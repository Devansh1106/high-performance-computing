/* Parallely solving heat eqn in 2D domain explicitly (FTCS) */

#ifndef M_PI
    #define M_PI 3.14159265358979323846
#endif

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>

void update_heat_2d(double* u_new, 
                    double* initial_guess,
                    double* final_sol,
                    int n, double total_time,
                    double alpha, 
                    double delta_t,
                    int rank, int size,
                    MPI_Comm comm);

void exact_solution_2d(double* u_exact,
                       int n,
                       double total_time);

double compute_Linf_error(double* numerical,
                          double* exact,
                          int n);

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm comm = MPI_COMM_WORLD;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    double start_time   = 0.0;
    double end_time     = 0.0, time = 0.0;
    /* This code has been tested for n = [10, 20, 40, 80] 
        Computed convergence rate is: ~ 2.00 */
    int n               = 120;
    double* u_new       = NULL;
    double* u           = NULL;
    double* initial_guess = NULL;
    double* final_sol     = NULL;
    double total_time     = 0.01, t = 0.0;
    double c              = 1.0;
    double alpha          = 0.2;
    double dx             = 1.0 / (n - 1);
    double delta_t        = (alpha * dx * dx) / c;

    // distribution of initial guess only
    /* initial_guess is a n x n matrix flattened into a 1D vector in row-major format */
    int rem        = n % size;
    int base       = (n/size);
    int local_rows = base + (rank < rem ? 1 : 0);

    // memory allocation on each rank
    initial_guess = calloc((local_rows + 2) * n, sizeof(double));
    u_new         = calloc((local_rows + 2) * n, sizeof(double));
    if (rank == 0) {
        final_sol = malloc(n * n * sizeof(double));
    }

    // Compute starting row for each rank
    int start_row = (rank*base) + (rank < rem ? rank : rem);
    // skip ghost row j = 0
    for (int j = 1; j <= local_rows; j++) {
        int global_row = start_row + (j-1);
        double y = global_row*dx;
        for (int i = 0; i < n; i++) {
            double x = i * dx;
            initial_guess[j*n + i] = sin(M_PI * x) * sin(M_PI * y);
        }
    }
    start_time = MPI_Wtime();

    update_heat_2d(u_new, 
                   initial_guess, 
                   final_sol, n, 
                   total_time, 
                   alpha, 
                   delta_t, 
                   rank, size, 
                   comm);

    end_time    = MPI_Wtime();
    time        = end_time - start_time;

    if (rank == 0){
        double* u_exact = malloc(n * n * sizeof(double));
        exact_solution_2d(u_exact, n, total_time);
        double Linf_error = compute_Linf_error(final_sol, u_exact, n);
        printf("L_inf error = %.10e\n", Linf_error);
        free(u_exact);

        // priting the sol
        // for (size_t i = 0; i < n; i++) {
        //     for (int j = 0; j < n; j++) {
        //         printf("%6.2f ", final_sol[i * n + j]);
        //     }
        //     printf("\n");
        // }
        printf("Time taken: %f.\n", time);
    }

    free(initial_guess);
    free(u_new);
    if (rank == 0){ 
        free(final_sol); 
    }
    MPI_Finalize();
    return 0;
}

void update_heat_2d(double* u_new, 
                    double* initial_guess, 
                    double* final_sol, 
                    int n, double total_time, 
                    double alpha, 
                    double delta_t,
                    int rank, int size, 
                    MPI_Comm comm){

    MPI_Status status;

    int rem     = n % size;
    int base    = (n/size);
    int local_rows  = base + (rank < rem ? 1 : 0);
    int step        = 0;
    int* recvcounts = NULL;
    int* displs     = NULL;

    int top_neighbor    = (rank == 0) ? MPI_PROC_NULL : rank - 1; // MPI_PROC_NULL: null value of rank
    int bottom_neighbor = (rank == size - 1) ? MPI_PROC_NULL : rank + 1;
    int total_steps     = ceil(total_time / delta_t);

    for (size_t i = 0; i < (local_rows + 2) * n; i++) 
        u_new[i] = initial_guess[i];

    while (step < total_steps){
        // communication with top_neighbour  
        MPI_Sendrecv(&initial_guess[n], 
                    n, MPI_DOUBLE, 
                    top_neighbor, 
                    0, &initial_guess[0], 
                    n, MPI_DOUBLE, 
                    top_neighbor, 
                    0, comm, 
                    &status);

        // communication with bottom_neighbour
        MPI_Sendrecv(&initial_guess[(local_rows)*n], 
                    n, MPI_DOUBLE, 
                    bottom_neighbor, 
                    0, &initial_guess[(local_rows+1) * n],
                    n, MPI_DOUBLE, 
                    bottom_neighbor, 
                    0, comm,
                    &status);

        int row_start   = 1;
        int row_end     = local_rows;

        if (rank == 0) 
            row_start = 2;          // don't compute row 1 (global top boundary for rank 0)
        if (rank == size - 1) 
            row_end = local_rows - 1;

        for (size_t j = row_start; j <= row_end; j++){
            for (size_t i = 1; i < n - 1; i++){
                int k        = j*n + i;
                double right = initial_guess[k+1];
                double left  = initial_guess[k-1];
                double diag  = initial_guess[k];
                double up    = initial_guess[(j+1)*n + i];
                double down  = initial_guess[(j-1)*n + i];

                u_new[k] = diag + alpha * (up + down + left + right - (4*diag));
            }
        }
        double* temp  = initial_guess;
        initial_guess = u_new;
        u_new         = temp;
        step          += 1;
    }
    if (rank == 0) {
        recvcounts = malloc(size * sizeof(int));
        displs     = malloc(size * sizeof(int));

        int offset = 0;
        for (int i = 0; i < size; i++){
            int rows      = base + (i < rem ? 1 : 0);
            recvcounts[i] = rows * n;
            displs[i]     = offset;
            offset        += recvcounts[i];
        }
    }
    MPI_Gatherv(&initial_guess[n], 
                (local_rows * n), 
                MPI_DOUBLE, 
                final_sol, 
                recvcounts, 
                displs, 
                MPI_DOUBLE, 
                0, comm);

    if (rank == 0){
        printf("Iterations: %d.\n", step);
        free(recvcounts);
        free(displs);
    }
}

void exact_solution_2d(double* u_exact,
                       int n,
                       double total_time){
    double dx = 1.0 / (n - 1);
    for (int j = 0; j < n; j++) {
        double y = j * dx;
        for (int i = 0; i < n; i++) {
            double x = i * dx;
            u_exact[j*n + i] = sin(M_PI * x) *
                               sin(M_PI * y) *
                               exp(-2.0 * M_PI * M_PI * total_time);
        }
    }
}

double compute_Linf_error(double* numerical,
                          double* exact,
                          int n){
    double max_error = 0.0;
    for (int j = 0; j < n; j++) {
        for (int i = 0; i < n; i++) {
            double diff = fabs(numerical[j*n + i] -
                               exact[j*n + i]);
            if (diff > max_error)
                max_error = diff;
        }
    }
    return max_error;
}
