/* Parallely solving heat eqn in 2D domain implicitly (BTCS) */

#ifndef M_PI
    #define M_PI 3.14159265358979323846
#endif

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <mpi.h>

double* matrix_vec_prod(double* A, 
                        double* vec, 
                        int n, 
                        double* result_vec, 
                        int rank, 
                        int size, 
                        MPI_Comm comm);
double gauss_jacobi(double* A, 
                    double* rhs_vec, 
                    double* initial_guess, 
                    int n, 
                    int max_iter, 
                    double tol, 
                    int rank, 
                    int size, 
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
    int n               = 20;
    int N               = n * n;
    double* A           = NULL;
    double* rhs_vec     = NULL;
    double* initial_guess = NULL;
    double* A_copy      = NULL;

    double max_tol = 1e-4;
    int max_iter = 10000;
    // Load balancing
    int rem = N % size;
    int local_n = (N/size) + (rank < rem ? 1 : 0);

    double total_time     = 0.01, t = 0.0;
    double c              = 1.0;
    double alpha          = 0.2;
    double dx             = 1.0 / (n - 1);
    double delta_t        = (alpha * dx * dx) / c;
    int total_steps       = ceil(total_time / delta_t);
    int step              = 0.0;

    if (rank == 0){
        A               = malloc(N * N * sizeof(double));
        rhs_vec         = malloc(N * sizeof(double));
        initial_guess   = calloc(N, sizeof(double));
        A_copy          = malloc(N * N * sizeof(double));

        for (int i = 0; i < n; i++) {
            double y = i * dx;
            for (int j = 0; j < n; j++) {
                double x = j * dx;
                int row            = i*n + j;
                initial_guess[row] = sin(M_PI * x) * sin(M_PI * y);
                // boundary nodes (Dirichlet u=0)
                if (i == 0 || i == n-1 || j == 0 || j == n-1) {
                    A[row*N + row] = 1.0;
                    rhs_vec[row]   = 0.0;
                } 
                else {
                    // diagonal
                    A[row*N + row] = 1.0 + 4.0*alpha;
                    // left
                    A[row*N + (row - 1)] = -alpha;
                    // right
                    A[row*N + (row + 1)] = -alpha;
                    // up
                    A[row*N + (row - n)] = -alpha;
                    // down
                    A[row*N + (row + n)] = -alpha;
                }
            }
        }
    }
    else{
        A             = malloc(local_n * N * sizeof(double));
        A_copy        = malloc(local_n * N * sizeof(double));
        rhs_vec       = malloc(N * sizeof(double));
        initial_guess = calloc(N, sizeof(double));
    }

    while (step < total_steps){
        memcpy(rhs_vec, initial_guess, N*sizeof(double));

        if (rank == 0)
            memcpy(A_copy, A, N * N * sizeof(double));

        time = gauss_jacobi(A_copy, rhs_vec, initial_guess, N, max_iter, max_tol, rank, size, comm);
        step++;

        if (rank == 0) 
            printf("\nAt step %d \t Time taken is: %f sec.\n", step, time);
    }
    if (rank == 0){
        double* u_exact = malloc(n * n * sizeof(double));
        exact_solution_2d(u_exact, n, total_time);
        double Linf_error = compute_Linf_error(initial_guess, u_exact, n);
        printf("L_inf error = %.10e\n", Linf_error);
        free(u_exact);

        // for (size_t i = 0; i < n; i++) {
        //     for (int j = 0; j < n; j++) {
        //         printf("%6.2f ", initial_guess[i * n + j]);
        //     }
        //     printf("\n");
        // }
    }

    free(A);
    free(rhs_vec);
    free(A_copy);
    free(initial_guess);

    MPI_Finalize();
    return 0;
}

double gauss_jacobi(double* A, 
                    double* rhs_vec, 
                    double* initial_guess, 
                    int n, 
                    int max_iter, 
                    double max_tol, 
                    int rank, 
                    int size, 
                    MPI_Comm comm){

    double start_time = 0.0;
    double end_time   = 0.0;
    double tol        = 1.0;
    double sum        = 0.0;
    int* sendcount1   = NULL;
    int* displs1      = NULL;
    int* sendcount2   = NULL;
    int* displs2      = NULL; 
    double* result_vec = NULL;
    int flag          = -1;
    double* diag      = NULL;
    double* temp      = NULL;
    int iter          = 0;
    
    // Load balancing
    int rem     = n % size;
    int local_n = (n/size) + (rank < rem ? 1 : 0);

    diag        = malloc(n * sizeof(double));
    result_vec  = calloc(local_n, sizeof(double));
    temp        = calloc(local_n, sizeof(double));
    sendcount1  = malloc(size * sizeof(int));
    displs1     = malloc(size * sizeof(int));
    sendcount2  = malloc(size * sizeof(int));
    displs2     = malloc(size * sizeof(int));

    if (rank == 0){
        for(size_t i = 0; i < n; i++){
            diag[i]     = A[i*n + i];
            A[i*n + i]  = 0.0;
        }    
    }
    for(size_t i = 0; i < size; i++){
        int count     = (n/size) + (i < rem ? 1 : 0);
        sendcount1[i] = count * n;
        sendcount2[i] = count;
        displs1[i]    = (i == 0) ? 0 : displs1[i - 1] + sendcount1[i - 1];
        displs2[i]    = (i == 0) ? 0 : displs2[i - 1] + sendcount2[i - 1];
    }

    start_time = MPI_Wtime();
    MPI_Scatterv(A, 
                sendcount1,
                displs1, 
                MPI_DOUBLE, 
                A, 
                (local_n * n), 
                MPI_DOUBLE, 
                0, 
                comm);

    MPI_Scatterv(diag,
                sendcount2, 
                displs2, 
                MPI_DOUBLE, 
                diag, 
                local_n, 
                MPI_DOUBLE, 
                0, 
                comm);

    MPI_Bcast(rhs_vec, n, MPI_DOUBLE, 0, comm);
    MPI_Bcast(initial_guess, n, MPI_DOUBLE, 0, comm); 

    while((tol > max_tol) && (iter != max_iter)){
        sum = 0.0;
        result_vec = matrix_vec_prod(A, 
                                    initial_guess, 
                                    n, 
                                    result_vec, 
                                    rank, 
                                    size, 
                                    comm);

        for (size_t i = 0; i < local_n; i++){
            temp[i] = rhs_vec[displs2[rank] + i] - result_vec[i];
            temp[i] = temp[i] / diag[i];
            sum     += fabs(temp[i] - initial_guess[displs2[rank] + i]);
        }
        MPI_Allgatherv(temp, 
                        local_n, 
                        MPI_DOUBLE, 
                        initial_guess, 
                        sendcount2, 
                        displs2, 
                        MPI_DOUBLE, 
                        comm);

        MPI_Allreduce(&sum, &tol, 1, MPI_DOUBLE, MPI_SUM, comm);
        iter++;
    }
    end_time = MPI_Wtime();
    if (rank == 0){
        if (iter == max_iter){
            printf("Failed to converge, max iterations reached!\n");
        }
    }
    free(result_vec);
    free(temp);
    free(diag);
    return (end_time - start_time);
}

double* matrix_vec_prod(double* A, 
                        double* vec, 
                        int n, 
                        double* 
                        result_vec, 
                        int rank, 
                        int size, 
                        MPI_Comm comm){
    int rem     = n % size;
    int local_n = (n/size) + (rank < rem ? 1 : 0);
    // reseting is needed otherwise it will have the value from the previous call
    for(size_t i = 0; i < local_n; i++) {
        result_vec[i] = 0.0;
    }

    for(size_t i = 0; i < local_n; i++){
        for(size_t j = 0; j < n; j++){
            result_vec[i] += A[i * n + j] * vec[j];
        }
    }
    return result_vec;
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
