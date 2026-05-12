// parallel gauss-jacobi method (when n % size == 0)

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>

double* matrix_vec_prod(double* A, double* vec, int n, double* result_vec, int size, MPI_Comm comm);

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm comm = MPI_COMM_WORLD;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    double* A=NULL;
    double* rhs_vec=NULL;
    double* diag=NULL;
    double* initial_guess=NULL;
    double* result_vec=NULL;
    double* temp=NULL;
    int n = 1000; 
    int local_n = n/size;            // n % size should be 0 for now
    double start_time = 0.0;
    double end_time = 0.0, time = 0.0;
    double tol = 1.0;
    int iter = 0.0;
    double sum = 0.0;
    int flag = -1;
    int max_iter = 10000;

    if (rank == 0){
        A = malloc(n * n * sizeof(double));
        rhs_vec = malloc(n * sizeof(double));
        diag = malloc(n * sizeof(double));
        initial_guess = calloc(n, sizeof(double));
        result_vec = calloc(local_n, sizeof(double));
        temp = calloc(local_n, sizeof(double));

        for(size_t i = 0; i < n; i++){
            for(size_t j = 0; j < n; j++){
                A[i * n + j] = 1.0;
            }
            A[i * n + i] = 0.0;
            diag[i] = n;
            rhs_vec[i] = 1.0;
        }
        start_time = MPI_Wtime();
        MPI_Scatter(A, (local_n * n), MPI_DOUBLE, A, (local_n * n), MPI_DOUBLE, 0, comm);
        MPI_Scatter(diag, local_n, MPI_DOUBLE, diag, local_n, MPI_DOUBLE, 0, comm);
        MPI_Bcast(rhs_vec, n, MPI_DOUBLE, 0, comm);
        MPI_Bcast(initial_guess, n, MPI_DOUBLE, 0, comm); 
        end_time = MPI_Wtime();
        time = start_time - end_time;

    }
    else{
        A = malloc(local_n * n * sizeof(double));
        rhs_vec = malloc(n * sizeof(double));
        initial_guess = calloc(n, sizeof(double));
        result_vec = calloc(local_n, sizeof(double));
        diag = malloc(local_n * sizeof(double));
        temp = calloc(local_n, sizeof(double));        

        MPI_Scatter(A, (local_n * n), MPI_DOUBLE, A, (local_n * n), MPI_DOUBLE, 0, comm);
        MPI_Scatter(diag, local_n, MPI_DOUBLE, diag, local_n, MPI_DOUBLE, 0, comm);
        MPI_Bcast(rhs_vec, n, MPI_DOUBLE, 0, comm);
        MPI_Bcast(initial_guess, n, MPI_DOUBLE, 0, comm); 
    }

    start_time = MPI_Wtime();
    while((tol > 1e-4) && (iter != max_iter)){
        sum = 0.0;
        result_vec = matrix_vec_prod(A, initial_guess, n, result_vec, size, comm);
        for (size_t i = 0; i < local_n; i++){
            temp[i] = rhs_vec[rank*local_n + i] - result_vec[i];
            temp[i] = temp[i] / diag[i];
            sum += fabs(temp[i] - initial_guess[rank*local_n + i]);
        }
        MPI_Allgather(temp, local_n, MPI_DOUBLE, initial_guess, local_n, MPI_DOUBLE, comm);
        MPI_Allreduce(&sum, &tol, 1, MPI_DOUBLE, MPI_SUM, comm);
        printf("\nTol: %f \t Rank:\t%d", tol, rank);
        printf("\t Iter: \t%d\n", iter);
        iter++;
    }
    end_time = MPI_Wtime();
    time += (start_time - end_time);
    if (rank == 0){
        if (iter == max_iter){
            printf("Failed to converge, max iterations reached!\n");
        }
        else{
            printf("Solution: \n");
            // for (size_t i = 0; i < 100; i++){
            //     printf("%f ", initial_guess[i]);
            // }
        }
        printf("\nTime taken is: %f sec.\n", end_time - start_time);
    }
    free(A);
    free(rhs_vec);
    free(result_vec);
    free(temp);
    free(diag);
    free(initial_guess);

    MPI_Finalize();
    return 0;
}

double* matrix_vec_prod(double* A, double* vec, int n, double* result_vec, int size, MPI_Comm comm){
    int local_n = n/size;            // n % size should be 0 for now
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