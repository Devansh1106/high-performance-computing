// matrix-matrix product (parallel): block distribution
// Matrix A has been distributed and matrix B has been broadcasted to everyone

/* Flattening of A is row-major
and flattening of B is column-major */

#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm comm = MPI_COMM_WORLD;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    double* A=NULL;
    double* B=NULL;
    double* result_mat=NULL;
    int n = 700; 
    int local_n_rows = n/size;            // n % size should be 0 for now
    double start_time = 0.0;
    double end_time = 0.0, time = 0.0;

    if (rank == 0){
        A = calloc(n * n , sizeof(double));
        B = malloc(n * n * sizeof(double));
        result_mat = calloc((n * n), sizeof(double));
        for(size_t i = 0; i < n; i++){
            for(size_t j = 0; j < n; j++){
                B[i * n + j] = 2.0;
            }
            A[i * n] = 1.0;
        }
        start_time = MPI_Wtime();
        MPI_Scatter(A, (local_n_rows * n), MPI_DOUBLE, A, (local_n_rows * n), MPI_DOUBLE, 0, comm);
        MPI_Bcast(B, (n*n), MPI_DOUBLE, 0, comm);
        time = start_time - end_time;
    }
    else{
        A = malloc(local_n_rows * n * sizeof(double));
        B = malloc(n * n * sizeof(double));
        result_mat = calloc((local_n_rows * n), sizeof(double));

        MPI_Scatter(A, (local_n_rows * n), MPI_DOUBLE, A, (local_n_rows * n), MPI_DOUBLE, 0, comm);
        MPI_Bcast(B, (n*n), MPI_DOUBLE, 0, comm);
    }

    // matrix-matrix multiplication
    start_time = MPI_Wtime();
    for (size_t k = 0; k < local_n_rows; k++){
        for(size_t i = 0; i < n; i++){
            for(size_t j = 0; j < n; j++){
                result_mat[k * n + i] += A[k * n + j] * B[i * n + j];
            }
        }
    }
    MPI_Gather(result_mat, (local_n_rows*n), MPI_DOUBLE, result_mat, (local_n_rows*n), MPI_DOUBLE, 0, comm);
    end_time = MPI_Wtime();
    time += (end_time - start_time);

    if (rank == 0){
        // for (size_t i = 0; i < n; i++){
        //     for (size_t j = 0; j < n; j++){
        //         printf("%f\t", result_mat[j]);
        //     }
        //     printf("\n");
        // }
        printf("\nTime taken is: %f sec.\n", time);
    }
    free(A);
    free(B);
    free(result_mat);

    MPI_Finalize();
    return 0;
}