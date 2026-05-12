// matrix vector product (parallel): Only matrix is distributed and full vector is broadcasted on all ranks (Block distribution)

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
    int* vec=NULL;
    double* result_vec=NULL;
    int n = 16384*2;
    int local_n = n/size;            // n % size should be 0 for now
    double start_time = 0.0;
    double end_time = 0.0, time = 0.0;

    if (rank == 0){
        A = malloc(n * n * sizeof(double));
        vec = malloc(n * sizeof(double));
        result_vec = calloc(n, sizeof(double));
        for(size_t i = 0; i < n; i++){
            for(size_t j = 0; j < n; j++){
                A[i * n + j] = 1.0;
            }
            vec[i] = 1.0;
        }
        start_time = MPI_Wtime();
        MPI_Scatter(A, (local_n * n), MPI_DOUBLE, A, (local_n * n), MPI_DOUBLE, 0, comm);
        MPI_Bcast(vec, n, MPI_DOUBLE, 0, comm); 
        end_time = MPI_Wtime();
        time = end_time - start_time;
    }
    else{
        A = malloc(local_n * n * sizeof(double));
        vec = malloc(n * sizeof(double));
        result_vec = calloc(local_n, sizeof(double));

        MPI_Scatter(A, (local_n * n), MPI_DOUBLE, A, (local_n * n), MPI_DOUBLE, 0, comm);
        MPI_Bcast(vec, n, MPI_DOUBLE, 0, comm);
    }

    start_time = MPI_Wtime();
    for(size_t i = 0; i < local_n; i++){
        for(size_t j = 0; j < n; j++){
            // without this third loop, we won't see scalability because of low compute.
            // for (int k = 0; k < 50; k++)
            //     result_vec[i] += A[i*n + j] * vec[j];
            result_vec[i] += A[i * n + j] * vec[j];
        }
    }
    MPI_Gather(result_vec, local_n, MPI_DOUBLE, result_vec, local_n, MPI_DOUBLE, 0, comm);
    end_time = MPI_Wtime();
    time += (end_time - start_time);

    if (rank == 0){
        // for (size_t i = 0; i < n; i++){
        //     printf("%f ", result_vec[i]);
        // }
        printf("Time taken is: %f sec.\n", time);
    }
    free(A);
    free(vec);
    free(result_vec);

    MPI_Finalize();
    return 0;
}