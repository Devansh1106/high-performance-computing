#include <stdlib.h>
#include <stdio.h>
#include <mpi.h>

int main (int argc, char** argv){
    MPI_Init(&argc, &argv);
    MPI_Comm comm = MPI_COMM_WORLD;
    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);
    srand(3);
    double* A = NULL;
    double* vec = NULL;
    double* result_vec = NULL;
    int n = 10000;
    int rem = n % size;
    int local_size = (n/size) + ((rank < rem) ? 1 : 0);
    int* sendcount1 = NULL;
    int* displs1 = NULL; 
    int* sendcount2 = NULL;
    int* displs2 = NULL; 
    double start_time = 0.0, end_time = 0.0;


    if (rank == 0){
        A = malloc(n * n * sizeof(double));
        vec = malloc(n * sizeof(double));
        result_vec = calloc(n , sizeof(double));
        sendcount1 = malloc(size * sizeof(int));
        displs1 = malloc(size * sizeof(int));
        sendcount2 = malloc(size * sizeof(int));
        displs2 = malloc(size * sizeof(int));

        for (int i = 0; i < size; i++){
            int count = (n/size) + ((i < rem) ? 1 : 0);
            // matrix
            sendcount1[i] = count * n;
            displs1[i] = (i == 0) ? 0 : displs1[i-1] + sendcount1[i-1];
            sendcount2[i] = count;
            displs2[i] = (i == 0) ? 0 : displs2[i-1] + sendcount2[i-1];
        }
        for (int i = 0; i < n; i++){
            for (int j = 0; j < n; j++){
                A[i*n + j] = (double)rand() / RAND_MAX; // inclusive 1
                // A[i*n + j] = 1.0;
            }
            vec[i] = (double)rand() / RAND_MAX;
            // vec[i] = 0.0;
        }
    }
    else{
        A = malloc(local_size * n * sizeof(double));
        vec = malloc(n * sizeof(double));
        result_vec = calloc(local_size , sizeof(double));
    }


    start_time = MPI_Wtime();
    // matrix
    MPI_Scatterv(A, sendcount1, displs1, MPI_DOUBLE, A, local_size * n, MPI_DOUBLE, 0, comm);
    MPI_Bcast(vec, n, MPI_DOUBLE, 0, comm);
    
    for (int i = 0; i < local_size; i++){
        for (int j = 0; j < n; j++){
            // without this third loop, we won't see scalability because of low compute.
            // for (int k = 0; k < 50; k++)
            //     result_vec[i] += A[i*n + j] * vec[j];
            result_vec[i] += A[i*n + j] * vec[j];
        }
    }
    MPI_Gatherv(result_vec, local_size, MPI_DOUBLE, result_vec, sendcount2, displs2, MPI_DOUBLE, 0, comm);


    end_time = MPI_Wtime();


    if (rank == 0){
        for (int i = 0; i < 50; i++){
            printf("%f\t", result_vec[i]);
        }
    }
    printf("\n");
    if (rank == 0) printf("Time taken: %f.\n", end_time - start_time);
    free(A); free(result_vec); free(vec);
    if (rank == 0){
        free(sendcount1);
        free(displs1);
        free(sendcount2);
        free(displs2);
    }
    MPI_Finalize();
    return 0;
}