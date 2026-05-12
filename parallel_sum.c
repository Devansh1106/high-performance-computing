
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
    srand(3);

    double* A = NULL;
    double n = 1000000;
    double sum = 0.0, final_sum = 0.0;

    double start_time = MPI_Wtime();
    if (rank == 0){
        A = malloc(n * sizeof(double));
        for (size_t i = 0; i < n; i++){
            // A[i] = (rand() % 10) + 1;
            A[i] = 1;
        }
        MPI_Scatter(A, n/size, MPI_DOUBLE, A, n/size, MPI_DOUBLE, 0, comm);  // n % size == 0 for now
    }
    else{
        A = malloc((n/size) * sizeof(double));
        MPI_Scatter(A, n/size, MPI_DOUBLE, A, n/size, MPI_DOUBLE, 0, comm);
    }
    for (size_t i = 0; i < n/size; i++){
        sum += A[i];
    }
    MPI_Reduce(&sum, &final_sum, 1, MPI_DOUBLE, MPI_SUM, 0, comm);
    if (rank == 0){
        printf("Sum of the entries of the vector is: %f.\n", final_sum);
    }
    double end_time = MPI_Wtime();
    free(A);
    printf("Time taken: %f sec.\n", end_time - start_time);

    MPI_Finalize();
    return 0;
}