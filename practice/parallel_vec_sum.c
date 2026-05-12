#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

int main(int argc, char** argv){
    MPI_Init(&argc, &argv);
    MPI_Comm comm = MPI_COMM_WORLD;
    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);
    srand(3);
    int n = 100000;
    double* vec = NULL;
    int* sendcount = NULL;
    int* displs = NULL;
    int rem = n % size;
    int local_size = (n/size) + (rank < rem ? 1 : 0);
    double ans = 0.0;
    // if (rank == size - 1){
    // }
    if (rank == 0){
        vec = malloc(n * sizeof(double));
        sendcount = malloc(size * sizeof(int));
        displs = malloc(size * sizeof(int));
        for (int i = 0; i < size; i++){
            int count = (n/size) + (i < rem ? 1 : 0);
            sendcount[i] = count;
            displs[i] = (i == 0) ? 0 : displs[i-1] + sendcount[i-1];
        }

        for (size_t i = 0; i < n; i++){
            vec[i] = (double)rand() / RAND_MAX;
        }
        MPI_Scatterv(vec, sendcount, displs, MPI_DOUBLE, vec, local_size, MPI_DOUBLE, 0, comm);
    }
    else{
        vec = malloc(local_size * sizeof(double));
        MPI_Scatterv(vec, sendcount, displs, MPI_DOUBLE, vec, local_size, MPI_DOUBLE, 0, comm);
    }
    double sum = 0.0;
    for (int i = 0; i < local_size; i++){
        sum += vec[i];
    }
    MPI_Reduce(&sum, &ans, 1, MPI_DOUBLE, MPI_SUM, 0, comm);
    if (rank == 0){
        printf("Total sum: %f.\n", ans);
        free(sendcount);
        free(displs);
    }
    free(vec);
    MPI_Finalize();
    return 0;
}