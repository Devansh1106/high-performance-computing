// ping pong for any size

#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);
    int rank, size, a, b;
    MPI_Comm comm = MPI_COMM_WORLD;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);
    MPI_Status status;
    int n = 10;
    double start_time = MPI_Wtime();
    for (size_t i = 0; i < n; i++){
        // a = (rank+1)%size;
        a = rank;
        // b = (rank-1+size)%size;
        MPI_Send(&a, 1, MPI_INT, (rank+1)%size, 0, comm);
        MPI_Recv(&b, 1, MPI_INT, (rank-1+size)%size, 0, comm, &status);

        printf("Done %d and value %d\n", rank, b);
        // printf("Done %d and value %d\n", rank, a);
    }
    double end_time = MPI_Wtime();
    if (rank == 0){ printf("Time: %f\n", end_time - start_time);}
    MPI_Finalize();
    return 0;
}

