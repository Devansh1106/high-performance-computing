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
    a = rank;
    MPI_Sendrecv(&a, 1, MPI_INT, (rank+1)%size, 0, &b, 1, MPI_INT, (rank-1+size)%size, 0, comm, &status);
    // MPI_Send(&a, 1, MPI_INT, (rank+1)%size, 0, comm);
    // MPI_Recv(&b, 1, MPI_INT, (rank-1+size)%size, 0, comm, &status);

    printf("Done %d and value %d\n", rank, b);
    MPI_Finalize();
    return 0;
}

