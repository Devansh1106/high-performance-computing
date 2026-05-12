#include <stdio.h>
#include <mpi.h>

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm comm = MPI_COMM_WORLD;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);
    int ranks[size];
    int sizes[size];

    // if (rank > 0){
    MPI_Gather(&rank, 1, MPI_INT, ranks, 1, MPI_INT, 0, comm);
    MPI_Gather(&size, 1, MPI_INT, sizes, 1, MPI_INT, 0, comm);
    // }

    if (rank == 0){
        for(size_t i = 0; i < size; i++){
            printf("Ranks and size are %d and %d.\n", ranks[i], sizes[i]);
        }
    }
    MPI_Finalize();
    return 0;
}