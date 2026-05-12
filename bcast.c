#include <stdio.h>
#include <mpi.h>

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm comm = MPI_COMM_WORLD;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);
    int a[4];

    if (rank == 0){
        a[0] = 12; a[2] = 12; a[3] = 12; a[1] = 12;
    }

    // MPI_Barrier(comm);
    MPI_Bcast(a, 4, MPI_INT, 0, comm);
    
    printf("I am rank %d.\n", rank);
    for (size_t i = 0; i < size; i++){
        printf("%d\n", a[i]);
    }

    MPI_Finalize();
    return 0;
}