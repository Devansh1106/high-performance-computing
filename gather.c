#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);
    int rank, size;
    int* a=NULL;
    MPI_Comm comm = MPI_COMM_WORLD;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);
    int n = 4;

    if (rank != 0){
        a = malloc(n * sizeof(int));
    }
    else{
        a = malloc(size * n * sizeof(int));
    }

    for (size_t i = 0; i < n; i++){
        a[i] = rank * 12;
    }

    MPI_Gather(a, n, MPI_INT, a, n, MPI_INT, 0, comm);
    
    if (rank == 0){
        printf("I am rank %d.\n", rank);
        for (size_t i = 0; i < n*size; i++){
            printf("%d\n", a[i]);
        }
    }
    free(a);
    MPI_Finalize();
    return 0;
}