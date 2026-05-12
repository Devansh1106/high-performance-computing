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
    int n = 8;

    if (rank == 0){
        a = malloc(n * sizeof(int));
        for (size_t i = 0; i < n; i++){
            a[i] = 12;
        }
    }
    else{
        a = malloc(2 * sizeof(int));
    }

    MPI_Scatter(a, 2, MPI_INT, a, 2, MPI_INT, 0, comm);
    
    printf("I am rank %d.\n", rank);
    for (size_t i = 0; i < 2; i++){
        printf("%d\n", a[i]);
    }
    free(a);
    MPI_Finalize();
    return 0;
}