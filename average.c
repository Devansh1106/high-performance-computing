#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);
    int rank, size;
    int* a=NULL;
    int n = 1000000;
    int sum = 0.0, final_sum = 0.0;
    int* sums=NULL;
    MPI_Comm comm = MPI_COMM_WORLD;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);
    int m = n/size;

    if (rank == 0){
        a = malloc(n * sizeof(int));
        sums = malloc(size * sizeof(int));
        for (size_t i = 0; i < n; i++){
            a[i] = 12;
        }
    }
    else{
        a = malloc(m * sizeof(int));
    }

    MPI_Scatter(a, m, MPI_INT, a, m, MPI_INT, 0, comm);
    
    for (size_t i = 0; i < m; i++){
        sum = sum + a[i];
    }
    MPI_Gather(&sum, 1, MPI_INT, sums, 1, MPI_INT, 0, comm);
    if (rank == 0){
        for (size_t i = 0; i < size; i++){
            final_sum = final_sum + sums[i];
        }
        printf("Average is %d.\n", final_sum/n);
        free(sums);
    }


    free(a);
    MPI_Finalize();
    return 0;
}