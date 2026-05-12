#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

int main (int argc, char **argv)
{
    MPI_Init(&argc, &argv);
    int rank, size;
    int message[2];
    int tag, dest, source;
    MPI_Status status;
    // MPI_COMM_WORLD comm;

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (size == 1){
        printf("This program requires more than 1 process.\n");
        MPI_Finalize();
        exit(0);
    }
    if (rank != 0){
        message[0] = rank;
        message[1] = size;
        MPI_Send(message, 2, MPI_INT, 0, tag, MPI_COMM_WORLD);
    }
    else{
        for (int src=1; src <= size-1; src++){
            MPI_Recv(message, 2, MPI_INT, src, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
            printf("I am rank %d out of size %d.\n", message[0], message[1]);
        }
    }

    MPI_Finalize();
    return 0;
}