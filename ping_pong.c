// ping pong for size=2 only

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
    int n = 100000;
    double start_time = MPI_Wtime();
    for (size_t i = 0; i < n; i++){
        if (rank == 0){
            a = 1234;
            b = 0;
            MPI_Send(&a, 1, MPI_INT, 1, 0, comm);
            MPI_Recv(&b, 1, MPI_INT, 1, 0, comm, &status);
        }
        else{
            b = 7890;
            a = 0;
            MPI_Recv(&a, 1, MPI_INT, 0, 0, comm, &status);
            MPI_Send(&b, 1, MPI_INT, 0, 0, comm);
        }
        if (rank == 0) { printf("Done %d and value %d\n", rank, b);}
        else{ printf("Done %d and value %d\n", rank, a);}
    }
    double end_time = MPI_Wtime();
    if (rank == 0){ printf("Time: %f\n", end_time - start_time);}
    MPI_Finalize();
    return 0;
}

