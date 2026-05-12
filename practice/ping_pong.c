#include <stdio.h>
#include <mpi.h>

int main(int argc, char** argv)
{
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm comm = MPI_COMM_WORLD;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);
    int msg[2];
    MPI_Status status;
    double start_time = 0.0, end_time = 0.0;
    if (rank == 0) {
        msg[0] = rank; msg[1] = size;
    }
    start_time = MPI_Wtime();
    for (size_t i = 0; i < 100; i++){
        if (rank == 0){
            MPI_Send(msg, 2, MPI_INT, 1, 0, comm);
            MPI_Recv(msg, 2, MPI_INT, size-1, 0, comm, &status);    
        }
        else 
        {
            MPI_Recv(msg, 2, MPI_INT, (rank+size-1)%size, 0, comm, &status);
            MPI_Send(msg, 2, MPI_INT, (rank+1)%size, 0, comm);
        }
    }
    end_time = MPI_Wtime();

    printf("Rank: %d \t data: %d %d\n", rank, msg[0], msg[1]);
    if (rank == 0){
        printf("Time taken: %f.\n", end_time - start_time);
    }
    MPI_Finalize();
    return 0;
}