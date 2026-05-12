// Trapezoidal rule in [0,1] using MPI (corrected)

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>

double fun(double x){
    return (pow(x,2.0))*exp(-x);
}

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm comm = MPI_COMM_WORLD;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    int n = 10000;     // number of subintervals
    double a = 0.0, b = 1.0;
    double h = (b - a) / n;

    double local_area = 0.0, area = 0.0;

    /* Load balancing (in case n % size != 0) */
    int rem     = n % size;
    int local_n = (n/size) + (rank < rem ? 1 : 0);

    // this logic is next-level
    int start = (rank < rem)
                ? rank * local_n
                : rank * local_n + rem;

    int end = start + local_n;

    double time = MPI_Wtime();

    /* Trapezoidal summation over assigned indices */
    for(int i = start; i < end; i++){
        double x_i = a + i*h;

        if(i == 0 || i == n)
            local_area += fun(x_i) / 2.0;
        else
            local_area += fun(x_i);
    }

    local_area *= h;

    MPI_Reduce(&local_area, &area, 1,
               MPI_DOUBLE, MPI_SUM, 0, comm);

    double end_time = MPI_Wtime();

    if (rank == 0){
        printf("%f sec\n", end_time - time);
        printf("Integral value: %.12f\n", area);
    }

    MPI_Finalize();
    return 0;
}