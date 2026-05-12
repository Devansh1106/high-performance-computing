// Calculating average and standard deviation

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);
    int rank, size;
    double* a=NULL;
    int n = 1000000;
    double sum = 0.0, final_sum = 0.0, avg = 0.0, sd = 0.0;
    double* sums=NULL;
    MPI_Comm comm = MPI_COMM_WORLD;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);
    int m = n/size;
    srand(3);

    // Creating the array
    if (rank == 0){
        a = malloc(n * sizeof(double));
        sums = malloc(size * sizeof(double));
        for (size_t i = 0; i < n; i++){
            a[i] = rand();
        }
    }
    else{
        a = malloc(m * sizeof(double));
    }

    // Distributed average of the elements of the array
    MPI_Scatter(a, m, MPI_DOUBLE, a, m, MPI_DOUBLE, 0, comm);    
    
    for (size_t i = 0; i < m; i++){
        sum = sum + a[i];
    }
    MPI_Gather(&sum, 1, MPI_DOUBLE, sums, 1, MPI_DOUBLE, 0, comm);
    if (rank == 0){
        for (size_t i = 0; i < size; i++){
            final_sum = final_sum + sums[i];
        }
        avg = final_sum/n;
        printf("Average is %f.\n", avg);
    }

    // Broadcasting the average and Standard deviation calculation
    MPI_Bcast(&avg, 1, MPI_DOUBLE, 0, comm);
    sum = 0.0;
    final_sum = 0.0;
    for (size_t j = 0; j < m; j++){
        sum += pow((a[j] - avg), 2.0);  
    }
    MPI_Gather(&sum, 1, MPI_DOUBLE, sums, 1, MPI_DOUBLE, 0, comm);
    if (rank == 0){
        for (size_t i = 0; i < size; i++){
            final_sum = final_sum + sums[i];
        }
        final_sum = sqrt(final_sum);
        sd = final_sum/(n-1);
        printf("Standard deviation is %f.\n", sd);
        free(sums);
    }


    free(a);
    MPI_Finalize();
    return 0;
}