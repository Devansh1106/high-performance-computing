// Minimum, Maximum and Sum using MPI

#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

struct {
    int val;
    int rank;
} max_in, max_out, min_in, min_out;


int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);
    int rank, size;
    double start_time, end_time, elapsed = 0.0;
    MPI_Comm comm = MPI_COMM_WORLD;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);
    long long int n = 1e8;
    long long local_sum = 0.0;
    long long local_prod = 1.0;
    int local_max, local_min;
    int max, min;
    long long int sum = 0, prod = 1.0;
    int* a=NULL;

    if (rank == 0){
        start_time = MPI_Wtime();
        a = malloc(n * sizeof(int));
        for (size_t i = 0; i < n; i++){
            a[i] = (rand() % 100) + 1;
        }
        MPI_Scatter(a, n/size, MPI_INT, a, n/size, MPI_INT, 0, comm);
        local_max = a[0];
        local_min = a[0];
        for(size_t i = 0; i < n/size; i++){
            local_sum += a[i];
            local_prod *= a[i];
            // local min
            if (local_max < a[i]){
                local_max = a[i];
            }
            // local max
            if (local_min > a[i]){
                local_min = a[i];
            }
        }
        max_in.val = local_max;
        max_in.rank = rank;
        min_in.val = local_min;
        min_in.rank = rank;
        end_time = MPI_Wtime();
        elapsed = end_time - start_time;
    }
    else {
        start_time = MPI_Wtime();
        a = malloc((1e9/size) * sizeof(int));
        MPI_Scatter(a, n/size, MPI_INT, a, n/size, MPI_INT, 0, comm);

        local_max = a[0];
        local_min = a[0];
        for(size_t i = 0; i < n/size; i++){
            local_sum += a[i];
            local_prod *= a[i];
            // local min
            if (local_max < a[i]){
                local_max = a[i];
            }
            // local max
            if (local_min > a[i]){
                local_min = a[i];
            }
        }
        max_in.val = local_max;
        max_in.rank = rank;
        min_in.val = local_min;
        min_in.rank = rank;
        end_time = MPI_Wtime();
        elapsed = end_time - start_time;
    }
    printf("Local sum of elements of array: %lld on rank %d.\n", local_sum, rank);
    printf("Local product of elements of array: %lld on rank %d.\n", local_prod, rank);

    start_time = MPI_Wtime();
    MPI_Reduce(&local_sum, &sum, 1, MPI_INT, MPI_SUM, 0, comm);
    MPI_Reduce(&local_prod, &prod, 1, MPI_INT, MPI_PROD, 0, comm);
    MPI_Reduce(&max_in, &max_out, 1, MPI_2INT, MPI_MAXLOC, 0, comm);
    MPI_Reduce(&min_in, &min_out, 1, MPI_2INT, MPI_MINLOC, 0, comm);
    end_time = MPI_Wtime();
    elapsed += (start_time - end_time);
    

    if (rank == 0){
        printf("Global max is %d on the rank %d.\n", max_out.val, max_out.rank);
        printf("Global min is %d on the rank %d.\n", min_out.val, min_out.rank);
        printf("Summation of all the elements of the array is %lld.\n", sum);
        printf("Product of all the elements of the array is %lld.\n", prod); 
    }
    MPI_Barrier(comm);
    printf("Time taken by rank %d is: %f seconds.\n", rank, elapsed);
    free(a);
    MPI_Finalize();
    return 0;
}