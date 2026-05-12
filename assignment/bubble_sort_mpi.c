/* parallel bubble sort using odd-even transposition */

#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

void bubble_sort(double* arr, 
                int local_n);
void odd_even_merge(double* arr, 
                    int size, 
                    int rank, 
                    int n, 
                    MPI_Comm comm);

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm comm = MPI_COMM_WORLD;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);
    srand(3);

    double start_time = 0.0;
    double end_time   = 0.0, time = 0.0;
    int n             = 10000;
    double* arr;

    // Load balancing
    int rem     = n % size;
    int local_n = (n/size) + (rank < rem ? 1 : 0);
    
    int* sendcounts;
    int* displs;
    sendcounts = malloc(size * sizeof(int));
    displs     = malloc(size * sizeof(int));
    for(size_t i = 0; i < size; i++){
        sendcounts[i] = (n/size) + (i < rem ? 1 : 0);        // each rank will have their own local_n (potentially different that others)
        displs[i]     = (i == 0) ? 0 : displs[i - 1] + sendcounts[i - 1];
    }

    if (rank == 0){
        arr = malloc(n * sizeof(double));
        for (size_t i = 0; i < n; i++){
            arr[i] = rand() % 100;
        }
    }
    else{
        arr = malloc(local_n * sizeof(double));
    }

    // printing unsorted array
    // if (rank == 0){
    //     printf("Unsorted array: \n");
    //     for (size_t i = 0; i < n; i++){
    //         printf("%f  ", arr[i]);
    //     }
    //     printf("\n");
    // }
    start_time = MPI_Wtime();
    MPI_Scatterv(arr, 
                sendcounts, 
                displs,
                MPI_DOUBLE, 
                arr, 
                local_n, 
                MPI_DOUBLE, 
                0, comm);

    bubble_sort(arr, local_n);

    odd_even_merge(arr, 
                   size, 
                   rank, 
                   n, comm);

    MPI_Gatherv(arr, 
                local_n, 
                MPI_DOUBLE, 
                arr, 
                sendcounts, 
                displs, 
                MPI_DOUBLE, 
                0, comm);

    end_time = MPI_Wtime();
    time     = end_time - start_time;
    if (rank == 0){
        // printf("Sorted array: \n");
        // for (size_t i = 0; i < n; i++){
        //     printf("%.2f  ", arr[i]);
        // }
        // printf("\n");
        printf("Time taken: %f.\n", time);
    }
    free(arr);
    free(sendcounts);
    free(displs);
    MPI_Finalize();
    return 0;
}

void bubble_sort(double* arr, int local_n){
    double temp;
    for (size_t k = 0; k < local_n; k++){
        for (size_t i = local_n-1; i > k; i--){
            if (arr[i] < arr[i-1]){
                temp     = arr[i];
                arr[i]   = arr[i-1];
                arr[i-1] = temp;
            }
        }
    }
}

void odd_even_merge(double* arr, 
                    int size, 
                    int rank, 
                    int n, 
                    MPI_Comm comm){

    MPI_Status status;
    int partner_n   = 0;
    int partner;
    int rem         = n % size;
    int local_n     = (n/size) + (rank < rem ? 1 : 0);
    int max_n       = (n/size) + 1;
    /* 
        each local array has 1 or 2 extra length due to uneven distribution 
        based on local_n so took max{1,2} for sizeof temp variable
    */
    double* temp    = malloc((2 * max_n) * sizeof(double));
    double* recvbuf = malloc(max_n * sizeof(double));
    
    for (size_t p = 0; p < size; p++){
        partner = -1;
        if (p % 2 == 0) {   // even phase
            if (rank % 2 == 0) {    // even ranks
                if (rank + 1 < size)
                    partner = rank + 1;
            }
            else{   // odd ranks
                partner = rank - 1;
            }
        }
        else{   // odd phase
            if (rank % 2 == 1) {    // odd ranks
                if (rank + 1 < size)
                    partner = rank + 1;
            }
            else{   // even ranks
                partner = rank - 1;
            }
        }
        if (partner >= 0){
            partner_n = (n/size) + (partner < rem ? 1 : 0);
            MPI_Sendrecv(arr, 
                        local_n, 
                        MPI_DOUBLE, 
                        partner, 0, 
                        recvbuf, 
                        partner_n, 
                        MPI_DOUBLE, 
                        partner, 0, 
                        comm, 
                        &status);

            // merging two sorted arrays
            int i = 0, j = 0, k = 0;
            while(i < local_n && j < partner_n){
                if  (arr[i] <= recvbuf[j]){
                    temp[k] = arr[i];
                    k++; i++;
                }
                else {
                    temp[k] = recvbuf[j];
                    k++; j++;
                }                
            }
            // copying remaining elements
            while (i < local_n){
                temp[k] = arr[i];
                k++; i++;
            }
            // copying remaining elements
            while (j < partner_n){
                temp[k] = recvbuf[j];
                k++; j++;
            }
            if (rank < partner) {
                for (size_t i = 0; i < local_n; i++)
                    arr[i] = temp[i];
            }
            else{
                for (size_t i = 0; i < local_n; i++)
                    arr[i] = temp[partner_n + i];
            }
        }
    }
    free(temp);
    free(recvbuf);
}