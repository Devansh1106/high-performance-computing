// parallel gauss-jacobi method (when n % size can be anything)

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>

double* matrix_vec_prod(double* A, double* vec, int n, double* result_vec, int rank, int size, MPI_Comm comm);
void gauss_jacobi(double* A, double* rhs_vec, double* initial_guess, int n, int max_iter, double tol, int rank, int size, MPI_Comm comm);


int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm comm = MPI_COMM_WORLD;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    double* A=NULL;
    double* rhs_vec=NULL;
    double* initial_guess=NULL;
    // double* sol=NULL;
    int n = 1000; 
    double max_tol = 1e-4;
    int max_iter = 10000;
    // Load balancing
    int rem = n % size;
    int local_n = (n/size) + (rank < rem ? 1 : 0);

    if (rank == 0){
        A = malloc(n * n * sizeof(double));
        rhs_vec = malloc(n * sizeof(double));
        initial_guess = calloc(n, sizeof(double)); // 0 as initial guess

        for(size_t i = 0; i < n; i++){
            for(size_t j = 0; j < n; j++){
                A[i * n + j] = 1.0;
            }
            A[i * n + i] = n;
            rhs_vec[i] = 1.0;
        }
        gauss_jacobi(A, rhs_vec, initial_guess, n, max_iter, max_tol, rank, size, comm);
    }
    else{
        A = malloc(local_n * n * sizeof(double));
        rhs_vec = malloc(n * sizeof(double));
        initial_guess = calloc(n, sizeof(double));
        gauss_jacobi(A, rhs_vec, initial_guess, n, max_iter, max_tol, rank, size, comm);
    }
    free(A);
    free(rhs_vec);
    free(initial_guess);
    MPI_Finalize();
    return 0;
}

void gauss_jacobi(double* A, double* rhs_vec, double* initial_guess, int n, int max_iter, double max_tol, int rank, int size, MPI_Comm comm){
    double start_time = 0.0;
    double end_time = 0.0, time = 0.0;
    int iter = 0.0;
    double tol = 1.0;
    double sum = 0.0;
    int* sendcount1 =NULL;
    int* displs1    =NULL;
    int* sendcount2 =NULL;
    int* displs2    =NULL; 
    double* result_vec=NULL;
    int flag = -1;
    double* diag=NULL;
    double* temp=NULL;
    
    // Load balancing
    int rem     = n % size;
    int local_n = (n/size) + (rank < rem ? 1 : 0);

    diag        = malloc(n * sizeof(double));
    result_vec  = calloc(local_n, sizeof(double));
    temp        = calloc(local_n, sizeof(double));
    sendcount1  = malloc(size * sizeof(int));
    displs1     = malloc(size * sizeof(int));
    sendcount2  = malloc(size * sizeof(int));
    displs2     = malloc(size * sizeof(int));

    if (rank == 0){
        for(size_t i = 0; i < n; i++){
            diag[i]     = A[i*n + i];
            A[i*n + i]  = 0.0;
        }    
    }
    for(size_t i = 0; i < size; i++){
        int count = (n/size) + (i < rem ? 1 : 0);
        sendcount1[i] = count * n;
        sendcount2[i] = count;
        displs1[i] = (i == 0) ? 0 : displs1[i - 1] + sendcount1[i - 1];
        displs2[i] = (i == 0) ? 0 : displs2[i - 1] + sendcount2[i - 1];
    }

    start_time = MPI_Wtime();
    MPI_Scatterv(A, sendcount1, displs1, MPI_DOUBLE, A, (local_n * n), MPI_DOUBLE, 0, comm);
    MPI_Scatterv(diag, sendcount2, displs2, MPI_DOUBLE, diag, local_n, MPI_DOUBLE, 0, comm);
    MPI_Bcast(rhs_vec, n, MPI_DOUBLE, 0, comm);
    MPI_Bcast(initial_guess, n, MPI_DOUBLE, 0, comm); 

    while((tol > max_tol) && (iter != max_iter)){
        sum = 0.0;
        result_vec = matrix_vec_prod(A, initial_guess, n, result_vec, rank, size, comm);
        for (size_t i = 0; i < local_n; i++){
            temp[i] = rhs_vec[displs2[rank] + i] - result_vec[i];
            temp[i] = temp[i] / diag[i];
            sum     += fabs(temp[i] - initial_guess[displs2[rank] + i]);
        }
        MPI_Allgatherv(temp, local_n, MPI_DOUBLE, initial_guess, sendcount2, displs2, MPI_DOUBLE, comm);
        MPI_Allreduce(&sum, &tol, 1, MPI_DOUBLE, MPI_SUM, comm);
        printf("\nTol: %f \t Rank:\t%d", tol, rank);
        printf("\t Iter: \t%d\n", iter);
        iter++;
    }
    end_time = MPI_Wtime();
    time = (end_time - start_time);
    if (rank == 0){
        if (iter == max_iter){
            printf("Failed to converge, max iterations reached!\n");
        }
        else{
            printf("Solution: \n");
            for (size_t i = 0; i < 100; i++){
                printf("%f ", initial_guess[i]);
            }
        }
        printf("\nTime taken is: %f sec.\n", time);
    }
    free(result_vec);
    free(temp);
    free(diag);
    // return initial_guess;
}

double* matrix_vec_prod(double* A, double* vec, int n, double* result_vec, int rank, int size, MPI_Comm comm){
    int rem = n % size;
    int local_n = (n/size) + (rank < rem ? 1 : 0);           // n % size should be 0 for now
    // reseting is needed otherwise it will have the value from the previous call
    for(size_t i = 0; i < local_n; i++) {
        result_vec[i] = 0.0;
    }

    for(size_t i = 0; i < local_n; i++){
        for(size_t j = 0; j < n; j++){
            result_vec[i] += A[i * n + j] * vec[j];
        }
    }
    return result_vec;
}