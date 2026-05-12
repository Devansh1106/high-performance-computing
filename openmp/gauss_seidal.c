#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>

int main(){
    int N;
    printf("Enter N: \n");
    scanf("%d", &N);
    double* A = malloc(N * N * sizeof(double));
    double* b = malloc(N * sizeof(double));
    double* initial_guess = malloc(N * sizeof(double));
    double* x_old = malloc(N * sizeof(double));
    // double A[N][N], b[N], initial_guess[N], x_old[N];

    for (int i = 0; i < N; i++){
        for (int j = 0; j < N; j++){
            A[i*N+j] = 1; 
        }
        A[i*N+i] = N+2; 
        b[i] = 2.5 + N;
        initial_guess[i] = 0.0;
    }

    double tol = 1e-1;
    double error = 1.0;

    double start = omp_get_wtime();
    int iter = 0;
    do {
        iter++;
        // copy old solution
        #pragma omp parallel for
        for (int i=0; i<N; i++){
            x_old[i] = initial_guess[i];
        }
        for (int i=0; i<N; i++){
            double sum = 0;

            // j < i (new values)
            #pragma omp parallel for reduction(+:sum)
            for (int j=0; j<i; j++){
                sum += A[i*N+j] * initial_guess[j];
            }

            // j > i (old values)
            #pragma omp parallel for reduction(+:sum)
            for (int j=i+1; j<N; j++){
                sum += A[i*N+j] * x_old[j];
            }

            initial_guess[i] = (b[i] - sum) / A[i*N+i];
        }

        // error computation
        error = 0; // imp to reset
        #pragma omp parallel for reduction(+:error)
        for (int i = 0; i<N; i++){
            error += (initial_guess[i] - x_old[i]) * (initial_guess[i] - x_old[i]);
        }
        error = sqrt(error);
    } while (error > tol);

    double end = omp_get_wtime();
    printf("Iterations: %d\n", iter);
    printf("Time taken %f\n", end-start);

    // for (int i=0;i<N;i++)
    //     printf("%f ", initial_guess[i]);



    return 0;
}