#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <math.h>

int main () {
    int N;
    printf("enter N:\n");
    scanf("%d", &N);
    double* A = malloc(sizeof(double) * N * N);
    double* b = malloc(sizeof(double) * N);
    double* x_old = malloc(sizeof(double) * N);
    double* x_new = malloc(sizeof(double) * N);


    for (int i = 0; i < N; i++){
        for (int j = 0; j < N; j++){
            A[i*N +j] = 1;
        }
        A[i*N+i] = N+2;
        b[i]= 2.5+N;
        x_old[i] = 0;
    }

    double tol = 1e-1;
    double error;
    int iter = 0;

    double start = omp_get_wtime();

    do {
        iter++;
        #pragma omp parallel for
        for (int i = 0; i<N; i++){
            double sum = 0;
            for (int j = 0; j<N; j++){
                if (j != i)
                    sum += A[i*N+j] * x_old[j];
            }
            x_new[i] = (b[i] - sum) / A[i*N +i];
        }
        error = 0; // imp dont forget
        #pragma omp parallel for reduction(+:error)
        for (int i = 0; i<N; i++){
            error += (x_new[i] - x_old[i]) * (x_new[i] - x_old[i]); 
        }

        error = sqrt(error);

        // update sol
        #pragma omp parallel for
        for (int i = 0; i<N; i++)
            x_old[i] = x_new[i];
    } while (error > tol);

    double end = omp_get_wtime();
    printf("Time taken: %f\n", end-start);
    printf("Iterations: %d\n", iter);

    // for (int i=0;i<N;i++)
    //     printf("%f ", x_new[i]);

    free(A); free(b); free(x_new); free(x_old);

    return 0;
}