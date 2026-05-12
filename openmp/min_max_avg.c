#include <stdio.h>
#include <omp.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>

#define N 10000000
int main(){
    srand(3);
    double* a = malloc(N *sizeof(double));
    double sum=0, min=1e9, max=-1e9;

    for (int i = 0; i < N; i++){
        a[i] = (double)rand() / (double)RAND_MAX;
    }

    double start_time = omp_get_wtime();

    #pragma omp parallel for reduction(+:sum)
    for (int i=0; i < N; i++)
        sum += a[i];

    #pragma omp parallel for reduction(min:min)
    for (int i = 0; i < N; i++)
        if (a[i] < min) min = a[i];

    #pragma omp parallel for reduction(max:max)
    for (int i = 0; i<N; i++)
        if (a[i] > max) max = a[i];

    double mean = sum/N;

    double norm=0;
    #pragma omp parallel for reduction(+:norm)
    for (int i=0; i< N;i++)
        norm += a[i]*a[i];

    norm = sqrt(norm);
    double end_time = omp_get_wtime();

    printf("Time taken: %f\n", end_time - start_time);

    printf("Norm=%f, Min=%f, Max=%f, Mean=%f", norm, min, max, mean);
    free(a);
    return 0;
}