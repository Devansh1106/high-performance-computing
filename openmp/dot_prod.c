#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

int main(int argc, char **argv){
    long long int c = 0;
    double start, end;
    double elapsed = 0.0;
    int n = 1e7;

    start = omp_get_wtime();
    #pragma omp parallel
    {   
        int thread_id = omp_get_thread_num();
        long long int local_sum = 0;
        #pragma omp for
        for (size_t i = 0; i < n; i++){
            unsigned int seed = i + 13;
            long long int a = rand_r(&seed) % 3;
            long long int b = rand_r(&seed) % 3;
            // long long int a = 1;
            // long long int b = 1;
            local_sum += a * b;
        }

        #pragma omp atomic
        c += local_sum;

        if (thread_id == 0)
            printf("Total number of threads are: %d.\n", omp_get_num_threads());
    }
    end = omp_get_wtime();
    printf("Answer is: %lld  ", c);

    elapsed = (end - start);
    printf("Time taken is: %.6f\n", elapsed);
    return 0;
}
