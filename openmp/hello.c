#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

int main(int argc, char **argv){
    int num_threads, thread_id;

    #pragma omp parallel private(num_threads, thread_id)
    {
        thread_id = omp_get_thread_num();
        printf("Hello from %d\n", thread_id);
        if (thread_id == 0){
            num_threads = omp_get_num_threads();
            printf("\nTotal threads: %d\n", num_threads);
        }
    }
    return 0;
}
