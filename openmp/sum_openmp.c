#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

int main(int argc, char **argv){
    int* a;
    int* b;
    int* c;
    // int i;
    a = calloc(100, sizeof(int));
    b = calloc(100, sizeof(int));
    c = calloc(100, sizeof(int));

    #pragma omp parallel
    {
        #pragma omp for
        for (int i = 0; i < 100; i++){
            a[i] = b[i] + c[i];
        }
    }
    return 0;
}
