// serial dot product

#include <stdio.h>
#include <stdlib.h>
#include <time.h>

int main()
{
    int* a;
    int* b;
    int c = 0;
    int n = 1e6;
    double start, end;
    srand(3);
    a = malloc(n * sizeof(int));
    b = malloc(n * sizeof(int));
    for (size_t i = 0; i < n; i++){
        // a[i] = rand() % 1000;
        // b[i] = rand() % 1000;
        a[i] = 1;
        b[i] = 1;
    }
    start = clock();
    for (size_t i = 0; i < n; i++){
        c += a[i] * b[i];
    }
    end = clock();
    printf("%f seconds\n", ((double)(end, start))/CLOCKS_PER_SEC);
    printf("Dot product is: %d.\n", c);

    return 0;
}