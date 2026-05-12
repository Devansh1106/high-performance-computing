#include <stdio.h>
#include <mpi.h>
#include <time.h>

#define row 50
#define col 50000

int main(int* argc, char** argv){
    double *A, *B, *R;
    A = (double*)malloc(row*col *sizeof(double));
    B = (double*)malloc(row*col *sizeof(double));
    R = (double*)calloc(row*col, sizeof(double));

    clock_t start = clock();
    bp2p(A, B, R);
    clock_t end = clock();
    printf("Time ")

    

    return 0;
}