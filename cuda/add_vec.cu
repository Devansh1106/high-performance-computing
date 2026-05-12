// Addition of vectors (executing many blocks and 1 thread)
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

__global__ void add(int* a, int* b, int* c){
  c[blockIdx.x] = a[blockIdx.x] + b[blockIdx.x];
}

#define N 512
int main(){
  int *a, *b, *c;      // host copy
  srand(3);
  int *_a, *_b, *_c;  // device copy
  int size = N * sizeof(int);

  // Allocate space for device copies of a, b, c
  cudaMalloc((void** )&_a, size);
  cudaMalloc((void** )&_b, size);
  cudaMalloc((void** )&_c, size);

  // Allocate space for host
  a = (int *)malloc(size);
  b = (int *)malloc(size);
  c = (int *)malloc(size);

  // random values
  for (int i = 0; i < N; i++){
    a[i] = rand() % 10;
    b[i] = rand() % 10;
  }

  // copy inputs to device
  cudaMemcpy(_a, a, size, cudaMemcpyHostToDevice);
  cudaMemcpy(_b, b, size, cudaMemcpyHostToDevice);

  // launch kernel
  // N refers to N parallel launch of kernel and each parallel call
  // refers to a block
  add<<<N, 1>>>(_a, _b, _c);

  // copy result back to host
  cudaMemcpy(c, _c, size, cudaMemcpyDeviceToHost);
  printf("%d %d %d\n", a[0], a[1], a[2]);
  printf("%d %d %d\n", b[0], b[1], b[2]);
  printf("%d %d %d\n", c[0], c[1], c[2]);

  // free
  free(a); free(b); free(c); // freeing at the host
  cudaFree(_a); cudaFree(_b), cudaFree(_c); // freeing at the device
  return 0;
}