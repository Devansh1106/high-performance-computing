#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

__global__ void add(int* a, int* b, int* c){
  *c = *a + *b;
}


int main(){
  int a, b, c;      // host copy
  int *_a, *_b, *_c;  // device copy
  int size = sizeof(int);

  // Allocate space for device copies of a, b, c
  cudaMalloc((void** )&_a, size);
  cudaMalloc((void** )&_b, size);
  cudaMalloc((void** )&_c, size);

  a = 2; b = 3;

  // copy inputs to device
  cudaMemcpy(_a, &a, size, cudaMemcpyHostToDevice);
  cudaMemcpy(_b, &b, size, cudaMemcpyHostToDevice);

  // launch kernel
  add<<<1, 1>>>(_a, _b, _c);

  // copy result back to host
  cudaMemcpy(&c, _c, size, cudaMemcpyDeviceToHost);
  printf("%d", c);

  // free
  cudaFree(_a); cudaFree(_b), cudaFree(_c);
  return 0;
}