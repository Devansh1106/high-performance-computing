// matrix vector product using cuda

#include <stdio.h>
#include <cuda_runtime.h>
#include <stdlib.h>
#include <time.h>
#include <float.h>

#define n 100
#define threads_per_block 64

__global__ void mat_vec(double* res, double* A, double* vec){
  int i = threadIdx.x + blockDim.x * blockIdx.x;
  if (i < n){
    double sum = 0.0;
    for (int j = 0; j < n; j++){
      sum += A[i*n + j] * vec[j];
    }
    res[i] = sum;
  }
}

int main () {
  double *A, *vec, *res;
  srand(3);
  double *d_A, *d_vec, *d_res;
  int size = n * sizeof(double);
  A = (double*) malloc(n * n * sizeof(double));
  vec = (double*)  malloc(size);
  res = (double*) calloc(size, sizeof(double));

  cudaMalloc((void**)&d_A, n*n*sizeof(double));
  cudaMalloc((void**)&d_vec, n*sizeof(double));
  cudaMalloc((void**)&d_res, n*sizeof(double));

  for ( int i = 0; i < n; i++){
    for (int j = 0; j < n; j++){
      A[i*n + j] = (double)rand() / RAND_MAX;
    }
    vec[i] = (double)rand() / RAND_MAX;
  }
  cudaMemcpy(d_A, A, n * size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_vec, vec, size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_res, res, size, cudaMemcpyHostToDevice);
  cudaMemset(d_res, 0, size);

  int blocks_needed = (n + threads_per_block - 1) / threads_per_block;
  clock_t start = clock();
  mat_vec<<<blocks_needed, threads_per_block>>>(d_res, d_A, d_vec);
  cudaDeviceSynchronize();
  clock_t end = clock();
  printf("Time: %f\n", ((double)(end-start)/CLOCKS_PER_SEC));

  cudaMemcpy(res, d_res, size, cudaMemcpyDeviceToHost);
  printf("%f\t %f", res[12], res[99]);


  return 0;
}