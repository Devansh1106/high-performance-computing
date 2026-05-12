/* parallel reduction Sum using Tree based reduction */
#include<stdio.h>
#include<stdlib.h>
#include <cuda_runtime.h>

__global__ void vec_sum(int* vec, int* sum, unsigned int n){
  for (unsigned int s = blockDim.x >> 1; s > 0; s >>= 1){
    if (threadIdx.x < s){
      vec[threadIdx.x] += vec[threadIdx.x + s];
    }
    __syncthreads();
  }
  if (threadIdx.x == 0){
    *sum = vec[0];
  }
}


#define n 512
int main(){
  srand(3);
  int *vec, sum = 0;    // host copy
  int *_vec, *_sum;    // device  copy
  int size = n * sizeof(int);

  // host memory allocation
  vec = (int *)malloc(size);
  for (int i = 0; i < n; i++){
    vec[i] = rand() % 10;
  }

  // device memory allocation
  cudaMalloc((void **)&_vec, size);
  cudaMalloc((void **)&_sum, sizeof(int));

  cudaMemcpy(_vec, vec, size, cudaMemcpyHostToDevice);

  int threads_per_block = 512;
  // int blocks_needed = (n + threads_per_block - 1) / threads_per_block;
  vec_sum<<<1, threads_per_block>>>(_vec, _sum, n);

  // copying result to host
  cudaMemcpy(&sum, _sum, sizeof(int), cudaMemcpyDeviceToHost);
  printf("Sum is %d.\n", sum);

  free(vec);
  cudaFree(_vec); cudaFree(_sum);
  return 0;
}