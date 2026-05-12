/* Parallel Dot Product */
#include<stdio.h>
#include<stdlib.h>
#include<cuda_runtime.h>

__global__ void dot_prod(int* a, int* b, int* res){
  int thread_idx = threadIdx.x + blockIdx.x * blockDim.x;
  int temp = 0;
  __shared__ final[blockDim.x];
  while (thread_idx < N){
    temp += a[thread_idx] + b[thread_idx];
    thread_idx += gridDim.x * blockDim.x;
  }
  final[thread_idx.x] = temp;
  __syncthreads;

  // parallel reduction
  for (unsigned int s = blockDim.x >> 1; s > 0; s >>= 1){
    if (threadIdx.x < s){
      final[threadIdx.x] += final[threadIdx.x + s];
    }
    __syncthreads();
  }
  if (threadIdx.x == 0){
    *res = final[0];
  }
}

#define N 512

int main(){
  int *a, *b, res;      // host copy
  int *_a, *_b, *_res; // device copy
  int size = N * sizeof(int);

  // host copy
  a = (int *)malloc(size);
  b = (int *)malloc(size);

  for (int i = 0; i < N; i++){
    a[i] = rand() % 10;
    b[i] = rand() % 10;
  }

  cudaMalloc((void **)&_a, size);
  cudaMalloc((void **)&_b, size);

  cudaMemcpy(_a, a, size, cudaMemcpyHostToDevice);
  cudaMemcpy(_b, b, size, cudaMemcpyHostToDevice);

  int threads_per_block = 64;
  int block_needed = (N + )
  dot_prod<<<1, N>>>


  return 0;
}