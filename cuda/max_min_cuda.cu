#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda_runtime.h>
#define n 50000

__global__ void _sum(int* vec, int* sum){
  int ix = threadIdx.x + blockIdx.x * blockDim.x;

  if (ix < n){
    atomicAdd(sum, vec[ix]);
  }
}

__global__ void _min(int* vec, int* min){
  int ix = threadIdx.x + blockIdx.x * blockDim.x;

  if (ix < n){
    atomicMin(min, vec[ix]);
  }
}

__global__ void _max(int* vec, int* max){
  int ix = threadIdx.x + blockIdx.x * blockDim.x;

  if (ix < n){
    atomicMax(max, vec[ix]);
  }
}

int main (){
  srand(3);
  int *vec;
  int min=1e9, max=-1e9, sum=0;
  int size = n * sizeof(int);

  int *d_vec;
  int *d_min, *d_max, *d_sum;

  vec = (int *)malloc(n * sizeof(int));
  cudaMalloc((void**)&d_vec, n * sizeof(int));
  cudaMalloc((void**)&d_sum, sizeof(int));
  cudaMalloc((void**)&d_min, sizeof(int));
  cudaMalloc((void**)&d_max, sizeof(int));



  for(int i=0; i<n; i++){
    vec[i] = rand() % 10;
  }

  cudaMemcpy(d_vec, vec, size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_min, &min, sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(d_max, &max, sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(d_sum, &sum, sizeof(int), cudaMemcpyHostToDevice);

  int threads_per_block = 64;
  int blocks_needed = (n + threads_per_block - 1) / threads_per_block;
  // max
  clock_t start1 = clock();
  _max<<<blocks_needed, threads_per_block>>>(d_vec, d_max);
  cudaDeviceSynchronize();
  clock_t end1 = clock();
  cudaMemcpy(&max, d_max, sizeof(int), cudaMemcpyDeviceToHost);
  double time1 = ((double) (end1- start1))/CLOCKS_PER_SEC;
  printf("Time: %f\n", time1);
  printf("Max is: %d\n", max);

  // min
  clock_t start2 = clock();
  _min<<<blocks_needed, threads_per_block>>>(d_vec, d_min);
  cudaDeviceSynchronize();
  clock_t end2 = clock();
  cudaMemcpy(&min, d_min, sizeof(int), cudaMemcpyDeviceToHost);
  double time2 = ((double) (end2 - start2))/CLOCKS_PER_SEC;
  printf("Time: %f\n", time2);
  printf("Min is: %d\n", min);

  // sum
  clock_t start3 = clock();
  _sum<<<blocks_needed, threads_per_block>>>(d_vec, d_sum);
  cudaDeviceSynchronize();
  clock_t end3 = clock();
  cudaMemcpy(&sum, d_sum, sizeof(int), cudaMemcpyDeviceToHost);
  double time3 = ((double) (end3 - start3))/CLOCKS_PER_SEC;
  printf("Time: %f\n", time3);
  printf("sum is: %d\n", sum);

  free(vec);
  cudaFree(d_vec); cudaFree(d_sum); cudaFree(d_max); cudaFree(d_min);

  return 0;
}