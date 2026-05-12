// calculate the value of pi using its approximation

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <math.h>
#include <time.h>

#define n 1000000

__global__ void pi(double* res){
  int i = threadIdx.x + blockDim.x * blockIdx.x;
  if (i < n){
    double x = (double)i / n;
    double h = 1 - x*x;
    double area = sqrt(h) / n;
    atomicAdd(res, area);
  }
}
int main(){
  double res = 0;
  double* d_res;

  cudaMalloc((void**)&d_res, sizeof(double));

  int threads_per_block = 256;
  int blocks_needed = (n + threads_per_block - 1)/ threads_per_block;
  pi<<<blocks_needed, threads_per_block>>>(d_res);
  cudaMemcpy(&res, d_res, sizeof(double), cudaMemcpyDeviceToHost);
  printf("Pi value is: %f\n", 4*res);

  cudaFree(d_res);
  return 0;
}