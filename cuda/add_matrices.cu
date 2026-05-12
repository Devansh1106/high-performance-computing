/* Adding two matrices */

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

__global__ void mat_sum(int* A, int* B, int* sum, int N){

  // calculate 2D block id
  int block_id = blockIdx.y * gridDim.x + blockIdx.x;

  // calculate 3D thread id within block
  int thread_id = threadIdx.z * blockDim.y * blockDim.x
                  + threadIdx.y * blockDim.x
                  + threadIdx.x;
  int threads_per_block = blockDim.x * blockDim.y * blockDim.z;

  // global index
  int i = thread_id + threads_per_block * block_id;

  if (i < N * N){
    sum[i] = A[i] + B[i];
  }
}

int main (){
  srand(3);
  int *A, *B, *sum; // host copy
  int *_A, *_B, *_sum;  // device copy
  int N = 512;
  int size = N * N * sizeof(int);

  // host memory allocation
  A = (int *)malloc(size);
  B = (int *)malloc(size);
  sum = (int *)malloc(size);

  for (int i = 0; i < N; i++){
    for (int j = 0; j < N; j++){
      A[i * N + j] = rand() % 10;
      B[i * N + j] = rand() % 10;

      // A[i * N + j] = 1;
      // B[i * N + j] = 0;
    }
  }

  // device memory allocation
  cudaMalloc((void **)&_A, size);
  cudaMalloc((void **)&_B, size);
  cudaMalloc((void **)&_sum, size);

  // input data copy from host to device
  cudaMemcpy(_A, A, size, cudaMemcpyHostToDevice);
  cudaMemcpy(_B, B, size, cudaMemcpyHostToDevice);

  // automatic block and grid size calculation
  int total_elements = N * N;
  int threads_per_block = 256; // 32 * 4 * 2
  int blocks_needed = (total_elements + threads_per_block - 1) / threads_per_block;

  dim3 grid(blocks_needed / 4, 4); // both / x, x should be same; block config in 2D
  dim3 block(32, 4, 2); // thread config in 3D; product of all these dims = threads_per_block
  mat_sum<<<grid, block>>>(_A, _B, _sum, N);

  // result copy from device to host
  cudaMemcpy(sum, _sum, size, cudaMemcpyDeviceToHost);

  for (int i = 0; i < 10; i++){
    for (int j = 0; j < 10; j++){
      printf("%d ", sum[i*N + j]);
    }
    printf("\n");
  }
  free(A); free(B); free(sum);
  cudaFree(_A); cudaFree(_B); cudaFree(_sum);
  return 0;
}