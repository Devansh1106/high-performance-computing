// %%writefile multi_gpu_matvec.cu
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <time.h>

/* ---------------------------------------------------------
 * Serial Matrix-Vector Multiplication (CPU)
 * --------------------------------------------------------- */
void serialMatVec(const float* A, const float* x, float* y, int M, int N) {
    for (int i = 0; i < M; ++i) {
        float sum = 0.0f;
        for (int j = 0; j < N; ++j) {
            sum += A[i * N + j] * x[j];
        }
        y[i] = sum;
    }
}

/* ---------------------------------------------------------
 * Compute Kernel: Local Matrix-Vector Multiplication
 * --------------------------------------------------------- */
__global__ void matVecMulKernel(const float* A, const float* x, float* y, int numRows, int N) {
    // Calculate global row index for this thread
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Boundary check to prevent memory access violations
    if (row < numRows) {
        float sum = 0.0f;
        for (int j = 0; j < N; ++j) {
            sum += A[row * N + j] * x[j];
        }
        y[row] = sum;
    }
}

/* ---------------------------------------------------------
 * Host Function: Multi-GPU Orchestration
 * --------------------------------------------------------- */
float multiGpuMatVec(const float* h_A, const float* h_x, float* h_y, int M, int N) {
    int numGPUs;
    cudaGetDeviceCount(&numGPUs);
    
    if (numGPUs == 0) {
        printf("Error: No CUDA-capable devices found.\n");
        return 0.0f;
    }

    // Assuming M is perfectly divisible by the number of GPUs for this baseline
    // numGPUs = 1;
    int rowsPerGPU = M / numGPUs; 
    printf("Number of GPUs: %d\n", numGPUs);

    // Allocate host arrays to hold device pointers and streams using standard C malloc
    float** d_A = (float**)malloc(numGPUs * sizeof(float*));
    float** d_x = (float**)malloc(numGPUs * sizeof(float*));
    float** d_y = (float**)malloc(numGPUs * sizeof(float*));
    cudaStream_t* streams = (cudaStream_t*)malloc(numGPUs * sizeof(cudaStream_t));

    /* ---------------------------------------------------------
     * Task 1 & 2: Partition Matrix Rows and Broadcast Vector x
     * --------------------------------------------------------- */
    for (int d = 0; d < numGPUs; ++d) {
        cudaSetDevice(d);
        cudaStreamCreate(&streams[d]);

        // Allocate memory on the current device
        cudaMalloc((void**)&d_A[d], rowsPerGPU * N * sizeof(float));
        cudaMalloc((void**)&d_x[d], N * sizeof(float));
        cudaMalloc((void**)&d_y[d], rowsPerGPU * sizeof(float));
    }

    cudaEvent_t start, stop;
    cudaSetDevice(0);
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start); // timer starts

    for (int d = 0; d < numGPUs; ++d) {
        cudaSetDevice(d);

        // Scatter row-blocks of A
        size_t offset_A = d * rowsPerGPU * N;
        cudaMemcpyAsync(d_A[d], h_A + offset_A, rowsPerGPU * N * sizeof(float), cudaMemcpyHostToDevice, streams[d]);

        // Broadcast the full vector x
        cudaMemcpyAsync(d_x[d], h_x, N * sizeof(float), cudaMemcpyHostToDevice, streams[d]);
    }

    /* ---------------------------------------------------------
     * Task 3a: Perform Local Computation
     * --------------------------------------------------------- */
    int threadsPerBlock = 256;
    int blocksPerGrid = (rowsPerGPU + threadsPerBlock - 1) / threadsPerBlock;

    for (int d = 0; d < numGPUs; ++d) {
        cudaSetDevice(d);
        matVecMulKernel<<<blocksPerGrid, threadsPerBlock, 0, streams[d]>>>(
            d_A[d], d_x[d], d_y[d], rowsPerGPU, N
        );
    }

    /* ---------------------------------------------------------
     * Task 3b: Gather Results
     * --------------------------------------------------------- */
    for (int d = 0; d < numGPUs; ++d) {
        cudaSetDevice(d);
        
        // Offset for the output vector y based on the GPU index
        size_t offset_y = d * rowsPerGPU;
        cudaMemcpyAsync(h_y + offset_y, d_y[d], rowsPerGPU * sizeof(float), cudaMemcpyDeviceToHost, streams[d]);
    }

    /* ---------------------------------------------------------
     * Synchronization
     * --------------------------------------------------------- */
    for (int d = 0; d < numGPUs; ++d) {
        cudaSetDevice(d);
        // Wait for all asynchronous operations on this stream to finish
        cudaStreamSynchronize(streams[d]);
    }

    cudaSetDevice(0);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    /* ---------------------------------------------------------
     * Cleanup
     * --------------------------------------------------------- */
    for (int d = 0; d < numGPUs; ++d) {
        cudaSetDevice(d);
        
        // Free device memory
        cudaFree(d_A[d]);
        cudaFree(d_x[d]);
        cudaFree(d_y[d]);
        cudaStreamDestroy(streams[d]);
    }

    // Free host memory arrays
    free(d_A);
    free(d_x);
    free(d_y);
    free(streams);

    return milliseconds;
}

int main() {
    int M = 8192;
    int N = 8192;

    float *h_A, *h_x, *h_y_gpu;
    
    // Allocate pinned memory for asynchronous transfers
    cudaMallocHost((void**)&h_A, M * N * sizeof(float));
    cudaMallocHost((void**)&h_x, N * sizeof(float));
    cudaMallocHost((void**)&h_y_gpu, M * sizeof(float));
    float* h_y_cpu = (float*)malloc(M * sizeof(float));
                    

    // matrix
    // for (int i = 0; i < M * N; ++i) 
        // h_A[i] = 1.0f;

    // vector
    // for (int i = 0; i < N; ++i)
        // h_x[i] = 1.0f;
    
    srand(42);
    
    // matrix
    for (int i = 0; i < M * N; ++i)
        h_A[i] = (float)rand() / RAND_MAX;
    
    // vector
    for (int i = 0; i < N; ++i)
        h_x[i] = (float)rand() / RAND_MAX;

    // CPU IMPLEMENTATION (Serial)
    clock_t cpu_start = clock();
    serialMatVec(h_A, h_x, h_y_cpu, M, N);
    clock_t cpu_end = clock();
    double cpu_ms = ((double)(cpu_end-cpu_start) / CLOCKS_PER_SEC) * 1000.0;

    // GPU IMPLEMENTATION
    float gpu_ms = multiGpuMatVec(h_A, h_x, h_y_gpu, M, N);

    printf("M = N =   %d\n", M);
    printf("Serial Time: (CPU) %f ms\n", cpu_ms);
    printf("Multi-GPU Time: %f ms\n", gpu_ms);
    printf("Verification: y[0](gpu) = %f and (cpu) %f\n", h_y_gpu[0], h_y_cpu[0]);
    // printf("Verification: y[0](gpu) = %f\n", h_y_gpu[0]);
    

    free(h_A); free(h_x); free(h_y_gpu); free(h_y_cpu);
    return 0;
}