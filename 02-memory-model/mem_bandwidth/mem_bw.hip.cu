// mem_bw.hip.cu — 显存带宽测试
// 编译: hipcc mem_bw.hip.cu -o mem_bw && ./mem_bw

#include <hip/hip_runtime.h>
#include <stdio.h>
#include <hip/hip_runtime_api.h>

#define N (1024 * 1024 * 64)  // 256MB
#define Iterations 100

__global__ void write_kernel(float* data, float val) {
    int i = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
    if (i < N) {
        data[i] = val;
    }
}

__global__ void read_kernel(float* data, float* out) {
    int i = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
    if (i < N) {
        *out += data[i];
    }
}

int main() {
    float *d_data;
    float result = 0.0f;

    hipMalloc(&d_data, N * sizeof(float));

    // Warm-up
    hipLaunchKernelGGL(write_kernel, dim3(N/256), dim3(256), 0, 0, d_data, 1.0f);
    hipDeviceSynchronize();

    // ---- Write bandwidth ----
    hipEvent_t start, stop;
    hipEventCreate(&start);
    hipEventCreate(&stop);

    hipEventRecord(start);
    for (int i = 0; i < Iterations; i++) {
        hipLaunchKernelGGL(write_kernel, dim3(N/256), dim3(256), 0, 0, d_data, (float)i);
    }
    hipEventRecord(stop);
    hipDeviceSynchronize();

    float ms;
    hipEventElapsedTime(&ms, start, stop);
    double write_bw = (double)N * sizeof(float) * Iterations / (ms / 1000.0) / 1e9;
    printf("Write bandwidth: %.2f GB/s  (%.2f GB/s theoretical max)\n",
           write_bw, 1000.0);  // RX 9070 XT ~1000 GB/s

    // ---- Read bandwidth ----
    hipEventRecord(start);
    for (int i = 0; i < Iterations; i++) {
        hipLaunchKernelGGL(read_kernel, dim3(N/256), dim3(256), 0, 0, d_data, &result);
    }
    hipEventRecord(stop);
    hipDeviceSynchronize();

    hipEventElapsedTime(&ms, start, stop);
    double read_bw = (double)N * sizeof(float) * Iterations / (ms / 1000.0) / 1e9;
    printf("Read bandwidth:  %.2f GB/s\n", read_bw);

    hipFree(d_data);
    hipEventDestroy(start);
    hipEventDestroy(stop);

    return 0;
}
