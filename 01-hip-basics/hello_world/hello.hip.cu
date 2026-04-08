// hello.hip.cu — 第一个 HIP 程序
// 编译: hipcc hello.hip.cu -o hello && ./hello

#include <hip/hip_runtime.h>
#include <stdio.h>

__global__ void hello() {
    int tid = hipThreadIdx_x;
    printf("Hello from GPU thread %d (block %d)\n",
           tid, hipBlockIdx_x);
}

int main() {
    printf("=== HIP Hello World ===\n");
    printf("Launching 2 blocks × 4 threads = 8 total threads\n\n");

    // dim3: (blocks, threads_per_block)
    hipLaunchKernelGGL(hello, dim3(2), dim3(4), 0, 0);

    hipDeviceSynchronize();
    printf("\nDone.\n");
    return 0;
}
