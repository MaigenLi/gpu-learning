# 01 — HIP 基础

## 目标
- 掌握 HIP 程序的基本结构
- 会编译、运行、简单调优
- 理解 thread/warp/block/grid 层次

## 包含内容

```
01-hip-basics/
├── hello_world/           # 第一个 HIP kernel
├── vector_add/            # 向量加法 — 最基础的 data parallel kernel
├── matrix_mul/            # 矩阵乘法 — 理解 thread 映射
└── README.md              # 本文件
```

## Hello World

```cpp
// hello.hip.cu
#include <hip/hip_runtime.h>
#include <stdio.h>

__global__ void hello() {
    printf("Hello from GPU thread %d\n", hipThreadIdx_x);
}

int main() {
    printf("Launching 1 block × 4 threads\n");
    hello<<<1, 4>>>();
    hipDeviceSynchronize();
    return 0;
}
```

编译运行：
```bash
hipcc hello.hip.cu -o hello && ./hello
```

## CUDA → HIP 快速对照

| CUDA | HIP |
|------|-----|
| `cudaMalloc` | `hipMalloc` |
| `cudaMemcpy` | `hipMemcpy` |
| `cudaFree` | `hipFree` |
| `cudaDeviceSynchronize` | `hipDeviceSynchronize` |
| `__global__` | `__global__` (相同) |
| `threadIdx.x` | `hipThreadIdx_x` |
| `blockIdx.x` | `hipBlockIdx_x` |
| `blockDim.x` | `hipBlockDim_x` |
| `gridDim.x` | `hipGridDim_x` |

## 参考

- HIP Programming Guide: `$ROCM_PATH/docs/html/hip/index.html`
- 官方 Samples: `$ROCM_PATH/hip/samples/`
