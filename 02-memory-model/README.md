# 02 — 内存模型

## AMD GPU 内存层次

```
Host (CPU DDR)
    ↓ hipMemcpy
GPU Global Memory (GDDR7, ~1000 GB/s)
    ↓
L2 Cache (per CU, shared)
    ↓
L1 / Shared Memory (per CU, 64KB, programmable)
    ↓
Registers (per thread, 256 registers max)
    ↓
ALU (Vector / Scalar)
```

## 关键概念

| 概念 | 大小 | 延迟 | 说明 |
|------|------|------|------|
| **Global Memory** | 16GB (RX 9070 XT) | ~600ns | GPU 上主要内存，需 hipMemcpy |
| **Shared Memory** | 64KB/CU | ~4 cycles | 同 workgroup 共享，手动 `__shared__` |
| **Local Memory** | 自动溢出 | high | register 不够时 spill 到 GDDR |
| **L2 Cache** | 8MB | ~100 cycles | 所有 CU 共享 |
| **HBM** | 显存带宽 | 1000+ GB/s | high bandwidth memory |

## Wavefront vs Warp

- **NVIDIA Warp**: 32 threads 同步执行
- **AMD Wavefront**: **64 threads** 同步执行（重要区别！）
- RDNA 4: Wavefront size = 64

## 重点实验

```
02-memory-model/
├── coalesced_access/       # 内存合并访问优化
├── bank_conflicts/         # Shared memory bank conflict 分析
├── mem_bandwidth/          # 带宽测试 (global mem vs shared mem)
└── README.md
```
