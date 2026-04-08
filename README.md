# GPU Learning — ROCm / HIP 学习目录

## 目录结构

```
gpu-learning/
├── 01-hip-basics/         # HIP 基础：kernel 编写、编译、运行
├── 02-memory-model/       # 内存模型：Global/Shared/Local/Register
├── 03-parallel-patterns/  # 并行模式：Reduction、Scan、矩阵分块
├── 04-math-libraries/    # 数学库：ROCBLAS、MIOpen、RCCL
├── 05-multi-gpu/          # 多卡编程：peer access、RCCL
├── 06-profiling-debugging/# 性能分析：rocprof、rocgdb、NSYS
├── 07-openmp-offload/      # OpenMP Offload 替代路径
├── projects/              # 自己的项目
├── notes/                 # 学习笔记、架构笔记
└── references/            # 官方文档、PDF 保存位置
```

## 学习路径建议

```
第一阶段 → HIP 基础 (01)
           ↓
第二阶段 → 内存模型 (02) + 并行模式 (03)
           ↓
第三阶段 → 数学库 (04) + 多卡 (05)
           ↓
第四阶段 → Profiling (06) + OpenMP (07)
```

## 环境

- ROCm: 7.2.0
- HIP: 7.2.26015
- GPU: AMD Radeon RX 9070 XT × 2 (gfx1201 / RDNA 4)
- 编译器: hipcc (AMD clang 22.0.0)

## 常用命令

```bash
# 编译 HIP 程序
hipcc -o app main.hip.cu

# 查看 GPU 状态
rocm-smi

# 运行 rocprof
rocprof --stats ./app

# 查看 HIP 语法转换 (CUDA → HIP)
hipify-clang input.cu --output=output.hip.cu
```
