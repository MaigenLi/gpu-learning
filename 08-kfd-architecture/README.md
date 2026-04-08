# 08 — KFD 架构 (Kernel Fusion Driver)

## KFD 在整个 AMD GPU 软件栈中的位置

```
应用层 (HIP / OpenMP / OpenCL)
         ↓
  ROCm Runtime (amdhip64)
         ↓
  HSA Runtime (hsa-runtime)
         ↓
  HSAKMT Thunk (libhsakmt.so)          ← 用户态 / 内核态通信抽象层
         ↓  ioctl / mmap
  KFD (Kernel Fusion Driver)            ← Linux 内核模块 (amdgpu.ko 的一部分)
         ↓
  GPU Hardware (GC / SDMA / PME)       ← 硬件
```

**一句话概括**：KFD 是 Linux 内核驱动，提供 GPU 底层资源管理（队列、内存、调度中断）。HSAKMT 是它的用户态封装，ROCm Runtime 依赖 HSAKMT 来管理 GPU。

---

## 核心概念

### 1. KFD vs amdgpu

| | KFD | amdgpu |
|--|-----|--------|
| 层级 | 内核驱动（GPU 资源管理层） | 图形驱动（显存管理、显示）|
| 职责 | GPU 调度、队列、计算内存分配 | 显存管理、显示、上下文 |
| 接口 | IOCTL（通过 `/dev/kfd`） | DRM IOCTL（通过 `/dev/dri/card*`）|
| 用途 | 计算（HPC/ML） | 图形 + 计算 |

两者共享 GPU 硬件，通过 DRM 机制协同。

### 2. 用户态通信路径

```
用户空间                    内核空间
┌──────────────────┐       ┌──────────────────────────┐
│   HSAKMT API     │       │                          │
│  (hsakmt.h)      │ ioctl │   KFD (amdgpu.ko)        │
│                  │ ────→ │   - 队列管理              │
│  hsaKmtOpenKFD() │       │   - 内存分配 (VRAM/GTT)   │
│  hsaKmtCreateQueue│      │   - GPU 中断处理           │
│  hsaKmtAllocMem()│       │   - SVM (Shared Virtual   │
└──────────────────┘       │     Memory)               │
                           └──────────────────────────┘
```

关键设备节点：`/dev/kfd`

### 3. 队列 (Queue)

- KFD 通过 **AQL (AMD Queue Language)** 队列工作
- 支持队列类型：`COMPUTE`（默认）、`SDMA`（数据搬运）、`SDMA_XGMI`（多卡）
- 每个队列有独立 **Doorbell**（GPU → CPU 通知机制）
- 队列通过 IOCTL 创建/销毁/更新

### 4. 内存模型

```
GPU Virtual Address Space
├── SVM (Shared Virtual Memory)     ← 用户空间 ptr，KFD 自动做 VA→PA 映射
├── GPU VRAM (显存)                ← KFD 分配的 GPU 本地内存
└── GTT (Global Transfer Table)    ← 可被 GPU 访问的系统内存页
```

### 5. 中断与事件

- GPU 中断 → KFD → 用户态事件信号（HSA Signal）
- 通过 `hsaKmtCreateEvent` 创建事件，绑定到特定 GPU 中断源

---

## IOCTL 体系（关键接口）

所有 IOCTL 通过 `/dev/kfd` 触发，查看 `kfd_ioctl.h`：

| IOCTL | 用途 |
|-------|------|
| `KFD_IOC_GET_VERSION` | 查询 KFD 版本 |
| `KFD_IOC_CREATE_QUEUE` | 创建计算队列 |
| `KFD_IOC_DESTROY_QUEUE` | 销毁队列 |
| `KFD_IOC_SET_CU_MASK` | 设置 CU 掩码（控制用到哪些 CU）|
| `KFD_IOC_ALLOC_MEM` | 分配 GPU 内存（VRAM/GTT/SVM）|
| `KFD_IOC_FREE_MEM` | 释放内存 |
| `KFD_IOC_MAP_MEMORY` | 将内存映射到 GPU VA |
| `KFD_IOC_CREATE_EVENT` | 创建事件信号 |
| `KFD_IOC_SET_EVENT` / `KFD_IOC_RESET_EVENT` | 触发/重置事件 |
| `KFD_IOC_GET_PROCESS_APERTURES` | 获取进程显存可见性 |

---

## 重要文件

```
/dev/kfd                      # KFD 主设备节点
/procfs/kfd/                   # KFD 统计信息（部分内核）
/sys/class/kfd/                # KFD sysfs 接口

# ROCm 头文件
/opt/rocm/include/hsakmt/      # HSAKMT API (用户空间调用 KFD)
/opt/rocm/include/rocm_smi/kfd_ioctl.h   # KFD IOCTL 定义
/opt/rocm/include/hsa/hsa.h    # HSA Runtime API
/opt/rocm/include/hsa/amd_hsa_common.h   # AMD HSA 通用结构
```

---

## 代码目录

```
08-kfd-architecture/
├── docs/
│   └── KFD_ARCHITECTURE.md    # 本文档（详细架构图解）
├── code/
│   ├── 01-kfd-basics/        # KFD 打开/版本查询/基本结构
│   ├── 02-queue-management/  # 队列创建、销毁、更新
│   ├── 03-memory-model/      # 内存分配、SVM 映射
│   └── 04-events-interrupts/ # 事件机制、信号
└── README.md
```

---

## 学习路径

```
第一阶段：KFD 基础 → 理解 /dev/kfd 打开、版本协商
第二阶段：队列管理 → AQL 队列、Doorbell、队列优先级
第三阶段：内存模型 → VRAM/GTT/SVM 区别，内存分配 IOCTL
第四阶段：事件机制 → GPU 中断如何传到用户空间
```

---

## 参考资料

- KFD IOCTL 完整定义：`/opt/rocm/include/rocm_smi/kfd_ioctl.h`
- HSAKMT API：`/opt/rocm/include/hsakmt/hsakmt.h`
- AMD GPU 架构文档：https://gpuopen.com/amd-gcn3-architecture/（参考 ISA 文档）
- ROCm 官方文档：https://docs.amd.com/
- HSA 规范：https://hsafoundation.com/（HSA System Specification PDF）
