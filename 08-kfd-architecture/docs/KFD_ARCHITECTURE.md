# KFD 架构详解

## 1. 什么是 KFD

KFD（Kernel Fusion Driver）是 AMD GPU Linux 内核驱动的一部分，专门负责 GPU **计算资源**的管理和调度。它是 AMD ROCm 计算栈的最底层内核组件。

**两层含义：**
- **Kernel** — 运行在 Linux 内核空间
- **Fusion Driver** — "Fusion" 指的是 AMD APU/Fusion 架构（CPU+GPU 融合），KFD 最初为 APU 设计，后来扩展到独立 GPU

---

## 2. KFD 与 amdgpu 的关系

```
amdgpu.ko（Linux DRM 驱动）
    │
    ├── 图形子模块（DRM）     → /dev/dri/card*  → Mesa/ Vulkan/ OpenGL
    │   - 显存管理（VRAM allocations via GEM）
    │   - 显示输出（CRTC/Encoder）
    │   - GEM objects / TTM 内存管理
    │
    └── 计算子模块（KFD）     → /dev/kfd        → ROCm / HSA Runtime
        - 调度器（GPU scheduler）
        - 队列管理（AQL queues）
        - 计算内存分配（SVM / VRAM / GTT）
        - 中断处理（GPU → CPU 信号）
        - GPU 虚拟内存管理
```

两者共享同一硬件（GPU），通过 DRM 机制共享显存资源。

---

## 3. 从 HIP 到 KFD 的调用链

```
HIP API (hipMalloc / hipLaunchKernel)
         ↓
ROCm Runtime (amdhip64.so)
         ↓
HSA Runtime (hsa-runtime.so)
    - AQL 队列操作
    - 信号量管理
    - 指令预取/分发
         ↓
HSAKMT Thunk (libhsakmt.so)
    - ioctl(fd, KFD_IOC_*, ...)
    - mmap(/dev/kfd, ...)
         ↓
KFD (amdgpu.ko 内)
    - 调度队列到硬件 Command Processor
    - 分配/映射 GPU 内存
    - 中断处理（完成信号）
```

---

## 4. 队列 (Queue) 架构

### AQL 队列

KFD 使用 AMD Queue Language (AQL) 描述工作。每个队列：
- 是一个环形缓冲区（Ring Buffer），CPU 写命令，GPU 读命令
- 有独立的 Doorbell Page（GPU 通知 CPU 的机制）
- 支持优先级（0-15，15 最高）

### 队列创建流程

```
用户态                    内核 KFD
┌─────────────────┐
│ hsaKmtCreateQueue│
└────────┬────────┘
         │ ioctl(KFD_IOC_CREATE_QUEUE)
         │ ring_base, gpu_id, queue_type, priority
         ▼
┌─────────────────────────────────┐
│ KFD:                               │
│  1. 分配 queue_id                   │
│  2. 分配 doorbell page (GPU VA)     │
│  3. 分配 AQL ring buffer            │
│  4. 注册到 GPU scheduler            │
└────────┬───────────────────────────┘
         │ 返回 queue_id, doorbell_offset
         ▼
  用户拿到 doorbell_offset，写到这里
  即可触发 GPU 执行
```

### Doorbell 机制

Doorbell 类似于 PCIe 门铃：
1. CPU 写命令到 Ring Buffer
2. CPU 写 Doorbell（写入一个特定偏移的内存页）
3. GPU 收到 Doorbell 中断，开始处理

---

## 5. 内存模型

### 三种 GPU 内存类型

| 类型 | 位置 | 访问速度 | 管理方式 |
|------|------|---------|---------|
| **VRAM** | GPU 显存 | 最快 (~1000 GB/s) | KFD 显式分配 (`KFD_IOC_ALLOC_MEM`) |
| **GTT** | 系统内存（通过 PCIe GART） | 中等 (~20 GB/s) | KFD 分配，系统页表映射 |
| **SVM** | 用户空间虚拟地址 | 由 KFD 动态映射 | `KFD_IOC_MAP_MEMORY` 动态绑定 |

### SVM (Shared Virtual Memory)

SVM 是 KFD 最重要的内存特性之一 —— 允许 GPU 和 CPU 共享同一个虚拟地址空间：

```
用户空间指针 p = malloc(N);
                   │
                   │ hsaKmtMapMemoryToGPU(p)
                   ▼
KFD: 为 p 的页分配 GPU 页表条目（虚拟地址 → 系统物理页）
     GPU 现在可以直接用 p 访问同一块内存，无需显式拷贝
```

### 内存分配 IOCTL

```
KFD_IOC_ALLOC_MEM_ARGS:
  - size:          分配大小
  - type:          VRAM / GTT / SVM
  - page_size:     4KB / 64KB (large page)
  - flags:         CONTIGUOUS / COARSE_GRAIN / FINE_GRAIN
  - gpu_id:        分配到哪个 GPU
  → 返回 gpu_address (GPU 虚拟地址)
```

---

## 6. 中断与事件机制

GPU 完成工作后，通过中断通知 CPU：

```
GPU Hardware
    │
    │ 调度器中断 (DCE - Dispatch Controller)
    ▼
KFD (中断处理例程)
    │
    │ 解析中断源（哪个 queue 完成 / 哪个 signal）
    ▼
用户态事件 (HSA Signal)
    │
    │ hsaKmtWaitForEvent(ev)
    ▼
CPU 线程唤醒
```

---

## 7. 多 GPU (Crossfire/XGMI)

KFD 支持多 GPU 场景：
- **Same-node multi-GPU**：通过 PCIe/XGMI 互联，内存统一寻址
- **KFD_IOC_QUEUE_TYPE_SDMA_XGMI**：跨 GPU 数据搬运
- **RCCL**：基于 HSA 通信库的集体操作

---

## 8. 关键 IOCTL 一览

```c
// 查询 KFD 版本
KFD_IOC_GET_VERSION        →  { major_version, minor_version }

// 队列管理
KFD_IOC_CREATE_QUEUE        →  { ring_base, doorbell_offset, queue_id }
KFD_IOC_DESTROY_QUEUE       →  { queue_id }
KFD_IOC_UPDATE_QUEUE        →  { queue_id, ring_base, priority }
KFD_IOC_SET_CU_MASK         →  { queue_id, cu_mask_ptr }
KFD_IOC_GET_QUEUE_WAVE_MAPPING  // 查询 queue 占用的 CU

// 内存管理
KFD_IOC_ALLOC_MEM           →  { size, type, flags, gpu_address }
KFD_IOC_FREE_MEM            →  { gpu_address }
KFD_IOC_MAP_MEMORY          →  { cpu_address, gpu_address, size }
KFD_IOC_UNMAP_MEMORY        →  { cpu_address, gpu_address, size }
KFD_IOC_GET_PROCESS_APERTURES // 获取进程可见显存区域

// 事件
KFD_IOC_CREATE_EVENT         →  { event_id, event_slot_index }
KFD_IOC_DESTROY_EVENT       →  { event_id }
KFD_IOC_SET_EVENT           →  { event_id }
KFD_IOC_RESET_EVENT         →  { event_id }
KFD_IOC_WAIT_EVENTS         →  { wait_count, event_list[] }

// SMI (System Management)
KFD_IOC_GET_SMI_CONFIG      →  { ... }
KFD_IOC_SET_SMI_CONFIG      →  { ... }
```

---

## 9. 查看 KFD 状态

```bash
# 查看 GPU KFD 信息
cat /sys/class/kfd/kfd/topology/   # GPU 拓扑
cat /sys/class/kfd/kfd/properties/ # KFD 属性

# 查看 KFD 设备
ls -la /dev/kfd

# rocminfo 显示 KFD 发现的 GPU
rocminfo

# amdgpu 驱动信息
dmesg | grep -i amdgpu | tail -20
```
