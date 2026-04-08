/*
 * kfd_memory.c — KFD 内存模型演示
 *
 * 编译: gcc -o kfd_memory kfd_memory.c -lhsakmt -lpthread
 * 运行: sudo ./kfd_memory
 *
 * 演示三种 GPU 内存分配方式：
 * 1. HSAKMT GPU Local (VRAM)
 * 2. HSAKMT Coarse Grain (系统内存, GPU 不可直接访问，需拷贝)
 * 3. SVM (Shared Virtual Memory) - 用户指针直接映射到 GPU VA
 */

#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <unistd.h>
#include <fcntl.h>
#include <errno.h>
#include <sys/ioctl.h>
#include <sys/mman.h>

#include <rocm_smi/kfd_ioctl.h>
#include <hsakmt/hsakmt.h>

#define HSAKMT_CHECK(call) do { \
    HSAKMT_STATUS _err = (call); \
    if (_err != HSAKMT_STATUS_SUCCESS) { \
        fprintf(stderr, "HSAKMT error %d at %s:%d: %s\n", \
                _err, __FILE__, __LINE__, #call); \
        exit(1); \
    } \
} while (0)

static void dump_buffer_info(const char *name, HsaMemoryBuffer *buf) {
    printf("  %-20s VA: 0x%016llx  Size: %zu bytes\n",
           name,
           (unsigned long long)buf->MemoryAddress,
           buf->Size);
}

int main(int argc, char *argv[]) {
    printf("=== KFD 内存模型演示 ===\n\n");

    HSAKMT_CHECK(hsaKmtOpenKFD());

    /* 获取 GPU Agent */
    HsaSystemInfo sys_info;
    HSAKMT_CHECK(hsaKmtGetSystemInfo(&sys_info, sizeof(sys_info)));
    HsaAgent *agents = calloc(sys_info.AgentCount, sizeof(HsaAgent));
    HSAKMT_CHECK(hsaKmtEnumerateAgent(sys_info.AgentCount, agents));

    HsaAgent *gpu_agent = NULL;
    for (uint32_t i = 0; i < sys_info.AgentCount; i++) {
        if (agents[i].AgentId.FamilyId != 0) {
            gpu_agent = &agents[i];
            break;
        }
    }
    if (!gpu_agent) { fprintf(stderr, "No GPU\n"); return 1; }

    const size_t SIZE = 4096 * 4;  // 16KB
    void *host_ptr;

    /* =============================================
     * 方法 1: GPU Local 分配 (VRAM)
     * ============================================= */
    printf("--- 1. GPU Local 分配 (VRAM) ---\n");
    HsaMemoryBuffer vram_buf;
    vram_buf.Size = SIZE;
    vram_buf.Alignment = 4096;
    HSAKMT_CHECK(hsaKmtAllocMemory(SIZE, 4096,
                                    HSA_MEMORY_FLAGS_GPU_LOCALLOC,
                                    &vram_buf));
    dump_buffer_info("VRAM buffer", &vram_buf);

    /* 映射到 GPU VA 空间 */
    HSAKMT_CHECK(hsaKmtMapMemoryToGPU(&vram_buf, NULL));
    printf("  GPU VA: 0x%016llx (mapped)\n",
           (unsigned long long)vram_buf.GpuVirtualAddress);

    /* 填充数据 */
    memset(vram_buf.MemoryAddress, 0xAB, SIZE);

    /* =============================================
     * 方法 2: Coarse Grain 系统内存
     * (不可直接被 GPU 访问，需要通过 GPU 拷贝)
     * ============================================= */
    printf("\n--- 2. Coarse Grain 系统内存 ---\n");
    HsaMemoryBuffer coarse_buf;
    coarse_buf.Size = SIZE;
    coarse_buf.Alignment = 4096;
    HSAKMT_CHECK(hsaKmtAllocMemory(SIZE, 4096,
                                    HSA_MEMORY_FLAGS_COARSE_GRAIN,
                                    &coarse_buf));
    dump_buffer_info("Coarse buffer", &coarse_buf);

    /* Coarse Grain 不需要 MapToGPU，直接用 CPU 填充 */
    memset(coarse_buf.MemoryAddress, 0xCD, SIZE);
    printf("  (Coarse grain: CPU 直接访问，GPU 需通过 PCIe DMA 拷贝)\n");

    /* =============================================
     * 方法 3: SVM (Shared Virtual Memory)
     * 用户空间 malloc 指针，直接映射到 GPU VA
     * ============================================= */
    printf("\n--- 3. SVM (用户指针直接映射到 GPU) ---\n");

    /* 分配对齐内存 (SVM 需要对齐) */
    if (posix_memalign(&host_ptr, 4096, SIZE) != 0) {
        perror("posix_memalign");
        return 1;
    }
    memset(host_ptr, 0xEF, SIZE);
    printf("  用户指针:     %p (CPU 虚拟地址)\n", host_ptr);

    HsaMemoryBuffer svm_buf;
    svm_buf.MemoryAddress = host_ptr;
    svm_buf.Size = SIZE;

    /* SVM 映射到 GPU */
    HSAKMT_CHECK(hsaKmtMapMemoryToGPU(&svm_buf, NULL));
    printf("  GPU VA:       0x%016llx (同一个指针，GPU/CPU 共享)\n",
           (unsigned long long)svm_buf.GpuVirtualAddress);

    /* =============================================
     * 查询进程显存可见性 ( apertures )
     * ============================================= */
    printf("\n--- 4. 进程显存可见性 ---\n");

    int kfd_fd = open("/dev/kfd", O_RDWR);
    if (kfd_fd >= 0) {
        struct kfd_ioctl_get_process_apertures_args args;
        if (ioctl(kfd_fd, KFD_IOC_GET_PROCESS_APERTURES, &args) == 0) {
            printf("  l0 GPU: 0x%016llx (size: %llu MB)\n",
                   (unsigned long long)args.process_vm_reserved_address,
                   (unsigned long long)args.process_vm_reserved_size / 1024 / 1024);
            printf("  l1 GPU: 0x%016llx (size: %llu MB)\n",
                   (unsigned long long)args.process_gpu_virtual_size,
                   (unsigned long long)args.process_gpu_virtual_size / 1024 / 1024);
        }
        close(kfd_fd);
    }

    /* =============================================
     * 清理
     * ============================================= */
    printf("\n--- 清理 ---\n");

    HSAKMT_CHECK(hsaKmtUnmapMemoryToGPU(&vram_buf));
    HSAKMT_CHECK(hsaKmtFreeMemory(&vram_buf, vram_buf.Size));
    printf("  VRAM buffer 释放\n");

    HSAKMT_CHECK(hsaKmtFreeMemory(&coarse_buf, coarse_buf.Size));
    printf("  Coarse buffer 释放\n");

    HSAKMT_CHECK(hsaKmtUnmapMemoryToGPU(&svm_buf));
    free(host_ptr);
    printf("  SVM host pointer 释放\n");

    free(agents);
    HSAKMT_CHECK(hsaKmtCloseKFD());

    printf("\n=== 完成 ===\n");
    return 0;
}
