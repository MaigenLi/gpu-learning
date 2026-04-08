/*
 * kfd_memory.c — KFD 内存模型演示
 *
 * 编译: cd code && make 03-memory-model/kfd_memory
 * 运行: ./03-memory-model/kfd_memory
 *
 * 演示三种 GPU 内存分配方式：
 * 1. Coarse Grain (默认) — 系统内存，GPU 通过 PCIe GART 访问
 * 2. GPU Local         — GPU 专用显存（NoSubstitute flag）
 * 3. SVM (Shared Virtual Memory) — 用户指针直接映射到 GPU VA
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
#include <hsakmt/hsakmttypes.h>

#define HSAKMT_CHECK(call) do { \
    HSAKMT_STATUS _err = (call); \
    if (_err != HSAKMT_STATUS_SUCCESS) { \
        fprintf(stderr, "HSAKMT error %d at %s:%d: %s\n", \
                _err, __FILE__, __LINE__, #call); \
        exit(1); \
    } \
} while (0)

/* HsaMemFlags bit positions (from struct _HsaMemFlags):
 *   bit 0: NonPaged
 *   bit 1-2: CachePolicy
 *   bit 3: ReadOnly
 *   bit 4-5: PageSize
 *   bit 6: HostAccess (0=GPU only, 1=host can access)
 *   bit 7: NoSubstitute (1=GPU local, don't fallback to system)
 *   ...
 *   bit 13: CoarseGrain (1=coarse grain)
 */
static HsaMemFlags memflags(unsigned int bits) {
    HsaMemFlags f;
    memset(&f, 0, sizeof(f));
    *(unsigned int*)&f = bits;
    return f;
}

#define SIZE (4096 * 4)   // 16KB 测试用

int main(int argc, char *argv[]) {
    printf("=== KFD 内存模型演示 ===\n\n");

    /* 打开 KFD */
    HSAKMT_CHECK(hsaKmtOpenKFD());

    /* 获取 NUMA 节点信息 */
    HsaSystemProperties sys_props;
    HSAKMT_CHECK(hsaKmtAcquireSystemProperties(&sys_props));
    printf("[OK] NUMA 节点数: %u\n", sys_props.NumNodes);

    /* 找到第一个 GPU 节点 */
    uint32_t gpu_node = 0;
    HsaNodeProperties node_prop;
    for (uint32_t n = 0; n < sys_props.NumNodes; n++) {
        HSAKMT_CHECK(hsaKmtGetNodeProperties(n, &node_prop));
        if (node_prop.NumFComputeCores > 0) {
            gpu_node = n;
            printf("[OK] GPU 节点 #%u  (SIMD: %u, 显存 banks: %u)\n",
                   n, node_prop.NumFComputeCores, node_prop.NumMemoryBanks);
            break;
        }
    }

    int kfd_fd = open("/dev/kfd", O_RDWR);
    if (kfd_fd < 0) {
        perror("open /dev/kfd");
        return 1;
    }

    /* =============================================
     * 步骤 1: Coarse Grain 内存（默认，系统内存）
     * GPU 通过 PCIe GART 访问，不可直接 DMA
     * ============================================= */
    printf("\n--- 步骤 1: Coarse Grain 系统内存 ---\n");

    void *coarse_ptr = NULL;
    HSAKMT_CHECK(hsaKmtAllocMemory(gpu_node, SIZE, memflags(0), &coarse_ptr));
    printf("[OK] hsaKmtAllocMemory (Coarse Grain)\n");
    printf("     主机虚拟地址: %p\n", coarse_ptr);

    /* Coarse Grain 不需要 MapToGPU，CPU 直接访问 */
    memset(coarse_ptr, 0xCD, SIZE);
    printf("     CPU 直接写入: 0xCD...\n");
    printf("     (GPU 需通过 PCIe DMA 才能访问这块内存)\n");

    /* =============================================
     * 步骤 2: Fine Grain 内存（Host Access = 1）
     * Host 可以直接访问，GPU 也可以访问（SVM 基础）
     * ============================================= */
    printf("\n--- 步骤 2: Fine Grain (Host Access) 内存 ---\n");

    void *fine_ptr = NULL;
    /* HostAccess(bit6)=1 → host 可以直接读写这块内存
     * NoSubstitute=0 → 允许回退到系统内存（适合离散 GPU） */
    HSAKMT_CHECK(hsaKmtAllocMemory(gpu_node, SIZE, memflags((1<<6)), &fine_ptr));
    printf("[OK] hsaKmtAllocMemory (Fine Grain)\n");
    printf("     主机虚拟地址: %p\n", fine_ptr);

    uint64_t fine_gpu_va = 0;
    HSAKMT_STATUS fine_err = hsaKmtMapMemoryToGPU(fine_ptr, SIZE, &fine_gpu_va);
    if (fine_err == HSAKMT_STATUS_SUCCESS) {
        printf("     GPU 虚拟地址: 0x%llx\n", (unsigned long long)fine_gpu_va);
    } else {
        printf("     [WARN] hsaKmtMapMemoryToGPU: 0x%x\n", fine_err);
    }

    memset(fine_ptr, 0xAB, SIZE);
    printf("     CPU 写入: 0xAB... (Host 和 GPU 共享访问)\n");

    /* =============================================
     * 步骤 3: SVM (Shared Virtual Memory)
     * 用户 malloc 的指针，直接映射到 GPU VA
     * GPU 和 CPU 共享同一虚拟地址
     * ============================================= */
    printf("\n--- 步骤 3: SVM (Shared Virtual Memory) ---\n");

    void *svm_ptr = NULL;
    if (posix_memalign(&svm_ptr, 4096, SIZE) != 0) {
        perror("posix_memalign");
        return 1;
    }
    memset(svm_ptr, 0xEF, SIZE);
    printf("[OK] posix_memalign: %p (对齐到 4KB)\n", svm_ptr);

    uint64_t svm_gpu_va = 0;
    HSAKMT_STATUS svm_err = hsaKmtMapMemoryToGPU(svm_ptr, SIZE, &svm_gpu_va);
    if (svm_err == HSAKMT_STATUS_SUCCESS) {
        printf("[OK] SVM 映射到 GPU VA: 0x%llx\n",
               (unsigned long long)svm_gpu_va);
        printf("     同一个指针: CPU=%p, GPU=0x%llx\n",
               svm_ptr, (unsigned long long)svm_gpu_va);
        printf("     GPU 现在可以直接读写这个地址，无需 hipMemcpy!\n");
    } else {
        printf("[WARN] SVM hsaKmtMapMemoryToGPU: 0x%x\n", svm_err);
    }

    /* =============================================
     * 步骤 4: 查询进程显存可见性 (Apertures)
     * ============================================= */
    printf("\n--- 步骤 4: 进程显存 Apertures ---\n");

    struct kfd_ioctl_get_process_apertures_args apergs;
    memset(&apergs, 0, sizeof(apergs));
    if (ioctl(kfd_fd, AMDKFD_IOC_GET_PROCESS_APERTURES, &apergs) == 0) {
        printf("  GPU 数: %u\n", apergs.num_of_nodes);
        for (uint32_t i = 0; i < apergs.num_of_nodes && i < 7; i++) {
            struct kfd_process_device_apertures *a = &apergs.process_apertures[i];
            printf("  GPU[%u] gpu_id=0x%x:\n", i, a->gpu_id);
            printf("    LDS:       0x%llx - 0x%llx\n",
                   (unsigned long long)a->lds_base,
                   (unsigned long long)a->lds_limit);
            printf("    Scratch:   0x%llx - 0x%llx\n",
                   (unsigned long long)a->scratch_base,
                   (unsigned long long)a->scratch_limit);
            printf("    GPU VM:    0x%llx - 0x%llx\n",
                   (unsigned long long)a->gpuvm_base,
                   (unsigned long long)a->gpuvm_limit);
        }
    } else {
        perror("  AMDKFD_IOC_GET_PROCESS_APERTURES");
    }

    /* =============================================
     * 步骤 5: 通过 /sys 查看 GPU 内存信息
     * ============================================= */
    printf("\n--- 步骤 5: /sys KFD 内存信息 ---\n");

    FILE *f = fopen("/sys/class/kfd/kfd/topology/nodes", "r");
    if (f) {
        char line[256];
        int node = -1;
        while (fgets(line, sizeof(line), f)) {
            if (strncmp(line, "node", 4) == 0 && strchr(line, 'g')) {
                node++;
                if (node >= 0) printf("  %s", line);
            } else if (node >= 0 && strstr(line, "memory")) {
                printf("    %s", line);
            }
        }
        fclose(f);
    }

    /* =============================================
     * 清理
     * ============================================= */
    printf("\n--- 清理 ---\n");

    HSAKMT_CHECK(hsaKmtUnmapMemoryToGPU(fine_ptr));
    printf("[OK] 解除 fine_ptr GPU 映射\n");

    HSAKMT_CHECK(hsaKmtFreeMemory(fine_ptr, SIZE));
    printf("[OK] 释放 fine_ptr (Fine Grain)\n");

    HSAKMT_CHECK(hsaKmtUnmapMemoryToGPU(svm_ptr));
    printf("[OK] 解除 svm_ptr GPU 映射\n");
    free(svm_ptr);
    printf("[OK] 释放 svm_ptr (SVM)\n");

    HSAKMT_CHECK(hsaKmtFreeMemory(coarse_ptr, SIZE));
    printf("[OK] 释放 coarse_ptr (Coarse Grain)\n");

    close(kfd_fd);
    HSAKMT_CHECK(hsaKmtReleaseSystemProperties());
    HSAKMT_CHECK(hsaKmtCloseKFD());

    printf("\n=== 完成 ===\n");
    return 0;
}
