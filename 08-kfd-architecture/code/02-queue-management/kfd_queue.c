/*
 * kfd_queue.c — KFD 队列管理演示
 *
 * 编译: cd code && make 02-queue-management/kfd_queue
 * 运行: sudo ./02-queue-management/kfd_queue
 *
 * 演示：
 * 1. 通过 HSAKMT 创建/查询/销毁队列
 * 2. 分配 GPU 本地内存作为 Ring Buffer
 * 3. 内存映射到 GPU VA
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
#include <sys/stat.h>
#include <sys/types.h>

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

/* Ring Buffer 大小（必须页对齐）*/
#define RING_SIZE (4096 * 4)

int main(int argc, char *argv[]) {
    printf("=== KFD 队列管理演示 ===\n\n");

    /* 打开 KFD */
    HSAKMT_CHECK(hsaKmtOpenKFD());

    /* 获取 GPU 节点信息 */
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
            printf("[OK] GPU 节点 #%u  (SIMD: %u, WaveFrontSize: %u)\n",
                   n, node_prop.NumFComputeCores, node_prop.WaveFrontSize);
            break;
        }
    }

    /* =========================================
     * 步骤 1: 分配 Ring Buffer 内存
     * ========================================= */
    printf("\n--- 步骤 1: 分配 Ring Buffer ---\n");

    void *ring_ptr = NULL;
    HSAKMT_CHECK(hsaKmtAllocMemory(
        gpu_node,
        RING_SIZE,
        (HsaMemFlags){0},
        &ring_ptr
    ));
    printf("[OK] hsaKmtAllocMemory: host VA = %p\n", ring_ptr);

    /* 映射到 GPU VA */
    uint64_t gpu_ring_va = 0;
    HSAKMT_STATUS map_err = hsaKmtMapMemoryToGPU(ring_ptr, RING_SIZE, &gpu_ring_va);
    if (map_err == HSAKMT_STATUS_SUCCESS) {
        printf("[OK] hsaKmtMapMemoryToGPU: GPU VA = 0x%llx\n",
               (unsigned long long)gpu_ring_va);
    } else {
        printf("[WARN] hsaKmtMapMemoryToGPU: 0x%x (GPU VA 可能需要从 QueueResource 获取)\n",
               map_err);
    }

    /* =========================================
     * 步骤 2: 创建队列
     * ========================================= */
    printf("\n--- 步骤 2: 创建队列 ---\n");

    HsaQueueResource qres;
    memset(&qres, 0, sizeof(qres));

    HSAKMT_CHECK(hsaKmtCreateQueue(
        gpu_node,                      // NodeId
        HSA_QUEUE_COMPUTE,             // Type (1 = PM4 Compute Queue)
        100,                           // QueuePercentage
        HSA_QUEUE_PRIORITY_NORMAL,     // Priority
        ring_ptr,                       // QueueAddress (host VA)
        RING_SIZE,                     // QueueSizeInBytes
        NULL,                          // Event (同步创建)
        &qres                          // OUT
    ));

    printf("[OK] 队列创建成功!\n");
    printf("     QueueId:        %lu\n", (unsigned long)qres.QueueId);
    printf("     GPU Ring VA:    0x%llx\n", (unsigned long long)gpu_ring_va);
    printf("     Doorbell VA:    0x%llx\n",
           (unsigned long long)qres.QueueDoorBell);
    printf("     WritePtr VA:    0x%llx\n",
           (unsigned long long)qres.QueueWptrValue);
    printf("     ReadPtr VA:     0x%llx\n",
           (unsigned long long)qres.QueueRptrValue);

    /* =========================================
     * 步骤 3: 写 AQL NOP 包到 Ring Buffer
     * ========================================= */
    printf("\n--- 步骤 3: 写 AQL NOP 包 ---\n");

    /*
     * AQL (AMD Queue Language) 环形缓冲区
     * 最简单的包: NOP (什么都不做，用于同步)
     * header = 0x8081 表示 NOP
     */
    volatile uint64_t *pkt = (volatile uint64_t *)ring_ptr;
    pkt[0] = 0x8081000000000000ULL;   // AQL NOP packet

    __sync_synchronize();
    printf("[OK] 写入 AQL NOP: pkt[0] = 0x%016llx\n",
           (unsigned long long)pkt[0]);
    printf("     (实际 AQL 包格式参见 amd_hsa_queue.h)\n");

    /* =========================================
     * 步骤 4: Ring Doorbell 触发 GPU
     * ========================================= */
    printf("\n--- 步骤 4: Ring Doorbell ---\n");

    volatile uint64_t *doorbell = (volatile uint64_t *)qres.QueueDoorBell;
    *doorbell = 0x1ULL;   // 写 1 触发 GPU 调度器
    printf("[OK] 门铃触发: *0x%llx = 1\n",
           (unsigned long long)qres.QueueDoorBell);
    printf("     GPU 调度器收到中断，开始处理 Ring 中的 AQL 包\n");

    /* =========================================
     * 步骤 5: 查询队列状态
     * ========================================= */
    printf("\n--- 步骤 5: 查询队列状态 ---\n");

    HsaQueueInfo qinfo;
    memset(&qinfo, 0, sizeof(qinfo));

    /* 轮询 Read Pointer（演示用）
     * QueueRptrValue 是 GPU VA，映射到 host 后可直接读取
     * 注意：如果 host 无法直接访问该 VA，显示 0x0 是正常的
     */
    volatile uint64_t *rptr = (volatile uint64_t *)qres.QueueRptrValue;
    uint64_t rptr_val = *rptr;
    printf("     ReadPointer(host view): 0x%llx\n",
           (unsigned long long)rptr_val);

    HSAKMT_STATUS info_err = hsaKmtGetQueueInfo(qres.QueueId, &qinfo);
    if (info_err == HSAKMT_STATUS_SUCCESS) {
        printf("     队列错误状态: 0x%x\n", qinfo.QueueDetailError);
        printf("     队列类型扩展: 0x%x\n", qinfo.QueueTypeExtended);
    } else {
        printf("     [WARN] hsaKmtGetQueueInfo: 0x%x (GPU 仍在执行中，正常)\n",
               info_err);
    }

    /* =========================================
     * 步骤 6: 销毁队列
     * ========================================= */
    printf("\n--- 步骤 6: 销毁队列 ---\n");

    HSAKMT_CHECK(hsaKmtDestroyQueue(qres.QueueId));
    printf("[OK] 队列已销毁 (QueueId=%lu)\n",
           (unsigned long)qres.QueueId);

    /* =========================================
     * 清理
     * ========================================= */
    printf("\n--- 清理 ---\n");

    HSAKMT_CHECK(hsaKmtUnmapMemoryToGPU(ring_ptr));
    printf("[OK] 解除 GPU 映射\n");

    HSAKMT_CHECK(hsaKmtFreeMemory(ring_ptr, RING_SIZE));
    printf("[OK] 释放 Ring Buffer\n");

    HSAKMT_CHECK(hsaKmtReleaseSystemProperties());
    HSAKMT_CHECK(hsaKmtCloseKFD());

    printf("\n=== 完成 ===\n");
    return 0;
}
