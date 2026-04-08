/*
 * kfd_queue.c — KFD 队列管理演示
 *
 * 编译: gcc -o kfd_queue kfd_queue.c -lhsakmt -lpthread
 * 运行: sudo ./kfd_queue
 *
 * 本程序演示：
 * 1. 分配 AQL 环形缓冲区
 * 2. 创建 KFD 队列
 * 3. 查询/更新/销毁队列
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
#include <pthread.h>

#include <rocm_smi/kfd_ioctl.h>
#include <hsakmt/hsakmt.h>

#define HSAKMT_CHECK(call) do { \
    HSAKMT_STATUS _err = (call); \
    if (_err != HSAKMT_STATUS_SUCCESS) { \
        fprintf(stderr, "HSAKMT error %d at %s:%d\n", _err, __FILE__, __LINE__); \
        exit(1); \
    } \
} while (0)

/* Ring Buffer 大小（必须 4096 对齐）*/
#define RING_SIZE (4096 * 4)
#define DOORBELL_PAGE_SIZE (4096)

/* AQL 包格式（简化版，实际参考 amd_hsa_queue.h）*/
typedef struct {
    uint16_t header;          // AQL 包头
    uint16_t dst_ops;         // 目标操作
    uint64_t dst_address;     // 目标地址
    uint64_t src_address;     // 源地址
    uint64_t src_size;        // 传输大小
    uint32_t control;        // 控制字段
    uint32_t reserved;
} AQL_Packet;

int main(int argc, char *argv[]) {
    printf("=== KFD 队列管理演示 ===\n\n");

    HSAKMT_CHECK(hsaKmtOpenKFD());

    /* 获取第一个 GPU Agent */
    HsaSystemInfo sys_info;
    HSAKMT_CHECK(hsaKmtGetSystemInfo(&sys_info, sizeof(sys_info)));

    HsaAgent *agents = calloc(sys_info.AgentCount, sizeof(HsaAgent));
    HSAKMT_CHECK(hsaKmtEnumerateAgent(sys_info.AgentCount, agents));

    HsaAgent *gpu_agent = NULL;
    uint32_t gpu_id = 0;
    for (uint32_t i = 0; i < sys_info.AgentCount; i++) {
        if (agents[i].AgentId.FamilyId != 0 &&
            agents[i].AgentId.HsaVersion > HSA_VERSION_1_0) {
            gpu_agent = &agents[i];
            gpu_id = agents[i].LocationId & 0xFFFF;
            break;
        }
    }

    if (!gpu_agent) {
        fprintf(stderr, "No compute GPU found\n");
        return 1;
    }
    printf("[OK] 使用 GPU ID: 0x%x\n", gpu_id);

    /* =========================================
     * 步骤 1: 分配 Ring Buffer（Host 内存，可被 GPU 访问）
     * ========================================= */

    int kfd_fd = open("/dev/kfd", O_RDWR);
    if (kfd_fd < 0) {
        perror("open /dev/kfd");
        return 1;
    }

    /* Ring Buffer 必须页对齐，使用 posix_memalign */
    void *ring_buffer = NULL;
    if (posix_memalign(&ring_buffer, 4096, RING_SIZE) != 0) {
        perror("posix_memalign");
        return 1;
    }
    memset(ring_buffer, 0, RING_SIZE);
    printf("[OK] Ring Buffer 分配: %p, 大小: %d bytes\n",
           ring_buffer, RING_SIZE);

    /* 为 Ring Buffer 创建 KFD 内存分配 */
    struct kfd_ioctl_alloc_memory_args alloc_args = {
        .size =            RING_SIZE,
        .flags =           0x0,  // 默认 flags
        .type =            0,    // HSA memory type (未指定)
    };

    /* 通过 HSAKMT 分配 GPU 可访问内存 */
    HsaMemoryBuffer buf;
    buf.Size = RING_SIZE;
    buf.Alignment = 4096;
    HSAKMT_CHECK(hsaKmtAllocMemory(RING_SIZE, 4096,
                                    HSA_MEMORY_FLAGS_GPU_LOCALLOC,
                                    &buf));
    printf("[OK] HSAKMT 分配 GPU 内存: %llx, 大小: %zu\n",
           (unsigned long long)buf.MemoryAddress, buf.Size);

    /* 映射到 GPU VA */
    HSAKMT_CHECK(hsaKmtMapMemoryToGPU(&buf, NULL));
    printf("[OK] 映射到 GPU VA: %llx\n",
           (unsigned long long)buf.GpuVirtualAddress);

    /* =========================================
     * 步骤 2: 创建 Doorbell 页面
     * ========================================= */

    void *doorbell = mmap(NULL, DOORBELL_PAGE_SIZE,
                          PROT_READ | PROT_WRITE,
                          MAP_PRIVATE | MAP_ANONYMOUS,
                          -1, 0);
    if (doorbell == MAP_FAILED) {
        perror("mmap doorbell");
        return 1;
    }
    printf("[OK] Doorbell 页面: %p\n", doorbell);

    /* 通过 KFD IOCTL 分配 Doorbell */
    struct kfd_ioctl_create_queue_args queue_args = {
        .ring_base_address =   (uint64_t)buf.GpuVirtualAddress,
        .write_pointer_address = 0,   // KFD 填写
        .read_pointer_address =  0,   // KFD 填写
        .doorbell_offset =      (uint64_t)doorbell,  // KFD 填写
        .ring_size =            RING_SIZE,
        .gpu_id =               gpu_id,
        .queue_type =           KFD_IOC_QUEUE_TYPE_COMPUTE,
        .queue_percentage =     KFD_MAX_QUEUE_PERCENTAGE,
        .queue_priority =       KFD_MAX_QUEUE_PRIORITY,
        .queue_id =             0,     // KFD 填写
        .eop_buffer_address =   0,
        .eop_buffer_size =      0,
        .ctx_save_restore_address = 0,
        .ctx_save_restore_size = 0,
        .ctl_stack_size =       0,
    };

    int ret = ioctl(kfd_fd, KFD_IOC_CREATE_QUEUE, &queue_args);
    if (ret < 0) {
        perror("KFD_IOC_CREATE_QUEUE failed");
        return 1;
    }
    printf("[OK] 队列创建成功!\n");
    printf("     queue_id:          %u\n", queue_args.queue_id);
    printf("     doorbell_offset:   0x%llx\n",
           (unsigned long long)queue_args.doorbell_offset);
    printf("     write_pointer_va:  0x%llx\n",
           (unsigned long long)queue_args.write_pointer_address);
    printf("     read_pointer_va:   0x%llx\n",
           (unsigned long long)queue_args.read_pointer_address);

    /* =========================================
     * 步骤 3: 写一个 NOP AQL 包到 Ring Buffer
     * （实际工作负载会在这里）
     * ========================================= */

    AQL_Packet *pkt = (AQL_Packet *)ring_buffer;
    pkt->header = 0x8081;  // AQL NOP 包
    pkt->dst_ops = 0;
    printf("[OK] 写入 NOP AQL 包到 ring buffer\n");

    /* 刷新缓存，确保 GPU 看到数据 */
    __builtin___sync_synchronize();

    /* Ring doorbell - 通知 GPU 开始处理 */
    volatile uint64_t *db = (volatile uint64_t *)
        ((uint8_t *)queue_args.doorbell_offset);
    *db = 0x00000001;  // 写 1 到 doorbell，触发调度
    printf("[OK] Ring doorbell\n");

    /* =========================================
     * 步骤 4: 等待 GPU 完成（轮询 Read Pointer）
     * ========================================= */

    printf("     等待 GPU 完成...\n");
    volatile uint64_t *rp = (volatile uint64_t *)
        ((uint8_t *)queue_args.read_pointer_address);
    while (*rp == 0) {
        usleep(100);
    }
    printf("[OK] GPU 完成! Read Pointer = %lu\n", (unsigned long)*rp);

    /* =========================================
     * 步骤 5: 销毁队列
     * ========================================= */

    struct kfd_ioctl_destroy_queue_args dq_args = {
        .queue_id = queue_args.queue_id,
    };
    ret = ioctl(kfd_fd, KFD_IOC_DESTROY_QUEUE, &dq_args);
    if (ret < 0) {
        perror("KFD_IOC_DESTROY_QUEUE failed");
    } else {
        printf("[OK] 队列已销毁 (queue_id=%u)\n", queue_args.queue_id);
    }

    /* 清理资源 */
    hsaKmtUnmapMemoryToGPU(&buf);
    hsaKmtFreeMemory(&buf, buf.Size);
    free(ring_buffer);
    munmap(doorbell, DOORBELL_PAGE_SIZE);
    free(agents);
    close(kfd_fd);
    HSAKMT_CHECK(hsaKmtCloseKFD());

    printf("\n=== 完成 ===\n");
    return 0;
}
