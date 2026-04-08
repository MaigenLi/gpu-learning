/*
 * kfd_version.c — KFD 基础：打开设备 + 查询版本 + 枚举 GPU 节点
 *
 * 编译: cd code && make 01-kfd-basics/kfd_version
 * 运行: sudo ./01-kfd-basics/kfd_version
 *
 * 演示：
 * 1. 打开 /dev/kfd 设备
 * 2. 查询 KFD IOCTL 版本
 * 3. 通过 HSAKMT API 枚举 GPU 节点和属性
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

/* HSA 类型简写 */
typedef HSAuint32 u32;
typedef HSAuint64 u64;

/* KFD IOCTL 版本 */
static void kfd_ioctl_version(int fd) {
    printf("--- 直接 IOCTL ---\n");

    struct kfd_ioctl_get_version_args args;
    int ret = ioctl(fd, AMDKFD_IOC_GET_VERSION, &args);
    if (ret < 0) {
        perror("AMDKFD_IOC_GET_VERSION failed");
        return;
    }
    printf("  KFD IOCTL 版本: %u.%u\n",
           args.major_version, args.minor_version);
}

/* 通过 HSAKMT 枚举节点 */
static void enumerate_nodes(void) {
    printf("\n--- HSAKMT 节点枚举 ---\n");

    HsaVersionInfo ver;
    HSAKMT_CHECK(hsaKmtGetVersion(&ver));
    printf("  HSAKMT 版本: %u.%u\n",
           ver.KernelInterfaceMajorVersion,
           ver.KernelInterfaceMinorVersion);

    HsaSystemProperties sys_props;
    HSAKMT_CHECK(hsaKmtAcquireSystemProperties(&sys_props));
    printf("  NUMA 节点数: %u\n", sys_props.NumNodes);

    for (u32 node_id = 0; node_id < sys_props.NumNodes; node_id++) {
        HsaNodeProperties prop;
        HSAKMT_CHECK(hsaKmtGetNodeProperties(node_id, &prop));

        if (prop.NumFComputeCores == 0) {
            continue;  // 跳过 CPU-only 节点
        }

        printf("\n  === GPU 节点 #%u ===\n", node_id);
        printf("    NumFComputeCores (SIMD数): %u\n", prop.NumFComputeCores);
        printf("    WaveFrontSize:             %u\n", prop.WaveFrontSize);
        printf("    MaxWavesPerSIMD:           %u\n", prop.MaxWavesPerSIMD);
        printf("    NumMemoryBanks:             %u\n", prop.NumMemoryBanks);
        printf("    NumCaches:                  %u\n", prop.NumCaches);
        printf("    NumIOLinks:                 %u\n", prop.NumIOLinks);
        printf("    LDS Size:                  %u KB/SIMD\n", prop.LDSSizeInKB);
        printf("    GDS Size:                  %u KB\n", prop.GDSSizeInKB);
        printf("    CComputeIdLo:              0x%x\n", prop.CComputeIdLo);
        printf("    FComputeIdLo:              0x%x\n", prop.FComputeIdLo);

        /* 引擎版本 */
        printf("    GFX Engine:  %u.%u.%u (uCode: %u)\n",
               prop.EngineId.ui32.Major,
               prop.EngineId.ui32.Minor,
               prop.EngineId.ui32.Stepping,
               prop.EngineId.ui32.uCode);

        /* 显存信息 */
        printf("    显存 banks: %u\n", prop.NumMemoryBanks);

        for (u32 b = 0; b < prop.NumMemoryBanks && b < 4; b++) {
            HsaMemoryProperties mem_prop;
            HSAKMT_CHECK(hsaKmtGetNodeMemoryProperties(node_id, b, &mem_prop));
            printf("    Bank[%u]: Size: %llu MB  Width: %u bits\n",
                   b,
                   (unsigned long long)mem_prop.SizeInBytes / 1024 / 1024,
                   mem_prop.Width);
        }
    }

    HSAKMT_CHECK(hsaKmtReleaseSystemProperties());
}

/* 通过 /sys 查看拓扑 */
static void show_sysfs_topology(void) {
    printf("\n--- /sys/class/kfd/topology ---\n");

    FILE *f = fopen("/sys/class/kfd/kfd/topology/nodes", "r");
    if (!f) {
        printf("  (无法读取 /sys/class/kfd/kfd/topology/nodes)\n");
        return;
    }

    char line[256];
    while (fgets(line, sizeof(line), f)) {
        /* 找节点 */
        if (strncmp(line, "node", 4) == 0) {
            printf("  %s", line);
        }
    }
    fclose(f);

    /* GPU 属性 */
    FILE *p = fopen("/sys/class/kfd/kfd/properties", "r");
    if (p) {
        while (fgets(line, sizeof(line), p)) {
            printf("  %s", line);
        }
        fclose(p);
    }
}

int main(int argc, char *argv[]) {
    printf("=== KFD 基础演示 ===\n\n");

    /* 打开 KFD */
    HSAKMT_CHECK(hsaKmtOpenKFD());
    printf("[OK] hsaKmtOpenKFD() 成功\n");

    /* 方法1: IOCTL */
    int kfd_fd = open("/dev/kfd", O_RDWR);
    if (kfd_fd >= 0) {
        kfd_ioctl_version(kfd_fd);
        close(kfd_fd);
    } else {
        perror("open /dev/kfd");
    }

    /* 方法2: HSAKMT */
    enumerate_nodes();

    /* 方法3: sysfs */
    show_sysfs_topology();

    HSAKMT_CHECK(hsaKmtCloseKFD());
    printf("\n[OK] hsaKmtCloseKFD() 成功\n");
    printf("\n=== 完成 ===\n");
    return 0;
}
