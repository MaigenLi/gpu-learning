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

/* KFD IOCTL GPU 详细信息 */
static void kfd_ioctl_gpu_info(int kfd_fd) {
    printf("\n--- KFD IOCTL GPU 信息 ---\n");

    /* 获取所有 GPU 的 apertures（包含 gpu_id）*/
    struct kfd_ioctl_get_process_apertures_args apergs;
    memset(&apergs, 0, sizeof(apergs));
    if (ioctl(kfd_fd, AMDKFD_IOC_GET_PROCESS_APERTURES, &apergs) < 0) {
        perror("  AMDKFD_IOC_GET_PROCESS_APERTURES");
        return;
    }
    printf("  发现 GPU 数: %u\n\n", apergs.num_of_nodes);

    for (unsigned i = 0; i < apergs.num_of_nodes && i < 7; i++) {
        struct kfd_process_device_apertures *a = &apergs.process_apertures[i];
        printf("  === GPU[%u] gpu_id=0x%x ===\n", i, a->gpu_id);

        /* 时钟计数器 */
        struct kfd_ioctl_get_clock_counters_args clock_args = {0};
        clock_args.gpu_id = a->gpu_id;
        if (ioctl(kfd_fd, AMDKFD_IOC_GET_CLOCK_COUNTERS, &clock_args) == 0) {
            double gpu_mhz = clock_args.gpu_clock_counter / 1e6;
            double sys_mhz = clock_args.system_clock_freq / 1e6;
            printf("    GPU Clock:     %.2f MHz (counter=%llu)\n",
                   gpu_mhz, (unsigned long long)clock_args.gpu_clock_counter);
            printf("    CPU Clock:     %llu\n",
                   (unsigned long long)clock_args.cpu_clock_counter);
            printf("    System Clock:  %.2f MHz (freq=%.2f MHz)\n",
                   clock_args.system_clock_counter / 1e6, sys_mhz);
        } else {
            perror("    AMDKFD_IOC_GET_CLOCK_COUNTERS");
        }

        /* Tile 配置 */
        __u32 tile_config[32] = {0};
        __u32 macro_tile[16] = {0};
        struct kfd_ioctl_get_tile_config_args tile_args = {0};
        tile_args.tile_config_ptr = (uint64_t)tile_config;
        tile_args.macro_tile_config_ptr = (uint64_t)macro_tile;
        tile_args.num_tile_configs = 32;
        tile_args.num_macro_tile_configs = 16;
        tile_args.gpu_id = a->gpu_id;
        if (ioctl(kfd_fd, AMDKFD_IOC_GET_TILE_CONFIG, &tile_args) == 0) {
            printf("    Tile Config:\n");
            printf("      Banks:      %u\n", tile_args.num_banks);
            printf("      Ranks:      %u\n", tile_args.num_ranks);
            printf("      AddrConfig: 0x%x\n", tile_args.gb_addr_config);
            printf("      TileConfigs filled: %u\n", tile_args.num_tile_configs);
            if (tile_args.num_tile_configs > 0) {
                printf("      Tile[0]:    0x%x\n", tile_config[0]);
            }
            printf("      MacroTileConfigs filled: %u\n",
                   tile_args.num_macro_tile_configs);
        } else {
            perror("    AMDKFD_IOC_GET_TILE_CONFIG");
        }

        /* 可用显存 */
        struct kfd_ioctl_get_available_memory_args mem_args = {0};
        mem_args.gpu_id = a->gpu_id;
        if (ioctl(kfd_fd, AMDKFD_IOC_AVAILABLE_MEMORY, &mem_args) == 0) {
            double avail_mb = (double)mem_args.available / 1024 / 1024;
            printf("    可用显存:   %.2f MB\n", avail_mb);
        } else {
            perror("    AMDKFD_IOC_GET_AVAILABLE_MEMORY");
        }

        /* GPU VM Aperture */
        printf("    GPU VM:     0x%llx - 0x%llx\n",
               (unsigned long long)a->gpuvm_base,
               (unsigned long long)a->gpuvm_limit);
        printf("    Scratch:    0x%llx - 0x%llx\n",
               (unsigned long long)a->scratch_base,
               (unsigned long long)a->scratch_limit);
        printf("    LDS:        0x%llx - 0x%llx\n",
               (unsigned long long)a->lds_base,
               (unsigned long long)a->lds_limit);
        printf("\n");
    }
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

    /* KFD 设备 IOCTL */
    int kfd_fd = open("/dev/kfd", O_RDWR);
    if (kfd_fd >= 0) {
        kfd_ioctl_version(kfd_fd);
        kfd_ioctl_gpu_info(kfd_fd);   // ← 新增：GPU 时钟/Tile/显存
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
