# KFD 代码示例 Makefile
# 用法: make [target]  或  make all

ROCM_PATH ?= /opt/rocm-7.2.0
HSAKMT_INC = -I$(ROCM_PATH)/include
HSAKMT_LIB = -L$(ROCM_PATH)/lib -lhsakmt
CC = gcc
CFLAGS = -Wall -O2 -g $(HSAKMT_INC)
LDFLAGS = -lpthread $(HSAKMT_LIB) -ldrm -ldrm_amdgpu -lnuma

# 所有示例
TARGETS = \
	01-kfd-basics/kfd_version \
	02-queue-management/kfd_queue \
	03-memory-model/kfd_memory

.PHONY: all clean

all: $(TARGETS)

# KFD 基础示例
01-kfd-basics/kfd_version: 01-kfd-basics/kfd_version.c
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

# 队列管理示例
02-queue-management/kfd_queue: 02-queue-management/kfd_queue.c
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

# 内存模型示例
03-memory-model/kfd_memory: 03-memory-model/kfd_memory.c
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

clean:
	rm -f $(TARGETS)
	rm -f 01-kfd-basics/kfd_version
	rm -f 02-queue-management/kfd_queue
	rm -f 03-memory-model/kfd_memory
