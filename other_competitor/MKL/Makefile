CC          = /home/wangpy/zjw/intel/oneapi/2025.0/bin/dpcpp
CFLAGS      = -O2 -g
MKLPATH     = /home/wangpy/zjw/oneapi/mkl/2025.0
LDFLAGS     = -L$(MKLPATH)/lib/intel64
INCFLAGS    = -I$(MKLPATH)/include
OTHER_FLGAS = -qmkl=parallel

# 自动获取所有.c文件
SRC := $(wildcard *.c)
# 生成对应的目标列表（文件名去后缀）
EXECS := $(SRC:.c=)

# 默认构建所有可执行文件
all: $(EXECS)

# 模式规则：每个.c生成同名可执行文件
%: %.c
	$(CC) $(CFLAGS) $(LDFLAGS) $(INCFLAGS) $(OTHER_FLGAS) $^ -o $@

# 清理所有生成的可执行文件
clean:
	rm -f $(EXECS)