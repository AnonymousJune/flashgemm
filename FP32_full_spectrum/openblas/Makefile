# 路径配置
BLIS_PREFIX = /home/wangpy/zjw/install/OpenBLAS
BLIS_INC    = $(BLIS_PREFIX)/include
BLIS_LIB    = $(BLIS_PREFIX)/lib/libopenblas.a

OTHER_LIBS  = -lm -fopenmp -lpthread

CC          = gcc
CFLAGS      = -O2 -g -fopenmp -mavx512f -I$(BLIS_INC)
LINKER      = $(CC)

# 自动获取所有 .c 文件并生成对应的目标文件和可执行文件名
SRCS        = $(wildcard *.c)         # 当前目录下所有 .c 文件
OBJS        = $(SRCS:.c=.o)           # 生成对应 .o 文件
TARGETS     = $(SRCS:.c=.out)         # 生成与 .c 同名的 .out 文件

# 默认目标：编译所有源文件
all: $(TARGETS)

# 编译规则：.c -> .o
%.o: %.c
	$(CC) $(CFLAGS) -fopenmp -c $< -o $@

# 链接规则：.o -> .out（每个 .o 生成独立的 .out 文件）
%.out: %.o
	$(LINKER) $^ $(BLIS_LIB) $(OTHER_LIBS) -o $@

# 清理规则
clean: 
	rm -f *.o *.out