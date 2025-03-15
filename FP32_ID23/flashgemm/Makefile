BLIS_PREFIX = $ /home/wangpy/zjw/install/OpenBLAS
BLIS_INC    = $(BLIS_PREFIX)/include
BLIS_LIB    = $(BLIS_PREFIX)/lib/libopenblas.a

OTHER_LIBS  = -lm -fopenmp

CC          = g++
CFLAGS      = -O2 -g -fopenmp -mavx512f -I$(BLIS_INC)
GDB_FLAGS   = -gdwarf-5 -gstrict-dwarf 
LINKER      = $(CC)

OBJS        = FP32_ID23.o

%.o: %.c
	 $(CC) $(CFLAGS) $(GDB_FLAGS) -fopenmp -c $< -o $@

all: $(OBJS)
	$(LINKER) $(OBJS) $(BLIS_LIB) $(OTHER_LIBS) $(GDB_FLAGS) -o a.out
clean: 
	rm -rf *.o *.out
