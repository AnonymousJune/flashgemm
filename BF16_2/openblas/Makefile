BLIS_PREFIX = $ /home/wangpy/zjw/install/OpenBLAS
BLIS_INC    = $(BLIS_PREFIX)/include/openblas
BLIS_LIB    = $(BLIS_PREFIX)/lib/libopenblas.a

OTHER_LIBS  = -lm -fopenmp -lpthread

CC          = gcc
CFLAGS      = -O2 -g -fopenmp -mavx512f -I$(BLIS_INC)
LINKER      = $(CC)

OBJS        = test.o

%.o: %.c
	 $(CC) $(CFLAGS) -fopenmp -c $< -o $@

all: $(OBJS)
	$(LINKER) $(OBJS) $(BLIS_LIB) $(OTHER_LIBS) -o a.out
clean: 
	rm -rf *.o *.out
