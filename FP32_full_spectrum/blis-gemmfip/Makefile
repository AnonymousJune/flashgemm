BLIS_PREFIX = $ /home/wangpy/zjw/install/BLIS-gemmfip
BLIS_INC    = $(BLIS_PREFIX)/include/blis
BLIS_LIB    = $(BLIS_PREFIX)/lib/libblis.a

OTHER_LIBS  = -lm -fopenmp -lpthread

CC          = g++
CFLAGS      = -O2 -g -fopenmp -mavx512f -I$(BLIS_INC)
LINKER      = $(CC)

OBJS        = FP32_ID23.o

%.o: %.c
	 $(CC) $(CFLAGS) -fopenmp -c $< -o $@

all: $(OBJS)
	$(LINKER) $(OBJS) $(BLIS_LIB) $(OTHER_LIBS) -o a.out
clean: 
	rm -rf *.o *.out
