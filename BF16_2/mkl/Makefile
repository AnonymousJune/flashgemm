# source /home/wangpy/zjw/intel/oneapi/setvars.sh
BINARY_NAME = testMKL
CC			= /home/wangpy/zjw/intel/oneapi/2025.0/bin/dpcpp
CFLAGS		= -O2 -g 
MKLPATH		= /home/wangpy/zjw/oneapi/mkl/2025.0
LDFLAGS		= -L$(MKLPATH)/lib/intel64  -lmkl_intel_lp64 -lmkl_intel_thread -lmkl_core -liomp5 -lpthread -lm -ldl
INCFLAGS	= -I$(MKLPATH)/include


OTHER_FLGAS  =  -DMKL_ILP64  -qmkl-ilp64=parallel
OTHER_LIBS   = 

SRC			= $(wildcard *.c)
build : $(BINARY_NAME)

$(BINARY_NAME): $(SRC)
	$(CC) $(CFLAGS) $(LDFLAGS) $(INCFLAGS) $(OTHER_FLGAS) $(OTHER_LIBS) $(SRC) -o $(BINARY_NAME)

clean:
	rm $(BINARY_NAME)
