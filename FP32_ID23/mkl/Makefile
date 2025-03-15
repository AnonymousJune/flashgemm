# source /home/wangpy/zjw/intel/oneapi/setvars.sh
BINARY_NAME = testMKL
CC			= /home/wangpy/zjw/intel/oneapi/2024.2/bin/dpcpp
CFLAGS		= -O2 -g
MKLPATH		= /home/wangpy/zjw/oneapi/mkl/2024.2
LDFLAGS		= -L$(MKLPATH)/lib/intel64
INCFLAGS	= -I$(MKLPATH)/include


OTHER_FLGAS  = -qmkl=parallel
OTHER_LIBS   = 

SRC			= $(wildcard *.c)
build : $(BINARY_NAME)

$(BINARY_NAME): $(SRC)
	$(CC) $(CFLAGS) $(LDFLAGS) $(INCFLAGS) $(OTHER_FLGAS) $(OTHER_LIBS) $(SRC) -o $(BINARY_NAME)

clean:
	rm $(BINARY_NAME)
