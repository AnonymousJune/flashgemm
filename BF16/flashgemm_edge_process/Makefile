# source /home/wangpy/zjw/intel/oneapi/setvars.sh
MKLROOT		= /home/wangpy/zjw/intel/oneapi/mkl/2025.0
LDFLAGS		= -m64  -L${MKLROOT}/lib/intel64 -Wl,--no-as-needed -lmkl_intel_ilp64 -lmkl_sequential -lmkl_core -lmkl_blacs_openmpi_ilp64 -lpthread -lm -ldl
INCFLAGS	= -I$(MKLROOT)/include

OTHER_LIBS  = -lm -fopenmp  -lmpi_cxx -lmpi

CC          = g++
CFLAGS      = -O2 -g -fopenmp -mavx512f -DMKL_ILP64 -m64 $(INCFLAGS)
GDB_FLAGS   = -gdwarf-5 -gstrict-dwarf
LINKER      = $(CC)

OBJS        = test.o

%.o: %.c
	 $(CC) $(CFLAGS) $(GDB_FLAGS) -fopenmp -c $< -o $@

all: $(OBJS)
	$(LINKER) $(OBJS) $(GDB_FLAGS) -o a.out $(LDFLAGS) $(OTHER_LIBS)
clean: 
	rm -rf *.o *.out
