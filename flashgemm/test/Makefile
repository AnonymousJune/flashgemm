MKLROOT		= /home/wangpy/zjw/intel/oneapi/mkl/2025.0
LDFLAGS		= # -m64 -L${MKLROOT}/lib/intel64 -Wl,--no-as-needed -lmkl_intel_ilp64 -lmkl_sequential -lmkl_core -lmkl_blacs_openmpi_ilp64 -lpthread -lm -ldl
INCFLAGS	= # -I$(MKLROOT)/include

OTHER_LIBS  = -lm -fopenmp # -lmpi_cxx -lmpi

CC          = g++
CFLAGS      = -O2 -g -w -fopenmp -mavx512f # -DMKL_ILP64 -m64 $(INCFLAGS)
GDB_FLAGS   = # -gdwarf-5 -gstrict-dwarf
LINKER      = $(CC)

# SRCS        = $(wildcard *.c)
SRCS        = test_f32.c
OBJS        = $(SRCS:.c=.o)
TARGETS     = $(SRCS:.c=.out)

%.o: %.c
	$(CC) $(CFLAGS) $(GDB_FLAGS) -fopenmp -c $< -o $@

%.out: %.o
	$(LINKER) $< $(GDB_FLAGS) -o $@ $(LDFLAGS) $(OTHER_LIBS)

all: $(TARGETS)

clean:
	rm -rf *.o *.out
