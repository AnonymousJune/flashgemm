BINARY_NAME = gemm
CC          = /home/wangpy/zjw/intel/oneapi/2025.0/bin/dpcpp
CFLAGS      = -O2 -g -march=native -fno-strict-aliasing -mavx512f -mavx512vnni -mavx -mavxvnni
GDB_FLAGS   = -gdwarf-5 -gstrict-dwarf
FLASHGEMM_DIR = /home/wangpy/zjw/install/flashgemm

MKLPATH     = /home/wangpy/zjw/intel/oneapi/mkl/2025.0
LDFLAGS     = -L$(MKLPATH)/lib/intel64 
INCFLAGS    = -I$(MKLPATH)/include -I$(FLASHGEMM_DIR)/include

OTHER_FLGAS = -qmkl=parallel
OTHER_LIBS  = $(FLASHGEMM_DIR)/lib/libflashgemm.a

SRC         = $(wildcard *.c)
OBJ         = $(SRC:.c=.o)

# Target to build the binary
build: $(BINARY_NAME)

$(BINARY_NAME): $(OBJ)
	$(CC) $(LDFLAGS) $(OBJ) $(OTHER_LIBS) $(OTHER_FLGAS) -o $@

# Pattern rule to compile source files to object files
%.o: %.c
	$(CC) $(CFLAGS) $(GDB_FLAGS) $(INCFLAGS) -c $< -o $@

# Clean up generated files
clean:
	rm -f $(BINARY_NAME) $(OBJ)
