# Makefile for compiling static library with OpenMP

# Compiler settings
CC = gcc
CFLAGS = -O2 -g -mavx512f -fopenmp -Wall -I.
AR = ar
ARFLAGS = rcs
INSTALL_DIR = /home/wangpy/zjw/install/flashgemm

# Library name
LIB_NAME = flashgemm

# Source files (only flashgemm.c needs to be compiled)
SRCS = flashgemm.c
OBJS = $(SRCS:.c=.o)

all: lib$(LIB_NAME).a

lib$(LIB_NAME).a: $(OBJS)
	$(AR) $(ARFLAGS) $@ $^

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -f $(OBJS) lib$(LIB_NAME).a

install:
	mkdir -p $(INSTALL_DIR)/lib $(INSTALL_DIR)/include
	cp lib$(LIB_NAME).a $(INSTALL_DIR)/lib/
	cp ./*.h ./*.c $(INSTALL_DIR)/include/

uninstall:
	rm -f $(INSTALL_DIR)/lib/*
	rm -f $(INSTALL_DIR)/include/*


.PHONY: all clean install uninstall
