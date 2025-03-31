export LD_LIBRARY_PATH=/home/wangpy/zjw/install/oneDNN/lib:$LD_LIBRARY_PATH
export ONEDNN_MAX_CPU_ISA=AVX512_CORE_VNNI
export ONEDNN_CPU_ISA_HINTS=no_hints

export GOMP_CPU_AFFINITY="0-31"
export OMP_NUM_THREADS=32

export KMP_HW_SUBSET=1T
export OMP_PROC_BIND=spread
export OMP_PLACES=threads
numactl --membind 0 --cpunodebind 0 ./matmul