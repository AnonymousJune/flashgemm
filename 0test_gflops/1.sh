# from /home/wangpy/zjw/download/oneDNN/doc/performance_considerations/perf_settings.md

# Several Cores Within a NUMA Domain, with a single hardware thread per core
export KMP_HW_SUBSET=1T # Use 1 hardware thread per core
export OMP_PROC_BIND=spread
export OMP_PLACES=threads
export OMP_NUM_THREADS=1 # desired number of cores to use
numactl --membind 1 --cpunodebind 1 ./a.out

# Single NUMA Domain
# export OMP_PROC_BIND=spread
# export OMP_PLACES=threads
# export OMP_NUM_THREADS=24 # number of cores in NUMA domain 0
# numactl --membind 0 --cpunodebind 0 ./matmul