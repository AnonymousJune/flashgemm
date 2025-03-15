export GOMP_CPU_AFFINITY="0-23"
export OMP_NUM_THREADS=24
export BLIS_NUM_THREADS=24

export KMP_HW_SUBSET=1T # Use 1 hardware thread per core
export OMP_PROC_BIND=spread
export OMP_PLACES=threads
numactl --membind 0 --cpunodebind 0 ./a.out
