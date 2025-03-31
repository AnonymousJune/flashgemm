export KMP_HW_SUBSET=1T
export OMP_PROC_BIND=spread
export OMP_PLACES=threads
export OMP_NUM_THREADS=24 
numactl --membind 0 --cpunodebind 0 ./a.out