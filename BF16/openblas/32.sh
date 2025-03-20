export GOMP_CPU_AFFINITY="0-31"
export OMP_NUM_THREADS=32 
numactl --membind 0 --cpunodebind 0 ./a.out
