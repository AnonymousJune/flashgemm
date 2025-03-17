export GOMP_CPU_AFFINITY="0-23"
export OMP_NUM_THREADS=24 
numactl --membind 0 --cpunodebind 0 ./a.out
