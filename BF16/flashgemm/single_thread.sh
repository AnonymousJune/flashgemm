export GOMP_CPU_AFFINITY="47"
export OMP_NUM_THREADS=1 
numactl --membind 0 --cpunodebind 0 ./a.out
