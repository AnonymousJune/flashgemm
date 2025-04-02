export GOMP_CPU_AFFINITY="0-47"
export OMP_NUM_THREADS=48 
numactl --membind 0 --cpunodebind 0 ./f32_blis.out
numactl --membind 0 --cpunodebind 0 ./f32_3_blis.out
