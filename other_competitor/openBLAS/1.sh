export GOMP_CPU_AFFINITY="31"
export OMP_NUM_THREADS=1 
numactl --membind 0 --cpunodebind 0 ./f32_openblas.out
numactl --membind 0 --cpunodebind 0 ./f32_3_openblas.out
numactl --membind 0 --cpunodebind 0 ./bf16_openblas.out
numactl --membind 0 --cpunodebind 0 ./bf16_3_openblas.out
