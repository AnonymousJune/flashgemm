export GOMP_CPU_AFFINITY="0-31"
export OMP_NUM_THREADS=32 
numactl --membind 0 --cpunodebind 0 ./test_bf16_3.out
numactl --membind 0 --cpunodebind 0 ./test_bf16.out
numactl --membind 0 --cpunodebind 0 ./test_int8_3.out
numactl --membind 0 --cpunodebind 0 ./test_int8.out
