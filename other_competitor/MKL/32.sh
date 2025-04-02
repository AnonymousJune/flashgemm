export MKL_DYNAMIC=FALSE
export MKL_ENABLE_INSTRUCTIONS=AVX512
export GOMP_CPU_AFFINITY="0-31"
export MKL_NUM_THREADS=32
numactl --membind 0 --cpunodebind 0 ./f32_mkl
numactl --membind 0 --cpunodebind 0 ./f32_3_mkl
numactl --membind 0 --cpunodebind 0 ./bf16_mkl
numactl --membind 0 --cpunodebind 0 ./bf16_3_mkl
numactl --membind 0 --cpunodebind 0 ./int8_mkl
numactl --membind 0 --cpunodebind 0 ./int8_3_mkl
