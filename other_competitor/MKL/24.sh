export MKL_DYNAMIC=FALSE
export GOMP_CPU_AFFINITY="0-23"
export OMP_NUM_THREADS=24 
export KMP_HW_SUBSET=1T
export OMP_PROC_BIND=spread
export OMP_PLACES=threads
export MKL_NUM_THREADS=24
numactl --membind 0 --cpunodebind 0 ./f32_mkl.out
numactl --membind 0 --cpunodebind 0 ./f32_mkl
numactl --membind 0 --cpunodebind 0 ./f32_3_mkl
numactl --membind 0 --cpunodebind 0 ./bf16_mkl
numactl --membind 0 --cpunodebind 0 ./bf16_3_mkl
numactl --membind 0 --cpunodebind 0 ./int8_mkl
numactl --membind 0 --cpunodebind 0 ./int8_3_mkl
