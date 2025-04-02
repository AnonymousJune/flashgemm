export LD_LIBRARY_PATH=/home/wangpy/zjw/install/oneDNN/lib:$LD_LIBRARY_PATH
export ONEDNN_MAX_CPU_ISA=AVX512_CORE_FP16

export GOMP_CPU_AFFINITY="23"
export OMP_NUM_THREADS=1

export KMP_HW_SUBSET=1T
export OMP_PROC_BIND=spread
export OMP_PLACES=threads
numactl --membind 0 --cpunodebind 0 ./f32_3_onednn.out
numactl --membind 0 --cpunodebind 0 ./f32_onednn.out
numactl --membind 0 --cpunodebind 0 ./bf16_3_onednn.out
numactl --membind 0 --cpunodebind 0 ./bf16_onednn.out
numactl --membind 0 --cpunodebind 0 ./int8_3_onednn.out
numactl --membind 0 --cpunodebind 0 ./int8_onednn.out