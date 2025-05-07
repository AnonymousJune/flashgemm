export GOMP_CPU_AFFINITY="0-11"
export OMP_NUM_THREADS=12
export KMP_HW_SUBSET=1T # Use 1 hardware thread per core
export OMP_PROC_BIND=spread
export OMP_PLACES=threads
./test_f32.out
./test_f32_3.out
./test_bf16.out
./test_bf16_3.out
./test_int8.out
./test_int8_3.out
