export KMP_HW_SUBSET=1T
export OMP_PROC_BIND=spread
export OMP_PLACES=threads
export OMP_NUM_THREADS=24 
./test_f32.out
./test_f32_3.out
./test_bf16.out
./test_bf16_3.out
./test_int8.out
./test_int8_3.out