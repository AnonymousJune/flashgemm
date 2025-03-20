#ifndef FLASHGEMM_H
#define FLASHGEMM_H

#ifdef __cplusplus
extern "C" {
#endif

#include <omp.h>
#include "./PACK_x86.c"
#include "./kernel_bf16.c"

#define GEMM_K 1024
#define GEMM_M 516

extern int flashgemm_thread_num; // init here is wrong

void flashgemm_set_thread_num(int num);

void flashgemm_multi_bf16bf16f32_type_a(float *C, uint16_t *A_bf16, uint16_t *B_bf16, long M, long N, long K, int beta);


#ifdef __cplusplus
}
#endif

#endif