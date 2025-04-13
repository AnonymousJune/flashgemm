#ifndef FLASHGEMM_H
#define FLASHGEMM_H

#ifdef __cplusplus
extern "C" {
#endif

#include <omp.h>
#include "./PACK_x86.h"
#include "./kernel_f32.h"
#include "./kernel_bf16.h"
#include "./kernel_int8.h"

#define GEMM_K 1024
#define GEMM_M 516

extern int flashgemm_thread_num; // init here is wrong

void flashgemm_set_thread_num(int num);

void flashgemm_multi_f32f32f32(float *C, float *B, long N, int num_gemm, ...);

void flashgemm_multi_bf16bf16f32(float *C, uint16_t *B, long N, int num_gemm, ...);

void flashgemm_multi_int8uint8int32(int *C, uint8_t *B, long N, int num_gemm, ...);


#ifdef __cplusplus
}
#endif

#endif