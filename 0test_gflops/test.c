#include <stdio.h>
#include <time.h>
#include <math.h>
#include <stdint.h>
#include <sys/time.h>

static double gtod_ref_time_sec = 0.0;
double dclock()
{
	double the_time, norm_sec;
	struct timeval tv;

	gettimeofday(&tv, NULL);

	if (gtod_ref_time_sec == 0.0)
		gtod_ref_time_sec = (double)tv.tv_sec;

	norm_sec = (double)tv.tv_sec - gtod_ref_time_sec;
	the_time = norm_sec + tv.tv_usec * 1.0e-6;
	return the_time;
}

void test_fp32(long loop)
{
	asm volatile(

		"   mov 			%[loop], %%rax   	 					\n"

		"MAIN_FP32:														\n"
		
		"	vfmadd231ps 	%%zmm0, %%zmm1, %%zmm2	 				\n"
		"	vfmadd231ps 	%%zmm3, %%zmm4, %%zmm5	 				\n"
		"	vfmadd231ps 	%%zmm6, %%zmm7, %%zmm8	 				\n"
		"	vfmadd231ps 	%%zmm9, %%zmm10, %%zmm11	 				\n"

		"	subq		$1, %%rax									\n"

		"	vfmadd231ps 	%%zmm12, %%zmm13, %%zmm14	 				\n"
		"	vfmadd231ps 	%%zmm15, %%zmm16, %%zmm17	 				\n"
		"	vfmadd231ps 	%%zmm18, %%zmm19, %%zmm20	 				\n"
		"	vfmadd231ps 	%%zmm21, %%zmm22, %%zmm23	 				\n"

		"	vfmadd231ps 	%%zmm24, %%zmm25, %%zmm26	 				\n"
		"	vfmadd231ps 	%%zmm27, %%zmm28, %%zmm29	 				\n"
		
		"	cmpq		$0, %%rax  								\n"
		"	jg 			MAIN_FP32									\n"
		:
		:
		[loop] "m"(loop)
		: "rax", "rbx", "rcx", "rdx", "rdi", "rsi", "rbp", "r8", "r9", "r10", "r11", "r12",
		  "r13", "r14", "r15", "zmm0", "zmm1", "zmm2", "zmm3", "zmm4", "zmm5",
		  "zmm6", "zmm7", "zmm8", "zmm9", "zmm10", "zmm11", "zmm12", "zmm13",
		  "zmm14", "zmm15", "zmm16", "zmm17", "zmm18", "zmm19", "zmm20", "zmm21",
		  "zmm22", "zmm23", "zmm24", "zmm25", "zmm26", "zmm27", "zmm28", "zmm29",
		  "zmm30", "zmm31", "memory");
}

void test_int32(long loop)
{
	asm volatile(

		"   mov 			%[loop], %%rax   	 					\n"

		"MAIN_INT32:														\n"
		
		"	vpmaddwd  	%%zmm0, %%zmm1, %%zmm2	 				\n"
		"	vpmaddwd  	%%zmm3, %%zmm4, %%zmm5	 				\n"
		"	vpmaddwd  	%%zmm6, %%zmm7, %%zmm8	 				\n"
		"	vpmaddwd  	%%zmm9, %%zmm10, %%zmm11	 				\n"

		"	subq		$1, %%rax									\n"

		"	vpmaddwd  	%%zmm12, %%zmm13, %%zmm14	 				\n"
		"	vpmaddwd  	%%zmm15, %%zmm16, %%zmm17	 				\n"
		"	vpmaddwd  	%%zmm18, %%zmm19, %%zmm20	 				\n"
		"	vpmaddwd  	%%zmm21, %%zmm22, %%zmm23	 				\n"

		"	vpmaddwd  	%%zmm24, %%zmm25, %%zmm26	 				\n"
		"	vpmaddwd  	%%zmm27, %%zmm28, %%zmm29	 				\n"
		
		"	cmpq		$0, %%rax  								\n"
		"	jg 			MAIN_INT32									\n"
		:
		:
		[loop] "m"(loop)
		: "rax", "rbx", "rcx", "rdx", "rdi", "rsi", "rbp", "r8", "r9", "r10", "r11", "r12",
		  "r13", "r14", "r15", "zmm0", "zmm1", "zmm2", "zmm3", "zmm4", "zmm5",
		  "zmm6", "zmm7", "zmm8", "zmm9", "zmm10", "zmm11", "zmm12", "zmm13",
		  "zmm14", "zmm15", "zmm16", "zmm17", "zmm18", "zmm19", "zmm20", "zmm21",
		  "zmm22", "zmm23", "zmm24", "zmm25", "zmm26", "zmm27", "zmm28", "zmm29",
		  "zmm30", "zmm31", "memory");
}


void test_bf16(long loop)
{
	asm volatile(

		"   mov 			%[loop], %%rax   	 					\n"

		"MAIN_BF16:														\n"
		
		"	vdpbf16ps 	%%zmm0, %%zmm1, %%zmm2	 				\n"
		"	vdpbf16ps 	%%zmm3, %%zmm4, %%zmm5	 				\n"
		"	vdpbf16ps 	%%zmm6, %%zmm7, %%zmm8	 				\n"
		"	vdpbf16ps 	%%zmm9, %%zmm10, %%zmm11	 				\n"

		"	subq		$1, %%rax									\n"

		"	vdpbf16ps 	%%zmm12, %%zmm13, %%zmm14	 				\n"
		"	vdpbf16ps 	%%zmm15, %%zmm16, %%zmm17	 				\n"
		"	vdpbf16ps 	%%zmm18, %%zmm19, %%zmm20	 				\n"
		"	vdpbf16ps 	%%zmm21, %%zmm22, %%zmm23	 				\n"

		"	vdpbf16ps 	%%zmm24, %%zmm25, %%zmm26	 				\n"
		"	vdpbf16ps 	%%zmm27, %%zmm28, %%zmm29	 				\n"

		"	cmpq		$0, %%rax  								\n"
		"	jg 			MAIN_BF16									\n"
		:
		:
		[loop] "m"(loop)
		: "rax", "rbx", "rcx", "rdx", "rdi", "rsi", "rbp", "r8", "r9", "r10", "r11", "r12",
		  "r13", "r14", "r15", "zmm0", "zmm1", "zmm2", "zmm3", "zmm4", "zmm5",
		  "zmm6", "zmm7", "zmm8", "zmm9", "zmm10", "zmm11", "zmm12", "zmm13",
		  "zmm14", "zmm15", "zmm16", "zmm17", "zmm18", "zmm19", "zmm20", "zmm21",
		  "zmm22", "zmm23", "zmm24", "zmm25", "zmm26", "zmm27", "zmm28", "zmm29",
		  "zmm30", "zmm31", "memory");
}

void test_int8_1(long loop)
{
	asm volatile(

		"   mov 			%[loop], %%rax   	 					\n"

		"MAIN_INT8:														\n"
		
		"	vpdpbusd 	%%zmm0, %%zmm1, %%zmm2	 				\n"
		"	vpdpbusd 	%%zmm3, %%zmm4, %%zmm5	 				\n"
		"	vpdpbusd 	%%zmm6, %%zmm7, %%zmm8	 				\n"
		"	vpdpbusd 	%%zmm9, %%zmm10, %%zmm11	 				\n"

		"	subq		$1, %%rax									\n"

		"	vpdpbusd 	%%zmm12, %%zmm13, %%zmm14	 				\n"
		"	vpdpbusd 	%%zmm15, %%zmm16, %%zmm17	 				\n"
		"	vpdpbusd 	%%zmm18, %%zmm19, %%zmm20	 				\n"
		"	vpdpbusd 	%%zmm21, %%zmm22, %%zmm23	 				\n"

		"	vpdpbusd 	%%zmm24, %%zmm25, %%zmm26	 				\n"
		"	vpdpbusd 	%%zmm27, %%zmm28, %%zmm29	 				\n"

		"	cmpq		$0, %%rax  								\n"
		"	jg 			MAIN_INT8									\n"
		:
		:
		[loop] "m"(loop)
		: "rax", "rbx", "rcx", "rdx", "rdi", "rsi", "rbp", "r8", "r9", "r10", "r11", "r12",
		  "r13", "r14", "r15", "zmm0", "zmm1", "zmm2", "zmm3", "zmm4", "zmm5",
		  "zmm6", "zmm7", "zmm8", "zmm9", "zmm10", "zmm11", "zmm12", "zmm13",
		  "zmm14", "zmm15", "zmm16", "zmm17", "zmm18", "zmm19", "zmm20", "zmm21",
		  "zmm22", "zmm23", "zmm24", "zmm25", "zmm26", "zmm27", "zmm28", "zmm29",
		  "zmm30", "zmm31", "memory");
}

void test_int8_2(long loop)
{
	asm volatile(

		"   mov 			%[loop], %%rax   	 					\n"

		"MAIN_INT82:														\n"
		
		"	vpdpbusd 	%%zmm0, %%zmm1, %%zmm10	 				\n"
		"	vpdpbusd 	%%zmm2, %%zmm3, %%zmm11	 				\n"
		"	vpdpbusd 	%%zmm4, %%zmm5, %%zmm12	 				\n"
		"	vpdpbusd 	%%zmm6, %%zmm7, %%zmm13	 				\n"

		"	subq		$1, %%rax									\n"

		"	vpdpbusd 	%%zmm0, %%zmm1, %%zmm10	 				\n"
		"	vpdpbusd 	%%zmm2, %%zmm3, %%zmm11	 				\n"
		"	vpdpbusd 	%%zmm4, %%zmm5, %%zmm12	 				\n"
		"	vpdpbusd 	%%zmm6, %%zmm7, %%zmm13	 				\n"

		"	vpdpbusd 	%%zmm0, %%zmm1, %%zmm10	 				\n"
		"	vpdpbusd 	%%zmm2, %%zmm3, %%zmm11	 				\n"
		"	vpdpbusd 	%%zmm4, %%zmm5, %%zmm12	 				\n"
		"	vpdpbusd 	%%zmm6, %%zmm7, %%zmm13	 				\n"

		"	vpdpbusd 	%%zmm0, %%zmm1, %%zmm10	 				\n"
		"	vpdpbusd 	%%zmm2, %%zmm3, %%zmm11	 				\n"
		"	vpdpbusd 	%%zmm4, %%zmm5, %%zmm12	 				\n"
		"	vpdpbusd 	%%zmm6, %%zmm7, %%zmm13	 				\n"

		"	cmpq		$0, %%rax  								\n"
		"	jg 			MAIN_INT82									\n"
		:
		:
		[loop] "m"(loop)
		: "rax", "rbx", "rcx", "rdx", "rdi", "rsi", "rbp", "r8", "r9", "r10", "r11", "r12",
		  "r13", "r14", "r15", "zmm0", "zmm1", "zmm2", "zmm3", "zmm4", "zmm5",
		  "zmm6", "zmm7", "zmm8", "zmm9", "zmm10", "zmm11", "zmm12", "zmm13",
		  "zmm14", "zmm15", "zmm16", "zmm17", "zmm18", "zmm19", "zmm20", "zmm21",
		  "zmm22", "zmm23", "zmm24", "zmm25", "zmm26", "zmm27", "zmm28", "zmm29",
		  "zmm30", "zmm31", "memory");
}


int main()
{

	long loop = 1000000;
	double ops_fp32 = 16 * 10 * 2 * 1.0e-09 * loop;
	double ops_int32 = 16 * 10 * 2 * 1.0e-09 * loop;
	double ops_bf16 = 32 * 10 * 2 * 1.0e-09 * loop;
	double ops_int8 = 64 * 10 * 2 * 1.0e-09 * loop;

	// vfmadd231ps 计时
	double start_f32 = dclock();
	test_fp32(loop);
	double cost_f32 = dclock() - start_f32;
	printf("FP32_GFLOPS(vfmadd231ps): cost=%.3f Gflops = %.3f\n", cost_f32, ops_fp32 / cost_f32);

	// vpmaddwd  计时
	double start_int32 = dclock();
	test_int32(loop);
	double cost_int32 = dclock() - start_int32;
	printf("INT32_GFLOPS(vpmaddwd): cost=%.3f Gflops = %.3f\n", cost_int32, ops_int32 / cost_int32);

	// vdpbf16ps 计时
	double start_bf16 = dclock();
	test_bf16(loop);
	double cost_bf16 = dclock() - start_bf16;
	printf("BF16_GFLOPS(vdpbf16ps): cost=%.3f Gflops = %.3f\n", cost_bf16, ops_bf16 / cost_bf16);

	// vpdpbusd 计时
	double start_int8 = dclock();
	test_int8_1(loop);
	double cost_int8 = dclock() - start_int8;
	printf("INT8_GFLOPS(vpdpbusd)1: cost=%.3f Gflops = %.3f\n", cost_int8, ops_int8 / cost_int8);

	// vpdpbusd 计时
	int ops_int8_2 = 64 * 16 * 2 * 1.0e-09 * loop;
	double start_int8_2 = dclock();
	test_int8_2(loop);
	double cost_int8_2 = dclock() - start_int8_2;
	printf("INT8_GFLOPS(vpdpbusd)2: cost=%.3f Gflops = %.3f\n", cost_int8_2, ops_int8_2 / cost_int8_2);

	return 0;
}