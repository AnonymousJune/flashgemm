#include <stdlib.h>
#include <stdio.h>
#include <cstdint>
#include <immintrin.h>
#include <string.h>
// #include "mkl.h"
#include "../src/flashgemm.c"
// #include "utils.h"

using namespace std;
#define PEAK_GFLOPS 3.6
#define NUM 24

int GEMM1[30] = {
	32,65536,288,
	144,16384,1296,
	192,16384,1728,
	192,4096,4800,
	336,4096,8400,
	32,50176,288,
	96,12544,864,
	144,12544,1296,
	144,3136,1296,
	192,3136,1728};

int GEMM2[30] = {
	24,65536,32,
	32,16384,144,
	32,16384,192,
	56,4096,192,
	56,4096,336,
	16,50176,32,
	24,12544,96,
	24,12544,144,
	32,3136,144,
	32,3136,192};

int GEMM3[30] = {
	144,65536,24,
	192,16384,32,
	192,16384,32,
	336,4096,56,
	336,4096,56,
	96,50176,16,
	144,12544,24,
	144, 12544, 24,
	192, 3136, 32,
	192, 3136, 32};

int main()
{
	omp_set_num_threads(NUM);
	flashgemm_set_thread_num(NUM);
	// mkl_set_num_threads(NUM);

	int loop = 10, beta = 0;
	double start, cost;
	double gflops;
	bool flag = true;

	FILE *fp;
	if ((fp = fopen("../result/bf16_3_flashgemm.txt", "w")) == NULL)
	{
		puts("Fail to open file!");
		exit(0);
	}

	// int j = 1;
	for (int j = 0; j < 10; j++)
	{
		long M1 = GEMM1[j * 3];
		long M2 = GEMM2[j * 3];
		long M3 = GEMM3[j * 3];
		long N = GEMM1[j * 3 + 1];
		long K1 = GEMM1[j * 3 + 2];
		long K2 = GEMM2[j * 3 + 2];
		long K3 = GEMM3[j * 3 + 2];

		// long M1 = 24;
		// long M2 = 12 + 8;
		// long M3 = 12 + 4;
		// long N = 32;
		// long K1 = 32;
		// long K2 = 24;
		// long K3 = 12 + 8;

		double ops = (double)(M1 * K1 + M2 * K2 + M3 * K3) * N * 1.0e-09 * 2;

		void *ptrA1, *ptrA2, *ptrA3, *ptrB;
		posix_memalign(&ptrA1, 64, M1 * K1 * sizeof(uint16_t));
		posix_memalign(&ptrA2, 64, M2 * K2 * sizeof(uint16_t));
		posix_memalign(&ptrA3, 64, M3 * K3 * sizeof(uint16_t));
		posix_memalign(&ptrB, 64, N * K1 * sizeof(uint16_t));
		uint16_t *A1 = (uint16_t *)ptrA1;
		uint16_t *A2 = (uint16_t *)ptrA2;
		uint16_t *A3 = (uint16_t *)ptrA3;
		uint16_t *B = (uint16_t *)ptrB;

		float *C = (float *)malloc(M3 * N * sizeof(float));
		float *C_MKL1_f32 = (float *)malloc(M1 * N * sizeof(float));
		float *C_MKL2_f32 = (float *)malloc(M2 * N * sizeof(float));
		float *C_MKL3_f32 = (float *)malloc(M3 * N * sizeof(float));
		uint16_t *C_MKL1_f16 = (uint16_t *)malloc(M1 * N * sizeof(uint16_t));
		uint16_t *C_MKL2_f16 = (uint16_t *)malloc(M2 * N * sizeof(uint16_t));

		random_matrix_bf16(K1, N, B);
		random_matrix_bf16(M1, K1, A1);
		random_matrix_bf16(M2, K2, A2);
		random_matrix_bf16(M3, K3, A3);

		// regular1_matrix_bf16(K1, N, B);
		// regular1_matrix_bf16(M1, K1, A1);
		// regular1_matrix_bf16(M2, K2, A2);
		// regular1_matrix_bf16(M3, K3, A3);

		// printf("\nA1:\n");
		// show_matrix_bf16(M1, K1, A1);
		// printf("\nA2:\n");
		// show_matrix_bf16(M2, K2, A2);
		// printf("\nA3:\n");
		// show_matrix_bf16(M3, K3, A3);
		// printf("\nB:\n");
		// show_matrix_bf16(K1, N, B);

		// test result
		flashgemm_multi_bf16bf16f32(C, B, N, 3, A1, M1, K1, A2, M2, K2, A3, M3, K3); // 3 is the number of GEMMs

		// cblas_gemm_bf16bf16f32(CblasRowMajor, CblasNoTrans, CblasNoTrans, M1, N, K1, 1.0, A1, K1, B, N, 0, C_MKL1_f32, N);
		// printf("\nC_MKL1_f32:\n");
		// show_matrix_fp32(M1, N, C_MKL1_f32);
		// f32_matrix_to_bf16_matrix(C_MKL1_f32, C_MKL1_f16, M1, N);
		// show_matrix_bf16(M1, N, C_MKL1_f16);

		// cblas_gemm_bf16bf16f32(CblasRowMajor, CblasNoTrans, CblasNoTrans, M2, N, K2, 1.0, A2, K1, C_MKL1_f16, N, 0, C_MKL2_f32, N);
		// printf("\nC_MKL2_f32:\n");
		// show_matrix_fp32(M2, N, C_MKL2_f32);
		// f32_matrix_to_bf16_matrix(C_MKL2_f32, C_MKL2_f16, M2, N);
		// show_matrix_bf16(M2, N, C_MKL2_f16);
		
		// cblas_gemm_bf16bf16f32(CblasRowMajor, CblasNoTrans, CblasNoTrans,
		// 											 M3, N, K3, 1.0, A3, K1, C_MKL2_f16, N, 0, C_MKL3_f32, N);

		// printf("\nC:\n");
		// show_matrix_fp32(M3, N, C);
		// printf("\nC_MKL3_f32:\n");
		// show_matrix_fp32(M3, N, C_MKL3_f32);

		// flag = Check_result(C, C_MKL3_f32, M3, N);

		// warm up
		for (int i = 0; i <= 5; i++)
		{
			flashgemm_multi_bf16bf16f32(C, B, N, 3, A1, M1, K1, A2, M2, K2, A3, M3, K3); // 3 is the number of GEMMs
		}

		// test time
		start = dclock();
		for (int i = 0; i <= loop; i++)
		{
			flashgemm_multi_bf16bf16f32(C, B, N, 3, A1, M1, K1, A2, M2, K2, A3, M3, K3); // 3 is the number of GEMMs
		}
		cost = (dclock() - start) / loop;

		if (flag)
		{
			printf("bf16:  N=%-6d M1= %-6d K1=%-6d M2= %-6d K2=%-6d M3= %-6d K3=%-6d flops = %-10.3lf effic= %.3lf %\n",
						 N, M1, K1, M2, K2, M3, K3, ops / cost, ops / cost / (PEAK_GFLOPS * 32 * 2 * 2) * 100 / NUM);
			fprintf(fp, "%.3lf\n", ops / cost);
		}
		else
		{
			printf("bf16:  N=%-6d M1= %-6d K1=%-6d M2= %-6d K2=%-6d M3= %-6d K3=%-6d error!!!\n",
						 N, M1, K1, M2, K2, M3, K3);
			fprintf(fp, "error! \n");
		}

		free(A1);
		free(A2);
		free(A3);
		free(B);
		free(C);
		free(C_MKL1_f32);
		free(C_MKL2_f32);
		free(C_MKL3_f32);
		free(C_MKL1_f16);
		free(C_MKL2_f16);
	}

	return 0;
}
