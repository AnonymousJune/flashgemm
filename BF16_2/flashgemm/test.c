#include <stdlib.h>
#include <stdio.h>
#include <cstdint>
#include <immintrin.h>
#include <string.h>
#include "mkl.h"
#include "./src/flashgemm.c"
#include "utils.h"

using namespace std;
#define PEAK_GFLOPS 2.6
#define NUM 32

int GEMM1[30] = {
		32, 12544, 288, // ID1-10
		144, 3136, 1296,
		192, 3136, 1728,
		192, 784, 4800,
		336, 784, 8400,
		32, 16384, 288,
		96, 4096, 864,
		144, 4096, 1296,
		144, 1024, 1296,
		192, 1024, 1728};

int GEMM2[30] = {
		24, 12544, 32,
		32, 3136, 144,
		32, 3136, 192,
		56, 784, 192,
		56, 784, 336,
		16, 16384, 32,
		24, 4096, 96,
		24, 4096, 144,
		32, 1024, 144,
		32, 1024, 192};

int GEMM3[30] = {
		144, 12544, 24,
		192, 3236, 32,
		192, 3136, 32,
		336, 784, 56,
		336, 784, 56,
		96, 16384, 16,
		144, 4096, 24,
		144, 4096, 24,
		192, 1024, 32,
		192, 1024, 32};

int main()
{
	omp_set_num_threads(NUM);
	flashgemm_set_thread_num(NUM);
	// mkl_set_num_threads(NUM);

	int loop = 10, beta = 0;
	double start, cost;
	double gflops;

	FILE *fp;
	if ((fp = fopen("./bf16_flashgemm.txt", "w")) == NULL)
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

		// long M = 12;
		// long N = 48;
		// long K = 16;

		double ops = (double)M * N * K * 1.0e-09 * 2;

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
		random_matrix_f32(M3, N, C);
		memcpy(C_MKL, C, M3 * N * sizeof(float));

		// regular_matrix_bf16(K, N, B);
		// regular_matrix_bf16(M, K, A);
		// regular_matrix_f32(M, N, C);
		// memcpy(C_MKL, C, M * N * sizeof(float));

		// printf("\nA:\n");
		// show_matrix_bf16(M, K, A);
		// printf("\nB:\n");
		// show_matrix_bf16(K, N, B);
		// printf("\nC init:\n");
		// show_matrix_fp32(M, N, C);

		// test result
		flashgemm_multi_bf16bf16f32_type_a(C, B, N, A1, M1, K1, A2, M2, K2, A3, M3, K3);
		
		cblas_gemm_bf16bf16f32(CblasRowMajor, CblasNoTrans, CblasNoTrans,
													 M1, N, K1, 1.0, A1, K1, B, N, 0, C_MKL1_f32, N);
		f32_matrix_to_bf16_matrix(C_MKL1_f16, C_MKL1_f32, M1, N);
		cblas_gemm_bf16bf16f32(CblasRowMajor, CblasNoTrans, CblasNoTrans,
													 M2, N, K2, 1.0, A2, K1, C_MKL1_f16, N, 0, C_MKL2_f32, N);
		f32_matrix_to_bf16_matrix(C_MKL2_f16, C_MKL2_f32, M2, N);
		cblas_gemm_bf16bf16f32(CblasRowMajor, CblasNoTrans, CblasNoTrans,
													 M3, N, K3, 1.0, A3, K1, C_MKL2_f16, N, 0, C_MKL3_f32, N);
		// show_matrix_fp32(M, N, C);
		// printf("\nC_MKL:\n");
		// show_matrix_fp32(M, N, C_MKL);

		// bool flag = Check_result(C, C_MKL, M, N);

		// warm up
		// for (int i = 0; i <= 5; i++)
		// {
		// 	flashgemm_single_bf16bf16f32(C, A, B, M, N, K, beta);
		// }

		// // test time
		// start = dclock();
		// for (int i = 0; i <= loop; i++)
		// {
		// 	flashgemm_single_bf16bf16f32(C, A, B, M, N, K, beta);
		// }
		// cost = (dclock() - start) / loop;

		// if (flag)
		// {
		printf("bf16:  M= %-10d N=%-10d K=%-10d flops = %-10.3lf effic= %.3lf %\n",
					 M, N, K, ops / cost, ops / cost / (PEAK_GFLOPS * 32 * 2 * 2) * 100 / NUM);
		fprintf(fp, "%.3lf\n", ops / cost / (PEAK_GFLOPS * 32 * 2 * 2) * 100 / NUM);
		// }
		// else
		// {
		// 	printf("bf16: id=%-3d M= %-8d N=%-8d K=%-8d error!!!\n", j, M, N, K);
		// 	fprintf(fp, "error! \n");
		// }

		free(ptrA);
		free(ptrB);
		free(C);
		free(C_MKL);
	}

	return 0;
}
