#include <stdlib.h>
#include <stdio.h>
#include <cstdint>
#include <immintrin.h>
#include <string.h>
// #include "mkl.h"
#include "../src/flashgemm.c"

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
		192, , 32,
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
	bool flag = true;

	FILE *fp;
	if ((fp = fopen("../result/int8_3_flashgemm.txt", "w")) == NULL)
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
		posix_memalign(&ptrA1, 64, M1 * K1 * sizeof(int8_t));
		posix_memalign(&ptrA2, 64, M2 * K2 * sizeof(int8_t));
		posix_memalign(&ptrA3, 64, M3 * K3 * sizeof(int8_t));
		posix_memalign(&ptrB, 64, N * K1 * sizeof(uint8_t));
		int8_t *A1 = (int8_t *)ptrA1;
		int8_t *A2 = (int8_t *)ptrA2;
		int8_t *A3 = (int8_t *)ptrA3;
		uint8_t *B = (uint8_t *)ptrB;

		int *C = (int *)malloc(M3 * N * sizeof(int));
		int *C_MKL1_int32 = (int *)malloc(M1 * N * sizeof(int));
		int *C_MKL2_int32 = (int *)malloc(M2 * N * sizeof(int));
		int *C_MKL3_int32 = (int *)malloc(M3 * N * sizeof(int));
		uint8_t *C_MKL1_int8 = (uint8_t *)malloc(M1 * N * sizeof(uint8_t));
		uint8_t *C_MKL2_int8 = (uint8_t *)malloc(M2 * N * sizeof(uint8_t));

		random_matrix_int8(M1, K1, A1);
		random_matrix_int8(M2, K2, A2);
		random_matrix_int8(M3, K3, A3);
		random_matrix_uint8(K1, N, B);

		// regular1_matrix_int8(K1, N, B);
		// regular1_matrix_int8(M1, K1, A1);
		// regular1_matrix_int8(M2, K2, A2);
		// regular1_matrix_int8(M3, K3, A3);

		// printf("\nA1:\n");
		// show_matrix_int8(M1, K1, A1);
		// printf("\nA2:\n");
		// show_matrix_int8(M2, K2, A2);
		// printf("\nA3:\n");
		// show_matrix_int8(M3, K3, A3);
		// printf("\nB:\n");
		// show_matrix_int8(K1, N, B);

		// test result
		flashgemm_multi_int8uint8int32(C, B, N, 3, A1, M1, K1, A2, M2, K2, A3, M3, K3); // 3 is the number of GEMMs

		// cblas_gemm_int8int8int32(CblasRowMajor, CblasNoTrans, CblasNoTrans, M1, N, K1, 1.0, A1, K1, B, N, 0, C_MKL1_int32, N);
		// printf("\nC_MKL1_int32:\n");
		// show_matrix_fp32(M1, N, C_MKL1_int32);
		// int32_matrix_to_int8_matrix(C_MKL1_int32, C_MKL1_int8, M1, N);
		// show_matrix_int8(M1, N, C_MKL1_int8);

		// cblas_gemm_int8int8int32(CblasRowMajor, CblasNoTrans, CblasNoTrans, M2, N, K2, 1.0, A2, K1, C_MKL1_int8, N, 0, C_MKL2_int32, N);
		// printf("\nC_MKL2_int32:\n");
		// show_matrix_fp32(M2, N, C_MKL2_int32);
		// int32_matrix_to_int8_matrix(C_MKL2_int32, C_MKL2_int8, M2, N);
		// show_matrix_int8(M2, N, C_MKL2_int8);

		// cblas_gemm_int8int8int32(CblasRowMajor, CblasNoTrans, CblasNoTrans,
		// 											 M3, N, K3, 1.0, A3, K1, C_MKL2_int8, N, 0, C_MKL3_int32, N);

		// printf("\nC:\n");
		// show_matrix_fp32(M3, N, C);
		// printf("\nC_MKL3_int32:\n");
		// show_matrix_fp32(M3, N, C_MKL3_int32);

		// flag = Check_result(C, C_MKL3_int32, M3, N);

		// warm up
		for (int i = 0; i <= 5; i++)
		{
			flashgemm_multi_int8uint8int32(C, B, N, 3, A1, M1, K1, A2, M2, K2, A3, M3, K3); // 3 is the number of GEMMs
		}

		// test time
		start = dclock();
		for (int i = 0; i <= loop; i++)
		{
			flashgemm_multi_int8uint8int32(C, B, N, 3, A1, M1, K1, A2, M2, K2, A3, M3, K3); // 3 is the number of GEMMs
		}
		cost = (dclock() - start) / loop;

		if (flag)
		{
			printf("int8:  N=%-10d M1= %-10d K1=%-10d M2= %-10d K2=%-10d M3= %-10d K3=%-10d flops = %-10.3lf effic= %.3lf %\n",
						 N, M1, K1, M2, K2, M3, K3, ops / cost, ops / cost / (PEAK_GFLOPS * 32 * 4 * 2) * 100 / NUM);
			fprintf(fp, "%.3lf \n", ops / cost);
		}
		else
		{
			printf("int8:  N=%-10d M1= %-10d K1=%-10d M2= %-10d K2=%-10d M3= %-10d K3=%-10d error!!!\n",
						 N, M1, K1, M2, K2, M3, K3);
			fprintf(fp, "error! \n");
		}

		free(A1);
		free(A2);
		free(A3);
		free(B);
		free(C);
		free(C_MKL1_int32);
		free(C_MKL2_int32);
		free(C_MKL3_int32);
		free(C_MKL1_int8);
		free(C_MKL2_int8);
	}

	return 0;
}
