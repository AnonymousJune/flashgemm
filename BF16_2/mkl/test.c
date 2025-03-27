#include <stdio.h>
#include <sys/time.h>
#include <stdlib.h>
#include "mkl.h"
#include <omp.h>
#include <math.h>
#include "utils.h"

#define NUM 32
#define PEAK_GFLOPS 2.6

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
	mkl_set_num_threads(NUM);
	omp_set_num_threads(NUM);

	int i, j, k, jj, pc;
	int loop = 10;
	long M, N, K;
	double start, cost;
	double gflops;
	long lda, ldb, ldc;
	int flag = 0;
	float alpha = 1.0, beta = 0.0;
	int sizea, sizeb, sizec;

	FILE *fp;
	if ((fp = fopen("./bf16_mkl.txt", "w")) == NULL)
	{
		puts("Fail to open file!");
		exit(0);
	}

	for (j = 0; j < 10; j++)
	{
		long M1 = GEMM1[j * 3];
		long M2 = GEMM2[j * 3];
		long M3 = GEMM3[j * 3];
		long N = GEMM1[j * 3 + 1];
		long K1 = GEMM1[j * 3 + 2];
		long K2 = GEMM2[j * 3 + 2];
		long K3 = GEMM3[j * 3 + 2];

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

		float *C_MKL1_f32 = (float *)malloc(M1 * N * sizeof(float));
		float *C_MKL2_f32 = (float *)malloc(M2 * N * sizeof(float));
		float *C_MKL3_f32 = (float *)malloc(M3 * N * sizeof(float));
		uint16_t *C_MKL1_f16 = (uint16_t *)malloc(M1 * N * sizeof(uint16_t));
		uint16_t *C_MKL2_f16 = (uint16_t *)malloc(M2 * N * sizeof(uint16_t));

		random_matrix_bf16(K1, N, B);
		random_matrix_bf16(M1, K1, A1);
		random_matrix_bf16(M2, K2, A2);
		random_matrix_bf16(M3, K3, A3);

		for (i = 0; i < 3; i++)
		{
			// GEMM 1
			cblas_gemm_bf16bf16f32(CblasRowMajor, CblasNoTrans, CblasNoTrans, M1, N, K1, 1.0, A1, K1, B, N, 0, C_MKL1_f32, N);
			// f32_matrix_to_bf16_matrix(C_MKL1_f32, C_MKL1_f16, M1, N);
			// GEMM 2
			cblas_gemm_bf16bf16f32(CblasRowMajor, CblasNoTrans, CblasNoTrans, M2, N, K2, 1.0, A2, K1, C_MKL1_f16, N, 0, C_MKL2_f32, N);
			// f32_matrix_to_bf16_matrix(C_MKL2_f32, C_MKL2_f16, M2, N);
			// GEMM 3
			cblas_gemm_bf16bf16f32(CblasRowMajor, CblasNoTrans, CblasNoTrans, M3, N, K3, 1.0, A3, K1, C_MKL2_f16, N, 0, C_MKL3_f32, N);
		}

		start = dclock();
		for (i = 0; i < loop; i++)
		{
			// GEMM 1
			cblas_gemm_bf16bf16f32(CblasRowMajor, CblasNoTrans, CblasNoTrans, M1, N, K1, 1.0, A1, K1, B, N, 0, C_MKL1_f32, N);
			// f32_matrix_to_bf16_matrix(C_MKL1_f32, C_MKL1_f16, M1, N);
			// GEMM 2
			cblas_gemm_bf16bf16f32(CblasRowMajor, CblasNoTrans, CblasNoTrans, M2, N, K2, 1.0, A2, K1, C_MKL1_f16, N, 0, C_MKL2_f32, N);
			// f32_matrix_to_bf16_matrix(C_MKL2_f32, C_MKL2_f16, M2, N);
			// GEMM 3
			cblas_gemm_bf16bf16f32(CblasRowMajor, CblasNoTrans, CblasNoTrans, M3, N, K3, 1.0, A3, K1, C_MKL2_f16, N, 0, C_MKL3_f32, N);
		}
		cost = (dclock() - start) / loop;

		printf("mkl_bf16:  N=%-10d M1= %-10d K1=%-10d M2= %-10d K2=%-10d M3= %-10d K3=%-10d flops = %-10.3lf effic= %.3lf %\n",
					 N, M1, K1, M2, K2, M3, K3, ops / cost, ops / cost / (PEAK_GFLOPS * 32 * 2 * 2) * 100 / NUM);
		fprintf(fp, "%.3lf\n", ops / cost / (PEAK_GFLOPS * 32 * 2 * 2) * 100 / NUM);

		free(A1);
		free(A2);
		free(A3);
		free(B);
		free(C_MKL1_f32);
		free(C_MKL2_f32);
		free(C_MKL3_f32);
		free(C_MKL1_f16);
		free(C_MKL2_f16);
	}
	fclose(fp);

	return 0;
}
