#include <stdio.h>
#include <sys/time.h>
#include <stdlib.h>
#include "cblas.h"
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
		192, 3136 , 32,
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
	openblas_set_num_threads(NUM);
	omp_set_num_threads(NUM);

	int i, j, k, jj, pc;
	int loop = 10;
	long M, N, K;
	double start, cost;
	double gflops;
	long lda, ldb, ldc;
	int flag = 0;

	FILE *fp;
	if ((fp = fopen("../result/f32_3_openblas.txt", "w")) == NULL)
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
		posix_memalign(&ptrA1, 64, M1 * K1 * sizeof(float));
		posix_memalign(&ptrA2, 64, M2 * K2 * sizeof(float));
		posix_memalign(&ptrA3, 64, M3 * K3 * sizeof(float));
		posix_memalign(&ptrB, 64, N * K1 * sizeof(float));
		float *A1 = (float *)ptrA1;
		float *A2 = (float *)ptrA2;
		float *A3 = (float *)ptrA3;
		float *B = (float *)ptrB;

		float *C1_f32 = (float *)malloc(M1 * N * sizeof(float));
		float *C2_f32 = (float *)malloc(M2 * N * sizeof(float));
		float *C3_f32 = (float *)malloc(M3 * N * sizeof(float));
		float *C1_f16 = (float *)malloc(M1 * N * sizeof(float));
		float *C2_f16 = (float *)malloc(M2 * N * sizeof(float));

		random_matrix_f32(K1, N, B);
		random_matrix_f32(M1, K1, A1);
		random_matrix_f32(M2, K2, A2);
		random_matrix_f32(M3, K3, A3);

		for (i = 0; i < loop; i++)
		{
			cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M1, N, K1,
									 1.0, A1, K1, B, N, 0.0, C1_f32, N);
			// f32_matrix_to_f32_matrix(C1_f32, C1_f16, M1, N);
			cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M2, N, K2,
									 1.0, A2, K2, C1_f16, N, 0.0, C2_f32, N);
			// f32_matrix_to_f32_matrix(C2_f32, C2_f16, M1, N);
			cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M3, N, K3,
									 1.0, A3, K3, C2_f16, N, 0.0, C3_f32, N);
		}

		start = dclock();
		for (i = 0; i < loop; i++)
		{
			cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M1, N, K1,
									 1.0, A1, K1, B, N, 0.0, C1_f32, N);
			// f32_matrix_to_f32_matrix(C1_f32, C1_f16, M1, N);
			cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M2, N, K2,
									 1.0, A2, K2, C1_f16, N, 0.0, C2_f32, N);
			// f32_matrix_to_f32_matrix(C2_f32, C2_f16, M1, N);
			cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M3, N, K3,
									 1.0, A3, K3, C2_f16, N, 0.0, C3_f32, N);
		}
		cost = (dclock() - start) / loop;

		printf("openblas_f32_3:  N=%-10d M1= %-10d K1=%-10d M2= %-10d K2=%-10d M3= %-10d K3=%-10d flops = %-10.3lf effic= %.3lf %\n",
					 N, M1, K1, M2, K2, M3, K3, ops / cost, ops / cost / (PEAK_GFLOPS * 32 * 2) * 100 / NUM);
		fprintf(fp, "%.3lf\n", ops / cost);

		free(A1);
		free(A2);
		free(A3);
		free(B);
		free(C1_f32);
		free(C2_f32);
		free(C3_f32);
		free(C1_f16);
		free(C2_f16);
	}
	fclose(fp);

	return 0;
}