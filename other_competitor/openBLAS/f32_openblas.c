#include <stdio.h>
#include <sys/time.h>
#include <stdlib.h>
#include "cblas.h"
#include <omp.h>
#include <math.h>
#include "utils.h"

#define NUM 32
#define PEAK_GFLOPS 2.6

int GEMM[60] = {
		32, 12544, 288, // ID1-10
		144, 3136, 1296,
		192, 3136, 1728,
		192, 784, 4800,
		336, 784, 8400,
		32, 16384, 288,
		96, 4096, 864,
		144, 4096, 1296,
		144, 1024, 1296,
		192, 1024, 1728,
		128, 128, 512, // ID11-20
		256, 256, 512,
		512, 512, 512,
		1024, 1024, 512,
		2048, 2048, 512,
		128, 16384, 512,
		256, 16384, 512,
		512, 16384, 512,
		1024, 16384, 512,
		2048, 16384, 512};

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
	if ((fp = fopen("../result/f32_openblas.txt", "w")) == NULL)
	{
		puts("Fail to open file!");
		exit(0);
	}

	for (j = 0; j < 20; j++)
	{
		M = GEMM[j * 3];
		N = GEMM[j * 3 + 1];
		K = GEMM[j * 3 + 2];

		lda = K;
		ldb = N;
		ldc = N;
		double ops = (double)M * N * K * 1.0e-09 * 2;

		float *A = (float *)malloc(M * K * sizeof(float));
		float *B = (float *)malloc(K * N * sizeof(float));
		float *C = (float *)malloc(M * N * sizeof(float));

		random_matrix_f32(K, N, B);
		random_matrix_f32(M, K, A);
		random_matrix_f32(M, N, C);

		for (i = 0; i < loop; i++)
		{
			cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, N, K,
									 1.0, A, lda, B, ldb, 0.0, C, ldc);
		}

		start = dclock();
		for (i = 0; i < loop; i++)
		{
			cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, N, K,
									 1.0, A, lda, B, ldb, 0.0, C, ldc);
		}
		cost = (dclock() - start) / loop;

		printf("openblas_f32:  M= %-10d N=%-10d K=%-10d flops = %-10.3lf effic= %.3lf %\n",
					 M, N, K, ops / cost, ops / cost / (PEAK_GFLOPS * 32 * 2) * 100 / NUM);
		fprintf(fp, "%.3lf\n", ops / cost);

		free(A);
		free(B);
		free(C);
	}
	fclose(fp);

	return 0;
}