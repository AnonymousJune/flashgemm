#include <stdio.h>
#include <sys/time.h>
#include <stdlib.h>
#include "mkl.h"
#include <omp.h>
#include <math.h>
#include "utils.h"

#define NUM 32
#define PEAK_GFLOPS 2.6

int MNK[60] = {
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
		2048, 16384, 512
};

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
	if ((fp = fopen("../result/bf16_mkl.txt", "w")) == NULL)
	{
		puts("Fail to open file!");
		exit(0);
	}

	for (j = 0; j < 20; j++)
	{
		M = MNK[j * 3];
		N = MNK[j * 3 + 1];
		K = MNK[j * 3 + 2];

		lda = K;
		ldb = N;
		ldc = N;
		sizea = M * K;
		sizeb = K * N;
		sizec = M * N;
		double ops = (double)M * N * K * 1.0e-09 * 2;

		uint16_t *A = (uint16_t *)malloc(M * K * sizeof(uint16_t));
		uint16_t *B = (uint16_t *)malloc(K * N * sizeof(uint16_t));
		float *C = (float *)malloc(M * N * sizeof(float));

		random_matrix_bf16(K, N, B);
		random_matrix_bf16(M, K, A);
		random_matrix_f32(M, N, C);

		for (i = 0; i < 3; i++)
		{
			cblas_gemm_bf16bf16f32(CblasRowMajor, CblasNoTrans, CblasNoTrans,
														 M, N, K, alpha, A, K, B, N, beta, C, N);
		}

		start = dclock();
		for (i = 0; i < loop; i++)
		{
			cblas_gemm_bf16bf16f32(CblasRowMajor, CblasNoTrans, CblasNoTrans,
														 M, N, K, alpha, A, K, B, N, beta, C, N);
		}
		cost = (dclock() - start) / loop;

		printf("bf16_mkl:  M= %-10d N=%-10d K=%-10d flops = %-10.3lf effic= %.3lf %\n",
					 M, N, K, ops / cost, ops / cost / (PEAK_GFLOPS * 32 * 2 * 2) * 100 / NUM);
		fprintf(fp, "%.3lf\n", ops / cost);

		free(A);
		free(B);
		free(C);
	}
	fclose(fp);

	return 0;
}
