#include <stdio.h>
#include <sys/time.h>
#include <stdlib.h>
#include "mkl.h"
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
	const MKL_INT8 ao = 0;
	const MKL_INT8 bo = 0;
	MKL_INT32 co = 0;
	CBLAS_OFFSET offsetc = CblasFixOffset;

	FILE *fp;
	if ((fp = fopen("../result/int8_mkl.txt", "w")) == NULL)
	{
		puts("Fail to open file!");
		exit(0);
	}

	for (j = 0; j < 20; j++)
	{
		long M = GEMM[j * 3];
		long N = GEMM[j * 3 + 1];
		long K = GEMM[j * 3 + 2];

		double ops = (double)(M * K) * N * 1.0e-09 * 2;

		void *ptrA, *ptrB;
		posix_memalign(&ptrA, 64, M * K * sizeof(int8_t));
		posix_memalign(&ptrB, 64, N * K * sizeof(uint8_t));
		int8_t *A = (int8_t *)ptrA;
		uint8_t *B = (uint8_t *)ptrB;
		int *C = (int *)malloc(M * N * sizeof(int));

		random_matrix_uint8(K, N, B);
		random_matrix_int8(M, K, A);

		for (i = 0; i < 5; i++)
		{
			cblas_gemm_s8u8s32(CblasRowMajor, CblasNoTrans, CblasNoTrans, offsetc, M, N, K, 1.0, A, K, ao, B, N, bo, 0, C, N, &co);
		}

		start = dclock();
		for (i = 0; i < loop; i++)
		{
			cblas_gemm_s8u8s32(CblasRowMajor, CblasNoTrans, CblasNoTrans, offsetc, M, N, K, 1.0, A, K, ao, B, N, bo, 0, C, N, &co);
		}
		cost = (dclock() - start) / loop;

		printf("int8_mkl:  N=%-10d M= %-10d K=%-10d flops=%-10.3lf effic= %.3lf %\n", N, M, K, ops / cost, ops / cost / (PEAK_GFLOPS * 32 * 4 * 2) * 100 / NUM);
		fprintf(fp, "%.3lf \n", ops / cost);

		free(A);
		free(B);
		free(C);
	}
	fclose(fp);

	return 0;
}
