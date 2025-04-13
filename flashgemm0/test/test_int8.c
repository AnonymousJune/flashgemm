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
	omp_set_num_threads(NUM);
	flashgemm_set_thread_num(NUM);
	// mkl_set_num_threads(NUM);

	int loop = 10, beta = 0;
	double start, cost;
	double gflops;
	bool flag = true;

	FILE *fp;
	if ((fp = fopen("../result/int8_flashgemm.txt", "w")) == NULL)
	{
		puts("Fail to open file!");
		exit(0);
	}

	// int j = 1;
	for (int j = 0; j < 20; j++)
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

		random_matrix_int8(M, K, A);
		random_matrix_uint8(K, N, B);

		// warm up
		for (int i = 0; i <= 5; i++)
		{
			flashgemm_multi_int8uint8int32(C, B, N, 1, A, M, K); // 1 is the number of GEMMs
		}

		// test time
		start = dclock();
		for (int i = 0; i <= loop; i++)
		{
			flashgemm_multi_int8uint8int32(C, B, N, 1, A, M, K); // 1 is the number of GEMMs
		}
		cost = (dclock() - start) / loop;

		if (flag)
		{
			printf("int8:  N=%-10d M= %-10d K=%-10d gflops = %-10.3lf effic= %.3lf %\n", N, M, K, ops / cost, ops / cost / (PEAK_GFLOPS * 32 * 4 * 2) * 100 / NUM);
			fprintf(fp, "%.3lf \n", ops / cost);
		}
		else
		{
			printf("int8:  N=%-10d M= %-10d K=%-10d error!!!\n", N, M, K);
			fprintf(fp, "error! \n");
		}

		free(A);
		free(B);
		free(C);
	}

	return 0;
}
