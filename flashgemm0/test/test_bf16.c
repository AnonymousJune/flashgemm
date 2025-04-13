#include <stdlib.h>
#include <stdio.h>
#include <cstdint>
#include <immintrin.h>
#include <string.h>
// #include "mkl.h"
#include "../src/flashgemm.c"
// #include "utils.h"

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

	int loop = 10, beta = 1;
	double start, cost;
	double gflops;

	FILE *fp;
	if ((fp = fopen("../result/bf16_flashgemm.txt", "w")) == NULL)
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

		double ops = (double)M * N * K * 1.0e-09 * 2;

		void *ptrA, *ptrB;
		posix_memalign(&ptrA, 64, M * K * sizeof(uint16_t));
		posix_memalign(&ptrB, 64, N * K * sizeof(uint16_t));
		uint16_t *A = (uint16_t *)ptrA;
		uint16_t *B = (uint16_t *)ptrB;

		float *C = (float *)malloc(M * N * sizeof(float));

		random_matrix_bf16(K, N, B);
		random_matrix_bf16(M, K, A);

		// warm up
		for (int i = 0; i <= 5; i++)
		{
			flashgemm_multi_bf16bf16f32(C, B, N, 1, A, M, K);
		}

		// test time
		start = dclock();
		for (int i = 0; i <= loop; i++)
		{
			flashgemm_multi_bf16bf16f32(C, B, N, 1, A, M, K);
		}
		cost = (dclock() - start) / loop;

		printf("bf16:  M= %-10d N=%-10d K=%-10d flops = %-10.3lf effic= %.3lf %\n",
					 M, N, K, ops / cost, ops / cost / (PEAK_GFLOPS * 32 * 2 * 2) * 100 / NUM);
		fprintf(fp, "%.3lf\n", ops / cost);

		free(ptrA);
		free(ptrB);
		free(C);
	}

	return 0;
}
