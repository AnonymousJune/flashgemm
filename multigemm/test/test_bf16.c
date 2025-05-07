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

int GEMM[60] = {
	32,65536,288,24,65536,32,
144,16384,1296,32,16384,144,
192,16384,1728,32,16384,192, 
192,4096,4800,56,4096,192,
336,4096,8400,56,4096,336,
32,50176,288,16,50176,32,
96,12544,864,24,12544,96,
144,12544,1296,24,12544,144,
144,3136,1296,32,3136,144,
192,3136,1728,32,3136,192};

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

		printf("bf16:  M= %-6d N=%-6d K=%-6d flops = %-10.3lf effic= %.3lf %\n",
					 M, N, K, ops / cost, ops / cost / (PEAK_GFLOPS * 32 * 2 * 2) * 100 / NUM);
		fprintf(fp, "%.3lf\n", ops / cost);

		free(ptrA);
		free(ptrB);
		free(C);
	}

	return 0;
}
