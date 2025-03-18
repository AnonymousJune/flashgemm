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
#define NUM 24

int MNK[30] = {
	32, 12544, 288,
	144, 3136, 1296,
	192, 3136, 1728,
	192, 784, 4800,
	336, 784, 8400,
	32, 16384, 288,
	96, 4096, 864,
	144, 4096, 1296,
	144, 1024, 1296,
	192, 1024, 1728,
};

int main()
{
	omp_set_num_threads(NUM);
	flashgemm_set_thread_num(NUM);
	// mkl_set_num_threads(NUM);

	int loop = 10, beta = 1;
	double start, cost;
	double gflops;

	FILE *fp;
	if ((fp = fopen("./result.txt", "w")) == NULL)
	{
		puts("Fail to open file!");
		exit(0);
	}

	// int j = 1;
	for (int j = 0; j < 10; j++)
	{
		long M = MNK[j * 3];
		long N = MNK[j * 3 + 1];
		long K = MNK[j * 3 + 2];

		// long M = 12;
		// long N = 48;
		// long K = 16;

		double ops = (double)M * N * K * 1.0e-09 * 2;

		void *ptrA, *ptrB;
		posix_memalign(&ptrA, 64, M * K * sizeof(uint16_t));
		posix_memalign(&ptrB, 64, N * K * sizeof(uint16_t));
		uint16_t *A = (uint16_t *)ptrA;
		uint16_t *B = (uint16_t *)ptrB;

		float *C = (float *)malloc(M * N * sizeof(float));
		float *C_MKL = (float *)malloc(M * N * sizeof(float));

		random_matrix_bf16(K, N, B);
		random_matrix_bf16(M, K, A);
		random_matrix_f32(M, N, C);
		memcpy(C_MKL, C, M * N * sizeof(float));

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
		flashgemm_single_bf16bf16f32(C, A, B, M, N, K, beta);
		cblas_gemm_bf16bf16f32(CblasRowMajor, CblasNoTrans, CblasNoTrans, 
											 M, N, K, 1.0, A, K, B, N, beta, C_MKL, N);
		// printf("\nC:\n");
		// show_matrix_fp32(M, N, C);
		// printf("\nC_MKL:\n");
		// show_matrix_fp32(M, N, C_MKL);
		
		bool flag = Check_result(C, C_MKL, M, N);
		
		// warm up
		for (int i = 0; i <= 5; i++){
			flashgemm_single_bf16bf16f32(C, A, B, M, N, K, beta);
		}

		// test time
		start = dclock();
		for (int i = 0; i <= loop; i++){
			flashgemm_single_bf16bf16f32(C, A, B, M, N, K, beta);
		}
		cost = (dclock() - start) / loop;

		if (flag)
		{
			printf("bf16: id=%-3d M= %-8d N=%-8d K=%-8d effic= %.3lf\n",
						 j, M, N, K, ops / cost / ((PEAK_GFLOPS * 64 * 2)) * 100 / NUM);
			fprintf(fp, "%.3lf \n", ops / cost / ((PEAK_GFLOPS * 64 * 2)) * 100 / NUM);
		}else{
			printf("bf16: id=%-3d M= %-8d N=%-8d K=%-8d error!!!\n", j, M, N, K);
			fprintf(fp, "error! \n");
		}

		free(ptrA);
		free(ptrB);
		free(C);
		free(C_MKL);
	}

	return 0;
}
