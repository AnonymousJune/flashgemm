#include <stdlib.h>
#include <stdio.h>
#include <cstdint>
#include <immintrin.h>
#include <string.h>
#include "mkl.h"
#include "../src/flashgemm.c"
// #include "utils.h"

using namespace std;
#define PEAK_GFLOPS 2.6
#define NUM 1

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
		192, 3136, 32,
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
	if ((fp = fopen("../result/f32_3_flashgemm.txt", "w")) == NULL)
	{
		puts("Fail to open file!");
		exit(0);
	}

	int j = 1;
	// for (int j = 0; j < 10; j++)
	{
		// long M1 = GEMM1[j * 3];
		// long M2 = GEMM2[j * 3];
		// long M3 = GEMM3[j * 3];
		// long N = GEMM1[j * 3 + 1];
		// long K1 = GEMM1[j * 3 + 2];
		// long K2 = GEMM2[j * 3 + 2];
		// long K3 = GEMM3[j * 3 + 2];

		long M1 = 24;
		long M2 = 12 + 8;
		long M3 = 12 + 4;
		long N = 32;
		long K1 = 32;
		long K2 = 24;
		long K3 = 12 + 8;

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

		float *C = (float *)malloc(M3 * N * sizeof(float));
		float *C_MKL1_f32 = (float *)malloc(M1 * N * sizeof(float));
		float *C_MKL2_f32 = (float *)malloc(M2 * N * sizeof(float));
		float *C_MKL3_f32 = (float *)malloc(M3 * N * sizeof(float));

		// random_matrix_f32(K1, N, B);
		// random_matrix_f32(M1, K1, A1);
		// random_matrix_f32(M2, K2, A2);
		// random_matrix_f32(M3, K3, A3);

		regular_matrix_f32(K1, N, B);
		regular_matrix_f32(M1, K1, A1);
		regular_matrix_f32(M2, K2, A2);
		regular_matrix_f32(M3, K3, A3);

		printf("\nA1:\n");
		show_matrix_fp32(M1, K1, A1);
		printf("\nA2:\n");
		show_matrix_fp32(M2, K2, A2);
		printf("\nA3:\n");
		show_matrix_fp32(M3, K3, A3);
		printf("\nB:\n");
		show_matrix_fp32(K1, N, B);

		cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M1, N, K1, 1, A1, K1, B, N, 0, C_MKL1_f32, N);
		cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M2, N, K2, 1, A2, K2, C_MKL1_f32, N, 0, C_MKL2_f32, N);
		cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M3, N, K3, 1, A3, K3, C_MKL2_f32, N, 0, C_MKL3_f32, N);

		// test result
		flashgemm_multi_f32f32f32(C, B, N, 3, A1, M1, K1, A2, M2, K2, A3, M3, K3); // 3 is the number of GEMMs
		
		printf("\nC_MKL1_f32:\n");
		show_matrix_fp32(M1, N, C_MKL1_f32);

		printf("\nC_MKL2_f32:\n");
		show_matrix_fp32(M2, N, C_MKL2_f32);

		printf("\nC:\n");
		show_matrix_fp32(M3, N, C);

		printf("\nC_MKL3_f32:\n");
		show_matrix_fp32(M3, N, C_MKL3_f32);

		flag = Check_result(C, C_MKL3_f32, M3, N);

		// // warm up
		// for (int i = 0; i <= 5; i++)
		// {
		// 	flashgemm_multi_f32f32f32(C, B, N, 3, A1, M1, K1, A2, M2, K2, A3, M3, K3); // 3 is the number of GEMMs
		// }

		// // test time
		// start = dclock();
		// for (int i = 0; i <= loop; i++)
		// {
		// 	flashgemm_multi_f32f32f32(C, B, N, 3, A1, M1, K1, A2, M2, K2, A3, M3, K3); // 3 is the number of GEMMs
		// }
		// cost = (dclock() - start) / loop;

		if (flag)
		{
			printf("f32:  N=%-10d M1= %-10d K1=%-10d M2= %-10d K2=%-10d M3= %-10d K3=%-10d flops = %-10.3lf effic= %.3lf %\n",
						 N, M1, K1, M2, K2, M3, K3, ops / cost, ops / cost / (PEAK_GFLOPS * 32 * 2 * 2) * 100 / NUM);
		}
		else
		{
			printf("f32:  N=%-10d M1= %-10d K1=%-10d M2= %-10d K2=%-10d M3= %-10d K3=%-10d error!!!\n",
						 N, M1, K1, M2, K2, M3, K3);
		}

		free(A1);
		free(A2);
		free(A3);
		free(B);
		free(C);
		free(C_MKL1_f32);
		free(C_MKL2_f32);
		free(C_MKL3_f32);
	}

	return 0;
}
