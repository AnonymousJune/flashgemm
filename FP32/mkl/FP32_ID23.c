// 杨维铃 2023.10.25

#include <stdio.h>
#include <sys/time.h>
#include <stdlib.h>
#include "mkl.h"
#include <omp.h>
// #include <cblas.h>
#include <math.h>

#define NUM 24
#define PEAK_GFLOPS 3.8
static double gtod_ref_time_sec = 0.0;

double dclock()
{
	double the_time, norm_sec;
	struct timeval tv;

	gettimeofday(&tv, NULL);

	if (gtod_ref_time_sec == 0.0)
		gtod_ref_time_sec = (double)tv.tv_sec;

	norm_sec = (double)tv.tv_sec - gtod_ref_time_sec;

	the_time = norm_sec + tv.tv_usec * 1.0e-6;

	return the_time;
}

void random_matrix(int m, int n, float *a)
{
	double drand48();
	int i, j;

	for (i = 0; i < m; i++)
		for (j = 0; j < n; j++)
			a[i * n + j] = (float)drand48() - 0.5;
}

long MNK[69] = {128, 2048, 4096,
								320, 3072, 4096,
								1632, 36548, 1024,
								2048, 4096, 32,
								31999, 1024, 80,
								84, 1024, 4096,
								256, 256, 2048,
								1024, 16, 500000,
								35, 8457, 2560,
								64, 18496, 128,
								128, 4624, 256,
								256, 1156, 512,
								512, 289, 1024,
								256, 3136, 64,
								64, 3136, 64,
								64, 3136, 256,
								512, 784, 128,
								128, 784, 512,
								32, 16384, 288,
								144, 4096, 1296,
								144, 1024, 1296,
								192, 1024, 1728,
								96, 4096, 864};

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
	if ((fp = fopen("../result/results/mkl.txt", "w")) == NULL)
	{
		puts("Fail to open file!");
		exit(0);
	}

	for (j = 0; j < 23; j++)
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

		float *A = (float *)malloc(M * K * sizeof(float));
		float *B = (float *)malloc(K * N * sizeof(float));
		float *C = (float *)malloc(M * N * sizeof(float));
		// float *C1 = ( float * ) malloc( M * N * sizeof( float ) );

		random_matrix(M, K, A);
		random_matrix(K, N, B);

		for (i = 0; i < loop; i++)
		{
			cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
									M, N, K, alpha, A, K, B, N, beta, C, N);
		}

		start = dclock();
		for (i = 0; i < loop; i++)
		{
			cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
									M, N, K, alpha, A, K, B, N, beta, C, N);
		}
		cost = (dclock() - start) / loop;

		printf("MKL:  M= %-10d N=%-10d K=%-10d flops = %-10.3lf effic= %.3lf %\n",
					 M, N, K, ops / cost, ops / cost / (PEAK_GFLOPS * 64) * 100 / NUM);
		fprintf(fp, "%.3lf \n", ops / cost);

		free(A);
		free(B);
		free(C);
	}
	fclose(fp);

	return 0;
}
