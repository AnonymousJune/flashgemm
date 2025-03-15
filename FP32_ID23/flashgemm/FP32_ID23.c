#include "./src/NN_mp.h"
#include "./src/utils.h"
#include "./src/set_TmTn.h"
#include <sys/time.h>
#include <stdlib.h>
#include <cblas.h>

#define NUM 24
#define PEAK_GFLOPS 3.8

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

void LibShalom_sgemm_mp(int transa, int transb, float *C, float *A,
						float *B, long M, long N, long K)

{
	if (/*(transa == 0) &&*/ (transb == 1)) // NT mode
	{
		Dete_grad_N_threads_nums(NUM, M, N, transa, transb);
		// SGEMM_NT_mp(C, A, B, M ,N , K);
	}
	else if (/*(transa == 0) &&*/ (transb == 0)) // NN mode
	{

		// M >> N
		if ((M / N) >= 8)
		{
			// Dete_grad_M_threads_nums(NUM, M, N, transa, transb);
			// TODO
			Tm = NUM;
			Tn = NUM / Tm;
			Small_MGN_NN_SGEMM(C, A, B, M, N, K, NUM, Tm, Tn);
		}
		else
		{
			Dete_grad_N_threads_nums(NUM, M, N, transa, transb);
			Small_NGM_NN_SGEMM(C, A, B, M, N, K, NUM, Tm, Tn);
		}
	}
}

int main()
{

	openblas_set_num_threads(NUM);
	LibShalom_set_thread_nums(NUM);

	int i, j, k, jj, pc;
	int loop = 10;
	long M, N, K;
	double start, cost = 1;
	double gflops;
	long lda, ldb, ldc;
	int flag = 0;

	FILE *fp;
	if ((fp = fopen("../results/flashgemm.txt", "w")) == NULL)
	{
		puts("Fail to open file!");
		exit(0);
	}

	// j = 4 - 1;
	for (j = 0; j < 23; j++)
	// for (j = 0; j < 33; j++)
	{
		M = MNK[j * 3];
		N = MNK[j * 3 + 1];
		K = MNK[j * 3 + 2];

		// M = 12 * j + 24;
		// N = 7680;
		// K = 9600;

		lda = K;
		ldb = N;
		ldc = N;
		double ops = (double)M * N * K * 1.0e-09 * 2;

		float *A = (float *)malloc(M * K * sizeof(float));
		float *B = (float *)malloc(K * N * sizeof(float));
		float *Bc = (float *)malloc(K * N * sizeof(float));
		float *C = (float *)malloc(M * N * sizeof(float));
		float *C1 = (float *)malloc(M * N * sizeof(float));

		random_matrix(M, K, A);
		random_matrix(K, N, B);

		// LibShalom_sgemm_mp(0, 0, C, A, B, M, N, K);

		for (i = 0; i < 10; i++)
		{
			LibShalom_sgemm_mp(0, 0, C, A, B, M, N, K);
		}

		start = dclock();
		for (i = 0; i < loop; i++)
		{
			LibShalom_sgemm_mp(0, 0, C, A, B, M, N, K);
		}
		cost = (dclock() - start) / loop;

		cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, N, K,
					1.0, A, lda, B, ldb, 0.0, C1, ldc);

		if (Check_result(C, C1, M, N))
		{
			printf("multicore: id=%-2d M= %-10d N=%-10d K=%-10d effic= %.3lf \n",
				   j + 1, M, N, K, ops / cost / (PEAK_GFLOPS*64) * 100 / NUM  );
			fprintf(fp, "%.3lf \n", ops / cost / (PEAK_GFLOPS*64) * 100 / NUM  );
		}
		else
		{
			printf("multicore: id=%-2d M= %-10d N=%-10d K=%-10d effic= %.3lf error!\n",
				   j + 1, M, N, K, ops / cost / (PEAK_GFLOPS*64) * 100 / NUM  );
			fprintf(fp, "%.3lf\n", ops / cost / (PEAK_GFLOPS*64) * 100 / NUM  );
		}

		free(A);
		free(B);
		free(C);
		free(Bc);
		free(C1);
	}
	fclose(fp);

	return 0;
}