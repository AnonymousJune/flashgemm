#include "./flashgemm.h"

int flashgemm_thread_num = 1;

void flashgemm_set_thread_num(int num)
{
	flashgemm_thread_num = num;
}

bool flashgemm_test_dimention(int M, int N, int K)
{
	bool flag = true;
	if (M % 4 != 0)
	{
		printf("M must be a multiple of 4\n");
		flag = false;
	}
	if (N % 32 != 0)
	{
		printf("N must be a multiple of 32\n");
		flag = false;
	}
	if (K % 32 != 0)
	{
		printf("K must be a multiple of 32\n");
		flag = false;
	}
	return flag;
}

// beta = 0/1, alpha = 1
void flashgemm_single_bf16bf16f32(float *C, uint16_t *A_bf16, uint16_t *B_bf16, long M, long N, long K, int beta)
{
	if(!flashgemm_test_dimention(M, N, K)){
		printf("dimention error\n");
		return;
	}
	if(N > M){
		flashgemm_single_bf16bf16f32_MlN(C, A_bf16, B_bf16, M, N, K, beta);
	}
	else{
		flashgemm_single_bf16bf16f32_MgN(C, A_bf16, B_bf16, M, N, K, beta);
	}
}

// M << N
void flashgemm_single_bf16bf16f32_MlN(float *C, uint16_t *A_bf16, uint16_t *B_bf16, long M, long N, long K, int beta)
{
	int NUM = flashgemm_thread_num;
	long Nb = N / NUM;
	float *A = (float *)A_bf16;
	void *ptrA, *ptrB;
	posix_memalign(&ptrA, 64, M * K * sizeof(uint16_t));
	posix_memalign(&ptrB, 64, NUM * GEMM_K * 32 * sizeof(uint16_t));
	float *Ac = (float *)ptrA;
	uint16_t *Bc = (uint16_t *)ptrB;

	int Num_K_block = K / GEMM_K;
	int Edge_K = (K % GEMM_K) / 2;
	int Num_M_block = M / 12;
	int Edge_M = M % 12;
	int Num_blocks0 = Num_K_block * Num_M_block;
	if (Edge_M > 0)
	{
		Num_blocks0 = Num_blocks0 + Num_K_block;
		Num_M_block = Num_M_block + 1;
	}
	int Num_blocks = Num_blocks0;
	if (Edge_K > 0)
	{
		Num_blocks = Num_blocks0 + Num_M_block;
	}

#pragma omp parallel num_threads(NUM)
	{
		long i, j, k, ii, jj, kk, mc, nc, kc, mr, nr;
		int size_block_m;
		int id = omp_get_thread_num();
		uint16_t *temp_Bc = Bc + id * GEMM_K * 32;

		for (i = id; i < Num_blocks; i = i + NUM)
		{
			int start_M = (i % Num_M_block) * 12;
			int start_K = (i / Num_M_block) * GEMM_K / 2; // TODO
			size_block_m = 12;

			float *AA = A + start_M * K / 2 + start_K;
			float *AAc;

			if (Edge_M > 0 && (i % Num_M_block) == (Num_M_block - 1))
			{
				size_block_m = Edge_M;
			}

			if (Edge_K > 0 && i >= Num_blocks0)
			{
				AAc = Ac + start_K * M + start_M * Edge_K; // note
				FLASHGEMM_NPACK(AA, AAc, size_block_m, Edge_K, K / 2);
			}
			else
			{
				AAc = Ac + start_K * M + start_M * GEMM_K / 2; // note
				FLASHGEMM_NPACK(AA, AAc, size_block_m, GEMM_K / 2, K / 2);
			}
		}

#pragma omp barrier

		for (kk = 0; kk < K; kk = kk + kc)
		{
			kc = GEMM_K;
			if (K - kk < GEMM_K)
				kc = K - kk;

			uint16_t *temp_A = (uint16_t *)Ac + kk * M;

			for (j = 0; j < Nb; j = j + nr)
			{
				nr = 32;
				if (Nb - j < 32)
					nr = Nb - j;
				float *temp_C = C + id * Nb + j;
				uint16_t *temp_B = B_bf16 + kk * N + id * Nb + j;
				if (nr == 32)
				{
					FLASHGEMM_BF16_KERNELm12xn32xk2(temp_C, temp_A, temp_B, M, Nb, kc, N, temp_Bc, kk || beta);
				}
			}
		}
	}

	free(ptrA);
	free(ptrB);
}

// M >> N
void flashgemm_single_bf16bf16f32_MgN(float *C, uint16_t *A_bf16, uint16_t *B_bf16, long M, long N, long K, int beta){
	printf("TODO, M >> N\n");
}