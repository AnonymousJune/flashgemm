#include "./flashgemm.h"
#include "../utils.h"
#include <math.h>

int flashgemm_thread_num = 1;

void flashgemm_set_thread_num(int num)
{
	flashgemm_thread_num = num;
}

bool flashgemm_test_dimention(int M, int N, int K)
{
	bool flag = true;
	if (K % 2 != 0)
	{
		printf("K must be a multiple of 2\n");
		flag = false;
	}
	return flag;
}

// beta = 0/1, alpha = 1
void flashgemm_single_bf16bf16f32(float *C, uint16_t *A_bf16, uint16_t *B_bf16, long M, long N, long K, int beta)
{
	if (!flashgemm_test_dimention(M, N, K))
	{
		printf("dimention error\n");
		return;
	}
	// if (N > M)
	// {
	flashgemm_single_bf16bf16f32_MlN(C, A_bf16, B_bf16, M, N, K, beta);
	// }
	// else
	// {
	// 	flashgemm_single_bf16bf16f32_MgN(C, A_bf16, B_bf16, M, N, K, beta);
	// }
}

// M << N
void flashgemm_single_bf16bf16f32_MlN(float *C, uint16_t *A_bf16, uint16_t *B_bf16, long M, long N, long K, int beta)
{
	int NUM = flashgemm_thread_num;
	float *A = (float *)A_bf16;
	void *ptrA, *ptrB;
	int K_Ac = ((K + 31) / 32) * 32; // for edge process K%32!=0
	int M_Ac = ((M + 3) / 4) * 4;	 // for edge process M%4!=0
	posix_memalign(&ptrA, 64, M_Ac * K_Ac * sizeof(uint16_t));
	posix_memalign(&ptrB, 64, NUM * GEMM_K * 32 * sizeof(uint16_t));
	float *Ac = (float *)ptrA;
	uint16_t *Bc = (uint16_t *)ptrB;

	// printf("Ac: %p, Bc: %p, C: %p\n", Ac, Bc, C);

	// prepare to block by dimention N
	int nb, ne, num_n;

	nb = ((N % NUM == 0) ? N / NUM : ceil(N / NUM) + 15) / 16 * 16;
	ne = N % nb;
	num_n = N / nb + ((ne == 0) ? 0 : 1);

	// printf("nb: %d, ne: %d, num_n: %d\n", nb, ne, num_n);

	// prepare to pack A
	int Num_K_block = K / GEMM_K;
	int Edge_K = (K_Ac % GEMM_K) / 2;
	int Num_M_block = M / 12;
	int Edge_M = M % 12;
	int Num_blocks0 = Num_K_block * Num_M_block;
	if (Edge_M > 0)
	{
		Num_blocks0 += Num_K_block; // 0
		Num_M_block += 1;			// 1
	}
	int Num_blocks = Num_blocks0;
	if (Edge_K > 0)
	{
		Num_blocks = Num_blocks0 + Num_M_block; // 1
	}

#pragma omp parallel num_threads(NUM)
	{
		long i, j, k, ii, jj, kk, mc, nc, kc, kc_Ac, mr, nr;
		int size_block_m;
		int id = omp_get_thread_num();
		uint16_t *temp_Bc = Bc + id * GEMM_K * 32;

		int NB = (id < num_n - 1) ? nb : ((id == num_n - 1 && ne > 0) ? ne : nb);
		// printf("id: %d, NB: %d\n", id, NB);

		for (i = id; i < Num_blocks; i = i + NUM)
		{
			int start_M = (i % Num_M_block) * 12;
			int start_K = (i / Num_M_block) * GEMM_K / 2;
			size_block_m = 12;

			float *AA = A + start_M * K / 2 + start_K;
			float *AAc;

			if (Edge_M > 0 && (i % Num_M_block) == (Num_M_block - 1))
			{
				size_block_m = Edge_M;
			}

			if (Edge_K > 0 && i >= Num_blocks0)
			{
				AAc = Ac + start_K * M_Ac + start_M * Edge_K; // note
				FLASHGEMM_NPACK(AA, AAc, size_block_m, Edge_K, K / 2);
			}
			else
			{
				AAc = Ac + start_K * M_Ac + start_M * GEMM_K / 2; // note
				FLASHGEMM_NPACK(AA, AAc, size_block_m, GEMM_K / 2, K / 2);
			}
		}

		printf("Ac:\n");
		show_matrix_bf16(M_Ac, K_Ac, (uint16_t *)Ac);
#pragma omp barrier

		for (kk = 0; kk < K; kk = kk + kc)
		{
			kc = GEMM_K;
			kc_Ac = GEMM_K;
			if (K - kk < GEMM_K)
			{
				kc_Ac = K_Ac - kk;
				kc = K - kk;
			}

			uint16_t *temp_A = (uint16_t *)Ac + kk * M_Ac;

			for (j = 0; j < NB; j = j + nr)
			{
				nr = 32;
				if (NB - j < 32)
					nr = NB - j;
				float *temp_C = C + id * nb + j;
				uint16_t *temp_B = B_bf16 + kk * N + id * nb + j;
				// printf("id: %d, NB: %d, kk: %d, j: %d, nr: %d, temp_C: %p, temp_B: %p\n", id, NB, kk, j, nr, temp_C, temp_B);
				if (nr == 32)
				{
					FLASHGEMM_BF16_KERNELm12xn32(temp_C, temp_A, temp_B, M, kc, kc_Ac, N, temp_Bc, kk || beta);
				}
				else if (nr > 16 and nr < 32)
				{
					FLASHGEMM_BF16_KERNELm12xn16_edge(temp_C, temp_A, temp_B, M, kc, kc_Ac, N, temp_Bc, kk || beta, 16);
					FLASHGEMM_BF16_KERNELm12xn16_edge(temp_C + 16, temp_A, temp_B + 16, M, kc, kc_Ac, N, temp_Bc, kk || beta, nr - 16);
				}
				else
				{
					FLASHGEMM_BF16_KERNELm12xn16_edge(temp_C, temp_A, temp_B, M, kc, kc_Ac, N, temp_Bc, kk || beta, nr);
				}
			}
		}
	}

	free(ptrA);
	free(ptrB);
}

// M >> N
void flashgemm_single_bf16bf16f32_MgN(float *C, uint16_t *A_bf16, uint16_t *B_bf16, long M, long N, long K, int beta)
{
	printf("TODO, M >> N\n");
}