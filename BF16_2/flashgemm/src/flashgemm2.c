#include "./flashgemm.h"
#include <math.h>
#include <stdarg.h>

int flashgemm_thread_num = 1;

void flashgemm_set_thread_num(int num)
{
	flashgemm_thread_num = num;
}

static bool flashgemm_test_dimention(int M, int N, int K)
{
	bool flag = true;
	if (K % 2 != 0)
	{
		printf("K must be a multiple of 2\n");
		flag = false;
	}
	return flag;
}

static void pack_A_blocks(uint16_t *A_bf16, float *Ac, int M, int K, int M_Ac, int K_Ac, int id, int NUM)
{
	// Prepare for pack A
	int Num_K_block = K / GEMM_K;
	int Edge_K = (K_Ac % GEMM_K) / 2;
	int Num_M_block = M / 12;
	int Edge_M = M % 12;
	int Num_blocks0 = Num_K_block * Num_M_block;
	if (Edge_M > 0)
	{
		Num_blocks0 += Num_K_block;
		Num_M_block += 1;
	}
	int Num_blocks = Num_blocks0;
	if (Edge_K > 0)
	{
		Num_blocks += Num_M_block;
	}

	// Pack A blocks
	for (int i = id; i < Num_blocks; i += NUM)
	{
		int start_M = (i % Num_M_block) * 12;
		int start_K = (i / Num_M_block) * GEMM_K / 2;
		int size_block_m = 12;

		float *AA = (float *)A_bf16 + start_M * K / 2 + start_K;
		float *AAc;

		if (Edge_M > 0 && (i % Num_M_block) == (Num_M_block - 1))
		{
			size_block_m = Edge_M;
		}

		if (Edge_K > 0 && i >= Num_blocks0)
		{
			AAc = Ac + start_K * M_Ac + start_M * Edge_K;
			FLASHGEMM_NPACK(AA, AAc, size_block_m, Edge_K, K / 2);
		}
		else
		{
			AAc = Ac + start_K * M_Ac + start_M * GEMM_K / 2;
			FLASHGEMM_NPACK(AA, AAc, size_block_m, GEMM_K / 2, K / 2);
		}
	}
}

void flashgemm_multi_bf16bf16f32_type_a(float *C, uint16_t *B, long N, ...)
{
	va_list args;
	va_start(args, N);

	// Count the number of additional arguments
	int count = 0;
	va_list args_copy;
	va_copy(args_copy, args);
	while (va_arg(args_copy, void *) != NULL)
	{
		count++;
	}
	va_end(args_copy);

	// Check if the number of arguments is a multiple of 3
	if (count % 3 != 0)
	{
		printf("Error: The number of additional arguments must be a multiple of 3.\n");
		va_end(args);
		return;
	}

	// Process each group of (A, M, K)
	int num_gemm = count / 3;

	// Allocate arrays to store processed data
	uint16_t **A_bf16s = (uint16_t **)malloc(num_gemm * sizeof(uint16_t *));
	long *Ms = (long *)malloc(num_gemm * sizeof(long));
	long *Ks = (long *)malloc(num_gemm * sizeof(long));
	void **ptrAs = (void **)malloc(num_gemm * sizeof(void *));
	void **ptrBs = (void **)malloc(num_gemm * sizeof(void *));
	float **Acs = (float **)malloc(num_gemm * sizeof(float *));
	uint16_t **Bcs = (uint16_t **)malloc(num_gemm * sizeof(uint16_t *));
	int *K_Acs = (int *)malloc(num_gemm * sizeof(int));
	int *M_Acs = (int *)malloc(num_gemm * sizeof(int));

	int NUM = flashgemm_thread_num;
	int nb = (ceil(N / NUM) + 15) / 16 * 16;
	int ne = N % nb;
	int num_n = N / nb + ((ne == 0) ? 0 : 1);

	// First loop: preprocess data and store in arrays
	for (int i = 0; i < num_gemm; i++)
	{
		A_bf16s[i] = va_arg(args, uint16_t *);
		Ms[i] = va_arg(args, long);
		Ks[i] = va_arg(args, long);

		// Validate dimensions
		if (!flashgemm_test_dimention(Ms[i], N, Ks[i]))
		{
			printf("Dimension error for gemm %d\n", i + 1);
			return;
		}

		// Prepare for GEMM
		posix_memalign(&ptrAs[i], 64, ((Ms[i] + 3) / 4) * 4 * ((Ks[i] + 31) / 32) * 32 * sizeof(uint16_t));
		posix_memalign(&ptrBs[i], 64, NUM * Ks[i] * 32 * sizeof(uint16_t));
		Acs[i] = (float *)ptrAs[i];
		Bcs[i] = (uint16_t *)ptrBs[i];
		K_Acs[i] = ((Ks[i] + 31) / 32) * 32;
		M_Acs[i] = ((Ms[i] + 3) / 4) * 4;
	}

	// Second loop: perform GEMM using preprocessed data

#pragma omp parallel num_threads(NUM)
	{
		int id = omp_get_thread_num();

		// Pack A blocks
		for (int i = 0; i < num_gemm; i++)
		{
			pack_A_blocks(A_bf16s[i], Acs[i], Ms[i], Ks[i], M_Acs[i], K_Acs[i], id, NUM);
		}

#pragma omp barrier // Synchronize threads after packing A blocks

			int NB = (id < num_n - 1) ? nb : ((id == num_n - 1 && ne > 0) ? ne : nb);

			for (int j = 0; j < NB; j += nr)
			{
				int nr = (NB - j < 32) ? (NB - j) : 32;
				float *temp_C = C + id * nb + j;
				for (int i = 0; i < num_gemm; i++)
				{
					uint16_t *temp_B = B + kk * N + id * nb + j;
					uint16_t *temp_Bc = Bcs[i] + id * GEMM_K * 32;
					uint16_t *temp_Cc = (i == num_gemm - 1)? NULL : (Bcs[i+1] + id * GEMM_K * 32);

					if (nr == 32)
					{
							FLASHGEMM_BF16_KERNELm12xn32(temp_C, temp_Cc, Acs[i], temp_B, Ms[i], Ks[i], K_Acs[i], N, temp_Bc, kk || 0, i == 0, i == num_gemm - 1);
					}
					else if (nr > 16 && nr < 32)
					{
						FLASHGEMM_BF16_KERNELm12xn16_edge(temp_C, temp_Cc, Acs[i], temp_B, Ms[i], Ks[i], K_Acs[i], N, temp_Bc, kk || 0, 16, i == 0, i == num_gemm - 1);
						FLASHGEMM_BF16_KERNELm12xn16_edge(temp_C + 16, temp_Cc, Acs[i], temp_B + 16, Ms[i], Ks[i], K_Acs[i], N, temp_Bc, kk || 0, nr - 16, i == 0, i == num_gemm - 1);
					}
					else
					{
						FLASHGEMM_BF16_KERNELm12xn16_edge(temp_C, temp_Cc, Acs[i], temp_B, Ms[i], Ks[i], K_Acs[i], N, temp_Bc, kk || 0, nr, i == 0, i == num_gemm - 1);
					}
				}
			}
		}
	}

	// Free allocated arrays
	free(A_bf16s);
	free(Ms);
	free(Ks);
	free(ptrAs);
	free(ptrBs);
	free(Acs);
	free(Bcs);
	free(K_Acs);
	free(M_Acs);

	va_end(args);
}
