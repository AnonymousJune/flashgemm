#include "./flashgemm.h"
#include "./utils.h"
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

void flashgemm_multi_f32f32f32(float *C, float *B, long N, int num_gemm, ...)
{
	va_list args;
	va_start(args, num_gemm);

	// Allocate arrays to store processed data
	float *A_f32s[num_gemm];
	long Ms[num_gemm];
	long Ks[num_gemm];
	void *ptrAs[num_gemm];
	void *ptrBs[num_gemm];
	float *Acs[num_gemm];
	float *Bcs[num_gemm];
	int K_Acs[num_gemm];
	int M_Acs[num_gemm];

	int NUM = flashgemm_thread_num;
	int nb, ne, num_n;
	if (((N + 15) / 16) % NUM == 0)
	{
		nb = (((N + 15) / 16) / NUM) * 16;
	}
	else
	{
		nb = (((N + 15) / 16) + NUM - 1) / NUM * 16;
	}
	ne = N % nb;
	num_n = N / nb + ((ne == 0) ? 0 : 1);

	// First loop: preprocess data and store in arrays
	for (int i = 0; i < num_gemm; i++)
	{
		A_f32s[i] = va_arg(args, float *);
		Ms[i] = va_arg(args, long);
		M_Acs[i] = ((Ms[i] + 3) / 4) * 4;

		Ks[i] = va_arg(args, long);
		K_Acs[i] = ((Ks[i] + 15) / 16) * 16; // K_Acs[i] % 16 ==0

		// Validate dimensions
		if (!flashgemm_test_dimention(Ms[i], N, Ks[i]))
		{
			printf("Dimension error for gemm %d\n", i + 1);
			return;
		}

		// Prepare for GEMM
		posix_memalign(&ptrAs[i], 64, M_Acs[i] * K_Acs[i] * sizeof(float));
		posix_memalign(&ptrBs[i], 64, NUM * K_Acs[i] * 32 * sizeof(float));
		Acs[i] = (float *)ptrAs[i];
		Bcs[i] = (float *)ptrBs[i];
	}

	va_end(args);

#pragma omp parallel num_threads(NUM)
	{
		int id = omp_get_thread_num();
		int nr, NB;

		// Pack A blocks
		for (int i = 0; i < num_gemm; i++)
		{
			// Moved functionality of pack_A_blocks here
			int Num_blocks = Ms[i] / 12 + ((Ms[i] % 12 == 0) ? 0 : 1);

			for (int j = id; j < Num_blocks; j += NUM)
			{
				int start_M = j * 12;
				int size_block_m = 12;

				float *AA = (float *)A_f32s[i] + start_M * Ks[i];
				float *AAc = Acs[i] + start_M * K_Acs[i];

				if (j == Num_blocks - 1 && Ms[i] % 12 != 0)
				{
					size_block_m = Ms[i] % 12;
				}

				FLASHGEMM_NPACK(AA, AAc, size_block_m, K_Acs[i], Ks[i]);
			}
		}

#pragma omp barrier // Synchronize threads after packing A blocks

		if (id < num_n - 1)
		{
			NB = nb;
		}
		else if (id == num_n - 1)
		{
			NB = (ne > 0) ? ne : nb;
		}
		else
		{
			NB = 0;
		}
		// printf("thread%d,NB:%d\n", id, NB);

		for (int j = 0; j < NB; j += nr)
		{
			nr = (NB - j < 32) ? (NB - j) : 32;
			float *temp_C = C + id * nb + j;
			float *temp_B = B + id * nb + j;
			for (int i = 0; i < num_gemm; i++)
			{
				float *temp_Ac = (float *)Acs[i];
				float *temp_Bc = Bcs[i] + id * K_Acs[i] * 32;
				float *temp_Cc = (i == num_gemm - 1) ? NULL : (Bcs[i + 1] + id * K_Acs[i + 1] * 32);

				if (nr == 32)
				{
					FLASHGEMM_F32_KERNELm12xn32(temp_C, temp_Cc, temp_Ac, temp_B, Ms[i], Ks[i], K_Acs[i], N, temp_Bc, i == 0, i == num_gemm - 1);
				}
				else if (nr > 16 && nr < 32)
				{
					FLASHGEMM_F32_KERNELm12xn16_edge(temp_C, temp_Cc, temp_Ac, temp_B, Ms[i], Ks[i], K_Acs[i], N, temp_Bc, 16, i == 0, i == num_gemm - 1);
					FLASHGEMM_F32_KERNELm12xn16_edge(temp_C + 16, temp_Cc, temp_Ac, temp_B + 16, Ms[i], Ks[i], K_Acs[i], N, temp_Bc, nr - 16, i == 0, i == num_gemm - 1);
				}
				else
				{
					FLASHGEMM_F32_KERNELm12xn16_edge(temp_C, temp_Cc, temp_Ac, temp_B, Ms[i], Ks[i], K_Acs[i], N, temp_Bc, nr, i == 0, i == num_gemm - 1);
				}
			}
		}
	}
	// Free allocated memory
	for (int i = 0; i < num_gemm; i++)
	{
		free(ptrAs[i]);
		free(ptrBs[i]);
	}
}

void flashgemm_multi_bf16bf16f32(float *C, uint16_t *B, long N, int num_gemm, ...)
{
	va_list args;
	va_start(args, num_gemm);

	// Allocate arrays to store processed data
	uint16_t *A_bf16s[num_gemm];
	long Ms[num_gemm];
	long Ks[num_gemm];
	void *ptrAs[num_gemm];
	void *ptrBs[num_gemm];
	float *Acs[num_gemm];
	uint16_t *Bcs[num_gemm];
	int K_Acs[num_gemm];
	int M_Acs[num_gemm];

	int NUM = flashgemm_thread_num;
	int nb, ne, num_n;
	if (((N + 15) / 16) % NUM == 0)
	{
		nb = (((N + 15) / 16) / NUM) * 16;
	}
	else
	{
		nb = (((N + 15) / 16) + NUM - 1) / NUM * 16;
	}
	ne = N % nb;
	num_n = N / nb + ((ne == 0) ? 0 : 1);

	// First loop: preprocess data and store in arrays
	for (int i = 0; i < num_gemm; i++)
	{
		A_bf16s[i] = va_arg(args, uint16_t *);
		Ms[i] = va_arg(args, long);
		M_Acs[i] = ((Ms[i] + 3) / 4) * 4;

		Ks[i] = va_arg(args, long);
		K_Acs[i] = ((Ks[i] + 31) / 32) * 32;

		// Validate dimensions
		if (!flashgemm_test_dimention(Ms[i], N, Ks[i]))
		{
			printf("Dimension error for gemm %d\n", i + 1);
			return;
		}

		// Prepare for GEMM
		posix_memalign(&ptrAs[i], 64, M_Acs[i] * K_Acs[i] * sizeof(uint16_t));
		posix_memalign(&ptrBs[i], 64, NUM * K_Acs[i] * 32 * sizeof(uint16_t));
		Acs[i] = (float *)ptrAs[i];
		Bcs[i] = (uint16_t *)ptrBs[i];
	}

	va_end(args);

#pragma omp parallel num_threads(NUM)
	{
		int id = omp_get_thread_num();
		int nr, NB;

		// Pack A blocks
		for (int i = 0; i < num_gemm; i++)
		{
			// Moved functionality of pack_A_blocks here
			int Num_blocks = Ms[i] / 12 + ((Ms[i] % 12 == 0) ? 0 : 1);

			for (int j = id; j < Num_blocks; j += NUM)
			{
				int start_M = j * 12;
				int size_block_m = 12;

				float *AA = (float *)A_bf16s[i] + start_M * Ks[i] / 2;
				float *AAc = Acs[i] + start_M * K_Acs[i] / 2;

				if (j == Num_blocks - 1 && Ms[i] % 12 != 0)
				{
					size_block_m = Ms[i] % 12;
				}

				FLASHGEMM_NPACK(AA, AAc, size_block_m, K_Acs[i] / 2, Ks[i] / 2);
			}
		}

#pragma omp barrier // Synchronize threads after packing A blocks

		if (id < num_n - 1)
		{
			NB = nb;
		}
		else if (id == num_n - 1)
		{
			NB = (ne > 0) ? ne : nb;
		}
		else
		{
			NB = 0;
		}
		// printf("thread%d,NB:%d\n", id, NB);

		for (int j = 0; j < NB; j += nr)
		{
			nr = (NB - j < 32) ? (NB - j) : 32;
			float *temp_C = C + id * nb + j;
			uint16_t *temp_B = B + id * nb + j;
			for (int i = 0; i < num_gemm; i++)
			{
				uint16_t *temp_Ac = (uint16_t *)Acs[i];
				uint16_t *temp_Bc = Bcs[i] + id * K_Acs[i] * 32;
				uint16_t *temp_Cc = (i == num_gemm - 1) ? NULL : (Bcs[i + 1] + id * K_Acs[i + 1] * 32);

				if (nr == 32)
				{
					FLASHGEMM_BF16_KERNELm12xn32(temp_C, temp_Cc, temp_Ac, temp_B, Ms[i], Ks[i], K_Acs[i], N, temp_Bc, i == 0, i == num_gemm - 1);
				}
				else if (nr > 16 && nr < 32)
				{
					FLASHGEMM_BF16_KERNELm12xn16_edge(temp_C, temp_Cc, temp_Ac, temp_B, Ms[i], Ks[i], K_Acs[i], N, temp_Bc, 16, i == 0, i == num_gemm - 1);
					FLASHGEMM_BF16_KERNELm12xn16_edge(temp_C + 16, temp_Cc, temp_Ac, temp_B + 16, Ms[i], Ks[i], K_Acs[i], N, temp_Bc, nr - 16, i == 0, i == num_gemm - 1);
				}
				else
				{
					FLASHGEMM_BF16_KERNELm12xn16_edge(temp_C, temp_Cc, temp_Ac, temp_B, Ms[i], Ks[i], K_Acs[i], N, temp_Bc, nr, i == 0, i == num_gemm - 1);
				}
			}
		}
	}
	// Free allocated memory
	for (int i = 0; i < num_gemm; i++)
	{
		free(Acs[i]);
		free(Bcs[i]);
	}
}

void flashgemm_multi_int8uint8int32(int *C, uint8_t *B, long N, int num_gemm, ...)
{
	va_list args;
	va_start(args, num_gemm);

	// Allocate arrays to store processed data
	int8_t *A_int8s[num_gemm];
	long Ms[num_gemm];
	long Ks[num_gemm];
	void *ptrAs[num_gemm];
	void *ptrBs[num_gemm];
	int *Acs[num_gemm];
	uint8_t *Bcs[num_gemm];
	int K_Acs[num_gemm];
	int M_Acs[num_gemm];

	int NUM = flashgemm_thread_num;
	int nb, ne, num_n;
	if (((N + 15) / 16) % NUM == 0)
	{
		nb = (((N + 15) / 16) / NUM) * 16;
	}
	else
	{
		nb = (((N + 15) / 16) + NUM - 1) / NUM * 16;
	}
	ne = N % nb;
	num_n = N / nb + ((ne == 0) ? 0 : 1);

	// First loop: preprocess data and store in arrays
	for (int i = 0; i < num_gemm; i++)
	{
		A_int8s[i] = va_arg(args, int8_t *);
		Ms[i] = va_arg(args, long);
		M_Acs[i] = ((Ms[i] + 3) / 4) * 4;

		Ks[i] = va_arg(args, long);
		K_Acs[i] = ((Ks[i] + 63) / 64) * 64; // MODIFIED

		// Validate dimensions
		if (!flashgemm_test_dimention(Ms[i], N, Ks[i]))
		{
			printf("Dimension error for gemm %d\n", i + 1);
			return;
		}

		// Prepare for GEMM
		posix_memalign(&ptrAs[i], 64, M_Acs[i] * K_Acs[i] * sizeof(uint8_t));
		posix_memalign(&ptrBs[i], 64, NUM * K_Acs[i] * 32 * sizeof(uint8_t));
		Acs[i] = (int *)ptrAs[i];
		Bcs[i] = (uint8_t *)ptrBs[i];
	}

	va_end(args);

#pragma omp parallel num_threads(NUM)
	{
		int id = omp_get_thread_num();
		int nr, NB;

		// Pack A blocks
		for (int i = 0; i < num_gemm; i++)
		{
			// Moved functionality of pack_A_blocks here
			int Num_blocks = Ms[i] / 12 + ((Ms[i] % 12 == 0) ? 0 : 1);

			for (int j = id; j < Num_blocks; j += NUM)
			{
				int start_M = j * 12;
				int size_block_m = 12;

				int *AA = (int *)A_int8s[i] + start_M * Ks[i] / 4;
				int *AAc = Acs[i] + start_M * K_Acs[i] / 4;

				if (j == Num_blocks - 1 && Ms[i] % 12 != 0)
				{
					size_block_m = Ms[i] % 12;
				}

				FLASHGEMM_NPACK((float *)AA, (float *)AAc, size_block_m, K_Acs[i] / 4, Ks[i] / 4);
			}
		}

#pragma omp barrier // Synchronize threads after packing A blocks

		if (id < num_n - 1)
		{
			NB = nb;
		}
		else if (id == num_n - 1)
		{
			NB = (ne > 0) ? ne : nb;
		}
		else
		{
			NB = 0;
		}
		// printf("thread%d,NB:%d\n", id, NB);

		for (int j = 0; j < NB; j += nr)
		{
			nr = (NB - j < 32) ? (NB - j) : 32;
			int *temp_C = C + id * nb + j;
			uint8_t *temp_B = B + id * nb + j;
			for (int i = 0; i < num_gemm; i++)
			{
				int8_t *temp_Ac = (int8_t *)Acs[i];
				uint8_t *temp_Bc = Bcs[i] + id * K_Acs[i] * 32;
				uint8_t *temp_Cc = (i == num_gemm - 1) ? NULL : (Bcs[i + 1] + id * K_Acs[i + 1] * 32);

				if (nr == 32)
				{
					FLASHGEMM_INT8_KERNELm12xn32(temp_C, temp_Cc, temp_Ac, temp_B, Ms[i], Ks[i], K_Acs[i], N, temp_Bc, i == 0, i == num_gemm - 1);
				}
				else if (nr > 16 && nr < 32)
				{
					FLASHGEMM_INT8_KERNELm12xn16_edge(temp_C, temp_Cc, temp_Ac, temp_B, Ms[i], Ks[i], K_Acs[i], N, temp_Bc, 16, i == 0, i == num_gemm - 1);
					FLASHGEMM_INT8_KERNELm12xn16_edge(temp_C + 16, temp_Cc, temp_Ac, temp_B + 16, Ms[i], Ks[i], K_Acs[i], N, temp_Bc, nr - 16, i == 0, i == num_gemm - 1);
				}
				else
				{
					FLASHGEMM_INT8_KERNELm12xn16_edge(temp_C, temp_Cc, temp_Ac, temp_B, Ms[i], Ks[i], K_Acs[i], N, temp_Bc, nr, i == 0, i == num_gemm - 1);
				}
			}
		}
	}
	// Free allocated memory
	for (int i = 0; i < num_gemm; i++)
	{
		free(Acs[i]);
		free(Bcs[i]);
	}
}
