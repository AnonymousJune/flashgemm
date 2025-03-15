#include "PACK_x86.h"
#include "PACK_B.h"
#include "kernel.h"
#include "kernelm4.h"
#include "kernelMGN.h"
#include "set_block.h"
#include <stdlib.h>
#include <omp.h>

// N >> M
void Small_NGM_NN_SGEMM(float *C, float *A, float *B, long M, long N, long K, int NUM, int Tm, int Tn)
{
    int GEMM_K = 2048, GEMM_M = 516 , GEMM_N = 4096;
    setMBlock(M, Tm);
    setNBlock(N, Tn);
    
    void *ptr;
	float *Ac = (float *)malloc(K * M * sizeof(float));
	posix_memalign(&ptr, 64, NUM * GEMM_K * 32 * sizeof(float));
	float *Bc = (float *)ptr;
	long Num_K_block = K / GEMM_K;
	long Edge_K = K % GEMM_K;

	long Num_M_block_mb = mb / 12; // M -> mb
	long Edge_M_mb = mb % 12;	   // M -> mb
	long Num_blocks0_mb = Num_K_block * Num_M_block_mb;
	if (Edge_M_mb > 0)
	{
		Num_blocks0_mb = Num_blocks0_mb + Num_K_block;
		Num_M_block_mb = Num_M_block_mb + 1;
	}
	long Num_blocks_mb = Num_blocks0_mb;
	if (Edge_K > 0)
	{
		Num_blocks_mb = Num_blocks0_mb + Num_M_block_mb;
	}

	long Num_M_block_me = me / 12; // M -> me
	long Edge_M_me = me % 12;	   // M -> me
	long Num_blocks0_me = Num_K_block * Num_M_block_me;
	if (Edge_M_me > 0)
	{
		Num_blocks0_me = Num_blocks0_me + Num_K_block;
		Num_M_block_me = Num_M_block_me + 1;
	}
	long Num_blocks_me = Num_blocks0_me;
	if (Edge_K > 0)
	{
		Num_blocks_me = Num_blocks0_me + Num_M_block_me;
	}

	long All_mb_blocks = Num_blocks_mb * (Tm - 1);
	long All_blocks;
	if (me == 0)
	{
		All_mb_blocks += Num_blocks_mb;
		All_blocks = All_mb_blocks;
	}
	else
		All_blocks = All_mb_blocks + Num_blocks_me;

#pragma omp parallel num_threads(NUM)
	{
		int i, j;
		int id = omp_get_thread_num();

		long ii, jj, kk, iis, jjs, kks, n_to, m_to, k_to;
		long jjj;

		long nc, mc, kc;
		int mr, nr;

		long Num_M_block, Num_blocks0, Num_blocks;

		for (i = id; i < All_blocks; i = i + NUM)
		{
			int tm = i / Num_blocks_mb;
			long blocks_tm = i - tm * Num_blocks_mb;
			long mb_tm = mb;
			long Edge_M = Edge_M_mb;
			if (tm == Tm - 1 && me != 0)
			{
				Num_M_block = Num_M_block_me;
				Num_blocks0 = Num_blocks0_me;
				Edge_M = Edge_M_me;
				mb_tm = me;
			}
			else
			{
				Num_M_block = Num_M_block_mb;
				Num_blocks0 = Num_blocks0_mb;
			}

			long start_M = (blocks_tm % Num_M_block) * 12;	   // note
			long start_K = (blocks_tm / Num_M_block) * GEMM_K; // note
			int size_block_m = 12;
			float *AA = A + tm * mb * K + start_M * K + start_K; // address
			float *AAc;
			if (Edge_M > 0 && (blocks_tm % Num_M_block) == (Num_M_block - 1))
			{
				size_block_m = Edge_M;
			}
			if (Edge_K > 0 && blocks_tm >= Num_blocks0)
			{
				AAc = Ac + tm * mb * K + start_K * mb_tm + start_M * Edge_K; // note
				NPACK(AA, AAc, size_block_m, Edge_K, K);
			}
			else
			{
				AAc = Ac + tm * mb * K + start_K * mb_tm + start_M * GEMM_K; // note
				NPACK(AA, AAc, size_block_m, GEMM_K, K);
			}
		}

// 此处同步的意义？？
#pragma omp barrier

		int mb_ = mb;

		jjs = (id % Tn) * nb;
		if ((ne != 0) && ((id + 1) % Tn) == 0)
			n_to = jjs + ne;
		else
			n_to = jjs + nb;

		iis = id / Tn * mb;
		if ((me != 0) && (id / Tn == (Tm - 1)))
		{
			m_to = iis + me;
			mb_ = me;
		}

		else
			m_to = iis + mb;

		kks = 0;
		k_to = K;

		float *temp_Bc = Bc + id * GEMM_K * 32;

		for (jj = jjs; jj < n_to; jj = jj + nc)
		{
			nc = GEMM_N;
			if (n_to - jj < GEMM_N)
			{
				nc = n_to - jj;
			}

			for (kk = kks; kk < k_to; kk = kk + kc)
			{
				kc = GEMM_K;
				if (k_to - kk < GEMM_K)
					kc = k_to - kk;

				float *BB = B + kk * N + jj;

				for (ii = iis; ii < m_to; ii = ii + mc)
				{
					mc = GEMM_M;
					if (m_to - ii < GEMM_M)
					{
						mc = m_to - ii;
					}
					float *AAc = Ac + (ii / mb) * mb * K + kk * mb_ + (ii % mb) * kc;
					float *temp_CC = C + ii * N + jj;
					float *temp_BB = BB;
					if (mc == 4)
					{
						SMM_NN_KERNELm4xn64(temp_CC, AAc, temp_BB, mc, nc, kc, N, K, kk != kks);
						continue;
					}
					for (jjj = 0; jjj < nc; jjj = jjj + nr)
					{
						nr = 32;
						if (nc - jjj < 32)
							nr = nc - jjj;
						if (nr == 32)
						{
							SMM_NN_KERNELm12xn32(temp_CC, AAc, temp_BB, mc, nc, kc, N, K, temp_Bc, kk != kks);
							temp_BB += 32;
							temp_CC += 32;
						}
						else
						{
							int edge_Nc = nr;
							if (edge_Nc >= 16)
							{
								SMM_NN_KERNELm12xn16(temp_CC, AAc, temp_BB, mc, nc, kc, N, K, temp_Bc, kk != kks);
								edge_Nc -= 16;
								temp_BB += 16;
								temp_CC += 16;
							}

							if (edge_Nc >= 8)
							{
								SMM_NN_KERNELm12xn8(temp_CC, AAc, temp_BB, mc, nc, kc, N, K, temp_Bc, kk != kks);
								edge_Nc -= 8;
								temp_BB += 8;
								temp_CC += 8;
							}

							if (edge_Nc >= 4)
							{
								SMM_NN_KERNELm12xn4(temp_CC, AAc, temp_BB, mc, nc, kc, N, K, temp_Bc, kk != kks);
								edge_Nc -= 4;
								temp_BB += 4;
								temp_CC += 4;
							}

							if (edge_Nc >= 1)
							{
								SMM_NN_KERNELm12xn1(temp_CC, AAc, temp_BB, mc, nc, kc, N, K, temp_Bc, kk != kks);
								edge_Nc -= 1;
								temp_BB += 1;
								temp_CC += 1;
							}

							if (edge_Nc >= 1)
							{
								SMM_NN_KERNELm12xn1(temp_CC, AAc, temp_BB, mc, nc, kc, N, K, temp_Bc, kk != kks);
								edge_Nc -= 1;
								temp_BB += 1;
								temp_CC += 1;
							}

							if (edge_Nc >= 1)
							{
								SMM_NN_KERNELm12xn1(temp_CC, AAc, temp_BB, mc, nc, kc, N, K, temp_Bc, kk != kks);
								edge_Nc -= 1;
								temp_BB += 1;
								temp_CC += 1;
							}
						}
					}
				}
			}
		}
	}
	free(Ac);
	free(Bc);
}


// M >> N
void Small_MGN_NN_SGEMM(float *C, float *A, float *B, long M, long N, long K, int NUM, int Tm, int Tn)
{
	int GEMM_K = 2048, GEMM_M = 6144, GEMM_N = 1024;
	void *ptr;
	float *Bc = (float *)malloc(K * N * sizeof(float));
	posix_memalign(&ptr, 64, NUM * GEMM_K * 12 * sizeof(float));
	float *Ac = (float *)ptr;

	setMBlock(M, Tm);
	setNBlock(N, Tn);

	// prepare to pack B
	long Num_K_block = K / GEMM_K;
	long Edge_K = K % GEMM_K;

	long Num_N_block = N / 32;
	long Edge_N = N % 32;
	long Num_blocks0 = Num_K_block * Num_N_block;
	if (Edge_N > 0)
	{
		Num_blocks0 = Num_blocks0 + Num_K_block;
		Num_N_block = Num_N_block + 1;
	}
	long Num_blocks = Num_blocks0;
	if (Edge_K > 0)
	{
		Num_blocks = Num_blocks0 + Num_N_block;
	}

#pragma omp parallel num_threads(NUM)
	{
		int i, kk, size_block_n;
		// int id = 0;
		int id = omp_get_thread_num(); // TODO
		int mr, nr, mc, nc, kc;
		float *temp_Ac = Ac + id * GEMM_K * 12;

		// pack B
		for (i = id; i < Num_blocks; i = i + NUM)
		{
			long start_N = (i % Num_N_block) * 32;	   // note
			long start_K = (i / Num_N_block) * GEMM_K; // note
			size_block_n = 32;

			float *BB = B + start_K * N + start_N; // address
			float *BBc;							   // 打包后的数据存放在内存中？？

			if (Edge_N > 0 && (i % Num_N_block) == (Num_N_block - 1))
			{
				size_block_n = Edge_N;
			}

			if (Edge_K > 0 && i >= Num_blocks0)
			{
				BBc = Bc + start_K * N + start_N * Edge_K; // note
														   // printf("NPACKB_EDGE:%-3d BB:%-10x BBc:%-10x\n", i, BB, BBc);
				NPACKB(BB, BBc, size_block_n, Edge_K, N);
			}
			else
			{
				BBc = Bc + start_K * N + start_N * GEMM_K; // note
														   // printf("NPACKB:%-3d BB:%-10x BBc:%-10x\n", i, BB, BBc);
				NPACKB(BB, BBc, size_block_n, GEMM_K, N);
			}
		}

		// printf("\n\nBc:\n");
		// print_matrix(Bc, K, N);

#pragma omp barrier
		float *temp_C, *temp_A, *temp_Bc;

		int Mb = mb, iis = id * mb;
		if ((me != 0) && (id == (Tm - 1)))
		{
			Mb = me;
		}

		for (kk = 0; kk < K; kk = kk + kc)
		{
			kc = GEMM_K;
			if (K - kk < GEMM_K)
				kc = K - kk;

			float *temp_Bc = Bc + kk * N;

			for (i = 0; i < Mb; i = i + mr)
			{
				mr = 12;
				if (Mb - i < 12)
					mr = Mb - i;
				float *temp_C = C + (id * mb + i) * N;
				float *temp_A = A + (id * mb + i) * K + kk;
				if (mr == 12)
				{
					MGN_KERNEL12x32(temp_C, temp_A, temp_Bc, Mb, N, kc, N, K, temp_Ac, kk);
					// free(C);
				}
				else
				{
					int edge_Mc = mr;
					if (edge_Mc >= 8)
					{
						MGN_KERNEL8x32(temp_C, temp_A, temp_Bc, Mb, N, kc, N, K, temp_Ac, kk);
						edge_Mc -= 8;
						temp_A += 8 * K;
						temp_C += 8 * N;
						// free(C);
					}
					if (edge_Mc >= 4)
					{
						MGN_KERNEL4x32(temp_C, temp_A, temp_Bc, Mb, N, kc, N, K, temp_Ac, kk);
						edge_Mc -= 4;
						temp_A += 4 * K;
						temp_C += 4 * N;
					}
					if (edge_Mc >= 1)
					{
						MGN_KERNEL1x32(temp_C, temp_A, temp_Bc, Mb, N, kc, N, K, temp_Ac, kk);
						edge_Mc -= 1;
						temp_A += 1 * K;
						temp_C += 1 * N;
					}
					if (edge_Mc >= 1)
					{
						MGN_KERNEL1x32(temp_C, temp_A, temp_Bc, Mb, N, kc, N, K, temp_Ac, kk);
						edge_Mc -= 1;
						temp_A += 1 * K;
						temp_C += 1 * N;
					}
					if (edge_Mc >= 1)
					{
						MGN_KERNEL1x32(temp_C, temp_A, temp_Bc, Mb, N, kc, N, K, temp_Ac, kk);
					}
				}
			}
		}
	}
	free(Ac);
	free(Bc);
}

