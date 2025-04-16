#include <immintrin.h>

static void NPACK_B_K16N32(long N, long LN, long K, float *B, float *Bc)
{
  for (long k = 0; k < K; k += 8)
  {
    float *B_ptr = B + k * LN;        // 当前 B 的起始地址
    float *Bc_ptr = Bc + k * 32;      // 当前 Bc 的起始地址
    for (long n = 0; n < N; n += 32)
    {
      // 加载 B 的 8 行数据到寄存器
      __m512 row00 = _mm512_loadu_ps(B_ptr + 0 * LN);
      __m512 row01 = _mm512_loadu_ps(B_ptr + 0 * LN + 16);
      __m512 row10 = _mm512_loadu_ps(B_ptr + 1 * LN);
      __m512 row11 = _mm512_loadu_ps(B_ptr + 1 * LN + 16);
      __m512 row20 = _mm512_loadu_ps(B_ptr + 2 * LN);
      __m512 row21 = _mm512_loadu_ps(B_ptr + 2 * LN + 16);
      __m512 row30 = _mm512_loadu_ps(B_ptr + 3 * LN);
      __m512 row31 = _mm512_loadu_ps(B_ptr + 3 * LN + 16);
      __m512 row40 = _mm512_loadu_ps(B_ptr + 4 * LN);
      __m512 row41 = _mm512_loadu_ps(B_ptr + 4 * LN + 16);
      __m512 row50 = _mm512_loadu_ps(B_ptr + 5 * LN);
      __m512 row51 = _mm512_loadu_ps(B_ptr + 5 * LN + 16);
      __m512 row60 = _mm512_loadu_ps(B_ptr + 6 * LN);
      __m512 row61 = _mm512_loadu_ps(B_ptr + 6 * LN + 16);
      __m512 row70 = _mm512_loadu_ps(B_ptr + 7 * LN);
      __m512 row71 = _mm512_loadu_ps(B_ptr + 7 * LN + 16);

      // 将数据存储到 Bc
      _mm512_storeu_ps(Bc_ptr + 0 * 16, row00);
      _mm512_storeu_ps(Bc_ptr + 1 * 16, row01);
      _mm512_storeu_ps(Bc_ptr + 2 * 16, row10);
      _mm512_storeu_ps(Bc_ptr + 3 * 16, row11);
      _mm512_storeu_ps(Bc_ptr + 4 * 16, row20);
      _mm512_storeu_ps(Bc_ptr + 5 * 16, row21);
      _mm512_storeu_ps(Bc_ptr + 6 * 16, row30);
      _mm512_storeu_ps(Bc_ptr + 7 * 16, row31);
      _mm512_storeu_ps(Bc_ptr + 8 * 16, row40);
      _mm512_storeu_ps(Bc_ptr + 9 * 16, row41);
      _mm512_storeu_ps(Bc_ptr + 10 * 16, row50);
      _mm512_storeu_ps(Bc_ptr + 11 * 16, row51);
      _mm512_storeu_ps(Bc_ptr + 12 * 16, row60);
      _mm512_storeu_ps(Bc_ptr + 13 * 16, row61);
      _mm512_storeu_ps(Bc_ptr + 14 * 16, row70);
      _mm512_storeu_ps(Bc_ptr + 15 * 16, row71);

      // 更新指针
      B_ptr += 32;
      Bc_ptr += 32 * K;
    }
  }
}