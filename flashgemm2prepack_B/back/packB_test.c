#include <stdlib.h>
#include <stdio.h>
#include <cstdint>
#include <immintrin.h>
#include <string.h>
#include "packB_f32.h"
#include "utils.h"

using namespace std;

int main()
{
  long N = 64;
  long K = 16;

  void *ptrB, *ptrBc;
  posix_memalign(&ptrB, 64, K * N * sizeof(float));
  posix_memalign(&ptrBc, 64, K * N * sizeof(float));
  float *B = (float*)ptrB;
  float *Bc = (float*)ptrBc;

  regular_matrix_f32(K, N, B);
  printf("\nB:\n");
  show_matrix_fp32(K, N, B);

  NPACK_B_K16N32(N, N, K, B, Bc);
  printf("\nBc:\n");
  show_matrix_fp32(K, N, Bc);

  free(ptrB);
  free(ptrBc);

  return 0;
}
