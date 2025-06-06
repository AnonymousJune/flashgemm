#include <stdint.h>

static void FLASHGEMM_F32_KERNELm12xn32(float *C, float *Cc, float *A, float *B, long M, long K, long LK, long LN, float *Bc, bool is_start_gemm, bool is_end_gemm)
{
	asm volatile(
			".macro    f32_kernel_m12n32_pack_1                          \n"

			"   vbroadcastss    8(%%rax), %%zmm2                         \n" // A2
			"   vfmadd231ps        %%zmm0, %%zmm4, %%zmm8                \n" // A0*(B0-15)
			"   vfmadd231ps        %%zmm0, %%zmm5, %%zmm9                \n" // A0*(B16-31)

			"   vbroadcastss    12(%%rax), %%zmm3                        \n" // A3
			"   vfmadd231ps        %%zmm1, %%zmm4, %%zmm10               \n" // A1*(B0-15)
			"   vfmadd231ps        %%zmm1, %%zmm5, %%zmm11               \n" // A1*(B16-31)

			"   prefetcht0         256(%%rax)                            \n"

			"   vbroadcastss    16(%%rax), %%zmm0                        \n" // A4
			"   vfmadd231ps        %%zmm2, %%zmm4, %%zmm12               \n"
			"   vfmadd231ps        %%zmm2, %%zmm5, %%zmm13               \n"

			"   prefetcht2         128(%%rbx)                            \n"

			"   vbroadcastss    20(%%rax), %%zmm1                        \n" // A5
			"   vfmadd231ps        %%zmm3, %%zmm4, %%zmm14               \n"
			"   vfmadd231ps        %%zmm3, %%zmm5, %%zmm15               \n"

			"   vbroadcastss    24(%%rax), %%zmm2                        \n" // A6
			"   vfmadd231ps        %%zmm0, %%zmm4, %%zmm16               \n"
			"   vfmadd231ps        %%zmm0, %%zmm5, %%zmm17               \n"

			"   prefetcht2         192(%%rbx)                            \n"

			"   vbroadcastss    28(%%rax), %%zmm3                        \n" // A7
			"   vfmadd231ps        %%zmm1, %%zmm4, %%zmm18               \n"
			"   vfmadd231ps        %%zmm1, %%zmm5, %%zmm19               \n"

			"   vbroadcastss    32(%%rax), %%zmm0                        \n" // A8
			"   vfmadd231ps        %%zmm2, %%zmm4, %%zmm20               \n"
			"   vfmadd231ps        %%zmm2, %%zmm5, %%zmm21               \n"

			"    leaq      (%%rbx, %%r8, 4), %%rbx                       \n" // B

			"   vbroadcastss    36(%%rax), %%zmm1                        \n" // A9
			"   vfmadd231ps        %%zmm3, %%zmm4, %%zmm22               \n"
			"   vfmadd231ps        %%zmm3, %%zmm5, %%zmm23               \n"

			"   vbroadcastss    40(%%rax), %%zmm2                        \n" // A10
			"   vfmadd231ps        %%zmm0, %%zmm4, %%zmm24               \n"
			"   vfmadd231ps        %%zmm0, %%zmm5, %%zmm25               \n"

			"   vbroadcastss    44(%%rax), %%zmm3                        \n" // A11
			"   vfmadd231ps        %%zmm1, %%zmm4, %%zmm26               \n"
			"   vmovups         (%%rbx), %%zmm6                          \n" // next B0
			"    addq              $48, %%rax                            \n" // 下一组A(已读12个)
			"   vfmadd231ps        %%zmm1, %%zmm5, %%zmm27               \n"

			"   vbroadcastss    (%%rax), %%zmm0                          \n" // next A0
			"   vfmadd231ps        %%zmm2, %%zmm4, %%zmm28               \n"
			"   vmovups         64(%%rbx), %%zmm7                        \n" // next B1
			"   vfmadd231ps        %%zmm2, %%zmm5, %%zmm29               \n"

			"   vbroadcastss    4(%%rax), %%zmm1                         \n" // next A1
			"   vfmadd231ps        %%zmm3, %%zmm4, %%zmm30               \n"
			"   vmovups         %%zmm4, (%%rbp)                          \n" // pack B0 to Bc
			"   vfmadd231ps        %%zmm3, %%zmm5, %%zmm31               \n"
			"   vmovups         %%zmm5, 64(%%rbp)                        \n" // pack B1 to Bc
			"    addq              $128, %%rbp                           \n"

			".endm                                                       \n"

			".macro    f32_kernel_m12n32_pack_2                          \n"

			"   vbroadcastss    8(%%rax), %%zmm2                         \n" // next A2
			"   vfmadd231ps        %%zmm0, %%zmm6, %%zmm8                \n"
			"   vfmadd231ps        %%zmm0, %%zmm7, %%zmm9                \n"

			"   prefetcht0         256(%%rax)                            \n"

			"   vbroadcastss    12(%%rax), %%zmm3                        \n" // next A3
			"   vfmadd231ps        %%zmm1, %%zmm6, %%zmm10               \n"
			"   vfmadd231ps        %%zmm1, %%zmm7, %%zmm11               \n"

			"   prefetcht2         128(%%rbx)                            \n"

			"   vbroadcastss    16(%%rax), %%zmm0                        \n" // next A4
			"   vfmadd231ps        %%zmm2, %%zmm6, %%zmm12               \n"
			"   vfmadd231ps        %%zmm2, %%zmm7, %%zmm13               \n"

			"   prefetcht2         192(%%rbx)                            \n"

			"   vbroadcastss    20(%%rax), %%zmm1                        \n" // next A5
			"   vfmadd231ps        %%zmm3, %%zmm6, %%zmm14               \n"
			"   vfmadd231ps        %%zmm3, %%zmm7, %%zmm15               \n"

			"   vbroadcastss    24(%%rax), %%zmm2                        \n" // next A6
			"   vfmadd231ps        %%zmm0, %%zmm6, %%zmm16               \n"
			"   vfmadd231ps        %%zmm0, %%zmm7, %%zmm17               \n"

			"   vbroadcastss    28(%%rax), %%zmm3                        \n" // next A7
			"   vfmadd231ps        %%zmm1, %%zmm6, %%zmm18               \n"
			"   vfmadd231ps        %%zmm1, %%zmm7, %%zmm19               \n"

			"   vbroadcastss    32(%%rax), %%zmm0                        \n" // next A8
			"   vfmadd231ps        %%zmm2, %%zmm6, %%zmm20               \n"
			"   vfmadd231ps        %%zmm2, %%zmm7, %%zmm21               \n"

			"    leaq      (%%rbx, %%r8, 4), %%rbx                       \n" // B

			"   vbroadcastss    36(%%rax), %%zmm1                        \n" // next A9
			"   vfmadd231ps        %%zmm3, %%zmm6, %%zmm22               \n"
			"   vfmadd231ps        %%zmm3, %%zmm7, %%zmm23               \n"

			"   vbroadcastss    40(%%rax), %%zmm2                        \n" // next A10
			"   vfmadd231ps        %%zmm0, %%zmm6, %%zmm24               \n"
			"   vfmadd231ps        %%zmm0, %%zmm7, %%zmm25               \n"

			"   vbroadcastss    44(%%rax), %%zmm3                        \n" // next A11
			"   vfmadd231ps        %%zmm1, %%zmm6, %%zmm26               \n"
			"   vmovups         (%%rbx), %%zmm4                          \n" // next next B0
			"    addq              $48, %%rax                            \n" // 下一组A(已读12个)
			"   vfmadd231ps        %%zmm1, %%zmm7, %%zmm27               \n"

			"   vbroadcastss    (%%rax), %%zmm0                          \n" // next next A0
			"   vfmadd231ps        %%zmm2, %%zmm6, %%zmm28               \n"
			"   vmovups         64(%%rbx), %%zmm5                        \n" // next next B1
			"   vfmadd231ps        %%zmm2, %%zmm7, %%zmm29               \n"

			"   vbroadcastss    4(%%rax), %%zmm1                         \n" // next next A1
			"   vfmadd231ps        %%zmm3, %%zmm6, %%zmm30               \n"
			"   vmovups         %%zmm6, (%%rbp)                          \n" // pack B0 to Bc
			"   vfmadd231ps        %%zmm3, %%zmm7, %%zmm31               \n"
			"   vmovups         %%zmm7, 64(%%rbp)                        \n" // pack B1 to Bc
			"    addq              $128, %%rbp                           \n"

			".endm                                                       \n"

			".macro    f32_kernel_m12n32_pack_1_end                      \n"

			"   vbroadcastss    8(%%rax), %%zmm2                         \n" // A2
			"   vfmadd231ps        %%zmm0, %%zmm4, %%zmm8                \n" // A0*(B0-15)
			"   vfmadd231ps        %%zmm0, %%zmm5, %%zmm9                \n" // A0*(B16-31)

			"   vbroadcastss    12(%%rax), %%zmm3                        \n" // A3
			"   vfmadd231ps        %%zmm1, %%zmm4, %%zmm10               \n" // A1*(B0-15)
			"   vfmadd231ps        %%zmm1, %%zmm5, %%zmm11               \n" // A1*(B16-31)

			"   prefetcht0         256(%%rax)                            \n"

			"   vbroadcastss    16(%%rax), %%zmm0                        \n" // A4
			"   vfmadd231ps        %%zmm2, %%zmm4, %%zmm12               \n"
			"   vfmadd231ps        %%zmm2, %%zmm5, %%zmm13               \n"

			"   prefetcht2         128(%%rbx)                            \n"

			"   vbroadcastss    20(%%rax), %%zmm1                        \n" // A5
			"   vfmadd231ps        %%zmm3, %%zmm4, %%zmm14               \n"
			"   vfmadd231ps        %%zmm3, %%zmm5, %%zmm15               \n"

			"   vbroadcastss    24(%%rax), %%zmm2                        \n" // A6
			"   vfmadd231ps        %%zmm0, %%zmm4, %%zmm16               \n"
			"   vfmadd231ps        %%zmm0, %%zmm5, %%zmm17               \n"

			"   prefetcht2         192(%%rbx)                            \n"

			"   vbroadcastss    28(%%rax), %%zmm3                        \n" // A7
			"   vfmadd231ps        %%zmm1, %%zmm4, %%zmm18               \n"
			"   vfmadd231ps        %%zmm1, %%zmm5, %%zmm19               \n"

			"   vbroadcastss    32(%%rax), %%zmm0                        \n" // A8
			"   vfmadd231ps        %%zmm2, %%zmm4, %%zmm20               \n"
			"   vfmadd231ps        %%zmm2, %%zmm5, %%zmm21               \n"

			"   vbroadcastss    36(%%rax), %%zmm1                        \n" // A9
			"   vfmadd231ps        %%zmm3, %%zmm4, %%zmm22               \n"
			"   vfmadd231ps        %%zmm3, %%zmm5, %%zmm23               \n"

			"   vbroadcastss    40(%%rax), %%zmm2                        \n" // A10
			"   vfmadd231ps        %%zmm0, %%zmm4, %%zmm24               \n"
			"   vfmadd231ps        %%zmm0, %%zmm5, %%zmm25               \n"

			"   vbroadcastss    44(%%rax), %%zmm3                        \n" // A11
			"   vfmadd231ps        %%zmm1, %%zmm4, %%zmm26               \n"
			"   vfmadd231ps        %%zmm1, %%zmm5, %%zmm27               \n"
			
			"    addq              $48, %%rax                            \n" // 下一组A(已读12个)
			
			"   vmovups         %%zmm4, (%%rbp)                          \n" // pack B0 to Bc
			"   vfmadd231ps        %%zmm2, %%zmm4, %%zmm28               \n"
			"   vfmadd231ps        %%zmm2, %%zmm5, %%zmm29               \n"

			"   vmovups         %%zmm5, 64(%%rbp)                        \n" // pack B1 to Bc
			"   vfmadd231ps        %%zmm3, %%zmm4, %%zmm30               \n"
			"   vfmadd231ps        %%zmm3, %%zmm5, %%zmm31               \n"

			".endm                                                       \n"

			".macro    f32_kernel_m12n32_pack_2_end                      \n"

			"   vbroadcastss    8(%%rax), %%zmm2                         \n" // next A2
			"   vfmadd231ps        %%zmm0, %%zmm6, %%zmm8                \n"
			"   vfmadd231ps        %%zmm0, %%zmm7, %%zmm9                \n"

			"   prefetcht0         256(%%rax)                            \n"

			"   vbroadcastss    12(%%rax), %%zmm3                        \n" // next A3
			"   vfmadd231ps        %%zmm1, %%zmm6, %%zmm10               \n"
			"   vfmadd231ps        %%zmm1, %%zmm7, %%zmm11               \n"

			"   prefetcht2         128(%%rbx)                            \n"

			"   vbroadcastss    16(%%rax), %%zmm0                        \n" // next A4
			"   vfmadd231ps        %%zmm2, %%zmm6, %%zmm12               \n"
			"   vfmadd231ps        %%zmm2, %%zmm7, %%zmm13               \n"

			"   prefetcht2         192(%%rbx)                            \n"

			"   vbroadcastss    20(%%rax), %%zmm1                        \n" // next A5
			"   vfmadd231ps        %%zmm3, %%zmm6, %%zmm14               \n"
			"   vfmadd231ps        %%zmm3, %%zmm7, %%zmm15               \n"

			"   vbroadcastss    24(%%rax), %%zmm2                        \n" // next A6
			"   vfmadd231ps        %%zmm0, %%zmm6, %%zmm16               \n"
			"   vfmadd231ps        %%zmm0, %%zmm7, %%zmm17               \n"

			"   vbroadcastss    28(%%rax), %%zmm3                        \n" // next A7
			"   vfmadd231ps        %%zmm1, %%zmm6, %%zmm18               \n"
			"   vfmadd231ps        %%zmm1, %%zmm7, %%zmm19               \n"

			"   vbroadcastss    32(%%rax), %%zmm0                        \n" // next A8
			"   vfmadd231ps        %%zmm2, %%zmm6, %%zmm20               \n"
			"   vfmadd231ps        %%zmm2, %%zmm7, %%zmm21               \n"

			"   vbroadcastss    36(%%rax), %%zmm1                        \n" // next A9
			"   vfmadd231ps        %%zmm3, %%zmm6, %%zmm22               \n"
			"   vfmadd231ps        %%zmm3, %%zmm7, %%zmm23               \n"

			"   vbroadcastss    40(%%rax), %%zmm2                        \n" // next A10
			"   vfmadd231ps        %%zmm0, %%zmm6, %%zmm24               \n"
			"   vfmadd231ps        %%zmm0, %%zmm7, %%zmm25               \n"

			"   vbroadcastss    44(%%rax), %%zmm3                        \n" // next A11
			"   vfmadd231ps        %%zmm1, %%zmm6, %%zmm26               \n"
			"   vfmadd231ps        %%zmm1, %%zmm7, %%zmm27               \n"
			
			"    addq              $48, %%rax                            \n" // 下一组A(已读12个)
			
			"   vmovups         %%zmm6, (%%rbp)                          \n" // pack B0 to Bc
			"   vfmadd231ps        %%zmm2, %%zmm6, %%zmm28               \n"
			"   vfmadd231ps        %%zmm2, %%zmm7, %%zmm29               \n"

			"   vmovups         %%zmm7, 64(%%rbp)                        \n" // pack B1 to Bc
			"   vfmadd231ps        %%zmm3, %%zmm6, %%zmm30               \n"
			"   vfmadd231ps        %%zmm3, %%zmm7, %%zmm31               \n"

			".endm                                                       \n"

			// ---------------------------------------------------------------

			".macro    f32_kernel_m12n32_1                               \n"
			"   vbroadcastss    8(%%rax), %%zmm2                         \n"
			"   vfmadd231ps        %%zmm0, %%zmm4, %%zmm8                \n"
			"   vfmadd231ps        %%zmm0, %%zmm5, %%zmm9                \n"

			"   vbroadcastss    12(%%rax), %%zmm3                        \n"

			"   vfmadd231ps        %%zmm1, %%zmm4, %%zmm10               \n"
			"   vfmadd231ps        %%zmm1, %%zmm5, %%zmm11               \n"

			"   prefetcht0         256(%%rax)                            \n"

			"   vbroadcastss    16(%%rax), %%zmm0                        \n"
			"   vfmadd231ps        %%zmm2, %%zmm4, %%zmm12               \n"
			"   vfmadd231ps        %%zmm2, %%zmm5, %%zmm13               \n"

			"   vbroadcastss    20(%%rax), %%zmm1                        \n"
			"   vfmadd231ps        %%zmm3, %%zmm4, %%zmm14               \n"
			"   vfmadd231ps        %%zmm3, %%zmm5, %%zmm15               \n"

			"   vbroadcastss    24(%%rax), %%zmm2                        \n"
			"   vfmadd231ps        %%zmm0, %%zmm4, %%zmm16               \n"
			"   vfmadd231ps        %%zmm0, %%zmm5, %%zmm17               \n"

			"   vbroadcastss    28(%%rax), %%zmm3                        \n"
			"   vfmadd231ps        %%zmm1, %%zmm4, %%zmm18               \n"
			"   vfmadd231ps        %%zmm1, %%zmm5, %%zmm19               \n"

			"   vbroadcastss    32(%%rax), %%zmm0                        \n"
			"   vfmadd231ps        %%zmm2, %%zmm4, %%zmm20               \n"
			"   vfmadd231ps        %%zmm2, %%zmm5, %%zmm21               \n"

			"    addq              $128, %%rbx                           \n"

			"   vbroadcastss    36(%%rax), %%zmm1                        \n"
			"   vfmadd231ps        %%zmm3, %%zmm4, %%zmm22               \n"
			"   vfmadd231ps        %%zmm3, %%zmm5, %%zmm23               \n"
      "   prefetcht0         64(%%rbx)                            \n"

			"   vbroadcastss    40(%%rax), %%zmm2                        \n"
			"   vfmadd231ps        %%zmm0, %%zmm4, %%zmm24               \n"
			"   vfmadd231ps        %%zmm0, %%zmm5, %%zmm25               \n"

			"   vbroadcastss    44(%%rax), %%zmm3                        \n"
			"   vfmadd231ps        %%zmm1, %%zmm4, %%zmm26               \n"
			"   vmovups           (%%rbx), %%zmm6                        \n"
			"    addq              $48, %%rax                            \n"
			"   vfmadd231ps        %%zmm1, %%zmm5, %%zmm27               \n"

			"   vbroadcastss    (%%rax), %%zmm0                          \n"
			"   vfmadd231ps        %%zmm2, %%zmm4, %%zmm28               \n"
			"   vmovups           64(%%rbx), %%zmm7                      \n"
			"   vfmadd231ps        %%zmm2, %%zmm5, %%zmm29               \n"

			"   vbroadcastss    4(%%rax), %%zmm1                         \n"
			"   vfmadd231ps        %%zmm3, %%zmm4, %%zmm30               \n"
			"   vfmadd231ps        %%zmm3, %%zmm5, %%zmm31               \n"
			".endm                                                       \n"

			".macro    f32_kernel_m12n32_2                               \n"
			"   vbroadcastss    8(%%rax), %%zmm2                         \n"
			"   vfmadd231ps        %%zmm0, %%zmm6, %%zmm8                \n"
			"   vfmadd231ps        %%zmm0, %%zmm7, %%zmm9                \n"

			"   vbroadcastss    12(%%rax), %%zmm3                        \n"
			"   vfmadd231ps        %%zmm1, %%zmm6, %%zmm10               \n"
			"   vfmadd231ps        %%zmm1, %%zmm7, %%zmm11               \n"

			"   prefetcht0         256(%%rax)                            \n"

			"   vbroadcastss    16(%%rax), %%zmm0                        \n"
			"   vfmadd231ps        %%zmm2, %%zmm6, %%zmm12               \n"
			"   vfmadd231ps        %%zmm2, %%zmm7, %%zmm13               \n"

			"   vbroadcastss    20(%%rax), %%zmm1                        \n"
			"   vfmadd231ps        %%zmm3, %%zmm6, %%zmm14               \n"
			"   vfmadd231ps        %%zmm3, %%zmm7, %%zmm15               \n"

			"   vbroadcastss    24(%%rax), %%zmm2                        \n"
			"   vfmadd231ps        %%zmm0, %%zmm6, %%zmm16               \n"
			"   vfmadd231ps        %%zmm0, %%zmm7, %%zmm17               \n"

			"   vbroadcastss    28(%%rax), %%zmm3                        \n"
			"   vfmadd231ps        %%zmm1, %%zmm6, %%zmm18               \n"
			"   vfmadd231ps        %%zmm1, %%zmm7, %%zmm19               \n"

			"   vbroadcastss    32(%%rax), %%zmm0                        \n"
			"   vfmadd231ps        %%zmm2, %%zmm6, %%zmm20               \n"
			"   vfmadd231ps        %%zmm2, %%zmm7, %%zmm21               \n"

			"   addq              $128, %%rbx                            \n"

			"   vbroadcastss    36(%%rax), %%zmm1                        \n"
			"   vfmadd231ps        %%zmm3, %%zmm6, %%zmm22               \n"
			"   vfmadd231ps        %%zmm3, %%zmm7, %%zmm23               \n"
      "   prefetcht0         64(%%rbx)                            \n"
			
			"   vbroadcastss    40(%%rax), %%zmm2                        \n"
			"   vfmadd231ps        %%zmm0, %%zmm6, %%zmm24               \n"
			"   vfmadd231ps        %%zmm0, %%zmm7, %%zmm25               \n"

			"   vbroadcastss    44(%%rax), %%zmm3                        \n"
			"   vfmadd231ps        %%zmm1, %%zmm6, %%zmm26               \n"
			"   vmovups           (%%rbx), %%zmm4                        \n"
			"    addq              $48, %%rax                            \n"
			"   vfmadd231ps        %%zmm1, %%zmm7, %%zmm27               \n"

			"   vbroadcastss    (%%rax), %%zmm0                          \n"
			"   vfmadd231ps        %%zmm2, %%zmm6, %%zmm28               \n"
			"   vmovups           64(%%rbx), %%zmm5                      \n"
			"   vfmadd231ps        %%zmm2, %%zmm7, %%zmm29               \n"

			"   vbroadcastss    4(%%rax), %%zmm1                         \n"
			"   vfmadd231ps        %%zmm3, %%zmm6, %%zmm30               \n"
			"   vfmadd231ps        %%zmm3, %%zmm7, %%zmm31               \n"
			".endm                                                       \n"

			".macro    f32_kernel_m12n32_end                             \n"
			"   vbroadcastss    8(%%rax), %%zmm2                         \n"
			"   vfmadd231ps        %%zmm0, %%zmm6, %%zmm8                \n"
			"   vfmadd231ps        %%zmm0, %%zmm7, %%zmm9                \n"
			"   vbroadcastss    12(%%rax), %%zmm3                        \n"

			"   vfmadd231ps        %%zmm1, %%zmm6, %%zmm10               \n"
			"   vfmadd231ps        %%zmm1, %%zmm7, %%zmm11               \n"

			"   prefetcht0         256(%%rax)                            \n"

			"   vbroadcastss    16(%%rax), %%zmm0                        \n"
			"   vfmadd231ps        %%zmm2, %%zmm6, %%zmm12               \n"
			"   vfmadd231ps        %%zmm2, %%zmm7, %%zmm13               \n"

			"   vbroadcastss    20(%%rax), %%zmm1                        \n"
			"   vfmadd231ps        %%zmm3, %%zmm6, %%zmm14               \n"
			"   vfmadd231ps        %%zmm3, %%zmm7, %%zmm15               \n"

			"   vbroadcastss    24(%%rax), %%zmm2                        \n"
			"   vfmadd231ps        %%zmm0, %%zmm6, %%zmm16               \n"
			"   vfmadd231ps        %%zmm0, %%zmm7, %%zmm17               \n"

			"   vbroadcastss    28(%%rax), %%zmm3                        \n"
			"   vfmadd231ps        %%zmm1, %%zmm6, %%zmm18               \n"
			"   vfmadd231ps        %%zmm1, %%zmm7, %%zmm19               \n"

			"   vbroadcastss    32(%%rax), %%zmm0                        \n"
			"   vfmadd231ps        %%zmm2, %%zmm6, %%zmm20               \n"
			"   vfmadd231ps        %%zmm2, %%zmm7, %%zmm21               \n"

			"   vbroadcastss    36(%%rax), %%zmm1                        \n"
			"   vfmadd231ps        %%zmm3, %%zmm6, %%zmm22               \n"
			"   vfmadd231ps        %%zmm3, %%zmm7, %%zmm23               \n"

			"   vbroadcastss    40(%%rax), %%zmm2                        \n"
			"   vfmadd231ps        %%zmm0, %%zmm6, %%zmm24               \n"
			"   vfmadd231ps        %%zmm0, %%zmm7, %%zmm25               \n"

			"   vbroadcastss    44(%%rax), %%zmm3                        \n"
			"   vfmadd231ps        %%zmm1, %%zmm6, %%zmm26               \n"
			"   addq               $48, %%rax                            \n"
			"   vfmadd231ps        %%zmm1, %%zmm7, %%zmm27               \n"

			"   vfmadd231ps        %%zmm2, %%zmm6, %%zmm28               \n"
			"   vfmadd231ps        %%zmm2, %%zmm7, %%zmm29               \n"

			"   vfmadd231ps        %%zmm3, %%zmm6, %%zmm30               \n"
			"   vfmadd231ps        %%zmm3, %%zmm7, %%zmm31               \n"
			".endm                                                       \n"

			".macro    f32_save_c_m12n32                                 \n"
			"   vmovups         %%zmm8, (%%r10)                          \n"
			"   vmovups         %%zmm9, 64(%%r10)                        \n"
			"   vmovups         %%zmm10, (%%r11)                         \n"
			"   vmovups         %%zmm11, 64(%%r11)                       \n"
			"   vmovups         %%zmm12, (%%r12)                         \n"
			"   vmovups         %%zmm13, 64(%%r12)                       \n"
			"   vmovups         %%zmm14, (%%r13)                         \n"
			"   vmovups         %%zmm15, 64(%%r13)                       \n"

			"   leaq  (%%r13, %%r8, 4), %%r10                            \n" // C0
			"   leaq     (%%r10, %%r8, 4), %%r11                         \n" // C1
			"   leaq     (%%r11, %%r8, 4), %%r12                         \n" // C2
			"   leaq     (%%r12, %%r8, 4), %%r13                         \n" // C3

			"   vmovups         %%zmm16, (%%r10)                         \n"
			"   vmovups         %%zmm17, 64(%%r10)                       \n"
			"   vmovups         %%zmm18, (%%r11)                         \n"
			"   vmovups         %%zmm19, 64(%%r11)                       \n"
			"   vmovups         %%zmm20, (%%r12)                         \n"
			"   vmovups         %%zmm21, 64(%%r12)                       \n"
			"   vmovups         %%zmm22, (%%r13)                         \n"
			"   vmovups         %%zmm23, 64(%%r13)                       \n"

			"   leaq  (%%r13, %%r8, 4), %%r10                            \n" // C0
			"   leaq     (%%r10, %%r8, 4), %%r11                         \n" // C1
			"   leaq     (%%r11, %%r8, 4), %%r12                         \n" // C2
			"   leaq     (%%r12, %%r8, 4), %%r13                         \n" // C3

			"   vmovups         %%zmm24, (%%r10)                         \n"
			"   vmovups         %%zmm25, 64(%%r10)                       \n"
			"   vmovups         %%zmm26, (%%r11)                         \n"
			"   vmovups         %%zmm27, 64(%%r11)                       \n"
			"   subq             $12, %%rdi                              \n"
			"   vmovups         %%zmm28, (%%r12)                         \n"
			"   vmovups         %%zmm29, 64(%%r12)                       \n"
			"   vmovups         %%zmm30, (%%r13)                         \n"
			"   vmovups         %%zmm31, 64(%%r13)                       \n"

			"   leaq      (%%r13, %%r8, 4), %%rcx                        \n" // C0
			".endm                                                       \n"

			".macro    f32_save_c_m12n32_2                               \n"
			"   vmovups         %%zmm8, (%%r10)                          \n"
			"   vmovups         %%zmm9, 64(%%r10)                        \n"
			"   vmovups         %%zmm10, 128(%%r10)                      \n"
			"   vmovups         %%zmm11, 192(%%r10)                      \n"
			"   vmovups         %%zmm12, 256(%%r10)                      \n"
			"   vmovups         %%zmm13, 320(%%r10)                      \n"
			"   vmovups         %%zmm14, 384(%%r10)                      \n"
			"   vmovups         %%zmm15, 448(%%r10)                      \n"

			"   addq            $512, %%r10                              \n"

			"   vmovups         %%zmm16, (%%r10)                         \n"
			"   vmovups         %%zmm17, 64(%%r10)                       \n"
			"   vmovups         %%zmm18, 128(%%r10)                      \n"
			"   vmovups         %%zmm19, 192(%%r10)                      \n"
			"   vmovups         %%zmm20, 256(%%r10)                      \n"
			"   vmovups         %%zmm21, 320(%%r10)                      \n"
			"   vmovups         %%zmm22, 384(%%r10)                      \n"
			"   vmovups         %%zmm23, 448(%%r10)                      \n"

			"   addq            $512, %%r10                              \n"

			"   vmovups         %%zmm24, (%%r10)                         \n"
			"   vmovups         %%zmm25, 64(%%r10)                       \n"
			"   vmovups         %%zmm26, 128(%%r10)                      \n"
			"   vmovups         %%zmm27, 192(%%r10)                      \n"
			"   vmovups         %%zmm28, 256(%%r10)                      \n"
			"   vmovups         %%zmm29, 320(%%r10)                      \n"
			"   vmovups         %%zmm30, 384(%%r10)                      \n"
			"   vmovups         %%zmm31, 448(%%r10)                      \n"

			"   addq            $512, %%r10                              \n"
			"   subq            $12, %%rdi                               \n"
			".endm                                                       \n"

			//-----------------------------------------------------------------

			".macro    f32_kernel_m8n32_1                                \n"
			"   vbroadcastss    8(%%rax), %%zmm2                         \n"
			"   vfmadd231ps        %%zmm0, %%zmm4, %%zmm8                \n"
			"   vfmadd231ps        %%zmm0, %%zmm5, %%zmm9                \n"

			"   vbroadcastss    12(%%rax), %%zmm3                        \n"
			"   vfmadd231ps        %%zmm1, %%zmm4, %%zmm10               \n"
			"   vfmadd231ps        %%zmm1, %%zmm5, %%zmm11               \n"

			"   prefetcht0      256(%%rax)                               \n"
			"   addq             $128, %%rbx                             \n"

			"   vbroadcastss    16(%%rax), %%zmm0                        \n"
			"   vfmadd231ps        %%zmm2, %%zmm4, %%zmm12               \n"
			"   vfmadd231ps        %%zmm2, %%zmm5, %%zmm13               \n"
			"   vmovups          (%%rbx), %%zmm6                         \n"

			"   vbroadcastss    20(%%rax), %%zmm1                        \n"
			"   vfmadd231ps        %%zmm3, %%zmm4, %%zmm14               \n"
			"   vfmadd231ps        %%zmm3, %%zmm5, %%zmm15               \n"

			"   vbroadcastss    24(%%rax), %%zmm2                        \n"
			"   vfmadd231ps        %%zmm0, %%zmm4, %%zmm16               \n"
			"   vfmadd231ps        %%zmm0, %%zmm5, %%zmm17               \n"
			"   vmovups          64(%%rbx), %%zmm7                       \n"

			"   vbroadcastss    28(%%rax), %%zmm3                        \n"
			"   vfmadd231ps        %%zmm1, %%zmm4, %%zmm18               \n"
			"   vfmadd231ps        %%zmm1, %%zmm5, %%zmm19               \n"

			"   addq              $32, %%rax                             \n"
			"   vbroadcastss     (%%rax), %%zmm0                         \n"
			"   vfmadd231ps        %%zmm2, %%zmm4, %%zmm20               \n"
			"   vfmadd231ps        %%zmm2, %%zmm5, %%zmm21               \n"

			"   vbroadcastss     4(%%rax), %%zmm1                        \n"
			"   vfmadd231ps        %%zmm3, %%zmm4, %%zmm22               \n"
			"   vfmadd231ps        %%zmm3, %%zmm5, %%zmm23               \n"
			".endm                                                       \n"

			".macro    f32_kernel_m8n32_2                                \n"
			"   vbroadcastss    8(%%rax), %%zmm2                         \n"
			"   vfmadd231ps        %%zmm0, %%zmm6, %%zmm8                \n"
			"   vfmadd231ps        %%zmm0, %%zmm7, %%zmm9                \n"

			"   vbroadcastss    12(%%rax), %%zmm3                        \n"
			"   vfmadd231ps        %%zmm1, %%zmm6, %%zmm10               \n"
			"   vfmadd231ps        %%zmm1, %%zmm7, %%zmm11               \n"

			"   prefetcht0         256(%%rax)                            \n"
			"   addq              $128, %%rbx                            \n"

			"   vbroadcastss    16(%%rax), %%zmm0                        \n"
			"   vfmadd231ps        %%zmm2, %%zmm6, %%zmm12               \n"
			"   vfmadd231ps        %%zmm2, %%zmm7, %%zmm13               \n"

			"   vbroadcastss    20(%%rax), %%zmm1                        \n"
			"   vfmadd231ps        %%zmm3, %%zmm6, %%zmm14               \n"
			"   vfmadd231ps        %%zmm3, %%zmm7, %%zmm15               \n"

			"   vbroadcastss    24(%%rax), %%zmm2                        \n"
			"   vfmadd231ps        %%zmm0, %%zmm6, %%zmm16               \n"
			"   vfmadd231ps        %%zmm0, %%zmm7, %%zmm17               \n"
			"   vmovups         (%%rbx), %%zmm4                          \n"

			"   vbroadcastss    28(%%rax), %%zmm3                        \n"
			"   vfmadd231ps        %%zmm1, %%zmm6, %%zmm18               \n"
			"   vfmadd231ps        %%zmm1, %%zmm7, %%zmm19               \n"
			"   vmovups         64(%%rbx), %%zmm5                        \n"

			"   addq             $32, %%rax                              \n"

			"   vbroadcastss    (%%rax), %%zmm0                          \n"
			"   vfmadd231ps        %%zmm2, %%zmm6, %%zmm20               \n"
			"   vfmadd231ps        %%zmm2, %%zmm7, %%zmm21               \n"

			"   vbroadcastss    4(%%rax), %%zmm1                         \n"
			"   vfmadd231ps        %%zmm3, %%zmm6, %%zmm22               \n"
			"   vfmadd231ps        %%zmm3, %%zmm7, %%zmm23               \n"
			".endm                                                       \n"

			".macro    f32_kernel_m8n32_end                              \n"
			"   vbroadcastss    8(%%rax), %%zmm2                         \n"
			"   vfmadd231ps        %%zmm0, %%zmm6, %%zmm8                \n"
			"   vfmadd231ps        %%zmm0, %%zmm7, %%zmm9                \n"

			"   vbroadcastss    12(%%rax), %%zmm3                        \n"
			"   vfmadd231ps        %%zmm1, %%zmm6, %%zmm10               \n"
			"   vfmadd231ps        %%zmm1, %%zmm7, %%zmm11               \n"

			"   prefetcht0         256(%%rax)                            \n"

			"   vbroadcastss    16(%%rax), %%zmm0                        \n"
			"   vfmadd231ps        %%zmm2, %%zmm6, %%zmm12               \n"
			"   vfmadd231ps        %%zmm2, %%zmm7, %%zmm13               \n"

			"   vbroadcastss    20(%%rax), %%zmm1                        \n"
			"   vfmadd231ps        %%zmm3, %%zmm6, %%zmm14               \n"
			"   vfmadd231ps        %%zmm3, %%zmm7, %%zmm15               \n"

			"   vbroadcastss    24(%%rax), %%zmm2                        \n"
			"   vfmadd231ps        %%zmm0, %%zmm6, %%zmm16               \n"
			"   vfmadd231ps        %%zmm0, %%zmm7, %%zmm17               \n"

			"   vbroadcastss    28(%%rax), %%zmm3                        \n"
			"   vfmadd231ps        %%zmm1, %%zmm6, %%zmm18               \n"
			"   vfmadd231ps        %%zmm1, %%zmm7, %%zmm19               \n"

			"   addq             $32, %%rax                              \n"

			"   vfmadd231ps        %%zmm2, %%zmm6, %%zmm20               \n"
			"   vfmadd231ps        %%zmm2, %%zmm7, %%zmm21               \n"

			"   vfmadd231ps        %%zmm3, %%zmm6, %%zmm22               \n"
			"   vfmadd231ps        %%zmm3, %%zmm7, %%zmm23               \n"
			".endm                                                       \n"

			".macro    f32_save_c_m8n32                                  \n"
			"   vmovups         %%zmm8, (%%r10)                          \n"
			"   vmovups         %%zmm9, 64(%%r10)                        \n"
			"   vmovups         %%zmm10, (%%r11)                         \n"
			"   vmovups         %%zmm11, 64(%%r11)                       \n"
			"   vmovups         %%zmm12, (%%r12)                         \n"
			"   vmovups         %%zmm13, 64(%%r12)                       \n"
			"   vmovups         %%zmm14, (%%r13)                         \n"
			"   vmovups         %%zmm15, 64(%%r13)                       \n"

			"   leaq     (%%r13, %%r8, 4), %%r10                         \n" // C0
			"   leaq     (%%r10, %%r8, 4), %%r11                         \n" // C1
			"   leaq     (%%r11, %%r8, 4), %%r12                         \n" // C2
			"   leaq     (%%r12, %%r8, 4), %%r13                         \n" // C3

			"   vmovups         %%zmm16, (%%r10)                         \n"
			"   vmovups         %%zmm17, 64(%%r10)                       \n"
			"   vmovups         %%zmm18, (%%r11)                         \n"
			"   vmovups         %%zmm19, 64(%%r11)                       \n"
			"   vmovups         %%zmm20, (%%r12)                         \n"
			"   vmovups         %%zmm21, 64(%%r12)                       \n"
			"   vmovups         %%zmm22, (%%r13)                         \n"
			"   vmovups         %%zmm23, 64(%%r13)                       \n"

			"   subq            $8, %%rdi                                \n"
			"   leaq      (%%r13, %%r8, 4), %%rcx                        \n" // C0
			".endm                                                       \n"

			".macro    f32_save_c_m8n32_2                                \n"
			"   vmovups         %%zmm8, (%%r10)                          \n"
			"   vmovups         %%zmm9, 64(%%r10)                        \n"
			"   vmovups         %%zmm10, 128(%%r10)                      \n"
			"   vmovups         %%zmm11, 192(%%r10)                      \n"
			"   vmovups         %%zmm12, 256(%%r10)                      \n"
			"   vmovups         %%zmm13, 320(%%r10)                      \n"
			"   vmovups         %%zmm14, 384(%%r10)                      \n"
			"   vmovups         %%zmm15, 448(%%r10)                      \n"

			"   addq            $512, %%r10                              \n"

			"   vmovups         %%zmm16, (%%r10)                         \n"
			"   vmovups         %%zmm17, 64(%%r10)                       \n"
			"   vmovups         %%zmm18, 128(%%r10)                      \n"
			"   vmovups         %%zmm19, 192(%%r10)                      \n"
			"   vmovups         %%zmm20, 256(%%r10)                      \n"
			"   vmovups         %%zmm21, 320(%%r10)                      \n"
			"   vmovups         %%zmm22, 384(%%r10)                      \n"
			"   vmovups         %%zmm23, 448(%%r10)                      \n"

			"   addq            $512, %%r10                              \n"
			"   subq            $8, %%rdi                                \n"
			".endm                                                       \n"

			//-----------------------------------------------------------------

			".macro    f32_kernel_m4n32_1                                \n"
			"   vbroadcastss    8(%%rax), %%zmm2                         \n"
			"   vfmadd231ps        %%zmm0, %%zmm4, %%zmm8                \n"
			"   vfmadd231ps        %%zmm0, %%zmm5, %%zmm9                \n"
			"   addq             $128, %%rbx                             \n"
			"   vmovups          (%%rbx), %%zmm6                         \n"

			"   vbroadcastss    12(%%rax), %%zmm3                        \n"
			"   vfmadd231ps        %%zmm1, %%zmm4, %%zmm10               \n"
			"   vfmadd231ps        %%zmm1, %%zmm5, %%zmm11               \n"
			"   vmovups          64(%%rbx), %%zmm7                       \n"

			"   prefetcht0      256(%%rax)                               \n"
			"   addq              $16, %%rax                             \n"

			"   vbroadcastss     (%%rax), %%zmm0                         \n"
			"   vfmadd231ps        %%zmm2, %%zmm4, %%zmm12               \n"
			"   vfmadd231ps        %%zmm2, %%zmm5, %%zmm13               \n"

			"   vbroadcastss     4(%%rax), %%zmm1                        \n"
			"   vfmadd231ps        %%zmm3, %%zmm4, %%zmm14               \n"
			"   vfmadd231ps        %%zmm3, %%zmm5, %%zmm15               \n"
			".endm                                                       \n"

			".macro    f32_kernel_m4n32_2                                \n"
			"   vbroadcastss     8(%%rax), %%zmm2                        \n"
			"   vfmadd231ps        %%zmm0, %%zmm6, %%zmm8                \n"
			"   vfmadd231ps        %%zmm0, %%zmm7, %%zmm9                \n"
			"   addq             $128, %%rbx                             \n"
			"   vmovups          (%%rbx), %%zmm4                         \n"

			"   vbroadcastss     12(%%rax), %%zmm3                       \n"
			"   vfmadd231ps        %%zmm1, %%zmm6, %%zmm10               \n"
			"   vfmadd231ps        %%zmm1, %%zmm7, %%zmm11               \n"
			"   vmovups          64(%%rbx), %%zmm5                       \n"

			"   prefetcht0         256(%%rax)                            \n"
			"   addq             $16, %%rax                              \n"

			"   vbroadcastss     (%%rax), %%zmm0                         \n"
			"   vfmadd231ps        %%zmm2, %%zmm6, %%zmm12               \n"
			"   vfmadd231ps        %%zmm2, %%zmm7, %%zmm13               \n"

			"   vbroadcastss     4(%%rax), %%zmm1                        \n"
			"   vfmadd231ps        %%zmm3, %%zmm6, %%zmm14               \n"
			"   vfmadd231ps        %%zmm3, %%zmm7, %%zmm15               \n"
			".endm                                                       \n"

			".macro    f32_kernel_m4n32_end                              \n"
			"   vbroadcastss    8(%%rax), %%zmm2                         \n"
			"   vfmadd231ps        %%zmm0, %%zmm6, %%zmm8                \n"
			"   vfmadd231ps        %%zmm0, %%zmm7, %%zmm9                \n"

			"   vbroadcastss    12(%%rax), %%zmm3                        \n"
			"   vfmadd231ps        %%zmm1, %%zmm6, %%zmm10               \n"
			"   vfmadd231ps        %%zmm1, %%zmm7, %%zmm11               \n"

			"   prefetcht0         256(%%rax)                            \n"

			"   addq             $16, %%rax                              \n"
			"   vbroadcastss     (%%rax), %%zmm0                         \n"
			"   vfmadd231ps        %%zmm2, %%zmm6, %%zmm12               \n"
			"   vfmadd231ps        %%zmm2, %%zmm7, %%zmm13               \n"

			"   vbroadcastss     4(%%rax), %%zmm1                        \n"
			"   vfmadd231ps        %%zmm3, %%zmm6, %%zmm14               \n"
			"   vfmadd231ps        %%zmm3, %%zmm7, %%zmm15               \n"
			".endm                                                       \n"

			".macro    f32_save_c_m4n32                                  \n"
			"   vmovups         %%zmm8, (%%r10)                          \n"
			"   vmovups         %%zmm9, 64(%%r10)                        \n"
			"   vmovups         %%zmm10, (%%r11)                         \n"
			"   vmovups         %%zmm11, 64(%%r11)                       \n"
			"   vmovups         %%zmm12, (%%r12)                         \n"
			"   vmovups         %%zmm13, 64(%%r12)                       \n"
			"   vmovups         %%zmm14, (%%r13)                         \n"
			"   vmovups         %%zmm15, 64(%%r13)                       \n"

			"   subq            $4, %%rdi                                \n"
			"   leaq      (%%r13, %%r8, 4), %%rcx                        \n" // C0
			".endm                                                       \n"

			".macro    f32_save_c_m4n32_2                                \n"
			"   vmovups         %%zmm8, (%%r10)                          \n"
			"   vmovups         %%zmm9, 64(%%r10)                        \n"
			"   vmovups         %%zmm10, 128(%%r10)                      \n"
			"   vmovups         %%zmm11, 192(%%r10)                      \n"
			"   vmovups         %%zmm12, 256(%%r10)                      \n"
			"   vmovups         %%zmm13, 320(%%r10)                      \n"
			"   vmovups         %%zmm14, 384(%%r10)                      \n"
			"   vmovups         %%zmm15, 448(%%r10)                      \n"

			"   addq            $512, %%r10                              \n"
			"   subq            $4, %%rdi                                \n"
			".endm                                                       \n"

			//-----------------------------------------------------------------
			//-----------------------------------------------------------------

			"GEMM_F32_N32:                                               \n"
			"   mov     %[C], %%rcx                                      \n"
			"   mov     %[Cc], %%r10                                     \n"
			"   mov     %[A], %%rax                                      \n"
			"   mov     %[B], %%rbx                                      \n"

			"   prefetcht0         (%%rax)                               \n"

			"   mov     %[K], %%rdx                                      \n"
			"   mov     %[LN], %%r8                                      \n"
			"   mov     %[Bc], %%r14                                     \n"
			"   mov     %[M], %%rdi                                      \n"

			"   mov     %[LK], %%r15                                     \n"
			"   mov     %%rax, %%r9                                      \n"

			"   prefetcht0         (%%rbx)                               \n"
			"   mov     %%rdx, %%rsi                                     \n"

			"   mov    %[is_start_gemm], %%r12                           \n"
			"   test   $1, %%r12                                         \n"
			"   jz     F32_BEGIN_M12N32                                  \n"

			//-----------------------------------------------------------------

			"F32_BEGIN_PACK_N32:                                         \n"
			"   mov     %%rsi, %%rdx                                     \n" // K
			"   mov     %%r14, %%rbp                                     \n" // Bc

			"   vmovups (%%rbx), %%zmm4                                  \n"
			"   vmovups 64(%%rbx), %%zmm5                                \n"

			"   vpxorq         %%zmm8, %%zmm8, %%zmm8                    \n"
			"   vpxorq         %%zmm9, %%zmm9, %%zmm9                    \n"
			"   vpxorq         %%zmm10, %%zmm10, %%zmm10                 \n"
			"   vpxorq         %%zmm11, %%zmm11, %%zmm11                 \n"
			"   vpxorq         %%zmm12, %%zmm12, %%zmm12                 \n"
			"   vpxorq         %%zmm13, %%zmm13, %%zmm13                 \n"
			"   vpxorq         %%zmm14, %%zmm14, %%zmm14                 \n"
			"   vpxorq         %%zmm15, %%zmm15, %%zmm15                 \n"
			"   vpxorq         %%zmm16, %%zmm16, %%zmm16                 \n"
			"   vpxorq         %%zmm17, %%zmm17, %%zmm17                 \n"
			"   vpxorq         %%zmm18, %%zmm18, %%zmm18                 \n"
			"   vpxorq         %%zmm19, %%zmm19, %%zmm19                 \n"
			"   vbroadcastss    (%%rax), %%zmm0                          \n" // A0
			"   vbroadcastss    4(%%rax), %%zmm1                         \n" // A1
			"   vpxorq         %%zmm20, %%zmm20, %%zmm20                 \n"
			"   vpxorq         %%zmm21, %%zmm21, %%zmm21                 \n"
			"   vpxorq         %%zmm22, %%zmm22, %%zmm22                 \n"
			"   vpxorq         %%zmm23, %%zmm23, %%zmm23                 \n"
			"   vpxorq         %%zmm24, %%zmm24, %%zmm24                 \n"
			"   vpxorq         %%zmm25, %%zmm25, %%zmm25                 \n"
			"   vpxorq         %%zmm26, %%zmm26, %%zmm26                 \n"
			"   vpxorq         %%zmm27, %%zmm27, %%zmm27                 \n"
			"   vpxorq         %%zmm28, %%zmm28, %%zmm28                 \n"
			"   vpxorq         %%zmm29, %%zmm29, %%zmm29                 \n"
			"   vpxorq         %%zmm30, %%zmm30, %%zmm30                 \n"
			"   vpxorq         %%zmm31, %%zmm31, %%zmm31                 \n"
			"   cmp     $8, %%rdx                                        \n"
			"   jb      F32_PACK_MAIN_M12N32K1                           \n"
			"   subq    $8, %%rdx                                        \n"

			"F32_PACK_MAIN_M12N32K8:                                     \n"
			"   f32_kernel_m12n32_pack_1                                 \n"
			"   f32_kernel_m12n32_pack_2                                 \n"
			"   f32_kernel_m12n32_pack_1                                 \n"
			"   f32_kernel_m12n32_pack_2                                 \n"
			"   f32_kernel_m12n32_pack_1                                 \n"
			"   f32_kernel_m12n32_pack_2                                 \n"
			"   f32_kernel_m12n32_pack_1                                 \n"
			"   cmp     $0, %%rdx                                        \n"
			"   je      F32_PACK_MAIN_END_1                              \n"
			"   f32_kernel_m12n32_pack_2                                 \n"
			"   cmp     $8, %%rdx                                        \n"
			"   jb      F32_PACK_MAIN_M12N32K1                           \n"
			"   subq    $8, %%rdx                                        \n"
			"   jmp     F32_PACK_MAIN_M12N32K8                           \n"

			"F32_PACK_MAIN_M12N32K1:                                     \n"
			"   cmp     $1, %%rdx                                        \n"
			"   je      F32_PACK_MAIN_END_2                              \n"
			"   f32_kernel_m12n32_pack_1                                 \n"
			"   subq    $1, %%rdx                                        \n"
			"   cmp     $1, %%rdx                                        \n"
			"   je      F32_PACK_MAIN_END_1                              \n"
			"   f32_kernel_m12n32_pack_2                                 \n"
			"   subq    $1, %%rdx                                        \n"
			"   jmp     F32_PACK_MAIN_M12N32K1                           \n"

			"F32_PACK_MAIN_END_1:                                        \n"
			"   f32_kernel_m12n32_pack_2_end                             \n"
			"   jmp     F32_BEGIN_SAVE_M12N32                            \n"

			"F32_PACK_MAIN_END_2:                                        \n"
			"   f32_kernel_m12n32_pack_1_end                             \n"
			"   jmp     F32_BEGIN_SAVE_M12N32                            \n"

			//-----------------------------------------------------------------

			"F32_BEGIN_M12N32:                                           \n"
			"   cmpq    $12, %%rdi                                       \n"
			"   jb      F32_BEGIN_M8N32                                  \n"

			"   mov     %%r14, %%rbx                                     \n" // Bc
			"   mov     %%rsi, %%rdx                                     \n" // K
			"   vmovups        (%%rbx), %%zmm4                           \n" // B0-15
			"   vmovups     64(%%rbx), %%zmm5                            \n" // B16-31
			"   vpxorq         %%zmm8, %%zmm8, %%zmm8                    \n"
			"   vpxorq         %%zmm9, %%zmm9, %%zmm9                    \n"
			"   vpxorq         %%zmm10, %%zmm10, %%zmm10                 \n"
			"   vpxorq         %%zmm11, %%zmm11, %%zmm11                 \n"
			"   vpxorq         %%zmm12, %%zmm12, %%zmm12                 \n"
			"   vpxorq         %%zmm13, %%zmm13, %%zmm13                 \n"
			"   vpxorq         %%zmm14, %%zmm14, %%zmm14                 \n"
			"   vpxorq         %%zmm15, %%zmm15, %%zmm15                 \n"
			"   vpxorq         %%zmm16, %%zmm16, %%zmm16                 \n"
			"   vpxorq         %%zmm17, %%zmm17, %%zmm17                 \n"
			"   vpxorq         %%zmm18, %%zmm18, %%zmm18                 \n"
			"   vpxorq         %%zmm19, %%zmm19, %%zmm19                 \n"
			"   vbroadcastss    (%%rax), %%zmm0                          \n" // A0
			"   vbroadcastss    4(%%rax), %%zmm1                         \n" // A1
			"   vpxorq         %%zmm20, %%zmm20, %%zmm20                 \n"
			"   vpxorq         %%zmm21, %%zmm21, %%zmm21                 \n"
			"   vpxorq         %%zmm22, %%zmm22, %%zmm22                 \n"
			"   vpxorq         %%zmm23, %%zmm23, %%zmm23                 \n"
			"   vpxorq         %%zmm24, %%zmm24, %%zmm24                 \n"
			"   vpxorq         %%zmm25, %%zmm25, %%zmm25                 \n"
			"   vpxorq         %%zmm26, %%zmm26, %%zmm26                 \n"
			"   vpxorq         %%zmm27, %%zmm27, %%zmm27                 \n"
			"   vpxorq         %%zmm28, %%zmm28, %%zmm28                 \n"
			"   vpxorq         %%zmm29, %%zmm29, %%zmm29                 \n"
			"   vpxorq         %%zmm30, %%zmm30, %%zmm30                 \n"
			"   vpxorq         %%zmm31, %%zmm31, %%zmm31                 \n"
			"   cmp     $8, %%rdx                                        \n"
			"   jb      F32_MAIN_M12N32K1                                \n"
			"   subq    $8, %%rdx                                        \n"

			"F32_MAIN_K_M12N32K8:                                        \n" // loop K+=4
			"   f32_kernel_m12n32_1                                      \n"
			"   f32_kernel_m12n32_2                                      \n"
			"   f32_kernel_m12n32_1                                      \n"
			"   f32_kernel_m12n32_2                                      \n"
			"   f32_kernel_m12n32_1                                      \n"
			"   f32_kernel_m12n32_2                                      \n"
			"   f32_kernel_m12n32_1                                      \n"
			"   f32_kernel_m12n32_2                                      \n"
			"   cmp     $0, %%rdx                                        \n"
			"   je      F32_BEGIN_SAVE_M12N32                            \n"
			"   cmp     $8, %%rdx                                        \n"
			"   jb      F32_MAIN_M12N32K1                                \n"
			"   subq    $8, %%rdx                                        \n"
			"   jmp     F32_MAIN_K_M12N32K8                              \n"

			"F32_MAIN_M12N32K1:                                          \n"
			"   f32_kernel_m12n32_1                                      \n"
			"   subq    $1, %%rdx                                        \n"
			"   cmp     $0, %%rdx                                        \n"
			"   je      F32_BEGIN_SAVE_M12N32                            \n"
			"   f32_kernel_m12n32_2                                      \n"
			"   subq    $1, %%rdx                                        \n"
			"   cmp     $0, %%rdx                                        \n"
			"   je      F32_BEGIN_SAVE_M12N32                            \n"
			"   jmp     F32_MAIN_M12N32K1                                \n"

			"F32_BEGIN_SAVE_M12N32:                                      \n"
			"   mov    %[is_end_gemm], %%r13                             \n"
			"   test      $1, %%r13                                      \n"
			"   jz       F32_SAVE_C_M12N32_2                             \n"
			"   mov      %%rcx, %%r10                                    \n" // C0
			"   leaq     (%%r10, %%r8, 4), %%r11                         \n" // C1
			"   leaq     (%%r11, %%r8, 4), %%r12                         \n" // C2
			"   leaq     (%%r12, %%r8, 4), %%r13                         \n" // C3

			"F32_SAVE_C_M12N32:                                          \n"
			"   f32_save_c_m12n32                                        \n"
			"   imul     $48, %%r15, %%r11                               \n" // temp use %%r11
			"   add      %%r11, %%r9                                     \n"
			"   movq     %%r9, %%rax                                     \n"
			"   jmp     F32_BEGIN_M12N32                                 \n"

			"F32_SAVE_C_M12N32_2:                                        \n"
			"   f32_save_c_m12n32_2                                      \n"
			"   imul     $48, %%r15, %%r11                               \n" // temp use %%r11
			"   add      %%r11, %%r9                                     \n"
			"   movq     %%r9, %%rax                                     \n"
			"   jmp     F32_BEGIN_M12N32                                 \n"

			//-----------------------------------------------------------------

			"F32_BEGIN_M8N32:                                            \n"
			"   cmpq    $8, %%rdi                                        \n"
			"   jb      F32_BEGIN_M4N32                                  \n"

			"   mov     %%r14, %%rbx                                     \n" // Bc
			"   mov     %%rsi, %%rdx                                     \n" // K
			"   vmovups        (%%rbx), %%zmm4                           \n" // B0-15
			"   vmovups     64(%%rbx), %%zmm5                            \n" // B16-31
			"   vpxorq         %%zmm8, %%zmm8, %%zmm8                    \n"
			"   vpxorq         %%zmm9, %%zmm9, %%zmm9                    \n"
			"   vpxorq         %%zmm10, %%zmm10, %%zmm10                 \n"
			"   vpxorq         %%zmm11, %%zmm11, %%zmm11                 \n"
			"   vpxorq         %%zmm12, %%zmm12, %%zmm12                 \n"
			"   vpxorq         %%zmm13, %%zmm13, %%zmm13                 \n"
			"   vpxorq         %%zmm14, %%zmm14, %%zmm14                 \n"
			"   vpxorq         %%zmm15, %%zmm15, %%zmm15                 \n"
			"   vbroadcastss    (%%rax), %%zmm0                          \n" // A0
			"   vbroadcastss    4(%%rax), %%zmm1                         \n" // A1
			"   vpxorq         %%zmm16, %%zmm16, %%zmm16                 \n"
			"   vpxorq         %%zmm17, %%zmm17, %%zmm17                 \n"
			"   vpxorq         %%zmm18, %%zmm18, %%zmm18                 \n"
			"   vpxorq         %%zmm19, %%zmm19, %%zmm19                 \n"
			"   vpxorq         %%zmm20, %%zmm20, %%zmm20                 \n"
			"   vpxorq         %%zmm21, %%zmm21, %%zmm21                 \n"
			"   vpxorq         %%zmm22, %%zmm22, %%zmm22                 \n"
			"   vpxorq         %%zmm23, %%zmm23, %%zmm23                 \n"
			"   cmpq    $8, %%rdx                                        \n"
			"   jb      F32_MAIN_M8N32K1                                 \n"
			"   subq    $8, %%rdx                                        \n"

			"F32_MAIN_K_M8N32K8:                                         \n"
			"   f32_kernel_m8n32_1                                       \n"
			"   f32_kernel_m8n32_2                                       \n"
			"   f32_kernel_m8n32_1                                       \n"
			"   f32_kernel_m8n32_2                                       \n"
			"   f32_kernel_m8n32_1                                       \n"
			"   f32_kernel_m8n32_2                                       \n"
			"   f32_kernel_m8n32_1                                       \n"
			"   f32_kernel_m8n32_2                                       \n"
			"   cmp     $0, %%rdx                                        \n"
			"   je      F32_BEGIN_SAVE_M8N32                             \n"
			"   cmpq    $8, %%rdx                                        \n"
			"   jb      F32_MAIN_M8N32K1                                 \n"
			"   subq  $8, %%rdx                                          \n"
			"   jmp   F32_MAIN_K_M8N32K8                                 \n"

			"F32_MAIN_M8N32K1:                                           \n"
			"   f32_kernel_m8n32_1                                       \n"
			"   subq    $1, %%rdx                                        \n"
			"   cmp     $0, %%rdx                                        \n"
			"   je      F32_BEGIN_SAVE_M8N32                             \n"
			"   f32_kernel_m8n32_2                                       \n"
			"   subq    $1, %%rdx                                        \n"
			"   cmp     $0, %%rdx                                        \n"
			"   je      F32_BEGIN_SAVE_M8N32                             \n"
			"   jmp     F32_MAIN_M8N32K1                                 \n"

			"F32_BEGIN_SAVE_M8N32:                                       \n"
			"   mov    %[is_end_gemm], %%r13                             \n"
			"   test      $1, %%r13                                      \n"
			"   jz       F32_SAVE_C_M8N32_2                              \n"
			"   mov      %%rcx, %%r10                                    \n" // C0
			"   leaq     (%%r10, %%r8, 4), %%r11                         \n" // C1
			"   leaq     (%%r11, %%r8, 4), %%r12                         \n" // C2
			"   leaq     (%%r12, %%r8, 4), %%r13                         \n" // C3

			"F32_SAVE_C_M8N32:                                           \n"
			"   f32_save_c_m8n32                                         \n"
			"   imul     $32, %%r15, %%r11                               \n" // temp use %%r11
			"   add      %%r11, %%r9                                     \n"
			"   movq     %%r9, %%rax                                     \n"
			"   jmp      F32_BEGIN_M8N32                                 \n"

			"F32_SAVE_C_M8N32_2:                                         \n"
			"   f32_save_c_m8n32_2                                       \n"
			"   imul     $32, %%r15, %%r11                               \n" // temp use %%r11
			"   add      %%r11, %%r9                                     \n"
			"   movq     %%r9, %%rax                                     \n"
			"   jmp      F32_BEGIN_M8N32                                 \n"

			//-----------------------------------------------------------------

			"F32_BEGIN_M4N32:                                            \n"
			"   cmpq    $0, %%rdi                                        \n"
			"   je      F32_END_N32                                      \n"
			"   mov     %%r14, %%rbx                                     \n" // Bc
			"   mov     %%rsi, %%rdx                                     \n" // K
			"   vmovups        (%%rbx), %%zmm4                           \n" // B0-15
			"   vmovups     64(%%rbx), %%zmm5                            \n" // B16-31
			"   vbroadcastss    (%%rax), %%zmm0                          \n" // A0
			"   vbroadcastss    4(%%rax), %%zmm1                         \n" // A1
			"   vpxorq         %%zmm8, %%zmm8, %%zmm8                    \n"
			"   vpxorq         %%zmm9, %%zmm9, %%zmm9                    \n"
			"   vpxorq         %%zmm10, %%zmm10, %%zmm10                 \n"
			"   vpxorq         %%zmm11, %%zmm11, %%zmm11                 \n"
			"   vpxorq         %%zmm12, %%zmm12, %%zmm12                 \n"
			"   vpxorq         %%zmm13, %%zmm13, %%zmm13                 \n"
			"   vpxorq         %%zmm14, %%zmm14, %%zmm14                 \n"
			"   vpxorq         %%zmm15, %%zmm15, %%zmm15                 \n"
			"   cmpq    $8, %%rdx                                        \n"
			"   jb      F32_MAIN_M4N32K1                                 \n"
			"   subq  $8, %%rdx                                          \n"

			"F32_MAIN_K_M4N32K8:                                         \n" // loop K+=4
			"   f32_kernel_m4n32_1                                       \n"
			"   f32_kernel_m4n32_2                                       \n"
			"   f32_kernel_m4n32_1                                       \n"
			"   f32_kernel_m4n32_2                                       \n"
			"   f32_kernel_m4n32_1                                       \n"
			"   f32_kernel_m4n32_2                                       \n"
			"   f32_kernel_m4n32_1                                       \n"
			"   f32_kernel_m4n32_2                                       \n"
			"   cmp     $0, %%rdx                                        \n"
			"   je      F32_BEGIN_SAVE_M4N32                             \n"
			"   cmpq    $8, %%rdx                                        \n"
			"   jb      F32_MAIN_M4N32K1                                 \n"
			"   subq  $8, %%rdx                                          \n"
			"   jmp   F32_MAIN_K_M4N32K8                                 \n"

			"F32_MAIN_M4N32K1:                                           \n"
			"   f32_kernel_m4n32_1                                       \n"
			"   subq    $1, %%rdx                                        \n"
			"   cmp     $0, %%rdx                                        \n"
			"   je      F32_BEGIN_SAVE_M4N32                             \n"
			"   f32_kernel_m4n32_2                                       \n"
			"   subq    $1, %%rdx                                        \n"
			"   cmp     $0, %%rdx                                        \n"
			"   je      F32_BEGIN_SAVE_M4N32                             \n"
			"   jmp     F32_MAIN_M4N32K1                                 \n"

			"F32_BEGIN_SAVE_M4N32:                                       \n"
			"   mov    %[is_end_gemm], %%r13                             \n"
			"   test     $1, %%r13                                       \n"
			"   jz       F32_SAVE_C_M4N32_2                              \n"
			"   mov      %%rcx, %%r10                                    \n" // C0
			"   leaq     (%%r10, %%r8, 4), %%r11                         \n" // C1
			"   leaq     (%%r11, %%r8, 4), %%r12                         \n" // C2
			"   leaq     (%%r12, %%r8, 4), %%r13                         \n" // C3

			"F32_SAVE_C_M4N32:                                           \n"
			"   cmpq     $3, %%rdi                                       \n"
			"   je       F32_SAVE_C_M3N32                                \n"
			"   cmpq     $2, %%rdi                                       \n"
			"   je       F32_SAVE_C_M2N32                                \n"
			"   cmpq     $1, %%rdi                                       \n"
			"   je       F32_SAVE_C_M1N32                                \n"
			"   f32_save_c_m4n32                                         \n"
			"   imul     $16, %%r15, %%r11                               \n" // temp use %%r11
			"   add      %%r11, %%r9                                     \n"
			"   movq     %%r9, %%rax                                     \n"
			"   jmp      F32_BEGIN_M4N32                                 \n"

			"F32_SAVE_C_M3N32:                                           \n"
			"   vmovups         %%zmm12, (%%r12)                         \n"
			"   vmovups         %%zmm13, 64(%%r12)                       \n"

			"F32_SAVE_C_M2N32:                                           \n"
			"   vmovups         %%zmm10, (%%r11)                         \n"
			"   vmovups         %%zmm11, 64(%%r11)                       \n"

			"F32_SAVE_C_M1N32:                                           \n"
			"   vmovups         %%zmm8, (%%r10)                          \n"
			"   vmovups         %%zmm9, 64(%%r10)                        \n"
			"   jmp      F32_END_N32                                     \n"

			"F32_SAVE_C_M4N32_2:                                         \n"
			"   cmpq     $3, %%rdi                                       \n"
			"   je       F32_SAVE_C_M3N32_2                              \n"
			"   cmpq     $2, %%rdi                                       \n"
			"   je       F32_SAVE_C_M2N32_2                              \n"
			"   cmpq     $1, %%rdi                                       \n"
			"   je       F32_SAVE_C_M1N32_2                              \n"
			"   f32_save_c_m4n32_2                                       \n"
			"   imul     $16, %%r15, %%r11                               \n" // temp use %%r11
			"   add      %%r11, %%r9                                     \n"
			"   movq     %%r9, %%rax                                     \n"
			"   jmp      F32_BEGIN_M4N32                                 \n"

			"F32_SAVE_C_M3N32_2:                                         \n"
			"   vmovups         %%zmm12, 256(%%r10)                      \n"
			"   vmovups         %%zmm13, 320(%%r10)                      \n"

			"F32_SAVE_C_M2N32_2:                                         \n"
			"   vmovups         %%zmm10, 128(%%r10)                      \n"
			"   vmovups         %%zmm11, 192(%%r10)                      \n"

			"F32_SAVE_C_M1N32_2:                                         \n"
			"   vmovups         %%zmm8, (%%r10)                          \n"
			"   vmovups         %%zmm9, 64(%%r10)                        \n"

			"F32_END_N32:                                                \n"

			:
			:
			[C] "m"(C),
			[Cc] "m"(Cc),
			[A] "m"(A),
			[B] "m"(B),
			[M] "m"(M),
			[K] "m"(K),
			[LK] "m"(LK),
			[LN] "m"(LN),
			[Bc] "m"(Bc),
			[is_start_gemm] "m"(is_start_gemm),
			[is_end_gemm] "m"(is_end_gemm)
			: "rax", "rbx", "rcx", "rdx", "rdi", "rsi", "rbp", "r8", "r9", "r10", "r11", "r12",
				"r13", "r14", "r15", "zmm0", "zmm1", "zmm2", "zmm3", "zmm4", "zmm5",
				"zmm6", "zmm7", "zmm8", "zmm9", "zmm10", "zmm11", "zmm12", "zmm13",
				"zmm14", "zmm15", "zmm16", "zmm17", "zmm18", "zmm19", "zmm20", "zmm21",
				"zmm22", "zmm23", "zmm24", "zmm25", "zmm26", "zmm27", "zmm28", "zmm29",
				"zmm30", "zmm31", "memory", "ymm0", "ymm1", "ymm2", "ymm3", "ymm4", "ymm5", "ymm6", "ymm7",
				"xmm0", "xmm1", "xmm2", "xmm3", "xmm4", "xmm5", "xmm6", "xmm7", "k1");
}

static void FLASHGEMM_F32_KERNELm12xn16_edge(float *C, float *Cc, float *A, float *B, long M, long K, long LK, long LN, float *Bc, long nr, bool is_start_gemm, bool is_end_gemm)
{
	asm volatile(
			".macro    f32_kernel_m12n16_pack_1                          \n"
			"   vbroadcastss    8(%%rax), %%zmm2                         \n" // A2
			"   vfmadd231ps        %%zmm0, %%zmm4, %%zmm8                \n" // A0*(B0-15)

			"   vbroadcastss    12(%%rax), %%zmm3                        \n" // A3
			"   vfmadd231ps        %%zmm1, %%zmm4, %%zmm10               \n" // A1*(B0-15)

			"   prefetcht0         256(%%rax)                            \n"

			"   vbroadcastss    16(%%rax), %%zmm0                        \n" // A4
			"   vfmadd231ps        %%zmm2, %%zmm4, %%zmm12               \n"

			"   prefetcht2         64(%%rbx)                             \n"

			"   vbroadcastss    20(%%rax), %%zmm1                        \n" // A5
			"   vfmadd231ps        %%zmm3, %%zmm4, %%zmm14               \n"

			"   vbroadcastss    24(%%rax), %%zmm2                        \n" // A6
			"   vfmadd231ps        %%zmm0, %%zmm4, %%zmm16               \n"

			"   vbroadcastss    28(%%rax), %%zmm3                        \n" // A7
			"   vfmadd231ps        %%zmm1, %%zmm4, %%zmm18               \n"

			"   vbroadcastss    32(%%rax), %%zmm0                        \n" // A8
			"   vfmadd231ps        %%zmm2, %%zmm4, %%zmm20               \n"

			"    leaq      (%%rbx, %%r8, 4), %%rbx                       \n" // B

			"   vbroadcastss    36(%%rax), %%zmm1                        \n" // A9
			"   vfmadd231ps        %%zmm3, %%zmm4, %%zmm22               \n"

			"   vbroadcastss    40(%%rax), %%zmm2                        \n" // A10
			"   vfmadd231ps        %%zmm0, %%zmm4, %%zmm24               \n"

			"   vbroadcastss    44(%%rax), %%zmm3                        \n" // A11
			"   vfmadd231ps        %%zmm1, %%zmm4, %%zmm26               \n"
			"   vmovups         (%%rbx), %%zmm6                          \n" // next B0
			"    addq              $48, %%rax                            \n" // 下一组A(已读12个)

			"   vbroadcastss    (%%rax), %%zmm0                          \n" // next A0
			"   vfmadd231ps        %%zmm2, %%zmm4, %%zmm28               \n"

			"   vbroadcastss    4(%%rax), %%zmm1                         \n" // next A1
			"   vfmadd231ps        %%zmm3, %%zmm4, %%zmm30               \n"
			"   vmovups         %%zmm4, (%%rbp)                          \n" // pack B0 to Bc
			"    addq              $64, %%rbp                            \n"

			".endm                                                       \n"

			".macro    f32_kernel_m12n16_pack_2                          \n"

			"   vbroadcastss    8(%%rax), %%zmm2                         \n" // next A2
			"   vfmadd231ps        %%zmm0, %%zmm6, %%zmm8                \n"

			"   prefetcht0         256(%%rax)                            \n"

			"   vbroadcastss    12(%%rax), %%zmm3                        \n" // next A3
			"   vfmadd231ps        %%zmm1, %%zmm6, %%zmm10               \n"

			"   prefetcht2         64(%%rbx)                             \n"

			"   vbroadcastss    16(%%rax), %%zmm0                        \n" // next A4
			"   vfmadd231ps        %%zmm2, %%zmm6, %%zmm12               \n"

			"   vbroadcastss    20(%%rax), %%zmm1                        \n" // next A5
			"   vfmadd231ps        %%zmm3, %%zmm6, %%zmm14               \n"

			"   vbroadcastss    24(%%rax), %%zmm2                        \n" // next A6
			"   vfmadd231ps        %%zmm0, %%zmm6, %%zmm16               \n"

			"   vbroadcastss    28(%%rax), %%zmm3                        \n" // next A7
			"   vfmadd231ps        %%zmm1, %%zmm6, %%zmm18               \n"

			"   vbroadcastss    32(%%rax), %%zmm0                        \n" // next A8
			"   vfmadd231ps        %%zmm2, %%zmm6, %%zmm20               \n"

			"    leaq      (%%rbx, %%r8, 4), %%rbx                       \n" // B

			"   vbroadcastss    36(%%rax), %%zmm1                        \n" // next A9
			"   vfmadd231ps        %%zmm3, %%zmm6, %%zmm22               \n"

			"   vbroadcastss    40(%%rax), %%zmm2                        \n" // next A10
			"   vfmadd231ps        %%zmm0, %%zmm6, %%zmm24               \n"

			"   vbroadcastss    44(%%rax), %%zmm3                        \n" // next A11
			"   vfmadd231ps        %%zmm1, %%zmm6, %%zmm26               \n"
			"   vmovups         (%%rbx), %%zmm4                          \n" // next next B0
			"    addq              $48, %%rax                            \n" // 下一组A(已读12个)

			"   vbroadcastss    (%%rax), %%zmm0                          \n" // next next A0
			"   vfmadd231ps        %%zmm2, %%zmm6, %%zmm28               \n"

			"   vbroadcastss    4(%%rax), %%zmm1                         \n" // next next A1
			"   vfmadd231ps        %%zmm3, %%zmm6, %%zmm30               \n"
			"   vmovups         %%zmm6, (%%rbp)                          \n" // pack B0 to Bc
			"    addq              $64, %%rbp                            \n"

			".endm                                                       \n"

			".macro    f32_kernel_m12n16_pack_1_end                      \n"

			"   vbroadcastss    8(%%rax), %%zmm2                         \n" // A2
			"   vfmadd231ps        %%zmm0, %%zmm4, %%zmm8                \n" // A0*(B0-15)

			"   vbroadcastss    12(%%rax), %%zmm3                        \n" // A3
			"   vfmadd231ps        %%zmm1, %%zmm4, %%zmm10               \n" // A1*(B0-15)

			"   prefetcht0         256(%%rax)                            \n"

			"   vbroadcastss    16(%%rax), %%zmm0                        \n" // A4
			"   vfmadd231ps        %%zmm2, %%zmm4, %%zmm12               \n"

			"   prefetcht2         64(%%rbx)                             \n"

			"   vbroadcastss    20(%%rax), %%zmm1                        \n" // A5
			"   vfmadd231ps        %%zmm3, %%zmm4, %%zmm14               \n"

			"   vbroadcastss    24(%%rax), %%zmm2                        \n" // A6
			"   vfmadd231ps        %%zmm0, %%zmm4, %%zmm16               \n"

			"   vbroadcastss    28(%%rax), %%zmm3                        \n" // A7
			"   vfmadd231ps        %%zmm1, %%zmm4, %%zmm18               \n"

			"   vbroadcastss    32(%%rax), %%zmm0                        \n" // A8
			"   vfmadd231ps        %%zmm2, %%zmm4, %%zmm20               \n"

			"   vbroadcastss    36(%%rax), %%zmm1                        \n" // A9
			"   vfmadd231ps        %%zmm3, %%zmm4, %%zmm22               \n"

			"   vbroadcastss    40(%%rax), %%zmm2                        \n" // A10
			"   vfmadd231ps        %%zmm0, %%zmm4, %%zmm24               \n"

			"   vbroadcastss    44(%%rax), %%zmm3                        \n" // A11
			"   vfmadd231ps        %%zmm1, %%zmm4, %%zmm26               \n"
			"    addq              $48, %%rax                            \n" // 下一组A(已读12个)

			"   vbroadcastss    (%%rax), %%zmm0                          \n" // next A0
			"   vfmadd231ps        %%zmm2, %%zmm4, %%zmm28               \n"

			"   vbroadcastss    4(%%rax), %%zmm1                         \n" // next A1
			"   vfmadd231ps        %%zmm3, %%zmm4, %%zmm30               \n"
			"   vmovups         %%zmm4, (%%rbp)                          \n" // pack B0 to Bc

			".endm                                                       \n"

			".macro    f32_kernel_m12n16_pack_2_end                      \n"

			"   vbroadcastss    8(%%rax), %%zmm2                         \n" // next A2
			"   vfmadd231ps        %%zmm0, %%zmm6, %%zmm8                \n"

			"   prefetcht0         256(%%rax)                            \n"

			"   vbroadcastss    12(%%rax), %%zmm3                        \n" // next A3
			"   vfmadd231ps        %%zmm1, %%zmm6, %%zmm10               \n"

			"   prefetcht2         64(%%rbx)                             \n"

			"   vbroadcastss    16(%%rax), %%zmm0                        \n" // next A4
			"   vfmadd231ps        %%zmm2, %%zmm6, %%zmm12               \n"

			"   vbroadcastss    20(%%rax), %%zmm1                        \n" // next A5
			"   vfmadd231ps        %%zmm3, %%zmm6, %%zmm14               \n"

			"   vbroadcastss    24(%%rax), %%zmm2                        \n" // next A6
			"   vfmadd231ps        %%zmm0, %%zmm6, %%zmm16               \n"

			"   vbroadcastss    28(%%rax), %%zmm3                        \n" // next A7
			"   vfmadd231ps        %%zmm1, %%zmm6, %%zmm18               \n"

			"   vbroadcastss    32(%%rax), %%zmm0                        \n" // next A8
			"   vfmadd231ps        %%zmm2, %%zmm6, %%zmm20               \n"

			"   vbroadcastss    36(%%rax), %%zmm1                        \n" // next A9
			"   vfmadd231ps        %%zmm3, %%zmm6, %%zmm22               \n"

			"   vbroadcastss    40(%%rax), %%zmm2                        \n" // next A10
			"   vfmadd231ps        %%zmm0, %%zmm6, %%zmm24               \n"

			"   vbroadcastss    44(%%rax), %%zmm3                        \n" // next A11
			"   vfmadd231ps        %%zmm1, %%zmm6, %%zmm26               \n"
			"    addq              $48, %%rax                            \n" // 下一组A(已读12个)

			"   vbroadcastss    (%%rax), %%zmm0                          \n" // next next A0
			"   vfmadd231ps        %%zmm2, %%zmm6, %%zmm28               \n"

			"   vbroadcastss    4(%%rax), %%zmm1                         \n" // next next A1
			"   vfmadd231ps        %%zmm3, %%zmm6, %%zmm30               \n"
			"   vmovups         %%zmm6, (%%rbp)                          \n" // pack B0 to Bc

			".endm                                                       \n"

			// ---------------------------------------------------------------

			".macro    f32_kernel_m12n16_1                               \n"
			"   vbroadcastss    8(%%rax), %%zmm2                         \n"
			"   vfmadd231ps        %%zmm0, %%zmm4, %%zmm8                \n"

			"   vbroadcastss    12(%%rax), %%zmm3                        \n"
			"   vfmadd231ps        %%zmm1, %%zmm4, %%zmm10               \n"

			"   prefetcht0         256(%%rax)                            \n"

			"   vbroadcastss    16(%%rax), %%zmm0                        \n"
			"   vfmadd231ps        %%zmm2, %%zmm4, %%zmm12               \n"

			"   vbroadcastss    20(%%rax), %%zmm1                        \n"
			"   vfmadd231ps        %%zmm3, %%zmm4, %%zmm14               \n"

			"   vbroadcastss    24(%%rax), %%zmm2                        \n"
			"   vfmadd231ps        %%zmm0, %%zmm4, %%zmm16               \n"

			"   vbroadcastss    28(%%rax), %%zmm3                        \n"
			"   vfmadd231ps        %%zmm1, %%zmm4, %%zmm18               \n"

			"   vbroadcastss    32(%%rax), %%zmm0                        \n"
			"   vfmadd231ps        %%zmm2, %%zmm4, %%zmm20               \n"

			"   addq              $64, %%rbx                             \n"
			"   vmovups           (%%rbx), %%zmm6                        \n"

			"   vbroadcastss    36(%%rax), %%zmm1                        \n"
			"   vfmadd231ps        %%zmm3, %%zmm4, %%zmm22               \n"

			"   vbroadcastss    40(%%rax), %%zmm2                        \n"
			"   vfmadd231ps        %%zmm0, %%zmm4, %%zmm24               \n"

			"   vbroadcastss    44(%%rax), %%zmm3                        \n"
			"   vfmadd231ps        %%zmm1, %%zmm4, %%zmm26               \n"

			"   addq              $48, %%rax                             \n"

			"   vbroadcastss    (%%rax), %%zmm0                          \n"
			"   vfmadd231ps        %%zmm2, %%zmm4, %%zmm28               \n"

			"   vbroadcastss    4(%%rax), %%zmm1                         \n"
			"   vfmadd231ps        %%zmm3, %%zmm4, %%zmm30               \n"
			".endm                                                       \n"

			".macro    f32_kernel_m12n16_2                               \n"
			"   vbroadcastss    8(%%rax), %%zmm2                         \n"
			"   vfmadd231ps        %%zmm0, %%zmm6, %%zmm8                \n"

			"   vbroadcastss    12(%%rax), %%zmm3                        \n"
			"   vfmadd231ps        %%zmm1, %%zmm6, %%zmm10               \n"

			"   prefetcht0         256(%%rax)                            \n"

			"   vbroadcastss    16(%%rax), %%zmm0                        \n"
			"   vfmadd231ps        %%zmm2, %%zmm6, %%zmm12               \n"

			"   vbroadcastss    20(%%rax), %%zmm1                        \n"
			"   vfmadd231ps        %%zmm3, %%zmm6, %%zmm14               \n"

			"   vbroadcastss    24(%%rax), %%zmm2                        \n"
			"   vfmadd231ps        %%zmm0, %%zmm6, %%zmm16               \n"

			"   vbroadcastss    28(%%rax), %%zmm3                        \n"
			"   vfmadd231ps        %%zmm1, %%zmm6, %%zmm18               \n"

			"   vbroadcastss    32(%%rax), %%zmm0                        \n"
			"   vfmadd231ps        %%zmm2, %%zmm6, %%zmm20               \n"

			"   addq              $64, %%rbx                             \n"
			"   vmovups           (%%rbx), %%zmm4                        \n"

			"   vbroadcastss    36(%%rax), %%zmm1                        \n"
			"   vfmadd231ps        %%zmm3, %%zmm6, %%zmm22               \n"

			"   vbroadcastss    40(%%rax), %%zmm2                        \n"
			"   vfmadd231ps        %%zmm0, %%zmm6, %%zmm24               \n"

			"   vbroadcastss    44(%%rax), %%zmm3                        \n"
			"   vfmadd231ps        %%zmm1, %%zmm6, %%zmm26               \n"

			"   addq              $48, %%rax                             \n"

			"   vbroadcastss    (%%rax), %%zmm0                          \n"
			"   vfmadd231ps        %%zmm2, %%zmm6, %%zmm28               \n"

			"   vbroadcastss    4(%%rax), %%zmm1                         \n"
			"   vfmadd231ps        %%zmm3, %%zmm6, %%zmm30               \n"
			".endm                                                       \n"

			".macro    f32_kernel_m12n16_end                             \n"
			"   vbroadcastss    8(%%rax), %%zmm2                         \n"
			"   vfmadd231ps        %%zmm0, %%zmm6, %%zmm8                \n"

			"   vbroadcastss    12(%%rax), %%zmm3                        \n"
			"   vfmadd231ps        %%zmm1, %%zmm6, %%zmm10               \n"

			"   prefetcht0         256(%%rax)                            \n"

			"   vbroadcastss    16(%%rax), %%zmm0                        \n"
			"   vfmadd231ps        %%zmm2, %%zmm6, %%zmm12               \n"

			"   vbroadcastss    20(%%rax), %%zmm1                        \n"
			"   vfmadd231ps        %%zmm3, %%zmm6, %%zmm14               \n"

			"   vbroadcastss    24(%%rax), %%zmm2                        \n"
			"   vfmadd231ps        %%zmm0, %%zmm6, %%zmm16               \n"

			"   vbroadcastss    28(%%rax), %%zmm3                        \n"
			"   vfmadd231ps        %%zmm1, %%zmm6, %%zmm18               \n"

			"   vbroadcastss    32(%%rax), %%zmm0                        \n"
			"   vfmadd231ps        %%zmm2, %%zmm6, %%zmm20               \n"

			"   vbroadcastss    36(%%rax), %%zmm1                        \n"
			"   vfmadd231ps        %%zmm3, %%zmm6, %%zmm22               \n"

			"   vbroadcastss    40(%%rax), %%zmm2                        \n"
			"   vfmadd231ps        %%zmm0, %%zmm6, %%zmm24               \n"

			"   vbroadcastss    44(%%rax), %%zmm3                        \n"
			"   vfmadd231ps        %%zmm1, %%zmm6, %%zmm26               \n"
			"   addq               $48, %%rax                            \n"

			"   vfmadd231ps        %%zmm2, %%zmm6, %%zmm28               \n"
			"   vfmadd231ps        %%zmm3, %%zmm6, %%zmm30               \n"
			".endm                                                       \n"

			".macro    f32_save_c_m12n16                                 \n"
			"   vmovups         %%zmm8, (%%r10)%{%%k1%}                  \n"
			"   vmovups         %%zmm10, (%%r11)%{%%k1%}                 \n"
			"   vmovups         %%zmm12, (%%r12)%{%%k1%}                 \n"
			"   vmovups         %%zmm14, (%%r13)%{%%k1%}                 \n"

			"   leaq  (%%r13, %%r8, 4), %%r10                            \n" // C0
			"   leaq     (%%r10, %%r8, 4), %%r11                         \n" // C1
			"   leaq     (%%r11, %%r8, 4), %%r12                         \n" // C2
			"   leaq     (%%r12, %%r8, 4), %%r13                         \n" // C3

			"   vmovups         %%zmm16, (%%r10)%{%%k1%}                 \n"
			"   vmovups         %%zmm18, (%%r11)%{%%k1%}                 \n"
			"   vmovups         %%zmm20, (%%r12)%{%%k1%}                 \n"
			"   vmovups         %%zmm22, (%%r13)%{%%k1%}                 \n"

			"   leaq  (%%r13, %%r8, 4), %%r10                            \n" // C0
			"   leaq     (%%r10, %%r8, 4), %%r11                         \n" // C1
			"   leaq     (%%r11, %%r8, 4), %%r12                         \n" // C2
			"   leaq     (%%r12, %%r8, 4), %%r13                         \n" // C3

			"   vmovups         %%zmm24, (%%r10)%{%%k1%}                 \n"
			"   vmovups         %%zmm26, (%%r11)%{%%k1%}                 \n"
			"   subq             $12, %%rdi                              \n"
			"   vmovups         %%zmm28, (%%r12)%{%%k1%}                 \n"
			"   vmovups         %%zmm30, (%%r13)%{%%k1%}                 \n"

			"   leaq      (%%r13, %%r8, 4), %%rcx                        \n" // C0
			".endm                                                       \n"

			".macro    f32_save_c_m12n16_2                               \n"
			"   vmovups         %%zmm8, (%%r10)%{%%k1%}                  \n"
			"   vmovups         %%zmm10, 64(%%r10)%{%%k1%}               \n"
			"   vmovups         %%zmm12, 128(%%r10)%{%%k1%}              \n"
			"   vmovups         %%zmm14, 192(%%r10)%{%%k1%}              \n"

			"   addq            $256, %%r10                              \n"

			"   vmovups         %%zmm16, (%%r10)%{%%k1%}                 \n"
			"   vmovups         %%zmm18, 64(%%r10)%{%%k1%}               \n"
			"   vmovups         %%zmm20, 128(%%r10)%{%%k1%}              \n"
			"   vmovups         %%zmm22, 192(%%r10)%{%%k1%}              \n"

			"   addq            $256, %%r10                              \n"

			"   vmovups         %%zmm24, (%%r10)%{%%k1%}                 \n"
			"   vmovups         %%zmm26, 64(%%r10)%{%%k1%}               \n"
			"   vmovups         %%zmm28, 128(%%r10)%{%%k1%}              \n"
			"   vmovups         %%zmm30, 192(%%r10)%{%%k1%}              \n"

			"   addq            $256, %%r10                              \n"
			"   subq            $12, %%rdi                               \n"
			".endm                                                       \n"

			//-----------------------------------------------------------------

			".macro    f32_kernel_m8n16_1                                \n"
			"   vbroadcastss    8(%%rax), %%zmm2                         \n"
			"   vfmadd231ps        %%zmm0, %%zmm4, %%zmm8                \n"

			"   vbroadcastss    12(%%rax), %%zmm3                        \n"
			"   vfmadd231ps        %%zmm1, %%zmm4, %%zmm10               \n"

			"   prefetcht0      256(%%rax)                               \n"
			"   addq             $64, %%rbx                              \n"

			"   vbroadcastss    16(%%rax), %%zmm0                        \n"
			"   vfmadd231ps        %%zmm2, %%zmm4, %%zmm12               \n"

			"   vmovups          (%%rbx), %%zmm6                         \n"

			"   vbroadcastss    20(%%rax), %%zmm1                        \n"
			"   vfmadd231ps        %%zmm3, %%zmm4, %%zmm14               \n"

			"   vbroadcastss    24(%%rax), %%zmm2                        \n"
			"   vfmadd231ps        %%zmm0, %%zmm4, %%zmm16               \n"

			"   vbroadcastss    28(%%rax), %%zmm3                        \n"
			"   vfmadd231ps        %%zmm1, %%zmm4, %%zmm18               \n"

			"   addq              $32, %%rax                             \n"
			"   vbroadcastss     (%%rax), %%zmm0                         \n"
			"   vfmadd231ps        %%zmm2, %%zmm4, %%zmm20               \n"

			"   vbroadcastss     4(%%rax), %%zmm1                        \n"
			"   vfmadd231ps        %%zmm3, %%zmm4, %%zmm22               \n"
			".endm                                                       \n"

			".macro    f32_kernel_m8n16_2                                \n"
			"   vbroadcastss    8(%%rax), %%zmm2                         \n"
			"   vfmadd231ps        %%zmm0, %%zmm6, %%zmm8                \n"

			"   vbroadcastss    12(%%rax), %%zmm3                        \n"
			"   vfmadd231ps        %%zmm1, %%zmm6, %%zmm10               \n"

			"   prefetcht0         256(%%rax)                            \n"
			"   addq              $64, %%rbx                             \n"

			"   vbroadcastss    16(%%rax), %%zmm0                        \n"
			"   vfmadd231ps        %%zmm2, %%zmm6, %%zmm12               \n"

			"   vbroadcastss    20(%%rax), %%zmm1                        \n"
			"   vfmadd231ps        %%zmm3, %%zmm6, %%zmm14               \n"

			"   vbroadcastss    24(%%rax), %%zmm2                        \n"
			"   vfmadd231ps        %%zmm0, %%zmm6, %%zmm16               \n"

			"   vmovups         (%%rbx), %%zmm4                          \n"

			"   vbroadcastss    28(%%rax), %%zmm3                        \n"
			"   vfmadd231ps        %%zmm1, %%zmm6, %%zmm18               \n"

			"   addq             $32, %%rax                              \n"

			"   vbroadcastss    (%%rax), %%zmm0                          \n"
			"   vfmadd231ps        %%zmm2, %%zmm6, %%zmm20               \n"

			"   vbroadcastss    4(%%rax), %%zmm1                         \n"
			"   vfmadd231ps        %%zmm3, %%zmm6, %%zmm22               \n"
			".endm                                                       \n"

			".macro    f32_kernel_m8n16_end                              \n"
			"   vbroadcastss    8(%%rax), %%zmm2                         \n"
			"   vfmadd231ps        %%zmm0, %%zmm6, %%zmm8                \n"

			"   vbroadcastss    12(%%rax), %%zmm3                        \n"
			"   vfmadd231ps        %%zmm1, %%zmm6, %%zmm10               \n"

			"   prefetcht0         256(%%rax)                            \n"

			"   vbroadcastss    16(%%rax), %%zmm0                        \n"
			"   vfmadd231ps        %%zmm2, %%zmm6, %%zmm12               \n"

			"   vbroadcastss    20(%%rax), %%zmm1                        \n"
			"   vfmadd231ps        %%zmm3, %%zmm6, %%zmm14               \n"

			"   vbroadcastss    24(%%rax), %%zmm2                        \n"
			"   vfmadd231ps        %%zmm0, %%zmm6, %%zmm16               \n"

			"   vbroadcastss    28(%%rax), %%zmm3                        \n"
			"   vfmadd231ps        %%zmm1, %%zmm6, %%zmm18               \n"

			"   addq             $32, %%rax                              \n"

			"   vfmadd231ps        %%zmm2, %%zmm6, %%zmm20               \n"
			"   vfmadd231ps        %%zmm3, %%zmm6, %%zmm22               \n"
			".endm                                                       \n"

			".macro    f32_save_c_m8n16                                  \n"
			"   vmovups         %%zmm8, (%%r10)%{%%k1%}                  \n"
			"   vmovups         %%zmm10, (%%r11)%{%%k1%}                 \n"
			"   vmovups         %%zmm12, (%%r12)%{%%k1%}                 \n"
			"   vmovups         %%zmm14, (%%r13)%{%%k1%}                 \n"

			"   leaq     (%%r13, %%r8, 4), %%r10                         \n" // C0
			"   leaq     (%%r10, %%r8, 4), %%r11                         \n" // C1
			"   leaq     (%%r11, %%r8, 4), %%r12                         \n" // C2
			"   leaq     (%%r12, %%r8, 4), %%r13                         \n" // C3

			"   vmovups         %%zmm16, (%%r10)%{%%k1%}                 \n"
			"   vmovups         %%zmm18, (%%r11)%{%%k1%}                 \n"
			"   vmovups         %%zmm20, (%%r12)%{%%k1%}                 \n"
			"   vmovups         %%zmm22, (%%r13)%{%%k1%}                 \n"

			"   subq            $8, %%rdi                                \n"
			"   leaq      (%%r13, %%r8, 4), %%rcx                        \n" // C0
			".endm                                                       \n"

			".macro    f32_save_c_m8n16_2                                \n"
			"   vmovups         %%zmm8, (%%r10)%{%%k1%}                  \n"
			"   vmovups         %%zmm10, 64(%%r10)%{%%k1%}               \n"
			"   vmovups         %%zmm12, 128(%%r10)%{%%k1%}              \n"
			"   vmovups         %%zmm14, 192(%%r10)%{%%k1%}              \n"

			"   addq            $256, %%r10                              \n"

			"   vmovups         %%zmm16, (%%r10)%{%%k1%}                 \n"
			"   vmovups         %%zmm18, 64(%%r10)%{%%k1%}               \n"
			"   vmovups         %%zmm20, 128(%%r10)%{%%k1%}              \n"
			"   vmovups         %%zmm22, 192(%%r10)%{%%k1%}              \n"

			"   addq            $256, %%r10                              \n"
			"   subq            $8, %%rdi                                \n"
			".endm                                                       \n"

			//-----------------------------------------------------------------

			".macro    f32_kernel_m4n16_1                                \n"
			"   vbroadcastss    8(%%rax), %%zmm2                         \n"
			"   vfmadd231ps        %%zmm0, %%zmm4, %%zmm8                \n"
			"   addq             $64, %%rbx                              \n"
			"   vmovups          (%%rbx), %%zmm6                         \n"

			"   vbroadcastss    12(%%rax), %%zmm3                        \n"
			"   vfmadd231ps        %%zmm1, %%zmm4, %%zmm10               \n"

			"   prefetcht0      256(%%rax)                               \n"
			"   addq              $16, %%rax                             \n"

			"   vbroadcastss     (%%rax), %%zmm0                         \n"
			"   vfmadd231ps        %%zmm2, %%zmm4, %%zmm12               \n"

			"   vbroadcastss     4(%%rax), %%zmm1                        \n"
			"   vfmadd231ps        %%zmm3, %%zmm4, %%zmm14               \n"
			".endm                                                       \n"

			".macro    f32_kernel_m4n16_2                                \n"
			"   vbroadcastss     8(%%rax), %%zmm2                        \n"
			"   vfmadd231ps        %%zmm0, %%zmm6, %%zmm8                \n"

			"   addq             $64, %%rbx                              \n"
			"   vmovups          (%%rbx), %%zmm4                         \n"

			"   vbroadcastss     12(%%rax), %%zmm3                       \n"
			"   vfmadd231ps        %%zmm1, %%zmm6, %%zmm10               \n"

			"   prefetcht0         256(%%rax)                            \n"
			"   addq             $16, %%rax                              \n"

			"   vbroadcastss     (%%rax), %%zmm0                         \n"
			"   vfmadd231ps        %%zmm2, %%zmm6, %%zmm12               \n"

			"   vbroadcastss     4(%%rax), %%zmm1                        \n"
			"   vfmadd231ps        %%zmm3, %%zmm6, %%zmm14               \n"
			".endm                                                       \n"

			".macro    f32_kernel_m4n16_end                              \n"
			"   vbroadcastss    8(%%rax), %%zmm2                         \n"
			"   vfmadd231ps        %%zmm0, %%zmm6, %%zmm8                \n"
			"   vfmadd231ps        %%zmm0, %%zmm7, %%zmm9                \n"

			"   vbroadcastss    12(%%rax), %%zmm3                        \n"
			"   vfmadd231ps        %%zmm1, %%zmm6, %%zmm10               \n"
			"   vfmadd231ps        %%zmm1, %%zmm7, %%zmm11               \n"

			"   prefetcht0         256(%%rax)                            \n"

			"   addq             $16, %%rax                              \n"
			"   vbroadcastss     (%%rax), %%zmm0                         \n"
			"   vfmadd231ps        %%zmm2, %%zmm6, %%zmm12               \n"
			"   vfmadd231ps        %%zmm2, %%zmm7, %%zmm13               \n"

			"   vbroadcastss     4(%%rax), %%zmm1                        \n"
			"   vfmadd231ps        %%zmm3, %%zmm6, %%zmm14               \n"
			"   vfmadd231ps        %%zmm3, %%zmm7, %%zmm15               \n"
			".endm                                                       \n"

			".macro    f32_save_c_m4n16                                  \n"
			"   vmovups         %%zmm8, (%%r10)%{%%k1%}                  \n"
			"   vmovups         %%zmm10, (%%r11)%{%%k1%}                 \n"
			"   vmovups         %%zmm12, (%%r12)%{%%k1%}                 \n"
			"   vmovups         %%zmm14, (%%r13)%{%%k1%}                 \n"

			"   subq            $4, %%rdi                                \n"
			"   leaq      (%%r13, %%r8, 4), %%rcx                        \n" // C0
			".endm                                                       \n"

			".macro    f32_save_c_m4n16_2                                \n"
			"   vmovups         %%zmm8, (%%r10)%{%%k1%}                  \n"
			"   vmovups         %%zmm10, 64(%%r10)%{%%k1%}               \n"
			"   vmovups         %%zmm12, 128(%%r10)%{%%k1%}              \n"
			"   vmovups         %%zmm14, 192(%%r10)%{%%k1%}              \n"

			"   addq            $256, %%r10                              \n"
			"   subq            $4, %%rdi                                \n"
			".endm                                                       \n"

			//-----------------------------------------------------------------
			//-----------------------------------------------------------------

			"GEMM_F32_N16:                                               \n"
			"   mov    %[nr], %%rdx                                      \n"
			"   mov    %%dl, %%cl                                        \n"
			"   mov    $1, %%eax                                         \n"
			"   shl    %%cl, %%eax                                       \n"
			"   dec    %%eax                                             \n"
			"   kmovw  %%eax, %%k1                                       \n"

			"   mov     %[C], %%rcx                                      \n"
			"   mov     %[Cc], %%r10                                     \n"
			"   mov     %[A], %%rax                                      \n"
			"   mov     %[B], %%rbx                                      \n"

			"   prefetcht0         (%%rax)                               \n"

			"   mov     %[K], %%rdx                                      \n"
			"   mov     %[LN], %%r8                                      \n"
			"   mov     %[Bc], %%r14                                     \n"
			"   mov     %[M], %%rdi                                      \n"

			"   mov     %[LK], %%r15                                     \n"
			"   mov     %%rax, %%r9                                      \n"

			"   prefetcht0         (%%rbx)                               \n"
			"   mov     %%rdx, %%rsi                                     \n"

			"   mov    %[is_start_gemm], %%r12                           \n"
			"   test   $1, %%r12                                         \n"
			"   jz     F32_BEGIN_M12N16                                  \n"

			"   mov    %[nr], %%rdx                                      \n"

			//-----------------------------------------------------------------

			"F32_BEGIN_PACK_N16:                                         \n"
			"   mov     %%rsi, %%rdx                                     \n" // K
			"   mov     %%r14, %%rbp                                     \n" // Bc

			"   vmovups (%%rbx), %%zmm4                                  \n"

			"   vpxorq         %%zmm8, %%zmm8, %%zmm8                    \n"
			"   vpxorq         %%zmm10, %%zmm10, %%zmm10                 \n"
			"   vpxorq         %%zmm12, %%zmm12, %%zmm12                 \n"
			"   vpxorq         %%zmm14, %%zmm14, %%zmm14                 \n"
			"   vpxorq         %%zmm16, %%zmm16, %%zmm16                 \n"
			"   vpxorq         %%zmm18, %%zmm18, %%zmm18                 \n"
			"   vbroadcastss    (%%rax), %%zmm0                          \n" // A0
			"   vbroadcastss    4(%%rax), %%zmm1                         \n" // A1
			"   vpxorq         %%zmm20, %%zmm20, %%zmm20                 \n"
			"   vpxorq         %%zmm22, %%zmm22, %%zmm22                 \n"
			"   vpxorq         %%zmm24, %%zmm24, %%zmm24                 \n"
			"   vpxorq         %%zmm26, %%zmm26, %%zmm26                 \n"
			"   vpxorq         %%zmm28, %%zmm28, %%zmm28                 \n"
			"   vpxorq         %%zmm30, %%zmm30, %%zmm30                 \n"
			"   cmp     $8, %%rdx                                        \n"
			"   jb      F32_PACK_MAIN_M12N16K1                           \n"
			"   subq    $8, %%rdx                                        \n"

			"F32_PACK_MAIN_M12N16K8:                                     \n"
			"   f32_kernel_m12n16_pack_1                                 \n"
			"   f32_kernel_m12n16_pack_2                                 \n"
			"   f32_kernel_m12n16_pack_1                                 \n"
			"   f32_kernel_m12n16_pack_2                                 \n"
			"   f32_kernel_m12n16_pack_1                                 \n"
			"   f32_kernel_m12n16_pack_2                                 \n"
			"   f32_kernel_m12n16_pack_1                                 \n"
			"   cmp     $0, %%rdx                                        \n"
			"   je      F32_N16_PACK_MAIN_END_1                          \n"
			"   f32_kernel_m12n16_pack_2                                 \n"
			"   cmp     $8, %%rdx                                        \n"
			"   jb      F32_PACK_MAIN_M12N16K1                           \n"
			"   subq    $8, %%rdx                                        \n"
			"   jmp     F32_PACK_MAIN_M12N16K8                           \n"

			"F32_PACK_MAIN_M12N16K1:                                     \n"
			"   cmp     $1, %%rdx                                        \n"
			"   je      F32_N16_PACK_MAIN_END_2                          \n"
			"   f32_kernel_m12n16_pack_1                                 \n"
			"   subq    $1, %%rdx                                        \n"
			"   cmp     $1, %%rdx                                        \n"
			"   je      F32_N16_PACK_MAIN_END_1                          \n"
			"   f32_kernel_m12n16_pack_2                                 \n"
			"   subq    $1, %%rdx                                        \n"
			"   jmp     F32_PACK_MAIN_M12N16K1                           \n"

			"F32_N16_PACK_MAIN_END_1:                                    \n"
			"   f32_kernel_m12n16_pack_2_end                             \n"
			"   jmp     F32_BEGIN_SAVE_M12N16                            \n"

			"F32_N16_PACK_MAIN_END_2:                                    \n"
			"   f32_kernel_m12n16_pack_1_end                             \n"
			"   jmp     F32_BEGIN_SAVE_M12N16                            \n"

			//-----------------------------------------------------------------

			"F32_BEGIN_M12N16:                                           \n"
			"   cmpq    $12, %%rdi                                       \n"
			"   jb      F32_BEGIN_M8N16                                  \n"

			"   mov     %%r14, %%rbx                                     \n" // Bc
			"   mov     %%rsi, %%rdx                                     \n" // K
			"   vmovups        (%%rbx), %%zmm4                           \n" // B0-15
			"   vpxorq         %%zmm8, %%zmm8, %%zmm8                    \n"
			"   vpxorq         %%zmm10, %%zmm10, %%zmm10                 \n"
			"   vpxorq         %%zmm12, %%zmm12, %%zmm12                 \n"
			"   vpxorq         %%zmm14, %%zmm14, %%zmm14                 \n"
			"   vpxorq         %%zmm16, %%zmm16, %%zmm16                 \n"
			"   vpxorq         %%zmm18, %%zmm18, %%zmm18                 \n"
			"   vbroadcastss    (%%rax), %%zmm0                          \n" // A0
			"   vbroadcastss    4(%%rax), %%zmm1                         \n" // A1
			"   vpxorq         %%zmm20, %%zmm20, %%zmm20                 \n"
			"   vpxorq         %%zmm22, %%zmm22, %%zmm22                 \n"
			"   vpxorq         %%zmm24, %%zmm24, %%zmm24                 \n"
			"   vpxorq         %%zmm26, %%zmm26, %%zmm26                 \n"
			"   vpxorq         %%zmm28, %%zmm28, %%zmm28                 \n"
			"   vpxorq         %%zmm30, %%zmm30, %%zmm30                 \n"
			"   cmp     $8, %%rdx                                        \n"
			"   jb      F32_MAIN_M12N16K1                                \n"
			"   subq    $8, %%rdx                                        \n"

			"F32_MAIN_K_M12N16K8:                                        \n" // loop K+=4
			"   f32_kernel_m12n16_1                                      \n"
			"   f32_kernel_m12n16_2                                      \n"
			"   f32_kernel_m12n16_1                                      \n"
			"   f32_kernel_m12n16_2                                      \n"
			"   f32_kernel_m12n16_1                                      \n"
			"   f32_kernel_m12n16_2                                      \n"
			"   f32_kernel_m12n16_1                                      \n"
			"   f32_kernel_m12n16_2                                      \n"
			"   cmp     $0, %%rdx                                        \n"
			"   je      F32_BEGIN_SAVE_M12N16                            \n"
			"   cmp     $8, %%rdx                                        \n"
			"   jb      F32_MAIN_M12N16K1                                \n"
			"   subq    $8, %%rdx                                        \n"
			"   jmp     F32_MAIN_K_M12N16K8                              \n"

			"F32_MAIN_M12N16K1:                                          \n"
			"   f32_kernel_m12n16_1                                      \n"
			"   subq    $1, %%rdx                                        \n"
			"   cmp     $0, %%rdx                                        \n"
			"   je      F32_BEGIN_SAVE_M12N16                            \n"
			"   f32_kernel_m12n16_2                                      \n"
			"   subq    $1, %%rdx                                        \n"
			"   cmp     $0, %%rdx                                        \n"
			"   je      F32_BEGIN_SAVE_M12N16                            \n"
			"   jmp     F32_MAIN_M12N16K1                                \n"

			"F32_BEGIN_SAVE_M12N16:                                      \n"
			"   mov    %[is_end_gemm], %%r13                             \n"
			"   test      $1, %%r13                                      \n"
			"   jz       F32_SAVE_C_M12N16_2                             \n"
			"   mov      %%rcx, %%r10                                    \n" // C0
			"   leaq     (%%r10, %%r8, 4), %%r11                         \n" // C1
			"   leaq     (%%r11, %%r8, 4), %%r12                         \n" // C2
			"   leaq     (%%r12, %%r8, 4), %%r13                         \n" // C3

			"F32_SAVE_C_M12N16:                                          \n"
			"   f32_save_c_m12n16                                        \n"
			"   imul     $48, %%r15, %%r11                               \n" // temp use %%r11
			"   add      %%r11, %%r9                                     \n"
			"   movq     %%r9, %%rax                                     \n"
			"   jmp     F32_BEGIN_M12N16                                 \n"

			"F32_SAVE_C_M12N16_2:                                        \n"
			"   f32_save_c_m12n16_2                                      \n"
			"   imul     $48, %%r15, %%r11                               \n" // temp use %%r11
			"   add      %%r11, %%r9                                     \n"
			"   movq     %%r9, %%rax                                     \n"
			"   jmp     F32_BEGIN_M12N16                                 \n"

			//-----------------------------------------------------------------

			"F32_BEGIN_M8N16:                                            \n"
			"   cmpq    $8, %%rdi                                        \n"
			"   jb      F32_BEGIN_M4N16                                  \n"

			"   mov     %%r14, %%rbx                                     \n" // Bc
			"   mov     %%rsi, %%rdx                                     \n" // K
			"   vmovups        (%%rbx), %%zmm4                           \n" // B0-15
			"   vpxorq         %%zmm8, %%zmm8, %%zmm8                    \n"
			"   vpxorq         %%zmm10, %%zmm10, %%zmm10                 \n"
			"   vpxorq         %%zmm12, %%zmm12, %%zmm12                 \n"
			"   vpxorq         %%zmm14, %%zmm14, %%zmm14                 \n"
			"   vbroadcastss    (%%rax), %%zmm0                          \n" // A0
			"   vbroadcastss    4(%%rax), %%zmm1                         \n" // A1
			"   vpxorq         %%zmm16, %%zmm16, %%zmm16                 \n"
			"   vpxorq         %%zmm18, %%zmm18, %%zmm18                 \n"
			"   vpxorq         %%zmm20, %%zmm20, %%zmm20                 \n"
			"   vpxorq         %%zmm22, %%zmm22, %%zmm22                 \n"
			"   cmpq    $8, %%rdx                                        \n"
			"   jb      F32_MAIN_M8N16K1                                 \n"
			"   subq    $8, %%rdx                                        \n"

			"F32_MAIN_K_M8N16K8:                                         \n"
			"   f32_kernel_m8n16_1                                       \n"
			"   f32_kernel_m8n16_2                                       \n"
			"   f32_kernel_m8n16_1                                       \n"
			"   f32_kernel_m8n16_2                                       \n"
			"   f32_kernel_m8n16_1                                       \n"
			"   f32_kernel_m8n16_2                                       \n"
			"   f32_kernel_m8n16_1                                       \n"
			"   f32_kernel_m8n16_2                                       \n"
			"   cmp     $0, %%rdx                                        \n"
			"   je      F32_BEGIN_SAVE_M8N16                             \n"
			"   cmpq    $8, %%rdx                                        \n"
			"   jb      F32_MAIN_M8N16K1                                 \n"
			"   subq  $8, %%rdx                                          \n"
			"   jmp   F32_MAIN_K_M8N16K8                                 \n"

			"F32_MAIN_M8N16K1:                                           \n"
			"   f32_kernel_m8n16_1                                       \n"
			"   subq    $1, %%rdx                                        \n"
			"   cmp     $0, %%rdx                                        \n"
			"   je      F32_BEGIN_SAVE_M8N16                             \n"
			"   f32_kernel_m8n16_2                                       \n"
			"   subq    $1, %%rdx                                        \n"
			"   cmp     $0, %%rdx                                        \n"
			"   je      F32_BEGIN_SAVE_M8N16                             \n"
			"   jmp     F32_MAIN_M8N16K1                                 \n"

			"F32_BEGIN_SAVE_M8N16:                                       \n"
			"   mov    %[is_end_gemm], %%r13                             \n"
			"   test      $1, %%r13                                      \n"
			"   jz       F32_SAVE_C_M8N16_2                              \n"
			"   mov      %%rcx, %%r10                                    \n" // C0
			"   leaq     (%%r10, %%r8, 4), %%r11                         \n" // C1
			"   leaq     (%%r11, %%r8, 4), %%r12                         \n" // C2
			"   leaq     (%%r12, %%r8, 4), %%r13                         \n" // C3

			"F32_SAVE_C_M8N16:                                           \n"
			"   f32_save_c_m8n16                                         \n"
			"   imul     $32, %%r15, %%r11                               \n" // temp use %%r11
			"   add      %%r11, %%r9                                     \n"
			"   movq     %%r9, %%rax                                     \n"
			"   jmp      F32_BEGIN_M8N16                                 \n"

			"F32_SAVE_C_M8N16_2:                                         \n"
			"   f32_save_c_m8n16_2                                       \n"
			"   imul     $32, %%r15, %%r11                               \n" // temp use %%r11
			"   add      %%r11, %%r9                                     \n"
			"   movq     %%r9, %%rax                                     \n"
			"   jmp      F32_BEGIN_M8N16                                 \n"

			//-----------------------------------------------------------------

			"F32_BEGIN_M4N16:                                            \n"
			"   cmpq    $0, %%rdi                                        \n"
			"   je      F32_END_N16                                      \n"
			"   mov     %%r14, %%rbx                                     \n" // Bc
			"   mov     %%rsi, %%rdx                                     \n" // K
			"   vmovups        (%%rbx), %%zmm4                           \n" // B0-15
			"   vbroadcastss    (%%rax), %%zmm0                          \n" // A0
			"   vbroadcastss    4(%%rax), %%zmm1                         \n" // A1
			"   vpxorq         %%zmm8, %%zmm8, %%zmm8                    \n"
			"   vpxorq         %%zmm10, %%zmm10, %%zmm10                 \n"
			"   vpxorq         %%zmm12, %%zmm12, %%zmm12                 \n"
			"   vpxorq         %%zmm14, %%zmm14, %%zmm14                 \n"
			"   cmpq    $8, %%rdx                                        \n"
			"   jb      F32_MAIN_M4N16K1                                 \n"
			"   subq  $8, %%rdx                                          \n"

			"F32_MAIN_K_M4N16K8:                                         \n" // loop K+=4
			"   f32_kernel_m4n16_1                                       \n"
			"   f32_kernel_m4n16_2                                       \n"
			"   f32_kernel_m4n16_1                                       \n"
			"   f32_kernel_m4n16_2                                       \n"
			"   f32_kernel_m4n16_1                                       \n"
			"   f32_kernel_m4n16_2                                       \n"
			"   f32_kernel_m4n16_1                                       \n"
			"   f32_kernel_m4n16_2                                       \n"
			"   cmp     $0, %%rdx                                        \n"
			"   je      F32_BEGIN_SAVE_M4N16                             \n"
			"   cmpq    $8, %%rdx                                        \n"
			"   jb      F32_MAIN_M4N16K1                                 \n"
			"   subq  $8, %%rdx                                          \n"
			"   jmp   F32_MAIN_K_M4N16K8                                 \n"

			"F32_MAIN_M4N16K1:                                           \n"
			"   f32_kernel_m4n16_1                                       \n"
			"   subq    $1, %%rdx                                        \n"
			"   cmp     $0, %%rdx                                        \n"
			"   je      F32_BEGIN_SAVE_M4N16                             \n"
			"   f32_kernel_m4n16_2                                       \n"
			"   subq    $1, %%rdx                                        \n"
			"   cmp     $0, %%rdx                                        \n"
			"   je      F32_BEGIN_SAVE_M4N16                             \n"
			"   jmp     F32_MAIN_M4N16K1                                 \n"

			"F32_BEGIN_SAVE_M4N16:                                       \n"
			"   mov    %[is_end_gemm], %%r13                             \n"
			"   test     $1, %%r13                                       \n"
			"   jz       F32_SAVE_C_M4N16_2                              \n"
			"   mov      %%rcx, %%r10                                    \n" // C0
			"   leaq     (%%r10, %%r8, 4), %%r11                         \n" // C1
			"   leaq     (%%r11, %%r8, 4), %%r12                         \n" // C2
			"   leaq     (%%r12, %%r8, 4), %%r13                         \n" // C3

			"F32_SAVE_C_M4N16:                                           \n"
			"   cmpq     $3, %%rdi                                       \n"
			"   je       F32_SAVE_C_M3N16                                \n"
			"   cmpq     $2, %%rdi                                       \n"
			"   je       F32_SAVE_C_M2N16                                \n"
			"   cmpq     $1, %%rdi                                       \n"
			"   je       F32_SAVE_C_M1N16                                \n"
			"   f32_save_c_m4n16                                         \n"
			"   imul     $16, %%r15, %%r11                               \n" // temp use %%r11
			"   add      %%r11, %%r9                                     \n"
			"   movq     %%r9, %%rax                                     \n"
			"   jmp      F32_BEGIN_M4N16                                 \n"

			"F32_SAVE_C_M3N16:                                           \n"
			"   vmovups         %%zmm12, (%%r12)                         \n"
			"   vmovups         %%zmm13, 64(%%r12)                       \n"

			"F32_SAVE_C_M2N16:                                           \n"
			"   vmovups         %%zmm10, (%%r11)                         \n"
			"   vmovups         %%zmm11, 64(%%r11)                       \n"

			"F32_SAVE_C_M1N16:                                           \n"
			"   vmovups         %%zmm8, (%%r10)                          \n"
			"   vmovups         %%zmm9, 64(%%r10)                        \n"
			"   jmp      F32_END_N16                                     \n"

			"F32_SAVE_C_M4N16_2:                                         \n"
			"   cmpq     $3, %%rdi                                       \n"
			"   je       F32_SAVE_C_M3N16_2                              \n"
			"   cmpq     $2, %%rdi                                       \n"
			"   je       F32_SAVE_C_M2N16_2                              \n"
			"   cmpq     $1, %%rdi                                       \n"
			"   je       F32_SAVE_C_M1N16_2                              \n"
			"   f32_save_c_m4n16_2                                       \n"
			"   imul     $16, %%r15, %%r11                               \n" // temp use %%r11
			"   add      %%r11, %%r9                                     \n"
			"   movq     %%r9, %%rax                                     \n"
			"   jmp      F32_BEGIN_M4N16                                 \n"

			"F32_SAVE_C_M3N16_2:                                         \n"
			"   vmovups         %%zmm12, 256(%%r10)                      \n"
			"   vmovups         %%zmm13, 320(%%r10)                      \n"

			"F32_SAVE_C_M2N16_2:                                         \n"
			"   vmovups         %%zmm10, 128(%%r10)                      \n"
			"   vmovups         %%zmm11, 192(%%r10)                      \n"

			"F32_SAVE_C_M1N16_2:                                         \n"
			"   vmovups         %%zmm8, (%%r10)                          \n"
			"   vmovups         %%zmm9, 64(%%r10)                        \n"

			"F32_END_N16:                                                \n"

			:
			:
			[C] "m"(C),
			[Cc] "m"(Cc),
			[A] "m"(A),
			[B] "m"(B),
			[M] "m"(M),
			[K] "m"(K),
			[LK] "m"(LK),
			[LN] "m"(LN),
			[Bc] "m"(Bc),
			[nr] "m"(nr),
			[is_start_gemm] "m"(is_start_gemm),
			[is_end_gemm] "m"(is_end_gemm)
			: "rax", "rbx", "rcx", "rdx", "rdi", "rsi", "rbp", "r8", "r9", "r10", "r11", "r12",
				"r13", "r14", "r15", "zmm0", "zmm1", "zmm2", "zmm3", "zmm4", "zmm5",
				"zmm6", "zmm7", "zmm8", "zmm9", "zmm10", "zmm11", "zmm12", "zmm13",
				"zmm14", "zmm15", "zmm16", "zmm17", "zmm18", "zmm19", "zmm20", "zmm21",
				"zmm22", "zmm23", "zmm24", "zmm25", "zmm26", "zmm27", "zmm28", "zmm29",
				"zmm30", "zmm31", "memory", "xmm0", "xmm1", "xmm2", "xmm3", "xmm6", "xmm7");
}