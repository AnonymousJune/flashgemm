#include <stdint.h>

static void FLASHGEMM_BF16_KERNELm12xn32(float *C, uint16_t *Cc, uint16_t *A, uint16_t *B, long M, long K, long LK, long LN, uint16_t *Bc, bool is_start_gemm, bool is_end_gemm)
{
	asm volatile(
			".macro bf16_pack_b_n32                                      \n"
			"   vpunpcklwd  %%zmm7, %%zmm6, %%zmm4                       \n"
			"   vpunpckhwd  %%zmm7, %%zmm6, %%zmm5                       \n"

			"   vextracti32x4   $0x1, %%zmm4, %%xmm0                     \n"
			"   vextracti32x4   $0x2, %%zmm4, %%xmm1                     \n"
			"   vextracti32x4   $0x3, %%zmm4, %%xmm2                     \n"

			"   vextracti32x4   $0x0, %%zmm5, %%xmm3                     \n"
			"   vextracti32x4   $0x1, %%zmm5, %%xmm6                     \n"
			"   vextracti32x4   $0x2, %%zmm5, %%xmm7                     \n"

			"   vinserti32x4    $0x2, %%xmm0, %%zmm4, %%zmm4             \n"
			"   vinserti32x4    $0x0, %%xmm1, %%zmm5, %%zmm5             \n"
			"   vinserti32x4    $0x2, %%xmm2, %%zmm5, %%zmm5             \n"

			"   vbroadcastss    (%%rax), %%zmm0                          \n" // A0 in zmm0
			"   vbroadcastss    4(%%rax), %%zmm1                         \n" // A1 in zmm1

			"   vinserti32x4    $0x1, %%xmm3, %%zmm4, %%zmm4             \n"
			"   vinserti32x4    $0x3, %%xmm6, %%zmm4, %%zmm4             \n"
			"   vinserti32x4    $0x1, %%xmm7, %%zmm5, %%zmm5             \n"

			"   vmovups         %%zmm4, (%%rbp)                          \n" // store back B
			"   vmovups         %%zmm5, 64(%%rbp)                        \n" // store back B
			"   addq            $128, %%rbp                              \n"
			".endm                                                       \n"

			".macro bf16_kernel_m12n32k2_pack                            \n"
			"   bf16_pack_b_n32                                          \n"

			"   vbroadcastss    8(%%rax), %%zmm2                         \n"
			"   vdpbf16ps        %%zmm0, %%zmm4, %%zmm8                  \n"
			"   vdpbf16ps        %%zmm0, %%zmm5, %%zmm9                  \n"

			"   vbroadcastss    12(%%rax), %%zmm3                        \n"
			"   vdpbf16ps        %%zmm1, %%zmm4, %%zmm10                 \n"
			"   vdpbf16ps        %%zmm1, %%zmm5, %%zmm11                 \n"

			"   prefetcht0         256(%%rax)                            \n"

			"   vbroadcastss    16(%%rax), %%zmm0                        \n"
			"   vdpbf16ps        %%zmm2, %%zmm4, %%zmm12                 \n"
			"   vdpbf16ps        %%zmm2, %%zmm5, %%zmm13                 \n"

			"   vbroadcastss    20(%%rax), %%zmm1                        \n"
			"   vdpbf16ps        %%zmm3, %%zmm4, %%zmm14                 \n"
			"   vdpbf16ps        %%zmm3, %%zmm5, %%zmm15                 \n"

			"   vbroadcastss    24(%%rax), %%zmm2                        \n"
			"   vdpbf16ps        %%zmm0, %%zmm4, %%zmm16                 \n"
			"   vdpbf16ps        %%zmm0, %%zmm5, %%zmm17                 \n"

			"   vbroadcastss    28(%%rax), %%zmm3                        \n"
			"   vdpbf16ps        %%zmm1, %%zmm4, %%zmm18                 \n"
			"   vdpbf16ps        %%zmm1, %%zmm5, %%zmm19                 \n"

			"   vmovdqu16     (%%rbx), %%zmm6                            \n" // load next B
			"   prefetcht2  64(%%rbx)                                    \n"
			"   leaq        (%%rbx, %%r8, 2), %%rbx                      \n"

			"   vbroadcastss    32(%%rax), %%zmm0                        \n"
			"   vdpbf16ps        %%zmm2, %%zmm4, %%zmm20                 \n"
			"   vdpbf16ps        %%zmm2, %%zmm5, %%zmm21                 \n"

			"   vbroadcastss    36(%%rax), %%zmm1                        \n"
			"   vdpbf16ps        %%zmm3, %%zmm4, %%zmm22                 \n"
			"   vdpbf16ps        %%zmm3, %%zmm5, %%zmm23                 \n"

			"   vbroadcastss    40(%%rax), %%zmm2                        \n"
			"   vdpbf16ps        %%zmm0, %%zmm4, %%zmm24                 \n"
			"   vdpbf16ps        %%zmm0, %%zmm5, %%zmm25                 \n"

			"   vmovdqu16         (%%rbx), %%zmm7                        \n" // load next B
			"   prefetcht2  64(%%rbx)                                    \n"
			"   leaq        (%%rbx, %%r8, 2), %%rbx                      \n"

			"   vbroadcastss    44(%%rax), %%zmm3                        \n"
			"   vdpbf16ps        %%zmm1, %%zmm4, %%zmm26                 \n"
			"   vdpbf16ps        %%zmm1, %%zmm5, %%zmm27                 \n"
			"   addq          $48, %%rax                                 \n"

			"   vdpbf16ps        %%zmm2, %%zmm4, %%zmm28                 \n"
			"   vdpbf16ps        %%zmm2, %%zmm5, %%zmm29                 \n"

			"   vdpbf16ps        %%zmm3, %%zmm4, %%zmm30                 \n"
			"   vdpbf16ps        %%zmm3, %%zmm5, %%zmm31                 \n"
			".endm                                                       \n"

			".macro bf16_kernel_m12n32k2_pack_end                        \n" // deference is no prefetch A and B
			"   bf16_pack_b_n32                                          \n"

			"   vbroadcastss    8(%%rax), %%zmm2                         \n"
			"   vdpbf16ps        %%zmm0, %%zmm4, %%zmm8                  \n"
			"   vdpbf16ps        %%zmm0, %%zmm5, %%zmm9                  \n"

			"   vbroadcastss    12(%%rax), %%zmm3                        \n"
			"   vdpbf16ps        %%zmm1, %%zmm4, %%zmm10                 \n"
			"   vdpbf16ps        %%zmm1, %%zmm5, %%zmm11                 \n"

			"   prefetcht0         256(%%rax)                            \n"

			"   vbroadcastss    16(%%rax), %%zmm0                        \n"
			"   vdpbf16ps        %%zmm2, %%zmm4, %%zmm12                 \n"
			"   vdpbf16ps        %%zmm2, %%zmm5, %%zmm13                 \n"

			"   vbroadcastss    20(%%rax), %%zmm1                        \n"
			"   vdpbf16ps        %%zmm3, %%zmm4, %%zmm14                 \n"
			"   vdpbf16ps        %%zmm3, %%zmm5, %%zmm15                 \n"

			"   vbroadcastss    24(%%rax), %%zmm2                        \n"
			"   vdpbf16ps        %%zmm0, %%zmm4, %%zmm16                 \n"
			"   vdpbf16ps        %%zmm0, %%zmm5, %%zmm17                 \n"

			"   vbroadcastss    28(%%rax), %%zmm3                        \n"
			"   vdpbf16ps        %%zmm1, %%zmm4, %%zmm18                 \n"
			"   vdpbf16ps        %%zmm1, %%zmm5, %%zmm19                 \n"

			"   vbroadcastss    32(%%rax), %%zmm0                        \n"
			"   vdpbf16ps        %%zmm2, %%zmm4, %%zmm20                 \n"
			"   vdpbf16ps        %%zmm2, %%zmm5, %%zmm21                 \n"

			"   vbroadcastss    36(%%rax), %%zmm1                        \n"
			"   vdpbf16ps        %%zmm3, %%zmm4, %%zmm22                 \n"
			"   vdpbf16ps        %%zmm3, %%zmm5, %%zmm23                 \n"

			"   vbroadcastss    40(%%rax), %%zmm2                        \n"
			"   vdpbf16ps        %%zmm0, %%zmm4, %%zmm24                 \n"
			"   vdpbf16ps        %%zmm0, %%zmm5, %%zmm25                 \n"

			"   vbroadcastss    44(%%rax), %%zmm3                        \n"
			"   vdpbf16ps        %%zmm1, %%zmm4, %%zmm26                 \n"
			"   vdpbf16ps        %%zmm1, %%zmm5, %%zmm27                 \n"
			"   addq          $48, %%rax                                 \n"

			"   vdpbf16ps        %%zmm2, %%zmm4, %%zmm28                 \n"
			"   vdpbf16ps        %%zmm2, %%zmm5, %%zmm29                 \n"

			"   vdpbf16ps        %%zmm3, %%zmm4, %%zmm30                 \n"
			"   vdpbf16ps        %%zmm3, %%zmm5, %%zmm31                 \n"
			".endm                                                       \n"

			".macro    bf16_kernel_m12n32k2_1                            \n"
			"   vbroadcastss    8(%%rax), %%zmm2                         \n"
			"   vdpbf16ps        %%zmm0, %%zmm4, %%zmm8                  \n"
			"   vdpbf16ps        %%zmm0, %%zmm5, %%zmm9                  \n"

			"   vbroadcastss    12(%%rax), %%zmm3                        \n"

			"   vdpbf16ps        %%zmm1, %%zmm4, %%zmm10                 \n"
			"   vdpbf16ps        %%zmm1, %%zmm5, %%zmm11                 \n"

			"   prefetcht0         256(%%rax)                            \n"

			"   vbroadcastss    16(%%rax), %%zmm0                        \n"
			"   vdpbf16ps        %%zmm2, %%zmm4, %%zmm12                 \n"
			"   vdpbf16ps        %%zmm2, %%zmm5, %%zmm13                 \n"

			"   vbroadcastss    20(%%rax), %%zmm1                        \n"
			"   vdpbf16ps        %%zmm3, %%zmm4, %%zmm14                 \n"
			"   vdpbf16ps        %%zmm3, %%zmm5, %%zmm15                 \n"

			"   vbroadcastss    24(%%rax), %%zmm2                        \n"
			"   vdpbf16ps        %%zmm0, %%zmm4, %%zmm16                 \n"
			"   vdpbf16ps        %%zmm0, %%zmm5, %%zmm17                 \n"

			"   vbroadcastss    28(%%rax), %%zmm3                        \n"
			"   vdpbf16ps        %%zmm1, %%zmm4, %%zmm18                 \n"
			"   vdpbf16ps        %%zmm1, %%zmm5, %%zmm19                 \n"

			"   vbroadcastss    32(%%rax), %%zmm0                        \n"
			"   vdpbf16ps        %%zmm2, %%zmm4, %%zmm20                 \n"
			"   vdpbf16ps        %%zmm2, %%zmm5, %%zmm21                 \n"

			"    addq              $128, %%rbx                           \n"

			"   vbroadcastss    36(%%rax), %%zmm1                        \n"
			"   vdpbf16ps        %%zmm3, %%zmm4, %%zmm22                 \n"
			"   vdpbf16ps        %%zmm3, %%zmm5, %%zmm23                 \n"

			"   vbroadcastss    40(%%rax), %%zmm2                        \n"
			"   vdpbf16ps        %%zmm0, %%zmm4, %%zmm24                 \n"
			"   vdpbf16ps        %%zmm0, %%zmm5, %%zmm25                 \n"

			"   vbroadcastss    44(%%rax), %%zmm3                        \n"
			"   vdpbf16ps        %%zmm1, %%zmm4, %%zmm26                 \n"
			"   vmovdqu16         (%%rbx), %%zmm6                        \n"
			"    addq              $48, %%rax                            \n"
			"   vdpbf16ps        %%zmm1, %%zmm5, %%zmm27                 \n"

			"   vbroadcastss    (%%rax), %%zmm0                          \n"
			"   vdpbf16ps        %%zmm2, %%zmm4, %%zmm28                 \n"
			"   vmovdqu16         64(%%rbx), %%zmm7                      \n"
			"   vdpbf16ps        %%zmm2, %%zmm5, %%zmm29                 \n"

			"   vbroadcastss    4(%%rax), %%zmm1                         \n"
			"   vdpbf16ps        %%zmm3, %%zmm4, %%zmm30                 \n"
			"   vdpbf16ps        %%zmm3, %%zmm5, %%zmm31                 \n"
			".endm                                                       \n"

			".macro    bf16_kernel_m12n32k2_2                            \n"
			"   vbroadcastss    8(%%rax), %%zmm2                         \n"
			"   vdpbf16ps        %%zmm0, %%zmm6, %%zmm8                  \n"
			"   vdpbf16ps        %%zmm0, %%zmm7, %%zmm9                  \n"

			"   vbroadcastss    12(%%rax), %%zmm3                        \n"
			"   vdpbf16ps        %%zmm1, %%zmm6, %%zmm10                 \n"
			"   vdpbf16ps        %%zmm1, %%zmm7, %%zmm11                 \n"

			"   prefetcht0         256(%%rax)                            \n"

			"   vbroadcastss    16(%%rax), %%zmm0                        \n"
			"   vdpbf16ps        %%zmm2, %%zmm6, %%zmm12                 \n"
			"   vdpbf16ps        %%zmm2, %%zmm7, %%zmm13                 \n"

			"   vbroadcastss    20(%%rax), %%zmm1                        \n"
			"   vdpbf16ps        %%zmm3, %%zmm6, %%zmm14                 \n"
			"   vdpbf16ps        %%zmm3, %%zmm7, %%zmm15                 \n"

			"   vbroadcastss    24(%%rax), %%zmm2                        \n"
			"   vdpbf16ps        %%zmm0, %%zmm6, %%zmm16                 \n"
			"   vdpbf16ps        %%zmm0, %%zmm7, %%zmm17                 \n"

			"   vbroadcastss    28(%%rax), %%zmm3                        \n"
			"   vdpbf16ps        %%zmm1, %%zmm6, %%zmm18                 \n"
			"   vdpbf16ps        %%zmm1, %%zmm7, %%zmm19                 \n"

			"   vbroadcastss    32(%%rax), %%zmm0                        \n"
			"   vdpbf16ps        %%zmm2, %%zmm6, %%zmm20                 \n"
			"   vdpbf16ps        %%zmm2, %%zmm7, %%zmm21                 \n"

			"   addq              $128, %%rbx                            \n"

			"   vbroadcastss    36(%%rax), %%zmm1                        \n"
			"   vdpbf16ps        %%zmm3, %%zmm6, %%zmm22                 \n"
			"   vdpbf16ps        %%zmm3, %%zmm7, %%zmm23                 \n"

			"   vbroadcastss    40(%%rax), %%zmm2                        \n"
			"   vdpbf16ps        %%zmm0, %%zmm6, %%zmm24                 \n"
			"   vdpbf16ps        %%zmm0, %%zmm7, %%zmm25                 \n"

			"   vbroadcastss    44(%%rax), %%zmm3                        \n"
			"   vdpbf16ps        %%zmm1, %%zmm6, %%zmm26                 \n"
			"   vmovdqu16         (%%rbx), %%zmm4                        \n"
			"    addq              $48, %%rax                            \n"
			"   vdpbf16ps        %%zmm1, %%zmm7, %%zmm27                 \n"

			"   vbroadcastss    (%%rax), %%zmm0                          \n"
			"   vdpbf16ps        %%zmm2, %%zmm6, %%zmm28                 \n"
			"   vmovdqu16         64(%%rbx), %%zmm5                      \n"
			"   vdpbf16ps        %%zmm2, %%zmm7, %%zmm29                 \n"

			"   vbroadcastss    4(%%rax), %%zmm1                         \n"
			"   vdpbf16ps        %%zmm3, %%zmm6, %%zmm30                 \n"
			"   vdpbf16ps        %%zmm3, %%zmm7, %%zmm31                 \n"
			".endm                                                       \n"

			".macro    bf16_kernel_m12n32k2_end                          \n"
			"   vbroadcastss    8(%%rax), %%zmm2                         \n"
			"   vdpbf16ps        %%zmm0, %%zmm6, %%zmm8                  \n"
			"   vdpbf16ps        %%zmm0, %%zmm7, %%zmm9                  \n"
			"   vbroadcastss    12(%%rax), %%zmm3                        \n"

			"   vdpbf16ps        %%zmm1, %%zmm6, %%zmm10                 \n"
			"   vdpbf16ps        %%zmm1, %%zmm7, %%zmm11                 \n"

			"   prefetcht0         256(%%rax)                            \n"

			"   vbroadcastss    16(%%rax), %%zmm0                        \n"
			"   vdpbf16ps        %%zmm2, %%zmm6, %%zmm12                 \n"
			"   vdpbf16ps        %%zmm2, %%zmm7, %%zmm13                 \n"

			"   vbroadcastss    20(%%rax), %%zmm1                        \n"
			"   vdpbf16ps        %%zmm3, %%zmm6, %%zmm14                 \n"
			"   vdpbf16ps        %%zmm3, %%zmm7, %%zmm15                 \n"

			"   vbroadcastss    24(%%rax), %%zmm2                        \n"
			"   vdpbf16ps        %%zmm0, %%zmm6, %%zmm16                 \n"
			"   vdpbf16ps        %%zmm0, %%zmm7, %%zmm17                 \n"

			"   vbroadcastss    28(%%rax), %%zmm3                        \n"
			"   vdpbf16ps        %%zmm1, %%zmm6, %%zmm18                 \n"
			"   vdpbf16ps        %%zmm1, %%zmm7, %%zmm19                 \n"

			"   vbroadcastss    32(%%rax), %%zmm0                        \n"
			"   vdpbf16ps        %%zmm2, %%zmm6, %%zmm20                 \n"
			"   vdpbf16ps        %%zmm2, %%zmm7, %%zmm21                 \n"

			"   vbroadcastss    36(%%rax), %%zmm1                        \n"
			"   vdpbf16ps        %%zmm3, %%zmm6, %%zmm22                 \n"
			"   vdpbf16ps        %%zmm3, %%zmm7, %%zmm23                 \n"

			"   vbroadcastss    40(%%rax), %%zmm2                        \n"
			"   vdpbf16ps        %%zmm0, %%zmm6, %%zmm24                 \n"
			"   vdpbf16ps        %%zmm0, %%zmm7, %%zmm25                 \n"

			"   vbroadcastss    44(%%rax), %%zmm3                        \n"
			"   vdpbf16ps        %%zmm1, %%zmm6, %%zmm26                 \n"
			"    addq              $48, %%rax                            \n"
			"   vdpbf16ps        %%zmm1, %%zmm7, %%zmm27                 \n"

			"   vdpbf16ps        %%zmm2, %%zmm6, %%zmm28                 \n"
			"   vdpbf16ps        %%zmm2, %%zmm7, %%zmm29                 \n"

			"   vdpbf16ps        %%zmm3, %%zmm6, %%zmm30                 \n"
			"   vdpbf16ps        %%zmm3, %%zmm7, %%zmm31                 \n"
			".endm                                                       \n"

			".macro    bf16_save_c_m12n32                                \n"
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
			"    subq             $12, %%rdi                             \n"
			"   vmovups         %%zmm28, (%%r12)                         \n"
			"   vmovups         %%zmm29, 64(%%r12)                       \n"
			"   vmovups         %%zmm30, (%%r13)                         \n"
			"   vmovups         %%zmm31, 64(%%r13)                       \n"

			"    leaq      (%%r13, %%r8, 4), %%rcx                       \n" // C0
			".endm                                                       \n"

			".macro    bf16_save_c_m12n32_2                              \n"
			"   vcvtne2ps2bf16  %%zmm8, %%zmm9, %%zmm0                   \n"
			"   vcvtne2ps2bf16  %%zmm10, %%zmm11, %%zmm1                 \n"
			"   vcvtne2ps2bf16  %%zmm12, %%zmm13, %%zmm2                 \n"
			"   vcvtne2ps2bf16  %%zmm14, %%zmm15, %%zmm3                 \n"
			"   vmovups         %%zmm0, (%%r10)                          \n"
			"   vmovups         %%zmm1, 64(%%r10)                        \n"
			"   vmovups         %%zmm2, 128(%%r10)                       \n"
			"   vmovups         %%zmm3, 192(%%r10)                       \n"

			"   addq           $256, %%r10                               \n"

			"   vcvtne2ps2bf16  %%zmm16, %%zmm17, %%zmm4                 \n"
			"   vcvtne2ps2bf16  %%zmm18, %%zmm19, %%zmm5                 \n"
			"   vcvtne2ps2bf16  %%zmm20, %%zmm21, %%zmm6                 \n"
			"   vcvtne2ps2bf16  %%zmm22, %%zmm23, %%zmm7                 \n"
			"   vmovups         %%zmm4, (%%r10)                          \n"
			"   vmovups         %%zmm5, 64(%%r10)                        \n"
			"   vmovups         %%zmm6, 128(%%r10)                       \n"
			"   vmovups         %%zmm7, 192(%%r10)                       \n"

			"   addq           $256, %%r10                               \n"

			"   vcvtne2ps2bf16  %%zmm24, %%zmm25, %%zmm8                 \n"
			"   vcvtne2ps2bf16  %%zmm26, %%zmm27, %%zmm9                 \n"
			"   vcvtne2ps2bf16  %%zmm28, %%zmm29, %%zmm10                \n"
			"   vcvtne2ps2bf16  %%zmm30, %%zmm31, %%zmm11                \n"
			"   vmovups         %%zmm8, (%%r10)                          \n"
			"   vmovups         %%zmm9, 64(%%r10)                        \n"
			"   vmovups         %%zmm10, 128(%%r10)                      \n"
			"   vmovups         %%zmm11, 192(%%r10)                      \n"

			"   subq           $12, %%rdi                                \n"
			"   addq           $256, %%r10                               \n"
			".endm                                                       \n"

			//-----------------------------------------------------------------

			".macro    bf16_kernel_m8n32k2_1                             \n"
			"   vbroadcastss    8(%%rax), %%zmm2                         \n"
			"   vdpbf16ps        %%zmm0, %%zmm4, %%zmm8                  \n"
			"   vdpbf16ps        %%zmm0, %%zmm5, %%zmm9                  \n"

			"   vbroadcastss    12(%%rax), %%zmm3                        \n"
			"   vdpbf16ps        %%zmm1, %%zmm4, %%zmm10                 \n"
			"   vdpbf16ps        %%zmm1, %%zmm5, %%zmm11                 \n"

			"   prefetcht0      256(%%rax)                               \n"
			"   addq             $128, %%rbx                             \n"

			"   vbroadcastss    16(%%rax), %%zmm0                        \n"
			"   vdpbf16ps        %%zmm2, %%zmm4, %%zmm12                 \n"
			"   vdpbf16ps        %%zmm2, %%zmm5, %%zmm13                 \n"
			"   vmovdqu16        (%%rbx), %%zmm6                         \n"

			"   vbroadcastss    20(%%rax), %%zmm1                        \n"
			"   vdpbf16ps        %%zmm3, %%zmm4, %%zmm14                 \n"
			"   vdpbf16ps        %%zmm3, %%zmm5, %%zmm15                 \n"

			"   vbroadcastss    24(%%rax), %%zmm2                        \n"
			"   vdpbf16ps        %%zmm0, %%zmm4, %%zmm16                 \n"
			"   vdpbf16ps        %%zmm0, %%zmm5, %%zmm17                 \n"
			"   vmovdqu16        64(%%rbx), %%zmm7                       \n"

			"   vbroadcastss    28(%%rax), %%zmm3                        \n"
			"   vdpbf16ps        %%zmm1, %%zmm4, %%zmm18                 \n"
			"   vdpbf16ps        %%zmm1, %%zmm5, %%zmm19                 \n"

			"   addq              $32, %%rax                             \n"
			"   vbroadcastss     (%%rax), %%zmm0                         \n"
			"   vdpbf16ps        %%zmm2, %%zmm4, %%zmm20                 \n"
			"   vdpbf16ps        %%zmm2, %%zmm5, %%zmm21                 \n"

			"   vbroadcastss     4(%%rax), %%zmm1                        \n"
			"   vdpbf16ps        %%zmm3, %%zmm4, %%zmm22                 \n"
			"   vdpbf16ps        %%zmm3, %%zmm5, %%zmm23                 \n"
			".endm                                                       \n"

			".macro    bf16_kernel_m8n32k2_2                             \n"
			"   vbroadcastss    8(%%rax), %%zmm2                         \n"
			"   vdpbf16ps        %%zmm0, %%zmm6, %%zmm8                  \n"
			"   vdpbf16ps        %%zmm0, %%zmm7, %%zmm9                  \n"

			"   vbroadcastss    12(%%rax), %%zmm3                        \n"
			"   vdpbf16ps        %%zmm1, %%zmm6, %%zmm10                 \n"
			"   vdpbf16ps        %%zmm1, %%zmm7, %%zmm11                 \n"

			"   prefetcht0         256(%%rax)                            \n"
			"   addq              $128, %%rbx                            \n"

			"   vbroadcastss    16(%%rax), %%zmm0                        \n"
			"   vdpbf16ps        %%zmm2, %%zmm6, %%zmm12                 \n"
			"   vdpbf16ps        %%zmm2, %%zmm7, %%zmm13                 \n"

			"   vbroadcastss    20(%%rax), %%zmm1                        \n"
			"   vdpbf16ps        %%zmm3, %%zmm6, %%zmm14                 \n"
			"   vdpbf16ps        %%zmm3, %%zmm7, %%zmm15                 \n"

			"   vbroadcastss    24(%%rax), %%zmm2                        \n"
			"   vdpbf16ps        %%zmm0, %%zmm6, %%zmm16                 \n"
			"   vdpbf16ps        %%zmm0, %%zmm7, %%zmm17                 \n"
			"   vmovdqu16       (%%rbx), %%zmm4                          \n"

			"   vbroadcastss    28(%%rax), %%zmm3                        \n"
			"   vdpbf16ps        %%zmm1, %%zmm6, %%zmm18                 \n"
			"   vdpbf16ps        %%zmm1, %%zmm7, %%zmm19                 \n"
			"   vmovdqu16       64(%%rbx), %%zmm5                        \n"

			"   addq             $32, %%rax                              \n"

			"   vbroadcastss    (%%rax), %%zmm0                          \n"
			"   vdpbf16ps        %%zmm2, %%zmm6, %%zmm20                 \n"
			"   vdpbf16ps        %%zmm2, %%zmm7, %%zmm21                 \n"

			"   vbroadcastss    4(%%rax), %%zmm1                         \n"
			"   vdpbf16ps        %%zmm3, %%zmm6, %%zmm22                 \n"
			"   vdpbf16ps        %%zmm3, %%zmm7, %%zmm23                 \n"
			".endm                                                       \n"

			".macro    bf16_kernel_m8n32k2_end                           \n"
			"   vbroadcastss    8(%%rax), %%zmm2                         \n"
			"   vdpbf16ps        %%zmm0, %%zmm6, %%zmm8                  \n"
			"   vdpbf16ps        %%zmm0, %%zmm7, %%zmm9                  \n"

			"   vbroadcastss    12(%%rax), %%zmm3                        \n"
			"   vdpbf16ps        %%zmm1, %%zmm6, %%zmm10                 \n"
			"   vdpbf16ps        %%zmm1, %%zmm7, %%zmm11                 \n"

			"   prefetcht0         256(%%rax)                            \n"

			"   vbroadcastss    16(%%rax), %%zmm0                        \n"
			"   vdpbf16ps        %%zmm2, %%zmm6, %%zmm12                 \n"
			"   vdpbf16ps        %%zmm2, %%zmm7, %%zmm13                 \n"

			"   vbroadcastss    20(%%rax), %%zmm1                        \n"
			"   vdpbf16ps        %%zmm3, %%zmm6, %%zmm14                 \n"
			"   vdpbf16ps        %%zmm3, %%zmm7, %%zmm15                 \n"

			"   vbroadcastss    24(%%rax), %%zmm2                        \n"
			"   vdpbf16ps        %%zmm0, %%zmm6, %%zmm16                 \n"
			"   vdpbf16ps        %%zmm0, %%zmm7, %%zmm17                 \n"

			"   vbroadcastss    28(%%rax), %%zmm3                        \n"
			"   vdpbf16ps        %%zmm1, %%zmm6, %%zmm18                 \n"
			"   vdpbf16ps        %%zmm1, %%zmm7, %%zmm19                 \n"

			"   addq             $32, %%rax                              \n"

			"   vdpbf16ps        %%zmm2, %%zmm6, %%zmm20                 \n"
			"   vdpbf16ps        %%zmm2, %%zmm7, %%zmm21                 \n"

			"   vdpbf16ps        %%zmm3, %%zmm6, %%zmm22                 \n"
			"   vdpbf16ps        %%zmm3, %%zmm7, %%zmm23                 \n"
			".endm                                                       \n"

			".macro    bf16_save_c_m8n32                                 \n"
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

			".macro    bf16_save_c_m8n32_2                               \n"
			"   vcvtne2ps2bf16  %%zmm8, %%zmm9, %%zmm0                   \n"
			"   vcvtne2ps2bf16  %%zmm10, %%zmm11, %%zmm1                 \n"
			"   vcvtne2ps2bf16  %%zmm12, %%zmm13, %%zmm2                 \n"
			"   vcvtne2ps2bf16  %%zmm14, %%zmm15, %%zmm3                 \n"
			"   vmovups         %%zmm0, (%%r10)                          \n"
			"   vmovups         %%zmm1, 64(%%r10)                        \n"
			"   vmovups         %%zmm2, 128(%%r10)                       \n"
			"   vmovups         %%zmm3, 192(%%r10)                       \n"

			"   addq           $256, %%r10                               \n"

			"   vcvtne2ps2bf16  %%zmm16, %%zmm17, %%zmm4                 \n"
			"   vcvtne2ps2bf16  %%zmm18, %%zmm19, %%zmm5                 \n"
			"   vcvtne2ps2bf16  %%zmm20, %%zmm21, %%zmm6                 \n"
			"   vcvtne2ps2bf16  %%zmm22, %%zmm23, %%zmm7                 \n"
			"   vmovups         %%zmm4, (%%r10)                          \n"
			"   vmovups         %%zmm5, 64(%%r10)                        \n"
			"   vmovups         %%zmm6, 128(%%r10)                       \n"
			"   vmovups         %%zmm7, 192(%%r10)                       \n"

			"   subq           $8, %%rdi                                 \n"
			"   addq           $256, %%r10                               \n"
			".endm                                                       \n"

			//-----------------------------------------------------------------

			".macro    bf16_kernel_m4n32k2_1                             \n"
			"   vbroadcastss    8(%%rax), %%zmm2                         \n"
			"   vdpbf16ps        %%zmm0, %%zmm4, %%zmm8                  \n"
			"   vdpbf16ps        %%zmm0, %%zmm5, %%zmm9                  \n"
			"   addq             $128, %%rbx                             \n"
			"   vmovdqu16        (%%rbx), %%zmm6                         \n"

			"   vbroadcastss    12(%%rax), %%zmm3                        \n"
			"   vdpbf16ps        %%zmm1, %%zmm4, %%zmm10                 \n"
			"   vdpbf16ps        %%zmm1, %%zmm5, %%zmm11                 \n"
			"   vmovdqu16        64(%%rbx), %%zmm7                       \n"

			"   prefetcht0      256(%%rax)                               \n"
			"   addq              $16, %%rax                             \n"

			"   vbroadcastss     (%%rax), %%zmm0                         \n"
			"   vdpbf16ps        %%zmm2, %%zmm4, %%zmm12                 \n"
			"   vdpbf16ps        %%zmm2, %%zmm5, %%zmm13                 \n"

			"   vbroadcastss     4(%%rax), %%zmm1                        \n"
			"   vdpbf16ps        %%zmm3, %%zmm4, %%zmm14                 \n"
			"   vdpbf16ps        %%zmm3, %%zmm5, %%zmm15                 \n"
			".endm                                                       \n"

			".macro    bf16_kernel_m4n32k2_2                             \n"
			"   vbroadcastss     8(%%rax), %%zmm2                        \n"
			"   vdpbf16ps        %%zmm0, %%zmm6, %%zmm8                  \n"
			"   vdpbf16ps        %%zmm0, %%zmm7, %%zmm9                  \n"
			"   addq             $128, %%rbx                             \n"
			"   vmovdqu16        (%%rbx), %%zmm4                         \n"

			"   vbroadcastss     12(%%rax), %%zmm3                       \n"
			"   vdpbf16ps        %%zmm1, %%zmm6, %%zmm10                 \n"
			"   vdpbf16ps        %%zmm1, %%zmm7, %%zmm11                 \n"
			"   vmovdqu16        64(%%rbx), %%zmm5                       \n"

			"   prefetcht0         256(%%rax)                            \n"
			"   addq             $16, %%rax                              \n"

			"   vbroadcastss     (%%rax), %%zmm0                         \n"
			"   vdpbf16ps        %%zmm2, %%zmm6, %%zmm12                 \n"
			"   vdpbf16ps        %%zmm2, %%zmm7, %%zmm13                 \n"

			"   vbroadcastss     4(%%rax), %%zmm1                        \n"
			"   vdpbf16ps        %%zmm3, %%zmm6, %%zmm14                 \n"
			"   vdpbf16ps        %%zmm3, %%zmm7, %%zmm15                 \n"
			".endm                                                       \n"

			".macro    bf16_kernel_m4n32k2_end                           \n"
			"   vbroadcastss    8(%%rax), %%zmm2                         \n"
			"   vdpbf16ps        %%zmm0, %%zmm6, %%zmm8                  \n"
			"   vdpbf16ps        %%zmm0, %%zmm7, %%zmm9                  \n"

			"   vbroadcastss    12(%%rax), %%zmm3                        \n"
			"   vdpbf16ps        %%zmm1, %%zmm6, %%zmm10                 \n"
			"   vdpbf16ps        %%zmm1, %%zmm7, %%zmm11                 \n"

			"   prefetcht0         256(%%rax)                            \n"

			"   addq             $16, %%rax                              \n"
			"   vbroadcastss     (%%rax), %%zmm0                         \n"
			"   vdpbf16ps        %%zmm2, %%zmm6, %%zmm12                 \n"
			"   vdpbf16ps        %%zmm2, %%zmm7, %%zmm13                 \n"

			"   vbroadcastss     4(%%rax), %%zmm1                        \n"
			"   vdpbf16ps        %%zmm3, %%zmm6, %%zmm14                 \n"
			"   vdpbf16ps        %%zmm3, %%zmm7, %%zmm15                 \n"
			".endm                                                       \n"

			".macro    bf16_save_c_m4n32                                 \n"
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

			".macro    bf16_save_c_m4n32_2                               \n"
			"   vcvtne2ps2bf16  %%zmm8, %%zmm9, %%zmm0                   \n"
			"   vcvtne2ps2bf16  %%zmm10, %%zmm11, %%zmm1                 \n"
			"   vcvtne2ps2bf16  %%zmm12, %%zmm13, %%zmm2                 \n"
			"   vcvtne2ps2bf16  %%zmm14, %%zmm15, %%zmm3                 \n"
			"   vmovups         %%zmm0, (%%r10)                          \n"
			"   vmovups         %%zmm1, 64(%%r10)                        \n"
			"   vmovups         %%zmm2, 128(%%r10)                       \n"
			"   vmovups         %%zmm3, 192(%%r10)                       \n"

			"   addq           $256, %%r10                               \n"
			"   subq           $4, %%rdi                                 \n"
			".endm                                                       \n"

			//-----------------------------------------------------------------
			//-----------------------------------------------------------------

			"GEMM_BF16_N32:                                              \n"
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
			"   jz     BF16_BEGIN_M12N32                                 \n"

			//-----------------------------------------------------------------

			"BF16_BEGIN_PACK_N32:                                        \n"
			"   mov     %%r14, %%rbp                                     \n" // Bc

			"   vmovdqu16 (%%rbx), %%zmm6                                \n"
			"   prefetcht2  64(%%rbx)                                    \n"
			"   leaq    (%%rbx, %%r8, 2), %%rbx                          \n"
			"   vmovdqu16 (%%rbx), %%zmm7                                \n"
			"   prefetcht2  64(%%rbx)                                    \n"
			"   leaq    (%%rbx, %%r8, 2), %%rbx                          \n"

			"   mov     %%rsi, %%rdx                                     \n" // K
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
			"   cmp     $16, %%rdx                                       \n"
			"   jb      BF16_PACK_MAIN_M12N32K2                          \n"
			"   subq    $16, %%rdx                                       \n"

			"BF16_PACK_MAIN_M12N32K16:                                   \n"
			"   bf16_kernel_m12n32k2_pack                                \n"
			"   bf16_kernel_m12n32k2_pack                                \n"
			"   bf16_kernel_m12n32k2_pack                                \n"
			"   bf16_kernel_m12n32k2_pack                                \n"
			"   bf16_kernel_m12n32k2_pack                                \n"
			"   bf16_kernel_m12n32k2_pack                                \n"
			"   bf16_kernel_m12n32k2_pack                                \n"
			"   cmp     $0, %%rdx                                        \n"
			"   je      BF16_PACK_SAVEC_M12N32                           \n"
			"   bf16_kernel_m12n32k2_pack                                \n"
			"   cmp     $16, %%rdx                                       \n"
			"   jb      BF16_PACK_MAIN_M12N32K2                          \n"
			"   subq    $16, %%rdx                                       \n"
			"   jmp     BF16_PACK_MAIN_M12N32K16                         \n"

			"BF16_PACK_MAIN_M12N32K2:                                    \n"
			"   subq    $2, %%rdx                                        \n"
			"   cmp     $0, %%rdx                                        \n"
			"   je      BF16_PACK_SAVEC_M12N32                           \n"
			"   bf16_kernel_m12n32k2_pack                                \n"
			"   jmp     BF16_PACK_MAIN_M12N32K2                          \n"

			"BF16_PACK_SAVEC_M12N32:                                     \n"
			"    bf16_kernel_m12n32k2_pack_end                           \n"
			"    jmp        BF16_BEGIN_SAVE_M12N32                       \n"

			//-----------------------------------------------------------------

			"BF16_BEGIN_M12N32:                                          \n"
			"   cmpq    $12, %%rdi                                       \n"
			"   jb      BF16_BEGIN_M8N32                                 \n"

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
			"   cmp     $16, %%rdx                                       \n"
			"   jb      BF16_MAIN_M12N32K2                               \n"
			"   subq    $16, %%rdx                                       \n"

			"BF16_MAIN_K_M12N32K16:                                      \n" // loop K+=4
			"   bf16_kernel_m12n32k2_1                                   \n"
			"   bf16_kernel_m12n32k2_2                                   \n"
			"   bf16_kernel_m12n32k2_1                                   \n"
			"   bf16_kernel_m12n32k2_2                                   \n"
			"   bf16_kernel_m12n32k2_1                                   \n"
			"   bf16_kernel_m12n32k2_2                                   \n"
			"   bf16_kernel_m12n32k2_1                                   \n"
			"   bf16_kernel_m12n32k2_2                                   \n"
			"   cmp     $0, %%rdx                                        \n"
			"   je      BF16_BEGIN_SAVE_M12N32                           \n"
			"   cmp     $16, %%rdx                                       \n"
			"   jb      BF16_MAIN_M12N32K2                               \n"
			"   subq  $16, %%rdx                                         \n"
			"   jmp   BF16_MAIN_K_M12N32K16                              \n"

			"BF16_MAIN_M12N32K2:                                         \n"
			"   bf16_kernel_m12n32k2_1                                   \n"
			"   subq    $2, %%rdx                                        \n"
			"   cmp     $0, %%rdx                                        \n"
			"   je      BF16_BEGIN_SAVE_M12N32                           \n"
			"   bf16_kernel_m12n32k2_2                                   \n"
			"   subq    $2, %%rdx                                        \n"
			"   cmp     $0, %%rdx                                        \n"
			"   je      BF16_BEGIN_SAVE_M12N32                           \n"
			"   jmp     BF16_MAIN_M12N32K2                               \n"

			"BF16_BEGIN_SAVE_M12N32:                                     \n"
			"   mov    %[is_end_gemm], %%r13                             \n"
			"   test      $1, %%r13                                      \n"
			"   jz       BF16_SAVE_C_M12N32_2                            \n"
			"   mov      %%rcx, %%r10                                    \n" // C0
			"   leaq     (%%r10, %%r8, 4), %%r11                         \n" // C1
			"   leaq     (%%r11, %%r8, 4), %%r12                         \n" // C2
			"   leaq     (%%r12, %%r8, 4), %%r13                         \n" // C3

			"BF16_SAVE_C_M12N32:                                         \n"
			"   bf16_save_c_m12n32                                       \n"
			"   imul     $24, %%r15, %%r11                               \n" // temp use %%r11
			"   add      %%r11, %%r9                                     \n"
			"   movq     %%r9, %%rax                                     \n"
			"   jmp     BF16_BEGIN_M12N32                                \n"

			"BF16_SAVE_C_M12N32_2:                                       \n"
			"   bf16_save_c_m12n32_2                                     \n"
			"   imul     $24, %%r15, %%r11                               \n" // temp use %%r11
			"   add      %%r11, %%r9                                     \n"
			"   movq     %%r9, %%rax                                     \n"
			"   jmp     BF16_BEGIN_M12N32                                \n"

			//-----------------------------------------------------------------

			"BF16_BEGIN_M8N32:                                           \n"
			"   cmpq    $8, %%rdi                                        \n"
			"   jb      BF16_BEGIN_M4N32                                 \n"

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
			"   cmpq    $16, %%rdx                                       \n"
			"   jb      BF16_MAIN_M8N32K2                                \n"
			"   subq    $16, %%rdx                                       \n"

			"BF16_MAIN_K_M8N32K16:                                       \n"
			"   bf16_kernel_m8n32k2_1                                    \n"
			"   bf16_kernel_m8n32k2_2                                    \n"
			"   bf16_kernel_m8n32k2_1                                    \n"
			"   bf16_kernel_m8n32k2_2                                    \n"
			"   bf16_kernel_m8n32k2_1                                    \n"
			"   bf16_kernel_m8n32k2_2                                    \n"
			"   bf16_kernel_m8n32k2_1                                    \n"
			"   bf16_kernel_m8n32k2_2                                    \n"
			"   cmp     $0, %%rdx                                        \n"
			"   je      BF16_BEGIN_SAVE_M8N32                            \n"
			"   cmpq    $16, %%rdx                                       \n"
			"   jb      BF16_MAIN_M8N32K2                                \n"
			"   subq  $16, %%rdx                                         \n"
			"   jmp   BF16_MAIN_K_M8N32K16                               \n"

			"BF16_MAIN_M8N32K2:                                          \n"
			"   bf16_kernel_m8n32k2_1                                    \n"
			"   subq    $2, %%rdx                                        \n"
			"   cmp     $0, %%rdx                                        \n"
			"   je      BF16_BEGIN_SAVE_M8N32                            \n"
			"   bf16_kernel_m8n32k2_2                                    \n"
			"   subq    $2, %%rdx                                        \n"
			"   cmp     $0, %%rdx                                        \n"
			"   je      BF16_BEGIN_SAVE_M8N32                            \n"
			"   jmp     BF16_MAIN_M8N32K2                                \n"

			"BF16_BEGIN_SAVE_M8N32:                                      \n"
			"   mov    %[is_end_gemm], %%r13                             \n"
			"   test      $1, %%r13                                      \n"
			"   jz       BF16_SAVE_C_M8N32_2                             \n"
			"   mov      %%rcx, %%r10                                    \n" // C0
			"   leaq     (%%r10, %%r8, 4), %%r11                         \n" // C1
			"   leaq     (%%r11, %%r8, 4), %%r12                         \n" // C2
			"   leaq     (%%r12, %%r8, 4), %%r13                         \n" // C3

			"BF16_SAVE_C_M8N32:                                          \n"
			"   bf16_save_c_m8n32                                        \n"
			"   imul     $16, %%r15, %%r11                               \n" // temp use %%r11
			"   add      %%r11, %%r9                                     \n"
			"   movq     %%r9, %%rax                                     \n"
			"   jmp      BF16_BEGIN_M8N32                                \n"

			"BF16_SAVE_C_M8N32_2:                                        \n"
			"   bf16_save_c_m8n32_2                                      \n"
			"   imul     $16, %%r15, %%r11                               \n" // temp use %%r11
			"   add      %%r11, %%r9                                     \n"
			"   movq     %%r9, %%rax                                     \n"
			"   jmp      BF16_BEGIN_M8N32                                \n"

			//-----------------------------------------------------------------

			"BF16_BEGIN_M4N32:                                           \n"
			"   cmpq    $0, %%rdi                                        \n"
			"   je      BF16_END_N32                                     \n"
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
			"   cmpq    $16, %%rdx                                       \n"
			"   jb      BF16_MAIN_M4N32K2                                \n"
			"   subq  $16, %%rdx                                         \n"

			"BF16_MAIN_K_M4N32K16:                                       \n" // loop K+=4
			"   bf16_kernel_m4n32k2_1                                    \n"
			"   bf16_kernel_m4n32k2_2                                    \n"
			"   bf16_kernel_m4n32k2_1                                    \n"
			"   bf16_kernel_m4n32k2_2                                    \n"
			"   bf16_kernel_m4n32k2_1                                    \n"
			"   bf16_kernel_m4n32k2_2                                    \n"
			"   bf16_kernel_m4n32k2_1                                    \n"
			"   bf16_kernel_m4n32k2_2                                    \n"
			"   cmp     $0, %%rdx                                        \n"
			"   je      BF16_BEGIN_SAVE_M4N32                            \n"
			"   cmpq    $16, %%rdx                                       \n"
			"   jb      BF16_MAIN_M4N32K2                                \n"
			"   subq  $16, %%rdx                                         \n"
			"   jmp   BF16_MAIN_K_M4N32K16                               \n"

			"BF16_MAIN_M4N32K2:                                          \n"
			"   bf16_kernel_m4n32k2_1                                    \n"
			"   subq    $2, %%rdx                                        \n"
			"   cmp     $0, %%rdx                                        \n"
			"   je      BF16_BEGIN_SAVE_M4N32                            \n"
			"   bf16_kernel_m4n32k2_2                                    \n"
			"   subq    $2, %%rdx                                        \n"
			"   cmp     $0, %%rdx                                        \n"
			"   je      BF16_BEGIN_SAVE_M4N32                            \n"
			"   jmp     BF16_MAIN_M4N32K2                                \n"

			"BF16_BEGIN_SAVE_M4N32:                                      \n"
			"   mov    %[is_end_gemm], %%r13                             \n"
			"   test     $1, %%r13                                       \n"
			"   jz       BF16_SAVE_C_M4N32_2                             \n"
			"   mov      %%rcx, %%r10                                    \n" // C0
			"   leaq     (%%r10, %%r8, 4), %%r11                         \n" // C1
			"   leaq     (%%r11, %%r8, 4), %%r12                         \n" // C2
			"   leaq     (%%r12, %%r8, 4), %%r13                         \n" // C3

			"BF16_SAVE_C_M4N32:                                          \n"
			"   cmpq     $3, %%rdi                                       \n"
			"   je       BF16_SAVE_C_M3N32                               \n"
			"   cmpq     $2, %%rdi                                       \n"
			"   je       BF16_SAVE_C_M2N32                               \n"
			"   cmpq     $1, %%rdi                                       \n"
			"   je       BF16_SAVE_C_M1N32                               \n"
			"   bf16_save_c_m4n32                                        \n"
			"   imul     $8, %%r15, %%r11                                \n" // temp use %%r11
			"   add      %%r11, %%r9                                     \n"
			"   movq     %%r9, %%rax                                     \n"
			"   jmp      BF16_BEGIN_M4N32                                \n"

			"BF16_SAVE_C_M3N32:                                          \n"
			"   vmovups         %%zmm12, (%%r12)                         \n"
			"   vmovups         %%zmm13, 64(%%r12)                       \n"

			"BF16_SAVE_C_M2N32:                                          \n"
			"   vmovups         %%zmm10, (%%r11)                         \n"
			"   vmovups         %%zmm11, 64(%%r11)                       \n"

			"BF16_SAVE_C_M1N32:                                          \n"
			"   vmovups         %%zmm8, (%%r10)                          \n"
			"   vmovups         %%zmm9, 64(%%r10)                        \n"
			"   jmp      BF16_END_N32                                    \n"

			"BF16_SAVE_C_M4N32_2:                                        \n"
			"   cmpq     $3, %%rdi                                       \n"
			"   je       BF16_SAVE_C_M3N32_2                             \n"
			"   cmpq     $2, %%rdi                                       \n"
			"   je       BF16_SAVE_C_M2N32_2                             \n"
			"   cmpq     $1, %%rdi                                       \n"
			"   je       BF16_SAVE_C_M1N32_2                             \n"
			"   bf16_save_c_m4n32_2                                      \n"
			"   imul     $8, %%r15, %%r11                                \n" // temp use %%r11
			"   add      %%r11, %%r9                                     \n"
			"   movq     %%r9, %%rax                                     \n"
			"   jmp      BF16_BEGIN_M4N32                                \n"

			"BF16_SAVE_C_M3N32_2:                                        \n"
			"   vcvtne2ps2bf16  %%zmm12, %%zmm13, %%zmm2                 \n"
			"   vmovups         %%zmm2, 128(%%r10)                       \n"

			"BF16_SAVE_C_M2N32_2:                                        \n"
			"   vcvtne2ps2bf16  %%zmm10, %%zmm11, %%zmm1                 \n"
			"   vmovups         %%zmm1, 64(%%r10)                        \n"

			"BF16_SAVE_C_M1N32_2:                                        \n"
			"   vcvtne2ps2bf16  %%zmm8, %%zmm9, %%zmm0                   \n"
			"   vmovups         %%zmm0, (%%r10)                          \n"

			"BF16_END_N32:                                               \n"

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

static void FLASHGEMM_BF16_KERNELm12xn16_edge(float *C, uint16_t *Cc, uint16_t *A, uint16_t *B, long M, long K, long LK, long LN, uint16_t *Bc, long nr, bool is_start_gemm, bool is_end_gemm)
{
	asm volatile(
			".macro bf16_pack_b_n16                                      \n"
			"   vbroadcastss    (%%rax), %%zmm0                          \n" // A0 in zmm0
			"   vbroadcastss    4(%%rax), %%zmm1                         \n" // A1 in zmm1

			"   vpunpcklwd  %%ymm7, %%ymm6, %%ymm4                       \n"
			"   vpunpckhwd  %%ymm7, %%ymm6, %%ymm5                       \n"

			"   vextractf128   $0x1, %%ymm4, %%xmm6                      \n"
			"   vextractf128   $0x1, %%ymm5, %%xmm7                      \n"

			"   vinserti32x4    $0x2, %%xmm6, %%zmm4, %%zmm4             \n"
			"   vinserti32x4    $0x1, %%xmm5, %%zmm4, %%zmm4             \n"
			"   vinserti32x4    $0x3, %%xmm7, %%zmm4, %%zmm4             \n"

			"   vmovups         %%zmm4, (%%rbp)                          \n" // store back B
			"   addq            $64, %%rbp                               \n"
			".endm                                                       \n"

			".macro bf16_kernel_m12n16k2_pack                            \n"
			"   bf16_pack_b_n16                                          \n"

			"   vbroadcastss    8(%%rax), %%zmm2                         \n"
			"   vdpbf16ps        %%zmm0, %%zmm4, %%zmm8                  \n"

			"   vbroadcastss    12(%%rax), %%zmm3                        \n"
			"   vdpbf16ps        %%zmm1, %%zmm4, %%zmm10                 \n"

			"   prefetcht0         256(%%rax)                            \n"

			"   vbroadcastss    16(%%rax), %%zmm0                        \n"
			"   vdpbf16ps        %%zmm2, %%zmm4, %%zmm12                 \n"

			"   vbroadcastss    20(%%rax), %%zmm1                        \n"
			"   vdpbf16ps        %%zmm3, %%zmm4, %%zmm14                 \n"

			"   vbroadcastss    24(%%rax), %%zmm2                        \n"
			"   vdpbf16ps        %%zmm0, %%zmm4, %%zmm16                 \n"

			"   vbroadcastss    28(%%rax), %%zmm3                        \n"
			"   vdpbf16ps        %%zmm1, %%zmm4, %%zmm18                 \n"

			"   vmovdqu16     (%%rbx), %%ymm6                            \n" // load next B
			"   prefetcht2  32(%%rbx)                                    \n"
			"   leaq        (%%rbx, %%r8, 2), %%rbx                      \n"

			"   vbroadcastss    32(%%rax), %%zmm0                        \n"
			"   vdpbf16ps        %%zmm2, %%zmm4, %%zmm20                 \n"

			"   vbroadcastss    36(%%rax), %%zmm1                        \n"
			"   vdpbf16ps        %%zmm3, %%zmm4, %%zmm22                 \n"

			"   vbroadcastss    40(%%rax), %%zmm2                        \n"
			"   vdpbf16ps        %%zmm0, %%zmm4, %%zmm24                 \n"

			"   vmovdqu16         (%%rbx), %%ymm7                        \n" // load next B
			"   prefetcht2  32(%%rbx)                                    \n"
			"   leaq        (%%rbx, %%r8, 2), %%rbx                      \n"

			"   vbroadcastss    44(%%rax), %%zmm3                        \n"
			"   vdpbf16ps        %%zmm1, %%zmm4, %%zmm26                 \n"
			"   addq          $48, %%rax                                 \n"

			"   vdpbf16ps        %%zmm2, %%zmm4, %%zmm28                 \n"

			"   vdpbf16ps        %%zmm3, %%zmm4, %%zmm30                 \n"
			".endm                                                       \n"

			".macro bf16_kernel_m12n16k2_pack_end                        \n" // deference is no prefetch A and B
			"   bf16_pack_b_n16                                          \n"

			"   vbroadcastss    8(%%rax), %%zmm2                         \n"
			"   vdpbf16ps        %%zmm0, %%zmm4, %%zmm8                  \n"

			"   vbroadcastss    12(%%rax), %%zmm3                        \n"
			"   vdpbf16ps        %%zmm1, %%zmm4, %%zmm10                 \n"

			"   prefetcht0         256(%%rax)                            \n"

			"   vbroadcastss    16(%%rax), %%zmm0                        \n"
			"   vdpbf16ps        %%zmm2, %%zmm4, %%zmm12                 \n"

			"   vbroadcastss    20(%%rax), %%zmm1                        \n"
			"   vdpbf16ps        %%zmm3, %%zmm4, %%zmm14                 \n"

			"   vbroadcastss    24(%%rax), %%zmm2                        \n"
			"   vdpbf16ps        %%zmm0, %%zmm4, %%zmm16                 \n"

			"   vbroadcastss    28(%%rax), %%zmm3                        \n"
			"   vdpbf16ps        %%zmm1, %%zmm4, %%zmm18                 \n"

			"   vbroadcastss    32(%%rax), %%zmm0                        \n"
			"   vdpbf16ps        %%zmm2, %%zmm4, %%zmm20                 \n"

			"   vbroadcastss    36(%%rax), %%zmm1                        \n"
			"   vdpbf16ps        %%zmm3, %%zmm4, %%zmm22                 \n"

			"   vbroadcastss    40(%%rax), %%zmm2                        \n"
			"   vdpbf16ps        %%zmm0, %%zmm4, %%zmm24                 \n"

			"   vbroadcastss    44(%%rax), %%zmm3                        \n"
			"   vdpbf16ps        %%zmm1, %%zmm4, %%zmm26                 \n"

			"   addq          $48, %%rax                                 \n"

			"   vdpbf16ps        %%zmm2, %%zmm4, %%zmm28                 \n"

			"   vdpbf16ps        %%zmm3, %%zmm4, %%zmm30                 \n"
			".endm                                                       \n"

			".macro    bf16_kernel_m12n16k2_1                            \n"
			"   vbroadcastss    8(%%rax), %%zmm2                         \n"
			"   vdpbf16ps        %%zmm0, %%zmm4, %%zmm8                  \n"

			"   vbroadcastss    12(%%rax), %%zmm3                        \n"
			"   vdpbf16ps        %%zmm1, %%zmm4, %%zmm10                 \n"

			"   prefetcht0         256(%%rax)                            \n"

			"   vbroadcastss    16(%%rax), %%zmm0                        \n"
			"   vdpbf16ps        %%zmm2, %%zmm4, %%zmm12                 \n"

			"   vbroadcastss    20(%%rax), %%zmm1                        \n"
			"   vdpbf16ps        %%zmm3, %%zmm4, %%zmm14                 \n"

			"   vbroadcastss    24(%%rax), %%zmm2                        \n"
			"   vdpbf16ps        %%zmm0, %%zmm4, %%zmm16                 \n"

			"   vbroadcastss    28(%%rax), %%zmm3                        \n"
			"   vdpbf16ps        %%zmm1, %%zmm4, %%zmm18                 \n"

			"   vbroadcastss    32(%%rax), %%zmm0                        \n"
			"   vdpbf16ps        %%zmm2, %%zmm4, %%zmm20                 \n"

			"    addq              $64, %%rbx                            \n"

			"   vbroadcastss    36(%%rax), %%zmm1                        \n"
			"   vdpbf16ps        %%zmm3, %%zmm4, %%zmm22                 \n"

			"   vbroadcastss    40(%%rax), %%zmm2                        \n"
			"   vdpbf16ps        %%zmm0, %%zmm4, %%zmm24                 \n"

			"   vbroadcastss    44(%%rax), %%zmm3                        \n"
			"   vdpbf16ps        %%zmm1, %%zmm4, %%zmm26                 \n"
			"   vmovdqu16         (%%rbx), %%zmm6                        \n"
			"    addq              $48, %%rax                            \n"

			"   vbroadcastss    (%%rax), %%zmm0                          \n"
			"   vdpbf16ps        %%zmm2, %%zmm4, %%zmm28                 \n"

			"   vbroadcastss    4(%%rax), %%zmm1                         \n"
			"   vdpbf16ps        %%zmm3, %%zmm4, %%zmm30                 \n"
			".endm                                                       \n"

			".macro    bf16_kernel_m12n16k2_2                            \n"
			"   vbroadcastss    8(%%rax), %%zmm2                         \n"
			"   vdpbf16ps        %%zmm0, %%zmm6, %%zmm8                  \n"

			"   vbroadcastss    12(%%rax), %%zmm3                        \n"
			"   vdpbf16ps        %%zmm1, %%zmm6, %%zmm10                 \n"

			"   prefetcht0         256(%%rax)                            \n"

			"   vbroadcastss    16(%%rax), %%zmm0                        \n"
			"   vdpbf16ps        %%zmm2, %%zmm6, %%zmm12                 \n"

			"   vbroadcastss    20(%%rax), %%zmm1                        \n"
			"   vdpbf16ps        %%zmm3, %%zmm6, %%zmm14                 \n"

			"   vbroadcastss    24(%%rax), %%zmm2                        \n"
			"   vdpbf16ps        %%zmm0, %%zmm6, %%zmm16                 \n"

			"   vbroadcastss    28(%%rax), %%zmm3                        \n"
			"   vdpbf16ps        %%zmm1, %%zmm6, %%zmm18                 \n"

			"   vbroadcastss    32(%%rax), %%zmm0                        \n"
			"   vdpbf16ps        %%zmm2, %%zmm6, %%zmm20                 \n"

			"   addq              $32, %%rbx                             \n"

			"   vbroadcastss    36(%%rax), %%zmm1                        \n"
			"   vdpbf16ps        %%zmm3, %%zmm6, %%zmm22                 \n"

			"   vbroadcastss    40(%%rax), %%zmm2                        \n"
			"   vdpbf16ps        %%zmm0, %%zmm6, %%zmm24                 \n"

			"   vbroadcastss    44(%%rax), %%zmm3                        \n"
			"   vdpbf16ps        %%zmm1, %%zmm6, %%zmm26                 \n"
			"   vmovdqu16         (%%rbx), %%zmm4                        \n"
			"    addq              $48, %%rax                            \n"

			"   vbroadcastss    (%%rax), %%zmm0                          \n"
			"   vdpbf16ps        %%zmm2, %%zmm6, %%zmm28                 \n"

			"   vbroadcastss    4(%%rax), %%zmm1                         \n"
			"   vdpbf16ps        %%zmm3, %%zmm6, %%zmm30                 \n"
			".endm                                                       \n"

			".macro    bf16_kernel_m12n16k2_end                          \n"
			"   vbroadcastss    8(%%rax), %%zmm2                         \n"
			"   vdpbf16ps        %%zmm0, %%zmm6, %%zmm8                  \n"

			"   vbroadcastss    12(%%rax), %%zmm3                        \n"
			"   vdpbf16ps        %%zmm1, %%zmm6, %%zmm10                 \n"

			"   prefetcht0         256(%%rax)                            \n"

			"   vbroadcastss    16(%%rax), %%zmm0                        \n"
			"   vdpbf16ps        %%zmm2, %%zmm6, %%zmm12                 \n"

			"   vbroadcastss    20(%%rax), %%zmm1                        \n"
			"   vdpbf16ps        %%zmm3, %%zmm6, %%zmm14                 \n"

			"   vbroadcastss    24(%%rax), %%zmm2                        \n"
			"   vdpbf16ps        %%zmm0, %%zmm6, %%zmm16                 \n"

			"   vbroadcastss    28(%%rax), %%zmm3                        \n"
			"   vdpbf16ps        %%zmm1, %%zmm6, %%zmm18                 \n"

			"   vbroadcastss    32(%%rax), %%zmm0                        \n"
			"   vdpbf16ps        %%zmm2, %%zmm6, %%zmm20                 \n"

			"   vbroadcastss    36(%%rax), %%zmm1                        \n"
			"   vdpbf16ps        %%zmm3, %%zmm6, %%zmm22                 \n"

			"   vbroadcastss    40(%%rax), %%zmm2                        \n"
			"   vdpbf16ps        %%zmm0, %%zmm6, %%zmm24                 \n"

			"   vbroadcastss    44(%%rax), %%zmm3                        \n"
			"   vdpbf16ps        %%zmm1, %%zmm6, %%zmm26                 \n"
			"    addq              $48, %%rax                            \n"

			"   vdpbf16ps        %%zmm2, %%zmm6, %%zmm28                 \n"

			"   vdpbf16ps        %%zmm3, %%zmm6, %%zmm30                 \n"
			".endm                                                       \n"

			".macro    bf16_save_c_m12n16                                \n"
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

			".macro    bf16_save_c_m12n16_2                              \n"
			"   vcvtne2ps2bf16  %%zmm8, %%zmm10, %%zmm0                  \n"
			"   vcvtne2ps2bf16  %%zmm12, %%zmm14, %%zmm1                 \n"
			"   vmovups         %%zmm0, (%%r10)                          \n"
			"   vmovups         %%zmm1, 64(%%r10)                        \n"

			"   addq           $128, %%r10                               \n"

			"   vcvtne2ps2bf16  %%zmm16, %%zmm18, %%zmm4                 \n"
			"   vcvtne2ps2bf16  %%zmm20, %%zmm22, %%zmm5                 \n"
			"   vmovups         %%zmm4, (%%r10)                          \n"
			"   vmovups         %%zmm5, 64(%%r10)                        \n"

			"   addq           $128, %%r10                               \n"

			"   vcvtne2ps2bf16  %%zmm24, %%zmm26, %%zmm8                 \n"
			"   vcvtne2ps2bf16  %%zmm28, %%zmm30, %%zmm9                 \n"
			"   vmovups         %%zmm8, (%%r10)                          \n"
			"   vmovups         %%zmm9, 64(%%r10)                        \n"

			"   subq           $12, %%rdi                                \n"
			"   addq           $128, %%r10                               \n"
			".endm                                                       \n"

			//-----------------------------------------------------------------

			".macro    bf16_kernel_m8n16k2_1                             \n"
			"   vbroadcastss    8(%%rax), %%zmm2                         \n"
			"   vdpbf16ps        %%zmm0, %%zmm4, %%zmm8                  \n"

			"   vbroadcastss    12(%%rax), %%zmm3                        \n"
			"   vdpbf16ps        %%zmm1, %%zmm4, %%zmm10                 \n"

			"   prefetcht0      256(%%rax)                               \n"
			"   addq             $64, %%rbx                              \n"

			"   vbroadcastss    16(%%rax), %%zmm0                        \n"
			"   vdpbf16ps        %%zmm2, %%zmm4, %%zmm12                 \n"
			"   vmovdqu16        (%%rbx), %%zmm6                         \n"

			"   vbroadcastss    20(%%rax), %%zmm1                        \n"
			"   vdpbf16ps        %%zmm3, %%zmm4, %%zmm14                 \n"

			"   vbroadcastss    24(%%rax), %%zmm2                        \n"
			"   vdpbf16ps        %%zmm0, %%zmm4, %%zmm16                 \n"

			"   vbroadcastss    28(%%rax), %%zmm3                        \n"
			"   vdpbf16ps        %%zmm1, %%zmm4, %%zmm18                 \n"

			"   addq              $32, %%rax                             \n"
			"   vbroadcastss     (%%rax), %%zmm0                         \n"
			"   vdpbf16ps        %%zmm2, %%zmm4, %%zmm20                 \n"

			"   vbroadcastss     4(%%rax), %%zmm1                        \n"
			"   vdpbf16ps        %%zmm3, %%zmm4, %%zmm22                 \n"
			".endm                                                       \n"

			".macro    bf16_kernel_m8n16k2_2                             \n"
			"   vbroadcastss    8(%%rax), %%zmm2                         \n"
			"   vdpbf16ps        %%zmm0, %%zmm6, %%zmm8                  \n"

			"   vbroadcastss    12(%%rax), %%zmm3                        \n"
			"   vdpbf16ps        %%zmm1, %%zmm6, %%zmm10                 \n"

			"   prefetcht0         256(%%rax)                            \n"
			"   addq              $64, %%rbx                             \n"

			"   vbroadcastss    16(%%rax), %%zmm0                        \n"
			"   vdpbf16ps        %%zmm2, %%zmm6, %%zmm12                 \n"

			"   vbroadcastss    20(%%rax), %%zmm1                        \n"
			"   vdpbf16ps        %%zmm3, %%zmm6, %%zmm14                 \n"

			"   vbroadcastss    24(%%rax), %%zmm2                        \n"
			"   vdpbf16ps        %%zmm0, %%zmm6, %%zmm16                 \n"
			"   vmovdqu16       (%%rbx), %%zmm4                          \n"

			"   vbroadcastss    28(%%rax), %%zmm3                        \n"
			"   vdpbf16ps        %%zmm1, %%zmm6, %%zmm18                 \n"

			"   addq             $32, %%rax                              \n"

			"   vbroadcastss    (%%rax), %%zmm0                          \n"
			"   vdpbf16ps        %%zmm2, %%zmm6, %%zmm20                 \n"

			"   vbroadcastss    4(%%rax), %%zmm1                         \n"
			"   vdpbf16ps        %%zmm3, %%zmm6, %%zmm22                 \n"
			".endm                                                       \n"

			".macro    bf16_kernel_m8n16k2_end                           \n"
			"   vbroadcastss    8(%%rax), %%zmm2                         \n"
			"   vdpbf16ps        %%zmm0, %%zmm6, %%zmm8                  \n"

			"   vbroadcastss    12(%%rax), %%zmm3                        \n"
			"   vdpbf16ps        %%zmm1, %%zmm6, %%zmm10                 \n"

			"   prefetcht0         256(%%rax)                            \n"

			"   vbroadcastss    16(%%rax), %%zmm0                        \n"
			"   vdpbf16ps        %%zmm2, %%zmm6, %%zmm12                 \n"

			"   vbroadcastss    20(%%rax), %%zmm1                        \n"
			"   vdpbf16ps        %%zmm3, %%zmm6, %%zmm14                 \n"

			"   vbroadcastss    24(%%rax), %%zmm2                        \n"
			"   vdpbf16ps        %%zmm0, %%zmm6, %%zmm16                 \n"

			"   vbroadcastss    28(%%rax), %%zmm3                        \n"
			"   vdpbf16ps        %%zmm1, %%zmm6, %%zmm18                 \n"

			"   addq             $32, %%rax                              \n"

			"   vdpbf16ps        %%zmm2, %%zmm6, %%zmm20                 \n"
			"   vdpbf16ps        %%zmm3, %%zmm6, %%zmm22                 \n"
			".endm                                                       \n"

			".macro    bf16_save_c_m8n16                                 \n"
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

			".macro    bf16_save_c_m8n16_2                               \n"
			"   vcvtne2ps2bf16  %%zmm8, %%zmm10, %%zmm0                  \n"
			"   vcvtne2ps2bf16  %%zmm12, %%zmm14, %%zmm1                 \n"
			"   vmovups         %%zmm0, (%%r10)                          \n"
			"   vmovups         %%zmm1, 64(%%r10)                        \n"

			"   addq            $128, %%r10                              \n"

			"   vcvtne2ps2bf16  %%zmm16, %%zmm18, %%zmm4                 \n"
			"   vcvtne2ps2bf16  %%zmm20, %%zmm22, %%zmm5                 \n"
			"   vmovups         %%zmm4, (%%r10)                          \n"
			"   vmovups         %%zmm5, 64(%%r10)                        \n"

			"   subq           $8, %%rdi                                 \n"
			"   addq           $128, %%r10                               \n"
			".endm                                                       \n"

			//-----------------------------------------------------------------

			".macro    bf16_kernel_m4n16k2_1                             \n"
			"   vbroadcastss    8(%%rax), %%zmm2                         \n"
			"   vdpbf16ps        %%zmm0, %%zmm4, %%zmm8                  \n"

			"   addq             $64, %%rbx                              \n"
			"   vmovdqu16        (%%rbx), %%zmm6                         \n"

			"   vbroadcastss    12(%%rax), %%zmm3                        \n"
			"   vdpbf16ps        %%zmm1, %%zmm4, %%zmm10                 \n"

			"   prefetcht0      256(%%rax)                               \n"
			"   addq              $16, %%rax                             \n"

			"   vbroadcastss     (%%rax), %%zmm0                         \n"
			"   vdpbf16ps        %%zmm2, %%zmm4, %%zmm12                 \n"

			"   vbroadcastss     4(%%rax), %%zmm1                        \n"
			"   vdpbf16ps        %%zmm3, %%zmm4, %%zmm14                 \n"
			".endm                                                       \n"

			".macro    bf16_kernel_m4n16k2_2                             \n"
			"   vbroadcastss     8(%%rax), %%zmm2                        \n"
			"   vdpbf16ps        %%zmm0, %%zmm6, %%zmm8                  \n"

			"   addq             $64, %%rbx                              \n"
			"   vmovdqu16        (%%rbx), %%zmm4                         \n"

			"   vbroadcastss     12(%%rax), %%zmm3                       \n"
			"   vdpbf16ps        %%zmm1, %%zmm6, %%zmm10                 \n"

			"   prefetcht0        256(%%rax)                             \n"
			"   addq             $16, %%rax                              \n"

			"   vbroadcastss     (%%rax), %%zmm0                         \n"
			"   vdpbf16ps        %%zmm2, %%zmm6, %%zmm12                 \n"

			"   vbroadcastss     4(%%rax), %%zmm1                        \n"
			"   vdpbf16ps        %%zmm3, %%zmm6, %%zmm14                 \n"
			".endm                                                       \n"

			".macro    bf16_kernel_m4n16k2_end                           \n"
			"   vbroadcastss    8(%%rax), %%zmm2                         \n"
			"   vdpbf16ps        %%zmm0, %%zmm6, %%zmm8                  \n"

			"   vbroadcastss    12(%%rax), %%zmm3                        \n"
			"   vdpbf16ps        %%zmm1, %%zmm6, %%zmm10                 \n"

			"   prefetcht0         256(%%rax)                            \n"

			"   addq             $16, %%rax                              \n"
			"   vbroadcastss     (%%rax), %%zmm0                         \n"
			"   vdpbf16ps        %%zmm2, %%zmm6, %%zmm12                 \n"

			"   vbroadcastss     4(%%rax), %%zmm1                        \n"
			"   vdpbf16ps        %%zmm3, %%zmm6, %%zmm14                 \n"
			".endm                                                       \n"

			".macro    bf16_save_c_m4n16                                 \n"
			"   vmovups         %%zmm8, (%%r10)%{%%k1%}                  \n"
			"   vmovups         %%zmm10, (%%r11)%{%%k1%}                 \n"
			"   vmovups         %%zmm12, (%%r12)%{%%k1%}                 \n"
			"   vmovups         %%zmm14, (%%r13)%{%%k1%}                 \n"

			"   subq            $4, %%rdi                                \n"
			"   leaq      (%%r13, %%r8, 4), %%rcx                        \n" // C0
			".endm                                                       \n"

			".macro    bf16_save_c_m4n16_2                               \n"
			"   vcvtne2ps2bf16  %%zmm8, %%zmm10, %%zmm0                  \n"
			"   vcvtne2ps2bf16  %%zmm12, %%zmm14, %%zmm1                 \n"
			"   vmovups         %%zmm0, (%%r10)                          \n"
			"   vmovups         %%zmm1, 64(%%r10)                        \n"

			"   addq           $128, %%r10                               \n"
			"   subq           $4, %%rdi                                 \n"
			".endm                                                       \n"

			//-----------------------------------------------------------------
			//-----------------------------------------------------------------

			"GEMM_BF16_N16:                                              \n"
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
			"   jz     BF16_BEGIN_M12N16                                 \n"

			//-----------------------------------------------------------------

			"BF16_BEGIN_PACK_N16:                                        \n"
			"   mov     %%r14, %%rbp                                     \n" // Bc

			"   vmovdqu16 (%%rbx), %%ymm6                                \n"
			"   leaq    (%%rbx, %%r8, 2), %%rbx                          \n"
			"   vmovdqu16 (%%rbx), %%ymm7                                \n"
			"   leaq    (%%rbx, %%r8, 2), %%rbx                          \n"

			"   mov     %%rsi, %%rdx                                     \n" // K
			"   vpxorq         %%zmm8, %%zmm8, %%zmm8                    \n"
			"   vpxorq         %%zmm10, %%zmm10, %%zmm10                 \n"
			"   vpxorq         %%zmm12, %%zmm12, %%zmm12                 \n"
			"   vpxorq         %%zmm14, %%zmm14, %%zmm14                 \n"
			"   vpxorq         %%zmm16, %%zmm16, %%zmm16                 \n"
			"   vpxorq         %%zmm18, %%zmm18, %%zmm18                 \n"
			"   vpxorq         %%zmm20, %%zmm20, %%zmm20                 \n"
			"   vpxorq         %%zmm22, %%zmm22, %%zmm22                 \n"
			"   vpxorq         %%zmm24, %%zmm24, %%zmm24                 \n"
			"   vpxorq         %%zmm26, %%zmm26, %%zmm26                 \n"
			"   vpxorq         %%zmm28, %%zmm28, %%zmm28                 \n"
			"   vpxorq         %%zmm30, %%zmm30, %%zmm30                 \n"
			"   cmp     $16, %%rdx                                       \n"
			"   jb      BF16_PACK_MAIN_M12N16K2                          \n"
			"   subq    $16, %%rdx                                       \n"

			"BF16_PACK_MAIN_M12N16K16:                                   \n"
			"   bf16_kernel_m12n16k2_pack                                \n"
			"   bf16_kernel_m12n16k2_pack                                \n"
			"   bf16_kernel_m12n16k2_pack                                \n"
			"   bf16_kernel_m12n16k2_pack                                \n"
			"   bf16_kernel_m12n16k2_pack                                \n"
			"   bf16_kernel_m12n16k2_pack                                \n"
			"   bf16_kernel_m12n16k2_pack                                \n"
			"   cmp     $0, %%rdx                                        \n"
			"   je      BF16_PACK_SAVEC_M12N16                           \n"
			"   bf16_kernel_m12n16k2_pack                                \n"
			"   cmp     $16, %%rdx                                       \n"
			"   jb      BF16_PACK_MAIN_M12N16K2                          \n"
			"   subq    $16, %%rdx                                       \n"
			"   jmp     BF16_PACK_MAIN_M12N16K16                         \n"

			"BF16_PACK_MAIN_M12N16K2:                                    \n"
			"   subq    $2, %%rdx                                        \n"
			"   cmp     $0, %%rdx                                        \n"
			"   je      BF16_PACK_SAVEC_M12N16                           \n"
			"   bf16_kernel_m12n16k2_pack                                \n"
			"   jmp     BF16_PACK_MAIN_M12N16K2                          \n"

			"BF16_PACK_SAVEC_M12N16:                                     \n"
			"    bf16_kernel_m12n16k2_pack_end                           \n"
			"    jmp        BF16_BEGIN_SAVE_M12N16                       \n"

			//-----------------------------------------------------------------

			"BF16_BEGIN_M12N16:                                          \n"
			"   cmpq    $12, %%rdi                                       \n"
			"   jb      BF16_BEGIN_M8N16                                 \n"

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
			"   cmp     $16, %%rdx                                       \n"
			"   jb      BF16_MAIN_M12N16K2                               \n"
			"   subq    $16, %%rdx                                       \n"

			"BF16_MAIN_K_M12N16K16:                                      \n" // loop K+=4
			"   bf16_kernel_m12n16k2_1                                   \n"
			"   bf16_kernel_m12n16k2_2                                   \n"
			"   bf16_kernel_m12n16k2_1                                   \n"
			"   bf16_kernel_m12n16k2_2                                   \n"
			"   bf16_kernel_m12n16k2_1                                   \n"
			"   bf16_kernel_m12n16k2_2                                   \n"
			"   bf16_kernel_m12n16k2_1                                   \n"
			"   bf16_kernel_m12n16k2_2                                   \n"
			"   cmp     $0, %%rdx                                        \n"
			"   je      BF16_BEGIN_SAVE_M12N16                           \n"
			"   cmp     $16, %%rdx                                       \n"
			"   jb      BF16_MAIN_M12N16K2                               \n"
			"   subq  $16, %%rdx                                         \n"
			"   jmp   BF16_MAIN_K_M12N16K16                              \n"

			"BF16_MAIN_M12N16K2:                                         \n"
			"   bf16_kernel_m12n16k2_1                                   \n"
			"   subq    $2, %%rdx                                        \n"
			"   cmp     $0, %%rdx                                        \n"
			"   je      BF16_BEGIN_SAVE_M12N16                           \n"
			"   bf16_kernel_m12n16k2_2                                   \n"
			"   subq    $2, %%rdx                                        \n"
			"   cmp     $0, %%rdx                                        \n"
			"   je      BF16_BEGIN_SAVE_M12N16                           \n"
			"   jmp     BF16_MAIN_M12N16K2                               \n"

			"BF16_BEGIN_SAVE_M12N16:                                     \n"
			"   mov    %[is_end_gemm], %%r13                             \n"
			"   test      $1, %%r13                                      \n"
			"   jz       BF16_SAVE_C_M12N16_2                            \n"
			"   mov      %%rcx, %%r10                                    \n" // C0
			"   leaq     (%%r10, %%r8, 4), %%r11                         \n" // C1
			"   leaq     (%%r11, %%r8, 4), %%r12                         \n" // C2
			"   leaq     (%%r12, %%r8, 4), %%r13                         \n" // C3

			"BF16_SAVE_C_M12N16:                                         \n"
			"   bf16_save_c_m12n16                                       \n"
			"   imul     $24, %%r15, %%r11                               \n" // temp use %%r11
			"   add      %%r11, %%r9                                     \n"
			"   movq     %%r9, %%rax                                     \n"
			"   jmp     BF16_BEGIN_M12N16                                \n"

			"BF16_SAVE_C_M12N16_2:                                       \n"
			"   bf16_save_c_m12n16_2                                     \n"
			"   imul     $24, %%r15, %%r11                               \n" // temp use %%r11
			"   add      %%r11, %%r9                                     \n"
			"   movq     %%r9, %%rax                                     \n"
			"   jmp     BF16_BEGIN_M12N16                                \n"

			//-----------------------------------------------------------------

			"BF16_BEGIN_M8N16:                                           \n"
			"   cmpq    $8, %%rdi                                        \n"
			"   jb      BF16_BEGIN_M4N16                                 \n"

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
			"   cmpq    $16, %%rdx                                       \n"
			"   jb      BF16_MAIN_M8N16K2                                \n"
			"   subq    $16, %%rdx                                       \n"

			"BF16_MAIN_K_M8N16K16:                                       \n"
			"   bf16_kernel_m8n16k2_1                                    \n"
			"   bf16_kernel_m8n16k2_2                                    \n"
			"   bf16_kernel_m8n16k2_1                                    \n"
			"   bf16_kernel_m8n16k2_2                                    \n"
			"   bf16_kernel_m8n16k2_1                                    \n"
			"   bf16_kernel_m8n16k2_2                                    \n"
			"   bf16_kernel_m8n16k2_1                                    \n"
			"   bf16_kernel_m8n16k2_2                                    \n"
			"   cmp     $0, %%rdx                                        \n"
			"   je      BF16_BEGIN_SAVE_M8N16                            \n"
			"   cmpq    $16, %%rdx                                       \n"
			"   jb      BF16_MAIN_M8N16K2                                \n"
			"   subq  $16, %%rdx                                         \n"
			"   jmp   BF16_MAIN_K_M8N16K16                               \n"

			"BF16_MAIN_M8N16K2:                                          \n"
			"   bf16_kernel_m8n16k2_1                                    \n"
			"   subq    $2, %%rdx                                        \n"
			"   cmp     $0, %%rdx                                        \n"
			"   je      BF16_BEGIN_SAVE_M8N16                            \n"
			"   bf16_kernel_m8n16k2_2                                    \n"
			"   subq    $2, %%rdx                                        \n"
			"   cmp     $0, %%rdx                                        \n"
			"   je      BF16_BEGIN_SAVE_M8N16                            \n"
			"   jmp     BF16_MAIN_M8N16K2                                \n"

			"BF16_BEGIN_SAVE_M8N16:                                      \n"
			"   mov    %[is_end_gemm], %%r13                             \n"
			"   test      $1, %%r13                                      \n"
			"   jz       BF16_SAVE_C_M8N16_2                             \n"
			"   mov      %%rcx, %%r10                                    \n" // C0
			"   leaq     (%%r10, %%r8, 4), %%r11                         \n" // C1
			"   leaq     (%%r11, %%r8, 4), %%r12                         \n" // C2
			"   leaq     (%%r12, %%r8, 4), %%r13                         \n" // C3

			"BF16_SAVE_C_M8N16:                                          \n"
			"   bf16_save_c_m8n16                                        \n"
			"   imul     $16, %%r15, %%r11                               \n" // temp use %%r11
			"   add      %%r11, %%r9                                     \n"
			"   movq     %%r9, %%rax                                     \n"
			"   jmp      BF16_BEGIN_M8N16                                \n"

			"BF16_SAVE_C_M8N16_2:                                        \n"
			"   bf16_save_c_m8n16_2                                      \n"
			"   imul     $16, %%r15, %%r11                               \n" // temp use %%r11
			"   add      %%r11, %%r9                                     \n"
			"   movq     %%r9, %%rax                                     \n"
			"   jmp      BF16_BEGIN_M8N16                                \n"

			//-----------------------------------------------------------------

			"BF16_BEGIN_M4N16:                                           \n"
			"   cmpq    $0, %%rdi                                        \n"
			"   je      BF16_END_N16                                     \n"
			"   mov     %%r14, %%rbx                                     \n" // Bc
			"   mov     %%rsi, %%rdx                                     \n" // K
			"   vmovups         (%%rbx), %%zmm4                          \n" // B0-15
			"   vbroadcastss    (%%rax), %%zmm0                          \n" // A0
			"   vbroadcastss    4(%%rax), %%zmm1                         \n" // A1
			"   vpxorq         %%zmm8, %%zmm8, %%zmm8                    \n"
			"   vpxorq         %%zmm10, %%zmm10, %%zmm10                 \n"
			"   vpxorq         %%zmm12, %%zmm12, %%zmm12                 \n"
			"   vpxorq         %%zmm14, %%zmm14, %%zmm14                 \n"
			"   cmpq    $16, %%rdx                                       \n"
			"   jb      BF16_MAIN_M4N16K2                                \n"
			"   subq  $16, %%rdx                                         \n"

			"BF16_MAIN_K_M4N16K16:                                       \n" // loop K+=4
			"   bf16_kernel_m4n16k2_1                                    \n"
			"   bf16_kernel_m4n16k2_2                                    \n"
			"   bf16_kernel_m4n16k2_1                                    \n"
			"   bf16_kernel_m4n16k2_2                                    \n"
			"   bf16_kernel_m4n16k2_1                                    \n"
			"   bf16_kernel_m4n16k2_2                                    \n"
			"   bf16_kernel_m4n16k2_1                                    \n"
			"   bf16_kernel_m4n16k2_2                                    \n"
			"   cmp     $0, %%rdx                                        \n"
			"   je      BF16_BEGIN_SAVE_M4N16                            \n"
			"   cmpq    $16, %%rdx                                       \n"
			"   jb      BF16_MAIN_M4N16K2                                \n"
			"   subq  $16, %%rdx                                         \n"
			"   jmp   BF16_MAIN_K_M4N16K16                               \n"

			"BF16_MAIN_M4N16K2:                                          \n"
			"   bf16_kernel_m4n16k2_1                                    \n"
			"   subq    $2, %%rdx                                        \n"
			"   cmp     $0, %%rdx                                        \n"
			"   je      BF16_BEGIN_SAVE_M4N16                            \n"
			"   bf16_kernel_m4n16k2_2                                    \n"
			"   subq    $2, %%rdx                                        \n"
			"   cmp     $0, %%rdx                                        \n"
			"   je      BF16_BEGIN_SAVE_M4N16                            \n"
			"   jmp     BF16_MAIN_M4N16K2                                \n"

			"BF16_BEGIN_SAVE_M4N16:                                      \n"
			"   mov    %[is_end_gemm], %%r13                             \n"
			"   test     $1, %%r13                                       \n"
			"   jz       BF16_SAVE_C_M4N16_2                             \n"
			"   mov      %%rcx, %%r10                                    \n" // C0
			"   leaq     (%%r10, %%r8, 4), %%r11                         \n" // C1
			"   leaq     (%%r11, %%r8, 4), %%r12                         \n" // C2
			"   leaq     (%%r12, %%r8, 4), %%r13                         \n" // C3

			"BF16_SAVE_C_M4N16:                                          \n"
			"   cmpq     $3, %%rdi                                       \n"
			"   je       BF16_SAVE_C_M3N16                               \n"
			"   cmpq     $2, %%rdi                                       \n"
			"   je       BF16_SAVE_C_M2N16                               \n"
			"   cmpq     $1, %%rdi                                       \n"
			"   je       BF16_SAVE_C_M1N16                               \n"
			"   bf16_save_c_m4n16                                        \n"
			"   imul     $8, %%r15, %%r11                                \n" // temp use %%r11
			"   add      %%r11, %%r9                                     \n"
			"   movq     %%r9, %%rax                                     \n"
			"   jmp      BF16_BEGIN_M4N16                                \n"

			"BF16_SAVE_C_M3N16:                                          \n"
			"   vmovups         %%zmm12, (%%r12)%{%%k1%}                 \n"

			"BF16_SAVE_C_M2N16:                                          \n"
			"   vmovups         %%zmm10, (%%r11)%{%%k1%}                 \n"

			"BF16_SAVE_C_M1N16:                                          \n"
			"   vmovups         %%zmm8, (%%r10)%{%%k1%}                  \n"
			"   jmp      BF16_END_N16                                    \n"

			"BF16_SAVE_C_M4N16_2:                                        \n"
			"   cmpq     $3, %%rdi                                       \n"
			"   je       BF16_SAVE_C_M3N16_2                             \n"
			"   cmpq     $2, %%rdi                                       \n"
			"   je       BF16_SAVE_C_M2N16_2                             \n"
			"   cmpq     $1, %%rdi                                       \n"
			"   je       BF16_SAVE_C_M1N16_2                             \n"
			"   bf16_save_c_m4n16_2                                      \n"
			"   imul     $8, %%r15, %%r11                                \n" // temp use %%r11
			"   add      %%r11, %%r9                                     \n"
			"   movq     %%r9, %%rax                                     \n"
			"   jmp      BF16_BEGIN_M4N16                                \n"

			"BF16_SAVE_C_M3N16_2:                                        \n"
			"   vcvtneps2bf16   %%zmm12, %%ymm2                          \n"
			"   vmovups         %%ymm2, 64(%%r10)                        \n"

			"BF16_SAVE_C_M2N16_2:                                        \n"
			"   vcvtneps2bf16   %%zmm10, %%ymm1                          \n"
			"   vmovups         %%ymm1, 32(%%r10)                        \n"

			"BF16_SAVE_C_M1N16_2:                                        \n"
			"   vcvtneps2bf16   %%zmm8, %%ymm0                           \n"
			"   vmovups         %%ymm0, (%%r10)                          \n"

			"BF16_END_N16:                                               \n"

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