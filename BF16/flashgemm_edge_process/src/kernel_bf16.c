#include <stdint.h>

static void FLASHGEMM_BF16_KERNELm12xn32xk2(float *C, uint16_t *A, uint16_t *B, long M, long N, long K, long LN, uint16_t *Bc, long k_tag) // edge case m4n32k2
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

			"   prefetcht0         64(%%rbx)                             \n"

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

			"   prefetcht0         64(%%rbx)                             \n"

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

			".macro    bf16_add_c_m12n32                                 \n"
			"   vmovups         (%%r10), %%zmm0                          \n"
			"   vaddps             %%zmm0, %%zmm8, %%zmm8                \n"
			"   vmovups         64(%%r10), %%zmm1                        \n"
			"   vaddps             %%zmm1, %%zmm9, %%zmm9                \n"
			"   vmovups         (%%r11), %%zmm2                          \n"
			"   vaddps             %%zmm2, %%zmm10, %%zmm10              \n"
			"   vmovups         64(%%r11), %%zmm3                        \n"
			"   vaddps             %%zmm3, %%zmm11, %%zmm11              \n"
			"   vmovups         (%%r12), %%zmm4                          \n"
			"   vaddps             %%zmm4, %%zmm12, %%zmm12              \n"
			"   vmovups         64(%%r12), %%zmm5                        \n"
			"   vaddps             %%zmm5, %%zmm13, %%zmm13              \n"
			"   vmovups         (%%r13), %%zmm6                          \n"
			"   vaddps             %%zmm6, %%zmm14, %%zmm14              \n"
			"   vmovups         64(%%r13), %%zmm7                        \n"
			"   vaddps             %%zmm7, %%zmm15, %%zmm15              \n"

			"   leaq          (%%r13, %%r8, 4), %%r10                    \n" // C0
			"   leaq             (%%r10, %%r8, 4), %%r11                 \n" // C1
			"   leaq             (%%r11, %%r8, 4), %%r12                 \n" // C2
			"   leaq             (%%r12, %%r8, 4), %%r13                 \n" // C3

			"   vmovups         (%%r10), %%zmm0                          \n"
			"   vaddps             %%zmm0, %%zmm16, %%zmm16              \n"
			"   vmovups         64(%%r10), %%zmm1                        \n"
			"   vaddps             %%zmm1, %%zmm17, %%zmm17              \n"
			"   vmovups         (%%r11), %%zmm2                          \n"
			"   vaddps             %%zmm2, %%zmm18, %%zmm18              \n"
			"   vmovups         64(%%r11), %%zmm3                        \n"
			"   vaddps             %%zmm3, %%zmm19, %%zmm19              \n"

			"   vmovups         (%%r12), %%zmm4                          \n"
			"   vaddps             %%zmm4, %%zmm20, %%zmm20              \n"
			"   vmovups         64(%%r12), %%zmm5                        \n"
			"   vaddps             %%zmm5, %%zmm21, %%zmm21              \n"
			"   vmovups         (%%r13), %%zmm6                          \n"
			"   vaddps             %%zmm6, %%zmm22, %%zmm22              \n"
			"   vmovups         64(%%r13), %%zmm7                        \n"
			"   vaddps             %%zmm7, %%zmm23, %%zmm23              \n"

			"   leaq          (%%r13, %%r8, 4), %%r10                    \n" // C0
			"   leaq             (%%r10, %%r8, 4), %%r11                 \n" // C1
			"   leaq             (%%r11, %%r8, 4), %%r12                 \n" // C2
			"   leaq             (%%r12, %%r8, 4), %%r13                 \n" // C3

			"   vmovups         (%%r10), %%zmm0                          \n"
			"   vaddps             %%zmm0, %%zmm24, %%zmm24              \n"
			"   vmovups         64(%%r10), %%zmm1                        \n"
			"   vaddps             %%zmm1, %%zmm25, %%zmm25              \n"
			"   vmovups         (%%r11), %%zmm2                          \n"
			"   vaddps             %%zmm2, %%zmm26, %%zmm26              \n"
			"   vmovups         64(%%r11), %%zmm3                        \n"
			"   vaddps             %%zmm3, %%zmm27, %%zmm27              \n"

			"   vmovups         (%%r12), %%zmm4                          \n"
			"   vaddps             %%zmm4, %%zmm28, %%zmm28              \n"
			"   vmovups         64(%%r12), %%zmm5                        \n"
			"   vaddps             %%zmm5, %%zmm29, %%zmm29              \n"
			"   vmovups         (%%r13), %%zmm6                          \n"
			"   vaddps             %%zmm6, %%zmm30, %%zmm30              \n"
			"   vmovups         64(%%r13), %%zmm7                        \n"
			"   vaddps             %%zmm7, %%zmm31, %%zmm31              \n"

			"   mov      %%rcx, %%r10                                    \n" // C0
			"   leaq     (%%r10, %%r8, 4), %%r11                         \n" // C1
			"   leaq     (%%r11, %%r8, 4), %%r12                         \n" // C2
			"   leaq     (%%r12, %%r8, 4), %%r13                         \n" // C3
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

			"    leaq  (%%r13, %%r8, 4), %%r10                           \n" // C0
			"    leaq     (%%r10, %%r8, 4), %%r11                        \n" // C1
			"    leaq     (%%r11, %%r8, 4), %%r12                        \n" // C2
			"    leaq     (%%r12, %%r8, 4), %%r13                        \n" // C3

			"   vmovups         %%zmm16, (%%r10)                         \n"
			"   vmovups         %%zmm17, 64(%%r10)                       \n"
			"   vmovups         %%zmm18, (%%r11)                         \n"
			"   vmovups         %%zmm19, 64(%%r11)                       \n"
			"   vmovups         %%zmm20, (%%r12)                         \n"
			"   vmovups         %%zmm21, 64(%%r12)                       \n"
			"   vmovups         %%zmm22, (%%r13)                         \n"
			"   vmovups         %%zmm23, 64(%%r13)                       \n"

			"    leaq  (%%r13, %%r8, 4), %%r10                           \n" // C0
			"    leaq     (%%r10, %%r8, 4), %%r11                        \n" // C1
			"    leaq     (%%r11, %%r8, 4), %%r12                        \n" // C2
			"    leaq     (%%r12, %%r8, 4), %%r13                        \n" // C3

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
			"   prefetcht0       64(%%rbx)                               \n"

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
			"   prefetcht0         64(%%rbx)                             \n"

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

			".macro    bf16_add_c_m8n32                                  \n"
			"   vmovups         (%%r10), %%zmm0                          \n"
			"   vaddps             %%zmm0, %%zmm8, %%zmm8                \n"
			"   vmovups         64(%%r10), %%zmm1                        \n"
			"   vaddps             %%zmm1, %%zmm9, %%zmm9                \n"
			"   vmovups         (%%r11), %%zmm2                          \n"
			"   vaddps             %%zmm2, %%zmm10, %%zmm10              \n"
			"   vmovups         64(%%r11), %%zmm3                        \n"
			"   vaddps             %%zmm3, %%zmm11, %%zmm11              \n"
			"   vmovups         (%%r12), %%zmm4                          \n"
			"   vaddps             %%zmm4, %%zmm12, %%zmm12              \n"
			"   vmovups         64(%%r12), %%zmm5                        \n"
			"   vaddps             %%zmm5, %%zmm13, %%zmm13              \n"
			"   vmovups         (%%r13), %%zmm6                          \n"
			"   vaddps             %%zmm6, %%zmm14, %%zmm14              \n"
			"   vmovups         64(%%r13), %%zmm7                        \n"
			"   vaddps             %%zmm7, %%zmm15, %%zmm15              \n"

			"   leaq          (%%r13, %%r8, 4), %%r10                    \n" // C0
			"   leaq             (%%r10, %%r8, 4), %%r11                 \n" // C1
			"   leaq             (%%r11, %%r8, 4), %%r12                 \n" // C2
			"   leaq             (%%r12, %%r8, 4), %%r13                 \n" // C3

			"   vmovups         (%%r10), %%zmm0                          \n"
			"   vaddps             %%zmm0, %%zmm16, %%zmm16              \n"
			"   vmovups         64(%%r10), %%zmm1                        \n"
			"   vaddps             %%zmm1, %%zmm17, %%zmm17              \n"
			"   vmovups         (%%r11), %%zmm2                          \n"
			"   vaddps             %%zmm2, %%zmm18, %%zmm18              \n"
			"   vmovups         64(%%r11), %%zmm3                        \n"
			"   vaddps             %%zmm3, %%zmm19, %%zmm19              \n"

			"   vmovups         (%%r12), %%zmm4                          \n"
			"   vaddps             %%zmm4, %%zmm20, %%zmm20              \n"
			"   vmovups         64(%%r12), %%zmm5                        \n"
			"   vaddps             %%zmm5, %%zmm21, %%zmm21              \n"
			"   vmovups         (%%r13), %%zmm6                          \n"
			"   vaddps             %%zmm6, %%zmm22, %%zmm22              \n"
			"   vmovups         64(%%r13), %%zmm7                        \n"
			"   vaddps             %%zmm7, %%zmm23, %%zmm23              \n"

			"   mov               %%rcx, %%r10                           \n" // C0
			"   leaq             (%%r10, %%r8, 4), %%r11                 \n" // C1
			"   leaq             (%%r11, %%r8, 4), %%r12                 \n" // C2
			"   leaq             (%%r12, %%r8, 4), %%r13                 \n" // C3
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

			//-----------------------------------------------------------------

			".macro    bf16_kernel_m4n32k2_1                             \n"
			"   vbroadcastss    8(%%rax), %%zmm2                         \n"
			"   vdpbf16ps        %%zmm0, %%zmm4, %%zmm8                  \n"
			"   vdpbf16ps        %%zmm0, %%zmm5, %%zmm9                  \n"
			"   addq             $128, %%rbx                             \n"
			"   prefetcht0       64(%%rbx)                               \n"
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
			"   prefetcht0       64(%%rbx)                               \n"
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

			".macro    bf16_add_c_m4n32                                  \n"
			"   vmovups         (%%r10), %%zmm0                          \n"
			"   vaddps             %%zmm0, %%zmm8, %%zmm8                \n"
			"   vmovups         64(%%r10), %%zmm1                        \n"
			"   vaddps             %%zmm1, %%zmm9, %%zmm9                \n"
			"   vmovups         (%%r11), %%zmm2                          \n"
			"   vaddps             %%zmm2, %%zmm10, %%zmm10              \n"
			"   vmovups         64(%%r11), %%zmm3                        \n"
			"   vaddps             %%zmm3, %%zmm11, %%zmm11              \n"
			"   vmovups         (%%r12), %%zmm4                          \n"
			"   vaddps             %%zmm4, %%zmm12, %%zmm12              \n"
			"   vmovups         64(%%r12), %%zmm5                        \n"
			"   vaddps             %%zmm5, %%zmm13, %%zmm13              \n"
			"   vmovups         (%%r13), %%zmm6                          \n"
			"   vaddps             %%zmm6, %%zmm14, %%zmm14              \n"
			"   vmovups         64(%%r13), %%zmm7                        \n"
			"   vaddps             %%zmm7, %%zmm15, %%zmm15              \n"

			"   mov               %%rcx, %%r10                           \n" // C0
			"   leaq             (%%r10, %%r8, 4), %%r11                 \n" // C1
			"   leaq             (%%r11, %%r8, 4), %%r12                 \n" // C2
			"   leaq             (%%r12, %%r8, 4), %%r13                 \n" // C3
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

			//-----------------------------------------------------------------
			//-----------------------------------------------------------------

			"GEMM_BF16_N32:                                              \n"
			"   mov     %[C], %%rcx                                      \n"
			"   mov     %[A], %%rax                                      \n"
			"   mov     %[B], %%rbx                                      \n"

			"   prefetcht0         (%%rax)                               \n"

			"   mov     %[K], %%rdx                                      \n"
			"   mov     %[LN], %%r8                                      \n"
			"   mov     %[Bc], %%r14                                     \n"
			"   mov     %[M], %%rdi                                      \n"
			"   mov     %[k_tag], %%r15                                  \n"

			"   prefetcht0         (%%rbx)                               \n"
			"   mov     %%rbx, %%r9                                      \n"
			"   mov     %%rdx, %%rsi                                     \n"

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
			"   mov     %%rcx, %%r13                                     \n"
			"   subq    $16, %%rdx                                       \n"

			"BF16_PACK_MAIN_M12N32:                                      \n"
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
			"   subq    $16, %%rdx                                       \n"
			"   cmp     $192, %%rdx                                      \n" // 192/16 = 12 rows of C data
			"   jbe     BF16_PACK_K_PREFETCH_C_M12N32                    \n" // prefeth C before loop end
			"   jmp     BF16_PACK_MAIN_M12N32                            \n"

			"BF16_PACK_K_PREFETCH_C_M12N32:                              \n"
			"   prefetcht2         (%%r13)                               \n"
			"   prefetcht2         64(%%r13)                             \n"
			"   leaq     (%%r13, %%r8, 4), %%r13                         \n"
			"   jmp BF16_PACK_MAIN_M12N32                                \n"

			"BF16_PACK_SAVEC_M12N32:                                     \n"
			"    bf16_kernel_m12n32k2_pack_end                           \n"
			"    jmp        BF16_BEGIN_SAVE_M12N32                       \n"

			//-----------------------------------------------------------------

			"BF16_BEGIN_M12:                                             \n"
			"   cmpq    $12, %%rdi                                       \n"
			"   jb      BF16_BEGIN_M8                                    \n"

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
			"   mov     %%rcx, %%r13                                     \n"
			"   subq    $16, %%rdx                                       \n"

			"BF16_MAIN_K_M12N32:                                         \n" // loop K+=4
			"   bf16_kernel_m12n32k2_1                                   \n"
			"   bf16_kernel_m12n32k2_2                                   \n"
			"   bf16_kernel_m12n32k2_1                                   \n"
			"   bf16_kernel_m12n32k2_2                                   \n"
			"   bf16_kernel_m12n32k2_1                                   \n"
			"   bf16_kernel_m12n32k2_2                                   \n"
			"   bf16_kernel_m12n32k2_1                                   \n"
			"   cmp     $0, %%rdx                                        \n"
			"   je      BF16_MAIN_K_M12N32_EDGE                          \n"
			"   bf16_kernel_m12n32k2_2                                   \n"
			"   subq  $16, %%rdx                                         \n"
			"   cmp   $192, %%rdx                                        \n" // 192/16 = 12 rows of C data
			"   jbe   BF16_K_PREFETCH_C_M12N32                           \n"
			"   jmp   BF16_MAIN_K_M12N32                                 \n"

			"BF16_K_PREFETCH_C_M12N32:                                   \n"
			"   prefetcht2         (%%r13)                               \n"
			"   prefetcht2         64(%%r13)                             \n"
			"   leaq     (%%r13, %%r8, 4), %%r13                         \n"
			"   jmp BF16_MAIN_K_M12N32                                   \n"

			"BF16_MAIN_K_M12N32_EDGE:                                    \n"
			"   bf16_kernel_m12n32k2_end                                 \n"

			"BF16_BEGIN_SAVE_M12N32:                                     \n"
			"   mov      %%rcx, %%r10                                    \n" // C0
			"   leaq     (%%r10, %%r8, 4), %%r11                         \n" // C1
			"   leaq     (%%r11, %%r8, 4), %%r12                         \n" // C2
			"   leaq     (%%r12, %%r8, 4), %%r13                         \n" // C3
			"   cmp      $0, %%r15                                       \n"
			"   je       BF16_SAVE_C_M12N32                              \n"
			"   bf16_add_c_m12n32                                        \n"

			"BF16_SAVE_C_M12N32:                                         \n"
			"   bf16_save_c_m12n32                                       \n"
			"   jmp     BF16_BEGIN_M12                                   \n"

			//-----------------------------------------------------------------

			"BF16_BEGIN_M8:                                              \n"
			"   cmpq    $8, %%rdi                                        \n"
			"   jb      BF16_BEGIN_M4                                    \n"

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
			"   mov     %%rcx, %%r13                                     \n"
			"   subq    $16, %%rdx                                       \n"

			"BF16_MAIN_K_M8N32:                                          \n"
			"   bf16_kernel_m8n32k2_1                                    \n"
			"   bf16_kernel_m8n32k2_2                                    \n"
			"   bf16_kernel_m8n32k2_1                                    \n"
			"   bf16_kernel_m8n32k2_2                                    \n"
			"   bf16_kernel_m8n32k2_1                                    \n"
			"   bf16_kernel_m8n32k2_2                                    \n"
			"   bf16_kernel_m8n32k2_1                                    \n"
			"   cmp     $0, %%rdx                                        \n"
			"   je      BF16_MAIN_K_M8N32_EDGE                           \n"
			"   bf16_kernel_m8n32k2_2                                    \n"
			"   subq  $16, %%rdx                                         \n"
			"   cmp   $128, %%rdx                                        \n" // 128/16 = 8 rows of C data
			"   jbe   BF16_K_PREFETCH_C_M8N32                            \n"
			"   jmp   BF16_MAIN_K_M8N32                                  \n"

			"BF16_K_PREFETCH_C_M8N32:                                    \n"
			"   prefetcht2         (%%r13)                               \n"
			"   prefetcht2         64(%%r13)                             \n"
			"   leaq     (%%r13, %%r8, 4), %%r13                         \n"
			"   jmp BF16_MAIN_K_M8N32                                    \n"

			"BF16_MAIN_K_M8N32_EDGE:                                     \n"
			"   bf16_kernel_m8n32k2_end                                  \n"

			"BF16_BEGIN_SAVE_M8N32:                                      \n"
			"   mov      %%rcx, %%r10                                    \n" // C0
			"   leaq     (%%r10, %%r8, 4), %%r11                         \n" // C1
			"   leaq     (%%r11, %%r8, 4), %%r12                         \n" // C2
			"   leaq     (%%r12, %%r8, 4), %%r13                         \n" // C3
			"   cmp      $0, %%r15                                       \n"
			"   je       BF16_SAVE_C_M8N32                               \n"
			"   bf16_add_c_m8n32                                         \n"

			"BF16_SAVE_C_M8N32:                                          \n"
			"   bf16_save_c_m8n32                                        \n"
			"   jmp     BF16_END_N32                                     \n"

			//-----------------------------------------------------------------

			"BF16_BEGIN_M4:                                              \n"
			"   cmpq    $4, %%rdi                                        \n"
			"   jb      BF16_END_N32                                     \n"

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

			"   mov     %%rcx, %%r13                                     \n"
			"   subq  $16, %%rdx                                         \n"

			"BF16_MAIN_K_M4N32:                                          \n" // loop K+=4
			"   bf16_kernel_m4n32k2_1                                    \n"
			"   bf16_kernel_m4n32k2_2                                    \n"
			"   bf16_kernel_m4n32k2_1                                    \n"
			"   bf16_kernel_m4n32k2_2                                    \n"
			"   bf16_kernel_m4n32k2_1                                    \n"
			"   bf16_kernel_m4n32k2_2                                    \n"
			"   bf16_kernel_m4n32k2_1                                    \n"
			"   cmp     $0, %%rdx                                        \n"
			"   je      BF16_MAIN_K_M4N32_EDGE                           \n"
			"   bf16_kernel_m4n32k2_2                                    \n"
			"   subq  $16, %%rdx                                         \n"
			"   cmp   $64, %%rdx                                         \n" // 64/16 = 4 rows of C data
			"   jbe   BF16_K_PREFETCH_C_M4N32                            \n"
			"   jmp   BF16_MAIN_K_M4N32                                  \n"

			"BF16_K_PREFETCH_C_M4N32:                                    \n"
			"   prefetcht2         (%%r13)                               \n"
			"   prefetcht2         64(%%r13)                             \n"
			"   leaq     (%%r13, %%r8, 4), %%r13                         \n"
			"   jmp BF16_MAIN_K_M4N32                                    \n"

			"BF16_MAIN_K_M4N32_EDGE:                                     \n"
			"   bf16_kernel_m4n32k2_end                                  \n"

			"BF16_BEGIN_SAVE_M4N32:                                      \n"
			"   mov      %%rcx, %%r10                                    \n" // C0
			"   leaq     (%%r10, %%r8, 4), %%r11                         \n" // C1
			"   leaq     (%%r11, %%r8, 4), %%r12                         \n" // C2
			"   leaq     (%%r12, %%r8, 4), %%r13                         \n" // C3
			"   cmp      $0, %%r15                                       \n"
			"   je       BF16_SAVE_C_M4N32                               \n"
			"   bf16_add_c_m4n32                                         \n"

			"BF16_SAVE_C_M4N32:                                          \n"
			"   bf16_save_c_m4n32                                        \n"
			"   jmp     BF16_END_N32                                     \n"

			"BF16_END_N32:                                               \n"

			:
			:
			[C] "m"(C),
			[A] "m"(A),
			[B] "m"(B),
			[M] "m"(M),
			[N] "m"(N),
			[K] "m"(K),
			[LN] "m"(LN),
			[Bc] "m"(Bc),
			[k_tag] "m"(k_tag)
			: "rax", "rbx", "rcx", "rdx", "rdi", "rsi", "rbp", "r8", "r9", "r10", "r11", "r12",
				"r13", "r14", "r15", "zmm0", "zmm1", "zmm2", "zmm3", "zmm4", "zmm5",
				"zmm6", "zmm7", "zmm8", "zmm9", "zmm10", "zmm11", "zmm12", "zmm13",
				"zmm14", "zmm15", "zmm16", "zmm17", "zmm18", "zmm19", "zmm20", "zmm21",
				"zmm22", "zmm23", "zmm24", "zmm25", "zmm26", "zmm27", "zmm28", "zmm29",
				"zmm30", "zmm31", "memory", "xmm0", "xmm1", "xmm2", "xmm3", "xmm6", "xmm7");
}
