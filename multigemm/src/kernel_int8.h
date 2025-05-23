#include <cstdint>

void FLASHGEMM_INT8_KERNELm12xn32(int *C, uint8_t *Cc, int8_t *A, uint8_t *B, long M, long K, long LK, long LN, uint8_t *Bc, bool is_start_gemm, bool is_end_gemm)
{
	asm volatile(
			".macro int8_pack_b_n32                                      \n"
			"  vmovups (%%rbx), %%ymm0                                   \n"
			"  prefetcht2  64(%%rbx)                                     \n"
			"  leaq    (%%rbx, %%r8, 1), %%rbx                           \n"
			"  vmovups (%%rbx), %%ymm1                                   \n"
			"  prefetcht2  64(%%rbx)                                     \n"
			"  leaq    (%%rbx, %%r8, 1), %%rbx                           \n"

			"  vpunpcklbw  %%ymm1, %%ymm0, %%ymm4                        \n"
			"  vpunpckhbw  %%ymm1, %%ymm0, %%ymm5                        \n"

			"  vmovups (%%rbx), %%ymm2                                   \n"
			"  prefetcht2  64(%%rbx)                                     \n"
			"  leaq    (%%rbx, %%r8, 1), %%rbx                           \n"
			"  vmovups (%%rbx), %%ymm3                                   \n"
			"  prefetcht2  64(%%rbx)                                     \n"
			"  leaq    (%%rbx, %%r8, 1), %%rbx                           \n"

			"  vpunpcklbw  %%ymm3, %%ymm2, %%ymm6                        \n"
			"  vpunpckhbw  %%ymm3, %%ymm2, %%ymm7                        \n"

			"  vpunpcklwd  %%ymm6, %%ymm4, %%ymm0                        \n"
			"  vpunpckhwd  %%ymm6, %%ymm4, %%ymm1                        \n"
			"  vpunpcklwd  %%ymm7, %%ymm5, %%ymm2                        \n"
			"  vpunpckhwd  %%ymm7, %%ymm5, %%ymm3                        \n"

			"  vextracti32x4   $0x1, %%ymm0, %%xmm4                      \n"
			"  vextracti32x4   $0x1, %%ymm1, %%xmm5                      \n"
			"  vextracti32x4   $0x1, %%ymm2, %%xmm6                      \n"
			"  vextracti32x4   $0x1, %%ymm3, %%xmm7                      \n"

			"  vinserti32x4    $0x1, %%xmm1, %%zmm0, %%zmm0              \n"
			"  vinserti32x4    $0x2, %%xmm2, %%zmm0, %%zmm0              \n"
			"  vbroadcastss    (%%rax), %%zmm2                           \n" // A0
			"  vinserti32x4    $0x3, %%xmm3, %%zmm0, %%zmm0              \n"
			"  vbroadcastss    4(%%rax), %%zmm3                          \n" // A1
			"  vmovups         %%zmm0, (%%rbp)                           \n"

			"  vinserti32x4    $0x1, %%xmm5, %%zmm4, %%zmm4              \n"
			"  vinserti32x4    $0x2, %%xmm6, %%zmm4, %%zmm4              \n"
			"  vinserti32x4    $0x3, %%xmm7, %%zmm4, %%zmm4              \n"
			"  vmovups         %%zmm4, 64(%%rbp)                         \n"
			"  addq            $128, %%rbp                               \n"

			".endm                                                       \n"

			".macro    int8_kernel_m12n32k4_pack                         \n"
			"  int8_pack_b_n32                                           \n"

			"  vbroadcastss    8(%%rax), %%zmm5                          \n" // A2
			"  vpdpbusd    %%zmm0, %%zmm2, %%zmm8                        \n" // A0*(B0-15)
			"  vpdpbusd    %%zmm4, %%zmm2, %%zmm9                        \n" // A0*(B16-31)

			"  vbroadcastss    12(%%rax), %%zmm6                         \n" // A3
			"  vpdpbusd    %%zmm0, %%zmm3, %%zmm10                       \n" // A1*(B0-15)
			"  vpdpbusd    %%zmm4, %%zmm3, %%zmm11                       \n" // A1*(B16-31)

			"  prefetcht0         256(%%rax)                             \n"

			"  vbroadcastss    16(%%rax), %%zmm2                         \n" // A4
			"  vpdpbusd    %%zmm0, %%zmm5, %%zmm12                       \n"
			"  vpdpbusd    %%zmm4, %%zmm5, %%zmm13                       \n"

			"  vbroadcastss    20(%%rax), %%zmm3                         \n" // A5
			"  vpdpbusd    %%zmm0, %%zmm6, %%zmm14                       \n"
			"  vpdpbusd    %%zmm4, %%zmm6, %%zmm15                       \n"

			"  vbroadcastss    24(%%rax), %%zmm5                         \n" // A6
			"  vpdpbusd    %%zmm0, %%zmm2, %%zmm16                       \n"
			"  vpdpbusd    %%zmm4, %%zmm2, %%zmm17                       \n"

			"  vbroadcastss    28(%%rax), %%zmm6                         \n" // A7
			"  vpdpbusd    %%zmm0, %%zmm3, %%zmm18                       \n"
			"  vpdpbusd    %%zmm4, %%zmm3, %%zmm19                       \n"

			"  vbroadcastss    32(%%rax), %%zmm2                         \n" // A8
			"  vpdpbusd    %%zmm0, %%zmm5, %%zmm20                       \n"
			"  vpdpbusd    %%zmm4, %%zmm5, %%zmm21                       \n"

			"  vbroadcastss    36(%%rax), %%zmm3                         \n" // A9
			"  vpdpbusd    %%zmm0, %%zmm6, %%zmm22                       \n"
			"  vpdpbusd    %%zmm4, %%zmm6, %%zmm23                       \n"

			"  vbroadcastss    40(%%rax), %%zmm5                         \n" // A10
			"  vpdpbusd    %%zmm0, %%zmm2, %%zmm24                       \n"
			"  vpdpbusd    %%zmm4, %%zmm2, %%zmm25                       \n"

			"  prefetcht0         384(%%rax)                             \n"

			"  vbroadcastss    44(%%rax), %%zmm6                         \n" // A11
			"  addq              $48, %%rax                              \n"
			"  vpdpbusd    %%zmm0, %%zmm3, %%zmm26                       \n"
			"  vpdpbusd    %%zmm4, %%zmm3, %%zmm27                       \n"

			"  vbroadcastss    (%%rax), %%zmm2                           \n" // next A0
			"  vpdpbusd    %%zmm0, %%zmm5, %%zmm28                       \n"
			"  vpdpbusd    %%zmm4, %%zmm5, %%zmm29                       \n"

			"  vbroadcastss    4(%%rax), %%zmm3                          \n" // next A1
			"  vpdpbusd    %%zmm0, %%zmm6, %%zmm30                       \n"
			"  vpdpbusd    %%zmm4, %%zmm6, %%zmm31                       \n"

			".endm                                                       \n"

			".macro    int8_kernel_m12n32k4_k1                           \n"

			"   vbroadcastss    8(%%rax), %%zmm2                         \n"
			"   vpdpbusd    %%zmm4, %%zmm0, %%zmm8                       \n"
			"   vpdpbusd    %%zmm5, %%zmm0, %%zmm9                       \n"

			"   vbroadcastss    12(%%rax), %%zmm3                        \n"

			"   vpdpbusd    %%zmm4, %%zmm1, %%zmm10                      \n"
			"   vpdpbusd    %%zmm5, %%zmm1, %%zmm11                      \n"

			"   prefetcht0         256(%%rax)                            \n"

			"   vbroadcastss    16(%%rax), %%zmm0                        \n"
			"   vpdpbusd    %%zmm4, %%zmm2, %%zmm12                      \n"
			"   vpdpbusd    %%zmm5, %%zmm2, %%zmm13                      \n"

			"   vbroadcastss    20(%%rax), %%zmm1                        \n"
			"   vpdpbusd    %%zmm4, %%zmm3, %%zmm14                      \n"
			"   vpdpbusd    %%zmm5, %%zmm3, %%zmm15                      \n"

			"   vbroadcastss    24(%%rax), %%zmm2                        \n"
			"   vpdpbusd    %%zmm4, %%zmm0, %%zmm16                      \n"
			"   vpdpbusd    %%zmm5, %%zmm0, %%zmm17                      \n"

			"   vbroadcastss    28(%%rax), %%zmm3                        \n"
			"   vpdpbusd    %%zmm4, %%zmm1, %%zmm18                      \n"
			"   vpdpbusd    %%zmm5, %%zmm1, %%zmm19                      \n"

			"   vbroadcastss    32(%%rax), %%zmm0                        \n"
			"   vpdpbusd    %%zmm4, %%zmm2, %%zmm20                      \n"
			"   vpdpbusd    %%zmm5, %%zmm2, %%zmm21                      \n"

			"    addq              $128, %%rbx                           \n"

			"   vbroadcastss    36(%%rax), %%zmm1                        \n"
			"   vpdpbusd    %%zmm4, %%zmm3, %%zmm22                      \n"
			"   vpdpbusd    %%zmm5, %%zmm3, %%zmm23                      \n"

			"   prefetcht0         64(%%rbx)                             \n"

			"   vbroadcastss    40(%%rax), %%zmm2                        \n"
			"   vpdpbusd    %%zmm4, %%zmm0, %%zmm24                      \n"
			"   vpdpbusd    %%zmm5, %%zmm0, %%zmm25                      \n"

			"   vbroadcastss    44(%%rax), %%zmm3                        \n"
			"   vpdpbusd    %%zmm4, %%zmm1, %%zmm26                      \n"
			"   vmovups         (%%rbx), %%zmm6                          \n"
			"    addq              $48, %%rax                            \n"
			"   vpdpbusd    %%zmm5, %%zmm1, %%zmm27                      \n"

			"   vbroadcastss    (%%rax), %%zmm0                          \n"
			"   vpdpbusd    %%zmm4, %%zmm2, %%zmm28                      \n"
			"   vmovups         64(%%rbx), %%zmm7                        \n"
			"   vpdpbusd    %%zmm5, %%zmm2, %%zmm29                      \n"

			"   vbroadcastss    4(%%rax), %%zmm1                         \n"
			"   vpdpbusd    %%zmm4, %%zmm3, %%zmm30                      \n"
			"   vpdpbusd    %%zmm5, %%zmm3, %%zmm31                      \n"

			".endm                                                       \n"

			".macro    int8_kernel_m12n32k4_k2                           \n"

			"   vbroadcastss    8(%%rax), %%zmm2                         \n"
			"   vpdpbusd    %%zmm6, %%zmm0, %%zmm8                       \n"
			"   vpdpbusd    %%zmm7, %%zmm0, %%zmm9                       \n"

			"   vbroadcastss    12(%%rax), %%zmm3                        \n"
			"   vpdpbusd    %%zmm6, %%zmm1, %%zmm10                      \n"
			"   vpdpbusd    %%zmm7, %%zmm1, %%zmm11                      \n"

			"   prefetcht0         256(%%rax)                            \n"

			"   vbroadcastss    16(%%rax), %%zmm0                        \n"
			"   vpdpbusd    %%zmm6, %%zmm2, %%zmm12                      \n"
			"   vpdpbusd    %%zmm7, %%zmm2, %%zmm13                      \n"

			"   vbroadcastss    20(%%rax), %%zmm1                        \n"
			"   vpdpbusd    %%zmm6, %%zmm3, %%zmm14                      \n"
			"   vpdpbusd    %%zmm7, %%zmm3, %%zmm15                      \n"

			"   vbroadcastss    24(%%rax), %%zmm2                        \n"
			"   vpdpbusd    %%zmm6, %%zmm0, %%zmm16                      \n"
			"   vpdpbusd    %%zmm7, %%zmm0, %%zmm17                      \n"

			"   vbroadcastss    28(%%rax), %%zmm3                        \n"
			"   vpdpbusd    %%zmm6, %%zmm1, %%zmm18                      \n"
			"   vpdpbusd    %%zmm7, %%zmm1, %%zmm19                      \n"

			"   vbroadcastss    32(%%rax), %%zmm0                        \n"
			"   vpdpbusd    %%zmm6, %%zmm2, %%zmm20                      \n"
			"   vpdpbusd    %%zmm7, %%zmm2, %%zmm21                      \n"

			"    addq              $128, %%rbx                           \n"

			"   vbroadcastss    36(%%rax), %%zmm1                        \n"
			"   vpdpbusd    %%zmm6, %%zmm3, %%zmm22                      \n"
			"   vpdpbusd    %%zmm7, %%zmm3, %%zmm23                      \n"

			"   prefetcht0         64(%%rbx)                             \n"

			"   vbroadcastss    40(%%rax), %%zmm2                        \n"
			"   vpdpbusd    %%zmm6, %%zmm0, %%zmm24                      \n"
			"   vpdpbusd    %%zmm7, %%zmm0, %%zmm25                      \n"

			"   vbroadcastss    44(%%rax), %%zmm3                        \n"
			"   vpdpbusd    %%zmm6, %%zmm1, %%zmm26                      \n"
			"   vmovups         (%%rbx), %%zmm4                          \n"
			"    addq              $48, %%rax                            \n"
			"   vpdpbusd    %%zmm7, %%zmm1, %%zmm27                      \n"

			"   vbroadcastss    (%%rax), %%zmm0                          \n"
			"   vpdpbusd    %%zmm6, %%zmm2, %%zmm28                      \n"
			"   vmovups         64(%%rbx), %%zmm5                        \n"
			"   vpdpbusd    %%zmm7, %%zmm2, %%zmm29                      \n"

			"   vbroadcastss    4(%%rax), %%zmm1                         \n"
			"   vpdpbusd    %%zmm6, %%zmm3, %%zmm30                      \n"
			"   vpdpbusd    %%zmm7, %%zmm3, %%zmm31                      \n"

			".endm                                                       \n"

			".macro    int8_kernel_m12n32k4_end                          \n"

			"   vbroadcastss    8(%%rax), %%zmm2                         \n"
			"   vpdpbusd    %%zmm6, %%zmm0, %%zmm8                       \n"
			"   vpdpbusd    %%zmm7, %%zmm0, %%zmm9                       \n"
			"   vbroadcastss    12(%%rax), %%zmm3                        \n"

			"   vpdpbusd    %%zmm6, %%zmm1, %%zmm10                      \n"
			"   vpdpbusd    %%zmm7, %%zmm1, %%zmm11                      \n"

			"   prefetcht0         256(%%rax)                            \n"

			"   vbroadcastss    16(%%rax), %%zmm0                        \n"
			"   vpdpbusd    %%zmm6, %%zmm2, %%zmm12                      \n"
			"   vpdpbusd    %%zmm7, %%zmm2, %%zmm13                      \n"

			"   vbroadcastss    20(%%rax), %%zmm1                        \n"
			"   vpdpbusd    %%zmm6, %%zmm3, %%zmm14                      \n"
			"   vpdpbusd    %%zmm7, %%zmm3, %%zmm15                      \n"

			"   vbroadcastss    24(%%rax), %%zmm2                        \n"
			"   vpdpbusd    %%zmm6, %%zmm0, %%zmm16                      \n"
			"   vpdpbusd    %%zmm7, %%zmm0, %%zmm17                      \n"

			"   vbroadcastss    28(%%rax), %%zmm3                        \n"
			"   vpdpbusd    %%zmm6, %%zmm1, %%zmm18                      \n"
			"   vpdpbusd    %%zmm7, %%zmm1, %%zmm19                      \n"

			"   vbroadcastss    32(%%rax), %%zmm0                        \n"
			"   vpdpbusd    %%zmm6, %%zmm2, %%zmm20                      \n"
			"   vpdpbusd    %%zmm7, %%zmm2, %%zmm21                      \n"

			"   vbroadcastss    36(%%rax), %%zmm1                        \n"
			"   vpdpbusd    %%zmm6, %%zmm3, %%zmm22                      \n"
			"   vpdpbusd    %%zmm7, %%zmm3, %%zmm23                      \n"

			"   vbroadcastss    40(%%rax), %%zmm2                        \n"
			"   vpdpbusd    %%zmm6, %%zmm0, %%zmm24                      \n"
			"   vpdpbusd    %%zmm7, %%zmm0, %%zmm25                      \n"

			"   vbroadcastss    44(%%rax), %%zmm3                        \n"
			"   vpdpbusd    %%zmm6, %%zmm1, %%zmm26                      \n"
			"    addq              $48, %%rax                            \n"
			"   vpdpbusd    %%zmm7, %%zmm1, %%zmm27                      \n"

			"   vpdpbusd    %%zmm6, %%zmm2, %%zmm28                      \n"
			"   vpdpbusd    %%zmm7, %%zmm2, %%zmm29                      \n"

			"   vpdpbusd    %%zmm6, %%zmm3, %%zmm30                      \n"
			"   vpdpbusd    %%zmm7, %%zmm3, %%zmm31                      \n"

			".endm                                                       \n"

			".macro    int8_save_c_m12n32                                \n"
			"  vmovups         %%zmm8, (%%r10)                           \n"
			"  vmovups         %%zmm9, 64(%%r10)                         \n"
			"  vmovups         %%zmm10, (%%r11)                          \n"
			"  vmovups         %%zmm11, 64(%%r11)                        \n"
			"  vmovups         %%zmm12, (%%r12)                          \n"
			"  vmovups         %%zmm13, 64(%%r12)                        \n"
			"  vmovups         %%zmm14, (%%r13)                          \n"
			"  vmovups         %%zmm15, 64(%%r13)                        \n"

			"  leaq     (%%r13, %%r8, 4), %%r10                          \n" // C0
			"  leaq     (%%r10, %%r8, 4), %%r11                          \n" // C1
			"  leaq     (%%r11, %%r8, 4), %%r12                          \n" // C2
			"  leaq     (%%r12, %%r8, 4), %%r13                          \n" // C3

			"  vmovups         %%zmm16, (%%r10)                          \n"
			"  vmovups         %%zmm17, 64(%%r10)                        \n"
			"  vmovups         %%zmm18, (%%r11)                          \n"
			"  vmovups         %%zmm19, 64(%%r11)                        \n"
			"  vmovups         %%zmm20, (%%r12)                          \n"
			"  vmovups         %%zmm21, 64(%%r12)                        \n"
			"  vmovups         %%zmm22, (%%r13)                          \n"
			"  vmovups         %%zmm23, 64(%%r13)                        \n"

			"  leaq     (%%r13, %%r8, 4), %%r10                          \n" // C0
			"  leaq     (%%r10, %%r8, 4), %%r11                          \n" // C1
			"  leaq     (%%r11, %%r8, 4), %%r12                          \n" // C2
			"  leaq     (%%r12, %%r8, 4), %%r13                          \n" // C3

			"  vmovups         %%zmm24, (%%r10)                          \n"
			"  vmovups         %%zmm25, 64(%%r10)                        \n"
			"  vmovups         %%zmm26, (%%r11)                          \n"
			"  vmovups         %%zmm27, 64(%%r11)                        \n"
			"  subq             $12, %%rdi                               \n"
			"  vmovups         %%zmm28, (%%r12)                          \n"
			"  vmovups         %%zmm29, 64(%%r12)                        \n"
			"  vmovups         %%zmm30, (%%r13)                          \n"
			"  vmovups         %%zmm31, 64(%%r13)                        \n"

			"  leaq      (%%r13, %%r8, 4), %%rcx                         \n" // C0

			".endm                                                       \n"

			".macro    int8_save_c_m12n32_2                              \n"
			"  vpmovsdb   %%zmm8, %%xmm0                                 \n"
			"  vpmovsdb   %%zmm9, %%xmm1                                 \n"
			"  vpmovsdb   %%zmm10, %%xmm2                                \n"
			"  vpmovsdb   %%zmm11, %%xmm3                                \n"
			"  vinserti32x4    $0x1, %%xmm1, %%zmm0, %%zmm0              \n"
			"  vinserti32x4    $0x2, %%xmm2, %%zmm0, %%zmm0              \n"
			"  vinserti32x4    $0x3, %%xmm3, %%zmm0, %%zmm0              \n"
			"  vmovups         %%zmm0, (%%r10)                           \n"
			"  vpmovsdb   %%zmm12, %%xmm4                                \n"
			"  vpmovsdb   %%zmm13, %%xmm5                                \n"
			"  vpmovsdb   %%zmm14, %%xmm6                                \n"
			"  vpmovsdb   %%zmm15, %%xmm7                                \n"
			"  vinserti32x4    $0x1, %%xmm5, %%zmm4, %%zmm4              \n"
			"  vinserti32x4    $0x2, %%xmm6, %%zmm4, %%zmm4              \n"
			"  vinserti32x4    $0x3, %%xmm7, %%zmm4, %%zmm4              \n"
			"  vmovups         %%zmm4, 64(%%r10)                         \n"
			"  addq            $128, %%r10                               \n"

			"  vpmovsdb   %%zmm16, %%xmm8                                \n"
			"  vpmovsdb   %%zmm17, %%xmm9                                \n"
			"  vpmovsdb   %%zmm18, %%xmm10                               \n"
			"  vpmovsdb   %%zmm19, %%xmm11                               \n"
			"  vinserti32x4    $0x1, %%xmm9, %%zmm8, %%zmm8              \n"
			"  vinserti32x4    $0x2, %%xmm10, %%zmm8, %%zmm8             \n"
			"  vinserti32x4    $0x3, %%xmm11, %%zmm8, %%zmm8             \n"
			"  vmovups         %%zmm8, (%%r10)                           \n"
			"  vpmovsdb   %%zmm20, %%xmm12                               \n"
			"  vpmovsdb   %%zmm21, %%xmm13                               \n"
			"  vpmovsdb   %%zmm22, %%xmm14                               \n"
			"  vpmovsdb   %%zmm23, %%xmm15                               \n"
			"  vinserti32x4    $0x1, %%xmm13, %%zmm12, %%zmm12           \n"
			"  vinserti32x4    $0x2, %%xmm14, %%zmm12, %%zmm12           \n"
			"  vinserti32x4    $0x3, %%xmm15, %%zmm12, %%zmm12           \n"
			"  vmovups         %%zmm12, 64(%%r10)                        \n"
			"  addq            $128, %%r10                               \n"

			"  vpmovsdb   %%zmm24, %%xmm16                               \n"
			"  vpmovsdb   %%zmm25, %%xmm17                               \n"
			"  vpmovsdb   %%zmm26, %%xmm18                               \n"
			"  vpmovsdb   %%zmm27, %%xmm19                               \n"
			"  vinserti32x4    $0x1, %%xmm17, %%zmm16, %%zmm16           \n"
			"  vinserti32x4    $0x2, %%xmm18, %%zmm16, %%zmm16           \n"
			"  vinserti32x4    $0x3, %%xmm19, %%zmm16, %%zmm16           \n"
			"  vmovups         %%zmm16, (%%r10)                          \n"
			"  vpmovsdb   %%zmm28, %%xmm20                               \n"
			"  vpmovsdb   %%zmm29, %%xmm21                               \n"
			"  vpmovsdb   %%zmm30, %%xmm22                               \n"
			"  vpmovsdb   %%zmm31, %%xmm23                               \n"
			"  vinserti32x4    $0x1, %%xmm21, %%zmm20, %%zmm20           \n"
			"  vinserti32x4    $0x2, %%xmm22, %%zmm20, %%zmm20           \n"
			"  vinserti32x4    $0x3, %%xmm23, %%zmm20, %%zmm20           \n"
			"  vmovups         %%zmm20, 64(%%r10)                        \n"
			"  addq           $128, %%r10                                \n"

			"  subq           $12, %%rdi                                 \n"
			".endm                                                       \n"

			//-----------------------------------------------------------------

			".macro    int8_kernel_m8n32k4_k1                            \n"
			"  vbroadcastss    8(%%rax), %%zmm2                          \n"
			"  vpdpbusd    %%zmm4, %%zmm0, %%zmm8                        \n"
			"  vpdpbusd    %%zmm5, %%zmm0, %%zmm9                        \n"

			"  vbroadcastss    12(%%rax), %%zmm3                         \n"

			"  vpdpbusd    %%zmm4, %%zmm1, %%zmm10                       \n"
			"  vpdpbusd    %%zmm5, %%zmm1, %%zmm11                       \n"

			"  prefetcht0         256(%%rax)                             \n"

			"  vbroadcastss    16(%%rax), %%zmm0                         \n"
			"  vpdpbusd    %%zmm4, %%zmm2, %%zmm12                       \n"
			"  vpdpbusd    %%zmm5, %%zmm2, %%zmm13                       \n"

			"  vmovups         (%%rbx), %%zmm6                           \n"

			"  vbroadcastss    20(%%rax), %%zmm1                         \n"
			"  vpdpbusd    %%zmm4, %%zmm3, %%zmm14                       \n"
			"  vpdpbusd    %%zmm5, %%zmm3, %%zmm15                       \n"

			"  vbroadcastss    24(%%rax), %%zmm2                         \n"
			"  vpdpbusd    %%zmm4, %%zmm0, %%zmm16                       \n"
			"  vpdpbusd    %%zmm5, %%zmm0, %%zmm17                       \n"

			"  vmovups         64(%%rbx), %%zmm7                         \n"

			"  vbroadcastss    28(%%rax), %%zmm3                         \n"
			"  vpdpbusd    %%zmm4, %%zmm1, %%zmm18                       \n"
			"  vpdpbusd    %%zmm5, %%zmm1, %%zmm19                       \n"

			"  addq              $32, %%rax                              \n"

			"  vbroadcastss    (%%rax), %%zmm0                           \n"
			"  vpdpbusd    %%zmm4, %%zmm2, %%zmm20                       \n"
			"  vpdpbusd    %%zmm5, %%zmm2, %%zmm21                       \n"

			"  addq              $128, %%rbx                             \n"

			"  vbroadcastss    4(%%rax), %%zmm1                          \n"
			"  vpdpbusd    %%zmm4, %%zmm3, %%zmm22                       \n"
			"  vpdpbusd    %%zmm5, %%zmm3, %%zmm23                       \n"
			".endm                                                       \n"

			".macro    int8_kernel_m8n32k4_k2                            \n"
			"  vbroadcastss    8(%%rax), %%zmm2                          \n"
			"  vpdpbusd    %%zmm6, %%zmm0, %%zmm8                        \n"
			"  vpdpbusd    %%zmm7, %%zmm0, %%zmm9                        \n"

			"  vbroadcastss    12(%%rax), %%zmm3                         \n"

			"  vpdpbusd    %%zmm6, %%zmm1, %%zmm10                       \n"
			"  vpdpbusd    %%zmm7, %%zmm1, %%zmm11                       \n"

			"  prefetcht0         256(%%rax)                             \n"

			"  vbroadcastss    16(%%rax), %%zmm0                         \n"
			"  vpdpbusd    %%zmm6, %%zmm2, %%zmm12                       \n"
			"  vpdpbusd    %%zmm7, %%zmm2, %%zmm13                       \n"

			"  vmovups         (%%rbx), %%zmm4                           \n"

			"  vbroadcastss    20(%%rax), %%zmm1                         \n"
			"  vpdpbusd    %%zmm6, %%zmm3, %%zmm14                       \n"
			"  vpdpbusd    %%zmm7, %%zmm3, %%zmm15                       \n"

			"  vbroadcastss    24(%%rax), %%zmm2                         \n"
			"  vpdpbusd    %%zmm6, %%zmm0, %%zmm16                       \n"
			"  vpdpbusd    %%zmm7, %%zmm0, %%zmm17                       \n"

			"  vmovups         64(%%rbx), %%zmm5                         \n"

			"  vbroadcastss    28(%%rax), %%zmm3                         \n"
			"  vpdpbusd    %%zmm6, %%zmm1, %%zmm18                       \n"
			"  vpdpbusd    %%zmm7, %%zmm1, %%zmm19                       \n"

			"  addq              $32, %%rax                              \n"

			"  vbroadcastss    (%%rax), %%zmm0                           \n"
			"  vpdpbusd    %%zmm6, %%zmm2, %%zmm20                       \n"
			"  vpdpbusd    %%zmm7, %%zmm2, %%zmm21                       \n"

			"  addq              $128, %%rbx                             \n"

			"  vbroadcastss    4(%%rax), %%zmm1                          \n"
			"  vpdpbusd    %%zmm6, %%zmm3, %%zmm22                       \n"
			"  vpdpbusd    %%zmm7, %%zmm3, %%zmm23                       \n"
			".endm                                                       \n"

			".macro    int8_kernel_m8n32k4_end                           \n"
			"  vbroadcastss    8(%%rax), %%zmm2                          \n"
			"  vpdpbusd    %%zmm6, %%zmm0, %%zmm8                        \n"
			"  vpdpbusd    %%zmm7, %%zmm0, %%zmm9                        \n"

			"  vbroadcastss    12(%%rax), %%zmm3                         \n"
			"  vpdpbusd    %%zmm6, %%zmm1, %%zmm10                       \n"
			"  vpdpbusd    %%zmm7, %%zmm1, %%zmm11                       \n"

			"  prefetcht0         256(%%rax)                             \n"

			"  vbroadcastss    16(%%rax), %%zmm0                         \n"
			"  vpdpbusd    %%zmm6, %%zmm2, %%zmm12                       \n"
			"  vpdpbusd    %%zmm7, %%zmm2, %%zmm13                       \n"

			"  vbroadcastss    20(%%rax), %%zmm1                         \n"
			"  vpdpbusd    %%zmm6, %%zmm3, %%zmm14                       \n"
			"  vpdpbusd    %%zmm7, %%zmm3, %%zmm15                       \n"

			"  vbroadcastss    24(%%rax), %%zmm2                         \n"
			"  vpdpbusd    %%zmm6, %%zmm0, %%zmm16                       \n"
			"  vpdpbusd    %%zmm7, %%zmm0, %%zmm17                       \n"

			"  vbroadcastss    28(%%rax), %%zmm3                         \n"
			"  vpdpbusd    %%zmm6, %%zmm1, %%zmm18                       \n"
			"  vpdpbusd    %%zmm7, %%zmm1, %%zmm19                       \n"

			"  addq              $32, %%rax                              \n"

			"  vpdpbusd    %%zmm6, %%zmm2, %%zmm20                       \n"
			"  vpdpbusd    %%zmm7, %%zmm2, %%zmm21                       \n"

			"  vpdpbusd    %%zmm6, %%zmm3, %%zmm22                       \n"
			"  vpdpbusd    %%zmm7, %%zmm3, %%zmm23                       \n"
			".endm                                                       \n"

			".macro    int8_save_c_m8n32                                 \n"
			"  vmovups         %%zmm8, (%%r10)                           \n"
			"  vmovups         %%zmm9, 64(%%r10)                         \n"
			"  vmovups         %%zmm10, (%%r11)                          \n"
			"  vmovups         %%zmm11, 64(%%r11)                        \n"
			"  vmovups         %%zmm12, (%%r12)                          \n"
			"  vmovups         %%zmm13, 64(%%r12)                        \n"
			"  vmovups         %%zmm14, (%%r13)                          \n"
			"  vmovups         %%zmm15, 64(%%r13)                        \n"

			"  leaq     (%%r13, %%r8, 4), %%r10                          \n" // C0
			"  leaq     (%%r10, %%r8, 4), %%r11                          \n" // C1
			"  leaq     (%%r11, %%r8, 4), %%r12                          \n" // C2
			"  leaq     (%%r12, %%r8, 4), %%r13                          \n" // C3

			"  vmovups         %%zmm16, (%%r10)                          \n"
			"  vmovups         %%zmm17, 64(%%r10)                        \n"
			"  vmovups         %%zmm18, (%%r11)                          \n"
			"  vmovups         %%zmm19, 64(%%r11)                        \n"
			"  vmovups         %%zmm20, (%%r12)                          \n"
			"  vmovups         %%zmm21, 64(%%r12)                        \n"
			"  vmovups         %%zmm22, (%%r13)                          \n"
			"  vmovups         %%zmm23, 64(%%r13)                        \n"

			"  subq             $8, %%rdi                                \n"
			"  leaq      (%%r13, %%r8, 4), %%rcx                         \n" // C0
			".endm                                                       \n"

			".macro    int8_save_c_m8n32_2                               \n"
			"   vpmovsdb   %%zmm8, %%xmm0                                \n"
			"   vpmovsdb   %%zmm9, %%xmm1                                \n"
			"   vpmovsdb   %%zmm10, %%xmm2                               \n"
			"   vpmovsdb   %%zmm11, %%xmm3                               \n"
			"   vinserti32x4    $0x1, %%xmm1, %%zmm0, %%zmm0             \n"
			"   vinserti32x4    $0x2, %%xmm2, %%zmm0, %%zmm0             \n"
			"   vinserti32x4    $0x3, %%xmm3, %%zmm0, %%zmm0             \n"
			"   vmovups         %%zmm0, (%%r10)                          \n"
			"   vpmovsdb   %%zmm12, %%xmm4                               \n"
			"   vpmovsdb   %%zmm13, %%xmm5                               \n"
			"   vpmovsdb   %%zmm14, %%xmm6                               \n"
			"   vpmovsdb   %%zmm15, %%xmm7                               \n"
			"   vinserti32x4    $0x1, %%xmm5, %%zmm4, %%zmm4             \n"
			"   vinserti32x4    $0x2, %%xmm6, %%zmm4, %%zmm4             \n"
			"   vinserti32x4    $0x3, %%xmm7, %%zmm4, %%zmm4             \n"
			"   vmovups         %%zmm4, 64(%%r10)                        \n"
			"   addq            $128, %%r10                              \n"

			"   vpmovsdb   %%zmm16, %%xmm8                               \n"
			"   vpmovsdb   %%zmm17, %%xmm9                               \n"
			"   vpmovsdb   %%zmm18, %%xmm10                              \n"
			"   vpmovsdb   %%zmm19, %%xmm11                              \n"
			"   vinserti32x4    $0x1, %%xmm9, %%zmm8, %%zmm8             \n"
			"   vinserti32x4    $0x2, %%xmm10, %%zmm8, %%zmm8            \n"
			"   vinserti32x4    $0x3, %%xmm11, %%zmm8, %%zmm8            \n"
			"   vmovups         %%zmm8, (%%r10)                          \n"
			"   vpmovsdb   %%zmm20, %%xmm12                              \n"
			"   vpmovsdb   %%zmm21, %%xmm13                              \n"
			"   vpmovsdb   %%zmm22, %%xmm14                              \n"
			"   vpmovsdb   %%zmm23, %%xmm15                              \n"
			"   vinserti32x4    $0x1, %%xmm13, %%zmm12, %%zmm12          \n"
			"   vinserti32x4    $0x2, %%xmm14, %%zmm12, %%zmm12          \n"
			"   vinserti32x4    $0x3, %%xmm15, %%zmm12, %%zmm12          \n"
			"   vmovups         %%zmm12, 64(%%r10)                       \n"
			"   addq            $128, %%r10                              \n"

			"   subq           $8, %%rdi                                 \n"
			".endm                                                       \n"

			//-----------------------------------------------------------------

			".macro    int8_kernel_m4n32k4_k1                            \n"
			"  vbroadcastss    8(%%rax), %%zmm2                          \n"
			"  vpdpbusd    %%zmm4, %%zmm0, %%zmm8                        \n"
			"  vpdpbusd    %%zmm5, %%zmm0, %%zmm9                        \n"

			"  vmovups         (%%rbx), %%zmm6                           \n"

			"  vbroadcastss    12(%%rax), %%zmm3                         \n"
			"  vpdpbusd    %%zmm4, %%zmm1, %%zmm10                       \n"
			"  vpdpbusd    %%zmm5, %%zmm1, %%zmm11                       \n"

			"  vmovups         64(%%rbx), %%zmm7                         \n"
			"  prefetcht0      256(%%rax)                                \n"

			"  addq            $16, %%rax                                \n"
			"  vbroadcastss    (%%rax), %%zmm0                           \n"
			"  vpdpbusd    %%zmm4, %%zmm2, %%zmm12                       \n"
			"  vpdpbusd    %%zmm5, %%zmm2, %%zmm13                       \n"

			"  vbroadcastss    4(%%rax), %%zmm1                          \n"
			"  vpdpbusd    %%zmm4, %%zmm3, %%zmm14                       \n"
			"  vpdpbusd    %%zmm5, %%zmm3, %%zmm15                       \n"

			"  addq              $128, %%rbx                             \n"
			".endm                                                       \n"

			".macro    int8_kernel_m4n32k4_k2                            \n"
			"  vbroadcastss    8(%%rax), %%zmm2                          \n"
			"  vpdpbusd    %%zmm6, %%zmm0, %%zmm8                        \n"
			"  vpdpbusd    %%zmm7, %%zmm0, %%zmm9                        \n"

			"  vmovups         (%%rbx), %%zmm4                           \n"

			"  vbroadcastss    12(%%rax), %%zmm3                         \n"
			"  vpdpbusd    %%zmm6, %%zmm1, %%zmm10                       \n"
			"  vpdpbusd    %%zmm7, %%zmm1, %%zmm11                       \n"

			"  vmovups         64(%%rbx), %%zmm5                         \n"
			"  prefetcht0      256(%%rax)                                \n"

			"  addq            $16, %%rax                                \n"
			"  vbroadcastss    (%%rax), %%zmm0                           \n"
			"  vpdpbusd    %%zmm6, %%zmm2, %%zmm12                       \n"
			"  vpdpbusd    %%zmm7, %%zmm2, %%zmm13                       \n"

			"  vbroadcastss    4(%%rax), %%zmm1                          \n"
			"  vpdpbusd    %%zmm6, %%zmm3, %%zmm14                       \n"
			"  vpdpbusd    %%zmm7, %%zmm3, %%zmm15                       \n"

			"  addq              $128, %%rbx                             \n"
			".endm                                                       \n"

			".macro    int8_kernel_m4n32k4_end                           \n"
			"  broadcastss     8(%%rax), %%zmm2                          \n"
			"  vpdpbusd    %%zmm6, %%zmm0, %%zmm8                        \n"
			"  vpdpbusd    %%zmm7, %%zmm0, %%zmm9                        \n"

			"  vbroadcastss    12(%%rax), %%zmm3                         \n"
			"  vpdpbusd    %%zmm6, %%zmm1, %%zmm10                       \n"
			"  vpdpbusd    %%zmm7, %%zmm1, %%zmm11                       \n"

			"  prefetcht0      256(%%rax)                                \n"

			"  vpdpbusd    %%zmm6, %%zmm2, %%zmm12                       \n"
			"  vpdpbusd    %%zmm7, %%zmm2, %%zmm13                       \n"

			"  vpdpbusd    %%zmm6, %%zmm3, %%zmm14                       \n"
			"  vpdpbusd    %%zmm7, %%zmm3, %%zmm15                       \n"
			".endm                                                       \n"

			".macro    int8_save_c_m4n32                                 \n"
			"  vmovups         %%zmm8, (%%r10)                           \n"
			"  vmovups         %%zmm9, 64(%%r10)                         \n"
			"  vmovups         %%zmm10, (%%r11)                          \n"
			"  vmovups         %%zmm11, 64(%%r11)                        \n"
			"  vmovups         %%zmm12, (%%r12)                          \n"
			"  vmovups         %%zmm13, 64(%%r12)                        \n"
			"  vmovups         %%zmm14, (%%r13)                          \n"
			"  vmovups         %%zmm15, 64(%%r13)                        \n"

			"  subq             $4, %%rdi                                \n"
			"  leaq      (%%r13, %%r8, 4), %%rcx                         \n" // C0
			".endm                                                       \n"

			".macro    int8_save_c_m4n32_2                               \n"
			"   vpmovsdb   %%zmm8, %%xmm16                               \n"
			"   vpmovsdb   %%zmm9, %%xmm17                               \n"
			"   vpmovsdb   %%zmm10, %%xmm18                              \n"
			"   vpmovsdb   %%zmm11, %%xmm19                              \n"
			"   vinserti32x4    $0x1, %%xmm17, %%zmm16, %%zmm16          \n"
			"   vinserti32x4    $0x2, %%xmm18, %%zmm16, %%zmm16          \n"
			"   vinserti32x4    $0x3, %%xmm19, %%zmm16, %%zmm16          \n"
			"   vmovups         %%zmm16, (%%r10)                         \n"
			"   vpmovsdb   %%zmm12, %%xmm20                              \n"
			"   vpmovsdb   %%zmm13, %%xmm21                              \n"
			"   vpmovsdb   %%zmm14, %%xmm22                              \n"
			"   vpmovsdb   %%zmm15, %%xmm23                              \n"
			"   vinserti32x4    $0x1, %%xmm21, %%zmm20, %%zmm20          \n"
			"   vinserti32x4    $0x2, %%xmm22, %%zmm20, %%zmm20          \n"
			"   vinserti32x4    $0x3, %%xmm23, %%zmm20, %%zmm20          \n"
			"   vmovups         %%zmm20, 64(%%r10)                       \n"
			"   addq            $128, %%r10                              \n"
			"   subq            $4, %%rdi                                \n"
			".endm                                                       \n"

			//-----------------------------------------------------------------

			"GEMM_INT8_N32:                                              \n"
			"  mov     %[C], %%rcx                                       \n"
			"  mov     %[Cc], %%r10                                      \n"
			"  mov     %[A], %%rax                                       \n"
			"  mov     %[B], %%rbx                                       \n"

			"  prefetcht0         (%%rax)                                \n"

			"  mov     %[K], %%rdx                                       \n"
			"  mov     %[LN], %%r8                                       \n"
			"  mov     %[Bc], %%r14                                      \n"
			"  mov     %[M], %%rdi                                       \n"

			"  mov     %[LK], %%r15                                      \n"
			"  mov     %%rax, %%r9                                       \n"
			"  prefetcht0         (%%rbx)                                \n"
			"  mov     %%rdx, %%rsi                                      \n"

			"  mov    %[is_start_gemm], %%r12                            \n"
			"  test   $1, %%r12                                          \n"
			"  jz     INT8_BEGIN_M12N32                                  \n"

			"INT8_BEGIN_PACK_N32:                                        \n"
			"   mov     %%r14, %%rbp                                     \n" // Bc
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
			"   cmp     $32, %%rdx                                       \n"
			"   jb      INT8_PACK_MAIN_M12N32K4                          \n"
			"   subq    $32, %%rdx                                       \n"

			"INT8_PACK_MAIN_M12N32K32:                                   \n"
			"   int8_kernel_m12n32k4_pack                                \n"
			"   int8_kernel_m12n32k4_pack                                \n"
			"   int8_kernel_m12n32k4_pack                                \n"
			"   int8_kernel_m12n32k4_pack                                \n"
			"   int8_kernel_m12n32k4_pack                                \n"
			"   int8_kernel_m12n32k4_pack                                \n"
			"   int8_kernel_m12n32k4_pack                                \n"
			"   int8_kernel_m12n32k4_pack                                \n"
			"   cmp     $0, %%rdx                                        \n"
			"   je      INT8_BEGIN_SAVE_M12N32                           \n"
			"   cmp     $32, %%rdx                                       \n"
			"   jb      INT8_PACK_MAIN_M12N32K4                          \n"
			"   subq    $32, %%rdx                                       \n"
			"   jmp     INT8_PACK_MAIN_M12N32K32                         \n"

			"INT8_PACK_MAIN_M12N32K4:                                    \n"
			"   subq    $4, %%rdx                                        \n"
			"   cmp     $0, %%rdx                                        \n"
			"   je      INT8_BEGIN_SAVE_M12N32                           \n"
			"   int8_kernel_m12n32k4_pack                                \n"
			"   jmp     INT8_PACK_MAIN_M12N32K4                          \n"

			//-----------------------------------------------------------------

			"INT8_BEGIN_M12N32:                                          \n"
			"   cmpq    $12, %%rdi                                       \n"
			"   jb      INT8_BEGIN_M8N32                                 \n"

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
			"   cmp     $32, %%rdx                                       \n"
			"   jb      INT8_MAIN_M12N32K4                               \n"
			"   subq    $32, %%rdx                                       \n"

			"INT8_MAIN_M12N32K32:                                        \n"
			"   int8_kernel_m12n32k4_k1                                  \n"
			"   int8_kernel_m12n32k4_k2                                  \n"
			"   int8_kernel_m12n32k4_k1                                  \n"
			"   int8_kernel_m12n32k4_k2                                  \n"
			"   int8_kernel_m12n32k4_k1                                  \n"
			"   int8_kernel_m12n32k4_k2                                  \n"
			"   int8_kernel_m12n32k4_k1                                  \n"
			"   int8_kernel_m12n32k4_k2                                  \n"
			"   cmp     $0, %%rdx                                        \n"
			"   je      INT8_BEGIN_SAVE_M12N32                           \n"
			"   cmp     $32, %%rdx                                       \n"
			"   jb      INT8_MAIN_M12N32K4                               \n"
			"   subq  $32, %%rdx                                         \n"
			"   jmp   INT8_MAIN_M12N32K32                                \n"

			"INT8_MAIN_M12N32K4:                                         \n"
			"   int8_kernel_m12n32k4_k1                                  \n"
			"   subq    $4, %%rdx                                        \n"
			"   cmp     $0, %%rdx                                        \n"
			"   je      INT8_BEGIN_SAVE_M12N32                           \n"
			"   int8_kernel_m12n32k4_k2                                  \n"
			"   subq    $4, %%rdx                                        \n"
			"   cmp     $0, %%rdx                                        \n"
			"   je      INT8_BEGIN_SAVE_M12N32                           \n"
			"   jmp     INT8_MAIN_M12N32K4                               \n"

			"INT8_BEGIN_SAVE_M12N32:                                     \n"
			"   mov      %[is_end_gemm], %%r13                           \n"
			"   test     $1, %%r13                                       \n"
			"   jz       INT8_SAVE_C_M12N32_2                            \n"
			"   mov      %%rcx, %%r10                                    \n" // C0
			"   leaq     (%%r10, %%r8, 4), %%r11                         \n" // C1
			"   leaq     (%%r11, %%r8, 4), %%r12                         \n" // C2
			"   leaq     (%%r12, %%r8, 4), %%r13                         \n" // C3

			"INT8_SAVE_C_M12N32:                                         \n"
			"  int8_save_c_m12n32                                        \n"
			"  imul     $12, %%r15, %%r11                                \n" // temp use %%r11
			"  add      %%r11, %%r9                                      \n"
			"  movq     %%r9, %%rax                                      \n"
			"  jmp      INT8_BEGIN_M12N32                                \n"

			"INT8_SAVE_C_M12N32_2:                                       \n"
			"  int8_save_c_m12n32_2                                      \n"
			"  imul     $12, %%r15, %%r11                                \n" // temp use %%r11
			"  add      %%r11, %%r9                                      \n"
			"  movq     %%r9, %%rax                                      \n"
			"  jmp      INT8_BEGIN_M12N32                                \n"

			//-----------------------------------------------------------------

			"INT8_BEGIN_M8N32:                                           \n"
			"   cmpq    $8, %%rdi                                        \n"
			"   jb      INT8_BEGIN_M4N32                                 \n"

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
			"   cmp     $32, %%rdx                                       \n"
			"   jb      INT8_MAIN_M8N32K4                                \n"
			"   subq    $32, %%rdx                                       \n"

			"INT8_MAIN_M8N32K32:                                         \n"
			"   int8_kernel_m8n32k4_k1                                   \n"
			"   int8_kernel_m8n32k4_k2                                   \n"
			"   int8_kernel_m8n32k4_k1                                   \n"
			"   int8_kernel_m8n32k4_k2                                   \n"
			"   int8_kernel_m8n32k4_k1                                   \n"
			"   int8_kernel_m8n32k4_k2                                   \n"
			"   int8_kernel_m8n32k4_k1                                   \n"
			"   int8_kernel_m8n32k4_k2                                   \n"
			"   cmp     $0, %%rdx                                        \n"
			"   je      INT8_BEGIN_SAVE_M8N32                            \n"
			"   cmp     $32, %%rdx                                       \n"
			"   jb      INT8_MAIN_M8N32K4                                \n"
			"   subq  $32, %%rdx                                         \n"
			"   jmp   INT8_MAIN_M8N32K32                                 \n"

			"INT8_MAIN_M8N32K4:                                          \n"
			"   int8_kernel_m8n32k4_k1                                   \n"
			"   subq    $4, %%rdx                                        \n"
			"   cmp     $0, %%rdx                                        \n"
			"   je      INT8_BEGIN_SAVE_M8N32                            \n"
			"   int8_kernel_m8n32k4_k2                                   \n"
			"   subq    $4, %%rdx                                        \n"
			"   cmp     $0, %%rdx                                        \n"
			"   je      INT8_BEGIN_SAVE_M8N32                            \n"
			"   jmp     INT8_MAIN_M8N32K4                                \n"

			"INT8_BEGIN_SAVE_M8N32:                                      \n"
			"   mov      %[is_end_gemm], %%r13                           \n"
			"   test     $1, %%r13                                       \n"
			"   jz       INT8_SAVE_C_M8N32_2                             \n"
			"   mov      %%rcx, %%r10                                    \n" // C0
			"   leaq     (%%r10, %%r8, 4), %%r11                         \n" // C1
			"   leaq     (%%r11, %%r8, 4), %%r12                         \n" // C2
			"   leaq     (%%r12, %%r8, 4), %%r13                         \n" // C3

			"INT8_SAVE_C_M8N32:                                          \n"
			"  int8_save_c_m8n32                                         \n"
			"  imul     $8, %%r15, %%r11                                 \n" // temp use %%r11
			"  add      %%r11, %%r9                                      \n"
			"  movq     %%r9, %%rax                                      \n"
			"  jmp      INT8_BEGIN_M8N32                                 \n"

			"INT8_SAVE_C_M8N32_2:                                        \n"
			"  int8_save_c_m8n32_2                                       \n"
			"  imul     $8, %%r15, %%r11                                 \n" // temp use %%r11
			"  add      %%r11, %%r9                                      \n"
			"  movq     %%r9, %%rax                                      \n"
			"  jmp      INT8_BEGIN_M8N32                                 \n"

			//-----------------------------------------------------------------

			"INT8_BEGIN_M4N32:                                           \n"
			"   cmpq    $4, %%rdi                                        \n"
			"   jb      INT8_END_N32                                     \n"
			"   mov     %%r14, %%rbx                                     \n" // Bc
			"   mov     %%rsi, %%rdx                                     \n" // K
			"   vmovups        (%%rbx), %%zmm4                           \n" // B0-15
			"   vmovups     64(%%rbx), %%zmm5                            \n" // B16-31
			"   vpxorq         %%zmm8, %%zmm8, %%zmm8                    \n"
			"   vpxorq         %%zmm9, %%zmm9, %%zmm9                    \n"
			"   vpxorq         %%zmm10, %%zmm10, %%zmm10                 \n"
			"   vpxorq         %%zmm11, %%zmm11, %%zmm11                 \n"
			"   vbroadcastss    (%%rax), %%zmm0                          \n" // A0
			"   vbroadcastss    4(%%rax), %%zmm1                         \n" // A1
			"   vpxorq         %%zmm12, %%zmm12, %%zmm12                 \n"
			"   vpxorq         %%zmm13, %%zmm13, %%zmm13                 \n"
			"   vpxorq         %%zmm14, %%zmm14, %%zmm14                 \n"
			"   vpxorq         %%zmm15, %%zmm15, %%zmm15                 \n"
			"   cmp     $32, %%rdx                                       \n"
			"   jb      INT8_MAIN_M4N32K4                                \n"
			"   subq    $32, %%rdx                                       \n"

			"INT8_MAIN_M4N32K32:                                         \n"
			"   int8_kernel_m4n32k4_k1                                   \n"
			"   int8_kernel_m4n32k4_k2                                   \n"
			"   int8_kernel_m4n32k4_k1                                   \n"
			"   int8_kernel_m4n32k4_k2                                   \n"
			"   int8_kernel_m4n32k4_k1                                   \n"
			"   int8_kernel_m4n32k4_k2                                   \n"
			"   int8_kernel_m4n32k4_k1                                   \n"
			"   int8_kernel_m4n32k4_k2                                   \n"
			"   cmp     $0, %%rdx                                        \n"
			"   je      INT8_BEGIN_SAVE_M4N32                            \n"
			"   cmp     $32, %%rdx                                       \n"
			"   jb      INT8_MAIN_M4N32K4                                \n"
			"   subq  $32, %%rdx                                         \n"
			"   jmp   INT8_MAIN_M4N32K32                                 \n"

			"INT8_MAIN_M4N32K4:                                          \n"
			"   int8_kernel_m4n32k4_k1                                   \n"
			"   subq    $4, %%rdx                                        \n"
			"   cmp     $0, %%rdx                                        \n"
			"   je      INT8_BEGIN_SAVE_M4N32                            \n"
			"   int8_kernel_m4n32k4_k2                                   \n"
			"   subq    $4, %%rdx                                        \n"
			"   cmp     $0, %%rdx                                        \n"
			"   je      INT8_BEGIN_SAVE_M4N32                            \n"
			"   jmp     INT8_MAIN_M4N32K4                                \n"

			"INT8_BEGIN_SAVE_M4N32:                                      \n"
			"   mov      %[is_end_gemm], %%r13                           \n"
			"   test     $1, %%r13                                       \n"
			"   jz INT8_SAVE_C_M4N32_2                                   \n"
			"   mov      %%rcx, %%r10                                    \n" // C0
			"   leaq     (%%r10, %%r8, 4), %%r11                         \n" // C1
			"   leaq     (%%r11, %%r8, 4), %%r12                         \n" // C2
			"   leaq     (%%r12, %%r8, 4), %%r13                         \n" // C3

			"INT8_SAVE_C_M4N32:                                          \n"
			"   cmpq     $3, %%rdi                                       \n"
			"   je       INT8_SAVE_C_M3N32                               \n"
			"   cmpq     $2, %%rdi                                       \n"
			"   je       INT8_SAVE_C_M2N32                               \n"
			"   cmpq     $1, %%rdi                                       \n"
			"   je       INT8_SAVE_C_M1N32                               \n"
			"   int8_save_c_m4n32                                        \n"
			"   imul     $4, %%r15, %%r11                                \n" // temp use %%r11
			"   add      %%r11, %%r9                                     \n"
			"   movq     %%r9, %%rax                                     \n"
			"   jmp      INT8_BEGIN_M4N32                                \n"

			"INT8_SAVE_C_M3N32:                                          \n"
			"   vmovups         %%zmm12, (%%r12)                         \n"
			"   vmovups         %%zmm13, 64(%%r12)                       \n"

			"INT8_SAVE_C_M2N32:                                          \n"
			"   vmovups         %%zmm10, (%%r11)                         \n"
			"   vmovups         %%zmm11, 64(%%r11)                       \n"

			"INT8_SAVE_C_M1N32:                                          \n"
			"   vmovups         %%zmm8, (%%r10)                          \n"
			"   vmovups         %%zmm9, 64(%%r10)                        \n"
			"   jmp      INT8_END_N32                                    \n"

			"INT8_SAVE_C_M4N32_2:                                        \n"
			"   cmpq     $3, %%rdi                                       \n"
			"   je       INT8_SAVE_C_M3N32_2                             \n"
			"   cmpq     $2, %%rdi                                       \n"
			"   je       INT8_SAVE_C_M2N32_2                             \n"
			"   cmpq     $1, %%rdi                                       \n"
			"   je       INT8_SAVE_C_M1N32_2                             \n"
			"   int8_save_c_m4n32_2                                      \n"
			"   imul     $4, %%r15, %%r11                                \n" // temp use %%r11
			"   add      %%r11, %%r9                                     \n"
			"   movq     %%r9, %%rax                                     \n"
			"   jmp      INT8_BEGIN_M4N32                                \n"

			"INT8_SAVE_C_M3N32_2:                                        \n"
			"   vpmovsdb   %%zmm12, %%xmm20                              \n"
			"   vpmovsdb   %%zmm13, %%xmm21                              \n"
			"   vmovups    %%xmm20, 64(%%r10)                            \n"
			"   vmovups    %%xmm21, 80(%%r10)                            \n"

			"INT8_SAVE_C_M2N32_2:                                        \n"
			"   vpmovsdb   %%zmm10, %%xmm18                              \n"
			"   vpmovsdb   %%zmm11, %%xmm19                              \n"
			"   vmovups    %%xmm18, 32(%%r10)                            \n"
			"   vmovups    %%xmm19, 48(%%r10)                            \n"

			"INT8_SAVE_C_M1N32_2:                                        \n"
			"   vpmovsdb   %%zmm8, %%xmm16                               \n"
			"   vpmovsdb   %%zmm9, %%xmm17                               \n"
			"   vmovups    %%xmm16, (%%r10)                              \n"
			"   vmovups    %%xmm17, 16(%%r10)                            \n"

			"INT8_END_N32:                                               \n"

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

void FLASHGEMM_INT8_KERNELm12xn16_edge(int *C, uint8_t *Cc, int8_t *A, uint8_t *B, long M, long K, long LK, long LN, uint8_t *Bc, long nr, bool is_start_gemm, bool is_end_gemm)
{
	asm volatile(

			".macro int8_pack_b_n16                                      \n"
			"  vmovups (%%rbx), %%xmm9                                   \n"
			"  prefetcht2  64(%%rbx)                                     \n"
			"  leaq    (%%rbx, %%r8, 1), %%rbx                           \n"
			"  vmovups (%%rbx), %%xmm11                                  \n"
			"  prefetcht2  64(%%rbx)                                     \n"
			"  leaq    (%%rbx, %%r8, 1), %%rbx                           \n"

			"  vpunpcklbw  %%xmm11, %%xmm9, %%xmm17                      \n"
			"  vpunpckhbw  %%xmm11, %%xmm9, %%xmm19                      \n"

			"  vmovups (%%rbx), %%xmm13                                  \n"
			"  prefetcht2  64(%%rbx)                                     \n"
			"  leaq    (%%rbx, %%r8, 1), %%rbx                           \n"
			"  vmovups (%%rbx), %%xmm15                                  \n"
			"  prefetcht2  64(%%rbx)                                     \n"
			"  leaq    (%%rbx, %%r8, 1), %%rbx                           \n"

			"  vpunpcklbw  %%xmm15, %%xmm13, %%xmm21                     \n"
			"  vpunpckhbw  %%xmm15, %%xmm13, %%xmm23                     \n"

			"  vpunpcklwd  %%xmm21, %%xmm17, %%xmm25                     \n"
			"  vpunpckhwd  %%xmm21, %%xmm17, %%xmm27                     \n"
			"  vpunpcklwd  %%xmm23, %%xmm19, %%xmm29                     \n"
			"  vpunpckhwd  %%xmm23, %%xmm19, %%xmm31                     \n"

			"  vinserti32x4    $0x0, %%xmm25, %%zmm0, %%zmm0             \n"
			"  vinserti32x4    $0x1, %%xmm27, %%zmm0, %%zmm0             \n"
			"  vinserti32x4    $0x2, %%xmm29, %%zmm0, %%zmm0             \n"
			"  vbroadcastss    (%%rax), %%zmm2                           \n" // A0
			"  vinserti32x4    $0x3, %%xmm31, %%zmm0, %%zmm0             \n"
			"  vbroadcastss    4(%%rax), %%zmm3                          \n" // A1
			"  vmovups         %%zmm0, (%%rbp)                           \n"

			"  addq            $64, %%rbp                                \n"

			".endm                                                       \n"

			".macro    int8_kernel_m12n16k4_pack                         \n"
			"  int8_pack_b_n16                                           \n"

			"  vbroadcastss    8(%%rax), %%zmm5                          \n" // A2
			"  vpdpbusd    %%zmm0, %%zmm2, %%zmm8                        \n" // A0*(B0-15)

			"  vbroadcastss    12(%%rax), %%zmm6                         \n" // A3
			"  vpdpbusd    %%zmm0, %%zmm3, %%zmm10                       \n" // A1*(B0-15)

			"  prefetcht0         256(%%rax)                             \n"

			"  vbroadcastss    16(%%rax), %%zmm2                         \n" // A4
			"  vpdpbusd    %%zmm0, %%zmm5, %%zmm12                       \n"

			"  vbroadcastss    20(%%rax), %%zmm3                         \n" // A5
			"  vpdpbusd    %%zmm0, %%zmm6, %%zmm14                       \n"

			"  vbroadcastss    24(%%rax), %%zmm5                         \n" // A6
			"  vpdpbusd    %%zmm0, %%zmm2, %%zmm16                       \n"

			"  vbroadcastss    28(%%rax), %%zmm6                         \n" // A7
			"  vpdpbusd    %%zmm0, %%zmm3, %%zmm18                       \n"

			"  vbroadcastss    32(%%rax), %%zmm2                         \n" // A8
			"  vpdpbusd    %%zmm0, %%zmm5, %%zmm20                       \n"

			"  vbroadcastss    36(%%rax), %%zmm3                         \n" // A9
			"  vpdpbusd    %%zmm0, %%zmm6, %%zmm22                       \n"

			"  vbroadcastss    40(%%rax), %%zmm5                         \n" // A10
			"  vpdpbusd    %%zmm0, %%zmm2, %%zmm24                       \n"

			"  prefetcht0         384(%%rax)                             \n"

			"  vbroadcastss    44(%%rax), %%zmm6                         \n" // A11
			"  addq              $48, %%rax                              \n"
			"  vpdpbusd    %%zmm0, %%zmm3, %%zmm26                       \n"

			"  vbroadcastss    (%%rax), %%zmm2                           \n" // next A0
			"  vpdpbusd    %%zmm0, %%zmm5, %%zmm28                       \n"

			"  vbroadcastss    4(%%rax), %%zmm3                          \n" // next A1
			"  vpdpbusd    %%zmm0, %%zmm6, %%zmm30                       \n"
			".endm                                                       \n"

			".macro    int8_kernel_m12n16k4_k1                           \n"

			"   vbroadcastss    8(%%rax), %%zmm2                         \n"
			"   vpdpbusd    %%zmm4, %%zmm0, %%zmm8                       \n"

			"   vbroadcastss    12(%%rax), %%zmm3                        \n"

			"   vpdpbusd    %%zmm4, %%zmm1, %%zmm10                      \n"

			"   prefetcht0         256(%%rax)                            \n"

			"   vbroadcastss    16(%%rax), %%zmm0                        \n"
			"   vpdpbusd    %%zmm4, %%zmm2, %%zmm12                      \n"

			"   vbroadcastss    20(%%rax), %%zmm1                        \n"
			"   vpdpbusd    %%zmm4, %%zmm3, %%zmm14                      \n"

			"   vbroadcastss    24(%%rax), %%zmm2                        \n"
			"   vpdpbusd    %%zmm4, %%zmm0, %%zmm16                      \n"

			"   vbroadcastss    28(%%rax), %%zmm3                        \n"
			"   vpdpbusd    %%zmm4, %%zmm1, %%zmm18                      \n"

			"   vbroadcastss    32(%%rax), %%zmm0                        \n"
			"   vpdpbusd    %%zmm4, %%zmm2, %%zmm20                      \n"

			"    addq              $64, %%rbx                            \n"

			"   vbroadcastss    36(%%rax), %%zmm1                        \n"
			"   vpdpbusd    %%zmm4, %%zmm3, %%zmm22                      \n"

			"   prefetcht0         64(%%rbx)                             \n"

			"   vbroadcastss    40(%%rax), %%zmm2                        \n"
			"   vpdpbusd    %%zmm4, %%zmm0, %%zmm24                      \n"

			"   vbroadcastss    44(%%rax), %%zmm3                        \n"
			"   vpdpbusd    %%zmm4, %%zmm1, %%zmm26                      \n"
			"   vmovups         (%%rbx), %%zmm6                          \n"
			"    addq              $48, %%rax                            \n"

			"   vbroadcastss    (%%rax), %%zmm0                          \n"
			"   vpdpbusd    %%zmm4, %%zmm2, %%zmm28                      \n"

			"   vbroadcastss    4(%%rax), %%zmm1                         \n"
			"   vpdpbusd    %%zmm4, %%zmm3, %%zmm30                      \n"
			".endm                                                       \n"

			".macro    int8_kernel_m12n16k4_k2                           \n"
			"   vbroadcastss    8(%%rax), %%zmm2                         \n"
			"   vpdpbusd    %%zmm6, %%zmm0, %%zmm8                       \n"

			"   vbroadcastss    12(%%rax), %%zmm3                        \n"
			"   vpdpbusd    %%zmm6, %%zmm1, %%zmm10                      \n"

			"   prefetcht0         256(%%rax)                            \n"

			"   vbroadcastss    16(%%rax), %%zmm0                        \n"
			"   vpdpbusd    %%zmm6, %%zmm2, %%zmm12                      \n"

			"   vbroadcastss    20(%%rax), %%zmm1                        \n"
			"   vpdpbusd    %%zmm6, %%zmm3, %%zmm14                      \n"

			"   vbroadcastss    24(%%rax), %%zmm2                        \n"
			"   vpdpbusd    %%zmm6, %%zmm0, %%zmm16                      \n"

			"   vbroadcastss    28(%%rax), %%zmm3                        \n"
			"   vpdpbusd    %%zmm6, %%zmm1, %%zmm18                      \n"

			"   vbroadcastss    32(%%rax), %%zmm0                        \n"
			"   vpdpbusd    %%zmm6, %%zmm2, %%zmm20                      \n"

			"   addq              $64, %%rbx                             \n"

			"   vbroadcastss    36(%%rax), %%zmm1                        \n"
			"   vpdpbusd    %%zmm6, %%zmm3, %%zmm22                      \n"

			"   prefetcht0         64(%%rbx)                             \n"

			"   vbroadcastss    40(%%rax), %%zmm2                        \n"
			"   vpdpbusd    %%zmm6, %%zmm0, %%zmm24                      \n"

			"   vbroadcastss    44(%%rax), %%zmm3                        \n"
			"   vpdpbusd    %%zmm6, %%zmm1, %%zmm26                      \n"
			"   vmovups         (%%rbx), %%zmm4                          \n"
			"   addq           $48, %%rax                                \n"

			"   vbroadcastss    (%%rax), %%zmm0                          \n"
			"   vpdpbusd    %%zmm6, %%zmm2, %%zmm28                      \n"

			"   vbroadcastss    4(%%rax), %%zmm1                         \n"
			"   vpdpbusd    %%zmm6, %%zmm3, %%zmm30                      \n"
			".endm                                                       \n"

			".macro    int8_kernel_m12n16k4_end                          \n"

			"   vbroadcastss    8(%%rax), %%zmm2                         \n"
			"   vpdpbusd    %%zmm6, %%zmm0, %%zmm8                       \n"

			"   vbroadcastss    12(%%rax), %%zmm3                        \n"
			"   vpdpbusd    %%zmm6, %%zmm1, %%zmm10                      \n"

			"   prefetcht0         256(%%rax)                            \n"

			"   vbroadcastss    16(%%rax), %%zmm0                        \n"
			"   vpdpbusd    %%zmm6, %%zmm2, %%zmm12                      \n"

			"   vbroadcastss    20(%%rax), %%zmm1                        \n"
			"   vpdpbusd    %%zmm6, %%zmm3, %%zmm14                      \n"

			"   vbroadcastss    24(%%rax), %%zmm2                        \n"
			"   vpdpbusd    %%zmm6, %%zmm0, %%zmm16                      \n"

			"   vbroadcastss    28(%%rax), %%zmm3                        \n"
			"   vpdpbusd    %%zmm6, %%zmm1, %%zmm18                      \n"

			"   vbroadcastss    32(%%rax), %%zmm0                        \n"
			"   vpdpbusd    %%zmm6, %%zmm2, %%zmm20                      \n"

			"   vbroadcastss    36(%%rax), %%zmm1                        \n"
			"   vpdpbusd    %%zmm6, %%zmm3, %%zmm22                      \n"

			"   vbroadcastss    40(%%rax), %%zmm2                        \n"
			"   vpdpbusd    %%zmm6, %%zmm0, %%zmm24                      \n"

			"   vbroadcastss    44(%%rax), %%zmm3                        \n"
			"   vpdpbusd    %%zmm6, %%zmm1, %%zmm26                      \n"
			"   addq              $48, %%rax                             \n"

			"   vpdpbusd    %%zmm6, %%zmm2, %%zmm28                      \n"

			"   vpdpbusd    %%zmm6, %%zmm3, %%zmm30                      \n"
			".endm                                                       \n"

			".macro    int8_save_c_m12n16                                \n"
			"  vmovups         %%zmm8, (%%r10)                           \n"
			"  vmovups         %%zmm10, (%%r11)                          \n"
			"  vmovups         %%zmm12, (%%r12)                          \n"
			"  vmovups         %%zmm14, (%%r13)                          \n"

			"  leaq     (%%r13, %%r8, 4), %%r10                          \n" // C0
			"  leaq     (%%r10, %%r8, 4), %%r11                          \n" // C1
			"  leaq     (%%r11, %%r8, 4), %%r12                          \n" // C2
			"  leaq     (%%r12, %%r8, 4), %%r13                          \n" // C3

			"  vmovups         %%zmm16, (%%r10)%{%%k1%}                  \n"
			"  vmovups         %%zmm18, (%%r11)%{%%k1%}                  \n"
			"  vmovups         %%zmm20, (%%r12)%{%%k1%}                  \n"
			"  vmovups         %%zmm22, (%%r13)%{%%k1%}                  \n"

			"  leaq     (%%r13, %%r8, 4), %%r10                          \n" // C0
			"  leaq     (%%r10, %%r8, 4), %%r11                          \n" // C1
			"  leaq     (%%r11, %%r8, 4), %%r12                          \n" // C2
			"  leaq     (%%r12, %%r8, 4), %%r13                          \n" // C3

			"  vmovups         %%zmm24, (%%r10)%{%%k1%}                  \n"
			"  vmovups         %%zmm26, (%%r11)%{%%k1%}                  \n"
			"  subq             $12, %%rdi                               \n"
			"  vmovups         %%zmm28, (%%r12)%{%%k1%}                  \n"
			"  vmovups         %%zmm30, (%%r13)%{%%k1%}                  \n"

			"  leaq      (%%r13, %%r8, 4), %%rcx                         \n" // C0
			".endm                                                       \n"

			".macro    int8_save_c_m12n16_2                              \n"
			"  vpmovsdb   %%zmm8, %%xmm0                                 \n" // int32 convert to int8
			"  vpmovsdb   %%zmm10, %%xmm1                                \n"
			"  vpmovsdb   %%zmm12, %%xmm2                                \n"
			"  vpmovsdb   %%zmm14, %%xmm3                                \n"
			"  vpunpcklbw %%xmm0, %%xmm1, %%xmm4                         \n" // interleave 4*int8
			"  vpunpckhbw %%xmm0, %%xmm1, %%xmm5                         \n"
			"  vpunpcklbw %%xmm2, %%xmm3, %%xmm6                         \n"
			"  vpunpckhbw %%xmm2, %%xmm3, %%xmm7                         \n"
			"  vpunpcklwd %%xmm4, %%xmm6, %%xmm13                        \n"
			"  vpunpckhwd %%xmm4, %%xmm6, %%xmm15                        \n"
			"  vpunpcklwd %%xmm5, %%xmm7, %%xmm17                        \n"
			"  vpunpckhwd %%xmm5, %%xmm7, %%xmm19                        \n"
			"  vinserti32x4    $0x1, %%xmm15, %%zmm13, %%zmm13           \n"
			"  vinserti32x4    $0x2, %%xmm17, %%zmm13, %%zmm13           \n"
			"  vinserti32x4    $0x3, %%xmm19, %%zmm13, %%zmm13           \n"
			"  vmovups         %%zmm13, (%%r10)                          \n"

			"  vpmovsdb   %%zmm16, %%xmm21                               \n" // int32 convert to int8
			"  vpmovsdb   %%zmm18, %%xmm23                               \n"
			"  vpmovsdb   %%zmm20, %%xmm25                               \n"
			"  vpmovsdb   %%zmm22, %%xmm27                               \n"
			"  vpunpcklbw %%xmm21, %%xmm23, %%xmm29                      \n" // interleave 4*int8
			"  vpunpckhbw %%xmm21, %%xmm23, %%xmm31                      \n"
			"  vpunpcklbw %%xmm25, %%xmm27, %%xmm0                       \n"
			"  vpunpckhbw %%xmm25, %%xmm27, %%xmm1                       \n"
			"  vpunpcklwd %%xmm29, %%xmm0, %%xmm2                        \n"
			"  vpunpckhwd %%xmm29, %%xmm0, %%xmm3                        \n"
			"  vpunpcklwd %%xmm31, %%xmm1, %%xmm4                        \n"
			"  vpunpckhwd %%xmm31, %%xmm1, %%xmm5                        \n"
			"  vinserti32x4    $0x1, %%xmm3, %%zmm2, %%zmm2              \n"
			"  vinserti32x4    $0x2, %%xmm4, %%zmm2, %%zmm2              \n"
			"  vinserti32x4    $0x3, %%xmm5, %%zmm2, %%zmm2              \n"
			"  vmovups         %%zmm2, 64(%%r10)                         \n"

			"  vpmovsdb   %%zmm24, %%xmm6                                \n" // int32 convert to int8
			"  vpmovsdb   %%zmm26, %%xmm7                                \n"
			"  vpmovsdb   %%zmm28, %%xmm8                                \n"
			"  vpmovsdb   %%zmm30, %%xmm9                                \n"
			"  vpunpcklbw %%xmm6, %%xmm7, %%xmm11                        \n" // interleave 4*int8
			"  vpunpckhbw %%xmm6, %%xmm7, %%xmm12                        \n"
			"  vpunpcklbw %%xmm8, %%xmm9, %%xmm13                        \n"
			"  vpunpckhbw %%xmm8, %%xmm9, %%xmm14                        \n"
			"  vpunpcklwd %%xmm11, %%xmm13, %%xmm15                      \n"
			"  vpunpckhwd %%xmm11, %%xmm13, %%xmm16                      \n"
			"  vpunpcklwd %%xmm12, %%xmm14, %%xmm17                      \n"
			"  vpunpckhwd %%xmm12, %%xmm14, %%xmm18                      \n"
			"  vinserti32x4    $0x1, %%xmm16, %%zmm15, %%zmm15           \n"
			"  vinserti32x4    $0x2, %%xmm17, %%zmm15, %%zmm15           \n"
			"  vinserti32x4    $0x3, %%xmm18, %%zmm15, %%zmm15           \n"
			"  vmovups         %%zmm15, 128(%%r10)                       \n"

			"  addq           $192, %%r10                                \n"
			"  subq           $12, %%rdi                                 \n"
			".endm                                                       \n"

			//-----------------------------------------------------------------

			".macro    int8_kernel_m8n16k4_k1                            \n"
			"  vbroadcastss    8(%%rax), %%zmm2                          \n"
			"  vpdpbusd    %%zmm4, %%zmm0, %%zmm8                        \n"

			"  vbroadcastss    12(%%rax), %%zmm3                         \n"
			"  vpdpbusd    %%zmm4, %%zmm1, %%zmm10                       \n"

			"  prefetcht0         256(%%rax)                             \n"

			"  vbroadcastss    16(%%rax), %%zmm0                         \n"
			"  vpdpbusd    %%zmm4, %%zmm2, %%zmm12                       \n"

			"  vmovups         (%%rbx), %%zmm6                           \n"

			"  vbroadcastss    20(%%rax), %%zmm1                         \n"
			"  vpdpbusd    %%zmm4, %%zmm3, %%zmm14                       \n"

			"  vbroadcastss    24(%%rax), %%zmm2                         \n"
			"  vpdpbusd    %%zmm4, %%zmm0, %%zmm16                       \n"

			"  vbroadcastss    28(%%rax), %%zmm3                         \n"
			"  vpdpbusd    %%zmm4, %%zmm1, %%zmm18                       \n"

			"  addq              $32, %%rax                              \n"

			"  vbroadcastss    (%%rax), %%zmm0                           \n"
			"  vpdpbusd    %%zmm4, %%zmm2, %%zmm20                       \n"

			"  addq              $64, %%rbx                              \n"

			"  vbroadcastss    4(%%rax), %%zmm1                          \n"
			"  vpdpbusd    %%zmm4, %%zmm3, %%zmm22                       \n"
			".endm                                                       \n"

			".macro    int8_kernel_m8n16k4_k2                            \n"
			"  vbroadcastss    8(%%rax), %%zmm2                          \n"
			"  vpdpbusd    %%zmm6, %%zmm0, %%zmm8                        \n"

			"  vbroadcastss    12(%%rax), %%zmm3                         \n"
			"  vpdpbusd    %%zmm6, %%zmm1, %%zmm10                       \n"

			"  prefetcht0         256(%%rax)                             \n"

			"  vbroadcastss    16(%%rax), %%zmm0                         \n"
			"  vpdpbusd    %%zmm6, %%zmm2, %%zmm12                       \n"

			"  vmovups         (%%rbx), %%zmm4                           \n"

			"  vbroadcastss    20(%%rax), %%zmm1                         \n"
			"  vpdpbusd    %%zmm6, %%zmm3, %%zmm14                       \n"

			"  vbroadcastss    24(%%rax), %%zmm2                         \n"
			"  vpdpbusd    %%zmm6, %%zmm0, %%zmm16                       \n"

			"  vmovups         64(%%rbx), %%zmm5                         \n"

			"  vbroadcastss    28(%%rax), %%zmm3                         \n"
			"  vpdpbusd    %%zmm6, %%zmm1, %%zmm18                       \n"

			"  addq              $32, %%rax                              \n"

			"  vbroadcastss    (%%rax), %%zmm0                           \n"
			"  vpdpbusd    %%zmm6, %%zmm2, %%zmm20                       \n"

			"  addq              $64, %%rbx                              \n"

			"  vbroadcastss    4(%%rax), %%zmm1                          \n"
			"  vpdpbusd    %%zmm6, %%zmm3, %%zmm22                       \n"
			".endm                                                       \n"

			".macro    int8_kernel_m8n16k4_end                           \n"
			"  vbroadcastss    8(%%rax), %%zmm2                          \n"
			"  vpdpbusd    %%zmm6, %%zmm0, %%zmm8                        \n"

			"  vbroadcastss    12(%%rax), %%zmm3                         \n"
			"  vpdpbusd    %%zmm6, %%zmm1, %%zmm10                       \n"

			"  prefetcht0         256(%%rax)                             \n"

			"  vbroadcastss    16(%%rax), %%zmm0                         \n"
			"  vpdpbusd    %%zmm6, %%zmm2, %%zmm12                       \n"

			"  vbroadcastss    20(%%rax), %%zmm1                         \n"
			"  vpdpbusd    %%zmm6, %%zmm3, %%zmm14                       \n"

			"  vbroadcastss    24(%%rax), %%zmm2                         \n"
			"  vpdpbusd    %%zmm6, %%zmm0, %%zmm16                       \n"

			"  vbroadcastss    28(%%rax), %%zmm3                         \n"
			"  vpdpbusd    %%zmm6, %%zmm1, %%zmm18                       \n"

			"  addq              $32, %%rax                              \n"

			"  vpdpbusd    %%zmm6, %%zmm2, %%zmm20                       \n"
			"  vpdpbusd    %%zmm6, %%zmm3, %%zmm22                       \n"
			".endm                                                       \n"

			".macro    int8_save_c_m8n16                                 \n"
			"  vmovups         %%zmm8, (%%r10)                           \n"
			"  vmovups         %%zmm10, (%%r11)                          \n"
			"  vmovups         %%zmm12, (%%r12)                          \n"
			"  vmovups         %%zmm14, (%%r13)                          \n"

			"  leaq     (%%r13, %%r8, 4), %%r10                          \n" // C0
			"  leaq     (%%r10, %%r8, 4), %%r11                          \n" // C1
			"  leaq     (%%r11, %%r8, 4), %%r12                          \n" // C2
			"  leaq     (%%r12, %%r8, 4), %%r13                          \n" // C3

			"  vmovups         %%zmm16, (%%r10)%{%%k1%}                  \n"
			"  vmovups         %%zmm18, (%%r11)%{%%k1%}                  \n"
			"  vmovups         %%zmm20, (%%r12)%{%%k1%}                  \n"
			"  vmovups         %%zmm22, (%%r13)%{%%k1%}                  \n"

			"  subq             $8, %%rdi                                \n"
			"  leaq      (%%r13, %%r8, 4), %%rcx                         \n" // C0
			".endm                                                       \n"

			".macro    int8_save_c_m8n16_2                               \n"
			"  vpmovsdb   %%zmm8, %%xmm0                                 \n" // int32 convert to int8
			"  vpmovsdb   %%zmm10, %%xmm1                                \n"
			"  vpmovsdb   %%zmm12, %%xmm2                                \n"
			"  vpmovsdb   %%zmm14, %%xmm3                                \n"
			"  vpunpcklbw %%xmm0, %%xmm1, %%xmm4                         \n" // interleave 4*int8
			"  vpunpckhbw %%xmm0, %%xmm1, %%xmm5                         \n"
			"  vpunpcklbw %%xmm2, %%xmm3, %%xmm6                         \n"
			"  vpunpckhbw %%xmm2, %%xmm3, %%xmm7                         \n"
			"  vpunpcklwd %%xmm4, %%xmm6, %%xmm13                        \n"
			"  vpunpckhwd %%xmm4, %%xmm6, %%xmm15                        \n"
			"  vpunpcklwd %%xmm5, %%xmm7, %%xmm17                        \n"
			"  vpunpckhwd %%xmm5, %%xmm7, %%xmm19                        \n"
			"  vinserti32x4    $0x1, %%xmm15, %%zmm13, %%zmm13           \n"
			"  vinserti32x4    $0x2, %%xmm17, %%zmm13, %%zmm13           \n"
			"  vinserti32x4    $0x3, %%xmm19, %%zmm13, %%zmm13           \n"
			"  vmovups         %%zmm13, (%%r10)                          \n"

			"  vpmovsdb   %%zmm16, %%xmm21                               \n" // int32 convert to int8
			"  vpmovsdb   %%zmm18, %%xmm23                               \n"
			"  vpmovsdb   %%zmm20, %%xmm25                               \n"
			"  vpmovsdb   %%zmm22, %%xmm27                               \n"
			"  vpunpcklbw %%xmm21, %%xmm23, %%xmm29                      \n" // interleave 4*int8
			"  vpunpckhbw %%xmm21, %%xmm23, %%xmm31                      \n"
			"  vpunpcklbw %%xmm25, %%xmm27, %%xmm0                       \n"
			"  vpunpckhbw %%xmm25, %%xmm27, %%xmm1                       \n"
			"  vpunpcklwd %%xmm29, %%xmm0, %%xmm2                        \n"
			"  vpunpckhwd %%xmm29, %%xmm0, %%xmm3                        \n"
			"  vpunpcklwd %%xmm31, %%xmm1, %%xmm4                        \n"
			"  vpunpckhwd %%xmm31, %%xmm1, %%xmm5                        \n"
			"  vinserti32x4    $0x1, %%xmm3, %%zmm2, %%zmm2              \n"
			"  vinserti32x4    $0x2, %%xmm4, %%zmm2, %%zmm2              \n"
			"  vinserti32x4    $0x3, %%xmm5, %%zmm2, %%zmm2              \n"
			"  vmovups         %%zmm2, 64(%%r10)                         \n"

			"  addq           $128, %%r10                                \n"
			"  subq           $8, %%rdi                                  \n"
			".endm                                                       \n"

			//-----------------------------------------------------------------

			".macro    int8_kernel_m4n16k4_k1                            \n"
			"  vbroadcastss    8(%%rax), %%zmm2                          \n"
			"  vpdpbusd    %%zmm4, %%zmm0, %%zmm8                        \n"

			"  vmovups         (%%rbx), %%zmm6                           \n"
			"  vbroadcastss    12(%%rax), %%zmm3                         \n"
			"  vpdpbusd    %%zmm4, %%zmm1, %%zmm10                       \n"

			"  vmovups         64(%%rbx), %%zmm7                         \n"
			"  prefetcht0      256(%%rax)                                \n"

			"  addq            $16, %%rax                                \n"
			"  vbroadcastss    (%%rax), %%zmm0                           \n"
			"  vpdpbusd    %%zmm4, %%zmm2, %%zmm12                       \n"

			"  vbroadcastss    4(%%rax), %%zmm1                          \n"
			"  vpdpbusd    %%zmm4, %%zmm3, %%zmm14                       \n"

			"  addq              $64, %%rbx                              \n"
			".endm                                                       \n"

			".macro    int8_kernel_m4n16k4_k2                            \n"
			"  vbroadcastss    8(%%rax), %%zmm2                          \n"
			"  vpdpbusd    %%zmm6, %%zmm0, %%zmm8                        \n"

			"  vmovups         (%%rbx), %%zmm4                           \n"

			"  vbroadcastss    12(%%rax), %%zmm3                         \n"
			"  vpdpbusd    %%zmm6, %%zmm1, %%zmm10                       \n"

			"  vmovups         64(%%rbx), %%zmm5                         \n"
			"  prefetcht0      256(%%rax)                                \n"

			"  addq            $16, %%rax                                \n"
			"  vbroadcastss    (%%rax), %%zmm0                           \n"
			"  vpdpbusd    %%zmm6, %%zmm2, %%zmm12                       \n"

			"  vbroadcastss    4(%%rax), %%zmm1                          \n"
			"  vpdpbusd    %%zmm6, %%zmm3, %%zmm14                       \n"

			"  addq            $64, %%rbx                                \n"
			".endm                                                       \n"

			".macro    int8_kernel_m4n16k4_end                           \n"
			"  broadcastss     8(%%rax), %%zmm2                          \n"
			"  vpdpbusd    %%zmm6, %%zmm0, %%zmm8                        \n"

			"  vbroadcastss    12(%%rax), %%zmm3                         \n"
			"  vpdpbusd    %%zmm6, %%zmm1, %%zmm10                       \n"

			"  prefetcht0      256(%%rax)                                \n"

			"  vpdpbusd    %%zmm6, %%zmm2, %%zmm12                       \n"
			"  vpdpbusd    %%zmm6, %%zmm3, %%zmm14                       \n"
			".endm                                                       \n"

			".macro    int8_save_c_m4n16                                 \n"
			"  vmovups         %%zmm8, (%%r10)%{%%k1%}                   \n"
			"  vmovups         %%zmm10, (%%r11)%{%%k1%}                  \n"
			"  vmovups         %%zmm12, (%%r12)%{%%k1%}                  \n"
			"  vmovups         %%zmm14, (%%r13)%{%%k1%}                  \n"

			"  subq             $4, %%rdi                                \n"
			"  leaq      (%%r13, %%r8, 4), %%rcx                         \n" // C0
			".endm                                                       \n"

			".macro    int8_save_c_m4n16_2                               \n"
			"  vpmovsdb   %%zmm8, %%xmm0                                 \n" // int32 convert to int8
			"  vpmovsdb   %%zmm10, %%xmm1                                \n"
			"  vpmovsdb   %%zmm12, %%xmm2                                \n"
			"  vpmovsdb   %%zmm14, %%xmm3                                \n"
			"  vpunpcklbw %%xmm0, %%xmm1, %%xmm4                         \n" // interleave 4*int8
			"  vpunpckhbw %%xmm0, %%xmm1, %%xmm5                         \n"
			"  vpunpcklbw %%xmm2, %%xmm3, %%xmm6                         \n"
			"  vpunpckhbw %%xmm2, %%xmm3, %%xmm7                         \n"
			"  vpunpcklwd %%xmm4, %%xmm6, %%xmm13                        \n"
			"  vpunpckhwd %%xmm4, %%xmm6, %%xmm15                        \n"
			"  vpunpcklwd %%xmm5, %%xmm7, %%xmm17                        \n"
			"  vpunpckhwd %%xmm5, %%xmm7, %%xmm19                        \n"
			"  vinserti32x4    $0x1, %%xmm15, %%zmm13, %%zmm13           \n"
			"  vinserti32x4    $0x2, %%xmm17, %%zmm13, %%zmm13           \n"
			"  vinserti32x4    $0x3, %%xmm19, %%zmm13, %%zmm13           \n"
			"  vmovups         %%zmm13, (%%r10)                          \n"

			"  addq           $64, %%r10                                 \n"
			"  subq           $4, %%rdi                                  \n"
			".endm                                                       \n"

			//-----------------------------------------------------------------

			"GEMM_INT8_N16:                                              \n"
			"  mov    %[nr], %%rdx                                       \n"
			"  mov    %%dl, %%cl                                         \n"
			"  mov    $1, %%eax                                          \n"
			"  shl    %%cl, %%eax                                        \n"
			"  dec    %%eax                                              \n"
			"  kmovw  %%eax, %%k1                                        \n"

			"  mov     %[C], %%rcx                                       \n"
			"  mov     %[Cc], %%r10                                      \n"
			"  mov     %[A], %%rax                                       \n"
			"  mov     %[B], %%rbx                                       \n"

			"  prefetcht0         (%%rax)                                \n"

			"  mov     %[K], %%rdx                                       \n"
			"  mov     %[LN], %%r8                                       \n"
			"  mov     %[Bc], %%r14                                      \n"
			"  mov     %[M], %%rdi                                       \n"

			"  mov     %[LK], %%r15                                      \n"
			"  mov     %%rax, %%r9                                       \n"
			"  prefetcht0         (%%rbx)                                \n"
			"  mov     %%rdx, %%rsi                                      \n"

			"  mov    %[is_start_gemm], %%r12                            \n"
			"  test   $1, %%r12                                          \n"
			"  jz     INT8_BEGIN_M12N16                                  \n"

			"INT8_BEGIN_PACK_N16:                                        \n"
			"   mov     %%r14, %%rbp                                     \n" // Bc
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
			"   cmp     $32, %%rdx                                       \n"
			"   jb      INT8_PACK_MAIN_M12N16K4                          \n"
			"   subq    $32, %%rdx                                       \n"

			"INT8_PACK_MAIN_M12N16K32:                                   \n"
			"   int8_kernel_m12n16k4_pack                                \n"
			"   int8_kernel_m12n16k4_pack                                \n"
			"   int8_kernel_m12n16k4_pack                                \n"
			"   int8_kernel_m12n16k4_pack                                \n"
			"   int8_kernel_m12n16k4_pack                                \n"
			"   int8_kernel_m12n16k4_pack                                \n"
			"   int8_kernel_m12n16k4_pack                                \n"
			"   int8_kernel_m12n16k4_pack                                \n"
			"   cmp     $0, %%rdx                                        \n"
			"   je      INT8_BEGIN_SAVE_M12N16                           \n"
			"   cmp     $32, %%rdx                                       \n"
			"   jb      INT8_PACK_MAIN_M12N16K4                          \n"
			"   subq    $32, %%rdx                                       \n"
			"   jmp     INT8_PACK_MAIN_M12N16K32                         \n"

			"INT8_PACK_MAIN_M12N16K4:                                    \n"
			"   subq    $4, %%rdx                                        \n"
			"   cmp     $0, %%rdx                                        \n"
			"   je      INT8_BEGIN_SAVE_M12N16                           \n"
			"   int8_kernel_m12n16k4_pack                                \n"
			"   jmp     INT8_PACK_MAIN_M12N16K4                          \n"

			//-----------------------------------------------------------------

			"INT8_BEGIN_M12N16:                                          \n"
			"   cmpq    $12, %%rdi                                       \n"
			"   jb      INT8_BEGIN_M8N16                                 \n"

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
			"   cmp     $32, %%rdx                                       \n"
			"   jb      INT8_MAIN_M12N16K4                               \n"
			"   subq    $32, %%rdx                                       \n"

			"INT8_MAIN_M12N16K32:                                        \n"
			"   int8_kernel_m12n16k4_k1                                  \n"
			"   int8_kernel_m12n16k4_k2                                  \n"
			"   int8_kernel_m12n16k4_k1                                  \n"
			"   int8_kernel_m12n16k4_k2                                  \n"
			"   int8_kernel_m12n16k4_k1                                  \n"
			"   int8_kernel_m12n16k4_k2                                  \n"
			"   int8_kernel_m12n16k4_k1                                  \n"
			"   int8_kernel_m12n16k4_k2                                  \n"
			"   cmp     $0, %%rdx                                        \n"
			"   je      INT8_BEGIN_SAVE_M12N16                           \n"
			"   cmp     $32, %%rdx                                       \n"
			"   jb      INT8_MAIN_M12N16K4                               \n"
			"   subq  $32, %%rdx                                         \n"
			"   jmp   INT8_MAIN_M12N16K32                                \n"

			"INT8_MAIN_M12N16K4:                                         \n"
			"   int8_kernel_m12n16k4_k1                                  \n"
			"   subq    $4, %%rdx                                        \n"
			"   cmp     $0, %%rdx                                        \n"
			"   je      INT8_BEGIN_SAVE_M12N16                           \n"
			"   int8_kernel_m12n16k4_k2                                  \n"
			"   subq    $4, %%rdx                                        \n"
			"   cmp     $0, %%rdx                                        \n"
			"   je      INT8_BEGIN_SAVE_M12N16                           \n"
			"   jmp     INT8_MAIN_M12N16K4                               \n"

			"INT8_BEGIN_SAVE_M12N16:                                     \n"
			"   mov      %[is_end_gemm], %%r13                           \n"
			"   test     $1, %%r13                                       \n"
			"   jz       INT8_SAVE_C_M12N16_2                            \n"
			"   mov      %%rcx, %%r10                                    \n" // C0
			"   leaq     (%%r10, %%r8, 4), %%r11                         \n" // C1
			"   leaq     (%%r11, %%r8, 4), %%r12                         \n" // C2
			"   leaq     (%%r12, %%r8, 4), %%r13                         \n" // C3

			"INT8_SAVE_C_M12N16:                                         \n"
			"  int8_save_c_m12n16                                        \n"
			"  imul     $12, %%r15, %%r11                                \n" // temp use %%r11
			"  add      %%r11, %%r9                                      \n"
			"  movq     %%r9, %%rax                                      \n"
			"  jmp      INT8_BEGIN_M12N16                                \n"

			"INT8_SAVE_C_M12N16_2:                                       \n"
			"  int8_save_c_m12n16_2                                      \n"
			"  imul     $12, %%r15, %%r11                                \n" // temp use %%r11
			"  add      %%r11, %%r9                                      \n"
			"  movq     %%r9, %%rax                                      \n"
			"  jmp      INT8_BEGIN_M12N16                                \n"

			//-----------------------------------------------------------------

			"INT8_BEGIN_M8N16:                                           \n"
			"   cmpq    $8, %%rdi                                        \n"
			"   jb      INT8_BEGIN_M4N16                                 \n"

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
			"   cmp     $32, %%rdx                                       \n"
			"   jb      INT8_MAIN_M8N16K4                                \n"
			"   subq    $32, %%rdx                                       \n"

			"INT8_MAIN_M8N16K32:                                         \n"
			"   int8_kernel_m8n16k4_k1                                   \n"
			"   int8_kernel_m8n16k4_k2                                   \n"
			"   int8_kernel_m8n16k4_k1                                   \n"
			"   int8_kernel_m8n16k4_k2                                   \n"
			"   int8_kernel_m8n16k4_k1                                   \n"
			"   int8_kernel_m8n16k4_k2                                   \n"
			"   int8_kernel_m8n16k4_k1                                   \n"
			"   int8_kernel_m8n16k4_k2                                   \n"
			"   cmp     $0, %%rdx                                        \n"
			"   je      INT8_BEGIN_SAVE_M8N16                            \n"
			"   cmp     $32, %%rdx                                       \n"
			"   jb      INT8_MAIN_M8N16K4                                \n"
			"   subq  $32, %%rdx                                         \n"
			"   jmp   INT8_MAIN_M8N16K32                                 \n"

			"INT8_MAIN_M8N16K4:                                          \n"
			"   int8_kernel_m8n16k4_k1                                   \n"
			"   subq    $4, %%rdx                                        \n"
			"   cmp     $0, %%rdx                                        \n"
			"   je      INT8_BEGIN_SAVE_M8N16                            \n"
			"   int8_kernel_m8n16k4_k2                                   \n"
			"   subq    $4, %%rdx                                        \n"
			"   cmp     $0, %%rdx                                        \n"
			"   je      INT8_BEGIN_SAVE_M8N16                            \n"
			"   jmp     INT8_MAIN_M8N16K4                                \n"

			"INT8_BEGIN_SAVE_M8N16:                                      \n"
			"   mov      %[is_end_gemm], %%r13                           \n"
			"   test     $1, %%r13                                       \n"
			"   jz       INT8_SAVE_C_M8N16_2                             \n"
			"   mov      %%rcx, %%r10                                    \n" // C0
			"   leaq     (%%r10, %%r8, 4), %%r11                         \n" // C1
			"   leaq     (%%r11, %%r8, 4), %%r12                         \n" // C2
			"   leaq     (%%r12, %%r8, 4), %%r13                         \n" // C3

			"INT8_SAVE_C_M8N16:                                          \n"
			"  int8_save_c_m8n16                                         \n"
			"  imul     $8, %%r15, %%r11                                 \n" // temp use %%r11
			"  add      %%r11, %%r9                                      \n"
			"  movq     %%r9, %%rax                                      \n"
			"  jmp      INT8_BEGIN_M8N16                                 \n"

			"INT8_SAVE_C_M8N16_2:                                        \n"
			"  int8_save_c_m8n16_2                                       \n"
			"  imul     $8, %%r15, %%r11                                 \n" // temp use %%r11
			"  add      %%r11, %%r9                                      \n"
			"  movq     %%r9, %%rax                                      \n"
			"  jmp      INT8_BEGIN_M8N16                                 \n"

			//-----------------------------------------------------------------

			"INT8_BEGIN_M4N16:                                           \n"
			"   cmpq    $4, %%rdi                                        \n"
			"   jb      INT8_END_N16                                     \n"
			"   mov     %%r14, %%rbx                                     \n" // Bc
			"   mov     %%rsi, %%rdx                                     \n" // K
			"   vmovups        (%%rbx), %%zmm4                           \n" // B0-15
			"   vmovups     64(%%rbx), %%zmm5                            \n" // B16-31
			"   vpxorq         %%zmm8, %%zmm8, %%zmm8                    \n"
			"   vpxorq         %%zmm9, %%zmm9, %%zmm9                    \n"
			"   vpxorq         %%zmm10, %%zmm10, %%zmm10                 \n"
			"   vpxorq         %%zmm11, %%zmm11, %%zmm11                 \n"
			"   vbroadcastss    (%%rax), %%zmm0                          \n" // A0
			"   vbroadcastss    4(%%rax), %%zmm1                         \n" // A1
			"   vpxorq         %%zmm12, %%zmm12, %%zmm12                 \n"
			"   vpxorq         %%zmm13, %%zmm13, %%zmm13                 \n"
			"   vpxorq         %%zmm14, %%zmm14, %%zmm14                 \n"
			"   vpxorq         %%zmm15, %%zmm15, %%zmm15                 \n"
			"   cmp     $32, %%rdx                                       \n"
			"   jb      INT8_MAIN_M4N16K4                                \n"
			"   subq    $32, %%rdx                                       \n"

			"INT8_MAIN_M4N16K32:                                         \n"
			"   int8_kernel_m4n16k4_k1                                   \n"
			"   int8_kernel_m4n16k4_k2                                   \n"
			"   int8_kernel_m4n16k4_k1                                   \n"
			"   int8_kernel_m4n16k4_k2                                   \n"
			"   int8_kernel_m4n16k4_k1                                   \n"
			"   int8_kernel_m4n16k4_k2                                   \n"
			"   int8_kernel_m4n16k4_k1                                   \n"
			"   int8_kernel_m4n16k4_k2                                   \n"
			"   cmp     $0, %%rdx                                        \n"
			"   je      INT8_BEGIN_SAVE_M4N16                            \n"
			"   cmp     $32, %%rdx                                       \n"
			"   jb      INT8_MAIN_M4N16K4                                \n"
			"   subq  $32, %%rdx                                         \n"
			"   jmp   INT8_MAIN_M4N16K32                                 \n"

			"INT8_MAIN_M4N16K4:                                          \n"
			"   int8_kernel_m4n16k4_k1                                   \n"
			"   subq    $4, %%rdx                                        \n"
			"   cmp     $0, %%rdx                                        \n"
			"   je      INT8_BEGIN_SAVE_M4N16                            \n"
			"   int8_kernel_m4n16k4_k2                                   \n"
			"   subq    $4, %%rdx                                        \n"
			"   cmp     $0, %%rdx                                        \n"
			"   je      INT8_BEGIN_SAVE_M4N16                            \n"
			"   jmp     INT8_MAIN_M4N16K4                                \n"

			"INT8_BEGIN_SAVE_M4N16:                                      \n"
			"   mov      %[is_end_gemm], %%r13                           \n"
			"   test     $1, %%r13                                       \n"
			"   jz INT8_SAVE_C_M4N16_2                                   \n"
			"   mov      %%rcx, %%r10                                    \n" // C0
			"   leaq     (%%r10, %%r8, 4), %%r11                         \n" // C1
			"   leaq     (%%r11, %%r8, 4), %%r12                         \n" // C2
			"   leaq     (%%r12, %%r8, 4), %%r13                         \n" // C3

			"INT8_SAVE_C_M4N16:                                          \n"
			"   cmpq     $3, %%rdi                                       \n"
			"   je       INT8_SAVE_C_M3N16                               \n"
			"   cmpq     $2, %%rdi                                       \n"
			"   je       INT8_SAVE_C_M2N16                               \n"
			"   cmpq     $1, %%rdi                                       \n"
			"   je       INT8_SAVE_C_M1N16                               \n"
			"   int8_save_c_m4n16                                        \n"
			"   imul     $4, %%r15, %%r11                                \n" // temp use %%r11
			"   add      %%r11, %%r9                                     \n"
			"   movq     %%r9, %%rax                                     \n"
			"   jmp      INT8_BEGIN_M4N16                                \n"

			"INT8_SAVE_C_M3N16:                                          \n"
			"   vmovups         %%zmm12, (%%r12)%{%%k1%}                 \n"
			"   vmovups         %%zmm13, 64(%%r12)%{%%k1%}               \n"

			"INT8_SAVE_C_M2N16:                                          \n"
			"   vmovups         %%zmm10, (%%r11)%{%%k1%}                 \n"
			"   vmovups         %%zmm11, 64(%%r11)%{%%k1%}               \n"

			"INT8_SAVE_C_M1N16:                                          \n"
			"   vmovups         %%zmm8, (%%r10)%{%%k1%}                  \n"
			"   vmovups         %%zmm9, 64(%%r10)%{%%k1%}                \n"
			"   jmp      INT8_END_N16                                    \n"

			"INT8_SAVE_C_M4N16_2:                                        \n"
			"   cmpq     $3, %%rdi                                       \n"
			"   je       INT8_SAVE_C_M3N16_2                             \n"
			"   cmpq     $2, %%rdi                                       \n"
			"   je       INT8_SAVE_C_M2N16_2                             \n"
			"   cmpq     $1, %%rdi                                       \n"
			"   je       INT8_SAVE_C_M1N16_2                             \n"
			"   int8_save_c_m4n16_2                                      \n"
			"   imul     $4, %%r15, %%r11                                \n" // temp use %%r11
			"   add      %%r11, %%r9                                     \n"
			"   movq     %%r9, %%rax                                     \n"
			"   jmp      INT8_BEGIN_M4N16                                \n"

			"INT8_SAVE_C_M3N16_2:                                        \n"
			"   vpmovsdb   %%zmm12, %%xmm20                              \n"
			"   vpmovsdb   %%zmm13, %%xmm21                              \n"
			"   vmovups    %%xmm20, 64(%%r10)                            \n"
			"   vmovups    %%xmm21, 80(%%r10)                            \n"

			"INT8_SAVE_C_M2N16_2:                                        \n"
			"   vpmovsdb   %%zmm10, %%xmm18                              \n"
			"   vpmovsdb   %%zmm11, %%xmm19                              \n"
			"   vmovups    %%xmm18, 32(%%r10)                            \n"
			"   vmovups    %%xmm19, 48(%%r10)                            \n"

			"INT8_SAVE_C_M1N16_2:                                        \n"
			"   vpmovsdb   %%zmm8, %%xmm16                               \n"
			"   vpmovsdb   %%zmm9, %%xmm17                               \n"
			"   vmovups    %%xmm16, (%%r10)                              \n"
			"   vmovups    %%xmm17, 16(%%r10)                            \n"

			"INT8_END_N16:                                               \n"

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
				"zmm30", "zmm31", "memory", "ymm0", "ymm1", "ymm2", "ymm3", "ymm4", "ymm5", "ymm6", "ymm7",
				"xmm0", "xmm1", "xmm2", "xmm3", "xmm4", "xmm5", "xmm6", "xmm7", "k1");
}
