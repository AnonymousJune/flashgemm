void MGN_KERNEL12x32(float *C, float *A, float *B, long M, long N, long K, long LN, long LK, float *Ac, long k_tag);
void MGN_KERNELm8xn32(float *C, float *A, float *B, long M, long N, long K, long LN, long LK, float *Ac, long k_tag);
void MGN_KERNELm4xn32(float *C, float *A, float *B, long M, long N, long K, long LN, long LK, float *Ac, long k_tag);
// void MGN_KERNELm2xn32(float *C, float *A, float *B, long M, long N, long K, long LN, long LK, float *Ac, long k_tag);
void MGN_KERNEL1x32(float *C, float *A, float *B, long M, long N, long K, long LN, long LK, float *Ac, long k_tag);

void MGN_KERNEL12x32(float *C, float *A, float *B, long M, long N, long K, long LN, long LK, float *Ac, long k_tag)
{

	asm volatile(
		".macro    KERNEL12x32_K1_MGN                                \n"

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

		"   prefetcht0         64(%%rbx)                             \n"

		"   vbroadcastss    40(%%rax), %%zmm2                        \n"
		"   vfmadd231ps        %%zmm0, %%zmm4, %%zmm24               \n"
		"   vfmadd231ps        %%zmm0, %%zmm5, %%zmm25               \n"

		"   vbroadcastss    44(%%rax), %%zmm3                        \n"
		"   vfmadd231ps        %%zmm1, %%zmm4, %%zmm26               \n"
		"   vmovups         (%%rbx), %%zmm6                          \n"
		"    addq              $48, %%rax                            \n"
		"   vfmadd231ps        %%zmm1, %%zmm5, %%zmm27               \n"

		"   vbroadcastss    (%%rax), %%zmm0                          \n"
		"   vfmadd231ps        %%zmm2, %%zmm4, %%zmm28               \n"
		"   vmovups         64(%%rbx), %%zmm7                        \n"
		"   vfmadd231ps        %%zmm2, %%zmm5, %%zmm29               \n"

		"   vbroadcastss    4(%%rax), %%zmm1                         \n"
		"   vfmadd231ps        %%zmm3, %%zmm4, %%zmm30               \n"
		"   vfmadd231ps        %%zmm3, %%zmm5, %%zmm31               \n"

		".endm                                                       \n"

		".macro    KERNEL12x32_K2_MGN                                \n"

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

		"    addq              $128, %%rbx                           \n"

		"   vbroadcastss    36(%%rax), %%zmm1                        \n"
		"   vfmadd231ps        %%zmm3, %%zmm6, %%zmm22               \n"
		"   vfmadd231ps        %%zmm3, %%zmm7, %%zmm23               \n"

		"   prefetcht0         64(%%rbx)                             \n"

		"   vbroadcastss    40(%%rax), %%zmm2                        \n"
		"   vfmadd231ps        %%zmm0, %%zmm6, %%zmm24               \n"
		"   vfmadd231ps        %%zmm0, %%zmm7, %%zmm25               \n"

		"   vbroadcastss    44(%%rax), %%zmm3                        \n"
		"   vfmadd231ps        %%zmm1, %%zmm6, %%zmm26               \n"
		"   vmovups         (%%rbx), %%zmm4                          \n"
		"    addq              $48, %%rax                            \n"
		"   vfmadd231ps        %%zmm1, %%zmm7, %%zmm27               \n"

		"   vbroadcastss    (%%rax), %%zmm0                          \n"
		"   vfmadd231ps        %%zmm2, %%zmm6, %%zmm28               \n"
		"   vmovups         64(%%rbx), %%zmm5                        \n"
		"   vfmadd231ps        %%zmm2, %%zmm7, %%zmm29               \n"

		"   vbroadcastss    4(%%rax), %%zmm1                         \n"
		"   vfmadd231ps        %%zmm3, %%zmm6, %%zmm30               \n"
		"   vfmadd231ps        %%zmm3, %%zmm7, %%zmm31               \n"

		".endm                                                       \n"

		".macro    KERNEL12x32_END_K_MGN                             \n"

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
		// addq              $48, %%rax
		"   addq              $128, %%rbx                            \n" // TODO
		"   vfmadd231ps        %%zmm1, %%zmm7, %%zmm27               \n"

		"   vfmadd231ps        %%zmm2, %%zmm6, %%zmm28               \n"
		"   vfmadd231ps        %%zmm2, %%zmm7, %%zmm29               \n"

		"   vfmadd231ps        %%zmm3, %%zmm6, %%zmm30               \n"
		"   vfmadd231ps        %%zmm3, %%zmm7, %%zmm31               \n"

		".endm                                                       \n"

		".macro    ADD_C_12x32_MGN                                   \n"

		"   vmovups         (%%r10), %%zmm0                          \n"
		"    vaddps             %%zmm0, %%zmm8, %%zmm8               \n"
		"   vmovups         64(%%r10), %%zmm1                        \n"
		"    vaddps             %%zmm1, %%zmm9, %%zmm9               \n"
		"   vmovups         (%%r11), %%zmm2                          \n"
		"    vaddps             %%zmm2, %%zmm10, %%zmm10             \n"
		"   vmovups         64(%%r11), %%zmm3                        \n"
		"    vaddps             %%zmm3, %%zmm11, %%zmm11             \n"
		"   vmovups         (%%r12), %%zmm4                          \n"
		"    vaddps             %%zmm4, %%zmm12, %%zmm12             \n"
		"   vmovups         64(%%r12), %%zmm5                        \n"
		"    vaddps             %%zmm5, %%zmm13, %%zmm13             \n"
		"   vmovups         (%%r13), %%zmm6                          \n"
		"    vaddps             %%zmm6, %%zmm14, %%zmm14             \n"
		"   vmovups         64(%%r13), %%zmm7                        \n"
		"    vaddps             %%zmm7, %%zmm15, %%zmm15             \n"

		"    leaq              (%%r13, %%r8, 4), %%r10               \n" // C0
		"    leaq             (%%r10, %%r8, 4), %%r11                \n" // C1
		"    leaq             (%%r11, %%r8, 4), %%r12                \n" // C2
		"    leaq             (%%r12, %%r8, 4), %%r13                \n" // C3

		"   vmovups         (%%r10), %%zmm0                          \n"
		"    vaddps             %%zmm0, %%zmm16, %%zmm16             \n"
		"   vmovups         64(%%r10), %%zmm1                        \n"
		"    vaddps             %%zmm1, %%zmm17, %%zmm17             \n"
		"   vmovups         (%%r11), %%zmm2                          \n"
		"    vaddps             %%zmm2, %%zmm18, %%zmm18             \n"
		"   vmovups         64(%%r11), %%zmm3                        \n"
		"    vaddps             %%zmm3, %%zmm19, %%zmm19             \n"

		"   vmovups         (%%r12), %%zmm4                          \n"
		"    vaddps             %%zmm4, %%zmm20, %%zmm20             \n"
		"   vmovups         64(%%r12), %%zmm5                        \n"
		"    vaddps             %%zmm5, %%zmm21, %%zmm21             \n"
		"   vmovups         (%%r13), %%zmm6                          \n"
		"    vaddps             %%zmm6, %%zmm22, %%zmm22             \n"
		"   vmovups         64(%%r13), %%zmm7                        \n"
		"    vaddps             %%zmm7, %%zmm23, %%zmm23             \n"

		"    leaq              (%%r13, %%r8, 4), %%r10               \n" // C0
		"    leaq             (%%r10, %%r8, 4), %%r11                \n" // C1
		"    leaq             (%%r11, %%r8, 4), %%r12                \n" // C2
		"    leaq             (%%r12, %%r8, 4), %%r13                \n" // C3

		"   vmovups         (%%r10), %%zmm0                          \n"
		"    vaddps             %%zmm0, %%zmm24, %%zmm24             \n"
		"   vmovups         64(%%r10), %%zmm1                        \n"
		"    vaddps             %%zmm1, %%zmm25, %%zmm25             \n"
		"   vmovups         (%%r11), %%zmm2                          \n"
		"    vaddps             %%zmm2, %%zmm26, %%zmm26             \n"
		"   vmovups         64(%%r11), %%zmm3                        \n"
		"    vaddps             %%zmm3, %%zmm27, %%zmm27             \n"

		"   vmovups         (%%r12), %%zmm4                          \n"
		"    vaddps             %%zmm4, %%zmm28, %%zmm28             \n"
		"   vmovups         64(%%r12), %%zmm5                        \n"
		"    vaddps             %%zmm5, %%zmm29, %%zmm29             \n"
		"   vmovups         (%%r13), %%zmm6                          \n"
		"    vaddps             %%zmm6, %%zmm30, %%zmm30             \n"
		"   vmovups         64(%%r13), %%zmm7                        \n"
		"    vaddps             %%zmm7, %%zmm31, %%zmm31             \n"

		"    mov      %%rcx, %%r10                                   \n" // C0
		"    leaq     (%%r10, %%r8, 4), %%r11                        \n" // C1
		"    leaq     (%%r11, %%r8, 4), %%r12                        \n" // C2
		"    leaq     (%%r12, %%r8, 4), %%r13                        \n" // C3

		".endm                                                       \n"

		".macro    SAVE_12x32_MGN                                    \n"

		"   vmovups         %%zmm8, (%%r10)                          \n"
		"   vmovups         %%zmm9, 64(%%r10)                        \n"
		"   vmovups         %%zmm10, (%%r11)                         \n"
		"   vmovups         %%zmm11, 64(%%r11)                       \n"
		"   vmovups         %%zmm12, (%%r12)                         \n"
		"   vmovups         %%zmm13, 64(%%r12)                       \n"
		"   vmovups         %%zmm14, (%%r13)                         \n"
		"   vmovups         %%zmm15, 64(%%r13)                       \n"

		"    leaq         (%%r13, %%r8, 4), %%r10                    \n" // C0
		"    leaq           (%%r10, %%r8, 4), %%r11                  \n" // C1
		"    leaq           (%%r11, %%r8, 4), %%r12                  \n" // C2
		"    leaq           (%%r12, %%r8, 4), %%r13                  \n" // C3

		"   vmovups         %%zmm16, (%%r10)                         \n"
		"   vmovups         %%zmm17, 64(%%r10)                       \n"
		"   vmovups         %%zmm18, (%%r11)                         \n"
		"   vmovups         %%zmm19, 64(%%r11)                       \n"
		"   vmovups         %%zmm20, (%%r12)                         \n"
		"   vmovups         %%zmm21, 64(%%r12)                       \n"
		"   vmovups         %%zmm22, (%%r13)                         \n"
		"   vmovups         %%zmm23, 64(%%r13)                       \n"

		"    leaq         (%%r13, %%r8, 4), %%r10                    \n" // C0
		"    leaq           (%%r10, %%r8, 4), %%r11                  \n" // C1
		"    leaq           (%%r11, %%r8, 4), %%r12                  \n" // C2
		"    leaq           (%%r12, %%r8, 4), %%r13                  \n" // C3

		"   vmovups         %%zmm24, (%%r10)                         \n"
		"   vmovups         %%zmm25, 64(%%r10)                       \n"
		"   vmovups         %%zmm26, (%%r11)                         \n"
		"   vmovups         %%zmm27, 64(%%r11)                       \n"
		"    subq             $32, %%rdi                             \n" // TODO 12->32
		"   vmovups         %%zmm28, (%%r12)                         \n"
		"   vmovups         %%zmm29, 64(%%r12)                       \n"
		"   vmovups         %%zmm30, (%%r13)                         \n"
		"   vmovups         %%zmm31, 64(%%r13)                       \n"

		"    addq         $128, %%rcx                                \n" // C0

		".endm                                                       \n"

		//-----------------------------------------------------------------

		".macro    KERNEL12x16_K1_MGN                                \n"

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

		"    addq              $64, %%rbx                            \n"

		"   vbroadcastss    36(%%rax), %%zmm1                        \n"
		"   vfmadd231ps        %%zmm3, %%zmm4, %%zmm22               \n"

		"   prefetcht0         256(%%rbx)                            \n"

		"   vbroadcastss    40(%%rax), %%zmm2                        \n"
		"   vfmadd231ps        %%zmm0, %%zmm4, %%zmm24               \n"

		"   vbroadcastss    44(%%rax), %%zmm3                        \n"
		"   vfmadd231ps        %%zmm1, %%zmm4, %%zmm26               \n"
		"   vmovups            (%%rbx), %%zmm6                       \n"
		"    addq              $48, %%rax                            \n"

		"   vbroadcastss    (%%rax), %%zmm0                          \n"
		"   vfmadd231ps        %%zmm2, %%zmm4, %%zmm28               \n"

		"   vbroadcastss    4(%%rax), %%zmm1                         \n"
		"   vfmadd231ps        %%zmm3, %%zmm4, %%zmm30               \n"

		".endm                                                       \n"

		".macro    KERNEL12x16_K2_MGN                                \n"

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

		"    addq              $64, %%rbx                            \n"

		"   vbroadcastss    36(%%rax), %%zmm1                        \n"
		"   vfmadd231ps        %%zmm3, %%zmm6, %%zmm22               \n"

		"   prefetcht0         256(%%rbx)                            \n"

		"   vbroadcastss    40(%%rax), %%zmm2                        \n"
		"   vfmadd231ps        %%zmm0, %%zmm6, %%zmm24               \n"

		"   vbroadcastss    44(%%rax), %%zmm3                        \n"
		"   vfmadd231ps        %%zmm1, %%zmm6, %%zmm26               \n"
		"   vmovups         (%%rbx), %%zmm4                          \n"
		"    addq              $48, %%rax                            \n"

		"   vbroadcastss    (%%rax), %%zmm0                          \n"
		"   vfmadd231ps        %%zmm2, %%zmm6, %%zmm28               \n"

		"   vbroadcastss    4(%%rax), %%zmm1                         \n"
		"   vfmadd231ps        %%zmm3, %%zmm6, %%zmm30               \n"

		".endm                                                       \n"

		".macro    KERNEL12x16_END_K_MGN                             \n"

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

		"   addq              $64, %%rbx                             \n" // TODO

		"   vfmadd231ps        %%zmm2, %%zmm6, %%zmm28               \n"

		"   vfmadd231ps        %%zmm3, %%zmm6, %%zmm30               \n"

		".endm                                                       \n"

		".macro    ADD_C_12x16_MGN                                   \n"

		"   vmovups         (%%r10), %%zmm0                          \n"
		"    vaddps             %%zmm0, %%zmm8, %%zmm8               \n"
		"   vmovups         (%%r11), %%zmm2                          \n"
		"    vaddps             %%zmm2, %%zmm10, %%zmm10             \n"
		"   vmovups         (%%r12), %%zmm4                          \n"
		"    vaddps             %%zmm4, %%zmm12, %%zmm12             \n"
		"   vmovups         (%%r13), %%zmm6                          \n"
		"    vaddps             %%zmm6, %%zmm14, %%zmm14             \n"

		"    leaq          (%%r13, %%r8, 4), %%r10                   \n" // C0
		"    leaq             (%%r10, %%r8, 4), %%r11                \n" // C1
		"    leaq             (%%r11, %%r8, 4), %%r12                \n" // C2
		"    leaq             (%%r12, %%r8, 4), %%r13                \n" // C3

		"   vmovups         (%%r10), %%zmm0                          \n"
		"    vaddps             %%zmm0, %%zmm16, %%zmm16             \n"
		"   vmovups         (%%r11), %%zmm2                          \n"
		"    vaddps             %%zmm2, %%zmm18, %%zmm18             \n"
		"   vmovups         (%%r12), %%zmm4                          \n"
		"    vaddps             %%zmm4, %%zmm20, %%zmm20             \n"
		"   vmovups         (%%r13), %%zmm6                          \n"
		"    vaddps             %%zmm6, %%zmm22, %%zmm22             \n"

		"    leaq          (%%r13, %%r8, 4), %%r10                   \n" // C0
		"    leaq             (%%r10, %%r8, 4), %%r11                \n" // C1
		"    leaq             (%%r11, %%r8, 4), %%r12                \n" // C2
		"    leaq             (%%r12, %%r8, 4), %%r13                \n" // C3

		"   vmovups         (%%r10), %%zmm0                          \n"
		"    vaddps             %%zmm0, %%zmm24, %%zmm24             \n"
		"   vmovups         (%%r11), %%zmm2                          \n"
		"    vaddps             %%zmm2, %%zmm26, %%zmm26             \n"
		"   vmovups         (%%r12), %%zmm4                          \n"
		"    vaddps             %%zmm4, %%zmm28, %%zmm28             \n"
		"   vmovups         (%%r13), %%zmm6                          \n"
		"    vaddps             %%zmm6, %%zmm30, %%zmm30             \n"

		"    mov         %%rcx, %%r10                                \n" // C0
		"    leaq           (%%r10, %%r8, 4), %%r11                  \n" // C1
		"    leaq           (%%r11, %%r8, 4), %%r12                  \n" // C2
		"    leaq         (%%r12, %%r8, 4), %%r13                    \n" // C3

		".endm                                                       \n"

		".macro    SAVE_12x16_MGN                                    \n"

		"   vmovups         %%zmm8, (%%r10)                          \n"
		"   vmovups         %%zmm10, (%%r11)                         \n"
		"   vmovups         %%zmm12, (%%r12)                         \n"
		"   vmovups         %%zmm14, (%%r13)                         \n"

		"    leaq         (%%r13, %%r8, 4), %%r10                    \n" // C0
		"    leaq           (%%r10, %%r8, 4), %%r11                  \n" // C1
		"    leaq           (%%r11, %%r8, 4), %%r12                  \n" // C2
		"    leaq           (%%r12, %%r8, 4), %%r13                  \n" // C3

		"   vmovups         %%zmm16, (%%r10)                         \n"
		"   vmovups         %%zmm18, (%%r11)                         \n"
		"   vmovups         %%zmm20, (%%r12)                         \n"
		"   vmovups         %%zmm22, (%%r13)                         \n"

		"    leaq         (%%r13, %%r8, 4), %%r10                    \n" // C0
		"    leaq           (%%r10, %%r8, 4), %%r11                  \n" // C1
		"    leaq           (%%r11, %%r8, 4), %%r12                  \n" // C2
		"    leaq           (%%r12, %%r8, 4), %%r13                  \n" // C3

		"   vmovups         %%zmm24, (%%r10)                         \n"
		"   vmovups         %%zmm26, (%%r11)                         \n"
		"    subq             $16, %%rdi                             \n" // TODO 12->16
		"   vmovups         %%zmm28, (%%r12)                         \n"
		"   vmovups         %%zmm30, (%%r13)                         \n"

		"    addq         $64, %%rcx                                 \n" // C0

		".endm                                                       \n"

		//-----------------------------------------------------------------

		".macro    KERNEL12x1_K1_MGN                                 \n"

		"   movss    8(%%rax), %%xmm2                                \n"
		"   vfmadd231ps        %%xmm0, %%xmm4, %%xmm8                \n"

		"   movss    12(%%rax), %%xmm3                               \n"
		"   vfmadd231ps        %%xmm1, %%xmm4, %%xmm10               \n"

		"   prefetcht0         256(%%rax)                            \n"

		"   movss    16(%%rax), %%xmm0                               \n"
		"   vfmadd231ps        %%xmm2, %%xmm4, %%xmm12               \n"

		"   movss    20(%%rax), %%xmm1                               \n"
		"   vfmadd231ps        %%xmm3, %%xmm4, %%xmm14               \n"

		"   movss    24(%%rax), %%xmm2                               \n"
		"   vfmadd231ps        %%xmm0, %%xmm4, %%xmm16               \n"

		"   movss    28(%%rax), %%xmm3                               \n"
		"   vfmadd231ps        %%xmm1, %%xmm4, %%xmm18               \n"

		"   movss    32(%%rax), %%xmm0                               \n"
		"   vfmadd231ps        %%xmm2, %%xmm4, %%xmm20               \n"

		"    addq              $4, %%rbx                             \n" // TODO

		"   movss    36(%%rax), %%xmm1                               \n"
		"   vfmadd231ps        %%xmm3, %%xmm4, %%xmm22               \n"

		"   prefetcht0         256(%%rbx)                            \n"

		"   movss    40(%%rax), %%xmm2                               \n"
		"   vfmadd231ps        %%xmm0, %%xmm4, %%xmm24               \n"

		"   movss    44(%%rax), %%xmm3                               \n"
		"   vfmadd231ps        %%xmm1, %%xmm4, %%xmm26               \n"
		"   movss            (%%rbx), %%xmm6                         \n"
		"    addq              $48, %%rax                            \n"

		"   movss    (%%rax), %%xmm0                                 \n"
		"   vfmadd231ps        %%xmm2, %%xmm4, %%xmm28               \n"

		"   movss    4(%%rax), %%xmm1                                \n"
		"   vfmadd231ps        %%xmm3, %%xmm4, %%xmm30               \n"

		".endm                                                       \n"

		".macro    KERNEL12x1_K2_MGN                                 \n"

		"   movss    8(%%rax), %%xmm2                                \n"
		"   vfmadd231ps        %%xmm0, %%xmm6, %%xmm8                \n"

		"   movss    12(%%rax), %%xmm3                               \n"
		"   vfmadd231ps        %%xmm1, %%xmm6, %%xmm10               \n"

		"   prefetcht0         256(%%rax)                            \n"

		"   movss    16(%%rax), %%xmm0                               \n"
		"   vfmadd231ps        %%xmm2, %%xmm6, %%xmm12               \n"

		"   movss    20(%%rax), %%xmm1                               \n"
		"   vfmadd231ps        %%xmm3, %%xmm6, %%xmm14               \n"

		"   movss    24(%%rax), %%xmm2                               \n"
		"   vfmadd231ps        %%xmm0, %%xmm6, %%xmm16               \n"

		"   movss    28(%%rax), %%xmm3                               \n"
		"   vfmadd231ps        %%xmm1, %%xmm6, %%xmm18               \n"

		"   movss    32(%%rax), %%xmm0                               \n"
		"   vfmadd231ps        %%xmm2, %%xmm6, %%xmm20               \n"

		"    addq              $4, %%rbx                             \n" // TODO

		"   movss    36(%%rax), %%xmm1                               \n"
		"   vfmadd231ps        %%xmm3, %%xmm6, %%xmm22               \n"

		"   prefetcht0         256(%%rbx)                            \n"

		"   movss    40(%%rax), %%xmm2                               \n"
		"   vfmadd231ps        %%xmm0, %%xmm6, %%xmm24               \n"

		"   movss    44(%%rax), %%xmm3                               \n"
		"   vfmadd231ps        %%xmm1, %%xmm6, %%xmm26               \n"
		"   movss         (%%rbx), %%xmm4                            \n"
		"    addq              $48, %%rax                            \n"

		"   movss    (%%rax), %%xmm0                                 \n"
		"   vfmadd231ps        %%xmm2, %%xmm6, %%xmm28               \n"

		"   movss    4(%%rax), %%xmm1                                \n"
		"   vfmadd231ps        %%xmm3, %%xmm6, %%xmm30               \n"

		".endm                                                       \n"

		".macro    KERNEL12x1_END_K_MGN                              \n"

		"   movss    8(%%rax), %%xmm2                                \n"
		"   vfmadd231ps        %%xmm0, %%xmm6, %%xmm8                \n"

		"   movss    12(%%rax), %%xmm3                               \n"
		"   vfmadd231ps        %%xmm1, %%xmm6, %%xmm10               \n"

		"   prefetcht0         256(%%rax)                            \n"

		"   movss    16(%%rax), %%xmm0                               \n"
		"   vfmadd231ps        %%xmm2, %%xmm6, %%xmm12               \n"

		"   movss    20(%%rax), %%xmm1                               \n"
		"   vfmadd231ps        %%xmm3, %%xmm6, %%xmm14               \n"

		"   movss    24(%%rax), %%xmm2                               \n"
		"   vfmadd231ps        %%xmm0, %%xmm6, %%xmm16               \n"

		"   movss    28(%%rax), %%xmm3                               \n"
		"   vfmadd231ps        %%xmm1, %%xmm6, %%xmm18               \n"

		"   movss    32(%%rax), %%xmm0                               \n"
		"   vfmadd231ps        %%xmm2, %%xmm6, %%xmm20               \n"

		"   movss    36(%%rax), %%xmm1                               \n"
		"   vfmadd231ps        %%xmm3, %%xmm6, %%xmm22               \n"

		"   movss    40(%%rax), %%xmm2                               \n"
		"   vfmadd231ps        %%xmm0, %%xmm6, %%xmm24               \n"

		"   movss    44(%%rax), %%xmm3                               \n"
		"   vfmadd231ps        %%xmm1, %%xmm6, %%xmm26               \n"

		"   addq              $4, %%rbx                              \n" // TODO

		"   vfmadd231ps        %%xmm2, %%xmm6, %%xmm28               \n"

		"   vfmadd231ps        %%xmm3, %%xmm6, %%xmm30               \n"

		".endm                                                       \n"

		".macro    ADD_C_12x1_MGN                                    \n"

		"   movss         (%%r10), %%xmm0                            \n"
		"    vaddps             %%xmm0, %%xmm8, %%xmm8               \n"
		"   movss         (%%r11), %%xmm2                            \n"
		"    vaddps             %%xmm2, %%xmm10, %%xmm10             \n"
		"   movss         (%%r12), %%xmm4                            \n"
		"    vaddps             %%xmm4, %%xmm12, %%xmm12             \n"
		"   movss         (%%r13), %%xmm6                            \n"
		"    vaddps             %%xmm6, %%xmm14, %%xmm14             \n"

		"    leaq          (%%r13, %%r8, 4), %%r10                   \n" // C0
		"    leaq             (%%r10, %%r8, 4), %%r11                \n" // C1
		"    leaq             (%%r11, %%r8, 4), %%r12                \n" // C2
		"    leaq             (%%r12, %%r8, 4), %%r13                \n" // C3

		"   movss         (%%r10), %%xmm0                            \n"
		"    vaddps             %%xmm0, %%xmm16, %%xmm16             \n"
		"   movss         (%%r11), %%xmm2                            \n"
		"    vaddps             %%xmm2, %%xmm18, %%xmm18             \n"
		"   movss         (%%r12), %%xmm4                            \n"
		"    vaddps             %%xmm4, %%xmm20, %%xmm20             \n"
		"   movss         (%%r13), %%xmm6                            \n"
		"    vaddps             %%xmm6, %%xmm22, %%xmm22             \n"

		"    leaq          (%%r13, %%r8, 4), %%r10                   \n" // C0
		"    leaq             (%%r10, %%r8, 4), %%r11                \n" // C1
		"    leaq             (%%r11, %%r8, 4), %%r12                \n" // C2
		"    leaq             (%%r12, %%r8, 4), %%r13                \n" // C3

		"   movss         (%%r10), %%xmm0                            \n"
		"    vaddps             %%xmm0, %%xmm24, %%xmm24             \n"
		"   movss         (%%r11), %%xmm2                            \n"
		"    vaddps             %%xmm2, %%xmm26, %%xmm26             \n"
		"   movss         (%%r12), %%xmm4                            \n"
		"    vaddps             %%xmm4, %%xmm28, %%xmm28             \n"
		"   movss         (%%r13), %%xmm6                            \n"
		"    vaddps             %%xmm6, %%xmm30, %%xmm30             \n"

		"    mov         %%rcx, %%r10                                \n" // C0
		"    leaq           (%%r10, %%r8, 4), %%r11                  \n" // C1
		"    leaq           (%%r11, %%r8, 4), %%r12                  \n" // C2
		"    leaq         (%%r12, %%r8, 4), %%r13                    \n" // C3

		".endm                                                       \n"

		".macro    SAVE_12x1_MGN                                     \n"

		"   vmovss         %%xmm8, (%%r10)                           \n"
		"   vmovss         %%xmm10, (%%r11)                          \n"
		"   vmovss         %%xmm12, (%%r12)                          \n"
		"   vmovss         %%xmm14, (%%r13)                          \n"

		"    leaq         (%%r13, %%r8, 4), %%r10                    \n" // C0
		"    leaq           (%%r10, %%r8, 4), %%r11                  \n" // C1
		"    leaq           (%%r11, %%r8, 4), %%r12                  \n" // C2
		"    leaq           (%%r12, %%r8, 4), %%r13                  \n" // C3

		"   vmovss         %%xmm16, (%%r10)                          \n"
		"   vmovss         %%xmm18, (%%r11)                          \n"
		"   vmovss         %%xmm20, (%%r12)                          \n"
		"   vmovss         %%xmm22, (%%r13)                          \n"

		"    leaq         (%%r13, %%r8, 4), %%r10                    \n" // C0
		"    leaq           (%%r10, %%r8, 4), %%r11                  \n" // C1
		"    leaq           (%%r11, %%r8, 4), %%r12                  \n" // C2
		"    leaq           (%%r12, %%r8, 4), %%r13                  \n" // C3

		"   vmovss         %%xmm24, (%%r10)                          \n"
		"   vmovss         %%xmm26, (%%r11)                          \n"
		"    subq             $1, %%rdi                              \n" // TODO
		"   vmovss         %%xmm28, (%%r12)                          \n"
		"   vmovss         %%xmm30, (%%r13)                          \n"

		"    addq         $4, %%rcx                                  \n" // TODO

		".endm                                                       \n"

		//-----------------------------------------------------------------
		// pack A
		"    movl %[LK], %%r8d                                       \n"
		"    movl %[LK], %%r15d                                      \n"

		"    mov %[Ac], %%r9                                         \n"
		"    movl %[K], %%r10d                                       \n"
		"    mov %[A], %%r11                                         \n"
		"    mov %[A], %%r12                                         \n"
		"    mov %[A], %%r13                                         \n"
		"    mov %[A], %%r14                                         \n"

		"    mov %[A], %%rbx                                         \n"
		"    mov %[A], %%rcx                                         \n"
		"    mov %[A], %%rdx                                         \n"
		"    mov %[A], %%rdi                                         \n"

		"    shr $4, %%r10                                           \n"
		"    shl $2, %%r8                                            \n"
		"    shl $3, %%r15                                           \n"
		"    add %%r8, %%r14                                         \n"
		"    add %%r8, %%rdi                                         \n"
		"    add %%r8, %%r12                                         \n"
		"    add %%r8, %%rcx                                         \n"
		"    add %%r15, %%r13                                        \n"
		"    add %%r15, %%rdx                                        \n"
		"    shl $2, %%r8                                            \n"
		"    add %%r15, %%r14                                        \n"
		"    add %%r15, %%rdi                                        \n"
		"    add %%r8, %%rbx                                         \n"
		"    add %%r8, %%rcx                                         \n"
		"    add %%r8, %%rdx                                         \n"
		"    add %%r8, %%rdi                                         \n"

		"    shl $1, %%r15                                           \n"

		"    vmovups (%%r11), %%zmm2                                 \n"
		"    vmovups (%%r12), %%zmm3                                 \n"
		"    vmovups (%%r13), %%zmm4                                 \n"
		"    vmovups (%%r14), %%zmm5                                 \n"

		"    sub $64, %%r15                                          \n"

		"    vmovups (%%rbx), %%zmm6                                 \n"
		"    vmovups (%%rcx), %%zmm7                                 \n"
		"    vmovups (%%rdx), %%zmm8                                 \n"
		"    vmovups (%%rdi), %%zmm9                                 \n"

		"    mov %%r15, %%rsi                                        \n"

		"    jmp NPACK_M12_MGN                                       \n"

		"NPACK_PRE_M12_MGN:                                          \n"
		"    add $64, %%r11                                          \n"
		"    add $64, %%r12                                          \n"
		"    add $64, %%r13                                          \n"
		"    add $64, %%r14                                          \n"
		"    sub %%rsi, %%rbx                                        \n"
		"    sub %%rsi, %%rcx                                        \n"
		"    sub %%rsi, %%rdx                                        \n"
		"    sub %%rsi, %%rdi                                        \n"

		"    add $768, %%r9                                          \n"

		"    vmovups (%%r11), %%zmm2                                 \n"
		"    vmovups (%%r12), %%zmm3                                 \n"
		"    vmovups (%%r13), %%zmm4                                 \n"
		"    vmovups (%%r14), %%zmm5                                 \n"
		"    vmovups (%%rbx), %%zmm6                                 \n"
		"    vmovups (%%rcx), %%zmm7                                 \n"
		"    vmovups (%%rdx), %%zmm8                                 \n"
		"    vmovups (%%rdi), %%zmm9                                 \n"

		"NPACK_M12_MGN:                                              \n"

		"    movl     $0xaa, %%eax                                   \n"
		"    movl     $0xcc, %%r15d                                  \n"

		"    kmovd    %%eax, %%k1                                    \n"
		"    kmovd    %%r15d, %%k2                                   \n"

		"    vunpcklps %%zmm3, %%zmm2, %%zmm0                        \n"
		"    vunpcklps %%zmm7, %%zmm6, %%zmm1                        \n"

		"    movl     $0x33, %%eax                                   \n"

		"    vunpcklps %%zmm5, %%zmm4, %%zmm26                       \n"
		"    vunpcklps %%zmm9, %%zmm8, %%zmm27                       \n"

		"    kmovd    %%eax, %%k3                                    \n"

		"    vpermq  $0x80, %%zmm26, %%zmm0%{%%k1%}                  \n"
		"    vmovups %%zmm0, %%zmm28                                 \n"
		"    vpermq  $0x80, %%zmm27, %%zmm1%{%%k1%}                  \n"
		"    vmovups %%zmm1, %%zmm29                                 \n"

		"    movl     $0x33, %%r15d                                  \n"

		"    vpermq  $0x40, %%zmm1, %%zmm0%{%%k2%}                   \n"
		"    vpermq  $0x0e, %%zmm28, %%zmm29%{%%k3%}                 \n"

		"    vmovups %%ymm0, (%%r9)                                  \n"
		"    vextractf64x4 $0x1, %%zmm0,  %%ymm30                    \n"
		"    vmovups %%ymm30, 384(%%r9)                              \n"

		"    vmovups %%ymm29, 192(%%r9)                              \n"
		"    vextractf64x4 $0x1, %%zmm29,  %%ymm31                   \n"
		"    vmovups %%ymm31, 576(%%r9)                              \n"

		"    movl     $0x55, %%eax                                   \n"
		"    movl     $0xcc, %%r15d                                  \n"

		"    kmovd    %%eax, %%k1                                    \n"
		"    kmovd    %%r15d, %%k2                                   \n"

		"    vunpcklps %%zmm3, %%zmm2, %%zmm0                        \n"
		"    vunpcklps %%zmm7, %%zmm6, %%zmm1                        \n"

		"    movl     $0x33, %%eax                                   \n"

		"    vunpcklps %%zmm5, %%zmm4, %%zmm26                       \n"
		"    vunpcklps %%zmm9, %%zmm8, %%zmm27                       \n"

		"    kmovd    %%eax, %%k3                                    \n"

		"    vpermq  $0x31, %%zmm0, %%zmm26%{%%k1%}                  \n"
		"    vmovups %%zmm26, %%zmm28                                \n"
		"    vpermq  $0x31, %%zmm1, %%zmm27%{%%k1%}                  \n"
		"    vmovups %%zmm27, %%zmm29                                \n"

		"    vpermq  $0x40, %%zmm27, %%zmm26%{%%k2%}                 \n"
		"    vpermq  $0x0e, %%zmm28, %%zmm29%{%%k3%}                 \n"

		"    vmovups %%ymm26, 48(%%r9)                               \n"
		"    vextractf64x4  $0x1,%%zmm26, %%ymm30                    \n"
		"    vmovups %%ymm30, 432(%%r9)                              \n"

		"    vmovups %%ymm29, 240(%%r9)                              \n"
		"    vextractf64x4 $0x1, %%zmm29,  %%ymm31                   \n"
		"    vmovups %%ymm31, 624(%%r9)                              \n"

		"    movl     $0xaa, %%eax                                   \n"
		"    movl     $0xcc, %%r15d                                  \n"

		"    kmovd    %%eax, %%k1                                    \n"
		"    kmovd    %%r15d, %%k2                                   \n"

		"    vunpckhps %%zmm3, %%zmm2, %%zmm0                        \n"
		"    vunpckhps %%zmm7, %%zmm6, %%zmm1                        \n"

		"    movl     $0x33, %%eax                                   \n"

		"    vunpckhps %%zmm5, %%zmm4, %%zmm26                       \n"
		"    vunpckhps %%zmm9, %%zmm8, %%zmm27                       \n"

		"    kmovd    %%eax, %%k3                                    \n"

		"    vpermq    $0x80, %%zmm26, %%zmm0%{%%k1%}                \n"
		"    vmovups %%zmm0, %%zmm28                                 \n"
		"    vpermq    $0x80, %%zmm27, %%zmm1%{%%k1%}                \n"
		"    vmovups %%zmm1, %%zmm29                                 \n"

		"    vpermq  $0x40, %%zmm1, %%zmm0%{%%k2%}                   \n"
		"    vpermq  $0x0e, %%zmm28, %%zmm29%{%%k3%}                 \n"

		"    vmovups %%ymm0, 96(%%r9)                                \n"
		"    vextractf64x4 $0x1, %%zmm0,  %%ymm30                    \n"
		"    vmovups %%ymm30, 480(%%r9)                              \n"

		"    vmovups %%ymm29, 288(%%r9)                              \n"
		"    vextractf64x4 $0x1, %%zmm29,  %%ymm31                   \n"
		"    vmovups %%ymm31, 672(%%r9)                              \n"

		"    movl     $0x55, %%eax                                   \n"
		"    movl     $0xcc, %%r15d                                  \n"

		"    kmovd    %%eax, %%k1                                    \n"
		"    kmovd    %%r15d, %%k2                                   \n"

		"    vunpckhps %%zmm3, %%zmm2, %%zmm0                        \n"
		"    vunpckhps %%zmm7, %%zmm6, %%zmm1                        \n"

		"    movl     $0x33, %%eax                                   \n"

		"    vunpckhps %%zmm5, %%zmm4, %%zmm26                       \n"
		"    vunpckhps %%zmm9, %%zmm8, %%zmm27                       \n"

		"    kmovd    %%eax, %%k3                                    \n"

		"    vpermq  $0x31, %%zmm0, %%zmm26%{%%k1%}                  \n"
		"    vmovups %%zmm26, %%zmm28                                \n"
		"    vpermq  $0x31, %%zmm1, %%zmm27%{%%k1%}                  \n"
		"    vmovups %%zmm27, %%zmm29                                \n"

		"    vpermq  $0x40, %%zmm27, %%zmm26%{%%k2%}                 \n"
		"    vpermq  $0x0e, %%zmm28, %%zmm29%{%%k3%}                 \n"

		"    vmovups %%ymm26, 144(%%r9)                              \n"
		"    vextractf64x4  $0x1,%%zmm26, %%ymm30                    \n"
		"    vmovups %%ymm30, 528(%%r9)                              \n"

		"    vmovups %%ymm29, 336(%%r9)                              \n"
		"    vextractf64x4 $0x1, %%zmm29,  %%ymm31                   \n"
		"    vmovups %%ymm31, 720(%%r9)                              \n"

		"    add %%r8, %%rbx                                         \n"
		"    add %%r8, %%rcx                                         \n"
		"    add %%r8, %%rdx                                         \n"
		"    add %%r8, %%rdi                                         \n"

		"    vmovups (%%rbx), %%zmm2                                 \n"
		"    vmovups (%%rcx), %%zmm3                                 \n"
		"    vmovups (%%rdx), %%zmm4                                 \n"
		"    vmovups (%%rdi), %%zmm5                                 \n"

		"    movl     $0xaa, %%eax                                   \n"
		"    kmovd    %%eax, %%k1                                    \n"
		"    vunpcklps %%zmm3, %%zmm2, %%zmm0                        \n"
		"    vunpcklps %%zmm5, %%zmm4, %%zmm26                       \n"

		"    vpermq    $0x80, %%zmm26, %%zmm0%{%%k1%}                \n"
		"    vmovups %%xmm0, 32(%%r9)                                \n"
		"    vextractf32x4  $0x1, %%zmm0, %%xmm28                    \n"
		"    vextractf32x4  $0x2, %%zmm0, %%xmm29                    \n"
		"    vextractf32x4  $0x3, %%zmm0, %%xmm30                    \n"

		"    vmovups %%xmm28, 224(%%r9)                              \n"
		"    vmovups %%xmm29, 416(%%r9)                              \n"
		"    vmovups %%xmm30, 608(%%r9)                              \n"

		"    movl     $0x55, %%eax                                   \n"
		"    kmovd    %%eax, %%k1                                    \n"

		"    vunpcklps %%zmm3, %%zmm2, %%zmm0                        \n"
		"    vunpcklps %%zmm5, %%zmm4, %%zmm26                       \n"

		"    vpermq  $0x31, %%zmm0, %%zmm26%{%%k1%}                  \n"
		"    vmovups %%xmm26, 80(%%r9)                               \n"
		"    vextractf32x4  $0x1, %%zmm26, %%xmm28                   \n"
		"    vextractf32x4  $0x2, %%zmm26, %%xmm29                   \n"
		"    vextractf32x4  $0x3, %%zmm26, %%xmm30                   \n"

		"    vmovups %%xmm28, 272(%%r9)                              \n"
		"    vmovups %%xmm29, 464(%%r9)                              \n"
		"    vmovups %%xmm30, 656(%%r9)                              \n"

		"    movl     $0xaa, %%eax                                   \n"
		"    kmovd    %%eax, %%k1                                    \n"
		"    vunpckhps %%zmm3, %%zmm2, %%zmm0                        \n"
		"    vunpckhps %%zmm5, %%zmm4, %%zmm26                       \n"
		"    vpermq    $0x80, %%zmm26, %%zmm0%{%%k1%}                \n"
		"    vmovups %%xmm0, 128(%%r9)                               \n"
		"    vextractf32x4  $0x1, %%zmm0, %%xmm28                    \n"
		"    vextractf32x4  $0x2, %%zmm0, %%xmm29                    \n"
		"    vextractf32x4  $0x3, %%zmm0, %%xmm30                    \n"

		"    vmovups %%xmm28, 320(%%r9)                              \n"
		"    vmovups %%xmm29, 512(%%r9)                              \n"
		"    vmovups %%xmm30, 704(%%r9)                              \n"

		"    movl     $0x55, %%eax                                   \n"
		"    kmovd    %%eax, %%k1                                    \n"

		"    vunpckhps %%zmm3, %%zmm2, %%zmm0                        \n"
		"    vunpckhps %%zmm5, %%zmm4, %%zmm26                       \n"

		"    vpermq  $0x31, %%zmm0, %%zmm26%{%%k1%}                  \n"
		"    vmovups %%xmm26, 176(%%r9)                              \n"
		"    vextractf32x4  $0x1, %%zmm26, %%xmm28                   \n"
		"    vextractf32x4  $0x2, %%zmm26, %%xmm29                   \n"
		"    vextractf32x4  $0x3, %%zmm26, %%xmm30                   \n"
		"    sub     $1, %%r10                                       \n"
		"    vmovups %%xmm28, 368(%%r9)                              \n"
		"    vmovups %%xmm29, 560(%%r9)                              \n"
		"    vmovups %%xmm30, 752(%%r9)                              \n"

		"    je  NPACK_END_M12_MGN                                   \n"
		"    jmp NPACK_PRE_M12_MGN                                   \n"

		"NPACK_END_M12_MGN:                                          \n"

		//-----------------------------------------------------------------

		"SMM_NN_KERNEL12x32_MGN:                                     \n"

		"   mov     %[C], %%rcx                                      \n"
		"   mov     %[Ac], %%rax                                     \n"
		"   mov     %[B], %%rbx                                      \n"

		"   prefetcht0         (%%rax)                               \n"

		"    mov     %[K], %%rdx                                     \n" // K(kc)
		"    mov      %[LN], %%r8                                    \n"
		"   mov      %[Ac], %%r14                                    \n" // 存储Ac地址
		"    movq  %[N], %%rdi                                       \n"
		"    mov     %[k_tag], %%r15                                 \n" // kk=0把C存回内存, 否则加回对应的C位置

		"   prefetcht0         (%%rbx)                               \n"

		"    mov     %%rdx, %%rsi                                    \n" // K

		//-----------------------------------------------------------------

		"BEGIN_N32_MGN:                                              \n"
		"   cmpq     $32, %%rdi                                      \n" // N % 16
		"    jb          BEGIN_N16_MGN                               \n"

		"    mov     %%r14, %%rax                                    \n" // Ac
		"   prefetcht0         (%%rax)                               \n"

		"    mov      %%rcx, %%r10                                   \n" // C0
		"   prefetcht1         (%%r10)                               \n"
		"    leaq     (%%r10, %%r8, 4), %%r11                        \n" // C1
		"   prefetcht1         (%%r11)                               \n"
		"    leaq     (%%r11, %%r8, 4), %%r12                        \n" // C2
		"   prefetcht1         (%%r12)                               \n"
		"    leaq     (%%r12, %%r8, 4), %%r13                        \n" // C3
		"   prefetcht1         (%%r13)                               \n"

		"    mov     %%rsi, %%rdx                                    \n" // K

		"   vmovups        (%%rbx), %%zmm4                           \n" // B0-15
		"    vpxorq         %%zmm8, %%zmm8, %%zmm8                   \n"
		"    vpxorq         %%zmm9, %%zmm9, %%zmm9                   \n"
		"    vpxorq         %%zmm10, %%zmm10, %%zmm10                \n"
		"    vpxorq         %%zmm11, %%zmm11, %%zmm11                \n"
		"   vmovups     64(%%rbx), %%zmm5                            \n"
		"    vpxorq         %%zmm12, %%zmm12, %%zmm12                \n"
		"    vpxorq         %%zmm13, %%zmm13, %%zmm13                \n"
		"    vpxorq         %%zmm14, %%zmm14, %%zmm14                \n"
		"    vpxorq         %%zmm15, %%zmm15, %%zmm15                \n"
		"    vpxorq         %%zmm16, %%zmm16, %%zmm16                \n"
		"    vpxorq         %%zmm17, %%zmm17, %%zmm17                \n"
		"    vpxorq         %%zmm18, %%zmm18, %%zmm18                \n"
		"    vpxorq         %%zmm19, %%zmm19, %%zmm19                \n"

		"   vbroadcastss    (%%rax), %%zmm0                          \n" // A0
		"   vbroadcastss    4(%%rax), %%zmm1                         \n" // A1

		"    vpxorq         %%zmm20, %%zmm20, %%zmm20                \n"
		"    vpxorq         %%zmm21, %%zmm21, %%zmm21                \n"
		"    vpxorq         %%zmm22, %%zmm22, %%zmm22                \n"
		"    vpxorq         %%zmm23, %%zmm23, %%zmm23                \n"
		"    vpxorq         %%zmm24, %%zmm24, %%zmm24                \n"
		"    vpxorq         %%zmm25, %%zmm25, %%zmm25                \n"
		"    vpxorq         %%zmm26, %%zmm26, %%zmm26                \n"
		"    vpxorq         %%zmm27, %%zmm27, %%zmm27                \n"
		"    vpxorq         %%zmm28, %%zmm28, %%zmm28                \n"
		"    vpxorq         %%zmm29, %%zmm29, %%zmm29                \n"
		"    vpxorq         %%zmm30, %%zmm30, %%zmm30                \n"
		"    vpxorq         %%zmm31, %%zmm31, %%zmm31                \n"

		"    subq     $8, %%rdx                                      \n"

		"K_N32_PREFETCH_C_MGN:                                       \n"
		"   leaq     (%%r13, %%r8, 4), %%r13                         \n"
		"   prefetcht2         (%%r13)                               \n"
		"   prefetcht2         64(%%r13)                             \n"

		"MAIN_K_N32_MGN:                                             \n"

		"    KERNEL12x32_K1_MGN                                      \n"
		"    KERNEL12x32_K2_MGN                                      \n"
		"    KERNEL12x32_K1_MGN                                      \n"
		"    KERNEL12x32_K2_MGN                                      \n"
		"    KERNEL12x32_K1_MGN                                      \n"
		"    KERNEL12x32_K2_MGN                                      \n"
		"    KERNEL12x32_K1_MGN                                      \n"
		"   cmp     $0, %%rdx                                        \n"
		"    je         EDGE_K_N32_MGN                               \n"
		"    KERNEL12x32_K2_MGN                                      \n"
		"   subq     $8, %%rdx                                       \n"
		"   cmp   $64, %%rdx                                         \n"
		"   jbe     K_N32_PREFETCH_C_MGN                             \n"
		"   jmp     MAIN_K_N32_MGN                                   \n"

		"EDGE_K_N32_MGN:                                             \n"
		"   leaq     (%%r12, %%r8, 4), %%r13                         \n"
		"    KERNEL12x32_END_K_MGN                                   \n"

		"BEGIN_SAVE_N32_MGN:                                         \n"
		"    cmp     $0, %%r15                                       \n"
		"    je      SAVE_C_N32_MGN                                  \n"
		"    ADD_C_12x32_MGN                                         \n"

		"SAVE_C_N32_MGN:                                             \n"
		"    SAVE_12x32_MGN                                          \n"
		"    cmpq      $32, %%rdi                                    \n"
		"    jnb     BEGIN_N32_MGN                                   \n" // 不小于（或等于）则跳转

		//----------------------------------------------------------------

		"BEGIN_N16_MGN:                                              \n"
		"   cmpq     $16, %%rdi                                      \n" // N % 16
		"    jb          BEGIN_N1_MGN                                \n" // TODO

		"    mov     %%r14, %%rax                                    \n" // Ac
		"   prefetcht0         (%%rax)                               \n"

		"    mov      %%rcx, %%r10                                   \n" // C0
		"   prefetcht1         (%%r10)                               \n"
		"    leaq     (%%r10, %%r8, 4), %%r11                        \n" // C1
		"   prefetcht1         (%%r11)                               \n"
		"    leaq     (%%r11, %%r8, 4), %%r12                        \n" // C2
		"   prefetcht1         (%%r12)                               \n"
		"    leaq     (%%r12, %%r8, 4), %%r13                        \n" // C3
		"   prefetcht1         (%%r13)                               \n"

		"    mov     %%rsi, %%rdx                                    \n" // K

		"   vmovups        (%%rbx), %%zmm4                           \n" // B0-15

		"   vpxorq         %%zmm8, %%zmm8, %%zmm8                    \n"
		"    vpxorq         %%zmm10, %%zmm10, %%zmm10                \n"
		"    vpxorq         %%zmm12, %%zmm12, %%zmm12                \n"
		"    vpxorq         %%zmm14, %%zmm14, %%zmm14                \n"
		"    vpxorq         %%zmm16, %%zmm16, %%zmm16                \n"
		"    vpxorq         %%zmm18, %%zmm18, %%zmm18                \n"

		"   vbroadcastss    (%%rax), %%zmm0                          \n" // A0
		"   vbroadcastss    4(%%rax), %%zmm1                         \n" // A1

		"    vpxorq         %%zmm20, %%zmm20, %%zmm20                \n"
		"    vpxorq         %%zmm22, %%zmm22, %%zmm22                \n"
		"    vpxorq         %%zmm24, %%zmm24, %%zmm24                \n"
		"    vpxorq         %%zmm26, %%zmm26, %%zmm26                \n"
		"    vpxorq         %%zmm28, %%zmm28, %%zmm28                \n"
		"    vpxorq         %%zmm30, %%zmm30, %%zmm30                \n"

		"    subq     $8, %%rdx                                      \n"

		"K_N16_PREFETCH_C_MGN:                                       \n"
		"   leaq     (%%r13, %%r8, 4), %%r13                         \n"
		"   prefetcht2         (%%r13)                               \n"
		"   prefetcht2         64(%%r13)                             \n"

		"MAIN_K_N16_MGN:                                             \n"

		"    KERNEL12x16_K1_MGN                                      \n"
		"    KERNEL12x16_K2_MGN                                      \n"
		"    KERNEL12x16_K1_MGN                                      \n"
		"    KERNEL12x16_K2_MGN                                      \n"
		"    KERNEL12x16_K1_MGN                                      \n"
		"    KERNEL12x16_K2_MGN                                      \n"
		"    KERNEL12x16_K1_MGN                                      \n"
		"   cmp     $0, %%rdx                                        \n"
		"    je         EDGE_K_N16_MGN                               \n"
		"    KERNEL12x16_K2_MGN                                      \n"
		"   subq     $8, %%rdx                                       \n"
		"   cmp   $32, %%rdx                                         \n"
		"   jbe     K_N16_PREFETCH_C_MGN                             \n"
		"   jmp     MAIN_K_N16_MGN                                   \n"

		"EDGE_K_N16_MGN:                                             \n"
		"   leaq     (%%r12, %%r8, 4), %%r13                         \n"
		"    KERNEL12x16_END_K_MGN                                   \n"
		"    cmp     $0, %%r15                                       \n"
		"    je      SAVE_C_N16_MGN                                  \n"
		"    ADD_C_12x16_MGN                                         \n"

		"SAVE_C_N16_MGN:                                             \n"
		"    SAVE_12x16_MGN                                          \n"

		//----------------------------------------------------------------

		"BEGIN_N1_MGN:                                               \n"
		"   cmpq     $1, %%rdi                                       \n" // N % 16
		"    jb          END_M12N1_MGN                               \n" // TODO

		"    mov     %%r14, %%rax                                    \n" // Ac
		"   prefetcht0         (%%rax)                               \n"

		"    mov      %%rcx, %%r10                                   \n" // C0
		"   prefetcht1         (%%r10)                               \n"
		"    leaq     (%%r10, %%r8, 4), %%r11                        \n" // C1
		"   prefetcht1         (%%r11)                               \n"
		"    leaq     (%%r11, %%r8, 4), %%r12                        \n" // C2
		"   prefetcht1         (%%r12)                               \n"
		"    leaq     (%%r12, %%r8, 4), %%r13                        \n" // C3
		"   prefetcht1         (%%r13)                               \n"

		"    mov     %%rsi, %%rdx                                    \n" // K

		"   movss        (%%rbx), %%xmm4                             \n" // B0

		"   vpxorq         %%xmm8, %%xmm8, %%xmm8                    \n"
		"    vpxorq         %%xmm10, %%xmm10, %%xmm10                \n"
		"    vpxorq         %%xmm12, %%xmm12, %%xmm12                \n"
		"    vpxorq         %%xmm14, %%xmm14, %%xmm14                \n"
		"    vpxorq         %%xmm16, %%xmm16, %%xmm16                \n"
		"    vpxorq         %%xmm18, %%xmm18, %%xmm18                \n"

		"   movss    (%%rax), %%xmm0                                 \n" // A0
		"   movss    4(%%rax), %%xmm1                                \n" // A1

		"    vpxorq         %%xmm20, %%xmm20, %%xmm20                \n"
		"    vpxorq         %%xmm22, %%xmm22, %%xmm22                \n"
		"    vpxorq         %%xmm24, %%xmm24, %%xmm24                \n"
		"    vpxorq         %%xmm26, %%xmm26, %%xmm26                \n"
		"    vpxorq         %%xmm28, %%xmm28, %%xmm28                \n"
		"    vpxorq         %%xmm30, %%xmm30, %%xmm30                \n"

		"    subq     $8, %%rdx                                      \n"

		"K_N1_PREFETCH_C_MGN:                                        \n"
		"   leaq     (%%r13, %%r8, 4), %%r13                         \n"
		"   prefetcht2         (%%r13)                               \n"
		"   prefetcht2         64(%%r13)                             \n"

		"MAIN_K_N1_MGN:                                              \n"

		"    KERNEL12x1_K1_MGN                                       \n"
		"    KERNEL12x1_K2_MGN                                       \n"
		"    KERNEL12x1_K1_MGN                                       \n"
		"    KERNEL12x1_K2_MGN                                       \n"
		"    KERNEL12x1_K1_MGN                                       \n"
		"    KERNEL12x1_K2_MGN                                       \n"
		"    KERNEL12x1_K1_MGN                                       \n"
		"   cmp     $0, %%rdx                                        \n"
		"    je         EDGE_K_N1_MGN                                \n"
		"    KERNEL12x1_K2_MGN                                       \n"
		"   subq     $8, %%rdx                                       \n"
		"   cmp   $32, %%rdx                                         \n"
		"   jbe     K_N1_PREFETCH_C_MGN                              \n"
		"   jmp     MAIN_K_N1_MGN                                    \n"

		"EDGE_K_N1_MGN:                                              \n"
		"   leaq     (%%r12, %%r8, 4), %%r13                         \n"
		"    KERNEL12x1_END_K_MGN                                    \n"
		"    cmp     $0, %%r15                                       \n"
		"    je      SAVE_C_N1_MGN                                   \n"
		"    ADD_C_12x1_MGN                                          \n"

		"SAVE_C_N1_MGN:                                              \n"
		"    SAVE_12x1_MGN                                           \n"

		//----------------------------------------------------------------

		"END_M12N1_MGN:                                              \n"

		:
		:
		[C] "m"(C),
		[A] "m"(A),
		[B] "m"(B),
		[M] "m"(M),
		[N] "m"(N),
		[K] "m"(K),
		[LN] "m"(LN),
		[LK] "m"(LK),
		[Ac] "m"(Ac),
		[k_tag] "m"(k_tag)
		: "rax", "rbx", "rcx", "rdx", "rdi", "rsi", "rbp", "r8", "r9", "r10", "r11", "r12",
		  "r13", "r14", "r15", "zmm0", "zmm1", "zmm2", "zmm3", "zmm4", "zmm5",
		  "zmm6", "zmm7", "zmm8", "zmm9", "zmm10", "zmm11", "zmm12", "zmm13",
		  "zmm14", "zmm15", "zmm16", "zmm17", "zmm18", "zmm19", "zmm20", "zmm21",
		  "zmm22", "zmm23", "zmm24", "zmm25", "zmm26", "zmm27", "zmm28", "zmm29",
		  "zmm30", "zmm31", "memory"

	);
}

void MGN_KERNEL8x32(float *C, float *A, float *B, long M, long N, long K, long LN, long LK, float *Ac, long k_tag)
{

	asm volatile(
		".macro    KERNEL8x32_K1_MGN                                 \n"

		"    vbroadcastss    8(%%rax), %%zmm2                        \n"
		"    vfmadd231ps        %%zmm0, %%zmm4, %%zmm8               \n"
		"    vfmadd231ps        %%zmm0, %%zmm5, %%zmm9               \n"

		"    vbroadcastss    12(%%rax), %%zmm3                       \n"
		"    vfmadd231ps        %%zmm1, %%zmm4, %%zmm10              \n"
		"    vfmadd231ps        %%zmm1, %%zmm5, %%zmm11              \n"

		"    prefetcht0         256(%%rax)                           \n"

		"    vbroadcastss    16(%%rax), %%zmm0                       \n"
		"    vfmadd231ps        %%zmm2, %%zmm4, %%zmm12              \n"
		"    vfmadd231ps        %%zmm2, %%zmm5, %%zmm13              \n"

		"    vbroadcastss    20(%%rax), %%zmm1                       \n"
		"    vfmadd231ps        %%zmm3, %%zmm4, %%zmm14              \n"
		"    vfmadd231ps        %%zmm3, %%zmm5, %%zmm15              \n"

		"    vbroadcastss    24(%%rax), %%zmm2                       \n"
		"    vfmadd231ps        %%zmm0, %%zmm4, %%zmm16              \n"
		"    vfmadd231ps        %%zmm0, %%zmm5, %%zmm17              \n"

		"    vbroadcastss    28(%%rax), %%zmm3                       \n"
		"    vfmadd231ps        %%zmm1, %%zmm4, %%zmm18              \n"
		"    vfmadd231ps        %%zmm1, %%zmm5, %%zmm19              \n"

		"    addq              $32, %%rax                            \n"
		"    addq              $128, %%rbx                           \n"

		"    vbroadcastss        (%%rax), %%zmm0                     \n"
		"    vmovups         (%%rbx), %%zmm6                         \n"
		"    vfmadd231ps        %%zmm2, %%zmm4, %%zmm20              \n"
		"    vfmadd231ps        %%zmm2, %%zmm5, %%zmm21              \n"

		"    vbroadcastss        4(%%rax), %%zmm1                    \n"
		"    vmovups         64(%%rbx), %%zmm7                       \n"
		"    vfmadd231ps        %%zmm3, %%zmm4, %%zmm22              \n"
		"    vfmadd231ps        %%zmm3, %%zmm5, %%zmm23              \n"

		"    prefetcht0         256(%%rbx)                           \n"

		".endm                                                       \n"

		".macro    KERNEL8x32_K2_MGN                                 \n"

		"    vbroadcastss    8(%%rax), %%zmm2                        \n"
		"    vfmadd231ps        %%zmm0, %%zmm6, %%zmm8               \n"
		"    vfmadd231ps        %%zmm0, %%zmm7, %%zmm9               \n"

		"    vbroadcastss    12(%%rax), %%zmm3                       \n"
		"    vfmadd231ps        %%zmm1, %%zmm6, %%zmm10              \n"
		"    vfmadd231ps        %%zmm1, %%zmm7, %%zmm11              \n"

		"    prefetcht0         256(%%rax)                           \n"

		"    vbroadcastss    16(%%rax), %%zmm0                       \n"
		"    vfmadd231ps        %%zmm2, %%zmm6, %%zmm12              \n"
		"    vfmadd231ps        %%zmm2, %%zmm7, %%zmm13              \n"

		"    vbroadcastss    20(%%rax), %%zmm1                       \n"
		"    vfmadd231ps        %%zmm3, %%zmm6, %%zmm14              \n"
		"    vfmadd231ps        %%zmm3, %%zmm7, %%zmm15              \n"

		"    vbroadcastss    24(%%rax), %%zmm2                       \n"
		"    vfmadd231ps        %%zmm0, %%zmm6, %%zmm16              \n"
		"    vfmadd231ps        %%zmm0, %%zmm7, %%zmm17              \n"

		"    vbroadcastss    28(%%rax), %%zmm3                       \n"
		"    vfmadd231ps        %%zmm1, %%zmm6, %%zmm18              \n"
		"    vfmadd231ps        %%zmm1, %%zmm7, %%zmm19              \n"

		"    addq              $32, %%rax                            \n"
		"    addq              $128, %%rbx                           \n"

		"    vbroadcastss        (%%rax), %%zmm0                     \n"
		"    vmovups         (%%rbx), %%zmm4                         \n"
		"    vfmadd231ps        %%zmm2, %%zmm6, %%zmm20              \n"
		"    vfmadd231ps        %%zmm2, %%zmm7, %%zmm21              \n"

		"    vbroadcastss        4(%%rax), %%zmm1                    \n"
		"    vmovups         64(%%rbx), %%zmm5                       \n"
		"    vfmadd231ps        %%zmm3, %%zmm6, %%zmm22              \n"
		"    vfmadd231ps        %%zmm3, %%zmm7, %%zmm23              \n"

		"    prefetcht0         256(%%rbx)                           \n"

		".endm                                                       \n"

		".macro    KERNEL8x32_END_K_MGN                              \n"

		"    vbroadcastss    8(%%rax), %%zmm2                        \n"
		"    vfmadd231ps        %%zmm0, %%zmm6, %%zmm8               \n"
		"    vfmadd231ps        %%zmm0, %%zmm7, %%zmm9               \n"

		"    vbroadcastss    12(%%rax), %%zmm3                       \n"
		"    vfmadd231ps        %%zmm1, %%zmm6, %%zmm10              \n"
		"    vfmadd231ps        %%zmm1, %%zmm7, %%zmm11              \n"

		"    prefetcht0         256(%%rax)                           \n"

		"    vbroadcastss    16(%%rax), %%zmm0                       \n"
		"    vfmadd231ps        %%zmm2, %%zmm6, %%zmm12              \n"
		"    vfmadd231ps        %%zmm2, %%zmm7, %%zmm13              \n"

		"    vbroadcastss    20(%%rax), %%zmm1                       \n"
		"    vfmadd231ps        %%zmm3, %%zmm6, %%zmm14              \n"
		"    vfmadd231ps        %%zmm3, %%zmm7, %%zmm15              \n"

		"    vbroadcastss    24(%%rax), %%zmm2                       \n"
		"    vfmadd231ps        %%zmm0, %%zmm6, %%zmm16              \n"
		"    vfmadd231ps        %%zmm0, %%zmm7, %%zmm17              \n"

		"    vbroadcastss    28(%%rax), %%zmm3                       \n"
		"    vfmadd231ps        %%zmm1, %%zmm6, %%zmm18              \n"
		"    vfmadd231ps        %%zmm1, %%zmm7, %%zmm19              \n"

		"    vfmadd231ps        %%zmm2, %%zmm6, %%zmm20              \n"
		"    vfmadd231ps        %%zmm2, %%zmm7, %%zmm21              \n"

		"    vfmadd231ps        %%zmm3, %%zmm6, %%zmm22              \n"
		"    vfmadd231ps        %%zmm3, %%zmm7, %%zmm23              \n"

		"    addq              $128, %%rbx                           \n" // TODO

		".endm                                                       \n"

		".macro    ADD_C_8x32_MGN                                    \n"

		"    vmovups         (%%r10), %%zmm0                         \n"
		"    vaddps             %%zmm0, %%zmm8, %%zmm8               \n"
		"    vmovups         64(%%r10), %%zmm1                       \n"
		"    vaddps             %%zmm1, %%zmm9, %%zmm9               \n"
		"    vmovups         (%%r11), %%zmm2                         \n"
		"    vaddps             %%zmm2, %%zmm10, %%zmm10             \n"
		"    vmovups         64(%%r11), %%zmm3                       \n"
		"    vaddps             %%zmm3, %%zmm11, %%zmm11             \n"
		"    vmovups         (%%r12), %%zmm4                         \n"
		"    vaddps             %%zmm4, %%zmm12, %%zmm12             \n"
		"    vmovups         64(%%r12), %%zmm5                       \n"
		"    vaddps             %%zmm5, %%zmm13, %%zmm13             \n"
		"    vmovups         (%%r13), %%zmm6                         \n"
		"    vaddps             %%zmm6, %%zmm14, %%zmm14             \n"
		"    vmovups         64(%%r13), %%zmm7                       \n"
		"    vaddps             %%zmm7, %%zmm15, %%zmm15             \n"

		"    leaq              (%%r13, %%r8, 4), %%r10               \n" // C0
		"    leaq             (%%r10, %%r8, 4), %%r11                \n" // C1
		"    leaq             (%%r11, %%r8, 4), %%r12                \n" // C2
		"    leaq             (%%r12, %%r8, 4), %%r13                \n" // C3

		"    vmovups         (%%r10), %%zmm0                         \n"
		"    vaddps             %%zmm0, %%zmm16, %%zmm16             \n"
		"    vmovups         64(%%r10), %%zmm1                       \n"
		"    vaddps             %%zmm1, %%zmm17, %%zmm17             \n"
		"    vmovups         (%%r11), %%zmm2                         \n"
		"    vaddps             %%zmm2, %%zmm18, %%zmm18             \n"
		"    vmovups         64(%%r11), %%zmm3                       \n"
		"    vaddps             %%zmm3, %%zmm19, %%zmm19             \n"

		"    vmovups         (%%r12), %%zmm4                         \n"
		"    vaddps             %%zmm4, %%zmm20, %%zmm20             \n"
		"    vmovups         64(%%r12), %%zmm5                       \n"
		"    vaddps             %%zmm5, %%zmm21, %%zmm21             \n"
		"    vmovups         (%%r13), %%zmm6                         \n"
		"    vaddps             %%zmm6, %%zmm22, %%zmm22             \n"
		"    vmovups         64(%%r13), %%zmm7                       \n"
		"    vaddps             %%zmm7, %%zmm23, %%zmm23             \n"

		"    mov      %%rcx, %%r10                                   \n" // C0
		"    leaq     (%%r10, %%r8, 4), %%r11                        \n" // C1
		"    leaq     (%%r11, %%r8, 4), %%r12                        \n" // C2
		"    leaq     (%%r12, %%r8, 4), %%r13                        \n" // C3

		".endm                                                       \n"

		".macro    SAVE_8x32_MGN                                     \n"

		"    vmovups         %%zmm8, (%%r10)                         \n"
		"    vmovups         %%zmm9, 64(%%r10)                       \n"
		"    vmovups         %%zmm10, (%%r11)                        \n"
		"    vmovups         %%zmm11, 64(%%r11)                      \n"
		"    vmovups         %%zmm12, (%%r12)                        \n"
		"    vmovups         %%zmm13, 64(%%r12)                      \n"
		"    vmovups         %%zmm14, (%%r13)                        \n"
		"    vmovups         %%zmm15, 64(%%r13)                      \n"

		"    leaq             (%%r13, %%r8, 4), %%r10                \n" // C0
		"    leaq         (%%r10, %%r8, 4), %%r11                    \n" // C1
		"    leaq         (%%r11, %%r8, 4), %%r12                    \n" // C2
		"    leaq         (%%r12, %%r8, 4), %%r13                    \n" // C3

		"    vmovups         %%zmm16, (%%r10)                        \n"
		"    vmovups         %%zmm17, 64(%%r10)                      \n"
		"    vmovups         %%zmm18, (%%r11)                        \n"
		"    vmovups         %%zmm19, 64(%%r11)                      \n"
		"    vmovups         %%zmm20, (%%r12)                        \n"
		"    vmovups         %%zmm21, 64(%%r12)                      \n"
		"    vmovups         %%zmm22, (%%r13)                        \n"
		"    vmovups         %%zmm23, 64(%%r13)                      \n"

		"    subq         $32, %%rdi                                 \n" // TODO
		"    addq             $128, %%rcx                            \n" // C0

		".endm                                                       \n"

		//-----------------------------------------------------------------

		".macro    KERNEL8x16_K1_MGN                                 \n"

		"    vbroadcastss    8(%%rax), %%zmm2                        \n"
		"    vfmadd231ps        %%zmm0, %%zmm4, %%zmm8               \n"

		"    vbroadcastss    12(%%rax), %%zmm3                       \n"
		"    vfmadd231ps        %%zmm1, %%zmm4, %%zmm10              \n"

		"    prefetcht0         256(%%rax)                           \n"

		"    vbroadcastss    16(%%rax), %%zmm0                       \n"
		"    vfmadd231ps        %%zmm2, %%zmm4, %%zmm12              \n"

		"    vbroadcastss    20(%%rax), %%zmm1                       \n"
		"    vfmadd231ps        %%zmm3, %%zmm4, %%zmm14              \n"

		"    vbroadcastss    24(%%rax), %%zmm2                       \n"
		"    vfmadd231ps        %%zmm0, %%zmm4, %%zmm16              \n"

		"    vbroadcastss    28(%%rax), %%zmm3                       \n"
		"    vfmadd231ps        %%zmm1, %%zmm4, %%zmm18              \n"

		"    addq              $32, %%rax                            \n"
		"    addq              $64, %%rbx                            \n"

		"    vbroadcastss    (%%rax), %%zmm0                         \n"
		"    vmovups            (%%rbx), %%zmm6                      \n"
		"    vfmadd231ps        %%zmm2, %%zmm4, %%zmm20              \n"

		"    vbroadcastss    4(%%rax), %%zmm1                        \n"
		"    vfmadd231ps        %%zmm3, %%zmm4, %%zmm22              \n"

		"    prefetcht0         256(%%rbx)                           \n"

		".endm                                                       \n"

		".macro    KERNEL8x16_K2_MGN                                 \n"

		"    vbroadcastss    8(%%rax), %%zmm2                        \n"
		"    vfmadd231ps        %%zmm0, %%zmm6, %%zmm8               \n"

		"    vbroadcastss    12(%%rax), %%zmm3                       \n"
		"    vfmadd231ps        %%zmm1, %%zmm6, %%zmm10              \n"

		"    prefetcht0         256(%%rax)                           \n"

		"    vbroadcastss    16(%%rax), %%zmm0                       \n"
		"    vfmadd231ps        %%zmm2, %%zmm6, %%zmm12              \n"

		"    vbroadcastss    20(%%rax), %%zmm1                       \n"
		"    vfmadd231ps        %%zmm3, %%zmm6, %%zmm14              \n"

		"    vbroadcastss    24(%%rax), %%zmm2                       \n"
		"    vfmadd231ps        %%zmm0, %%zmm6, %%zmm16              \n"

		"    vbroadcastss    28(%%rax), %%zmm3                       \n"
		"    vfmadd231ps        %%zmm1, %%zmm6, %%zmm18              \n"

		"    addq              $32, %%rax                            \n"
		"    addq              $64, %%rbx                            \n"

		"    vbroadcastss    (%%rax), %%zmm0                         \n"
		"    vmovups         (%%rbx), %%zmm4                         \n"
		"    vfmadd231ps        %%zmm2, %%zmm6, %%zmm20              \n"

		"    vbroadcastss    4(%%rax), %%zmm1                        \n"
		"    vfmadd231ps        %%zmm3, %%zmm6, %%zmm22              \n"

		"    prefetcht0         256(%%rbx)                           \n"

		".endm                                                       \n"

		".macro    KERNEL8x16_END_K_MGN                              \n"

		"    vbroadcastss    8(%%rax), %%zmm2                        \n"
		"    vfmadd231ps        %%zmm0, %%zmm6, %%zmm8               \n"

		"    vbroadcastss    12(%%rax), %%zmm3                       \n"
		"    vfmadd231ps        %%zmm1, %%zmm6, %%zmm10              \n"

		"    prefetcht0         256(%%rax)                           \n"

		"    vbroadcastss    16(%%rax), %%zmm0                       \n"
		"    vfmadd231ps        %%zmm2, %%zmm6, %%zmm12              \n"

		"    vbroadcastss    20(%%rax), %%zmm1                       \n"
		"    vfmadd231ps        %%zmm3, %%zmm6, %%zmm14              \n"

		"    vbroadcastss    24(%%rax), %%zmm2                       \n"
		"    vfmadd231ps        %%zmm0, %%zmm6, %%zmm16              \n"

		"    vbroadcastss    28(%%rax), %%zmm3                       \n"
		"    vfmadd231ps        %%zmm1, %%zmm6, %%zmm18              \n"

		"    vfmadd231ps        %%zmm2, %%zmm6, %%zmm20              \n"
		"    vfmadd231ps        %%zmm3, %%zmm6, %%zmm22              \n"

		"    addq              $64, %%rbx                            \n" // TODO

		".endm                                                       \n"

		".macro    ADD_C_8x16_MGN                                    \n"

		"    vmovups     (%%r10), %%zmm0                             \n"
		"    vaddps         %%zmm0, %%zmm8, %%zmm8                   \n"
		"    vmovups     (%%r11), %%zmm2                             \n"
		"    vaddps         %%zmm2, %%zmm10, %%zmm10                 \n"
		"    vmovups     (%%r12), %%zmm4                             \n"
		"    vaddps         %%zmm4, %%zmm12, %%zmm12                 \n"
		"    vmovups     (%%r13), %%zmm6                             \n"
		"    vaddps         %%zmm6, %%zmm14, %%zmm14                 \n"

		"    leaq          (%%r13, %%r8, 4), %%r10                   \n" // C0
		"    leaq         (%%r10, %%r8, 4), %%r11                    \n" // C1
		"    leaq         (%%r11, %%r8, 4), %%r12                    \n" // C2
		"    leaq         (%%r12, %%r8, 4), %%r13                    \n" // C3

		"    vmovups     (%%r10), %%zmm0                             \n"
		"    vaddps         %%zmm0, %%zmm16, %%zmm16                 \n"
		"    vmovups     (%%r11), %%zmm2                             \n"
		"    vaddps         %%zmm2, %%zmm18, %%zmm18                 \n"
		"    vmovups     (%%r12), %%zmm4                             \n"
		"    vaddps         %%zmm4, %%zmm20, %%zmm20                 \n"
		"    vmovups     (%%r13), %%zmm6                             \n"
		"    vaddps         %%zmm6, %%zmm22, %%zmm22                 \n"

		"    mov            %%rcx, %%r10                             \n" // C0
		"    leaq        (%%r10, %%r8, 4), %%r11                     \n" // C1
		"    leaq        (%%r11, %%r8, 4), %%r12                     \n" // C2
		"    leaq        (%%r12, %%r8, 4), %%r13                     \n" // C3

		".endm                                                       \n"

		".macro    SAVE_8x16_MGN                                     \n"

		"    vmovups         %%zmm8, (%%r10)                         \n"
		"    vmovups         %%zmm10, (%%r11)                        \n"
		"    vmovups         %%zmm12, (%%r12)                        \n"
		"    vmovups         %%zmm14, (%%r13)                        \n"

		"    leaq                 (%%r13, %%r8, 4), %%r10            \n" // C0
		"    leaq               (%%r10, %%r8, 4), %%r11              \n" // C1
		"    leaq               (%%r11, %%r8, 4), %%r12              \n" // C2
		"    leaq               (%%r12, %%r8, 4), %%r13              \n" // C3

		"    vmovups         %%zmm16, (%%r10)                        \n"
		"    vmovups         %%zmm18, (%%r11)                        \n"
		"    vmovups         %%zmm20, (%%r12)                        \n"
		"    vmovups         %%zmm22, (%%r13)                        \n"

		"    subq             $16, %%rdi                             \n"
		"    addq                 $64, %%rcx                         \n" // C0

		".endm                                                       \n"

		//-----------------------------------------------------------------

		".macro    KERNEL8x1_K1_MGN                                  \n"

		"    movss            8(%%rax), %%xmm2                       \n"
		"    vfmadd231ps        %%xmm0, %%xmm4, %%xmm8               \n"

		"    movss            12(%%rax), %%xmm3                      \n"
		"    vfmadd231ps        %%xmm1, %%xmm4, %%xmm10              \n"

		"    prefetcht0         256(%%rax)                           \n"

		"    movss            16(%%rax), %%xmm0                      \n"
		"    vfmadd231ps        %%xmm2, %%xmm4, %%xmm12              \n"

		"    movss            20(%%rax), %%xmm1                      \n"
		"    vfmadd231ps        %%xmm3, %%xmm4, %%xmm14              \n"

		"    movss            24(%%rax), %%xmm2                      \n"
		"    vfmadd231ps        %%xmm0, %%xmm4, %%xmm16              \n"

		"    movss            28(%%rax), %%xmm3                      \n"
		"    vfmadd231ps        %%xmm1, %%xmm4, %%xmm18              \n"

		"    addq              $32, %%rax                            \n"
		"    addq              $4, %%rbx                             \n" // TODO

		"    movss            (%%rax), %%xmm0                        \n"
		"    movss                (%%rbx), %%xmm6                    \n"
		"    vfmadd231ps        %%xmm2, %%xmm4, %%xmm20              \n"

		"    movss            4(%%rax), %%xmm1                       \n"
		"    vfmadd231ps        %%xmm3, %%xmm4, %%xmm22              \n"

		"    prefetcht0         256(%%rbx)                           \n"

		".endm                                                       \n"

		".macro    KERNEL8x1_K2_MGN                                  \n"

		"    movss            8(%%rax), %%xmm2                       \n"
		"    vfmadd231ps        %%xmm0, %%xmm6, %%xmm8               \n"

		"    movss            12(%%rax), %%xmm3                      \n"
		"    vfmadd231ps        %%xmm1, %%xmm6, %%xmm10              \n"

		"    prefetcht0         256(%%rax)                           \n"

		"    movss            16(%%rax), %%xmm0                      \n"
		"    vfmadd231ps        %%xmm2, %%xmm6, %%xmm12              \n"

		"    movss            20(%%rax), %%xmm1                      \n"
		"    vfmadd231ps        %%xmm3, %%xmm6, %%xmm14              \n"

		"    movss            24(%%rax), %%xmm2                      \n"
		"    vfmadd231ps        %%xmm0, %%xmm6, %%xmm16              \n"

		"    movss            28(%%rax), %%xmm3                      \n"
		"    vfmadd231ps        %%xmm1, %%xmm6, %%xmm18              \n"

		"    addq              $32, %%rax                            \n"
		"    addq              $4, %%rbx                             \n" // TODO

		"    movss            (%%rax), %%xmm0                        \n"
		"    movss             (%%rbx), %%xmm4                       \n"
		"    vfmadd231ps        %%xmm2, %%xmm6, %%xmm20              \n"

		"    movss            4(%%rax), %%xmm1                       \n"
		"    vfmadd231ps        %%xmm3, %%xmm6, %%xmm22              \n"

		"    prefetcht0         256(%%rbx)                           \n"

		".endm                                                       \n"

		".macro    KERNEL8x1_END_K_MGN                               \n"

		"    movss            8(%%rax), %%xmm2                       \n"
		"    vfmadd231ps        %%xmm0, %%xmm6, %%xmm8               \n"

		"    movss            12(%%rax), %%xmm3                      \n"
		"    vfmadd231ps        %%xmm1, %%xmm6, %%xmm10              \n"

		"    prefetcht0         256(%%rax)                           \n"

		"    movss            16(%%rax), %%xmm0                      \n"
		"    vfmadd231ps        %%xmm2, %%xmm6, %%xmm12              \n"

		"    movss            20(%%rax), %%xmm1                      \n"
		"    vfmadd231ps        %%xmm3, %%xmm6, %%xmm14              \n"

		"    movss            24(%%rax), %%xmm2                      \n"
		"    vfmadd231ps        %%xmm0, %%xmm6, %%xmm16              \n"

		"    movss            28(%%rax), %%xmm3                      \n"
		"    vfmadd231ps        %%xmm1, %%xmm6, %%xmm18              \n"

		"    vfmadd231ps        %%xmm2, %%xmm6, %%xmm20              \n"
		"    vfmadd231ps        %%xmm3, %%xmm6, %%xmm22              \n"

		"    addq              $4, %%rbx                             \n" // TODO

		".endm                                                       \n"

		".macro    ADD_C_8x1_MGN                                     \n"

		"    movss             (%%r10), %%xmm0                       \n"
		"    vaddps             %%xmm0, %%xmm8, %%xmm8               \n"
		"    movss             (%%r11), %%xmm2                       \n"
		"    vaddps             %%xmm2, %%xmm10, %%xmm10             \n"
		"    movss             (%%r12), %%xmm4                       \n"
		"    vaddps             %%xmm4, %%xmm12, %%xmm12             \n"
		"    movss             (%%r13), %%xmm6                       \n"
		"    vaddps             %%xmm6, %%xmm14, %%xmm14             \n"

		"    leaq              (%%r13, %%r8, 4), %%r10               \n" // C0
		"    leaq             (%%r10, %%r8, 4), %%r11                \n" // C1
		"    leaq             (%%r11, %%r8, 4), %%r12                \n" // C2
		"    leaq             (%%r12, %%r8, 4), %%r13                \n" // C3

		"    movss             (%%r10), %%xmm0                       \n"
		"    vaddps             %%xmm0, %%xmm16, %%xmm16             \n"
		"    movss             (%%r11), %%xmm2                       \n"
		"    vaddps             %%xmm2, %%xmm18, %%xmm18             \n"
		"    movss             (%%r12), %%xmm4                       \n"
		"    vaddps             %%xmm4, %%xmm20, %%xmm20             \n"
		"    movss             (%%r13), %%xmm6                       \n"
		"    vaddps             %%xmm6, %%xmm22, %%xmm22             \n"

		"    mov             %%rcx, %%r10                            \n" // C0
		"    leaq               (%%r10, %%r8, 4), %%r11              \n" // C1
		"    leaq               (%%r11, %%r8, 4), %%r12              \n" // C2
		"    leaq             (%%r12, %%r8, 4), %%r13                \n" // C3

		".endm                                                       \n"

		".macro    SAVE_8x1_MGN                                      \n"

		"    vmovss         %%xmm8, (%%r10)                          \n"
		"    vmovss         %%xmm10, (%%r11)                         \n"
		"    vmovss         %%xmm12, (%%r12)                         \n"
		"    vmovss         %%xmm14, (%%r13)                         \n"

		"    leaq         (%%r13, %%r8, 4), %%r10                    \n" // C0
		"    leaq        (%%r10, %%r8, 4), %%r11                     \n" // C1
		"    leaq        (%%r11, %%r8, 4), %%r12                     \n" // C2
		"    leaq        (%%r12, %%r8, 4), %%r13                     \n" // C3

		"    vmovss         %%xmm16, (%%r10)                         \n"
		"    vmovss         %%xmm18, (%%r11)                         \n"
		"    vmovss         %%xmm20, (%%r12)                         \n"
		"    vmovss         %%xmm22, (%%r13)                         \n"

		"    subq         $1, %%rdi                                  \n"
		"    addq             $4, %%rcx                              \n"

		".endm                                                       \n"

		//-----------------------------------------------------------------
		// pack A
		"    movl %[LK], %%r8d                                       \n"
		"    movl %[LK], %%r15d                                      \n"

		"    mov %[Ac], %%r9                                         \n"
		"    movl %[K], %%r10d                                       \n"
		"    mov %[A], %%r11                                         \n"
		"    mov %[A], %%r12                                         \n"
		"    mov %[A], %%r13                                         \n"
		"    mov %[A], %%r14                                         \n"

		"    mov %[A], %%rbx                                         \n"
		"    mov %[A], %%rcx                                         \n"
		"    mov %[A], %%rdx                                         \n"
		"    mov %[A], %%rdi                                         \n"

		"    shr $4, %%r10                                           \n"
		"    shl $2, %%r8                                            \n"
		"    shl $3, %%r15                                           \n"
		"    add %%r8, %%r14                                         \n"
		"    add %%r8, %%rdi                                         \n"
		"    add %%r8, %%r12                                         \n"
		"    add %%r8, %%rcx                                         \n"
		"    add %%r15, %%r13                                        \n"
		"    add %%r15, %%rdx                                        \n"
		"    shl $2, %%r8                                            \n"
		"    add %%r15, %%r14                                        \n"
		"    add %%r15, %%rdi                                        \n"
		"    add %%r8, %%rbx                                         \n"
		"    add %%r8, %%rcx                                         \n"
		"    add %%r8, %%rdx                                         \n"
		"    add %%r8, %%rdi                                         \n"

		"    shl $1, %%r15                                           \n"

		"    vmovups (%%r11), %%zmm2                                 \n"
		"    vmovups (%%r12), %%zmm3                                 \n"
		"    vmovups (%%r13), %%zmm4                                 \n"
		"    vmovups (%%r14), %%zmm5                                 \n"

		"    sub $64, %%r15                                          \n"

		"    vmovups (%%rbx), %%zmm6                                 \n"
		"    vmovups (%%rcx), %%zmm7                                 \n"
		"    vmovups (%%rdx), %%zmm8                                 \n"
		"    vmovups (%%rdi), %%zmm9                                 \n"

		"    mov %%r15, %%rsi                                        \n"

		"    jmp NPACK_M8_MGN                                        \n"
		"    movl %[LK], %%r8d                                       \n"
		"    movl %[LK], %%r15d                                      \n"

		"    mov %[Ac], %%r9                                         \n"
		"    movl %[K], %%r10d                                       \n"
		"    mov %[A], %%r11                                         \n"
		"    mov %[A], %%r12                                         \n"
		"    mov %[A], %%r13                                         \n"
		"    mov %[A], %%r14                                         \n"

		"    mov %[A], %%rbx                                         \n"
		"    mov %[A], %%rcx                                         \n"
		"    mov %[A], %%rdx                                         \n"
		"    mov %[A], %%rdi                                         \n"

		"    shr $4, %%r10                                           \n"
		"    shl $2, %%r8                                            \n"
		"    shl $3, %%r15                                           \n"
		"    add %%r8, %%r14                                         \n"
		"    add %%r8, %%rdi                                         \n"
		"    add %%r8, %%r12                                         \n"
		"    add %%r8, %%rcx                                         \n"
		"    add %%r15, %%r13                                        \n"
		"    add %%r15, %%rdx                                        \n"
		"    shl $2, %%r8                                            \n"
		"    add %%r15, %%r14                                        \n"
		"    add %%r15, %%rdi                                        \n"
		"    add %%r8, %%rbx                                         \n"
		"    add %%r8, %%rcx                                         \n"
		"    add %%r8, %%rdx                                         \n"
		"    add %%r8, %%rdi                                         \n"

		"    vmovups (%%r11), %%zmm2                                 \n"
		"    vmovups (%%r12), %%zmm3                                 \n"
		"    vmovups (%%r13), %%zmm4                                 \n"
		"    vmovups (%%r14), %%zmm5                                 \n"
		"    vmovups (%%rbx), %%zmm6                                 \n"
		"    vmovups (%%rcx), %%zmm7                                 \n"
		"    vmovups (%%rdx), %%zmm8                                 \n"
		"    vmovups (%%rdi), %%zmm9                                 \n"
		"    jmp NPACK_M8_MGN                                        \n"

		"NPACK_Pre_M8_MGN:                                           \n"
		"   add $64, %%r11                                           \n"
		"   add $64, %%r12                                           \n"
		"   add $64, %%r13                                           \n"
		"   add $64, %%r14                                           \n"
		"   add $64, %%rbx                                           \n"
		"   add $64, %%rcx                                           \n"
		"   add $64, %%rdx                                           \n"
		"   add $64, %%rdi                                           \n"
		"   add $512, %%r9                                           \n"
		"   vmovups (%%r11), %%zmm2                                  \n"
		"   vmovups (%%r12), %%zmm3                                  \n"
		"   vmovups (%%r13), %%zmm4                                  \n"
		"   vmovups (%%r14), %%zmm5                                  \n"
		"   vmovups (%%rbx), %%zmm6                                  \n"
		"   vmovups (%%rcx), %%zmm7                                  \n"
		"   vmovups (%%rdx), %%zmm8                                  \n"
		"   vmovups (%%rdi), %%zmm9                                  \n"

		"NPACK_M8_MGN:                                               \n"

		"   movl    $0xaa, %%eax                                     \n"
		"   movl    $0xcc, %%r8d                                     \n"
		"   movl    $0x33, %%r15d                                    \n"

		"   kmovd   %%eax, %%k1                                      \n"
		"   kmovd   %%r8d, %%k2                                      \n"
		"   kmovd   %%r15d, %%k3                                     \n"

		"   vunpcklps %%zmm3, %%zmm2, %%zmm0                         \n"
		"   vunpcklps %%zmm7, %%zmm6, %%zmm1                         \n"

		"   vunpcklps %%zmm5, %%zmm4, %%zmm26                        \n"
		"   vunpcklps %%zmm9, %%zmm8, %%zmm27                        \n"

		"   vpermq  $0x80, %%zmm26, %%zmm0%{%%k1%}                   \n"
		"   vmovups %%zmm0, %%zmm28                                  \n"
		"   vpermq  $0x80, %%zmm27, %%zmm1%{%%k1%}                   \n"
		"   vmovups %%zmm1, %%zmm29                                  \n"

		"   vpermq  $0x40, %%zmm1, %%zmm0%{%%k2%}                    \n"
		"   vpermq  $0x0e, %%zmm28, %%zmm29%{%%k3%}                  \n"

		"   vmovups %%ymm0, (%%r9)                                   \n"
		"   vextractf64x4 $0x1, %%zmm0,  %%ymm30                     \n"
		"   vmovups %%ymm30, 256(%%r9)                               \n"

		"   vmovups %%ymm29, 128(%%r9)                               \n"
		"   vextractf64x4 $0x1, %%zmm29,  %%ymm31                    \n"
		"   vmovups %%ymm31, 384(%%r9)                               \n"

		"   movl    $0x55, %%eax                                     \n"
		"   movl    $0xcc, %%r8d                                     \n"
		"   movl    $0x33, %%r15d                                    \n"

		"   kmovd   %%eax, %%k1                                      \n"
		"   kmovd   %%r8d, %%k2                                      \n"
		"   kmovd   %%r15d, %%k3                                     \n"

		"   vunpcklps %%zmm3, %%zmm2, %%zmm0                         \n"
		"   vunpcklps %%zmm7, %%zmm6, %%zmm1                         \n"

		"   vunpcklps %%zmm5, %%zmm4, %%zmm26                        \n"
		"   vunpcklps %%zmm9, %%zmm8, %%zmm27                        \n"

		"   vpermq  $0x31, %%zmm0, %%zmm26%{%%k1%}                   \n"
		"   vmovups %%zmm26, %%zmm28                                 \n"
		"   vpermq  $0x31, %%zmm1, %%zmm27%{%%k1%}                   \n"
		"   vmovups %%zmm27, %%zmm29                                 \n"

		"   vpermq  $0x40, %%zmm27, %%zmm26%{%%k2%}                  \n"
		"   vpermq  $0x0e, %%zmm28, %%zmm29%{%%k3%}                  \n"

		"   vmovups %%ymm26, 32(%%r9)                                \n"
		"   vextractf64x4  $0x1,%%zmm26, %%ymm30                     \n"
		"   vmovups %%ymm30, 288(%%r9)                               \n"

		"   vmovups %%ymm29, 160(%%r9)                               \n"
		"   vextractf64x4 $0x1, %%zmm29,  %%ymm31                    \n"
		"   vmovups %%ymm31, 416(%%r9)                               \n"

		"   movl    $0xaa, %%eax                                     \n"
		"   movl    $0xcc, %%r8d                                     \n"
		"   movl    $0x33, %%r15d                                    \n"

		"   kmovd   %%eax, %%k1                                      \n"
		"   kmovd   %%r8d, %%k2                                      \n"
		"   kmovd   %%r15d, %%k3                                     \n"

		"   vunpckhps %%zmm3, %%zmm2, %%zmm0                         \n"
		"   vunpckhps %%zmm7, %%zmm6, %%zmm1                         \n"

		"   vunpckhps %%zmm5, %%zmm4, %%zmm26                        \n"
		"   vunpckhps %%zmm9, %%zmm8, %%zmm27                        \n"

		"   vpermq   $0x80, %%zmm26, %%zmm0%{%%k1%}                  \n"
		"   vmovups %%zmm0, %%zmm28                                  \n"
		"   vpermq   $0x80, %%zmm27, %%zmm1%{%%k1%}                  \n"
		"   vmovups %%zmm1, %%zmm29                                  \n"

		"   vpermq  $0x40, %%zmm1, %%zmm0%{%%k2%}                    \n"
		"   vpermq  $0x0e, %%zmm28, %%zmm29%{%%k3%}                  \n"

		"   vmovups %%ymm0, 64(%%r9)                                 \n"
		"   vextractf64x4 $0x1, %%zmm0,  %%ymm30                     \n"
		"   vmovups %%ymm30, 320(%%r9)                               \n"

		"   vmovups %%ymm29, 192(%%r9)                               \n"
		"   vextractf64x4 $0x1, %%zmm29,  %%ymm31                    \n"
		"   vmovups %%ymm31, 448(%%r9)                               \n"

		"   movl    $0x55, %%eax                                     \n"
		"   movl    $0xcc, %%r8d                                     \n"
		"   movl    $0x33, %%r15d                                    \n"

		"   kmovd   %%eax, %%k1                                      \n"
		"   kmovd   %%r8d, %%k2                                      \n"
		"   kmovd   %%r15d, %%k3                                     \n"

		"   vunpckhps %%zmm3, %%zmm2, %%zmm0                         \n"
		"   vunpckhps %%zmm7, %%zmm6, %%zmm1                         \n"

		"   vunpckhps %%zmm5, %%zmm4, %%zmm26                        \n"
		"   vunpckhps %%zmm9, %%zmm8, %%zmm27                        \n"

		"   vpermq  $0x31, %%zmm0, %%zmm26%{%%k1%}                   \n"
		"   vmovups %%zmm26, %%zmm28                                 \n"
		"   vpermq  $0x31, %%zmm1, %%zmm27%{%%k1%}                   \n"
		"   vmovups %%zmm27, %%zmm29                                 \n"

		"   vpermq  $0x40, %%zmm27, %%zmm26%{%%k2%}                  \n"
		"   vpermq  $0x0e, %%zmm28, %%zmm29%{%%k3%}                  \n"

		"   vmovups %%ymm26, 96(%%r9)                                \n"
		"   vextractf64x4  $0x1,%%zmm26, %%ymm30                     \n"
		"   vmovups %%ymm30, 352(%%r9)                               \n"

		"   sub    $1, %%r10                                         \n"

		"   vmovups %%ymm29, 224(%%r9)                               \n"
		"   vextractf64x4 $0x1, %%zmm29,  %%ymm31                    \n"
		"   vmovups %%ymm31, 480(%%r9)                               \n"

		"   je  NPACK_END_M8_MGN                                     \n"
		"   jmp NPACK_Pre_M8_MGN                                     \n"

		"NPACK_END_M8_MGN:                                           \n"

		//-----------------------------------------------------------------

		"SMM_KERNEL8x32_MGN:                                         \n"

		"   mov     %[C], %%rcx                                      \n"
		"   mov     %[Ac], %%rax                                     \n"
		"   mov     %[B], %%rbx                                      \n"

		"   prefetcht0         (%%rax)                               \n"

		"    mov     %[K], %%rdx                                     \n" // K(kc)
		"    mov      %[LN], %%r8                                    \n"
		"   mov      %[Ac], %%r14                                    \n" // 存储Ac地址
		"    movq  %[N], %%rdi                                       \n"
		"    mov     %[k_tag], %%r15                                 \n" // kk=0把C存回内存, 否则加回对应的C位置

		"   prefetcht0         (%%rbx)                               \n"

		"    mov     %%rdx, %%rsi                                    \n" // K

		//-----------------------------------------------------------------

		"BEGIN_M8N32_MGN:                                            \n"
		"       cmpq     $32, %%rdi                                  \n" // N % 32
		"    jb          BEGIN_M8N16_MGN                             \n"

		"    mov     %%r14, %%rax                                    \n" // Ac
		"       prefetcht0         (%%rax)                           \n"

		"    mov      %%rcx, %%r10                                   \n" // C0
		"       prefetcht1         (%%r10)                           \n"
		"    leaq     (%%r10, %%r8, 4), %%r11                        \n" // C1
		"       prefetcht1         (%%r11)                           \n"
		"    leaq     (%%r11, %%r8, 4), %%r12                        \n" // C2
		"       prefetcht1         (%%r12)                           \n"
		"    leaq     (%%r12, %%r8, 4), %%r13                        \n" // C3
		"       prefetcht1         (%%r13)                           \n"

		"    mov     %%rsi, %%rdx                                    \n" // K

		"       vmovups        (%%rbx), %%zmm4                       \n" // B0-15
		"    vpxorq         %%zmm8, %%zmm8, %%zmm8                   \n"
		"    vpxorq         %%zmm9, %%zmm9, %%zmm9                   \n"
		"    vpxorq         %%zmm10, %%zmm10, %%zmm10                \n"
		"    vpxorq         %%zmm11, %%zmm11, %%zmm11                \n"
		"       vmovups     64(%%rbx), %%zmm5                        \n"
		"    vpxorq         %%zmm12, %%zmm12, %%zmm12                \n"
		"    vpxorq         %%zmm13, %%zmm13, %%zmm13                \n"
		"    vpxorq         %%zmm14, %%zmm14, %%zmm14                \n"
		"    vpxorq         %%zmm15, %%zmm15, %%zmm15                \n"

		"    vbroadcastss    (%%rax), %%zmm0                         \n" // A0
		"       vbroadcastss    4(%%rax), %%zmm1                     \n" // A1

		"    vpxorq         %%zmm16, %%zmm16, %%zmm16                \n"
		"    vpxorq         %%zmm17, %%zmm17, %%zmm17                \n"
		"    vpxorq         %%zmm18, %%zmm18, %%zmm18                \n"
		"    vpxorq         %%zmm19, %%zmm19, %%zmm19                \n"
		"    vpxorq         %%zmm20, %%zmm20, %%zmm20                \n"
		"    vpxorq         %%zmm21, %%zmm21, %%zmm21                \n"
		"    vpxorq         %%zmm22, %%zmm22, %%zmm22                \n"
		"    vpxorq         %%zmm23, %%zmm23, %%zmm23                \n"

		"    subq     $8, %%rdx                                      \n"

		"K_M8N32_PREFETCH_C_MGN:                                     \n"
		"       leaq     (%%r13, %%r8, 4), %%r13                     \n"
		"       prefetcht2         (%%r13)                           \n"
		"       prefetcht2         64(%%r13)                         \n"

		"MAIN_K_M8N32_MGN:                                           \n"

		"    KERNEL8x32_K1_MGN                                       \n"
		"    KERNEL8x32_K2_MGN                                       \n"
		"    KERNEL8x32_K1_MGN                                       \n"
		"    KERNEL8x32_K2_MGN                                       \n"
		"    KERNEL8x32_K1_MGN                                       \n"
		"    KERNEL8x32_K2_MGN                                       \n"
		"    KERNEL8x32_K1_MGN                                       \n"
		"       cmp     $0, %%rdx                                    \n"
		"    je         EDGE_K_M8N32_MGN                             \n"
		"    KERNEL8x32_K2_MGN                                       \n"
		"       subq     $8, %%rdx                                   \n"
		"       cmp       $64, %%rdx                                 \n"
		"       jbe     K_M8N32_PREFETCH_C_MGN                       \n"
		"       jmp     MAIN_K_M8N32_MGN                             \n"

		"EDGE_K_M8N32_MGN:                                           \n"
		"       leaq     (%%r12, %%r8, 4), %%r13                     \n"
		"    KERNEL8x32_END_K_MGN                                    \n"

		"BEGIN_SAVE_M8N32_MGN:                                       \n"
		"    cmp     $0, %%r15                                       \n"
		"    je      SAVE_C_M8N32_MGN                                \n"
		"    ADD_C_8x32_MGN                                          \n"

		"SAVE_C_M8N32_MGN:                                           \n"
		"    SAVE_8x32_MGN                                           \n"
		"    cmpq      $32, %%rdi                                    \n"
		"    jnb     BEGIN_M8N32_MGN                                 \n" // 不小于（或等于）则跳转

		//----------------------------------------------------------------

		"BEGIN_M8N16_MGN:                                            \n"
		"       cmpq     $16, %%rdi                                  \n" // N % 16
		"    jb          BEGIN_M8N1_MGN                              \n" // TODO

		"    mov     %%r14, %%rax                                    \n" // Ac
		"       prefetcht0         (%%rax)                           \n"

		"    mov      %%rcx, %%r10                                   \n" // C0
		"       prefetcht1         (%%r10)                           \n"
		"    leaq     (%%r10, %%r8, 4), %%r11                        \n" // C1
		"       prefetcht1         (%%r11)                           \n"
		"    leaq     (%%r11, %%r8, 4), %%r12                        \n" // C2
		"       prefetcht1         (%%r12)                           \n"
		"    leaq     (%%r12, %%r8, 4), %%r13                        \n" // C3
		"       prefetcht1         (%%r13)                           \n"

		"    mov     %%rsi, %%rdx                                    \n" // K

		"       vmovups            (%%rbx), %%zmm4                   \n" // B0-15
		"       vbroadcastss    (%%rax), %%zmm0                      \n" // A0
		"    vbroadcastss    4(%%rax), %%zmm1                        \n" // A1

		"       vpxorq         %%zmm8, %%zmm8, %%zmm8                \n"
		"    vpxorq         %%zmm10, %%zmm10, %%zmm10                \n"
		"    vpxorq         %%zmm12, %%zmm12, %%zmm12                \n"
		"    vpxorq         %%zmm14, %%zmm14, %%zmm14                \n"
		"    vpxorq         %%zmm16, %%zmm16, %%zmm16                \n"
		"    vpxorq         %%zmm18, %%zmm18, %%zmm18                \n"
		"    vpxorq         %%zmm20, %%zmm20, %%zmm20                \n"
		"    vpxorq         %%zmm22, %%zmm22, %%zmm22                \n"

		"    subq     $8, %%rdx                                      \n"

		"K_M8N16_PREFETCH_C_MGN:                                     \n"
		"   leaq     (%%r13, %%r8, 4), %%r13                         \n"
		"   prefetcht2         (%%r13)                               \n"
		"   prefetcht2         64(%%r13)                             \n"

		"MAIN_K_M8N16_MGN:                                           \n"

		"    KERNEL8x16_K1_MGN                                       \n"
		"    KERNEL8x16_K2_MGN                                       \n"
		"    KERNEL8x16_K1_MGN                                       \n"
		"    KERNEL8x16_K2_MGN                                       \n"
		"    KERNEL8x16_K1_MGN                                       \n"
		"    KERNEL8x16_K2_MGN                                       \n"
		"    KERNEL8x16_K1_MGN                                       \n"
		"   cmp     $0, %%rdx                                        \n"
		"    je         EDGE_K_M8N16_MGN                             \n"
		"    KERNEL8x16_K2_MGN                                       \n"
		"   subq     $8, %%rdx                                       \n"
		"   cmp   $32, %%rdx                                         \n"
		"   jbe     K_M8N16_PREFETCH_C_MGN                           \n"
		"   jmp     MAIN_K_M8N16_MGN                                 \n"

		"EDGE_K_M8N16_MGN:                                           \n"
		"   leaq     (%%r12, %%r8, 4), %%r13                         \n"
		"    KERNEL8x16_END_K_MGN                                    \n"
		"    cmp     $0, %%r15                                       \n"
		"    je      SAVE_C_M8N16_MGN                                \n"
		"    ADD_C_8x16_MGN                                          \n"

		"SAVE_C_M8N16_MGN:                                           \n"
		"    SAVE_8x16_MGN                                           \n"

		//----------------------------------------------------------------

		"BEGIN_M8N1_MGN:                                             \n"
		"       cmpq     $1, %%rdi                                   \n" // N % 16
		"    jb          END_M8N1_MGN                                \n" // TODO

		"    mov     %%r14, %%rax                                    \n" // Ac
		"       prefetcht0         (%%rax)                           \n"

		"    mov      %%rcx, %%r10                                   \n" // C0
		"       prefetcht1         (%%r10)                           \n"
		"    leaq     (%%r10, %%r8, 4), %%r11                        \n" // C1
		"       prefetcht1         (%%r11)                           \n"
		"    leaq     (%%r11, %%r8, 4), %%r12                        \n" // C2
		"       prefetcht1         (%%r12)                           \n"
		"    leaq     (%%r12, %%r8, 4), %%r13                        \n" // C3
		"       prefetcht1         (%%r13)                           \n"

		"    mov     %%rsi, %%rdx                                    \n" // K

		"       movss        (%%rbx), %%xmm4                         \n" // B0
		"       movss    (%%rax), %%xmm0                             \n" // A0
		"       movss    4(%%rax), %%xmm1                            \n" // A1

		"       vpxorq         %%xmm8, %%xmm8, %%xmm8                \n"
		"    vpxorq         %%xmm10, %%xmm10, %%xmm10                \n"
		"    vpxorq         %%xmm12, %%xmm12, %%xmm12                \n"
		"    vpxorq         %%xmm14, %%xmm14, %%xmm14                \n"
		"    vpxorq         %%xmm16, %%xmm16, %%xmm16                \n"
		"    vpxorq         %%xmm18, %%xmm18, %%xmm18                \n"
		"    vpxorq         %%xmm20, %%xmm20, %%xmm20                \n"
		"    vpxorq         %%xmm22, %%xmm22, %%xmm22                \n"

		"    subq     $8, %%rdx                                      \n"

		"K_M8N1_PREFETCH_C_MGN:                                      \n"
		"   leaq     (%%r13, %%r8, 4), %%r13                         \n"
		"   prefetcht2         (%%r13)                               \n"
		"   prefetcht2         64(%%r13)                             \n"

		"MAIN_K_M8N1_MGN:                                            \n"

		"    KERNEL8x1_K1_MGN                                        \n"
		"    KERNEL8x1_K2_MGN                                        \n"
		"    KERNEL8x1_K1_MGN                                        \n"
		"    KERNEL8x1_K2_MGN                                        \n"
		"    KERNEL8x1_K1_MGN                                        \n"
		"    KERNEL8x1_K2_MGN                                        \n"
		"    KERNEL8x1_K1_MGN                                        \n"
		"   cmp     $0, %%rdx                                        \n"
		"    je         EDGE_K_M8N1_MGN                              \n"
		"    KERNEL8x1_K2_MGN                                        \n"
		"   subq     $8, %%rdx                                       \n"
		"   cmp   $32, %%rdx                                         \n"
		"   jbe     K_M8N1_PREFETCH_C_MGN                            \n"
		"   jmp     MAIN_K_M8N1_MGN                                  \n"

		"EDGE_K_M8N1_MGN:                                            \n"
		"   leaq     (%%r12, %%r8, 4), %%r13                         \n"
		"    KERNEL8x1_END_K_MGN                                     \n"
		"    cmp     $0, %%r15                                       \n"
		"    je      SAVE_C_M8N1_MGN                                 \n"
		"    ADD_C_8x1_MGN                                           \n"

		"SAVE_C_M8N1_MGN:                                            \n"
		"    SAVE_8x1_MGN                                            \n"

		//----------------------------------------------------------------

		"END_M8N1_MGN:                                               \n"

		:
		:
		[C] "m"(C),
		[A] "m"(A),
		[B] "m"(B),
		[M] "m"(M),
		[N] "m"(N),
		[K] "m"(K),
		[LN] "m"(LN),
		[LK] "m"(LK),
		[Ac] "m"(Ac),
		[k_tag] "m"(k_tag)
		: "rax", "rbx", "rcx", "rdx", "rdi", "rsi", "rbp", "r8", "r9", "r10", "r11", "r12",
		  "r13", "r14", "r15", "zmm0", "zmm1", "zmm2", "zmm3", "zmm4", "zmm5",
		  "zmm6", "zmm7", "zmm8", "zmm9", "zmm10", "zmm11", "zmm12", "zmm13",
		  "zmm14", "zmm15", "zmm16", "zmm17", "zmm18", "zmm19", "zmm20", "zmm21",
		  "zmm22", "zmm23", "zmm24", "zmm25", "zmm26", "zmm27", "zmm28", "zmm29",
		  "zmm30", "zmm31", "memory"

	);
}

void MGN_KERNEL4x32(float *C, float *A, float *B, long M, long N, long K, long LN, long LK, float *Ac, long k_tag)
{

	asm volatile(
		".macro    KERNEL4x32_K1_MGN                                 \n"

		"    vbroadcastss    8(%%rax), %%zmm2                        \n"
		"    vfmadd231ps        %%zmm0, %%zmm4, %%zmm8               \n"
		"    vfmadd231ps        %%zmm0, %%zmm5, %%zmm9               \n"

		"    vbroadcastss    12(%%rax), %%zmm3                       \n"
		"    vfmadd231ps        %%zmm1, %%zmm4, %%zmm10              \n"
		"    vfmadd231ps        %%zmm1, %%zmm5, %%zmm11              \n"

		"    prefetcht0         256(%%rax)                           \n"

		"    addq              $16, %%rax                            \n"
		"    addq              $128, %%rbx                           \n"

		"    vbroadcastss    (%%rax), %%zmm0                         \n"
		"    vmovups         (%%rbx), %%zmm6                         \n"
		"    vfmadd231ps        %%zmm2, %%zmm4, %%zmm12              \n"
		"    vfmadd231ps        %%zmm2, %%zmm5, %%zmm13              \n"

		"    vbroadcastss    4(%%rax), %%zmm1                        \n"
		"    vmovups         64(%%rbx), %%zmm7                       \n"
		"    vfmadd231ps        %%zmm3, %%zmm4, %%zmm14              \n"
		"    vfmadd231ps        %%zmm3, %%zmm5, %%zmm15              \n"

		"    prefetcht0         256(%%rbx)                           \n"

		".endm                                                       \n"

		".macro    KERNEL4x32_K2_MGN                                 \n"

		"    vbroadcastss    8(%%rax), %%zmm2                        \n"
		"    vfmadd231ps        %%zmm0, %%zmm6, %%zmm8               \n"
		"    vfmadd231ps        %%zmm0, %%zmm7, %%zmm9               \n"

		"    vbroadcastss    12(%%rax), %%zmm3                       \n"
		"    vfmadd231ps        %%zmm1, %%zmm6, %%zmm10              \n"
		"    vfmadd231ps        %%zmm1, %%zmm7, %%zmm11              \n"

		"    prefetcht0         256(%%rax)                           \n"

		"    addq              $16, %%rax                            \n"
		"    addq              $128, %%rbx                           \n"

		"    vbroadcastss    (%%rax), %%zmm0                         \n"
		"    vmovups         (%%rbx), %%zmm4                         \n"
		"    vfmadd231ps        %%zmm2, %%zmm6, %%zmm12              \n"
		"    vfmadd231ps        %%zmm2, %%zmm7, %%zmm13              \n"

		"    vbroadcastss    4(%%rax), %%zmm1                        \n"
		"    vmovups         64(%%rbx), %%zmm5                       \n"
		"    vfmadd231ps        %%zmm3, %%zmm6, %%zmm14              \n"
		"    vfmadd231ps        %%zmm3, %%zmm7, %%zmm15              \n"

		"    prefetcht0         256(%%rbx)                           \n"

		".endm                                                       \n"

		".macro    KERNEL4x32_END_K_MGN                              \n"

		"    vbroadcastss    8(%%rax), %%zmm2                        \n"
		"    vfmadd231ps        %%zmm0, %%zmm6, %%zmm8               \n"
		"    vfmadd231ps        %%zmm0, %%zmm7, %%zmm9               \n"

		"    vbroadcastss    12(%%rax), %%zmm3                       \n"
		"    vfmadd231ps        %%zmm1, %%zmm6, %%zmm10              \n"
		"    vfmadd231ps        %%zmm1, %%zmm7, %%zmm11              \n"

		"    vfmadd231ps        %%zmm2, %%zmm6, %%zmm12              \n"
		"    vfmadd231ps        %%zmm2, %%zmm7, %%zmm13              \n"

		"    vfmadd231ps        %%zmm3, %%zmm6, %%zmm14              \n"
		"    vfmadd231ps        %%zmm3, %%zmm7, %%zmm15              \n"

		"    addq              $128, %%rbx                           \n" // TODO

		".endm                                                       \n"

		".macro    ADD_C_4x32_MGN                                    \n"

		"    vmovups         (%%r10), %%zmm0                         \n"
		"    vaddps             %%zmm0, %%zmm8, %%zmm8               \n"
		"    vmovups         64(%%r10), %%zmm1                       \n"
		"    vaddps             %%zmm1, %%zmm9, %%zmm9               \n"
		"    vmovups         (%%r11), %%zmm2                         \n"
		"    vaddps             %%zmm2, %%zmm10, %%zmm10             \n"
		"    vmovups         64(%%r11), %%zmm3                       \n"
		"    vaddps             %%zmm3, %%zmm11, %%zmm11             \n"
		"    vmovups         (%%r12), %%zmm4                         \n"
		"    vaddps             %%zmm4, %%zmm12, %%zmm12             \n"
		"    vmovups         64(%%r12), %%zmm5                       \n"
		"    vaddps             %%zmm5, %%zmm13, %%zmm13             \n"
		"    vmovups         (%%r13), %%zmm6                         \n"
		"    vaddps             %%zmm6, %%zmm14, %%zmm14             \n"
		"    vmovups         64(%%r13), %%zmm7                       \n"
		"    vaddps             %%zmm7, %%zmm15, %%zmm15             \n"

		"    mov      %%rcx, %%r10                                   \n" // C0
		"    leaq     (%%r10, %%r8, 4), %%r11                        \n" // C1
		"    leaq     (%%r11, %%r8, 4), %%r12                        \n" // C2
		"    leaq     (%%r12, %%r8, 4), %%r13                        \n" // C3

		".endm                                                       \n"

		".macro    SAVE_4x32_MGN                                     \n"

		"    vmovups         %%zmm8, (%%r10)                         \n"
		"    vmovups         %%zmm9, 64(%%r10)                       \n"
		"    vmovups         %%zmm10, (%%r11)                        \n"
		"    vmovups         %%zmm11, 64(%%r11)                      \n"
		"    vmovups         %%zmm12, (%%r12)                        \n"
		"    vmovups         %%zmm13, 64(%%r12)                      \n"
		"    vmovups         %%zmm14, (%%r13)                        \n"
		"    vmovups         %%zmm15, 64(%%r13)                      \n"

		"    subq         $32, %%rdi                                 \n" // TODO
		"    addq             $128, %%rcx                            \n" // C0

		".endm                                                       \n"

		//-----------------------------------------------------------------

		".macro    KERNEL4x16_K1_MGN                                 \n"

		"    vbroadcastss    8(%%rax), %%zmm2                        \n"
		"    vfmadd231ps        %%zmm0, %%zmm4, %%zmm8               \n"

		"    vbroadcastss    12(%%rax), %%zmm3                       \n"
		"    vfmadd231ps        %%zmm1, %%zmm4, %%zmm10              \n"

		"    prefetcht0         256(%%rax)                           \n"
		"    addq              $16, %%rax                            \n"
		"    addq              $64, %%rbx                            \n"

		"    vbroadcastss    (%%rax), %%zmm0                         \n"
		"    vmovups         (%%rbx), %%zmm6                         \n"
		"    vfmadd231ps        %%zmm2, %%zmm4, %%zmm12              \n"

		"    vbroadcastss    4(%%rax), %%zmm1                        \n"
		"    vfmadd231ps        %%zmm3, %%zmm4, %%zmm14              \n"

		"    prefetcht0         256(%%rbx)                           \n"

		".endm                                                       \n"

		".macro    KERNEL4x16_K2_MGN                                 \n"

		"    vbroadcastss    8(%%rax), %%zmm2                        \n"
		"    vfmadd231ps        %%zmm0, %%zmm6, %%zmm8               \n"

		"    vbroadcastss    12(%%rax), %%zmm3                       \n"
		"    vfmadd231ps        %%zmm1, %%zmm6, %%zmm10              \n"

		"    prefetcht0         256(%%rax)                           \n"
		"    addq              $16, %%rax                            \n"
		"    addq              $64, %%rbx                            \n"

		"    vbroadcastss    (%%rax), %%zmm0                         \n"
		"    vmovups         (%%rbx), %%zmm4                         \n"
		"    vfmadd231ps        %%zmm2, %%zmm6, %%zmm12              \n"

		"    vbroadcastss    4(%%rax), %%zmm1                        \n"
		"    vfmadd231ps        %%zmm3, %%zmm6, %%zmm14              \n"

		"    prefetcht0         256(%%rbx)                           \n"

		".endm                                                       \n"

		".macro    KERNEL4x16_END_K_MGN                              \n"

		"    vbroadcastss    8(%%rax), %%zmm2                        \n"
		"    vfmadd231ps        %%zmm0, %%zmm6, %%zmm8               \n"

		"    vbroadcastss    12(%%rax), %%zmm3                       \n"
		"    vfmadd231ps        %%zmm1, %%zmm6, %%zmm10              \n"

		"    prefetcht0         256(%%rax)                           \n"

		"    vfmadd231ps        %%zmm2, %%zmm6, %%zmm12              \n"

		"    vfmadd231ps        %%zmm3, %%zmm6, %%zmm14              \n"

		"    addq              $64, %%rbx                            \n" // TODO

		".endm                                                       \n"

		".macro    ADD_C_4x16_MGN                                    \n"

		"    vmovups     (%%r10), %%zmm0                             \n"
		"    vaddps         %%zmm0, %%zmm8, %%zmm8                   \n"
		"    vmovups     (%%r11), %%zmm2                             \n"
		"    vaddps         %%zmm2, %%zmm10, %%zmm10                 \n"
		"    vmovups     (%%r12), %%zmm4                             \n"
		"    vaddps         %%zmm4, %%zmm12, %%zmm12                 \n"
		"    vmovups     (%%r13), %%zmm6                             \n"
		"    vaddps         %%zmm6, %%zmm14, %%zmm14                 \n"

		"    mov            %%rcx, %%r10                             \n" // C0
		"    leaq        (%%r10, %%r8, 4), %%r11                     \n" // C1
		"    leaq        (%%r11, %%r8, 4), %%r12                     \n" // C2
		"    leaq        (%%r12, %%r8, 4), %%r13                     \n" // C3

		".endm                                                       \n"

		".macro    SAVE_4x16_MGN                                     \n"

		"    vmovups         %%zmm8, (%%r10)                         \n"
		"    vmovups         %%zmm10, (%%r11)                        \n"
		"    vmovups         %%zmm12, (%%r12)                        \n"
		"    vmovups         %%zmm14, (%%r13)                        \n"

		"    subq             $16, %%rdi                             \n"
		"    addq                 $64, %%rcx                         \n" // C0

		".endm                                                       \n"

		//-----------------------------------------------------------------

		".macro    KERNEL4x1_K1_MGN                                  \n"

		"    movss            8(%%rax), %%xmm2                       \n"
		"    vfmadd231ps        %%xmm0, %%xmm4, %%xmm8               \n"

		"    movss            12(%%rax), %%xmm3                      \n"
		"    vfmadd231ps        %%xmm1, %%xmm4, %%xmm10              \n"

		"    prefetcht0         256(%%rax)                           \n"
		"    addq              $16, %%rax                            \n"
		"    addq              $4, %%rbx                             \n" // TODO

		"    movss            (%%rax), %%xmm0                        \n"
		"    movss                (%%rbx), %%xmm6                    \n"
		"    vfmadd231ps        %%xmm2, %%xmm4, %%xmm12              \n"

		"    movss            4(%%rax), %%xmm1                       \n"
		"    vfmadd231ps        %%xmm3, %%xmm4, %%xmm14              \n"

		"    prefetcht0         256(%%rbx)                           \n"

		".endm                                                       \n"

		".macro    KERNEL4x1_K2_MGN                                  \n"

		"    movss            8(%%rax), %%xmm2                       \n"
		"    vfmadd231ps        %%xmm0, %%xmm6, %%xmm8               \n"

		"    movss            12(%%rax), %%xmm3                      \n"
		"    vfmadd231ps        %%xmm1, %%xmm6, %%xmm10              \n"

		"    prefetcht0         256(%%rax)                           \n"
		"    addq              $16, %%rax                            \n"
		"    addq              $4, %%rbx                             \n" // TODO

		"    movss            (%%rax), %%xmm0                        \n"
		"    movss             (%%rbx), %%xmm4                       \n"
		"    vfmadd231ps        %%xmm2, %%xmm6, %%xmm12              \n"

		"    movss            4(%%rax), %%xmm1                       \n"
		"    vfmadd231ps        %%xmm3, %%xmm6, %%xmm14              \n"

		"    prefetcht0         256(%%rbx)                           \n"

		".endm                                                       \n"

		".macro    KERNEL4x1_END_K_MGN                               \n"

		"    movss            8(%%rax), %%xmm2                       \n"
		"    vfmadd231ps        %%xmm0, %%xmm6, %%xmm8               \n"

		"    movss            12(%%rax), %%xmm3                      \n"
		"    vfmadd231ps        %%xmm1, %%xmm6, %%xmm10              \n"

		"    prefetcht0         256(%%rax)                           \n"

		"    vfmadd231ps        %%xmm2, %%xmm6, %%xmm12              \n"
		"    vfmadd231ps        %%xmm3, %%xmm6, %%xmm14              \n"

		"    addq              $4, %%rbx                             \n" // TODO

		".endm                                                       \n"

		".macro    ADD_C_4x1_MGN                                     \n"

		"    movss             (%%r10), %%xmm0                       \n"
		"    vaddps             %%xmm0, %%xmm8, %%xmm8               \n"
		"    movss             (%%r11), %%xmm2                       \n"
		"    vaddps             %%xmm2, %%xmm10, %%xmm10             \n"
		"    movss             (%%r12), %%xmm4                       \n"
		"    vaddps             %%xmm4, %%xmm12, %%xmm12             \n"
		"    movss             (%%r13), %%xmm6                       \n"
		"    vaddps             %%xmm6, %%xmm14, %%xmm14             \n"

		"    mov             %%rcx, %%r10                            \n" // C0
		"    leaq               (%%r10, %%r8, 4), %%r11              \n" // C1
		"    leaq               (%%r11, %%r8, 4), %%r12              \n" // C2
		"    leaq             (%%r12, %%r8, 4), %%r13                \n" // C3

		".endm                                                       \n"

		".macro    SAVE_4x1_MGN                                      \n"

		"    vmovss         %%xmm8, (%%r10)                          \n"
		"    vmovss         %%xmm10, (%%r11)                         \n"
		"    vmovss         %%xmm12, (%%r12)                         \n"
		"    vmovss         %%xmm14, (%%r13)                         \n"

		"    subq         $1, %%rdi                                  \n"
		"    addq             $4, %%rcx                              \n"

		".endm                                                       \n"

		//-----------------------------------------------------------------
		// pack A
		"   movl %[LK], %%r8d                                        \n"
		"   movl %[LK], %%r15d                                       \n"

		"   mov %[Ac], %%r9                                          \n"
		"   movl %[K], %%r10d                                        \n"
		"   mov %[A], %%r11                                          \n"
		"   mov %[A], %%r12                                          \n"
		"   mov %[A], %%r13                                          \n"
		"   mov %[A], %%r14                                          \n"

		"   shr $4, %%r10                                            \n" // K循环边界
		"   shl $2, %%r8                                             \n"
		"   shl $3, %%r15                                            \n"

		"   add %%r8, %%r14                                          \n"
		"   add %%r8, %%r12                                          \n"
		"   add %%r15, %%r13                                         \n"
		"   add %%r15, %%r14                                         \n"

		"   vmovups (%%r11), %%zmm2                                  \n"
		"   vmovups (%%r12), %%zmm3                                  \n"
		"   vmovups (%%r13), %%zmm4                                  \n"
		"   vmovups (%%r14), %%zmm5                                  \n"
		"   jmp NPACK_M4_MGN                                         \n"

		"NPACK_Pre_M4_MGN:                                           \n"
		"   add $64, %%r11                                           \n" // K维度上后16个元素位置
		"   add $64, %%r12                                           \n" // K维度上后16个元素位置
		"   add $64, %%r13                                           \n" // K维度上后16个元素位置
		"   add $64, %%r14                                           \n" // K维度上后16个元素位置
		"   add $256, %%r9                                           \n" // Ac指针后移
		"   vmovups (%%r11), %%zmm2                                  \n"
		"   vmovups (%%r12), %%zmm3                                  \n"
		"   vmovups (%%r13), %%zmm4                                  \n"
		"   vmovups (%%r14), %%zmm5                                  \n"

		"NPACK_M4_MGN:                                               \n"

		"   movl    $0xaa, %%eax                                     \n"
		"   kmovd   %%eax, %%k1                                      \n"
		"   vunpcklps %%zmm3, %%zmm2, %%zmm0                         \n" // 2元素连续
		"   vunpcklps %%zmm5, %%zmm4, %%zmm26                        \n"

		"   vpermq   $0x80, %%zmm26, %%zmm0%{%%k1%}                  \n" // 4元素连续
		"   vmovups %%xmm0, (%%r9)                                   \n"
		"   vextractf32x4  $0x1, %%zmm0, %%xmm28                     \n"
		"   vextractf32x4  $0x2, %%zmm0, %%xmm29                     \n"
		"   vextractf32x4  $0x3, %%zmm0, %%xmm30                     \n"

		"   vmovups %%xmm28, 64(%%r9)                                \n"
		"   vmovups %%xmm29, 128(%%r9)                               \n"
		"   vmovups %%xmm30, 192(%%r9)                               \n"

		"   movl    $0x55, %%eax                                     \n"
		"   kmovd   %%eax, %%k1                                      \n"

		"   vunpcklps %%zmm3, %%zmm2, %%zmm0                         \n"
		"   vunpcklps %%zmm5, %%zmm4, %%zmm26                        \n"

		"   vpermq  $0x31, %%zmm0, %%zmm26%{%%k1%}                   \n"
		"   vmovups %%xmm26, 16(%%r9)                                \n"
		"   vextractf32x4  $0x1, %%zmm26, %%xmm28                    \n"
		"   vextractf32x4  $0x2, %%zmm26, %%xmm29                    \n"
		"   vextractf32x4  $0x3, %%zmm26, %%xmm30                    \n"

		"   vmovups %%xmm28, 80(%%r9)                                \n"
		"   vmovups %%xmm29, 144(%%r9)                               \n"
		"   vmovups %%xmm30, 208(%%r9)                               \n"

		"   movl    $0xaa, %%eax                                     \n"
		"   kmovd   %%eax, %%k1                                      \n"
		"   vunpckhps %%zmm3, %%zmm2, %%zmm0                         \n"
		"   vunpckhps %%zmm5, %%zmm4, %%zmm26                        \n"
		"   vpermq   $0x80, %%zmm26, %%zmm0%{%%k1%}                  \n"
		"   vmovups %%xmm0, 32(%%r9)                                 \n"
		"   vextractf32x4  $0x1, %%zmm0, %%xmm28                     \n"
		"   vextractf32x4  $0x2, %%zmm0, %%xmm29                     \n"
		"   vextractf32x4  $0x3, %%zmm0, %%xmm30                     \n"

		"   vmovups %%xmm28, 96(%%r9)                                \n"
		"   vmovups %%xmm29, 160(%%r9)                               \n"
		"   vmovups %%xmm30, 224(%%r9)                               \n"
		"   movl    $0x55, %%eax                                     \n"
		"   kmovd   %%eax, %%k1                                      \n"

		"   vunpckhps %%zmm3, %%zmm2, %%zmm0                         \n"
		"   vunpckhps %%zmm5, %%zmm4, %%zmm26                        \n"

		"   vpermq  $0x31, %%zmm0, %%zmm26%{%%k1%}                   \n"
		"   vmovups %%xmm26, 48(%%r9)                                \n"
		"   vextractf32x4  $0x1, %%zmm26, %%xmm28                    \n"
		"   vextractf32x4  $0x2, %%zmm26, %%xmm29                    \n"
		"   vextractf32x4  $0x3, %%zmm26, %%xmm30                    \n"
		"   sub    $1, %%r10                                         \n"
		"   vmovups %%xmm28, 112(%%r9)                               \n"
		"   vmovups %%xmm29, 176(%%r9)                               \n"
		"   vmovups %%xmm30, 240(%%r9)                               \n"

		"   je  NPACK_END_M4_MGN                                     \n"
		"   jmp NPACK_Pre_M4_MGN                                     \n"

		"NPACK_END_M4_MGN:                                           \n"

		//-----------------------------------------------------------------

		"SMM_KERNEL4x32_MGN:                                         \n"

		"    mov     %[C], %%rcx                                     \n"
		"    mov     %[Ac], %%rax                                    \n"
		"    mov     %[B], %%rbx                                     \n"

		"    prefetcht0         (%%rax)                              \n"

		"    mov     %[K], %%rdx                                     \n" // K(kc)
		"    mov      %[LN], %%r8                                    \n"
		"    mov      %[Ac], %%r14                                   \n" // 存储Ac地址
		"    movq      %[N], %%rdi                                   \n"
		"    mov     %[k_tag], %%r15                                 \n" // kk=0把C存回内存, 否则加回对应的C位置

		"    prefetcht0         (%%rbx)                              \n"
		"    mov     %%rdx, %%rsi                                    \n" // K

		//-----------------------------------------------------------------

		"BEGIN_M4N32_MGN:                                            \n"
		"       cmpq     $32, %%rdi                                  \n" // N % 32
		"    jb          BEGIN_M4N16_MGN                             \n"

		"    mov     %%r14, %%rax                                    \n" // Ac
		"       prefetcht0         (%%rax)                           \n"

		"    mov      %%rcx, %%r10                                   \n" // C0
		"       prefetcht1         (%%r10)                           \n"
		"    leaq     (%%r10, %%r8, 4), %%r11                        \n" // C1
		"       prefetcht1         (%%r11)                           \n"
		"    leaq     (%%r11, %%r8, 4), %%r12                        \n" // C2
		"       prefetcht1         (%%r12)                           \n"
		"    leaq     (%%r12, %%r8, 4), %%r13                        \n" // C3
		"       prefetcht1         (%%r13)                           \n"

		"    mov     %%rsi, %%rdx                                    \n" // K

		"       vmovups            (%%rbx), %%zmm4                   \n" // B0-15
		"    vbroadcastss    (%%rax), %%zmm0                         \n" // A0
		"    vpxorq             %%zmm8, %%zmm8, %%zmm8               \n"
		"    vpxorq             %%zmm9, %%zmm9, %%zmm9               \n"
		"    vpxorq             %%zmm10, %%zmm10, %%zmm10            \n"
		"    vpxorq             %%zmm11, %%zmm11, %%zmm11            \n"

		"       vmovups         64(%%rbx), %%zmm5                    \n"
		"    vbroadcastss    4(%%rax), %%zmm1                        \n" // A1
		"    vpxorq             %%zmm12, %%zmm12, %%zmm12            \n"
		"    vpxorq             %%zmm13, %%zmm13, %%zmm13            \n"
		"    vpxorq             %%zmm14, %%zmm14, %%zmm14            \n"
		"    vpxorq             %%zmm15, %%zmm15, %%zmm15            \n"

		"    subq     $8, %%rdx                                      \n"

		"K_M4N32_PREFETCH_C_MGN:                                     \n"
		"       leaq     (%%r13, %%r8, 4), %%r13                     \n"
		"       prefetcht2         (%%r13)                           \n"
		"       prefetcht2         64(%%r13)                         \n"

		"MAIN_K_M4N32_MGN:                                           \n"

		"    KERNEL4x32_K1_MGN                                       \n"
		"    KERNEL4x32_K2_MGN                                       \n"
		"    KERNEL4x32_K1_MGN                                       \n"
		"    KERNEL4x32_K2_MGN                                       \n"
		"    KERNEL4x32_K1_MGN                                       \n"
		"    KERNEL4x32_K2_MGN                                       \n"
		"    KERNEL4x32_K1_MGN                                       \n"
		"       cmp     $0, %%rdx                                    \n"
		"    je         EDGE_K_M4N32_MGN                             \n"
		"    KERNEL4x32_K2_MGN                                       \n"
		"       subq     $8, %%rdx                                   \n"
		"       cmp       $64, %%rdx                                 \n"
		"       jbe     K_M4N32_PREFETCH_C_MGN                       \n"
		"       jmp     MAIN_K_M4N32_MGN                             \n"

		"EDGE_K_M4N32_MGN:                                           \n"
		"       leaq     (%%r12, %%r8, 4), %%r13                     \n"
		"    KERNEL4x32_END_K_MGN                                    \n"

		"BEGIN_SAVE_M4N32_MGN:                                       \n"
		"    cmp     $0, %%r15                                       \n"
		"    je      SAVE_C_M4N32_MGN                                \n"
		"    ADD_C_4x32_MGN                                          \n"

		"SAVE_C_M4N32_MGN:                                           \n"
		"    SAVE_4x32_MGN                                           \n"
		"    cmpq      $32, %%rdi                                    \n"
		"    jnb     BEGIN_M4N32_MGN                                 \n" // 不小于（或等于）则跳转

		//----------------------------------------------------------------

		"BEGIN_M4N16_MGN:                                            \n"
		"       cmpq     $16, %%rdi                                  \n" // N % 16
		"    jb          BEGIN_M4N1_MGN                              \n" // TODO

		"    mov     %%r14, %%rax                                    \n" // Ac
		"       prefetcht0         (%%rax)                           \n"

		"    mov      %%rcx, %%r10                                   \n" // C0
		"       prefetcht1         (%%r10)                           \n"
		"    leaq     (%%r10, %%r8, 4), %%r11                        \n" // C1
		"       prefetcht1         (%%r11)                           \n"
		"    leaq     (%%r11, %%r8, 4), %%r12                        \n" // C2
		"       prefetcht1         (%%r12)                           \n"
		"    leaq     (%%r12, %%r8, 4), %%r13                        \n" // C3
		"       prefetcht1         (%%r13)                           \n"

		"    mov     %%rsi, %%rdx                                    \n" // K

		"       vmovups            (%%rbx), %%zmm4                   \n" // B0-15
		"       vbroadcastss    (%%rax), %%zmm0                      \n" // A0
		"    vbroadcastss    4(%%rax), %%zmm1                        \n" // A1

		"       vpxorq         %%zmm8, %%zmm8, %%zmm8                \n"
		"    vpxorq         %%zmm10, %%zmm10, %%zmm10                \n"
		"    vpxorq         %%zmm12, %%zmm12, %%zmm12                \n"
		"    vpxorq         %%zmm14, %%zmm14, %%zmm14                \n"

		"    subq     $8, %%rdx                                      \n"

		"K_M4N16_PREFETCH_C_MGN:                                     \n"
		"   leaq     (%%r13, %%r8, 4), %%r13                         \n"
		"   prefetcht2         (%%r13)                               \n"
		"   prefetcht2         64(%%r13)                             \n"

		"MAIN_K_M4N16_MGN:                                           \n"

		"    KERNEL4x16_K1_MGN                                       \n"
		"    KERNEL4x16_K2_MGN                                       \n"
		"    KERNEL4x16_K1_MGN                                       \n"
		"    KERNEL4x16_K2_MGN                                       \n"
		"    KERNEL4x16_K1_MGN                                       \n"
		"    KERNEL4x16_K2_MGN                                       \n"
		"    KERNEL4x16_K1_MGN                                       \n"
		"   cmp     $0, %%rdx                                        \n"
		"    je         EDGE_K_M4N16_MGN                             \n"
		"    KERNEL4x16_K2_MGN                                       \n"
		"   subq     $8, %%rdx                                       \n"
		"   cmp   $32, %%rdx                                         \n"
		"   jbe     K_M4N16_PREFETCH_C_MGN                           \n"
		"   jmp     MAIN_K_M4N16_MGN                                 \n"

		"EDGE_K_M4N16_MGN:                                           \n"
		"   leaq     (%%r12, %%r8, 4), %%r13                         \n"
		"    KERNEL4x16_END_K_MGN                                    \n"
		"    cmp     $0, %%r15                                       \n"
		"    je      SAVE_C_M4N16_MGN                                \n"
		"    ADD_C_4x16_MGN                                          \n"

		"SAVE_C_M4N16_MGN:                                           \n"
		"    SAVE_4x16_MGN                                           \n"

		//----------------------------------------------------------------

		"BEGIN_M4N1_MGN:                                             \n"
		"       cmpq     $1, %%rdi                                   \n" // N % 16
		"    jb          END_M4N1_MGN                                \n" // TODO

		"    mov     %%r14, %%rax                                    \n" // Ac
		"       prefetcht0         (%%rax)                           \n"

		"    mov      %%rcx, %%r10                                   \n" // C0
		"       prefetcht1         (%%r10)                           \n"
		"    leaq     (%%r10, %%r8, 4), %%r11                        \n" // C1
		"       prefetcht1         (%%r11)                           \n"
		"    leaq     (%%r11, %%r8, 4), %%r12                        \n" // C2
		"       prefetcht1         (%%r12)                           \n"
		"    leaq     (%%r12, %%r8, 4), %%r13                        \n" // C3
		"       prefetcht1         (%%r13)                           \n"

		"    mov     %%rsi, %%rdx                                    \n" // K

		"       movss        (%%rbx), %%xmm4                         \n" // B0
		"       movss    (%%rax), %%xmm0                             \n" // A0
		"       movss    4(%%rax), %%xmm1                            \n" // A1

		"       vpxorq         %%xmm8, %%xmm8, %%xmm8                \n"
		"    vpxorq         %%xmm10, %%xmm10, %%xmm10                \n"
		"    vpxorq         %%xmm12, %%xmm12, %%xmm12                \n"
		"    vpxorq         %%xmm14, %%xmm14, %%xmm14                \n"

		"    subq     $8, %%rdx                                      \n"

		"K_M4N1_PREFETCH_C_MGN:                                      \n"
		"   leaq     (%%r13, %%r8, 4), %%r13                         \n"
		"   prefetcht2         (%%r13)                               \n"
		"   prefetcht2         64(%%r13)                             \n"

		"MAIN_K_M4N1_MGN:                                            \n"

		"    KERNEL4x1_K1_MGN                                        \n"
		"    KERNEL4x1_K2_MGN                                        \n"
		"    KERNEL4x1_K1_MGN                                        \n"
		"    KERNEL4x1_K2_MGN                                        \n"
		"    KERNEL4x1_K1_MGN                                        \n"
		"    KERNEL4x1_K2_MGN                                        \n"
		"    KERNEL4x1_K1_MGN                                        \n"
		"   cmp     $0, %%rdx                                        \n"
		"    je         EDGE_K_M4N1_MGN                              \n"
		"    KERNEL4x1_K2_MGN                                        \n"
		"   subq     $8, %%rdx                                       \n"
		"   cmp   $32, %%rdx                                         \n"
		"   jbe     K_M4N1_PREFETCH_C_MGN                            \n"
		"   jmp     MAIN_K_M4N1_MGN                                  \n"

		"EDGE_K_M4N1_MGN:                                            \n"
		"   leaq     (%%r12, %%r8, 4), %%r13                         \n"
		"    KERNEL4x1_END_K_MGN                                     \n"
		"    cmp     $0, %%r15                                       \n"
		"    je      SAVE_C_M4N1_MGN                                 \n"
		"    ADD_C_4x1_MGN                                           \n"

		"SAVE_C_M4N1_MGN:                                            \n"
		"    SAVE_4x1_MGN                                            \n"

		//----------------------------------------------------------------

		"END_M4N1_MGN:                                               \n"

		:
		:
		[C] "m"(C),
		[A] "m"(A),
		[B] "m"(B),
		[M] "m"(M),
		[N] "m"(N),
		[K] "m"(K),
		[LN] "m"(LN),
		[LK] "m"(LK),
		[Ac] "m"(Ac),
		[k_tag] "m"(k_tag)
		: "rax", "rbx", "rcx", "rdx", "rdi", "rsi", "rbp", "r8", "r9", "r10", "r11", "r12",
		  "r13", "r14", "r15", "zmm0", "zmm1", "zmm2", "zmm3", "zmm4", "zmm5",
		  "zmm6", "zmm7", "zmm8", "zmm9", "zmm10", "zmm11", "zmm12", "zmm13",
		  "zmm14", "zmm15", "zmm16", "zmm17", "zmm18", "zmm19", "zmm20", "zmm21",
		  "zmm22", "zmm23", "zmm24", "zmm25", "zmm26", "zmm27", "zmm28", "zmm29",
		  "zmm30", "zmm31", "memory"

	);
}

void MGN_KERNEL1x32(float *C, float *A, float *B, long M, long N, long K, long LN, long LK, float *Ac, long k_tag)
{

	asm volatile(
		".macro    KERNEL1x32_K1_MGN                                 \n"
		"    vfmadd231ps        %%zmm0, %%zmm4, %%zmm8               \n"
		"    vfmadd231ps        %%zmm0, %%zmm5, %%zmm9               \n"
		"    prefetcht0         256(%%rax)                           \n"

		"    addq              $4, %%rax                             \n"
		"    addq              $128, %%rbx                           \n"

		"    vbroadcastss    (%%rax), %%zmm1                         \n"
		"    vmovups         (%%rbx), %%zmm6                         \n"
		"    vmovups         64(%%rbx), %%zmm7                       \n"

		"    prefetcht0         256(%%rbx)                           \n"

		".endm                                                       \n"

		".macro    KERNEL1x32_K2_MGN                                 \n"

		"    vfmadd231ps        %%zmm1, %%zmm6, %%zmm8               \n"
		"    vfmadd231ps        %%zmm1, %%zmm7, %%zmm9               \n"
		"    prefetcht0         256(%%rax)                           \n"

		"    addq              $4, %%rax                             \n"
		"    addq              $128, %%rbx                           \n"

		"    vbroadcastss    (%%rax), %%zmm0                         \n"
		"    vmovups         (%%rbx), %%zmm4                         \n"
		"    vmovups         64(%%rbx), %%zmm5                       \n"

		"    prefetcht0         256(%%rbx)                           \n"

		".endm                                                       \n"

		".macro    KERNEL1x32_END_K_MGN                              \n"

		"    vfmadd231ps        %%zmm1, %%zmm6, %%zmm8               \n"
		"    vfmadd231ps        %%zmm1, %%zmm7, %%zmm9               \n"

		"    addq              $128, %%rbx                           \n" // TODO

		".endm                                                       \n"

		".macro    ADD_C_1x32_MGN                                    \n"

		"    vmovups         (%%r10), %%zmm0                         \n"
		"    vaddps             %%zmm0, %%zmm8, %%zmm8               \n"
		"    vmovups         64(%%r10), %%zmm1                       \n"
		"    vaddps             %%zmm1, %%zmm9, %%zmm9               \n"

		"    mov      %%rcx, %%r10                                   \n" // C0

		".endm                                                       \n"

		".macro    SAVE_1x32_MGN                                     \n"

		"    vmovups         %%zmm8, (%%r10)                         \n"
		"    vmovups         %%zmm9, 64(%%r10)                       \n"

		"    subq         $32, %%rdi                                 \n" // TODO
		"    addq             $128, %%rcx                            \n" // C0

		".endm                                                       \n"

		//-----------------------------------------------------------------

		".macro    KERNEL1x16_K1_MGN                                 \n"

		"    vfmadd231ps        %%zmm0, %%zmm4, %%zmm8               \n"

		"    prefetcht0         256(%%rax)                           \n"
		"    addq              $4, %%rax                             \n"
		"    addq              $64, %%rbx                            \n"

		"    vbroadcastss    (%%rax), %%zmm1                         \n"
		"    vmovups         (%%rbx), %%zmm6                         \n"

		"    prefetcht0         256(%%rbx)                           \n"

		".endm                                                       \n"

		".macro    KERNEL1x16_K2_MGN                                 \n"

		"    vfmadd231ps        %%zmm1, %%zmm6, %%zmm8               \n"

		"    prefetcht0         256(%%rax)                           \n"
		"    addq              $4, %%rax                             \n"
		"    addq              $64, %%rbx                            \n"

		"    vbroadcastss    (%%rax), %%zmm0                         \n"
		"    vmovups         (%%rbx), %%zmm4                         \n"

		"    prefetcht0         256(%%rbx)                           \n"

		".endm                                                       \n"

		".macro    KERNEL1x16_END_K_MGN                              \n"

		"    vfmadd231ps        %%zmm1, %%zmm6, %%zmm8               \n"

		"    addq              $64, %%rbx                            \n" // TODO

		".endm                                                       \n"

		".macro    ADD_C_1x16_MGN                                    \n"

		"    vmovups     (%%r10), %%zmm0                             \n"
		"    vaddps         %%zmm0, %%zmm8, %%zmm8                   \n"

		"    mov            %%rcx, %%r10                             \n" // C0

		".endm                                                       \n"

		".macro    SAVE_1x16_MGN                                     \n"

		"    vmovups         %%zmm8, (%%r10)                         \n"

		"    subq             $16, %%rdi                             \n"
		"    addq                 $64, %%rcx                         \n" // C0

		".endm                                                       \n"

		//-----------------------------------------------------------------

		".macro    KERNEL1x1_K1_MGN                                  \n"

		"    vfmadd231ps        %%xmm0, %%xmm4, %%xmm8               \n"

		"    prefetcht0         256(%%rax)                           \n"
		"    addq              $4, %%rax                             \n"
		"    addq              $4, %%rbx                             \n" // TODO

		"    movss            (%%rax), %%xmm1                        \n"
		"    movss                (%%rbx), %%xmm6                    \n"

		"    prefetcht0         256(%%rbx)                           \n"

		".endm                                                       \n"

		".macro    KERNEL1x1_K2_MGN                                  \n"

		"    vfmadd231ps        %%xmm1, %%xmm6, %%xmm8               \n"

		"    prefetcht0         256(%%rax)                           \n"
		"    addq              $4, %%rax                             \n"
		"    addq              $4, %%rbx                             \n" // TODO

		"    movss            (%%rax), %%xmm0                        \n"
		"    movss             (%%rbx), %%xmm4                       \n"

		"    prefetcht0         256(%%rbx)                           \n"

		".endm                                                       \n"

		".macro    KERNEL1x1_END_K_MGN                               \n"

		"    vfmadd231ps        %%xmm1, %%xmm6, %%xmm8               \n"

		"    addq              $4, %%rbx                             \n" // TODO

		".endm                                                       \n"

		".macro    ADD_C_1x1_MGN                                     \n"

		"    movss             (%%r10), %%xmm0                       \n"
		"    vaddps             %%xmm0, %%xmm8, %%xmm8               \n"

		"    mov             %%rcx, %%r10                            \n" // C0

		".endm                                                       \n"

		".macro    SAVE_1x1_MGN                                      \n"

		"    vmovss         %%xmm8, (%%r10)                          \n"

		"    subq         $1, %%rdi                                  \n"
		"    addq             $4, %%rcx                              \n"

		".endm                                                       \n"

		//-----------------------------------------------------------------

		"SMM_KERNEL1x32_MGN:                                         \n"

		"    mov     %[C], %%rcx                                     \n"
		"    mov     %[A], %%rax                                     \n"
		"    mov     %[B], %%rbx                                     \n"

		"    prefetcht0         (%%rax)                              \n"

		"    mov     %[K], %%rdx                                     \n" // K(kc)
		"    mov      %[LN], %%r8                                    \n"
		"    mov      %[A], %%r14                                    \n" // 存储Ac地址
		"    movq      %[N], %%rdi                                   \n"
		"    mov     %[k_tag], %%r15                                 \n" // kk=0把C存回内存, 否则加回对应的C位置

		"    prefetcht0         (%%rbx)                              \n"
		"    mov     %%rdx, %%rsi                                    \n" // K

		//-----------------------------------------------------------------

		"BEGIN_M1N32_MGN:                                            \n"
		"       cmpq     $32, %%rdi                                  \n" // N % 32
		"    jb          BEGIN_M1N16_MGN                             \n"

		"    mov     %%r14, %%rax                                    \n" // Ac
		"       prefetcht0         (%%rax)                           \n"

		"    mov      %%rcx, %%r10                                   \n" // C0
		"       prefetcht1         (%%r10)                           \n"

		"    mov     %%rsi, %%rdx                                    \n" // K

		"       vmovups            (%%rbx), %%zmm4                   \n" // B0-15
		"    vbroadcastss    (%%rax), %%zmm0                         \n" // A0
		"    vpxorq             %%zmm8, %%zmm8, %%zmm8               \n"
		"    vpxorq             %%zmm9, %%zmm9, %%zmm9               \n"

		"       vmovups         64(%%rbx), %%zmm5                    \n"

		"    subq     $8, %%rdx                                      \n"

		"K_M1N32_PREFETCH_C_MGN:                                     \n"
		"       leaq     (%%r13, %%r8, 4), %%r13                     \n"
		"       prefetcht2         (%%r13)                           \n"
		"       prefetcht2         64(%%r13)                         \n"

		"MAIN_K_M1N32_MGN:                                           \n"

		"    KERNEL1x32_K1_MGN                                       \n"
		"    KERNEL1x32_K2_MGN                                       \n"
		"    KERNEL1x32_K1_MGN                                       \n"
		"    KERNEL1x32_K2_MGN                                       \n"
		"    KERNEL1x32_K1_MGN                                       \n"
		"    KERNEL1x32_K2_MGN                                       \n"
		"    KERNEL1x32_K1_MGN                                       \n"
		"       cmp     $0, %%rdx                                    \n"
		"    je         EDGE_K_M1N32_MGN                             \n"
		"    KERNEL1x32_K2_MGN                                       \n"
		"       subq     $8, %%rdx                                   \n"
		"       cmp       $64, %%rdx                                 \n"
		"       jbe     K_M1N32_PREFETCH_C_MGN                       \n"
		"       jmp     MAIN_K_M1N32_MGN                             \n"

		"EDGE_K_M1N32_MGN:                                           \n"
		"       leaq     (%%r12, %%r8, 4), %%r13                     \n"
		"    KERNEL1x32_END_K_MGN                                    \n"

		"BEGIN_SAVE_M1N32_MGN:                                       \n"
		"    cmp     $0, %%r15                                       \n"
		"    je      SAVE_C_M1N32_MGN                                \n"
		"    ADD_C_1x32_MGN                                          \n"

		"SAVE_C_M1N32_MGN:                                           \n"
		"    SAVE_1x32_MGN                                           \n"
		"    cmpq      $32, %%rdi                                    \n"
		"    jnb     BEGIN_M1N32_MGN                                 \n" // 不小于（或等于）则跳转

		//----------------------------------------------------------------

		"BEGIN_M1N16_MGN:                                            \n"
		"       cmpq     $16, %%rdi                                  \n" // N % 16
		"    jb          BEGIN_M1N1_MGN                              \n" // TODO

		"    mov     %%r14, %%rax                                    \n" // Ac
		"       prefetcht0         (%%rax)                           \n"

		"    mov      %%rcx, %%r10                                   \n" // C0
		"       prefetcht1         (%%r10)                           \n"
		"    mov     %%rsi, %%rdx                                    \n" // K

		"       vmovups            (%%rbx), %%zmm4                   \n" // B0-15
		"       vbroadcastss    (%%rax), %%zmm0                      \n" // A0

		"       vpxorq         %%zmm8, %%zmm8, %%zmm8                \n"

		"    subq     $8, %%rdx                                      \n"

		"K_M1N16_PREFETCH_C_MGN:                                     \n"
		"   leaq     (%%r13, %%r8, 4), %%r13                         \n"
		"   prefetcht2         (%%r13)                               \n"
		"   prefetcht2         64(%%r13)                             \n"

		"MAIN_K_M1N16_MGN:                                           \n"

		"    KERNEL1x16_K1_MGN                                       \n"
		"    KERNEL1x16_K2_MGN                                       \n"
		"    KERNEL1x16_K1_MGN                                       \n"
		"    KERNEL1x16_K2_MGN                                       \n"
		"    KERNEL1x16_K1_MGN                                       \n"
		"    KERNEL1x16_K2_MGN                                       \n"
		"    KERNEL1x16_K1_MGN                                       \n"
		"   cmp     $0, %%rdx                                        \n"
		"    je         EDGE_K_M1N16_MGN                             \n"
		"    KERNEL1x16_K2_MGN                                       \n"
		"   subq     $8, %%rdx                                       \n"
		"   cmp   $32, %%rdx                                         \n"
		"   jbe     K_M1N16_PREFETCH_C_MGN                           \n"
		"   jmp     MAIN_K_M1N16_MGN                                 \n"

		"EDGE_K_M1N16_MGN:                                           \n"
		"   leaq     (%%r12, %%r8, 4), %%r13                         \n"
		"    KERNEL1x16_END_K_MGN                                    \n"
		"    cmp     $0, %%r15                                       \n"
		"    je      SAVE_C_M1N16_MGN                                \n"
		"    ADD_C_1x16_MGN                                          \n"

		"SAVE_C_M1N16_MGN:                                           \n"
		"    SAVE_1x16_MGN                                           \n"

		//----------------------------------------------------------------

		"BEGIN_M1N1_MGN:                                             \n"
		"       cmpq     $1, %%rdi                                   \n" // N % 16
		"    jb          END_M1N1_MGN                                \n" // TODO

		"    mov     %%r14, %%rax                                    \n" // Ac
		"       prefetcht0         (%%rax)                           \n"

		"    mov      %%rcx, %%r10                                   \n" // C0
		"       prefetcht1         (%%r10)                           \n"

		"    mov     %%rsi, %%rdx                                    \n" // K

		"       movss        (%%rbx), %%xmm4                         \n" // B0
		"       movss    (%%rax), %%xmm0                             \n" // A0

		"       vpxorq         %%xmm8, %%xmm8, %%xmm8                \n"

		"    subq     $8, %%rdx                                      \n"

		"K_M1N1_PREFETCH_C_MGN:                                      \n"
		"   leaq     (%%r13, %%r8, 4), %%r13                         \n"
		"   prefetcht2         (%%r13)                               \n"
		"   prefetcht2         64(%%r13)                             \n"

		"MAIN_K_M1N1_MGN:                                            \n"

		"    KERNEL1x1_K1_MGN                                        \n"
		"    KERNEL1x1_K2_MGN                                        \n"
		"    KERNEL1x1_K1_MGN                                        \n"
		"    KERNEL1x1_K2_MGN                                        \n"
		"    KERNEL1x1_K1_MGN                                        \n"
		"    KERNEL1x1_K2_MGN                                        \n"
		"    KERNEL1x1_K1_MGN                                        \n"
		"   cmp     $0, %%rdx                                        \n"
		"    je         EDGE_K_M1N1_MGN                              \n"
		"    KERNEL1x1_K2_MGN                                        \n"
		"   subq     $8, %%rdx                                       \n"
		"   cmp   $32, %%rdx                                         \n"
		"   jbe     K_M1N1_PREFETCH_C_MGN                            \n"
		"   jmp     MAIN_K_M1N1_MGN                                  \n"

		"EDGE_K_M1N1_MGN:                                            \n"
		"   leaq     (%%r12, %%r8, 4), %%r13                         \n"
		"    KERNEL1x1_END_K_MGN                                     \n"
		"    cmp     $0, %%r15                                       \n"
		"    je      SAVE_C_M1N1_MGN                                 \n"
		"    ADD_C_1x1_MGN                                           \n"

		"SAVE_C_M1N1_MGN:                                            \n"
		"    SAVE_1x1_MGN                                            \n"

		//----------------------------------------------------------------

		"END_M1N1_MGN:                                               \n"

		:
		:
		[C] "m"(C),
		[A] "m"(A),
		[B] "m"(B),
		[M] "m"(M),
		[N] "m"(N),
		[K] "m"(K),
		[LN] "m"(LN),
		[LK] "m"(LK),
		[Ac] "m"(Ac),
		[k_tag] "m"(k_tag)
		: "rax", "rbx", "rcx", "rdx", "rdi", "rsi", "rbp", "r8", "r9", "r10", "r11", "r12",
		  "r13", "r14", "r15", "zmm0", "zmm1", "zmm2", "zmm3", "zmm4", "zmm5",
		  "zmm6", "zmm7", "zmm8", "zmm9", "zmm10", "zmm11", "zmm12", "zmm13",
		  "zmm14", "zmm15", "zmm16", "zmm17", "zmm18", "zmm19", "zmm20", "zmm21",
		  "zmm22", "zmm23", "zmm24", "zmm25", "zmm26", "zmm27", "zmm28", "zmm29",
		  "zmm30", "zmm31", "memory"

	);
}