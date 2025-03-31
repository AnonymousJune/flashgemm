void SMM_NN_KERNELm12xn32(float *C, float *A, float *B, long M, long N, long K, long LN, long LK, float *Bc, long k_tag);
void SMM_NN_KERNELm12xn16(float *C, float *A, float *B, long M, long N, long K, long LN, long LK, float *Bc, long k_tag);
void SMM_NN_KERNELm12xn8(float *C, float *A, float *B, long M, long N, long K, long LN, long LK, float *Bc, long k_tag);
void SMM_NN_KERNELm12xn4(float *C, float *A, float *B, long M, long N, long K, long LN, long LK, float *Bc, long k_tag);
// void SMM_NN_KERNELm12xn2(float *C, float *A, float *B, long M, long N, long K, long LN, long LK, float *Bc, long k_tag);
void SMM_NN_KERNELm12xn1(float *C, float *A, float *B, long M, long N, long K, long LN, long LK, float *Bc, long k_tag);

void SMM_NN_KERNELm12xn32(float *C, float *A, float *B, long M,
						  long N, long K, long LN, long LK, float *Bc, long k_tag)
{

	asm volatile(
		// 将C预取隐藏在k维计算中，预取到L3 Cache，条件预取
		// 加入直接打包Bc，解决只能处理M>24的问题
		".macro    KERNEL12x32_PACK_K1                               \n"

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

		"   prefetcht0         384(%%rax)                            \n"

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

		".macro    KERNEL12x32_PACK_K2                               \n"

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

		".macro    KERNEL12x32_PACK_END_K                            \n"

		"   vbroadcastss    8(%%rax), %%zmm2                         \n" // next A2

		"   vfmadd231ps        %%zmm0, %%zmm6, %%zmm8                \n"
		"   vfmadd231ps        %%zmm0, %%zmm7, %%zmm9                \n"

		"   vbroadcastss    12(%%rax), %%zmm3                        \n" // next A3

		"   vfmadd231ps        %%zmm1, %%zmm6, %%zmm10               \n"
		"   vfmadd231ps        %%zmm1, %%zmm7, %%zmm11               \n"

		"   prefetcht0         256(%%rax)                            \n"

		"   vbroadcastss    16(%%rax), %%zmm0                        \n" // next A4
		"   vfmadd231ps        %%zmm2, %%zmm6, %%zmm12               \n"
		"   vfmadd231ps        %%zmm2, %%zmm7, %%zmm13               \n"

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
		"    addq              $48, %%rax                            \n"
		"   vfmadd231ps        %%zmm1, %%zmm7, %%zmm27               \n"

		"   vmovups         %%zmm6, (%%rbp)                          \n" // pack B0 to Bc
		"   vfmadd231ps        %%zmm2, %%zmm6, %%zmm28               \n"
		"   vfmadd231ps        %%zmm2, %%zmm7, %%zmm29               \n"

		"   vmovups         %%zmm7, 64(%%rbp)                        \n" // pack B0 to Bc
		"   vfmadd231ps        %%zmm3, %%zmm6, %%zmm30               \n"
		"   vfmadd231ps        %%zmm3, %%zmm7, %%zmm31               \n"

		".endm                                                       \n"

		//-----------------------------------------------------------------

		".macro    KERNEL12x32_K1                                    \n"

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

		".macro    KERNEL12x32_K2                                    \n"

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

		".macro    KERNEL12x32_END_K                                 \n"

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
		"    addq              $48, %%rax                            \n"
		"   vfmadd231ps        %%zmm1, %%zmm7, %%zmm27               \n"

		"   vfmadd231ps        %%zmm2, %%zmm6, %%zmm28               \n"
		"   vfmadd231ps        %%zmm2, %%zmm7, %%zmm29               \n"

		"   vfmadd231ps        %%zmm3, %%zmm6, %%zmm30               \n"
		"   vfmadd231ps        %%zmm3, %%zmm7, %%zmm31               \n"

		".endm                                                       \n"

		".macro    ADD_C_12x32                                       \n"

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

		".macro    SAVE_12x32                                        \n"

		"   vmovups         %%zmm8, (%%r10)                          \n"
		"   vmovups         %%zmm9, 64(%%r10)                        \n"
		"   vmovups         %%zmm10, (%%r11)                         \n"
		"   vmovups         %%zmm11, 64(%%r11)                       \n"
		"   vmovups         %%zmm12, (%%r12)                         \n"
		"   vmovups         %%zmm13, 64(%%r12)                       \n"
		"   vmovups         %%zmm14, (%%r13)                         \n"
		"   vmovups         %%zmm15, 64(%%r13)                       \n"

		"    leaq      (%%r13, %%r8, 4), %%r10                       \n" // C0
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

		"    leaq      (%%r13, %%r8, 4), %%r10                       \n" // C0
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

		".macro    KERNEL8x32_K1                                     \n"

		"   vbroadcastss    8(%%rax), %%zmm2                         \n"
		"   vfmadd231ps        %%zmm0, %%zmm4, %%zmm16               \n"
		"   vfmadd231ps        %%zmm0, %%zmm5, %%zmm17               \n"

		"   vbroadcastss    12(%%rax), %%zmm3                        \n"
		"   vfmadd231ps        %%zmm1, %%zmm4, %%zmm18               \n"
		"   vfmadd231ps        %%zmm1, %%zmm5, %%zmm19               \n"

		"   prefetcht0         256(%%rax)                            \n"

		"   vbroadcastss    16(%%rax), %%zmm8                        \n"
		"   vfmadd231ps        %%zmm2, %%zmm4, %%zmm20               \n"
		"   vfmadd231ps        %%zmm2, %%zmm5, %%zmm21               \n"

		"    addq              $128, %%rbx                           \n"

		"   vbroadcastss    20(%%rax), %%zmm9                        \n"
		"   vfmadd231ps        %%zmm3, %%zmm4, %%zmm22               \n"
		"   vfmadd231ps        %%zmm3, %%zmm5, %%zmm23               \n"

		"   prefetcht0         64(%%rbx)                             \n"

		"   vbroadcastss    24(%%rax), %%zmm10                       \n"
		"   vfmadd231ps        %%zmm8, %%zmm4, %%zmm24               \n"
		"   vfmadd231ps        %%zmm8, %%zmm5, %%zmm25               \n"

		"   vbroadcastss    28(%%rax), %%zmm11                       \n"
		"   vfmadd231ps        %%zmm9, %%zmm4, %%zmm26               \n"
		"   vmovups         (%%rbx), %%zmm6                          \n"
		"    addq              $32, %%rax                            \n"
		"   vfmadd231ps        %%zmm9, %%zmm5, %%zmm27               \n"

		"   vbroadcastss    (%%rax), %%zmm0                          \n"
		"   vfmadd231ps        %%zmm10, %%zmm4, %%zmm28              \n"
		"   vmovups         64(%%rbx), %%zmm7                        \n"
		"   vfmadd231ps        %%zmm10, %%zmm5, %%zmm29              \n"

		"   vbroadcastss    4(%%rax), %%zmm1                         \n"
		"   vfmadd231ps        %%zmm11, %%zmm4, %%zmm30              \n"
		"   vfmadd231ps        %%zmm11, %%zmm5, %%zmm31              \n"

		".endm                                                       \n"

		".macro    KERNEL8x32_K2                                     \n"

		"   vbroadcastss    8(%%rax), %%zmm2                         \n"
		"   vfmadd231ps        %%zmm0, %%zmm6, %%zmm16               \n"
		"   vfmadd231ps        %%zmm0, %%zmm7, %%zmm17               \n"

		"   vbroadcastss    12(%%rax), %%zmm3                        \n"
		"   vfmadd231ps        %%zmm1, %%zmm6, %%zmm18               \n"
		"   vfmadd231ps        %%zmm1, %%zmm7, %%zmm19               \n"

		"   prefetcht0         256(%%rax)                            \n"

		"   vbroadcastss    16(%%rax), %%zmm8                        \n"
		"   vfmadd231ps        %%zmm2, %%zmm6, %%zmm20               \n"
		"   vfmadd231ps        %%zmm2, %%zmm7, %%zmm21               \n"

		"    addq              $128, %%rbx                           \n"

		"   vbroadcastss    20(%%rax), %%zmm9                        \n"
		"   vfmadd231ps        %%zmm3, %%zmm6, %%zmm22               \n"
		"   vfmadd231ps        %%zmm3, %%zmm7, %%zmm23               \n"

		"   prefetcht0         64(%%rbx)                             \n"

		"   vbroadcastss    24(%%rax), %%zmm10                       \n"
		"   vfmadd231ps        %%zmm8, %%zmm6, %%zmm24               \n"
		"   vfmadd231ps        %%zmm8, %%zmm7, %%zmm25               \n"

		"   vbroadcastss    28(%%rax), %%zmm11                       \n"
		"   vfmadd231ps        %%zmm9, %%zmm6, %%zmm26               \n"
		"   vmovups         (%%rbx), %%zmm4                          \n"
		"    addq              $32, %%rax                            \n"
		"   vfmadd231ps        %%zmm9, %%zmm7, %%zmm27               \n"

		"   vbroadcastss    (%%rax), %%zmm0                          \n"
		"   vfmadd231ps        %%zmm10, %%zmm6, %%zmm28              \n"
		"   vmovups         64(%%rbx), %%zmm5                        \n"
		"   vfmadd231ps        %%zmm10, %%zmm7, %%zmm29              \n"

		"   vbroadcastss    4(%%rax), %%zmm1                         \n"
		"   vfmadd231ps        %%zmm11, %%zmm6, %%zmm30              \n"
		"   vfmadd231ps        %%zmm11, %%zmm7, %%zmm31              \n"

		".endm                                                       \n"

		".macro    KERNEL8x32_END_K                                  \n"

		"   vbroadcastss    8(%%rax), %%zmm2                         \n"
		"   vfmadd231ps        %%zmm0, %%zmm6, %%zmm16               \n"
		"   vfmadd231ps        %%zmm0, %%zmm7, %%zmm17               \n"

		"   vbroadcastss    12(%%rax), %%zmm3                        \n"
		"   vfmadd231ps        %%zmm1, %%zmm6, %%zmm18               \n"
		"   vfmadd231ps        %%zmm1, %%zmm7, %%zmm19               \n"

		"   prefetcht0         256(%%rax)                            \n"

		"   vbroadcastss    16(%%rax), %%zmm8                        \n"
		"   vfmadd231ps        %%zmm2, %%zmm6, %%zmm20               \n"
		"   vfmadd231ps        %%zmm2, %%zmm7, %%zmm21               \n"

		"   vbroadcastss    20(%%rax), %%zmm9                        \n"
		"   vfmadd231ps        %%zmm3, %%zmm6, %%zmm22               \n"
		"   vfmadd231ps        %%zmm3, %%zmm7, %%zmm23               \n"

		"   prefetcht0         64(%%rbx)                             \n"

		"   vbroadcastss    24(%%rax), %%zmm10                       \n"
		"   vfmadd231ps        %%zmm8, %%zmm6, %%zmm24               \n"
		"   vfmadd231ps        %%zmm8, %%zmm7, %%zmm25               \n"

		"   vbroadcastss    28(%%rax), %%zmm11                       \n"
		"   vfmadd231ps        %%zmm9, %%zmm6, %%zmm26               \n"
		"    addq              $32, %%rax                            \n"
		"   vfmadd231ps        %%zmm9, %%zmm7, %%zmm27               \n"

		"   vfmadd231ps        %%zmm10, %%zmm6, %%zmm28              \n"
		"   vfmadd231ps        %%zmm10, %%zmm7, %%zmm29              \n"
		"   vfmadd231ps        %%zmm11, %%zmm6, %%zmm30              \n"
		"   vfmadd231ps        %%zmm11, %%zmm7, %%zmm31              \n"

		".endm                                                       \n"

		".macro    ADD_C_8x32                                        \n"

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

		".macro    SAVE_8x32                                         \n"

		"   vmovups         %%zmm16, (%%r10)                         \n"
		"   vmovups         %%zmm17, 64(%%r10)                       \n"
		"   vmovups         %%zmm18, (%%r11)                         \n"
		"   vmovups         %%zmm19, 64(%%r11)                       \n"
		"   vmovups         %%zmm20, (%%r12)                         \n"
		"   vmovups         %%zmm21, 64(%%r12)                       \n"
		"   vmovups         %%zmm22, (%%r13)                         \n"
		"   vmovups         %%zmm23, 64(%%r13)                       \n"

		"    leaq      (%%r13, %%r8, 4), %%r10                       \n" // C0
		"    leaq     (%%r10, %%r8, 4), %%r11                        \n" // C1
		"    leaq     (%%r11, %%r8, 4), %%r12                        \n" // C2
		"    leaq     (%%r12, %%r8, 4), %%r13                        \n" // C3

		"   vmovups         %%zmm24, (%%r10)                         \n"
		"   vmovups         %%zmm25, 64(%%r10)                       \n"
		"   vmovups         %%zmm26, (%%r11)                         \n"
		"   vmovups         %%zmm27, 64(%%r11)                       \n"
		"    subq             $8, %%rdi                              \n"
		"   vmovups         %%zmm28, (%%r12)                         \n"
		"   vmovups         %%zmm29, 64(%%r12)                       \n"
		"   vmovups         %%zmm30, (%%r13)                         \n"
		"   vmovups         %%zmm31, 64(%%r13)                       \n"

		"    leaq      (%%r13, %%r8, 4), %%rcx                       \n" // C0

		".endm                                                       \n"

		//---------------------------------------------------------------

		".macro    KERNEL4x32_K1                                     \n"

		"   vbroadcastss    8(%%rax), %%zmm2                         \n"
		"   vfmadd231ps        %%zmm0, %%zmm4, %%zmm24               \n"
		"    addq              $128, %%rbx                           \n"
		"   vfmadd231ps        %%zmm0, %%zmm5, %%zmm25               \n"

		"   vbroadcastss    12(%%rax), %%zmm3                        \n"
		"   vfmadd231ps        %%zmm1, %%zmm4, %%zmm26               \n"
		"    addq              $16, %%rax                            \n"
		"   vfmadd231ps        %%zmm1, %%zmm5, %%zmm27               \n"

		"   prefetcht0         256(%%rax)                            \n"

		"   vbroadcastss    (%%rax), %%zmm0                          \n"
		"   vfmadd231ps        %%zmm2, %%zmm4, %%zmm28               \n"
		"   vmovups         (%%rbx), %%zmm6                          \n"
		"   vfmadd231ps        %%zmm2, %%zmm5, %%zmm29               \n"

		"   vmovups         64(%%rbx), %%zmm7                        \n"
		"   vfmadd231ps        %%zmm3, %%zmm4, %%zmm30               \n"
		"   vbroadcastss    4(%%rax), %%zmm1                         \n"
		"   vfmadd231ps        %%zmm3, %%zmm5, %%zmm31               \n"
		"   prefetcht0         64(%%rbx)                             \n"

		".endm                                                       \n"

		".macro    KERNEL4x32_K2                                     \n"

		"   vbroadcastss    8(%%rax), %%zmm2                         \n"
		"   vfmadd231ps        %%zmm0, %%zmm6, %%zmm24               \n"
		"    addq              $128, %%rbx                           \n"
		"   vfmadd231ps        %%zmm0, %%zmm7, %%zmm25               \n"

		"   vbroadcastss    12(%%rax), %%zmm3                        \n"
		"   vfmadd231ps        %%zmm1, %%zmm6, %%zmm26               \n"
		"    addq              $16, %%rax                            \n"
		"   vfmadd231ps        %%zmm1, %%zmm7, %%zmm27               \n"

		"   prefetcht0         256(%%rax)                            \n"

		"   vbroadcastss    (%%rax), %%zmm0                          \n"
		"   vfmadd231ps        %%zmm2, %%zmm6, %%zmm28               \n"
		"   vmovups         (%%rbx), %%zmm4                          \n"
		"   vfmadd231ps        %%zmm2, %%zmm7, %%zmm29               \n"

		"    vfmadd231ps        %%zmm3, %%zmm6, %%zmm30              \n"
		"   vmovups         64(%%rbx), %%zmm5                        \n"
		"    vfmadd231ps        %%zmm3, %%zmm7, %%zmm31              \n"
		"   vbroadcastss    4(%%rax), %%zmm1                         \n"
		"   prefetcht0         64(%%rbx)                             \n"

		".endm                                                       \n"

		".macro    KERNEL4x32_END_K                                  \n"

		"   vbroadcastss    8(%%rax), %%zmm2                         \n"
		"   vfmadd231ps        %%zmm0, %%zmm6, %%zmm24               \n"
		"   vfmadd231ps        %%zmm0, %%zmm7, %%zmm25               \n"

		"   vbroadcastss    12(%%rax), %%zmm3                        \n"
		"   vfmadd231ps        %%zmm1, %%zmm6, %%zmm26               \n"
		"    addq              $16, %%rax                            \n"
		"   vfmadd231ps        %%zmm1, %%zmm7, %%zmm27               \n"

		"   vfmadd231ps        %%zmm2, %%zmm6, %%zmm28               \n"
		"   vfmadd231ps        %%zmm2, %%zmm7, %%zmm29               \n"

		"    vfmadd231ps        %%zmm3, %%zmm6, %%zmm30              \n"
		"    vfmadd231ps        %%zmm3, %%zmm7, %%zmm31              \n"

		".endm                                                       \n"

		".macro    ADD_C_4x32                                        \n"

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

		".macro    SAVE_4x32                                         \n"

		"   vmovups         %%zmm24, (%%r10)                         \n"
		"   vmovups         %%zmm25, 64(%%r10)                       \n"
		"   vmovups         %%zmm26, (%%r11)                         \n"
		"   vmovups         %%zmm27, 64(%%r11)                       \n"
		"    subq             $4, %%rdi                              \n"
		"   vmovups         %%zmm28, (%%r12)                         \n"
		"   vmovups         %%zmm29, 64(%%r12)                       \n"
		"   vmovups         %%zmm30, (%%r13)                         \n"
		"   vmovups         %%zmm31, 64(%%r13)                       \n"

		"    leaq      (%%r13, %%r8, 4), %%rcx                       \n" // C0

		".endm                                                       \n"

		//-----------------------------------------------------------------

		".macro    KERNEL1x32_K1                                     \n"
		"   vbroadcastss    4(%%rax), %%zmm2                         \n"
		"   vfmadd231ps        %%zmm0, %%zmm4, %%zmm24               \n"
		"    addq              $128, %%rbx                           \n"
		"   vmovups         (%%rbx), %%zmm6                          \n"
		"   prefetcht0         64(%%rbx)                             \n"
		"   vfmadd231ps        %%zmm0, %%zmm5, %%zmm25               \n"

		"    addq              $4, %%rax                             \n"
		"   vmovups         64(%%rbx), %%zmm7                        \n"
		"   prefetcht0         64(%%rax)                             \n"
		".endm                                                       \n"

		".macro    KERNEL1x32_K2                                     \n"
		"   vbroadcastss    4(%%rax), %%zmm0                         \n"
		"   vfmadd231ps        %%zmm2, %%zmm6, %%zmm24               \n"
		"    addq              $128, %%rbx                           \n"
		"   vmovups         (%%rbx), %%zmm4                          \n"
		"   prefetcht0         64(%%rbx)                             \n"
		"   vfmadd231ps        %%zmm2, %%zmm7, %%zmm25               \n"

		"    addq              $4, %%rax                             \n"
		"   vmovups         64(%%rbx), %%zmm5                        \n"
		"   prefetcht0         64(%%rax)                             \n"
		".endm                                                       \n"

		".macro    KERNEL1x32_END_K                                  \n"
		"   vbroadcastss    4(%%rax), %%zmm0                         \n"
		"   vfmadd231ps        %%zmm2, %%zmm6, %%zmm24               \n"
		"   vfmadd231ps        %%zmm2, %%zmm7, %%zmm25               \n"

		"    addq              $4, %%rax                             \n"

		".endm                                                       \n"

		".macro    ADD_C_1x32                                        \n"
		"   vmovups         (%%r10), %%zmm10                         \n"
		"    vaddps             %%zmm10, %%zmm24, %%zmm24            \n"
		"   vmovups         64(%%r10), %%zmm11                       \n"
		"    vaddps             %%zmm11, %%zmm25, %%zmm25            \n"

		"    mov      %%rcx, %%r10                                   \n" // C0
		".endm                                                       \n"

		".macro    SAVE_1x32                                         \n"
		"   vmovups         %%zmm24, (%%r10)                         \n"
		"   vmovups         %%zmm25, 64(%%r10)                       \n"
		"    subq             $1, %%rdi                              \n"

		"    leaq      (%%r10, %%r8, 4), %%rcx                       \n" // next C0
		".endm                                                       \n"

		//-----------------------------------------------------------------

		"SMM_NN_KERNEL12x32:                                         \n"

		"   mov     %[C], %%rcx                                      \n"
		"   mov     %[A], %%rax                                      \n"
		"   mov     %[B], %%rbx                                      \n"

		"   prefetcht0         (%%rax)                               \n"

		"    mov     %[K], %%rdx                                     \n" // K(kc)
		"    mov      %[LN], %%r8                                    \n"
		"    mov      %[Bc], %%r14                                   \n"
		"    movq  %[M], %%rdi                                       \n"
		"    mov     %[k_tag], %%r15                                 \n" // kk=0把C存回内存, 否则加回对应的C位置

		"   prefetcht0         (%%rbx)                               \n"
		"    mov     %%rbx, %%r9                                     \n" // B
		"    mov     %%rdx, %%rsi                                    \n" // K
		"   mov     %%r14, %%rbp                                     \n"

		//-----------------------------------------------------------------

		"BEGIN_PACK:                                                 \n"
		"   cmpq      $12, %%rdi                                     \n" // FIXME
		"    jb          DIRECT_PACK                                 \n"

		"    mov     %%r9, %%rbx                                     \n" // B
		"   prefetcht0         (%%rbx)                               \n"

		"    mov      %%rcx, %%r10                                   \n" // C0
		"   prefetcht2         (%%r10)                               \n"
		"    leaq     (%%r10, %%r8, 4), %%r11                        \n" // C1
		"   prefetcht2         (%%r11)                               \n"
		"    leaq     (%%r11, %%r8, 4), %%r12                        \n" // C2
		"   prefetcht2         (%%r12)                               \n"
		"    leaq     (%%r12, %%r8, 4), %%r13                        \n" // C3
		"   prefetcht2         (%%r13)                               \n"

		"    mov     %%rsi, %%rdx                                    \n" // K
		"    mov     %%r14, %%rbp                                    \n" // Bc

		"   vmovups        (%%rbx), %%zmm4                           \n" // B0-15
		"    vpxorq         %%zmm8, %%zmm8, %%zmm8                   \n"
		"    vpxorq         %%zmm9, %%zmm9, %%zmm9                   \n"
		"    vpxorq         %%zmm10, %%zmm10, %%zmm10                \n"
		"    vpxorq         %%zmm11, %%zmm11, %%zmm11                \n"
		"   vmovups     64(%%rbx), %%zmm5                            \n" // B16-31

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
		"   prefetcht2         64(%%r10)                             \n"
		"    vpxorq         %%zmm22, %%zmm22, %%zmm22                \n"
		"    vpxorq         %%zmm23, %%zmm23, %%zmm23                \n"
		"   prefetcht2         64(%%r11)                             \n"
		"    vpxorq         %%zmm24, %%zmm24, %%zmm24                \n"
		"    vpxorq         %%zmm25, %%zmm25, %%zmm25                \n"
		"   prefetcht2         64(%%r12)                             \n"
		"    vpxorq         %%zmm26, %%zmm26, %%zmm26                \n"
		"    vpxorq         %%zmm27, %%zmm27, %%zmm27                \n"
		"   prefetcht2         64(%%r13)                             \n"
		"    vpxorq         %%zmm28, %%zmm28, %%zmm28                \n"
		"    vpxorq         %%zmm29, %%zmm29, %%zmm29                \n"
		"    vpxorq         %%zmm30, %%zmm30, %%zmm30                \n"
		"    vpxorq         %%zmm31, %%zmm31, %%zmm31                \n" // C:zmm8-31(24个)

		"    subq     $8, %%rdx                                      \n" // K-=8, cant process K<8 and K%8<>0

		"PACK_K_PREFETCH_C:                                          \n"
		"   leaq     (%%r13, %%r8, 4), %%r13                         \n"
		"   prefetcht2         (%%r13)                               \n"
		"   prefetcht2         64(%%r13)                             \n"

		"MAIN_PACK_K:                                                \n"

		"    KERNEL12x32_PACK_K1                                     \n"
		"    KERNEL12x32_PACK_K2                                     \n"
		"    KERNEL12x32_PACK_K1                                     \n"
		"    KERNEL12x32_PACK_K2                                     \n"
		"    KERNEL12x32_PACK_K1                                     \n"
		"    KERNEL12x32_PACK_K2                                     \n"
		"    KERNEL12x32_PACK_K1                                     \n"
		"   cmp     $0, %%rdx                                        \n"
		"    je         EDGE_PACK_K                                  \n"
		"    KERNEL12x32_PACK_K2                                     \n"
		"    subq     $8, %%rdx                                      \n"
		"   cmp   $64, %%rdx                                         \n"
		"   jbe     PACK_K_PREFETCH_C                                \n"
		"   jmp     MAIN_PACK_K                                      \n"

		"EDGE_PACK_K:                                                \n"
		"   leaq     (%%r12, %%r8, 4), %%r13                         \n"
		"    KERNEL12x32_PACK_END_K                                  \n"
		"    jmp      BEGIN_SAVE                                     \n"

		//-----------------------------------------------------------------

		"DIRECT_PACK:                                                \n"
		"   cmp           $0, %%rdx                                  \n"
		"   je          BEGIN_M8                                     \n"

		"   vmovups        (%%rbx), %%zmm4                           \n" // B0-15
		"   vmovups        64(%%rbx), %%zmm5                         \n" // B16-31
		"   vmovups         %%zmm4, (%%rbp)                          \n" // pack B0 to Bc
		"   vmovups         %%zmm5, 64(%%rbp)                        \n" // pack B1 to Bc
		"   leaq         (%%rbx, %%r8, 4), %%rbx                     \n" // next B
		"   addq          $128, %%rbp                                \n" // next Bc
		"   prefetcht2     128(%%rbx)                                \n"
		"   prefetcht2     192(%%rbx)                                \n"

		"   vmovups        (%%rbx), %%zmm6                           \n" // B0-15
		"   vmovups        64(%%rbx), %%zmm7                         \n" // B16-31
		"   vmovups         %%zmm6, (%%rbp)                          \n" // pack B0 to Bc
		"   vmovups         %%zmm7, 64(%%rbp)                        \n" // pack B1 to Bc
		"   leaq         (%%rbx, %%r8, 4), %%rbx                     \n" // next B
		"   addq          $128, %%rbp                                \n" // next Bc
		"   prefetcht2     128(%%rbx)                                \n"
		"   prefetcht2     192(%%rbx)                                \n"

		"   vmovups        (%%rbx), %%zmm8                           \n" // B0-15
		"   vmovups        64(%%rbx), %%zmm9                         \n" // B16-31
		"   vmovups         %%zmm8, (%%rbp)                          \n" // pack B0 to Bc
		"   vmovups         %%zmm9, 64(%%rbp)                        \n" // pack B1 to Bc
		"   leaq         (%%rbx, %%r8, 4), %%rbx                     \n" // next B
		"   addq          $128, %%rbp                                \n" // next Bc
		"   prefetcht2     128(%%rbx)                                \n"
		"   prefetcht2     192(%%rbx)                                \n"

		"   vmovups        (%%rbx), %%zmm10                          \n" // B0-15
		"   vmovups        64(%%rbx), %%zmm11                        \n" // B16-31
		"   vmovups         %%zmm10, (%%rbp)                         \n" // pack B0 to Bc
		"   vmovups         %%zmm11, 64(%%rbp)                       \n" // pack B1 to Bc
		"   leaq         (%%rbx, %%r8, 4), %%rbx                     \n" // next B
		"   addq          $128, %%rbp                                \n" // next Bc
		"   prefetcht2     128(%%rbx)                                \n"
		"   prefetcht2     192(%%rbx)                                \n"

		"   subq        $4, %%rdx                                    \n"
		"   jmp         DIRECT_PACK                                  \n"

		//-----------------------------------------------------------------
		"BEGIN_M:                                                    \n"
		"   cmpq      $12, %%rdi                                     \n" // FIXME
		"    jb       BEGIN_M8                                       \n"

		"    mov     %%r14, %%rbx                                    \n" // Bc
		"   prefetcht0         (%%rbx)                               \n"

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
		"   vmovups     64(%%rbx), %%zmm5                            \n" // B16-31

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

		"K_M12_PREFETCH_C:                                           \n"
		"   leaq     (%%r13, %%r8, 4), %%r13                         \n"
		"   prefetcht2         (%%r13)                               \n"
		"   prefetcht2         64(%%r13)                             \n"

		"MAIN_K_M12:                                                 \n"

		"    KERNEL12x32_K1                                          \n"
		"    KERNEL12x32_K2                                          \n"
		"    KERNEL12x32_K1                                          \n"
		"    KERNEL12x32_K2                                          \n"
		"    KERNEL12x32_K1                                          \n"
		"    KERNEL12x32_K2                                          \n"
		"    KERNEL12x32_K1                                          \n"
		"   cmp     $0, %%rdx                                        \n"
		"    je         EDGE_K                                       \n"
		"    KERNEL12x32_K2                                          \n"
		"   subq     $8, %%rdx                                       \n"
		"   cmp   $64, %%rdx                                         \n"
		"   jbe     K_M12_PREFETCH_C                                 \n"
		"   jmp     MAIN_K_M12                                       \n"

		"EDGE_K:                                                     \n"
		"   leaq     (%%r12, %%r8, 4), %%r13                         \n"
		"    KERNEL12x32_END_K                                       \n"

		"BEGIN_SAVE:                                                 \n"
		"    cmp     $0, %%r15                                       \n"
		"    je      SAVE_C                                          \n"
		"    ADD_C_12x32                                             \n"

		"SAVE_C:                                                     \n"
		"    SAVE_12x32                                              \n"
		"    cmpq      $12, %%rdi                                    \n"
		"    jnb     BEGIN_M                                         \n" // 不小于（或等于）则跳转

		//-----------------------------------------------------------------

		"BEGIN_M8:                                                   \n"
		"   cmpq      $8, %%rdi                                      \n" // M % 8
		"    jb       BEGIN_M4                                       \n" // 小于则跳转!!!改了

		"    mov     %%r14, %%rbx                                    \n" // Bc
		"   prefetcht0         (%%rbx)                               \n"

		"    mov      %%rcx, %%r10                                   \n" // C0
		"   prefetcht2         (%%r10)                               \n"
		"    leaq     (%%r10, %%r8, 4), %%r11                        \n" // C1
		"   prefetcht2         (%%r11)                               \n"
		"    leaq     (%%r11, %%r8, 4), %%r12                        \n" // C2
		"   prefetcht2         (%%r12)                               \n"
		"    leaq     (%%r12, %%r8, 4), %%r13                        \n" // C3
		"   prefetcht2         (%%r13)                               \n"

		"    mov     %%rsi, %%rdx                                    \n" // K

		"   vmovups        (%%rbx), %%zmm4                           \n"
		"    vpxorq         %%zmm16, %%zmm16, %%zmm16                \n"
		"    vpxorq         %%zmm17, %%zmm17, %%zmm17                \n"
		"   vmovups     64(%%rbx), %%zmm5                            \n"
		"    vpxorq         %%zmm18, %%zmm18, %%zmm18                \n"
		"    vpxorq         %%zmm19, %%zmm19, %%zmm19                \n"

		"   vbroadcastss    (%%rax), %%zmm0                          \n"
		"   vbroadcastss    4(%%rax), %%zmm1                         \n"

		"    vpxorq         %%zmm20, %%zmm20, %%zmm20                \n"
		"    vpxorq         %%zmm21, %%zmm21, %%zmm21                \n"
		"   prefetcht2         64(%%r10)                             \n"
		"    vpxorq         %%zmm22, %%zmm22, %%zmm22                \n"
		"    vpxorq         %%zmm23, %%zmm23, %%zmm23                \n"
		"   prefetcht2         64(%%r11)                             \n"
		"    vpxorq         %%zmm24, %%zmm24, %%zmm24                \n"
		"    vpxorq         %%zmm25, %%zmm25, %%zmm25                \n"
		"   prefetcht2         64(%%r12)                             \n"
		"    vpxorq         %%zmm26, %%zmm26, %%zmm26                \n"
		"    vpxorq         %%zmm27, %%zmm27, %%zmm27                \n"
		"   prefetcht2         64(%%r13)                             \n"
		"    vpxorq         %%zmm28, %%zmm28, %%zmm28                \n"
		"    vpxorq         %%zmm29, %%zmm29, %%zmm29                \n"
		"    vpxorq         %%zmm30, %%zmm30, %%zmm30                \n"
		"    vpxorq         %%zmm31, %%zmm31, %%zmm31                \n"

		"    subq     $8, %%rdx                                      \n"

		"K_M8_PREFETCH_C:                                            \n"
		"   leaq     (%%r13, %%r8, 4), %%r13                         \n"
		"   prefetcht2         (%%r13)                               \n"
		"   prefetcht2         64(%%r13)                             \n"

		"MAIN_K_M8:                                                  \n"

		"    KERNEL8x32_K1                                           \n"
		"    KERNEL8x32_K2                                           \n"
		"    KERNEL8x32_K1                                           \n"
		"    KERNEL8x32_K2                                           \n"
		"    KERNEL8x32_K1                                           \n"
		"    KERNEL8x32_K2                                           \n"
		"    KERNEL8x32_K1                                           \n"
		"   cmp     $0, %%rdx                                        \n"
		"    je         EDGE_K_M8                                    \n"
		"    KERNEL8x32_K2                                           \n"
		"    subq     $8, %%rdx                                      \n"
		"   cmp   $32, %%rdx                                         \n"
		"   jbe     K_M8_PREFETCH_C                                  \n"
		"   jmp     MAIN_K_M8                                        \n"

		"EDGE_K_M8:                                                  \n"
		"   leaq     (%%r12, %%r8, 4), %%r13                         \n"
		"    KERNEL8x32_END_K                                        \n"

		"    cmp     $0, %%r15                                       \n"
		"    je      SAVE_C_8x32                                     \n"
		"    ADD_C_8x32                                              \n"

		"SAVE_C_8x32:                                                \n"
		"    SAVE_8x32                                               \n"

		//----------------------------------------------------------------

		"BEGIN_M4:                                                   \n"

		"    cmpq      $4, %%rdi                                     \n" // M % 4
		"    jb       BEGIN_M1                                       \n" // 小于则跳转!!!改了

		"    mov     %%r14, %%rbx                                    \n" // Bc
		"   prefetcht0         (%%rbx)                               \n"

		"    mov      %%rcx, %%r10                                   \n" // C0
		"   prefetcht2         (%%r10)                               \n"
		"    leaq     (%%r10, %%r8, 4), %%r11                        \n" // C1
		"   prefetcht2         (%%r11)                               \n"
		"    leaq     (%%r11, %%r8, 4), %%r12                        \n" // C2
		"   prefetcht2         (%%r12)                               \n"
		"    leaq     (%%r12, %%r8, 4), %%r13                        \n" // C3
		"   prefetcht2         (%%r13)                               \n"

		"    mov     %%rsi, %%rdx                                    \n" // K

		"   vmovups        (%%rbx), %%zmm4                           \n"
		"   vmovups     64(%%rbx), %%zmm5                            \n"

		"   prefetcht2         64(%%r10)                             \n"
		"    vpxorq         %%zmm24, %%zmm24, %%zmm24                \n"
		"    vpxorq         %%zmm25, %%zmm25, %%zmm25                \n"
		"   vbroadcastss    (%%rax), %%zmm0                          \n"
		"   prefetcht2         64(%%r11)                             \n"
		"    vpxorq         %%zmm26, %%zmm26, %%zmm26                \n"
		"    vpxorq         %%zmm27, %%zmm27, %%zmm27                \n"
		"   vbroadcastss    4(%%rax), %%zmm1                         \n"
		"   prefetcht2         64(%%r12)                             \n"
		"    vpxorq         %%zmm28, %%zmm28, %%zmm28                \n"
		"    vpxorq         %%zmm29, %%zmm29, %%zmm29                \n"
		"   prefetcht2         64(%%r13)                             \n"
		"    vpxorq         %%zmm30, %%zmm30, %%zmm30                \n"
		"    vpxorq         %%zmm31, %%zmm31, %%zmm31                \n"

		"    subq     $8, %%rdx                                      \n"

		"MAIN_K_M4:                                                  \n"

		"    KERNEL4x32_K1                                           \n"
		"    KERNEL4x32_K2                                           \n"
		"    KERNEL4x32_K1                                           \n"
		"    KERNEL4x32_K2                                           \n"
		"    KERNEL4x32_K1                                           \n"
		"    KERNEL4x32_K2                                           \n"
		"    KERNEL4x32_K1                                           \n"
		"   cmp     $0, %%rdx                                        \n"
		"    je         EDGE_K_M4                                    \n"
		"    KERNEL4x32_K2                                           \n"

		"    subq     $8, %%rdx                                      \n"
		"   jmp     MAIN_K_M4                                        \n"

		"EDGE_K_M4:                                                  \n"

		"    KERNEL4x32_END_K                                        \n"

		"    cmp     $0, %%r15                                       \n"
		"    je      SAVE_C_4x32                                     \n"
		"    ADD_C_4x32                                              \n"

		"SAVE_C_4x32:                                                \n"
		"    SAVE_4x32                                               \n"

		//----------------------------------------------------------------

		"BEGIN_M1:                                                   \n"
		"    cmpq      $1, %%rdi                                     \n"
		"    jb       END_M                                          \n" // 小于则跳转

		"    mov     %%r14, %%rbx                                    \n" // Bc
		"   prefetcht0         (%%rbx)                               \n"

		"    mov      %%rcx, %%r10                                   \n" // C0
		"   prefetcht2         (%%r10)                               \n"

		"    mov     %%rsi, %%rdx                                    \n" // K

		"   vmovups        (%%rbx), %%zmm4                           \n" // B0-15
		"   vmovups     64(%%rbx), %%zmm5                            \n" // B16-31

		"   prefetcht2         64(%%r10)                             \n"
		"    vpxorq         %%zmm24, %%zmm24, %%zmm24                \n"
		"    vpxorq         %%zmm25, %%zmm25, %%zmm25                \n"
		"   vbroadcastss    (%%rax), %%zmm0                          \n" // A0

		"    subq     $8, %%rdx                                      \n"

		"MAIN_K_M1:                                                  \n"
		"    KERNEL1x32_K1                                           \n"
		"    KERNEL1x32_K2                                           \n"
		"    KERNEL1x32_K1                                           \n"
		"    KERNEL1x32_K2                                           \n"
		"    KERNEL1x32_K1                                           \n"
		"    KERNEL1x32_K2                                           \n"
		"    KERNEL1x32_K1                                           \n"
		"   cmp     $0, %%rdx                                        \n"
		"    je    EDGE_K_M1                                         \n"
		"    KERNEL1x32_K2                                           \n"

		"    subq     $8, %%rdx                                      \n"
		"   jmp   MAIN_K_M1                                          \n"

		"EDGE_K_M1:                                                  \n"
		"   KERNEL1x32_END_K                                         \n"

		"    cmp     $0, %%r15                                       \n"
		"    je      SAVE_C_1x32                                     \n"
		"    ADD_C_1x32                                              \n"

		"SAVE_C_1x32:                                                \n"
		"   SAVE_1x32                                                \n"

		"   cmpq      $1, %%rdi                                      \n"
		"    jnb     BEGIN_M1                                        \n" // 不小于（或等于）则跳转

		//----------------------------------------------------------------

		"END_M:                                                      \n"

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
		[Bc] "m"(Bc),
		[k_tag] "m"(k_tag)
		: "rax", "rbx", "rcx", "rdx", "rdi", "rsi", "rbp", "r8", "r9", "r10", "r11", "r12",
		  "r13", "r14", "r15", "zmm0", "zmm1", "zmm2", "zmm3", "zmm4", "zmm5",
		  "zmm6", "zmm7", "zmm8", "zmm9", "zmm10", "zmm11", "zmm12", "zmm13",
		  "zmm14", "zmm15", "zmm16", "zmm17", "zmm18", "zmm19", "zmm20", "zmm21",
		  "zmm22", "zmm23", "zmm24", "zmm25", "zmm26", "zmm27", "zmm28", "zmm29",
		  "zmm30", "zmm31", "memory"

	);
}

void SMM_NN_KERNELm12xn16(float *C, float *A, float *B, long M,
						  long N, long K, long LN, long LK, float *Bc, long k_tag)
{

	asm volatile(
		".macro    KERNEL12x16_PACK_K1                               \n"

		"   vbroadcastss    8(%%rax), %%zmm2                         \n"
		"   vfmadd231ps        %%zmm0, %%zmm4, %%zmm20               \n"

		"   vbroadcastss    12(%%rax), %%zmm3                        \n"
		"   vfmadd231ps        %%zmm1, %%zmm4, %%zmm21               \n"

		"   prefetcht0         256(%%rax)                            \n"

		"   vbroadcastss    16(%%rax), %%zmm0                        \n"
		"   vfmadd231ps        %%zmm2, %%zmm4, %%zmm22               \n"

		"   vbroadcastss    20(%%rax), %%zmm1                        \n"
		"   vfmadd231ps        %%zmm3, %%zmm4, %%zmm23               \n"

		"   vbroadcastss    24(%%rax), %%zmm2                        \n"
		"   vfmadd231ps        %%zmm0, %%zmm4, %%zmm24               \n"

		"   vbroadcastss    28(%%rax), %%zmm3                        \n"
		"   vfmadd231ps        %%zmm1, %%zmm4, %%zmm25               \n"

		"   vbroadcastss    32(%%rax), %%zmm0                        \n"
		"   vfmadd231ps        %%zmm2, %%zmm4, %%zmm26               \n"

		"    leaq      (%%rbx, %%r8, 4), %%rbx                       \n" // B

		"   vbroadcastss    36(%%rax), %%zmm1                        \n"
		"   vfmadd231ps        %%zmm3, %%zmm4, %%zmm27               \n"

		"   vbroadcastss    40(%%rax), %%zmm2                        \n"
		"   vfmadd231ps        %%zmm0, %%zmm4, %%zmm28               \n"

		"   vbroadcastss    44(%%rax), %%zmm3                        \n"
		"   vfmadd231ps        %%zmm1, %%zmm4, %%zmm29               \n"
		"   vmovups         (%%rbx), %%zmm6                          \n"
		"    addq              $48, %%rax                            \n"

		"   vbroadcastss    (%%rax), %%zmm0                          \n"
		"   vfmadd231ps        %%zmm2, %%zmm4, %%zmm30               \n"

		"   vbroadcastss    4(%%rax), %%zmm1                         \n"
		"   vfmadd231ps        %%zmm3, %%zmm4, %%zmm31               \n"
		"   vmovups         %%zmm4, (%%rbp)                          \n"
		"    addq              $64, %%rbp                            \n"

		".endm                                                       \n"

		".macro    KERNEL12x16_PACK_K2                               \n"

		"   vbroadcastss    8(%%rax), %%zmm2                         \n"
		"   vfmadd231ps        %%zmm0, %%zmm6, %%zmm20               \n"

		"   vbroadcastss    12(%%rax), %%zmm3                        \n"
		"   vfmadd231ps        %%zmm1, %%zmm6, %%zmm21               \n"

		"   vbroadcastss    16(%%rax), %%zmm0                        \n"
		"   vfmadd231ps        %%zmm2, %%zmm6, %%zmm22               \n"

		"   prefetcht0         256(%%rax)                            \n"

		"   vbroadcastss    20(%%rax), %%zmm1                        \n"
		"   vfmadd231ps        %%zmm3, %%zmm6, %%zmm23               \n"

		"   vbroadcastss    24(%%rax), %%zmm2                        \n"
		"   vfmadd231ps        %%zmm0, %%zmm6, %%zmm24               \n"

		"   vbroadcastss    28(%%rax), %%zmm3                        \n"
		"   vfmadd231ps        %%zmm1, %%zmm6, %%zmm25               \n"

		"   vbroadcastss    32(%%rax), %%zmm0                        \n"
		"   vfmadd231ps        %%zmm2, %%zmm6, %%zmm26               \n"

		"    leaq      (%%rbx, %%r8, 4), %%rbx                       \n" // B

		"   vbroadcastss    36(%%rax), %%zmm1                        \n"
		"   vfmadd231ps        %%zmm3, %%zmm6, %%zmm27               \n"

		"   prefetcht0         64(%%rbx)                             \n"

		"   vbroadcastss    40(%%rax), %%zmm2                        \n"
		"   vfmadd231ps        %%zmm0, %%zmm6, %%zmm28               \n"

		"   vbroadcastss    44(%%rax), %%zmm3                        \n"
		"   vfmadd231ps        %%zmm1, %%zmm6, %%zmm29               \n"
		"   vmovups         (%%rbx), %%zmm4                          \n"
		"    addq              $48, %%rax                            \n"

		"   vbroadcastss    (%%rax), %%zmm0                          \n"
		"   vfmadd231ps        %%zmm2, %%zmm6, %%zmm30               \n"

		"   vbroadcastss    4(%%rax), %%zmm1                         \n"
		"   vfmadd231ps        %%zmm3, %%zmm6, %%zmm31               \n"
		"   vmovups         %%zmm6, (%%rbp)                          \n"
		"    addq              $64, %%rbp                            \n"

		".endm                                                       \n"

		".macro    KERNEL12x16_PACK_END_K                            \n"

		"   vbroadcastss    8(%%rax), %%zmm2                         \n"
		"   vfmadd231ps        %%zmm0, %%zmm6, %%zmm20               \n"

		"   vbroadcastss    12(%%rax), %%zmm3                        \n"
		"   vfmadd231ps        %%zmm1, %%zmm6, %%zmm21               \n"

		"   vbroadcastss    16(%%rax), %%zmm0                        \n"
		"   vfmadd231ps        %%zmm2, %%zmm6, %%zmm22               \n"

		"   prefetcht0         256(%%rax)                            \n"

		"   vbroadcastss    20(%%rax), %%zmm1                        \n"
		"   vfmadd231ps        %%zmm3, %%zmm6, %%zmm23               \n"

		"   vbroadcastss    24(%%rax), %%zmm2                        \n"
		"   vfmadd231ps        %%zmm0, %%zmm6, %%zmm24               \n"

		"   vbroadcastss    28(%%rax), %%zmm3                        \n"
		"   vfmadd231ps        %%zmm1, %%zmm6, %%zmm25               \n"

		"   vbroadcastss    32(%%rax), %%zmm0                        \n"
		"   vfmadd231ps        %%zmm2, %%zmm6, %%zmm26               \n"

		"   vbroadcastss    36(%%rax), %%zmm1                        \n"
		"   vfmadd231ps        %%zmm3, %%zmm6, %%zmm27               \n"

		"   vbroadcastss    40(%%rax), %%zmm2                        \n"
		"   vfmadd231ps        %%zmm0, %%zmm6, %%zmm28               \n"

		"   vbroadcastss    44(%%rax), %%zmm3                        \n"
		"   vfmadd231ps        %%zmm1, %%zmm6, %%zmm29               \n"
		"    addq              $48, %%rax                            \n"

		"   vfmadd231ps        %%zmm2, %%zmm6, %%zmm30               \n"
		"   vmovups         %%zmm6, (%%rbp)                          \n"
		"   vfmadd231ps        %%zmm3, %%zmm6, %%zmm31               \n"

		".endm                                                       \n"

		//-----------------------------------------------------------------------

		".macro    KERNEL12x16_K1                                    \n"

		"   vbroadcastss    8(%%rax), %%zmm2                         \n"
		"   vfmadd231ps        %%zmm0, %%zmm4, %%zmm20               \n"

		"   vbroadcastss    12(%%rax), %%zmm3                        \n"
		"   vfmadd231ps        %%zmm1, %%zmm4, %%zmm21               \n"

		"   prefetcht0         256(%%rax)                            \n"

		"   vbroadcastss    16(%%rax), %%zmm0                        \n"
		"   vfmadd231ps        %%zmm2, %%zmm4, %%zmm22               \n"

		"   vbroadcastss    20(%%rax), %%zmm1                        \n"
		"   vfmadd231ps        %%zmm3, %%zmm4, %%zmm23               \n"

		"   vbroadcastss    24(%%rax), %%zmm2                        \n"
		"   vfmadd231ps        %%zmm0, %%zmm4, %%zmm24               \n"

		"   vbroadcastss    28(%%rax), %%zmm3                        \n"
		"   vfmadd231ps        %%zmm1, %%zmm4, %%zmm25               \n"

		"   vbroadcastss    32(%%rax), %%zmm0                        \n"
		"   vfmadd231ps        %%zmm2, %%zmm4, %%zmm26               \n"

		"    addq              $64, %%rbx                            \n" // B

		"   vbroadcastss    36(%%rax), %%zmm1                        \n"
		"   vfmadd231ps        %%zmm3, %%zmm4, %%zmm27               \n"

		"   vbroadcastss    40(%%rax), %%zmm2                        \n"
		"   vfmadd231ps        %%zmm0, %%zmm4, %%zmm28               \n"

		"   vbroadcastss    44(%%rax), %%zmm3                        \n"
		"   vfmadd231ps        %%zmm1, %%zmm4, %%zmm29               \n"
		"   vmovups         (%%rbx), %%zmm6                          \n"
		"    addq              $48, %%rax                            \n"

		"   vbroadcastss    (%%rax), %%zmm0                          \n"
		"   vfmadd231ps        %%zmm2, %%zmm4, %%zmm30               \n"

		"   vbroadcastss    4(%%rax), %%zmm1                         \n"
		"   vfmadd231ps        %%zmm3, %%zmm4, %%zmm31               \n"

		".endm                                                       \n"

		".macro    KERNEL12x16_K2                                    \n"

		"   vbroadcastss    8(%%rax), %%zmm2                         \n"
		"   vfmadd231ps        %%zmm0, %%zmm6, %%zmm20               \n"

		"   vbroadcastss    12(%%rax), %%zmm3                        \n"
		"   vfmadd231ps        %%zmm1, %%zmm6, %%zmm21               \n"

		"   vbroadcastss    16(%%rax), %%zmm0                        \n"
		"   vfmadd231ps        %%zmm2, %%zmm6, %%zmm22               \n"

		"   prefetcht0         256(%%rax)                            \n"

		"   vbroadcastss    20(%%rax), %%zmm1                        \n"
		"   vfmadd231ps        %%zmm3, %%zmm6, %%zmm23               \n"

		"   vbroadcastss    24(%%rax), %%zmm2                        \n"
		"   vfmadd231ps        %%zmm0, %%zmm6, %%zmm24               \n"

		"   vbroadcastss    28(%%rax), %%zmm3                        \n"
		"   vfmadd231ps        %%zmm1, %%zmm6, %%zmm25               \n"

		"   vbroadcastss    32(%%rax), %%zmm0                        \n"
		"   vfmadd231ps        %%zmm2, %%zmm6, %%zmm26               \n"

		"    addq             $64, %%rbx                             \n"

		"   vbroadcastss    36(%%rax), %%zmm1                        \n"
		"   vfmadd231ps        %%zmm3, %%zmm6, %%zmm27               \n"

		"   prefetcht0         64(%%rbx)                             \n"

		"   vbroadcastss    40(%%rax), %%zmm2                        \n"
		"   vfmadd231ps        %%zmm0, %%zmm6, %%zmm28               \n"

		"   vbroadcastss    44(%%rax), %%zmm3                        \n"
		"   vfmadd231ps        %%zmm1, %%zmm6, %%zmm29               \n"
		"   vmovups         (%%rbx), %%zmm4                          \n"
		"    addq              $48, %%rax                            \n"

		"   vbroadcastss    (%%rax), %%zmm0                          \n"
		"   vfmadd231ps        %%zmm2, %%zmm6, %%zmm30               \n"

		"   vbroadcastss    4(%%rax), %%zmm1                         \n"
		"   vfmadd231ps        %%zmm3, %%zmm6, %%zmm31               \n"

		".endm                                                       \n"

		".macro    KERNEL12x16_END_K                                 \n"

		"   vbroadcastss    8(%%rax), %%zmm2                         \n"
		"   vfmadd231ps        %%zmm0, %%zmm6, %%zmm20               \n"

		"   vbroadcastss    12(%%rax), %%zmm3                        \n"
		"   vfmadd231ps        %%zmm1, %%zmm6, %%zmm21               \n"

		"   vbroadcastss    16(%%rax), %%zmm0                        \n"
		"   vfmadd231ps        %%zmm2, %%zmm6, %%zmm22               \n"

		"   prefetcht0         256(%%rax)                            \n"

		"   vbroadcastss    20(%%rax), %%zmm1                        \n"
		"   vfmadd231ps        %%zmm3, %%zmm6, %%zmm23               \n"

		"   vbroadcastss    24(%%rax), %%zmm2                        \n"
		"   vfmadd231ps        %%zmm0, %%zmm6, %%zmm24               \n"

		"   vbroadcastss    28(%%rax), %%zmm3                        \n"
		"   vfmadd231ps        %%zmm1, %%zmm6, %%zmm25               \n"

		"   vbroadcastss    32(%%rax), %%zmm0                        \n"
		"   vfmadd231ps        %%zmm2, %%zmm6, %%zmm26               \n"

		"   vbroadcastss    36(%%rax), %%zmm1                        \n"
		"   vfmadd231ps        %%zmm3, %%zmm6, %%zmm27               \n"

		"   vbroadcastss    40(%%rax), %%zmm2                        \n"
		"   vfmadd231ps        %%zmm0, %%zmm6, %%zmm28               \n"

		"   vbroadcastss    44(%%rax), %%zmm3                        \n"
		"   vfmadd231ps        %%zmm1, %%zmm6, %%zmm29               \n"
		"   vmovups         (%%rbx), %%zmm4                          \n"
		"    addq              $48, %%rax                            \n"

		"   vbroadcastss    (%%rax), %%zmm0                          \n"
		"   vfmadd231ps        %%zmm2, %%zmm6, %%zmm30               \n"

		"   vbroadcastss    4(%%rax), %%zmm1                         \n"
		"   vfmadd231ps        %%zmm3, %%zmm6, %%zmm31               \n"

		".endm                                                       \n"

		".macro    ADD_C_12x16                                       \n"

		"   vmovups         (%%r10), %%zmm0                          \n"
		"    vaddps             %%zmm0, %%zmm20, %%zmm20             \n"
		"   vmovups         (%%r11), %%zmm1                          \n"
		"    vaddps             %%zmm1, %%zmm21, %%zmm21             \n"
		"   vmovups         (%%r12), %%zmm2                          \n"
		"    vaddps             %%zmm2, %%zmm22, %%zmm22             \n"
		"   vmovups         (%%r13), %%zmm3                          \n"
		"    vaddps             %%zmm3, %%zmm23, %%zmm23             \n"

		"    leaq              (%%r13, %%r8, 4), %%r10               \n" // C0
		"    leaq             (%%r10, %%r8, 4), %%r11                \n" // C1
		"    leaq             (%%r11, %%r8, 4), %%r12                \n" // C2
		"    leaq             (%%r12, %%r8, 4), %%r13                \n" // C3

		"   vmovups         (%%r10), %%zmm4                          \n"
		"    vaddps             %%zmm4, %%zmm24, %%zmm24             \n"
		"   vmovups         (%%r11), %%zmm5                          \n"
		"    vaddps             %%zmm5, %%zmm25, %%zmm25             \n"
		"   vmovups         (%%r12), %%zmm6                          \n"
		"    vaddps             %%zmm6, %%zmm26, %%zmm26             \n"
		"   vmovups         (%%r13), %%zmm7                          \n"
		"    vaddps             %%zmm7, %%zmm27, %%zmm27             \n"

		"    leaq              (%%r13, %%r8, 4), %%r10               \n" // C0
		"    leaq             (%%r10, %%r8, 4), %%r11                \n" // C1
		"    leaq             (%%r11, %%r8, 4), %%r12                \n" // C2
		"    leaq             (%%r12, %%r8, 4), %%r13                \n" // C3

		"   vmovups         (%%r10), %%zmm0                          \n"
		"    vaddps             %%zmm0, %%zmm28, %%zmm28             \n"
		"   vmovups         (%%r11), %%zmm1                          \n"
		"    vaddps             %%zmm1, %%zmm29, %%zmm29             \n"
		"   vmovups         (%%r12), %%zmm2                          \n"
		"    vaddps             %%zmm2, %%zmm30, %%zmm30             \n"
		"   vmovups         (%%r13), %%zmm3                          \n"
		"    vaddps             %%zmm3, %%zmm31, %%zmm31             \n"

		"    mov      %%rcx, %%r10                                   \n" // C0
		"    leaq     (%%r10, %%r8, 4), %%r11                        \n" // C1
		"    leaq     (%%r11, %%r8, 4), %%r12                        \n" // C2
		"    leaq     (%%r12, %%r8, 4), %%r13                        \n" // C3

		".endm                                                       \n"

		".macro    SAVE_12x16                                        \n"

		"   vmovups         %%zmm20, (%%r10)                         \n"
		"   vmovups         %%zmm21, (%%r11)                         \n"
		"    subq             $12, %%rdi                             \n"
		"   vmovups         %%zmm22, (%%r12)                         \n"
		"   vmovups         %%zmm23, (%%r13)                         \n"

		"    leaq      (%%r13, %%r8, 4), %%r10                       \n" // C0
		"    leaq     (%%r10, %%r8, 4), %%r11                        \n" // C1
		"    leaq     (%%r11, %%r8, 4), %%r12                        \n" // C2
		"    leaq     (%%r12, %%r8, 4), %%r13                        \n" // C3

		"   vmovups         %%zmm24, (%%r10)                         \n"
		"   vmovups         %%zmm25, (%%r11)                         \n"
		"   vmovups         %%zmm26, (%%r12)                         \n"
		"   vmovups         %%zmm27, (%%r13)                         \n"

		"    leaq      (%%r13, %%r8, 4), %%r10                       \n" // C0
		"    leaq     (%%r10, %%r8, 4), %%r11                        \n" // C1
		"    leaq     (%%r11, %%r8, 4), %%r12                        \n" // C2
		"    leaq     (%%r12, %%r8, 4), %%r13                        \n" // C3

		"   vmovups         %%zmm28, (%%r10)                         \n"
		"   vmovups         %%zmm29, (%%r11)                         \n"
		"   vmovups         %%zmm30, (%%r12)                         \n"
		"   vmovups         %%zmm31, (%%r13)                         \n"

		"    leaq      (%%r13, %%r8, 4), %%rcx                       \n" // C0

		".endm                                                       \n"

		//-----------------------------------------------------------------------

		".macro    KERNEL8x16_K1                                     \n"

		"   vbroadcastss    8(%%rax), %%zmm2                         \n"
		"   vfmadd231ps        %%zmm0, %%zmm4, %%zmm24               \n"

		"   vbroadcastss    12(%%rax), %%zmm3                        \n"
		"   vfmadd231ps        %%zmm1, %%zmm4, %%zmm25               \n"

		"   prefetcht0         256(%%rax)                            \n"

		"   vbroadcastss    16(%%rax), %%zmm8                        \n"
		"   vfmadd231ps        %%zmm2, %%zmm4, %%zmm26               \n"

		"   vbroadcastss    20(%%rax), %%zmm9                        \n"
		"   vfmadd231ps        %%zmm3, %%zmm4, %%zmm27               \n"

		"   vbroadcastss    24(%%rax), %%zmm10                       \n"
		"   vfmadd231ps        %%zmm8, %%zmm4, %%zmm28               \n"

		"   prefetcht0         64(%%rbx)                             \n"

		"   vbroadcastss    28(%%rax), %%zmm11                       \n"
		"   vfmadd231ps        %%zmm9, %%zmm4, %%zmm29               \n"
		"    addq              $64, %%rbx                            \n" // B
		"   vfmadd231ps        %%zmm10, %%zmm4, %%zmm30              \n"

		"    addq              $32, %%rax                            \n"
		"   vmovups         (%%rbx), %%zmm6                          \n"
		"   vfmadd231ps        %%zmm11, %%zmm4, %%zmm31              \n"

		"   vbroadcastss    (%%rax), %%zmm0                          \n"
		"   vbroadcastss    4(%%rax), %%zmm1                         \n"

		".endm                                                       \n"

		".macro    KERNEL8x16_K2                                     \n"

		"   vbroadcastss    8(%%rax), %%zmm2                         \n"
		"   vfmadd231ps        %%zmm0, %%zmm6, %%zmm24               \n"

		"   vbroadcastss    12(%%rax), %%zmm3                        \n"
		"   vfmadd231ps        %%zmm1, %%zmm6, %%zmm25               \n"

		"   prefetcht0         256(%%rax)                            \n"

		"   vbroadcastss    16(%%rax), %%zmm8                        \n"
		"   vfmadd231ps        %%zmm2, %%zmm6, %%zmm26               \n"

		"   vbroadcastss    20(%%rax), %%zmm9                        \n"
		"   vfmadd231ps        %%zmm3, %%zmm6, %%zmm27               \n"

		"   vbroadcastss    24(%%rax), %%zmm10                       \n"
		"   vfmadd231ps        %%zmm8, %%zmm6, %%zmm28               \n"

		"   prefetcht0         64(%%rbx)                             \n"

		"   vbroadcastss    28(%%rax), %%zmm11                       \n"
		"   vfmadd231ps        %%zmm9, %%zmm6, %%zmm29               \n"
		"    addq              $64, %%rbx                            \n" // B
		"   vfmadd231ps        %%zmm10, %%zmm6, %%zmm30              \n"

		"    addq              $32, %%rax                            \n"
		"   vmovups         (%%rbx), %%zmm4                          \n"
		"   vfmadd231ps        %%zmm11, %%zmm6, %%zmm31              \n"

		"   vbroadcastss    (%%rax), %%zmm0                          \n"
		"   vbroadcastss    4(%%rax), %%zmm1                         \n"

		".endm                                                       \n"

		".macro    KERNEL8x16_END_K                                  \n"

		"   vbroadcastss    8(%%rax), %%zmm2                         \n"
		"   vfmadd231ps        %%zmm0, %%zmm6, %%zmm24               \n"

		"   vbroadcastss    12(%%rax), %%zmm3                        \n"
		"   vfmadd231ps        %%zmm1, %%zmm6, %%zmm25               \n"

		"   prefetcht0         256(%%rax)                            \n"

		"   vbroadcastss    16(%%rax), %%zmm8                        \n"
		"   vfmadd231ps        %%zmm2, %%zmm6, %%zmm26               \n"

		"   vbroadcastss    20(%%rax), %%zmm9                        \n"
		"   vfmadd231ps        %%zmm3, %%zmm6, %%zmm27               \n"

		"   vbroadcastss    24(%%rax), %%zmm10                       \n"
		"   vfmadd231ps        %%zmm8, %%zmm6, %%zmm28               \n"

		"   vbroadcastss    28(%%rax), %%zmm11                       \n"
		"   vfmadd231ps        %%zmm9, %%zmm6, %%zmm29               \n"
		"   vfmadd231ps        %%zmm10, %%zmm6, %%zmm30              \n"

		"    addq              $32, %%rax                            \n"
		"   vfmadd231ps        %%zmm11, %%zmm6, %%zmm31              \n"

		".endm                                                       \n"

		".macro    ADD_C_8x16                                        \n"

		"   vmovups         (%%r10), %%zmm4                          \n"
		"    vaddps             %%zmm4, %%zmm24, %%zmm24             \n"
		"   vmovups         (%%r11), %%zmm5                          \n"
		"    vaddps             %%zmm5, %%zmm25, %%zmm25             \n"
		"   vmovups         (%%r12), %%zmm6                          \n"
		"    vaddps             %%zmm6, %%zmm26, %%zmm26             \n"
		"   vmovups         (%%r13), %%zmm7                          \n"
		"    vaddps             %%zmm7, %%zmm27, %%zmm27             \n"

		"    leaq              (%%r13, %%r8, 4), %%r10               \n" // C0
		"    leaq             (%%r10, %%r8, 4), %%r11                \n" // C1
		"    leaq             (%%r11, %%r8, 4), %%r12                \n" // C2
		"    leaq             (%%r12, %%r8, 4), %%r13                \n" // C3

		"   vmovups         (%%r10), %%zmm0                          \n"
		"    vaddps             %%zmm0, %%zmm28, %%zmm28             \n"
		"   vmovups         (%%r11), %%zmm1                          \n"
		"    vaddps             %%zmm1, %%zmm29, %%zmm29             \n"
		"   vmovups         (%%r12), %%zmm2                          \n"
		"    vaddps             %%zmm2, %%zmm30, %%zmm30             \n"
		"   vmovups         (%%r13), %%zmm3                          \n"
		"    vaddps             %%zmm3, %%zmm31, %%zmm31             \n"

		"    mov      %%rcx, %%r10                                   \n" // C0
		"    leaq     (%%r10, %%r8, 4), %%r11                        \n" // C1
		"    leaq     (%%r11, %%r8, 4), %%r12                        \n" // C2
		"    leaq     (%%r12, %%r8, 4), %%r13                        \n" // C3

		".endm                                                       \n"

		".macro    SAVE_8x16                                         \n"

		"   vmovups         %%zmm24, (%%r10)                         \n"
		"   vmovups         %%zmm25, (%%r11)                         \n"
		"    subq             $8, %%rdi                              \n"
		"   vmovups         %%zmm26, (%%r12)                         \n"
		"   vmovups         %%zmm27, (%%r13)                         \n"

		"    leaq      (%%r13, %%r8, 4), %%r10                       \n" // C0
		"    leaq     (%%r10, %%r8, 4), %%r11                        \n" // C1
		"    leaq     (%%r11, %%r8, 4), %%r12                        \n" // C2
		"    leaq     (%%r12, %%r8, 4), %%r13                        \n" // C3

		"   vmovups         %%zmm28, (%%r10)                         \n"
		"   vmovups         %%zmm29, (%%r11)                         \n"
		"   vmovups         %%zmm30, (%%r12)                         \n"
		"   vmovups         %%zmm31, (%%r13)                         \n"

		"    leaq      (%%r13, %%r8, 4), %%rcx                       \n" // C0

		".endm                                                       \n"

		//-----------------------------------------------------------------------

		".macro    KERNEL4x16_K1                                     \n"

		"   vbroadcastss    8(%%rax), %%zmm2                         \n"
		"   vfmadd231ps        %%zmm0, %%zmm4, %%zmm28               \n"

		"   vbroadcastss    12(%%rax), %%zmm3                        \n"
		"   vfmadd231ps        %%zmm1, %%zmm4, %%zmm29               \n"
		"    addq              $16, %%rax                            \n"

		"   prefetcht0         256(%%rax)                            \n"
		"    addq              $64, %%rbx                            \n" // B
		"   vbroadcastss    (%%rax), %%zmm0                          \n"
		"   vfmadd231ps        %%zmm2, %%zmm4, %%zmm30               \n"

		"   vbroadcastss    4(%%rax), %%zmm1                         \n"
		"   vfmadd231ps        %%zmm3, %%zmm4, %%zmm31               \n"

		"   prefetcht0         64(%%rbx)                             \n"
		"   vmovups         (%%rbx), %%zmm6                          \n"

		".endm                                                       \n"

		".macro    KERNEL4x16_K2                                     \n"

		"   vbroadcastss    8(%%rax), %%zmm2                         \n"
		"   vfmadd231ps        %%zmm0, %%zmm6, %%zmm28               \n"

		"   vbroadcastss    12(%%rax), %%zmm3                        \n"
		"   vfmadd231ps        %%zmm1, %%zmm6, %%zmm29               \n"
		"    addq              $16, %%rax                            \n"

		"   prefetcht0         256(%%rax)                            \n"
		"    addq              $64, %%rbx                            \n" // B
		"   vbroadcastss    (%%rax), %%zmm0                          \n"
		"   vfmadd231ps        %%zmm2, %%zmm6, %%zmm30               \n"

		"   vbroadcastss    4(%%rax), %%zmm1                         \n"
		"   vfmadd231ps        %%zmm3, %%zmm6, %%zmm31               \n"

		"   vmovups         (%%rbx), %%zmm4                          \n"
		"   prefetcht0         64(%%rbx)                             \n"

		".endm                                                       \n"

		".macro    KERNEL4x16_END_K                                  \n"

		"   vbroadcastss    8(%%rax), %%zmm2                         \n"
		"   vfmadd231ps        %%zmm0, %%zmm6, %%zmm28               \n"

		"   vbroadcastss    12(%%rax), %%zmm3                        \n"
		"   vfmadd231ps        %%zmm1, %%zmm6, %%zmm29               \n"
		"    addq              $16, %%rax                            \n"

		"   vfmadd231ps        %%zmm2, %%zmm6, %%zmm30               \n"
		"   vfmadd231ps        %%zmm3, %%zmm6, %%zmm31               \n"

		".endm                                                       \n"

		".macro    ADD_C_4x16                                        \n"

		"   vmovups         (%%r10), %%zmm0                          \n"
		"    vaddps             %%zmm0, %%zmm28, %%zmm28             \n"
		"   vmovups         (%%r11), %%zmm1                          \n"
		"    vaddps             %%zmm1, %%zmm29, %%zmm29             \n"
		"   vmovups         (%%r12), %%zmm2                          \n"
		"    vaddps             %%zmm2, %%zmm30, %%zmm30             \n"
		"   vmovups         (%%r13), %%zmm3                          \n"
		"    vaddps             %%zmm3, %%zmm31, %%zmm31             \n"

		"    mov      %%rcx, %%r10                                   \n" // C0
		"    leaq     (%%r10, %%r8, 4), %%r11                        \n" // C1
		"    leaq     (%%r11, %%r8, 4), %%r12                        \n" // C2
		"    leaq     (%%r12, %%r8, 4), %%r13                        \n" // C3

		".endm                                                       \n"

		".macro    SAVE_4x16                                         \n"
		"   subq             $4, %%rdi                               \n"

		"   vmovups         %%zmm28, (%%r10)                         \n"
		"   vmovups         %%zmm29, (%%r11)                         \n"
		"   vmovups         %%zmm30, (%%r12)                         \n"
		"   vmovups         %%zmm31, (%%r13)                         \n"

		"    leaq      (%%r13, %%r8, 4), %%rcx                       \n" // C0

		".endm                                                       \n"

		//-----------------------------------------------------------------------

		".macro    KERNEL1x16_K1                                     \n"
		"   prefetcht0         256(%%rax)                            \n"
		"   addq              $4, %%rax                              \n"

		"   vbroadcastss    (%%rax), %%zmm2                          \n"
		"   vfmadd231ps        %%zmm0, %%zmm4, %%zmm30               \n"

		"    addq              $64, %%rbx                            \n" // B
		"   vmovups            (%%rbx), %%zmm6                       \n"
		"   prefetcht0         64(%%rbx)                             \n"

		".endm                                                       \n"

		".macro    KERNEL1x16_K2                                     \n"
		"   prefetcht0         256(%%rax)                            \n"
		"    addq              $4, %%rax                             \n"

		"   vbroadcastss    (%%rax), %%zmm0                          \n"
		"   vfmadd231ps        %%zmm2, %%zmm6, %%zmm30               \n"

		"    addq              $64, %%rbx                            \n" // B
		"   vmovups            (%%rbx), %%zmm4                       \n"
		"   prefetcht0         64(%%rbx)                             \n"

		".endm                                                       \n"

		".macro    KERNEL1x16_END_K                                  \n"
		"   vfmadd231ps        %%zmm2, %%zmm6, %%zmm30               \n"
		"    addq              $4, %%rax                             \n"
		".endm                                                       \n"

		".macro    ADD_C_1x16                                        \n"
		"   vmovups         (%%r10), %%zmm2                          \n"
		"    vaddps             %%zmm2, %%zmm30, %%zmm30             \n"
		"    mov      %%rcx, %%r10                                   \n" // C0
		".endm                                                       \n"

		".macro    SAVE_1x16                                         \n"
		"   vmovups         %%zmm30, (%%r10)                         \n"
		"   subq             $1, %%rdi                               \n"
		"    leaq      (%%r10, %%r8, 4), %%rcx                       \n" // C0

		".endm                                                       \n"

		//-----------------------------------------------------------------

		"SMM_NN_KERNEL12x16:                                         \n"

		"   mov     %[C], %%rcx                                      \n"
		"   mov     %[A], %%rax                                      \n"
		"   mov     %[B], %%rbx                                      \n"

		"   prefetcht0         (%%rax)                               \n"

		"    mov     %[K], %%rdx                                     \n" // K
		"    mov      %[LN], %%r8                                    \n"
		"    mov      %[Bc], %%r14                                   \n"
		"    mov      %[M], %%rdi                                    \n"
		"    mov     %[k_tag], %%r15                                 \n"

		"   prefetcht0         (%%rbx)                               \n"
		"    mov     %%rbx, %%r9                                     \n" // B
		"    mov     %%rdx, %%rsi                                    \n" // K

		"BEGIN_PACK12x16:                                            \n"

		"    mov     %%r9, %%rbx                                     \n" // B
		"   prefetcht0         (%%rbx)                               \n"

		"    mov      %%rcx, %%r10                                   \n" // C0
		"   prefetcht1         (%%r10)                               \n"
		"    leaq     (%%r10, %%r8, 4), %%r11                        \n" // C1
		"   prefetcht1         (%%r11)                               \n"
		"    leaq     (%%r11, %%r8, 4), %%r12                        \n" // C2
		"   prefetcht1         (%%r12)                               \n"
		"    leaq     (%%r12, %%r8, 4), %%r13                        \n" // C3
		"   prefetcht1         (%%r13)                               \n"

		"    mov     %%rsi, %%rdx                                    \n" // K
		"    mov     %%r14, %%rbp                                    \n" // Bc

		"   vmovups        (%%rbx), %%zmm4                           \n"
		"    vpxorq         %%zmm20, %%zmm20, %%zmm20                \n"
		"    vpxorq         %%zmm21, %%zmm21, %%zmm21                \n"
		"   vbroadcastss    (%%rax), %%zmm0                          \n"
		"    vpxorq         %%zmm22, %%zmm22, %%zmm22                \n"
		"    vpxorq         %%zmm23, %%zmm23, %%zmm23                \n"
		"   vbroadcastss    4(%%rax), %%zmm1                         \n"
		"    vpxorq         %%zmm24, %%zmm24, %%zmm24                \n"
		"    vpxorq         %%zmm25, %%zmm25, %%zmm25                \n"
		"    vpxorq         %%zmm26, %%zmm26, %%zmm26                \n"
		"    vpxorq         %%zmm27, %%zmm27, %%zmm27                \n"
		"    vpxorq         %%zmm28, %%zmm28, %%zmm28                \n"
		"    vpxorq         %%zmm29, %%zmm29, %%zmm29                \n"
		"    vpxorq         %%zmm30, %%zmm30, %%zmm30                \n"
		"    vpxorq         %%zmm31, %%zmm31, %%zmm31                \n"

		"    subq     $8, %%rdx                                      \n"

		"MAIN_PACK_K12x16:                                           \n"

		"    KERNEL12x16_PACK_K1                                     \n"
		"    KERNEL12x16_PACK_K2                                     \n"
		"    KERNEL12x16_PACK_K1                                     \n"
		"    KERNEL12x16_PACK_K2                                     \n"
		"    KERNEL12x16_PACK_K1                                     \n"
		"    KERNEL12x16_PACK_K2                                     \n"
		"    KERNEL12x16_PACK_K1                                     \n"
		"   cmp     $0, %%rdx                                        \n"
		"    je         EDGE_PACK_K12x16                             \n"
		"    KERNEL12x16_PACK_K2                                     \n"

		"    subq     $8, %%rdx                                      \n"
		"   jmp     MAIN_PACK_K12x16                                 \n"

		"EDGE_PACK_K12x16:                                           \n"

		"    KERNEL12x16_PACK_END_K                                  \n"
		"    jmp      BEGIN_SAVE_12x16                               \n"

		//-----------------------------------------------------------------

		"BEGIN_M12x16:                                               \n"

		"    mov     %%r14, %%rbx                                    \n" // Bc
		"   prefetcht0         (%%rbx)                               \n"

		"    mov      %%rcx, %%r10                                   \n" // C0
		"   prefetcht1         (%%r10)                               \n"
		"    leaq     (%%r10, %%r8, 4), %%r11                        \n" // C1
		"   prefetcht1         (%%r11)                               \n"
		"    leaq     (%%r11, %%r8, 4), %%r12                        \n" // C2
		"   prefetcht1         (%%r12)                               \n"
		"    leaq     (%%r12, %%r8, 4), %%r13                        \n" // C3
		"   prefetcht1         (%%r13)                               \n"

		"    mov     %%rsi, %%rdx                                    \n" // K

		"   vmovups        (%%rbx), %%zmm4                           \n"
		"    vpxorq         %%zmm20, %%zmm20, %%zmm20                \n"
		"    vpxorq         %%zmm21, %%zmm21, %%zmm21                \n"
		"    vpxorq         %%zmm22, %%zmm22, %%zmm22                \n"
		"   vbroadcastss    (%%rax), %%zmm0                          \n"
		"    vpxorq         %%zmm23, %%zmm23, %%zmm23                \n"
		"    vpxorq         %%zmm24, %%zmm24, %%zmm24                \n"
		"   vbroadcastss    4(%%rax), %%zmm1                         \n"
		"    vpxorq         %%zmm25, %%zmm25, %%zmm25                \n"
		"    vpxorq         %%zmm26, %%zmm26, %%zmm26                \n"
		"    vpxorq         %%zmm27, %%zmm27, %%zmm27                \n"
		"    vpxorq         %%zmm28, %%zmm28, %%zmm28                \n"
		"    vpxorq         %%zmm29, %%zmm29, %%zmm29                \n"
		"    vpxorq         %%zmm30, %%zmm30, %%zmm30                \n"
		"    vpxorq         %%zmm31, %%zmm31, %%zmm31                \n"

		"    subq     $8, %%rdx                                      \n"

		"MAIN_K12x16:                                                \n"

		"    KERNEL12x16_K1                                          \n"
		"    KERNEL12x16_K2                                          \n"
		"    KERNEL12x16_K1                                          \n"
		"    KERNEL12x16_K2                                          \n"
		"    KERNEL12x16_K1                                          \n"
		"    KERNEL12x16_K2                                          \n"
		"    KERNEL12x16_K1                                          \n"
		"   cmp     $0, %%rdx                                        \n"
		"    je         EDGE_K12x16                                  \n"
		"    KERNEL12x16_K2                                          \n"

		"    subq     $8, %%rdx                                      \n"
		"   jmp     MAIN_K12x16                                      \n"

		"EDGE_K12x16:                                                \n"

		"    KERNEL12x16_END_K                                       \n"

		"BEGIN_SAVE_12x16:                                           \n"
		"    cmp     $0, %%r15                                       \n"
		"    je      SAVE_C12x16                                     \n"
		"    ADD_C_12x16                                             \n"

		"SAVE_C12x16:                                                \n"
		"    SAVE_12x16                                              \n"

		"    cmpq      $12, %%rdi                                    \n"
		"    jnb     BEGIN_M12x16                                    \n" // 不小于（或等于）则跳转

		//------------------------------------------------------------------

		"BEGIN_M8_N16:                                               \n"
		"   cmpq      $8, %%rdi                                      \n" // M % 8
		"    jb       BEGIN_M4_N16                                   \n" // 小于则跳转

		"    mov     %%r14, %%rbx                                    \n" // Bc
		"   prefetcht0         (%%rbx)                               \n"

		"    mov      %%rcx, %%r10                                   \n" // C0
		"   prefetcht1         (%%r10)                               \n"
		"    leaq     (%%r10, %%r8, 4), %%r11                        \n" // C1
		"   prefetcht1         (%%r11)                               \n"
		"    leaq     (%%r11, %%r8, 4), %%r12                        \n" // C2
		"   prefetcht1         (%%r12)                               \n"
		"    leaq     (%%r12, %%r8, 4), %%r13                        \n" // C3
		"   prefetcht1         (%%r13)                               \n"

		"    mov     %%rsi, %%rdx                                    \n" // K
		"   vmovups        (%%rbx), %%zmm4                           \n"
		"    vpxorq         %%zmm24, %%zmm24, %%zmm24                \n"
		"    vpxorq         %%zmm25, %%zmm25, %%zmm25                \n"
		"   vbroadcastss    (%%rax), %%zmm0                          \n"
		"    vpxorq         %%zmm26, %%zmm26, %%zmm26                \n"
		"    vpxorq         %%zmm27, %%zmm27, %%zmm27                \n"
		"   vbroadcastss    4(%%rax), %%zmm1                         \n"
		"    vpxorq         %%zmm28, %%zmm28, %%zmm28                \n"
		"    vpxorq         %%zmm29, %%zmm29, %%zmm29                \n"
		"    vpxorq         %%zmm30, %%zmm30, %%zmm30                \n"
		"    vpxorq         %%zmm31, %%zmm31, %%zmm31                \n"

		"    subq     $8, %%rdx                                      \n"

		"MAIN_K_M8_N16:                                              \n"

		"    KERNEL8x16_K1                                           \n"
		"    KERNEL8x16_K2                                           \n"
		"    KERNEL8x16_K1                                           \n"
		"    KERNEL8x16_K2                                           \n"
		"    KERNEL8x16_K1                                           \n"
		"    KERNEL8x16_K2                                           \n"
		"    KERNEL8x16_K1                                           \n"
		"   cmp     $0, %%rdx                                        \n"
		"    je         EDGE_K_M8_N16                                \n"
		"    KERNEL8x16_K2                                           \n"

		"    subq     $8, %%rdx                                      \n"
		"   jmp     MAIN_K_M8_N16                                    \n"

		"EDGE_K_M8_N16:                                              \n"

		"    KERNEL8x16_END_K                                        \n"

		"    cmp     $0, %%r15                                       \n"
		"    je      SAVE_C_8x16                                     \n"
		"    ADD_C_8x16                                              \n"

		"SAVE_C_8x16:                                                \n"
		"    SAVE_8x16                                               \n"

		//----------------------------------------------------------------

		"BEGIN_M4_N16:                                               \n"

		"    cmpq      $4, %%rdi                                     \n" // M % 4
		"    jb       BEGIN_M1_N16                                   \n"

		"    mov     %%r14, %%rbx                                    \n" // Bc
		"   prefetcht0         (%%rbx)                               \n"

		"    mov      %%rcx, %%r10                                   \n" // C0
		"   prefetcht1         (%%r10)                               \n"
		"    leaq     (%%r10, %%r8, 4), %%r11                        \n" // C1
		"   prefetcht1         (%%r11)                               \n"
		"    leaq     (%%r11, %%r8, 4), %%r12                        \n" // C2
		"   prefetcht1         (%%r12)                               \n"
		"    leaq     (%%r12, %%r8, 4), %%r13                        \n" // C3
		"   prefetcht1         (%%r13)                               \n"

		"    mov     %%rsi, %%rdx                                    \n" // K

		"   vmovups        (%%rbx), %%zmm4                           \n"
		"   vbroadcastss    (%%rax), %%zmm0                          \n"
		"    vpxorq         %%zmm28, %%zmm28, %%zmm28                \n"
		"    vpxorq         %%zmm29, %%zmm29, %%zmm29                \n"
		"   vbroadcastss    4(%%rax), %%zmm1                         \n"
		"    vpxorq         %%zmm30, %%zmm30, %%zmm30                \n"
		"    vpxorq         %%zmm31, %%zmm31, %%zmm31                \n"

		"    subq     $8, %%rdx                                      \n"

		"MAIN_K_M4_N16:                                              \n"

		"    KERNEL4x16_K1                                           \n"
		"    KERNEL4x16_K2                                           \n"
		"    KERNEL4x16_K1                                           \n"
		"    KERNEL4x16_K2                                           \n"
		"    KERNEL4x16_K1                                           \n"
		"    KERNEL4x16_K2                                           \n"
		"    KERNEL4x16_K1                                           \n"
		"   cmp     $0, %%rdx                                        \n"
		"    je         EDGE_K_M4_N16                                \n"
		"    KERNEL4x16_K2                                           \n"

		"    subq     $8, %%rdx                                      \n"
		"   jmp     MAIN_K_M4_N16                                    \n"

		"EDGE_K_M4_N16:                                              \n"

		"    KERNEL4x16_END_K                                        \n"

		"    cmp     $0, %%r15                                       \n"
		"    je      SAVE_C_4x16                                     \n"
		"    ADD_C_4x16                                              \n"

		"SAVE_C_4x16:                                                \n"
		"    SAVE_4x16                                               \n"

		//----------------------------------------------------------------

		"BEGIN_M1_N16:                                               \n"
		"    cmpq      $1, %%rdi                                     \n" // M % 1
		"    jb       END_M_N16                                      \n"

		"    mov     %%r14, %%rbx                                    \n" // Bc
		"   prefetcht0         (%%rbx)                               \n"

		"    mov      %%rcx, %%r10                                   \n" // C0
		"   prefetcht1         (%%r10)                               \n"

		"    mov     %%rsi, %%rdx                                    \n" // K

		"   vmovups        (%%rbx), %%zmm4                           \n"
		"   vbroadcastss    (%%rax), %%zmm0                          \n" // A0
		"    vpxorq         %%zmm30, %%zmm30, %%zmm30                \n"

		"    subq     $8, %%rdx                                      \n"

		"MAIN_K_M1_N16:                                              \n"
		"    KERNEL1x16_K1                                           \n"
		"    KERNEL1x16_K2                                           \n"
		"    KERNEL1x16_K1                                           \n"
		"    KERNEL1x16_K2                                           \n"
		"    KERNEL1x16_K1                                           \n"
		"    KERNEL1x16_K2                                           \n"
		"    KERNEL1x16_K1                                           \n"
		"   cmp     $0, %%rdx                                        \n"
		"    je         EDGE_K_M1_N16                                \n"
		"    KERNEL1x16_K2                                           \n"

		"    subq     $8, %%rdx                                      \n"
		"   jmp     MAIN_K_M1_N16                                    \n"

		"EDGE_K_M1_N16:                                              \n"
		"    KERNEL1x16_END_K                                        \n"

		"    cmp     $0, %%r15                                       \n"
		"    je      SAVE_C_1x16                                     \n"
		"    ADD_C_1x16                                              \n"

		"SAVE_C_1x16:                                                \n"
		"    SAVE_1x16                                               \n"
		"   cmpq      $1, %%rdi                                      \n"
		"    jnb     BEGIN_M1_N16                                    \n" // 不小于（或等于）则跳转

		//-----------------------------------------------------------------

		"END_M_N16:                                                  \n"

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
		[Bc] "m"(Bc),
		[k_tag] "m"(k_tag)
		: "rax", "rbx", "rcx", "rdx", "rdi", "rsi", "rbp", "r8", "r9", "r10", "r11", "r12",
		  "r13", "r14", "r15", "zmm0", "zmm1", "zmm2", "zmm3", "zmm4", "zmm5",
		  "zmm6", "zmm7", "zmm8", "zmm9", "zmm10", "zmm11", "zmm12", "zmm13",
		  "zmm14", "zmm15", "zmm16", "zmm17", "zmm18", "zmm19", "zmm20", "zmm21",
		  "zmm22", "zmm23", "zmm24", "zmm25", "zmm26", "zmm27", "zmm28", "zmm29",
		  "zmm30", "zmm31", "memory");
}

void SMM_NN_KERNELm12xn8(float *C, float *A, float *B, long M, long N, long K, long LN, long LK, float *Bc, long k_tag)
{
	asm volatile(
		".macro    KERNEL12x8_PACK_K1                                \n"

		"   vbroadcastss    8(%%rax), %%ymm2                         \n"
		"   vfmadd231ps        %%ymm0, %%ymm4, %%ymm20               \n"

		"   vbroadcastss    12(%%rax), %%ymm3                        \n"
		"   vfmadd231ps        %%ymm1, %%ymm4, %%ymm21               \n"

		"   prefetcht0         256(%%rax)                            \n"

		"   vbroadcastss    16(%%rax), %%ymm0                        \n"
		"   vfmadd231ps        %%ymm2, %%ymm4, %%ymm22               \n"

		"   vbroadcastss    20(%%rax), %%ymm1                        \n"
		"   vfmadd231ps        %%ymm3, %%ymm4, %%ymm23               \n"

		"   vbroadcastss    24(%%rax), %%ymm2                        \n"
		"   vfmadd231ps        %%ymm0, %%ymm4, %%ymm24               \n"

		"   vbroadcastss    28(%%rax), %%ymm3                        \n"
		"   vfmadd231ps        %%ymm1, %%ymm4, %%ymm25               \n"

		"   vbroadcastss    32(%%rax), %%ymm0                        \n"
		"   vfmadd231ps        %%ymm2, %%ymm4, %%ymm26               \n"

		"    leaq      (%%rbx, %%r8, 4), %%rbx                       \n" // B

		"   vbroadcastss    36(%%rax), %%ymm1                        \n"
		"   vfmadd231ps        %%ymm3, %%ymm4, %%ymm27               \n"

		"   vbroadcastss    40(%%rax), %%ymm2                        \n"
		"   vfmadd231ps        %%ymm0, %%ymm4, %%ymm28               \n"

		"   vbroadcastss    44(%%rax), %%ymm3                        \n"
		"   vfmadd231ps        %%ymm1, %%ymm4, %%ymm29               \n"
		"   vmovups         (%%rbx), %%ymm6                          \n"
		"    addq              $48, %%rax                            \n"

		"   vbroadcastss    (%%rax), %%ymm0                          \n"
		"   vfmadd231ps        %%ymm2, %%ymm4, %%ymm30               \n"

		"   vbroadcastss    4(%%rax), %%ymm1                         \n"
		"   vfmadd231ps        %%ymm3, %%ymm4, %%ymm31               \n"
		"   vmovups         %%ymm4, (%%rbp)                          \n"
		"    addq              $32, %%rbp                            \n" // 64->32

		".endm                                                       \n"

		".macro    KERNEL12x8_PACK_K2                                \n"

		"   vbroadcastss    8(%%rax), %%ymm2                         \n"
		"   vfmadd231ps        %%ymm0, %%ymm6, %%ymm20               \n"

		"   vbroadcastss    12(%%rax), %%ymm3                        \n"
		"   vfmadd231ps        %%ymm1, %%ymm6, %%ymm21               \n"

		"   vbroadcastss    16(%%rax), %%ymm0                        \n"
		"   vfmadd231ps        %%ymm2, %%ymm6, %%ymm22               \n"

		"   prefetcht0         256(%%rax)                            \n"

		"   vbroadcastss    20(%%rax), %%ymm1                        \n"
		"   vfmadd231ps        %%ymm3, %%ymm6, %%ymm23               \n"

		"   vbroadcastss    24(%%rax), %%ymm2                        \n"
		"   vfmadd231ps        %%ymm0, %%ymm6, %%ymm24               \n"

		"   vbroadcastss    28(%%rax), %%ymm3                        \n"
		"   vfmadd231ps        %%ymm1, %%ymm6, %%ymm25               \n"

		"   vbroadcastss    32(%%rax), %%ymm0                        \n"
		"   vfmadd231ps        %%ymm2, %%ymm6, %%ymm26               \n"

		"    leaq      (%%rbx, %%r8, 4), %%rbx                       \n" // B

		"   vbroadcastss    36(%%rax), %%ymm1                        \n"
		"   vfmadd231ps        %%ymm3, %%ymm6, %%ymm27               \n"

		"   prefetcht0         32(%%rbx)                             \n" //

		"   vbroadcastss    40(%%rax), %%ymm2                        \n"
		"   vfmadd231ps        %%ymm0, %%ymm6, %%ymm28               \n"

		"   vbroadcastss    44(%%rax), %%ymm3                        \n"
		"   vfmadd231ps        %%ymm1, %%ymm6, %%ymm29               \n"
		"   vmovups         (%%rbx), %%ymm4                          \n"
		"    addq              $48, %%rax                            \n"

		"   vbroadcastss    (%%rax), %%ymm0                          \n"
		"   vfmadd231ps        %%ymm2, %%ymm6, %%ymm30               \n"

		"   vbroadcastss    4(%%rax), %%ymm1                         \n"
		"   vfmadd231ps        %%ymm3, %%ymm6, %%ymm31               \n"
		"   vmovups         %%ymm6, (%%rbp)                          \n"
		"    addq              $32, %%rbp                            \n" // 64->32

		".endm                                                       \n"

		".macro    KERNEL12x8_PACK_END_K                             \n"

		"   vbroadcastss    8(%%rax), %%ymm2                         \n"
		"   vfmadd231ps        %%ymm0, %%ymm6, %%ymm20               \n"

		"   vbroadcastss    12(%%rax), %%ymm3                        \n"
		"   vfmadd231ps        %%ymm1, %%ymm6, %%ymm21               \n"

		"   vbroadcastss    16(%%rax), %%ymm0                        \n"
		"   vfmadd231ps        %%ymm2, %%ymm6, %%ymm22               \n"

		"   prefetcht0         256(%%rax)                            \n"

		"   vbroadcastss    20(%%rax), %%ymm1                        \n"
		"   vfmadd231ps        %%ymm3, %%ymm6, %%ymm23               \n"

		"   vbroadcastss    24(%%rax), %%ymm2                        \n"
		"   vfmadd231ps        %%ymm0, %%ymm6, %%ymm24               \n"

		"   vbroadcastss    28(%%rax), %%ymm3                        \n"
		"   vfmadd231ps        %%ymm1, %%ymm6, %%ymm25               \n"

		"   vbroadcastss    32(%%rax), %%ymm0                        \n"
		"   vfmadd231ps        %%ymm2, %%ymm6, %%ymm26               \n"

		"   vbroadcastss    36(%%rax), %%ymm1                        \n"
		"   vfmadd231ps        %%ymm3, %%ymm6, %%ymm27               \n"

		"   vbroadcastss    40(%%rax), %%ymm2                        \n"
		"   vfmadd231ps        %%ymm0, %%ymm6, %%ymm28               \n"

		"   vbroadcastss    44(%%rax), %%ymm3                        \n"
		"   vfmadd231ps        %%ymm1, %%ymm6, %%ymm29               \n"
		"    addq              $48, %%rax                            \n"

		"   vfmadd231ps        %%ymm2, %%ymm6, %%ymm30               \n"
		"   vmovups         %%ymm6, (%%rbp)                          \n"
		"   vfmadd231ps        %%ymm3, %%ymm6, %%ymm31               \n"

		".endm                                                       \n"

		//-----------------------------------------------------------------------

		".macro    KERNEL12x8_K1                                     \n"

		"   vbroadcastss    8(%%rax), %%ymm2                         \n"
		"   vfmadd231ps        %%ymm0, %%ymm4, %%ymm20               \n"

		"   vbroadcastss    12(%%rax), %%ymm3                        \n"
		"   vfmadd231ps        %%ymm1, %%ymm4, %%ymm21               \n"

		"   prefetcht0         256(%%rax)                            \n"

		"   vbroadcastss    16(%%rax), %%ymm0                        \n"
		"   vfmadd231ps        %%ymm2, %%ymm4, %%ymm22               \n"

		"   vbroadcastss    20(%%rax), %%ymm1                        \n"
		"   vfmadd231ps        %%ymm3, %%ymm4, %%ymm23               \n"

		"   vbroadcastss    24(%%rax), %%ymm2                        \n"
		"   vfmadd231ps        %%ymm0, %%ymm4, %%ymm24               \n"

		"   vbroadcastss    28(%%rax), %%ymm3                        \n"
		"   vfmadd231ps        %%ymm1, %%ymm4, %%ymm25               \n"

		"   vbroadcastss    32(%%rax), %%ymm0                        \n"
		"   vfmadd231ps        %%ymm2, %%ymm4, %%ymm26               \n"

		"    addq              $32, %%rbx                            \n" // B

		"   vbroadcastss    36(%%rax), %%ymm1                        \n"
		"   vfmadd231ps        %%ymm3, %%ymm4, %%ymm27               \n"

		"   vbroadcastss    40(%%rax), %%ymm2                        \n"
		"   vfmadd231ps        %%ymm0, %%ymm4, %%ymm28               \n"

		"   vbroadcastss    44(%%rax), %%ymm3                        \n"
		"   vfmadd231ps        %%ymm1, %%ymm4, %%ymm29               \n"
		"   vmovups         (%%rbx), %%ymm6                          \n"
		"    addq              $48, %%rax                            \n"

		"   vbroadcastss    (%%rax), %%ymm0                          \n"
		"   vfmadd231ps        %%ymm2, %%ymm4, %%ymm30               \n"

		"   vbroadcastss    4(%%rax), %%ymm1                         \n"
		"   vfmadd231ps        %%ymm3, %%ymm4, %%ymm31               \n"

		".endm                                                       \n"

		".macro    KERNEL12x8_K2                                     \n"

		"   vbroadcastss    8(%%rax), %%ymm2                         \n"
		"   vfmadd231ps        %%ymm0, %%ymm6, %%ymm20               \n"

		"   vbroadcastss    12(%%rax), %%ymm3                        \n"
		"   vfmadd231ps        %%ymm1, %%ymm6, %%ymm21               \n"

		"   vbroadcastss    16(%%rax), %%ymm0                        \n"
		"   vfmadd231ps        %%ymm2, %%ymm6, %%ymm22               \n"

		"   prefetcht0         256(%%rax)                            \n"

		"   vbroadcastss    20(%%rax), %%ymm1                        \n"
		"   vfmadd231ps        %%ymm3, %%ymm6, %%ymm23               \n"

		"   vbroadcastss    24(%%rax), %%ymm2                        \n"
		"   vfmadd231ps        %%ymm0, %%ymm6, %%ymm24               \n"

		"   vbroadcastss    28(%%rax), %%ymm3                        \n"
		"   vfmadd231ps        %%ymm1, %%ymm6, %%ymm25               \n"

		"   vbroadcastss    32(%%rax), %%ymm0                        \n"
		"   vfmadd231ps        %%ymm2, %%ymm6, %%ymm26               \n"

		"    addq             $32, %%rbx                             \n" //

		"   vbroadcastss    36(%%rax), %%ymm1                        \n"
		"   vfmadd231ps        %%ymm3, %%ymm6, %%ymm27               \n"

		"   prefetcht0         32(%%rbx)                             \n" //

		"   vbroadcastss    40(%%rax), %%ymm2                        \n"
		"   vfmadd231ps        %%ymm0, %%ymm6, %%ymm28               \n"

		"   vbroadcastss    44(%%rax), %%ymm3                        \n"
		"   vfmadd231ps        %%ymm1, %%ymm6, %%ymm29               \n"
		"   vmovups         (%%rbx), %%ymm4                          \n"
		"    addq              $48, %%rax                            \n"

		"   vbroadcastss    (%%rax), %%ymm0                          \n"
		"   vfmadd231ps        %%ymm2, %%ymm6, %%ymm30               \n"

		"   vbroadcastss    4(%%rax), %%ymm1                         \n"
		"   vfmadd231ps        %%ymm3, %%ymm6, %%ymm31               \n"

		".endm                                                       \n"

		".macro    KERNEL12x8_END_K                                  \n"

		"   vbroadcastss    8(%%rax), %%ymm2                         \n"
		"   vfmadd231ps        %%ymm0, %%ymm6, %%ymm20               \n"

		"   vbroadcastss    12(%%rax), %%ymm3                        \n"
		"   vfmadd231ps        %%ymm1, %%ymm6, %%ymm21               \n"

		"   vbroadcastss    16(%%rax), %%ymm0                        \n"
		"   vfmadd231ps        %%ymm2, %%ymm6, %%ymm22               \n"

		"   prefetcht0         256(%%rax)                            \n"

		"   vbroadcastss    20(%%rax), %%ymm1                        \n"
		"   vfmadd231ps        %%ymm3, %%ymm6, %%ymm23               \n"

		"   vbroadcastss    24(%%rax), %%ymm2                        \n"
		"   vfmadd231ps        %%ymm0, %%ymm6, %%ymm24               \n"

		"   vbroadcastss    28(%%rax), %%ymm3                        \n"
		"   vfmadd231ps        %%ymm1, %%ymm6, %%ymm25               \n"

		"   vbroadcastss    32(%%rax), %%ymm0                        \n"
		"   vfmadd231ps        %%ymm2, %%ymm6, %%ymm26               \n"

		"   vbroadcastss    36(%%rax), %%ymm1                        \n"
		"   vfmadd231ps        %%ymm3, %%ymm6, %%ymm27               \n"

		"   vbroadcastss    40(%%rax), %%ymm2                        \n"
		"   vfmadd231ps        %%ymm0, %%ymm6, %%ymm28               \n"

		"   vbroadcastss    44(%%rax), %%ymm3                        \n"
		"   vfmadd231ps        %%ymm1, %%ymm6, %%ymm29               \n"
		"   vmovups         (%%rbx), %%ymm4                          \n"
		"    addq              $48, %%rax                            \n"

		"   vbroadcastss    (%%rax), %%ymm0                          \n"
		"   vfmadd231ps        %%ymm2, %%ymm6, %%ymm30               \n"

		"   vbroadcastss    4(%%rax), %%ymm1                         \n"
		"   vfmadd231ps        %%ymm3, %%ymm6, %%ymm31               \n"

		".endm                                                       \n"

		".macro    ADD_C_12x8                                        \n"

		"   vmovups         (%%r10), %%ymm0                          \n"
		"    vaddps             %%ymm0, %%ymm20, %%ymm20             \n"
		"   vmovups         (%%r11), %%ymm1                          \n"
		"    vaddps             %%ymm1, %%ymm21, %%ymm21             \n"
		"   vmovups         (%%r12), %%ymm2                          \n"
		"    vaddps             %%ymm2, %%ymm22, %%ymm22             \n"
		"   vmovups         (%%r13), %%ymm3                          \n"
		"    vaddps             %%ymm3, %%ymm23, %%ymm23             \n"

		"    leaq              (%%r13, %%r8, 4), %%r10               \n" // C0
		"    leaq             (%%r10, %%r8, 4), %%r11                \n" // C1
		"    leaq             (%%r11, %%r8, 4), %%r12                \n" // C2
		"    leaq             (%%r12, %%r8, 4), %%r13                \n" // C3

		"   vmovups         (%%r10), %%ymm4                          \n"
		"    vaddps             %%ymm4, %%ymm24, %%ymm24             \n"
		"   vmovups         (%%r11), %%ymm5                          \n"
		"    vaddps             %%ymm5, %%ymm25, %%ymm25             \n"
		"   vmovups         (%%r12), %%ymm6                          \n"
		"    vaddps             %%ymm6, %%ymm26, %%ymm26             \n"
		"   vmovups         (%%r13), %%ymm7                          \n"
		"    vaddps             %%ymm7, %%ymm27, %%ymm27             \n"

		"    leaq              (%%r13, %%r8, 4), %%r10               \n" // C0
		"    leaq             (%%r10, %%r8, 4), %%r11                \n" // C1
		"    leaq             (%%r11, %%r8, 4), %%r12                \n" // C2
		"    leaq             (%%r12, %%r8, 4), %%r13                \n" // C3

		"   vmovups         (%%r10), %%ymm0                          \n"
		"    vaddps             %%ymm0, %%ymm28, %%ymm28             \n"
		"   vmovups         (%%r11), %%ymm1                          \n"
		"    vaddps             %%ymm1, %%ymm29, %%ymm29             \n"
		"   vmovups         (%%r12), %%ymm2                          \n"
		"    vaddps             %%ymm2, %%ymm30, %%ymm30             \n"
		"   vmovups         (%%r13), %%ymm3                          \n"
		"    vaddps             %%ymm3, %%ymm31, %%ymm31             \n"

		"    mov      %%rcx, %%r10                                   \n" // C0
		"    leaq     (%%r10, %%r8, 4), %%r11                        \n" // C1
		"    leaq     (%%r11, %%r8, 4), %%r12                        \n" // C2
		"    leaq     (%%r12, %%r8, 4), %%r13                        \n" // C3

		".endm                                                       \n"

		".macro    SAVE_12x8                                         \n"

		"   vmovups         %%ymm20, (%%r10)                         \n"
		"   vmovups         %%ymm21, (%%r11)                         \n"
		"    subq             $12, %%rdi                             \n"
		"   vmovups         %%ymm22, (%%r12)                         \n"
		"   vmovups         %%ymm23, (%%r13)                         \n"

		"    leaq      (%%r13, %%r8, 4), %%r10                       \n" // C0
		"    leaq     (%%r10, %%r8, 4), %%r11                        \n" // C1
		"    leaq     (%%r11, %%r8, 4), %%r12                        \n" // C2
		"    leaq     (%%r12, %%r8, 4), %%r13                        \n" // C3

		"   vmovups         %%ymm24, (%%r10)                         \n"
		"   vmovups         %%ymm25, (%%r11)                         \n"
		"   vmovups         %%ymm26, (%%r12)                         \n"
		"   vmovups         %%ymm27, (%%r13)                         \n"

		"    leaq      (%%r13, %%r8, 4), %%r10                       \n" // C0
		"    leaq     (%%r10, %%r8, 4), %%r11                        \n" // C1
		"    leaq     (%%r11, %%r8, 4), %%r12                        \n" // C2
		"    leaq     (%%r12, %%r8, 4), %%r13                        \n" // C3

		"   vmovups         %%ymm28, (%%r10)                         \n"
		"   vmovups         %%ymm29, (%%r11)                         \n"
		"   vmovups         %%ymm30, (%%r12)                         \n"
		"   vmovups         %%ymm31, (%%r13)                         \n"

		"    leaq      (%%r13, %%r8, 4), %%rcx                       \n" // C0

		".endm                                                       \n"

		//-----------------------------------------------------------------------

		".macro    KERNEL8x8_K1                                      \n"

		"   vbroadcastss    8(%%rax), %%ymm2                         \n"
		"   vfmadd231ps        %%ymm0, %%ymm4, %%ymm24               \n"

		"   vbroadcastss    12(%%rax), %%ymm3                        \n"
		"   vfmadd231ps        %%ymm1, %%ymm4, %%ymm25               \n"

		"   prefetcht0         256(%%rax)                            \n"

		"   vbroadcastss    16(%%rax), %%ymm8                        \n"
		"   vfmadd231ps        %%ymm2, %%ymm4, %%ymm26               \n"

		"   vbroadcastss    20(%%rax), %%ymm9                        \n"
		"   vfmadd231ps        %%ymm3, %%ymm4, %%ymm27               \n"

		"   vbroadcastss    24(%%rax), %%ymm10                       \n"
		"   vfmadd231ps        %%ymm8, %%ymm4, %%ymm28               \n"

		"   prefetcht0         32(%%rbx)                             \n" //

		"   vbroadcastss    28(%%rax), %%ymm11                       \n"
		"   vfmadd231ps        %%ymm9, %%ymm4, %%ymm29               \n"
		"    addq              $32, %%rbx                            \n" //
		"   vfmadd231ps        %%ymm10, %%ymm4, %%ymm30              \n"

		"    addq              $32, %%rax                            \n"
		"   vmovups         (%%rbx), %%ymm6                          \n"
		"   vfmadd231ps        %%ymm11, %%ymm4, %%ymm31              \n"

		"   vbroadcastss    (%%rax), %%ymm0                          \n"
		"   vbroadcastss    4(%%rax), %%ymm1                         \n"

		".endm                                                       \n"

		".macro    KERNEL8x8_K2                                      \n"

		"   vbroadcastss    8(%%rax), %%ymm2                         \n"
		"   vfmadd231ps        %%ymm0, %%ymm6, %%ymm24               \n"

		"   vbroadcastss    12(%%rax), %%ymm3                        \n"
		"   vfmadd231ps        %%ymm1, %%ymm6, %%ymm25               \n"

		"   prefetcht0         256(%%rax)                            \n"

		"   vbroadcastss    16(%%rax), %%ymm8                        \n"
		"   vfmadd231ps        %%ymm2, %%ymm6, %%ymm26               \n"

		"   vbroadcastss    20(%%rax), %%ymm9                        \n"
		"   vfmadd231ps        %%ymm3, %%ymm6, %%ymm27               \n"

		"   vbroadcastss    24(%%rax), %%ymm10                       \n"
		"   vfmadd231ps        %%ymm8, %%ymm6, %%ymm28               \n"

		"   prefetcht0         32(%%rbx)                             \n" //

		"   vbroadcastss    28(%%rax), %%ymm11                       \n"
		"   vfmadd231ps        %%ymm9, %%ymm6, %%ymm29               \n"
		"    addq              $32, %%rbx                            \n" // B
		"   vfmadd231ps        %%ymm10, %%ymm6, %%ymm30              \n"

		"    addq              $32, %%rax                            \n"
		"   vmovups         (%%rbx), %%ymm4                          \n"
		"   vfmadd231ps        %%ymm11, %%ymm6, %%ymm31              \n"

		"   vbroadcastss    (%%rax), %%ymm0                          \n"
		"   vbroadcastss    4(%%rax), %%ymm1                         \n"

		".endm                                                       \n"

		".macro    KERNEL8x8_END_K                                   \n"

		"   vbroadcastss    8(%%rax), %%ymm2                         \n"
		"   vfmadd231ps        %%ymm0, %%ymm6, %%ymm24               \n"

		"   vbroadcastss    12(%%rax), %%ymm3                        \n"
		"   vfmadd231ps        %%ymm1, %%ymm6, %%ymm25               \n"

		"   prefetcht0         256(%%rax)                            \n"

		"   vbroadcastss    16(%%rax), %%ymm8                        \n"
		"   vfmadd231ps        %%ymm2, %%ymm6, %%ymm26               \n"

		"   vbroadcastss    20(%%rax), %%ymm9                        \n"
		"   vfmadd231ps        %%ymm3, %%ymm6, %%ymm27               \n"

		"   vbroadcastss    24(%%rax), %%ymm10                       \n"
		"   vfmadd231ps        %%ymm8, %%ymm6, %%ymm28               \n"

		"   vbroadcastss    28(%%rax), %%ymm11                       \n"
		"   vfmadd231ps        %%ymm9, %%ymm6, %%ymm29               \n"
		"   vfmadd231ps        %%ymm10, %%ymm6, %%ymm30              \n"

		"    addq              $32, %%rax                            \n"
		"   vfmadd231ps        %%ymm11, %%ymm6, %%ymm31              \n"

		".endm                                                       \n"

		".macro    ADD_C_8x8                                         \n"

		"   vmovups         (%%r10), %%ymm4                          \n"
		"    vaddps             %%ymm4, %%ymm24, %%ymm24             \n"
		"   vmovups         (%%r11), %%ymm5                          \n"
		"    vaddps             %%ymm5, %%ymm25, %%ymm25             \n"
		"   vmovups         (%%r12), %%ymm6                          \n"
		"    vaddps             %%ymm6, %%ymm26, %%ymm26             \n"
		"   vmovups         (%%r13), %%ymm7                          \n"
		"    vaddps             %%ymm7, %%ymm27, %%ymm27             \n"

		"    leaq              (%%r13, %%r8, 4), %%r10               \n" // C0
		"    leaq             (%%r10, %%r8, 4), %%r11                \n" // C1
		"    leaq             (%%r11, %%r8, 4), %%r12                \n" // C2
		"    leaq             (%%r12, %%r8, 4), %%r13                \n" // C3

		"   vmovups         (%%r10), %%ymm0                          \n"
		"    vaddps             %%ymm0, %%ymm28, %%ymm28             \n"
		"   vmovups         (%%r11), %%ymm1                          \n"
		"    vaddps             %%ymm1, %%ymm29, %%ymm29             \n"
		"   vmovups         (%%r12), %%ymm2                          \n"
		"    vaddps             %%ymm2, %%ymm30, %%ymm30             \n"
		"   vmovups         (%%r13), %%ymm3                          \n"
		"    vaddps             %%ymm3, %%ymm31, %%ymm31             \n"

		"    mov      %%rcx, %%r10                                   \n" // C0
		"    leaq     (%%r10, %%r8, 4), %%r11                        \n" // C1
		"    leaq     (%%r11, %%r8, 4), %%r12                        \n" // C2
		"    leaq     (%%r12, %%r8, 4), %%r13                        \n" // C3

		".endm                                                       \n"

		".macro    SAVE_8x8                                          \n"

		"   vmovups         %%ymm24, (%%r10)                         \n"
		"   vmovups         %%ymm25, (%%r11)                         \n"
		"    subq             $8, %%rdi                              \n"
		"   vmovups         %%ymm26, (%%r12)                         \n"
		"   vmovups         %%ymm27, (%%r13)                         \n"

		"    leaq      (%%r13, %%r8, 4), %%r10                       \n" // C0
		"    leaq     (%%r10, %%r8, 4), %%r11                        \n" // C1
		"    leaq     (%%r11, %%r8, 4), %%r12                        \n" // C2
		"    leaq     (%%r12, %%r8, 4), %%r13                        \n" // C3

		"   vmovups         %%ymm28, (%%r10)                         \n"
		"   vmovups         %%ymm29, (%%r11)                         \n"
		"   vmovups         %%ymm30, (%%r12)                         \n"
		"   vmovups         %%ymm31, (%%r13)                         \n"

		"    leaq      (%%r13, %%r8, 4), %%rcx                       \n" // C0

		".endm                                                       \n"

		//-----------------------------------------------------------------------

		".macro    KERNEL4x8_K1                                      \n"

		"   vbroadcastss    8(%%rax), %%ymm2                         \n"
		"   vfmadd231ps        %%ymm0, %%ymm4, %%ymm28               \n"

		"   vbroadcastss    12(%%rax), %%ymm3                        \n"
		"   vfmadd231ps        %%ymm1, %%ymm4, %%ymm29               \n"
		"    addq              $16, %%rax                            \n"

		"   prefetcht0         256(%%rax)                            \n"
		"    addq              $32, %%rbx                            \n" // B
		"   vbroadcastss    (%%rax), %%ymm0                          \n"
		"   vfmadd231ps        %%ymm2, %%ymm4, %%ymm30               \n"

		"   vbroadcastss    4(%%rax), %%ymm1                         \n"
		"   vfmadd231ps        %%ymm3, %%ymm4, %%ymm31               \n"

		"   prefetcht0         32(%%rbx)                             \n"
		"   vmovups         (%%rbx), %%ymm6                          \n"

		".endm                                                       \n"

		".macro    KERNEL4x8_K2                                      \n"

		"   vbroadcastss    8(%%rax), %%ymm2                         \n"
		"   vfmadd231ps        %%ymm0, %%ymm6, %%ymm28               \n"

		"   vbroadcastss    12(%%rax), %%ymm3                        \n"
		"   vfmadd231ps        %%ymm1, %%ymm6, %%ymm29               \n"
		"    addq              $16, %%rax                            \n"

		"   prefetcht0         256(%%rax)                            \n"
		"    addq              $32, %%rbx                            \n" // B
		"   vbroadcastss    (%%rax), %%ymm0                          \n"
		"   vfmadd231ps        %%ymm2, %%ymm6, %%ymm30               \n"

		"   vbroadcastss    4(%%rax), %%ymm1                         \n"
		"   vfmadd231ps        %%ymm3, %%ymm6, %%ymm31               \n"

		"   vmovups         (%%rbx), %%ymm4                          \n"
		"   prefetcht0         32(%%rbx)                             \n"

		".endm                                                       \n"

		".macro    KERNEL4x8_END_K                                   \n"

		"   vbroadcastss    8(%%rax), %%ymm2                         \n"
		"   vfmadd231ps        %%ymm0, %%ymm6, %%ymm28               \n"

		"   vbroadcastss    12(%%rax), %%ymm3                        \n"
		"   vfmadd231ps        %%ymm1, %%ymm6, %%ymm29               \n"
		"    addq              $16, %%rax                            \n"

		"   vfmadd231ps        %%ymm2, %%ymm6, %%ymm30               \n"
		"   vfmadd231ps        %%ymm3, %%ymm6, %%ymm31               \n"

		".endm                                                       \n"

		".macro    ADD_C_4x8                                         \n"

		"   vmovups         (%%r10), %%ymm0                          \n"
		"    vaddps             %%ymm0, %%ymm28, %%ymm28             \n"
		"   vmovups         (%%r11), %%ymm1                          \n"
		"    vaddps             %%ymm1, %%ymm29, %%ymm29             \n"
		"   vmovups         (%%r12), %%ymm2                          \n"
		"    vaddps             %%ymm2, %%ymm30, %%ymm30             \n"
		"   vmovups         (%%r13), %%ymm3                          \n"
		"    vaddps             %%ymm3, %%ymm31, %%ymm31             \n"

		"    mov      %%rcx, %%r10                                   \n" // C0
		"    leaq     (%%r10, %%r8, 4), %%r11                        \n" // C1
		"    leaq     (%%r11, %%r8, 4), %%r12                        \n" // C2
		"    leaq     (%%r12, %%r8, 4), %%r13                        \n" // C3

		".endm                                                       \n"

		".macro    SAVE_4x8                                          \n"
		"   subq             $4, %%rdi                               \n"

		"   vmovups         %%ymm28, (%%r10)                         \n"
		"   vmovups         %%ymm29, (%%r11)                         \n"
		"   vmovups         %%ymm30, (%%r12)                         \n"
		"   vmovups         %%ymm31, (%%r13)                         \n"

		"    leaq      (%%r13, %%r8, 4), %%rcx                       \n" // C0

		".endm                                                       \n"

		//-----------------------------------------------------------------

		".macro    KERNEL1x8_K1                                      \n"
		"   prefetcht0         256(%%rax)                            \n"
		"   addq              $4, %%rax                              \n"

		"   vbroadcastss    (%%rax), %%ymm2                          \n"
		"   vfmadd231ps        %%ymm0, %%ymm4, %%ymm30               \n"

		"    addq              $32, %%rbx                            \n" // B
		"   vmovups            (%%rbx), %%ymm6                       \n"
		"   prefetcht0         32(%%rbx)                             \n"

		".endm                                                       \n"

		".macro    KERNEL1x8_K2                                      \n"
		"   prefetcht0         256(%%rax)                            \n"
		"    addq              $4, %%rax                             \n"

		"   vbroadcastss    (%%rax), %%ymm0                          \n"
		"   vfmadd231ps        %%ymm2, %%ymm6, %%ymm30               \n"

		"    addq              $32, %%rbx                            \n" // B
		"   vmovups            (%%rbx), %%ymm4                       \n"
		"   prefetcht0         32(%%rbx)                             \n"

		".endm                                                       \n"

		".macro    KERNEL1x8_END_K                                   \n"
		"   vfmadd231ps        %%ymm2, %%ymm6, %%ymm30               \n"
		"    addq              $4, %%rax                             \n"
		".endm                                                       \n"

		".macro    ADD_C_1x8                                         \n"
		"   vmovups         (%%r10), %%ymm2                          \n"
		"    vaddps             %%ymm2, %%ymm30, %%ymm30             \n"
		"    mov      %%rcx, %%r10                                   \n" // C0
		".endm                                                       \n"

		".macro    SAVE_1x8                                          \n"
		"   vmovups         %%ymm30, (%%r10)                         \n"
		"   subq             $1, %%rdi                               \n"
		"    leaq      (%%r10, %%r8, 4), %%rcx                       \n" // C0

		".endm                                                       \n"

		//-----------------------------------------------------------------

		"SMM_NN_KERNEL12x8:                                          \n"

		"   mov     %[C], %%rcx                                      \n"
		"   mov     %[A], %%rax                                      \n"
		"   mov     %[B], %%rbx                                      \n"

		"   prefetcht0         (%%rax)                               \n"

		"    mov     %[K], %%rdx                                     \n" // K
		"    mov      %[LN], %%r8                                    \n"
		"    mov      %[Bc], %%r14                                   \n"
		"    mov      %[M], %%rdi                                    \n"
		"    mov     %[k_tag], %%r15                                 \n"

		"   prefetcht0         (%%rbx)                               \n"
		"    mov     %%rbx, %%r9                                     \n" // B
		"    mov     %%rdx, %%rsi                                    \n" // K

		"BEGIN_PACK12x8:                                             \n"

		"    mov     %%r9, %%rbx                                     \n" // B
		"   prefetcht0         (%%rbx)                               \n"

		"    mov      %%rcx, %%r10                                   \n" // C0
		"   prefetcht1         (%%r10)                               \n"
		"    leaq     (%%r10, %%r8, 4), %%r11                        \n" // C1
		"   prefetcht1         (%%r11)                               \n"
		"    leaq     (%%r11, %%r8, 4), %%r12                        \n" // C2
		"   prefetcht1         (%%r12)                               \n"
		"    leaq     (%%r12, %%r8, 4), %%r13                        \n" // C3
		"   prefetcht1         (%%r13)                               \n"

		"    mov     %%rsi, %%rdx                                    \n" // K
		"    mov     %%r14, %%rbp                                    \n" // Bc

		"   vmovups        (%%rbx), %%ymm4                           \n"
		"    vpxorq         %%ymm20, %%ymm20, %%ymm20                \n"
		"    vpxorq         %%ymm21, %%ymm21, %%ymm21                \n"
		"   vbroadcastss    (%%rax), %%ymm0                          \n"
		"    vpxorq         %%ymm22, %%ymm22, %%ymm22                \n"
		"    vpxorq         %%ymm23, %%ymm23, %%ymm23                \n"
		"   vbroadcastss    4(%%rax), %%ymm1                         \n"
		"    vpxorq         %%ymm24, %%ymm24, %%ymm24                \n"
		"    vpxorq         %%ymm25, %%ymm25, %%ymm25                \n"
		"    vpxorq         %%ymm26, %%ymm26, %%ymm26                \n"
		"    vpxorq         %%ymm27, %%ymm27, %%ymm27                \n"
		"    vpxorq         %%ymm28, %%ymm28, %%ymm28                \n"
		"    vpxorq         %%ymm29, %%ymm29, %%ymm29                \n"
		"    vpxorq         %%ymm30, %%ymm30, %%ymm30                \n"
		"    vpxorq         %%ymm31, %%ymm31, %%ymm31                \n"

		"    subq     $8, %%rdx                                      \n"

		"MAIN_PACK_K12x8:                                            \n"

		"    KERNEL12x8_PACK_K1                                      \n"
		"    KERNEL12x8_PACK_K2                                      \n"
		"    KERNEL12x8_PACK_K1                                      \n"
		"    KERNEL12x8_PACK_K2                                      \n"
		"    KERNEL12x8_PACK_K1                                      \n"
		"    KERNEL12x8_PACK_K2                                      \n"
		"    KERNEL12x8_PACK_K1                                      \n"
		"   cmp     $0, %%rdx                                        \n"
		"    je         EDGE_PACK_K12x8                              \n"
		"    KERNEL12x8_PACK_K2                                      \n"

		"    subq     $8, %%rdx                                      \n"
		"   jmp     MAIN_PACK_K12x8                                  \n"

		"EDGE_PACK_K12x8:                                            \n"

		"    KERNEL12x8_PACK_END_K                                   \n"
		"    jmp      BEGIN_SAVE_12x8                                \n"

		//-----------------------------------------------------------------

		"BEGIN_M12x8:                                                \n"

		"    mov     %%r14, %%rbx                                    \n" // Bc
		"   prefetcht0         (%%rbx)                               \n"

		"    mov      %%rcx, %%r10                                   \n" // C0
		"   prefetcht1         (%%r10)                               \n"
		"    leaq     (%%r10, %%r8, 4), %%r11                        \n" // C1
		"   prefetcht1         (%%r11)                               \n"
		"    leaq     (%%r11, %%r8, 4), %%r12                        \n" // C2
		"   prefetcht1         (%%r12)                               \n"
		"    leaq     (%%r12, %%r8, 4), %%r13                        \n" // C3
		"   prefetcht1         (%%r13)                               \n"

		"    mov     %%rsi, %%rdx                                    \n" // K

		"   vmovups        (%%rbx), %%ymm4                           \n"
		"    vpxorq         %%ymm20, %%ymm20, %%ymm20                \n"
		"    vpxorq         %%ymm21, %%ymm21, %%ymm21                \n"
		"    vpxorq         %%ymm22, %%ymm22, %%ymm22                \n"
		"   vbroadcastss    (%%rax), %%ymm0                          \n"
		"    vpxorq         %%ymm23, %%ymm23, %%ymm23                \n"
		"    vpxorq         %%ymm24, %%ymm24, %%ymm24                \n"
		"   vbroadcastss    4(%%rax), %%ymm1                         \n"
		"    vpxorq         %%ymm25, %%ymm25, %%ymm25                \n"
		"    vpxorq         %%ymm26, %%ymm26, %%ymm26                \n"
		"    vpxorq         %%ymm27, %%ymm27, %%ymm27                \n"
		"    vpxorq         %%ymm28, %%ymm28, %%ymm28                \n"
		"    vpxorq         %%ymm29, %%ymm29, %%ymm29                \n"
		"    vpxorq         %%ymm30, %%ymm30, %%ymm30                \n"
		"    vpxorq         %%ymm31, %%ymm31, %%ymm31                \n"

		"    subq     $8, %%rdx                                      \n"

		"MAIN_K12x8:                                                 \n"

		"    KERNEL12x8_K1                                           \n"
		"    KERNEL12x8_K2                                           \n"
		"    KERNEL12x8_K1                                           \n"
		"    KERNEL12x8_K2                                           \n"
		"    KERNEL12x8_K1                                           \n"
		"    KERNEL12x8_K2                                           \n"
		"    KERNEL12x8_K1                                           \n"
		"   cmp     $0, %%rdx                                        \n"
		"    je         EDGE_K12x8                                   \n"
		"    KERNEL12x8_K2                                           \n"

		"    subq     $8, %%rdx                                      \n"
		"   jmp     MAIN_K12x8                                       \n"

		"EDGE_K12x8:                                                 \n"

		"    KERNEL12x8_END_K                                        \n"

		"BEGIN_SAVE_12x8:                                            \n"
		"    cmp     $0, %%r15                                       \n"
		"    je      SAVE_C12x8                                      \n"
		"    ADD_C_12x8                                              \n"

		"SAVE_C12x8:                                                 \n"
		"    SAVE_12x8                                               \n"

		"    cmpq      $12, %%rdi                                    \n"
		"    jnb     BEGIN_M12x8                                     \n" // 不小于（或等于）则跳转

		//------------------------------------------------------------------

		"BEGIN_M8_N8:                                                \n"
		"   cmpq      $8, %%rdi                                      \n" // M % 8
		"    jb       BEGIN_M4_N8                                    \n" // 小于则跳转

		"    mov     %%r14, %%rbx                                    \n" // Bc
		"   prefetcht0         (%%rbx)                               \n"

		"    mov      %%rcx, %%r10                                   \n" // C0
		"   prefetcht1         (%%r10)                               \n"
		"    leaq     (%%r10, %%r8, 4), %%r11                        \n" // C1
		"   prefetcht1         (%%r11)                               \n"
		"    leaq     (%%r11, %%r8, 4), %%r12                        \n" // C2
		"   prefetcht1         (%%r12)                               \n"
		"    leaq     (%%r12, %%r8, 4), %%r13                        \n" // C3
		"   prefetcht1         (%%r13)                               \n"

		"    mov     %%rsi, %%rdx                                    \n" // K
		"   vmovups        (%%rbx), %%ymm4                           \n"
		"    vpxorq         %%ymm24, %%ymm24, %%ymm24                \n"
		"    vpxorq         %%ymm25, %%ymm25, %%ymm25                \n"
		"   vbroadcastss    (%%rax), %%ymm0                          \n"
		"    vpxorq         %%ymm26, %%ymm26, %%ymm26                \n"
		"    vpxorq         %%ymm27, %%ymm27, %%ymm27                \n"
		"   vbroadcastss    4(%%rax), %%ymm1                         \n"
		"    vpxorq         %%ymm28, %%ymm28, %%ymm28                \n"
		"    vpxorq         %%ymm29, %%ymm29, %%ymm29                \n"
		"    vpxorq         %%ymm30, %%ymm30, %%ymm30                \n"
		"    vpxorq         %%ymm31, %%ymm31, %%ymm31                \n"

		"    subq     $8, %%rdx                                      \n"

		"MAIN_K_M8_N8:                                               \n"

		"    KERNEL8x8_K1                                            \n"
		"    KERNEL8x8_K2                                            \n"
		"    KERNEL8x8_K1                                            \n"
		"    KERNEL8x8_K2                                            \n"
		"    KERNEL8x8_K1                                            \n"
		"    KERNEL8x8_K2                                            \n"
		"    KERNEL8x8_K1                                            \n"
		"   cmp     $0, %%rdx                                        \n"
		"    je         EDGE_K_M8_N8                                 \n"
		"    KERNEL8x8_K2                                            \n"

		"    subq     $8, %%rdx                                      \n"
		"   jmp     MAIN_K_M8_N8                                     \n"

		"EDGE_K_M8_N8:                                               \n"

		"    KERNEL8x8_END_K                                         \n"

		"    cmp     $0, %%r15                                       \n"
		"    je      SAVE_C_8x8                                      \n"
		"    ADD_C_8x8                                               \n"

		"SAVE_C_8x8:                                                 \n"
		"    SAVE_8x8                                                \n"

		//----------------------------------------------------------------

		"BEGIN_M4_N8:                                                \n"

		"    cmpq      $4, %%rdi                                     \n" // M % 4
		"    jb       BEGIN_M1_N8                                    \n"

		"    mov     %%r14, %%rbx                                    \n" // Bc
		"   prefetcht0         (%%rbx)                               \n"

		"    mov      %%rcx, %%r10                                   \n" // C0
		"   prefetcht1         (%%r10)                               \n"
		"    leaq     (%%r10, %%r8, 4), %%r11                        \n" // C1
		"   prefetcht1         (%%r11)                               \n"
		"    leaq     (%%r11, %%r8, 4), %%r12                        \n" // C2
		"   prefetcht1         (%%r12)                               \n"
		"    leaq     (%%r12, %%r8, 4), %%r13                        \n" // C3
		"   prefetcht1         (%%r13)                               \n"

		"    mov     %%rsi, %%rdx                                    \n" // K

		"   vmovups        (%%rbx), %%ymm4                           \n"
		"   vbroadcastss    (%%rax), %%ymm0                          \n"
		"    vpxorq         %%ymm28, %%ymm28, %%ymm28                \n"
		"    vpxorq         %%ymm29, %%ymm29, %%ymm29                \n"
		"   vbroadcastss    4(%%rax), %%ymm1                         \n"
		"    vpxorq         %%ymm30, %%ymm30, %%ymm30                \n"
		"    vpxorq         %%ymm31, %%ymm31, %%ymm31                \n"

		"    subq     $8, %%rdx                                      \n"

		"MAIN_K_M4_N8:                                               \n"

		"    KERNEL4x8_K1                                            \n"
		"    KERNEL4x8_K2                                            \n"
		"    KERNEL4x8_K1                                            \n"
		"    KERNEL4x8_K2                                            \n"
		"    KERNEL4x8_K1                                            \n"
		"    KERNEL4x8_K2                                            \n"
		"    KERNEL4x8_K1                                            \n"
		"   cmp     $0, %%rdx                                        \n"
		"    je         EDGE_K_M4_N8                                 \n"
		"    KERNEL4x8_K2                                            \n"

		"    subq     $8, %%rdx                                      \n"
		"   jmp     MAIN_K_M4_N8                                     \n"

		"EDGE_K_M4_N8:                                               \n"

		"    KERNEL4x8_END_K                                         \n"

		"    cmp     $0, %%r15                                       \n"
		"    je      SAVE_C_4x8                                      \n"
		"    ADD_C_4x8                                               \n"

		"SAVE_C_4x8:                                                 \n"
		"    SAVE_4x8                                                \n"

		//----------------------------------------------------------------

		"BEGIN_M1_N8:                                                \n"
		"    cmpq      $1, %%rdi                                     \n" // M % 1
		"    jb       END_M_N8                                       \n"

		"    mov     %%r14, %%rbx                                    \n" // Bc
		"   prefetcht0         (%%rbx)                               \n"

		"    mov      %%rcx, %%r10                                   \n" // C0
		"   prefetcht1         (%%r10)                               \n"

		"    mov     %%rsi, %%rdx                                    \n" // K

		"   vmovups        (%%rbx), %%ymm4                           \n"
		"   vbroadcastss    (%%rax), %%ymm0                          \n" // A0
		"    vpxorq         %%ymm30, %%ymm30, %%ymm30                \n"

		"    subq     $8, %%rdx                                      \n"

		"MAIN_K_M1_N8:                                               \n"
		"    KERNEL1x8_K1                                            \n"
		"    KERNEL1x8_K2                                            \n"
		"    KERNEL1x8_K1                                            \n"
		"    KERNEL1x8_K2                                            \n"
		"    KERNEL1x8_K1                                            \n"
		"    KERNEL1x8_K2                                            \n"
		"    KERNEL1x8_K1                                            \n"
		"   cmp     $0, %%rdx                                        \n"
		"    je         EDGE_K_M1_N8                                 \n"
		"    KERNEL1x8_K2                                            \n"

		"    subq     $8, %%rdx                                      \n"
		"   jmp     MAIN_K_M1_N8                                     \n"

		"EDGE_K_M1_N8:                                               \n"
		"    KERNEL1x8_END_K                                         \n"

		"    cmp     $0, %%r15                                       \n"
		"    je      SAVE_C_1x8                                      \n"
		"    ADD_C_1x8                                               \n"

		"SAVE_C_1x8:                                                 \n"
		"    SAVE_1x8                                                \n"
		"   cmpq      $1, %%rdi                                      \n"
		"    jnb     BEGIN_M1_N8                                     \n" // 不小于（或等于）则跳转

		//-----------------------------------------------------------------

		"END_M_N8:                                                   \n"

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
		[Bc] "m"(Bc),
		[k_tag] "m"(k_tag)
		: "rax", "rbx", "rcx", "rdx", "rdi", "rsi", "rbp", "r8", "r9", "r10", "r11", "r12",
		  "r13", "r14", "r15", "ymm0", "ymm1", "ymm2", "ymm3", "ymm4", "ymm5",
		  "ymm6", "ymm7", "ymm8", "ymm9", "ymm10", "ymm11", "ymm12", "ymm13",
		  "ymm14", "ymm15", "ymm16", "ymm17", "ymm18", "ymm19", "ymm20", "ymm21",
		  "ymm22", "ymm23", "ymm24", "ymm25", "ymm26", "ymm27", "ymm28", "ymm29",
		  "ymm30", "ymm31", "memory");
}

void SMM_NN_KERNELm12xn4(float *C, float *A, float *B, long M, long N, long K, long LN, long LK, float *Bc, long k_tag)
{
	asm volatile(
		".macro    KERNEL12x4_PACK_K1                                \n"

		"   vbroadcastss    8(%%rax), %%xmm2                         \n"
		"   vfmadd231ps        %%xmm0, %%xmm4, %%xmm20               \n"

		"   vbroadcastss    12(%%rax), %%xmm3                        \n"
		"   vfmadd231ps        %%xmm1, %%xmm4, %%xmm21               \n"

		"   prefetcht0         256(%%rax)                            \n"

		"   vbroadcastss    16(%%rax), %%xmm0                        \n"
		"   vfmadd231ps        %%xmm2, %%xmm4, %%xmm22               \n"

		"   vbroadcastss    20(%%rax), %%xmm1                        \n"
		"   vfmadd231ps        %%xmm3, %%xmm4, %%xmm23               \n"

		"   vbroadcastss    24(%%rax), %%xmm2                        \n"
		"   vfmadd231ps        %%xmm0, %%xmm4, %%xmm24               \n"

		"   vbroadcastss    28(%%rax), %%xmm3                        \n"
		"   vfmadd231ps        %%xmm1, %%xmm4, %%xmm25               \n"

		"   vbroadcastss    32(%%rax), %%xmm0                        \n"
		"   vfmadd231ps        %%xmm2, %%xmm4, %%xmm26               \n"

		"    leaq      (%%rbx, %%r8, 4), %%rbx                       \n" // B

		"   vbroadcastss    36(%%rax), %%xmm1                        \n"
		"   vfmadd231ps        %%xmm3, %%xmm4, %%xmm27               \n"

		"   vbroadcastss    40(%%rax), %%xmm2                        \n"
		"   vfmadd231ps        %%xmm0, %%xmm4, %%xmm28               \n"

		"   vbroadcastss    44(%%rax), %%xmm3                        \n"
		"   vfmadd231ps        %%xmm1, %%xmm4, %%xmm29               \n"
		"   vmovups         (%%rbx), %%xmm6                          \n"
		"    addq              $48, %%rax                            \n"

		"   vbroadcastss    (%%rax), %%xmm0                          \n"
		"   vfmadd231ps        %%xmm2, %%xmm4, %%xmm30               \n"

		"   vbroadcastss    4(%%rax), %%xmm1                         \n"
		"   vfmadd231ps        %%xmm3, %%xmm4, %%xmm31               \n"
		"   vmovups         %%xmm4, (%%rbp)                          \n"
		"    addq              $16, %%rbp                            \n" // 64->16

		".endm                                                       \n"

		".macro    KERNEL12x4_PACK_K2                                \n"

		"   vbroadcastss    8(%%rax), %%xmm2                         \n"
		"   vfmadd231ps        %%xmm0, %%xmm6, %%xmm20               \n"

		"   vbroadcastss    12(%%rax), %%xmm3                        \n"
		"   vfmadd231ps        %%xmm1, %%xmm6, %%xmm21               \n"

		"   vbroadcastss    16(%%rax), %%xmm0                        \n"
		"   vfmadd231ps        %%xmm2, %%xmm6, %%xmm22               \n"

		"   prefetcht0         256(%%rax)                            \n"

		"   vbroadcastss    20(%%rax), %%xmm1                        \n"
		"   vfmadd231ps        %%xmm3, %%xmm6, %%xmm23               \n"

		"   vbroadcastss    24(%%rax), %%xmm2                        \n"
		"   vfmadd231ps        %%xmm0, %%xmm6, %%xmm24               \n"

		"   vbroadcastss    28(%%rax), %%xmm3                        \n"
		"   vfmadd231ps        %%xmm1, %%xmm6, %%xmm25               \n"

		"   vbroadcastss    32(%%rax), %%xmm0                        \n"
		"   vfmadd231ps        %%xmm2, %%xmm6, %%xmm26               \n"

		"    leaq      (%%rbx, %%r8, 4), %%rbx                       \n" // B

		"   vbroadcastss    36(%%rax), %%xmm1                        \n"
		"   vfmadd231ps        %%xmm3, %%xmm6, %%xmm27               \n"

		"   prefetcht0         16(%%rbx)                             \n" //

		"   vbroadcastss    40(%%rax), %%xmm2                        \n"
		"   vfmadd231ps        %%xmm0, %%xmm6, %%xmm28               \n"

		"   vbroadcastss    44(%%rax), %%xmm3                        \n"
		"   vfmadd231ps        %%xmm1, %%xmm6, %%xmm29               \n"
		"   vmovups         (%%rbx), %%xmm4                          \n"
		"    addq              $48, %%rax                            \n"

		"   vbroadcastss    (%%rax), %%xmm0                          \n"
		"   vfmadd231ps        %%xmm2, %%xmm6, %%xmm30               \n"

		"   vbroadcastss    4(%%rax), %%xmm1                         \n"
		"   vfmadd231ps        %%xmm3, %%xmm6, %%xmm31               \n"
		"   vmovups         %%xmm6, (%%rbp)                          \n"
		"    addq              $16, %%rbp                            \n" // 64->16

		".endm                                                       \n"

		".macro    KERNEL12x4_PACK_END_K                             \n"

		"   vbroadcastss    8(%%rax), %%xmm2                         \n"
		"   vfmadd231ps        %%xmm0, %%xmm6, %%xmm20               \n"

		"   vbroadcastss    12(%%rax), %%xmm3                        \n"
		"   vfmadd231ps        %%xmm1, %%xmm6, %%xmm21               \n"

		"   vbroadcastss    16(%%rax), %%xmm0                        \n"
		"   vfmadd231ps        %%xmm2, %%xmm6, %%xmm22               \n"

		"   prefetcht0         256(%%rax)                            \n"

		"   vbroadcastss    20(%%rax), %%xmm1                        \n"
		"   vfmadd231ps        %%xmm3, %%xmm6, %%xmm23               \n"

		"   vbroadcastss    24(%%rax), %%xmm2                        \n"
		"   vfmadd231ps        %%xmm0, %%xmm6, %%xmm24               \n"

		"   vbroadcastss    28(%%rax), %%xmm3                        \n"
		"   vfmadd231ps        %%xmm1, %%xmm6, %%xmm25               \n"

		"   vbroadcastss    32(%%rax), %%xmm0                        \n"
		"   vfmadd231ps        %%xmm2, %%xmm6, %%xmm26               \n"

		"   vbroadcastss    36(%%rax), %%xmm1                        \n"
		"   vfmadd231ps        %%xmm3, %%xmm6, %%xmm27               \n"

		"   vbroadcastss    40(%%rax), %%xmm2                        \n"
		"   vfmadd231ps        %%xmm0, %%xmm6, %%xmm28               \n"

		"   vbroadcastss    44(%%rax), %%xmm3                        \n"
		"   vfmadd231ps        %%xmm1, %%xmm6, %%xmm29               \n"
		"    addq              $48, %%rax                            \n"

		"   vfmadd231ps        %%xmm2, %%xmm6, %%xmm30               \n"
		"   vmovups         %%xmm6, (%%rbp)                          \n"
		"   vfmadd231ps        %%xmm3, %%xmm6, %%xmm31               \n"

		".endm                                                       \n"

		//-----------------------------------------------------------------------

		".macro    KERNEL12x4_K1                                     \n"

		"   vbroadcastss    8(%%rax), %%xmm2                         \n"
		"   vfmadd231ps        %%xmm0, %%xmm4, %%xmm20               \n"

		"   vbroadcastss    12(%%rax), %%xmm3                        \n"
		"   vfmadd231ps        %%xmm1, %%xmm4, %%xmm21               \n"

		"   prefetcht0         256(%%rax)                            \n"

		"   vbroadcastss    16(%%rax), %%xmm0                        \n"
		"   vfmadd231ps        %%xmm2, %%xmm4, %%xmm22               \n"

		"   vbroadcastss    20(%%rax), %%xmm1                        \n"
		"   vfmadd231ps        %%xmm3, %%xmm4, %%xmm23               \n"

		"   vbroadcastss    24(%%rax), %%xmm2                        \n"
		"   vfmadd231ps        %%xmm0, %%xmm4, %%xmm24               \n"

		"   vbroadcastss    28(%%rax), %%xmm3                        \n"
		"   vfmadd231ps        %%xmm1, %%xmm4, %%xmm25               \n"

		"   vbroadcastss    32(%%rax), %%xmm0                        \n"
		"   vfmadd231ps        %%xmm2, %%xmm4, %%xmm26               \n"

		"    addq              $16, %%rbx                            \n" // B

		"   vbroadcastss    36(%%rax), %%xmm1                        \n"
		"   vfmadd231ps        %%xmm3, %%xmm4, %%xmm27               \n"

		"   vbroadcastss    40(%%rax), %%xmm2                        \n"
		"   vfmadd231ps        %%xmm0, %%xmm4, %%xmm28               \n"

		"   vbroadcastss    44(%%rax), %%xmm3                        \n"
		"   vfmadd231ps        %%xmm1, %%xmm4, %%xmm29               \n"
		"   vmovups         (%%rbx), %%xmm6                          \n"
		"    addq              $48, %%rax                            \n"

		"   vbroadcastss    (%%rax), %%xmm0                          \n"
		"   vfmadd231ps        %%xmm2, %%xmm4, %%xmm30               \n"

		"   vbroadcastss    4(%%rax), %%xmm1                         \n"
		"   vfmadd231ps        %%xmm3, %%xmm4, %%xmm31               \n"

		".endm                                                       \n"

		".macro    KERNEL12x4_K2                                     \n"

		"   vbroadcastss    8(%%rax), %%xmm2                         \n"
		"   vfmadd231ps        %%xmm0, %%xmm6, %%xmm20               \n"

		"   vbroadcastss    12(%%rax), %%xmm3                        \n"
		"   vfmadd231ps        %%xmm1, %%xmm6, %%xmm21               \n"

		"   vbroadcastss    16(%%rax), %%xmm0                        \n"
		"   vfmadd231ps        %%xmm2, %%xmm6, %%xmm22               \n"

		"   prefetcht0         256(%%rax)                            \n"

		"   vbroadcastss    20(%%rax), %%xmm1                        \n"
		"   vfmadd231ps        %%xmm3, %%xmm6, %%xmm23               \n"

		"   vbroadcastss    24(%%rax), %%xmm2                        \n"
		"   vfmadd231ps        %%xmm0, %%xmm6, %%xmm24               \n"

		"   vbroadcastss    28(%%rax), %%xmm3                        \n"
		"   vfmadd231ps        %%xmm1, %%xmm6, %%xmm25               \n"

		"   vbroadcastss    32(%%rax), %%xmm0                        \n"
		"   vfmadd231ps        %%xmm2, %%xmm6, %%xmm26               \n"

		"    addq             $16, %%rbx                             \n" //

		"   vbroadcastss    36(%%rax), %%xmm1                        \n"
		"   vfmadd231ps        %%xmm3, %%xmm6, %%xmm27               \n"

		"   prefetcht0         16(%%rbx)                             \n" //

		"   vbroadcastss    40(%%rax), %%xmm2                        \n"
		"   vfmadd231ps        %%xmm0, %%xmm6, %%xmm28               \n"

		"   vbroadcastss    44(%%rax), %%xmm3                        \n"
		"   vfmadd231ps        %%xmm1, %%xmm6, %%xmm29               \n"
		"   vmovups         (%%rbx), %%xmm4                          \n"
		"    addq              $48, %%rax                            \n"

		"   vbroadcastss    (%%rax), %%xmm0                          \n"
		"   vfmadd231ps        %%xmm2, %%xmm6, %%xmm30               \n"

		"   vbroadcastss    4(%%rax), %%xmm1                         \n"
		"   vfmadd231ps        %%xmm3, %%xmm6, %%xmm31               \n"

		".endm                                                       \n"

		".macro    KERNEL12x4_END_K                                  \n"

		"   vbroadcastss    8(%%rax), %%xmm2                         \n"
		"   vfmadd231ps        %%xmm0, %%xmm6, %%xmm20               \n"

		"   vbroadcastss    12(%%rax), %%xmm3                        \n"
		"   vfmadd231ps        %%xmm1, %%xmm6, %%xmm21               \n"

		"   vbroadcastss    16(%%rax), %%xmm0                        \n"
		"   vfmadd231ps        %%xmm2, %%xmm6, %%xmm22               \n"

		"   prefetcht0         256(%%rax)                            \n"

		"   vbroadcastss    20(%%rax), %%xmm1                        \n"
		"   vfmadd231ps        %%xmm3, %%xmm6, %%xmm23               \n"

		"   vbroadcastss    24(%%rax), %%xmm2                        \n"
		"   vfmadd231ps        %%xmm0, %%xmm6, %%xmm24               \n"

		"   vbroadcastss    28(%%rax), %%xmm3                        \n"
		"   vfmadd231ps        %%xmm1, %%xmm6, %%xmm25               \n"

		"   vbroadcastss    32(%%rax), %%xmm0                        \n"
		"   vfmadd231ps        %%xmm2, %%xmm6, %%xmm26               \n"

		"   vbroadcastss    36(%%rax), %%xmm1                        \n"
		"   vfmadd231ps        %%xmm3, %%xmm6, %%xmm27               \n"

		"   vbroadcastss    40(%%rax), %%xmm2                        \n"
		"   vfmadd231ps        %%xmm0, %%xmm6, %%xmm28               \n"

		"   vbroadcastss    44(%%rax), %%xmm3                        \n"
		"   vfmadd231ps        %%xmm1, %%xmm6, %%xmm29               \n"
		"   vmovups         (%%rbx), %%xmm4                          \n"
		"    addq              $48, %%rax                            \n"

		"   vbroadcastss    (%%rax), %%xmm0                          \n"
		"   vfmadd231ps        %%xmm2, %%xmm6, %%xmm30               \n"

		"   vbroadcastss    4(%%rax), %%xmm1                         \n"
		"   vfmadd231ps        %%xmm3, %%xmm6, %%xmm31               \n"

		".endm                                                       \n"

		".macro    ADD_C_12x4                                        \n"

		"   vmovups         (%%r10), %%xmm0                          \n"
		"    vaddps             %%xmm0, %%xmm20, %%xmm20             \n"
		"   vmovups         (%%r11), %%xmm1                          \n"
		"    vaddps             %%xmm1, %%xmm21, %%xmm21             \n"
		"   vmovups         (%%r12), %%xmm2                          \n"
		"    vaddps             %%xmm2, %%xmm22, %%xmm22             \n"
		"   vmovups         (%%r13), %%xmm3                          \n"
		"    vaddps             %%xmm3, %%xmm23, %%xmm23             \n"

		"    leaq              (%%r13, %%r8, 4), %%r10               \n" // C0
		"    leaq             (%%r10, %%r8, 4), %%r11                \n" // C1
		"    leaq             (%%r11, %%r8, 4), %%r12                \n" // C2
		"    leaq             (%%r12, %%r8, 4), %%r13                \n" // C3

		"   vmovups         (%%r10), %%xmm4                          \n"
		"    vaddps             %%xmm4, %%xmm24, %%xmm24             \n"
		"   vmovups         (%%r11), %%xmm5                          \n"
		"    vaddps             %%xmm5, %%xmm25, %%xmm25             \n"
		"   vmovups         (%%r12), %%xmm6                          \n"
		"    vaddps             %%xmm6, %%xmm26, %%xmm26             \n"
		"   vmovups         (%%r13), %%xmm7                          \n"
		"    vaddps             %%xmm7, %%xmm27, %%xmm27             \n"

		"    leaq              (%%r13, %%r8, 4), %%r10               \n" // C0
		"    leaq             (%%r10, %%r8, 4), %%r11                \n" // C1
		"    leaq             (%%r11, %%r8, 4), %%r12                \n" // C2
		"    leaq             (%%r12, %%r8, 4), %%r13                \n" // C3

		"   vmovups         (%%r10), %%xmm0                          \n"
		"    vaddps             %%xmm0, %%xmm28, %%xmm28             \n"
		"   vmovups         (%%r11), %%xmm1                          \n"
		"    vaddps             %%xmm1, %%xmm29, %%xmm29             \n"
		"   vmovups         (%%r12), %%xmm2                          \n"
		"    vaddps             %%xmm2, %%xmm30, %%xmm30             \n"
		"   vmovups         (%%r13), %%xmm3                          \n"
		"    vaddps             %%xmm3, %%xmm31, %%xmm31             \n"

		"    mov      %%rcx, %%r10                                   \n" // C0
		"    leaq     (%%r10, %%r8, 4), %%r11                        \n" // C1
		"    leaq     (%%r11, %%r8, 4), %%r12                        \n" // C2
		"    leaq     (%%r12, %%r8, 4), %%r13                        \n" // C3

		".endm                                                       \n"

		".macro    SAVE_12x4                                         \n"

		"   vmovups         %%xmm20, (%%r10)                         \n"
		"   vmovups         %%xmm21, (%%r11)                         \n"
		"    subq             $12, %%rdi                             \n"
		"   vmovups         %%xmm22, (%%r12)                         \n"
		"   vmovups         %%xmm23, (%%r13)                         \n"

		"    leaq      (%%r13, %%r8, 4), %%r10                       \n" // C0
		"    leaq     (%%r10, %%r8, 4), %%r11                        \n" // C1
		"    leaq     (%%r11, %%r8, 4), %%r12                        \n" // C2
		"    leaq     (%%r12, %%r8, 4), %%r13                        \n" // C3

		"   vmovups         %%xmm24, (%%r10)                         \n"
		"   vmovups         %%xmm25, (%%r11)                         \n"
		"   vmovups         %%xmm26, (%%r12)                         \n"
		"   vmovups         %%xmm27, (%%r13)                         \n"

		"    leaq      (%%r13, %%r8, 4), %%r10                       \n" // C0
		"    leaq     (%%r10, %%r8, 4), %%r11                        \n" // C1
		"    leaq     (%%r11, %%r8, 4), %%r12                        \n" // C2
		"    leaq     (%%r12, %%r8, 4), %%r13                        \n" // C3

		"   vmovups         %%xmm28, (%%r10)                         \n"
		"   vmovups         %%xmm29, (%%r11)                         \n"
		"   vmovups         %%xmm30, (%%r12)                         \n"
		"   vmovups         %%xmm31, (%%r13)                         \n"

		"    leaq      (%%r13, %%r8, 4), %%rcx                       \n" // C0

		".endm                                                       \n"

		//-----------------------------------------------------------------------

		".macro    KERNEL8x4_K1                                      \n"

		"   vbroadcastss    8(%%rax), %%xmm2                         \n"
		"   vfmadd231ps        %%xmm0, %%xmm4, %%xmm24               \n"

		"   vbroadcastss    12(%%rax), %%xmm3                        \n"
		"   vfmadd231ps        %%xmm1, %%xmm4, %%xmm25               \n"

		"   prefetcht0         256(%%rax)                            \n"

		"   vbroadcastss    16(%%rax), %%xmm8                        \n"
		"   vfmadd231ps        %%xmm2, %%xmm4, %%xmm26               \n"

		"   vbroadcastss    20(%%rax), %%xmm9                        \n"
		"   vfmadd231ps        %%xmm3, %%xmm4, %%xmm27               \n"

		"   vbroadcastss    24(%%rax), %%xmm10                       \n"
		"   vfmadd231ps        %%xmm8, %%xmm4, %%xmm28               \n"

		"   prefetcht0         16(%%rbx)                             \n" //

		"   vbroadcastss    28(%%rax), %%xmm11                       \n"
		"   vfmadd231ps        %%xmm9, %%xmm4, %%xmm29               \n"
		"    addq              $16, %%rbx                            \n" //
		"   vfmadd231ps        %%xmm10, %%xmm4, %%xmm30              \n"

		"    addq              $32, %%rax                            \n"
		"   vmovups         (%%rbx), %%xmm6                          \n"
		"   vfmadd231ps        %%xmm11, %%xmm4, %%xmm31              \n"

		"   vbroadcastss    (%%rax), %%xmm0                          \n"
		"   vbroadcastss    4(%%rax), %%xmm1                         \n"

		".endm                                                       \n"

		".macro    KERNEL8x4_K2                                      \n"

		"   vbroadcastss    8(%%rax), %%xmm2                         \n"
		"   vfmadd231ps        %%xmm0, %%xmm6, %%xmm24               \n"

		"   vbroadcastss    12(%%rax), %%xmm3                        \n"
		"   vfmadd231ps        %%xmm1, %%xmm6, %%xmm25               \n"

		"   prefetcht0         256(%%rax)                            \n"

		"   vbroadcastss    16(%%rax), %%xmm8                        \n"
		"   vfmadd231ps        %%xmm2, %%xmm6, %%xmm26               \n"

		"   vbroadcastss    20(%%rax), %%xmm9                        \n"
		"   vfmadd231ps        %%xmm3, %%xmm6, %%xmm27               \n"

		"   vbroadcastss    24(%%rax), %%xmm10                       \n"
		"   vfmadd231ps        %%xmm8, %%xmm6, %%xmm28               \n"

		"   prefetcht0         16(%%rbx)                             \n" //

		"   vbroadcastss    28(%%rax), %%xmm11                       \n"
		"   vfmadd231ps        %%xmm9, %%xmm6, %%xmm29               \n"
		"    addq              $16, %%rbx                            \n" // B
		"   vfmadd231ps        %%xmm10, %%xmm6, %%xmm30              \n"

		"    addq              $32, %%rax                            \n"
		"   vmovups         (%%rbx), %%xmm4                          \n"
		"   vfmadd231ps        %%xmm11, %%xmm6, %%xmm31              \n"

		"   vbroadcastss    (%%rax), %%xmm0                          \n"
		"   vbroadcastss    4(%%rax), %%xmm1                         \n"

		".endm                                                       \n"

		".macro    KERNEL8x4_END_K                                   \n"

		"   vbroadcastss    8(%%rax), %%xmm2                         \n"
		"   vfmadd231ps        %%xmm0, %%xmm6, %%xmm24               \n"

		"   vbroadcastss    12(%%rax), %%xmm3                        \n"
		"   vfmadd231ps        %%xmm1, %%xmm6, %%xmm25               \n"

		"   prefetcht0         256(%%rax)                            \n"

		"   vbroadcastss    16(%%rax), %%xmm8                        \n"
		"   vfmadd231ps        %%xmm2, %%xmm6, %%xmm26               \n"

		"   vbroadcastss    20(%%rax), %%xmm9                        \n"
		"   vfmadd231ps        %%xmm3, %%xmm6, %%xmm27               \n"

		"   vbroadcastss    24(%%rax), %%xmm10                       \n"
		"   vfmadd231ps        %%xmm8, %%xmm6, %%xmm28               \n"

		"   vbroadcastss    28(%%rax), %%xmm11                       \n"
		"   vfmadd231ps        %%xmm9, %%xmm6, %%xmm29               \n"
		"   vfmadd231ps        %%xmm10, %%xmm6, %%xmm30              \n"

		"    addq              $32, %%rax                            \n"
		"   vfmadd231ps        %%xmm11, %%xmm6, %%xmm31              \n"

		".endm                                                       \n"

		".macro    ADD_C_8x4                                         \n"

		"   vmovups         (%%r10), %%xmm4                          \n"
		"    vaddps             %%xmm4, %%xmm24, %%xmm24             \n"
		"   vmovups         (%%r11), %%xmm5                          \n"
		"    vaddps             %%xmm5, %%xmm25, %%xmm25             \n"
		"   vmovups         (%%r12), %%xmm6                          \n"
		"    vaddps             %%xmm6, %%xmm26, %%xmm26             \n"
		"   vmovups         (%%r13), %%xmm7                          \n"
		"    vaddps             %%xmm7, %%xmm27, %%xmm27             \n"

		"    leaq              (%%r13, %%r8, 4), %%r10               \n" // C0
		"    leaq             (%%r10, %%r8, 4), %%r11                \n" // C1
		"    leaq             (%%r11, %%r8, 4), %%r12                \n" // C2
		"    leaq             (%%r12, %%r8, 4), %%r13                \n" // C3

		"   vmovups         (%%r10), %%xmm0                          \n"
		"    vaddps             %%xmm0, %%xmm28, %%xmm28             \n"
		"   vmovups         (%%r11), %%xmm1                          \n"
		"    vaddps             %%xmm1, %%xmm29, %%xmm29             \n"
		"   vmovups         (%%r12), %%xmm2                          \n"
		"    vaddps             %%xmm2, %%xmm30, %%xmm30             \n"
		"   vmovups         (%%r13), %%xmm3                          \n"
		"    vaddps             %%xmm3, %%xmm31, %%xmm31             \n"

		"    mov      %%rcx, %%r10                                   \n" // C0
		"    leaq     (%%r10, %%r8, 4), %%r11                        \n" // C1
		"    leaq     (%%r11, %%r8, 4), %%r12                        \n" // C2
		"    leaq     (%%r12, %%r8, 4), %%r13                        \n" // C3

		".endm                                                       \n"

		".macro    SAVE_8x4                                          \n"

		"   vmovups         %%xmm24, (%%r10)                         \n"
		"   vmovups         %%xmm25, (%%r11)                         \n"
		"    subq             $8, %%rdi                              \n"
		"   vmovups         %%xmm26, (%%r12)                         \n"
		"   vmovups         %%xmm27, (%%r13)                         \n"

		"    leaq      (%%r13, %%r8, 4), %%r10                       \n" // C0
		"    leaq     (%%r10, %%r8, 4), %%r11                        \n" // C1
		"    leaq     (%%r11, %%r8, 4), %%r12                        \n" // C2
		"    leaq     (%%r12, %%r8, 4), %%r13                        \n" // C3

		"   vmovups         %%xmm28, (%%r10)                         \n"
		"   vmovups         %%xmm29, (%%r11)                         \n"
		"   vmovups         %%xmm30, (%%r12)                         \n"
		"   vmovups         %%xmm31, (%%r13)                         \n"

		"    leaq      (%%r13, %%r8, 4), %%rcx                       \n" // C0

		".endm                                                       \n"

		//-----------------------------------------------------------------------

		".macro    KERNEL4x4_K1                                      \n"

		"   vbroadcastss    8(%%rax), %%xmm2                         \n"
		"   vfmadd231ps        %%xmm0, %%xmm4, %%xmm28               \n"

		"   vbroadcastss    12(%%rax), %%xmm3                        \n"
		"   vfmadd231ps        %%xmm1, %%xmm4, %%xmm29               \n"
		"    addq              $16, %%rax                            \n"

		"   prefetcht0         256(%%rax)                            \n"
		"    addq              $16, %%rbx                            \n" // B
		"   vbroadcastss    (%%rax), %%xmm0                          \n"
		"   vfmadd231ps        %%xmm2, %%xmm4, %%xmm30               \n"

		"   vbroadcastss    4(%%rax), %%xmm1                         \n"
		"   vfmadd231ps        %%xmm3, %%xmm4, %%xmm31               \n"

		"   prefetcht0         16(%%rbx)                             \n"
		"   vmovups         (%%rbx), %%xmm6                          \n"

		".endm                                                       \n"

		".macro    KERNEL4x4_K2                                      \n"

		"   vbroadcastss    8(%%rax), %%xmm2                         \n"
		"   vfmadd231ps        %%xmm0, %%xmm6, %%xmm28               \n"

		"   vbroadcastss    12(%%rax), %%xmm3                        \n"
		"   vfmadd231ps        %%xmm1, %%xmm6, %%xmm29               \n"
		"    addq              $16, %%rax                            \n"

		"   prefetcht0         256(%%rax)                            \n"
		"    addq              $16, %%rbx                            \n" // B
		"   vbroadcastss    (%%rax), %%xmm0                          \n"
		"   vfmadd231ps        %%xmm2, %%xmm6, %%xmm30               \n"

		"   vbroadcastss    4(%%rax), %%xmm1                         \n"
		"   vfmadd231ps        %%xmm3, %%xmm6, %%xmm31               \n"

		"   vmovups         (%%rbx), %%xmm4                          \n"
		"   prefetcht0         16(%%rbx)                             \n"

		".endm                                                       \n"

		".macro    KERNEL4x4_END_K                                   \n"

		"   vbroadcastss    8(%%rax), %%xmm2                         \n"
		"   vfmadd231ps        %%xmm0, %%xmm6, %%xmm28               \n"

		"   vbroadcastss    12(%%rax), %%xmm3                        \n"
		"   vfmadd231ps        %%xmm1, %%xmm6, %%xmm29               \n"
		"    addq              $16, %%rax                            \n"

		"   vfmadd231ps        %%xmm2, %%xmm6, %%xmm30               \n"
		"   vfmadd231ps        %%xmm3, %%xmm6, %%xmm31               \n"

		".endm                                                       \n"

		".macro    ADD_C_4x4                                         \n"

		"   vmovups         (%%r10), %%xmm0                          \n"
		"    vaddps             %%xmm0, %%xmm28, %%xmm28             \n"
		"   vmovups         (%%r11), %%xmm1                          \n"
		"    vaddps             %%xmm1, %%xmm29, %%xmm29             \n"
		"   vmovups         (%%r12), %%xmm2                          \n"
		"    vaddps             %%xmm2, %%xmm30, %%xmm30             \n"
		"   vmovups         (%%r13), %%xmm3                          \n"
		"    vaddps             %%xmm3, %%xmm31, %%xmm31             \n"

		"    mov      %%rcx, %%r10                                   \n" // C0
		"    leaq     (%%r10, %%r8, 4), %%r11                        \n" // C1
		"    leaq     (%%r11, %%r8, 4), %%r12                        \n" // C2
		"    leaq     (%%r12, %%r8, 4), %%r13                        \n" // C3

		".endm                                                       \n"

		".macro    SAVE_4x4                                          \n"
		"   subq             $4, %%rdi                               \n"

		"   vmovups         %%xmm28, (%%r10)                         \n"
		"   vmovups         %%xmm29, (%%r11)                         \n"
		"   vmovups         %%xmm30, (%%r12)                         \n"
		"   vmovups         %%xmm31, (%%r13)                         \n"

		"    leaq      (%%r13, %%r8, 4), %%rcx                       \n" // C0

		".endm                                                       \n"

		//-----------------------------------------------------------------------

		".macro    KERNEL1x4_K1                                      \n"
		"   prefetcht0         256(%%rax)                            \n"
		"   addq              $4, %%rax                              \n"

		"   vbroadcastss    (%%rax), %%xmm2                          \n"
		"   vfmadd231ps        %%xmm0, %%xmm4, %%xmm30               \n"

		"    addq              $16, %%rbx                            \n" // B
		"   vmovups            (%%rbx), %%xmm6                       \n"
		"   prefetcht0         16(%%rbx)                             \n"

		".endm                                                       \n"

		".macro    KERNEL1x4_K2                                      \n"
		"   prefetcht0         256(%%rax)                            \n"
		"    addq              $4, %%rax                             \n"

		"   vbroadcastss    (%%rax), %%xmm0                          \n"
		"   vfmadd231ps        %%xmm2, %%xmm6, %%xmm30               \n"

		"    addq              $16, %%rbx                            \n" // B
		"   vmovups            (%%rbx), %%xmm4                       \n"
		"   prefetcht0         16(%%rbx)                             \n"

		".endm                                                       \n"

		".macro    KERNEL1x4_END_K                                   \n"
		"   vfmadd231ps        %%xmm2, %%xmm6, %%xmm30               \n"
		"    addq              $4, %%rax                             \n"
		".endm                                                       \n"

		".macro    ADD_C_1x4                                         \n"
		"   vmovups         (%%r10), %%xmm2                          \n"
		"    vaddps             %%xmm2, %%xmm30, %%xmm30             \n"
		"    mov      %%rcx, %%r10                                   \n" // C0
		".endm                                                       \n"

		".macro    SAVE_1x4                                          \n"
		"   vmovups         %%xmm30, (%%r10)                         \n"
		"   subq             $1, %%rdi                               \n"
		"    leaq      (%%r10, %%r8, 4), %%rcx                       \n" // C0

		".endm                                                       \n"

		//-----------------------------------------------------------------

		"SMM_NN_KERNEL12x4:                                          \n"

		"   mov     %[C], %%rcx                                      \n"
		"   mov     %[A], %%rax                                      \n"
		"   mov     %[B], %%rbx                                      \n"

		"   prefetcht0         (%%rax)                               \n"

		"    mov     %[K], %%rdx                                     \n" // K
		"    mov      %[LN], %%r8                                    \n"
		"    mov      %[Bc], %%r14                                   \n"
		"    mov      %[M], %%rdi                                    \n"
		"    mov     %[k_tag], %%r15                                 \n"

		"   prefetcht0         (%%rbx)                               \n"
		"    mov     %%rbx, %%r9                                     \n" // B
		"    mov     %%rdx, %%rsi                                    \n" // K

		"BEGIN_PACK12x4:                                             \n"

		"    mov     %%r9, %%rbx                                     \n" // B
		"   prefetcht0         (%%rbx)                               \n"

		"    mov      %%rcx, %%r10                                   \n" // C0
		"   prefetcht1         (%%r10)                               \n"
		"    leaq     (%%r10, %%r8, 4), %%r11                        \n" // C1
		"   prefetcht1         (%%r11)                               \n"
		"    leaq     (%%r11, %%r8, 4), %%r12                        \n" // C2
		"   prefetcht1         (%%r12)                               \n"
		"    leaq     (%%r12, %%r8, 4), %%r13                        \n" // C3
		"   prefetcht1         (%%r13)                               \n"

		"    mov     %%rsi, %%rdx                                    \n" // K
		"    mov     %%r14, %%rbp                                    \n" // Bc

		"   vmovups        (%%rbx), %%xmm4                           \n"
		"    vpxorq         %%xmm20, %%xmm20, %%xmm20                \n"
		"    vpxorq         %%xmm21, %%xmm21, %%xmm21                \n"
		"   vbroadcastss    (%%rax), %%xmm0                          \n"
		"    vpxorq         %%xmm22, %%xmm22, %%xmm22                \n"
		"    vpxorq         %%xmm23, %%xmm23, %%xmm23                \n"
		"   vbroadcastss    4(%%rax), %%xmm1                         \n"
		"    vpxorq         %%xmm24, %%xmm24, %%xmm24                \n"
		"    vpxorq         %%xmm25, %%xmm25, %%xmm25                \n"
		"    vpxorq         %%xmm26, %%xmm26, %%xmm26                \n"
		"    vpxorq         %%xmm27, %%xmm27, %%xmm27                \n"
		"    vpxorq         %%xmm28, %%xmm28, %%xmm28                \n"
		"    vpxorq         %%xmm29, %%xmm29, %%xmm29                \n"
		"    vpxorq         %%xmm30, %%xmm30, %%xmm30                \n"
		"    vpxorq         %%xmm31, %%xmm31, %%xmm31                \n"

		"    subq     $8, %%rdx                                      \n"

		"MAIN_PACK_K12x4:                                            \n"

		"    KERNEL12x4_PACK_K1                                      \n"
		"    KERNEL12x4_PACK_K2                                      \n"
		"    KERNEL12x4_PACK_K1                                      \n"
		"    KERNEL12x4_PACK_K2                                      \n"
		"    KERNEL12x4_PACK_K1                                      \n"
		"    KERNEL12x4_PACK_K2                                      \n"
		"    KERNEL12x4_PACK_K1                                      \n"
		"   cmp     $0, %%rdx                                        \n"
		"    je         EDGE_PACK_K12x4                              \n"
		"    KERNEL12x4_PACK_K2                                      \n"

		"    subq     $8, %%rdx                                      \n"
		"   jmp     MAIN_PACK_K12x4                                  \n"

		"EDGE_PACK_K12x4:                                            \n"

		"    KERNEL12x4_PACK_END_K                                   \n"
		"    jmp      BEGIN_SAVE_12x4                                \n"

		//-----------------------------------------------------------------

		"BEGIN_M12x4:                                                \n"

		"    mov     %%r14, %%rbx                                    \n" // Bc
		"   prefetcht0         (%%rbx)                               \n"

		"    mov      %%rcx, %%r10                                   \n" // C0
		"   prefetcht1         (%%r10)                               \n"
		"    leaq     (%%r10, %%r8, 4), %%r11                        \n" // C1
		"   prefetcht1         (%%r11)                               \n"
		"    leaq     (%%r11, %%r8, 4), %%r12                        \n" // C2
		"   prefetcht1         (%%r12)                               \n"
		"    leaq     (%%r12, %%r8, 4), %%r13                        \n" // C3
		"   prefetcht1         (%%r13)                               \n"

		"    mov     %%rsi, %%rdx                                    \n" // K

		"   vmovups        (%%rbx), %%xmm4                           \n"
		"    vpxorq         %%xmm20, %%xmm20, %%xmm20                \n"
		"    vpxorq         %%xmm21, %%xmm21, %%xmm21                \n"
		"    vpxorq         %%xmm22, %%xmm22, %%xmm22                \n"
		"   vbroadcastss    (%%rax), %%xmm0                          \n"
		"    vpxorq         %%xmm23, %%xmm23, %%xmm23                \n"
		"    vpxorq         %%xmm24, %%xmm24, %%xmm24                \n"
		"   vbroadcastss    4(%%rax), %%xmm1                         \n"
		"    vpxorq         %%xmm25, %%xmm25, %%xmm25                \n"
		"    vpxorq         %%xmm26, %%xmm26, %%xmm26                \n"
		"    vpxorq         %%xmm27, %%xmm27, %%xmm27                \n"
		"    vpxorq         %%xmm28, %%xmm28, %%xmm28                \n"
		"    vpxorq         %%xmm29, %%xmm29, %%xmm29                \n"
		"    vpxorq         %%xmm30, %%xmm30, %%xmm30                \n"
		"    vpxorq         %%xmm31, %%xmm31, %%xmm31                \n"

		"    subq     $8, %%rdx                                      \n"

		"MAIN_K12x4:                                                 \n"

		"    KERNEL12x4_K1                                           \n"
		"    KERNEL12x4_K2                                           \n"
		"    KERNEL12x4_K1                                           \n"
		"    KERNEL12x4_K2                                           \n"
		"    KERNEL12x4_K1                                           \n"
		"    KERNEL12x4_K2                                           \n"
		"    KERNEL12x4_K1                                           \n"
		"   cmp     $0, %%rdx                                        \n"
		"    je         EDGE_K12x4                                   \n"
		"    KERNEL12x4_K2                                           \n"

		"    subq     $8, %%rdx                                      \n"
		"   jmp     MAIN_K12x4                                       \n"

		"EDGE_K12x4:                                                 \n"

		"    KERNEL12x4_END_K                                        \n"

		"BEGIN_SAVE_12x4:                                            \n"
		"    cmp     $0, %%r15                                       \n"
		"    je      SAVE_C12x4                                      \n"
		"    ADD_C_12x4                                              \n"

		"SAVE_C12x4:                                                 \n"
		"    SAVE_12x4                                               \n"

		"    cmpq      $12, %%rdi                                    \n"
		"    jnb     BEGIN_M12x4                                     \n" // 不小于（或等于）则跳转

		//------------------------------------------------------------------

		"BEGIN_M8_N4:                                                \n"
		"   cmpq      $8, %%rdi                                      \n" // M % 8
		"    jb       BEGIN_M4_N4                                    \n" // 小于则跳转

		"    mov     %%r14, %%rbx                                    \n" // Bc
		"   prefetcht0         (%%rbx)                               \n"

		"    mov      %%rcx, %%r10                                   \n" // C0
		"   prefetcht1         (%%r10)                               \n"
		"    leaq     (%%r10, %%r8, 4), %%r11                        \n" // C1
		"   prefetcht1         (%%r11)                               \n"
		"    leaq     (%%r11, %%r8, 4), %%r12                        \n" // C2
		"   prefetcht1         (%%r12)                               \n"
		"    leaq     (%%r12, %%r8, 4), %%r13                        \n" // C3
		"   prefetcht1         (%%r13)                               \n"

		"    mov     %%rsi, %%rdx                                    \n" // K
		"   vmovups        (%%rbx), %%xmm4                           \n"
		"    vpxorq         %%xmm24, %%xmm24, %%xmm24                \n"
		"    vpxorq         %%xmm25, %%xmm25, %%xmm25                \n"
		"   vbroadcastss    (%%rax), %%xmm0                          \n"
		"    vpxorq         %%xmm26, %%xmm26, %%xmm26                \n"
		"    vpxorq         %%xmm27, %%xmm27, %%xmm27                \n"
		"   vbroadcastss    4(%%rax), %%xmm1                         \n"
		"    vpxorq         %%xmm28, %%xmm28, %%xmm28                \n"
		"    vpxorq         %%xmm29, %%xmm29, %%xmm29                \n"
		"    vpxorq         %%xmm30, %%xmm30, %%xmm30                \n"
		"    vpxorq         %%xmm31, %%xmm31, %%xmm31                \n"

		"    subq     $8, %%rdx                                      \n"

		"MAIN_K_M8_N4:                                               \n"

		"    KERNEL8x4_K1                                            \n"
		"    KERNEL8x4_K2                                            \n"
		"    KERNEL8x4_K1                                            \n"
		"    KERNEL8x4_K2                                            \n"
		"    KERNEL8x4_K1                                            \n"
		"    KERNEL8x4_K2                                            \n"
		"    KERNEL8x4_K1                                            \n"
		"   cmp     $0, %%rdx                                        \n"
		"    je         EDGE_K_M8_N4                                 \n"
		"    KERNEL8x4_K2                                            \n"

		"    subq     $8, %%rdx                                      \n"
		"   jmp     MAIN_K_M8_N4                                     \n"

		"EDGE_K_M8_N4:                                               \n"

		"    KERNEL8x4_END_K                                         \n"

		"    cmp     $0, %%r15                                       \n"
		"    je      SAVE_C_8x4                                      \n"
		"    ADD_C_8x4                                               \n"

		"SAVE_C_8x4:                                                 \n"
		"    SAVE_8x4                                                \n"

		//----------------------------------------------------------------

		"BEGIN_M4_N4:                                                \n"

		"    cmpq      $4, %%rdi                                     \n" // M % 4
		"    jb       BEGIN_M1_N4                                    \n"

		"    mov     %%r14, %%rbx                                    \n" // Bc
		"   prefetcht0         (%%rbx)                               \n"

		"    mov      %%rcx, %%r10                                   \n" // C0
		"   prefetcht1         (%%r10)                               \n"
		"    leaq     (%%r10, %%r8, 4), %%r11                        \n" // C1
		"   prefetcht1         (%%r11)                               \n"
		"    leaq     (%%r11, %%r8, 4), %%r12                        \n" // C2
		"   prefetcht1         (%%r12)                               \n"
		"    leaq     (%%r12, %%r8, 4), %%r13                        \n" // C3
		"   prefetcht1         (%%r13)                               \n"

		"    mov     %%rsi, %%rdx                                    \n" // K

		"   vmovups        (%%rbx), %%xmm4                           \n"
		"   vbroadcastss    (%%rax), %%xmm0                          \n"
		"    vpxorq         %%xmm28, %%xmm28, %%xmm28                \n"
		"    vpxorq         %%xmm29, %%xmm29, %%xmm29                \n"
		"   vbroadcastss    4(%%rax), %%xmm1                         \n"
		"    vpxorq         %%xmm30, %%xmm30, %%xmm30                \n"
		"    vpxorq         %%xmm31, %%xmm31, %%xmm31                \n"

		"    subq     $8, %%rdx                                      \n"

		"MAIN_K_M4_N4:                                               \n"

		"    KERNEL4x4_K1                                            \n"
		"    KERNEL4x4_K2                                            \n"
		"    KERNEL4x4_K1                                            \n"
		"    KERNEL4x4_K2                                            \n"
		"    KERNEL4x4_K1                                            \n"
		"    KERNEL4x4_K2                                            \n"
		"    KERNEL4x4_K1                                            \n"
		"   cmp     $0, %%rdx                                        \n"
		"    je         EDGE_K_M4_N4                                 \n"
		"    KERNEL4x4_K2                                            \n"

		"    subq     $8, %%rdx                                      \n"
		"   jmp     MAIN_K_M4_N4                                     \n"

		"EDGE_K_M4_N4:                                               \n"

		"    KERNEL4x4_END_K                                         \n"

		"    cmp     $0, %%r15                                       \n"
		"    je      SAVE_C_4x4                                      \n"
		"    ADD_C_4x4                                               \n"

		"SAVE_C_4x4:                                                 \n"
		"    SAVE_4x4                                                \n"

		//----------------------------------------------------------------

		"BEGIN_M1_N4:                                                \n"
		"    cmpq      $1, %%rdi                                     \n" // M % 1
		"    jb       END_M_N4                                       \n"

		"    mov     %%r14, %%rbx                                    \n" // Bc
		"   prefetcht0         (%%rbx)                               \n"

		"    mov      %%rcx, %%r10                                   \n" // C0
		"   prefetcht1         (%%r10)                               \n"

		"    mov     %%rsi, %%rdx                                    \n" // K

		"   vmovups        (%%rbx), %%xmm4                           \n"
		"   vbroadcastss    (%%rax), %%xmm0                          \n" // A0
		"    vpxorq         %%xmm30, %%xmm30, %%xmm30                \n"

		"    subq     $8, %%rdx                                      \n"

		"MAIN_K_M1_N4:                                               \n"
		"    KERNEL1x4_K1                                            \n"
		"    KERNEL1x4_K2                                            \n"
		"    KERNEL1x4_K1                                            \n"
		"    KERNEL1x4_K2                                            \n"
		"    KERNEL1x4_K1                                            \n"
		"    KERNEL1x4_K2                                            \n"
		"    KERNEL1x4_K1                                            \n"
		"   cmp     $0, %%rdx                                        \n"
		"    je         EDGE_K_M1_N4                                 \n"
		"    KERNEL1x4_K2                                            \n"

		"    subq     $8, %%rdx                                      \n"
		"   jmp     MAIN_K_M1_N4                                     \n"

		"EDGE_K_M1_N4:                                               \n"
		"    KERNEL1x4_END_K                                         \n"

		"    cmp     $0, %%r15                                       \n"
		"    je      SAVE_C_1x4                                      \n"
		"    ADD_C_1x4                                               \n"

		"SAVE_C_1x4:                                                 \n"
		"    SAVE_1x4                                                \n"
		"   cmpq      $1, %%rdi                                      \n"
		"    jnb     BEGIN_M1_N4                                     \n" // 不小于（或等于）则跳转

		//-----------------------------------------------------------------

		"END_M_N4:                                                   \n"

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
		[Bc] "m"(Bc),
		[k_tag] "m"(k_tag)
		: "rax", "rbx", "rcx", "rdx", "rdi", "rsi", "rbp", "r8", "r9", "r10", "r11", "r12",
		  "r13", "r14", "r15", "xmm0", "xmm1", "xmm2", "xmm3", "xmm4", "xmm5",
		  "xmm6", "xmm7", "xmm8", "xmm9", "xmm10", "xmm11", "xmm12", "xmm13",
		  "xmm14", "xmm15", "xmm16", "xmm17", "xmm18", "xmm19", "xmm20", "xmm21",
		  "xmm22", "xmm23", "xmm24", "xmm25", "xmm26", "xmm27", "xmm28", "xmm29",
		  "xmm30", "xmm31", "memory");
}

// void SMM_NN_KERNELm12xn2(float *C, float *A, float *B, long M, long N, long K, long LN, long LK, float *Bc, long k_tag)
// {
// 	asm volatile(
// 		:
// 		:
// 		[C] "m"(C),
// 		[A] "m"(A),
// 		[B] "m"(B),
// 		[M] "m"(M),
// 		[N] "m"(N),
// 		[K] "m"(K),
// 		[LN] "m"(LN),
// 		[LK] "m"(LK),
// 		[Bc] "m"(Bc),
// 		[k_tag] "m"(k_tag)
// 		: "rax", "rbx", "rcx", "rdx", "rdi", "rsi", "rbp", "r8", "r9", "r10", "r11", "r12",
// 		  "r13", "r14", "r15", "xmm0", "xmm1", "xmm2", "xmm3", "xmm4", "xmm5",
// 		  "xmm6", "xmm7", "xmm8", "xmm9", "xmm10", "xmm11", "xmm12", "xmm13",
// 		  "xmm14", "xmm15", "xmm16", "xmm17", "xmm18", "xmm19", "xmm20", "xmm21",
// 		  "xmm22", "xmm23", "xmm24", "xmm25", "xmm26", "xmm27", "xmm28", "xmm29",
// 		  "xmm30", "xmm31", "memory");
// }

void SMM_NN_KERNELm12xn1(float *C, float *A, float *B, long M, long N, long K, long LN, long LK, float *Bc, long k_tag)
{
	asm volatile(
		".macro    KERNEL12x1_PACK_K1                                \n"

		"   movss           8(%%rax), %%xmm2                         \n"
		"   vfmadd231ps              %%xmm0, %%xmm4, %%xmm20         \n"

		"   movss           12(%%rax), %%xmm3                        \n"
		"   vfmadd231ps              %%xmm1, %%xmm4, %%xmm21         \n"

		"   prefetcht0         256(%%rax)                            \n"

		"   movss           16(%%rax), %%xmm0                        \n"
		"   vfmadd231ps              %%xmm2, %%xmm4, %%xmm22         \n"

		"   movss           20(%%rax), %%xmm1                        \n"
		"   vfmadd231ps              %%xmm3, %%xmm4, %%xmm23         \n"

		"   movss           24(%%rax), %%xmm2                        \n"
		"   vfmadd231ps              %%xmm0, %%xmm4, %%xmm24         \n"

		"   movss           28(%%rax), %%xmm3                        \n"
		"   vfmadd231ps              %%xmm1, %%xmm4, %%xmm25         \n"

		"   movss           32(%%rax), %%xmm0                        \n"
		"   vfmadd231ps              %%xmm2, %%xmm4, %%xmm26         \n"

		"    leaq      (%%rbx, %%r8, 4), %%rbx                       \n" // B

		"   movss           36(%%rax), %%xmm1                        \n"
		"   vfmadd231ps              %%xmm3, %%xmm4, %%xmm27         \n"

		"   movss           40(%%rax), %%xmm2                        \n"
		"   vfmadd231ps              %%xmm0, %%xmm4, %%xmm28         \n"

		"   movss           44(%%rax), %%xmm3                        \n"
		"   vfmadd231ps              %%xmm1, %%xmm4, %%xmm29         \n"
		"   movss           (%%rbx), %%xmm6                          \n"
		"    addq              $48, %%rax                            \n"

		"   movss           (%%rax), %%xmm0                          \n"
		"   vfmadd231ps              %%xmm2, %%xmm4, %%xmm30         \n"

		"   movss           4(%%rax), %%xmm1                         \n"
		"   vfmadd231ps              %%xmm3, %%xmm4, %%xmm31         \n"
		"   vmovss           %%xmm4, (%%rbp)                         \n"
		"    addq              $4, %%rbp                             \n" // 64->16

		".endm                                                       \n"

		".macro    KERNEL12x1_PACK_K2                                \n"

		"   movss           8(%%rax), %%xmm2                         \n"
		"   vfmadd231ps              %%xmm0, %%xmm6, %%xmm20         \n"

		"   movss           12(%%rax), %%xmm3                        \n"
		"   vfmadd231ps              %%xmm1, %%xmm6, %%xmm21         \n"

		"   movss           16(%%rax), %%xmm0                        \n"
		"   vfmadd231ps              %%xmm2, %%xmm6, %%xmm22         \n"

		"   prefetcht0         256(%%rax)                            \n"

		"   movss           20(%%rax), %%xmm1                        \n"
		"   vfmadd231ps              %%xmm3, %%xmm6, %%xmm23         \n"

		"   movss           24(%%rax), %%xmm2                        \n"
		"   vfmadd231ps              %%xmm0, %%xmm6, %%xmm24         \n"

		"   movss           28(%%rax), %%xmm3                        \n"
		"   vfmadd231ps              %%xmm1, %%xmm6, %%xmm25         \n"

		"   movss           32(%%rax), %%xmm0                        \n"
		"   vfmadd231ps              %%xmm2, %%xmm6, %%xmm26         \n"

		"    leaq      (%%rbx, %%r8, 4), %%rbx                       \n" // B

		"   movss           36(%%rax), %%xmm1                        \n"
		"   vfmadd231ps              %%xmm3, %%xmm6, %%xmm27         \n"

		"   prefetcht0         4(%%rbx)                              \n" //

		"   movss           40(%%rax), %%xmm2                        \n"
		"   vfmadd231ps              %%xmm0, %%xmm6, %%xmm28         \n"

		"   movss           44(%%rax), %%xmm3                        \n"
		"   vfmadd231ps              %%xmm1, %%xmm6, %%xmm29         \n"
		"   movss           (%%rbx), %%xmm4                          \n"
		"    addq              $48, %%rax                            \n"

		"   movss           (%%rax), %%xmm0                          \n"
		"   vfmadd231ps              %%xmm2, %%xmm6, %%xmm30         \n"

		"   movss           4(%%rax), %%xmm1                         \n"
		"   vfmadd231ps              %%xmm3, %%xmm6, %%xmm31         \n"
		"   vmovss           %%xmm6, (%%rbp)                         \n"
		"    addq              $4, %%rbp                             \n" // 64->16

		".endm                                                       \n"

		".macro    KERNEL12x1_PACK_END_K                             \n"

		"   movss           8(%%rax), %%xmm2                         \n"
		"   vfmadd231ps              %%xmm0, %%xmm6, %%xmm20         \n"

		"   movss           12(%%rax), %%xmm3                        \n"
		"   vfmadd231ps              %%xmm1, %%xmm6, %%xmm21         \n"

		"   movss           16(%%rax), %%xmm0                        \n"
		"   vfmadd231ps              %%xmm2, %%xmm6, %%xmm22         \n"

		"   prefetcht0         256(%%rax)                            \n"

		"   movss           20(%%rax), %%xmm1                        \n"
		"   vfmadd231ps              %%xmm3, %%xmm6, %%xmm23         \n"

		"   movss           24(%%rax), %%xmm2                        \n"
		"   vfmadd231ps              %%xmm0, %%xmm6, %%xmm24         \n"

		"   movss           28(%%rax), %%xmm3                        \n"
		"   vfmadd231ps              %%xmm1, %%xmm6, %%xmm25         \n"

		"   movss           32(%%rax), %%xmm0                        \n"
		"   vfmadd231ps              %%xmm2, %%xmm6, %%xmm26         \n"

		"   movss           36(%%rax), %%xmm1                        \n"
		"   vfmadd231ps              %%xmm3, %%xmm6, %%xmm27         \n"

		"   movss           40(%%rax), %%xmm2                        \n"
		"   vfmadd231ps              %%xmm0, %%xmm6, %%xmm28         \n"

		"   movss           44(%%rax), %%xmm3                        \n"
		"   vfmadd231ps              %%xmm1, %%xmm6, %%xmm29         \n"
		"    addq              $48, %%rax                            \n"

		"   vfmadd231ps              %%xmm2, %%xmm6, %%xmm30         \n"
		"   vmovss           %%xmm6, (%%rbp)                         \n"
		"   vfmadd231ps              %%xmm3, %%xmm6, %%xmm31         \n"

		".endm                                                       \n"

		//-----------------------------------------------------------------------

		".macro    KERNEL12x1_K1                                     \n"

		"   movss           8(%%rax), %%xmm2                         \n"
		"   vfmadd231ps              %%xmm0, %%xmm4, %%xmm20         \n"

		"   movss           12(%%rax), %%xmm3                        \n"
		"   vfmadd231ps              %%xmm1, %%xmm4, %%xmm21         \n"

		"   prefetcht0         256(%%rax)                            \n"

		"   movss           16(%%rax), %%xmm0                        \n"
		"   vfmadd231ps              %%xmm2, %%xmm4, %%xmm22         \n"

		"   movss           20(%%rax), %%xmm1                        \n"
		"   vfmadd231ps              %%xmm3, %%xmm4, %%xmm23         \n"

		"   movss           24(%%rax), %%xmm2                        \n"
		"   vfmadd231ps              %%xmm0, %%xmm4, %%xmm24         \n"

		"   movss           28(%%rax), %%xmm3                        \n"
		"   vfmadd231ps              %%xmm1, %%xmm4, %%xmm25         \n"

		"   movss           32(%%rax), %%xmm0                        \n"
		"   vfmadd231ps              %%xmm2, %%xmm4, %%xmm26         \n"

		"    addq              $4, %%rbx                             \n" // B

		"   movss           36(%%rax), %%xmm1                        \n"
		"   vfmadd231ps              %%xmm3, %%xmm4, %%xmm27         \n"

		"   movss           40(%%rax), %%xmm2                        \n"
		"   vfmadd231ps              %%xmm0, %%xmm4, %%xmm28         \n"

		"   movss           44(%%rax), %%xmm3                        \n"
		"   vfmadd231ps              %%xmm1, %%xmm4, %%xmm29         \n"
		"   movss           (%%rbx), %%xmm6                          \n"
		"    addq              $48, %%rax                            \n"

		"   movss           (%%rax), %%xmm0                          \n"
		"   vfmadd231ps              %%xmm2, %%xmm4, %%xmm30         \n"

		"   movss           4(%%rax), %%xmm1                         \n"
		"   vfmadd231ps              %%xmm3, %%xmm4, %%xmm31         \n"

		".endm                                                       \n"

		".macro    KERNEL12x1_K2                                     \n"

		"   movss           8(%%rax), %%xmm2                         \n"
		"   vfmadd231ps              %%xmm0, %%xmm6, %%xmm20         \n"

		"   movss           12(%%rax), %%xmm3                        \n"
		"   vfmadd231ps              %%xmm1, %%xmm6, %%xmm21         \n"

		"   movss           16(%%rax), %%xmm0                        \n"
		"   vfmadd231ps              %%xmm2, %%xmm6, %%xmm22         \n"

		"   prefetcht0         256(%%rax)                            \n"

		"   movss           20(%%rax), %%xmm1                        \n"
		"   vfmadd231ps              %%xmm3, %%xmm6, %%xmm23         \n"

		"   movss           24(%%rax), %%xmm2                        \n"
		"   vfmadd231ps              %%xmm0, %%xmm6, %%xmm24         \n"

		"   movss           28(%%rax), %%xmm3                        \n"
		"   vfmadd231ps              %%xmm1, %%xmm6, %%xmm25         \n"

		"   movss           32(%%rax), %%xmm0                        \n"
		"   vfmadd231ps              %%xmm2, %%xmm6, %%xmm26         \n"

		"    addq             $4, %%rbx                              \n" //

		"   movss           36(%%rax), %%xmm1                        \n"
		"   vfmadd231ps              %%xmm3, %%xmm6, %%xmm27         \n"

		"   prefetcht0         4(%%rbx)                              \n" //

		"   movss           40(%%rax), %%xmm2                        \n"
		"   vfmadd231ps              %%xmm0, %%xmm6, %%xmm28         \n"

		"   movss           44(%%rax), %%xmm3                        \n"
		"   vfmadd231ps              %%xmm1, %%xmm6, %%xmm29         \n"
		"   movss           (%%rbx), %%xmm4                          \n"
		"    addq              $48, %%rax                            \n"

		"   movss           (%%rax), %%xmm0                          \n"
		"   vfmadd231ps              %%xmm2, %%xmm6, %%xmm30         \n"

		"   movss           4(%%rax), %%xmm1                         \n"
		"   vfmadd231ps              %%xmm3, %%xmm6, %%xmm31         \n"

		".endm                                                       \n"

		".macro    KERNEL12x1_END_K                                  \n"

		"   movss           8(%%rax), %%xmm2                         \n"
		"   vfmadd231ps              %%xmm0, %%xmm6, %%xmm20         \n"

		"   movss           12(%%rax), %%xmm3                        \n"
		"   vfmadd231ps              %%xmm1, %%xmm6, %%xmm21         \n"

		"   movss           16(%%rax), %%xmm0                        \n"
		"   vfmadd231ps              %%xmm2, %%xmm6, %%xmm22         \n"

		"   prefetcht0         256(%%rax)                            \n"

		"   movss           20(%%rax), %%xmm1                        \n"
		"   vfmadd231ps              %%xmm3, %%xmm6, %%xmm23         \n"

		"   movss           24(%%rax), %%xmm2                        \n"
		"   vfmadd231ps              %%xmm0, %%xmm6, %%xmm24         \n"

		"   movss           28(%%rax), %%xmm3                        \n"
		"   vfmadd231ps              %%xmm1, %%xmm6, %%xmm25         \n"

		"   movss           32(%%rax), %%xmm0                        \n"
		"   vfmadd231ps              %%xmm2, %%xmm6, %%xmm26         \n"

		"   movss           36(%%rax), %%xmm1                        \n"
		"   vfmadd231ps              %%xmm3, %%xmm6, %%xmm27         \n"

		"   movss           40(%%rax), %%xmm2                        \n"
		"   vfmadd231ps              %%xmm0, %%xmm6, %%xmm28         \n"

		"   movss           44(%%rax), %%xmm3                        \n"
		"   vfmadd231ps              %%xmm1, %%xmm6, %%xmm29         \n"
		"   movss           (%%rbx), %%xmm4                          \n"
		"    addq              $48, %%rax                            \n"

		"   movss           (%%rax), %%xmm0                          \n"
		"   vfmadd231ps              %%xmm2, %%xmm6, %%xmm30         \n"

		"   movss           4(%%rax), %%xmm1                         \n"
		"   vfmadd231ps              %%xmm3, %%xmm6, %%xmm31         \n"

		".endm                                                       \n"

		".macro    ADD_C_12x1                                        \n"

		"   movss           (%%r10), %%xmm0                          \n"
		"    vaddps             %%xmm0, %%xmm20, %%xmm20             \n"
		"   movss           (%%r11), %%xmm1                          \n"
		"    vaddps             %%xmm1, %%xmm21, %%xmm21             \n"
		"   movss           (%%r12), %%xmm2                          \n"
		"    vaddps             %%xmm2, %%xmm22, %%xmm22             \n"
		"   movss           (%%r13), %%xmm3                          \n"
		"    vaddps             %%xmm3, %%xmm23, %%xmm23             \n"

		"    leaq              (%%r13, %%r8, 4), %%r10               \n" // C0
		"    leaq             (%%r10, %%r8, 4), %%r11                \n" // C1
		"    leaq             (%%r11, %%r8, 4), %%r12                \n" // C2
		"    leaq             (%%r12, %%r8, 4), %%r13                \n" // C3

		"   movss           (%%r10), %%xmm4                          \n"
		"    vaddps             %%xmm4, %%xmm24, %%xmm24             \n"
		"   movss           (%%r11), %%xmm5                          \n"
		"    vaddps             %%xmm5, %%xmm25, %%xmm25             \n"
		"   movss           (%%r12), %%xmm6                          \n"
		"    vaddps             %%xmm6, %%xmm26, %%xmm26             \n"
		"   movss           (%%r13), %%xmm7                          \n"
		"    vaddps             %%xmm7, %%xmm27, %%xmm27             \n"

		"    leaq              (%%r13, %%r8, 4), %%r10               \n" // C0
		"    leaq             (%%r10, %%r8, 4), %%r11                \n" // C1
		"    leaq             (%%r11, %%r8, 4), %%r12                \n" // C2
		"    leaq             (%%r12, %%r8, 4), %%r13                \n" // C3

		"   movss           (%%r10), %%xmm0                          \n"
		"    vaddps             %%xmm0, %%xmm28, %%xmm28             \n"
		"   movss           (%%r11), %%xmm1                          \n"
		"    vaddps             %%xmm1, %%xmm29, %%xmm29             \n"
		"   movss           (%%r12), %%xmm2                          \n"
		"    vaddps             %%xmm2, %%xmm30, %%xmm30             \n"
		"   movss           (%%r13), %%xmm3                          \n"
		"    vaddps             %%xmm3, %%xmm31, %%xmm31             \n"

		"    mov      %%rcx, %%r10                                   \n" // C0
		"    leaq     (%%r10, %%r8, 4), %%r11                        \n" // C1
		"    leaq     (%%r11, %%r8, 4), %%r12                        \n" // C2
		"    leaq     (%%r12, %%r8, 4), %%r13                        \n" // C3

		".endm                                                       \n"

		".macro    SAVE_12x1                                         \n"

		"   vmovss           %%xmm20, (%%r10)                        \n"
		"   vmovss           %%xmm21, (%%r11)                        \n"
		"    subq             $12, %%rdi                             \n"
		"   vmovss           %%xmm22, (%%r12)                        \n"
		"   vmovss           %%xmm23, (%%r13)                        \n"

		"    leaq      (%%r13, %%r8, 4), %%r10                       \n" // C0
		"    leaq     (%%r10, %%r8, 4), %%r11                        \n" // C1
		"    leaq     (%%r11, %%r8, 4), %%r12                        \n" // C2
		"    leaq     (%%r12, %%r8, 4), %%r13                        \n" // C3

		"   vmovss           %%xmm24, (%%r10)                        \n"
		"   vmovss           %%xmm25, (%%r11)                        \n"
		"   vmovss           %%xmm26, (%%r12)                        \n"
		"   vmovss           %%xmm27, (%%r13)                        \n"

		"    leaq      (%%r13, %%r8, 4), %%r10                       \n" // C0
		"    leaq     (%%r10, %%r8, 4), %%r11                        \n" // C1
		"    leaq     (%%r11, %%r8, 4), %%r12                        \n" // C2
		"    leaq     (%%r12, %%r8, 4), %%r13                        \n" // C3

		"   vmovss           %%xmm28, (%%r10)                        \n"
		"   vmovss           %%xmm29, (%%r11)                        \n"
		"   vmovss           %%xmm30, (%%r12)                        \n"
		"   vmovss           %%xmm31, (%%r13)                        \n"

		"    leaq      (%%r13, %%r8, 4), %%rcx                       \n" // C0

		".endm                                                       \n"

		//-----------------------------------------------------------------------

		".macro    KERNEL8x1_K1                                      \n"

		"   movss           8(%%rax), %%xmm2                         \n"
		"   vfmadd231ps              %%xmm0, %%xmm4, %%xmm24         \n"

		"   movss           12(%%rax), %%xmm3                        \n"
		"   vfmadd231ps              %%xmm1, %%xmm4, %%xmm25         \n"

		"   prefetcht0         256(%%rax)                            \n"

		"   movss           16(%%rax), %%xmm8                        \n"
		"   vfmadd231ps              %%xmm2, %%xmm4, %%xmm26         \n"

		"   movss           20(%%rax), %%xmm9                        \n"
		"   vfmadd231ps              %%xmm3, %%xmm4, %%xmm27         \n"

		"   movss           24(%%rax), %%xmm10                       \n"
		"   vfmadd231ps              %%xmm8, %%xmm4, %%xmm28         \n"

		"   prefetcht0         4(%%rbx)                              \n" //

		"   movss           28(%%rax), %%xmm11                       \n"
		"   vfmadd231ps              %%xmm9, %%xmm4, %%xmm29         \n"
		"    addq              $4, %%rbx                             \n" //
		"   vfmadd231ps              %%xmm10, %%xmm4, %%xmm30        \n"

		"    addq              $32, %%rax                            \n"
		"   movss           (%%rbx), %%xmm6                          \n"
		"   vfmadd231ps              %%xmm11, %%xmm4, %%xmm31        \n"

		"   movss           (%%rax), %%xmm0                          \n"
		"   movss           4(%%rax), %%xmm1                         \n"

		".endm                                                       \n"

		".macro    KERNEL8x1_K2                                      \n"

		"   movss           8(%%rax), %%xmm2                         \n"
		"   vfmadd231ps              %%xmm0, %%xmm6, %%xmm24         \n"

		"   movss           12(%%rax), %%xmm3                        \n"
		"   vfmadd231ps              %%xmm1, %%xmm6, %%xmm25         \n"

		"   prefetcht0         256(%%rax)                            \n"

		"   movss           16(%%rax), %%xmm8                        \n"
		"   vfmadd231ps              %%xmm2, %%xmm6, %%xmm26         \n"

		"   movss           20(%%rax), %%xmm9                        \n"
		"   vfmadd231ps              %%xmm3, %%xmm6, %%xmm27         \n"

		"   movss           24(%%rax), %%xmm10                       \n"
		"   vfmadd231ps              %%xmm8, %%xmm6, %%xmm28         \n"

		"   prefetcht0         4(%%rbx)                              \n" //

		"   movss           28(%%rax), %%xmm11                       \n"
		"   vfmadd231ps              %%xmm9, %%xmm6, %%xmm29         \n"
		"    addq              $4, %%rbx                             \n" // B
		"   vfmadd231ps              %%xmm10, %%xmm6, %%xmm30        \n"

		"    addq              $32, %%rax                            \n"
		"   movss           (%%rbx), %%xmm4                          \n"
		"   vfmadd231ps              %%xmm11, %%xmm6, %%xmm31        \n"

		"   movss           (%%rax), %%xmm0                          \n"
		"   movss           4(%%rax), %%xmm1                         \n"

		".endm                                                       \n"

		".macro    KERNEL8x1_END_K                                   \n"

		"   movss           8(%%rax), %%xmm2                         \n"
		"   vfmadd231ps              %%xmm0, %%xmm6, %%xmm24         \n"

		"   movss           12(%%rax), %%xmm3                        \n"
		"   vfmadd231ps              %%xmm1, %%xmm6, %%xmm25         \n"

		"   prefetcht0         256(%%rax)                            \n"

		"   movss           16(%%rax), %%xmm8                        \n"
		"   vfmadd231ps              %%xmm2, %%xmm6, %%xmm26         \n"

		"   movss           20(%%rax), %%xmm9                        \n"
		"   vfmadd231ps              %%xmm3, %%xmm6, %%xmm27         \n"

		"   movss           24(%%rax), %%xmm10                       \n"
		"   vfmadd231ps              %%xmm8, %%xmm6, %%xmm28         \n"

		"   movss           28(%%rax), %%xmm11                       \n"
		"   vfmadd231ps              %%xmm9, %%xmm6, %%xmm29         \n"
		"   vfmadd231ps              %%xmm10, %%xmm6, %%xmm30        \n"

		"    addq              $32, %%rax                            \n"
		"   vfmadd231ps              %%xmm11, %%xmm6, %%xmm31        \n"

		".endm                                                       \n"

		".macro    ADD_C_8x1                                         \n"

		"   movss           (%%r10), %%xmm4                          \n"
		"    vaddps             %%xmm4, %%xmm24, %%xmm24             \n"
		"   movss           (%%r11), %%xmm5                          \n"
		"    vaddps             %%xmm5, %%xmm25, %%xmm25             \n"
		"   movss           (%%r12), %%xmm6                          \n"
		"    vaddps             %%xmm6, %%xmm26, %%xmm26             \n"
		"   movss           (%%r13), %%xmm7                          \n"
		"    vaddps             %%xmm7, %%xmm27, %%xmm27             \n"

		"    leaq              (%%r13, %%r8, 4), %%r10               \n" // C0
		"    leaq             (%%r10, %%r8, 4), %%r11                \n" // C1
		"    leaq             (%%r11, %%r8, 4), %%r12                \n" // C2
		"    leaq             (%%r12, %%r8, 4), %%r13                \n" // C3

		"   movss           (%%r10), %%xmm0                          \n"
		"    vaddps             %%xmm0, %%xmm28, %%xmm28             \n"
		"   movss           (%%r11), %%xmm1                          \n"
		"    vaddps             %%xmm1, %%xmm29, %%xmm29             \n"
		"   movss           (%%r12), %%xmm2                          \n"
		"    vaddps             %%xmm2, %%xmm30, %%xmm30             \n"
		"   movss           (%%r13), %%xmm3                          \n"
		"    vaddps             %%xmm3, %%xmm31, %%xmm31             \n"

		"    mov      %%rcx, %%r10                                   \n" // C0
		"    leaq     (%%r10, %%r8, 4), %%r11                        \n" // C1
		"    leaq     (%%r11, %%r8, 4), %%r12                        \n" // C2
		"    leaq     (%%r12, %%r8, 4), %%r13                        \n" // C3

		".endm                                                       \n"

		".macro    SAVE_8x1                                          \n"

		"   vmovss           %%xmm24, (%%r10)                        \n"
		"   vmovss           %%xmm25, (%%r11)                        \n"
		"    subq             $8, %%rdi                              \n"
		"   vmovss           %%xmm26, (%%r12)                        \n"
		"   vmovss           %%xmm27, (%%r13)                        \n"

		"    leaq      (%%r13, %%r8, 4), %%r10                       \n" // C0
		"    leaq     (%%r10, %%r8, 4), %%r11                        \n" // C1
		"    leaq     (%%r11, %%r8, 4), %%r12                        \n" // C2
		"    leaq     (%%r12, %%r8, 4), %%r13                        \n" // C3

		"   vmovss           %%xmm28, (%%r10)                        \n"
		"   vmovss           %%xmm29, (%%r11)                        \n"
		"   vmovss           %%xmm30, (%%r12)                        \n"
		"   vmovss           %%xmm31, (%%r13)                        \n"

		"    leaq      (%%r13, %%r8, 4), %%rcx                       \n" // C0

		".endm                                                       \n"

		//-----------------------------------------------------------------------

		".macro    KERNEL4x1_K1                                      \n"

		"   movss           8(%%rax), %%xmm2                         \n"
		"   vfmadd231ps              %%xmm0, %%xmm4, %%xmm28         \n"

		"   movss           12(%%rax), %%xmm3                        \n"
		"   vfmadd231ps              %%xmm1, %%xmm4, %%xmm29         \n"
		"    addq              $16, %%rax                            \n"

		"   prefetcht0         256(%%rax)                            \n"
		"    addq              $4, %%rbx                             \n" // B
		"   movss           (%%rax), %%xmm0                          \n"
		"   vfmadd231ps              %%xmm2, %%xmm4, %%xmm30         \n"

		"   movss           4(%%rax), %%xmm1                         \n"
		"   vfmadd231ps              %%xmm3, %%xmm4, %%xmm31         \n"

		"   prefetcht0         4(%%rbx)                              \n"
		"   movss           (%%rbx), %%xmm6                          \n"

		".endm                                                       \n"

		".macro    KERNEL4x1_K2                                      \n"

		"   movss           8(%%rax), %%xmm2                         \n"
		"   vfmadd231ps              %%xmm0, %%xmm6, %%xmm28         \n"

		"   movss           12(%%rax), %%xmm3                        \n"
		"   vfmadd231ps              %%xmm1, %%xmm6, %%xmm29         \n"
		"    addq              $16, %%rax                            \n"

		"   prefetcht0         256(%%rax)                            \n"
		"    addq              $4, %%rbx                             \n" // B
		"   movss           (%%rax), %%xmm0                          \n"
		"   vfmadd231ps              %%xmm2, %%xmm6, %%xmm30         \n"

		"   movss           4(%%rax), %%xmm1                         \n"
		"   vfmadd231ps              %%xmm3, %%xmm6, %%xmm31         \n"

		"   movss           (%%rbx), %%xmm4                          \n"
		"   prefetcht0         4(%%rbx)                              \n"

		".endm                                                       \n"

		".macro    KERNEL4x1_END_K                                   \n"

		"   movss           8(%%rax), %%xmm2                         \n"
		"   vfmadd231ps              %%xmm0, %%xmm6, %%xmm28         \n"

		"   movss           12(%%rax), %%xmm3                        \n"
		"   vfmadd231ps              %%xmm1, %%xmm6, %%xmm29         \n"
		"    addq              $16, %%rax                            \n"

		"   vfmadd231ps              %%xmm2, %%xmm6, %%xmm30         \n"
		"   vfmadd231ps              %%xmm3, %%xmm6, %%xmm31         \n"

		".endm                                                       \n"

		".macro    ADD_C_4x1                                         \n"

		"   movss           (%%r10), %%xmm0                          \n"
		"    vaddps             %%xmm0, %%xmm28, %%xmm28             \n"
		"   movss           (%%r11), %%xmm1                          \n"
		"    vaddps             %%xmm1, %%xmm29, %%xmm29             \n"
		"   movss           (%%r12), %%xmm2                          \n"
		"    vaddps             %%xmm2, %%xmm30, %%xmm30             \n"
		"   movss           (%%r13), %%xmm3                          \n"
		"    vaddps             %%xmm3, %%xmm31, %%xmm31             \n"

		"    mov      %%rcx, %%r10                                   \n" // C0
		"    leaq     (%%r10, %%r8, 4), %%r11                        \n" // C1
		"    leaq     (%%r11, %%r8, 4), %%r12                        \n" // C2
		"    leaq     (%%r12, %%r8, 4), %%r13                        \n" // C3

		".endm                                                       \n"

		".macro    SAVE_4x1                                          \n"
		"   subq             $4, %%rdi                               \n"

		"   vmovss           %%xmm28, (%%r10)                        \n"
		"   vmovss           %%xmm29, (%%r11)                        \n"
		"   vmovss           %%xmm30, (%%r12)                        \n"
		"   vmovss           %%xmm31, (%%r13)                        \n"

		"    leaq      (%%r13, %%r8, 4), %%rcx                       \n" // C0

		".endm                                                       \n"

		//-----------------------------------------------------------------

		".macro    KERNEL1x1_K1                                      \n"
		"   prefetcht0         256(%%rax)                            \n"
		"   addq              $4, %%rax                              \n"

		"   movss           (%%rax), %%xmm2                          \n"
		"   vfmadd231ps              %%xmm0, %%xmm4, %%xmm30         \n"

		"    addq              $4, %%rbx                             \n" // B
		"   movss              (%%rbx), %%xmm6                       \n"
		"   prefetcht0         4(%%rbx)                              \n"

		".endm                                                       \n"

		".macro    KERNEL1x1_K2                                      \n"
		"   prefetcht0         256(%%rax)                            \n"
		"    addq              $4, %%rax                             \n"

		"   movss           (%%rax), %%xmm0                          \n"
		"   vfmadd231ps              %%xmm2, %%xmm6, %%xmm30         \n"

		"    addq              $4, %%rbx                             \n" // B
		"   movss              (%%rbx), %%xmm4                       \n"
		"   prefetcht0         4(%%rbx)                              \n"

		".endm                                                       \n"

		".macro    KERNEL1x1_END_K                                   \n"
		"   vfmadd231ps              %%xmm2, %%xmm6, %%xmm30         \n"
		"    addq              $4, %%rax                             \n"
		".endm                                                       \n"

		".macro    ADD_C_1x1                                         \n"
		"   movss           (%%r10), %%xmm2                          \n"
		"    vaddps             %%xmm2, %%xmm30, %%xmm30             \n"
		"    mov      %%rcx, %%r10                                   \n" // C0
		".endm                                                       \n"

		".macro    SAVE_1x1                                          \n"
		"   vmovss           %%xmm30, (%%r10)                        \n"
		"   subq             $1, %%rdi                               \n"
		"    leaq      (%%r10, %%r8, 4), %%rcx                       \n" // C0

		".endm                                                       \n"

		//-----------------------------------------------------------------

		"SMM_NN_KERNEL12x1:                                          \n"

		"   mov     %[C], %%rcx                                      \n"
		"   mov     %[A], %%rax                                      \n"
		"   mov     %[B], %%rbx                                      \n"

		"   prefetcht0         (%%rax)                               \n"

		"    mov     %[K], %%rdx                                     \n" // K
		"    mov      %[LN], %%r8                                    \n"
		"    mov      %[Bc], %%r14                                   \n"
		"    mov      %[M], %%rdi                                    \n"
		"    mov     %[k_tag], %%r15                                 \n"

		"   prefetcht0         (%%rbx)                               \n"
		"    mov     %%rbx, %%r9                                     \n" // B
		"    mov     %%rdx, %%rsi                                    \n" // K

		"BEGIN_PACK12x1:                                             \n"

		"    mov     %%r9, %%rbx                                     \n" // B
		"   prefetcht0         (%%rbx)                               \n"

		"    mov      %%rcx, %%r10                                   \n" // C0
		"   prefetcht1         (%%r10)                               \n"
		"    leaq     (%%r10, %%r8, 4), %%r11                        \n" // C1
		"   prefetcht1         (%%r11)                               \n"
		"    leaq     (%%r11, %%r8, 4), %%r12                        \n" // C2
		"   prefetcht1         (%%r12)                               \n"
		"    leaq     (%%r12, %%r8, 4), %%r13                        \n" // C3
		"   prefetcht1         (%%r13)                               \n"

		"    mov     %%rsi, %%rdx                                    \n" // K
		"    mov     %%r14, %%rbp                                    \n" // Bc

		"   movss          (%%rbx), %%xmm4                           \n"
		"    vpxorq         %%xmm20, %%xmm20, %%xmm20                \n"
		"    vpxorq         %%xmm21, %%xmm21, %%xmm21                \n"
		"   movss           (%%rax), %%xmm0                          \n"
		"    vpxorq         %%xmm22, %%xmm22, %%xmm22                \n"
		"    vpxorq         %%xmm23, %%xmm23, %%xmm23                \n"
		"   movss           4(%%rax), %%xmm1                         \n"
		"    vpxorq         %%xmm24, %%xmm24, %%xmm24                \n"
		"    vpxorq         %%xmm25, %%xmm25, %%xmm25                \n"
		"    vpxorq         %%xmm26, %%xmm26, %%xmm26                \n"
		"    vpxorq         %%xmm27, %%xmm27, %%xmm27                \n"
		"    vpxorq         %%xmm28, %%xmm28, %%xmm28                \n"
		"    vpxorq         %%xmm29, %%xmm29, %%xmm29                \n"
		"    vpxorq         %%xmm30, %%xmm30, %%xmm30                \n"
		"    vpxorq         %%xmm31, %%xmm31, %%xmm31                \n"

		"    subq     $8, %%rdx                                      \n"

		"MAIN_PACK_K12x1:                                            \n"

		"    KERNEL12x1_PACK_K1                                      \n"
		"    KERNEL12x1_PACK_K2                                      \n"
		"    KERNEL12x1_PACK_K1                                      \n"
		"    KERNEL12x1_PACK_K2                                      \n"
		"    KERNEL12x1_PACK_K1                                      \n"
		"    KERNEL12x1_PACK_K2                                      \n"
		"    KERNEL12x1_PACK_K1                                      \n"
		"   cmp     $0, %%rdx                                        \n"
		"    je         EDGE_PACK_K12x1                              \n"
		"    KERNEL12x1_PACK_K2                                      \n"

		"    subq     $8, %%rdx                                      \n"
		"   jmp     MAIN_PACK_K12x1                                  \n"

		"EDGE_PACK_K12x1:                                            \n"

		"    KERNEL12x1_PACK_END_K                                   \n"
		"    jmp      BEGIN_SAVE_12x1                                \n"

		//-----------------------------------------------------------------

		"BEGIN_M12x1:                                                \n"

		"    mov     %%r14, %%rbx                                    \n" // Bc
		"   prefetcht0         (%%rbx)                               \n"

		"    mov      %%rcx, %%r10                                   \n" // C0
		"   prefetcht1         (%%r10)                               \n"
		"    leaq     (%%r10, %%r8, 4), %%r11                        \n" // C1
		"   prefetcht1         (%%r11)                               \n"
		"    leaq     (%%r11, %%r8, 4), %%r12                        \n" // C2
		"   prefetcht1         (%%r12)                               \n"
		"    leaq     (%%r12, %%r8, 4), %%r13                        \n" // C3
		"   prefetcht1         (%%r13)                               \n"

		"    mov     %%rsi, %%rdx                                    \n" // K

		"   movss          (%%rbx), %%xmm4                           \n"
		"    vpxorq         %%xmm20, %%xmm20, %%xmm20                \n"
		"    vpxorq         %%xmm21, %%xmm21, %%xmm21                \n"
		"    vpxorq         %%xmm22, %%xmm22, %%xmm22                \n"
		"   movss           (%%rax), %%xmm0                          \n"
		"    vpxorq         %%xmm23, %%xmm23, %%xmm23                \n"
		"    vpxorq         %%xmm24, %%xmm24, %%xmm24                \n"
		"   movss           4(%%rax), %%xmm1                         \n"
		"    vpxorq         %%xmm25, %%xmm25, %%xmm25                \n"
		"    vpxorq         %%xmm26, %%xmm26, %%xmm26                \n"
		"    vpxorq         %%xmm27, %%xmm27, %%xmm27                \n"
		"    vpxorq         %%xmm28, %%xmm28, %%xmm28                \n"
		"    vpxorq         %%xmm29, %%xmm29, %%xmm29                \n"
		"    vpxorq         %%xmm30, %%xmm30, %%xmm30                \n"
		"    vpxorq         %%xmm31, %%xmm31, %%xmm31                \n"

		"    subq     $8, %%rdx                                      \n"

		"MAIN_K12x1:                                                 \n"

		"    KERNEL12x1_K1                                           \n"
		"    KERNEL12x1_K2                                           \n"
		"    KERNEL12x1_K1                                           \n"
		"    KERNEL12x1_K2                                           \n"
		"    KERNEL12x1_K1                                           \n"
		"    KERNEL12x1_K2                                           \n"
		"    KERNEL12x1_K1                                           \n"
		"   cmp     $0, %%rdx                                        \n"
		"    je         EDGE_K12x1                                   \n"
		"    KERNEL12x1_K2                                           \n"

		"    subq     $8, %%rdx                                      \n"
		"   jmp     MAIN_K12x1                                       \n"

		"EDGE_K12x1:                                                 \n"

		"    KERNEL12x1_END_K                                        \n"

		"BEGIN_SAVE_12x1:                                            \n"
		"    cmp     $0, %%r15                                       \n"
		"    je      SAVE_C12x1                                      \n"
		"    ADD_C_12x1                                              \n"

		"SAVE_C12x1:                                                 \n"
		"    SAVE_12x1                                               \n"

		"    cmpq      $12, %%rdi                                    \n"
		"    jnb     BEGIN_M12x1                                     \n" // 不小于（或等于）则跳转

		//------------------------------------------------------------------

		"BEGIN_M8_N1:                                                \n"
		"   cmpq      $8, %%rdi                                      \n" // M % 8
		"    jb       BEGIN_M4_N1                                    \n" // 小于则跳转

		"    mov     %%r14, %%rbx                                    \n" // Bc
		"   prefetcht0         (%%rbx)                               \n"

		"    mov      %%rcx, %%r10                                   \n" // C0
		"   prefetcht1         (%%r10)                               \n"
		"    leaq     (%%r10, %%r8, 4), %%r11                        \n" // C1
		"   prefetcht1         (%%r11)                               \n"
		"    leaq     (%%r11, %%r8, 4), %%r12                        \n" // C2
		"   prefetcht1         (%%r12)                               \n"
		"    leaq     (%%r12, %%r8, 4), %%r13                        \n" // C3
		"   prefetcht1         (%%r13)                               \n"

		"    mov     %%rsi, %%rdx                                    \n" // K
		"   movss          (%%rbx), %%xmm4                           \n"
		"    vpxorq         %%xmm24, %%xmm24, %%xmm24                \n"
		"    vpxorq         %%xmm25, %%xmm25, %%xmm25                \n"
		"   movss           (%%rax), %%xmm0                          \n"
		"    vpxorq         %%xmm26, %%xmm26, %%xmm26                \n"
		"    vpxorq         %%xmm27, %%xmm27, %%xmm27                \n"
		"   movss           4(%%rax), %%xmm1                         \n"
		"    vpxorq         %%xmm28, %%xmm28, %%xmm28                \n"
		"    vpxorq         %%xmm29, %%xmm29, %%xmm29                \n"
		"    vpxorq         %%xmm30, %%xmm30, %%xmm30                \n"
		"    vpxorq         %%xmm31, %%xmm31, %%xmm31                \n"

		"    subq     $8, %%rdx                                      \n"

		"MAIN_K_M8_N1:                                               \n"

		"    KERNEL8x1_K1                                            \n"
		"    KERNEL8x1_K2                                            \n"
		"    KERNEL8x1_K1                                            \n"
		"    KERNEL8x1_K2                                            \n"
		"    KERNEL8x1_K1                                            \n"
		"    KERNEL8x1_K2                                            \n"
		"    KERNEL8x1_K1                                            \n"
		"   cmp     $0, %%rdx                                        \n"
		"    je         EDGE_K_M8_N1                                 \n"
		"    KERNEL8x1_K2                                            \n"

		"    subq     $8, %%rdx                                      \n"
		"   jmp     MAIN_K_M8_N1                                     \n"

		"EDGE_K_M8_N1:                                               \n"

		"    KERNEL8x1_END_K                                         \n"

		"    cmp     $0, %%r15                                       \n"
		"    je      SAVE_C_8x1                                      \n"
		"    ADD_C_8x1                                               \n"

		"SAVE_C_8x1:                                                 \n"
		"    SAVE_8x1                                                \n"

		//----------------------------------------------------------------

		"BEGIN_M4_N1:                                                \n"

		"    cmpq      $4, %%rdi                                     \n" // M % 4
		"    jb       BEGIN_M1_N1                                    \n"

		"    mov     %%r14, %%rbx                                    \n" // Bc
		"   prefetcht0         (%%rbx)                               \n"

		"    mov      %%rcx, %%r10                                   \n" // C0
		"   prefetcht1         (%%r10)                               \n"
		"    leaq     (%%r10, %%r8, 4), %%r11                        \n" // C1
		"   prefetcht1         (%%r11)                               \n"
		"    leaq     (%%r11, %%r8, 4), %%r12                        \n" // C2
		"   prefetcht1         (%%r12)                               \n"
		"    leaq     (%%r12, %%r8, 4), %%r13                        \n" // C3
		"   prefetcht1         (%%r13)                               \n"

		"    mov     %%rsi, %%rdx                                    \n" // K

		"   movss          (%%rbx), %%xmm4                           \n"
		"   movss           (%%rax), %%xmm0                          \n"
		"    vpxorq         %%xmm28, %%xmm28, %%xmm28                \n"
		"    vpxorq         %%xmm29, %%xmm29, %%xmm29                \n"
		"   movss           4(%%rax), %%xmm1                         \n"
		"    vpxorq         %%xmm30, %%xmm30, %%xmm30                \n"
		"    vpxorq         %%xmm31, %%xmm31, %%xmm31                \n"

		"    subq     $8, %%rdx                                      \n"

		"MAIN_K_M4_N1:                                               \n"

		"    KERNEL4x1_K1                                            \n"
		"    KERNEL4x1_K2                                            \n"
		"    KERNEL4x1_K1                                            \n"
		"    KERNEL4x1_K2                                            \n"
		"    KERNEL4x1_K1                                            \n"
		"    KERNEL4x1_K2                                            \n"
		"    KERNEL4x1_K1                                            \n"
		"   cmp     $0, %%rdx                                        \n"
		"    je         EDGE_K_M4_N1                                 \n"
		"    KERNEL4x1_K2                                            \n"

		"    subq     $8, %%rdx                                      \n"
		"   jmp     MAIN_K_M4_N1                                     \n"

		"EDGE_K_M4_N1:                                               \n"

		"    KERNEL4x1_END_K                                         \n"

		"    cmp     $0, %%r15                                       \n"
		"    je      SAVE_C_4x1                                      \n"
		"    ADD_C_4x1                                               \n"

		"SAVE_C_4x1:                                                 \n"
		"    SAVE_4x1                                                \n"

		//----------------------------------------------------------------

		"BEGIN_M1_N1:                                                \n"
		"    cmpq      $1, %%rdi                                     \n" // M % 1
		"    jb       END_M_N1                                       \n"

		"    mov     %%r14, %%rbx                                    \n" // Bc
		"   prefetcht0         (%%rbx)                               \n"

		"    mov      %%rcx, %%r10                                   \n" // C0
		"   prefetcht1         (%%r10)                               \n"

		"    mov     %%rsi, %%rdx                                    \n" // K

		"   movss          (%%rbx), %%xmm4                           \n"
		"   movss           (%%rax), %%xmm0                          \n" // A0
		"    vpxorq         %%xmm30, %%xmm30, %%xmm30                \n"

		"    subq     $8, %%rdx                                      \n"

		"MAIN_K_M1_N1:                                               \n"
		"    KERNEL1x1_K1                                            \n"
		"    KERNEL1x1_K2                                            \n"
		"    KERNEL1x1_K1                                            \n"
		"    KERNEL1x1_K2                                            \n"
		"    KERNEL1x1_K1                                            \n"
		"    KERNEL1x1_K2                                            \n"
		"    KERNEL1x1_K1                                            \n"
		"   cmp     $0, %%rdx                                        \n"
		"    je         EDGE_K_M1_N1                                 \n"
		"    KERNEL1x1_K2                                            \n"

		"    subq     $8, %%rdx                                      \n"
		"   jmp     MAIN_K_M1_N1                                     \n"

		"EDGE_K_M1_N1:                                               \n"
		"    KERNEL1x1_END_K                                         \n"

		"    cmp     $0, %%r15                                       \n"
		"    je      SAVE_C_1x1                                      \n"
		"    ADD_C_1x1                                               \n"

		"SAVE_C_1x1:                                                 \n"
		"    SAVE_1x1                                                \n"
		"   cmpq      $1, %%rdi                                      \n"
		"    jnb     BEGIN_M1_N1                                     \n" // 不小于（或等于）则跳转

		//-----------------------------------------------------------------

		"END_M_N1:                                                   \n"

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
		[Bc] "m"(Bc),
		[k_tag] "m"(k_tag)
		: "rax", "rbx", "rcx", "rdx", "rdi", "rsi", "rbp", "r8", "r9", "r10", "r11", "r12",
		  "r13", "r14", "r15", "xmm0", "xmm1", "xmm2", "xmm3", "xmm4", "xmm5",
		  "xmm6", "xmm7", "xmm8", "xmm9", "xmm10", "xmm11", "xmm12", "xmm13",
		  "xmm14", "xmm15", "xmm16", "xmm17", "xmm18", "xmm19", "xmm20", "xmm21",
		  "xmm22", "xmm23", "xmm24", "xmm25", "xmm26", "xmm27", "xmm28", "xmm29",
		  "xmm30", "xmm31", "memory");
}
