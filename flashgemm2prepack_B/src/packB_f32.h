static void NPACK_B_K16N32(long N, long LN, long K, float *B, float *Bc)
{
  asm volatile(
      "BEGIN_NPACK_B_matrix:                                       \n"
      "   mov %[N], %%rsi                                          \n"
      "   mov %[LN], %%rdi                                         \n"
      "   mov %[K], %%rdx                                          \n"
      "   mov %[B], %%rbx                                          \n"
      "   mov %[Bc], %%rbp                                         \n"

      "   mov  %%rdx, %%r8                                         \n"
      "   imul $128, %%r8                                          \n" // Bc+=4(size_of_float)*K*32(nr)

      "   mov  %%rdi, %%r9                                         \n"
      "   imul $32, %%r9                                           \n" // B+=4(size_of_float)*LN*8(loop_k)

      "   mov  %%rbp, %%r10                                        \n"
      "   mov  %%rbx, %%r11                                        \n"

      "   mov %%rsi, %%rcx                                         \n"

      "NPACK_B_N32:                                                \n"
      "   movq %%rbx, %%r12                                        \n"
      "   leaq (%%r12, %%rdi, 4), %%r13                            \n"
      "   leaq (%%r13, %%rdi, 4), %%r14                            \n"
      "   leaq (%%r14, %%rdi, 4), %%r15                            \n"

      "   vmovups (%%r12), %%zmm0                                  \n"
      "   vmovups 64(%%r12), %%zmm1                                \n"
      "   vmovups (%%r13), %%zmm2                                  \n"
      "   vmovups 64(%%r13), %%zmm3                                \n"
      "   vmovups (%%r14), %%zmm4                                  \n"
      "   vmovups 64(%%r14), %%zmm5                                \n"
      "   vmovups (%%r15), %%zmm6                                  \n"
      "   vmovups 64(%%r15), %%zmm7                                \n"

      "   vmovups %%zmm0, (%%rbp)                                  \n"
      "   vmovups %%zmm1, 64(%%rbp)                                \n"
      "   vmovups %%zmm2, 128(%%rbp)                               \n"
      "   vmovups %%zmm3, 192(%%rbp)                               \n"
      "   vmovups %%zmm4, 256(%%rbp)                               \n"
      "   vmovups %%zmm5, 320(%%rbp)                               \n"
      "   vmovups %%zmm6, 384(%%rbp)                               \n"
      "   vmovups %%zmm7, 448(%%rbp)                               \n"

      "   leaq (%%r15, %%rdi, 4), %%r12                            \n"
      "   leaq (%%r12, %%rdi, 4), %%r13                            \n"
      "   leaq (%%r13, %%rdi, 4), %%r14                            \n"
      "   leaq (%%r14, %%rdi, 4), %%r15                            \n"

      "   vmovups (%%r12), %%zmm8                                  \n"
      "   vmovups 64(%%r12), %%zmm9                                \n"
      "   vmovups (%%r13), %%zmm10                                 \n"
      "   vmovups 64(%%r13), %%zmm11                               \n"
      "   vmovups (%%r14), %%zmm12                                 \n"
      "   vmovups 64(%%r14), %%zmm13                               \n"
      "   vmovups (%%r15), %%zmm14                                 \n"
      "   vmovups 64(%%r15), %%zmm15                               \n"

      "   vmovups %%zmm8, 512(%%rbp)                               \n"
      "   vmovups %%zmm9, 576(%%rbp)                               \n"
      "   vmovups %%zmm10, 640(%%rbp)                              \n"
      "   vmovups %%zmm11, 704(%%rbp)                              \n"
      "   vmovups %%zmm12, 768(%%rbp)                              \n"
      "   vmovups %%zmm13, 832(%%rbp)                              \n"
      "   vmovups %%zmm14, 896(%%rbp)                              \n"
      "   vmovups %%zmm15, 960(%%rbp)                              \n"

      "   addq $128, %%rbx                                         \n" // 4*32
      "   addq %%r8, %%rbp                                         \n"

      "   subq $32, %%rcx                                          \n"
      "   cmp  $0, %%rcx                                           \n"
      "   jg   NPACK_B_N32                                         \n"

      "NPACK_B_K8:                                                 \n"
      "   subq $8, %%rdx                                           \n"
      "   cmp  $0, %%rdx                                           \n"
      "   je   NPACK_B_END                                         \n"
      "   addq %%r9, %%r11                                         \n"
      "   mov  %%r11, %%rbx                                        \n"
      "   addq $1024, %%r10                                        \n" // K(8)*nr(32)*size_of_float(4)
      "   mov  %%r10, %%rbp                                        \n"
      "   mov  %%rsi, %%rcx                                        \n"
      "   jmp  NPACK_B_N32                                         \n"

      "NPACK_B_END:                                                \n"

      :
      :
      [N] "m"(N),
      [LN] "m"(LN),
      [K] "m"(K),
      [B] "m"(B),
      [Bc] "m"(Bc)
      : "rax", "rbx", "rcx", "rdx", "rdi", "rsi", "r8", "r9", "r10", "r11", "r12", "r13", "r14", "r15", "zmm0", "zmm1", "zmm2", "zmm3", "zmm4", "zmm5", "zmm6", "zmm7", "zmm8", "zmm9", "zmm10", "zmm11", "zmm12", "zmm13", "zmm14", "zmm15", "zmm16", "zmm17", "zmm18", "zmm19", "zmm20", "zmm21", "zmm22", "zmm23", "zmm24", "zmm25", "zmm26", "zmm27", "zmm28", "zmm29", "zmm30", "zmm31", "memory", "k0", "k1", "k2", "k3", "k4");
}
