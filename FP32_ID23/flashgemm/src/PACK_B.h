#include <string.h>
#include <stdio.h>

void NPACKB_nkx32(float *B, float *Bc, int K, int LN);
void NPACKB_nkx16(float *B, float *Bc, int K, int LN);
void NPACKB_nkx1(float *B, float *Bc, int K, int LN);

void NPACKB(float *B, float *Bc, int N, int K, int LN)
{
    float *temp_B = B;
    float *temp_Bc = Bc;

    if (N == 32)
    {
        // printf("N = %d, pack n 32\n",N);
        NPACKB_nkx32(temp_B, temp_Bc, K, LN);
        return;
    }
    if (N >= 16)
    {
        // printf("N = %d,pack m8\n",N);
        NPACKB_nkx16(temp_B, temp_Bc, K, LN);
        temp_B = temp_B + 16;
        temp_Bc = temp_Bc + 16 * K;
        N = N - 16;
    }
    // if (N >= 4)
    // {
    //     // printf("N = %d,pack m4\n",N);
    //     NPACK_m4xk(temp_B, temp_Bc, K, LN);
    //     temp_B = temp_B + 4 * LN;
    //     temp_Bc = temp_Bc + 4 * K;
    //     N = N - 4;
    // }
    if (N >= 1)
    {
        // printf("N = %d,pack m1\n",N);
        NPACKB_nkx1(temp_B, temp_Bc, K, LN);
        temp_B = temp_B + 1;
        temp_Bc = temp_Bc + 1 * K;
        N = N - 1;
    }
    if (N >= 1)
    {
        // printf("N = %d,pack m1\n",N);
        NPACKB_nkx1(temp_B, temp_Bc, K, LN);
        temp_B = temp_B + 1;
        temp_Bc = temp_Bc + 1 * K;
        N = N - 1;
    }
    if (N >= 1)
    {
        // printf("N = %d,pack m1\n",N);
        NPACKB_nkx1(temp_B, temp_Bc, K, LN);
    }
}

void NPACKB_nkx32(float *B, float *Bc, int K, int LN)
{
    asm volatile(
        "   movl %[LN], %%r8d                                        \n"
        "   movl %[LN], %%r15d                                       \n"

        "   mov %[Bc], %%r9                                          \n"
        "   movl %[K], %%r10d                                        \n"

        "   mov %[B], %%r11                                          \n"
        "   mov %[B], %%r12                                          \n"
        "   mov %[B], %%r13                                          \n"
        "   mov %[B], %%r14                                          \n"

        "   mov %[B], %%rbx                                          \n"
        "   mov %[B], %%rcx                                          \n"
        "   mov %[B], %%rdx                                          \n"
        "   mov %[B], %%rdi                                          \n"

        "   shl $2, %%r8                                             \n" // LN*4
        "   shl $3, %%r15                                            \n" // LN*8
        "   add %%r8, %%r12                                          \n"
        "   add %%r8, %%r14                                          \n"
        "   add %%r8, %%rcx                                          \n"
        "   add %%r8, %%rdi                                          \n"
        "   add %%r15, %%r13                                         \n"
        "   add %%r15, %%r14                                         \n"
        "   shl $2, %%r8                                             \n" // LN*16
        "   add %%r15, %%rdx                                         \n"
        "   add %%r15, %%rdi                                         \n"
        "   add %%r8, %%rbx                                          \n"
        "   add %%r8, %%rcx                                          \n"
        "   add %%r8, %%rdx                                          \n"
        "   add %%r8, %%rdi                                          \n"
        "   shl $1, %%r8                                             \n" // LN*32

        "NPACK_B32_8:                                                \n"

        "   vmovups (%%r11), %%zmm2                                  \n"
        "   vmovups 64(%%r11), %%zmm10                               \n"
        "   vmovups %%zmm2, (%%r9)                                   \n"
        "   vmovups %%zmm10, 64(%%r9)                                \n"
        "   add     $128, %%r9                                       \n"
        "   add     %%r8, %%r11                                      \n"

        "   vmovups (%%r12), %%zmm3                                  \n"
        "   vmovups 64(%%r12), %%zmm11                               \n"
        "   vmovups %%zmm3, (%%r9)                                   \n"
        "   vmovups %%zmm11, 64(%%r9)                                \n"
        "   add     $128, %%r9                                       \n"
        "   add     %%r8, %%r12                                      \n"

        "   vmovups (%%r13), %%zmm4                                  \n"
        "   vmovups 64(%%r13), %%zmm12                               \n"
        "   vmovups %%zmm4, (%%r9)                                   \n"
        "   vmovups %%zmm12, 64(%%r9)                                \n"
        "   add     $128, %%r9                                       \n"
        "   add     %%r8, %%r13                                      \n"

        "   vmovups (%%r14), %%zmm5                                  \n"
        "   vmovups 64(%%r14), %%zmm13                               \n"
        "   vmovups %%zmm5, (%%r9)                                   \n"
        "   vmovups %%zmm13, 64(%%r9)                                \n"
        "   add     $128, %%r9                                       \n"
        "   add     %%r8, %%r14                                      \n"

        "   vmovups (%%rbx), %%zmm6                                  \n"
        "   vmovups 64(%%rbx), %%zmm14                               \n"
        "   vmovups %%zmm6, (%%r9)                                   \n"
        "   vmovups %%zmm14, 64(%%r9)                                \n"
        "   add     $128, %%r9                                       \n"
        "   add     %%r8, %%rbx                                      \n"

        "   vmovups (%%rcx), %%zmm7                                  \n"
        "   vmovups 64(%%rcx), %%zmm15                               \n"
        "   vmovups %%zmm7, (%%r9)                                   \n"
        "   vmovups %%zmm15, 64(%%r9)                                \n"
        "   add     $128, %%r9                                       \n"
        "   add     %%r8, %%rcx                                      \n"

        "   vmovups (%%rdx), %%zmm8                                  \n"
        "   vmovups 64(%%rdx), %%zmm16                               \n"
        "   vmovups %%zmm8, (%%r9)                                   \n"
        "   vmovups %%zmm16, 64(%%r9)                                \n"
        "   add     $128, %%r9                                       \n"
        "   add     %%r8, %%rdx                                      \n"

        "   vmovups (%%rdi), %%zmm9                                  \n"
        "   vmovups 64(%%rdi), %%zmm17                               \n"
        "   vmovups %%zmm9, (%%r9)                                   \n"
        "   vmovups %%zmm17, 64(%%r9)                                \n"
        "   add     $128, %%r9                                       \n"
        "   add     %%r8, %%rdi                                      \n"

        "   sub    $8, %%r10                                         \n"

        "   je  NPACK_B32_END_8                                      \n"
        "   jmp NPACK_B32_8                                          \n"

        "NPACK_B32_END_8:                                            \n"

        :

        :
        [B] "m"(B),
        [Bc] "m"(Bc),
        [K] "m"(K),
        [LN] "m"(LN)

        : "rax", "rbx", "rcx", "rdx", "rdi", "rsi", "r8", "r9", "r10", "r11", "r12", "r13", "r14", "r15", "zmm0", "zmm1", "zmm2", "zmm3", "zmm4", "zmm5", "zmm6", "zmm7", "zmm8", "zmm9", "zmm10", "zmm11", "zmm12", "zmm13", "zmm14", "zmm15", "zmm16", "zmm17", "zmm18", "zmm19", "zmm20", "zmm21", "zmm22", "zmm23", "zmm24", "zmm25", "zmm26", "zmm27", "zmm28", "zmm29", "zmm30", "zmm31", "memory", "k0", "k1", "k2", "k3", "k4");
}

void NPACKB_nkx16(float *B, float *Bc, int K, int LN)
{
    asm volatile(
        "   movl %[LN], %%r8d                                        \n"
        "   movl %[LN], %%r15d                                       \n"

        "   mov %[Bc], %%r9                                          \n"
        "   movl %[K], %%r10d                                        \n"

        "   mov %[B], %%r11                                          \n"
        "   mov %[B], %%r12                                          \n"
        "   mov %[B], %%r13                                          \n"
        "   mov %[B], %%r14                                          \n"

        "   mov %[B], %%rbx                                          \n"
        "   mov %[B], %%rcx                                          \n"
        "   mov %[B], %%rdx                                          \n"
        "   mov %[B], %%rdi                                          \n"

        "   shl $2, %%r8                                             \n" // LN*4
        "   shl $3, %%r15                                            \n" // LN*8
        "   add %%r8, %%r12                                          \n"
        "   add %%r8, %%r14                                          \n"
        "   add %%r8, %%rcx                                          \n"
        "   add %%r8, %%rdi                                          \n"
        "   add %%r15, %%r13                                         \n"
        "   add %%r15, %%r14                                         \n"
        "   shl $2, %%r8                                             \n" // LN*16
        "   add %%r15, %%rdx                                         \n"
        "   add %%r15, %%rdi                                         \n"
        "   add %%r8, %%rbx                                          \n"
        "   add %%r8, %%rcx                                          \n"
        "   add %%r8, %%rdx                                          \n"
        "   add %%r8, %%rdi                                          \n"
        "   shl $1, %%r8                                             \n" // LN*32

        "NPACK_B16:                                                  \n"

        "   vmovups (%%r11), %%zmm2                                  \n"
        "   vmovups %%zmm2, (%%r9)                                   \n"
        "   add     $64, %%r9                                        \n"
        "   add     %%r8, %%r11                                      \n"

        "   vmovups (%%r12), %%zmm3                                  \n"
        "   vmovups %%zmm3, (%%r9)                                   \n"
        "   add     $64, %%r9                                        \n"
        "   add     %%r8, %%r12                                      \n"

        "   vmovups (%%r13), %%zmm4                                  \n"
        "   vmovups %%zmm4, (%%r9)                                   \n"
        "   add     $64, %%r9                                        \n"
        "   add     %%r8, %%r13                                      \n"

        "   vmovups (%%r14), %%zmm5                                  \n"
        "   vmovups %%zmm5, (%%r9)                                   \n"
        "   add     $64, %%r9                                        \n"
        "   add     %%r8, %%r14                                      \n"

        "   vmovups (%%rbx), %%zmm6                                  \n"
        "   vmovups %%zmm6, (%%r9)                                   \n"
        "   add     $64, %%r9                                        \n"
        "   add     %%r8, %%rbx                                      \n"

        "   vmovups (%%rcx), %%zmm7                                  \n"
        "   vmovups %%zmm7, (%%r9)                                   \n"
        "   add     $64, %%r9                                        \n"
        "   add     %%r8, %%rcx                                      \n"

        "   vmovups (%%rdx), %%zmm8                                  \n"
        "   vmovups %%zmm8, (%%r9)                                   \n"
        "   add     $64, %%r9                                        \n"
        "   add     %%r8, %%rdx                                      \n"

        "   vmovups (%%rdi), %%zmm9                                  \n"
        "   vmovups %%zmm9, (%%r9)                                   \n"
        "   add     $64, %%r9                                        \n"
        "   add     %%r8, %%rdi                                      \n"

        "   sub    $8, %%r10                                         \n"

        "   je  NPACK_B16_END                                        \n"
        "   jmp NPACK_B16                                            \n"

        "NPACK_B16_END:                                              \n"

        :

        :
        [B] "m"(B),
        [Bc] "m"(Bc),
        [K] "m"(K),
        [LN] "m"(LN)

        : "rax", "rbx", "rcx", "rdx", "rdi", "rsi", "r8", "r9", "r10", "r11", "r12", "r13", "r14", "r15", "zmm0", "zmm1", "zmm2", "zmm3", "zmm4", "zmm5", "zmm6", "zmm7", "zmm8", "zmm9", "zmm10", "zmm11", "zmm12", "zmm13", "zmm14", "zmm15", "zmm16", "zmm17", "zmm18", "zmm19", "zmm20", "zmm21", "zmm22", "zmm23", "zmm24", "zmm25", "zmm26", "zmm27", "zmm28", "zmm29", "zmm30", "zmm31", "memory", "k0", "k1", "k2", "k3", "k4");
}

void NPACKB_nkx1(float *B, float *Bc, int K, int LN)
{
    asm volatile(
        "   movl %[LN], %%r8d                                        \n"
        "   movl %[LN], %%r15d                                       \n"

        "   mov %[Bc], %%r9                                          \n"
        "   movl %[K], %%r10d                                        \n"

        "   mov %[B], %%r11                                          \n"
        "   mov %[B], %%r12                                          \n"
        "   mov %[B], %%r13                                          \n"
        "   mov %[B], %%r14                                          \n"

        "   mov %[B], %%rbx                                          \n"
        "   mov %[B], %%rcx                                          \n"
        "   mov %[B], %%rdx                                          \n"
        "   mov %[B], %%rdi                                          \n"

        "   shl $2, %%r8                                             \n" // LN*4
        "   shl $3, %%r15                                            \n" // LN*8
        "   add %%r8, %%r12                                          \n"
        "   add %%r8, %%r14                                          \n"
        "   add %%r8, %%rcx                                          \n"
        "   add %%r8, %%rdi                                          \n"
        "   add %%r15, %%r13                                         \n"
        "   add %%r15, %%r14                                         \n"
        "   shl $2, %%r8                                             \n" // LN*16
        "   add %%r15, %%rdx                                         \n"
        "   add %%r15, %%rdi                                         \n"
        "   add %%r8, %%rbx                                          \n"
        "   add %%r8, %%rcx                                          \n"
        "   add %%r8, %%rdx                                          \n"
        "   add %%r8, %%rdi                                          \n"
        "   shl $1, %%r8                                             \n" // LN*32

        "NPACK_B1:                                                   \n"

        "   movss (%%r11), %%xmm2                                    \n"
        "   vmovss %%xmm2, (%%r9)                                    \n"
        "   add     $4, %%r9                                         \n"
        "   add     %%r8, %%r11                                      \n"

        "   movss (%%r12), %%xmm3                                    \n"
        "   vmovss %%xmm3, (%%r9)                                    \n"
        "   add     $4, %%r9                                         \n"
        "   add     %%r8, %%r12                                      \n"

        "   movss (%%r13), %%xmm4                                    \n"
        "   vmovss %%xmm4, (%%r9)                                    \n"
        "   add     $4, %%r9                                         \n"
        "   add     %%r8, %%r13                                      \n"

        "   movss (%%r14), %%xmm5                                    \n"
        "   vmovss %%xmm5, (%%r9)                                    \n"
        "   add     $4, %%r9                                         \n"
        "   add     %%r8, %%r14                                      \n"

        "   movss (%%rbx), %%xmm6                                    \n"
        "   vmovss %%xmm6, (%%r9)                                    \n"
        "   add     $4, %%r9                                         \n"
        "   add     %%r8, %%rbx                                      \n"

        "   movss (%%rcx), %%xmm7                                    \n"
        "   vmovss %%xmm7, (%%r9)                                    \n"
        "   add     $4, %%r9                                         \n"
        "   add     %%r8, %%rcx                                      \n"

        "   movss (%%rdx), %%xmm8                                    \n"
        "   vmovss %%xmm8, (%%r9)                                    \n"
        "   add     $4, %%r9                                         \n"
        "   add     %%r8, %%rdx                                      \n"

        "   movss (%%rdi), %%xmm9                                    \n"
        "   vmovss %%xmm9, (%%r9)                                    \n"
        "   add     $4, %%r9                                         \n"
        "   add     %%r8, %%rdi                                      \n"

        "   sub    $8, %%r10                                         \n"

        "   je  NPACK_B1_END                                         \n"
        "   jmp NPACK_B1                                             \n"

        "NPACK_B1_END:                                               \n"

        :

        :
        [B] "m"(B),
        [Bc] "m"(Bc),
        [K] "m"(K),
        [LN] "m"(LN)

        : "rax", "rbx", "rcx", "rdx", "rdi", "rsi", "r8", "r9", "r10", "r11", "r12", "r13", "r14", "r15", "zmm0", "zmm1", "zmm2", "zmm3", "zmm4", "zmm5", "zmm6", "zmm7", "zmm8", "zmm9", "zmm10", "zmm11", "zmm12", "zmm13", "zmm14", "zmm15", "zmm16", "zmm17", "zmm18", "zmm19", "zmm20", "zmm21", "zmm22", "zmm23", "zmm24", "zmm25", "zmm26", "zmm27", "zmm28", "zmm29", "zmm30", "zmm31", "memory", "k0", "k1", "k2", "k3", "k4");
}
