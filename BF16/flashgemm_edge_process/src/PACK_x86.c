#include <string.h>

static void NPACK_m1xk(float *A, float *Ac, int K, int LK)
{
    memcpy(Ac, A, K * 4);
}

static void NPACK_m4xk(float *A, float *Ac, int K, int LK)
{
    asm volatile(
        "   movl %[LK], %%r8d    \n"
        "   movl %[LK], %%r15d    \n"

        "   mov %[Ac], %%r9    \n"
        "   movl %[K], %%r10d   \n"
        "   mov %[A], %%r11    \n"
        "   mov %[A], %%r12    \n"
        "   mov %[A], %%r13    \n"
        "   mov %[A], %%r14    \n"

        "   shr $4, %%r10       \n"
        "   shl $2, %%r8       \n"
        "   shl $3, %%r15       \n"

        "   add %%r8, %%r14     \n"
        "   add %%r8, %%r12     \n"
        "   add %%r15, %%r13     \n"
        "   add %%r15, %%r14     \n"

        "   vmovups (%%r11), %%zmm2       \n"
        "   vmovups (%%r12), %%zmm3       \n"
        "   vmovups (%%r13), %%zmm4       \n"
        "   vmovups (%%r14), %%zmm5       \n"
        "   jmp NPACK                   \n"

        "NPACK_Pre:              \n"
        "   add $64, %%r11      \n"
        "   add $64, %%r12      \n"
        "   add $64, %%r13      \n"
        "   add $64, %%r14      \n"
        "   add $256, %%r9      \n"
        "   vmovups (%%r11), %%zmm2       \n"
        "   vmovups (%%r12), %%zmm3       \n"
        "   vmovups (%%r13), %%zmm4       \n"
        "   vmovups (%%r14), %%zmm5       \n"
        "NPACK:                 \n"

        "   movl    $0xaa, %%eax       \n"
        "   kmovd   %%eax, %%k1            \n"
        "   vunpcklps %%zmm3, %%zmm2, %%zmm0    \n"
        "   vunpcklps %%zmm5, %%zmm4, %%zmm26    \n"

        "   vpermq   $0x80, %%zmm26, %%zmm0%{%%k1%}      \n"
        "   vmovups %%xmm0, (%%r9)         \n"
        "   vextractf32x4  $0x1, %%zmm0, %%xmm28      \n"
        "   vextractf32x4  $0x2, %%zmm0, %%xmm29      \n"
        "   vextractf32x4  $0x3, %%zmm0, %%xmm30      \n"

        "   vmovups %%xmm28, 64(%%r9)         \n"
        "   vmovups %%xmm29, 128(%%r9)         \n"
        "   vmovups %%xmm30, 192(%%r9)         \n"

        "   movl    $0x55, %%eax       \n"
        "   kmovd   %%eax, %%k1            \n"

        "   vunpcklps %%zmm3, %%zmm2, %%zmm0    \n"
        "   vunpcklps %%zmm5, %%zmm4, %%zmm26    \n"

        "   vpermq  $0x31, %%zmm0, %%zmm26%{%%k1%}      \n"
        "   vmovups %%xmm26, 16(%%r9)         \n"
        "   vextractf32x4  $0x1, %%zmm26, %%xmm28      \n"
        "   vextractf32x4  $0x2, %%zmm26, %%xmm29      \n"
        "   vextractf32x4  $0x3, %%zmm26, %%xmm30      \n"

        "   vmovups %%xmm28, 80(%%r9)         \n"
        "   vmovups %%xmm29, 144(%%r9)         \n"
        "   vmovups %%xmm30, 208(%%r9)         \n"

        "   movl    $0xaa, %%eax       \n"
        "   kmovd   %%eax, %%k1            \n"
        "   vunpckhps %%zmm3, %%zmm2, %%zmm0    \n"
        "   vunpckhps %%zmm5, %%zmm4, %%zmm26    \n"
        "   vpermq   $0x80, %%zmm26, %%zmm0%{%%k1%}      \n"
        "   vmovups %%xmm0, 32(%%r9)         \n"
        "   vextractf32x4  $0x1, %%zmm0, %%xmm28      \n"
        "   vextractf32x4  $0x2, %%zmm0, %%xmm29      \n"
        "   vextractf32x4  $0x3, %%zmm0, %%xmm30      \n"

        "   vmovups %%xmm28, 96(%%r9)         \n"
        "   vmovups %%xmm29, 160(%%r9)         \n"
        "   vmovups %%xmm30, 224(%%r9)         \n"
        "   movl    $0x55, %%eax       \n"
        "   kmovd   %%eax, %%k1            \n"

        "   vunpckhps %%zmm3, %%zmm2, %%zmm0    \n"
        "   vunpckhps %%zmm5, %%zmm4, %%zmm26    \n"

        "   vpermq  $0x31, %%zmm0, %%zmm26%{%%k1%}      \n"
        "   vmovups %%xmm26, 48(%%r9)         \n"
        "   vextractf32x4  $0x1, %%zmm26, %%xmm28      \n"
        "   vextractf32x4  $0x2, %%zmm26, %%xmm29      \n"
        "   vextractf32x4  $0x3, %%zmm26, %%xmm30      \n"
        "   sub    $1, %%r10       \n"
        "   vmovups %%xmm28, 112(%%r9)         \n"
        "   vmovups %%xmm29, 176(%%r9)         \n"
        "   vmovups %%xmm30, 240(%%r9)         \n"

        "   je  NPACK_END           \n"
        "   jmp NPACK_Pre           \n"

        "NPACK_END:                         \n"
        :

        :
        [A] "m"(A),
        [Ac] "m"(Ac),
        [K] "m"(K),
        [LK] "m"(LK)

        : "rax", "rbx", "rcx", "rdx", "rdi", "rsi", "r8", "r9", "r10", "r11", "r12", "r13", "r14", "r15", "zmm0", "zmm1", "zmm2", "zmm3", "zmm4", "zmm5", "zmm6", "zmm7", "zmm8", "zmm9", "zmm10", "zmm11", "zmm12", "zmm13", "zmm14", "zmm15", "zmm16", "zmm17", "zmm18", "zmm19", "zmm20", "zmm21", "zmm22", "zmm23", "zmm24", "zmm25", "zmm26", "zmm27", "zmm28", "zmm29", "zmm30", "zmm31", "memory", "k0", "k1", "k2", "k3", "k4");
}

static void NPACK_m8xk(float *A, float *Ac, int K, int LK)
{
    asm volatile(
        "   movl %[LK], %%r8d    \n"
        "   movl %[LK], %%r15d    \n"

        "   mov %[Ac], %%r9    \n"
        "   movl %[K], %%r10d   \n"
        "   mov %[A], %%r11    \n"
        "   mov %[A], %%r12    \n"
        "   mov %[A], %%r13    \n"
        "   mov %[A], %%r14    \n"

        "   mov %[A], %%rbx    \n"
        "   mov %[A], %%rcx    \n"
        "   mov %[A], %%rdx    \n"
        "   mov %[A], %%rdi    \n"

        "   shr $4, %%r10       \n"
        "   shl $2, %%r8       \n"
        "   shl $3, %%r15       \n"
        "   add %%r8, %%r14     \n"
        "   add %%r8, %%rdi     \n"
        "   add %%r8, %%r12     \n"
        "   add %%r8, %%rcx     \n"
        "   add %%r15, %%r13     \n"
        "   add %%r15, %%rdx     \n"
        "   shl $2, %%r8       \n"
        "   add %%r15, %%r14     \n"
        "   add %%r15, %%rdi     \n"
        "   add %%r8, %%rbx     \n"
        "   add %%r8, %%rcx     \n"
        "   add %%r8, %%rdx     \n"
        "   add %%r8, %%rdi     \n"

        "   vmovups (%%r11), %%zmm2       \n"
        "   vmovups (%%r12), %%zmm3       \n"
        "   vmovups (%%r13), %%zmm4       \n"
        "   vmovups (%%r14), %%zmm5       \n"
        "   vmovups (%%rbx), %%zmm6       \n"
        "   vmovups (%%rcx), %%zmm7       \n"
        "   vmovups (%%rdx), %%zmm8       \n"
        "   vmovups (%%rdi), %%zmm9       \n"
        "   jmp NPACK_8                   \n"

        "NPACK_Pre_8:              \n"
        "   add $64, %%r11      \n"
        "   add $64, %%r12      \n"
        "   add $64, %%r13      \n"
        "   add $64, %%r14      \n"
        "   add $64, %%rbx      \n"
        "   add $64, %%rcx      \n"
        "   add $64, %%rdx      \n"
        "   add $64, %%rdi      \n"
        "   add $512, %%r9      \n"
        "   vmovups (%%r11), %%zmm2       \n"
        "   vmovups (%%r12), %%zmm3       \n"
        "   vmovups (%%r13), %%zmm4       \n"
        "   vmovups (%%r14), %%zmm5       \n"
        "   vmovups (%%rbx), %%zmm6       \n"
        "   vmovups (%%rcx), %%zmm7       \n"
        "   vmovups (%%rdx), %%zmm8       \n"
        "   vmovups (%%rdi), %%zmm9       \n"
        "NPACK_8:                 \n"

        "   movl    $0xaa, %%eax       \n"
        "   movl    $0xcc, %%r8d         \n"
        "   movl    $0x33, %%r15d           \n"

        "   kmovd   %%eax, %%k1            \n"
        "   kmovd   %%r8d, %%k2             \n"
        "   kmovd   %%r15d, %%k3             \n"

        "   vunpcklps %%zmm3, %%zmm2, %%zmm0    \n"
        "   vunpcklps %%zmm7, %%zmm6, %%zmm1    \n"

        "   vunpcklps %%zmm5, %%zmm4, %%zmm26    \n"
        "   vunpcklps %%zmm9, %%zmm8, %%zmm27    \n"

        "   vpermq  $0x80, %%zmm26, %%zmm0%{%%k1%}      \n"
        "   vmovups %%zmm0, %%zmm28     \n"
        "   vpermq  $0x80, %%zmm27, %%zmm1%{%%k1%}      \n"
        "   vmovups %%zmm1, %%zmm29     \n"

        "   vpermq  $0x40, %%zmm1, %%zmm0%{%%k2%}      \n"
        "   vpermq  $0x0e, %%zmm28, %%zmm29%{%%k3%}        \n"

        "   vmovups %%ymm0, (%%r9)         \n"
        "   vextractf64x4 $0x1, %%zmm0,  %%ymm30      \n"
        "   vmovups %%ymm30, 256(%%r9)            \n"

        "   vmovups %%ymm29, 128(%%r9)         \n"
        "   vextractf64x4 $0x1, %%zmm29,  %%ymm31      \n"
        "   vmovups %%ymm31, 384(%%r9)            \n"

        "   movl    $0x55, %%eax       \n"
        "   movl    $0xcc, %%r8d         \n"
        "   movl    $0x33, %%r15d           \n"

        "   kmovd   %%eax, %%k1            \n"
        "   kmovd   %%r8d, %%k2             \n"
        "   kmovd   %%r15d, %%k3             \n"

        "   vunpcklps %%zmm3, %%zmm2, %%zmm0    \n"
        "   vunpcklps %%zmm7, %%zmm6, %%zmm1    \n"

        "   vunpcklps %%zmm5, %%zmm4, %%zmm26    \n"
        "   vunpcklps %%zmm9, %%zmm8, %%zmm27    \n"

        "   vpermq  $0x31, %%zmm0, %%zmm26%{%%k1%}      \n"
        "   vmovups %%zmm26, %%zmm28     \n"
        "   vpermq  $0x31, %%zmm1, %%zmm27%{%%k1%}      \n"
        "   vmovups %%zmm27, %%zmm29     \n"

        "   vpermq  $0x40, %%zmm27, %%zmm26%{%%k2%}      \n"
        "   vpermq  $0x0e, %%zmm28, %%zmm29%{%%k3%}        \n"

        "   vmovups %%ymm26, 32(%%r9)         \n"
        "   vextractf64x4  $0x1,%%zmm26, %%ymm30      \n"
        "   vmovups %%ymm30, 288(%%r9)            \n"

        "   vmovups %%ymm29, 160(%%r9)         \n"
        "   vextractf64x4 $0x1, %%zmm29,  %%ymm31      \n"
        "   vmovups %%ymm31, 416(%%r9)            \n"

        "   movl    $0xaa, %%eax       \n"
        "   movl    $0xcc, %%r8d         \n"
        "   movl    $0x33, %%r15d           \n"

        "   kmovd   %%eax, %%k1            \n"
        "   kmovd   %%r8d, %%k2             \n"
        "   kmovd   %%r15d, %%k3             \n"

        "   vunpckhps %%zmm3, %%zmm2, %%zmm0    \n"
        "   vunpckhps %%zmm7, %%zmm6, %%zmm1    \n"

        "   vunpckhps %%zmm5, %%zmm4, %%zmm26    \n"
        "   vunpckhps %%zmm9, %%zmm8, %%zmm27    \n"

        "   vpermq   $0x80, %%zmm26, %%zmm0%{%%k1%}      \n"
        "   vmovups %%zmm0, %%zmm28     \n"
        "   vpermq   $0x80, %%zmm27, %%zmm1%{%%k1%}      \n"
        "   vmovups %%zmm1, %%zmm29     \n"

        "   vpermq  $0x40, %%zmm1, %%zmm0%{%%k2%}      \n"
        "   vpermq  $0x0e, %%zmm28, %%zmm29%{%%k3%}        \n"

        "   vmovups %%ymm0, 64(%%r9)         \n"
        "   vextractf64x4 $0x1, %%zmm0,  %%ymm30      \n"
        "   vmovups %%ymm30, 320(%%r9)            \n"

        "   vmovups %%ymm29, 192(%%r9)         \n"
        "   vextractf64x4 $0x1, %%zmm29,  %%ymm31      \n"
        "   vmovups %%ymm31, 448(%%r9)            \n"

        "   movl    $0x55, %%eax       \n"
        "   movl    $0xcc, %%r8d         \n"
        "   movl    $0x33, %%r15d           \n"

        "   kmovd   %%eax, %%k1            \n"
        "   kmovd   %%r8d, %%k2             \n"
        "   kmovd   %%r15d, %%k3             \n"

        "   vunpckhps %%zmm3, %%zmm2, %%zmm0    \n"
        "   vunpckhps %%zmm7, %%zmm6, %%zmm1    \n"

        "   vunpckhps %%zmm5, %%zmm4, %%zmm26    \n"
        "   vunpckhps %%zmm9, %%zmm8, %%zmm27    \n"

        "   vpermq  $0x31, %%zmm0, %%zmm26%{%%k1%}      \n"
        "   vmovups %%zmm26, %%zmm28     \n"
        "   vpermq  $0x31, %%zmm1, %%zmm27%{%%k1%}      \n"
        "   vmovups %%zmm27, %%zmm29     \n"

        "   vpermq  $0x40, %%zmm27, %%zmm26%{%%k2%}      \n"
        "   vpermq  $0x0e, %%zmm28, %%zmm29%{%%k3%}        \n"

        "   vmovups %%ymm26, 96(%%r9)         \n"
        "   vextractf64x4  $0x1,%%zmm26, %%ymm30      \n"
        "   vmovups %%ymm30, 352(%%r9)            \n"

        "   sub    $1, %%r10       \n"

        "   vmovups %%ymm29, 224(%%r9)         \n"
        "   vextractf64x4 $0x1, %%zmm29,  %%ymm31      \n"
        "   vmovups %%ymm31, 480(%%r9)            \n"

        "   je  NPACK_END_8           \n"
        "   jmp NPACK_Pre_8           \n"

        "NPACK_END_8:                         \n"
        :

        :
        [A] "m"(A),
        [Ac] "m"(Ac),
        [K] "m"(K),
        [LK] "m"(LK)

        : "rax", "rbx", "rcx", "rdx", "rdi", "rsi", "r8", "r9", "r10", "r11", "r12", "r13", "r14", "r15", "zmm0", "zmm1", "zmm2", "zmm3", "zmm4", "zmm5", "zmm6", "zmm7", "zmm8", "zmm9", "zmm10", "zmm11", "zmm12", "zmm13", "zmm14", "zmm15", "zmm16", "zmm17", "zmm18", "zmm19", "zmm20", "zmm21", "zmm22", "zmm23", "zmm24", "zmm25", "zmm26", "zmm27", "zmm28", "zmm29", "zmm30", "zmm31", "memory", "k0", "k1", "k2", "k3", "k4");
}

static void NPACK_m12xk(float *A, float *Ac, int K, int LK)
{
    asm volatile(
        "   movl %[LK], %%r8d    \n"
        "   movl %[LK], %%r15d    \n"

        "   mov %[Ac], %%r9    \n"
        "   movl %[K], %%r10d   \n"
        "   mov %[A], %%r11    \n"
        "   mov %[A], %%r12    \n"
        "   mov %[A], %%r13    \n"
        "   mov %[A], %%r14    \n"

        "   mov %[A], %%rbx    \n"
        "   mov %[A], %%rcx    \n"
        "   mov %[A], %%rdx    \n"
        "   mov %[A], %%rdi    \n"

        "   shr $4, %%r10       \n"
        "   shl $2, %%r8       \n"
        "   shl $3, %%r15       \n"
        "   add %%r8, %%r14     \n"
        "   add %%r8, %%rdi     \n"
        "   add %%r8, %%r12     \n"
        "   add %%r8, %%rcx     \n"
        "   add %%r15, %%r13     \n"
        "   add %%r15, %%rdx     \n"
        "   shl $2, %%r8       \n"
        "   add %%r15, %%r14     \n"
        "   add %%r15, %%rdi     \n"
        "   add %%r8, %%rbx     \n"
        "   add %%r8, %%rcx     \n"
        "   add %%r8, %%rdx     \n"
        "   add %%r8, %%rdi     \n"

        "   shl $1, %%r15       \n"

        "   vmovups (%%r11), %%zmm2       \n"  // 512/16=32个K
        "   vmovups (%%r12), %%zmm3       \n"
        "   vmovups (%%r13), %%zmm4       \n"
        "   vmovups (%%r14), %%zmm5       \n"

        "   sub $64, %%r15              \n"

        "   vmovups (%%rbx), %%zmm6       \n"
        "   vmovups (%%rcx), %%zmm7       \n"
        "   vmovups (%%rdx), %%zmm8       \n"
        "   vmovups (%%rdi), %%zmm9       \n"

        "   mov %%r15, %%rsi              \n"

        "   jmp NPACK_12                   \n"

        "NPACK_Pre_12:              \n"
        "   add $64, %%r11      \n"
        "   add $64, %%r12      \n"
        "   add $64, %%r13      \n"
        "   add $64, %%r14      \n"
        "   sub %%rsi, %%rbx      \n"
        "   sub %%rsi, %%rcx      \n"
        "   sub %%rsi, %%rdx      \n"
        "   sub %%rsi, %%rdi      \n"

        "   add $768, %%r9      \n"

        "   vmovups (%%r11), %%zmm2       \n"
        "   vmovups (%%r12), %%zmm3       \n"
        "   vmovups (%%r13), %%zmm4       \n"
        "   vmovups (%%r14), %%zmm5       \n"
        "   vmovups (%%rbx), %%zmm6       \n"
        "   vmovups (%%rcx), %%zmm7       \n"
        "   vmovups (%%rdx), %%zmm8       \n"
        "   vmovups (%%rdi), %%zmm9       \n"
        "NPACK_12:                 \n"

        "   movl    $0xaa, %%eax       \n"
        "   movl    $0xcc, %%r15d         \n"

        "   kmovd   %%eax, %%k1            \n"
        "   kmovd   %%r15d, %%k2             \n"

        "   vunpcklps %%zmm3, %%zmm2, %%zmm0    \n"
        "   vunpcklps %%zmm7, %%zmm6, %%zmm1    \n"

        "   movl    $0x33, %%eax           \n"

        "   vunpcklps %%zmm5, %%zmm4, %%zmm26    \n"
        "   vunpcklps %%zmm9, %%zmm8, %%zmm27    \n"

        "   kmovd   %%eax, %%k3             \n"

        "   vpermq  $0x80, %%zmm26, %%zmm0%{%%k1%}      \n"
        "   vmovups %%zmm0, %%zmm28     \n"
        "   vpermq  $0x80, %%zmm27, %%zmm1%{%%k1%}      \n"
        "   vmovups %%zmm1, %%zmm29     \n"

        "   movl    $0x33, %%r15d           \n"

        "   vpermq  $0x40, %%zmm1, %%zmm0%{%%k2%}      \n"
        "   vpermq  $0x0e, %%zmm28, %%zmm29%{%%k3%}        \n"

        "   vmovups %%ymm0, (%%r9)         \n"
        "   vextractf64x4 $0x1, %%zmm0,  %%ymm30      \n"
        "   vmovups %%ymm30, 384(%%r9)            \n"

        "   vmovups %%ymm29, 192(%%r9)         \n"
        "   vextractf64x4 $0x1, %%zmm29,  %%ymm31      \n"
        "   vmovups %%ymm31, 576(%%r9)            \n"

        "   movl    $0x55, %%eax       \n"
        "   movl    $0xcc, %%r15d         \n"

        "   kmovd   %%eax, %%k1            \n"
        "   kmovd   %%r15d, %%k2             \n"

        "   vunpcklps %%zmm3, %%zmm2, %%zmm0    \n"
        "   vunpcklps %%zmm7, %%zmm6, %%zmm1    \n"

        "   movl    $0x33, %%eax           \n"

        "   vunpcklps %%zmm5, %%zmm4, %%zmm26    \n"
        "   vunpcklps %%zmm9, %%zmm8, %%zmm27    \n"

        "   kmovd   %%eax, %%k3             \n"

        "   vpermq  $0x31, %%zmm0, %%zmm26%{%%k1%}      \n"
        "   vmovups %%zmm26, %%zmm28     \n"
        "   vpermq  $0x31, %%zmm1, %%zmm27%{%%k1%}      \n"
        "   vmovups %%zmm27, %%zmm29     \n"

        "   vpermq  $0x40, %%zmm27, %%zmm26%{%%k2%}      \n"
        "   vpermq  $0x0e, %%zmm28, %%zmm29%{%%k3%}        \n"

        "   vmovups %%ymm26, 48(%%r9)         \n"
        "   vextractf64x4  $0x1,%%zmm26, %%ymm30      \n"
        "   vmovups %%ymm30, 432(%%r9)            \n"

        "   vmovups %%ymm29, 240(%%r9)         \n"
        "   vextractf64x4 $0x1, %%zmm29,  %%ymm31      \n"
        "   vmovups %%ymm31, 624(%%r9)            \n"

        "   movl    $0xaa, %%eax       \n"
        "   movl    $0xcc, %%r15d         \n"

        "   kmovd   %%eax, %%k1            \n"
        "   kmovd   %%r15d, %%k2             \n"

        "   vunpckhps %%zmm3, %%zmm2, %%zmm0    \n"
        "   vunpckhps %%zmm7, %%zmm6, %%zmm1    \n"

        "   movl    $0x33, %%eax           \n"

        "   vunpckhps %%zmm5, %%zmm4, %%zmm26    \n"
        "   vunpckhps %%zmm9, %%zmm8, %%zmm27    \n"

        "   kmovd   %%eax, %%k3             \n"

        "   vpermq   $0x80, %%zmm26, %%zmm0%{%%k1%}      \n"
        "   vmovups %%zmm0, %%zmm28     \n"
        "   vpermq   $0x80, %%zmm27, %%zmm1%{%%k1%}      \n"
        "   vmovups %%zmm1, %%zmm29     \n"

        "   vpermq  $0x40, %%zmm1, %%zmm0%{%%k2%}      \n"
        "   vpermq  $0x0e, %%zmm28, %%zmm29%{%%k3%}        \n"

        "   vmovups %%ymm0, 96(%%r9)         \n"
        "   vextractf64x4 $0x1, %%zmm0,  %%ymm30      \n"
        "   vmovups %%ymm30, 480(%%r9)            \n"

        "   vmovups %%ymm29, 288(%%r9)         \n"
        "   vextractf64x4 $0x1, %%zmm29,  %%ymm31      \n"
        "   vmovups %%ymm31, 672(%%r9)            \n"

        "   movl    $0x55, %%eax       \n"
        "   movl    $0xcc, %%r15d         \n"

        "   kmovd   %%eax, %%k1            \n"
        "   kmovd   %%r15d, %%k2             \n"

        "   vunpckhps %%zmm3, %%zmm2, %%zmm0    \n"
        "   vunpckhps %%zmm7, %%zmm6, %%zmm1    \n"

        "   movl    $0x33, %%eax           \n"

        "   vunpckhps %%zmm5, %%zmm4, %%zmm26    \n"
        "   vunpckhps %%zmm9, %%zmm8, %%zmm27    \n"

        "   kmovd   %%eax, %%k3             \n"

        "   vpermq  $0x31, %%zmm0, %%zmm26%{%%k1%}      \n"
        "   vmovups %%zmm26, %%zmm28     \n"
        "   vpermq  $0x31, %%zmm1, %%zmm27%{%%k1%}      \n"
        "   vmovups %%zmm27, %%zmm29     \n"

        "   vpermq  $0x40, %%zmm27, %%zmm26%{%%k2%}      \n"
        "   vpermq  $0x0e, %%zmm28, %%zmm29%{%%k3%}        \n"

        "   vmovups %%ymm26, 144(%%r9)         \n"
        "   vextractf64x4  $0x1,%%zmm26, %%ymm30      \n"
        "   vmovups %%ymm30, 528(%%r9)            \n"

        "   vmovups %%ymm29, 336(%%r9)         \n"
        "   vextractf64x4 $0x1, %%zmm29,  %%ymm31      \n"
        "   vmovups %%ymm31, 720(%%r9)            \n"

        "   add %%r8, %%rbx     \n"
        "   add %%r8, %%rcx     \n"
        "   add %%r8, %%rdx     \n"
        "   add %%r8, %%rdi     \n"

        "   vmovups (%%rbx), %%zmm2       \n"
        "   vmovups (%%rcx), %%zmm3       \n"
        "   vmovups (%%rdx), %%zmm4       \n"
        "   vmovups (%%rdi), %%zmm5       \n"

        "   movl    $0xaa, %%eax       \n"
        "   kmovd   %%eax, %%k1            \n"
        "   vunpcklps %%zmm3, %%zmm2, %%zmm0    \n"
        "   vunpcklps %%zmm5, %%zmm4, %%zmm26    \n"

        "   vpermq   $0x80, %%zmm26, %%zmm0%{%%k1%}      \n"
        "   vmovups %%xmm0, 32(%%r9)         \n"
        "   vextractf32x4  $0x1, %%zmm0, %%xmm28      \n"
        "   vextractf32x4  $0x2, %%zmm0, %%xmm29      \n"
        "   vextractf32x4  $0x3, %%zmm0, %%xmm30      \n"

        "   vmovups %%xmm28, 224(%%r9)         \n"
        "   vmovups %%xmm29, 416(%%r9)         \n"
        "   vmovups %%xmm30, 608(%%r9)         \n"

        "   movl    $0x55, %%eax       \n"
        "   kmovd   %%eax, %%k1            \n"

        "   vunpcklps %%zmm3, %%zmm2, %%zmm0    \n"
        "   vunpcklps %%zmm5, %%zmm4, %%zmm26    \n"

        "   vpermq  $0x31, %%zmm0, %%zmm26%{%%k1%}      \n"
        "   vmovups %%xmm26, 80(%%r9)         \n"
        "   vextractf32x4  $0x1, %%zmm26, %%xmm28      \n"
        "   vextractf32x4  $0x2, %%zmm26, %%xmm29      \n"
        "   vextractf32x4  $0x3, %%zmm26, %%xmm30      \n"

        "   vmovups %%xmm28, 272(%%r9)         \n"
        "   vmovups %%xmm29, 464(%%r9)         \n"
        "   vmovups %%xmm30, 656(%%r9)         \n"

        "   movl    $0xaa, %%eax       \n"
        "   kmovd   %%eax, %%k1            \n"
        "   vunpckhps %%zmm3, %%zmm2, %%zmm0    \n"
        "   vunpckhps %%zmm5, %%zmm4, %%zmm26    \n"
        "   vpermq   $0x80, %%zmm26, %%zmm0%{%%k1%}      \n"
        "   vmovups %%xmm0, 128(%%r9)         \n"
        "   vextractf32x4  $0x1, %%zmm0, %%xmm28      \n"
        "   vextractf32x4  $0x2, %%zmm0, %%xmm29      \n"
        "   vextractf32x4  $0x3, %%zmm0, %%xmm30      \n"

        "   vmovups %%xmm28, 320(%%r9)         \n"
        "   vmovups %%xmm29, 512(%%r9)         \n"
        "   vmovups %%xmm30, 704(%%r9)         \n"

        "   movl    $0x55, %%eax       \n"
        "   kmovd   %%eax, %%k1            \n"

        "   vunpckhps %%zmm3, %%zmm2, %%zmm0    \n"
        "   vunpckhps %%zmm5, %%zmm4, %%zmm26    \n"

        "   vpermq  $0x31, %%zmm0, %%zmm26%{%%k1%}      \n"
        "   vmovups %%xmm26, 176(%%r9)         \n"
        "   vextractf32x4  $0x1, %%zmm26, %%xmm28      \n"
        "   vextractf32x4  $0x2, %%zmm26, %%xmm29      \n"
        "   vextractf32x4  $0x3, %%zmm26, %%xmm30      \n"
        "   sub    $1, %%r10       \n"
        "   vmovups %%xmm28, 368(%%r9)         \n"
        "   vmovups %%xmm29, 560(%%r9)         \n"
        "   vmovups %%xmm30, 752(%%r9)         \n"

        "   je  NPACK_END_12           \n"
        "   jmp NPACK_Pre_12           \n"

        "NPACK_END_12:                         \n"
        :

        :
        [A] "m"(A),
        [Ac] "m"(Ac),
        [K] "m"(K),
        [LK] "m"(LK)

        : "rax", "rbx", "rcx", "rdx", "rdi", "rsi", "r8", "r9", "r10", "r11", "r12", "r13", "r14", "r15", "zmm0", "zmm1", "zmm2", "zmm3", "zmm4", "zmm5", "zmm6", "zmm7", "zmm8", "zmm9", "zmm10", "zmm11", "zmm12", "zmm13", "zmm14", "zmm15", "zmm16", "zmm17", "zmm18", "zmm19", "zmm20", "zmm21", "zmm22", "zmm23", "zmm24", "zmm25", "zmm26", "zmm27", "zmm28", "zmm29", "zmm30", "zmm31", "memory", "k0", "k1", "k2", "k3", "k4");
}

static void FLASHGEMM_NPACK(float *A, float *Ac, int M, int K, int LK)
{
    float *temp_A = A;
    float *temp_Ac = Ac;

    if (M == 12)
    {
        // printf("M = %d,pack m12\n",M);
        NPACK_m12xk(temp_A, temp_Ac, K, LK);
        return;
    }
    if (M >= 8)
    {
        // printf("M = %d,pack m8\n",M);
        NPACK_m8xk(temp_A, temp_Ac, K, LK);
        temp_A = temp_A + 8 * LK;
        temp_Ac = temp_Ac + 8 * K;
        M = M - 8;
    }
    if (M >= 4)
    {
        // printf("M = %d,pack m4\n",M);
        NPACK_m4xk(temp_A, temp_Ac, K, LK);
        temp_A = temp_A + 4 * LK;
        temp_Ac = temp_Ac + 4 * K;
        M = M - 4;
    }
    if (M >= 1)
    {
        // printf("M = %d,pack m1\n",M);
        NPACK_m1xk(temp_A, temp_Ac, K, LK);
        temp_A = temp_A + 1 * LK;
        temp_Ac = temp_Ac + 1 * K;
        M = M - 1;
    }
    if (M >= 1)
    {
        // printf("M = %d,pack m1\n",M);
        NPACK_m1xk(temp_A, temp_Ac, K, LK);
        temp_A = temp_A + 1 * LK;
        temp_Ac = temp_Ac + 1 * K;
        M = M - 1;
    }
    if (M >= 1)
    {
        // printf("M = %d,pack m1\n",M);
        NPACK_m1xk(temp_A, temp_Ac, K, LK);
    }
}
