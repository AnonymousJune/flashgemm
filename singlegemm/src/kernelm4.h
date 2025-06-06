void SMM_NN_KERNELm4xn64(float *C, float *Ac, float *B, long M, long N, long K, long LN, long LK, long k_tag);

void SMM_NN_KERNELm4xn64(float *C, float *Ac, float *B, long M, long N, long K, long LN, long LK, long k_tag)
{
    asm volatile(
        // 共4x64/4x16两种核，仅适用于M=4的情况
        // Ac已打包(zmm0-7)，B不打包(zmm8-15)，C(zmm16-31)
        // Bc, %%r14, %%rbp没有用了
        // rbx: 存放当前k行开始的B指针
        // rbp: 存放预取k行的B指针
        // r9: 存放当前[kc,nr]B块开始的指针
        //-----------------------------------------------------------------

        ".macro    KERNEL4x64_K1_MLN                                 \n"
        "    addq              $16, %%rax                            \n"
        "    leaq           (%%rbx, %%r8, 4), %%rbx                  \n"
        "    leaq           (%%rbp, %%r8, 4), %%rbp                  \n"

        "    vfmadd231ps        %%zmm0, %%zmm8, %%zmm16              \n"
        "    vfmadd231ps        %%zmm0, %%zmm9, %%zmm17              \n"
        "    vfmadd231ps        %%zmm0, %%zmm10, %%zmm18             \n"
        "    vfmadd231ps        %%zmm0, %%zmm11, %%zmm19             \n"
        "    vbroadcastss    (%%rax), %%zmm4                         \n"
        "    vmovups            (%%rbx), %%zmm12                     \n"
        "    prefetcht1      (%%rbp)                                 \n"
        "    prefetcht2     256(%%rbx)                               \n"

        "    vfmadd231ps        %%zmm1, %%zmm8, %%zmm20              \n"
        "    vfmadd231ps        %%zmm1, %%zmm9, %%zmm21              \n"
        "    vfmadd231ps        %%zmm1, %%zmm10, %%zmm22             \n"
        "    vfmadd231ps        %%zmm1, %%zmm11, %%zmm23             \n"
        "    vbroadcastss    4(%%rax), %%zmm5                        \n"
        "    vmovups            64(%%rbx), %%zmm13                   \n"
        "    prefetcht1      64(%%rbp)                               \n"
        "    prefetcht2     320(%%rbx)                               \n"

        "    vfmadd231ps        %%zmm2, %%zmm8, %%zmm24              \n"
        "    vfmadd231ps        %%zmm2, %%zmm9, %%zmm25              \n"
        "    vfmadd231ps        %%zmm2, %%zmm10, %%zmm26             \n"
        "    vfmadd231ps        %%zmm2, %%zmm11, %%zmm27             \n"
        "    vbroadcastss    8(%%rax), %%zmm6                        \n"
        "    vmovups            128(%%rbx), %%zmm14                  \n"
        "    prefetcht1      128(%%rbp)                              \n"
        "    prefetcht2     384(%%rbx)                               \n"

        "    vfmadd231ps        %%zmm3, %%zmm8, %%zmm28              \n"
        "    vfmadd231ps        %%zmm3, %%zmm9, %%zmm29              \n"
        "    vfmadd231ps        %%zmm3, %%zmm10, %%zmm30             \n"
        "    vfmadd231ps        %%zmm3, %%zmm11, %%zmm31             \n"
        "    vbroadcastss    12(%%rax), %%zmm7                       \n"
        "    vmovups            192(%%rbx), %%zmm15                  \n"
        "    prefetcht1      192(%%rbp)                              \n"
        "    prefetcht2     448(%%rbx)                               \n"

        ".endm                                                       \n"

        ".macro    KERNEL4x64_K2_MLN                                 \n"
        "    addq              $16, %%rax                            \n"
        "    leaq           (%%rbx, %%r8, 4), %%rbx                  \n"
        "    leaq           (%%rbp, %%r8, 4), %%rbp                  \n"

        "    vfmadd231ps        %%zmm4, %%zmm12, %%zmm16             \n"
        "    vfmadd231ps        %%zmm4, %%zmm13, %%zmm17             \n"
        "    vfmadd231ps        %%zmm4, %%zmm14, %%zmm18             \n"
        "    vfmadd231ps        %%zmm4, %%zmm15, %%zmm19             \n"
        "    vbroadcastss    (%%rax), %%zmm0                         \n"
        "    vmovups            (%%rbx), %%zmm8                      \n"
        "    prefetcht1      (%%rbp)                                 \n"
        "    prefetcht2     256(%%rbx)                               \n"

        "    vfmadd231ps        %%zmm5, %%zmm12, %%zmm20             \n"
        "    vfmadd231ps        %%zmm5, %%zmm13, %%zmm21             \n"
        "    vfmadd231ps        %%zmm5, %%zmm14, %%zmm22             \n"
        "    vfmadd231ps        %%zmm5, %%zmm15, %%zmm23             \n"
        "    vbroadcastss    4(%%rax), %%zmm1                        \n"
        "    vmovups            64(%%rbx), %%zmm9                    \n"
        "    prefetcht1      64(%%rbp)                               \n"
        "    prefetcht2     320(%%rbx)                               \n"

        "    vfmadd231ps        %%zmm6, %%zmm12, %%zmm24             \n"
        "    vfmadd231ps        %%zmm6, %%zmm13, %%zmm25             \n"
        "    vfmadd231ps        %%zmm6, %%zmm14, %%zmm26             \n"
        "    vfmadd231ps        %%zmm6, %%zmm15, %%zmm27             \n"
        "    vbroadcastss    8(%%rax), %%zmm2                        \n"
        "    vmovups            128(%%rbx), %%zmm10                  \n"
        "    prefetcht1      128(%%rbp)                              \n"
        "    prefetcht2     384(%%rbx)                               \n"

        "    vfmadd231ps        %%zmm7, %%zmm12, %%zmm28             \n"
        "    vfmadd231ps        %%zmm7, %%zmm13, %%zmm29             \n"
        "    vfmadd231ps        %%zmm7, %%zmm14, %%zmm30             \n"
        "    vfmadd231ps        %%zmm7, %%zmm15, %%zmm31             \n"
        "    vbroadcastss    12(%%rax), %%zmm3                       \n"
        "    vmovups            192(%%rbx), %%zmm11                  \n"
        "    prefetcht1      192(%%rbp)                              \n"
        "    prefetcht2     448(%%rbx)                               \n"

        ".endm                                                       \n"

        ".macro    KERNEL4x64_END_K_MLN                              \n"

        "    vfmadd231ps        %%zmm4, %%zmm12, %%zmm16             \n"
        "    vfmadd231ps        %%zmm4, %%zmm13, %%zmm17             \n"
        "    vfmadd231ps        %%zmm4, %%zmm14, %%zmm18             \n"
        "    vfmadd231ps        %%zmm4, %%zmm15, %%zmm19             \n"

        "    vfmadd231ps        %%zmm5, %%zmm12, %%zmm20             \n"
        "    vfmadd231ps        %%zmm5, %%zmm13, %%zmm21             \n"
        "    vfmadd231ps        %%zmm5, %%zmm14, %%zmm22             \n"
        "    vfmadd231ps        %%zmm5, %%zmm15, %%zmm23             \n"

        "    vfmadd231ps        %%zmm6, %%zmm12, %%zmm24             \n"
        "    vfmadd231ps        %%zmm6, %%zmm13, %%zmm25             \n"
        "    vfmadd231ps        %%zmm6, %%zmm14, %%zmm26             \n"
        "    vfmadd231ps        %%zmm6, %%zmm15, %%zmm27             \n"

        "    vfmadd231ps        %%zmm7, %%zmm12, %%zmm28             \n"
        "    vfmadd231ps        %%zmm7, %%zmm13, %%zmm29             \n"
        "    vfmadd231ps        %%zmm7, %%zmm14, %%zmm30             \n"
        "    vfmadd231ps        %%zmm7, %%zmm15, %%zmm31             \n"

        ".endm                                                       \n"

        ".macro    ADD_C_4x64_MLN                                    \n"

        "    vmovups         (%%r10), %%zmm0                         \n"
        "    vaddps             %%zmm0, %%zmm16, %%zmm16             \n"
        "    vmovups         64(%%r10), %%zmm1                       \n"
        "    vaddps             %%zmm1, %%zmm17, %%zmm17             \n"
        "    vmovups         128(%%r10), %%zmm2                      \n"
        "    vaddps             %%zmm2, %%zmm18, %%zmm18             \n"
        "    vmovups         192(%%r10), %%zmm3                      \n"
        "    vaddps             %%zmm3, %%zmm19, %%zmm19             \n"

        "    vmovups         (%%r11), %%zmm4                         \n"
        "    vaddps             %%zmm4, %%zmm20, %%zmm20             \n"
        "    vmovups         64(%%r11), %%zmm5                       \n"
        "    vaddps             %%zmm5, %%zmm21, %%zmm21             \n"
        "    vmovups         128(%%r11), %%zmm6                      \n"
        "    vaddps             %%zmm6, %%zmm22, %%zmm22             \n"
        "    vmovups         192(%%r11), %%zmm7                      \n"
        "    vaddps             %%zmm7, %%zmm23, %%zmm23             \n"

        "    vmovups         (%%r12), %%zmm8                         \n"
        "    vaddps             %%zmm8, %%zmm24, %%zmm24             \n"
        "    vmovups         64(%%r12), %%zmm9                       \n"
        "    vaddps             %%zmm9, %%zmm25, %%zmm25             \n"
        "    vmovups         128(%%r12), %%zmm10                     \n"
        "    vaddps             %%zmm10, %%zmm26, %%zmm26            \n"
        "    vmovups         192(%%r12), %%zmm11                     \n"
        "    vaddps             %%zmm11, %%zmm27, %%zmm27            \n"

        "    vmovups         (%%r13), %%zmm12                        \n"
        "    vaddps             %%zmm12, %%zmm28, %%zmm28            \n"
        "    vmovups         64(%%r13), %%zmm13                      \n"
        "    vaddps             %%zmm13, %%zmm29, %%zmm29            \n"
        "    vmovups         128(%%r13), %%zmm14                     \n"
        "    vaddps             %%zmm14, %%zmm30, %%zmm30            \n"
        "    vmovups         192(%%r13), %%zmm15                     \n"
        "    vaddps             %%zmm15, %%zmm31, %%zmm31            \n"

        ".endm                                                       \n"

        ".macro    SAVE_4x64_MLN                                     \n"

        "    vmovups         %%zmm16, (%%r10)                        \n"
        "    vmovups         %%zmm17, 64(%%r10)                      \n"
        "    vmovups         %%zmm18, 128(%%r10)                     \n"
        "    vmovups         %%zmm19, 192(%%r10)                     \n"

        "    vmovups         %%zmm20, (%%r11)                        \n"
        "    vmovups         %%zmm21, 64(%%r11)                      \n"
        "    vmovups         %%zmm22, 128(%%r11)                     \n"
        "    vmovups         %%zmm23, 192(%%r11)                     \n"

        "    vmovups         %%zmm24, (%%r12)                        \n"
        "    vmovups         %%zmm25, 64(%%r12)                      \n"
        "    vmovups         %%zmm26, 128(%%r12)                     \n"
        "    vmovups         %%zmm27, 192(%%r12)                     \n"

        "    vmovups         %%zmm28, (%%r13)                        \n"
        "    vmovups         %%zmm29, 64(%%r13)                      \n"
        "    vmovups         %%zmm30, 128(%%r13)                     \n"
        "    vmovups         %%zmm31, 192(%%r13)                     \n"

        "    subq            $64, %%rdi                              \n" // TODO
        "    addq            $256, %%rcx                             \n" // C0
        "    addq              $256, %%r9                            \n" // TODO

        ".endm                                                       \n"

        //-----------------------------------------------------------------

        ".macro    KERNEL4x16_K1_MLN                                 \n"

        "    addq              $16, %%rax                            \n"
        "    leaq           (%%rbx, %%r8, 4), %%rbx                  \n"
        "    leaq           (%%rbp, %%r8, 4), %%rbp                  \n"

        "    vfmadd231ps        %%zmm0, %%zmm8, %%zmm16              \n"
        "    vbroadcastss    (%%rax), %%zmm4                         \n"
        "    vmovups         (%%rbx), %%zmm12                        \n"
        "    prefetcht1        (%%rbp)                               \n"
        "    prefetcht2        256(%%rbx)                            \n"

        "    vfmadd231ps        %%zmm1, %%zmm8, %%zmm20              \n"
        "    vbroadcastss    4(%%rax), %%zmm5                        \n"

        "    vfmadd231ps        %%zmm2, %%zmm8, %%zmm24              \n"
        "    vbroadcastss    8(%%rax), %%zmm6                        \n"

        "    vfmadd231ps        %%zmm3, %%zmm8, %%zmm28              \n"
        "    vbroadcastss    12(%%rax), %%zmm7                       \n"

        ".endm                                                       \n"

        ".macro    KERNEL4x16_K2_MLN                                 \n"

        "    addq              $16, %%rax                            \n"
        "    leaq           (%%rbx, %%r8, 4), %%rbx                  \n"
        "    leaq           (%%rbp, %%r8, 4), %%rbp                  \n"

        "    vfmadd231ps        %%zmm4, %%zmm12, %%zmm16             \n"
        "    vbroadcastss    (%%rax), %%zmm0                         \n"
        "    vmovups         (%%rbx), %%zmm12                        \n"
        "    prefetcht1        (%%rbp)                               \n"
        "    prefetcht2        256(%%rbx)                            \n"

        "    vfmadd231ps        %%zmm5, %%zmm12, %%zmm20             \n"
        "    vbroadcastss    4(%%rax), %%zmm1                        \n"

        "    vfmadd231ps        %%zmm6, %%zmm12, %%zmm24             \n"
        "    vbroadcastss    8(%%rax), %%zmm2                        \n"

        "    vfmadd231ps        %%zmm7, %%zmm12, %%zmm28             \n"
        "    vbroadcastss    12(%%rax), %%zmm3                       \n"

        ".endm                                                       \n"

        ".macro    KERNEL4x16_END_K_MLN                              \n"

        "    vfmadd231ps        %%zmm4, %%zmm12, %%zmm16             \n"
        "    vfmadd231ps        %%zmm5, %%zmm12, %%zmm20             \n"
        "    vfmadd231ps        %%zmm6, %%zmm12, %%zmm24             \n"
        "    vfmadd231ps        %%zmm7, %%zmm12, %%zmm28             \n"

        ".endm                                                       \n"

        ".macro    ADD_C_4x16_MLN                                    \n"

        "    vmovups         (%%r10), %%zmm0                         \n"
        "    vaddps             %%zmm0, %%zmm16, %%zmm16             \n"

        "    vmovups         (%%r11), %%zmm4                         \n"
        "    vaddps             %%zmm4, %%zmm20, %%zmm20             \n"

        "    vmovups         (%%r12), %%zmm8                         \n"
        "    vaddps             %%zmm8, %%zmm24, %%zmm24             \n"

        "    vmovups         (%%r13), %%zmm12                        \n"
        "    vaddps             %%zmm12, %%zmm28, %%zmm28            \n"

        ".endm                                                       \n"

        ".macro    SAVE_4x16_MLN                                     \n"

        "    vmovups         %%zmm16, (%%r10)                        \n"
        "    vmovups         %%zmm20, (%%r11)                        \n"
        "    vmovups         %%zmm24, (%%r12)                        \n"
        "    vmovups         %%zmm28, (%%r13)                        \n"

        "    subq             $16, %%rdi                             \n"
        "    addq                 $64, %%rcx                         \n" // C0
        "    addq              $64, %%r9                             \n" // TODO
        ".endm                                                       \n"

        //-----------------------------------------------------------------

        "SMM_KERNEL4x64_MLN:                                         \n"

        "    mov     %[C], %%rcx                                     \n"
        "    mov     %[Ac], %%rax                                    \n"
        "    mov     %[B], %%rbx                                     \n"

        "    prefetcht0         (%%rax)                              \n"
        "    prefetcht0         (%%rbx)                              \n"

        "    mov     %[K], %%rdx                                     \n" // K(kc)
        "    mov      %[LN], %%r8                                    \n"
        "    mov      %%rax, %%r14                                   \n" // 存储Ac地址
        "    movq    %[N], %%rdi                                     \n"
        "    mov     %[k_tag], %%r15                                 \n" // kk=0把C存回内存, 否则加回对应的C位置

        "    mov   %%r8, %%rsi                                       \n"
        "    shl   $5, %%rsi                                         \n" // TODO B预取的间隔
        "    mov     %%rbx, %%r9                                     \n" // B备份

        //-----------------------------------------------------------------

        "BEGIN_M4N64_MLN:                                            \n"
        "    cmpq     $64, %%rdi                                     \n"
        "    jb          BEGIN_M4N16_MLN                             \n"
        "    mov        %[K], %%rdx                                  \n" // K

        "    mov   %%r9, %%rbx                                       \n" // B
        "    vmovups            (%%rbx), %%zmm8                      \n" // B0
        "    vmovups         64(%%rbx), %%zmm9                       \n" // B1
        "    vmovups         128(%%rbx), %%zmm10                     \n" // B2
        "    vmovups         192(%%rbx), %%zmm11                     \n" // B3

        "    mov     %%r14, %%rax                                    \n" // Ac
        "    vbroadcastss    (%%rax), %%zmm0                         \n" // A0
        "    vbroadcastss    4(%%rax), %%zmm1                        \n" // A1
        "    vbroadcastss    8(%%rax), %%zmm2                        \n" // A2
        "    vbroadcastss    12(%%rax), %%zmm3                       \n" // A3

        "    mov   %%r9, %%rbp                                       \n"
        "    addq  %%rsi, %%rbp                                      \n" // B prefetch
        "    prefetcht1       (%%rbp)                                \n"
        "    prefetcht1       64(%%rbp)                              \n"
        "    prefetcht1       128(%%rbp)                             \n"
        "    prefetcht1       192(%%rbp)                             \n"

        "    vpxorq             %%zmm16, %%zmm16, %%zmm16            \n"
        "    vpxorq             %%zmm17, %%zmm17, %%zmm17            \n"
        "    vpxorq             %%zmm18, %%zmm18, %%zmm18            \n"
        "    vpxorq             %%zmm19, %%zmm19, %%zmm19            \n"
        "    vpxorq             %%zmm20, %%zmm20, %%zmm20            \n"
        "    vpxorq             %%zmm21, %%zmm21, %%zmm21            \n"
        "    vpxorq             %%zmm22, %%zmm22, %%zmm22            \n"
        "    vpxorq             %%zmm23, %%zmm23, %%zmm23            \n"
        "    vpxorq             %%zmm24, %%zmm24, %%zmm24            \n"
        "    vpxorq             %%zmm25, %%zmm25, %%zmm25            \n"
        "    vpxorq             %%zmm26, %%zmm26, %%zmm26            \n"
        "    vpxorq             %%zmm27, %%zmm27, %%zmm27            \n"
        "    vpxorq             %%zmm28, %%zmm28, %%zmm28            \n"
        "    vpxorq             %%zmm29, %%zmm29, %%zmm29            \n"
        "    vpxorq             %%zmm30, %%zmm30, %%zmm30            \n"
        "    vpxorq             %%zmm31, %%zmm31, %%zmm31            \n"

        "    mov      %%rcx, %%r10                                   \n" // C0
        "    prefetcht2         (%%r10)                              \n"
        "    prefetcht2         64(%%r10)                            \n"
        "    prefetcht2         128(%%r10)                           \n"
        "    prefetcht2         192(%%r10)                           \n"
        "    leaq     (%%r10, %%r8, 4), %%r11                        \n" // C1
        "    prefetcht2         (%%r11)                              \n"
        "    prefetcht2         64(%%r11)                            \n"
        "    prefetcht2         128(%%r11)                           \n"
        "    prefetcht2         192(%%r11)                           \n"
        "    leaq     (%%r11, %%r8, 4), %%r12                        \n" // C2
        "    prefetcht2         (%%r12)                              \n"
        "    prefetcht2         64(%%r12)                            \n"
        "    prefetcht2         128(%%r12)                           \n"
        "    prefetcht2         192(%%r12)                           \n"
        "    leaq     (%%r12, %%r8, 4), %%r13                        \n" // C3
        "    prefetcht2         (%%r13)                              \n"
        "    prefetcht2         64(%%r13)                            \n"
        "    prefetcht2         128(%%r13)                           \n"
        "    prefetcht2         192(%%r13)                           \n"

        "    subq     $8, %%rdx                                      \n"

        "MAIN_K_M4N64_MLN:                                           \n"

        "    KERNEL4x64_K1_MLN                                       \n"
        "    KERNEL4x64_K2_MLN                                       \n"
        "    KERNEL4x64_K1_MLN                                       \n"
        "    KERNEL4x64_K2_MLN                                       \n"
        "    KERNEL4x64_K1_MLN                                       \n"
        "    KERNEL4x64_K2_MLN                                       \n"
        "    KERNEL4x64_K1_MLN                                       \n"
        "    cmp     $0, %%rdx                                       \n"
        "    je         EDGE_K_M4N64_MLN                             \n"
        "    KERNEL4x64_K2_MLN                                       \n"
        "    subq     $8, %%rdx                                      \n"
        "    jmp     MAIN_K_M4N64_MLN                                \n"

        "EDGE_K_M4N64_MLN:                                           \n"
        "    KERNEL4x64_END_K_MLN                                    \n"

        "BEGIN_SAVE_M4N64_MLN:                                       \n"
        "    cmp     $0, %%r15                                       \n"
        "    je      SAVE_C_M4N64_MLN                                \n"
        "    ADD_C_4x64_MLN                                          \n"

        "SAVE_C_M4N64_MLN:                                           \n"
        "    SAVE_4x64_MLN                                           \n"
        "    cmpq      $64, %%rdi                                    \n" // TODO
        "    jnb     BEGIN_M4N64_MLN                                 \n" // 不小于（或等于）则跳转

        //----------------------------------------------------------------

        "BEGIN_M4N16_MLN:                                            \n"
        "    cmpq        $16, %%rdi                                  \n"
        "    jb            END_M4N64_MLN                             \n"
        "    mov            %[K], %%rdx                              \n" // K

        "    mov            %%r9, %%rbx                              \n" // B
        "    vmovups            (%%rbx), %%zmm8                      \n" // B0

        "    mov     %%r14, %%rax                                    \n" // Ac
        "    vbroadcastss    (%%rax), %%zmm0                         \n" // A0
        "    vbroadcastss    4(%%rax), %%zmm1                        \n" // A1
        "    vbroadcastss    8(%%rax), %%zmm2                        \n" // A2
        "    vbroadcastss    12(%%rax), %%zmm3                       \n" // A3

        "    mov   %%r9, %%rbp                                       \n"
        "    addq  %%rsi, %%rbp                                      \n" // B prefetch
        "    prefetcht1       (%%rbp)                                \n"

        "    vpxorq             %%zmm16, %%zmm16, %%zmm16            \n"
        "    vpxorq             %%zmm17, %%zmm17, %%zmm17            \n"
        "    vpxorq             %%zmm18, %%zmm18, %%zmm18            \n"
        "    vpxorq             %%zmm19, %%zmm19, %%zmm19            \n"
        "    vpxorq             %%zmm20, %%zmm20, %%zmm20            \n"
        "    vpxorq             %%zmm21, %%zmm21, %%zmm21            \n"
        "    vpxorq             %%zmm22, %%zmm22, %%zmm22            \n"
        "    vpxorq             %%zmm23, %%zmm23, %%zmm23            \n"
        "    vpxorq             %%zmm24, %%zmm24, %%zmm24            \n"
        "    vpxorq             %%zmm25, %%zmm25, %%zmm25            \n"
        "    vpxorq             %%zmm26, %%zmm26, %%zmm26            \n"
        "    vpxorq             %%zmm27, %%zmm27, %%zmm27            \n"
        "    vpxorq             %%zmm28, %%zmm28, %%zmm28            \n"
        "    vpxorq             %%zmm29, %%zmm29, %%zmm29            \n"
        "    vpxorq             %%zmm30, %%zmm30, %%zmm30            \n"
        "    vpxorq             %%zmm31, %%zmm31, %%zmm31            \n"

        "    mov      %%rcx, %%r10                                   \n" // C0
        "    prefetcht2         (%%r10)                              \n"
        "    leaq     (%%r10, %%r8, 4), %%r11                        \n" // C1
        "    prefetcht2         (%%r11)                              \n"
        "    leaq     (%%r11, %%r8, 4), %%r12                        \n" // C2
        "    prefetcht2         (%%r12)                              \n"
        "    leaq     (%%r12, %%r8, 4), %%r13                        \n" // C3
        "    prefetcht2         (%%r13)                              \n"

        "    subq     $8, %%rdx                                      \n"

        "MAIN_K_M4N16_MLN:                                           \n"
        "    KERNEL4x16_K1_MLN                                       \n"
        "    KERNEL4x16_K2_MLN                                       \n"
        "    KERNEL4x16_K1_MLN                                       \n"
        "    KERNEL4x16_K2_MLN                                       \n"
        "    KERNEL4x16_K1_MLN                                       \n"
        "    KERNEL4x16_K2_MLN                                       \n"
        "    KERNEL4x16_K1_MLN                                       \n"
        "   cmp     $0, %%rdx                                        \n"
        "    je         EDGE_K_M4N16_MLN                             \n"
        "    KERNEL4x16_K2_MLN                                       \n"
        "   subq     $8, %%rdx                                       \n"
        "   jmp   MAIN_K_M4N16_MLN                                   \n"

        "EDGE_K_M4N16_MLN:                                           \n"
        "    KERNEL4x16_END_K_MLN                                    \n"
        "    cmp     $0, %%r15                                       \n"
        "    je      SAVE_C_M4N16_MLN                                \n"
        "    ADD_C_4x16_MLN                                          \n"

        "SAVE_C_M4N16_MLN:                                           \n"
        "    SAVE_4x16_MLN                                           \n"

        //----------------------------------------------------------------

        "END_M4N64_MLN:                                              \n"

        :
        :
        [C] "m"(C),
        [Ac] "m"(Ac),
        [B] "m"(B),
        [M] "m"(M),
        [N] "m"(N),
        [K] "m"(K),
        [LN] "m"(LN),
        [LK] "m"(LK),
        [k_tag] "m"(k_tag)
        : "rax", "rbx", "rcx", "rdx", "rdi", "rsi", "rbp", "r8", "r9", "r10", "r11", "r12",
          "r13", "r14", "r15", "xmm0", "xmm1", "xmm2", "xmm3", "xmm4", "xmm5",
          "xmm6", "xmm7", "xmm8", "xmm9", "xmm10", "xmm11", "xmm12", "xmm13",
          "xmm14", "xmm15", "xmm16", "xmm17", "xmm18", "xmm19", "xmm20", "xmm21",
          "xmm22", "xmm23", "xmm24", "xmm25", "xmm26", "xmm27", "xmm28", "xmm29",
          "xmm30", "xmm31", "memory"

    );
}
