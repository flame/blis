/*
   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2022-2023, Advanced Micro Devices, Inc. All rights reserved.

   Redistribution and use in source and binary forms, with or without
   modification, are permitted provided that the following conditions are
   met:
    - Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.
    - Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.
    - Neither the name(s) of the copyright holder(s) nor the names of its
      contributors may be used to endorse or promote products derived
      from this software without specific prior written permission.
   THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
   "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
   LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
   A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
   HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
   SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
   LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
   DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
   THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
   (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
   OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/
#include "blis.h"
#define BLIS_ASM_SYNTAX_ATT
#include "bli_x86_asm_macros.h"
/*
   rrc:
     --------        ------        | | | | | | | |
     --------        ------        | | | | | | | |
     --------   +=   ------ ...    | | | | | | | |
     --------        ------        | | | | | | | |
     --------        ------              :
     --------        ------              :
   Assumptions:
   - C is row-stored and B is column-stored;
   - A is row-stored;
   - m0 and n0 are at most MR and NR, respectively.
   Therefore, this (r)ow-preferential microkernel is well-suited for
   a dot-product-based accumulation that performs vector loads from
   both A and B.
*/

void bli_zgemmsup_rd_zen_asm_2x4
     (
       conj_t       conja,
       conj_t       conjb,
       dim_t        m0,
       dim_t        n0,
       dim_t        k0,
       dcomplex*    restrict alpha,
       dcomplex*    restrict a, inc_t rs_a0, inc_t cs_a0,
       dcomplex*    restrict b, inc_t rs_b0, inc_t cs_b0,
       dcomplex*    restrict beta,
       dcomplex*    restrict c, inc_t rs_c0, inc_t cs_c0,
       auxinfo_t* restrict data,
       cntx_t*    restrict cntx
     )
{
    uint64_t k_iter8 = k0 / 8;
    uint64_t k_left8 = k0 % 8;
    uint64_t k_iter4 = k_left8 / 4;
    uint64_t k_left4 = k_left8 % 4;

    uint64_t rs_a   = rs_a0;
    uint64_t cs_b   = cs_b0;
    uint64_t rs_c   = rs_c0;

    // Checking whether generic/special case handling is required for beta scaling
    char beta_mul_type = (beta->real == 0.0 && beta->imag == 0.0)? BLIS_MUL_ZERO : BLIS_MUL_DEFAULT;

    //-----------------------------------------------------------//
    // Inline assembly implementation

    begin_asm()
    mov(var(rs_a), r8)             // load rs_a
    lea(mem(, r8, 8), r8)
    lea(mem(, r8, 2), r8)          // r8 = sizeof(dcomplex)*rs_a

    mov(var(cs_b), r10)
    lea(mem(, r10, 8), r10)
    lea(mem(, r10, 2), r10)       // r10 = sizeof(dcomplex)*cs_b

    mov(var(rs_c), rdi)
    lea(mem(, rdi, 8), rdi)
    lea(mem(, rdi, 2), rdi)       // rdi = sizeof(dcomplex)*rs_c

    mov(imm(0), r15)               // jj = 0
    label(.ZLOOP3X4J)              // LOOP OVER jj = [ 0 1 ... ]
    mov(var(a), r14)               // r14 = addr of a
    mov(var(b), r11)               // r11 = addr of b
    mov(var(c), r12)               // r12 = addr of c

    lea(mem(, r15, 1), rsi)
    imul(imm(1*16), rsi)            // rsi = 16*jj
    lea(mem(r12, rsi, 1), r12)    // r12 += 16*jj

    lea(mem(, r15, 1), rsi)
    imul(r10, rsi)                 // rsi = 16*jj
    lea(mem(r11, rsi, 1), r11)    // r12 += cs_b*jj

    vzeroall()                      // Reset all ymm registers
    mov(r12, rcx)                  // rcx = c_iijj;
    mov(r11, rbx)                  // rbx = b_jj;
    mov(r14, rax)                  // rax = a_ii;

    mov(var(k_iter8), rsi)        // i = k_iter8;
    test(rsi, rsi)                 // Check i via logical AND
    je(.ZLOOPKLEFT8)               // If i=0 jmp to k_iter4 loop

    label(.ZLOOPKITER8)            // MAIN LOOP

    // ---------------------------------- Iteration 0
    vmovupd(mem(rax), ymm0)
    vmovupd(mem(rax, r8, 1), ymm1)
    add(imm(2*16), rax)            // a += 2*sizeof(dcomplex)*cs_a = 2*16;

    vmovupd(mem(rbx), ymm3)
    vfmadd231pd(ymm0, ymm3, ymm4)
    vfmadd231pd(ymm1, ymm3, ymm5)

    vpermilpd(imm(0x5), ymm3, ymm3)
    vfmadd231pd(ymm0, ymm3, ymm7)
    vfmadd231pd(ymm1, ymm3, ymm8)

    vmovupd(mem(rbx, r10, 1), ymm3)
    vfmadd231pd(ymm0, ymm3, ymm10)
    vfmadd231pd(ymm1, ymm3, ymm11)

    vpermilpd(imm(0x5), ymm3, ymm3)
    vfmadd231pd(ymm0, ymm3, ymm13)
    vfmadd231pd(ymm1, ymm3, ymm14)
    add(imm(2*16), rbx)          // b += 2*sizeof(dcomplex)*rs_b = 2*16;

    // ---------------------------------- Iteration 1
    vmovupd(mem(rax), ymm0)
    vmovupd(mem(rax, r8, 1), ymm1)
    add(imm(2*16), rax)            // a += 2*sizeof(dcomplex)*cs_a = 2*16;

    vmovupd(mem(rbx), ymm3)
    vfmadd231pd(ymm0, ymm3, ymm4)
    vfmadd231pd(ymm1, ymm3, ymm5)

    vpermilpd(imm(0x5), ymm3, ymm3)
    vfmadd231pd(ymm0, ymm3, ymm7)
    vfmadd231pd(ymm1, ymm3, ymm8)

    vmovupd(mem(rbx, r10, 1), ymm3)
    vfmadd231pd(ymm0, ymm3, ymm10)
    vfmadd231pd(ymm1, ymm3, ymm11)

    vpermilpd(imm(0x5), ymm3, ymm3)
    vfmadd231pd(ymm0, ymm3, ymm13)
    vfmadd231pd(ymm1, ymm3, ymm14)
    add(imm(2*16), rbx)          // b += 2*sizeof(dcomplex)*rs_b = 2*16;

    // ---------------------------------- Iteration 2
    vmovupd(mem(rax), ymm0)
    vmovupd(mem(rax, r8, 1), ymm1)
    add(imm(2*16), rax)            // a += 2*sizeof(dcomplex)*cs_a = 2*16;

    vmovupd(mem(rbx), ymm3)
    vfmadd231pd(ymm0, ymm3, ymm4)
    vfmadd231pd(ymm1, ymm3, ymm5)

    vpermilpd(imm(0x5), ymm3, ymm3)
    vfmadd231pd(ymm0, ymm3, ymm7)
    vfmadd231pd(ymm1, ymm3, ymm8)

    vmovupd(mem(rbx, r10, 1), ymm3)
    vfmadd231pd(ymm0, ymm3, ymm10)
    vfmadd231pd(ymm1, ymm3, ymm11)

    vpermilpd(imm(0x5), ymm3, ymm3)
    vfmadd231pd(ymm0, ymm3, ymm13)
    vfmadd231pd(ymm1, ymm3, ymm14)
    add(imm(2*16), rbx)          // b += 2*sizeof(dcomplex)*rs_b = 2*16;

    // ---------------------------------- Iteration 4
    vmovupd(mem(rax), ymm0)
    vmovupd(mem(rax, r8, 1), ymm1)
    add(imm(2*16), rax)            // a += 2*sizeof(dcomplex)*cs_a = 2*16;

    vmovupd(mem(rbx), ymm3)
    vfmadd231pd(ymm0, ymm3, ymm4)
    vfmadd231pd(ymm1, ymm3, ymm5)

    vpermilpd(imm(0x5), ymm3, ymm3)
    vfmadd231pd(ymm0, ymm3, ymm7)
    vfmadd231pd(ymm1, ymm3, ymm8)

    vmovupd(mem(rbx, r10, 1), ymm3)
    vfmadd231pd(ymm0, ymm3, ymm10)
    vfmadd231pd(ymm1, ymm3, ymm11)

    vpermilpd(imm(0x5), ymm3, ymm3)
    vfmadd231pd(ymm0, ymm3, ymm13)
    vfmadd231pd(ymm1, ymm3, ymm14)
    add(imm(2*16), rbx)          // b += 2*sizeof(dcomplex)*rs_b = 2*16;

    dec(rsi)
    jne(.ZLOOPKITER8)

    label(.ZLOOPKLEFT8)

    mov(var(k_iter4), rsi)      // i = k_iter4;
    test(rsi, rsi)               // Check i via logical AND
    je(.ZLOOPKLEFT4)             // If i=0 jmp to k_left loop
    label(.ZLOOPKITER4)
    // ---------------------------------- Iteration 0
    vmovupd(mem(rax), ymm0)
    vmovupd(mem(rax, r8, 1), ymm1)
    add(imm(2*16), rax)            // a += 2*sizeof(dcomplex)*cs_a = 2*16;

    vmovupd(mem(rbx), ymm3)
    vfmadd231pd(ymm0, ymm3, ymm4)
    vfmadd231pd(ymm1, ymm3, ymm5)

    vpermilpd(imm(0x5), ymm3, ymm3)
    vfmadd231pd(ymm0, ymm3, ymm7)
    vfmadd231pd(ymm1, ymm3, ymm8)

    vmovupd(mem(rbx, r10, 1), ymm3)
    vfmadd231pd(ymm0, ymm3, ymm10)
    vfmadd231pd(ymm1, ymm3, ymm11)

    vpermilpd(imm(0x5), ymm3, ymm3)
    vfmadd231pd(ymm0, ymm3, ymm13)
    vfmadd231pd(ymm1, ymm3, ymm14)
    add(imm(2*16), rbx)          // b += 2*sizeof(dcomplex)*rs_b = 2*16;

    // ---------------------------------- Iteration 1
    vmovupd(mem(rax), ymm0)
    vmovupd(mem(rax, r8, 1), ymm1)
    add(imm(2*16), rax)            // a += 2*sizeof(dcomplex)*cs_a = 2*16;

    vmovupd(mem(rbx), ymm3)
    vfmadd231pd(ymm0, ymm3, ymm4)
    vfmadd231pd(ymm1, ymm3, ymm5)

    vpermilpd(imm(0x5), ymm3, ymm3)
    vfmadd231pd(ymm0, ymm3, ymm7)
    vfmadd231pd(ymm1, ymm3, ymm8)

    vmovupd(mem(rbx, r10, 1), ymm3)
    vfmadd231pd(ymm0, ymm3, ymm10)
    vfmadd231pd(ymm1, ymm3, ymm11)

    vpermilpd(imm(0x5), ymm3, ymm3)
    vfmadd231pd(ymm0, ymm3, ymm13)
    vfmadd231pd(ymm1, ymm3, ymm14)
    add(imm(2*16), rbx)          // b += 2*sizeof(dcomplex)*rs_b = 2*16;

    dec(rsi)
    jne(.ZLOOPKITER4)

    label(.ZLOOPKLEFT4)
    mov(var(k_left4), rsi)      // i = k_left4;
    test(rsi, rsi)               // Check i via logical AND
    je(.ZPOSTACCUM)             // If i=0 jmp to accumalation
    label(.ZLOOPKLEFT)

    vmovupd(mem(rax), xmm0)
    vmovupd(mem(rax, r8, 1), xmm1)
    add(imm(1*16), rax)          // a += 1*sizeof(dcomplex)*cs_a = 1*16*1;

    vmovupd(mem(rbx), xmm3)
    vfmadd231pd(ymm3, ymm0, ymm4)
    vfmadd231pd(ymm3, ymm1, ymm5)

    vpermilpd(imm(0x1), xmm3, xmm3)
    vfmadd231pd(ymm3, ymm0, ymm7)
    vfmadd231pd(ymm3, ymm1, ymm8)

    vmovupd(mem(rbx, r10, 1), xmm3)
    vfmadd231pd(ymm3, ymm0, ymm10)
    vfmadd231pd(ymm3, ymm1, ymm11)

    vpermilpd(imm(0x1), xmm3, xmm3)
    vfmadd231pd(ymm3, ymm0, ymm13)
    vfmadd231pd(ymm3, ymm1, ymm14)
    add(imm(1*16), rbx)          // b += 2*sizeof(dcomplex)*rs_b = 1*16*1;

    dec(rsi)
    jne(.ZLOOPKLEFT)

    label(.ZPOSTACCUM)
    vhsubpd( ymm10, ymm4, ymm0 )
    vextractf128(imm(1), ymm0, xmm1 )
    vaddpd( xmm0, xmm1, xmm0 ) // xmm0 = sum(ymm4) sum(ymm10)
    vhaddpd( ymm13, ymm7, ymm2 )
    vextractf128(imm(1), ymm2, xmm1 )
    vaddpd( xmm2, xmm1, xmm2 )  // xmm2 = sum(ymm7) sum(ymm13)

    vshufpd(imm(0),xmm2,xmm0,xmm4)  // xmm4 = sum(ymm4) sum(ymm7)
    vshufpd(imm(3),xmm2,xmm0,xmm7)  // xmm7 = sum(ymm10) sum(ymm13)

    vhsubpd( ymm11, ymm5, ymm0 )
    vextractf128(imm(1), ymm0, xmm1 )
    vaddpd( xmm0, xmm1, xmm0 ) // xmm0 = sum(ymm5) sum(ymm11)
    vhaddpd( ymm14, ymm8, ymm2 )
    vextractf128(imm(1), ymm2, xmm1 )
    vaddpd( xmm2, xmm1, xmm2 ) // xmm2 = sum(ymm8) sum(ymm14)

    vshufpd(imm(0),xmm2,xmm0,xmm5) // xmm5 = sum(ymm5) sum(ymm8)
    vshufpd(imm(3),xmm2,xmm0,xmm8) // xmm8 = sum(ymm11) sum(ymm14)

    vinsertf128(imm(1),xmm7,ymm4,ymm4)
    vinsertf128(imm(1),xmm8,ymm5,ymm5)

    //Scaling with alpha
    mov(var(alpha), rax)
    vbroadcastsd(mem(rax), ymm0)    // ymm0 = real(alpha)
    vbroadcastsd(mem(rax, 8), ymm1) // ymm1 = imag(alpha)

    vpermilpd(imm(0x5), ymm4, ymm10)
    vpermilpd(imm(0x5), ymm5, ymm11)

    vmulpd(ymm0, ymm4, ymm4)
    vmulpd(ymm1, ymm10, ymm10)
    vaddsubpd(ymm10, ymm4, ymm4)

    vmulpd(ymm0, ymm5, ymm5)
    vmulpd(ymm1, ymm11, ymm11)
    vaddsubpd(ymm11, ymm5, ymm5)

    // Scaling with beta
    mov(var(beta_mul_type), al)
    cmp(imm(0), al) // Checking if beta = 0.0
    je(.BETA_ZERO)
    mov(var(beta), rbx)
    vbroadcastsd(mem(rbx), ymm1)    // ymm1 = real(beta)
    vbroadcastsd(mem(rbx, 8), ymm2) // ymm2 = imag(beta)

    vmovupd(mem(rcx), ymm0)
    vpermilpd(imm(0x5), ymm0, ymm3)
    vmulpd(ymm1, ymm0, ymm0)
    vmulpd(ymm2, ymm3, ymm3)
    vaddsubpd(ymm3, ymm0, ymm0)
    vaddpd(ymm0,ymm4,ymm0)
    vmovupd(ymm0, mem(rcx))
    add(rdi, rcx)

    vmovupd(mem(rcx), ymm0)
    vpermilpd(imm(0x5), ymm0, ymm3)
    vmulpd(ymm1, ymm0, ymm0)
    vmulpd(ymm2, ymm3, ymm3)
    vaddsubpd(ymm3, ymm0, ymm0)
    vaddpd(ymm0,ymm5,ymm0)
    vmovupd(ymm0, mem(rcx))

    jmp(.ZDONE)

    label(.BETA_ZERO)

    //Storing in C
    vmovupd(ymm4, mem(rcx))
    add(rdi, rcx)

    vmovupd(ymm5, mem(rcx))

    label(.ZDONE)

    add(imm(2), r15)              // jj += 2
    cmp(imm(4), r15)
    jl(.ZLOOP3X4J)                // Iterate again if jj < 4
    label(.ZRETURN)

    end_asm(
    : // output operands (none)
    : // input operands
      [beta_mul_type] "m" (beta_mul_type),
      [k_iter8] "m" (k_iter8),
      [k_left8] "m" (k_left8),
      [k_iter4] "m" (k_iter4),
      [k_left4] "m" (k_left4),
      [alpha]  "m" (alpha),
      [a]      "m" (a),
      [rs_a]   "m" (rs_a),
      [b]      "m" (b),
      [cs_b]   "m" (cs_b),
      [beta]   "m" (beta),
      [c]      "m" (c),
      [rs_c]   "m" (rs_c)
    : // register clobber list
      "rax", "rbx", "rdx", "rcx", "rsi", "rdi",
      "r8", "r9", "r10", "r12", "r14", "r15", "r11",
      "xmm0", "xmm1", "xmm2", "xmm3",
      "xmm4", "xmm5", "xmm6", "xmm7",
      "xmm8", "xmm9", "xmm10", "xmm11",
      "xmm12", "xmm13", "xmm14", "xmm15",
      "ymm0", "ymm1", "ymm2", "ymm3",
      "ymm4", "ymm5", "ymm7", "ymm8",
      "ymm10", "ymm11", "ymm13", "ymm14",
      "memory"
    )

}

void bli_zgemmsup_rd_zen_asm_1x4
     (
       conj_t       conja,
       conj_t       conjb,
       dim_t        m0,
       dim_t        n0,
       dim_t        k0,
       dcomplex*    restrict alpha,
       dcomplex*    restrict a, inc_t rs_a0, inc_t cs_a0,
       dcomplex*    restrict b, inc_t rs_b0, inc_t cs_b0,
       dcomplex*    restrict beta,
       dcomplex*    restrict c, inc_t rs_c0, inc_t cs_c0,
       auxinfo_t* restrict data,
       cntx_t*    restrict cntx
     )
{
    uint64_t k_iter8 = k0 / 8;
    uint64_t k_left8 = k0 % 8;
    uint64_t k_iter4 = k_left8 / 4;
    uint64_t k_left4 = k_left8 % 4;

    uint64_t rs_a   = rs_a0;
    uint64_t cs_b   = cs_b0;
    uint64_t rs_c   = rs_c0;

    // Checking whether generic/special case handling is required for beta scaling
    char beta_mul_type = (beta->real == 0.0 && beta->imag == 0.0)? BLIS_MUL_ZERO : BLIS_MUL_DEFAULT;

    //-----------------------------------------------------------//
    // Inline assembly implementation

    begin_asm()
    mov(var(rs_a), r8)             // load rs_a
    lea(mem(, r8, 8), r8)
    lea(mem(, r8, 2), r8)          // r8 = sizeof(dcomplex)*rs_a

    mov(var(cs_b), r10)
    lea(mem(, r10, 8), r10)
    lea(mem(, r10, 2), r10)       // r10 = sizeof(dcomplex)*cs_b

    mov(var(rs_c), rdi)
    lea(mem(, rdi, 8), rdi)
    lea(mem(, rdi, 2), rdi)       // rdi = sizeof(dcomplex)*rs_c

    mov(imm(0), r15)               // jj = 0
    label(.ZLOOP3X4J)              // LOOP OVER jj = [ 0 1 ... ]
    mov(var(a), r14)               // r14 = addr of a
    mov(var(b), r11)               // r11 = addr of b
    mov(var(c), r12)               // r12 = addr of c

    lea(mem(, r15, 1), rsi)
    imul(imm(1*16), rsi)            // rsi = 16*jj
    lea(mem(r12, rsi, 1), r12)    // r12 += 16*jj

    lea(mem(, r15, 1), rsi)
    imul(r10, rsi)                 // rsi = 16*jj
    lea(mem(r11, rsi, 1), r11)    // r12 += cs_b*jj

    vzeroall()                      // Reset all ymm registers
    mov(r12, rcx)                  // rcx = c_iijj;
    mov(r11, rbx)                  // rbx = b_jj;
    mov(r14, rax)                  // rax = a_ii;

    mov(var(k_iter8), rsi)        // i = k_iter8;
    test(rsi, rsi)                 // Check i via logical AND
    je(.ZLOOPKLEFT8)               // If i=0 jmp to k_iter4 loop

    label(.ZLOOPKITER8)            // MAIN LOOP

    // ---------------------------------- Iteration 0
    vmovupd(mem(rax), ymm0)
    add(imm(2*16), rax)            // a += 2*sizeof(dcomplex)*cs_a = 2*16;

    vmovupd(mem(rbx), ymm3)
    vfmadd231pd(ymm0, ymm3, ymm4)

    vpermilpd(imm(0x5), ymm3, ymm3)
    vfmadd231pd(ymm0, ymm3, ymm7)

    vmovupd(mem(rbx, r10, 1), ymm3)
    vfmadd231pd(ymm0, ymm3, ymm10)

    vpermilpd(imm(0x5), ymm3, ymm3)
    vfmadd231pd(ymm0, ymm3, ymm13)
    add(imm(2*16), rbx)          // b += 2*sizeof(dcomplex)*rs_b = 2*16;

    // ---------------------------------- Iteration 1
    vmovupd(mem(rax), ymm0)
    add(imm(2*16), rax)            // a += 2*sizeof(dcomplex)*cs_a = 2*16;

    vmovupd(mem(rbx), ymm3)
    vfmadd231pd(ymm0, ymm3, ymm4)

    vpermilpd(imm(0x5), ymm3, ymm3)
    vfmadd231pd(ymm0, ymm3, ymm7)

    vmovupd(mem(rbx, r10, 1), ymm3)
    vfmadd231pd(ymm0, ymm3, ymm10)

    vpermilpd(imm(0x5), ymm3, ymm3)
    vfmadd231pd(ymm0, ymm3, ymm13)
    add(imm(2*16), rbx)          // b += 2*sizeof(dcomplex)*rs_b = 2*16;

    // ---------------------------------- Iteration 2
    vmovupd(mem(rax), ymm0)
    add(imm(2*16), rax)            // a += 2*sizeof(dcomplex)*cs_a = 2*16;

    vmovupd(mem(rbx), ymm3)
    vfmadd231pd(ymm0, ymm3, ymm4)

    vpermilpd(imm(0x5), ymm3, ymm3)
    vfmadd231pd(ymm0, ymm3, ymm7)

    vmovupd(mem(rbx, r10, 1), ymm3)
    vfmadd231pd(ymm0, ymm3, ymm10)

    vpermilpd(imm(0x5), ymm3, ymm3)
    vfmadd231pd(ymm0, ymm3, ymm13)
    add(imm(2*16), rbx)          // b += 2*sizeof(dcomplex)*rs_b = 2*16;

    // ---------------------------------- Iteration 4
    vmovupd(mem(rax), ymm0)
    add(imm(2*16), rax)            // a += 2*sizeof(dcomplex)*cs_a = 2*16;

    vmovupd(mem(rbx), ymm3)
    vfmadd231pd(ymm0, ymm3, ymm4)

    vpermilpd(imm(0x5), ymm3, ymm3)
    vfmadd231pd(ymm0, ymm3, ymm7)

    vmovupd(mem(rbx, r10, 1), ymm3)
    vfmadd231pd(ymm0, ymm3, ymm10)

    vpermilpd(imm(0x5), ymm3, ymm3)
    vfmadd231pd(ymm0, ymm3, ymm13)
    add(imm(2*16), rbx)          // b += 2*sizeof(dcomplex)*rs_b = 2*16;

    dec(rsi)
    jne(.ZLOOPKITER8)

    label(.ZLOOPKLEFT8)

    mov(var(k_iter4), rsi)      // i = k_iter4;
    test(rsi, rsi)               // Check i via logical AND
    je(.ZLOOPKLEFT4)             // If i=0 jmp to k_left loop
    label(.ZLOOPKITER4)
    // ---------------------------------- Iteration 0
    vmovupd(mem(rax), ymm0)
    add(imm(2*16), rax)            // a += 2*sizeof(dcomplex)*cs_a = 2*16;

    vmovupd(mem(rbx), ymm3)
    vfmadd231pd(ymm0, ymm3, ymm4)

    vpermilpd(imm(0x5), ymm3, ymm3)
    vfmadd231pd(ymm0, ymm3, ymm7)

    vmovupd(mem(rbx, r10, 1), ymm3)
    vfmadd231pd(ymm0, ymm3, ymm10)

    vpermilpd(imm(0x5), ymm3, ymm3)
    vfmadd231pd(ymm0, ymm3, ymm13)
    add(imm(2*16), rbx)          // b += 2*sizeof(dcomplex)*rs_b = 2*16;

    // ---------------------------------- Iteration 1
    vmovupd(mem(rax), ymm0)
    add(imm(2*16), rax)            // a += 2*sizeof(dcomplex)*cs_a = 2*16;

    vmovupd(mem(rbx), ymm3)
    vfmadd231pd(ymm0, ymm3, ymm4)

    vpermilpd(imm(0x5), ymm3, ymm3)
    vfmadd231pd(ymm0, ymm3, ymm7)

    vmovupd(mem(rbx, r10, 1), ymm3)
    vfmadd231pd(ymm0, ymm3, ymm10)

    vpermilpd(imm(0x5), ymm3, ymm3)
    vfmadd231pd(ymm0, ymm3, ymm13)
    add(imm(2*16), rbx)          // b += 2*sizeof(dcomplex)*rs_b = 2*16;

    dec(rsi)
    jne(.ZLOOPKITER4)

    label(.ZLOOPKLEFT4)
    mov(var(k_left4), rsi)      // i = k_left4;
    test(rsi, rsi)               // Check i via logical AND
    je(.ZPOSTACCUM)             // If i=0 jmp to accumalation
    label(.ZLOOPKLEFT)

    vmovupd(mem(rax), xmm0)
    add(imm(1*16), rax)          // a += 1*sizeof(dcomplex)*cs_a = 1*16*1;

    vmovupd(mem(rbx), xmm3)
    vfmadd231pd(ymm3, ymm0, ymm4)

    vpermilpd(imm(0x1), xmm3, xmm3)
    vfmadd231pd(ymm3, ymm0, ymm7)

    vmovupd(mem(rbx, r10, 1), xmm3)
    vfmadd231pd(ymm3, ymm0, ymm10)

    vpermilpd(imm(0x1), xmm3, xmm3)
    vfmadd231pd(ymm3, ymm0, ymm13)
    add(imm(1*16), rbx)          // b += 2*sizeof(dcomplex)*rs_b = 1*16*1;

    dec(rsi)
    jne(.ZLOOPKLEFT)

    label(.ZPOSTACCUM)
    vhsubpd( ymm10, ymm4, ymm0 )
    vextractf128(imm(1), ymm0, xmm1 )
    vaddpd( xmm0, xmm1, xmm0 ) // xmm0 = sum(ymm4) sum(ymm10)
    vhaddpd( ymm13, ymm7, ymm2 )
    vextractf128(imm(1), ymm2, xmm1 )
    vaddpd( xmm2, xmm1, xmm2 )  // xmm2 = sum(ymm7) sum(ymm13)

    vshufpd(imm(0),xmm2,xmm0,xmm4)  // xmm4 = sum(ymm4) sum(ymm7)
    vshufpd(imm(3),xmm2,xmm0,xmm7)  // xmm7 = sum(ymm10) sum(ymm13)

    vinsertf128(imm(1),xmm7,ymm4,ymm4)

    //Scaling with alpha
    mov(var(alpha), rax)
    vbroadcastsd(mem(rax), ymm0)    // ymm0 = real(alpha)
    vbroadcastsd(mem(rax, 8), ymm1) // ymm1 = imag(alpha)

    vpermilpd(imm(0x5), ymm4, ymm10)

    vmulpd(ymm0, ymm4, ymm4)
    vmulpd(ymm1, ymm10, ymm10)
    vaddsubpd(ymm10, ymm4, ymm4)

    // Scaling with beta
    mov(var(beta_mul_type), al)
    cmp(imm(0), al) // Checking if beta = 0.0
    je(.BETA_ZERO)
    mov(var(beta), rbx)
    vbroadcastsd(mem(rbx), ymm1)    // ymm1 = real(beta)
    vbroadcastsd(mem(rbx, 8), ymm2) // ymm2 = imag(beta)

    vmovupd(mem(rcx), ymm0)
    vpermilpd(imm(0x5), ymm0, ymm3)
    vmulpd(ymm1, ymm0, ymm0)
    vmulpd(ymm2, ymm3, ymm3)
    vaddsubpd(ymm3, ymm0, ymm0)
    vaddpd(ymm0,ymm4,ymm0)
    vmovupd(ymm0, mem(rcx))

    jmp(.ZDONE)

    label(.BETA_ZERO)

    //Storing in C
    vmovupd(ymm4, mem(rcx))

    label(.ZDONE)

    add(imm(2), r15)              // jj += 2
    cmp(imm(4), r15)
    jl(.ZLOOP3X4J)                // Iterate again if jj < 4
    label(.ZRETURN)

    end_asm(
    : // output operands (none)
    : // input operands
      [beta_mul_type] "m" (beta_mul_type),
      [k_iter8] "m" (k_iter8),
      [k_left8] "m" (k_left8),
      [k_iter4] "m" (k_iter4),
      [k_left4] "m" (k_left4),
      [alpha]  "m" (alpha),
      [a]      "m" (a),
      [rs_a]   "m" (rs_a),
      [b]      "m" (b),
      [cs_b]   "m" (cs_b),
      [beta]   "m" (beta),
      [c]      "m" (c),
      [rs_c]   "m" (rs_c)
    : // register clobber list
      "rax", "rbx", "rdx", "rcx", "rsi", "rdi",
      "r8", "r9", "r10", "r12", "r14", "r15", "r11",
      "xmm0", "xmm1", "xmm2", "xmm3",
      "xmm4", "xmm5", "xmm6", "xmm7",
      "xmm8", "xmm9", "xmm10", "xmm11",
      "xmm12", "xmm13", "xmm14", "xmm15",
      "ymm0", "ymm1", "ymm2", "ymm3",
      "ymm4", "ymm7", "ymm10", "ymm13",
      "memory"
    )

}

void bli_zgemmsup_rd_zen_asm_2x2
     (
       conj_t       conja,
       conj_t       conjb,
       dim_t        m0,
       dim_t        n0,
       dim_t        k0,
       dcomplex*    restrict alpha,
       dcomplex*    restrict a, inc_t rs_a0, inc_t cs_a0,
       dcomplex*    restrict b, inc_t rs_b0, inc_t cs_b0,
       dcomplex*    restrict beta,
       dcomplex*    restrict c, inc_t rs_c0, inc_t cs_c0,
       auxinfo_t* restrict data,
       cntx_t*    restrict cntx
     )
{

    uint64_t k_iter8 = k0 / 8;
    uint64_t k_left8 = k0 % 8;
    uint64_t k_iter4 = k_left8 / 4;
    uint64_t k_left4 = k_left8 % 4;

    uint64_t rs_a   = rs_a0;
    uint64_t cs_b   = cs_b0;
    uint64_t rs_c   = rs_c0;

    // Checking whether generic/special case handling is required for beta scaling
    char beta_mul_type = (beta->real == 0.0 && beta->imag == 0.0)? BLIS_MUL_ZERO : BLIS_MUL_DEFAULT;

    //-----------------------------------------------------------//
    // Inline assembly implementation

    begin_asm()
    mov(var(rs_a), r8)             // load rs_a
    lea(mem(, r8, 8), r8)
    lea(mem(, r8, 2), r8)          // r8 = sizeof(dcomplex)*rs_a

    mov(var(cs_b), r10)
    lea(mem(, r10, 8), r10)
    lea(mem(, r10, 2), r10)       // r10 = sizeof(dcomplex)*cs_b

    mov(var(rs_c), rdi)
    lea(mem(, rdi, 8), rdi)
    lea(mem(, rdi, 2), rdi)       // rdi = sizeof(dcomplex)*rs_c

    mov(var(a), r14)               // r14 = addr of a
    mov(var(b), r11)               // r11 = addr of b
    mov(var(c), r12)               // r12 = addr of c

    vzeroall()                      // Reset all ymm registers
    mov(r12, rcx)                  // rcx = c_iijj;
    mov(r11, rbx)                  // rbx = b_jj;
    mov(r14, rax)                  // rax = a_ii;

    mov(var(k_iter8), rsi)        // i = k_iter8;
    test(rsi, rsi)                 // Check i via logical AND
    je(.ZLOOPKLEFT8)               // If i=0 jmp to k_iter4 loop

    label(.ZLOOPKITER8)            // MAIN LOOP

    // ---------------------------------- Iteration 0
    vmovupd(mem(rax), ymm0)
    vmovupd(mem(rax, r8, 1), ymm1)
    add(imm(2*16), rax)            // a += 2*sizeof(dcomplex)*cs_a = 2*16;

    vmovupd(mem(rbx), ymm3)
    vfmadd231pd(ymm0, ymm3, ymm4)
    vfmadd231pd(ymm1, ymm3, ymm5)

    vpermilpd(imm(0x5), ymm3, ymm3)
    vfmadd231pd(ymm0, ymm3, ymm7)
    vfmadd231pd(ymm1, ymm3, ymm8)

    vmovupd(mem(rbx, r10, 1), ymm3)
    vfmadd231pd(ymm0, ymm3, ymm10)
    vfmadd231pd(ymm1, ymm3, ymm11)

    vpermilpd(imm(0x5), ymm3, ymm3)
    vfmadd231pd(ymm0, ymm3, ymm13)
    vfmadd231pd(ymm1, ymm3, ymm14)
    add(imm(2*16), rbx)          // b += 2*sizeof(dcomplex)*rs_b = 2*16;

    // ---------------------------------- Iteration 1
    vmovupd(mem(rax), ymm0)
    vmovupd(mem(rax, r8, 1), ymm1)
    add(imm(2*16), rax)            // a += 2*sizeof(dcomplex)*cs_a = 2*16;

    vmovupd(mem(rbx), ymm3)
    vfmadd231pd(ymm0, ymm3, ymm4)
    vfmadd231pd(ymm1, ymm3, ymm5)

    vpermilpd(imm(0x5), ymm3, ymm3)
    vfmadd231pd(ymm0, ymm3, ymm7)
    vfmadd231pd(ymm1, ymm3, ymm8)

    vmovupd(mem(rbx, r10, 1), ymm3)
    vfmadd231pd(ymm0, ymm3, ymm10)
    vfmadd231pd(ymm1, ymm3, ymm11)

    vpermilpd(imm(0x5), ymm3, ymm3)
    vfmadd231pd(ymm0, ymm3, ymm13)
    vfmadd231pd(ymm1, ymm3, ymm14)
    add(imm(2*16), rbx)          // b += 2*sizeof(dcomplex)*rs_b = 2*16;

    // ---------------------------------- Iteration 2
    vmovupd(mem(rax), ymm0)
    vmovupd(mem(rax, r8, 1), ymm1)
    add(imm(2*16), rax)            // a += 2*sizeof(dcomplex)*cs_a = 2*16;

    vmovupd(mem(rbx), ymm3)
    vfmadd231pd(ymm0, ymm3, ymm4)
    vfmadd231pd(ymm1, ymm3, ymm5)

    vpermilpd(imm(0x5), ymm3, ymm3)
    vfmadd231pd(ymm0, ymm3, ymm7)
    vfmadd231pd(ymm1, ymm3, ymm8)

    vmovupd(mem(rbx, r10, 1), ymm3)
    vfmadd231pd(ymm0, ymm3, ymm10)
    vfmadd231pd(ymm1, ymm3, ymm11)

    vpermilpd(imm(0x5), ymm3, ymm3)
    vfmadd231pd(ymm0, ymm3, ymm13)
    vfmadd231pd(ymm1, ymm3, ymm14)
    add(imm(2*16), rbx)          // b += 2*sizeof(dcomplex)*rs_b = 2*16;

    // ---------------------------------- Iteration 3
    vmovupd(mem(rax), ymm0)
    vmovupd(mem(rax, r8, 1), ymm1)
    add(imm(2*16), rax)            // a += 2*sizeof(dcomplex)*cs_a = 2*16;

    vmovupd(mem(rbx), ymm3)
    vfmadd231pd(ymm0, ymm3, ymm4)
    vfmadd231pd(ymm1, ymm3, ymm5)

    vpermilpd(imm(0x5), ymm3, ymm3)
    vfmadd231pd(ymm0, ymm3, ymm7)
    vfmadd231pd(ymm1, ymm3, ymm8)

    vmovupd(mem(rbx, r10, 1), ymm3)
    vfmadd231pd(ymm0, ymm3, ymm10)
    vfmadd231pd(ymm1, ymm3, ymm11)

    vpermilpd(imm(0x5), ymm3, ymm3)
    vfmadd231pd(ymm0, ymm3, ymm13)
    vfmadd231pd(ymm1, ymm3, ymm14)
    add(imm(2*16), rbx)          // b += 2*sizeof(dcomplex)*rs_b = 2*16;

    dec(rsi)
    jne(.ZLOOPKITER8)

    label(.ZLOOPKLEFT8)

    mov(var(k_iter4), rsi)      // i = k_iter4;
    test(rsi, rsi)               // Check i via logical AND
    je(.ZLOOPKLEFT4)             // If i=0 jmp to k_left loop
    label(.ZLOOPKITER4)
    // ---------------------------------- Iteration 0
    vmovupd(mem(rax), ymm0)
    vmovupd(mem(rax, r8, 1), ymm1)
    add(imm(2*16), rax)            // a += 2*sizeof(dcomplex)*cs_a = 2*16;

    vmovupd(mem(rbx), ymm3)
    vfmadd231pd(ymm0, ymm3, ymm4)
    vfmadd231pd(ymm1, ymm3, ymm5)

    vpermilpd(imm(0x5), ymm3, ymm3)
    vfmadd231pd(ymm0, ymm3, ymm7)
    vfmadd231pd(ymm1, ymm3, ymm8)

    vmovupd(mem(rbx, r10, 1), ymm3)
    vfmadd231pd(ymm0, ymm3, ymm10)
    vfmadd231pd(ymm1, ymm3, ymm11)

    vpermilpd(imm(0x5), ymm3, ymm3)
    vfmadd231pd(ymm0, ymm3, ymm13)
    vfmadd231pd(ymm1, ymm3, ymm14)
    add(imm(2*16), rbx)          // b += 2*sizeof(dcomplex)*rs_b = 2*16;

    // ---------------------------------- Iteration 1
    vmovupd(mem(rax), ymm0)
    vmovupd(mem(rax, r8, 1), ymm1)
    add(imm(2*16), rax)            // a += 2*sizeof(dcomplex)*cs_a = 2*16;

    vmovupd(mem(rbx), ymm3)
    vfmadd231pd(ymm0, ymm3, ymm4)
    vfmadd231pd(ymm1, ymm3, ymm5)

    vpermilpd(imm(0x5), ymm3, ymm3)
    vfmadd231pd(ymm0, ymm3, ymm7)
    vfmadd231pd(ymm1, ymm3, ymm8)

    vmovupd(mem(rbx, r10, 1), ymm3)
    vfmadd231pd(ymm0, ymm3, ymm10)
    vfmadd231pd(ymm1, ymm3, ymm11)

    vpermilpd(imm(0x5), ymm3, ymm3)
    vfmadd231pd(ymm0, ymm3, ymm13)
    vfmadd231pd(ymm1, ymm3, ymm14)
    add(imm(2*16), rbx)          // b += 2*sizeof(dcomplex)*rs_b = 2*16;

    dec(rsi)
    jne(.ZLOOPKITER4)

    label(.ZLOOPKLEFT4)
    mov(var(k_left4), rsi)      // i = k_left4;
    test(rsi, rsi)               // Check i via logical AND
    je(.ZPOSTACCUM)             // If i=0 jmp to accumalation
    label(.ZLOOPKLEFT)

    vmovupd(mem(rax), xmm0)
    vmovupd(mem(rax, r8, 1), xmm1)
    add(imm(1*16), rax)          // a += 1*sizeof(dcomplex)*cs_a = 1*16*1;

    vmovupd(mem(rbx), xmm3)
    vfmadd231pd(ymm3, ymm0, ymm4)
    vfmadd231pd(ymm3, ymm1, ymm5)

    vpermilpd(imm(0x1), xmm3, xmm3)
    vfmadd231pd(ymm3, ymm0, ymm7)
    vfmadd231pd(ymm3, ymm1, ymm8)

    vmovupd(mem(rbx, r10, 1), xmm3)
    vfmadd231pd(ymm3, ymm0, ymm10)
    vfmadd231pd(ymm3, ymm1, ymm11)

    vpermilpd(imm(0x1), xmm3, xmm3)
    vfmadd231pd(ymm3, ymm0, ymm13)
    vfmadd231pd(ymm3, ymm1, ymm14)
    add(imm(1*16), rbx)          // b += 2*sizeof(dcomplex)*rs_b = 1*16*1;

    dec(rsi)
    jne(.ZLOOPKLEFT)

    label(.ZPOSTACCUM)
    vhsubpd( ymm10, ymm4, ymm0 )
    vextractf128(imm(1), ymm0, xmm1 )
    vaddpd( xmm0, xmm1, xmm0 ) // xmm0 = sum(ymm4) sum(ymm10)
    vhaddpd( ymm13, ymm7, ymm2 )
    vextractf128(imm(1), ymm2, xmm1 )
    vaddpd( xmm2, xmm1, xmm2 )  // xmm2 = sum(ymm7) sum(ymm13)

    vshufpd(imm(0),xmm2,xmm0,xmm4)  // xmm4 = sum(ymm4) sum(ymm7)
    vshufpd(imm(3),xmm2,xmm0,xmm7)  // xmm7 = sum(ymm10) sum(ymm13)

    vhsubpd( ymm11, ymm5, ymm0 )
    vextractf128(imm(1), ymm0, xmm1 )
    vaddpd( xmm0, xmm1, xmm0 ) // xmm0 = sum(ymm5) sum(ymm11)
    vhaddpd( ymm14, ymm8, ymm2 )
    vextractf128(imm(1), ymm2, xmm1 )
    vaddpd( xmm2, xmm1, xmm2 ) // xmm2 = sum(ymm8) sum(ymm14)

    vshufpd(imm(0),xmm2,xmm0,xmm5) // xmm5 = sum(ymm5) sum(ymm8)
    vshufpd(imm(3),xmm2,xmm0,xmm8) // xmm8 = sum(ymm11) sum(ymm14)

    vinsertf128(imm(1),xmm7,ymm4,ymm4)
    vinsertf128(imm(1),xmm8,ymm5,ymm5)

    //Scaling with alpha
    mov(var(alpha), rax)
    vbroadcastsd(mem(rax), ymm0)    // ymm0 = real(alpha)
    vbroadcastsd(mem(rax, 8), ymm1) // ymm1 = imag(alpha)

    vpermilpd(imm(0x5), ymm4, ymm10)
    vpermilpd(imm(0x5), ymm5, ymm11)

    vmulpd(ymm0, ymm4, ymm4)
    vmulpd(ymm1, ymm10, ymm10)
    vaddsubpd(ymm10, ymm4, ymm4)

    vmulpd(ymm0, ymm5, ymm5)
    vmulpd(ymm1, ymm11, ymm11)
    vaddsubpd(ymm11, ymm5, ymm5)

    // Scaling with beta
    mov(var(beta_mul_type), al)
    cmp(imm(0), al) // Checking if beta = 0.0
    je(.BETA_ZERO)
    mov(var(beta), rbx)
    vbroadcastsd(mem(rbx), ymm1)    // ymm1 = real(beta)
    vbroadcastsd(mem(rbx, 8), ymm2) // ymm2 = imag(beta)

    vmovupd(mem(rcx), ymm0)
    vpermilpd(imm(0x5), ymm0, ymm3)
    vmulpd(ymm1, ymm0, ymm0)
    vmulpd(ymm2, ymm3, ymm3)
    vaddsubpd(ymm3, ymm0, ymm0)
    vaddpd(ymm0,ymm4,ymm0)
    vmovupd(ymm0, mem(rcx))
    add(rdi, rcx)

    vmovupd(mem(rcx), ymm0)
    vpermilpd(imm(0x5), ymm0, ymm3)
    vmulpd(ymm1, ymm0, ymm0)
    vmulpd(ymm2, ymm3, ymm3)
    vaddsubpd(ymm3, ymm0, ymm0)
    vaddpd(ymm0,ymm5,ymm0)
    vmovupd(ymm0, mem(rcx))

    jmp(.ZDONE)

    label(.BETA_ZERO)

    //Storing in C
    vmovupd(ymm4, mem(rcx))
    add(rdi, rcx)

    vmovupd(ymm5, mem(rcx))

    label(.ZDONE)

    end_asm(
    : // output operands (none)
    : // input operands
      [beta_mul_type] "m" (beta_mul_type),
      [k_iter8] "m" (k_iter8),
      [k_left8] "m" (k_left8),
      [k_iter4] "m" (k_iter4),
      [k_left4] "m" (k_left4),
      [alpha]  "m" (alpha),
      [a]      "m" (a),
      [rs_a]   "m" (rs_a),
      [b]      "m" (b),
      [cs_b]   "m" (cs_b),
      [beta]   "m" (beta),
      [c]      "m" (c),
      [rs_c]   "m" (rs_c)
    : // register clobber list
      "rax", "rbx", "rdx", "rcx", "rsi", "rdi",
      "r8", "r9", "r10", "r12", "r14", "r15", "r11",
      "xmm0", "xmm1", "xmm2", "xmm3",
      "xmm4", "xmm5", "xmm6", "xmm7",
      "xmm8", "xmm9", "xmm10", "xmm11",
      "xmm12", "xmm13", "xmm14", "xmm15",
      "ymm0", "ymm1", "ymm2", "ymm3",
      "ymm4", "ymm5", "ymm7", "ymm8",
      "ymm10", "ymm11", "ymm13", "ymm14",
      "memory"
    )

}

void bli_zgemmsup_rd_zen_asm_1x2
     (
       conj_t       conja,
       conj_t       conjb,
       dim_t        m0,
       dim_t        n0,
       dim_t        k0,
       dcomplex*    restrict alpha,
       dcomplex*    restrict a, inc_t rs_a0, inc_t cs_a0,
       dcomplex*    restrict b, inc_t rs_b0, inc_t cs_b0,
       dcomplex*    restrict beta,
       dcomplex*    restrict c, inc_t rs_c0, inc_t cs_c0,
       auxinfo_t* restrict data,
       cntx_t*    restrict cntx
     )
{

    uint64_t k_iter8 = k0 / 8;
    uint64_t k_left8 = k0 % 8;
    uint64_t k_iter4 = k_left8 / 4;
    uint64_t k_left4 = k_left8 % 4;

    uint64_t rs_a   = rs_a0;
    uint64_t cs_b   = cs_b0;
    uint64_t rs_c   = rs_c0;

    // Checking whether generic/special case handling is required for beta scaling
    char beta_mul_type = (beta->real == 0.0 && beta->imag == 0.0)? BLIS_MUL_ZERO : BLIS_MUL_DEFAULT;

    //-----------------------------------------------------------//
    // Inline assembly implementation

    begin_asm()
    mov(var(rs_a), r8)             // load rs_a
    lea(mem(, r8, 8), r8)
    lea(mem(, r8, 2), r8)          // r8 = sizeof(dcomplex)*rs_a

    mov(var(cs_b), r10)
    lea(mem(, r10, 8), r10)
    lea(mem(, r10, 2), r10)       // r10 = sizeof(dcomplex)*cs_b

    mov(var(rs_c), rdi)
    lea(mem(, rdi, 8), rdi)
    lea(mem(, rdi, 2), rdi)       // rdi = sizeof(dcomplex)*rs_c

    mov(var(a), r14)               // r14 = addr of a
    mov(var(b), r11)               // r11 = addr of b
    mov(var(c), r12)               // r12 = addr of c

    vzeroall()                      // Reset all ymm registers
    mov(r12, rcx)                  // rcx = c_iijj;
    mov(r11, rbx)                  // rbx = b_jj;
    mov(r14, rax)                  // rax = a_ii;

    mov(var(k_iter8), rsi)        // i = k_iter8;
    test(rsi, rsi)                 // Check i via logical AND
    je(.ZLOOPKLEFT8)               // If i=0 jmp to k_iter4 loop

    label(.ZLOOPKITER8)            // MAIN LOOP

    // ---------------------------------- Iteration 0
    vmovupd(mem(rax), ymm0)
    add(imm(2*16), rax)            // a += 2*sizeof(dcomplex)*cs_a = 2*16;

    vmovupd(mem(rbx), ymm3)
    vfmadd231pd(ymm0, ymm3, ymm4)

    vpermilpd(imm(0x5), ymm3, ymm3)
    vfmadd231pd(ymm0, ymm3, ymm7)

    vmovupd(mem(rbx, r10, 1), ymm3)
    vfmadd231pd(ymm0, ymm3, ymm10)

    vpermilpd(imm(0x5), ymm3, ymm3)
    vfmadd231pd(ymm0, ymm3, ymm13)
    add(imm(2*16), rbx)          // b += 2*sizeof(dcomplex)*rs_b = 2*16;

    // ---------------------------------- Iteration 1
    vmovupd(mem(rax), ymm0)
    add(imm(2*16), rax)            // a += 2*sizeof(dcomplex)*cs_a = 2*16;

    vmovupd(mem(rbx), ymm3)
    vfmadd231pd(ymm0, ymm3, ymm4)

    vpermilpd(imm(0x5), ymm3, ymm3)
    vfmadd231pd(ymm0, ymm3, ymm7)

    vmovupd(mem(rbx, r10, 1), ymm3)
    vfmadd231pd(ymm0, ymm3, ymm10)

    vpermilpd(imm(0x5), ymm3, ymm3)
    vfmadd231pd(ymm0, ymm3, ymm13)
    add(imm(2*16), rbx)          // b += 2*sizeof(dcomplex)*rs_b = 2*16;

    // ---------------------------------- Iteration 2
    vmovupd(mem(rax), ymm0)
    add(imm(2*16), rax)            // a += 2*sizeof(dcomplex)*cs_a = 2*16;

    vmovupd(mem(rbx), ymm3)
    vfmadd231pd(ymm0, ymm3, ymm4)

    vpermilpd(imm(0x5), ymm3, ymm3)
    vfmadd231pd(ymm0, ymm3, ymm7)

    vmovupd(mem(rbx, r10, 1), ymm3)
    vfmadd231pd(ymm0, ymm3, ymm10)

    vpermilpd(imm(0x5), ymm3, ymm3)
    vfmadd231pd(ymm0, ymm3, ymm13)
    add(imm(2*16), rbx)          // b += 2*sizeof(dcomplex)*rs_b = 2*16;

    // ---------------------------------- Iteration 3
    vmovupd(mem(rax), ymm0)
    add(imm(2*16), rax)            // a += 2*sizeof(dcomplex)*cs_a = 2*16;

    vmovupd(mem(rbx), ymm3)
    vfmadd231pd(ymm0, ymm3, ymm4)

    vpermilpd(imm(0x5), ymm3, ymm3)
    vfmadd231pd(ymm0, ymm3, ymm7)

    vmovupd(mem(rbx, r10, 1), ymm3)
    vfmadd231pd(ymm0, ymm3, ymm10)

    vpermilpd(imm(0x5), ymm3, ymm3)
    vfmadd231pd(ymm0, ymm3, ymm13)
    add(imm(2*16), rbx)          // b += 2*sizeof(dcomplex)*rs_b = 2*16;

    dec(rsi)
    jne(.ZLOOPKITER8)

    label(.ZLOOPKLEFT8)

    mov(var(k_iter4), rsi)      // i = k_iter4;
    test(rsi, rsi)               // Check i via logical AND
    je(.ZLOOPKLEFT4)             // If i=0 jmp to k_left loop
    label(.ZLOOPKITER4)
    // ---------------------------------- Iteration 0
    vmovupd(mem(rax), ymm0)
    add(imm(2*16), rax)            // a += 2*sizeof(dcomplex)*cs_a = 2*16;

    vmovupd(mem(rbx), ymm3)
    vfmadd231pd(ymm0, ymm3, ymm4)

    vpermilpd(imm(0x5), ymm3, ymm3)
    vfmadd231pd(ymm0, ymm3, ymm7)

    vmovupd(mem(rbx, r10, 1), ymm3)
    vfmadd231pd(ymm0, ymm3, ymm10)

    vpermilpd(imm(0x5), ymm3, ymm3)
    vfmadd231pd(ymm0, ymm3, ymm13)
    add(imm(2*16), rbx)          // b += 2*sizeof(dcomplex)*rs_b = 2*16;

    // ---------------------------------- Iteration 1
    vmovupd(mem(rax), ymm0)
    add(imm(2*16), rax)            // a += 2*sizeof(dcomplex)*cs_a = 2*16;

    vmovupd(mem(rbx), ymm3)
    vfmadd231pd(ymm0, ymm3, ymm4)

    vpermilpd(imm(0x5), ymm3, ymm3)
    vfmadd231pd(ymm0, ymm3, ymm7)

    vmovupd(mem(rbx, r10, 1), ymm3)
    vfmadd231pd(ymm0, ymm3, ymm10)

    vpermilpd(imm(0x5), ymm3, ymm3)
    vfmadd231pd(ymm0, ymm3, ymm13)
    add(imm(2*16), rbx)          // b += 2*sizeof(dcomplex)*rs_b = 2*16;

    dec(rsi)
    jne(.ZLOOPKITER4)

    label(.ZLOOPKLEFT4)
    mov(var(k_left4), rsi)      // i = k_left4;
    test(rsi, rsi)               // Check i via logical AND
    je(.ZPOSTACCUM)             // If i=0 jmp to accumalation
    label(.ZLOOPKLEFT)

    vmovupd(mem(rax), xmm0)
    add(imm(1*16), rax)          // a += 1*sizeof(dcomplex)*cs_a = 1*16*1;

    vmovupd(mem(rbx), xmm3)
    vfmadd231pd(ymm3, ymm0, ymm4)

    vpermilpd(imm(0x1), xmm3, xmm3)
    vfmadd231pd(ymm3, ymm0, ymm7)

    vmovupd(mem(rbx, r10, 1), xmm3)
    vfmadd231pd(ymm3, ymm0, ymm10)

    vpermilpd(imm(0x1), xmm3, xmm3)
    vfmadd231pd(ymm3, ymm0, ymm13)
    add(imm(1*16), rbx)          // b += 2*sizeof(dcomplex)*rs_b = 1*16*1;

    dec(rsi)
    jne(.ZLOOPKLEFT)

    label(.ZPOSTACCUM)
    vhsubpd( ymm10, ymm4, ymm0 )
    vextractf128(imm(1), ymm0, xmm1 )
    vaddpd( xmm0, xmm1, xmm0 ) // xmm0 = sum(ymm4) sum(ymm10)
    vhaddpd( ymm13, ymm7, ymm2 )
    vextractf128(imm(1), ymm2, xmm1 )
    vaddpd( xmm2, xmm1, xmm2 )  // xmm2 = sum(ymm7) sum(ymm13)

    vshufpd(imm(0),xmm2,xmm0,xmm4)  // xmm4 = sum(ymm4) sum(ymm7)
    vshufpd(imm(3),xmm2,xmm0,xmm7)  // xmm7 = sum(ymm10) sum(ymm13)

    vinsertf128(imm(1),xmm7,ymm4,ymm4)

    //Scaling with alpha
    mov(var(alpha), rax)
    vbroadcastsd(mem(rax), ymm0)    // ymm0 = real(alpha)
    vbroadcastsd(mem(rax, 8), ymm1) // ymm1 = imag(alpha)

    vpermilpd(imm(0x5), ymm4, ymm10)

    vmulpd(ymm0, ymm4, ymm4)
    vmulpd(ymm1, ymm10, ymm10)
    vaddsubpd(ymm10, ymm4, ymm4)

    // Scaling with beta
    mov(var(beta_mul_type), al)
    cmp(imm(0), al) // Checking if beta = 0.0
    je(.BETA_ZERO)
    mov(var(beta), rbx)
    vbroadcastsd(mem(rbx), ymm1)    // ymm1 = real(beta)
    vbroadcastsd(mem(rbx, 8), ymm2) // ymm2 = imag(beta)

    vmovupd(mem(rcx), ymm0)
    vpermilpd(imm(0x5), ymm0, ymm3)
    vmulpd(ymm1, ymm0, ymm0)
    vmulpd(ymm2, ymm3, ymm3)
    vaddsubpd(ymm3, ymm0, ymm0)
    vaddpd(ymm0,ymm4,ymm0)
    vmovupd(ymm0, mem(rcx))

    jmp(.ZDONE)

    label(.BETA_ZERO)

    //Storing in C
    vmovupd(ymm4, mem(rcx))

    label(.ZDONE)

    end_asm(
    : // output operands (none)
    : // input operands
      [beta_mul_type] "m" (beta_mul_type),
      [k_iter8] "m" (k_iter8),
      [k_left8] "m" (k_left8),
      [k_iter4] "m" (k_iter4),
      [k_left4] "m" (k_left4),
      [alpha]  "m" (alpha),
      [a]      "m" (a),
      [rs_a]   "m" (rs_a),
      [b]      "m" (b),
      [cs_b]   "m" (cs_b),
      [beta]   "m" (beta),
      [c]      "m" (c),
      [rs_c]   "m" (rs_c)
    : // register clobber list
      "rax", "rbx", "rdx", "rcx", "rsi", "rdi",
      "r8", "r9", "r10", "r12", "r14", "r15", "r11",
      "xmm0", "xmm1", "xmm2", "xmm3",
      "xmm4", "xmm5", "xmm6", "xmm7",
      "xmm8", "xmm9", "xmm10", "xmm11",
      "xmm12", "xmm13", "xmm14", "xmm15",
      "ymm0", "ymm1", "ymm2", "ymm3",
      "ymm4", "ymm7", "ymm10", "ymm13",
      "memory"
    )

}
