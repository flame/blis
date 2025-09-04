/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2022 - 2023, Advanced Micro Devices, Inc. All rights reserved.

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
     --------        ------               :
     --------        ------               :

   Assumptions:
   - C is row-stored and B is column-stored;
   - A is row-stored;
   - m0 and n0 are at most MR and NR, respectively.
   Therefore, this (r)ow-preferential microkernel is well-suited for
   a dot-product-based accumulation that performs vector loads from
   both A and B.

   NOTE: These kernels implicitly support column-oriented IO, implemented
   via an a high-level transposition of the entire operation. A and B will
   effectively remain row- and column-stored, respectively, but C will then
   effectively appear column-stored. Thus, this kernel may be used for both
   rrc and crc cases.
*/

void bli_zgemmsup_rd_zen_asm_3x4m
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
    // Checking for edge case in n dimension in order to
    // dispatch 3x?m fringe kernels, as required.
    uint64_t n_left = n0 % 4;

    if ( n_left )
    {
        dcomplex* restrict cij = c;
        dcomplex* restrict bj  = b;
        dcomplex* restrict ai  = a;

        if ( 2 <= n_left )
        {
             const dim_t nr_cur = 2;

             bli_zgemmsup_rd_zen_asm_3x2m
             (
               conja, conjb, m0, nr_cur, k0,
               alpha, ai, rs_a0, cs_a0, bj, rs_b0, cs_b0,
               beta, cij, rs_c0, cs_c0, data, cntx
             );
             cij += nr_cur*cs_c0; bj += nr_cur*cs_b0; n_left -= nr_cur;
        }
        if ( 1 == n_left )
        {
             bli_zgemv_ex
             (
               BLIS_NO_TRANSPOSE, conjb, m0, k0,
               alpha, ai, rs_a0, cs_a0, bj, rs_b0,
               beta, cij, rs_c0, cntx, NULL
             );
        }
        return;
    }

    uint64_t k_iter8 = k0 / 8;
    uint64_t k_left8 = k0 % 8;
    uint64_t k_iter4 = k_left8 / 4;
    uint64_t k_left4 = k_left8 % 4;

    uint64_t m_iter = m0 / 3;
    uint64_t m_left = m0 % 3;

    uint64_t rs_a   = rs_a0;
    uint64_t cs_b   = cs_b0;
    uint64_t rs_c   = rs_c0;

    // Redirecting to m fringe kernels if m_iter = 0
    if ( m_iter == 0 ) goto consider_edge_cases;

    char alpha_mul_type = BLIS_MUL_DEFAULT;
    char beta_mul_type  = BLIS_MUL_DEFAULT;

    // Dealing with special cases of alpha and beta
    if( alpha->imag == 0.0 ) // If alpha is real
    {
      if( alpha->real == 1.0 ) alpha_mul_type = BLIS_MUL_ONE;
      else if( alpha->real == -1.0 )  alpha_mul_type = BLIS_MUL_MINUS_ONE;
    }

    if( beta->imag == 0.0 ) // If beta is real
    {
        if( beta->real == 1.0 )       beta_mul_type = BLIS_MUL_ONE;
        else if( beta->real == -1.0 ) beta_mul_type = BLIS_MUL_MINUS_ONE;
        else if( beta->real == 0.0 )  beta_mul_type = BLIS_MUL_ZERO;
    }

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

    mov(var(m_iter), r9)           // ii = m_iter
    label(.ZLOOP3X4I)              // LOOP OVER ii
    vzeroall()                      // Reset all ymm registers
    mov(r12, rcx)                  // rcx = c_iijj;
    mov(r11, rbx)                  // rbx = b_jj;
    mov(r14, rax)                  // rax = a_ii;

    mov(var(k_iter8), rsi)        // i = k_iter8;
    test(rsi, rsi)                 // Check i via logical AND
    je(.ZLOOPKLEFT8)               // If i=0 jmp to k_iter4 loop

    label(.ZLOOPKITER8)            // MAIN LOOP
    /*
      Load 3 rows from matrix A using ymm0-ymm2.
      Load 2 columns from B one at a time using ymm3.
      Compute point wise pdt of ymm0-ymm2 with ymm3.
      This gives the real part of result, in ymm4-ymm6 and ymm10-ymm12.

      Permute ymm3 after point wise pdt with ymm0-ymm2.
      Compute another set of point wise pdt in ymm7-ymm9 and ymm13-ymm15.

      Cumulative sum of these registers will give the real and imaginary parts
      of the result of dot product.
    */

    // ---------------------------------- Iteration 0
    vmovupd(mem(rax), ymm0)
    vmovupd(mem(rax, r8, 1), ymm1)
    vmovupd(mem(rax, r8, 2), ymm2)
    add(imm(2*16), rax)            // a += 2*sizeof(dcomplex)*cs_a = 2*16;

    vmovupd(mem(rbx), ymm3)
    vfmadd231pd(ymm0, ymm3, ymm4)
    vfmadd231pd(ymm1, ymm3, ymm5)
    vfmadd231pd(ymm2, ymm3, ymm6)

    vpermilpd(imm(0x5), ymm3, ymm3)
    vfmadd231pd(ymm0, ymm3, ymm7)
    vfmadd231pd(ymm1, ymm3, ymm8)
    vfmadd231pd(ymm2, ymm3, ymm9)

    vmovupd(mem(rbx, r10, 1), ymm3)
    vfmadd231pd(ymm0, ymm3, ymm10)
    vfmadd231pd(ymm1, ymm3, ymm11)
    vfmadd231pd(ymm2, ymm3, ymm12)

    vpermilpd(imm(0x5), ymm3, ymm3)
    vfmadd231pd(ymm0, ymm3, ymm13)
    vfmadd231pd(ymm1, ymm3, ymm14)
    vfmadd231pd(ymm2, ymm3, ymm15)
    add(imm(2*16), rbx)          // b += 2*sizeof(dcomplex)*rs_b = 2*16;

    // ---------------------------------- Iteration 1
    vmovupd(mem(rax), ymm0)
    vmovupd(mem(rax, r8, 1), ymm1)
    vmovupd(mem(rax, r8, 2), ymm2)
    add(imm(2*16), rax)            // a += 2*sizeof(dcomplex)*cs_a = 2*16;

    vmovupd(mem(rbx), ymm3)
    vfmadd231pd(ymm0, ymm3, ymm4)
    vfmadd231pd(ymm1, ymm3, ymm5)
    vfmadd231pd(ymm2, ymm3, ymm6)

    vpermilpd(imm(0x5), ymm3, ymm3)
    vfmadd231pd(ymm0, ymm3, ymm7)
    vfmadd231pd(ymm1, ymm3, ymm8)
    vfmadd231pd(ymm2, ymm3, ymm9)

    vmovupd(mem(rbx, r10, 1), ymm3)
    vfmadd231pd(ymm0, ymm3, ymm10)
    vfmadd231pd(ymm1, ymm3, ymm11)
    vfmadd231pd(ymm2, ymm3, ymm12)

    vpermilpd(imm(0x5), ymm3, ymm3)
    vfmadd231pd(ymm0, ymm3, ymm13)
    vfmadd231pd(ymm1, ymm3, ymm14)
    vfmadd231pd(ymm2, ymm3, ymm15)
    add(imm(2*16), rbx)          // b += 2*sizeof(dcomplex)*rs_b = 2*16;

    // ---------------------------------- Iteration 2
    vmovupd(mem(rax), ymm0)
    vmovupd(mem(rax, r8, 1), ymm1)
    vmovupd(mem(rax, r8, 2), ymm2)
    add(imm(2*16), rax)            // a += 2*sizeof(dcomplex)*cs_a = 2*16;

    vmovupd(mem(rbx), ymm3)
    vfmadd231pd(ymm0, ymm3, ymm4)
    vfmadd231pd(ymm1, ymm3, ymm5)
    vfmadd231pd(ymm2, ymm3, ymm6)

    vpermilpd(imm(0x5), ymm3, ymm3)
    vfmadd231pd(ymm0, ymm3, ymm7)
    vfmadd231pd(ymm1, ymm3, ymm8)
    vfmadd231pd(ymm2, ymm3, ymm9)

    vmovupd(mem(rbx, r10, 1), ymm3)
    vfmadd231pd(ymm0, ymm3, ymm10)
    vfmadd231pd(ymm1, ymm3, ymm11)
    vfmadd231pd(ymm2, ymm3, ymm12)

    vpermilpd(imm(0x5), ymm3, ymm3)
    vfmadd231pd(ymm0, ymm3, ymm13)
    vfmadd231pd(ymm1, ymm3, ymm14)
    vfmadd231pd(ymm2, ymm3, ymm15)
    add(imm(2*16), rbx)          // b += 2*sizeof(dcomplex)*rs_b = 2*16;

    // ---------------------------------- Iteration 4
    vmovupd(mem(rax), ymm0)
    vmovupd(mem(rax, r8, 1), ymm1)
    vmovupd(mem(rax, r8, 2), ymm2)
    add(imm(2*16), rax)            // a += 2*sizeof(dcomplex)*cs_a = 2*16;

    vmovupd(mem(rbx), ymm3)
    vfmadd231pd(ymm0, ymm3, ymm4)
    vfmadd231pd(ymm1, ymm3, ymm5)
    vfmadd231pd(ymm2, ymm3, ymm6)

    vpermilpd(imm(0x5), ymm3, ymm3)
    vfmadd231pd(ymm0, ymm3, ymm7)
    vfmadd231pd(ymm1, ymm3, ymm8)
    vfmadd231pd(ymm2, ymm3, ymm9)

    vmovupd(mem(rbx, r10, 1), ymm3)
    vfmadd231pd(ymm0, ymm3, ymm10)
    vfmadd231pd(ymm1, ymm3, ymm11)
    vfmadd231pd(ymm2, ymm3, ymm12)

    vpermilpd(imm(0x5), ymm3, ymm3)
    vfmadd231pd(ymm0, ymm3, ymm13)
    vfmadd231pd(ymm1, ymm3, ymm14)
    vfmadd231pd(ymm2, ymm3, ymm15)
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
    vmovupd(mem(rax, r8, 2), ymm2)
    add(imm(2*16), rax)            // a += 2*sizeof(dcomplex)*cs_a = 2*16;

    vmovupd(mem(rbx), ymm3)
    vfmadd231pd(ymm0, ymm3, ymm4)
    vfmadd231pd(ymm1, ymm3, ymm5)
    vfmadd231pd(ymm2, ymm3, ymm6)

    vpermilpd(imm(0x5), ymm3, ymm3)
    vfmadd231pd(ymm0, ymm3, ymm7)
    vfmadd231pd(ymm1, ymm3, ymm8)
    vfmadd231pd(ymm2, ymm3, ymm9)

    vmovupd(mem(rbx, r10, 1), ymm3)
    vfmadd231pd(ymm0, ymm3, ymm10)
    vfmadd231pd(ymm1, ymm3, ymm11)
    vfmadd231pd(ymm2, ymm3, ymm12)

    vpermilpd(imm(0x5), ymm3, ymm3)
    vfmadd231pd(ymm0, ymm3, ymm13)
    vfmadd231pd(ymm1, ymm3, ymm14)
    vfmadd231pd(ymm2, ymm3, ymm15)
    add(imm(2*16), rbx)          // b += 2*sizeof(dcomplex)*rs_b = 2*16;

    // ---------------------------------- Iteration 1
    vmovupd(mem(rax), ymm0)
    vmovupd(mem(rax, r8, 1), ymm1)
    vmovupd(mem(rax, r8, 2), ymm2)
    add(imm(2*16), rax)            // a += 2*sizeof(dcomplex)*cs_a = 2*16;

    vmovupd(mem(rbx), ymm3)
    vfmadd231pd(ymm0, ymm3, ymm4)
    vfmadd231pd(ymm1, ymm3, ymm5)
    vfmadd231pd(ymm2, ymm3, ymm6)

    vpermilpd(imm(0x5), ymm3, ymm3)
    vfmadd231pd(ymm0, ymm3, ymm7)
    vfmadd231pd(ymm1, ymm3, ymm8)
    vfmadd231pd(ymm2, ymm3, ymm9)

    vmovupd(mem(rbx, r10, 1), ymm3)
    vfmadd231pd(ymm0, ymm3, ymm10)
    vfmadd231pd(ymm1, ymm3, ymm11)
    vfmadd231pd(ymm2, ymm3, ymm12)

    vpermilpd(imm(0x5), ymm3, ymm3)
    vfmadd231pd(ymm0, ymm3, ymm13)
    vfmadd231pd(ymm1, ymm3, ymm14)
    vfmadd231pd(ymm2, ymm3, ymm15)
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
    vmovupd(mem(rax, r8, 2), xmm2)
    add(imm(1*16), rax)          // a += 1*sizeof(dcomplex)*cs_a = 1*16*1;

    vmovupd(mem(rbx), xmm3)
    vfmadd231pd(ymm3, ymm0, ymm4)
    vfmadd231pd(ymm3, ymm1, ymm5)
    vfmadd231pd(ymm3, ymm2, ymm6)

    vpermilpd(imm(0x1), xmm3, xmm3)
    vfmadd231pd(ymm3, ymm0, ymm7)
    vfmadd231pd(ymm3, ymm1, ymm8)
    vfmadd231pd(ymm3, ymm2, ymm9)

    vmovupd(mem(rbx, r10, 1), xmm3)
    vfmadd231pd(ymm3, ymm0, ymm10)
    vfmadd231pd(ymm3, ymm1, ymm11)
    vfmadd231pd(ymm3, ymm2, ymm12)

    vpermilpd(imm(0x1), xmm3, xmm3)
    vfmadd231pd(ymm3, ymm0, ymm13)
    vfmadd231pd(ymm3, ymm1, ymm14)
    vfmadd231pd(ymm3, ymm2, ymm15)
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

    vhsubpd( ymm12, ymm6, ymm0 )
    vextractf128(imm(1), ymm0, xmm1 )
    vaddpd( xmm0, xmm1, xmm0 ) // xmm0 = sum(ymm6) sum(ymm12)
    vhaddpd( ymm15, ymm9, ymm2 )
    vextractf128(imm(1), ymm2, xmm1 )
    vaddpd( xmm2, xmm1, xmm2 ) // xmm2 = sum(ymm9) sum(ymm15)

    vshufpd(imm(0),xmm2,xmm0,xmm6) // xmm6 = sum(ymm6) sum(ymm9)
    vshufpd(imm(3),xmm2,xmm0,xmm9) // xmm9 = sum(ymm12) sum(ymm15)

    vinsertf128(imm(1),xmm7,ymm4,ymm4)
    vinsertf128(imm(1),xmm8,ymm5,ymm5)
    vinsertf128(imm(1),xmm9,ymm6,ymm6)

    //Scaling with alpha
    mov(var(alpha_mul_type), al)
    cmp(imm(0xFF), al) // Checking if alpha = -1.0
    jne(.ALPHA_NOT_MINUS1)

    vxorpd(ymm0, ymm0, ymm0)
    vsubpd(ymm4, ymm0, ymm4)      // ymm4 = -ymm4
    vsubpd(ymm5, ymm0, ymm5)      // ymm5 = -ymm5
    vsubpd(ymm6, ymm0, ymm6)      // ymm6 = -ymm6

    jmp(.BETA_SCALING)

    label(.ALPHA_NOT_MINUS1)
    cmp(imm(2), al) // Checking for BLIS_MUL_DEFAULT
    jne(.BETA_SCALING)
    mov(var(alpha), rax)
    vbroadcastsd(mem(rax), ymm0)    // ymm0 = real(alpha)
    vbroadcastsd(mem(rax, 8), ymm1) // ymm1 = imag(alpha)

    vpermilpd(imm(0x5), ymm4, ymm10)
    vpermilpd(imm(0x5), ymm5, ymm11)
    vpermilpd(imm(0x5), ymm6, ymm12)

    vmulpd(ymm0, ymm4, ymm4)
    vmulpd(ymm1, ymm10, ymm10)
    vaddsubpd(ymm10, ymm4, ymm4)

    vmulpd(ymm0, ymm5, ymm5)
    vmulpd(ymm1, ymm11, ymm11)
    vaddsubpd(ymm11, ymm5, ymm5)

    vmulpd(ymm0, ymm6, ymm6)
    vmulpd(ymm1, ymm12, ymm12)
    vaddsubpd(ymm12, ymm6, ymm6)

    label(.BETA_SCALING)
    // Scaling with beta
    mov(var(beta_mul_type), al)
    cmp(imm(0), al) // Checking if beta = 0.0
    je(.BETA_ZERO)
    cmp(imm(2), al) // Checking for BLIS_MUL_DEFAULT
    je(.BETA_NOT_REAL_ONE)
    cmp(imm(0xFF), al)
    je(.BETA_REAL_MINUS1) // Checking if beta = -1.0
    // Handling when beta == 1
    vmovupd(mem(rcx), ymm0)
    vaddpd(ymm0,ymm4,ymm0)
    vmovupd(ymm0, mem(rcx))
    add(rdi, rcx)

    vmovupd(mem(rcx), ymm0)
    vaddpd(ymm0,ymm5,ymm0)
    vmovupd(ymm0, mem(rcx))
    add(rdi, rcx)

    vmovupd(mem(rcx), ymm0)
    vaddpd(ymm0,ymm6,ymm0)
    vmovupd(ymm0, mem(rcx))

    jmp(.ZDONE)

    // Handling when beta == -1
    label(.BETA_REAL_MINUS1)
    vmovupd(mem(rcx), ymm0)
    vsubpd(ymm0,ymm4,ymm0)
    vmovupd(ymm0, mem(rcx))
    add(rdi, rcx)

    vmovupd(mem(rcx), ymm0)
    vsubpd(ymm0,ymm5,ymm0)
    vmovupd(ymm0, mem(rcx))
    add(rdi, rcx)

    vmovupd(mem(rcx), ymm0)
    vsubpd(ymm0,ymm6,ymm0)
    vmovupd(ymm0, mem(rcx))

    jmp(.ZDONE)

    label(.BETA_NOT_REAL_ONE)
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
    add(rdi, rcx)

    vmovupd(mem(rcx), ymm0)
    vpermilpd(imm(0x5), ymm0, ymm3)
    vmulpd(ymm1, ymm0, ymm0)
    vmulpd(ymm2, ymm3, ymm3)
    vaddsubpd(ymm3, ymm0, ymm0)
    vaddpd(ymm0,ymm6,ymm0)
    vmovupd(ymm0, mem(rcx))

    jmp(.ZDONE)

    label(.BETA_ZERO)

    //Storing in C
    vmovupd(ymm4, mem(rcx))
    add(rdi, rcx)

    vmovupd(ymm5, mem(rcx))
    add(rdi, rcx)

    vmovupd(ymm6, mem(rcx))

    label(.ZDONE)
    lea(mem(r12, rdi, 2), r12)
    lea(mem(r12, rdi, 1), r12)    // c_ii = r12 += 3*rs_c

    lea(mem(r14, r8,  2), r14)
    lea(mem(r14, r8,  1), r14)    // a_ii = r14 += 3*rs_a

    dec(r9)                       // ii -= 1;
    jne(.ZLOOP3X4I)               // Iterating again if ii != 0

    add(imm(2), r15)              // jj += 2
    cmp(imm(4), r15)
    jl(.ZLOOP3X4J)                // Iterate again if jj < 4
    label(.ZRETURN)

    end_asm(
    : // output operands (none)
    : // input operands
      [alpha_mul_type] "m" (alpha_mul_type),
      [beta_mul_type] "m" (beta_mul_type),
      [m_iter] "m" (m_iter),
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
      "ymm4", "ymm5", "ymm6", "ymm7",
      "ymm8", "ymm9", "ymm10", "ymm11",
      "ymm12", "ymm13", "ymm14", "ymm15",
      "memory"
    )

    // Handling edge cases in m dimension if they exist
    consider_edge_cases:
    if ( m_left )
    {
        const dim_t      nr_cur = 4;
        const dim_t      i_edge = m0 - ( dim_t )m_left;

        dcomplex* restrict cij = c + i_edge*rs_c;
        dcomplex* restrict bj  = b;
        dcomplex* restrict ai  = a + i_edge*rs_a;

        if ( 2 == m_left )
        {
             const dim_t mr_cur = 2;

             bli_zgemmsup_rd_zen_asm_2x4
             (
               conja, conjb, mr_cur, nr_cur, k0,
               alpha, ai, rs_a0, cs_a0, bj, rs_b0, cs_b0,
               beta, cij, rs_c0, cs_c0, data, cntx
             );
        }
        if ( 1 == m_left )
        {
             const dim_t mr_cur = 1;

             bli_zgemmsup_rd_zen_asm_1x4
             (
               conja, conjb, mr_cur, nr_cur, k0,
               alpha, ai, rs_a0, cs_a0, bj, rs_b0, cs_b0,
               beta, cij, rs_c0, cs_c0, data, cntx
             );
        }
    }

}

void bli_zgemmsup_rd_zen_asm_3x2m
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

    uint64_t m_iter = m0 / 3;
    uint64_t m_left = m0 % 3;

    uint64_t rs_a   = rs_a0;
    uint64_t cs_b   = cs_b0;
    uint64_t rs_c   = rs_c0;

    if ( m_iter == 0 ) goto consider_edge_cases;

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

    mov(var(m_iter), r9)           // ii = m_iter
    label(.ZLOOP3X4I)              // LOOP OVER ii
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
    vmovupd(mem(rax, r8, 2), ymm2)
    add(imm(2*16), rax)            // a += 2*sizeof(dcomplex)*cs_a = 2*16;

    vmovupd(mem(rbx), ymm3)
    vfmadd231pd(ymm0, ymm3, ymm4)
    vfmadd231pd(ymm1, ymm3, ymm5)
    vfmadd231pd(ymm2, ymm3, ymm6)

    vpermilpd(imm(0x5), ymm3, ymm3)
    vfmadd231pd(ymm0, ymm3, ymm7)
    vfmadd231pd(ymm1, ymm3, ymm8)
    vfmadd231pd(ymm2, ymm3, ymm9)

    vmovupd(mem(rbx, r10, 1), ymm3)
    vfmadd231pd(ymm0, ymm3, ymm10)
    vfmadd231pd(ymm1, ymm3, ymm11)
    vfmadd231pd(ymm2, ymm3, ymm12)

    vpermilpd(imm(0x5), ymm3, ymm3)
    vfmadd231pd(ymm0, ymm3, ymm13)
    vfmadd231pd(ymm1, ymm3, ymm14)
    vfmadd231pd(ymm2, ymm3, ymm15)
    add(imm(2*16), rbx)          // b += 2*sizeof(dcomplex)*rs_b = 2*16;

    // ---------------------------------- Iteration 1
    vmovupd(mem(rax), ymm0)
    vmovupd(mem(rax, r8, 1), ymm1)
    vmovupd(mem(rax, r8, 2), ymm2)
    add(imm(2*16), rax)            // a += 2*sizeof(dcomplex)*cs_a = 2*16;

    vmovupd(mem(rbx), ymm3)
    vfmadd231pd(ymm0, ymm3, ymm4)
    vfmadd231pd(ymm1, ymm3, ymm5)
    vfmadd231pd(ymm2, ymm3, ymm6)

    vpermilpd(imm(0x5), ymm3, ymm3)
    vfmadd231pd(ymm0, ymm3, ymm7)
    vfmadd231pd(ymm1, ymm3, ymm8)
    vfmadd231pd(ymm2, ymm3, ymm9)

    vmovupd(mem(rbx, r10, 1), ymm3)
    vfmadd231pd(ymm0, ymm3, ymm10)
    vfmadd231pd(ymm1, ymm3, ymm11)
    vfmadd231pd(ymm2, ymm3, ymm12)

    vpermilpd(imm(0x5), ymm3, ymm3)
    vfmadd231pd(ymm0, ymm3, ymm13)
    vfmadd231pd(ymm1, ymm3, ymm14)
    vfmadd231pd(ymm2, ymm3, ymm15)
    add(imm(2*16), rbx)          // b += 2*sizeof(dcomplex)*rs_b = 2*16;

    // ---------------------------------- Iteration 2
    vmovupd(mem(rax), ymm0)
    vmovupd(mem(rax, r8, 1), ymm1)
    vmovupd(mem(rax, r8, 2), ymm2)
    add(imm(2*16), rax)            // a += 2*sizeof(dcomplex)*cs_a = 2*16;

    vmovupd(mem(rbx), ymm3)
    vfmadd231pd(ymm0, ymm3, ymm4)
    vfmadd231pd(ymm1, ymm3, ymm5)
    vfmadd231pd(ymm2, ymm3, ymm6)

    vpermilpd(imm(0x5), ymm3, ymm3)
    vfmadd231pd(ymm0, ymm3, ymm7)
    vfmadd231pd(ymm1, ymm3, ymm8)
    vfmadd231pd(ymm2, ymm3, ymm9)

    vmovupd(mem(rbx, r10, 1), ymm3)
    vfmadd231pd(ymm0, ymm3, ymm10)
    vfmadd231pd(ymm1, ymm3, ymm11)
    vfmadd231pd(ymm2, ymm3, ymm12)

    vpermilpd(imm(0x5), ymm3, ymm3)
    vfmadd231pd(ymm0, ymm3, ymm13)
    vfmadd231pd(ymm1, ymm3, ymm14)
    vfmadd231pd(ymm2, ymm3, ymm15)
    add(imm(2*16), rbx)          // b += 2*sizeof(dcomplex)*rs_b = 2*16;

    // ---------------------------------- Iteration 4
    vmovupd(mem(rax), ymm0)
    vmovupd(mem(rax, r8, 1), ymm1)
    vmovupd(mem(rax, r8, 2), ymm2)
    add(imm(2*16), rax)            // a += 2*sizeof(dcomplex)*cs_a = 2*16;

    vmovupd(mem(rbx), ymm3)
    vfmadd231pd(ymm0, ymm3, ymm4)
    vfmadd231pd(ymm1, ymm3, ymm5)
    vfmadd231pd(ymm2, ymm3, ymm6)

    vpermilpd(imm(0x5), ymm3, ymm3)
    vfmadd231pd(ymm0, ymm3, ymm7)
    vfmadd231pd(ymm1, ymm3, ymm8)
    vfmadd231pd(ymm2, ymm3, ymm9)

    vmovupd(mem(rbx, r10, 1), ymm3)
    vfmadd231pd(ymm0, ymm3, ymm10)
    vfmadd231pd(ymm1, ymm3, ymm11)
    vfmadd231pd(ymm2, ymm3, ymm12)

    vpermilpd(imm(0x5), ymm3, ymm3)
    vfmadd231pd(ymm0, ymm3, ymm13)
    vfmadd231pd(ymm1, ymm3, ymm14)
    vfmadd231pd(ymm2, ymm3, ymm15)
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
    vmovupd(mem(rax, r8, 2), ymm2)
    add(imm(2*16), rax)            // a += 2*sizeof(dcomplex)*cs_a = 2*16;

    vmovupd(mem(rbx), ymm3)
    vfmadd231pd(ymm0, ymm3, ymm4)
    vfmadd231pd(ymm1, ymm3, ymm5)
    vfmadd231pd(ymm2, ymm3, ymm6)

    vpermilpd(imm(0x5), ymm3, ymm3)
    vfmadd231pd(ymm0, ymm3, ymm7)
    vfmadd231pd(ymm1, ymm3, ymm8)
    vfmadd231pd(ymm2, ymm3, ymm9)

    vmovupd(mem(rbx, r10, 1), ymm3)
    vfmadd231pd(ymm0, ymm3, ymm10)
    vfmadd231pd(ymm1, ymm3, ymm11)
    vfmadd231pd(ymm2, ymm3, ymm12)

    vpermilpd(imm(0x5), ymm3, ymm3)
    vfmadd231pd(ymm0, ymm3, ymm13)
    vfmadd231pd(ymm1, ymm3, ymm14)
    vfmadd231pd(ymm2, ymm3, ymm15)
    add(imm(2*16), rbx)          // b += 2*sizeof(dcomplex)*rs_b = 2*16;

    // ---------------------------------- Iteration 1
    vmovupd(mem(rax), ymm0)
    vmovupd(mem(rax, r8, 1), ymm1)
    vmovupd(mem(rax, r8, 2), ymm2)
    add(imm(2*16), rax)            // a += 2*sizeof(dcomplex)*cs_a = 2*16;

    vmovupd(mem(rbx), ymm3)
    vfmadd231pd(ymm0, ymm3, ymm4)
    vfmadd231pd(ymm1, ymm3, ymm5)
    vfmadd231pd(ymm2, ymm3, ymm6)

    vpermilpd(imm(0x5), ymm3, ymm3)
    vfmadd231pd(ymm0, ymm3, ymm7)
    vfmadd231pd(ymm1, ymm3, ymm8)
    vfmadd231pd(ymm2, ymm3, ymm9)

    vmovupd(mem(rbx, r10, 1), ymm3)
    vfmadd231pd(ymm0, ymm3, ymm10)
    vfmadd231pd(ymm1, ymm3, ymm11)
    vfmadd231pd(ymm2, ymm3, ymm12)

    vpermilpd(imm(0x5), ymm3, ymm3)
    vfmadd231pd(ymm0, ymm3, ymm13)
    vfmadd231pd(ymm1, ymm3, ymm14)
    vfmadd231pd(ymm2, ymm3, ymm15)
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
    vmovupd(mem(rax, r8, 2), xmm2)
    add(imm(1*16), rax)          // a += 1*sizeof(dcomplex)*cs_a = 1*16*1;

    vmovupd(mem(rbx), xmm3)
    vfmadd231pd(ymm3, ymm0, ymm4)
    vfmadd231pd(ymm3, ymm1, ymm5)
    vfmadd231pd(ymm3, ymm2, ymm6)

    vpermilpd(imm(0x1), xmm3, xmm3)
    vfmadd231pd(ymm3, ymm0, ymm7)
    vfmadd231pd(ymm3, ymm1, ymm8)
    vfmadd231pd(ymm3, ymm2, ymm9)

    vmovupd(mem(rbx, r10, 1), xmm3)
    vfmadd231pd(ymm3, ymm0, ymm10)
    vfmadd231pd(ymm3, ymm1, ymm11)
    vfmadd231pd(ymm3, ymm2, ymm12)

    vpermilpd(imm(0x1), xmm3, xmm3)
    vfmadd231pd(ymm3, ymm0, ymm13)
    vfmadd231pd(ymm3, ymm1, ymm14)
    vfmadd231pd(ymm3, ymm2, ymm15)
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

    vhsubpd( ymm12, ymm6, ymm0 )
    vextractf128(imm(1), ymm0, xmm1 )
    vaddpd( xmm0, xmm1, xmm0 ) // xmm0 = sum(ymm6) sum(ymm12)
    vhaddpd( ymm15, ymm9, ymm2 )
    vextractf128(imm(1), ymm2, xmm1 )
    vaddpd( xmm2, xmm1, xmm2 ) // xmm2 = sum(ymm9) sum(ymm15)

    vshufpd(imm(0),xmm2,xmm0,xmm6) // xmm6 = sum(ymm6) sum(ymm9)
    vshufpd(imm(3),xmm2,xmm0,xmm9) // xmm9 = sum(ymm12) sum(ymm15)

    vinsertf128(imm(1),xmm7,ymm4,ymm4)
    vinsertf128(imm(1),xmm8,ymm5,ymm5)
    vinsertf128(imm(1),xmm9,ymm6,ymm6)

    //Scaling with alpha
    mov(var(alpha), rax)
    vbroadcastsd(mem(rax), ymm0)    // ymm0 = real(alpha)
    vbroadcastsd(mem(rax, 8), ymm1) // ymm1 = imag(alpha)

    vpermilpd(imm(0x5), ymm4, ymm10)
    vpermilpd(imm(0x5), ymm5, ymm11)
    vpermilpd(imm(0x5), ymm6, ymm12)

    vmulpd(ymm0, ymm4, ymm4)
    vmulpd(ymm1, ymm10, ymm10)
    vaddsubpd(ymm10, ymm4, ymm4)

    vmulpd(ymm0, ymm5, ymm5)
    vmulpd(ymm1, ymm11, ymm11)
    vaddsubpd(ymm11, ymm5, ymm5)

    vmulpd(ymm0, ymm6, ymm6)
    vmulpd(ymm1, ymm12, ymm12)
    vaddsubpd(ymm12, ymm6, ymm6)

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
    add(rdi, rcx)

    vmovupd(mem(rcx), ymm0)
    vpermilpd(imm(0x5), ymm0, ymm3)
    vmulpd(ymm1, ymm0, ymm0)
    vmulpd(ymm2, ymm3, ymm3)
    vaddsubpd(ymm3, ymm0, ymm0)
    vaddpd(ymm0,ymm6,ymm0)
    vmovupd(ymm0, mem(rcx))

    jmp(.ZDONE)

    label(.BETA_ZERO)

    //Storing in C
    vmovupd(ymm4, mem(rcx))
    add(rdi, rcx)

    vmovupd(ymm5, mem(rcx))
    add(rdi, rcx)

    vmovupd(ymm6, mem(rcx))

    label(.ZDONE)

    lea(mem(r12, rdi, 2), r12)
    lea(mem(r12, rdi, 1), r12)    // c_ii = r12 += 3*rs_c

    lea(mem(r14, r8,  2), r14)
    lea(mem(r14, r8,  1), r14)    // a_ii = r14 += 3*rs_a

    dec(r9)                       // ii -= 1;
    jne(.ZLOOP3X4I)               // Iterating again if ii != 0
    label(.ZRETURN)

    end_asm(
    : // output operands (none)
    : // input operands
      [beta_mul_type] "m" (beta_mul_type),
      [m_iter] "m" (m_iter),
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
      "ymm4", "ymm5", "ymm6", "ymm7",
      "ymm8", "ymm9", "ymm10", "ymm11",
      "ymm12", "ymm13", "ymm14", "ymm15",
      "memory"
    )

    // Handling edge cases in m dimension if they exist
    consider_edge_cases:
    if ( m_left )
    {
        const dim_t      nr_cur = 2;
        const dim_t      i_edge = m0 - ( dim_t )m_left;

        dcomplex* restrict cij = c + i_edge*rs_c;
        dcomplex* restrict bj  = b;
        dcomplex* restrict ai  = a + i_edge*rs_a;

        if ( 2 == m_left )
        {
             const dim_t mr_cur = 2;

             bli_zgemmsup_rd_zen_asm_2x2
             (
               conja, conjb, mr_cur, nr_cur, k0,
               alpha, ai, rs_a0, cs_a0, bj, rs_b0, cs_b0,
               beta, cij, rs_c0, cs_c0, data, cntx
             );
        }
        if ( 1 == m_left )
        {
             const dim_t mr_cur = 1;

             bli_zgemmsup_rd_zen_asm_1x2
             (
               conja, conjb, mr_cur, nr_cur, k0,
               alpha, ai, rs_a0, cs_a0, bj, rs_b0, cs_b0,
               beta, cij, rs_c0, cs_c0, data, cntx
             );
        }
    }

}
