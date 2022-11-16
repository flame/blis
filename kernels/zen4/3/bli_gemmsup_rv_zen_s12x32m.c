/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2022, Advanced Micro Devices, Inc. All rights reserved.

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

#define NR 32

/*
   rrr:
	 --------        ------        --------      
	 --------        ------        --------      
	 --------   +=   ------ ...    --------      
	 --------        ------        --------      
	 --------        ------            :         
	 --------        ------            :         
   Assumptions:
   - B is row-stored;
   - A is row-stored;
   - m0 and n0 are at most MR (12) and NR (32), respectively.
   Therefore, this (r)ow-preferential kernel is well-suited for contiguous
   (v)ector loads on B and single-element broadcasts from A.

   NOTE: These kernels currently do not have in-register transpose 
   implemented and hence they do not support column-oriented IO.
*/

void bli_sgemmsup_rv_zen_asm_12x32m
(
       conj_t    conja,
       conj_t    conjb,
       dim_t     m0,
       dim_t     n0,
       dim_t     k0,
       float*    restrict alpha,
       float*    restrict a, inc_t rs_a0, inc_t cs_a0,
       float*    restrict b, inc_t rs_b0, inc_t cs_b0,
       float*    restrict beta,
       float*    restrict c, inc_t rs_c0, inc_t cs_c0,
       auxinfo_t* restrict data,
       cntx_t*    restrict cntx
     )
{
  uint64_t n_left = n0 % NR;  //n0 is expected to be n0<=NR

  // First check whether this is a edge case in the n dimension.
  // If so, dispatch other 12x?m kernels, as needed.
  if (n_left )
  {
    float*  cij = c;
    float*  bj  = b;
    float*  ai  = a;

    if ( 16 <= n_left )
    {
      const dim_t nr_cur = 16;
      bli_sgemmsup_rv_zen_asm_12x16m(conja,conjb,m0,nr_cur,k0,
                                    alpha,ai,rs_a0,cs_a0,
                                    bj,rs_b0,cs_b0,beta,
                                    cij,rs_c0,cs_c0,
                                    data,cntx);
      cij += nr_cur*cs_c0; bj += nr_cur*cs_b0; 
      n_left -= nr_cur;
    }
    if ( 8 <= n_left )
    {
      const dim_t nr_cur = 8;
      bli_sgemmsup_rv_zen_asm_12x8m(conja,conjb,m0,nr_cur,k0,
                                    alpha,ai,rs_a0,cs_a0,
                                    bj,rs_b0,cs_b0,beta,
                                    cij,rs_c0,cs_c0,
                                    data,cntx);
      cij += nr_cur*cs_c0; bj += nr_cur*cs_b0; 
      n_left -= nr_cur;
    }
    if ( 4 <= n_left )
    {
      const dim_t nr_cur = 4;
      bli_sgemmsup_rv_zen_asm_12x4m(conja,conjb,m0,nr_cur,k0,
                                    alpha,ai,rs_a0,cs_a0,
                                    bj,rs_b0,cs_b0,beta,
                                    cij,rs_c0,cs_c0,
                                    data,cntx);
      cij += nr_cur*cs_c0; bj += nr_cur*cs_b0; 
      n_left -= nr_cur;
    }
    if ( 2 <= n_left )
    {
      const dim_t nr_cur = 2;
      bli_sgemmsup_rv_zen_asm_12x2m(conja,conjb,m0,nr_cur,k0,
                                    alpha,ai,rs_a0,cs_a0,
                                    bj,rs_b0,cs_b0,beta,
                                    cij,rs_c0,cs_c0,
                                    data,cntx);
      cij += nr_cur*cs_c0; bj += nr_cur*cs_b0; 
      n_left -= nr_cur;
    }
    if (1 <= n_left)
    {
      const dim_t nr_cur = 1;
      dim_t ps_a0 = bli_auxinfo_ps_a( data );
      if ( ps_a0 == 12 * rs_a0 )
      {
          bli_sgemv_ex
          (
              BLIS_NO_TRANSPOSE, conjb, m0, k0,
              alpha, ai, rs_a0, cs_a0, bj, rs_b0,
              beta, cij, rs_c0, cntx, NULL
          );
      }
      else
      {
          const dim_t mr = 12;
      
          // Since A is packed into row panels, we must use a loop over
          // gemv.
          dim_t m_iter = ( m0 + mr - 1 ) / mr;
          dim_t m_left =   m0            % mr;
      
          float* restrict ai_ii  = ai;
          float* restrict cij_ii = cij;
      
          for ( dim_t ii = 0; ii < m_iter; ii += 1 )
          {
            dim_t mr_cur = ( bli_is_not_edge_f( ii, m_iter, m_left )
                             ? mr : m_left );
            
            bli_sgemv_ex 
            (
              BLIS_NO_TRANSPOSE, conjb, mr_cur, k0,
              alpha, ai_ii, rs_a0, cs_a0, bj, rs_b0,
              beta, cij_ii, rs_c0, cntx, NULL
            );
            cij_ii += mr_cur*rs_c0; ai_ii += ps_a0;
          }            
      }
      n_left -= nr_cur;
    }
    if (n0/NR == 0) {
      return;
    }
  }

  uint64_t k_iter = k0;

  uint64_t m_iter = m0 / 12;
  uint64_t m_left = m0 % 12;

  uint64_t rs_a   = rs_a0;
  uint64_t cs_a   = cs_a0;
  uint64_t rs_b   = rs_b0;
  uint64_t cs_b   = cs_b0;
  uint64_t rs_c   = rs_c0;
  uint64_t cs_c   = cs_c0;

  float *abuf = a;
  float *bbuf = b;
  float *cbuf = c;

  if ( m_iter == 0 ) goto consider_edge_cases;

  /*Produce MRXNR outputs */
  // -------------------------------------------------------------------------
  begin_asm()

  mov(var(rs_a), r8)                 // load rs_a
  lea(mem(, r8, 4), r8)              // rs_a *= sizeof(dt)
  mov(var(rs_b), r9)                 // load rs_b
  lea(mem(, r9, 4), r9)              // rs_b *= sizeof(dt)
  mov(var(cs_a), r10)                // load cs_a
  lea(mem(, r10, 4), r10)            // cs_a *= sizeof(dt)
  mov(var(rs_c), rdi)                // load rs_c
  lea(mem(, rdi, 4), rdi)            // rs_c *= sizeof(float)


  lea(mem(r8, r8, 2), r13)           // r13 = 3*rs_a
  lea(mem(r8, r8, 4), r15)           // r15 = 5*rs_a
  lea(mem(r13, r13, 1), r12)         // r12 = 6*rs_a
  sub(r12,r10)                       // r10 = cs_a-6*rs_a

  mov( var( m_iter ), r11 )        // load m_iter

  label( .M_LOOP_ITER )

  vxorps( zmm0,zmm0,zmm0 )
  vxorps( zmm1,zmm1,zmm1 )
  vxorps( zmm2,zmm2,zmm2 )
  vxorps( zmm3,zmm3,zmm3 )
  vxorps( zmm4,zmm4,zmm4 )
  vxorps( zmm5,zmm5,zmm5 )
  vxorps( zmm6,zmm6,zmm6 )
  vxorps( zmm7,zmm7,zmm7 )
  vxorps( zmm8,zmm8,zmm8 )
  vxorps( zmm9,zmm9,zmm9 )
  vxorps( zmm10,zmm10,zmm10 )
  vxorps( zmm11,zmm11,zmm11 )
  vxorps( zmm12,zmm12,zmm12 )
  vxorps( zmm13,zmm13,zmm13 )
  vxorps( zmm14,zmm14,zmm14 )
  vxorps( zmm15,zmm15,zmm15 )
  vxorps( zmm16,zmm16,zmm16 )
  vxorps( zmm17,zmm17,zmm17 )
  vxorps( zmm18,zmm18,zmm18 )
  vxorps( zmm19,zmm19,zmm19 )
  vxorps( zmm20,zmm20,zmm20 )
  vxorps( zmm21,zmm21,zmm21 )
  vxorps( zmm22,zmm22,zmm22 )
  vxorps( zmm23,zmm23,zmm23 )
  vxorps( zmm24,zmm24,zmm24 )
  vxorps( zmm25,zmm25,zmm25 )
  vxorps( zmm26,zmm26,zmm26 )
  vxorps( zmm27,zmm27,zmm27 )
  vxorps( zmm28,zmm28,zmm28 )
  vxorps( zmm29,zmm29,zmm29 )
  vxorps( zmm30,zmm30,zmm30 )
  vxorps( zmm31,zmm31,zmm31 )
  mov( var(abuf), rax )   // load address of a
  mov( var(bbuf), rbx )   // load address of b
  mov( var(cbuf), rcx )   // load address of c

  mov( var( k_iter ), rsi )   // load k_iter
  test( rsi,rsi )
  mov( var(alpha), rdx )  // load address of alpha
  vbroadcastss( (rdx),zmm28 )

  label( .K_LOOP_ITER )

  vmovups( (rbx),zmm0 )
  vmovups( 0x40(rbx),zmm1 )
  vbroadcastss( (rax),zmm2 )
  vfmadd231ps( zmm0,zmm2,zmm4 )
  vfmadd231ps( zmm1,zmm2,zmm5 )
  vbroadcastss(mem(rax, r8, 1),zmm29 )
  vfmadd231ps( zmm0,zmm29,zmm6 )
  vfmadd231ps( zmm1,zmm29,zmm7 )
  vbroadcastss(mem(rax, r8, 2),zmm30 )
  vfmadd231ps( zmm0,zmm30,zmm8 )
  vfmadd231ps( zmm1,zmm30,zmm9 )
  vbroadcastss(mem(rax, r13, 1 ),zmm31 )
  vfmadd231ps( zmm0,zmm31,zmm10 )
  vfmadd231ps( zmm1,zmm31,zmm11 )
  vbroadcastss(mem(rax, r8, 4),zmm2 )
  vfmadd231ps( zmm0,zmm2,zmm12 )
  vfmadd231ps( zmm1,zmm2,zmm13 )
  vbroadcastss(mem(rax, r15, 1 ),zmm29 )
  vfmadd231ps( zmm0,zmm29,zmm14 )
  vfmadd231ps( zmm1,zmm29,zmm15 )
  add( r12, rax)
  vbroadcastss(mem(rax       ),zmm30 )
  vfmadd231ps( zmm0,zmm30,zmm16 )
  vfmadd231ps( zmm1,zmm30,zmm17 )
  vbroadcastss(mem(rax, r8, 1),zmm31 )
  vfmadd231ps( zmm0,zmm31,zmm18 )
  vfmadd231ps( zmm1,zmm31,zmm19 )
  vbroadcastss(mem(rax, r8, 2),zmm2 )
  vfmadd231ps( zmm0,zmm2,zmm20 )
  vfmadd231ps( zmm1,zmm2,zmm21 )
  vbroadcastss(mem(rax, r13, 1 ),zmm29 )
  vfmadd231ps( zmm0,zmm29,zmm22 )
  vfmadd231ps( zmm1,zmm29,zmm23 )
  vbroadcastss(mem(rax, r8, 4),zmm30 )
  vfmadd231ps( zmm0,zmm30,zmm24 )
  vfmadd231ps( zmm1,zmm30,zmm25 )
  vbroadcastss(mem(rax, r15, 1 ),zmm31 )
  vfmadd231ps( zmm0,zmm31,zmm26 )
  vfmadd231ps( zmm1,zmm31,zmm27 )
  
  add( r9, rbx)
  add( r10, rax)
  dec( rsi )
  jne( .K_LOOP_ITER )

  // Scale by alpha
  vmulps( zmm28,zmm4,zmm4 )
  vmulps( zmm28,zmm5,zmm5 )
  vmulps( zmm28,zmm6,zmm6 )
  vmulps( zmm28,zmm7,zmm7 )
  vmulps( zmm28,zmm8,zmm8 )
  vmulps( zmm28,zmm9,zmm9 )
  vmulps( zmm28,zmm10,zmm10 )
  vmulps( zmm28,zmm11,zmm11 )
  vmulps( zmm28,zmm12,zmm12 )
  vmulps( zmm28,zmm13,zmm13 )
  vmulps( zmm28,zmm14,zmm14 )
  vmulps( zmm28,zmm15,zmm15 )
  vmulps( zmm28,zmm16,zmm16 )
  vmulps( zmm28,zmm17,zmm17 )
  vmulps( zmm28,zmm18,zmm18 )
  vmulps( zmm28,zmm19,zmm19 )
  vmulps( zmm28,zmm20,zmm20 )
  vmulps( zmm28,zmm21,zmm21 )
  vmulps( zmm28,zmm22,zmm22 )
  vmulps( zmm28,zmm23,zmm23 )
  vmulps( zmm28,zmm24,zmm24 )
  vmulps( zmm28,zmm25,zmm25 )
  vmulps( zmm28,zmm26,zmm26 )
  vmulps( zmm28,zmm27,zmm27 )
  
  mov( var(beta), rdx )  // load address of beta
  vbroadcastss( (rdx),zmm28 )

  vxorps(xmm0, xmm0, xmm0)
  vucomiss(xmm0, xmm28)
  je(.SBETAZERO)

  vmovups( (rcx),zmm0 )
  vfmadd231ps( zmm28,zmm0,zmm4 )
  vmovups( zmm4,(rcx) )
  vmovups( 0x40(rcx),zmm0 )
  vfmadd231ps( zmm28,zmm0,zmm5 )
  vmovups( zmm5,0x40(rcx) )
  add(rdi, rcx)
  vmovups( mem(rcx),zmm0 )
  vfmadd231ps( zmm28,zmm0,zmm6 )
  vmovups( zmm6,mem(rcx) )
  vmovups( 0x40(rcx),zmm0 )
  vfmadd231ps( zmm28,zmm0,zmm7 )
  vmovups( zmm7,0x40(rcx) )
  add(rdi, rcx)
  vmovups( mem(rcx),zmm0 )
  vfmadd231ps( zmm28,zmm0,zmm8 )
  vmovups( zmm8,mem(rcx) )
  vmovups( 0x40(rcx),zmm0 )
  vfmadd231ps( zmm28,zmm0,zmm9 )
  vmovups( zmm9,0x40(rcx) )
  add(rdi, rcx)
  vmovups( mem(rcx),zmm0 )
  vfmadd231ps( zmm28,zmm0,zmm10 )
  vmovups( zmm10,mem(rcx) )
  vmovups( 0x40(rcx),zmm0 )
  vfmadd231ps( zmm28,zmm0,zmm11 )
  vmovups( zmm11,0x40(rcx) )
  add(rdi, rcx)
  vmovups( mem(rcx),zmm0 )
  vfmadd231ps( zmm28,zmm0,zmm12 )
  vmovups( zmm12,mem(rcx) )
  vmovups( 0x40(rcx),zmm0 )
  vfmadd231ps( zmm28,zmm0,zmm13 )
  vmovups( zmm13,0x40(rcx) )
  add(rdi, rcx)
  vmovups( mem(rcx),zmm0 )
  vfmadd231ps( zmm28,zmm0,zmm14 )
  vmovups( zmm14,mem(rcx) )
  vmovups( 0x40(rcx),zmm0 )
  vfmadd231ps( zmm28,zmm0,zmm15 )
  vmovups( zmm15,0x40(rcx) )
  add(rdi, rcx)
  vmovups( mem(rcx),zmm0 )
  vfmadd231ps( zmm28,zmm0,zmm16 )
  vmovups( zmm16,mem(rcx) )
  vmovups( 0x40(rcx),zmm0 )
  vfmadd231ps( zmm28,zmm0,zmm17 )
  vmovups( zmm17,0x40(rcx) )
  add(rdi, rcx)
  vmovups( mem(rcx),zmm0 )
  vfmadd231ps( zmm28,zmm0,zmm18 )
  vmovups( zmm18,mem(rcx) )
  vmovups( 0x40(rcx),zmm0 )
  vfmadd231ps( zmm28,zmm0,zmm19 )
  vmovups( zmm19,0x40(rcx) )
  add(rdi, rcx)
  vmovups( mem(rcx),zmm0 )
  vfmadd231ps( zmm28,zmm0,zmm20 )
  vmovups( zmm20,mem(rcx) )
  vmovups( 0x40(rcx),zmm0 )
  vfmadd231ps( zmm28,zmm0,zmm21 )
  vmovups( zmm21,0x40(rcx) )
  add(rdi, rcx)
  vmovups( mem(rcx),zmm0 )
  vfmadd231ps( zmm28,zmm0,zmm22 )
  vmovups( zmm22,mem(rcx) )
  vmovups( 0x40(rcx),zmm0 )
  vfmadd231ps( zmm28,zmm0,zmm23 )
  vmovups( zmm23,0x40(rcx) )
  add(rdi, rcx)
  vmovups( mem(rcx),zmm0 )
  vfmadd231ps( zmm28,zmm0,zmm24 )
  vmovups( zmm24,mem(rcx) )
  vmovups( 0x40(rcx),zmm0 )
  vfmadd231ps( zmm28,zmm0,zmm25 )
  vmovups( zmm25,0x40(rcx) )
  add(rdi, rcx)
  vmovups( mem(rcx),zmm0 )
  vfmadd231ps( zmm28,zmm0,zmm26 )
  vmovups( zmm26,mem(rcx) )
  vmovups( 0x40(rcx),zmm0 )
  vfmadd231ps( zmm28,zmm0,zmm27 )
  vmovups( zmm27,0x40(rcx) )
  add(rdi, rcx)
  jmp(.SDONE)

  label(.SBETAZERO)

  vmovups( zmm4,(rcx) )
  vmovups( zmm5,0x40(rcx) )
  add(rdi, rcx)
  vmovups( zmm6,mem(rcx) )
  vmovups( zmm7,0x40(rcx) )
  add(rdi, rcx)
  vmovups( zmm8,mem(rcx) )
  vmovups( zmm9,0x40(rcx) )
  add(rdi, rcx)
  vmovups( zmm10,mem(rcx) )
  vmovups( zmm11,0x40(rcx) )
  add(rdi, rcx)
  vmovups( zmm12,mem(rcx) )
  vmovups( zmm13,0x40(rcx) )
  add(rdi, rcx)
  vmovups( zmm14,mem(rcx) )
  vmovups( zmm15,0x40(rcx) )
  add(rdi, rcx)
  vmovups( zmm16,mem(rcx) )
  vmovups( zmm17,0x40(rcx) )
  add(rdi, rcx)
  vmovups( zmm18,mem(rcx) )
  vmovups( zmm19,0x40(rcx) )
  add(rdi, rcx)
  vmovups( zmm20,mem(rcx) )
  vmovups( zmm21,0x40(rcx) )
  add(rdi, rcx)
  vmovups( zmm22,mem(rcx) )
  vmovups( zmm23,0x40(rcx) )
  add(rdi, rcx)
  vmovups( zmm24,mem(rcx) )
  vmovups( zmm25,0x40(rcx) )
  add(rdi, rcx)
  vmovups( zmm26,mem(rcx) )
  vmovups( zmm27,0x40(rcx) )
  add(rdi, rcx)

  label(.SDONE)

  lea(mem(, r8, 4), rdx )            // rs_a * 4
  lea(mem(rdx, r8, 8), rdx )         // rdx = rs_a * 12
  mov( var(abuf), rax )              // load address of a
  add( rdx, rax )                    // a += rs_a * MR
  mov( rax, var(abuf) )              // store updated a

  lea(mem(, rdi, 4), rdx )           // rs_c * 4
  lea(mem(rdx, rdi, 8), rdx )        // rdx = rs_c * 12
  mov( var(cbuf), rcx )              // load address of c
  add( rdx, rcx )                    // c += rs_c * MR
  mov( rcx, var(cbuf) )              // store updated c

  dec( r11 )
  jne( .M_LOOP_ITER )

  end_asm(
   : // output operands (none)
   : // input operands
     [k_iter] "m" (k_iter),
     [a]      "m" (a),
     [rs_a]   "m" (rs_a),
     [cs_a]   "m" (cs_a),
     [b]      "m" (b),
     [rs_b]   "m" (rs_b),
     [cs_b]   "m" (cs_b),
     [alpha]  "m" (alpha),
     [beta]   "m" (beta),
     [c]      "m" (c),
     [rs_c]   "m" (rs_c),
     [cs_c]   "m" (cs_c),
     [n0]     "m" (n0),
     [m0]     "m" (m0),
     [m_iter] "m" (m_iter),
     [abuf]   "m" (abuf),
     [bbuf]   "m" (bbuf),
     [cbuf]   "m" (cbuf)
   : // register clobber list
   "rax", "rbx", "rcx", "rdx", "rsi", "rdi",
   "r8", "r9", "r10", "r11", "r12", "r13", "r14", "r15",
   "zmm0", "zmm1", "zmm2", "zmm3",
   "zmm4", "zmm5", "zmm6", "zmm7", "zmm8", "zmm9", "zmm10",
   "zmm11", "zmm12", "zmm13", "zmm14", "zmm15",
   "zmm16", "zmm17", "zmm18", "zmm19",
   "zmm20", "zmm21", "zmm22", "zmm23", "zmm24", "zmm25", "zmm26",
   "zmm27", "zmm28", "zmm29", "zmm30", "zmm31",
   "memory"
   )

  consider_edge_cases:

  // Handle edge cases in the m dimension, if they exist.
  if (m_left)
  {
    const dim_t      i_edge = m0 - ( dim_t )m_left;

    float* restrict cij = c + i_edge*rs_c;
    float* restrict ai  = a + i_edge*rs_a;
    float* restrict bj  = b;

    if (8 <= m_left)
    {
      const dim_t      mr_cur = 8;
      bli_sgemmsup_rv_zen_asm_8x32m(conja, conjb, mr_cur, n0, k0, alpha,
                                    ai, rs_a0, cs_a0,
                                    bj, rs_b0, cs_b0,
                                    beta,
                                    cij, rs_c0, cs_c0,
                                    data, cntx);
      cij += mr_cur * rs_c; ai += mr_cur * rs_a;
      m_left -= mr_cur;
    }
    if (4 <= m_left)
    {
      const dim_t      mr_cur = 4;
      bli_sgemmsup_rv_zen_asm_4x32m(conja, conjb, mr_cur, n0, k0, alpha,
                                    ai, rs_a0, cs_a0,
                                    bj, rs_b0, cs_b0,
                                    beta,
                                    cij, rs_c0, cs_c0,
                                    data, cntx);
      cij += mr_cur * rs_c; ai += mr_cur * rs_a;
      m_left -= mr_cur;
    }
    if (2 <= m_left)
    {
      const dim_t      mr_cur = 2;
      bli_sgemmsup_rv_zen_asm_2x32m(conja, conjb, mr_cur, n0, k0, alpha,
                                    ai, rs_a0, cs_a0,
                                    bj, rs_b0, cs_b0,
                                    beta,
                                    cij, rs_c0, cs_c0,
                                    data, cntx);
      cij += mr_cur * rs_c; ai += mr_cur * rs_a;
      m_left -= mr_cur;
    }
    if (1 <= m_left)
    {
      const dim_t      mr_cur = 1;
      bli_sgemmsup_rv_zen_asm_1x32m(conja, conjb, mr_cur, n0, k0, alpha,
                                    ai, rs_a0, cs_a0,
                                    bj, rs_b0, cs_b0,
                                    beta,
                                    cij, rs_c0, cs_c0,
                                    data, cntx);
      cij += mr_cur * rs_c; ai += mr_cur * rs_a;
      m_left -= mr_cur;
    }
  }
}

void bli_sgemmsup_rv_zen_asm_8x32m
(
       conj_t    conja,
       conj_t    conjb,
       dim_t     m0,
       dim_t     n0,
       dim_t     k0,
       float*    restrict alpha,
       float*    restrict a, inc_t rs_a0, inc_t cs_a0,
       float*    restrict b, inc_t rs_b0, inc_t cs_b0,
       float*    restrict beta,
       float*    restrict c, inc_t rs_c0, inc_t cs_c0,
       auxinfo_t* restrict data,
       cntx_t*    restrict cntx
     )
{
  uint64_t k_iter = k0;

  uint64_t rs_a   = rs_a0;
  uint64_t cs_a   = cs_a0;
  uint64_t rs_b   = rs_b0;
  uint64_t cs_b   = cs_b0;
  uint64_t rs_c   = rs_c0;
  uint64_t cs_c   = cs_c0;

  float *abuf = a;
  float *bbuf = b;
  float *cbuf = c;

  /*Produce MRXNR outputs */
  // -------------------------------------------------------------------------
  begin_asm()

  mov(var(rs_a), r8)                 // load rs_a
  lea(mem(, r8, 4), r8)              // rs_a *= sizeof(dt)
  mov(var(rs_b), r9)                 // load rs_b
  lea(mem(, r9, 4), r9)              // rs_b *= sizeof(dt)
  mov(var(cs_a), r10)                // load cs_a
  lea(mem(, r10, 4), r10)            // cs_a *= sizeof(dt)
  mov(var(rs_c), rdi)                // load rs_c
  lea(mem(, rdi, 4), rdi)            // rs_c *= sizeof(float)


  lea(mem(r8, r8, 2), r13)           // r13 = 3*rs_a
  lea(mem(r8, r8, 4), r15)           // r15 = 5*rs_a
  lea(mem(r13, r13, 1), r12)         // r12 = 6*rs_a
  sub(r12,r10)                       // r10 = cs_a-6*rs_a
  vxorps( zmm0,zmm0,zmm0 )
  vxorps( zmm1,zmm1,zmm1 )
  vxorps( zmm2,zmm2,zmm2 )
  vxorps( zmm3,zmm3,zmm3 )
  vxorps( zmm4,zmm4,zmm4 )
  vxorps( zmm5,zmm5,zmm5 )
  vxorps( zmm6,zmm6,zmm6 )
  vxorps( zmm7,zmm7,zmm7 )
  vxorps( zmm8,zmm8,zmm8 )
  vxorps( zmm9,zmm9,zmm9 )
  vxorps( zmm10,zmm10,zmm10 )
  vxorps( zmm11,zmm11,zmm11 )
  vxorps( zmm12,zmm12,zmm12 )
  vxorps( zmm13,zmm13,zmm13 )
  vxorps( zmm14,zmm14,zmm14 )
  vxorps( zmm15,zmm15,zmm15 )
  vxorps( zmm16,zmm16,zmm16 )
  vxorps( zmm17,zmm17,zmm17 )
  vxorps( zmm18,zmm18,zmm18 )
  vxorps( zmm19,zmm19,zmm19 )
  vxorps( zmm20,zmm20,zmm20 )
  vxorps( zmm21,zmm21,zmm21 )
  vxorps( zmm22,zmm22,zmm22 )
  vxorps( zmm23,zmm23,zmm23 )
  vxorps( zmm24,zmm24,zmm24 )
  vxorps( zmm25,zmm25,zmm25 )
  vxorps( zmm26,zmm26,zmm26 )
  vxorps( zmm27,zmm27,zmm27 )
  vxorps( zmm28,zmm28,zmm28 )
  vxorps( zmm29,zmm29,zmm29 )
  vxorps( zmm30,zmm30,zmm30 )
  vxorps( zmm31,zmm31,zmm31 )
  mov( var(abuf), rax )   // load address of a
  mov( var(bbuf), rbx )   // load address of b
  mov( var(cbuf), rcx )   // load address of c

  mov( var( k_iter ), rsi )   // load k_iter
  test( rsi,rsi )
  mov( var(alpha), rdx )  // load address of alpha
  vbroadcastss( (rdx),zmm28 )

  label( .K_LOOP_ITER )

  vmovups( (rbx),zmm0 )
  vmovups( 0x40(rbx),zmm1 )
  vbroadcastss( (rax),zmm2 )
  vfmadd231ps( zmm0,zmm2,zmm4 )
  vfmadd231ps( zmm1,zmm2,zmm5 )
  vbroadcastss(mem(rax, r8, 1),zmm29 )
  vfmadd231ps( zmm0,zmm29,zmm6 )
  vfmadd231ps( zmm1,zmm29,zmm7 )
  vbroadcastss(mem(rax, r8, 2),zmm30 )
  vfmadd231ps( zmm0,zmm30,zmm8 )
  vfmadd231ps( zmm1,zmm30,zmm9 )
  vbroadcastss(mem(rax, r13, 1 ),zmm31 )
  vfmadd231ps( zmm0,zmm31,zmm10 )
  vfmadd231ps( zmm1,zmm31,zmm11 )
  vbroadcastss(mem(rax, r8, 4),zmm2 )
  vfmadd231ps( zmm0,zmm2,zmm12 )
  vfmadd231ps( zmm1,zmm2,zmm13 )
  vbroadcastss(mem(rax, r15, 1 ),zmm29 )
  vfmadd231ps( zmm0,zmm29,zmm14 )
  vfmadd231ps( zmm1,zmm29,zmm15 )
  add( r12, rax)
  vbroadcastss(mem(rax       ),zmm30 )
  vfmadd231ps( zmm0,zmm30,zmm16 )
  vfmadd231ps( zmm1,zmm30,zmm17 )
  vbroadcastss(mem(rax, r8, 1),zmm31 )
  vfmadd231ps( zmm0,zmm31,zmm18 )
  vfmadd231ps( zmm1,zmm31,zmm19 )
  
  add( r9, rbx)
  add( r10, rax)
  dec( rsi )
  jne( .K_LOOP_ITER )

  // Scale by alpha
  vmulps( zmm28,zmm4,zmm4 )
  vmulps( zmm28,zmm5,zmm5 )
  vmulps( zmm28,zmm6,zmm6 )
  vmulps( zmm28,zmm7,zmm7 )
  vmulps( zmm28,zmm8,zmm8 )
  vmulps( zmm28,zmm9,zmm9 )
  vmulps( zmm28,zmm10,zmm10 )
  vmulps( zmm28,zmm11,zmm11 )
  vmulps( zmm28,zmm12,zmm12 )
  vmulps( zmm28,zmm13,zmm13 )
  vmulps( zmm28,zmm14,zmm14 )
  vmulps( zmm28,zmm15,zmm15 )
  vmulps( zmm28,zmm16,zmm16 )
  vmulps( zmm28,zmm17,zmm17 )
  vmulps( zmm28,zmm18,zmm18 )
  vmulps( zmm28,zmm19,zmm19 )
  
  mov( var(beta), rdx )  // load address of beta
  vbroadcastss( (rdx),zmm28 )

  vxorps(xmm0, xmm0, xmm0)
  vucomiss(xmm0, xmm28)
  je(.SBETAZERO)

  vmovups( (rcx),zmm0 )
  vfmadd231ps( zmm28,zmm0,zmm4 )
  vmovups( zmm4,(rcx) )
  vmovups( 0x40(rcx),zmm0 )
  vfmadd231ps( zmm28,zmm0,zmm5 )
  vmovups( zmm5,0x40(rcx) )
  add(rdi, rcx)
  vmovups( mem(rcx),zmm0 )
  vfmadd231ps( zmm28,zmm0,zmm6 )
  vmovups( zmm6,mem(rcx) )
  vmovups( 0x40(rcx),zmm0 )
  vfmadd231ps( zmm28,zmm0,zmm7 )
  vmovups( zmm7,0x40(rcx) )
  add(rdi, rcx)
  vmovups( mem(rcx),zmm0 )
  vfmadd231ps( zmm28,zmm0,zmm8 )
  vmovups( zmm8,mem(rcx) )
  vmovups( 0x40(rcx),zmm0 )
  vfmadd231ps( zmm28,zmm0,zmm9 )
  vmovups( zmm9,0x40(rcx) )
  add(rdi, rcx)
  vmovups( mem(rcx),zmm0 )
  vfmadd231ps( zmm28,zmm0,zmm10 )
  vmovups( zmm10,mem(rcx) )
  vmovups( 0x40(rcx),zmm0 )
  vfmadd231ps( zmm28,zmm0,zmm11 )
  vmovups( zmm11,0x40(rcx) )
  add(rdi, rcx)
  vmovups( mem(rcx),zmm0 )
  vfmadd231ps( zmm28,zmm0,zmm12 )
  vmovups( zmm12,mem(rcx) )
  vmovups( 0x40(rcx),zmm0 )
  vfmadd231ps( zmm28,zmm0,zmm13 )
  vmovups( zmm13,0x40(rcx) )
  add(rdi, rcx)
  vmovups( mem(rcx),zmm0 )
  vfmadd231ps( zmm28,zmm0,zmm14 )
  vmovups( zmm14,mem(rcx) )
  vmovups( 0x40(rcx),zmm0 )
  vfmadd231ps( zmm28,zmm0,zmm15 )
  vmovups( zmm15,0x40(rcx) )
  add(rdi, rcx)
  vmovups( mem(rcx),zmm0 )
  vfmadd231ps( zmm28,zmm0,zmm16 )
  vmovups( zmm16,mem(rcx) )
  vmovups( 0x40(rcx),zmm0 )
  vfmadd231ps( zmm28,zmm0,zmm17 )
  vmovups( zmm17,0x40(rcx) )
  add(rdi, rcx)
  vmovups( mem(rcx),zmm0 )
  vfmadd231ps( zmm28,zmm0,zmm18 )
  vmovups( zmm18,mem(rcx) )
  vmovups( 0x40(rcx),zmm0 )
  vfmadd231ps( zmm28,zmm0,zmm19 )
  vmovups( zmm19,0x40(rcx) )
  add(rdi, rcx)
  jmp(.SDONE)

  label(.SBETAZERO)

  vmovups( zmm4,(rcx) )
  vmovups( zmm5,0x40(rcx) )
  add(rdi, rcx)
  vmovups( zmm6,mem(rcx) )
  vmovups( zmm7,0x40(rcx) )
  add(rdi, rcx)
  vmovups( zmm8,mem(rcx) )
  vmovups( zmm9,0x40(rcx) )
  add(rdi, rcx)
  vmovups( zmm10,mem(rcx) )
  vmovups( zmm11,0x40(rcx) )
  add(rdi, rcx)
  vmovups( zmm12,mem(rcx) )
  vmovups( zmm13,0x40(rcx) )
  add(rdi, rcx)
  vmovups( zmm14,mem(rcx) )
  vmovups( zmm15,0x40(rcx) )
  add(rdi, rcx)
  vmovups( zmm16,mem(rcx) )
  vmovups( zmm17,0x40(rcx) )
  add(rdi, rcx)
  vmovups( zmm18,mem(rcx) )
  vmovups( zmm19,0x40(rcx) )
  add(rdi, rcx)

  label(.SDONE)


  end_asm(
   : // output operands (none)
   : // input operands
     [k_iter] "m" (k_iter),
     [a]      "m" (a),
     [rs_a]   "m" (rs_a),
     [cs_a]   "m" (cs_a),
     [b]      "m" (b),
     [rs_b]   "m" (rs_b),
     [cs_b]   "m" (cs_b),
     [alpha]  "m" (alpha),
     [beta]   "m" (beta),
     [c]      "m" (c),
     [rs_c]   "m" (rs_c),
     [cs_c]   "m" (cs_c),
     [n0]     "m" (n0),
     [m0]     "m" (m0),
     [abuf]   "m" (abuf),
     [bbuf]   "m" (bbuf),
     [cbuf]   "m" (cbuf)
   : // register clobber list
   "rax", "rbx", "rcx", "rdx", "rsi", "rdi",
   "r8", "r9", "r10", "r11", "r12", "r13", "r14", "r15",
   "zmm0", "zmm1", "zmm2", "zmm3",
   "zmm4", "zmm5", "zmm6", "zmm7", "zmm8", "zmm9", "zmm10",
   "zmm11", "zmm12", "zmm13", "zmm14", "zmm15",
   "zmm16", "zmm17", "zmm18", "zmm19",
   "zmm20", "zmm21", "zmm22", "zmm23", "zmm24", "zmm25", "zmm26",
   "zmm27", "zmm28", "zmm29", "zmm30", "zmm31",
   "memory"
   )
      
}

void bli_sgemmsup_rv_zen_asm_4x32m
(
       conj_t    conja,
       conj_t    conjb,
       dim_t     m0,
       dim_t     n0,
       dim_t     k0,
       float*    restrict alpha,
       float*    restrict a, inc_t rs_a0, inc_t cs_a0,
       float*    restrict b, inc_t rs_b0, inc_t cs_b0,
       float*    restrict beta,
       float*    restrict c, inc_t rs_c0, inc_t cs_c0,
       auxinfo_t* restrict data,
       cntx_t*    restrict cntx
     )
{
  uint64_t k_iter = k0;

  uint64_t rs_a   = rs_a0;
  uint64_t cs_a   = cs_a0;
  uint64_t rs_b   = rs_b0;
  uint64_t cs_b   = cs_b0;
  uint64_t rs_c   = rs_c0;
  uint64_t cs_c   = cs_c0;

  float *abuf = a;
  float *bbuf = b;
  float *cbuf = c;

  /*Produce MRXNR outputs */
  // -------------------------------------------------------------------------
  begin_asm()

  mov(var(rs_a), r8)                 // load rs_a
  lea(mem(, r8, 4), r8)              // rs_a *= sizeof(dt)
  mov(var(rs_b), r9)                 // load rs_b
  lea(mem(, r9, 4), r9)              // rs_b *= sizeof(dt)
  mov(var(cs_a), r10)                // load cs_a
  lea(mem(, r10, 4), r10)            // cs_a *= sizeof(dt)
  mov(var(rs_c), rdi)                // load rs_c
  lea(mem(, rdi, 4), rdi)            // rs_c *= sizeof(float)


  lea(mem(r8, r8, 2), r13)           // r13 = 3*rs_a
  lea(mem(r8, r8, 4), r15)           // r15 = 5*rs_a
  vxorps( zmm0,zmm0,zmm0 )
  vxorps( zmm1,zmm1,zmm1 )
  vxorps( zmm2,zmm2,zmm2 )
  vxorps( zmm3,zmm3,zmm3 )
  vxorps( zmm4,zmm4,zmm4 )
  vxorps( zmm5,zmm5,zmm5 )
  vxorps( zmm6,zmm6,zmm6 )
  vxorps( zmm7,zmm7,zmm7 )
  vxorps( zmm8,zmm8,zmm8 )
  vxorps( zmm9,zmm9,zmm9 )
  vxorps( zmm10,zmm10,zmm10 )
  vxorps( zmm11,zmm11,zmm11 )
  vxorps( zmm12,zmm12,zmm12 )
  vxorps( zmm13,zmm13,zmm13 )
  vxorps( zmm14,zmm14,zmm14 )
  vxorps( zmm15,zmm15,zmm15 )
  vxorps( zmm16,zmm16,zmm16 )
  vxorps( zmm17,zmm17,zmm17 )
  vxorps( zmm18,zmm18,zmm18 )
  vxorps( zmm19,zmm19,zmm19 )
  vxorps( zmm20,zmm20,zmm20 )
  vxorps( zmm21,zmm21,zmm21 )
  vxorps( zmm22,zmm22,zmm22 )
  vxorps( zmm23,zmm23,zmm23 )
  vxorps( zmm24,zmm24,zmm24 )
  vxorps( zmm25,zmm25,zmm25 )
  vxorps( zmm26,zmm26,zmm26 )
  vxorps( zmm27,zmm27,zmm27 )
  vxorps( zmm28,zmm28,zmm28 )
  vxorps( zmm29,zmm29,zmm29 )
  vxorps( zmm30,zmm30,zmm30 )
  vxorps( zmm31,zmm31,zmm31 )
  mov( var(abuf), rax )   // load address of a
  mov( var(bbuf), rbx )   // load address of b
  mov( var(cbuf), rcx )   // load address of c

  mov( var( k_iter ), rsi )   // load k_iter
  test( rsi,rsi )
  mov( var(alpha), rdx )  // load address of alpha
  vbroadcastss( (rdx),zmm28 )

  label( .K_LOOP_ITER )

  vmovups( (rbx),zmm0 )
  vmovups( 0x40(rbx),zmm1 )
  vbroadcastss( (rax),zmm2 )
  vfmadd231ps( zmm0,zmm2,zmm4 )
  vfmadd231ps( zmm1,zmm2,zmm5 )
  vbroadcastss(mem(rax, r8, 1),zmm29 )
  vfmadd231ps( zmm0,zmm29,zmm6 )
  vfmadd231ps( zmm1,zmm29,zmm7 )
  vbroadcastss(mem(rax, r8, 2),zmm30 )
  vfmadd231ps( zmm0,zmm30,zmm8 )
  vfmadd231ps( zmm1,zmm30,zmm9 )
  vbroadcastss(mem(rax, r13, 1 ),zmm31 )
  vfmadd231ps( zmm0,zmm31,zmm10 )
  vfmadd231ps( zmm1,zmm31,zmm11 )
  
  add( r9, rbx)
  add( r10, rax)
  dec( rsi )
  jne( .K_LOOP_ITER )

  // Scale by alpha
  vmulps( zmm28,zmm4,zmm4 )
  vmulps( zmm28,zmm5,zmm5 )
  vmulps( zmm28,zmm6,zmm6 )
  vmulps( zmm28,zmm7,zmm7 )
  vmulps( zmm28,zmm8,zmm8 )
  vmulps( zmm28,zmm9,zmm9 )
  vmulps( zmm28,zmm10,zmm10 )
  vmulps( zmm28,zmm11,zmm11 )
  
  mov( var(beta), rdx )  // load address of beta
  vbroadcastss( (rdx),zmm28 )

  vxorps(xmm0, xmm0, xmm0)
  vucomiss(xmm0, xmm28)
  je(.SBETAZERO)

  vmovups( (rcx),zmm0 )
  vfmadd231ps( zmm28,zmm0,zmm4 )
  vmovups( zmm4,(rcx) )
  vmovups( 0x40(rcx),zmm0 )
  vfmadd231ps( zmm28,zmm0,zmm5 )
  vmovups( zmm5,0x40(rcx) )
  add(rdi, rcx)
  vmovups( mem(rcx),zmm0 )
  vfmadd231ps( zmm28,zmm0,zmm6 )
  vmovups( zmm6,mem(rcx) )
  vmovups( 0x40(rcx),zmm0 )
  vfmadd231ps( zmm28,zmm0,zmm7 )
  vmovups( zmm7,0x40(rcx) )
  add(rdi, rcx)
  vmovups( mem(rcx),zmm0 )
  vfmadd231ps( zmm28,zmm0,zmm8 )
  vmovups( zmm8,mem(rcx) )
  vmovups( 0x40(rcx),zmm0 )
  vfmadd231ps( zmm28,zmm0,zmm9 )
  vmovups( zmm9,0x40(rcx) )
  add(rdi, rcx)
  vmovups( mem(rcx),zmm0 )
  vfmadd231ps( zmm28,zmm0,zmm10 )
  vmovups( zmm10,mem(rcx) )
  vmovups( 0x40(rcx),zmm0 )
  vfmadd231ps( zmm28,zmm0,zmm11 )
  vmovups( zmm11,0x40(rcx) )
  add(rdi, rcx)
  jmp(.SDONE)

  label(.SBETAZERO)

  vmovups( zmm4,(rcx) )
  vmovups( zmm5,0x40(rcx) )
  add(rdi, rcx)
  vmovups( zmm6,mem(rcx) )
  vmovups( zmm7,0x40(rcx) )
  add(rdi, rcx)
  vmovups( zmm8,mem(rcx) )
  vmovups( zmm9,0x40(rcx) )
  add(rdi, rcx)
  vmovups( zmm10,mem(rcx) )
  vmovups( zmm11,0x40(rcx) )
  add(rdi, rcx)

  label(.SDONE)


  end_asm(
   : // output operands (none)
   : // input operands
     [k_iter] "m" (k_iter),
     [a]      "m" (a),
     [rs_a]   "m" (rs_a),
     [cs_a]   "m" (cs_a),
     [b]      "m" (b),
     [rs_b]   "m" (rs_b),
     [cs_b]   "m" (cs_b),
     [alpha]  "m" (alpha),
     [beta]   "m" (beta),
     [c]      "m" (c),
     [rs_c]   "m" (rs_c),
     [cs_c]   "m" (cs_c),
     [n0]     "m" (n0),
     [m0]     "m" (m0),
     [abuf]   "m" (abuf),
     [bbuf]   "m" (bbuf),
     [cbuf]   "m" (cbuf)
   : // register clobber list
   "rax", "rbx", "rcx", "rdx", "rsi", "rdi",
   "r8", "r9", "r10", "r11", "r12", "r13", "r14", "r15",
   "zmm0", "zmm1", "zmm2", "zmm3",
   "zmm4", "zmm5", "zmm6", "zmm7", "zmm8", "zmm9", "zmm10",
   "zmm11", "zmm12", "zmm13", "zmm14", "zmm15",
   "zmm16", "zmm17", "zmm18", "zmm19",
   "zmm20", "zmm21", "zmm22", "zmm23", "zmm24", "zmm25", "zmm26",
   "zmm27", "zmm28", "zmm29", "zmm30", "zmm31",
   "memory"
   )

}

void bli_sgemmsup_rv_zen_asm_2x32m
(
       conj_t    conja,
       conj_t    conjb,
       dim_t     m0,
       dim_t     n0,
       dim_t     k0,
       float*    restrict alpha,
       float*    restrict a, inc_t rs_a0, inc_t cs_a0,
       float*    restrict b, inc_t rs_b0, inc_t cs_b0,
       float*    restrict beta,
       float*    restrict c, inc_t rs_c0, inc_t cs_c0,
       auxinfo_t* restrict data,
       cntx_t*    restrict cntx
     )
{
  uint64_t k_iter = k0;

  uint64_t rs_a   = rs_a0;
  uint64_t cs_a   = cs_a0;
  uint64_t rs_b   = rs_b0;
  uint64_t cs_b   = cs_b0;
  uint64_t rs_c   = rs_c0;
  uint64_t cs_c   = cs_c0;

  float *abuf = a;
  float *bbuf = b;
  float *cbuf = c;

  /*Produce MRXNR outputs */
  // -------------------------------------------------------------------------
  begin_asm()

  mov(var(rs_a), r8)                 // load rs_a
  lea(mem(, r8, 4), r8)              // rs_a *= sizeof(dt)
  mov(var(rs_b), r9)                 // load rs_b
  lea(mem(, r9, 4), r9)              // rs_b *= sizeof(dt)
  mov(var(cs_a), r10)                // load cs_a
  lea(mem(, r10, 4), r10)            // cs_a *= sizeof(dt)
  mov(var(rs_c), rdi)                // load rs_c
  lea(mem(, rdi, 4), rdi)            // rs_c *= sizeof(float)


  lea(mem(r8, r8, 2), r13)           // r13 = 3*rs_a
  lea(mem(r8, r8, 4), r15)           // r15 = 5*rs_a
  vxorps( zmm0,zmm0,zmm0 )
  vxorps( zmm1,zmm1,zmm1 )
  vxorps( zmm2,zmm2,zmm2 )
  vxorps( zmm3,zmm3,zmm3 )
  vxorps( zmm4,zmm4,zmm4 )
  vxorps( zmm5,zmm5,zmm5 )
  vxorps( zmm6,zmm6,zmm6 )
  vxorps( zmm7,zmm7,zmm7 )
  vxorps( zmm8,zmm8,zmm8 )
  vxorps( zmm9,zmm9,zmm9 )
  vxorps( zmm10,zmm10,zmm10 )
  vxorps( zmm11,zmm11,zmm11 )
  vxorps( zmm12,zmm12,zmm12 )
  vxorps( zmm13,zmm13,zmm13 )
  vxorps( zmm14,zmm14,zmm14 )
  vxorps( zmm15,zmm15,zmm15 )
  vxorps( zmm16,zmm16,zmm16 )
  vxorps( zmm17,zmm17,zmm17 )
  vxorps( zmm18,zmm18,zmm18 )
  vxorps( zmm19,zmm19,zmm19 )
  vxorps( zmm20,zmm20,zmm20 )
  vxorps( zmm21,zmm21,zmm21 )
  vxorps( zmm22,zmm22,zmm22 )
  vxorps( zmm23,zmm23,zmm23 )
  vxorps( zmm24,zmm24,zmm24 )
  vxorps( zmm25,zmm25,zmm25 )
  vxorps( zmm26,zmm26,zmm26 )
  vxorps( zmm27,zmm27,zmm27 )
  vxorps( zmm28,zmm28,zmm28 )
  vxorps( zmm29,zmm29,zmm29 )
  vxorps( zmm30,zmm30,zmm30 )
  vxorps( zmm31,zmm31,zmm31 )
  mov( var(abuf), rax )   // load address of a
  mov( var(bbuf), rbx )   // load address of b
  mov( var(cbuf), rcx )   // load address of c

  mov( var( k_iter ), rsi )   // load k_iter
  test( rsi,rsi )
  mov( var(alpha), rdx )  // load address of alpha
  vbroadcastss( (rdx),zmm28 )

  label( .K_LOOP_ITER )

  vmovups( (rbx),zmm0 )
  vmovups( 0x40(rbx),zmm1 )
  vbroadcastss( (rax),zmm2 )
  vfmadd231ps( zmm0,zmm2,zmm4 )
  vfmadd231ps( zmm1,zmm2,zmm5 )
  vbroadcastss(mem(rax, r8, 1),zmm29 )
  vfmadd231ps( zmm0,zmm29,zmm6 )
  vfmadd231ps( zmm1,zmm29,zmm7 )
  
  add( r9, rbx)
  add( r10, rax)
  dec( rsi )
  jne( .K_LOOP_ITER )

  // Scale by alpha
  vmulps( zmm28,zmm4,zmm4 )
  vmulps( zmm28,zmm5,zmm5 )
  vmulps( zmm28,zmm6,zmm6 )
  vmulps( zmm28,zmm7,zmm7 )
  
  mov( var(beta), rdx )  // load address of beta
  vbroadcastss( (rdx),zmm28 )

  vxorps(xmm0, xmm0, xmm0)
  vucomiss(xmm0, xmm28)
  je(.SBETAZERO)

  vmovups( (rcx),zmm0 )
  vfmadd231ps( zmm28,zmm0,zmm4 )
  vmovups( zmm4,(rcx) )
  vmovups( 0x40(rcx),zmm0 )
  vfmadd231ps( zmm28,zmm0,zmm5 )
  vmovups( zmm5,0x40(rcx) )
  add(rdi, rcx)
  vmovups( mem(rcx),zmm0 )
  vfmadd231ps( zmm28,zmm0,zmm6 )
  vmovups( zmm6,mem(rcx) )
  vmovups( 0x40(rcx),zmm0 )
  vfmadd231ps( zmm28,zmm0,zmm7 )
  vmovups( zmm7,0x40(rcx) )
  add(rdi, rcx)
  jmp(.SDONE)

  label(.SBETAZERO)

  vmovups( zmm4,(rcx) )
  vmovups( zmm5,0x40(rcx) )
  add(rdi, rcx)
  vmovups( zmm6,mem(rcx) )
  vmovups( zmm7,0x40(rcx) )
  add(rdi, rcx)

  label(.SDONE)


  end_asm(
   : // output operands (none)
   : // input operands
     [k_iter] "m" (k_iter),
     [a]      "m" (a),
     [rs_a]   "m" (rs_a),
     [cs_a]   "m" (cs_a),
     [b]      "m" (b),
     [rs_b]   "m" (rs_b),
     [cs_b]   "m" (cs_b),
     [alpha]  "m" (alpha),
     [beta]   "m" (beta),
     [c]      "m" (c),
     [rs_c]   "m" (rs_c),
     [cs_c]   "m" (cs_c),
     [n0]     "m" (n0),
     [m0]     "m" (m0),
     [abuf]   "m" (abuf),
     [bbuf]   "m" (bbuf),
     [cbuf]   "m" (cbuf)
   : // register clobber list
   "rax", "rbx", "rcx", "rdx", "rsi", "rdi",
   "r8", "r9", "r10", "r11", "r12", "r13", "r14", "r15",
   "zmm0", "zmm1", "zmm2", "zmm3",
   "zmm4", "zmm5", "zmm6", "zmm7", "zmm8", "zmm9", "zmm10",
   "zmm11", "zmm12", "zmm13", "zmm14", "zmm15",
   "zmm16", "zmm17", "zmm18", "zmm19",
   "zmm20", "zmm21", "zmm22", "zmm23", "zmm24", "zmm25", "zmm26",
   "zmm27", "zmm28", "zmm29", "zmm30", "zmm31",
   "memory"
   )

}

void bli_sgemmsup_rv_zen_asm_1x32m
(
       conj_t    conja,
       conj_t    conjb,
       dim_t     m0,
       dim_t     n0,
       dim_t     k0,
       float*    restrict alpha,
       float*    restrict a, inc_t rs_a0, inc_t cs_a0,
       float*    restrict b, inc_t rs_b0, inc_t cs_b0,
       float*    restrict beta,
       float*    restrict c, inc_t rs_c0, inc_t cs_c0,
       auxinfo_t* restrict data,
       cntx_t*    restrict cntx
     )
{
  uint64_t k_iter = k0;

  uint64_t rs_a   = rs_a0;
  uint64_t cs_a   = cs_a0;
  uint64_t rs_b   = rs_b0;
  uint64_t cs_b   = cs_b0;
  uint64_t rs_c   = rs_c0;
  uint64_t cs_c   = cs_c0;

  float *abuf = a;
  float *bbuf = b;
  float *cbuf = c;

  /*Produce MRXNR outputs */
  // -------------------------------------------------------------------------
  begin_asm()

  mov(var(rs_a), r8)                 // load rs_a
  lea(mem(, r8, 4), r8)              // rs_a *= sizeof(dt)
  mov(var(rs_b), r9)                 // load rs_b
  lea(mem(, r9, 4), r9)              // rs_b *= sizeof(dt)
  mov(var(cs_a), r10)                // load cs_a
  lea(mem(, r10, 4), r10)            // cs_a *= sizeof(dt)
  mov(var(rs_c), rdi)                // load rs_c
  lea(mem(, rdi, 4), rdi)            // rs_c *= sizeof(float)


  lea(mem(r8, r8, 2), r13)           // r13 = 3*rs_a
  lea(mem(r8, r8, 4), r15)           // r15 = 5*rs_a
  vxorps( zmm0,zmm0,zmm0 )
  vxorps( zmm1,zmm1,zmm1 )
  vxorps( zmm2,zmm2,zmm2 )
  vxorps( zmm3,zmm3,zmm3 )
  vxorps( zmm4,zmm4,zmm4 )
  vxorps( zmm5,zmm5,zmm5 )
  vxorps( zmm6,zmm6,zmm6 )
  vxorps( zmm7,zmm7,zmm7 )
  vxorps( zmm8,zmm8,zmm8 )
  vxorps( zmm9,zmm9,zmm9 )
  vxorps( zmm10,zmm10,zmm10 )
  vxorps( zmm11,zmm11,zmm11 )
  vxorps( zmm12,zmm12,zmm12 )
  vxorps( zmm13,zmm13,zmm13 )
  vxorps( zmm14,zmm14,zmm14 )
  vxorps( zmm15,zmm15,zmm15 )
  vxorps( zmm16,zmm16,zmm16 )
  vxorps( zmm17,zmm17,zmm17 )
  vxorps( zmm18,zmm18,zmm18 )
  vxorps( zmm19,zmm19,zmm19 )
  vxorps( zmm20,zmm20,zmm20 )
  vxorps( zmm21,zmm21,zmm21 )
  vxorps( zmm22,zmm22,zmm22 )
  vxorps( zmm23,zmm23,zmm23 )
  vxorps( zmm24,zmm24,zmm24 )
  vxorps( zmm25,zmm25,zmm25 )
  vxorps( zmm26,zmm26,zmm26 )
  vxorps( zmm27,zmm27,zmm27 )
  vxorps( zmm28,zmm28,zmm28 )
  vxorps( zmm29,zmm29,zmm29 )
  vxorps( zmm30,zmm30,zmm30 )
  vxorps( zmm31,zmm31,zmm31 )
  mov( var(abuf), rax )   // load address of a
  mov( var(bbuf), rbx )   // load address of b
  mov( var(cbuf), rcx )   // load address of c

  mov( var( k_iter ), rsi )   // load k_iter
  test( rsi,rsi )
  mov( var(alpha), rdx )  // load address of alpha
  vbroadcastss( (rdx),zmm28 )

  label( .K_LOOP_ITER )

  vmovups( (rbx),zmm0 )
  vmovups( 0x40(rbx),zmm1 )
  vbroadcastss( (rax),zmm2 )
  vfmadd231ps( zmm0,zmm2,zmm4 )
  vfmadd231ps( zmm1,zmm2,zmm5 )
  
  add( r9, rbx)
  add( r10, rax)
  dec( rsi )
  jne( .K_LOOP_ITER )

  // Scale by alpha
  vmulps( zmm28,zmm4,zmm4 )
  vmulps( zmm28,zmm5,zmm5 )
  
  mov( var(beta), rdx )  // load address of beta
  vbroadcastss( (rdx),zmm28 )

  vxorps(xmm0, xmm0, xmm0)
  vucomiss(xmm0, xmm28)
  je(.SBETAZERO)

  vmovups( (rcx),zmm0 )
  vfmadd231ps( zmm28,zmm0,zmm4 )
  vmovups( zmm4,(rcx) )
  vmovups( 0x40(rcx),zmm0 )
  vfmadd231ps( zmm28,zmm0,zmm5 )
  vmovups( zmm5,0x40(rcx) )
  add(rdi, rcx)
  jmp(.SDONE)

  label(.SBETAZERO)

  vmovups( zmm4,(rcx) )
  vmovups( zmm5,0x40(rcx) )
  add(rdi, rcx)

  label(.SDONE)


  end_asm(
   : // output operands (none)
   : // input operands
     [k_iter] "m" (k_iter),
     [a]      "m" (a),
     [rs_a]   "m" (rs_a),
     [cs_a]   "m" (cs_a),
     [b]      "m" (b),
     [rs_b]   "m" (rs_b),
     [cs_b]   "m" (cs_b),
     [alpha]  "m" (alpha),
     [beta]   "m" (beta),
     [c]      "m" (c),
     [rs_c]   "m" (rs_c),
     [cs_c]   "m" (cs_c),
     [n0]     "m" (n0),
     [m0]     "m" (m0),
     [abuf]   "m" (abuf),
     [bbuf]   "m" (bbuf),
     [cbuf]   "m" (cbuf)
   : // register clobber list
   "rax", "rbx", "rcx", "rdx", "rsi", "rdi",
   "r8", "r9", "r10", "r11", "r12", "r13", "r14", "r15",
   "zmm0", "zmm1", "zmm2", "zmm3",
   "zmm4", "zmm5", "zmm6", "zmm7", "zmm8", "zmm9", "zmm10",
   "zmm11", "zmm12", "zmm13", "zmm14", "zmm15",
   "zmm16", "zmm17", "zmm18", "zmm19",
   "zmm20", "zmm21", "zmm22", "zmm23", "zmm24", "zmm25", "zmm26",
   "zmm27", "zmm28", "zmm29", "zmm30", "zmm31",
   "memory"
   )

}

void bli_sgemmsup_rv_zen_asm_12x16m
(
       conj_t    conja,
       conj_t    conjb,
       dim_t     m0,
       dim_t     n0,
       dim_t     k0,
       float*    restrict alpha,
       float*    restrict a, inc_t rs_a0, inc_t cs_a0,
       float*    restrict b, inc_t rs_b0, inc_t cs_b0,
       float*    restrict beta,
       float*    restrict c, inc_t rs_c0, inc_t cs_c0,
       auxinfo_t* restrict data,
       cntx_t*    restrict cntx
     )
{
  uint64_t k_iter = k0;

  uint64_t m_iter = m0 / 12;
  uint64_t m_left = m0 % 12;

  uint64_t rs_a   = rs_a0;
  uint64_t cs_a   = cs_a0;
  uint64_t rs_b   = rs_b0;
  uint64_t cs_b   = cs_b0;
  uint64_t rs_c   = rs_c0;
  uint64_t cs_c   = cs_c0;

  float *abuf = a;
  float *bbuf = b;
  float *cbuf = c;

  if ( m_iter == 0 ) goto consider_edge_cases;

  /*Produce MRXNR outputs */
  // -------------------------------------------------------------------------
  begin_asm()

  mov(var(rs_a), r8)                 // load rs_a
  lea(mem(, r8, 4), r8)              // rs_a *= sizeof(dt)
  mov(var(rs_b), r9)                 // load rs_b
  lea(mem(, r9, 4), r9)              // rs_b *= sizeof(dt)
  mov(var(cs_a), r10)                // load cs_a
  lea(mem(, r10, 4), r10)            // cs_a *= sizeof(dt)
  mov(var(rs_c), rdi)                // load rs_c
  lea(mem(, rdi, 4), rdi)            // rs_c *= sizeof(float)


  lea(mem(r8, r8, 2), r13)           // r13 = 3*rs_a
  lea(mem(r8, r8, 4), r15)           // r15 = 5*rs_a
  lea(mem(r13, r13, 1), r12)         // r12 = 6*rs_a
  sub(r12,r10)                       // r10 = cs_a-6*rs_a

  mov( var( m_iter ), r11 )        // load m_iter

  label( .M_LOOP_ITER )

  vxorps( zmm0,zmm0,zmm0 )
  vxorps( zmm1,zmm1,zmm1 )
  vxorps( zmm2,zmm2,zmm2 )
  vxorps( zmm3,zmm3,zmm3 )
  vxorps( zmm4,zmm4,zmm4 )
  vxorps( zmm5,zmm5,zmm5 )
  vxorps( zmm6,zmm6,zmm6 )
  vxorps( zmm7,zmm7,zmm7 )
  vxorps( zmm8,zmm8,zmm8 )
  vxorps( zmm9,zmm9,zmm9 )
  vxorps( zmm10,zmm10,zmm10 )
  vxorps( zmm11,zmm11,zmm11 )
  vxorps( zmm12,zmm12,zmm12 )
  vxorps( zmm13,zmm13,zmm13 )
  vxorps( zmm14,zmm14,zmm14 )
  vxorps( zmm15,zmm15,zmm15 )
  vxorps( zmm16,zmm16,zmm16 )
  vxorps( zmm17,zmm17,zmm17 )
  vxorps( zmm18,zmm18,zmm18 )
  vxorps( zmm19,zmm19,zmm19 )
  vxorps( zmm20,zmm20,zmm20 )
  vxorps( zmm21,zmm21,zmm21 )
  vxorps( zmm22,zmm22,zmm22 )
  vxorps( zmm23,zmm23,zmm23 )
  vxorps( zmm24,zmm24,zmm24 )
  vxorps( zmm25,zmm25,zmm25 )
  vxorps( zmm26,zmm26,zmm26 )
  vxorps( zmm27,zmm27,zmm27 )
  vxorps( zmm28,zmm28,zmm28 )
  vxorps( zmm29,zmm29,zmm29 )
  vxorps( zmm30,zmm30,zmm30 )
  vxorps( zmm31,zmm31,zmm31 )
  mov( var(abuf), rax )   // load address of a
  mov( var(bbuf), rbx )   // load address of b
  mov( var(cbuf), rcx )   // load address of c

  mov( var( k_iter ), rsi )   // load k_iter
  test( rsi,rsi )
  mov( var(alpha), rdx )  // load address of alpha
  vbroadcastss( (rdx),zmm28 )

  label( .K_LOOP_ITER )

  vmovups( (rbx),zmm0 )
  vbroadcastss( (rax),zmm2 )
  vfmadd231ps( zmm0,zmm2,zmm4 )
  vbroadcastss(mem(rax, r8, 1),zmm29 )
  vfmadd231ps( zmm0,zmm29,zmm6 )
  vbroadcastss(mem(rax, r8, 2),zmm30 )
  vfmadd231ps( zmm0,zmm30,zmm8 )
  vbroadcastss(mem(rax, r13, 1 ),zmm31 )
  vfmadd231ps( zmm0,zmm31,zmm10 )
  vbroadcastss(mem(rax, r8, 4),zmm2 )
  vfmadd231ps( zmm0,zmm2,zmm12 )
  vbroadcastss(mem(rax, r15, 1 ),zmm29 )
  vfmadd231ps( zmm0,zmm29,zmm14 )
  add( r12, rax)
  vbroadcastss(mem(rax       ),zmm30 )
  vfmadd231ps( zmm0,zmm30,zmm16 )
  vbroadcastss(mem(rax, r8, 1),zmm31 )
  vfmadd231ps( zmm0,zmm31,zmm18 )
  vbroadcastss(mem(rax, r8, 2),zmm2 )
  vfmadd231ps( zmm0,zmm2,zmm20 )
  vbroadcastss(mem(rax, r13, 1 ),zmm29 )
  vfmadd231ps( zmm0,zmm29,zmm22 )
  vbroadcastss(mem(rax, r8, 4),zmm30 )
  vfmadd231ps( zmm0,zmm30,zmm24 )
  vbroadcastss(mem(rax, r15, 1 ),zmm31 )
  vfmadd231ps( zmm0,zmm31,zmm26 )
  
  add( r9, rbx)
  add( r10, rax)
  dec( rsi )
  jne( .K_LOOP_ITER )

  // Scale by alpha
  vmulps( zmm28,zmm4,zmm4 )
  vmulps( zmm28,zmm6,zmm6 )
  vmulps( zmm28,zmm8,zmm8 )
  vmulps( zmm28,zmm10,zmm10 )
  vmulps( zmm28,zmm12,zmm12 )
  vmulps( zmm28,zmm14,zmm14 )
  vmulps( zmm28,zmm16,zmm16 )
  vmulps( zmm28,zmm18,zmm18 )
  vmulps( zmm28,zmm20,zmm20 )
  vmulps( zmm28,zmm22,zmm22 )
  vmulps( zmm28,zmm24,zmm24 )
  vmulps( zmm28,zmm26,zmm26 )
  
  mov( var(beta), rdx )  // load address of beta
  vbroadcastss( (rdx),zmm28 )

  vxorps(xmm0, xmm0, xmm0)
  vucomiss(xmm0, xmm28)
  je(.SBETAZERO)

  vmovups( (rcx),zmm0 )
  vfmadd231ps( zmm28,zmm0,zmm4 )
  vmovups( zmm4,(rcx) )
  add(rdi, rcx)
  vmovups( mem(rcx),zmm0 )
  vfmadd231ps( zmm28,zmm0,zmm6 )
  vmovups( zmm6,mem(rcx) )
  add(rdi, rcx)
  vmovups( mem(rcx),zmm0 )
  vfmadd231ps( zmm28,zmm0,zmm8 )
  vmovups( zmm8,mem(rcx) )
  add(rdi, rcx)
  vmovups( mem(rcx),zmm0 )
  vfmadd231ps( zmm28,zmm0,zmm10 )
  vmovups( zmm10,mem(rcx) )
  add(rdi, rcx)
  vmovups( mem(rcx),zmm0 )
  vfmadd231ps( zmm28,zmm0,zmm12 )
  vmovups( zmm12,mem(rcx) )
  add(rdi, rcx)
  vmovups( mem(rcx),zmm0 )
  vfmadd231ps( zmm28,zmm0,zmm14 )
  vmovups( zmm14,mem(rcx) )
  add(rdi, rcx)
  vmovups( mem(rcx),zmm0 )
  vfmadd231ps( zmm28,zmm0,zmm16 )
  vmovups( zmm16,mem(rcx) )
  add(rdi, rcx)
  vmovups( mem(rcx),zmm0 )
  vfmadd231ps( zmm28,zmm0,zmm18 )
  vmovups( zmm18,mem(rcx) )
  add(rdi, rcx)
  vmovups( mem(rcx),zmm0 )
  vfmadd231ps( zmm28,zmm0,zmm20 )
  vmovups( zmm20,mem(rcx) )
  add(rdi, rcx)
  vmovups( mem(rcx),zmm0 )
  vfmadd231ps( zmm28,zmm0,zmm22 )
  vmovups( zmm22,mem(rcx) )
  add(rdi, rcx)
  vmovups( mem(rcx),zmm0 )
  vfmadd231ps( zmm28,zmm0,zmm24 )
  vmovups( zmm24,mem(rcx) )
  add(rdi, rcx)
  vmovups( mem(rcx),zmm0 )
  vfmadd231ps( zmm28,zmm0,zmm26 )
  vmovups( zmm26,mem(rcx) )
  add(rdi, rcx)
  jmp(.SDONE)

  label(.SBETAZERO)

  vmovups( zmm4,(rcx) )
  add(rdi, rcx)
  vmovups( zmm6,mem(rcx) )
  add(rdi, rcx)
  vmovups( zmm8,mem(rcx) )
  add(rdi, rcx)
  vmovups( zmm10,mem(rcx) )
  add(rdi, rcx)
  vmovups( zmm12,mem(rcx) )
  add(rdi, rcx)
  vmovups( zmm14,mem(rcx) )
  add(rdi, rcx)
  vmovups( zmm16,mem(rcx) )
  add(rdi, rcx)
  vmovups( zmm18,mem(rcx) )
  add(rdi, rcx)
  vmovups( zmm20,mem(rcx) )
  add(rdi, rcx)
  vmovups( zmm22,mem(rcx) )
  add(rdi, rcx)
  vmovups( zmm24,mem(rcx) )
  add(rdi, rcx)
  vmovups( zmm26,mem(rcx) )
  add(rdi, rcx)

  label(.SDONE)

  lea(mem(, r8, 4), rdx )            // rs_a * 4
  lea(mem(rdx, r8, 8), rdx )         // rdx = rs_a * 12
  mov( var(abuf), rax )              // load address of a
  add( rdx, rax )                    // a += rs_a * MR
  mov( rax, var(abuf) )              // store updated a

  lea(mem(, rdi, 4), rdx )           // rs_c * 4
  lea(mem(rdx, rdi, 8), rdx )        // rdx = rs_c * 12
  mov( var(cbuf), rcx )              // load address of c
  add( rdx, rcx )                    // c += rs_c * MR
  mov( rcx, var(cbuf) )              // store updated c

  dec( r11 )
  jne( .M_LOOP_ITER )

  end_asm(
    : // output operands (none)
    : // input operands
      [k_iter] "m" (k_iter),
      [a]      "m" (a),
      [rs_a]   "m" (rs_a),
      [cs_a]   "m" (cs_a),
      [b]      "m" (b),
      [rs_b]   "m" (rs_b),
      [cs_b]   "m" (cs_b),
      [alpha]  "m" (alpha),
      [beta]   "m" (beta),
      [c]      "m" (c),
      [rs_c]   "m" (rs_c),
      [cs_c]   "m" (cs_c),
      [n0]     "m" (n0),
      [m0]     "m" (m0),
      [m_iter] "m" (m_iter),
      [abuf]   "m" (abuf),
      [bbuf]   "m" (bbuf),
      [cbuf]   "m" (cbuf)
    : // register clobber list
    "rax", "rbx", "rcx", "rdx", "rsi", "rdi",
    "r8", "r9", "r10", "r11", "r12", "r13", "r14", "r15",
    "zmm0", "zmm1", "zmm2", "zmm3",
    "zmm4", "zmm5", "zmm6", "zmm7", "zmm8", "zmm9", "zmm10",
    "zmm11", "zmm12", "zmm13", "zmm14", "zmm15",
    "zmm16", "zmm17", "zmm18", "zmm19",
    "zmm20", "zmm21", "zmm22", "zmm23", "zmm24", "zmm25", "zmm26",
    "zmm27", "zmm28", "zmm29", "zmm30", "zmm31",
    "memory"
    )

  consider_edge_cases:

  // Handle edge cases in the m dimension, if they exist.
  if (m_left)
  {
    const dim_t      i_edge = m0 - ( dim_t )m_left;

    float* restrict cij = c + i_edge*rs_c;
    float* restrict ai  = a + i_edge*rs_a;
    float* restrict bj  = b;

    if (8 <= m_left)
    {
      const dim_t      mr_cur = 8;
      bli_sgemmsup_rv_zen_asm_8x16m(conja, conjb, mr_cur, n0, k0, alpha,
                                    ai, rs_a0, cs_a0,
                                    bj, rs_b0, cs_b0,
                                    beta,
                                    cij, rs_c0, cs_c0,
                                    data, cntx);
      cij += mr_cur * rs_c; ai += mr_cur * rs_a;
      m_left -= mr_cur;
    }
    if (4 <= m_left)
    {
      const dim_t      mr_cur = 4;
      bli_sgemmsup_rv_zen_asm_4x16m(conja, conjb, mr_cur, n0, k0, alpha,
                                    ai, rs_a0, cs_a0,
                                    bj, rs_b0, cs_b0,
                                    beta,
                                    cij, rs_c0, cs_c0,
                                    data, cntx);
      cij += mr_cur * rs_c; ai += mr_cur * rs_a;
      m_left -= mr_cur;
    }
    if (2 <= m_left)
    {
      const dim_t      mr_cur = 2;
      bli_sgemmsup_rv_zen_asm_2x16m(conja, conjb, mr_cur, n0, k0, alpha,
                                    ai, rs_a0, cs_a0,
                                    bj, rs_b0, cs_b0,
                                    beta,
                                    cij, rs_c0, cs_c0,
                                    data, cntx);
      cij += mr_cur * rs_c; ai += mr_cur * rs_a;
      m_left -= mr_cur;
    }
    if (1 <= m_left)
    {
      const dim_t      mr_cur = 1;
      bli_sgemmsup_rv_zen_asm_1x16m(conja, conjb, mr_cur, n0, k0, alpha,
                                    ai, rs_a0, cs_a0,
                                    bj, rs_b0, cs_b0,
                                    beta,
                                    cij, rs_c0, cs_c0,
                                    data, cntx);
      cij += mr_cur * rs_c; ai += mr_cur * rs_a;
      m_left -= mr_cur;
    }
  }
}

void bli_sgemmsup_rv_zen_asm_8x16m
(
       conj_t    conja,
       conj_t    conjb,
       dim_t     m0,
       dim_t     n0,
       dim_t     k0,
       float*    restrict alpha,
       float*    restrict a, inc_t rs_a0, inc_t cs_a0,
       float*    restrict b, inc_t rs_b0, inc_t cs_b0,
       float*    restrict beta,
       float*    restrict c, inc_t rs_c0, inc_t cs_c0,
       auxinfo_t* restrict data,
       cntx_t*    restrict cntx
     )
{
  uint64_t k_iter = k0;

  uint64_t rs_a   = rs_a0;
  uint64_t cs_a   = cs_a0;
  uint64_t rs_b   = rs_b0;
  uint64_t cs_b   = cs_b0;
  uint64_t rs_c   = rs_c0;
  uint64_t cs_c   = cs_c0;

  float *abuf = a;
  float *bbuf = b;
  float *cbuf = c;

  /*Produce MRXNR outputs */
  // -------------------------------------------------------------------------
  begin_asm()

  mov(var(rs_a), r8)                 // load rs_a
  lea(mem(, r8, 4), r8)              // rs_a *= sizeof(dt)
  mov(var(rs_b), r9)                 // load rs_b
  lea(mem(, r9, 4), r9)              // rs_b *= sizeof(dt)
  mov(var(cs_a), r10)                // load cs_a
  lea(mem(, r10, 4), r10)            // cs_a *= sizeof(dt)
  mov(var(rs_c), rdi)                // load rs_c
  lea(mem(, rdi, 4), rdi)            // rs_c *= sizeof(float)


  lea(mem(r8, r8, 2), r13)           // r13 = 3*rs_a
  lea(mem(r8, r8, 4), r15)           // r15 = 5*rs_a
  lea(mem(r13, r13, 1), r12)         // r12 = 6*rs_a
  sub(r12,r10)                       // r10 = cs_a-6*rs_a
  vxorps( zmm0,zmm0,zmm0 )
  vxorps( zmm1,zmm1,zmm1 )
  vxorps( zmm2,zmm2,zmm2 )
  vxorps( zmm3,zmm3,zmm3 )
  vxorps( zmm4,zmm4,zmm4 )
  vxorps( zmm5,zmm5,zmm5 )
  vxorps( zmm6,zmm6,zmm6 )
  vxorps( zmm7,zmm7,zmm7 )
  vxorps( zmm8,zmm8,zmm8 )
  vxorps( zmm9,zmm9,zmm9 )
  vxorps( zmm10,zmm10,zmm10 )
  vxorps( zmm11,zmm11,zmm11 )
  vxorps( zmm12,zmm12,zmm12 )
  vxorps( zmm13,zmm13,zmm13 )
  vxorps( zmm14,zmm14,zmm14 )
  vxorps( zmm15,zmm15,zmm15 )
  vxorps( zmm16,zmm16,zmm16 )
  vxorps( zmm17,zmm17,zmm17 )
  vxorps( zmm18,zmm18,zmm18 )
  vxorps( zmm19,zmm19,zmm19 )
  vxorps( zmm20,zmm20,zmm20 )
  vxorps( zmm21,zmm21,zmm21 )
  vxorps( zmm22,zmm22,zmm22 )
  vxorps( zmm23,zmm23,zmm23 )
  vxorps( zmm24,zmm24,zmm24 )
  vxorps( zmm25,zmm25,zmm25 )
  vxorps( zmm26,zmm26,zmm26 )
  vxorps( zmm27,zmm27,zmm27 )
  vxorps( zmm28,zmm28,zmm28 )
  vxorps( zmm29,zmm29,zmm29 )
  vxorps( zmm30,zmm30,zmm30 )
  vxorps( zmm31,zmm31,zmm31 )
  mov( var(abuf), rax )   // load address of a
  mov( var(bbuf), rbx )   // load address of b
  mov( var(cbuf), rcx )   // load address of c

  mov( var( k_iter ), rsi )   // load k_iter
  test( rsi,rsi )
  mov( var(alpha), rdx )  // load address of alpha
  vbroadcastss( (rdx),zmm28 )

  label( .K_LOOP_ITER )

  vmovups( (rbx),zmm0 )
  vbroadcastss( (rax),zmm2 )
  vfmadd231ps( zmm0,zmm2,zmm4 )
  vbroadcastss(mem(rax, r8, 1),zmm29 )
  vfmadd231ps( zmm0,zmm29,zmm6 )
  vbroadcastss(mem(rax, r8, 2),zmm30 )
  vfmadd231ps( zmm0,zmm30,zmm8 )
  vbroadcastss(mem(rax, r13, 1 ),zmm31 )
  vfmadd231ps( zmm0,zmm31,zmm10 )
  vbroadcastss(mem(rax, r8, 4),zmm2 )
  vfmadd231ps( zmm0,zmm2,zmm12 )
  vbroadcastss(mem(rax, r15, 1 ),zmm29 )
  vfmadd231ps( zmm0,zmm29,zmm14 )
  add( r12, rax)
  vbroadcastss(mem(rax       ),zmm30 )
  vfmadd231ps( zmm0,zmm30,zmm16 )
  vbroadcastss(mem(rax, r8, 1),zmm31 )
  vfmadd231ps( zmm0,zmm31,zmm18 )
  
  add( r9, rbx)
  add( r10, rax)
  dec( rsi )
  jne( .K_LOOP_ITER )

  // Scale by alpha
  vmulps( zmm28,zmm4,zmm4 )
  vmulps( zmm28,zmm6,zmm6 )
  vmulps( zmm28,zmm8,zmm8 )
  vmulps( zmm28,zmm10,zmm10 )
  vmulps( zmm28,zmm12,zmm12 )
  vmulps( zmm28,zmm14,zmm14 )
  vmulps( zmm28,zmm16,zmm16 )
  vmulps( zmm28,zmm18,zmm18 )
  
  mov( var(beta), rdx )  // load address of beta
  vbroadcastss( (rdx),zmm28 )

  vxorps(xmm0, xmm0, xmm0)
  vucomiss(xmm0, xmm28)
  je(.SBETAZERO)

  vmovups( (rcx),zmm0 )
  vfmadd231ps( zmm28,zmm0,zmm4 )
  vmovups( zmm4,(rcx) )
  add(rdi, rcx)
  vmovups( mem(rcx),zmm0 )
  vfmadd231ps( zmm28,zmm0,zmm6 )
  vmovups( zmm6,mem(rcx) )
  add(rdi, rcx)
  vmovups( mem(rcx),zmm0 )
  vfmadd231ps( zmm28,zmm0,zmm8 )
  vmovups( zmm8,mem(rcx) )
  add(rdi, rcx)
  vmovups( mem(rcx),zmm0 )
  vfmadd231ps( zmm28,zmm0,zmm10 )
  vmovups( zmm10,mem(rcx) )
  add(rdi, rcx)
  vmovups( mem(rcx),zmm0 )
  vfmadd231ps( zmm28,zmm0,zmm12 )
  vmovups( zmm12,mem(rcx) )
  add(rdi, rcx)
  vmovups( mem(rcx),zmm0 )
  vfmadd231ps( zmm28,zmm0,zmm14 )
  vmovups( zmm14,mem(rcx) )
  add(rdi, rcx)
  vmovups( mem(rcx),zmm0 )
  vfmadd231ps( zmm28,zmm0,zmm16 )
  vmovups( zmm16,mem(rcx) )
  add(rdi, rcx)
  vmovups( mem(rcx),zmm0 )
  vfmadd231ps( zmm28,zmm0,zmm18 )
  vmovups( zmm18,mem(rcx) )
  add(rdi, rcx)
  jmp(.SDONE)

  label(.SBETAZERO)

  vmovups( zmm4,(rcx) )
  add(rdi, rcx)
  vmovups( zmm6,mem(rcx) )
  add(rdi, rcx)
  vmovups( zmm8,mem(rcx) )
  add(rdi, rcx)
  vmovups( zmm10,mem(rcx) )
  add(rdi, rcx)
  vmovups( zmm12,mem(rcx) )
  add(rdi, rcx)
  vmovups( zmm14,mem(rcx) )
  add(rdi, rcx)
  vmovups( zmm16,mem(rcx) )
  add(rdi, rcx)
  vmovups( zmm18,mem(rcx) )
  add(rdi, rcx)

  label(.SDONE)


  end_asm(
   : // output operands (none)
   : // input operands
     [k_iter] "m" (k_iter),
     [a]      "m" (a),
     [rs_a]   "m" (rs_a),
     [cs_a]   "m" (cs_a),
     [b]      "m" (b),
     [rs_b]   "m" (rs_b),
     [cs_b]   "m" (cs_b),
     [alpha]  "m" (alpha),
     [beta]   "m" (beta),
     [c]      "m" (c),
     [rs_c]   "m" (rs_c),
     [cs_c]   "m" (cs_c),
     [n0]     "m" (n0),
     [m0]     "m" (m0),
     [abuf]   "m" (abuf),
     [bbuf]   "m" (bbuf),
     [cbuf]   "m" (cbuf)
   : // register clobber list
   "rax", "rbx", "rcx", "rdx", "rsi", "rdi",
   "r8", "r9", "r10", "r11", "r12", "r13", "r14", "r15",
   "zmm0", "zmm1", "zmm2", "zmm3",
   "zmm4", "zmm5", "zmm6", "zmm7", "zmm8", "zmm9", "zmm10",
   "zmm11", "zmm12", "zmm13", "zmm14", "zmm15",
   "zmm16", "zmm17", "zmm18", "zmm19",
   "zmm20", "zmm21", "zmm22", "zmm23", "zmm24", "zmm25", "zmm26",
   "zmm27", "zmm28", "zmm29", "zmm30", "zmm31",
   "memory"
   )

}

void bli_sgemmsup_rv_zen_asm_4x16m
(
       conj_t    conja,
       conj_t    conjb,
       dim_t     m0,
       dim_t     n0,
       dim_t     k0,
       float*    restrict alpha,
       float*    restrict a, inc_t rs_a0, inc_t cs_a0,
       float*    restrict b, inc_t rs_b0, inc_t cs_b0,
       float*    restrict beta,
       float*    restrict c, inc_t rs_c0, inc_t cs_c0,
       auxinfo_t* restrict data,
       cntx_t*    restrict cntx
     )
{
  uint64_t k_iter = k0;

  uint64_t rs_a   = rs_a0;
  uint64_t cs_a   = cs_a0;
  uint64_t rs_b   = rs_b0;
  uint64_t cs_b   = cs_b0;
  uint64_t rs_c   = rs_c0;
  uint64_t cs_c   = cs_c0;

  float *abuf = a;
  float *bbuf = b;
  float *cbuf = c;

  /*Produce MRXNR outputs */
  // -------------------------------------------------------------------------
  begin_asm()

  mov(var(rs_a), r8)                 // load rs_a
  lea(mem(, r8, 4), r8)              // rs_a *= sizeof(dt)
  mov(var(rs_b), r9)                 // load rs_b
  lea(mem(, r9, 4), r9)              // rs_b *= sizeof(dt)
  mov(var(cs_a), r10)                // load cs_a
  lea(mem(, r10, 4), r10)            // cs_a *= sizeof(dt)
  mov(var(rs_c), rdi)                // load rs_c
  lea(mem(, rdi, 4), rdi)            // rs_c *= sizeof(float)


  lea(mem(r8, r8, 2), r13)           // r13 = 3*rs_a
  lea(mem(r8, r8, 4), r15)           // r15 = 5*rs_a
  vxorps( zmm0,zmm0,zmm0 )
  vxorps( zmm1,zmm1,zmm1 )
  vxorps( zmm2,zmm2,zmm2 )
  vxorps( zmm3,zmm3,zmm3 )
  vxorps( zmm4,zmm4,zmm4 )
  vxorps( zmm5,zmm5,zmm5 )
  vxorps( zmm6,zmm6,zmm6 )
  vxorps( zmm7,zmm7,zmm7 )
  vxorps( zmm8,zmm8,zmm8 )
  vxorps( zmm9,zmm9,zmm9 )
  vxorps( zmm10,zmm10,zmm10 )
  vxorps( zmm11,zmm11,zmm11 )
  vxorps( zmm12,zmm12,zmm12 )
  vxorps( zmm13,zmm13,zmm13 )
  vxorps( zmm14,zmm14,zmm14 )
  vxorps( zmm15,zmm15,zmm15 )
  vxorps( zmm16,zmm16,zmm16 )
  vxorps( zmm17,zmm17,zmm17 )
  vxorps( zmm18,zmm18,zmm18 )
  vxorps( zmm19,zmm19,zmm19 )
  vxorps( zmm20,zmm20,zmm20 )
  vxorps( zmm21,zmm21,zmm21 )
  vxorps( zmm22,zmm22,zmm22 )
  vxorps( zmm23,zmm23,zmm23 )
  vxorps( zmm24,zmm24,zmm24 )
  vxorps( zmm25,zmm25,zmm25 )
  vxorps( zmm26,zmm26,zmm26 )
  vxorps( zmm27,zmm27,zmm27 )
  vxorps( zmm28,zmm28,zmm28 )
  vxorps( zmm29,zmm29,zmm29 )
  vxorps( zmm30,zmm30,zmm30 )
  vxorps( zmm31,zmm31,zmm31 )
  mov( var(abuf), rax )   // load address of a
  mov( var(bbuf), rbx )   // load address of b
  mov( var(cbuf), rcx )   // load address of c

  mov( var( k_iter ), rsi )   // load k_iter
  test( rsi,rsi )
  mov( var(alpha), rdx )  // load address of alpha
  vbroadcastss( (rdx),zmm28 )

  label( .K_LOOP_ITER )

  vmovups( (rbx),zmm0 )
  vbroadcastss( (rax),zmm2 )
  vfmadd231ps( zmm0,zmm2,zmm4 )
  vbroadcastss(mem(rax, r8, 1),zmm29 )
  vfmadd231ps( zmm0,zmm29,zmm6 )
  vbroadcastss(mem(rax, r8, 2),zmm30 )
  vfmadd231ps( zmm0,zmm30,zmm8 )
  vbroadcastss(mem(rax, r13, 1 ),zmm31 )
  vfmadd231ps( zmm0,zmm31,zmm10 )
  
  add( r9, rbx)
  add( r10, rax)
  dec( rsi )
  jne( .K_LOOP_ITER )

  // Scale by alpha
  vmulps( zmm28,zmm4,zmm4 )
  vmulps( zmm28,zmm6,zmm6 )
  vmulps( zmm28,zmm8,zmm8 )
  vmulps( zmm28,zmm10,zmm10 )
  
  mov( var(beta), rdx )  // load address of beta
  vbroadcastss( (rdx),zmm28 )

  vxorps(xmm0, xmm0, xmm0)
  vucomiss(xmm0, xmm28)
  je(.SBETAZERO)

  vmovups( (rcx),zmm0 )
  vfmadd231ps( zmm28,zmm0,zmm4 )
  vmovups( zmm4,(rcx) )
  add(rdi, rcx)
  vmovups( mem(rcx),zmm0 )
  vfmadd231ps( zmm28,zmm0,zmm6 )
  vmovups( zmm6,mem(rcx) )
  add(rdi, rcx)
  vmovups( mem(rcx),zmm0 )
  vfmadd231ps( zmm28,zmm0,zmm8 )
  vmovups( zmm8,mem(rcx) )
  add(rdi, rcx)
  vmovups( mem(rcx),zmm0 )
  vfmadd231ps( zmm28,zmm0,zmm10 )
  vmovups( zmm10,mem(rcx) )
  add(rdi, rcx)
  jmp(.SDONE)

  label(.SBETAZERO)

  vmovups( zmm4,(rcx) )
  add(rdi, rcx)
  vmovups( zmm6,mem(rcx) )
  add(rdi, rcx)
  vmovups( zmm8,mem(rcx) )
  add(rdi, rcx)
  vmovups( zmm10,mem(rcx) )
  add(rdi, rcx)

  label(.SDONE)


  end_asm(
   : // output operands (none)
   : // input operands
     [k_iter] "m" (k_iter),
     [a]      "m" (a),
     [rs_a]   "m" (rs_a),
     [cs_a]   "m" (cs_a),
     [b]      "m" (b),
     [rs_b]   "m" (rs_b),
     [cs_b]   "m" (cs_b),
     [alpha]  "m" (alpha),
     [beta]   "m" (beta),
     [c]      "m" (c),
     [rs_c]   "m" (rs_c),
     [cs_c]   "m" (cs_c),
     [n0]     "m" (n0),
     [m0]     "m" (m0),
     [abuf]   "m" (abuf),
     [bbuf]   "m" (bbuf),
     [cbuf]   "m" (cbuf)
   : // register clobber list
   "rax", "rbx", "rcx", "rdx", "rsi", "rdi",
   "r8", "r9", "r10", "r11", "r12", "r13", "r14", "r15",
   "zmm0", "zmm1", "zmm2", "zmm3",
   "zmm4", "zmm5", "zmm6", "zmm7", "zmm8", "zmm9", "zmm10",
   "zmm11", "zmm12", "zmm13", "zmm14", "zmm15",
   "zmm16", "zmm17", "zmm18", "zmm19",
   "zmm20", "zmm21", "zmm22", "zmm23", "zmm24", "zmm25", "zmm26",
   "zmm27", "zmm28", "zmm29", "zmm30", "zmm31",
   "memory"
   )

}

void bli_sgemmsup_rv_zen_asm_2x16m
(
       conj_t    conja,
       conj_t    conjb,
       dim_t     m0,
       dim_t     n0,
       dim_t     k0,
       float*    restrict alpha,
       float*    restrict a, inc_t rs_a0, inc_t cs_a0,
       float*    restrict b, inc_t rs_b0, inc_t cs_b0,
       float*    restrict beta,
       float*    restrict c, inc_t rs_c0, inc_t cs_c0,
       auxinfo_t* restrict data,
       cntx_t*    restrict cntx
     )
{
  uint64_t k_iter = k0;

  uint64_t rs_a   = rs_a0;
  uint64_t cs_a   = cs_a0;
  uint64_t rs_b   = rs_b0;
  uint64_t cs_b   = cs_b0;
  uint64_t rs_c   = rs_c0;
  uint64_t cs_c   = cs_c0;

  float *abuf = a;
  float *bbuf = b;
  float *cbuf = c;

  /*Produce MRXNR outputs */
  // -------------------------------------------------------------------------
  begin_asm()

  mov(var(rs_a), r8)                 // load rs_a
  lea(mem(, r8, 4), r8)              // rs_a *= sizeof(dt)
  mov(var(rs_b), r9)                 // load rs_b
  lea(mem(, r9, 4), r9)              // rs_b *= sizeof(dt)
  mov(var(cs_a), r10)                // load cs_a
  lea(mem(, r10, 4), r10)            // cs_a *= sizeof(dt)
  mov(var(rs_c), rdi)                // load rs_c
  lea(mem(, rdi, 4), rdi)            // rs_c *= sizeof(float)


  lea(mem(r8, r8, 2), r13)           // r13 = 3*rs_a
  lea(mem(r8, r8, 4), r15)           // r15 = 5*rs_a
  vxorps( zmm0,zmm0,zmm0 )
  vxorps( zmm1,zmm1,zmm1 )
  vxorps( zmm2,zmm2,zmm2 )
  vxorps( zmm3,zmm3,zmm3 )
  vxorps( zmm4,zmm4,zmm4 )
  vxorps( zmm5,zmm5,zmm5 )
  vxorps( zmm6,zmm6,zmm6 )
  vxorps( zmm7,zmm7,zmm7 )
  vxorps( zmm8,zmm8,zmm8 )
  vxorps( zmm9,zmm9,zmm9 )
  vxorps( zmm10,zmm10,zmm10 )
  vxorps( zmm11,zmm11,zmm11 )
  vxorps( zmm12,zmm12,zmm12 )
  vxorps( zmm13,zmm13,zmm13 )
  vxorps( zmm14,zmm14,zmm14 )
  vxorps( zmm15,zmm15,zmm15 )
  vxorps( zmm16,zmm16,zmm16 )
  vxorps( zmm17,zmm17,zmm17 )
  vxorps( zmm18,zmm18,zmm18 )
  vxorps( zmm19,zmm19,zmm19 )
  vxorps( zmm20,zmm20,zmm20 )
  vxorps( zmm21,zmm21,zmm21 )
  vxorps( zmm22,zmm22,zmm22 )
  vxorps( zmm23,zmm23,zmm23 )
  vxorps( zmm24,zmm24,zmm24 )
  vxorps( zmm25,zmm25,zmm25 )
  vxorps( zmm26,zmm26,zmm26 )
  vxorps( zmm27,zmm27,zmm27 )
  vxorps( zmm28,zmm28,zmm28 )
  vxorps( zmm29,zmm29,zmm29 )
  vxorps( zmm30,zmm30,zmm30 )
  vxorps( zmm31,zmm31,zmm31 )
  mov( var(abuf), rax )   // load address of a
  mov( var(bbuf), rbx )   // load address of b
  mov( var(cbuf), rcx )   // load address of c

  mov( var( k_iter ), rsi )   // load k_iter
  test( rsi,rsi )
  mov( var(alpha), rdx )  // load address of alpha
  vbroadcastss( (rdx),zmm28 )

  label( .K_LOOP_ITER )

  vmovups( (rbx),zmm0 )
  vbroadcastss( (rax),zmm2 )
  vfmadd231ps( zmm0,zmm2,zmm4 )
  vbroadcastss(mem(rax, r8, 1),zmm29 )
  vfmadd231ps( zmm0,zmm29,zmm6 )
  
  add( r9, rbx)
  add( r10, rax)
  dec( rsi )
  jne( .K_LOOP_ITER )

  // Scale by alpha
  vmulps( zmm28,zmm4,zmm4 )
  vmulps( zmm28,zmm6,zmm6 )
  
  mov( var(beta), rdx )  // load address of beta
  vbroadcastss( (rdx),zmm28 )

  vxorps(xmm0, xmm0, xmm0)
  vucomiss(xmm0, xmm28)
  je(.SBETAZERO)

  vmovups( (rcx),zmm0 )
  vfmadd231ps( zmm28,zmm0,zmm4 )
  vmovups( zmm4,(rcx) )
  add(rdi, rcx)
  vmovups( mem(rcx),zmm0 )
  vfmadd231ps( zmm28,zmm0,zmm6 )
  vmovups( zmm6,mem(rcx) )
  add(rdi, rcx)
  jmp(.SDONE)

  label(.SBETAZERO)

  vmovups( zmm4,(rcx) )
  add(rdi, rcx)
  vmovups( zmm6,mem(rcx) )
  add(rdi, rcx)

  label(.SDONE)


  end_asm(
   : // output operands (none)
   : // input operands
     [k_iter] "m" (k_iter),
     [a]      "m" (a),
     [rs_a]   "m" (rs_a),
     [cs_a]   "m" (cs_a),
     [b]      "m" (b),
     [rs_b]   "m" (rs_b),
     [cs_b]   "m" (cs_b),
     [alpha]  "m" (alpha),
     [beta]   "m" (beta),
     [c]      "m" (c),
     [rs_c]   "m" (rs_c),
     [cs_c]   "m" (cs_c),
     [n0]     "m" (n0),
     [m0]     "m" (m0),
     [abuf]   "m" (abuf),
     [bbuf]   "m" (bbuf),
     [cbuf]   "m" (cbuf)
   : // register clobber list
   "rax", "rbx", "rcx", "rdx", "rsi", "rdi",
   "r8", "r9", "r10", "r11", "r12", "r13", "r14", "r15",
   "zmm0", "zmm1", "zmm2", "zmm3",
   "zmm4", "zmm5", "zmm6", "zmm7", "zmm8", "zmm9", "zmm10",
   "zmm11", "zmm12", "zmm13", "zmm14", "zmm15",
   "zmm16", "zmm17", "zmm18", "zmm19",
   "zmm20", "zmm21", "zmm22", "zmm23", "zmm24", "zmm25", "zmm26",
   "zmm27", "zmm28", "zmm29", "zmm30", "zmm31",
   "memory"
   )

}

void bli_sgemmsup_rv_zen_asm_1x16m
(
       conj_t    conja,
       conj_t    conjb,
       dim_t     m0,
       dim_t     n0,
       dim_t     k0,
       float*    restrict alpha,
       float*    restrict a, inc_t rs_a0, inc_t cs_a0,
       float*    restrict b, inc_t rs_b0, inc_t cs_b0,
       float*    restrict beta,
       float*    restrict c, inc_t rs_c0, inc_t cs_c0,
       auxinfo_t* restrict data,
       cntx_t*    restrict cntx
     )
{
  uint64_t k_iter = k0;

  uint64_t rs_a   = rs_a0;
  uint64_t cs_a   = cs_a0;
  uint64_t rs_b   = rs_b0;
  uint64_t cs_b   = cs_b0;
  uint64_t rs_c   = rs_c0;
  uint64_t cs_c   = cs_c0;

  float *abuf = a;
  float *bbuf = b;
  float *cbuf = c;

  /*Produce MRXNR outputs */
  // -------------------------------------------------------------------------
  begin_asm()

  mov(var(rs_a), r8)                 // load rs_a
  lea(mem(, r8, 4), r8)              // rs_a *= sizeof(dt)
  mov(var(rs_b), r9)                 // load rs_b
  lea(mem(, r9, 4), r9)              // rs_b *= sizeof(dt)
  mov(var(cs_a), r10)                // load cs_a
  lea(mem(, r10, 4), r10)            // cs_a *= sizeof(dt)
  mov(var(rs_c), rdi)                // load rs_c
  lea(mem(, rdi, 4), rdi)            // rs_c *= sizeof(float)


  lea(mem(r8, r8, 2), r13)           // r13 = 3*rs_a
  lea(mem(r8, r8, 4), r15)           // r15 = 5*rs_a
  vxorps( zmm0,zmm0,zmm0 )
  vxorps( zmm1,zmm1,zmm1 )
  vxorps( zmm2,zmm2,zmm2 )
  vxorps( zmm3,zmm3,zmm3 )
  vxorps( zmm4,zmm4,zmm4 )
  vxorps( zmm5,zmm5,zmm5 )
  vxorps( zmm6,zmm6,zmm6 )
  vxorps( zmm7,zmm7,zmm7 )
  vxorps( zmm8,zmm8,zmm8 )
  vxorps( zmm9,zmm9,zmm9 )
  vxorps( zmm10,zmm10,zmm10 )
  vxorps( zmm11,zmm11,zmm11 )
  vxorps( zmm12,zmm12,zmm12 )
  vxorps( zmm13,zmm13,zmm13 )
  vxorps( zmm14,zmm14,zmm14 )
  vxorps( zmm15,zmm15,zmm15 )
  vxorps( zmm16,zmm16,zmm16 )
  vxorps( zmm17,zmm17,zmm17 )
  vxorps( zmm18,zmm18,zmm18 )
  vxorps( zmm19,zmm19,zmm19 )
  vxorps( zmm20,zmm20,zmm20 )
  vxorps( zmm21,zmm21,zmm21 )
  vxorps( zmm22,zmm22,zmm22 )
  vxorps( zmm23,zmm23,zmm23 )
  vxorps( zmm24,zmm24,zmm24 )
  vxorps( zmm25,zmm25,zmm25 )
  vxorps( zmm26,zmm26,zmm26 )
  vxorps( zmm27,zmm27,zmm27 )
  vxorps( zmm28,zmm28,zmm28 )
  vxorps( zmm29,zmm29,zmm29 )
  vxorps( zmm30,zmm30,zmm30 )
  vxorps( zmm31,zmm31,zmm31 )
  mov( var(abuf), rax )   // load address of a
  mov( var(bbuf), rbx )   // load address of b
  mov( var(cbuf), rcx )   // load address of c

  mov( var( k_iter ), rsi )   // load k_iter
  test( rsi,rsi )
  mov( var(alpha), rdx )  // load address of alpha
  vbroadcastss( (rdx),zmm28 )

  label( .K_LOOP_ITER )

  vmovups( (rbx),zmm0 )
  vbroadcastss( (rax),zmm2 )
  vfmadd231ps( zmm0,zmm2,zmm4 )
  
  add( r9, rbx)
  add( r10, rax)
  dec( rsi )
  jne( .K_LOOP_ITER )

  // Scale by alpha
  vmulps( zmm28,zmm4,zmm4 )
  
  mov( var(beta), rdx )  // load address of beta
  vbroadcastss( (rdx),zmm28 )

  vxorps(xmm0, xmm0, xmm0)
  vucomiss(xmm0, xmm28)
  je(.SBETAZERO)

  vmovups( (rcx),zmm0 )
  vfmadd231ps( zmm28,zmm0,zmm4 )
  vmovups( zmm4,(rcx) )
  add(rdi, rcx)
  jmp(.SDONE)

  label(.SBETAZERO)

  vmovups( zmm4,(rcx) )
  add(rdi, rcx)

  label(.SDONE)


  end_asm(
   : // output operands (none)
   : // input operands
     [k_iter] "m" (k_iter),
     [a]      "m" (a),
     [rs_a]   "m" (rs_a),
     [cs_a]   "m" (cs_a),
     [b]      "m" (b),
     [rs_b]   "m" (rs_b),
     [cs_b]   "m" (cs_b),
     [alpha]  "m" (alpha),
     [beta]   "m" (beta),
     [c]      "m" (c),
     [rs_c]   "m" (rs_c),
     [cs_c]   "m" (cs_c),
     [n0]     "m" (n0),
     [m0]     "m" (m0),
     [abuf]   "m" (abuf),
     [bbuf]   "m" (bbuf),
     [cbuf]   "m" (cbuf)
   : // register clobber list
   "rax", "rbx", "rcx", "rdx", "rsi", "rdi",
   "r8", "r9", "r10", "r11", "r12", "r13", "r14", "r15",
   "zmm0", "zmm1", "zmm2", "zmm3",
   "zmm4", "zmm5", "zmm6", "zmm7", "zmm8", "zmm9", "zmm10",
   "zmm11", "zmm12", "zmm13", "zmm14", "zmm15",
   "zmm16", "zmm17", "zmm18", "zmm19",
   "zmm20", "zmm21", "zmm22", "zmm23", "zmm24", "zmm25", "zmm26",
   "zmm27", "zmm28", "zmm29", "zmm30", "zmm31",
   "memory"
   )
}

void bli_sgemmsup_rv_zen_asm_12x8m
(
       conj_t    conja,
       conj_t    conjb,
       dim_t     m0,
       dim_t     n0,
       dim_t     k0,
       float*    restrict alpha,
       float*    restrict a, inc_t rs_a0, inc_t cs_a0,
       float*    restrict b, inc_t rs_b0, inc_t cs_b0,
       float*    restrict beta,
       float*    restrict c, inc_t rs_c0, inc_t cs_c0,
       auxinfo_t* restrict data,
       cntx_t*    restrict cntx
     )
{
  uint64_t k_iter = k0;

  uint64_t m_iter = m0 / 12;
  uint64_t m_left = m0 % 12;

  uint64_t rs_a   = rs_a0;
  uint64_t cs_a   = cs_a0;
  uint64_t rs_b   = rs_b0;
  uint64_t cs_b   = cs_b0;
  uint64_t rs_c   = rs_c0;
  uint64_t cs_c   = cs_c0;

  float *abuf = a;
  float *bbuf = b;
  float *cbuf = c;

  if ( m_iter == 0 ) goto consider_edge_cases;

  /*Produce MRXNR outputs */
  // -------------------------------------------------------------------------
  begin_asm()

  mov(var(rs_a), r8)                 // load rs_a
  lea(mem(, r8, 4), r8)              // rs_a *= sizeof(dt)
  mov(var(rs_b), r9)                 // load rs_b
  lea(mem(, r9, 4), r9)              // rs_b *= sizeof(dt)
  mov(var(cs_a), r10)                // load cs_a
  lea(mem(, r10, 4), r10)            // cs_a *= sizeof(dt)
  mov(var(rs_c), rdi)                // load rs_c
  lea(mem(, rdi, 4), rdi)            // rs_c *= sizeof(float)


  lea(mem(r8, r8, 2), r13)           // r13 = 3*rs_a
  lea(mem(r8, r8, 4), r15)           // r15 = 5*rs_a
  lea(mem(r13, r13, 1), r12)         // r12 = 6*rs_a
  sub(r12,r10)                       // r10 = cs_a-6*rs_a

  mov( var( m_iter ), r11 )        // load m_iter

  label( .M_LOOP_ITER )

  vxorps( ymm0,ymm0,ymm0 )
  vxorps( ymm1,ymm1,ymm1 )
  vxorps( ymm2,ymm2,ymm2 )
  vxorps( ymm3,ymm3,ymm3 )
  vxorps( ymm4,ymm4,ymm4 )
  vxorps( ymm5,ymm5,ymm5 )
  vxorps( ymm6,ymm6,ymm6 )
  vxorps( ymm7,ymm7,ymm7 )
  vxorps( ymm8,ymm8,ymm8 )
  vxorps( ymm9,ymm9,ymm9 )
  vxorps( ymm10,ymm10,ymm10 )
  vxorps( ymm11,ymm11,ymm11 )
  vxorps( ymm12,ymm12,ymm12 )
  vxorps( ymm13,ymm13,ymm13 )
  vxorps( ymm14,ymm14,ymm14 )
  vxorps( ymm15,ymm15,ymm15 )
  mov( var(abuf), rax )   // load address of a
  mov( var(bbuf), rbx )   // load address of b
  mov( var(cbuf), rcx )   // load address of c

  mov( var( k_iter ), rsi )   // load k_iter
  test( rsi,rsi )
  mov( var(alpha), rdx )  // load address of alpha
  vbroadcastss( (rdx),ymm14 )

  label( .K_LOOP_ITER )

  vmovups( (rbx),ymm0 )
  vbroadcastss( (rax),ymm1 )
  vfmadd231ps( ymm0,ymm1,ymm2 )
  vbroadcastss(mem(rax, r8, 1),ymm15 )
  vfmadd231ps( ymm0,ymm15,ymm3 )
  vbroadcastss(mem(rax, r8, 2),ymm1 )
  vfmadd231ps( ymm0,ymm1,ymm4 )
  vbroadcastss(mem(rax, r13, 1 ),ymm15 )
  vfmadd231ps( ymm0,ymm15,ymm5 )
  vbroadcastss(mem(rax, r8, 4),ymm1 )
  vfmadd231ps( ymm0,ymm1,ymm6 )
  vbroadcastss(mem(rax, r15, 1 ),ymm15 )
  vfmadd231ps( ymm0,ymm15,ymm7 )
  add( r12, rax)
  vbroadcastss(mem(rax       ),ymm1 )
  vfmadd231ps( ymm0,ymm1,ymm8 )
  vbroadcastss(mem(rax, r8, 1),ymm15 )
  vfmadd231ps( ymm0,ymm15,ymm9 )
  vbroadcastss(mem(rax, r8, 2),ymm1 )
  vfmadd231ps( ymm0,ymm1,ymm10 )
  vbroadcastss(mem(rax, r13, 1 ),ymm15 )
  vfmadd231ps( ymm0,ymm15,ymm11 )
  vbroadcastss(mem(rax, r8, 4),ymm1 )
  vfmadd231ps( ymm0,ymm1,ymm12 )
  vbroadcastss(mem(rax, r15, 1 ),ymm15 )
  vfmadd231ps( ymm0,ymm15,ymm13 )
  
  add( r9, rbx)
  add( r10, rax)
  dec( rsi )
  jne( .K_LOOP_ITER )

  // Scale by alpha
  vmulps( ymm14,ymm2,ymm2 )
  vmulps( ymm14,ymm3,ymm3 )
  vmulps( ymm14,ymm4,ymm4 )
  vmulps( ymm14,ymm5,ymm5 )
  vmulps( ymm14,ymm6,ymm6 )
  vmulps( ymm14,ymm7,ymm7 )
  vmulps( ymm14,ymm8,ymm8 )
  vmulps( ymm14,ymm9,ymm9 )
  vmulps( ymm14,ymm10,ymm10 )
  vmulps( ymm14,ymm11,ymm11 )
  vmulps( ymm14,ymm12,ymm12 )
  vmulps( ymm14,ymm13,ymm13 )
  
  mov( var(beta), rdx )  // load address of beta
  vbroadcastss( (rdx),ymm14 )

  vxorps(xmm0, xmm0, xmm0)
  vucomiss(xmm0, xmm14)
  je(.SBETAZERO)

  vmovups( (rcx),ymm0 )
  vfmadd231ps( ymm14,ymm0,ymm2 )
  vmovups( ymm2,(rcx) )
  add(rdi, rcx)
  vmovups( mem(rcx),ymm0 )
  vfmadd231ps( ymm14,ymm0,ymm3 )
  vmovups( ymm3,mem(rcx) )
  add(rdi, rcx)
  vmovups( mem(rcx),ymm0 )
  vfmadd231ps( ymm14,ymm0,ymm4 )
  vmovups( ymm4,mem(rcx) )
  add(rdi, rcx)
  vmovups( mem(rcx),ymm0 )
  vfmadd231ps( ymm14,ymm0,ymm5 )
  vmovups( ymm5,mem(rcx) )
  add(rdi, rcx)
  vmovups( mem(rcx),ymm0 )
  vfmadd231ps( ymm14,ymm0,ymm6 )
  vmovups( ymm6,mem(rcx) )
  add(rdi, rcx)
  vmovups( mem(rcx),ymm0 )
  vfmadd231ps( ymm14,ymm0,ymm7 )
  vmovups( ymm7,mem(rcx) )
  add(rdi, rcx)
  vmovups( mem(rcx),ymm0 )
  vfmadd231ps( ymm14,ymm0,ymm8 )
  vmovups( ymm8,mem(rcx) )
  add(rdi, rcx)
  vmovups( mem(rcx),ymm0 )
  vfmadd231ps( ymm14,ymm0,ymm9 )
  vmovups( ymm9,mem(rcx) )
  add(rdi, rcx)
  vmovups( mem(rcx),ymm0 )
  vfmadd231ps( ymm14,ymm0,ymm10 )
  vmovups( ymm10,mem(rcx) )
  add(rdi, rcx)
  vmovups( mem(rcx),ymm0 )
  vfmadd231ps( ymm14,ymm0,ymm11 )
  vmovups( ymm11,mem(rcx) )
  add(rdi, rcx)
  vmovups( mem(rcx),ymm0 )
  vfmadd231ps( ymm14,ymm0,ymm12 )
  vmovups( ymm12,mem(rcx) )
  add(rdi, rcx)
  vmovups( mem(rcx),ymm0 )
  vfmadd231ps( ymm14,ymm0,ymm13 )
  vmovups( ymm13,mem(rcx) )
  add(rdi, rcx)
  jmp(.SDONE)

  label(.SBETAZERO)

  vmovups( ymm2,(rcx) )
  add(rdi, rcx)
  vmovups( ymm3,mem(rcx) )
  add(rdi, rcx)
  vmovups( ymm4,mem(rcx) )
  add(rdi, rcx)
  vmovups( ymm5,mem(rcx) )
  add(rdi, rcx)
  vmovups( ymm6,mem(rcx) )
  add(rdi, rcx)
  vmovups( ymm7,mem(rcx) )
  add(rdi, rcx)
  vmovups( ymm8,mem(rcx) )
  add(rdi, rcx)
  vmovups( ymm9,mem(rcx) )
  add(rdi, rcx)
  vmovups( ymm10,mem(rcx) )
  add(rdi, rcx)
  vmovups( ymm11,mem(rcx) )
  add(rdi, rcx)
  vmovups( ymm12,mem(rcx) )
  add(rdi, rcx)
  vmovups( ymm13,mem(rcx) )
  add(rdi, rcx)

  label(.SDONE)

  lea(mem(, r8, 4), rdx )            // rs_a * 4
  lea(mem(rdx, r8, 8), rdx )         // rdx = rs_a * 12
  mov( var(abuf), rax )              // load address of a
  add( rdx, rax )                    // a += rs_a * MR
  mov( rax, var(abuf) )              // store updated a

  lea(mem(, rdi, 4), rdx )           // rs_c * 4
  lea(mem(rdx, rdi, 8), rdx )        // rdx = rs_c * 12
  mov( var(cbuf), rcx )              // load address of c
  add( rdx, rcx )                    // c += rs_c * MR
  mov( rcx, var(cbuf) )              // store updated c

  dec( r11 )
  jne( .M_LOOP_ITER )

  end_asm(
    : // output operands (none)
    : // input operands
     [k_iter] "m" (k_iter),
     [a]      "m" (a),
     [rs_a]   "m" (rs_a),
     [cs_a]   "m" (cs_a),
     [b]      "m" (b),
     [rs_b]   "m" (rs_b),
     [cs_b]   "m" (cs_b),
     [alpha]  "m" (alpha),
     [beta]   "m" (beta),
     [c]      "m" (c),
     [rs_c]   "m" (rs_c),
     [cs_c]   "m" (cs_c),
     [n0]     "m" (n0),
     [m0]     "m" (m0),
     [m_iter] "m" (m_iter),
     [abuf]   "m" (abuf),
     [bbuf]   "m" (bbuf),
     [cbuf]   "m" (cbuf)
   : // register clobber list
   "rax", "rbx", "rcx", "rdx", "rsi", "rdi",
   "r8", "r9", "r10", "r11", "r12", "r13", "r14", "r15",
   "ymm0", "ymm1", "ymm2", "ymm3",
   "ymm4", "ymm5", "ymm6", "ymm7", "ymm8", "ymm9", "ymm10",
   "ymm11", "ymm12", "ymm13", "ymm14", "ymm15",
   "memory"
   )

  consider_edge_cases:

  // Handle edge cases in the m dimension, if they exist.
  if (m_left)
  {
    const dim_t      i_edge = m0 - ( dim_t )m_left;

    float* restrict cij = c + i_edge*rs_c;
    float* restrict ai  = a + i_edge*rs_a;
    float* restrict bj  = b;

    if (8 <= m_left)
    {
      const dim_t      mr_cur = 8;
      bli_sgemmsup_rv_zen_asm_8x8m(conja, conjb, mr_cur, n0, k0, alpha,
                                   ai, rs_a0, cs_a0,
                                   bj, rs_b0, cs_b0,
                                   beta,
                                   cij, rs_c0, cs_c0,
                                   data, cntx);
      cij += mr_cur * rs_c; ai += mr_cur * rs_a;
      m_left -= mr_cur;
    }
    if (4 <= m_left) {
      const dim_t      mr_cur = 4;
      bli_sgemmsup_rv_zen_asm_4x8(conja, conjb, mr_cur, n0, k0, alpha,
                                  ai, rs_a0, cs_a0,
                                  bj, rs_b0, cs_b0,
                                  beta,
                                  cij, rs_c0, cs_c0,
                                  data, cntx);
      cij += mr_cur * rs_c; ai += mr_cur * rs_a;
      m_left -= mr_cur;
    }
    if (2 <= m_left) {
      const dim_t      mr_cur = 2;
      bli_sgemmsup_rv_zen_asm_2x8(conja, conjb, mr_cur, n0, k0, alpha,
                                  ai, rs_a0, cs_a0,
                                  bj, rs_b0, cs_b0,
                                  beta,
                                  cij, rs_c0, cs_c0,
                                  data, cntx);
      cij += mr_cur * rs_c; ai += mr_cur * rs_a;
      m_left -= mr_cur;
    }
    if (1 <= m_left) {
      const dim_t      mr_cur = 1;
      bli_sgemmsup_rv_zen_asm_1x8(conja, conjb, mr_cur, n0, k0, alpha,
                                  ai, rs_a0, cs_a0,
                                  bj, rs_b0, cs_b0,
                                  beta,
                                  cij, rs_c0, cs_c0,
                                  data, cntx);
      cij += mr_cur * rs_c; ai += mr_cur * rs_a;
      m_left -= mr_cur;
    }
  }

}

void bli_sgemmsup_rv_zen_asm_8x8m
(
       conj_t    conja,
       conj_t    conjb,
       dim_t     m0,
       dim_t     n0,
       dim_t     k0,
       float*    restrict alpha,
       float*    restrict a, inc_t rs_a0, inc_t cs_a0,
       float*    restrict b, inc_t rs_b0, inc_t cs_b0,
       float*    restrict beta,
       float*    restrict c, inc_t rs_c0, inc_t cs_c0,
       auxinfo_t* restrict data,
       cntx_t*    restrict cntx
     )
{
  uint64_t k_iter = k0;

  uint64_t rs_a   = rs_a0;
  uint64_t cs_a   = cs_a0;
  uint64_t rs_b   = rs_b0;
  uint64_t cs_b   = cs_b0;
  uint64_t rs_c   = rs_c0;
  uint64_t cs_c   = cs_c0;

  float *abuf = a;
  float *bbuf = b;
  float *cbuf = c;

  /*Produce MRXNR outputs */
  // -------------------------------------------------------------------------
  begin_asm()

  mov(var(rs_a), r8)                 // load rs_a
  lea(mem(, r8, 4), r8)              // rs_a *= sizeof(dt)
  mov(var(rs_b), r9)                 // load rs_b
  lea(mem(, r9, 4), r9)              // rs_b *= sizeof(dt)
  mov(var(cs_a), r10)                // load cs_a
  lea(mem(, r10, 4), r10)            // cs_a *= sizeof(dt)
  mov(var(rs_c), rdi)                // load rs_c
  lea(mem(, rdi, 4), rdi)            // rs_c *= sizeof(float)


  lea(mem(r8, r8, 2), r13)           // r13 = 3*rs_a
  lea(mem(r8, r8, 4), r15)           // r15 = 5*rs_a
  lea(mem(r13, r13, 1), r12)         // r12 = 6*rs_a
  sub(r12,r10)                       // r10 = cs_a-6*rs_a
  vxorps( ymm0,ymm0,ymm0 )
  vxorps( ymm1,ymm1,ymm1 )
  vxorps( ymm2,ymm2,ymm2 )
  vxorps( ymm3,ymm3,ymm3 )
  vxorps( ymm4,ymm4,ymm4 )
  vxorps( ymm5,ymm5,ymm5 )
  vxorps( ymm6,ymm6,ymm6 )
  vxorps( ymm7,ymm7,ymm7 )
  vxorps( ymm8,ymm8,ymm8 )
  vxorps( ymm9,ymm9,ymm9 )
  vxorps( ymm10,ymm10,ymm10 )
  vxorps( ymm11,ymm11,ymm11 )
  vxorps( ymm12,ymm12,ymm12 )
  vxorps( ymm13,ymm13,ymm13 )
  vxorps( ymm14,ymm14,ymm14 )
  vxorps( ymm15,ymm15,ymm15 )
  mov( var(abuf), rax )   // load address of a
  mov( var(bbuf), rbx )   // load address of b
  mov( var(cbuf), rcx )   // load address of c

  mov( var( k_iter ), rsi )   // load k_iter
  test( rsi,rsi )
  mov( var(alpha), rdx )  // load address of alpha
  vbroadcastss( (rdx),ymm14 )

  label( .K_LOOP_ITER )

  vmovups( (rbx),ymm0 )
  vbroadcastss( (rax),ymm1 )
  vfmadd231ps( ymm0,ymm1,ymm2 )
  vbroadcastss(mem(rax, r8, 1),ymm15 )
  vfmadd231ps( ymm0,ymm15,ymm3 )
  vbroadcastss(mem(rax, r8, 2),ymm1 )
  vfmadd231ps( ymm0,ymm1,ymm4 )
  vbroadcastss(mem(rax, r13, 1 ),ymm15 )
  vfmadd231ps( ymm0,ymm15,ymm5 )
  vbroadcastss(mem(rax, r8, 4),ymm1 )
  vfmadd231ps( ymm0,ymm1,ymm6 )
  vbroadcastss(mem(rax, r15, 1 ),ymm15 )
  vfmadd231ps( ymm0,ymm15,ymm7 )
  add( r12, rax)
  vbroadcastss(mem(rax       ),ymm1 )
  vfmadd231ps( ymm0,ymm1,ymm8 )
  vbroadcastss(mem(rax, r8, 1),ymm15 )
  vfmadd231ps( ymm0,ymm15,ymm9 )
  
  add( r9, rbx)
  add( r10, rax)
  dec( rsi )
  jne( .K_LOOP_ITER )

  // Scale by alpha
  vmulps( ymm14,ymm2,ymm2 )
  vmulps( ymm14,ymm3,ymm3 )
  vmulps( ymm14,ymm4,ymm4 )
  vmulps( ymm14,ymm5,ymm5 )
  vmulps( ymm14,ymm6,ymm6 )
  vmulps( ymm14,ymm7,ymm7 )
  vmulps( ymm14,ymm8,ymm8 )
  vmulps( ymm14,ymm9,ymm9 )
  
  mov( var(beta), rdx )  // load address of beta
  vbroadcastss( (rdx),ymm14 )

  vxorps(xmm0, xmm0, xmm0)
  vucomiss(xmm0, xmm14)
  je(.SBETAZERO)

  vmovups( (rcx),ymm0 )
  vfmadd231ps( ymm14,ymm0,ymm2 )
  vmovups( ymm2,(rcx) )
  add(rdi, rcx)
  vmovups( mem(rcx),ymm0 )
  vfmadd231ps( ymm14,ymm0,ymm3 )
  vmovups( ymm3,mem(rcx) )
  add(rdi, rcx)
  vmovups( mem(rcx),ymm0 )
  vfmadd231ps( ymm14,ymm0,ymm4 )
  vmovups( ymm4,mem(rcx) )
  add(rdi, rcx)
  vmovups( mem(rcx),ymm0 )
  vfmadd231ps( ymm14,ymm0,ymm5 )
  vmovups( ymm5,mem(rcx) )
  add(rdi, rcx)
  vmovups( mem(rcx),ymm0 )
  vfmadd231ps( ymm14,ymm0,ymm6 )
  vmovups( ymm6,mem(rcx) )
  add(rdi, rcx)
  vmovups( mem(rcx),ymm0 )
  vfmadd231ps( ymm14,ymm0,ymm7 )
  vmovups( ymm7,mem(rcx) )
  add(rdi, rcx)
  vmovups( mem(rcx),ymm0 )
  vfmadd231ps( ymm14,ymm0,ymm8 )
  vmovups( ymm8,mem(rcx) )
  add(rdi, rcx)
  vmovups( mem(rcx),ymm0 )
  vfmadd231ps( ymm14,ymm0,ymm9 )
  vmovups( ymm9,mem(rcx) )
  add(rdi, rcx)
  jmp(.SDONE)

  label(.SBETAZERO)

  vmovups( ymm2,(rcx) )
  add(rdi, rcx)
  vmovups( ymm3,mem(rcx) )
  add(rdi, rcx)
  vmovups( ymm4,mem(rcx) )
  add(rdi, rcx)
  vmovups( ymm5,mem(rcx) )
  add(rdi, rcx)
  vmovups( ymm6,mem(rcx) )
  add(rdi, rcx)
  vmovups( ymm7,mem(rcx) )
  add(rdi, rcx)
  vmovups( ymm8,mem(rcx) )
  add(rdi, rcx)
  vmovups( ymm9,mem(rcx) )
  add(rdi, rcx)

  label(.SDONE)


  end_asm(
    : // output operands (none)
    : // input operands
     [k_iter] "m" (k_iter),
     [a]      "m" (a),
     [rs_a]   "m" (rs_a),
     [cs_a]   "m" (cs_a),
     [b]      "m" (b),
     [rs_b]   "m" (rs_b),
     [cs_b]   "m" (cs_b),
     [alpha]  "m" (alpha),
     [beta]   "m" (beta),
     [c]      "m" (c),
     [rs_c]   "m" (rs_c),
     [cs_c]   "m" (cs_c),
     [n0]     "m" (n0),
     [m0]     "m" (m0),
     [abuf]   "m" (abuf),
     [bbuf]   "m" (bbuf),
     [cbuf]   "m" (cbuf)
   : // register clobber list
   "rax", "rbx", "rcx", "rdx", "rsi", "rdi",
   "r8", "r9", "r10", "r11", "r12", "r13", "r14", "r15",
   "ymm0", "ymm1", "ymm2", "ymm3",
   "ymm4", "ymm5", "ymm6", "ymm7", "ymm8", "ymm9", "ymm10",
   "ymm11", "ymm12", "ymm13", "ymm14", "ymm15",
   "memory"
   )

}

void bli_sgemmsup_rv_zen_asm_12x4m
(
       conj_t    conja,
       conj_t    conjb,
       dim_t     m0,
       dim_t     n0,
       dim_t     k0,
       float*    restrict alpha,
       float*    restrict a, inc_t rs_a0, inc_t cs_a0,
       float*    restrict b, inc_t rs_b0, inc_t cs_b0,
       float*    restrict beta,
       float*    restrict c, inc_t rs_c0, inc_t cs_c0,
       auxinfo_t* restrict data,
       cntx_t*    restrict cntx
     )
{
  uint64_t k_iter = k0;

  uint64_t m_iter = m0 / 12;
  uint64_t m_left = m0 % 12;

  uint64_t rs_a   = rs_a0;
  uint64_t cs_a   = cs_a0;
  uint64_t rs_b   = rs_b0;
  uint64_t cs_b   = cs_b0;
  uint64_t rs_c   = rs_c0;
  uint64_t cs_c   = cs_c0;

  float *abuf = a;
  float *bbuf = b;
  float *cbuf = c;

  if ( m_iter == 0 ) goto consider_edge_cases;

  /*Produce MRXNR outputs */
  // -------------------------------------------------------------------------
  begin_asm()

  mov(var(rs_a), r8)                 // load rs_a
  lea(mem(, r8, 4), r8)              // rs_a *= sizeof(dt)
  mov(var(rs_b), r9)                 // load rs_b
  lea(mem(, r9, 4), r9)              // rs_b *= sizeof(dt)
  mov(var(cs_a), r10)                // load cs_a
  lea(mem(, r10, 4), r10)            // cs_a *= sizeof(dt)
  mov(var(rs_c), rdi)                // load rs_c
  lea(mem(, rdi, 4), rdi)            // rs_c *= sizeof(float)


  lea(mem(r8, r8, 2), r13)           // r13 = 3*rs_a
  lea(mem(r8, r8, 4), r15)           // r15 = 5*rs_a
  lea(mem(r13, r13, 1), r12)         // r12 = 6*rs_a
  sub(r12,r10)                       // r10 = cs_a-6*rs_a

  mov( var( m_iter ), r11 )        // load m_iter

  label( .M_LOOP_ITER )

  vxorps( xmm0,xmm0,xmm0 )
  vxorps( xmm1,xmm1,xmm1 )
  vxorps( xmm2,xmm2,xmm2 )
  vxorps( xmm3,xmm3,xmm3 )
  vxorps( xmm4,xmm4,xmm4 )
  vxorps( xmm5,xmm5,xmm5 )
  vxorps( xmm6,xmm6,xmm6 )
  vxorps( xmm7,xmm7,xmm7 )
  vxorps( xmm8,xmm8,xmm8 )
  vxorps( xmm9,xmm9,xmm9 )
  vxorps( xmm10,xmm10,xmm10 )
  vxorps( xmm11,xmm11,xmm11 )
  vxorps( xmm12,xmm12,xmm12 )
  vxorps( xmm13,xmm13,xmm13 )
  vxorps( xmm14,xmm14,xmm14 )
  vxorps( xmm15,xmm15,xmm15 )
  mov( var(abuf), rax )   // load address of a
  mov( var(bbuf), rbx )   // load address of b
  mov( var(cbuf), rcx )   // load address of c

  mov( var( k_iter ), rsi )   // load k_iter
  test( rsi,rsi )
  mov( var(alpha), rdx )  // load address of alpha
  vbroadcastss( (rdx),xmm14 )

  label( .K_LOOP_ITER )

  vmovups( (rbx),xmm0 )
  vbroadcastss( (rax),xmm1 )
  vfmadd231ps( xmm0,xmm1,xmm2 )
  vbroadcastss(mem(rax, r8, 1),xmm15 )
  vfmadd231ps( xmm0,xmm15,xmm3 )
  vbroadcastss(mem(rax, r8, 2),xmm1 )
  vfmadd231ps( xmm0,xmm1,xmm4 )
  vbroadcastss(mem(rax, r13, 1 ),xmm15 )
  vfmadd231ps( xmm0,xmm15,xmm5 )
  vbroadcastss(mem(rax, r8, 4),xmm1 )
  vfmadd231ps( xmm0,xmm1,xmm6 )
  vbroadcastss(mem(rax, r15, 1 ),xmm15 )
  vfmadd231ps( xmm0,xmm15,xmm7 )
  add( r12, rax)
  vbroadcastss(mem(rax       ),xmm1 )
  vfmadd231ps( xmm0,xmm1,xmm8 )
  vbroadcastss(mem(rax, r8, 1),xmm15 )
  vfmadd231ps( xmm0,xmm15,xmm9 )
  vbroadcastss(mem(rax, r8, 2),xmm1 )
  vfmadd231ps( xmm0,xmm1,xmm10 )
  vbroadcastss(mem(rax, r13, 1 ),xmm15 )
  vfmadd231ps( xmm0,xmm15,xmm11 )
  vbroadcastss(mem(rax, r8, 4),xmm1 )
  vfmadd231ps( xmm0,xmm1,xmm12 )
  vbroadcastss(mem(rax, r15, 1 ),xmm15 )
  vfmadd231ps( xmm0,xmm15,xmm13 )
  
  add( r9, rbx)
  add( r10, rax)
  dec( rsi )
  jne( .K_LOOP_ITER )

  // Scale by alpha
  vmulps( xmm14,xmm2,xmm2 )
  vmulps( xmm14,xmm3,xmm3 )
  vmulps( xmm14,xmm4,xmm4 )
  vmulps( xmm14,xmm5,xmm5 )
  vmulps( xmm14,xmm6,xmm6 )
  vmulps( xmm14,xmm7,xmm7 )
  vmulps( xmm14,xmm8,xmm8 )
  vmulps( xmm14,xmm9,xmm9 )
  vmulps( xmm14,xmm10,xmm10 )
  vmulps( xmm14,xmm11,xmm11 )
  vmulps( xmm14,xmm12,xmm12 )
  vmulps( xmm14,xmm13,xmm13 )
  
  mov( var(beta), rdx )  // load address of beta
  vbroadcastss( (rdx),xmm14 )

  vxorps(xmm0, xmm0, xmm0)
  vucomiss(xmm0, xmm14)
  je(.SBETAZERO)

  vmovups( (rcx),xmm0 )
  vfmadd231ps( xmm14,xmm0,xmm2 )
  vmovups( xmm2,(rcx) )
  add(rdi, rcx)
  vmovups( mem(rcx),xmm0 )
  vfmadd231ps( xmm14,xmm0,xmm3 )
  vmovups( xmm3,mem(rcx) )
  add(rdi, rcx)
  vmovups( mem(rcx),xmm0 )
  vfmadd231ps( xmm14,xmm0,xmm4 )
  vmovups( xmm4,mem(rcx) )
  add(rdi, rcx)
  vmovups( mem(rcx),xmm0 )
  vfmadd231ps( xmm14,xmm0,xmm5 )
  vmovups( xmm5,mem(rcx) )
  add(rdi, rcx)
  vmovups( mem(rcx),xmm0 )
  vfmadd231ps( xmm14,xmm0,xmm6 )
  vmovups( xmm6,mem(rcx) )
  add(rdi, rcx)
  vmovups( mem(rcx),xmm0 )
  vfmadd231ps( xmm14,xmm0,xmm7 )
  vmovups( xmm7,mem(rcx) )
  add(rdi, rcx)
  vmovups( mem(rcx),xmm0 )
  vfmadd231ps( xmm14,xmm0,xmm8 )
  vmovups( xmm8,mem(rcx) )
  add(rdi, rcx)
  vmovups( mem(rcx),xmm0 )
  vfmadd231ps( xmm14,xmm0,xmm9 )
  vmovups( xmm9,mem(rcx) )
  add(rdi, rcx)
  vmovups( mem(rcx),xmm0 )
  vfmadd231ps( xmm14,xmm0,xmm10 )
  vmovups( xmm10,mem(rcx) )
  add(rdi, rcx)
  vmovups( mem(rcx),xmm0 )
  vfmadd231ps( xmm14,xmm0,xmm11 )
  vmovups( xmm11,mem(rcx) )
  add(rdi, rcx)
  vmovups( mem(rcx),xmm0 )
  vfmadd231ps( xmm14,xmm0,xmm12 )
  vmovups( xmm12,mem(rcx) )
  add(rdi, rcx)
  vmovups( mem(rcx),xmm0 )
  vfmadd231ps( xmm14,xmm0,xmm13 )
  vmovups( xmm13,mem(rcx) )
  add(rdi, rcx)
  jmp(.SDONE)

  label(.SBETAZERO)

  vmovups( xmm2,(rcx) )
  add(rdi, rcx)
  vmovups( xmm3,mem(rcx) )
  add(rdi, rcx)
  vmovups( xmm4,mem(rcx) )
  add(rdi, rcx)
  vmovups( xmm5,mem(rcx) )
  add(rdi, rcx)
  vmovups( xmm6,mem(rcx) )
  add(rdi, rcx)
  vmovups( xmm7,mem(rcx) )
  add(rdi, rcx)
  vmovups( xmm8,mem(rcx) )
  add(rdi, rcx)
  vmovups( xmm9,mem(rcx) )
  add(rdi, rcx)
  vmovups( xmm10,mem(rcx) )
  add(rdi, rcx)
  vmovups( xmm11,mem(rcx) )
  add(rdi, rcx)
  vmovups( xmm12,mem(rcx) )
  add(rdi, rcx)
  vmovups( xmm13,mem(rcx) )
  add(rdi, rcx)

  label(.SDONE)

  lea(mem(, r8, 4), rdx )            // rs_a * 4
  lea(mem(rdx, r8, 8), rdx )         // rdx = rs_a * 12
  mov( var(abuf), rax )              // load address of a
  add( rdx, rax )                    // a += rs_a * MR
  mov( rax, var(abuf) )              // store updated a

  lea(mem(, rdi, 4), rdx )           // rs_c * 4
  lea(mem(rdx, rdi, 8), rdx )        // rdx = rs_c * 12
  mov( var(cbuf), rcx )              // load address of c
  add( rdx, rcx )                    // c += rs_c * MR
  mov( rcx, var(cbuf) )              // store updated c

  dec( r11 )
  jne( .M_LOOP_ITER )

  end_asm(
    : // output operands (none)
    : // input operands
     [k_iter] "m" (k_iter),
     [a]      "m" (a),
     [rs_a]   "m" (rs_a),
     [cs_a]   "m" (cs_a),
     [b]      "m" (b),
     [rs_b]   "m" (rs_b),
     [cs_b]   "m" (cs_b),
     [alpha]  "m" (alpha),
     [beta]   "m" (beta),
     [c]      "m" (c),
     [rs_c]   "m" (rs_c),
     [cs_c]   "m" (cs_c),
     [n0]     "m" (n0),
     [m0]     "m" (m0),
     [m_iter] "m" (m_iter),
     [abuf]   "m" (abuf),
     [bbuf]   "m" (bbuf),
     [cbuf]   "m" (cbuf)
   : // register clobber list
   "rax", "rbx", "rcx", "rdx", "rsi", "rdi",
   "r8", "r9", "r10", "r11", "r12", "r13", "r14", "r15",
   "ymm0", "ymm1", "ymm2", "ymm3",
   "ymm4", "ymm5", "ymm6", "ymm7", "ymm8", "ymm9", "ymm10",
   "ymm11", "ymm12", "ymm13", "ymm14", "ymm15",
   "memory"
   )

  consider_edge_cases:

  // Handle edge cases in the m dimension, if they exist.
  if (m_left)
  {
    const dim_t      i_edge = m0 - ( dim_t )m_left;

    float* restrict cij = c + i_edge*rs_c;
    float* restrict ai  = a + i_edge*rs_a;
    float* restrict bj  = b;

    if (8 <= m_left)
    {
      const dim_t      mr_cur = 8;
      bli_sgemmsup_rv_zen_asm_8x4m(conja, conjb, mr_cur, n0, k0, alpha,
                                   ai, rs_a0, cs_a0,
                                   bj, rs_b0, cs_b0,
                                   beta,
                                   cij, rs_c0, cs_c0,
                                   data, cntx);
      cij += mr_cur * rs_c; ai += mr_cur * rs_a;
      m_left -= mr_cur;
    }
    if (4 <= m_left) {
      const dim_t      mr_cur = 4;
      bli_sgemmsup_rv_zen_asm_4x4(conja, conjb, mr_cur, n0, k0, alpha,
                                  ai, rs_a0, cs_a0,
                                  bj, rs_b0, cs_b0,
                                  beta,
                                  cij, rs_c0, cs_c0,
                                  data, cntx);
      cij += mr_cur * rs_c; ai += mr_cur * rs_a;
      m_left -= mr_cur;
    }
    if (2 <= m_left) {
      const dim_t      mr_cur = 2;
      bli_sgemmsup_rv_zen_asm_2x4(conja, conjb, mr_cur, n0, k0, alpha,
                                  ai, rs_a0, cs_a0,
                                  bj, rs_b0, cs_b0,
                                  beta,
                                  cij, rs_c0, cs_c0,
                                  data, cntx);
      cij += mr_cur * rs_c; ai += mr_cur * rs_a;
      m_left -= mr_cur;
    }
    if (1 <= m_left) {
      const dim_t      mr_cur = 1;
      bli_sgemmsup_rv_zen_asm_1x4(conja, conjb, mr_cur, n0, k0, alpha,
                                  ai, rs_a0, cs_a0,
                                  bj, rs_b0, cs_b0,
                                  beta,
                                  cij, rs_c0, cs_c0,
                                  data, cntx);
      cij += mr_cur * rs_c; ai += mr_cur * rs_a;
      m_left -= mr_cur;
    }
  }
}

void bli_sgemmsup_rv_zen_asm_8x4m
(
       conj_t    conja,
       conj_t    conjb,
       dim_t     m0,
       dim_t     n0,
       dim_t     k0,
       float*    restrict alpha,
       float*    restrict a, inc_t rs_a0, inc_t cs_a0,
       float*    restrict b, inc_t rs_b0, inc_t cs_b0,
       float*    restrict beta,
       float*    restrict c, inc_t rs_c0, inc_t cs_c0,
       auxinfo_t* restrict data,
       cntx_t*    restrict cntx
     )
{
  uint64_t k_iter = k0;

  uint64_t rs_a   = rs_a0;
  uint64_t cs_a   = cs_a0;
  uint64_t rs_b   = rs_b0;
  uint64_t cs_b   = cs_b0;
  uint64_t rs_c   = rs_c0;
  uint64_t cs_c   = cs_c0;

  float *abuf = a;
  float *bbuf = b;
  float *cbuf = c;

  /*Produce MRXNR outputs */
  // -------------------------------------------------------------------------
  begin_asm()

  mov(var(rs_a), r8)                 // load rs_a
  lea(mem(, r8, 4), r8)              // rs_a *= sizeof(dt)
  mov(var(rs_b), r9)                 // load rs_b
  lea(mem(, r9, 4), r9)              // rs_b *= sizeof(dt)
  mov(var(cs_a), r10)                // load cs_a
  lea(mem(, r10, 4), r10)            // cs_a *= sizeof(dt)
  mov(var(rs_c), rdi)                // load rs_c
  lea(mem(, rdi, 4), rdi)            // rs_c *= sizeof(float)


  lea(mem(r8, r8, 2), r13)           // r13 = 3*rs_a
  lea(mem(r8, r8, 4), r15)           // r15 = 5*rs_a
  lea(mem(r13, r13, 1), r12)         // r12 = 6*rs_a
  sub(r12,r10)                       // r10 = cs_a-6*rs_a
  vxorps( xmm0,xmm0,xmm0 )
  vxorps( xmm1,xmm1,xmm1 )
  vxorps( xmm2,xmm2,xmm2 )
  vxorps( xmm3,xmm3,xmm3 )
  vxorps( xmm4,xmm4,xmm4 )
  vxorps( xmm5,xmm5,xmm5 )
  vxorps( xmm6,xmm6,xmm6 )
  vxorps( xmm7,xmm7,xmm7 )
  vxorps( xmm8,xmm8,xmm8 )
  vxorps( xmm9,xmm9,xmm9 )
  vxorps( xmm10,xmm10,xmm10 )
  vxorps( xmm11,xmm11,xmm11 )
  vxorps( xmm12,xmm12,xmm12 )
  vxorps( xmm13,xmm13,xmm13 )
  vxorps( xmm14,xmm14,xmm14 )
  vxorps( xmm15,xmm15,xmm15 )
  mov( var(abuf), rax )   // load address of a
  mov( var(bbuf), rbx )   // load address of b
  mov( var(cbuf), rcx )   // load address of c

  mov( var( k_iter ), rsi )   // load k_iter
  test( rsi,rsi )
  mov( var(alpha), rdx )  // load address of alpha
  vbroadcastss( (rdx),xmm14 )

  label( .K_LOOP_ITER )

  vmovups( (rbx),xmm0 )
  vbroadcastss( (rax),xmm1 )
  vfmadd231ps( xmm0,xmm1,xmm2 )
  vbroadcastss(mem(rax, r8, 1),xmm15 )
  vfmadd231ps( xmm0,xmm15,xmm3 )
  vbroadcastss(mem(rax, r8, 2),xmm1 )
  vfmadd231ps( xmm0,xmm1,xmm4 )
  vbroadcastss(mem(rax, r13, 1 ),xmm15 )
  vfmadd231ps( xmm0,xmm15,xmm5 )
  vbroadcastss(mem(rax, r8, 4),xmm1 )
  vfmadd231ps( xmm0,xmm1,xmm6 )
  vbroadcastss(mem(rax, r15, 1 ),xmm15 )
  vfmadd231ps( xmm0,xmm15,xmm7 )
  add( r12, rax)
  vbroadcastss(mem(rax       ),xmm1 )
  vfmadd231ps( xmm0,xmm1,xmm8 )
  vbroadcastss(mem(rax, r8, 1),xmm15 )
  vfmadd231ps( xmm0,xmm15,xmm9 )
  
  add( r9, rbx)
  add( r10, rax)
  dec( rsi )
  jne( .K_LOOP_ITER )

  // Scale by alpha
  vmulps( xmm14,xmm2,xmm2 )
  vmulps( xmm14,xmm3,xmm3 )
  vmulps( xmm14,xmm4,xmm4 )
  vmulps( xmm14,xmm5,xmm5 )
  vmulps( xmm14,xmm6,xmm6 )
  vmulps( xmm14,xmm7,xmm7 )
  vmulps( xmm14,xmm8,xmm8 )
  vmulps( xmm14,xmm9,xmm9 )
  
  mov( var(beta), rdx )  // load address of beta
  vbroadcastss( (rdx),xmm14 )

  vxorps(xmm0, xmm0, xmm0)
  vucomiss(xmm0, xmm14)
  je(.SBETAZERO)

  vmovups( (rcx),xmm0 )
  vfmadd231ps( xmm14,xmm0,xmm2 )
  vmovups( xmm2,(rcx) )
  add(rdi, rcx)
  vmovups( mem(rcx),xmm0 )
  vfmadd231ps( xmm14,xmm0,xmm3 )
  vmovups( xmm3,mem(rcx) )
  add(rdi, rcx)
  vmovups( mem(rcx),xmm0 )
  vfmadd231ps( xmm14,xmm0,xmm4 )
  vmovups( xmm4,mem(rcx) )
  add(rdi, rcx)
  vmovups( mem(rcx),xmm0 )
  vfmadd231ps( xmm14,xmm0,xmm5 )
  vmovups( xmm5,mem(rcx) )
  add(rdi, rcx)
  vmovups( mem(rcx),xmm0 )
  vfmadd231ps( xmm14,xmm0,xmm6 )
  vmovups( xmm6,mem(rcx) )
  add(rdi, rcx)
  vmovups( mem(rcx),xmm0 )
  vfmadd231ps( xmm14,xmm0,xmm7 )
  vmovups( xmm7,mem(rcx) )
  add(rdi, rcx)
  vmovups( mem(rcx),xmm0 )
  vfmadd231ps( xmm14,xmm0,xmm8 )
  vmovups( xmm8,mem(rcx) )
  add(rdi, rcx)
  vmovups( mem(rcx),xmm0 )
  vfmadd231ps( xmm14,xmm0,xmm9 )
  vmovups( xmm9,mem(rcx) )
  add(rdi, rcx)
  jmp(.SDONE)

  label(.SBETAZERO)

  vmovups( xmm2,(rcx) )
  add(rdi, rcx)
  vmovups( xmm3,mem(rcx) )
  add(rdi, rcx)
  vmovups( xmm4,mem(rcx) )
  add(rdi, rcx)
  vmovups( xmm5,mem(rcx) )
  add(rdi, rcx)
  vmovups( xmm6,mem(rcx) )
  add(rdi, rcx)
  vmovups( xmm7,mem(rcx) )
  add(rdi, rcx)
  vmovups( xmm8,mem(rcx) )
  add(rdi, rcx)
  vmovups( xmm9,mem(rcx) )
  add(rdi, rcx)

  label(.SDONE)


  end_asm(
    : // output operands (none)
    : // input operands
     [k_iter] "m" (k_iter),
     [a]      "m" (a),
     [rs_a]   "m" (rs_a),
     [cs_a]   "m" (cs_a),
     [b]      "m" (b),
     [rs_b]   "m" (rs_b),
     [cs_b]   "m" (cs_b),
     [alpha]  "m" (alpha),
     [beta]   "m" (beta),
     [c]      "m" (c),
     [rs_c]   "m" (rs_c),
     [cs_c]   "m" (cs_c),
     [n0]     "m" (n0),
     [m0]     "m" (m0),
     [abuf]   "m" (abuf),
     [bbuf]   "m" (bbuf),
     [cbuf]   "m" (cbuf)
   : // register clobber list
   "rax", "rbx", "rcx", "rdx", "rsi", "rdi",
   "r8", "r9", "r10", "r11", "r12", "r13", "r14", "r15",
   "ymm0", "ymm1", "ymm2", "ymm3",
   "ymm4", "ymm5", "ymm6", "ymm7", "ymm8", "ymm9", "ymm10",
   "ymm11", "ymm12", "ymm13", "ymm14", "ymm15",
   "memory"
   )
}

void bli_sgemmsup_rv_zen_asm_12x2m
(
       conj_t    conja,
       conj_t    conjb,
       dim_t     m0,
       dim_t     n0,
       dim_t     k0,
       float*    restrict alpha,
       float*    restrict a, inc_t rs_a0, inc_t cs_a0,
       float*    restrict b, inc_t rs_b0, inc_t cs_b0,
       float*    restrict beta,
       float*    restrict c, inc_t rs_c0, inc_t cs_c0,
       auxinfo_t* restrict data,
       cntx_t*    restrict cntx
     )
{
  uint64_t k_iter = k0;

  uint64_t m_iter = m0 / 12;
  uint64_t m_left = m0 % 12;

  uint64_t rs_a   = rs_a0;
  uint64_t cs_a   = cs_a0;
  uint64_t rs_b   = rs_b0;
  uint64_t cs_b   = cs_b0;
  uint64_t rs_c   = rs_c0;
  uint64_t cs_c   = cs_c0;

  float *abuf = a;
  float *bbuf = b;
  float *cbuf = c;

  if ( m_iter == 0 ) goto consider_edge_cases;

  /*Produce MRXNR outputs */
  // -------------------------------------------------------------------------
  begin_asm()

  mov(var(rs_a), r8)                 // load rs_a
  lea(mem(, r8, 4), r8)              // rs_a *= sizeof(dt)
  mov(var(rs_b), r9)                 // load rs_b
  lea(mem(, r9, 4), r9)              // rs_b *= sizeof(dt)
  mov(var(cs_a), r10)                // load cs_a
  lea(mem(, r10, 4), r10)            // cs_a *= sizeof(dt)
  mov(var(rs_c), rdi)                // load rs_c
  lea(mem(, rdi, 4), rdi)            // rs_c *= sizeof(float)


  lea(mem(r8, r8, 2), r13)           // r13 = 3*rs_a
  lea(mem(r8, r8, 4), r15)           // r15 = 5*rs_a
  lea(mem(r13, r13, 1), r12)         // r12 = 6*rs_a
  sub(r12,r10)                       // r10 = cs_a-6*rs_a

  mov( var( m_iter ), r11 )        // load m_iter

  label( .M_LOOP_ITER )

  vxorps( xmm0,xmm0,xmm0 )
  vxorps( xmm1,xmm1,xmm1 )
  vxorps( xmm2,xmm2,xmm2 )
  vxorps( xmm3,xmm3,xmm3 )
  vxorps( xmm4,xmm4,xmm4 )
  vxorps( xmm5,xmm5,xmm5 )
  vxorps( xmm6,xmm6,xmm6 )
  vxorps( xmm7,xmm7,xmm7 )
  vxorps( xmm8,xmm8,xmm8 )
  vxorps( xmm9,xmm9,xmm9 )
  vxorps( xmm10,xmm10,xmm10 )
  vxorps( xmm11,xmm11,xmm11 )
  vxorps( xmm12,xmm12,xmm12 )
  vxorps( xmm13,xmm13,xmm13 )
  vxorps( xmm14,xmm14,xmm14 )
  vxorps( xmm15,xmm15,xmm15 )
  mov( var(abuf), rax )   // load address of a
  mov( var(bbuf), rbx )   // load address of b
  mov( var(cbuf), rcx )   // load address of c

  mov( var( k_iter ), rsi )   // load k_iter
  test( rsi,rsi )
  mov( var(alpha), rdx )  // load address of alpha
  vbroadcastss( (rdx),xmm14 )

  label( .K_LOOP_ITER )

  vmovups( (rbx),xmm0 )
  vbroadcastss( (rax),xmm1 )
  vfmadd231ps( xmm0,xmm1,xmm2 )
  vbroadcastss(mem(rax, r8, 1),xmm15 )
  vfmadd231ps( xmm0,xmm15,xmm3 )
  vbroadcastss(mem(rax, r8, 2),xmm1 )
  vfmadd231ps( xmm0,xmm1,xmm4 )
  vbroadcastss(mem(rax, r13, 1 ),xmm15 )
  vfmadd231ps( xmm0,xmm15,xmm5 )
  vbroadcastss(mem(rax, r8, 4),xmm1 )
  vfmadd231ps( xmm0,xmm1,xmm6 )
  vbroadcastss(mem(rax, r15, 1 ),xmm15 )
  vfmadd231ps( xmm0,xmm15,xmm7 )
  add( r12, rax)
  vbroadcastss(mem(rax       ),xmm1 )
  vfmadd231ps( xmm0,xmm1,xmm8 )
  vbroadcastss(mem(rax, r8, 1),xmm15 )
  vfmadd231ps( xmm0,xmm15,xmm9 )
  vbroadcastss(mem(rax, r8, 2),xmm1 )
  vfmadd231ps( xmm0,xmm1,xmm10 )
  vbroadcastss(mem(rax, r13, 1 ),xmm15 )
  vfmadd231ps( xmm0,xmm15,xmm11 )
  vbroadcastss(mem(rax, r8, 4),xmm1 )
  vfmadd231ps( xmm0,xmm1,xmm12 )
  vbroadcastss(mem(rax, r15, 1 ),xmm15 )
  vfmadd231ps( xmm0,xmm15,xmm13 )
  
  add( r9, rbx)
  add( r10, rax)
  dec( rsi )
  jne( .K_LOOP_ITER )

  // Scale by alpha
  vmulps( xmm14,xmm2,xmm2 )
  vmulps( xmm14,xmm3,xmm3 )
  vmulps( xmm14,xmm4,xmm4 )
  vmulps( xmm14,xmm5,xmm5 )
  vmulps( xmm14,xmm6,xmm6 )
  vmulps( xmm14,xmm7,xmm7 )
  vmulps( xmm14,xmm8,xmm8 )
  vmulps( xmm14,xmm9,xmm9 )
  vmulps( xmm14,xmm10,xmm10 )
  vmulps( xmm14,xmm11,xmm11 )
  vmulps( xmm14,xmm12,xmm12 )
  vmulps( xmm14,xmm13,xmm13 )
  
  mov( var(beta), rdx )  // load address of beta
  vbroadcastss( (rdx),xmm14 )

  vxorps(xmm0, xmm0, xmm0)
  vucomiss(xmm0, xmm14)
  je(.SBETAZERO)

  vmovsd( (rcx),xmm0 )
  vfmadd231ps( xmm14,xmm0,xmm2 )
  vmovsd( xmm2,(rcx) )
  add(rdi, rcx)
  vmovsd( mem(rcx),xmm0 )
  vfmadd231ps( xmm14,xmm0,xmm3 )
  vmovsd( xmm3,mem(rcx) )
  add(rdi, rcx)
  vmovsd( mem(rcx),xmm0 )
  vfmadd231ps( xmm14,xmm0,xmm4 )
  vmovsd( xmm4,mem(rcx) )
  add(rdi, rcx)
  vmovsd( mem(rcx),xmm0 )
  vfmadd231ps( xmm14,xmm0,xmm5 )
  vmovsd( xmm5,mem(rcx) )
  add(rdi, rcx)
  vmovsd( mem(rcx),xmm0 )
  vfmadd231ps( xmm14,xmm0,xmm6 )
  vmovsd( xmm6,mem(rcx) )
  add(rdi, rcx)
  vmovsd( mem(rcx),xmm0 )
  vfmadd231ps( xmm14,xmm0,xmm7 )
  vmovsd( xmm7,mem(rcx) )
  add(rdi, rcx)
  vmovsd( mem(rcx),xmm0 )
  vfmadd231ps( xmm14,xmm0,xmm8 )
  vmovsd( xmm8,mem(rcx) )
  add(rdi, rcx)
  vmovsd( mem(rcx),xmm0 )
  vfmadd231ps( xmm14,xmm0,xmm9 )
  vmovsd( xmm9,mem(rcx) )
  add(rdi, rcx)
  vmovsd( mem(rcx),xmm0 )
  vfmadd231ps( xmm14,xmm0,xmm10 )
  vmovsd( xmm10,mem(rcx) )
  add(rdi, rcx)
  vmovsd( mem(rcx),xmm0 )
  vfmadd231ps( xmm14,xmm0,xmm11 )
  vmovsd( xmm11,mem(rcx) )
  add(rdi, rcx)
  vmovsd( mem(rcx),xmm0 )
  vfmadd231ps( xmm14,xmm0,xmm12 )
  vmovsd( xmm12,mem(rcx) )
  add(rdi, rcx)
  vmovsd( mem(rcx),xmm0 )
  vfmadd231ps( xmm14,xmm0,xmm13 )
  vmovsd( xmm13,mem(rcx) )
  add(rdi, rcx)
  jmp(.SDONE)

  label(.SBETAZERO)

  vmovsd( xmm2,(rcx) )
  add(rdi, rcx)
  vmovsd( xmm3,mem(rcx) )
  add(rdi, rcx)
  vmovsd( xmm4,mem(rcx) )
  add(rdi, rcx)
  vmovsd( xmm5,mem(rcx) )
  add(rdi, rcx)
  vmovsd( xmm6,mem(rcx) )
  add(rdi, rcx)
  vmovsd( xmm7,mem(rcx) )
  add(rdi, rcx)
  vmovsd( xmm8,mem(rcx) )
  add(rdi, rcx)
  vmovsd( xmm9,mem(rcx) )
  add(rdi, rcx)
  vmovsd( xmm10,mem(rcx) )
  add(rdi, rcx)
  vmovsd( xmm11,mem(rcx) )
  add(rdi, rcx)
  vmovsd( xmm12,mem(rcx) )
  add(rdi, rcx)
  vmovsd( xmm13,mem(rcx) )
  add(rdi, rcx)

  label(.SDONE)

  lea(mem(, r8, 4), rdx )            // rs_a * 4
  lea(mem(rdx, r8, 8), rdx )         // rdx = rs_a * 12
  mov( var(abuf), rax )              // load address of a
  add( rdx, rax )                    // a += rs_a * MR
  mov( rax, var(abuf) )              // store updated a

  lea(mem(, rdi, 4), rdx )           // rs_c * 4
  lea(mem(rdx, rdi, 8), rdx )        // rdx = rs_c * 12
  mov( var(cbuf), rcx )              // load address of c
  add( rdx, rcx )                    // c += rs_c * MR
  mov( rcx, var(cbuf) )              // store updated c

  dec( r11 )
  jne( .M_LOOP_ITER )

  end_asm(
    : // output operands (none)
    : // input operands
     [k_iter] "m" (k_iter),
     [a]      "m" (a),
     [rs_a]   "m" (rs_a),
     [cs_a]   "m" (cs_a),
     [b]      "m" (b),
     [rs_b]   "m" (rs_b),
     [cs_b]   "m" (cs_b),
     [alpha]  "m" (alpha),
     [beta]   "m" (beta),
     [c]      "m" (c),
     [rs_c]   "m" (rs_c),
     [cs_c]   "m" (cs_c),
     [n0]     "m" (n0),
     [m0]     "m" (m0),
     [m_iter] "m" (m_iter),
     [abuf]   "m" (abuf),
     [bbuf]   "m" (bbuf),
     [cbuf]   "m" (cbuf)
   : // register clobber list
   "rax", "rbx", "rcx", "rdx", "rsi", "rdi",
   "r8", "r9", "r10", "r11", "r12", "r13", "r14", "r15",
   "ymm0", "ymm1", "ymm2", "ymm3",
   "ymm4", "ymm5", "ymm6", "ymm7", "ymm8", "ymm9", "ymm10",
   "ymm11", "ymm12", "ymm13", "ymm14", "ymm15",
   "memory"
   )

  consider_edge_cases:

  // Handle edge cases in the m dimension, if they exist.
  if (m_left)
  {
    const dim_t      i_edge = m0 - ( dim_t )m_left;

    float* restrict cij = c + i_edge*rs_c;
    float* restrict ai  = a + i_edge*rs_a;
    float* restrict bj  = b;

    if (8 <= m_left)
    {
      const dim_t      mr_cur = 8;
      bli_sgemmsup_rv_zen_asm_8x2m(conja, conjb, mr_cur, n0, k0, alpha,
                                   ai, rs_a0, cs_a0,
                                   bj, rs_b0, cs_b0,
                                   beta,
                                   cij, rs_c0, cs_c0,
                                   data, cntx);
      cij += mr_cur * rs_c; ai += mr_cur * rs_a;
      m_left -= mr_cur;
    }
    if (4 <= m_left) {
      const dim_t      mr_cur = 4;
      bli_sgemmsup_rv_zen_asm_4x2(conja, conjb, mr_cur, n0, k0, alpha,
                                  ai, rs_a0, cs_a0,
                                  bj, rs_b0, cs_b0,
                                  beta,
                                  cij, rs_c0, cs_c0,
                                  data, cntx);
      cij += mr_cur * rs_c; ai += mr_cur * rs_a;
      m_left -= mr_cur;
    }
    if (2 <= m_left) {
      const dim_t      mr_cur = 2;
      bli_sgemmsup_rv_zen_asm_2x2(conja, conjb, mr_cur, n0, k0, alpha,
                                  ai, rs_a0, cs_a0,
                                  bj, rs_b0, cs_b0,
                                  beta,
                                  cij, rs_c0, cs_c0,
                                  data, cntx);
      cij += mr_cur * rs_c; ai += mr_cur * rs_a;
      m_left -= mr_cur;
    }
    if (1 <= m_left) {
      const dim_t      mr_cur = 1;
      bli_sgemmsup_rv_zen_asm_1x2(conja, conjb, mr_cur, n0, k0, alpha,
                                  ai, rs_a0, cs_a0,
                                  bj, rs_b0, cs_b0,
                                  beta,
                                  cij, rs_c0, cs_c0,
                                  data, cntx);
      cij += mr_cur * rs_c; ai += mr_cur * rs_a;
      m_left -= mr_cur;
    }
  }
}

void bli_sgemmsup_rv_zen_asm_8x2m
(
       conj_t    conja,
       conj_t    conjb,
       dim_t     m0,
       dim_t     n0,
       dim_t     k0,
       float*    restrict alpha,
       float*    restrict a, inc_t rs_a0, inc_t cs_a0,
       float*    restrict b, inc_t rs_b0, inc_t cs_b0,
       float*    restrict beta,
       float*    restrict c, inc_t rs_c0, inc_t cs_c0,
       auxinfo_t* restrict data,
       cntx_t*    restrict cntx
     )
{
  uint64_t k_iter = k0;

  uint64_t rs_a   = rs_a0;
  uint64_t cs_a   = cs_a0;
  uint64_t rs_b   = rs_b0;
  uint64_t cs_b   = cs_b0;
  uint64_t rs_c   = rs_c0;
  uint64_t cs_c   = cs_c0;

  float *abuf = a;
  float *bbuf = b;
  float *cbuf = c;

  /*Produce MRXNR outputs */
  // -------------------------------------------------------------------------
  begin_asm()

  mov(var(rs_a), r8)                 // load rs_a
  lea(mem(, r8, 4), r8)              // rs_a *= sizeof(dt)
  mov(var(rs_b), r9)                 // load rs_b
  lea(mem(, r9, 4), r9)              // rs_b *= sizeof(dt)
  mov(var(cs_a), r10)                // load cs_a
  lea(mem(, r10, 4), r10)            // cs_a *= sizeof(dt)
  mov(var(rs_c), rdi)                // load rs_c
  lea(mem(, rdi, 4), rdi)            // rs_c *= sizeof(float)


  lea(mem(r8, r8, 2), r13)           // r13 = 3*rs_a
  lea(mem(r8, r8, 4), r15)           // r15 = 5*rs_a
  lea(mem(r13, r13, 1), r12)         // r12 = 6*rs_a
  sub(r12,r10)                       // r10 = cs_a-6*rs_a
  vxorps( xmm0,xmm0,xmm0 )
  vxorps( xmm1,xmm1,xmm1 )
  vxorps( xmm2,xmm2,xmm2 )
  vxorps( xmm3,xmm3,xmm3 )
  vxorps( xmm4,xmm4,xmm4 )
  vxorps( xmm5,xmm5,xmm5 )
  vxorps( xmm6,xmm6,xmm6 )
  vxorps( xmm7,xmm7,xmm7 )
  vxorps( xmm8,xmm8,xmm8 )
  vxorps( xmm9,xmm9,xmm9 )
  vxorps( xmm10,xmm10,xmm10 )
  vxorps( xmm11,xmm11,xmm11 )
  vxorps( xmm12,xmm12,xmm12 )
  vxorps( xmm13,xmm13,xmm13 )
  vxorps( xmm14,xmm14,xmm14 )
  vxorps( xmm15,xmm15,xmm15 )
  mov( var(abuf), rax )   // load address of a
  mov( var(bbuf), rbx )   // load address of b
  mov( var(cbuf), rcx )   // load address of c

  mov( var( k_iter ), rsi )   // load k_iter
  test( rsi,rsi )
  mov( var(alpha), rdx )  // load address of alpha
  vbroadcastss( (rdx),xmm14 )

  label( .K_LOOP_ITER )

  vmovups( (rbx),xmm0 )
  vbroadcastss( (rax),xmm1 )
  vfmadd231ps( xmm0,xmm1,xmm2 )
  vbroadcastss(mem(rax, r8, 1),xmm15 )
  vfmadd231ps( xmm0,xmm15,xmm3 )
  vbroadcastss(mem(rax, r8, 2),xmm1 )
  vfmadd231ps( xmm0,xmm1,xmm4 )
  vbroadcastss(mem(rax, r13, 1 ),xmm15 )
  vfmadd231ps( xmm0,xmm15,xmm5 )
  vbroadcastss(mem(rax, r8, 4),xmm1 )
  vfmadd231ps( xmm0,xmm1,xmm6 )
  vbroadcastss(mem(rax, r15, 1 ),xmm15 )
  vfmadd231ps( xmm0,xmm15,xmm7 )
  add( r12, rax)
  vbroadcastss(mem(rax       ),xmm1 )
  vfmadd231ps( xmm0,xmm1,xmm8 )
  vbroadcastss(mem(rax, r8, 1),xmm15 )
  vfmadd231ps( xmm0,xmm15,xmm9 )
  
  add( r9, rbx)
  add( r10, rax)
  dec( rsi )
  jne( .K_LOOP_ITER )

  // Scale by alpha
  vmulps( xmm14,xmm2,xmm2 )
  vmulps( xmm14,xmm3,xmm3 )
  vmulps( xmm14,xmm4,xmm4 )
  vmulps( xmm14,xmm5,xmm5 )
  vmulps( xmm14,xmm6,xmm6 )
  vmulps( xmm14,xmm7,xmm7 )
  vmulps( xmm14,xmm8,xmm8 )
  vmulps( xmm14,xmm9,xmm9 )
  
  mov( var(beta), rdx )  // load address of beta
  vbroadcastss( (rdx),xmm14 )

  vxorps(xmm0, xmm0, xmm0)
  vucomiss(xmm0, xmm14)
  je(.SBETAZERO)

  vmovsd( (rcx),xmm0 )
  vfmadd231ps( xmm14,xmm0,xmm2 )
  vmovsd( xmm2,(rcx) )
  add(rdi, rcx)
  vmovsd( mem(rcx),xmm0 )
  vfmadd231ps( xmm14,xmm0,xmm3 )
  vmovsd( xmm3,mem(rcx) )
  add(rdi, rcx)
  vmovsd( mem(rcx),xmm0 )
  vfmadd231ps( xmm14,xmm0,xmm4 )
  vmovsd( xmm4,mem(rcx) )
  add(rdi, rcx)
  vmovsd( mem(rcx),xmm0 )
  vfmadd231ps( xmm14,xmm0,xmm5 )
  vmovsd( xmm5,mem(rcx) )
  add(rdi, rcx)
  vmovsd( mem(rcx),xmm0 )
  vfmadd231ps( xmm14,xmm0,xmm6 )
  vmovsd( xmm6,mem(rcx) )
  add(rdi, rcx)
  vmovsd( mem(rcx),xmm0 )
  vfmadd231ps( xmm14,xmm0,xmm7 )
  vmovsd( xmm7,mem(rcx) )
  add(rdi, rcx)
  vmovsd( mem(rcx),xmm0 )
  vfmadd231ps( xmm14,xmm0,xmm8 )
  vmovsd( xmm8,mem(rcx) )
  add(rdi, rcx)
  vmovsd( mem(rcx),xmm0 )
  vfmadd231ps( xmm14,xmm0,xmm9 )
  vmovsd( xmm9,mem(rcx) )
  add(rdi, rcx)
  jmp(.SDONE)

  label(.SBETAZERO)

  vmovsd( xmm2,(rcx) )
  add(rdi, rcx)
  vmovsd( xmm3,mem(rcx) )
  add(rdi, rcx)
  vmovsd( xmm4,mem(rcx) )
  add(rdi, rcx)
  vmovsd( xmm5,mem(rcx) )
  add(rdi, rcx)
  vmovsd( xmm6,mem(rcx) )
  add(rdi, rcx)
  vmovsd( xmm7,mem(rcx) )
  add(rdi, rcx)
  vmovsd( xmm8,mem(rcx) )
  add(rdi, rcx)
  vmovsd( xmm9,mem(rcx) )
  add(rdi, rcx)

  label(.SDONE)


  end_asm(
    : // output operands (none)
    : // input operands
     [k_iter] "m" (k_iter),
     [a]      "m" (a),
     [rs_a]   "m" (rs_a),
     [cs_a]   "m" (cs_a),
     [b]      "m" (b),
     [rs_b]   "m" (rs_b),
     [cs_b]   "m" (cs_b),
     [alpha]  "m" (alpha),
     [beta]   "m" (beta),
     [c]      "m" (c),
     [rs_c]   "m" (rs_c),
     [cs_c]   "m" (cs_c),
     [n0]     "m" (n0),
     [m0]     "m" (m0),
     [abuf]   "m" (abuf),
     [bbuf]   "m" (bbuf),
     [cbuf]   "m" (cbuf)
   : // register clobber list
   "rax", "rbx", "rcx", "rdx", "rsi", "rdi",
   "r8", "r9", "r10", "r11", "r12", "r13", "r14", "r15",
   "ymm0", "ymm1", "ymm2", "ymm3",
   "ymm4", "ymm5", "ymm6", "ymm7", "ymm8", "ymm9", "ymm10",
   "ymm11", "ymm12", "ymm13", "ymm14", "ymm15",
   "memory"
   )
   
}
