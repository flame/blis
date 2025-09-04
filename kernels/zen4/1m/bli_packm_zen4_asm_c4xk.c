/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.

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

/**
 * Shuffle 2 scomplex elements selected by imm8 from S1 and S2,
 * and store the results in D1
 * S1 : 1  9 3 11 5 13 7 15
 * S2 : 2 10 4 12 6 14 8 16
 * D1 : 1  9  5  13  2  10  6  14
 * D2 : 3 11  7  15  4  12  8  16
*/
#define SHUFFLE_DATA(S1, S2, D1, D2, S3, S4, D3, D4) \
\
    VSHUFF64X2(IMM(0x88), ZMM(S1), ZMM(S2), ZMM(D1)) \
    VSHUFF64X2(IMM(0xDD), ZMM(S1), ZMM(S2), ZMM(D2)) \
    VSHUFF64X2(IMM(0x88), ZMM(S3), ZMM(S4), ZMM(D3)) \
    VSHUFF64X2(IMM(0xDD), ZMM(S3), ZMM(S4), ZMM(D4)) \

/**
 * Unpacks and interleave low half and high half of each
 * 128-bit lane in S1 and S2 and store into D1 and D2
 * respectively.
 * S1 : 1  2  3  4  5  6  7  8
 * S2 : 9 10 11 12 13 14 15 16
 * D1 : 1  9 3 11 5 13 7 15
 * D2 : 2 10 4 12 6 14 8 16
*/
#define UNPACK_LO_HIGH(S1, S2, D1, D2, S3, S4, D3, D4) \
\
    vunpcklpd(zmm(S1),  zmm(S2),  zmm(D1)) \
    vunpckhpd(zmm(S1),  zmm(S2),  zmm(D2)) \
    vunpcklpd(zmm(S3),  zmm(S4),  zmm(D3)) \
    vunpckhpd(zmm(S3),  zmm(S4),  zmm(D4))

void bli_cpackm_zen4_asm_4xk
     (
       conj_t              conja,
       pack_t              schema,
       dim_t               cdim0,
       dim_t               k0,
       dim_t               k0_max,
       scomplex*  restrict kappa,
       scomplex*  restrict a, inc_t inca0, inc_t lda0,
       scomplex*  restrict p,              inc_t ldp0,
       cntx_t*    restrict cntx
     )
{
    // This is the panel dimension assumed by the packm kernel.
    const dim_t      mnr   = 4;

    // Typecast local copies of integers in case dim_t and inc_t are a
    // different size than is expected by load instructions.
    // NOTE : k_iter is in blocks of 8, due to the SIMD width of scomplex
    //        in a 512-register. This way, we could still perform AVX512 loads
    //        and stores in case of the matrix being in row-major format.
    const uint64_t k_iter = k0 / 8;
    const uint64_t k_left = k0 % 8;
	 /**
     * Preparing the mask for k_left, since we are computing in blocks of 8.
     * For  the edge cases, mask is set to load and store only the leftover elements.
	 */
    uint16_t one = 1;
	  uint16_t mask = ( one << ( 2 * k_left ) ) - one;

    // NOTE: For the purposes of the comments in this packm kernel, we
    // interpret inca and lda as rs_a and cs_a, respectively, and similarly
    // interpret ldp as cs_p (with rs_p implicitly unit). Thus, when reading
    // this packm kernel, you should think of the operation as packing an
    // m x n micropanel, where m and n are tiny and large, respectively, and
    // where elements of each column of the packed matrix P are contiguous.
    // (This packm kernel can still be used to pack micropanels of matrix B
    // in a gemm operation.)
    const uint64_t inca   = inca0;
    const uint64_t lda    = lda0;
    const uint64_t ldp    = ldp0;

    const bool     gs     = ( inca0 != 1 && lda0 != 1 );

    // NOTE: If/when this kernel ever supports scaling by kappa within the
    // assembly region, this constraint should be lifted.
    const bool     unitk  = bli_ceq1( *kappa );

    // -------------------------------------------------------------------------
    if ( cdim0 == mnr && !gs && !conja && unitk )
    {
      begin_asm()

      mov(var(mask), rdx)                // load mask
      kmovw(edx, k(2))                   // move mask to k2 register

      mov(var(a), rax)                   // load address of source buffer
		  mov(var(a), r13)                   // load address of source buffer

      mov(var(inca), r8)                 // load inca
      mov(var(lda), r10)                 // load lda
      lea(mem(   , r8,  8), r8)          // inca *= sizeof(scomplex)
      lea(mem(   , r10, 8), r10)         // lda *= sizeof(scomplex)

      mov(var(p), rbx)                   // load address of p.

      lea(mem(   , r8, 8), r14)          // r14 = 8*inca

      cmp(imm(8), r8)                    // set ZF if (8*inca) == 8.
      jz(.CCOLUNIT)                      // jump to column storage case

      // -- kappa unit, row storage on A -------------------------------

      label(.CROWUNIT)

      lea(mem(r8,  r8,  2), r12)         // r12 = 3*inca

      mov(var(k_iter), rsi)              // i = k_iter;
      test(rsi, rsi)                     // check i via logical AND.
      je(.CCONKLEFTROWU)                 // if i == 0, jump to code that
                                         // contains the k_left loop.
      label(.CKITERROWU)                 // MAIN LOOP (k_iter)

      /**
       * Load first 4 rows of matrix.
       * Set 4 additional registers to zero
       * Transpose 8x8(by extending 4x8 with 0.0 padding ) tile
         and store it back to destination buffer.
		  */
      vmovups(mem(rax,         0), zmm6)
      vmovups(mem(rax,  r8, 1, 0), zmm8)
      vmovups(mem(rax,  r8, 2, 0), zmm10)
      vmovups(mem(rax, r12, 1, 0), zmm12)
      vxorps(zmm14, zmm14, zmm14)
      vxorps(zmm16, zmm16, zmm16)
      vxorps(zmm18, zmm18, zmm18)
      vxorps(zmm20, zmm20, zmm20)

      /* Transpose the 8x8 matrix onto another set of 8 registers */
      /*
        Input :
        zmm6  --> row1
        zmm8  --> row2
        zmm10 --> row3
        zmm12 --> row4
        zmm14 --> 0.0f
        zmm16 --> 0.0f
        zmm18 --> 0.0f
        zmm20 --> 0.0f

        Output(after transpose) :
        zmm0  zmm4  zmm2  zmm6  zmm1  zmm5  zmm3  zmm7
          |     |     |     |     |     |     |     |
          |     |     |     |     |     |     |     |
          V     V     V     V     V     V     V     V
        col1   col2  col3  col4  col5  col6  col7  col8 

        Every column(register) will have the last 256-bit lane as 0.0f
        Thus, we only store the YMM registers to the destination buffer.   

      */
      UNPACK_LO_HIGH(8, 6, 0, 1, 12, 10, 2, 3)
      SHUFFLE_DATA(2, 0, 4, 5, 3, 1, 30, 31)
      UNPACK_LO_HIGH(16, 14, 0, 1, 20, 18, 2, 3)
      SHUFFLE_DATA(2, 0, 6, 8, 3, 1, 10, 12)
      SHUFFLE_DATA(6, 4, 0, 1, 8, 5, 2, 3)
      SHUFFLE_DATA(10, 30, 4, 5, 12, 31, 6, 8)

      /* Store the 256-bit lanes(YMM) onto the destination */
      vmovups(ymm0, mem(rbx, 0*32))
      vmovups(ymm4, mem(rbx, 1*32))
      vmovups(ymm2, mem(rbx, 2*32))
      vmovups(ymm6, mem(rbx, 3*32))
      vmovups(ymm1, mem(rbx, 4*32))
      vmovups(ymm5, mem(rbx, 5*32))
      vmovups(ymm3, mem(rbx, 6*32))
      vmovups(ymm8, mem(rbx, 7*32))

      add(imm(8*8), r13)
      mov(r13, rax)                      // a += 8*8*lda
      add(imm(8*8*4), rbx)               // p += 8*ldp

      dec(rsi)                           // i -= 1;
      jne(.CKITERROWU)                   // iterate again if i != 0.

      label(.CCONKLEFTROWU)

      mov(var(k_left), rsi)              // i = k_left;
      test(rsi, rsi)                     // check i via logical AND.
      je(.CDONE)                         // if i == 0, we're done; jump to end.
                                         // else, we prepare to enter k_left loop.

      label(.CKLEFTROWU)                 // EDGE LOOP (k_left)
 
      LABEL(.UPDATEL1)
      /* Move the first 4xk_left block of data */
      vmovups(mem(rax,         0), zmm6 MASK_KZ(2))
      vmovups(mem(rax,  r8, 1, 0), zmm8 MASK_KZ(2))
      vmovups(mem(rax,  r8, 2, 0), zmm10 MASK_KZ(2))
      vmovups(mem(rax, r12, 1, 0), zmm12 MASK_KZ(2))
      vxorps(zmm14, zmm14, zmm14)
      vxorps(zmm16, zmm16, zmm16)
      vxorps(zmm18, zmm18, zmm18)
      vxorps(zmm20, zmm20, zmm20)

      /*
        Input :
        zmm6  --> row1(masked loads)
        zmm8  --> row2(masked loads)
        zmm10 --> row3(masked loads)
        zmm12 --> row4(masked loads)
        zmm14 --> 0.0f
        zmm16 --> 0.0f
        zmm18 --> 0.0f
        zmm20 --> 0.0f

        Output(after transpose) :
        zmm0  zmm4  zmm2  zmm6  zmm1  zmm5  zmm3  zmm7
          |     |     |     |     |     |     |     |
          |     |     |     |     |     |     |     |
          V     V     V     V     V     V     V     V
        col1   col2  col3  col4  col5  col6  col7  col8 

        Every column(register) will have the last 256-bit lane as 0.0f
        Thus, we only store the YMM registers to the destination buffer.   

      */
      UNPACK_LO_HIGH(8, 6, 0, 1, 12, 10, 2, 3)
      SHUFFLE_DATA(2, 0, 4, 5, 3, 1, 30, 31)
      UNPACK_LO_HIGH(16, 14, 0, 1, 20, 18, 2, 3)
      SHUFFLE_DATA(2, 0, 6, 8, 3, 1, 10, 12)
      SHUFFLE_DATA(6, 4, 0, 1, 8, 5, 2, 3)
      SHUFFLE_DATA(10, 30, 4, 5, 12, 31, 6, 8)

      cmp(imm(7), rsi)
      JZ(.UPDATE7L1)
      cmp(imm(6), rsi)
      JZ(.UPDATE6L1)
      cmp(imm(5), rsi)
      JZ(.UPDATE5L1)
      cmp(imm(4), rsi)
      JZ(.UPDATE4L1)
      cmp(imm(3), rsi)
      JZ(.UPDATE3L1)
      cmp(imm(2), rsi)
      JZ(.UPDATE2L1)
      cmp(imm(1), rsi)
      JZ(.UPDATE1L1)

      LABEL(.UPDATE7L1)
  		// Update 4x7 tile to destination buffer
      vmovups(ymm0, mem(rbx, 0*32))
      vmovups(ymm4, mem(rbx, 1*32))
      vmovups(ymm2, mem(rbx, 2*32))
      vmovups(ymm6, mem(rbx, 3*32))
      vmovups(ymm1, mem(rbx, 4*32))
      vmovups(ymm5, mem(rbx, 5*32))
      vmovups(ymm3, mem(rbx, 6*32))
      jmp(.CDONE)

      LABEL(.UPDATE6L1)
  		// Update 4x6 tile to destination buffer
      vmovups(ymm0, mem(rbx, 0*32))
      vmovups(ymm4, mem(rbx, 1*32))
      vmovups(ymm2, mem(rbx, 2*32))
      vmovups(ymm6, mem(rbx, 3*32))
      vmovups(ymm1, mem(rbx, 4*32))
      vmovups(ymm5, mem(rbx, 5*32))
      jmp(.CDONE)

      LABEL(.UPDATE5L1)
  		// Update 4x5 tile to destination buffer
      vmovups(ymm0, mem(rbx, 0*32))
      vmovups(ymm4, mem(rbx, 1*32))
      vmovups(ymm2, mem(rbx, 2*32))
      vmovups(ymm6, mem(rbx, 3*32))
      vmovups(ymm1, mem(rbx, 4*32))
      jmp(.CDONE)

      LABEL(.UPDATE4L1)
  		// Update 4x4 tile to destination buffer
      vmovups(ymm0, mem(rbx, 0*32))
      vmovups(ymm4, mem(rbx, 1*32))
      vmovups(ymm2, mem(rbx, 2*32))
      vmovups(ymm6, mem(rbx, 3*32))
      jmp(.CDONE)

      LABEL(.UPDATE3L1)
  		// Update 4x3 tile to destination buffer
      vmovups(ymm0, mem(rbx, 0*32))
      vmovups(ymm4, mem(rbx, 1*32))
      vmovups(ymm2, mem(rbx, 2*32))
      jmp(.CDONE)

      LABEL(.UPDATE2L1)
  		// Update 4x2 tile to destination buffer
      vmovups(ymm0, mem(rbx, 0*32))
      vmovups(ymm4, mem(rbx, 1*32))
      jmp(.CDONE)

      LABEL(.UPDATE1L1)
  		// Update 4x1 tile to destination buffer
      vmovups(ymm0, mem(rbx, 0*32))
      jmp(.CDONE)

      // -- column storage on A --------------------------------------

      label(.CCOLUNIT)

      mov(var(ldp), r8)                  // load ldp
		  lea(mem(, r8,  8), r8)             // r8 *= sizeof(scomplex)

      mov(var(k_iter), rsi)              // i = k_iter;
      test(rsi, rsi)                     // check i via logical AND.
      je(.CCONKLEFTCOLU)                 // if i == 0, jump to code that
                                         // contains the k_left loop.

      label(.CKITERCOLU)                 // MAIN LOOP (k_iter)

      /* Load/store a column of C using YMM regsiters */
      /* Unroll-1 */
      vmovups(mem(rax), ymm6)
      vmovups(ymm6, mem(rbx))
      add(r10, rax)
      add(r8, rbx)

      /* Unroll-2 */
      vmovups(mem(rax), ymm6)
      vmovups(ymm6, mem(rbx))
      add(r10, rax)
      add(r8, rbx)
      
      /* Unroll-3 */
      vmovups(mem(rax), ymm6)
      vmovups(ymm6, mem(rbx))
      add(r10, rax)
      add(r8, rbx)

      /* Unroll-4 */
      vmovups(mem(rax), ymm6)
      vmovups(ymm6, mem(rbx))
      add(r10, rax)
      add(r8, rbx)

      /* Unroll-5 */
      vmovups(mem(rax), ymm6)
      vmovups(ymm6, mem(rbx))
      add(r10, rax)
      add(r8, rbx)

      /* Unroll-6 */
      vmovups(mem(rax), ymm6)
      vmovups(ymm6, mem(rbx))
      add(r10, rax)
      add(r8, rbx)
      
      /* Unroll-7 */
      vmovups(mem(rax), ymm6)
      vmovups(ymm6, mem(rbx))
      add(r10, rax)
      add(r8, rbx)

      /* Unroll-8 */
      vmovups(mem(rax), ymm6)
      vmovups(ymm6, mem(rbx))
      add(r10, rax)
      add(r8, rbx)

      dec(rsi)                           // i -= 1;
      jne(.CKITERCOLU)                   // iterate again if i != 0.

      label(.CCONKLEFTCOLU)

      mov(var(k_left), rsi)              // i = k_left;
      test(rsi, rsi)                     // check i via logical AND.
      je(.CDONE)                         // if i == 0, we're done; jump to end.
                                          // else, we prepare to enter k_left loop.

      label(.CKLEFTCOLU)                 // EDGE LOOP (k_left)

      /* Load/store a column of C using YMM register */
      vmovups(mem(rax), ymm6)
      vmovups(ymm6, mem(rbx))
      add(r10, rax)
      add(r8, rbx)

      dec(rsi)                           // i -= 1;
      jne(.CKLEFTCOLU)                   // iterate again if i != 0.

      label(.CDONE)

      end_asm(
      : // output operands (none)
      : // input operands
		    [mask] "m" (mask),
        [k_iter] "m" (k_iter),
        [k_left] "m" (k_left),
        [a]      "m" (a),
        [inca]   "m" (inca),
        [lda]    "m" (lda),
        [p]      "m" (p),
        [ldp]    "m" (ldp)
      : // register clobber list
        "rax", "rbx", "rcx", "rdx", "rsi",
        "r8", "r10", "r12", "r13", "r14",
        "xmm0", "xmm1", "xmm2", "xmm3",
        "ymm0", "ymm1", "ymm2", "ymm3",
        "ymm4", "ymm5", "ymm6", "ymm8",
        "zmm0", "zmm1", "zmm2", "zmm3",
        "zmm4", "zmm5", "zmm6", "zmm7",
        "zmm8", "zmm10", "zmm12", "zmm14",
        "zmm16", "zmm18", "zmm20", "zmm30",
        "zmm31", "k2", "memory"
      )
    }
    else // if ( cdim0 < mnr || gs || bli_does_conj( conja ) || !unitk )
    {
      PASTEMAC(cscal2m,BLIS_TAPI_EX_SUF)
      (
        0,
        BLIS_NONUNIT_DIAG,
        BLIS_DENSE,
        ( trans_t )conja,
        cdim0,
        k0,
        kappa,
        a, inca0, lda0,
        p,     1, ldp0,
        cntx,
        NULL
      );

      if ( cdim0 < mnr )
      {
          // Handle zero-filling along the "long" edge of the micropanel.

          const dim_t        i      = cdim0;
          const dim_t        m_edge = mnr - cdim0;
          const dim_t        n_edge = k0_max;
          scomplex* restrict p_edge = p + (i  )*1;

          bli_cset0s_mxn
          (
            m_edge,
            n_edge,
            p_edge, 1, ldp
          );
      }
    }

    if ( k0 < k0_max )
    {
      // Handle zero-filling along the "short" (far) edge of the micropanel.

      const dim_t        j      = k0;
      const dim_t        m_edge = mnr;
      const dim_t        n_edge = k0_max - k0;
      scomplex* restrict p_edge = p + (j  )*ldp;

      bli_cset0s_mxn
      (
        m_edge,
        n_edge,
        p_edge, 1, ldp
      );
    }
}

