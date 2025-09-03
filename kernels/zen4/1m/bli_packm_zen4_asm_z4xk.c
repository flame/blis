/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.

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

/******************************************************/
/* Transpose contents of R0, R1, R2, R3 and store     */
/* the result to same register                        */
/* Transpose 4x4 register                             */
/* Input R0 = Ar0 Ai0 Ar1 Ai1 Ar2 Ai2 Ar3 Ai3         */
/* Input R1 = Ar4 Ai4 Ar5 Ai5 Ar6 Ai6 Ar7 Ai7         */
/* Input R2 = Ar8 Ai8 Ar9 Ai9 Ar10 Ai10 Ar11 Ai11     */
/* Input R3 = Ar12 Ai12 Ar13 Ai13 Ar14 Ai14 Ar15 Ai15 */
/* ZMM4 = Ar0 Ai0 Ar2 Ai2 Ar4 Ai4 Ar6 Ai6            */
/* ZMM5 = Ar1 Ai1 Ar3 Ai3 Ar5 Ai5 Ar7 Ai7            */
/* ZMM6 = Ar8 Ai8 Ar10 Ai10 Ar12 Ai12 Ar14 Ai14      */
/* ZMM7 = Ar9 Ai9 Ar11 Ai11 Ar13 Ai13 Ar15 Ai15      */
/* Output R0 = Ar0 Ai0 Ar4 Ai4 Ar8 Ai8 Ar12 Ai12      */
/* Output R1 = Ar1 Ai1 Ar5 Ai5 Ar9 Ai9 Ar13 Ai13      */
/* Output R2 = Ar2 Ai2 Ar6 Ai6 Ar10 Ai10 Ar14 Ai14    */
/* Output R3 = Ar3 Ai3 Ar7 Ai7 Ar11 Ai11 Ar15 Ai15    */
/******************************************************/
#define TRANSPOSE(R0, R1, R2, R3) \
    VSHUFF64X2(IMM(0x88), ZMM(R1), ZMM(R0), ZMM(4)) \
    VSHUFF64X2(IMM(0xDD), ZMM(R1), ZMM(R0), ZMM(5)) \
    VSHUFF64X2(IMM(0x88), ZMM(R3), ZMM(R2), ZMM(6)) \
    VSHUFF64X2(IMM(0xDD), ZMM(R3), ZMM(R2), ZMM(7)) \
    VSHUFF64X2(IMM(0x88), ZMM(6), ZMM(4), ZMM(R0)) \
    VSHUFF64X2(IMM(0xDD), ZMM(6), ZMM(4), ZMM(R2)) \
    VSHUFF64X2(IMM(0x88), ZMM(7), ZMM(5), ZMM(R1)) \
    VSHUFF64X2(IMM(0xDD), ZMM(7), ZMM(5), ZMM(R3))

void bli_zpackm_zen4_asm_4xk
     (
       conj_t              conja,
       pack_t              schema,
       dim_t               cdim0,
       dim_t               k0,
       dim_t               k0_max,
       dcomplex*  restrict kappa,
       dcomplex*  restrict a, inc_t inca0, inc_t lda0,
       dcomplex*  restrict p,              inc_t ldp0,
       cntx_t*    restrict cntx
     )
{
    // This is the panel dimension assumed by the packm kernel.
    const dim_t      mnr   = 4;

    // Typecast local copies of integers in case dim_t and inc_t are a
    // different size than is expected by load instructions.
    const uint64_t k_iter = k0 / 4;
    const uint64_t k_left = k0 % 4;

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
    const bool     unitk  = bli_zeq1( *kappa );

    // -------------------------------------------------------------------------
    if ( cdim0 == mnr && !gs && !conja && unitk )
    {
        begin_asm()

        mov(var(a), rax)                   // load address of a.

        mov(var(inca), r8)                 // load inca
        mov(var(lda), r10)                 // load lda
        lea(mem(   , r8,  2), r8)
        lea(mem(   , r8,  8), r8)          // inca *= sizeof(dcomplex)
        lea(mem(   , r10, 2), r10)
        lea(mem(   , r10, 8), r10)         // lda *= sizeof(dcomplex)

        mov(var(p), rbx)                   // load address of p.

        lea(mem(   , r10, 4), r14)         // r14 = 4*lda

        cmp(imm(16), r8)                   // set ZF if (16*inca) == 16.
        jz(.ZCOLUNIT)                      // jump to column storage case

        // -- row storage on A -----------------------------------------

        label(.ZROWUNIT)

        lea(mem(r8,  r8,  2), r12)         // r12 = 3*inca

        mov(var(k_iter), rsi)              // i = k_iter;
        test(rsi, rsi)                     // check i via logical AND.
        je(.ZCONKLEFTROWU)                 // if i == 0, jump to code that
                                           // contains the k_left loop.
        label(.ZKITERROWU)                 // MAIN LOOP (k_iter)

        vmovupd(mem(rax,         0), zmm0)
        vmovupd(mem(rax,  r8, 1, 0), zmm1)
        vmovupd(mem(rax,  r8, 2, 0), zmm2)
        vmovupd(mem(rax, r12, 1, 0), zmm3)

        TRANSPOSE(0, 1, 2, 3)

        vmovupd(zmm0, mem(rbx, 0*64))
        vmovupd(zmm1, mem(rbx, 1*64))
        vmovupd(zmm2, mem(rbx, 2*64))
        vmovupd(zmm3, mem(rbx, 3*64))

        add(r14, rax)                      // a += 4*lda;

        add(imm(4*4*16), rbx)              // p += 4*ldp = 4*4;

        dec(rsi)                           // i -= 1;
        jne(.ZKITERROWU)                   // iterate again if i != 0.

        label(.ZCONKLEFTROWU)

        mov(var(k_left), rsi)              // i = k_left;
        test(rsi, rsi)                     // check i via logical AND.
        je(.ZDONE)                         // if i == 0, we're done; jump to end.
                                           // else, we prepare to enter k_left loop.

        label(.ZKLEFTROWU)                 // EDGE LOOP (k_left)

        vmovups(mem(rax,         0), xmm0)
        vmovups(mem(rax,  r8, 1, 0), xmm1)
        vmovups(mem(rax,  r8, 2, 0), xmm2)
        vmovups(mem(rax, r12, 1, 0), xmm3)

        add(r10, rax)                      // a += lda;

        vmovups(xmm0, mem(rbx, 0*16+0*64))
        vmovups(xmm1, mem(rbx, 1*16+0*64))
        vmovups(xmm2, mem(rbx, 2*16+0*64))
        vmovups(xmm3, mem(rbx, 3*16+0*64))

        add(imm(4*16), rbx)                // p += ldp = 4;

        dec(rsi)                           // i -= 1;
        jne(.ZKLEFTROWU)                   // iterate again if i != 0.

        jmp(.ZDONE)                        // jump to end.

        // -- column storage on A --------------------------------------

        label(.ZCOLUNIT)

        lea(mem(r10, r10, 2), r13)         // r13 = 3*lda

        mov(var(k_iter), rsi)              // i = k_iter;
        test(rsi, rsi)                     // check i via logical AND.
        je(.ZCONKLEFTCOLU)                 // if i == 0, jump to code that
                                           // contains the k_left loop.

        label(.ZKITERCOLU)                 // MAIN LOOP (k_iter)

        vmovupd(mem(rax,          0), zmm0)
        vmovupd(zmm0, mem(rbx, 0*64))

        vmovupd(mem(rax, r10, 1,  0), zmm1)
        vmovupd(zmm1, mem(rbx, 1*64))

        vmovupd(mem(rax, r10, 2,  0), zmm2)
        vmovupd(zmm2, mem(rbx, 2*64))

        vmovupd(mem(rax, r13, 1,  0), zmm3)
        add(r14, rax)                      // a += 4*lda;
        vmovupd(zmm3, mem(rbx, 3*64))
        add(imm(4*4*16), rbx)               // p += 4*ldp = 4*4;

        dec(rsi)                           // i -= 1;
        jne(.ZKITERCOLU)                   // iterate again if i != 0.

        label(.ZCONKLEFTCOLU)

        mov(var(k_left), rsi)              // i = k_left;
        test(rsi, rsi)                     // check i via logical AND.
        je(.ZDONE)                         // if i == 0, we're done; jump to end.
                                           // else, we prepare to enter k_left loop.

        label(.ZKLEFTCOLU)                 // EDGE LOOP (k_left)

        vmovupd(mem(rax,          0), zmm0)
        add(r10, rax)                      // a += lda;
        vmovupd(zmm0, mem(rbx))
        add(imm(4*16), rbx)                // p += ldp = 4;

        dec(rsi)                           // i -= 1;
        jne(.ZKLEFTCOLU)                   // iterate again if i != 0.

        label(.ZDONE)

        end_asm(
        : // output operands (none)
        : // input operands
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
          "zmm0", "zmm1", "zmm2", "zmm3",
          "zmm4", "zmm5", "zmm6", "zmm7",
          "memory"
        )
    }
    else // if ( cdim0 < mnr || gs || bli_does_conj( conja ) || !unitk )
    {
        PASTEMAC(zscal2m,BLIS_TAPI_EX_SUF)
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
            dcomplex* restrict p_edge = p + (i  )*1;

            bli_zset0s_mxn
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
        dcomplex* restrict p_edge = p + (j  )*ldp;

        bli_zset0s_mxn
        (
          m_edge,
          n_edge,
          p_edge, 1, ldp
        );
    }
}

