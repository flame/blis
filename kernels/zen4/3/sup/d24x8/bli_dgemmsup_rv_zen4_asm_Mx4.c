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
    - Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.
      documentation and/or other materials provided with the distribution.
    - Neither the name(s) of the copyright holder(s) nor the names of its
      contributors may be used to endorse or promote products derived
      from this software without specific prior written permission.
   THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
   AS IS AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
   LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
   A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE UNIVERSITY
   OF TEXAS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
   EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
   PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
   PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
   OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
   (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
   OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/


#include "blis.h"
#define BLIS_ASM_SYNTAX_ATT
#include "bli_x86_asm_macros.h"
#define TAIL_NITER 3

/* These kernels Assume that A matrix needs to be in col-major order
 * B matrix can be col/row-major
 * C matrix can be col/row-major though support for row-major order will
 * be added by a separate commit.
 * Prefetch for C is done assuming that C is col-stored.
 * Prefetch of B is done assuming that the matrix is col-stored.
 * Prefetch for B and C matrices when row-stored is yet to be added.
 * Prefetch of A matrix is not done in edge-case kernels.
 */

void bli_dgemmsup_rv_zen4_asm_24x4
(
       conj_t    conja,
       conj_t    conjb,
       dim_t     m0,
       dim_t     n0,
       dim_t     k0,
       double*    restrict alpha,
       double*    restrict a, inc_t rs_a0, inc_t cs_a0,
       double*    restrict b, inc_t rs_b0, inc_t cs_b0,
       double*    restrict beta,
       double*    restrict c, inc_t rs_c0, inc_t cs_c0,
       auxinfo_t* restrict data,
       cntx_t*    restrict cntx
     )
{
    AOCL_DTL_TRACE_ENTRY(AOCL_DTL_LEVEL_TRACE_7);
    // Typecast local copies of integers in case dim_t and inc_t are a
    // different size than is expected by load instructions.
    uint64_t rs_a   = rs_a0;
    uint64_t cs_a   = cs_a0;
    uint64_t rs_b   = rs_b0;
    uint64_t cs_b   = cs_b0;
    uint64_t rs_c   = rs_c0;
    uint64_t cs_c   = cs_c0;

    uint64_t ps_a = bli_auxinfo_ps_a( data );
    uint64_t ps_a8  = ps_a * sizeof( double );

    uint64_t k_iter = (uint64_t)k0 / 8;
    uint64_t k_left = (uint64_t)k0 % 8;

    /* 8 double precision elements can be loaded into a 512-bit register
     * So, we use an 8-bit mask to specify which elements to be loaded/stored
     * into/from the register. m_left % 8 specifies how many number of elements
     *  are to be loaded/stored into/from the last register.
     * For example, if m_left = 19, m0 & 7 becomes 3 which indicates that 3 elements
     * have to be loaded/stored into/from register, so shift 0xff(11111111) by (8-3)
     *  times to the right which makes the mask to be (00000111)
     */
    uint8_t mask = 0xff >> (0x8 - (m0 & 7)); // calculate mask based on m_left
    // For special cases where m_left = 24/16/8, all 8 elements have to be loaded/stored
    // So, mask becomes 0xff(11111111)
    if (mask == 0) mask = 0xff;

        // -------------------------------------------------------------------------
    begin_asm()

    mov(var(a), rax)                // load address of a
    mov(var(cs_a), r10)             // load cs_a
    mov(var(b), rbx)                // load address of b
    mov(var(rs_b), r8)              // load rs_b
    mov(var(cs_b), r9)              // load cs_b
    mov(var(c), rcx)                // load address of c
    mov(var(cs_c), rdi)             // load cs_c
    mov(var(mask), rdx)             // load mask
    kmovw(edx, k(2))                // move mask to k2 register
    lea(mem(, r8, 8), r8)           // rs_b *= sizeof(double)
    lea(mem(, r9, 8), r9)           // cs_b *= sizeof(double)
    lea(mem(, r10, 8), r10)         // cs_a *= sizeof(double)
    lea(mem(, rdi, 8), rdi)         // cs_c *= sizeof(double)
    lea(mem(r9, r9, 2 ), r13)       // r13 = 3*cs_b
    lea(mem(rcx, 7*8), rdx)         // C for prefetching
    lea(mem(rbx, r8, 8, 7*8), r11)  // r11 = rbx + 8*rs_b(B for prefetching)

    /* Register usage: zmm0-5 are used to load A matrix
     *                 zmm6-29 are used for accumulation
     *                 zmm30-31 are used for broadcasting B matrix
     */

    // zero out all accumulation registers
    vxorpd(zmm6, zmm6, zmm6)
    vxorpd(zmm7, zmm7, zmm7)
    vxorpd(zmm28, zmm28, zmm28)
    vxorpd(zmm8, zmm8, zmm8)
    vxorpd(zmm9, zmm9, zmm9)
    vxorpd(zmm29, zmm29, zmm29)
    vxorpd(zmm10, zmm10, zmm10)
    vxorpd(zmm11, zmm11, zmm11)
    vxorpd(zmm26, zmm26, zmm26)
    vxorpd(zmm12, zmm12, zmm12)
    vxorpd(zmm13, zmm13, zmm13)
    vxorpd(zmm27,zmm27, zmm27)

    // K is unrolled by 8 to facilitate prefetch of B
    // Assuming B to be col-stored, for each iteration of K,
    //one cacheline of B_next is prefetched where b_next = b + (unroll)*rs_b
    label(.DLOOPKITER)                                     // main loop
    mov(var(k_iter), rsi)                                  // i = k_iter
    sub(imm( 4+TAIL_NITER), rsi)                           // i -= NR + TAIL_NITER
    jle(.PREFETCHLOOP)                                     // jump if i <= 0

    label(.LOOP1)

        // ---------------------------------- iteration 1

        vmovupd( mem(rax),zmm0 )                           // load A
        vmovupd( 0x40(rax),zmm1 )
        vmovupd( 0x80(rax),zmm2 MASK_(k(2)) MASK_(z) )     // Load A with mask and zero hint
        add( r10,rax )                                     // a += cs_a
        vmovupd( mem(rax),zmm3 )                           // load A
        vmovupd( 0x40(rax),zmm4 )
        vmovupd( 0x80(rax),zmm5 MASK_(k(2)) MASK_(z) )     // Load A with mask and zero hint
        add( r10,rax )                                     // a += cs_a
        prefetch( 0,mem(r11) )                             // prefetch B
        vbroadcastsd( mem(rbx),zmm30 )
        vbroadcastsd( mem(rbx,r9,1),zmm31 )
        vfmadd231pd( zmm0,zmm30,zmm6 )
        vfmadd231pd( zmm1,zmm30,zmm7 )
        vfmadd231pd( zmm2,zmm30,zmm28 )
        vbroadcastsd( mem(rbx,r9,2),zmm30 )
        vfmadd231pd( zmm0,zmm31,zmm8 )
        vfmadd231pd( zmm1,zmm31,zmm9 )
        vfmadd231pd( zmm2,zmm31,zmm29 )
        vbroadcastsd( mem(rbx,r13,1),zmm31 )
        vfmadd231pd( zmm0,zmm30,zmm10 )
        vfmadd231pd( zmm1,zmm30,zmm11 )
        vfmadd231pd( zmm2,zmm30,zmm26 )
        add( r8,rbx )                                     // b += rs_b
        vfmadd231pd( zmm0,zmm31,zmm12 )
        vfmadd231pd( zmm1,zmm31,zmm13 )
        vfmadd231pd( zmm2,zmm31,zmm27 )

        // ---------------------------------- iteration 2

        vmovupd( mem(rax),zmm0 )                           // load A
        vmovupd( 0x40(rax),zmm1 )
        vmovupd( 0x80(rax),zmm2 MASK_(k(2)) MASK_(z) )     // Load A with mask and zero hint
        add( r10,rax )                                     // a += cs_a
        prefetch( 0,mem(r11,r9,1) )                             // prefetch B
        vbroadcastsd( mem(rbx),zmm30 )
        vbroadcastsd( mem(rbx,r9,1),zmm31 )
        vfmadd231pd( zmm3,zmm30,zmm6 )
        vfmadd231pd( zmm4,zmm30,zmm7 )
        vfmadd231pd( zmm5,zmm30,zmm28 )
        vbroadcastsd( mem(rbx,r9,2),zmm30 )
        vfmadd231pd( zmm3,zmm31,zmm8 )
        vfmadd231pd( zmm4,zmm31,zmm9 )
        vfmadd231pd( zmm5,zmm31,zmm29 )
        vbroadcastsd( mem(rbx,r13,1),zmm31 )
        vfmadd231pd( zmm3,zmm30,zmm10 )
        vfmadd231pd( zmm4,zmm30,zmm11 )
        vfmadd231pd( zmm5,zmm30,zmm26 )
        add( r8,rbx )                                     // b += rs_b
        vfmadd231pd( zmm3,zmm31,zmm12 )
        vfmadd231pd( zmm4,zmm31,zmm13 )
        vfmadd231pd( zmm5,zmm31,zmm27 )

        // ---------------------------------- iteration 3

        vmovupd( mem(rax),zmm3 )                           // load A
        vmovupd( 0x40(rax),zmm4 )
        vmovupd( 0x80(rax),zmm5 MASK_(k(2)) MASK_(z) )     // Load A with mask and zero hint
        add( r10,rax )                                     // a += cs_a
        prefetch( 0,mem(r11,r9,2) )                             // prefetch B
        vbroadcastsd( mem(rbx),zmm30 )
        vbroadcastsd( mem(rbx,r9,1),zmm31 )
        vfmadd231pd( zmm0,zmm30,zmm6 )
        vfmadd231pd( zmm1,zmm30,zmm7 )
        vfmadd231pd( zmm2,zmm30,zmm28 )
        vbroadcastsd( mem(rbx,r9,2),zmm30 )
        vfmadd231pd( zmm0,zmm31,zmm8 )
        vfmadd231pd( zmm1,zmm31,zmm9 )
        vfmadd231pd( zmm2,zmm31,zmm29 )
        vbroadcastsd( mem(rbx,r13,1),zmm31 )
        vfmadd231pd( zmm0,zmm30,zmm10 )
        vfmadd231pd( zmm1,zmm30,zmm11 )
        vfmadd231pd( zmm2,zmm30,zmm26 )
        add( r8,rbx )                                     // b += rs_b
        vfmadd231pd( zmm0,zmm31,zmm12 )
        vfmadd231pd( zmm1,zmm31,zmm13 )
        vfmadd231pd( zmm2,zmm31,zmm27 )

        // ---------------------------------- iteration 4

        vmovupd( mem(rax),zmm0 )                           // load A
        vmovupd( 0x40(rax),zmm1 )
        vmovupd( 0x80(rax),zmm2 MASK_(k(2)) MASK_(z) )     // Load A with mask and zero hint
        add( r10,rax )                                     // a += cs_a
        prefetch( 0,mem(r11,r13,1) )                             // prefetch B
        vbroadcastsd( mem(rbx),zmm30 )
        vbroadcastsd( mem(rbx,r9,1),zmm31 )
        vfmadd231pd( zmm3,zmm30,zmm6 )
        vfmadd231pd( zmm4,zmm30,zmm7 )
        vfmadd231pd( zmm5,zmm30,zmm28 )
        vbroadcastsd( mem(rbx,r9,2),zmm30 )
        vfmadd231pd( zmm3,zmm31,zmm8 )
        vfmadd231pd( zmm4,zmm31,zmm9 )
        vfmadd231pd( zmm5,zmm31,zmm29 )
        vbroadcastsd( mem(rbx,r13,1),zmm31 )
        vfmadd231pd( zmm3,zmm30,zmm10 )
        vfmadd231pd( zmm4,zmm30,zmm11 )
        vfmadd231pd( zmm5,zmm30,zmm26 )
        add( r8,rbx )                                     // b += rs_b
        vfmadd231pd( zmm3,zmm31,zmm12 )
        vfmadd231pd( zmm4,zmm31,zmm13 )
        vfmadd231pd( zmm5,zmm31,zmm27 )

        // ---------------------------------- iteration 5

        vmovupd( mem(rax),zmm3 )                           // load A
        vmovupd( 0x40(rax),zmm4 )
        vmovupd( 0x80(rax),zmm5 MASK_(k(2)) MASK_(z) )     // Load A with mask and zero hint
        add( r10,rax )                                     // a += cs_a
        vbroadcastsd( mem(rbx),zmm30 )
        vbroadcastsd( mem(rbx,r9,1),zmm31 )
        vfmadd231pd( zmm0,zmm30,zmm6 )
        vfmadd231pd( zmm1,zmm30,zmm7 )
        vfmadd231pd( zmm2,zmm30,zmm28 )
        vbroadcastsd( mem(rbx,r9,2),zmm30 )
        vfmadd231pd( zmm0,zmm31,zmm8 )
        vfmadd231pd( zmm1,zmm31,zmm9 )
        vfmadd231pd( zmm2,zmm31,zmm29 )
        vbroadcastsd( mem(rbx,r13,1),zmm31 )
        vfmadd231pd( zmm0,zmm30,zmm10 )
        vfmadd231pd( zmm1,zmm30,zmm11 )
        vfmadd231pd( zmm2,zmm30,zmm26 )
        add( r8,rbx )                                     // b += rs_b
        vfmadd231pd( zmm0,zmm31,zmm12 )
        vfmadd231pd( zmm1,zmm31,zmm13 )
        vfmadd231pd( zmm2,zmm31,zmm27 )

        // ---------------------------------- iteration 6

        vmovupd( mem(rax),zmm0 )                           // load A
        vmovupd( 0x40(rax),zmm1 )
        vmovupd( 0x80(rax),zmm2 MASK_(k(2)) MASK_(z) )     // Load A with mask and zero hint
        add( r10,rax )                                     // a += cs_a
        vbroadcastsd( mem(rbx),zmm30 )
        vbroadcastsd( mem(rbx,r9,1),zmm31 )
        vfmadd231pd( zmm3,zmm30,zmm6 )
        vfmadd231pd( zmm4,zmm30,zmm7 )
        vfmadd231pd( zmm5,zmm30,zmm28 )
        vbroadcastsd( mem(rbx,r9,2),zmm30 )
        vfmadd231pd( zmm3,zmm31,zmm8 )
        vfmadd231pd( zmm4,zmm31,zmm9 )
        vfmadd231pd( zmm5,zmm31,zmm29 )
        vbroadcastsd( mem(rbx,r13,1),zmm31 )
        vfmadd231pd( zmm3,zmm30,zmm10 )
        vfmadd231pd( zmm4,zmm30,zmm11 )
        vfmadd231pd( zmm5,zmm30,zmm26 )
        add( r8,rbx )                                     // b += rs_b
        vfmadd231pd( zmm3,zmm31,zmm12 )
        vfmadd231pd( zmm4,zmm31,zmm13 )
        vfmadd231pd( zmm5,zmm31,zmm27 )

        // ---------------------------------- iteration 7

        vmovupd( mem(rax),zmm3 )                           // load A
        vmovupd( 0x40(rax),zmm4 )
        vmovupd( 0x80(rax),zmm5 MASK_(k(2)) MASK_(z) )     // Load A with mask and zero hint
        add( r10,rax )                                     // a += cs_a
        vbroadcastsd( mem(rbx),zmm30 )
        vbroadcastsd( mem(rbx,r9,1),zmm31 )
        vfmadd231pd( zmm0,zmm30,zmm6 )
        vfmadd231pd( zmm1,zmm30,zmm7 )
        vfmadd231pd( zmm2,zmm30,zmm28 )
        vbroadcastsd( mem(rbx,r9,2),zmm30 )
        vfmadd231pd( zmm0,zmm31,zmm8 )
        vfmadd231pd( zmm1,zmm31,zmm9 )
        vfmadd231pd( zmm2,zmm31,zmm29 )
        vbroadcastsd( mem(rbx,r13,1),zmm31 )
        vfmadd231pd( zmm0,zmm30,zmm10 )
        vfmadd231pd( zmm1,zmm30,zmm11 )
        vfmadd231pd( zmm2,zmm30,zmm26 )
        add( r8,rbx )                                     // b += rs_b
        vfmadd231pd( zmm0,zmm31,zmm12 )
        vfmadd231pd( zmm1,zmm31,zmm13 )
        vfmadd231pd( zmm2,zmm31,zmm27 )

        // ---------------------------------- iteration 8

        vbroadcastsd( mem(rbx),zmm30 )
        vbroadcastsd( mem(rbx,r9,1),zmm31 )
        vfmadd231pd( zmm3,zmm30,zmm6 )
        vfmadd231pd( zmm4,zmm30,zmm7 )
        vfmadd231pd( zmm5,zmm30,zmm28 )
        vbroadcastsd( mem(rbx,r9,2),zmm30 )
        vfmadd231pd( zmm3,zmm31,zmm8 )
        vfmadd231pd( zmm4,zmm31,zmm9 )
        vfmadd231pd( zmm5,zmm31,zmm29 )
        vbroadcastsd( mem(rbx,r13,1),zmm31 )
        vfmadd231pd( zmm3,zmm30,zmm10 )
        vfmadd231pd( zmm4,zmm30,zmm11 )
        vfmadd231pd( zmm5,zmm30,zmm26 )
        add( r8,rbx )                                     // b += rs_b
        vfmadd231pd( zmm3,zmm31,zmm12 )
        vfmadd231pd( zmm4,zmm31,zmm13 )
        vfmadd231pd( zmm5,zmm31,zmm27 )
        lea(mem(r11,r8,8), r11)                            // b_next += 8*rs_b
        dec(rsi)                                           // i -= 1
    jnz(.LOOP1)                                            // iterate again if i != 0.

    label(.PREFETCHLOOP)
    add(imm(4), rsi)                                       // i += NR
    jle(.TAILITER)                                         // jump if i <= 0.

    label(.LOOP2)

        // ---------------------------------- iteration 1
        prefetchw0( mem(rdx))                              // prefetch C
        vmovupd( mem(rax),zmm0 )                           // load A
        vmovupd( 0x40(rax),zmm1 )
        vmovupd( 0x80(rax),zmm2 MASK_(k(2)) MASK_(z) )     // Load A with mask and zero hint
        add( r10,rax )                                     // a += cs_a
        vmovupd( mem(rax),zmm3 )                           // load A
        vmovupd( 0x40(rax),zmm4 )
        vmovupd( 0x80(rax),zmm5 MASK_(k(2)) MASK_(z) )     // Load A with mask and zero hint
        add( r10,rax )                                     // a += cs_a
        prefetch( 0,mem(r11) )                             // prefetch B
        vbroadcastsd( mem(rbx),zmm30 )
        vbroadcastsd( mem(rbx,r9,1),zmm31 )
        vfmadd231pd( zmm0,zmm30,zmm6 )
        vfmadd231pd( zmm1,zmm30,zmm7 )
        vfmadd231pd( zmm2,zmm30,zmm28 )
        vbroadcastsd( mem(rbx,r9,2),zmm30 )
        vfmadd231pd( zmm0,zmm31,zmm8 )
        vfmadd231pd( zmm1,zmm31,zmm9 )
        vfmadd231pd( zmm2,zmm31,zmm29 )
        vbroadcastsd( mem(rbx,r13,1),zmm31 )
        vfmadd231pd( zmm0,zmm30,zmm10 )
        vfmadd231pd( zmm1,zmm30,zmm11 )
        vfmadd231pd( zmm2,zmm30,zmm26 )
        add( r8,rbx )                                     // b += rs_b
        vfmadd231pd( zmm0,zmm31,zmm12 )
        vfmadd231pd( zmm1,zmm31,zmm13 )
        vfmadd231pd( zmm2,zmm31,zmm27 )

        // ---------------------------------- iteration 2
        prefetchw0( mem(rdx, 64))                          // prefetch C
        vmovupd( mem(rax),zmm0 )                           // load A
        vmovupd( 0x40(rax),zmm1 )
        vmovupd( 0x80(rax),zmm2 MASK_(k(2)) MASK_(z) )     // Load A with mask and zero hint
        add( r10,rax )                                     // a += cs_a
        prefetch( 0,mem(r11,r9,1) )                             // prefetch B
        vbroadcastsd( mem(rbx),zmm30 )
        vbroadcastsd( mem(rbx,r9,1),zmm31 )
        vfmadd231pd( zmm3,zmm30,zmm6 )
        vfmadd231pd( zmm4,zmm30,zmm7 )
        vfmadd231pd( zmm5,zmm30,zmm28 )
        vbroadcastsd( mem(rbx,r9,2),zmm30 )
        vfmadd231pd( zmm3,zmm31,zmm8 )
        vfmadd231pd( zmm4,zmm31,zmm9 )
        vfmadd231pd( zmm5,zmm31,zmm29 )
        vbroadcastsd( mem(rbx,r13,1),zmm31 )
        vfmadd231pd( zmm3,zmm30,zmm10 )
        vfmadd231pd( zmm4,zmm30,zmm11 )
        vfmadd231pd( zmm5,zmm30,zmm26 )
        add( r8,rbx )                                     // b += rs_b
        vfmadd231pd( zmm3,zmm31,zmm12 )
        vfmadd231pd( zmm4,zmm31,zmm13 )
        vfmadd231pd( zmm5,zmm31,zmm27 )

        // ---------------------------------- iteration 3
        prefetchw0( mem(rdx, 128))                        // prefetch C
        vmovupd( mem(rax),zmm3 )                           // load A
        vmovupd( 0x40(rax),zmm4 )
        vmovupd( 0x80(rax),zmm5 MASK_(k(2)) MASK_(z) )     // Load A with mask and zero hint
        add( r10,rax )                                     // a += cs_a
        prefetch( 0,mem(r11,r9,2) )                             // prefetch B
        vbroadcastsd( mem(rbx),zmm30 )
        vbroadcastsd( mem(rbx,r9,1),zmm31 )
        vfmadd231pd( zmm0,zmm30,zmm6 )
        vfmadd231pd( zmm1,zmm30,zmm7 )
        vfmadd231pd( zmm2,zmm30,zmm28 )
        vbroadcastsd( mem(rbx,r9,2),zmm30 )
        vfmadd231pd( zmm0,zmm31,zmm8 )
        vfmadd231pd( zmm1,zmm31,zmm9 )
        vfmadd231pd( zmm2,zmm31,zmm29 )
        vbroadcastsd( mem(rbx,r13,1),zmm31 )
        vfmadd231pd( zmm0,zmm30,zmm10 )
        vfmadd231pd( zmm1,zmm30,zmm11 )
        vfmadd231pd( zmm2,zmm30,zmm26 )
        add( r8,rbx )                                     // b += rs_b
        vfmadd231pd( zmm0,zmm31,zmm12 )
        vfmadd231pd( zmm1,zmm31,zmm13 )
        vfmadd231pd( zmm2,zmm31,zmm27 )

        // ---------------------------------- iteration 4
        vmovupd( mem(rax),zmm0 )                           // load A
        vmovupd( 0x40(rax),zmm1 )
        vmovupd( 0x80(rax),zmm2 MASK_(k(2)) MASK_(z) )     // Load A with mask and zero hint
        add( r10,rax )                                     // a += cs_a
        prefetch( 0,mem(r11,r13,1) )                             // prefetch B
        vbroadcastsd( mem(rbx),zmm30 )
        vbroadcastsd( mem(rbx,r9,1),zmm31 )
        vfmadd231pd( zmm3,zmm30,zmm6 )
        vfmadd231pd( zmm4,zmm30,zmm7 )
        vfmadd231pd( zmm5,zmm30,zmm28 )
        vbroadcastsd( mem(rbx,r9,2),zmm30 )
        vfmadd231pd( zmm3,zmm31,zmm8 )
        vfmadd231pd( zmm4,zmm31,zmm9 )
        vfmadd231pd( zmm5,zmm31,zmm29 )
        vbroadcastsd( mem(rbx,r13,1),zmm31 )
        vfmadd231pd( zmm3,zmm30,zmm10 )
        vfmadd231pd( zmm4,zmm30,zmm11 )
        vfmadd231pd( zmm5,zmm30,zmm26 )
        add( r8,rbx )                                     // b += rs_b
        vfmadd231pd( zmm3,zmm31,zmm12 )
        vfmadd231pd( zmm4,zmm31,zmm13 )
        vfmadd231pd( zmm5,zmm31,zmm27 )

        // ---------------------------------- iteration 5
        vmovupd( mem(rax),zmm3 )                           // load A
        vmovupd( 0x40(rax),zmm4 )
        vmovupd( 0x80(rax),zmm5 MASK_(k(2)) MASK_(z) )     // Load A with mask and zero hint
        add( r10,rax )                                     // a += cs_a
        vbroadcastsd( mem(rbx),zmm30 )
        vbroadcastsd( mem(rbx,r9,1),zmm31 )
        vfmadd231pd( zmm0,zmm30,zmm6 )
        vfmadd231pd( zmm1,zmm30,zmm7 )
        vfmadd231pd( zmm2,zmm30,zmm28 )
        vbroadcastsd( mem(rbx,r9,2),zmm30 )
        vfmadd231pd( zmm0,zmm31,zmm8 )
        vfmadd231pd( zmm1,zmm31,zmm9 )
        vfmadd231pd( zmm2,zmm31,zmm29 )
        vbroadcastsd( mem(rbx,r13,1),zmm31 )
        vfmadd231pd( zmm0,zmm30,zmm10 )
        vfmadd231pd( zmm1,zmm30,zmm11 )
        vfmadd231pd( zmm2,zmm30,zmm26 )
        add( r8,rbx )                                     // b += rs_b
        vfmadd231pd( zmm0,zmm31,zmm12 )
        vfmadd231pd( zmm1,zmm31,zmm13 )
        vfmadd231pd( zmm2,zmm31,zmm27 )

        // ---------------------------------- iteration 6
        vmovupd( mem(rax),zmm0 )                           // load A
        vmovupd( 0x40(rax),zmm1 )
        vmovupd( 0x80(rax),zmm2 MASK_(k(2)) MASK_(z) )     // Load A with mask and zero hint
        add( r10,rax )                                     // a += cs_a
        vbroadcastsd( mem(rbx),zmm30 )
        vbroadcastsd( mem(rbx,r9,1),zmm31 )
        vfmadd231pd( zmm3,zmm30,zmm6 )
        vfmadd231pd( zmm4,zmm30,zmm7 )
        vfmadd231pd( zmm5,zmm30,zmm28 )
        vbroadcastsd( mem(rbx,r9,2),zmm30 )
        vfmadd231pd( zmm3,zmm31,zmm8 )
        vfmadd231pd( zmm4,zmm31,zmm9 )
        vfmadd231pd( zmm5,zmm31,zmm29 )
        vbroadcastsd( mem(rbx,r13,1),zmm31 )
        vfmadd231pd( zmm3,zmm30,zmm10 )
        vfmadd231pd( zmm4,zmm30,zmm11 )
        vfmadd231pd( zmm5,zmm30,zmm26 )
        add( r8,rbx )                                     // b += rs_b
        vfmadd231pd( zmm3,zmm31,zmm12 )
        vfmadd231pd( zmm4,zmm31,zmm13 )
        vfmadd231pd( zmm5,zmm31,zmm27 )

        // ---------------------------------- iteration 7
        vmovupd( mem(rax),zmm3 )                           // load A
        vmovupd( 0x40(rax),zmm4 )
        vmovupd( 0x80(rax),zmm5 MASK_(k(2)) MASK_(z) )     // Load A with mask and zero hint
        add( r10,rax )                                     // a += cs_a
        vbroadcastsd( mem(rbx),zmm30 )
        vbroadcastsd( mem(rbx,r9,1),zmm31 )
        vfmadd231pd( zmm0,zmm30,zmm6 )
        vfmadd231pd( zmm1,zmm30,zmm7 )
        vfmadd231pd( zmm2,zmm30,zmm28 )
        vbroadcastsd( mem(rbx,r9,2),zmm30 )
        vfmadd231pd( zmm0,zmm31,zmm8 )
        vfmadd231pd( zmm1,zmm31,zmm9 )
        vfmadd231pd( zmm2,zmm31,zmm29 )
        vbroadcastsd( mem(rbx,r13,1),zmm31 )
        vfmadd231pd( zmm0,zmm30,zmm10 )
        vfmadd231pd( zmm1,zmm30,zmm11 )
        vfmadd231pd( zmm2,zmm30,zmm26 )
        add( r8,rbx )                                     // b += rs_b
        vfmadd231pd( zmm0,zmm31,zmm12 )
        vfmadd231pd( zmm1,zmm31,zmm13 )
        vfmadd231pd( zmm2,zmm31,zmm27 )

        // ---------------------------------- iteration 8
        vbroadcastsd( mem(rbx),zmm30 )
        vbroadcastsd( mem(rbx,r9,1),zmm31 )
        vfmadd231pd( zmm3,zmm30,zmm6 )
        vfmadd231pd( zmm4,zmm30,zmm7 )
        vfmadd231pd( zmm5,zmm30,zmm28 )
        vbroadcastsd( mem(rbx,r9,2),zmm30 )
        vfmadd231pd( zmm3,zmm31,zmm8 )
        vfmadd231pd( zmm4,zmm31,zmm9 )
        vfmadd231pd( zmm5,zmm31,zmm29 )
        vbroadcastsd( mem(rbx,r13,1),zmm31 )
        vfmadd231pd( zmm3,zmm30,zmm10 )
        vfmadd231pd( zmm4,zmm30,zmm11 )
        vfmadd231pd( zmm5,zmm30,zmm26 )
        add( r8,rbx )                                     // b += rs_b
        vfmadd231pd( zmm3,zmm31,zmm12 )
        vfmadd231pd( zmm4,zmm31,zmm13 )
        vfmadd231pd( zmm5,zmm31,zmm27 )
        lea(mem(rdx, rdi, 1), rdx)                         // C += cs_c
        lea(mem(r11,r8,8), r11)                            // b_next += 8*rs_b
        sub(imm(1), rsi)                                   // i -= 1
    jnz(.LOOP2)                                            // iterate again if i != 0.
    label(.TAILITER)
    add(imm(TAIL_NITER), rsi)                              // i += TAIL_NITER
    jle(.TAIL)                                             // jump if i <= 0

    label(.LOOP3)

        // ---------------------------------- iteration 1
        vmovupd( mem(rax),zmm0 )                           // load A
        vmovupd( 0x40(rax),zmm1 )
        vmovupd( 0x80(rax),zmm2 MASK_(k(2)) MASK_(z) )     // Load A with mask and zero hint
        add( r10,rax )                                     // a += cs_a
        vmovupd( mem(rax),zmm3 )                           // load A
        vmovupd( 0x40(rax),zmm4 )
        vmovupd( 0x80(rax),zmm5 MASK_(k(2)) MASK_(z) )     // Load A with mask and zero hint
        add( r10,rax )                                     // a += cs_a
        prefetch( 0,mem(r11) )                             // prefetch B
        vbroadcastsd( mem(rbx),zmm30 )
        vbroadcastsd( mem(rbx,r9,1),zmm31 )
        vfmadd231pd( zmm0,zmm30,zmm6 )
        vfmadd231pd( zmm1,zmm30,zmm7 )
        vfmadd231pd( zmm2,zmm30,zmm28 )
        vbroadcastsd( mem(rbx,r9,2),zmm30 )
        vfmadd231pd( zmm0,zmm31,zmm8 )
        vfmadd231pd( zmm1,zmm31,zmm9 )
        vfmadd231pd( zmm2,zmm31,zmm29 )
        vbroadcastsd( mem(rbx,r13,1),zmm31 )
        vfmadd231pd( zmm0,zmm30,zmm10 )
        vfmadd231pd( zmm1,zmm30,zmm11 )
        vfmadd231pd( zmm2,zmm30,zmm26 )
        add( r8,rbx )                                     // b += rs_b
        vfmadd231pd( zmm0,zmm31,zmm12 )
        vfmadd231pd( zmm1,zmm31,zmm13 )
        vfmadd231pd( zmm2,zmm31,zmm27 )

        // ---------------------------------- iteration 2
        vmovupd( mem(rax),zmm0 )                           // load A
        vmovupd( 0x40(rax),zmm1 )
        vmovupd( 0x80(rax),zmm2 MASK_(k(2)) MASK_(z) )     // Load A with mask and zero hint
        add( r10,rax )                                     // a += cs_a
        prefetch( 0,mem(r11,r9,1) )                             // prefetch B
        vbroadcastsd( mem(rbx),zmm30 )
        vbroadcastsd( mem(rbx,r9,1),zmm31 )
        vfmadd231pd( zmm3,zmm30,zmm6 )
        vfmadd231pd( zmm4,zmm30,zmm7 )
        vfmadd231pd( zmm5,zmm30,zmm28 )
        vbroadcastsd( mem(rbx,r9,2),zmm30 )
        vfmadd231pd( zmm3,zmm31,zmm8 )
        vfmadd231pd( zmm4,zmm31,zmm9 )
        vfmadd231pd( zmm5,zmm31,zmm29 )
        vbroadcastsd( mem(rbx,r13,1),zmm31 )
        vfmadd231pd( zmm3,zmm30,zmm10 )
        vfmadd231pd( zmm4,zmm30,zmm11 )
        vfmadd231pd( zmm5,zmm30,zmm26 )
        add( r8,rbx )                                     // b += rs_b
        vfmadd231pd( zmm3,zmm31,zmm12 )
        vfmadd231pd( zmm4,zmm31,zmm13 )
        vfmadd231pd( zmm5,zmm31,zmm27 )

        // ---------------------------------- iteration 3
        vmovupd( mem(rax),zmm3 )                           // load A
        vmovupd( 0x40(rax),zmm4 )
        vmovupd( 0x80(rax),zmm5 MASK_(k(2)) MASK_(z) )     // Load A with mask and zero hint
        add( r10,rax )                                     // a += cs_a
        prefetch( 0,mem(r11,r9,2) )                             // prefetch B
        vbroadcastsd( mem(rbx),zmm30 )
        vbroadcastsd( mem(rbx,r9,1),zmm31 )
        vfmadd231pd( zmm0,zmm30,zmm6 )
        vfmadd231pd( zmm1,zmm30,zmm7 )
        vfmadd231pd( zmm2,zmm30,zmm28 )
        vbroadcastsd( mem(rbx,r9,2),zmm30 )
        vfmadd231pd( zmm0,zmm31,zmm8 )
        vfmadd231pd( zmm1,zmm31,zmm9 )
        vfmadd231pd( zmm2,zmm31,zmm29 )
        vbroadcastsd( mem(rbx,r13,1),zmm31 )
        vfmadd231pd( zmm0,zmm30,zmm10 )
        vfmadd231pd( zmm1,zmm30,zmm11 )
        vfmadd231pd( zmm2,zmm30,zmm26 )
        add( r8,rbx )                                     // b += rs_b
        vfmadd231pd( zmm0,zmm31,zmm12 )
        vfmadd231pd( zmm1,zmm31,zmm13 )
        vfmadd231pd( zmm2,zmm31,zmm27 )

        // ---------------------------------- iteration 4
        vmovupd( mem(rax),zmm0 )                           // load A
        vmovupd( 0x40(rax),zmm1 )
        vmovupd( 0x80(rax),zmm2 MASK_(k(2)) MASK_(z) )     // Load A with mask and zero hint
        add( r10,rax )                                     // a += cs_a
        prefetch( 0,mem(r11,r13,1) )                             // prefetch B
        vbroadcastsd( mem(rbx),zmm30 )
        vbroadcastsd( mem(rbx,r9,1),zmm31 )
        vfmadd231pd( zmm3,zmm30,zmm6 )
        vfmadd231pd( zmm4,zmm30,zmm7 )
        vfmadd231pd( zmm5,zmm30,zmm28 )
        vbroadcastsd( mem(rbx,r9,2),zmm30 )
        vfmadd231pd( zmm3,zmm31,zmm8 )
        vfmadd231pd( zmm4,zmm31,zmm9 )
        vfmadd231pd( zmm5,zmm31,zmm29 )
        vbroadcastsd( mem(rbx,r13,1),zmm31 )
        vfmadd231pd( zmm3,zmm30,zmm10 )
        vfmadd231pd( zmm4,zmm30,zmm11 )
        vfmadd231pd( zmm5,zmm30,zmm26 )
        add( r8,rbx )                                     // b += rs_b
        vfmadd231pd( zmm3,zmm31,zmm12 )
        vfmadd231pd( zmm4,zmm31,zmm13 )
        vfmadd231pd( zmm5,zmm31,zmm27 )

        // ---------------------------------- iteration 5
        vmovupd( mem(rax),zmm3 )                           // load A
        vmovupd( 0x40(rax),zmm4 )
        vmovupd( 0x80(rax),zmm5 MASK_(k(2)) MASK_(z) )     // Load A with mask and zero hint
        add( r10,rax )                                     // a += cs_a
        vbroadcastsd( mem(rbx),zmm30 )
        vbroadcastsd( mem(rbx,r9,1),zmm31 )
        vfmadd231pd( zmm0,zmm30,zmm6 )
        vfmadd231pd( zmm1,zmm30,zmm7 )
        vfmadd231pd( zmm2,zmm30,zmm28 )
        vbroadcastsd( mem(rbx,r9,2),zmm30 )
        vfmadd231pd( zmm0,zmm31,zmm8 )
        vfmadd231pd( zmm1,zmm31,zmm9 )
        vfmadd231pd( zmm2,zmm31,zmm29 )
        vbroadcastsd( mem(rbx,r13,1),zmm31 )
        vfmadd231pd( zmm0,zmm30,zmm10 )
        vfmadd231pd( zmm1,zmm30,zmm11 )
        vfmadd231pd( zmm2,zmm30,zmm26 )
        add( r8,rbx )                                     // b += rs_b
        vfmadd231pd( zmm0,zmm31,zmm12 )
        vfmadd231pd( zmm1,zmm31,zmm13 )
        vfmadd231pd( zmm2,zmm31,zmm27 )

        // ---------------------------------- iteration 6
        vmovupd( mem(rax),zmm0 )                           // load A
        vmovupd( 0x40(rax),zmm1 )
        vmovupd( 0x80(rax),zmm2 MASK_(k(2)) MASK_(z) )     // Load A with mask and zero hint
        add( r10,rax )                                     // a += cs_a
        vbroadcastsd( mem(rbx),zmm30 )
        vbroadcastsd( mem(rbx,r9,1),zmm31 )
        vfmadd231pd( zmm3,zmm30,zmm6 )
        vfmadd231pd( zmm4,zmm30,zmm7 )
        vfmadd231pd( zmm5,zmm30,zmm28 )
        vbroadcastsd( mem(rbx,r9,2),zmm30 )
        vfmadd231pd( zmm3,zmm31,zmm8 )
        vfmadd231pd( zmm4,zmm31,zmm9 )
        vfmadd231pd( zmm5,zmm31,zmm29 )
        vbroadcastsd( mem(rbx,r13,1),zmm31 )
        vfmadd231pd( zmm3,zmm30,zmm10 )
        vfmadd231pd( zmm4,zmm30,zmm11 )
        vfmadd231pd( zmm5,zmm30,zmm26 )
        add( r8,rbx )                                     // b += rs_b
        vfmadd231pd( zmm3,zmm31,zmm12 )
        vfmadd231pd( zmm4,zmm31,zmm13 )
        vfmadd231pd( zmm5,zmm31,zmm27 )

        // ---------------------------------- iteration 7
        vmovupd( mem(rax),zmm3 )                           // load A
        vmovupd( 0x40(rax),zmm4 )
        vmovupd( 0x80(rax),zmm5 MASK_(k(2)) MASK_(z) )     // Load A with mask and zero hint
        add( r10,rax )                                     // a += cs_a
        vbroadcastsd( mem(rbx),zmm30 )
        vbroadcastsd( mem(rbx,r9,1),zmm31 )
        vfmadd231pd( zmm0,zmm30,zmm6 )
        vfmadd231pd( zmm1,zmm30,zmm7 )
        vfmadd231pd( zmm2,zmm30,zmm28 )
        vbroadcastsd( mem(rbx,r9,2),zmm30 )
        vfmadd231pd( zmm0,zmm31,zmm8 )
        vfmadd231pd( zmm1,zmm31,zmm9 )
        vfmadd231pd( zmm2,zmm31,zmm29 )
        vbroadcastsd( mem(rbx,r13,1),zmm31 )
        vfmadd231pd( zmm0,zmm30,zmm10 )
        vfmadd231pd( zmm1,zmm30,zmm11 )
        vfmadd231pd( zmm2,zmm30,zmm26 )
        add( r8,rbx )                                     // b += rs_b
        vfmadd231pd( zmm0,zmm31,zmm12 )
        vfmadd231pd( zmm1,zmm31,zmm13 )
        vfmadd231pd( zmm2,zmm31,zmm27 )

        // ---------------------------------- iteration 8
        vbroadcastsd( mem(rbx),zmm30 )
        vbroadcastsd( mem(rbx,r9,1),zmm31 )
        vfmadd231pd( zmm3,zmm30,zmm6 )
        vfmadd231pd( zmm4,zmm30,zmm7 )
        vfmadd231pd( zmm5,zmm30,zmm28 )
        vbroadcastsd( mem(rbx,r9,2),zmm30 )
        vfmadd231pd( zmm3,zmm31,zmm8 )
        vfmadd231pd( zmm4,zmm31,zmm9 )
        vfmadd231pd( zmm5,zmm31,zmm29 )
        vbroadcastsd( mem(rbx,r13,1),zmm31 )
        vfmadd231pd( zmm3,zmm30,zmm10 )
        vfmadd231pd( zmm4,zmm30,zmm11 )
        vfmadd231pd( zmm5,zmm30,zmm26 )
        add( r8,rbx )                                     // b += rs_b
        vfmadd231pd( zmm3,zmm31,zmm12 )
        vfmadd231pd( zmm4,zmm31,zmm13 )
        vfmadd231pd( zmm5,zmm31,zmm27 )
        lea(mem(r11,r8,8), r11)                            // b_next += 8*rs_b
        dec(rsi)                                           // i -= 1
    jnz(.LOOP3)                                            // iterate again if i != 0.


    label(.TAIL)
    mov(var(k_left), rsi)                                  // i = k_left
    test(rsi, rsi)                                         // check i via logical AND
    je(.DPOSTACCUM)                                        // if i == 0, jump to post-accumulation

    label(.DLOOPKLEFT)                                     // k_left loop
        vmovupd( mem(rax),zmm0 )                           // load A
        vmovupd( 0x40(rax),zmm1 )
        vmovupd( 0x80(rax),zmm2 MASK_(k(2)) MASK_(z) )     // Load A with mask and zero hint
        add( r10,rax )                                     // a += cs_a
        vbroadcastsd( mem(rbx),zmm30 )
        vbroadcastsd( mem(rbx,r9,1),zmm31 )
        vfmadd231pd( zmm0,zmm30,zmm6 )
        vfmadd231pd( zmm1,zmm30,zmm7 )
        vfmadd231pd( zmm2,zmm30,zmm28 )
        vbroadcastsd( mem(rbx,r9,2),zmm30 )
        vfmadd231pd( zmm0,zmm31,zmm8 )
        vfmadd231pd( zmm1,zmm31,zmm9 )
        vfmadd231pd( zmm2,zmm31,zmm29 )
        vbroadcastsd( mem(rbx,r13,1),zmm31 )
        vfmadd231pd( zmm0,zmm30,zmm10 )
        vfmadd231pd( zmm1,zmm30,zmm11 )
        vfmadd231pd( zmm2,zmm30,zmm26 )
        add( r8,rbx )                                     // b += rs_b
        vfmadd231pd( zmm0,zmm31,zmm12 )
        vfmadd231pd( zmm1,zmm31,zmm13 )
        vfmadd231pd( zmm2,zmm31,zmm27 )
        dec(rsi)                                           // i -= 1
    jne(.DLOOPKLEFT)                                       // iterate again if i != 0.


    label(.DPOSTACCUM)
    mov(var(alpha), rdx)                                   // load address of alpha
    vbroadcastsd(mem(rdx), zmm30)                           // broadcast alpha
    mov(var(beta), rax)                                    // load address of beta
    vbroadcastsd(mem(rax), zmm31)                           // broadcast beta

    // scale by alpha
    vmulpd( zmm30,zmm6,zmm6 )
    vmulpd( zmm30,zmm7,zmm7 )
    vmulpd( zmm30,zmm28,zmm28 )
    vmulpd( zmm30,zmm8,zmm8 )
    vmulpd( zmm30,zmm9,zmm9 )
    vmulpd( zmm30,zmm29,zmm29 )
    vmulpd( zmm30,zmm10,zmm10 )
    vmulpd( zmm30,zmm11,zmm11 )
    vmulpd( zmm30,zmm26,zmm26 )
    vmulpd( zmm30,zmm12,zmm12 )
    vmulpd( zmm30,zmm13,zmm13 )
    vmulpd( zmm30,zmm27,zmm27 )


    mov(var(rs_c), rsi)                                    // load rs_c
    lea(mem(, rsi, 8), rsi)                                // rsi = rs_c * sizeof(double)
    lea(mem(rdi, rdi, 2), r13)                             // r13 = 3*cs_c
    vxorpd(ymm2, ymm2, ymm2)
    vucomisd(xmm2, xmm31)                                   // set ZF if beta == 0
    je(.DBETAZERO)                                         // if ZF == 1, jump to beta == 0 case


    cmp(imm(8), rdi)                                       // set ZF if (8*cs_c) == 8


    jz(.DROWSTORED)                                        // jump to row storage case

    label(.DCOLSTORED)
    vmovupd( mem(rcx),zmm0)
    vfmadd231pd( zmm0,zmm31,zmm6)
    vmovupd( zmm6,(rcx))
    vmovupd( 0x40(rcx),zmm1)
    vfmadd231pd( zmm1,zmm31,zmm7)
    vmovupd( zmm7,0x40(rcx))
    vmovupd( 0x80(rcx),zmm2 MASK_(k(2)) MASK_(z))        // Load C using mask and zero hint
    vfmadd231pd( zmm2,zmm31,zmm28)
    vmovupd( zmm28,0x80(rcx) MASK_(k(2)))                // store to C with mask
    vmovupd( mem(rcx,rdi,1),zmm3)
    vfmadd231pd( zmm3,zmm31,zmm8)
    vmovupd( zmm8,(rcx,rdi,1))
    vmovupd( 0x40(rcx,rdi,1),zmm4)
    vfmadd231pd( zmm4,zmm31,zmm9)
    vmovupd( zmm9,0x40(rcx,rdi,1))
    vmovupd( 0x80(rcx,rdi,1),zmm5 MASK_(k(2)) MASK_(z))        // Load C using mask and zero hint
    vfmadd231pd( zmm5,zmm31,zmm29)
    vmovupd( zmm29,0x80(rcx,rdi,1) MASK_(k(2)))                // store to C with mask
    vmovupd( mem(rcx,rdi,2),zmm0)
    vfmadd231pd( zmm0,zmm31,zmm10)
    vmovupd( zmm10,(rcx,rdi,2))
    vmovupd( 0x40(rcx,rdi,2),zmm1)
    vfmadd231pd( zmm1,zmm31,zmm11)
    vmovupd( zmm11,0x40(rcx,rdi,2))
    vmovupd( 0x80(rcx,rdi,2),zmm2 MASK_(k(2)) MASK_(z))        // Load C using mask and zero hint
    vfmadd231pd( zmm2,zmm31,zmm26)
    vmovupd( zmm26,0x80(rcx,rdi,2) MASK_(k(2)))                // store to C with mask
    vmovupd( mem(rcx,r13,1),zmm3)
    vfmadd231pd( zmm3,zmm31,zmm12)
    vmovupd( zmm12,(rcx,r13,1))
    vmovupd( 0x40(rcx,r13,1),zmm4)
    vfmadd231pd( zmm4,zmm31,zmm13)
    vmovupd( zmm13,0x40(rcx,r13,1))
    vmovupd( 0x80(rcx,r13,1),zmm5 MASK_(k(2)) MASK_(z))        // Load C using mask and zero hint
    vfmadd231pd( zmm5,zmm31,zmm27)
    vmovupd( zmm27,0x80(rcx,r13,1) MASK_(k(2)))                // store to C with mask

    jmp(.DDONE)                                           // jump to end.

    label(.DROWSTORED)

    // yet to be implemented
    jmp(.DDONE)                                          // jump to end.


    label(.DBETAZERO)
    cmp(imm(8), rdi)                                     // set ZF if (8*cs_c) == 8

    jz(.DROWSTORBZ)                                      // jump to row storage case
    label(.DCOLSTORBZ)
    vmovupd( zmm6,(rcx))
    vmovupd( zmm7,0x40(rcx))
    vmovupd( zmm28,0x80(rcx) MASK_(k(2)))                // store to C with mask
    vmovupd( zmm8,(rcx,rdi,1))
    vmovupd( zmm9,0x40(rcx,rdi,1))
    vmovupd( zmm29,0x80(rcx,rdi,1) MASK_(k(2)))                // store to C with mask
    vmovupd( zmm10,(rcx,rdi,2))
    vmovupd( zmm11,0x40(rcx,rdi,2))
    vmovupd( zmm26,0x80(rcx,rdi,2) MASK_(k(2)))                // store to C with mask
    vmovupd( zmm12,(rcx,r13,1))
    vmovupd( zmm13,0x40(rcx,r13,1))
    vmovupd( zmm27,0x80(rcx,r13,1) MASK_(k(2)))                // store to C with mask

    jmp(.DDONE)                                          // jump to end.


    label(.DROWSTORBZ)

    // yet to be implemented
    label(.DDONE)


    vzeroupper()

    end_asm(
      : // output operands (none)
      : // input operands
        [k_iter] "m" (k_iter),
        [k_left] "m" (k_left),
        [a]      "m" (a),
        [rs_a]   "m" (rs_a),
        [cs_a]   "m" (cs_a),
        [ps_a8]  "m" (ps_a8),
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
        [mask]   "m" (mask)
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
    AOCL_DTL_TRACE_EXIT(AOCL_DTL_LEVEL_TRACE_7);
}


void bli_dgemmsup_rv_zen4_asm_16x4
(
       conj_t    conja,
       conj_t    conjb,
       dim_t     m0,
       dim_t     n0,
       dim_t     k0,
       double*    restrict alpha,
       double*    restrict a, inc_t rs_a0, inc_t cs_a0,
       double*    restrict b, inc_t rs_b0, inc_t cs_b0,
       double*    restrict beta,
       double*    restrict c, inc_t rs_c0, inc_t cs_c0,
       auxinfo_t* restrict data,
       cntx_t*    restrict cntx
     )
{
    AOCL_DTL_TRACE_ENTRY(AOCL_DTL_LEVEL_TRACE_7);
    // Typecast local copies of integers in case dim_t and inc_t are a
    // different size than is expected by load instructions.
    uint64_t rs_a   = rs_a0;
    uint64_t cs_a   = cs_a0;
    uint64_t rs_b   = rs_b0;
    uint64_t cs_b   = cs_b0;
    uint64_t rs_c   = rs_c0;
    uint64_t cs_c   = cs_c0;

    uint64_t ps_a = bli_auxinfo_ps_a( data );
    uint64_t ps_a8  = ps_a * sizeof( double );

    uint64_t k_iter = (uint64_t)k0 / 8;
    uint64_t k_left = (uint64_t)k0 % 8;

    /* 8 double precision elements can be loaded into a 512-bit register
     * So, we use an 8-bit mask to specify which elements to be loaded/stored
     * into/from the register. m_left % 8 specifies how many number of elements
     *  are to be loaded/stored into/from the last register.
     * For example, if m_left = 19, m0 & 7 becomes 3 which indicates that 3 elements
     * have to be loaded/stored into/from register, so shift 0xff(11111111) by (8-3)
     *  times to the right which makes the mask to be (00000111)
     */
    uint8_t mask = 0xff >> (0x8 - (m0 & 7)); // calculate mask based on m_left
    // For special cases where m_left = 24/16/8, all 8 elements have to be loaded/stored
    // So, mask becomes 0xff(11111111)
    if (mask == 0) mask = 0xff;

        // -------------------------------------------------------------------------
    begin_asm()

    mov(var(a), rax)                // load address of a
    mov(var(cs_a), r10)             // load cs_a
    mov(var(b), rbx)                // load address of b
    mov(var(rs_b), r8)              // load rs_b
    mov(var(cs_b), r9)              // load cs_b
    mov(var(c), rcx)                // load address of c
    mov(var(cs_c), rdi)             // load cs_c
    mov(var(mask), rdx)             // load mask
    kmovw(edx, k(2))                // move mask to k2 register
    lea(mem(, r8, 8), r8)           // rs_b *= sizeof(double)
    lea(mem(, r9, 8), r9)           // cs_b *= sizeof(double)
    lea(mem(, r10, 8), r10)         // cs_a *= sizeof(double)
    lea(mem(, rdi, 8), rdi)         // cs_c *= sizeof(double)
    lea(mem(r9, r9, 2 ), r13)       // r13 = 3*cs_b
    lea(mem(rcx, 7*8), rdx)         // C for prefetching
    lea(mem(rbx, r8, 8, 7*8), r11)  // r11 = rbx + 8*rs_b(B for prefetching)

    /* Register usage: zmm0-5 are used to load A matrix
     *                 zmm6-29 are used for accumulation
     *                 zmm30-31 are used for broadcasting B matrix
     */

    // zero out all accumulation registers
    vxorpd(zmm6, zmm6, zmm6)
    vxorpd(zmm7, zmm7, zmm7)
    vxorpd(zmm8, zmm8, zmm8)
    vxorpd(zmm9, zmm9, zmm9)
    vxorpd(zmm10, zmm10, zmm10)
    vxorpd(zmm11, zmm11, zmm11)
    vxorpd(zmm12, zmm12, zmm12)
    vxorpd(zmm13, zmm13, zmm13)

    // K is unrolled by 8 to facilitate prefetch of B
    // Assuming B to be col-stored, for each iteration of K,
    //one cacheline of B_next is prefetched where b_next = b + (unroll)*rs_b
    label(.DLOOPKITER)                                     // main loop
    mov(var(k_iter), rsi)                                  // i = k_iter
    sub(imm( 4+TAIL_NITER), rsi)                           // i -= NR + TAIL_NITER
    jle(.PREFETCHLOOP)                                     // jump if i <= 0

    label(.LOOP1)

        // ---------------------------------- iteration 1

        vmovupd( mem(rax),zmm0 )                           // load A
        vmovupd( 0x40(rax),zmm1 MASK_(k(2)) MASK_(z) )     // Load A with mask and zero hint
        add( r10,rax )                                     // a += cs_a
        vmovupd( mem(rax),zmm3 )                           // load A
        vmovupd( 0x40(rax),zmm4 MASK_(k(2)) MASK_(z) )     // Load A with mask and zero hint
        add( r10,rax )                                     // a += cs_a
        prefetch( 0,mem(r11) )                             // prefetch B
        vbroadcastsd( mem(rbx),zmm30 )
        vbroadcastsd( mem(rbx,r9,1),zmm31 )
        vfmadd231pd( zmm0,zmm30,zmm6 )
        vfmadd231pd( zmm1,zmm30,zmm7 )
        vbroadcastsd( mem(rbx,r9,2),zmm30 )
        vfmadd231pd( zmm0,zmm31,zmm8 )
        vfmadd231pd( zmm1,zmm31,zmm9 )
        vbroadcastsd( mem(rbx,r13,1),zmm31 )
        vfmadd231pd( zmm0,zmm30,zmm10 )
        vfmadd231pd( zmm1,zmm30,zmm11 )
        add( r8,rbx )                                     // b += rs_b
        vfmadd231pd( zmm0,zmm31,zmm12 )
        vfmadd231pd( zmm1,zmm31,zmm13 )

        // ---------------------------------- iteration 2

        vmovupd( mem(rax),zmm0 )                           // load A
        vmovupd( 0x40(rax),zmm1 MASK_(k(2)) MASK_(z) )     // Load A with mask and zero hint
        add( r10,rax )                                     // a += cs_a
        prefetch( 0,mem(r11,r9,1) )                             // prefetch B
        vbroadcastsd( mem(rbx),zmm30 )
        vbroadcastsd( mem(rbx,r9,1),zmm31 )
        vfmadd231pd( zmm3,zmm30,zmm6 )
        vfmadd231pd( zmm4,zmm30,zmm7 )
        vbroadcastsd( mem(rbx,r9,2),zmm30 )
        vfmadd231pd( zmm3,zmm31,zmm8 )
        vfmadd231pd( zmm4,zmm31,zmm9 )
        vbroadcastsd( mem(rbx,r13,1),zmm31 )
        vfmadd231pd( zmm3,zmm30,zmm10 )
        vfmadd231pd( zmm4,zmm30,zmm11 )
        add( r8,rbx )                                     // b += rs_b
        vfmadd231pd( zmm3,zmm31,zmm12 )
        vfmadd231pd( zmm4,zmm31,zmm13 )

        // ---------------------------------- iteration 3

        vmovupd( mem(rax),zmm3 )                           // load A
        vmovupd( 0x40(rax),zmm4 MASK_(k(2)) MASK_(z) )     // Load A with mask and zero hint
        add( r10,rax )                                     // a += cs_a
        prefetch( 0,mem(r11,r9,2) )                             // prefetch B
        vbroadcastsd( mem(rbx),zmm30 )
        vbroadcastsd( mem(rbx,r9,1),zmm31 )
        vfmadd231pd( zmm0,zmm30,zmm6 )
        vfmadd231pd( zmm1,zmm30,zmm7 )
        vbroadcastsd( mem(rbx,r9,2),zmm30 )
        vfmadd231pd( zmm0,zmm31,zmm8 )
        vfmadd231pd( zmm1,zmm31,zmm9 )
        vbroadcastsd( mem(rbx,r13,1),zmm31 )
        vfmadd231pd( zmm0,zmm30,zmm10 )
        vfmadd231pd( zmm1,zmm30,zmm11 )
        add( r8,rbx )                                     // b += rs_b
        vfmadd231pd( zmm0,zmm31,zmm12 )
        vfmadd231pd( zmm1,zmm31,zmm13 )

        // ---------------------------------- iteration 4

        vmovupd( mem(rax),zmm0 )                           // load A
        vmovupd( 0x40(rax),zmm1 MASK_(k(2)) MASK_(z) )     // Load A with mask and zero hint
        add( r10,rax )                                     // a += cs_a
        prefetch( 0,mem(r11,r13,1) )                             // prefetch B
        vbroadcastsd( mem(rbx),zmm30 )
        vbroadcastsd( mem(rbx,r9,1),zmm31 )
        vfmadd231pd( zmm3,zmm30,zmm6 )
        vfmadd231pd( zmm4,zmm30,zmm7 )
        vbroadcastsd( mem(rbx,r9,2),zmm30 )
        vfmadd231pd( zmm3,zmm31,zmm8 )
        vfmadd231pd( zmm4,zmm31,zmm9 )
        vbroadcastsd( mem(rbx,r13,1),zmm31 )
        vfmadd231pd( zmm3,zmm30,zmm10 )
        vfmadd231pd( zmm4,zmm30,zmm11 )
        add( r8,rbx )                                     // b += rs_b
        vfmadd231pd( zmm3,zmm31,zmm12 )
        vfmadd231pd( zmm4,zmm31,zmm13 )

        // ---------------------------------- iteration 5

        vmovupd( mem(rax),zmm3 )                           // load A
        vmovupd( 0x40(rax),zmm4 MASK_(k(2)) MASK_(z) )     // Load A with mask and zero hint
        add( r10,rax )                                     // a += cs_a
        vbroadcastsd( mem(rbx),zmm30 )
        vbroadcastsd( mem(rbx,r9,1),zmm31 )
        vfmadd231pd( zmm0,zmm30,zmm6 )
        vfmadd231pd( zmm1,zmm30,zmm7 )
        vbroadcastsd( mem(rbx,r9,2),zmm30 )
        vfmadd231pd( zmm0,zmm31,zmm8 )
        vfmadd231pd( zmm1,zmm31,zmm9 )
        vbroadcastsd( mem(rbx,r13,1),zmm31 )
        vfmadd231pd( zmm0,zmm30,zmm10 )
        vfmadd231pd( zmm1,zmm30,zmm11 )
        add( r8,rbx )                                     // b += rs_b
        vfmadd231pd( zmm0,zmm31,zmm12 )
        vfmadd231pd( zmm1,zmm31,zmm13 )

        // ---------------------------------- iteration 6

        vmovupd( mem(rax),zmm0 )                           // load A
        vmovupd( 0x40(rax),zmm1 MASK_(k(2)) MASK_(z) )     // Load A with mask and zero hint
        add( r10,rax )                                     // a += cs_a
        vbroadcastsd( mem(rbx),zmm30 )
        vbroadcastsd( mem(rbx,r9,1),zmm31 )
        vfmadd231pd( zmm3,zmm30,zmm6 )
        vfmadd231pd( zmm4,zmm30,zmm7 )
        vbroadcastsd( mem(rbx,r9,2),zmm30 )
        vfmadd231pd( zmm3,zmm31,zmm8 )
        vfmadd231pd( zmm4,zmm31,zmm9 )
        vbroadcastsd( mem(rbx,r13,1),zmm31 )
        vfmadd231pd( zmm3,zmm30,zmm10 )
        vfmadd231pd( zmm4,zmm30,zmm11 )
        add( r8,rbx )                                     // b += rs_b
        vfmadd231pd( zmm3,zmm31,zmm12 )
        vfmadd231pd( zmm4,zmm31,zmm13 )

        // ---------------------------------- iteration 7

        vmovupd( mem(rax),zmm3 )                           // load A
        vmovupd( 0x40(rax),zmm4 MASK_(k(2)) MASK_(z) )     // Load A with mask and zero hint
        add( r10,rax )                                     // a += cs_a
        vbroadcastsd( mem(rbx),zmm30 )
        vbroadcastsd( mem(rbx,r9,1),zmm31 )
        vfmadd231pd( zmm0,zmm30,zmm6 )
        vfmadd231pd( zmm1,zmm30,zmm7 )
        vbroadcastsd( mem(rbx,r9,2),zmm30 )
        vfmadd231pd( zmm0,zmm31,zmm8 )
        vfmadd231pd( zmm1,zmm31,zmm9 )
        vbroadcastsd( mem(rbx,r13,1),zmm31 )
        vfmadd231pd( zmm0,zmm30,zmm10 )
        vfmadd231pd( zmm1,zmm30,zmm11 )
        add( r8,rbx )                                     // b += rs_b
        vfmadd231pd( zmm0,zmm31,zmm12 )
        vfmadd231pd( zmm1,zmm31,zmm13 )

        // ---------------------------------- iteration 8

        vbroadcastsd( mem(rbx),zmm30 )
        vbroadcastsd( mem(rbx,r9,1),zmm31 )
        vfmadd231pd( zmm3,zmm30,zmm6 )
        vfmadd231pd( zmm4,zmm30,zmm7 )
        vbroadcastsd( mem(rbx,r9,2),zmm30 )
        vfmadd231pd( zmm3,zmm31,zmm8 )
        vfmadd231pd( zmm4,zmm31,zmm9 )
        vbroadcastsd( mem(rbx,r13,1),zmm31 )
        vfmadd231pd( zmm3,zmm30,zmm10 )
        vfmadd231pd( zmm4,zmm30,zmm11 )
        add( r8,rbx )                                     // b += rs_b
        vfmadd231pd( zmm3,zmm31,zmm12 )
        vfmadd231pd( zmm4,zmm31,zmm13 )
        lea(mem(r11,r8,8), r11)                            // b_next += 8*rs_b
        dec(rsi)                                           // i -= 1
    jnz(.LOOP1)                                            // iterate again if i != 0.

    label(.PREFETCHLOOP)
    add(imm(4), rsi)                                       // i += NR
    jle(.TAILITER)                                         // jump if i <= 0.

    label(.LOOP2)

        // ---------------------------------- iteration 1
        prefetchw0( mem(rdx))                              // prefetch C
        vmovupd( mem(rax),zmm0 )                           // load A
        vmovupd( 0x40(rax),zmm1 MASK_(k(2)) MASK_(z) )     // Load A with mask and zero hint
        add( r10,rax )                                     // a += cs_a
        vmovupd( mem(rax),zmm3 )                           // load A
        vmovupd( 0x40(rax),zmm4 MASK_(k(2)) MASK_(z) )     // Load A with mask and zero hint
        add( r10,rax )                                     // a += cs_a
        prefetch( 0,mem(r11) )                             // prefetch B
        vbroadcastsd( mem(rbx),zmm30 )
        vbroadcastsd( mem(rbx,r9,1),zmm31 )
        vfmadd231pd( zmm0,zmm30,zmm6 )
        vfmadd231pd( zmm1,zmm30,zmm7 )
        vbroadcastsd( mem(rbx,r9,2),zmm30 )
        vfmadd231pd( zmm0,zmm31,zmm8 )
        vfmadd231pd( zmm1,zmm31,zmm9 )
        vbroadcastsd( mem(rbx,r13,1),zmm31 )
        vfmadd231pd( zmm0,zmm30,zmm10 )
        vfmadd231pd( zmm1,zmm30,zmm11 )
        add( r8,rbx )                                     // b += rs_b
        vfmadd231pd( zmm0,zmm31,zmm12 )
        vfmadd231pd( zmm1,zmm31,zmm13 )

        // ---------------------------------- iteration 2
        prefetchw0( mem(rdx, 64))                          // prefetch C
        vmovupd( mem(rax),zmm0 )                           // load A
        vmovupd( 0x40(rax),zmm1 MASK_(k(2)) MASK_(z) )     // Load A with mask and zero hint
        add( r10,rax )                                     // a += cs_a
        prefetch( 0,mem(r11,r9,1) )                             // prefetch B
        vbroadcastsd( mem(rbx),zmm30 )
        vbroadcastsd( mem(rbx,r9,1),zmm31 )
        vfmadd231pd( zmm3,zmm30,zmm6 )
        vfmadd231pd( zmm4,zmm30,zmm7 )
        vbroadcastsd( mem(rbx,r9,2),zmm30 )
        vfmadd231pd( zmm3,zmm31,zmm8 )
        vfmadd231pd( zmm4,zmm31,zmm9 )
        vbroadcastsd( mem(rbx,r13,1),zmm31 )
        vfmadd231pd( zmm3,zmm30,zmm10 )
        vfmadd231pd( zmm4,zmm30,zmm11 )
        add( r8,rbx )                                     // b += rs_b
        vfmadd231pd( zmm3,zmm31,zmm12 )
        vfmadd231pd( zmm4,zmm31,zmm13 )

        // ---------------------------------- iteration 3
        vmovupd( mem(rax),zmm3 )                           // load A
        vmovupd( 0x40(rax),zmm4 MASK_(k(2)) MASK_(z) )     // Load A with mask and zero hint
        add( r10,rax )                                     // a += cs_a
        prefetch( 0,mem(r11,r9,2) )                             // prefetch B
        vbroadcastsd( mem(rbx),zmm30 )
        vbroadcastsd( mem(rbx,r9,1),zmm31 )
        vfmadd231pd( zmm0,zmm30,zmm6 )
        vfmadd231pd( zmm1,zmm30,zmm7 )
        vbroadcastsd( mem(rbx,r9,2),zmm30 )
        vfmadd231pd( zmm0,zmm31,zmm8 )
        vfmadd231pd( zmm1,zmm31,zmm9 )
        vbroadcastsd( mem(rbx,r13,1),zmm31 )
        vfmadd231pd( zmm0,zmm30,zmm10 )
        vfmadd231pd( zmm1,zmm30,zmm11 )
        add( r8,rbx )                                     // b += rs_b
        vfmadd231pd( zmm0,zmm31,zmm12 )
        vfmadd231pd( zmm1,zmm31,zmm13 )

        // ---------------------------------- iteration 4
        vmovupd( mem(rax),zmm0 )                           // load A
        vmovupd( 0x40(rax),zmm1 MASK_(k(2)) MASK_(z) )     // Load A with mask and zero hint
        add( r10,rax )                                     // a += cs_a
        prefetch( 0,mem(r11,r13,1) )                             // prefetch B
        vbroadcastsd( mem(rbx),zmm30 )
        vbroadcastsd( mem(rbx,r9,1),zmm31 )
        vfmadd231pd( zmm3,zmm30,zmm6 )
        vfmadd231pd( zmm4,zmm30,zmm7 )
        vbroadcastsd( mem(rbx,r9,2),zmm30 )
        vfmadd231pd( zmm3,zmm31,zmm8 )
        vfmadd231pd( zmm4,zmm31,zmm9 )
        vbroadcastsd( mem(rbx,r13,1),zmm31 )
        vfmadd231pd( zmm3,zmm30,zmm10 )
        vfmadd231pd( zmm4,zmm30,zmm11 )
        add( r8,rbx )                                     // b += rs_b
        vfmadd231pd( zmm3,zmm31,zmm12 )
        vfmadd231pd( zmm4,zmm31,zmm13 )

        // ---------------------------------- iteration 5
        vmovupd( mem(rax),zmm3 )                           // load A
        vmovupd( 0x40(rax),zmm4 MASK_(k(2)) MASK_(z) )     // Load A with mask and zero hint
        add( r10,rax )                                     // a += cs_a
        vbroadcastsd( mem(rbx),zmm30 )
        vbroadcastsd( mem(rbx,r9,1),zmm31 )
        vfmadd231pd( zmm0,zmm30,zmm6 )
        vfmadd231pd( zmm1,zmm30,zmm7 )
        vbroadcastsd( mem(rbx,r9,2),zmm30 )
        vfmadd231pd( zmm0,zmm31,zmm8 )
        vfmadd231pd( zmm1,zmm31,zmm9 )
        vbroadcastsd( mem(rbx,r13,1),zmm31 )
        vfmadd231pd( zmm0,zmm30,zmm10 )
        vfmadd231pd( zmm1,zmm30,zmm11 )
        add( r8,rbx )                                     // b += rs_b
        vfmadd231pd( zmm0,zmm31,zmm12 )
        vfmadd231pd( zmm1,zmm31,zmm13 )

        // ---------------------------------- iteration 6
        vmovupd( mem(rax),zmm0 )                           // load A
        vmovupd( 0x40(rax),zmm1 MASK_(k(2)) MASK_(z) )     // Load A with mask and zero hint
        add( r10,rax )                                     // a += cs_a
        vbroadcastsd( mem(rbx),zmm30 )
        vbroadcastsd( mem(rbx,r9,1),zmm31 )
        vfmadd231pd( zmm3,zmm30,zmm6 )
        vfmadd231pd( zmm4,zmm30,zmm7 )
        vbroadcastsd( mem(rbx,r9,2),zmm30 )
        vfmadd231pd( zmm3,zmm31,zmm8 )
        vfmadd231pd( zmm4,zmm31,zmm9 )
        vbroadcastsd( mem(rbx,r13,1),zmm31 )
        vfmadd231pd( zmm3,zmm30,zmm10 )
        vfmadd231pd( zmm4,zmm30,zmm11 )
        add( r8,rbx )                                     // b += rs_b
        vfmadd231pd( zmm3,zmm31,zmm12 )
        vfmadd231pd( zmm4,zmm31,zmm13 )

        // ---------------------------------- iteration 7
        vmovupd( mem(rax),zmm3 )                           // load A
        vmovupd( 0x40(rax),zmm4 MASK_(k(2)) MASK_(z) )     // Load A with mask and zero hint
        add( r10,rax )                                     // a += cs_a
        vbroadcastsd( mem(rbx),zmm30 )
        vbroadcastsd( mem(rbx,r9,1),zmm31 )
        vfmadd231pd( zmm0,zmm30,zmm6 )
        vfmadd231pd( zmm1,zmm30,zmm7 )
        vbroadcastsd( mem(rbx,r9,2),zmm30 )
        vfmadd231pd( zmm0,zmm31,zmm8 )
        vfmadd231pd( zmm1,zmm31,zmm9 )
        vbroadcastsd( mem(rbx,r13,1),zmm31 )
        vfmadd231pd( zmm0,zmm30,zmm10 )
        vfmadd231pd( zmm1,zmm30,zmm11 )
        add( r8,rbx )                                     // b += rs_b
        vfmadd231pd( zmm0,zmm31,zmm12 )
        vfmadd231pd( zmm1,zmm31,zmm13 )

        // ---------------------------------- iteration 8
        vbroadcastsd( mem(rbx),zmm30 )
        vbroadcastsd( mem(rbx,r9,1),zmm31 )
        vfmadd231pd( zmm3,zmm30,zmm6 )
        vfmadd231pd( zmm4,zmm30,zmm7 )
        vbroadcastsd( mem(rbx,r9,2),zmm30 )
        vfmadd231pd( zmm3,zmm31,zmm8 )
        vfmadd231pd( zmm4,zmm31,zmm9 )
        vbroadcastsd( mem(rbx,r13,1),zmm31 )
        vfmadd231pd( zmm3,zmm30,zmm10 )
        vfmadd231pd( zmm4,zmm30,zmm11 )
        add( r8,rbx )                                     // b += rs_b
        vfmadd231pd( zmm3,zmm31,zmm12 )
        vfmadd231pd( zmm4,zmm31,zmm13 )
        lea(mem(rdx, rdi, 1), rdx)                         // C += cs_c
        lea(mem(r11,r8,8), r11)                            // b_next += 8*rs_b
        sub(imm(1), rsi)                                   // i -= 1
    jnz(.LOOP2)                                            // iterate again if i != 0.
    label(.TAILITER)
    add(imm(TAIL_NITER), rsi)                              // i += TAIL_NITER
    jle(.TAIL)                                             // jump if i <= 0

    label(.LOOP3)

        // ---------------------------------- iteration 1
        vmovupd( mem(rax),zmm0 )                           // load A
        vmovupd( 0x40(rax),zmm1 MASK_(k(2)) MASK_(z) )     // Load A with mask and zero hint
        add( r10,rax )                                     // a += cs_a
        vmovupd( mem(rax),zmm3 )                           // load A
        vmovupd( 0x40(rax),zmm4 MASK_(k(2)) MASK_(z) )     // Load A with mask and zero hint
        add( r10,rax )                                     // a += cs_a
        prefetch( 0,mem(r11) )                             // prefetch B
        vbroadcastsd( mem(rbx),zmm30 )
        vbroadcastsd( mem(rbx,r9,1),zmm31 )
        vfmadd231pd( zmm0,zmm30,zmm6 )
        vfmadd231pd( zmm1,zmm30,zmm7 )
        vbroadcastsd( mem(rbx,r9,2),zmm30 )
        vfmadd231pd( zmm0,zmm31,zmm8 )
        vfmadd231pd( zmm1,zmm31,zmm9 )
        vbroadcastsd( mem(rbx,r13,1),zmm31 )
        vfmadd231pd( zmm0,zmm30,zmm10 )
        vfmadd231pd( zmm1,zmm30,zmm11 )
        add( r8,rbx )                                     // b += rs_b
        vfmadd231pd( zmm0,zmm31,zmm12 )
        vfmadd231pd( zmm1,zmm31,zmm13 )

        // ---------------------------------- iteration 2
        vmovupd( mem(rax),zmm0 )                           // load A
        vmovupd( 0x40(rax),zmm1 MASK_(k(2)) MASK_(z) )     // Load A with mask and zero hint
        add( r10,rax )                                     // a += cs_a
        prefetch( 0,mem(r11,r9,1) )                             // prefetch B
        vbroadcastsd( mem(rbx),zmm30 )
        vbroadcastsd( mem(rbx,r9,1),zmm31 )
        vfmadd231pd( zmm3,zmm30,zmm6 )
        vfmadd231pd( zmm4,zmm30,zmm7 )
        vbroadcastsd( mem(rbx,r9,2),zmm30 )
        vfmadd231pd( zmm3,zmm31,zmm8 )
        vfmadd231pd( zmm4,zmm31,zmm9 )
        vbroadcastsd( mem(rbx,r13,1),zmm31 )
        vfmadd231pd( zmm3,zmm30,zmm10 )
        vfmadd231pd( zmm4,zmm30,zmm11 )
        add( r8,rbx )                                     // b += rs_b
        vfmadd231pd( zmm3,zmm31,zmm12 )
        vfmadd231pd( zmm4,zmm31,zmm13 )

        // ---------------------------------- iteration 3
        vmovupd( mem(rax),zmm3 )                           // load A
        vmovupd( 0x40(rax),zmm4 MASK_(k(2)) MASK_(z) )     // Load A with mask and zero hint
        add( r10,rax )                                     // a += cs_a
        prefetch( 0,mem(r11,r9,2) )                             // prefetch B
        vbroadcastsd( mem(rbx),zmm30 )
        vbroadcastsd( mem(rbx,r9,1),zmm31 )
        vfmadd231pd( zmm0,zmm30,zmm6 )
        vfmadd231pd( zmm1,zmm30,zmm7 )
        vbroadcastsd( mem(rbx,r9,2),zmm30 )
        vfmadd231pd( zmm0,zmm31,zmm8 )
        vfmadd231pd( zmm1,zmm31,zmm9 )
        vbroadcastsd( mem(rbx,r13,1),zmm31 )
        vfmadd231pd( zmm0,zmm30,zmm10 )
        vfmadd231pd( zmm1,zmm30,zmm11 )
        add( r8,rbx )                                     // b += rs_b
        vfmadd231pd( zmm0,zmm31,zmm12 )
        vfmadd231pd( zmm1,zmm31,zmm13 )

        // ---------------------------------- iteration 4
        vmovupd( mem(rax),zmm0 )                           // load A
        vmovupd( 0x40(rax),zmm1 MASK_(k(2)) MASK_(z) )     // Load A with mask and zero hint
        add( r10,rax )                                     // a += cs_a
        prefetch( 0,mem(r11,r13,1) )                             // prefetch B
        vbroadcastsd( mem(rbx),zmm30 )
        vbroadcastsd( mem(rbx,r9,1),zmm31 )
        vfmadd231pd( zmm3,zmm30,zmm6 )
        vfmadd231pd( zmm4,zmm30,zmm7 )
        vbroadcastsd( mem(rbx,r9,2),zmm30 )
        vfmadd231pd( zmm3,zmm31,zmm8 )
        vfmadd231pd( zmm4,zmm31,zmm9 )
        vbroadcastsd( mem(rbx,r13,1),zmm31 )
        vfmadd231pd( zmm3,zmm30,zmm10 )
        vfmadd231pd( zmm4,zmm30,zmm11 )
        add( r8,rbx )                                     // b += rs_b
        vfmadd231pd( zmm3,zmm31,zmm12 )
        vfmadd231pd( zmm4,zmm31,zmm13 )

        // ---------------------------------- iteration 5
        vmovupd( mem(rax),zmm3 )                           // load A
        vmovupd( 0x40(rax),zmm4 MASK_(k(2)) MASK_(z) )     // Load A with mask and zero hint
        add( r10,rax )                                     // a += cs_a
        vbroadcastsd( mem(rbx),zmm30 )
        vbroadcastsd( mem(rbx,r9,1),zmm31 )
        vfmadd231pd( zmm0,zmm30,zmm6 )
        vfmadd231pd( zmm1,zmm30,zmm7 )
        vbroadcastsd( mem(rbx,r9,2),zmm30 )
        vfmadd231pd( zmm0,zmm31,zmm8 )
        vfmadd231pd( zmm1,zmm31,zmm9 )
        vbroadcastsd( mem(rbx,r13,1),zmm31 )
        vfmadd231pd( zmm0,zmm30,zmm10 )
        vfmadd231pd( zmm1,zmm30,zmm11 )
        add( r8,rbx )                                     // b += rs_b
        vfmadd231pd( zmm0,zmm31,zmm12 )
        vfmadd231pd( zmm1,zmm31,zmm13 )

        // ---------------------------------- iteration 6
        vmovupd( mem(rax),zmm0 )                           // load A
        vmovupd( 0x40(rax),zmm1 MASK_(k(2)) MASK_(z) )     // Load A with mask and zero hint
        add( r10,rax )                                     // a += cs_a
        vbroadcastsd( mem(rbx),zmm30 )
        vbroadcastsd( mem(rbx,r9,1),zmm31 )
        vfmadd231pd( zmm3,zmm30,zmm6 )
        vfmadd231pd( zmm4,zmm30,zmm7 )
        vbroadcastsd( mem(rbx,r9,2),zmm30 )
        vfmadd231pd( zmm3,zmm31,zmm8 )
        vfmadd231pd( zmm4,zmm31,zmm9 )
        vbroadcastsd( mem(rbx,r13,1),zmm31 )
        vfmadd231pd( zmm3,zmm30,zmm10 )
        vfmadd231pd( zmm4,zmm30,zmm11 )
        add( r8,rbx )                                     // b += rs_b
        vfmadd231pd( zmm3,zmm31,zmm12 )
        vfmadd231pd( zmm4,zmm31,zmm13 )

        // ---------------------------------- iteration 7
        vmovupd( mem(rax),zmm3 )                           // load A
        vmovupd( 0x40(rax),zmm4 MASK_(k(2)) MASK_(z) )     // Load A with mask and zero hint
        add( r10,rax )                                     // a += cs_a
        vbroadcastsd( mem(rbx),zmm30 )
        vbroadcastsd( mem(rbx,r9,1),zmm31 )
        vfmadd231pd( zmm0,zmm30,zmm6 )
        vfmadd231pd( zmm1,zmm30,zmm7 )
        vbroadcastsd( mem(rbx,r9,2),zmm30 )
        vfmadd231pd( zmm0,zmm31,zmm8 )
        vfmadd231pd( zmm1,zmm31,zmm9 )
        vbroadcastsd( mem(rbx,r13,1),zmm31 )
        vfmadd231pd( zmm0,zmm30,zmm10 )
        vfmadd231pd( zmm1,zmm30,zmm11 )
        add( r8,rbx )                                     // b += rs_b
        vfmadd231pd( zmm0,zmm31,zmm12 )
        vfmadd231pd( zmm1,zmm31,zmm13 )

        // ---------------------------------- iteration 8
        vbroadcastsd( mem(rbx),zmm30 )
        vbroadcastsd( mem(rbx,r9,1),zmm31 )
        vfmadd231pd( zmm3,zmm30,zmm6 )
        vfmadd231pd( zmm4,zmm30,zmm7 )
        vbroadcastsd( mem(rbx,r9,2),zmm30 )
        vfmadd231pd( zmm3,zmm31,zmm8 )
        vfmadd231pd( zmm4,zmm31,zmm9 )
        vbroadcastsd( mem(rbx,r13,1),zmm31 )
        vfmadd231pd( zmm3,zmm30,zmm10 )
        vfmadd231pd( zmm4,zmm30,zmm11 )
        add( r8,rbx )                                     // b += rs_b
        vfmadd231pd( zmm3,zmm31,zmm12 )
        vfmadd231pd( zmm4,zmm31,zmm13 )
        lea(mem(r11,r8,8), r11)                            // b_next += 8*rs_b
        dec(rsi)                                           // i -= 1
    jnz(.LOOP3)                                            // iterate again if i != 0.


    label(.TAIL)
    mov(var(k_left), rsi)                                  // i = k_left
    test(rsi, rsi)                                         // check i via logical AND
    je(.DPOSTACCUM)                                        // if i == 0, jump to post-accumulation

    label(.DLOOPKLEFT)                                     // k_left loop
        vmovupd( mem(rax),zmm0 )                           // load A
        vmovupd( 0x40(rax),zmm1 MASK_(k(2)) MASK_(z) )     // Load A with mask and zero hint
        add( r10,rax )                                     // a += cs_a
        vbroadcastsd( mem(rbx),zmm30 )
        vbroadcastsd( mem(rbx,r9,1),zmm31 )
        vfmadd231pd( zmm0,zmm30,zmm6 )
        vfmadd231pd( zmm1,zmm30,zmm7 )
        vbroadcastsd( mem(rbx,r9,2),zmm30 )
        vfmadd231pd( zmm0,zmm31,zmm8 )
        vfmadd231pd( zmm1,zmm31,zmm9 )
        vbroadcastsd( mem(rbx,r13,1),zmm31 )
        vfmadd231pd( zmm0,zmm30,zmm10 )
        vfmadd231pd( zmm1,zmm30,zmm11 )
        add( r8,rbx )                                     // b += rs_b
        vfmadd231pd( zmm0,zmm31,zmm12 )
        vfmadd231pd( zmm1,zmm31,zmm13 )
        dec(rsi)                                           // i -= 1
    jne(.DLOOPKLEFT)                                       // iterate again if i != 0.


    label(.DPOSTACCUM)
    mov(var(alpha), rdx)                                   // load address of alpha
    vbroadcastsd(mem(rdx), zmm30)                           // broadcast alpha
    mov(var(beta), rax)                                    // load address of beta
    vbroadcastsd(mem(rax), zmm31)                           // broadcast beta

    // scale by alpha
    vmulpd( zmm30,zmm6,zmm6 )
    vmulpd( zmm30,zmm7,zmm7 )
    vmulpd( zmm30,zmm8,zmm8 )
    vmulpd( zmm30,zmm9,zmm9 )
    vmulpd( zmm30,zmm10,zmm10 )
    vmulpd( zmm30,zmm11,zmm11 )
    vmulpd( zmm30,zmm12,zmm12 )
    vmulpd( zmm30,zmm13,zmm13 )


    mov(var(rs_c), rsi)                                    // load rs_c
    lea(mem(, rsi, 8), rsi)                                // rsi = rs_c * sizeof(double)
    lea(mem(rdi, rdi, 2), r13)                             // r13 = 3*cs_c
    vxorpd(ymm2, ymm2, ymm2)
    vucomisd(xmm2, xmm31)                                   // set ZF if beta == 0
    je(.DBETAZERO)                                         // if ZF == 1, jump to beta == 0 case


    cmp(imm(8), rdi)                                       // set ZF if (8*cs_c) == 8


    jz(.DROWSTORED)                                        // jump to row storage case

    label(.DCOLSTORED)
    vmovupd( mem(rcx),zmm0)
    vfmadd231pd( zmm0,zmm31,zmm6)
    vmovupd( zmm6,(rcx))
    vmovupd( 0x40(rcx),zmm1 MASK_(k(2)) MASK_(z))        // Load C using mask and zero hint
    vfmadd231pd( zmm1,zmm31,zmm7)
    vmovupd( zmm7,0x40(rcx) MASK_(k(2)))                // store to C with mask
    vmovupd( mem(rcx,rdi,1),zmm3)
    vfmadd231pd( zmm3,zmm31,zmm8)
    vmovupd( zmm8,(rcx,rdi,1))
    vmovupd( 0x40(rcx,rdi,1),zmm4 MASK_(k(2)) MASK_(z))        // Load C using mask and zero hint
    vfmadd231pd( zmm4,zmm31,zmm9)
    vmovupd( zmm9,0x40(rcx,rdi,1) MASK_(k(2)))                // store to C with mask
    vmovupd( mem(rcx,rdi,2),zmm0)
    vfmadd231pd( zmm0,zmm31,zmm10)
    vmovupd( zmm10,(rcx,rdi,2))
    vmovupd( 0x40(rcx,rdi,2),zmm1 MASK_(k(2)) MASK_(z))        // Load C using mask and zero hint
    vfmadd231pd( zmm1,zmm31,zmm11)
    vmovupd( zmm11,0x40(rcx,rdi,2) MASK_(k(2)))                // store to C with mask
    vmovupd( mem(rcx,r13,1),zmm3)
    vfmadd231pd( zmm3,zmm31,zmm12)
    vmovupd( zmm12,(rcx,r13,1))
    vmovupd( 0x40(rcx,r13,1),zmm4 MASK_(k(2)) MASK_(z))        // Load C using mask and zero hint
    vfmadd231pd( zmm4,zmm31,zmm13)
    vmovupd( zmm13,0x40(rcx,r13,1) MASK_(k(2)))                // store to C with mask

    jmp(.DDONE)                                           // jump to end.

    label(.DROWSTORED)

    // yet to be implemented
    jmp(.DDONE)                                          // jump to end.


    label(.DBETAZERO)
    cmp(imm(8), rdi)                                     // set ZF if (8*cs_c) == 8

    jz(.DROWSTORBZ)                                      // jump to row storage case
    label(.DCOLSTORBZ)
    vmovupd( zmm6,(rcx))
    vmovupd( zmm7,0x40(rcx) MASK_(k(2)))                // store to C with mask
    vmovupd( zmm8,(rcx,rdi,1))
    vmovupd( zmm9,0x40(rcx,rdi,1) MASK_(k(2)))                // store to C with mask
    vmovupd( zmm10,(rcx,rdi,2))
    vmovupd( zmm11,0x40(rcx,rdi,2) MASK_(k(2)))                // store to C with mask
    vmovupd( zmm12,(rcx,r13,1))
    vmovupd( zmm13,0x40(rcx,r13,1) MASK_(k(2)))                // store to C with mask

    jmp(.DDONE)                                          // jump to end.


    label(.DROWSTORBZ)

    // yet to be implemented
    label(.DDONE)


    vzeroupper()

    end_asm(
      : // output operands (none)
      : // input operands
        [k_iter] "m" (k_iter),
        [k_left] "m" (k_left),
        [a]      "m" (a),
        [rs_a]   "m" (rs_a),
        [cs_a]   "m" (cs_a),
        [ps_a8]  "m" (ps_a8),
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
        [mask]   "m" (mask)
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
    AOCL_DTL_TRACE_EXIT(AOCL_DTL_LEVEL_TRACE_7);
}


void bli_dgemmsup_rv_zen4_asm_8x4
(
       conj_t    conja,
       conj_t    conjb,
       dim_t     m0,
       dim_t     n0,
       dim_t     k0,
       double*    restrict alpha,
       double*    restrict a, inc_t rs_a0, inc_t cs_a0,
       double*    restrict b, inc_t rs_b0, inc_t cs_b0,
       double*    restrict beta,
       double*    restrict c, inc_t rs_c0, inc_t cs_c0,
       auxinfo_t* restrict data,
       cntx_t*    restrict cntx
     )
{
    AOCL_DTL_TRACE_ENTRY(AOCL_DTL_LEVEL_TRACE_7);
    // Typecast local copies of integers in case dim_t and inc_t are a
    // different size than is expected by load instructions.
    uint64_t rs_a   = rs_a0;
    uint64_t cs_a   = cs_a0;
    uint64_t rs_b   = rs_b0;
    uint64_t cs_b   = cs_b0;
    uint64_t rs_c   = rs_c0;
    uint64_t cs_c   = cs_c0;

    uint64_t ps_a = bli_auxinfo_ps_a( data );
    uint64_t ps_a8  = ps_a * sizeof( double );

    uint64_t k_iter = (uint64_t)k0 / 8;
    uint64_t k_left = (uint64_t)k0 % 8;

    /* 8 double precision elements can be loaded into a 512-bit register
     * So, we use an 8-bit mask to specify which elements to be loaded/stored
     * into/from the register. m_left % 8 specifies how many number of elements
     *  are to be loaded/stored into/from the last register.
     * For example, if m_left = 19, m0 & 7 becomes 3 which indicates that 3 elements
     * have to be loaded/stored into/from register, so shift 0xff(11111111) by (8-3)
     *  times to the right which makes the mask to be (00000111)
     */
    uint8_t mask = 0xff >> (0x8 - (m0 & 7)); // calculate mask based on m_left
    // For special cases where m_left = 24/16/8, all 8 elements have to be loaded/stored
    // So, mask becomes 0xff(11111111)
    if (mask == 0) mask = 0xff;

        // -------------------------------------------------------------------------
    begin_asm()

    mov(var(a), rax)                // load address of a
    mov(var(cs_a), r10)             // load cs_a
    mov(var(b), rbx)                // load address of b
    mov(var(rs_b), r8)              // load rs_b
    mov(var(cs_b), r9)              // load cs_b
    mov(var(c), rcx)                // load address of c
    mov(var(cs_c), rdi)             // load cs_c
    mov(var(mask), rdx)             // load mask
    kmovw(edx, k(2))                // move mask to k2 register
    lea(mem(, r8, 8), r8)           // rs_b *= sizeof(double)
    lea(mem(, r9, 8), r9)           // cs_b *= sizeof(double)
    lea(mem(, r10, 8), r10)         // cs_a *= sizeof(double)
    lea(mem(, rdi, 8), rdi)         // cs_c *= sizeof(double)
    lea(mem(r9, r9, 2 ), r13)       // r13 = 3*cs_b
    lea(mem(rcx, 7*8), rdx)         // C for prefetching
    lea(mem(rbx, r8, 8, 7*8), r11)  // r11 = rbx + 8*rs_b(B for prefetching)

    /* Register usage: zmm0-5 are used to load A matrix
     *                 zmm6-29 are used for accumulation
     *                 zmm30-31 are used for broadcasting B matrix
     */

    // zero out all accumulation registers
    vxorpd(zmm6, zmm6, zmm6)
    vxorpd(zmm8, zmm8, zmm8)
    vxorpd(zmm10, zmm10, zmm10)
    vxorpd(zmm12, zmm12, zmm12)

    // K is unrolled by 8 to facilitate prefetch of B
    // Assuming B to be col-stored, for each iteration of K,
    //one cacheline of B_next is prefetched where b_next = b + (unroll)*rs_b
    label(.DLOOPKITER)                                     // main loop
    mov(var(k_iter), rsi)                                  // i = k_iter
    sub(imm( 4+TAIL_NITER), rsi)                           // i -= NR + TAIL_NITER
    jle(.PREFETCHLOOP)                                     // jump if i <= 0

    label(.LOOP1)

        // ---------------------------------- iteration 1

        vmovupd( mem(rax),zmm0 MASK_(k(2)) MASK_(z) )                           // load A     // Load A with mask and zero hint
        add( r10,rax )                                     // a += cs_a
        vmovupd( mem(rax),zmm3 MASK_(k(2)) MASK_(z) )                           // load A     // Load A with mask and zero hint
        add( r10,rax )                                     // a += cs_a
        prefetch( 0,mem(r11) )                             // prefetch B
        vbroadcastsd( mem(rbx),zmm30 )
        vbroadcastsd( mem(rbx,r9,1),zmm31 )
        vfmadd231pd( zmm0,zmm30,zmm6 )
        vbroadcastsd( mem(rbx,r9,2),zmm30 )
        vfmadd231pd( zmm0,zmm31,zmm8 )
        vbroadcastsd( mem(rbx,r13,1),zmm31 )
        vfmadd231pd( zmm0,zmm30,zmm10 )
        add( r8,rbx )                                     // b += rs_b
        vfmadd231pd( zmm0,zmm31,zmm12 )

        // ---------------------------------- iteration 2

        vmovupd( mem(rax),zmm0 MASK_(k(2)) MASK_(z) )                           // load A     // Load A with mask and zero hint
        add( r10,rax )                                     // a += cs_a
        prefetch( 0,mem(r11,r9,1) )                             // prefetch B
        vbroadcastsd( mem(rbx),zmm30 )
        vbroadcastsd( mem(rbx,r9,1),zmm31 )
        vfmadd231pd( zmm3,zmm30,zmm6 )
        vbroadcastsd( mem(rbx,r9,2),zmm30 )
        vfmadd231pd( zmm3,zmm31,zmm8 )
        vbroadcastsd( mem(rbx,r13,1),zmm31 )
        vfmadd231pd( zmm3,zmm30,zmm10 )
        add( r8,rbx )                                     // b += rs_b
        vfmadd231pd( zmm3,zmm31,zmm12 )

        // ---------------------------------- iteration 3

        vmovupd( mem(rax),zmm3 MASK_(k(2)) MASK_(z) )                           // load A     // Load A with mask and zero hint
        add( r10,rax )                                     // a += cs_a
        prefetch( 0,mem(r11,r9,2) )                             // prefetch B
        vbroadcastsd( mem(rbx),zmm30 )
        vbroadcastsd( mem(rbx,r9,1),zmm31 )
        vfmadd231pd( zmm0,zmm30,zmm6 )
        vbroadcastsd( mem(rbx,r9,2),zmm30 )
        vfmadd231pd( zmm0,zmm31,zmm8 )
        vbroadcastsd( mem(rbx,r13,1),zmm31 )
        vfmadd231pd( zmm0,zmm30,zmm10 )
        add( r8,rbx )                                     // b += rs_b
        vfmadd231pd( zmm0,zmm31,zmm12 )

        // ---------------------------------- iteration 4

        vmovupd( mem(rax),zmm0 MASK_(k(2)) MASK_(z) )                           // load A     // Load A with mask and zero hint
        add( r10,rax )                                     // a += cs_a
        prefetch( 0,mem(r11,r13,1) )                             // prefetch B
        vbroadcastsd( mem(rbx),zmm30 )
        vbroadcastsd( mem(rbx,r9,1),zmm31 )
        vfmadd231pd( zmm3,zmm30,zmm6 )
        vbroadcastsd( mem(rbx,r9,2),zmm30 )
        vfmadd231pd( zmm3,zmm31,zmm8 )
        vbroadcastsd( mem(rbx,r13,1),zmm31 )
        vfmadd231pd( zmm3,zmm30,zmm10 )
        add( r8,rbx )                                     // b += rs_b
        vfmadd231pd( zmm3,zmm31,zmm12 )

        // ---------------------------------- iteration 5

        vmovupd( mem(rax),zmm3 MASK_(k(2)) MASK_(z) )                           // load A     // Load A with mask and zero hint
        add( r10,rax )                                     // a += cs_a
        vbroadcastsd( mem(rbx),zmm30 )
        vbroadcastsd( mem(rbx,r9,1),zmm31 )
        vfmadd231pd( zmm0,zmm30,zmm6 )
        vbroadcastsd( mem(rbx,r9,2),zmm30 )
        vfmadd231pd( zmm0,zmm31,zmm8 )
        vbroadcastsd( mem(rbx,r13,1),zmm31 )
        vfmadd231pd( zmm0,zmm30,zmm10 )
        add( r8,rbx )                                     // b += rs_b
        vfmadd231pd( zmm0,zmm31,zmm12 )

        // ---------------------------------- iteration 6

        vmovupd( mem(rax),zmm0 MASK_(k(2)) MASK_(z) )                           // load A     // Load A with mask and zero hint
        add( r10,rax )                                     // a += cs_a
        vbroadcastsd( mem(rbx),zmm30 )
        vbroadcastsd( mem(rbx,r9,1),zmm31 )
        vfmadd231pd( zmm3,zmm30,zmm6 )
        vbroadcastsd( mem(rbx,r9,2),zmm30 )
        vfmadd231pd( zmm3,zmm31,zmm8 )
        vbroadcastsd( mem(rbx,r13,1),zmm31 )
        vfmadd231pd( zmm3,zmm30,zmm10 )
        add( r8,rbx )                                     // b += rs_b
        vfmadd231pd( zmm3,zmm31,zmm12 )

        // ---------------------------------- iteration 7

        vmovupd( mem(rax),zmm3 MASK_(k(2)) MASK_(z) )                           // load A     // Load A with mask and zero hint
        add( r10,rax )                                     // a += cs_a
        vbroadcastsd( mem(rbx),zmm30 )
        vbroadcastsd( mem(rbx,r9,1),zmm31 )
        vfmadd231pd( zmm0,zmm30,zmm6 )
        vbroadcastsd( mem(rbx,r9,2),zmm30 )
        vfmadd231pd( zmm0,zmm31,zmm8 )
        vbroadcastsd( mem(rbx,r13,1),zmm31 )
        vfmadd231pd( zmm0,zmm30,zmm10 )
        add( r8,rbx )                                     // b += rs_b
        vfmadd231pd( zmm0,zmm31,zmm12 )

        // ---------------------------------- iteration 8

        vbroadcastsd( mem(rbx),zmm30 )
        vbroadcastsd( mem(rbx,r9,1),zmm31 )
        vfmadd231pd( zmm3,zmm30,zmm6 )
        vbroadcastsd( mem(rbx,r9,2),zmm30 )
        vfmadd231pd( zmm3,zmm31,zmm8 )
        vbroadcastsd( mem(rbx,r13,1),zmm31 )
        vfmadd231pd( zmm3,zmm30,zmm10 )
        add( r8,rbx )                                     // b += rs_b
        vfmadd231pd( zmm3,zmm31,zmm12 )
        lea(mem(r11,r8,8), r11)                            // b_next += 8*rs_b
        dec(rsi)                                           // i -= 1
    jnz(.LOOP1)                                            // iterate again if i != 0.

    label(.PREFETCHLOOP)
    add(imm(4), rsi)                                       // i += NR
    jle(.TAILITER)                                         // jump if i <= 0.

    label(.LOOP2)

        // ---------------------------------- iteration 1
        prefetchw0( mem(rdx))                              // prefetch C
        vmovupd( mem(rax),zmm0 MASK_(k(2)) MASK_(z) )                           // load A     // Load A with mask and zero hint
        add( r10,rax )                                     // a += cs_a
        vmovupd( mem(rax),zmm3 MASK_(k(2)) MASK_(z) )                           // load A     // Load A with mask and zero hint
        add( r10,rax )                                     // a += cs_a
        prefetch( 0,mem(r11) )                             // prefetch B
        vbroadcastsd( mem(rbx),zmm30 )
        vbroadcastsd( mem(rbx,r9,1),zmm31 )
        vfmadd231pd( zmm0,zmm30,zmm6 )
        vbroadcastsd( mem(rbx,r9,2),zmm30 )
        vfmadd231pd( zmm0,zmm31,zmm8 )
        vbroadcastsd( mem(rbx,r13,1),zmm31 )
        vfmadd231pd( zmm0,zmm30,zmm10 )
        add( r8,rbx )                                     // b += rs_b
        vfmadd231pd( zmm0,zmm31,zmm12 )

        // ---------------------------------- iteration 2
        vmovupd( mem(rax),zmm0 MASK_(k(2)) MASK_(z) )                           // load A     // Load A with mask and zero hint
        add( r10,rax )                                     // a += cs_a
        prefetch( 0,mem(r11,r9,1) )                             // prefetch B
        vbroadcastsd( mem(rbx),zmm30 )
        vbroadcastsd( mem(rbx,r9,1),zmm31 )
        vfmadd231pd( zmm3,zmm30,zmm6 )
        vbroadcastsd( mem(rbx,r9,2),zmm30 )
        vfmadd231pd( zmm3,zmm31,zmm8 )
        vbroadcastsd( mem(rbx,r13,1),zmm31 )
        vfmadd231pd( zmm3,zmm30,zmm10 )
        add( r8,rbx )                                     // b += rs_b
        vfmadd231pd( zmm3,zmm31,zmm12 )

        // ---------------------------------- iteration 3
        vmovupd( mem(rax),zmm3 MASK_(k(2)) MASK_(z) )                           // load A     // Load A with mask and zero hint
        add( r10,rax )                                     // a += cs_a
        prefetch( 0,mem(r11,r9,2) )                             // prefetch B
        vbroadcastsd( mem(rbx),zmm30 )
        vbroadcastsd( mem(rbx,r9,1),zmm31 )
        vfmadd231pd( zmm0,zmm30,zmm6 )
        vbroadcastsd( mem(rbx,r9,2),zmm30 )
        vfmadd231pd( zmm0,zmm31,zmm8 )
        vbroadcastsd( mem(rbx,r13,1),zmm31 )
        vfmadd231pd( zmm0,zmm30,zmm10 )
        add( r8,rbx )                                     // b += rs_b
        vfmadd231pd( zmm0,zmm31,zmm12 )

        // ---------------------------------- iteration 4
        vmovupd( mem(rax),zmm0 MASK_(k(2)) MASK_(z) )                           // load A     // Load A with mask and zero hint
        add( r10,rax )                                     // a += cs_a
        prefetch( 0,mem(r11,r13,1) )                             // prefetch B
        vbroadcastsd( mem(rbx),zmm30 )
        vbroadcastsd( mem(rbx,r9,1),zmm31 )
        vfmadd231pd( zmm3,zmm30,zmm6 )
        vbroadcastsd( mem(rbx,r9,2),zmm30 )
        vfmadd231pd( zmm3,zmm31,zmm8 )
        vbroadcastsd( mem(rbx,r13,1),zmm31 )
        vfmadd231pd( zmm3,zmm30,zmm10 )
        add( r8,rbx )                                     // b += rs_b
        vfmadd231pd( zmm3,zmm31,zmm12 )

        // ---------------------------------- iteration 5
        vmovupd( mem(rax),zmm3 MASK_(k(2)) MASK_(z) )                           // load A     // Load A with mask and zero hint
        add( r10,rax )                                     // a += cs_a
        vbroadcastsd( mem(rbx),zmm30 )
        vbroadcastsd( mem(rbx,r9,1),zmm31 )
        vfmadd231pd( zmm0,zmm30,zmm6 )
        vbroadcastsd( mem(rbx,r9,2),zmm30 )
        vfmadd231pd( zmm0,zmm31,zmm8 )
        vbroadcastsd( mem(rbx,r13,1),zmm31 )
        vfmadd231pd( zmm0,zmm30,zmm10 )
        add( r8,rbx )                                     // b += rs_b
        vfmadd231pd( zmm0,zmm31,zmm12 )

        // ---------------------------------- iteration 6
        vmovupd( mem(rax),zmm0 MASK_(k(2)) MASK_(z) )                           // load A     // Load A with mask and zero hint
        add( r10,rax )                                     // a += cs_a
        vbroadcastsd( mem(rbx),zmm30 )
        vbroadcastsd( mem(rbx,r9,1),zmm31 )
        vfmadd231pd( zmm3,zmm30,zmm6 )
        vbroadcastsd( mem(rbx,r9,2),zmm30 )
        vfmadd231pd( zmm3,zmm31,zmm8 )
        vbroadcastsd( mem(rbx,r13,1),zmm31 )
        vfmadd231pd( zmm3,zmm30,zmm10 )
        add( r8,rbx )                                     // b += rs_b
        vfmadd231pd( zmm3,zmm31,zmm12 )

        // ---------------------------------- iteration 7
        vmovupd( mem(rax),zmm3 MASK_(k(2)) MASK_(z) )                           // load A     // Load A with mask and zero hint
        add( r10,rax )                                     // a += cs_a
        vbroadcastsd( mem(rbx),zmm30 )
        vbroadcastsd( mem(rbx,r9,1),zmm31 )
        vfmadd231pd( zmm0,zmm30,zmm6 )
        vbroadcastsd( mem(rbx,r9,2),zmm30 )
        vfmadd231pd( zmm0,zmm31,zmm8 )
        vbroadcastsd( mem(rbx,r13,1),zmm31 )
        vfmadd231pd( zmm0,zmm30,zmm10 )
        add( r8,rbx )                                     // b += rs_b
        vfmadd231pd( zmm0,zmm31,zmm12 )

        // ---------------------------------- iteration 8
        vbroadcastsd( mem(rbx),zmm30 )
        vbroadcastsd( mem(rbx,r9,1),zmm31 )
        vfmadd231pd( zmm3,zmm30,zmm6 )
        vbroadcastsd( mem(rbx,r9,2),zmm30 )
        vfmadd231pd( zmm3,zmm31,zmm8 )
        vbroadcastsd( mem(rbx,r13,1),zmm31 )
        vfmadd231pd( zmm3,zmm30,zmm10 )
        add( r8,rbx )                                     // b += rs_b
        vfmadd231pd( zmm3,zmm31,zmm12 )
        lea(mem(rdx, rdi, 1), rdx)                         // C += cs_c
        lea(mem(r11,r8,8), r11)                            // b_next += 8*rs_b
        sub(imm(1), rsi)                                   // i -= 1
    jnz(.LOOP2)                                            // iterate again if i != 0.
    label(.TAILITER)
    add(imm(TAIL_NITER), rsi)                              // i += TAIL_NITER
    jle(.TAIL)                                             // jump if i <= 0

    label(.LOOP3)

        // ---------------------------------- iteration 1
        vmovupd( mem(rax),zmm0 MASK_(k(2)) MASK_(z) )                           // load A     // Load A with mask and zero hint
        add( r10,rax )                                     // a += cs_a
        vmovupd( mem(rax),zmm3 MASK_(k(2)) MASK_(z) )                           // load A     // Load A with mask and zero hint
        add( r10,rax )                                     // a += cs_a
        prefetch( 0,mem(r11) )                             // prefetch B
        vbroadcastsd( mem(rbx),zmm30 )
        vbroadcastsd( mem(rbx,r9,1),zmm31 )
        vfmadd231pd( zmm0,zmm30,zmm6 )
        vbroadcastsd( mem(rbx,r9,2),zmm30 )
        vfmadd231pd( zmm0,zmm31,zmm8 )
        vbroadcastsd( mem(rbx,r13,1),zmm31 )
        vfmadd231pd( zmm0,zmm30,zmm10 )
        add( r8,rbx )                                     // b += rs_b
        vfmadd231pd( zmm0,zmm31,zmm12 )

        // ---------------------------------- iteration 2
        vmovupd( mem(rax),zmm0 MASK_(k(2)) MASK_(z) )                           // load A     // Load A with mask and zero hint
        add( r10,rax )                                     // a += cs_a
        prefetch( 0,mem(r11,r9,1) )                             // prefetch B
        vbroadcastsd( mem(rbx),zmm30 )
        vbroadcastsd( mem(rbx,r9,1),zmm31 )
        vfmadd231pd( zmm3,zmm30,zmm6 )
        vbroadcastsd( mem(rbx,r9,2),zmm30 )
        vfmadd231pd( zmm3,zmm31,zmm8 )
        vbroadcastsd( mem(rbx,r13,1),zmm31 )
        vfmadd231pd( zmm3,zmm30,zmm10 )
        add( r8,rbx )                                     // b += rs_b
        vfmadd231pd( zmm3,zmm31,zmm12 )

        // ---------------------------------- iteration 3
        vmovupd( mem(rax),zmm3 MASK_(k(2)) MASK_(z) )                           // load A     // Load A with mask and zero hint
        add( r10,rax )                                     // a += cs_a
        prefetch( 0,mem(r11,r9,2) )                             // prefetch B
        vbroadcastsd( mem(rbx),zmm30 )
        vbroadcastsd( mem(rbx,r9,1),zmm31 )
        vfmadd231pd( zmm0,zmm30,zmm6 )
        vbroadcastsd( mem(rbx,r9,2),zmm30 )
        vfmadd231pd( zmm0,zmm31,zmm8 )
        vbroadcastsd( mem(rbx,r13,1),zmm31 )
        vfmadd231pd( zmm0,zmm30,zmm10 )
        add( r8,rbx )                                     // b += rs_b
        vfmadd231pd( zmm0,zmm31,zmm12 )

        // ---------------------------------- iteration 4
        vmovupd( mem(rax),zmm0 MASK_(k(2)) MASK_(z) )                           // load A     // Load A with mask and zero hint
        add( r10,rax )                                     // a += cs_a
        prefetch( 0,mem(r11,r13,1) )                             // prefetch B
        vbroadcastsd( mem(rbx),zmm30 )
        vbroadcastsd( mem(rbx,r9,1),zmm31 )
        vfmadd231pd( zmm3,zmm30,zmm6 )
        vbroadcastsd( mem(rbx,r9,2),zmm30 )
        vfmadd231pd( zmm3,zmm31,zmm8 )
        vbroadcastsd( mem(rbx,r13,1),zmm31 )
        vfmadd231pd( zmm3,zmm30,zmm10 )
        add( r8,rbx )                                     // b += rs_b
        vfmadd231pd( zmm3,zmm31,zmm12 )

        // ---------------------------------- iteration 5
        vmovupd( mem(rax),zmm3 MASK_(k(2)) MASK_(z) )                           // load A     // Load A with mask and zero hint
        add( r10,rax )                                     // a += cs_a
        vbroadcastsd( mem(rbx),zmm30 )
        vbroadcastsd( mem(rbx,r9,1),zmm31 )
        vfmadd231pd( zmm0,zmm30,zmm6 )
        vbroadcastsd( mem(rbx,r9,2),zmm30 )
        vfmadd231pd( zmm0,zmm31,zmm8 )
        vbroadcastsd( mem(rbx,r13,1),zmm31 )
        vfmadd231pd( zmm0,zmm30,zmm10 )
        add( r8,rbx )                                     // b += rs_b
        vfmadd231pd( zmm0,zmm31,zmm12 )

        // ---------------------------------- iteration 6
        vmovupd( mem(rax),zmm0 MASK_(k(2)) MASK_(z) )                           // load A     // Load A with mask and zero hint
        add( r10,rax )                                     // a += cs_a
        vbroadcastsd( mem(rbx),zmm30 )
        vbroadcastsd( mem(rbx,r9,1),zmm31 )
        vfmadd231pd( zmm3,zmm30,zmm6 )
        vbroadcastsd( mem(rbx,r9,2),zmm30 )
        vfmadd231pd( zmm3,zmm31,zmm8 )
        vbroadcastsd( mem(rbx,r13,1),zmm31 )
        vfmadd231pd( zmm3,zmm30,zmm10 )
        add( r8,rbx )                                     // b += rs_b
        vfmadd231pd( zmm3,zmm31,zmm12 )

        // ---------------------------------- iteration 7
        vmovupd( mem(rax),zmm3 MASK_(k(2)) MASK_(z) )                           // load A     // Load A with mask and zero hint
        add( r10,rax )                                     // a += cs_a
        vbroadcastsd( mem(rbx),zmm30 )
        vbroadcastsd( mem(rbx,r9,1),zmm31 )
        vfmadd231pd( zmm0,zmm30,zmm6 )
        vbroadcastsd( mem(rbx,r9,2),zmm30 )
        vfmadd231pd( zmm0,zmm31,zmm8 )
        vbroadcastsd( mem(rbx,r13,1),zmm31 )
        vfmadd231pd( zmm0,zmm30,zmm10 )
        add( r8,rbx )                                     // b += rs_b
        vfmadd231pd( zmm0,zmm31,zmm12 )

        // ---------------------------------- iteration 8
        vbroadcastsd( mem(rbx),zmm30 )
        vbroadcastsd( mem(rbx,r9,1),zmm31 )
        vfmadd231pd( zmm3,zmm30,zmm6 )
        vbroadcastsd( mem(rbx,r9,2),zmm30 )
        vfmadd231pd( zmm3,zmm31,zmm8 )
        vbroadcastsd( mem(rbx,r13,1),zmm31 )
        vfmadd231pd( zmm3,zmm30,zmm10 )
        add( r8,rbx )                                     // b += rs_b
        vfmadd231pd( zmm3,zmm31,zmm12 )
        lea(mem(r11,r8,8), r11)                            // b_next += 8*rs_b
        dec(rsi)                                           // i -= 1
    jnz(.LOOP3)                                            // iterate again if i != 0.


    label(.TAIL)
    mov(var(k_left), rsi)                                  // i = k_left
    test(rsi, rsi)                                         // check i via logical AND
    je(.DPOSTACCUM)                                        // if i == 0, jump to post-accumulation

    label(.DLOOPKLEFT)                                     // k_left loop
        vmovupd( mem(rax),zmm0 MASK_(k(2)) MASK_(z) )                           // load A     // Load A with mask and zero hint
        add( r10,rax )                                     // a += cs_a
        vbroadcastsd( mem(rbx),zmm30 )
        vbroadcastsd( mem(rbx,r9,1),zmm31 )
        vfmadd231pd( zmm0,zmm30,zmm6 )
        vbroadcastsd( mem(rbx,r9,2),zmm30 )
        vfmadd231pd( zmm0,zmm31,zmm8 )
        vbroadcastsd( mem(rbx,r13,1),zmm31 )
        vfmadd231pd( zmm0,zmm30,zmm10 )
        add( r8,rbx )                                     // b += rs_b
        vfmadd231pd( zmm0,zmm31,zmm12 )
        dec(rsi)                                           // i -= 1
    jne(.DLOOPKLEFT)                                       // iterate again if i != 0.


    label(.DPOSTACCUM)
    mov(var(alpha), rdx)                                   // load address of alpha
    vbroadcastsd(mem(rdx), zmm30)                           // broadcast alpha
    mov(var(beta), rax)                                    // load address of beta
    vbroadcastsd(mem(rax), zmm31)                           // broadcast beta

    // scale by alpha
    vmulpd( zmm30,zmm6,zmm6 )
    vmulpd( zmm30,zmm8,zmm8 )
    vmulpd( zmm30,zmm10,zmm10 )
    vmulpd( zmm30,zmm12,zmm12 )


    mov(var(rs_c), rsi)                                    // load rs_c
    lea(mem(, rsi, 8), rsi)                                // rsi = rs_c * sizeof(double)
    lea(mem(rdi, rdi, 2), r13)                             // r13 = 3*cs_c
    vxorpd(ymm2, ymm2, ymm2)
    vucomisd(xmm2, xmm31)                                   // set ZF if beta == 0
    je(.DBETAZERO)                                         // if ZF == 1, jump to beta == 0 case


    cmp(imm(8), rdi)                                       // set ZF if (8*cs_c) == 8


    jz(.DROWSTORED)                                        // jump to row storage case

    label(.DCOLSTORED)
    vmovupd( mem(rcx),zmm0 MASK_(k(2)) MASK_(z))        // Load C using mask and zero hint
    vfmadd231pd( zmm0,zmm31,zmm6)
    vmovupd( zmm6,(rcx) MASK_(k(2)))                // store to C with mask
    vmovupd( mem(rcx,rdi,1),zmm3 MASK_(k(2)) MASK_(z))        // Load C using mask and zero hint
    vfmadd231pd( zmm3,zmm31,zmm8)
    vmovupd( zmm8,(rcx,rdi,1) MASK_(k(2)))                // store to C with mask
    vmovupd( mem(rcx,rdi,2),zmm0 MASK_(k(2)) MASK_(z))        // Load C using mask and zero hint
    vfmadd231pd( zmm0,zmm31,zmm10)
    vmovupd( zmm10,(rcx,rdi,2) MASK_(k(2)))                // store to C with mask
    vmovupd( mem(rcx,r13,1),zmm3 MASK_(k(2)) MASK_(z))        // Load C using mask and zero hint
    vfmadd231pd( zmm3,zmm31,zmm12)
    vmovupd( zmm12,(rcx,r13,1) MASK_(k(2)))                // store to C with mask

    jmp(.DDONE)                                           // jump to end.

    label(.DROWSTORED)

    // yet to be implemented
    jmp(.DDONE)                                          // jump to end.


    label(.DBETAZERO)
    cmp(imm(8), rdi)                                     // set ZF if (8*cs_c) == 8

    jz(.DROWSTORBZ)                                      // jump to row storage case
    label(.DCOLSTORBZ)
    vmovupd( zmm6,(rcx) MASK_(k(2)))                // store to C with mask
    vmovupd( zmm8,(rcx,rdi,1) MASK_(k(2)))                // store to C with mask
    vmovupd( zmm10,(rcx,rdi,2) MASK_(k(2)))                // store to C with mask
    vmovupd( zmm12,(rcx,r13,1) MASK_(k(2)))                // store to C with mask

    jmp(.DDONE)                                          // jump to end.


    label(.DROWSTORBZ)

    // yet to be implemented
    label(.DDONE)


    vzeroupper()

    end_asm(
      : // output operands (none)
      : // input operands
        [k_iter] "m" (k_iter),
        [k_left] "m" (k_left),
        [a]      "m" (a),
        [rs_a]   "m" (rs_a),
        [cs_a]   "m" (cs_a),
        [ps_a8]  "m" (ps_a8),
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
        [mask]   "m" (mask)
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
    AOCL_DTL_TRACE_EXIT(AOCL_DTL_LEVEL_TRACE_7);
}
