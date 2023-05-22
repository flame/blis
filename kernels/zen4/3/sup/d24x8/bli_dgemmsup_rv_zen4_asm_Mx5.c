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

/**
 * Shuffle 2 double-precision elements selected by imm8 from S1 and S2,
 * and store the results in D1.
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
    vunpcklpd( zmm(S1),  zmm(S2),  zmm(D1)) \
    vunpckhpd( zmm(S1),  zmm(S2),  zmm(D2)) \
    vunpcklpd( zmm(S3),  zmm(S4),  zmm(D3)) \
    vunpckhpd( zmm(S3),  zmm(S4),  zmm(D4))

/**
 * mask register is set, stores the fma result back to C
*/
#define UPDATE_MASKED_C_8_BZ \
    vmovupd( zmm0, mem(rcx) MASK_(k(3))) \
\
    vmovupd( zmm4, mem(rcx, rsi, 1) MASK_(k(3))) \
\
    vmovupd( zmm2, mem(rcx, rsi, 2) MASK_(k(3)) ) \
\
    vmovupd( zmm6, mem(rcx, r12, 1) MASK_(k(3)) ) \
\
    vmovupd( zmm1, mem(rcx, rsi, 4) MASK_(k(3))) \
\
    vmovupd( zmm5, mem(rcx, r13, 1) MASK_(k(3))) \
\
    vmovupd( zmm3, mem(rcx, r12, 2) MASK_(k(3))) \
\
    vmovupd( zmm8, mem(rcx, rdx, 1) MASK_(k(3))) \
    add(r14, rcx)

/**
 * mask register is set, stores the fma result back to C
*/
#define UPDATE_MASKED_C_7_BZ \
    vmovupd( zmm0, mem(rcx) MASK_(k(3))) \
\
    vmovupd( zmm4, mem(rcx, rsi, 1) MASK_(k(3))) \
\
    vmovupd( zmm2, mem(rcx, rsi, 2) MASK_(k(3)) ) \
\
    vmovupd( zmm6, mem(rcx, r12, 1) MASK_(k(3)) ) \
\
    vmovupd( zmm1, mem(rcx, rsi, 4) MASK_(k(3))) \
\
    vmovupd( zmm5, mem(rcx, r13, 1) MASK_(k(3))) \
\
    vmovupd( zmm3, mem(rcx, r12, 2) MASK_(k(3)))

/**
 * mask register is set, stores the fma result back to C
*/
#define UPDATE_MASKED_C_6_BZ \
    vmovupd( zmm0, mem(rcx) MASK_(k(3))) \
\
    vmovupd( zmm4, mem(rcx, rsi, 1) MASK_(k(3))) \
\
    vmovupd( zmm2, mem(rcx, rsi, 2) MASK_(k(3)) ) \
\
    vmovupd( zmm6, mem(rcx, r12, 1) MASK_(k(3)) ) \
\
    vmovupd( zmm1, mem(rcx, rsi, 4) MASK_(k(3))) \
\
    vmovupd( zmm5, mem(rcx, r13, 1) MASK_(k(3)))

/**
 * mask register is set, stores the fma result back to C
*/
#define UPDATE_MASKED_C_5_BZ \
    vmovupd( zmm0, mem(rcx) MASK_(k(3))) \
\
    vmovupd( zmm4, mem(rcx, rsi, 1) MASK_(k(3))) \
\
    vmovupd( zmm2, mem(rcx, rsi, 2) MASK_(k(3)) ) \
\
    vmovupd( zmm6, mem(rcx, r12, 1) MASK_(k(3)) ) \
\
    vmovupd( zmm1, mem(rcx, rsi, 4) MASK_(k(3)))

/**
 * mask register is set, stores the fma result back to C
*/
#define UPDATE_MASKED_C_4_BZ \
    vmovupd( zmm0, mem(rcx) MASK_(k(3))) \
\
    vmovupd( zmm4, mem(rcx, rsi, 1) MASK_(k(3))) \
\
    vmovupd( zmm2, mem(rcx, rsi, 2) MASK_(k(3)) ) \
\
    vmovupd( zmm6, mem(rcx, r12, 1) MASK_(k(3)) )

/**
 * mask register is set, stores the fma result back to C
*/
#define UPDATE_MASKED_C_3_BZ \
    vmovupd( zmm0, mem(rcx) MASK_(k(3))) \
\
    vmovupd( zmm4, mem(rcx, rsi, 1) MASK_(k(3))) \
\
    vmovupd( zmm2, mem(rcx, rsi, 2) MASK_(k(3)) )

/**
 * mask register is set, stores the fma result back to C
*/
#define UPDATE_MASKED_C_2_BZ \
    vmovupd( zmm0, mem(rcx) MASK_(k(3))) \
\
    vmovupd( zmm4, mem(rcx, rsi, 1) MASK_(k(3)))

/**
 * mask register is set, stores the fma result back to C
*/
#define UPDATE_MASKED_C_1_BZ \
\
    vmovupd( zmm0, mem(rcx) MASK_(k(3)))

/**
 * Loads elements from C row only if correspondnig bits in
 * mask register is set, Scales it with Beta and adds FMA result to it.
 * Stores back the C row.
*/
#define UPDATE_MASKED_C_8 \
\
    vmovupd( mem(rcx), zmm30 MASK_KZ(3) ) \
    vfmadd231pd( zmm31,zmm30,zmm0 ) \
\
    vmovupd( mem(rcx, rsi, 1, 0), zmm10 MASK_KZ(3) ) \
    vfmadd231pd( zmm31,zmm10,zmm4 ) \
\
    vmovupd( mem(rcx, rsi, 2, 0), zmm12 MASK_KZ(3) ) \
    vfmadd231pd( zmm31,zmm12,zmm2 ) \
\
    vmovupd( mem(rcx, r12, 1, 0), zmm16 MASK_KZ(3) ) \
    vfmadd231pd( zmm31,zmm16,zmm6 ) \
\
    vmovupd( mem(rcx, rsi, 4, 0), zmm14 MASK_KZ(3) ) \
    vfmadd231pd( zmm31,zmm14,zmm1 ) \
\
    vmovupd( mem(rcx, r13, 1, 0), zmm18 MASK_KZ(3) ) \
    vfmadd231pd( zmm31,zmm18,zmm5 ) \
\
    vmovupd( mem(rcx, r12, 2, 0), zmm10 MASK_KZ(3) ) \
    vfmadd231pd( zmm31,zmm10,zmm3 ) \
\
    vmovupd( mem(rcx, rdx, 1, 0), zmm12 MASK_KZ(3) ) \
    vfmadd231pd( zmm31,zmm12,zmm8 ) \
\
    vmovupd( zmm0, (rcx) MASK_(k(3)))            /*Stores back to C*/\
    vmovupd( zmm4, (rcx, rsi, 1) MASK_(k(3)))\
    vmovupd( zmm2, (rcx, rsi, 2) MASK_(k(3)))\
    vmovupd( zmm6, (rcx, r12, 1) MASK_(k(3)))\
    vmovupd( zmm1, (rcx, rsi, 4) MASK_(k(3)))\
    vmovupd( zmm5, (rcx, r13, 1) MASK_(k(3)))\
    vmovupd( zmm3, (rcx, r12, 2) MASK_(k(3)))\
    vmovupd( zmm8, (rcx, rdx, 1) MASK_(k(3)))\
    add(r14, rcx)

/**
 * Loads elements from C row only if correspondnig bits in
 * mask register is set, Scales it with Beta and adds FMA result to it.
 * Stores back the C row.
*/
#define UPDATE_MASKED_C_7 \
\
    vmovupd( mem(rcx), zmm30 MASK_KZ(3) ) \
    vfmadd231pd( zmm31,zmm30,zmm0 ) \
\
    vmovupd( mem(rcx, rsi, 1, 0), zmm10 MASK_KZ(3) ) \
    vfmadd231pd( zmm31,zmm10,zmm4 ) \
\
    vmovupd( mem(rcx, rsi, 2, 0), zmm12 MASK_KZ(3) ) \
    vfmadd231pd( zmm31,zmm12,zmm2 ) \
\
    vmovupd( mem(rcx, r12, 1, 0), zmm16 MASK_KZ(3) ) \
    vfmadd231pd( zmm31,zmm16,zmm6 ) \
\
    vmovupd( mem(rcx, rsi, 4, 0), zmm14 MASK_KZ(3) ) \
    vfmadd231pd( zmm31,zmm14,zmm1 ) \
\
    vmovupd( mem(rcx, r13, 1, 0), zmm18 MASK_KZ(3) ) \
    vfmadd231pd( zmm31,zmm18,zmm5 ) \
\
    vmovupd( mem(rcx, r12, 2, 0), zmm10 MASK_KZ(3) ) \
    vfmadd231pd( zmm31,zmm10,zmm3 ) \
\
    vmovupd( zmm0, (rcx) MASK_(k(3)))            /*Stores back to C*/\
    vmovupd( zmm4, (rcx, rsi, 1) MASK_(k(3)))\
    vmovupd( zmm2, (rcx, rsi, 2) MASK_(k(3)))\
    vmovupd( zmm6, (rcx, r12, 1) MASK_(k(3)))\
    vmovupd( zmm1, (rcx, rsi, 4) MASK_(k(3)))\
    vmovupd( zmm5, (rcx, r13, 1) MASK_(k(3)))\
    vmovupd( zmm3, (rcx, r12, 2) MASK_(k(3)))

/**
 * Loads elements from C row only if correspondnig bits in
 * mask register is set, Scales it with Beta and adds FMA result to it.
 * Stores back the C row.
*/
#define UPDATE_MASKED_C_6 \
\
    vmovupd( mem(rcx), zmm30 MASK_KZ(3) ) \
    vfmadd231pd( zmm31,zmm30,zmm0 ) \
\
    vmovupd( mem(rcx, rsi, 1, 0), zmm10 MASK_KZ(3) ) \
    vfmadd231pd( zmm31,zmm10,zmm4 ) \
\
    vmovupd( mem(rcx, rsi, 2, 0), zmm12 MASK_KZ(3) ) \
    vfmadd231pd( zmm31,zmm12,zmm2 ) \
\
    vmovupd( mem(rcx, r12, 1, 0), zmm16 MASK_KZ(3) ) \
    vfmadd231pd( zmm31,zmm16,zmm6 ) \
\
    vmovupd( mem(rcx, rsi, 4, 0), zmm14 MASK_KZ(3) ) \
    vfmadd231pd( zmm31,zmm14,zmm1 ) \
\
    vmovupd( mem(rcx, r13, 1, 0), zmm18 MASK_KZ(3) ) \
    vfmadd231pd( zmm31,zmm18,zmm5 ) \
\
    vmovupd( zmm0, (rcx) MASK_(k(3)))            /*Stores back to C*/\
    vmovupd( zmm4, (rcx, rsi, 1) MASK_(k(3)))\
    vmovupd( zmm2, (rcx, rsi, 2) MASK_(k(3)))\
    vmovupd( zmm6, (rcx, r12, 1) MASK_(k(3)))\
    vmovupd( zmm1, (rcx, rsi, 4) MASK_(k(3)))\
    vmovupd( zmm5, (rcx, r13, 1) MASK_(k(3)))

/**
 * Loads elements from C row only if correspondnig bits in
 * mask register is set, Scales it with Beta and adds FMA result to it.
 * Stores back the C row.
*/
#define UPDATE_MASKED_C_5 \
\
    vmovupd( mem(rcx), zmm30 MASK_KZ(3) ) \
    vfmadd231pd( zmm31,zmm30,zmm0 ) \
\
    vmovupd( mem(rcx, rsi, 1, 0), zmm10 MASK_KZ(3) ) \
    vfmadd231pd( zmm31,zmm10,zmm4 ) \
\
    vmovupd( mem(rcx, rsi, 2, 0), zmm12 MASK_KZ(3) ) \
    vfmadd231pd( zmm31,zmm12,zmm2 ) \
\
    vmovupd( mem(rcx, r12, 1, 0), zmm16 MASK_KZ(3) ) \
    vfmadd231pd( zmm31,zmm16,zmm6 ) \
\
    vmovupd( mem(rcx, rsi, 4, 0), zmm14 MASK_KZ(3) ) \
    vfmadd231pd( zmm31,zmm14,zmm1 ) \
\
    vmovupd( zmm0, (rcx) MASK_(k(3)))            /*Stores back to C*/\
    vmovupd( zmm4, (rcx, rsi, 1) MASK_(k(3)))\
    vmovupd( zmm2, (rcx, rsi, 2) MASK_(k(3)))\
    vmovupd( zmm6, (rcx, r12, 1) MASK_(k(3)))\
    vmovupd( zmm1, (rcx, rsi, 4) MASK_(k(3)))

/**
 * Loads elements from C row only if correspondnig bits in
 * mask register is set, Scales it with Beta and adds FMA result to it.
 * Stores back the C row.
*/
#define UPDATE_MASKED_C_4 \
\
    vmovupd( mem(rcx), zmm30 MASK_KZ(3) ) \
    vfmadd231pd( zmm31,zmm30,zmm0 ) \
\
    vmovupd( mem(rcx, rsi, 1, 0), zmm10 MASK_KZ(3) ) \
    vfmadd231pd( zmm31,zmm10,zmm4 ) \
\
    vmovupd( mem(rcx, rsi, 2, 0), zmm12 MASK_KZ(3) ) \
    vfmadd231pd( zmm31,zmm12,zmm2 ) \
\
    vmovupd( mem(rcx, r12, 1, 0), zmm16 MASK_KZ(3) ) \
    vfmadd231pd( zmm31,zmm16,zmm6 ) \
\
    vmovupd( zmm0, (rcx) MASK_(k(3)))            /*Stores back to C*/\
    vmovupd( zmm4, (rcx, rsi, 1) MASK_(k(3)))\
    vmovupd( zmm2, (rcx, rsi, 2) MASK_(k(3)))\
    vmovupd( zmm6, (rcx, r12, 1) MASK_(k(3)))

/**
 * Loads elements from C row only if correspondnig bits in
 * mask register is set, Scales it with Beta and adds FMA result to it.
 * Stores back the C row.
*/
#define UPDATE_MASKED_C_3 \
\
    vmovupd( mem(rcx), zmm30 MASK_KZ(3) ) \
    vfmadd231pd( zmm31,zmm30,zmm0 ) \
\
    vmovupd( mem(rcx, rsi, 1, 0), zmm10 MASK_KZ(3) ) \
    vfmadd231pd( zmm31,zmm10,zmm4 ) \
\
    vmovupd( mem(rcx, rsi, 2, 0), zmm12 MASK_KZ(3) ) \
    vfmadd231pd( zmm31,zmm12,zmm2 ) \
\
    vmovupd( zmm0, (rcx) MASK_(k(3)))            /*Stores back to C*/\
    vmovupd( zmm4, (rcx, rsi, 1) MASK_(k(3)))\
    vmovupd( zmm2, (rcx, rsi, 2) MASK_(k(3)))

/**
 * Loads elements from C row only if correspondnig bits in
 * mask register is set, Scales it with Beta and adds FMA result to it.
 * Stores back the C row.
*/
#define UPDATE_MASKED_C_2 \
\
    vmovupd( mem(rcx), zmm30 MASK_KZ(3) ) \
    vfmadd231pd( zmm31,zmm30,zmm0 ) \
\
    vmovupd( mem(rcx, rsi, 1, 0), zmm10 MASK_KZ(3) ) \
    vfmadd231pd( zmm31,zmm10,zmm4 ) \
\
    vmovupd( zmm0, (rcx) MASK_(k(3)))            /*Stores back to C*/\
    vmovupd( zmm4, (rcx, rsi, 1) MASK_(k(3)))

/**
 * Loads elements from C row only if correspondnig bits in
 * mask register is set, Scales it with Beta and adds FMA result to it.
 * Stores back the C row.
*/
#define UPDATE_MASKED_C_1 \
\
    vmovupd( mem(rcx), zmm30 MASK_KZ(3) ) \
    vfmadd231pd( zmm31,zmm30,zmm0 ) \
\
    vmovupd( zmm0, (rcx) MASK_(k(3)))            /*Stores back to C*/

/* These kernels Assume that A matrix needs to be in col-major order
 * B matrix can be col/row-major
 * C matrix can be col/row-major
 * Prefetch for C is done assuming that C is col-stored.
 * Prefetch of B is done assuming that the matrix is col-stored.
 * Prefetch for B and C matrices when row-stored is yet to be added.
 * Prefetch of A matrix is not done in edge-case kernels.
 */

void bli_dgemmsup_rv_zen4_asm_24x5
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

    uint8_t mask_n0 = 0xff >> (0x8 - (n0 & 7)); // calculate mask based on n_left
    // For special cases where n_left = 8, all 8 elements have to be loaded/stored
    // So, mask becomes 0xff(11111111)
    if (mask_n0 == 0) mask_n0 = 0xff;

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
    mov(var(mask_n0), rdx)          // load mask
    kmovw(edx, k(3))                // move mask to k3 register
    lea(mem(, r8, 8), r8)           // rs_b *= sizeof(double)
    lea(mem(, r9, 8), r9)           // cs_b *= sizeof(double)
    lea(mem(, r10, 8), r10)         // cs_a *= sizeof(double)
    lea(mem(, rdi, 8), rdi)         // cs_c *= sizeof(double)
    lea(mem(r9, r9, 2 ), r13)       // r13 = 3*cs_b
    // if n > 4, a second pointer(r12) which points to rbx + 4*cs_b
    //is also used to traverse B matrix
    lea(mem(rbx, r9, 4), r12)       // r12 = rbx + 4*cs_b
    lea(mem(rcx, 7*8), rdx)         // C for prefetching
    lea(mem(rbx, r8, 8, 7*8), r11)  // r11 = rbx + 8*rs_b(B for prefetching)
    // if n > 4, a second pointer which point to r11 + 4*cs_b
    //is also used to prefetch from B matrix
    lea(mem(r11, r9, 4), r15)       // r15 = r11 + 4* cs_b(B for prefetching)

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
    vxorpd(zmm14, zmm14, zmm14)
    vxorpd(zmm15, zmm15, zmm15)
    vxorpd(zmm24, zmm24, zmm24)

    // K is unrolled by 8 to facilitate prefetch of B
    // Assuming B to be col-stored, for each iteration of K,
    //one cacheline of B_next is prefetched where b_next = b + (unroll)*rs_b
    label(.DLOOPKITER)                                     // main loop
    mov(var(k_iter), rsi)                                  // i = k_iter
    sub(imm( 5+TAIL_NITER), rsi)                           // i -= NR + TAIL_NITER
    jle(.PREFETCHLOOP)                                     // jump if i <= 0

    label(.LOOP1)

        // ---------------------------------- iteration 1

        vmovupd( mem(rax),zmm0 )                           // load A
        vmovupd( 0x40(rax),zmm1 )
        vmovupd( 0x80(rax),zmm2 MASK_KZ(2) )     // Load A with mask and zero hint
        add( r10,rax )                                     // a += cs_a
        vmovupd( mem(rax),zmm3 )                           // load A
        vmovupd( 0x40(rax),zmm4 )
        vmovupd( 0x80(rax),zmm5 MASK_KZ(2) )     // Load A with mask and zero hint
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
        vbroadcastsd( mem(r12),zmm30 )
        add( r8,rbx )                                     // b += rs_b
        vfmadd231pd( zmm0,zmm31,zmm12 )
        vfmadd231pd( zmm1,zmm31,zmm13 )
        vfmadd231pd( zmm2,zmm31,zmm27 )
        add( r8,r12 )                                     // second pointer of b += rs_b
        vfmadd231pd( zmm0,zmm30,zmm14 )
        vfmadd231pd( zmm1,zmm30,zmm15 )
        vfmadd231pd( zmm2,zmm30,zmm24 )

        // ---------------------------------- iteration 2

        vmovupd( mem(rax),zmm0 )                           // load A
        vmovupd( 0x40(rax),zmm1 )
        vmovupd( 0x80(rax),zmm2 MASK_KZ(2) )     // Load A with mask and zero hint
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
        vbroadcastsd( mem(r12),zmm30 )
        add( r8,rbx )                                     // b += rs_b
        vfmadd231pd( zmm3,zmm31,zmm12 )
        vfmadd231pd( zmm4,zmm31,zmm13 )
        vfmadd231pd( zmm5,zmm31,zmm27 )
        add( r8,r12 )                                     // second pointer of b += rs_b
        vfmadd231pd( zmm3,zmm30,zmm14 )
        vfmadd231pd( zmm4,zmm30,zmm15 )
        vfmadd231pd( zmm5,zmm30,zmm24 )

        // ---------------------------------- iteration 3

        vmovupd( mem(rax),zmm3 )                           // load A
        vmovupd( 0x40(rax),zmm4 )
        vmovupd( 0x80(rax),zmm5 MASK_KZ(2) )     // Load A with mask and zero hint
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
        vbroadcastsd( mem(r12),zmm30 )
        add( r8,rbx )                                     // b += rs_b
        vfmadd231pd( zmm0,zmm31,zmm12 )
        vfmadd231pd( zmm1,zmm31,zmm13 )
        vfmadd231pd( zmm2,zmm31,zmm27 )
        add( r8,r12 )                                     // second pointer of b += rs_b
        vfmadd231pd( zmm0,zmm30,zmm14 )
        vfmadd231pd( zmm1,zmm30,zmm15 )
        vfmadd231pd( zmm2,zmm30,zmm24 )

        // ---------------------------------- iteration 4

        vmovupd( mem(rax),zmm0 )                           // load A
        vmovupd( 0x40(rax),zmm1 )
        vmovupd( 0x80(rax),zmm2 MASK_KZ(2) )     // Load A with mask and zero hint
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
        vbroadcastsd( mem(r12),zmm30 )
        add( r8,rbx )                                     // b += rs_b
        vfmadd231pd( zmm3,zmm31,zmm12 )
        vfmadd231pd( zmm4,zmm31,zmm13 )
        vfmadd231pd( zmm5,zmm31,zmm27 )
        add( r8,r12 )                                     // second pointer of b += rs_b
        vfmadd231pd( zmm3,zmm30,zmm14 )
        vfmadd231pd( zmm4,zmm30,zmm15 )
        vfmadd231pd( zmm5,zmm30,zmm24 )

        // ---------------------------------- iteration 5

        vmovupd( mem(rax),zmm3 )                           // load A
        vmovupd( 0x40(rax),zmm4 )
        vmovupd( 0x80(rax),zmm5 MASK_KZ(2) )     // Load A with mask and zero hint
        add( r10,rax )                                     // a += cs_a
        prefetch( 0,mem(r15) )                             // prefetch B
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
        vbroadcastsd( mem(r12),zmm30 )
        add( r8,rbx )                                     // b += rs_b
        vfmadd231pd( zmm0,zmm31,zmm12 )
        vfmadd231pd( zmm1,zmm31,zmm13 )
        vfmadd231pd( zmm2,zmm31,zmm27 )
        add( r8,r12 )                                     // second pointer of b += rs_b
        vfmadd231pd( zmm0,zmm30,zmm14 )
        vfmadd231pd( zmm1,zmm30,zmm15 )
        vfmadd231pd( zmm2,zmm30,zmm24 )

        // ---------------------------------- iteration 6

        vmovupd( mem(rax),zmm0 )                           // load A
        vmovupd( 0x40(rax),zmm1 )
        vmovupd( 0x80(rax),zmm2 MASK_KZ(2) )     // Load A with mask and zero hint
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
        vbroadcastsd( mem(r12),zmm30 )
        add( r8,rbx )                                     // b += rs_b
        vfmadd231pd( zmm3,zmm31,zmm12 )
        vfmadd231pd( zmm4,zmm31,zmm13 )
        vfmadd231pd( zmm5,zmm31,zmm27 )
        add( r8,r12 )                                     // second pointer of b += rs_b
        vfmadd231pd( zmm3,zmm30,zmm14 )
        vfmadd231pd( zmm4,zmm30,zmm15 )
        vfmadd231pd( zmm5,zmm30,zmm24 )

        // ---------------------------------- iteration 7

        vmovupd( mem(rax),zmm3 )                           // load A
        vmovupd( 0x40(rax),zmm4 )
        vmovupd( 0x80(rax),zmm5 MASK_KZ(2) )     // Load A with mask and zero hint
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
        vbroadcastsd( mem(r12),zmm30 )
        add( r8,rbx )                                     // b += rs_b
        vfmadd231pd( zmm0,zmm31,zmm12 )
        vfmadd231pd( zmm1,zmm31,zmm13 )
        vfmadd231pd( zmm2,zmm31,zmm27 )
        add( r8,r12 )                                     // second pointer of b += rs_b
        vfmadd231pd( zmm0,zmm30,zmm14 )
        vfmadd231pd( zmm1,zmm30,zmm15 )
        vfmadd231pd( zmm2,zmm30,zmm24 )

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
        vbroadcastsd( mem(r12),zmm30 )
        add( r8,rbx )                                     // b += rs_b
        vfmadd231pd( zmm3,zmm31,zmm12 )
        vfmadd231pd( zmm4,zmm31,zmm13 )
        vfmadd231pd( zmm5,zmm31,zmm27 )
        add( r8,r12 )                                     // second pointer of b += rs_b
        vfmadd231pd( zmm3,zmm30,zmm14 )
        vfmadd231pd( zmm4,zmm30,zmm15 )
        vfmadd231pd( zmm5,zmm30,zmm24 )
        lea(mem(r11,r8,8), r11)                            // b_next += 8*rs_b
        lea(mem(r15,r8,8), r15)                            // second pointer to b_next += 8*rs_b
        dec(rsi)                                           // i -= 1
    jnz(.LOOP1)                                            // iterate again if i != 0.

    label(.PREFETCHLOOP)
    add(imm(5), rsi)                                       // i += NR
    jle(.TAILITER)                                         // jump if i <= 0.

    label(.LOOP2)

        // ---------------------------------- iteration 1
        prefetchw0( mem(rdx))                              // prefetch C
        vmovupd( mem(rax),zmm0 )                           // load A
        vmovupd( 0x40(rax),zmm1 )
        vmovupd( 0x80(rax),zmm2 MASK_KZ(2) )     // Load A with mask and zero hint
        add( r10,rax )                                     // a += cs_a
        vmovupd( mem(rax),zmm3 )                           // load A
        vmovupd( 0x40(rax),zmm4 )
        vmovupd( 0x80(rax),zmm5 MASK_KZ(2) )     // Load A with mask and zero hint
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
        vbroadcastsd( mem(r12),zmm30 )
        add( r8,rbx )                                     // b += rs_b
        vfmadd231pd( zmm0,zmm31,zmm12 )
        vfmadd231pd( zmm1,zmm31,zmm13 )
        vfmadd231pd( zmm2,zmm31,zmm27 )
        add( r8,r12 )                                     // second pointer of b += rs_b
        vfmadd231pd( zmm0,zmm30,zmm14 )
        vfmadd231pd( zmm1,zmm30,zmm15 )
        vfmadd231pd( zmm2,zmm30,zmm24 )

        // ---------------------------------- iteration 2
        prefetchw0( mem(rdx, 64))                          // prefetch C
        vmovupd( mem(rax),zmm0 )                           // load A
        vmovupd( 0x40(rax),zmm1 )
        vmovupd( 0x80(rax),zmm2 MASK_KZ(2) )     // Load A with mask and zero hint
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
        vbroadcastsd( mem(r12),zmm30 )
        add( r8,rbx )                                     // b += rs_b
        vfmadd231pd( zmm3,zmm31,zmm12 )
        vfmadd231pd( zmm4,zmm31,zmm13 )
        vfmadd231pd( zmm5,zmm31,zmm27 )
        add( r8,r12 )                                     // second pointer of b += rs_b
        vfmadd231pd( zmm3,zmm30,zmm14 )
        vfmadd231pd( zmm4,zmm30,zmm15 )
        vfmadd231pd( zmm5,zmm30,zmm24 )

        // ---------------------------------- iteration 3
        prefetchw0( mem(rdx, 128))                        // prefetch C
        vmovupd( mem(rax),zmm3 )                           // load A
        vmovupd( 0x40(rax),zmm4 )
        vmovupd( 0x80(rax),zmm5 MASK_KZ(2) )     // Load A with mask and zero hint
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
        vbroadcastsd( mem(r12),zmm30 )
        add( r8,rbx )                                     // b += rs_b
        vfmadd231pd( zmm0,zmm31,zmm12 )
        vfmadd231pd( zmm1,zmm31,zmm13 )
        vfmadd231pd( zmm2,zmm31,zmm27 )
        add( r8,r12 )                                     // second pointer of b += rs_b
        vfmadd231pd( zmm0,zmm30,zmm14 )
        vfmadd231pd( zmm1,zmm30,zmm15 )
        vfmadd231pd( zmm2,zmm30,zmm24 )

        // ---------------------------------- iteration 4
        vmovupd( mem(rax),zmm0 )                           // load A
        vmovupd( 0x40(rax),zmm1 )
        vmovupd( 0x80(rax),zmm2 MASK_KZ(2) )     // Load A with mask and zero hint
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
        vbroadcastsd( mem(r12),zmm30 )
        add( r8,rbx )                                     // b += rs_b
        vfmadd231pd( zmm3,zmm31,zmm12 )
        vfmadd231pd( zmm4,zmm31,zmm13 )
        vfmadd231pd( zmm5,zmm31,zmm27 )
        add( r8,r12 )                                     // second pointer of b += rs_b
        vfmadd231pd( zmm3,zmm30,zmm14 )
        vfmadd231pd( zmm4,zmm30,zmm15 )
        vfmadd231pd( zmm5,zmm30,zmm24 )

        // ---------------------------------- iteration 5
        vmovupd( mem(rax),zmm3 )                           // load A
        vmovupd( 0x40(rax),zmm4 )
        vmovupd( 0x80(rax),zmm5 MASK_KZ(2) )     // Load A with mask and zero hint
        add( r10,rax )                                     // a += cs_a
        prefetch( 0,mem(r15) )                             // prefetch B
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
        vbroadcastsd( mem(r12),zmm30 )
        add( r8,rbx )                                     // b += rs_b
        vfmadd231pd( zmm0,zmm31,zmm12 )
        vfmadd231pd( zmm1,zmm31,zmm13 )
        vfmadd231pd( zmm2,zmm31,zmm27 )
        add( r8,r12 )                                     // second pointer of b += rs_b
        vfmadd231pd( zmm0,zmm30,zmm14 )
        vfmadd231pd( zmm1,zmm30,zmm15 )
        vfmadd231pd( zmm2,zmm30,zmm24 )

        // ---------------------------------- iteration 6
        vmovupd( mem(rax),zmm0 )                           // load A
        vmovupd( 0x40(rax),zmm1 )
        vmovupd( 0x80(rax),zmm2 MASK_KZ(2) )     // Load A with mask and zero hint
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
        vbroadcastsd( mem(r12),zmm30 )
        add( r8,rbx )                                     // b += rs_b
        vfmadd231pd( zmm3,zmm31,zmm12 )
        vfmadd231pd( zmm4,zmm31,zmm13 )
        vfmadd231pd( zmm5,zmm31,zmm27 )
        add( r8,r12 )                                     // second pointer of b += rs_b
        vfmadd231pd( zmm3,zmm30,zmm14 )
        vfmadd231pd( zmm4,zmm30,zmm15 )
        vfmadd231pd( zmm5,zmm30,zmm24 )

        // ---------------------------------- iteration 7
        vmovupd( mem(rax),zmm3 )                           // load A
        vmovupd( 0x40(rax),zmm4 )
        vmovupd( 0x80(rax),zmm5 MASK_KZ(2) )     // Load A with mask and zero hint
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
        vbroadcastsd( mem(r12),zmm30 )
        add( r8,rbx )                                     // b += rs_b
        vfmadd231pd( zmm0,zmm31,zmm12 )
        vfmadd231pd( zmm1,zmm31,zmm13 )
        vfmadd231pd( zmm2,zmm31,zmm27 )
        add( r8,r12 )                                     // second pointer of b += rs_b
        vfmadd231pd( zmm0,zmm30,zmm14 )
        vfmadd231pd( zmm1,zmm30,zmm15 )
        vfmadd231pd( zmm2,zmm30,zmm24 )

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
        vbroadcastsd( mem(r12),zmm30 )
        add( r8,rbx )                                     // b += rs_b
        vfmadd231pd( zmm3,zmm31,zmm12 )
        vfmadd231pd( zmm4,zmm31,zmm13 )
        vfmadd231pd( zmm5,zmm31,zmm27 )
        add( r8,r12 )                                     // second pointer of b += rs_b
        vfmadd231pd( zmm3,zmm30,zmm14 )
        vfmadd231pd( zmm4,zmm30,zmm15 )
        vfmadd231pd( zmm5,zmm30,zmm24 )
        lea(mem(rdx, rdi, 1), rdx)                         // C += cs_c
        lea(mem(r11,r8,8), r11)                            // b_next += 8*rs_b
        lea(mem(r15,r8,8), r15)                            // second pointer of b_next += 8*rs_b
        sub(imm(1), rsi)                                   // i -= 1
    jnz(.LOOP2)                                            // iterate again if i != 0.
    label(.TAILITER)
    add(imm(TAIL_NITER), rsi)                              // i += TAIL_NITER
    jle(.TAIL)                                             // jump if i <= 0

    label(.LOOP3)

        // ---------------------------------- iteration 1
        vmovupd( mem(rax),zmm0 )                           // load A
        vmovupd( 0x40(rax),zmm1 )
        vmovupd( 0x80(rax),zmm2 MASK_KZ(2) )     // Load A with mask and zero hint
        add( r10,rax )                                     // a += cs_a
        vmovupd( mem(rax),zmm3 )                           // load A
        vmovupd( 0x40(rax),zmm4 )
        vmovupd( 0x80(rax),zmm5 MASK_KZ(2) )     // Load A with mask and zero hint
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
        vbroadcastsd( mem(r12),zmm30 )
        add( r8,rbx )                                     // b += rs_b
        vfmadd231pd( zmm0,zmm31,zmm12 )
        vfmadd231pd( zmm1,zmm31,zmm13 )
        vfmadd231pd( zmm2,zmm31,zmm27 )
        add( r8,r12 )                                     // second pointer of b += rs_b
        vfmadd231pd( zmm0,zmm30,zmm14 )
        vfmadd231pd( zmm1,zmm30,zmm15 )
        vfmadd231pd( zmm2,zmm30,zmm24 )

        // ---------------------------------- iteration 2
        vmovupd( mem(rax),zmm0 )                           // load A
        vmovupd( 0x40(rax),zmm1 )
        vmovupd( 0x80(rax),zmm2 MASK_KZ(2) )     // Load A with mask and zero hint
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
        vbroadcastsd( mem(r12),zmm30 )
        add( r8,rbx )                                     // b += rs_b
        vfmadd231pd( zmm3,zmm31,zmm12 )
        vfmadd231pd( zmm4,zmm31,zmm13 )
        vfmadd231pd( zmm5,zmm31,zmm27 )
        add( r8,r12 )                                     // second pointer of b += rs_b
        vfmadd231pd( zmm3,zmm30,zmm14 )
        vfmadd231pd( zmm4,zmm30,zmm15 )
        vfmadd231pd( zmm5,zmm30,zmm24 )

        // ---------------------------------- iteration 3
        vmovupd( mem(rax),zmm3 )                           // load A
        vmovupd( 0x40(rax),zmm4 )
        vmovupd( 0x80(rax),zmm5 MASK_KZ(2) )     // Load A with mask and zero hint
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
        vbroadcastsd( mem(r12),zmm30 )
        add( r8,rbx )                                     // b += rs_b
        vfmadd231pd( zmm0,zmm31,zmm12 )
        vfmadd231pd( zmm1,zmm31,zmm13 )
        vfmadd231pd( zmm2,zmm31,zmm27 )
        add( r8,r12 )                                     // second pointer of b += rs_b
        vfmadd231pd( zmm0,zmm30,zmm14 )
        vfmadd231pd( zmm1,zmm30,zmm15 )
        vfmadd231pd( zmm2,zmm30,zmm24 )

        // ---------------------------------- iteration 4
        vmovupd( mem(rax),zmm0 )                           // load A
        vmovupd( 0x40(rax),zmm1 )
        vmovupd( 0x80(rax),zmm2 MASK_KZ(2) )     // Load A with mask and zero hint
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
        vbroadcastsd( mem(r12),zmm30 )
        add( r8,rbx )                                     // b += rs_b
        vfmadd231pd( zmm3,zmm31,zmm12 )
        vfmadd231pd( zmm4,zmm31,zmm13 )
        vfmadd231pd( zmm5,zmm31,zmm27 )
        add( r8,r12 )                                     // second pointer of b += rs_b
        vfmadd231pd( zmm3,zmm30,zmm14 )
        vfmadd231pd( zmm4,zmm30,zmm15 )
        vfmadd231pd( zmm5,zmm30,zmm24 )

        // ---------------------------------- iteration 5
        vmovupd( mem(rax),zmm3 )                           // load A
        vmovupd( 0x40(rax),zmm4 )
        vmovupd( 0x80(rax),zmm5 MASK_KZ(2) )     // Load A with mask and zero hint
        add( r10,rax )                                     // a += cs_a
        prefetch( 0,mem(r15) )                             // prefetch B
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
        vbroadcastsd( mem(r12),zmm30 )
        add( r8,rbx )                                     // b += rs_b
        vfmadd231pd( zmm0,zmm31,zmm12 )
        vfmadd231pd( zmm1,zmm31,zmm13 )
        vfmadd231pd( zmm2,zmm31,zmm27 )
        add( r8,r12 )                                     // second pointer of b += rs_b
        vfmadd231pd( zmm0,zmm30,zmm14 )
        vfmadd231pd( zmm1,zmm30,zmm15 )
        vfmadd231pd( zmm2,zmm30,zmm24 )

        // ---------------------------------- iteration 6
        vmovupd( mem(rax),zmm0 )                           // load A
        vmovupd( 0x40(rax),zmm1 )
        vmovupd( 0x80(rax),zmm2 MASK_KZ(2) )     // Load A with mask and zero hint
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
        vbroadcastsd( mem(r12),zmm30 )
        add( r8,rbx )                                     // b += rs_b
        vfmadd231pd( zmm3,zmm31,zmm12 )
        vfmadd231pd( zmm4,zmm31,zmm13 )
        vfmadd231pd( zmm5,zmm31,zmm27 )
        add( r8,r12 )                                     // second pointer of b += rs_b
        vfmadd231pd( zmm3,zmm30,zmm14 )
        vfmadd231pd( zmm4,zmm30,zmm15 )
        vfmadd231pd( zmm5,zmm30,zmm24 )

        // ---------------------------------- iteration 7
        vmovupd( mem(rax),zmm3 )                           // load A
        vmovupd( 0x40(rax),zmm4 )
        vmovupd( 0x80(rax),zmm5 MASK_KZ(2) )     // Load A with mask and zero hint
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
        vbroadcastsd( mem(r12),zmm30 )
        add( r8,rbx )                                     // b += rs_b
        vfmadd231pd( zmm0,zmm31,zmm12 )
        vfmadd231pd( zmm1,zmm31,zmm13 )
        vfmadd231pd( zmm2,zmm31,zmm27 )
        add( r8,r12 )                                     // second pointer of b += rs_b
        vfmadd231pd( zmm0,zmm30,zmm14 )
        vfmadd231pd( zmm1,zmm30,zmm15 )
        vfmadd231pd( zmm2,zmm30,zmm24 )

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
        vbroadcastsd( mem(r12),zmm30 )
        add( r8,rbx )                                     // b += rs_b
        vfmadd231pd( zmm3,zmm31,zmm12 )
        vfmadd231pd( zmm4,zmm31,zmm13 )
        vfmadd231pd( zmm5,zmm31,zmm27 )
        add( r8,r12 )                                     // second pointer of b += rs_b
        vfmadd231pd( zmm3,zmm30,zmm14 )
        vfmadd231pd( zmm4,zmm30,zmm15 )
        vfmadd231pd( zmm5,zmm30,zmm24 )
        lea(mem(r11,r8,8), r11)                            // b_next += 8*rs_b
        lea(mem(r15,r8,8), r15)                            // Second pointer of b_next += 8*rs_b
        dec(rsi)                                           // i -= 1
    jnz(.LOOP3)                                            // iterate again if i != 0.


    label(.TAIL)
    mov(var(k_left), rsi)                                  // i = k_left
    test(rsi, rsi)                                         // check i via logical AND
    je(.DPOSTACCUM)                                        // if i == 0, jump to post-accumulation

    label(.DLOOPKLEFT)                                     // k_left loop
        vmovupd( mem(rax),zmm0 )                           // load A
        vmovupd( 0x40(rax),zmm1 )
        vmovupd( 0x80(rax),zmm2 MASK_KZ(2) )     // Load A with mask and zero hint
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
        vbroadcastsd( mem(r12),zmm30 )
        add( r8,rbx )                                     // b += rs_b
        vfmadd231pd( zmm0,zmm31,zmm12 )
        vfmadd231pd( zmm1,zmm31,zmm13 )
        vfmadd231pd( zmm2,zmm31,zmm27 )
        add( r8,r12 )                                     // second pointer of b += rs_b
        vfmadd231pd( zmm0,zmm30,zmm14 )
        vfmadd231pd( zmm1,zmm30,zmm15 )
        vfmadd231pd( zmm2,zmm30,zmm24 )
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
    vmulpd( zmm30,zmm14,zmm14 )
    vmulpd( zmm30,zmm15,zmm15 )
    vmulpd( zmm30,zmm24,zmm24 )


    mov(var(rs_c), rsi)                                    // load rs_c
    lea(mem(, rsi, 8), rsi)                                // rsi = rs_c * sizeof(double)
    lea(mem(rcx, rdi, 4), rdx)                             // rdx = rcx + 4 * cs_c
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
    vmovupd( 0x80(rcx),zmm2 MASK_KZ(2))        // Load C using mask and zero hint
    vfmadd231pd( zmm2,zmm31,zmm28)
    vmovupd( zmm28,0x80(rcx) MASK_(k(2)))                // store to C with mask
    vmovupd( mem(rcx,rdi,1),zmm3)
    vfmadd231pd( zmm3,zmm31,zmm8)
    vmovupd( zmm8,(rcx,rdi,1))
    vmovupd( 0x40(rcx,rdi,1),zmm4)
    vfmadd231pd( zmm4,zmm31,zmm9)
    vmovupd( zmm9,0x40(rcx,rdi,1))
    vmovupd( 0x80(rcx,rdi,1),zmm5 MASK_KZ(2))        // Load C using mask and zero hint
    vfmadd231pd( zmm5,zmm31,zmm29)
    vmovupd( zmm29,0x80(rcx,rdi,1) MASK_(k(2)))                // store to C with mask
    vmovupd( mem(rcx,rdi,2),zmm0)
    vfmadd231pd( zmm0,zmm31,zmm10)
    vmovupd( zmm10,(rcx,rdi,2))
    vmovupd( 0x40(rcx,rdi,2),zmm1)
    vfmadd231pd( zmm1,zmm31,zmm11)
    vmovupd( zmm11,0x40(rcx,rdi,2))
    vmovupd( 0x80(rcx,rdi,2),zmm2 MASK_KZ(2))        // Load C using mask and zero hint
    vfmadd231pd( zmm2,zmm31,zmm26)
    vmovupd( zmm26,0x80(rcx,rdi,2) MASK_(k(2)))                // store to C with mask
    vmovupd( mem(rcx,r13,1),zmm3)
    vfmadd231pd( zmm3,zmm31,zmm12)
    vmovupd( zmm12,(rcx,r13,1))
    vmovupd( 0x40(rcx,r13,1),zmm4)
    vfmadd231pd( zmm4,zmm31,zmm13)
    vmovupd( zmm13,0x40(rcx,r13,1))
    vmovupd( 0x80(rcx,r13,1),zmm5 MASK_KZ(2))        // Load C using mask and zero hint
    vfmadd231pd( zmm5,zmm31,zmm27)
    vmovupd( zmm27,0x80(rcx,r13,1) MASK_(k(2)))                // store to C with mask
    vmovupd( mem(rdx),zmm0)
    vfmadd231pd( zmm0,zmm31,zmm14)
    vmovupd( zmm14,(rdx))
    vmovupd( 0x40(rdx),zmm1)
    vfmadd231pd( zmm1,zmm31,zmm15)
    vmovupd( zmm15,0x40(rdx))
    vmovupd( 0x80(rdx),zmm2 MASK_KZ(2))        // Load C using mask and zero hint
    vfmadd231pd( zmm2,zmm31,zmm24)
    vmovupd( zmm24,0x80(rdx) MASK_(k(2)))                // store to C with mask

    jmp(.DDONE)                                           // jump to end.

    label(.DROWSTORED)
    lea(mem(rsi,  rsi,  2), r12)
    lea(mem(r12, rsi,  2), r13)
    lea(mem(r12, rsi,  4), rdx)
    lea(mem(   , rsi, 8), r14)
    UNPACK_LO_HIGH(8, 6, 0, 1, 12, 10, 2, 3)
    SHUFFLE_DATA(2, 0, 4, 5, 3, 1, 30, 31)

    vunpcklpd(zmm16, zmm14, zmm0)
    vunpckhpd(zmm16, zmm14, zmm1)
    SHUFFLE_DATA(2, 0, 6, 8, 3, 1, 10, 12)

    SHUFFLE_DATA(6, 4, 0, 1, 8, 5, 2, 3)
    SHUFFLE_DATA(10, 30, 4, 5, 12, 31, 6, 8)

    vbroadcastsd(mem(rax), zmm31)
    UPDATE_MASKED_C_8
    //First 8x5 tile updated

    UNPACK_LO_HIGH(9, 7, 0, 1, 13, 11, 2, 3)
    SHUFFLE_DATA(2, 0, 4, 5, 3, 1, 7, 9)

    vunpcklpd(zmm17, zmm15, zmm0)
    vunpckhpd(zmm17, zmm15, zmm1)
    SHUFFLE_DATA(2, 0, 6, 8, 3, 1, 10, 12)

    SHUFFLE_DATA(6, 4, 0, 1, 8, 5, 2, 3)
    SHUFFLE_DATA(10, 7, 4, 5, 12, 9, 6, 8)

    UPDATE_MASKED_C_8
    //Second 8x5 tile updated

    UNPACK_LO_HIGH(29, 28, 0, 1, 27, 26, 2, 3)
    SHUFFLE_DATA(2, 0, 4, 5, 3, 1, 7, 9)

    vunpcklpd(zmm25, zmm24, zmm0)
    vunpckhpd(zmm25, zmm24, zmm1)
    SHUFFLE_DATA(2, 0, 6, 8, 3, 1, 10, 12)

    SHUFFLE_DATA(6, 4, 0, 1, 8, 5, 2, 3)
    SHUFFLE_DATA(10, 7, 4, 5, 12, 9, 6, 8)

    mov(var(m0), rdi)
    sub(imm(16), rdi)

    cmp(imm(8), rdi)
    JZ(.UPDATE8)
    cmp(imm(7), rdi)
    JZ(.UPDATE7)
    cmp(imm(6), rdi)
    JZ(.UPDATE6)
    cmp(imm(5), rdi)
    JZ(.UPDATE5)
    cmp(imm(4), rdi)
    JZ(.UPDATE4)
    cmp(imm(3), rdi)
    JZ(.UPDATE3)
    cmp(imm(2), rdi)
    JZ(.UPDATE2)
    cmp(imm(1), rdi)
    JZ(.UPDATE1)
    cmp(imm(0), rdi)
    JZ(.UPDATE0)

    LABEL(.UPDATE8)
    UPDATE_MASKED_C_8
    jmp(.DDONE)

    LABEL(.UPDATE7)
    UPDATE_MASKED_C_7
    jmp(.DDONE)

    LABEL(.UPDATE6)
    UPDATE_MASKED_C_6
    jmp(.DDONE)                                              // jump to end.

    LABEL(.UPDATE5)
    UPDATE_MASKED_C_5
    jmp(.DDONE)                                              // jump to end.

    LABEL(.UPDATE4)
    UPDATE_MASKED_C_4
    jmp(.DDONE)                                              // jump to end.

    LABEL(.UPDATE3)
    UPDATE_MASKED_C_3
    jmp(.DDONE)                                              // jump to end.

    LABEL(.UPDATE2)
    UPDATE_MASKED_C_2
    jmp(.DDONE)                                              // jump to end.

    LABEL(.UPDATE1)
    UPDATE_MASKED_C_1
    jmp(.DDONE)                                              // jump to end.

    LABEL(.UPDATE0)
    //Third 8x8 tile updated
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
    vmovupd( zmm14,(rdx))
    vmovupd( zmm15,0x40(rdx))
    vmovupd( zmm24,0x80(rdx) MASK_(k(2)))                // store to C with mask

    jmp(.DDONE)                                          // jump to end.


    label(.DROWSTORBZ)
    lea(mem(rsi,  rsi,  2), r12)
    lea(mem(r12, rsi,  2), r13)
    lea(mem(r12, rsi,  4), rdx)
    lea(mem(   , rsi, 8), r14)
    UNPACK_LO_HIGH(8, 6, 0, 1, 12, 10, 2, 3)
    SHUFFLE_DATA(2, 0, 4, 5, 3, 1, 30, 31)

    vunpcklpd(zmm16, zmm14, zmm0)
    vunpckhpd(zmm16, zmm14, zmm1)
    SHUFFLE_DATA(2, 0, 6, 8, 3, 1, 10, 12)

    SHUFFLE_DATA(6, 4, 0, 1, 8, 5, 2, 3)
    SHUFFLE_DATA(10, 30, 4, 5, 12, 31, 6, 8)

    UPDATE_MASKED_C_8_BZ
    //First 8x5 tile updated

    UNPACK_LO_HIGH(9, 7, 0, 1, 13, 11, 2, 3)
    SHUFFLE_DATA(2, 0, 4, 5, 3, 1, 7, 9)

    vunpcklpd(zmm17, zmm15, zmm0)
    vunpckhpd(zmm17, zmm15, zmm1)
    SHUFFLE_DATA(2, 0, 6, 8, 3, 1, 10, 12)

    SHUFFLE_DATA(6, 4, 0, 1, 8, 5, 2, 3)
    SHUFFLE_DATA(10, 7, 4, 5, 12, 9, 6, 8)

    UPDATE_MASKED_C_8_BZ
    //Second 8x5 tile updated

    UNPACK_LO_HIGH(29, 28, 0, 1, 27, 26, 2, 3)
    SHUFFLE_DATA(2, 0, 4, 5, 3, 1, 7, 9)

    vunpcklpd(zmm25, zmm24, zmm0)
    vunpckhpd(zmm25, zmm24, zmm1)
    SHUFFLE_DATA(2, 0, 6, 8, 3, 1, 10, 12)

    SHUFFLE_DATA(6, 4, 0, 1, 8, 5, 2, 3)
    SHUFFLE_DATA(10, 7, 4, 5, 12, 9, 6, 8)

    mov(var(m0), rdi)
    sub(imm(16), rdi)

    cmp(imm(8), rdi)
    JZ(.UPDATE8BZ)
    cmp(imm(7), rdi)
    JZ(.UPDATE7BZ)
    cmp(imm(6), rdi)
    JZ(.UPDATE6BZ)
    cmp(imm(5), rdi)
    JZ(.UPDATE5BZ)
    cmp(imm(4), rdi)
    JZ(.UPDATE4BZ)
    cmp(imm(3), rdi)
    JZ(.UPDATE3BZ)
    cmp(imm(2), rdi)
    JZ(.UPDATE2BZ)
    cmp(imm(1), rdi)
    JZ(.UPDATE1BZ)
    cmp(imm(0), rdi)
    JZ(.UPDATE0BZ)

    LABEL(.UPDATE8BZ)
    UPDATE_MASKED_C_8_BZ
    jmp(.DDONE)

    LABEL(.UPDATE7BZ)
    UPDATE_MASKED_C_7_BZ
    jmp(.DDONE)

    LABEL(.UPDATE6BZ)
    UPDATE_MASKED_C_6_BZ
    jmp(.DDONE)                                              // jump to end.

    LABEL(.UPDATE5BZ)
    UPDATE_MASKED_C_5_BZ
    jmp(.DDONE)                                              // jump to end.

    LABEL(.UPDATE4BZ)
    UPDATE_MASKED_C_4_BZ
    jmp(.DDONE)                                              // jump to end.

    LABEL(.UPDATE3BZ)
    UPDATE_MASKED_C_3_BZ
    jmp(.DDONE)                                              // jump to end.

    LABEL(.UPDATE2BZ)
    UPDATE_MASKED_C_2_BZ
    jmp(.DDONE)                                              // jump to end.

    LABEL(.UPDATE1BZ)
    UPDATE_MASKED_C_1_BZ
    jmp(.DDONE)                                              // jump to end.

    LABEL(.UPDATE0BZ)
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
        [mask]   "m" (mask),
        [mask_n0]   "m" (mask_n0)
      : // register clobber list
        "rax", "rbx", "rcx", "rdx", "rsi", "rdi",
        "r8", "r9", "r10", "r11", "r12", "r13", "r14", "r15",
        "xmm2", "xmm31",
        "ymm2",
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


void bli_dgemmsup_rv_zen4_asm_16x5
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

    uint8_t mask_n0 = 0xff >> (0x8 - (n0 & 7)); // calculate mask based on n_left
    // For special cases where n_left = 8, all 8 elements have to be loaded/stored
    // So, mask becomes 0xff(11111111)
    if (mask_n0 == 0) mask_n0 = 0xff;

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
    mov(var(mask_n0), rdx)          // load mask
    kmovw(edx, k(3))                // move mask to k3 register
    lea(mem(, r8, 8), r8)           // rs_b *= sizeof(double)
    lea(mem(, r9, 8), r9)           // cs_b *= sizeof(double)
    lea(mem(, r10, 8), r10)         // cs_a *= sizeof(double)
    lea(mem(, rdi, 8), rdi)         // cs_c *= sizeof(double)
    lea(mem(r9, r9, 2 ), r13)       // r13 = 3*cs_b
    // if n > 4, a second pointer(r12) which points to rbx + 4*cs_b
    //is also used to traverse B matrix
    lea(mem(rbx, r9, 4), r12)       // r12 = rbx + 4*cs_b
    lea(mem(rcx, 7*8), rdx)         // C for prefetching
    lea(mem(rbx, r8, 8, 7*8), r11)  // r11 = rbx + 8*rs_b(B for prefetching)
    // if n > 4, a second pointer which point to r11 + 4*cs_b
    //is also used to prefetch from B matrix
    lea(mem(r11, r9, 4), r15)       // r15 = r11 + 4* cs_b(B for prefetching)

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
    vxorpd(zmm14, zmm14, zmm14)
    vxorpd(zmm15, zmm15, zmm15)

    // K is unrolled by 8 to facilitate prefetch of B
    // Assuming B to be col-stored, for each iteration of K,
    //one cacheline of B_next is prefetched where b_next = b + (unroll)*rs_b
    label(.DLOOPKITER)                                     // main loop
    mov(var(k_iter), rsi)                                  // i = k_iter
    sub(imm( 5+TAIL_NITER), rsi)                           // i -= NR + TAIL_NITER
    jle(.PREFETCHLOOP)                                     // jump if i <= 0

    label(.LOOP1)

        // ---------------------------------- iteration 1

        vmovupd( mem(rax),zmm0 )                           // load A
        vmovupd( 0x40(rax),zmm1 MASK_KZ(2) )     // Load A with mask and zero hint
        add( r10,rax )                                     // a += cs_a
        vmovupd( mem(rax),zmm3 )                           // load A
        vmovupd( 0x40(rax),zmm4 MASK_KZ(2) )     // Load A with mask and zero hint
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
        vbroadcastsd( mem(r12),zmm30 )
        add( r8,rbx )                                     // b += rs_b
        vfmadd231pd( zmm0,zmm31,zmm12 )
        vfmadd231pd( zmm1,zmm31,zmm13 )
        add( r8,r12 )                                     // second pointer of b += rs_b
        vfmadd231pd( zmm0,zmm30,zmm14 )
        vfmadd231pd( zmm1,zmm30,zmm15 )

        // ---------------------------------- iteration 2

        vmovupd( mem(rax),zmm0 )                           // load A
        vmovupd( 0x40(rax),zmm1 MASK_KZ(2) )     // Load A with mask and zero hint
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
        vbroadcastsd( mem(r12),zmm30 )
        add( r8,rbx )                                     // b += rs_b
        vfmadd231pd( zmm3,zmm31,zmm12 )
        vfmadd231pd( zmm4,zmm31,zmm13 )
        add( r8,r12 )                                     // second pointer of b += rs_b
        vfmadd231pd( zmm3,zmm30,zmm14 )
        vfmadd231pd( zmm4,zmm30,zmm15 )

        // ---------------------------------- iteration 3

        vmovupd( mem(rax),zmm3 )                           // load A
        vmovupd( 0x40(rax),zmm4 MASK_KZ(2) )     // Load A with mask and zero hint
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
        vbroadcastsd( mem(r12),zmm30 )
        add( r8,rbx )                                     // b += rs_b
        vfmadd231pd( zmm0,zmm31,zmm12 )
        vfmadd231pd( zmm1,zmm31,zmm13 )
        add( r8,r12 )                                     // second pointer of b += rs_b
        vfmadd231pd( zmm0,zmm30,zmm14 )
        vfmadd231pd( zmm1,zmm30,zmm15 )

        // ---------------------------------- iteration 4

        vmovupd( mem(rax),zmm0 )                           // load A
        vmovupd( 0x40(rax),zmm1 MASK_KZ(2) )     // Load A with mask and zero hint
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
        vbroadcastsd( mem(r12),zmm30 )
        add( r8,rbx )                                     // b += rs_b
        vfmadd231pd( zmm3,zmm31,zmm12 )
        vfmadd231pd( zmm4,zmm31,zmm13 )
        add( r8,r12 )                                     // second pointer of b += rs_b
        vfmadd231pd( zmm3,zmm30,zmm14 )
        vfmadd231pd( zmm4,zmm30,zmm15 )

        // ---------------------------------- iteration 5

        vmovupd( mem(rax),zmm3 )                           // load A
        vmovupd( 0x40(rax),zmm4 MASK_KZ(2) )     // Load A with mask and zero hint
        add( r10,rax )                                     // a += cs_a
        prefetch( 0,mem(r15) )                             // prefetch B
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
        vbroadcastsd( mem(r12),zmm30 )
        add( r8,rbx )                                     // b += rs_b
        vfmadd231pd( zmm0,zmm31,zmm12 )
        vfmadd231pd( zmm1,zmm31,zmm13 )
        add( r8,r12 )                                     // second pointer of b += rs_b
        vfmadd231pd( zmm0,zmm30,zmm14 )
        vfmadd231pd( zmm1,zmm30,zmm15 )

        // ---------------------------------- iteration 6

        vmovupd( mem(rax),zmm0 )                           // load A
        vmovupd( 0x40(rax),zmm1 MASK_KZ(2) )     // Load A with mask and zero hint
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
        vbroadcastsd( mem(r12),zmm30 )
        add( r8,rbx )                                     // b += rs_b
        vfmadd231pd( zmm3,zmm31,zmm12 )
        vfmadd231pd( zmm4,zmm31,zmm13 )
        add( r8,r12 )                                     // second pointer of b += rs_b
        vfmadd231pd( zmm3,zmm30,zmm14 )
        vfmadd231pd( zmm4,zmm30,zmm15 )

        // ---------------------------------- iteration 7

        vmovupd( mem(rax),zmm3 )                           // load A
        vmovupd( 0x40(rax),zmm4 MASK_KZ(2) )     // Load A with mask and zero hint
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
        vbroadcastsd( mem(r12),zmm30 )
        add( r8,rbx )                                     // b += rs_b
        vfmadd231pd( zmm0,zmm31,zmm12 )
        vfmadd231pd( zmm1,zmm31,zmm13 )
        add( r8,r12 )                                     // second pointer of b += rs_b
        vfmadd231pd( zmm0,zmm30,zmm14 )
        vfmadd231pd( zmm1,zmm30,zmm15 )

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
        vbroadcastsd( mem(r12),zmm30 )
        add( r8,rbx )                                     // b += rs_b
        vfmadd231pd( zmm3,zmm31,zmm12 )
        vfmadd231pd( zmm4,zmm31,zmm13 )
        add( r8,r12 )                                     // second pointer of b += rs_b
        vfmadd231pd( zmm3,zmm30,zmm14 )
        vfmadd231pd( zmm4,zmm30,zmm15 )
        lea(mem(r11,r8,8), r11)                            // b_next += 8*rs_b
        lea(mem(r15,r8,8), r15)                            // second pointer to b_next += 8*rs_b
        dec(rsi)                                           // i -= 1
    jnz(.LOOP1)                                            // iterate again if i != 0.

    label(.PREFETCHLOOP)
    add(imm(5), rsi)                                       // i += NR
    jle(.TAILITER)                                         // jump if i <= 0.

    label(.LOOP2)

        // ---------------------------------- iteration 1
        prefetchw0( mem(rdx))                              // prefetch C
        vmovupd( mem(rax),zmm0 )                           // load A
        vmovupd( 0x40(rax),zmm1 MASK_KZ(2) )     // Load A with mask and zero hint
        add( r10,rax )                                     // a += cs_a
        vmovupd( mem(rax),zmm3 )                           // load A
        vmovupd( 0x40(rax),zmm4 MASK_KZ(2) )     // Load A with mask and zero hint
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
        vbroadcastsd( mem(r12),zmm30 )
        add( r8,rbx )                                     // b += rs_b
        vfmadd231pd( zmm0,zmm31,zmm12 )
        vfmadd231pd( zmm1,zmm31,zmm13 )
        add( r8,r12 )                                     // second pointer of b += rs_b
        vfmadd231pd( zmm0,zmm30,zmm14 )
        vfmadd231pd( zmm1,zmm30,zmm15 )

        // ---------------------------------- iteration 2
        prefetchw0( mem(rdx, 64))                          // prefetch C
        vmovupd( mem(rax),zmm0 )                           // load A
        vmovupd( 0x40(rax),zmm1 MASK_KZ(2) )     // Load A with mask and zero hint
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
        vbroadcastsd( mem(r12),zmm30 )
        add( r8,rbx )                                     // b += rs_b
        vfmadd231pd( zmm3,zmm31,zmm12 )
        vfmadd231pd( zmm4,zmm31,zmm13 )
        add( r8,r12 )                                     // second pointer of b += rs_b
        vfmadd231pd( zmm3,zmm30,zmm14 )
        vfmadd231pd( zmm4,zmm30,zmm15 )

        // ---------------------------------- iteration 3
        vmovupd( mem(rax),zmm3 )                           // load A
        vmovupd( 0x40(rax),zmm4 MASK_KZ(2) )     // Load A with mask and zero hint
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
        vbroadcastsd( mem(r12),zmm30 )
        add( r8,rbx )                                     // b += rs_b
        vfmadd231pd( zmm0,zmm31,zmm12 )
        vfmadd231pd( zmm1,zmm31,zmm13 )
        add( r8,r12 )                                     // second pointer of b += rs_b
        vfmadd231pd( zmm0,zmm30,zmm14 )
        vfmadd231pd( zmm1,zmm30,zmm15 )

        // ---------------------------------- iteration 4
        vmovupd( mem(rax),zmm0 )                           // load A
        vmovupd( 0x40(rax),zmm1 MASK_KZ(2) )     // Load A with mask and zero hint
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
        vbroadcastsd( mem(r12),zmm30 )
        add( r8,rbx )                                     // b += rs_b
        vfmadd231pd( zmm3,zmm31,zmm12 )
        vfmadd231pd( zmm4,zmm31,zmm13 )
        add( r8,r12 )                                     // second pointer of b += rs_b
        vfmadd231pd( zmm3,zmm30,zmm14 )
        vfmadd231pd( zmm4,zmm30,zmm15 )

        // ---------------------------------- iteration 5
        vmovupd( mem(rax),zmm3 )                           // load A
        vmovupd( 0x40(rax),zmm4 MASK_KZ(2) )     // Load A with mask and zero hint
        add( r10,rax )                                     // a += cs_a
        prefetch( 0,mem(r15) )                             // prefetch B
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
        vbroadcastsd( mem(r12),zmm30 )
        add( r8,rbx )                                     // b += rs_b
        vfmadd231pd( zmm0,zmm31,zmm12 )
        vfmadd231pd( zmm1,zmm31,zmm13 )
        add( r8,r12 )                                     // second pointer of b += rs_b
        vfmadd231pd( zmm0,zmm30,zmm14 )
        vfmadd231pd( zmm1,zmm30,zmm15 )

        // ---------------------------------- iteration 6
        vmovupd( mem(rax),zmm0 )                           // load A
        vmovupd( 0x40(rax),zmm1 MASK_KZ(2) )     // Load A with mask and zero hint
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
        vbroadcastsd( mem(r12),zmm30 )
        add( r8,rbx )                                     // b += rs_b
        vfmadd231pd( zmm3,zmm31,zmm12 )
        vfmadd231pd( zmm4,zmm31,zmm13 )
        add( r8,r12 )                                     // second pointer of b += rs_b
        vfmadd231pd( zmm3,zmm30,zmm14 )
        vfmadd231pd( zmm4,zmm30,zmm15 )

        // ---------------------------------- iteration 7
        vmovupd( mem(rax),zmm3 )                           // load A
        vmovupd( 0x40(rax),zmm4 MASK_KZ(2) )     // Load A with mask and zero hint
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
        vbroadcastsd( mem(r12),zmm30 )
        add( r8,rbx )                                     // b += rs_b
        vfmadd231pd( zmm0,zmm31,zmm12 )
        vfmadd231pd( zmm1,zmm31,zmm13 )
        add( r8,r12 )                                     // second pointer of b += rs_b
        vfmadd231pd( zmm0,zmm30,zmm14 )
        vfmadd231pd( zmm1,zmm30,zmm15 )

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
        vbroadcastsd( mem(r12),zmm30 )
        add( r8,rbx )                                     // b += rs_b
        vfmadd231pd( zmm3,zmm31,zmm12 )
        vfmadd231pd( zmm4,zmm31,zmm13 )
        add( r8,r12 )                                     // second pointer of b += rs_b
        vfmadd231pd( zmm3,zmm30,zmm14 )
        vfmadd231pd( zmm4,zmm30,zmm15 )
        lea(mem(rdx, rdi, 1), rdx)                         // C += cs_c
        lea(mem(r11,r8,8), r11)                            // b_next += 8*rs_b
        lea(mem(r15,r8,8), r15)                            // second pointer of b_next += 8*rs_b
        sub(imm(1), rsi)                                   // i -= 1
    jnz(.LOOP2)                                            // iterate again if i != 0.
    label(.TAILITER)
    add(imm(TAIL_NITER), rsi)                              // i += TAIL_NITER
    jle(.TAIL)                                             // jump if i <= 0

    label(.LOOP3)

        // ---------------------------------- iteration 1
        vmovupd( mem(rax),zmm0 )                           // load A
        vmovupd( 0x40(rax),zmm1 MASK_KZ(2) )     // Load A with mask and zero hint
        add( r10,rax )                                     // a += cs_a
        vmovupd( mem(rax),zmm3 )                           // load A
        vmovupd( 0x40(rax),zmm4 MASK_KZ(2) )     // Load A with mask and zero hint
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
        vbroadcastsd( mem(r12),zmm30 )
        add( r8,rbx )                                     // b += rs_b
        vfmadd231pd( zmm0,zmm31,zmm12 )
        vfmadd231pd( zmm1,zmm31,zmm13 )
        add( r8,r12 )                                     // second pointer of b += rs_b
        vfmadd231pd( zmm0,zmm30,zmm14 )
        vfmadd231pd( zmm1,zmm30,zmm15 )

        // ---------------------------------- iteration 2
        vmovupd( mem(rax),zmm0 )                           // load A
        vmovupd( 0x40(rax),zmm1 MASK_KZ(2) )     // Load A with mask and zero hint
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
        vbroadcastsd( mem(r12),zmm30 )
        add( r8,rbx )                                     // b += rs_b
        vfmadd231pd( zmm3,zmm31,zmm12 )
        vfmadd231pd( zmm4,zmm31,zmm13 )
        add( r8,r12 )                                     // second pointer of b += rs_b
        vfmadd231pd( zmm3,zmm30,zmm14 )
        vfmadd231pd( zmm4,zmm30,zmm15 )

        // ---------------------------------- iteration 3
        vmovupd( mem(rax),zmm3 )                           // load A
        vmovupd( 0x40(rax),zmm4 MASK_KZ(2) )     // Load A with mask and zero hint
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
        vbroadcastsd( mem(r12),zmm30 )
        add( r8,rbx )                                     // b += rs_b
        vfmadd231pd( zmm0,zmm31,zmm12 )
        vfmadd231pd( zmm1,zmm31,zmm13 )
        add( r8,r12 )                                     // second pointer of b += rs_b
        vfmadd231pd( zmm0,zmm30,zmm14 )
        vfmadd231pd( zmm1,zmm30,zmm15 )

        // ---------------------------------- iteration 4
        vmovupd( mem(rax),zmm0 )                           // load A
        vmovupd( 0x40(rax),zmm1 MASK_KZ(2) )     // Load A with mask and zero hint
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
        vbroadcastsd( mem(r12),zmm30 )
        add( r8,rbx )                                     // b += rs_b
        vfmadd231pd( zmm3,zmm31,zmm12 )
        vfmadd231pd( zmm4,zmm31,zmm13 )
        add( r8,r12 )                                     // second pointer of b += rs_b
        vfmadd231pd( zmm3,zmm30,zmm14 )
        vfmadd231pd( zmm4,zmm30,zmm15 )

        // ---------------------------------- iteration 5
        vmovupd( mem(rax),zmm3 )                           // load A
        vmovupd( 0x40(rax),zmm4 MASK_KZ(2) )     // Load A with mask and zero hint
        add( r10,rax )                                     // a += cs_a
        prefetch( 0,mem(r15) )                             // prefetch B
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
        vbroadcastsd( mem(r12),zmm30 )
        add( r8,rbx )                                     // b += rs_b
        vfmadd231pd( zmm0,zmm31,zmm12 )
        vfmadd231pd( zmm1,zmm31,zmm13 )
        add( r8,r12 )                                     // second pointer of b += rs_b
        vfmadd231pd( zmm0,zmm30,zmm14 )
        vfmadd231pd( zmm1,zmm30,zmm15 )

        // ---------------------------------- iteration 6
        vmovupd( mem(rax),zmm0 )                           // load A
        vmovupd( 0x40(rax),zmm1 MASK_KZ(2) )     // Load A with mask and zero hint
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
        vbroadcastsd( mem(r12),zmm30 )
        add( r8,rbx )                                     // b += rs_b
        vfmadd231pd( zmm3,zmm31,zmm12 )
        vfmadd231pd( zmm4,zmm31,zmm13 )
        add( r8,r12 )                                     // second pointer of b += rs_b
        vfmadd231pd( zmm3,zmm30,zmm14 )
        vfmadd231pd( zmm4,zmm30,zmm15 )

        // ---------------------------------- iteration 7
        vmovupd( mem(rax),zmm3 )                           // load A
        vmovupd( 0x40(rax),zmm4 MASK_KZ(2) )     // Load A with mask and zero hint
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
        vbroadcastsd( mem(r12),zmm30 )
        add( r8,rbx )                                     // b += rs_b
        vfmadd231pd( zmm0,zmm31,zmm12 )
        vfmadd231pd( zmm1,zmm31,zmm13 )
        add( r8,r12 )                                     // second pointer of b += rs_b
        vfmadd231pd( zmm0,zmm30,zmm14 )
        vfmadd231pd( zmm1,zmm30,zmm15 )

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
        vbroadcastsd( mem(r12),zmm30 )
        add( r8,rbx )                                     // b += rs_b
        vfmadd231pd( zmm3,zmm31,zmm12 )
        vfmadd231pd( zmm4,zmm31,zmm13 )
        add( r8,r12 )                                     // second pointer of b += rs_b
        vfmadd231pd( zmm3,zmm30,zmm14 )
        vfmadd231pd( zmm4,zmm30,zmm15 )
        lea(mem(r11,r8,8), r11)                            // b_next += 8*rs_b
        lea(mem(r15,r8,8), r15)                            // Second pointer of b_next += 8*rs_b
        dec(rsi)                                           // i -= 1
    jnz(.LOOP3)                                            // iterate again if i != 0.


    label(.TAIL)
    mov(var(k_left), rsi)                                  // i = k_left
    test(rsi, rsi)                                         // check i via logical AND
    je(.DPOSTACCUM)                                        // if i == 0, jump to post-accumulation

    label(.DLOOPKLEFT)                                     // k_left loop
        vmovupd( mem(rax),zmm0 )                           // load A
        vmovupd( 0x40(rax),zmm1 MASK_KZ(2) )     // Load A with mask and zero hint
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
        vbroadcastsd( mem(r12),zmm30 )
        add( r8,rbx )                                     // b += rs_b
        vfmadd231pd( zmm0,zmm31,zmm12 )
        vfmadd231pd( zmm1,zmm31,zmm13 )
        add( r8,r12 )                                     // second pointer of b += rs_b
        vfmadd231pd( zmm0,zmm30,zmm14 )
        vfmadd231pd( zmm1,zmm30,zmm15 )
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
    vmulpd( zmm30,zmm14,zmm14 )
    vmulpd( zmm30,zmm15,zmm15 )


    mov(var(rs_c), rsi)                                    // load rs_c
    lea(mem(, rsi, 8), rsi)                                // rsi = rs_c * sizeof(double)
    lea(mem(rcx, rdi, 4), rdx)                             // rdx = rcx + 4 * cs_c
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
    vmovupd( 0x40(rcx),zmm1 MASK_KZ(2))        // Load C using mask and zero hint
    vfmadd231pd( zmm1,zmm31,zmm7)
    vmovupd( zmm7,0x40(rcx) MASK_(k(2)))                // store to C with mask
    vmovupd( mem(rcx,rdi,1),zmm3)
    vfmadd231pd( zmm3,zmm31,zmm8)
    vmovupd( zmm8,(rcx,rdi,1))
    vmovupd( 0x40(rcx,rdi,1),zmm4 MASK_KZ(2))        // Load C using mask and zero hint
    vfmadd231pd( zmm4,zmm31,zmm9)
    vmovupd( zmm9,0x40(rcx,rdi,1) MASK_(k(2)))                // store to C with mask
    vmovupd( mem(rcx,rdi,2),zmm0)
    vfmadd231pd( zmm0,zmm31,zmm10)
    vmovupd( zmm10,(rcx,rdi,2))
    vmovupd( 0x40(rcx,rdi,2),zmm1 MASK_KZ(2))        // Load C using mask and zero hint
    vfmadd231pd( zmm1,zmm31,zmm11)
    vmovupd( zmm11,0x40(rcx,rdi,2) MASK_(k(2)))                // store to C with mask
    vmovupd( mem(rcx,r13,1),zmm3)
    vfmadd231pd( zmm3,zmm31,zmm12)
    vmovupd( zmm12,(rcx,r13,1))
    vmovupd( 0x40(rcx,r13,1),zmm4 MASK_KZ(2))        // Load C using mask and zero hint
    vfmadd231pd( zmm4,zmm31,zmm13)
    vmovupd( zmm13,0x40(rcx,r13,1) MASK_(k(2)))                // store to C with mask
    vmovupd( mem(rdx),zmm0)
    vfmadd231pd( zmm0,zmm31,zmm14)
    vmovupd( zmm14,(rdx))
    vmovupd( 0x40(rdx),zmm1 MASK_KZ(2))        // Load C using mask and zero hint
    vfmadd231pd( zmm1,zmm31,zmm15)
    vmovupd( zmm15,0x40(rdx) MASK_(k(2)))                // store to C with mask

    jmp(.DDONE)                                           // jump to end.

    label(.DROWSTORED)
    // r12 = 3*rs_c
    lea(mem(rsi,  rsi,  2), r12)
    // r13 = 5*rs_c
    lea(mem(r12, rsi,  2), r13)
    // rdx = 7*rs_c
    lea(mem(r12, rsi,  4), rdx)
    lea(mem(   , rsi, 8), r14)
    UNPACK_LO_HIGH(8, 6, 0, 1, 12, 10, 2, 3)
    SHUFFLE_DATA(2, 0, 4, 5, 3, 1, 30, 31)

    vunpcklpd(zmm16, zmm14, zmm0)
    vunpckhpd(zmm16, zmm14, zmm1)
    SHUFFLE_DATA(2, 0, 6, 8, 3, 1, 10, 12)

    SHUFFLE_DATA(6, 4, 0, 1, 8, 5, 2, 3)
    SHUFFLE_DATA(10, 30, 4, 5, 12, 31, 6, 8)

    vbroadcastsd(mem(rax), zmm31)
    UPDATE_MASKED_C_8
    //First 8x5 tile updated

    UNPACK_LO_HIGH(9, 7, 0, 1, 13, 11, 2, 3)
    SHUFFLE_DATA(2, 0, 4, 5, 3, 1, 7, 9)

    vunpcklpd(zmm17, zmm15, zmm0)
    vunpckhpd(zmm17, zmm15, zmm1)
    SHUFFLE_DATA(2, 0, 6, 8, 3, 1, 10, 12)

    SHUFFLE_DATA(6, 4, 0, 1, 8, 5, 2, 3)
    SHUFFLE_DATA(10, 7, 4, 5, 12, 9, 6, 8)

    mov(var(m0), rdi)
    sub(imm(8), rdi)
    cmp(imm(8), rdi)
    JZ(.UPDATE8)
    cmp(imm(7), rdi)
    JZ(.UPDATE7)
    cmp(imm(6), rdi)
    JZ(.UPDATE6)
    cmp(imm(5), rdi)
    JZ(.UPDATE5)
    cmp(imm(4), rdi)
    JZ(.UPDATE4)
    cmp(imm(3), rdi)
    JZ(.UPDATE3)
    cmp(imm(2), rdi)
    JZ(.UPDATE2)
    cmp(imm(1), rdi)
    JZ(.UPDATE1)
    cmp(imm(0), rdi)
    JZ(.UPDATE0)

    LABEL(.UPDATE8)
    UPDATE_MASKED_C_8
    jmp(.DDONE)

    LABEL(.UPDATE7)
    UPDATE_MASKED_C_7
    jmp(.DDONE)

    LABEL(.UPDATE6)
    UPDATE_MASKED_C_6
    jmp(.DDONE)                                              // jump to end.

    LABEL(.UPDATE5)
    UPDATE_MASKED_C_5
    jmp(.DDONE)                                              // jump to end.

    LABEL(.UPDATE4)
    UPDATE_MASKED_C_4
    jmp(.DDONE)                                              // jump to end.

    LABEL(.UPDATE3)
    UPDATE_MASKED_C_3
    jmp(.DDONE)                                              // jump to end.

    LABEL(.UPDATE2)
    UPDATE_MASKED_C_2
    jmp(.DDONE)                                              // jump to end.

    LABEL(.UPDATE1)
    UPDATE_MASKED_C_1
    jmp(.DDONE)                                              // jump to end.

    LABEL(.UPDATE0)
    //Second 8x5 tile updated
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
    vmovupd( zmm14,(rdx))
    vmovupd( zmm15,0x40(rdx) MASK_(k(2)))                // store to C with mask

    jmp(.DDONE)                                          // jump to end.


    label(.DROWSTORBZ)
    // rdx = 3*rs_c
    lea(mem(rsi,  rsi,  2), r12)
    // rdx = 5*rs_c
    lea(mem(r12, rsi,  2), r13)
    // rdx = 7*rs_c
    lea(mem(r12, rsi,  4), rdx)
    lea(mem(   , rsi, 8), r14)
    UNPACK_LO_HIGH(8, 6, 0, 1, 12, 10, 2, 3)
    SHUFFLE_DATA(2, 0, 4, 5, 3, 1, 30, 31)

    vunpcklpd(zmm16, zmm14, zmm0)
    vunpckhpd(zmm16, zmm14, zmm1)
    SHUFFLE_DATA(2, 0, 6, 8, 3, 1, 10, 12)

    SHUFFLE_DATA(6, 4, 0, 1, 8, 5, 2, 3)
    SHUFFLE_DATA(10, 30, 4, 5, 12, 31, 6, 8)

    UPDATE_MASKED_C_8_BZ
    //First 8x5 tile updated

    UNPACK_LO_HIGH(9, 7, 0, 1, 13, 11, 2, 3)
    SHUFFLE_DATA(2, 0, 4, 5, 3, 1, 7, 9)

    vunpcklpd(zmm17, zmm15, zmm0)
    vunpckhpd(zmm17, zmm15, zmm1)
    SHUFFLE_DATA(2, 0, 6, 8, 3, 1, 10, 12)

    SHUFFLE_DATA(6, 4, 0, 1, 8, 5, 2, 3)
    SHUFFLE_DATA(10, 7, 4, 5, 12, 9, 6, 8)

    mov(var(m0), rdi)
    sub(imm(8), rdi)
    cmp(imm(8), rdi)
    JZ(.UPDATE8BZ)
    cmp(imm(7), rdi)
    JZ(.UPDATE7BZ)
    cmp(imm(6), rdi)
    JZ(.UPDATE6BZ)
    cmp(imm(5), rdi)
    JZ(.UPDATE5BZ)
    cmp(imm(4), rdi)
    JZ(.UPDATE4BZ)
    cmp(imm(3), rdi)
    JZ(.UPDATE3BZ)
    cmp(imm(2), rdi)
    JZ(.UPDATE2BZ)
    cmp(imm(1), rdi)
    JZ(.UPDATE1BZ)
    cmp(imm(0), rdi)
    JZ(.UPDATE0BZ)

    LABEL(.UPDATE8BZ)
    UPDATE_MASKED_C_8_BZ
    jmp(.DDONE)

    LABEL(.UPDATE7BZ)
    UPDATE_MASKED_C_7_BZ
    jmp(.DDONE)

    LABEL(.UPDATE6BZ)
    UPDATE_MASKED_C_6_BZ
    jmp(.DDONE)                                              // jump to end.

    LABEL(.UPDATE5BZ)
    UPDATE_MASKED_C_5_BZ
    jmp(.DDONE)                                              // jump to end.

    LABEL(.UPDATE4BZ)
    UPDATE_MASKED_C_4_BZ
    jmp(.DDONE)                                              // jump to end.

    LABEL(.UPDATE3BZ)
    UPDATE_MASKED_C_3_BZ
    jmp(.DDONE)                                              // jump to end.

    LABEL(.UPDATE2BZ)
    UPDATE_MASKED_C_2_BZ
    jmp(.DDONE)                                              // jump to end.

    LABEL(.UPDATE1BZ)
    UPDATE_MASKED_C_1_BZ
    jmp(.DDONE)                                              // jump to end.

    LABEL(.UPDATE0BZ)
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
        [mask]   "m" (mask),
        [mask_n0]   "m" (mask_n0)
      : // register clobber list
        "rax", "rbx", "rcx", "rdx", "rsi", "rdi",
        "r8", "r9", "r10", "r11", "r12", "r13", "r14", "r15",
        "xmm2", "xmm31",
        "ymm2",
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


void bli_dgemmsup_rv_zen4_asm_8x5
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

    uint8_t mask_n0 = 0xff >> (0x8 - (n0 & 7)); // calculate mask based on n_left
    // For special cases where n_left = 8, all 8 elements have to be loaded/stored
    // So, mask becomes 0xff(11111111)
    if (mask_n0 == 0) mask_n0 = 0xff;

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
    mov(var(mask_n0), rdx)          // load mask
    kmovw(edx, k(3))                // move mask to k3 register
    lea(mem(, r8, 8), r8)           // rs_b *= sizeof(double)
    lea(mem(, r9, 8), r9)           // cs_b *= sizeof(double)
    lea(mem(, r10, 8), r10)         // cs_a *= sizeof(double)
    lea(mem(, rdi, 8), rdi)         // cs_c *= sizeof(double)
    lea(mem(r9, r9, 2 ), r13)       // r13 = 3*cs_b
    // if n > 4, a second pointer(r12) which points to rbx + 4*cs_b
    //is also used to traverse B matrix
    lea(mem(rbx, r9, 4), r12)       // r12 = rbx + 4*cs_b
    lea(mem(rcx, 7*8), rdx)         // C for prefetching
    lea(mem(rbx, r8, 8, 7*8), r11)  // r11 = rbx + 8*rs_b(B for prefetching)
    // if n > 4, a second pointer which point to r11 + 4*cs_b
    //is also used to prefetch from B matrix
    lea(mem(r11, r9, 4), r15)       // r15 = r11 + 4* cs_b(B for prefetching)

    /* Register usage: zmm0-5 are used to load A matrix
     *                 zmm6-29 are used for accumulation
     *                 zmm30-31 are used for broadcasting B matrix
     */

    // zero out all accumulation registers
    vxorpd(zmm6, zmm6, zmm6)
    vxorpd(zmm8, zmm8, zmm8)
    vxorpd(zmm10, zmm10, zmm10)
    vxorpd(zmm12, zmm12, zmm12)
    vxorpd(zmm14, zmm14, zmm14)

    // K is unrolled by 8 to facilitate prefetch of B
    // Assuming B to be col-stored, for each iteration of K,
    //one cacheline of B_next is prefetched where b_next = b + (unroll)*rs_b
    label(.DLOOPKITER)                                     // main loop
    mov(var(k_iter), rsi)                                  // i = k_iter
    sub(imm( 5+TAIL_NITER), rsi)                           // i -= NR + TAIL_NITER
    jle(.PREFETCHLOOP)                                     // jump if i <= 0

    label(.LOOP1)

        // ---------------------------------- iteration 1

        vmovupd( mem(rax),zmm0 MASK_KZ(2) )                           // load A     // Load A with mask and zero hint
        add( r10,rax )                                     // a += cs_a
        vmovupd( mem(rax),zmm3 MASK_KZ(2) )                           // load A     // Load A with mask and zero hint
        add( r10,rax )                                     // a += cs_a
        prefetch( 0,mem(r11) )                             // prefetch B
        vbroadcastsd( mem(rbx),zmm30 )
        vbroadcastsd( mem(rbx,r9,1),zmm31 )
        vfmadd231pd( zmm0,zmm30,zmm6 )
        vbroadcastsd( mem(rbx,r9,2),zmm30 )
        vfmadd231pd( zmm0,zmm31,zmm8 )
        vbroadcastsd( mem(rbx,r13,1),zmm31 )
        vfmadd231pd( zmm0,zmm30,zmm10 )
        vbroadcastsd( mem(r12),zmm30 )
        add( r8,rbx )                                     // b += rs_b
        vfmadd231pd( zmm0,zmm31,zmm12 )
        add( r8,r12 )                                     // second pointer of b += rs_b
        vfmadd231pd( zmm0,zmm30,zmm14 )

        // ---------------------------------- iteration 2

        vmovupd( mem(rax),zmm0 MASK_KZ(2) )                           // load A     // Load A with mask and zero hint
        add( r10,rax )                                     // a += cs_a
        prefetch( 0,mem(r11,r9,1) )                             // prefetch B
        vbroadcastsd( mem(rbx),zmm30 )
        vbroadcastsd( mem(rbx,r9,1),zmm31 )
        vfmadd231pd( zmm3,zmm30,zmm6 )
        vbroadcastsd( mem(rbx,r9,2),zmm30 )
        vfmadd231pd( zmm3,zmm31,zmm8 )
        vbroadcastsd( mem(rbx,r13,1),zmm31 )
        vfmadd231pd( zmm3,zmm30,zmm10 )
        vbroadcastsd( mem(r12),zmm30 )
        add( r8,rbx )                                     // b += rs_b
        vfmadd231pd( zmm3,zmm31,zmm12 )
        add( r8,r12 )                                     // second pointer of b += rs_b
        vfmadd231pd( zmm3,zmm30,zmm14 )

        // ---------------------------------- iteration 3

        vmovupd( mem(rax),zmm3 MASK_KZ(2) )                           // load A     // Load A with mask and zero hint
        add( r10,rax )                                     // a += cs_a
        prefetch( 0,mem(r11,r9,2) )                             // prefetch B
        vbroadcastsd( mem(rbx),zmm30 )
        vbroadcastsd( mem(rbx,r9,1),zmm31 )
        vfmadd231pd( zmm0,zmm30,zmm6 )
        vbroadcastsd( mem(rbx,r9,2),zmm30 )
        vfmadd231pd( zmm0,zmm31,zmm8 )
        vbroadcastsd( mem(rbx,r13,1),zmm31 )
        vfmadd231pd( zmm0,zmm30,zmm10 )
        vbroadcastsd( mem(r12),zmm30 )
        add( r8,rbx )                                     // b += rs_b
        vfmadd231pd( zmm0,zmm31,zmm12 )
        add( r8,r12 )                                     // second pointer of b += rs_b
        vfmadd231pd( zmm0,zmm30,zmm14 )

        // ---------------------------------- iteration 4

        vmovupd( mem(rax),zmm0 MASK_KZ(2) )                           // load A     // Load A with mask and zero hint
        add( r10,rax )                                     // a += cs_a
        prefetch( 0,mem(r11,r13,1) )                             // prefetch B
        vbroadcastsd( mem(rbx),zmm30 )
        vbroadcastsd( mem(rbx,r9,1),zmm31 )
        vfmadd231pd( zmm3,zmm30,zmm6 )
        vbroadcastsd( mem(rbx,r9,2),zmm30 )
        vfmadd231pd( zmm3,zmm31,zmm8 )
        vbroadcastsd( mem(rbx,r13,1),zmm31 )
        vfmadd231pd( zmm3,zmm30,zmm10 )
        vbroadcastsd( mem(r12),zmm30 )
        add( r8,rbx )                                     // b += rs_b
        vfmadd231pd( zmm3,zmm31,zmm12 )
        add( r8,r12 )                                     // second pointer of b += rs_b
        vfmadd231pd( zmm3,zmm30,zmm14 )

        // ---------------------------------- iteration 5

        vmovupd( mem(rax),zmm3 MASK_KZ(2) )                           // load A     // Load A with mask and zero hint
        add( r10,rax )                                     // a += cs_a
        prefetch( 0,mem(r15) )                             // prefetch B
        vbroadcastsd( mem(rbx),zmm30 )
        vbroadcastsd( mem(rbx,r9,1),zmm31 )
        vfmadd231pd( zmm0,zmm30,zmm6 )
        vbroadcastsd( mem(rbx,r9,2),zmm30 )
        vfmadd231pd( zmm0,zmm31,zmm8 )
        vbroadcastsd( mem(rbx,r13,1),zmm31 )
        vfmadd231pd( zmm0,zmm30,zmm10 )
        vbroadcastsd( mem(r12),zmm30 )
        add( r8,rbx )                                     // b += rs_b
        vfmadd231pd( zmm0,zmm31,zmm12 )
        add( r8,r12 )                                     // second pointer of b += rs_b
        vfmadd231pd( zmm0,zmm30,zmm14 )

        // ---------------------------------- iteration 6

        vmovupd( mem(rax),zmm0 MASK_KZ(2) )                           // load A     // Load A with mask and zero hint
        add( r10,rax )                                     // a += cs_a
        vbroadcastsd( mem(rbx),zmm30 )
        vbroadcastsd( mem(rbx,r9,1),zmm31 )
        vfmadd231pd( zmm3,zmm30,zmm6 )
        vbroadcastsd( mem(rbx,r9,2),zmm30 )
        vfmadd231pd( zmm3,zmm31,zmm8 )
        vbroadcastsd( mem(rbx,r13,1),zmm31 )
        vfmadd231pd( zmm3,zmm30,zmm10 )
        vbroadcastsd( mem(r12),zmm30 )
        add( r8,rbx )                                     // b += rs_b
        vfmadd231pd( zmm3,zmm31,zmm12 )
        add( r8,r12 )                                     // second pointer of b += rs_b
        vfmadd231pd( zmm3,zmm30,zmm14 )

        // ---------------------------------- iteration 7

        vmovupd( mem(rax),zmm3 MASK_KZ(2) )                           // load A     // Load A with mask and zero hint
        add( r10,rax )                                     // a += cs_a
        vbroadcastsd( mem(rbx),zmm30 )
        vbroadcastsd( mem(rbx,r9,1),zmm31 )
        vfmadd231pd( zmm0,zmm30,zmm6 )
        vbroadcastsd( mem(rbx,r9,2),zmm30 )
        vfmadd231pd( zmm0,zmm31,zmm8 )
        vbroadcastsd( mem(rbx,r13,1),zmm31 )
        vfmadd231pd( zmm0,zmm30,zmm10 )
        vbroadcastsd( mem(r12),zmm30 )
        add( r8,rbx )                                     // b += rs_b
        vfmadd231pd( zmm0,zmm31,zmm12 )
        add( r8,r12 )                                     // second pointer of b += rs_b
        vfmadd231pd( zmm0,zmm30,zmm14 )

        // ---------------------------------- iteration 8

        vbroadcastsd( mem(rbx),zmm30 )
        vbroadcastsd( mem(rbx,r9,1),zmm31 )
        vfmadd231pd( zmm3,zmm30,zmm6 )
        vbroadcastsd( mem(rbx,r9,2),zmm30 )
        vfmadd231pd( zmm3,zmm31,zmm8 )
        vbroadcastsd( mem(rbx,r13,1),zmm31 )
        vfmadd231pd( zmm3,zmm30,zmm10 )
        vbroadcastsd( mem(r12),zmm30 )
        add( r8,rbx )                                     // b += rs_b
        vfmadd231pd( zmm3,zmm31,zmm12 )
        add( r8,r12 )                                     // second pointer of b += rs_b
        vfmadd231pd( zmm3,zmm30,zmm14 )
        lea(mem(r11,r8,8), r11)                            // b_next += 8*rs_b
        lea(mem(r15,r8,8), r15)                            // second pointer to b_next += 8*rs_b
        dec(rsi)                                           // i -= 1
    jnz(.LOOP1)                                            // iterate again if i != 0.

    label(.PREFETCHLOOP)
    add(imm(5), rsi)                                       // i += NR
    jle(.TAILITER)                                         // jump if i <= 0.

    label(.LOOP2)

        // ---------------------------------- iteration 1
        prefetchw0( mem(rdx))                              // prefetch C
        vmovupd( mem(rax),zmm0 MASK_KZ(2) )                           // load A     // Load A with mask and zero hint
        add( r10,rax )                                     // a += cs_a
        vmovupd( mem(rax),zmm3 MASK_KZ(2) )                           // load A     // Load A with mask and zero hint
        add( r10,rax )                                     // a += cs_a
        prefetch( 0,mem(r11) )                             // prefetch B
        vbroadcastsd( mem(rbx),zmm30 )
        vbroadcastsd( mem(rbx,r9,1),zmm31 )
        vfmadd231pd( zmm0,zmm30,zmm6 )
        vbroadcastsd( mem(rbx,r9,2),zmm30 )
        vfmadd231pd( zmm0,zmm31,zmm8 )
        vbroadcastsd( mem(rbx,r13,1),zmm31 )
        vfmadd231pd( zmm0,zmm30,zmm10 )
        vbroadcastsd( mem(r12),zmm30 )
        add( r8,rbx )                                     // b += rs_b
        vfmadd231pd( zmm0,zmm31,zmm12 )
        add( r8,r12 )                                     // second pointer of b += rs_b
        vfmadd231pd( zmm0,zmm30,zmm14 )

        // ---------------------------------- iteration 2
        vmovupd( mem(rax),zmm0 MASK_KZ(2) )                           // load A     // Load A with mask and zero hint
        add( r10,rax )                                     // a += cs_a
        prefetch( 0,mem(r11,r9,1) )                             // prefetch B
        vbroadcastsd( mem(rbx),zmm30 )
        vbroadcastsd( mem(rbx,r9,1),zmm31 )
        vfmadd231pd( zmm3,zmm30,zmm6 )
        vbroadcastsd( mem(rbx,r9,2),zmm30 )
        vfmadd231pd( zmm3,zmm31,zmm8 )
        vbroadcastsd( mem(rbx,r13,1),zmm31 )
        vfmadd231pd( zmm3,zmm30,zmm10 )
        vbroadcastsd( mem(r12),zmm30 )
        add( r8,rbx )                                     // b += rs_b
        vfmadd231pd( zmm3,zmm31,zmm12 )
        add( r8,r12 )                                     // second pointer of b += rs_b
        vfmadd231pd( zmm3,zmm30,zmm14 )

        // ---------------------------------- iteration 3
        vmovupd( mem(rax),zmm3 MASK_KZ(2) )                           // load A     // Load A with mask and zero hint
        add( r10,rax )                                     // a += cs_a
        prefetch( 0,mem(r11,r9,2) )                             // prefetch B
        vbroadcastsd( mem(rbx),zmm30 )
        vbroadcastsd( mem(rbx,r9,1),zmm31 )
        vfmadd231pd( zmm0,zmm30,zmm6 )
        vbroadcastsd( mem(rbx,r9,2),zmm30 )
        vfmadd231pd( zmm0,zmm31,zmm8 )
        vbroadcastsd( mem(rbx,r13,1),zmm31 )
        vfmadd231pd( zmm0,zmm30,zmm10 )
        vbroadcastsd( mem(r12),zmm30 )
        add( r8,rbx )                                     // b += rs_b
        vfmadd231pd( zmm0,zmm31,zmm12 )
        add( r8,r12 )                                     // second pointer of b += rs_b
        vfmadd231pd( zmm0,zmm30,zmm14 )

        // ---------------------------------- iteration 4
        vmovupd( mem(rax),zmm0 MASK_KZ(2) )                           // load A     // Load A with mask and zero hint
        add( r10,rax )                                     // a += cs_a
        prefetch( 0,mem(r11,r13,1) )                             // prefetch B
        vbroadcastsd( mem(rbx),zmm30 )
        vbroadcastsd( mem(rbx,r9,1),zmm31 )
        vfmadd231pd( zmm3,zmm30,zmm6 )
        vbroadcastsd( mem(rbx,r9,2),zmm30 )
        vfmadd231pd( zmm3,zmm31,zmm8 )
        vbroadcastsd( mem(rbx,r13,1),zmm31 )
        vfmadd231pd( zmm3,zmm30,zmm10 )
        vbroadcastsd( mem(r12),zmm30 )
        add( r8,rbx )                                     // b += rs_b
        vfmadd231pd( zmm3,zmm31,zmm12 )
        add( r8,r12 )                                     // second pointer of b += rs_b
        vfmadd231pd( zmm3,zmm30,zmm14 )

        // ---------------------------------- iteration 5
        vmovupd( mem(rax),zmm3 MASK_KZ(2) )                           // load A     // Load A with mask and zero hint
        add( r10,rax )                                     // a += cs_a
        prefetch( 0,mem(r15) )                             // prefetch B
        vbroadcastsd( mem(rbx),zmm30 )
        vbroadcastsd( mem(rbx,r9,1),zmm31 )
        vfmadd231pd( zmm0,zmm30,zmm6 )
        vbroadcastsd( mem(rbx,r9,2),zmm30 )
        vfmadd231pd( zmm0,zmm31,zmm8 )
        vbroadcastsd( mem(rbx,r13,1),zmm31 )
        vfmadd231pd( zmm0,zmm30,zmm10 )
        vbroadcastsd( mem(r12),zmm30 )
        add( r8,rbx )                                     // b += rs_b
        vfmadd231pd( zmm0,zmm31,zmm12 )
        add( r8,r12 )                                     // second pointer of b += rs_b
        vfmadd231pd( zmm0,zmm30,zmm14 )

        // ---------------------------------- iteration 6
        vmovupd( mem(rax),zmm0 MASK_KZ(2) )                           // load A     // Load A with mask and zero hint
        add( r10,rax )                                     // a += cs_a
        vbroadcastsd( mem(rbx),zmm30 )
        vbroadcastsd( mem(rbx,r9,1),zmm31 )
        vfmadd231pd( zmm3,zmm30,zmm6 )
        vbroadcastsd( mem(rbx,r9,2),zmm30 )
        vfmadd231pd( zmm3,zmm31,zmm8 )
        vbroadcastsd( mem(rbx,r13,1),zmm31 )
        vfmadd231pd( zmm3,zmm30,zmm10 )
        vbroadcastsd( mem(r12),zmm30 )
        add( r8,rbx )                                     // b += rs_b
        vfmadd231pd( zmm3,zmm31,zmm12 )
        add( r8,r12 )                                     // second pointer of b += rs_b
        vfmadd231pd( zmm3,zmm30,zmm14 )

        // ---------------------------------- iteration 7
        vmovupd( mem(rax),zmm3 MASK_KZ(2) )                           // load A     // Load A with mask and zero hint
        add( r10,rax )                                     // a += cs_a
        vbroadcastsd( mem(rbx),zmm30 )
        vbroadcastsd( mem(rbx,r9,1),zmm31 )
        vfmadd231pd( zmm0,zmm30,zmm6 )
        vbroadcastsd( mem(rbx,r9,2),zmm30 )
        vfmadd231pd( zmm0,zmm31,zmm8 )
        vbroadcastsd( mem(rbx,r13,1),zmm31 )
        vfmadd231pd( zmm0,zmm30,zmm10 )
        vbroadcastsd( mem(r12),zmm30 )
        add( r8,rbx )                                     // b += rs_b
        vfmadd231pd( zmm0,zmm31,zmm12 )
        add( r8,r12 )                                     // second pointer of b += rs_b
        vfmadd231pd( zmm0,zmm30,zmm14 )

        // ---------------------------------- iteration 8
        vbroadcastsd( mem(rbx),zmm30 )
        vbroadcastsd( mem(rbx,r9,1),zmm31 )
        vfmadd231pd( zmm3,zmm30,zmm6 )
        vbroadcastsd( mem(rbx,r9,2),zmm30 )
        vfmadd231pd( zmm3,zmm31,zmm8 )
        vbroadcastsd( mem(rbx,r13,1),zmm31 )
        vfmadd231pd( zmm3,zmm30,zmm10 )
        vbroadcastsd( mem(r12),zmm30 )
        add( r8,rbx )                                     // b += rs_b
        vfmadd231pd( zmm3,zmm31,zmm12 )
        add( r8,r12 )                                     // second pointer of b += rs_b
        vfmadd231pd( zmm3,zmm30,zmm14 )
        lea(mem(rdx, rdi, 1), rdx)                         // C += cs_c
        lea(mem(r11,r8,8), r11)                            // b_next += 8*rs_b
        lea(mem(r15,r8,8), r15)                            // second pointer of b_next += 8*rs_b
        sub(imm(1), rsi)                                   // i -= 1
    jnz(.LOOP2)                                            // iterate again if i != 0.
    label(.TAILITER)
    add(imm(TAIL_NITER), rsi)                              // i += TAIL_NITER
    jle(.TAIL)                                             // jump if i <= 0

    label(.LOOP3)

        // ---------------------------------- iteration 1
        vmovupd( mem(rax),zmm0 MASK_KZ(2) )                           // load A     // Load A with mask and zero hint
        add( r10,rax )                                     // a += cs_a
        vmovupd( mem(rax),zmm3 MASK_KZ(2) )                           // load A     // Load A with mask and zero hint
        add( r10,rax )                                     // a += cs_a
        prefetch( 0,mem(r11) )                             // prefetch B
        vbroadcastsd( mem(rbx),zmm30 )
        vbroadcastsd( mem(rbx,r9,1),zmm31 )
        vfmadd231pd( zmm0,zmm30,zmm6 )
        vbroadcastsd( mem(rbx,r9,2),zmm30 )
        vfmadd231pd( zmm0,zmm31,zmm8 )
        vbroadcastsd( mem(rbx,r13,1),zmm31 )
        vfmadd231pd( zmm0,zmm30,zmm10 )
        vbroadcastsd( mem(r12),zmm30 )
        add( r8,rbx )                                     // b += rs_b
        vfmadd231pd( zmm0,zmm31,zmm12 )
        add( r8,r12 )                                     // second pointer of b += rs_b
        vfmadd231pd( zmm0,zmm30,zmm14 )

        // ---------------------------------- iteration 2
        vmovupd( mem(rax),zmm0 MASK_KZ(2) )                           // load A     // Load A with mask and zero hint
        add( r10,rax )                                     // a += cs_a
        prefetch( 0,mem(r11,r9,1) )                             // prefetch B
        vbroadcastsd( mem(rbx),zmm30 )
        vbroadcastsd( mem(rbx,r9,1),zmm31 )
        vfmadd231pd( zmm3,zmm30,zmm6 )
        vbroadcastsd( mem(rbx,r9,2),zmm30 )
        vfmadd231pd( zmm3,zmm31,zmm8 )
        vbroadcastsd( mem(rbx,r13,1),zmm31 )
        vfmadd231pd( zmm3,zmm30,zmm10 )
        vbroadcastsd( mem(r12),zmm30 )
        add( r8,rbx )                                     // b += rs_b
        vfmadd231pd( zmm3,zmm31,zmm12 )
        add( r8,r12 )                                     // second pointer of b += rs_b
        vfmadd231pd( zmm3,zmm30,zmm14 )

        // ---------------------------------- iteration 3
        vmovupd( mem(rax),zmm3 MASK_KZ(2) )                           // load A     // Load A with mask and zero hint
        add( r10,rax )                                     // a += cs_a
        prefetch( 0,mem(r11,r9,2) )                             // prefetch B
        vbroadcastsd( mem(rbx),zmm30 )
        vbroadcastsd( mem(rbx,r9,1),zmm31 )
        vfmadd231pd( zmm0,zmm30,zmm6 )
        vbroadcastsd( mem(rbx,r9,2),zmm30 )
        vfmadd231pd( zmm0,zmm31,zmm8 )
        vbroadcastsd( mem(rbx,r13,1),zmm31 )
        vfmadd231pd( zmm0,zmm30,zmm10 )
        vbroadcastsd( mem(r12),zmm30 )
        add( r8,rbx )                                     // b += rs_b
        vfmadd231pd( zmm0,zmm31,zmm12 )
        add( r8,r12 )                                     // second pointer of b += rs_b
        vfmadd231pd( zmm0,zmm30,zmm14 )

        // ---------------------------------- iteration 4
        vmovupd( mem(rax),zmm0 MASK_KZ(2) )                           // load A     // Load A with mask and zero hint
        add( r10,rax )                                     // a += cs_a
        prefetch( 0,mem(r11,r13,1) )                             // prefetch B
        vbroadcastsd( mem(rbx),zmm30 )
        vbroadcastsd( mem(rbx,r9,1),zmm31 )
        vfmadd231pd( zmm3,zmm30,zmm6 )
        vbroadcastsd( mem(rbx,r9,2),zmm30 )
        vfmadd231pd( zmm3,zmm31,zmm8 )
        vbroadcastsd( mem(rbx,r13,1),zmm31 )
        vfmadd231pd( zmm3,zmm30,zmm10 )
        vbroadcastsd( mem(r12),zmm30 )
        add( r8,rbx )                                     // b += rs_b
        vfmadd231pd( zmm3,zmm31,zmm12 )
        add( r8,r12 )                                     // second pointer of b += rs_b
        vfmadd231pd( zmm3,zmm30,zmm14 )

        // ---------------------------------- iteration 5
        vmovupd( mem(rax),zmm3 MASK_KZ(2) )                           // load A     // Load A with mask and zero hint
        add( r10,rax )                                     // a += cs_a
        prefetch( 0,mem(r15) )                             // prefetch B
        vbroadcastsd( mem(rbx),zmm30 )
        vbroadcastsd( mem(rbx,r9,1),zmm31 )
        vfmadd231pd( zmm0,zmm30,zmm6 )
        vbroadcastsd( mem(rbx,r9,2),zmm30 )
        vfmadd231pd( zmm0,zmm31,zmm8 )
        vbroadcastsd( mem(rbx,r13,1),zmm31 )
        vfmadd231pd( zmm0,zmm30,zmm10 )
        vbroadcastsd( mem(r12),zmm30 )
        add( r8,rbx )                                     // b += rs_b
        vfmadd231pd( zmm0,zmm31,zmm12 )
        add( r8,r12 )                                     // second pointer of b += rs_b
        vfmadd231pd( zmm0,zmm30,zmm14 )

        // ---------------------------------- iteration 6
        vmovupd( mem(rax),zmm0 MASK_KZ(2) )                           // load A     // Load A with mask and zero hint
        add( r10,rax )                                     // a += cs_a
        vbroadcastsd( mem(rbx),zmm30 )
        vbroadcastsd( mem(rbx,r9,1),zmm31 )
        vfmadd231pd( zmm3,zmm30,zmm6 )
        vbroadcastsd( mem(rbx,r9,2),zmm30 )
        vfmadd231pd( zmm3,zmm31,zmm8 )
        vbroadcastsd( mem(rbx,r13,1),zmm31 )
        vfmadd231pd( zmm3,zmm30,zmm10 )
        vbroadcastsd( mem(r12),zmm30 )
        add( r8,rbx )                                     // b += rs_b
        vfmadd231pd( zmm3,zmm31,zmm12 )
        add( r8,r12 )                                     // second pointer of b += rs_b
        vfmadd231pd( zmm3,zmm30,zmm14 )

        // ---------------------------------- iteration 7
        vmovupd( mem(rax),zmm3 MASK_KZ(2) )                           // load A     // Load A with mask and zero hint
        add( r10,rax )                                     // a += cs_a
        vbroadcastsd( mem(rbx),zmm30 )
        vbroadcastsd( mem(rbx,r9,1),zmm31 )
        vfmadd231pd( zmm0,zmm30,zmm6 )
        vbroadcastsd( mem(rbx,r9,2),zmm30 )
        vfmadd231pd( zmm0,zmm31,zmm8 )
        vbroadcastsd( mem(rbx,r13,1),zmm31 )
        vfmadd231pd( zmm0,zmm30,zmm10 )
        vbroadcastsd( mem(r12),zmm30 )
        add( r8,rbx )                                     // b += rs_b
        vfmadd231pd( zmm0,zmm31,zmm12 )
        add( r8,r12 )                                     // second pointer of b += rs_b
        vfmadd231pd( zmm0,zmm30,zmm14 )

        // ---------------------------------- iteration 8
        vbroadcastsd( mem(rbx),zmm30 )
        vbroadcastsd( mem(rbx,r9,1),zmm31 )
        vfmadd231pd( zmm3,zmm30,zmm6 )
        vbroadcastsd( mem(rbx,r9,2),zmm30 )
        vfmadd231pd( zmm3,zmm31,zmm8 )
        vbroadcastsd( mem(rbx,r13,1),zmm31 )
        vfmadd231pd( zmm3,zmm30,zmm10 )
        vbroadcastsd( mem(r12),zmm30 )
        add( r8,rbx )                                     // b += rs_b
        vfmadd231pd( zmm3,zmm31,zmm12 )
        add( r8,r12 )                                     // second pointer of b += rs_b
        vfmadd231pd( zmm3,zmm30,zmm14 )
        lea(mem(r11,r8,8), r11)                            // b_next += 8*rs_b
        lea(mem(r15,r8,8), r15)                            // Second pointer of b_next += 8*rs_b
        dec(rsi)                                           // i -= 1
    jnz(.LOOP3)                                            // iterate again if i != 0.


    label(.TAIL)
    mov(var(k_left), rsi)                                  // i = k_left
    test(rsi, rsi)                                         // check i via logical AND
    je(.DPOSTACCUM)                                        // if i == 0, jump to post-accumulation

    label(.DLOOPKLEFT)                                     // k_left loop
        vmovupd( mem(rax),zmm0 MASK_KZ(2) )                           // load A     // Load A with mask and zero hint
        add( r10,rax )                                     // a += cs_a
        vbroadcastsd( mem(rbx),zmm30 )
        vbroadcastsd( mem(rbx,r9,1),zmm31 )
        vfmadd231pd( zmm0,zmm30,zmm6 )
        vbroadcastsd( mem(rbx,r9,2),zmm30 )
        vfmadd231pd( zmm0,zmm31,zmm8 )
        vbroadcastsd( mem(rbx,r13,1),zmm31 )
        vfmadd231pd( zmm0,zmm30,zmm10 )
        vbroadcastsd( mem(r12),zmm30 )
        add( r8,rbx )                                     // b += rs_b
        vfmadd231pd( zmm0,zmm31,zmm12 )
        add( r8,r12 )                                     // second pointer of b += rs_b
        vfmadd231pd( zmm0,zmm30,zmm14 )
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
    vmulpd( zmm30,zmm14,zmm14 )


    mov(var(rs_c), rsi)                                    // load rs_c
    lea(mem(, rsi, 8), rsi)                                // rsi = rs_c * sizeof(double)
    lea(mem(rcx, rdi, 4), rdx)                             // rdx = rcx + 4 * cs_c
    lea(mem(rdi, rdi, 2), r13)                             // r13 = 3*cs_c
    vxorpd(ymm2, ymm2, ymm2)
    vucomisd(xmm2, xmm31)                                   // set ZF if beta == 0
    je(.DBETAZERO)                                         // if ZF == 1, jump to beta == 0 case


    cmp(imm(8), rdi)                                       // set ZF if (8*cs_c) == 8


    jz(.DROWSTORED)                                        // jump to row storage case

    label(.DCOLSTORED)
    vmovupd( mem(rcx),zmm0 MASK_KZ(2))        // Load C using mask and zero hint
    vfmadd231pd( zmm0,zmm31,zmm6)
    vmovupd( zmm6,(rcx) MASK_(k(2)))                // store to C with mask
    vmovupd( mem(rcx,rdi,1),zmm3 MASK_KZ(2))        // Load C using mask and zero hint
    vfmadd231pd( zmm3,zmm31,zmm8)
    vmovupd( zmm8,(rcx,rdi,1) MASK_(k(2)))                // store to C with mask
    vmovupd( mem(rcx,rdi,2),zmm0 MASK_KZ(2))        // Load C using mask and zero hint
    vfmadd231pd( zmm0,zmm31,zmm10)
    vmovupd( zmm10,(rcx,rdi,2) MASK_(k(2)))                // store to C with mask
    vmovupd( mem(rcx,r13,1),zmm3 MASK_KZ(2))        // Load C using mask and zero hint
    vfmadd231pd( zmm3,zmm31,zmm12)
    vmovupd( zmm12,(rcx,r13,1) MASK_(k(2)))                // store to C with mask
    vmovupd( mem(rdx),zmm0 MASK_KZ(2))        // Load C using mask and zero hint
    vfmadd231pd( zmm0,zmm31,zmm14)
    vmovupd( zmm14,(rdx) MASK_(k(2)))                // store to C with mask

    jmp(.DDONE)                                           // jump to end.

    label(.DROWSTORED)
    // rdx = 3*rs_c
    lea(mem(rsi,  rsi,  2), r12)
    // rdx = 5*rs_c
    lea(mem(r12, rsi,  2), r13)
    // rdx = 7*rs_c
    lea(mem(r12, rsi,  4), rdx)
    lea(mem(   , rsi, 8), r14)
    UNPACK_LO_HIGH(8, 6, 0, 1, 12, 10, 2, 3)
    SHUFFLE_DATA(2, 0, 4, 5, 3, 1, 30, 31)

    vunpcklpd(zmm16, zmm14, zmm0)
    vunpckhpd(zmm16, zmm14, zmm1)
    SHUFFLE_DATA(2, 0, 6, 8, 3, 1, 10, 12)

    SHUFFLE_DATA(6, 4, 0, 1, 8, 5, 2, 3)
    SHUFFLE_DATA(10, 30, 4, 5, 12, 31, 6, 8)

    vbroadcastsd(mem(rax), zmm31)

    mov(var(m0), rdi)
    cmp(imm(8), rdi)
    JZ(.UPDATE8)
    cmp(imm(7), rdi)
    JZ(.UPDATE7)
    cmp(imm(6), rdi)
    JZ(.UPDATE6)
    cmp(imm(5), rdi)
    JZ(.UPDATE5)
    cmp(imm(4), rdi)
    JZ(.UPDATE4)
    cmp(imm(3), rdi)
    JZ(.UPDATE3)
    cmp(imm(2), rdi)
    JZ(.UPDATE2)
    cmp(imm(1), rdi)
    JZ(.UPDATE1)
    cmp(imm(0), rdi)
    JZ(.UPDATE0)

    LABEL(.UPDATE8)
    UPDATE_MASKED_C_8
    jmp(.DDONE)

    LABEL(.UPDATE7)
    UPDATE_MASKED_C_7
    jmp(.DDONE)

    LABEL(.UPDATE6)
    UPDATE_MASKED_C_6
    jmp(.DDONE)                                              // jump to end.

    LABEL(.UPDATE5)
    UPDATE_MASKED_C_5
    jmp(.DDONE)                                              // jump to end.

    LABEL(.UPDATE4)
    UPDATE_MASKED_C_4
    jmp(.DDONE)                                              // jump to end.

    LABEL(.UPDATE3)
    UPDATE_MASKED_C_3
    jmp(.DDONE)                                              // jump to end.

    LABEL(.UPDATE2)
    UPDATE_MASKED_C_2
    jmp(.DDONE)                                              // jump to end.

    LABEL(.UPDATE1)
    UPDATE_MASKED_C_1
    jmp(.DDONE)                                              // jump to end.

    LABEL(.UPDATE0)
    //8x5 tile updated
    jmp(.DDONE)                                          // jump to end.


    label(.DBETAZERO)
    cmp(imm(8), rdi)                                     // set ZF if (8*cs_c) == 8

    jz(.DROWSTORBZ)                                      // jump to row storage case
    label(.DCOLSTORBZ)
    vmovupd( zmm6,(rcx) MASK_(k(2)))                // store to C with mask
    vmovupd( zmm8,(rcx,rdi,1) MASK_(k(2)))                // store to C with mask
    vmovupd( zmm10,(rcx,rdi,2) MASK_(k(2)))                // store to C with mask
    vmovupd( zmm12,(rcx,r13,1) MASK_(k(2)))                // store to C with mask
    vmovupd( zmm14,(rdx) MASK_(k(2)))                // store to C with mask

    jmp(.DDONE)                                          // jump to end.


    label(.DROWSTORBZ)
    // rdx = 3*rs_c
    lea(mem(rsi,  rsi,  2), r12)
    // rdx = 5*rs_c
    lea(mem(r12, rsi,  2), r13)
    // rdx = 7*rs_c
    lea(mem(r12, rsi,  4), rdx)
    lea(mem(   , rsi, 8), r14)
    UNPACK_LO_HIGH(8, 6, 0, 1, 12, 10, 2, 3)
    SHUFFLE_DATA(2, 0, 4, 5, 3, 1, 30, 31)

    vunpcklpd(zmm16, zmm14, zmm0)
    vunpckhpd(zmm16, zmm14, zmm1)
    SHUFFLE_DATA(2, 0, 6, 8, 3, 1, 10, 12)

    SHUFFLE_DATA(6, 4, 0, 1, 8, 5, 2, 3)
    SHUFFLE_DATA(10, 30, 4, 5, 12, 31, 6, 8)

    mov(var(m0), rdi)
    cmp(imm(8), rdi)
    JZ(.UPDATE8BZ)
    cmp(imm(7), rdi)
    JZ(.UPDATE7BZ)
    cmp(imm(6), rdi)
    JZ(.UPDATE6BZ)
    cmp(imm(5), rdi)
    JZ(.UPDATE5BZ)
    cmp(imm(4), rdi)
    JZ(.UPDATE4BZ)
    cmp(imm(3), rdi)
    JZ(.UPDATE3BZ)
    cmp(imm(2), rdi)
    JZ(.UPDATE2BZ)
    cmp(imm(1), rdi)
    JZ(.UPDATE1BZ)
    cmp(imm(0), rdi)
    JZ(.UPDATE0BZ)

    LABEL(.UPDATE8BZ)
    UPDATE_MASKED_C_8_BZ
    jmp(.DDONE)

    LABEL(.UPDATE7BZ)
    UPDATE_MASKED_C_7_BZ
    jmp(.DDONE)

    LABEL(.UPDATE6BZ)
    UPDATE_MASKED_C_6_BZ
    jmp(.DDONE)                                              // jump to end.

    LABEL(.UPDATE5BZ)
    UPDATE_MASKED_C_5_BZ
    jmp(.DDONE)                                              // jump to end.

    LABEL(.UPDATE4BZ)
    UPDATE_MASKED_C_4_BZ
    jmp(.DDONE)                                              // jump to end.

    LABEL(.UPDATE3BZ)
    UPDATE_MASKED_C_3_BZ
    jmp(.DDONE)                                              // jump to end.

    LABEL(.UPDATE2BZ)
    UPDATE_MASKED_C_2_BZ
    jmp(.DDONE)                                              // jump to end.

    LABEL(.UPDATE1BZ)
    UPDATE_MASKED_C_1_BZ
    jmp(.DDONE)                                              // jump to end.

    LABEL(.UPDATE0BZ)
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
        [mask]   "m" (mask),
        [mask_n0]   "m" (mask_n0)
      : // register clobber list
        "rax", "rbx", "rcx", "rdx", "rsi", "rdi",
        "r8", "r9", "r10", "r11", "r12", "r13", "r14", "r15",
        "xmm2", "xmm31",
        "ymm2",
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
