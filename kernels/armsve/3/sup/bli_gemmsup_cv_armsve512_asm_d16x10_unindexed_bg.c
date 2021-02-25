/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2014, The University of Texas at Austin
   Copyright (C) 2020, Dept. Physics, The University of Tokyo

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
#include <assert.h>
#include "blis.h"

#define DGEMM_FMLA2(CCOLFH,CCOLLH,ACOLFH,ACOLLH,BV) \
" fmla     "#CCOLFH".d, p0/m, "#ACOLFH".d, "#BV".d\n\t" /* A Row 1:8  */ \
" fmla     "#CCOLLH".d, p0/m, "#ACOLLH".d, "#BV".d\n\t" /* A Row 9:15 */

#define DGEMM_FMLA2_LD1RD(CCOLFH,CCOLLH,ACOLFH,ACOLLH,BV,ADDRB,SHIFT) \
  DGEMM_FMLA2(CCOLFH,CCOLLH,ACOLFH,ACOLLH,BV) \
" ld1rd       "#BV".d, p0/z, ["#ADDRB", #"#SHIFT"]\n\t" /* Next B     */

#define DGEMM_FMLA2_LD1RD_FIRST_HALF(CCOLFH,CCOLLH,ACOLFH,ACOLLH,BV,ADDRB,SHIFT,ADDRC) \
" prfm            PSTL1STRM, ["#ADDRC"]           \n\t" /* Prefetch C column 0:7 */ \
  DGEMM_FMLA2_LD1RD(CCOLFH,CCOLLH,ACOLFH,ACOLLH,BV,ADDRB,SHIFT)
#define DGEMM_FMLA2_LD1RD_LAST__HALF(CCOLFH,CCOLLH,ACOLFH,ACOLLH,BV,ADDRB,SHIFT,ADDRC,LDC) \
" prfm            PSTL1STRM, ["#ADDRC", #64]      \n\t" /* Prefetch C column 8:15 */ \
" add             "#ADDRC", "#ADDRC", "#LDC"      \n\t" /* C column forward */ \
  DGEMM_FMLA2_LD1RD(CCOLFH,CCOLLH,ACOLFH,ACOLLH,BV,ADDRB,SHIFT)

#define DGEMM_BADDR_GENERIC(ADDRBIJ,ADDRB,DBBIT,SKIP) \
" mov             "#ADDRBIJ", #"#SKIP"            \n\t" \
" madd  "#ADDRBIJ", "#ADDRBIJ", "#DBBIT", "#ADDRB"\n\t"

#define DGEMM_2VX10_MKER_LOOP_PLAIN_1(C0FH,C1FH,C2FH,C3FH,C4FH,C5FH,C6FH,C7FH,C8FH,C9FH,C0LH,C1LH,C2LH,C3LH,C4LH,C5LH,C6LH,C7LH,C8LH,C9LH,ACOLFH,ACOLLH,BV0,BV1,BV2,BV3,BV4,BV5,BV6,BV7,ADDRB,TDB) \
" madd            x26, "#TDB", x12, xzr           \n\t" /* B's row stride in bits */ \
  DGEMM_BADDR_GENERIC(x24,ADDRB,x26,8) \
  DGEMM_FMLA2_LD1RD(C0FH,C0LH,ACOLFH,ACOLLH,BV0,x24,0) \
  DGEMM_BADDR_GENERIC(x24,ADDRB,x26,9) \
  DGEMM_FMLA2_LD1RD(C1FH,C1LH,ACOLFH,ACOLLH,BV1,x24,0) \
" madd            "#ADDRB", x5, x12, "#ADDRB"     \n\t" /* B address forward */ \
  DGEMM_BADDR_GENERIC(x24,ADDRB,x26,0) \
  DGEMM_FMLA2_LD1RD(C2FH,C2LH,ACOLFH,ACOLLH,BV2,x24,0) \
  DGEMM_BADDR_GENERIC(x24,ADDRB,x26,1) \
  DGEMM_FMLA2_LD1RD(C3FH,C3LH,ACOLFH,ACOLLH,BV3,x24,0) \
  DGEMM_BADDR_GENERIC(x24,ADDRB,x26,2) \
  DGEMM_FMLA2_LD1RD(C4FH,C4LH,ACOLFH,ACOLLH,BV4,x24,0) \
  DGEMM_BADDR_GENERIC(x24,ADDRB,x26,3) \
  DGEMM_FMLA2_LD1RD(C5FH,C5LH,ACOLFH,ACOLLH,BV5,x24,0) \
  DGEMM_BADDR_GENERIC(x24,ADDRB,x26,4) \
  DGEMM_FMLA2_LD1RD(C6FH,C6LH,ACOLFH,ACOLLH,BV6,x24,0) \
  DGEMM_BADDR_GENERIC(x24,ADDRB,x26,5) \
  DGEMM_FMLA2_LD1RD(C7FH,C7LH,ACOLFH,ACOLLH,BV7,x24,0) \
  \
  DGEMM_BADDR_GENERIC(x24,ADDRB,x26,6) \
  DGEMM_FMLA2_LD1RD(C8FH,C8LH,ACOLFH,ACOLLH,BV0,x24,0) \
  DGEMM_BADDR_GENERIC(x24,ADDRB,x26,7) \
  DGEMM_FMLA2_LD1RD(C9FH,C9LH,ACOLFH,ACOLLH,BV1,x24,0)

// Second through forth microkernels are the first one with B vectors rotated.
#define DGEMM_2VX10_MKER_LOOP_PLAIN_2(C0FH,C1FH,C2FH,C3FH,C4FH,C5FH,C6FH,C7FH,C8FH,C9FH,C0LH,C1LH,C2LH,C3LH,C4LH,C5LH,C6LH,C7LH,C8LH,C9LH,ACOLFH,ACOLLH,BV0,BV1,BV2,BV3,BV4,BV5,BV6,BV7,ADDRB,TDB) \
  DGEMM_2VX10_MKER_LOOP_PLAIN_1(C0FH,C1FH,C2FH,C3FH,C4FH,C5FH,C6FH,C7FH,C8FH,C9FH,C0LH,C1LH,C2LH,C3LH,C4LH,C5LH,C6LH,C7LH,C8LH,C9LH,ACOLFH,ACOLLH,BV2,BV3,BV4,BV5,BV6,BV7,BV0,BV1,ADDRB,TDB)

#define DGEMM_2VX10_MKER_LOOP_PLAIN_3(C0FH,C1FH,C2FH,C3FH,C4FH,C5FH,C6FH,C7FH,C8FH,C9FH,C0LH,C1LH,C2LH,C3LH,C4LH,C5LH,C6LH,C7LH,C8LH,C9LH,ACOLFH,ACOLLH,BV0,BV1,BV2,BV3,BV4,BV5,BV6,BV7,ADDRB,TDB) \
  DGEMM_2VX10_MKER_LOOP_PLAIN_1(C0FH,C1FH,C2FH,C3FH,C4FH,C5FH,C6FH,C7FH,C8FH,C9FH,C0LH,C1LH,C2LH,C3LH,C4LH,C5LH,C6LH,C7LH,C8LH,C9LH,ACOLFH,ACOLLH,BV4,BV5,BV6,BV7,BV0,BV1,BV2,BV3,ADDRB,TDB)

#define DGEMM_2VX10_MKER_LOOP_PLAIN_4(C0FH,C1FH,C2FH,C3FH,C4FH,C5FH,C6FH,C7FH,C8FH,C9FH,C0LH,C1LH,C2LH,C3LH,C4LH,C5LH,C6LH,C7LH,C8LH,C9LH,ACOLFH,ACOLLH,BV0,BV1,BV2,BV3,BV4,BV5,BV6,BV7,ADDRB,TDB) \
  DGEMM_2VX10_MKER_LOOP_PLAIN_1(C0FH,C1FH,C2FH,C3FH,C4FH,C5FH,C6FH,C7FH,C8FH,C9FH,C0LH,C1LH,C2LH,C3LH,C4LH,C5LH,C6LH,C7LH,C8LH,C9LH,ACOLFH,ACOLLH,BV6,BV7,BV0,BV1,BV2,BV3,BV4,BV5,ADDRB,TDB)
// NOTE:
//  The microkernel (PLAIN_1-4 as a whole) satisfies on entry/exit
//  (sth. akin to loop-invariant):
//   - BV[0-5] holds B[0:5, 4*k_cur]
//   - Stream LOAD stops at B[0, 4*k_cur+1]

// For rows left behind microkernels.
#define DGEMM_2VX10_MKER_LOOP_PLAIN_RESIDUAL(C0FH,C1FH,C2FH,C3FH,C4FH,C5FH,C6FH,C7FH,C8FH,C9FH,C0LH,C1LH,C2LH,C3LH,C4LH,C5LH,C6LH,C7LH,C8LH,C9LH,ACOLFH,ACOLLH,BV0,BV1,BV2,BV3,BV4,BV5,BV6,BV7,ADDRB,TDB) \
" madd            x26, "#TDB", x12, xzr           \n\t" /* B's row stride in bits */ \
  DGEMM_BADDR_GENERIC(x24,ADDRB,x26,8) \
  DGEMM_FMLA2_LD1RD(C0FH,C0LH,ACOLFH,ACOLLH,BV0,x24,0) \
  DGEMM_BADDR_GENERIC(x24,ADDRB,x26,9) \
  DGEMM_FMLA2_LD1RD(C1FH,C1LH,ACOLFH,ACOLLH,BV1,x24,0) \
" madd            "#ADDRB", x5, x12, "#ADDRB"     \n\t" /* B address forward */ \
  DGEMM_BADDR_GENERIC(x24,ADDRB,x26,0) \
  DGEMM_FMLA2_LD1RD(C8FH,C8LH,ACOLFH,ACOLLH,BV0,x24,0) \
  DGEMM_BADDR_GENERIC(x24,ADDRB,x26,1) \
  DGEMM_FMLA2_LD1RD(C9FH,C9LH,ACOLFH,ACOLLH,BV1,x24,0) \
  \
  DGEMM_BADDR_GENERIC(x24,ADDRB,x26,2) \
  DGEMM_FMLA2_LD1RD(C2FH,C2LH,ACOLFH,ACOLLH,BV2,x24,0) \
  DGEMM_BADDR_GENERIC(x24,ADDRB,x26,3) \
  DGEMM_FMLA2_LD1RD(C3FH,C3LH,ACOLFH,ACOLLH,BV3,x24,0) \
  DGEMM_BADDR_GENERIC(x24,ADDRB,x26,4) \
  DGEMM_FMLA2_LD1RD(C4FH,C4LH,ACOLFH,ACOLLH,BV4,x24,0) \
  DGEMM_BADDR_GENERIC(x24,ADDRB,x26,5) \
  DGEMM_FMLA2_LD1RD(C5FH,C5LH,ACOLFH,ACOLLH,BV5,x24,0) \
  DGEMM_BADDR_GENERIC(x24,ADDRB,x26,6) \
  DGEMM_FMLA2_LD1RD(C6FH,C6LH,ACOLFH,ACOLLH,BV6,x24,0) \
  DGEMM_BADDR_GENERIC(x24,ADDRB,x26,7) \
  DGEMM_FMLA2_LD1RD(C7FH,C7LH,ACOLFH,ACOLLH,BV7,x24,0)

// C-prefetching microkernel 1 and 2.
#define DGEMM_2VX10_MKER_LOOP_PREFETCH_CCOL_5X_1(C0FH,C1FH,C2FH,C3FH,C4FH,C5FH,C6FH,C7FH,C8FH,C9FH,C0LH,C1LH,C2LH,C3LH,C4LH,C5LH,C6LH,C7LH,C8LH,C9LH,ACOLFH,ACOLLH,ADDRC,LDC,BV0,BV1,BV2,BV3,BV4,BV5,BV6,BV7,ADDRB,TDB) \
" madd            x26, "#TDB", x12, xzr           \n\t" /* B's row stride in bits */ \
  DGEMM_BADDR_GENERIC(x24,ADDRB,x26,8) \
  DGEMM_FMLA2_LD1RD_FIRST_HALF(C0FH,C0LH,ACOLFH,ACOLLH,BV0,x24,0,ADDRC) \
  DGEMM_BADDR_GENERIC(x24,ADDRB,x26,9) \
  DGEMM_FMLA2_LD1RD_LAST__HALF(C1FH,C1LH,ACOLFH,ACOLLH,BV1,x24,0,ADDRC,LDC) \
" madd            "#ADDRB", x5, x12, "#ADDRB"     \n\t" /* B address forward */ \
  DGEMM_BADDR_GENERIC(x24,ADDRB,x26,0) \
  DGEMM_FMLA2_LD1RD_FIRST_HALF(C2FH,C2LH,ACOLFH,ACOLLH,BV2,x24,0,ADDRC) \
  DGEMM_BADDR_GENERIC(x24,ADDRB,x26,1) \
  DGEMM_FMLA2_LD1RD_LAST__HALF(C3FH,C3LH,ACOLFH,ACOLLH,BV3,x24,0,ADDRC,LDC) \
  DGEMM_BADDR_GENERIC(x24,ADDRB,x26,2) \
  DGEMM_FMLA2_LD1RD_FIRST_HALF(C4FH,C4LH,ACOLFH,ACOLLH,BV4,x24,0,ADDRC) \
  DGEMM_BADDR_GENERIC(x24,ADDRB,x26,3) \
  DGEMM_FMLA2_LD1RD_LAST__HALF(C5FH,C5LH,ACOLFH,ACOLLH,BV5,x24,0,ADDRC,LDC) \
  DGEMM_BADDR_GENERIC(x24,ADDRB,x26,4) \
  DGEMM_FMLA2_LD1RD_FIRST_HALF(C6FH,C6LH,ACOLFH,ACOLLH,BV6,x24,0,ADDRC) \
  DGEMM_BADDR_GENERIC(x24,ADDRB,x26,5) \
  DGEMM_FMLA2_LD1RD_LAST__HALF(C7FH,C7LH,ACOLFH,ACOLLH,BV7,x24,0,ADDRC,LDC) \
  \
  DGEMM_BADDR_GENERIC(x24,ADDRB,x26,6) \
  DGEMM_FMLA2_LD1RD_FIRST_HALF(C8FH,C8LH,ACOLFH,ACOLLH,BV0,x24,0,ADDRC) \
  DGEMM_BADDR_GENERIC(x24,ADDRB,x26,7) \
  DGEMM_FMLA2_LD1RD_LAST__HALF(C9FH,C9LH,ACOLFH,ACOLLH,BV1,x24,0,ADDRC,LDC)

#define DGEMM_2VX10_MKER_LOOP_PREFETCH_CCOL_5X_2(C0FH,C1FH,C2FH,C3FH,C4FH,C5FH,C6FH,C7FH,C8FH,C9FH,C0LH,C1LH,C2LH,C3LH,C4LH,C5LH,C6LH,C7LH,C8LH,C9LH,ACOLFH,ACOLLH,ADDRC,LDC,BV0,BV1,BV2,BV3,BV4,BV5,BV6,BV7,ADDRB,TDB) \
  DGEMM_2VX10_MKER_LOOP_PREFETCH_CCOL_5X_1(C0FH,C1FH,C2FH,C3FH,C4FH,C5FH,C6FH,C7FH,C8FH,C9FH,C0LH,C1LH,C2LH,C3LH,C4LH,C5LH,C6LH,C7LH,C8LH,C9LH,ACOLFH,ACOLLH,ADDRC,LDC,BV2,BV3,BV4,BV5,BV6,BV7,BV0,BV1,ADDRB,TDB)

#ifndef _A64FX
#define DGEMM_2VX10_BGEN_PREFETCH_NEXT_MKER(ZF,PFST,PLFT,ADDRB,TDB,TAG) \
" index           "#ZF".d, xzr, "#TDB"            \n\t" /* Prefetch B the same line... */ \
" add             x22, x16, "#ADDRB"              \n\t" /* For the next microkernel. */ \
" prfd            PLDL1KEEP, "#PFST", [x22, "#ZF".d, lsl #3] \n\t" /* [0:8] */ \
" madd            x22, "#TDB", x12, x22           \n\t" \
" prfd            PLDL1KEEP, "#PLFT", [x22, "#ZF".d, lsl #3] \n\t" /* [8:10] */ \
" add             x22, x17, "#ADDRB"              \n\t" /* For the next micropanel. */ \
" prfd            PLDL2KEEP, "#PFST", [x22, "#ZF".d, lsl #3] \n\t" /* [0:8] */ \
" madd            x22, "#TDB", x12, x22           \n\t" \
" prfd            PLDL2KEEP, "#PLFT", [x22, "#ZF".d, lsl #3] \n\t" /* [8:10] */
#else
#define DGEMM_2VX10_BGEN_PREFETCH_NEXT_MKER(ZF,PFST,PLFT,ADDRB,TDB,TAG) \
" index           "#ZF".d, xzr, "#TDB"            \n\t" \
" add             x22, x16, "#ADDRB"              \n\t" \
" mov             x27, 0x"#TAG"                   \n\t" \
" lsl             x27, x27, #56                   \n\t" \
" orr             x22, x22, x27                   \n\t" \
" prfd            PLDL1KEEP, "#PFST", [x22, "#ZF".d, lsl #3] \n\t" /* [0:8] */ \
" madd            x22, "#TDB", x12, x22           \n\t" \
" prfd            PLDL1KEEP, "#PLFT", [x22, "#ZF".d, lsl #3] \n\t" /* [8:10] */ \
" add             x22, x17, "#ADDRB"              \n\t" \
" mov             x27, 0x"#TAG"                   \n\t" \
" lsl             x27, x27, #56                   \n\t" \
" orr             x22, x22, x27                   \n\t" \
" prfd            PLDL2KEEP, "#PFST", [x22, "#ZF".d, lsl #3] \n\t" /* [0:8] */ \
" madd            x22, "#TDB", x12, x22           \n\t" \
" prfd            PLDL2KEEP, "#PLFT", [x22, "#ZF".d, lsl #3] \n\t" /* [8:10] */
#endif

/*
   o This implementation uses unindexed FMLA instructions in its kernel.
   o Runnable on ARMv8a with SVE 512 feature compiled with aarch64 GCC.
*/
void __attribute__ ((noinline)) bli_dgemmsup_cv_armsve512_16x10_asm_ukr_unindexed_bg
     (
       uint64_t            conja,
       uint64_t            conjb,
       uint64_t            m_loc,
       uint64_t            k_mker,
       uint64_t            k_left,
       double*    restrict alpha,
       double*    restrict aaddr, uint64_t rs_a, uint64_t cs_a, uint64_t ps_a,
       double*    restrict baddr, uint64_t rs_b, uint64_t cs_b, uint64_t ps_b,
       double*    restrict beta,
       double*    restrict caddr, uint64_t rs_c, uint64_t cs_c,
       double*             a_next,
       double*             b_next
     );

/*
   o 16x10 double precision supplimentary kernel.
   o C-wrap around the asm kernel for n-loop.

   Jan. 2021.
*/
void __attribute__ ((optimize(1))) bli_dgemmsup_cv_armsve512_16x10_unindexed_bg
     (
       conj_t              conja,
       conj_t              conjb,
       dim_t               m0,
       dim_t               n0,
       dim_t               k0,
       double*    restrict alpha,
       double*    restrict a, inc_t rs_a0, inc_t cs_a0,
       double*    restrict b, inc_t rs_b0, inc_t cs_b0,
       double*    restrict beta,
       double*    restrict c, inc_t rs_c0, inc_t cs_c0,
       auxinfo_t* restrict data,
       cntx_t*    restrict cntx
     )
{
  void* a_next0 = bli_auxinfo_next_a( data );
  void* b_next0 = bli_auxinfo_next_b( data );
  uint64_t ps_a = bli_auxinfo_ps_a( data );
  uint64_t ps_b = bli_auxinfo_ps_b( data );

  dim_t m0_mker = m0 / 16;
  dim_t m0_left = m0 % 16;
  dim_t n0_mker = n0 / 10;
  dim_t n0_left = n0 % 10;

  // Edge cases.
  if ( m0_left )
  {
    ++m0_mker;
  }
  if ( n0_left )
  {
    // A[:, ::]
    // B[::, n0_mker*10:n0] => n0_mker * 10 panels.
    // C[: , n0_mker*10:n0]
    double *ai = a;
    double *bi = b + n0_mker * ps_b;
    double *ci = c + 0 * rs_c0 + n0_mker * 10 * cs_c0;
    bli_dgemmsup_c_a64fx_ref // TODO: Fix function name.
    (
      conja, conjb,
      m0, n0_left, k0,
      alpha,
      ai, rs_a0, cs_a0,
      bi, rs_b0, cs_b0,
      beta,
      ci, rs_c0, cs_c0,
      data, // Ensure this is not used.
      cntx
    );
  }
  // Return if it's a pure edge case.
  if ( !(m0_mker * n0_mker) )
    return;

  // Be careful.
  // This kernel supports gc*.
  assert(1 == rs_a0);
  // assert(1 == cs_b0);

#ifdef _A64FX
  // Set A's prefetch controller and prefetch distance.
  uint64_t pf_ctrl_a, pf_dist_a;
  uint64_t pf_ctrl_b, pf_dist_b;
  pf_ctrl_a = ((uint64_t)0x8 << 60) | (0x8);
  pf_dist_a =((cs_a0 * 8) << 32) |
              (cs_a0 * 8 * 4);
  if ((cs_b0 * 8) & ((uint64_t)0x3f << 58))
    // TODO: Handle this situation.
    pf_ctrl_b = ((uint64_t)0x8 << 60) | (0x8);
  else
    pf_ctrl_b = ((uint64_t)0x8 << 60) | (cs_b0 * 8);
  pf_dist_b =((rs_b0 * 8) << 32) |
              (rs_b0 * 8 * 4);

  __asm__ (
  "msr  S3_3_C11_C6_2, %[ctrl_a] \n\t"
  "msr  S3_3_C11_C7_2, %[dist_a] \n\t"
  "msr  S3_3_C11_C6_3, %[ctrl_b] \n\t"
  "msr  S3_3_C11_C7_3, %[dist_b] \n\t"
: // output (none).
: // input
  [ctrl_a] "r" (pf_ctrl_a),
  [dist_a] "r" (pf_dist_a),
  [ctrl_b] "r" (pf_ctrl_b),
  [dist_b] "r" (pf_dist_b)
: // clobber
  );
#endif

  // Typecast local copies of integers in case dim_t and inc_t are a
  // different size than is expected by load instructions.
  uint64_t k_mker = k0 / 4;
  uint64_t k_left = k0 % 4;
  uint64_t rs_c   = rs_c0;
  uint64_t cs_c   = cs_c0;
  uint64_t rs_a   = 1;
  uint64_t cs_a   = cs_a0;
  uint64_t rs_b   = rs_b0;
  uint64_t cs_b   = cs_b0;

  for ( int64_t im0_mker = 0; im0_mker < m0_mker; ++im0_mker )
    for ( int64_t in0_mker = 0; in0_mker < n0_mker; ++in0_mker )
    {
      uint64_t m_loc = 16;
      if ( m0_left && im0_mker == m0_mker - 1 )
      {
        // Edge case.
        m_loc = m0_left;
      }

      double *ai = a + im0_mker * 16 * rs_a;
      double *bi = b + in0_mker * ps_b;
      double *ci = c + im0_mker * 16 * rs_a + in0_mker * 10 * cs_a;

      void* a_next = NULL;
      void* b_next = NULL;
      // NOTE: Prefetches at this moment are based on an assumption that
      //       im0_mker == 1.
      if ( in0_mker == n0_mker - 1 )
      {
        a_next = a_next0;
        b_next = b_next0;
      }

      bli_dgemmsup_cv_armsve512_16x10_asm_ukr_unindexed_bg
      (
        conja != BLIS_NO_CONJUGATE,
        conjb != BLIS_NO_CONJUGATE,
        m_loc, k_mker, k_left,
        alpha,
        ai, rs_a, cs_a, ps_a,
        bi, rs_b, cs_b, ps_b,
        beta,
        ci, rs_c, cs_c,
        a_next,
        b_next
      );
    }

}

void __attribute__ ((noinline)) bli_dgemmsup_cv_armsve512_16x10_asm_ukr_unindexed_bg
     (
       uint64_t            conja,
       uint64_t            conjb,
       uint64_t            m_loc,
       uint64_t            k_mker,
       uint64_t            k_left,
       double*    restrict alpha,
       double*    restrict aaddr, uint64_t rs_a, uint64_t cs_a, uint64_t ps_a,
       double*    restrict baddr, uint64_t rs_b, uint64_t cs_b, uint64_t ps_b,
       double*    restrict beta,
       double*    restrict caddr, uint64_t rs_c, uint64_t cs_c,
       double*             a_next,
       double*             b_next
     )
{ __asm__ volatile (
" ldr             x9,  %[m_loc]                   \n\t" // Shape M, from input
"                                                 \n\t" // Shape N is fixed to be 10
" ldr             x8,  %[k_left]                  \n\t" // Shape K to be contracted
" ldr             x21, %[k_mker]                  \n\t" // Size of K-microblock
"                                                 \n\t"
" ldr             x6,  %[caddr]                   \n\t" // Load address of C
" ldr             x20, %[rs_c]                    \n\t" // Row-skip of C
" ldr             x7,  %[cs_c]                    \n\t" // LdC, which is called column-skip in BLIS
" cmp             x20, #1                         \n\t"
"                                                 \n\t"
" ldr             x0,  %[alpha]                   \n\t" // Alpha address
" ldr             x1,  %[beta]                    \n\t" // Beta address
"                                                 \n\t"
" ldr             x2,  %[aaddr]                   \n\t" // Load address of A
" ldr             x3,  %[cs_a]                    \n\t" // Leading dimension of A
"                                                 \n\t" // Row-skip of A is fixed to be 1.
" ldr             x4,  %[baddr]                   \n\t" // Load address of B
" ldr             x5,  %[rs_b]                    \n\t" // Leading dimension of B
" ldr             x25, %[cs_b]                    \n\t" // Trailing dimension of B
"                                                 \n\t"
" ldr             x17, %[ps_b]                    \n\t" // B's panel stride
" ldr             x18, %[a_next]                  \n\t" // Pointer to next A pack
" ldr             x19, %[b_next]                  \n\t" // Pointer to next B pack
"                                                 \n\t"
#ifdef _A64FX
" mov x26, 0x3      \n\t" // A64FX: Use cache sector 3 for C_r microtile
" lsl x26, x26, 56  \n\t"
" orr x6, x6, x26   \n\t"
"                   \n\t"
// " mov x26, 0xb      \n\t" // A64FX: Prefetch injection mode for B_r microtile
// " lsl x26, x26, 60  \n\t" //        according to control#3 register.
" mov x26, 0x6      \n\t" // A64FX: Disable hardware prefetching for B.
" lsl x26, x26, 60  \n\t" //
// " mov x26, 0x2      \n\t" // A64FX: Tag B address
// " lsl x26, x26, 56  \n\t" //        as 2.
" orr x4, x4, x26   \n\t"
"                   \n\t"
// " mov x26, 0xa      \n\t" // A64FX: Prefetch injection mode for A_r microtile
// " lsl x26, x26, 60  \n\t" //        according to control#2 register.
" mov x26, 0x1      \n\t" // A64FX: Tag A address
" lsl x26, x26, 56  \n\t" //        as 1.
" orr x2, x2, x26   \n\t"
#endif
"                                                 \n\t"
" b.ne            C_PRFML2_STRIDED                \n\t"
"                                                 \n\t" // Registers occupied: X0-9, X20, X21
" mov             x10, #8                         \n\t" // Double in bytes, will be destroyed.
" madd            x22, x7, x10, xzr               \n\t"
"                                                 \n\t"
"                                                 \n\t" // C column 0 is x6
" add             x11, x6, x22                    \n\t" // C column 1
" add             x12, x11, x22                   \n\t" // C column 2
" add             x13, x12, x22                   \n\t" // C column 3
" add             x14, x13, x22                   \n\t" // C column 4
" prfm            PSTL2STRM, [x6]                 \n\t" // Prefetch C column 0
" prfm            PSTL2STRM, [x6, #64]            \n\t"
" prfm            PSTL2STRM, [x11]                \n\t" // Prefetch C column 1
" prfm            PSTL2STRM, [x11,#64]            \n\t"
" prfm            PSTL2STRM, [x12]                \n\t" // Prefetch C column 2
" prfm            PSTL2STRM, [x12,#64]            \n\t"
" prfm            PSTL2STRM, [x13]                \n\t" // Prefetch C column 3
" prfm            PSTL2STRM, [x13,#64]            \n\t"
" prfm            PSTL2STRM, [x14]                \n\t" // Prefetch C column 4
" prfm            PSTL2STRM, [x14,#64]            \n\t"
"                                                 \n\t"
" add             x10, x14, x22                   \n\t" // C column 5
" add             x11, x10, x22                   \n\t" // C column 6
" add             x12, x11, x22                   \n\t" // C column 7
" add             x13, x12, x22                   \n\t" // C column 8
" add             x14, x13, x22                   \n\t" // C column 9
" prfm            PSTL2STRM, [x10]                \n\t" // Prefetch C column 5
" prfm            PSTL2STRM, [x10,#64]            \n\t"
" prfm            PSTL2STRM, [x11]                \n\t" // Prefetch C column 6
" prfm            PSTL2STRM, [x11,#64]            \n\t"
" prfm            PSTL2STRM, [x12]                \n\t" // Prefetch C column 7
" prfm            PSTL2STRM, [x12,#64]            \n\t"
" prfm            PSTL2STRM, [x13]                \n\t" // Prefetch C column 8
" prfm            PSTL2STRM, [x13,#64]            \n\t"
" prfm            PSTL2STRM, [x14]                \n\t" // Prefetch C column 9
" prfm            PSTL2STRM, [x14,#64]            \n\t"
"                                                 \n\t"
" b               END_C_PRFML2                    \n\t"
"                                                 \n\t"
" C_PRFML2_STRIDED:                               \n\t"
" mov             x10, #8                         \n\t" // Double in bytes, will be destroyed.
"                                                 \n\t" //  == vector length in doubles.
" madd            x23, x20, x10, xzr              \n\t" // Column stride in bytes
" madd            x23, x23, x10, xzr              \n\t" // Vector length in memory
" madd            x22, x7, x10, xzr               \n\t"
" ptrue           p0.d, all                       \n\t"
"                                                 \n\t"
"                                                 \n\t" // Z30: index for prefetching C columns.
" index           z30.d, xzr, x20                 \n\t" // Generate indices.
"                                                 \n\t"
"                                                 \n\t" // 0:7  Column 0 is X6
" add             x11, x6, x23                    \n\t" // 8:16 Column 0
" add             x12, x6, x22                    \n\t" // 0:7  Column 1
" add             x13, x12, x23                   \n\t" // 8:16 Column 1
" add             x14, x12, x22                   \n\t" // 0:7  Column 2
" add             x15, x14, x23                   \n\t" // 8:16 Column 2
" prfd        PSTL2STRM, p0, [x6, z30.d, lsl #3]  \n\t" // Prefetch C column 0
" prfd        PSTL2STRM, p0, [x11, z30.d, lsl #3] \n\t"
" prfd        PSTL2STRM, p0, [x12, z30.d, lsl #3] \n\t" // Prefetch C column 1
" prfd        PSTL2STRM, p0, [x13, z30.d, lsl #3] \n\t"
" prfd        PSTL2STRM, p0, [x14, z30.d, lsl #3] \n\t" // Prefetch C column 2
" prfd        PSTL2STRM, p0, [x15, z30.d, lsl #3] \n\t"
"                                                 \n\t"
" add             x10, x14, x22                   \n\t" // 0:7  Column 3
" add             x11, x10, x23                   \n\t" // 8:16 Column 3
" add             x12, x10, x22                   \n\t" // 0:7  Column 4
" add             x13, x12, x23                   \n\t" // 8:16 Column 4
" add             x14, x12, x22                   \n\t" // 0:7  Column 5
" add             x15, x14, x23                   \n\t" // 8:16 Column 5
" prfd        PSTL2STRM, p0, [x10, z30.d, lsl #3] \n\t" // Prefetch C column 3
" prfd        PSTL2STRM, p0, [x11, z30.d, lsl #3] \n\t"
" prfd        PSTL2STRM, p0, [x12, z30.d, lsl #3] \n\t" // Prefetch C column 4
" prfd        PSTL2STRM, p0, [x13, z30.d, lsl #3] \n\t"
" prfd        PSTL2STRM, p0, [x14, z30.d, lsl #3] \n\t" // Prefetch C column 5
" prfd        PSTL2STRM, p0, [x15, z30.d, lsl #3] \n\t"
"                                                 \n\t"
" add             x10, x14, x22                   \n\t" // 0:7  Column 6
" add             x11, x10, x23                   \n\t" // 8:16 Column 6
" add             x12, x10, x22                   \n\t" // 0:7  Column 7
" add             x13, x12, x23                   \n\t" // 8:16 Column 7
" add             x14, x12, x22                   \n\t" // 0:7  Column 8
" add             x15, x14, x23                   \n\t" // 8:16 Column 8
" prfd        PSTL2STRM, p0, [x10, z30.d, lsl #3] \n\t" // Prefetch C column 6
" prfd        PSTL2STRM, p0, [x11, z30.d, lsl #3] \n\t"
" prfd        PSTL2STRM, p0, [x12, z30.d, lsl #3] \n\t" // Prefetch C column 7
" prfd        PSTL2STRM, p0, [x13, z30.d, lsl #3] \n\t"
" prfd        PSTL2STRM, p0, [x14, z30.d, lsl #3] \n\t" // Prefetch C column 8
" prfd        PSTL2STRM, p0, [x15, z30.d, lsl #3] \n\t"
"                                                 \n\t"
" add             x10, x14, x22                   \n\t" // 0:7  Column 9
" add             x11, x10, x23                   \n\t" // 8:16 Column 9
" prfd        PSTL2STRM, p0, [x10, z30.d, lsl #3] \n\t" // Prefetch C column 9
" prfd        PSTL2STRM, p0, [x11, z30.d, lsl #3] \n\t"
"                                                 \n\t"
" b               END_C_PRFML2                    \n\t"
"                                                 \n\t"
" END_C_PRFML2:                                   \n\t"
"                                                 \n\t"
" mov             x12, #2                         \n\t" // 10-8=2.
" mov             x11, xzr                        \n\t"
" incd            x11                             \n\t" // Determine vector length, in doubles.
"                                                 \n\t"
" whilelo         p0.d, xzr, x9                   \n\t" // First half from M argument.
" whilelo         p1.d, x11, x9                   \n\t" // Second half, could be all-false.
" whilelo         p2.d, xzr, x12                  \n\t" // 10-8=2.
"                                                 \n\t"
" mov             x12, #8                         \n\t" // Double in bytes
" mov             x16, #4                         \n\t"
" madd            x16, x16, x12, xzr              \n\t"
" madd            x16, x16, x5, xzr               \n\t" // K=4 microkernel skip in bits.
" fmov            d0, #1.0                        \n\t" // Exact floating-point 1.0.
" fmov            x14, d0                         \n\t" // Hard float to avoid conflict with SVE.
"                                                 \n\t"
"                                                 \n\t" // SVE Register configuration:
"                                                 \n\t" // Z[30-31]: A columns
"                                                 \n\t" // Z[0-5]: B elements broadcasted
"                                                 \n\t" // Z[6-29]: C change buffer
"                                                 \n\t"
" FIRST_BCOL:                                     \n\t" // Load B[0:8, 0].
" madd            x26, x25, x12, xzr              \n\t" // B's row stride in bits.
" ld1rd           z0.d, p0/z, [x4]                \n\t"
DGEMM_BADDR_GENERIC(x24,x4,x26,1)
" ld1rd           z1.d, p0/z, [x24]               \n\t"
DGEMM_BADDR_GENERIC(x24,x4,x26,2)
" ld1rd           z2.d, p0/z, [x24]               \n\t"
DGEMM_BADDR_GENERIC(x24,x4,x26,3)
" ld1rd           z3.d, p0/z, [x24]               \n\t"
DGEMM_BADDR_GENERIC(x24,x4,x26,4)
" ld1rd           z4.d, p0/z, [x24]               \n\t"
DGEMM_BADDR_GENERIC(x24,x4,x26,5)
" ld1rd           z5.d, p0/z, [x24]               \n\t"
DGEMM_BADDR_GENERIC(x24,x4,x26,6)
" ld1rd           z26.d, p0/z, [x24]              \n\t"
DGEMM_BADDR_GENERIC(x24,x4,x26,7)
" ld1rd           z27.d, p0/z, [x24]              \n\t"
" FIRST_ACOL:                                     \n\t" // Load of first & second column of A.
" ld1d            z28.d, p0/z, [x2]               \n\t"
" ld1d            z29.d, p1/z, [x2, #1, MUL VL]   \n\t"
"                                                 \n\t"
" CLEAR_CCOLS:                                    \n\t" // Clear registers.
" fmov            z6.d, p0/m, #0.0                \n\t"
" fmov            z7.d, p0/m, #0.0                \n\t"
" fmov            z8.d, p0/m, #0.0                \n\t"
" fmov            z9.d, p0/m, #0.0                \n\t"
" fmov            z10.d, p0/m, #0.0               \n\t"
" fmov            z11.d, p0/m, #0.0               \n\t"
" fmov            z12.d, p0/m, #0.0               \n\t"
" fmov            z13.d, p0/m, #0.0               \n\t"
" fmov            z14.d, p0/m, #0.0               \n\t"
" fmov            z15.d, p0/m, #0.0               \n\t"
" fmov            z16.d, p0/m, #0.0               \n\t"
" fmov            z17.d, p0/m, #0.0               \n\t"
" fmov            z18.d, p0/m, #0.0               \n\t"
" fmov            z19.d, p0/m, #0.0               \n\t"
" fmov            z20.d, p0/m, #0.0               \n\t"
" fmov            z21.d, p0/m, #0.0               \n\t"
" fmov            z22.d, p0/m, #0.0               \n\t"
" fmov            z23.d, p0/m, #0.0               \n\t"
" fmov            z24.d, p0/m, #0.0               \n\t"
" fmov            z25.d, p0/m, #0.0               \n\t"
"                                                 \n\t"
" cmp             x21, #0                         \n\t" // If no 4-microkernel can be applied
" b.eq            K_LEFT_LOOP                     \n\t"
"                                                 \n\t"
" K_MKER_LOOP:                                    \n\t" // Unroll the 4-loop.
" madd            x22, x21, x20, xzr              \n\t"
" cmp             x22, #1                         \n\t"
" b.eq            K_MKER_LOOP_FINAL               \n\t"
"                                                 \n\t"
DGEMM_2VX10_BGEN_PREFETCH_NEXT_MKER(z30,p0,p2,x4,x25,2)
" madd            x2, x3, x12, x2                 \n\t" // A address forward
" ld1d            z30.d, p0/z, [x2]               \n\t"
" ld1d            z31.d, p1/z, [x2, #1, MUL VL]   \n\t"
DGEMM_2VX10_MKER_LOOP_PLAIN_1(z6,z8,z10,z12,z14,z16,z18,z20,z22,z24,z7,z9,z11,z13,z15,z17,z19,z21,z23,z25,z28,z29,z0,z1,z2,z3,z4,z5,z26,z27,x4,x25)
"                                                 \n\t"
DGEMM_2VX10_BGEN_PREFETCH_NEXT_MKER(z28,p0,p2,x4,x25,4)
" madd            x2, x3, x12, x2                 \n\t" // A address forward
" ld1d            z28.d, p0/z, [x2]               \n\t"
" ld1d            z29.d, p1/z, [x2, #1, MUL VL]   \n\t"
DGEMM_2VX10_MKER_LOOP_PLAIN_2(z6,z8,z10,z12,z14,z16,z18,z20,z22,z24,z7,z9,z11,z13,z15,z17,z19,z21,z23,z25,z30,z31,z0,z1,z2,z3,z4,z5,z26,z27,x4,x25)
"                                                 \n\t"
DGEMM_2VX10_BGEN_PREFETCH_NEXT_MKER(z30,p0,p2,x4,x25,5)
" madd            x2, x3, x12, x2                 \n\t" // A address forward
" ld1d            z30.d, p0/z, [x2]               \n\t"
" ld1d            z31.d, p1/z, [x2, #1, MUL VL]   \n\t"
DGEMM_2VX10_MKER_LOOP_PLAIN_3(z6,z8,z10,z12,z14,z16,z18,z20,z22,z24,z7,z9,z11,z13,z15,z17,z19,z21,z23,z25,z28,z29,z0,z1,z2,z3,z4,z5,z26,z27,x4,x25)
"                                                 \n\t"
" sub             x10, x21, #1                    \n\t" // Before final replica,
" adds            x10, x10, x8                    \n\t" //  check if this iteration is final
" b.eq            FIN_LOOP_POPPED                 \n\t"
"                                                 \n\t"
DGEMM_2VX10_BGEN_PREFETCH_NEXT_MKER(z28,p0,p2,x4,x25,6)
" madd            x2, x3, x12, x2                 \n\t" // A address forward
" ld1d            z28.d, p0/z, [x2]               \n\t"
" ld1d            z29.d, p1/z, [x2, #1, MUL VL]   \n\t"
DGEMM_2VX10_MKER_LOOP_PLAIN_4(z6,z8,z10,z12,z14,z16,z18,z20,z22,z24,z7,z9,z11,z13,z15,z17,z19,z21,z23,z25,z30,z31,z0,z1,z2,z3,z4,z5,z26,z27,x4,x25)
"                                                 \n\t"
" subs            x21, x21, #1                    \n\t" // Decrease counter.
" b.ne            K_MKER_LOOP                     \n\t"
" b               K_LEFT_LOOP                     \n\t"
"                                                 \n\t"
" K_MKER_LOOP_FINAL:                              \n\t"
"                                                 \n\t" // In final M-Kernel, C microtiles
"                                                 \n\t" //   are prefetched instead of A & B.
"                                                 \n\t" // Still, A & B in K_left will be prefetched
"                                                 \n\t" //   to some extent by stream prefetcher.
" madd            x23, x7, x12, xzr               \n\t"
" mov             x22, x6                         \n\t" // Prepare C address for prefetching.
"                                                 \n\t"
" madd            x2, x3, x12, x2                 \n\t" // A address forward
" ld1d            z30.d, p0/z, [x2]               \n\t"
" ld1d            z31.d, p1/z, [x2, #1, MUL VL]   \n\t"
DGEMM_2VX10_MKER_LOOP_PREFETCH_CCOL_5X_1(z6,z8,z10,z12,z14,z16,z18,z20,z22,z24,z7,z9,z11,z13,z15,z17,z19,z21,z23,z25,z28,z29,x22,x23,z0,z1,z2,z3,z4,z5,z26,z27,x4,x25)
"                                                 \n\t"
" madd            x2, x3, x12, x2                 \n\t" // A address forward
" ld1d            z28.d, p0/z, [x2]               \n\t"
" ld1d            z29.d, p1/z, [x2, #1, MUL VL]   \n\t"
DGEMM_2VX10_MKER_LOOP_PREFETCH_CCOL_5X_2(z6,z8,z10,z12,z14,z16,z18,z20,z22,z24,z7,z9,z11,z13,z15,z17,z19,z21,z23,z25,z30,z31,x22,x23,z0,z1,z2,z3,z4,z5,z26,z27,x4,x25)
"                                                 \n\t"
" madd            x2, x3, x12, x2                 \n\t" // A address forward
" ld1d            z30.d, p0/z, [x2]               \n\t"
" ld1d            z31.d, p1/z, [x2, #1, MUL VL]   \n\t"
DGEMM_2VX10_MKER_LOOP_PLAIN_3(z6,z8,z10,z12,z14,z16,z18,z20,z22,z24,z7,z9,z11,z13,z15,z17,z19,z21,z23,z25,z28,z29,z0,z1,z2,z3,z4,z5,z26,z27,x4,x25)
"                                                 \n\t"
" sub             x10, x21, #1                    \n\t" // Before final replica,
" adds            x10, x10, x8                    \n\t" //  check if this iteration is final
" b.eq            FIN_LOOP_POPPED                 \n\t"
"                                                 \n\t"
" madd            x2, x3, x12, x2                 \n\t" // A address forward
" ld1d            z28.d, p0/z, [x2]               \n\t"
" ld1d            z29.d, p1/z, [x2, #1, MUL VL]   \n\t"
DGEMM_2VX10_MKER_LOOP_PLAIN_4(z6,z8,z10,z12,z14,z16,z18,z20,z22,z24,z7,z9,z11,z13,z15,z17,z19,z21,z23,z25,z30,z31,z0,z1,z2,z3,z4,z5,z26,z27,x4,x25)
"                                                 \n\t"
" subs            x21, x21, #1                    \n\t" // Decrease counter.
"                                                 \n\t" // One more microkernel for looping over
"                                                 \n\t" //   non-mker k values.
"                                                 \n\t"
" K_LEFT_LOOP:                                    \n\t" // Non-mker k values. Ensure that A stored in Z28/29.
"                                                 \n\t"
" cmp             x8, #0                          \n\t" // Spetial handler only for k0 == 0.
" b.eq            WRITE_MEM                       \n\t"
"                                                 \n\t"
" cmp             x8, #1                          \n\t" // If K=1.
" b.eq            FIN_LOOP                        \n\t"
"                                                 \n\t"
" madd            x2, x3, x12, x2                 \n\t" // A address forward
" ld1d            z30.d, p0/z, [x2]               \n\t"
" ld1d            z31.d, p1/z, [x2, #1, MUL VL]   \n\t"
DGEMM_2VX10_MKER_LOOP_PLAIN_RESIDUAL(z6,z8,z10,z12,z14,z16,z18,z20,z22,z24,z7,z9,z11,z13,z15,z17,z19,z21,z23,z25,z30,z31,z0,z1,z2,z3,z4,z5,z26,z27,x4,x25)
" mov             z28.d, z30.d                    \n\t" // Copy to Z28/29 for convenience.
" mov             z29.d, z31.d                    \n\t"
"                                                 \n\t"
" NEXT_ROW:                                       \n\t"
" sub             x8, x8, #1                      \n\t"
" cmp             x8, #1                          \n\t"
" b.ne            K_LEFT_LOOP                     \n\t" // Next column / row.
"                                                 \n\t"
" FIN_LOOP:                                       \n\t" // Final K-loop. Ensure that A is stored in Z30/31.
"                                                 \n\t"
"                                                 \n\t" // Final A & B[0:7, end] are already loaded
"                                                 \n\t"
" madd            x26, x25, x12, xzr              \n\t" // B's row stride in bits.
DGEMM_BADDR_GENERIC(x24,x4,x26,8)
DGEMM_FMLA2_LD1RD(z6,z7,z28,z29,z0,x24,0) // Column 0
DGEMM_BADDR_GENERIC(x24,x4,x26,9)
DGEMM_FMLA2_LD1RD(z8,z9,z28,z29,z1,x24,0) // Column 1
DGEMM_FMLA2(z10,z11,z28,z29,z2) // Column 2
DGEMM_FMLA2(z12,z13,z28,z29,z3) // Column 3
DGEMM_FMLA2(z14,z15,z28,z29,z4) // Column 4
DGEMM_FMLA2(z16,z17,z28,z29,z5) // Column 5
DGEMM_FMLA2(z18,z19,z28,z29,z26) // Column 6
DGEMM_FMLA2(z20,z21,z28,z29,z27) // Column 7
DGEMM_FMLA2(z22,z23,z28,z29,z0) // Column 8
DGEMM_FMLA2(z24,z25,z28,z29,z1) // Column 9
" b               WRITE_MEM                       \n\t"
"                                                 \n\t"
" FIN_LOOP_POPPED:                                \n\t" // Final K-loop popped from inside microkernels
"                                                 \n\t"
" madd            x26, x25, x12, xzr              \n\t" // B's row stride in bits.
DGEMM_BADDR_GENERIC(x24,x4,x26,8)
DGEMM_FMLA2_LD1RD(z6,z7,z30,z31,z26,x24,0) // Column 0
DGEMM_BADDR_GENERIC(x24,x4,x26,9)
DGEMM_FMLA2_LD1RD(z8,z9,z30,z31,z27,x24,0) // Column 1
DGEMM_FMLA2(z10,z11,z30,z31,z0) // Column 2
DGEMM_FMLA2(z12,z13,z30,z31,z1) // Column 3
DGEMM_FMLA2(z14,z15,z30,z31,z2) // Column 4
DGEMM_FMLA2(z16,z17,z30,z31,z3) // Column 5
DGEMM_FMLA2(z18,z19,z30,z31,z4) // Column 6
DGEMM_FMLA2(z20,z21,z30,z31,z5) // Column 7
DGEMM_FMLA2(z22,z23,z30,z31,z26) // Column 8
DGEMM_FMLA2(z24,z25,z30,z31,z27) // Column 9
"                                                 \n\t"
" WRITE_MEM:                                      \n\t"
"                                                 \n\t" // Override A and B buffers:
"                                                 \n\t" // Z[30-31]: extended alpha and beta.
"                                                 \n\t" // Z[0-1]: C memory buffer.
"                                                 \n\t"
" ldr             x15, [x0]                       \n\t" // Alpha, as 64-bits
" ld1rd           z30.d, p0/z, [x0]               \n\t" // Alpha, to the vector.
" ld1rd           z31.d, p0/z, [x1]               \n\t" // Beta, to the vector.
"                                                 \n\t"
" PRFM_NEXT:                                      \n\t" // Prefetch next A and B.
" prfm            PLDL2KEEP, [x18]                \n\t" // Prefetch 2 panels to
" prfm            PLDL2KEEP, [x18, #64]           \n\t" //   "confirm" the stream prefetcher.
" prfm            PLDL2KEEP, [x19]                \n\t"
" prfm            PLDL2KEEP, [x19, #64]           \n\t"
"                                                 \n\t"
"                                                 \n\t"
" cmp             x14, x15                        \n\t" // (R&)Write data back to C memory.
" b.eq            UNIT_ALPHA                      \n\t"
"                                                 \n\t"
" fmul            z6.d, z6.d, z30.d               \n\t" // Non-unit alpha case.
" fmul            z7.d, z7.d, z30.d               \n\t" // Scale all C change buffers.
" fmul            z8.d, z8.d, z30.d               \n\t"
" fmul            z9.d, z9.d, z30.d               \n\t"
" fmul            z10.d, z10.d, z30.d             \n\t"
" fmul            z11.d, z11.d, z30.d             \n\t"
" fmul            z12.d, z12.d, z30.d             \n\t"
" fmul            z13.d, z13.d, z30.d             \n\t"
" fmul            z14.d, z14.d, z30.d             \n\t"
" fmul            z15.d, z15.d, z30.d             \n\t"
" fmul            z16.d, z16.d, z30.d             \n\t"
" fmul            z17.d, z17.d, z30.d             \n\t"
" fmul            z18.d, z18.d, z30.d             \n\t"
" fmul            z19.d, z19.d, z30.d             \n\t"
" fmul            z20.d, z20.d, z30.d             \n\t"
" fmul            z21.d, z21.d, z30.d             \n\t"
" fmul            z22.d, z22.d, z30.d             \n\t"
" fmul            z23.d, z23.d, z30.d             \n\t"
" fmul            z24.d, z24.d, z30.d             \n\t"
" fmul            z25.d, z25.d, z30.d             \n\t"
"                                                 \n\t"
" UNIT_ALPHA:                                     \n\t" // Unit alpha case.
" cmp             x20, #1                         \n\t"
" b.ne            CS_CCOL                         \n\t"
"                                                 \n\t"
" CT_CCOL:                                        \n\t" // Contiguous columns.
"                                                 \n\t" // X10=12 counter no longer used.
"                                                 \n\t"
"                                                 \n\t" // Update C[:, 0:2]
"                                                 \n\t" // Z[0, 2, 4] = C[0:7,  0, 1, 2]
"                                                 \n\t" // Z[1, 3, 5] = C[8:15, 0, 1, 2]
"                                                 \n\t" // After Z[6-11] are used:
"                                                 \n\t" // Z[6, 8, 10] = C[0:7,  3, 4, 5]
"                                                 \n\t" // Z[7, 9, 11] = C[8:15, 3, 4, 5]
"                                                 \n\t"
" mov             x0, x6                          \n\t" // Clone address for R/W.
" mul             x22, x7, x12                    \n\t" // Row stride in bytes.
"                                                 \n\t"
" ld1d            z0.d, p0/z, [x0]                \n\t" // Read C[:, 0]
" ld1d            z1.d, p1/z, [x0, x11, lsl #3]   \n\t"
" add             x0, x22, x0                     \n\t" // Next column
" ld1d            z2.d, p0/z, [x0]                \n\t" // Read C[:, 1]
" ld1d            z3.d, p1/z, [x0, x11, lsl #3]   \n\t"
" add             x0, x22, x0                     \n\t" // Next column
" ld1d            z4.d, p0/z, [x0]                \n\t" // Read C[:, 2]
" ld1d            z5.d, p1/z, [x0, x11, lsl #3]   \n\t"
" add             x0, x22, x0                     \n\t" // Next column
"                                                 \n\t"
" fmad            z0.d, p0/m, z31.d, z6.d         \n\t" // Z6 used
" ld1d            z6.d, p0/z, [x0]                \n\t" // Read C[0:7, 3]
" fmad            z1.d, p1/m, z31.d, z7.d         \n\t" // Z7 used
" ld1d            z7.d, p1/z, [x0, x11, lsl #3]   \n\t" // Read C[8:15, 3]
" add             x0, x22, x0                     \n\t" // Next column
" fmad            z2.d, p0/m, z31.d, z8.d         \n\t" // Z8 used
" ld1d            z8.d, p0/z, [x0]                \n\t" // Read C[0:7, 4]
" fmad            z3.d, p1/m, z31.d, z9.d         \n\t" // Z9 used
" ld1d            z9.d, p1/z, [x0, x11, lsl #3]   \n\t" // Read C[8:15, 4]
" add             x0, x22, x0                     \n\t" // Next column
" fmad            z4.d, p0/m, z31.d, z10.d        \n\t" // Z10 used
" ld1d            z10.d, p0/z, [x0]               \n\t" // Read C[0:7, 5]
" fmad            z5.d, p1/m, z31.d, z11.d        \n\t" // Z11 used
" ld1d            z11.d, p1/z, [x0, x11, lsl #3]  \n\t" // Read C[8:15, 5]
" add             x0, x22, x0                     \n\t" // Next column
"                                                 \n\t"
" st1d            z0.d, p0, [x6]                  \n\t" // Write C[:, 0]
" st1d            z1.d, p1, [x6, x11, lsl #3]     \n\t"
" add             x6, x22, x6                     \n\t" // Next column
" st1d            z2.d, p0, [x6]                  \n\t" // Write C[:, 1]
" st1d            z3.d, p1, [x6, x11, lsl #3]     \n\t"
" add             x6, x22, x6                     \n\t" // Next column
" st1d            z4.d, p0, [x6]                  \n\t" // Write C[:, 2]
" st1d            z5.d, p1, [x6, x11, lsl #3]     \n\t"
" add             x6, x22, x6                     \n\t" // Next column
"                                                 \n\t"
"                                                 \n\t" // After Z[0-5], Z[12-17] are used:
"                                                 \n\t" // Z[0, 2, 4] = C[0:7,  6, 7, 8]
"                                                 \n\t" // Z[1, 3, 5] = C[8:15, 6, 7, 8]
"                                                 \n\t" // Z[12, 14, 16] = C[0:7,  9, 10, 11]
"                                                 \n\t" // Z[13, 15, 17] = C[8:15, 9, 10, 11]
"                                                 \n\t"
" fmad            z6.d, p0/m, z31.d, z12.d        \n\t" // Z12 used
" ld1d            z0.d, p0/z, [x0]                \n\t" // Read C[0:7, 6]
" fmad            z7.d, p1/m, z31.d, z13.d        \n\t" // Z13 used
" ld1d            z1.d, p1/z, [x0, x11, lsl #3]   \n\t" // Read C[8:15, 6]
" add             x0, x22, x0                     \n\t" // Next column
" fmad            z8.d, p0/m, z31.d, z14.d        \n\t" // Z14 used
" ld1d            z2.d, p0/z, [x0]                \n\t" // Read C[0:7, 7]
" fmad            z9.d, p1/m, z31.d, z15.d        \n\t" // Z15 used
" ld1d            z3.d, p1/z, [x0, x11, lsl #3]   \n\t" // Read C[8:15, 7]
" add             x0, x22, x0                     \n\t" // Next column
" fmad            z10.d, p0/m, z31.d, z16.d       \n\t" // Z16 used
" ld1d            z4.d, p0/z, [x0]                \n\t" // Read C[0:7, 8]
" fmad            z11.d, p1/m, z31.d, z17.d       \n\t" // Z17 used
" ld1d            z5.d, p1/z, [x0, x11, lsl #3]   \n\t" // Read C[8:15, 8]
" add             x0, x22, x0                     \n\t" // Next column
"                                                 \n\t"
" st1d            z6.d, p0, [x6]                  \n\t" // Write C[0:7, 3]
" ld1d            z12.d, p0/z, [x0]               \n\t" // Read C[0:7, 9]
" st1d            z7.d, p1, [x6, x11, lsl #3]     \n\t" // Write C[8:15, 3]
" ld1d            z13.d, p1/z, [x0, x11, lsl #3]  \n\t" // Read C[8:15, 9]
" add             x0, x22, x0                     \n\t" // Next column
" add             x6, x22, x6                     \n\t" // Next column
" st1d            z8.d, p0, [x6]                  \n\t" // Write C[0:7, 4]
" st1d            z9.d, p1, [x6, x11, lsl #3]     \n\t" // Write C[8:15, 4]
// " ld1d            z14.d, p0/z, [x0]               \n\t" // Read C[0:7, 10]
// " ld1d            z15.d, p1/z, [x0, x11, lsl #3]  \n\t" // Read C[8:15, 10]
// " add             x0, x22, x0                     \n\t" // Next column
" add             x6, x22, x6                     \n\t" // Next column
" st1d            z10.d, p0, [x6]                 \n\t" // Write C[0:7, 5]
" st1d            z11.d, p1, [x6, x11, lsl #3]    \n\t" // Write C[8:15, 5]
// " ld1d            z16.d, p0/z, [x0]               \n\t" // Read C[0:7, 11]
// " ld1d            z17.d, p1/z, [x0, x11, lsl #3]  \n\t" // Read C[8:15, 11]
// " add             x0, x22, x0                     \n\t" // Next column
" add             x6, x22, x6                     \n\t" // Next column

" fmad            z0.d, p0/m, z31.d, z18.d        \n\t"
" fmad            z1.d, p1/m, z31.d, z19.d        \n\t"
" fmad            z2.d, p0/m, z31.d, z20.d        \n\t"
" fmad            z3.d, p1/m, z31.d, z21.d        \n\t"
" fmad            z4.d, p0/m, z31.d, z22.d        \n\t"
" fmad            z5.d, p1/m, z31.d, z23.d        \n\t"
"                                                 \n\t"
" fmad            z12.d, p0/m, z31.d, z24.d       \n\t"
" fmad            z13.d, p1/m, z31.d, z25.d       \n\t"
// " fmad            z14.d, p0/m, z31.d, z26.d       \n\t"
// " fmad            z15.d, p1/m, z31.d, z27.d       \n\t"
// " fmad            z16.d, p0/m, z31.d, z28.d       \n\t"
// " fmad            z17.d, p1/m, z31.d, z29.d       \n\t"

" st1d            z0.d, p0, [x6]                  \n\t" // Write C[:, 6]
" st1d            z1.d, p1, [x6, x11, lsl #3]     \n\t"
" add             x6, x22, x6                     \n\t" // Next column
" st1d            z2.d, p0, [x6]                  \n\t" // Write C[:, 7]
" st1d            z3.d, p1, [x6, x11, lsl #3]     \n\t"
" add             x6, x22, x6                     \n\t" // Next column
" st1d            z4.d, p0, [x6]                  \n\t" // Write C[:, 8]
" st1d            z5.d, p1, [x6, x11, lsl #3]     \n\t"
" add             x6, x22, x6                     \n\t" // Next column
"                                                 \n\t"
" st1d            z12.d, p0, [x6]                 \n\t" // Write C[:, 9]
" st1d            z13.d, p1, [x6, x11, lsl #3]    \n\t"
" add             x6, x22, x6                     \n\t" // Next column
// " st1d            z14.d, p0, [x6]                 \n\t" // Write C[:, 10]
// " st1d            z15.d, p1, [x6, x11, lsl #3]    \n\t"
// " add             x6, x22, x6                     \n\t" // Next column
// " st1d            z16.d, p0, [x6]                 \n\t" // Write C[:, 11]
// " st1d            z17.d, p1, [x6, x11, lsl #3]    \n\t"
// " add             x6, x22, x6                     \n\t" // Next column
"                                                 \n\t"
"                                                 \n\t"
" b               END_WRITE_MEM                   \n\t"
"                                                 \n\t"
" CS_CCOL:                                        \n\t" // C has row-strides.
" mul             x23, x20, x12                   \n\t" // Column stride in bytes
" mul             x23, x23, x11                   \n\t" // Vector length in memory
" mul             x22, x7, x12                    \n\t" // Row stride in bytes
" mov             x0, x6                          \n\t" // Clone address for R/W.
" add             x10, x23, x0                    \n\t" // Address C[8:15, 0]
" add             x16, x23, x6                    \n\t" // Address C[8:15, 0]
"                                                 \n\t"
"                                                 \n\t" // Z30: index for loading C columns.
" index           z30.d, xzr, x20                 \n\t" // Generate indices.
"                                                 \n\t"
" ld1d          z0.d, p0/z, [x0, z30.d, lsl #3]   \n\t" // Read C[0:7, 0]
" ld1d          z1.d, p1/z, [x10, z30.d, lsl #3]  \n\t" // Read C[8:15, 0]
" add             x0, x22, x0                     \n\t" // Move to C[:, 1]
" add             x10, x23, x0                    \n\t" // Address C[8:15, 1]
" ld1d          z2.d, p0/z, [x0, z30.d, lsl #3]   \n\t" // Read C[0:7, 1]
" ld1d          z3.d, p1/z, [x10, z30.d, lsl #3]  \n\t" // Read C[8:15, 1]
" add             x0, x22, x0                     \n\t" // Move to C[:, 2]
" add             x10, x23, x0                    \n\t" // Address C[8:15, 2]
" ld1d          z4.d, p0/z, [x0, z30.d, lsl #3]   \n\t" // Read C[0:7, 2]
" ld1d          z5.d, p1/z, [x10, z30.d, lsl #3]  \n\t" // Read C[8:15, 2]
" add             x0, x22, x0                     \n\t" // Move to C[:, 3]
" add             x10, x23, x0                    \n\t" // Address C[8:15, 3]
"                                                 \n\t"
" fmad            z0.d, p0/m, z31.d, z6.d         \n\t" // Z6 used
" fmad            z1.d, p1/m, z31.d, z7.d         \n\t" // Z7 used
" ld1d          z6.d, p0/z, [x0, z30.d, lsl #3]   \n\t" // Read C[0:7, 3]
" ld1d          z7.d, p1/z, [x10, z30.d, lsl #3]  \n\t" // Read C[8:15, 3]
" add             x0, x22, x0                     \n\t" // Move to C[:, 4]
" add             x10, x23, x0                    \n\t" // Address C[8:15, 4]
" fmad            z2.d, p0/m, z31.d, z8.d         \n\t" // Z8 used
" fmad            z3.d, p1/m, z31.d, z9.d         \n\t" // Z9 used
" ld1d          z8.d, p0/z, [x0, z30.d, lsl #3]   \n\t" // Read C[0:7, 4]
" ld1d          z9.d, p1/z, [x10, z30.d, lsl #3]  \n\t" // Read C[8:15, 4]
" add             x0, x22, x0                     \n\t" // Move to C[:, 5]
" add             x10, x23, x0                    \n\t" // Address C[8:15, 5]
" fmad            z4.d, p0/m, z31.d, z10.d        \n\t" // Z10 used
" fmad            z5.d, p1/m, z31.d, z11.d        \n\t" // Z11 used
" ld1d          z10.d, p0/z, [x0, z30.d, lsl #3]  \n\t" // Read C[0:7, 5]
" ld1d          z11.d, p1/z, [x10, z30.d, lsl #3] \n\t" // Read C[8:15, 5]
" add             x0, x22, x0                     \n\t" // Move to C[:, 6]
" add             x10, x23, x0                    \n\t" // Address C[8:15, 6]
"                                                 \n\t"
" st1d            z0.d, p0, [x6, z30.d, lsl #3]   \n\t" // Write C[0:7, 0]
" st1d            z1.d, p1, [x16, z30.d, lsl #3]  \n\t" // Write C[8:15, 0]
" add             x6, x22, x6                     \n\t" // Move to C[:, 1]
" add             x16, x23, x6                    \n\t" // Address C[8:15, 1]
" st1d            z2.d, p0, [x6, z30.d, lsl #3]   \n\t" // Write C[0:7, 1]
" st1d            z3.d, p1, [x16, z30.d, lsl #3]  \n\t" // Write C[8:15, 1]
" add             x6, x22, x6                     \n\t" // Move to C[:, 2]
" add             x16, x23, x6                    \n\t" // Address C[8:15, 2]
" st1d            z4.d, p0, [x6, z30.d, lsl #3]   \n\t" // Write C[0:7, 2]
" st1d            z5.d, p1, [x16, z30.d, lsl #3]  \n\t" // Write C[8:15, 2]
" add             x6, x22, x6                     \n\t" // Move to C[:, 3]
" add             x16, x23, x6                    \n\t" // Address C[8:15, 3]
"                                                 \n\t"
" fmad            z6.d, p0/m, z31.d, z12.d        \n\t" // Z12 used
" fmad            z7.d, p1/m, z31.d, z13.d        \n\t" // Z13 used
" ld1d          z0.d, p0/z, [x0, z30.d, lsl #3]   \n\t" // Read C[0:7, 6]
" ld1d          z1.d, p1/z, [x10, z30.d, lsl #3]  \n\t" // Read C[8:15, 6]
" add             x0, x22, x0                     \n\t" // Move to C[:, 7]
" add             x10, x23, x0                    \n\t" // Address C[8:15, 7]
" fmad            z8.d, p0/m, z31.d, z14.d        \n\t" // Z14 used
" fmad            z9.d, p1/m, z31.d, z15.d        \n\t" // Z15 used
" ld1d          z2.d, p0/z, [x0, z30.d, lsl #3]   \n\t" // Read C[0:7, 7]
" ld1d          z3.d, p1/z, [x10, z30.d, lsl #3]  \n\t" // Read C[8:15, 7]
" add             x0, x22, x0                     \n\t" // Move to C[:, 8]
" add             x10, x23, x0                    \n\t" // Address C[8:15, 8]
" fmad            z10.d, p0/m, z31.d, z16.d       \n\t" // Z16 used
" fmad            z11.d, p1/m, z31.d, z17.d       \n\t" // Z17 used
" ld1d          z4.d, p0/z, [x0, z30.d, lsl #3]   \n\t" // Read C[0:7, 8]
" ld1d          z5.d, p1/z, [x10, z30.d, lsl #3]  \n\t" // Read C[8:15, 8]
" add             x0, x22, x0                     \n\t" // Move to C[:, 9]
" add             x10, x23, x0                    \n\t" // Address C[8:15, 9]
"                                                 \n\t"
" st1d            z6.d, p0, [x6, z30.d, lsl #3]   \n\t" // Write C[0:7, 3]
" st1d            z7.d, p1, [x16, z30.d, lsl #3]  \n\t" // Write C[8:15, 3]
" add             x6, x22, x6                     \n\t" // Move to C[:, 4]
" add             x16, x23, x6                    \n\t" // Address C[8:15, 4]
" ld1d          z12.d, p0/z, [x0, z30.d, lsl #3]  \n\t" // Read C[0:7, 9]
" ld1d          z13.d, p1/z, [x10, z30.d, lsl #3] \n\t" // Read C[8:15, 9]
" add             x0, x22, x0                     \n\t" // Move to C[:, 10]
" add             x10, x23, x0                    \n\t" // Address C[8:15, 10]
" st1d            z8.d, p0, [x6, z30.d, lsl #3]   \n\t" // Write C[0:7, 4]
" st1d            z9.d, p1, [x16, z30.d, lsl #3]  \n\t" // Write C[8:15, 4]
" add             x6, x22, x6                     \n\t" // Move to C[:, 5]
" add             x16, x23, x6                    \n\t" // Address C[8:15, 5]
// " ld1d          z14.d, p0/z, [x0, z30.d, lsl #3]  \n\t" // Read C[0:7, 10]
// " ld1d          z15.d, p1/z, [x10, z30.d, lsl #3] \n\t" // Read C[8:15, 10]
// " add             x0, x22, x0                     \n\t" // Move to C[:, 11]
// " add             x10, x23, x0                    \n\t" // Address C[8:15, 11]
" st1d            z10.d, p0, [x6, z30.d, lsl #3]  \n\t" // Write C[0:7, 5]
" st1d            z11.d, p1, [x16, z30.d, lsl #3] \n\t" // Write C[8:15, 5]
" add             x6, x22, x6                     \n\t" // Move to C[:, 6]
" add             x16, x23, x6                    \n\t" // Address C[8:15, 6]
// " ld1d          z16.d, p0/z, [x0, z30.d, lsl #3]  \n\t" // Read C[0:7, 11]
// " ld1d          z17.d, p1/z, [x10, z30.d, lsl #3] \n\t" // Read C[8:15, 11]
"                                                 \n\t"
" fmad            z0.d, p0/m, z31.d, z18.d        \n\t" // Z18 used
" fmad            z1.d, p1/m, z31.d, z19.d        \n\t" // Z19 used
" fmad            z2.d, p0/m, z31.d, z20.d        \n\t" // Z20 used
" fmad            z3.d, p1/m, z31.d, z21.d        \n\t" // Z21 used
" fmad            z4.d, p0/m, z31.d, z22.d        \n\t" // Z22 used
" fmad            z5.d, p1/m, z31.d, z23.d        \n\t" // Z23 used
"                                                 \n\t"
" fmad            z12.d, p0/m, z31.d, z24.d       \n\t" // Z24 used
" fmad            z13.d, p1/m, z31.d, z25.d       \n\t" // Z25 used
// " fmad            z14.d, p0/m, z31.d, z26.d       \n\t" // Z26 used
// " fmad            z15.d, p1/m, z31.d, z27.d       \n\t" // Z27 used
// " fmad            z16.d, p0/m, z31.d, z28.d       \n\t" // Z28 used
// " fmad            z17.d, p1/m, z31.d, z29.d       \n\t" // Z29 used
"                                                 \n\t"
" st1d            z0.d, p0, [x6, z30.d, lsl #3]   \n\t" // Write C[0:7, 6]
" st1d            z1.d, p1, [x16, z30.d, lsl #3]  \n\t" // Write C[8:15, 6]
" add             x6, x22, x6                     \n\t" // Move to C[:, 7]
" add             x16, x23, x6                    \n\t" // Address C[8:15, 7]
" st1d            z2.d, p0, [x6, z30.d, lsl #3]   \n\t" // Write C[0:7, 8]
" st1d            z3.d, p1, [x16, z30.d, lsl #3]  \n\t" // Write C[8:15, 8]
" add             x6, x22, x6                     \n\t" // Move to C[:, 8]
" add             x16, x23, x6                    \n\t" // Address C[8:15, 8]
" st1d            z4.d, p0, [x6, z30.d, lsl #3]   \n\t" // Write C[0:7, 8]
" st1d            z5.d, p1, [x16, z30.d, lsl #3]  \n\t" // Write C[8:15, 8]
" add             x6, x22, x6                     \n\t" // Move to C[:, 9]
" add             x16, x23, x6                    \n\t" // Address C[8:15, 9]
"                                                 \n\t"
" st1d            z12.d, p0, [x6, z30.d, lsl #3]  \n\t" // Write C[0:7, 9]
" st1d            z13.d, p1, [x16, z30.d, lsl #3] \n\t" // Write C[8:15, 9]
" add             x6, x22, x6                     \n\t" // Move to C[:, 10]
" add             x16, x23, x6                    \n\t" // Address C[8:15, 10]
// " st1d            z14.d, p0, [x6, z30.d, lsl #3]  \n\t" // Write C[0:7, 10]
// " st1d            z15.d, p1, [x16, z30.d, lsl #3] \n\t" // Write C[8:15, 10]
// " add             x6, x22, x6                     \n\t" // Move to C[:, 11]
// " add             x16, x23, x6                    \n\t" // Address C[8:15, 11]
// " st1d            z16.d, p0, [x6, z30.d, lsl #3]  \n\t" // Write C[0:7, 11]
// " st1d            z17.d, p1, [x16, z30.d, lsl #3] \n\t" // Write C[8:15, 11]
"                                                 \n\t"
"                                                 \n\t"
" END_WRITE_MEM:                                  \n\t" // End of computation.
" mov             x0, #0                          \n\t" // Return normal.
" b               END_EXEC                        \n\t"
" END_ERROR:                                      \n\t"
" mov             x0, #1                          \n\t" // Return error.
" END_EXEC:                                       \n\t"
:// output operands (none)
:// input operands
 [m_loc]  "m" (m_loc),  // 0
 [aaddr]  "m" (aaddr),  // 1
 [baddr]  "m" (baddr),  // 2
 [caddr]  "m" (caddr),  // 3
 [k_mker] "m" (k_mker), // 4
 [k_left] "m" (k_left), // 5
 [alpha]  "m" (alpha),  // 6
 [beta]   "m" (beta),   // 7
 [rs_c]   "m" (rs_c),   // 8
 [cs_c]   "m" (cs_c),   // 9
 [cs_a]   "m" (cs_a),   // 10
 [rs_b]   "m" (rs_b),   // 11
 [cs_b]   "m" (cs_b),   // 12
 [ps_b]   "m" (ps_b),   // 13
 [a_next] "m" (a_next), // 14
 [b_next] "m" (b_next)  // 15
:// Register clobber list
 "x0","x1","x2","x3","x4","x5","x6","x7","x8",
 "x9","x10","x11","x12","x14","x15",
 "x16","x17","x18","x19","x20","x21","x22","x23",
 "x24","x25", "x26", "x27", ///< Additional registers used when this kernel becomes a sup.
 "z0","z1","z2","z3","z4","z5","z6","z7",
 "z8","z9","z10","z11","z12","z13","z14","z15",
 "z16","z17","z18","z19",
 "z20","z21","z22","z23",
 "z24","z25","z26","z27",
 "z28","z29","z30","z31"
); }

