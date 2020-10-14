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
#include "blis.h"

# define DGEMM_2VX12_MKER_LOOP_PLAIN(C0FH,C1FH,C2FH,C3FH,C4FH,C5FH,C6FH,C7FH,C8FH,C9FH,C10FH,C11FH,C0LH,C1LH,C2LH,C3LH,C4LH,C5LH,C6LH,C7LH,C8LH,C9LH,C10LH,C11LH) \
" madd            x2, x3, x12, x2                 \n\t" /* A address forward */ \
" fmla       "#C0FH".d, p0/m, z30.d, z0.d         \n\t" /* Row 1:8 column 0  */ \
" fmla       "#C1FH".d, p0/m, z30.d, z1.d         \n\t" /* Row 1:8 column 1  */ \
" fmla       "#C2FH".d, p0/m, z30.d, z2.d         \n\t" /* Row 1:8 column 2  */ \
" fmla       "#C3FH".d, p0/m, z30.d, z3.d         \n\t" /* Row 1:8 column 3  */ \
" fmla       "#C4FH".d, p0/m, z30.d, z4.d         \n\t" /* Row 1:8 column 4  */ \
" fmla       "#C5FH".d, p0/m, z30.d, z5.d         \n\t" /* Row 1:8 column 5  */ \
" fmla       "#C0LH".d, p0/m, z31.d, z0.d         \n\t" /* Row 9:15 column 0 */ \
" ld1rd           z0.d, p0/z, [x4, #48]           \n\t" /* Load B[6, j]      */ \
" fmla       "#C1LH".d, p0/m, z31.d, z1.d         \n\t" /* Row 9:15 column 1 */ \
" ld1rd           z1.d, p0/z, [x4, #56]           \n\t" /* Load B[7, j]      */ \
" fmla       "#C2LH".d, p0/m, z31.d, z2.d         \n\t" /* Row 9:15 column 2 */ \
" ld1rd           z2.d, p0/z, [x4, #64]           \n\t" /* Load B[8, j]      */ \
" fmla       "#C3LH".d, p0/m, z31.d, z3.d         \n\t" /* Row 9:15 column 3 */ \
" ld1rd           z3.d, p0/z, [x4, #72]           \n\t" /* Load B[9, j]      */ \
" fmla       "#C4LH".d, p0/m, z31.d, z4.d         \n\t" /* Row 9:15 column 4 */ \
" ld1rd           z4.d, p0/z, [x4, #80]           \n\t" /* Load B[10, j]     */ \
" fmla       "#C5LH".d, p0/m, z31.d, z5.d         \n\t" /* Row 9:15 column 5 */ \
" ld1rd           z5.d, p0/z, [x4, #88]           \n\t" /* Load B[11, j]     */ \
\
" madd            x4, x5, x12, x4                 \n\t" /* B address forward */ \
" fmla       "#C6FH".d, p0/m, z30.d, z0.d        \n\t" /* Row 1:8 column 6  */ \
" fmla       "#C7FH".d, p0/m, z30.d, z1.d        \n\t" /* Row 1:8 column 7  */ \
" fmla       "#C8FH".d, p0/m, z30.d, z2.d        \n\t" /* Row 1:8 column 8  */ \
" fmla       "#C9FH".d, p0/m, z30.d, z3.d        \n\t" /* Row 1:8 column 9  */ \
" fmla      "#C10FH".d, p0/m, z30.d, z4.d        \n\t" /* Row 1:8 column 10 */ \
" fmla      "#C11FH".d, p0/m, z30.d, z5.d        \n\t" /* Row 1:8 column 11 */ \
" ld1d            z30.d, p0/z, [x2]               \n\t" /* Load next A column (first half) */ \
" fmla       "#C6LH".d, p0/m, z31.d, z0.d        \n\t" /* Row 9:15 column 6  */ \
" ld1rd           z0.d, p0/z, [x4, #0]            \n\t" /* Load B[0, j+1]     */ \
" fmla       "#C7LH".d, p0/m, z31.d, z1.d        \n\t" /* Row 9:15 column 7  */ \
" ld1rd           z1.d, p0/z, [x4, #8]            \n\t" /* Load B[1, j+1]     */ \
" fmla       "#C8LH".d, p0/m, z31.d, z2.d        \n\t" /* Row 9:15 column 8  */ \
" ld1rd           z2.d, p0/z, [x4, #16]           \n\t" /* Load B[2, j+1]     */ \
" fmla       "#C9LH".d, p0/m, z31.d, z3.d        \n\t" /* Row 9:15 column 9  */ \
" ld1rd           z3.d, p0/z, [x4, #24]           \n\t" /* Load B[3, j+1]     */ \
" fmla      "#C10LH".d, p0/m, z31.d, z4.d        \n\t" /* Row 9:15 column 10 */ \
" ld1rd           z4.d, p0/z, [x4, #32]           \n\t" /* Load B[4, j+1]     */ \
" fmla      "#C11LH".d, p0/m, z31.d, z5.d        \n\t" /* Row 9:15 column 11 */ \
" ld1rd           z5.d, p0/z, [x4, #40]           \n\t" /* Load B[5, j+1]     */ \
" ld1d            z31.d, p1/z, [x2, x11, lsl 3]   \n\t" /* Load next A column (last half) */ \
  /* End of Microkernel loop. */

# define DGEMM_2VX12_MKER_LOOP_PREFETCH_CCOL_6X(C0FH,C1FH,C2FH,C3FH,C4FH,C5FH,C6FH,C7FH,C8FH,C9FH,C10FH,C11FH,C0LH,C1LH,C2LH,C3LH,C4LH,C5LH,C6LH,C7LH,C8LH,C9LH,C10LH,C11LH,ADDRC,LDC) \
" madd            x2, x3, x12, x2                 \n\t" /* A address forward */ \
" fmla       "#C0FH".d, p0/m, z30.d, z0.d         \n\t" /* Row 1:8 column 0  */ \
" prfm            PSTL1STRM, ["#ADDRC"]           \n\t" /* Prefetch C column 0:7 */ \
" fmla       "#C1FH".d, p0/m, z30.d, z1.d         \n\t" /* Row 1:8 column 1  */ \
" prfm            PSTL1STRM, ["#ADDRC", #64]      \n\t" /* Prefetch C column 8:15 */ \
" add             "#ADDRC", "#ADDRC", "#LDC"      \n\t" /* C column forward */ \
" fmla       "#C2FH".d, p0/m, z30.d, z2.d         \n\t" /* Row 1:8 column 2  */ \
" prfm            PSTL1STRM, ["#ADDRC"]           \n\t" /* Prefetch C column 0:7 */ \
" fmla       "#C3FH".d, p0/m, z30.d, z3.d         \n\t" /* Row 1:8 column 3  */ \
" prfm            PSTL1STRM, ["#ADDRC", #64]      \n\t" /* Prefetch C column 8:15 */ \
" add             "#ADDRC", "#ADDRC", "#LDC"      \n\t" /* C column forward */ \
" fmla       "#C4FH".d, p0/m, z30.d, z4.d         \n\t" /* Row 1:8 column 4  */ \
" prfm            PSTL1STRM, ["#ADDRC"]           \n\t" /* Prefetch C column 0:7 */ \
" fmla       "#C5FH".d, p0/m, z30.d, z5.d         \n\t" /* Row 1:8 column 5  */ \
" prfm            PSTL1STRM, ["#ADDRC", #64]      \n\t" /* Prefetch C column 8:15 */ \
" add             "#ADDRC", "#ADDRC", "#LDC"      \n\t" /* C column forward */ \
" fmla       "#C0LH".d, p0/m, z31.d, z0.d         \n\t" /* Row 9:15 column 0 */ \
" ld1rd           z0.d, p0/z, [x4, #48]           \n\t" /* Load B[6, j]      */ \
" fmla       "#C1LH".d, p0/m, z31.d, z1.d         \n\t" /* Row 9:15 column 1 */ \
" ld1rd           z1.d, p0/z, [x4, #56]           \n\t" /* Load B[7, j]      */ \
" fmla       "#C2LH".d, p0/m, z31.d, z2.d         \n\t" /* Row 9:15 column 2 */ \
" ld1rd           z2.d, p0/z, [x4, #64]           \n\t" /* Load B[8, j]      */ \
" fmla       "#C3LH".d, p0/m, z31.d, z3.d         \n\t" /* Row 9:15 column 3 */ \
" ld1rd           z3.d, p0/z, [x4, #72]           \n\t" /* Load B[9, j]      */ \
" fmla       "#C4LH".d, p0/m, z31.d, z4.d         \n\t" /* Row 9:15 column 4 */ \
" ld1rd           z4.d, p0/z, [x4, #80]           \n\t" /* Load B[10, j]     */ \
" fmla       "#C5LH".d, p0/m, z31.d, z5.d         \n\t" /* Row 9:15 column 5 */ \
" ld1rd           z5.d, p0/z, [x4, #88]           \n\t" /* Load B[11, j]     */ \
\
" madd            x4, x5, x12, x4                 \n\t" /* B address forward */ \
" fmla       "#C6FH".d, p0/m, z30.d, z0.d        \n\t" /* Row 1:8 column 6  */ \
" prfm            PSTL1STRM, ["#ADDRC"]           \n\t" /* Prefetch C column 0:7 */ \
" fmla       "#C7FH".d, p0/m, z30.d, z1.d        \n\t" /* Row 1:8 column 7  */ \
" prfm            PSTL1STRM, ["#ADDRC", #64]      \n\t" /* Prefetch C column 8:15 */ \
" add             "#ADDRC", "#ADDRC", "#LDC"      \n\t" /* C column forward */ \
" fmla       "#C8FH".d, p0/m, z30.d, z2.d        \n\t" /* Row 1:8 column 8  */ \
" prfm            PSTL1STRM, ["#ADDRC"]           \n\t" /* Prefetch C column 0:7 */ \
" fmla       "#C9FH".d, p0/m, z30.d, z3.d        \n\t" /* Row 1:8 column 9  */ \
" prfm            PSTL1STRM, ["#ADDRC", #64]      \n\t" /* Prefetch C column 8:15 */ \
" add             "#ADDRC", "#ADDRC", "#LDC"      \n\t" /* C column forward */ \
" fmla      "#C10FH".d, p0/m, z30.d, z4.d        \n\t" /* Row 1:8 column 10 */ \
" prfm            PSTL1STRM, ["#ADDRC"]           \n\t" /* Prefetch C column 0:7 */ \
" fmla      "#C11FH".d, p0/m, z30.d, z5.d        \n\t" /* Row 1:8 column 11 */ \
" prfm            PSTL1STRM, ["#ADDRC", #64]      \n\t" /* Prefetch C column 8:15 */ \
" add             "#ADDRC", "#ADDRC", "#LDC"      \n\t" /* C column forward */ \
" ld1d            z30.d, p0/z, [x2]               \n\t" /* Load next A column (first half) */ \
" fmla       "#C6LH".d, p0/m, z31.d, z0.d        \n\t" /* Row 9:15 column 6  */ \
" ld1rd           z0.d, p0/z, [x4, #0]            \n\t" /* Load B[0, j+1]     */ \
" fmla       "#C7LH".d, p0/m, z31.d, z1.d        \n\t" /* Row 9:15 column 7  */ \
" ld1rd           z1.d, p0/z, [x4, #8]            \n\t" /* Load B[1, j+1]     */ \
" fmla       "#C8LH".d, p0/m, z31.d, z2.d        \n\t" /* Row 9:15 column 8  */ \
" ld1rd           z2.d, p0/z, [x4, #16]           \n\t" /* Load B[2, j+1]     */ \
" fmla       "#C9LH".d, p0/m, z31.d, z3.d        \n\t" /* Row 9:15 column 9  */ \
" ld1rd           z3.d, p0/z, [x4, #24]           \n\t" /* Load B[3, j+1]     */ \
" fmla      "#C10LH".d, p0/m, z31.d, z4.d        \n\t" /* Row 9:15 column 10 */ \
" ld1rd           z4.d, p0/z, [x4, #32]           \n\t" /* Load B[4, j+1]     */ \
" fmla      "#C11LH".d, p0/m, z31.d, z5.d        \n\t" /* Row 9:15 column 11 */ \
" ld1rd           z5.d, p0/z, [x4, #40]           \n\t" /* Load B[5, j+1]     */ \
" ld1d            z31.d, p1/z, [x2, x11, lsl 3]   \n\t" /* Load next A column (last half) */ \
  /* End of Microkernel loop + prefetch 6 colums of C. */


/*
   o 16x12 double precision micro-kernel
   o This implementation uses unindexed FMLA instructions in its kernel.
   o Runnable on ARMv8a with SVE 512 feature compiled with aarch64 GCC.
   o Tested on armie for SVE.
   o Tested & benchmarked on A64fx:
    - Single threaded GEMM( M=2000 N=1400 K=500 ) yields around 45GFlOps.

   Sept. 2020.
*/
void bli_dgemm_armsve512_asm_16x12_unindexed
     (
       dim_t               k0,
       double*    restrict alpha,
       double*    restrict a,
       double*    restrict b,
       double*    restrict beta,
       double*    restrict c, inc_t rs_c0, inc_t cs_c0,
       auxinfo_t* restrict data,
       cntx_t*    restrict cntx
     )
{
  void* a_next = bli_auxinfo_next_a( data );
  void* b_next = bli_auxinfo_next_b( data );

  // Typecast local copies of integers in case dim_t and inc_t are a
  // different size than is expected by load instructions.
  uint64_t k_mker = k0 / 4;
  uint64_t k_left = k0 % 4;
  uint64_t rs_c   = rs_c0;
  uint64_t cs_c   = cs_c0;

__asm__ volatile (
" mov             x9, #16                         \n\t" // Shape M, can be input
"                                                 \n\t" // Shape N is fixed to be 12
" ldr             x8, %[k_left]                   \n\t" // Shape K to be contracted
" ldr             x21, %[k_mker]                  \n\t" // Size of K-microblock
"                                                 \n\t"
" ldr             x20, %[rs_c]                    \n\t" // Row-skip of C
" ldr             x6, %[caddr]                    \n\t" // Load address of C
" ldr             x7, %[cs_c]                     \n\t" // LdC, which is called column-skip in BLIS
" cmp             x20, #1                         \n\t"
"                                                 \n\t"
" ldr             x0, %[alpha]                    \n\t" // Alpha address
" ldr             x1, %[beta]                     \n\t" // Beta address
"                                                 \n\t"
" ldr             x2, %[aaddr]                    \n\t" // Load address of A
" mov             x3, #16                         \n\t" // LdA is 16 from packing, can be input
" ldr             x4, %[baddr]                    \n\t" // Load address of B
" mov             x5, #12                         \n\t" // LdB is 12 from packing, can be input
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
" add             x15, x14, x22                   \n\t" // C column 5
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
" prfm            PSTL2STRM, [x15]                \n\t" // Prefetch C column 5
" prfm            PSTL2STRM, [x15,#64]            \n\t"
"                                                 \n\t"
" add             x10, x15, x22                   \n\t" // C column 6
" add             x11, x10, x22                   \n\t" // C column 7
" add             x12, x11, x22                   \n\t" // C column 8
" add             x13, x12, x22                   \n\t" // C column 9
" add             x14, x13, x22                   \n\t" // C column 10
" add             x15, x14, x22                   \n\t" // C column 11
" prfm            PSTL2STRM, [x10]                \n\t" // Prefetch C column 6
" prfm            PSTL2STRM, [x10,#64]            \n\t"
" prfm            PSTL2STRM, [x11]                \n\t" // Prefetch C column 7
" prfm            PSTL2STRM, [x11,#64]            \n\t"
" prfm            PSTL2STRM, [x12]                \n\t" // Prefetch C column 8
" prfm            PSTL2STRM, [x12,#64]            \n\t"
" prfm            PSTL2STRM, [x13]                \n\t" // Prefetch C column 9
" prfm            PSTL2STRM, [x13,#64]            \n\t"
" prfm            PSTL2STRM, [x14]                \n\t" // Prefetch C column 10
" prfm            PSTL2STRM, [x14,#64]            \n\t"
" prfm            PSTL2STRM, [x15]                \n\t" // Prefetch C column 11
" prfm            PSTL2STRM, [x15,#64]            \n\t"
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
" add             x12, x10, x22                   \n\t" // 0:7  Column 10
" add             x13, x12, x23                   \n\t" // 8:16 Column 10
" add             x14, x12, x22                   \n\t" // 0:7  Column 11
" add             x15, x14, x23                   \n\t" // 8:16 Column 11
" prfd        PSTL2STRM, p0, [x10, z30.d, lsl #3] \n\t" // Prefetch C column 9
" prfd        PSTL2STRM, p0, [x11, z30.d, lsl #3] \n\t"
" prfd        PSTL2STRM, p0, [x12, z30.d, lsl #3] \n\t" // Prefetch C column 10
" prfd        PSTL2STRM, p0, [x13, z30.d, lsl #3] \n\t"
" prfd        PSTL2STRM, p0, [x14, z30.d, lsl #3] \n\t" // Prefetch C column 11
" prfd        PSTL2STRM, p0, [x15, z30.d, lsl #3] \n\t"
"                                                 \n\t"
" b               END_C_PRFML2                    \n\t"
"                                                 \n\t"
" END_C_PRFML2:                                   \n\t"
"                                                 \n\t"
" ldr             x18, %[a_next]                  \n\t" // Pointer to next A pack
" ldr             x19, %[b_next]                  \n\t" // Pointer to next B pack
" mov             x12, #8                         \n\t" // Double in bytes
"                                                 \n\t"
" mov             x11, xzr                        \n\t"
" incd            x11                             \n\t" // Determine vector length, in doubles.
"                                                 \n\t"
" ptrue           p0.d, all                       \n\t" // First half is all-true.
" whilelo         p1.d, x11, x9                   \n\t" // Second half from M argument.
" fmov            d0, #1.0                        \n\t" // Exact floating-point 1.0.
" fmov            x14, d0                         \n\t" // Hard float to avoid conflict with SVE.
"                                                 \n\t"
" prfm            PLDL1STRM, [x2]                 \n\t" // Prefetch A for first microkernel
" prfm            PLDL1STRM, [x2, #64]            \n\t"
" prfm            PLDL1STRM, [x2, #128]           \n\t"
// " prfm            PLDL1STRM, [x2, #192]           \n\t"
// " prfm            PLDL1STRM, [x2, #256]           \n\t"
// " prfm            PLDL1STRM, [x2, #320]           \n\t"
// " prfm            PLDL1STRM, [x2, #384]           \n\t"
// " prfm            PLDL1STRM, [x2, #448]           \n\t"
" prfm            PLDL1STRM, [x4]                 \n\t" // Prefetch B for first microkernel
" prfm            PLDL1STRM, [x4, #64]            \n\t"
" prfm            PLDL1STRM, [x4, #128]           \n\t"
// " prfm            PLDL1STRM, [x4, #192]           \n\t"
// " prfm            PLDL1STRM, [x4, #256]           \n\t"
// " prfm            PLDL1STRM, [x4, #320]           \n\t"
"                                                 \n\t"
"                                                 \n\t" // SVE Register configuration:
"                                                 \n\t" // Z[30-31]: A columns
"                                                 \n\t" // Z[0-5]: B elements broadcasted
"                                                 \n\t" // Z[6-29]: C change buffer
"                                                 \n\t"
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
" FIRST_BCOL:                                     \n\t" // Load B[0:5, 0].
" ld1rd           z0.d, p0/z, [x4, #0]            \n\t"
" ld1rd           z1.d, p0/z, [x4, #8]            \n\t"
" ld1rd           z2.d, p0/z, [x4, #16]           \n\t"
" ld1rd           z3.d, p0/z, [x4, #24]           \n\t"
" ld1rd           z4.d, p0/z, [x4, #32]           \n\t"
" ld1rd           z5.d, p0/z, [x4, #40]           \n\t"
" FIRST_ACOL:                                     \n\t" // Load of first column of A.
" ld1d            z30.d, p0/z, [x2]               \n\t"
" ld1d            z31.d, p1/z, [x2, x11, lsl 3]   \n\t"
"                                                 \n\t"
" fmov            z26.d, p0/m, #0.0               \n\t"
" fmov            z27.d, p0/m, #0.0               \n\t"
" fmov            z28.d, p0/m, #0.0               \n\t"
" fmov            z29.d, p0/m, #0.0               \n\t"
"                                                 \n\t"
" cmp             x21, #0                         \n\t" // If no 4-microkernel can be applied
" b.eq            K_LEFT_LOOP                     \n\t"
"                                                 \n\t"
" K_MKER_LOOP:                                    \n\t" // Unroll the 4-loop.
" madd            x22, x21, x20, xzr              \n\t"
" cmp             x22, #1                         \n\t"
" b.eq            K_MKER_LOOP_FINAL               \n\t"
"                                                 \n\t"
"                                                 \n\t"
DGEMM_2VX12_MKER_LOOP_PLAIN(z6,z8,z10,z12,z14,z16,z18,z20,z22,z24,z26,z28,z7,z9,z11,z13,z15,z17,z19,z21,z23,z25,z27,z29)
DGEMM_2VX12_MKER_LOOP_PLAIN(z6,z8,z10,z12,z14,z16,z18,z20,z22,z24,z26,z28,z7,z9,z11,z13,z15,z17,z19,z21,z23,z25,z27,z29)
"                                                 \n\t"
// " prfm            PLDL1STRM, [x2, #384]           \n\t" // [NO_REPEAT] Prefetch A
// " prfm            PLDL1STRM, [x2, #448]           \n\t" // [NO_REPEAT] Prefetch A
// " prfm            PLDL1STRM, [x2, #512]           \n\t" // [NO_REPEAT] Prefetch A
// " prfm            PLDL1STRM, [x2, #576]           \n\t" // [NO_REPEAT] Prefetch A
// " prfm            PLDL1STRM, [x2, #640]           \n\t" // [NO_REPEAT] Prefetch A
// " prfm            PLDL1STRM, [x2, #704]           \n\t" // [NO_REPEAT] Prefetch A
// " prfm            PLDL1STRM, [x2, #768]           \n\t" // [NO_REPEAT] Prefetch A
// " prfm            PLDL1STRM, [x2, #832]           \n\t" // [NO_REPEAT] Prefetch A
// "                                                 \n\t"
// " prfm            PLDL1STRM, [x4, #192]           \n\t" // [NO_REPEAT] Prefetch B
// " prfm            PLDL1STRM, [x4, #256]           \n\t" // [NO_REPEAT] Prefetch B
// " prfm            PLDL1STRM, [x4, #320]           \n\t" // [NO_REPEAT] Prefetch B
// " prfm            PLDL1STRM, [x4, #384]           \n\t" // [NO_REPEAT] Prefetch B
// " prfm            PLDL1STRM, [x4, #448]           \n\t" // [NO_REPEAT] Prefetch B
// " prfm            PLDL1STRM, [x4, #512]           \n\t" // [NO_REPEAT] Prefetch B
"                                                 \n\t"
"                                                 \n\t"
DGEMM_2VX12_MKER_LOOP_PLAIN(z6,z8,z10,z12,z14,z16,z18,z20,z22,z24,z26,z28,z7,z9,z11,z13,z15,z17,z19,z21,z23,z25,z27,z29)
"                                                 \n\t"
" sub             x10, x21, #1                    \n\t" // Before final replica,
" adds            x10, x10, x8                    \n\t" //  check if this iteration is final
" b.eq            FIN_LOOP                        \n\t"
"                                                 \n\t"
DGEMM_2VX12_MKER_LOOP_PLAIN(z6,z8,z10,z12,z14,z16,z18,z20,z22,z24,z26,z28,z7,z9,z11,z13,z15,z17,z19,z21,z23,z25,z27,z29)
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
DGEMM_2VX12_MKER_LOOP_PREFETCH_CCOL_6X(z6,z8,z10,z12,z14,z16,z18,z20,z22,z24,z26,z28,z7,z9,z11,z13,z15,z17,z19,z21,z23,z25,z27,z29,x22,x23)
"                                                 \n\t"
DGEMM_2VX12_MKER_LOOP_PREFETCH_CCOL_6X(z6,z8,z10,z12,z14,z16,z18,z20,z22,z24,z26,z28,z7,z9,z11,z13,z15,z17,z19,z21,z23,z25,z27,z29,x22,x23)
"                                                 \n\t"
DGEMM_2VX12_MKER_LOOP_PLAIN(z6,z8,z10,z12,z14,z16,z18,z20,z22,z24,z26,z28,z7,z9,z11,z13,z15,z17,z19,z21,z23,z25,z27,z29)
"                                                 \n\t"
" sub             x10, x21, #1                    \n\t" // Before final replica,
" adds            x10, x10, x8                    \n\t" //  check if this iteration is final
" b.eq            FIN_LOOP                        \n\t"
"                                                 \n\t"
DGEMM_2VX12_MKER_LOOP_PLAIN(z6,z8,z10,z12,z14,z16,z18,z20,z22,z24,z26,z28,z7,z9,z11,z13,z15,z17,z19,z21,z23,z25,z27,z29)
"                                                 \n\t"
" subs            x21, x21, #1                    \n\t" // Decrease counter.
"                                                 \n\t" // One more microkernel for looping over
"                                                 \n\t" //   non-mker k values.
"                                                 \n\t"
" K_LEFT_LOOP:                                    \n\t"
"                                                 \n\t"
" cmp             x8, #0                          \n\t" // Spetial handler only for k0 == 0.
" b.eq            WRITE_MEM                       \n\t"
"                                                 \n\t"
" cmp             x8, #1                          \n\t" // If K=1.
" b.eq            FIN_LOOP                        \n\t"
"                                                 \n\t"
"                                                 \n\t"
DGEMM_2VX12_MKER_LOOP_PLAIN(z6,z8,z10,z12,z14,z16,z18,z20,z22,z24,z26,z28,z7,z9,z11,z13,z15,z17,z19,z21,z23,z25,z27,z29)
"                                                 \n\t"
" NEXT_ROW:                                       \n\t"
" sub             x8, x8, #1                      \n\t"
" cmp             x8, #1                          \n\t"
" b.ne            K_LEFT_LOOP                     \n\t" // Next column / row.
"                                                 \n\t"
" FIN_LOOP:                                       \n\t" // Final K-loop
"                                                 \n\t"
"                                                 \n\t" // Final A & B[0:5, end] are already loaded
" fmla            z6.d, p0/m, z30.d, z0.d         \n\t" // Row 1:8 column 0 and 1
" fmla            z8.d, p0/m, z30.d, z1.d         \n\t"
" fmla            z10.d, p0/m, z30.d, z2.d        \n\t" // Row 1:8 column 2 and 3
" fmla            z12.d, p0/m, z30.d, z3.d        \n\t"
" fmla            z14.d, p0/m, z30.d, z4.d        \n\t" // Row 1:8 column 4 and 5
" fmla            z16.d, p0/m, z30.d, z5.d        \n\t"
" fmla            z7.d, p0/m, z31.d, z0.d         \n\t" // Row 9:15 column 0 and 1
" ld1rd           z0.d, p0/z, [x4, #48]           \n\t" // Load B[6, end]
" fmla            z9.d, p0/m, z31.d, z1.d         \n\t"
" ld1rd           z1.d, p0/z, [x4, #56]           \n\t" // Load B[7, end]
" fmla            z11.d, p0/m, z31.d, z2.d        \n\t" // Row 9:15 column 2 and 3
" ld1rd           z2.d, p0/z, [x4, #64]           \n\t" // Load B[8, end]
" fmla            z13.d, p0/m, z31.d, z3.d        \n\t"
" ld1rd           z3.d, p0/z, [x4, #72]           \n\t" // Load B[9, end]
" fmla            z15.d, p0/m, z31.d, z4.d        \n\t" // Row 9:15 column 4 and 5
" ld1rd           z4.d, p0/z, [x4, #80]           \n\t" // Load B[10, end]
" fmla            z17.d, p0/m, z31.d, z5.d        \n\t"
" ld1rd           z5.d, p0/z, [x4, #88]           \n\t" // Load B[11, end]
"                                                 \n\t"
" fmla            z18.d, p0/m, z30.d, z0.d        \n\t" // Row 1:8 column 6 and 7
" fmla            z20.d, p0/m, z30.d, z1.d        \n\t"
" fmla            z22.d, p0/m, z30.d, z2.d        \n\t" // Row 1:8 column 8 and 9
" fmla            z24.d, p0/m, z30.d, z3.d        \n\t"
" fmla            z26.d, p0/m, z30.d, z4.d        \n\t" // Row 1:8 column 10 and 11
" fmla            z28.d, p0/m, z30.d, z5.d        \n\t"
" fmla            z19.d, p0/m, z31.d, z0.d        \n\t" // Row 9:15 column 6 and 7
" fmla            z21.d, p0/m, z31.d, z1.d        \n\t"
" fmla            z23.d, p0/m, z31.d, z2.d        \n\t" // Row 9:15 column 8 and 9
" fmla            z25.d, p0/m, z31.d, z3.d        \n\t"
" fmla            z27.d, p0/m, z31.d, z4.d        \n\t" // Row 9:15 column 10 and 11
" fmla            z29.d, p0/m, z31.d, z5.d        \n\t"
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
" prfm            PLDL2STRM, [x18]                \n\t" // Prefetch 2 panels to
" prfm            PLDL2STRM, [x18, #64]           \n\t" //   "confirm" the stream prefetcher.
" prfm            PLDL2STRM, [x18, #128]          \n\t"
" prfm            PLDL2STRM, [x19]                \n\t"
" prfm            PLDL2STRM, [x19, #64]           \n\t"
" prfm            PLDL2STRM, [x19, #128]          \n\t"
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
" fmul            z26.d, z26.d, z30.d             \n\t"
" fmul            z27.d, z27.d, z30.d             \n\t"
" fmul            z28.d, z28.d, z30.d             \n\t"
" fmul            z29.d, z29.d, z30.d             \n\t"
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
" ld1d            z14.d, p0/z, [x0]               \n\t" // Read C[0:7, 10]
" st1d            z9.d, p1, [x6, x11, lsl #3]     \n\t" // Write C[8:15, 4]
" ld1d            z15.d, p1/z, [x0, x11, lsl #3]  \n\t" // Read C[8:15, 10]
" add             x0, x22, x0                     \n\t" // Next column
" add             x6, x22, x6                     \n\t" // Next column
" st1d            z10.d, p0, [x6]                 \n\t" // Write C[0:7, 5]
" ld1d            z16.d, p0/z, [x0]               \n\t" // Read C[0:7, 11]
" st1d            z11.d, p1, [x6, x11, lsl #3]    \n\t" // Write C[8:15, 5]
" ld1d            z17.d, p1/z, [x0, x11, lsl #3]  \n\t" // Read C[8:15, 11]
" add             x0, x22, x0                     \n\t" // Next column
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
" fmad            z14.d, p0/m, z31.d, z26.d       \n\t"
" fmad            z15.d, p1/m, z31.d, z27.d       \n\t"
" fmad            z16.d, p0/m, z31.d, z28.d       \n\t"
" fmad            z17.d, p1/m, z31.d, z29.d       \n\t"

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
" st1d            z14.d, p0, [x6]                 \n\t" // Write C[:, 10]
" st1d            z15.d, p1, [x6, x11, lsl #3]    \n\t"
" add             x6, x22, x6                     \n\t" // Next column
" st1d            z16.d, p0, [x6]                 \n\t" // Write C[:, 11]
" st1d            z17.d, p1, [x6, x11, lsl #3]    \n\t"
" add             x6, x22, x6                     \n\t" // Next column
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
" ld1d          z14.d, p0/z, [x0, z30.d, lsl #3]  \n\t" // Read C[0:7, 10]
" ld1d          z15.d, p1/z, [x10, z30.d, lsl #3] \n\t" // Read C[8:15, 10]
" add             x0, x22, x0                     \n\t" // Move to C[:, 11]
" add             x10, x23, x0                    \n\t" // Address C[8:15, 11]
" st1d            z10.d, p0, [x6, z30.d, lsl #3]  \n\t" // Write C[0:7, 5]
" st1d            z11.d, p1, [x16, z30.d, lsl #3] \n\t" // Write C[8:15, 5]
" add             x6, x22, x6                     \n\t" // Move to C[:, 6]
" add             x16, x23, x6                    \n\t" // Address C[8:15, 6]
" ld1d          z16.d, p0/z, [x0, z30.d, lsl #3]  \n\t" // Read C[0:7, 11]
" ld1d          z17.d, p1/z, [x10, z30.d, lsl #3] \n\t" // Read C[8:15, 11]
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
" fmad            z14.d, p0/m, z31.d, z26.d       \n\t" // Z26 used
" fmad            z15.d, p1/m, z31.d, z27.d       \n\t" // Z27 used
" fmad            z16.d, p0/m, z31.d, z28.d       \n\t" // Z28 used
" fmad            z17.d, p1/m, z31.d, z29.d       \n\t" // Z29 used
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
" st1d            z14.d, p0, [x6, z30.d, lsl #3]  \n\t" // Write C[0:7, 10]
" st1d            z15.d, p1, [x16, z30.d, lsl #3] \n\t" // Write C[8:15, 10]
" add             x6, x22, x6                     \n\t" // Move to C[:, 11]
" add             x16, x23, x6                    \n\t" // Address C[8:15, 11]
" st1d            z16.d, p0, [x6, z30.d, lsl #3]  \n\t" // Write C[0:7, 11]
" st1d            z17.d, p1, [x16, z30.d, lsl #3] \n\t" // Write C[8:15, 11]
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
 [aaddr]  "m" (a),      // 0
 [baddr]  "m" (b),      // 1
 [caddr]  "m" (c),      // 2
 [k_mker] "m" (k_mker), // 3
 [k_left] "m" (k_left), // 4
 [alpha]  "m" (alpha),  // 5
 [beta]   "m" (beta),   // 6
 [rs_c]   "m" (rs_c),   // 6
 [cs_c]   "m" (cs_c),   // 7
 [a_next] "m" (a_next), // 8
 [b_next] "m" (b_next)  // 9
:// Register clobber list
 "x0","x1","x2","x3","x4","x5","x6","x7","x8",
 "x9","x10","x11","x12","x14","x15",
 "x16","x17","x18","x19","x20","x21","x22","x23",
 "z0","z1","z2","z3","z4","z5","z6","z7",
 "z8","z9","z10","z11","z12","z13","z14","z15",
 "z16","z17","z18","z19",
 "z20","z21","z22","z23",
 "z24","z25","z26","z27",
 "z28","z29","z30","z31" );

}
