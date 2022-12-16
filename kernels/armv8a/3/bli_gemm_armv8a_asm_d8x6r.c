/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2014, The University of Texas at Austin
   Copyright (C) 2021, The University of Tokyo

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

// Label locality & misc.
#include "armv8a_asm_utils.h"

// Nanokernel operations.
#include "armv8a_asm_d2x2.h"

/* Order of row-major SGEMM_12x8's execution in 4x5 blocks:
 *
 * +---+ +---+ 
 * | 0 | | 1 | 
 * +---+ +---+ 
 * +---+ +---+ 
 * | 2 | | 3 | 
 * +---+ +---+ 
 * +---+ +---+ 
 * | 4 | | 5 | 
 * +---+ +---+ 
 */
#define SGEMM_12X8_MKER_LOOP(SUFFIX,C00,C01,C10,C11,C20,C21,C30,C31,C40,C41,C50,C51,C60,C61,C70,C71,C80,C81,C90,C91,CA0,CA1,CB0,CB1,A0,A1,A2,B0,B1,AADDR,ASHIFT,BADDR,BSHIFT,LOADNEXT,CADDR,RSC,LASTB,PRFC) \
  SGEMM_4X4_NANOKERNEL_ ##SUFFIX (C00,C10,C20,C30,B0,A0) \
  GEMM_PRFC_FH_ ##PRFC (CADDR) \
  SGEMM_4X4_NANOKERNEL_ ##SUFFIX (C01,C11,C21,C31,B1,A0) \
  DGEMM_LOAD1V_ ##LOADNEXT (A0,AADDR,ASHIFT) /* Contiguous load is the same across S/D. */ \
  GEMM_PRFC_LH_FWD_ ##PRFC (CADDR,RSC,LASTB) \
  SGEMM_4X4_NANOKERNEL_ ##SUFFIX (C40,C50,C60,C70,B0,A1) \
  GEMM_PRFC_FH_ ##PRFC (CADDR) \
  SGEMM_4X4_NANOKERNEL_ ##SUFFIX (C41,C51,C61,C71,B1,A1) \
  DGEMM_LOAD1V_ ##LOADNEXT (A1,AADDR,ASHIFT+16) \
  GEMM_PRFC_LH_FWD_ ##PRFC (CADDR,RSC,LASTB) \
  SGEMM_4X4_NANOKERNEL_ ##SUFFIX (C80,C90,CA0,CB0,B0,A2) \
  DGEMM_LOAD1V_ ##LOADNEXT (B0,BADDR,BSHIFT) \
  GEMM_PRFC_FH_ ##PRFC (CADDR) \
  SGEMM_4X4_NANOKERNEL_ ##SUFFIX (C81,C91,CA1,CB1,B1,A2) \
  GEMM_PRFC_LH_FWD_ ##PRFC (CADDR,RSC,LASTB)

// For contiguous storage of C, SLOAD is the same as DLOAD.
#define SLOADC_2V_R_FWD(C0,C1,CADDR,CSHIFT,RSC) \
  DLOAD2V(C0,C1,CADDR,CSHIFT) \
" add  "#CADDR", "#CADDR", "#RSC" \n\t"
#define SSTOREC_2V_R_FWD(C0,C1,CADDR,CSHIFT,RSC) \
  DSTORE2V(C0,C1,CADDR,CSHIFT) \
" add  "#CADDR", "#CADDR", "#RSC" \n\t"

/* Order of row-major DGEMM_8x6's execution in 2x2 blocks:
 *
 * +---+ +---+ +---+
 * | 0 | | 2 | | 4 |
 * +---+ +---+ +---+
 * +---+ +---+ +---+
 * | 1 | | 3 | | 5 |
 * +---+ +---+ +---+
 * +---+ +---+ +---+
 * | 6 | | 8 | | 10|
 * +---+ +---+ +---+
 * +---+ +---+ +---+
 * | 7 | | 9 | | 11|
 * +---+ +---+ +---+
 *
 */
#define DGEMM_8X6_MKER_LOOP(SUFFIX,C00,C01,C02,C10,C11,C12,C20,C21,C22,C30,C31,C32,C40,C41,C42,C50,C51,C52,C60,C61,C62,C70,C71,C72,A0,A1,A2,A3,B0,B1,B2,AADDR,ASHIFT,BADDR,BSHIFT,LOADNEXT,CADDR,RSC,LASTB,PRFC) \
  GEMM_PRFC_FH_ ##PRFC (CADDR) \
  DGEMM_2X2_NANOKERNEL_ ##SUFFIX (C00,C10,B0,A0) \
  GEMM_PRFC_LH_FWD_ ##PRFC (CADDR,RSC,LASTB) \
  DGEMM_2X2_NANOKERNEL_ ##SUFFIX (C20,C30,B0,A1) \
  GEMM_PRFC_FH_ ##PRFC (CADDR) \
  DGEMM_2X2_NANOKERNEL_ ##SUFFIX (C01,C11,B1,A0) \
  GEMM_PRFC_LH_FWD_ ##PRFC (CADDR,RSC,LASTB) \
  DGEMM_2X2_NANOKERNEL_ ##SUFFIX (C21,C31,B1,A1) \
  GEMM_PRFC_FH_ ##PRFC (CADDR) \
  DGEMM_2X2_NANOKERNEL_ ##SUFFIX (C02,C12,B2,A0) \
  GEMM_PRFC_LH_FWD_ ##PRFC (CADDR,RSC,LASTB) \
  DGEMM_2X2_NANOKERNEL_ ##SUFFIX (C22,C32,B2,A1) \
  DGEMM_LOAD2V_ ##LOADNEXT (A0,A1,AADDR,ASHIFT) \
  DGEMM_2X2_NANOKERNEL_ ##SUFFIX (C40,C50,B0,A2) \
  GEMM_PRFC_FH_ ##PRFC (CADDR) \
  DGEMM_2X2_NANOKERNEL_ ##SUFFIX (C60,C70,B0,A3) \
  DGEMM_LOAD1V_ ##LOADNEXT (B0,BADDR,BSHIFT) \
  DGEMM_2X2_NANOKERNEL_ ##SUFFIX (C41,C51,B1,A2) \
  GEMM_PRFC_LH_FWD_ ##PRFC (CADDR,RSC,LASTB) \
  DGEMM_2X2_NANOKERNEL_ ##SUFFIX (C61,C71,B1,A3) \
  DGEMM_LOAD1V_ ##LOADNEXT (B1,BADDR,BSHIFT+16) \
  DGEMM_2X2_NANOKERNEL_ ##SUFFIX (C42,C52,B2,A2) \
  DGEMM_2X2_NANOKERNEL_ ##SUFFIX (C62,C72,B2,A3)

// Interleaving load or not.
#define DGEMM_LOAD1V_noload(V1,ADDR,IMM)
#define DGEMM_LOAD1V_load(V1,ADDR,IMM) \
  DLOAD1V(V1,ADDR,IMM)

#define DGEMM_LOAD2V_noload(V1,V2,ADDR,IMM)
#define DGEMM_LOAD2V_load(V1,V2,ADDR,IMM) \
  DGEMM_LOAD1V_load(V1,ADDR,IMM) \
  DGEMM_LOAD1V_load(V2,ADDR,IMM+16)

// Interleaving prefetch or not.
#define GEMM_PRFC_FH_noload(CADDR)
#define GEMM_PRFC_LH_FWD_noload(CADDR,RSC,LASTB)
#define GEMM_PRFC_FH_load(CADDR) \
" prfm PLDL1KEEP, ["#CADDR"]           \n\t"
#define GEMM_PRFC_LH_FWD_load(CADDR,RSC,LASTB) \
" prfm PLDL1KEEP, ["#CADDR", "#LASTB"] \n\t" \
" add  "#CADDR", "#CADDR", "#RSC"      \n\t"

// For contiguous storage of C.
#define DLOADC_3V_R_FWD(C0,C1,C2,CADDR,CSHIFT,RSC) \
  DLOAD2V(C0,C1,CADDR,CSHIFT) \
  DLOAD1V(C2,CADDR,CSHIFT+32) \
" add  "#CADDR", "#CADDR", "#RSC" \n\t"
#define DSTOREC_3V_R_FWD(C0,C1,C2,CADDR,CSHIFT,RSC) \
  DSTORE2V(C0,C1,CADDR,CSHIFT) \
  DSTORE1V(C2,CADDR,CSHIFT+32) \
" add  "#CADDR", "#CADDR", "#RSC" \n\t"

// Prefetch C.
#define PRFMC_FWD(CADDR,RSC,LASTB) \
" prfm PLDL1KEEP, ["#CADDR"]           \n\t" \
" prfm PLDL1KEEP, ["#CADDR", "#LASTB"] \n\t" \
" add  "#CADDR", "#CADDR", "#RSC"      \n\t"

void bli_sgemm_armv8a_asm_12x8r
     (
       dim_t               m,
       dim_t               n,
       dim_t               k,
       float*     restrict alpha,
       float*     restrict a,
       float*     restrict b,
       float*     restrict beta,
       float*     restrict c, inc_t rs_c0, inc_t cs_c0,
       auxinfo_t*          data,
       cntx_t*             cntx
     )
{
  const void* a_next = bli_auxinfo_next_a( data );
  const void* b_next = bli_auxinfo_next_b( data );

  // Typecast local copies of integers in case dim_t and inc_t are a
  // different size than is expected by load instructions.
  uint64_t k_mker = k / 4;
  uint64_t k_left = k % 4;
  uint64_t rs_c   = rs_c0;
  uint64_t cs_c   = cs_c0;

  GEMM_UKR_SETUP_CT( s, 12, 8, true );

  __asm__ volatile
  (
" ldr             x0, %[a]                        \n\t"
" ldr             x1, %[b]                        \n\t"
" mov             x2, #12                         \n\t" // Column-skip of A.
" mov             x3, #8                          \n\t" // Row-skip of B.
"                                                 \n\t"
" ldr             x5, %[c]                        \n\t"
" ldr             x6, %[rs_c]                     \n\t" // Row-skip of C. (column-skip == 1)
"                                                 \n\t"
"                                                 \n\t" // Multiply some address skips by sizeof(float).
" lsl             x2, x2, #2                      \n\t" // cs_a
" lsl             x3, x3, #2                      \n\t" // rs_b
" lsl             x6, x6, #2                      \n\t" // rs_c
"                                                 \n\t"
" mov             x9, x5                          \n\t"
"                                                 \n\t"
" ldr             x4, %[k_mker]                   \n\t" // Number of loops.
" ldr             x8, %[k_left]                   \n\t"
"                                                 \n\t"
// Storage scheme:
//  V[ 0:23] <- C
//  V[24:27] <- A
//  V[28:31] <- B
// Under this scheme, the following is defined:
#define SGEMM_12X8_MKER_LOOP_LOC(SUFFIX,A0,A1,A2,B0,B1,AADDR,ASHIFT,BADDR,BSHIFT,LOADNEXT,PRFC) \
  SGEMM_12X8_MKER_LOOP(SUFFIX,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,A0,A1,A2,B0,B1,AADDR,ASHIFT,BADDR,BSHIFT,LOADNEXT,x9,x6,32,PRFC)
// Load from memory.
LABEL(SLOAD_ABC)
"                                                 \n\t" // No-microkernel early return is a must
" cmp             x4, #0                          \n\t" //  to avoid out-of-boundary read.
BEQ(SK_LEFT_LOOP_INIT)
"                                                 \n\t"
" ldr             q24, [x0, #16*0]                \n\t" // Load A.
" ldr             q25, [x0, #16*1]                \n\t"
" ldr             q26, [x0, #16*2]                \n\t"
" add             x0, x0, x2                      \n\t"
" ldr             q27, [x0, #16*0]                \n\t"
"                                                 \n\t"
" ldr             q28, [x1, #16*0]                \n\t" // Load B.
" ldr             q29, [x1, #16*1]                \n\t"
" add             x1, x1, x3                      \n\t"
" ldr             q30, [x1, #16*0]                \n\t"
" ldr             q31, [x1, #16*1]                \n\t"
" add             x1, x1, x3                      \n\t"
//
// Microkernel is defined here as:
#define SGEMM_12X8_MKER_LOOP_LOC_FWD(SUFFIX,A0,A1,A2,B0,B1,PRFC) \
  SGEMM_12X8_MKER_LOOP_LOC(SUFFIX,A0,A1,A2,B0,B1,x0,16,x1,0,load,PRFC) \
 "add             x0, x0, x2                      \n\t" \
 "ldr             q"#A2", [x0, #16*0]             \n\t" \
 "ldr             q"#B1", [x1, #16*1]             \n\t" \
 "add             x1, x1, x3                      \n\t"
// Start microkernel loop -- Initial handled differently.
SGEMM_12X8_MKER_LOOP_LOC_FWD(INIT,24,25,26,28,29,load) // Interleaving C prefetch 03/12.
SGEMM_12X8_MKER_LOOP_LOC_FWD(PLAIN,27,24,25,30,31,load) // Interleaving C prefetch 06/12.
"                                                 \n\t" // Decrease counter before final replica.
" subs            x4, x4, #1                      \n\t" // Branch early to avoid reading excess mem.
BEQ(SFIN_MKER_LOOP)
SGEMM_12X8_MKER_LOOP_LOC_FWD(PLAIN,26,27,24,28,29,load) // Interleaving C prefetch 09/12.
SGEMM_12X8_MKER_LOOP_LOC_FWD(PLAIN,25,26,27,30,31,load) // Interleaving C prefetch 12/12.
//
// The microkernel loop.
LABEL(SK_MKER_LOOP)
SGEMM_12X8_MKER_LOOP_LOC_FWD(PLAIN,24,25,26,28,29,noload)
SGEMM_12X8_MKER_LOOP_LOC_FWD(PLAIN,27,24,25,30,31,noload)
"                                                 \n\t" // Decrease counter before final replica.
" subs            x4, x4, #1                      \n\t" // Branch early to avoid reading excess mem.
BEQ(SFIN_MKER_LOOP)
SGEMM_12X8_MKER_LOOP_LOC_FWD(PLAIN,26,27,24,28,29,noload)
SGEMM_12X8_MKER_LOOP_LOC_FWD(PLAIN,25,26,27,30,31,noload)
BRANCH(SK_MKER_LOOP)
//
// Final microkernel loop.
LABEL(SFIN_MKER_LOOP)
SGEMM_12X8_MKER_LOOP_LOC(PLAIN,26,27,24,28,29,xzr,-1,xzr,-1,noload,noload)
" ldr             q26, [x0, #16*1]                \n\t"
" ldr             q27, [x0, #16*2]                \n\t"
" add             x0, x0, x2                      \n\t"
SGEMM_12X8_MKER_LOOP_LOC(PLAIN,25,26,27,30,31,xzr,-1,xzr,-1,noload,noload)
//
// Loops left behind microkernels.
LABEL(SK_LEFT_LOOP)
" cmp             x8, #0                          \n\t" // End of exec.
BEQ(SWRITE_MEM_PREP)
" ldr             q24, [x0, #16*0]                \n\t" // Load A col.
" ldr             q25, [x0, #16*1]                \n\t"
" ldr             q26, [x0, #16*2]                \n\t"
" add             x0, x0, x2                      \n\t"
" ldr             q28, [x1, #16*0]                \n\t" // Load B row.
" ldr             q29, [x1, #16*1]                \n\t"
" add             x1, x1, x3                      \n\t"
" sub             x8, x8, #1                      \n\t"
SGEMM_12X8_MKER_LOOP_LOC(PLAIN,24,25,26,28,29,xzr,-1,xzr,-1,noload,noload)
BRANCH(SK_LEFT_LOOP)
//
// No microkernel 4-loop. Have to clear C rows in the first k_left.
LABEL(SK_LEFT_LOOP_INIT)
" cmp             x8, #0                          \n\t" // End of exec.
BEQ(SCLEAR_CCOLS)
" ldr             q24, [x0, #16*0]                \n\t" // Load A col.
" ldr             q25, [x0, #16*1]                \n\t"
" ldr             q26, [x0, #16*2]                \n\t"
" add             x0, x0, x2                      \n\t"
" ldr             q28, [x1, #16*0]                \n\t" // Load B row.
" ldr             q29, [x1, #16*1]                \n\t"
" add             x1, x1, x3                      \n\t"
" sub             x8, x8, #1                      \n\t"
SGEMM_12X8_MKER_LOOP_LOC(INIT,24,25,26,28,29,xzr,-1,xzr,-1,noload,noload)
BRANCH(SK_LEFT_LOOP)
//
// No FMUL at all to clear C up. Have to zeroize.
LABEL(SCLEAR_CCOLS)
CLEAR8V(0,1,2,3,4,5,6,7)
CLEAR8V(8,9,10,11,12,13,14,15)
CLEAR8V(16,17,18,19,20,21,22,23)
//
// Scale and write to memory.
LABEL(SWRITE_MEM_PREP)
" ldr             x4, %[alpha]                    \n\t" // Load alpha & beta (address).
" ldr             x8, %[beta]                     \n\t"
" ld1r            {v24.4s}, [x4]                  \n\t" // Load alpha & beta.
" ld1r            {v25.4s}, [x8]                  \n\t"
"                                                 \n\t"
LABEL(SPREFETCH_ABNEXT)
" ldr             x0, %[a_next]                   \n\t"
" ldr             x1, %[b_next]                   \n\t"
" prfm            PLDL1STRM, [x0, 64*0]           \n\t" // Do not know cache line size,
" prfm            PLDL1STRM, [x0, 64*1]           \n\t" //  issue some number of prfm instructions
" prfm            PLDL1STRM, [x0, 64*2]           \n\t" //  to try to activate hardware prefetcher.
" prfm            PLDL1STRM, [x1, 64*0]           \n\t"
" prfm            PLDL1STRM, [x1, 64*1]           \n\t"
" prfm            PLDL1STRM, [x1, 64*2]           \n\t"
" prfm            PLDL1STRM, [x1, 64*3]           \n\t"
"                                                 \n\t"
" fmov            d26, #1.0                       \n\t"
" fcvt            s26, d26                        \n\t"
" fcmp            s24, s26                        \n\t"
BEQ(SUNIT_ALPHA)
SSCALE8V(0,1,2,3,4,5,6,7,24,0)
SSCALE8V(8,9,10,11,12,13,14,15,24,0)
SSCALE8V(16,17,18,19,20,21,22,23,24,0)
LABEL(SUNIT_ALPHA)
"                                                 \n\t"
" mov             x9, x5                          \n\t" // C address for loading.
"                                                 \n\t" // C address for storing is x5 itself.
//
// Contiguous C-storage.
LABEL(SWRITE_MEM_R)
" fcmp            s25, #0.0                       \n\t" // Sets conditional flag whether *beta == 0.
"                                                 \n\t" // This conditional flag will be used
"                                                 \n\t" //  multiple times for skipping load.
// Row 0 & 1 & 2:
BEQ(SZERO_BETA_R_0_1_2)
SLOADC_2V_R_FWD(26,27,x9,0,x6)
SLOADC_2V_R_FWD(28,29,x9,0,x6)
SLOADC_2V_R_FWD(30,31,x9,0,x6)
SSCALEA2V(0,1,26,27,25,0)
SSCALEA2V(2,3,28,29,25,0)
SSCALEA2V(4,5,30,31,25,0)
LABEL(SZERO_BETA_R_0_1_2)
SSTOREC_2V_R_FWD(0,1,x5,0,x6)
SSTOREC_2V_R_FWD(2,3,x5,0,x6)
SSTOREC_2V_R_FWD(4,5,x5,0,x6)
// Row 3 & 4 & 5 & 6 & 7 & 8:
BEQ(SZERO_BETA_R_3_4_5_6_7_8)
SLOADC_2V_R_FWD(26,27,x9,0,x6)
SLOADC_2V_R_FWD(28,29,x9,0,x6)
SLOADC_2V_R_FWD(30,31,x9,0,x6)
SLOADC_2V_R_FWD(0,1,x9,0,x6)
SLOADC_2V_R_FWD(2,3,x9,0,x6)
SLOADC_2V_R_FWD(4,5,x9,0,x6)
SSCALEA4V(6,7,8,9,26,27,28,29,25,0)
SSCALEA4V(10,11,12,13,30,31,0,1,25,0)
SSCALEA4V(14,15,16,17,2,3,4,5,25,0)
LABEL(SZERO_BETA_R_3_4_5_6_7_8)
SSTOREC_2V_R_FWD(6,7,x5,0,x6)
SSTOREC_2V_R_FWD(8,9,x5,0,x6)
SSTOREC_2V_R_FWD(10,11,x5,0,x6)
SSTOREC_2V_R_FWD(12,13,x5,0,x6)
SSTOREC_2V_R_FWD(14,15,x5,0,x6)
SSTOREC_2V_R_FWD(16,17,x5,0,x6)
// Row 9 & 10 & 11
BEQ(SZERO_BETA_R_9_10_11)
SLOADC_2V_R_FWD(26,27,x9,0,x6)
SLOADC_2V_R_FWD(28,29,x9,0,x6)
SLOADC_2V_R_FWD(30,31,x9,0,x6)
SSCALEA2V(18,19,26,27,25,0)
SSCALEA2V(20,21,28,29,25,0)
SSCALEA2V(22,23,30,31,25,0)
LABEL(SZERO_BETA_R_9_10_11)
SSTOREC_2V_R_FWD(18,19,x5,0,x6)
SSTOREC_2V_R_FWD(20,21,x5,0,x6)
SSTOREC_2V_R_FWD(22,23,x5,0,x6)
// Done.
LABEL(SEND_WRITE_MEM)
:
: [a]      "m" (a),
  [b]      "m" (b),
  [c]      "m" (c),
  [rs_c]   "m" (rs_c),
  [k_mker] "m" (k_mker),
  [k_left] "m" (k_left),
  [alpha]  "m" (alpha),
  [beta]   "m" (beta),
  [a_next] "m" (a_next),
  [b_next] "m" (b_next)
: "x0","x1","x2","x3","x4","x5","x6","x8","x9",
  "v0","v1","v2","v3","v4","v5","v6","v7",
  "v8","v9","v10","v11","v12","v13","v14","v15",
  "v16","v17","v18","v19",
  "v20","v21","v22","v23",
  "v24","v25","v26","v27",
  "v28","v29","v30","v31"
  );

  GEMM_UKR_FLUSH_CT( s );
}

/*
 * Differences from the col-major 6x8 in HW modeling:
 * * Stream HW prefetcher is assumed s.t. PRFM instructions for packed A&B are omitted.
 */
void bli_dgemm_armv8a_asm_8x6r
     (
       dim_t               m,
       dim_t               n,
       dim_t               k,
       double*    restrict alpha,
       double*    restrict a,
       double*    restrict b,
       double*    restrict beta,
       double*    restrict c, inc_t rs_c0, inc_t cs_c0,
       auxinfo_t*          data,
       cntx_t*             cntx
     )
{
  const void* a_next = bli_auxinfo_next_a( data );
  const void* b_next = bli_auxinfo_next_b( data );

  // Typecast local copies of integers in case dim_t and inc_t are a
  // different size than is expected by load instructions.
  uint64_t k_mker = k / 4;
  uint64_t k_left = k % 4;
  uint64_t rs_c   = rs_c0;
  uint64_t cs_c   = cs_c0;
  // TODO: Aggregated str instructions.

  GEMM_UKR_SETUP_CT( d, 8, 6, true );

  __asm__ volatile
  (
" lsl             %3, %3, #3                      \n\t" // rs_c *= sizeof(double).
" mov             x9, %2                          \n\t" // Address of C for prefetching.
"                                                 \n\t"
// Storage scheme:
//  V[ 0:23] <- C
//  V[24:27] <- A
//  V[28:31] <- B
// Under this scheme, the following is defined:
#define DGEMM_8X6_MKER_LOOP_LOC(SUFFIX,A0,A1,A2,A3,B0,B1,B2,AADDR,ASHIFT,BADDR,BSHIFT,LOADNEXT,PRFC) \
  DGEMM_8X6_MKER_LOOP(SUFFIX,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,A0,A1,A2,A3,B0,B1,B2,AADDR,ASHIFT,BADDR,BSHIFT,LOADNEXT,x9,%3,40,PRFC)
// Load from memory.
LABEL(DLOAD_ABC)
"                                                 \n\t" // No-microkernel early return is a must
" cmp             %4, #0                          \n\t" //  to avoid out-of-boundary read.
BEQ(DK_LEFT_LOOP_INIT)
"                                                 \n\t"
" ldr             q24, [%0, #16*0]                \n\t" // Load A.
" ldr             q25, [%0, #16*1]                \n\t"
" ldr             q26, [%0, #16*2]                \n\t"
" ldr             q27, [%0, #16*3]                \n\t"
" add             %0, %0, #64                     \n\t"
"                                                 \n\t"
" ldr             q28, [%1, #16*0]                \n\t" // Load B.
" ldr             q29, [%1, #16*1]                \n\t"
" ldr             q30, [%1, #16*2]                \n\t"
" add             %1, %1, #48                     \n\t"
" ldr             q31, [%1, #16*0]                \n\t"
//
// Microkernel is defined here as:
#define DGEMM_8X6_MKER_LOOP_LOC_FWD(SUFFIX,A0,A1,A2,A3,B0,B1,B2,PRFC) \
  DGEMM_8X6_MKER_LOOP_LOC(SUFFIX,A0,A1,A2,A3,B0,B1,B2,%0,0,%1,16,load,PRFC) \
 "add             %1, %1, #48                     \n\t" \
 "ldr             q"#B2", [%1, #16*0]             \n\t" \
 "ldr             q"#A2", [%0, #16*2]             \n\t" \
 "ldr             q"#A3", [%0, #16*3]             \n\t" \
 "add             %0, %0, #64                     \n\t"
// Start microkernel loop -- Special treatment for the very first loop.
" subs            %4, %4, #1                      \n\t" // Decrease counter in advance.
DGEMM_8X6_MKER_LOOP_LOC_FWD(INIT,24,25,26,27,28,29,30,load) // Prefetch C 1-4/8.
DGEMM_8X6_MKER_LOOP_LOC_FWD(PLAIN,24,25,26,27,31,28,29,load) // Prefetch C 5-8/8.
BEQ(DFIN_MKER_LOOP) // Branch early to avoid reading excess mem.
DGEMM_8X6_MKER_LOOP_LOC_FWD(PLAIN,24,25,26,27,30,31,28,noload)
DGEMM_8X6_MKER_LOOP_LOC_FWD(PLAIN,24,25,26,27,29,30,31,noload)
// Start microkernel loop.
LABEL(DK_MKER_LOOP)
" subs            %4, %4, #1                      \n\t" // Decrease counter in advance.
DGEMM_8X6_MKER_LOOP_LOC_FWD(PLAIN,24,25,26,27,28,29,30,noload)
DGEMM_8X6_MKER_LOOP_LOC_FWD(PLAIN,24,25,26,27,31,28,29,noload)
BEQ(DFIN_MKER_LOOP) // Branch early to avoid reading excess mem.
DGEMM_8X6_MKER_LOOP_LOC_FWD(PLAIN,24,25,26,27,30,31,28,noload)
DGEMM_8X6_MKER_LOOP_LOC_FWD(PLAIN,24,25,26,27,29,30,31,noload)
BRANCH(DK_MKER_LOOP)
//
// Final microkernel loop.
LABEL(DFIN_MKER_LOOP)
DGEMM_8X6_MKER_LOOP_LOC(PLAIN,24,25,26,27,30,31,28,%0,0,%1,16,load,noload)
" add             %1, %1, #48                     \n\t"
" ldr             q26, [%0, #16*2]                \n\t"
" ldr             q27, [%0, #16*3]                \n\t"
" add             %0, %0, #64                     \n\t"
DGEMM_8X6_MKER_LOOP_LOC(PLAIN,24,25,26,27,29,30,31,xzr,-1,xzr,-1,noload,noload)
//
// Loops left behind microkernels.
LABEL(DK_LEFT_LOOP)
" cmp             %5, #0                          \n\t" // End of exec.
BEQ(DWRITE_MEM_PREP)
" ldr             q24, [%0, #16*0]                \n\t" // Load A col.
" ldr             q25, [%0, #16*1]                \n\t"
" ldr             q26, [%0, #16*2]                \n\t"
" ldr             q27, [%0, #16*3]                \n\t"
" add             %0, %0, #64                     \n\t"
" ldr             q28, [%1, #16*0]                \n\t" // Load B row.
" ldr             q29, [%1, #16*1]                \n\t"
" ldr             q30, [%1, #16*2]                \n\t"
" add             %1, %1, #48                     \n\t"
" sub             %5, %5, #1                      \n\t"
DGEMM_8X6_MKER_LOOP_LOC(PLAIN,24,25,26,27,28,29,30,xzr,-1,xzr,-1,noload,noload)
BRANCH(DK_LEFT_LOOP)
//
// No microkernel 4-loop. Have to clear C rows in the first k_left.
LABEL(DK_LEFT_LOOP_INIT)
" cmp             %5, #0                          \n\t" // End of exec.
BEQ(DCLEAR_CCOLS)
" ldr             q24, [%0, #16*0]                \n\t" // Load A col.
" ldr             q25, [%0, #16*1]                \n\t"
" ldr             q26, [%0, #16*2]                \n\t"
" ldr             q27, [%0, #16*3]                \n\t"
" add             %0, %0, #64                     \n\t"
" ldr             q28, [%1, #16*0]                \n\t" // Load B row.
" ldr             q29, [%1, #16*1]                \n\t"
" ldr             q30, [%1, #16*2]                \n\t"
" add             %1, %1, #48                     \n\t"
" sub             %5, %5, #1                      \n\t"
DGEMM_8X6_MKER_LOOP_LOC(INIT,24,25,26,27,28,29,30,xzr,-1,xzr,-1,noload,noload)
BRANCH(DK_LEFT_LOOP)
//
// No FMUL at all to clear C up. Have to zeroize.
LABEL(DCLEAR_CCOLS)
CLEAR8V(0,1,2,3,4,5,6,7)
CLEAR8V(8,9,10,11,12,13,14,15)
CLEAR8V(16,17,18,19,20,21,22,23)
//
// Scale and write to memory.
LABEL(DWRITE_MEM_PREP)
" ld1r            {v24.2d}, [%[alpha]]            \n\t" // Load alpha & beta.
" ld1r            {v25.2d}, [%[beta]]             \n\t"
"                                                 \n\t"
LABEL(DPREFETCH_ABNEXT)
" prfm            PLDL1STRM, [%[a_next], 64*0]    \n\t" // Do not know cache line size,
" prfm            PLDL1STRM, [%[a_next], 64*1]    \n\t" //  issue some number of prfm instructions
" prfm            PLDL1STRM, [%[a_next], 64*2]    \n\t" //  to try to activate hardware prefetcher.
" prfm            PLDL1STRM, [%[b_next], 64*0]    \n\t"
" prfm            PLDL1STRM, [%[b_next], 64*1]    \n\t"
" prfm            PLDL1STRM, [%[b_next], 64*2]    \n\t"
" prfm            PLDL1STRM, [%[b_next], 64*3]    \n\t"
"                                                 \n\t"
" fmov            d26, #1.0                       \n\t"
" fcmp            d24, d26                        \n\t"
BEQ(DUNIT_ALPHA)
DSCALE8V(0,1,2,3,4,5,6,7,24,0)
DSCALE8V(8,9,10,11,12,13,14,15,24,0)
DSCALE8V(16,17,18,19,20,21,22,23,24,0)
LABEL(DUNIT_ALPHA)
"                                                 \n\t"
" mov             x9, %2                          \n\t" // C address for loading.
"                                                 \n\t" // C address for storing is %2 itself.
//
// Contiguous C-storage.
LABEL(DWRITE_MEM_R)
" fcmp            d25, #0.0                       \n\t" // Sets conditional flag whether *beta == 0.
"                                                 \n\t" // This conditional flag will be used
"                                                 \n\t" //  multiple times for skipping load.
// Row 0 & 1:
BEQ(DZERO_BETA_R_0_1)
DLOADC_3V_R_FWD(26,27,28,x9,0,%3)
DLOADC_3V_R_FWD(29,30,31,x9,0,%3)
DSCALEA2V(0,1,26,27,25,0)
DSCALEA2V(2,3,28,29,25,0)
DSCALEA2V(4,5,30,31,25,0)
LABEL(DZERO_BETA_R_0_1)
DSTOREC_3V_R_FWD(0,1,2,%2,0,%3)
DSTOREC_3V_R_FWD(3,4,5,%2,0,%3)
// Row 2 & 3 & 4 & 5:
BEQ(DZERO_BETA_R_2_3_4_5)
DLOADC_3V_R_FWD(26,27,28,x9,0,%3)
DLOADC_3V_R_FWD(29,30,31,x9,0,%3)
DLOADC_3V_R_FWD(0,1,2,x9,0,%3)
DLOADC_3V_R_FWD(3,4,5,x9,0,%3)
DSCALEA4V(6,7,8,9,26,27,28,29,25,0)
DSCALEA4V(10,11,12,13,30,31,0,1,25,0)
DSCALEA4V(14,15,16,17,2,3,4,5,25,0)
LABEL(DZERO_BETA_R_2_3_4_5)
DSTOREC_3V_R_FWD(6,7,8,%2,0,%3)
DSTOREC_3V_R_FWD(9,10,11,%2,0,%3)
DSTOREC_3V_R_FWD(12,13,14,%2,0,%3)
DSTOREC_3V_R_FWD(15,16,17,%2,0,%3)
// Row 6 & 7
BEQ(DZERO_BETA_R_6_7)
DLOADC_3V_R_FWD(26,27,28,x9,0,%3)
DLOADC_3V_R_FWD(29,30,31,x9,0,%3)
DSCALEA2V(18,19,26,27,25,0)
DSCALEA2V(20,21,28,29,25,0)
DSCALEA2V(22,23,30,31,25,0)
LABEL(DZERO_BETA_R_6_7)
DSTOREC_3V_R_FWD(18,19,20,%2,0,%3)
DSTOREC_3V_R_FWD(21,22,23,%2,0,%3)
// Done.
LABEL(DEND_WRITE_MEM)
: "+r" (a),      // %0
  "+r" (b),      // %1
  "+r" (c),      // %2
  "+r" (rs_c),   // %3
  "+r" (k_mker), // %4
  "+r" (k_left), // %5
  [alpha]  "+r" (alpha),
  [beta]   "+r" (beta),
  [a_next] "+r" (a_next),
  [b_next] "+r" (b_next)
:
: "x9",
  "v0","v1","v2","v3","v4","v5","v6","v7",
  "v8","v9","v10","v11","v12","v13","v14","v15",
  "v16","v17","v18","v19",
  "v20","v21","v22","v23",
  "v24","v25","v26","v27",
  "v28","v29","v30","v31"
  );

  GEMM_UKR_FLUSH_CT( d );
}

