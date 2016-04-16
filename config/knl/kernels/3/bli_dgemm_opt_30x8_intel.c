/*

   BLIS    
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2014, The University of Texas at Austin

   Redistribution and use in source and binary forms, with or without
   modification, are permitted provided that the following conditions are
   met:
    - Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.
    - Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.
    - Neither the name of The University of Texas at Austin nor the names
      of its contributors may be used to endorse or promote products
      derived derived from this software without specific prior written permission.

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
#include <assert.h>


#define A_L1_PREFETCH_DIST 4
#define B_L1_PREFETCH_DIST 2
#define L2_PREFETCH_DIST  16 // Must be greater than 10, because of the way the loop is constructed.

//Alternate code path uused if C is not row-major
// r9 = c
// ebx = 0xff
// zmm30 = cs_c * 1...8
// r11 = rs_c
#define UPDATE_C_ROW_SCATTERED(NUM) \
\
    __asm kmovw k3, ebx \
    __asm GATHER##NUM: \
        __asm vgatherdpd zmm31{k3}, [r9 + zmm30*8] \
        __asm kortestw k3, k3 \
        __asm jnz k3, GATHER##NUM \
    \
    __asm vmulpd zmm##NUM, zmm##NUM, 0[r12]{1to8} /*scale by alpha*/ \
    __asm vfmadd132pd zmm31, zmm##NUM, 0[r13]{1to8} /*scale by beta, add in result*/\
    \
    __asm kmovw k3, ebx \
    __asm SCATTER##NUM: \
        __asm vscatterdpd [r9 + zmm30*8]{k3}, zmm31 \
        __asm kortestw k3, k3 \
        __asm jnz k3, SCATTER##NUM \
    __asm add r9, r11 \

// r12 = &alpha
// zmm31 = beta
// r9 = c
// r11 =   rs_c
// r10 = 3*rs_c
// rdi = 4*rs_c
#define UPDATE_C_4_ROWS(R1, R2, R3, R4) \
\
    __asm vmulpd zmm##R1, zmm##R1, 0[r12]{1to8} \
    __asm vmulpd zmm##R2, zmm##R2, 0[r12]{1to8} \
    __asm vmulpd zmm##R3, zmm##R3, 0[r12]{1to8} \
    __asm vmulpd zmm##R4, zmm##R4, 0[r12]{1to8} \
    __asm vfmadd231pd zmm##R1, zmm31, [r9+0] \
    __asm vfmadd231pd zmm##R2, zmm31, [r9+r11+0] \
    __asm vfmadd231pd zmm##R3, zmm31, [r9+2*r11+0] \
    __asm vfmadd231pd zmm##R4, zmm31, [r9+r10+0] \
    __asm vmovapd [r9+0], zmm##R1 \
    __asm vmovapd [r9+r11+0], zmm##R2 \
    __asm vmovapd [r9+2*r11+0], zmm##R3 \
    __asm vmovapd [r9+r10+0], zmm##R4 \
    __asm add r9, rdi

// r12 = &alpha
// zmm31 = beta
// r9 = c
// r11 = rs_c
#define UPDATE_C_2_ROWS(R1, R2) \
\
    __asm vmulpd zmm##R1, zmm##R1, 0[r12]{1to8} \
    __asm vmulpd zmm##R2, zmm##R2, 0[r12]{1to8} \
    __asm vfmadd231pd zmm##R1, zmm31, [r9+0] \
    __asm vfmadd231pd zmm##R2, zmm31, [r9+r11+0] \
    __asm vmovapd [r9+0], zmm##R1 \
    __asm vmovapd [r9+r11+0], zmm##R2

//One iteration of the k_r loop.
//Each iteration, we prefetch A into L1 and into L2
// r15 = a
// rbx = b
// r11 = rs_c
// r13 = L2_PREFETCH_DIST*8*8
// r14 = L2_PREFETCH_DIST*8*32
// 256 = 32*8 = dist. to next sliver of a
// 64 = 8*8 = dist. to next sliver of b
#define ONE_ITER_MAIN_LOOP(COUNTER, PC_L1, PC_L2) \
\
    /* Can this be pre-loaded for next iter. in zmm30? */ \
    __asm vmovapd zmm31, 0[rbx]                     \
                                                    \
    __asm vfmadd231pd zmm0, zmm31,  0*8[r15]{1to8}  \
    __asm vfmadd231pd zmm4, zmm31,  4*8[r15]{1to8}  \
    __asm vprefetch0 A_L1_PREFETCH_DIST*256[r15]    \
    __asm vfmadd231pd zmm5, zmm31,  5*8[r15]{1to8}  \
    __asm vprefetch0 A_L1_PREFETCH_DIST*256+64[r15] \
    __asm vfmadd231pd zmm6, zmm31,  6*8[r15]{1to8}  \
    __asm vprefetch0 A_L1_PREFETCH_DIST*256+128[r15]\
    __asm vfmadd231pd zmm7, zmm31,  7*8[r15]{1to8}  \
    __asm vprefetch0 A_L1_PREFETCH_DIST*256+192[r15]\
    __asm vfmadd231pd zmm8, zmm31,  8*8[r15]{1to8}  \
                                                    \
    __asm vprefetch1 0[r15 + r14]                   \
    __asm vfmadd231pd zmm9, zmm31,  9*8[r15]{1to8}  \
    if (PC_L1) __asm vprefetch0 0[rcx]              \
    __asm vfmadd231pd zmm1, zmm31, 1*8[r15]{1to8}   \
    if (PC_L1) __asm add rcx, r11                   \
    __asm vfmadd231pd zmm2, zmm31, 2*8[r15]{1to8}   \
    __asm vfmadd231pd zmm3, zmm31, 3*8[r15]{1to8}   \
    __asm vfmadd231pd zmm10, zmm31, 10*8[r15]{1to8} \
                                                    \
    __asm vprefetch1 64[r15 + r14]                  \
    __asm vfmadd231pd zmm11, zmm31, 11*8[r15]{1to8} \
    if (PC_L2) __asm vprefetch1 0[rcx]              \
    if (PC_L1) __asm vprefetch0 0[rcx]              \
    __asm vfmadd231pd zmm12, zmm31, 12*8[r15]{1to8} \
    if (PC_L1) __asm add rcx, r11                   \
    __asm vfmadd231pd zmm13, zmm31, 13*8[r15]{1to8} \
    __asm vfmadd231pd zmm14, zmm31, 14*8[r15]{1to8} \
    __asm vfmadd231pd zmm15, zmm31, 15*8[r15]{1to8} \
                                                    \
    __asm vprefetch1 2*64[r15 + r14]                \
    __asm vfmadd231pd zmm16, zmm31, 16*8[r15]{1to8} \
    if (PC_L1) __asm vprefetch0 0[rcx]              \
    __asm vfmadd231pd zmm17, zmm31, 17*8[r15]{1to8} \
    if (PC_L1) __asm add rcx, r11                   \
    __asm vfmadd231pd zmm18, zmm31, 18*8[r15]{1to8} \
    __asm vfmadd231pd zmm19, zmm31, 19*8[r15]{1to8} \
    __asm vfmadd231pd zmm20, zmm31, 20*8[r15]{1to8} \
                                                    \
    __asm vprefetch1 3*64[r15 + r14]                \
    __asm vfmadd231pd zmm21, zmm31, 21*8[r15]{1to8} \
    __asm add r15, r12                              \
    __asm vfmadd231pd zmm22, zmm31, -10*8[r15]{1to8}\
    __asm vfmadd231pd zmm23, zmm31, -9*8[r15]{1to8} \
    if (PC_L2) __asm add rcx, r11                   \
    __asm vfmadd231pd zmm24, zmm31, -8*8[r15]{1to8} \
    __asm dec COUNTER                               \
    __asm vfmadd231pd zmm25, zmm31, -7*8[r15]{1to8} \
                                                    \
    __asm vprefetch1 0[rbx + r13]                   \
    __asm vfmadd231pd zmm26, zmm31, -6*8[r15]{1to8} \
    __asm vprefetch0 B_L1_PREFETCH_DIST*8*8[rbx]    \
    __asm vfmadd231pd zmm27, zmm31, -5*8[r15]{1to8} \
    __asm add rbx, r9                               \
    __asm vfmadd231pd zmm28, zmm31, -4*8[r15]{1to8} \
    __asm cmp COUNTER, 0                            \
    __asm vfmadd231pd zmm29, zmm31, -3*8[r15]{1to8}

//This is an array used for the scattter/gather instructions.
extern int offsets[16];


//#define MONITORS
//#define LOOPMON
void bli_dgemm_opt_30x8(
                    dim_t            k,
                    double* restrict alpha,
                    double* restrict a,
                    double* restrict b,
                    double* restrict beta,
                    double* restrict c, inc_t rs_c, inc_t cs_c,
                    auxinfo_t*       data
                  )
{
    double * a_next = bli_auxinfo_next_a( data );
    double * b_next = bli_auxinfo_next_b( data );

    int * offsetPtr = &offsets[0];

#ifdef MONITORS
    int toph, topl, both, botl, midl, midh, mid2l, mid2h;
#endif
#ifdef LOOPMON
    int tlooph, tloopl, blooph, bloopl;
#endif
    
#ifdef MONITORS
    __asm rdtsc
    __asm mov topl, eax
    __asm mov toph, edx
#endif

    __asm vpxord  zmm0,  zmm0, zmm0
    __asm vmovaps zmm1,  zmm0  //clear out registers
    __asm vmovaps zmm2,  zmm0
    __asm mov rsi, k    //loop index
    __asm vmovaps zmm3,  zmm0

    __asm mov r11, rs_c           //load row stride
    __asm vmovaps zmm4,  zmm0
    __asm sal r11, 3              //scale row stride
    __asm vmovaps zmm5,  zmm0
    __asm mov r15, a              //load address of a
    __asm vmovaps zmm6,  zmm0
    __asm mov rbx, b              //load address of b
    __asm vmovaps zmm7,  zmm0

    __asm vmovaps zmm8,  zmm0
    __asm lea r10, [r11 + 2*r11 + 0] //r10 has 3 * r11
    __asm vmovaps zmm9,  zmm0
    __asm vmovaps zmm10, zmm0
    __asm mov rdi, r11
    __asm vmovaps zmm11, zmm0
    __asm sal rdi, 2              //rdi has 4*r11

    __asm vmovaps zmm12, zmm0
    __asm mov rcx, c              //load address of c for prefetching
    __asm vmovaps zmm13, zmm0
    __asm vmovaps zmm14, zmm0
    __asm mov r8, k
    __asm vmovaps zmm15, zmm0

    __asm vmovaps zmm16, zmm0
    __asm vmovaps zmm17, zmm0
    __asm mov r13, L2_PREFETCH_DIST*8*8
    __asm vmovaps zmm18, zmm0
    __asm mov r14, L2_PREFETCH_DIST*8*32
    __asm vmovaps zmm19, zmm0
    __asm vmovaps zmm20, zmm0
    __asm vmovaps zmm21, zmm0
    __asm vmovaps zmm22, zmm0

    __asm vmovaps zmm23, zmm0
    __asm sub r8, 30 + L2_PREFETCH_DIST       //Check if we have over 40 operations to do.
    __asm vmovaps zmm24, zmm0
    __asm mov r8, 30
    __asm vmovaps zmm25, zmm0
    __asm mov r9, 8*8                         //amount to increment b* by each iteration
    __asm vmovaps zmm26, zmm0
    __asm mov r12, 32*8                       //amount to increment a* by each iteration
    __asm vmovaps zmm27, zmm0
    __asm vmovaps zmm28, zmm0
    __asm vmovaps zmm29, zmm0

#ifdef MONITORS
    __asm rdtsc
    __asm mov midl, eax
    __asm mov midh, edx
#endif
    __asm jle CONSIDER_UNDER_40
    __asm sub rsi, 30 + L2_PREFETCH_DIST

    //First 30 iterations
    __asm LOOPREFECHCL2:
        ONE_ITER_MAIN_LOOP(r8, 0, 1)
    __asm jne LOOPREFECHCL2
    __asm mov rcx, c

    //Main Loop.
    __asm LOOPMAIN:
        ONE_ITER_MAIN_LOOP(rsi, 0, 0)
    __asm jne LOOPMAIN

    //Penultimate 22 iterations.
    //Break these off from the main loop to avoid prefetching extra shit.
    __asm mov r14, a_next
    __asm mov r13, b_next
    __asm sub r14, r15
    __asm sub r13, rbx

    __asm mov rsi, L2_PREFETCH_DIST-10
    __asm LOOPMAIN2:
        ONE_ITER_MAIN_LOOP(rsi, 0, 0)
    __asm jne LOOPMAIN2

    //Last 10 iterations
    __asm mov r8, 10
    __asm LOOPREFETCHCL1:
        ONE_ITER_MAIN_LOOP(r8, 1, 0)
    __asm jne LOOPREFETCHCL1

    __asm jmp POSTACCUM

    //Alternate main loop, with no prefetching of C
    //Used when <= 40 iterations
    __asm CONSIDER_UNDER_40:

    __asm mov rsi, k
    __asm test rsi, rsi
    __asm je POSTACCUM

    __asm LOOP_UNDER_40:
        ONE_ITER_MAIN_LOOP(rsi)
    __asm jne LOOP_UNDER_40

    __asm POSTACCUM:

#ifdef MONITORS
    __asm rdtsc
    __asm mov mid2l, eax
    __asm mov mid2h, edx
#endif

    __asm mov r9, c               //load address of c for update
    __asm mov r12, alpha          //load address of alpha

    // Check if C is row stride. If not, jump to the slow scattered update
    __asm mov r14, cs_c
    __asm dec r14
    __asm jne SCATTEREDUPDATE

    __asm mov r14, beta
    __asm vbroadcastsd zmm31, 0[r14]

    /* Alignment??? */
    UPDATE_C_4_ROWS( 0, 1, 2, 3)
    UPDATE_C_4_ROWS( 4, 5, 6, 7)
    UPDATE_C_4_ROWS( 8, 9,10,11)
    UPDATE_C_4_ROWS(12,13,14,15)
    UPDATE_C_4_ROWS(16,17,18,19)
    UPDATE_C_4_ROWS(20,21,22,23)
    UPDATE_C_4_ROWS(24,25,26,27)
    UPDATE_C_2_ROWS(28,29)

    __asm jmp END

    __asm SCATTEREDUPDATE:

    __asm mov r10, offsetPtr
    __asm vmovapd zmm31, 0[r10]
    __asm vpbroadcastd zmm30, cs_c
    __asm mov r13, beta
    __asm vpmulld zmm30, zmm31, zmm30

    __asm mov ebx, 255
    UPDATE_C_ROW_SCATTERED( 0)
    UPDATE_C_ROW_SCATTERED( 1)
    UPDATE_C_ROW_SCATTERED( 2)
    UPDATE_C_ROW_SCATTERED( 3)
    UPDATE_C_ROW_SCATTERED( 4)
    UPDATE_C_ROW_SCATTERED( 5)
    UPDATE_C_ROW_SCATTERED( 6)
    UPDATE_C_ROW_SCATTERED( 7)
    UPDATE_C_ROW_SCATTERED( 8)
    UPDATE_C_ROW_SCATTERED( 9)
    UPDATE_C_ROW_SCATTERED(10)
    UPDATE_C_ROW_SCATTERED(11)
    UPDATE_C_ROW_SCATTERED(12)
    UPDATE_C_ROW_SCATTERED(13)
    UPDATE_C_ROW_SCATTERED(14)
    UPDATE_C_ROW_SCATTERED(15)
    UPDATE_C_ROW_SCATTERED(16)
    UPDATE_C_ROW_SCATTERED(17)
    UPDATE_C_ROW_SCATTERED(18)
    UPDATE_C_ROW_SCATTERED(19)
    UPDATE_C_ROW_SCATTERED(20)
    UPDATE_C_ROW_SCATTERED(21)
    UPDATE_C_ROW_SCATTERED(22)
    UPDATE_C_ROW_SCATTERED(23)
    UPDATE_C_ROW_SCATTERED(24)
    UPDATE_C_ROW_SCATTERED(25)
    UPDATE_C_ROW_SCATTERED(26)
    UPDATE_C_ROW_SCATTERED(27)
    UPDATE_C_ROW_SCATTERED(28)
    UPDATE_C_ROW_SCATTERED(29)

    __asm END:

#ifdef MONITORS
    __asm rdtsc
    __asm mov botl, eax
    __asm mov both, edx
#endif

#ifdef LOOPMON
    printf("looptime = \t%d\n", bloopl - tloopl);
#endif
#ifdef MONITORS
    dim_t top = ((dim_t)toph << 32) | topl;
    dim_t mid = ((dim_t)midh << 32) | midl;
    dim_t mid2 = ((dim_t)mid2h << 32) | mid2l;
    dim_t bot = ((dim_t)both << 32) | botl;
    printf("setup =\t%u\tmain loop =\t%u\tcleanup=\t%u\ttotal=\t%u\n", mid - top, mid2 - mid, bot - mid2, bot - top);
#endif
}
