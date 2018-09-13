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

/**************************************************************************************
* Macro definitions
**************************************************************************************/

#define INIT4x4 \
"   vsub.f32        s16 , s16 , s16             \n\t" \
"   vmov.f32        s17, s16                    \n\t" \
"   vmov.f32        s18, s16                    \n\t" \
"   vmov.f32        s19, s16                    \n\t" \
"   vmov.f32        s20, s16                    \n\t" \
"   vmov.f32        s21, s16                    \n\t" \
"   vmov.f32        s22, s16                    \n\t" \
"   vmov.f32        s23, s16                    \n\t" \
"   vmov.f32        s24, s16                    \n\t" \
"   vmov.f32        s25, s16                    \n\t" \
"   vmov.f32        s26, s16                    \n\t" \
"   vmov.f32        s27, s16                    \n\t" \
"   vmov.f32        s28, s16                    \n\t" \
"   vmov.f32        s29, s16                    \n\t" \
"   vmov.f32        s30, s16                    \n\t" \
"   vmov.f32        s31, s16                    \n\t" \

#define KERNEL4x4_I \
"   pld [ r5 , #96 ]                         \n\t" \
"   fldmias r5!, { s0 - s1 }                    \n\t" \
"   pld [ r6 , #96 ]                         \n\t" \
"   fldmias r6!, { s8 - s9 }                    \n\t" \
\
"   fmuls   s16  , s0,  s8                      \n\t" \
"   fldmias r5!, { s2 - s3 }                    \n\t" \
"   fmuls   s17  , s1,  s8                      \n\t" \
"   fmuls   s18  , s2,  s8                      \n\t" \
"   fldmias r6!, { s10 - s11 }                  \n\t" \
"   fmuls   s19  , s3,  s8                      \n\t" \
\
"   fmuls   s20  , s0,  s9                      \n\t" \
"   fldmias r5!, { s4 - s5 }                    \n\t" \
"   fmuls   s21  , s1,  s9                      \n\t" \
"   fmuls   s22  , s2,  s9                      \n\t" \
"   fldmias r5!, { s6 - s7 }                    \n\t" \
"   fmuls   s23  , s3,  s9                      \n\t" \
\
"   fmuls   s24  , s0,  s10                     \n\t" \
"   fldmias r6!, { s12 - s13 }                  \n\t" \
"   fmuls   s25  , s1,  s10                     \n\t" \
"   fmuls   s26  , s2,  s10                     \n\t" \
"   fldmias r6!, { s14 - s15 }                  \n\t" \
"   fmuls   s27  , s3,  s10                     \n\t" \
\
"   fmuls   s28  , s0,  s11                     \n\t" \
"   fmuls   s29  , s1,  s11                     \n\t" \
"   fmuls   s30  , s2,  s11                     \n\t" \
"   fmuls   s31  , s3,  s11                     \n\t" \


#define KERNEL4x4_M2 \
"   pld [ r5 , #96 ]                         \n\t" \
"   fmacs   s16  , s4,  s12                     \n\t" \
"   fmacs   s17  , s5,  s12                     \n\t" \
"   fldmias r5!, { s0 - s3 }                    \n\t" \
"   fmacs   s18  , s6,  s12                     \n\t" \
"   pld [ r6 , #96 ]                         \n\t" \
"   fmacs   s19  , s7,  s12                     \n\t" \
\
"   fmacs   s20  , s4,  s13                     \n\t" \
"   fldmias r6!, { s8 - s11 }                   \n\t" \
"   fmacs   s21  , s5,  s13                     \n\t" \
"   fmacs   s22  , s6,  s13                     \n\t" \
"   fmacs   s23  , s7,  s13                     \n\t" \
\
"   fmacs   s24  , s4,  s14                     \n\t" \
"   fmacs   s25  , s5,  s14                     \n\t" \
"   fmacs   s26  , s6,  s14                     \n\t" \
"   fmacs   s27  , s7,  s14                     \n\t" \
\
"   fmacs   s28  , s4,  s15                     \n\t" \
"   fmacs   s29  , s5,  s15                     \n\t" \
"   fmacs   s30  , s6,  s15                     \n\t" \
"   fmacs   s31  , s7,  s15                     \n\t" \


#define KERNEL4x4_M1 \
"   fmacs   s16  , s0,  s8                      \n\t" \
"   fldmias r5!, { s4 - s7 }                    \n\t" \
"   fmacs   s17  , s1,  s8                      \n\t" \
"   fmacs   s18  , s2,  s8                      \n\t" \
"   fldmias r6!, { s12 - s15 }                  \n\t" \
"   fmacs   s19  , s3,  s8                      \n\t" \
\
"   fmacs   s20  , s0,  s9                      \n\t" \
"   fmacs   s21  , s1,  s9                      \n\t" \
"   fmacs   s22  , s2,  s9                      \n\t" \
"   fmacs   s23  , s3,  s9                      \n\t" \
\
"   fmacs   s24  , s0,  s10                     \n\t" \
"   fmacs   s25  , s1,  s10                     \n\t" \
"   fmacs   s26  , s2,  s10                     \n\t" \
"   fmacs   s27  , s3,  s10                     \n\t" \
\
"   fmacs   s28  , s0,  s11                     \n\t" \
"   fmacs   s29  , s1,  s11                     \n\t" \
"   fmacs   s30  , s2,  s11                     \n\t" \
"   fmacs   s31  , s3,  s11                     \n\t" \

#define KERNEL4x4_E \
"   fmacs   s16  , s4,  s12                     \n\t" \
"   fmacs   s17  , s5,  s12                     \n\t" \
"   fmacs   s18  , s6,  s12                     \n\t" \
"   fmacs   s19  , s7,  s12                     \n\t" \
\
"   fmacs   s20  , s4,  s13                     \n\t" \
"   fmacs   s21  , s5,  s13                     \n\t" \
"   fmacs   s22  , s6,  s13                     \n\t" \
"   fmacs   s23  , s7,  s13                     \n\t" \
\
"   fmacs   s24  , s4,  s14                     \n\t" \
"   fmacs   s25  , s5,  s14                     \n\t" \
"   fmacs   s26  , s6,  s14                     \n\t" \
"   fmacs   s27  , s7,  s14                     \n\t" \
\
"   fmacs   s28  , s4,  s15                     \n\t" \
"   fmacs   s29  , s5,  s15                     \n\t" \
"   fmacs   s30  , s6,  s15                     \n\t" \
"   fmacs   s31  , s7,  s15                     \n\t" \


#define KERNEL4x4_SUB \
"   flds    s8 , [ r6 ]                         \n\t" \
\
"   flds    s0 , [ r5 ]                         \n\t" \
"   flds    s1 , [ r5, #4 ]                     \n\t" \
\
"   fmacs   s16  , s0,  s8                      \n\t" \
"   flds    s2 , [ r5, #8 ]                     \n\t" \
"   fmacs   s17  , s1,  s8                      \n\t" \
"   flds    s3 , [ r5, #12 ]                    \n\t" \
"   fmacs   s18  , s2,  s8                      \n\t" \
"   flds    s9 , [ r6, #4 ]                     \n\t" \
"   fmacs   s19  , s3,  s8                      \n\t" \
\
"   flds    s10, [ r6, #8 ]                     \n\t" \
"   fmacs   s20  , s0,  s9                      \n\t" \
"   flds    s11, [ r6, #12 ]                    \n\t" \
"   fmacs   s21  , s1,  s9                      \n\t" \
"   fmacs   s22  , s2,  s9                      \n\t" \
"   fmacs   s23  , s3,  s9                      \n\t" \
\
"   fmacs   s24  , s0,  s10                     \n\t" \
"   fmacs   s25  , s1,  s10                     \n\t" \
"   fmacs   s26  , s2,  s10                     \n\t" \
"   fmacs   s27  , s3,  s10                     \n\t" \
\
"   fmacs   s28  , s0,  s11                     \n\t" \
"   fmacs   s29  , s1,  s11                     \n\t" \
"   add r5 , r5, #16                            \n\t" \
"   fmacs   s30  , s2,  s11                     \n\t" \
"   add r6 , r6, #16                            \n\t" \
"   fmacs   s31  , s3,  s11                     \n\t" \

#define ALPHA_SCALE \
"   fmuls s16, s16, s0                          \n\t" \
"   fmuls s17, s17, s0                          \n\t" \
"   fmuls s18, s18, s0                          \n\t" \
"   fmuls s19, s19, s0                          \n\t" \
"   fmuls s20, s20, s0                          \n\t" \
"   fmuls s21, s21, s0                          \n\t" \
"   fmuls s22, s22, s0                          \n\t" \
"   fmuls s23, s23, s0                          \n\t" \
"   fmuls s24, s24, s0                          \n\t" \
"   fmuls s25, s25, s0                          \n\t" \
"   fmuls s26, s26, s0                          \n\t" \
"   fmuls s27, s27, s0                          \n\t" \
"   fmuls s28, s28, s0                          \n\t" \
"   fmuls s29, s29, s0                          \n\t" \
"   fmuls s30, s30, s0                          \n\t" \
"   fmuls s31, s31, s0                          \n\t" \

#define STORE4x4 \
\
"   mov r0, r4                                  \n\t" \
"   mov r1, r5                                  \n\t" \
"   flds    s8, [ r4 ]                          \n\t" \
"   flds    s12, [ r5 ]                         \n\t" \
"   fmacs   s16, s8, s1                         \n\t" \
"   add r4, r4, r8                              \n\t" \
"   add r5, r5, r8                              \n\t" \
\
"   flds    s9, [ r4 ]                          \n\t" \
"   flds    s13, [ r5 ]                         \n\t" \
"   fmacs   s17, s9, s1                         \n\t" \
"   add r4, r4, r8                              \n\t" \
"   add r5, r5, r8                              \n\t" \
\
"   flds    s10, [ r4 ]                         \n\t" \
"   flds    s14, [ r5 ]                         \n\t" \
"   fmacs   s18, s10, s1                        \n\t" \
"   add r4, r4, r8                              \n\t" \
"   add r5, r5, r8                              \n\t" \
\
"   flds    s11, [ r4 ]                         \n\t" \
"   flds    s15, [ r5 ]                         \n\t" \
"   fmacs   s19, s11, s1                        \n\t" \
"   mov r4, r0                                  \n\t" \
"   mov r5, r1                                  \n\t" \
\
"   fsts    s16, [ r4 ]                          \n\t" \
"   add r4 , r4, r8                             \n\t" \
"   fsts    s17, [ r4 ]                          \n\t" \
"   add r4 , r4, r8                             \n\t" \
"   fsts    s18, [ r4 ]                         \n\t" \
"   add r4 , r4, r8                             \n\t" \
"   fsts    s19, [ r4 ]                         \n\t" \
\
"   mov r0, r6                                  \n\t" \
"   flds    s8, [ r6 ]                          \n\t" \
"   fmacs   s20, s12, s1                        \n\t" \
"   add r6, r6, r8                              \n\t" \
\
"   flds    s9, [ r6 ]                          \n\t" \
"   fmacs   s21, s13, s1                        \n\t" \
"   add r6, r6, r8                              \n\t" \
\
"   flds    s10, [ r6 ]                         \n\t" \
"   fmacs   s22, s14, s1                        \n\t" \
"   add r6, r6, r8                              \n\t" \
\
"   flds    s11, [ r6 ]                         \n\t" \
"   fmacs   s23, s15, s1                        \n\t" \
"   mov r6, r0                                  \n\t" \
\
"   fsts    s20, [ r5 ]                         \n\t" \
"   add r5 , r5, r8                             \n\t" \
"   fsts    s21, [ r5 ]                         \n\t" \
"   add r5 , r5, r8                             \n\t" \
"   fsts    s22, [ r5 ]                         \n\t" \
"   add r5 , r5, r8                             \n\t" \
"   fsts    s23, [ r5 ]                         \n\t" \
\
"   mov r0, r7                                  \n\t" \
"   flds    s12, [ r7 ]                         \n\t" \
"   fmacs   s24, s8, s1                          \n\t" \
"   add r7, r7, r8                              \n\t" \
\
"   flds    s13, [ r7 ]                         \n\t" \
"   fmacs   s25, s9, s1                          \n\t" \
"   add r7, r7, r8                              \n\t" \
\
"   flds    s14, [ r7 ]                         \n\t" \
"   fmacs   s26, s10, s1                        \n\t" \
"   add r7, r7, r8                              \n\t" \
\
"   flds    s15, [ r7 ]                         \n\t" \
"   fmacs   s27, s11, s1                        \n\t" \
"   mov r7, r0                                  \n\t" \
\
"   fsts    s24, [ r6 ]                          \n\t" \
"   fmacs   s28, s12, s1                        \n\t" \
"   add r6 , r6, r8                             \n\t" \
\
"   fsts    s25, [ r6 ]                          \n\t" \
"   fmacs   s29, s13, s1                        \n\t" \
"   add r6 , r6, r8                             \n\t" \
\
"   fsts    s26, [ r6 ]                         \n\t" \
"   fmacs   s30, s14, s1                        \n\t" \
"   add r6 , r6, r8                             \n\t" \
\
"   fsts    s27, [ r6 ]                         \n\t" \
"   fmacs   s31, s15, s1                        \n\t" \
"   fsts    s28, [ r7 ]                         \n\t" \
\
"   add r7 , r7, r8                             \n\t" \
"   fsts    s29, [ r7 ]                         \n\t" \
"   add r7 , r7, r8                             \n\t" \
"   fsts    s30, [ r7 ]                         \n\t" \
"   add r7 , r7, r8                             \n\t" \
"   fsts    s31, [ r7 ]                         \n\t" \

#define STORE4x4_BETAZERO \
\
"   fsts    s16, [ r4 ]                          \n\t" \
"   add r4 , r4, r8                             \n\t" \
"   fsts    s17, [ r4 ]                          \n\t" \
"   add r4 , r4, r8                             \n\t" \
"   fsts    s18, [ r4 ]                         \n\t" \
"   add r4 , r4, r8                             \n\t" \
"   fsts    s19, [ r4 ]                         \n\t" \
\
"   fsts    s20, [ r5 ]                         \n\t" \
"   add r5 , r5, r8                             \n\t" \
"   fsts    s21, [ r5 ]                         \n\t" \
"   add r5 , r5, r8                             \n\t" \
"   fsts    s22, [ r5 ]                         \n\t" \
"   add r5 , r5, r8                             \n\t" \
"   fsts    s23, [ r5 ]                         \n\t" \
\
"   fsts    s24, [ r6 ]                          \n\t" \
"   add r6 , r6, r8                             \n\t" \
"   fsts    s25, [ r6 ]                          \n\t" \
"   add r6 , r6, r8                             \n\t" \
"   fsts    s26, [ r6 ]                         \n\t" \
"   add r6 , r6, r8                             \n\t" \
"   fsts    s27, [ r6 ]                         \n\t" \
\
"   fsts    s28, [ r7 ]                         \n\t" \
"   add r7 , r7, r8                             \n\t" \
"   fsts    s29, [ r7 ]                         \n\t" \
"   add r7 , r7, r8                             \n\t" \
"   fsts    s30, [ r7 ]                         \n\t" \
"   add r7 , r7, r8                             \n\t" \
"   fsts    s31, [ r7 ]                         \n\t" \


/**************************************************************************************
* End of macro definitions
**************************************************************************************/

void bli_sgemm_arm32vfp_asm_4x4
     (
       dim_t               k0,
       float*     restrict alpha,
       float*     restrict a,
       float*     restrict b,
       float*     restrict beta,
       float*     restrict c, inc_t rs_c0, inc_t cs_c0,
       auxinfo_t* restrict data,
       cntx_t*    restrict cntx
     )
{
    void* a_next = bli_auxinfo_next_a( data );
    void* b_next = bli_auxinfo_next_b( data );

    // Typecast local copies of integers in case dim_t and inc_t are a
    // different size than is expected by load instructions.
    uint32_t k_iter = k0 / 8;
    uint32_t k_left = k0 % 8;
    uint32_t rs_c   = rs_c0;
    uint32_t cs_c   = cs_c0;
    float _one = 1.0;
    float *one = &_one;

__asm__ volatile (
" ldr r0,%[k_iter]                  \n\t"
" ldr r1,%[k_left]                  \n\t"
" ldr r2,%[alpha]                   \n\t"
" ldr r3,%[beta]                    \n\t"
" ldr r4,%[caddr]                   \n\t"
" ldr r5,%[aaddr]                   \n\t"
" ldr r6,%[baddr]                   \n\t"
" ldr r7,%[cs_c]                    \n\t"
" ldr r8,%[rs_c]                    \n\t"
" ldr r11,%[one]                    \n\t"
" cmp r0,#2                         \n\t"
" blt K_ITER_LE_TWO                 \n\t" // sgemm_kernel_L4_M4_32

    KERNEL4x4_I
    KERNEL4x4_M2
    KERNEL4x4_M1
    KERNEL4x4_M2

    KERNEL4x4_M1
    KERNEL4x4_M2
    KERNEL4x4_M1
    KERNEL4x4_M2

" subs r0, r0, #2                   \n\t"
" ble LAST_MAIN_LOOP                \n\t"

" .align 5                          \n\t"
" MAIN_LOOP:                        \n\t"

    KERNEL4x4_M1
    KERNEL4x4_M2
    KERNEL4x4_M1
    KERNEL4x4_M2

    KERNEL4x4_M1
    KERNEL4x4_M2
    KERNEL4x4_M1
    KERNEL4x4_M2

" subs r0, r0, #1                   \n\t"
" bgt MAIN_LOOP                     \n\t"

" LAST_MAIN_LOOP:                   \n\t"

    KERNEL4x4_M1
    KERNEL4x4_M2
    KERNEL4x4_M1
    KERNEL4x4_M2

    KERNEL4x4_M1
    KERNEL4x4_M2
    KERNEL4x4_M1
    KERNEL4x4_E

" b K_MAYBE_LEFT_LOOP               \n\t"

" K_ITER_LE_TWO:                    \n\t" // sgemm_kernel_L4_M4_32
" cmp r0, #0                        \n\t"
" beq K_ITER_ZERO                   \n\t" // sgemm_kernel_L4_M4_40

    KERNEL4x4_I
    KERNEL4x4_M2
    KERNEL4x4_M1
    KERNEL4x4_M2

    KERNEL4x4_M1
    KERNEL4x4_M2
    KERNEL4x4_M1
    KERNEL4x4_E

" b K_MAYBE_LEFT_LOOP               \n\t" // sgemm_kernel_L4_M4_44

" K_ITER_ZERO:                      \n\t" // sgemm_kernel_L4_M4_40
    INIT4x4

" K_MAYBE_LEFT_LOOP:                \n\t"
" cmp r1, #0                        \n\t"
" beq SAVE                          \n\t"

" K_LEFT_LOOP:                      \n\t"
    KERNEL4x4_SUB

" subs r1, r1, #1                   \n\t"
" bgt K_LEFT_LOOP                   \n\t"

" SAVE:                             \n\t"

" flds s0, [ r2 ]                   \n\t" // s0 <- alpha

" flds s1, [ r11 ]                  \n\t" // s1 <- 1.0
" vcmpe.f32 s0, s1                  \n\t"
" vmrs APSR_nzcv, FPSCR             \n\t"
" beq STORE                         \n\t" 

    ALPHA_SCALE

" STORE:                            \n\t"

" flds s1, [ r3 ]                   \n\t" // s1 <- beta
" lsl r7, r7, #2                    \n\t" // r7 *= sizeof(float)
" lsl r8, r8, #2                    \n\t" // r8 *= sizeof(float)
" add r5, r4, r7                    \n\t" // we no longer care for A and B
" add r6, r5, r7                    \n\t"
" add r7, r6, r7                    \n\t" // r4,r5,r6,r7 are now addr for cols of C
                                          // r0,r1,r2,r3 are now scratch
                                          // r8 is rs_c*sizeof(float)

" vcmpe.f32 s1, #0                  \n\t"
" vmrs APSR_nzcv, FPSCR             \n\t"
" beq STORE4x4_BETAZERO             \n\t" 

    STORE4x4

" b DONE                            \n\t"
" STORE4x4_BETAZERO:                \n\t"

    STORE4x4_BETAZERO

" DONE:                             \n\t"

:// Output (none)
:// Input
 [k_iter] "m" (k_iter), // r0
 [k_left] "m" (k_left), // r1
 [alpha]  "m" (alpha),  // r2
 [beta]   "m" (beta),   // r3
 [caddr]  "m" (c),      // r4
 [aaddr]  "m" (a),      // r5
 [baddr]  "m" (b),      // r6
 [cs_c]   "m" (cs_c),   // r7
 [rs_c]   "m" (rs_c),   // r8
 [a_next] "m" (a_next), // r9
 [b_next] "m" (b_next), // r10
 [one]    "m" (one)     // r11
: // Clobber
  "r0",  "r1",  "r2",  "r3",
  "r4",  "r5",  "r6",  "r7",
  "r8",  "r9", "r10", "r11",

  "s0",  "s1",  "s2",  "s3",
  "s4",  "s5",  "s6",  "s7",
  "s8",  "s9", "s10", "s11",
 "s12", "s13", "s14", "s15",
 "s16", "s17", "s18", "s19",
 "s20", "s21", "s22", "s23",
 "s24", "s25", "s26", "s27",
 "s28", "s29", "s30", "s31"
);
}
