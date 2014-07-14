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
   THEORY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
   (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
   OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

*/

#include "blis.h"

void bli_sgemmtrsm_l_opt_8x4(
                              dim_t              k,
                              float* restrict    alpha,
                              float* restrict    a10,
                              float* restrict    a11,
                              float* restrict    b01,
                              float* restrict    b11,
                              float* restrict    c11, inc_t rs_c, inc_t cs_c,
                              auxinfo_t*         data
                            )
{
	/* Just call the reference implementation. */
	BLIS_SGEMMTRSM_L_UKERNEL_REF( k,
	                         alpha,
	                         a10,
	                         a11,
	                         b01,
	                         b11,
	                         c11, rs_c, cs_c,
	                         data );
}

void bli_dgemmtrsm_l_opt_4x4(
                              dim_t              k,
                              double* restrict   alpha,
                              double* restrict   a10,
                              double* restrict   a11,
                              double* restrict   b01,
                              double* restrict   b11,
                              double* restrict   c11, inc_t rs_c, inc_t cs_c,
                              auxinfo_t*         data
                            )
{
	void*   b_next  = bli_auxinfo_next_b( data );

	dim_t   k_iter  = k / 4;
	dim_t   k_left  = k % 4;

	__asm__ volatile
	(
		"                                \n\t"
		"movq          %2, %%rax         \n\t" // load address of a10.
		"movq          %4, %%rbx         \n\t" // load address of b01.
		//"movq         %10, %%r9          \n\t" // load address of b_next.
		"                                \n\t"
		"subq    $-8 * 16, %%rax         \n\t" // increment pointers to allow byte
		"subq    $-8 * 16, %%rbx         \n\t" // offsets in the unrolled iterations.
		"                                \n\t"
		"movaps  -8 * 16(%%rax), %%xmm0  \n\t" // initialize loop by pre-loading elements
		"movaps  -7 * 16(%%rax), %%xmm1  \n\t" // of a and b.
		"movaps  -8 * 16(%%rbx), %%xmm2  \n\t"
		"                                \n\t"
		//"movq          %6, %%rcx         \n\t" // load address of c11
		//"movq          %9, %%rdi         \n\t" // load cs_c
		//"leaq        (,%%rdi,8), %%rdi   \n\t" // cs_c *= sizeof(double)
		//"leaq   (%%rcx,%%rdi,2), %%rdx   \n\t" // load address of c + 2*cs_c;
		"                                \n\t"
		//"prefetcht2   0 * 8(%%r9)        \n\t" // prefetch b_next
		"                                \n\t"
		"xorpd     %%xmm3,  %%xmm3       \n\t"
		"xorpd     %%xmm4,  %%xmm4       \n\t"
		"xorpd     %%xmm5,  %%xmm5       \n\t"
		"xorpd     %%xmm6,  %%xmm6       \n\t"
		"                                \n\t"
		//"prefetcht2   3 * 8(%%rcx)       \n\t" // prefetch c + 0*cs_c
		"xorpd     %%xmm8,  %%xmm8       \n\t"
		"movaps    %%xmm8,  %%xmm9       \n\t"
		//"prefetcht2   3 * 8(%%rcx,%%rdi) \n\t" // prefetch c + 1*cs_c
		"movaps    %%xmm8, %%xmm10       \n\t"
		"movaps    %%xmm8, %%xmm11       \n\t"
		//"prefetcht2   3 * 8(%%rdx)       \n\t" // prefetch c + 2*cs_c
		"movaps    %%xmm8, %%xmm12       \n\t"
		"movaps    %%xmm8, %%xmm13       \n\t"
		//"prefetcht2   3 * 8(%%rdx,%%rdi) \n\t" // prefetch c + 3*cs_c
		"movaps    %%xmm8, %%xmm14       \n\t"
		"movaps    %%xmm8, %%xmm15       \n\t"
		"                                \n\t"
		"                                \n\t"
		"                                \n\t"
		"movq      %0, %%rsi             \n\t" // i = k_iter;
		"testq  %%rsi, %%rsi             \n\t" // check i via logical AND.
		"je     .CONSIDERKLEFT           \n\t" // if i == 0, jump to code that
		"                                \n\t" // contains the k_left loop.
		"                                \n\t"
		"                                \n\t"
		".LOOPKITER:                     \n\t" // MAIN LOOP
		"                                \n\t"
		//"prefetcht0 1264(%%rax)          \n\t"
		"prefetcht0  (4*35+1) * 8(%%rax) \n\t"
		"                                \n\t"
		"addpd   %%xmm3, %%xmm11         \n\t" // iteration 0
		"movaps  -7 * 16(%%rbx), %%xmm3  \n\t"
		"addpd   %%xmm4, %%xmm15         \n\t"
		"movaps  %%xmm2, %%xmm4          \n\t"
		"pshufd   $0x4e, %%xmm2, %%xmm7  \n\t"
		"mulpd   %%xmm0, %%xmm2          \n\t"
		"mulpd   %%xmm1, %%xmm4          \n\t"
		"                                \n\t"
		"addpd   %%xmm5, %%xmm10         \n\t"
		"addpd   %%xmm6, %%xmm14         \n\t"
		"movaps  %%xmm7, %%xmm6          \n\t"
		"mulpd   %%xmm0, %%xmm7          \n\t"
		"mulpd   %%xmm1, %%xmm6          \n\t"
		"                                \n\t"
		"addpd   %%xmm2, %%xmm9          \n\t"
		"movaps  -6 * 16(%%rbx), %%xmm2  \n\t"
		"addpd   %%xmm4, %%xmm13         \n\t"
		"movaps  %%xmm3, %%xmm4          \n\t"
		"pshufd   $0x4e, %%xmm3, %%xmm5  \n\t"
		"mulpd   %%xmm0, %%xmm3          \n\t"
		"mulpd   %%xmm1, %%xmm4          \n\t"
		"                                \n\t"
		"addpd   %%xmm7, %%xmm8          \n\t"
		"addpd   %%xmm6, %%xmm12         \n\t"
		"movaps  %%xmm5, %%xmm6          \n\t"
		"mulpd   %%xmm0, %%xmm5          \n\t"
		"movaps  -6 * 16(%%rax), %%xmm0  \n\t"
		"mulpd   %%xmm1, %%xmm6          \n\t"
		"movaps  -5 * 16(%%rax), %%xmm1  \n\t"
		"                                \n\t"
		"                                \n\t"
		"addpd   %%xmm3, %%xmm11         \n\t" // iteration 1
		"movaps  -5 * 16(%%rbx), %%xmm3  \n\t"
		"addpd   %%xmm4, %%xmm15         \n\t"
		"movaps  %%xmm2, %%xmm4          \n\t"
		"pshufd   $0x4e, %%xmm2, %%xmm7  \n\t"
		"mulpd   %%xmm0, %%xmm2          \n\t"
		"mulpd   %%xmm1, %%xmm4          \n\t"
		"                                \n\t"
		"addpd   %%xmm5, %%xmm10         \n\t"
		"addpd   %%xmm6, %%xmm14         \n\t"
		"movaps  %%xmm7, %%xmm6          \n\t"
		"mulpd   %%xmm0, %%xmm7          \n\t"
		"mulpd   %%xmm1, %%xmm6          \n\t"
		"                                \n\t"
		"addpd   %%xmm2, %%xmm9          \n\t"
		"movaps  -4 * 16(%%rbx), %%xmm2  \n\t"
		"addpd   %%xmm4, %%xmm13         \n\t"
		"movaps  %%xmm3, %%xmm4          \n\t"
		"pshufd   $0x4e, %%xmm3, %%xmm5  \n\t"
		"mulpd   %%xmm0, %%xmm3          \n\t"
		"mulpd   %%xmm1, %%xmm4          \n\t"
		"                                \n\t"
		"addpd   %%xmm7, %%xmm8          \n\t"
		"addpd   %%xmm6, %%xmm12         \n\t"
		"movaps  %%xmm5, %%xmm6          \n\t"
		"mulpd   %%xmm0, %%xmm5          \n\t"
		"movaps  -4 * 16(%%rax), %%xmm0  \n\t"
		"mulpd   %%xmm1, %%xmm6          \n\t"
		"movaps  -3 * 16(%%rax), %%xmm1  \n\t"
		"                                \n\t"
		//"prefetcht0 1328(%%rax)          \n\t"
		"prefetcht0  (4*37+1) * 8(%%rax) \n\t"
		"                                \n\t"
		"addpd   %%xmm3, %%xmm11         \n\t" // iteration 2
		"movaps  -3 * 16(%%rbx), %%xmm3  \n\t"
		"addpd   %%xmm4, %%xmm15         \n\t"
		"movaps  %%xmm2, %%xmm4          \n\t"
		"pshufd   $0x4e, %%xmm2, %%xmm7  \n\t"
		"mulpd   %%xmm0, %%xmm2          \n\t"
		"mulpd   %%xmm1, %%xmm4          \n\t"
		"                                \n\t"
		"addpd   %%xmm5, %%xmm10         \n\t"
		"addpd   %%xmm6, %%xmm14         \n\t"
		"movaps  %%xmm7, %%xmm6          \n\t"
		"mulpd   %%xmm0, %%xmm7          \n\t"
		"mulpd   %%xmm1, %%xmm6          \n\t"
		"                                \n\t"
		"addpd   %%xmm2, %%xmm9          \n\t"
		"movaps  -2 * 16(%%rbx), %%xmm2  \n\t"
		"addpd   %%xmm4, %%xmm13         \n\t"
		"movaps  %%xmm3, %%xmm4          \n\t"
		"pshufd   $0x4e, %%xmm3, %%xmm5  \n\t"
		"mulpd   %%xmm0, %%xmm3          \n\t"
		"mulpd   %%xmm1, %%xmm4          \n\t"
		"                                \n\t"
		"addpd   %%xmm7, %%xmm8          \n\t"
		"addpd   %%xmm6, %%xmm12         \n\t"
		"movaps  %%xmm5, %%xmm6          \n\t"
		"mulpd   %%xmm0, %%xmm5          \n\t"
		"movaps  -2 * 16(%%rax), %%xmm0  \n\t"
		"mulpd   %%xmm1, %%xmm6          \n\t"
		"movaps  -1 * 16(%%rax), %%xmm1  \n\t"
		"                                \n\t"
		"                                \n\t"
		"addpd   %%xmm3, %%xmm11         \n\t" // iteration 3
		"movaps  -1 * 16(%%rbx), %%xmm3  \n\t"
		"addpd   %%xmm4, %%xmm15         \n\t"
		"movaps  %%xmm2, %%xmm4          \n\t"
		"pshufd   $0x4e, %%xmm2, %%xmm7  \n\t"
		"mulpd   %%xmm0, %%xmm2          \n\t"
		"mulpd   %%xmm1, %%xmm4          \n\t"
		"                                \n\t"
		"subq  $-4 * 4 * 8, %%rax        \n\t" // a += 4*4 (unroll x mr)
		"                                \n\t"
		"addpd   %%xmm5, %%xmm10         \n\t"
		"addpd   %%xmm6, %%xmm14         \n\t"
		"movaps  %%xmm7, %%xmm6          \n\t"
		"mulpd   %%xmm0, %%xmm7          \n\t"
		"mulpd   %%xmm1, %%xmm6          \n\t"
		"                                \n\t"
		//"subq  $-4 * 4 * 8, %%r9         \n\t" // b_next += 4*4 (unroll x nr)
		"                                \n\t"
		"addpd   %%xmm2, %%xmm9          \n\t"
		"movaps   0 * 16(%%rbx), %%xmm2  \n\t"
		"addpd   %%xmm4, %%xmm13         \n\t"
		"movaps  %%xmm3, %%xmm4          \n\t"
		"pshufd   $0x4e, %%xmm3, %%xmm5  \n\t"
		"mulpd   %%xmm0, %%xmm3          \n\t"
		"mulpd   %%xmm1, %%xmm4          \n\t"
		"                                \n\t"
		"subq  $-4 * 4 * 8, %%rbx        \n\t" // b += 4*4 (unroll x nr)
		"                                \n\t"
		"addpd   %%xmm7, %%xmm8          \n\t"
		"addpd   %%xmm6, %%xmm12         \n\t"
		"movaps  %%xmm5, %%xmm6          \n\t"
		"mulpd   %%xmm0, %%xmm5          \n\t"
		"movaps  -8 * 16(%%rax), %%xmm0  \n\t"
		"mulpd   %%xmm1, %%xmm6          \n\t"
		"movaps  -7 * 16(%%rax), %%xmm1  \n\t"
		"                                \n\t"
		//"prefetcht2        0 * 8(%%r9)   \n\t" // prefetch b_next[0]
		//"prefetcht2        8 * 8(%%r9)   \n\t" // prefetch b_next[8]
		"                                \n\t"
		"                                \n\t"
		"decq   %%rsi                    \n\t" // i -= 1;
		"jne    .LOOPKITER               \n\t" // iterate again if i != 0.
		"                                \n\t"
		"                                \n\t"
		"                                \n\t"
		".CONSIDERKLEFT:                 \n\t"
		"                                \n\t"
		"movq      %1, %%rsi             \n\t" // i = k_left;
		"testq  %%rsi, %%rsi             \n\t" // check i via logical AND.
		"je     .POSTACCUM               \n\t" // if i == 0, we're done; jump to end.
		"                                \n\t" // else, we prepare to enter k_left loop.
		"                                \n\t"
		"                                \n\t"
		".LOOPKLEFT:                     \n\t" // EDGE LOOP
		"                                \n\t"
		"addpd   %%xmm3, %%xmm11         \n\t" // iteration 0
		"movaps  -7 * 16(%%rbx), %%xmm3  \n\t"
		"addpd   %%xmm4, %%xmm15         \n\t"
		"movaps  %%xmm2, %%xmm4          \n\t"
		"pshufd   $0x4e, %%xmm2, %%xmm7  \n\t"
		"mulpd   %%xmm0, %%xmm2          \n\t"
		"mulpd   %%xmm1, %%xmm4          \n\t"
		"                                \n\t"
		"addpd   %%xmm5, %%xmm10         \n\t"
		"addpd   %%xmm6, %%xmm14         \n\t"
		"movaps  %%xmm7, %%xmm6          \n\t"
		"mulpd   %%xmm0, %%xmm7          \n\t"
		"mulpd   %%xmm1, %%xmm6          \n\t"
		"                                \n\t"
		"addpd   %%xmm2, %%xmm9          \n\t"
		"movaps  -6 * 16(%%rbx), %%xmm2  \n\t"
		"addpd   %%xmm4, %%xmm13         \n\t"
		"movaps  %%xmm3, %%xmm4          \n\t"
		"pshufd   $0x4e, %%xmm3, %%xmm5  \n\t"
		"mulpd   %%xmm0, %%xmm3          \n\t"
		"mulpd   %%xmm1, %%xmm4          \n\t"
		"                                \n\t"
		"addpd   %%xmm7, %%xmm8          \n\t"
		"addpd   %%xmm6, %%xmm12         \n\t"
		"movaps  %%xmm5, %%xmm6          \n\t"
		"mulpd   %%xmm0, %%xmm5          \n\t"
		"movaps  -6 * 16(%%rax), %%xmm0  \n\t"
		"mulpd   %%xmm1, %%xmm6          \n\t"
		"movaps  -5 * 16(%%rax), %%xmm1  \n\t"
		"                                \n\t"
		"                                \n\t"
		"subq  $-4 * 1 * 8, %%rax        \n\t" // a += 4 (1 x mr)
		"subq  $-4 * 1 * 8, %%rbx        \n\t" // b += 4 (1 x nr)
		"                                \n\t"
		"                                \n\t"
		"decq   %%rsi                    \n\t" // i -= 1;
		"jne    .LOOPKLEFT               \n\t" // iterate again if i != 0.
		"                                \n\t"
		"                                \n\t"
		"                                \n\t"
		".POSTACCUM:                     \n\t"
		"                                \n\t"
		"addpd   %%xmm3, %%xmm11         \n\t"
		"addpd   %%xmm4, %%xmm15         \n\t"
		"addpd   %%xmm5, %%xmm10         \n\t"
		"addpd   %%xmm6, %%xmm14         \n\t"
		"                                \n\t"
		"                                \n\t"
		"                                \n\t"
		"movq      %5, %%rbx             \n\t" // load address of b11.
		"                                \n\t"
		"                                \n\t" // xmm8:   xmm9:   xmm10:  xmm11:
		"                                \n\t" // ( ab01  ( ab00  ( ab03  ( ab02
		"                                \n\t" //   ab10 )  ab11 )  ab12 )  ab13 )
		"                                \n\t" //
		"                                \n\t" // xmm12:  xmm13:  xmm14:  xmm15:
		"                                \n\t" // ( ab21  ( ab20  ( ab23  ( ab22
		"                                \n\t" //   ab30 )  ab31 )  ab32 )  ab33 )
		"movaps   %%xmm9,  %%xmm0        \n\t"
		"movaps   %%xmm8,  %%xmm1        \n\t"
		"unpcklpd %%xmm8,  %%xmm0        \n\t"
		"unpckhpd %%xmm9,  %%xmm1        \n\t"
		"                                \n\t"
		"movaps   %%xmm11, %%xmm4        \n\t"
		"movaps   %%xmm10, %%xmm5        \n\t"
		"unpcklpd %%xmm10, %%xmm4        \n\t"
		"unpckhpd %%xmm11, %%xmm5        \n\t"
		"                                \n\t"
		"movaps   %%xmm13, %%xmm2        \n\t"
		"movaps   %%xmm12, %%xmm3        \n\t"
		"unpcklpd %%xmm12, %%xmm2        \n\t"
		"unpckhpd %%xmm13, %%xmm3        \n\t"
		"                                \n\t"
		"movaps   %%xmm15, %%xmm6        \n\t"
		"movaps   %%xmm14, %%xmm7        \n\t"
		"unpcklpd %%xmm14, %%xmm6        \n\t"
		"unpckhpd %%xmm15, %%xmm7        \n\t"
		"                                \n\t"
		"                                \n\t" // xmm0: ( ab00 ab01 ) xmm4: ( ab02 ab03 )
		"                                \n\t" // xmm1: ( ab10 ab11 ) xmm5: ( ab12 ab13 )
		"                                \n\t" // xmm2: ( ab20 ab21 ) xmm6: ( ab22 ab23 )
		"                                \n\t" // xmm3: ( ab30 ab31 ) xmm7: ( ab32 ab33 )
		"                                \n\t"
		"movq    %9, %%rax               \n\t" // load address of alpha
		"movddup (%%rax), %%xmm15        \n\t" // load alpha and duplicate
		"                                \n\t"
		"movaps  0 * 16(%%rbx), %%xmm8   \n\t" 
		"movaps  1 * 16(%%rbx), %%xmm12  \n\t"
		"mulpd    %%xmm15, %%xmm8        \n\t" // xmm8  = alpha * ( beta00 beta01 )
		"mulpd    %%xmm15, %%xmm12       \n\t" // xmm12 = alpha * ( beta02 beta03 )
		"movaps  2 * 16(%%rbx), %%xmm9   \n\t"
		"movaps  3 * 16(%%rbx), %%xmm13  \n\t"
		"mulpd    %%xmm15, %%xmm9        \n\t" // xmm9  = alpha * ( beta10 beta11 )
		"mulpd    %%xmm15, %%xmm13       \n\t" // xmm13 = alpha * ( beta12 beta13 )
		"movaps  4 * 16(%%rbx), %%xmm10  \n\t"
		"movaps  5 * 16(%%rbx), %%xmm14  \n\t"
		"mulpd    %%xmm15, %%xmm10       \n\t" // xmm10 = alpha * ( beta20 beta21 )
		"mulpd    %%xmm15, %%xmm14       \n\t" // xmm14 = alpha * ( beta22 beta23 )
		"movaps  6 * 16(%%rbx), %%xmm11  \n\t"
		"mulpd    %%xmm15, %%xmm11       \n\t" // xmm11 = alpha * ( beta30 beta31 )
		"mulpd   7 * 16(%%rbx), %%xmm15  \n\t" // xmm15 = alpha * ( beta32 beta33 )
		"                                \n\t"
		"                                \n\t" // (Now scaled by alpha:)
		"                                \n\t" // xmm8:  ( beta00 beta01 ) xmm12: ( beta02 beta03 )
		"                                \n\t" // xmm9:  ( beta10 beta11 ) xmm13: ( beta12 beta13 )
		"                                \n\t" // xmm10: ( beta20 beta21 ) xmm14: ( beta22 beta23 )
		"                                \n\t" // xmm11: ( beta30 beta31 ) xmm15: ( beta32 beta33 )
		"                                \n\t"
		"subpd    %%xmm0, %%xmm8         \n\t" // xmm8  -= xmm0
		"subpd    %%xmm1, %%xmm9         \n\t" // xmm9  -= xmm1
		"subpd    %%xmm2, %%xmm10        \n\t" // xmm10 -= xmm2
		"subpd    %%xmm3, %%xmm11        \n\t" // xmm11 -= xmm3
		"subpd    %%xmm4, %%xmm12        \n\t" // xmm12 -= xmm4
		"subpd    %%xmm5, %%xmm13        \n\t" // xmm13 -= xmm5
		"subpd    %%xmm6, %%xmm14        \n\t" // xmm14 -= xmm6
		"subpd    %%xmm7, %%xmm15        \n\t" // xmm15 -= xmm7
		"                                \n\t"
		"                                \n\t"
		"                                \n\t"
		".TRSM:                          \n\t"
		"                                \n\t"
		"                                \n\t"
		"movq     %3, %%rax                \n\t" // load address of a11
		"movq     %6, %%rcx                \n\t" // load address of c11
		"                                  \n\t"
		"movq     %7, %%rsi                \n\t" // load rs_c
		"movq     %8, %%rdi                \n\t" // load cs_c
		"salq     $3, %%rsi                \n\t" // rs_c *= sizeof( double )
		"salq     $3, %%rdi                \n\t" // cs_c *= sizeof( double )
		"                                  \n\t"
		"leaq   (%%rcx,%%rdi,2), %%rdx     \n\t" // c11_2 = c11 + 2*cs_c
		"                                  \n\t"
		"                                  \n\t"
		"                                  \n\t"
		"                                  \n\t" // iteration 0
		"                                  \n\t"
		"movddup (0+0*4)*8(%%rax), %%xmm0  \n\t" // load xmm0 = (1/alpha00)
		"                                  \n\t"
		"mulpd    %%xmm0, %%xmm8           \n\t" // xmm8  *= (1/alpha00);
		"mulpd    %%xmm0, %%xmm12          \n\t" // xmm12 *= (1/alpha00);
		"                                  \n\t"
		"movaps   %%xmm8,  0 * 16(%%rbx)   \n\t" // store ( beta00 beta01 ) = xmm8
		"movaps   %%xmm12, 1 * 16(%%rbx)   \n\t" // store ( beta02 beta03 ) = xmm12
		"movlpd   %%xmm8,  (%%rcx)         \n\t" // store ( gamma00 ) = xmm8[0]
		"movhpd   %%xmm8,  (%%rcx,%%rdi)   \n\t" // store ( gamma01 ) = xmm8[1]
		"movlpd   %%xmm12, (%%rdx)         \n\t" // store ( gamma02 ) = xmm12[0]
		"movhpd   %%xmm12, (%%rdx,%%rdi)   \n\t" // store ( gamma03 ) = xmm12[1]
		"addq     %%rsi, %%rcx             \n\t" // c11   += rs_c
		"addq     %%rsi, %%rdx             \n\t" // c11_2 += rs_c
		"                                  \n\t"
		"                                  \n\t"
		"                                  \n\t"
		"                                  \n\t" // iteration 1
		"                                  \n\t"
		"movddup (1+0*4)*8(%%rax), %%xmm0  \n\t" // load xmm0 = alpha10
		"movddup (1+1*4)*8(%%rax), %%xmm1  \n\t" // load xmm1 = (1/alpha11)
		"                                  \n\t"
		"movaps   %%xmm0,  %%xmm4          \n\t" // xmm4 = xmm0
		"mulpd    %%xmm8,  %%xmm0          \n\t" // xmm0 = alpha10 * ( beta00 beta01 )
		"mulpd    %%xmm12, %%xmm4          \n\t" // xmm4 = alpha10 * ( beta02 beta03 )
		"subpd    %%xmm0,  %%xmm9          \n\t" // xmm9  -= xmm0
		"subpd    %%xmm4,  %%xmm13         \n\t" // xmm13 -= xmm4
		"mulpd    %%xmm1,  %%xmm9          \n\t" // xmm9  *= (1/alpha11);
		"mulpd    %%xmm1,  %%xmm13         \n\t" // xmm13 *= (1/alpha11);
		"                                  \n\t"
		"movaps   %%xmm9,  2 * 16(%%rbx)   \n\t" // store ( beta10 beta11 ) = xmm9
		"movaps   %%xmm13, 3 * 16(%%rbx)   \n\t" // store ( beta12 beta13 ) = xmm13
		"movlpd   %%xmm9,  (%%rcx)         \n\t" // store ( gamma10 ) = xmm9[0]
		"movhpd   %%xmm9,  (%%rcx,%%rdi)   \n\t" // store ( gamma11 ) = xmm9[1]
		"movlpd   %%xmm13, (%%rdx)         \n\t" // store ( gamma12 ) = xmm13[0]
		"movhpd   %%xmm13, (%%rdx,%%rdi)   \n\t" // store ( gamma13 ) = xmm13[1]
		"addq     %%rsi, %%rcx             \n\t" // c11   += rs_c
		"addq     %%rsi, %%rdx             \n\t" // c11_2 += rs_c
		"                                  \n\t"
		"                                  \n\t"
		"                                  \n\t"
		"                                  \n\t" // iteration 2
		"                                  \n\t"
		"movddup (2+0*4)*8(%%rax), %%xmm0  \n\t" // load xmm0 = alpha20
		"movddup (2+1*4)*8(%%rax), %%xmm1  \n\t" // load xmm1 = alpha21
		"movddup (2+2*4)*8(%%rax), %%xmm2  \n\t" // load xmm2 = (1/alpha22)
		"                                  \n\t"
		"movaps   %%xmm0,  %%xmm4          \n\t" // xmm4 = xmm0
		"movaps   %%xmm1,  %%xmm5          \n\t" // xmm5 = xmm1
		"mulpd    %%xmm8,  %%xmm0          \n\t" // xmm0 = alpha20 * ( beta00 beta01 )
		"mulpd    %%xmm12, %%xmm4          \n\t" // xmm4 = alpha20 * ( beta02 beta03 )
		"mulpd    %%xmm9,  %%xmm1          \n\t" // xmm1 = alpha21 * ( beta10 beta11 )
		"mulpd    %%xmm13, %%xmm5          \n\t" // xmm5 = alpha21 * ( beta12 beta13 )
		"addpd    %%xmm1,  %%xmm0          \n\t" // xmm0 += xmm1;
		"addpd    %%xmm5,  %%xmm4          \n\t" // xmm4 += xmm5;
		"subpd    %%xmm0,  %%xmm10         \n\t" // xmm10 -= xmm0
		"subpd    %%xmm4,  %%xmm14         \n\t" // xmm14 -= xmm4
		"mulpd    %%xmm2,  %%xmm10         \n\t" // xmm10 *= (1/alpha22);
		"mulpd    %%xmm2,  %%xmm14         \n\t" // xmm14 *= (1/alpha22);
		"                                  \n\t"
		"movaps   %%xmm10, 4 * 16(%%rbx)   \n\t" // store ( beta20 beta21 ) = xmm10
		"movaps   %%xmm14, 5 * 16(%%rbx)   \n\t" // store ( beta22 beta23 ) = xmm14
		"movlpd   %%xmm10, (%%rcx)         \n\t" // store ( gamma20 ) = xmm10[0]
		"movhpd   %%xmm10, (%%rcx,%%rdi)   \n\t" // store ( gamma21 ) = xmm10[1]
		"movlpd   %%xmm14, (%%rdx)         \n\t" // store ( gamma22 ) = xmm14[0]
		"movhpd   %%xmm14, (%%rdx,%%rdi)   \n\t" // store ( gamma23 ) = xmm14[1]
		"addq     %%rsi, %%rcx             \n\t" // c11   += rs_c
		"addq     %%rsi, %%rdx             \n\t" // c11_2 += rs_c
		"                                  \n\t"
		"                                  \n\t"
		"                                  \n\t"
		"                                  \n\t" // iteration 3
		"                                  \n\t"
		"movddup (3+0*4)*8(%%rax), %%xmm0  \n\t" // load xmm0 = alpha30
		"movddup (3+1*4)*8(%%rax), %%xmm1  \n\t" // load xmm1 = alpha31
		"movddup (3+2*4)*8(%%rax), %%xmm2  \n\t" // load xmm2 = alpha32
		"movddup (3+3*4)*8(%%rax), %%xmm3  \n\t" // load xmm3 = (1/alpha33)
		"                                  \n\t"
		"movaps   %%xmm0,  %%xmm4          \n\t" // xmm4 = xmm0
		"movaps   %%xmm1,  %%xmm5          \n\t" // xmm5 = xmm1
		"movaps   %%xmm2,  %%xmm6          \n\t" // xmm6 = xmm2
		"mulpd    %%xmm8,  %%xmm0          \n\t" // xmm0 = alpha30 * ( beta00 beta01 )
		"mulpd    %%xmm12, %%xmm4          \n\t" // xmm4 = alpha30 * ( beta02 beta03 )
		"mulpd    %%xmm9,  %%xmm1          \n\t" // xmm1 = alpha31 * ( beta10 beta11 )
		"mulpd    %%xmm13, %%xmm5          \n\t" // xmm5 = alpha31 * ( beta12 beta13 )
		"mulpd    %%xmm10, %%xmm2          \n\t" // xmm2 = alpha32 * ( beta20 beta21 )
		"mulpd    %%xmm14, %%xmm6          \n\t" // xmm6 = alpha32 * ( beta22 beta23 )
		"addpd    %%xmm1,  %%xmm0          \n\t" // xmm0 += xmm1;
		"addpd    %%xmm5,  %%xmm4          \n\t" // xmm4 += xmm5;
		"addpd    %%xmm2,  %%xmm0          \n\t" // xmm0 += xmm2;
		"addpd    %%xmm6,  %%xmm4          \n\t" // xmm4 += xmm6;
		"subpd    %%xmm0,  %%xmm11         \n\t" // xmm11 -= xmm0
		"subpd    %%xmm4,  %%xmm15         \n\t" // xmm15 -= xmm4
		"mulpd    %%xmm3,  %%xmm11         \n\t" // xmm11 *= (1/alpha33);
		"mulpd    %%xmm3,  %%xmm15         \n\t" // xmm15 *= (1/alpha33);
		"                                  \n\t"
		"movaps   %%xmm11, 6 * 16(%%rbx)   \n\t" // store ( beta30 beta31 ) = xmm11
		"movaps   %%xmm15, 7 * 16(%%rbx)   \n\t" // store ( beta32 beta33 ) = xmm15
		"movlpd   %%xmm11, (%%rcx)         \n\t" // store ( gamma30 ) = xmm11[0]
		"movhpd   %%xmm11, (%%rcx,%%rdi)   \n\t" // store ( gamma31 ) = xmm11[1]
		"movlpd   %%xmm15, (%%rdx)         \n\t" // store ( gamma32 ) = xmm15[0]
		"movhpd   %%xmm15, (%%rdx,%%rdi)   \n\t" // store ( gamma33 ) = xmm15[1]
		"                                  \n\t"
		"                                  \n\t"
		"                                  \n\t"

		: // output operands (none)
		: // input operands
		  "m" (k_iter), // 0
		  "m" (k_left), // 1
		  "m" (a10),    // 2
		  "m" (a11),    // 3
		  "m" (b01),    // 4
		  "m" (b11),    // 5
		  "m" (c11),    // 6
		  "m" (rs_c),   // 7
		  "m" (cs_c),   // 8
		  "m" (alpha),  // 9
		  "m" (b_next)  // 10
		: // register clobber list
		  "rax", "rbx", "rcx", "rdx", "rsi", "rdi", //"r8", "r9", "r10",
		  "xmm0", "xmm1", "xmm2", "xmm3",
		  "xmm4", "xmm5", "xmm6", "xmm7",
		  "xmm8", "xmm9", "xmm10", "xmm11",
		  "xmm12", "xmm13", "xmm14", "xmm15",
		  "memory"
	);

}

void bli_cgemmtrsm_l_opt_4x2(
                              dim_t              k,
                              scomplex* restrict alpha,
                              scomplex* restrict a10,
                              scomplex* restrict a11,
                              scomplex* restrict b01,
                              scomplex* restrict b11,
                              scomplex* restrict c11, inc_t rs_c, inc_t cs_c,
                              auxinfo_t*         data
                            )
{
	/* Just call the reference implementation. */
	BLIS_CGEMMTRSM_L_UKERNEL_REF( k,
	                         alpha,
	                         a10,
	                         a11,
	                         b01,
	                         b11,
	                         c11, rs_c, cs_c,
	                         data );
}

void bli_zgemmtrsm_l_opt_2x2(
                              dim_t              k,
                              dcomplex* restrict alpha,
                              dcomplex* restrict a10,
                              dcomplex* restrict a11,
                              dcomplex* restrict b01,
                              dcomplex* restrict b11,
                              dcomplex* restrict c11, inc_t rs_c, inc_t cs_c,
                              auxinfo_t*         data
                            )
{
	/* Just call the reference implementation. */
	BLIS_ZGEMMTRSM_L_UKERNEL_REF( k,
	                         alpha,
	                         a10,
	                         a11,
	                         b01,
	                         b11,
	                         c11, rs_c, cs_c,
	                         data );
}

