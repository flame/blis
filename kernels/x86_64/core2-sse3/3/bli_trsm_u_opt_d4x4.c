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

void bli_strsm_u_opt_8x4(
                          float* restrict    a11,
                          float* restrict    b11,
                          float* restrict    c11, inc_t rs_c, inc_t cs_c,
                          auxinfo_t*         data
                        )
{
	/* Just call the reference implementation. */
	BLIS_STRSM_U_UKERNEL_REF( a11,
	                     b11,
	                     c11, rs_c, cs_c,
	                     data );
}

void bli_dtrsm_u_opt_4x4(
                          double* restrict   a11,
                          double* restrict   b11,
                          double* restrict   c11, inc_t rs_c, inc_t cs_c,
                          auxinfo_t*         data
                        )
{
	__asm__ volatile
	(
		"                                  \n\t"
		"movq      %1, %%rbx               \n\t" // load address of b11.
		"                                  \n\t"
		"movaps  0 * 16(%%rbx), %%xmm8     \n\t" // xmm8  = ( beta00 beta01 )
		"movaps  1 * 16(%%rbx), %%xmm12    \n\t" // xmm9  = ( beta02 beta03 )
		"movaps  2 * 16(%%rbx), %%xmm9     \n\t" // xmm10 = ( beta10 beta11 )
		"movaps  3 * 16(%%rbx), %%xmm13    \n\t" // xmm11 = ( beta12 beta13 )
		"movaps  4 * 16(%%rbx), %%xmm10    \n\t" // xmm12 = ( beta20 beta21 )
		"movaps  5 * 16(%%rbx), %%xmm14    \n\t" // xmm13 = ( beta22 beta23 )
		"movaps  6 * 16(%%rbx), %%xmm11    \n\t" // xmm14 = ( beta30 beta31 )
		"movaps  7 * 16(%%rbx), %%xmm15    \n\t" // xmm15 = ( beta32 beta33 )
		"                                  \n\t"
		"                                  \n\t"
		"                                  \n\t"
		"movq     %0, %%rax                \n\t" // load address of a11
		"movq     %2, %%rcx                \n\t" // load address of c11
		"                                  \n\t"
		"movq     %3, %%rsi                \n\t" // load rs_c
		"movq     %4, %%rdi                \n\t" // load cs_c
		"salq     $3, %%rsi                \n\t" // rs_c *= sizeof( double )
		"salq     $3, %%rdi                \n\t" // cs_c *= sizeof( double )
		"                                  \n\t"
		"addq  %%rsi, %%rcx                \n\t" // c11 += (4-1)*rs_c
		"addq  %%rsi, %%rcx                \n\t"
		"addq  %%rsi, %%rcx                \n\t"
		"leaq   (%%rcx,%%rdi,2), %%rdx     \n\t" // c11_2 = c11 + 2*cs_c;
		"                                  \n\t"
		"                                  \n\t"
		"                                  \n\t"
		"                                  \n\t" // iteration 0
		"                                  \n\t"
		"movddup (3+3*4)*8(%%rax), %%xmm3  \n\t" // load xmm3 = (1/alpha33)
		"                                  \n\t"
		"mulpd    %%xmm3, %%xmm11          \n\t" // xmm11 *= (1/alpha33);
		"mulpd    %%xmm3, %%xmm15          \n\t" // xmm15 *= (1/alpha33);
		"                                  \n\t"
		"movaps   %%xmm11, 6 * 16(%%rbx)   \n\t" // store ( beta30 beta31 ) = xmm11
		"movaps   %%xmm15, 7 * 16(%%rbx)   \n\t" // store ( beta32 beta33 ) = xmm15
		"movlpd   %%xmm11, (%%rcx)         \n\t" // store ( gamma30 ) = xmm11[0]
		"movhpd   %%xmm11, (%%rcx,%%rdi)   \n\t" // store ( gamma31 ) = xmm11[1]
		"movlpd   %%xmm15, (%%rdx)         \n\t" // store ( gamma32 ) = xmm15[0]
		"movhpd   %%xmm15, (%%rdx,%%rdi)   \n\t" // store ( gamma33 ) = xmm15[1]
		"subq     %%rsi, %%rcx             \n\t" // c11   -= rs_c
		"subq     %%rsi, %%rdx             \n\t" // c11_2 -= rs_c
		"                                  \n\t"
		"                                  \n\t"
		"                                  \n\t"
		"                                  \n\t" // iteration 1
		"                                  \n\t"
		"movddup (2+2*4)*8(%%rax), %%xmm2  \n\t" // load xmm2 = (1/alpha22)
		"movddup (2+3*4)*8(%%rax), %%xmm3  \n\t" // load xmm3 = alpha23
		"                                  \n\t"
		"movaps   %%xmm3,  %%xmm7          \n\t" // xmm7 = xmm3
		"mulpd    %%xmm11, %%xmm3          \n\t" // xmm3 = alpha23 * ( beta30 beta31 )
		"mulpd    %%xmm15, %%xmm7          \n\t" // xmm7 = alpha23 * ( beta32 beta33 )
		"subpd    %%xmm3,  %%xmm10         \n\t" // xmm10 -= xmm3
		"subpd    %%xmm7,  %%xmm14         \n\t" // xmm14 -= xmm7
		"mulpd    %%xmm2,  %%xmm10         \n\t" // xmm10 *= (1/alpha22);
		"mulpd    %%xmm2,  %%xmm14         \n\t" // xmm14 *= (1/alpha22);
		"                                  \n\t"
		"movaps   %%xmm10, 4 * 16(%%rbx)   \n\t" // store ( beta20 beta21 ) = xmm10
		"movaps   %%xmm14, 5 * 16(%%rbx)   \n\t" // store ( beta22 beta23 ) = xmm14
		"movlpd   %%xmm10, (%%rcx)         \n\t" // store ( gamma20 ) = xmm10[0]
		"movhpd   %%xmm10, (%%rcx,%%rdi)   \n\t" // store ( gamma21 ) = xmm10[1]
		"movlpd   %%xmm14, (%%rdx)         \n\t" // store ( gamma22 ) = xmm14[0]
		"movhpd   %%xmm14, (%%rdx,%%rdi)   \n\t" // store ( gamma23 ) = xmm14[1]
		"subq     %%rsi, %%rcx             \n\t" // c11   -= rs_c
		"subq     %%rsi, %%rdx             \n\t" // c11_2 -= rs_c
		"                                  \n\t"
		"                                  \n\t"
		"                                  \n\t"
		"                                  \n\t" // iteration 2
		"                                  \n\t"
		"movddup (1+1*4)*8(%%rax), %%xmm1  \n\t" // load xmm1 = (1/alpha11)
		"movddup (1+2*4)*8(%%rax), %%xmm2  \n\t" // load xmm2 = alpha12
		"movddup (1+3*4)*8(%%rax), %%xmm3  \n\t" // load xmm3 = alpha13
		"                                  \n\t"
		"movaps   %%xmm2,  %%xmm6          \n\t" // xmm6 = xmm2
		"movaps   %%xmm3,  %%xmm7          \n\t" // xmm7 = xmm3
		"mulpd    %%xmm10, %%xmm2          \n\t" // xmm2 = alpha12 * ( beta20 beta21 )
		"mulpd    %%xmm14, %%xmm6          \n\t" // xmm6 = alpha12 * ( beta22 beta23 )
		"mulpd    %%xmm11, %%xmm3          \n\t" // xmm3 = alpha13 * ( beta30 beta31 )
		"mulpd    %%xmm15, %%xmm7          \n\t" // xmm7 = alpha13 * ( beta32 beta33 )
		"addpd    %%xmm3,  %%xmm2          \n\t" // xmm2 += xmm3;
		"addpd    %%xmm7,  %%xmm6          \n\t" // xmm6 += xmm7;
		"subpd    %%xmm2,  %%xmm9          \n\t" // xmm9  -= xmm2
		"subpd    %%xmm6,  %%xmm13         \n\t" // xmm13 -= xmm6
		"mulpd    %%xmm1,  %%xmm9          \n\t" // xmm9  *= (1/alpha11);
		"mulpd    %%xmm1,  %%xmm13         \n\t" // xmm13 *= (1/alpha11);
		"                                  \n\t"
		"movaps   %%xmm9,  2 * 16(%%rbx)   \n\t" // store ( beta10 beta11 ) = xmm9
		"movaps   %%xmm13, 3 * 16(%%rbx)   \n\t" // store ( beta12 beta13 ) = xmm13
		"movlpd   %%xmm9,  (%%rcx)         \n\t" // store ( gamma10 ) = xmm9[0]
		"movhpd   %%xmm9,  (%%rcx,%%rdi)   \n\t" // store ( gamma11 ) = xmm9[1]
		"movlpd   %%xmm13, (%%rdx)         \n\t" // store ( gamma12 ) = xmm13[0]
		"movhpd   %%xmm13, (%%rdx,%%rdi)   \n\t" // store ( gamma13 ) = xmm13[1]
		"subq     %%rsi, %%rcx             \n\t" // c11   -= rs_c
		"subq     %%rsi, %%rdx             \n\t" // c11_2 -= rs_c
		"                                  \n\t"
		"                                  \n\t"
		"                                  \n\t"
		"                                  \n\t" // iteration 3
		"                                  \n\t"
		"movddup (0+0*4)*8(%%rax), %%xmm0  \n\t" // load xmm0 = (1/alpha00)
		"movddup (0+1*4)*8(%%rax), %%xmm1  \n\t" // load xmm1 = alpha01
		"movddup (0+2*4)*8(%%rax), %%xmm2  \n\t" // load xmm2 = alpha02
		"movddup (0+3*4)*8(%%rax), %%xmm3  \n\t" // load xmm3 = alpha03
		"                                  \n\t"
		"movaps   %%xmm1,  %%xmm5          \n\t" // xmm5 = xmm1
		"movaps   %%xmm2,  %%xmm6          \n\t" // xmm6 = xmm2
		"movaps   %%xmm3,  %%xmm7          \n\t" // xmm7 = xmm3
		"mulpd    %%xmm9,  %%xmm1          \n\t" // xmm1 = alpha01 * ( beta10 beta11 )
		"mulpd    %%xmm13, %%xmm5          \n\t" // xmm5 = alpha01 * ( beta12 beta13 )
		"mulpd    %%xmm10, %%xmm2          \n\t" // xmm2 = alpha02 * ( beta20 beta21 )
		"mulpd    %%xmm14, %%xmm6          \n\t" // xmm6 = alpha02 * ( beta22 beta23 )
		"mulpd    %%xmm11, %%xmm3          \n\t" // xmm3 = alpha03 * ( beta30 beta31 )
		"mulpd    %%xmm15, %%xmm7          \n\t" // xmm7 = alpha03 * ( beta32 beta33 )
		"addpd    %%xmm2,  %%xmm1          \n\t" // xmm1 += xmm2;
		"addpd    %%xmm6,  %%xmm5          \n\t" // xmm5 += xmm6;
		"addpd    %%xmm3,  %%xmm1          \n\t" // xmm1 += xmm3;
		"addpd    %%xmm7,  %%xmm5          \n\t" // xmm5 += xmm7;
		"subpd    %%xmm1,  %%xmm8          \n\t" // xmm8  -= xmm1
		"subpd    %%xmm5,  %%xmm12         \n\t" // xmm12 -= xmm5
		"mulpd    %%xmm0,  %%xmm8          \n\t" // xmm8  *= (1/alpha00);
		"mulpd    %%xmm0,  %%xmm12         \n\t" // xmm12 *= (1/alpha00);
		"                                  \n\t"
		"movaps   %%xmm8,  0 * 16(%%rbx)   \n\t" // store ( beta00 beta01 ) = xmm8
		"movaps   %%xmm12, 1 * 16(%%rbx)   \n\t" // store ( beta02 beta03 ) = xmm12
		"movlpd   %%xmm8,  (%%rcx)         \n\t" // store ( gamma00 ) = xmm8[0]
		"movhpd   %%xmm8,  (%%rcx,%%rdi)   \n\t" // store ( gamma01 ) = xmm8[1]
		"movlpd   %%xmm12, (%%rdx)         \n\t" // store ( gamma02 ) = xmm12[0]
		"movhpd   %%xmm12, (%%rdx,%%rdi)   \n\t" // store ( gamma03 ) = xmm12[1]
		"                                  \n\t"
		"                                  \n\t"
		"                                  \n\t"

		: // output operands (none)
		: // input operands
		  "m" (a11),    // 0
		  "m" (b11),    // 1
		  "m" (c11),    // 2
		  "m" (rs_c),   // 3
		  "m" (cs_c)    // 4
		: // register clobber list
		  "rax", "rbx", "rcx", "rdx", "rsi", "rdi",
		  "xmm0", "xmm1", "xmm2", "xmm3",
		  "xmm4", "xmm5", "xmm6", "xmm7",
		  "xmm8", "xmm9", "xmm10", "xmm11",
		  "xmm12", "xmm13", "xmm14", "xmm15",
		  "memory"
	);

}

void bli_ctrsm_u_opt_4x2(
                          scomplex* restrict a11,
                          scomplex* restrict b11,
                          scomplex* restrict c11, inc_t rs_c, inc_t cs_c,
                          auxinfo_t*         data
                        )
{
	/* Just call the reference implementation. */
	BLIS_CTRSM_U_UKERNEL_REF( a11,
	                     b11,
	                     c11, rs_c, cs_c,
	                     data );
}

void bli_ztrsm_u_opt_2x2(
                          dcomplex* restrict a11,
                          dcomplex* restrict b11,
                          dcomplex* restrict c11, inc_t rs_c, inc_t cs_c,
                          auxinfo_t*         data
                        )
{
	/* Just call the reference implementation. */
	BLIS_ZTRSM_U_UKERNEL_REF( a11,
	                     b11,
	                     c11, rs_c, cs_c,
	                     data );
}

