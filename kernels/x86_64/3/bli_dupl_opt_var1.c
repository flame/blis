/*

   BLIS    
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2013, The University of Texas

   Redistribution and use in source and binary forms, with or without
   modification, are permitted provided that the following conditions are
   met:
    - Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.
    - Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.
    - Neither the name of The University of Texas nor the names of its
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
   THEORY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
   (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
   OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

*/

#include "blis.h"

void bli_sdupl_opt_var1(
                         dim_t     n_elem,
                         float*    b,
                         float*    bd
                       )
{
	bli_check_error_code( BLIS_NOT_YET_IMPLEMENTED );
}

void bli_ddupl_opt_var1(
                         dim_t     n_elem,
                         double*   b,
                         double*   bd
                       )
{
	dim_t n_iter = n_elem / 16;
	dim_t n_left = n_elem % 16;

	__asm__ volatile
	(
		"                                  \n\t"
		"movq      %2, %%rax               \n\t" // load address of b.
		"movq      %3, %%rbx               \n\t" // load address of bd.
		"                                  \n\t"
		"addq     $8 * 16, %%rax           \n\t" // increment pointers to allow byte
		"addq     $8 * 16, %%rbx           \n\t" // offsets in the unrolled iterations.
		"                                  \n\t"
		"movq      %0, %%rsi               \n\t" // i = n_iter;
		"testq  %%rsi, %%rsi               \n\t" // check n_iter via logical AND.
		"je     .CONSIDERNLEFT             \n\t" // if i == 0, jump to code that
		"                                  \n\t" // contains the n_left loop.
		"                                  \n\t"
		"                                  \n\t"
		".LOOPNITER:                       \n\t" // MAIN LOOP
		"                                  \n\t"
		"movapd  -8 * 16(%%rax),  %%xmm1   \n\t"
		"movapd  -7 * 16(%%rax),  %%xmm3   \n\t"
		"movapd  -6 * 16(%%rax),  %%xmm5   \n\t"
		"movapd  -5 * 16(%%rax),  %%xmm7   \n\t"
		"                                  \n\t"
		"movapd  -4 * 16(%%rax),  %%xmm9   \n\t"
		"movapd  -3 * 16(%%rax), %%xmm11   \n\t"
		"movapd  -2 * 16(%%rax), %%xmm13   \n\t"
		"movapd  -1 * 16(%%rax), %%xmm15   \n\t"
		"                                  \n\t"
		"                                  \n\t"
		"movddup    %%xmm1,  %%xmm0        \n\t"
		"unpckhpd   %%xmm1,  %%xmm1        \n\t"
		"movddup    %%xmm3,  %%xmm2        \n\t"
		"unpckhpd   %%xmm3,  %%xmm3        \n\t"
		"movddup    %%xmm5,  %%xmm4        \n\t"
		"unpckhpd   %%xmm5,  %%xmm5        \n\t"
		"movddup    %%xmm7,  %%xmm6        \n\t"
		"unpckhpd   %%xmm7,  %%xmm7        \n\t"
		"                                  \n\t"
		"movddup    %%xmm9,  %%xmm8        \n\t"
		"unpckhpd   %%xmm9,  %%xmm9        \n\t"
		"movddup   %%xmm11, %%xmm10        \n\t"
		"unpckhpd  %%xmm11, %%xmm11        \n\t"
		"movddup   %%xmm13, %%xmm12        \n\t"
		"unpckhpd  %%xmm13, %%xmm13        \n\t"
		"movddup   %%xmm15, %%xmm14        \n\t"
		"unpckhpd  %%xmm15, %%xmm15        \n\t"
		"                                  \n\t"
		"                                  \n\t"
		"movapd   %%xmm0, -8 * 16(%%rbx)   \n\t"
		"movapd   %%xmm1, -7 * 16(%%rbx)   \n\t"
		"movapd   %%xmm2, -6 * 16(%%rbx)   \n\t"
		"movapd   %%xmm3, -5 * 16(%%rbx)   \n\t"
		"movapd   %%xmm4, -4 * 16(%%rbx)   \n\t"
		"movapd   %%xmm5, -3 * 16(%%rbx)   \n\t"
		"movapd   %%xmm6, -2 * 16(%%rbx)   \n\t"
		"movapd   %%xmm7, -1 * 16(%%rbx)   \n\t"
		"                                  \n\t"
		"movapd   %%xmm8,  0 * 16(%%rbx)   \n\t"
		"movapd   %%xmm9,  1 * 16(%%rbx)   \n\t"
		"movapd  %%xmm10,  2 * 16(%%rbx)   \n\t"
		"movapd  %%xmm11,  3 * 16(%%rbx)   \n\t"
		"movapd  %%xmm12,  4 * 16(%%rbx)   \n\t"
		"movapd  %%xmm13,  5 * 16(%%rbx)   \n\t"
		"movapd  %%xmm14,  6 * 16(%%rbx)   \n\t"
		"movapd  %%xmm15,  7 * 16(%%rbx)   \n\t"
		"                                  \n\t"
		"addq     $8 * 16, %%rax           \n\t" // b += 16;
		"addq    $16 * 16, %%rbx           \n\t" // bd += 16*2;
		"                                  \n\t"
		"decq   %%rsi                      \n\t" // i -= 1;
		"jne    .LOOPNITER                 \n\t" // iterate again if i != 0.
		"                                  \n\t"
		"                                  \n\t"
		"                                  \n\t"
		".CONSIDERNLEFT:                   \n\t"
		"                                  \n\t"
		"movq      %1, %%rsi               \n\t" // i = n_left;
		"testq  %%rsi, %%rsi               \n\t" // check n_left via logical AND.
		"je     .DONE                      \n\t" // if i == 0, we're done; jump to end.
		"                                  \n\t" // else, we prepare to enter n_left loop.
		"                                  \n\t"
		"                                  \n\t"
		".LOOPNLEFT:                       \n\t" // EDGE LOOP
		"                                  \n\t"
		"movddup  -8 * 16(%%rax), %%xmm0   \n\t"
		"addq      $8, %%rax               \n\t" // b += 1;
		"                                  \n\t"
		"movapd   %%xmm0, -8 * 16(%%rbx)   \n\t"
		"addq     $16, %%rbx               \n\t" // bd += 2;
		"                                  \n\t"
		"decq   %%rsi                      \n\t" // i -= 1;
		"jne    .LOOPNLEFT                 \n\t" // iterate again if i != 0.
		"                                  \n\t"
		"                                  \n\t"
		"                                  \n\t"
		".DONE:                            \n\t"
		"                                  \n\t"

		: // output operands (none)
		: // input operands
		  "r" (n_iter),
		  "r" (n_left),
		  "m" (b),
		  "m" (bd)
		: // register clobber list
		  "rax", "rbx", "rsi",
		  "xmm0", "xmm1", "xmm2", "xmm3",
		  "xmm4", "xmm5", "xmm6", "xmm7",
		  "xmm8", "xmm9", "xmm10", "xmm11",
		  "xmm12", "xmm13", "xmm14", "xmm15",
		  "memory"
	);
	
}

void bli_cdupl_opt_var1(
                         dim_t     k,
                         scomplex* b,
                         scomplex* bd
                       )
{
	bli_check_error_code( BLIS_NOT_YET_IMPLEMENTED );
}

void bli_zdupl_opt_var1(
                         dim_t     k,
                         dcomplex* b,
                         dcomplex* bd
                       )
{
	bli_check_error_code( BLIS_NOT_YET_IMPLEMENTED );
}

