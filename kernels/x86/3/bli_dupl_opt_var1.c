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
	dim_t n_iter = n_elem / 8;
	dim_t n_left = n_elem % 8;

	__asm__ volatile
	(
		"                                \n\t"
		"movl     %2, %%eax              \n\t" // load address of b.
		"movl     %3, %%ebx              \n\t" // load address of bd.
		"                                \n\t"
		"                                \n\t"
		"                                \n\t"
		"movl      %0, %%esi             \n\t" // i = n_iter;
		"testl  %%esi, %%esi             \n\t" // check n_iter via logical AND.
		"je     .CONSIDERNLEFT           \n\t" // if i == 0, jump to code that
		"                                \n\t" // contains the n_left loop.
		"                                \n\t"
		"                                \n\t"
		".LOOPNITER:                     \n\t" // MAIN LOOP
		"                                \n\t"
		"movddup  0 * 8(%%eax), %%xmm0   \n\t"
		"movddup  1 * 8(%%eax), %%xmm1   \n\t"
		"movddup  2 * 8(%%eax), %%xmm2   \n\t"
		"movddup  3 * 8(%%eax), %%xmm3   \n\t"
		"movddup  4 * 8(%%eax), %%xmm4   \n\t"
		"movddup  5 * 8(%%eax), %%xmm5   \n\t"
		"movddup  6 * 8(%%eax), %%xmm6   \n\t"
		"movddup  7 * 8(%%eax), %%xmm7   \n\t"
		"addl     $64, %%eax             \n\t" // b += 8;
		"                                \n\t"
		"movapd   %%xmm0, 0 * 16(%%ebx)  \n\t"
		"movapd   %%xmm1, 1 * 16(%%ebx)  \n\t"
		"movapd   %%xmm2, 2 * 16(%%ebx)  \n\t"
		"movapd   %%xmm3, 3 * 16(%%ebx)  \n\t"
		"movapd   %%xmm4, 4 * 16(%%ebx)  \n\t"
		"movapd   %%xmm5, 5 * 16(%%ebx)  \n\t"
		"movapd   %%xmm6, 6 * 16(%%ebx)  \n\t"
		"movapd   %%xmm7, 7 * 16(%%ebx)  \n\t"
		"addl    $128, %%ebx             \n\t" // bd += 16;
		"                                \n\t"
		"decl   %%esi                    \n\t" // i -= 1;
		"jne    .LOOPNITER               \n\t" // iterate again if i != 0.
		"                                \n\t"
		"                                \n\t"
		"                                \n\t"
		".CONSIDERNLEFT:                 \n\t"
		"                                \n\t"
		"movl      %1, %%esi             \n\t" // i = n_left;
		"testl  %%esi, %%esi             \n\t" // check n_left via logical AND.
		"je     .DONE                    \n\t" // if i == 0, we're done; jump to end.
		"                                \n\t" // else, we prepare to enter n_left loop.
		"                                \n\t"
		"                                \n\t"
		".LOOPNLEFT:                     \n\t" // EDGE LOOP
		"                                \n\t"
		"movddup  0 * 8(%%eax), %%xmm0   \n\t"
		"addl      $8, %%eax             \n\t" // b += 1;
		"                                \n\t"
		"movapd   %%xmm0, 0 * 16(%%ebx)  \n\t"
		"addl     $16, %%ebx             \n\t" // bd += 2;
		"                                \n\t"
		"decl   %%esi                    \n\t" // i -= 1;
		"jne    .LOOPNLEFT               \n\t" // iterate again if i != 0.
		"                                \n\t"
		"                                \n\t"
		"                                \n\t"
		".DONE:                          \n\t"
		"                                \n\t"

		: // output operands (none)
		: // input operands
		  "r" (n_iter),
		  "r" (n_left),
		  "m" (b),
		  "m" (bd)
		: // register clobber list
		  "eax", "ebx", "esi",
		  "xmm0", "xmm1", "xmm2", "xmm3",
		  "xmm4", "xmm5", "xmm6", "xmm7",
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

