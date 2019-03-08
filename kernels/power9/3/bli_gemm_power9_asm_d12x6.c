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

void bli_dgemm_power9_asm_12x6
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
	//void*   a_next = bli_auxinfo_next_a( data );
	//void*   b_next = bli_auxinfo_next_b( data );

	// Typecast local copies of integers in case dim_t and inc_t are a
	// different size than is expected by load instructions.
	uint64_t k_iter = k0 / 4;
	uint64_t k_left = k0 % 4;
	uint64_t rs_c   = rs_c0;
	uint64_t cs_c   = cs_c0;

	__asm__ volatile
	(
	"                                               \n\t"
	"ld                %%r26, %6                    \n\t"
  "li                %%r27, 1                     \n\t"
  "lxv               %%v0,  0(%%r26)              \n\t" 
  "std               %%r27, 4(%%r26)              \n\t"
	// "                                            \n\t"
  // "                                            \n\t"
  // "ld                r26, %6                   \n\t" // load C
  // "                                            \n\t"
  // "ld                r28, %2                   \n\t" // load A
	// "ld                r27, %3                   \n\t" // load B
  // "                                            \n\t"
  // "                                            \n\t" 
  // "lxv               0, 0(r26)                 \n\t" // load elems of C 
  // "lxv               1, 1(r26)                 \n\t"
  // "                                            \n\t"
  // "                                            \n\t"
  // "                                            \n\t"
  // "                                            \n\t"
  // "                                            \n\t"
  // "li                r4, %1                    \n\t" // load k_iter
  // "mtctr             r4                        \n\t"
  // "loop:                                       \n\t"
  // "                                            \n\t"
  // "lxv               36, 0(r28)                \n\t" // load col of a
  // "lxv               37, 1(r28)                \n\t" // load col of a
  // "                                            \n\t"
  // "lxvdsx            47, 0, r27                \n\t" // broadcast b
  // "addi              r27,r27,8                 \n\t" // inc b
  // "                                            \n\t"
  // "xvmaddadp         0,36,47                   \n\t"
  // "xvmaddadp         1,37,47                   \n\t"
  // "                                            \n\t"
  // "lxvdsx            47, 0, r27                \n\t" // broadcast b
  // "addi              r27,r27,8                 \n\t" // inc b
  // "                                            \n\t"
  // "xvmaddadp         0,36,47                   \n\t"
  // "xvmaddadp         1,37,47                   \n\t"
  // "                                            \n\t"
  // "                                            \n\t"
  // "addi              r28,r28,32                \n\t" // move A forward
  // "bdnz loop                                   \n\t"
  // "                                            \n\t"
  // "                                            \n\t"
  // "                                            \n\t"
  // "                                            \n\t"
  // "                                            \n\t"
  // "                                            \n\t"
  // "li                r5,16                     \n\t" // use as offset
  // "                                            \n\t"
  // "stxvd2x           0,0,r26                   \n\t" // store c back into memory 
  // "stxvd2x           1,r5,r26                  \n\t"
  // "                                            \n\t"
  // "                                            \n\t"
  


	: // output operands (none)
	: // input operands
	  "m" (k_iter), // 0
	  "m" (k_left), // 1
	  "m" (a),      // 2
	  "m" (b),      // 3
	  "m" (alpha),  // 4
	  "m" (beta),   // 5
	  "m" (c),      // 6
	  "m" (rs_c),   // 7
	  "m" (cs_c)/*,   // 8
	  "m" (b_next), // 9
	  "m" (a_next)*/  // 10
	: // register clobber list
	/*Unclobberable registers: r1 (stk ptr), r2(toc ptr), r11(env ptr),
	r13(64b mode thread local data ptr, r30(stk frm ptr), r31(stk frm ptr) */
  "r4", "r5", "r26", "r27", "r28", "0", "1", "36", "37", "47"
	);
}



