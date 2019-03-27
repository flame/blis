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

void bli_dgemm_power9_asm_2x2
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
	uint64_t k_iter = k0;
	uint64_t k_left = k0 % 4;
	uint64_t rs_c   = rs_c0;
	uint64_t cs_c   = cs_c0;

	__asm__ volatile
	(
	"                                               \n\t"
	"ld               %%r26, %6                     \n\t" // c
  "ld               %%r27, %2                     \n\t" // a
  "ld               %%r28, %3                     \n\t" // b
  "                                               \n\t" 
  "                                               \n\t" // indices
  "li               %%r10,0                       \n\t" // for b
  "li               %%r14,32                      \n\t" // for c
  "li               %%r15,0                       \n\t" // for c
  "li               %%r16,32                       \n\t" // for a
  "lxvd2x           %%vs3, %%r15, %%r26           \n\t" // load col of c
  // "lxvd2x           %%vs4, %%r14, %%r26           \n\t" // load col of c
  "                                               \n\t"
  // "ld               %%r30, %0                     \n\t" // k_iter
  // "mtctr            %%r30                         \n\t"
  // "loop:                                          \n\t"
  "                                               \n\t"
  "lxvd2x           %%vs0, %%r16, %%r27           \n\t" // load col of a
  // "                                                \n\t"
  // "                                                \n\t"
  "lxvdsx           %%vs1, %%r10, %%r28           \n\t" // splat an elem of b
  "xvmaddadp        %%vs3, %%vs0, %%vs1           \n\t" // mult a * b 
  // "                                                \n\t"
  // "addi             %%r10, %%r10, 8               \n\t" // go to next elem of b
  // // "                                                \n\t"
  // "lxvdsx           %%vs1, %%r10, %%r28           \n\t" // splat new elem
  // "xvmaddadp        %%vs4, %%vs0, %%vs1           \n\t" // a * b
  // // "                                                \n\t"
  // "addi             %%r16, %%r16, 32              \n\t" // move a to next col
  // "addi             %%r10, %%r10, 24              \n\t" // move b to next row
  // // "                                                \n\t"
  // "bdnz             loop                          \n\t"
  "stxvd2x          %%vs3, %%r14, %%r26           \n\t" 
  // "stxvd2x          %%vs3, %%r14, %%r26           \n\t" 
  // "                                                \n\t"
  

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
  "r26", "r27", "r28", "r10", "r14", "r15", "r16", "r30", "vs0", "vs1", "vs3"
	);
}



