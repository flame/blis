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
	"ld               %%r1, %6                      \n\t" // load ptr of C
  "ld               %%r2, %2                      \n\t" // load ptr of A
  "ld               %%r3, %3                      \n\t" // load ptr of B
  "                                               \n\t" 
  "                                               \n\t" // Create indices
  "li               %%r10,0                       \n\t" // for C (1st col)
  "li               %%r11,16                      \n\t" // for C (2nd col)
  "                                               \n\t"
  "li               %%r20,0                       \n\t" // for A 
  "li               %%r30,0                       \n\t" // for B
  "                                               \n\t"
  "lxv           %%vs0, 0(%%r1)            \n\t" // Load 1st col of C
  "lxv           %%vs1, 16(%%r1)            \n\t" // Load 2nd col of C
  "                                               \n\t"
  "ld               %%r9, %0                      \n\t" // Set k_iter to be loop counter
  "mtctr            %%r9                          \n\t"
  "                                               \n\t"
  "K_ITER_LOOP:                                   \n\t" // Begin k_iter loop
  "                                               \n\t"
  "lxv              %%vs20, 0(%%r2)           \n\t" // Load a new col of A
  "                                               \n\t"
  "lxvdsx           %%vs30, %%r30, %%r3           \n\t" // Broadcast elem of B
  "xvmaddadp        %%vs0, %%vs20, %%vs30         \n\t" // FMA (C += AB) - 1st 
  "                                               \n\t"
  "addi             %%r30, %%r30, 8               \n\t" // Move B's index
  "                                               \n\t"
  "lxvdsx           %%vs30, %%r30, %%r3           \n\t" // Broadcast elem of B
  "xvmaddadp        %%vs1, %%vs20, %%vs30         \n\t" // FMA (C += AB) - 2nd
  "                                               \n\t"
  "addi             %%r2, %%r2, 16              \n\t" // Move A's index to new col
  "addi             %%r30, %%r30, 8               \n\t" // Move B's index to new row
  "                                               \n\t"
  "bdnz             K_ITER_LOOP                   \n\t"
  "                                               \n\t"
  "                                               \n\t"
  "stxv          %%vs0, 0(%%r1)            \n\t" // Store updated C in memory
  "stxv          %%vs1, 16(%%r1)            \n\t"
  "                                               \n\t"
  

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
  /* unclobberable regs: r2(PIC reg), */
  "r1", "r3", "r9", "r10", "r11", "r12", "r13", "r20", "r30", "vs20", "vs30", "vs0", "vs1"
	);
}