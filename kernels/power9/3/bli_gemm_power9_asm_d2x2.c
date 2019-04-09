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

void bli_dgemm_power9_asm_4x6
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
  "                                               \n\t"
  "li               %%r30,0                       \n\t" // for B
  "                                               \n\t"
  "lxv           %%vs0, 0(%%r1)                   \n\t" // Load 1_1 col of C
  "lxv           %%vs1, 16(%%r1)                  \n\t" // Load 1_2 col of C
  "lxv           %%vs2, 32(%%r1)                  \n\t" // Load 2_1 col of C
  "lxv           %%vs3, 48(%%r1)                  \n\t" // Load 2_2 col of C
  "lxv           %%vs4, 64(%%r1)                  \n\t" // Load 3_1 col of C
  "lxv           %%vs5, 80(%%r1)                  \n\t" // Load 3_2 col of C
  "lxv           %%vs6, 96(%%r1)                  \n\t" // Load 4_1 col of C
  "lxv           %%vs7, 112(%%r1)                 \n\t" // Load 4_2 col of C
  "lxv           %%vs8, 128(%%r1)                 \n\t" // Load 5_1 col of C
  "lxv           %%vs9, 144(%%r1)                 \n\t" // Load 5_2 col of C
  "lxv           %%vs10, 160(%%r1)                \n\t" // Load 6_1 col of C
  "lxv           %%vs11, 176(%%r1)                \n\t" // Load 6_2 col of C
  "                                               \n\t"
  "ld               %%r9, %0                      \n\t" // Set k_iter to be loop counter
  "mtctr            %%r9                          \n\t"
  "                                               \n\t"
  "K_ITER_LOOP:                                   \n\t" // Begin k_iter loop
  "                                               \n\t"
  "lxv           %%vs36, 0(%%r2)                  \n\t" // Load a new col of A
  "lxv           %%vs37, 16(%%r2)                 \n\t" // Load a new col of A
  "                                               \n\t"
  "                                               \n\t"
  "                                               \n\t"
  "                                               \n\t"
  "lxvdsx           %%vs48, %%r30, %%r3           \n\t" // Broadcast elem of B
  "xvmaddadp        %%vs0, %%vs36, %%vs48         \n\t" // FMA (C += AB) - 1st 
  "xvmaddadp        %%vs1, %%vs37, %%vs48         \n\t" // FMA (C += AB) - 2nd
  "                                               \n\t"
  "                                               \n\t"
  "                                               \n\t"
  "addi             %%r30, %%r30, 8               \n\t" // Move B's index
  "                                               \n\t"
  "lxvdsx           %%vs48, %%r30, %%r3           \n\t" // Broadcast elem of B
  "xvmaddadp        %%vs2, %%vs36, %%vs48         \n\t" // FMA (C += AB) - 2nd
  "xvmaddadp        %%vs3, %%vs37, %%vs48         \n\t" // FMA (C += AB) - 2nd
  "                                               \n\t"
  "                                               \n\t"
  "                                               \n\t"
  "addi             %%r30, %%r30, 8               \n\t" // Move B's index
  "                                               \n\t"
  "lxvdsx           %%vs48, %%r30, %%r3           \n\t" // Broadcast elem of B
  "xvmaddadp        %%vs4, %%vs36, %%vs48         \n\t" // FMA (C += AB) - 2nd
  "xvmaddadp        %%vs5, %%vs37, %%vs48         \n\t" // FMA (C += AB) - 2nd
  "                                               \n\t"
  "                                               \n\t"
  "                                               \n\t"
  "addi             %%r30, %%r30, 8               \n\t" // Move B's index
  "                                               \n\t"
  "lxvdsx           %%vs48, %%r30, %%r3           \n\t" // Broadcast elem of B
  "xvmaddadp        %%vs6, %%vs36, %%vs48         \n\t" // FMA (C += AB) - 2nd
  "xvmaddadp        %%vs7, %%vs37, %%vs48         \n\t" // FMA (C += AB) - 2nd
  "                                               \n\t"
  "                                               \n\t"
  "                                               \n\t"
  "addi             %%r30, %%r30, 8               \n\t" // Move B's index
  "                                               \n\t"
  "lxvdsx           %%vs48, %%r30, %%r3           \n\t" // Broadcast elem of B
  "xvmaddadp        %%vs8, %%vs36, %%vs48         \n\t" // FMA (C += AB) - 2nd
  "xvmaddadp        %%vs9, %%vs37, %%vs48         \n\t" // FMA (C += AB) - 2nd
  "                                               \n\t"
  "                                               \n\t"
  "                                               \n\t"
  "addi             %%r30, %%r30, 8               \n\t" // Move B's index
  "                                               \n\t"
  "lxvdsx           %%vs48, %%r30, %%r3           \n\t" // Broadcast elem of B
  "xvmaddadp        %%vs10, %%vs36, %%vs48         \n\t" // FMA (C += AB) - 2nd
  "xvmaddadp        %%vs11, %%vs37, %%vs48         \n\t" // FMA (C += AB) - 2nd
  "                                               \n\t"
  "                                               \n\t"
  "                                               \n\t"
  "                                               \n\t"
  "                                               \n\t"
  "                                               \n\t"
  "addi             %%r2, %%r2, 32              \n\t" // Move A's index to new col
  "addi             %%r30, %%r30, 8               \n\t" // Move B's index to new row
  "                                               \n\t"
  "bdnz             K_ITER_LOOP                   \n\t"
  "                                               \n\t"
  "                                               \n\t"
  "stxv           %%vs0, 0(%%r1)                   \n\t" // Store 1_1 col of C
  "stxv           %%vs1, 16(%%r1)                  \n\t" // Store 1_2 col of C
  "stxv           %%vs2, 32(%%r1)                  \n\t" // Store 2_1 col of C
  "stxv           %%vs3, 48(%%r1)                  \n\t" // Store 2_2 col of C
  "stxv           %%vs4, 64(%%r1)                  \n\t" // Store 3_1 col of C
  "stxv           %%vs5, 80(%%r1)                  \n\t" // Store 3_2 col of C
  "stxv           %%vs6, 96(%%r1)                  \n\t" // Store 4_1 col of C
  "stxv           %%vs7, 112(%%r1)                 \n\t" // Store 4_2 col of C
  "stxv           %%vs8, 128(%%r1)                 \n\t" // Store 5_1 col of C
  "stxv           %%vs9, 144(%%r1)                 \n\t" // Store 5_2 col of C
  "stxv           %%vs10, 160(%%r1)                \n\t" // Store 6_1 col of C
  "stxv           %%vs11, 176(%%r1)                \n\t" // Store 6_2 col of C
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
  "r1", "r3", "r9", "r30", 
  "vs0", "vs1", "vs3", "vs4", "vs5", "vs6", "vs7", "vs8", "vs9", "vs10", "vs11",
  "vs36", "vs37", "vs48"
	);
}