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
  "lxv              %%vs0, 0(%%r1)                \n\t" // Load 1_1 col of C
  "lxv              %%vs1, 16(%%r1)               \n\t" // Load 1_2 col of C
  "lxv              %%vs2, 32(%%r1)               \n\t" // Load 1_3 col of C
  "lxv              %%vs3, 48(%%r1)               \n\t" // Load 1_4 col of C
  "lxv              %%vs4, 64(%%r1)               \n\t" // Load 1_5 col of C
  "lxv              %%vs5, 80(%%r1)               \n\t" // Load 1_6 col of C
  "lxv              %%vs6, 96(%%r1)               \n\t" // Load 2_1 col of C
  "lxv              %%vs7, 112(%%r1)              \n\t" // Load 2_2 col of C
  "lxv              %%vs8, 128(%%r1)              \n\t" // Load 2_3 col of C
  "lxv              %%vs9, 144(%%r1)              \n\t" // Load 2_4 col of C
  "lxv              %%vs10, 160(%%r1)             \n\t" // Load 2_5 col of C
  "lxv              %%vs11, 176(%%r1)             \n\t" // Load 2_6 col of C
  "lxv              %%vs12, 192(%%r1)             \n\t" // Load 3_1 col of C
  "lxv              %%vs13, 208(%%r1)             \n\t" // Load 3_2 col of C
  "lxv              %%vs14, 224(%%r1)             \n\t" // Load 3_3 col of C
  "lxv              %%vs15, 240(%%r1)             \n\t" // Load 3_4 col of C
  "lxv              %%vs16, 256(%%r1)             \n\t" // Load 3_5 col of C
  "lxv              %%vs17, 272(%%r1)             \n\t" // Load 3_6 col of C
  "lxv              %%vs18, 288(%%r1)             \n\t" // Load 4_1 col of C
  "lxv              %%vs19, 304(%%r1)             \n\t" // Load 4_2 col of C
  "lxv              %%vs20, 320(%%r1)             \n\t" // Load 4_3 col of C
  "lxv              %%vs21, 336(%%r1)             \n\t" // Load 4_4 col of C
  "lxv              %%vs22, 352(%%r1)             \n\t" // Load 4_5 col of C
  "lxv              %%vs23, 368(%%r1)             \n\t" // Load 4_6 col of C
  "lxv              %%vs24, 384(%%r1)             \n\t" // Load 5_1 col of C
  "lxv              %%vs25, 400(%%r1)             \n\t" // Load 6_2 col of C
  "lxv              %%vs26, 416(%%r1)             \n\t" // Load 5_3 col of C
  "lxv              %%vs27, 432(%%r1)             \n\t" // Load 5_4 col of C
  "lxv              %%vs28, 448(%%r1)             \n\t" // Load 5_5 col of C
  "lxv              %%vs29, 464(%%r1)             \n\t" // Load 5_6 col of C
  "lxv              %%vs30, 480(%%r1)             \n\t" // Load 6_1 col of C
  "lxv              %%vs31, 496(%%r1)             \n\t" // Load 6_2 col of C
  "lxv              %%vs32, 512(%%r1)             \n\t" // Load 6_3 col of C
  "lxv              %%vs33, 528(%%r1)             \n\t" // Load 6_4 col of C
  "lxv              %%vs34, 544(%%r1)             \n\t" // Load 6_5 col of C
  "lxv              %%vs35, 560(%%r1)             \n\t" // Load 6_6 col of C
  "                                               \n\t"
  "ld               %%r9, %0                      \n\t" // Set k_iter to be loop counter
  "mtctr            %%r9                          \n\t"
  "                                               \n\t"
  "K_ITER_LOOP:                                   \n\t" // Begin k_iter loop
  "                                               \n\t"
  "                                               \n\t"
  "                                               \n\t"
  "lxv           %%vs36, 0(%%r2)                  \n\t" // Load a new col of A
  "lxv           %%vs37, 16(%%r2)                 \n\t" // Load a new col of A
  "lxv           %%vs38, 32(%%r2)                 \n\t" // Load a new col of A
  "lxv           %%vs39, 48(%%r2)                 \n\t" // Load a new col of A
  "lxv           %%vs40, 64(%%r2)                 \n\t" // Load a new col of A
  "lxv           %%vs41, 80(%%r2)                 \n\t" // Load a new col of A
  "                                               \n\t"
  "                                               \n\t"
  "                                               \n\t"
  "                                               \n\t"
  "lxvdsx           %%vs48, %%r30, %%r3           \n\t" // Broadcast elem of B
  "xvmaddadp        %%vs0, %%vs36, %%vs48         \n\t" // FMA (C += AB) - 1st
  "xvmaddadp        %%vs1, %%vs37, %%vs48         \n\t" 
  "xvmaddadp        %%vs2, %%vs38, %%vs48         \n\t"
  "xvmaddadp        %%vs3, %%vs39, %%vs48         \n\t"
  "xvmaddadp        %%vs4, %%vs40, %%vs48         \n\t"
  "xvmaddadp        %%vs5, %%vs41, %%vs48         \n\t"
  "                                               \n\t"
  "                                               \n\t"
  "                                               \n\t"
  "addi             %%r30, %%r30, 8               \n\t" // Move B's index
  "                                               \n\t"
  "lxvdsx           %%vs48, %%r30, %%r3           \n\t" // Broadcast elem of B
  "xvmaddadp        %%vs6, %%vs36, %%vs48         \n\t" // FMA (C += AB) - 1st
  "xvmaddadp        %%vs7, %%vs37, %%vs48         \n\t" 
  "xvmaddadp        %%vs8, %%vs38, %%vs48         \n\t"
  "xvmaddadp        %%vs9, %%vs39, %%vs48         \n\t"
  "xvmaddadp        %%vs10, %%vs40, %%vs48        \n\t"
  "xvmaddadp        %%vs11, %%vs41, %%vs48        \n\t"
  "                                               \n\t"
  "                                               \n\t"
  "                                               \n\t"
  "addi             %%r30, %%r30, 8               \n\t" // Move B's index
  "                                               \n\t"
  "lxvdsx           %%vs48, %%r30, %%r3           \n\t" // Broadcast elem of B
  "xvmaddadp        %%vs12, %%vs36, %%vs48        \n\t" // FMA (C += AB) - 1st
  "xvmaddadp        %%vs13, %%vs37, %%vs48        \n\t" 
  "xvmaddadp        %%vs14, %%vs38, %%vs48        \n\t"
  "xvmaddadp        %%vs15, %%vs39, %%vs48        \n\t"
  "xvmaddadp        %%vs16, %%vs40, %%vs48        \n\t"
  "xvmaddadp        %%vs17, %%vs41, %%vs48        \n\t"
  "                                               \n\t"
  "                                               \n\t"
  "                                               \n\t"
  "addi             %%r30, %%r30, 8               \n\t" // Move B's index
  "                                               \n\t"
  "lxvdsx           %%vs48, %%r30, %%r3           \n\t" // Broadcast elem of B
  "xvmaddadp        %%vs18, %%vs36, %%vs48        \n\t" // FMA (C += AB) - 1st
  "xvmaddadp        %%vs19, %%vs37, %%vs48        \n\t" 
  "xvmaddadp        %%vs20, %%vs38, %%vs48        \n\t"
  "xvmaddadp        %%vs21, %%vs39, %%vs48        \n\t"
  "xvmaddadp        %%vs22, %%vs40, %%vs48        \n\t"
  "xvmaddadp        %%vs23, %%vs41, %%vs48        \n\t"
  "                                               \n\t"
  "                                               \n\t"
  "                                               \n\t"
  "addi             %%r30, %%r30, 8               \n\t" // Move B's index
  "                                               \n\t"
  "lxvdsx           %%vs48, %%r30, %%r3           \n\t" // Broadcast elem of B
  "xvmaddadp        %%vs24, %%vs36, %%vs48        \n\t" // FMA (C += AB) - 1st
  "xvmaddadp        %%vs25, %%vs37, %%vs48        \n\t" 
  "xvmaddadp        %%vs26, %%vs38, %%vs48        \n\t"
  "xvmaddadp        %%vs27, %%vs39, %%vs48        \n\t"
  "xvmaddadp        %%vs28, %%vs40, %%vs48        \n\t"
  "xvmaddadp        %%vs29, %%vs41, %%vs48        \n\t"
  "                                               \n\t"
  "                                               \n\t"
  "                                               \n\t"
  "addi             %%r30, %%r30, 8               \n\t" // Move B's index
  "                                               \n\t"
  "lxvdsx           %%vs48, %%r30, %%r3           \n\t" // Broadcast elem of B
  "xvmaddadp        %%vs30, %%vs36, %%vs48        \n\t" // FMA (C += AB) - 1st
  "xvmaddadp        %%vs31, %%vs37, %%vs48        \n\t" 
  "xvmaddadp        %%vs32, %%vs38, %%vs48        \n\t"
  "xvmaddadp        %%vs33, %%vs39, %%vs48        \n\t"
  "xvmaddadp        %%vs34, %%vs40, %%vs48        \n\t"
  "xvmaddadp        %%vs35, %%vs41, %%vs48        \n\t"
  "                                               \n\t"
  "                                               \n\t"
  "                                               \n\t"
  "                                               \n\t"
  "                                               \n\t"
  "                                               \n\t"
  "addi             %%r2, %%r2, 96              \n\t" // Move A's index to new col
  "addi             %%r30, %%r30, 8               \n\t" // Move B's index to new row
  "                                               \n\t"
  "bdnz             K_ITER_LOOP                   \n\t"
  "                                               \n\t"
  "                                               \n\t"
  "stxv              %%vs0, 0(%%r1)                \n\t" // Store C
  "stxv              %%vs1, 16(%%r1)               \n\t" 
  "stxv              %%vs2, 32(%%r1)               \n\t" 
  "stxv              %%vs3, 48(%%r1)               \n\t" 
  "stxv              %%vs4, 64(%%r1)               \n\t" 
  "stxv              %%vs5, 80(%%r1)               \n\t" 
  "stxv              %%vs6, 96(%%r1)               \n\t" 
  "stxv              %%vs7, 112(%%r1)              \n\t" 
  "stxv              %%vs8, 128(%%r1)              \n\t" 
  "stxv              %%vs9, 144(%%r1)              \n\t" 
  "stxv              %%vs10, 160(%%r1)             \n\t" 
  "stxv              %%vs11, 176(%%r1)             \n\t" 
  "stxv              %%vs12, 192(%%r1)             \n\t" 
  "stxv              %%vs13, 208(%%r1)             \n\t" 
  "stxv              %%vs14, 224(%%r1)             \n\t" 
  "stxv              %%vs15, 240(%%r1)             \n\t" 
  "stxv              %%vs16, 256(%%r1)             \n\t" 
  "stxv              %%vs17, 272(%%r1)             \n\t" 
  "stxv              %%vs18, 288(%%r1)             \n\t" 
  "stxv              %%vs19, 304(%%r1)             \n\t" 
  "stxv              %%vs20, 320(%%r1)             \n\t" 
  "stxv              %%vs21, 336(%%r1)             \n\t" 
  "stxv              %%vs22, 352(%%r1)             \n\t" 
  "stxv              %%vs23, 368(%%r1)             \n\t" 
  "stxv              %%vs24, 384(%%r1)             \n\t" 
  "stxv              %%vs25, 400(%%r1)             \n\t" 
  "stxv              %%vs26, 416(%%r1)             \n\t" 
  "stxv              %%vs27, 432(%%r1)             \n\t" 
  "stxv              %%vs28, 448(%%r1)             \n\t" 
  "stxv              %%vs29, 464(%%r1)             \n\t" 
  "stxv              %%vs30, 480(%%r1)             \n\t" 
  "stxv              %%vs31, 496(%%r1)             \n\t" 
  "stxv              %%vs32, 512(%%r1)             \n\t" 
  "stxv              %%vs33, 528(%%r1)             \n\t" 
  "stxv              %%vs34, 544(%%r1)             \n\t" 
  "stxv              %%vs35, 560(%%r1)             \n\t" 
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
  "r1", "r3", "r9", "r10", "r11", "r14", "r15", "r16", "r17", "r20", "r30", 
  "vs36", "vs48", "vs0", "vs1", "vs2", "vs3", "vs4", "vs5"
	);
}