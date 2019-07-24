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
#include "bli_pwr9_asm_macros_12x6.h"

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

	#if 1
	uint64_t k_iter = k0 / 2;
	uint64_t k_left = k0 % 2;
	#else
	uint64_t k_iter = 0;
	uint64_t k_left = k0;
	#endif

  	uint64_t rs_c   = rs_c0;
	uint64_t cs_c   = cs_c0;


	__asm__ volatile
	(
	"                                               \n\t"
  	"                                               \n\t"
  	"                                               \n\t" // Offsets for B
  	"li               %%r22,0                       \n\t" // 0
  	"li               %%r23,8                       \n\t" // 1
  	"li               %%r24,16                      \n\t" // 2
  	"li               %%r25,24                      \n\t" // 3
  	"li               %%r26,32                      \n\t" // 4
  	"li               %%r27,40                      \n\t" // 5
  	"                                               \n\t"
  	"ld               %%r17, %0                     \n\t" // load k_iter
  	"ld               %%r18, %1                     \n\t" // load k_left
  	"                                               \n\t"
	"ld               %%r10, %8                     \n\t" // load cs_c
  	"ld               %%r9, %7                      \n\t" // load rs_c
  	"                                               \n\t"
  	"slwi             %%r10, %%r10, 3               \n\t" // mul by size of elem
  	"slwi             %%r9, %%r9, 3                 \n\t" // mul by size of elem
  	"                                               \n\t"
  	"ld               %%r7, %2                      \n\t" // load ptr of A
  	"ld               %%r8, %3                      \n\t" // load ptr of B
  	"ld               %%r16, %6                     \n\t" // load ptr of C
  	"                                               \n\t"
	"                                               \n\t"
	"                                               \n\t"
	"                                               \n\t"
	"                                               \n\t"
	"                                               \n\t"
	"                                               \n\t"
	"                                               \n\t"
	"                                               \n\t"
	"                                               \n\t"
	"                                               \n\t"
	ZERO_OUT_VREG                                             // Zero out vec regs
	"                                               \n\t"
	"                                               \n\t"
	"                                               \n\t"
	PRELOAD_A_B
	"                                               \n\t"
	"addi             %%r28, %%r8, 48               \n\t"
	"addi             %%r8, %%r8, 96                \n\t"
	"addi             %%r7, %%r7, 96                \n\t"
	"                                               \n\t"
	"                                               \n\t"
	"                                               \n\t"
	"                                               \n\t"
	"                                               \n\t"
  	"                                               \n\t"
  	"                                               \n\t"
  	"                                               \n\t"
  	"cmpwi            %%r0, %%r17, 0                \n\t"
  	"beq              %%r0, DPRELOOPKLEFT           \n\t"
  	"mtctr            %%r17                         \n\t"
  	"                                               \n\t"
	"                                               \n\t"
	"                                               \n\t"
	"                                               \n\t"
	"                                               \n\t"
	"                                               \n\t"
	"                                               \n\t"
	"                                               \n\t"
	"                                               \n\t"
	"                                               \n\t"
	"                                               \n\t"
	"                                               \n\t"
	"                                               \n\t"
	"                                               \n\t"
	"                                               \n\t"
  	"                                               \n\t" // k_iter loop does A*B 
  	"DLOOPKITER:                                    \n\t" // Begin k_iter loop
  	"                                               \n\t"
	LOAD_UPDATE_32
  	"                                               \n\t"
  	"bdnz             DLOOPKITER                    \n\t"
    "                                               \n\t"
  	"                                               \n\t"
	"                                               \n\t"
	"                                               \n\t"
	"                                               \n\t"
	"                                               \n\t"
	"                                               \n\t"
	"                                               \n\t"
	"                                               \n\t"
	"                                               \n\t"
	"                                               \n\t"
	"                                               \n\t"
	"                                               \n\t"
	"                                               \n\t"
	"                                               \n\t"
	"                                               \n\t"
	"                                               \n\t"
	"                                               \n\t"
  	"DPRELOOPKLEFT:                                 \n\t"
  	"                                               \n\t"
  	"cmpwi            %%r0, %%r18, 0                \n\t"
  	"beq              %%r0, DPOSTACCUM              \n\t"
  	"mtctr            %%r18                         \n\t"
  	"                                               \n\t"
  	"DLOOPKLEFT:                                    \n\t" // EDGE LOOP
	"                                               \n\t"
  LOAD_UPDATE_1
  	"                                               \n\t"
  	"bdnz             DLOOPKLEFT                    \n\t"
  	"                                               \n\t"
	"                                               \n\t"
	"                                               \n\t"
	"                                               \n\t"
	"                                               \n\t"
	"                                               \n\t"
	"                                               \n\t"
	"                                               \n\t"
	"                                               \n\t"
	"                                               \n\t"
  	"DPOSTACCUM:                                    \n\t"
  	"                                               \n\t"
  	"ld               %%r0, %4                      \n\t" // load ptr for alpha
  	"ld               %%r28, %5                     \n\t" // load ptr for beta
	"ld               %%r26, 0(%%r28)               \n\t" // load val of beta
  	"                                               \n\t"
  	"lxvdsx           %%vs48, 0, %%r0               \n\t" // splat alpha
  	"lxvdsx           %%vs59, 0, %%r28              \n\t" // splat beta
  	"                                               \n\t"
  SCALE_ALPHA
  	"                                               \n\t"
  	"                                               \n\t"
  	"                                               \n\t"
  	"cmpdi            %%r0, %%r26, 0                \n\t"
  	"beq              %%r0, DBETAZERO               \n\t" // jump to BZ case if beta = 0
  	"                                               \n\t"
  	"ld               %%r22, %6                     \n\t" // load ptr for C (used as offset)
  	"                                               \n\t"
  	"cmpwi            %%r0, %%r9, 8                 \n\t"
  	"beq              DCOLSTOREDBNZ                 \n\t" // jump to COLstore case, if rs_c = 8
  	"                                               \n\t"
  	"                                               \n\t"
	"                                               \n\t"
	"                                               \n\t"
	"                                               \n\t"
	"                                               \n\t"
	"                                               \n\t"
	"                                               \n\t"
	"                                               \n\t"
	"                                               \n\t"
	"                                               \n\t"
	"                                               \n\t"
	"                                               \n\t"
	"                                               \n\t"
	"                                               \n\t"
	"                                               \n\t"
	"                                               \n\t"
	"                                               \n\t"
	"                                               \n\t"
	"                                               \n\t"
  	"DGENSTOREDBNZ:                                 \n\t"
	#if 0
  	"                                               \n\t" // create offset regs
  	"slwi            %%r12, %%r9, 1                 \n\t"
  	"add             %%r23, %%r22, %%r12            \n\t" // c + rs_c * 2
  	"add             %%r24, %%r23, %%r12           	\n\t" // c + rs_c * 4
  	"add             %%r25, %%r24, %%r12           	\n\t" // c + rs_c * 6 
  	"add             %%r26, %%r25, %%r12           	\n\t" // c + rs_c * 8
  	"add             %%r27, %%r26, %%r12           	\n\t" // c + rs_c * 10
  	"                                              	\n\t"
  GENLOAD_SCALE_UPDATE                                	// (1) load, scale, and move offsets of C
  	"                                               \n\t"
	"xvadddp          %%vs0, %%vs0, %%vs36   	    \n\t" 
  	"xvadddp          %%vs1, %%vs1, %%vs37   	  	\n\t" 
  	"xvadddp          %%vs2, %%vs2, %%vs38   	  	\n\t" 
  	"xvadddp          %%vs3, %%vs3, %%vs39	        \n\t" 
  	"xvadddp          %%vs4, %%vs4, %%vs40   	  	\n\t" 
  	"xvadddp          %%vs5, %%vs5, %%vs41   	  	\n\t"
  	"                                               \n\t"
  	"                                               \n\t"
  GENLOAD_SCALE_UPDATE                                  // (2) load, scale, and move offsets of C
  	"                                               \n\t"
  	"xvadddp          %%vs6, %%vs6, %%vs36          \n\t" 
  	"xvadddp          %%vs7, %%vs7, %%vs37          \n\t" 
  	"xvadddp          %%vs8, %%vs8, %%vs38          \n\t" 
  	"xvadddp          %%vs9, %%vs9, %%vs39          \n\t" 
  	"xvadddp          %%vs10, %%vs10, %%vs40        \n\t" 
  	"xvadddp          %%vs11, %%vs11, %%vs41        \n\t"
  	"                                               \n\t"
  	"                                               \n\t"
  GENLOAD_SCALE_UPDATE                              // (3) load, scale, and move offsets of C
  	"                                               \n\t"
  	"xvadddp          %%vs12, %%vs12, %%vs36        \n\t"
  	"xvadddp          %%vs13, %%vs13, %%vs37        \n\t"
  	"xvadddp          %%vs14, %%vs14, %%vs38        \n\t"
  	"xvadddp          %%vs15, %%vs15, %%vs39        \n\t"
  	"xvadddp          %%vs16, %%vs16, %%vs40        \n\t"
  	"xvadddp          %%vs17, %%vs17, %%vs41        \n\t"
  	"                                               \n\t"
  	"                                          	    \n\t"
  GENLOAD_SCALE_UPDATE                              // (4) load, scale, and move offsets of C
  	"                                               \n\t"
  	"xvadddp          %%vs18, %%vs18, %%vs36        \n\t"
  	"xvadddp          %%vs19, %%vs19, %%vs37        \n\t"
  	"xvadddp          %%vs20, %%vs20, %%vs38        \n\t"
  	"xvadddp          %%vs21, %%vs21, %%vs39        \n\t"
  	"xvadddp          %%vs22, %%vs22, %%vs40        \n\t"
  	"xvadddp          %%vs23, %%vs23, %%vs41        \n\t"
  	"                                               \n\t"
  	"                                               \n\t"
  GENLOAD_SCALE_UPDATE                                 // (5) load, scale, and move offsets of C
  	"                                               \n\t"
  	"xvadddp          %%vs24, %%vs24, %%vs36        \n\t"
  	"xvadddp          %%vs25, %%vs25, %%vs37        \n\t"
  	"xvadddp          %%vs26, %%vs26, %%vs38        \n\t"
  	"xvadddp          %%vs27, %%vs27, %%vs39        \n\t"
  	"xvadddp          %%vs28, %%vs28, %%vs40        \n\t"
  	"xvadddp          %%vs29, %%vs29, %%vs41        \n\t"
  	"                                               \n\t"
  	"                                               \n\t"
  GENLOAD_SCALE_UPDATE                                 // (6) load, scale, and move offsets of C
  	"                                               \n\t"
  	"xvadddp          %%vs30, %%vs30, %%vs36        \n\t"
  	"xvadddp          %%vs31, %%vs31, %%vs37        \n\t"
  	"xvadddp          %%vs32, %%vs32, %%vs38        \n\t"
  	"xvadddp          %%vs33, %%vs33, %%vs39        \n\t"
  	"xvadddp          %%vs34, %%vs34, %%vs40        \n\t"
  	"xvadddp          %%vs35, %%vs35, %%vs41        \n\t"
  	"                                               \n\t"
	#endif
  	"b                DGENSTORED                    \n\t"
  	"                                               \n\t"
  	"                                               \n\t"
	"                                               \n\t"
  	"                                               \n\t"
	"                                               \n\t"
  	"                                               \n\t"
	"                                               \n\t"
  	"                                               \n\t"
	"                                               \n\t"
  	"                                               \n\t"
  	"DCOLSTOREDBNZ:                                 \n\t"
  	"                                               \n\t"
  	"add              %%r23, %%r22, %%r10           \n\t" // c + cs_c
  	"add              %%r24, %%r23, %%r10           \n\t" // c + cs_c * 2
  	"                                               \n\t"
  	"DADDTOC:                                       \n\t" // C = beta*C + alpha*(AB)
  	"                                               \n\t"
 DCOL_LOAD_C
  	"add             %%r22, %%r24, %%r10            \n\t" // c + cs_c * 3
  	"add             %%r23, %%r22, %%r10            \n\t" // c + cs_c * 4
  	"add             %%r24, %%r23, %%r10            \n\t" // c + cs_c * 5
 DCOL_SCALE_BETA
  	"            	                                   \n\t"
  	"xvadddp          %%vs0, %%vs0, %%vs36   	       \n\t" // Begin adding to C
  	"xvadddp          %%vs1, %%vs1, %%vs37   	       \n\t" 
  	"xvadddp          %%vs2, %%vs2, %%vs38   	       \n\t" 
  	"xvadddp          %%vs3, %%vs3, %%vs39   	       \n\t" 
  	"xvadddp          %%vs4, %%vs4, %%vs40   	       \n\t" 
  	"xvadddp          %%vs5, %%vs5, %%vs41   	       \n\t" 
  	"xvadddp          %%vs6, %%vs6, %%vs42   	       \n\t" 
  	"xvadddp          %%vs7, %%vs7, %%vs43     	       \n\t" 
  	"xvadddp          %%vs8, %%vs8, %%vs44    	       \n\t" 
  	"xvadddp          %%vs9, %%vs9, %%vs45   	       \n\t" 
  	"xvadddp          %%vs10, %%vs10, %%vs46   	   	   \n\t" 
  	"xvadddp          %%vs11, %%vs11, %%vs47   	       \n\t" 
  	"xvadddp          %%vs12, %%vs12, %%vs48   	       \n\t" 
  	"xvadddp          %%vs13, %%vs13, %%vs49   	       \n\t" 
  	"xvadddp          %%vs14, %%vs14, %%vs50   	       \n\t" 
  	"xvadddp          %%vs15, %%vs15, %%vs51   	       \n\t" 
  	"xvadddp          %%vs16, %%vs16, %%vs52   	       \n\t" 
  	"xvadddp          %%vs17, %%vs17, %%vs53   	       \n\t" 
  	"                                                  \n\t"
  DCOL_LOAD_C
  DCOL_SCALE_BETA
  	"                                                 \n\t"
  	"xvadddp          %%vs18, %%vs18, %%vs36   	      \n\t" 
  	"xvadddp          %%vs19, %%vs19, %%vs37   	      \n\t" 
  	"xvadddp          %%vs20, %%vs20, %%vs38   	      \n\t" 
  	"xvadddp          %%vs21, %%vs21, %%vs39   	      \n\t" 
  	"xvadddp          %%vs22, %%vs22, %%vs40   	      \n\t" 
  	"xvadddp          %%vs23, %%vs23, %%vs41   	      \n\t" 
  	"xvadddp          %%vs24, %%vs24, %%vs42   	      \n\t" 
  	"xvadddp          %%vs25, %%vs25, %%vs43   	      \n\t" 
  	"xvadddp          %%vs26, %%vs26, %%vs44          \n\t" 
  	"xvadddp          %%vs27, %%vs27, %%vs45   	      \n\t" 
  	"xvadddp          %%vs28, %%vs28, %%vs46   	      \n\t" 
  	"xvadddp          %%vs29, %%vs29, %%vs47   	      \n\t" 
  	"xvadddp          %%vs30, %%vs30, %%vs48   	      \n\t" 
  	"xvadddp          %%vs31, %%vs31, %%vs49   	      \n\t" 
  	"xvadddp          %%vs32, %%vs32, %%vs50   	      \n\t" 
  	"xvadddp          %%vs33, %%vs33, %%vs51   	      \n\t" 
  	"xvadddp          %%vs34, %%vs34, %%vs52   	      \n\t" 
  	"xvadddp          %%vs35, %%vs35, %%vs53   	      \n\t"
  	"                                                 \n\t"
  	"b                DCOLSTORED                      \n\t"
  	"                                                 \n\t"
  	"                                                 \n\t"
  	"                                                 \n\t"
  	"                                                 \n\t"
  	"                                                 \n\t"
  	"DBETAZERO:                                       \n\t" // beta=0 case
  	"                                                 \n\t" 
  	"cmpwi            %%r0, %%r9, 8                   \n\t" // if rs_c == 8,
  	"beq              DCOLSTORED                      \n\t" // C is col stored
  	"                                                 \n\t"
  	"DGENSTORED:                                      \n\t"
	#if 0
  	"                                                 \n\t"
  	"ld              %%r22, %6                        \n\t" // load c
  	"slwi            %%r12, %%r9, 1                   \n\t"
  	"add             %%r23, %%r22, %%r12              \n\t" // c + rs_c * 2
  	"add             %%r24, %%r23, %%r12              \n\t" // c + rs_c * 4
  	"add             %%r25, %%r24, %%r12              \n\t" // c + rs_c * 6 
  	"add             %%r26, %%r25, %%r12              \n\t" // c + rs_c * 8
  	"add             %%r27, %%r26, %%r12              \n\t" // c + rs_c * 10
  	"                                                 \n\t"
  	"                                                 \n\t"
  	"stxsdx          %%vs0, %%r9, %%r22               \n\t" 
  	"xxswapd         %%vs0, %%vs0		              \n\t" 
  	"stxsdx          %%vs0, 0, %%r22                  \n\t" 
  	"stxsdx          %%vs1, %%r9, %%r23               \n\t" 
  	"xxswapd         %%vs1, %%vs1		              \n\t" 
  	"stxsdx          %%vs1, 0, %%r23                  \n\t" 
  	"stxsdx          %%vs2, %%r9, %%r24               \n\t" 
  	"xxswapd         %%vs2, %%vs2		              \n\t" 
  	"stxsdx          %%vs2, 0, %%r24                  \n\t" 
  	"stxsdx          %%vs3, %%r9, %%r25               \n\t" 
  	"xxswapd         %%vs3, %%vs3		              \n\t" 
  	"stxsdx          %%vs3, 0, %%r25                  \n\t" 
  	"stxsdx          %%vs4, %%r9, %%r26               \n\t" 
  	"xxswapd         %%vs4, %%vs4		              \n\t" 
  	"stxsdx          %%vs4, 0, %%r26                  \n\t" 
  	"stxsdx          %%vs5, %%r9, %%r27               \n\t" 
  	"xxswapd         %%vs5, %%vs5		              \n\t" 
  	"stxsdx          %%vs5, 0, %%r27                  \n\t" 
  	"                                                 \n\t"
 GEN_NEXT_COL_CMATRIX 
  	"                                               \n\t"
  	"stxsdx          %%vs6, %%r9, %%r22             \n\t" 
  	"xxswapd         %%vs6, %%vs6		              \n\t" 
  	"stxsdx          %%vs6, 0, %%r22                \n\t" 
  	"stxsdx          %%vs7, %%r9, %%r23             \n\t" 
  	"xxswapd         %%vs7, %%vs7		              \n\t" 
  	"stxsdx          %%vs7, 0, %%r23                \n\t" 
  	"stxsdx          %%vs8, %%r9, %%r24             \n\t" 
  	"xxswapd         %%vs8, %%vs8		              \n\t" 
  	"stxsdx          %%vs8, 0, %%r24                \n\t" 
  	"stxsdx          %%vs9, %%r9, %%r25             \n\t" 
  	"xxswapd         %%vs9, %%vs9		              \n\t" 
  	"stxsdx          %%vs9, 0, %%r25                \n\t" 
  	"stxsdx          %%vs10, %%r9, %%r26            \n\t" 
  	"xxswapd         %%vs10, %%vs10		          \n\t" 
  	"stxsdx          %%vs10, 0, %%r26               \n\t" 
 	"stxsdx          %%vs11, %%r9, %%r27            \n\t" 
  	"xxswapd         %%vs11, %%vs11		          \n\t" 
  	"stxsdx          %%vs11, 0, %%r27               \n\t" 
  	"                                               \n\t"
 GEN_NEXT_COL_CMATRIX 
  	"                                               \n\t"
  	"stxsdx          %%vs12, %%r9, %%r22            \n\t" 
  	"xxswapd         %%vs12, %%vs12		          \n\t" 
  	"stxsdx          %%vs12, 0, %%r22               \n\t" 
  	"stxsdx          %%vs13, %%r9, %%r23            \n\t" 
  	"xxswapd         %%vs13, %%vs13		          \n\t" 
  	"stxsdx          %%vs13, 0, %%r23               \n\t" 
  	"stxsdx          %%vs14, %%r9, %%r24            \n\t" 
  	"xxswapd         %%vs14, %%vs14		          \n\t" 
  	"stxsdx          %%vs14, 0, %%r24               \n\t" 
  	"stxsdx          %%vs15, %%r9, %%r25            \n\t" 
  	"xxswapd         %%vs15, %%vs15		          \n\t" 
  	"stxsdx          %%vs15, 0, %%r25               \n\t" 
  	"stxsdx          %%vs16, %%r9, %%r26            \n\t" 
  	"xxswapd         %%vs16, %%vs16		          \n\t" 
  	"stxsdx          %%vs16, 0, %%r26               \n\t" 
  	"stxsdx          %%vs17, %%r9, %%r27            \n\t" 
  	"xxswapd         %%vs17, %%vs17		          \n\t" 
  	"stxsdx          %%vs17, 0, %%r27               \n\t" 
  	"                                               \n\t"
 GEN_NEXT_COL_CMATRIX 
  	"                                               \n\t"
  	"stxsdx          %%vs18, %%r9, %%r22            \n\t" 
  	"xxswapd         %%vs18, %%vs18		          \n\t" 
  	"stxsdx          %%vs18, 0, %%r22               \n\t" 
  	"stxsdx          %%vs19, %%r9, %%r23            \n\t" 
  	"xxswapd         %%vs19, %%vs19		          \n\t" 
  	"stxsdx          %%vs19, 0, %%r23               \n\t" 
  	"stxsdx          %%vs20, %%r9, %%r24            \n\t" 
  	"xxswapd         %%vs20, %%vs20		          \n\t" 
  	"stxsdx          %%vs20, 0, %%r24               \n\t" 
  	"stxsdx          %%vs21, %%r9, %%r25            \n\t" 
  	"xxswapd         %%vs21, %%vs21		          \n\t" 
  	"stxsdx          %%vs21, 0, %%r25               \n\t" 
  	"stxsdx          %%vs22, %%r9, %%r26            \n\t" 
  	"xxswapd         %%vs22, %%vs22		          \n\t" 
  	"stxsdx          %%vs22, 0, %%r26               \n\t" 
  	"stxsdx          %%vs23, %%r9, %%r27            \n\t" 
  	"xxswapd         %%vs23, %%vs23		          \n\t" 
  	"stxsdx          %%vs23, 0, %%r27               \n\t" 
  	"                                               \n\t"
 GEN_NEXT_COL_CMATRIX 
  	"                                               \n\t"
  	"stxsdx          %%vs24, %%r9, %%r22            \n\t" 
  	"xxswapd         %%vs24, %%vs24		          \n\t" 
  	"stxsdx          %%vs24, 0, %%r22               \n\t" 
  	"stxsdx          %%vs25, %%r9, %%r23            \n\t" 
  	"xxswapd         %%vs25, %%vs25		          \n\t" 
  	"stxsdx          %%vs25, 0, %%r23               \n\t" 
  	"stxsdx          %%vs26, %%r9, %%r24            \n\t" 
  	"xxswapd         %%vs26, %%vs26		          \n\t" 
  	"stxsdx          %%vs26, 0, %%r24               \n\t" 
  	"stxsdx          %%vs27, %%r9, %%r25            \n\t" 
  	"xxswapd         %%vs27, %%vs27		          \n\t" 
  	"stxsdx          %%vs27, 0, %%r25               \n\t" 
  	"stxsdx          %%vs28, %%r9, %%r26            \n\t" 
  	"xxswapd         %%vs28, %%vs28	              \n\t" 
  	"stxsdx          %%vs28, 0, %%r26               \n\t" 
  	"stxsdx          %%vs29, %%r9, %%r27            \n\t" 
  	"xxswapd         %%vs29, %%vs29		          \n\t" 
  	"stxsdx          %%vs29, 0, %%r27               \n\t" 
  	"                                               \n\t"
 GEN_NEXT_COL_CMATRIX 
  	"                                               \n\t"
  	"stxsdx          %%vs30, %%r9, %%r22            \n\t" 
  	"xxswapd         %%vs30, %%vs30		          \n\t" 
  	"stxsdx          %%vs30, 0, %%r22               \n\t" 
  	"stxsdx          %%vs31, %%r9, %%r23            \n\t" 
  	"xxswapd         %%vs31, %%vs31		          \n\t" 
  	"stxsdx          %%vs31, 0, %%r23               \n\t" 
  	"stxsdx          %%vs32, %%r9, %%r24            \n\t" 
  	"xxswapd         %%vs32, %%vs32		          \n\t" 
  	"stxsdx          %%vs32, 0, %%r24               \n\t" 
  	"stxsdx          %%vs33, %%r9, %%r25            \n\t" 
  	"xxswapd         %%vs33, %%vs33		          \n\t" 
  	"stxsdx          %%vs33, 0, %%r25               \n\t" 
  	"stxsdx          %%vs34, %%r9, %%r26            \n\t" 
  	"xxswapd         %%vs34, %%vs34	              \n\t" 
  	"stxsdx          %%vs34, 0, %%r26               \n\t" 
  	"stxsdx          %%vs35, %%r9, %%r27            \n\t" 
  	"xxswapd         %%vs35, %%vs35		          \n\t" 
  	"stxsdx          %%vs35, 0, %%r27               \n\t"
  	"                                               \n\t"
	#endif
  	"b               DDONE                          \n\t"
  	"                                               \n\t"
  	"DCOLSTORED:                                    \n\t"
  	"                                               \n\t" // create offset regs
  	"add              %%r17, %%r16, %%r10           \n\t" // c + cs_c
  	"add              %%r18, %%r17, %%r10           \n\t" // c + cs_c * 2 
  	"add              %%r19, %%r18, %%r10           \n\t" // c + cs_c * 3
  	"add              %%r20, %%r19, %%r10           \n\t" // c + cs_c * 4
  	"add              %%r21, %%r20, %%r10           \n\t" // c + cs_c * 5
  	"                                               \n\t"
 	DCOL_STORE
  	"                                               \n\t"
  	"DDONE:                                         \n\t"  
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
  /* unclobberable regs: r2, r3, r4, r5, r6, r13, r14, r15, r30, r31 */
  "r0",  "r7", "r8", "r9",
  "r10", "r12","r16", "r17", "r18", "r19", 
  "r20", "r21", "r22", "r23", "r24", "r25", "r26", "r27", "r28"

  #if XLC
  ,"f0", "f1", "f2", "f3", "f4", "f5", "f6", "f7", "f8", "f9"
  , "f10", "f11", "f12", "f13", "f14", "f15", "f16", "f17", "f18", "f19"
  , "f20" ,"f21", "f22", "f23", "f24", "f25", "f26", "f27", "f28", "f29"
  , "f30" ,"f31"
  , "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9"
  , "v10", "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v19"
  , "v20", "v21", "v22", "v23", "v24", "v25", "v26", "v27", "v28", "v29"
  , "v30", "v31"
  #else
  , "vs0", "vs1", "vs2", "vs3", "vs4", "vs5", "vs6", "vs7", "vs8", "vs9"
  , "vs10", "vs11", "vs12", "vs13", "vs14", "vs15", "vs16", "vs17", "vs18", "vs19"
  , "vs20", "vs21", "vs22", "vs23", "vs24", "vs25", "vs26", "vs27", "vs28", "vs29"
  , "vs30", "vs31", "vs32", "vs33", "vs34", "vs35", "vs36", "vs37", "vs38", "vs39"
  , "vs40", "vs41", "vs42", "vs43", "vs44", "vs45", "vs46", "vs47", "vs48", "vs49"
  , "vs50", "vs51", "vs52", "vs53"
  

  #endif
  );
}
