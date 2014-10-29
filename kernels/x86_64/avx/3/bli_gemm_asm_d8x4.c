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
      derived from this software without specific prior written permission.

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

/* NOTE: The micro-kernels in this file were partially inspired by portions
   of code found in OpenBLAS 0.2.8 (http://www.openblas.net/). -FGVZ */

#include "blis.h"

void bli_sgemm_asm_8x8(
                        dim_t              k,
                        float* restrict    alpha,
                        float* restrict    a,
                        float* restrict    b,
                        float* restrict    beta,
                        float* restrict    c, inc_t rs_c, inc_t cs_c,
                        auxinfo_t*         data
                      )
{
	//void*   a_next = bli_auxinfo_next_a( data );
	//void*   b_next = bli_auxinfo_next_b( data );

	dim_t   k_iter = k / 4;
	dim_t   k_left = k % 4;

	__asm__ volatile
	(
	"                                            \n\t"
	"                                            \n\t"
	"movq                %2, %%rax               \n\t" // load address of a.
	"movq                %3, %%rbx               \n\t" // load address of b.
	//"movq                %9, %%r15               \n\t" // load address of b_next.
	"                                            \n\t"
	"vmovaps    0 * 32(%%rax), %%ymm0            \n\t" // initialize loop by pre-loading
	"vmovsldup  0 * 32(%%rbx), %%ymm2            \n\t" // elements of a and b.
	"vpermilps   $0x4e, %%ymm2, %%ymm3           \n\t"
	"                                            \n\t"
	"movq                %6, %%rcx               \n\t" // load address of c
	"movq                %8, %%rdi               \n\t" // load cs_c
	"leaq        (,%%rdi,4), %%rdi               \n\t" // cs_c *= sizeof(float)
	"leaq   (%%rcx,%%rdi,4), %%r10               \n\t" // load address of c + 4*cs_c;
	"                                            \n\t"
	"leaq   (%%rdi,%%rdi,2), %%r14               \n\t" // r14 = 3*cs_c;
	"prefetcht0   7 * 8(%%rcx)                   \n\t" // prefetch c + 0*cs_c
	"prefetcht0   7 * 8(%%rcx,%%rdi)             \n\t" // prefetch c + 1*cs_c
	"prefetcht0   7 * 8(%%rcx,%%rdi,2)           \n\t" // prefetch c + 2*cs_c
	"prefetcht0   7 * 8(%%rcx,%%r14)             \n\t" // prefetch c + 3*cs_c
	"prefetcht0   7 * 8(%%r10)                   \n\t" // prefetch c + 4*cs_c
	"prefetcht0   7 * 8(%%r10,%%rdi)             \n\t" // prefetch c + 5*cs_c
	"prefetcht0   7 * 8(%%r10,%%rdi,2)           \n\t" // prefetch c + 6*cs_c
	"prefetcht0   7 * 8(%%r10,%%r14)             \n\t" // prefetch c + 7*cs_c
	"                                            \n\t"
	"vxorps    %%ymm8,  %%ymm8,  %%ymm8          \n\t"
	"vxorps    %%ymm9,  %%ymm9,  %%ymm9          \n\t"
	"vxorps    %%ymm10, %%ymm10, %%ymm10         \n\t"
	"vxorps    %%ymm11, %%ymm11, %%ymm11         \n\t"
	"vxorps    %%ymm12, %%ymm12, %%ymm12         \n\t"
	"vxorps    %%ymm13, %%ymm13, %%ymm13         \n\t"
	"vxorps    %%ymm14, %%ymm14, %%ymm14         \n\t"
	"vxorps    %%ymm15, %%ymm15, %%ymm15         \n\t"
	"                                            \n\t"
	"                                            \n\t"
	"                                            \n\t"
	"movq      %0, %%rsi                         \n\t" // i = k_iter;
	"testq  %%rsi, %%rsi                         \n\t" // check i via logical AND.
	"je     .SCONSIDKLEFT                        \n\t" // if i == 0, jump to code that
	"                                            \n\t" // contains the k_left loop.
	"                                            \n\t"
	"                                            \n\t"
	".SLOOPKITER:                                \n\t" // MAIN LOOP
	"                                            \n\t"
	"                                            \n\t"
	"                                            \n\t" // iteration 0
	"prefetcht0  16 * 32(%%rax)                  \n\t"
	"vmulps            %%ymm0,  %%ymm2, %%ymm6   \n\t"
	"vperm2f128 $0x03, %%ymm2,  %%ymm2, %%ymm4   \n\t"
	"vmovshdup  0 * 32(%%rbx),  %%ymm2           \n\t"
	"vmulps            %%ymm0,  %%ymm3, %%ymm7   \n\t"
	"vperm2f128 $0x03, %%ymm3,  %%ymm3, %%ymm5   \n\t"
	"vaddps            %%ymm15, %%ymm6, %%ymm15  \n\t"
	"vaddps            %%ymm13, %%ymm7, %%ymm13  \n\t"
	"                                            \n\t"
	"vmovaps    1 * 32(%%rax),  %%ymm1           \n\t"
	"vpermilps  $0x4e, %%ymm2,  %%ymm3           \n\t"
	"vmulps            %%ymm0,  %%ymm4, %%ymm6   \n\t"
	"vmulps            %%ymm0,  %%ymm5, %%ymm7   \n\t"
	"vaddps            %%ymm11, %%ymm6, %%ymm11  \n\t"
	"vaddps            %%ymm9,  %%ymm7, %%ymm9   \n\t"
	"                                            \n\t"
	"vmulps            %%ymm0,  %%ymm2, %%ymm6   \n\t"
	"vperm2f128 $0x03, %%ymm2,  %%ymm2, %%ymm4   \n\t"
	"vmovsldup  1 * 32(%%rbx),  %%ymm2           \n\t"
	"vmulps            %%ymm0,  %%ymm3, %%ymm7   \n\t"
	"vperm2f128 $0x03, %%ymm3,  %%ymm3, %%ymm5   \n\t"
	"vaddps            %%ymm14, %%ymm6, %%ymm14  \n\t"
	"vaddps            %%ymm12, %%ymm7, %%ymm12  \n\t"
	"                                            \n\t"
	"vpermilps  $0x4e, %%ymm2,  %%ymm3           \n\t"
	"vmulps            %%ymm0,  %%ymm4, %%ymm6   \n\t"
	"vmulps            %%ymm0,  %%ymm5, %%ymm7   \n\t"
	"vaddps            %%ymm10, %%ymm6, %%ymm10  \n\t"
	"vaddps            %%ymm8,  %%ymm7, %%ymm8   \n\t"
	"                                            \n\t"
	"                                            \n\t" // iteration 1
	"vmulps            %%ymm1,  %%ymm2, %%ymm6   \n\t"
	"vperm2f128 $0x03, %%ymm2,  %%ymm2, %%ymm4   \n\t"
	"vmovshdup  1 * 32(%%rbx), %%ymm2            \n\t"
	"vmulps            %%ymm1,  %%ymm3, %%ymm7   \n\t"
	"vperm2f128 $0x03, %%ymm3,  %%ymm3, %%ymm5   \n\t"
	"vaddps            %%ymm15, %%ymm6, %%ymm15  \n\t"
	"vaddps            %%ymm13, %%ymm7, %%ymm13  \n\t"
	"                                            \n\t"
	"vmovaps    2 * 32(%%rax),  %%ymm0           \n\t"
	"vpermilps  $0x4e, %%ymm2,  %%ymm3           \n\t"
	"vmulps            %%ymm1,  %%ymm4, %%ymm6   \n\t"
	"vmulps            %%ymm1,  %%ymm5, %%ymm7   \n\t"
	"vaddps            %%ymm11, %%ymm6, %%ymm11  \n\t"
	"vaddps            %%ymm9,  %%ymm7, %%ymm9   \n\t"
	"                                            \n\t"
	"vmulps            %%ymm1,  %%ymm2, %%ymm6   \n\t"
	"vperm2f128 $0x03, %%ymm2,  %%ymm2, %%ymm4   \n\t"
	"vmovsldup  2 * 32(%%rbx),  %%ymm2           \n\t"
	"vmulps            %%ymm1,  %%ymm3, %%ymm7   \n\t"
	"vperm2f128 $0x03, %%ymm3,  %%ymm3, %%ymm5   \n\t"
	"vaddps            %%ymm14, %%ymm6, %%ymm14  \n\t"
	"vaddps            %%ymm12, %%ymm7, %%ymm12  \n\t"
	"                                            \n\t"
	"vpermilps  $0x4e, %%ymm2,  %%ymm3           \n\t"
	"vmulps            %%ymm1,  %%ymm4, %%ymm6   \n\t"
	"vmulps            %%ymm1,  %%ymm5, %%ymm7   \n\t"
	"vaddps            %%ymm10, %%ymm6, %%ymm10  \n\t"
	"vaddps            %%ymm8,  %%ymm7, %%ymm8   \n\t"
	"                                            \n\t"
	"                                            \n\t"
	"                                            \n\t" // iteration 2
	"prefetcht0  18 * 32(%%rax)                  \n\t"
	"vmulps            %%ymm0,  %%ymm2, %%ymm6   \n\t"
	"vperm2f128 $0x03, %%ymm2,  %%ymm2, %%ymm4   \n\t"
	"vmovshdup  2 * 32(%%rbx),  %%ymm2           \n\t"
	"vmulps            %%ymm0,  %%ymm3, %%ymm7   \n\t"
	"vperm2f128 $0x03, %%ymm3,  %%ymm3, %%ymm5   \n\t"
	"vaddps            %%ymm15, %%ymm6, %%ymm15  \n\t"
	"vaddps            %%ymm13, %%ymm7, %%ymm13  \n\t"
	"                                            \n\t"
	"vmovaps    3 * 32(%%rax),  %%ymm1           \n\t"
	"addq           $4 * 8 * 4, %%rax            \n\t" // a += 4*8 (unroll x mr)
	"vpermilps  $0x4e, %%ymm2,  %%ymm3           \n\t"
	"vmulps            %%ymm0,  %%ymm4, %%ymm6   \n\t"
	"vmulps            %%ymm0,  %%ymm5, %%ymm7   \n\t"
	"vaddps            %%ymm11, %%ymm6, %%ymm11  \n\t"
	"vaddps            %%ymm9,  %%ymm7, %%ymm9   \n\t"
	"                                            \n\t"
	"vmulps            %%ymm0,  %%ymm2, %%ymm6   \n\t"
	"vperm2f128 $0x03, %%ymm2,  %%ymm2, %%ymm4   \n\t"
	"vmovsldup  3 * 32(%%rbx),  %%ymm2           \n\t"
	"vmulps            %%ymm0,  %%ymm3, %%ymm7   \n\t"
	"vperm2f128 $0x03, %%ymm3,  %%ymm3, %%ymm5   \n\t"
	"vaddps            %%ymm14, %%ymm6, %%ymm14  \n\t"
	"vaddps            %%ymm12, %%ymm7, %%ymm12  \n\t"
	"                                            \n\t"
	"vpermilps  $0x4e, %%ymm2,  %%ymm3           \n\t"
	"vmulps            %%ymm0,  %%ymm4, %%ymm6   \n\t"
	"vmulps            %%ymm0,  %%ymm5, %%ymm7   \n\t"
	"vaddps            %%ymm10, %%ymm6, %%ymm10  \n\t"
	"vaddps            %%ymm8,  %%ymm7, %%ymm8   \n\t"
	"                                            \n\t"
	"                                            \n\t"
	"                                            \n\t" // iteration 3
	"vmulps            %%ymm1,  %%ymm2, %%ymm6   \n\t"
	"vperm2f128 $0x03, %%ymm2,  %%ymm2, %%ymm4   \n\t"
	"vmovshdup  3 * 32(%%rbx), %%ymm2            \n\t"
	"addq           $4 * 8 * 4, %%rbx            \n\t" // b += 4*8 (unroll x nr)
	"vmulps            %%ymm1,  %%ymm3, %%ymm7   \n\t"
	"vperm2f128 $0x03, %%ymm3,  %%ymm3, %%ymm5   \n\t"
	"vaddps            %%ymm15, %%ymm6, %%ymm15  \n\t"
	"vaddps            %%ymm13, %%ymm7, %%ymm13  \n\t"
	"                                            \n\t"
	"vmovaps    0 * 32(%%rax),  %%ymm0           \n\t"
	"vpermilps  $0x4e, %%ymm2,  %%ymm3           \n\t"
	"vmulps            %%ymm1,  %%ymm4, %%ymm6   \n\t"
	"vmulps            %%ymm1,  %%ymm5, %%ymm7   \n\t"
	"vaddps            %%ymm11, %%ymm6, %%ymm11  \n\t"
	"vaddps            %%ymm9,  %%ymm7, %%ymm9   \n\t"
	"                                            \n\t"
	"vmulps            %%ymm1,  %%ymm2, %%ymm6   \n\t"
	"vperm2f128 $0x03, %%ymm2,  %%ymm2, %%ymm4   \n\t"
	"vmovsldup  0 * 32(%%rbx),  %%ymm2           \n\t"
	"vmulps            %%ymm1,  %%ymm3, %%ymm7   \n\t"
	"vperm2f128 $0x03, %%ymm3,  %%ymm3, %%ymm5   \n\t"
	"vaddps            %%ymm14, %%ymm6, %%ymm14  \n\t"
	"vaddps            %%ymm12, %%ymm7, %%ymm12  \n\t"
	"                                            \n\t"
	"vpermilps  $0x4e, %%ymm2,  %%ymm3           \n\t"
	"vmulps            %%ymm1,  %%ymm4, %%ymm6   \n\t"
	"vmulps            %%ymm1,  %%ymm5, %%ymm7   \n\t"
	"vaddps            %%ymm10, %%ymm6, %%ymm10  \n\t"
	"vaddps            %%ymm8,  %%ymm7, %%ymm8   \n\t"
	"                                            \n\t"
	"                                            \n\t"
	"                                            \n\t"
	"                                            \n\t"
	"decq   %%rsi                                \n\t" // i -= 1;
	"jne    .SLOOPKITER                          \n\t" // iterate again if i != 0.
	"                                            \n\t"
	"                                            \n\t"
	"                                            \n\t"
	"                                            \n\t"
	"                                            \n\t"
	"                                            \n\t"
	".SCONSIDKLEFT:                              \n\t"
	"                                            \n\t"
	"movq      %1, %%rsi                         \n\t" // i = k_left;
	"testq  %%rsi, %%rsi                         \n\t" // check i via logical AND.
	"je     .SPOSTACCUM                          \n\t" // if i == 0, we're done; jump to end.
	"                                            \n\t" // else, we prepare to enter k_left loop.
	"                                            \n\t"
	"                                            \n\t"
	".SLOOPKLEFT:                                \n\t" // EDGE LOOP
	"                                            \n\t"
	"                                            \n\t"
	"prefetcht0  16 * 32(%%rax)                  \n\t"
	"vmulps            %%ymm0,  %%ymm2, %%ymm6   \n\t"
	"vperm2f128  $0x3, %%ymm2,  %%ymm2, %%ymm4   \n\t"
	"vmovshdup  0 * 32(%%rbx),  %%ymm2           \n\t"
	"vmulps            %%ymm0,  %%ymm3, %%ymm7   \n\t"
	"vperm2f128  $0x3, %%ymm3,  %%ymm3, %%ymm5   \n\t"
	"vaddps            %%ymm15, %%ymm6, %%ymm15  \n\t"
	"vaddps            %%ymm13, %%ymm7, %%ymm13  \n\t"
	"                                            \n\t"
	"vmovaps    1 * 32(%%rax),  %%ymm1           \n\t"
	"addq           $8 * 1 * 4, %%rax            \n\t" // a += 8 (1 x mr)
	"vpermilps  $0x4e, %%ymm2,  %%ymm3           \n\t"
	"vmulps            %%ymm0,  %%ymm4, %%ymm6   \n\t"
	"vmulps            %%ymm0,  %%ymm5, %%ymm7   \n\t"
	"vaddps            %%ymm11, %%ymm6, %%ymm11  \n\t"
	"vaddps            %%ymm9,  %%ymm7, %%ymm9   \n\t"
	"                                            \n\t"
	"vmulps            %%ymm0,  %%ymm2, %%ymm6   \n\t"
	"vperm2f128  $0x3, %%ymm2,  %%ymm2, %%ymm4   \n\t"
	"vmovsldup  1 * 32(%%rbx),  %%ymm2           \n\t"
	"addq           $8 * 1 * 4, %%rbx            \n\t" // b += 8 (1 x nr)
	"vmulps            %%ymm0,  %%ymm3, %%ymm7   \n\t"
	"vperm2f128  $0x3, %%ymm3,  %%ymm3, %%ymm5   \n\t"
	"vaddps            %%ymm14, %%ymm6, %%ymm14  \n\t"
	"vaddps            %%ymm12, %%ymm7, %%ymm12  \n\t"
	"                                            \n\t"
	"vpermilps  $0x4e, %%ymm2,  %%ymm3           \n\t"
	"vmulps            %%ymm0,  %%ymm4, %%ymm6   \n\t"
	"vmulps            %%ymm0,  %%ymm5, %%ymm7   \n\t"
	"vmovaps           %%ymm1,  %%ymm0           \n\t"
	"vaddps            %%ymm10, %%ymm6, %%ymm10  \n\t"
	"vaddps            %%ymm8,  %%ymm7, %%ymm8   \n\t"
	"                                            \n\t"
	"                                            \n\t"
	"                                            \n\t"
	"decq   %%rsi                                \n\t" // i -= 1;
	"jne    .SLOOPKLEFT                          \n\t" // iterate again if i != 0.
	"                                            \n\t"
	"                                            \n\t"
	"                                            \n\t"
	".SPOSTACCUM:                                \n\t"
	"                                            \n\t"
	"                                            \n\t" // ymm15:  ymm13:  ymm11:  ymm9:
	"                                            \n\t" // ( ab00  ( ab02  ( ab04  ( ab06
	"                                            \n\t" //   ab10    ab12    ab14    ab16  
	"                                            \n\t" //   ab22    ab20    ab26    ab24
	"                                            \n\t" //   ab32    ab30    ab36    ab34
	"                                            \n\t" //   ab44    ab46    ab40    ab42
	"                                            \n\t" //   ab54    ab56    ab50    ab52  
	"                                            \n\t" //   ab66    ab64    ab62    ab60
	"                                            \n\t" //   ab76 )  ab74 )  ab72 )  ab70 )
	"                                            \n\t"
	"                                            \n\t" // ymm14:  ymm12:  ymm10:  ymm8:
	"                                            \n\t" // ( ab01  ( ab03  ( ab05  ( ab07
	"                                            \n\t" //   ab11    ab13    ab15    ab17  
	"                                            \n\t" //   ab23    ab21    ab27    ab25
	"                                            \n\t" //   ab33    ab31    ab37    ab35
	"                                            \n\t" //   ab45    ab47    ab41    ab43
	"                                            \n\t" //   ab55    ab57    ab51    ab53  
	"                                            \n\t" //   ab67    ab65    ab63    ab61
	"                                            \n\t" //   ab77 )  ab75 )  ab73 )  ab71 )
	"                                            \n\t"
	"vmovaps          %%ymm15, %%ymm7            \n\t"
	"vshufps   $0xe4, %%ymm13, %%ymm15, %%ymm15  \n\t"
	"vshufps   $0xe4, %%ymm7,  %%ymm13, %%ymm13  \n\t"
	"                                            \n\t"
	"vmovaps          %%ymm11, %%ymm7            \n\t"
	"vshufps   $0xe4, %%ymm9,  %%ymm11, %%ymm11  \n\t"
	"vshufps   $0xe4, %%ymm7,  %%ymm9,  %%ymm9   \n\t"
	"                                            \n\t"
	"vmovaps          %%ymm14, %%ymm7            \n\t"
	"vshufps   $0xe4, %%ymm12, %%ymm14, %%ymm14  \n\t"
	"vshufps   $0xe4, %%ymm7,  %%ymm12, %%ymm12  \n\t"
	"                                            \n\t"
	"vmovaps          %%ymm10, %%ymm7            \n\t"
	"vshufps   $0xe4, %%ymm8,  %%ymm10, %%ymm10  \n\t"
	"vshufps   $0xe4, %%ymm7,  %%ymm8,  %%ymm8   \n\t"
	"                                            \n\t"
	"                                            \n\t" // ymm15:  ymm13:  ymm11:  ymm9:
	"                                            \n\t" // ( ab00  ( ab02  ( ab04  ( ab06
	"                                            \n\t" //   ab10    ab12    ab14    ab16  
	"                                            \n\t" //   ab20    ab22    ab24    ab26
	"                                            \n\t" //   ab30    ab32    ab34    ab36
	"                                            \n\t" //   ab44    ab46    ab40    ab42
	"                                            \n\t" //   ab54    ab56    ab50    ab52  
	"                                            \n\t" //   ab64    ab66    ab60    ab62
	"                                            \n\t" //   ab74 )  ab76 )  ab70 )  ab72 )
	"                                            \n\t"
	"                                            \n\t" // ymm14:  ymm12:  ymm10:  ymm8:
	"                                            \n\t" // ( ab01  ( ab03  ( ab05  ( ab07
	"                                            \n\t" //   ab11    ab13    ab15    ab17  
	"                                            \n\t" //   ab21    ab23    ab25    ab27
	"                                            \n\t" //   ab31    ab33    ab35    ab37
	"                                            \n\t" //   ab45    ab47    ab41    ab43
	"                                            \n\t" //   ab55    ab57    ab51    ab53  
	"                                            \n\t" //   ab65    ab67    ab61    ab63
	"                                            \n\t" //   ab75 )  ab77 )  ab71 )  ab73 )
	"                                            \n\t"
	"vmovaps           %%ymm15, %%ymm7           \n\t"
	"vperm2f128 $0x30, %%ymm11, %%ymm15, %%ymm15 \n\t"
	"vperm2f128 $0x12, %%ymm11, %%ymm7,  %%ymm11 \n\t"
	"                                            \n\t"
	"vmovaps           %%ymm13, %%ymm7           \n\t"
	"vperm2f128 $0x30, %%ymm9,  %%ymm13, %%ymm13 \n\t"
	"vperm2f128 $0x12, %%ymm9,  %%ymm7,  %%ymm9  \n\t"
	"                                            \n\t"
	"vmovaps           %%ymm14, %%ymm7           \n\t"
	"vperm2f128 $0x30, %%ymm10, %%ymm14, %%ymm14 \n\t"
	"vperm2f128 $0x12, %%ymm10, %%ymm7,  %%ymm10 \n\t"
	"                                            \n\t"
	"vmovaps           %%ymm12, %%ymm7           \n\t"
	"vperm2f128 $0x30, %%ymm8,  %%ymm12, %%ymm12 \n\t"
	"vperm2f128 $0x12, %%ymm8,  %%ymm7,  %%ymm8  \n\t"
	"                                            \n\t"
	"                                            \n\t" // ymm15:  ymm13:  ymm11:  ymm9:
	"                                            \n\t" // ( ab00  ( ab02  ( ab04  ( ab06
	"                                            \n\t" //   ab10    ab12    ab14    ab16  
	"                                            \n\t" //   ab20    ab22    ab24    ab26
	"                                            \n\t" //   ab30    ab32    ab34    ab36
	"                                            \n\t" //   ab40    ab42    ab44    ab46
	"                                            \n\t" //   ab50    ab52    ab54    ab56  
	"                                            \n\t" //   ab60    ab62    ab64    ab66
	"                                            \n\t" //   ab70 )  ab72 )  ab74 )  ab76 )
	"                                            \n\t"
	"                                            \n\t" // ymm14:  ymm12:  ymm10:  ymm8:
	"                                            \n\t" // ( ab01  ( ab03  ( ab05  ( ab07
	"                                            \n\t" //   ab11    ab13    ab15    ab17  
	"                                            \n\t" //   ab21    ab23    ab25    ab27
	"                                            \n\t" //   ab31    ab33    ab35    ab37
	"                                            \n\t" //   ab41    ab43    ab45    ab47
	"                                            \n\t" //   ab51    ab53    ab55    ab57  
	"                                            \n\t" //   ab61    ab63    ab65    ab67
	"                                            \n\t" //   ab71 )  ab73 )  ab75 )  ab77 )
	"                                            \n\t"
	"                                            \n\t"
	"                                            \n\t"
	"movq         %4, %%rax                      \n\t" // load address of alpha
	"movq         %5, %%rbx                      \n\t" // load address of beta 
	"vbroadcastss    (%%rax), %%ymm0             \n\t" // load alpha and duplicate
	"vbroadcastss    (%%rbx), %%ymm4             \n\t" // load beta and duplicate
	"                                            \n\t"
	"vmulps           %%ymm0,  %%ymm8,  %%ymm8   \n\t" // scale by alpha
	"vmulps           %%ymm0,  %%ymm9,  %%ymm9   \n\t"
	"vmulps           %%ymm0,  %%ymm10, %%ymm10  \n\t"
	"vmulps           %%ymm0,  %%ymm11, %%ymm11  \n\t"
	"vmulps           %%ymm0,  %%ymm12, %%ymm12  \n\t"
	"vmulps           %%ymm0,  %%ymm13, %%ymm13  \n\t"
	"vmulps           %%ymm0,  %%ymm14, %%ymm14  \n\t"
	"vmulps           %%ymm0,  %%ymm15, %%ymm15  \n\t"
	"                                            \n\t"
	"                                            \n\t"
	"                                            \n\t"
	"                                            \n\t"
	"                                            \n\t"
	"                                            \n\t"
	"movq                %7, %%rsi               \n\t" // load rs_c
	"leaq        (,%%rsi,4), %%rsi               \n\t" // rsi = rs_c * sizeof(float)
	"                                            \n\t"
	"leaq   (%%rcx,%%rsi,4), %%rdx               \n\t" // load address of c + 4*rs_c;
	"                                            \n\t"
	"leaq        (,%%rsi,2), %%r12               \n\t" // r12 = 2*rs_c;
	"leaq   (%%r12,%%rsi,1), %%r13               \n\t" // r13 = 3*rs_c;
	"                                            \n\t"
	"                                            \n\t"
	"                                            \n\t"
	"                                            \n\t" // determine if
	"                                            \n\t" //    c    % 32 == 0, AND
	"                                            \n\t" //  4*cs_c % 32 == 0, AND
	"                                            \n\t" //    rs_c      == 1
	"                                            \n\t" // ie: aligned, ldim aligned, and
	"                                            \n\t" // column-stored
	"                                            \n\t"
	"cmpq       $4, %%rsi                        \n\t" // set ZF if (4*rs_c) == 4.
	"sete           %%bl                         \n\t" // bl = ( ZF == 1 ? 1 : 0 );
	"testq     $31, %%rcx                        \n\t" // set ZF if c & 32 is zero.
	"setz           %%bh                         \n\t" // bh = ( ZF == 0 ? 1 : 0 );
	"testq     $31, %%rdi                        \n\t" // set ZF if (4*cs_c) & 32 is zero.
	"setz           %%al                         \n\t" // al = ( ZF == 0 ? 1 : 0 );
	"                                            \n\t" // and(bl,bh) followed by
	"                                            \n\t" // and(bh,al) will reveal result
	"                                            \n\t"
	"                                            \n\t" // now avoid loading C if beta == 0
	"                                            \n\t"
	"vxorps    %%ymm0,  %%ymm0,  %%ymm0          \n\t" // set ymm0 to zero.
	"vucomiss  %%xmm0,  %%xmm4                   \n\t" // set ZF if beta == 0.
	"je      .SBETAZERO                          \n\t" // if ZF = 1, jump to beta == 0 case
	"                                            \n\t"
	"                                            \n\t"
	"                                            \n\t" // check if aligned/column-stored
	"andb     %%bl, %%bh                         \n\t" // set ZF if bl & bh == 1.
	"andb     %%bh, %%al                         \n\t" // set ZF if bh & al == 1.
	"jne     .SCOLSTORED                         \n\t" // jump to column storage case
	"                                            \n\t"
	"                                            \n\t"
	"                                            \n\t"
	".SGENSTORED:                                \n\t"
	"                                            \n\t"
	"                                            \n\t" // update c00:c70
	"vmovlps    (%%rcx),        %%xmm0,  %%xmm0  \n\t"
	"vmovhps    (%%rcx,%%rsi),  %%xmm0,  %%xmm0  \n\t"
	"vmovlps    (%%rcx,%%r12),  %%xmm1,  %%xmm1  \n\t"
	"vmovhps    (%%rcx,%%r13),  %%xmm1,  %%xmm1  \n\t"
	"vshufps    $0x88, %%xmm1,  %%xmm0,  %%xmm0  \n\t"
	"vmovlps    (%%rdx),        %%xmm2,  %%xmm2  \n\t"
	"vmovhps    (%%rdx,%%rsi),  %%xmm2,  %%xmm2  \n\t"
	"vmovlps    (%%rdx,%%r12),  %%xmm3,  %%xmm3  \n\t"
	"vmovhps    (%%rdx,%%r13),  %%xmm3,  %%xmm3  \n\t"
	"vshufps    $0x88, %%xmm3,  %%xmm2,  %%xmm2  \n\t"
	"vperm2f128 $0x20, %%ymm2,  %%ymm0,  %%ymm0  \n\t"
	"                                            \n\t"
	"vmulps            %%ymm4,  %%ymm0,  %%ymm0  \n\t" // scale by beta,
	"vaddps            %%ymm15, %%ymm0,  %%ymm0  \n\t" // add the gemm result,
	"                                            \n\t"
	"vextractf128  $1, %%ymm0,  %%xmm2           \n\t"
	"vmovss            %%xmm0, (%%rcx)           \n\t"
	"vpermilps  $0x39, %%xmm0,  %%xmm1           \n\t"
	"vmovss            %%xmm1, (%%rcx,%%rsi)     \n\t"
	"vpermilps  $0x39, %%xmm1,  %%xmm0           \n\t"
	"vmovss            %%xmm0, (%%rcx,%%r12)     \n\t"
	"vpermilps  $0x39, %%xmm0,  %%xmm1           \n\t"
	"vmovss            %%xmm1, (%%rcx,%%r13)     \n\t"
	"vmovss            %%xmm2, (%%rdx)           \n\t"
	"vpermilps  $0x39, %%xmm2,  %%xmm3           \n\t"
	"vmovss            %%xmm3, (%%rdx,%%rsi)     \n\t"
	"vpermilps  $0x39, %%xmm3,  %%xmm2           \n\t"
	"vmovss            %%xmm2, (%%rdx,%%r12)     \n\t"
	"vpermilps  $0x39, %%xmm2,  %%xmm3           \n\t"
	"vmovss            %%xmm3, (%%rdx,%%r13)     \n\t"
	"                                            \n\t"
	"addq      %%rdi, %%rcx                      \n\t" // c += cs_c;
	"addq      %%rdi, %%rdx                      \n\t" // c += cs_c;
	"                                            \n\t"
	"                                            \n\t"
	"                                            \n\t" // update c01:c71
	"vmovlps    (%%rcx),        %%xmm0,  %%xmm0  \n\t"
	"vmovhps    (%%rcx,%%rsi),  %%xmm0,  %%xmm0  \n\t"
	"vmovlps    (%%rcx,%%r12),  %%xmm1,  %%xmm1  \n\t"
	"vmovhps    (%%rcx,%%r13),  %%xmm1,  %%xmm1  \n\t"
	"vshufps    $0x88, %%xmm1,  %%xmm0,  %%xmm0  \n\t"
	"vmovlps    (%%rdx),        %%xmm2,  %%xmm2  \n\t"
	"vmovhps    (%%rdx,%%rsi),  %%xmm2,  %%xmm2  \n\t"
	"vmovlps    (%%rdx,%%r12),  %%xmm3,  %%xmm3  \n\t"
	"vmovhps    (%%rdx,%%r13),  %%xmm3,  %%xmm3  \n\t"
	"vshufps    $0x88, %%xmm3,  %%xmm2,  %%xmm2  \n\t"
	"vperm2f128 $0x20, %%ymm2,  %%ymm0,  %%ymm0  \n\t"
	"                                            \n\t"
	"vmulps            %%ymm4,  %%ymm0,  %%ymm0  \n\t" // scale by beta,
	"vaddps            %%ymm14, %%ymm0,  %%ymm0  \n\t" // add the gemm result,
	"                                            \n\t"
	"vextractf128  $1, %%ymm0,  %%xmm2           \n\t"
	"vmovss            %%xmm0, (%%rcx)           \n\t"
	"vpermilps  $0x39, %%xmm0,  %%xmm1           \n\t"
	"vmovss            %%xmm1, (%%rcx,%%rsi)     \n\t"
	"vpermilps  $0x39, %%xmm1,  %%xmm0           \n\t"
	"vmovss            %%xmm0, (%%rcx,%%r12)     \n\t"
	"vpermilps  $0x39, %%xmm0,  %%xmm1           \n\t"
	"vmovss            %%xmm1, (%%rcx,%%r13)     \n\t"
	"vmovss            %%xmm2, (%%rdx)           \n\t"
	"vpermilps  $0x39, %%xmm2,  %%xmm3           \n\t"
	"vmovss            %%xmm3, (%%rdx,%%rsi)     \n\t"
	"vpermilps  $0x39, %%xmm3,  %%xmm2           \n\t"
	"vmovss            %%xmm2, (%%rdx,%%r12)     \n\t"
	"vpermilps  $0x39, %%xmm2,  %%xmm3           \n\t"
	"vmovss            %%xmm3, (%%rdx,%%r13)     \n\t"
	"                                            \n\t"
	"addq      %%rdi, %%rcx                      \n\t" // c += cs_c;
	"addq      %%rdi, %%rdx                      \n\t" // c += cs_c;
	"                                            \n\t"
	"                                            \n\t"
	"                                            \n\t" // update c02:c72
	"vmovlps    (%%rcx),        %%xmm0,  %%xmm0  \n\t"
	"vmovhps    (%%rcx,%%rsi),  %%xmm0,  %%xmm0  \n\t"
	"vmovlps    (%%rcx,%%r12),  %%xmm1,  %%xmm1  \n\t"
	"vmovhps    (%%rcx,%%r13),  %%xmm1,  %%xmm1  \n\t"
	"vshufps    $0x88, %%xmm1,  %%xmm0,  %%xmm0  \n\t"
	"vmovlps    (%%rdx),        %%xmm2,  %%xmm2  \n\t"
	"vmovhps    (%%rdx,%%rsi),  %%xmm2,  %%xmm2  \n\t"
	"vmovlps    (%%rdx,%%r12),  %%xmm3,  %%xmm3  \n\t"
	"vmovhps    (%%rdx,%%r13),  %%xmm3,  %%xmm3  \n\t"
	"vshufps    $0x88, %%xmm3,  %%xmm2,  %%xmm2  \n\t"
	"vperm2f128 $0x20, %%ymm2,  %%ymm0,  %%ymm0  \n\t"
	"                                            \n\t"
	"vmulps            %%ymm4,  %%ymm0,  %%ymm0  \n\t" // scale by beta,
	"vaddps            %%ymm13, %%ymm0,  %%ymm0  \n\t" // add the gemm result,
	"                                            \n\t"
	"vextractf128  $1, %%ymm0,  %%xmm2           \n\t"
	"vmovss            %%xmm0, (%%rcx)           \n\t"
	"vpermilps  $0x39, %%xmm0,  %%xmm1           \n\t"
	"vmovss            %%xmm1, (%%rcx,%%rsi)     \n\t"
	"vpermilps  $0x39, %%xmm1,  %%xmm0           \n\t"
	"vmovss            %%xmm0, (%%rcx,%%r12)     \n\t"
	"vpermilps  $0x39, %%xmm0,  %%xmm1           \n\t"
	"vmovss            %%xmm1, (%%rcx,%%r13)     \n\t"
	"vmovss            %%xmm2, (%%rdx)           \n\t"
	"vpermilps  $0x39, %%xmm2,  %%xmm3           \n\t"
	"vmovss            %%xmm3, (%%rdx,%%rsi)     \n\t"
	"vpermilps  $0x39, %%xmm3,  %%xmm2           \n\t"
	"vmovss            %%xmm2, (%%rdx,%%r12)     \n\t"
	"vpermilps  $0x39, %%xmm2,  %%xmm3           \n\t"
	"vmovss            %%xmm3, (%%rdx,%%r13)     \n\t"
	"                                            \n\t"
	"addq      %%rdi, %%rcx                      \n\t" // c += cs_c;
	"addq      %%rdi, %%rdx                      \n\t" // c += cs_c;
	"                                            \n\t"
	"                                            \n\t"
	"                                            \n\t" // update c03:c73
	"vmovlps    (%%rcx),        %%xmm0,  %%xmm0  \n\t"
	"vmovhps    (%%rcx,%%rsi),  %%xmm0,  %%xmm0  \n\t"
	"vmovlps    (%%rcx,%%r12),  %%xmm1,  %%xmm1  \n\t"
	"vmovhps    (%%rcx,%%r13),  %%xmm1,  %%xmm1  \n\t"
	"vshufps    $0x88, %%xmm1,  %%xmm0,  %%xmm0  \n\t"
	"vmovlps    (%%rdx),        %%xmm2,  %%xmm2  \n\t"
	"vmovhps    (%%rdx,%%rsi),  %%xmm2,  %%xmm2  \n\t"
	"vmovlps    (%%rdx,%%r12),  %%xmm3,  %%xmm3  \n\t"
	"vmovhps    (%%rdx,%%r13),  %%xmm3,  %%xmm3  \n\t"
	"vshufps    $0x88, %%xmm3,  %%xmm2,  %%xmm2  \n\t"
	"vperm2f128 $0x20, %%ymm2,  %%ymm0,  %%ymm0  \n\t"
	"                                            \n\t"
	"vmulps            %%ymm4,  %%ymm0,  %%ymm0  \n\t" // scale by beta,
	"vaddps            %%ymm12, %%ymm0,  %%ymm0  \n\t" // add the gemm result,
	"                                            \n\t"
	"vextractf128  $1, %%ymm0,  %%xmm2           \n\t"
	"vmovss            %%xmm0, (%%rcx)           \n\t"
	"vpermilps  $0x39, %%xmm0,  %%xmm1           \n\t"
	"vmovss            %%xmm1, (%%rcx,%%rsi)     \n\t"
	"vpermilps  $0x39, %%xmm1,  %%xmm0           \n\t"
	"vmovss            %%xmm0, (%%rcx,%%r12)     \n\t"
	"vpermilps  $0x39, %%xmm0,  %%xmm1           \n\t"
	"vmovss            %%xmm1, (%%rcx,%%r13)     \n\t"
	"vmovss            %%xmm2, (%%rdx)           \n\t"
	"vpermilps  $0x39, %%xmm2,  %%xmm3           \n\t"
	"vmovss            %%xmm3, (%%rdx,%%rsi)     \n\t"
	"vpermilps  $0x39, %%xmm3,  %%xmm2           \n\t"
	"vmovss            %%xmm2, (%%rdx,%%r12)     \n\t"
	"vpermilps  $0x39, %%xmm2,  %%xmm3           \n\t"
	"vmovss            %%xmm3, (%%rdx,%%r13)     \n\t"
	"                                            \n\t"
	"addq      %%rdi, %%rcx                      \n\t" // c += cs_c;
	"addq      %%rdi, %%rdx                      \n\t" // c += cs_c;
	"                                            \n\t"
	"                                            \n\t"
	"                                            \n\t" // update c04:c74
	"vmovlps    (%%rcx),        %%xmm0,  %%xmm0  \n\t"
	"vmovhps    (%%rcx,%%rsi),  %%xmm0,  %%xmm0  \n\t"
	"vmovlps    (%%rcx,%%r12),  %%xmm1,  %%xmm1  \n\t"
	"vmovhps    (%%rcx,%%r13),  %%xmm1,  %%xmm1  \n\t"
	"vshufps    $0x88, %%xmm1,  %%xmm0,  %%xmm0  \n\t"
	"vmovlps    (%%rdx),        %%xmm2,  %%xmm2  \n\t"
	"vmovhps    (%%rdx,%%rsi),  %%xmm2,  %%xmm2  \n\t"
	"vmovlps    (%%rdx,%%r12),  %%xmm3,  %%xmm3  \n\t"
	"vmovhps    (%%rdx,%%r13),  %%xmm3,  %%xmm3  \n\t"
	"vshufps    $0x88, %%xmm3,  %%xmm2,  %%xmm2  \n\t"
	"vperm2f128 $0x20, %%ymm2,  %%ymm0,  %%ymm0  \n\t"
	"                                            \n\t"
	"vmulps            %%ymm4,  %%ymm0,  %%ymm0  \n\t" // scale by beta,
	"vaddps            %%ymm11, %%ymm0,  %%ymm0  \n\t" // add the gemm result,
	"                                            \n\t"
	"vextractf128  $1, %%ymm0,  %%xmm2           \n\t"
	"vmovss            %%xmm0, (%%rcx)           \n\t"
	"vpermilps  $0x39, %%xmm0,  %%xmm1           \n\t"
	"vmovss            %%xmm1, (%%rcx,%%rsi)     \n\t"
	"vpermilps  $0x39, %%xmm1,  %%xmm0           \n\t"
	"vmovss            %%xmm0, (%%rcx,%%r12)     \n\t"
	"vpermilps  $0x39, %%xmm0,  %%xmm1           \n\t"
	"vmovss            %%xmm1, (%%rcx,%%r13)     \n\t"
	"vmovss            %%xmm2, (%%rdx)           \n\t"
	"vpermilps  $0x39, %%xmm2,  %%xmm3           \n\t"
	"vmovss            %%xmm3, (%%rdx,%%rsi)     \n\t"
	"vpermilps  $0x39, %%xmm3,  %%xmm2           \n\t"
	"vmovss            %%xmm2, (%%rdx,%%r12)     \n\t"
	"vpermilps  $0x39, %%xmm2,  %%xmm3           \n\t"
	"vmovss            %%xmm3, (%%rdx,%%r13)     \n\t"
	"                                            \n\t"
	"addq      %%rdi, %%rcx                      \n\t" // c += cs_c;
	"addq      %%rdi, %%rdx                      \n\t" // c += cs_c;
	"                                            \n\t"
	"                                            \n\t"
	"                                            \n\t" // update c05:c75
	"vmovlps    (%%rcx),        %%xmm0,  %%xmm0  \n\t"
	"vmovhps    (%%rcx,%%rsi),  %%xmm0,  %%xmm0  \n\t"
	"vmovlps    (%%rcx,%%r12),  %%xmm1,  %%xmm1  \n\t"
	"vmovhps    (%%rcx,%%r13),  %%xmm1,  %%xmm1  \n\t"
	"vshufps    $0x88, %%xmm1,  %%xmm0,  %%xmm0  \n\t"
	"vmovlps    (%%rdx),        %%xmm2,  %%xmm2  \n\t"
	"vmovhps    (%%rdx,%%rsi),  %%xmm2,  %%xmm2  \n\t"
	"vmovlps    (%%rdx,%%r12),  %%xmm3,  %%xmm3  \n\t"
	"vmovhps    (%%rdx,%%r13),  %%xmm3,  %%xmm3  \n\t"
	"vshufps    $0x88, %%xmm3,  %%xmm2,  %%xmm2  \n\t"
	"vperm2f128 $0x20, %%ymm2,  %%ymm0,  %%ymm0  \n\t"
	"                                            \n\t"
	"vmulps            %%ymm4,  %%ymm0,  %%ymm0  \n\t" // scale by beta,
	"vaddps            %%ymm10, %%ymm0,  %%ymm0  \n\t" // add the gemm result,
	"                                            \n\t"
	"vextractf128  $1, %%ymm0,  %%xmm2           \n\t"
	"vmovss            %%xmm0, (%%rcx)           \n\t"
	"vpermilps  $0x39, %%xmm0,  %%xmm1           \n\t"
	"vmovss            %%xmm1, (%%rcx,%%rsi)     \n\t"
	"vpermilps  $0x39, %%xmm1,  %%xmm0           \n\t"
	"vmovss            %%xmm0, (%%rcx,%%r12)     \n\t"
	"vpermilps  $0x39, %%xmm0,  %%xmm1           \n\t"
	"vmovss            %%xmm1, (%%rcx,%%r13)     \n\t"
	"vmovss            %%xmm2, (%%rdx)           \n\t"
	"vpermilps  $0x39, %%xmm2,  %%xmm3           \n\t"
	"vmovss            %%xmm3, (%%rdx,%%rsi)     \n\t"
	"vpermilps  $0x39, %%xmm3,  %%xmm2           \n\t"
	"vmovss            %%xmm2, (%%rdx,%%r12)     \n\t"
	"vpermilps  $0x39, %%xmm2,  %%xmm3           \n\t"
	"vmovss            %%xmm3, (%%rdx,%%r13)     \n\t"
	"                                            \n\t"
	"addq      %%rdi, %%rcx                      \n\t" // c += cs_c;
	"addq      %%rdi, %%rdx                      \n\t" // c += cs_c;
	"                                            \n\t"
	"                                            \n\t"
	"                                            \n\t" // update c06:c76
	"vmovlps    (%%rcx),        %%xmm0,  %%xmm0  \n\t"
	"vmovhps    (%%rcx,%%rsi),  %%xmm0,  %%xmm0  \n\t"
	"vmovlps    (%%rcx,%%r12),  %%xmm1,  %%xmm1  \n\t"
	"vmovhps    (%%rcx,%%r13),  %%xmm1,  %%xmm1  \n\t"
	"vshufps    $0x88, %%xmm1,  %%xmm0,  %%xmm0  \n\t"
	"vmovlps    (%%rdx),        %%xmm2,  %%xmm2  \n\t"
	"vmovhps    (%%rdx,%%rsi),  %%xmm2,  %%xmm2  \n\t"
	"vmovlps    (%%rdx,%%r12),  %%xmm3,  %%xmm3  \n\t"
	"vmovhps    (%%rdx,%%r13),  %%xmm3,  %%xmm3  \n\t"
	"vshufps    $0x88, %%xmm3,  %%xmm2,  %%xmm2  \n\t"
	"vperm2f128 $0x20, %%ymm2,  %%ymm0,  %%ymm0  \n\t"
	"                                            \n\t"
	"vmulps            %%ymm4,  %%ymm0,  %%ymm0  \n\t" // scale by beta,
	"vaddps            %%ymm9,  %%ymm0,  %%ymm0  \n\t" // add the gemm result,
	"                                            \n\t"
	"vextractf128  $1, %%ymm0,  %%xmm2           \n\t"
	"vmovss            %%xmm0, (%%rcx)           \n\t"
	"vpermilps  $0x39, %%xmm0,  %%xmm1           \n\t"
	"vmovss            %%xmm1, (%%rcx,%%rsi)     \n\t"
	"vpermilps  $0x39, %%xmm1,  %%xmm0           \n\t"
	"vmovss            %%xmm0, (%%rcx,%%r12)     \n\t"
	"vpermilps  $0x39, %%xmm0,  %%xmm1           \n\t"
	"vmovss            %%xmm1, (%%rcx,%%r13)     \n\t"
	"vmovss            %%xmm2, (%%rdx)           \n\t"
	"vpermilps  $0x39, %%xmm2,  %%xmm3           \n\t"
	"vmovss            %%xmm3, (%%rdx,%%rsi)     \n\t"
	"vpermilps  $0x39, %%xmm3,  %%xmm2           \n\t"
	"vmovss            %%xmm2, (%%rdx,%%r12)     \n\t"
	"vpermilps  $0x39, %%xmm2,  %%xmm3           \n\t"
	"vmovss            %%xmm3, (%%rdx,%%r13)     \n\t"
	"                                            \n\t"
	"addq      %%rdi, %%rcx                      \n\t" // c += cs_c;
	"addq      %%rdi, %%rdx                      \n\t" // c += cs_c;
	"                                            \n\t"
	"                                            \n\t"
	"                                            \n\t" // update c07:c77
	"vmovlps    (%%rcx),        %%xmm0,  %%xmm0  \n\t"
	"vmovhps    (%%rcx,%%rsi),  %%xmm0,  %%xmm0  \n\t"
	"vmovlps    (%%rcx,%%r12),  %%xmm1,  %%xmm1  \n\t"
	"vmovhps    (%%rcx,%%r13),  %%xmm1,  %%xmm1  \n\t"
	"vshufps    $0x88, %%xmm1,  %%xmm0,  %%xmm0  \n\t"
	"vmovlps    (%%rdx),        %%xmm2,  %%xmm2  \n\t"
	"vmovhps    (%%rdx,%%rsi),  %%xmm2,  %%xmm2  \n\t"
	"vmovlps    (%%rdx,%%r12),  %%xmm3,  %%xmm3  \n\t"
	"vmovhps    (%%rdx,%%r13),  %%xmm3,  %%xmm3  \n\t"
	"vshufps    $0x88, %%xmm3,  %%xmm2,  %%xmm2  \n\t"
	"vperm2f128 $0x20, %%ymm2,  %%ymm0,  %%ymm0  \n\t"
	"                                            \n\t"
	"vmulps            %%ymm4,  %%ymm0,  %%ymm0  \n\t" // scale by beta,
	"vaddps            %%ymm8,  %%ymm0,  %%ymm0  \n\t" // add the gemm result,
	"                                            \n\t"
	"vextractf128  $1, %%ymm0,  %%xmm2           \n\t"
	"vmovss            %%xmm0, (%%rcx)           \n\t"
	"vpermilps  $0x39, %%xmm0,  %%xmm1           \n\t"
	"vmovss            %%xmm1, (%%rcx,%%rsi)     \n\t"
	"vpermilps  $0x39, %%xmm1,  %%xmm0           \n\t"
	"vmovss            %%xmm0, (%%rcx,%%r12)     \n\t"
	"vpermilps  $0x39, %%xmm0,  %%xmm1           \n\t"
	"vmovss            %%xmm1, (%%rcx,%%r13)     \n\t"
	"vmovss            %%xmm2, (%%rdx)           \n\t"
	"vpermilps  $0x39, %%xmm2,  %%xmm3           \n\t"
	"vmovss            %%xmm3, (%%rdx,%%rsi)     \n\t"
	"vpermilps  $0x39, %%xmm3,  %%xmm2           \n\t"
	"vmovss            %%xmm2, (%%rdx,%%r12)     \n\t"
	"vpermilps  $0x39, %%xmm2,  %%xmm3           \n\t"
	"vmovss            %%xmm3, (%%rdx,%%r13)     \n\t"
	"                                            \n\t"
	"                                            \n\t"
	"                                            \n\t"
	"jmp    .SDONE                               \n\t" // jump to end.
	"                                            \n\t"
	"                                            \n\t"
	"                                            \n\t"
	".SCOLSTORED:                                \n\t"
	"                                            \n\t"
	"                                            \n\t"
	"vmovaps    (%%rcx),       %%ymm0            \n\t" // load c00:c70,
	"vmulps           %%ymm4,  %%ymm0,  %%ymm0   \n\t" // scale by beta,
	"vaddps           %%ymm15, %%ymm0,  %%ymm0   \n\t" // add the gemm result,
	"vmovaps          %%ymm0,  (%%rcx)           \n\t" // and store back to memory.
	"addq      %%rdi, %%rcx                      \n\t" // c += cs_c;
	"                                            \n\t"
	"vmovaps    (%%rcx),       %%ymm1            \n\t" // load c01:c71,
	"vmulps           %%ymm4,  %%ymm1,  %%ymm1   \n\t" // scale by beta,
	"vaddps           %%ymm14, %%ymm1,  %%ymm1   \n\t" // add the gemm result,
	"vmovaps          %%ymm1,  (%%rcx)           \n\t" // and store back to memory.
	"addq      %%rdi, %%rcx                      \n\t" // c += cs_c;
	"                                            \n\t"
	"vmovaps    (%%rcx),       %%ymm0            \n\t" // load c02:c72,
	"vmulps           %%ymm4,  %%ymm0,  %%ymm0   \n\t" // scale by beta,
	"vaddps           %%ymm13, %%ymm0,  %%ymm0   \n\t" // add the gemm result,
	"vmovaps          %%ymm0,  (%%rcx)           \n\t" // and store back to memory.
	"addq      %%rdi, %%rcx                      \n\t" // c += cs_c;
	"                                            \n\t"
	"vmovaps    (%%rcx),       %%ymm1            \n\t" // load c03:c73,
	"vmulps           %%ymm4,  %%ymm1,  %%ymm1   \n\t" // scale by beta,
	"vaddps           %%ymm12, %%ymm1,  %%ymm1   \n\t" // add the gemm result,
	"vmovaps          %%ymm1,  (%%rcx)           \n\t" // and store back to memory.
	"addq      %%rdi, %%rcx                      \n\t" // c += cs_c;
	"                                            \n\t"
	"vmovaps    (%%rcx),       %%ymm0            \n\t" // load c04:c74,
	"vmulps           %%ymm4,  %%ymm0,  %%ymm0   \n\t" // scale by beta,
	"vaddps           %%ymm11, %%ymm0,  %%ymm0   \n\t" // add the gemm result,
	"vmovaps          %%ymm0,  (%%rcx)           \n\t" // and store back to memory.
	"addq      %%rdi, %%rcx                      \n\t" // c += cs_c;
	"                                            \n\t"
	"vmovaps    (%%rcx),       %%ymm1            \n\t" // load c05:c75,
	"vmulps           %%ymm4,  %%ymm1,  %%ymm1   \n\t" // scale by beta,
	"vaddps           %%ymm10, %%ymm1,  %%ymm1   \n\t" // add the gemm result,
	"vmovaps          %%ymm1,  (%%rcx)           \n\t" // and store back to memory.
	"addq      %%rdi, %%rcx                      \n\t" // c += cs_c;
	"                                            \n\t"
	"vmovaps    (%%rcx),       %%ymm0            \n\t" // load c06:c76,
	"vmulps           %%ymm4,  %%ymm0,  %%ymm0   \n\t" // scale by beta,
	"vaddps           %%ymm9,  %%ymm0,  %%ymm0   \n\t" // add the gemm result,
	"vmovaps          %%ymm0,  (%%rcx)           \n\t" // and store back to memory.
	"addq      %%rdi, %%rcx                      \n\t" // c += cs_c;
	"                                            \n\t"
	"vmovaps    (%%rcx),       %%ymm1            \n\t" // load c07:c77,
	"vmulps           %%ymm4,  %%ymm1,  %%ymm1   \n\t" // scale by beta,
	"vaddps           %%ymm8,  %%ymm1,  %%ymm1   \n\t" // add the gemm result,
	"vmovaps          %%ymm1,  (%%rcx)           \n\t" // and store back to memory.
	"                                            \n\t"
	"                                            \n\t"
	"jmp    .SDONE                               \n\t" // jump to end.
	"                                            \n\t"
	"                                            \n\t"
	"                                            \n\t"
	"                                            \n\t"
	".SBETAZERO:                                 \n\t"
	"                                            \n\t" // check if aligned/column-stored
	"andb     %%bl, %%bh                         \n\t" // set ZF if bl & bh == 1.
	"andb     %%bh, %%al                         \n\t" // set ZF if bh & al == 1.
	"jne     .SCOLSTORBZ                         \n\t" // jump to column storage case
	"                                            \n\t"
	"                                            \n\t"
	"                                            \n\t"
	".SGENSTORBZ:                                \n\t"
	"                                            \n\t"
	"                                            \n\t" // update c00:c70
	"vmovapd           %%ymm15, %%ymm0           \n\t"
	"vextractf128  $1, %%ymm0,  %%xmm2           \n\t"
	"vmovss            %%xmm0, (%%rcx)           \n\t"
	"vpermilps  $0x39, %%xmm0,  %%xmm1           \n\t"
	"vmovss            %%xmm1, (%%rcx,%%rsi)     \n\t"
	"vpermilps  $0x39, %%xmm1,  %%xmm0           \n\t"
	"vmovss            %%xmm0, (%%rcx,%%r12)     \n\t"
	"vpermilps  $0x39, %%xmm0,  %%xmm1           \n\t"
	"vmovss            %%xmm1, (%%rcx,%%r13)     \n\t"
	"vmovss            %%xmm2, (%%rdx)           \n\t"
	"vpermilps  $0x39, %%xmm2,  %%xmm3           \n\t"
	"vmovss            %%xmm3, (%%rdx,%%rsi)     \n\t"
	"vpermilps  $0x39, %%xmm3,  %%xmm2           \n\t"
	"vmovss            %%xmm2, (%%rdx,%%r12)     \n\t"
	"vpermilps  $0x39, %%xmm2,  %%xmm3           \n\t"
	"vmovss            %%xmm3, (%%rdx,%%r13)     \n\t"
	"                                            \n\t"
	"addq      %%rdi, %%rcx                      \n\t" // c += cs_c;
	"addq      %%rdi, %%rdx                      \n\t" // c += cs_c;
	"                                            \n\t"
	"                                            \n\t"
	"                                            \n\t" // update c01:c71
	"vmovapd           %%ymm14, %%ymm0           \n\t"
	"vextractf128  $1, %%ymm0,  %%xmm2           \n\t"
	"vmovss            %%xmm0, (%%rcx)           \n\t"
	"vpermilps  $0x39, %%xmm0,  %%xmm1           \n\t"
	"vmovss            %%xmm1, (%%rcx,%%rsi)     \n\t"
	"vpermilps  $0x39, %%xmm1,  %%xmm0           \n\t"
	"vmovss            %%xmm0, (%%rcx,%%r12)     \n\t"
	"vpermilps  $0x39, %%xmm0,  %%xmm1           \n\t"
	"vmovss            %%xmm1, (%%rcx,%%r13)     \n\t"
	"vmovss            %%xmm2, (%%rdx)           \n\t"
	"vpermilps  $0x39, %%xmm2,  %%xmm3           \n\t"
	"vmovss            %%xmm3, (%%rdx,%%rsi)     \n\t"
	"vpermilps  $0x39, %%xmm3,  %%xmm2           \n\t"
	"vmovss            %%xmm2, (%%rdx,%%r12)     \n\t"
	"vpermilps  $0x39, %%xmm2,  %%xmm3           \n\t"
	"vmovss            %%xmm3, (%%rdx,%%r13)     \n\t"
	"                                            \n\t"
	"addq      %%rdi, %%rcx                      \n\t" // c += cs_c;
	"addq      %%rdi, %%rdx                      \n\t" // c += cs_c;
	"                                            \n\t"
	"                                            \n\t"
	"                                            \n\t" // update c02:c72
	"vmovapd           %%ymm13, %%ymm0           \n\t"
	"vextractf128  $1, %%ymm0,  %%xmm2           \n\t"
	"vmovss            %%xmm0, (%%rcx)           \n\t"
	"vpermilps  $0x39, %%xmm0,  %%xmm1           \n\t"
	"vmovss            %%xmm1, (%%rcx,%%rsi)     \n\t"
	"vpermilps  $0x39, %%xmm1,  %%xmm0           \n\t"
	"vmovss            %%xmm0, (%%rcx,%%r12)     \n\t"
	"vpermilps  $0x39, %%xmm0,  %%xmm1           \n\t"
	"vmovss            %%xmm1, (%%rcx,%%r13)     \n\t"
	"vmovss            %%xmm2, (%%rdx)           \n\t"
	"vpermilps  $0x39, %%xmm2,  %%xmm3           \n\t"
	"vmovss            %%xmm3, (%%rdx,%%rsi)     \n\t"
	"vpermilps  $0x39, %%xmm3,  %%xmm2           \n\t"
	"vmovss            %%xmm2, (%%rdx,%%r12)     \n\t"
	"vpermilps  $0x39, %%xmm2,  %%xmm3           \n\t"
	"vmovss            %%xmm3, (%%rdx,%%r13)     \n\t"
	"                                            \n\t"
	"addq      %%rdi, %%rcx                      \n\t" // c += cs_c;
	"addq      %%rdi, %%rdx                      \n\t" // c += cs_c;
	"                                            \n\t"
	"                                            \n\t"
	"                                            \n\t" // update c03:c73
	"vmovapd           %%ymm12, %%ymm0           \n\t"
	"vextractf128  $1, %%ymm0,  %%xmm2           \n\t"
	"vmovss            %%xmm0, (%%rcx)           \n\t"
	"vpermilps  $0x39, %%xmm0,  %%xmm1           \n\t"
	"vmovss            %%xmm1, (%%rcx,%%rsi)     \n\t"
	"vpermilps  $0x39, %%xmm1,  %%xmm0           \n\t"
	"vmovss            %%xmm0, (%%rcx,%%r12)     \n\t"
	"vpermilps  $0x39, %%xmm0,  %%xmm1           \n\t"
	"vmovss            %%xmm1, (%%rcx,%%r13)     \n\t"
	"vmovss            %%xmm2, (%%rdx)           \n\t"
	"vpermilps  $0x39, %%xmm2,  %%xmm3           \n\t"
	"vmovss            %%xmm3, (%%rdx,%%rsi)     \n\t"
	"vpermilps  $0x39, %%xmm3,  %%xmm2           \n\t"
	"vmovss            %%xmm2, (%%rdx,%%r12)     \n\t"
	"vpermilps  $0x39, %%xmm2,  %%xmm3           \n\t"
	"vmovss            %%xmm3, (%%rdx,%%r13)     \n\t"
	"                                            \n\t"
	"addq      %%rdi, %%rcx                      \n\t" // c += cs_c;
	"addq      %%rdi, %%rdx                      \n\t" // c += cs_c;
	"                                            \n\t"
	"                                            \n\t"
	"                                            \n\t" // update c04:c74
	"vmovapd           %%ymm11, %%ymm0           \n\t"
	"vextractf128  $1, %%ymm0,  %%xmm2           \n\t"
	"vmovss            %%xmm0, (%%rcx)           \n\t"
	"vpermilps  $0x39, %%xmm0,  %%xmm1           \n\t"
	"vmovss            %%xmm1, (%%rcx,%%rsi)     \n\t"
	"vpermilps  $0x39, %%xmm1,  %%xmm0           \n\t"
	"vmovss            %%xmm0, (%%rcx,%%r12)     \n\t"
	"vpermilps  $0x39, %%xmm0,  %%xmm1           \n\t"
	"vmovss            %%xmm1, (%%rcx,%%r13)     \n\t"
	"vmovss            %%xmm2, (%%rdx)           \n\t"
	"vpermilps  $0x39, %%xmm2,  %%xmm3           \n\t"
	"vmovss            %%xmm3, (%%rdx,%%rsi)     \n\t"
	"vpermilps  $0x39, %%xmm3,  %%xmm2           \n\t"
	"vmovss            %%xmm2, (%%rdx,%%r12)     \n\t"
	"vpermilps  $0x39, %%xmm2,  %%xmm3           \n\t"
	"vmovss            %%xmm3, (%%rdx,%%r13)     \n\t"
	"                                            \n\t"
	"addq      %%rdi, %%rcx                      \n\t" // c += cs_c;
	"addq      %%rdi, %%rdx                      \n\t" // c += cs_c;
	"                                            \n\t"
	"                                            \n\t"
	"                                            \n\t" // update c05:c75
	"vmovapd           %%ymm10, %%ymm0           \n\t"
	"vextractf128  $1, %%ymm0,  %%xmm2           \n\t"
	"vmovss            %%xmm0, (%%rcx)           \n\t"
	"vpermilps  $0x39, %%xmm0,  %%xmm1           \n\t"
	"vmovss            %%xmm1, (%%rcx,%%rsi)     \n\t"
	"vpermilps  $0x39, %%xmm1,  %%xmm0           \n\t"
	"vmovss            %%xmm0, (%%rcx,%%r12)     \n\t"
	"vpermilps  $0x39, %%xmm0,  %%xmm1           \n\t"
	"vmovss            %%xmm1, (%%rcx,%%r13)     \n\t"
	"vmovss            %%xmm2, (%%rdx)           \n\t"
	"vpermilps  $0x39, %%xmm2,  %%xmm3           \n\t"
	"vmovss            %%xmm3, (%%rdx,%%rsi)     \n\t"
	"vpermilps  $0x39, %%xmm3,  %%xmm2           \n\t"
	"vmovss            %%xmm2, (%%rdx,%%r12)     \n\t"
	"vpermilps  $0x39, %%xmm2,  %%xmm3           \n\t"
	"vmovss            %%xmm3, (%%rdx,%%r13)     \n\t"
	"                                            \n\t"
	"addq      %%rdi, %%rcx                      \n\t" // c += cs_c;
	"addq      %%rdi, %%rdx                      \n\t" // c += cs_c;
	"                                            \n\t"
	"                                            \n\t"
	"                                            \n\t" // update c06:c76
	"vmovapd           %%ymm9,  %%ymm0           \n\t"
	"vextractf128  $1, %%ymm0,  %%xmm2           \n\t"
	"vmovss            %%xmm0, (%%rcx)           \n\t"
	"vpermilps  $0x39, %%xmm0,  %%xmm1           \n\t"
	"vmovss            %%xmm1, (%%rcx,%%rsi)     \n\t"
	"vpermilps  $0x39, %%xmm1,  %%xmm0           \n\t"
	"vmovss            %%xmm0, (%%rcx,%%r12)     \n\t"
	"vpermilps  $0x39, %%xmm0,  %%xmm1           \n\t"
	"vmovss            %%xmm1, (%%rcx,%%r13)     \n\t"
	"vmovss            %%xmm2, (%%rdx)           \n\t"
	"vpermilps  $0x39, %%xmm2,  %%xmm3           \n\t"
	"vmovss            %%xmm3, (%%rdx,%%rsi)     \n\t"
	"vpermilps  $0x39, %%xmm3,  %%xmm2           \n\t"
	"vmovss            %%xmm2, (%%rdx,%%r12)     \n\t"
	"vpermilps  $0x39, %%xmm2,  %%xmm3           \n\t"
	"vmovss            %%xmm3, (%%rdx,%%r13)     \n\t"
	"                                            \n\t"
	"addq      %%rdi, %%rcx                      \n\t" // c += cs_c;
	"addq      %%rdi, %%rdx                      \n\t" // c += cs_c;
	"                                            \n\t"
	"                                            \n\t"
	"                                            \n\t" // update c07:c77
	"vmovapd           %%ymm8,  %%ymm0           \n\t"
	"vextractf128  $1, %%ymm0,  %%xmm2           \n\t"
	"vmovss            %%xmm0, (%%rcx)           \n\t"
	"vpermilps  $0x39, %%xmm0,  %%xmm1           \n\t"
	"vmovss            %%xmm1, (%%rcx,%%rsi)     \n\t"
	"vpermilps  $0x39, %%xmm1,  %%xmm0           \n\t"
	"vmovss            %%xmm0, (%%rcx,%%r12)     \n\t"
	"vpermilps  $0x39, %%xmm0,  %%xmm1           \n\t"
	"vmovss            %%xmm1, (%%rcx,%%r13)     \n\t"
	"vmovss            %%xmm2, (%%rdx)           \n\t"
	"vpermilps  $0x39, %%xmm2,  %%xmm3           \n\t"
	"vmovss            %%xmm3, (%%rdx,%%rsi)     \n\t"
	"vpermilps  $0x39, %%xmm3,  %%xmm2           \n\t"
	"vmovss            %%xmm2, (%%rdx,%%r12)     \n\t"
	"vpermilps  $0x39, %%xmm2,  %%xmm3           \n\t"
	"vmovss            %%xmm3, (%%rdx,%%r13)     \n\t"
	"                                            \n\t"
	"                                            \n\t"
	"jmp    .SDONE                               \n\t" // jump to end.
	"                                            \n\t"
	"                                            \n\t"
	"                                            \n\t"
	".SCOLSTORBZ:                                \n\t"
	"                                            \n\t"
	"                                            \n\t"
	"vmovaps          %%ymm15, (%%rcx)           \n\t" // and store back to memory.
	"addq      %%rdi, %%rcx                      \n\t" // c += cs_c;
	"                                            \n\t"
	"vmovaps          %%ymm14, (%%rcx)           \n\t" // and store back to memory.
	"addq      %%rdi, %%rcx                      \n\t" // c += cs_c;
	"                                            \n\t"
	"vmovaps          %%ymm13, (%%rcx)           \n\t" // and store back to memory.
	"addq      %%rdi, %%rcx                      \n\t" // c += cs_c;
	"                                            \n\t"
	"vmovaps          %%ymm12, (%%rcx)           \n\t" // and store back to memory.
	"addq      %%rdi, %%rcx                      \n\t" // c += cs_c;
	"                                            \n\t"
	"vmovaps          %%ymm11, (%%rcx)           \n\t" // and store back to memory.
	"addq      %%rdi, %%rcx                      \n\t" // c += cs_c;
	"                                            \n\t"
	"vmovaps          %%ymm10, (%%rcx)           \n\t" // and store back to memory.
	"addq      %%rdi, %%rcx                      \n\t" // c += cs_c;
	"                                            \n\t"
	"vmovaps          %%ymm9,  (%%rcx)           \n\t" // and store back to memory.
	"addq      %%rdi, %%rcx                      \n\t" // c += cs_c;
	"                                            \n\t"
	"vmovaps          %%ymm8,  (%%rcx)           \n\t" // and store back to memory.
	"                                            \n\t"
	"                                            \n\t"
	"                                            \n\t"
	"                                            \n\t"
	"                                            \n\t"
	".SDONE:                                     \n\t"
	"                                            \n\t"

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
	  "rax", "rbx", "rcx", "rdx", "rsi", "rdi", 
	  "r8", "r9", "r10", "r11", "r12", "r13", "r14", "r15",
	  "xmm0", "xmm1", "xmm2", "xmm3",
	  "xmm4", "xmm5", "xmm6", "xmm7",
	  "xmm8", "xmm9", "xmm10", "xmm11",
	  "xmm12", "xmm13", "xmm14", "xmm15",
	  "memory"
	);
}

void bli_dgemm_asm_8x4(
                        dim_t              k,
                        double* restrict   alpha,
                        double* restrict   a,
                        double* restrict   b,
                        double* restrict   beta,
                        double* restrict   c, inc_t rs_c, inc_t cs_c,
                        auxinfo_t*         data
                      )
{
	//void*   a_next = bli_auxinfo_next_a( data );
	void*   b_next = bli_auxinfo_next_b( data );

	dim_t   k_iter = k / 4;
	dim_t   k_left = k % 4;

	__asm__ volatile
	(
	"                                            \n\t"
	"                                            \n\t"
	"movq                %2, %%rax               \n\t" // load address of a.
	"movq                %3, %%rbx               \n\t" // load address of b.
	"movq                %9, %%r15               \n\t" // load address of b_next.
	//"movq               %10, %%r14               \n\t" // load address of a_next.
	"addq          $-4 * 64, %%r15               \n\t"
	"                                            \n\t"
	"vmovapd   0 * 32(%%rax), %%ymm0             \n\t" // initialize loop by pre-loading
	"vmovapd   0 * 32(%%rbx), %%ymm2             \n\t" // elements of a and b.
	"vpermilpd  $0x5, %%ymm2, %%ymm3             \n\t"
	"                                            \n\t"
	"movq                %6, %%rcx               \n\t" // load address of c
	"movq                %8, %%rdi               \n\t" // load cs_c
	"leaq        (,%%rdi,8), %%rdi               \n\t" // cs_c *= sizeof(double)
	"leaq   (%%rcx,%%rdi,2), %%r10               \n\t" // load address of c + 2*cs_c;
	"                                            \n\t"
	"prefetcht0   3 * 8(%%rcx)                   \n\t" // prefetch c + 0*cs_c
	"prefetcht0   3 * 8(%%rcx,%%rdi)             \n\t" // prefetch c + 1*cs_c
	"prefetcht0   3 * 8(%%r10)                   \n\t" // prefetch c + 2*cs_c
	"prefetcht0   3 * 8(%%r10,%%rdi)             \n\t" // prefetch c + 3*cs_c
	"                                            \n\t"
	"vxorpd    %%ymm8,  %%ymm8,  %%ymm8          \n\t"
	"vxorpd    %%ymm9,  %%ymm9,  %%ymm9          \n\t"
	"vxorpd    %%ymm10, %%ymm10, %%ymm10         \n\t"
	"vxorpd    %%ymm11, %%ymm11, %%ymm11         \n\t"
	"vxorpd    %%ymm12, %%ymm12, %%ymm12         \n\t"
	"vxorpd    %%ymm13, %%ymm13, %%ymm13         \n\t"
	"vxorpd    %%ymm14, %%ymm14, %%ymm14         \n\t"
	"vxorpd    %%ymm15, %%ymm15, %%ymm15         \n\t"
	"                                            \n\t"
	"                                            \n\t"
	"                                            \n\t"
	"movq      %0, %%rsi                         \n\t" // i = k_iter;
	"testq  %%rsi, %%rsi                         \n\t" // check i via logical AND.
	"je     .DCONSIDKLEFT                        \n\t" // if i == 0, jump to code that
	"                                            \n\t" // contains the k_left loop.
	"                                            \n\t"
	"                                            \n\t"
	".DLOOPKITER:                                \n\t" // MAIN LOOP
	"                                            \n\t"
	"addq         $4 * 4 * 8,  %%r15             \n\t" // b_next += 4*4 (unroll x nr)
	"                                            \n\t"
	"                                            \n\t" // iteration 0
	"vmovapd   1 * 32(%%rax),  %%ymm1            \n\t"
	"vmulpd           %%ymm0,  %%ymm2,  %%ymm6   \n\t"
	"vperm2f128 $0x3, %%ymm2,  %%ymm2,  %%ymm4   \n\t"
	"vmulpd           %%ymm0,  %%ymm3,  %%ymm7   \n\t"
	"vperm2f128 $0x3, %%ymm3,  %%ymm3,  %%ymm5   \n\t"
	"vaddpd           %%ymm15, %%ymm6,  %%ymm15  \n\t"
	"vaddpd           %%ymm13, %%ymm7,  %%ymm13  \n\t"
	"                                            \n\t"
	"prefetcht0  16 * 32(%%rax)                  \n\t"
	"vmulpd           %%ymm1,  %%ymm2,  %%ymm6   \n\t"
	"vmovapd   1 * 32(%%rbx),  %%ymm2            \n\t"
	"vmulpd           %%ymm1,  %%ymm3,  %%ymm7   \n\t"
	"vpermilpd  $0x5, %%ymm2,  %%ymm3            \n\t"
	"vaddpd           %%ymm14, %%ymm6,  %%ymm14  \n\t"
	"vaddpd           %%ymm12, %%ymm7,  %%ymm12  \n\t"
	"                                            \n\t"
	"vmulpd           %%ymm0,  %%ymm4,  %%ymm6   \n\t"
	"vmulpd           %%ymm0,  %%ymm5,  %%ymm7   \n\t"
	"vmovapd   2 * 32(%%rax),  %%ymm0            \n\t"
	"vaddpd           %%ymm11, %%ymm6,  %%ymm11  \n\t"
	"vaddpd           %%ymm9,  %%ymm7,  %%ymm9   \n\t"
	"prefetcht0   0 * 32(%%r15)                  \n\t" // prefetch b_next[0*4]
	"                                            \n\t"
	"vmulpd           %%ymm1,  %%ymm4,  %%ymm6   \n\t"
	"vmulpd           %%ymm1,  %%ymm5,  %%ymm7   \n\t"
	"vaddpd           %%ymm10, %%ymm6,  %%ymm10  \n\t"
	"vaddpd           %%ymm8,  %%ymm7,  %%ymm8   \n\t"
	"                                            \n\t"
	"                                            \n\t"
	"                                            \n\t" // iteration 1
	"vmovapd   3 * 32(%%rax),  %%ymm1            \n\t"
	"vmulpd           %%ymm0,  %%ymm2,  %%ymm6   \n\t"
	"vperm2f128 $0x3, %%ymm2,  %%ymm2,  %%ymm4   \n\t"
	"vmulpd           %%ymm0,  %%ymm3,  %%ymm7   \n\t"
	"vperm2f128 $0x3, %%ymm3,  %%ymm3,  %%ymm5   \n\t"
	"vaddpd           %%ymm15, %%ymm6,  %%ymm15  \n\t"
	"vaddpd           %%ymm13, %%ymm7,  %%ymm13  \n\t"
	"                                            \n\t"
	"prefetcht0  18 * 32(%%rax)                  \n\t"
	"vmulpd           %%ymm1,  %%ymm2,  %%ymm6   \n\t"
	"vmovapd   2 * 32(%%rbx),  %%ymm2            \n\t"
	"vmulpd           %%ymm1,  %%ymm3,  %%ymm7   \n\t"
	"vpermilpd  $0x5, %%ymm2,  %%ymm3            \n\t"
	"vaddpd           %%ymm14, %%ymm6,  %%ymm14  \n\t"
	"vaddpd           %%ymm12, %%ymm7,  %%ymm12  \n\t"
	"                                            \n\t"
	"vmulpd           %%ymm0,  %%ymm4,  %%ymm6   \n\t"
	"vmulpd           %%ymm0,  %%ymm5,  %%ymm7   \n\t"
	"vmovapd   4 * 32(%%rax),  %%ymm0            \n\t"
	"vaddpd           %%ymm11, %%ymm6,  %%ymm11  \n\t"
	"vaddpd           %%ymm9,  %%ymm7,  %%ymm9   \n\t"
	"                                            \n\t"
	"vmulpd           %%ymm1,  %%ymm4,  %%ymm6   \n\t"
	"vmulpd           %%ymm1,  %%ymm5,  %%ymm7   \n\t"
	"vaddpd           %%ymm10, %%ymm6,  %%ymm10  \n\t"
	"vaddpd           %%ymm8,  %%ymm7,  %%ymm8   \n\t"
	"                                            \n\t"
	"                                            \n\t"
	"                                            \n\t" // iteration 2
	"vmovapd   5 * 32(%%rax),  %%ymm1            \n\t"
	"vmulpd           %%ymm0,  %%ymm2,  %%ymm6   \n\t"
	"vperm2f128 $0x3, %%ymm2,  %%ymm2,  %%ymm4   \n\t"
	"vmulpd           %%ymm0,  %%ymm3,  %%ymm7   \n\t"
	"vperm2f128 $0x3, %%ymm3,  %%ymm3,  %%ymm5   \n\t"
	"vaddpd           %%ymm15, %%ymm6,  %%ymm15  \n\t"
	"vaddpd           %%ymm13, %%ymm7,  %%ymm13  \n\t"
	"                                            \n\t"
	"prefetcht0  20 * 32(%%rax)                  \n\t"
	"vmulpd           %%ymm1,  %%ymm2,  %%ymm6   \n\t"
	"vmovapd   3 * 32(%%rbx),  %%ymm2            \n\t"
	"addq         $4 * 4 * 8,  %%rbx             \n\t" // b += 4*4 (unroll x nr)
	"vmulpd           %%ymm1,  %%ymm3,  %%ymm7   \n\t"
	"vpermilpd  $0x5, %%ymm2,  %%ymm3            \n\t"
	"vaddpd           %%ymm14, %%ymm6,  %%ymm14  \n\t"
	"vaddpd           %%ymm12, %%ymm7,  %%ymm12  \n\t"
	"                                            \n\t"
	"vmulpd           %%ymm0,  %%ymm4,  %%ymm6   \n\t"
	"vmulpd           %%ymm0,  %%ymm5,  %%ymm7   \n\t"
	"vmovapd   6 * 32(%%rax),  %%ymm0            \n\t"
	"vaddpd           %%ymm11, %%ymm6,  %%ymm11  \n\t"
	"vaddpd           %%ymm9,  %%ymm7,  %%ymm9   \n\t"
	"prefetcht0   2 * 32(%%r15)                  \n\t" // prefetch b_next[2*4]
	"                                            \n\t"
	"vmulpd           %%ymm1,  %%ymm4,  %%ymm6   \n\t"
	"vmulpd           %%ymm1,  %%ymm5,  %%ymm7   \n\t"
	"vaddpd           %%ymm10, %%ymm6,  %%ymm10  \n\t"
	"vaddpd           %%ymm8,  %%ymm7,  %%ymm8   \n\t"
	"                                            \n\t"
	"                                            \n\t"
	"                                            \n\t" // iteration 3
	"vmovapd   7 * 32(%%rax),  %%ymm1            \n\t"
	"addq         $4 * 8 * 8,  %%rax             \n\t" // a += 4*8 (unroll x mr)
	"vmulpd           %%ymm0,  %%ymm2,  %%ymm6   \n\t"
	"vperm2f128 $0x3, %%ymm2,  %%ymm2,  %%ymm4   \n\t"
	"vmulpd           %%ymm0,  %%ymm3,  %%ymm7   \n\t"
	"vperm2f128 $0x3, %%ymm3,  %%ymm3,  %%ymm5   \n\t"
	"vaddpd           %%ymm15, %%ymm6,  %%ymm15  \n\t"
	"vaddpd           %%ymm13, %%ymm7,  %%ymm13  \n\t"
	"                                            \n\t"
	//"prefetcht0  22 * 32(%%rax)                  \n\t"
	"prefetcht0  14 * 32(%%rax)                  \n\t"
	"vmulpd           %%ymm1,  %%ymm2,  %%ymm6   \n\t"
	"vmovapd   0 * 32(%%rbx),  %%ymm2            \n\t"
	"vmulpd           %%ymm1,  %%ymm3,  %%ymm7   \n\t"
	"vpermilpd  $0x5, %%ymm2,  %%ymm3            \n\t"
	"vaddpd           %%ymm14, %%ymm6,  %%ymm14  \n\t"
	"vaddpd           %%ymm12, %%ymm7,  %%ymm12  \n\t"
	"                                            \n\t"
	"vmulpd           %%ymm0,  %%ymm4,  %%ymm6   \n\t"
	"vmulpd           %%ymm0,  %%ymm5,  %%ymm7   \n\t"
	"vmovapd   0 * 32(%%rax),  %%ymm0            \n\t"
	"vaddpd           %%ymm11, %%ymm6,  %%ymm11  \n\t"
	"vaddpd           %%ymm9,  %%ymm7,  %%ymm9   \n\t"
	"                                            \n\t"
	"vmulpd           %%ymm1,  %%ymm4,  %%ymm6   \n\t"
	"vmulpd           %%ymm1,  %%ymm5,  %%ymm7   \n\t"
	"vaddpd           %%ymm10, %%ymm6,  %%ymm10  \n\t"
	"vaddpd           %%ymm8,  %%ymm7,  %%ymm8   \n\t"
	"                                            \n\t"
	"                                            \n\t"
	"                                            \n\t"
	//"addq   $4 * 8 * 8, %%rax                    \n\t" // a      += 4*8 (unroll x mr)
	//"addq   $4 * 4 * 8, %%rbx                    \n\t" // b      += 4*4 (unroll x nr)
	"                                            \n\t"
	"decq   %%rsi                                \n\t" // i -= 1;
	"jne    .DLOOPKITER                          \n\t" // iterate again if i != 0.
	"                                            \n\t"
	"                                            \n\t"
	"                                            \n\t"
	"                                            \n\t"
	"                                            \n\t"
	"                                            \n\t"
	".DCONSIDKLEFT:                              \n\t"
	"                                            \n\t"
	"movq      %1, %%rsi                         \n\t" // i = k_left;
	"testq  %%rsi, %%rsi                         \n\t" // check i via logical AND.
	"je     .DPOSTACCUM                          \n\t" // if i == 0, we're done; jump to end.
	"                                            \n\t" // else, we prepare to enter k_left loop.
	"                                            \n\t"
	"                                            \n\t"
	".DLOOPKLEFT:                                \n\t" // EDGE LOOP
	"                                            \n\t"
	"vmovapd   1 * 32(%%rax),  %%ymm1            \n\t"
	"addq         $8 * 1 * 8,  %%rax             \n\t" // a += 8 (1 x mr)
	"vmulpd           %%ymm0,  %%ymm2, %%ymm6    \n\t"
	"vperm2f128 $0x3, %%ymm2,  %%ymm2, %%ymm4    \n\t"
	"vmulpd           %%ymm0,  %%ymm3, %%ymm7    \n\t"
	"vperm2f128 $0x3, %%ymm3,  %%ymm3, %%ymm5    \n\t"
	"vaddpd           %%ymm15, %%ymm6, %%ymm15   \n\t"
	"vaddpd           %%ymm13, %%ymm7, %%ymm13   \n\t"
	"                                            \n\t"
	"prefetcht0  14 * 32(%%rax)                  \n\t"
	"vmulpd           %%ymm1,  %%ymm2, %%ymm6    \n\t"
	"vmovapd   1 * 32(%%rbx),  %%ymm2            \n\t"
	"addq         $4 * 1 * 8,  %%rbx             \n\t" // b += 4 (1 x nr)
	"vmulpd           %%ymm1,  %%ymm3, %%ymm7    \n\t"
	"vpermilpd  $0x5, %%ymm2,  %%ymm3            \n\t"
	"vaddpd           %%ymm14, %%ymm6, %%ymm14   \n\t"
	"vaddpd           %%ymm12, %%ymm7, %%ymm12   \n\t"
	"                                            \n\t"
	"vmulpd           %%ymm0,  %%ymm4, %%ymm6    \n\t"
	"vmulpd           %%ymm0,  %%ymm5, %%ymm7    \n\t"
	"vmovapd   0 * 32(%%rax),  %%ymm0            \n\t"
	"vaddpd           %%ymm11, %%ymm6, %%ymm11   \n\t"
	"vaddpd           %%ymm9,  %%ymm7, %%ymm9    \n\t"
	"                                            \n\t"
	"vmulpd           %%ymm1,  %%ymm4, %%ymm6    \n\t"
	"vmulpd           %%ymm1,  %%ymm5, %%ymm7    \n\t"
	"vaddpd           %%ymm10, %%ymm6, %%ymm10   \n\t"
	"vaddpd           %%ymm8,  %%ymm7, %%ymm8    \n\t"
	"                                            \n\t"
	"                                            \n\t"
	"decq   %%rsi                                \n\t" // i -= 1;
	"jne    .DLOOPKLEFT                          \n\t" // iterate again if i != 0.
	"                                            \n\t"
	"                                            \n\t"
	"                                            \n\t"
	".DPOSTACCUM:                                \n\t"
	"                                            \n\t"
	"                                            \n\t"
	"                                            \n\t" // ymm15:  ymm13:  ymm11:  ymm9:
	"                                            \n\t" // ( ab00  ( ab01  ( ab02  ( ab03
	"                                            \n\t" //   ab11    ab10    ab13    ab12  
	"                                            \n\t" //   ab22    ab23    ab20    ab21
	"                                            \n\t" //   ab33 )  ab32 )  ab31 )  ab30 )
	"                                            \n\t"
	"                                            \n\t" // ymm14:  ymm12:  ymm10:  ymm8:
	"                                            \n\t" // ( ab40  ( ab41  ( ab42  ( ab43
	"                                            \n\t" //   ab51    ab50    ab53    ab52  
	"                                            \n\t" //   ab62    ab63    ab60    ab61
	"                                            \n\t" //   ab73 )  ab72 )  ab71 )  ab70 )
	"                                            \n\t"
	"vmovapd          %%ymm15, %%ymm7            \n\t"
	"vshufpd    $0xa, %%ymm15, %%ymm13, %%ymm15  \n\t"
	"vshufpd    $0xa, %%ymm13, %%ymm7,  %%ymm13  \n\t"
	"                                            \n\t"
	"vmovapd          %%ymm11, %%ymm7            \n\t"
	"vshufpd    $0xa, %%ymm11, %%ymm9,  %%ymm11  \n\t"
	"vshufpd    $0xa, %%ymm9,  %%ymm7,  %%ymm9   \n\t"
	"                                            \n\t"
	"vmovapd          %%ymm14, %%ymm7            \n\t"
	"vshufpd    $0xa, %%ymm14, %%ymm12, %%ymm14  \n\t"
	"vshufpd    $0xa, %%ymm12, %%ymm7,  %%ymm12  \n\t"
	"                                            \n\t"
	"vmovapd          %%ymm10, %%ymm7            \n\t"
	"vshufpd    $0xa, %%ymm10, %%ymm8,  %%ymm10  \n\t"
	"vshufpd    $0xa, %%ymm8,  %%ymm7,  %%ymm8   \n\t"
	"                                            \n\t"
	"                                            \n\t" // ymm15:  ymm13:  ymm11:  ymm9:
	"                                            \n\t" // ( ab01  ( ab00  ( ab03  ( ab02
	"                                            \n\t" //   ab11    ab10    ab13    ab12  
	"                                            \n\t" //   ab23    ab22    ab21    ab20
	"                                            \n\t" //   ab33 )  ab32 )  ab31 )  ab30 )
	"                                            \n\t"
	"                                            \n\t" // ymm14:  ymm12:  ymm10:  ymm8:
	"                                            \n\t" // ( ab41  ( ab40  ( ab43  ( ab42
	"                                            \n\t" //   ab51    ab50    ab53    ab52  
	"                                            \n\t" //   ab63    ab62    ab61    ab60
	"                                            \n\t" //   ab73 )  ab72 )  ab71 )  ab70 )
	"                                            \n\t"
	"vmovapd           %%ymm15, %%ymm7           \n\t"
	"vperm2f128 $0x30, %%ymm15, %%ymm11, %%ymm15 \n\t"
	"vperm2f128 $0x12, %%ymm7,  %%ymm11, %%ymm11 \n\t"
	"                                            \n\t"
	"vmovapd           %%ymm13, %%ymm7           \n\t"
	"vperm2f128 $0x30, %%ymm13, %%ymm9,  %%ymm13 \n\t"
	"vperm2f128 $0x12, %%ymm7,  %%ymm9,  %%ymm9  \n\t"
	"                                            \n\t"
	"vmovapd           %%ymm14, %%ymm7           \n\t"
	"vperm2f128 $0x30, %%ymm14, %%ymm10, %%ymm14 \n\t"
	"vperm2f128 $0x12, %%ymm7,  %%ymm10, %%ymm10 \n\t"
	"                                            \n\t"
	"vmovapd           %%ymm12, %%ymm7           \n\t"
	"vperm2f128 $0x30, %%ymm12, %%ymm8,  %%ymm12 \n\t"
	"vperm2f128 $0x12, %%ymm7,  %%ymm8,  %%ymm8  \n\t"
	"                                            \n\t"
	"                                            \n\t" // ymm9:   ymm11:  ymm13:  ymm15:
	"                                            \n\t" // ( ab00  ( ab01  ( ab02  ( ab03
	"                                            \n\t" //   ab10    ab11    ab12    ab13  
	"                                            \n\t" //   ab20    ab21    ab22    ab23
	"                                            \n\t" //   ab30 )  ab31 )  ab32 )  ab33 )
	"                                            \n\t"
	"                                            \n\t" // ymm8:   ymm10:  ymm12:  ymm14:
	"                                            \n\t" // ( ab40  ( ab41  ( ab42  ( ab43
	"                                            \n\t" //   ab50    ab51    ab52    ab53  
	"                                            \n\t" //   ab60    ab61    ab62    ab63
	"                                            \n\t" //   ab70 )  ab71 )  ab72 )  ab73 )
	"                                            \n\t"
	"                                            \n\t"
	"movq         %4, %%rax                      \n\t" // load address of alpha
	"movq         %5, %%rbx                      \n\t" // load address of beta 
	"vbroadcastsd    (%%rax), %%ymm0             \n\t" // load alpha and duplicate
	"vbroadcastsd    (%%rbx), %%ymm2             \n\t" // load beta and duplicate
	"                                            \n\t"
	"vmulpd           %%ymm0,  %%ymm8,  %%ymm8   \n\t" // scale by alpha
	"vmulpd           %%ymm0,  %%ymm9,  %%ymm9   \n\t"
	"vmulpd           %%ymm0,  %%ymm10, %%ymm10  \n\t"
	"vmulpd           %%ymm0,  %%ymm11, %%ymm11  \n\t"
	"vmulpd           %%ymm0,  %%ymm12, %%ymm12  \n\t"
	"vmulpd           %%ymm0,  %%ymm13, %%ymm13  \n\t"
	"vmulpd           %%ymm0,  %%ymm14, %%ymm14  \n\t"
	"vmulpd           %%ymm0,  %%ymm15, %%ymm15  \n\t"
	"                                            \n\t"
	"                                            \n\t"
	"                                            \n\t"
	"                                            \n\t"
	"                                            \n\t"
	"                                            \n\t"
	"movq                %7, %%rsi               \n\t" // load rs_c
	"leaq        (,%%rsi,8), %%rsi               \n\t" // rsi = rs_c * sizeof(double)
	"                                            \n\t"
	"leaq   (%%rcx,%%rsi,4), %%rdx               \n\t" // load address of c + 4*rs_c;
	"                                            \n\t"
	"leaq        (,%%rsi,2), %%r12               \n\t" // r12 = 2*rs_c;
	"leaq   (%%r12,%%rsi,1), %%r13               \n\t" // r13 = 3*rs_c;
	"                                            \n\t"
	"                                            \n\t"
	"                                            \n\t"
	"                                            \n\t" // determine if
	"                                            \n\t" //    c    % 32 == 0, AND
	"                                            \n\t" //  8*cs_c % 32 == 0, AND
	"                                            \n\t" //    rs_c      == 1
	"                                            \n\t" // ie: aligned, ldim aligned, and
	"                                            \n\t" // column-stored
	"                                            \n\t"
	"cmpq       $8, %%rsi                        \n\t" // set ZF if (8*rs_c) == 8.
	"sete           %%bl                         \n\t" // bl = ( ZF == 1 ? 1 : 0 );
	"testq     $31, %%rcx                        \n\t" // set ZF if c & 32 is zero.
	"setz           %%bh                         \n\t" // bh = ( ZF == 0 ? 1 : 0 );
	"testq     $31, %%rdi                        \n\t" // set ZF if (8*cs_c) & 32 is zero.
	"setz           %%al                         \n\t" // al = ( ZF == 0 ? 1 : 0 );
	"                                            \n\t" // and(bl,bh) followed by
	"                                            \n\t" // and(bh,al) will reveal result
	"                                            \n\t"
	"                                            \n\t" // now avoid loading C if beta == 0
	"                                            \n\t"
	"vxorpd    %%ymm0,  %%ymm0,  %%ymm0          \n\t" // set ymm0 to zero.
	"vucomisd  %%xmm0,  %%xmm2                   \n\t" // set ZF if beta == 0.
	"je      .DBETAZERO                          \n\t" // if ZF = 1, jump to beta == 0 case
	"                                            \n\t"
	"                                            \n\t"
	"                                            \n\t" // check if aligned/column-stored
	"andb     %%bl, %%bh                         \n\t" // set ZF if bl & bh == 1.
	"andb     %%bh, %%al                         \n\t" // set ZF if bh & al == 1.
	"jne     .DCOLSTORED                         \n\t" // jump to column storage case
	"                                            \n\t"
	"                                            \n\t"
	"                                            \n\t"
	".DGENSTORED:                                \n\t"
	"                                            \n\t" // update c00:c33
	"                                            \n\t"
	"vextractf128 $1, %%ymm9,  %%xmm1            \n\t"
	"vmovlpd    (%%rcx),       %%xmm0,  %%xmm0   \n\t" // load c00 and c10,
	"vmovhpd    (%%rcx,%%rsi), %%xmm0,  %%xmm0   \n\t"
	"vmulpd           %%xmm2,  %%xmm0,  %%xmm0   \n\t" // scale by beta,
	"vaddpd           %%xmm9,  %%xmm0,  %%xmm0   \n\t" // add the gemm result,
	"vmovlpd          %%xmm0,  (%%rcx)           \n\t" // and store back to memory.
	"vmovhpd          %%xmm0,  (%%rcx,%%rsi)     \n\t"
	"vmovlpd    (%%rcx,%%r12), %%xmm0,  %%xmm0   \n\t" // load c20 and c30,
	"vmovhpd    (%%rcx,%%r13), %%xmm0,  %%xmm0   \n\t"
	"vmulpd           %%xmm2,  %%xmm0,  %%xmm0   \n\t" // scale by beta,
	"vaddpd           %%xmm1,  %%xmm0,  %%xmm0   \n\t" // add the gemm result,
	"vmovlpd          %%xmm0,  (%%rcx,%%r12)     \n\t" // and store back to memory.
	"vmovhpd          %%xmm0,  (%%rcx,%%r13)     \n\t"
	"addq      %%rdi, %%rcx                      \n\t" // c += cs_c;
	"                                            \n\t"
	"vextractf128 $1, %%ymm11, %%xmm1            \n\t"
	"vmovlpd    (%%rcx),       %%xmm0,  %%xmm0   \n\t" // load c01 and c11,
	"vmovhpd    (%%rcx,%%rsi), %%xmm0,  %%xmm0   \n\t"
	"vmulpd           %%xmm2,  %%xmm0,  %%xmm0   \n\t" // scale by beta,
	"vaddpd           %%xmm11, %%xmm0,  %%xmm0   \n\t" // add the gemm result,
	"vmovlpd          %%xmm0,  (%%rcx)           \n\t" // and store back to memory.
	"vmovhpd          %%xmm0,  (%%rcx,%%rsi)     \n\t"
	"vmovlpd    (%%rcx,%%r12), %%xmm0,  %%xmm0   \n\t" // load c21 and c31,
	"vmovhpd    (%%rcx,%%r13), %%xmm0,  %%xmm0   \n\t"
	"vmulpd           %%xmm2,  %%xmm0,  %%xmm0   \n\t" // scale by beta,
	"vaddpd           %%xmm1,  %%xmm0,  %%xmm0   \n\t" // add the gemm result,
	"vmovlpd          %%xmm0,  (%%rcx,%%r12)     \n\t" // and store back to memory.
	"vmovhpd          %%xmm0,  (%%rcx,%%r13)     \n\t"
	"addq      %%rdi, %%rcx                      \n\t" // c += cs_c;
	"                                            \n\t"
	"vextractf128 $1, %%ymm13, %%xmm1            \n\t"
	"vmovlpd    (%%rcx),       %%xmm0,  %%xmm0   \n\t" // load c02 and c12,
	"vmovhpd    (%%rcx,%%rsi), %%xmm0,  %%xmm0   \n\t"
	"vmulpd           %%xmm2,  %%xmm0,  %%xmm0   \n\t" // scale by beta,
	"vaddpd           %%xmm13, %%xmm0,  %%xmm0   \n\t" // add the gemm result,
	"vmovlpd          %%xmm0,  (%%rcx)           \n\t" // and store back to memory.
	"vmovhpd          %%xmm0,  (%%rcx,%%rsi)     \n\t"
	"vmovlpd    (%%rcx,%%r12), %%xmm0,  %%xmm0   \n\t" // load c22 and c32,
	"vmovhpd    (%%rcx,%%r13), %%xmm0,  %%xmm0   \n\t"
	"vmulpd           %%xmm2,  %%xmm0,  %%xmm0   \n\t" // scale by beta,
	"vaddpd           %%xmm1,  %%xmm0,  %%xmm0   \n\t" // add the gemm result,
	"vmovlpd          %%xmm0,  (%%rcx,%%r12)     \n\t" // and store back to memory.
	"vmovhpd          %%xmm0,  (%%rcx,%%r13)     \n\t"
	"addq      %%rdi, %%rcx                      \n\t" // c += cs_c;
	"                                            \n\t"
	"vextractf128 $1, %%ymm15, %%xmm1            \n\t"
	"vmovlpd    (%%rcx),       %%xmm0,  %%xmm0   \n\t" // load c03 and c13,
	"vmovhpd    (%%rcx,%%rsi), %%xmm0,  %%xmm0   \n\t"
	"vmulpd           %%xmm2,  %%xmm0,  %%xmm0   \n\t" // scale by beta,
	"vaddpd           %%xmm15, %%xmm0,  %%xmm0   \n\t" // add the gemm result,
	"vmovlpd          %%xmm0,  (%%rcx)           \n\t" // and store back to memory.
	"vmovhpd          %%xmm0,  (%%rcx,%%rsi)     \n\t"
	"vmovlpd    (%%rcx,%%r12), %%xmm0,  %%xmm0   \n\t" // load c23 and c33,
	"vmovhpd    (%%rcx,%%r13), %%xmm0,  %%xmm0   \n\t"
	"vmulpd           %%xmm2,  %%xmm0,  %%xmm0   \n\t" // scale by beta,
	"vaddpd           %%xmm1,  %%xmm0,  %%xmm0   \n\t" // add the gemm result,
	"vmovlpd          %%xmm0,  (%%rcx,%%r12)     \n\t" // and store back to memory.
	"vmovhpd          %%xmm0,  (%%rcx,%%r13)     \n\t"
	"                                            \n\t"
	"                                            \n\t" // update c40:c73
	"                                            \n\t"
	"vextractf128 $1, %%ymm8,  %%xmm1            \n\t"
	"vmovlpd    (%%rdx),       %%xmm0,  %%xmm0   \n\t" // load c40 and c50,
	"vmovhpd    (%%rdx,%%rsi), %%xmm0,  %%xmm0   \n\t"
	"vmulpd           %%xmm2,  %%xmm0,  %%xmm0   \n\t" // scale by beta,
	"vaddpd           %%xmm8,  %%xmm0,  %%xmm0   \n\t" // add the gemm result,
	"vmovlpd          %%xmm0,  (%%rdx)           \n\t" // and store back to memory.
	"vmovhpd          %%xmm0,  (%%rdx,%%rsi)     \n\t"
	"vmovlpd    (%%rdx,%%r12), %%xmm0,  %%xmm0   \n\t" // load c60 and c70,
	"vmovhpd    (%%rdx,%%r13), %%xmm0,  %%xmm0   \n\t"
	"vmulpd           %%xmm2,  %%xmm0,  %%xmm0   \n\t" // scale by beta,
	"vaddpd           %%xmm1,  %%xmm0,  %%xmm0   \n\t" // add the gemm result,
	"vmovlpd          %%xmm0,  (%%rdx,%%r12)     \n\t" // and store back to memory.
	"vmovhpd          %%xmm0,  (%%rdx,%%r13)     \n\t"
	"addq      %%rdi, %%rdx                      \n\t" // c += cs_c;
	"                                            \n\t"
	"vextractf128 $1, %%ymm10, %%xmm1            \n\t"
	"vmovlpd    (%%rdx),       %%xmm0,  %%xmm0   \n\t" // load c41 and c51,
	"vmovhpd    (%%rdx,%%rsi), %%xmm0,  %%xmm0   \n\t"
	"vmulpd           %%xmm2,  %%xmm0,  %%xmm0   \n\t" // scale by beta,
	"vaddpd           %%xmm10, %%xmm0,  %%xmm0   \n\t" // add the gemm result,
	"vmovlpd          %%xmm0,  (%%rdx)           \n\t" // and store back to memory.
	"vmovhpd          %%xmm0,  (%%rdx,%%rsi)     \n\t"
	"vmovlpd    (%%rdx,%%r12), %%xmm0,  %%xmm0   \n\t" // load c61 and c71,
	"vmovhpd    (%%rdx,%%r13), %%xmm0,  %%xmm0   \n\t"
	"vmulpd           %%xmm2,  %%xmm0,  %%xmm0   \n\t" // scale by beta,
	"vaddpd           %%xmm1,  %%xmm0,  %%xmm0   \n\t" // add the gemm result,
	"vmovlpd          %%xmm0,  (%%rdx,%%r12)     \n\t" // and store back to memory.
	"vmovhpd          %%xmm0,  (%%rdx,%%r13)     \n\t"
	"addq      %%rdi, %%rdx                      \n\t" // c += cs_c;
	"                                            \n\t"
	"vextractf128 $1, %%ymm12, %%xmm1            \n\t"
	"vmovlpd    (%%rdx),       %%xmm0,  %%xmm0   \n\t" // load c42 and c52,
	"vmovhpd    (%%rdx,%%rsi), %%xmm0,  %%xmm0   \n\t"
	"vmulpd           %%xmm2,  %%xmm0,  %%xmm0   \n\t" // scale by beta,
	"vaddpd           %%xmm12, %%xmm0,  %%xmm0   \n\t" // add the gemm result,
	"vmovlpd          %%xmm0,  (%%rdx)           \n\t" // and store back to memory.
	"vmovhpd          %%xmm0,  (%%rdx,%%rsi)     \n\t"
	"vmovlpd    (%%rdx,%%r12), %%xmm0,  %%xmm0   \n\t" // load c62 and c72,
	"vmovhpd    (%%rdx,%%r13), %%xmm0,  %%xmm0   \n\t"
	"vmulpd           %%xmm2,  %%xmm0,  %%xmm0   \n\t" // scale by beta,
	"vaddpd           %%xmm1,  %%xmm0,  %%xmm0   \n\t" // add the gemm result,
	"vmovlpd          %%xmm0,  (%%rdx,%%r12)     \n\t" // and store back to memory.
	"vmovhpd          %%xmm0,  (%%rdx,%%r13)     \n\t"
	"addq      %%rdi, %%rdx                      \n\t" // c += cs_c;
	"                                            \n\t"
	"vextractf128 $1, %%ymm14, %%xmm1            \n\t"
	"vmovlpd    (%%rdx),       %%xmm0,  %%xmm0   \n\t" // load c43 and c53,
	"vmovhpd    (%%rdx,%%rsi), %%xmm0,  %%xmm0   \n\t"
	"vmulpd           %%xmm2,  %%xmm0,  %%xmm0   \n\t" // scale by beta,
	"vaddpd           %%xmm14, %%xmm0,  %%xmm0   \n\t" // add the gemm result,
	"vmovlpd          %%xmm0,  (%%rdx)           \n\t" // and store back to memory.
	"vmovhpd          %%xmm0,  (%%rdx,%%rsi)     \n\t"
	"vmovlpd    (%%rdx,%%r12), %%xmm0,  %%xmm0   \n\t" // load c63 and c73,
	"vmovhpd    (%%rdx,%%r13), %%xmm0,  %%xmm0   \n\t"
	"vmulpd           %%xmm2,  %%xmm0,  %%xmm0   \n\t" // scale by beta,
	"vaddpd           %%xmm1,  %%xmm0,  %%xmm0   \n\t" // add the gemm result,
	"vmovlpd          %%xmm0,  (%%rdx,%%r12)     \n\t" // and store back to memory.
	"vmovhpd          %%xmm0,  (%%rdx,%%r13)     \n\t"
	"                                            \n\t"
	"                                            \n\t"
	"jmp    .DDONE                               \n\t" // jump to end.
	"                                            \n\t"
	"                                            \n\t"
	"                                            \n\t"
	".DCOLSTORED:                                \n\t"
	"                                            \n\t" // update c00:c33
	"                                            \n\t"
	"vmovapd    (%%rcx),       %%ymm0            \n\t" // load c00:c30,
	"vmulpd           %%ymm2,  %%ymm0,  %%ymm0   \n\t" // scale by beta,
	"vaddpd           %%ymm9,  %%ymm0,  %%ymm0   \n\t" // add the gemm result,
	"vmovapd          %%ymm0,  (%%rcx)           \n\t" // and store back to memory.
	"addq      %%rdi, %%rcx                      \n\t" // c += cs_c;
	"                                            \n\t"
	"vmovapd    (%%rcx),       %%ymm0            \n\t" // load c01:c31,
	"vmulpd           %%ymm2,  %%ymm0,  %%ymm0   \n\t" // scale by beta,
	"vaddpd           %%ymm11, %%ymm0,  %%ymm0   \n\t" // add the gemm result,
	"vmovapd          %%ymm0,  (%%rcx)           \n\t" // and store back to memory.
	"addq      %%rdi, %%rcx                      \n\t" // c += cs_c;
	"                                            \n\t"
	"vmovapd    (%%rcx),       %%ymm0            \n\t" // load c02:c32,
	"vmulpd           %%ymm2,  %%ymm0,  %%ymm0   \n\t" // scale by beta,
	"vaddpd           %%ymm13, %%ymm0,  %%ymm0   \n\t" // add the gemm result,
	"vmovapd          %%ymm0,  (%%rcx)           \n\t" // and store back to memory.
	"addq      %%rdi, %%rcx                      \n\t" // c += cs_c;
	"                                            \n\t"
	"vmovapd    (%%rcx),       %%ymm0            \n\t" // load c03:c33,
	"vmulpd           %%ymm2,  %%ymm0,  %%ymm0   \n\t" // scale by beta,
	"vaddpd           %%ymm15, %%ymm0,  %%ymm0   \n\t" // add the gemm result,
	"vmovapd          %%ymm0,  (%%rcx)           \n\t" // and store back to memory.
	"                                            \n\t"
	"                                            \n\t" // update c40:c73
	"                                            \n\t"
	"vmovapd    (%%rdx),       %%ymm0            \n\t" // load c40:c70,
	"vmulpd           %%ymm2,  %%ymm0,  %%ymm0   \n\t" // scale by beta,
	"vaddpd           %%ymm8,  %%ymm0,  %%ymm0   \n\t" // add the gemm result,
	"vmovapd          %%ymm0,  (%%rdx)           \n\t" // and store back to memory.
	"addq      %%rdi, %%rdx                      \n\t" // c += cs_c;
	"                                            \n\t"
	"vmovapd    (%%rdx),       %%ymm0            \n\t" // load c41:c71,
	"vmulpd           %%ymm2,  %%ymm0,  %%ymm0   \n\t" // scale by beta,
	"vaddpd           %%ymm10, %%ymm0,  %%ymm0   \n\t" // add the gemm result,
	"vmovapd          %%ymm0,  (%%rdx)           \n\t" // and store back to memory.
	"addq      %%rdi, %%rdx                      \n\t" // c += cs_c;
	"                                            \n\t"
	"vmovapd    (%%rdx),       %%ymm0            \n\t" // load c42:c72,
	"vmulpd           %%ymm2,  %%ymm0,  %%ymm0   \n\t" // scale by beta,
	"vaddpd           %%ymm12, %%ymm0,  %%ymm0   \n\t" // add the gemm result,
	"vmovapd          %%ymm0,  (%%rdx)           \n\t" // and store back to memory.
	"addq      %%rdi, %%rdx                      \n\t" // c += cs_c;
	"                                            \n\t"
	"vmovapd    (%%rdx),       %%ymm0            \n\t" // load c43:c73,
	"vmulpd           %%ymm2,  %%ymm0,  %%ymm0   \n\t" // scale by beta,
	"vaddpd           %%ymm14, %%ymm0,  %%ymm0   \n\t" // add the gemm result,
	"vmovapd          %%ymm0,  (%%rdx)           \n\t" // and store back to memory.
	"                                            \n\t"
	"                                            \n\t"
	"jmp    .DDONE                               \n\t" // jump to end.
	"                                            \n\t"
	"                                            \n\t"
	"                                            \n\t"
	"                                            \n\t"
	".DBETAZERO:                                 \n\t"
	"                                            \n\t" // check if aligned/column-stored
	"andb     %%bl, %%bh                         \n\t" // set ZF if bl & bh == 1.
	"andb     %%bh, %%al                         \n\t" // set ZF if bh & al == 1.
	"jne     .DCOLSTORBZ                         \n\t" // jump to column storage case
	"                                            \n\t"
	"                                            \n\t"
	"                                            \n\t"
	".DGENSTORBZ:                                \n\t"
	"                                            \n\t" // update c00:c33
	"                                            \n\t"
	"vextractf128 $1, %%ymm9,  %%xmm1            \n\t"
	"vmovlpd          %%xmm9,  (%%rcx)           \n\t" // store to c00:c30
	"vmovhpd          %%xmm9,  (%%rcx,%%rsi)     \n\t"
	"vmovlpd          %%xmm1,  (%%rcx,%%r12)     \n\t"
	"vmovhpd          %%xmm1,  (%%rcx,%%r13)     \n\t"
	"addq      %%rdi, %%rcx                      \n\t" // c += cs_c;
	"                                            \n\t"
	"vextractf128 $1, %%ymm11, %%xmm1            \n\t"
	"vmovlpd          %%xmm11, (%%rcx)           \n\t" // store to c01:c31
	"vmovhpd          %%xmm11, (%%rcx,%%rsi)     \n\t"
	"vmovlpd          %%xmm1,  (%%rcx,%%r12)     \n\t"
	"vmovhpd          %%xmm1,  (%%rcx,%%r13)     \n\t"
	"addq      %%rdi, %%rcx                      \n\t" // c += cs_c;
	"                                            \n\t"
	"vextractf128 $1, %%ymm13, %%xmm1            \n\t"
	"vmovlpd          %%xmm13, (%%rcx)           \n\t" // store to c02:c32
	"vmovhpd          %%xmm13, (%%rcx,%%rsi)     \n\t"
	"vmovlpd          %%xmm1,  (%%rcx,%%r12)     \n\t"
	"vmovhpd          %%xmm1,  (%%rcx,%%r13)     \n\t"
	"addq      %%rdi, %%rcx                      \n\t" // c += cs_c;
	"                                            \n\t"
	"vextractf128 $1, %%ymm15, %%xmm1            \n\t"
	"vmovlpd          %%xmm15, (%%rcx)           \n\t" // store to c03:c33
	"vmovhpd          %%xmm15, (%%rcx,%%rsi)     \n\t"
	"vmovlpd          %%xmm1,  (%%rcx,%%r12)     \n\t"
	"vmovhpd          %%xmm1,  (%%rcx,%%r13)     \n\t"
	"                                            \n\t"
	"                                            \n\t" // update c40:c73
	"                                            \n\t"
	"vextractf128 $1, %%ymm8,  %%xmm1            \n\t"
	"vmovlpd          %%xmm8,  (%%rdx)           \n\t" // store to c40:c70
	"vmovhpd          %%xmm8,  (%%rdx,%%rsi)     \n\t"
	"vmovlpd          %%xmm1,  (%%rdx,%%r12)     \n\t"
	"vmovhpd          %%xmm1,  (%%rdx,%%r13)     \n\t"
	"addq      %%rdi, %%rdx                      \n\t" // c += cs_c;
	"                                            \n\t"
	"vextractf128 $1, %%ymm10, %%xmm1            \n\t"
	"vmovlpd          %%xmm10, (%%rdx)           \n\t" // store to c41:c71
	"vmovhpd          %%xmm10, (%%rdx,%%rsi)     \n\t"
	"vmovlpd          %%xmm1,  (%%rdx,%%r12)     \n\t"
	"vmovhpd          %%xmm1,  (%%rdx,%%r13)     \n\t"
	"addq      %%rdi, %%rdx                      \n\t" // c += cs_c;
	"                                            \n\t"
	"vextractf128 $1, %%ymm12, %%xmm1            \n\t"
	"vmovlpd          %%xmm12, (%%rdx)           \n\t" // store to c42:c72
	"vmovhpd          %%xmm12, (%%rdx,%%rsi)     \n\t"
	"vmovlpd          %%xmm1,  (%%rdx,%%r12)     \n\t"
	"vmovhpd          %%xmm1,  (%%rdx,%%r13)     \n\t"
	"addq      %%rdi, %%rdx                      \n\t" // c += cs_c;
	"                                            \n\t"
	"vextractf128 $1, %%ymm14, %%xmm1            \n\t"
	"vmovlpd          %%xmm14, (%%rdx)           \n\t" // store to c43:c73
	"vmovhpd          %%xmm14, (%%rdx,%%rsi)     \n\t"
	"vmovlpd          %%xmm1,  (%%rdx,%%r12)     \n\t"
	"vmovhpd          %%xmm1,  (%%rdx,%%r13)     \n\t"
	"                                            \n\t"
	"                                            \n\t"
	"jmp    .DDONE                               \n\t" // jump to end.
	"                                            \n\t"
	"                                            \n\t"
	"                                            \n\t"
	".DCOLSTORBZ:                                \n\t"
	"                                            \n\t" // update c00:c33
	"                                            \n\t"
	"vmovapd          %%ymm9,  (%%rcx)           \n\t" // store c00:c30
	"addq      %%rdi, %%rcx                      \n\t" // c += cs_c;
	"                                            \n\t"
	"vmovapd          %%ymm11, (%%rcx)           \n\t" // store c01:c31
	"addq      %%rdi, %%rcx                      \n\t" // c += cs_c;
	"                                            \n\t"
	"vmovapd          %%ymm13, (%%rcx)           \n\t" // store c02:c32
	"addq      %%rdi, %%rcx                      \n\t" // c += cs_c;
	"                                            \n\t"
	"vmovapd          %%ymm15, (%%rcx)           \n\t" // store c03:c33
	"                                            \n\t"
	"                                            \n\t" // update c40:c73
	"                                            \n\t"
	"vmovapd          %%ymm8,  (%%rdx)           \n\t" // store c40:c70
	"addq      %%rdi, %%rdx                      \n\t" // c += cs_c;
	"                                            \n\t"
	"vmovapd          %%ymm10, (%%rdx)           \n\t" // store c41:c71
	"addq      %%rdi, %%rdx                      \n\t" // c += cs_c;
	"                                            \n\t"
	"vmovapd          %%ymm12, (%%rdx)           \n\t" // store c42:c72
	"addq      %%rdi, %%rdx                      \n\t" // c += cs_c;
	"                                            \n\t"
	"vmovapd          %%ymm14, (%%rdx)           \n\t" // store c43:c73
	"                                            \n\t"
	"                                            \n\t"
	"                                            \n\t"
	"                                            \n\t"
	"                                            \n\t"
	".DDONE:                                     \n\t"
	"                                            \n\t"

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
	  "m" (cs_c),   // 8
	  "m" (b_next)/*, // 9
	  "m" (a_next)*/  // 10
	: // register clobber list
	  "rax", "rbx", "rcx", "rdx", "rsi", "rdi", 
	  "r8", "r9", "r10", "r11", "r12", "r13", "r14", "r15",
	  "xmm0", "xmm1", "xmm2", "xmm3",
	  "xmm4", "xmm5", "xmm6", "xmm7",
	  "xmm8", "xmm9", "xmm10", "xmm11",
	  "xmm12", "xmm13", "xmm14", "xmm15",
	  "memory"
	);
}

void bli_cgemm_asm_8x4(
                        dim_t              k,
                        scomplex* restrict alpha,
                        scomplex* restrict a,
                        scomplex* restrict b,
                        scomplex* restrict beta,
                        scomplex* restrict c, inc_t rs_c, inc_t cs_c,
                        auxinfo_t*         data
                      )
{
	//void*   a_next = bli_auxinfo_next_a( data );
	void*   b_next = bli_auxinfo_next_b( data );

	dim_t   k_iter = k / 4;
	dim_t   k_left = k % 4;

	__asm__ volatile
	(
	"                                            \n\t"
	"                                            \n\t"
	"movq                %2, %%rax               \n\t" // load address of a.
	"movq                %3, %%rbx               \n\t" // load address of b.
	"movq                %9, %%r15               \n\t" // load address of b_next.
	//"movq               %10, %%r14               \n\t" // load address of a_next.
	"addq          $-4 * 64, %%r15               \n\t"
	"                                            \n\t"
	"vmovaps        0 * 32(%%rax), %%ymm0        \n\t" // initialize loop by pre-loading
	"vmovsldup      0 * 32(%%rbx), %%ymm2        \n\t"
	"vpermilps     $0x4e, %%ymm2,  %%ymm3        \n\t"
	"                                            \n\t"
	"movq                %6, %%rcx               \n\t" // load address of c
	"movq                %8, %%rdi               \n\t" // load cs_c
	"leaq        (,%%rdi,8), %%rdi               \n\t" // cs_c *= sizeof(scomplex)
	"leaq   (%%rcx,%%rdi,2), %%r10               \n\t" // load address of c + 2*cs_c;
	"                                            \n\t"
	"prefetcht0   3 * 8(%%rcx)                   \n\t" // prefetch c + 0*cs_c
	"prefetcht0   3 * 8(%%rcx,%%rdi)             \n\t" // prefetch c + 1*cs_c
	"prefetcht0   3 * 8(%%r10)                   \n\t" // prefetch c + 2*cs_c
	"prefetcht0   3 * 8(%%r10,%%rdi)             \n\t" // prefetch c + 3*cs_c
	"                                            \n\t"
	"vxorps    %%ymm8,  %%ymm8,  %%ymm8          \n\t"
	"vxorps    %%ymm9,  %%ymm9,  %%ymm9          \n\t"
	"vxorps    %%ymm10, %%ymm10, %%ymm10         \n\t"
	"vxorps    %%ymm11, %%ymm11, %%ymm11         \n\t"
	"vxorps    %%ymm12, %%ymm12, %%ymm12         \n\t"
	"vxorps    %%ymm13, %%ymm13, %%ymm13         \n\t"
	"vxorps    %%ymm14, %%ymm14, %%ymm14         \n\t"
	"vxorps    %%ymm15, %%ymm15, %%ymm15         \n\t"
	"                                            \n\t"
	"                                            \n\t"
	"                                            \n\t"
	"movq      %0, %%rsi                         \n\t" // i = k_iter;
	"testq  %%rsi, %%rsi                         \n\t" // check i via logical AND.
	"je     .CCONSIDKLEFT                        \n\t" // if i == 0, jump to code that
	"                                            \n\t" // contains the k_left loop.
	"                                            \n\t"
	"                                            \n\t"
	".CLOOPKITER:                                \n\t" // MAIN LOOP
	"                                            \n\t"
	"addq         $4 * 4 * 8,  %%r15             \n\t" // b_next += 4*4 (unroll x nr)
	"                                            \n\t"
	"                                            \n\t" // iteration 0
	"prefetcht0     8 * 32(%%rax)                \n\t"
	"vmovaps        1 * 32(%%rax),      %%ymm1   \n\t"
	"vmulps           %%ymm0,  %%ymm2,  %%ymm6   \n\t"
	"vperm2f128 $0x3, %%ymm2,  %%ymm2,  %%ymm4   \n\t"
	"vmulps           %%ymm0,  %%ymm3,  %%ymm7   \n\t"
	"vperm2f128 $0x3, %%ymm3,  %%ymm3,  %%ymm5   \n\t"
	"vaddps           %%ymm6,  %%ymm15, %%ymm15  \n\t"
	"vaddps           %%ymm7,  %%ymm13, %%ymm13  \n\t"
	"                                            \n\t"
	"vmulps           %%ymm1,  %%ymm2,  %%ymm6   \n\t"
	"vmovshdup      0 * 32(%%rbx),      %%ymm2   \n\t"
	"vmulps           %%ymm1,  %%ymm3,  %%ymm7   \n\t"
	"vpermilps $0x4e, %%ymm2,  %%ymm3            \n\t"
	"vaddps           %%ymm6,  %%ymm14, %%ymm14  \n\t"
	"vaddps           %%ymm7,  %%ymm12, %%ymm12  \n\t"
	"                                            \n\t"
	"vmulps           %%ymm0,  %%ymm4,  %%ymm6   \n\t"
	"vmulps           %%ymm0,  %%ymm5,  %%ymm7   \n\t"
	"vpermilps $0xb1, %%ymm0,  %%ymm0            \n\t"
	"vaddps           %%ymm6,  %%ymm11, %%ymm11  \n\t"
	"vaddps           %%ymm7,  %%ymm9,  %%ymm9   \n\t"
	"                                            \n\t"
	"vmulps           %%ymm1,  %%ymm4,  %%ymm6   \n\t"
	"vperm2f128 $0x3, %%ymm2,  %%ymm2,  %%ymm4   \n\t"
	"vmulps           %%ymm1,  %%ymm5,  %%ymm7   \n\t"
	"vperm2f128 $0x3, %%ymm3,  %%ymm3,  %%ymm5   \n\t"
	"vaddps           %%ymm6,  %%ymm10, %%ymm10  \n\t"
	"vaddps           %%ymm7,  %%ymm8,  %%ymm8   \n\t"
	"prefetcht0   0 * 32(%%r15)                  \n\t" // prefetch b_next[0*4]
	"                                            \n\t"
	"vpermilps $0xb1, %%ymm1,  %%ymm1            \n\t"
	"vmulps           %%ymm0,  %%ymm2,  %%ymm6   \n\t"
	"vmulps           %%ymm0,  %%ymm3,  %%ymm7   \n\t"
	"vaddsubps        %%ymm6,  %%ymm15, %%ymm15  \n\t"
	"vaddsubps        %%ymm7,  %%ymm13, %%ymm13  \n\t"
	"                                            \n\t"
	"vmulps           %%ymm1,  %%ymm2,  %%ymm6   \n\t"
	"vmovsldup      1 * 32(%%rbx),      %%ymm2   \n\t"
	"vmulps           %%ymm1,  %%ymm3,  %%ymm7   \n\t"
	"vpermilps $0x4e, %%ymm2,  %%ymm3            \n\t"
	"vaddsubps        %%ymm6,  %%ymm14, %%ymm14  \n\t"
	"vaddsubps        %%ymm7,  %%ymm12, %%ymm12  \n\t"
	"                                            \n\t"
	"vmulps           %%ymm0,  %%ymm4,  %%ymm6   \n\t"
	"vmulps           %%ymm0,  %%ymm5,  %%ymm7   \n\t"
	"vmovaps        2 * 32(%%rax),      %%ymm0   \n\t"
	"vaddsubps        %%ymm6,  %%ymm11, %%ymm11  \n\t"
	"vaddsubps        %%ymm7,  %%ymm9,  %%ymm9   \n\t"
	"                                            \n\t"
	"vmulps           %%ymm1,  %%ymm4,  %%ymm6   \n\t"
	"vmulps           %%ymm1,  %%ymm5,  %%ymm7   \n\t"
	"vaddsubps        %%ymm6,  %%ymm10, %%ymm10  \n\t"
	"vaddsubps        %%ymm7,  %%ymm8,  %%ymm8   \n\t"
	"                                            \n\t"
	"                                            \n\t"
	"                                            \n\t" // iteration 1
	"prefetcht0    10 * 32(%%rax)                \n\t"
	"vmovaps        3 * 32(%%rax),      %%ymm1   \n\t"
	"vmulps           %%ymm0,  %%ymm2,  %%ymm6   \n\t"
	"vperm2f128 $0x3, %%ymm2,  %%ymm2,  %%ymm4   \n\t"
	"vmulps           %%ymm0,  %%ymm3,  %%ymm7   \n\t"
	"vperm2f128 $0x3, %%ymm3,  %%ymm3,  %%ymm5   \n\t"
	"vaddps           %%ymm6,  %%ymm15, %%ymm15  \n\t"
	"vaddps           %%ymm7,  %%ymm13, %%ymm13  \n\t"
	"                                            \n\t"
	"vmulps           %%ymm1,  %%ymm2,  %%ymm6   \n\t"
	"vmovshdup      1 * 32(%%rbx),      %%ymm2   \n\t"
	"vmulps           %%ymm1,  %%ymm3,  %%ymm7   \n\t"
	"vpermilps $0x4e, %%ymm2,  %%ymm3            \n\t"
	"vaddps           %%ymm6,  %%ymm14, %%ymm14  \n\t"
	"vaddps           %%ymm7,  %%ymm12, %%ymm12  \n\t"
	"                                            \n\t"
	"vmulps           %%ymm0,  %%ymm4,  %%ymm6   \n\t"
	"vmulps           %%ymm0,  %%ymm5,  %%ymm7   \n\t"
	"vpermilps $0xb1, %%ymm0,  %%ymm0            \n\t"
	"vaddps           %%ymm6,  %%ymm11, %%ymm11  \n\t"
	"vaddps           %%ymm7,  %%ymm9,  %%ymm9   \n\t"
	"                                            \n\t"
	"vmulps           %%ymm1,  %%ymm4,  %%ymm6   \n\t"
	"vperm2f128 $0x3, %%ymm2,  %%ymm2,  %%ymm4   \n\t"
	"vmulps           %%ymm1,  %%ymm5,  %%ymm7   \n\t"
	"vperm2f128 $0x3, %%ymm3,  %%ymm3,  %%ymm5   \n\t"
	"vaddps           %%ymm6,  %%ymm10, %%ymm10  \n\t"
	"vaddps           %%ymm7,  %%ymm8,  %%ymm8   \n\t"
	"                                            \n\t"
	"vpermilps $0xb1, %%ymm1,  %%ymm1            \n\t"
	"vmulps           %%ymm0,  %%ymm2,  %%ymm6   \n\t"
	"vmulps           %%ymm0,  %%ymm3,  %%ymm7   \n\t"
	"vaddsubps        %%ymm6,  %%ymm15, %%ymm15  \n\t"
	"vaddsubps        %%ymm7,  %%ymm13, %%ymm13  \n\t"
	"                                            \n\t"
	"vmulps           %%ymm1,  %%ymm2,  %%ymm6   \n\t"
	"vmovsldup      2 * 32(%%rbx),      %%ymm2   \n\t"
	"vmulps           %%ymm1,  %%ymm3,  %%ymm7   \n\t"
	"vpermilps $0x4e, %%ymm2,  %%ymm3            \n\t"
	"vaddsubps        %%ymm6,  %%ymm14, %%ymm14  \n\t"
	"vaddsubps        %%ymm7,  %%ymm12, %%ymm12  \n\t"
	"                                            \n\t"
	"vmulps           %%ymm0,  %%ymm4,  %%ymm6   \n\t"
	"vmulps           %%ymm0,  %%ymm5,  %%ymm7   \n\t"
	"vmovaps        4 * 32(%%rax),      %%ymm0   \n\t"
	"vaddsubps        %%ymm6,  %%ymm11, %%ymm11  \n\t"
	"vaddsubps        %%ymm7,  %%ymm9,  %%ymm9   \n\t"
	"                                            \n\t"
	"vmulps           %%ymm1,  %%ymm4,  %%ymm6   \n\t"
	"vmulps           %%ymm1,  %%ymm5,  %%ymm7   \n\t"
	"vaddsubps        %%ymm6,  %%ymm10, %%ymm10  \n\t"
	"vaddsubps        %%ymm7,  %%ymm8,  %%ymm8   \n\t"
	"                                            \n\t"
	"                                            \n\t"
	"                                            \n\t" // iteration 2
	"prefetcht0    12 * 32(%%rax)                \n\t"
	"vmovaps        5 * 32(%%rax),      %%ymm1   \n\t"
	"vmulps           %%ymm0,  %%ymm2,  %%ymm6   \n\t"
	"vperm2f128 $0x3, %%ymm2,  %%ymm2,  %%ymm4   \n\t"
	"vmulps           %%ymm0,  %%ymm3,  %%ymm7   \n\t"
	"vperm2f128 $0x3, %%ymm3,  %%ymm3,  %%ymm5   \n\t"
	"vaddps           %%ymm6,  %%ymm15, %%ymm15  \n\t"
	"vaddps           %%ymm7,  %%ymm13, %%ymm13  \n\t"
	"                                            \n\t"
	"vmulps           %%ymm1,  %%ymm2,  %%ymm6   \n\t"
	"vmovshdup      2 * 32(%%rbx),      %%ymm2   \n\t"
	"vmulps           %%ymm1,  %%ymm3,  %%ymm7   \n\t"
	"vpermilps $0x4e, %%ymm2,  %%ymm3            \n\t"
	"vaddps           %%ymm6,  %%ymm14, %%ymm14  \n\t"
	"vaddps           %%ymm7,  %%ymm12, %%ymm12  \n\t"
	"                                            \n\t"
	"vmulps           %%ymm0,  %%ymm4,  %%ymm6   \n\t"
	"vmulps           %%ymm0,  %%ymm5,  %%ymm7   \n\t"
	"vpermilps $0xb1, %%ymm0,  %%ymm0            \n\t"
	"vaddps           %%ymm6,  %%ymm11, %%ymm11  \n\t"
	"vaddps           %%ymm7,  %%ymm9,  %%ymm9   \n\t"
	"                                            \n\t"
	"vmulps           %%ymm1,  %%ymm4,  %%ymm6   \n\t"
	"vperm2f128 $0x3, %%ymm2,  %%ymm2,  %%ymm4   \n\t"
	"vmulps           %%ymm1,  %%ymm5,  %%ymm7   \n\t"
	"vperm2f128 $0x3, %%ymm3,  %%ymm3,  %%ymm5   \n\t"
	"vaddps           %%ymm6,  %%ymm10, %%ymm10  \n\t"
	"vaddps           %%ymm7,  %%ymm8,  %%ymm8   \n\t"
	"prefetcht0   2 * 32(%%r15)                  \n\t" // prefetch b_next[2*4]
	"                                            \n\t"
	"vpermilps $0xb1, %%ymm1,  %%ymm1            \n\t"
	"vmulps           %%ymm0,  %%ymm2,  %%ymm6   \n\t"
	"vmulps           %%ymm0,  %%ymm3,  %%ymm7   \n\t"
	"vaddsubps        %%ymm6,  %%ymm15, %%ymm15  \n\t"
	"vaddsubps        %%ymm7,  %%ymm13, %%ymm13  \n\t"
	"                                            \n\t"
	"vmulps           %%ymm1,  %%ymm2,  %%ymm6   \n\t"
	"vmovsldup      3 * 32(%%rbx),      %%ymm2   \n\t"
	"vmulps           %%ymm1,  %%ymm3,  %%ymm7   \n\t"
	"vpermilps $0x4e, %%ymm2,  %%ymm3            \n\t"
	"vaddsubps        %%ymm6,  %%ymm14, %%ymm14  \n\t"
	"vaddsubps        %%ymm7,  %%ymm12, %%ymm12  \n\t"
	"                                            \n\t"
	"vmulps           %%ymm0,  %%ymm4,  %%ymm6   \n\t"
	"vmulps           %%ymm0,  %%ymm5,  %%ymm7   \n\t"
	"vmovaps        6 * 32(%%rax),      %%ymm0   \n\t"
	"vaddsubps        %%ymm6,  %%ymm11, %%ymm11  \n\t"
	"vaddsubps        %%ymm7,  %%ymm9,  %%ymm9   \n\t"
	"                                            \n\t"
	"vmulps           %%ymm1,  %%ymm4,  %%ymm6   \n\t"
	"vmulps           %%ymm1,  %%ymm5,  %%ymm7   \n\t"
	"vaddsubps        %%ymm6,  %%ymm10, %%ymm10  \n\t"
	"vaddsubps        %%ymm7,  %%ymm8,  %%ymm8   \n\t"
	"                                            \n\t"
	"                                            \n\t"
	"                                            \n\t" // iteration 3
	"prefetcht0    14 * 32(%%rax)                \n\t"
	"vmovaps        7 * 32(%%rax),      %%ymm1   \n\t"
	"vmulps           %%ymm0,  %%ymm2,  %%ymm6   \n\t"
	"vperm2f128 $0x3, %%ymm2,  %%ymm2,  %%ymm4   \n\t"
	"vmulps           %%ymm0,  %%ymm3,  %%ymm7   \n\t"
	"vperm2f128 $0x3, %%ymm3,  %%ymm3,  %%ymm5   \n\t"
	"vaddps           %%ymm6,  %%ymm15, %%ymm15  \n\t"
	"vaddps           %%ymm7,  %%ymm13, %%ymm13  \n\t"
	"                                            \n\t"
	"vmulps           %%ymm1,  %%ymm2,  %%ymm6   \n\t"
	"vmovshdup      3 * 32(%%rbx),      %%ymm2   \n\t"
	"vmulps           %%ymm1,  %%ymm3,  %%ymm7   \n\t"
	"vpermilps $0x4e, %%ymm2,  %%ymm3            \n\t"
	"vaddps           %%ymm6,  %%ymm14, %%ymm14  \n\t"
	"vaddps           %%ymm7,  %%ymm12, %%ymm12  \n\t"
	"                                            \n\t"
	"vmulps           %%ymm0,  %%ymm4,  %%ymm6   \n\t"
	"vmulps           %%ymm0,  %%ymm5,  %%ymm7   \n\t"
	"vpermilps $0xb1, %%ymm0,  %%ymm0            \n\t"
	"vaddps           %%ymm6,  %%ymm11, %%ymm11  \n\t"
	"vaddps           %%ymm7,  %%ymm9,  %%ymm9   \n\t"
	"                                            \n\t"
	"vmulps           %%ymm1,  %%ymm4,  %%ymm6   \n\t"
	"vperm2f128 $0x3, %%ymm2,  %%ymm2,  %%ymm4   \n\t"
	"vmulps           %%ymm1,  %%ymm5,  %%ymm7   \n\t"
	"vperm2f128 $0x3, %%ymm3,  %%ymm3,  %%ymm5   \n\t"
	"vaddps           %%ymm6,  %%ymm10, %%ymm10  \n\t"
	"vaddps           %%ymm7,  %%ymm8,  %%ymm8   \n\t"
	"                                            \n\t"
	"vpermilps $0xb1, %%ymm1,  %%ymm1            \n\t"
	"vmulps           %%ymm0,  %%ymm2,  %%ymm6   \n\t"
	"vmulps           %%ymm0,  %%ymm3,  %%ymm7   \n\t"
	"vaddsubps        %%ymm6,  %%ymm15, %%ymm15  \n\t"
	"vaddsubps        %%ymm7,  %%ymm13, %%ymm13  \n\t"
	"                                            \n\t"
	"vmulps           %%ymm1,  %%ymm2,  %%ymm6   \n\t"
	"vmovsldup      4 * 32(%%rbx),      %%ymm2   \n\t"
	"vmulps           %%ymm1,  %%ymm3,  %%ymm7   \n\t"
	"vpermilps $0x4e, %%ymm2,  %%ymm3            \n\t"
	"vaddsubps        %%ymm6,  %%ymm14, %%ymm14  \n\t"
	"vaddsubps        %%ymm7,  %%ymm12, %%ymm12  \n\t"
	"                                            \n\t"
	"vmulps           %%ymm0,  %%ymm4,  %%ymm6   \n\t"
	"vmulps           %%ymm0,  %%ymm5,  %%ymm7   \n\t"
	"vmovaps        8 * 32(%%rax),      %%ymm0   \n\t"
	"vaddsubps        %%ymm6,  %%ymm11, %%ymm11  \n\t"
	"vaddsubps        %%ymm7,  %%ymm9,  %%ymm9   \n\t"
	"                                            \n\t"
	"vmulps           %%ymm1,  %%ymm4,  %%ymm6   \n\t"
	"vmulps           %%ymm1,  %%ymm5,  %%ymm7   \n\t"
	"vaddsubps        %%ymm6,  %%ymm10, %%ymm10  \n\t"
	"vaddsubps        %%ymm7,  %%ymm8,  %%ymm8   \n\t"
	"                                            \n\t"
	"                                            \n\t"
	"addq          $8 * 4 * 8, %%rax             \n\t" // a += 8*4 (unroll x mr)
	"addq          $4 * 4 * 8, %%rbx             \n\t" // b += 4*4 (unroll x nr)
	"                                            \n\t"
	"                                            \n\t"
	"decq   %%rsi                                \n\t" // i -= 1;
	"jne    .CLOOPKITER                          \n\t" // iterate again if i != 0.
	"                                            \n\t"
	"                                            \n\t"
	"                                            \n\t"
	"                                            \n\t"
	"                                            \n\t"
	"                                            \n\t"
	".CCONSIDKLEFT:                              \n\t"
	"                                            \n\t"
	"movq      %1, %%rsi                         \n\t" // i = k_left;
	"testq  %%rsi, %%rsi                         \n\t" // check i via logical AND.
	"je     .CPOSTACCUM                          \n\t" // if i == 0, we're done; jump to end.
	"                                            \n\t" // else, we prepare to enter k_left loop.
	"                                            \n\t"
	"                                            \n\t"
	".CLOOPKLEFT:                                \n\t" // EDGE LOOP
	"                                            \n\t"
	"                                            \n\t" // iteration 0
	"prefetcht0     8 * 32(%%rax)                \n\t"
	"vmovaps        1 * 32(%%rax),      %%ymm1   \n\t"
	"vmulps           %%ymm0,  %%ymm2,  %%ymm6   \n\t"
	"vperm2f128 $0x3, %%ymm2,  %%ymm2,  %%ymm4   \n\t"
	"vmulps           %%ymm0,  %%ymm3,  %%ymm7   \n\t"
	"vperm2f128 $0x3, %%ymm3,  %%ymm3,  %%ymm5   \n\t"
	"vaddps           %%ymm6,  %%ymm15, %%ymm15  \n\t"
	"vaddps           %%ymm7,  %%ymm13, %%ymm13  \n\t"
	"                                            \n\t"
	"vmulps           %%ymm1,  %%ymm2,  %%ymm6   \n\t"
	"vmovshdup      0 * 32(%%rbx),      %%ymm2   \n\t"
	"vmulps           %%ymm1,  %%ymm3,  %%ymm7   \n\t"
	"vpermilps $0x4e, %%ymm2,  %%ymm3            \n\t"
	"vaddps           %%ymm6,  %%ymm14, %%ymm14  \n\t"
	"vaddps           %%ymm7,  %%ymm12, %%ymm12  \n\t"
	"                                            \n\t"
	"vmulps           %%ymm0,  %%ymm4,  %%ymm6   \n\t"
	"vmulps           %%ymm0,  %%ymm5,  %%ymm7   \n\t"
	"vpermilps $0xb1, %%ymm0,  %%ymm0            \n\t"
	"vaddps           %%ymm6,  %%ymm11, %%ymm11  \n\t"
	"vaddps           %%ymm7,  %%ymm9,  %%ymm9   \n\t"
	"                                            \n\t"
	"vmulps           %%ymm1,  %%ymm4,  %%ymm6   \n\t"
	"vperm2f128 $0x3, %%ymm2,  %%ymm2,  %%ymm4   \n\t"
	"vmulps           %%ymm1,  %%ymm5,  %%ymm7   \n\t"
	"vperm2f128 $0x3, %%ymm3,  %%ymm3,  %%ymm5   \n\t"
	"vaddps           %%ymm6,  %%ymm10, %%ymm10  \n\t"
	"vaddps           %%ymm7,  %%ymm8,  %%ymm8   \n\t"
	"                                            \n\t"
	"vpermilps $0xb1, %%ymm1,  %%ymm1            \n\t"
	"vmulps           %%ymm0,  %%ymm2,  %%ymm6   \n\t"
	"vmulps           %%ymm0,  %%ymm3,  %%ymm7   \n\t"
	"vaddsubps        %%ymm6,  %%ymm15, %%ymm15  \n\t"
	"vaddsubps        %%ymm7,  %%ymm13, %%ymm13  \n\t"
	"                                            \n\t"
	"vmulps           %%ymm1,  %%ymm2,  %%ymm6   \n\t"
	"vmovsldup      1 * 32(%%rbx),      %%ymm2   \n\t"
	"vmulps           %%ymm1,  %%ymm3,  %%ymm7   \n\t"
	"vpermilps $0x4e, %%ymm2,  %%ymm3            \n\t"
	"vaddsubps        %%ymm6,  %%ymm14, %%ymm14  \n\t"
	"vaddsubps        %%ymm7,  %%ymm12, %%ymm12  \n\t"
	"                                            \n\t"
	"vmulps           %%ymm0,  %%ymm4,  %%ymm6   \n\t"
	"vmulps           %%ymm0,  %%ymm5,  %%ymm7   \n\t"
	"vmovaps        2 * 32(%%rax),      %%ymm0   \n\t"
	"vaddsubps        %%ymm6,  %%ymm11, %%ymm11  \n\t"
	"vaddsubps        %%ymm7,  %%ymm9,  %%ymm9   \n\t"
	"                                            \n\t"
	"vmulps           %%ymm1,  %%ymm4,  %%ymm6   \n\t"
	"vmulps           %%ymm1,  %%ymm5,  %%ymm7   \n\t"
	"vaddsubps        %%ymm6,  %%ymm10, %%ymm10  \n\t"
	"vaddsubps        %%ymm7,  %%ymm8,  %%ymm8   \n\t"
	"                                            \n\t"
	"                                            \n\t"
	"addq          $8 * 1 * 8, %%rax             \n\t" // a += 8 (1 x mr)
	"addq          $4 * 1 * 8, %%rbx             \n\t" // b += 4 (1 x nr)
	"                                            \n\t"
	"                                            \n\t"
	"decq   %%rsi                                \n\t" // i -= 1;
	"jne    .CLOOPKLEFT                          \n\t" // iterate again if i != 0.
	"                                            \n\t"
	"                                            \n\t"
	"                                            \n\t"
	".CPOSTACCUM:                                \n\t"
	"                                            \n\t"
	"                                            \n\t" // ymm15:  ymm13:  ymm11:  ymm9:
	"                                            \n\t" // ( ab00  ( ab01  ( ab02  ( ab03 
	"                                            \n\t" //   ab10    ab11    ab12    ab13 
	"                                            \n\t" //   ab21    ab20    ab23    ab22 
	"                                            \n\t" //   ab31    ab30    ab33    ab32 
	"                                            \n\t" //   ab42    ab43    ab40    ab41 
	"                                            \n\t" //   ab52    ab53    ab50    ab51 
	"                                            \n\t" //   ab63    ab62    ab61    ab60 
	"                                            \n\t" //   ab73 )  ab72 )  ab71 )  ab70 )
	"                                            \n\t"
	"                                            \n\t" // ymm14:  ymm12:  ymm10:  ymm8:
	"                                            \n\t" // ( ab80  ( ab81  ( ab82  ( ab83 
	"                                            \n\t" //   ab90    ab91    ab92    ab93 
	"                                            \n\t" //   aba1    aba0    aba3    aba2 
	"                                            \n\t" //   abb1    abb0    abb3    abb2 
	"                                            \n\t" //   abc2    abc3    abc0    abc1 
	"                                            \n\t" //   abd2    abd3    abd0    abd1 
	"                                            \n\t" //   abe3    abe2    abe1    abe0 
	"                                            \n\t" //   abf3    abf2    abf1    abf0 )
	"                                            \n\t"
	"vmovaps          %%ymm15, %%ymm7            \n\t"
	"vshufps   $0xe4, %%ymm13, %%ymm15, %%ymm15  \n\t"
	"vshufps   $0xe4, %%ymm7,  %%ymm13, %%ymm13  \n\t"
	"                                            \n\t"
	"vmovaps          %%ymm11, %%ymm7            \n\t"
	"vshufps   $0xe4, %%ymm9,  %%ymm11, %%ymm11  \n\t"
	"vshufps   $0xe4, %%ymm7,  %%ymm9,  %%ymm9   \n\t"
	"                                            \n\t"
	"vmovaps          %%ymm14, %%ymm7            \n\t"
	"vshufps   $0xe4, %%ymm12, %%ymm14, %%ymm14  \n\t"
	"vshufps   $0xe4, %%ymm7,  %%ymm12, %%ymm12  \n\t"
	"                                            \n\t"
	"vmovaps          %%ymm10, %%ymm7            \n\t"
	"vshufps   $0xe4, %%ymm8,  %%ymm10, %%ymm10  \n\t"
	"vshufps   $0xe4, %%ymm7,  %%ymm8,  %%ymm8   \n\t"
	"                                            \n\t"
	"                                            \n\t" // ymm15:  ymm13:  ymm11:  ymm9:
	"                                            \n\t" // ( ab00  ( ab01  ( ab02  ( ab03 
	"                                            \n\t" //   ab10    ab11    ab12    ab13 
	"                                            \n\t" //   ab20    ab21    ab22    ab23 
	"                                            \n\t" //   ab30    ab31    ab32    ab33 
	"                                            \n\t" //   ab42    ab43    ab40    ab41 
	"                                            \n\t" //   ab52    ab53    ab50    ab51 
	"                                            \n\t" //   ab62    ab63    ab60    ab61 
	"                                            \n\t" //   ab72 )  ab73 )  ab70 )  ab71 )
	"                                            \n\t"
	"                                            \n\t" // ymm14:  ymm12:  ymm10:  ymm8:
	"                                            \n\t" // ( ab80  ( ab81  ( ab82  ( ab83 
	"                                            \n\t" //   ab90    ab91    ab92    ab93 
	"                                            \n\t" //   aba0    aba1    aba2    aba3 
	"                                            \n\t" //   abb0    abb1    abb2    abb3 
	"                                            \n\t" //   abc2    abc3    abc0    abc1 
	"                                            \n\t" //   abd2    abd3    abd0    abd1 
	"                                            \n\t" //   abe2    abe3    abe0    abe1 
	"                                            \n\t" //   abf2 )  abf3 )  abf0 )  abf1 )
	"                                            \n\t"
	"vmovaps           %%ymm15, %%ymm7           \n\t"
	"vperm2f128 $0x12, %%ymm15, %%ymm11, %%ymm15 \n\t"
	"vperm2f128 $0x30, %%ymm7,  %%ymm11, %%ymm11 \n\t"
	"                                            \n\t"
	"vmovaps           %%ymm13, %%ymm7           \n\t"
	"vperm2f128 $0x12, %%ymm13, %%ymm9,  %%ymm13 \n\t"
	"vperm2f128 $0x30, %%ymm7,  %%ymm9,  %%ymm9  \n\t"
	"                                            \n\t"
	"vmovaps           %%ymm14, %%ymm7           \n\t"
	"vperm2f128 $0x12, %%ymm14, %%ymm10, %%ymm14 \n\t"
	"vperm2f128 $0x30, %%ymm7,  %%ymm10, %%ymm10 \n\t"
	"                                            \n\t"
	"vmovaps           %%ymm12, %%ymm7           \n\t"
	"vperm2f128 $0x12, %%ymm12, %%ymm8,  %%ymm12 \n\t"
	"vperm2f128 $0x30, %%ymm7,  %%ymm8,  %%ymm8  \n\t"
	"                                            \n\t"
	"                                            \n\t" // ymm15:  ymm13:  ymm11:  ymm9:
	"                                            \n\t" // ( ab00  ( ab01  ( ab02  ( ab03 
	"                                            \n\t" //   ab10    ab11    ab12    ab13 
	"                                            \n\t" //   ab20    ab21    ab22    ab23 
	"                                            \n\t" //   ab30    ab31    ab32    ab33 
	"                                            \n\t" //   ab40    ab41    ab42    ab43 
	"                                            \n\t" //   ab50    ab51    ab52    ab53 
	"                                            \n\t" //   ab60    ab61    ab62    ab63 
	"                                            \n\t" //   ab70 )  ab71 )  ab72 )  ab73 )
	"                                            \n\t"
	"                                            \n\t" // ymm14:  ymm12:  ymm10:  ymm8:
	"                                            \n\t" // ( ab80  ( ab81  ( ab82  ( ab83 
	"                                            \n\t" //   ab90    ab91    ab92    ab93 
	"                                            \n\t" //   aba0    aba1    aba2    aba3 
	"                                            \n\t" //   abb0    abb1    abb2    abb3 
	"                                            \n\t" //   abc0    abc1    abc2    abc3 
	"                                            \n\t" //   abd0    abd1    abd2    abd3 
	"                                            \n\t" //   abe0    abe1    abe2    abe3 
	"                                            \n\t" //   abf0 )  abf1 )  abf2 )  abf3 )
	"                                            \n\t"
	"                                            \n\t"
	"                                            \n\t"
	"                                            \n\t"
	"                                            \n\t" // scale by alpha
	"                                            \n\t"
	"movq         %4, %%rax                      \n\t" // load address of alpha
	"vbroadcastss    (%%rax), %%ymm7             \n\t" // load alpha_r and duplicate
	"vbroadcastss   4(%%rax), %%ymm6             \n\t" // load alpha_i and duplicate
	"                                            \n\t"
	"vpermilps $0xb1, %%ymm15, %%ymm3            \n\t"
	"vmulps           %%ymm7,  %%ymm15, %%ymm15  \n\t"
	"vmulps           %%ymm6,  %%ymm3,  %%ymm3   \n\t"
	"vaddsubps        %%ymm3,  %%ymm15, %%ymm15  \n\t"
	"                                            \n\t"
	"vpermilps $0xb1, %%ymm14, %%ymm2            \n\t"
	"vmulps           %%ymm7,  %%ymm14, %%ymm14  \n\t"
	"vmulps           %%ymm6,  %%ymm2,  %%ymm2   \n\t"
	"vaddsubps        %%ymm2,  %%ymm14, %%ymm14  \n\t"
	"                                            \n\t"
	"vpermilps $0xb1, %%ymm13, %%ymm1            \n\t"
	"vmulps           %%ymm7,  %%ymm13, %%ymm13  \n\t"
	"vmulps           %%ymm6,  %%ymm1,  %%ymm1   \n\t"
	"vaddsubps        %%ymm1,  %%ymm13, %%ymm13  \n\t"
	"                                            \n\t"
	"vpermilps $0xb1, %%ymm12, %%ymm0            \n\t"
	"vmulps           %%ymm7,  %%ymm12, %%ymm12  \n\t"
	"vmulps           %%ymm6,  %%ymm0,  %%ymm0   \n\t"
	"vaddsubps        %%ymm0,  %%ymm12, %%ymm12  \n\t"
	"                                            \n\t"
	"vpermilps $0xb1, %%ymm11, %%ymm3            \n\t"
	"vmulps           %%ymm7,  %%ymm11, %%ymm11  \n\t"
	"vmulps           %%ymm6,  %%ymm3,  %%ymm3   \n\t"
	"vaddsubps        %%ymm3,  %%ymm11, %%ymm11  \n\t"
	"                                            \n\t"
	"vpermilps $0xb1, %%ymm10, %%ymm2            \n\t"
	"vmulps           %%ymm7,  %%ymm10, %%ymm10  \n\t"
	"vmulps           %%ymm6,  %%ymm2,  %%ymm2   \n\t"
	"vaddsubps        %%ymm2,  %%ymm10, %%ymm10  \n\t"
	"                                            \n\t"
	"vpermilps $0xb1, %%ymm9,  %%ymm1            \n\t"
	"vmulps           %%ymm7,  %%ymm9,  %%ymm9   \n\t"
	"vmulps           %%ymm6,  %%ymm1,  %%ymm1   \n\t"
	"vaddsubps        %%ymm1,  %%ymm9,  %%ymm9   \n\t"
	"                                            \n\t"
	"vpermilps $0xb1, %%ymm8,  %%ymm0            \n\t"
	"vmulps           %%ymm7,  %%ymm8,  %%ymm8   \n\t"
	"vmulps           %%ymm6,  %%ymm0,  %%ymm0   \n\t"
	"vaddsubps        %%ymm0,  %%ymm8,  %%ymm8   \n\t"
	"                                            \n\t"
	"                                            \n\t"
	"                                            \n\t"
	"                                            \n\t"
	"movq         %5, %%rbx                      \n\t" // load address of beta 
	"vbroadcastss    (%%rbx), %%ymm7             \n\t" // load beta_r and duplicate
	"vbroadcastss   4(%%rbx), %%ymm6             \n\t" // load beta_i and duplicate
	"                                            \n\t"
	"                                            \n\t"
	"                                            \n\t"
	"                                            \n\t"
	"                                            \n\t"
	"                                            \n\t"
	"                                            \n\t"
	"movq                %7, %%rsi               \n\t" // load rs_c
	"leaq        (,%%rsi,8), %%rsi               \n\t" // rsi = rs_c * sizeof(scomplex)
	"                                            \n\t"
	"leaq   (%%rcx,%%rsi,4), %%rdx               \n\t" // load address of c + 4*rs_c;
	"                                            \n\t"
	"leaq        (,%%rsi,2), %%r12               \n\t" // r12 = 2*rs_c;
	"leaq   (%%r12,%%rsi,1), %%r13               \n\t" // r13 = 3*rs_c;
	"                                            \n\t"
	"                                            \n\t"
	"                                            \n\t"
	"                                            \n\t" // determine if
	"                                            \n\t" //    c    % 32 == 0, AND
	"                                            \n\t" //  8*cs_c % 32 == 0, AND
	"                                            \n\t" //    rs_c      == 1
	"                                            \n\t" // ie: aligned, ldim aligned, and
	"                                            \n\t" // column-stored
	"                                            \n\t"
	"cmpq       $8, %%rsi                        \n\t" // set ZF if (8*rs_c) == 8.
	"sete           %%bl                         \n\t" // bl = ( ZF == 1 ? 1 : 0 );
	"testq     $31, %%rcx                        \n\t" // set ZF if c & 32 is zero.
	"setz           %%bh                         \n\t" // bh = ( ZF == 0 ? 1 : 0 );
	"testq     $31, %%rdi                        \n\t" // set ZF if (8*cs_c) & 32 is zero.
	"setz           %%al                         \n\t" // al = ( ZF == 0 ? 1 : 0 );
	"                                            \n\t" // and(bl,bh) followed by
	"                                            \n\t" // and(bh,al) will reveal result
	"                                            \n\t"
	"                                            \n\t" // now avoid loading C if beta == 0
	"                                            \n\t"
	"vxorps    %%ymm0,  %%ymm0,  %%ymm0          \n\t" // set ymm0 to zero.
	"vucomiss  %%xmm0,  %%xmm7                   \n\t" // set ZF if beta_r == 0.
	"sete       %%r8b                            \n\t" // r8b = ( ZF == 1 ? 1 : 0 );
	"vucomiss  %%xmm0,  %%xmm6                   \n\t" // set ZF if beta_i == 0.
	"sete       %%r9b                            \n\t" // r9b = ( ZF == 1 ? 1 : 0 );
	"andb       %%r8b, %%r9b                     \n\t" // set ZF if r8b & r9b == 1.
	"jne      .CBETAZERO                         \n\t" // if ZF = 0, jump to beta == 0 case
	"                                            \n\t"
	"                                            \n\t"
	"                                            \n\t" // check if aligned/column-stored
	"andb     %%bl, %%bh                         \n\t" // set ZF if bl & bh == 1.
	"andb     %%bh, %%al                         \n\t" // set ZF if bh & al == 1.
	"jne     .CCOLSTORED                         \n\t" // jump to column storage case
	"                                            \n\t"
	"                                            \n\t"
	"                                            \n\t"
	".CGENSTORED:                                \n\t"
	"                                            \n\t"
	"                                            \n\t" // update c00:c70
	"                                            \n\t"
	"vmovlpd    (%%rcx),       %%xmm0,  %%xmm0   \n\t" // load (c00,10) into xmm0[0:1]
	"vmovhpd    (%%rcx,%%rsi), %%xmm0,  %%xmm0   \n\t" // load (c20,30) into xmm0[2:3]
	"vmovlpd    (%%rcx,%%r12), %%xmm2,  %%xmm2   \n\t" // load (c40,50) into xmm2[0:1]
	"vmovhpd    (%%rcx,%%r13), %%xmm2,  %%xmm2   \n\t" // load (c60,70) into xmm2[2:3]
	"vinsertf128  $1, %%xmm2,  %%ymm0,  %%ymm0   \n\t" // ymm0 := (ymm0[0:3],xmm2)
	"vpermilps $0xb1, %%ymm0,  %%ymm2            \n\t" // scale ymm0 by beta
	"vmulps           %%ymm7,  %%ymm0,  %%ymm0   \n\t"
	"vmulps           %%ymm6,  %%ymm2,  %%ymm2   \n\t"
	"vaddsubps        %%ymm2,  %%ymm0,  %%ymm0   \n\t"
	"vaddps           %%ymm15, %%ymm0,  %%ymm0   \n\t" // add the gemm result to ymm0
	"vextractf128 $1, %%ymm0,  %%xmm2            \n\t" // xmm2 := ymm0[4:7]
	"vmovlpd          %%xmm0,  (%%rcx)           \n\t" // store (c00,c10)
	"vmovhpd          %%xmm0,  (%%rcx,%%rsi)     \n\t" // store (c20,c30)
	"vmovlpd          %%xmm2,  (%%rcx,%%r12)     \n\t" // store (c40,c50)
	"vmovhpd          %%xmm2,  (%%rcx,%%r13)     \n\t" // store (c60,c70)
	"addq      %%rdi, %%rcx                      \n\t" // c += cs_c;
	"                                            \n\t"
	"                                            \n\t" // update c80:cf0
	"                                            \n\t"
	"vmovlpd    (%%rdx),       %%xmm0,  %%xmm0   \n\t" // load (c80,90) into xmm0[0:1]
	"vmovhpd    (%%rdx,%%rsi), %%xmm0,  %%xmm0   \n\t" // load (ca0,b0) into xmm0[2:3]
	"vmovlpd    (%%rdx,%%r12), %%xmm2,  %%xmm2   \n\t" // load (cc0,d0) into xmm2[0:1]
	"vmovhpd    (%%rdx,%%r13), %%xmm2,  %%xmm2   \n\t" // load (ce0,f0) into xmm2[2:3]
	"vinsertf128  $1, %%xmm2,  %%ymm0,  %%ymm0   \n\t" // ymm0 := (ymm0[0:3],xmm2)
	"vpermilps $0xb1, %%ymm0,  %%ymm2            \n\t" // scale ymm0 by beta
	"vmulps           %%ymm7,  %%ymm0,  %%ymm0   \n\t"
	"vmulps           %%ymm6,  %%ymm2,  %%ymm2   \n\t"
	"vaddsubps        %%ymm2,  %%ymm0,  %%ymm0   \n\t"
	"vaddps           %%ymm14, %%ymm0,  %%ymm0   \n\t" // add the gemm result to ymm0
	"vextractf128 $1, %%ymm0,  %%xmm2            \n\t" // xmm2 := ymm0[4:7]
	"vmovlpd          %%xmm0,  (%%rdx)           \n\t" // store (c80,c90)
	"vmovhpd          %%xmm0,  (%%rdx,%%rsi)     \n\t" // store (ca0,cb0)
	"vmovlpd          %%xmm2,  (%%rdx,%%r12)     \n\t" // store (cc0,cd0)
	"vmovhpd          %%xmm2,  (%%rdx,%%r13)     \n\t" // store (ce0,cf0)
	"addq      %%rdi, %%rdx                      \n\t" // c += cs_c;
	"                                            \n\t"
	"                                            \n\t" // update c01:c71
	"                                            \n\t"
	"vmovlpd    (%%rcx),       %%xmm0,  %%xmm0   \n\t" // load (c01,11) into xmm0[0:1]
	"vmovhpd    (%%rcx,%%rsi), %%xmm0,  %%xmm0   \n\t" // load (c21,31) into xmm0[2:3]
	"vmovlpd    (%%rcx,%%r12), %%xmm2,  %%xmm2   \n\t" // load (c41,51) into xmm2[0:1]
	"vmovhpd    (%%rcx,%%r13), %%xmm2,  %%xmm2   \n\t" // load (c61,71) into xmm2[2:3]
	"vinsertf128  $1, %%xmm2,  %%ymm0,  %%ymm0   \n\t" // ymm0 := (ymm0[0:3],xmm2)
	"vpermilps $0xb1, %%ymm0,  %%ymm2            \n\t" // scale ymm0 by beta
	"vmulps           %%ymm7,  %%ymm0,  %%ymm0   \n\t"
	"vmulps           %%ymm6,  %%ymm2,  %%ymm2   \n\t"
	"vaddsubps        %%ymm2,  %%ymm0,  %%ymm0   \n\t"
	"vaddps           %%ymm13, %%ymm0,  %%ymm0   \n\t" // add the gemm result to ymm0
	"vextractf128 $1, %%ymm0,  %%xmm2            \n\t" // xmm2 := ymm0[4:7]
	"vmovlpd          %%xmm0,  (%%rcx)           \n\t" // store (c01,c11)
	"vmovhpd          %%xmm0,  (%%rcx,%%rsi)     \n\t" // store (c21,c31)
	"vmovlpd          %%xmm2,  (%%rcx,%%r12)     \n\t" // store (c41,c51)
	"vmovhpd          %%xmm2,  (%%rcx,%%r13)     \n\t" // store (c61,c71)
	"addq      %%rdi, %%rcx                      \n\t" // c += cs_c;
	"                                            \n\t"
	"                                            \n\t" // update c81:cf1
	"                                            \n\t"
	"vmovlpd    (%%rdx),       %%xmm0,  %%xmm0   \n\t" // load (c81,91) into xmm0[0:1]
	"vmovhpd    (%%rdx,%%rsi), %%xmm0,  %%xmm0   \n\t" // load (ca1,b1) into xmm0[2:3]
	"vmovlpd    (%%rdx,%%r12), %%xmm2,  %%xmm2   \n\t" // load (cc1,d1) into xmm2[0:1]
	"vmovhpd    (%%rdx,%%r13), %%xmm2,  %%xmm2   \n\t" // load (ce1,f1) into xmm2[2:3]
	"vinsertf128  $1, %%xmm2,  %%ymm0,  %%ymm0   \n\t" // ymm0 := (ymm0[0:3],xmm2)
	"vpermilps $0xb1, %%ymm0,  %%ymm2            \n\t" // scale ymm0 by beta
	"vmulps           %%ymm7,  %%ymm0,  %%ymm0   \n\t"
	"vmulps           %%ymm6,  %%ymm2,  %%ymm2   \n\t"
	"vaddsubps        %%ymm2,  %%ymm0,  %%ymm0   \n\t"
	"vaddps           %%ymm12, %%ymm0,  %%ymm0   \n\t" // add the gemm result to ymm0
	"vextractf128 $1, %%ymm0,  %%xmm2            \n\t" // xmm2 := ymm0[4:7]
	"vmovlpd          %%xmm0,  (%%rdx)           \n\t" // store (c81,c91)
	"vmovhpd          %%xmm0,  (%%rdx,%%rsi)     \n\t" // store (ca1,cb1)
	"vmovlpd          %%xmm2,  (%%rdx,%%r12)     \n\t" // store (cc1,cd1)
	"vmovhpd          %%xmm2,  (%%rdx,%%r13)     \n\t" // store (ce1,cf1)
	"addq      %%rdi, %%rdx                      \n\t" // c += cs_c;
	"                                            \n\t"
	"                                            \n\t" // update c02:c72
	"                                            \n\t"
	"vmovlpd    (%%rcx),       %%xmm0,  %%xmm0   \n\t" // load (c02,12) into xmm0[0:1]
	"vmovhpd    (%%rcx,%%rsi), %%xmm0,  %%xmm0   \n\t" // load (c22,32) into xmm0[2:3]
	"vmovlpd    (%%rcx,%%r12), %%xmm2,  %%xmm2   \n\t" // load (c42,52) into xmm2[0:1]
	"vmovhpd    (%%rcx,%%r13), %%xmm2,  %%xmm2   \n\t" // load (c62,72) into xmm2[2:3]
	"vinsertf128  $1, %%xmm2,  %%ymm0,  %%ymm0   \n\t" // ymm0 := (ymm0[0:3],xmm2)
	"vpermilps $0xb1, %%ymm0,  %%ymm2            \n\t" // scale ymm0 by beta
	"vmulps           %%ymm7,  %%ymm0,  %%ymm0   \n\t"
	"vmulps           %%ymm6,  %%ymm2,  %%ymm2   \n\t"
	"vaddsubps        %%ymm2,  %%ymm0,  %%ymm0   \n\t"
	"vaddps           %%ymm11, %%ymm0,  %%ymm0   \n\t" // add the gemm result to ymm0
	"vextractf128 $1, %%ymm0,  %%xmm2            \n\t" // xmm2 := ymm0[4:7]
	"vmovlpd          %%xmm0,  (%%rcx)           \n\t" // store (c02,c12)
	"vmovhpd          %%xmm0,  (%%rcx,%%rsi)     \n\t" // store (c22,c32)
	"vmovlpd          %%xmm2,  (%%rcx,%%r12)     \n\t" // store (c42,c52)
	"vmovhpd          %%xmm2,  (%%rcx,%%r13)     \n\t" // store (c62,c72)
	"addq      %%rdi, %%rcx                      \n\t" // c += cs_c;
	"                                            \n\t"
	"                                            \n\t" // update c82:cf2
	"                                            \n\t"
	"vmovlpd    (%%rdx),       %%xmm0,  %%xmm0   \n\t" // load (c82,92) into xmm0[0:1]
	"vmovhpd    (%%rdx,%%rsi), %%xmm0,  %%xmm0   \n\t" // load (ca2,b2) into xmm0[2:3]
	"vmovlpd    (%%rdx,%%r12), %%xmm2,  %%xmm2   \n\t" // load (cc2,d2) into xmm2[0:1]
	"vmovhpd    (%%rdx,%%r13), %%xmm2,  %%xmm2   \n\t" // load (ce2,f2) into xmm2[2:3]
	"vinsertf128  $1, %%xmm2,  %%ymm0,  %%ymm0   \n\t" // ymm0 := (ymm0[0:3],xmm2)
	"vpermilps $0xb1, %%ymm0,  %%ymm2            \n\t" // scale ymm0 by beta
	"vmulps           %%ymm7,  %%ymm0,  %%ymm0   \n\t"
	"vmulps           %%ymm6,  %%ymm2,  %%ymm2   \n\t"
	"vaddsubps        %%ymm2,  %%ymm0,  %%ymm0   \n\t"
	"vaddps           %%ymm10, %%ymm0,  %%ymm0   \n\t" // add the gemm result to ymm0
	"vextractf128 $1, %%ymm0,  %%xmm2            \n\t" // xmm2 := ymm0[4:7]
	"vmovlpd          %%xmm0,  (%%rdx)           \n\t" // store (c82,c92)
	"vmovhpd          %%xmm0,  (%%rdx,%%rsi)     \n\t" // store (ca2,cb2)
	"vmovlpd          %%xmm2,  (%%rdx,%%r12)     \n\t" // store (cc2,cd2)
	"vmovhpd          %%xmm2,  (%%rdx,%%r13)     \n\t" // store (ce2,cf2)
	"addq      %%rdi, %%rdx                      \n\t" // c += cs_c;
	"                                            \n\t"
	"                                            \n\t" // update c03:c73
	"                                            \n\t"
	"vmovlpd    (%%rcx),       %%xmm0,  %%xmm0   \n\t" // load (c03,13) into xmm0[0:1]
	"vmovhpd    (%%rcx,%%rsi), %%xmm0,  %%xmm0   \n\t" // load (c23,33) into xmm0[2:3]
	"vmovlpd    (%%rcx,%%r12), %%xmm2,  %%xmm2   \n\t" // load (c43,53) into xmm2[0:1]
	"vmovhpd    (%%rcx,%%r13), %%xmm2,  %%xmm2   \n\t" // load (c63,73) into xmm2[2:3]
	"vinsertf128  $1, %%xmm2,  %%ymm0,  %%ymm0   \n\t" // ymm0 := (ymm0[0:3],xmm2)
	"vpermilps $0xb1, %%ymm0,  %%ymm2            \n\t" // scale ymm0 by beta
	"vmulps           %%ymm7,  %%ymm0,  %%ymm0   \n\t"
	"vmulps           %%ymm6,  %%ymm2,  %%ymm2   \n\t"
	"vaddsubps        %%ymm2,  %%ymm0,  %%ymm0   \n\t"
	"vaddps           %%ymm9,  %%ymm0,  %%ymm0   \n\t" // add the gemm result to ymm0
	"vextractf128 $1, %%ymm0,  %%xmm2            \n\t" // xmm2 := ymm0[4:7]
	"vmovlpd          %%xmm0,  (%%rcx)           \n\t" // store (c03,c13)
	"vmovhpd          %%xmm0,  (%%rcx,%%rsi)     \n\t" // store (c23,c33)
	"vmovlpd          %%xmm2,  (%%rcx,%%r12)     \n\t" // store (c43,c53)
	"vmovhpd          %%xmm2,  (%%rcx,%%r13)     \n\t" // store (c63,c73)
	"addq      %%rdi, %%rcx                      \n\t" // c += cs_c;
	"                                            \n\t"
	"                                            \n\t" // update c83:cf3
	"                                            \n\t"
	"vmovlpd    (%%rdx),       %%xmm0,  %%xmm0   \n\t" // load (c83,93) into xmm0[0:1]
	"vmovhpd    (%%rdx,%%rsi), %%xmm0,  %%xmm0   \n\t" // load (ca3,b3) into xmm0[2:3]
	"vmovlpd    (%%rdx,%%r12), %%xmm2,  %%xmm2   \n\t" // load (cc3,d3) into xmm2[0:1]
	"vmovhpd    (%%rdx,%%r13), %%xmm2,  %%xmm2   \n\t" // load (ce3,f3) into xmm2[2:3]
	"vinsertf128  $1, %%xmm2,  %%ymm0,  %%ymm0   \n\t" // ymm0 := (ymm0[0:3],xmm2)
	"vpermilps $0xb1, %%ymm0,  %%ymm2            \n\t" // scale ymm0 by beta
	"vmulps           %%ymm7,  %%ymm0,  %%ymm0   \n\t"
	"vmulps           %%ymm6,  %%ymm2,  %%ymm2   \n\t"
	"vaddsubps        %%ymm2,  %%ymm0,  %%ymm0   \n\t"
	"vaddps           %%ymm8,  %%ymm0,  %%ymm0   \n\t" // add the gemm result to ymm0
	"vextractf128 $1, %%ymm0,  %%xmm2            \n\t" // xmm2 := ymm0[4:7]
	"vmovlpd          %%xmm0,  (%%rdx)           \n\t" // store (c83,c93)
	"vmovhpd          %%xmm0,  (%%rdx,%%rsi)     \n\t" // store (ca3,cb3)
	"vmovlpd          %%xmm2,  (%%rdx,%%r12)     \n\t" // store (cc3,cd3)
	"vmovhpd          %%xmm2,  (%%rdx,%%r13)     \n\t" // store (ce3,cf3)
	"addq      %%rdi, %%rdx                      \n\t" // c += cs_c;
	"                                            \n\t"
	"                                            \n\t"
	"                                            \n\t"
	"jmp    .CDONE                               \n\t" // jump to end.
	"                                            \n\t"
	"                                            \n\t"
	"                                            \n\t"
	".CCOLSTORED:                                \n\t"
	"                                            \n\t"
	"                                            \n\t" // update c00:c70
	"                                            \n\t"
	"vmovaps    (%%rcx),       %%ymm0            \n\t" // load c00:c70 into ymm0
	"vpermilps $0xb1, %%ymm0,  %%ymm2            \n\t" // scale ymm0 by beta
	"vmulps           %%ymm7,  %%ymm0,  %%ymm0   \n\t"
	"vmulps           %%ymm6,  %%ymm2,  %%ymm2   \n\t"
	"vaddsubps        %%ymm2,  %%ymm0,  %%ymm0   \n\t"
	"vaddps           %%ymm15, %%ymm0,  %%ymm0   \n\t" // add the gemm result to ymm0
	"vmovaps          %%ymm0,  (%%rcx)           \n\t" // store c00:c70
	"addq      %%rdi, %%rcx                      \n\t" // c += cs_c;
	"                                            \n\t"
	"                                            \n\t" // update c80:cf0
	"                                            \n\t"
	"vmovaps    (%%rdx),       %%ymm0            \n\t" // load c80:f0 into ymm0
	"vpermilps $0xb1, %%ymm0,  %%ymm2            \n\t" // scale ymm0 by beta
	"vmulps           %%ymm7,  %%ymm0,  %%ymm0   \n\t"
	"vmulps           %%ymm6,  %%ymm2,  %%ymm2   \n\t"
	"vaddsubps        %%ymm2,  %%ymm0,  %%ymm0   \n\t"
	"vaddps           %%ymm14, %%ymm0,  %%ymm0   \n\t" // add the gemm result to ymm0
	"vmovaps          %%ymm0,  (%%rdx)           \n\t" // store c80:cf0
	"addq      %%rdi, %%rdx                      \n\t" // c += cs_c;
	"                                            \n\t"
	"                                            \n\t" // update c00:c70
	"                                            \n\t"
	"vmovaps    (%%rcx),       %%ymm0            \n\t" // load c01:c71 into ymm0
	"vpermilps $0xb1, %%ymm0,  %%ymm2            \n\t" // scale ymm0 by beta
	"vmulps           %%ymm7,  %%ymm0,  %%ymm0   \n\t"
	"vmulps           %%ymm6,  %%ymm2,  %%ymm2   \n\t"
	"vaddsubps        %%ymm2,  %%ymm0,  %%ymm0   \n\t"
	"vaddps           %%ymm13, %%ymm0,  %%ymm0   \n\t" // add the gemm result to ymm0
	"vmovaps          %%ymm0,  (%%rcx)           \n\t" // store c01:c71
	"addq      %%rdi, %%rcx                      \n\t" // c += cs_c;
	"                                            \n\t"
	"                                            \n\t" // update c81:cf1
	"                                            \n\t"
	"vmovaps    (%%rdx),       %%ymm0            \n\t" // load c81:f1 into ymm0
	"vpermilps $0xb1, %%ymm0,  %%ymm2            \n\t" // scale ymm0 by beta
	"vmulps           %%ymm7,  %%ymm0,  %%ymm0   \n\t"
	"vmulps           %%ymm6,  %%ymm2,  %%ymm2   \n\t"
	"vaddsubps        %%ymm2,  %%ymm0,  %%ymm0   \n\t"
	"vaddps           %%ymm12, %%ymm0,  %%ymm0   \n\t" // add the gemm result to ymm0
	"vmovaps          %%ymm0,  (%%rdx)           \n\t" // store c81:cf1
	"addq      %%rdi, %%rdx                      \n\t" // c += cs_c;
	"                                            \n\t"
	"                                            \n\t" // update c02:c72
	"                                            \n\t"
	"vmovaps    (%%rcx),       %%ymm0            \n\t" // load c02:c72 into ymm0
	"vpermilps $0xb1, %%ymm0,  %%ymm2            \n\t" // scale ymm0 by beta
	"vmulps           %%ymm7,  %%ymm0,  %%ymm0   \n\t"
	"vmulps           %%ymm6,  %%ymm2,  %%ymm2   \n\t"
	"vaddsubps        %%ymm2,  %%ymm0,  %%ymm0   \n\t"
	"vaddps           %%ymm11, %%ymm0,  %%ymm0   \n\t" // add the gemm result to ymm0
	"vmovaps          %%ymm0,  (%%rcx)           \n\t" // store c02:c72
	"addq      %%rdi, %%rcx                      \n\t" // c += cs_c;
	"                                            \n\t"
	"                                            \n\t" // update c82:cf2
	"                                            \n\t"
	"vmovaps    (%%rdx),       %%ymm0            \n\t" // load c82:f2 into ymm0
	"vpermilps $0xb1, %%ymm0,  %%ymm2            \n\t" // scale ymm0 by beta
	"vmulps           %%ymm7,  %%ymm0,  %%ymm0   \n\t"
	"vmulps           %%ymm6,  %%ymm2,  %%ymm2   \n\t"
	"vaddsubps        %%ymm2,  %%ymm0,  %%ymm0   \n\t"
	"vaddps           %%ymm10, %%ymm0,  %%ymm0   \n\t" // add the gemm result to ymm0
	"vmovaps          %%ymm0,  (%%rdx)           \n\t" // store c82:cf2
	"addq      %%rdi, %%rdx                      \n\t" // c += cs_c;
	"                                            \n\t"
	"                                            \n\t" // update c03:c73
	"                                            \n\t"
	"vmovaps    (%%rcx),       %%ymm0            \n\t" // load c03:c73 into ymm0
	"vpermilps $0xb1, %%ymm0,  %%ymm2            \n\t" // scale ymm0 by beta
	"vmulps           %%ymm7,  %%ymm0,  %%ymm0   \n\t"
	"vmulps           %%ymm6,  %%ymm2,  %%ymm2   \n\t"
	"vaddsubps        %%ymm2,  %%ymm0,  %%ymm0   \n\t"
	"vaddps           %%ymm9,  %%ymm0,  %%ymm0   \n\t" // add the gemm result to ymm0
	"vmovaps          %%ymm0,  (%%rcx)           \n\t" // store c03:c73
	"addq      %%rdi, %%rcx                      \n\t" // c += cs_c;
	"                                            \n\t"
	"                                            \n\t" // update c83:cf3
	"                                            \n\t"
	"vmovaps    (%%rdx),       %%ymm0            \n\t" // load c83:f3 into ymm0
	"vpermilps $0xb1, %%ymm0,  %%ymm2            \n\t" // scale ymm0 by beta
	"vmulps           %%ymm7,  %%ymm0,  %%ymm0   \n\t"
	"vmulps           %%ymm6,  %%ymm2,  %%ymm2   \n\t"
	"vaddsubps        %%ymm2,  %%ymm0,  %%ymm0   \n\t"
	"vaddps           %%ymm8,  %%ymm0,  %%ymm0   \n\t" // add the gemm result to ymm0
	"vmovaps          %%ymm0,  (%%rdx)           \n\t" // store c83:cf3
	"addq      %%rdi, %%rdx                      \n\t" // c += cs_c;
	"                                            \n\t"
	"                                            \n\t"
	"                                            \n\t"
	"jmp    .CDONE                               \n\t" // jump to end.
	"                                            \n\t"
	"                                            \n\t"
	"                                            \n\t"
	".CBETAZERO:                                 \n\t"
	"                                            \n\t" // check if aligned/column-stored
	"                                            \n\t" // check if aligned/column-stored
	"andb     %%bl, %%bh                         \n\t" // set ZF if bl & bh == 1.
	"andb     %%bh, %%al                         \n\t" // set ZF if bh & al == 1.
	"jne     .CCOLSTORBZ                         \n\t" // jump to column storage case
	"                                            \n\t"
	"                                            \n\t"
	"                                            \n\t"
	".CGENSTORBZ:                                \n\t"
	"                                            \n\t"
	"                                            \n\t" // update c00:c70
	"                                            \n\t"
	"vextractf128 $1, %%ymm15, %%xmm2            \n\t" // xmm2 := ymm0[4:7]
	"vmovlpd          %%xmm15, (%%rcx)           \n\t" // store (c00,c10)
	"vmovhpd          %%xmm15, (%%rcx,%%rsi)     \n\t" // store (c20,c30)
	"vmovlpd          %%xmm2,  (%%rcx,%%r12)     \n\t" // store (c40,c50)
	"vmovhpd          %%xmm2,  (%%rcx,%%r13)     \n\t" // store (c60,c70)
	"addq      %%rdi, %%rcx                      \n\t" // c += cs_c;
	"                                            \n\t"
	"                                            \n\t" // update c80:cf0
	"                                            \n\t"
	"vextractf128 $1, %%ymm14, %%xmm2            \n\t" // xmm2 := ymm0[4:7]
	"vmovlpd          %%xmm14, (%%rdx)           \n\t" // store (c80,c90)
	"vmovhpd          %%xmm14, (%%rdx,%%rsi)     \n\t" // store (ca0,cb0)
	"vmovlpd          %%xmm2,  (%%rdx,%%r12)     \n\t" // store (cc0,cd0)
	"vmovhpd          %%xmm2,  (%%rdx,%%r13)     \n\t" // store (ce0,cf0)
	"addq      %%rdi, %%rdx                      \n\t" // c += cs_c;
	"                                            \n\t"
	"                                            \n\t" // update c01:c71
	"                                            \n\t"
	"vextractf128 $1, %%ymm13, %%xmm2            \n\t" // xmm2 := ymm0[4:7]
	"vmovlpd          %%xmm13, (%%rcx)           \n\t" // store (c01,c11)
	"vmovhpd          %%xmm13, (%%rcx,%%rsi)     \n\t" // store (c21,c31)
	"vmovlpd          %%xmm2,  (%%rcx,%%r12)     \n\t" // store (c41,c51)
	"vmovhpd          %%xmm2,  (%%rcx,%%r13)     \n\t" // store (c61,c71)
	"addq      %%rdi, %%rcx                      \n\t" // c += cs_c;
	"                                            \n\t"
	"                                            \n\t" // update c81:cf1
	"                                            \n\t"
	"vextractf128 $1, %%ymm12, %%xmm2            \n\t" // xmm2 := ymm0[4:7]
	"vmovlpd          %%xmm12, (%%rdx)           \n\t" // store (c81,c91)
	"vmovhpd          %%xmm12, (%%rdx,%%rsi)     \n\t" // store (ca1,cb1)
	"vmovlpd          %%xmm2,  (%%rdx,%%r12)     \n\t" // store (cc1,cd1)
	"vmovhpd          %%xmm2,  (%%rdx,%%r13)     \n\t" // store (ce1,cf1)
	"addq      %%rdi, %%rdx                      \n\t" // c += cs_c;
	"                                            \n\t"
	"                                            \n\t" // update c02:c72
	"                                            \n\t"
	"vextractf128 $1, %%ymm11, %%xmm2            \n\t" // xmm2 := ymm0[4:7]
	"vmovlpd          %%xmm11, (%%rcx)           \n\t" // store (c02,c12)
	"vmovhpd          %%xmm11, (%%rcx,%%rsi)     \n\t" // store (c22,c32)
	"vmovlpd          %%xmm2,  (%%rcx,%%r12)     \n\t" // store (c42,c52)
	"vmovhpd          %%xmm2,  (%%rcx,%%r13)     \n\t" // store (c62,c72)
	"addq      %%rdi, %%rcx                      \n\t" // c += cs_c;
	"                                            \n\t"
	"                                            \n\t" // update c82:cf2
	"                                            \n\t"
	"vextractf128 $1, %%ymm10, %%xmm2            \n\t" // xmm2 := ymm0[4:7]
	"vmovlpd          %%xmm10, (%%rdx)           \n\t" // store (c82,c92)
	"vmovhpd          %%xmm10, (%%rdx,%%rsi)     \n\t" // store (ca2,cb2)
	"vmovlpd          %%xmm2,  (%%rdx,%%r12)     \n\t" // store (cc2,cd2)
	"vmovhpd          %%xmm2,  (%%rdx,%%r13)     \n\t" // store (ce2,cf2)
	"addq      %%rdi, %%rdx                      \n\t" // c += cs_c;
	"                                            \n\t"
	"                                            \n\t" // update c03:c73
	"                                            \n\t"
	"vextractf128 $1, %%ymm9,  %%xmm2            \n\t" // xmm2 := ymm0[4:7]
	"vmovlpd          %%xmm9,  (%%rcx)           \n\t" // store (c03,c13)
	"vmovhpd          %%xmm9,  (%%rcx,%%rsi)     \n\t" // store (c23,c33)
	"vmovlpd          %%xmm2,  (%%rcx,%%r12)     \n\t" // store (c43,c53)
	"vmovhpd          %%xmm2,  (%%rcx,%%r13)     \n\t" // store (c63,c73)
	"addq      %%rdi, %%rcx                      \n\t" // c += cs_c;
	"                                            \n\t"
	"                                            \n\t" // update c83:cf3
	"                                            \n\t"
	"vextractf128 $1, %%ymm8,  %%xmm2            \n\t" // xmm2 := ymm0[4:7]
	"vmovlpd          %%xmm8,  (%%rdx)           \n\t" // store (c83,c93)
	"vmovhpd          %%xmm8,  (%%rdx,%%rsi)     \n\t" // store (ca3,cb3)
	"vmovlpd          %%xmm2,  (%%rdx,%%r12)     \n\t" // store (cc3,cd3)
	"vmovhpd          %%xmm2,  (%%rdx,%%r13)     \n\t" // store (ce3,cf3)
	"addq      %%rdi, %%rdx                      \n\t" // c += cs_c;
	"                                            \n\t"
	"                                            \n\t"
	"                                            \n\t"
	"jmp    .CDONE                               \n\t" // jump to end.
	"                                            \n\t"
	"                                            \n\t"
	"                                            \n\t"
	".CCOLSTORBZ:                                \n\t"
	"                                            \n\t"
	"                                            \n\t"
	"vmovaps          %%ymm15, (%%rcx)           \n\t" // store c00:c70
	"addq      %%rdi, %%rcx                      \n\t" // c += cs_c;
	"                                            \n\t"
	"vmovaps          %%ymm14, (%%rdx)           \n\t" // store c80:cf0
	"addq      %%rdi, %%rdx                      \n\t" // c += cs_c;
	"                                            \n\t"
	"vmovaps          %%ymm13, (%%rcx)           \n\t" // store c01:c71
	"addq      %%rdi, %%rcx                      \n\t" // c += cs_c;
	"                                            \n\t"
	"vmovaps          %%ymm12, (%%rdx)           \n\t" // store c81:cf1
	"addq      %%rdi, %%rdx                      \n\t" // c += cs_c;
	"                                            \n\t"
	"vmovaps          %%ymm11, (%%rcx)           \n\t" // store c02:c72
	"addq      %%rdi, %%rcx                      \n\t" // c += cs_c;
	"                                            \n\t"
	"vmovaps          %%ymm10, (%%rdx)           \n\t" // store c82:cf2
	"addq      %%rdi, %%rdx                      \n\t" // c += cs_c;
	"                                            \n\t"
	"vmovaps          %%ymm9,  (%%rcx)           \n\t" // store c03:c73
	"addq      %%rdi, %%rcx                      \n\t" // c += cs_c;
	"                                            \n\t"
	"vmovaps          %%ymm8,  (%%rdx)           \n\t" // store c83:cf3
	"addq      %%rdi, %%rdx                      \n\t" // c += cs_c;
	"                                            \n\t"
	"                                            \n\t"
	"                                            \n\t"
	"                                            \n\t"
	"                                            \n\t"
	".CDONE:                                     \n\t"
	"                                            \n\t"

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
	  "m" (cs_c),   // 8
	  "m" (b_next)/*, // 9
	  "m" (a_next)*/  // 10
	: // register clobber list
	  "rax", "rbx", "rcx", "rdx", "rsi", "rdi", 
	  "r8", "r9", "r10", "r11", "r12", "r13", "r14", "r15",
	  "xmm0", "xmm1", "xmm2", "xmm3",
	  "xmm4", "xmm5", "xmm6", "xmm7",
	  "xmm8", "xmm9", "xmm10", "xmm11",
	  "xmm12", "xmm13", "xmm14", "xmm15",
	  "memory"
	);
}



void bli_zgemm_asm_4x4(
                        dim_t              k,
                        dcomplex* restrict alpha,
                        dcomplex* restrict a,
                        dcomplex* restrict b,
                        dcomplex* restrict beta,
                        dcomplex* restrict c, inc_t rs_c, inc_t cs_c,
                        auxinfo_t*         data
                      )
{
	//void*   a_next = bli_auxinfo_next_a( data );
	//void*   b_next = bli_auxinfo_next_b( data );

	dim_t   k_iter = k / 4;
	dim_t   k_left = k % 4;

	__asm__ volatile
	(
	"                                            \n\t"
	"                                            \n\t"
	"movq                %2, %%rax               \n\t" // load address of a.
	"movq                %3, %%rbx               \n\t" // load address of b.
	//"movq                %9, %%r15               \n\t" // load address of b_next.
	//"movq               %10, %%r14               \n\t" // load address of a_next.
	"                                            \n\t"
	"vmovapd        0 * 32(%%rax), %%ymm0        \n\t" // initialize loop by pre-loading
	"vmovddup   0 + 0 * 32(%%rbx), %%ymm2        \n\t"
	"vmovddup   0 + 1 * 32(%%rbx), %%ymm3        \n\t"
	"                                            \n\t"
	"movq                %6, %%rcx               \n\t" // load address of c
	"movq                %8, %%rdi               \n\t" // load cs_c
	"leaq        (,%%rdi,8), %%rdi               \n\t" // cs_c *= sizeof(dcomplex)
	"leaq        (,%%rdi,2), %%rdi               \n\t"
	"leaq   (%%rcx,%%rdi,2), %%r10               \n\t" // load address of c + 2*cs_c;
	"                                            \n\t"
	"prefetcht0   3 * 8(%%rcx)                   \n\t" // prefetch c + 0*cs_c
	"prefetcht0   3 * 8(%%rcx,%%rdi)             \n\t" // prefetch c + 1*cs_c
	"prefetcht0   3 * 8(%%r10)                   \n\t" // prefetch c + 2*cs_c
	"prefetcht0   3 * 8(%%r10,%%rdi)             \n\t" // prefetch c + 3*cs_c
	"                                            \n\t"
	"vxorpd    %%ymm8,  %%ymm8,  %%ymm8          \n\t"
	"vxorpd    %%ymm9,  %%ymm9,  %%ymm9          \n\t"
	"vxorpd    %%ymm10, %%ymm10, %%ymm10         \n\t"
	"vxorpd    %%ymm11, %%ymm11, %%ymm11         \n\t"
	"vxorpd    %%ymm12, %%ymm12, %%ymm12         \n\t"
	"vxorpd    %%ymm13, %%ymm13, %%ymm13         \n\t"
	"vxorpd    %%ymm14, %%ymm14, %%ymm14         \n\t"
	"vxorpd    %%ymm15, %%ymm15, %%ymm15         \n\t"
	"                                            \n\t"
	"                                            \n\t"
	"                                            \n\t"
	"movq      %0, %%rsi                         \n\t" // i = k_iter;
	"testq  %%rsi, %%rsi                         \n\t" // check i via logical AND.
	"je     .ZCONSIDKLEFT                        \n\t" // if i == 0, jump to code that
	"                                            \n\t" // contains the k_left loop.
	"                                            \n\t"
	"                                            \n\t"
	".ZLOOPKITER:                                \n\t" // MAIN LOOP
	"                                            \n\t"
	"                                            \n\t"
	"                                            \n\t" // iteration 0
	"vmovapd        1 * 32(%%rax),      %%ymm1   \n\t"
	"vmulpd           %%ymm0,  %%ymm2,  %%ymm6   \n\t"
	"vperm2f128 $0x3, %%ymm2,  %%ymm2,  %%ymm4   \n\t"
	"vmulpd           %%ymm0,  %%ymm3,  %%ymm7   \n\t"
	"vperm2f128 $0x3, %%ymm3,  %%ymm3,  %%ymm5   \n\t"
	"vaddpd           %%ymm6,  %%ymm15, %%ymm15  \n\t"
	"vaddpd           %%ymm7,  %%ymm11, %%ymm11  \n\t"
	"                                            \n\t"
	"prefetcht0    16 * 32(%%rax)                \n\t"
	"vmulpd           %%ymm1,  %%ymm2,  %%ymm6   \n\t"
	"vmovddup   8 + 0 * 32(%%rbx),      %%ymm2   \n\t"
	"vmulpd           %%ymm1,  %%ymm3,  %%ymm7   \n\t"
	"vmovddup   8 + 1 * 32(%%rbx),      %%ymm3   \n\t"
	"vaddpd           %%ymm6,  %%ymm14, %%ymm14  \n\t"
	"vaddpd           %%ymm7,  %%ymm10, %%ymm10  \n\t"
	"                                            \n\t"
	"vmulpd           %%ymm0,  %%ymm4,  %%ymm6   \n\t"
	"vmulpd           %%ymm0,  %%ymm5,  %%ymm7   \n\t"
	"vpermilpd  $0x5, %%ymm0,  %%ymm0            \n\t"
	"vaddpd           %%ymm6,  %%ymm13, %%ymm13  \n\t"
	"vaddpd           %%ymm7,  %%ymm9,  %%ymm9   \n\t"
	"                                            \n\t"
	"vmulpd           %%ymm1,  %%ymm4,  %%ymm6   \n\t"
	"vperm2f128 $0x3, %%ymm2,  %%ymm2,  %%ymm4   \n\t"
	"vmulpd           %%ymm1,  %%ymm5,  %%ymm7   \n\t"
	"vperm2f128 $0x3, %%ymm3,  %%ymm3,  %%ymm5   \n\t"
	"vaddpd           %%ymm6,  %%ymm12, %%ymm12  \n\t"
	"vaddpd           %%ymm7,  %%ymm8,  %%ymm8   \n\t"
	"                                            \n\t"
	"vpermilpd  $0x5, %%ymm1,  %%ymm1            \n\t"
	"vmulpd           %%ymm0,  %%ymm2,  %%ymm6   \n\t"
	"vmulpd           %%ymm0,  %%ymm3,  %%ymm7   \n\t"
	"vaddsubpd        %%ymm6,  %%ymm15, %%ymm15  \n\t"
	"vaddsubpd        %%ymm7,  %%ymm11, %%ymm11  \n\t"
	"                                            \n\t"
	"vmulpd           %%ymm1,  %%ymm2,  %%ymm6   \n\t"
	"vmovddup   0 + 2 * 32(%%rbx),      %%ymm2   \n\t"
	"vmulpd           %%ymm1,  %%ymm3,  %%ymm7   \n\t"
	"vmovddup   0 + 3 * 32(%%rbx),      %%ymm3   \n\t"
	"vaddsubpd        %%ymm6,  %%ymm14, %%ymm14  \n\t"
	"vaddsubpd        %%ymm7,  %%ymm10, %%ymm10  \n\t"
	"                                            \n\t"
	"vmulpd           %%ymm0,  %%ymm4,  %%ymm6   \n\t"
	"vmulpd           %%ymm0,  %%ymm5,  %%ymm7   \n\t"
	"vmovapd        2 * 32(%%rax),      %%ymm0   \n\t"
	"vaddsubpd        %%ymm6,  %%ymm13, %%ymm13  \n\t"
	"vaddsubpd        %%ymm7,  %%ymm9,  %%ymm9   \n\t"
	"                                            \n\t"
	"vmulpd           %%ymm1,  %%ymm4,  %%ymm6   \n\t"
	"vmulpd           %%ymm1,  %%ymm5,  %%ymm7   \n\t"
	"vaddsubpd        %%ymm6,  %%ymm12, %%ymm12  \n\t"
	"vaddsubpd        %%ymm7,  %%ymm8,  %%ymm8   \n\t"
	"                                            \n\t"
	"                                            \n\t"
	"                                            \n\t" // iteration 1
	"vmovapd        3 * 32(%%rax),      %%ymm1   \n\t"
	"vmulpd           %%ymm0,  %%ymm2,  %%ymm6   \n\t"
	"vperm2f128 $0x3, %%ymm2,  %%ymm2,  %%ymm4   \n\t"
	"vmulpd           %%ymm0,  %%ymm3,  %%ymm7   \n\t"
	"vperm2f128 $0x3, %%ymm3,  %%ymm3,  %%ymm5   \n\t"
	"vaddpd           %%ymm6,  %%ymm15, %%ymm15  \n\t"
	"vaddpd           %%ymm7,  %%ymm11, %%ymm11  \n\t"
	"                                            \n\t"
	"prefetcht0    18 * 32(%%rax)                \n\t"
	"vmulpd           %%ymm1,  %%ymm2,  %%ymm6   \n\t"
	"vmovddup   8 + 2 * 32(%%rbx),      %%ymm2   \n\t"
	"vmulpd           %%ymm1,  %%ymm3,  %%ymm7   \n\t"
	"vmovddup   8 + 3 * 32(%%rbx),      %%ymm3   \n\t"
	"vaddpd           %%ymm6,  %%ymm14, %%ymm14  \n\t"
	"vaddpd           %%ymm7,  %%ymm10, %%ymm10  \n\t"
	"                                            \n\t"
	"vmulpd           %%ymm0,  %%ymm4,  %%ymm6   \n\t"
	"vmulpd           %%ymm0,  %%ymm5,  %%ymm7   \n\t"
	"vpermilpd  $0x5, %%ymm0,  %%ymm0            \n\t"
	"vaddpd           %%ymm6,  %%ymm13, %%ymm13  \n\t"
	"vaddpd           %%ymm7,  %%ymm9,  %%ymm9   \n\t"
	"                                            \n\t"
	"vmulpd           %%ymm1,  %%ymm4,  %%ymm6   \n\t"
	"vperm2f128 $0x3, %%ymm2,  %%ymm2,  %%ymm4   \n\t"
	"vmulpd           %%ymm1,  %%ymm5,  %%ymm7   \n\t"
	"vperm2f128 $0x3, %%ymm3,  %%ymm3,  %%ymm5   \n\t"
	"vaddpd           %%ymm6,  %%ymm12, %%ymm12  \n\t"
	"vaddpd           %%ymm7,  %%ymm8,  %%ymm8   \n\t"
	"                                            \n\t"
	"vpermilpd  $0x5, %%ymm1,  %%ymm1            \n\t"
	"vmulpd           %%ymm0,  %%ymm2,  %%ymm6   \n\t"
	"vmulpd           %%ymm0,  %%ymm3,  %%ymm7   \n\t"
	"vaddsubpd        %%ymm6,  %%ymm15, %%ymm15  \n\t"
	"vaddsubpd        %%ymm7,  %%ymm11, %%ymm11  \n\t"
	"                                            \n\t"
	"vmulpd           %%ymm1,  %%ymm2,  %%ymm6   \n\t"
	"vmovddup   0 + 4 * 32(%%rbx),      %%ymm2   \n\t"
	"vmulpd           %%ymm1,  %%ymm3,  %%ymm7   \n\t"
	"vmovddup   0 + 5 * 32(%%rbx),      %%ymm3   \n\t"
	"vaddsubpd        %%ymm6,  %%ymm14, %%ymm14  \n\t"
	"vaddsubpd        %%ymm7,  %%ymm10, %%ymm10  \n\t"
	"                                            \n\t"
	"vmulpd           %%ymm0,  %%ymm4,  %%ymm6   \n\t"
	"vmulpd           %%ymm0,  %%ymm5,  %%ymm7   \n\t"
	"vmovapd        4 * 32(%%rax),      %%ymm0   \n\t"
	"vaddsubpd        %%ymm6,  %%ymm13, %%ymm13  \n\t"
	"vaddsubpd        %%ymm7,  %%ymm9,  %%ymm9   \n\t"
	"                                            \n\t"
	"vmulpd           %%ymm1,  %%ymm4,  %%ymm6   \n\t"
	"vmulpd           %%ymm1,  %%ymm5,  %%ymm7   \n\t"
	"vaddsubpd        %%ymm6,  %%ymm12, %%ymm12  \n\t"
	"vaddsubpd        %%ymm7,  %%ymm8,  %%ymm8   \n\t"
	"                                            \n\t"
	"                                            \n\t"
	"                                            \n\t" // iteration 2
	"vmovapd        5 * 32(%%rax),      %%ymm1   \n\t"
	"vmulpd           %%ymm0,  %%ymm2,  %%ymm6   \n\t"
	"vperm2f128 $0x3, %%ymm2,  %%ymm2,  %%ymm4   \n\t"
	"vmulpd           %%ymm0,  %%ymm3,  %%ymm7   \n\t"
	"vperm2f128 $0x3, %%ymm3,  %%ymm3,  %%ymm5   \n\t"
	"vaddpd           %%ymm6,  %%ymm15, %%ymm15  \n\t"
	"vaddpd           %%ymm7,  %%ymm11, %%ymm11  \n\t"
	"                                            \n\t"
	"prefetcht0    20 * 32(%%rax)                \n\t"
	"vmulpd           %%ymm1,  %%ymm2,  %%ymm6   \n\t"
	"vmovddup   8 + 4 * 32(%%rbx),      %%ymm2   \n\t"
	"vmulpd           %%ymm1,  %%ymm3,  %%ymm7   \n\t"
	"vmovddup   8 + 5 * 32(%%rbx),      %%ymm3   \n\t"
	"vaddpd           %%ymm6,  %%ymm14, %%ymm14  \n\t"
	"vaddpd           %%ymm7,  %%ymm10, %%ymm10  \n\t"
	"                                            \n\t"
	"vmulpd           %%ymm0,  %%ymm4,  %%ymm6   \n\t"
	"vmulpd           %%ymm0,  %%ymm5,  %%ymm7   \n\t"
	"vpermilpd  $0x5, %%ymm0,  %%ymm0            \n\t"
	"vaddpd           %%ymm6,  %%ymm13, %%ymm13  \n\t"
	"vaddpd           %%ymm7,  %%ymm9,  %%ymm9   \n\t"
	"                                            \n\t"
	"vmulpd           %%ymm1,  %%ymm4,  %%ymm6   \n\t"
	"vperm2f128 $0x3, %%ymm2,  %%ymm2,  %%ymm4   \n\t"
	"vmulpd           %%ymm1,  %%ymm5,  %%ymm7   \n\t"
	"vperm2f128 $0x3, %%ymm3,  %%ymm3,  %%ymm5   \n\t"
	"vaddpd           %%ymm6,  %%ymm12, %%ymm12  \n\t"
	"vaddpd           %%ymm7,  %%ymm8,  %%ymm8   \n\t"
	"                                            \n\t"
	"vpermilpd  $0x5, %%ymm1,  %%ymm1            \n\t"
	"vmulpd           %%ymm0,  %%ymm2,  %%ymm6   \n\t"
	"vmulpd           %%ymm0,  %%ymm3,  %%ymm7   \n\t"
	"vaddsubpd        %%ymm6,  %%ymm15, %%ymm15  \n\t"
	"vaddsubpd        %%ymm7,  %%ymm11, %%ymm11  \n\t"
	"                                            \n\t"
	"vmulpd           %%ymm1,  %%ymm2,  %%ymm6   \n\t"
	"vmovddup   0 + 6 * 32(%%rbx),      %%ymm2   \n\t"
	"vmulpd           %%ymm1,  %%ymm3,  %%ymm7   \n\t"
	"vmovddup   0 + 7 * 32(%%rbx),      %%ymm3   \n\t"
	"vaddsubpd        %%ymm6,  %%ymm14, %%ymm14  \n\t"
	"vaddsubpd        %%ymm7,  %%ymm10, %%ymm10  \n\t"
	"                                            \n\t"
	"vmulpd           %%ymm0,  %%ymm4,  %%ymm6   \n\t"
	"vmulpd           %%ymm0,  %%ymm5,  %%ymm7   \n\t"
	"vmovapd        6 * 32(%%rax),      %%ymm0   \n\t"
	"vaddsubpd        %%ymm6,  %%ymm13, %%ymm13  \n\t"
	"vaddsubpd        %%ymm7,  %%ymm9,  %%ymm9   \n\t"
	"                                            \n\t"
	"vmulpd           %%ymm1,  %%ymm4,  %%ymm6   \n\t"
	"vmulpd           %%ymm1,  %%ymm5,  %%ymm7   \n\t"
	"vaddsubpd        %%ymm6,  %%ymm12, %%ymm12  \n\t"
	"vaddsubpd        %%ymm7,  %%ymm8,  %%ymm8   \n\t"
	"                                            \n\t"
	"                                            \n\t"
	"                                            \n\t" // iteration 3
	"vmovapd        7 * 32(%%rax),      %%ymm1   \n\t"
	"vmulpd           %%ymm0,  %%ymm2,  %%ymm6   \n\t"
	"vperm2f128 $0x3, %%ymm2,  %%ymm2,  %%ymm4   \n\t"
	"vmulpd           %%ymm0,  %%ymm3,  %%ymm7   \n\t"
	"vperm2f128 $0x3, %%ymm3,  %%ymm3,  %%ymm5   \n\t"
	"vaddpd           %%ymm6,  %%ymm15, %%ymm15  \n\t"
	"vaddpd           %%ymm7,  %%ymm11, %%ymm11  \n\t"
	"                                            \n\t"
	"prefetcht0    22 * 32(%%rax)                \n\t"
	"vmulpd           %%ymm1,  %%ymm2,  %%ymm6   \n\t"
	"vmovddup   8 + 6 * 32(%%rbx),      %%ymm2   \n\t"
	"vmulpd           %%ymm1,  %%ymm3,  %%ymm7   \n\t"
	"vmovddup   8 + 7 * 32(%%rbx),      %%ymm3   \n\t"
	"vaddpd           %%ymm6,  %%ymm14, %%ymm14  \n\t"
	"vaddpd           %%ymm7,  %%ymm10, %%ymm10  \n\t"
	"                                            \n\t"
	"vmulpd           %%ymm0,  %%ymm4,  %%ymm6   \n\t"
	"vmulpd           %%ymm0,  %%ymm5,  %%ymm7   \n\t"
	"vpermilpd  $0x5, %%ymm0,  %%ymm0            \n\t"
	"vaddpd           %%ymm6,  %%ymm13, %%ymm13  \n\t"
	"vaddpd           %%ymm7,  %%ymm9,  %%ymm9   \n\t"
	"                                            \n\t"
	"vmulpd           %%ymm1,  %%ymm4,  %%ymm6   \n\t"
	"vperm2f128 $0x3, %%ymm2,  %%ymm2,  %%ymm4   \n\t"
	"vmulpd           %%ymm1,  %%ymm5,  %%ymm7   \n\t"
	"vperm2f128 $0x3, %%ymm3,  %%ymm3,  %%ymm5   \n\t"
	"vaddpd           %%ymm6,  %%ymm12, %%ymm12  \n\t"
	"vaddpd           %%ymm7,  %%ymm8,  %%ymm8   \n\t"
	"                                            \n\t"
	"vpermilpd  $0x5, %%ymm1,  %%ymm1            \n\t"
	"vmulpd           %%ymm0,  %%ymm2,  %%ymm6   \n\t"
	"vmulpd           %%ymm0,  %%ymm3,  %%ymm7   \n\t"
	"vaddsubpd        %%ymm6,  %%ymm15, %%ymm15  \n\t"
	"vaddsubpd        %%ymm7,  %%ymm11, %%ymm11  \n\t"
	"                                            \n\t"
	"vmulpd           %%ymm1,  %%ymm2,  %%ymm6   \n\t"
	"vmovddup   0 + 8 * 32(%%rbx),      %%ymm2   \n\t"
	"vmulpd           %%ymm1,  %%ymm3,  %%ymm7   \n\t"
	"vmovddup   0 + 9 * 32(%%rbx),      %%ymm3   \n\t"
	"vaddsubpd        %%ymm6,  %%ymm14, %%ymm14  \n\t"
	"vaddsubpd        %%ymm7,  %%ymm10, %%ymm10  \n\t"
	"                                            \n\t"
	"vmulpd           %%ymm0,  %%ymm4,  %%ymm6   \n\t"
	"vmulpd           %%ymm0,  %%ymm5,  %%ymm7   \n\t"
	"vmovapd        8 * 32(%%rax),      %%ymm0   \n\t"
	"vaddsubpd        %%ymm6,  %%ymm13, %%ymm13  \n\t"
	"vaddsubpd        %%ymm7,  %%ymm9,  %%ymm9   \n\t"
	"                                            \n\t"
	"vmulpd           %%ymm1,  %%ymm4,  %%ymm6   \n\t"
	"vmulpd           %%ymm1,  %%ymm5,  %%ymm7   \n\t"
	"vaddsubpd        %%ymm6,  %%ymm12, %%ymm12  \n\t"
	"vaddsubpd        %%ymm7,  %%ymm8,  %%ymm8   \n\t"
	"                                            \n\t"
	"                                            \n\t"
	"addq         $4 * 4 * 16, %%rbx             \n\t" // b += 4*4 (unroll x nr)
	"addq         $4 * 4 * 16, %%rax             \n\t" // a += 4*4 (unroll x mr)
	"                                            \n\t"
	"                                            \n\t"
	"decq   %%rsi                                \n\t" // i -= 1;
	"jne    .ZLOOPKITER                          \n\t" // iterate again if i != 0.
	"                                            \n\t"
	"                                            \n\t"
	"                                            \n\t"
	"                                            \n\t"
	"                                            \n\t"
	"                                            \n\t"
	".ZCONSIDKLEFT:                              \n\t"
	"                                            \n\t"
	"movq      %1, %%rsi                         \n\t" // i = k_left;
	"testq  %%rsi, %%rsi                         \n\t" // check i via logical AND.
	"je     .ZPOSTACCUM                          \n\t" // if i == 0, we're done; jump to end.
	"                                            \n\t" // else, we prepare to enter k_left loop.
	"                                            \n\t"
	"                                            \n\t"
	".ZLOOPKLEFT:                                \n\t" // EDGE LOOP
	"                                            \n\t"
	"                                            \n\t" // iteration 0
	"vmovapd        1 * 32(%%rax),      %%ymm1   \n\t"
	"vmulpd           %%ymm0,  %%ymm2,  %%ymm6   \n\t"
	"vperm2f128 $0x3, %%ymm2,  %%ymm2,  %%ymm4   \n\t"
	"vmulpd           %%ymm0,  %%ymm3,  %%ymm7   \n\t"
	"vperm2f128 $0x3, %%ymm3,  %%ymm3,  %%ymm5   \n\t"
	"vaddpd           %%ymm6,  %%ymm15, %%ymm15  \n\t"
	"vaddpd           %%ymm7,  %%ymm11, %%ymm11  \n\t"
	"                                            \n\t"
	"prefetcht0    16 * 32(%%rax)                \n\t"
	"vmulpd           %%ymm1,  %%ymm2,  %%ymm6   \n\t"
	"vmovddup   8 + 0 * 32(%%rbx),      %%ymm2   \n\t"
	"vmulpd           %%ymm1,  %%ymm3,  %%ymm7   \n\t"
	"vmovddup   8 + 1 * 32(%%rbx),      %%ymm3   \n\t"
	"vaddpd           %%ymm6,  %%ymm14, %%ymm14  \n\t"
	"vaddpd           %%ymm7,  %%ymm10, %%ymm10  \n\t"
	"                                            \n\t"
	"vmulpd           %%ymm0,  %%ymm4,  %%ymm6   \n\t"
	"vmulpd           %%ymm0,  %%ymm5,  %%ymm7   \n\t"
	"vpermilpd  $0x5, %%ymm0,  %%ymm0            \n\t"
	"vaddpd           %%ymm6,  %%ymm13, %%ymm13  \n\t"
	"vaddpd           %%ymm7,  %%ymm9,  %%ymm9   \n\t"
	"                                            \n\t"
	"vmulpd           %%ymm1,  %%ymm4,  %%ymm6   \n\t"
	"vperm2f128 $0x3, %%ymm2,  %%ymm2,  %%ymm4   \n\t"
	"vmulpd           %%ymm1,  %%ymm5,  %%ymm7   \n\t"
	"vperm2f128 $0x3, %%ymm3,  %%ymm3,  %%ymm5   \n\t"
	"vaddpd           %%ymm6,  %%ymm12, %%ymm12  \n\t"
	"vaddpd           %%ymm7,  %%ymm8,  %%ymm8   \n\t"
	"                                            \n\t"
	"vpermilpd  $0x5, %%ymm1,  %%ymm1            \n\t"
	"vmulpd           %%ymm0,  %%ymm2,  %%ymm6   \n\t"
	"vmulpd           %%ymm0,  %%ymm3,  %%ymm7   \n\t"
	"vaddsubpd        %%ymm6,  %%ymm15, %%ymm15  \n\t"
	"vaddsubpd        %%ymm7,  %%ymm11, %%ymm11  \n\t"
	"                                            \n\t"
	"vmulpd           %%ymm1,  %%ymm2,  %%ymm6   \n\t"
	"vmovddup   0 + 2 * 32(%%rbx),      %%ymm2   \n\t"
	"vmulpd           %%ymm1,  %%ymm3,  %%ymm7   \n\t"
	"vmovddup   0 + 3 * 32(%%rbx),      %%ymm3   \n\t"
	"vaddsubpd        %%ymm6,  %%ymm14, %%ymm14  \n\t"
	"vaddsubpd        %%ymm7,  %%ymm10, %%ymm10  \n\t"
	"                                            \n\t"
	"vmulpd           %%ymm0,  %%ymm4,  %%ymm6   \n\t"
	"vmulpd           %%ymm0,  %%ymm5,  %%ymm7   \n\t"
	"vmovapd        2 * 32(%%rax),      %%ymm0   \n\t"
	"vaddsubpd        %%ymm6,  %%ymm13, %%ymm13  \n\t"
	"vaddsubpd        %%ymm7,  %%ymm9,  %%ymm9   \n\t"
	"                                            \n\t"
	"vmulpd           %%ymm1,  %%ymm4,  %%ymm6   \n\t"
	"vmulpd           %%ymm1,  %%ymm5,  %%ymm7   \n\t"
	"vaddsubpd        %%ymm6,  %%ymm12, %%ymm12  \n\t"
	"vaddsubpd        %%ymm7,  %%ymm8,  %%ymm8   \n\t"
	"                                            \n\t"
	"                                            \n\t"
	"addq         $4 * 1 * 16, %%rax             \n\t" // a += 4 (1 x mr)
	"addq         $4 * 1 * 16, %%rbx             \n\t" // b += 4 (1 x nr)
	"                                            \n\t"
	"                                            \n\t"
	"decq   %%rsi                                \n\t" // i -= 1;
	"jne    .ZLOOPKLEFT                          \n\t" // iterate again if i != 0.
	"                                            \n\t"
	"                                            \n\t"
	"                                            \n\t"
	".ZPOSTACCUM:                                \n\t"
	"                                            \n\t"
	"                                            \n\t" // ymm15:  ymm13:  ymm11:  ymm9:
	"                                            \n\t" // ( ab00  ( ab01  ( ab02  ( ab03
	"                                            \n\t" //   ab10    ab11    ab12    ab13  
	"                                            \n\t" //   ab21    ab20    ab23    ab22
	"                                            \n\t" //   ab31 )  ab30 )  ab33 )  ab32 )
	"                                            \n\t"
	"                                            \n\t" // ymm14:  ymm12:  ymm10:  ymm8:
	"                                            \n\t" // ( ab40  ( ab41  ( ab42  ( ab43
	"                                            \n\t" //   ab50    ab51    ab52    ab53  
	"                                            \n\t" //   ab61    ab60    ab63    ab62
	"                                            \n\t" //   ab71 )  ab70 )  ab73 )  ab72 )
	"                                            \n\t"
	"                                            \n\t"
	"vmovapd           %%ymm15, %%ymm7           \n\t"
	"vperm2f128 $0x12, %%ymm15, %%ymm13, %%ymm15 \n\t"
	"vperm2f128 $0x30, %%ymm7,  %%ymm13, %%ymm13 \n\t"
	"                                            \n\t"
	"vmovapd           %%ymm11, %%ymm7           \n\t"
	"vperm2f128 $0x12, %%ymm11, %%ymm9,  %%ymm11 \n\t"
	"vperm2f128 $0x30, %%ymm7,  %%ymm9,  %%ymm9  \n\t"
	"                                            \n\t"
	"vmovapd           %%ymm14, %%ymm7           \n\t"
	"vperm2f128 $0x12, %%ymm14, %%ymm12, %%ymm14 \n\t"
	"vperm2f128 $0x30, %%ymm7,  %%ymm12, %%ymm12 \n\t"
	"                                            \n\t"
	"vmovapd           %%ymm10, %%ymm7           \n\t"
	"vperm2f128 $0x12, %%ymm10, %%ymm8,  %%ymm10 \n\t"
	"vperm2f128 $0x30, %%ymm7,  %%ymm8,  %%ymm8  \n\t"
	"                                            \n\t"
	"                                            \n\t"
	"                                            \n\t" // ymm15:  ymm13:  ymm11:  ymm9:
	"                                            \n\t" // ( ab00  ( ab01  ( ab02  ( ab03
	"                                            \n\t" //   ab10    ab11    ab12    ab13  
	"                                            \n\t" //   ab20    ab21    ab22    ab23
	"                                            \n\t" //   ab30 )  ab31 )  ab32 )  ab33 )
	"                                            \n\t"
	"                                            \n\t" // ymm14:  ymm12:  ymm10:  ymm8:
	"                                            \n\t" // ( ab40  ( ab41  ( ab42  ( ab43
	"                                            \n\t" //   ab50    ab51    ab52    ab53  
	"                                            \n\t" //   ab60    ab61    ab62    ab63
	"                                            \n\t" //   ab70 )  ab71 )  ab72 )  ab73 )
	"                                            \n\t"
	"                                            \n\t"
	"                                            \n\t" // scale by alpha
	"                                            \n\t"
	"movq         %4, %%rax                      \n\t" // load address of alpha
	"vbroadcastsd    (%%rax), %%ymm7             \n\t" // load alpha_r and duplicate
	"vbroadcastsd   8(%%rax), %%ymm6             \n\t" // load alpha_i and duplicate
	"                                            \n\t"
	"vpermilpd  $0x5, %%ymm15, %%ymm3            \n\t"
	"vmulpd           %%ymm7,  %%ymm15, %%ymm15  \n\t"
	"vmulpd           %%ymm6,  %%ymm3,  %%ymm3   \n\t"
	"vaddsubpd        %%ymm3,  %%ymm15, %%ymm15  \n\t"
	"                                            \n\t"
	"vpermilpd  $0x5, %%ymm14, %%ymm2            \n\t"
	"vmulpd           %%ymm7,  %%ymm14, %%ymm14  \n\t"
	"vmulpd           %%ymm6,  %%ymm2,  %%ymm2   \n\t"
	"vaddsubpd        %%ymm2,  %%ymm14, %%ymm14  \n\t"
	"                                            \n\t"
	"vpermilpd  $0x5, %%ymm13, %%ymm1            \n\t"
	"vmulpd           %%ymm7,  %%ymm13, %%ymm13  \n\t"
	"vmulpd           %%ymm6,  %%ymm1,  %%ymm1   \n\t"
	"vaddsubpd        %%ymm1,  %%ymm13, %%ymm13  \n\t"
	"                                            \n\t"
	"vpermilpd  $0x5, %%ymm12, %%ymm0            \n\t"
	"vmulpd           %%ymm7,  %%ymm12, %%ymm12  \n\t"
	"vmulpd           %%ymm6,  %%ymm0,  %%ymm0   \n\t"
	"vaddsubpd        %%ymm0,  %%ymm12, %%ymm12  \n\t"
	"                                            \n\t"
	"vpermilpd  $0x5, %%ymm11, %%ymm3            \n\t"
	"vmulpd           %%ymm7,  %%ymm11, %%ymm11  \n\t"
	"vmulpd           %%ymm6,  %%ymm3,  %%ymm3   \n\t"
	"vaddsubpd        %%ymm3,  %%ymm11, %%ymm11  \n\t"
	"                                            \n\t"
	"vpermilpd  $0x5, %%ymm10, %%ymm2            \n\t"
	"vmulpd           %%ymm7,  %%ymm10, %%ymm10  \n\t"
	"vmulpd           %%ymm6,  %%ymm2,  %%ymm2   \n\t"
	"vaddsubpd        %%ymm2,  %%ymm10, %%ymm10  \n\t"
	"                                            \n\t"
	"vpermilpd  $0x5, %%ymm9,  %%ymm1            \n\t"
	"vmulpd           %%ymm7,  %%ymm9,  %%ymm9   \n\t"
	"vmulpd           %%ymm6,  %%ymm1,  %%ymm1   \n\t"
	"vaddsubpd        %%ymm1,  %%ymm9,  %%ymm9   \n\t"
	"                                            \n\t"
	"vpermilpd  $0x5, %%ymm8,  %%ymm0            \n\t"
	"vmulpd           %%ymm7,  %%ymm8,  %%ymm8   \n\t"
	"vmulpd           %%ymm6,  %%ymm0,  %%ymm0   \n\t"
	"vaddsubpd        %%ymm0,  %%ymm8,  %%ymm8   \n\t"
	"                                            \n\t"
	"                                            \n\t"
	"                                            \n\t"
	"                                            \n\t"
	"movq         %5, %%rbx                      \n\t" // load address of beta 
	"vbroadcastsd    (%%rbx), %%ymm7             \n\t" // load beta_r and duplicate
	"vbroadcastsd   8(%%rbx), %%ymm6             \n\t" // load beta_i and duplicate
	"                                            \n\t"
	"                                            \n\t"
	"                                            \n\t"
	"                                            \n\t"
	"                                            \n\t"
	"                                            \n\t"
	"                                            \n\t"
	"movq                %7, %%rsi               \n\t" // load rs_c
	"leaq        (,%%rsi,8), %%rsi               \n\t" // rsi = rs_c * sizeof(dcomplex)
	"leaq        (,%%rsi,2), %%rsi               \n\t"
	"leaq   (%%rcx,%%rsi,2), %%rdx               \n\t" // load address of c + 2*rs_c;
	"                                            \n\t"
	"                                            \n\t"
	"                                            \n\t"
	"                                            \n\t"
	"                                            \n\t"
	"                                            \n\t"
	"                                            \n\t" // determine if
	"                                            \n\t" //    c    % 32 == 0, AND
	"                                            \n\t" // 16*cs_c % 32 == 0, AND
	"                                            \n\t" //    rs_c      == 1
	"                                            \n\t" // ie: aligned, ldim aligned, and
	"                                            \n\t" // column-stored
	"                                            \n\t"
	"cmpq      $16, %%rsi                        \n\t" // set ZF if (16*rs_c) == 16.
	"sete           %%bl                         \n\t" // bl = ( ZF == 1 ? 1 : 0 );
	"testq     $31, %%rcx                        \n\t" // set ZF if c & 32 is zero.
	"setz           %%bh                         \n\t" // bh = ( ZF == 0 ? 1 : 0 );
	"testq     $31, %%rdi                        \n\t" // set ZF if (16*cs_c) & 32 is zero.
	"setz           %%al                         \n\t" // al = ( ZF == 0 ? 1 : 0 );
	"                                            \n\t" // and(bl,bh) followed by
	"                                            \n\t" // and(bh,al) will reveal result
	"                                            \n\t"
	"                                            \n\t" // now avoid loading C if beta == 0
	"                                            \n\t"
	"vxorpd    %%ymm0,  %%ymm0,  %%ymm0          \n\t" // set ymm0 to zero.
	"vucomisd  %%xmm0,  %%xmm7                   \n\t" // set ZF if beta_r == 0.
	"sete       %%r8b                            \n\t" // r8b = ( ZF == 1 ? 1 : 0 );
	"vucomisd  %%xmm0,  %%xmm6                   \n\t" // set ZF if beta_i == 0.
	"sete       %%r9b                            \n\t" // r9b = ( ZF == 1 ? 1 : 0 );
	"andb       %%r8b, %%r9b                     \n\t" // set ZF if r8b & r9b == 1.
	"jne      .ZBETAZERO                         \n\t" // if ZF = 0, jump to beta == 0 case
	"                                            \n\t"
	"                                            \n\t"
	"                                            \n\t" // check if aligned/column-stored
	"andb     %%bl, %%bh                         \n\t" // set ZF if bl & bh == 1.
	"andb     %%bh, %%al                         \n\t" // set ZF if bh & al == 1.
	"jne     .ZCOLSTORED                         \n\t" // jump to column storage case
	"                                            \n\t"
	"                                            \n\t"
	"                                            \n\t"
	".ZGENSTORED:                                \n\t"
	"                                            \n\t" // update c00:c30
	"                                            \n\t"
	"vmovupd    (%%rcx),       %%xmm0            \n\t" // load (c00,c10) into xmm0
	"vmovupd    (%%rcx,%%rsi), %%xmm2            \n\t" // load (c20,c30) into xmm2
	"vinsertf128  $1, %%xmm2,  %%ymm0,  %%ymm0   \n\t" // ymm0 := (ymm0[0:1],xmm2)
	"vpermilpd  $0x5, %%ymm0,  %%ymm2            \n\t" // scale ymm0 by beta
	"vmulpd           %%ymm7,  %%ymm0,  %%ymm0   \n\t"
	"vmulpd           %%ymm6,  %%ymm2,  %%ymm2   \n\t"
	"vaddsubpd        %%ymm2,  %%ymm0,  %%ymm0   \n\t"
	"vaddpd           %%ymm15, %%ymm0,  %%ymm0   \n\t" // add the gemm result to ymm0
	"vextractf128 $1, %%ymm0,  %%xmm2            \n\t" // xmm2 := ymm0[2:3]
	"vmovupd          %%xmm0,  (%%rcx)           \n\t" // store (c00,c10)
	"vmovupd          %%xmm2,  (%%rcx,%%rsi)     \n\t" // store (c20,c30)
	"addq      %%rdi, %%rcx                      \n\t" // c += cs_c;
	"                                            \n\t"
	"                                            \n\t" // update c40:c70
	"                                            \n\t"
	"vmovupd    (%%rdx),       %%xmm0            \n\t" // load (c40,c50) into xmm0
	"vmovupd    (%%rdx,%%rsi), %%xmm2            \n\t" // load (c60,c70) into xmm2
	"vinsertf128  $1, %%xmm2,  %%ymm0,  %%ymm0   \n\t" // ymm0 := (ymm0[0:1],xmm2)
	"vpermilpd  $0x5, %%ymm0,  %%ymm2            \n\t" // scale ymm0 by beta
	"vmulpd           %%ymm7,  %%ymm0,  %%ymm0   \n\t"
	"vmulpd           %%ymm6,  %%ymm2,  %%ymm2   \n\t"
	"vaddsubpd        %%ymm2,  %%ymm0,  %%ymm0   \n\t"
	"vaddpd           %%ymm14, %%ymm0,  %%ymm0   \n\t" // add the gemm result to ymm0
	"vextractf128 $1, %%ymm0,  %%xmm2            \n\t" // xmm2 := ymm0[2:3]
	"vmovupd          %%xmm0,  (%%rdx)           \n\t" // store (c40,c50)
	"vmovupd          %%xmm2,  (%%rdx,%%rsi)     \n\t" // store (c60,c70)
	"addq      %%rdi, %%rdx                      \n\t" // c += cs_c;
	"                                            \n\t"
	"                                            \n\t" // update c01:c31
	"                                            \n\t"
	"vmovupd    (%%rcx),       %%xmm0            \n\t" // load (c01,c11) into xmm0
	"vmovupd    (%%rcx,%%rsi), %%xmm2            \n\t" // load (c21,c31) into xmm2
	"vinsertf128  $1, %%xmm2,  %%ymm0,  %%ymm0   \n\t" // ymm0 := (ymm0[0:1],xmm2)
	"vpermilpd  $0x5, %%ymm0,  %%ymm2            \n\t" // scale ymm0 by beta
	"vmulpd           %%ymm7,  %%ymm0,  %%ymm0   \n\t"
	"vmulpd           %%ymm6,  %%ymm2,  %%ymm2   \n\t"
	"vaddsubpd        %%ymm2,  %%ymm0,  %%ymm0   \n\t"
	"vaddpd           %%ymm13, %%ymm0,  %%ymm0   \n\t" // add the gemm result to ymm0
	"vextractf128 $1, %%ymm0,  %%xmm2            \n\t" // xmm2 := ymm0[2:3]
	"vmovupd          %%xmm0,  (%%rcx)           \n\t" // store (c01,c11)
	"vmovupd          %%xmm2,  (%%rcx,%%rsi)     \n\t" // store (c21,c31)
	"addq      %%rdi, %%rcx                      \n\t" // c += cs_c;
	"                                            \n\t"
	"                                            \n\t" // update c41:c71
	"                                            \n\t"
	"vmovupd    (%%rdx),       %%xmm0            \n\t" // load (c41,c51) into xmm0
	"vmovupd    (%%rdx,%%rsi), %%xmm2            \n\t" // load (c61,c71) into xmm2
	"vinsertf128  $1, %%xmm2,  %%ymm0,  %%ymm0   \n\t" // ymm0 := (ymm0[0:1],xmm2)
	"vpermilpd  $0x5, %%ymm0,  %%ymm2            \n\t" // scale ymm0 by beta
	"vmulpd           %%ymm7,  %%ymm0,  %%ymm0   \n\t"
	"vmulpd           %%ymm6,  %%ymm2,  %%ymm2   \n\t"
	"vaddsubpd        %%ymm2,  %%ymm0,  %%ymm0   \n\t"
	"vaddpd           %%ymm12, %%ymm0,  %%ymm0   \n\t" // add the gemm result to ymm0
	"vextractf128 $1, %%ymm0,  %%xmm2            \n\t" // xmm2 := ymm0[2:3]
	"vmovupd          %%xmm0,  (%%rdx)           \n\t" // store (c41,c51)
	"vmovupd          %%xmm2,  (%%rdx,%%rsi)     \n\t" // store (c61,c71)
	"addq      %%rdi, %%rdx                      \n\t" // c += cs_c;
	"                                            \n\t"
	"                                            \n\t" // update c02:c32
	"                                            \n\t"
	"vmovupd    (%%rcx),       %%xmm0            \n\t" // load (c02,c12) into xmm0
	"vmovupd    (%%rcx,%%rsi), %%xmm2            \n\t" // load (c22,c32) into xmm2
	"vinsertf128  $1, %%xmm2,  %%ymm0,  %%ymm0   \n\t" // ymm0 := (ymm0[0:1],xmm2)
	"vpermilpd  $0x5, %%ymm0,  %%ymm2            \n\t" // scale ymm0 by beta
	"vmulpd           %%ymm7,  %%ymm0,  %%ymm0   \n\t"
	"vmulpd           %%ymm6,  %%ymm2,  %%ymm2   \n\t"
	"vaddsubpd        %%ymm2,  %%ymm0,  %%ymm0   \n\t"
	"vaddpd           %%ymm11, %%ymm0,  %%ymm0   \n\t" // add the gemm result to ymm0
	"vextractf128 $1, %%ymm0,  %%xmm2            \n\t" // xmm2 := ymm0[2:3]
	"vmovupd          %%xmm0,  (%%rcx)           \n\t" // store (c02,c12)
	"vmovupd          %%xmm2,  (%%rcx,%%rsi)     \n\t" // store (c22,c32)
	"addq      %%rdi, %%rcx                      \n\t" // c += cs_c;
	"                                            \n\t"
	"                                            \n\t" // update c42:c72
	"                                            \n\t"
	"vmovupd    (%%rdx),       %%xmm0            \n\t" // load (c42,c52) into xmm0
	"vmovupd    (%%rdx,%%rsi), %%xmm2            \n\t" // load (c62,c72) into xmm2
	"vinsertf128  $1, %%xmm2,  %%ymm0,  %%ymm0   \n\t" // ymm0 := (ymm0[0:1],xmm2)
	"vpermilpd  $0x5, %%ymm0,  %%ymm2            \n\t" // scale ymm0 by beta
	"vmulpd           %%ymm7,  %%ymm0,  %%ymm0   \n\t"
	"vmulpd           %%ymm6,  %%ymm2,  %%ymm2   \n\t"
	"vaddsubpd        %%ymm2,  %%ymm0,  %%ymm0   \n\t"
	"vaddpd           %%ymm10, %%ymm0,  %%ymm0   \n\t" // add the gemm result to ymm0
	"vextractf128 $1, %%ymm0,  %%xmm2            \n\t" // xmm2 := ymm0[2:3]
	"vmovupd          %%xmm0,  (%%rdx)           \n\t" // store (c42,c52)
	"vmovupd          %%xmm2,  (%%rdx,%%rsi)     \n\t" // store (c62,c72)
	"addq      %%rdi, %%rdx                      \n\t" // c += cs_c;
	"                                            \n\t"
	"                                            \n\t" // update c03:c33
	"                                            \n\t"
	"vmovupd    (%%rcx),       %%xmm0            \n\t" // load (c03,c13) into xmm0
	"vmovupd    (%%rcx,%%rsi), %%xmm2            \n\t" // load (c23,c33) into xmm2
	"vinsertf128  $1, %%xmm2,  %%ymm0,  %%ymm0   \n\t" // ymm0 := (ymm0[0:1],xmm2)
	"vpermilpd  $0x5, %%ymm0,  %%ymm2            \n\t" // scale ymm0 by beta
	"vmulpd           %%ymm7,  %%ymm0,  %%ymm0   \n\t"
	"vmulpd           %%ymm6,  %%ymm2,  %%ymm2   \n\t"
	"vaddsubpd        %%ymm2,  %%ymm0,  %%ymm0   \n\t"
	"vaddpd           %%ymm9,  %%ymm0,  %%ymm0   \n\t" // add the gemm result to ymm0
	"vextractf128 $1, %%ymm0,  %%xmm2            \n\t" // xmm2 := ymm0[2:3]
	"vmovupd          %%xmm0,  (%%rcx)           \n\t" // store (c03,c13)
	"vmovupd          %%xmm2,  (%%rcx,%%rsi)     \n\t" // store (c23,c33)
	"addq      %%rdi, %%rcx                      \n\t" // c += cs_c;
	"                                            \n\t"
	"                                            \n\t" // update c43:c73
	"                                            \n\t"
	"vmovupd    (%%rdx),       %%xmm0            \n\t" // load (c43,c53) into xmm0
	"vmovupd    (%%rdx,%%rsi), %%xmm2            \n\t" // load (c63,c73) into xmm2
	"vinsertf128  $1, %%xmm2,  %%ymm0,  %%ymm0   \n\t" // ymm0 := (ymm0[0:1],xmm2)
	"vpermilpd  $0x5, %%ymm0,  %%ymm2            \n\t" // scale ymm0 by beta
	"vmulpd           %%ymm7,  %%ymm0,  %%ymm0   \n\t"
	"vmulpd           %%ymm6,  %%ymm2,  %%ymm2   \n\t"
	"vaddsubpd        %%ymm2,  %%ymm0,  %%ymm0   \n\t"
	"vaddpd           %%ymm8,  %%ymm0,  %%ymm0   \n\t" // add the gemm result to ymm0
	"vextractf128 $1, %%ymm0,  %%xmm2            \n\t" // xmm2 := ymm0[2:3]
	"vmovupd          %%xmm0,  (%%rdx)           \n\t" // store (c43,c53)
	"vmovupd          %%xmm2,  (%%rdx,%%rsi)     \n\t" // store (c63,c73)
	"                                            \n\t"
	"                                            \n\t"
	"                                            \n\t"
	"jmp    .ZDONE                               \n\t" // jump to end.
	"                                            \n\t"
	"                                            \n\t"
	"                                            \n\t"
	".ZCOLSTORED:                                \n\t"
	"                                            \n\t" // update c00:c30
	"                                            \n\t"
	"vmovapd    (%%rcx),       %%ymm0            \n\t" // load c00:c30 into ymm0
	"vpermilpd  $0x5, %%ymm0,  %%ymm2            \n\t" // scale ymm0 by beta
	"vmulpd           %%ymm7,  %%ymm0,  %%ymm0   \n\t"
	"vmulpd           %%ymm6,  %%ymm2,  %%ymm2   \n\t"
	"vaddsubpd        %%ymm2,  %%ymm0,  %%ymm0   \n\t"
	"vaddpd           %%ymm15, %%ymm0,  %%ymm0   \n\t" // add the gemm result to ymm0
	"vmovapd          %%ymm0,  (%%rcx)           \n\t" // store c00:c30
	"addq      %%rdi, %%rcx                      \n\t" // c += cs_c;
	"                                            \n\t"
	"                                            \n\t" // update c40:c70
	"                                            \n\t"
	"vmovapd    (%%rdx),       %%ymm0            \n\t" // load c40:c70 into ymm0
	"vpermilpd  $0x5, %%ymm0,  %%ymm2            \n\t" // scale ymm0 by beta
	"vmulpd           %%ymm7,  %%ymm0,  %%ymm0   \n\t"
	"vmulpd           %%ymm6,  %%ymm2,  %%ymm2   \n\t"
	"vaddsubpd        %%ymm2,  %%ymm0,  %%ymm0   \n\t"
	"vaddpd           %%ymm14, %%ymm0,  %%ymm0   \n\t" // add the gemm result to ymm0
	"vmovapd          %%ymm0,  (%%rdx)           \n\t" // store c40:c70
	"addq      %%rdi, %%rdx                      \n\t" // c += cs_c;
	"                                            \n\t"
	"                                            \n\t" // update c01:c31
	"                                            \n\t"
	"vmovapd    (%%rcx),       %%ymm0            \n\t" // load c01:c31 into ymm0
	"vpermilpd  $0x5, %%ymm0,  %%ymm2            \n\t" // scale ymm0 by beta
	"vmulpd           %%ymm7,  %%ymm0,  %%ymm0   \n\t"
	"vmulpd           %%ymm6,  %%ymm2,  %%ymm2   \n\t"
	"vaddsubpd        %%ymm2,  %%ymm0,  %%ymm0   \n\t"
	"vaddpd           %%ymm13, %%ymm0,  %%ymm0   \n\t" // add the gemm result to ymm0
	"vmovapd          %%ymm0,  (%%rcx)           \n\t" // store c01:c31
	"addq      %%rdi, %%rcx                      \n\t" // c += cs_c;
	"                                            \n\t"
	"                                            \n\t" // update c41:c71
	"                                            \n\t"
	"vmovapd    (%%rdx),       %%ymm0            \n\t" // load c41:c71 into ymm0
	"vpermilpd  $0x5, %%ymm0,  %%ymm2            \n\t" // scale ymm0 by beta
	"vmulpd           %%ymm7,  %%ymm0,  %%ymm0   \n\t"
	"vmulpd           %%ymm6,  %%ymm2,  %%ymm2   \n\t"
	"vaddsubpd        %%ymm2,  %%ymm0,  %%ymm0   \n\t"
	"vaddpd           %%ymm12, %%ymm0,  %%ymm0   \n\t" // add the gemm result to ymm0
	"vmovapd          %%ymm0,  (%%rdx)           \n\t" // store c41:c71
	"addq      %%rdi, %%rdx                      \n\t" // c += cs_c;
	"                                            \n\t"
	"                                            \n\t" // update c02:c32
	"                                            \n\t"
	"vmovapd    (%%rcx),       %%ymm0            \n\t" // load c02:c32 into ymm0
	"vpermilpd  $0x5, %%ymm0,  %%ymm2            \n\t" // scale ymm0 by beta
	"vmulpd           %%ymm7,  %%ymm0,  %%ymm0   \n\t"
	"vmulpd           %%ymm6,  %%ymm2,  %%ymm2   \n\t"
	"vaddsubpd        %%ymm2,  %%ymm0,  %%ymm0   \n\t"
	"vaddpd           %%ymm11, %%ymm0,  %%ymm0   \n\t" // add the gemm result to ymm0
	"vmovapd          %%ymm0,  (%%rcx)           \n\t" // store c02:c32
	"addq      %%rdi, %%rcx                      \n\t" // c += cs_c;
	"                                            \n\t"
	"                                            \n\t" // update c42:c72
	"                                            \n\t"
	"vmovapd    (%%rdx),       %%ymm0            \n\t" // load c42:c72 into ymm0
	"vpermilpd  $0x5, %%ymm0,  %%ymm2            \n\t" // scale ymm0 by beta
	"vmulpd           %%ymm7,  %%ymm0,  %%ymm0   \n\t"
	"vmulpd           %%ymm6,  %%ymm2,  %%ymm2   \n\t"
	"vaddsubpd        %%ymm2,  %%ymm0,  %%ymm0   \n\t"
	"vaddpd           %%ymm10, %%ymm0,  %%ymm0   \n\t" // add the gemm result to ymm0
	"vmovapd          %%ymm0,  (%%rdx)           \n\t" // store c42:c72
	"addq      %%rdi, %%rdx                      \n\t" // c += cs_c;
	"                                            \n\t"
	"                                            \n\t" // update c03:c33
	"                                            \n\t"
	"vmovapd    (%%rcx),       %%ymm0            \n\t" // load c03:c33 into ymm0
	"vpermilpd  $0x5, %%ymm0,  %%ymm2            \n\t" // scale ymm0 by beta
	"vmulpd           %%ymm7,  %%ymm0,  %%ymm0   \n\t"
	"vmulpd           %%ymm6,  %%ymm2,  %%ymm2   \n\t"
	"vaddsubpd        %%ymm2,  %%ymm0,  %%ymm0   \n\t"
	"vaddpd           %%ymm9,  %%ymm0,  %%ymm0   \n\t" // add the gemm result to ymm0
	"vmovapd          %%ymm0,  (%%rcx)           \n\t" // store c03:c33
	"addq      %%rdi, %%rcx                      \n\t" // c += cs_c;
	"                                            \n\t"
	"                                            \n\t" // update c43:c73
	"                                            \n\t"
	"vmovapd    (%%rdx),       %%ymm0            \n\t" // load c43:c73 into ymm0
	"vpermilpd  $0x5, %%ymm0,  %%ymm2            \n\t" // scale ymm0 by beta
	"vmulpd           %%ymm7,  %%ymm0,  %%ymm0   \n\t"
	"vmulpd           %%ymm6,  %%ymm2,  %%ymm2   \n\t"
	"vaddsubpd        %%ymm2,  %%ymm0,  %%ymm0   \n\t"
	"vaddpd           %%ymm8,  %%ymm0,  %%ymm0   \n\t" // add the gemm result to ymm0
	"vmovapd          %%ymm0,  (%%rdx)           \n\t" // store c43:c73
	"                                            \n\t"
	"                                            \n\t"
	"                                            \n\t"
	"jmp    .ZDONE                               \n\t" // jump to end.
	"                                            \n\t"
	"                                            \n\t"
	"                                            \n\t"
	".ZBETAZERO:                                 \n\t"
	"                                            \n\t" // check if aligned/column-stored
	"                                            \n\t" // check if aligned/column-stored
	"andb     %%bl, %%bh                         \n\t" // set ZF if bl & bh == 1.
	"andb     %%bh, %%al                         \n\t" // set ZF if bh & al == 1.
	"jne     .ZCOLSTORBZ                         \n\t" // jump to column storage case
	"                                            \n\t"
	"                                            \n\t"
	"                                            \n\t"
	".ZGENSTORBZ:                                \n\t"
	"                                            \n\t" // update c00:c30
	"                                            \n\t"
	"vextractf128 $1, %%ymm15, %%xmm2            \n\t"
	"vmovupd          %%xmm15, (%%rcx)           \n\t" // store (c00,c10)
	"vmovupd          %%xmm2,  (%%rcx,%%rsi)     \n\t" // store (c20,c30)
	"addq      %%rdi, %%rcx                      \n\t" // c += cs_c;
	"                                            \n\t"
	"                                            \n\t" // update c40:c70
	"                                            \n\t"
	"vextractf128 $1, %%ymm14, %%xmm2            \n\t"
	"vmovupd          %%xmm14, (%%rdx)           \n\t" // store (c40,c50)
	"vmovupd          %%xmm2,  (%%rdx,%%rsi)     \n\t" // store (c60,c70)
	"addq      %%rdi, %%rdx                      \n\t" // c += cs_c;
	"                                            \n\t"
	"                                            \n\t" // update c01:c31
	"                                            \n\t"
	"vextractf128 $1, %%ymm13, %%xmm2            \n\t"
	"vmovupd          %%xmm13, (%%rcx)           \n\t" // store (c01,c11)
	"vmovupd          %%xmm2,  (%%rcx,%%rsi)     \n\t" // store (c21,c31)
	"addq      %%rdi, %%rcx                      \n\t" // c += cs_c;
	"                                            \n\t"
	"                                            \n\t" // update c41:c71
	"                                            \n\t"
	"vextractf128 $1, %%ymm12, %%xmm2            \n\t"
	"vmovupd          %%xmm12, (%%rdx)           \n\t" // store (c41,c51)
	"vmovupd          %%xmm2,  (%%rdx,%%rsi)     \n\t" // store (c61,c71)
	"addq      %%rdi, %%rdx                      \n\t" // c += cs_c;
	"                                            \n\t"
	"                                            \n\t" // update c02:c32
	"                                            \n\t"
	"vextractf128 $1, %%ymm11, %%xmm2            \n\t"
	"vmovupd          %%xmm11, (%%rcx)           \n\t" // store (c02,c12)
	"vmovupd          %%xmm2,  (%%rcx,%%rsi)     \n\t" // store (c22,c32)
	"addq      %%rdi, %%rcx                      \n\t" // c += cs_c;
	"                                            \n\t"
	"                                            \n\t" // update c42:c72
	"                                            \n\t"
	"vextractf128 $1, %%ymm10, %%xmm2            \n\t"
	"vmovupd          %%xmm10, (%%rdx)           \n\t" // store (c42,c52)
	"vmovupd          %%xmm2,  (%%rdx,%%rsi)     \n\t" // store (c62,c72)
	"addq      %%rdi, %%rdx                      \n\t" // c += cs_c;
	"                                            \n\t"
	"                                            \n\t" // update c03:c33
	"                                            \n\t"
	"vextractf128 $1, %%ymm9,  %%xmm2            \n\t"
	"vmovupd          %%xmm9,  (%%rcx)           \n\t" // store (c03,c13)
	"vmovupd          %%xmm2,  (%%rcx,%%rsi)     \n\t" // store (c23,c33)
	"addq      %%rdi, %%rcx                      \n\t" // c += cs_c;
	"                                            \n\t"
	"                                            \n\t" // update c43:c73
	"                                            \n\t"
	"vextractf128 $1, %%ymm8,  %%xmm2            \n\t"
	"vmovupd          %%xmm8,  (%%rdx)           \n\t" // store (c43,c53)
	"vmovupd          %%xmm2,  (%%rdx,%%rsi)     \n\t" // store (c63,c73)
	"                                            \n\t"
	"                                            \n\t"
	"                                            \n\t"
	"jmp    .ZDONE                               \n\t" // jump to end.
	"                                            \n\t"
	"                                            \n\t"
	"                                            \n\t"
	".ZCOLSTORBZ:                                \n\t"
	"                                            \n\t"
	"                                            \n\t"
	"vmovapd          %%ymm15, (%%rcx)           \n\t" // store c00:c30
	"addq      %%rdi, %%rcx                      \n\t" // c += cs_c;
	"                                            \n\t"
	"vmovapd          %%ymm14, (%%rdx)           \n\t" // store c40:c70
	"addq      %%rdi, %%rdx                      \n\t" // c += cs_c;
	"                                            \n\t"
	"vmovapd          %%ymm13, (%%rcx)           \n\t" // store c01:c31
	"addq      %%rdi, %%rcx                      \n\t" // c += cs_c;
	"                                            \n\t"
	"vmovapd          %%ymm12, (%%rdx)           \n\t" // store c41:c71
	"addq      %%rdi, %%rdx                      \n\t" // c += cs_c;
	"                                            \n\t"
	"vmovapd          %%ymm11, (%%rcx)           \n\t" // store c02:c32
	"addq      %%rdi, %%rcx                      \n\t" // c += cs_c;
	"                                            \n\t"
	"vmovapd          %%ymm10, (%%rdx)           \n\t" // store c42:c72
	"addq      %%rdi, %%rdx                      \n\t" // c += cs_c;
	"                                            \n\t"
	"vmovapd          %%ymm9,  (%%rcx)           \n\t" // store c03:c33
	"addq      %%rdi, %%rcx                      \n\t" // c += cs_c;
	"                                            \n\t"
	"vmovapd          %%ymm8,  (%%rdx)           \n\t" // store c43:c73
	"                                            \n\t"
	"                                            \n\t"
	"                                            \n\t"
	"                                            \n\t"
	"                                            \n\t"
	".ZDONE:                                     \n\t"
	"                                            \n\t"

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
	  "rax", "rbx", "rcx", "rdx", "rsi", "rdi", 
	  "r8", "r9", "r10", "r11", "r12", "r13", "r14", "r15",
	  "xmm0", "xmm1", "xmm2", "xmm3",
	  "xmm4", "xmm5", "xmm6", "xmm7",
	  "xmm8", "xmm9", "xmm10", "xmm11",
	  "xmm12", "xmm13", "xmm14", "xmm15",
	  "memory"
	);
}

