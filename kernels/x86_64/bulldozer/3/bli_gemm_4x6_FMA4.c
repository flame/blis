/*

   BLIS    
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2014, The University of Texas

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
   THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
   (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
   OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

*/

#include "blis.h"

void bli_sgemm_8x8_FMA4(
                        dim_t              k,
                        float* restrict    alpha,
                        float* restrict    a,
                        float* restrict    b,
                        float* restrict    beta,
                        float* restrict    c, inc_t rs_c, inc_t cs_c,
                        auxinfo_t*         data
                      )
{
	dim_t   k_iter = k / 4;
	dim_t   k_left = k % 4;

	__asm__ volatile
	(
	"                                           \n\t"
	"movq                %2, %%rax              \n\t" // load address of a.
	"movq                %3, %%rbx              \n\t" // load address of b.
	"                                           \n\t"
	"vmovaps    0 * 32(%%rax), %%ymm0           \n\t" // initialize loop by pre-loading
	"vmovsldup  0 * 32(%%rbx), %%ymm2			\n\t" // elements of a and b.
	"vpermilps   $0x4e, %%ymm2, %%ymm3          \n\t"
	"                                           \n\t"
	"movq                %6, %%rcx              \n\t" // load address of c
	"movq                %8, %%rdi              \n\t" // load cs_c
	"leaq        (,%%rdi,4), %%rdi              \n\t" // cs_c *= sizeof(float)
	"leaq   (%%rcx,%%rdi,4), %%r10              \n\t" // load address of c + 4*cs_c;
	"                                           \n\t"
	"leaq   (%%rdi,%%rdi,2), %%r14              \n\t" // r14 = 3*cs_c;
	"prefetcht0   7 * 8(%%rcx)                  \n\t" // prefetch c + 0*cs_c
	"prefetcht0   7 * 8(%%rcx,%%rdi)            \n\t" // prefetch c + 1*cs_c
	"prefetcht0   7 * 8(%%rcx,%%rdi,2)          \n\t" // prefetch c + 2*cs_c
	"prefetcht0   7 * 8(%%rcx,%%r14)            \n\t" // prefetch c + 3*cs_c
	"prefetcht0   7 * 8(%%r10)                  \n\t" // prefetch c + 4*cs_c
	"prefetcht0   7 * 8(%%r10,%%rdi)            \n\t" // prefetch c + 5*cs_c
	"prefetcht0   7 * 8(%%r10,%%rdi,2)          \n\t" // prefetch c + 6*cs_c
	"prefetcht0   7 * 8(%%r10,%%r14)            \n\t" // prefetch c + 7*cs_c
	"                                           \n\t"
	"vxorps    %%ymm8,  %%ymm8,  %%ymm8         \n\t"
	"vxorps    %%ymm9,  %%ymm9,  %%ymm9         \n\t"
	"vxorps    %%ymm10, %%ymm10, %%ymm10        \n\t"
	"vxorps    %%ymm11, %%ymm11, %%ymm11        \n\t"
	"vxorps    %%ymm12, %%ymm12, %%ymm12        \n\t"
	"vxorps    %%ymm13, %%ymm13, %%ymm13        \n\t"
	"vxorps    %%ymm14, %%ymm14, %%ymm14        \n\t"
	"vxorps    %%ymm15, %%ymm15, %%ymm15        \n\t"
	"                                           \n\t"
	"                                           \n\t"
	"movq      %0, %%rsi                        \n\t" // i = k_iter;
	"testq  %%rsi, %%rsi                        \n\t" // check i via logical AND.
	"je     .SCONSIDKLEFT                       \n\t" // if i == 0, jump to code that
	"                                           \n\t" // contains the k_left loop.
	"                                            \n\t"
	".SLOOPKITER:                                \n\t" // MAIN LOOP
	"                                            \n\t"
	"                                            \n\t" // iteration 0
	"prefetcht0  16 * 32(%%rax)                  \n\t"
	"vfmaddps	%%ymm15, %%ymm0,  %%ymm2, %%ymm15\n\t"
	"vperm2f128 $0x03, %%ymm2,  %%ymm2, %%ymm4   \n\t"
	"vmovshdup  0 * 32(%%rbx),  %%ymm2          \n\t"
	"vfmaddps	%%ymm13, %%ymm0,  %%ymm3, %%ymm13\n\t"
	"vperm2f128 $0x03, %%ymm3,  %%ymm3, %%ymm5   \n\t"
	"                                            \n\t"
	"vmovaps    1 * 32(%%rax),  %%ymm1           \n\t"
	"vpermilps  $0x4e, %%ymm2,  %%ymm3           \n\t"
	"vfmaddps	%%ymm11, %%ymm0,  %%ymm4, %%ymm11\n\t"
	"vfmaddps	%%ymm9, %%ymm0,  %%ymm5, %%ymm9	\n\t"
	"                                            \n\t"
	"vfmaddps	%%ymm14, %%ymm0,  %%ymm2, %%ymm14\n\t"
	"vperm2f128 $0x03, %%ymm2,  %%ymm2, %%ymm4   \n\t"
	"vmovsldup  1 * 32(%%rbx),  %%ymm2           \n\t"
	"vfmaddps	%%ymm12, %%ymm0,  %%ymm3, %%ymm12\n\t"
	"vperm2f128 $0x03, %%ymm3,  %%ymm3, %%ymm5   \n\t"
	"                                            \n\t"
	"vpermilps  $0x4e, %%ymm2,  %%ymm3           \n\t"
	"vfmaddps	%%ymm10, %%ymm0,  %%ymm4, %%ymm10\n\t"
	"vfmaddps	%%ymm8, %%ymm0,  %%ymm5, %%ymm8\n\t"
	"                                            \n\t"
	"                                            \n\t" // iteration 1
	"vfmaddps	%%ymm15, %%ymm1,  %%ymm2, %%ymm15\n\t"
	"vperm2f128 $0x03, %%ymm2,  %%ymm2, %%ymm4   \n\t"
	"vmovshdup  1 * 32(%%rbx), %%ymm2            \n\t"
	"vfmaddps	%%ymm13, %%ymm1,  %%ymm3, %%ymm13\n\t"
	"vperm2f128 $0x03, %%ymm3,  %%ymm3, %%ymm5   \n\t"
	"                                            \n\t"
	"vmovaps    2 * 32(%%rax),  %%ymm0           \n\t"
	"vpermilps  $0x4e, %%ymm2,  %%ymm3           \n\t"
	"vfmaddps	%%ymm11, %%ymm1,  %%ymm4, %%ymm11\n\t"
	"vfmaddps	%%ymm9, %%ymm1,  %%ymm5, %%ymm9\n\t"
	"                                            \n\t"
	"vfmaddps	%%ymm14, %%ymm1,  %%ymm2, %%ymm14\n\t"
	"vperm2f128 $0x03, %%ymm2,  %%ymm2, %%ymm4   \n\t"
	"vmovsldup  2 * 32(%%rbx),  %%ymm2           \n\t"
	"vfmaddps	%%ymm12, %%ymm1,  %%ymm3, %%ymm12\n\t"
	"vperm2f128 $0x03, %%ymm3,  %%ymm3, %%ymm5   \n\t"
	"                                            \n\t"
	"vpermilps  $0x4e, %%ymm2,  %%ymm3           \n\t"
	"vfmaddps	%%ymm10, %%ymm1,  %%ymm4, %%ymm10\n\t"
	"vfmaddps	%%ymm8, %%ymm1,  %%ymm5, %%ymm8\n\t"
	"                                            \n\t"
	"                                            \n\t" // iteration 2
	"prefetcht0  18 * 32(%%rax)                  \n\t"
	"vfmaddps	%%ymm15, %%ymm0,  %%ymm2, %%ymm15\n\t"
	"vperm2f128 $0x03, %%ymm2,  %%ymm2, %%ymm4   \n\t"
	"vmovshdup  2 * 32(%%rbx),  %%ymm2           \n\t"
	"vfmaddps	%%ymm13, %%ymm0,  %%ymm3, %%ymm13\n\t"
	"vperm2f128 $0x03, %%ymm3,  %%ymm3, %%ymm5   \n\t"
	"                                            \n\t"
	"vmovaps    3 * 32(%%rax),  %%ymm1           \n\t"
	"addq           $4 * 8 * 4, %%rax            \n\t" // a += 4*8 (unroll x mr)
	"vpermilps  $0x4e, %%ymm2,  %%ymm3           \n\t"
	"vfmaddps	%%ymm11, %%ymm0,  %%ymm4, %%ymm11\n\t"
	"vfmaddps	%%ymm9, %%ymm0,  %%ymm5, %%ymm9\n\t"
	"                                            \n\t"
	"vfmaddps	%%ymm14, %%ymm0,  %%ymm2, %%ymm14\n\t"
	"vperm2f128 $0x03, %%ymm2,  %%ymm2, %%ymm4   \n\t"
	"vmovsldup  3 * 32(%%rbx),  %%ymm2           \n\t"
	"vfmaddps	%%ymm12, %%ymm0,  %%ymm3, %%ymm12\n\t"
	"vperm2f128 $0x03, %%ymm3,  %%ymm3, %%ymm5   \n\t"
	"                                            \n\t"
	"vpermilps  $0x4e, %%ymm2,  %%ymm3           \n\t"
	"vfmaddps	%%ymm10, %%ymm0,  %%ymm4, %%ymm10\n\t"
	"vfmaddps	%%ymm8, %%ymm0,  %%ymm5, %%ymm8\n\t"
	"                                            \n\t"
	"                                            \n\t" // iteration 3
	"vfmaddps	%%ymm15, %%ymm1,  %%ymm2, %%ymm15\n\t"
	"vperm2f128 $0x03, %%ymm2,  %%ymm2, %%ymm4   \n\t"
	"vmovshdup  3 * 32(%%rbx), %%ymm2            \n\t"
	"addq           $4 * 8 * 4, %%rbx            \n\t" // b += 4*8 (unroll x nr)
	"vfmaddps	%%ymm13, %%ymm1,  %%ymm3, %%ymm13\n\t"
	"vperm2f128 $0x03, %%ymm3,  %%ymm3, %%ymm5   \n\t"
	"                                            \n\t"
	"vmovaps    0 * 32(%%rax),  %%ymm0           \n\t"
	"vpermilps  $0x4e, %%ymm2,  %%ymm3           \n\t"
	"vfmaddps	%%ymm11, %%ymm1,  %%ymm4, %%ymm11\n\t"
	"vfmaddps	%%ymm9, %%ymm1,  %%ymm5, %%ymm9\n\t"
	"                                            \n\t"
	"vfmaddps	%%ymm14, %%ymm1,  %%ymm2, %%ymm14\n\t"
	"vperm2f128 $0x03, %%ymm2,  %%ymm2, %%ymm4   \n\t"
	"vmovsldup  0 * 32(%%rbx),  %%ymm2           \n\t"
	"vfmaddps	%%ymm12, %%ymm1,  %%ymm3, %%ymm12\n\t"
	"vperm2f128 $0x03, %%ymm3,  %%ymm3, %%ymm5   \n\t"
	"                                            \n\t"
	"vpermilps  $0x4e, %%ymm2,  %%ymm3           \n\t"
	"vfmaddps	%%ymm10, %%ymm1,  %%ymm4, %%ymm10\n\t"
	"vfmaddps	%%ymm8, %%ymm1,  %%ymm5, %%ymm8\n\t"
	"                                            \n\t"
	"                                            \n\t"
	"                                            \n\t"
	"decq   %%rsi                                \n\t" // i -= 1;
	"jne    .SLOOPKITER                          \n\t" // iterate again if i != 0.
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
	".SLOOPKLEFT:                                \n\t" // EDGE LOOP
	"                                            \n\t"
	"prefetcht0  16 * 32(%%rax)                  \n\t"
	"vfmaddps	%%ymm15, %%ymm0,  %%ymm2, %%ymm15\n\t"
	"vperm2f128  $0x3, %%ymm2,  %%ymm2, %%ymm4   \n\t"
	"vmovshdup  0 * 32(%%rbx),  %%ymm2           \n\t"
	"vfmaddps	%%ymm13, %%ymm0,  %%ymm3, %%ymm13\n\t"
	"vperm2f128  $0x3, %%ymm3,  %%ymm3, %%ymm5   \n\t"
	"                                            \n\t"
	"vmovaps    1 * 32(%%rax),  %%ymm1           \n\t"
	"addq           $8 * 1 * 4, %%rax            \n\t" // a += 8 (1 x mr)
	"vpermilps  $0x4e, %%ymm2,  %%ymm3           \n\t"
	"vfmaddps	%%ymm11, %%ymm0,  %%ymm4, %%ymm11\n\t"
	"vfmaddps	%%ymm9, %%ymm0,  %%ymm5, %%ymm9\n\t"
	"                                            \n\t"
	"vfmaddps	%%ymm14, %%ymm0,  %%ymm2, %%ymm14\n\t"
	"vperm2f128  $0x3, %%ymm2,  %%ymm2, %%ymm4   \n\t"
	"vmovsldup  1 * 32(%%rbx),  %%ymm2           \n\t"
	"addq           $8 * 1 * 4, %%rbx            \n\t" // b += 8 (1 x nr)
	"vfmaddps	%%ymm12, %%ymm0,  %%ymm3, %%ymm12\n\t"
	"vmulps            %%ymm0,  %%ymm3, %%ymm7   \n\t"
	"                                            \n\t"
	"vpermilps  $0x4e, %%ymm2,  %%ymm3           \n\t"
	"vfmaddps	%%ymm10, %%ymm0,  %%ymm4, %%ymm10\n\t"
	"vfmaddps	%%ymm8, %%ymm0,  %%ymm5, %%ymm8\n\t"
	"vmovaps           %%ymm1,  %%ymm0           \n\t"
	"                                            \n\t"
	"                                            \n\t"
	"decq   %%rsi                                \n\t" // i -= 1;
	"jne    .SLOOPKLEFT                          \n\t" // iterate again if i != 0.
	"                                            \n\t"
	"                                            \n\t"
	".SPOSTACCUM:                                \n\t"
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
	"movq                %7, %%rsi               \n\t" // load rs_c
	"leaq        (,%%rsi,4), %%rsi               \n\t" // rsi = rs_c * sizeof(float)
	"                                            \n\t"
	"leaq   (%%rcx,%%rsi,4), %%rdx               \n\t" // load address of c + 4*rs_c;
	"                                            \n\t"
	"leaq        (,%%rsi,2), %%r12               \n\t" // r12 = 2*rs_c;
	"leaq   (%%r12,%%rsi,1), %%r13               \n\t" // r13 = 3*rs_c;
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
	".SGENSTORED:                                \n\t"
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
	"vfmaddps	%%ymm15, %%ymm0,  %%ymm4, %%ymm0\n\t"	// scale by beta and add the gemm result,
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

#undef KERNEL4x6_1
#undef KERNEL4x6_2
#undef KERNEL4x6_3
#undef KERNEL4x6_4

#define KERNEL4x6_1(xx) \
		".align 4                       					\n\t"\
		"vmovddup -8 * 8(%%rax), %%xmm0   					\n\t"\
		"vfmaddpd %%xmm4,  %%xmm1, %%xmm0, %%xmm4		\n\t"\
		"vfmaddpd %%xmm5,  %%xmm2, %%xmm0, %%xmm5		\n\t"\
		"vfmaddpd %%xmm6,  %%xmm3, %%xmm0, %%xmm6		        \n\t"\
		"vmovddup -7 * 8(%%rax), %%xmm0   					\n\t"\
		"vfmaddpd %%xmm7,  %%xmm1, %%xmm0, %%xmm7		        \n\t"\
		"prefetcht0 128(%%rax)								\n\t"\
		"vfmaddpd %%xmm8,  %%xmm2, %%xmm0, %%xmm8		        \n\t"\
		"vfmaddpd %%xmm9,  %%xmm3, %%xmm0, %%xmm9                \n\t"\
		"vmovddup -6 * 8(%%rax), %%xmm0   					\n\t"\
		"vfmaddpd %%xmm10,  %%xmm1, %%xmm0, %%xmm10               \n\t"\
		"vfmaddpd %%xmm11,  %%xmm2, %%xmm0, %%xmm11               \n\t"\
		"vfmaddpd %%xmm12,  %%xmm3, %%xmm0, %%xmm12               \n\t"\
		"vmovddup -5 * 8(%%rax), %%xmm0   	                \n\t"\
		"vfmaddpd %%xmm13,  %%xmm1, %%xmm0, %%xmm13		        \n\t"\
		"vmovaps -6 * 8(%%rbx), %%xmm1   					\n\t"\
		"vfmaddpd %%xmm14,  %%xmm2, %%xmm0, %%xmm14		        \n\t"\
		"vmovaps -4 * 8(%%rbx), %%xmm2   			        \n\t"\
		"vfmaddpd %%xmm15,  %%xmm3, %%xmm0, %%xmm15		        \n\t"\
		"vmovaps -2 * 8(%%rbx), %%xmm3                      \n\t"

#define KERNEL4x6_2(xx) \
		"vmovddup -4 * 8(%%rax), %%xmm0                     \n\t"\
		"vfmaddpd %%xmm4,  %%xmm1, %%xmm0, %%xmm4                \n\t"\
		"prefetcht0 192(%%rax)                              \n\t"\
		"vfmaddpd %%xmm5,  %%xmm2, %%xmm0, %%xmm5                \n\t"\
		"vfmaddpd %%xmm6,  %%xmm3, %%xmm0, %%xmm6                \n\t"\
		"vmovddup -3 * 8(%%rax), %%xmm0                     \n\t"\
		"vfmaddpd %%xmm7,  %%xmm1, %%xmm0, %%xmm7                \n\t"\
		"vfmaddpd %%xmm8,  %%xmm2, %%xmm0, %%xmm8                \n\t"\
		"vfmaddpd %%xmm9,  %%xmm3, %%xmm0, %%xmm9                \n\t"\
		"vmovddup -2 * 8(%%rax), %%xmm0                     \n\t"\
		"vfmaddpd %%xmm10,  %%xmm1, %%xmm0, %%xmm10               \n\t"\
		"vfmaddpd %%xmm11,  %%xmm2, %%xmm0, %%xmm11               \n\t"\
		"vfmaddpd %%xmm12,  %%xmm3, %%xmm0, %%xmm12               \n\t"\
		"vmovddup -1 * 8(%%rax), %%xmm0                     \n\t"\
		"vfmaddpd %%xmm13,  %%xmm1, %%xmm0, %%xmm13               \n\t"\
		"vmovaps 0 * 8(%%rbx), %%xmm1                       \n\t"\
		"vfmaddpd %%xmm14,  %%xmm2, %%xmm0, %%xmm14               \n\t"\
		"vmovaps 2 * 8(%%rbx), %%xmm2                       \n\t"\
		"vfmaddpd %%xmm15, %%xmm3, %%xmm0, %%xmm15                \n\t"\
		"vmovaps 4 * 8(%%rbx), %%xmm3                       \n\t"\

#define KERNEL4x6_3(xx) \
		"vmovddup  0 * 8(%%rax), %%xmm0   					\n\t"\
		"vfmaddpd %%xmm4,  %%xmm1, %%xmm0, %%xmm4		        \n\t"\
		"vfmaddpd %%xmm5,  %%xmm2, %%xmm0, %%xmm5		        \n\t"\
		"vfmaddpd %%xmm6,  %%xmm3, %%xmm0, %%xmm6		        \n\t"\
		"vmovddup  1 * 8(%%rax), %%xmm0   					\n\t"\
		"vfmaddpd %%xmm7,  %%xmm1, %%xmm0, %%xmm7		        \n\t"\
		"prefetcht0 224(%%rax)								\n\t"\
		"vfmaddpd %%xmm8,  %%xmm2, %%xmm0, %%xmm8		        \n\t"\
		"vfmaddpd %%xmm9,  %%xmm3, %%xmm0, %%xmm9                \n\t"\
		"vmovddup  2 * 8(%%rax), %%xmm0   					\n\t"\
		"vfmaddpd %%xmm10,  %%xmm1, %%xmm0, %%xmm10               \n\t"\
		"vfmaddpd %%xmm11,  %%xmm2, %%xmm0, %%xmm11               \n\t"\
		"vfmaddpd %%xmm12,  %%xmm3, %%xmm0, %%xmm12               \n\t"\
		"vmovddup  3 * 8(%%rax), %%xmm0   	                \n\t"\
		"vfmaddpd %%xmm13,  %%xmm1, %%xmm0, %%xmm13		        \n\t"\
		"vmovaps  6 * 8(%%rbx), %%xmm1   					\n\t"\
		"vfmaddpd %%xmm14,  %%xmm2, %%xmm0, %%xmm14		        \n\t"\
		"vmovaps  8 * 8(%%rbx), %%xmm2   			        \n\t"\
		"vfmaddpd %%xmm15,  %%xmm3, %%xmm0, %%xmm15		        \n\t"\
		"vmovaps 10 * 8(%%rbx), %%xmm3                      \n\t"

#define KERNEL4x6_4(xx) \
		"vmovddup  4 * 8(%%rax), %%xmm0                     \n\t"\
		"vfmaddpd %%xmm4,  %%xmm1, %%xmm0, %%xmm4                \n\t"\
		"prefetcht0 224(%%rax)                              \n\t"\
		"vfmaddpd %%xmm5,  %%xmm2, %%xmm0, %%xmm5                \n\t"\
		"vfmaddpd %%xmm6,  %%xmm3, %%xmm0, %%xmm6                \n\t"\
		"vmovddup  5 * 8(%%rax), %%xmm0                     \n\t"\
		"vfmaddpd %%xmm7,  %%xmm1, %%xmm0, %%xmm7                \n\t"\
		"vfmaddpd %%xmm8,  %%xmm2, %%xmm0, %%xmm8                \n\t"\
		"vfmaddpd %%xmm9,  %%xmm3, %%xmm0, %%xmm9                \n\t"\
		"vmovddup  6 * 8(%%rax), %%xmm0                     \n\t"\
		"vfmaddpd %%xmm10,  %%xmm1, %%xmm0, %%xmm10               \n\t"\
		"vfmaddpd %%xmm11,  %%xmm2, %%xmm0, %%xmm11               \n\t"\
		"vfmaddpd %%xmm12,  %%xmm3, %%xmm0, %%xmm12               \n\t"\
		"vmovddup  7 * 8(%%rax), %%xmm0                     \n\t"\
		"vfmaddpd %%xmm13,  %%xmm1, %%xmm0, %%xmm13               \n\t"\
		"vmovaps  12 * 8(%%rbx), %%xmm1                       \n\t"\
		"vfmaddpd %%xmm14,  %%xmm2, %%xmm0, %%xmm14               \n\t"\
		"vmovaps  14 * 8(%%rbx), %%xmm2                       \n\t"\
		"vfmaddpd %%xmm15, %%xmm3, %%xmm0, %%xmm15                \n\t"\
		"addq       $16*8, %%rax		                    \n\t"\
		"vmovaps  16 * 8(%%rbx), %%xmm3                       \n\t"\
		"addq       $24*8, %%rbx		                    \n\t"

void bli_dgemm_4x6_FMA4(
                    dim_t              k,
                    double*   restrict alpha,
                    double*   restrict a,
                    double*   restrict b,
                    double*   restrict beta,
                    double*   restrict c, inc_t rs_c, inc_t cs_c,
                    auxinfo_t*         data
                  )
{
	dim_t k_iter  = k / 12;
	dim_t k_left  = k % 12;

	__asm__ 
	(	
		"                                \n\t"
		"                                \n\t"
		"vzeroall                        \n\t"
		"movq          %3, %%rbx         \n\t" // load address of b.
		"movq          %2, %%rax         \n\t" // load address of a.
		"prefetcht0 64(%%rax)								\n\t"
		"                                \n\t"
		"                                \n\t"
		"vmovaps 0 * 8(%%rbx), %%xmm1   					\n\t"
		"vmovaps 2 * 8(%%rbx), %%xmm2   					\n\t"
		"vmovaps 4 * 8(%%rbx), %%xmm3   					\n\t"
		"addq		$12*8, %%rbx		                    \n\t"
		"addq		$8*8, %%rax 		                    \n\t"
		"                                                   \n\t"
		"movq      %0, %%rsi                                \n\t" // i = k_iter; notice %0 not $0
		"testq  %%rsi, %%rsi                                \n\t"
		"je .CONSIDERKLEFT   								\n\t"
		"                       					        \n\t"
		".align 32              					        \n\t"
		".LOOPKITER:                                        \n\t" // MAIN LOOP
		"                       					        \n\t"
        KERNEL4x6_1(xx)
        KERNEL4x6_2(xx)
        KERNEL4x6_3(xx)
        KERNEL4x6_4(xx)
        KERNEL4x6_1(xx)
        KERNEL4x6_2(xx)
        KERNEL4x6_3(xx)
        KERNEL4x6_4(xx)
        KERNEL4x6_1(xx)
        KERNEL4x6_2(xx)
        KERNEL4x6_3(xx)
        KERNEL4x6_4(xx)
		"                       					        \n\t"
		"decq %%rsi             					        \n\t"
		"jne .LOOPKITER										\n\t"
		"                       					        \n\t"
		".CONSIDERKLEFT:           					        \n\t"
		"                       					        \n\t"
		"movq %1, %%rsi            					        \n\t"
		"testq %%rsi, %%rsi       					        \n\t" 
		".LOOPKLEFT:               					        \n\t"
		"je .POSTACCUM            					        \n\t"
		"                       					        \n\t"
        KERNEL4x6_1(xx)
		"addq     $6*8, %%rbx		                        \n\t"
		"addq       $4*8, %%rax		                        \n\t"
		"                       					        \n\t"
		"decq %%rsi             					        \n\t"
		"jmp .LOOPKLEFT               						\n\t" // iterate again if i != 0.
		"                       					        \n\t"
		".POSTACCUM:                    \n\t"
		"                               \n\t"
		"                               \n\t"
		"movq    %7, %%rsi              \n\t" // load cs_c
		"movq    %8, %%rdi              \n\t" // load rs_c
		"vmovddup (%4), %%xmm2		    \n\t" //load alpha
		"vmovddup (%5), %%xmm3		    \n\t" //load beta
		"movq    %6, %%rcx          	\n\t" // load address of c
		"salq    $3, %%rsi              \n\t" // cs_c *= sizeof(double)
		"salq    $3, %%rdi              \n\t" // rs_c *= sizeof(double)
		"leaq    (%%rcx, %%rdi,2), %%rdx 	\n\t"
		"                                	\n\t"
		"vmovlpd  (%%rcx),       %%xmm0, %%xmm0   	\n\t" 		
		"vmovlpd  (%%rdx),       %%xmm1, %%xmm1   	\n\t" 			
		"vmovhpd  (%%rcx,%%rdi), %%xmm0, %%xmm0   	\n\t"
		"vmovhpd  (%%rdx,%%rdi), %%xmm1, %%xmm1   	\n\t"
		"leaq     (%%rdx, %%rdi,2), %%r8 	\n\t"
		"vmulpd   %%xmm2,  %%xmm4, %%xmm4         	\n\t"			// scale by alpha,
		"vmulpd   %%xmm2,  %%xmm5, %%xmm5         	\n\t"			// scale by alpha,
		"vfmaddpd %%xmm4, %%xmm0, %%xmm3, %%xmm4         \n\t"	// scale by beta, and add the gemm result
		"vmovlpd  (%%r8),       %%xmm0, %%xmm0   	\n\t" 			
		"vfmaddpd %%xmm5, %%xmm1, %%xmm3, %%xmm5         \n\t"	// scale by beta, and add the gemm result
		"vmovhpd  (%%r8,%%rdi), %%xmm0, %%xmm0   	\n\t"
		"vmovlpd  %%xmm4,  (%%rcx)        	\n\t" 			// and store back to memory.
		"vmovlpd  %%xmm5,  (%%rdx)        	\n\t" 			// and store back to memory.
		"vmovhpd  %%xmm4,  (%%rcx,%%rdi)  	\n\t"
		"addq %%rsi, %%rcx				   	\n\t" 
		"vmovhpd  %%xmm5,  (%%rdx,%%rdi)  	\n\t"
		"addq %%rsi, %%rdx				   	\n\t" 
		"                                	\n\t"
		"vmulpd   %%xmm2,  %%xmm6, %%xmm6         	\n\t"			// scale by alpha,
		"vfmaddpd %%xmm6,   %%xmm0, %%xmm3, %%xmm6       \n\t"	// scale by beta, and add the gemm result
		"vmovlpd  %%xmm6,  (%%r8)        	\n\t" 			// and store back to memory.
		"vmovhpd  %%xmm6,  (%%r8,%%rdi)  	\n\t"
		"addq %%rsi, %%r8				   	\n\t" 
		"                                	\n\t"
		"                                	\n\t"
		"vmovlpd  (%%rcx),       %%xmm0, %%xmm0   	\n\t" 			
		"vmovlpd  (%%rdx),       %%xmm1, %%xmm1   	\n\t" 			
		"vmovlpd  (%%r8),        %%xmm4, %%xmm4   	\n\t" 			
		"vmovhpd  (%%rcx,%%rdi), %%xmm0, %%xmm0   	\n\t"
		"vmovhpd  (%%rdx,%%rdi), %%xmm1, %%xmm1   	\n\t"
		"vmovhpd  (%%r8,%%rdi),  %%xmm4, %%xmm4   	\n\t"
		"vmulpd   %%xmm2,  %%xmm7, %%xmm7         	\n\t"			// scale by alpha,
		"vmulpd   %%xmm2,  %%xmm8, %%xmm8         	\n\t"			// scale by alpha,
		"vmulpd   %%xmm2,  %%xmm9, %%xmm9         	\n\t"			// scale by alpha,
		"vfmaddpd %%xmm7,   %%xmm0, %%xmm3, %%xmm7       \n\t"	// scale by beta, and add the gemm result
		"vfmaddpd %%xmm8,   %%xmm1, %%xmm3, %%xmm8       \n\t"	// scale by beta, and add the gemm result
		"vfmaddpd %%xmm9,   %%xmm4, %%xmm3, %%xmm9       \n\t"	// scale by beta, and add the gemm result
		"vmovlpd  %%xmm7,  (%%rcx)        	\n\t" 			// and store back to memory.
		"vmovlpd  %%xmm8,  (%%rdx)        	\n\t" 			// and store back to memory.
		"vmovlpd  %%xmm9,  (%%r8)        	\n\t" 			// and store back to memory.
		"vmovhpd  %%xmm7,  (%%rcx,%%rdi)  	\n\t"
		"addq %%rsi, %%rcx				   	\n\t" 
		"vmovhpd  %%xmm8,  (%%rdx,%%rdi)  	\n\t"
		"addq %%rsi, %%rdx				   	\n\t" 
		"vmovhpd  %%xmm9,  (%%r8,%%rdi)  	\n\t"
		"addq %%rsi, %%r8				   	\n\t" 
		"                                	\n\t"
		"                                	\n\t"
		"vmovlpd  (%%rcx),       %%xmm0, %%xmm0   	\n\t" 			
		"vmovlpd  (%%rdx),       %%xmm1, %%xmm1   	\n\t" 			
		"vmovlpd  (%%r8),       %%xmm4, %%xmm4   	\n\t" 			
		"vmovhpd  (%%rcx,%%rdi), %%xmm0, %%xmm0   	\n\t"
		"vmovhpd  (%%rdx,%%rdi), %%xmm1, %%xmm1   	\n\t"
		"vmovhpd  (%%r8,%%rdi), %%xmm4, %%xmm4   	\n\t"
		"vmulpd   %%xmm2,  %%xmm10, %%xmm10         	\n\t"			// scale by alpha,
		"vmulpd   %%xmm2,  %%xmm11, %%xmm11         	\n\t"			// scale by alpha,
		"vmulpd   %%xmm2,  %%xmm12, %%xmm12         	\n\t"			// scale by alpha,
		"vfmaddpd %%xmm10,   %%xmm0, %%xmm3, %%xmm10      \n\t"	// scale by beta, and add the gemm result
		"vfmaddpd %%xmm11,   %%xmm1, %%xmm3, %%xmm11      \n\t"	// scale by beta, and add the gemm result
		"vfmaddpd %%xmm12,   %%xmm4, %%xmm3, %%xmm12      \n\t"	// scale by beta, and add the gemm result
		"vmovlpd  %%xmm10,  (%%rcx)        	\n\t" 			// and store back to memory.
		"vmovlpd  %%xmm11,  (%%rdx)        	\n\t" 			// and store back to memory.
		"vmovlpd  %%xmm12,  (%%r8)        	\n\t" 			// and store back to memory.
		"vmovhpd  %%xmm10,  (%%rcx,%%rdi)  	\n\t"
		"addq %%rsi, %%rcx				   	\n\t" 
		"vmovhpd  %%xmm11,  (%%rdx,%%rdi)  	\n\t"
		"addq %%rsi, %%rdx				   	\n\t" 
		"vmovhpd  %%xmm12,  (%%r8,%%rdi)  	\n\t"
		"addq %%rsi, %%r8				   	\n\t" 
		"                                	\n\t"
		"                                	\n\t"
		"vmovlpd  (%%rcx),       %%xmm0, %%xmm0   	\n\t" 			
		"vmovlpd  (%%rdx),       %%xmm1, %%xmm1   	\n\t" 			
		"vmovlpd  (%%r8),       %%xmm4, %%xmm4   	\n\t" 			
		"vmovhpd  (%%rcx,%%rdi), %%xmm0, %%xmm0   	\n\t"
		"vmovhpd  (%%rdx,%%rdi), %%xmm1, %%xmm1   	\n\t"
		"vmovhpd  (%%r8,%%rdi), %%xmm4, %%xmm4   	\n\t"
		"vmulpd   %%xmm2,  %%xmm13, %%xmm13         	\n\t"			// scale by alpha,
		"vmulpd   %%xmm2,  %%xmm14, %%xmm14         	\n\t"			// scale by alpha,
		"vmulpd   %%xmm2,  %%xmm15, %%xmm15         	\n\t"			// scale by alpha,
		"vfmaddpd %%xmm13,   %%xmm0, %%xmm3, %%xmm13          \n\t"	// scale by beta, and add the gemm result
		"vfmaddpd %%xmm14,   %%xmm1, %%xmm3, %%xmm14          \n\t"	// scale by beta, and add the gemm result
		"vfmaddpd %%xmm15,   %%xmm4, %%xmm3, %%xmm15          \n\t"	// scale by beta, and add the gemm result
		"vmovlpd  %%xmm13,  (%%rcx)        	\n\t" 			// and store back to memory.
		"vmovlpd  %%xmm14,  (%%rdx)        	\n\t" 			// and store back to memory.
		"vmovlpd  %%xmm15,  (%%r8)        	\n\t" 			// and store back to memory.
		"vmovhpd  %%xmm13,  (%%rcx,%%rdi)  	\n\t"
		"vmovhpd  %%xmm14,  (%%rdx,%%rdi)  	\n\t"
		"vmovhpd  %%xmm15,  (%%r8,%%rdi)  	\n\t" 

		: // output operands (none)
		: // input operands
		  "r" (k_iter),
		  "r" (k_left),
		  "r" (a),
		  "r" (b),
		  "r" (alpha),
		  "r" (beta),
		  "r" (c),
		  "m" (rs_c),
		  "m" (cs_c)
		: // register clobber list
		  "rax", "rbx", "rcx", "rdx", "rsi", "rdi", "r8",
		  "xmm0", "xmm1", "xmm2", "xmm3",
		  "xmm4", "xmm5", "xmm6", "xmm7",
		  "xmm8", "xmm9", "xmm10", "xmm11",
		  "xmm12", "xmm13", "xmm14", "xmm15",
		  "memory"
	);
}
