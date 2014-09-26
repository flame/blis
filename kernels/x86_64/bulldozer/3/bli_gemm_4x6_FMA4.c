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
