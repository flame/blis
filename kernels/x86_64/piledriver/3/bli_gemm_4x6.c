/*
   libflame
   An object-based infrastructure for developing high-performance
   dense linear algebra libraries.

   Copyright (C) 2011, The University of Texas

   libflame is free software; you can redistribute it and/or modify
   it under the terms of the GNU Lesser General Public License as
   published by the Free Software Foundation; either version 2.1 of
   the License, or (at your option) any later version.

   libflame is distributed in the hope that it will be useful, but
   WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
   Lesser General Public License for more details.

   You should have received a copy of the GNU Lesser General Public
   License along with libflame; if you did not receive a copy, see
   http://www.gnu.org/licenses/.

   For more information, please contact us at flame@cs.utexas.edu or
   send mail to:

   Field G. Van Zee and/or
   Robert A. van de Geijn
   The University of Texas at Austin
   Department of Computer Sciences
   1 University Station C0500
   Austin TX 78712
*/

#include "blis.h"



 void bli_sgemm_4x6(
                    dim_t     k,
                    float*    alpha,
                    float*    a,
                    float*    b,
                    float*    beta,
                    float*    c, inc_t rs_c, inc_t cs_c,
                    float* a_next, float* b_next,
                    dim_t thread_id
                  )
{
	bli_check_error_code( BLIS_NOT_YET_IMPLEMENTED );
}

 void bli_dgemm_4x6(
                    dim_t     k,
                    double*   alpha,
                    double*   a,
                    double*   b,
                    double*   beta,
                    double*   c, inc_t rs_c, inc_t cs_c,
                    double* a_next, double* b_next,
                    dim_t thread_id
                  )
{
    dim_t   k_iter;
	dim_t   k_left;
	k_iter  = k / 16;
	k_left  = k % 16;

	__asm__ 
	(	
		"                                \n\t"
		"                                \n\t"
		"movq          %3, %%rbx         \n\t" // load address of b.
		"vzeroall                        \n\t"
		"movq          %2, %%rax         \n\t" // load address of a.
		"movq      %0, %%rsi             \n\t" // i = k_iter; notice %0 not $0
		"testq  %%rsi, %%rsi			 \n\t"
		"prefetcht0 64(%%rax)								\n\t"
		"                                \n\t"
		"                                \n\t"
		"vmovapd 0 * 8(%%rbx), %%xmm1   					\n\t"
		"vmovapd 2 * 8(%%rbx), %%xmm2   					\n\t"
		"vmovapd 4 * 8(%%rbx), %%xmm3   					\n\t"
		"                                \n\t"
		".LOOPKITER:                     \n\t" // MAIN LOOP
		"je .CONSIDERKLEFT   								\n\t"
		"                       					        \n\t"
		"vmovddup 0 * 8(%%rax), %%xmm0   					\n\t" //iteration 0
		"vfmadd231pd  %%xmm1, %%xmm0, %%xmm4		    \n\t"
		"addq		$12*8, %%rbx		                    \n\t" 
		"vfmadd231pd  %%xmm2, %%xmm0, %%xmm5		    \n\t"
		"addq		$8*8, %%rax		                        \n\t" 
		"vfmadd231pd  %%xmm3, %%xmm0, %%xmm6		    \n\t"
		"vmovddup -7 * 8(%%rax), %%xmm0   					\n\t"
		"vfmadd231pd  %%xmm1, %%xmm0, %%xmm7		    \n\t"
		"prefetcht0 128(%%rax)								\n\t"
		"vfmadd231pd  %%xmm2, %%xmm0, %%xmm8		    \n\t"
		"vfmadd231pd  %%xmm3, %%xmm0, %%xmm9		    \n\t"
		"vmovddup -6 * 8(%%rax), %%xmm0   					\n\t"
		"vfmadd231pd  %%xmm1, %%xmm0, %%xmm10		    \n\t"
		"vfmadd231pd  %%xmm2, %%xmm0, %%xmm11		    \n\t"
		"vfmadd231pd  %%xmm3, %%xmm0, %%xmm12		    \n\t"
		"vmovddup -5 * 8(%%rax), %%xmm0   					\n\t"
		"vfmadd231pd  %%xmm1, %%xmm0, %%xmm13		    \n\t"
		"vmovapd -6 * 8(%%rbx), %%xmm1   					\n\t"
		"vfmadd231pd  %%xmm2, %%xmm0, %%xmm14		    \n\t"
		"vmovapd -4 * 8(%%rbx), %%xmm2   					\n\t"
		"vfmadd231pd  %%xmm3, %%xmm0, %%xmm15		    \n\t"
		"vmovapd -2 * 8(%%rbx), %%xmm3   					\n\t"
		"                       					        \n\t"
		"                       					        \n\t"
		"                       					        \n\t"
		"vmovddup -4 * 8(%%rax), %%xmm0   					\n\t" //iteration 1
		"vfmadd231pd  %%xmm1, %%xmm0, %%xmm4		    \n\t"
		"prefetcht0 192(%%rax)								\n\t"
		"vfmadd231pd  %%xmm2, %%xmm0, %%xmm5		    \n\t"
		"vfmadd231pd  %%xmm3, %%xmm0, %%xmm6		    \n\t"
		"vmovddup -3 * 8(%%rax), %%xmm0   					\n\t"
		"vfmadd231pd  %%xmm1, %%xmm0, %%xmm7		    \n\t"
		"vfmadd231pd  %%xmm2, %%xmm0, %%xmm8		    \n\t"
		"vfmadd231pd  %%xmm3, %%xmm0, %%xmm9		    \n\t"
		"vmovddup -2 * 8(%%rax), %%xmm0   					\n\t"
		"vfmadd231pd  %%xmm1, %%xmm0, %%xmm10		    \n\t"
		"vfmadd231pd  %%xmm2, %%xmm0, %%xmm11		    \n\t"
		"vfmadd231pd  %%xmm3, %%xmm0, %%xmm12		    \n\t"
		"vmovddup -1 * 8(%%rax), %%xmm0   					\n\t"
		"vfmadd231pd  %%xmm1, %%xmm0, %%xmm13		    \n\t"
		"vmovapd 0 * 8(%%rbx), %%xmm1   					\n\t"
		"vfmadd231pd  %%xmm2, %%xmm0, %%xmm14		    \n\t"
		"vmovapd 2 * 8(%%rbx), %%xmm2   					\n\t"
		"vfmadd231pd %%xmm3, %%xmm0, %%xmm15		    \n\t"
		"vmovapd 4 * 8(%%rbx), %%xmm3   					\n\t"
		"                       					        \n\t"
		"                       					        \n\t"
		"                       					        \n\t"
		"                       					        \n\t"
		"                       					        \n\t"
		"                       					        \n\t"
		"                       					        \n\t"
		"                       					        \n\t"
		"vmovddup 0 * 8(%%rax), %%xmm0   					\n\t" //iteration 0
		"vfmadd231pd  %%xmm1, %%xmm0, %%xmm4		    \n\t"
		"addq		$12*8, %%rbx		                    \n\t" 
		"vfmadd231pd  %%xmm2, %%xmm0, %%xmm5		    \n\t"
		"addq		$8*8, %%rax		                        \n\t" 
		"vfmadd231pd  %%xmm3, %%xmm0, %%xmm6		    \n\t"
		"vmovddup -7 * 8(%%rax), %%xmm0   					\n\t"
		"vfmadd231pd  %%xmm1, %%xmm0, %%xmm7		    \n\t"
		"prefetcht0 128(%%rax)								\n\t"
		"vfmadd231pd  %%xmm2, %%xmm0, %%xmm8		    \n\t"
		"vfmadd231pd  %%xmm3, %%xmm0, %%xmm9		    \n\t"
		"vmovddup -6 * 8(%%rax), %%xmm0   					\n\t"
		"vfmadd231pd  %%xmm1, %%xmm0, %%xmm10		    \n\t"
		"vfmadd231pd  %%xmm2, %%xmm0, %%xmm11		    \n\t"
		"vfmadd231pd  %%xmm3, %%xmm0, %%xmm12		    \n\t"
		"vmovddup -5 * 8(%%rax), %%xmm0   					\n\t"
		"vfmadd231pd  %%xmm1, %%xmm0, %%xmm13		    \n\t"
		"vmovapd -6 * 8(%%rbx), %%xmm1   					\n\t"
		"vfmadd231pd  %%xmm2, %%xmm0, %%xmm14		    \n\t"
		"vmovapd -4 * 8(%%rbx), %%xmm2   					\n\t"
		"vfmadd231pd  %%xmm3, %%xmm0, %%xmm15		    \n\t"
		"vmovapd -2 * 8(%%rbx), %%xmm3   					\n\t"
		"                       					        \n\t"
		"                       					        \n\t"
		"                       					        \n\t"
		"vmovddup -4 * 8(%%rax), %%xmm0   					\n\t" //iteration 1
		"vfmadd231pd  %%xmm1, %%xmm0, %%xmm4		    \n\t"
		"prefetcht0 192(%%rax)								\n\t"
		"vfmadd231pd  %%xmm2, %%xmm0, %%xmm5		    \n\t"
		"vfmadd231pd  %%xmm3, %%xmm0, %%xmm6		    \n\t"
		"vmovddup -3 * 8(%%rax), %%xmm0   					\n\t"
		"vfmadd231pd  %%xmm1, %%xmm0, %%xmm7		    \n\t"
		"vfmadd231pd  %%xmm2, %%xmm0, %%xmm8		    \n\t"
		"vfmadd231pd  %%xmm3, %%xmm0, %%xmm9		    \n\t"
		"vmovddup -2 * 8(%%rax), %%xmm0   					\n\t"
		"vfmadd231pd  %%xmm1, %%xmm0, %%xmm10		    \n\t"
		"vfmadd231pd  %%xmm2, %%xmm0, %%xmm11		    \n\t"
		"vfmadd231pd  %%xmm3, %%xmm0, %%xmm12		    \n\t"
		"vmovddup -1 * 8(%%rax), %%xmm0   					\n\t"
		"vfmadd231pd  %%xmm1, %%xmm0, %%xmm13		    \n\t"
		"vmovapd 0 * 8(%%rbx), %%xmm1   					\n\t"
		"vfmadd231pd  %%xmm2, %%xmm0, %%xmm14		    \n\t"
		"vmovapd 2 * 8(%%rbx), %%xmm2   					\n\t"
		"vfmadd231pd %%xmm3, %%xmm0, %%xmm15		    \n\t"
		"vmovapd 4 * 8(%%rbx), %%xmm3   					\n\t"
		"                       					        \n\t"
		"                       					        \n\t"
		"                       					        \n\t"
		"                       					        \n\t"
		"                       					        \n\t"
		"                       					        \n\t"
		"                       					        \n\t"
		"                       					        \n\t"
		"vmovddup 0 * 8(%%rax), %%xmm0   					\n\t" //iteration 0
		"vfmadd231pd  %%xmm1, %%xmm0, %%xmm4		    \n\t"
		"addq		$12*8, %%rbx		                    \n\t" 
		"vfmadd231pd  %%xmm2, %%xmm0, %%xmm5		    \n\t"
		"addq		$8*8, %%rax		                        \n\t" 
		"vfmadd231pd  %%xmm3, %%xmm0, %%xmm6		    \n\t"
		"vmovddup -7 * 8(%%rax), %%xmm0   					\n\t"
		"vfmadd231pd  %%xmm1, %%xmm0, %%xmm7		    \n\t"
		"prefetcht0 128(%%rax)								\n\t"
		"vfmadd231pd  %%xmm2, %%xmm0, %%xmm8		    \n\t"
		"vfmadd231pd  %%xmm3, %%xmm0, %%xmm9		    \n\t"
		"vmovddup -6 * 8(%%rax), %%xmm0   					\n\t"
		"vfmadd231pd  %%xmm1, %%xmm0, %%xmm10		    \n\t"
		"vfmadd231pd  %%xmm2, %%xmm0, %%xmm11		    \n\t"
		"vfmadd231pd  %%xmm3, %%xmm0, %%xmm12		    \n\t"
		"vmovddup -5 * 8(%%rax), %%xmm0   					\n\t"
		"vfmadd231pd  %%xmm1, %%xmm0, %%xmm13		    \n\t"
		"vmovapd -6 * 8(%%rbx), %%xmm1   					\n\t"
		"vfmadd231pd  %%xmm2, %%xmm0, %%xmm14		    \n\t"
		"vmovapd -4 * 8(%%rbx), %%xmm2   					\n\t"
		"vfmadd231pd  %%xmm3, %%xmm0, %%xmm15		    \n\t"
		"vmovapd -2 * 8(%%rbx), %%xmm3   					\n\t"
		"                       					        \n\t"
		"                       					        \n\t"
		"                       					        \n\t"
		"vmovddup -4 * 8(%%rax), %%xmm0   					\n\t" //iteration 1
		"vfmadd231pd  %%xmm1, %%xmm0, %%xmm4		    \n\t"
		"prefetcht0 192(%%rax)								\n\t"
		"vfmadd231pd  %%xmm2, %%xmm0, %%xmm5		    \n\t"
		"vfmadd231pd  %%xmm3, %%xmm0, %%xmm6		    \n\t"
		"vmovddup -3 * 8(%%rax), %%xmm0   					\n\t"
		"vfmadd231pd  %%xmm1, %%xmm0, %%xmm7		    \n\t"
		"vfmadd231pd  %%xmm2, %%xmm0, %%xmm8		    \n\t"
		"vfmadd231pd  %%xmm3, %%xmm0, %%xmm9		    \n\t"
		"vmovddup -2 * 8(%%rax), %%xmm0   					\n\t"
		"vfmadd231pd  %%xmm1, %%xmm0, %%xmm10		    \n\t"
		"vfmadd231pd  %%xmm2, %%xmm0, %%xmm11		    \n\t"
		"vfmadd231pd  %%xmm3, %%xmm0, %%xmm12		    \n\t"
		"vmovddup -1 * 8(%%rax), %%xmm0   					\n\t"
		"vfmadd231pd  %%xmm1, %%xmm0, %%xmm13		    \n\t"
		"vmovapd 0 * 8(%%rbx), %%xmm1   					\n\t"
		"vfmadd231pd  %%xmm2, %%xmm0, %%xmm14		    \n\t"
		"vmovapd 2 * 8(%%rbx), %%xmm2   					\n\t"
		"vfmadd231pd %%xmm3, %%xmm0, %%xmm15		    \n\t"
		"vmovapd 4 * 8(%%rbx), %%xmm3   					\n\t"
		"                       					        \n\t"
		"                       					        \n\t"
		"                       					        \n\t"
		"                       					        \n\t"
		"                       					        \n\t"
		"                       					        \n\t"
		"                       					        \n\t"
		"                       					        \n\t"
		"vmovddup 0 * 8(%%rax), %%xmm0   					\n\t" //iteration 0
		"vfmadd231pd  %%xmm1, %%xmm0, %%xmm4		    \n\t"
		"addq		$12*8, %%rbx		                    \n\t" 
		"vfmadd231pd  %%xmm2, %%xmm0, %%xmm5		    \n\t"
		"addq		$8*8, %%rax		                        \n\t" 
		"vfmadd231pd  %%xmm3, %%xmm0, %%xmm6		    \n\t"
		"vmovddup -7 * 8(%%rax), %%xmm0   					\n\t"
		"vfmadd231pd  %%xmm1, %%xmm0, %%xmm7		    \n\t"
		"prefetcht0 128(%%rax)								\n\t"
		"vfmadd231pd  %%xmm2, %%xmm0, %%xmm8		    \n\t"
		"vfmadd231pd  %%xmm3, %%xmm0, %%xmm9		    \n\t"
		"vmovddup -6 * 8(%%rax), %%xmm0   					\n\t"
		"vfmadd231pd  %%xmm1, %%xmm0, %%xmm10		    \n\t"
		"vfmadd231pd  %%xmm2, %%xmm0, %%xmm11		    \n\t"
		"vfmadd231pd  %%xmm3, %%xmm0, %%xmm12		    \n\t"
		"vmovddup -5 * 8(%%rax), %%xmm0   					\n\t"
		"vfmadd231pd  %%xmm1, %%xmm0, %%xmm13		    \n\t"
		"vmovapd -6 * 8(%%rbx), %%xmm1   					\n\t"
		"vfmadd231pd  %%xmm2, %%xmm0, %%xmm14		    \n\t"
		"vmovapd -4 * 8(%%rbx), %%xmm2   					\n\t"
		"vfmadd231pd  %%xmm3, %%xmm0, %%xmm15		    \n\t"
		"vmovapd -2 * 8(%%rbx), %%xmm3   					\n\t"
		"                       					        \n\t"
		"                       					        \n\t"
		"                       					        \n\t"
		"vmovddup -4 * 8(%%rax), %%xmm0   					\n\t" //iteration 1
		"vfmadd231pd  %%xmm1, %%xmm0, %%xmm4		    \n\t"
		"prefetcht0 192(%%rax)								\n\t"
		"vfmadd231pd  %%xmm2, %%xmm0, %%xmm5		    \n\t"
		"vfmadd231pd  %%xmm3, %%xmm0, %%xmm6		    \n\t"
		"vmovddup -3 * 8(%%rax), %%xmm0   					\n\t"
		"vfmadd231pd  %%xmm1, %%xmm0, %%xmm7		    \n\t"
		"vfmadd231pd  %%xmm2, %%xmm0, %%xmm8		    \n\t"
		"vfmadd231pd  %%xmm3, %%xmm0, %%xmm9		    \n\t"
		"vmovddup -2 * 8(%%rax), %%xmm0   					\n\t"
		"vfmadd231pd  %%xmm1, %%xmm0, %%xmm10		    \n\t"
		"vfmadd231pd  %%xmm2, %%xmm0, %%xmm11		    \n\t"
		"vfmadd231pd  %%xmm3, %%xmm0, %%xmm12		    \n\t"
		"vmovddup -1 * 8(%%rax), %%xmm0   					\n\t"
		"vfmadd231pd  %%xmm1, %%xmm0, %%xmm13		    \n\t"
		"vmovapd 0 * 8(%%rbx), %%xmm1   					\n\t"
		"vfmadd231pd  %%xmm2, %%xmm0, %%xmm14		    \n\t"
		"vmovapd 2 * 8(%%rbx), %%xmm2   					\n\t"
		"vfmadd231pd %%xmm3, %%xmm0, %%xmm15		    \n\t"
		"vmovapd 4 * 8(%%rbx), %%xmm3   					\n\t"
		"                       					        \n\t"
		"                       					        \n\t"
		"                       					        \n\t"
		"                       					        \n\t"
		"                       					        \n\t"
		"                       					        \n\t"
		"                       					        \n\t"
		"                       					        \n\t"
		"vmovddup 0 * 8(%%rax), %%xmm0   					\n\t" //iteration 0
		"vfmadd231pd  %%xmm1, %%xmm0, %%xmm4		    \n\t"
		"addq		$12*8, %%rbx		                    \n\t" 
		"vfmadd231pd  %%xmm2, %%xmm0, %%xmm5		    \n\t"
		"addq		$8*8, %%rax		                        \n\t" 
		"vfmadd231pd  %%xmm3, %%xmm0, %%xmm6		    \n\t"
		"vmovddup -7 * 8(%%rax), %%xmm0   					\n\t"
		"vfmadd231pd  %%xmm1, %%xmm0, %%xmm7		    \n\t"
		"prefetcht0 128(%%rax)								\n\t"
		"vfmadd231pd  %%xmm2, %%xmm0, %%xmm8		    \n\t"
		"vfmadd231pd  %%xmm3, %%xmm0, %%xmm9		    \n\t"
		"vmovddup -6 * 8(%%rax), %%xmm0   					\n\t"
		"vfmadd231pd  %%xmm1, %%xmm0, %%xmm10		    \n\t"
		"vfmadd231pd  %%xmm2, %%xmm0, %%xmm11		    \n\t"
		"vfmadd231pd  %%xmm3, %%xmm0, %%xmm12		    \n\t"
		"vmovddup -5 * 8(%%rax), %%xmm0   					\n\t"
		"vfmadd231pd  %%xmm1, %%xmm0, %%xmm13		    \n\t"
		"vmovapd -6 * 8(%%rbx), %%xmm1   					\n\t"
		"vfmadd231pd  %%xmm2, %%xmm0, %%xmm14		    \n\t"
		"vmovapd -4 * 8(%%rbx), %%xmm2   					\n\t"
		"vfmadd231pd  %%xmm3, %%xmm0, %%xmm15		    \n\t"
		"vmovapd -2 * 8(%%rbx), %%xmm3   					\n\t"
		"                       					        \n\t"
		"                       					        \n\t"
		"                       					        \n\t"
		"vmovddup -4 * 8(%%rax), %%xmm0   					\n\t" //iteration 1
		"vfmadd231pd  %%xmm1, %%xmm0, %%xmm4		    \n\t"
		"prefetcht0 192(%%rax)								\n\t"
		"vfmadd231pd  %%xmm2, %%xmm0, %%xmm5		    \n\t"
		"vfmadd231pd  %%xmm3, %%xmm0, %%xmm6		    \n\t"
		"vmovddup -3 * 8(%%rax), %%xmm0   					\n\t"
		"vfmadd231pd  %%xmm1, %%xmm0, %%xmm7		    \n\t"
		"vfmadd231pd  %%xmm2, %%xmm0, %%xmm8		    \n\t"
		"vfmadd231pd  %%xmm3, %%xmm0, %%xmm9		    \n\t"
		"vmovddup -2 * 8(%%rax), %%xmm0   					\n\t"
		"vfmadd231pd  %%xmm1, %%xmm0, %%xmm10		    \n\t"
		"vfmadd231pd  %%xmm2, %%xmm0, %%xmm11		    \n\t"
		"vfmadd231pd  %%xmm3, %%xmm0, %%xmm12		    \n\t"
		"vmovddup -1 * 8(%%rax), %%xmm0   					\n\t"
		"vfmadd231pd  %%xmm1, %%xmm0, %%xmm13		    \n\t"
		"vmovapd 0 * 8(%%rbx), %%xmm1   					\n\t"
		"vfmadd231pd  %%xmm2, %%xmm0, %%xmm14		    \n\t"
		"vmovapd 2 * 8(%%rbx), %%xmm2   					\n\t"
		"vfmadd231pd %%xmm3, %%xmm0, %%xmm15		    \n\t"
		"vmovapd 4 * 8(%%rbx), %%xmm3   					\n\t"
		"                       					        \n\t"
		"                       					        \n\t"
		"                       					        \n\t"
		"                       					        \n\t"
		"                       					        \n\t"
		"                       					        \n\t"
		"                       					        \n\t"
		"                       					        \n\t"
		"vmovddup 0 * 8(%%rax), %%xmm0   					\n\t" //iteration 0
		"vfmadd231pd  %%xmm1, %%xmm0, %%xmm4		    \n\t"
		"addq		$12*8, %%rbx		                    \n\t" 
		"vfmadd231pd  %%xmm2, %%xmm0, %%xmm5		    \n\t"
		"addq		$8*8, %%rax		                        \n\t" 
		"vfmadd231pd  %%xmm3, %%xmm0, %%xmm6		    \n\t"
		"vmovddup -7 * 8(%%rax), %%xmm0   					\n\t"
		"vfmadd231pd  %%xmm1, %%xmm0, %%xmm7		    \n\t"
		"prefetcht0 128(%%rax)								\n\t"
		"vfmadd231pd  %%xmm2, %%xmm0, %%xmm8		    \n\t"
		"vfmadd231pd  %%xmm3, %%xmm0, %%xmm9		    \n\t"
		"vmovddup -6 * 8(%%rax), %%xmm0   					\n\t"
		"vfmadd231pd  %%xmm1, %%xmm0, %%xmm10		    \n\t"
		"vfmadd231pd  %%xmm2, %%xmm0, %%xmm11		    \n\t"
		"vfmadd231pd  %%xmm3, %%xmm0, %%xmm12		    \n\t"
		"vmovddup -5 * 8(%%rax), %%xmm0   					\n\t"
		"vfmadd231pd  %%xmm1, %%xmm0, %%xmm13		    \n\t"
		"vmovapd -6 * 8(%%rbx), %%xmm1   					\n\t"
		"vfmadd231pd  %%xmm2, %%xmm0, %%xmm14		    \n\t"
		"vmovapd -4 * 8(%%rbx), %%xmm2   					\n\t"
		"vfmadd231pd  %%xmm3, %%xmm0, %%xmm15		    \n\t"
		"vmovapd -2 * 8(%%rbx), %%xmm3   					\n\t"
		"                       					        \n\t"
		"                       					        \n\t"
		"                       					        \n\t"
		"vmovddup -4 * 8(%%rax), %%xmm0   					\n\t" //iteration 1
		"vfmadd231pd  %%xmm1, %%xmm0, %%xmm4		    \n\t"
		"prefetcht0 192(%%rax)								\n\t"
		"vfmadd231pd  %%xmm2, %%xmm0, %%xmm5		    \n\t"
		"vfmadd231pd  %%xmm3, %%xmm0, %%xmm6		    \n\t"
		"vmovddup -3 * 8(%%rax), %%xmm0   					\n\t"
		"vfmadd231pd  %%xmm1, %%xmm0, %%xmm7		    \n\t"
		"vfmadd231pd  %%xmm2, %%xmm0, %%xmm8		    \n\t"
		"vfmadd231pd  %%xmm3, %%xmm0, %%xmm9		    \n\t"
		"vmovddup -2 * 8(%%rax), %%xmm0   					\n\t"
		"vfmadd231pd  %%xmm1, %%xmm0, %%xmm10		    \n\t"
		"vfmadd231pd  %%xmm2, %%xmm0, %%xmm11		    \n\t"
		"vfmadd231pd  %%xmm3, %%xmm0, %%xmm12		    \n\t"
		"vmovddup -1 * 8(%%rax), %%xmm0   					\n\t"
		"vfmadd231pd  %%xmm1, %%xmm0, %%xmm13		    \n\t"
		"vmovapd 0 * 8(%%rbx), %%xmm1   					\n\t"
		"vfmadd231pd  %%xmm2, %%xmm0, %%xmm14		    \n\t"
		"vmovapd 2 * 8(%%rbx), %%xmm2   					\n\t"
		"vfmadd231pd %%xmm3, %%xmm0, %%xmm15		    \n\t"
		"vmovapd 4 * 8(%%rbx), %%xmm3   					\n\t"
		"                       					        \n\t"
		"                       					        \n\t"
		"                       					        \n\t"
		"                       					        \n\t"
		"                       					        \n\t"
		"                       					        \n\t"
		"                       					        \n\t"
		"                       					        \n\t"
		"vmovddup 0 * 8(%%rax), %%xmm0   					\n\t" //iteration 0
		"vfmadd231pd  %%xmm1, %%xmm0, %%xmm4		    \n\t"
		"addq		$12*8, %%rbx		                    \n\t" 
		"vfmadd231pd  %%xmm2, %%xmm0, %%xmm5		    \n\t"
		"addq		$8*8, %%rax		                        \n\t" 
		"vfmadd231pd  %%xmm3, %%xmm0, %%xmm6		    \n\t"
		"vmovddup -7 * 8(%%rax), %%xmm0   					\n\t"
		"vfmadd231pd  %%xmm1, %%xmm0, %%xmm7		    \n\t"
		"prefetcht0 128(%%rax)								\n\t"
		"vfmadd231pd  %%xmm2, %%xmm0, %%xmm8		    \n\t"
		"vfmadd231pd  %%xmm3, %%xmm0, %%xmm9		    \n\t"
		"vmovddup -6 * 8(%%rax), %%xmm0   					\n\t"
		"vfmadd231pd  %%xmm1, %%xmm0, %%xmm10		    \n\t"
		"vfmadd231pd  %%xmm2, %%xmm0, %%xmm11		    \n\t"
		"vfmadd231pd  %%xmm3, %%xmm0, %%xmm12		    \n\t"
		"vmovddup -5 * 8(%%rax), %%xmm0   					\n\t"
		"vfmadd231pd  %%xmm1, %%xmm0, %%xmm13		    \n\t"
		"vmovapd -6 * 8(%%rbx), %%xmm1   					\n\t"
		"vfmadd231pd  %%xmm2, %%xmm0, %%xmm14		    \n\t"
		"vmovapd -4 * 8(%%rbx), %%xmm2   					\n\t"
		"vfmadd231pd  %%xmm3, %%xmm0, %%xmm15		    \n\t"
		"vmovapd -2 * 8(%%rbx), %%xmm3   					\n\t"
		"                       					        \n\t"
		"                       					        \n\t"
		"                       					        \n\t"
		"vmovddup -4 * 8(%%rax), %%xmm0   					\n\t" //iteration 1
		"vfmadd231pd  %%xmm1, %%xmm0, %%xmm4		    \n\t"
		"prefetcht0 192(%%rax)								\n\t"
		"vfmadd231pd  %%xmm2, %%xmm0, %%xmm5		    \n\t"
		"vfmadd231pd  %%xmm3, %%xmm0, %%xmm6		    \n\t"
		"vmovddup -3 * 8(%%rax), %%xmm0   					\n\t"
		"vfmadd231pd  %%xmm1, %%xmm0, %%xmm7		    \n\t"
		"vfmadd231pd  %%xmm2, %%xmm0, %%xmm8		    \n\t"
		"vfmadd231pd  %%xmm3, %%xmm0, %%xmm9		    \n\t"
		"vmovddup -2 * 8(%%rax), %%xmm0   					\n\t"
		"vfmadd231pd  %%xmm1, %%xmm0, %%xmm10		    \n\t"
		"vfmadd231pd  %%xmm2, %%xmm0, %%xmm11		    \n\t"
		"vfmadd231pd  %%xmm3, %%xmm0, %%xmm12		    \n\t"
		"vmovddup -1 * 8(%%rax), %%xmm0   					\n\t"
		"vfmadd231pd  %%xmm1, %%xmm0, %%xmm13		    \n\t"
		"vmovapd 0 * 8(%%rbx), %%xmm1   					\n\t"
		"vfmadd231pd  %%xmm2, %%xmm0, %%xmm14		    \n\t"
		"vmovapd 2 * 8(%%rbx), %%xmm2   					\n\t"
		"vfmadd231pd %%xmm3, %%xmm0, %%xmm15		    \n\t"
		"vmovapd 4 * 8(%%rbx), %%xmm3   					\n\t"
		"                       					        \n\t"
		"                       					        \n\t"
		"                       					        \n\t"
		"                       					        \n\t"
		"                       					        \n\t"
		"                       					        \n\t"
		"                       					        \n\t"
		"                       					        \n\t"
		"vmovddup 0 * 8(%%rax), %%xmm0   					\n\t" //iteration 0
		"vfmadd231pd  %%xmm1, %%xmm0, %%xmm4		    \n\t"
		"addq		$12*8, %%rbx		                    \n\t" 
		"vfmadd231pd  %%xmm2, %%xmm0, %%xmm5		    \n\t"
		"addq		$8*8, %%rax		                        \n\t" 
		"vfmadd231pd  %%xmm3, %%xmm0, %%xmm6		    \n\t"
        "decq %%rsi \n\t"
		"vmovddup -7 * 8(%%rax), %%xmm0   					\n\t"
		"vfmadd231pd  %%xmm1, %%xmm0, %%xmm7		    \n\t"
		"prefetcht0 128(%%rax)								\n\t"
		"vfmadd231pd  %%xmm2, %%xmm0, %%xmm8		    \n\t"
		"vfmadd231pd  %%xmm3, %%xmm0, %%xmm9		    \n\t"
		"vmovddup -6 * 8(%%rax), %%xmm0   					\n\t"
		"vfmadd231pd  %%xmm1, %%xmm0, %%xmm10		    \n\t"
		"vfmadd231pd  %%xmm2, %%xmm0, %%xmm11		    \n\t"
		"vfmadd231pd  %%xmm3, %%xmm0, %%xmm12		    \n\t"
		"vmovddup -5 * 8(%%rax), %%xmm0   					\n\t"
		"vfmadd231pd  %%xmm1, %%xmm0, %%xmm13		    \n\t"
		"vmovapd -6 * 8(%%rbx), %%xmm1   					\n\t"
		"vfmadd231pd  %%xmm2, %%xmm0, %%xmm14		    \n\t"
		"vmovapd -4 * 8(%%rbx), %%xmm2   					\n\t"
		"vfmadd231pd  %%xmm3, %%xmm0, %%xmm15		    \n\t"
		"vmovapd -2 * 8(%%rbx), %%xmm3   					\n\t"
		"                       					        \n\t"
		"                       					        \n\t"
		"                       					        \n\t"
		"vmovddup -4 * 8(%%rax), %%xmm0   					\n\t" //iteration 1
		"vfmadd231pd  %%xmm1, %%xmm0, %%xmm4		    \n\t"
		"prefetcht0 192(%%rax)								\n\t"
		"vfmadd231pd  %%xmm2, %%xmm0, %%xmm5		    \n\t"
		"vfmadd231pd  %%xmm3, %%xmm0, %%xmm6		    \n\t"
		"vmovddup -3 * 8(%%rax), %%xmm0   					\n\t"
		"vfmadd231pd  %%xmm1, %%xmm0, %%xmm7		    \n\t"
		"vfmadd231pd  %%xmm2, %%xmm0, %%xmm8		    \n\t"
		"vfmadd231pd  %%xmm3, %%xmm0, %%xmm9		    \n\t"
		"vmovddup -2 * 8(%%rax), %%xmm0   					\n\t"
		"vfmadd231pd  %%xmm1, %%xmm0, %%xmm10		    \n\t"
		"vfmadd231pd  %%xmm2, %%xmm0, %%xmm11		    \n\t"
		"vfmadd231pd  %%xmm3, %%xmm0, %%xmm12		    \n\t"
		"vmovddup -1 * 8(%%rax), %%xmm0   					\n\t"
		"vfmadd231pd  %%xmm1, %%xmm0, %%xmm13		    \n\t"
		"vmovapd 0 * 8(%%rbx), %%xmm1   					\n\t"
		"vfmadd231pd  %%xmm2, %%xmm0, %%xmm14		    \n\t"
		"vmovapd 2 * 8(%%rbx), %%xmm2   					\n\t"
		"vfmadd231pd %%xmm3, %%xmm0, %%xmm15		    \n\t"
		"vmovapd 4 * 8(%%rbx), %%xmm3   					\n\t"
		"                       					        \n\t"
		"                       					        \n\t"
		"                       					        \n\t"
		"                       					        \n\t"
		"                       					        \n\t"
		"                       					        \n\t"
		"                       					        \n\t"
		"                       					        \n\t"
		"jmp .LOOPKITER										\n\t"
		"                       					        \n\t"
		".CONSIDERKLEFT:           					        \n\t"
		"                       					        \n\t"
		"movq %1, %%rsi            					        \n\t"
		"testq %%rsi, %%rsi       					        \n\t" 
		".LOOPKLEFT:               					        \n\t"
		"je .POSTACCUM            					        \n\t"
		"                       					        \n\t"
		"vmovddup 0 * 8(%%rax), %%xmm0   					\n\t" //iteration 0
		"vfmadd231pd  %%xmm1, %%xmm0, %%xmm4		    \n\t"
		"addq		$6*8, %%rbx		                    \n\t" 
		"vfmadd231pd  %%xmm2, %%xmm0, %%xmm5		    \n\t"
		"addq		$4*8, %%rax		                        \n\t" 
		"vfmadd231pd  %%xmm3, %%xmm0, %%xmm6		    \n\t"
        "decq %%rsi \n\t"
		"vmovddup -3 * 8(%%rax), %%xmm0   					\n\t"
		"vfmadd231pd  %%xmm1, %%xmm0, %%xmm7		    \n\t"
		"prefetcht0 128(%%rax)								\n\t"
		"vfmadd231pd  %%xmm2, %%xmm0, %%xmm8		    \n\t"
		"vfmadd231pd  %%xmm3, %%xmm0, %%xmm9		    \n\t"
		"vmovddup -2 * 8(%%rax), %%xmm0   					\n\t"
		"vfmadd231pd  %%xmm1, %%xmm0, %%xmm10		    \n\t"
		"vfmadd231pd  %%xmm2, %%xmm0, %%xmm11		    \n\t"
		"vfmadd231pd  %%xmm3, %%xmm0, %%xmm12		    \n\t"
		"vmovddup -1 * 8(%%rax), %%xmm0   					\n\t"
		"vfmadd231pd  %%xmm1, %%xmm0, %%xmm13		    \n\t"
		"vmovapd 0 * 8(%%rbx), %%xmm1   					\n\t"
		"vfmadd231pd  %%xmm2, %%xmm0, %%xmm14		    \n\t"
		"vmovapd 2 * 8(%%rbx), %%xmm2   					\n\t"
		"vfmadd231pd  %%xmm3, %%xmm0, %%xmm15		    \n\t"
		"vmovapd 4 * 8(%%rbx), %%xmm3   					\n\t"
		"                       					        \n\t"
		"jmp .LOOPKLEFT               						\n\t" // iterate again if i != 0.
		"                       					        \n\t"
		".POSTACCUM:                    \n\t"
		"                               \n\t"
		"                               \n\t"
		"movq    %7, %%rsi              \n\t" // load cs_c
		"vmovddup (%4), %%xmm2		\n\t" //load alpha
		"movq    %8, %%rdi              \n\t" // load rs_c
		"vmovddup (%5), %%xmm3		\n\t" //load beta
		"movq    %6, %%rcx          		\n\t" // load address of c
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
		"vfmadd231pd %%xmm0, %%xmm3, %%xmm4\n\t"	// scale by beta, and add the gemm result
		"vmovlpd  (%%r8),       %%xmm0, %%xmm0   	\n\t" 			
		"vfmadd231pd %%xmm1, %%xmm3, %%xmm5\n\t"	// scale by beta, and add the gemm result
		"vmovhpd  (%%r8,%%rdi), %%xmm0, %%xmm0   	\n\t"
		"vmovlpd  %%xmm4,  (%%rcx)        	\n\t" 			// and store back to memory.
		"vmovlpd  %%xmm5,  (%%rdx)        	\n\t" 			// and store back to memory.
		"vmovhpd  %%xmm4,  (%%rcx,%%rdi)  	\n\t"
		"addq %%rsi, %%rcx				   	\n\t" 
		"vmovhpd  %%xmm5,  (%%rdx,%%rdi)  	\n\t"
		"addq %%rsi, %%rdx				   	\n\t" 
		"                                	\n\t"
		"vmulpd   %%xmm2,  %%xmm6, %%xmm6         	\n\t"			// scale by alpha,
		"vfmadd231pd   %%xmm0, %%xmm3, %%xmm6\n\t"	// scale by beta, and add the gemm result
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
		"vfmadd231pd   %%xmm0, %%xmm3, %%xmm7\n\t"	// scale by beta, and add the gemm result
		"vfmadd231pd   %%xmm1, %%xmm3, %%xmm8\n\t"	// scale by beta, and add the gemm result
		"vfmadd231pd   %%xmm4, %%xmm3, %%xmm9\n\t"	// scale by beta, and add the gemm result
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
		"vfmadd231pd   %%xmm0, %%xmm3, %%xmm10\n\t"	// scale by beta, and add the gemm result
		"vfmadd231pd   %%xmm1, %%xmm3, %%xmm11\n\t"	// scale by beta, and add the gemm result
		"vfmadd231pd   %%xmm4, %%xmm3, %%xmm12\n\t"	// scale by beta, and add the gemm result
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
		"vfmadd231pd   %%xmm0, %%xmm3, %%xmm13\n\t"	// scale by beta, and add the gemm result
		"vfmadd231pd   %%xmm1, %%xmm3, %%xmm14\n\t"	// scale by beta, and add the gemm result
		"vfmadd231pd   %%xmm4, %%xmm3, %%xmm15\n\t"	// scale by beta, and add the gemm result
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

 void bli_cgemm_4x6(
                    dim_t     k,
                    scomplex* alpha,
                    scomplex* a,
                    scomplex* b,
                    scomplex* beta,
                    scomplex* c, inc_t rs_c, inc_t cs_c,
                    scomplex* a_next, scomplex* b_next,
                    dim_t thread_id
                  )
{
	bli_check_error_code( BLIS_NOT_YET_IMPLEMENTED );
}

 void bli_zgemm_4x6(
                    dim_t     k,
                    dcomplex* alpha,
                    dcomplex* a,
                    dcomplex* b,
                    dcomplex* beta,
                    dcomplex* c, inc_t rs_c, inc_t cs_c,
                    dcomplex* a_next, dcomplex* b_next,
                    dim_t thread_id
                  )
{
	bli_check_error_code( BLIS_NOT_YET_IMPLEMENTED );
}

