/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2014, The University of Texas at Austin
   Copyright (C) 2019, Advanced Micro Devices, Inc.

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

#define BLIS_ASM_SYNTAX_ATT
#include "bli_x86_asm_macros.h"

/*
   rrc:
     --------        ------        | | | | | | | |
     --------        ------        | | | | | | | |
     --------   +=   ------ ...    | | | | | | | |
     --------        ------        | | | | | | | |
     --------        ------              :
     --------        ------              :

   Assumptions:
   - C is row-stored and B is column-stored;
   - A is row-stored;
   - m0 and n0 are at most MR and NR, respectively.
   Therefore, this (r)ow-preferential microkernel is well-suited for
   a dot-product-based accumulation that performs vector loads from
   both A and B.

   NOTE: These kernels implicitly support column-oriented IO, implemented
   via an a high-level transposition of the entire operation. A and B will
   effectively remain row- and column-stored, respectively, but C will then
   effectively appear column-stored. Thus, this kernel may be used for both
   rrc and crc cases.
*/

// Prototype reference microkernels.
GEMMSUP_KER_PROT( double,   d, gemmsup_r_haswell_ref )


void bli_dgemmsup_rd_haswell_asm_6x8n
     (
       conj_t              conja,
       conj_t              conjb,
       dim_t               m0,
       dim_t               n0,
       dim_t               k0,
       double*    restrict alpha,
       double*    restrict a, inc_t rs_a0, inc_t cs_a0,
       double*    restrict b, inc_t rs_b0, inc_t cs_b0,
       double*    restrict beta,
       double*    restrict c, inc_t rs_c0, inc_t cs_c0,
       auxinfo_t* restrict data,
       cntx_t*    restrict cntx
     )
{
	uint64_t m_left = m0 % 6;

	// First check whether this is a edge case in the n dimension. If so,
	// dispatch other ?x8m kernels, as needed.
	if ( m_left )
	{
		double* restrict cij = c;
		double* restrict bj  = b;
		double* restrict ai  = a;

#if 1
		// We add special handling for slightly inflated MR blocksizes
		// at edge cases, up to a maximum of 9.
		if ( 6 < m0 )
		{
			dgemmsup_ker_ft ker_fp1 = NULL;
			dgemmsup_ker_ft ker_fp2 = NULL;
			dim_t           mr1, mr2;

			// These kernels don't make any attempt to optimize the cases of
			// inflated MR blocksizes because they don't benefit from the
			// load balancing that the "rv" kernels do. That is, if m0 = 7,
			// there is no benefit to executing that case as 4x8n followed
			// by 3x8n because 4x8n isn't implemented, and more generally
			// because these kernels are implemented as loops over their
			// true blocksizes, which are MR=3 NR=4.
			if ( m0 == 7 )
			{
				mr1 = 6; mr2 = 1;
				ker_fp1 = bli_dgemmsup_rd_haswell_asm_6x8n;
				ker_fp2 = bli_dgemmsup_rd_haswell_asm_1x8n;
			}
			else if ( m0 == 8 )
			{
				mr1 = 6; mr2 = 2;
				ker_fp1 = bli_dgemmsup_rd_haswell_asm_6x8n;
				ker_fp2 = bli_dgemmsup_rd_haswell_asm_2x8n;
			}
			else // if ( m0 == 9 )
			{
				mr1 = 6; mr2 = 3;
				ker_fp1 = bli_dgemmsup_rd_haswell_asm_6x8n;
				ker_fp2 = bli_dgemmsup_rd_haswell_asm_3x8n;
			}

			ker_fp1
			(
			  conja, conjb, mr1, n0, k0,
			  alpha, ai, rs_a0, cs_a0, bj, rs_b0, cs_b0,
			  beta, cij, rs_c0, cs_c0, data, cntx
			);
			cij += mr1*rs_c0; ai += mr1*rs_a0;

			ker_fp2
			(
			  conja, conjb, mr2, n0, k0,
			  alpha, ai, rs_a0, cs_a0, bj, rs_b0, cs_b0,
			  beta, cij, rs_c0, cs_c0, data, cntx
			);

			return;
		}
#endif

		if ( 3 <= m_left )
		{
			const dim_t mr_cur = 3;

			bli_dgemmsup_rd_haswell_asm_3x8n
			//bli_dgemmsup_r_haswell_ref
			(
			  conja, conjb, mr_cur, n0, k0,
			  alpha, ai, rs_a0, cs_a0, bj, rs_b0, cs_b0,
			  beta, cij, rs_c0, cs_c0, data, cntx
			);
			cij += mr_cur*rs_c0; ai += mr_cur*rs_a0; m_left -= mr_cur;
		}
		if ( 2 <= m_left )
		{
			const dim_t mr_cur = 2;

			bli_dgemmsup_rd_haswell_asm_2x8n
			//bli_dgemmsup_r_haswell_ref
			(
			  conja, conjb, mr_cur, n0, k0,
			  alpha, ai, rs_a0, cs_a0, bj, rs_b0, cs_b0,
			  beta, cij, rs_c0, cs_c0, data, cntx
			);
			cij += mr_cur*rs_c0; ai += mr_cur*rs_a0; m_left -= mr_cur;
		}
		if ( 1 == m_left )
		{
			#if 1
			const dim_t mr_cur = 1;

			bli_dgemmsup_rd_haswell_asm_1x8n
			//bli_dgemmsup_r_haswell_ref
			(
			  conja, conjb, mr_cur, n0, k0,
			  alpha, ai, rs_a0, cs_a0, bj, rs_b0, cs_b0,
			  beta, cij, rs_c0, cs_c0, data, cntx
			);
			#else
			bli_dgemv_ex
			(
			  BLIS_TRANSPOSE, conja, k0, n0,
			  alpha, bj, rs_b0, cs_b0, ai, cs_a0,
			  beta, cij, cs_c0, cntx, NULL
			);
			#endif
		}
		return;
	}

	//void*    a_next = bli_auxinfo_next_a( data );
	//void*    b_next = bli_auxinfo_next_b( data );

	// Typecast local copies of integers in case dim_t and inc_t are a
	// different size than is expected by load instructions.
	uint64_t k_iter16 = k0 / 16;
	uint64_t k_left16 = k0 % 16;
	uint64_t k_iter4  = k_left16 / 4;
	uint64_t k_left1  = k_left16 % 4;

	uint64_t n_iter = n0 / 4;
	uint64_t n_left = n0 % 4;

	uint64_t rs_a   = rs_a0;
	uint64_t cs_a   = cs_a0;
	uint64_t rs_b   = rs_b0;
	uint64_t cs_b   = cs_b0;
	uint64_t rs_c   = rs_c0;
	uint64_t cs_c   = cs_c0;

	if ( n_iter == 0 ) goto consider_edge_cases;

	// -------------------------------------------------------------------------

	begin_asm()

	//vzeroall()                         // zero all xmm/ymm registers.
	
	//mov(var(a), rdx)                   // load address of a.
	mov(var(rs_a), r8)                 // load rs_a
	//mov(var(cs_a), r9)                 // load cs_a
	lea(mem(, r8, 8), r8)              // rs_a *= sizeof(double)
	//lea(mem(, r9, 8), r9)              // cs_a *= sizeof(double)

	//lea(mem(r8, r8, 2), r13)           // r13 = 3*rs_a
	//lea(mem(r8, r8, 4), r15)           // r15 = 5*rs_a

	//mov(var(b), r14)                   // load address of b.
	//mov(var(rs_b), r10)                // load rs_b
	mov(var(cs_b), r11)                // load cs_b
	//lea(mem(, r10, 8), r10)            // rs_b *= sizeof(double)
	lea(mem(, r11, 8), r11)            // cs_b *= sizeof(double)

	lea(mem(r11, r11, 2), r13)         // r13 = 3*cs_b
	//lea(mem(r8,  r8,  2), r10)         // r10 = 3*rs_a
	

	//mov(var(c), r12)                   // load address of c
	mov(var(rs_c), rdi)                // load rs_c
	lea(mem(, rdi, 8), rdi)            // rs_c *= sizeof(double)



	// r12 = rcx = c
	// rdx = rax = a
	// r14 = rbx = b
	// r9  = m dim index ii
	// r15 = n dim index jj

	mov(imm(0), r9)                    // ii = 0;

	label(.DLOOP3X4I)                  // LOOP OVER ii = [ 0 1 ... ]



	mov(var(a), rdx)                   // load address of a
	mov(var(b), r14)                   // load address of b
	mov(var(c), r12)                   // load address of c

	lea(mem(   , r9,  1), rsi)         // rsi = r9 = 3*ii;
	imul(rdi, rsi)                     // rsi *= rs_c
	lea(mem(r12, rsi, 1), r12)         // r12 = c + 3*ii*rs_c;

	lea(mem(   , r9,  1), rsi)         // rsi = r9 = 3*ii;
	imul(r8,  rsi)                     // rsi *= rs_a;
	lea(mem(rdx, rsi, 1), rdx)         // rax = a + 3*ii*rs_a;



	mov(var(n_iter), r15)              // jj = n_iter;

	label(.DLOOP3X4J)                  // LOOP OVER jj = [ n_iter ... 1 0 ]



#if 0
	vzeroall()                         // zero all xmm/ymm registers.
#else
	                                   // skylake can execute 3 vxorpd ipc with
	                                   // a latency of 1 cycle, while vzeroall
	                                   // has a latency of 12 cycles.
	vxorpd(ymm4,  ymm4,  ymm4)
	vxorpd(ymm5,  ymm5,  ymm5)
	vxorpd(ymm6,  ymm6,  ymm6)
	vxorpd(ymm7,  ymm7,  ymm7)
	vxorpd(ymm8,  ymm8,  ymm8)
	vxorpd(ymm9,  ymm9,  ymm9)
	vxorpd(ymm10, ymm10, ymm10)
	vxorpd(ymm11, ymm11, ymm11)
	vxorpd(ymm12, ymm12, ymm12)
	vxorpd(ymm13, ymm13, ymm13)
	vxorpd(ymm14, ymm14, ymm14)
	vxorpd(ymm15, ymm15, ymm15)
#endif


	lea(mem(r12), rcx)                 // rcx = c_iijj;
	lea(mem(rdx), rax)                 // rax = a_ii;
	lea(mem(r14), rbx)                 // rbx = b_jj;


#if 1
	//mov(var(rs_c), rdi)                // load rs_c
	//lea(mem(, rdi, 8), rdi)            // rs_c *= sizeof(double)
	prefetch(0, mem(rcx,         3*8)) // prefetch c + 0*rs_c
	prefetch(0, mem(rcx, rdi, 1, 3*8)) // prefetch c + 1*rs_c
	prefetch(0, mem(rcx, rdi, 2, 3*8)) // prefetch c + 2*rs_c
#endif
	//lea(mem(r8,  r8,  4), rbp)         // rbp = 5*rs_a
	lea(mem(rbx, r11, 4), r10)         // r10 = rbx + 4*cs_b

	

	
	mov(var(k_iter16), rsi)            // i = k_iter16;
	test(rsi, rsi)                     // check i via logical AND.
	je(.DCONSIDKITER4)                 // if i == 0, jump to code that
	                                   // contains the k_iter4 loop.
	
	
	label(.DLOOPKITER16)               // MAIN LOOP
	
	
	// ---------------------------------- iteration 0

#if 1
	prefetch(0, mem(r10,         0*8)) // prefetch rbx + 4*cs_b
	prefetch(0, mem(r10, r11, 1, 0*8)) // prefetch rbx + 5*cs_b
#endif

	vmovupd(mem(rax       ), ymm0)
	vmovupd(mem(rax, r8, 1), ymm1)
	vmovupd(mem(rax, r8, 2), ymm2)
	add(imm(4*8), rax)                 // a += 4*cs_a = 4*8;

	vmovupd(mem(rbx        ), ymm3)
	vfmadd231pd(ymm0, ymm3, ymm4)
	vfmadd231pd(ymm1, ymm3, ymm5)
	vfmadd231pd(ymm2, ymm3, ymm6)

	vmovupd(mem(rbx, r11, 1), ymm3)
	vfmadd231pd(ymm0, ymm3, ymm7)
	vfmadd231pd(ymm1, ymm3, ymm8)
	vfmadd231pd(ymm2, ymm3, ymm9)

	vmovupd(mem(rbx, r11, 2), ymm3)
	vfmadd231pd(ymm0, ymm3, ymm10)
	vfmadd231pd(ymm1, ymm3, ymm11)
	vfmadd231pd(ymm2, ymm3, ymm12)

	vmovupd(mem(rbx, r13, 1), ymm3)
	add(imm(4*8), rbx)                 // b += 4*rs_b = 4*8;
	vfmadd231pd(ymm0, ymm3, ymm13)
	vfmadd231pd(ymm1, ymm3, ymm14)
	vfmadd231pd(ymm2, ymm3, ymm15)

	
	// ---------------------------------- iteration 1

#if 1
	prefetch(0, mem(r10, r11, 2, 0*8)) // prefetch rbx + 6*cs_b
	prefetch(0, mem(r10, r13, 1, 0*8)) // prefetch rbx + 7*cs_b
#endif

	vmovupd(mem(rax       ), ymm0)
	vmovupd(mem(rax, r8, 1), ymm1)
	vmovupd(mem(rax, r8, 2), ymm2)
	add(imm(4*8), rax)                 // a += 4*cs_a = 4*8;

	vmovupd(mem(rbx        ), ymm3)
	vfmadd231pd(ymm0, ymm3, ymm4)
	vfmadd231pd(ymm1, ymm3, ymm5)
	vfmadd231pd(ymm2, ymm3, ymm6)

	vmovupd(mem(rbx, r11, 1), ymm3)
	vfmadd231pd(ymm0, ymm3, ymm7)
	vfmadd231pd(ymm1, ymm3, ymm8)
	vfmadd231pd(ymm2, ymm3, ymm9)

	vmovupd(mem(rbx, r11, 2), ymm3)
	vfmadd231pd(ymm0, ymm3, ymm10)
	vfmadd231pd(ymm1, ymm3, ymm11)
	vfmadd231pd(ymm2, ymm3, ymm12)

	vmovupd(mem(rbx, r13, 1), ymm3)
	add(imm(4*8), rbx)                 // b += 4*rs_b = 4*8;
	vfmadd231pd(ymm0, ymm3, ymm13)
	vfmadd231pd(ymm1, ymm3, ymm14)
	vfmadd231pd(ymm2, ymm3, ymm15)


	// ---------------------------------- iteration 2
	
#if 1
	prefetch(0, mem(r10,         8*8)) // prefetch rbx + 4*cs_b + 8*rs_b
	prefetch(0, mem(r10, r11, 1, 8*8)) // prefetch rbx + 5*cs_b + 8*rs_b
#endif

	vmovupd(mem(rax       ), ymm0)
	vmovupd(mem(rax, r8, 1), ymm1)
	vmovupd(mem(rax, r8, 2), ymm2)
	add(imm(4*8), rax)                 // a += 4*cs_a = 4*8;

	vmovupd(mem(rbx        ), ymm3)
	vfmadd231pd(ymm0, ymm3, ymm4)
	vfmadd231pd(ymm1, ymm3, ymm5)
	vfmadd231pd(ymm2, ymm3, ymm6)

	vmovupd(mem(rbx, r11, 1), ymm3)
	vfmadd231pd(ymm0, ymm3, ymm7)
	vfmadd231pd(ymm1, ymm3, ymm8)
	vfmadd231pd(ymm2, ymm3, ymm9)

	vmovupd(mem(rbx, r11, 2), ymm3)
	vfmadd231pd(ymm0, ymm3, ymm10)
	vfmadd231pd(ymm1, ymm3, ymm11)
	vfmadd231pd(ymm2, ymm3, ymm12)

	vmovupd(mem(rbx, r13, 1), ymm3)
	add(imm(4*8), rbx)                 // b += 4*rs_b = 4*8;
	vfmadd231pd(ymm0, ymm3, ymm13)
	vfmadd231pd(ymm1, ymm3, ymm14)
	vfmadd231pd(ymm2, ymm3, ymm15)


	// ---------------------------------- iteration 3

#if 1
	prefetch(0, mem(r10, r11, 2, 8*8)) // prefetch rbx + 6*cs_b + 8*rs_b
	prefetch(0, mem(r10, r13, 1, 8*8)) // prefetch rbx + 7*cs_b + 8*rs_b
	add(imm(16*8), r10)                 // r10 += 8*rs_b = 8*8;
#endif

	vmovupd(mem(rax       ), ymm0)
	vmovupd(mem(rax, r8, 1), ymm1)
	vmovupd(mem(rax, r8, 2), ymm2)
	add(imm(4*8), rax)                 // a += 4*cs_a = 4*8;

	vmovupd(mem(rbx        ), ymm3)
	vfmadd231pd(ymm0, ymm3, ymm4)
	vfmadd231pd(ymm1, ymm3, ymm5)
	vfmadd231pd(ymm2, ymm3, ymm6)

	vmovupd(mem(rbx, r11, 1), ymm3)
	vfmadd231pd(ymm0, ymm3, ymm7)
	vfmadd231pd(ymm1, ymm3, ymm8)
	vfmadd231pd(ymm2, ymm3, ymm9)

	vmovupd(mem(rbx, r11, 2), ymm3)
	vfmadd231pd(ymm0, ymm3, ymm10)
	vfmadd231pd(ymm1, ymm3, ymm11)
	vfmadd231pd(ymm2, ymm3, ymm12)

	vmovupd(mem(rbx, r13, 1), ymm3)
	add(imm(4*8), rbx)                 // b += 4*rs_b = 4*8;
	vfmadd231pd(ymm0, ymm3, ymm13)
	vfmadd231pd(ymm1, ymm3, ymm14)
	vfmadd231pd(ymm2, ymm3, ymm15)

	

	dec(rsi)                           // i -= 1;
	jne(.DLOOPKITER16)                 // iterate again if i != 0.
	
	
	
	
	
	
	label(.DCONSIDKITER4)
	
	mov(var(k_iter4), rsi)             // i = k_iter4;
	test(rsi, rsi)                     // check i via logical AND.
	je(.DCONSIDKLEFT1)                 // if i == 0, jump to code that
	                                   // considers k_left1 loop.
	                                   // else, we prepare to enter k_iter4 loop.
	
	
	label(.DLOOPKITER4)                // EDGE LOOP (ymm)
	
	vmovupd(mem(rax       ), ymm0)
	vmovupd(mem(rax, r8, 1), ymm1)
	vmovupd(mem(rax, r8, 2), ymm2)
	add(imm(4*8), rax)                 // a += 4*cs_a = 4*8;
	
	vmovupd(mem(rbx        ), ymm3)
	vfmadd231pd(ymm0, ymm3, ymm4)
	vfmadd231pd(ymm1, ymm3, ymm5)
	vfmadd231pd(ymm2, ymm3, ymm6)

	vmovupd(mem(rbx, r11, 1), ymm3)
	vfmadd231pd(ymm0, ymm3, ymm7)
	vfmadd231pd(ymm1, ymm3, ymm8)
	vfmadd231pd(ymm2, ymm3, ymm9)

	vmovupd(mem(rbx, r11, 2), ymm3)
	vfmadd231pd(ymm0, ymm3, ymm10)
	vfmadd231pd(ymm1, ymm3, ymm11)
	vfmadd231pd(ymm2, ymm3, ymm12)

	vmovupd(mem(rbx, r13, 1), ymm3)
	add(imm(4*8), rbx)                 // b += 4*rs_b = 4*8;
	vfmadd231pd(ymm0, ymm3, ymm13)
	vfmadd231pd(ymm1, ymm3, ymm14)
	vfmadd231pd(ymm2, ymm3, ymm15)

	
	dec(rsi)                           // i -= 1;
	jne(.DLOOPKITER4)                  // iterate again if i != 0.
	
	
	

	label(.DCONSIDKLEFT1)
	
	mov(var(k_left1), rsi)             // i = k_left1;
	test(rsi, rsi)                     // check i via logical AND.
	je(.DPOSTACCUM)                    // if i == 0, we're done; jump to end.
	                                   // else, we prepare to enter k_left1 loop.
	
	


	label(.DLOOPKLEFT1)                // EDGE LOOP (scalar)
	                                   // NOTE: We must use ymm registers here bc
	                                   // using the xmm registers would zero out the
	                                   // high bits of the destination registers,
	                                   // which would destory intermediate results.
	
	vmovsd(mem(rax       ), xmm0)
	vmovsd(mem(rax, r8, 1), xmm1)
	vmovsd(mem(rax, r8, 2), xmm2)
	add(imm(1*8), rax)                 // a += 1*cs_a = 1*8;
	
	vmovsd(mem(rbx        ), xmm3)
	vfmadd231pd(ymm0, ymm3, ymm4)
	vfmadd231pd(ymm1, ymm3, ymm5)
	vfmadd231pd(ymm2, ymm3, ymm6)

	vmovsd(mem(rbx, r11, 1), xmm3)
	vfmadd231pd(ymm0, ymm3, ymm7)
	vfmadd231pd(ymm1, ymm3, ymm8)
	vfmadd231pd(ymm2, ymm3, ymm9)

	vmovsd(mem(rbx, r11, 2), xmm3)
	vfmadd231pd(ymm0, ymm3, ymm10)
	vfmadd231pd(ymm1, ymm3, ymm11)
	vfmadd231pd(ymm2, ymm3, ymm12)

	vmovsd(mem(rbx, r13, 1), xmm3)
	add(imm(1*8), rbx)                 // b += 1*rs_b = 1*8;
	vfmadd231pd(ymm0, ymm3, ymm13)
	vfmadd231pd(ymm1, ymm3, ymm14)
	vfmadd231pd(ymm2, ymm3, ymm15)

	
	dec(rsi)                           // i -= 1;
	jne(.DLOOPKLEFT1)                  // iterate again if i != 0.
	
	
	




	label(.DPOSTACCUM)
	
	                                   // ymm4  ymm7  ymm10 ymm13  
	                                   // ymm5  ymm8  ymm11 ymm14
	                                   // ymm6  ymm9  ymm12 ymm15
	
	vhaddpd( ymm7, ymm4, ymm0 )
	vextractf128(imm(1), ymm0, xmm1 )
	vaddpd( xmm0, xmm1, xmm0 )

	vhaddpd( ymm13, ymm10, ymm2 )
	vextractf128(imm(1), ymm2, xmm1 )
	vaddpd( xmm2, xmm1, xmm2 )

	vperm2f128(imm(0x20), ymm2, ymm0, ymm4 )
	                                   // ymm4[0] = sum(ymm4);  ymm4[1] = sum(ymm7)
	                                   // ymm4[2] = sum(ymm10); ymm4[3] = sum(ymm13)

	vhaddpd( ymm8, ymm5, ymm0 )
	vextractf128(imm(1), ymm0, xmm1 )
	vaddpd( xmm0, xmm1, xmm0 )

	vhaddpd( ymm14, ymm11, ymm2 )
	vextractf128(imm(1), ymm2, xmm1 )
	vaddpd( xmm2, xmm1, xmm2 )

	vperm2f128(imm(0x20), ymm2, ymm0, ymm5 )
	                                   // ymm5[0] = sum(ymm5);  ymm5[1] = sum(ymm8)
	                                   // ymm5[2] = sum(ymm11); ymm5[3] = sum(ymm14)

	vhaddpd( ymm9, ymm6, ymm0 )
	vextractf128(imm(1), ymm0, xmm1 )
	vaddpd( xmm0, xmm1, xmm0 )

	vhaddpd( ymm15, ymm12, ymm2 )
	vextractf128(imm(1), ymm2, xmm1 )
	vaddpd( xmm2, xmm1, xmm2 )

	vperm2f128(imm(0x20), ymm2, ymm0, ymm6 )
	                                   // ymm6[0] = sum(ymm6);  ymm6[1] = sum(ymm9)
	                                   // ymm6[2] = sum(ymm12); ymm6[3] = sum(ymm15)



	
	//mov(var(rs_c), rdi)                // load rs_c
	//lea(mem(, rdi, 8), rdi)            // rs_c *= sizeof(double)

	mov(var(alpha), rax)               // load address of alpha
	mov(var(beta), rbx)                // load address of beta
	vbroadcastsd(mem(rax), ymm0)       // load alpha and duplicate
	vbroadcastsd(mem(rbx), ymm3)       // load beta and duplicate
	
	vmulpd(ymm0, ymm4, ymm4)           // scale by alpha
	vmulpd(ymm0, ymm5, ymm5)
	vmulpd(ymm0, ymm6, ymm6)
	
	
	
	
	
	
	//mov(var(cs_c), rsi)                // load cs_c
	//lea(mem(, rsi, 8), rsi)            // rsi = cs_c * sizeof(double)
	
	
	
	                                   // now avoid loading C if beta == 0
	
	vxorpd(ymm0, ymm0, ymm0)           // set ymm0 to zero.
	vucomisd(xmm0, xmm3)               // set ZF if beta == 0.
	je(.DBETAZERO)                     // if ZF = 1, jump to beta == 0 case
	

	
	label(.DROWSTORED)
	
	
	vfmadd231pd(mem(rcx), ymm3, ymm4)
	vmovupd(ymm4, mem(rcx))
	add(rdi, rcx)
	
	vfmadd231pd(mem(rcx), ymm3, ymm5)
	vmovupd(ymm5, mem(rcx))
	add(rdi, rcx)
	
	vfmadd231pd(mem(rcx), ymm3, ymm6)
	vmovupd(ymm6, mem(rcx))
	//add(rdi, rcx)
	
	
	
	jmp(.DDONE)                        // jump to end.
	
	
	
	
	label(.DBETAZERO)
	

	
	label(.DROWSTORBZ)
	
	
	vmovupd(ymm4, mem(rcx))
	add(rdi, rcx)
	
	vmovupd(ymm5, mem(rcx))
	add(rdi, rcx)
	
	vmovupd(ymm6, mem(rcx))
	//add(rdi, rcx)
	
	
	
	
	label(.DDONE)
	
	


	add(imm(4*8), r12)                 // c_jj = r12 += 4*cs_c

	lea(mem(r14, r11, 4), r14)         // b_jj = r14 += 4*cs_b

	dec(r15)                           // jj -= 1;
	jne(.DLOOP3X4J)                    // iterate again if jj != 0.



	add(imm(3), r9)                    // ii += 3;
	cmp(imm(6), r9)                    // compare ii to 6
	jl(.DLOOP3X4I)                     // if ii < 6, jump to beginning
	                                   // of ii loop; otherwise, loop ends.




	label(.DRETURN)

	

    end_asm(
	: // output operands (none)
	: // input operands
      [n_iter] "m" (n_iter),
      [k_iter16] "m" (k_iter16),
      [k_iter4] "m" (k_iter4),
      [k_left1] "m" (k_left1),
      [a]      "m" (a),
      [rs_a]   "m" (rs_a),
      [cs_a]   "m" (cs_a),
      [b]      "m" (b),
      [rs_b]   "m" (rs_b),
      [cs_b]   "m" (cs_b),
      [alpha]  "m" (alpha),
      [beta]   "m" (beta),
      [c]      "m" (c),
      [rs_c]   "m" (rs_c),
      [cs_c]   "m" (cs_c)/*,
      [a_next] "m" (a_next),
      [b_next] "m" (b_next)*/
	: // register clobber list
	  "rax", "rbx", "rcx", "rdx", "rsi", "rdi", "rbp",
	  "r8", "r9", "r10", "r11", "r12", "r13", "r14", "r15",
	  "xmm0", "xmm1", "xmm2", "xmm3",
	  "xmm4", "xmm5", "xmm6", "xmm7",
	  "xmm8", "xmm9", "xmm10", "xmm11",
	  "xmm12", "xmm13", "xmm14", "xmm15",
	  "memory"
	)

	consider_edge_cases:

	// Handle edge cases in the n dimension, if they exist.
	if ( n_left )
	{
		const dim_t      mr_cur = 6;
		const dim_t      j_edge = n0 - ( dim_t )n_left;

		double* restrict cij = c + j_edge*cs_c;
		double* restrict ai  = a;
		double* restrict bj  = b + j_edge*cs_b;

		if ( 2 <= n_left )
		{
			const dim_t nr_cur = 2;

			bli_dgemmsup_rd_haswell_asm_6x2
			//bli_dgemmsup_r_haswell_ref
			(
			  conja, conjb, mr_cur, nr_cur, k0,
			  alpha, ai, rs_a0, cs_a0, bj, rs_b0, cs_b0,
			  beta, cij, rs_c0, cs_c0, data, cntx
			);
			cij += nr_cur*cs_c0; bj += nr_cur*cs_b0; n_left -= nr_cur;
		}
		if ( 1 == n_left )
		{
			#if 1
			const dim_t nr_cur = 1;

			bli_dgemmsup_rd_haswell_asm_6x1
			//bli_dgemmsup_r_haswell_ref
			(
			  conja, conjb, mr_cur, nr_cur, k0,
			  alpha, ai, rs_a0, cs_a0, bj, rs_b0, cs_b0,
			  beta, cij, rs_c0, cs_c0, data, cntx
			);
			#else
			bli_dgemv_ex
			(
			  BLIS_NO_TRANSPOSE, conjb, mr_cur, k0,
			  alpha, ai, rs_a0, cs_a0, bj, rs_b0,
			  beta, cij, rs_c0, cntx, NULL
			);
			#endif
		}
	}
}

void bli_dgemmsup_rd_haswell_asm_3x8n
     (
       conj_t              conja,
       conj_t              conjb,
       dim_t               m0,
       dim_t               n0,
       dim_t               k0,
       double*    restrict alpha,
       double*    restrict a, inc_t rs_a0, inc_t cs_a0,
       double*    restrict b, inc_t rs_b0, inc_t cs_b0,
       double*    restrict beta,
       double*    restrict c, inc_t rs_c0, inc_t cs_c0,
       auxinfo_t* restrict data,
       cntx_t*    restrict cntx
     )
{
	//void*    a_next = bli_auxinfo_next_a( data );
	//void*    b_next = bli_auxinfo_next_b( data );

	// Typecast local copies of integers in case dim_t and inc_t are a
	// different size than is expected by load instructions.
	uint64_t k_iter16 = k0 / 16;
	uint64_t k_left16 = k0 % 16;
	uint64_t k_iter4  = k_left16 / 4;
	uint64_t k_left1  = k_left16 % 4;

	uint64_t n_iter = n0 / 4;
	uint64_t n_left = n0 % 4;

	uint64_t rs_a   = rs_a0;
	uint64_t cs_a   = cs_a0;
	uint64_t rs_b   = rs_b0;
	uint64_t cs_b   = cs_b0;
	uint64_t rs_c   = rs_c0;
	uint64_t cs_c   = cs_c0;

	if ( n_iter == 0 ) goto consider_edge_cases;

	// -------------------------------------------------------------------------

	begin_asm()

	//vzeroall()                         // zero all xmm/ymm registers.
	
	mov(var(a), rdx)                   // load address of a.
	mov(var(rs_a), r8)                 // load rs_a
	//mov(var(cs_a), r9)                 // load cs_a
	lea(mem(, r8, 8), r8)              // rs_a *= sizeof(double)
	//lea(mem(, r9, 8), r9)              // cs_a *= sizeof(double)

	//lea(mem(r8, r8, 2), r13)           // r13 = 3*rs_a
	//lea(mem(r8, r8, 4), r15)           // r15 = 5*rs_a

	mov(var(b), r14)                   // load address of b.
	//mov(var(rs_b), r10)                // load rs_b
	mov(var(cs_b), r11)                // load cs_b
	//lea(mem(, r10, 8), r10)            // rs_b *= sizeof(double)
	lea(mem(, r11, 8), r11)            // cs_b *= sizeof(double)

	lea(mem(r11, r11, 2), r13)         // r13 = 3*cs_b
	//lea(mem(r8,  r8,  2), r10)         // r10 = 3*rs_a
	

	mov(var(c), r12)                   // load address of c
	mov(var(rs_c), rdi)                // load rs_c
	lea(mem(, rdi, 8), rdi)            // rs_c *= sizeof(double)



	// r12 = rcx = c
	// rdx = rax = a
	// r14 = rbx = b
	// r9  = unused
	// r15 = n dim index jj

	mov(var(n_iter), r15)              // jj = n_iter;

	label(.DLOOP3X4J)                  // LOOP OVER jj = [ n_iter ... 1 0 ]



#if 0
	vzeroall()                         // zero all xmm/ymm registers.
#else
	                                   // skylake can execute 3 vxorpd ipc with
	                                   // a latency of 1 cycle, while vzeroall
	                                   // has a latency of 12 cycles.
	vxorpd(ymm4,  ymm4,  ymm4)
	vxorpd(ymm5,  ymm5,  ymm5)
	vxorpd(ymm6,  ymm6,  ymm6)
	vxorpd(ymm7,  ymm7,  ymm7)
	vxorpd(ymm8,  ymm8,  ymm8)
	vxorpd(ymm9,  ymm9,  ymm9)
	vxorpd(ymm10, ymm10, ymm10)
	vxorpd(ymm11, ymm11, ymm11)
	vxorpd(ymm12, ymm12, ymm12)
	vxorpd(ymm13, ymm13, ymm13)
	vxorpd(ymm14, ymm14, ymm14)
	vxorpd(ymm15, ymm15, ymm15)
#endif


	lea(mem(r12), rcx)                 // rcx = c_iijj;
	lea(mem(rdx), rax)                 // rax = a_ii;
	lea(mem(r14), rbx)                 // rbx = b_jj;


#if 1
	//mov(var(rs_c), rdi)                // load rs_c
	//lea(mem(, rdi, 8), rdi)            // rs_c *= sizeof(double)
	prefetch(0, mem(rcx,         3*8)) // prefetch c + 0*rs_c
	prefetch(0, mem(rcx, rdi, 1, 3*8)) // prefetch c + 1*rs_c
	prefetch(0, mem(rcx, rdi, 2, 3*8)) // prefetch c + 2*rs_c
#endif
	//lea(mem(r8,  r8,  4), rbp)         // rbp = 5*rs_a
	lea(mem(rbx, r11, 4), r10)         // r10 = rbx + 4*cs_b

	

	
	mov(var(k_iter16), rsi)            // i = k_iter16;
	test(rsi, rsi)                     // check i via logical AND.
	je(.DCONSIDKITER4)                 // if i == 0, jump to code that
	                                   // contains the k_iter4 loop.
	
	
	label(.DLOOPKITER16)               // MAIN LOOP
	
	
	// ---------------------------------- iteration 0

#if 1
	prefetch(0, mem(r10,         0*8)) // prefetch rbx + 4*cs_b
	prefetch(0, mem(r10, r11, 1, 0*8)) // prefetch rbx + 5*cs_b
#endif

	vmovupd(mem(rax       ), ymm0)
	vmovupd(mem(rax, r8, 1), ymm1)
	vmovupd(mem(rax, r8, 2), ymm2)
	add(imm(4*8), rax)                 // a += 4*cs_a = 4*8;

	vmovupd(mem(rbx        ), ymm3)
	vfmadd231pd(ymm0, ymm3, ymm4)
	vfmadd231pd(ymm1, ymm3, ymm5)
	vfmadd231pd(ymm2, ymm3, ymm6)

	vmovupd(mem(rbx, r11, 1), ymm3)
	vfmadd231pd(ymm0, ymm3, ymm7)
	vfmadd231pd(ymm1, ymm3, ymm8)
	vfmadd231pd(ymm2, ymm3, ymm9)

	vmovupd(mem(rbx, r11, 2), ymm3)
	vfmadd231pd(ymm0, ymm3, ymm10)
	vfmadd231pd(ymm1, ymm3, ymm11)
	vfmadd231pd(ymm2, ymm3, ymm12)

	vmovupd(mem(rbx, r13, 1), ymm3)
	add(imm(4*8), rbx)                 // b += 4*rs_b = 4*8;
	vfmadd231pd(ymm0, ymm3, ymm13)
	vfmadd231pd(ymm1, ymm3, ymm14)
	vfmadd231pd(ymm2, ymm3, ymm15)

	
	// ---------------------------------- iteration 1

#if 1
	prefetch(0, mem(r10, r11, 2, 0*8)) // prefetch rbx + 6*cs_b
	prefetch(0, mem(r10, r13, 1, 0*8)) // prefetch rbx + 7*cs_b
#endif

	vmovupd(mem(rax       ), ymm0)
	vmovupd(mem(rax, r8, 1), ymm1)
	vmovupd(mem(rax, r8, 2), ymm2)
	add(imm(4*8), rax)                 // a += 4*cs_a = 4*8;

	vmovupd(mem(rbx        ), ymm3)
	vfmadd231pd(ymm0, ymm3, ymm4)
	vfmadd231pd(ymm1, ymm3, ymm5)
	vfmadd231pd(ymm2, ymm3, ymm6)

	vmovupd(mem(rbx, r11, 1), ymm3)
	vfmadd231pd(ymm0, ymm3, ymm7)
	vfmadd231pd(ymm1, ymm3, ymm8)
	vfmadd231pd(ymm2, ymm3, ymm9)

	vmovupd(mem(rbx, r11, 2), ymm3)
	vfmadd231pd(ymm0, ymm3, ymm10)
	vfmadd231pd(ymm1, ymm3, ymm11)
	vfmadd231pd(ymm2, ymm3, ymm12)

	vmovupd(mem(rbx, r13, 1), ymm3)
	add(imm(4*8), rbx)                 // b += 4*rs_b = 4*8;
	vfmadd231pd(ymm0, ymm3, ymm13)
	vfmadd231pd(ymm1, ymm3, ymm14)
	vfmadd231pd(ymm2, ymm3, ymm15)


	// ---------------------------------- iteration 2
	
#if 1
	prefetch(0, mem(r10,         8*8)) // prefetch rbx + 4*cs_b + 8*rs_b
	prefetch(0, mem(r10, r11, 1, 8*8)) // prefetch rbx + 5*cs_b + 8*rs_b
#endif

	vmovupd(mem(rax       ), ymm0)
	vmovupd(mem(rax, r8, 1), ymm1)
	vmovupd(mem(rax, r8, 2), ymm2)
	add(imm(4*8), rax)                 // a += 4*cs_a = 4*8;

	vmovupd(mem(rbx        ), ymm3)
	vfmadd231pd(ymm0, ymm3, ymm4)
	vfmadd231pd(ymm1, ymm3, ymm5)
	vfmadd231pd(ymm2, ymm3, ymm6)

	vmovupd(mem(rbx, r11, 1), ymm3)
	vfmadd231pd(ymm0, ymm3, ymm7)
	vfmadd231pd(ymm1, ymm3, ymm8)
	vfmadd231pd(ymm2, ymm3, ymm9)

	vmovupd(mem(rbx, r11, 2), ymm3)
	vfmadd231pd(ymm0, ymm3, ymm10)
	vfmadd231pd(ymm1, ymm3, ymm11)
	vfmadd231pd(ymm2, ymm3, ymm12)

	vmovupd(mem(rbx, r13, 1), ymm3)
	add(imm(4*8), rbx)                 // b += 4*rs_b = 4*8;
	vfmadd231pd(ymm0, ymm3, ymm13)
	vfmadd231pd(ymm1, ymm3, ymm14)
	vfmadd231pd(ymm2, ymm3, ymm15)


	// ---------------------------------- iteration 3

#if 1
	prefetch(0, mem(r10, r11, 2, 8*8)) // prefetch rbx + 6*cs_b + 8*rs_b
	prefetch(0, mem(r10, r13, 1, 8*8)) // prefetch rbx + 7*cs_b + 8*rs_b
	add(imm(16*8), r10)                 // r10 += 8*rs_b = 8*8;
#endif

	vmovupd(mem(rax       ), ymm0)
	vmovupd(mem(rax, r8, 1), ymm1)
	vmovupd(mem(rax, r8, 2), ymm2)
	add(imm(4*8), rax)                 // a += 4*cs_a = 4*8;

	vmovupd(mem(rbx        ), ymm3)
	vfmadd231pd(ymm0, ymm3, ymm4)
	vfmadd231pd(ymm1, ymm3, ymm5)
	vfmadd231pd(ymm2, ymm3, ymm6)

	vmovupd(mem(rbx, r11, 1), ymm3)
	vfmadd231pd(ymm0, ymm3, ymm7)
	vfmadd231pd(ymm1, ymm3, ymm8)
	vfmadd231pd(ymm2, ymm3, ymm9)

	vmovupd(mem(rbx, r11, 2), ymm3)
	vfmadd231pd(ymm0, ymm3, ymm10)
	vfmadd231pd(ymm1, ymm3, ymm11)
	vfmadd231pd(ymm2, ymm3, ymm12)

	vmovupd(mem(rbx, r13, 1), ymm3)
	add(imm(4*8), rbx)                 // b += 4*rs_b = 4*8;
	vfmadd231pd(ymm0, ymm3, ymm13)
	vfmadd231pd(ymm1, ymm3, ymm14)
	vfmadd231pd(ymm2, ymm3, ymm15)

	

	dec(rsi)                           // i -= 1;
	jne(.DLOOPKITER16)                 // iterate again if i != 0.
	
	
	
	
	
	
	label(.DCONSIDKITER4)
	
	mov(var(k_iter4), rsi)             // i = k_iter4;
	test(rsi, rsi)                     // check i via logical AND.
	je(.DCONSIDKLEFT1)                 // if i == 0, jump to code that
	                                   // considers k_left1 loop.
	                                   // else, we prepare to enter k_iter4 loop.
	
	
	label(.DLOOPKITER4)                // EDGE LOOP (ymm)
	
	vmovupd(mem(rax       ), ymm0)
	vmovupd(mem(rax, r8, 1), ymm1)
	vmovupd(mem(rax, r8, 2), ymm2)
	add(imm(4*8), rax)                 // a += 4*cs_a = 4*8;
	
	vmovupd(mem(rbx        ), ymm3)
	vfmadd231pd(ymm0, ymm3, ymm4)
	vfmadd231pd(ymm1, ymm3, ymm5)
	vfmadd231pd(ymm2, ymm3, ymm6)

	vmovupd(mem(rbx, r11, 1), ymm3)
	vfmadd231pd(ymm0, ymm3, ymm7)
	vfmadd231pd(ymm1, ymm3, ymm8)
	vfmadd231pd(ymm2, ymm3, ymm9)

	vmovupd(mem(rbx, r11, 2), ymm3)
	vfmadd231pd(ymm0, ymm3, ymm10)
	vfmadd231pd(ymm1, ymm3, ymm11)
	vfmadd231pd(ymm2, ymm3, ymm12)

	vmovupd(mem(rbx, r13, 1), ymm3)
	add(imm(4*8), rbx)                 // b += 4*rs_b = 4*8;
	vfmadd231pd(ymm0, ymm3, ymm13)
	vfmadd231pd(ymm1, ymm3, ymm14)
	vfmadd231pd(ymm2, ymm3, ymm15)

	
	dec(rsi)                           // i -= 1;
	jne(.DLOOPKITER4)                  // iterate again if i != 0.
	
	
	

	label(.DCONSIDKLEFT1)
	
	mov(var(k_left1), rsi)             // i = k_left1;
	test(rsi, rsi)                     // check i via logical AND.
	je(.DPOSTACCUM)                    // if i == 0, we're done; jump to end.
	                                   // else, we prepare to enter k_left1 loop.
	
	


	label(.DLOOPKLEFT1)                // EDGE LOOP (scalar)
	                                   // NOTE: We must use ymm registers here bc
	                                   // using the xmm registers would zero out the
	                                   // high bits of the destination registers,
	                                   // which would destory intermediate results.
	
	vmovsd(mem(rax       ), xmm0)
	vmovsd(mem(rax, r8, 1), xmm1)
	vmovsd(mem(rax, r8, 2), xmm2)
	add(imm(1*8), rax)                 // a += 1*cs_a = 1*8;
	
	vmovsd(mem(rbx        ), xmm3)
	vfmadd231pd(ymm0, ymm3, ymm4)
	vfmadd231pd(ymm1, ymm3, ymm5)
	vfmadd231pd(ymm2, ymm3, ymm6)

	vmovsd(mem(rbx, r11, 1), xmm3)
	vfmadd231pd(ymm0, ymm3, ymm7)
	vfmadd231pd(ymm1, ymm3, ymm8)
	vfmadd231pd(ymm2, ymm3, ymm9)

	vmovsd(mem(rbx, r11, 2), xmm3)
	vfmadd231pd(ymm0, ymm3, ymm10)
	vfmadd231pd(ymm1, ymm3, ymm11)
	vfmadd231pd(ymm2, ymm3, ymm12)

	vmovsd(mem(rbx, r13, 1), xmm3)
	add(imm(1*8), rbx)                 // b += 1*rs_b = 1*8;
	vfmadd231pd(ymm0, ymm3, ymm13)
	vfmadd231pd(ymm1, ymm3, ymm14)
	vfmadd231pd(ymm2, ymm3, ymm15)

	
	dec(rsi)                           // i -= 1;
	jne(.DLOOPKLEFT1)                  // iterate again if i != 0.
	
	
	




	label(.DPOSTACCUM)
	
	                                   // ymm4  ymm7  ymm10 ymm13  
	                                   // ymm5  ymm8  ymm11 ymm14
	                                   // ymm6  ymm9  ymm12 ymm15
	
	vhaddpd( ymm7, ymm4, ymm0 )
	vextractf128(imm(1), ymm0, xmm1 )
	vaddpd( xmm0, xmm1, xmm0 )

	vhaddpd( ymm13, ymm10, ymm2 )
	vextractf128(imm(1), ymm2, xmm1 )
	vaddpd( xmm2, xmm1, xmm2 )

	vperm2f128(imm(0x20), ymm2, ymm0, ymm4 )
	                                   // ymm4[0] = sum(ymm4);  ymm4[1] = sum(ymm7)
	                                   // ymm4[2] = sum(ymm10); ymm4[3] = sum(ymm13)

	vhaddpd( ymm8, ymm5, ymm0 )
	vextractf128(imm(1), ymm0, xmm1 )
	vaddpd( xmm0, xmm1, xmm0 )

	vhaddpd( ymm14, ymm11, ymm2 )
	vextractf128(imm(1), ymm2, xmm1 )
	vaddpd( xmm2, xmm1, xmm2 )

	vperm2f128(imm(0x20), ymm2, ymm0, ymm5 )
	                                   // ymm5[0] = sum(ymm5);  ymm5[1] = sum(ymm8)
	                                   // ymm5[2] = sum(ymm11); ymm5[3] = sum(ymm14)

	vhaddpd( ymm9, ymm6, ymm0 )
	vextractf128(imm(1), ymm0, xmm1 )
	vaddpd( xmm0, xmm1, xmm0 )

	vhaddpd( ymm15, ymm12, ymm2 )
	vextractf128(imm(1), ymm2, xmm1 )
	vaddpd( xmm2, xmm1, xmm2 )

	vperm2f128(imm(0x20), ymm2, ymm0, ymm6 )
	                                   // ymm6[0] = sum(ymm6);  ymm6[1] = sum(ymm9)
	                                   // ymm6[2] = sum(ymm12); ymm6[3] = sum(ymm15)


	

	//mov(var(rs_c), rdi)                // load rs_c
	//lea(mem(, rdi, 8), rdi)            // rs_c *= sizeof(double)

	mov(var(alpha), rax)               // load address of alpha
	mov(var(beta), rbx)                // load address of beta
	vbroadcastsd(mem(rax), ymm0)       // load alpha and duplicate
	vbroadcastsd(mem(rbx), ymm3)       // load beta and duplicate
	
	vmulpd(ymm0, ymm4, ymm4)           // scale by alpha
	vmulpd(ymm0, ymm5, ymm5)
	vmulpd(ymm0, ymm6, ymm6)
	
	
	
	
	
	
	//mov(var(cs_c), rsi)                // load cs_c
	//lea(mem(, rsi, 8), rsi)            // rsi = cs_c * sizeof(double)
	
	
	
	                                   // now avoid loading C if beta == 0
	
	vxorpd(ymm0, ymm0, ymm0)           // set ymm0 to zero.
	vucomisd(xmm0, xmm3)               // set ZF if beta == 0.
	je(.DBETAZERO)                     // if ZF = 1, jump to beta == 0 case
	

	
	label(.DROWSTORED)
	
	
	vfmadd231pd(mem(rcx), ymm3, ymm4)
	vmovupd(ymm4, mem(rcx))
	add(rdi, rcx)
	
	vfmadd231pd(mem(rcx), ymm3, ymm5)
	vmovupd(ymm5, mem(rcx))
	add(rdi, rcx)
	
	vfmadd231pd(mem(rcx), ymm3, ymm6)
	vmovupd(ymm6, mem(rcx))
	//add(rdi, rcx)
	
	
	
	jmp(.DDONE)                        // jump to end.
	
	
	
	
	label(.DBETAZERO)
	

	
	label(.DROWSTORBZ)
	
	
	vmovupd(ymm4, mem(rcx))
	add(rdi, rcx)
	
	vmovupd(ymm5, mem(rcx))
	add(rdi, rcx)
	
	vmovupd(ymm6, mem(rcx))
	//add(rdi, rcx)
	
	
	
	
	label(.DDONE)
	
	


	add(imm(4*8), r12)                 // c_jj = r12 += 4*cs_c

	lea(mem(r14, r11, 4), r14)         // b_jj = r14 += 4*cs_b

	dec(r15)                           // jj -= 1;
	jne(.DLOOP3X4J)                    // iterate again if jj != 0.





	label(.DRETURN)

	

    end_asm(
	: // output operands (none)
	: // input operands
      [n_iter] "m" (n_iter),
      [k_iter16] "m" (k_iter16),
      [k_iter4] "m" (k_iter4),
      [k_left1] "m" (k_left1),
      [a]      "m" (a),
      [rs_a]   "m" (rs_a),
      [cs_a]   "m" (cs_a),
      [b]      "m" (b),
      [rs_b]   "m" (rs_b),
      [cs_b]   "m" (cs_b),
      [alpha]  "m" (alpha),
      [beta]   "m" (beta),
      [c]      "m" (c),
      [rs_c]   "m" (rs_c),
      [cs_c]   "m" (cs_c)/*,
      [a_next] "m" (a_next),
      [b_next] "m" (b_next)*/
	: // register clobber list
	  "rax", "rbx", "rcx", "rdx", "rsi", "rdi", "rbp",
	  "r8", "r9", "r10", "r11", "r12", "r13", "r14", "r15",
	  "xmm0", "xmm1", "xmm2", "xmm3",
	  "xmm4", "xmm5", "xmm6", "xmm7",
	  "xmm8", "xmm9", "xmm10", "xmm11",
	  "xmm12", "xmm13", "xmm14", "xmm15",
	  "memory"
	)

	consider_edge_cases:

	// Handle edge cases in the n dimension, if they exist.
	if ( n_left )
	{
		const dim_t      mr_cur = 3;
		const dim_t      j_edge = n0 - ( dim_t )n_left;

		double* restrict cij = c + j_edge*cs_c;
		double* restrict ai  = a;
		double* restrict bj  = b + j_edge*cs_b;

		if ( 2 <= n_left )
		{
			const dim_t nr_cur = 2;

			bli_dgemmsup_rd_haswell_asm_3x2
			//bli_dgemmsup_r_haswell_ref
			(
			  conja, conjb, mr_cur, nr_cur, k0,
			  alpha, ai, rs_a0, cs_a0, bj, rs_b0, cs_b0,
			  beta, cij, rs_c0, cs_c0, data, cntx
			);
			cij += nr_cur*cs_c0; bj += nr_cur*cs_b0; n_left -= nr_cur;
		}
		if ( 1 == n_left )
		{
			#if 1
			const dim_t nr_cur = 1;

			bli_dgemmsup_rd_haswell_asm_3x1
			//bli_dgemmsup_r_haswell_ref_3x1
			//bli_dgemmsup_r_haswell_ref
			(
			  conja, conjb, mr_cur, nr_cur, k0,
			  alpha, ai, rs_a0, cs_a0, bj, rs_b0, cs_b0,
			  beta, cij, rs_c0, cs_c0, data, cntx
			);
			#else
			bli_dgemv_ex
			(
			  BLIS_NO_TRANSPOSE, conjb, mr_cur, k0,
			  alpha, ai, rs_a0, cs_a0, bj, rs_b0,
			  beta, cij, rs_c0, cntx, NULL
			);
			#endif
		}
	}
}

void bli_dgemmsup_rd_haswell_asm_2x8n
     (
       conj_t              conja,
       conj_t              conjb,
       dim_t               m0,
       dim_t               n0,
       dim_t               k0,
       double*    restrict alpha,
       double*    restrict a, inc_t rs_a0, inc_t cs_a0,
       double*    restrict b, inc_t rs_b0, inc_t cs_b0,
       double*    restrict beta,
       double*    restrict c, inc_t rs_c0, inc_t cs_c0,
       auxinfo_t* restrict data,
       cntx_t*    restrict cntx
     )
{
	//void*    a_next = bli_auxinfo_next_a( data );
	//void*    b_next = bli_auxinfo_next_b( data );

	// Typecast local copies of integers in case dim_t and inc_t are a
	// different size than is expected by load instructions.
	uint64_t k_iter16 = k0 / 16;
	uint64_t k_left16 = k0 % 16;
	uint64_t k_iter4  = k_left16 / 4;
	uint64_t k_left1  = k_left16 % 4;

	uint64_t n_iter = n0 / 4;
	uint64_t n_left = n0 % 4;

	uint64_t rs_a   = rs_a0;
	uint64_t cs_a   = cs_a0;
	uint64_t rs_b   = rs_b0;
	uint64_t cs_b   = cs_b0;
	uint64_t rs_c   = rs_c0;
	uint64_t cs_c   = cs_c0;

	if ( n_iter == 0 ) goto consider_edge_cases;

	// -------------------------------------------------------------------------

	begin_asm()

	//vzeroall()                         // zero all xmm/ymm registers.
	
	mov(var(a), rdx)                   // load address of a.
	mov(var(rs_a), r8)                 // load rs_a
	//mov(var(cs_a), r9)                 // load cs_a
	lea(mem(, r8, 8), r8)              // rs_a *= sizeof(double)
	//lea(mem(, r9, 8), r9)              // cs_a *= sizeof(double)

	//lea(mem(r8, r8, 2), r13)           // r13 = 3*rs_a
	//lea(mem(r8, r8, 4), r15)           // r15 = 5*rs_a

	mov(var(b), r14)                   // load address of b.
	//mov(var(rs_b), r10)                // load rs_b
	mov(var(cs_b), r11)                // load cs_b
	//lea(mem(, r10, 8), r10)            // rs_b *= sizeof(double)
	lea(mem(, r11, 8), r11)            // cs_b *= sizeof(double)

	lea(mem(r11, r11, 2), r13)         // r13 = 3*cs_b
	//lea(mem(r8,  r8,  2), r10)         // r10 = 3*rs_a
	

	mov(var(c), r12)                   // load address of c
	mov(var(rs_c), rdi)                // load rs_c
	lea(mem(, rdi, 8), rdi)            // rs_c *= sizeof(double)



	// r12 = rcx = c
	// rdx = rax = a
	// r14 = rbx = b
	// r9  = unused
	// r15 = n dim index jj

	mov(var(n_iter), r15)              // jj = n_iter;

	label(.DLOOP3X4J)                  // LOOP OVER jj = [ n_iter ... 1 0 ]



#if 0
	vzeroall()                         // zero all xmm/ymm registers.
#else
	                                   // skylake can execute 3 vxorpd ipc with
	                                   // a latency of 1 cycle, while vzeroall
	                                   // has a latency of 12 cycles.
	vxorpd(ymm4,  ymm4,  ymm4)
	vxorpd(ymm5,  ymm5,  ymm5)
	vxorpd(ymm7,  ymm7,  ymm7)
	vxorpd(ymm8,  ymm8,  ymm8)
	vxorpd(ymm10, ymm10, ymm10)
	vxorpd(ymm11, ymm11, ymm11)
	vxorpd(ymm13, ymm13, ymm13)
	vxorpd(ymm14, ymm14, ymm14)
#endif


	lea(mem(r12), rcx)                 // rcx = c_iijj;
	lea(mem(rdx), rax)                 // rax = a_ii;
	lea(mem(r14), rbx)                 // rbx = b_jj;


#if 1
	//mov(var(rs_c), rdi)                // load rs_c
	//lea(mem(, rdi, 8), rdi)            // rs_c *= sizeof(double)
	prefetch(0, mem(rcx, 3*8))         // prefetch c + 0*rs_c
	prefetch(0, mem(rcx, rdi, 1, 3*8)) // prefetch c + 1*rs_c
#endif
	//lea(mem(r8,  r8,  4), rbp)         // rbp = 5*rs_a
	lea(mem(rbx, r11, 4), r10)         // r10 = rbx + 4*cs_b

	

	
	mov(var(k_iter16), rsi)            // i = k_iter16;
	test(rsi, rsi)                     // check i via logical AND.
	je(.DCONSIDKITER4)                 // if i == 0, jump to code that
	                                   // contains the k_iter4 loop.
	
	
	label(.DLOOPKITER16)               // MAIN LOOP
	
	
	// ---------------------------------- iteration 0

#if 1
	prefetch(0, mem(r10,         0*8)) // prefetch rbx + 4*cs_b
	prefetch(0, mem(r10, r11, 1, 0*8)) // prefetch rbx + 5*cs_b
#endif

	vmovupd(mem(rax       ), ymm0)
	vmovupd(mem(rax, r8, 1), ymm1)
	add(imm(4*8), rax)                 // a += 4*cs_a = 4*8;

	vmovupd(mem(rbx        ), ymm3)
	vfmadd231pd(ymm0, ymm3, ymm4)
	vfmadd231pd(ymm1, ymm3, ymm5)

	vmovupd(mem(rbx, r11, 1), ymm3)
	vfmadd231pd(ymm0, ymm3, ymm7)
	vfmadd231pd(ymm1, ymm3, ymm8)

	vmovupd(mem(rbx, r11, 2), ymm3)
	vfmadd231pd(ymm0, ymm3, ymm10)
	vfmadd231pd(ymm1, ymm3, ymm11)

	vmovupd(mem(rbx, r13, 1), ymm3)
	add(imm(4*8), rbx)                 // b += 4*rs_b = 4*8;
	vfmadd231pd(ymm0, ymm3, ymm13)
	vfmadd231pd(ymm1, ymm3, ymm14)

	
	// ---------------------------------- iteration 1

#if 1
	prefetch(0, mem(r10, r11, 2, 0*8)) // prefetch rbx + 6*cs_b
	prefetch(0, mem(r10, r13, 1, 0*8)) // prefetch rbx + 7*cs_b
#endif

	vmovupd(mem(rax       ), ymm0)
	vmovupd(mem(rax, r8, 1), ymm1)
	add(imm(4*8), rax)                 // a += 4*cs_a = 4*8;

	vmovupd(mem(rbx        ), ymm3)
	vfmadd231pd(ymm0, ymm3, ymm4)
	vfmadd231pd(ymm1, ymm3, ymm5)

	vmovupd(mem(rbx, r11, 1), ymm3)
	vfmadd231pd(ymm0, ymm3, ymm7)
	vfmadd231pd(ymm1, ymm3, ymm8)

	vmovupd(mem(rbx, r11, 2), ymm3)
	vfmadd231pd(ymm0, ymm3, ymm10)
	vfmadd231pd(ymm1, ymm3, ymm11)

	vmovupd(mem(rbx, r13, 1), ymm3)
	add(imm(4*8), rbx)                 // b += 4*rs_b = 4*8;
	vfmadd231pd(ymm0, ymm3, ymm13)
	vfmadd231pd(ymm1, ymm3, ymm14)


	// ---------------------------------- iteration 2
	
#if 1
	prefetch(0, mem(r10,         8*8)) // prefetch rbx + 4*cs_b + 8*rs_b
	prefetch(0, mem(r10, r11, 1, 8*8)) // prefetch rbx + 5*cs_b + 8*rs_b
#endif

	vmovupd(mem(rax       ), ymm0)
	vmovupd(mem(rax, r8, 1), ymm1)
	add(imm(4*8), rax)                 // a += 4*cs_a = 4*8;

	vmovupd(mem(rbx        ), ymm3)
	vfmadd231pd(ymm0, ymm3, ymm4)
	vfmadd231pd(ymm1, ymm3, ymm5)

	vmovupd(mem(rbx, r11, 1), ymm3)
	vfmadd231pd(ymm0, ymm3, ymm7)
	vfmadd231pd(ymm1, ymm3, ymm8)

	vmovupd(mem(rbx, r11, 2), ymm3)
	vfmadd231pd(ymm0, ymm3, ymm10)
	vfmadd231pd(ymm1, ymm3, ymm11)

	vmovupd(mem(rbx, r13, 1), ymm3)
	add(imm(4*8), rbx)                 // b += 4*rs_b = 4*8;
	vfmadd231pd(ymm0, ymm3, ymm13)
	vfmadd231pd(ymm1, ymm3, ymm14)


	// ---------------------------------- iteration 3

#if 1
	prefetch(0, mem(r10, r11, 2, 8*8)) // prefetch rbx + 6*cs_b + 8*rs_b
	prefetch(0, mem(r10, r13, 1, 8*8)) // prefetch rbx + 7*cs_b + 8*rs_b
	add(imm(16*8), r10)                 // r10 += 8*rs_b = 8*8;
#endif

	vmovupd(mem(rax       ), ymm0)
	vmovupd(mem(rax, r8, 1), ymm1)
	add(imm(4*8), rax)                 // a += 4*cs_a = 4*8;

	vmovupd(mem(rbx        ), ymm3)
	vfmadd231pd(ymm0, ymm3, ymm4)
	vfmadd231pd(ymm1, ymm3, ymm5)

	vmovupd(mem(rbx, r11, 1), ymm3)
	vfmadd231pd(ymm0, ymm3, ymm7)
	vfmadd231pd(ymm1, ymm3, ymm8)

	vmovupd(mem(rbx, r11, 2), ymm3)
	vfmadd231pd(ymm0, ymm3, ymm10)
	vfmadd231pd(ymm1, ymm3, ymm11)

	vmovupd(mem(rbx, r13, 1), ymm3)
	add(imm(4*8), rbx)                 // b += 4*rs_b = 4*8;
	vfmadd231pd(ymm0, ymm3, ymm13)
	vfmadd231pd(ymm1, ymm3, ymm14)

	

	dec(rsi)                           // i -= 1;
	jne(.DLOOPKITER16)                 // iterate again if i != 0.
	
	
	
	
	
	
	label(.DCONSIDKITER4)
	
	mov(var(k_iter4), rsi)             // i = k_iter4;
	test(rsi, rsi)                     // check i via logical AND.
	je(.DCONSIDKLEFT1)                 // if i == 0, jump to code that
	                                   // considers k_left1 loop.
	                                   // else, we prepare to enter k_iter4 loop.
	
	
	label(.DLOOPKITER4)                // EDGE LOOP (ymm)
	
	vmovupd(mem(rax       ), ymm0)
	vmovupd(mem(rax, r8, 1), ymm1)
	add(imm(4*8), rax)                 // a += 4*cs_a = 4*8;
	
	vmovupd(mem(rbx        ), ymm3)
	vfmadd231pd(ymm0, ymm3, ymm4)
	vfmadd231pd(ymm1, ymm3, ymm5)

	vmovupd(mem(rbx, r11, 1), ymm3)
	vfmadd231pd(ymm0, ymm3, ymm7)
	vfmadd231pd(ymm1, ymm3, ymm8)

	vmovupd(mem(rbx, r11, 2), ymm3)
	vfmadd231pd(ymm0, ymm3, ymm10)
	vfmadd231pd(ymm1, ymm3, ymm11)

	vmovupd(mem(rbx, r13, 1), ymm3)
	add(imm(4*8), rbx)                 // b += 4*rs_b = 4*8;
	vfmadd231pd(ymm0, ymm3, ymm13)
	vfmadd231pd(ymm1, ymm3, ymm14)

	
	dec(rsi)                           // i -= 1;
	jne(.DLOOPKITER4)                  // iterate again if i != 0.
	
	
	

	label(.DCONSIDKLEFT1)
	
	mov(var(k_left1), rsi)             // i = k_left1;
	test(rsi, rsi)                     // check i via logical AND.
	je(.DPOSTACCUM)                    // if i == 0, we're done; jump to end.
	                                   // else, we prepare to enter k_left1 loop.
	
	


	label(.DLOOPKLEFT1)                // EDGE LOOP (scalar)
	                                   // NOTE: We must use ymm registers here bc
	                                   // using the xmm registers would zero out the
	                                   // high bits of the destination registers,
	                                   // which would destory intermediate results.
	
	vmovsd(mem(rax       ), xmm0)
	vmovsd(mem(rax, r8, 1), xmm1)
	add(imm(1*8), rax)                 // a += 1*cs_a = 1*8;
	
	vmovsd(mem(rbx        ), xmm3)
	vfmadd231pd(ymm0, ymm3, ymm4)
	vfmadd231pd(ymm1, ymm3, ymm5)

	vmovsd(mem(rbx, r11, 1), xmm3)
	vfmadd231pd(ymm0, ymm3, ymm7)
	vfmadd231pd(ymm1, ymm3, ymm8)

	vmovsd(mem(rbx, r11, 2), xmm3)
	vfmadd231pd(ymm0, ymm3, ymm10)
	vfmadd231pd(ymm1, ymm3, ymm11)

	vmovsd(mem(rbx, r13, 1), xmm3)
	add(imm(1*8), rbx)                 // b += 1*rs_b = 1*8;
	vfmadd231pd(ymm0, ymm3, ymm13)
	vfmadd231pd(ymm1, ymm3, ymm14)

	
	dec(rsi)                           // i -= 1;
	jne(.DLOOPKLEFT1)                  // iterate again if i != 0.
	
	
	




	label(.DPOSTACCUM)
	
	                                   // ymm4  ymm7  ymm10 ymm13  
	                                   // ymm5  ymm8  ymm11 ymm14
	
	vhaddpd( ymm7, ymm4, ymm0 )
	vextractf128(imm(1), ymm0, xmm1 )
	vaddpd( xmm0, xmm1, xmm0 )

	vhaddpd( ymm13, ymm10, ymm2 )
	vextractf128(imm(1), ymm2, xmm1 )
	vaddpd( xmm2, xmm1, xmm2 )

	vperm2f128(imm(0x20), ymm2, ymm0, ymm4 )
	                                   // ymm4[0] = sum(ymm4);  ymm4[1] = sum(ymm7)
	                                   // ymm4[2] = sum(ymm10); ymm4[3] = sum(ymm13)

	vhaddpd( ymm8, ymm5, ymm0 )
	vextractf128(imm(1), ymm0, xmm1 )
	vaddpd( xmm0, xmm1, xmm0 )

	vhaddpd( ymm14, ymm11, ymm2 )
	vextractf128(imm(1), ymm2, xmm1 )
	vaddpd( xmm2, xmm1, xmm2 )

	vperm2f128(imm(0x20), ymm2, ymm0, ymm5 )
	                                   // ymm5[0] = sum(ymm5);  ymm5[1] = sum(ymm8)
	                                   // ymm5[2] = sum(ymm11); ymm5[3] = sum(ymm14)



	
	//mov(var(rs_c), rdi)                // load rs_c
	//lea(mem(, rdi, 8), rdi)            // rs_c *= sizeof(double)

	mov(var(alpha), rax)               // load address of alpha
	mov(var(beta), rbx)                // load address of beta
	vbroadcastsd(mem(rax), ymm0)       // load alpha and duplicate
	vbroadcastsd(mem(rbx), ymm3)       // load beta and duplicate
	
	vmulpd(ymm0, ymm4, ymm4)           // scale by alpha
	vmulpd(ymm0, ymm5, ymm5)
	
	
	
	
	
	
	//mov(var(cs_c), rsi)                // load cs_c
	//lea(mem(, rsi, 8), rsi)            // rsi = cs_c * sizeof(double)
	
	
	
	                                   // now avoid loading C if beta == 0
	
	vxorpd(ymm0, ymm0, ymm0)           // set ymm0 to zero.
	vucomisd(xmm0, xmm3)               // set ZF if beta == 0.
	je(.DBETAZERO)                     // if ZF = 1, jump to beta == 0 case
	

	
	label(.DROWSTORED)
	
	
	vfmadd231pd(mem(rcx), ymm3, ymm4)
	vmovupd(ymm4, mem(rcx))
	add(rdi, rcx)
	
	vfmadd231pd(mem(rcx), ymm3, ymm5)
	vmovupd(ymm5, mem(rcx))
	//add(rdi, rcx)
	
	
	
	jmp(.DDONE)                        // jump to end.
	
	
	
	
	label(.DBETAZERO)
	

	
	label(.DROWSTORBZ)
	
	
	vmovupd(ymm4, mem(rcx))
	add(rdi, rcx)
	
	vmovupd(ymm5, mem(rcx))
	//add(rdi, rcx)
	
	
	
	
	label(.DDONE)
	
	


	add(imm(4*8), r12)                 // c_jj = r12 += 4*cs_c

	lea(mem(r14, r11, 4), r14)         // b_jj = r14 += 4*cs_b

	dec(r15)                           // jj -= 1;
	jne(.DLOOP3X4J)                    // iterate again if jj != 0.





	label(.DRETURN)

	

    end_asm(
	: // output operands (none)
	: // input operands
      [n_iter] "m" (n_iter),
      [k_iter16] "m" (k_iter16),
      [k_iter4] "m" (k_iter4),
      [k_left1] "m" (k_left1),
      [a]      "m" (a),
      [rs_a]   "m" (rs_a),
      [cs_a]   "m" (cs_a),
      [b]      "m" (b),
      [rs_b]   "m" (rs_b),
      [cs_b]   "m" (cs_b),
      [alpha]  "m" (alpha),
      [beta]   "m" (beta),
      [c]      "m" (c),
      [rs_c]   "m" (rs_c),
      [cs_c]   "m" (cs_c)/*,
      [a_next] "m" (a_next),
      [b_next] "m" (b_next)*/
	: // register clobber list
	  "rax", "rbx", "rcx", "rdx", "rsi", "rdi", "rbp",
	  "r8", "r9", "r10", "r11", "r12", "r13", "r14", "r15",
	  "xmm0", "xmm1", "xmm2", "xmm3",
	  "xmm4", "xmm5", "xmm6", "xmm7",
	  "xmm8", "xmm9", "xmm10", "xmm11",
	  "xmm12", "xmm13", "xmm14", "xmm15",
	  "memory"
	)

	consider_edge_cases:

	// Handle edge cases in the n dimension, if they exist.
	if ( n_left )
	{
		const dim_t      mr_cur = 2;
		const dim_t      j_edge = n0 - ( dim_t )n_left;

		double* restrict cij = c + j_edge*cs_c;
		double* restrict ai  = a;
		double* restrict bj  = b + j_edge*cs_b;

		if ( 2 <= n_left )
		{
			const dim_t nr_cur = 2;

			bli_dgemmsup_rd_haswell_asm_2x2
			//bli_dgemmsup_r_haswell_ref
			(
			  conja, conjb, mr_cur, nr_cur, k0,
			  alpha, ai, rs_a0, cs_a0, bj, rs_b0, cs_b0,
			  beta, cij, rs_c0, cs_c0, data, cntx
			);
			cij += nr_cur*cs_c0; bj += nr_cur*cs_b0; n_left -= nr_cur;
		}
		if ( 1 == n_left )
		{
			#if 1
			const dim_t nr_cur = 1;

			bli_dgemmsup_rd_haswell_asm_2x1
			//bli_dgemmsup_r_haswell_ref
			(
			  conja, conjb, mr_cur, nr_cur, k0,
			  alpha, ai, rs_a0, cs_a0, bj, rs_b0, cs_b0,
			  beta, cij, rs_c0, cs_c0, data, cntx
			);
			#else
			bli_dgemv_ex
			(
			  BLIS_NO_TRANSPOSE, conjb, mr_cur, k0,
			  alpha, ai, rs_a0, cs_a0, bj, rs_b0,
			  beta, cij, rs_c0, cntx, NULL
			);
			#endif
		}
	}
}

void bli_dgemmsup_rd_haswell_asm_1x8n
     (
       conj_t              conja,
       conj_t              conjb,
       dim_t               m0,
       dim_t               n0,
       dim_t               k0,
       double*    restrict alpha,
       double*    restrict a, inc_t rs_a0, inc_t cs_a0,
       double*    restrict b, inc_t rs_b0, inc_t cs_b0,
       double*    restrict beta,
       double*    restrict c, inc_t rs_c0, inc_t cs_c0,
       auxinfo_t* restrict data,
       cntx_t*    restrict cntx
     )
{
	//void*    a_next = bli_auxinfo_next_a( data );
	//void*    b_next = bli_auxinfo_next_b( data );

	// Typecast local copies of integers in case dim_t and inc_t are a
	// different size than is expected by load instructions.
	uint64_t k_iter16 = k0 / 16;
	uint64_t k_left16 = k0 % 16;
	uint64_t k_iter4  = k_left16 / 4;
	uint64_t k_left1  = k_left16 % 4;

	uint64_t n_iter = n0 / 4;
	uint64_t n_left = n0 % 4;

	uint64_t rs_a   = rs_a0;
	uint64_t cs_a   = cs_a0;
	uint64_t rs_b   = rs_b0;
	uint64_t cs_b   = cs_b0;
	uint64_t rs_c   = rs_c0;
	uint64_t cs_c   = cs_c0;

	if ( n_iter == 0 ) goto consider_edge_cases;

	// -------------------------------------------------------------------------

	begin_asm()

	//vzeroall()                         // zero all xmm/ymm registers.
	
	mov(var(a), rdx)                   // load address of a.
	//mov(var(rs_a), r8)                 // load rs_a
	//mov(var(cs_a), r9)                 // load cs_a
	//lea(mem(, r8, 8), r8)              // rs_a *= sizeof(double)
	//lea(mem(, r9, 8), r9)              // cs_a *= sizeof(double)

	//lea(mem(r8, r8, 2), r13)           // r13 = 3*rs_a
	//lea(mem(r8, r8, 4), r15)           // r15 = 5*rs_a

	mov(var(b), r14)                   // load address of b.
	//mov(var(rs_b), r10)                // load rs_b
	mov(var(cs_b), r11)                // load cs_b
	//lea(mem(, r10, 8), r10)            // rs_b *= sizeof(double)
	lea(mem(, r11, 8), r11)            // cs_b *= sizeof(double)

	lea(mem(r11, r11, 2), r13)         // r13 = 3*cs_b
	//lea(mem(r8,  r8,  2), r10)         // r10 = 3*rs_a
	

	mov(var(c), r12)                   // load address of c
	mov(var(rs_c), rdi)                // load rs_c
	lea(mem(, rdi, 8), rdi)            // rs_c *= sizeof(double)



	// r12 = rcx = c
	// rdx = rax = a
	// r14 = rbx = b
	// r9  = unused
	// r15 = n dim index jj

	mov(var(n_iter), r15)              // jj = n_iter;

	label(.DLOOP3X4J)                  // LOOP OVER jj = [ n_iter ... 1 0 ]



#if 0
	vzeroall()                         // zero all xmm/ymm registers.
#else
	                                   // skylake can execute 3 vxorpd ipc with
	                                   // a latency of 1 cycle, while vzeroall
	                                   // has a latency of 12 cycles.
	vxorpd(ymm4,  ymm4,  ymm4)
	vxorpd(ymm7,  ymm7,  ymm7)
	vxorpd(ymm10, ymm10, ymm10)
	vxorpd(ymm13, ymm13, ymm13)
#endif


	lea(mem(r12), rcx)                 // rcx = c_iijj;
	lea(mem(rdx), rax)                 // rax = a_ii;
	lea(mem(r14), rbx)                 // rbx = b_jj;


#if 1
	//mov(var(rs_c), rdi)                // load rs_c
	//lea(mem(, rdi, 8), rdi)            // rs_c *= sizeof(double)
	prefetch(0, mem(rcx,         3*8)) // prefetch c + 0*rs_c
#endif
	//lea(mem(r8,  r8,  4), rbp)         // rbp = 5*rs_a
	lea(mem(rbx, r11, 4), r10)         // r10 = rbx + 4*cs_b

	

	
	mov(var(k_iter16), rsi)            // i = k_iter16;
	test(rsi, rsi)                     // check i via logical AND.
	je(.DCONSIDKITER4)                 // if i == 0, jump to code that
	                                   // contains the k_iter4 loop.
	
	
	label(.DLOOPKITER16)               // MAIN LOOP
	
	
	// ---------------------------------- iteration 0

#if 1
	prefetch(0, mem(r10,         0*8)) // prefetch rbx + 4*cs_b
	prefetch(0, mem(r10, r11, 1, 0*8)) // prefetch rbx + 5*cs_b
#endif

	vmovupd(mem(rax       ), ymm0)
	add(imm(4*8), rax)                 // a += 4*cs_a = 4*8;

	vmovupd(mem(rbx        ), ymm3)
	vfmadd231pd(ymm0, ymm3, ymm4)

	vmovupd(mem(rbx, r11, 1), ymm3)
	vfmadd231pd(ymm0, ymm3, ymm7)

	vmovupd(mem(rbx, r11, 2), ymm3)
	vfmadd231pd(ymm0, ymm3, ymm10)

	vmovupd(mem(rbx, r13, 1), ymm3)
	add(imm(4*8), rbx)                 // b += 4*rs_b = 4*8;
	vfmadd231pd(ymm0, ymm3, ymm13)

	
	// ---------------------------------- iteration 1

#if 1
	prefetch(0, mem(r10, r11, 2, 0*8)) // prefetch rbx + 6*cs_b
	prefetch(0, mem(r10, r13, 1, 0*8)) // prefetch rbx + 7*cs_b
#endif

	vmovupd(mem(rax       ), ymm0)
	add(imm(4*8), rax)                 // a += 4*cs_a = 4*8;

	vmovupd(mem(rbx        ), ymm3)
	vfmadd231pd(ymm0, ymm3, ymm4)

	vmovupd(mem(rbx, r11, 1), ymm3)
	vfmadd231pd(ymm0, ymm3, ymm7)

	vmovupd(mem(rbx, r11, 2), ymm3)
	vfmadd231pd(ymm0, ymm3, ymm10)

	vmovupd(mem(rbx, r13, 1), ymm3)
	add(imm(4*8), rbx)                 // b += 4*rs_b = 4*8;
	vfmadd231pd(ymm0, ymm3, ymm13)


	// ---------------------------------- iteration 2
	
#if 1
	prefetch(0, mem(r10,         8*8)) // prefetch rbx + 4*cs_b + 8*rs_b
	prefetch(0, mem(r10, r11, 1, 8*8)) // prefetch rbx + 5*cs_b + 8*rs_b
#endif

	vmovupd(mem(rax       ), ymm0)
	add(imm(4*8), rax)                 // a += 4*cs_a = 4*8;

	vmovupd(mem(rbx        ), ymm3)
	vfmadd231pd(ymm0, ymm3, ymm4)

	vmovupd(mem(rbx, r11, 1), ymm3)
	vfmadd231pd(ymm0, ymm3, ymm7)

	vmovupd(mem(rbx, r11, 2), ymm3)
	vfmadd231pd(ymm0, ymm3, ymm10)

	vmovupd(mem(rbx, r13, 1), ymm3)
	add(imm(4*8), rbx)                 // b += 4*rs_b = 4*8;
	vfmadd231pd(ymm0, ymm3, ymm13)


	// ---------------------------------- iteration 3

#if 1
	prefetch(0, mem(r10, r11, 2, 8*8)) // prefetch rbx + 6*cs_b + 8*rs_b
	prefetch(0, mem(r10, r13, 1, 8*8)) // prefetch rbx + 7*cs_b + 8*rs_b
	add(imm(16*8), r10)                 // r10 += 8*rs_b = 8*8;
#endif

	vmovupd(mem(rax       ), ymm0)
	add(imm(4*8), rax)                 // a += 4*cs_a = 4*8;

	vmovupd(mem(rbx        ), ymm3)
	vfmadd231pd(ymm0, ymm3, ymm4)

	vmovupd(mem(rbx, r11, 1), ymm3)
	vfmadd231pd(ymm0, ymm3, ymm7)

	vmovupd(mem(rbx, r11, 2), ymm3)
	vfmadd231pd(ymm0, ymm3, ymm10)

	vmovupd(mem(rbx, r13, 1), ymm3)
	add(imm(4*8), rbx)                 // b += 4*rs_b = 4*8;
	vfmadd231pd(ymm0, ymm3, ymm13)

	

	dec(rsi)                           // i -= 1;
	jne(.DLOOPKITER16)                 // iterate again if i != 0.
	
	
	
	
	
	
	label(.DCONSIDKITER4)
	
	mov(var(k_iter4), rsi)             // i = k_iter4;
	test(rsi, rsi)                     // check i via logical AND.
	je(.DCONSIDKLEFT1)                 // if i == 0, jump to code that
	                                   // considers k_left1 loop.
	                                   // else, we prepare to enter k_iter4 loop.
	
	
	label(.DLOOPKITER4)                // EDGE LOOP (ymm)
	
	vmovupd(mem(rax       ), ymm0)
	add(imm(4*8), rax)                 // a += 4*cs_a = 4*8;
	
	vmovupd(mem(rbx        ), ymm3)
	vfmadd231pd(ymm0, ymm3, ymm4)

	vmovupd(mem(rbx, r11, 1), ymm3)
	vfmadd231pd(ymm0, ymm3, ymm7)

	vmovupd(mem(rbx, r11, 2), ymm3)
	vfmadd231pd(ymm0, ymm3, ymm10)

	vmovupd(mem(rbx, r13, 1), ymm3)
	add(imm(4*8), rbx)                 // b += 4*rs_b = 4*8;
	vfmadd231pd(ymm0, ymm3, ymm13)

	
	dec(rsi)                           // i -= 1;
	jne(.DLOOPKITER4)                  // iterate again if i != 0.
	
	
	

	label(.DCONSIDKLEFT1)
	
	mov(var(k_left1), rsi)             // i = k_left1;
	test(rsi, rsi)                     // check i via logical AND.
	je(.DPOSTACCUM)                    // if i == 0, we're done; jump to end.
	                                   // else, we prepare to enter k_left1 loop.
	
	


	label(.DLOOPKLEFT1)                // EDGE LOOP (scalar)
	                                   // NOTE: We must use ymm registers here bc
	                                   // using the xmm registers would zero out the
	                                   // high bits of the destination registers,
	                                   // which would destory intermediate results.
	
	vmovsd(mem(rax       ), xmm0)
	add(imm(1*8), rax)                 // a += 1*cs_a = 1*8;
	
	vmovsd(mem(rbx        ), xmm3)
	vfmadd231pd(ymm0, ymm3, ymm4)

	vmovsd(mem(rbx, r11, 1), xmm3)
	vfmadd231pd(ymm0, ymm3, ymm7)

	vmovsd(mem(rbx, r11, 2), xmm3)
	vfmadd231pd(ymm0, ymm3, ymm10)

	vmovsd(mem(rbx, r13, 1), xmm3)
	add(imm(1*8), rbx)                 // b += 1*rs_b = 1*8;
	vfmadd231pd(ymm0, ymm3, ymm13)

	
	dec(rsi)                           // i -= 1;
	jne(.DLOOPKLEFT1)                  // iterate again if i != 0.
	
	
	




	label(.DPOSTACCUM)
	
	                                   // ymm4  ymm7  ymm10 ymm13  
	
	vhaddpd( ymm7, ymm4, ymm0 )
	vextractf128(imm(1), ymm0, xmm1 )
	vaddpd( xmm0, xmm1, xmm0 )

	vhaddpd( ymm13, ymm10, ymm2 )
	vextractf128(imm(1), ymm2, xmm1 )
	vaddpd( xmm2, xmm1, xmm2 )

	vperm2f128(imm(0x20), ymm2, ymm0, ymm4 )
	                                   // ymm4[0] = sum(ymm4);  ymm4[1] = sum(ymm7)
	                                   // ymm4[2] = sum(ymm10); ymm4[3] = sum(ymm13)



	
	//mov(var(rs_c), rdi)                // load rs_c
	//lea(mem(, rdi, 8), rdi)            // rs_c *= sizeof(double)

	mov(var(alpha), rax)               // load address of alpha
	mov(var(beta), rbx)                // load address of beta
	vbroadcastsd(mem(rax), ymm0)       // load alpha and duplicate
	vbroadcastsd(mem(rbx), ymm3)       // load beta and duplicate
	
	vmulpd(ymm0, ymm4, ymm4)           // scale by alpha
	
	
	
	
	
	
	//mov(var(cs_c), rsi)                // load cs_c
	//lea(mem(, rsi, 8), rsi)            // rsi = cs_c * sizeof(double)
	
	
	
	                                   // now avoid loading C if beta == 0
	
	vxorpd(ymm0, ymm0, ymm0)           // set ymm0 to zero.
	vucomisd(xmm0, xmm3)               // set ZF if beta == 0.
	je(.DBETAZERO)                     // if ZF = 1, jump to beta == 0 case
	

	
	label(.DROWSTORED)
	
	
	vfmadd231pd(mem(rcx), ymm3, ymm4)
	vmovupd(ymm4, mem(rcx))
	//add(rdi, rcx)
	
	
	
	jmp(.DDONE)                        // jump to end.
	
	
	
	
	label(.DBETAZERO)
	

	
	label(.DROWSTORBZ)
	
	
	vmovupd(ymm4, mem(rcx))
	//add(rdi, rcx)
	
	
	
	
	label(.DDONE)
	
	


	add(imm(4*8), r12)                 // c_jj = r12 += 4*cs_c

	lea(mem(r14, r11, 4), r14)         // b_jj = r14 += 4*cs_b

	dec(r15)                           // jj -= 1;
	jne(.DLOOP3X4J)                    // iterate again if jj != 0.





	label(.DRETURN)

	

    end_asm(
	: // output operands (none)
	: // input operands
      [n_iter] "m" (n_iter),
      [k_iter16] "m" (k_iter16),
      [k_iter4] "m" (k_iter4),
      [k_left1] "m" (k_left1),
      [a]      "m" (a),
      [rs_a]   "m" (rs_a),
      [cs_a]   "m" (cs_a),
      [b]      "m" (b),
      [rs_b]   "m" (rs_b),
      [cs_b]   "m" (cs_b),
      [alpha]  "m" (alpha),
      [beta]   "m" (beta),
      [c]      "m" (c),
      [rs_c]   "m" (rs_c),
      [cs_c]   "m" (cs_c)/*,
      [a_next] "m" (a_next),
      [b_next] "m" (b_next)*/
	: // register clobber list
	  "rax", "rbx", "rcx", "rdx", "rsi", "rdi", "rbp",
	  "r8", "r9", "r10", "r11", "r12", "r13", "r14", "r15",
	  "xmm0", "xmm1", "xmm2", "xmm3",
	  "xmm4", "xmm5", "xmm6", "xmm7",
	  "xmm8", "xmm9", "xmm10", "xmm11",
	  "xmm12", "xmm13", "xmm14", "xmm15",
	  "memory"
	)

	consider_edge_cases:

	// Handle edge cases in the n dimension, if they exist.
	if ( n_left )
	{
		const dim_t      mr_cur = 1;
		const dim_t      j_edge = n0 - ( dim_t )n_left;

		double* restrict cij = c + j_edge*cs_c;
		double* restrict ai  = a;
		double* restrict bj  = b + j_edge*cs_b;

		if ( 2 <= n_left )
		{
			const dim_t nr_cur = 2;

			bli_dgemmsup_rd_haswell_asm_1x2
			//bli_dgemmsup_r_haswell_ref
			(
			  conja, conjb, mr_cur, nr_cur, k0,
			  alpha, ai, rs_a0, cs_a0, bj, rs_b0, cs_b0,
			  beta, cij, rs_c0, cs_c0, data, cntx
			);
			cij += nr_cur*cs_c0; bj += nr_cur*cs_b0; n_left -= nr_cur;
		}
		if ( 1 == n_left )
		{
			#if 1
			const dim_t nr_cur = 1;

			bli_dgemmsup_rd_haswell_asm_1x1
			//bli_dgemmsup_r_haswell_ref
			(
			  conja, conjb, mr_cur, nr_cur, k0,
			  alpha, ai, rs_a0, cs_a0, bj, rs_b0, cs_b0,
			  beta, cij, rs_c0, cs_c0, data, cntx
			);
			#else
			bli_ddotxv_ex
			(
			  conja, conjb, k0,
			  alpha, ai, cs_a0, bj, rs_b0,
			  beta, cij, cntx, NULL
			);
			#endif
		}
	}
}

