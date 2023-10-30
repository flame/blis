/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2014, The University of Texas at Austin
   Copyright (C) 2019 - 2023, Advanced Micro Devices, Inc. All rights reserved.

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
*/

// Prototype reference microkernels.
GEMMSUP_KER_PROT( float,    s, gemmsup_r_haswell_ref )
GEMMSUP_KER_PROT( double,   d, gemmsup_r_haswell_ref )
GEMMSUP_KER_PROT( scomplex, c, gemmsup_r_haswell_ref )
GEMMSUP_KER_PROT( dcomplex, z, gemmsup_r_haswell_ref )

// Define parameters and variables for edge case kernel map.
#define NUM_MR 4
#define NUM_NR 4
#define FUNCPTR_T dgemmsup_ker_ft

static dim_t mrs[NUM_MR] = { 6, 3, 2, 1 };
static dim_t nrs[NUM_NR] = { 8, 4, 2, 1 };
static FUNCPTR_T kmap[NUM_MR][NUM_NR] =
{     /*  8                                4                                2                                1  */
/* 6 */ { bli_dgemmsup_rd_haswell_asm_6x8, bli_dgemmsup_rd_haswell_asm_6x4, bli_dgemmsup_rd_haswell_asm_6x2, bli_dgemmsup_r_haswell_ref },
/* 3 */ { bli_dgemmsup_rd_haswell_asm_3x8, bli_dgemmsup_rd_haswell_asm_3x4, bli_dgemmsup_rd_haswell_asm_3x2, bli_dgemmsup_r_haswell_ref },
/* 2 */ { bli_dgemmsup_rd_haswell_asm_2x8, bli_dgemmsup_rd_haswell_asm_2x4, bli_dgemmsup_rd_haswell_asm_2x2, bli_dgemmsup_r_haswell_ref },
/* 1 */ { bli_dgemmsup_rd_haswell_asm_1x8, bli_dgemmsup_rd_haswell_asm_1x4, bli_dgemmsup_rd_haswell_asm_1x2, bli_dgemmsup_r_haswell_ref }
};


void bli_dgemmsup_rd_haswell_asm_6x8
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
	// Use a reference kernel if this is an edge case in the m or n
	// dimensions.
	if ( m0 < 6 || n0 < 8 )
	{
		dim_t            n_left = n0;
		double* restrict cj     = c;
		double* restrict bj     = b;

		// Iterate across columns (corresponding to elements of nrs) until
		// n_left is zero.
		for ( dim_t j = 0; n_left != 0; ++j )
		{
			const dim_t nr_cur = nrs[ j ];

			// Once we find the value of nrs that is less than (or equal to)
			// n_left, we use the kernels in that column.
			if ( nr_cur <= n_left )
			{
				dim_t            m_left = m0;
				double* restrict cij    = cj;
				double* restrict ai     = a;

				// Iterate down the current column (corresponding to elements
				// of mrs) until m_left is zero.
				for ( dim_t i = 0; m_left != 0; ++i )
				{
					const dim_t mr_cur = mrs[ i ];

					// Once we find the value of mrs that is less than (or equal
					// to) m_left, we select that kernel.
					if ( mr_cur <= m_left )
					{
						FUNCPTR_T ker_fp = kmap[i][j];

						//printf( "executing %d x %d sup kernel.\n", (int)mr_cur, (int)nr_cur );

						// Call the kernel using current mrs and nrs values.
						ker_fp
						(
						  conja, conjb, mr_cur, nr_cur, k0,
						  alpha, ai, rs_a0, cs_a0, bj, rs_b0, cs_b0,
						  beta, cij, rs_c0, cs_c0, data, cntx
						);

						// Advance C and A pointers by the mrs and nrs we just
						// used, and decrement m_left.
						cij += mr_cur*rs_c0; ai += mr_cur*rs_a0; m_left -= mr_cur;
					} 
				}

				// Advance C and B pointers by the mrs and nrs we just used, and
				// decrement n_left.
				cj += nr_cur*cs_c0; bj += nr_cur*cs_b0; n_left -= nr_cur;
			}
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

	uint64_t rs_a   = rs_a0;
	uint64_t cs_a   = cs_a0;
	uint64_t rs_b   = rs_b0;
	uint64_t cs_b   = cs_b0;
	uint64_t rs_c   = rs_c0;
	uint64_t cs_c   = cs_c0;

	// -------------------------------------------------------------------------

	begin_asm()

	//vzeroall()                         // zero all xmm/ymm registers.
	
	mov(var(a), r12)                   // load address of a.
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
	

	mov(var(c), r10)                   // load address of c
	mov(var(rs_c), rdi)                // load rs_c
	lea(mem(, rdi, 8), rdi)            // rs_c *= sizeof(double)



	// r10 = rcx = c
	// r12 = rax = a
	// r14 = rbx = b
	// r9  = m dim index ii
	// r15 = n dim index jj

#if 1
	mov(imm(0), r9)                    // ii = 0;

	label(.DLOOP3X4I)                  // LOOP OVER ii = [ 0 1 ... ]



	lea(mem(   , r9,  1), rsi)         // rsi = r9 = 3*ii;
	imul(rdi, rsi)                     // rsi *= rs_c;
	lea(mem(r10, rsi, 1), rdx)         // rdx = c_jj + 3*ii*rs_c;

	lea(mem(   , r9,  1), rsi)         // rsi = r9 = 3*ii;
	imul(r8,  rsi)                     // rsi *= rs_a;
	lea(mem(r12, rsi, 1), r12)         // rax = a + 3*ii*rs_a;



	mov(imm(0), r15)                   // jj = 0;

	label(.DLOOP3X4J)                  // LOOP OVER jj = [ 0 1 ... ]

	vzeroall()                         // zero all xmm/ymm registers.


	lea(mem(   , r15, 1), rsi)         // rsi = r15 = 4*jj;
	imul(imm(1*8), rsi)                // rsi *= cs_c = 1*8
	lea(mem(rdx, rsi, 1), rcx)         // rcx = c + 4*jj*cs_c;

	lea(mem(   , r15, 1), rsi)         // rsi = r15 = 4*jj;
	imul(r11, rsi)                     // rsi *= cs_b;
	lea(mem(r14, rsi, 1), rbx)         // rbx = b + 4*jj*cs_b;

	lea(mem(   , r12, 1), rax)         // rax = a_ii;
#endif

#if 0
	prefetch(0, mem(rcx, 7*8))         // prefetch c + 0*rs_c
	prefetch(0, mem(rcx, rdi, 1, 7*8)) // prefetch c + 1*rs_c
	prefetch(0, mem(rcx, rdi, 2, 7*8)) // prefetch c + 2*rs_c
#else
	prefetch(0, mem(rcx, 3*8))         // prefetch c + 0*rs_c
	prefetch(0, mem(rcx, rdi, 1, 3*8)) // prefetch c + 1*rs_c
	prefetch(0, mem(rcx, rdi, 2, 3*8)) // prefetch c + 2*rs_c
#endif

	

	
	mov(var(k_iter16), rsi)            // i = k_iter16;
	test(rsi, rsi)                     // check i via logical AND.
	je(.DCONSIDKITER4)                 // if i == 0, jump to code that
	                                   // contains the k_iter4 loop.
	
	
	label(.DLOOPKITER16)               // MAIN LOOP
	
	
	// ---------------------------------- iteration 0

	vmovupd(mem(rax       ), ymm0)
	vmovupd(mem(rax, r8, 1), ymm1)
	vmovupd(mem(rax, r8, 2), ymm2)
	add(imm(4*8), rax)                 // a += 4*cs_b = 4*8;

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

	vmovupd(mem(rax       ), ymm0)
	vmovupd(mem(rax, r8, 1), ymm1)
	vmovupd(mem(rax, r8, 2), ymm2)
	add(imm(4*8), rax)                 // a += 4*cs_b = 4*8;

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
	
	vmovupd(mem(rax       ), ymm0)
	vmovupd(mem(rax, r8, 1), ymm1)
	vmovupd(mem(rax, r8, 2), ymm2)
	add(imm(4*8), rax)                 // a += 4*cs_b = 4*8;

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

	vmovupd(mem(rax       ), ymm0)
	vmovupd(mem(rax, r8, 1), ymm1)
	vmovupd(mem(rax, r8, 2), ymm2)
	add(imm(4*8), rax)                 // a += 4*cs_b = 4*8;

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
	add(imm(4*8), rax)                 // a += 4*cs_b = 4*8;
	
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
	add(imm(1*8), rax)                 // a += 1*cs_b = 1*8;
	
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
	vaddpd( xmm0, xmm1, xmm0 )         // xmm0[0] = sum(ymm4); xmm0[1] = sum(ymm7)

	vhaddpd( ymm13, ymm10, ymm2 )
	vextractf128(imm(1), ymm2, xmm1 )
	vaddpd( xmm2, xmm1, xmm2 )         // xmm2[0] = sum(ymm10); xmm2[1] = sum(ymm13)

	vperm2f128(imm(0x20), ymm2, ymm0, ymm4 )


	vhaddpd( ymm8, ymm5, ymm0 )
	vextractf128(imm(1), ymm0, xmm1 )
	vaddpd( xmm0, xmm1, xmm0 )

	vhaddpd( ymm14, ymm11, ymm2 )
	vextractf128(imm(1), ymm2, xmm1 )
	vaddpd( xmm2, xmm1, xmm2 )

	vperm2f128(imm(0x20), ymm2, ymm0, ymm5 )


	vhaddpd( ymm9, ymm6, ymm0 )
	vextractf128(imm(1), ymm0, xmm1 )
	vaddpd( xmm0, xmm1, xmm0 )

	vhaddpd( ymm15, ymm12, ymm2 )
	vextractf128(imm(1), ymm2, xmm1 )
	vaddpd( xmm2, xmm1, xmm2 )

	vperm2f128(imm(0x20), ymm2, ymm0, ymm6 )

	                                   // ymm4 = sum(ymm4) sum(ymm7) sum(ymm10) sum(ymm13)
	                                   // ymm5 = sum(ymm5) sum(ymm8) sum(ymm11) sum(ymm14)
	                                   // ymm6 = sum(ymm6) sum(ymm9) sum(ymm12) sum(ymm15)


	
	mov(var(alpha), rax)               // load address of alpha
	mov(var(beta), rbx)                // load address of beta
	vbroadcastsd(mem(rax), ymm0)       // load alpha and duplicate
	vbroadcastsd(mem(rbx), ymm3)       // load beta and duplicate
	
	vmulpd(ymm0, ymm4, ymm4)           // scale by alpha
	vmulpd(ymm0, ymm5, ymm5)
	vmulpd(ymm0, ymm6, ymm6)
	
	
	
	
	
	
	mov(var(cs_c), rsi)                // load cs_c
	lea(mem(, rsi, 8), rsi)            // rsi = cs_c * sizeof(double)
	
	
	
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
	
	


#if 1
	add(imm(4), r15)                   // jj += 4;
	cmp(imm(4), r15)                   // compare jj to 4
	jle(.DLOOP3X4J)                    // if jj <= 4, jump to beginning
	                                   // of jj loop; otherwise, loop ends.

	add(imm(3), r9)                    // ii += 3;
	cmp(imm(3), r9)                    // compare ii to 3
	jle(.DLOOP3X4I)                    // if ii <= 3, jump to beginning
#endif



	label(.DRETURN)

	

    end_asm(
	: // output operands (none)
	: // input operands
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
	  "rax", "rbx", "rcx", "rdx", "rsi", "rdi", 
	  "r8", "r9", "r10", "r11", "r12", "r13", "r14", "r15",
	  "xmm0", "xmm1", "xmm2", "xmm3",
	  "xmm4", "xmm5", "xmm6", "xmm7",
	  "xmm8", "xmm9", "xmm10", "xmm11",
	  "xmm12", "xmm13", "xmm14", "xmm15",
	  "ymm0", "ymm1", "ymm2", "ymm3", "ymm4", "ymm5", "ymm6", "ymm7",
	  "ymm8", "ymm9", "ymm10", "ymm11", "ymm12", "ymm13", "ymm14",
	  "ymm15", "memory"
	)
}

void bli_dgemmsup_rd_haswell_asm_3x8
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

	uint64_t rs_a   = rs_a0;
	uint64_t cs_a   = cs_a0;
	uint64_t rs_b   = rs_b0;
	uint64_t cs_b   = cs_b0;
	uint64_t rs_c   = rs_c0;
	uint64_t cs_c   = cs_c0;

	// -------------------------------------------------------------------------

	begin_asm()

	//vzeroall()                         // zero all xmm/ymm registers.
	
	mov(var(a), r12)                   // load address of a.
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
	

	mov(var(c), r10)                   // load address of c
	mov(var(rs_c), rdi)                // load rs_c
	lea(mem(, rdi, 8), rdi)            // rs_c *= sizeof(double)



	// r10 = rcx = c
	// r12 = rax = a
	// r14 = rbx = b
	// r9  = m dim index ii
	// r15 = n dim index jj

	mov(imm(0), r15)                   // jj = 0;

	label(.DLOOP3X4J)                  // LOOP OVER jj = [ 0 1 ... ]



	vzeroall()                         // zero all xmm/ymm registers.

	lea(mem(   , r15, 1), rsi)         // rsi = r15 = 4*jj;
	imul(imm(1*8), rsi)                // rsi *= cs_c = 1*8
	lea(mem(r10, rsi, 1), rcx)         // rcx = c + 4*jj*cs_c;

	lea(mem(   , r15, 1), rsi)         // rsi = r15 = 4*jj;
	imul(r11, rsi)                     // rsi *= cs_b;
	lea(mem(r14, rsi, 1), rbx)         // rbx = b + 4*jj*cs_b;

	lea(mem(   , r12, 1), rax)         // rax = a;

	prefetch(0, mem(rcx, 7*8))         // prefetch c + 0*rs_c
	prefetch(0, mem(rcx, rdi, 1, 7*8)) // prefetch c + 1*rs_c
	prefetch(0, mem(rcx, rdi, 2, 7*8)) // prefetch c + 2*rs_c

	
	
	
	mov(var(k_iter16), rsi)            // i = k_iter16;
	test(rsi, rsi)                     // check i via logical AND.
	je(.DCONSIDKITER4)                 // if i == 0, jump to code that
	                                   // contains the k_iter4 loop.
	
	
	label(.DLOOPKITER16)               // MAIN LOOP
	
	
	// ---------------------------------- iteration 0

	vmovupd(mem(rax       ), ymm0)
	vmovupd(mem(rax, r8, 1), ymm1)
	vmovupd(mem(rax, r8, 2), ymm2)
	add(imm(4*8), rax)                 // a += 4*cs_b = 4*8;

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

	vmovupd(mem(rax       ), ymm0)
	vmovupd(mem(rax, r8, 1), ymm1)
	vmovupd(mem(rax, r8, 2), ymm2)
	add(imm(4*8), rax)                 // a += 4*cs_b = 4*8;

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
	
	vmovupd(mem(rax       ), ymm0)
	vmovupd(mem(rax, r8, 1), ymm1)
	vmovupd(mem(rax, r8, 2), ymm2)
	add(imm(4*8), rax)                 // a += 4*cs_b = 4*8;

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

	vmovupd(mem(rax       ), ymm0)
	vmovupd(mem(rax, r8, 1), ymm1)
	vmovupd(mem(rax, r8, 2), ymm2)
	add(imm(4*8), rax)                 // a += 4*cs_b = 4*8;

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
	add(imm(4*8), rax)                 // a += 4*cs_b = 4*8;
	
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
	add(imm(1*8), rax)                 // a += 1*cs_b = 1*8;
	
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
	vaddpd( xmm0, xmm1, xmm0 )         // xmm0[0] = sum(ymm4); xmm0[1] = sum(ymm7)

	vhaddpd( ymm13, ymm10, ymm2 )
	vextractf128(imm(1), ymm2, xmm1 )
	vaddpd( xmm2, xmm1, xmm2 )         // xmm2[0] = sum(ymm10); xmm2[1] = sum(ymm13)

	vperm2f128(imm(0x20), ymm2, ymm0, ymm4 )


	vhaddpd( ymm8, ymm5, ymm0 )
	vextractf128(imm(1), ymm0, xmm1 )
	vaddpd( xmm0, xmm1, xmm0 )

	vhaddpd( ymm14, ymm11, ymm2 )
	vextractf128(imm(1), ymm2, xmm1 )
	vaddpd( xmm2, xmm1, xmm2 )

	vperm2f128(imm(0x20), ymm2, ymm0, ymm5 )


	vhaddpd( ymm9, ymm6, ymm0 )
	vextractf128(imm(1), ymm0, xmm1 )
	vaddpd( xmm0, xmm1, xmm0 )

	vhaddpd( ymm15, ymm12, ymm2 )
	vextractf128(imm(1), ymm2, xmm1 )
	vaddpd( xmm2, xmm1, xmm2 )

	vperm2f128(imm(0x20), ymm2, ymm0, ymm6 )

	                                   // ymm4 = sum(ymm4) sum(ymm7) sum(ymm10) sum(ymm13)
	                                   // ymm5 = sum(ymm5) sum(ymm8) sum(ymm11) sum(ymm14)
	                                   // ymm6 = sum(ymm6) sum(ymm9) sum(ymm12) sum(ymm15)


	
	mov(var(alpha), rax)               // load address of alpha
	mov(var(beta), rbx)                // load address of beta
	vbroadcastsd(mem(rax), ymm0)       // load alpha and duplicate
	vbroadcastsd(mem(rbx), ymm3)       // load beta and duplicate
	
	vmulpd(ymm0, ymm4, ymm4)           // scale by alpha
	vmulpd(ymm0, ymm5, ymm5)
	vmulpd(ymm0, ymm6, ymm6)
	
	
	
	
	
	
	mov(var(cs_c), rsi)                // load cs_c
	lea(mem(, rsi, 8), rsi)            // rsi = cs_c * sizeof(double)
	
	
	
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
	
	


	add(imm(4), r15)                   // jj += 4;
	cmp(imm(4), r15)                   // compare jj to 4
	jle(.DLOOP3X4J)                    // if jj <= 4, jump to beginning
	                                   // of jj loop; otherwise, loop ends.



	label(.DRETURN)

	

    end_asm(
	: // output operands (none)
	: // input operands
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
	  "rax", "rbx", "rcx", "rdx", "rsi", "rdi", 
	  "r8", "r9", "r10", "r11", "r12", "r13", "r14", "r15",
	  "xmm0", "xmm1", "xmm2", "xmm3",
	  "xmm4", "xmm5", "xmm6", "xmm7",
	  "xmm8", "xmm9", "xmm10", "xmm11",
	  "xmm12", "xmm13", "xmm14", "xmm15",
	  "ymm0", "ymm1", "ymm2", "ymm3", "ymm4", "ymm5", "ymm6", "ymm7",
	  "ymm8", "ymm9", "ymm10", "ymm11", "ymm12", "ymm13", "ymm14",
	  "ymm15", "memory"
	)
}

void bli_dgemmsup_rd_haswell_asm_2x8
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
#if 0
	bli_dgemmsup_r_haswell_ref
	(
	  conja, conjb, m0, n0, k0,
	  alpha, a, rs_a0, cs_a0, b, rs_b0, cs_b0,
	  beta, c, rs_c0, cs_c0, data, cntx
	);
	return;
#endif
	//void*    a_next = bli_auxinfo_next_a( data );
	//void*    b_next = bli_auxinfo_next_b( data );

	// Typecast local copies of integers in case dim_t and inc_t are a
	// different size than is expected by load instructions.
	uint64_t k_iter16 = k0 / 16;
	uint64_t k_left16 = k0 % 16;
	uint64_t k_iter4  = k_left16 / 4;
	uint64_t k_left1  = k_left16 % 4;

	uint64_t rs_a   = rs_a0;
	uint64_t cs_a   = cs_a0;
	uint64_t rs_b   = rs_b0;
	uint64_t cs_b   = cs_b0;
	uint64_t rs_c   = rs_c0;
	uint64_t cs_c   = cs_c0;

	// -------------------------------------------------------------------------

	begin_asm()

	//vzeroall()                         // zero all xmm/ymm registers.
	
	mov(var(a), r12)                   // load address of a.
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
	

	mov(var(c), r10)                   // load address of c
	mov(var(rs_c), rdi)                // load rs_c
	lea(mem(, rdi, 8), rdi)            // rs_c *= sizeof(double)



	// r10 = rcx = c
	// r12 = rax = a
	// r14 = rbx = b
	// r9  = m dim index ii
	// r15 = n dim index jj

	mov(imm(0), r15)                   // jj = 0;

	label(.DLOOP3X4J)                  // LOOP OVER jj = [ 0 1 ... ]



	vzeroall()                         // zero all xmm/ymm registers.

	lea(mem(   , r15, 1), rsi)         // rsi = r15 = 4*jj;
	imul(imm(1*8), rsi)                // rsi *= cs_c = 1*8
	lea(mem(r10, rsi, 1), rcx)         // rcx = c + 4*jj*cs_c;

	lea(mem(   , r15, 1), rsi)         // rsi = r15 = 4*jj;
	imul(r11, rsi)                     // rsi *= cs_b;
	lea(mem(r14, rsi, 1), rbx)         // rbx = b + 4*jj*cs_b;

	lea(mem(   , r12, 1), rax)         // rax = a;

	prefetch(0, mem(rcx, 7*8))         // prefetch c + 0*rs_c
	prefetch(0, mem(rcx, rdi, 1, 7*8)) // prefetch c + 1*rs_c

	
	
	
	mov(var(k_iter16), rsi)            // i = k_iter16;
	test(rsi, rsi)                     // check i via logical AND.
	je(.DCONSIDKITER4)                 // if i == 0, jump to code that
	                                   // contains the k_iter4 loop.
	
	
	label(.DLOOPKITER16)               // MAIN LOOP
	
	
	// ---------------------------------- iteration 0

	vmovupd(mem(rax       ), ymm0)
	vmovupd(mem(rax, r8, 1), ymm1)
	add(imm(4*8), rax)                 // a += 4*cs_b = 4*8;

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

	vmovupd(mem(rax       ), ymm0)
	vmovupd(mem(rax, r8, 1), ymm1)
	add(imm(4*8), rax)                 // a += 4*cs_b = 4*8;

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
	
	vmovupd(mem(rax       ), ymm0)
	vmovupd(mem(rax, r8, 1), ymm1)
	add(imm(4*8), rax)                 // a += 4*cs_b = 4*8;

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

	vmovupd(mem(rax       ), ymm0)
	vmovupd(mem(rax, r8, 1), ymm1)
	add(imm(4*8), rax)                 // a += 4*cs_b = 4*8;

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
	add(imm(4*8), rax)                 // a += 4*cs_b = 4*8;
	
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
	add(imm(1*8), rax)                 // a += 1*cs_b = 1*8;
	
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
	vaddpd( xmm0, xmm1, xmm0 )         // xmm0[0] = sum(ymm4); xmm0[1] = sum(ymm7)

	vhaddpd( ymm13, ymm10, ymm2 )
	vextractf128(imm(1), ymm2, xmm1 )
	vaddpd( xmm2, xmm1, xmm2 )         // xmm2[0] = sum(ymm10); xmm2[1] = sum(ymm13)

	vperm2f128(imm(0x20), ymm2, ymm0, ymm4 )


	vhaddpd( ymm8, ymm5, ymm0 )
	vextractf128(imm(1), ymm0, xmm1 )
	vaddpd( xmm0, xmm1, xmm0 )

	vhaddpd( ymm14, ymm11, ymm2 )
	vextractf128(imm(1), ymm2, xmm1 )
	vaddpd( xmm2, xmm1, xmm2 )

	vperm2f128(imm(0x20), ymm2, ymm0, ymm5 )

	                                   // ymm4 = sum(ymm4) sum(ymm7) sum(ymm10) sum(ymm13)
	                                   // ymm5 = sum(ymm5) sum(ymm8) sum(ymm11) sum(ymm14)


	
	mov(var(alpha), rax)               // load address of alpha
	mov(var(beta), rbx)                // load address of beta
	vbroadcastsd(mem(rax), ymm0)       // load alpha and duplicate
	vbroadcastsd(mem(rbx), ymm3)       // load beta and duplicate
	
	vmulpd(ymm0, ymm4, ymm4)           // scale by alpha
	vmulpd(ymm0, ymm5, ymm5)
	
	
	
	
	
	
	mov(var(cs_c), rsi)                // load cs_c
	lea(mem(, rsi, 8), rsi)            // rsi = cs_c * sizeof(double)
	
	
	
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
	
	


	add(imm(4), r15)                   // jj += 4;
	cmp(imm(4), r15)                   // compare jj to 4
	jle(.DLOOP3X4J)                    // if jj <= 4, jump to beginning
	                                   // of jj loop; otherwise, loop ends.



	label(.DRETURN)

	

    end_asm(
	: // output operands (none)
	: // input operands
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
	  "rax", "rbx", "rcx", "rdx", "rsi", "rdi", 
	  "r8", "r9", "r10", "r11", "r12", "r13", "r14", "r15",
	  "xmm0", "xmm1", "xmm2", "xmm3",
	  "xmm4", "xmm5", "xmm6", "xmm7",
	  "xmm8", "xmm9", "xmm10", "xmm11",
	  "xmm12", "xmm13", "xmm14", "xmm15",
	  "ymm0", "ymm1", "ymm2", "ymm3", "ymm4", "ymm5", "ymm7", "ymm8",
	  "ymm10", "ymm11", "ymm13", "ymm14", "memory"
	)
}

void bli_dgemmsup_rd_haswell_asm_1x8
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
#if 0
	bli_dgemmsup_r_haswell_ref
	(
	  conja, conjb, m0, n0, k0,
	  alpha, a, rs_a0, cs_a0, b, rs_b0, cs_b0,
	  beta, c, rs_c0, cs_c0, data, cntx
	);
	return;
#endif
	//void*    a_next = bli_auxinfo_next_a( data );
	//void*    b_next = bli_auxinfo_next_b( data );

	// Typecast local copies of integers in case dim_t and inc_t are a
	// different size than is expected by load instructions.
	uint64_t k_iter16 = k0 / 16;
	uint64_t k_left16 = k0 % 16;
	uint64_t k_iter4  = k_left16 / 4;
	uint64_t k_left1  = k_left16 % 4;

	uint64_t rs_a   = rs_a0;
	uint64_t cs_a   = cs_a0;
	uint64_t rs_b   = rs_b0;
	uint64_t cs_b   = cs_b0;
	uint64_t rs_c   = rs_c0;
	uint64_t cs_c   = cs_c0;

	// -------------------------------------------------------------------------

	begin_asm()

	//vzeroall()                         // zero all xmm/ymm registers.
	
	mov(var(a), r12)                   // load address of a.
	//mov(var(rs_a), r8)                 // load rs_a
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
	

	mov(var(c), r10)                   // load address of c
	//mov(var(rs_c), rdi)                // load rs_c
	//lea(mem(, rdi, 8), rdi)            // rs_c *= sizeof(double)



	// r10 = rcx = c
	// r12 = rax = a
	// r14 = rbx = b
	// r9  = m dim index ii
	// r15 = n dim index jj

	mov(imm(0), r15)                   // jj = 0;

	label(.DLOOP3X4J)                  // LOOP OVER jj = [ 0 1 ... ]



	vzeroall()                         // zero all xmm/ymm registers.

	lea(mem(   , r15, 1), rsi)         // rsi = r15 = 4*jj;
	imul(imm(1*8), rsi)                // rsi *= cs_c = 1*8
	lea(mem(r10, rsi, 1), rcx)         // rcx = c + 4*jj*cs_c;

	lea(mem(   , r15, 1), rsi)         // rsi = r15 = 4*jj;
	imul(r11, rsi)                     // rsi *= cs_b;
	lea(mem(r14, rsi, 1), rbx)         // rbx = b + 4*jj*cs_b;

	lea(mem(   , r12, 1), rax)         // rax = a;

	prefetch(0, mem(rcx, 7*8))         // prefetch c + 0*rs_c

	
	
	
	mov(var(k_iter16), rsi)            // i = k_iter16;
	test(rsi, rsi)                     // check i via logical AND.
	je(.DCONSIDKITER4)                 // if i == 0, jump to code that
	                                   // contains the k_iter4 loop.
	
	
	label(.DLOOPKITER16)               // MAIN LOOP
	
	
	// ---------------------------------- iteration 0

	vmovupd(mem(rax       ), ymm0)
	add(imm(4*8), rax)                 // a += 4*cs_b = 4*8;

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

	vmovupd(mem(rax       ), ymm0)
	add(imm(4*8), rax)                 // a += 4*cs_b = 4*8;

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
	
	vmovupd(mem(rax       ), ymm0)
	add(imm(4*8), rax)                 // a += 4*cs_b = 4*8;

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

	vmovupd(mem(rax       ), ymm0)
	add(imm(4*8), rax)                 // a += 4*cs_b = 4*8;

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
	add(imm(4*8), rax)                 // a += 4*cs_b = 4*8;
	
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
	add(imm(1*8), rax)                 // a += 1*cs_b = 1*8;
	
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
	vaddpd( xmm0, xmm1, xmm0 )         // xmm0[0] = sum(ymm4); xmm0[1] = sum(ymm7)

	vhaddpd( ymm13, ymm10, ymm2 )
	vextractf128(imm(1), ymm2, xmm1 )
	vaddpd( xmm2, xmm1, xmm2 )         // xmm2[0] = sum(ymm10); xmm2[1] = sum(ymm13)

	vperm2f128(imm(0x20), ymm2, ymm0, ymm4 )

	                                   // ymm4 = sum(ymm4) sum(ymm7) sum(ymm10) sum(ymm13)


	
	mov(var(alpha), rax)               // load address of alpha
	mov(var(beta), rbx)                // load address of beta
	vbroadcastsd(mem(rax), ymm0)       // load alpha and duplicate
	vbroadcastsd(mem(rbx), ymm3)       // load beta and duplicate
	
	vmulpd(ymm0, ymm4, ymm4)           // scale by alpha
	
	
	
	
	
	
	mov(var(cs_c), rsi)                // load cs_c
	lea(mem(, rsi, 8), rsi)            // rsi = cs_c * sizeof(double)
	
	
	
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
	
	


	add(imm(4), r15)                   // jj += 4;
	cmp(imm(4), r15)                   // compare jj to 4
	jle(.DLOOP3X4J)                    // if jj <= 4, jump to beginning
	                                   // of jj loop; otherwise, loop ends.



	label(.DRETURN)

	

    end_asm(
	: // output operands (none)
	: // input operands
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
	  "rax", "rbx", "rcx", "rdx", "rsi", "rdi", 
	  "r8", "r9", "r10", "r11", "r12", "r13", "r14", "r15",
	  "xmm0", "xmm1", "xmm2", "xmm3",
	  "xmm4", "xmm5", "xmm6", "xmm7",
	  "xmm8", "xmm9", "xmm10", "xmm11",
	  "xmm12", "xmm13", "xmm14", "xmm15",
	  "ymm0", "ymm2", "ymm3", "ymm4", "ymm7", "ymm10", "ymm13", "memory"
	)
}

void bli_dgemmsup_rd_haswell_asm_6x4
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
#if 0
	bli_dgemmsup_r_haswell_ref
	(
	  conja, conjb, m0, n0, k0,
	  alpha, a, rs_a0, cs_a0, b, rs_b0, cs_b0,
	  beta, c, rs_c0, cs_c0, data, cntx
	);
	return;
#endif
	//void*    a_next = bli_auxinfo_next_a( data );
	//void*    b_next = bli_auxinfo_next_b( data );

	// Typecast local copies of integers in case dim_t and inc_t are a
	// different size than is expected by load instructions.
	uint64_t k_iter16 = k0 / 16;
	uint64_t k_left16 = k0 % 16;
	uint64_t k_iter4  = k_left16 / 4;
	uint64_t k_left1  = k_left16 % 4;

	uint64_t rs_a   = rs_a0;
	uint64_t cs_a   = cs_a0;
	uint64_t rs_b   = rs_b0;
	uint64_t cs_b   = cs_b0;
	uint64_t rs_c   = rs_c0;
	uint64_t cs_c   = cs_c0;

	// -------------------------------------------------------------------------

	begin_asm()

	//vzeroall()                         // zero all xmm/ymm registers.
	
	mov(var(a), r12)                   // load address of a.
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
	

	mov(var(c), r10)                   // load address of c
	mov(var(rs_c), rdi)                // load rs_c
	lea(mem(, rdi, 8), rdi)            // rs_c *= sizeof(double)



	// r10 = rcx = c
	// r12 = rax = a
	// r14 = rbx = b
	// r9  = m dim index ii
	// r15 = n dim index jj

	mov(imm(0), r9)                    // ii = 0;

	label(.DLOOP3X4I)                  // LOOP OVER ii = [ 0 1 ... ]



	vzeroall()                         // zero all xmm/ymm registers.

	lea(mem(   , r9,  1), rsi)         // rsi = r9 = 3*ii;
	imul(rdi, rsi)                     // rsi *= rs_c;
	lea(mem(r10, rsi, 1), rcx)         // rcx = c + 3*ii*rs_c;

	lea(mem(   , r9,  1), rsi)         // rsi = r9 = 3*ii;
	imul(r8,  rsi)                     // rsi *= rs_a;
	lea(mem(r12, rsi, 1), rax)         // rax = a + 3*ii*rs_a;

	lea(mem(   , r14, 1), rbx)         // rbx = b;

	prefetch(0, mem(rcx, 7*8))         // prefetch c + 0*rs_c
	prefetch(0, mem(rcx, rdi, 1, 7*8)) // prefetch c + 1*rs_c
	prefetch(0, mem(rcx, rdi, 2, 7*8)) // prefetch c + 2*rs_c
	

	
	
	mov(var(k_iter16), rsi)            // i = k_iter16;
	test(rsi, rsi)                     // check i via logical AND.
	je(.DCONSIDKITER4)                 // if i == 0, jump to code that
	                                   // contains the k_iter4 loop.
	
	
	label(.DLOOPKITER16)               // MAIN LOOP
	
	
	// ---------------------------------- iteration 0

	vmovupd(mem(rax       ), ymm0)
	vmovupd(mem(rax, r8, 1), ymm1)
	vmovupd(mem(rax, r8, 2), ymm2)
	add(imm(4*8), rax)                 // a += 4*cs_b = 4*8;

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

	vmovupd(mem(rax       ), ymm0)
	vmovupd(mem(rax, r8, 1), ymm1)
	vmovupd(mem(rax, r8, 2), ymm2)
	add(imm(4*8), rax)                 // a += 4*cs_b = 4*8;

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
	
	vmovupd(mem(rax       ), ymm0)
	vmovupd(mem(rax, r8, 1), ymm1)
	vmovupd(mem(rax, r8, 2), ymm2)
	add(imm(4*8), rax)                 // a += 4*cs_b = 4*8;

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

	vmovupd(mem(rax       ), ymm0)
	vmovupd(mem(rax, r8, 1), ymm1)
	vmovupd(mem(rax, r8, 2), ymm2)
	add(imm(4*8), rax)                 // a += 4*cs_b = 4*8;

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
	add(imm(4*8), rax)                 // a += 4*cs_b = 4*8;
	
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
	add(imm(1*8), rax)                 // a += 1*cs_b = 1*8;
	
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
	vaddpd( xmm0, xmm1, xmm0 )         // xmm0[0] = sum(ymm4); xmm0[1] = sum(ymm7)

	vhaddpd( ymm13, ymm10, ymm2 )
	vextractf128(imm(1), ymm2, xmm1 )
	vaddpd( xmm2, xmm1, xmm2 )         // xmm2[0] = sum(ymm10); xmm2[1] = sum(ymm13)

	vperm2f128(imm(0x20), ymm2, ymm0, ymm4 )


	vhaddpd( ymm8, ymm5, ymm0 )
	vextractf128(imm(1), ymm0, xmm1 )
	vaddpd( xmm0, xmm1, xmm0 )

	vhaddpd( ymm14, ymm11, ymm2 )
	vextractf128(imm(1), ymm2, xmm1 )
	vaddpd( xmm2, xmm1, xmm2 )

	vperm2f128(imm(0x20), ymm2, ymm0, ymm5 )


	vhaddpd( ymm9, ymm6, ymm0 )
	vextractf128(imm(1), ymm0, xmm1 )
	vaddpd( xmm0, xmm1, xmm0 )

	vhaddpd( ymm15, ymm12, ymm2 )
	vextractf128(imm(1), ymm2, xmm1 )
	vaddpd( xmm2, xmm1, xmm2 )

	vperm2f128(imm(0x20), ymm2, ymm0, ymm6 )

	                                   // ymm4 = sum(ymm4) sum(ymm7) sum(ymm10) sum(ymm13)
	                                   // ymm5 = sum(ymm5) sum(ymm8) sum(ymm11) sum(ymm14)
	                                   // ymm6 = sum(ymm6) sum(ymm9) sum(ymm12) sum(ymm15)


	
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
	
	


	add(imm(3), r9)                    // ii += 3;
	cmp(imm(3), r9)                    // compare ii to 3
	jle(.DLOOP3X4I)                    // if ii <= 3, jump to beginning
	                                   // of ii loop; otherwise, loop ends.




	label(.DRETURN)

	

    end_asm(
	: // output operands (none)
	: // input operands
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
	  "rax", "rbx", "rcx", "rdx", "rsi", "rdi", 
	  "r8", "r9", "r10", "r11", "r12", "r13", "r14", "r15",
	  "xmm0", "xmm1", "xmm2", "xmm3",
	  "xmm4", "xmm5", "xmm6", "xmm7",
	  "xmm8", "xmm9", "xmm10", "xmm11",
	  "xmm12", "xmm13", "xmm14", "xmm15",
	  "ymm0", "ymm1", "ymm2", "ymm3", "ymm4", "ymm5", "ymm6", "ymm7",
	  "ymm8", "ymm9", "ymm10", "ymm11", "ymm12", "ymm13", "ymm14",
	  "ymm15", "memory"
	)
}

void bli_dgemmsup_rd_haswell_asm_3x4
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
#if 0
	bli_dgemmsup_r_haswell_ref
	(
	  conja, conjb, m0, n0, k0,
	  alpha, a, rs_a0, cs_a0, b, rs_b0, cs_b0,
	  beta, c, rs_c0, cs_c0, data, cntx
	);
	return;
#endif
	//void*    a_next = bli_auxinfo_next_a( data );
	//void*    b_next = bli_auxinfo_next_b( data );

	// Typecast local copies of integers in case dim_t and inc_t are a
	// different size than is expected by load instructions.
	uint64_t k_iter16 = k0 / 16;
	uint64_t k_left16 = k0 % 16;
	uint64_t k_iter4  = k_left16 / 4;
	uint64_t k_left1  = k_left16 % 4;

	uint64_t rs_a   = rs_a0;
	uint64_t cs_a   = cs_a0;
	uint64_t rs_b   = rs_b0;
	uint64_t cs_b   = cs_b0;
	uint64_t rs_c   = rs_c0;
	uint64_t cs_c   = cs_c0;

	// -------------------------------------------------------------------------

	begin_asm()
	
	vzeroall()                         // zero all xmm/ymm registers.
	
	mov(var(a), rax)                   // load address of a.
	mov(var(rs_a), r8)                 // load rs_a
	//mov(var(cs_a), r9)                 // load cs_a
	lea(mem(, r8, 8), r8)              // rs_a *= sizeof(double)
	//lea(mem(, r9, 8), r9)              // cs_a *= sizeof(double)

	//lea(mem(r8, r8, 2), r13)           // r13 = 3*rs_a
	//lea(mem(r8, r8, 4), r15)           // r15 = 5*rs_a

	mov(var(b), rbx)                   // load address of b.
	//mov(var(rs_b), r10)                // load rs_b
	mov(var(cs_b), r11)                // load cs_b
	//lea(mem(, r10, 8), r10)            // rs_b *= sizeof(double)
	lea(mem(, r11, 8), r11)            // cs_b *= sizeof(double)

	lea(mem(r11, r11, 2), r13)         // r13 = 3*cs_b
	
	                                   // initialize loop by pre-loading
	                                   // a column of a.

	mov(var(c), rcx)                   // load address of c
	mov(var(rs_c), rdi)                // load rs_c
	lea(mem(, rdi, 8), rdi)            // rs_c *= sizeof(double)

	prefetch(0, mem(rcx, 7*8))         // prefetch c + 0*rs_c
	prefetch(0, mem(rcx, rdi, 1, 7*8)) // prefetch c + 1*rs_c
	prefetch(0, mem(rcx, rdi, 2, 7*8)) // prefetch c + 2*rs_c
	
	
	
	mov(var(k_iter16), rsi)            // i = k_iter16;
	test(rsi, rsi)                     // check i via logical AND.
	je(.DCONSIDKITER4)                 // if i == 0, jump to code that
	                                   // contains the k_iter4 loop.
	
	
	label(.DLOOPKITER16)               // MAIN LOOP
	
	
	// ---------------------------------- iteration 0

	vmovupd(mem(rax       ), ymm0)
	vmovupd(mem(rax, r8, 1), ymm1)
	vmovupd(mem(rax, r8, 2), ymm2)
	add(imm(4*8), rax)                 // a += 4*cs_b = 4*8;

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

	vmovupd(mem(rax       ), ymm0)
	vmovupd(mem(rax, r8, 1), ymm1)
	vmovupd(mem(rax, r8, 2), ymm2)
	add(imm(4*8), rax)                 // a += 4*cs_b = 4*8;

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
	
	vmovupd(mem(rax       ), ymm0)
	vmovupd(mem(rax, r8, 1), ymm1)
	vmovupd(mem(rax, r8, 2), ymm2)
	add(imm(4*8), rax)                 // a += 4*cs_b = 4*8;

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

	vmovupd(mem(rax       ), ymm0)
	vmovupd(mem(rax, r8, 1), ymm1)
	vmovupd(mem(rax, r8, 2), ymm2)
	add(imm(4*8), rax)                 // a += 4*cs_b = 4*8;

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
	add(imm(4*8), rax)                 // a += 4*cs_b = 4*8;
	
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
	add(imm(1*8), rax)                 // a += 1*cs_b = 1*8;
	
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
	vaddpd( xmm0, xmm1, xmm0 )         // xmm0[0] = sum(ymm4); xmm0[1] = sum(ymm7)

	vhaddpd( ymm13, ymm10, ymm2 )
	vextractf128(imm(1), ymm2, xmm1 )
	vaddpd( xmm2, xmm1, xmm2 )         // xmm2[0] = sum(ymm10); xmm2[1] = sum(ymm13)

	vperm2f128(imm(0x20), ymm2, ymm0, ymm4 )


	vhaddpd( ymm8, ymm5, ymm0 )
	vextractf128(imm(1), ymm0, xmm1 )
	vaddpd( xmm0, xmm1, xmm0 )

	vhaddpd( ymm14, ymm11, ymm2 )
	vextractf128(imm(1), ymm2, xmm1 )
	vaddpd( xmm2, xmm1, xmm2 )

	vperm2f128(imm(0x20), ymm2, ymm0, ymm5 )


	vhaddpd( ymm9, ymm6, ymm0 )
	vextractf128(imm(1), ymm0, xmm1 )
	vaddpd( xmm0, xmm1, xmm0 )

	vhaddpd( ymm15, ymm12, ymm2 )
	vextractf128(imm(1), ymm2, xmm1 )
	vaddpd( xmm2, xmm1, xmm2 )

	vperm2f128(imm(0x20), ymm2, ymm0, ymm6 )

	                                   // ymm4 = sum(ymm4) sum(ymm7) sum(ymm10) sum(ymm13)
	                                   // ymm5 = sum(ymm5) sum(ymm8) sum(ymm11) sum(ymm14)
	                                   // ymm6 = sum(ymm6) sum(ymm9) sum(ymm12) sum(ymm15)


	
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
	
	

    end_asm(
	: // output operands (none)
	: // input operands
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
	  "rax", "rbx", "rcx", "rdx", "rsi", "rdi", 
	  "r8", "r9", "r10", "r11", "r12", "r13", "r14", "r15",
	  "xmm0", "xmm1", "xmm2", "xmm3",
	  "xmm4", "xmm5", "xmm6", "xmm7",
	  "xmm8", "xmm9", "xmm10", "xmm11",
	  "xmm12", "xmm13", "xmm14", "xmm15",
	  "ymm0", "ymm1", "ymm2", "ymm3", "ymm4", "ymm5", "ymm6", "ymm7",
	  "ymm8", "ymm9", "ymm10", "ymm11", "ymm12", "ymm13", "ymm14",
	  "ymm15", "memory"
	)
}

void bli_dgemmsup_rd_haswell_asm_2x4
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
#if 0
	bli_dgemmsup_r_haswell_ref
	(
	  conja, conjb, m0, n0, k0,
	  alpha, a, rs_a0, cs_a0, b, rs_b0, cs_b0,
	  beta, c, rs_c0, cs_c0, data, cntx
	);
	return;
#endif
	//void*    a_next = bli_auxinfo_next_a( data );
	//void*    b_next = bli_auxinfo_next_b( data );

	// Typecast local copies of integers in case dim_t and inc_t are a
	// different size than is expected by load instructions.
	uint64_t k_iter16 = k0 / 16;
	uint64_t k_left16 = k0 % 16;
	uint64_t k_iter4  = k_left16 / 4;
	uint64_t k_left1  = k_left16 % 4;

	uint64_t rs_a   = rs_a0;
	uint64_t cs_a   = cs_a0;
	uint64_t rs_b   = rs_b0;
	uint64_t cs_b   = cs_b0;
	uint64_t rs_c   = rs_c0;
	uint64_t cs_c   = cs_c0;

	// -------------------------------------------------------------------------

	begin_asm()
	
	vzeroall()                         // zero all xmm/ymm registers.
	
	mov(var(a), rax)                   // load address of a.
	mov(var(rs_a), r8)                 // load rs_a
	//mov(var(cs_a), r9)                 // load cs_a
	lea(mem(, r8, 8), r8)              // rs_a *= sizeof(double)
	//lea(mem(, r9, 8), r9)              // cs_a *= sizeof(double)

	//lea(mem(r8, r8, 2), r13)           // r13 = 3*rs_a
	//lea(mem(r8, r8, 4), r15)           // r15 = 5*rs_a

	mov(var(b), rbx)                   // load address of b.
	//mov(var(rs_b), r10)                // load rs_b
	mov(var(cs_b), r11)                // load cs_b
	//lea(mem(, r10, 8), r10)            // rs_b *= sizeof(double)
	lea(mem(, r11, 8), r11)            // cs_b *= sizeof(double)

	lea(mem(r11, r11, 2), r13)         // r13 = 3*cs_b
	
	                                   // initialize loop by pre-loading
	                                   // a column of a.

	mov(var(c), rcx)                   // load address of c
	mov(var(rs_c), rdi)                // load rs_c
	lea(mem(, rdi, 8), rdi)            // rs_c *= sizeof(double)

	prefetch(0, mem(rcx, 7*8))         // prefetch c + 0*rs_c
	prefetch(0, mem(rcx, rdi, 1, 7*8)) // prefetch c + 1*rs_c
	
	
	
	mov(var(k_iter16), rsi)            // i = k_iter16;
	test(rsi, rsi)                     // check i via logical AND.
	je(.DCONSIDKITER4)                 // if i == 0, jump to code that
	                                   // contains the k_iter4 loop.
	
	
	label(.DLOOPKITER16)               // MAIN LOOP
	
	
	// ---------------------------------- iteration 0

	vmovupd(mem(rax       ), ymm0)
	vmovupd(mem(rax, r8, 1), ymm1)
	add(imm(4*8), rax)                 // a += 4*cs_b = 4*8;

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

	vmovupd(mem(rax       ), ymm0)
	vmovupd(mem(rax, r8, 1), ymm1)
	add(imm(4*8), rax)                 // a += 4*cs_b = 4*8;

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
	
	vmovupd(mem(rax       ), ymm0)
	vmovupd(mem(rax, r8, 1), ymm1)
	add(imm(4*8), rax)                 // a += 4*cs_b = 4*8;

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

	vmovupd(mem(rax       ), ymm0)
	vmovupd(mem(rax, r8, 1), ymm1)
	add(imm(4*8), rax)                 // a += 4*cs_b = 4*8;

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
	add(imm(4*8), rax)                 // a += 4*cs_b = 4*8;
	
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
	add(imm(1*8), rax)                 // a += 1*cs_b = 1*8;
	
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
	vaddpd( xmm0, xmm1, xmm0 )         // xmm0[0] = sum(ymm4); xmm0[1] = sum(ymm7)

	vhaddpd( ymm13, ymm10, ymm2 )
	vextractf128(imm(1), ymm2, xmm1 )
	vaddpd( xmm2, xmm1, xmm2 )         // xmm2[0] = sum(ymm10); xmm2[1] = sum(ymm13)

	vperm2f128(imm(0x20), ymm2, ymm0, ymm4 )


	vhaddpd( ymm8, ymm5, ymm0 )
	vextractf128(imm(1), ymm0, xmm1 )
	vaddpd( xmm0, xmm1, xmm0 )

	vhaddpd( ymm14, ymm11, ymm2 )
	vextractf128(imm(1), ymm2, xmm1 )
	vaddpd( xmm2, xmm1, xmm2 )

	vperm2f128(imm(0x20), ymm2, ymm0, ymm5 )

	                                   // ymm4 = sum(ymm4) sum(ymm7) sum(ymm10) sum(ymm13)
	                                   // ymm5 = sum(ymm5) sum(ymm8) sum(ymm11) sum(ymm14)


	
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
	
	

    end_asm(
	: // output operands (none)
	: // input operands
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
	  "rax", "rbx", "rcx", "rdx", "rsi", "rdi", 
	  "r8", "r9", "r10", "r11", "r12", "r13", "r14", "r15",
	  "xmm0", "xmm1", "xmm2", "xmm3",
	  "xmm4", "xmm5", "xmm6", "xmm7",
	  "xmm8", "xmm9", "xmm10", "xmm11",
	  "xmm12", "xmm13", "xmm14", "xmm15",
	  "ymm0", "ymm1", "ymm2", "ymm3", "ymm4", "ymm5", "ymm7", "ymm8",
	  "ymm10", "ymm11", "ymm13", "ymm14", "memory"
	)
}

void bli_dgemmsup_rd_haswell_asm_1x4
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
#if 0
	bli_dgemmsup_r_haswell_ref
	(
	  conja, conjb, m0, n0, k0,
	  alpha, a, rs_a0, cs_a0, b, rs_b0, cs_b0,
	  beta, c, rs_c0, cs_c0, data, cntx
	);
	return;
#endif
	//void*    a_next = bli_auxinfo_next_a( data );
	//void*    b_next = bli_auxinfo_next_b( data );

	// Typecast local copies of integers in case dim_t and inc_t are a
	// different size than is expected by load instructions.
	uint64_t k_iter16 = k0 / 16;
	uint64_t k_left16 = k0 % 16;
	uint64_t k_iter4  = k_left16 / 4;
	uint64_t k_left1  = k_left16 % 4;

	uint64_t rs_a   = rs_a0;
	uint64_t cs_a   = cs_a0;
	uint64_t rs_b   = rs_b0;
	uint64_t cs_b   = cs_b0;
	uint64_t rs_c   = rs_c0;
	uint64_t cs_c   = cs_c0;

/*
   rrc:
     --------        -- -- --        | | | |
     --------        -- -- -- ...    | | | |
     --------   +=   -- -- --        | | | |
     --------                        | | | |
     --------                           :
     --------                           :
*/
	// -------------------------------------------------------------------------

	begin_asm()
	
	vzeroall()                         // zero all xmm/ymm registers.
	
	mov(var(a), rax)                   // load address of a.
	mov(var(rs_a), r8)                 // load rs_a
	//mov(var(cs_a), r9)                 // load cs_a
	lea(mem(, r8, 8), r8)              // rs_a *= sizeof(double)
	//lea(mem(, r9, 8), r9)              // cs_a *= sizeof(double)

	//lea(mem(r8, r8, 2), r13)           // r13 = 3*rs_a
	//lea(mem(r8, r8, 4), r15)           // r15 = 5*rs_a

	mov(var(b), rbx)                   // load address of b.
	//mov(var(rs_b), r10)                // load rs_b
	mov(var(cs_b), r11)                // load cs_b
	//lea(mem(, r10, 8), r10)            // rs_b *= sizeof(double)
	lea(mem(, r11, 8), r11)            // cs_b *= sizeof(double)

	lea(mem(r11, r11, 2), r13)         // r13 = 3*cs_b
	
	                                   // initialize loop by pre-loading
	                                   // a column of a.

	mov(var(c), rcx)                   // load address of c
	//mov(var(rs_c), rdi)                // load rs_c
	//lea(mem(, rdi, 8), rdi)            // rs_c *= sizeof(double)

	prefetch(0, mem(rcx, 7*8))         // prefetch c + 0*rs_c
	
	
	
	mov(var(k_iter16), rsi)            // i = k_iter16;
	test(rsi, rsi)                     // check i via logical AND.
	je(.DCONSIDKITER4)                 // if i == 0, jump to code that
	                                   // contains the k_iter4 loop.
	
	
	label(.DLOOPKITER16)               // MAIN LOOP
	
	
	// ---------------------------------- iteration 0

	vmovupd(mem(rax       ), ymm0)
	add(imm(4*8), rax)                 // a += 4*cs_b = 4*8;

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

	vmovupd(mem(rax       ), ymm0)
	add(imm(4*8), rax)                 // a += 4*cs_b = 4*8;

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
	
	vmovupd(mem(rax       ), ymm0)
	add(imm(4*8), rax)                 // a += 4*cs_b = 4*8;

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

	vmovupd(mem(rax       ), ymm0)
	add(imm(4*8), rax)                 // a += 4*cs_b = 4*8;

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
	add(imm(4*8), rax)                 // a += 4*cs_b = 4*8;
	
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
	add(imm(1*8), rax)                 // a += 1*cs_b = 1*8;
	
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
	vaddpd( xmm0, xmm1, xmm0 )         // xmm0[0] = sum(ymm4); xmm0[1] = sum(ymm7)

	vhaddpd( ymm13, ymm10, ymm2 )
	vextractf128(imm(1), ymm2, xmm1 )
	vaddpd( xmm2, xmm1, xmm2 )         // xmm2[0] = sum(ymm10); xmm2[1] = sum(ymm13)

	vperm2f128(imm(0x20), ymm2, ymm0, ymm4 )

	                                   // ymm4 = sum(ymm4) sum(ymm7) sum(ymm10) sum(ymm13)


	
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
	
	

    end_asm(
	: // output operands (none)
	: // input operands
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
	  "rax", "rbx", "rcx", "rdx", "rsi", "rdi", 
	  "r8", "r9", "r10", "r11", "r12", "r13", "r14", "r15",
	  "xmm0", "xmm1", "xmm2", "xmm3",
	  "xmm4", "xmm5", "xmm6", "xmm7",
	  "xmm8", "xmm9", "xmm10", "xmm11",
	  "xmm12", "xmm13", "xmm14", "xmm15",
	  "ymm0", "ymm2", "ymm3", "ymm4", "ymm7", "ymm10", "ymm13", "memory"
	)
}

void bli_dgemmsup_rd_haswell_asm_6x2
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
#if 0
	bli_dgemmsup_r_haswell_ref
	(
	  conja, conjb, m0, n0, k0,
	  alpha, a, rs_a0, cs_a0, b, rs_b0, cs_b0,
	  beta, c, rs_c0, cs_c0, data, cntx
	);
	return;
#endif
	//void*    a_next = bli_auxinfo_next_a( data );
	//void*    b_next = bli_auxinfo_next_b( data );

	// Typecast local copies of integers in case dim_t and inc_t are a
	// different size than is expected by load instructions.
	uint64_t k_iter16 = k0 / 16;
	uint64_t k_left16 = k0 % 16;
	uint64_t k_iter4  = k_left16 / 4;
	uint64_t k_left1  = k_left16 % 4;

	uint64_t rs_a   = rs_a0;
	uint64_t cs_a   = cs_a0;
	uint64_t rs_b   = rs_b0;
	uint64_t cs_b   = cs_b0;
	uint64_t rs_c   = rs_c0;
	uint64_t cs_c   = cs_c0;

/*
   rrc:
     --------        -- -- --        | |
     --------        -- -- -- ...    | |
     --------   +=   -- -- --        | |
     --------        -- -- --        | |
     --------        -- -- --         :
     --------        -- -- --         :
*/
	// -------------------------------------------------------------------------

	begin_asm()
	
	vzeroall()                         // zero all xmm/ymm registers.
	
	mov(var(a), rax)                   // load address of a.
	mov(var(rs_a), r8)                 // load rs_a
	//mov(var(cs_a), r9)                 // load cs_a
	lea(mem(, r8, 8), r8)              // rs_a *= sizeof(double)
	//lea(mem(, r9, 8), r9)              // cs_a *= sizeof(double)

	lea(mem(r8, r8, 2), r13)           // r13 = 3*rs_a
	lea(mem(r8, r8, 4), r15)           // r15 = 5*rs_a

	mov(var(b), rbx)                   // load address of b.
	//mov(var(rs_b), r10)                // load rs_b
	mov(var(cs_b), r11)                // load cs_b
	//lea(mem(, r10, 8), r10)            // rs_b *= sizeof(double)
	lea(mem(, r11, 8), r11)            // cs_b *= sizeof(double)

	//lea(mem(r11, r11, 2), r13)         // r13 = 3*cs_b
	
	                                   // initialize loop by pre-loading
	                                   // a column of a.

	mov(var(c), rcx)                   // load address of c
	mov(var(rs_c), rdi)                // load rs_c
	lea(mem(, rdi, 8), rdi)            // rs_c *= sizeof(double)

	lea(mem(rcx, rdi, 2), rdx)         //
	lea(mem(rdx, rdi, 1), rdx)         // rdx = c + 3*rs_c;
	prefetch(0, mem(rcx, 7*8))         // prefetch c + 0*rs_c
	prefetch(0, mem(rcx, rdi, 1, 7*8)) // prefetch c + 1*rs_c
	prefetch(0, mem(rcx, rdi, 2, 7*8)) // prefetch c + 2*rs_c
	prefetch(0, mem(rdx, 7*8))         // prefetch c + 3*rs_c
	prefetch(0, mem(rdx, rdi, 1, 7*8)) // prefetch c + 4*rs_c
	prefetch(0, mem(rdx, rdi, 2, 7*8)) // prefetch c + 5*rs_c
	

	
	
	mov(var(k_iter16), rsi)            // i = k_iter16;
	test(rsi, rsi)                     // check i via logical AND.
	je(.DCONSIDKITER4)                 // if i == 0, jump to code that
	                                   // contains the k_iter4 loop.
	
	
	label(.DLOOPKITER16)               // MAIN LOOP
	
	
	// ---------------------------------- iteration 0

	vmovupd(mem(rbx        ), ymm0)
	vmovupd(mem(rbx, r11, 1), ymm1)
	add(imm(4*8), rbx)                 // b += 4*rs_b = 4*8;

	vmovupd(mem(rax        ), ymm3)
	vfmadd231pd(ymm0, ymm3, ymm4)
	vfmadd231pd(ymm1, ymm3, ymm5)

	vmovupd(mem(rax, r8,  1), ymm3)
	vfmadd231pd(ymm0, ymm3, ymm6)
	vfmadd231pd(ymm1, ymm3, ymm7)

	vmovupd(mem(rax, r8,  2), ymm3)
	vfmadd231pd(ymm0, ymm3, ymm8)
	vfmadd231pd(ymm1, ymm3, ymm9)

	vmovupd(mem(rax, r13, 1), ymm3)
	vfmadd231pd(ymm0, ymm3, ymm10)
	vfmadd231pd(ymm1, ymm3, ymm11)

	vmovupd(mem(rax, r8,  4), ymm3)
	vfmadd231pd(ymm0, ymm3, ymm12)
	vfmadd231pd(ymm1, ymm3, ymm13)

	vmovupd(mem(rax, r15, 1), ymm3)
	add(imm(4*8), rax)                 // a += 4*cs_a = 4*8;
	vfmadd231pd(ymm0, ymm3, ymm14)
	vfmadd231pd(ymm1, ymm3, ymm15)

	
	// ---------------------------------- iteration 1

	vmovupd(mem(rbx        ), ymm0)
	vmovupd(mem(rbx, r11, 1), ymm1)
	add(imm(4*8), rbx)                 // b += 4*rs_b = 4*8;

	vmovupd(mem(rax        ), ymm3)
	vfmadd231pd(ymm0, ymm3, ymm4)
	vfmadd231pd(ymm1, ymm3, ymm5)

	vmovupd(mem(rax, r8,  1), ymm3)
	vfmadd231pd(ymm0, ymm3, ymm6)
	vfmadd231pd(ymm1, ymm3, ymm7)

	vmovupd(mem(rax, r8,  2), ymm3)
	vfmadd231pd(ymm0, ymm3, ymm8)
	vfmadd231pd(ymm1, ymm3, ymm9)

	vmovupd(mem(rax, r13, 1), ymm3)
	vfmadd231pd(ymm0, ymm3, ymm10)
	vfmadd231pd(ymm1, ymm3, ymm11)

	vmovupd(mem(rax, r8,  4), ymm3)
	vfmadd231pd(ymm0, ymm3, ymm12)
	vfmadd231pd(ymm1, ymm3, ymm13)

	vmovupd(mem(rax, r15, 1), ymm3)
	add(imm(4*8), rax)                 // a += 4*cs_a = 4*8;
	vfmadd231pd(ymm0, ymm3, ymm14)
	vfmadd231pd(ymm1, ymm3, ymm15)


	// ---------------------------------- iteration 2
	
	vmovupd(mem(rbx        ), ymm0)
	vmovupd(mem(rbx, r11, 1), ymm1)
	add(imm(4*8), rbx)                 // b += 4*rs_b = 4*8;

	vmovupd(mem(rax        ), ymm3)
	vfmadd231pd(ymm0, ymm3, ymm4)
	vfmadd231pd(ymm1, ymm3, ymm5)

	vmovupd(mem(rax, r8,  1), ymm3)
	vfmadd231pd(ymm0, ymm3, ymm6)
	vfmadd231pd(ymm1, ymm3, ymm7)

	vmovupd(mem(rax, r8,  2), ymm3)
	vfmadd231pd(ymm0, ymm3, ymm8)
	vfmadd231pd(ymm1, ymm3, ymm9)

	vmovupd(mem(rax, r13, 1), ymm3)
	vfmadd231pd(ymm0, ymm3, ymm10)
	vfmadd231pd(ymm1, ymm3, ymm11)

	vmovupd(mem(rax, r8,  4), ymm3)
	vfmadd231pd(ymm0, ymm3, ymm12)
	vfmadd231pd(ymm1, ymm3, ymm13)

	vmovupd(mem(rax, r15, 1), ymm3)
	add(imm(4*8), rax)                 // a += 4*cs_a = 4*8;
	vfmadd231pd(ymm0, ymm3, ymm14)
	vfmadd231pd(ymm1, ymm3, ymm15)


	// ---------------------------------- iteration 3

	vmovupd(mem(rbx        ), ymm0)
	vmovupd(mem(rbx, r11, 1), ymm1)
	add(imm(4*8), rbx)                 // b += 4*rs_b = 4*8;

	vmovupd(mem(rax        ), ymm3)
	vfmadd231pd(ymm0, ymm3, ymm4)
	vfmadd231pd(ymm1, ymm3, ymm5)

	vmovupd(mem(rax, r8,  1), ymm3)
	vfmadd231pd(ymm0, ymm3, ymm6)
	vfmadd231pd(ymm1, ymm3, ymm7)

	vmovupd(mem(rax, r8,  2), ymm3)
	vfmadd231pd(ymm0, ymm3, ymm8)
	vfmadd231pd(ymm1, ymm3, ymm9)

	vmovupd(mem(rax, r13, 1), ymm3)
	vfmadd231pd(ymm0, ymm3, ymm10)
	vfmadd231pd(ymm1, ymm3, ymm11)

	vmovupd(mem(rax, r8,  4), ymm3)
	vfmadd231pd(ymm0, ymm3, ymm12)
	vfmadd231pd(ymm1, ymm3, ymm13)

	vmovupd(mem(rax, r15, 1), ymm3)
	add(imm(4*8), rax)                 // a += 4*cs_a = 4*8;
	vfmadd231pd(ymm0, ymm3, ymm14)
	vfmadd231pd(ymm1, ymm3, ymm15)

	

	dec(rsi)                           // i -= 1;
	jne(.DLOOPKITER16)                 // iterate again if i != 0.
	
	
	
	
	
	
	label(.DCONSIDKITER4)
	
	mov(var(k_iter4), rsi)             // i = k_iter4;
	test(rsi, rsi)                     // check i via logical AND.
	je(.DCONSIDKLEFT1)                 // if i == 0, jump to code that
	                                   // considers k_left1 loop.
	                                   // else, we prepare to enter k_iter4 loop.
	
	
	label(.DLOOPKITER4)                // EDGE LOOP (ymm)
	
	vmovupd(mem(rbx        ), ymm0)
	vmovupd(mem(rbx, r11, 1), ymm1)
	add(imm(4*8), rbx)                 // b += 4*rs_b = 4*8;

	vmovupd(mem(rax        ), ymm3)
	vfmadd231pd(ymm0, ymm3, ymm4)
	vfmadd231pd(ymm1, ymm3, ymm5)

	vmovupd(mem(rax, r8,  1), ymm3)
	vfmadd231pd(ymm0, ymm3, ymm6)
	vfmadd231pd(ymm1, ymm3, ymm7)

	vmovupd(mem(rax, r8,  2), ymm3)
	vfmadd231pd(ymm0, ymm3, ymm8)
	vfmadd231pd(ymm1, ymm3, ymm9)

	vmovupd(mem(rax, r13, 1), ymm3)
	vfmadd231pd(ymm0, ymm3, ymm10)
	vfmadd231pd(ymm1, ymm3, ymm11)

	vmovupd(mem(rax, r8,  4), ymm3)
	vfmadd231pd(ymm0, ymm3, ymm12)
	vfmadd231pd(ymm1, ymm3, ymm13)

	vmovupd(mem(rax, r15, 1), ymm3)
	add(imm(4*8), rax)                 // a += 4*cs_a = 4*8;
	vfmadd231pd(ymm0, ymm3, ymm14)
	vfmadd231pd(ymm1, ymm3, ymm15)

	
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
	
	vmovsd(mem(rbx        ), xmm0)
	vmovsd(mem(rbx, r11, 1), xmm1)
	add(imm(1*8), rbx)                 // b += 1*rs_b = 1*8;

	vmovsd(mem(rax        ), xmm3)
	vfmadd231pd(ymm0, ymm3, ymm4)
	vfmadd231pd(ymm1, ymm3, ymm5)

	vmovsd(mem(rax, r8,  1), xmm3)
	vfmadd231pd(ymm0, ymm3, ymm6)
	vfmadd231pd(ymm1, ymm3, ymm7)

	vmovsd(mem(rax, r8,  2), xmm3)
	vfmadd231pd(ymm0, ymm3, ymm8)
	vfmadd231pd(ymm1, ymm3, ymm9)

	vmovsd(mem(rax, r13, 1), xmm3)
	vfmadd231pd(ymm0, ymm3, ymm10)
	vfmadd231pd(ymm1, ymm3, ymm11)

	vmovsd(mem(rax, r8,  4), xmm3)
	vfmadd231pd(ymm0, ymm3, ymm12)
	vfmadd231pd(ymm1, ymm3, ymm13)

	vmovsd(mem(rax, r15, 1), xmm3)
	add(imm(1*8), rax)                 // a += 1*cs_a = 1*8;
	vfmadd231pd(ymm0, ymm3, ymm14)
	vfmadd231pd(ymm1, ymm3, ymm15)

	
	dec(rsi)                           // i -= 1;
	jne(.DLOOPKLEFT1)                  // iterate again if i != 0.
	
	
	




	label(.DPOSTACCUM)

	                                   // ymm4  ymm5
	                                   // ymm6  ymm7
	                                   // ymm8  ymm9
	                                   // ymm10 ymm11
	                                   // ymm12 ymm13
	                                   // ymm14 ymm15
	
	vhaddpd( ymm5, ymm4, ymm0 )
	vextractf128(imm(1), ymm0, xmm1 )
	vaddpd( xmm0, xmm1, xmm4 )         // xmm0[0] = sum(ymm4); xmm0[1] = sum(ymm5)

	vhaddpd( ymm7, ymm6, ymm0 )
	vextractf128(imm(1), ymm0, xmm1 )
	vaddpd( xmm0, xmm1, xmm6 )

	vhaddpd( ymm9, ymm8, ymm0 )
	vextractf128(imm(1), ymm0, xmm1 )
	vaddpd( xmm0, xmm1, xmm8 )

	vhaddpd( ymm11, ymm10, ymm0 )
	vextractf128(imm(1), ymm0, xmm1 )
	vaddpd( xmm0, xmm1, xmm10 )

	vhaddpd( ymm13, ymm12, ymm0 )
	vextractf128(imm(1), ymm0, xmm1 )
	vaddpd( xmm0, xmm1, xmm12 )

	vhaddpd( ymm15, ymm14, ymm0 )
	vextractf128(imm(1), ymm0, xmm1 )
	vaddpd( xmm0, xmm1, xmm14 )

	                                   // xmm4  = sum(ymm4)  sum(ymm5)
	                                   // xmm6  = sum(ymm6)  sum(ymm7)
	                                   // xmm8  = sum(ymm8)  sum(ymm9)
	                                   // xmm10 = sum(ymm10) sum(ymm11)
	                                   // xmm12 = sum(ymm12) sum(ymm13)
	                                   // xmm14 = sum(ymm14) sum(ymm15)


	
	mov(var(alpha), rax)               // load address of alpha
	mov(var(beta), rbx)                // load address of beta
	vbroadcastsd(mem(rax), ymm0)       // load alpha and duplicate
	vbroadcastsd(mem(rbx), ymm3)       // load beta and duplicate
	
	vmulpd(xmm0, xmm4,  xmm4)          // scale by alpha
	vmulpd(xmm0, xmm6,  xmm6)
	vmulpd(xmm0, xmm8,  xmm8)
	vmulpd(xmm0, xmm10, xmm10)
	vmulpd(xmm0, xmm12, xmm12)
	vmulpd(xmm0, xmm14, xmm14)
	
	
	
	
	
	
	//mov(var(cs_c), rsi)                // load cs_c
	//lea(mem(, rsi, 8), rsi)            // rsi = cs_c * sizeof(double)
	
	
	
	                                   // now avoid loading C if beta == 0
	
	vxorpd(ymm0, ymm0, ymm0)           // set ymm0 to zero.
	vucomisd(xmm0, xmm3)               // set ZF if beta == 0.
	je(.DBETAZERO)                     // if ZF = 1, jump to beta == 0 case
	

	
	label(.DROWSTORED)
	
	
	vfmadd231pd(mem(rcx), xmm3, xmm4)
	vmovupd(xmm4, mem(rcx))
	add(rdi, rcx)
	
	vfmadd231pd(mem(rcx), xmm3, xmm6)
	vmovupd(xmm6, mem(rcx))
	add(rdi, rcx)
	
	vfmadd231pd(mem(rcx), xmm3, xmm8)
	vmovupd(xmm8, mem(rcx))
	add(rdi, rcx)
	
	vfmadd231pd(mem(rcx), xmm3, xmm10)
	vmovupd(xmm10, mem(rcx))
	add(rdi, rcx)
	
	vfmadd231pd(mem(rcx), xmm3, xmm12)
	vmovupd(xmm12, mem(rcx))
	add(rdi, rcx)
	
	vfmadd231pd(mem(rcx), xmm3, xmm14)
	vmovupd(xmm14, mem(rcx))
	//add(rdi, rcx)
	
	
	
	jmp(.DDONE)                        // jump to end.
	
	
	
	
	label(.DBETAZERO)
	

	
	label(.DROWSTORBZ)
	
	
	vmovupd(xmm4, mem(rcx))
	add(rdi, rcx)
	
	vmovupd(xmm6, mem(rcx))
	add(rdi, rcx)
	
	vmovupd(xmm8, mem(rcx))
	add(rdi, rcx)
	
	vmovupd(xmm10, mem(rcx))
	add(rdi, rcx)
	
	vmovupd(xmm12, mem(rcx))
	add(rdi, rcx)
	
	vmovupd(xmm14, mem(rcx))
	//add(rdi, rcx)
	
	
	
	
	label(.DDONE)
	
	

    end_asm(
	: // output operands (none)
	: // input operands
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
	  "rax", "rbx", "rcx", "rdx", "rsi", "rdi", 
	  "r8", "r9", "r10", "r11", "r12", "r13", "r14", "r15",
	  "xmm0", "xmm1", "xmm2", "xmm3",
	  "xmm4", "xmm5", "xmm6", "xmm7",
	  "xmm8", "xmm9", "xmm10", "xmm11",
	  "xmm12", "xmm13", "xmm14", "xmm15",
	  "ymm0", "ymm1", "ymm3", "ymm4", "ymm5", "ymm6", "ymm7", "ymm8",
	  "ymm9", "ymm10", "ymm11", "ymm12", "ymm13", "ymm14", "ymm15", "memory"
	)
}

void bli_dgemmsup_rd_haswell_asm_3x2
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
#if 0
	bli_dgemmsup_r_haswell_ref
	(
	  conja, conjb, m0, n0, k0,
	  alpha, a, rs_a0, cs_a0, b, rs_b0, cs_b0,
	  beta, c, rs_c0, cs_c0, data, cntx
	);
	return;
#endif
	//void*    a_next = bli_auxinfo_next_a( data );
	//void*    b_next = bli_auxinfo_next_b( data );

	// Typecast local copies of integers in case dim_t and inc_t are a
	// different size than is expected by load instructions.
	uint64_t k_iter16 = k0 / 16;
	uint64_t k_left16 = k0 % 16;
	uint64_t k_iter4  = k_left16 / 4;
	uint64_t k_left1  = k_left16 % 4;

	uint64_t rs_a   = rs_a0;
	uint64_t cs_a   = cs_a0;
	uint64_t rs_b   = rs_b0;
	uint64_t cs_b   = cs_b0;
	uint64_t rs_c   = rs_c0;
	uint64_t cs_c   = cs_c0;

/*
   rrc:
     --------        -- -- --        | |
     --------        -- -- -- ...    | |
     --------   +=   -- -- --        | |
     --------        -- -- --        | |
     --------        -- -- --         :
     --------        -- -- --         :
*/
	// -------------------------------------------------------------------------

	begin_asm()
	
	vzeroall()                         // zero all xmm/ymm registers.
	
	mov(var(a), rax)                   // load address of a.
	mov(var(rs_a), r8)                 // load rs_a
	//mov(var(cs_a), r9)                 // load cs_a
	lea(mem(, r8, 8), r8)              // rs_a *= sizeof(double)
	//lea(mem(, r9, 8), r9)              // cs_a *= sizeof(double)

	//lea(mem(r8, r8, 2), r13)           // r13 = 3*rs_a
	//lea(mem(r8, r8, 4), r15)           // r15 = 5*rs_a

	mov(var(b), rbx)                   // load address of b.
	//mov(var(rs_b), r10)                // load rs_b
	mov(var(cs_b), r11)                // load cs_b
	//lea(mem(, r10, 8), r10)            // rs_b *= sizeof(double)
	lea(mem(, r11, 8), r11)            // cs_b *= sizeof(double)

	//lea(mem(r11, r11, 2), r13)         // r13 = 3*cs_b
	
	                                   // initialize loop by pre-loading
	                                   // a column of a.

	mov(var(c), rcx)                   // load address of c
	mov(var(rs_c), rdi)                // load rs_c
	lea(mem(, rdi, 8), rdi)            // rs_c *= sizeof(double)

	//lea(mem(rcx, rdi, 2), rdx)         //
	//lea(mem(rdx, rdi, 1), rdx)         // rdx = c + 3*rs_c;
	prefetch(0, mem(rcx, 7*8))         // prefetch c + 0*rs_c
	prefetch(0, mem(rcx, rdi, 1, 7*8)) // prefetch c + 1*rs_c
	prefetch(0, mem(rcx, rdi, 2, 7*8)) // prefetch c + 2*rs_c
	

	
	
	mov(var(k_iter16), rsi)            // i = k_iter16;
	test(rsi, rsi)                     // check i via logical AND.
	je(.DCONSIDKITER4)                 // if i == 0, jump to code that
	                                   // contains the k_iter4 loop.
	
	
	label(.DLOOPKITER16)               // MAIN LOOP
	
	
	// ---------------------------------- iteration 0

	vmovupd(mem(rbx        ), ymm0)
	vmovupd(mem(rbx, r11, 1), ymm1)
	add(imm(4*8), rbx)                 // b += 4*rs_b = 4*8;

	vmovupd(mem(rax        ), ymm3)
	vfmadd231pd(ymm0, ymm3, ymm4)
	vfmadd231pd(ymm1, ymm3, ymm5)

	vmovupd(mem(rax, r8,  1), ymm3)
	vfmadd231pd(ymm0, ymm3, ymm6)
	vfmadd231pd(ymm1, ymm3, ymm7)

	vmovupd(mem(rax, r8,  2), ymm3)
	add(imm(4*8), rax)                 // a += 4*cs_a = 4*8;
	vfmadd231pd(ymm0, ymm3, ymm8)
	vfmadd231pd(ymm1, ymm3, ymm9)

	
	// ---------------------------------- iteration 1

	vmovupd(mem(rbx        ), ymm0)
	vmovupd(mem(rbx, r11, 1), ymm1)
	add(imm(4*8), rbx)                 // b += 4*rs_b = 4*8;

	vmovupd(mem(rax        ), ymm3)
	vfmadd231pd(ymm0, ymm3, ymm4)
	vfmadd231pd(ymm1, ymm3, ymm5)

	vmovupd(mem(rax, r8,  1), ymm3)
	vfmadd231pd(ymm0, ymm3, ymm6)
	vfmadd231pd(ymm1, ymm3, ymm7)

	vmovupd(mem(rax, r8,  2), ymm3)
	add(imm(4*8), rax)                 // a += 4*cs_a = 4*8;
	vfmadd231pd(ymm0, ymm3, ymm8)
	vfmadd231pd(ymm1, ymm3, ymm9)


	// ---------------------------------- iteration 2
	
	vmovupd(mem(rbx        ), ymm0)
	vmovupd(mem(rbx, r11, 1), ymm1)
	add(imm(4*8), rbx)                 // b += 4*rs_b = 4*8;

	vmovupd(mem(rax        ), ymm3)
	vfmadd231pd(ymm0, ymm3, ymm4)
	vfmadd231pd(ymm1, ymm3, ymm5)

	vmovupd(mem(rax, r8,  1), ymm3)
	vfmadd231pd(ymm0, ymm3, ymm6)
	vfmadd231pd(ymm1, ymm3, ymm7)

	vmovupd(mem(rax, r8,  2), ymm3)
	add(imm(4*8), rax)                 // a += 4*cs_a = 4*8;
	vfmadd231pd(ymm0, ymm3, ymm8)
	vfmadd231pd(ymm1, ymm3, ymm9)


	// ---------------------------------- iteration 3

	vmovupd(mem(rbx        ), ymm0)
	vmovupd(mem(rbx, r11, 1), ymm1)
	add(imm(4*8), rbx)                 // b += 4*rs_b = 4*8;

	vmovupd(mem(rax        ), ymm3)
	vfmadd231pd(ymm0, ymm3, ymm4)
	vfmadd231pd(ymm1, ymm3, ymm5)

	vmovupd(mem(rax, r8,  1), ymm3)
	vfmadd231pd(ymm0, ymm3, ymm6)
	vfmadd231pd(ymm1, ymm3, ymm7)

	vmovupd(mem(rax, r8,  2), ymm3)
	add(imm(4*8), rax)                 // a += 4*cs_a = 4*8;
	vfmadd231pd(ymm0, ymm3, ymm8)
	vfmadd231pd(ymm1, ymm3, ymm9)

	

	dec(rsi)                           // i -= 1;
	jne(.DLOOPKITER16)                 // iterate again if i != 0.
	
	
	
	
	
	
	label(.DCONSIDKITER4)
	
	mov(var(k_iter4), rsi)             // i = k_iter4;
	test(rsi, rsi)                     // check i via logical AND.
	je(.DCONSIDKLEFT1)                 // if i == 0, jump to code that
	                                   // considers k_left1 loop.
	                                   // else, we prepare to enter k_iter4 loop.
	
	
	label(.DLOOPKITER4)                // EDGE LOOP (ymm)
	
	vmovupd(mem(rbx        ), ymm0)
	vmovupd(mem(rbx, r11, 1), ymm1)
	add(imm(4*8), rbx)                 // b += 4*rs_b = 4*8;

	vmovupd(mem(rax        ), ymm3)
	vfmadd231pd(ymm0, ymm3, ymm4)
	vfmadd231pd(ymm1, ymm3, ymm5)

	vmovupd(mem(rax, r8,  1), ymm3)
	vfmadd231pd(ymm0, ymm3, ymm6)
	vfmadd231pd(ymm1, ymm3, ymm7)

	vmovupd(mem(rax, r8,  2), ymm3)
	add(imm(4*8), rax)                 // a += 4*cs_a = 4*8;
	vfmadd231pd(ymm0, ymm3, ymm8)
	vfmadd231pd(ymm1, ymm3, ymm9)

	
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
	
	vmovsd(mem(rbx        ), xmm0)
	vmovsd(mem(rbx, r11, 1), xmm1)
	add(imm(1*8), rbx)                 // b += 1*rs_b = 1*8;

	vmovsd(mem(rax        ), xmm3)
	vfmadd231pd(ymm0, ymm3, ymm4)
	vfmadd231pd(ymm1, ymm3, ymm5)

	vmovsd(mem(rax, r8,  1), xmm3)
	vfmadd231pd(ymm0, ymm3, ymm6)
	vfmadd231pd(ymm1, ymm3, ymm7)

	vmovsd(mem(rax, r8,  2), xmm3)
	add(imm(1*8), rax)                 // a += 1*cs_a = 1*8;
	vfmadd231pd(ymm0, ymm3, ymm8)
	vfmadd231pd(ymm1, ymm3, ymm9)

	
	dec(rsi)                           // i -= 1;
	jne(.DLOOPKLEFT1)                  // iterate again if i != 0.
	
	
	




	label(.DPOSTACCUM)

	                                   // ymm4  ymm5
	                                   // ymm6  ymm7
	                                   // ymm8  ymm9
	
	vhaddpd( ymm5, ymm4, ymm0 )
	vextractf128(imm(1), ymm0, xmm1 )
	vaddpd( xmm0, xmm1, xmm4 )         // xmm0[0] = sum(ymm4); xmm0[1] = sum(ymm5)

	vhaddpd( ymm7, ymm6, ymm0 )
	vextractf128(imm(1), ymm0, xmm1 )
	vaddpd( xmm0, xmm1, xmm6 )

	vhaddpd( ymm9, ymm8, ymm0 )
	vextractf128(imm(1), ymm0, xmm1 )
	vaddpd( xmm0, xmm1, xmm8 )

	                                   // xmm4  = sum(ymm4)  sum(ymm5)
	                                   // xmm6  = sum(ymm6)  sum(ymm7)
	                                   // xmm8  = sum(ymm8)  sum(ymm9)


	
	mov(var(alpha), rax)               // load address of alpha
	mov(var(beta), rbx)                // load address of beta
	vbroadcastsd(mem(rax), ymm0)       // load alpha and duplicate
	vbroadcastsd(mem(rbx), ymm3)       // load beta and duplicate
	
	vmulpd(xmm0, xmm4,  xmm4)          // scale by alpha
	vmulpd(xmm0, xmm6,  xmm6)
	vmulpd(xmm0, xmm8,  xmm8)
	
	
	
	
	
	
	//mov(var(cs_c), rsi)                // load cs_c
	//lea(mem(, rsi, 8), rsi)            // rsi = cs_c * sizeof(double)
	
	
	
	                                   // now avoid loading C if beta == 0
	
	vxorpd(ymm0, ymm0, ymm0)           // set ymm0 to zero.
	vucomisd(xmm0, xmm3)               // set ZF if beta == 0.
	je(.DBETAZERO)                     // if ZF = 1, jump to beta == 0 case
	

	
	label(.DROWSTORED)
	
	
	vfmadd231pd(mem(rcx), xmm3, xmm4)
	vmovupd(xmm4, mem(rcx))
	add(rdi, rcx)
	
	vfmadd231pd(mem(rcx), xmm3, xmm6)
	vmovupd(xmm6, mem(rcx))
	add(rdi, rcx)
	
	vfmadd231pd(mem(rcx), xmm3, xmm8)
	vmovupd(xmm8, mem(rcx))
	//add(rdi, rcx)
	
	
	
	jmp(.DDONE)                        // jump to end.
	
	
	
	
	label(.DBETAZERO)
	

	
	label(.DROWSTORBZ)
	
	
	vmovupd(xmm4, mem(rcx))
	add(rdi, rcx)
	
	vmovupd(xmm6, mem(rcx))
	add(rdi, rcx)
	
	vmovupd(xmm8, mem(rcx))
	//add(rdi, rcx)
	
	
	
	
	label(.DDONE)
	
	

    end_asm(
	: // output operands (none)
	: // input operands
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
	  "rax", "rbx", "rcx", "rdx", "rsi", "rdi", 
	  "r8", "r9", "r10", "r11", "r12", "r13", "r14", "r15",
	  "xmm0", "xmm1", "xmm2", "xmm3",
	  "xmm4", "xmm5", "xmm6", "xmm7",
	  "xmm8", "xmm9", "xmm10", "xmm11",
	  "xmm12", "xmm13", "xmm14", "xmm15",
	  "ymm0", "ymm1", "ymm3", "ymm4", "ymm5", "ymm6", "ymm7", "ymm8",
	  "ymm9", "memory"
	)
}

void bli_dgemmsup_rd_haswell_asm_2x2
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
#if 0
	bli_dgemmsup_r_haswell_ref
	(
	  conja, conjb, m0, n0, k0,
	  alpha, a, rs_a0, cs_a0, b, rs_b0, cs_b0,
	  beta, c, rs_c0, cs_c0, data, cntx
	);
	return;
#endif
	//void*    a_next = bli_auxinfo_next_a( data );
	//void*    b_next = bli_auxinfo_next_b( data );

	// Typecast local copies of integers in case dim_t and inc_t are a
	// different size than is expected by load instructions.
	uint64_t k_iter16 = k0 / 16;
	uint64_t k_left16 = k0 % 16;
	uint64_t k_iter4  = k_left16 / 4;
	uint64_t k_left1  = k_left16 % 4;

	uint64_t rs_a   = rs_a0;
	uint64_t cs_a   = cs_a0;
	uint64_t rs_b   = rs_b0;
	uint64_t cs_b   = cs_b0;
	uint64_t rs_c   = rs_c0;
	uint64_t cs_c   = cs_c0;

/*
   rrc:
     --------        -- -- --        | |
     --------        -- -- -- ...    | |
     --------   +=   -- -- --        | |
     --------        -- -- --        | |
     --------        -- -- --         :
     --------        -- -- --         :
*/
	// -------------------------------------------------------------------------

	begin_asm()
	
	vzeroall()                         // zero all xmm/ymm registers.
	
	mov(var(a), rax)                   // load address of a.
	mov(var(rs_a), r8)                 // load rs_a
	//mov(var(cs_a), r9)                 // load cs_a
	lea(mem(, r8, 8), r8)              // rs_a *= sizeof(double)
	//lea(mem(, r9, 8), r9)              // cs_a *= sizeof(double)

	//lea(mem(r8, r8, 2), r13)           // r13 = 3*rs_a
	//lea(mem(r8, r8, 4), r15)           // r15 = 5*rs_a

	mov(var(b), rbx)                   // load address of b.
	//mov(var(rs_b), r10)                // load rs_b
	mov(var(cs_b), r11)                // load cs_b
	//lea(mem(, r10, 8), r10)            // rs_b *= sizeof(double)
	lea(mem(, r11, 8), r11)            // cs_b *= sizeof(double)

	//lea(mem(r11, r11, 2), r13)         // r13 = 3*cs_b
	
	                                   // initialize loop by pre-loading
	                                   // a column of a.

	mov(var(c), rcx)                   // load address of c
	mov(var(rs_c), rdi)                // load rs_c
	lea(mem(, rdi, 8), rdi)            // rs_c *= sizeof(double)

	//lea(mem(rcx, rdi, 2), rdx)         //
	//lea(mem(rdx, rdi, 1), rdx)         // rdx = c + 3*rs_c;
	prefetch(0, mem(rcx, 7*8))         // prefetch c + 0*rs_c
	prefetch(0, mem(rcx, rdi, 1, 7*8)) // prefetch c + 1*rs_c
	

	
	
	mov(var(k_iter16), rsi)            // i = k_iter16;
	test(rsi, rsi)                     // check i via logical AND.
	je(.DCONSIDKITER4)                 // if i == 0, jump to code that
	                                   // contains the k_iter4 loop.
	
	
	label(.DLOOPKITER16)               // MAIN LOOP
	
	
	// ---------------------------------- iteration 0

	vmovupd(mem(rbx        ), ymm0)
	vmovupd(mem(rbx, r11, 1), ymm1)
	add(imm(4*8), rbx)                 // b += 4*rs_b = 4*8;

	vmovupd(mem(rax        ), ymm3)
	vfmadd231pd(ymm0, ymm3, ymm4)
	vfmadd231pd(ymm1, ymm3, ymm5)

	vmovupd(mem(rax, r8,  1), ymm3)
	add(imm(4*8), rax)                 // a += 4*cs_a = 4*8;
	vfmadd231pd(ymm0, ymm3, ymm6)
	vfmadd231pd(ymm1, ymm3, ymm7)

	
	// ---------------------------------- iteration 1

	vmovupd(mem(rbx        ), ymm0)
	vmovupd(mem(rbx, r11, 1), ymm1)
	add(imm(4*8), rbx)                 // b += 4*rs_b = 4*8;

	vmovupd(mem(rax        ), ymm3)
	vfmadd231pd(ymm0, ymm3, ymm4)
	vfmadd231pd(ymm1, ymm3, ymm5)

	vmovupd(mem(rax, r8,  1), ymm3)
	add(imm(4*8), rax)                 // a += 4*cs_a = 4*8;
	vfmadd231pd(ymm0, ymm3, ymm6)
	vfmadd231pd(ymm1, ymm3, ymm7)


	// ---------------------------------- iteration 2
	
	vmovupd(mem(rbx        ), ymm0)
	vmovupd(mem(rbx, r11, 1), ymm1)
	add(imm(4*8), rbx)                 // b += 4*rs_b = 4*8;

	vmovupd(mem(rax        ), ymm3)
	vfmadd231pd(ymm0, ymm3, ymm4)
	vfmadd231pd(ymm1, ymm3, ymm5)

	vmovupd(mem(rax, r8,  1), ymm3)
	add(imm(4*8), rax)                 // a += 4*cs_a = 4*8;
	vfmadd231pd(ymm0, ymm3, ymm6)
	vfmadd231pd(ymm1, ymm3, ymm7)


	// ---------------------------------- iteration 3

	vmovupd(mem(rbx        ), ymm0)
	vmovupd(mem(rbx, r11, 1), ymm1)
	add(imm(4*8), rbx)                 // b += 4*rs_b = 4*8;

	vmovupd(mem(rax        ), ymm3)
	vfmadd231pd(ymm0, ymm3, ymm4)
	vfmadd231pd(ymm1, ymm3, ymm5)

	vmovupd(mem(rax, r8,  1), ymm3)
	add(imm(4*8), rax)                 // a += 4*cs_a = 4*8;
	vfmadd231pd(ymm0, ymm3, ymm6)
	vfmadd231pd(ymm1, ymm3, ymm7)

	

	dec(rsi)                           // i -= 1;
	jne(.DLOOPKITER16)                 // iterate again if i != 0.
	
	
	
	
	
	
	label(.DCONSIDKITER4)
	
	mov(var(k_iter4), rsi)             // i = k_iter4;
	test(rsi, rsi)                     // check i via logical AND.
	je(.DCONSIDKLEFT1)                 // if i == 0, jump to code that
	                                   // considers k_left1 loop.
	                                   // else, we prepare to enter k_iter4 loop.
	
	
	label(.DLOOPKITER4)                // EDGE LOOP (ymm)
	
	vmovupd(mem(rbx        ), ymm0)
	vmovupd(mem(rbx, r11, 1), ymm1)
	add(imm(4*8), rbx)                 // b += 4*rs_b = 4*8;

	vmovupd(mem(rax        ), ymm3)
	vfmadd231pd(ymm0, ymm3, ymm4)
	vfmadd231pd(ymm1, ymm3, ymm5)

	vmovupd(mem(rax, r8,  1), ymm3)
	add(imm(4*8), rax)                 // a += 4*cs_a = 4*8;
	vfmadd231pd(ymm0, ymm3, ymm6)
	vfmadd231pd(ymm1, ymm3, ymm7)

	
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
	
	vmovsd(mem(rbx        ), xmm0)
	vmovsd(mem(rbx, r11, 1), xmm1)
	add(imm(1*8), rbx)                 // b += 1*rs_b = 1*8;

	vmovsd(mem(rax        ), xmm3)
	vfmadd231pd(ymm0, ymm3, ymm4)
	vfmadd231pd(ymm1, ymm3, ymm5)

	vmovsd(mem(rax, r8,  1), xmm3)
	add(imm(1*8), rax)                 // a += 1*cs_a = 1*8;
	vfmadd231pd(ymm0, ymm3, ymm6)
	vfmadd231pd(ymm1, ymm3, ymm7)

	
	dec(rsi)                           // i -= 1;
	jne(.DLOOPKLEFT1)                  // iterate again if i != 0.
	
	
	




	label(.DPOSTACCUM)

	                                   // ymm4  ymm5
	                                   // ymm6  ymm7
	
	vhaddpd( ymm5, ymm4, ymm0 )
	vextractf128(imm(1), ymm0, xmm1 )
	vaddpd( xmm0, xmm1, xmm4 )         // xmm0[0] = sum(ymm4); xmm0[1] = sum(ymm5)

	vhaddpd( ymm7, ymm6, ymm0 )
	vextractf128(imm(1), ymm0, xmm1 )
	vaddpd( xmm0, xmm1, xmm6 )

	                                   // xmm4  = sum(ymm4)  sum(ymm5)
	                                   // xmm6  = sum(ymm6)  sum(ymm7)


	
	mov(var(alpha), rax)               // load address of alpha
	mov(var(beta), rbx)                // load address of beta
	vbroadcastsd(mem(rax), ymm0)       // load alpha and duplicate
	vbroadcastsd(mem(rbx), ymm3)       // load beta and duplicate
	
	vmulpd(xmm0, xmm4,  xmm4)          // scale by alpha
	vmulpd(xmm0, xmm6,  xmm6)
	
	
	
	
	
	
	//mov(var(cs_c), rsi)                // load cs_c
	//lea(mem(, rsi, 8), rsi)            // rsi = cs_c * sizeof(double)
	
	
	
	                                   // now avoid loading C if beta == 0
	
	vxorpd(ymm0, ymm0, ymm0)           // set ymm0 to zero.
	vucomisd(xmm0, xmm3)               // set ZF if beta == 0.
	je(.DBETAZERO)                     // if ZF = 1, jump to beta == 0 case
	

	
	label(.DROWSTORED)
	
	
	vfmadd231pd(mem(rcx), xmm3, xmm4)
	vmovupd(xmm4, mem(rcx))
	add(rdi, rcx)
	
	vfmadd231pd(mem(rcx), xmm3, xmm6)
	vmovupd(xmm6, mem(rcx))
	//add(rdi, rcx)
	
	
	
	jmp(.DDONE)                        // jump to end.
	
	
	
	
	label(.DBETAZERO)
	

	
	label(.DROWSTORBZ)
	
	
	vmovupd(xmm4, mem(rcx))
	add(rdi, rcx)
	
	vmovupd(xmm6, mem(rcx))
	//add(rdi, rcx)
	
	
	
	
	label(.DDONE)
	
	

    end_asm(
	: // output operands (none)
	: // input operands
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
	  "rax", "rbx", "rcx", "rdx", "rsi", "rdi", 
	  "r8", "r9", "r10", "r11", "r12", "r13", "r14", "r15",
	  "xmm0", "xmm1", "xmm2", "xmm3",
	  "xmm4", "xmm5", "xmm6", "xmm7",
	  "xmm8", "xmm9", "xmm10", "xmm11",
	  "xmm12", "xmm13", "xmm14", "xmm15",
	  "ymm0", "ymm1", "ymm3", "ymm4", "ymm5", "ymm6", "ymm7", "memory"
	)
}

void bli_dgemmsup_rd_haswell_asm_1x2
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
#if 0
	bli_dgemmsup_r_haswell_ref
	(
	  conja, conjb, m0, n0, k0,
	  alpha, a, rs_a0, cs_a0, b, rs_b0, cs_b0,
	  beta, c, rs_c0, cs_c0, data, cntx
	);
	return;
#endif
	//void*    a_next = bli_auxinfo_next_a( data );
	//void*    b_next = bli_auxinfo_next_b( data );

	// Typecast local copies of integers in case dim_t and inc_t are a
	// different size than is expected by load instructions.
	uint64_t k_iter16 = k0 / 16;
	uint64_t k_left16 = k0 % 16;
	uint64_t k_iter4  = k_left16 / 4;
	uint64_t k_left1  = k_left16 % 4;

	uint64_t rs_a   = rs_a0;
	uint64_t cs_a   = cs_a0;
	uint64_t rs_b   = rs_b0;
	uint64_t cs_b   = cs_b0;
	uint64_t rs_c   = rs_c0;
	uint64_t cs_c   = cs_c0;

/*
   rrc:
     --------        -- -- --        | |
     --------        -- -- -- ...    | |
     --------   +=   -- -- --        | |
     --------        -- -- --        | |
     --------        -- -- --         :
     --------        -- -- --         :
*/
	// -------------------------------------------------------------------------

	begin_asm()
	
	vzeroall()                         // zero all xmm/ymm registers.
	
	mov(var(a), rax)                   // load address of a.
	//mov(var(rs_a), r8)                 // load rs_a
	//mov(var(cs_a), r9)                 // load cs_a
	//lea(mem(, r8, 8), r8)              // rs_a *= sizeof(double)
	//lea(mem(, r9, 8), r9)              // cs_a *= sizeof(double)

	//lea(mem(r8, r8, 2), r13)           // r13 = 3*rs_a
	//lea(mem(r8, r8, 4), r15)           // r15 = 5*rs_a

	mov(var(b), rbx)                   // load address of b.
	//mov(var(rs_b), r10)                // load rs_b
	mov(var(cs_b), r11)                // load cs_b
	//lea(mem(, r10, 8), r10)            // rs_b *= sizeof(double)
	lea(mem(, r11, 8), r11)            // cs_b *= sizeof(double)

	//lea(mem(r11, r11, 2), r13)         // r13 = 3*cs_b
	
	                                   // initialize loop by pre-loading
	                                   // a column of a.

	mov(var(c), rcx)                   // load address of c
	//mov(var(rs_c), rdi)                // load rs_c
	//lea(mem(, rdi, 8), rdi)            // rs_c *= sizeof(double)

	//lea(mem(rcx, rdi, 2), rdx)         //
	//lea(mem(rdx, rdi, 1), rdx)         // rdx = c + 3*rs_c;
	prefetch(0, mem(rcx, 7*8))         // prefetch c + 0*rs_c
	

	
	
	mov(var(k_iter16), rsi)            // i = k_iter16;
	test(rsi, rsi)                     // check i via logical AND.
	je(.DCONSIDKITER4)                 // if i == 0, jump to code that
	                                   // contains the k_iter4 loop.
	
	
	label(.DLOOPKITER16)               // MAIN LOOP
	
	
	// ---------------------------------- iteration 0

	vmovupd(mem(rbx        ), ymm0)
	vmovupd(mem(rbx, r11, 1), ymm1)
	add(imm(4*8), rbx)                 // b += 4*rs_b = 4*8;

	vmovupd(mem(rax        ), ymm3)
	add(imm(4*8), rax)                 // a += 4*cs_a = 4*8;
	vfmadd231pd(ymm0, ymm3, ymm4)
	vfmadd231pd(ymm1, ymm3, ymm5)

	
	// ---------------------------------- iteration 1

	vmovupd(mem(rbx        ), ymm0)
	vmovupd(mem(rbx, r11, 1), ymm1)
	add(imm(4*8), rbx)                 // b += 4*rs_b = 4*8;

	vmovupd(mem(rax        ), ymm3)
	add(imm(4*8), rax)                 // a += 4*cs_a = 4*8;
	vfmadd231pd(ymm0, ymm3, ymm4)
	vfmadd231pd(ymm1, ymm3, ymm5)


	// ---------------------------------- iteration 2
	
	vmovupd(mem(rbx        ), ymm0)
	vmovupd(mem(rbx, r11, 1), ymm1)
	add(imm(4*8), rbx)                 // b += 4*rs_b = 4*8;

	vmovupd(mem(rax        ), ymm3)
	add(imm(4*8), rax)                 // a += 4*cs_a = 4*8;
	vfmadd231pd(ymm0, ymm3, ymm4)
	vfmadd231pd(ymm1, ymm3, ymm5)


	// ---------------------------------- iteration 3

	vmovupd(mem(rbx        ), ymm0)
	vmovupd(mem(rbx, r11, 1), ymm1)
	add(imm(4*8), rbx)                 // b += 4*rs_b = 4*8;

	vmovupd(mem(rax        ), ymm3)
	add(imm(4*8), rax)                 // a += 4*cs_a = 4*8;
	vfmadd231pd(ymm0, ymm3, ymm4)
	vfmadd231pd(ymm1, ymm3, ymm5)

	

	dec(rsi)                           // i -= 1;
	jne(.DLOOPKITER16)                 // iterate again if i != 0.
	
	
	
	
	
	
	label(.DCONSIDKITER4)
	
	mov(var(k_iter4), rsi)             // i = k_iter4;
	test(rsi, rsi)                     // check i via logical AND.
	je(.DCONSIDKLEFT1)                 // if i == 0, jump to code that
	                                   // considers k_left1 loop.
	                                   // else, we prepare to enter k_iter4 loop.
	
	
	label(.DLOOPKITER4)                // EDGE LOOP (ymm)
	
	vmovupd(mem(rbx        ), ymm0)
	vmovupd(mem(rbx, r11, 1), ymm1)
	add(imm(4*8), rbx)                 // b += 4*rs_b = 4*8;

	vmovupd(mem(rax        ), ymm3)
	add(imm(4*8), rax)                 // a += 4*cs_a = 4*8;
	vfmadd231pd(ymm0, ymm3, ymm4)
	vfmadd231pd(ymm1, ymm3, ymm5)

	
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
	
	vmovsd(mem(rbx        ), xmm0)
	vmovsd(mem(rbx, r11, 1), xmm1)
	add(imm(1*8), rbx)                 // b += 1*rs_b = 1*8;

	vmovsd(mem(rax        ), xmm3)
	add(imm(1*8), rax)                 // a += 1*cs_a = 1*8;
	vfmadd231pd(ymm0, ymm3, ymm4)
	vfmadd231pd(ymm1, ymm3, ymm5)

	
	dec(rsi)                           // i -= 1;
	jne(.DLOOPKLEFT1)                  // iterate again if i != 0.
	
	
	




	label(.DPOSTACCUM)

	                                   // ymm4  ymm5
	
	vhaddpd( ymm5, ymm4, ymm0 )
	vextractf128(imm(1), ymm0, xmm1 )
	vaddpd( xmm0, xmm1, xmm4 )         // xmm0[0] = sum(ymm4); xmm0[1] = sum(ymm5)

	                                   // xmm4  = sum(ymm4)  sum(ymm5)


	
	mov(var(alpha), rax)               // load address of alpha
	mov(var(beta), rbx)                // load address of beta
	vbroadcastsd(mem(rax), ymm0)       // load alpha and duplicate
	vbroadcastsd(mem(rbx), ymm3)       // load beta and duplicate
	
	vmulpd(xmm0, xmm4,  xmm4)          // scale by alpha
	
	
	
	
	
	
	//mov(var(cs_c), rsi)                // load cs_c
	//lea(mem(, rsi, 8), rsi)            // rsi = cs_c * sizeof(double)
	
	
	
	                                   // now avoid loading C if beta == 0
	
	vxorpd(ymm0, ymm0, ymm0)           // set ymm0 to zero.
	vucomisd(xmm0, xmm3)               // set ZF if beta == 0.
	je(.DBETAZERO)                     // if ZF = 1, jump to beta == 0 case
	

	
	label(.DROWSTORED)
	
	
	vfmadd231pd(mem(rcx), xmm3, xmm4)
	vmovupd(xmm4, mem(rcx))
	//add(rdi, rcx)
	
	
	
	jmp(.DDONE)                        // jump to end.
	
	
	
	
	label(.DBETAZERO)
	

	
	label(.DROWSTORBZ)
	
	
	vmovupd(xmm4, mem(rcx))
	//add(rdi, rcx)
	
	
	
	
	label(.DDONE)
	
	

    end_asm(
	: // output operands (none)
	: // input operands
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
	  "rax", "rbx", "rcx", "rdx", "rsi", "rdi", 
	  "r8", "r9", "r10", "r11", "r12", "r13", "r14", "r15",
	  "xmm0", "xmm1", "xmm2", "xmm3",
	  "xmm4", "xmm5", "xmm6", "xmm7",
	  "xmm8", "xmm9", "xmm10", "xmm11",
	  "xmm12", "xmm13", "xmm14", "xmm15",
	  "ymm0", "ymm1", "ymm3", "ymm4", "ymm5", "memory"
	)
}

