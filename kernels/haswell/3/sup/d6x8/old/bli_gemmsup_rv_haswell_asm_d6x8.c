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
   rrr:
	 --------        ------        --------      
	 --------        ------        --------      
	 --------   +=   ------ ...    --------      
	 --------        ------        --------      
	 --------        ------            :         
	 --------        ------            :         

   rcr:
	 --------        | | | |       --------      
	 --------        | | | |       --------      
	 --------   +=   | | | | ...   --------      
	 --------        | | | |       --------      
	 --------        | | | |           :         
	 --------        | | | |           :         

   Assumptions:
   - B is row-stored;
   - A is row- or column-stored;
   - m0 and n0 are at most MR and NR, respectively.
   Therefore, this (r)ow-preferential kernel is well-suited for contiguous
   (v)ector loads on B and single-element broadcasts from A.

   NOTE: These kernels explicitly support column-oriented IO, implemented
   via an in-register transpose. And thus they also support the crr and
   ccr cases, though only crr is ever utilized (because ccr is handled by
   transposing the operation and executing rcr, which does not incur the
   cost of the in-register transpose).

   crr:
	 | | | | | | | |       ------        --------      
	 | | | | | | | |       ------        --------      
	 | | | | | | | |  +=   ------ ...    --------      
	 | | | | | | | |       ------        --------      
	 | | | | | | | |       ------            :         
	 | | | | | | | |       ------            :         
*/

// Prototype reference microkernels.
GEMMSUP_KER_PROT( double,   d, gemmsup_r_haswell_ref )

// Define parameters and variables for edge case kernel map.
#define NUM_MR 4
#define NUM_NR 4
#define FUNCPTR_T dgemmsup_ker_ft

static dim_t mrs[NUM_MR] = { 6, 4, 2, 1 };
static dim_t nrs[NUM_NR] = { 8, 4, 2, 1 };
static FUNCPTR_T kmap[NUM_MR][NUM_NR] =
{     /*  8                                4                                2                                1  */
/* 6 */ { bli_dgemmsup_rv_haswell_asm_6x8, bli_dgemmsup_rv_haswell_asm_6x4, bli_dgemmsup_rv_haswell_asm_6x2, bli_dgemmsup_r_haswell_ref_6x1 },
/* 4 */	{ bli_dgemmsup_rv_haswell_asm_4x8, bli_dgemmsup_rv_haswell_asm_4x4, bli_dgemmsup_rv_haswell_asm_4x2, bli_dgemmsup_r_haswell_ref_4x1 },
/* 2 */	{ bli_dgemmsup_rv_haswell_asm_2x8, bli_dgemmsup_rv_haswell_asm_2x4, bli_dgemmsup_rv_haswell_asm_2x2, bli_dgemmsup_r_haswell_ref_2x1 },
/* 1 */	{ bli_dgemmsup_rv_haswell_asm_1x8, bli_dgemmsup_rv_haswell_asm_1x4, bli_dgemmsup_rv_haswell_asm_1x2, bli_dgemmsup_r_haswell_ref_1x1 },
};


void bli_dgemmsup_rv_haswell_asm_6x8
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
	AOCL_DTL_TRACE_ENTRY(AOCL_DTL_LEVEL_TRACE_7);
#if 0
	bli_dgemmsup_r_haswell_ref
	(
	  conja, conjb, m0, n0, k0,
	  alpha, a, rs_a0, cs_a0, b, rs_b0, cs_b0,
	  beta, c, rs_c0, cs_c0, data, cntx
	);
	return;
#endif

	// Use a reference kernel if this is an edge case in the m or n
	// dimensions.
	if ( m0 < 6 || n0 < 8 )
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
	uint64_t k_iter = k0 / 4;
	uint64_t k_left = k0 % 4;

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
	mov(var(cs_a), r9)                 // load cs_a
	lea(mem(, r8, 8), r8)              // rs_a *= sizeof(double)
	lea(mem(, r9, 8), r9)              // cs_a *= sizeof(double)

	lea(mem(r8, r8, 2), r13)           // r13 = 3*rs_a
	lea(mem(r8, r8, 4), r15)           // r15 = 5*rs_a

	mov(var(b), rbx)                   // load address of b.
	mov(var(rs_b), r10)                // load rs_b
	//mov(var(cs_b), r11)                // load cs_b
	lea(mem(, r10, 8), r10)            // rs_b *= sizeof(double)
	//lea(mem(, r11, 8), r11)            // cs_b *= sizeof(double)
	
	                                   // NOTE: We cannot pre-load elements of a or b
	                                   // because it could eventually, in the last
	                                   // unrolled iter or the cleanup loop, result
	                                   // in reading beyond the bounds allocated mem
	                                   // (the likely result: a segmentation fault).

	mov(var(c), rcx)                   // load address of c
	mov(var(rs_c), rdi)                // load rs_c
	lea(mem(, rdi, 8), rdi)            // rs_c *= sizeof(double)

#if 0
	lea(mem(rcx, rdi, 2), rdx)         //
	lea(mem(rdx, rdi, 1), rdx)         // rdx = c + 3*rs_c;
	prefetch(0, mem(rcx, 7*8))         // prefetch c + 0*rs_c
	prefetch(0, mem(rcx, rdi, 1, 7*8)) // prefetch c + 1*rs_c
	prefetch(0, mem(rcx, rdi, 2, 7*8)) // prefetch c + 2*rs_c
	prefetch(0, mem(rdx, 7*8))         // prefetch c + 3*rs_c
	prefetch(0, mem(rdx, rdi, 1, 7*8)) // prefetch c + 4*rs_c
	prefetch(0, mem(rdx, rdi, 2, 7*8)) // prefetch c + 5*rs_c
#else
	cmp(imm(8), rdi)                   // set ZF if (8*rs_c) == 8.
	jz(.DCOLPFETCH)                    // jump to column storage case
	label(.DROWPFETCH)                 // row-stored prefetching on c

	lea(mem(rcx, rdi, 2), rdx)         //
	lea(mem(rdx, rdi, 1), rdx)         // rdx = c + 3*rs_c;
	prefetch(0, mem(rcx, 7*8))         // prefetch c + 0*rs_c
	prefetch(0, mem(rcx, rdi, 1, 7*8)) // prefetch c + 1*rs_c
	prefetch(0, mem(rcx, rdi, 2, 7*8)) // prefetch c + 2*rs_c
	prefetch(0, mem(rdx, 7*8))         // prefetch c + 3*rs_c
	prefetch(0, mem(rdx, rdi, 1, 7*8)) // prefetch c + 4*rs_c
	prefetch(0, mem(rdx, rdi, 2, 7*8)) // prefetch c + 5*rs_c

	jmp(.DPOSTPFETCH)                  // jump to end of prefetching c
	label(.DCOLPFETCH)                 // column-stored prefetching c

	mov(var(cs_c), rsi)                // load cs_c to rsi (temporarily)
	lea(mem(, rsi, 8), rsi)            // cs_c *= sizeof(double)
	lea(mem(rcx, rsi, 2), rdx)         //
	lea(mem(rdx, rsi, 1), rdx)         // rdx = c + 3*cs_c;
	prefetch(0, mem(rcx, 4*8))         // prefetch c + 0*cs_c
	prefetch(0, mem(rcx, rsi, 1, 4*8)) // prefetch c + 1*cs_c
	prefetch(0, mem(rcx, rsi, 2, 4*8)) // prefetch c + 2*cs_c
	prefetch(0, mem(rdx, 4*8))         // prefetch c + 3*cs_c
	prefetch(0, mem(rdx, rsi, 1, 4*8)) // prefetch c + 4*cs_c
	prefetch(0, mem(rdx, rsi, 2, 4*8)) // prefetch c + 5*cs_c
	lea(mem(rdx, rsi, 2), rdx)         // rdx = c + 5*cs_c;
	prefetch(0, mem(rdx, rsi, 1, 4*8)) // prefetch c + 6*cs_c
	prefetch(0, mem(rdx, rsi, 2, 4*8)) // prefetch c + 7*cs_c

	label(.DPOSTPFETCH)                // done prefetching c
#endif


#if 1
	lea(mem(rax, r9,  8), rdx)         //
	lea(mem(rdx, r9,  8), rdx)         // rdx = a + 16*cs_a;
#endif
	
	
	
	
	mov(var(k_iter), rsi)              // i = k_iter;
	test(rsi, rsi)                     // check i via logical AND.
	je(.DCONSIDKLEFT)                  // if i == 0, jump to code that
	                                   // contains the k_left loop.
	
	
	label(.DLOOPKITER)                 // MAIN LOOP
	
	
	// ---------------------------------- iteration 0

#if 1
	prefetch(0, mem(rdx, 5*8))
#endif
	
	vmovupd(mem(rbx,  0*32), ymm0)
	vmovupd(mem(rbx,  1*32), ymm1)
	add(r10, rbx)                      // b += rs_b;

	vbroadcastsd(mem(rax        ), ymm2)
	vbroadcastsd(mem(rax, r8,  1), ymm3)
	vfmadd231pd(ymm0, ymm2, ymm4)
	vfmadd231pd(ymm1, ymm2, ymm5)
	vfmadd231pd(ymm0, ymm3, ymm6)
	vfmadd231pd(ymm1, ymm3, ymm7)
	
	vbroadcastsd(mem(rax, r8,  2), ymm2)
	vbroadcastsd(mem(rax, r13, 1), ymm3)
	vfmadd231pd(ymm0, ymm2, ymm8)
	vfmadd231pd(ymm1, ymm2, ymm9)
	vfmadd231pd(ymm0, ymm3, ymm10)
	vfmadd231pd(ymm1, ymm3, ymm11)
	
	vbroadcastsd(mem(rax, r8,  4), ymm2)
	vbroadcastsd(mem(rax, r15, 1), ymm3)
	add(r9, rax)                       // a += cs_a;
	vfmadd231pd(ymm0, ymm2, ymm12)
	vfmadd231pd(ymm1, ymm2, ymm13)
	vfmadd231pd(ymm0, ymm3, ymm14)
	vfmadd231pd(ymm1, ymm3, ymm15)

	
	// ---------------------------------- iteration 1

#if 0
	prefetch(0, mem(rdx, r9, 1, 5*8))
#endif

	vmovupd(mem(rbx,  0*32), ymm0)
	vmovupd(mem(rbx,  1*32), ymm1)
	add(r10, rbx)                      // b += rs_b;

	vbroadcastsd(mem(rax        ), ymm2)
	vbroadcastsd(mem(rax, r8,  1), ymm3)
	vfmadd231pd(ymm0, ymm2, ymm4)
	vfmadd231pd(ymm1, ymm2, ymm5)
	vfmadd231pd(ymm0, ymm3, ymm6)
	vfmadd231pd(ymm1, ymm3, ymm7)
	
	vbroadcastsd(mem(rax, r8,  2), ymm2)
	vbroadcastsd(mem(rax, r13, 1), ymm3)
	vfmadd231pd(ymm0, ymm2, ymm8)
	vfmadd231pd(ymm1, ymm2, ymm9)
	vfmadd231pd(ymm0, ymm3, ymm10)
	vfmadd231pd(ymm1, ymm3, ymm11)
	
	vbroadcastsd(mem(rax, r8,  4), ymm2)
	vbroadcastsd(mem(rax, r15, 1), ymm3)
	add(r9, rax)                       // a += cs_a;
	vfmadd231pd(ymm0, ymm2, ymm12)
	vfmadd231pd(ymm1, ymm2, ymm13)
	vfmadd231pd(ymm0, ymm3, ymm14)
	vfmadd231pd(ymm1, ymm3, ymm15)
	
	
	// ---------------------------------- iteration 2

#if 1
	prefetch(0, mem(rdx, r9, 2, 5*8))
#endif
	
	vmovupd(mem(rbx,  0*32), ymm0)
	vmovupd(mem(rbx,  1*32), ymm1)
	add(r10, rbx)                      // b += rs_b;

	vbroadcastsd(mem(rax        ), ymm2)
	vbroadcastsd(mem(rax, r8,  1), ymm3)
	vfmadd231pd(ymm0, ymm2, ymm4)
	vfmadd231pd(ymm1, ymm2, ymm5)
	vfmadd231pd(ymm0, ymm3, ymm6)
	vfmadd231pd(ymm1, ymm3, ymm7)
	
	vbroadcastsd(mem(rax, r8,  2), ymm2)
	vbroadcastsd(mem(rax, r13, 1), ymm3)
	vfmadd231pd(ymm0, ymm2, ymm8)
	vfmadd231pd(ymm1, ymm2, ymm9)
	vfmadd231pd(ymm0, ymm3, ymm10)
	vfmadd231pd(ymm1, ymm3, ymm11)
	
	vbroadcastsd(mem(rax, r8,  4), ymm2)
	vbroadcastsd(mem(rax, r15, 1), ymm3)
	add(r9, rax)                       // a += cs_a;
	vfmadd231pd(ymm0, ymm2, ymm12)
	vfmadd231pd(ymm1, ymm2, ymm13)
	vfmadd231pd(ymm0, ymm3, ymm14)
	vfmadd231pd(ymm1, ymm3, ymm15)
	

	// ---------------------------------- iteration 3

#if 1
	lea(mem(rdx, r9,  4), rdx)         // a_prefetch += 4*cs_a;
#endif

	vmovupd(mem(rbx, 0*32), ymm0)
	vmovupd(mem(rbx, 1*32), ymm1)
	add(r10, rbx)                      // b += rs_b;

	vbroadcastsd(mem(rax        ), ymm2)
	vbroadcastsd(mem(rax, r8,  1), ymm3)
	vfmadd231pd(ymm0, ymm2, ymm4)
	vfmadd231pd(ymm1, ymm2, ymm5)
	vfmadd231pd(ymm0, ymm3, ymm6)
	vfmadd231pd(ymm1, ymm3, ymm7)
	
	vbroadcastsd(mem(rax, r8,  2), ymm2)
	vbroadcastsd(mem(rax, r13, 1), ymm3)
	vfmadd231pd(ymm0, ymm2, ymm8)
	vfmadd231pd(ymm1, ymm2, ymm9)
	vfmadd231pd(ymm0, ymm3, ymm10)
	vfmadd231pd(ymm1, ymm3, ymm11)
	
	vbroadcastsd(mem(rax, r8,  4), ymm2)
	vbroadcastsd(mem(rax, r15, 1), ymm3)
	add(r9, rax)                       // a += cs_a;
	vfmadd231pd(ymm0, ymm2, ymm12)
	vfmadd231pd(ymm1, ymm2, ymm13)
	vfmadd231pd(ymm0, ymm3, ymm14)
	vfmadd231pd(ymm1, ymm3, ymm15)
	
	
	
	dec(rsi)                           // i -= 1;
	jne(.DLOOPKITER)                   // iterate again if i != 0.
	
	
	
	
	
	
	label(.DCONSIDKLEFT)
	
	mov(var(k_left), rsi)              // i = k_left;
	test(rsi, rsi)                     // check i via logical AND.
	je(.DPOSTACCUM)                    // if i == 0, we're done; jump to end.
	                                   // else, we prepare to enter k_left loop.
	
	
	label(.DLOOPKLEFT)                 // EDGE LOOP
	
	vmovupd(mem(rbx,  0*32), ymm0)
	vmovupd(mem(rbx,  1*32), ymm1)
	add(r10, rbx)                      // b += rs_b;
	
	vbroadcastsd(mem(rax        ), ymm2)
	vbroadcastsd(mem(rax, r8,  1), ymm3)
	vfmadd231pd(ymm0, ymm2, ymm4)
	vfmadd231pd(ymm1, ymm2, ymm5)
	vfmadd231pd(ymm0, ymm3, ymm6)
	vfmadd231pd(ymm1, ymm3, ymm7)
	
	vbroadcastsd(mem(rax, r8,  2), ymm2)
	vbroadcastsd(mem(rax, r13, 1), ymm3)
	vfmadd231pd(ymm0, ymm2, ymm8)
	vfmadd231pd(ymm1, ymm2, ymm9)
	vfmadd231pd(ymm0, ymm3, ymm10)
	vfmadd231pd(ymm1, ymm3, ymm11)
	
	vbroadcastsd(mem(rax, r8,  4), ymm2)
	vbroadcastsd(mem(rax, r15, 1), ymm3)
	add(r9, rax)                       // a += cs_a;
	vfmadd231pd(ymm0, ymm2, ymm12)
	vfmadd231pd(ymm1, ymm2, ymm13)
	vfmadd231pd(ymm0, ymm3, ymm14)
	vfmadd231pd(ymm1, ymm3, ymm15)
	
	
	dec(rsi)                           // i -= 1;
	jne(.DLOOPKLEFT)                   // iterate again if i != 0.
	
	
	
	label(.DPOSTACCUM)

	
	
	mov(var(alpha), rax)               // load address of alpha
	mov(var(beta), rbx)                // load address of beta
	vbroadcastsd(mem(rax), ymm0)       // load alpha and duplicate
	vbroadcastsd(mem(rbx), ymm3)       // load beta and duplicate
	
	vmulpd(ymm0, ymm4, ymm4)           // scale by alpha
	vmulpd(ymm0, ymm5, ymm5)
	vmulpd(ymm0, ymm6, ymm6)
	vmulpd(ymm0, ymm7, ymm7)
	vmulpd(ymm0, ymm8, ymm8)
	vmulpd(ymm0, ymm9, ymm9)
	vmulpd(ymm0, ymm10, ymm10)
	vmulpd(ymm0, ymm11, ymm11)
	vmulpd(ymm0, ymm12, ymm12)
	vmulpd(ymm0, ymm13, ymm13)
	vmulpd(ymm0, ymm14, ymm14)
	vmulpd(ymm0, ymm15, ymm15)
	
	
	
	
	
	
	mov(var(cs_c), rsi)                // load cs_c
	lea(mem(, rsi, 8), rsi)            // rsi = cs_c * sizeof(double)
	
	//lea(mem(rcx, rsi, 4), rdx)         // load address of c +  4*cs_c;
	lea(mem(rcx, rdi, 4), rdx)         // load address of c +  4*rs_c;

	lea(mem(rsi, rsi, 2), rax)         // rax = 3*cs_c;
	
	
	
	                                   // now avoid loading C if beta == 0
	
	vxorpd(ymm0, ymm0, ymm0)           // set ymm0 to zero.
	vucomisd(xmm0, xmm3)               // set ZF if beta == 0.
	je(.DBETAZERO)                     // if ZF = 1, jump to beta == 0 case



	cmp(imm(8), rdi)                   // set ZF if (8*rs_c) == 8.
	jz(.DCOLSTORED)                    // jump to column storage case
	

	
	label(.DROWSTORED)
	
	
	vfmadd231pd(mem(rcx), ymm3, ymm4)
	vmovupd(ymm4, mem(rcx))

	vfmadd231pd(mem(rcx, rsi, 4), ymm3, ymm5)
	vmovupd(ymm5, mem(rcx, rsi, 4))
	add(rdi, rcx)
	
	
	vfmadd231pd(mem(rcx), ymm3, ymm6)
	vmovupd(ymm6, mem(rcx))

	vfmadd231pd(mem(rcx, rsi, 4), ymm3, ymm7)
	vmovupd(ymm7, mem(rcx, rsi, 4))
	add(rdi, rcx)
	
	
	vfmadd231pd(mem(rcx), ymm3, ymm8)
	vmovupd(ymm8, mem(rcx))

	vfmadd231pd(mem(rcx, rsi, 4), ymm3, ymm9)
	vmovupd(ymm9, mem(rcx, rsi, 4))
	add(rdi, rcx)
	
	
	vfmadd231pd(mem(rcx), ymm3, ymm10)
	vmovupd(ymm10, mem(rcx))

	vfmadd231pd(mem(rcx, rsi, 4), ymm3, ymm11)
	vmovupd(ymm11, mem(rcx, rsi, 4))
	add(rdi, rcx)
	
	
	vfmadd231pd(mem(rcx), ymm3, ymm12)
	vmovupd(ymm12, mem(rcx))

	vfmadd231pd(mem(rcx, rsi, 4), ymm3, ymm13)
	vmovupd(ymm13, mem(rcx, rsi, 4))
	add(rdi, rcx)
	
	
	vfmadd231pd(mem(rcx), ymm3, ymm14)
	vmovupd(ymm14, mem(rcx))

	vfmadd231pd(mem(rcx, rsi, 4), ymm3, ymm15)
	vmovupd(ymm15, mem(rcx, rsi, 4))
	//add(rdi, rcx)
	
	
	jmp(.DDONE)                        // jump to end.



	label(.DCOLSTORED)


	vunpcklpd(ymm6, ymm4, ymm0)
	vunpckhpd(ymm6, ymm4, ymm1)
	vunpcklpd(ymm10, ymm8, ymm2)
	vunpckhpd(ymm10, ymm8, ymm3)
	vinsertf128(imm(0x1), xmm2, ymm0, ymm4)
	vinsertf128(imm(0x1), xmm3, ymm1, ymm6)
	vperm2f128(imm(0x31), ymm2, ymm0, ymm8)
	vperm2f128(imm(0x31), ymm3, ymm1, ymm10)

	vbroadcastsd(mem(rbx), ymm3)

	vfmadd231pd(mem(rcx), ymm3, ymm4)
	vfmadd231pd(mem(rcx, rsi, 1), ymm3, ymm6)
	vfmadd231pd(mem(rcx, rsi, 2), ymm3, ymm8)
	vfmadd231pd(mem(rcx, rax, 1), ymm3, ymm10)
	vmovupd(ymm4, mem(rcx))
	vmovupd(ymm6, mem(rcx, rsi, 1))
	vmovupd(ymm8, mem(rcx, rsi, 2))
	vmovupd(ymm10, mem(rcx, rax, 1))

	lea(mem(rcx, rsi, 4), rcx)

	vunpcklpd(ymm14, ymm12, ymm0)
	vunpckhpd(ymm14, ymm12, ymm1)
	vextractf128(imm(0x1), ymm0, xmm2)
	vextractf128(imm(0x1), ymm1, xmm4)

	vfmadd231pd(mem(rdx), xmm3, xmm0)
	vfmadd231pd(mem(rdx, rsi, 1), xmm3, xmm1)
	vfmadd231pd(mem(rdx, rsi, 2), xmm3, xmm2)
	vfmadd231pd(mem(rdx, rax, 1), xmm3, xmm4)
	vmovupd(xmm0, mem(rdx))
	vmovupd(xmm1, mem(rdx, rsi, 1))
	vmovupd(xmm2, mem(rdx, rsi, 2))
	vmovupd(xmm4, mem(rdx, rax, 1))

	lea(mem(rdx, rsi, 4), rdx)


	vunpcklpd(ymm7, ymm5, ymm0)
	vunpckhpd(ymm7, ymm5, ymm1)
	vunpcklpd(ymm11, ymm9, ymm2)
	vunpckhpd(ymm11, ymm9, ymm3)
	vinsertf128(imm(0x1), xmm2, ymm0, ymm5)
	vinsertf128(imm(0x1), xmm3, ymm1, ymm7)
	vperm2f128(imm(0x31), ymm2, ymm0, ymm9)
	vperm2f128(imm(0x31), ymm3, ymm1, ymm11)

	vbroadcastsd(mem(rbx), ymm3)

	vfmadd231pd(mem(rcx), ymm3, ymm5)
	vfmadd231pd(mem(rcx, rsi, 1), ymm3, ymm7)
	vfmadd231pd(mem(rcx, rsi, 2), ymm3, ymm9)
	vfmadd231pd(mem(rcx, rax, 1), ymm3, ymm11)
	vmovupd(ymm5, mem(rcx))
	vmovupd(ymm7, mem(rcx, rsi, 1))
	vmovupd(ymm9, mem(rcx, rsi, 2))
	vmovupd(ymm11, mem(rcx, rax, 1))

	//lea(mem(rcx, rsi, 4), rcx)

	vunpcklpd(ymm15, ymm13, ymm0)
	vunpckhpd(ymm15, ymm13, ymm1)
	vextractf128(imm(0x1), ymm0, xmm2)
	vextractf128(imm(0x1), ymm1, xmm4)

	vfmadd231pd(mem(rdx), xmm3, xmm0)
	vfmadd231pd(mem(rdx, rsi, 1), xmm3, xmm1)
	vfmadd231pd(mem(rdx, rsi, 2), xmm3, xmm2)
	vfmadd231pd(mem(rdx, rax, 1), xmm3, xmm4)
	vmovupd(xmm0, mem(rdx))
	vmovupd(xmm1, mem(rdx, rsi, 1))
	vmovupd(xmm2, mem(rdx, rsi, 2))
	vmovupd(xmm4, mem(rdx, rax, 1))

	//lea(mem(rdx, rsi, 4), rdx)


	jmp(.DDONE)                        // jump to end.
	
	
	
	
	label(.DBETAZERO)
	

	cmp(imm(8), rdi)                   // set ZF if (8*rs_c) == 8.
	jz(.DCOLSTORBZ)                    // jump to column storage case


	
	label(.DROWSTORBZ)
	
	
	vmovupd(ymm4, mem(rcx))
	vmovupd(ymm5, mem(rcx, rsi, 4))
	add(rdi, rcx)
	

	vmovupd(ymm6, mem(rcx))
	vmovupd(ymm7, mem(rcx, rsi, 4))
	add(rdi, rcx)
	
	
	vmovupd(ymm8, mem(rcx))
	vmovupd(ymm9, mem(rcx, rsi, 4))
	add(rdi, rcx)
	
	
	vmovupd(ymm10, mem(rcx))
	vmovupd(ymm11, mem(rcx, rsi, 4))
	add(rdi, rcx)
	
	
	vmovupd(ymm12, mem(rcx))
	vmovupd(ymm13, mem(rcx, rsi, 4))
	add(rdi, rcx)
	
	
	vmovupd(ymm14, mem(rcx))
	vmovupd(ymm15, mem(rcx, rsi, 4))
	//add(rdi, rcx)
	
	
	jmp(.DDONE)                        // jump to end.



	label(.DCOLSTORBZ)


	vunpcklpd(ymm6, ymm4, ymm0)
	vunpckhpd(ymm6, ymm4, ymm1)
	vunpcklpd(ymm10, ymm8, ymm2)
	vunpckhpd(ymm10, ymm8, ymm3)
	vinsertf128(imm(0x1), xmm2, ymm0, ymm4)
	vinsertf128(imm(0x1), xmm3, ymm1, ymm6)
	vperm2f128(imm(0x31), ymm2, ymm0, ymm8)
	vperm2f128(imm(0x31), ymm3, ymm1, ymm10)

	vmovupd(ymm4, mem(rcx))
	vmovupd(ymm6, mem(rcx, rsi, 1))
	vmovupd(ymm8, mem(rcx, rsi, 2))
	vmovupd(ymm10, mem(rcx, rax, 1))

	lea(mem(rcx, rsi, 4), rcx)

	vunpcklpd(ymm14, ymm12, ymm0)
	vunpckhpd(ymm14, ymm12, ymm1)
	vextractf128(imm(0x1), ymm0, xmm2)
	vextractf128(imm(0x1), ymm1, xmm4)

	vmovupd(xmm0, mem(rdx))
	vmovupd(xmm1, mem(rdx, rsi, 1))
	vmovupd(xmm2, mem(rdx, rsi, 2))
	vmovupd(xmm4, mem(rdx, rax, 1))

	lea(mem(rdx, rsi, 4), rdx)


	vunpcklpd(ymm7, ymm5, ymm0)
	vunpckhpd(ymm7, ymm5, ymm1)
	vunpcklpd(ymm11, ymm9, ymm2)
	vunpckhpd(ymm11, ymm9, ymm3)
	vinsertf128(imm(0x1), xmm2, ymm0, ymm5)
	vinsertf128(imm(0x1), xmm3, ymm1, ymm7)
	vperm2f128(imm(0x31), ymm2, ymm0, ymm9)
	vperm2f128(imm(0x31), ymm3, ymm1, ymm11)

	vmovupd(ymm5, mem(rcx))
	vmovupd(ymm7, mem(rcx, rsi, 1))
	vmovupd(ymm9, mem(rcx, rsi, 2))
	vmovupd(ymm11, mem(rcx, rax, 1))

	//lea(mem(rcx, rsi, 4), rcx)

	vunpcklpd(ymm15, ymm13, ymm0)
	vunpckhpd(ymm15, ymm13, ymm1)
	vextractf128(imm(0x1), ymm0, xmm2)
	vextractf128(imm(0x1), ymm1, xmm4)

	vmovupd(xmm0, mem(rdx))
	vmovupd(xmm1, mem(rdx, rsi, 1))
	vmovupd(xmm2, mem(rdx, rsi, 2))
	vmovupd(xmm4, mem(rdx, rax, 1))

	//lea(mem(rdx, rsi, 4), rdx)

	
	
	
	label(.DDONE)
	
	

    end_asm(
	: // output operands (none)
	: // input operands
      [k_iter] "m" (k_iter),
      [k_left] "m" (k_left),
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
	AOCL_DTL_TRACE_EXIT(AOCL_DTL_LEVEL_TRACE_7);
}

void bli_dgemmsup_rv_haswell_asm_5x8
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
	AOCL_DTL_TRACE_ENTRY(AOCL_DTL_LEVEL_TRACE_7);
	//void*    a_next = bli_auxinfo_next_a( data );
	//void*    b_next = bli_auxinfo_next_b( data );

	// Typecast local copies of integers in case dim_t and inc_t are a
	// different size than is expected by load instructions.
	uint64_t k_iter = k0 / 4;
	uint64_t k_left = k0 % 4;

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
	mov(var(cs_a), r9)                 // load cs_a
	lea(mem(, r8, 8), r8)              // rs_a *= sizeof(double)
	lea(mem(, r9, 8), r9)              // cs_a *= sizeof(double)
	
	lea(mem(r8, r8, 2), r13)           // r13 = 3*rs_a
	//lea(mem(r8, r8, 4), r15)           // r15 = 5*rs_a

	mov(var(b), rbx)                   // load address of b.
	mov(var(rs_b), r10)                // load rs_b
	//mov(var(cs_b), r11)                // load cs_b
	lea(mem(, r10, 8), r10)            // rs_b *= sizeof(double)
	//lea(mem(, r11, 8), r11)            // cs_b *= sizeof(double)

	                                   // NOTE: We cannot pre-load elements of a or b
	                                   // because it could eventually, in the last
	                                   // unrolled iter or the cleanup loop, result
	                                   // in reading beyond the bounds allocated mem
	                                   // (the likely result: a segmentation fault).

	mov(var(c), rcx)                   // load address of c
	mov(var(rs_c), rdi)                // load rs_c
	lea(mem(, rdi, 8), rdi)            // rs_c *= sizeof(double)



	cmp(imm(8), rdi)                   // set ZF if (8*rs_c) == 8.
	jz(.DCOLPFETCH)                    // jump to column storage case
	label(.DROWPFETCH)                 // row-stored prefetching on c

	lea(mem(rcx, rdi, 2), rdx)         //
	lea(mem(rdx, rdi, 1), rdx)         // rdx = c + 3*rs_c;
	prefetch(0, mem(rcx, 7*8))         // prefetch c + 0*rs_c
	prefetch(0, mem(rcx, rdi, 1, 7*8)) // prefetch c + 1*rs_c
	prefetch(0, mem(rcx, rdi, 2, 7*8)) // prefetch c + 2*rs_c
	prefetch(0, mem(rdx, 7*8))         // prefetch c + 3*rs_c
	prefetch(0, mem(rdx, rdi, 1, 7*8)) // prefetch c + 4*rs_c

	jmp(.DPOSTPFETCH)                  // jump to end of prefetching c
	label(.DCOLPFETCH)                 // column-stored prefetching c

	mov(var(cs_c), rsi)                // load cs_c to rsi (temporarily)
	lea(mem(, rsi, 8), rsi)            // cs_c *= sizeof(double)
	lea(mem(rcx, rsi, 2), rdx)         //
	lea(mem(rdx, rsi, 1), rdx)         // rdx = c + 3*cs_c;
	prefetch(0, mem(rcx, 4*8))         // prefetch c + 0*cs_c
	prefetch(0, mem(rcx, rsi, 1, 4*8)) // prefetch c + 1*cs_c
	prefetch(0, mem(rcx, rsi, 2, 4*8)) // prefetch c + 2*cs_c
	prefetch(0, mem(rdx, 4*8))         // prefetch c + 3*cs_c
	prefetch(0, mem(rdx, rsi, 1, 4*8)) // prefetch c + 4*cs_c
	prefetch(0, mem(rdx, rsi, 2, 4*8)) // prefetch c + 5*cs_c
	lea(mem(rdx, rsi, 2), rdx)         // rdx = c + 5*cs_c;
	prefetch(0, mem(rdx, rsi, 1, 4*8)) // prefetch c + 6*cs_c
	prefetch(0, mem(rdx, rsi, 2, 4*8)) // prefetch c + 7*cs_c

	label(.DPOSTPFETCH)                // done prefetching c

	
	
	
	mov(var(k_iter), rsi)              // i = k_iter;
	test(rsi, rsi)                     // check i via logical AND.
	je(.DCONSIDKLEFT)                  // if i == 0, jump to code that
	                                   // contains the k_left loop.
	
	
	label(.DLOOPKITER)                 // MAIN LOOP
	
	
	// ---------------------------------- iteration 0

	vmovupd(mem(rbx,  0*32), ymm0)
	vmovupd(mem(rbx,  1*32), ymm1)
	add(r10, rbx)                      // b += rs_b;

	vbroadcastsd(mem(rax        ), ymm2)
	vbroadcastsd(mem(rax, r8,  1), ymm3)
	vfmadd231pd(ymm0, ymm2, ymm4)
	vfmadd231pd(ymm1, ymm2, ymm5)
	vfmadd231pd(ymm0, ymm3, ymm6)
	vfmadd231pd(ymm1, ymm3, ymm7)
	
	vbroadcastsd(mem(rax, r8,  2), ymm2)
	vbroadcastsd(mem(rax, r13, 1), ymm3)
	vfmadd231pd(ymm0, ymm2, ymm8)
	vfmadd231pd(ymm1, ymm2, ymm9)
	vfmadd231pd(ymm0, ymm3, ymm10)
	vfmadd231pd(ymm1, ymm3, ymm11)
	
	vbroadcastsd(mem(rax, r8,  4), ymm2)
	add(r9, rax)                       // a += cs_a;
	vfmadd231pd(ymm0, ymm2, ymm12)
	vfmadd231pd(ymm1, ymm2, ymm13)

	
	// ---------------------------------- iteration 1

	vmovupd(mem(rbx,  0*32), ymm0)
	vmovupd(mem(rbx,  1*32), ymm1)
	add(r10, rbx)                      // b += rs_b;

	vbroadcastsd(mem(rax        ), ymm2)
	vbroadcastsd(mem(rax, r8,  1), ymm3)
	vfmadd231pd(ymm0, ymm2, ymm4)
	vfmadd231pd(ymm1, ymm2, ymm5)
	vfmadd231pd(ymm0, ymm3, ymm6)
	vfmadd231pd(ymm1, ymm3, ymm7)
	
	vbroadcastsd(mem(rax, r8,  2), ymm2)
	vbroadcastsd(mem(rax, r13, 1), ymm3)
	vfmadd231pd(ymm0, ymm2, ymm8)
	vfmadd231pd(ymm1, ymm2, ymm9)
	vfmadd231pd(ymm0, ymm3, ymm10)
	vfmadd231pd(ymm1, ymm3, ymm11)
	
	vbroadcastsd(mem(rax, r8,  4), ymm2)
	add(r9, rax)                       // a += cs_a;
	vfmadd231pd(ymm0, ymm2, ymm12)
	vfmadd231pd(ymm1, ymm2, ymm13)
	
	
	// ---------------------------------- iteration 2
	
	vmovupd(mem(rbx,  0*32), ymm0)
	vmovupd(mem(rbx,  1*32), ymm1)
	add(r10, rbx)                      // b += rs_b;

	vbroadcastsd(mem(rax        ), ymm2)
	vbroadcastsd(mem(rax, r8,  1), ymm3)
	vfmadd231pd(ymm0, ymm2, ymm4)
	vfmadd231pd(ymm1, ymm2, ymm5)
	vfmadd231pd(ymm0, ymm3, ymm6)
	vfmadd231pd(ymm1, ymm3, ymm7)
	
	vbroadcastsd(mem(rax, r8,  2), ymm2)
	vbroadcastsd(mem(rax, r13, 1), ymm3)
	vfmadd231pd(ymm0, ymm2, ymm8)
	vfmadd231pd(ymm1, ymm2, ymm9)
	vfmadd231pd(ymm0, ymm3, ymm10)
	vfmadd231pd(ymm1, ymm3, ymm11)
	
	vbroadcastsd(mem(rax, r8,  4), ymm2)
	add(r9, rax)                       // a += cs_a;
	vfmadd231pd(ymm0, ymm2, ymm12)
	vfmadd231pd(ymm1, ymm2, ymm13)
	

	// ---------------------------------- iteration 3

	vmovupd(mem(rbx, 0*32), ymm0)
	vmovupd(mem(rbx, 1*32), ymm1)
	add(r10, rbx)                      // b += rs_b;

	vbroadcastsd(mem(rax        ), ymm2)
	vbroadcastsd(mem(rax, r8,  1), ymm3)
	vfmadd231pd(ymm0, ymm2, ymm4)
	vfmadd231pd(ymm1, ymm2, ymm5)
	vfmadd231pd(ymm0, ymm3, ymm6)
	vfmadd231pd(ymm1, ymm3, ymm7)
	
	vbroadcastsd(mem(rax, r8,  2), ymm2)
	vbroadcastsd(mem(rax, r13, 1), ymm3)
	vfmadd231pd(ymm0, ymm2, ymm8)
	vfmadd231pd(ymm1, ymm2, ymm9)
	vfmadd231pd(ymm0, ymm3, ymm10)
	vfmadd231pd(ymm1, ymm3, ymm11)
	
	vbroadcastsd(mem(rax, r8,  4), ymm2)
	add(r9, rax)                       // a += cs_a;
	vfmadd231pd(ymm0, ymm2, ymm12)
	vfmadd231pd(ymm1, ymm2, ymm13)
	
	
	
	dec(rsi)                           // i -= 1;
	jne(.DLOOPKITER)                   // iterate again if i != 0.
	
	
	
	
	
	
	label(.DCONSIDKLEFT)
	
	mov(var(k_left), rsi)              // i = k_left;
	test(rsi, rsi)                     // check i via logical AND.
	je(.DPOSTACCUM)                    // if i == 0, we're done; jump to end.
	                                   // else, we prepare to enter k_left loop.
	
	
	label(.DLOOPKLEFT)                 // EDGE LOOP

	vmovupd(mem(rbx,  0*32), ymm0)
	vmovupd(mem(rbx,  1*32), ymm1)
	add(r10, rbx)                      // b += rs_b;
	
	vbroadcastsd(mem(rax        ), ymm2)
	vbroadcastsd(mem(rax, r8,  1), ymm3)
	vfmadd231pd(ymm0, ymm2, ymm4)
	vfmadd231pd(ymm1, ymm2, ymm5)
	vfmadd231pd(ymm0, ymm3, ymm6)
	vfmadd231pd(ymm1, ymm3, ymm7)
	
	vbroadcastsd(mem(rax, r8,  2), ymm2)
	vbroadcastsd(mem(rax, r13, 1), ymm3)
	vfmadd231pd(ymm0, ymm2, ymm8)
	vfmadd231pd(ymm1, ymm2, ymm9)
	vfmadd231pd(ymm0, ymm3, ymm10)
	vfmadd231pd(ymm1, ymm3, ymm11)
	
	vbroadcastsd(mem(rax, r8,  4), ymm2)
	add(r9, rax)                       // a += cs_a;
	vfmadd231pd(ymm0, ymm2, ymm12)
	vfmadd231pd(ymm1, ymm2, ymm13)
	
	
	dec(rsi)                           // i -= 1;
	jne(.DLOOPKLEFT)                   // iterate again if i != 0.
	
	
	
	label(.DPOSTACCUM)

	
	
	mov(var(alpha), rax)               // load address of alpha
	mov(var(beta), rbx)                // load address of beta
	vbroadcastsd(mem(rax), ymm0)       // load alpha and duplicate
	vbroadcastsd(mem(rbx), ymm3)       // load beta and duplicate
	
	vmulpd(ymm0, ymm4, ymm4)           // scale by alpha
	vmulpd(ymm0, ymm5, ymm5)
	vmulpd(ymm0, ymm6, ymm6)
	vmulpd(ymm0, ymm7, ymm7)
	vmulpd(ymm0, ymm8, ymm8)
	vmulpd(ymm0, ymm9, ymm9)
	vmulpd(ymm0, ymm10, ymm10)
	vmulpd(ymm0, ymm11, ymm11)
	vmulpd(ymm0, ymm12, ymm12)
	vmulpd(ymm0, ymm13, ymm13)
	
	
	
	
	
	
	mov(var(cs_c), rsi)                // load cs_c
	lea(mem(, rsi, 8), rsi)            // rsi = cs_c * sizeof(double)
	
	//lea(mem(rcx, rsi, 4), rdx)         // load address of c +  4*cs_c;
	lea(mem(rcx, rdi, 4), rdx)         // load address of c +  4*rs_c;

	lea(mem(rsi, rsi, 2), rax)         // rax = 3*cs_c;
	
	
	
	                                   // now avoid loading C if beta == 0
	
	vxorpd(ymm0, ymm0, ymm0)           // set ymm0 to zero.
	vucomisd(xmm0, xmm3)               // set ZF if beta == 0.
	je(.DBETAZERO)                     // if ZF = 1, jump to beta == 0 case



	cmp(imm(8), rdi)                   // set ZF if (8*rs_c) == 8.
	jz(.DCOLSTORED)                    // jump to column storage case
	

	
	label(.DROWSTORED)
	
	
	vfmadd231pd(mem(rcx), ymm3, ymm4)
	vmovupd(ymm4, mem(rcx))

	vfmadd231pd(mem(rcx, rsi, 4), ymm3, ymm5)
	vmovupd(ymm5, mem(rcx, rsi, 4))
	add(rdi, rcx)
	
	
	vfmadd231pd(mem(rcx), ymm3, ymm6)
	vmovupd(ymm6, mem(rcx))

	vfmadd231pd(mem(rcx, rsi, 4), ymm3, ymm7)
	vmovupd(ymm7, mem(rcx, rsi, 4))
	add(rdi, rcx)
	
	
	vfmadd231pd(mem(rcx), ymm3, ymm8)
	vmovupd(ymm8, mem(rcx))

	vfmadd231pd(mem(rcx, rsi, 4), ymm3, ymm9)
	vmovupd(ymm9, mem(rcx, rsi, 4))
	add(rdi, rcx)
	
	
	vfmadd231pd(mem(rcx), ymm3, ymm10)
	vmovupd(ymm10, mem(rcx))

	vfmadd231pd(mem(rcx, rsi, 4), ymm3, ymm11)
	vmovupd(ymm11, mem(rcx, rsi, 4))
	add(rdi, rcx)
	
	
	vfmadd231pd(mem(rcx), ymm3, ymm12)
	vmovupd(ymm12, mem(rcx))

	vfmadd231pd(mem(rcx, rsi, 4), ymm3, ymm13)
	vmovupd(ymm13, mem(rcx, rsi, 4))
	//add(rdi, rcx)
	
	
	jmp(.DDONE)                        // jump to end.



	label(.DCOLSTORED)


	vunpcklpd(ymm6, ymm4, ymm0)
	vunpckhpd(ymm6, ymm4, ymm1)
	vunpcklpd(ymm10, ymm8, ymm2)
	vunpckhpd(ymm10, ymm8, ymm3)
	vinsertf128(imm(0x1), xmm2, ymm0, ymm4)
	vinsertf128(imm(0x1), xmm3, ymm1, ymm6)
	vperm2f128(imm(0x31), ymm2, ymm0, ymm8)
	vperm2f128(imm(0x31), ymm3, ymm1, ymm10)

	vbroadcastsd(mem(rbx), ymm3)

	vfmadd231pd(mem(rcx), ymm3, ymm4)
	vfmadd231pd(mem(rcx, rsi, 1), ymm3, ymm6)
	vfmadd231pd(mem(rcx, rsi, 2), ymm3, ymm8)
	vfmadd231pd(mem(rcx, rax, 1), ymm3, ymm10)
	vmovupd(ymm4, mem(rcx))
	vmovupd(ymm6, mem(rcx, rsi, 1))
	vmovupd(ymm8, mem(rcx, rsi, 2))
	vmovupd(ymm10, mem(rcx, rax, 1))

	lea(mem(rcx, rsi, 4), rcx)

#if 0
	vunpcklpd(ymm14, ymm12, ymm0)
	vunpckhpd(ymm14, ymm12, ymm1)
	vextractf128(imm(0x1), ymm0, xmm2)
	vextractf128(imm(0x1), ymm1, xmm4)

	vfmadd231pd(mem(rdx), xmm3, xmm0)
	vfmadd231pd(mem(rdx, rsi, 1), xmm3, xmm1)
	vfmadd231pd(mem(rdx, rsi, 2), xmm3, xmm2)
	vfmadd231pd(mem(rdx, rax, 1), xmm3, xmm4)
	vmovupd(xmm0, mem(rdx))
	vmovupd(xmm1, mem(rdx, rsi, 1))
	vmovupd(xmm2, mem(rdx, rsi, 2))
	vmovupd(xmm4, mem(rdx, rax, 1))
#else
	vmovlpd(mem(rdx), xmm0, xmm0)
	vmovhpd(mem(rdx, rsi, 1), xmm0, xmm0)
	vmovlpd(mem(rdx, rsi, 2), xmm1, xmm1)
	vmovhpd(mem(rdx, rax, 1), xmm1, xmm1)
	vperm2f128(imm(0x20), ymm1, ymm0, ymm0)

	vfmadd213pd(ymm12, ymm3, ymm0)
	vextractf128(imm(1), ymm0, xmm1)
	vmovlpd(xmm0, mem(rdx))
	vmovhpd(xmm0, mem(rdx, rsi, 1))
	vmovlpd(xmm1, mem(rdx, rsi, 2))
	vmovhpd(xmm1, mem(rdx, rax, 1))
#endif

	lea(mem(rdx, rsi, 4), rdx)


	vunpcklpd(ymm7, ymm5, ymm0)
	vunpckhpd(ymm7, ymm5, ymm1)
	vunpcklpd(ymm11, ymm9, ymm2)
	vunpckhpd(ymm11, ymm9, ymm3)
	vinsertf128(imm(0x1), xmm2, ymm0, ymm5)
	vinsertf128(imm(0x1), xmm3, ymm1, ymm7)
	vperm2f128(imm(0x31), ymm2, ymm0, ymm9)
	vperm2f128(imm(0x31), ymm3, ymm1, ymm11)

	vbroadcastsd(mem(rbx), ymm3)

	vfmadd231pd(mem(rcx), ymm3, ymm5)
	vfmadd231pd(mem(rcx, rsi, 1), ymm3, ymm7)
	vfmadd231pd(mem(rcx, rsi, 2), ymm3, ymm9)
	vfmadd231pd(mem(rcx, rax, 1), ymm3, ymm11)
	vmovupd(ymm5, mem(rcx))
	vmovupd(ymm7, mem(rcx, rsi, 1))
	vmovupd(ymm9, mem(rcx, rsi, 2))
	vmovupd(ymm11, mem(rcx, rax, 1))

	//lea(mem(rcx, rsi, 4), rcx)

#if 0
	vunpcklpd(ymm15, ymm13, ymm0)
	vunpckhpd(ymm15, ymm13, ymm1)
	vextractf128(imm(0x1), ymm0, xmm2)
	vextractf128(imm(0x1), ymm1, xmm4)

	vfmadd231pd(mem(rdx), xmm3, xmm0)
	vfmadd231pd(mem(rdx, rsi, 1), xmm3, xmm1)
	vfmadd231pd(mem(rdx, rsi, 2), xmm3, xmm2)
	vfmadd231pd(mem(rdx, rax, 1), xmm3, xmm4)
	vmovupd(xmm0, mem(rdx))
	vmovupd(xmm1, mem(rdx, rsi, 1))
	vmovupd(xmm2, mem(rdx, rsi, 2))
	vmovupd(xmm4, mem(rdx, rax, 1))
#else
	vmovlpd(mem(rdx), xmm0, xmm0)
	vmovhpd(mem(rdx, rsi, 1), xmm0, xmm0)
	vmovlpd(mem(rdx, rsi, 2), xmm1, xmm1)
	vmovhpd(mem(rdx, rax, 1), xmm1, xmm1)
	vperm2f128(imm(0x20), ymm1, ymm0, ymm0)

	vfmadd213pd(ymm13, ymm3, ymm0)
	vextractf128(imm(1), ymm0, xmm1)
	vmovlpd(xmm0, mem(rdx))
	vmovhpd(xmm0, mem(rdx, rsi, 1))
	vmovlpd(xmm1, mem(rdx, rsi, 2))
	vmovhpd(xmm1, mem(rdx, rax, 1))
#endif

	//lea(mem(rdx, rsi, 4), rdx)


	jmp(.DDONE)                        // jump to end.
	
	
	
	
	label(.DBETAZERO)
	

	cmp(imm(8), rdi)                   // set ZF if (8*rs_c) == 8.
	jz(.DCOLSTORBZ)                    // jump to column storage case


	
	label(.DROWSTORBZ)
	
	
	vmovupd(ymm4, mem(rcx))
	vmovupd(ymm5, mem(rcx, rsi, 4))
	add(rdi, rcx)
	

	vmovupd(ymm6, mem(rcx))
	vmovupd(ymm7, mem(rcx, rsi, 4))
	add(rdi, rcx)
	
	
	vmovupd(ymm8, mem(rcx))
	vmovupd(ymm9, mem(rcx, rsi, 4))
	add(rdi, rcx)
	
	
	vmovupd(ymm10, mem(rcx))
	vmovupd(ymm11, mem(rcx, rsi, 4))
	add(rdi, rcx)
	
	
	vmovupd(ymm12, mem(rcx))
	vmovupd(ymm13, mem(rcx, rsi, 4))
	//add(rdi, rcx)
	
	
	jmp(.DDONE)                        // jump to end.



	label(.DCOLSTORBZ)


	vunpcklpd(ymm6, ymm4, ymm0)
	vunpckhpd(ymm6, ymm4, ymm1)
	vunpcklpd(ymm10, ymm8, ymm2)
	vunpckhpd(ymm10, ymm8, ymm3)
	vinsertf128(imm(0x1), xmm2, ymm0, ymm4)
	vinsertf128(imm(0x1), xmm3, ymm1, ymm6)
	vperm2f128(imm(0x31), ymm2, ymm0, ymm8)
	vperm2f128(imm(0x31), ymm3, ymm1, ymm10)

	vmovupd(ymm4, mem(rcx))
	vmovupd(ymm6, mem(rcx, rsi, 1))
	vmovupd(ymm8, mem(rcx, rsi, 2))
	vmovupd(ymm10, mem(rcx, rax, 1))

	lea(mem(rcx, rsi, 4), rcx)

#if 0
	vunpcklpd(ymm14, ymm12, ymm0)
	vunpckhpd(ymm14, ymm12, ymm1)
	vextractf128(imm(0x1), ymm0, xmm2)
	vextractf128(imm(0x1), ymm1, xmm4)

	vmovupd(xmm0, mem(rdx))
	vmovupd(xmm1, mem(rdx, rsi, 1))
	vmovupd(xmm2, mem(rdx, rsi, 2))
	vmovupd(xmm4, mem(rdx, rax, 1))
#else
	vmovupd(ymm12, ymm0)

	vextractf128(imm(1), ymm0, xmm1)
	vmovlpd(xmm0, mem(rcx))
	vmovhpd(xmm0, mem(rcx, rsi, 1))
	vmovlpd(xmm1, mem(rcx, rsi, 2))
	vmovhpd(xmm1, mem(rcx, rax, 1))
#endif

	lea(mem(rdx, rsi, 4), rdx)


	vunpcklpd(ymm7, ymm5, ymm0)
	vunpckhpd(ymm7, ymm5, ymm1)
	vunpcklpd(ymm11, ymm9, ymm2)
	vunpckhpd(ymm11, ymm9, ymm3)
	vinsertf128(imm(0x1), xmm2, ymm0, ymm5)
	vinsertf128(imm(0x1), xmm3, ymm1, ymm7)
	vperm2f128(imm(0x31), ymm2, ymm0, ymm9)
	vperm2f128(imm(0x31), ymm3, ymm1, ymm11)

	vmovupd(ymm5, mem(rcx))
	vmovupd(ymm7, mem(rcx, rsi, 1))
	vmovupd(ymm9, mem(rcx, rsi, 2))
	vmovupd(ymm11, mem(rcx, rax, 1))

	//lea(mem(rcx, rsi, 4), rcx)

#if 0
	vunpcklpd(ymm15, ymm13, ymm0)
	vunpckhpd(ymm15, ymm13, ymm1)
	vextractf128(imm(0x1), ymm0, xmm2)
	vextractf128(imm(0x1), ymm1, xmm4)

	vmovupd(xmm0, mem(rdx))
	vmovupd(xmm1, mem(rdx, rsi, 1))
	vmovupd(xmm2, mem(rdx, rsi, 2))
	vmovupd(xmm4, mem(rdx, rax, 1))
#else
	vmovupd(ymm13, ymm0)

	vextractf128(imm(1), ymm0, xmm1)
	vmovlpd(xmm0, mem(rdx))
	vmovhpd(xmm0, mem(rdx, rsi, 1))
	vmovlpd(xmm1, mem(rdx, rsi, 2))
	vmovhpd(xmm1, mem(rdx, rax, 1))
#endif

	//lea(mem(rdx, rsi, 4), rdx)

	
	
	
	label(.DDONE)



    end_asm(
	: // output operands (none)
	: // input operands
      [k_iter] "m" (k_iter),
      [k_left] "m" (k_left),
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
	  "ymm8", "ymm9", "ymm10", "ymm11", "ymm12", "ymm13", "memory"
	)
	AOCL_DTL_TRACE_EXIT(AOCL_DTL_LEVEL_TRACE_7);
}

void bli_dgemmsup_rv_haswell_asm_4x8
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
	AOCL_DTL_TRACE_ENTRY(AOCL_DTL_LEVEL_TRACE_7);
	//void*    a_next = bli_auxinfo_next_a( data );
	//void*    b_next = bli_auxinfo_next_b( data );

	// Typecast local copies of integers in case dim_t and inc_t are a
	// different size than is expected by load instructions.
	uint64_t k_iter = k0 / 4;
	uint64_t k_left = k0 % 4;

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
	mov(var(cs_a), r9)                 // load cs_a
	lea(mem(, r8, 8), r8)              // rs_a *= sizeof(double)
	lea(mem(, r9, 8), r9)              // cs_a *= sizeof(double)

	lea(mem(r8, r8, 2), r13)           // r13 = 3*rs_a
	//lea(mem(r8, r8, 4), r15)           // r15 = 5*rs_a

	mov(var(b), rbx)                   // load address of b.
	mov(var(rs_b), r10)                // load rs_b
	//mov(var(cs_b), r11)                // load cs_b
	lea(mem(, r10, 8), r10)            // rs_b *= sizeof(double)
	//lea(mem(, r11, 8), r11)            // cs_b *= sizeof(double)
	
	                                   // NOTE: We cannot pre-load elements of a or b
	                                   // because it could eventually, in the last
	                                   // unrolled iter or the cleanup loop, result
	                                   // in reading beyond the bounds allocated mem
	                                   // (the likely result: a segmentation fault).

	mov(var(c), rcx)                   // load address of c
	mov(var(rs_c), rdi)                // load rs_c
	lea(mem(, rdi, 8), rdi)            // rs_c *= sizeof(double)

#if 0
	lea(mem(rcx, rdi, 2), rdx)         //
	lea(mem(rdx, rdi, 1), rdx)         // rdx = c + 3*rs_c;
	prefetch(0, mem(rcx, 7*8))         // prefetch c + 0*rs_c
	prefetch(0, mem(rcx, rdi, 1, 7*8)) // prefetch c + 1*rs_c
	prefetch(0, mem(rcx, rdi, 2, 7*8)) // prefetch c + 2*rs_c
	prefetch(0, mem(rdx, 7*8))         // prefetch c + 3*rs_c
#else
	cmp(imm(8), rdi)                   // set ZF if (8*rs_c) == 8.
	jz(.DCOLPFETCH)                    // jump to column storage case
	label(.DROWPFETCH)                 // row-stored prefetching on c

	lea(mem(rcx, rdi, 2), rdx)         //
	lea(mem(rdx, rdi, 1), rdx)         // rdx = c + 3*rs_c;
	prefetch(0, mem(rcx, 7*8))         // prefetch c + 0*rs_c
	prefetch(0, mem(rcx, rdi, 1, 7*8)) // prefetch c + 1*rs_c
	prefetch(0, mem(rcx, rdi, 2, 7*8)) // prefetch c + 2*rs_c
	prefetch(0, mem(rdx, 7*8))         // prefetch c + 3*rs_c

	jmp(.DPOSTPFETCH)                  // jump to end of prefetching c
	label(.DCOLPFETCH)                 // column-stored prefetching c

	mov(var(cs_c), rsi)                // load cs_c to rsi (temporarily)
	lea(mem(, rsi, 8), rsi)            // cs_c *= sizeof(double)
	lea(mem(rcx, rsi, 2), rdx)         //
	lea(mem(rdx, rsi, 1), rdx)         // rdx = c + 3*cs_c;
	prefetch(0, mem(rcx, 3*8))         // prefetch c + 0*cs_c
	prefetch(0, mem(rcx, rsi, 1, 3*8)) // prefetch c + 1*cs_c
	prefetch(0, mem(rcx, rsi, 2, 3*8)) // prefetch c + 2*cs_c
	prefetch(0, mem(rdx, 3*8))         // prefetch c + 3*cs_c
	prefetch(0, mem(rdx, rsi, 1, 3*8)) // prefetch c + 4*cs_c
	prefetch(0, mem(rdx, rsi, 2, 3*8)) // prefetch c + 5*cs_c
	lea(mem(rdx, rsi, 2), rdx)         // rdx = c + 5*cs_c;
	prefetch(0, mem(rdx, rsi, 1, 3*8)) // prefetch c + 6*cs_c
	prefetch(0, mem(rdx, rsi, 2, 3*8)) // prefetch c + 7*cs_c

	label(.DPOSTPFETCH)                // done prefetching c
#endif
	
	
	
	
	mov(var(k_iter), rsi)              // i = k_iter;
	test(rsi, rsi)                     // check i via logical AND.
	je(.DCONSIDKLEFT)                  // if i == 0, jump to code that
	                                   // contains the k_left loop.
	
	
	label(.DLOOPKITER)                 // MAIN LOOP
	
	
	// ---------------------------------- iteration 0
	
	vmovupd(mem(rbx, 0*32), ymm0)
	vmovupd(mem(rbx, 1*32), ymm1)
	add(r10, rbx)                      // b += rs_b;

	vbroadcastsd(mem(rax        ), ymm2)
	vbroadcastsd(mem(rax, r8,  1), ymm3)
	vfmadd231pd(ymm0, ymm2, ymm4)
	vfmadd231pd(ymm1, ymm2, ymm5)
	vfmadd231pd(ymm0, ymm3, ymm6)
	vfmadd231pd(ymm1, ymm3, ymm7)
	
	vbroadcastsd(mem(rax, r8,  2), ymm2)
	vbroadcastsd(mem(rax, r13, 1), ymm3)
	add(r9, rax)                       // a += cs_a;
	vfmadd231pd(ymm0, ymm2, ymm8)
	vfmadd231pd(ymm1, ymm2, ymm9)
	vfmadd231pd(ymm0, ymm3, ymm10)
	vfmadd231pd(ymm1, ymm3, ymm11)
	
	
	// ---------------------------------- iteration 1

	vmovupd(mem(rbx, 0*32), ymm0)
	vmovupd(mem(rbx, 1*32), ymm1)
	add(r10, rbx)                      // b += rs_b;

	vbroadcastsd(mem(rax        ), ymm2)
	vbroadcastsd(mem(rax, r8,  1), ymm3)
	vfmadd231pd(ymm0, ymm2, ymm4)
	vfmadd231pd(ymm1, ymm2, ymm5)
	vfmadd231pd(ymm0, ymm3, ymm6)
	vfmadd231pd(ymm1, ymm3, ymm7)
	
	vbroadcastsd(mem(rax, r8,  2), ymm2)
	vbroadcastsd(mem(rax, r13, 1), ymm3)
	add(r9, rax)                       // a += cs_a;
	vfmadd231pd(ymm0, ymm2, ymm8)
	vfmadd231pd(ymm1, ymm2, ymm9)
	vfmadd231pd(ymm0, ymm3, ymm10)
	vfmadd231pd(ymm1, ymm3, ymm11)
	

	// ---------------------------------- iteration 2
	
	vmovupd(mem(rbx, 0*32), ymm0)
	vmovupd(mem(rbx, 1*32), ymm1)
	add(r10, rbx)                      // b += rs_b;

	vbroadcastsd(mem(rax        ), ymm2)
	vbroadcastsd(mem(rax, r8,  1), ymm3)
	vfmadd231pd(ymm0, ymm2, ymm4)
	vfmadd231pd(ymm1, ymm2, ymm5)
	vfmadd231pd(ymm0, ymm3, ymm6)
	vfmadd231pd(ymm1, ymm3, ymm7)
	
	vbroadcastsd(mem(rax, r8,  2), ymm2)
	vbroadcastsd(mem(rax, r13, 1), ymm3)
	add(r9, rax)                       // a += cs_a;
	vfmadd231pd(ymm0, ymm2, ymm8)
	vfmadd231pd(ymm1, ymm2, ymm9)
	vfmadd231pd(ymm0, ymm3, ymm10)
	vfmadd231pd(ymm1, ymm3, ymm11)
	

	// ---------------------------------- iteration 3

	vmovupd(mem(rbx, 0*32), ymm0)
	vmovupd(mem(rbx, 1*32), ymm1)
	add(r10, rbx)                      // b += rs_b;

	vbroadcastsd(mem(rax        ), ymm2)
	vbroadcastsd(mem(rax, r8,  1), ymm3)
	vfmadd231pd(ymm0, ymm2, ymm4)
	vfmadd231pd(ymm1, ymm2, ymm5)
	vfmadd231pd(ymm0, ymm3, ymm6)
	vfmadd231pd(ymm1, ymm3, ymm7)
	
	vbroadcastsd(mem(rax, r8,  2), ymm2)
	vbroadcastsd(mem(rax, r13, 1), ymm3)
	add(r9, rax)                       // a += cs_a;
	vfmadd231pd(ymm0, ymm2, ymm8)
	vfmadd231pd(ymm1, ymm2, ymm9)
	vfmadd231pd(ymm0, ymm3, ymm10)
	vfmadd231pd(ymm1, ymm3, ymm11)
	
	
	
	dec(rsi)                           // i -= 1;
	jne(.DLOOPKITER)                   // iterate again if i != 0.
	
	
	
	
	
	
	label(.DCONSIDKLEFT)
	
	mov(var(k_left), rsi)              // i = k_left;
	test(rsi, rsi)                     // check i via logical AND.
	je(.DPOSTACCUM)                    // if i == 0, we're done; jump to end.
	                                   // else, we prepare to enter k_left loop.
	
	
	label(.DLOOPKLEFT)                 // EDGE LOOP
	
	vmovupd(mem(rbx, 0*32), ymm0)
	vmovupd(mem(rbx, 1*32), ymm1)
	add(r10, rbx)                      // b += rs_b;

	vbroadcastsd(mem(rax        ), ymm2)
	vbroadcastsd(mem(rax, r8,  1), ymm3)
	vfmadd231pd(ymm0, ymm2, ymm4)
	vfmadd231pd(ymm1, ymm2, ymm5)
	vfmadd231pd(ymm0, ymm3, ymm6)
	vfmadd231pd(ymm1, ymm3, ymm7)
	
	vbroadcastsd(mem(rax, r8,  2), ymm2)
	vbroadcastsd(mem(rax, r13, 1), ymm3)
	add(r9, rax)                       // a += cs_a;
	vfmadd231pd(ymm0, ymm2, ymm8)
	vfmadd231pd(ymm1, ymm2, ymm9)
	vfmadd231pd(ymm0, ymm3, ymm10)
	vfmadd231pd(ymm1, ymm3, ymm11)
	
	
	dec(rsi)                           // i -= 1;
	jne(.DLOOPKLEFT)                   // iterate again if i != 0.
	
	
	
	label(.DPOSTACCUM)


	
	mov(var(alpha), rax)               // load address of alpha
	mov(var(beta), rbx)                // load address of beta
	vbroadcastsd(mem(rax), ymm0)       // load alpha and duplicate
	vbroadcastsd(mem(rbx), ymm3)       // load beta and duplicate
	
	vmulpd(ymm0, ymm4, ymm4)           // scale by alpha
	vmulpd(ymm0, ymm5, ymm5)
	vmulpd(ymm0, ymm6, ymm6)
	vmulpd(ymm0, ymm7, ymm7)
	vmulpd(ymm0, ymm8, ymm8)
	vmulpd(ymm0, ymm9, ymm9)
	vmulpd(ymm0, ymm10, ymm10)
	vmulpd(ymm0, ymm11, ymm11)
	
	
	
	
	
	
	mov(var(cs_c), rsi)                // load cs_c
	lea(mem(, rsi, 8), rsi)            // rsi = cs_c * sizeof(double)
	
	//lea(mem(rcx, rsi, 4), rdx)         // load address of c +  4*cs_c;
	//lea(mem(rcx, rdi, 4), rdx)         // load address of c +  4*rs_c;

	lea(mem(rsi, rsi, 2), rax)         // rax = 3*cs_c;
	
	
	
	                                   // now avoid loading C if beta == 0
	
	vxorpd(ymm0, ymm0, ymm0)           // set ymm0 to zero.
	vucomisd(xmm0, xmm3)               // set ZF if beta == 0.
	je(.DBETAZERO)                     // if ZF = 1, jump to beta == 0 case
	


	cmp(imm(8), rdi)                   // set ZF if (8*rs_c) == 8.
	jz(.DCOLSTORED)                    // jump to column storage case


	
	label(.DROWSTORED)
	
	
	vfmadd231pd(mem(rcx), ymm3, ymm4)
	vmovupd(ymm4, mem(rcx))

	vfmadd231pd(mem(rcx, rsi, 4), ymm3, ymm5)
	vmovupd(ymm5, mem(rcx, rsi, 4))
	add(rdi, rcx)


	vfmadd231pd(mem(rcx), ymm3, ymm6)
	vmovupd(ymm6, mem(rcx))

	vfmadd231pd(mem(rcx, rsi, 4), ymm3, ymm7)
	vmovupd(ymm7, mem(rcx, rsi, 4))
	add(rdi, rcx)


	vfmadd231pd(mem(rcx), ymm3, ymm8)
	vmovupd(ymm8, mem(rcx))

	vfmadd231pd(mem(rcx, rsi, 4), ymm3, ymm9)
	vmovupd(ymm9, mem(rcx, rsi, 4))
	add(rdi, rcx)


	vfmadd231pd(mem(rcx), ymm3, ymm10)
	vmovupd(ymm10, mem(rcx))

	vfmadd231pd(mem(rcx, rsi, 4), ymm3, ymm11)
	vmovupd(ymm11, mem(rcx, rsi, 4))
	//add(rdi, rcx)
	
	
	jmp(.DDONE)                        // jump to end.
	


	label(.DCOLSTORED)


	vunpcklpd(ymm6, ymm4, ymm0)
	vunpckhpd(ymm6, ymm4, ymm1)
	vunpcklpd(ymm10, ymm8, ymm2)
	vunpckhpd(ymm10, ymm8, ymm3)
	vinsertf128(imm(0x1), xmm2, ymm0, ymm4)
	vinsertf128(imm(0x1), xmm3, ymm1, ymm6)
	vperm2f128(imm(0x31), ymm2, ymm0, ymm8)
	vperm2f128(imm(0x31), ymm3, ymm1, ymm10)

	vbroadcastsd(mem(rbx), ymm3)

	vfmadd231pd(mem(rcx), ymm3, ymm4)
	vfmadd231pd(mem(rcx, rsi, 1), ymm3, ymm6)
	vfmadd231pd(mem(rcx, rsi, 2), ymm3, ymm8)
	vfmadd231pd(mem(rcx, rax, 1), ymm3, ymm10)
	vmovupd(ymm4, mem(rcx))
	vmovupd(ymm6, mem(rcx, rsi, 1))
	vmovupd(ymm8, mem(rcx, rsi, 2))
	vmovupd(ymm10, mem(rcx, rax, 1))

	lea(mem(rcx, rsi, 4), rcx)


	vunpcklpd(ymm7, ymm5, ymm0)
	vunpckhpd(ymm7, ymm5, ymm1)
	vunpcklpd(ymm11, ymm9, ymm2)
	vunpckhpd(ymm11, ymm9, ymm3)
	vinsertf128(imm(0x1), xmm2, ymm0, ymm5)
	vinsertf128(imm(0x1), xmm3, ymm1, ymm7)
	vperm2f128(imm(0x31), ymm2, ymm0, ymm9)
	vperm2f128(imm(0x31), ymm3, ymm1, ymm11)

	vbroadcastsd(mem(rbx), ymm3)

	vfmadd231pd(mem(rcx), ymm3, ymm5)
	vfmadd231pd(mem(rcx, rsi, 1), ymm3, ymm7)
	vfmadd231pd(mem(rcx, rsi, 2), ymm3, ymm9)
	vfmadd231pd(mem(rcx, rax, 1), ymm3, ymm11)
	vmovupd(ymm5, mem(rcx))
	vmovupd(ymm7, mem(rcx, rsi, 1))
	vmovupd(ymm9, mem(rcx, rsi, 2))
	vmovupd(ymm11, mem(rcx, rax, 1))

	//lea(mem(rcx, rsi, 4), rcx)


	jmp(.DDONE)                        // jump to end.


	
	
	label(.DBETAZERO)
	

	cmp(imm(8), rdi)                   // set ZF if (8*rs_c) == 8.
	jz(.DCOLSTORBZ)                    // jump to column storage case


	
	label(.DROWSTORBZ)
	
	
	vmovupd(ymm4, mem(rcx))
	vmovupd(ymm5, mem(rcx, rsi, 4))
	add(rdi, rcx)

	vmovupd(ymm6, mem(rcx))
	vmovupd(ymm7, mem(rcx, rsi, 4))
	add(rdi, rcx)

	vmovupd(ymm8, mem(rcx))
	vmovupd(ymm9, mem(rcx, rsi, 4))
	add(rdi, rcx)

	vmovupd(ymm10, mem(rcx))
	vmovupd(ymm11, mem(rcx, rsi, 4))
	//add(rdi, rcx)
	
	
	jmp(.DDONE)                        // jump to end.



	label(.DCOLSTORBZ)


	vunpcklpd(ymm6, ymm4, ymm0)
	vunpckhpd(ymm6, ymm4, ymm1)
	vunpcklpd(ymm10, ymm8, ymm2)
	vunpckhpd(ymm10, ymm8, ymm3)
	vinsertf128(imm(0x1), xmm2, ymm0, ymm4)
	vinsertf128(imm(0x1), xmm3, ymm1, ymm6)
	vperm2f128(imm(0x31), ymm2, ymm0, ymm8)
	vperm2f128(imm(0x31), ymm3, ymm1, ymm10)

	vmovupd(ymm4, mem(rcx))
	vmovupd(ymm6, mem(rcx, rsi, 1))
	vmovupd(ymm8, mem(rcx, rsi, 2))
	vmovupd(ymm10, mem(rcx, rax, 1))

	lea(mem(rcx, rsi, 4), rcx)


	vunpcklpd(ymm7, ymm5, ymm0)
	vunpckhpd(ymm7, ymm5, ymm1)
	vunpcklpd(ymm11, ymm9, ymm2)
	vunpckhpd(ymm11, ymm9, ymm3)
	vinsertf128(imm(0x1), xmm2, ymm0, ymm5)
	vinsertf128(imm(0x1), xmm3, ymm1, ymm7)
	vperm2f128(imm(0x31), ymm2, ymm0, ymm9)
	vperm2f128(imm(0x31), ymm3, ymm1, ymm11)

	vmovupd(ymm5, mem(rcx))
	vmovupd(ymm7, mem(rcx, rsi, 1))
	vmovupd(ymm9, mem(rcx, rsi, 2))
	vmovupd(ymm11, mem(rcx, rax, 1))

	//lea(mem(rcx, rsi, 4), rcx)

	
	
	
	label(.DDONE)
	
	

    end_asm(
	: // output operands (none)
	: // input operands
      [k_iter] "m" (k_iter),
      [k_left] "m" (k_left),
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
	  "ymm8", "ymm9", "ymm10", "ymm11", "memory"
	)
	AOCL_DTL_TRACE_EXIT(AOCL_DTL_LEVEL_TRACE_7);
}

void bli_dgemmsup_rv_haswell_asm_3x8
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
	AOCL_DTL_TRACE_ENTRY(AOCL_DTL_LEVEL_TRACE_7);
	//void*    a_next = bli_auxinfo_next_a( data );
	//void*    b_next = bli_auxinfo_next_b( data );

	// Typecast local copies of integers in case dim_t and inc_t are a
	// different size than is expected by load instructions.
	uint64_t k_iter = k0 / 4;
	uint64_t k_left = k0 % 4;

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
	mov(var(cs_a), r9)                 // load cs_a
	lea(mem(, r8, 8), r8)              // rs_a *= sizeof(double)
	lea(mem(, r9, 8), r9)              // cs_a *= sizeof(double)

	//lea(mem(r8, r8, 2), r13)           // r13 = 3*rs_a
	//lea(mem(r8, r8, 4), r15)           // r15 = 5*rs_a

	mov(var(b), rbx)                   // load address of b.
	mov(var(rs_b), r10)                // load rs_b
	//mov(var(cs_b), r11)                // load cs_b
	lea(mem(, r10, 8), r10)            // rs_b *= sizeof(double)
	//lea(mem(, r11, 8), r11)            // cs_b *= sizeof(double)
	
	                                   // NOTE: We cannot pre-load elements of a or b
	                                   // because it could eventually, in the last
	                                   // unrolled iter or the cleanup loop, result
	                                   // in reading beyond the bounds allocated mem
	                                   // (the likely result: a segmentation fault).

	mov(var(c), rcx)                   // load address of c
	mov(var(rs_c), rdi)                // load rs_c
	lea(mem(, rdi, 8), rdi)            // rs_c *= sizeof(double)

#if 0
	lea(mem(rcx, rdi, 2), rdx)         //
	lea(mem(rdx, rdi, 1), rdx)         // rdx = c + 3*rs_c;
	prefetch(0, mem(rcx, 7*8))         // prefetch c + 0*rs_c
	prefetch(0, mem(rcx, rdi, 1, 7*8)) // prefetch c + 1*rs_c
	prefetch(0, mem(rcx, rdi, 2, 7*8)) // prefetch c + 2*rs_c
	prefetch(0, mem(rdx, 7*8))         // prefetch c + 3*rs_c
#else
	cmp(imm(8), rdi)                   // set ZF if (8*rs_c) == 8.
	jz(.DCOLPFETCH)                    // jump to column storage case
	label(.DROWPFETCH)                 // row-stored prefetching on c

	//lea(mem(rcx, rdi, 2), rdx)         //
	//lea(mem(rdx, rdi, 1), rdx)         // rdx = c + 3*rs_c;
	prefetch(0, mem(rcx, 7*8))         // prefetch c + 0*rs_c
	prefetch(0, mem(rcx, rdi, 1, 7*8)) // prefetch c + 1*rs_c
	prefetch(0, mem(rcx, rdi, 2, 7*8)) // prefetch c + 2*rs_c

	jmp(.DPOSTPFETCH)                  // jump to end of prefetching c
	label(.DCOLPFETCH)                 // column-stored prefetching c

	mov(var(cs_c), rsi)                // load cs_c to rsi (temporarily)
	lea(mem(, rsi, 8), rsi)            // cs_c *= sizeof(double)
	lea(mem(rcx, rsi, 2), rdx)         //
	lea(mem(rdx, rsi, 1), rdx)         // rdx = c + 3*cs_c;
	prefetch(0, mem(rcx, 2*8))         // prefetch c + 0*cs_c
	prefetch(0, mem(rcx, rsi, 1, 2*8)) // prefetch c + 1*cs_c
	prefetch(0, mem(rcx, rsi, 2, 2*8)) // prefetch c + 2*cs_c
	prefetch(0, mem(rdx, 2*8))         // prefetch c + 3*cs_c
	prefetch(0, mem(rdx, rsi, 1, 2*8)) // prefetch c + 4*cs_c
	prefetch(0, mem(rdx, rsi, 2, 2*8)) // prefetch c + 5*cs_c
	lea(mem(rdx, rsi, 2), rdx)         // rdx = c + 5*cs_c;
	prefetch(0, mem(rdx, rsi, 1, 2*8)) // prefetch c + 6*cs_c
	prefetch(0, mem(rdx, rsi, 2, 2*8)) // prefetch c + 7*cs_c

	label(.DPOSTPFETCH)                // done prefetching c
#endif
	
	
	
	
	mov(var(k_iter), rsi)              // i = k_iter;
	test(rsi, rsi)                     // check i via logical AND.
	je(.DCONSIDKLEFT)                  // if i == 0, jump to code that
	                                   // contains the k_left loop.
	
	
	label(.DLOOPKITER)                 // MAIN LOOP
	
	
	// ---------------------------------- iteration 0
	
	vmovupd(mem(rbx, 0*32), ymm0)
	vmovupd(mem(rbx, 1*32), ymm1)
	add(r10, rbx)                      // b += rs_b;

	vbroadcastsd(mem(rax        ), ymm2)
	vbroadcastsd(mem(rax, r8,  1), ymm3)
	vfmadd231pd(ymm0, ymm2, ymm4)
	vfmadd231pd(ymm1, ymm2, ymm5)
	vfmadd231pd(ymm0, ymm3, ymm6)
	vfmadd231pd(ymm1, ymm3, ymm7)
	
	vbroadcastsd(mem(rax, r8,  2), ymm2)
	add(r9, rax)                       // a += cs_a;
	vfmadd231pd(ymm0, ymm2, ymm8)
	vfmadd231pd(ymm1, ymm2, ymm9)
	
	
	// ---------------------------------- iteration 1

	vmovupd(mem(rbx, 0*32), ymm0)
	vmovupd(mem(rbx, 1*32), ymm1)
	add(r10, rbx)                      // b += rs_b;

	vbroadcastsd(mem(rax        ), ymm2)
	vbroadcastsd(mem(rax, r8,  1), ymm3)
	vfmadd231pd(ymm0, ymm2, ymm4)
	vfmadd231pd(ymm1, ymm2, ymm5)
	vfmadd231pd(ymm0, ymm3, ymm6)
	vfmadd231pd(ymm1, ymm3, ymm7)
	
	vbroadcastsd(mem(rax, r8,  2), ymm2)
	add(r9, rax)                       // a += cs_a;
	vfmadd231pd(ymm0, ymm2, ymm8)
	vfmadd231pd(ymm1, ymm2, ymm9)
	

	// ---------------------------------- iteration 2
	
	vmovupd(mem(rbx, 0*32), ymm0)
	vmovupd(mem(rbx, 1*32), ymm1)
	add(r10, rbx)                      // b += rs_b;

	vbroadcastsd(mem(rax        ), ymm2)
	vbroadcastsd(mem(rax, r8,  1), ymm3)
	vfmadd231pd(ymm0, ymm2, ymm4)
	vfmadd231pd(ymm1, ymm2, ymm5)
	vfmadd231pd(ymm0, ymm3, ymm6)
	vfmadd231pd(ymm1, ymm3, ymm7)
	
	vbroadcastsd(mem(rax, r8,  2), ymm2)
	add(r9, rax)                       // a += cs_a;
	vfmadd231pd(ymm0, ymm2, ymm8)
	vfmadd231pd(ymm1, ymm2, ymm9)
	

	// ---------------------------------- iteration 3

	vmovupd(mem(rbx, 0*32), ymm0)
	vmovupd(mem(rbx, 1*32), ymm1)
	add(r10, rbx)                      // b += rs_b;

	vbroadcastsd(mem(rax        ), ymm2)
	vbroadcastsd(mem(rax, r8,  1), ymm3)
	vfmadd231pd(ymm0, ymm2, ymm4)
	vfmadd231pd(ymm1, ymm2, ymm5)
	vfmadd231pd(ymm0, ymm3, ymm6)
	vfmadd231pd(ymm1, ymm3, ymm7)
	
	vbroadcastsd(mem(rax, r8,  2), ymm2)
	add(r9, rax)                       // a += cs_a;
	vfmadd231pd(ymm0, ymm2, ymm8)
	vfmadd231pd(ymm1, ymm2, ymm9)
	
	
	
	dec(rsi)                           // i -= 1;
	jne(.DLOOPKITER)                   // iterate again if i != 0.
	
	
	
	
	
	
	label(.DCONSIDKLEFT)
	
	mov(var(k_left), rsi)              // i = k_left;
	test(rsi, rsi)                     // check i via logical AND.
	je(.DPOSTACCUM)                    // if i == 0, we're done; jump to end.
	                                   // else, we prepare to enter k_left loop.
	
	
	label(.DLOOPKLEFT)                 // EDGE LOOP
	
	vmovupd(mem(rbx, 0*32), ymm0)
	vmovupd(mem(rbx, 1*32), ymm1)
	add(r10, rbx)                      // b += rs_b;

	vbroadcastsd(mem(rax        ), ymm2)
	vbroadcastsd(mem(rax, r8,  1), ymm3)
	vfmadd231pd(ymm0, ymm2, ymm4)
	vfmadd231pd(ymm1, ymm2, ymm5)
	vfmadd231pd(ymm0, ymm3, ymm6)
	vfmadd231pd(ymm1, ymm3, ymm7)
	
	vbroadcastsd(mem(rax, r8,  2), ymm2)
	add(r9, rax)                       // a += cs_a;
	vfmadd231pd(ymm0, ymm2, ymm8)
	vfmadd231pd(ymm1, ymm2, ymm9)
	
	
	dec(rsi)                           // i -= 1;
	jne(.DLOOPKLEFT)                   // iterate again if i != 0.
	
	
	
	label(.DPOSTACCUM)


	
	mov(var(alpha), rax)               // load address of alpha
	mov(var(beta), rbx)                // load address of beta
	vbroadcastsd(mem(rax), ymm0)       // load alpha and duplicate
	vbroadcastsd(mem(rbx), ymm3)       // load beta and duplicate
	
	vmulpd(ymm0, ymm4, ymm4)           // scale by alpha
	vmulpd(ymm0, ymm5, ymm5)
	vmulpd(ymm0, ymm6, ymm6)
	vmulpd(ymm0, ymm7, ymm7)
	vmulpd(ymm0, ymm8, ymm8)
	vmulpd(ymm0, ymm9, ymm9)
	
	
	
	
	
	
	mov(var(cs_c), rsi)                // load cs_c
	lea(mem(, rsi, 8), rsi)            // rsi = cs_c * sizeof(double)
	
	//lea(mem(rcx, rsi, 4), rdx)         // load address of c +  4*cs_c;
	lea(mem(rcx, rdi, 2), rdx)         // load address of c +  2*rs_c;

	lea(mem(rsi, rsi, 2), rax)         // rax = 3*cs_c;
	
	
	
	                                   // now avoid loading C if beta == 0
	
	vxorpd(ymm0, ymm0, ymm0)           // set ymm0 to zero.
	vucomisd(xmm0, xmm3)               // set ZF if beta == 0.
	je(.DBETAZERO)                     // if ZF = 1, jump to beta == 0 case
	


	cmp(imm(8), rdi)                   // set ZF if (8*rs_c) == 8.
	jz(.DCOLSTORED)                    // jump to column storage case


	
	label(.DROWSTORED)
	
	
	vfmadd231pd(mem(rcx), ymm3, ymm4)
	vmovupd(ymm4, mem(rcx))

	vfmadd231pd(mem(rcx, rsi, 4), ymm3, ymm5)
	vmovupd(ymm5, mem(rcx, rsi, 4))
	add(rdi, rcx)


	vfmadd231pd(mem(rcx), ymm3, ymm6)
	vmovupd(ymm6, mem(rcx))

	vfmadd231pd(mem(rcx, rsi, 4), ymm3, ymm7)
	vmovupd(ymm7, mem(rcx, rsi, 4))
	add(rdi, rcx)


	vfmadd231pd(mem(rcx), ymm3, ymm8)
	vmovupd(ymm8, mem(rcx))

	vfmadd231pd(mem(rcx, rsi, 4), ymm3, ymm9)
	vmovupd(ymm9, mem(rcx, rsi, 4))
	//add(rdi, rcx)
	
	
	jmp(.DDONE)                        // jump to end.
	


	label(.DCOLSTORED)


	vunpcklpd(ymm6, ymm4, ymm0)
	vunpckhpd(ymm6, ymm4, ymm1)
	vunpcklpd(ymm10, ymm8, ymm2)
	vunpckhpd(ymm10, ymm8, ymm3)
	vinsertf128(imm(0x1), xmm2, ymm0, ymm4)
	vinsertf128(imm(0x1), xmm3, ymm1, ymm6)
	vperm2f128(imm(0x31), ymm2, ymm0, ymm8)
	vperm2f128(imm(0x31), ymm3, ymm1, ymm10)

	vextractf128(imm(0x1), ymm4, xmm12)
	vextractf128(imm(0x1), ymm6, xmm13)
	vextractf128(imm(0x1), ymm8, xmm14)
	vextractf128(imm(0x1), ymm10, xmm15)

	vbroadcastsd(mem(rbx), ymm3)

	vfmadd231pd(mem(rcx), xmm3, xmm4)
	vfmadd231pd(mem(rcx, rsi, 1), xmm3, xmm6)
	vfmadd231pd(mem(rcx, rsi, 2), xmm3, xmm8)
	vfmadd231pd(mem(rcx, rax, 1), xmm3, xmm10)
	vmovupd(xmm4, mem(rcx))
	vmovupd(xmm6, mem(rcx, rsi, 1))
	vmovupd(xmm8, mem(rcx, rsi, 2))
	vmovupd(xmm10, mem(rcx, rax, 1))

	lea(mem(rcx, rsi, 4), rcx)

	vfmadd231sd(mem(rdx), xmm3, xmm12)
	vfmadd231sd(mem(rdx, rsi, 1), xmm3, xmm13)
	vfmadd231sd(mem(rdx, rsi, 2), xmm3, xmm14)
	vfmadd231sd(mem(rdx, rax, 1), xmm3, xmm15)
	vmovsd(xmm12, mem(rdx))
	vmovsd(xmm13, mem(rdx, rsi, 1))
	vmovsd(xmm14, mem(rdx, rsi, 2))
	vmovsd(xmm15, mem(rdx, rax, 1))
	
	lea(mem(rdx, rsi, 4), rdx)



	vunpcklpd(ymm7, ymm5, ymm0)
	vunpckhpd(ymm7, ymm5, ymm1)
	vunpcklpd(ymm11, ymm9, ymm2)
	vunpckhpd(ymm11, ymm9, ymm3)
	vinsertf128(imm(0x1), xmm2, ymm0, ymm5)
	vinsertf128(imm(0x1), xmm3, ymm1, ymm7)
	vperm2f128(imm(0x31), ymm2, ymm0, ymm9)
	vperm2f128(imm(0x31), ymm3, ymm1, ymm11)

	vextractf128(imm(0x1), ymm5, xmm12)
	vextractf128(imm(0x1), ymm7, xmm13)
	vextractf128(imm(0x1), ymm9, xmm14)
	vextractf128(imm(0x1), ymm11, xmm15)

	vbroadcastsd(mem(rbx), ymm3)

	vfmadd231pd(mem(rcx), xmm3, xmm5)
	vfmadd231pd(mem(rcx, rsi, 1), xmm3, xmm7)
	vfmadd231pd(mem(rcx, rsi, 2), xmm3, xmm9)
	vfmadd231pd(mem(rcx, rax, 1), xmm3, xmm11)
	vmovupd(xmm5, mem(rcx))
	vmovupd(xmm7, mem(rcx, rsi, 1))
	vmovupd(xmm9, mem(rcx, rsi, 2))
	vmovupd(xmm11, mem(rcx, rax, 1))

	//lea(mem(rcx, rsi, 4), rcx)

	vfmadd231sd(mem(rdx), xmm3, xmm12)
	vfmadd231sd(mem(rdx, rsi, 1), xmm3, xmm13)
	vfmadd231sd(mem(rdx, rsi, 2), xmm3, xmm14)
	vfmadd231sd(mem(rdx, rax, 1), xmm3, xmm15)
	vmovsd(xmm12, mem(rdx))
	vmovsd(xmm13, mem(rdx, rsi, 1))
	vmovsd(xmm14, mem(rdx, rsi, 2))
	vmovsd(xmm15, mem(rdx, rax, 1))
	
	//lea(mem(rdx, rsi, 4), rdx)


	jmp(.DDONE)                        // jump to end.


	
	
	label(.DBETAZERO)
	

	cmp(imm(8), rdi)                   // set ZF if (8*rs_c) == 8.
	jz(.DCOLSTORBZ)                    // jump to column storage case


	
	label(.DROWSTORBZ)
	
	
	vmovupd(ymm4, mem(rcx))
	vmovupd(ymm5, mem(rcx, rsi, 4))
	add(rdi, rcx)

	vmovupd(ymm6, mem(rcx))
	vmovupd(ymm7, mem(rcx, rsi, 4))
	add(rdi, rcx)

	vmovupd(ymm8, mem(rcx))
	vmovupd(ymm9, mem(rcx, rsi, 4))
	//add(rdi, rcx)
	
	
	jmp(.DDONE)                        // jump to end.



	label(.DCOLSTORBZ)


	vunpcklpd(ymm6, ymm4, ymm0)
	vunpckhpd(ymm6, ymm4, ymm1)
	vunpcklpd(ymm10, ymm8, ymm2)
	vunpckhpd(ymm10, ymm8, ymm3)
	vinsertf128(imm(0x1), xmm2, ymm0, ymm4)
	vinsertf128(imm(0x1), xmm3, ymm1, ymm6)
	vperm2f128(imm(0x31), ymm2, ymm0, ymm8)
	vperm2f128(imm(0x31), ymm3, ymm1, ymm10)

	vextractf128(imm(0x1), ymm4, xmm12)
	vextractf128(imm(0x1), ymm6, xmm13)
	vextractf128(imm(0x1), ymm8, xmm14)
	vextractf128(imm(0x1), ymm10, xmm15)

	vmovupd(xmm4, mem(rcx))
	vmovupd(xmm6, mem(rcx, rsi, 1))
	vmovupd(xmm8, mem(rcx, rsi, 2))
	vmovupd(xmm10, mem(rcx, rax, 1))

	lea(mem(rcx, rsi, 4), rcx)

	vmovsd(xmm12, mem(rdx))
	vmovsd(xmm13, mem(rdx, rsi, 1))
	vmovsd(xmm14, mem(rdx, rsi, 2))
	vmovsd(xmm15, mem(rdx, rax, 1))

	lea(mem(rdx, rsi, 4), rdx)


	vunpcklpd(ymm7, ymm5, ymm0)
	vunpckhpd(ymm7, ymm5, ymm1)
	vunpcklpd(ymm11, ymm9, ymm2)
	vunpckhpd(ymm11, ymm9, ymm3)
	vinsertf128(imm(0x1), xmm2, ymm0, ymm5)
	vinsertf128(imm(0x1), xmm3, ymm1, ymm7)
	vperm2f128(imm(0x31), ymm2, ymm0, ymm9)
	vperm2f128(imm(0x31), ymm3, ymm1, ymm11)

	vextractf128(imm(0x1), ymm5, xmm12)
	vextractf128(imm(0x1), ymm7, xmm13)
	vextractf128(imm(0x1), ymm9, xmm14)
	vextractf128(imm(0x1), ymm11, xmm15)

	vmovupd(xmm5, mem(rcx))
	vmovupd(xmm7, mem(rcx, rsi, 1))
	vmovupd(xmm9, mem(rcx, rsi, 2))
	vmovupd(xmm11, mem(rcx, rax, 1))

	//lea(mem(rcx, rsi, 4), rcx)

	vmovsd(xmm12, mem(rdx))
	vmovsd(xmm13, mem(rdx, rsi, 1))
	vmovsd(xmm14, mem(rdx, rsi, 2))
	vmovsd(xmm15, mem(rdx, rax, 1))

	//lea(mem(rdx, rsi, 4), rdx)

	
	
	
	label(.DDONE)
	
	

    end_asm(
	: // output operands (none)
	: // input operands
      [k_iter] "m" (k_iter),
      [k_left] "m" (k_left),
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
	  "ymm8", "ymm9", "ymm10", "ymm11", "memory"
	)

    AOCL_DTL_TRACE_EXIT(AOCL_DTL_LEVEL_TRACE_7);
}

void bli_dgemmsup_rv_haswell_asm_2x8
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
	AOCL_DTL_TRACE_ENTRY(AOCL_DTL_LEVEL_TRACE_7);
	//void*    a_next = bli_auxinfo_next_a( data );
	//void*    b_next = bli_auxinfo_next_b( data );

	// Typecast local copies of integers in case dim_t and inc_t are a
	// different size than is expected by load instructions.
	uint64_t k_iter = k0 / 4;
	uint64_t k_left = k0 % 4;

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
	mov(var(cs_a), r9)                 // load cs_a
	lea(mem(, r8, 8), r8)              // rs_a *= sizeof(double)
	lea(mem(, r9, 8), r9)              // cs_a *= sizeof(double)

	//lea(mem(r8, r8, 2), r13)           // r13 = 3*rs_a
	//lea(mem(r8, r8, 4), r15)           // r15 = 5*rs_a

	mov(var(b), rbx)                   // load address of b.
	mov(var(rs_b), r10)                // load rs_b
	//mov(var(cs_b), r11)                // load cs_b
	lea(mem(, r10, 8), r10)            // rs_b *= sizeof(double)
	//lea(mem(, r11, 8), r11)            // cs_b *= sizeof(double)
	
	                                   // NOTE: We cannot pre-load elements of a or b
	                                   // because it could eventually, in the last
	                                   // unrolled iter or the cleanup loop, result
	                                   // in reading beyond the bounds allocated mem
	                                   // (the likely result: a segmentation fault).

	mov(var(c), rcx)                   // load address of c
	mov(var(rs_c), rdi)                // load rs_c
	lea(mem(, rdi, 8), rdi)            // rs_c *= sizeof(double)

#if 0
	//lea(mem(rcx, rdi, 2), rdx)         //
	//lea(mem(rdx, rdi, 1), rdx)         // rdx = c + 3*rs_c;
	prefetch(0, mem(rcx, 7*8))         // prefetch c + 0*rs_c
	prefetch(0, mem(rcx, rdi, 1, 7*8)) // prefetch c + 1*rs_c
#else
	cmp(imm(8), rdi)                   // set ZF if (8*rs_c) == 8.
	jz(.DCOLPFETCH)                    // jump to column storage case
	label(.DROWPFETCH)                 // row-stored prefetching on c

	//lea(mem(rcx, rdi, 2), rdx)         //
	//lea(mem(rdx, rdi, 1), rdx)         // rdx = c + 3*rs_c;
	prefetch(0, mem(rcx, 7*8))         // prefetch c + 0*rs_c
	prefetch(0, mem(rcx, rdi, 1, 7*8)) // prefetch c + 1*rs_c

	jmp(.DPOSTPFETCH)                  // jump to end of prefetching c
	label(.DCOLPFETCH)                 // column-stored prefetching c

	mov(var(cs_c), rsi)                // load cs_c to rsi (temporarily)
	lea(mem(, rsi, 8), rsi)            // cs_c *= sizeof(double)
	lea(mem(rcx, rsi, 2), rdx)         //
	lea(mem(rdx, rsi, 1), rdx)         // rdx = c + 3*cs_c;
	prefetch(0, mem(rcx, 1*8))         // prefetch c + 0*cs_c
	prefetch(0, mem(rcx, rsi, 1, 1*8)) // prefetch c + 1*cs_c
	prefetch(0, mem(rcx, rsi, 2, 1*8)) // prefetch c + 2*cs_c
	prefetch(0, mem(rdx, 1*8))         // prefetch c + 3*cs_c
	prefetch(0, mem(rdx, rsi, 1, 1*8)) // prefetch c + 4*cs_c
	prefetch(0, mem(rdx, rsi, 2, 1*8)) // prefetch c + 5*cs_c
	lea(mem(rdx, rsi, 2), rdx)         // rdx = c + 5*cs_c;
	prefetch(0, mem(rdx, rsi, 1, 1*8)) // prefetch c + 6*cs_c
	prefetch(0, mem(rdx, rsi, 2, 1*8)) // prefetch c + 7*cs_c

	label(.DPOSTPFETCH)                // done prefetching c
#endif
	
	
	
	
	mov(var(k_iter), rsi)              // i = k_iter;
	test(rsi, rsi)                     // check i via logical AND.
	je(.DCONSIDKLEFT)                  // if i == 0, jump to code that
	                                   // contains the k_left loop.
	
	
	label(.DLOOPKITER)                 // MAIN LOOP
	
	
	// ---------------------------------- iteration 0
	
	vmovupd(mem(rbx,  0*32), ymm0)
	vmovupd(mem(rbx,  1*32), ymm1)
	add(r10, rbx)                      // b += rs_b;

	vbroadcastsd(mem(rax        ), ymm2)
	vbroadcastsd(mem(rax, r8,  1), ymm3)
	add(r9, rax)                       // a += cs_a;
	vfmadd231pd(ymm0, ymm2, ymm4)
	vfmadd231pd(ymm1, ymm2, ymm5)
	vfmadd231pd(ymm0, ymm3, ymm6)
	vfmadd231pd(ymm1, ymm3, ymm7)
	
	
	// ---------------------------------- iteration 1

	vmovupd(mem(rbx,  0*32), ymm0)
	vmovupd(mem(rbx,  1*32), ymm1)
	add(r10, rbx)                      // b += rs_b;

	vbroadcastsd(mem(rax        ), ymm2)
	vbroadcastsd(mem(rax, r8,  1), ymm3)
	add(r9, rax)                       // a += cs_a;
	vfmadd231pd(ymm0, ymm2, ymm4)
	vfmadd231pd(ymm1, ymm2, ymm5)
	vfmadd231pd(ymm0, ymm3, ymm6)
	vfmadd231pd(ymm1, ymm3, ymm7)
	

	// ---------------------------------- iteration 2
	
	vmovupd(mem(rbx,  0*32), ymm0)
	vmovupd(mem(rbx,  1*32), ymm1)
	add(r10, rbx)                      // b += rs_b;

	vbroadcastsd(mem(rax        ), ymm2)
	vbroadcastsd(mem(rax, r8,  1), ymm3)
	add(r9, rax)                       // a += cs_a;
	vfmadd231pd(ymm0, ymm2, ymm4)
	vfmadd231pd(ymm1, ymm2, ymm5)
	vfmadd231pd(ymm0, ymm3, ymm6)
	vfmadd231pd(ymm1, ymm3, ymm7)
	

	// ---------------------------------- iteration 3

	vmovupd(mem(rbx,  0*32), ymm0)
	vmovupd(mem(rbx,  1*32), ymm1)
	add(r10, rbx)                      // b += rs_b;

	vbroadcastsd(mem(rax        ), ymm2)
	vbroadcastsd(mem(rax, r8,  1), ymm3)
	add(r9, rax)                       // a += cs_a;
	vfmadd231pd(ymm0, ymm2, ymm4)
	vfmadd231pd(ymm1, ymm2, ymm5)
	vfmadd231pd(ymm0, ymm3, ymm6)
	vfmadd231pd(ymm1, ymm3, ymm7)
	
	
	
	dec(rsi)                           // i -= 1;
	jne(.DLOOPKITER)                   // iterate again if i != 0.
	
	
	
	
	
	
	label(.DCONSIDKLEFT)
	
	mov(var(k_left), rsi)              // i = k_left;
	test(rsi, rsi)                     // check i via logical AND.
	je(.DPOSTACCUM)                    // if i == 0, we're done; jump to end.
	                                   // else, we prepare to enter k_left loop.
	
	
	label(.DLOOPKLEFT)                 // EDGE LOOP
	
	vmovupd(mem(rbx,  0*32), ymm0)
	vmovupd(mem(rbx,  1*32), ymm1)
	add(r10, rbx)                      // b += rs_b;

	vbroadcastsd(mem(rax        ), ymm2)
	vbroadcastsd(mem(rax, r8,  1), ymm3)
	add(r9, rax)                       // a += cs_a;
	vfmadd231pd(ymm0, ymm2, ymm4)
	vfmadd231pd(ymm1, ymm2, ymm5)
	vfmadd231pd(ymm0, ymm3, ymm6)
	vfmadd231pd(ymm1, ymm3, ymm7)
	
	
	dec(rsi)                           // i -= 1;
	jne(.DLOOPKLEFT)                   // iterate again if i != 0.
	
	
	
	label(.DPOSTACCUM)


	
	mov(var(alpha), rax)               // load address of alpha
	mov(var(beta), rbx)                // load address of beta
	vbroadcastsd(mem(rax), ymm0)       // load alpha and duplicate
	vbroadcastsd(mem(rbx), ymm3)       // load beta and duplicate
	
	vmulpd(ymm0, ymm4, ymm4)           // scale by alpha
	vmulpd(ymm0, ymm5, ymm5)
	vmulpd(ymm0, ymm6, ymm6)
	vmulpd(ymm0, ymm7, ymm7)
	
	
	
	
	
	
	mov(var(cs_c), rsi)                // load cs_c
	lea(mem(, rsi, 8), rsi)            // rsi = cs_c * sizeof(double)
	
	//lea(mem(rcx, rsi, 4), rdx)         // load address of c +  4*cs_c;
	//lea(mem(rcx, rdi, 4), rdx)         // load address of c +  4*rs_c;

	lea(mem(rsi, rsi, 2), rax)         // rax = 3*cs_c;
	
	
	
	                                   // now avoid loading C if beta == 0
	
	vxorpd(ymm0, ymm0, ymm0)           // set ymm0 to zero.
	vucomisd(xmm0, xmm3)               // set ZF if beta == 0.
	je(.DBETAZERO)                     // if ZF = 1, jump to beta == 0 case



	cmp(imm(8), rdi)                   // set ZF if (8*rs_c) == 8.
	jz(.DCOLSTORED)                    // jump to column storage case
	

	
	label(.DROWSTORED)
	
	
	vfmadd231pd(mem(rcx), ymm3, ymm4)
	vmovupd(ymm4, mem(rcx))

	vfmadd231pd(mem(rcx, rsi, 4), ymm3, ymm5)
	vmovupd(ymm5, mem(rcx, rsi, 4))
	add(rdi, rcx)


	vfmadd231pd(mem(rcx), ymm3, ymm6)
	vmovupd(ymm6, mem(rcx))

	vfmadd231pd(mem(rcx, rsi, 4), ymm3, ymm7)
	vmovupd(ymm7, mem(rcx, rsi, 4))
	//add(rdi, rcx)
	
	
	jmp(.DDONE)                        // jump to end.



	label(.DCOLSTORED)


	vunpcklpd(ymm6, ymm4, ymm0)
	vunpckhpd(ymm6, ymm4, ymm1)
	vextractf128(imm(0x1), ymm0, xmm2)
	vextractf128(imm(0x1), ymm1, xmm4)

	vfmadd231pd(mem(rcx), xmm3, xmm0)
	vfmadd231pd(mem(rcx, rsi, 1), xmm3, xmm1)
	vfmadd231pd(mem(rcx, rsi, 2), xmm3, xmm2)
	vfmadd231pd(mem(rcx, rax, 1), xmm3, xmm4)
	vmovupd(xmm0, mem(rcx))
	vmovupd(xmm1, mem(rcx, rsi, 1))
	vmovupd(xmm2, mem(rcx, rsi, 2))
	vmovupd(xmm4, mem(rcx, rax, 1))

	lea(mem(rcx, rsi, 4), rcx)


	vunpcklpd(ymm7, ymm5, ymm0)
	vunpckhpd(ymm7, ymm5, ymm1)
	vextractf128(imm(0x1), ymm0, xmm2)
	vextractf128(imm(0x1), ymm1, xmm4)

	vfmadd231pd(mem(rcx), xmm3, xmm0)
	vfmadd231pd(mem(rcx, rsi, 1), xmm3, xmm1)
	vfmadd231pd(mem(rcx, rsi, 2), xmm3, xmm2)
	vfmadd231pd(mem(rcx, rax, 1), xmm3, xmm4)
	vmovupd(xmm0, mem(rcx))
	vmovupd(xmm1, mem(rcx, rsi, 1))
	vmovupd(xmm2, mem(rcx, rsi, 2))
	vmovupd(xmm4, mem(rcx, rax, 1))

	//lea(mem(rcx, rsi, 4), rcx)


	jmp(.DDONE)                        // jump to end.


	
	
	label(.DBETAZERO)


	cmp(imm(8), rdi)                   // set ZF if (8*rs_c) == 8.
	jz(.DCOLSTORBZ)                    // jump to column storage case
	

	
	label(.DROWSTORBZ)
	
	
	vmovupd(ymm4, mem(rcx))
	vmovupd(ymm5, mem(rcx, rsi, 4))
	add(rdi, rcx)

	vmovupd(ymm6, mem(rcx))
	vmovupd(ymm7, mem(rcx, rsi, 4))
	//add(rdi, rcx)
	
	
	jmp(.DDONE)                        // jump to end.



	label(.DCOLSTORBZ)


	vunpcklpd(ymm6, ymm4, ymm0)
	vunpckhpd(ymm6, ymm4, ymm1)
	vextractf128(imm(0x1), ymm0, xmm2)
	vextractf128(imm(0x1), ymm1, xmm4)

	vmovupd(xmm0, mem(rcx))
	vmovupd(xmm1, mem(rcx, rsi, 1))
	vmovupd(xmm2, mem(rcx, rsi, 2))
	vmovupd(xmm4, mem(rcx, rax, 1))

	lea(mem(rcx, rsi, 4), rcx)


	vunpcklpd(ymm7, ymm5, ymm0)
	vunpckhpd(ymm7, ymm5, ymm1)
	vextractf128(imm(0x1), ymm0, xmm2)
	vextractf128(imm(0x1), ymm1, xmm4)

	vmovupd(xmm0, mem(rcx))
	vmovupd(xmm1, mem(rcx, rsi, 1))
	vmovupd(xmm2, mem(rcx, rsi, 2))
	vmovupd(xmm4, mem(rcx, rax, 1))

	//lea(mem(rcx, rsi, 4), rcx)

	
	
	
	label(.DDONE)
	
	

    end_asm(
	: // output operands (none)
	: // input operands
      [k_iter] "m" (k_iter),
      [k_left] "m" (k_left),
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
	  "ymm0", "ymm1", "ymm2", "ymm3", "ymm4", "ymm5", "ymm6", "ymm7", "memory"
	)
	AOCL_DTL_TRACE_EXIT(AOCL_DTL_LEVEL_TRACE_7);
}

void bli_dgemmsup_rv_haswell_asm_1x8
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
	AOCL_DTL_TRACE_ENTRY(AOCL_DTL_LEVEL_TRACE_7);
	//void*    a_next = bli_auxinfo_next_a( data );
	//void*    b_next = bli_auxinfo_next_b( data );

	// Typecast local copies of integers in case dim_t and inc_t are a
	// different size than is expected by load instructions.
	uint64_t k_iter = k0 / 4;
	uint64_t k_left = k0 % 4;

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
	mov(var(cs_a), r9)                 // load cs_a
	lea(mem(, r8, 8), r8)              // rs_a *= sizeof(double)
	lea(mem(, r9, 8), r9)              // cs_a *= sizeof(double)

	//lea(mem(r8, r8, 2), r13)           // r13 = 3*rs_a
	//lea(mem(r8, r8, 4), r15)           // r15 = 5*rs_a

	mov(var(b), rbx)                   // load address of b.
	mov(var(rs_b), r10)                // load rs_b
	//mov(var(cs_b), r11)                // load cs_b
	lea(mem(, r10, 8), r10)            // rs_b *= sizeof(double)
	//lea(mem(, r11, 8), r11)            // cs_b *= sizeof(double)
	
	                                   // NOTE: We cannot pre-load elements of a or b
	                                   // because it could eventually, in the last
	                                   // unrolled iter or the cleanup loop, result
	                                   // in reading beyond the bounds allocated mem
	                                   // (the likely result: a segmentation fault).

	mov(var(c), rcx)                   // load address of c
	mov(var(rs_c), rdi)                // load rs_c
	lea(mem(, rdi, 8), rdi)            // rs_c *= sizeof(double)

#if 0
	//lea(mem(rcx, rdi, 2), rdx)         //
	//lea(mem(rdx, rdi, 1), rdx)         // rdx = c + 3*rs_c;
	prefetch(0, mem(rcx, 7*8))         // prefetch c + 0*rs_c
#else
	cmp(imm(8), rdi)                   // set ZF if (8*rs_c) == 8.
	jz(.DCOLPFETCH)                    // jump to column storage case
	label(.DROWPFETCH)                 // row-stored prefetching on c

	//lea(mem(rcx, rdi, 2), rdx)         //
	//lea(mem(rdx, rdi, 1), rdx)         // rdx = c + 3*rs_c;
	prefetch(0, mem(rcx, 7*8))         // prefetch c + 0*rs_c

	jmp(.DPOSTPFETCH)                  // jump to end of prefetching c
	label(.DCOLPFETCH)                 // column-stored prefetching c

	mov(var(cs_c), rsi)                // load cs_c to rsi (temporarily)
	lea(mem(, rsi, 8), rsi)            // cs_c *= sizeof(double)
	lea(mem(rcx, rsi, 2), rdx)         //
	lea(mem(rdx, rsi, 1), rdx)         // rdx = c + 3*cs_c;
	prefetch(0, mem(rcx, 0*8))         // prefetch c + 0*cs_c
	prefetch(0, mem(rcx, rsi, 1, 0*8)) // prefetch c + 1*cs_c
	prefetch(0, mem(rcx, rsi, 2, 0*8)) // prefetch c + 2*cs_c
	prefetch(0, mem(rdx, 0*8))         // prefetch c + 3*cs_c
	prefetch(0, mem(rdx, rsi, 1, 0*8)) // prefetch c + 4*cs_c
	prefetch(0, mem(rdx, rsi, 2, 0*8)) // prefetch c + 5*cs_c
	lea(mem(rdx, rsi, 2), rdx)         // rdx = c + 5*cs_c;
	prefetch(0, mem(rdx, rsi, 1, 0*8)) // prefetch c + 6*cs_c
	prefetch(0, mem(rdx, rsi, 2, 0*8)) // prefetch c + 7*cs_c

	label(.DPOSTPFETCH)                // done prefetching c
#endif
	
	
	
	
	mov(var(k_iter), rsi)              // i = k_iter;
	test(rsi, rsi)                     // check i via logical AND.
	je(.DCONSIDKLEFT)                  // if i == 0, jump to code that
	                                   // contains the k_left loop.
	
	
	label(.DLOOPKITER)                 // MAIN LOOP
	
	
	// ---------------------------------- iteration 0
	
	vmovupd(mem(rbx,  0*32), ymm0)
	vmovupd(mem(rbx,  1*32), ymm1)
	add(r10, rbx)                      // b += rs_b;

	vbroadcastsd(mem(rax        ), ymm2)
	add(r9, rax)                       // a += cs_a;
	vfmadd231pd(ymm0, ymm2, ymm4)
	vfmadd231pd(ymm1, ymm2, ymm5)
	
	
	// ---------------------------------- iteration 1

	vmovupd(mem(rbx, 0*32), ymm0)
	vmovupd(mem(rbx, 1*32), ymm1)
	add(r10, rbx)                      // b += rs_b;
	
	vbroadcastsd(mem(rax        ), ymm2)
	add(r9, rax)                       // a += cs_a;
	vfmadd231pd(ymm0, ymm2, ymm4)
	vfmadd231pd(ymm1, ymm2, ymm5)
	

	// ---------------------------------- iteration 2
	
	vmovupd(mem(rbx, 0*32), ymm0)
	vmovupd(mem(rbx, 1*32), ymm1)
	add(r10, rbx)                      // b += rs_b;

	vbroadcastsd(mem(rax        ), ymm2)
	add(r9, rax)                       // a += cs_a;
	vfmadd231pd(ymm0, ymm2, ymm4)
	vfmadd231pd(ymm1, ymm2, ymm5)
	

	// ---------------------------------- iteration 3

	vmovupd(mem(rbx, 0*32), ymm0)
	vmovupd(mem(rbx, 1*32), ymm1)
	add(r10, rbx)                      // b += rs_b;

	vbroadcastsd(mem(rax        ), ymm2)
	add(r9, rax)                       // a += cs_a;
	vfmadd231pd(ymm0, ymm2, ymm4)
	vfmadd231pd(ymm1, ymm2, ymm5)
	
	
	
	dec(rsi)                           // i -= 1;
	jne(.DLOOPKITER)                   // iterate again if i != 0.
	
	
	
	
	
	
	label(.DCONSIDKLEFT)
	
	mov(var(k_left), rsi)              // i = k_left;
	test(rsi, rsi)                     // check i via logical AND.
	je(.DPOSTACCUM)                    // if i == 0, we're done; jump to end.
	                                   // else, we prepare to enter k_left loop.
	
	
	label(.DLOOPKLEFT)                 // EDGE LOOP
	
	vmovupd(mem(rbx,  0*32), ymm0)
	vmovupd(mem(rbx,  1*32), ymm1)
	add(r10, rbx)                      // b += rs_b;
	
	vbroadcastsd(mem(rax        ), ymm2)
	add(r9, rax)                       // a += cs_a;
	vfmadd231pd(ymm0, ymm2, ymm4)
	vfmadd231pd(ymm1, ymm2, ymm5)
	
	
	dec(rsi)                           // i -= 1;
	jne(.DLOOPKLEFT)                   // iterate again if i != 0.
	
	
	
	label(.DPOSTACCUM)



	mov(var(alpha), rax)               // load address of alpha
	mov(var(beta), rbx)                // load address of beta
	vbroadcastsd(mem(rax), ymm0)       // load alpha and duplicate
	vbroadcastsd(mem(rbx), ymm3)       // load beta and duplicate
	
	vmulpd(ymm0, ymm4, ymm4)           // scale by alpha
	vmulpd(ymm0, ymm5, ymm5)
	
	
	
	
	
	
	mov(var(cs_c), rsi)                // load cs_c
	lea(mem(, rsi, 8), rsi)            // rsi = cs_c * sizeof(double)
	
	//lea(mem(rcx, rsi, 4), rdx)         // load address of c +  4*cs_c;
	//lea(mem(rcx, rdi, 4), rdx)         // load address of c +  4*rs_c;

	lea(mem(rsi, rsi, 2), rax)         // rax = 3*cs_c;
	
	
	
	                                   // now avoid loading C if beta == 0
	
	vxorpd(ymm0, ymm0, ymm0)           // set ymm0 to zero.
	vucomisd(xmm0, xmm3)               // set ZF if beta == 0.
	je(.DBETAZERO)                     // if ZF = 1, jump to beta == 0 case


	cmp(imm(8), rdi)                   // set ZF if (8*rs_c) == 8.
	jz(.DCOLSTORED)                    // jump to column storage case
	

	
	label(.DROWSTORED)
	
	
	vfmadd231pd(mem(rcx), ymm3, ymm4)
	vmovupd(ymm4, mem(rcx))

	vfmadd231pd(mem(rcx, rsi, 4), ymm3, ymm5)
	vmovupd(ymm5, mem(rcx, rsi, 4))
	//add(rdi, rcx)
	
	
	jmp(.DDONE)                        // jump to end.



	label(.DCOLSTORED)


	vmovlpd(mem(rcx), xmm0, xmm0)
	vmovhpd(mem(rcx, rsi, 1), xmm0, xmm0)
	vmovlpd(mem(rcx, rsi, 2), xmm1, xmm1)
	vmovhpd(mem(rcx, rax, 1), xmm1, xmm1)
	vperm2f128(imm(0x20), ymm1, ymm0, ymm0)

	vfmadd213pd(ymm4, ymm3, ymm0)

	vextractf128(imm(1), ymm0, xmm1)
	vmovlpd(xmm0, mem(rcx))
	vmovhpd(xmm0, mem(rcx, rsi, 1))
	vmovlpd(xmm1, mem(rcx, rsi, 2))
	vmovhpd(xmm1, mem(rcx, rax, 1))
	
	lea(mem(rcx, rsi, 4), rcx)


	vmovlpd(mem(rcx), xmm0, xmm0)
	vmovhpd(mem(rcx, rsi, 1), xmm0, xmm0)
	vmovlpd(mem(rcx, rsi, 2), xmm1, xmm1)
	vmovhpd(mem(rcx, rax, 1), xmm1, xmm1)
	vperm2f128(imm(0x20), ymm1, ymm0, ymm0)

	vfmadd213pd(ymm5, ymm3, ymm0)

	vextractf128(imm(1), ymm0, xmm1)
	vmovlpd(xmm0, mem(rcx))
	vmovhpd(xmm0, mem(rcx, rsi, 1))
	vmovlpd(xmm1, mem(rcx, rsi, 2))
	vmovhpd(xmm1, mem(rcx, rax, 1))

	//lea(mem(rcx, rsi, 4), rcx)


	jmp(.DDONE)                        // jump to end.
	
	
	
	
	label(.DBETAZERO)


	cmp(imm(8), rdi)                   // set ZF if (8*rs_c) == 8.
	jz(.DCOLSTORBZ)                    // jump to column storage case
	

	
	label(.DROWSTORBZ)
	
	
	vmovupd(ymm4, mem(rcx))
	vmovupd(ymm5, mem(rcx, rsi, 4))
	//add(rdi, rcx)
	
	
	jmp(.DDONE)                        // jump to end.



	label(.DCOLSTORBZ)


	vmovupd(ymm4, ymm0)

	vextractf128(imm(1), ymm0, xmm1)
	vmovlpd(xmm0, mem(rcx))
	vmovhpd(xmm0, mem(rcx, rsi, 1))
	vmovlpd(xmm1, mem(rcx, rsi, 2))
	vmovhpd(xmm1, mem(rcx, rax, 1))

	lea(mem(rcx, rsi, 4), rcx)


	vmovupd(ymm5, ymm0)

	vextractf128(imm(1), ymm0, xmm1)
	vmovlpd(xmm0, mem(rcx))
	vmovhpd(xmm0, mem(rcx, rsi, 1))
	vmovlpd(xmm1, mem(rcx, rsi, 2))
	vmovhpd(xmm1, mem(rcx, rax, 1))

	//lea(mem(rcx, rsi, 4), rcx)
	

	
	
	
	label(.DDONE)
	
	

    end_asm(
	: // output operands (none)
	: // input operands
      [k_iter] "m" (k_iter),
      [k_left] "m" (k_left),
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
	  "ymm0", "ymm1", "ymm2", "ymm3", "ymm4", "ymm5", "memory"
	)
	AOCL_DTL_TRACE_EXIT(AOCL_DTL_LEVEL_TRACE_7);
}

void bli_dgemmsup_rv_haswell_asm_6x6
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
	AOCL_DTL_TRACE_ENTRY(AOCL_DTL_LEVEL_TRACE_7);
	//void*    a_next = bli_auxinfo_next_a( data );
	//void*    b_next = bli_auxinfo_next_b( data );

	// Typecast local copies of integers in case dim_t and inc_t are a
	// different size than is expected by load instructions.
	uint64_t k_iter = k0 / 4;
	uint64_t k_left = k0 % 4;

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
	mov(var(cs_a), r9)                 // load cs_a
	lea(mem(, r8, 8), r8)              // rs_a *= sizeof(double)
	lea(mem(, r9, 8), r9)              // cs_a *= sizeof(double)
	
	lea(mem(r8, r8, 2), r13)           // r13 = 3*rs_a
	lea(mem(r8, r8, 4), r15)           // r15 = 5*rs_a

	mov(var(b), rbx)                   // load address of b.
	mov(var(rs_b), r10)                // load rs_b
	//mov(var(cs_b), r11)                // load cs_b
	lea(mem(, r10, 8), r10)            // rs_b *= sizeof(double)
	//lea(mem(, r11, 8), r11)            // cs_b *= sizeof(double)

	                                   // NOTE: We cannot pre-load elements of a or b
	                                   // because it could eventually, in the last
	                                   // unrolled iter or the cleanup loop, result
	                                   // in reading beyond the bounds allocated mem
	                                   // (the likely result: a segmentation fault).

	mov(var(c), rcx)                   // load address of c
	mov(var(rs_c), rdi)                // load rs_c
	lea(mem(, rdi, 8), rdi)            // rs_c *= sizeof(double)



	cmp(imm(8), rdi)                   // set ZF if (8*rs_c) == 8.
	jz(.DCOLPFETCH)                    // jump to column storage case
	label(.DROWPFETCH)                 // row-stored prefetching on c

	lea(mem(rcx, rdi, 2), rdx)         //
	lea(mem(rdx, rdi, 1), rdx)         // rdx = c + 3*rs_c;
	prefetch(0, mem(rcx, 5*8))         // prefetch c + 0*rs_c
	prefetch(0, mem(rcx, rdi, 1, 5*8)) // prefetch c + 1*rs_c
	prefetch(0, mem(rcx, rdi, 2, 5*8)) // prefetch c + 2*rs_c
	prefetch(0, mem(rdx, 5*8))         // prefetch c + 3*rs_c
	prefetch(0, mem(rdx, rdi, 1, 5*8)) // prefetch c + 4*rs_c
	prefetch(0, mem(rdx, rdi, 2, 5*8)) // prefetch c + 5*rs_c

	jmp(.DPOSTPFETCH)                  // jump to end of prefetching c
	label(.DCOLPFETCH)                 // column-stored prefetching c

	mov(var(cs_c), rsi)                // load cs_c to rsi (temporarily)
	lea(mem(, rsi, 8), rsi)            // cs_c *= sizeof(double)
	lea(mem(rcx, rsi, 2), rdx)         //
	lea(mem(rdx, rsi, 1), rdx)         // rdx = c + 3*cs_c;
	prefetch(0, mem(rcx, 5*8))         // prefetch c + 0*cs_c
	prefetch(0, mem(rcx, rsi, 1, 5*8)) // prefetch c + 1*cs_c
	prefetch(0, mem(rcx, rsi, 2, 5*8)) // prefetch c + 2*cs_c
	prefetch(0, mem(rdx, 5*8))         // prefetch c + 3*cs_c
	prefetch(0, mem(rdx, rsi, 1, 5*8)) // prefetch c + 4*cs_c
	prefetch(0, mem(rdx, rsi, 2, 5*8)) // prefetch c + 5*cs_c

	label(.DPOSTPFETCH)                // done prefetching c


	
	
	
	mov(var(k_iter), rsi)              // i = k_iter;
	test(rsi, rsi)                     // check i via logical AND.
	je(.DCONSIDKLEFT)                  // if i == 0, jump to code that
	                                   // contains the k_left loop.
	
	
	label(.DLOOPKITER)                 // MAIN LOOP
	
	
	// ---------------------------------- iteration 0

	vmovupd(mem(rbx,  0*32), ymm0)
	vmovupd(mem(rbx,  1*32), xmm1)
	add(r10, rbx)                      // b += rs_b;

	vbroadcastsd(mem(rax        ), ymm2)
	vbroadcastsd(mem(rax, r8,  1), ymm3)
	vfmadd231pd(ymm0, ymm2, ymm4)
	vfmadd231pd(ymm1, ymm2, ymm5)
	vfmadd231pd(ymm0, ymm3, ymm6)
	vfmadd231pd(ymm1, ymm3, ymm7)
	
	vbroadcastsd(mem(rax, r8,  2), ymm2)
	vbroadcastsd(mem(rax, r13, 1), ymm3)
	vfmadd231pd(ymm0, ymm2, ymm8)
	vfmadd231pd(ymm1, ymm2, ymm9)
	vfmadd231pd(ymm0, ymm3, ymm10)
	vfmadd231pd(ymm1, ymm3, ymm11)
	
	vbroadcastsd(mem(rax, r8,  4), ymm2)
	vbroadcastsd(mem(rax, r15, 1), ymm3)
	add(r9, rax)                       // a += cs_a;
	vfmadd231pd(ymm0, ymm2, ymm12)
	vfmadd231pd(ymm1, ymm2, ymm13)
	vfmadd231pd(ymm0, ymm3, ymm14)
	vfmadd231pd(ymm1, ymm3, ymm15)

	
	// ---------------------------------- iteration 1

	vmovupd(mem(rbx,  0*32), ymm0)
	vmovupd(mem(rbx,  1*32), xmm1)
	add(r10, rbx)                      // b += rs_b;

	vbroadcastsd(mem(rax        ), ymm2)
	vbroadcastsd(mem(rax, r8,  1), ymm3)
	vfmadd231pd(ymm0, ymm2, ymm4)
	vfmadd231pd(ymm1, ymm2, ymm5)
	vfmadd231pd(ymm0, ymm3, ymm6)
	vfmadd231pd(ymm1, ymm3, ymm7)
	
	vbroadcastsd(mem(rax, r8,  2), ymm2)
	vbroadcastsd(mem(rax, r13, 1), ymm3)
	vfmadd231pd(ymm0, ymm2, ymm8)
	vfmadd231pd(ymm1, ymm2, ymm9)
	vfmadd231pd(ymm0, ymm3, ymm10)
	vfmadd231pd(ymm1, ymm3, ymm11)
	
	vbroadcastsd(mem(rax, r8,  4), ymm2)
	vbroadcastsd(mem(rax, r15, 1), ymm3)
	add(r9, rax)                       // a += cs_a;
	vfmadd231pd(ymm0, ymm2, ymm12)
	vfmadd231pd(ymm1, ymm2, ymm13)
	vfmadd231pd(ymm0, ymm3, ymm14)
	vfmadd231pd(ymm1, ymm3, ymm15)
	
	
	// ---------------------------------- iteration 2

	vmovupd(mem(rbx,  0*32), ymm0)
	vmovupd(mem(rbx,  1*32), xmm1)
	add(r10, rbx)                      // b += rs_b;

	vbroadcastsd(mem(rax        ), ymm2)
	vbroadcastsd(mem(rax, r8,  1), ymm3)
	vfmadd231pd(ymm0, ymm2, ymm4)
	vfmadd231pd(ymm1, ymm2, ymm5)
	vfmadd231pd(ymm0, ymm3, ymm6)
	vfmadd231pd(ymm1, ymm3, ymm7)
	
	vbroadcastsd(mem(rax, r8,  2), ymm2)
	vbroadcastsd(mem(rax, r13, 1), ymm3)
	vfmadd231pd(ymm0, ymm2, ymm8)
	vfmadd231pd(ymm1, ymm2, ymm9)
	vfmadd231pd(ymm0, ymm3, ymm10)
	vfmadd231pd(ymm1, ymm3, ymm11)
	
	vbroadcastsd(mem(rax, r8,  4), ymm2)
	vbroadcastsd(mem(rax, r15, 1), ymm3)
	add(r9, rax)                       // a += cs_a;
	vfmadd231pd(ymm0, ymm2, ymm12)
	vfmadd231pd(ymm1, ymm2, ymm13)
	vfmadd231pd(ymm0, ymm3, ymm14)
	vfmadd231pd(ymm1, ymm3, ymm15)
	

	// ---------------------------------- iteration 3

	vmovupd(mem(rbx, 0*32), ymm0)
	vmovupd(mem(rbx, 1*32), xmm1)
	add(r10, rbx)                      // b += rs_b;

	vbroadcastsd(mem(rax        ), ymm2)
	vbroadcastsd(mem(rax, r8,  1), ymm3)
	vfmadd231pd(ymm0, ymm2, ymm4)
	vfmadd231pd(ymm1, ymm2, ymm5)
	vfmadd231pd(ymm0, ymm3, ymm6)
	vfmadd231pd(ymm1, ymm3, ymm7)
	
	vbroadcastsd(mem(rax, r8,  2), ymm2)
	vbroadcastsd(mem(rax, r13, 1), ymm3)
	vfmadd231pd(ymm0, ymm2, ymm8)
	vfmadd231pd(ymm1, ymm2, ymm9)
	vfmadd231pd(ymm0, ymm3, ymm10)
	vfmadd231pd(ymm1, ymm3, ymm11)
	
	vbroadcastsd(mem(rax, r8,  4), ymm2)
	vbroadcastsd(mem(rax, r15, 1), ymm3)
	add(r9, rax)                       // a += cs_a;
	vfmadd231pd(ymm0, ymm2, ymm12)
	vfmadd231pd(ymm1, ymm2, ymm13)
	vfmadd231pd(ymm0, ymm3, ymm14)
	vfmadd231pd(ymm1, ymm3, ymm15)
	
	
	
	dec(rsi)                           // i -= 1;
	jne(.DLOOPKITER)                   // iterate again if i != 0.
	
	
	
	
	
	
	label(.DCONSIDKLEFT)
	
	mov(var(k_left), rsi)              // i = k_left;
	test(rsi, rsi)                     // check i via logical AND.
	je(.DPOSTACCUM)                    // if i == 0, we're done; jump to end.
	                                   // else, we prepare to enter k_left loop.
	
	
	label(.DLOOPKLEFT)                 // EDGE LOOP
	
	vmovupd(mem(rbx,  0*32), ymm0)
	vmovupd(mem(rbx,  1*32), xmm1)
	add(r10, rbx)                      // b += rs_b;
	
	vbroadcastsd(mem(rax        ), ymm2)
	vbroadcastsd(mem(rax, r8,  1), ymm3)
	vfmadd231pd(ymm0, ymm2, ymm4)
	vfmadd231pd(ymm1, ymm2, ymm5)
	vfmadd231pd(ymm0, ymm3, ymm6)
	vfmadd231pd(ymm1, ymm3, ymm7)
	
	vbroadcastsd(mem(rax, r8,  2), ymm2)
	vbroadcastsd(mem(rax, r13, 1), ymm3)
	vfmadd231pd(ymm0, ymm2, ymm8)
	vfmadd231pd(ymm1, ymm2, ymm9)
	vfmadd231pd(ymm0, ymm3, ymm10)
	vfmadd231pd(ymm1, ymm3, ymm11)
	
	vbroadcastsd(mem(rax, r8,  4), ymm2)
	vbroadcastsd(mem(rax, r15, 1), ymm3)
	add(r9, rax)                       // a += cs_a;
	vfmadd231pd(ymm0, ymm2, ymm12)
	vfmadd231pd(ymm1, ymm2, ymm13)
	vfmadd231pd(ymm0, ymm3, ymm14)
	vfmadd231pd(ymm1, ymm3, ymm15)
	
	
	dec(rsi)                           // i -= 1;
	jne(.DLOOPKLEFT)                   // iterate again if i != 0.
	
	
	
	label(.DPOSTACCUM)

	
	
	mov(var(alpha), rax)               // load address of alpha
	mov(var(beta), rbx)                // load address of beta
	vbroadcastsd(mem(rax), ymm0)       // load alpha and duplicate
	vbroadcastsd(mem(rbx), ymm3)       // load beta and duplicate
	
	vmulpd(ymm0, ymm4, ymm4)           // scale by alpha
	vmulpd(xmm0, xmm5, xmm5)
	vmulpd(ymm0, ymm6, ymm6)
	vmulpd(xmm0, xmm7, xmm7)
	vmulpd(ymm0, ymm8, ymm8)
	vmulpd(xmm0, xmm9, xmm9)
	vmulpd(ymm0, ymm10, ymm10)
	vmulpd(xmm0, xmm11, xmm11)
	vmulpd(ymm0, ymm12, ymm12)
	vmulpd(xmm0, xmm13, xmm13)
	vmulpd(ymm0, ymm14, ymm14)
	vmulpd(xmm0, xmm15, xmm15)
	
	
	
	
	
	
	mov(var(cs_c), rsi)                // load cs_c
	lea(mem(, rsi, 8), rsi)            // rsi = cs_c * sizeof(double)
	
	//lea(mem(rcx, rsi, 4), rdx)         // load address of c +  4*cs_c;
	lea(mem(rcx, rdi, 4), rdx)         // load address of c +  4*rs_c;

	lea(mem(rsi, rsi, 2), rax)         // rax = 3*cs_c;
	
	
	
	                                   // now avoid loading C if beta == 0
	
	vxorpd(ymm0, ymm0, ymm0)           // set ymm0 to zero.
	vucomisd(xmm0, xmm3)               // set ZF if beta == 0.
	je(.DBETAZERO)                     // if ZF = 1, jump to beta == 0 case



	cmp(imm(8), rdi)                   // set ZF if (8*rs_c) == 8.
	jz(.DCOLSTORED)                    // jump to column storage case
	

	
	label(.DROWSTORED)
	
	
	vfmadd231pd(mem(rcx), ymm3, ymm4)
	vmovupd(ymm4, mem(rcx))

	vfmadd231pd(mem(rcx, rsi, 4), xmm3, xmm5)
	vmovupd(xmm5, mem(rcx, rsi, 4))
	add(rdi, rcx)
	
	
	vfmadd231pd(mem(rcx), ymm3, ymm6)
	vmovupd(ymm6, mem(rcx))

	vfmadd231pd(mem(rcx, rsi, 4), xmm3, xmm7)
	vmovupd(xmm7, mem(rcx, rsi, 4))
	add(rdi, rcx)
	
	
	vfmadd231pd(mem(rcx), ymm3, ymm8)
	vmovupd(ymm8, mem(rcx))

	vfmadd231pd(mem(rcx, rsi, 4), xmm3, xmm9)
	vmovupd(xmm9, mem(rcx, rsi, 4))
	add(rdi, rcx)
	
	
	vfmadd231pd(mem(rcx), ymm3, ymm10)
	vmovupd(ymm10, mem(rcx))

	vfmadd231pd(mem(rcx, rsi, 4), xmm3, xmm11)
	vmovupd(xmm11, mem(rcx, rsi, 4))
	add(rdi, rcx)
	
	
	vfmadd231pd(mem(rcx), ymm3, ymm12)
	vmovupd(ymm12, mem(rcx))

	vfmadd231pd(mem(rcx, rsi, 4), xmm3, xmm13)
	vmovupd(xmm13, mem(rcx, rsi, 4))
	add(rdi, rcx)
	
	
	vfmadd231pd(mem(rcx), ymm3, ymm14)
	vmovupd(ymm14, mem(rcx))

	vfmadd231pd(mem(rcx, rsi, 4), xmm3, xmm15)
	vmovupd(xmm15, mem(rcx, rsi, 4))
	//add(rdi, rcx)
	
	
	jmp(.DDONE)                        // jump to end.



	label(.DCOLSTORED)


	vunpcklpd(ymm6, ymm4, ymm0)
	vunpckhpd(ymm6, ymm4, ymm1)
	vunpcklpd(ymm10, ymm8, ymm2)
	vunpckhpd(ymm10, ymm8, ymm3)
	vinsertf128(imm(0x1), xmm2, ymm0, ymm4)
	vinsertf128(imm(0x1), xmm3, ymm1, ymm6)
	vperm2f128(imm(0x31), ymm2, ymm0, ymm8)
	vperm2f128(imm(0x31), ymm3, ymm1, ymm10)

	vbroadcastsd(mem(rbx), ymm3)

	vfmadd231pd(mem(rcx), ymm3, ymm4)
	vfmadd231pd(mem(rcx, rsi, 1), ymm3, ymm6)
	vfmadd231pd(mem(rcx, rsi, 2), ymm3, ymm8)
	vfmadd231pd(mem(rcx, rax, 1), ymm3, ymm10)
	vmovupd(ymm4, mem(rcx))
	vmovupd(ymm6, mem(rcx, rsi, 1))
	vmovupd(ymm8, mem(rcx, rsi, 2))
	vmovupd(ymm10, mem(rcx, rax, 1))

	lea(mem(rcx, rsi, 4), rcx)

	vunpcklpd(ymm14, ymm12, ymm0)
	vunpckhpd(ymm14, ymm12, ymm1)
	vextractf128(imm(0x1), ymm0, xmm2)
	vextractf128(imm(0x1), ymm1, xmm4)

	vfmadd231pd(mem(rdx), xmm3, xmm0)
	vfmadd231pd(mem(rdx, rsi, 1), xmm3, xmm1)
	vfmadd231pd(mem(rdx, rsi, 2), xmm3, xmm2)
	vfmadd231pd(mem(rdx, rax, 1), xmm3, xmm4)
	vmovupd(xmm0, mem(rdx))
	vmovupd(xmm1, mem(rdx, rsi, 1))
	vmovupd(xmm2, mem(rdx, rsi, 2))
	vmovupd(xmm4, mem(rdx, rax, 1))

	lea(mem(rdx, rsi, 4), rdx)


	vunpcklpd(ymm7, ymm5, ymm0)
	vunpckhpd(ymm7, ymm5, ymm1)
	vunpcklpd(ymm11, ymm9, ymm2)
	vunpckhpd(ymm11, ymm9, ymm3)
	vinsertf128(imm(0x1), xmm2, ymm0, ymm5)
	vinsertf128(imm(0x1), xmm3, ymm1, ymm7)
	//vperm2f128(imm(0x31), ymm2, ymm0, ymm9)
	//vperm2f128(imm(0x31), ymm3, ymm1, ymm11)

	vbroadcastsd(mem(rbx), ymm3)

	vfmadd231pd(mem(rcx), ymm3, ymm5)
	vfmadd231pd(mem(rcx, rsi, 1), ymm3, ymm7)
	//vfmadd231pd(mem(rcx, rsi, 2), ymm3, ymm9)
	//vfmadd231pd(mem(rcx, rax, 1), ymm3, ymm11)
	vmovupd(ymm5, mem(rcx))
	vmovupd(ymm7, mem(rcx, rsi, 1))
	//vmovupd(ymm9, mem(rcx, rsi, 2))
	//vmovupd(ymm11, mem(rcx, rax, 1))

	//lea(mem(rcx, rsi, 4), rcx)

	vunpcklpd(ymm15, ymm13, ymm0)
	vunpckhpd(ymm15, ymm13, ymm1)
	//vextractf128(imm(0x1), ymm0, xmm2)
	//vextractf128(imm(0x1), ymm1, xmm4)

	vfmadd231pd(mem(rdx), xmm3, xmm0)
	vfmadd231pd(mem(rdx, rsi, 1), xmm3, xmm1)
	//vfmadd231pd(mem(rdx, rsi, 2), xmm3, xmm2)
	//vfmadd231pd(mem(rdx, rax, 1), xmm3, xmm4)
	vmovupd(xmm0, mem(rdx))
	vmovupd(xmm1, mem(rdx, rsi, 1))
	//vmovupd(xmm2, mem(rdx, rsi, 2))
	//vmovupd(xmm4, mem(rdx, rax, 1))

	//lea(mem(rdx, rsi, 4), rdx)


	jmp(.DDONE)                        // jump to end.
	
	
	
	
	label(.DBETAZERO)
	

	cmp(imm(8), rdi)                   // set ZF if (8*rs_c) == 8.
	jz(.DCOLSTORBZ)                    // jump to column storage case


	
	label(.DROWSTORBZ)
	
	
	vmovupd(ymm4, mem(rcx))
	vmovupd(xmm5, mem(rcx, rsi, 4))
	add(rdi, rcx)
	

	vmovupd(ymm6, mem(rcx))
	vmovupd(xmm7, mem(rcx, rsi, 4))
	add(rdi, rcx)
	
	
	vmovupd(ymm8, mem(rcx))
	vmovupd(xmm9, mem(rcx, rsi, 4))
	add(rdi, rcx)
	
	
	vmovupd(ymm10, mem(rcx))
	vmovupd(xmm11, mem(rcx, rsi, 4))
	add(rdi, rcx)
	
	
	vmovupd(ymm12, mem(rcx))
	vmovupd(xmm13, mem(rcx, rsi, 4))
	add(rdi, rcx)
	
	
	vmovupd(ymm14, mem(rcx))
	vmovupd(xmm15, mem(rcx, rsi, 4))
	//add(rdi, rcx)
	
	
	jmp(.DDONE)                        // jump to end.



	label(.DCOLSTORBZ)


	vunpcklpd(ymm6, ymm4, ymm0)
	vunpckhpd(ymm6, ymm4, ymm1)
	vunpcklpd(ymm10, ymm8, ymm2)
	vunpckhpd(ymm10, ymm8, ymm3)
	vinsertf128(imm(0x1), xmm2, ymm0, ymm4)
	vinsertf128(imm(0x1), xmm3, ymm1, ymm6)
	vperm2f128(imm(0x31), ymm2, ymm0, ymm8)
	vperm2f128(imm(0x31), ymm3, ymm1, ymm10)

	vmovupd(ymm4, mem(rcx))
	vmovupd(ymm6, mem(rcx, rsi, 1))
	vmovupd(ymm8, mem(rcx, rsi, 2))
	vmovupd(ymm10, mem(rcx, rax, 1))

	lea(mem(rcx, rsi, 4), rcx)

	vunpcklpd(ymm14, ymm12, ymm0)
	vunpckhpd(ymm14, ymm12, ymm1)
	vextractf128(imm(0x1), ymm0, xmm2)
	vextractf128(imm(0x1), ymm1, xmm4)

	vmovupd(xmm0, mem(rdx))
	vmovupd(xmm1, mem(rdx, rsi, 1))
	vmovupd(xmm2, mem(rdx, rsi, 2))
	vmovupd(xmm4, mem(rdx, rax, 1))

	lea(mem(rdx, rsi, 4), rdx)


	vunpcklpd(ymm7, ymm5, ymm0)
	vunpckhpd(ymm7, ymm5, ymm1)
	vunpcklpd(ymm11, ymm9, ymm2)
	vunpckhpd(ymm11, ymm9, ymm3)
	vinsertf128(imm(0x1), xmm2, ymm0, ymm5)
	vinsertf128(imm(0x1), xmm3, ymm1, ymm7)
	//vperm2f128(imm(0x31), ymm2, ymm0, ymm9)
	//vperm2f128(imm(0x31), ymm3, ymm1, ymm11)

	vmovupd(ymm5, mem(rcx))
	vmovupd(ymm7, mem(rcx, rsi, 1))
	//vmovupd(ymm9, mem(rcx, rsi, 2))
	//vmovupd(ymm11, mem(rcx, rax, 1))

	//lea(mem(rcx, rsi, 4), rcx)

	vunpcklpd(ymm15, ymm13, ymm0)
	vunpckhpd(ymm15, ymm13, ymm1)
	//vextractf128(imm(0x1), ymm0, xmm2)
	//vextractf128(imm(0x1), ymm1, xmm4)

	vmovupd(xmm0, mem(rdx))
	vmovupd(xmm1, mem(rdx, rsi, 1))
	//vmovupd(xmm2, mem(rdx, rsi, 2))
	//vmovupd(xmm4, mem(rdx, rax, 1))

	//lea(mem(rdx, rsi, 4), rdx)

	
	
	
	label(.DDONE)
	
	

    end_asm(
	: // output operands (none)
	: // input operands
      [k_iter] "m" (k_iter),
      [k_left] "m" (k_left),
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
	AOCL_DTL_TRACE_EXIT(AOCL_DTL_LEVEL_TRACE_7);
}

void bli_dgemmsup_rv_haswell_asm_5x6
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
	AOCL_DTL_TRACE_ENTRY(AOCL_DTL_LEVEL_TRACE_7);
	//void*    a_next = bli_auxinfo_next_a( data );
	//void*    b_next = bli_auxinfo_next_b( data );

	// Typecast local copies of integers in case dim_t and inc_t are a
	// different size than is expected by load instructions.
	uint64_t k_iter = k0 / 4;
	uint64_t k_left = k0 % 4;

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
	mov(var(cs_a), r9)                 // load cs_a
	lea(mem(, r8, 8), r8)              // rs_a *= sizeof(double)
	lea(mem(, r9, 8), r9)              // cs_a *= sizeof(double)
	
	lea(mem(r8, r8, 2), r13)           // r13 = 3*rs_a
	//lea(mem(r8, r8, 4), r15)           // r15 = 5*rs_a

	mov(var(b), rbx)                   // load address of b.
	mov(var(rs_b), r10)                // load rs_b
	//mov(var(cs_b), r11)                // load cs_b
	lea(mem(, r10, 8), r10)            // rs_b *= sizeof(double)
	//lea(mem(, r11, 8), r11)            // cs_b *= sizeof(double)

	                                   // NOTE: We cannot pre-load elements of a or b
	                                   // because it could eventually, in the last
	                                   // unrolled iter or the cleanup loop, result
	                                   // in reading beyond the bounds allocated mem
	                                   // (the likely result: a segmentation fault).

	mov(var(c), rcx)                   // load address of c
	mov(var(rs_c), rdi)                // load rs_c
	lea(mem(, rdi, 8), rdi)            // rs_c *= sizeof(double)



	cmp(imm(8), rdi)                   // set ZF if (8*rs_c) == 8.
	jz(.DCOLPFETCH)                    // jump to column storage case
	label(.DROWPFETCH)                 // row-stored prefetching on c

	lea(mem(rcx, rdi, 2), rdx)         //
	lea(mem(rdx, rdi, 1), rdx)         // rdx = c + 3*rs_c;
	prefetch(0, mem(rcx, 5*8))         // prefetch c + 0*rs_c
	prefetch(0, mem(rcx, rdi, 1, 5*8)) // prefetch c + 1*rs_c
	prefetch(0, mem(rcx, rdi, 2, 5*8)) // prefetch c + 2*rs_c
	prefetch(0, mem(rdx, 5*8))         // prefetch c + 3*rs_c
	prefetch(0, mem(rdx, rdi, 1, 5*8)) // prefetch c + 4*rs_c

	jmp(.DPOSTPFETCH)                  // jump to end of prefetching c
	label(.DCOLPFETCH)                 // column-stored prefetching c

	mov(var(cs_c), rsi)                // load cs_c to rsi (temporarily)
	lea(mem(, rsi, 8), rsi)            // cs_c *= sizeof(double)
	lea(mem(rcx, rsi, 2), rdx)         //
	lea(mem(rdx, rsi, 1), rdx)         // rdx = c + 3*cs_c;
	prefetch(0, mem(rcx, 4*8))         // prefetch c + 0*cs_c
	prefetch(0, mem(rcx, rsi, 1, 4*8)) // prefetch c + 1*cs_c
	prefetch(0, mem(rcx, rsi, 2, 4*8)) // prefetch c + 2*cs_c
	prefetch(0, mem(rdx, 4*8))         // prefetch c + 3*cs_c
	prefetch(0, mem(rdx, rsi, 1, 4*8)) // prefetch c + 4*cs_c
	prefetch(0, mem(rdx, rsi, 2, 4*8)) // prefetch c + 5*cs_c

	label(.DPOSTPFETCH)                // done prefetching c


	
	
	
	mov(var(k_iter), rsi)              // i = k_iter;
	test(rsi, rsi)                     // check i via logical AND.
	je(.DCONSIDKLEFT)                  // if i == 0, jump to code that
	                                   // contains the k_left loop.
	
	
	label(.DLOOPKITER)                 // MAIN LOOP
	
	
	// ---------------------------------- iteration 0

	vmovupd(mem(rbx,  0*32), ymm0)
	vmovupd(mem(rbx,  1*32), xmm1)
	add(r10, rbx)                      // b += rs_b;

	vbroadcastsd(mem(rax        ), ymm2)
	vbroadcastsd(mem(rax, r8,  1), ymm3)
	vfmadd231pd(ymm0, ymm2, ymm4)
	vfmadd231pd(ymm1, ymm2, ymm5)
	vfmadd231pd(ymm0, ymm3, ymm6)
	vfmadd231pd(ymm1, ymm3, ymm7)
	
	vbroadcastsd(mem(rax, r8,  2), ymm2)
	vbroadcastsd(mem(rax, r13, 1), ymm3)
	vfmadd231pd(ymm0, ymm2, ymm8)
	vfmadd231pd(ymm1, ymm2, ymm9)
	vfmadd231pd(ymm0, ymm3, ymm10)
	vfmadd231pd(ymm1, ymm3, ymm11)
	
	vbroadcastsd(mem(rax, r8,  4), ymm2)
	add(r9, rax)                       // a += cs_a;
	vfmadd231pd(ymm0, ymm2, ymm12)
	vfmadd231pd(ymm1, ymm2, ymm13)

	
	// ---------------------------------- iteration 1

	vmovupd(mem(rbx,  0*32), ymm0)
	vmovupd(mem(rbx,  1*32), xmm1)
	add(r10, rbx)                      // b += rs_b;

	vbroadcastsd(mem(rax        ), ymm2)
	vbroadcastsd(mem(rax, r8,  1), ymm3)
	vfmadd231pd(ymm0, ymm2, ymm4)
	vfmadd231pd(ymm1, ymm2, ymm5)
	vfmadd231pd(ymm0, ymm3, ymm6)
	vfmadd231pd(ymm1, ymm3, ymm7)
	
	vbroadcastsd(mem(rax, r8,  2), ymm2)
	vbroadcastsd(mem(rax, r13, 1), ymm3)
	vfmadd231pd(ymm0, ymm2, ymm8)
	vfmadd231pd(ymm1, ymm2, ymm9)
	vfmadd231pd(ymm0, ymm3, ymm10)
	vfmadd231pd(ymm1, ymm3, ymm11)
	
	vbroadcastsd(mem(rax, r8,  4), ymm2)
	add(r9, rax)                       // a += cs_a;
	vfmadd231pd(ymm0, ymm2, ymm12)
	vfmadd231pd(ymm1, ymm2, ymm13)
	
	
	// ---------------------------------- iteration 2

	vmovupd(mem(rbx,  0*32), ymm0)
	vmovupd(mem(rbx,  1*32), xmm1)
	add(r10, rbx)                      // b += rs_b;

	vbroadcastsd(mem(rax        ), ymm2)
	vbroadcastsd(mem(rax, r8,  1), ymm3)
	vfmadd231pd(ymm0, ymm2, ymm4)
	vfmadd231pd(ymm1, ymm2, ymm5)
	vfmadd231pd(ymm0, ymm3, ymm6)
	vfmadd231pd(ymm1, ymm3, ymm7)
	
	vbroadcastsd(mem(rax, r8,  2), ymm2)
	vbroadcastsd(mem(rax, r13, 1), ymm3)
	vfmadd231pd(ymm0, ymm2, ymm8)
	vfmadd231pd(ymm1, ymm2, ymm9)
	vfmadd231pd(ymm0, ymm3, ymm10)
	vfmadd231pd(ymm1, ymm3, ymm11)
	
	vbroadcastsd(mem(rax, r8,  4), ymm2)
	add(r9, rax)                       // a += cs_a;
	vfmadd231pd(ymm0, ymm2, ymm12)
	vfmadd231pd(ymm1, ymm2, ymm13)
	

	// ---------------------------------- iteration 3

	vmovupd(mem(rbx, 0*32), ymm0)
	vmovupd(mem(rbx, 1*32), xmm1)
	add(r10, rbx)                      // b += rs_b;

	vbroadcastsd(mem(rax        ), ymm2)
	vbroadcastsd(mem(rax, r8,  1), ymm3)
	vfmadd231pd(ymm0, ymm2, ymm4)
	vfmadd231pd(ymm1, ymm2, ymm5)
	vfmadd231pd(ymm0, ymm3, ymm6)
	vfmadd231pd(ymm1, ymm3, ymm7)
	
	vbroadcastsd(mem(rax, r8,  2), ymm2)
	vbroadcastsd(mem(rax, r13, 1), ymm3)
	vfmadd231pd(ymm0, ymm2, ymm8)
	vfmadd231pd(ymm1, ymm2, ymm9)
	vfmadd231pd(ymm0, ymm3, ymm10)
	vfmadd231pd(ymm1, ymm3, ymm11)
	
	vbroadcastsd(mem(rax, r8,  4), ymm2)
	add(r9, rax)                       // a += cs_a;
	vfmadd231pd(ymm0, ymm2, ymm12)
	vfmadd231pd(ymm1, ymm2, ymm13)
	
	
	
	dec(rsi)                           // i -= 1;
	jne(.DLOOPKITER)                   // iterate again if i != 0.
	
	
	
	
	
	
	label(.DCONSIDKLEFT)
	
	mov(var(k_left), rsi)              // i = k_left;
	test(rsi, rsi)                     // check i via logical AND.
	je(.DPOSTACCUM)                    // if i == 0, we're done; jump to end.
	                                   // else, we prepare to enter k_left loop.
	
	
	label(.DLOOPKLEFT)                 // EDGE LOOP
	
	vmovupd(mem(rbx,  0*32), ymm0)
	vmovupd(mem(rbx,  1*32), xmm1)
	add(r10, rbx)                      // b += rs_b;
	
	vbroadcastsd(mem(rax        ), ymm2)
	vbroadcastsd(mem(rax, r8,  1), ymm3)
	vfmadd231pd(ymm0, ymm2, ymm4)
	vfmadd231pd(ymm1, ymm2, ymm5)
	vfmadd231pd(ymm0, ymm3, ymm6)
	vfmadd231pd(ymm1, ymm3, ymm7)
	
	vbroadcastsd(mem(rax, r8,  2), ymm2)
	vbroadcastsd(mem(rax, r13, 1), ymm3)
	vfmadd231pd(ymm0, ymm2, ymm8)
	vfmadd231pd(ymm1, ymm2, ymm9)
	vfmadd231pd(ymm0, ymm3, ymm10)
	vfmadd231pd(ymm1, ymm3, ymm11)
	
	vbroadcastsd(mem(rax, r8,  4), ymm2)
	add(r9, rax)                       // a += cs_a;
	vfmadd231pd(ymm0, ymm2, ymm12)
	vfmadd231pd(ymm1, ymm2, ymm13)
	
	
	dec(rsi)                           // i -= 1;
	jne(.DLOOPKLEFT)                   // iterate again if i != 0.
	
	
	
	label(.DPOSTACCUM)

	
	
	mov(var(alpha), rax)               // load address of alpha
	mov(var(beta), rbx)                // load address of beta
	vbroadcastsd(mem(rax), ymm0)       // load alpha and duplicate
	vbroadcastsd(mem(rbx), ymm3)       // load beta and duplicate
	
	vmulpd(ymm0, ymm4, ymm4)           // scale by alpha
	vmulpd(xmm0, xmm5, xmm5)
	vmulpd(ymm0, ymm6, ymm6)
	vmulpd(xmm0, xmm7, xmm7)
	vmulpd(ymm0, ymm8, ymm8)
	vmulpd(xmm0, xmm9, xmm9)
	vmulpd(ymm0, ymm10, ymm10)
	vmulpd(xmm0, xmm11, xmm11)
	vmulpd(ymm0, ymm12, ymm12)
	vmulpd(xmm0, xmm13, xmm13)
	
	
	
	
	
	
	mov(var(cs_c), rsi)                // load cs_c
	lea(mem(, rsi, 8), rsi)            // rsi = cs_c * sizeof(double)
	
	//lea(mem(rcx, rsi, 4), rdx)         // load address of c +  4*cs_c;
	lea(mem(rcx, rdi, 4), rdx)         // load address of c +  4*rs_c;

	lea(mem(rsi, rsi, 2), rax)         // rax = 3*cs_c;
	
	
	
	                                   // now avoid loading C if beta == 0
	
	vxorpd(ymm0, ymm0, ymm0)           // set ymm0 to zero.
	vucomisd(xmm0, xmm3)               // set ZF if beta == 0.
	je(.DBETAZERO)                     // if ZF = 1, jump to beta == 0 case



	cmp(imm(8), rdi)                   // set ZF if (8*rs_c) == 8.
	jz(.DCOLSTORED)                    // jump to column storage case
	

	
	label(.DROWSTORED)
	
	
	vfmadd231pd(mem(rcx), ymm3, ymm4)
	vmovupd(ymm4, mem(rcx))

	vfmadd231pd(mem(rcx, rsi, 4), xmm3, xmm5)
	vmovupd(xmm5, mem(rcx, rsi, 4))
	add(rdi, rcx)
	
	
	vfmadd231pd(mem(rcx), ymm3, ymm6)
	vmovupd(ymm6, mem(rcx))

	vfmadd231pd(mem(rcx, rsi, 4), xmm3, xmm7)
	vmovupd(xmm7, mem(rcx, rsi, 4))
	add(rdi, rcx)
	
	
	vfmadd231pd(mem(rcx), ymm3, ymm8)
	vmovupd(ymm8, mem(rcx))

	vfmadd231pd(mem(rcx, rsi, 4), xmm3, xmm9)
	vmovupd(xmm9, mem(rcx, rsi, 4))
	add(rdi, rcx)
	
	
	vfmadd231pd(mem(rcx), ymm3, ymm10)
	vmovupd(ymm10, mem(rcx))

	vfmadd231pd(mem(rcx, rsi, 4), xmm3, xmm11)
	vmovupd(xmm11, mem(rcx, rsi, 4))
	add(rdi, rcx)
	
	
	vfmadd231pd(mem(rcx), ymm3, ymm12)
	vmovupd(ymm12, mem(rcx))

	vfmadd231pd(mem(rcx, rsi, 4), xmm3, xmm13)
	vmovupd(xmm13, mem(rcx, rsi, 4))
	//add(rdi, rcx)
	
	
	jmp(.DDONE)                        // jump to end.



	label(.DCOLSTORED)


	vunpcklpd(ymm6, ymm4, ymm0)
	vunpckhpd(ymm6, ymm4, ymm1)
	vunpcklpd(ymm10, ymm8, ymm2)
	vunpckhpd(ymm10, ymm8, ymm3)
	vinsertf128(imm(0x1), xmm2, ymm0, ymm4)
	vinsertf128(imm(0x1), xmm3, ymm1, ymm6)
	vperm2f128(imm(0x31), ymm2, ymm0, ymm8)
	vperm2f128(imm(0x31), ymm3, ymm1, ymm10)

	vbroadcastsd(mem(rbx), ymm3)

	vfmadd231pd(mem(rcx), ymm3, ymm4)
	vfmadd231pd(mem(rcx, rsi, 1), ymm3, ymm6)
	vfmadd231pd(mem(rcx, rsi, 2), ymm3, ymm8)
	vfmadd231pd(mem(rcx, rax, 1), ymm3, ymm10)
	vmovupd(ymm4, mem(rcx))
	vmovupd(ymm6, mem(rcx, rsi, 1))
	vmovupd(ymm8, mem(rcx, rsi, 2))
	vmovupd(ymm10, mem(rcx, rax, 1))

	lea(mem(rcx, rsi, 4), rcx)

#if 0
	vunpcklpd(ymm14, ymm12, ymm0)
	vunpckhpd(ymm14, ymm12, ymm1)
	vextractf128(imm(0x1), ymm0, xmm2)
	vextractf128(imm(0x1), ymm1, xmm4)

	vfmadd231pd(mem(rdx), xmm3, xmm0)
	vfmadd231pd(mem(rdx, rsi, 1), xmm3, xmm1)
	vfmadd231pd(mem(rdx, rsi, 2), xmm3, xmm2)
	vfmadd231pd(mem(rdx, rax, 1), xmm3, xmm4)
	vmovupd(xmm0, mem(rdx))
	vmovupd(xmm1, mem(rdx, rsi, 1))
	vmovupd(xmm2, mem(rdx, rsi, 2))
	vmovupd(xmm4, mem(rdx, rax, 1))
#else
	vmovlpd(mem(rdx), xmm0, xmm0)
	vmovhpd(mem(rdx, rsi, 1), xmm0, xmm0)
	vmovlpd(mem(rdx, rsi, 2), xmm1, xmm1)
	vmovhpd(mem(rdx, rax, 1), xmm1, xmm1)
	vperm2f128(imm(0x20), ymm1, ymm0, ymm0)

	vfmadd213pd(ymm12, ymm3, ymm0)
	vextractf128(imm(1), ymm0, xmm1)
	vmovlpd(xmm0, mem(rdx))
	vmovhpd(xmm0, mem(rdx, rsi, 1))
	vmovlpd(xmm1, mem(rdx, rsi, 2))
	vmovhpd(xmm1, mem(rdx, rax, 1))
#endif

	lea(mem(rdx, rsi, 4), rdx)


	vunpcklpd(ymm7, ymm5, ymm0)
	vunpckhpd(ymm7, ymm5, ymm1)
	vunpcklpd(ymm11, ymm9, ymm2)
	vunpckhpd(ymm11, ymm9, ymm3)
	vinsertf128(imm(0x1), xmm2, ymm0, ymm5)
	vinsertf128(imm(0x1), xmm3, ymm1, ymm7)
	//vperm2f128(imm(0x31), ymm2, ymm0, ymm9)
	//vperm2f128(imm(0x31), ymm3, ymm1, ymm11)

	vbroadcastsd(mem(rbx), ymm3)

	vfmadd231pd(mem(rcx), ymm3, ymm5)
	vfmadd231pd(mem(rcx, rsi, 1), ymm3, ymm7)
	//vfmadd231pd(mem(rcx, rsi, 2), ymm3, ymm9)
	//vfmadd231pd(mem(rcx, rax, 1), ymm3, ymm11)
	vmovupd(ymm5, mem(rcx))
	vmovupd(ymm7, mem(rcx, rsi, 1))
	//vmovupd(ymm9, mem(rcx, rsi, 2))
	//vmovupd(ymm11, mem(rcx, rax, 1))

	//lea(mem(rcx, rsi, 4), rcx)

#if 0
	vunpcklpd(ymm15, ymm13, ymm0)
	vunpckhpd(ymm15, ymm13, ymm1)
	//vextractf128(imm(0x1), ymm0, xmm2)
	//vextractf128(imm(0x1), ymm1, xmm4)

	vfmadd231pd(mem(rdx), xmm3, xmm0)
	vfmadd231pd(mem(rdx, rsi, 1), xmm3, xmm1)
	//vfmadd231pd(mem(rdx, rsi, 2), xmm3, xmm2)
	//vfmadd231pd(mem(rdx, rax, 1), xmm3, xmm4)
	vmovupd(xmm0, mem(rdx))
	vmovupd(xmm1, mem(rdx, rsi, 1))
	//vmovupd(xmm2, mem(rdx, rsi, 2))
	//vmovupd(xmm4, mem(rdx, rax, 1))
#else
	vmovlpd(mem(rdx), xmm0, xmm0)
	vmovhpd(mem(rdx, rsi, 1), xmm0, xmm0)
	//vmovlpd(mem(rdx, rsi, 2), xmm1, xmm1)
	//vmovhpd(mem(rdx, rax, 1), xmm1, xmm1)
	//vperm2f128(imm(0x20), ymm1, ymm0, ymm0)

	vfmadd213pd(xmm13, xmm3, xmm0)
	//vextractf128(imm(1), ymm0, xmm1)
	vmovlpd(xmm0, mem(rdx))
	vmovhpd(xmm0, mem(rdx, rsi, 1))
	//vmovlpd(xmm1, mem(rdx, rsi, 2))
	//vmovhpd(xmm1, mem(rdx, rax, 1))
#endif

	//lea(mem(rdx, rsi, 4), rdx)


	jmp(.DDONE)                        // jump to end.
	
	
	
	
	label(.DBETAZERO)
	

	cmp(imm(8), rdi)                   // set ZF if (8*rs_c) == 8.
	jz(.DCOLSTORBZ)                    // jump to column storage case


	
	label(.DROWSTORBZ)
	
	
	vmovupd(ymm4, mem(rcx))
	vmovupd(xmm5, mem(rcx, rsi, 4))
	add(rdi, rcx)
	

	vmovupd(ymm6, mem(rcx))
	vmovupd(xmm7, mem(rcx, rsi, 4))
	add(rdi, rcx)
	
	
	vmovupd(ymm8, mem(rcx))
	vmovupd(xmm9, mem(rcx, rsi, 4))
	add(rdi, rcx)
	
	
	vmovupd(ymm10, mem(rcx))
	vmovupd(xmm11, mem(rcx, rsi, 4))
	add(rdi, rcx)
	
	
	vmovupd(ymm12, mem(rcx))
	vmovupd(xmm13, mem(rcx, rsi, 4))
	//add(rdi, rcx)
	
	
	jmp(.DDONE)                        // jump to end.



	label(.DCOLSTORBZ)


	vunpcklpd(ymm6, ymm4, ymm0)
	vunpckhpd(ymm6, ymm4, ymm1)
	vunpcklpd(ymm10, ymm8, ymm2)
	vunpckhpd(ymm10, ymm8, ymm3)
	vinsertf128(imm(0x1), xmm2, ymm0, ymm4)
	vinsertf128(imm(0x1), xmm3, ymm1, ymm6)
	vperm2f128(imm(0x31), ymm2, ymm0, ymm8)
	vperm2f128(imm(0x31), ymm3, ymm1, ymm10)

	vmovupd(ymm4, mem(rcx))
	vmovupd(ymm6, mem(rcx, rsi, 1))
	vmovupd(ymm8, mem(rcx, rsi, 2))
	vmovupd(ymm10, mem(rcx, rax, 1))

	lea(mem(rcx, rsi, 4), rcx)

#if 0
	vunpcklpd(ymm14, ymm12, ymm0)
	vunpckhpd(ymm14, ymm12, ymm1)
	vextractf128(imm(0x1), ymm0, xmm2)
	vextractf128(imm(0x1), ymm1, xmm4)

	vmovupd(xmm0, mem(rdx))
	vmovupd(xmm1, mem(rdx, rsi, 1))
	vmovupd(xmm2, mem(rdx, rsi, 2))
	vmovupd(xmm4, mem(rdx, rax, 1))
#else
	vmovupd(ymm12, ymm0)

	vextractf128(imm(1), ymm0, xmm1)
	vmovlpd(xmm0, mem(rdx))
	vmovhpd(xmm0, mem(rdx, rsi, 1))
	vmovlpd(xmm1, mem(rdx, rsi, 2))
	vmovhpd(xmm1, mem(rdx, rax, 1))
#endif

	lea(mem(rdx, rsi, 4), rdx)


	vunpcklpd(ymm7, ymm5, ymm0)
	vunpckhpd(ymm7, ymm5, ymm1)
	vunpcklpd(ymm11, ymm9, ymm2)
	vunpckhpd(ymm11, ymm9, ymm3)
	vinsertf128(imm(0x1), xmm2, ymm0, ymm5)
	vinsertf128(imm(0x1), xmm3, ymm1, ymm7)
	//vperm2f128(imm(0x31), ymm2, ymm0, ymm9)
	//vperm2f128(imm(0x31), ymm3, ymm1, ymm11)

	vmovupd(ymm5, mem(rcx))
	vmovupd(ymm7, mem(rcx, rsi, 1))
	//vmovupd(ymm9, mem(rcx, rsi, 2))
	//vmovupd(ymm11, mem(rcx, rax, 1))

	//lea(mem(rcx, rsi, 4), rcx)

#if 0
	vunpcklpd(ymm15, ymm13, ymm0)
	vunpckhpd(ymm15, ymm13, ymm1)
	//vextractf128(imm(0x1), ymm0, xmm2)
	//vextractf128(imm(0x1), ymm1, xmm4)

	vmovupd(xmm0, mem(rdx))
	vmovupd(xmm1, mem(rdx, rsi, 1))
	//vmovupd(xmm2, mem(rdx, rsi, 2))
	//vmovupd(xmm4, mem(rdx, rax, 1))
#else
	vmovupd(ymm13, ymm0)

	//vextractf128(imm(1), ymm0, xmm1)
	vmovlpd(xmm0, mem(rdx))
	vmovhpd(xmm0, mem(rdx, rsi, 1))
	//vmovlpd(xmm1, mem(rdx, rsi, 2))
	//vmovhpd(xmm1, mem(rdx, rax, 1))
#endif

	//lea(mem(rdx, rsi, 4), rdx)

	
	
	
	label(.DDONE)
	
	

    end_asm(
	: // output operands (none)
	: // input operands
      [k_iter] "m" (k_iter),
      [k_left] "m" (k_left),
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
	  "ymm8", "ymm9", "ymm10", "ymm11", "ymm12", "ymm13", "memory"
	)

	AOCL_DTL_TRACE_EXIT(AOCL_DTL_LEVEL_TRACE_7);
}

void bli_dgemmsup_rv_haswell_asm_4x6
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
	AOCL_DTL_TRACE_ENTRY(AOCL_DTL_LEVEL_TRACE_7);
	//void*    a_next = bli_auxinfo_next_a( data );
	//void*    b_next = bli_auxinfo_next_b( data );

	// Typecast local copies of integers in case dim_t and inc_t are a
	// different size than is expected by load instructions.
	uint64_t k_iter = k0 / 4;
	uint64_t k_left = k0 % 4;

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
	mov(var(cs_a), r9)                 // load cs_a
	lea(mem(, r8, 8), r8)              // rs_a *= sizeof(double)
	lea(mem(, r9, 8), r9)              // cs_a *= sizeof(double)
	
	lea(mem(r8, r8, 2), r13)           // r13 = 3*rs_a
	//lea(mem(r8, r8, 4), r15)           // r15 = 5*rs_a

	mov(var(b), rbx)                   // load address of b.
	mov(var(rs_b), r10)                // load rs_b
	//mov(var(cs_b), r11)                // load cs_b
	lea(mem(, r10, 8), r10)            // rs_b *= sizeof(double)
	//lea(mem(, r11, 8), r11)            // cs_b *= sizeof(double)

	                                   // NOTE: We cannot pre-load elements of a or b
	                                   // because it could eventually, in the last
	                                   // unrolled iter or the cleanup loop, result
	                                   // in reading beyond the bounds allocated mem
	                                   // (the likely result: a segmentation fault).

	mov(var(c), rcx)                   // load address of c
	mov(var(rs_c), rdi)                // load rs_c
	lea(mem(, rdi, 8), rdi)            // rs_c *= sizeof(double)



	cmp(imm(8), rdi)                   // set ZF if (8*rs_c) == 8.
	jz(.DCOLPFETCH)                    // jump to column storage case
	label(.DROWPFETCH)                 // row-stored prefetching on c

	lea(mem(rcx, rdi, 2), rdx)         //
	lea(mem(rdx, rdi, 1), rdx)         // rdx = c + 3*rs_c;
	prefetch(0, mem(rcx, 5*8))         // prefetch c + 0*rs_c
	prefetch(0, mem(rcx, rdi, 1, 5*8)) // prefetch c + 1*rs_c
	prefetch(0, mem(rcx, rdi, 2, 5*8)) // prefetch c + 2*rs_c
	prefetch(0, mem(rdx, 5*8))         // prefetch c + 3*rs_c

	jmp(.DPOSTPFETCH)                  // jump to end of prefetching c
	label(.DCOLPFETCH)                 // column-stored prefetching c

	mov(var(cs_c), rsi)                // load cs_c to rsi (temporarily)
	lea(mem(, rsi, 8), rsi)            // cs_c *= sizeof(double)
	lea(mem(rcx, rsi, 2), rdx)         //
	lea(mem(rdx, rsi, 1), rdx)         // rdx = c + 3*cs_c;
	prefetch(0, mem(rcx, 3*8))         // prefetch c + 0*cs_c
	prefetch(0, mem(rcx, rsi, 1, 3*8)) // prefetch c + 1*cs_c
	prefetch(0, mem(rcx, rsi, 2, 3*8)) // prefetch c + 2*cs_c
	prefetch(0, mem(rdx, 3*8))         // prefetch c + 3*cs_c
	prefetch(0, mem(rdx, rsi, 1, 3*8)) // prefetch c + 4*cs_c
	prefetch(0, mem(rdx, rsi, 2, 3*8)) // prefetch c + 5*cs_c

	label(.DPOSTPFETCH)                // done prefetching c


	
	
	mov(var(k_iter), rsi)              // i = k_iter;
	test(rsi, rsi)                     // check i via logical AND.
	je(.DCONSIDKLEFT)                  // if i == 0, jump to code that
	                                   // contains the k_left loop.
	
	
	label(.DLOOPKITER)                 // MAIN LOOP
	
	
	// ---------------------------------- iteration 0

	vmovupd(mem(rbx,  0*32), ymm0)
	vmovupd(mem(rbx,  1*32), xmm1)
	add(r10, rbx)                      // b += rs_b;

	vbroadcastsd(mem(rax        ), ymm2)
	vbroadcastsd(mem(rax, r8,  1), ymm3)
	vfmadd231pd(ymm0, ymm2, ymm4)
	vfmadd231pd(ymm1, ymm2, ymm5)
	vfmadd231pd(ymm0, ymm3, ymm6)
	vfmadd231pd(ymm1, ymm3, ymm7)
	
	vbroadcastsd(mem(rax, r8,  2), ymm2)
	vbroadcastsd(mem(rax, r13, 1), ymm3)
	add(r9, rax)                       // a += cs_a;
	vfmadd231pd(ymm0, ymm2, ymm8)
	vfmadd231pd(ymm1, ymm2, ymm9)
	vfmadd231pd(ymm0, ymm3, ymm10)
	vfmadd231pd(ymm1, ymm3, ymm11)

	
	// ---------------------------------- iteration 1

	vmovupd(mem(rbx,  0*32), ymm0)
	vmovupd(mem(rbx,  1*32), xmm1)
	add(r10, rbx)                      // b += rs_b;

	vbroadcastsd(mem(rax        ), ymm2)
	vbroadcastsd(mem(rax, r8,  1), ymm3)
	vfmadd231pd(ymm0, ymm2, ymm4)
	vfmadd231pd(ymm1, ymm2, ymm5)
	vfmadd231pd(ymm0, ymm3, ymm6)
	vfmadd231pd(ymm1, ymm3, ymm7)
	
	vbroadcastsd(mem(rax, r8,  2), ymm2)
	vbroadcastsd(mem(rax, r13, 1), ymm3)
	add(r9, rax)                       // a += cs_a;
	vfmadd231pd(ymm0, ymm2, ymm8)
	vfmadd231pd(ymm1, ymm2, ymm9)
	vfmadd231pd(ymm0, ymm3, ymm10)
	vfmadd231pd(ymm1, ymm3, ymm11)
	
	
	// ---------------------------------- iteration 2

	vmovupd(mem(rbx,  0*32), ymm0)
	vmovupd(mem(rbx,  1*32), xmm1)
	add(r10, rbx)                      // b += rs_b;

	vbroadcastsd(mem(rax        ), ymm2)
	vbroadcastsd(mem(rax, r8,  1), ymm3)
	vfmadd231pd(ymm0, ymm2, ymm4)
	vfmadd231pd(ymm1, ymm2, ymm5)
	vfmadd231pd(ymm0, ymm3, ymm6)
	vfmadd231pd(ymm1, ymm3, ymm7)
	
	vbroadcastsd(mem(rax, r8,  2), ymm2)
	vbroadcastsd(mem(rax, r13, 1), ymm3)
	add(r9, rax)                       // a += cs_a;
	vfmadd231pd(ymm0, ymm2, ymm8)
	vfmadd231pd(ymm1, ymm2, ymm9)
	vfmadd231pd(ymm0, ymm3, ymm10)
	vfmadd231pd(ymm1, ymm3, ymm11)
	

	// ---------------------------------- iteration 3

	vmovupd(mem(rbx, 0*32), ymm0)
	vmovupd(mem(rbx, 1*32), xmm1)
	add(r10, rbx)                      // b += rs_b;

	vbroadcastsd(mem(rax        ), ymm2)
	vbroadcastsd(mem(rax, r8,  1), ymm3)
	vfmadd231pd(ymm0, ymm2, ymm4)
	vfmadd231pd(ymm1, ymm2, ymm5)
	vfmadd231pd(ymm0, ymm3, ymm6)
	vfmadd231pd(ymm1, ymm3, ymm7)
	
	vbroadcastsd(mem(rax, r8,  2), ymm2)
	vbroadcastsd(mem(rax, r13, 1), ymm3)
	add(r9, rax)                       // a += cs_a;
	vfmadd231pd(ymm0, ymm2, ymm8)
	vfmadd231pd(ymm1, ymm2, ymm9)
	vfmadd231pd(ymm0, ymm3, ymm10)
	vfmadd231pd(ymm1, ymm3, ymm11)
	
	
	
	dec(rsi)                           // i -= 1;
	jne(.DLOOPKITER)                   // iterate again if i != 0.
	
	
	
	
	
	
	label(.DCONSIDKLEFT)
	
	mov(var(k_left), rsi)              // i = k_left;
	test(rsi, rsi)                     // check i via logical AND.
	je(.DPOSTACCUM)                    // if i == 0, we're done; jump to end.
	                                   // else, we prepare to enter k_left loop.
	
	
	label(.DLOOPKLEFT)                 // EDGE LOOP
	
	vmovupd(mem(rbx,  0*32), ymm0)
	vmovupd(mem(rbx,  1*32), xmm1)
	add(r10, rbx)                      // b += rs_b;
	
	vbroadcastsd(mem(rax        ), ymm2)
	vbroadcastsd(mem(rax, r8,  1), ymm3)
	vfmadd231pd(ymm0, ymm2, ymm4)
	vfmadd231pd(ymm1, ymm2, ymm5)
	vfmadd231pd(ymm0, ymm3, ymm6)
	vfmadd231pd(ymm1, ymm3, ymm7)
	
	vbroadcastsd(mem(rax, r8,  2), ymm2)
	vbroadcastsd(mem(rax, r13, 1), ymm3)
	add(r9, rax)                       // a += cs_a;
	vfmadd231pd(ymm0, ymm2, ymm8)
	vfmadd231pd(ymm1, ymm2, ymm9)
	vfmadd231pd(ymm0, ymm3, ymm10)
	vfmadd231pd(ymm1, ymm3, ymm11)
	
	
	dec(rsi)                           // i -= 1;
	jne(.DLOOPKLEFT)                   // iterate again if i != 0.
	
	
	
	label(.DPOSTACCUM)

	
	
	mov(var(alpha), rax)               // load address of alpha
	mov(var(beta), rbx)                // load address of beta
	vbroadcastsd(mem(rax), ymm0)       // load alpha and duplicate
	vbroadcastsd(mem(rbx), ymm3)       // load beta and duplicate
	
	vmulpd(ymm0, ymm4, ymm4)           // scale by alpha
	vmulpd(xmm0, xmm5, xmm5)
	vmulpd(ymm0, ymm6, ymm6)
	vmulpd(xmm0, xmm7, xmm7)
	vmulpd(ymm0, ymm8, ymm8)
	vmulpd(xmm0, xmm9, xmm9)
	vmulpd(ymm0, ymm10, ymm10)
	vmulpd(xmm0, xmm11, xmm11)
	
	
	
	
	
	
	mov(var(cs_c), rsi)                // load cs_c
	lea(mem(, rsi, 8), rsi)            // rsi = cs_c * sizeof(double)
	
	//lea(mem(rcx, rsi, 4), rdx)         // load address of c +  4*cs_c;
	//lea(mem(rcx, rdi, 4), rdx)         // load address of c +  4*rs_c;

	lea(mem(rsi, rsi, 2), rax)         // rax = 3*cs_c;
	
	
	
	                                   // now avoid loading C if beta == 0
	
	vxorpd(ymm0, ymm0, ymm0)           // set ymm0 to zero.
	vucomisd(xmm0, xmm3)               // set ZF if beta == 0.
	je(.DBETAZERO)                     // if ZF = 1, jump to beta == 0 case



	cmp(imm(8), rdi)                   // set ZF if (8*rs_c) == 8.
	jz(.DCOLSTORED)                    // jump to column storage case
	

	
	label(.DROWSTORED)
	
	
	vfmadd231pd(mem(rcx), ymm3, ymm4)
	vmovupd(ymm4, mem(rcx))

	vfmadd231pd(mem(rcx, rsi, 4), xmm3, xmm5)
	vmovupd(xmm5, mem(rcx, rsi, 4))
	add(rdi, rcx)
	
	
	vfmadd231pd(mem(rcx), ymm3, ymm6)
	vmovupd(ymm6, mem(rcx))

	vfmadd231pd(mem(rcx, rsi, 4), xmm3, xmm7)
	vmovupd(xmm7, mem(rcx, rsi, 4))
	add(rdi, rcx)
	
	
	vfmadd231pd(mem(rcx), ymm3, ymm8)
	vmovupd(ymm8, mem(rcx))

	vfmadd231pd(mem(rcx, rsi, 4), xmm3, xmm9)
	vmovupd(xmm9, mem(rcx, rsi, 4))
	add(rdi, rcx)
	
	
	vfmadd231pd(mem(rcx), ymm3, ymm10)
	vmovupd(ymm10, mem(rcx))

	vfmadd231pd(mem(rcx, rsi, 4), xmm3, xmm11)
	vmovupd(xmm11, mem(rcx, rsi, 4))
	//add(rdi, rcx)
	
	
	jmp(.DDONE)                        // jump to end.



	label(.DCOLSTORED)


	vunpcklpd(ymm6, ymm4, ymm0)
	vunpckhpd(ymm6, ymm4, ymm1)
	vunpcklpd(ymm10, ymm8, ymm2)
	vunpckhpd(ymm10, ymm8, ymm3)
	vinsertf128(imm(0x1), xmm2, ymm0, ymm4)
	vinsertf128(imm(0x1), xmm3, ymm1, ymm6)
	vperm2f128(imm(0x31), ymm2, ymm0, ymm8)
	vperm2f128(imm(0x31), ymm3, ymm1, ymm10)

	vbroadcastsd(mem(rbx), ymm3)

	vfmadd231pd(mem(rcx), ymm3, ymm4)
	vfmadd231pd(mem(rcx, rsi, 1), ymm3, ymm6)
	vfmadd231pd(mem(rcx, rsi, 2), ymm3, ymm8)
	vfmadd231pd(mem(rcx, rax, 1), ymm3, ymm10)
	vmovupd(ymm4, mem(rcx))
	vmovupd(ymm6, mem(rcx, rsi, 1))
	vmovupd(ymm8, mem(rcx, rsi, 2))
	vmovupd(ymm10, mem(rcx, rax, 1))

	lea(mem(rcx, rsi, 4), rcx)


	vunpcklpd(ymm7, ymm5, ymm0)
	vunpckhpd(ymm7, ymm5, ymm1)
	vunpcklpd(ymm11, ymm9, ymm2)
	vunpckhpd(ymm11, ymm9, ymm3)
	vinsertf128(imm(0x1), xmm2, ymm0, ymm5)
	vinsertf128(imm(0x1), xmm3, ymm1, ymm7)
	//vperm2f128(imm(0x31), ymm2, ymm0, ymm9)
	//vperm2f128(imm(0x31), ymm3, ymm1, ymm11)

	vbroadcastsd(mem(rbx), ymm3)

	vfmadd231pd(mem(rcx), ymm3, ymm5)
	vfmadd231pd(mem(rcx, rsi, 1), ymm3, ymm7)
	//vfmadd231pd(mem(rcx, rsi, 2), ymm3, ymm9)
	//vfmadd231pd(mem(rcx, rax, 1), ymm3, ymm11)
	vmovupd(ymm5, mem(rcx))
	vmovupd(ymm7, mem(rcx, rsi, 1))
	//vmovupd(ymm9, mem(rcx, rsi, 2))
	//vmovupd(ymm11, mem(rcx, rax, 1))

	//lea(mem(rcx, rsi, 4), rcx)


	jmp(.DDONE)                        // jump to end.
	
	
	
	
	label(.DBETAZERO)
	

	cmp(imm(8), rdi)                   // set ZF if (8*rs_c) == 8.
	jz(.DCOLSTORBZ)                    // jump to column storage case


	
	label(.DROWSTORBZ)
	
	
	vmovupd(ymm4, mem(rcx))
	vmovupd(xmm5, mem(rcx, rsi, 4))
	add(rdi, rcx)
	

	vmovupd(ymm6, mem(rcx))
	vmovupd(xmm7, mem(rcx, rsi, 4))
	add(rdi, rcx)
	
	
	vmovupd(ymm8, mem(rcx))
	vmovupd(xmm9, mem(rcx, rsi, 4))
	add(rdi, rcx)
	
	
	vmovupd(ymm10, mem(rcx))
	vmovupd(xmm11, mem(rcx, rsi, 4))
	//add(rdi, rcx)
	
	
	jmp(.DDONE)                        // jump to end.



	label(.DCOLSTORBZ)


	vunpcklpd(ymm6, ymm4, ymm0)
	vunpckhpd(ymm6, ymm4, ymm1)
	vunpcklpd(ymm10, ymm8, ymm2)
	vunpckhpd(ymm10, ymm8, ymm3)
	vinsertf128(imm(0x1), xmm2, ymm0, ymm4)
	vinsertf128(imm(0x1), xmm3, ymm1, ymm6)
	vperm2f128(imm(0x31), ymm2, ymm0, ymm8)
	vperm2f128(imm(0x31), ymm3, ymm1, ymm10)

	vmovupd(ymm4, mem(rcx))
	vmovupd(ymm6, mem(rcx, rsi, 1))
	vmovupd(ymm8, mem(rcx, rsi, 2))
	vmovupd(ymm10, mem(rcx, rax, 1))

	lea(mem(rcx, rsi, 4), rcx)


	vunpcklpd(ymm7, ymm5, ymm0)
	vunpckhpd(ymm7, ymm5, ymm1)
	vunpcklpd(ymm11, ymm9, ymm2)
	vunpckhpd(ymm11, ymm9, ymm3)
	vinsertf128(imm(0x1), xmm2, ymm0, ymm5)
	vinsertf128(imm(0x1), xmm3, ymm1, ymm7)
	//vperm2f128(imm(0x31), ymm2, ymm0, ymm9)
	//vperm2f128(imm(0x31), ymm3, ymm1, ymm11)

	vmovupd(ymm5, mem(rcx))
	vmovupd(ymm7, mem(rcx, rsi, 1))
	//vmovupd(ymm9, mem(rcx, rsi, 2))
	//vmovupd(ymm11, mem(rcx, rax, 1))

	//lea(mem(rcx, rsi, 4), rcx)

	
	
	
	label(.DDONE)



    end_asm(
	: // output operands (none)
	: // input operands
      [k_iter] "m" (k_iter),
      [k_left] "m" (k_left),
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
	  "ymm8", "ymm9", "ymm10", "ymm11", "memory"
	)
	AOCL_DTL_TRACE_EXIT(AOCL_DTL_LEVEL_TRACE_7);
}

void bli_dgemmsup_rv_haswell_asm_3x6
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
	AOCL_DTL_TRACE_ENTRY(AOCL_DTL_LEVEL_TRACE_7);
	//void*    a_next = bli_auxinfo_next_a( data );
	//void*    b_next = bli_auxinfo_next_b( data );

	// Typecast local copies of integers in case dim_t and inc_t are a
	// different size than is expected by load instructions.
	uint64_t k_iter = k0 / 4;
	uint64_t k_left = k0 % 4;

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
	mov(var(cs_a), r9)                 // load cs_a
	lea(mem(, r8, 8), r8)              // rs_a *= sizeof(double)
	lea(mem(, r9, 8), r9)              // cs_a *= sizeof(double)

	//lea(mem(r8, r8, 2), r13)           // r13 = 3*rs_a
	//lea(mem(r8, r8, 4), r15)           // r15 = 5*rs_a

	mov(var(b), rbx)                   // load address of b.
	mov(var(rs_b), r10)                // load rs_b
	//mov(var(cs_b), r11)                // load cs_b
	lea(mem(, r10, 8), r10)            // rs_b *= sizeof(double)
	//lea(mem(, r11, 8), r11)            // cs_b *= sizeof(double)
	
	                                   // NOTE: We cannot pre-load elements of a or b
	                                   // because it could eventually, in the last
	                                   // unrolled iter or the cleanup loop, result
	                                   // in reading beyond the bounds allocated mem
	                                   // (the likely result: a segmentation fault).

	mov(var(c), rcx)                   // load address of c
	mov(var(rs_c), rdi)                // load rs_c
	lea(mem(, rdi, 8), rdi)            // rs_c *= sizeof(double)

#if 0
	lea(mem(rcx, rdi, 2), rdx)         //
	lea(mem(rdx, rdi, 1), rdx)         // rdx = c + 3*rs_c;
	prefetch(0, mem(rcx, 7*8))         // prefetch c + 0*rs_c
	prefetch(0, mem(rcx, rdi, 1, 7*8)) // prefetch c + 1*rs_c
	prefetch(0, mem(rcx, rdi, 2, 7*8)) // prefetch c + 2*rs_c
	prefetch(0, mem(rdx, 7*8))         // prefetch c + 3*rs_c
#else
	cmp(imm(8), rdi)                   // set ZF if (8*rs_c) == 8.
	jz(.DCOLPFETCH)                    // jump to column storage case
	label(.DROWPFETCH)                 // row-stored prefetching on c

	//lea(mem(rcx, rdi, 2), rdx)         //
	//lea(mem(rdx, rdi, 1), rdx)         // rdx = c + 3*rs_c;
	prefetch(0, mem(rcx, 5*8))         // prefetch c + 0*rs_c
	prefetch(0, mem(rcx, rdi, 1, 5*8)) // prefetch c + 1*rs_c
	prefetch(0, mem(rcx, rdi, 2, 5*8)) // prefetch c + 2*rs_c

	jmp(.DPOSTPFETCH)                  // jump to end of prefetching c
	label(.DCOLPFETCH)                 // column-stored prefetching c

	mov(var(cs_c), rsi)                // load cs_c to rsi (temporarily)
	lea(mem(, rsi, 8), rsi)            // cs_c *= sizeof(double)
	lea(mem(rcx, rsi, 2), rdx)         //
	lea(mem(rdx, rsi, 1), rdx)         // rdx = c + 3*cs_c;
	prefetch(0, mem(rcx, 2*8))         // prefetch c + 0*cs_c
	prefetch(0, mem(rcx, rsi, 1, 2*8)) // prefetch c + 1*cs_c
	prefetch(0, mem(rcx, rsi, 2, 2*8)) // prefetch c + 2*cs_c
	prefetch(0, mem(rdx, 2*8))         // prefetch c + 3*cs_c
	prefetch(0, mem(rdx, rsi, 1, 2*8)) // prefetch c + 4*cs_c
	prefetch(0, mem(rdx, rsi, 2, 2*8)) // prefetch c + 5*cs_c

	label(.DPOSTPFETCH)                // done prefetching c
#endif
	
	
	
	
	mov(var(k_iter), rsi)              // i = k_iter;
	test(rsi, rsi)                     // check i via logical AND.
	je(.DCONSIDKLEFT)                  // if i == 0, jump to code that
	                                   // contains the k_left loop.
	
	
	label(.DLOOPKITER)                 // MAIN LOOP
	
	
	// ---------------------------------- iteration 0
	
	vmovupd(mem(rbx, 0*32), ymm0)
	vmovupd(mem(rbx, 1*32), xmm1)
	add(r10, rbx)                      // b += rs_b;

	vbroadcastsd(mem(rax        ), ymm2)
	vbroadcastsd(mem(rax, r8,  1), ymm3)
	vfmadd231pd(ymm0, ymm2, ymm4)
	vfmadd231pd(ymm1, ymm2, ymm5)
	vfmadd231pd(ymm0, ymm3, ymm6)
	vfmadd231pd(ymm1, ymm3, ymm7)
	
	vbroadcastsd(mem(rax, r8,  2), ymm2)
	add(r9, rax)                       // a += cs_a;
	vfmadd231pd(ymm0, ymm2, ymm8)
	vfmadd231pd(ymm1, ymm2, ymm9)
	
	
	// ---------------------------------- iteration 1

	vmovupd(mem(rbx, 0*32), ymm0)
	vmovupd(mem(rbx, 1*32), xmm1)
	add(r10, rbx)                      // b += rs_b;

	vbroadcastsd(mem(rax        ), ymm2)
	vbroadcastsd(mem(rax, r8,  1), ymm3)
	vfmadd231pd(ymm0, ymm2, ymm4)
	vfmadd231pd(ymm1, ymm2, ymm5)
	vfmadd231pd(ymm0, ymm3, ymm6)
	vfmadd231pd(ymm1, ymm3, ymm7)
	
	vbroadcastsd(mem(rax, r8,  2), ymm2)
	add(r9, rax)                       // a += cs_a;
	vfmadd231pd(ymm0, ymm2, ymm8)
	vfmadd231pd(ymm1, ymm2, ymm9)
	

	// ---------------------------------- iteration 2
	
	vmovupd(mem(rbx, 0*32), ymm0)
	vmovupd(mem(rbx, 1*32), xmm1)
	add(r10, rbx)                      // b += rs_b;

	vbroadcastsd(mem(rax        ), ymm2)
	vbroadcastsd(mem(rax, r8,  1), ymm3)
	vfmadd231pd(ymm0, ymm2, ymm4)
	vfmadd231pd(ymm1, ymm2, ymm5)
	vfmadd231pd(ymm0, ymm3, ymm6)
	vfmadd231pd(ymm1, ymm3, ymm7)
	
	vbroadcastsd(mem(rax, r8,  2), ymm2)
	add(r9, rax)                       // a += cs_a;
	vfmadd231pd(ymm0, ymm2, ymm8)
	vfmadd231pd(ymm1, ymm2, ymm9)
	

	// ---------------------------------- iteration 3

	vmovupd(mem(rbx, 0*32), ymm0)
	vmovupd(mem(rbx, 1*32), xmm1)
	add(r10, rbx)                      // b += rs_b;

	vbroadcastsd(mem(rax        ), ymm2)
	vbroadcastsd(mem(rax, r8,  1), ymm3)
	vfmadd231pd(ymm0, ymm2, ymm4)
	vfmadd231pd(ymm1, ymm2, ymm5)
	vfmadd231pd(ymm0, ymm3, ymm6)
	vfmadd231pd(ymm1, ymm3, ymm7)
	
	vbroadcastsd(mem(rax, r8,  2), ymm2)
	add(r9, rax)                       // a += cs_a;
	vfmadd231pd(ymm0, ymm2, ymm8)
	vfmadd231pd(ymm1, ymm2, ymm9)
	
	
	
	dec(rsi)                           // i -= 1;
	jne(.DLOOPKITER)                   // iterate again if i != 0.
	
	
	
	
	
	
	label(.DCONSIDKLEFT)
	
	mov(var(k_left), rsi)              // i = k_left;
	test(rsi, rsi)                     // check i via logical AND.
	je(.DPOSTACCUM)                    // if i == 0, we're done; jump to end.
	                                   // else, we prepare to enter k_left loop.
	
	
	label(.DLOOPKLEFT)                 // EDGE LOOP
	
	vmovupd(mem(rbx, 0*32), ymm0)
	vmovupd(mem(rbx, 1*32), xmm1)
	add(r10, rbx)                      // b += rs_b;

	vbroadcastsd(mem(rax        ), ymm2)
	vbroadcastsd(mem(rax, r8,  1), ymm3)
	vfmadd231pd(ymm0, ymm2, ymm4)
	vfmadd231pd(ymm1, ymm2, ymm5)
	vfmadd231pd(ymm0, ymm3, ymm6)
	vfmadd231pd(ymm1, ymm3, ymm7)
	
	vbroadcastsd(mem(rax, r8,  2), ymm2)
	add(r9, rax)                       // a += cs_a;
	vfmadd231pd(ymm0, ymm2, ymm8)
	vfmadd231pd(ymm1, ymm2, ymm9)
	
	
	dec(rsi)                           // i -= 1;
	jne(.DLOOPKLEFT)                   // iterate again if i != 0.
	
	
	
	label(.DPOSTACCUM)


	
	mov(var(alpha), rax)               // load address of alpha
	mov(var(beta), rbx)                // load address of beta
	vbroadcastsd(mem(rax), ymm0)       // load alpha and duplicate
	vbroadcastsd(mem(rbx), ymm3)       // load beta and duplicate
	
	vmulpd(ymm0, ymm4, ymm4)           // scale by alpha
	vmulpd(xmm0, xmm5, xmm5)
	vmulpd(ymm0, ymm6, ymm6)
	vmulpd(xmm0, xmm7, xmm7)
	vmulpd(ymm0, ymm8, ymm8)
	vmulpd(xmm0, xmm9, xmm9)
	
	
	
	
	
	
	mov(var(cs_c), rsi)                // load cs_c
	lea(mem(, rsi, 8), rsi)            // rsi = cs_c * sizeof(double)
	
	//lea(mem(rcx, rsi, 4), rdx)         // load address of c +  4*cs_c;
	lea(mem(rcx, rdi, 2), rdx)         // load address of c +  2*rs_c;

	lea(mem(rsi, rsi, 2), rax)         // rax = 3*cs_c;
	
	
	
	                                   // now avoid loading C if beta == 0
	
	vxorpd(ymm0, ymm0, ymm0)           // set ymm0 to zero.
	vucomisd(xmm0, xmm3)               // set ZF if beta == 0.
	je(.DBETAZERO)                     // if ZF = 1, jump to beta == 0 case
	


	cmp(imm(8), rdi)                   // set ZF if (8*rs_c) == 8.
	jz(.DCOLSTORED)                    // jump to column storage case


	
	label(.DROWSTORED)
	
	
	vfmadd231pd(mem(rcx), ymm3, ymm4)
	vmovupd(ymm4, mem(rcx))

	vfmadd231pd(mem(rcx, rsi, 4), xmm3, xmm5)
	vmovupd(xmm5, mem(rcx, rsi, 4))
	add(rdi, rcx)


	vfmadd231pd(mem(rcx), ymm3, ymm6)
	vmovupd(ymm6, mem(rcx))

	vfmadd231pd(mem(rcx, rsi, 4), xmm3, xmm7)
	vmovupd(xmm7, mem(rcx, rsi, 4))
	add(rdi, rcx)


	vfmadd231pd(mem(rcx), ymm3, ymm8)
	vmovupd(ymm8, mem(rcx))

	vfmadd231pd(mem(rcx, rsi, 4), xmm3, xmm9)
	vmovupd(xmm9, mem(rcx, rsi, 4))
	//add(rdi, rcx)
	
	
	jmp(.DDONE)                        // jump to end.
	


	label(.DCOLSTORED)


	vunpcklpd(ymm6, ymm4, ymm0)
	vunpckhpd(ymm6, ymm4, ymm1)
	vunpcklpd(ymm10, ymm8, ymm2)
	vunpckhpd(ymm10, ymm8, ymm3)
	vinsertf128(imm(0x1), xmm2, ymm0, ymm4)
	vinsertf128(imm(0x1), xmm3, ymm1, ymm6)
	vperm2f128(imm(0x31), ymm2, ymm0, ymm8)
	vperm2f128(imm(0x31), ymm3, ymm1, ymm10)

	vextractf128(imm(0x1), ymm4, xmm12)
	vextractf128(imm(0x1), ymm6, xmm13)
	vextractf128(imm(0x1), ymm8, xmm14)
	vextractf128(imm(0x1), ymm10, xmm15)

	vbroadcastsd(mem(rbx), ymm3)

	vfmadd231pd(mem(rcx), xmm3, xmm4)
	vfmadd231pd(mem(rcx, rsi, 1), xmm3, xmm6)
	vfmadd231pd(mem(rcx, rsi, 2), xmm3, xmm8)
	vfmadd231pd(mem(rcx, rax, 1), xmm3, xmm10)
	vmovupd(xmm4, mem(rcx))
	vmovupd(xmm6, mem(rcx, rsi, 1))
	vmovupd(xmm8, mem(rcx, rsi, 2))
	vmovupd(xmm10, mem(rcx, rax, 1))

	lea(mem(rcx, rsi, 4), rcx)

	vfmadd231sd(mem(rdx), xmm3, xmm12)
	vfmadd231sd(mem(rdx, rsi, 1), xmm3, xmm13)
	vfmadd231sd(mem(rdx, rsi, 2), xmm3, xmm14)
	vfmadd231sd(mem(rdx, rax, 1), xmm3, xmm15)
	vmovsd(xmm12, mem(rdx))
	vmovsd(xmm13, mem(rdx, rsi, 1))
	vmovsd(xmm14, mem(rdx, rsi, 2))
	vmovsd(xmm15, mem(rdx, rax, 1))
	
	lea(mem(rdx, rsi, 4), rdx)



	vunpcklpd(ymm7, ymm5, ymm0)
	vunpckhpd(ymm7, ymm5, ymm1)
	vunpcklpd(ymm11, ymm9, ymm2)
	vunpckhpd(ymm11, ymm9, ymm3)
	vinsertf128(imm(0x1), xmm2, ymm0, ymm5)
	vinsertf128(imm(0x1), xmm3, ymm1, ymm7)
	vperm2f128(imm(0x31), ymm2, ymm0, ymm9)
	vperm2f128(imm(0x31), ymm3, ymm1, ymm11)

	vextractf128(imm(0x1), ymm5, xmm12)
	vextractf128(imm(0x1), ymm7, xmm13)
	//vextractf128(imm(0x1), ymm9, xmm14)
	//vextractf128(imm(0x1), ymm11, xmm15)

	vbroadcastsd(mem(rbx), ymm3)

	vfmadd231pd(mem(rcx), xmm3, xmm5)
	vfmadd231pd(mem(rcx, rsi, 1), xmm3, xmm7)
	vfmadd231pd(mem(rcx, rsi, 2), xmm3, xmm9)
	vfmadd231pd(mem(rcx, rax, 1), xmm3, xmm11)
	vmovupd(xmm5, mem(rcx))
	vmovupd(xmm7, mem(rcx, rsi, 1))
	//vmovupd(xmm9, mem(rcx, rsi, 2))
	//vmovupd(xmm11, mem(rcx, rax, 1))

	//lea(mem(rcx, rsi, 4), rcx)

	vfmadd231sd(mem(rdx), xmm3, xmm12)
	vfmadd231sd(mem(rdx, rsi, 1), xmm3, xmm13)
	//vfmadd231sd(mem(rdx, rsi, 2), xmm3, xmm14)
	//vfmadd231sd(mem(rdx, rax, 1), xmm3, xmm15)
	vmovsd(xmm12, mem(rdx))
	vmovsd(xmm13, mem(rdx, rsi, 1))
	//vmovsd(xmm14, mem(rdx, rsi, 2))
	//vmovsd(xmm15, mem(rdx, rax, 1))
	
	//lea(mem(rdx, rsi, 4), rdx)


	jmp(.DDONE)                        // jump to end.


	
	
	label(.DBETAZERO)
	

	cmp(imm(8), rdi)                   // set ZF if (8*rs_c) == 8.
	jz(.DCOLSTORBZ)                    // jump to column storage case


	
	label(.DROWSTORBZ)
	
	
	vmovupd(ymm4, mem(rcx))
	vmovupd(xmm5, mem(rcx, rsi, 4))
	add(rdi, rcx)

	vmovupd(ymm6, mem(rcx))
	vmovupd(xmm7, mem(rcx, rsi, 4))
	add(rdi, rcx)

	vmovupd(ymm8, mem(rcx))
	vmovupd(xmm9, mem(rcx, rsi, 4))
	//add(rdi, rcx)
	
	
	jmp(.DDONE)                        // jump to end.



	label(.DCOLSTORBZ)


	vunpcklpd(ymm6, ymm4, ymm0)
	vunpckhpd(ymm6, ymm4, ymm1)
	vunpcklpd(ymm10, ymm8, ymm2)
	vunpckhpd(ymm10, ymm8, ymm3)
	vinsertf128(imm(0x1), xmm2, ymm0, ymm4)
	vinsertf128(imm(0x1), xmm3, ymm1, ymm6)
	vperm2f128(imm(0x31), ymm2, ymm0, ymm8)
	vperm2f128(imm(0x31), ymm3, ymm1, ymm10)

	vextractf128(imm(0x1), ymm4, xmm12)
	vextractf128(imm(0x1), ymm6, xmm13)
	vextractf128(imm(0x1), ymm8, xmm14)
	vextractf128(imm(0x1), ymm10, xmm15)

	vmovupd(xmm4, mem(rcx))
	vmovupd(xmm6, mem(rcx, rsi, 1))
	vmovupd(xmm8, mem(rcx, rsi, 2))
	vmovupd(xmm10, mem(rcx, rax, 1))

	lea(mem(rcx, rsi, 4), rcx)

	vmovsd(xmm12, mem(rdx))
	vmovsd(xmm13, mem(rdx, rsi, 1))
	vmovsd(xmm14, mem(rdx, rsi, 2))
	vmovsd(xmm15, mem(rdx, rax, 1))

	lea(mem(rdx, rsi, 4), rdx)


	vunpcklpd(ymm7, ymm5, ymm0)
	vunpckhpd(ymm7, ymm5, ymm1)
	vunpcklpd(ymm11, ymm9, ymm2)
	vunpckhpd(ymm11, ymm9, ymm3)
	vinsertf128(imm(0x1), xmm2, ymm0, ymm5)
	vinsertf128(imm(0x1), xmm3, ymm1, ymm7)
	vperm2f128(imm(0x31), ymm2, ymm0, ymm9)
	vperm2f128(imm(0x31), ymm3, ymm1, ymm11)

	vextractf128(imm(0x1), ymm5, xmm12)
	vextractf128(imm(0x1), ymm7, xmm13)
	//vextractf128(imm(0x1), ymm9, xmm14)
	//vextractf128(imm(0x1), ymm11, xmm15)

	vmovupd(xmm5, mem(rcx))
	vmovupd(xmm7, mem(rcx, rsi, 1))
	//vmovupd(xmm9, mem(rcx, rsi, 2))
	//vmovupd(xmm11, mem(rcx, rax, 1))

	//lea(mem(rcx, rsi, 4), rcx)

	vmovsd(xmm12, mem(rdx))
	vmovsd(xmm13, mem(rdx, rsi, 1))
	//vmovsd(xmm14, mem(rdx, rsi, 2))
	//vmovsd(xmm15, mem(rdx, rax, 1))

	//lea(mem(rdx, rsi, 4), rdx)

	
	
	
	label(.DDONE)
	
	

    end_asm(
	: // output operands (none)
	: // input operands
      [k_iter] "m" (k_iter),
      [k_left] "m" (k_left),
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
	  "ymm8", "ymm9", "ymm10", "ymm11", "memory"
	)
	AOCL_DTL_TRACE_EXIT(AOCL_DTL_LEVEL_TRACE_7);
}

void bli_dgemmsup_rv_haswell_asm_2x6
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
	AOCL_DTL_TRACE_ENTRY(AOCL_DTL_LEVEL_TRACE_7);
	//void*    a_next = bli_auxinfo_next_a( data );
	//void*    b_next = bli_auxinfo_next_b( data );

	// Typecast local copies of integers in case dim_t and inc_t are a
	// different size than is expected by load instructions.
	uint64_t k_iter = k0 / 4;
	uint64_t k_left = k0 % 4;

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
	mov(var(cs_a), r9)                 // load cs_a
	lea(mem(, r8, 8), r8)              // rs_a *= sizeof(double)
	lea(mem(, r9, 8), r9)              // cs_a *= sizeof(double)
	
	//lea(mem(r8, r8, 2), r13)           // r13 = 3*rs_a
	//lea(mem(r8, r8, 4), r15)           // r15 = 5*rs_a

	mov(var(b), rbx)                   // load address of b.
	mov(var(rs_b), r10)                // load rs_b
	//mov(var(cs_b), r11)                // load cs_b
	lea(mem(, r10, 8), r10)            // rs_b *= sizeof(double)
	//lea(mem(, r11, 8), r11)            // cs_b *= sizeof(double)

	                                   // NOTE: We cannot pre-load elements of a or b
	                                   // because it could eventually, in the last
	                                   // unrolled iter or the cleanup loop, result
	                                   // in reading beyond the bounds allocated mem
	                                   // (the likely result: a segmentation fault).

	mov(var(c), rcx)                   // load address of c
	mov(var(rs_c), rdi)                // load rs_c
	lea(mem(, rdi, 8), rdi)            // rs_c *= sizeof(double)



	cmp(imm(8), rdi)                   // set ZF if (8*rs_c) == 8.
	jz(.DCOLPFETCH)                    // jump to column storage case
	label(.DROWPFETCH)                 // row-stored prefetching on c

	//lea(mem(rcx, rdi, 2), rdx)         //
	//lea(mem(rdx, rdi, 1), rdx)         // rdx = c + 3*rs_c;
	prefetch(0, mem(rcx, 5*8))         // prefetch c + 0*rs_c
	prefetch(0, mem(rcx, rdi, 1, 5*8)) // prefetch c + 1*rs_c

	jmp(.DPOSTPFETCH)                  // jump to end of prefetching c
	label(.DCOLPFETCH)                 // column-stored prefetching c

	mov(var(cs_c), rsi)                // load cs_c to rsi (temporarily)
	lea(mem(, rsi, 8), rsi)            // cs_c *= sizeof(double)
	lea(mem(rcx, rsi, 2), rdx)         //
	lea(mem(rdx, rsi, 1), rdx)         // rdx = c + 3*cs_c;
	prefetch(0, mem(rcx, 1*8))         // prefetch c + 0*cs_c
	prefetch(0, mem(rcx, rsi, 1, 1*8)) // prefetch c + 1*cs_c
	prefetch(0, mem(rcx, rsi, 2, 1*8)) // prefetch c + 2*cs_c
	prefetch(0, mem(rdx, 1*8))         // prefetch c + 3*cs_c
	prefetch(0, mem(rdx, rsi, 1, 1*8)) // prefetch c + 4*cs_c
	prefetch(0, mem(rdx, rsi, 2, 1*8)) // prefetch c + 5*cs_c

	label(.DPOSTPFETCH)                // done prefetching c


	
	
	mov(var(k_iter), rsi)              // i = k_iter;
	test(rsi, rsi)                     // check i via logical AND.
	je(.DCONSIDKLEFT)                  // if i == 0, jump to code that
	                                   // contains the k_left loop.
	
	
	label(.DLOOPKITER)                 // MAIN LOOP
	
	
	// ---------------------------------- iteration 0

	vmovupd(mem(rbx,  0*32), ymm0)
	vmovupd(mem(rbx,  1*32), xmm1)
	add(r10, rbx)                      // b += rs_b;

	vbroadcastsd(mem(rax        ), ymm2)
	vbroadcastsd(mem(rax, r8,  1), ymm3)
	add(r9, rax)                       // a += cs_a;
	vfmadd231pd(ymm0, ymm2, ymm4)
	vfmadd231pd(ymm1, ymm2, ymm5)
	vfmadd231pd(ymm0, ymm3, ymm6)
	vfmadd231pd(ymm1, ymm3, ymm7)

	
	// ---------------------------------- iteration 1

	vmovupd(mem(rbx,  0*32), ymm0)
	vmovupd(mem(rbx,  1*32), xmm1)
	add(r10, rbx)                      // b += rs_b;

	vbroadcastsd(mem(rax        ), ymm2)
	vbroadcastsd(mem(rax, r8,  1), ymm3)
	add(r9, rax)                       // a += cs_a;
	vfmadd231pd(ymm0, ymm2, ymm4)
	vfmadd231pd(ymm1, ymm2, ymm5)
	vfmadd231pd(ymm0, ymm3, ymm6)
	vfmadd231pd(ymm1, ymm3, ymm7)
	
	
	// ---------------------------------- iteration 2

	vmovupd(mem(rbx,  0*32), ymm0)
	vmovupd(mem(rbx,  1*32), xmm1)
	add(r10, rbx)                      // b += rs_b;

	vbroadcastsd(mem(rax        ), ymm2)
	vbroadcastsd(mem(rax, r8,  1), ymm3)
	add(r9, rax)                       // a += cs_a;
	vfmadd231pd(ymm0, ymm2, ymm4)
	vfmadd231pd(ymm1, ymm2, ymm5)
	vfmadd231pd(ymm0, ymm3, ymm6)
	vfmadd231pd(ymm1, ymm3, ymm7)
	

	// ---------------------------------- iteration 3

	vmovupd(mem(rbx, 0*32), ymm0)
	vmovupd(mem(rbx, 1*32), xmm1)
	add(r10, rbx)                      // b += rs_b;

	vbroadcastsd(mem(rax        ), ymm2)
	vbroadcastsd(mem(rax, r8,  1), ymm3)
	add(r9, rax)                       // a += cs_a;
	vfmadd231pd(ymm0, ymm2, ymm4)
	vfmadd231pd(ymm1, ymm2, ymm5)
	vfmadd231pd(ymm0, ymm3, ymm6)
	vfmadd231pd(ymm1, ymm3, ymm7)
	
	
	
	dec(rsi)                           // i -= 1;
	jne(.DLOOPKITER)                   // iterate again if i != 0.
	
	
	
	
	
	
	label(.DCONSIDKLEFT)
	
	mov(var(k_left), rsi)              // i = k_left;
	test(rsi, rsi)                     // check i via logical AND.
	je(.DPOSTACCUM)                    // if i == 0, we're done; jump to end.
	                                   // else, we prepare to enter k_left loop.
	
	
	label(.DLOOPKLEFT)                 // EDGE LOOP
	
	vmovupd(mem(rbx,  0*32), ymm0)
	vmovupd(mem(rbx,  1*32), xmm1)
	add(r10, rbx)                      // b += rs_b;
	
	vbroadcastsd(mem(rax        ), ymm2)
	vbroadcastsd(mem(rax, r8,  1), ymm3)
	add(r9, rax)                       // a += cs_a;
	vfmadd231pd(ymm0, ymm2, ymm4)
	vfmadd231pd(ymm1, ymm2, ymm5)
	vfmadd231pd(ymm0, ymm3, ymm6)
	vfmadd231pd(ymm1, ymm3, ymm7)
	
	
	dec(rsi)                           // i -= 1;
	jne(.DLOOPKLEFT)                   // iterate again if i != 0.
	
	
	
	label(.DPOSTACCUM)

	
	
	mov(var(alpha), rax)               // load address of alpha
	mov(var(beta), rbx)                // load address of beta
	vbroadcastsd(mem(rax), ymm0)       // load alpha and duplicate
	vbroadcastsd(mem(rbx), ymm3)       // load beta and duplicate
	
	vmulpd(ymm0, ymm4, ymm4)           // scale by alpha
	vmulpd(xmm0, xmm5, xmm5)
	vmulpd(ymm0, ymm6, ymm6)
	vmulpd(xmm0, xmm7, xmm7)
	
	
	
	
	
	
	mov(var(cs_c), rsi)                // load cs_c
	lea(mem(, rsi, 8), rsi)            // rsi = cs_c * sizeof(double)
	
	//lea(mem(rcx, rsi, 4), rdx)         // load address of c +  4*cs_c;
	//lea(mem(rcx, rdi, 4), rdx)         // load address of c +  4*rs_c;

	lea(mem(rsi, rsi, 2), rax)         // rax = 3*cs_c;
	
	
	
	                                   // now avoid loading C if beta == 0
	
	vxorpd(ymm0, ymm0, ymm0)           // set ymm0 to zero.
	vucomisd(xmm0, xmm3)               // set ZF if beta == 0.
	je(.DBETAZERO)                     // if ZF = 1, jump to beta == 0 case



	cmp(imm(8), rdi)                   // set ZF if (8*rs_c) == 8.
	jz(.DCOLSTORED)                    // jump to column storage case
	

	
	label(.DROWSTORED)
	
	
	vfmadd231pd(mem(rcx), ymm3, ymm4)
	vmovupd(ymm4, mem(rcx))

	vfmadd231pd(mem(rcx, rsi, 4), xmm3, xmm5)
	vmovupd(xmm5, mem(rcx, rsi, 4))
	add(rdi, rcx)
	
	
	vfmadd231pd(mem(rcx), ymm3, ymm6)
	vmovupd(ymm6, mem(rcx))

	vfmadd231pd(mem(rcx, rsi, 4), xmm3, xmm7)
	vmovupd(xmm7, mem(rcx, rsi, 4))
	//add(rdi, rcx)
	
	
	jmp(.DDONE)                        // jump to end.



	label(.DCOLSTORED)


	vunpcklpd(ymm6, ymm4, ymm0)
	vunpckhpd(ymm6, ymm4, ymm1)
	vunpcklpd(ymm10, ymm8, ymm2)
	vunpckhpd(ymm10, ymm8, ymm3)
	vinsertf128(imm(0x1), xmm2, ymm0, ymm4)
	vinsertf128(imm(0x1), xmm3, ymm1, ymm6)
	vperm2f128(imm(0x31), ymm2, ymm0, ymm8)
	vperm2f128(imm(0x31), ymm3, ymm1, ymm10)

	vbroadcastsd(mem(rbx), ymm3)

	vfmadd231pd(mem(rcx), xmm3, xmm4)
	vfmadd231pd(mem(rcx, rsi, 1), xmm3, xmm6)
	vfmadd231pd(mem(rcx, rsi, 2), xmm3, xmm8)
	vfmadd231pd(mem(rcx, rax, 1), xmm3, xmm10)
	vmovupd(xmm4, mem(rcx))
	vmovupd(xmm6, mem(rcx, rsi, 1))
	vmovupd(xmm8, mem(rcx, rsi, 2))
	vmovupd(xmm10, mem(rcx, rax, 1))

	lea(mem(rcx, rsi, 4), rcx)


	vunpcklpd(ymm7, ymm5, ymm0)
	vunpckhpd(ymm7, ymm5, ymm1)
	vunpcklpd(ymm11, ymm9, ymm2)
	vunpckhpd(ymm11, ymm9, ymm3)
	vinsertf128(imm(0x1), xmm2, ymm0, ymm5)
	vinsertf128(imm(0x1), xmm3, ymm1, ymm7)
	//vperm2f128(imm(0x31), ymm2, ymm0, ymm9)
	//vperm2f128(imm(0x31), ymm3, ymm1, ymm11)

	vbroadcastsd(mem(rbx), ymm3)

	vfmadd231pd(mem(rcx), xmm3, xmm5)
	vfmadd231pd(mem(rcx, rsi, 1), xmm3, xmm7)
	//vfmadd231pd(mem(rcx, rsi, 2), ymm3, ymm9)
	//vfmadd231pd(mem(rcx, rax, 1), ymm3, ymm11)
	vmovupd(xmm5, mem(rcx))
	vmovupd(xmm7, mem(rcx, rsi, 1))
	//vmovupd(ymm9, mem(rcx, rsi, 2))
	//vmovupd(ymm11, mem(rcx, rax, 1))

	//lea(mem(rcx, rsi, 4), rcx)


	jmp(.DDONE)                        // jump to end.
	
	
	
	
	label(.DBETAZERO)
	

	cmp(imm(8), rdi)                   // set ZF if (8*rs_c) == 8.
	jz(.DCOLSTORBZ)                    // jump to column storage case


	
	label(.DROWSTORBZ)
	
	
	vmovupd(ymm4, mem(rcx))
	vmovupd(xmm5, mem(rcx, rsi, 4))
	add(rdi, rcx)
	

	vmovupd(ymm6, mem(rcx))
	vmovupd(xmm7, mem(rcx, rsi, 4))
	//add(rdi, rcx)
	
	
	jmp(.DDONE)                        // jump to end.



	label(.DCOLSTORBZ)


	vunpcklpd(ymm6, ymm4, ymm0)
	vunpckhpd(ymm6, ymm4, ymm1)
	vunpcklpd(ymm10, ymm8, ymm2)
	vunpckhpd(ymm10, ymm8, ymm3)
	vinsertf128(imm(0x1), xmm2, ymm0, ymm4)
	vinsertf128(imm(0x1), xmm3, ymm1, ymm6)
	vperm2f128(imm(0x31), ymm2, ymm0, ymm8)
	vperm2f128(imm(0x31), ymm3, ymm1, ymm10)

	vmovupd(xmm4, mem(rcx))
	vmovupd(xmm6, mem(rcx, rsi, 1))
	vmovupd(xmm8, mem(rcx, rsi, 2))
	vmovupd(xmm10, mem(rcx, rax, 1))

	lea(mem(rcx, rsi, 4), rcx)


	vunpcklpd(ymm7, ymm5, ymm0)
	vunpckhpd(ymm7, ymm5, ymm1)
	vunpcklpd(ymm11, ymm9, ymm2)
	vunpckhpd(ymm11, ymm9, ymm3)
	vinsertf128(imm(0x1), xmm2, ymm0, ymm5)
	vinsertf128(imm(0x1), xmm3, ymm1, ymm7)
	//vperm2f128(imm(0x31), ymm2, ymm0, ymm9)
	//vperm2f128(imm(0x31), ymm3, ymm1, ymm11)

	vmovupd(xmm5, mem(rcx))
	vmovupd(xmm7, mem(rcx, rsi, 1))
	//vmovupd(ymm9, mem(rcx, rsi, 2))
	//vmovupd(ymm11, mem(rcx, rax, 1))

	//lea(mem(rcx, rsi, 4), rcx)

	
	
	
	label(.DDONE)



    end_asm(
	: // output operands (none)
	: // input operands
      [k_iter] "m" (k_iter),
      [k_left] "m" (k_left),
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
	  "ymm8", "ymm9", "ymm10", "ymm11", "memory"
	)
	AOCL_DTL_TRACE_EXIT(AOCL_DTL_LEVEL_TRACE_7);
}

void bli_dgemmsup_rv_haswell_asm_1x6
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
	AOCL_DTL_TRACE_ENTRY(AOCL_DTL_LEVEL_TRACE_7);
	//void*    a_next = bli_auxinfo_next_a( data );
	//void*    b_next = bli_auxinfo_next_b( data );

	// Typecast local copies of integers in case dim_t and inc_t are a
	// different size than is expected by load instructions.
	uint64_t k_iter = k0 / 4;
	uint64_t k_left = k0 % 4;

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
	mov(var(cs_a), r9)                 // load cs_a
	lea(mem(, r8, 8), r8)              // rs_a *= sizeof(double)
	lea(mem(, r9, 8), r9)              // cs_a *= sizeof(double)
	
	//lea(mem(r8, r8, 2), r13)           // r13 = 3*rs_a
	//lea(mem(r8, r8, 4), r15)           // r15 = 5*rs_a

	mov(var(b), rbx)                   // load address of b.
	mov(var(rs_b), r10)                // load rs_b
	//mov(var(cs_b), r11)                // load cs_b
	lea(mem(, r10, 8), r10)            // rs_b *= sizeof(double)
	//lea(mem(, r11, 8), r11)            // cs_b *= sizeof(double)

	                                   // NOTE: We cannot pre-load elements of a or b
	                                   // because it could eventually, in the last
	                                   // unrolled iter or the cleanup loop, result
	                                   // in reading beyond the bounds allocated mem
	                                   // (the likely result: a segmentation fault).

	mov(var(c), rcx)                   // load address of c
	mov(var(rs_c), rdi)                // load rs_c
	lea(mem(, rdi, 8), rdi)            // rs_c *= sizeof(double)



#if 1
	prefetch(0, mem(rcx, 5*8))         // prefetch c + 0*rs_c
#else
	cmp(imm(8), rdi)                   // set ZF if (8*rs_c) == 8.
	jz(.DCOLPFETCH)                    // jump to column storage case
	label(.DROWPFETCH)                 // row-stored prefetching on c

	//lea(mem(rcx, rdi, 2), rdx)         //
	//lea(mem(rdx, rdi, 1), rdx)         // rdx = c + 3*rs_c;
	prefetch(0, mem(rcx, 5*8))         // prefetch c + 0*rs_c

	jmp(.DPOSTPFETCH)                  // jump to end of prefetching c
	label(.DCOLPFETCH)                 // column-stored prefetching c

	mov(var(cs_c), rsi)                // load cs_c to rsi (temporarily)
	lea(mem(, rsi, 8), rsi)            // cs_c *= sizeof(double)
	lea(mem(rcx, rsi, 2), rdx)         //
	lea(mem(rdx, rsi, 1), rdx)         // rdx = c + 3*cs_c;
	prefetch(0, mem(rcx, 0*8))         // prefetch c + 0*cs_c
	prefetch(0, mem(rcx, rsi, 1, 0*8)) // prefetch c + 1*cs_c
	prefetch(0, mem(rcx, rsi, 2, 0*8)) // prefetch c + 2*cs_c
	prefetch(0, mem(rdx, 0*8))         // prefetch c + 3*cs_c
	prefetch(0, mem(rdx, rsi, 1, 0*8)) // prefetch c + 4*cs_c
	prefetch(0, mem(rdx, rsi, 2, 0*8)) // prefetch c + 5*cs_c

	label(.DPOSTPFETCH)                // done prefetching c
#endif


	
	
	mov(var(k_iter), rsi)              // i = k_iter;
	test(rsi, rsi)                     // check i via logical AND.
	je(.DCONSIDKLEFT)                  // if i == 0, jump to code that
	                                   // contains the k_left loop.
	
	
	label(.DLOOPKITER)                 // MAIN LOOP
	
	
	// ---------------------------------- iteration 0

	vmovupd(mem(rbx,  0*32), ymm0)
	vmovupd(mem(rbx,  1*32), xmm1)
	add(r10, rbx)                      // b += rs_b;

	vbroadcastsd(mem(rax        ), ymm2)
	vbroadcastsd(mem(rax, r8,  1), ymm3)
	add(r9, rax)                       // a += cs_a;
	vfmadd231pd(ymm0, ymm2, ymm4)
	vfmadd231pd(ymm1, ymm2, ymm5)

	
	// ---------------------------------- iteration 1

	vmovupd(mem(rbx,  0*32), ymm0)
	vmovupd(mem(rbx,  1*32), xmm1)
	add(r10, rbx)                      // b += rs_b;

	vbroadcastsd(mem(rax        ), ymm2)
	vbroadcastsd(mem(rax, r8,  1), ymm3)
	add(r9, rax)                       // a += cs_a;
	vfmadd231pd(ymm0, ymm2, ymm4)
	vfmadd231pd(ymm1, ymm2, ymm5)
	
	
	// ---------------------------------- iteration 2

	vmovupd(mem(rbx,  0*32), ymm0)
	vmovupd(mem(rbx,  1*32), xmm1)
	add(r10, rbx)                      // b += rs_b;

	vbroadcastsd(mem(rax        ), ymm2)
	vbroadcastsd(mem(rax, r8,  1), ymm3)
	add(r9, rax)                       // a += cs_a;
	vfmadd231pd(ymm0, ymm2, ymm4)
	vfmadd231pd(ymm1, ymm2, ymm5)
	

	// ---------------------------------- iteration 3

	vmovupd(mem(rbx, 0*32), ymm0)
	vmovupd(mem(rbx, 1*32), xmm1)
	add(r10, rbx)                      // b += rs_b;

	vbroadcastsd(mem(rax        ), ymm2)
	vbroadcastsd(mem(rax, r8,  1), ymm3)
	add(r9, rax)                       // a += cs_a;
	vfmadd231pd(ymm0, ymm2, ymm4)
	vfmadd231pd(ymm1, ymm2, ymm5)
	
	
	
	dec(rsi)                           // i -= 1;
	jne(.DLOOPKITER)                   // iterate again if i != 0.
	
	
	
	
	
	
	label(.DCONSIDKLEFT)
	
	mov(var(k_left), rsi)              // i = k_left;
	test(rsi, rsi)                     // check i via logical AND.
	je(.DPOSTACCUM)                    // if i == 0, we're done; jump to end.
	                                   // else, we prepare to enter k_left loop.
	
	
	label(.DLOOPKLEFT)                 // EDGE LOOP
	
	vmovupd(mem(rbx,  0*32), ymm0)
	vmovupd(mem(rbx,  1*32), xmm1)
	add(r10, rbx)                      // b += rs_b;
	
	vbroadcastsd(mem(rax        ), ymm2)
	vbroadcastsd(mem(rax, r8,  1), ymm3)
	add(r9, rax)                       // a += cs_a;
	vfmadd231pd(ymm0, ymm2, ymm4)
	vfmadd231pd(ymm1, ymm2, ymm5)
	
	
	dec(rsi)                           // i -= 1;
	jne(.DLOOPKLEFT)                   // iterate again if i != 0.
	
	
	
	label(.DPOSTACCUM)

	
	
	mov(var(alpha), rax)               // load address of alpha
	mov(var(beta), rbx)                // load address of beta
	vbroadcastsd(mem(rax), ymm0)       // load alpha and duplicate
	vbroadcastsd(mem(rbx), ymm3)       // load beta and duplicate
	
	vmulpd(ymm0, ymm4, ymm4)           // scale by alpha
	vmulpd(xmm0, xmm5, xmm5)
	
	
	
	
	
	
	mov(var(cs_c), rsi)                // load cs_c
	lea(mem(, rsi, 8), rsi)            // rsi = cs_c * sizeof(double)
	
	//lea(mem(rcx, rsi, 4), rdx)         // load address of c +  4*cs_c;
	//lea(mem(rcx, rdi, 4), rdx)         // load address of c +  4*rs_c;

	lea(mem(rsi, rsi, 2), rax)         // rax = 3*cs_c;
	
	
	
	                                   // now avoid loading C if beta == 0
	
	vxorpd(ymm0, ymm0, ymm0)           // set ymm0 to zero.
	vucomisd(xmm0, xmm3)               // set ZF if beta == 0.
	je(.DBETAZERO)                     // if ZF = 1, jump to beta == 0 case



	cmp(imm(8), rdi)                   // set ZF if (8*rs_c) == 8.
	jz(.DCOLSTORED)                    // jump to column storage case
	

	
	label(.DROWSTORED)
	
	
	vfmadd231pd(mem(rcx), ymm3, ymm4)
	vmovupd(ymm4, mem(rcx))

	vfmadd231pd(mem(rcx, rsi, 4), xmm3, xmm5)
	vmovupd(xmm5, mem(rcx, rsi, 4))
	//add(rdi, rcx)
	
	
	jmp(.DDONE)                        // jump to end.



	label(.DCOLSTORED)


	vmovlpd(mem(rcx), xmm0, xmm0)
	vmovhpd(mem(rcx, rsi, 1), xmm0, xmm0)
	vmovlpd(mem(rcx, rsi, 2), xmm1, xmm1)
	vmovhpd(mem(rcx, rax, 1), xmm1, xmm1)
	vperm2f128(imm(0x20), ymm1, ymm0, ymm0)

	vfmadd213pd(ymm4, ymm3, ymm0)

	vextractf128(imm(1), ymm0, xmm1)
	vmovlpd(xmm0, mem(rcx))
	vmovhpd(xmm0, mem(rcx, rsi, 1))
	vmovlpd(xmm1, mem(rcx, rsi, 2))
	vmovhpd(xmm1, mem(rcx, rax, 1))

	lea(mem(rcx, rsi, 4), rcx)


	vmovlpd(mem(rcx), xmm0, xmm0)
	vmovhpd(mem(rcx, rsi, 1), xmm0, xmm0)
	//vmovlpd(mem(rcx, rsi, 2), xmm1, xmm1)
	//vmovhpd(mem(rcx, rax, 1), xmm1, xmm1)
	//vperm2f128(imm(0x20), ymm1, ymm0, ymm0)

	vfmadd213pd(xmm5, xmm3, xmm0)

	//vextractf128(imm(1), ymm0, xmm1)
	vmovlpd(xmm0, mem(rcx))
	vmovhpd(xmm0, mem(rcx, rsi, 1))
	//vmovlpd(xmm1, mem(rcx, rsi, 2))
	//vmovhpd(xmm1, mem(rcx, rax, 1))

	//lea(mem(rcx, rsi, 4), rcx)


	jmp(.DDONE)                        // jump to end.
	
	
	
	
	label(.DBETAZERO)
	

	cmp(imm(8), rdi)                   // set ZF if (8*rs_c) == 8.
	jz(.DCOLSTORBZ)                    // jump to column storage case


	
	label(.DROWSTORBZ)
	
	
	vmovupd(ymm4, mem(rcx))
	vmovupd(xmm5, mem(rcx, rsi, 4))
	//add(rdi, rcx)
	
	
	jmp(.DDONE)                        // jump to end.



	label(.DCOLSTORBZ)


	vmovupd(ymm4, ymm0)

	vextractf128(imm(1), ymm0, xmm1)
	vmovlpd(xmm0, mem(rcx))
	vmovhpd(xmm0, mem(rcx, rsi, 1))
	vmovlpd(xmm1, mem(rcx, rsi, 2))
	vmovhpd(xmm1, mem(rcx, rax, 1))

	lea(mem(rcx, rsi, 4), rcx)


	vmovupd(xmm5, xmm0)

	//vextractf128(imm(1), ymm0, xmm1)
	vmovlpd(xmm0, mem(rcx))
	vmovhpd(xmm0, mem(rcx, rsi, 1))
	//vmovlpd(xmm1, mem(rcx, rsi, 2))
	//vmovhpd(xmm1, mem(rcx, rax, 1))

	//lea(mem(rcx, rsi, 4), rcx)

	
	
	
	label(.DDONE)



    end_asm(
	: // output operands (none)
	: // input operands
      [k_iter] "m" (k_iter),
      [k_left] "m" (k_left),
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
	  "ymm0", "ymm1", "ymm2", "ymm3", "ymm4", "ymm5", "memory"
	)
	AOCL_DTL_TRACE_EXIT(AOCL_DTL_LEVEL_TRACE_7);
}

void bli_dgemmsup_rv_haswell_asm_6x4
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
	AOCL_DTL_TRACE_ENTRY(AOCL_DTL_LEVEL_TRACE_7);
	//void*    a_next = bli_auxinfo_next_a( data );
	//void*    b_next = bli_auxinfo_next_b( data );

	// Typecast local copies of integers in case dim_t and inc_t are a
	// different size than is expected by load instructions.
	uint64_t k_iter = k0 / 4;
	uint64_t k_left = k0 % 4;

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
	mov(var(cs_a), r9)                 // load cs_a
	lea(mem(, r8, 8), r8)              // rs_a *= sizeof(double)
	lea(mem(, r9, 8), r9)              // cs_a *= sizeof(double)

	lea(mem(r8, r8, 2), r13)           // r13 = 3*rs_a
	lea(mem(r8, r8, 4), r15)           // r15 = 5*rs_a

	mov(var(b), rbx)                   // load address of b.
	mov(var(rs_b), r10)                // load rs_b
	//mov(var(cs_b), r11)                // load cs_b
	lea(mem(, r10, 8), r10)            // rs_b *= sizeof(double)
	//lea(mem(, r11, 8), r11)            // cs_b *= sizeof(double)
	
	                                   // NOTE: We cannot pre-load elements of a or b
	                                   // because it could eventually, in the last
	                                   // unrolled iter or the cleanup loop, result
	                                   // in reading beyond the bounds allocated mem
	                                   // (the likely result: a segmentation fault).

	mov(var(c), rcx)                   // load address of c
	mov(var(rs_c), rdi)                // load rs_c
	lea(mem(, rdi, 8), rdi)            // rs_c *= sizeof(double)

#if 0
	lea(mem(rcx, rdi, 2), rdx)         //
	lea(mem(rdx, rdi, 1), rdx)         // rdx = c + 3*rs_c;
	prefetch(0, mem(rcx, 7*8))         // prefetch c + 0*rs_c
	prefetch(0, mem(rcx, rdi, 1, 7*8)) // prefetch c + 1*rs_c
	prefetch(0, mem(rcx, rdi, 2, 7*8)) // prefetch c + 2*rs_c
	prefetch(0, mem(rdx, 7*8))         // prefetch c + 3*rs_c
	prefetch(0, mem(rdx, rdi, 1, 7*8)) // prefetch c + 4*rs_c
	prefetch(0, mem(rdx, rdi, 2, 7*8)) // prefetch c + 5*rs_c
#else
	cmp(imm(8), rdi)                   // set ZF if (8*rs_c) == 8.
	jz(.DCOLPFETCH)                    // jump to column storage case
	label(.DROWPFETCH)                 // row-stored prefetching on c

	lea(mem(rcx, rdi, 2), rdx)         //
	lea(mem(rdx, rdi, 1), rdx)         // rdx = c + 3*rs_c;
	prefetch(0, mem(rcx, 3*8))         // prefetch c + 0*rs_c
	prefetch(0, mem(rcx, rdi, 1, 3*8)) // prefetch c + 1*rs_c
	prefetch(0, mem(rcx, rdi, 2, 3*8)) // prefetch c + 2*rs_c
	prefetch(0, mem(rdx, 3*8))         // prefetch c + 3*rs_c
	prefetch(0, mem(rdx, rdi, 1, 3*8)) // prefetch c + 4*rs_c
	prefetch(0, mem(rdx, rdi, 2, 3*8)) // prefetch c + 5*rs_c

	jmp(.DPOSTPFETCH)                  // jump to end of prefetching c
	label(.DCOLPFETCH)                 // column-stored prefetching c

	mov(var(cs_c), rsi)                // load cs_c to rsi (temporarily)
	lea(mem(, rsi, 8), rsi)            // cs_c *= sizeof(double)
	lea(mem(rcx, rsi, 2), rdx)         //
	lea(mem(rdx, rsi, 1), rdx)         // rdx = c + 3*cs_c;
	prefetch(0, mem(rcx, 5*8))         // prefetch c + 0*cs_c
	prefetch(0, mem(rcx, rsi, 1, 5*8)) // prefetch c + 1*cs_c
	prefetch(0, mem(rcx, rsi, 2, 5*8)) // prefetch c + 2*cs_c
	prefetch(0, mem(rdx, 5*8))         // prefetch c + 3*cs_c

	label(.DPOSTPFETCH)                // done prefetching c
#endif
	
	
	
	
	mov(var(k_iter), rsi)              // i = k_iter;
	test(rsi, rsi)                     // check i via logical AND.
	je(.DCONSIDKLEFT)                  // if i == 0, jump to code that
	                                   // contains the k_left loop.
	
	
	label(.DLOOPKITER)                 // MAIN LOOP
	
	
	// ---------------------------------- iteration 0
	
	vmovupd(mem(rbx,  0*32), ymm0)
	add(r10, rbx)                      // b += rs_b;

	vbroadcastsd(mem(rax        ), ymm2)
	vbroadcastsd(mem(rax, r8,  1), ymm3)
	vfmadd231pd(ymm0, ymm2, ymm4)
	vfmadd231pd(ymm0, ymm3, ymm6)
	
	vbroadcastsd(mem(rax, r8,  2), ymm2)
	vbroadcastsd(mem(rax, r13, 1), ymm3)
	vfmadd231pd(ymm0, ymm2, ymm8)
	vfmadd231pd(ymm0, ymm3, ymm10)
	
	vbroadcastsd(mem(rax, r8,  4), ymm2)
	vbroadcastsd(mem(rax, r15, 1), ymm3)
	add(r9, rax)                       // a += cs_a;
	vfmadd231pd(ymm0, ymm2, ymm12)
	vfmadd231pd(ymm0, ymm3, ymm14)

	
	// ---------------------------------- iteration 1

	vmovupd(mem(rbx,  0*32), ymm0)
	add(r10, rbx)                      // b += rs_b;

	vbroadcastsd(mem(rax        ), ymm2)
	vbroadcastsd(mem(rax, r8,  1), ymm3)
	vfmadd231pd(ymm0, ymm2, ymm4)
	vfmadd231pd(ymm0, ymm3, ymm6)
	
	vbroadcastsd(mem(rax, r8,  2), ymm2)
	vbroadcastsd(mem(rax, r13, 1), ymm3)
	vfmadd231pd(ymm0, ymm2, ymm8)
	vfmadd231pd(ymm0, ymm3, ymm10)
	
	vbroadcastsd(mem(rax, r8,  4), ymm2)
	vbroadcastsd(mem(rax, r15, 1), ymm3)
	add(r9, rax)                       // a += cs_a;
	vfmadd231pd(ymm0, ymm2, ymm12)
	vfmadd231pd(ymm0, ymm3, ymm14)
	

	// ---------------------------------- iteration 2
	
	vmovupd(mem(rbx,  0*32), ymm0)
	add(r10, rbx)                      // b += rs_b;

	vbroadcastsd(mem(rax        ), ymm2)
	vbroadcastsd(mem(rax, r8,  1), ymm3)
	vfmadd231pd(ymm0, ymm2, ymm4)
	vfmadd231pd(ymm0, ymm3, ymm6)
	
	vbroadcastsd(mem(rax, r8,  2), ymm2)
	vbroadcastsd(mem(rax, r13, 1), ymm3)
	vfmadd231pd(ymm0, ymm2, ymm8)
	vfmadd231pd(ymm0, ymm3, ymm10)
	
	vbroadcastsd(mem(rax, r8,  4), ymm2)
	vbroadcastsd(mem(rax, r15, 1), ymm3)
	add(r9, rax)                       // a += cs_a;
	vfmadd231pd(ymm0, ymm2, ymm12)
	vfmadd231pd(ymm0, ymm3, ymm14)
	

	// ---------------------------------- iteration 3

	vmovupd(mem(rbx,  0*32), ymm0)
	add(r10, rbx)                      // b += rs_b;

	vbroadcastsd(mem(rax        ), ymm2)
	vbroadcastsd(mem(rax, r8,  1), ymm3)
	vfmadd231pd(ymm0, ymm2, ymm4)
	vfmadd231pd(ymm0, ymm3, ymm6)
	
	vbroadcastsd(mem(rax, r8,  2), ymm2)
	vbroadcastsd(mem(rax, r13, 1), ymm3)
	vfmadd231pd(ymm0, ymm2, ymm8)
	vfmadd231pd(ymm0, ymm3, ymm10)
	
	vbroadcastsd(mem(rax, r8,  4), ymm2)
	vbroadcastsd(mem(rax, r15, 1), ymm3)
	add(r9, rax)                       // a += cs_a;
	vfmadd231pd(ymm0, ymm2, ymm12)
	vfmadd231pd(ymm0, ymm3, ymm14)
	
	
	
	dec(rsi)                           // i -= 1;
	jne(.DLOOPKITER)                   // iterate again if i != 0.
	
	
	
	
	
	
	label(.DCONSIDKLEFT)
	
	mov(var(k_left), rsi)              // i = k_left;
	test(rsi, rsi)                     // check i via logical AND.
	je(.DPOSTACCUM)                    // if i == 0, we're done; jump to end.
	                                   // else, we prepare to enter k_left loop.
	
	
	label(.DLOOPKLEFT)                 // EDGE LOOP
	
	vmovupd(mem(rbx,  0*32), ymm0)
	add(r10, rbx)                      // b += rs_b;

	vbroadcastsd(mem(rax        ), ymm2)
	vbroadcastsd(mem(rax, r8,  1), ymm3)
	vfmadd231pd(ymm0, ymm2, ymm4)
	vfmadd231pd(ymm0, ymm3, ymm6)
	
	vbroadcastsd(mem(rax, r8,  2), ymm2)
	vbroadcastsd(mem(rax, r13, 1), ymm3)
	vfmadd231pd(ymm0, ymm2, ymm8)
	vfmadd231pd(ymm0, ymm3, ymm10)
	
	vbroadcastsd(mem(rax, r8,  4), ymm2)
	vbroadcastsd(mem(rax, r15, 1), ymm3)
	add(r9, rax)                       // a += cs_a;
	vfmadd231pd(ymm0, ymm2, ymm12)
	vfmadd231pd(ymm0, ymm3, ymm14)
	
	
	dec(rsi)                           // i -= 1;
	jne(.DLOOPKLEFT)                   // iterate again if i != 0.
	
	
	
	label(.DPOSTACCUM)


	
	mov(var(alpha), rax)               // load address of alpha
	mov(var(beta), rbx)                // load address of beta
	vbroadcastsd(mem(rax), ymm0)       // load alpha and duplicate
	vbroadcastsd(mem(rbx), ymm3)       // load beta and duplicate
	
	vmulpd(ymm0, ymm4, ymm4)           // scale by alpha
	vmulpd(ymm0, ymm6, ymm6)
	vmulpd(ymm0, ymm8, ymm8)
	vmulpd(ymm0, ymm10, ymm10)
	vmulpd(ymm0, ymm12, ymm12)
	vmulpd(ymm0, ymm14, ymm14)
	
	
	
	
	
	
	mov(var(cs_c), rsi)                // load cs_c
	lea(mem(, rsi, 8), rsi)            // rsi = cs_c * sizeof(double)
	
	//lea(mem(rcx, rsi, 4), rdx)         // load address of c +  4*cs_c;
	lea(mem(rcx, rdi, 4), rdx)         // load address of c +  4*rs_c;

	lea(mem(rsi, rsi, 2), rax)         // rax = 3*cs_c;
	
	
	
	                                   // now avoid loading C if beta == 0
	
	vxorpd(ymm0, ymm0, ymm0)           // set ymm0 to zero.
	vucomisd(xmm0, xmm3)               // set ZF if beta == 0.
	je(.DBETAZERO)                     // if ZF = 1, jump to beta == 0 case



	cmp(imm(8), rdi)                   // set ZF if (8*rs_c) == 8.
	jz(.DCOLSTORED)                    // jump to column storage case
	

	
	label(.DROWSTORED)
	
	
	vfmadd231pd(mem(rcx), ymm3, ymm4)
	vmovupd(ymm4, mem(rcx))
	add(rdi, rcx)
	
	
	vfmadd231pd(mem(rcx), ymm3, ymm6)
	vmovupd(ymm6, mem(rcx))
	add(rdi, rcx)
	
	
	vfmadd231pd(mem(rcx), ymm3, ymm8)
	vmovupd(ymm8, mem(rcx))
	add(rdi, rcx)
	
	
	vfmadd231pd(mem(rcx), ymm3, ymm10)
	vmovupd(ymm10, mem(rcx))
	add(rdi, rcx)
	
	
	vfmadd231pd(mem(rcx), ymm3, ymm12)
	vmovupd(ymm12, mem(rcx))
	add(rdi, rcx)
	
	
	vfmadd231pd(mem(rcx), ymm3, ymm14)
	vmovupd(ymm14, mem(rcx))
	//add(rdi, rcx)
	
	
	jmp(.DDONE)                        // jump to end.



	label(.DCOLSTORED)


	vunpcklpd(ymm6, ymm4, ymm0)
	vunpckhpd(ymm6, ymm4, ymm1)
	vunpcklpd(ymm10, ymm8, ymm2)
	vunpckhpd(ymm10, ymm8, ymm3)
	vinsertf128(imm(0x1), xmm2, ymm0, ymm4)
	vinsertf128(imm(0x1), xmm3, ymm1, ymm6)
	vperm2f128(imm(0x31), ymm2, ymm0, ymm8)
	vperm2f128(imm(0x31), ymm3, ymm1, ymm10)

	vbroadcastsd(mem(rbx), ymm3)

	vfmadd231pd(mem(rcx), ymm3, ymm4)
	vfmadd231pd(mem(rcx, rsi, 1), ymm3, ymm6)
	vfmadd231pd(mem(rcx, rsi, 2), ymm3, ymm8)
	vfmadd231pd(mem(rcx, rax, 1), ymm3, ymm10)
	vmovupd(ymm4, mem(rcx))
	vmovupd(ymm6, mem(rcx, rsi, 1))
	vmovupd(ymm8, mem(rcx, rsi, 2))
	vmovupd(ymm10, mem(rcx, rax, 1))

	//lea(mem(rcx, rsi, 4), rcx)

	vunpcklpd(ymm14, ymm12, ymm0)
	vunpckhpd(ymm14, ymm12, ymm1)
	vextractf128(imm(0x1), ymm0, xmm2)
	vextractf128(imm(0x1), ymm1, xmm4)

	vfmadd231pd(mem(rdx), xmm3, xmm0)
	vfmadd231pd(mem(rdx, rsi, 1), xmm3, xmm1)
	vfmadd231pd(mem(rdx, rsi, 2), xmm3, xmm2)
	vfmadd231pd(mem(rdx, rax, 1), xmm3, xmm4)
	vmovupd(xmm0, mem(rdx))
	vmovupd(xmm1, mem(rdx, rsi, 1))
	vmovupd(xmm2, mem(rdx, rsi, 2))
	vmovupd(xmm4, mem(rdx, rax, 1))

	//lea(mem(rdx, rsi, 4), rdx)


	jmp(.DDONE)                        // jump to end.
	
	
	
	
	label(.DBETAZERO)


	cmp(imm(8), rdi)                   // set ZF if (8*rs_c) == 8.
	jz(.DCOLSTORBZ)                    // jump to column storage case
	

	
	label(.DROWSTORBZ)
	
	
	vmovupd(ymm4, mem(rcx))
	add(rdi, rcx)
	
	vmovupd(ymm6, mem(rcx))
	add(rdi, rcx)
	
	
	vmovupd(ymm8, mem(rcx))
	add(rdi, rcx)
	
	vmovupd(ymm10, mem(rcx))
	add(rdi, rcx)
	
	
	vmovupd(ymm12, mem(rcx))
	add(rdi, rcx)
	
	vmovupd(ymm14, mem(rcx))
	//add(rdi, rcx)

	
	jmp(.DDONE)                        // jump to end.



	label(.DCOLSTORBZ)


	vunpcklpd(ymm6, ymm4, ymm0)
	vunpckhpd(ymm6, ymm4, ymm1)
	vunpcklpd(ymm10, ymm8, ymm2)
	vunpckhpd(ymm10, ymm8, ymm3)
	vinsertf128(imm(0x1), xmm2, ymm0, ymm4)
	vinsertf128(imm(0x1), xmm3, ymm1, ymm6)
	vperm2f128(imm(0x31), ymm2, ymm0, ymm8)
	vperm2f128(imm(0x31), ymm3, ymm1, ymm10)

	vmovupd(ymm4, mem(rcx))
	vmovupd(ymm6, mem(rcx, rsi, 1))
	vmovupd(ymm8, mem(rcx, rsi, 2))
	vmovupd(ymm10, mem(rcx, rax, 1))

	//lea(mem(rcx, rsi, 4), rcx)

	vunpcklpd(ymm14, ymm12, ymm0)
	vunpckhpd(ymm14, ymm12, ymm1)
	vextractf128(imm(0x1), ymm0, xmm2)
	vextractf128(imm(0x1), ymm1, xmm4)

	vmovupd(xmm0, mem(rdx))
	vmovupd(xmm1, mem(rdx, rsi, 1))
	vmovupd(xmm2, mem(rdx, rsi, 2))
	vmovupd(xmm4, mem(rdx, rax, 1))

	//lea(mem(rdx, rsi, 4), rdx)
	
	
	
	
	label(.DDONE)
	
	

    end_asm(
	: // output operands (none)
	: // input operands
      [k_iter] "m" (k_iter),
      [k_left] "m" (k_left),
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
	  "ymm0", "ymm1", "ymm2", "ymm3", "ymm4", "ymm6", "ymm8", "ymm10",
	  "ymm12", "ymm14", "memory"
	)
	AOCL_DTL_TRACE_EXIT(AOCL_DTL_LEVEL_TRACE_7);
}

void bli_dgemmsup_rv_haswell_asm_5x4
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
	AOCL_DTL_TRACE_ENTRY(AOCL_DTL_LEVEL_TRACE_7);
	//void*    a_next = bli_auxinfo_next_a( data );
	//void*    b_next = bli_auxinfo_next_b( data );

	// Typecast local copies of integers in case dim_t and inc_t are a
	// different size than is expected by load instructions.
	uint64_t k_iter = k0 / 4;
	uint64_t k_left = k0 % 4;

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
	mov(var(cs_a), r9)                 // load cs_a
	lea(mem(, r8, 8), r8)              // rs_a *= sizeof(double)
	lea(mem(, r9, 8), r9)              // cs_a *= sizeof(double)

	lea(mem(r8, r8, 2), r13)           // r13 = 3*rs_a
	//lea(mem(r8, r8, 4), r15)           // r15 = 5*rs_a

	mov(var(b), rbx)                   // load address of b.
	mov(var(rs_b), r10)                // load rs_b
	//mov(var(cs_b), r11)                // load cs_b
	lea(mem(, r10, 8), r10)            // rs_b *= sizeof(double)
	//lea(mem(, r11, 8), r11)            // cs_b *= sizeof(double)
	
	                                   // NOTE: We cannot pre-load elements of a or b
	                                   // because it could eventually, in the last
	                                   // unrolled iter or the cleanup loop, result
	                                   // in reading beyond the bounds allocated mem
	                                   // (the likely result: a segmentation fault).

	mov(var(c), rcx)                   // load address of c
	mov(var(rs_c), rdi)                // load rs_c
	lea(mem(, rdi, 8), rdi)            // rs_c *= sizeof(double)

#if 0
	lea(mem(rcx, rdi, 2), rdx)         //
	lea(mem(rdx, rdi, 1), rdx)         // rdx = c + 3*rs_c;
	prefetch(0, mem(rcx, 7*8))         // prefetch c + 0*rs_c
	prefetch(0, mem(rcx, rdi, 1, 7*8)) // prefetch c + 1*rs_c
	prefetch(0, mem(rcx, rdi, 2, 7*8)) // prefetch c + 2*rs_c
	prefetch(0, mem(rdx, 7*8))         // prefetch c + 3*rs_c
	prefetch(0, mem(rdx, rdi, 1, 7*8)) // prefetch c + 4*rs_c
	prefetch(0, mem(rdx, rdi, 2, 7*8)) // prefetch c + 5*rs_c
#else
	cmp(imm(8), rdi)                   // set ZF if (8*rs_c) == 8.
	jz(.DCOLPFETCH)                    // jump to column storage case
	label(.DROWPFETCH)                 // row-stored prefetching on c

	lea(mem(rcx, rdi, 2), rdx)         //
	lea(mem(rdx, rdi, 1), rdx)         // rdx = c + 3*rs_c;
	prefetch(0, mem(rcx, 3*8))         // prefetch c + 0*rs_c
	prefetch(0, mem(rcx, rdi, 1, 3*8)) // prefetch c + 1*rs_c
	prefetch(0, mem(rcx, rdi, 2, 3*8)) // prefetch c + 2*rs_c
	prefetch(0, mem(rdx, 3*8))         // prefetch c + 3*rs_c
	prefetch(0, mem(rdx, rdi, 1, 3*8)) // prefetch c + 4*rs_c

	jmp(.DPOSTPFETCH)                  // jump to end of prefetching c
	label(.DCOLPFETCH)                 // column-stored prefetching c

	mov(var(cs_c), rsi)                // load cs_c to rsi (temporarily)
	lea(mem(, rsi, 8), rsi)            // cs_c *= sizeof(double)
	lea(mem(rcx, rsi, 2), rdx)         //
	lea(mem(rdx, rsi, 1), rdx)         // rdx = c + 3*cs_c;
	prefetch(0, mem(rcx, 4*8))         // prefetch c + 0*cs_c
	prefetch(0, mem(rcx, rsi, 1, 4*8)) // prefetch c + 1*cs_c
	prefetch(0, mem(rcx, rsi, 2, 4*8)) // prefetch c + 2*cs_c
	prefetch(0, mem(rdx, 4*8))         // prefetch c + 3*cs_c

	label(.DPOSTPFETCH)                // done prefetching c
#endif



#if 0
	lea(mem(rax, r9,  8), rdx)         // use rdx for prefetching b.
	lea(mem(rdx, r9,  8), rdx)         // rdx = b + 16*rs_b;
#else
	#if 1
	mov(r9, rsi)                       // rsi = rs_b;
	sal(imm(5), rsi)                   // rsi = 16*rs_b;
	lea(mem(rax, rsi, 1), rdx)         // rdx = b + 16*rs_b;
	#endif
#endif
	
	
	
	
	mov(var(k_iter), rsi)              // i = k_iter;
	test(rsi, rsi)                     // check i via logical AND.
	je(.DCONSIDKLEFT)                  // if i == 0, jump to code that
	                                   // contains the k_left loop.
	
	
	label(.DLOOPKITER)                 // MAIN LOOP
	
	
	// ---------------------------------- iteration 0

	vmovupd(mem(rbx,  0*32), ymm0)
	add(r10, rbx)                      // b += rs_b;

	vbroadcastsd(mem(rax        ), ymm2)
	vbroadcastsd(mem(rax, r8,  1), ymm3)
	vfmadd231pd(ymm0, ymm2, ymm4)
	vfmadd231pd(ymm0, ymm3, ymm6)
	
	vbroadcastsd(mem(rax, r8,  2), ymm2)
	vbroadcastsd(mem(rax, r13, 1), ymm3)
	vfmadd231pd(ymm0, ymm2, ymm8)
	vfmadd231pd(ymm0, ymm3, ymm10)
	
	vbroadcastsd(mem(rax, r8,  4), ymm2)
	add(r9, rax)                       // a += cs_a;
	vfmadd231pd(ymm0, ymm2, ymm12)

	
	// ---------------------------------- iteration 1

	vmovupd(mem(rbx,  0*32), ymm0)
	add(r10, rbx)                      // b += rs_b;

	vbroadcastsd(mem(rax        ), ymm2)
	vbroadcastsd(mem(rax, r8,  1), ymm3)
	vfmadd231pd(ymm0, ymm2, ymm4)
	vfmadd231pd(ymm0, ymm3, ymm6)
	
	vbroadcastsd(mem(rax, r8,  2), ymm2)
	vbroadcastsd(mem(rax, r13, 1), ymm3)
	vfmadd231pd(ymm0, ymm2, ymm8)
	vfmadd231pd(ymm0, ymm3, ymm10)
	
	vbroadcastsd(mem(rax, r8,  4), ymm2)
	add(r9, rax)                       // a += cs_a;
	vfmadd231pd(ymm0, ymm2, ymm12)
	

	// ---------------------------------- iteration 2

	vmovupd(mem(rbx,  0*32), ymm0)
	add(r10, rbx)                      // b += rs_b;

	vbroadcastsd(mem(rax        ), ymm2)
	vbroadcastsd(mem(rax, r8,  1), ymm3)
	vfmadd231pd(ymm0, ymm2, ymm4)
	vfmadd231pd(ymm0, ymm3, ymm6)
	
	vbroadcastsd(mem(rax, r8,  2), ymm2)
	vbroadcastsd(mem(rax, r13, 1), ymm3)
	vfmadd231pd(ymm0, ymm2, ymm8)
	vfmadd231pd(ymm0, ymm3, ymm10)
	
	vbroadcastsd(mem(rax, r8,  4), ymm2)
	add(r9, rax)                       // a += cs_a;
	vfmadd231pd(ymm0, ymm2, ymm12)
	

	// ---------------------------------- iteration 3

	vmovupd(mem(rbx,  0*32), ymm0)
	add(r10, rbx)                      // b += rs_b;

	vbroadcastsd(mem(rax        ), ymm2)
	vbroadcastsd(mem(rax, r8,  1), ymm3)
	vfmadd231pd(ymm0, ymm2, ymm4)
	vfmadd231pd(ymm0, ymm3, ymm6)
	
	vbroadcastsd(mem(rax, r8,  2), ymm2)
	vbroadcastsd(mem(rax, r13, 1), ymm3)
	vfmadd231pd(ymm0, ymm2, ymm8)
	vfmadd231pd(ymm0, ymm3, ymm10)
	
	vbroadcastsd(mem(rax, r8,  4), ymm2)
	add(r9, rax)                       // a += cs_a;
	vfmadd231pd(ymm0, ymm2, ymm12)
	
	
	
	dec(rsi)                           // i -= 1;
	jne(.DLOOPKITER)                   // iterate again if i != 0.
	
	
	
	
	
	
	label(.DCONSIDKLEFT)
	
	mov(var(k_left), rsi)              // i = k_left;
	test(rsi, rsi)                     // check i via logical AND.
	je(.DPOSTACCUM)                    // if i == 0, we're done; jump to end.
	                                   // else, we prepare to enter k_left loop.
	
	
	label(.DLOOPKLEFT)                 // EDGE LOOP
	
	vmovupd(mem(rbx,  0*32), ymm0)
	add(r10, rbx)                      // b += rs_b;

	vbroadcastsd(mem(rax        ), ymm2)
	vbroadcastsd(mem(rax, r8,  1), ymm3)
	vfmadd231pd(ymm0, ymm2, ymm4)
	vfmadd231pd(ymm0, ymm3, ymm6)
	
	vbroadcastsd(mem(rax, r8,  2), ymm2)
	vbroadcastsd(mem(rax, r13, 1), ymm3)
	vfmadd231pd(ymm0, ymm2, ymm8)
	vfmadd231pd(ymm0, ymm3, ymm10)
	
	vbroadcastsd(mem(rax, r8,  4), ymm2)
	add(r9, rax)                       // a += cs_a;
	vfmadd231pd(ymm0, ymm2, ymm12)
	
	
	dec(rsi)                           // i -= 1;
	jne(.DLOOPKLEFT)                   // iterate again if i != 0.
	
	
	
	label(.DPOSTACCUM)


	
	mov(var(alpha), rax)               // load address of alpha
	mov(var(beta), rbx)                // load address of beta
	vbroadcastsd(mem(rax), ymm0)       // load alpha and duplicate
	vbroadcastsd(mem(rbx), ymm3)       // load beta and duplicate
	
	vmulpd(ymm0, ymm4, ymm4)           // scale by alpha
	vmulpd(ymm0, ymm6, ymm6)
	vmulpd(ymm0, ymm8, ymm8)
	vmulpd(ymm0, ymm10, ymm10)
	vmulpd(ymm0, ymm12, ymm12)
	
	
	
	
	
	
	mov(var(cs_c), rsi)                // load cs_c
	lea(mem(, rsi, 8), rsi)            // rsi = cs_c * sizeof(double)
	
	//lea(mem(rcx, rsi, 4), rdx)         // load address of c +  4*cs_c;
	lea(mem(rcx, rdi, 4), rdx)         // load address of c +  4*rs_c;

	lea(mem(rsi, rsi, 2), rax)         // rax = 3*cs_c;
	
	
	
	                                   // now avoid loading C if beta == 0
	
	vxorpd(ymm0, ymm0, ymm0)           // set ymm0 to zero.
	vucomisd(xmm0, xmm3)               // set ZF if beta == 0.
	je(.DBETAZERO)                     // if ZF = 1, jump to beta == 0 case



	cmp(imm(8), rdi)                   // set ZF if (8*rs_c) == 8.
	jz(.DCOLSTORED)                    // jump to column storage case
	

	
	label(.DROWSTORED)
	
	
	vfmadd231pd(mem(rcx), ymm3, ymm4)
	vmovupd(ymm4, mem(rcx))
	add(rdi, rcx)
	
	
	vfmadd231pd(mem(rcx), ymm3, ymm6)
	vmovupd(ymm6, mem(rcx))
	add(rdi, rcx)
	
	
	vfmadd231pd(mem(rcx), ymm3, ymm8)
	vmovupd(ymm8, mem(rcx))
	add(rdi, rcx)
	
	
	vfmadd231pd(mem(rcx), ymm3, ymm10)
	vmovupd(ymm10, mem(rcx))
	add(rdi, rcx)
	
	
	vfmadd231pd(mem(rcx), ymm3, ymm12)
	vmovupd(ymm12, mem(rcx))
	//add(rdi, rcx)
	
	
	jmp(.DDONE)                        // jump to end.



	label(.DCOLSTORED)


	vunpcklpd(ymm6, ymm4, ymm0)
	vunpckhpd(ymm6, ymm4, ymm1)
	vunpcklpd(ymm10, ymm8, ymm2)
	vunpckhpd(ymm10, ymm8, ymm3)
	vinsertf128(imm(0x1), xmm2, ymm0, ymm4)
	vinsertf128(imm(0x1), xmm3, ymm1, ymm6)
	vperm2f128(imm(0x31), ymm2, ymm0, ymm8)
	vperm2f128(imm(0x31), ymm3, ymm1, ymm10)

	vbroadcastsd(mem(rbx), ymm3)

	vfmadd231pd(mem(rcx), ymm3, ymm4)
	vfmadd231pd(mem(rcx, rsi, 1), ymm3, ymm6)
	vfmadd231pd(mem(rcx, rsi, 2), ymm3, ymm8)
	vfmadd231pd(mem(rcx, rax, 1), ymm3, ymm10)
	vmovupd(ymm4, mem(rcx))
	vmovupd(ymm6, mem(rcx, rsi, 1))
	vmovupd(ymm8, mem(rcx, rsi, 2))
	vmovupd(ymm10, mem(rcx, rax, 1))

	//lea(mem(rcx, rsi, 4), rcx)

#if 0
	vunpcklpd(ymm14, ymm12, ymm0)
	vunpckhpd(ymm14, ymm12, ymm1)
	vextractf128(imm(0x1), ymm0, xmm2)
	vextractf128(imm(0x1), ymm1, xmm4)

	vfmadd231pd(mem(rdx), xmm3, xmm0)
	vfmadd231pd(mem(rdx, rsi, 1), xmm3, xmm1)
	vfmadd231pd(mem(rdx, rsi, 2), xmm3, xmm2)
	vfmadd231pd(mem(rdx, rax, 1), xmm3, xmm4)
	vmovupd(xmm0, mem(rdx))
	vmovupd(xmm1, mem(rdx, rsi, 1))
	vmovupd(xmm2, mem(rdx, rsi, 2))
	vmovupd(xmm4, mem(rdx, rax, 1))
#else
	vmovlpd(mem(rdx), xmm0, xmm0)
	vmovhpd(mem(rdx, rsi, 1), xmm0, xmm0)
	vmovlpd(mem(rdx, rsi, 2), xmm1, xmm1)
	vmovhpd(mem(rdx, rax, 1), xmm1, xmm1)
	vperm2f128(imm(0x20), ymm1, ymm0, ymm0)

	vfmadd213pd(ymm12, ymm3, ymm0)
	vextractf128(imm(1), ymm0, xmm1)
	vmovlpd(xmm0, mem(rdx))
	vmovhpd(xmm0, mem(rdx, rsi, 1))
	vmovlpd(xmm1, mem(rdx, rsi, 2))
	vmovhpd(xmm1, mem(rdx, rax, 1))
#endif

	//lea(mem(rdx, rsi, 4), rdx)


	jmp(.DDONE)                        // jump to end.
	
	
	
	
	label(.DBETAZERO)


	cmp(imm(8), rdi)                   // set ZF if (8*rs_c) == 8.
	jz(.DCOLSTORBZ)                    // jump to column storage case
	

	
	label(.DROWSTORBZ)
	
	
	vmovupd(ymm4, mem(rcx))
	add(rdi, rcx)
	
	vmovupd(ymm6, mem(rcx))
	add(rdi, rcx)
	
	
	vmovupd(ymm8, mem(rcx))
	add(rdi, rcx)
	
	vmovupd(ymm10, mem(rcx))
	add(rdi, rcx)
	
	
	vmovupd(ymm12, mem(rcx))
	//add(rdi, rcx)

	
	jmp(.DDONE)                        // jump to end.



	label(.DCOLSTORBZ)


	vunpcklpd(ymm6, ymm4, ymm0)
	vunpckhpd(ymm6, ymm4, ymm1)
	vunpcklpd(ymm10, ymm8, ymm2)
	vunpckhpd(ymm10, ymm8, ymm3)
	vinsertf128(imm(0x1), xmm2, ymm0, ymm4)
	vinsertf128(imm(0x1), xmm3, ymm1, ymm6)
	vperm2f128(imm(0x31), ymm2, ymm0, ymm8)
	vperm2f128(imm(0x31), ymm3, ymm1, ymm10)

	vmovupd(ymm4, mem(rcx))
	vmovupd(ymm6, mem(rcx, rsi, 1))
	vmovupd(ymm8, mem(rcx, rsi, 2))
	vmovupd(ymm10, mem(rcx, rax, 1))

	//lea(mem(rcx, rsi, 4), rcx)

#if 0
	vunpcklpd(ymm14, ymm12, ymm0)
	vunpckhpd(ymm14, ymm12, ymm1)
	vextractf128(imm(0x1), ymm0, xmm2)
	vextractf128(imm(0x1), ymm1, xmm4)

	vmovupd(xmm0, mem(rdx))
	vmovupd(xmm1, mem(rdx, rsi, 1))
	vmovupd(xmm2, mem(rdx, rsi, 2))
	vmovupd(xmm4, mem(rdx, rax, 1))
#else
	vmovupd(ymm12, ymm0)

	vextractf128(imm(1), ymm0, xmm1)
	vmovlpd(xmm0, mem(rdx))
	vmovhpd(xmm0, mem(rdx, rsi, 1))
	vmovlpd(xmm1, mem(rdx, rsi, 2))
	vmovhpd(xmm1, mem(rdx, rax, 1))
#endif

	//lea(mem(rdx, rsi, 4), rdx)
	
	
	
	
	label(.DDONE)
	
	

    end_asm(
	: // output operands (none)
	: // input operands
      [k_iter] "m" (k_iter),
      [k_left] "m" (k_left),
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
	  "ymm0", "ymm1", "ymm2", "ymm3", "ymm4", "ymm6", "ymm8", "ymm10",
	  "ymm12", "memory"
	)
	AOCL_DTL_TRACE_EXIT(AOCL_DTL_LEVEL_TRACE_7);
}

void bli_dgemmsup_rv_haswell_asm_4x4
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
	AOCL_DTL_TRACE_ENTRY(AOCL_DTL_LEVEL_TRACE_7);
	//void*    a_next = bli_auxinfo_next_a( data );
	//void*    b_next = bli_auxinfo_next_b( data );

	// Typecast local copies of integers in case dim_t and inc_t are a
	// different size than is expected by load instructions.
	uint64_t k_iter = k0 / 4;
	uint64_t k_left = k0 % 4;

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
	mov(var(cs_a), r9)                 // load cs_a
	lea(mem(, r8, 8), r8)              // rs_a *= sizeof(double)
	lea(mem(, r9, 8), r9)              // cs_a *= sizeof(double)

	lea(mem(r8, r8, 2), r13)           // r13 = 3*rs_a
	//lea(mem(r8, r8, 4), r15)           // r15 = 5*rs_a

	mov(var(b), rbx)                   // load address of b.
	mov(var(rs_b), r10)                // load rs_b
	//mov(var(cs_b), r11)                // load cs_b
	lea(mem(, r10, 8), r10)            // rs_b *= sizeof(double)
	//lea(mem(, r11, 8), r11)            // cs_b *= sizeof(double)
	
	                                   // NOTE: We cannot pre-load elements of a or b
	                                   // because it could eventually, in the last
	                                   // unrolled iter or the cleanup loop, result
	                                   // in reading beyond the bounds allocated mem
	                                   // (the likely result: a segmentation fault).

	mov(var(c), rcx)                   // load address of c
	mov(var(rs_c), rdi)                // load rs_c
	lea(mem(, rdi, 8), rdi)            // rs_c *= sizeof(double)

#if 0
	lea(mem(rcx, rdi, 2), rdx)         //
	lea(mem(rdx, rdi, 1), rdx)         // rdx = c + 3*rs_c;
	prefetch(0, mem(rcx, 3*8))         // prefetch c + 0*rs_c
	prefetch(0, mem(rcx, rdi, 1, 3*8)) // prefetch c + 1*rs_c
	prefetch(0, mem(rcx, rdi, 2, 3*8)) // prefetch c + 2*rs_c
	prefetch(0, mem(rdx, 3*8))         // prefetch c + 3*rs_c
#else
	cmp(imm(8), rdi)                   // set ZF if (8*rs_c) == 8.
	jz(.DCOLPFETCH)                    // jump to column storage case
	label(.DROWPFETCH)                 // row-stored prefetching on c

	lea(mem(rcx, rdi, 2), rdx)         //
	lea(mem(rdx, rdi, 1), rdx)         // rdx = c + 3*rs_c;
	prefetch(0, mem(rcx, 3*8))         // prefetch c + 0*rs_c
	prefetch(0, mem(rcx, rdi, 1, 3*8)) // prefetch c + 1*rs_c
	prefetch(0, mem(rcx, rdi, 2, 3*8)) // prefetch c + 2*rs_c
	prefetch(0, mem(rdx, 3*8))         // prefetch c + 3*rs_c

	jmp(.DPOSTPFETCH)                  // jump to end of prefetching c
	label(.DCOLPFETCH)                 // column-stored prefetching c

	mov(var(cs_c), rsi)                // load cs_c to rsi (temporarily)
	lea(mem(, rsi, 8), rsi)            // cs_c *= sizeof(double)
	lea(mem(rcx, rsi, 2), rdx)         //
	lea(mem(rdx, rsi, 1), rdx)         // rdx = c + 3*cs_c;
	prefetch(0, mem(rcx, 3*8))         // prefetch c + 0*cs_c
	prefetch(0, mem(rcx, rsi, 1, 3*8)) // prefetch c + 1*cs_c
	prefetch(0, mem(rcx, rsi, 2, 3*8)) // prefetch c + 2*cs_c
	prefetch(0, mem(rdx, 3*8))         // prefetch c + 3*cs_c

	label(.DPOSTPFETCH)                // done prefetching c
#endif
	
	
	
	
	mov(var(k_iter), rsi)              // i = k_iter;
	test(rsi, rsi)                     // check i via logical AND.
	je(.DCONSIDKLEFT)                  // if i == 0, jump to code that
	                                   // contains the k_left loop.
	
	
	label(.DLOOPKITER)                 // MAIN LOOP
	
	
	// ---------------------------------- iteration 0
	
	vmovupd(mem(rbx,  0*32), ymm0)
	add(r10, rbx)                      // b += rs_b;

	vbroadcastsd(mem(rax        ), ymm2)
	vbroadcastsd(mem(rax, r8,  1), ymm3)
	vfmadd231pd(ymm0, ymm2, ymm4)
	vfmadd231pd(ymm0, ymm3, ymm6)
	
	vbroadcastsd(mem(rax, r8,  2), ymm2)
	vbroadcastsd(mem(rax, r13, 1), ymm3)
	add(r9, rax)                       // a += cs_a;
	vfmadd231pd(ymm0, ymm2, ymm8)
	vfmadd231pd(ymm0, ymm3, ymm10)
	
	
	// ---------------------------------- iteration 1

	vmovupd(mem(rbx, 0*32), ymm0)
	add(r10, rbx)                      // b += rs_b;
	
	vbroadcastsd(mem(rax        ), ymm2)
	vbroadcastsd(mem(rax, r8,  1), ymm3)
	vfmadd231pd(ymm0, ymm2, ymm4)
	vfmadd231pd(ymm0, ymm3, ymm6)
	
	vbroadcastsd(mem(rax, r8,  2), ymm2)
	vbroadcastsd(mem(rax, r13, 1), ymm3)
	add(r9, rax)                       // a += cs_a;
	vfmadd231pd(ymm0, ymm2, ymm8)
	vfmadd231pd(ymm0, ymm3, ymm10)
	

	// ---------------------------------- iteration 2
	
	vmovupd(mem(rbx, 0*32), ymm0)
	add(r10, rbx)                      // b += rs_b;
	
	vbroadcastsd(mem(rax        ), ymm2)
	vbroadcastsd(mem(rax, r8,  1), ymm3)
	vfmadd231pd(ymm0, ymm2, ymm4)
	vfmadd231pd(ymm0, ymm3, ymm6)
	
	vbroadcastsd(mem(rax, r8,  2), ymm2)
	vbroadcastsd(mem(rax, r13, 1), ymm3)
	add(r9, rax)                       // a += cs_a;
	vfmadd231pd(ymm0, ymm2, ymm8)
	vfmadd231pd(ymm0, ymm3, ymm10)
	

	// ---------------------------------- iteration 3

	vmovupd(mem(rbx, 0*32), ymm0)
	add(r10, rbx)                      // b += rs_b;
	
	vbroadcastsd(mem(rax        ), ymm2)
	vbroadcastsd(mem(rax, r8,  1), ymm3)
	vfmadd231pd(ymm0, ymm2, ymm4)
	vfmadd231pd(ymm0, ymm3, ymm6)
	
	vbroadcastsd(mem(rax, r8,  2), ymm2)
	vbroadcastsd(mem(rax, r13, 1), ymm3)
	add(r9, rax)                       // a += cs_a;
	vfmadd231pd(ymm0, ymm2, ymm8)
	vfmadd231pd(ymm0, ymm3, ymm10)
	
	
	
	dec(rsi)                           // i -= 1;
	jne(.DLOOPKITER)                   // iterate again if i != 0.
	
	
	
	
	
	
	label(.DCONSIDKLEFT)
	
	mov(var(k_left), rsi)              // i = k_left;
	test(rsi, rsi)                     // check i via logical AND.
	je(.DPOSTACCUM)                    // if i == 0, we're done; jump to end.
	                                   // else, we prepare to enter k_left loop.
	
	
	label(.DLOOPKLEFT)                 // EDGE LOOP
	
	vmovupd(mem(rbx,  0*32), ymm0)
	add(r10, rbx)                      // b += rs_b;
	
	vbroadcastsd(mem(rax        ), ymm2)
	vbroadcastsd(mem(rax, r8,  1), ymm3)
	vfmadd231pd(ymm0, ymm2, ymm4)
	vfmadd231pd(ymm0, ymm3, ymm6)
	
	vbroadcastsd(mem(rax, r8,  2), ymm2)
	vbroadcastsd(mem(rax, r13, 1), ymm3)
	add(r9, rax)                       // a += cs_a;
	vfmadd231pd(ymm0, ymm2, ymm8)
	vfmadd231pd(ymm0, ymm3, ymm10)
	
	
	dec(rsi)                           // i -= 1;
	jne(.DLOOPKLEFT)                   // iterate again if i != 0.
	
	
	
	label(.DPOSTACCUM)


	
	mov(var(alpha), rax)               // load address of alpha
	mov(var(beta), rbx)                // load address of beta
	vbroadcastsd(mem(rax), ymm0)       // load alpha and duplicate
	vbroadcastsd(mem(rbx), ymm3)       // load beta and duplicate
	
	vmulpd(ymm0, ymm4, ymm4)           // scale by alpha
	vmulpd(ymm0, ymm6, ymm6)
	vmulpd(ymm0, ymm8, ymm8)
	vmulpd(ymm0, ymm10, ymm10)
	
	
	
	
	
	
	mov(var(cs_c), rsi)                // load cs_c
	lea(mem(, rsi, 8), rsi)            // rsi = cs_c * sizeof(double)
	
	//lea(mem(rcx, rsi, 4), rdx)         // load address of c +  4*cs_c;
	//lea(mem(rcx, rdi, 4), r14)         // load address of c +  4*rs_c;

	lea(mem(rsi, rsi, 2), rax)         // rax = 3*cs_c;
	
	
	
	                                   // now avoid loading C if beta == 0
	
	vxorpd(ymm0, ymm0, ymm0)           // set ymm0 to zero.
	vucomisd(xmm0, xmm3)               // set ZF if beta == 0.
	je(.DBETAZERO)                     // if ZF = 1, jump to beta == 0 case
	

	
	cmp(imm(8), rdi)                   // set ZF if (8*rs_c) == 8.
	jz(.DCOLSTORED)                    // jump to column storage case



	label(.DROWSTORED)
	
	
	vfmadd231pd(mem(rcx), ymm3, ymm4)
	vmovupd(ymm4, mem(rcx))
	add(rdi, rcx)
	
	
	vfmadd231pd(mem(rcx), ymm3, ymm6)
	vmovupd(ymm6, mem(rcx))
	add(rdi, rcx)
	
	
	vfmadd231pd(mem(rcx), ymm3, ymm8)
	vmovupd(ymm8, mem(rcx))
	add(rdi, rcx)
	
	
	vfmadd231pd(mem(rcx), ymm3, ymm10)
	vmovupd(ymm10, mem(rcx))
	//add(rdi, rcx)
	
	
	jmp(.DDONE)                        // jump to end.



	label(.DCOLSTORED)


	vunpcklpd(ymm6, ymm4, ymm0)
	vunpckhpd(ymm6, ymm4, ymm1)
	vunpcklpd(ymm10, ymm8, ymm2)
	vunpckhpd(ymm10, ymm8, ymm3)
	vinsertf128(imm(0x1), xmm2, ymm0, ymm4)
	vinsertf128(imm(0x1), xmm3, ymm1, ymm6)
	vperm2f128(imm(0x31), ymm2, ymm0, ymm8)
	vperm2f128(imm(0x31), ymm3, ymm1, ymm10)

	vbroadcastsd(mem(rbx), ymm3)

	vfmadd231pd(mem(rcx), ymm3, ymm4)
	vfmadd231pd(mem(rcx, rsi, 1), ymm3, ymm6)
	vfmadd231pd(mem(rcx, rsi, 2), ymm3, ymm8)
	vfmadd231pd(mem(rcx, rax, 1), ymm3, ymm10)
	vmovupd(ymm4, mem(rcx))
	vmovupd(ymm6, mem(rcx, rsi, 1))
	vmovupd(ymm8, mem(rcx, rsi, 2))
	vmovupd(ymm10, mem(rcx, rax, 1))

	//lea(mem(rcx, rsi, 4), rcx)


	jmp(.DDONE)                        // jump to end.
	
	
	
	
	label(.DBETAZERO)


	cmp(imm(8), rdi)                   // set ZF if (8*rs_c) == 8.
	jz(.DCOLSTORBZ)                    // jump to column storage case
	

	
	label(.DROWSTORBZ)
	
	
	vmovupd(ymm4, mem(rcx))
	add(rdi, rcx)
	
	vmovupd(ymm6, mem(rcx))
	add(rdi, rcx)
	
	
	vmovupd(ymm8, mem(rcx))
	add(rdi, rcx)
	
	vmovupd(ymm10, mem(rcx))
	//add(rdi, rcx)


	jmp(.DDONE)                        // jump to end.



	label(.DCOLSTORBZ)
	
	
	vunpcklpd(ymm6, ymm4, ymm0)
	vunpckhpd(ymm6, ymm4, ymm1)
	vunpcklpd(ymm10, ymm8, ymm2)
	vunpckhpd(ymm10, ymm8, ymm3)
	vinsertf128(imm(0x1), xmm2, ymm0, ymm4)
	vinsertf128(imm(0x1), xmm3, ymm1, ymm6)
	vperm2f128(imm(0x31), ymm2, ymm0, ymm8)
	vperm2f128(imm(0x31), ymm3, ymm1, ymm10)

	vmovupd(ymm4, mem(rcx))
	vmovupd(ymm6, mem(rcx, rsi, 1))
	vmovupd(ymm8, mem(rcx, rsi, 2))
	vmovupd(ymm10, mem(rcx, rax, 1))

	//lea(mem(rcx, rsi, 4), rcx)

	
	
	
	label(.DDONE)
	
	

    end_asm(
	: // output operands (none)
	: // input operands
      [k_iter] "m" (k_iter),
      [k_left] "m" (k_left),
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
	  "ymm0", "ymm1", "ymm2", "ymm3", "ymm4", "ymm6", "ymm8", "ymm10", "memory"
	)
	AOCL_DTL_TRACE_EXIT(AOCL_DTL_LEVEL_TRACE_7);
}

void bli_dgemmsup_rv_haswell_asm_3x4
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
	AOCL_DTL_TRACE_ENTRY(AOCL_DTL_LEVEL_TRACE_7);
	//void*    a_next = bli_auxinfo_next_a( data );
	//void*    b_next = bli_auxinfo_next_b( data );

	// Typecast local copies of integers in case dim_t and inc_t are a
	// different size than is expected by load instructions.
	uint64_t k_iter = k0 / 4;
	uint64_t k_left = k0 % 4;

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
	mov(var(cs_a), r9)                 // load cs_a
	lea(mem(, r8, 8), r8)              // rs_a *= sizeof(double)
	lea(mem(, r9, 8), r9)              // cs_a *= sizeof(double)

	//lea(mem(r8, r8, 2), r13)           // r13 = 3*rs_a
	//lea(mem(r8, r8, 4), r15)           // r15 = 5*rs_a

	mov(var(b), rbx)                   // load address of b.
	mov(var(rs_b), r10)                // load rs_b
	//mov(var(cs_b), r11)                // load cs_b
	lea(mem(, r10, 8), r10)            // rs_b *= sizeof(double)
	//lea(mem(, r11, 8), r11)            // cs_b *= sizeof(double)
	
	                                   // NOTE: We cannot pre-load elements of a or b
	                                   // because it could eventually, in the last
	                                   // unrolled iter or the cleanup loop, result
	                                   // in reading beyond the bounds allocated mem
	                                   // (the likely result: a segmentation fault).

	mov(var(c), rcx)                   // load address of c
	mov(var(rs_c), rdi)                // load rs_c
	lea(mem(, rdi, 8), rdi)            // rs_c *= sizeof(double)

#if 0
	lea(mem(rcx, rdi, 2), rdx)         //
	lea(mem(rdx, rdi, 1), rdx)         // rdx = c + 3*rs_c;
	prefetch(0, mem(rcx, 7*8))         // prefetch c + 0*rs_c
	prefetch(0, mem(rcx, rdi, 1, 7*8)) // prefetch c + 1*rs_c
	prefetch(0, mem(rcx, rdi, 2, 7*8)) // prefetch c + 2*rs_c
	prefetch(0, mem(rdx, 7*8))         // prefetch c + 3*rs_c
#else
	cmp(imm(8), rdi)                   // set ZF if (8*rs_c) == 8.
	jz(.DCOLPFETCH)                    // jump to column storage case
	label(.DROWPFETCH)                 // row-stored prefetching on c

	//lea(mem(rcx, rdi, 2), rdx)         //
	//lea(mem(rdx, rdi, 1), rdx)         // rdx = c + 3*rs_c;
	prefetch(0, mem(rcx, 3*8))         // prefetch c + 0*rs_c
	prefetch(0, mem(rcx, rdi, 1, 3*8)) // prefetch c + 1*rs_c
	prefetch(0, mem(rcx, rdi, 2, 3*8)) // prefetch c + 2*rs_c

	jmp(.DPOSTPFETCH)                  // jump to end of prefetching c
	label(.DCOLPFETCH)                 // column-stored prefetching c

	mov(var(cs_c), rsi)                // load cs_c to rsi (temporarily)
	lea(mem(, rsi, 8), rsi)            // cs_c *= sizeof(double)
	lea(mem(rcx, rsi, 2), rdx)         //
	lea(mem(rdx, rsi, 1), rdx)         // rdx = c + 3*cs_c;
	prefetch(0, mem(rcx, 2*8))         // prefetch c + 0*cs_c
	prefetch(0, mem(rcx, rsi, 1, 2*8)) // prefetch c + 1*cs_c
	prefetch(0, mem(rcx, rsi, 2, 2*8)) // prefetch c + 2*cs_c
	prefetch(0, mem(rdx, 2*8))         // prefetch c + 3*cs_c

	label(.DPOSTPFETCH)                // done prefetching c
#endif
	
	
	
	
	mov(var(k_iter), rsi)              // i = k_iter;
	test(rsi, rsi)                     // check i via logical AND.
	je(.DCONSIDKLEFT)                  // if i == 0, jump to code that
	                                   // contains the k_left loop.
	
	
	label(.DLOOPKITER)                 // MAIN LOOP
	
	
	// ---------------------------------- iteration 0
	
	vmovupd(mem(rbx, 0*32), ymm0)
	add(r10, rbx)                      // b += rs_b;

	vbroadcastsd(mem(rax        ), ymm2)
	vbroadcastsd(mem(rax, r8,  1), ymm3)
	vfmadd231pd(ymm0, ymm2, ymm4)
	vfmadd231pd(ymm0, ymm3, ymm6)
	
	vbroadcastsd(mem(rax, r8,  2), ymm2)
	add(r9, rax)                       // a += cs_a;
	vfmadd231pd(ymm0, ymm2, ymm8)
	
	
	// ---------------------------------- iteration 1

	vmovupd(mem(rbx, 0*32), ymm0)
	add(r10, rbx)                      // b += rs_b;

	vbroadcastsd(mem(rax        ), ymm2)
	vbroadcastsd(mem(rax, r8,  1), ymm3)
	vfmadd231pd(ymm0, ymm2, ymm4)
	vfmadd231pd(ymm0, ymm3, ymm6)
	
	vbroadcastsd(mem(rax, r8,  2), ymm2)
	add(r9, rax)                       // a += cs_a;
	vfmadd231pd(ymm0, ymm2, ymm8)
	

	// ---------------------------------- iteration 2
	
	vmovupd(mem(rbx, 0*32), ymm0)
	add(r10, rbx)                      // b += rs_b;

	vbroadcastsd(mem(rax        ), ymm2)
	vbroadcastsd(mem(rax, r8,  1), ymm3)
	vfmadd231pd(ymm0, ymm2, ymm4)
	vfmadd231pd(ymm0, ymm3, ymm6)
	
	vbroadcastsd(mem(rax, r8,  2), ymm2)
	add(r9, rax)                       // a += cs_a;
	vfmadd231pd(ymm0, ymm2, ymm8)
	

	// ---------------------------------- iteration 3

	vmovupd(mem(rbx, 0*32), ymm0)
	add(r10, rbx)                      // b += rs_b;

	vbroadcastsd(mem(rax        ), ymm2)
	vbroadcastsd(mem(rax, r8,  1), ymm3)
	vfmadd231pd(ymm0, ymm2, ymm4)
	vfmadd231pd(ymm0, ymm3, ymm6)
	
	vbroadcastsd(mem(rax, r8,  2), ymm2)
	add(r9, rax)                       // a += cs_a;
	vfmadd231pd(ymm0, ymm2, ymm8)
	
	
	
	dec(rsi)                           // i -= 1;
	jne(.DLOOPKITER)                   // iterate again if i != 0.
	
	
	
	
	
	
	label(.DCONSIDKLEFT)
	
	mov(var(k_left), rsi)              // i = k_left;
	test(rsi, rsi)                     // check i via logical AND.
	je(.DPOSTACCUM)                    // if i == 0, we're done; jump to end.
	                                   // else, we prepare to enter k_left loop.
	
	
	label(.DLOOPKLEFT)                 // EDGE LOOP
	
	vmovupd(mem(rbx, 0*32), ymm0)
	add(r10, rbx)                      // b += rs_b;

	vbroadcastsd(mem(rax        ), ymm2)
	vbroadcastsd(mem(rax, r8,  1), ymm3)
	vfmadd231pd(ymm0, ymm2, ymm4)
	vfmadd231pd(ymm0, ymm3, ymm6)
	
	vbroadcastsd(mem(rax, r8,  2), ymm2)
	add(r9, rax)                       // a += cs_a;
	vfmadd231pd(ymm0, ymm2, ymm8)
	
	
	dec(rsi)                           // i -= 1;
	jne(.DLOOPKLEFT)                   // iterate again if i != 0.
	
	
	
	label(.DPOSTACCUM)


	
	mov(var(alpha), rax)               // load address of alpha
	mov(var(beta), rbx)                // load address of beta
	vbroadcastsd(mem(rax), ymm0)       // load alpha and duplicate
	vbroadcastsd(mem(rbx), ymm3)       // load beta and duplicate
	
	vmulpd(ymm0, ymm4, ymm4)           // scale by alpha
	vmulpd(ymm0, ymm6, ymm6)
	vmulpd(ymm0, ymm8, ymm8)
	
	
	
	
	
	
	mov(var(cs_c), rsi)                // load cs_c
	lea(mem(, rsi, 8), rsi)            // rsi = cs_c * sizeof(double)
	
	//lea(mem(rcx, rsi, 4), rdx)         // load address of c +  4*cs_c;
	lea(mem(rcx, rdi, 2), rdx)         // load address of c +  2*rs_c;

	lea(mem(rsi, rsi, 2), rax)         // rax = 3*cs_c;
	
	
	
	                                   // now avoid loading C if beta == 0
	
	vxorpd(ymm0, ymm0, ymm0)           // set ymm0 to zero.
	vucomisd(xmm0, xmm3)               // set ZF if beta == 0.
	je(.DBETAZERO)                     // if ZF = 1, jump to beta == 0 case
	


	cmp(imm(8), rdi)                   // set ZF if (8*rs_c) == 8.
	jz(.DCOLSTORED)                    // jump to column storage case


	
	label(.DROWSTORED)
	
	
	vfmadd231pd(mem(rcx), ymm3, ymm4)
	vmovupd(ymm4, mem(rcx))
	add(rdi, rcx)


	vfmadd231pd(mem(rcx), ymm3, ymm6)
	vmovupd(ymm6, mem(rcx))
	add(rdi, rcx)


	vfmadd231pd(mem(rcx), ymm3, ymm8)
	vmovupd(ymm8, mem(rcx))
	//add(rdi, rcx)
	
	
	jmp(.DDONE)                        // jump to end.
	


	label(.DCOLSTORED)


	vunpcklpd(ymm6, ymm4, ymm0)
	vunpckhpd(ymm6, ymm4, ymm1)
	vunpcklpd(ymm10, ymm8, ymm2)
	vunpckhpd(ymm10, ymm8, ymm3)
	vinsertf128(imm(0x1), xmm2, ymm0, ymm4)
	vinsertf128(imm(0x1), xmm3, ymm1, ymm6)
	vperm2f128(imm(0x31), ymm2, ymm0, ymm8)
	vperm2f128(imm(0x31), ymm3, ymm1, ymm10)

	vextractf128(imm(0x1), ymm4, xmm12)
	vextractf128(imm(0x1), ymm6, xmm13)
	vextractf128(imm(0x1), ymm8, xmm14)
	vextractf128(imm(0x1), ymm10, xmm15)

	vbroadcastsd(mem(rbx), ymm3)

	vfmadd231pd(mem(rcx), xmm3, xmm4)
	vfmadd231pd(mem(rcx, rsi, 1), xmm3, xmm6)
	vfmadd231pd(mem(rcx, rsi, 2), xmm3, xmm8)
	vfmadd231pd(mem(rcx, rax, 1), xmm3, xmm10)
	vmovupd(xmm4, mem(rcx))
	vmovupd(xmm6, mem(rcx, rsi, 1))
	vmovupd(xmm8, mem(rcx, rsi, 2))
	vmovupd(xmm10, mem(rcx, rax, 1))

	//lea(mem(rcx, rsi, 4), rcx)

	vfmadd231sd(mem(rdx), xmm3, xmm12)
	vfmadd231sd(mem(rdx, rsi, 1), xmm3, xmm13)
	vfmadd231sd(mem(rdx, rsi, 2), xmm3, xmm14)
	vfmadd231sd(mem(rdx, rax, 1), xmm3, xmm15)
	vmovsd(xmm12, mem(rdx))
	vmovsd(xmm13, mem(rdx, rsi, 1))
	vmovsd(xmm14, mem(rdx, rsi, 2))
	vmovsd(xmm15, mem(rdx, rax, 1))
	
	//lea(mem(rdx, rsi, 4), rdx)


	jmp(.DDONE)                        // jump to end.


	
	
	label(.DBETAZERO)
	

	cmp(imm(8), rdi)                   // set ZF if (8*rs_c) == 8.
	jz(.DCOLSTORBZ)                    // jump to column storage case


	
	label(.DROWSTORBZ)
	
	
	vmovupd(ymm4, mem(rcx))
	add(rdi, rcx)

	vmovupd(ymm6, mem(rcx))
	add(rdi, rcx)

	vmovupd(ymm8, mem(rcx))
	//add(rdi, rcx)
	
	
	jmp(.DDONE)                        // jump to end.



	label(.DCOLSTORBZ)


	vunpcklpd(ymm6, ymm4, ymm0)
	vunpckhpd(ymm6, ymm4, ymm1)
	vunpcklpd(ymm10, ymm8, ymm2)
	vunpckhpd(ymm10, ymm8, ymm3)
	vinsertf128(imm(0x1), xmm2, ymm0, ymm4)
	vinsertf128(imm(0x1), xmm3, ymm1, ymm6)
	vperm2f128(imm(0x31), ymm2, ymm0, ymm8)
	vperm2f128(imm(0x31), ymm3, ymm1, ymm10)

	vextractf128(imm(0x1), ymm4, xmm12)
	vextractf128(imm(0x1), ymm6, xmm13)
	vextractf128(imm(0x1), ymm8, xmm14)
	vextractf128(imm(0x1), ymm10, xmm15)

	vmovupd(xmm4, mem(rcx))
	vmovupd(xmm6, mem(rcx, rsi, 1))
	vmovupd(xmm8, mem(rcx, rsi, 2))
	vmovupd(xmm10, mem(rcx, rax, 1))

	//lea(mem(rcx, rsi, 4), rcx)

	vmovsd(xmm12, mem(rdx))
	vmovsd(xmm13, mem(rdx, rsi, 1))
	vmovsd(xmm14, mem(rdx, rsi, 2))
	vmovsd(xmm15, mem(rdx, rax, 1))

	//lea(mem(rdx, rsi, 4), rdx)

	
	
	
	label(.DDONE)
	
	

    end_asm(
	: // output operands (none)
	: // input operands
      [k_iter] "m" (k_iter),
      [k_left] "m" (k_left),
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
	  "ymm0", "ymm1", "ymm2", "ymm3", "ymm4", "ymm6", "ymm8", "ymm10", "memory"
	)
	AOCL_DTL_TRACE_EXIT(AOCL_DTL_LEVEL_TRACE_7);
}

void bli_dgemmsup_rv_haswell_asm_2x4
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
	AOCL_DTL_TRACE_ENTRY(AOCL_DTL_LEVEL_TRACE_7);
	//void*    a_next = bli_auxinfo_next_a( data );
	//void*    b_next = bli_auxinfo_next_b( data );

	// Typecast local copies of integers in case dim_t and inc_t are a
	// different size than is expected by load instructions.
	uint64_t k_iter = k0 / 4;
	uint64_t k_left = k0 % 4;

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
	mov(var(cs_a), r9)                 // load cs_a
	lea(mem(, r8, 8), r8)              // rs_a *= sizeof(double)
	lea(mem(, r9, 8), r9)              // cs_a *= sizeof(double)

	//lea(mem(r8, r8, 2), r13)           // r13 = 3*rs_a
	//lea(mem(r8, r8, 4), r15)           // r15 = 5*rs_a

	mov(var(b), rbx)                   // load address of b.
	mov(var(rs_b), r10)                // load rs_b
	//mov(var(cs_b), r11)                // load cs_b
	lea(mem(, r10, 8), r10)            // rs_b *= sizeof(double)
	//lea(mem(, r11, 8), r11)            // cs_b *= sizeof(double)
	
	                                   // NOTE: We cannot pre-load elements of a or b
	                                   // because it could eventually, in the last
	                                   // unrolled iter or the cleanup loop, result
	                                   // in reading beyond the bounds allocated mem
	                                   // (the likely result: a segmentation fault).

	mov(var(c), rcx)                   // load address of c
	mov(var(rs_c), rdi)                // load rs_c
	lea(mem(, rdi, 8), rdi)            // rs_c *= sizeof(double)

#if 0
	lea(mem(rcx, rdi, 2), rdx)         //
	//lea(mem(rdx, rdi, 1), rdx)         // rdx = c + 3*rs_c;
	prefetch(0, mem(rcx, 7*8))         // prefetch c + 0*rs_c
	prefetch(0, mem(rcx, rdi, 1, 7*8)) // prefetch c + 1*rs_c
#else
	cmp(imm(8), rdi)                   // set ZF if (8*rs_c) == 8.
	jz(.DCOLPFETCH)                    // jump to column storage case
	label(.DROWPFETCH)                 // row-stored prefetching on c

	//lea(mem(rcx, rdi, 2), rdx)         //
	//lea(mem(rdx, rdi, 1), rdx)         // rdx = c + 3*rs_c;
	prefetch(0, mem(rcx, 3*8))         // prefetch c + 0*rs_c
	prefetch(0, mem(rcx, rdi, 1, 3*8)) // prefetch c + 1*rs_c

	jmp(.DPOSTPFETCH)                  // jump to end of prefetching c
	label(.DCOLPFETCH)                 // column-stored prefetching c

	mov(var(cs_c), rsi)                // load cs_c to rsi (temporarily)
	lea(mem(, rsi, 8), rsi)            // cs_c *= sizeof(double)
	lea(mem(rcx, rsi, 2), rdx)         //
	lea(mem(rdx, rsi, 1), rdx)         // rdx = c + 3*cs_c;
	prefetch(0, mem(rcx, 1*8))         // prefetch c + 0*cs_c
	prefetch(0, mem(rcx, rsi, 1, 1*8)) // prefetch c + 1*cs_c
	prefetch(0, mem(rcx, rsi, 2, 1*8)) // prefetch c + 2*cs_c
	prefetch(0, mem(rdx, 1*8))         // prefetch c + 3*cs_c

	label(.DPOSTPFETCH)                // done prefetching c
#endif
	
	
	
	
	mov(var(k_iter), rsi)              // i = k_iter;
	test(rsi, rsi)                     // check i via logical AND.
	je(.DCONSIDKLEFT)                  // if i == 0, jump to code that
	                                   // contains the k_left loop.
	
	
	label(.DLOOPKITER)                 // MAIN LOOP
	
	
	// ---------------------------------- iteration 0
	
	vmovupd(mem(rbx,  0*32), ymm0)
	add(r10, rbx)                      // b += rs_b;

	vbroadcastsd(mem(rax        ), ymm2)
	vbroadcastsd(mem(rax, r8,  1), ymm3)
	add(r9, rax)                       // a += cs_a;
	vfmadd231pd(ymm0, ymm2, ymm4)
	vfmadd231pd(ymm0, ymm3, ymm6)
	
	
	// ---------------------------------- iteration 1

	vmovupd(mem(rbx,  0*32), ymm0)
	add(r10, rbx)                      // b += rs_b;

	vbroadcastsd(mem(rax        ), ymm2)
	vbroadcastsd(mem(rax, r8,  1), ymm3)
	add(r9, rax)                       // a += cs_a;
	vfmadd231pd(ymm0, ymm2, ymm4)
	vfmadd231pd(ymm0, ymm3, ymm6)
	

	// ---------------------------------- iteration 2
	
	vmovupd(mem(rbx,  0*32), ymm0)
	add(r10, rbx)                      // b += rs_b;

	vbroadcastsd(mem(rax        ), ymm2)
	vbroadcastsd(mem(rax, r8,  1), ymm3)
	add(r9, rax)                       // a += cs_a;
	vfmadd231pd(ymm0, ymm2, ymm4)
	vfmadd231pd(ymm0, ymm3, ymm6)
	

	// ---------------------------------- iteration 3

	vmovupd(mem(rbx,  0*32), ymm0)
	add(r10, rbx)                      // b += rs_b;

	vbroadcastsd(mem(rax        ), ymm2)
	vbroadcastsd(mem(rax, r8,  1), ymm3)
	add(r9, rax)                       // a += cs_a;
	vfmadd231pd(ymm0, ymm2, ymm4)
	vfmadd231pd(ymm0, ymm3, ymm6)
	
	
	
	dec(rsi)                           // i -= 1;
	jne(.DLOOPKITER)                   // iterate again if i != 0.
	
	
	
	
	
	
	label(.DCONSIDKLEFT)
	
	mov(var(k_left), rsi)              // i = k_left;
	test(rsi, rsi)                     // check i via logical AND.
	je(.DPOSTACCUM)                    // if i == 0, we're done; jump to end.
	                                   // else, we prepare to enter k_left loop.
	
	
	label(.DLOOPKLEFT)                 // EDGE LOOP
	
	vmovupd(mem(rbx,  0*32), ymm0)
	add(r10, rbx)                      // b += rs_b;

	vbroadcastsd(mem(rax        ), ymm2)
	vbroadcastsd(mem(rax, r8,  1), ymm3)
	add(r9, rax)                       // a += cs_a;
	vfmadd231pd(ymm0, ymm2, ymm4)
	vfmadd231pd(ymm0, ymm3, ymm6)
	
	
	dec(rsi)                           // i -= 1;
	jne(.DLOOPKLEFT)                   // iterate again if i != 0.
	
	
	
	label(.DPOSTACCUM)


	
	mov(var(alpha), rax)               // load address of alpha
	mov(var(beta), rbx)                // load address of beta
	vbroadcastsd(mem(rax), ymm0)       // load alpha and duplicate
	vbroadcastsd(mem(rbx), ymm3)       // load beta and duplicate
	
	vmulpd(ymm0, ymm4, ymm4)           // scale by alpha
	vmulpd(ymm0, ymm6, ymm6)
	
	
	
	
	
	
	mov(var(cs_c), rsi)                // load cs_c
	lea(mem(, rsi, 8), rsi)            // rsi = cs_c * sizeof(double)
	
	//lea(mem(rcx, rsi, 4), rdx)         // load address of c +  4*cs_c;
	//lea(mem(rcx, rdi, 4), r14)         // load address of c +  4*rs_c;

	lea(mem(rsi, rsi, 2), rax)         // rax = 3*cs_c;
	
	
	
	                                   // now avoid loading C if beta == 0
	
	vxorpd(ymm0, ymm0, ymm0)           // set ymm0 to zero.
	vucomisd(xmm0, xmm3)               // set ZF if beta == 0.
	je(.DBETAZERO)                     // if ZF = 1, jump to beta == 0 case



	cmp(imm(8), rdi)                   // set ZF if (8*rs_c) == 8.
	jz(.DCOLSTORED)                    // jump to column storage case
	

	
	label(.DROWSTORED)
	
	
	vfmadd231pd(mem(rcx), ymm3, ymm4)
	vmovupd(ymm4, mem(rcx))
	add(rdi, rcx)
	
	
	vfmadd231pd(mem(rcx), ymm3, ymm6)
	vmovupd(ymm6, mem(rcx))
	//add(rdi, rcx)
	
	
	jmp(.DDONE)                        // jump to end.



	label(.DCOLSTORED)


	vunpcklpd(ymm6, ymm4, ymm0)
	vunpckhpd(ymm6, ymm4, ymm1)
	vextractf128(imm(0x1), ymm0, xmm2)
	vextractf128(imm(0x1), ymm1, xmm4)

	vfmadd231pd(mem(rcx), xmm3, xmm0)
	vfmadd231pd(mem(rcx, rsi, 1), xmm3, xmm1)
	vfmadd231pd(mem(rcx, rsi, 2), xmm3, xmm2)
	vfmadd231pd(mem(rcx, rax, 1), xmm3, xmm4)
	vmovupd(xmm0, mem(rcx))
	vmovupd(xmm1, mem(rcx, rsi, 1))
	vmovupd(xmm2, mem(rcx, rsi, 2))
	vmovupd(xmm4, mem(rcx, rax, 1))

	//lea(mem(rcx, rsi, 4), rcx)


	jmp(.DDONE)                        // jump to end.
	
	
	
	
	label(.DBETAZERO)


	cmp(imm(8), rdi)                   // set ZF if (8*rs_c) == 8.
	jz(.DCOLSTORBZ)                    // jump to column storage case
	

	
	label(.DROWSTORBZ)
	
	
	vmovupd(ymm4, mem(rcx))
	add(rdi, rcx)
	
	vmovupd(ymm6, mem(rcx))
	//add(rdi, rcx)


	jmp(.DDONE)                        // jump to end.



	label(.DCOLSTORBZ)
	

	vunpcklpd(ymm6, ymm4, ymm0)
	vunpckhpd(ymm6, ymm4, ymm1)
	vextractf128(imm(0x1), ymm0, xmm2)
	vextractf128(imm(0x1), ymm1, xmm4)

	vmovupd(xmm0, mem(rcx))
	vmovupd(xmm1, mem(rcx, rsi, 1))
	vmovupd(xmm2, mem(rcx, rsi, 2))
	vmovupd(xmm4, mem(rcx, rax, 1))

	//lea(mem(rcx, rsi, 4), rcx)
	
	
	
	
	label(.DDONE)
	
	

    end_asm(
	: // output operands (none)
	: // input operands
      [k_iter] "m" (k_iter),
      [k_left] "m" (k_left),
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
	  "ymm0", "ymm1", "ymm2", "ymm3", "ymm4", "ymm6", "memory"
	)
	AOCL_DTL_TRACE_EXIT(AOCL_DTL_LEVEL_TRACE_7);
}

void bli_dgemmsup_rv_haswell_asm_1x4
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
	AOCL_DTL_TRACE_ENTRY(AOCL_DTL_LEVEL_TRACE_7);
	//void*    a_next = bli_auxinfo_next_a( data );
	//void*    b_next = bli_auxinfo_next_b( data );

	// Typecast local copies of integers in case dim_t and inc_t are a
	// different size than is expected by load instructions.
	uint64_t k_iter = k0 / 4;
	uint64_t k_left = k0 % 4;

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
	mov(var(cs_a), r9)                 // load cs_a
	lea(mem(, r8, 8), r8)              // rs_a *= sizeof(double)
	lea(mem(, r9, 8), r9)              // cs_a *= sizeof(double)

	//lea(mem(r8, r8, 2), r13)           // r13 = 3*rs_a
	//lea(mem(r8, r8, 4), r15)           // r15 = 5*rs_a

	mov(var(b), rbx)                   // load address of b.
	mov(var(rs_b), r10)                // load rs_b
	//mov(var(cs_b), r11)                // load cs_b
	lea(mem(, r10, 8), r10)            // rs_b *= sizeof(double)
	//lea(mem(, r11, 8), r11)            // cs_b *= sizeof(double)
	
	                                   // NOTE: We cannot pre-load elements of a or b
	                                   // because it could eventually, in the last
	                                   // unrolled iter or the cleanup loop, result
	                                   // in reading beyond the bounds allocated mem
	                                   // (the likely result: a segmentation fault).

	mov(var(c), rcx)                   // load address of c
	mov(var(rs_c), rdi)                // load rs_c
	lea(mem(, rdi, 8), rdi)            // rs_c *= sizeof(double)

#if 0
	lea(mem(rcx, rdi, 2), rdx)         //
	//lea(mem(rdx, rdi, 1), rdx)         // rdx = c + 3*rs_c;
	prefetch(0, mem(rcx, 7*8))         // prefetch c + 0*rs_c
#else
	cmp(imm(8), rdi)                   // set ZF if (8*rs_c) == 8.
	jz(.DCOLPFETCH)                    // jump to column storage case
	label(.DROWPFETCH)                 // row-stored prefetching on c

	//lea(mem(rcx, rdi, 2), rdx)         //
	//lea(mem(rdx, rdi, 1), rdx)         // rdx = c + 3*rs_c;
	prefetch(0, mem(rcx, 3*8))         // prefetch c + 0*rs_c

	jmp(.DPOSTPFETCH)                  // jump to end of prefetching c
	label(.DCOLPFETCH)                 // column-stored prefetching c

	mov(var(cs_c), rsi)                // load cs_c to rsi (temporarily)
	lea(mem(, rsi, 8), rsi)            // cs_c *= sizeof(double)
	lea(mem(rcx, rsi, 2), rdx)         //
	lea(mem(rdx, rsi, 1), rdx)         // rdx = c + 3*cs_c;
	prefetch(0, mem(rcx, 0*8))         // prefetch c + 0*cs_c
	prefetch(0, mem(rcx, rsi, 1, 0*8)) // prefetch c + 1*cs_c
	prefetch(0, mem(rcx, rsi, 2, 0*8)) // prefetch c + 2*cs_c
	prefetch(0, mem(rdx, 0*8))         // prefetch c + 3*cs_c

	label(.DPOSTPFETCH)                // done prefetching c
#endif
	
	
	
	
	mov(var(k_iter), rsi)              // i = k_iter;
	test(rsi, rsi)                     // check i via logical AND.
	je(.DCONSIDKLEFT)                  // if i == 0, jump to code that
	                                   // contains the k_left loop.
	
	
	label(.DLOOPKITER)                 // MAIN LOOP
	
	
	// ---------------------------------- iteration 0
	
	vmovupd(mem(rbx,  0*32), ymm0)
	add(r10, rbx)                      // b += rs_b;

	vbroadcastsd(mem(rax        ), ymm2)
	add(r9, rax)                       // a += cs_a;
	vfmadd231pd(ymm0, ymm2, ymm4)
	
	
	// ---------------------------------- iteration 1

	vmovupd(mem(rbx,  0*32), ymm0)
	add(r10, rbx)                      // b += rs_b;

	vbroadcastsd(mem(rax        ), ymm2)
	add(r9, rax)                       // a += cs_a;
	vfmadd231pd(ymm0, ymm2, ymm4)
	

	// ---------------------------------- iteration 2
	
	vmovupd(mem(rbx,  0*32), ymm0)
	add(r10, rbx)                      // b += rs_b;

	vbroadcastsd(mem(rax        ), ymm2)
	add(r9, rax)                       // a += cs_a;
	vfmadd231pd(ymm0, ymm2, ymm4)
	

	// ---------------------------------- iteration 3

	vmovupd(mem(rbx,  0*32), ymm0)
	add(r10, rbx)                      // b += rs_b;

	vbroadcastsd(mem(rax        ), ymm2)
	add(r9, rax)                       // a += cs_a;
	vfmadd231pd(ymm0, ymm2, ymm4)
	
	
	
	dec(rsi)                           // i -= 1;
	jne(.DLOOPKITER)                   // iterate again if i != 0.
	
	
	
	
	
	
	label(.DCONSIDKLEFT)
	
	mov(var(k_left), rsi)              // i = k_left;
	test(rsi, rsi)                     // check i via logical AND.
	je(.DPOSTACCUM)                    // if i == 0, we're done; jump to end.
	                                   // else, we prepare to enter k_left loop.
	
	
	label(.DLOOPKLEFT)                 // EDGE LOOP
	
	vmovupd(mem(rbx,  0*32), ymm0)
	add(r10, rbx)                      // b += rs_b;

	vbroadcastsd(mem(rax        ), ymm2)
	add(r9, rax)                       // a += cs_a;
	vfmadd231pd(ymm0, ymm2, ymm4)
	
	
	dec(rsi)                           // i -= 1;
	jne(.DLOOPKLEFT)                   // iterate again if i != 0.
	
	
	
	label(.DPOSTACCUM)


	
	mov(var(alpha), rax)               // load address of alpha
	mov(var(beta), rbx)                // load address of beta
	vbroadcastsd(mem(rax), ymm0)       // load alpha and duplicate
	vbroadcastsd(mem(rbx), ymm3)       // load beta and duplicate
	
	vmulpd(ymm0, ymm4, ymm4)           // scale by alpha
	
	
	
	
	
	
	mov(var(cs_c), rsi)                // load cs_c
	lea(mem(, rsi, 8), rsi)            // rsi = cs_c * sizeof(double)
	
	//lea(mem(rcx, rsi, 4), rdx)         // load address of c +  4*cs_c;
	//lea(mem(rcx, rdi, 4), r14)         // load address of c +  4*rs_c;

	lea(mem(rsi, rsi, 2), rax)         // rax = 3*cs_c;
	
	
	
	                                   // now avoid loading C if beta == 0
	
	vxorpd(ymm0, ymm0, ymm0)           // set ymm0 to zero.
	vucomisd(xmm0, xmm3)               // set ZF if beta == 0.
	je(.DBETAZERO)                     // if ZF = 1, jump to beta == 0 case


	cmp(imm(8), rdi)                   // set ZF if (8*rs_c) == 8.
	jz(.DCOLSTORED)                    // jump to column storage case
	

	
	label(.DROWSTORED)
	
	
	vfmadd231pd(mem(rcx), ymm3, ymm4)
	vmovupd(ymm4, mem(rcx))
	//add(rdi, rcx)
	
	
	jmp(.DDONE)                        // jump to end.



	label(.DCOLSTORED)


	vmovlpd(mem(rcx), xmm0, xmm0)
	vmovhpd(mem(rcx, rsi, 1), xmm0, xmm0)
	vmovlpd(mem(rcx, rsi, 2), xmm1, xmm1)
	vmovhpd(mem(rcx, rax, 1), xmm1, xmm1)
	vperm2f128(imm(0x20), ymm1, ymm0, ymm0)

	vfmadd213pd(ymm4, ymm3, ymm0)

	vextractf128(imm(1), ymm0, xmm1)
	vmovlpd(xmm0, mem(rcx))
	vmovhpd(xmm0, mem(rcx, rsi, 1))
	vmovlpd(xmm1, mem(rcx, rsi, 2))
	vmovhpd(xmm1, mem(rcx, rax, 1))
	
	//lea(mem(rcx, rsi, 4), rcx)


	jmp(.DDONE)                        // jump to end.
	
	
	
	
	label(.DBETAZERO)


	cmp(imm(8), rdi)                   // set ZF if (8*rs_c) == 8.
	jz(.DCOLSTORBZ)                    // jump to column storage case


	
	label(.DROWSTORBZ)
	
	
	vmovupd(ymm4, mem(rcx))
	//add(rdi, rcx)


	jmp(.DDONE)                        // jump to end.



	label(.DCOLSTORBZ)
	

	vmovupd(ymm4, ymm0)

	vextractf128(imm(1), ymm0, xmm1)
	vmovlpd(xmm0, mem(rcx))
	vmovhpd(xmm0, mem(rcx, rsi, 1))
	vmovlpd(xmm1, mem(rcx, rsi, 2))
	vmovhpd(xmm1, mem(rcx, rax, 1))

	//lea(mem(rcx, rsi, 4), rcx)
	
	
	
	
	
	label(.DDONE)
	
	

    end_asm(
	: // output operands (none)
	: // input operands
      [k_iter] "m" (k_iter),
      [k_left] "m" (k_left),
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
	  "ymm0", "ymm1", "ymm2", "ymm3", "ymm4", "memory"
	)
	AOCL_DTL_TRACE_EXIT(AOCL_DTL_LEVEL_TRACE_7);
}

void bli_dgemmsup_rv_haswell_asm_6x2
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
	AOCL_DTL_TRACE_ENTRY(AOCL_DTL_LEVEL_TRACE_7);
	//void*    a_next = bli_auxinfo_next_a( data );
	//void*    b_next = bli_auxinfo_next_b( data );

	// Typecast local copies of integers in case dim_t and inc_t are a
	// different size than is expected by load instructions.
	uint64_t k_iter = k0 / 4;
	uint64_t k_left = k0 % 4;

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
	mov(var(cs_a), r9)                 // load cs_a
	lea(mem(, r8, 8), r8)              // rs_a *= sizeof(double)
	lea(mem(, r9, 8), r9)              // cs_a *= sizeof(double)

	lea(mem(r8, r8, 2), r13)           // r13 = 3*rs_a
	lea(mem(r8, r8, 4), r15)           // r15 = 5*rs_a

	mov(var(b), rbx)                   // load address of b.
	mov(var(rs_b), r10)                // load rs_b
	//mov(var(cs_b), r11)                // load cs_b
	lea(mem(, r10, 8), r10)            // rs_b *= sizeof(double)
	//lea(mem(, r11, 8), r11)            // cs_b *= sizeof(double)
	
	                                   // NOTE: We cannot pre-load elements of a or b
	                                   // because it could eventually, in the last
	                                   // unrolled iter or the cleanup loop, result
	                                   // in reading beyond the bounds allocated mem
	                                   // (the likely result: a segmentation fault).

	mov(var(c), rcx)                   // load address of c
	mov(var(rs_c), rdi)                // load rs_c
	lea(mem(, rdi, 8), rdi)            // rs_c *= sizeof(double)

#if 0
	lea(mem(rcx, rdi, 2), rdx)         //
	lea(mem(rdx, rdi, 1), rdx)         // rdx = c + 3*rs_c;
	prefetch(0, mem(rcx, 7*8))         // prefetch c + 0*rs_c
	prefetch(0, mem(rcx, rdi, 1, 7*8)) // prefetch c + 1*rs_c
	prefetch(0, mem(rcx, rdi, 2, 7*8)) // prefetch c + 2*rs_c
	prefetch(0, mem(rdx, 7*8))         // prefetch c + 3*rs_c
	prefetch(0, mem(rdx, rdi, 1, 7*8)) // prefetch c + 4*rs_c
	prefetch(0, mem(rdx, rdi, 2, 7*8)) // prefetch c + 5*rs_c
#else
	cmp(imm(8), rdi)                   // set ZF if (8*rs_c) == 8.
	jz(.DCOLPFETCH)                    // jump to column storage case
	label(.DROWPFETCH)                 // row-stored prefetching on c

	lea(mem(rcx, rdi, 2), rdx)         //
	lea(mem(rdx, rdi, 1), rdx)         // rdx = c + 3*rs_c;
	prefetch(0, mem(rcx, 1*8))         // prefetch c + 0*rs_c
	prefetch(0, mem(rcx, rdi, 1, 1*8)) // prefetch c + 1*rs_c
	prefetch(0, mem(rcx, rdi, 2, 1*8)) // prefetch c + 2*rs_c
	prefetch(0, mem(rdx, 1*8))         // prefetch c + 3*rs_c
	prefetch(0, mem(rdx, rdi, 1, 1*8)) // prefetch c + 4*rs_c
	prefetch(0, mem(rdx, rdi, 2, 1*8)) // prefetch c + 5*rs_c

	jmp(.DPOSTPFETCH)                  // jump to end of prefetching c
	label(.DCOLPFETCH)                 // column-stored prefetching c

	mov(var(cs_c), rsi)                // load cs_c to rsi (temporarily)
	lea(mem(, rsi, 8), rsi)            // cs_c *= sizeof(double)
	//lea(mem(rcx, rsi, 2), rdx)         //
	//lea(mem(rdx, rsi, 1), rdx)         // rdx = c + 3*cs_c;
	prefetch(0, mem(rcx, 5*8))         // prefetch c + 0*cs_c
	prefetch(0, mem(rcx, rsi, 1, 5*8)) // prefetch c + 1*cs_c

	label(.DPOSTPFETCH)                // done prefetching c
#endif
	
	
	
	
	mov(var(k_iter), rsi)              // i = k_iter;
	test(rsi, rsi)                     // check i via logical AND.
	je(.DCONSIDKLEFT)                  // if i == 0, jump to code that
	                                   // contains the k_left loop.
	
	
	label(.DLOOPKITER)                 // MAIN LOOP
	
	
	// ---------------------------------- iteration 0
	
	vmovupd(mem(rbx,  0*32), xmm0)
	add(r10, rbx)                      // b += rs_b;

	vbroadcastsd(mem(rax        ), ymm2)
	vbroadcastsd(mem(rax, r8,  1), ymm3)
	vfmadd231pd(xmm0, xmm2, xmm4)
	vfmadd231pd(xmm0, xmm3, xmm6)
	
	vbroadcastsd(mem(rax, r8,  2), ymm2)
	vbroadcastsd(mem(rax, r13, 1), ymm3)
	vfmadd231pd(xmm0, xmm2, xmm8)
	vfmadd231pd(xmm0, xmm3, xmm10)
	
	vbroadcastsd(mem(rax, r8,  4), ymm2)
	vbroadcastsd(mem(rax, r15, 1), ymm3)
	add(r9, rax)                       // a += cs_a;
	vfmadd231pd(xmm0, xmm2, xmm12)
	vfmadd231pd(xmm0, xmm3, xmm14)

	
	// ---------------------------------- iteration 1

	vmovupd(mem(rbx,  0*32), xmm0)
	add(r10, rbx)                      // b += rs_b;

	vbroadcastsd(mem(rax        ), ymm2)
	vbroadcastsd(mem(rax, r8,  1), ymm3)
	vfmadd231pd(xmm0, xmm2, xmm4)
	vfmadd231pd(xmm0, xmm3, xmm6)
	
	vbroadcastsd(mem(rax, r8,  2), ymm2)
	vbroadcastsd(mem(rax, r13, 1), ymm3)
	vfmadd231pd(xmm0, xmm2, xmm8)
	vfmadd231pd(xmm0, xmm3, xmm10)
	
	vbroadcastsd(mem(rax, r8,  4), ymm2)
	vbroadcastsd(mem(rax, r15, 1), ymm3)
	add(r9, rax)                       // a += cs_a;
	vfmadd231pd(xmm0, xmm2, xmm12)
	vfmadd231pd(xmm0, xmm3, xmm14)
	

	// ---------------------------------- iteration 2
	
	vmovupd(mem(rbx,  0*32), xmm0)
	add(r10, rbx)                      // b += rs_b;

	vbroadcastsd(mem(rax        ), ymm2)
	vbroadcastsd(mem(rax, r8,  1), ymm3)
	vfmadd231pd(xmm0, xmm2, xmm4)
	vfmadd231pd(xmm0, xmm3, xmm6)
	
	vbroadcastsd(mem(rax, r8,  2), ymm2)
	vbroadcastsd(mem(rax, r13, 1), ymm3)
	vfmadd231pd(xmm0, xmm2, xmm8)
	vfmadd231pd(xmm0, xmm3, xmm10)
	
	vbroadcastsd(mem(rax, r8,  4), ymm2)
	vbroadcastsd(mem(rax, r15, 1), ymm3)
	add(r9, rax)                       // a += cs_a;
	vfmadd231pd(xmm0, xmm2, xmm12)
	vfmadd231pd(xmm0, xmm3, xmm14)
	

	// ---------------------------------- iteration 3

	vmovupd(mem(rbx,  0*32), xmm0)
	add(r10, rbx)                      // b += rs_b;

	vbroadcastsd(mem(rax        ), ymm2)
	vbroadcastsd(mem(rax, r8,  1), ymm3)
	vfmadd231pd(xmm0, xmm2, xmm4)
	vfmadd231pd(xmm0, xmm3, xmm6)
	
	vbroadcastsd(mem(rax, r8,  2), ymm2)
	vbroadcastsd(mem(rax, r13, 1), ymm3)
	vfmadd231pd(xmm0, xmm2, xmm8)
	vfmadd231pd(xmm0, xmm3, xmm10)
	
	vbroadcastsd(mem(rax, r8,  4), ymm2)
	vbroadcastsd(mem(rax, r15, 1), ymm3)
	add(r9, rax)                       // a += cs_a;
	vfmadd231pd(xmm0, xmm2, xmm12)
	vfmadd231pd(xmm0, xmm3, xmm14)
	
	
	
	dec(rsi)                           // i -= 1;
	jne(.DLOOPKITER)                   // iterate again if i != 0.
	
	
	
	
	
	
	label(.DCONSIDKLEFT)
	
	mov(var(k_left), rsi)              // i = k_left;
	test(rsi, rsi)                     // check i via logical AND.
	je(.DPOSTACCUM)                    // if i == 0, we're done; jump to end.
	                                   // else, we prepare to enter k_left loop.
	
	
	label(.DLOOPKLEFT)                 // EDGE LOOP
	
	vmovupd(mem(rbx,  0*32), xmm0)
	add(r10, rbx)                      // b += rs_b;

	vbroadcastsd(mem(rax        ), ymm2)
	vbroadcastsd(mem(rax, r8,  1), ymm3)
	vfmadd231pd(xmm0, xmm2, xmm4)
	vfmadd231pd(xmm0, xmm3, xmm6)
	
	vbroadcastsd(mem(rax, r8,  2), ymm2)
	vbroadcastsd(mem(rax, r13, 1), ymm3)
	vfmadd231pd(xmm0, xmm2, xmm8)
	vfmadd231pd(xmm0, xmm3, xmm10)
	
	vbroadcastsd(mem(rax, r8,  4), ymm2)
	vbroadcastsd(mem(rax, r15, 1), ymm3)
	add(r9, rax)                       // a += cs_a;
	vfmadd231pd(xmm0, xmm2, xmm12)
	vfmadd231pd(xmm0, xmm3, xmm14)
	
	
	dec(rsi)                           // i -= 1;
	jne(.DLOOPKLEFT)                   // iterate again if i != 0.
	
	
	
	label(.DPOSTACCUM)


	
	mov(var(alpha), rax)               // load address of alpha
	mov(var(beta), rbx)                // load address of beta
	vbroadcastsd(mem(rax), ymm0)       // load alpha and duplicate
	vbroadcastsd(mem(rbx), ymm3)       // load beta and duplicate
	
	vmulpd(xmm0, xmm4, xmm4)           // scale by alpha
	vmulpd(xmm0, xmm6, xmm6)
	vmulpd(xmm0, xmm8, xmm8)
	vmulpd(xmm0, xmm10, xmm10)
	vmulpd(xmm0, xmm12, xmm12)
	vmulpd(xmm0, xmm14, xmm14)
	
	
	
	
	
	
	mov(var(cs_c), rsi)                // load cs_c
	lea(mem(, rsi, 8), rsi)            // rsi = cs_c * sizeof(double)
	
	//lea(mem(rcx, rsi, 4), rdx)         // load address of c +  4*cs_c;
	lea(mem(rcx, rdi, 4), rdx)         // load address of c +  4*rs_c;

	//lea(mem(rsi, rsi, 2), rax)         // rax = 3*cs_c;
	
	
	
	                                   // now avoid loading C if beta == 0
	
	vxorpd(ymm0, ymm0, ymm0)           // set ymm0 to zero.
	vucomisd(xmm0, xmm3)               // set ZF if beta == 0.
	je(.DBETAZERO)                     // if ZF = 1, jump to beta == 0 case



	cmp(imm(8), rdi)                   // set ZF if (8*rs_c) == 8.
	jz(.DCOLSTORED)                    // jump to column storage case
	

	
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



	label(.DCOLSTORED)


	vunpcklpd(xmm6, xmm4, xmm0)
	vunpckhpd(xmm6, xmm4, xmm1)
	vunpcklpd(xmm10, xmm8, xmm2)
	vunpckhpd(xmm10, xmm8, xmm3)
	vinsertf128(imm(0x1), xmm2, ymm0, ymm4)
	vinsertf128(imm(0x1), xmm3, ymm1, ymm6)

	vbroadcastsd(mem(rbx), ymm3)

	vfmadd231pd(mem(rcx), ymm3, ymm4)
	vfmadd231pd(mem(rcx, rsi, 1), ymm3, ymm6)
	vmovupd(ymm4, mem(rcx))
	vmovupd(ymm6, mem(rcx, rsi, 1))

	//lea(mem(rcx, rsi, 4), rcx)

	vunpcklpd(xmm14, xmm12, xmm0)
	vunpckhpd(xmm14, xmm12, xmm1)

	vfmadd231pd(mem(rdx), xmm3, xmm0)
	vfmadd231pd(mem(rdx, rsi, 1), xmm3, xmm1)
	vmovupd(xmm0, mem(rdx))
	vmovupd(xmm1, mem(rdx, rsi, 1))

	//lea(mem(rdx, rsi, 4), rdx)


	jmp(.DDONE)                        // jump to end.
	
	
	
	
	label(.DBETAZERO)


	cmp(imm(8), rdi)                   // set ZF if (8*rs_c) == 8.
	jz(.DCOLSTORBZ)                    // jump to column storage case
	

	
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


	jmp(.DDONE)                        // jump to end.



	label(.DCOLSTORBZ)
	

	vunpcklpd(xmm6, xmm4, xmm0)
	vunpckhpd(xmm6, xmm4, xmm1)
	vunpcklpd(xmm10, xmm8, xmm2)
	vunpckhpd(xmm10, xmm8, xmm3)
	vinsertf128(imm(0x1), xmm2, ymm0, ymm4)
	vinsertf128(imm(0x1), xmm3, ymm1, ymm6)

	vmovupd(ymm4, mem(rcx))
	vmovupd(ymm6, mem(rcx, rsi, 1))

	//lea(mem(rcx, rsi, 4), rcx)

	vunpcklpd(xmm14, xmm12, xmm0)
	vunpckhpd(xmm14, xmm12, xmm1)

	vmovupd(xmm0, mem(rdx))
	vmovupd(xmm1, mem(rdx, rsi, 1))

	//lea(mem(rdx, rsi, 4), rdx)
	
	
	
	
	label(.DDONE)
	
	

    end_asm(
	: // output operands (none)
	: // input operands
      [k_iter] "m" (k_iter),
      [k_left] "m" (k_left),
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
	  "ymm0", "ymm1", "ymm2", "ymm3", "ymm4", "ymm6", "memory"
	)
	AOCL_DTL_TRACE_EXIT(AOCL_DTL_LEVEL_TRACE_7);
}

void bli_dgemmsup_rv_haswell_asm_5x2
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
	AOCL_DTL_TRACE_ENTRY(AOCL_DTL_LEVEL_TRACE_7);
	//void*    a_next = bli_auxinfo_next_a( data );
	//void*    b_next = bli_auxinfo_next_b( data );

	// Typecast local copies of integers in case dim_t and inc_t are a
	// different size than is expected by load instructions.
	uint64_t k_iter = k0 / 4;
	uint64_t k_left = k0 % 4;

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
	mov(var(cs_a), r9)                 // load cs_a
	lea(mem(, r8, 8), r8)              // rs_a *= sizeof(double)
	lea(mem(, r9, 8), r9)              // cs_a *= sizeof(double)

	lea(mem(r8, r8, 2), r13)           // r13 = 3*rs_a
	//lea(mem(r8, r8, 4), r15)           // r15 = 5*rs_a

	mov(var(b), rbx)                   // load address of b.
	mov(var(rs_b), r10)                // load rs_b
	//mov(var(cs_b), r11)                // load cs_b
	lea(mem(, r10, 8), r10)            // rs_b *= sizeof(double)
	//lea(mem(, r11, 8), r11)            // cs_b *= sizeof(double)
	
	                                   // NOTE: We cannot pre-load elements of a or b
	                                   // because it could eventually, in the last
	                                   // unrolled iter or the cleanup loop, result
	                                   // in reading beyond the bounds allocated mem
	                                   // (the likely result: a segmentation fault).

	mov(var(c), rcx)                   // load address of c
	mov(var(rs_c), rdi)                // load rs_c
	lea(mem(, rdi, 8), rdi)            // rs_c *= sizeof(double)

#if 0
	lea(mem(rcx, rdi, 2), rdx)         //
	lea(mem(rdx, rdi, 1), rdx)         // rdx = c + 3*rs_c;
	prefetch(0, mem(rcx, 7*8))         // prefetch c + 0*rs_c
	prefetch(0, mem(rcx, rdi, 1, 7*8)) // prefetch c + 1*rs_c
	prefetch(0, mem(rcx, rdi, 2, 7*8)) // prefetch c + 2*rs_c
	prefetch(0, mem(rdx, 7*8))         // prefetch c + 3*rs_c
	prefetch(0, mem(rdx, rdi, 1, 7*8)) // prefetch c + 4*rs_c
	prefetch(0, mem(rdx, rdi, 2, 7*8)) // prefetch c + 5*rs_c
#else
	cmp(imm(8), rdi)                   // set ZF if (8*rs_c) == 8.
	jz(.DCOLPFETCH)                    // jump to column storage case
	label(.DROWPFETCH)                 // row-stored prefetching on c

	lea(mem(rcx, rdi, 2), rdx)         //
	lea(mem(rdx, rdi, 1), rdx)         // rdx = c + 3*rs_c;
	prefetch(0, mem(rcx, 1*8))         // prefetch c + 0*rs_c
	prefetch(0, mem(rcx, rdi, 1, 1*8)) // prefetch c + 1*rs_c
	prefetch(0, mem(rcx, rdi, 2, 1*8)) // prefetch c + 2*rs_c
	prefetch(0, mem(rdx, 1*8))         // prefetch c + 3*rs_c
	prefetch(0, mem(rdx, rdi, 1, 1*8)) // prefetch c + 4*rs_c

	jmp(.DPOSTPFETCH)                  // jump to end of prefetching c
	label(.DCOLPFETCH)                 // column-stored prefetching c

	mov(var(cs_c), rsi)                // load cs_c to rsi (temporarily)
	lea(mem(, rsi, 8), rsi)            // cs_c *= sizeof(double)
	prefetch(0, mem(rcx, 4*8))         // prefetch c + 0*cs_c
	prefetch(0, mem(rcx, rsi, 1, 4*8)) // prefetch c + 1*cs_c

	label(.DPOSTPFETCH)                // done prefetching c
#endif

	
	
	
	
	mov(var(k_iter), rsi)              // i = k_iter;
	test(rsi, rsi)                     // check i via logical AND.
	je(.DCONSIDKLEFT)                  // if i == 0, jump to code that
	                                   // contains the k_left loop.
	
	
	label(.DLOOPKITER)                 // MAIN LOOP
	
	
	// ---------------------------------- iteration 0
	
	vmovupd(mem(rbx,  0*32), xmm0)
	add(r10, rbx)                      // b += rs_b;

	vbroadcastsd(mem(rax        ), ymm2)
	vbroadcastsd(mem(rax, r8,  1), ymm3)
	vfmadd231pd(xmm0, xmm2, xmm4)
	vfmadd231pd(xmm0, xmm3, xmm6)
	
	vbroadcastsd(mem(rax, r8,  2), ymm2)
	vbroadcastsd(mem(rax, r13, 1), ymm3)
	vfmadd231pd(xmm0, xmm2, xmm8)
	vfmadd231pd(xmm0, xmm3, xmm10)
	
	vbroadcastsd(mem(rax, r8,  4), ymm2)
	add(r9, rax)                       // a += cs_a;
	vfmadd231pd(xmm0, xmm2, xmm12)

	
	// ---------------------------------- iteration 1

	vmovupd(mem(rbx,  0*32), xmm0)
	add(r10, rbx)                      // b += rs_b;

	vbroadcastsd(mem(rax        ), ymm2)
	vbroadcastsd(mem(rax, r8,  1), ymm3)
	vfmadd231pd(xmm0, xmm2, xmm4)
	vfmadd231pd(xmm0, xmm3, xmm6)
	
	vbroadcastsd(mem(rax, r8,  2), ymm2)
	vbroadcastsd(mem(rax, r13, 1), ymm3)
	vfmadd231pd(xmm0, xmm2, xmm8)
	vfmadd231pd(xmm0, xmm3, xmm10)
	
	vbroadcastsd(mem(rax, r8,  4), ymm2)
	add(r9, rax)                       // a += cs_a;
	vfmadd231pd(xmm0, xmm2, xmm12)
	

	// ---------------------------------- iteration 2
	
	vmovupd(mem(rbx,  0*32), xmm0)
	add(r10, rbx)                      // b += rs_b;

	vbroadcastsd(mem(rax        ), ymm2)
	vbroadcastsd(mem(rax, r8,  1), ymm3)
	vfmadd231pd(xmm0, xmm2, xmm4)
	vfmadd231pd(xmm0, xmm3, xmm6)
	
	vbroadcastsd(mem(rax, r8,  2), ymm2)
	vbroadcastsd(mem(rax, r13, 1), ymm3)
	vfmadd231pd(xmm0, xmm2, xmm8)
	vfmadd231pd(xmm0, xmm3, xmm10)
	
	vbroadcastsd(mem(rax, r8,  4), ymm2)
	add(r9, rax)                       // a += cs_a;
	vfmadd231pd(xmm0, xmm2, xmm12)
	

	// ---------------------------------- iteration 3

	vmovupd(mem(rbx,  0*32), xmm0)
	add(r10, rbx)                      // b += rs_b;

	vbroadcastsd(mem(rax        ), ymm2)
	vbroadcastsd(mem(rax, r8,  1), ymm3)
	vfmadd231pd(xmm0, xmm2, xmm4)
	vfmadd231pd(xmm0, xmm3, xmm6)
	
	vbroadcastsd(mem(rax, r8,  2), ymm2)
	vbroadcastsd(mem(rax, r13, 1), ymm3)
	vfmadd231pd(xmm0, xmm2, xmm8)
	vfmadd231pd(xmm0, xmm3, xmm10)
	
	vbroadcastsd(mem(rax, r8,  4), ymm2)
	add(r9, rax)                       // a += cs_a;
	vfmadd231pd(xmm0, xmm2, xmm12)
	
	
	
	dec(rsi)                           // i -= 1;
	jne(.DLOOPKITER)                   // iterate again if i != 0.
	
	
	
	
	
	
	label(.DCONSIDKLEFT)
	
	mov(var(k_left), rsi)              // i = k_left;
	test(rsi, rsi)                     // check i via logical AND.
	je(.DPOSTACCUM)                    // if i == 0, we're done; jump to end.
	                                   // else, we prepare to enter k_left loop.
	
	
	label(.DLOOPKLEFT)                 // EDGE LOOP
	
	vmovupd(mem(rbx,  0*32), xmm0)
	add(r10, rbx)                      // b += rs_b;

	vbroadcastsd(mem(rax        ), ymm2)
	vbroadcastsd(mem(rax, r8,  1), ymm3)
	vfmadd231pd(xmm0, xmm2, xmm4)
	vfmadd231pd(xmm0, xmm3, xmm6)
	
	vbroadcastsd(mem(rax, r8,  2), ymm2)
	vbroadcastsd(mem(rax, r13, 1), ymm3)
	vfmadd231pd(xmm0, xmm2, xmm8)
	vfmadd231pd(xmm0, xmm3, xmm10)
	
	vbroadcastsd(mem(rax, r8,  4), ymm2)
	add(r9, rax)                       // a += cs_a;
	vfmadd231pd(xmm0, xmm2, xmm12)
	
	
	dec(rsi)                           // i -= 1;
	jne(.DLOOPKLEFT)                   // iterate again if i != 0.
	
	
	
	label(.DPOSTACCUM)


	
	mov(var(alpha), rax)               // load address of alpha
	mov(var(beta), rbx)                // load address of beta
	vbroadcastsd(mem(rax), ymm0)       // load alpha and duplicate
	vbroadcastsd(mem(rbx), ymm3)       // load beta and duplicate
	
	vmulpd(xmm0, xmm4, xmm4)           // scale by alpha
	vmulpd(xmm0, xmm6, xmm6)
	vmulpd(xmm0, xmm8, xmm8)
	vmulpd(xmm0, xmm10, xmm10)
	vmulpd(xmm0, xmm12, xmm12)
	
	
	
	
	
	
	mov(var(cs_c), rsi)                // load cs_c
	lea(mem(, rsi, 8), rsi)            // rsi = cs_c * sizeof(double)
	
	//lea(mem(rcx, rsi, 4), rdx)         // load address of c +  4*cs_c;
	lea(mem(rcx, rdi, 4), rdx)         // load address of c +  4*rs_c;

	//lea(mem(rsi, rsi, 2), rax)         // r13 = 3*cs_c;
	
	
	
	                                   // now avoid loading C if beta == 0
	
	vxorpd(ymm0, ymm0, ymm0)           // set ymm0 to zero.
	vucomisd(xmm0, xmm3)               // set ZF if beta == 0.
	je(.DBETAZERO)                     // if ZF = 1, jump to beta == 0 case



	cmp(imm(8), rdi)                   // set ZF if (8*rs_c) == 8.
	jz(.DCOLSTORED)                    // jump to column storage case
	

	
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
	//add(rdi, rcx)
	
	
	jmp(.DDONE)                        // jump to end.



	label(.DCOLSTORED)


	vunpcklpd(xmm6, xmm4, xmm0)
	vunpckhpd(xmm6, xmm4, xmm1)
	vunpcklpd(xmm10, xmm8, xmm2)
	vunpckhpd(xmm10, xmm8, xmm3)
	vinsertf128(imm(0x1), xmm2, ymm0, ymm4)
	vinsertf128(imm(0x1), xmm3, ymm1, ymm6)

	vbroadcastsd(mem(rbx), ymm3)

	vfmadd231pd(mem(rcx), ymm3, ymm4)
	vfmadd231pd(mem(rcx, rsi, 1), ymm3, ymm6)
	vmovupd(ymm4, mem(rcx))
	vmovupd(ymm6, mem(rcx, rsi, 1))

	//lea(mem(rcx, rsi, 4), rcx)

#if 0
	vunpcklpd(xmm14, xmm12, xmm0)
	vunpckhpd(xmm14, xmm12, xmm1)

	vfmadd231pd(mem(rdx), xmm3, xmm0)
	vfmadd231pd(mem(rdx, rsi, 1), xmm3, xmm1)
	vmovupd(xmm0, mem(rdx))
	vmovupd(xmm1, mem(rdx, rsi, 1))
#else
	vmovlpd(mem(rdx), xmm0, xmm0)
	vmovhpd(mem(rdx, rsi, 1), xmm0, xmm0)

	vfmadd213pd(xmm12, xmm3, xmm0)
	vmovlpd(xmm0, mem(rdx))
	vmovhpd(xmm0, mem(rdx, rsi, 1))
#endif

	//lea(mem(rdx, rsi, 4), rdx)


	jmp(.DDONE)                        // jump to end.
	
	
	
	
	label(.DBETAZERO)


	cmp(imm(8), rdi)                   // set ZF if (8*rs_c) == 8.
	jz(.DCOLSTORBZ)                    // jump to column storage case
	

	
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
	//add(rdi, rcx)


	jmp(.DDONE)                        // jump to end.



	label(.DCOLSTORBZ)
	

	vunpcklpd(xmm6, xmm4, xmm0)
	vunpckhpd(xmm6, xmm4, xmm1)
	vunpcklpd(xmm10, xmm8, xmm2)
	vunpckhpd(xmm10, xmm8, xmm3)
	vinsertf128(imm(0x1), xmm2, ymm0, ymm4)
	vinsertf128(imm(0x1), xmm3, ymm1, ymm6)

	vmovupd(ymm4, mem(rcx))
	vmovupd(ymm6, mem(rcx, rsi, 1))

	//lea(mem(rcx, rsi, 4), rcx)

#if 0
	vunpcklpd(xmm14, xmm12, xmm0)
	vunpckhpd(xmm14, xmm12, xmm1)

	vmovupd(xmm0, mem(rdx))
	vmovupd(xmm1, mem(rdx, rsi, 1))
#else
	vmovupd(xmm12, xmm0)

	vmovlpd(xmm0, mem(rdx))
	vmovhpd(xmm0, mem(rdx, rsi, 1))
#endif

	//lea(mem(rdx, rsi, 4), rdx)
	
	
	
	
	label(.DDONE)
	
	

    end_asm(
	: // output operands (none)
	: // input operands
      [k_iter] "m" (k_iter),
      [k_left] "m" (k_left),
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
	  "ymm0", "ymm1", "ymm2", "ymm3", "ymm4", "ymm6", "memory"
	)
	AOCL_DTL_TRACE_EXIT(AOCL_DTL_LEVEL_TRACE_7);
}

void bli_dgemmsup_rv_haswell_asm_4x2
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
	AOCL_DTL_TRACE_ENTRY(AOCL_DTL_LEVEL_TRACE_7);
	//void*    a_next = bli_auxinfo_next_a( data );
	//void*    b_next = bli_auxinfo_next_b( data );

	// Typecast local copies of integers in case dim_t and inc_t are a
	// different size than is expected by load instructions.
	uint64_t k_iter = k0 / 4;
	uint64_t k_left = k0 % 4;

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
	mov(var(cs_a), r9)                 // load cs_a
	lea(mem(, r8, 8), r8)              // rs_a *= sizeof(double)
	lea(mem(, r9, 8), r9)              // cs_a *= sizeof(double)

	lea(mem(r8, r8, 2), r13)           // r13 = 3*rs_a
	//lea(mem(r8, r8, 4), r15)           // r15 = 5*rs_a

	mov(var(b), rbx)                   // load address of b.
	mov(var(rs_b), r10)                // load rs_b
	//mov(var(cs_b), r11)                // load cs_b
	lea(mem(, r10, 8), r10)            // rs_b *= sizeof(double)
	//lea(mem(, r11, 8), r11)            // cs_b *= sizeof(double)
	
	                                   // NOTE: We cannot pre-load elements of a or b
	                                   // because it could eventually, in the last
	                                   // unrolled iter or the cleanup loop, result
	                                   // in reading beyond the bounds allocated mem
	                                   // (the likely result: a segmentation fault).

	mov(var(c), rcx)                   // load address of c
	mov(var(rs_c), rdi)                // load rs_c
	lea(mem(, rdi, 8), rdi)            // rs_c *= sizeof(double)

#if 0
	lea(mem(rcx, rdi, 2), rdx)         //
	lea(mem(rdx, rdi, 1), rdx)         // rdx = c + 3*rs_c;
	prefetch(0, mem(rcx, 7*8))         // prefetch c + 0*rs_c
	prefetch(0, mem(rcx, rdi, 1, 7*8)) // prefetch c + 1*rs_c
	prefetch(0, mem(rcx, rdi, 2, 7*8)) // prefetch c + 2*rs_c
	prefetch(0, mem(rdx, 7*8))         // prefetch c + 3*rs_c
#else
	cmp(imm(8), rdi)                   // set ZF if (8*rs_c) == 8.
	jz(.DCOLPFETCH)                    // jump to column storage case
	label(.DROWPFETCH)                 // row-stored prefetching on c

	lea(mem(rcx, rdi, 2), rdx)         //
	lea(mem(rdx, rdi, 1), rdx)         // rdx = c + 3*rs_c;
	prefetch(0, mem(rcx, 1*8))         // prefetch c + 0*rs_c
	prefetch(0, mem(rcx, rdi, 1, 1*8)) // prefetch c + 1*rs_c
	prefetch(0, mem(rcx, rdi, 2, 1*8)) // prefetch c + 2*rs_c
	prefetch(0, mem(rdx, 1*8))         // prefetch c + 3*rs_c

	jmp(.DPOSTPFETCH)                  // jump to end of prefetching c
	label(.DCOLPFETCH)                 // column-stored prefetching c

	mov(var(cs_c), rsi)                // load cs_c to rsi (temporarily)
	lea(mem(, rsi, 8), rsi)            // cs_c *= sizeof(double)
	lea(mem(rcx, rsi, 2), rdx)         //
	lea(mem(rdx, rsi, 1), rdx)         // rdx = c + 3*cs_c;
	prefetch(0, mem(rcx, 3*8))         // prefetch c + 0*cs_c
	prefetch(0, mem(rcx, rsi, 1, 3*8)) // prefetch c + 1*cs_c

	label(.DPOSTPFETCH)                // done prefetching c
#endif
	
	
	
	
	mov(var(k_iter), rsi)              // i = k_iter;
	test(rsi, rsi)                     // check i via logical AND.
	je(.DCONSIDKLEFT)                  // if i == 0, jump to code that
	                                   // contains the k_left loop.
	
	
	label(.DLOOPKITER)                 // MAIN LOOP
	
	
	// ---------------------------------- iteration 0
	
	vmovupd(mem(rbx,  0*32), xmm0)
	add(r10, rbx)                      // b += rs_b;

	vbroadcastsd(mem(rax        ), ymm2)
	vbroadcastsd(mem(rax, r8,  1), ymm3)
	vfmadd231pd(xmm0, xmm2, xmm4)
	vfmadd231pd(xmm0, xmm3, xmm6)
	
	vbroadcastsd(mem(rax, r8,  2), ymm2)
	vbroadcastsd(mem(rax, r13, 1), ymm3)
	add(r9, rax)                       // a += cs_a;
	vfmadd231pd(xmm0, xmm2, xmm8)
	vfmadd231pd(xmm0, xmm3, xmm10)

	
	// ---------------------------------- iteration 1

	vmovupd(mem(rbx,  0*32), xmm0)
	add(r10, rbx)                      // b += rs_b;

	vbroadcastsd(mem(rax        ), ymm2)
	vbroadcastsd(mem(rax, r8,  1), ymm3)
	vfmadd231pd(xmm0, xmm2, xmm4)
	vfmadd231pd(xmm0, xmm3, xmm6)
	
	vbroadcastsd(mem(rax, r8,  2), ymm2)
	vbroadcastsd(mem(rax, r13, 1), ymm3)
	add(r9, rax)                       // a += cs_a;
	vfmadd231pd(xmm0, xmm2, xmm8)
	vfmadd231pd(xmm0, xmm3, xmm10)
	

	// ---------------------------------- iteration 2
	
	vmovupd(mem(rbx,  0*32), xmm0)
	add(r10, rbx)                      // b += rs_b;

	vbroadcastsd(mem(rax        ), ymm2)
	vbroadcastsd(mem(rax, r8,  1), ymm3)
	vfmadd231pd(xmm0, xmm2, xmm4)
	vfmadd231pd(xmm0, xmm3, xmm6)
	
	vbroadcastsd(mem(rax, r8,  2), ymm2)
	vbroadcastsd(mem(rax, r13, 1), ymm3)
	add(r9, rax)                       // a += cs_a;
	vfmadd231pd(xmm0, xmm2, xmm8)
	vfmadd231pd(xmm0, xmm3, xmm10)
	

	// ---------------------------------- iteration 3

	vmovupd(mem(rbx,  0*32), xmm0)
	add(r10, rbx)                      // b += rs_b;

	vbroadcastsd(mem(rax        ), ymm2)
	vbroadcastsd(mem(rax, r8,  1), ymm3)
	vfmadd231pd(xmm0, xmm2, xmm4)
	vfmadd231pd(xmm0, xmm3, xmm6)
	
	vbroadcastsd(mem(rax, r8,  2), ymm2)
	vbroadcastsd(mem(rax, r13, 1), ymm3)
	add(r9, rax)                       // a += cs_a;
	vfmadd231pd(xmm0, xmm2, xmm8)
	vfmadd231pd(xmm0, xmm3, xmm10)
	
	
	
	dec(rsi)                           // i -= 1;
	jne(.DLOOPKITER)                   // iterate again if i != 0.
	
	
	
	
	
	
	label(.DCONSIDKLEFT)
	
	mov(var(k_left), rsi)              // i = k_left;
	test(rsi, rsi)                     // check i via logical AND.
	je(.DPOSTACCUM)                    // if i == 0, we're done; jump to end.
	                                   // else, we prepare to enter k_left loop.
	
	
	label(.DLOOPKLEFT)                 // EDGE LOOP
	
	vmovupd(mem(rbx,  0*32), xmm0)
	add(r10, rbx)                      // b += rs_b;
	
	vbroadcastsd(mem(rax        ), ymm2)
	vbroadcastsd(mem(rax, r8,  1), ymm3)
	vfmadd231pd(xmm0, xmm2, xmm4)
	vfmadd231pd(xmm0, xmm3, xmm6)
	
	vbroadcastsd(mem(rax, r8,  2), ymm2)
	vbroadcastsd(mem(rax, r13, 1), ymm3)
	add(r9, rax)                       // a += cs_a;
	vfmadd231pd(xmm0, xmm2, xmm8)
	vfmadd231pd(xmm0, xmm3, xmm10)
	
	
	dec(rsi)                           // i -= 1;
	jne(.DLOOPKLEFT)                   // iterate again if i != 0.
	
	
	
	label(.DPOSTACCUM)


	
	mov(var(alpha), rax)               // load address of alpha
	mov(var(beta), rbx)                // load address of beta
	vbroadcastsd(mem(rax), ymm0)       // load alpha and duplicate
	vbroadcastsd(mem(rbx), ymm3)       // load beta and duplicate
	
	vmulpd(xmm0, xmm4, xmm4)           // scale by alpha
	vmulpd(xmm0, xmm6, xmm6)
	vmulpd(xmm0, xmm8, xmm8)
	vmulpd(xmm0, xmm10, xmm10)
	
	
	
	
	
	
	mov(var(cs_c), rsi)                // load cs_c
	lea(mem(, rsi, 8), rsi)            // rsi = cs_c * sizeof(double)
	
	//lea(mem(rcx, rsi, 4), rdx)         // load address of c +  4*cs_c;
	//lea(mem(rcx, rdi, 4), r14)         // load address of c +  4*rs_c;

	//lea(mem(rsi, rsi, 2), rax)         // rax = 3*cs_c;
	
	
	
	                                   // now avoid loading C if beta == 0
	
	vxorpd(ymm0, ymm0, ymm0)           // set ymm0 to zero.
	vucomisd(xmm0, xmm3)               // set ZF if beta == 0.
	je(.DBETAZERO)                     // if ZF = 1, jump to beta == 0 case



	cmp(imm(8), rdi)                   // set ZF if (8*rs_c) == 8.
	jz(.DCOLSTORED)                    // jump to column storage case
	

	
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
	//add(rdi, rcx)
	
	
	jmp(.DDONE)                        // jump to end.



	label(.DCOLSTORED)


	vunpcklpd(xmm6, xmm4, xmm0)
	vunpckhpd(xmm6, xmm4, xmm1)
	vunpcklpd(xmm10, xmm8, xmm2)
	vunpckhpd(xmm10, xmm8, xmm3)
	vinsertf128(imm(0x1), xmm2, ymm0, ymm4)
	vinsertf128(imm(0x1), xmm3, ymm1, ymm6)

	vbroadcastsd(mem(rbx), ymm3)

	vfmadd231pd(mem(rcx), ymm3, ymm4)
	vfmadd231pd(mem(rcx, rsi, 1), ymm3, ymm6)
	vmovupd(ymm4, mem(rcx))
	vmovupd(ymm6, mem(rcx, rsi, 1))

	//lea(mem(rcx, rsi, 4), rcx)


	jmp(.DDONE)                        // jump to end.
	
	
	
	
	label(.DBETAZERO)


	cmp(imm(8), rdi)                   // set ZF if (8*rs_c) == 8.
	jz(.DCOLSTORBZ)                    // jump to column storage case
	

	
	label(.DROWSTORBZ)
	
	
	vmovupd(xmm4, mem(rcx))
	add(rdi, rcx)
	
	vmovupd(xmm6, mem(rcx))
	add(rdi, rcx)
	
	
	vmovupd(xmm8, mem(rcx))
	add(rdi, rcx)
	
	vmovupd(xmm10, mem(rcx))
	//add(rdi, rcx)


	jmp(.DDONE)                        // jump to end.



	label(.DCOLSTORBZ)
	
	
	vunpcklpd(xmm6, xmm4, xmm0)
	vunpckhpd(xmm6, xmm4, xmm1)
	vunpcklpd(xmm10, xmm8, xmm2)
	vunpckhpd(xmm10, xmm8, xmm3)
	vinsertf128(imm(0x1), xmm2, ymm0, ymm4)
	vinsertf128(imm(0x1), xmm3, ymm1, ymm6)

	vmovupd(ymm4, mem(rcx))
	vmovupd(ymm6, mem(rcx, rsi, 1))

	//lea(mem(rcx, rsi, 4), rcx)
	

	
	
	label(.DDONE)
	
	

    end_asm(
	: // output operands (none)
	: // input operands
      [k_iter] "m" (k_iter),
      [k_left] "m" (k_left),
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
	  "ymm0", "ymm1", "ymm2", "ymm3", "ymm4", "ymm6", "memory"
	)
	AOCL_DTL_TRACE_EXIT(AOCL_DTL_LEVEL_TRACE_7);
}

void bli_dgemmsup_rv_haswell_asm_3x2
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
	AOCL_DTL_TRACE_ENTRY(AOCL_DTL_LEVEL_TRACE_7);
	//void*    a_next = bli_auxinfo_next_a( data );
	//void*    b_next = bli_auxinfo_next_b( data );

	// Typecast local copies of integers in case dim_t and inc_t are a
	// different size than is expected by load instructions.
	uint64_t k_iter = k0 / 4;
	uint64_t k_left = k0 % 4;

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
	mov(var(cs_a), r9)                 // load cs_a
	lea(mem(, r8, 8), r8)              // rs_a *= sizeof(double)
	lea(mem(, r9, 8), r9)              // cs_a *= sizeof(double)

	//lea(mem(r8, r8, 2), r13)           // r13 = 3*rs_a
	//lea(mem(r8, r8, 4), r15)           // r15 = 5*rs_a

	mov(var(b), rbx)                   // load address of b.
	mov(var(rs_b), r10)                // load rs_b
	//mov(var(cs_b), r11)                // load cs_b
	lea(mem(, r10, 8), r10)            // rs_b *= sizeof(double)
	//lea(mem(, r11, 8), r11)            // cs_b *= sizeof(double)
	
	                                   // NOTE: We cannot pre-load elements of a or b
	                                   // because it could eventually, in the last
	                                   // unrolled iter or the cleanup loop, result
	                                   // in reading beyond the bounds allocated mem
	                                   // (the likely result: a segmentation fault).

	mov(var(c), rcx)                   // load address of c
	mov(var(rs_c), rdi)                // load rs_c
	lea(mem(, rdi, 8), rdi)            // rs_c *= sizeof(double)

#if 0
	lea(mem(rcx, rdi, 2), rdx)         //
	lea(mem(rdx, rdi, 1), rdx)         // rdx = c + 3*rs_c;
	prefetch(0, mem(rcx, 7*8))         // prefetch c + 0*rs_c
	prefetch(0, mem(rcx, rdi, 1, 7*8)) // prefetch c + 1*rs_c
	prefetch(0, mem(rcx, rdi, 2, 7*8)) // prefetch c + 2*rs_c
	prefetch(0, mem(rdx, 7*8))         // prefetch c + 3*rs_c
#else
	cmp(imm(8), rdi)                   // set ZF if (8*rs_c) == 8.
	jz(.DCOLPFETCH)                    // jump to column storage case
	label(.DROWPFETCH)                 // row-stored prefetching on c

	//lea(mem(rcx, rdi, 2), rdx)         //
	//lea(mem(rdx, rdi, 1), rdx)         // rdx = c + 3*rs_c;
	prefetch(0, mem(rcx, 1*8))         // prefetch c + 0*rs_c
	prefetch(0, mem(rcx, rdi, 1, 1*8)) // prefetch c + 1*rs_c
	prefetch(0, mem(rcx, rdi, 2, 1*8)) // prefetch c + 2*rs_c

	jmp(.DPOSTPFETCH)                  // jump to end of prefetching c
	label(.DCOLPFETCH)                 // column-stored prefetching c

	mov(var(cs_c), rsi)                // load cs_c to rsi (temporarily)
	lea(mem(, rsi, 8), rsi)            // cs_c *= sizeof(double)
	//lea(mem(rcx, rsi, 2), rdx)         //
	//lea(mem(rdx, rsi, 1), rdx)         // rdx = c + 3*cs_c;
	prefetch(0, mem(rcx, 2*8))         // prefetch c + 0*cs_c
	prefetch(0, mem(rcx, rsi, 1, 2*8)) // prefetch c + 1*cs_c

	label(.DPOSTPFETCH)                // done prefetching c
#endif
	
	
	
	
	mov(var(k_iter), rsi)              // i = k_iter;
	test(rsi, rsi)                     // check i via logical AND.
	je(.DCONSIDKLEFT)                  // if i == 0, jump to code that
	                                   // contains the k_left loop.
	
	
	label(.DLOOPKITER)                 // MAIN LOOP
	
	
	// ---------------------------------- iteration 0
	
	vmovupd(mem(rbx, 0*32), xmm0)
	add(r10, rbx)                      // b += rs_b;

	vbroadcastsd(mem(rax        ), ymm2)
	vbroadcastsd(mem(rax, r8,  1), ymm3)
	vfmadd231pd(xmm0, xmm2, xmm4)
	vfmadd231pd(xmm0, xmm3, xmm6)
	
	vbroadcastsd(mem(rax, r8,  2), ymm2)
	add(r9, rax)                       // a += cs_a;
	vfmadd231pd(xmm0, xmm2, xmm8)
	
	
	// ---------------------------------- iteration 1

	vmovupd(mem(rbx, 0*32), xmm0)
	add(r10, rbx)                      // b += rs_b;

	vbroadcastsd(mem(rax        ), ymm2)
	vbroadcastsd(mem(rax, r8,  1), ymm3)
	vfmadd231pd(xmm0, xmm2, xmm4)
	vfmadd231pd(xmm0, xmm3, xmm6)
	
	vbroadcastsd(mem(rax, r8,  2), ymm2)
	add(r9, rax)                       // a += cs_a;
	vfmadd231pd(xmm0, xmm2, xmm8)
	

	// ---------------------------------- iteration 2
	
	vmovupd(mem(rbx, 0*32), xmm0)
	add(r10, rbx)                      // b += rs_b;

	vbroadcastsd(mem(rax        ), ymm2)
	vbroadcastsd(mem(rax, r8,  1), ymm3)
	vfmadd231pd(xmm0, xmm2, xmm4)
	vfmadd231pd(xmm0, xmm3, xmm6)
	
	vbroadcastsd(mem(rax, r8,  2), ymm2)
	add(r9, rax)                       // a += cs_a;
	vfmadd231pd(xmm0, xmm2, xmm8)
	

	// ---------------------------------- iteration 3

	vmovupd(mem(rbx, 0*32), xmm0)
	add(r10, rbx)                      // b += rs_b;

	vbroadcastsd(mem(rax        ), ymm2)
	vbroadcastsd(mem(rax, r8,  1), ymm3)
	vfmadd231pd(xmm0, xmm2, xmm4)
	vfmadd231pd(xmm0, xmm3, xmm6)
	
	vbroadcastsd(mem(rax, r8,  2), ymm2)
	add(r9, rax)                       // a += cs_a;
	vfmadd231pd(xmm0, xmm2, xmm8)
	
	
	
	dec(rsi)                           // i -= 1;
	jne(.DLOOPKITER)                   // iterate again if i != 0.
	
	
	
	
	
	
	label(.DCONSIDKLEFT)
	
	mov(var(k_left), rsi)              // i = k_left;
	test(rsi, rsi)                     // check i via logical AND.
	je(.DPOSTACCUM)                    // if i == 0, we're done; jump to end.
	                                   // else, we prepare to enter k_left loop.
	
	
	label(.DLOOPKLEFT)                 // EDGE LOOP
	
	vmovupd(mem(rbx, 0*32), xmm0)
	add(r10, rbx)                      // b += rs_b;

	vbroadcastsd(mem(rax        ), ymm2)
	vbroadcastsd(mem(rax, r8,  1), ymm3)
	vfmadd231pd(xmm0, xmm2, xmm4)
	vfmadd231pd(xmm0, xmm3, xmm6)
	
	vbroadcastsd(mem(rax, r8,  2), ymm2)
	add(r9, rax)                       // a += cs_a;
	vfmadd231pd(xmm0, xmm2, xmm8)
	
	
	dec(rsi)                           // i -= 1;
	jne(.DLOOPKLEFT)                   // iterate again if i != 0.
	
	
	
	label(.DPOSTACCUM)


	
	mov(var(alpha), rax)               // load address of alpha
	mov(var(beta), rbx)                // load address of beta
	vbroadcastsd(mem(rax), ymm0)       // load alpha and duplicate
	vbroadcastsd(mem(rbx), ymm3)       // load beta and duplicate
	
	vmulpd(xmm0, xmm4, xmm4)           // scale by alpha
	vmulpd(xmm0, xmm6, xmm6)
	vmulpd(xmm0, xmm8, xmm8)
	
	
	
	
	
	
	mov(var(cs_c), rsi)                // load cs_c
	lea(mem(, rsi, 8), rsi)            // rsi = cs_c * sizeof(double)
	
	//lea(mem(rcx, rsi, 4), rdx)         // load address of c +  4*cs_c;
	lea(mem(rcx, rdi, 2), rdx)         // load address of c +  2*rs_c;

	lea(mem(rsi, rsi, 2), rax)         // rax = 3*cs_c;
	
	
	
	                                   // now avoid loading C if beta == 0
	
	vxorpd(ymm0, ymm0, ymm0)           // set ymm0 to zero.
	vucomisd(xmm0, xmm3)               // set ZF if beta == 0.
	je(.DBETAZERO)                     // if ZF = 1, jump to beta == 0 case
	


	cmp(imm(8), rdi)                   // set ZF if (8*rs_c) == 8.
	jz(.DCOLSTORED)                    // jump to column storage case


	
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
	


	label(.DCOLSTORED)


	vunpcklpd(ymm6, ymm4, ymm0)
	vunpckhpd(ymm6, ymm4, ymm1)
	vunpcklpd(ymm10, ymm8, ymm2)
	vunpckhpd(ymm10, ymm8, ymm3)
	vinsertf128(imm(0x1), xmm2, ymm0, ymm4)
	vinsertf128(imm(0x1), xmm3, ymm1, ymm6)
	//vperm2f128(imm(0x31), ymm2, ymm0, ymm8)
	//vperm2f128(imm(0x31), ymm3, ymm1, ymm10)

	vextractf128(imm(0x1), ymm4, xmm12)
	vextractf128(imm(0x1), ymm6, xmm13)
	//vextractf128(imm(0x1), ymm8, xmm14)
	//vextractf128(imm(0x1), ymm10, xmm15)

	vbroadcastsd(mem(rbx), ymm3)

	vfmadd231pd(mem(rcx), xmm3, xmm4)
	vfmadd231pd(mem(rcx, rsi, 1), xmm3, xmm6)
	//vfmadd231pd(mem(rcx, rsi, 2), xmm3, xmm8)
	//vfmadd231pd(mem(rcx, rax, 1), xmm3, xmm10)
	vmovupd(xmm4, mem(rcx))
	vmovupd(xmm6, mem(rcx, rsi, 1))
	//vmovupd(xmm8, mem(rcx, rsi, 2))
	//vmovupd(xmm10, mem(rcx, rax, 1))

	//lea(mem(rcx, rsi, 4), rcx)

	vfmadd231sd(mem(rdx), xmm3, xmm12)
	vfmadd231sd(mem(rdx, rsi, 1), xmm3, xmm13)
	//vfmadd231sd(mem(rdx, rsi, 2), xmm3, xmm14)
	//vfmadd231sd(mem(rdx, rax, 1), xmm3, xmm15)
	vmovsd(xmm12, mem(rdx))
	vmovsd(xmm13, mem(rdx, rsi, 1))
	//vmovsd(xmm14, mem(rdx, rsi, 2))
	//vmovsd(xmm15, mem(rdx, rax, 1))
	
	//lea(mem(rdx, rsi, 4), rdx)


	jmp(.DDONE)                        // jump to end.


	
	
	label(.DBETAZERO)
	

	cmp(imm(8), rdi)                   // set ZF if (8*rs_c) == 8.
	jz(.DCOLSTORBZ)                    // jump to column storage case


	
	label(.DROWSTORBZ)
	
	
	vmovupd(xmm4, mem(rcx))
	add(rdi, rcx)

	vmovupd(xmm6, mem(rcx))
	add(rdi, rcx)

	vmovupd(xmm8, mem(rcx))
	//add(rdi, rcx)
	
	
	jmp(.DDONE)                        // jump to end.



	label(.DCOLSTORBZ)


	vunpcklpd(ymm6, ymm4, ymm0)
	vunpckhpd(ymm6, ymm4, ymm1)
	vunpcklpd(ymm10, ymm8, ymm2)
	vunpckhpd(ymm10, ymm8, ymm3)
	vinsertf128(imm(0x1), xmm2, ymm0, ymm4)
	vinsertf128(imm(0x1), xmm3, ymm1, ymm6)
	//vperm2f128(imm(0x31), ymm2, ymm0, ymm8)
	//vperm2f128(imm(0x31), ymm3, ymm1, ymm10)

	vextractf128(imm(0x1), ymm4, xmm12)
	vextractf128(imm(0x1), ymm6, xmm13)
	//vextractf128(imm(0x1), ymm8, xmm14)
	//vextractf128(imm(0x1), ymm10, xmm15)

	vmovupd(xmm4, mem(rcx))
	vmovupd(xmm6, mem(rcx, rsi, 1))
	//vmovupd(xmm8, mem(rcx, rsi, 2))
	//vmovupd(xmm10, mem(rcx, rax, 1))

	//lea(mem(rcx, rsi, 4), rcx)

	vmovsd(xmm12, mem(rdx))
	vmovsd(xmm13, mem(rdx, rsi, 1))
	//vmovsd(xmm14, mem(rdx, rsi, 2))
	//vmovsd(xmm15, mem(rdx, rax, 1))

	//lea(mem(rdx, rsi, 4), rdx)

	
	
	
	label(.DDONE)
	
	

    end_asm(
	: // output operands (none)
	: // input operands
      [k_iter] "m" (k_iter),
      [k_left] "m" (k_left),
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
	  "ymm0", "ymm1", "ymm2", "ymm3", "ymm4", "ymm6", "ymm8", "ymm10", "memory"
	)
	AOCL_DTL_TRACE_EXIT(AOCL_DTL_LEVEL_TRACE_7);
}

void bli_dgemmsup_rv_haswell_asm_2x2
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
	AOCL_DTL_TRACE_ENTRY(AOCL_DTL_LEVEL_TRACE_7);
	//void*    a_next = bli_auxinfo_next_a( data );
	//void*    b_next = bli_auxinfo_next_b( data );

	// Typecast local copies of integers in case dim_t and inc_t are a
	// different size than is expected by load instructions.
	uint64_t k_iter = k0 / 4;
	uint64_t k_left = k0 % 4;

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
	mov(var(cs_a), r9)                 // load cs_a
	lea(mem(, r8, 8), r8)              // rs_a *= sizeof(double)
	lea(mem(, r9, 8), r9)              // cs_a *= sizeof(double)

	//lea(mem(r8, r8, 2), r13)           // r13 = 3*rs_a
	//lea(mem(r8, r8, 4), r15)           // r15 = 5*rs_a

	mov(var(b), rbx)                   // load address of b.
	mov(var(rs_b), r10)                // load rs_b
	//mov(var(cs_b), r11)                // load cs_b
	lea(mem(, r10, 8), r10)            // rs_b *= sizeof(double)
	//lea(mem(, r11, 8), r11)            // cs_b *= sizeof(double)
	
	                                   // NOTE: We cannot pre-load elements of a or b
	                                   // because it could eventually, in the last
	                                   // unrolled iter or the cleanup loop, result
	                                   // in reading beyond the bounds allocated mem
	                                   // (the likely result: a segmentation fault).

	mov(var(c), rcx)                   // load address of c
	mov(var(rs_c), rdi)                // load rs_c
	lea(mem(, rdi, 8), rdi)            // rs_c *= sizeof(double)

#if 0
	lea(mem(rcx, rdi, 2), rdx)         //
	//lea(mem(rdx, rdi, 1), rdx)         // rdx = c + 3*rs_c;
	prefetch(0, mem(rcx, 7*8))         // prefetch c + 0*rs_c
	prefetch(0, mem(rcx, rdi, 1, 7*8)) // prefetch c + 1*rs_c
#else
	cmp(imm(8), rdi)                   // set ZF if (8*rs_c) == 8.
	jz(.DCOLPFETCH)                    // jump to column storage case
	label(.DROWPFETCH)                 // row-stored prefetching on c

	//lea(mem(rcx, rdi, 2), rdx)         //
	//lea(mem(rdx, rdi, 1), rdx)         // rdx = c + 3*rs_c;
	prefetch(0, mem(rcx, 1*8))         // prefetch c + 0*rs_c
	prefetch(0, mem(rcx, rdi, 1, 1*8)) // prefetch c + 1*rs_c

	jmp(.DPOSTPFETCH)                  // jump to end of prefetching c
	label(.DCOLPFETCH)                 // column-stored prefetching c

	mov(var(cs_c), rsi)                // load cs_c to rsi (temporarily)
	lea(mem(, rsi, 8), rsi)            // cs_c *= sizeof(double)
	//lea(mem(rcx, rsi, 2), rdx)         //
	//lea(mem(rdx, rsi, 1), rdx)         // rdx = c + 3*cs_c;
	prefetch(0, mem(rcx, 1*8))         // prefetch c + 0*cs_c
	prefetch(0, mem(rcx, rsi, 1, 1*8)) // prefetch c + 1*cs_c

	label(.DPOSTPFETCH)                // done prefetching c
#endif
	
	
	
	
	mov(var(k_iter), rsi)              // i = k_iter;
	test(rsi, rsi)                     // check i via logical AND.
	je(.DCONSIDKLEFT)                  // if i == 0, jump to code that
	                                   // contains the k_left loop.
	
	
	label(.DLOOPKITER)                 // MAIN LOOP
	
	
	// ---------------------------------- iteration 0
	
	vmovupd(mem(rbx,  0*32), xmm0)
	add(r10, rbx)                      // b += rs_b;
	
	vbroadcastsd(mem(rax        ), ymm2)
	vbroadcastsd(mem(rax, r8,  1), ymm3)
	add(r9, rax)                       // a += cs_a;
	vfmadd231pd(xmm0, xmm2, xmm4)
	vfmadd231pd(xmm0, xmm3, xmm6)

	
	// ---------------------------------- iteration 1

	vmovupd(mem(rbx,  0*32), xmm0)
	add(r10, rbx)                      // b += rs_b;
	
	vbroadcastsd(mem(rax        ), ymm2)
	vbroadcastsd(mem(rax, r8,  1), ymm3)
	add(r9, rax)                       // a += cs_a;
	vfmadd231pd(xmm0, xmm2, xmm4)
	vfmadd231pd(xmm0, xmm3, xmm6)
	
	
	// ---------------------------------- iteration 2
	
	vmovupd(mem(rbx,  0*32), xmm0)
	add(r10, rbx)                      // b += rs_b;
	
	vbroadcastsd(mem(rax        ), ymm2)
	vbroadcastsd(mem(rax, r8,  1), ymm3)
	add(r9, rax)                       // a += cs_a;
	vfmadd231pd(xmm0, xmm2, xmm4)
	vfmadd231pd(xmm0, xmm3, xmm6)
	

	// ---------------------------------- iteration 3

	vmovupd(mem(rbx,  0*32), xmm0)
	add(r10, rbx)                      // b += rs_b;
	
	vbroadcastsd(mem(rax        ), ymm2)
	vbroadcastsd(mem(rax, r8,  1), ymm3)
	add(r9, rax)                       // a += cs_a;
	vfmadd231pd(xmm0, xmm2, xmm4)
	vfmadd231pd(xmm0, xmm3, xmm6)
	
	
	
	dec(rsi)                           // i -= 1;
	jne(.DLOOPKITER)                   // iterate again if i != 0.
	
	
	
	
	
	
	label(.DCONSIDKLEFT)
	
	mov(var(k_left), rsi)              // i = k_left;
	test(rsi, rsi)                     // check i via logical AND.
	je(.DPOSTACCUM)                    // if i == 0, we're done; jump to end.
	                                   // else, we prepare to enter k_left loop.
	
	
	label(.DLOOPKLEFT)                 // EDGE LOOP
	
	vmovupd(mem(rbx,  0*32), xmm0)
	add(r10, rbx)                      // b += rs_b;
	
	vbroadcastsd(mem(rax        ), ymm2)
	vbroadcastsd(mem(rax, r8,  1), ymm3)
	add(r9, rax)                       // a += cs_a;
	vfmadd231pd(xmm0, xmm2, xmm4)
	vfmadd231pd(xmm0, xmm3, xmm6)
	
	
	dec(rsi)                           // i -= 1;
	jne(.DLOOPKLEFT)                   // iterate again if i != 0.
	
	
	
	label(.DPOSTACCUM)


	
	mov(var(alpha), rax)               // load address of alpha
	mov(var(beta), rbx)                // load address of beta
	vbroadcastsd(mem(rax), ymm0)       // load alpha and duplicate
	vbroadcastsd(mem(rbx), ymm3)       // load beta and duplicate
	
	vmulpd(xmm0, xmm4, xmm4)           // scale by alpha
	vmulpd(xmm0, xmm6, xmm6)
	
	
	
	
	
	
	mov(var(cs_c), rsi)                // load cs_c
	lea(mem(, rsi, 8), rsi)            // rsi = cs_c * sizeof(double)
	
	//lea(mem(rcx, rsi, 4), rdx)         // load address of c +  4*cs_c;
	//lea(mem(rcx, rdi, 4), r14)         // load address of c +  4*rs_c;

	//lea(mem(rsi, rsi, 2), rax)         // rax = 3*cs_c;
	
	
	
	                                   // now avoid loading C if beta == 0
	
	vxorpd(ymm0, ymm0, ymm0)           // set ymm0 to zero.
	vucomisd(xmm0, xmm3)               // set ZF if beta == 0.
	je(.DBETAZERO)                     // if ZF = 1, jump to beta == 0 case



	cmp(imm(8), rdi)                   // set ZF if (8*rs_c) == 8.
	jz(.DCOLSTORED)                    // jump to column storage case
	

	
	label(.DROWSTORED)
	
	
	vfmadd231pd(mem(rcx), xmm3, xmm4)
	vmovupd(xmm4, mem(rcx))
	add(rdi, rcx)
	
	
	vfmadd231pd(mem(rcx), xmm3, xmm6)
	vmovupd(xmm6, mem(rcx))
	//add(rdi, rcx)
	
	
	jmp(.DDONE)                        // jump to end.



	label(.DCOLSTORED)


	vunpcklpd(xmm6, xmm4, xmm0)
	vunpckhpd(xmm6, xmm4, xmm1)

	vfmadd231pd(mem(rcx), xmm3, xmm0)
	vfmadd231pd(mem(rcx, rsi, 1), xmm3, xmm1)
	vmovupd(xmm0, mem(rcx))
	vmovupd(xmm1, mem(rcx, rsi, 1))

	//lea(mem(rcx, rsi, 4), rcx)


	jmp(.DDONE)                        // jump to end.
	
	
	
	label(.DBETAZERO)


	cmp(imm(8), rdi)                   // set ZF if (8*rs_c) == 8.
	jz(.DCOLSTORBZ)                    // jump to column storage case
	

	
	label(.DROWSTORBZ)
	
	
	vmovupd(xmm4, mem(rcx))
	add(rdi, rcx)
	
	vmovupd(xmm6, mem(rcx))
	//add(rdi, rcx)
	

	jmp(.DDONE)                        // jump to end.



	label(.DCOLSTORBZ)
	

	vunpcklpd(xmm6, xmm4, xmm0)
	vunpckhpd(xmm6, xmm4, xmm1)

	vmovupd(xmm0, mem(rcx))
	vmovupd(xmm1, mem(rcx, rsi, 1))

	//lea(mem(rcx, rsi, 4), rcx)
	
	
	
	
	label(.DDONE)
	
	

    end_asm(
	: // output operands (none)
	: // input operands
      [k_iter] "m" (k_iter),
      [k_left] "m" (k_left),
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
	  "ymm0", "ymm2", "ymm3", "memory"
	)
	AOCL_DTL_TRACE_EXIT(AOCL_DTL_LEVEL_TRACE_7);
}

void bli_dgemmsup_rv_haswell_asm_1x2
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
	AOCL_DTL_TRACE_ENTRY(AOCL_DTL_LEVEL_TRACE_7);
	//void*    a_next = bli_auxinfo_next_a( data );
	//void*    b_next = bli_auxinfo_next_b( data );

	// Typecast local copies of integers in case dim_t and inc_t are a
	// different size than is expected by load instructions.
	uint64_t k_iter = k0 / 4;
	uint64_t k_left = k0 % 4;

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
	mov(var(cs_a), r9)                 // load cs_a
	lea(mem(, r8, 8), r8)              // rs_a *= sizeof(double)
	lea(mem(, r9, 8), r9)              // cs_a *= sizeof(double)

	//lea(mem(r8, r8, 2), r13)           // r13 = 3*rs_a
	//lea(mem(r8, r8, 4), r15)           // r15 = 5*rs_a

	mov(var(b), rbx)                   // load address of b.
	mov(var(rs_b), r10)                // load rs_b
	//mov(var(cs_b), r11)                // load cs_b
	lea(mem(, r10, 8), r10)            // rs_b *= sizeof(double)
	//lea(mem(, r11, 8), r11)            // cs_b *= sizeof(double)
	
	                                   // NOTE: We cannot pre-load elements of a or b
	                                   // because it could eventually, in the last
	                                   // unrolled iter or the cleanup loop, result
	                                   // in reading beyond the bounds allocated mem
	                                   // (the likely result: a segmentation fault).

	mov(var(c), rcx)                   // load address of c
	mov(var(rs_c), rdi)                // load rs_c
	lea(mem(, rdi, 8), rdi)            // rs_c *= sizeof(double)

#if 0
	lea(mem(rcx, rdi, 2), rdx)         //
	//lea(mem(rdx, rdi, 1), rdx)         // rdx = c + 3*rs_c;
	prefetch(0, mem(rcx, 7*8))         // prefetch c + 0*rs_c
#else
	cmp(imm(8), rdi)                   // set ZF if (8*rs_c) == 8.
	jz(.DCOLPFETCH)                    // jump to column storage case
	label(.DROWPFETCH)                 // row-stored prefetching on c

	//lea(mem(rcx, rdi, 2), rdx)         //
	//lea(mem(rdx, rdi, 1), rdx)         // rdx = c + 3*rs_c;
	prefetch(0, mem(rcx, 1*8))         // prefetch c + 0*rs_c

	jmp(.DPOSTPFETCH)                  // jump to end of prefetching c
	label(.DCOLPFETCH)                 // column-stored prefetching c

	mov(var(cs_c), rsi)                // load cs_c to rsi (temporarily)
	lea(mem(, rsi, 8), rsi)            // cs_c *= sizeof(double)
	//lea(mem(rcx, rsi, 2), rdx)         //
	//lea(mem(rdx, rsi, 1), rdx)         // rdx = c + 3*cs_c;
	prefetch(0, mem(rcx, 0*8))         // prefetch c + 0*cs_c
	prefetch(0, mem(rcx, rsi, 1, 0*8)) // prefetch c + 1*cs_c

	label(.DPOSTPFETCH)                // done prefetching c
#endif
	
	
	
	
	mov(var(k_iter), rsi)              // i = k_iter;
	test(rsi, rsi)                     // check i via logical AND.
	je(.DCONSIDKLEFT)                  // if i == 0, jump to code that
	                                   // contains the k_left loop.
	
	
	label(.DLOOPKITER)                 // MAIN LOOP
	
	
	// ---------------------------------- iteration 0
	
	vmovupd(mem(rbx,  0*32), xmm0)
	add(r10, rbx)                      // b += rs_b;

	vbroadcastsd(mem(rax        ), ymm2)
	add(r9, rax)                       // a += cs_a;
	vfmadd231pd(xmm0, xmm2, xmm4)

	
	// ---------------------------------- iteration 1

	vmovupd(mem(rbx,  0*32), xmm0)
	add(r10, rbx)                      // b += rs_b;

	vbroadcastsd(mem(rax        ), ymm2)
	add(r9, rax)                       // a += cs_a;
	vfmadd231pd(xmm0, xmm2, xmm4)
	
	
	// ---------------------------------- iteration 2
	
	vmovupd(mem(rbx,  0*32), xmm0)
	add(r10, rbx)                      // b += rs_b;

	vbroadcastsd(mem(rax        ), ymm2)
	add(r9, rax)                       // a += cs_a;
	vfmadd231pd(xmm0, xmm2, xmm4)
	

	// ---------------------------------- iteration 3

	vmovupd(mem(rbx,  0*32), xmm0)
	add(r10, rbx)                      // b += rs_b;

	vbroadcastsd(mem(rax        ), ymm2)
	add(r9, rax)                       // a += cs_a;
	vfmadd231pd(xmm0, xmm2, xmm4)
	
	
	
	dec(rsi)                           // i -= 1;
	jne(.DLOOPKITER)                   // iterate again if i != 0.
	
	
	
	
	
	
	label(.DCONSIDKLEFT)
	
	mov(var(k_left), rsi)              // i = k_left;
	test(rsi, rsi)                     // check i via logical AND.
	je(.DPOSTACCUM)                    // if i == 0, we're done; jump to end.
	                                   // else, we prepare to enter k_left loop.
	
	
	label(.DLOOPKLEFT)                 // EDGE LOOP
	
	vmovupd(mem(rbx,  0*32), xmm0)
	add(r10, rbx)                      // b += rs_b;

	vbroadcastsd(mem(rax        ), ymm2)
	add(r9, rax)                       // a += cs_a;
	vfmadd231pd(xmm0, xmm2, xmm4)
	
	
	dec(rsi)                           // i -= 1;
	jne(.DLOOPKLEFT)                   // iterate again if i != 0.
	
	
	
	label(.DPOSTACCUM)


	
	mov(var(alpha), rax)               // load address of alpha
	mov(var(beta), rbx)                // load address of beta
	vbroadcastsd(mem(rax), ymm0)       // load alpha and duplicate
	vbroadcastsd(mem(rbx), ymm3)       // load beta and duplicate
	
	vmulpd(xmm0, xmm4, xmm4)           // scale by alpha
	
	
	
	
	
	
	mov(var(cs_c), rsi)                // load cs_c
	lea(mem(, rsi, 8), rsi)            // rsi = cs_c * sizeof(double)
	
	//lea(mem(rcx, rsi, 4), rdx)         // load address of c +  4*cs_c;
	//lea(mem(rcx, rdi, 4), r14)         // load address of c +  4*rs_c;

	//lea(mem(rsi, rsi, 2), rax)         // rax = 3*cs_c;
	
	
	
	                                   // now avoid loading C if beta == 0
	
	vxorpd(ymm0, ymm0, ymm0)           // set ymm0 to zero.
	vucomisd(xmm0, xmm3)               // set ZF if beta == 0.
	je(.DBETAZERO)                     // if ZF = 1, jump to beta == 0 case
	


	cmp(imm(8), rdi)                   // set ZF if (8*rs_c) == 8.
	jz(.DCOLSTORED)                    // jump to column storage case


	
	label(.DROWSTORED)
	
	
	vfmadd231pd(mem(rcx), xmm3, xmm4)
	vmovupd(xmm4, mem(rcx))
	//add(rdi, rcx)
	
	
	jmp(.DDONE)                        // jump to end.



	label(.DCOLSTORED)


	vmovlpd(mem(rcx), xmm0, xmm0)
	vmovhpd(mem(rcx, rsi, 1), xmm0, xmm0)

	vfmadd213pd(xmm4, xmm3, xmm0)

	vmovlpd(xmm0, mem(rcx))
	vmovhpd(xmm0, mem(rcx, rsi, 1))
	
	//lea(mem(rcx, rsi, 4), rcx)


	jmp(.DDONE)                        // jump to end.
	
	
	
	
	label(.DBETAZERO)


	cmp(imm(8), rdi)                   // set ZF if (8*rs_c) == 8.
	jz(.DCOLSTORBZ)                    // jump to column storage case
	

	
	label(.DROWSTORBZ)
	
	
	vmovupd(xmm4, mem(rcx))
	//add(rdi, rcx)


	jmp(.DDONE)                        // jump to end.
	


	label(.DCOLSTORBZ)
	

	vmovlpd(xmm4, mem(rcx))
	vmovhpd(xmm4, mem(rcx, rsi, 1))

	//lea(mem(rcx, rsi, 4), rcx)
	
	
	
	
	label(.DDONE)
	
	

    end_asm(
	: // output operands (none)
	: // input operands
      [k_iter] "m" (k_iter),
      [k_left] "m" (k_left),
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
	  "ymm0", "ymm2", "ymm3", "memory"
	)
	AOCL_DTL_TRACE_EXIT(AOCL_DTL_LEVEL_TRACE_7);
}

// -----------------------------------------------------------------------------

// NOTE: Normally, for any "?x1" kernel, we would call the reference kernel.
// However, at least one other subconfiguration (zen) uses this kernel set, so
// we need to be able to call a set of "?x1" kernels that we know will actually
// exist regardless of which subconfiguration these kernels were used by. Thus,
// the compromise employed here is to inline the reference kernel so it gets
// compiled as part of the haswell kernel set, and hence can unconditionally be
// called by other kernels within that kernel set.

#undef  GENTFUNC
#define GENTFUNC( ctype, ch, opname, mdim ) \
\
void PASTEMAC(ch,opname) \
     ( \
       conj_t              conja, \
       conj_t              conjb, \
       dim_t               m, \
       dim_t               n, \
       dim_t               k, \
       ctype*     restrict alpha, \
       ctype*     restrict a, inc_t rs_a, inc_t cs_a, \
       ctype*     restrict b, inc_t rs_b, inc_t cs_b, \
       ctype*     restrict beta, \
       ctype*     restrict c, inc_t rs_c, inc_t cs_c, \
       auxinfo_t* restrict data, \
       cntx_t*    restrict cntx \
     ) \
{ \
	AOCL_DTL_TRACE_ENTRY(AOCL_DTL_LEVEL_TRACE_7); \
	for ( dim_t i = 0; i < mdim; ++i ) \
	{ \
		ctype* restrict ci = &c[ i*rs_c ]; \
		ctype* restrict ai = &a[ i*rs_a ]; \
\
		/* for ( dim_t j = 0; j < 1; ++j ) */ \
		{ \
			ctype* restrict cij = ci /*[ j*cs_c ]*/ ; \
			ctype* restrict bj  = b  /*[ j*cs_b ]*/ ; \
			ctype           ab; \
\
			PASTEMAC(ch,set0s)( ab ); \
\
			/* Perform a dot product to update the (i,j) element of c. */ \
			for ( dim_t l = 0; l < k; ++l ) \
			{ \
				ctype* restrict aij = &ai[ l*cs_a ]; \
				ctype* restrict bij = &bj[ l*rs_b ]; \
\
				PASTEMAC(ch,dots)( *aij, *bij, ab ); \
			} \
\
			/* If beta is one, add ab into c. If beta is zero, overwrite c
			   with the result in ab. Otherwise, scale by beta and accumulate
			   ab to c. */ \
			if ( PASTEMAC(ch,eq1)( *beta ) ) \
			{ \
				PASTEMAC(ch,axpys)( *alpha, ab, *cij ); \
			} \
			else if ( PASTEMAC(d,eq0)( *beta ) ) \
			{ \
				PASTEMAC(ch,scal2s)( *alpha, ab, *cij ); \
			} \
			else \
			{ \
				PASTEMAC(ch,axpbys)( *alpha, ab, *beta, *cij ); \
			} \
		} \
	} \
	AOCL_DTL_TRACE_EXIT(AOCL_DTL_LEVEL_TRACE_7); \
}

GENTFUNC( double, d, gemmsup_r_haswell_ref_6x1, 6 )
GENTFUNC( double, d, gemmsup_r_haswell_ref_5x1, 5 )
GENTFUNC( double, d, gemmsup_r_haswell_ref_4x1, 4 )
GENTFUNC( double, d, gemmsup_r_haswell_ref_3x1, 3 )
GENTFUNC( double, d, gemmsup_r_haswell_ref_2x1, 2 )
GENTFUNC( double, d, gemmsup_r_haswell_ref_1x1, 1 )

