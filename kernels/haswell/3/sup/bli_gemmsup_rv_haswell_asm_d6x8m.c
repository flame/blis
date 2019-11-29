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

#if 0
// Define parameters and variables for edge case kernel map.
#define NUM_MR 4
#define NUM_NR 4
#define FUNCPTR_T dgemmsup_ker_ft

static dim_t mrs[NUM_MR] = { 6, 4, 2, 1 };
static dim_t nrs[NUM_NR] = { 8, 4, 2, 1 };
static FUNCPTR_T kmap[NUM_MR][NUM_NR] =
{     /*  8                                4                                2                                1  */
/* 6 */ { bli_dgemmsup_rv_haswell_asm_6x8m, bli_dgemmsup_rv_haswell_asm_6x4m, bli_dgemmsup_rv_haswell_asm_6x2m, bli_dgemmsup_r_haswell_ref_6x1 },
/* 4 */	{ bli_dgemmsup_rv_haswell_asm_4x8m, bli_dgemmsup_rv_haswell_asm_4x4m, bli_dgemmsup_rv_haswell_asm_4x2m, bli_dgemmsup_r_haswell_ref_4x1 },
/* 2 */	{ bli_dgemmsup_rv_haswell_asm_2x8m, bli_dgemmsup_rv_haswell_asm_2x4m, bli_dgemmsup_rv_haswell_asm_2x2m, bli_dgemmsup_r_haswell_ref_2x1 },
/* 1 */	{ bli_dgemmsup_rv_haswell_asm_1x8m, bli_dgemmsup_rv_haswell_asm_1x4m, bli_dgemmsup_rv_haswell_asm_1x2m, bli_dgemmsup_r_haswell_ref_1x1 },
};
#endif


void bli_dgemmsup_rv_haswell_asm_6x8m
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
	uint64_t n_left = n0 % 8;

	// First check whether this is a edge case in the n dimension. If so,
	// dispatch other 6x?m kernels, as needed.
	if ( n_left )
	{
		double* restrict cij = c;
		double* restrict bj  = b;
		double* restrict ai  = a;

		if ( 6 <= n_left )
		{
			const dim_t nr_cur = 6;

			bli_dgemmsup_rv_haswell_asm_6x6m
			(
			  conja, conjb, m0, nr_cur, k0,
			  alpha, ai, rs_a0, cs_a0, bj, rs_b0, cs_b0,
			  beta, cij, rs_c0, cs_c0, data, cntx
			);
			cij += nr_cur*cs_c0; bj += nr_cur*cs_b0; n_left -= nr_cur;
		}
		if ( 4 <= n_left )
		{
			const dim_t nr_cur = 4;

			bli_dgemmsup_rv_haswell_asm_6x4m
			(
			  conja, conjb, m0, nr_cur, k0,
			  alpha, ai, rs_a0, cs_a0, bj, rs_b0, cs_b0,
			  beta, cij, rs_c0, cs_c0, data, cntx
			);
			cij += nr_cur*cs_c0; bj += nr_cur*cs_b0; n_left -= nr_cur;
		}
		if ( 2 <= n_left )
		{
			const dim_t nr_cur = 2;

			bli_dgemmsup_rv_haswell_asm_6x2m
			(
			  conja, conjb, m0, nr_cur, k0,
			  alpha, ai, rs_a0, cs_a0, bj, rs_b0, cs_b0,
			  beta, cij, rs_c0, cs_c0, data, cntx
			);
			cij += nr_cur*cs_c0; bj += nr_cur*cs_b0; n_left -= nr_cur;
		}
		if ( 1 == n_left )
		{
#if 0
			const dim_t nr_cur = 1;

			bli_dgemmsup_r_haswell_ref
			(
			  conja, conjb, m0, nr_cur, k0,
			  alpha, ai, rs_a0, cs_a0, bj, rs_b0, cs_b0,
			  beta, cij, rs_c0, cs_c0, data, cntx
			);
#else
			dim_t ps_a0 = bli_auxinfo_ps_a( data );

			if ( ps_a0 == 6 * rs_a0 )
			{
				// Since A is not packed, we can use one gemv.
				bli_dgemv_ex
				(
				  BLIS_NO_TRANSPOSE, conjb, m0, k0,
				  alpha, ai, rs_a0, cs_a0, bj, rs_b0,
				  beta, cij, rs_c0, cntx, NULL
				);
			}
			else
			{
				const dim_t mr = 6;

				// Since A is packed into row panels, we must use a loop over
				// gemv.
				dim_t m_iter = ( m0 + mr - 1 ) / mr;
				dim_t m_left =   m0            % mr;

				double* restrict ai_ii  = ai;
				double* restrict cij_ii = cij;

				for ( dim_t ii = 0; ii < m_iter; ii += 1 )
				{
					dim_t mr_cur = ( bli_is_not_edge_f( ii, m_iter, m_left )
					                 ? mr : m_left );

					bli_dgemv_ex
					(
					  BLIS_NO_TRANSPOSE, conjb, mr_cur, k0,
					  alpha, ai_ii, rs_a0, cs_a0, bj, rs_b0,
					  beta, cij_ii, rs_c0, cntx, NULL
					);
					cij_ii += mr*rs_c0; ai_ii += ps_a0;
				}
			}
#endif
		}
		return;
	}

	//void*    a_next = bli_auxinfo_next_a( data );
	//void*    b_next = bli_auxinfo_next_b( data );

	// Typecast local copies of integers in case dim_t and inc_t are a
	// different size than is expected by load instructions.
	uint64_t k_iter = k0 / 4;
	uint64_t k_left = k0 % 4;

	uint64_t m_iter = m0 / 6;
	uint64_t m_left = m0 % 6;

	uint64_t rs_a   = rs_a0;
	uint64_t cs_a   = cs_a0;
	uint64_t rs_b   = rs_b0;
	uint64_t cs_b   = cs_b0;
	uint64_t rs_c   = rs_c0;
	uint64_t cs_c   = cs_c0;

	// Query the panel stride of A and convert it to units of bytes.
	uint64_t ps_a   = bli_auxinfo_ps_a( data );
	uint64_t ps_a8  = ps_a * sizeof( double );

	if ( m_iter == 0 ) goto consider_edge_cases;

	// -------------------------------------------------------------------------

	begin_asm()
	
	//vzeroall()                         // zero all xmm/ymm registers.
	
	mov(var(a), r14)                   // load address of a.
	mov(var(rs_a), r8)                 // load rs_a
	mov(var(cs_a), r9)                 // load cs_a
	lea(mem(, r8, 8), r8)              // rs_a *= sizeof(double)
	lea(mem(, r9, 8), r9)              // cs_a *= sizeof(double)
	
	lea(mem(r8, r8, 2), r13)           // r13 = 3*rs_a
	lea(mem(r8, r8, 4), r15)           // r15 = 5*rs_a

	//mov(var(b), rbx)                   // load address of b.
	mov(var(rs_b), r10)                // load rs_b
	//mov(var(cs_b), r11)                // load cs_b
	lea(mem(, r10, 8), r10)            // rs_b *= sizeof(double)
	//lea(mem(, r11, 8), r11)            // cs_b *= sizeof(double)

	                                   // NOTE: We cannot pre-load elements of a or b
	                                   // because it could eventually, in the last
	                                   // unrolled iter or the cleanup loop, result
	                                   // in reading beyond the bounds allocated mem
	                                   // (the likely result: a segmentation fault).

	mov(var(c), r12)                   // load address of c
	mov(var(rs_c), rdi)                // load rs_c
	lea(mem(, rdi, 8), rdi)            // rs_c *= sizeof(double)


	// During preamble and loops:
	// r12 = rcx = c
	// r14 = rax = a
	// read rbx from var(b) near beginning of loop
	// r11 = m dim index ii

	mov(var(m_iter), r11)              // ii = m_iter;

	label(.DLOOP6X8I)                  // LOOP OVER ii = [ m_iter ... 1 0 ]



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

	mov(var(b), rbx)                   // load address of b.
	//mov(r12, rcx)                      // reset rcx to current utile of c.
	mov(r14, rax)                      // reset rax to current upanel of a.



	cmp(imm(8), rdi)                   // set ZF if (8*rs_c) == 8.
	jz(.DCOLPFETCH)                    // jump to column storage case
	label(.DROWPFETCH)                 // row-stored prefetching on c

	lea(mem(r12, rdi, 2), rdx)         //
	lea(mem(rdx, rdi, 1), rdx)         // rdx = c + 3*rs_c;
	prefetch(0, mem(r12, 7*8))         // prefetch c + 0*rs_c
	prefetch(0, mem(r12, rdi, 1, 7*8)) // prefetch c + 1*rs_c
	prefetch(0, mem(r12, rdi, 2, 7*8)) // prefetch c + 2*rs_c
	prefetch(0, mem(rdx, 7*8))         // prefetch c + 3*rs_c
	prefetch(0, mem(rdx, rdi, 1, 7*8)) // prefetch c + 4*rs_c
	prefetch(0, mem(rdx, rdi, 2, 7*8)) // prefetch c + 5*rs_c

	jmp(.DPOSTPFETCH)                  // jump to end of prefetching c
	label(.DCOLPFETCH)                 // column-stored prefetching c

	mov(var(cs_c), rsi)                // load cs_c to rsi (temporarily)
	lea(mem(, rsi, 8), rsi)            // cs_c *= sizeof(double)
	lea(mem(r12, rsi, 2), rdx)         //
	lea(mem(rdx, rsi, 1), rdx)         // rdx = c + 3*cs_c;
	prefetch(0, mem(r12, 5*8))         // prefetch c + 0*cs_c
	prefetch(0, mem(r12, rsi, 1, 5*8)) // prefetch c + 1*cs_c
	prefetch(0, mem(r12, rsi, 2, 5*8)) // prefetch c + 2*cs_c
	prefetch(0, mem(rdx, 5*8))         // prefetch c + 3*cs_c
	prefetch(0, mem(rdx, rsi, 1, 5*8)) // prefetch c + 4*cs_c
	prefetch(0, mem(rdx, rsi, 2, 5*8)) // prefetch c + 5*cs_c
	lea(mem(rdx, rsi, 2), rdx)         // rdx = c + 5*cs_c;
	prefetch(0, mem(rdx, rsi, 1, 5*8)) // prefetch c + 6*cs_c
	prefetch(0, mem(rdx, rsi, 2, 5*8)) // prefetch c + 7*cs_c

	label(.DPOSTPFETCH)                // done prefetching c


#if 1
	lea(mem(r9, r9, 2), rcx)           // rcx = 3*cs_a;

	lea(mem(rax, r8,  4), rdx)         // use rdx for prefetching lines
	lea(mem(rdx, r8,  2), rdx)         // from next upanel of a.
#else
	lea(mem(rax, r9,  8), rdx)         // use rdx for prefetching a.
	lea(mem(rdx, r9,  8), rdx)         // rdx = a + 16*cs_a;
	//mov(r9, rsi)                       // rsi = cs_a;
	//sal(imm(4), rsi)                   // rsi = 16*cs_a;
	//lea(mem(rax, rsi, 1), rdx)         // rdx = a + 16*cs_a;
#endif
	
	
	
	
	mov(var(k_iter), rsi)              // i = k_iter;
	test(rsi, rsi)                     // check i via logical AND.
	je(.DCONSIDKLEFT)                  // if i == 0, jump to code that
	                                   // contains the k_left loop.
	
	
	label(.DLOOPKITER)                 // MAIN LOOP
	
	
	// ---------------------------------- iteration 0

#if 1
	prefetch(0, mem(rdx, 5*8))
	//prefetch(0, mem(rax, 5*8))
#else
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

#if 1
	prefetch(0, mem(rdx, r9, 1, 5*8))
	//prefetch(0, mem(rax, 5*8))
#else
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
	//prefetch(0, mem(rax, 5*8))
#else
	prefetch(0, mem(rdx, r9, 2, 5*8))
	//prefetch(0, mem(rdx, r9, 2))
	//lea(mem(rdx, r9,  4), rdx)         // rdx += 4*cs_a;
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
	prefetch(0, mem(rdx, rcx, 1, 5*8))
	lea(mem(rdx, r9,  4), rdx)         // a_prefetch += 4*cs_a;
	//prefetch(0, mem(rax, 5*8))
#else
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

	
	
	mov(r12, rcx)                      // reset rcx to current utile of c.
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




	lea(mem(r12, rdi, 4), r12)         //
	lea(mem(r12, rdi, 2), r12)         // c_ii = r12 += 6*rs_c

	//lea(mem(r14, r8,  4), r14)         //
	//lea(mem(r14, r8,  2), r14)         // a_ii = r14 += 6*rs_a
	mov(var(ps_a8), rax)               // load ps_a8
	lea(mem(r14, rax, 1), r14)         // a_ii = r14 += ps_a8

	dec(r11)                           // ii -= 1;
	jne(.DLOOP6X8I)                    // iterate again if ii != 0.




	label(.DRETURN)
	
	

    end_asm(
	: // output operands (none)
	: // input operands
      [m_iter] "m" (m_iter),
      [k_iter] "m" (k_iter),
      [k_left] "m" (k_left),
      [a]      "m" (a),
      [rs_a]   "m" (rs_a),
      [cs_a]   "m" (cs_a),
      [ps_a8]  "m" (ps_a8),
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
	  "memory"
	)

	consider_edge_cases:

	// Handle edge cases in the m dimension, if they exist.
	if ( m_left )
	{
		const dim_t      nr_cur = 8;
		const dim_t      i_edge = m0 - ( dim_t )m_left;

		double* restrict cij = c + i_edge*rs_c;
		//double* restrict ai  = a + i_edge*rs_a;
		//double* restrict ai  = a + ( i_edge / 6 ) * ps_a;
		double* restrict ai  = a + m_iter * ps_a;
		double* restrict bj  = b;

#if 0
		// We add special handling for slightly inflated MR blocksizes
		// at edge cases, up to a maximum of 9.
		if ( 6 < m_left )
		{
			dgemmsup_ker_ft ker_fp1 = NULL;
			dgemmsup_ker_ft ker_fp2 = NULL;
			dim_t           mr1, mr2;

			if ( m_left == 7 )
			{
				mr1 = 4; mr2 = 3;
				ker_fp1 = bli_dgemmsup_rv_haswell_asm_4x8;
				ker_fp2 = bli_dgemmsup_rv_haswell_asm_3x8;
			}
			else if ( m_left == 8 )
			{
				mr1 = 4; mr2 = 4;
				ker_fp1 = bli_dgemmsup_rv_haswell_asm_4x8;
				ker_fp2 = bli_dgemmsup_rv_haswell_asm_4x8;
			}
			else // if ( m_left == 9 )
			{
				mr1 = 4; mr2 = 5;
				ker_fp1 = bli_dgemmsup_rv_haswell_asm_4x8;
				ker_fp2 = bli_dgemmsup_rv_haswell_asm_5x8;
			}

			ker_fp1
			(
			  conja, conjb, mr1, nr_cur, k0,
			  alpha, ai, rs_a0, cs_a0, bj, rs_b0, cs_b0,
			  beta, cij, rs_c0, cs_c0, data, cntx
			);
			cij += mr1*rs_c0; ai += mr1*rs_a0;

			ker_fp2
			(
			  conja, conjb, mr2, nr_cur, k0,
			  alpha, ai, rs_a0, cs_a0, bj, rs_b0, cs_b0,
			  beta, cij, rs_c0, cs_c0, data, cntx
			);

			return;
		}
#endif

#if 1
		dgemmsup_ker_ft ker_fps[6] =
		{
		  NULL,
		  bli_dgemmsup_rv_haswell_asm_1x8,
		  bli_dgemmsup_rv_haswell_asm_2x8,
		  bli_dgemmsup_rv_haswell_asm_3x8,
		  bli_dgemmsup_rv_haswell_asm_4x8,
		  bli_dgemmsup_rv_haswell_asm_5x8
		};

		dgemmsup_ker_ft ker_fp = ker_fps[ m_left ];

		ker_fp
		(
		  conja, conjb, m_left, nr_cur, k0,
		  alpha, ai, rs_a0, cs_a0, bj, rs_b0, cs_b0,
		  beta, cij, rs_c0, cs_c0, data, cntx
		);

		return;
#else
		if ( 5 <= m_left )
		{
			const dim_t mr_cur = 5;

			bli_dgemmsup_rv_haswell_asm_5x8
			(
			  conja, conjb, mr_cur, nr_cur, k0,
			  alpha, ai, rs_a0, cs_a0, bj, rs_b0, cs_b0,
			  beta, cij, rs_c0, cs_c0, data, cntx
			);
			cij += mr_cur*rs_c0; ai += mr_cur*rs_a0; m_left -= mr_cur;
		}
		if ( 4 <= m_left )
		{
			const dim_t mr_cur = 4;

			bli_dgemmsup_rv_haswell_asm_4x8
			(
			  conja, conjb, mr_cur, nr_cur, k0,
			  alpha, ai, rs_a0, cs_a0, bj, rs_b0, cs_b0,
			  beta, cij, rs_c0, cs_c0, data, cntx
			);
			cij += mr_cur*rs_c0; ai += mr_cur*rs_a0; m_left -= mr_cur;
		}
		if ( 3 <= m_left )
		{
			const dim_t mr_cur = 3;

			bli_dgemmsup_rv_haswell_asm_3x8
			(
			  conja, conjb, mr_cur, nr_cur, k0,
			  alpha, ai, rs_a0, cs_a0, bj, rs_b0, cs_b0,
			  beta, cij, rs_c0, cs_c0, data, cntx
			);
			cij += mr_cur*rs_c0; ai += mr_cur*rs_a0; m_left -= mr_cur;
		}
		if ( 2 <= m_left )
		{
			const dim_t mr_cur = 2;

			bli_dgemmsup_rv_haswell_asm_2x8
			(
			  conja, conjb, mr_cur, nr_cur, k0,
			  alpha, ai, rs_a0, cs_a0, bj, rs_b0, cs_b0,
			  beta, cij, rs_c0, cs_c0, data, cntx
			);
			cij += mr_cur*rs_c0; ai += mr_cur*rs_a0; m_left -= mr_cur;
		}
		if ( 1 == m_left )
		{
			const dim_t mr_cur = 1;

			bli_dgemmsup_rv_haswell_asm_1x8
			(
			  conja, conjb, mr_cur, nr_cur, k0,
			  alpha, ai, rs_a0, cs_a0, bj, rs_b0, cs_b0,
			  beta, cij, rs_c0, cs_c0, data, cntx
			);
		}
#endif
	}
}

void bli_dgemmsup_rv_haswell_asm_6x6m
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
	uint64_t k_iter = k0 / 4;
	uint64_t k_left = k0 % 4;

	uint64_t m_iter = m0 / 6;
	uint64_t m_left = m0 % 6;

	uint64_t rs_a   = rs_a0;
	uint64_t cs_a   = cs_a0;
	uint64_t rs_b   = rs_b0;
	uint64_t cs_b   = cs_b0;
	uint64_t rs_c   = rs_c0;
	uint64_t cs_c   = cs_c0;

	// Query the panel stride of A and convert it to units of bytes.
	uint64_t ps_a   = bli_auxinfo_ps_a( data );
	uint64_t ps_a8  = ps_a * sizeof( double );

	if ( m_iter == 0 ) goto consider_edge_cases;

	// -------------------------------------------------------------------------

	begin_asm()
	
	//vzeroall()                         // zero all xmm/ymm registers.
	
	mov(var(a), r14)                   // load address of a.
	mov(var(rs_a), r8)                 // load rs_a
	mov(var(cs_a), r9)                 // load cs_a
	lea(mem(, r8, 8), r8)              // rs_a *= sizeof(double)
	lea(mem(, r9, 8), r9)              // cs_a *= sizeof(double)
	
	lea(mem(r8, r8, 2), r13)           // r13 = 3*rs_a
	lea(mem(r8, r8, 4), r15)           // r15 = 5*rs_a

	//mov(var(b), rbx)                   // load address of b.
	mov(var(rs_b), r10)                // load rs_b
	//mov(var(cs_b), r11)                // load cs_b
	lea(mem(, r10, 8), r10)            // rs_b *= sizeof(double)
	//lea(mem(, r11, 8), r11)            // cs_b *= sizeof(double)

	                                   // NOTE: We cannot pre-load elements of a or b
	                                   // because it could eventually, in the last
	                                   // unrolled iter or the cleanup loop, result
	                                   // in reading beyond the bounds allocated mem
	                                   // (the likely result: a segmentation fault).

	mov(var(c), r12)                   // load address of c
	mov(var(rs_c), rdi)                // load rs_c
	lea(mem(, rdi, 8), rdi)            // rs_c *= sizeof(double)


	// During preamble and loops:
	// r12 = rcx = c
	// r14 = rax = a
	// read rbx from var(b) near beginning of loop
	// r11 = m dim index ii

	mov(var(m_iter), r11)              // ii = m_iter;

	label(.DLOOP6X8I)                  // LOOP OVER ii = [ m_iter ... 1 0 ]



#if 0
	vzeroall()                         // zero all xmm/ymm registers.
#else
	                                   // skylake can execute 3 vxorpd ipc with
	                                   // a latency of 1 cycle, while vzeroall
	                                   // has a latency of 12 cycles.
	vxorpd(ymm1,  ymm1,  ymm1)         // zero ymm1 since we only use the lower
	vxorpd(ymm4,  ymm4,  ymm4)         // half (xmm1), and nans/infs may slow us
	vxorpd(ymm5,  ymm5,  ymm5)         // down.
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

	mov(var(b), rbx)                   // load address of b.
	//mov(r12, rcx)                      // reset rcx to current utile of c.
	mov(r14, rax)                      // reset rax to current upanel of a.



	cmp(imm(8), rdi)                   // set ZF if (8*rs_c) == 8.
	jz(.DCOLPFETCH)                    // jump to column storage case
	label(.DROWPFETCH)                 // row-stored prefetching on c

	lea(mem(r12, rdi, 2), rdx)         //
	lea(mem(rdx, rdi, 1), rdx)         // rdx = c + 3*rs_c;
	prefetch(0, mem(r12, 5*8))         // prefetch c + 0*rs_c
	prefetch(0, mem(r12, rdi, 1, 5*8)) // prefetch c + 1*rs_c
	prefetch(0, mem(r12, rdi, 2, 5*8)) // prefetch c + 2*rs_c
	prefetch(0, mem(rdx, 5*8))         // prefetch c + 3*rs_c
	prefetch(0, mem(rdx, rdi, 1, 5*8)) // prefetch c + 4*rs_c
	prefetch(0, mem(rdx, rdi, 2, 5*8)) // prefetch c + 5*rs_c

	jmp(.DPOSTPFETCH)                  // jump to end of prefetching c
	label(.DCOLPFETCH)                 // column-stored prefetching c

	mov(var(cs_c), rsi)                // load cs_c to rsi (temporarily)
	lea(mem(, rsi, 8), rsi)            // cs_c *= sizeof(double)
	lea(mem(r12, rsi, 2), rdx)         //
	lea(mem(rdx, rsi, 1), rdx)         // rdx = c + 3*cs_c;
	prefetch(0, mem(r12, 5*8))         // prefetch c + 0*cs_c
	prefetch(0, mem(r12, rsi, 1, 5*8)) // prefetch c + 1*cs_c
	prefetch(0, mem(r12, rsi, 2, 5*8)) // prefetch c + 2*cs_c
	prefetch(0, mem(rdx, 5*8))         // prefetch c + 3*cs_c
	prefetch(0, mem(rdx, rsi, 1, 5*8)) // prefetch c + 4*cs_c
	prefetch(0, mem(rdx, rsi, 2, 5*8)) // prefetch c + 5*cs_c

	label(.DPOSTPFETCH)                // done prefetching c


#if 1
	lea(mem(r9, r9, 2), rcx)           // rcx = 3*cs_a;

	lea(mem(rax, r8,  4), rdx)         // use rdx for prefetching lines
	lea(mem(rdx, r8,  2), rdx)         // from next upanel of a.
#else
	lea(mem(rax, r9,  8), rdx)         // use rdx for prefetching a.
	lea(mem(rdx, r9,  8), rdx)         // rdx = a + 16*cs_a;
	//mov(r9, rsi)                       // rsi = cs_a;
	//sal(imm(4), rsi)                   // rsi = 16*cs_a;
	//lea(mem(rax, rsi, 1), rdx)         // rdx = a + 16*cs_a;
#endif
	
	
	
	
	mov(var(k_iter), rsi)              // i = k_iter;
	test(rsi, rsi)                     // check i via logical AND.
	je(.DCONSIDKLEFT)                  // if i == 0, jump to code that
	                                   // contains the k_left loop.
	
	
	label(.DLOOPKITER)                 // MAIN LOOP
	
	
	// ---------------------------------- iteration 0

#if 1
	prefetch(0, mem(rdx, 5*8))
	//prefetch(0, mem(rax, 5*8))
#else
	prefetch(0, mem(rdx, 5*8))
#endif
	
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

#if 1
	prefetch(0, mem(rdx, r9, 1, 5*8))
	//prefetch(0, mem(rax, 5*8))
#else
#endif

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

#if 1
	prefetch(0, mem(rdx, r9, 2, 5*8))
	//prefetch(0, mem(rax, 5*8))
#else
	prefetch(0, mem(rdx, r9, 2, 5*8))
	//prefetch(0, mem(rdx, r9, 2))
	//lea(mem(rdx, r9,  4), rdx)         // rdx += 4*cs_a;
#endif
	
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

#if 1
	prefetch(0, mem(rdx, rcx, 1, 5*8))
	lea(mem(rdx, r9,  4), rdx)         // a_prefetch += 4*cs_a;
	//prefetch(0, mem(rax, 5*8))
#else
	lea(mem(rdx, r9,  4), rdx)         // a_prefetch += 4*cs_a;
#endif

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

	
	
	mov(r12, rcx)                      // reset rcx to current utile of c.
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




	lea(mem(r12, rdi, 4), r12)         //
	lea(mem(r12, rdi, 2), r12)         // c_ii = r12 += 6*rs_c

	//lea(mem(r14, r8,  4), r14)         //
	//lea(mem(r14, r8,  2), r14)         // a_ii = r14 += 6*rs_a
	mov(var(ps_a8), rax)               // load ps_a8
	lea(mem(r14, rax, 1), r14)         // a_ii = r14 += ps_a8

	dec(r11)                           // ii -= 1;
	jne(.DLOOP6X8I)                    // iterate again if ii != 0.




	label(.DRETURN)
	
	

    end_asm(
	: // output operands (none)
	: // input operands
      [m_iter] "m" (m_iter),
      [k_iter] "m" (k_iter),
      [k_left] "m" (k_left),
      [a]      "m" (a),
      [rs_a]   "m" (rs_a),
      [cs_a]   "m" (cs_a),
      [ps_a8]  "m" (ps_a8),
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
	  "memory"
	)

	consider_edge_cases:

	// Handle edge cases in the m dimension, if they exist.
	if ( m_left )
	{
		const dim_t      nr_cur = 6;
		const dim_t      i_edge = m0 - ( dim_t )m_left;

		double* restrict cij = c + i_edge*rs_c;
		//double* restrict ai  = a + i_edge*rs_a;
		//double* restrict ai  = a + ( i_edge / 6 ) * ps_a;
		double* restrict ai  = a + m_iter * ps_a;
		double* restrict bj  = b;

#if 0
		// We add special handling for slightly inflated MR blocksizes
		// at edge cases, up to a maximum of 9.
		if ( 6 < m_left )
		{
			dgemmsup_ker_ft ker_fp1 = NULL;
			dgemmsup_ker_ft ker_fp2 = NULL;
			dim_t           mr1, mr2;

			if ( m_left == 7 )
			{
				mr1 = 4; mr2 = 3;
				ker_fp1 = bli_dgemmsup_rv_haswell_asm_4x6;
				ker_fp2 = bli_dgemmsup_rv_haswell_asm_3x6;
			}
			else if ( m_left == 8 )
			{
				mr1 = 4; mr2 = 4;
				ker_fp1 = bli_dgemmsup_rv_haswell_asm_4x6;
				ker_fp2 = bli_dgemmsup_rv_haswell_asm_4x6;
			}
			else // if ( m_left == 9 )
			{
				mr1 = 4; mr2 = 5;
				ker_fp1 = bli_dgemmsup_rv_haswell_asm_4x6;
				ker_fp2 = bli_dgemmsup_rv_haswell_asm_5x6;
			}

			ker_fp1
			(
			  conja, conjb, mr1, nr_cur, k0,
			  alpha, ai, rs_a0, cs_a0, bj, rs_b0, cs_b0,
			  beta, cij, rs_c0, cs_c0, data, cntx
			);
			cij += mr1*rs_c0; ai += mr1*rs_a0;

			ker_fp2
			(
			  conja, conjb, mr2, nr_cur, k0,
			  alpha, ai, rs_a0, cs_a0, bj, rs_b0, cs_b0,
			  beta, cij, rs_c0, cs_c0, data, cntx
			);

			return;
		}
#endif

#if 1
		dgemmsup_ker_ft ker_fps[6] =
		{
		  NULL,
		  bli_dgemmsup_rv_haswell_asm_1x6,
		  bli_dgemmsup_rv_haswell_asm_2x6,
		  bli_dgemmsup_rv_haswell_asm_3x6,
		  bli_dgemmsup_rv_haswell_asm_4x6,
		  bli_dgemmsup_rv_haswell_asm_5x6
		};

		dgemmsup_ker_ft ker_fp = ker_fps[ m_left ];

		ker_fp
		(
		  conja, conjb, m_left, nr_cur, k0,
		  alpha, ai, rs_a0, cs_a0, bj, rs_b0, cs_b0,
		  beta, cij, rs_c0, cs_c0, data, cntx
		);

		return;
#else
		if ( 5 <= m_left )
		{
			const dim_t mr_cur = 5;

			bli_dgemmsup_rv_haswell_asm_5x6
			(
			  conja, conjb, mr_cur, nr_cur, k0,
			  alpha, ai, rs_a0, cs_a0, bj, rs_b0, cs_b0,
			  beta, cij, rs_c0, cs_c0, data, cntx
			);
			cij += mr_cur*rs_c0; ai += mr_cur*rs_a0; m_left -= mr_cur;
		}
		if ( 4 <= m_left )
		{
			const dim_t mr_cur = 4;

			bli_dgemmsup_rv_haswell_asm_4x6
			(
			  conja, conjb, mr_cur, nr_cur, k0,
			  alpha, ai, rs_a0, cs_a0, bj, rs_b0, cs_b0,
			  beta, cij, rs_c0, cs_c0, data, cntx
			);
			cij += mr_cur*rs_c0; ai += mr_cur*rs_a0; m_left -= mr_cur;
		}
		if ( 3 <= m_left )
		{
			const dim_t mr_cur = 3;

			bli_dgemmsup_rv_haswell_asm_3x6
			(
			  conja, conjb, mr_cur, nr_cur, k0,
			  alpha, ai, rs_a0, cs_a0, bj, rs_b0, cs_b0,
			  beta, cij, rs_c0, cs_c0, data, cntx
			);
			cij += mr_cur*rs_c0; ai += mr_cur*rs_a0; m_left -= mr_cur;
		}
		if ( 2 <= m_left )
		{
			const dim_t mr_cur = 2;

			bli_dgemmsup_rv_haswell_asm_2x6
			(
			  conja, conjb, mr_cur, nr_cur, k0,
			  alpha, ai, rs_a0, cs_a0, bj, rs_b0, cs_b0,
			  beta, cij, rs_c0, cs_c0, data, cntx
			);
			cij += mr_cur*rs_c0; ai += mr_cur*rs_a0; m_left -= mr_cur;
		}
		if ( 1 == m_left )
		{
			const dim_t mr_cur = 1;

			bli_dgemmsup_rv_haswell_asm_1x6
			(
			  conja, conjb, mr_cur, nr_cur, k0,
			  alpha, ai, rs_a0, cs_a0, bj, rs_b0, cs_b0,
			  beta, cij, rs_c0, cs_c0, data, cntx
			);
		}
#endif
	}
}

void bli_dgemmsup_rv_haswell_asm_6x4m
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
	uint64_t k_iter = k0 / 4;
	uint64_t k_left = k0 % 4;

	uint64_t m_iter = m0 / 6;
	uint64_t m_left = m0 % 6;

	uint64_t rs_a   = rs_a0;
	uint64_t cs_a   = cs_a0;
	uint64_t rs_b   = rs_b0;
	uint64_t cs_b   = cs_b0;
	uint64_t rs_c   = rs_c0;
	uint64_t cs_c   = cs_c0;

	// Query the panel stride of A and convert it to units of bytes.
	uint64_t ps_a   = bli_auxinfo_ps_a( data );
	uint64_t ps_a8  = ps_a * sizeof( double );

	if ( m_iter == 0 ) goto consider_edge_cases;

	// -------------------------------------------------------------------------

	begin_asm()
	
	//vzeroall()                         // zero all xmm/ymm registers.
	
	mov(var(a), r14)                   // load address of a.
	mov(var(rs_a), r8)                 // load rs_a
	mov(var(cs_a), r9)                 // load cs_a
	lea(mem(, r8, 8), r8)              // rs_a *= sizeof(double)
	lea(mem(, r9, 8), r9)              // cs_a *= sizeof(double)

	lea(mem(r8, r8, 2), r13)           // r13 = 3*rs_a
	lea(mem(r8, r8, 4), r15)           // r15 = 5*rs_a

	//mov(var(b), rbx)                   // load address of b.
	mov(var(rs_b), r10)                // load rs_b
	//mov(var(cs_b), r11)                // load cs_b
	lea(mem(, r10, 8), r10)            // rs_b *= sizeof(double)
	//lea(mem(, r11, 8), r11)            // cs_b *= sizeof(double)
	
	                                   // NOTE: We cannot pre-load elements of a or b
	                                   // because it could eventually, in the last
	                                   // unrolled iter or the cleanup loop, result
	                                   // in reading beyond the bounds allocated mem
	                                   // (the likely result: a segmentation fault).

	mov(var(c), r12)                   // load address of c
	mov(var(rs_c), rdi)                // load rs_c
	lea(mem(, rdi, 8), rdi)            // rs_c *= sizeof(double)


	// During preamble and loops:
	// r12 = rcx = c
	// r14 = rax = a
	// read rbx from var(b) near beginning of loop
	// r11 = m dim index ii

	mov(var(m_iter), r11)              // ii = m_iter;

	label(.DLOOP6X4I)                  // LOOP OVER ii = [ m_iter ... 1 0 ]


#if 0
	vzeroall()                         // zero all xmm/ymm registers.
#else
	                                   // skylake can execute 3 vxorpd ipc with
	                                   // a latency of 1 cycle, while vzeroall
	                                   // has a latency of 12 cycles.
	vxorpd(ymm4,  ymm4,  ymm4)
	vxorpd(ymm6,  ymm6,  ymm6)
	vxorpd(ymm8,  ymm8,  ymm8)
	vxorpd(ymm10, ymm10, ymm10)
	vxorpd(ymm12, ymm12, ymm12)
	vxorpd(ymm14, ymm14, ymm14)
#endif

	mov(var(b), rbx)                   // load address of b.
	//mov(r12, rcx)                      // reset rcx to current utile of c.
	mov(r14, rax)


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

	lea(mem(r12, rdi, 2), rdx)         //
	lea(mem(rdx, rdi, 1), rdx)         // rdx = c + 3*rs_c;
	prefetch(0, mem(r12, 3*8))         // prefetch c + 0*rs_c
	prefetch(0, mem(r12, rdi, 1, 3*8)) // prefetch c + 1*rs_c
	prefetch(0, mem(r12, rdi, 2, 3*8)) // prefetch c + 2*rs_c
	prefetch(0, mem(rdx, 3*8))         // prefetch c + 3*rs_c
	prefetch(0, mem(rdx, rdi, 1, 3*8)) // prefetch c + 4*rs_c
	prefetch(0, mem(rdx, rdi, 2, 3*8)) // prefetch c + 5*rs_c

	jmp(.DPOSTPFETCH)                  // jump to end of prefetching c
	label(.DCOLPFETCH)                 // column-stored prefetching c

	mov(var(cs_c), rsi)                // load cs_c to rsi (temporarily)
	lea(mem(, rsi, 8), rsi)            // cs_c *= sizeof(double)
	lea(mem(r12, rsi, 2), rdx)         //
	lea(mem(rdx, rsi, 1), rdx)         // rdx = c + 3*cs_c;
	prefetch(0, mem(r12, 5*8))         // prefetch c + 0*cs_c
	prefetch(0, mem(r12, rsi, 1, 5*8)) // prefetch c + 1*cs_c
	prefetch(0, mem(r12, rsi, 2, 5*8)) // prefetch c + 2*cs_c
	prefetch(0, mem(rdx, 5*8))         // prefetch c + 3*cs_c

	label(.DPOSTPFETCH)                // done prefetching c
#endif



#if 1
	lea(mem(r9, r9, 2), rcx)           // rcx = 3*cs_a;

	lea(mem(rax, r8,  4), rdx)         // use rdx for prefetching lines
	lea(mem(rdx, r8,  2), rdx)         // from next upanel of a.

	//lea(mem(rax, r9,  8), rdx)         // use rdx for prefetching a.
	//lea(mem(rdx, r9,  8), rdx)         // rdx = a + 16*cs_a;
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

#if 1
	prefetch(0, mem(rdx, r9, 1, 5*8))
#endif

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

#if 1
	prefetch(0, mem(rdx, r9, 2, 5*8))
#endif
	
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

#if 1
	prefetch(0, mem(rdx, rcx, 1, 5*8))
	lea(mem(rdx, r9,  4), rdx)         // rdx += 4*cs_a;
#endif

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
	
#if 1
	prefetch(0, mem(rdx, 5*8))
	add(r9, rdx)                       // rdx += cs_a;
#endif

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


	
	mov(r12, rcx)                      // reset rcx to current utile of c.
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



	
	lea(mem(r12, rdi, 4), r12)         //
	lea(mem(r12, rdi, 2), r12)         // c_ii = r12 += 6*rs_c

	//lea(mem(r14, r8,  4), r14)         //
	//lea(mem(r14, r8,  2), r14)         // a_ii = r14 += 6*rs_a
	mov(var(ps_a8), rax)               // load ps_a8
	lea(mem(r14, rax, 1), r14)         // a_ii = r14 += ps_a8

	dec(r11)                           // ii -= 1;
	jne(.DLOOP6X4I)                    // iterate again if ii != 0.




	label(.DRETURN)
	
	

    end_asm(
	: // output operands (none)
	: // input operands
      [m_iter] "m" (m_iter),
      [k_iter] "m" (k_iter),
      [k_left] "m" (k_left),
      [a]      "m" (a),
      [rs_a]   "m" (rs_a),
      [cs_a]   "m" (cs_a),
      [ps_a8]  "m" (ps_a8),
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
	  "memory"
	)

	consider_edge_cases:

	// Handle edge cases in the m dimension, if they exist.
	if ( m_left )
	{
		const dim_t      nr_cur = 4;
		const dim_t      i_edge = m0 - ( dim_t )m_left;

		double* restrict cij = c + i_edge*rs_c;
		//double* restrict ai  = a + i_edge*rs_a;
		//double* restrict ai  = a + ( i_edge / 6 ) * ps_a;
		double* restrict ai  = a + m_iter * ps_a;
		double* restrict bj  = b;

#if 0
		// We add special handling for slightly inflated MR blocksizes
		// at edge cases, up to a maximum of 9.
		if ( 6 < m_left )
		{
			dgemmsup_ker_ft ker_fp1 = NULL;
			dgemmsup_ker_ft ker_fp2 = NULL;
			dim_t           mr1, mr2;

			if ( m_left == 7 )
			{
				mr1 = 4; mr2 = 3;
				ker_fp1 = bli_dgemmsup_rv_haswell_asm_4x4;
				ker_fp2 = bli_dgemmsup_rv_haswell_asm_3x4;
			}
			else if ( m_left == 8 )
			{
				mr1 = 4; mr2 = 4;
				ker_fp1 = bli_dgemmsup_rv_haswell_asm_4x4;
				ker_fp2 = bli_dgemmsup_rv_haswell_asm_4x4;
			}
			else // if ( m_left == 9 )
			{
				mr1 = 4; mr2 = 5;
				ker_fp1 = bli_dgemmsup_rv_haswell_asm_4x4;
				ker_fp2 = bli_dgemmsup_rv_haswell_asm_5x4;
			}

			ker_fp1
			(
			  conja, conjb, mr1, nr_cur, k0,
			  alpha, ai, rs_a0, cs_a0, bj, rs_b0, cs_b0,
			  beta, cij, rs_c0, cs_c0, data, cntx
			);
			cij += mr1*rs_c0; ai += mr1*rs_a0;

			ker_fp2
			(
			  conja, conjb, mr2, nr_cur, k0,
			  alpha, ai, rs_a0, cs_a0, bj, rs_b0, cs_b0,
			  beta, cij, rs_c0, cs_c0, data, cntx
			);

			return;
		}
#endif

#if 1
		dgemmsup_ker_ft ker_fps[6] =
		{
		  NULL,
		  bli_dgemmsup_rv_haswell_asm_1x4,
		  bli_dgemmsup_rv_haswell_asm_2x4,
		  bli_dgemmsup_rv_haswell_asm_3x4,
		  bli_dgemmsup_rv_haswell_asm_4x4,
		  bli_dgemmsup_rv_haswell_asm_5x4
		};

		dgemmsup_ker_ft ker_fp = ker_fps[ m_left ];

		ker_fp
		(
		  conja, conjb, m_left, nr_cur, k0,
		  alpha, ai, rs_a0, cs_a0, bj, rs_b0, cs_b0,
		  beta, cij, rs_c0, cs_c0, data, cntx
		);

		return;
#else
		if ( 5 <= m_left )
		{
			const dim_t mr_cur = 5;

			bli_dgemmsup_rv_haswell_asm_5x4
			(
			  conja, conjb, mr_cur, nr_cur, k0,
			  alpha, ai, rs_a0, cs_a0, bj, rs_b0, cs_b0,
			  beta, cij, rs_c0, cs_c0, data, cntx
			);
			cij += mr_cur*rs_c0; ai += mr_cur*rs_a0; m_left -= mr_cur;
		}
		if ( 4 <= m_left )
		{
			const dim_t mr_cur = 4;

			bli_dgemmsup_rv_haswell_asm_4x4
			(
			  conja, conjb, mr_cur, nr_cur, k0,
			  alpha, ai, rs_a0, cs_a0, bj, rs_b0, cs_b0,
			  beta, cij, rs_c0, cs_c0, data, cntx
			);
			cij += mr_cur*rs_c0; ai += mr_cur*rs_a0; m_left -= mr_cur;
		}
		if ( 3 <= m_left )
		{
			const dim_t mr_cur = 3;

			bli_dgemmsup_rv_haswell_asm_3x4
			(
			  conja, conjb, mr_cur, nr_cur, k0,
			  alpha, ai, rs_a0, cs_a0, bj, rs_b0, cs_b0,
			  beta, cij, rs_c0, cs_c0, data, cntx
			);
			cij += mr_cur*rs_c0; ai += mr_cur*rs_a0; m_left -= mr_cur;
		}
		if ( 2 <= m_left )
		{
			const dim_t mr_cur = 2;

			bli_dgemmsup_rv_haswell_asm_2x4
			(
			  conja, conjb, mr_cur, nr_cur, k0,
			  alpha, ai, rs_a0, cs_a0, bj, rs_b0, cs_b0,
			  beta, cij, rs_c0, cs_c0, data, cntx
			);
			cij += mr_cur*rs_c0; ai += mr_cur*rs_a0; m_left -= mr_cur;
		}
		if ( 1 == m_left )
		{
			const dim_t mr_cur = 1;

			bli_dgemmsup_rv_haswell_asm_1x4
			(
			  conja, conjb, mr_cur, nr_cur, k0,
			  alpha, ai, rs_a0, cs_a0, bj, rs_b0, cs_b0,
			  beta, cij, rs_c0, cs_c0, data, cntx
			);
		}
#endif
	}
}

void bli_dgemmsup_rv_haswell_asm_6x2m
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
	uint64_t k_iter = k0 / 4;
	uint64_t k_left = k0 % 4;

	uint64_t m_iter = m0 / 6;
	uint64_t m_left = m0 % 6;

	uint64_t rs_a   = rs_a0;
	uint64_t cs_a   = cs_a0;
	uint64_t rs_b   = rs_b0;
	uint64_t cs_b   = cs_b0;
	uint64_t rs_c   = rs_c0;
	uint64_t cs_c   = cs_c0;

	// Query the panel stride of A and convert it to units of bytes.
	uint64_t ps_a   = bli_auxinfo_ps_a( data );
	uint64_t ps_a8  = ps_a * sizeof( double );

	if ( m_iter == 0 ) goto consider_edge_cases;

	// -------------------------------------------------------------------------

	begin_asm()
	
	//vzeroall()                         // zero all xmm/ymm registers.
	
	mov(var(a), r14)                   // load address of a.
	mov(var(rs_a), r8)                 // load rs_a
	mov(var(cs_a), r9)                 // load cs_a
	lea(mem(, r8, 8), r8)              // rs_a *= sizeof(double)
	lea(mem(, r9, 8), r9)              // cs_a *= sizeof(double)

	lea(mem(r8, r8, 2), r13)           // r13 = 3*rs_a
	lea(mem(r8, r8, 4), r15)           // r15 = 5*rs_a

	//mov(var(b), rbx)                   // load address of b.
	mov(var(rs_b), r10)                // load rs_b
	//mov(var(cs_b), r11)                // load cs_b
	lea(mem(, r10, 8), r10)            // rs_b *= sizeof(double)
	//lea(mem(, r11, 8), r11)            // cs_b *= sizeof(double)
	
	                                   // NOTE: We cannot pre-load elements of a or b
	                                   // because it could eventually, in the last
	                                   // unrolled iter or the cleanup loop, result
	                                   // in reading beyond the bounds allocated mem
	                                   // (the likely result: a segmentation fault).

	mov(var(c), r12)                   // load address of c
	mov(var(rs_c), rdi)                // load rs_c
	lea(mem(, rdi, 8), rdi)            // rs_c *= sizeof(double)


	// During preamble and loops:
	// r12 = rcx = c
	// r14 = rax = a
	// read rbx from var(b) near beginning of loop
	// r11 = m dim index ii

	mov(var(m_iter), r11)              // ii = m_iter;

	label(.DLOOP6X2I)                  // LOOP OVER ii = [ m_iter ... 1 0 ]


#if 0
	vzeroall()                         // zero all xmm/ymm registers.
#else
	                                   // skylake can execute 3 vxorpd ipc with
	                                   // a latency of 1 cycle, while vzeroall
	                                   // has a latency of 12 cycles.
	vxorpd(xmm4,  xmm4,  xmm4)
	vxorpd(xmm6,  xmm6,  xmm6)
	vxorpd(xmm8,  xmm8,  xmm8)
	vxorpd(xmm10, xmm10, xmm10)
	vxorpd(xmm12, xmm12, xmm12)
	vxorpd(xmm14, xmm14, xmm14)
#endif

	mov(var(b), rbx)                   // load address of b.
	//mov(r12, rcx)                      // reset rcx to current utile of c.
	mov(r14, rax)


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

	lea(mem(r12, rdi, 2), rdx)         //
	lea(mem(rdx, rdi, 1), rdx)         // rdx = c + 3*rs_c;
	prefetch(0, mem(r12, 1*8))         // prefetch c + 0*rs_c
	prefetch(0, mem(r12, rdi, 1, 1*8)) // prefetch c + 1*rs_c
	prefetch(0, mem(r12, rdi, 2, 1*8)) // prefetch c + 2*rs_c
	prefetch(0, mem(rdx, 1*8))         // prefetch c + 3*rs_c
	prefetch(0, mem(rdx, rdi, 1, 1*8)) // prefetch c + 4*rs_c
	prefetch(0, mem(rdx, rdi, 2, 1*8)) // prefetch c + 5*rs_c

	jmp(.DPOSTPFETCH)                  // jump to end of prefetching c
	label(.DCOLPFETCH)                 // column-stored prefetching c

	mov(var(cs_c), rsi)                // load cs_c to rsi (temporarily)
	lea(mem(, rsi, 8), rsi)            // cs_c *= sizeof(double)
	prefetch(0, mem(r12, 5*8))         // prefetch c + 0*cs_c
	prefetch(0, mem(r12, rsi, 1, 5*8)) // prefetch c + 1*cs_c

	label(.DPOSTPFETCH)                // done prefetching c
#endif


#if 1
	lea(mem(r9, r9, 2), rcx)           // rcx = 3*cs_a;

	lea(mem(rax, r8,  4), rdx)         // use rdx for prefetching lines
	lea(mem(rdx, r8,  2), rdx)         // from next upanel of a.

	//lea(mem(rax, r9,  8), rdx)         // use rdx for prefetching a.
	//lea(mem(rdx, r9,  8), rdx)         // rdx = a + 16*cs_a;
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

#if 1
    prefetch(0, mem(rdx, r9, 1, 5*8))
#endif

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

#if 1
	prefetch(0, mem(rdx, r9, 2, 5*8))
#endif
	
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

#if 1
	prefetch(0, mem(rdx, rcx, 1, 5*8))
	lea(mem(rdx, r9,  4), rdx)         // rdx += 4*cs_a;
#endif

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

#if 1
	prefetch(0, mem(rdx, 5*8))
	add(r9, rdx)                       // rdx += cs_a;
#endif
	
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


	
	mov(r12, rcx)                      // reset rcx to current utile of c.
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




	lea(mem(r12, rdi, 4), r12)         //
	lea(mem(r12, rdi, 2), r12)         // c_ii = r12 += 6*rs_c

	//lea(mem(r14, r8,  4), r14)         //
	//lea(mem(r14, r8,  2), r14)         // a_ii = r14 += 6*rs_a
	mov(var(ps_a8), rax)               // load ps_a8
	lea(mem(r14, rax, 1), r14)         // a_ii = r14 += ps_a8

	dec(r11)                           // ii -= 1;
	jne(.DLOOP6X2I)                    // iterate again if ii != 0.




	label(.DRETURN)

	

    end_asm(
	: // output operands (none)
	: // input operands
      [m_iter] "m" (m_iter),
      [k_iter] "m" (k_iter),
      [k_left] "m" (k_left),
      [a]      "m" (a),
      [rs_a]   "m" (rs_a),
      [cs_a]   "m" (cs_a),
      [ps_a8]  "m" (ps_a8),
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
	  "memory"
	)

	consider_edge_cases:

	// Handle edge cases in the m dimension, if they exist.
	if ( m_left )
	{
		const dim_t      nr_cur = 2;
		const dim_t      i_edge = m0 - ( dim_t )m_left;

		double* restrict cij = c + i_edge*rs_c;
		//double* restrict ai  = a + i_edge*rs_a;
		//double* restrict ai  = a + ( i_edge / 6 ) * ps_a;
		double* restrict ai  = a + m_iter * ps_a;
		double* restrict bj  = b;

#if 0
		// We add special handling for slightly inflated MR blocksizes
		// at edge cases, up to a maximum of 9.
		if ( 6 < m_left )
		{
			dgemmsup_ker_ft ker_fp1 = NULL;
			dgemmsup_ker_ft ker_fp2 = NULL;
			dim_t           mr1, mr2;

			if ( m_left == 7 )
			{
				mr1 = 4; mr2 = 3;
				ker_fp1 = bli_dgemmsup_rv_haswell_asm_4x2;
				ker_fp2 = bli_dgemmsup_rv_haswell_asm_3x2;
			}
			else if ( m_left == 8 )
			{
				mr1 = 4; mr2 = 4;
				ker_fp1 = bli_dgemmsup_rv_haswell_asm_4x2;
				ker_fp2 = bli_dgemmsup_rv_haswell_asm_4x2;
			}
			else // if ( m_left == 9 )
			{
				mr1 = 4; mr2 = 5;
				ker_fp1 = bli_dgemmsup_rv_haswell_asm_4x2;
				ker_fp2 = bli_dgemmsup_rv_haswell_asm_5x2;
			}

			ker_fp1
			(
			  conja, conjb, mr1, nr_cur, k0,
			  alpha, ai, rs_a0, cs_a0, bj, rs_b0, cs_b0,
			  beta, cij, rs_c0, cs_c0, data, cntx
			);
			cij += mr1*rs_c0; ai += mr1*rs_a0;

			ker_fp2
			(
			  conja, conjb, mr2, nr_cur, k0,
			  alpha, ai, rs_a0, cs_a0, bj, rs_b0, cs_b0,
			  beta, cij, rs_c0, cs_c0, data, cntx
			);

			return;
		}
#endif

#if 1
		dgemmsup_ker_ft ker_fps[6] =
		{
		  NULL,
		  bli_dgemmsup_rv_haswell_asm_1x2,
		  bli_dgemmsup_rv_haswell_asm_2x2,
		  bli_dgemmsup_rv_haswell_asm_3x2,
		  bli_dgemmsup_rv_haswell_asm_4x2,
		  bli_dgemmsup_rv_haswell_asm_5x2
		};

		dgemmsup_ker_ft ker_fp = ker_fps[ m_left ];

		ker_fp
		(
		  conja, conjb, m_left, nr_cur, k0,
		  alpha, ai, rs_a0, cs_a0, bj, rs_b0, cs_b0,
		  beta, cij, rs_c0, cs_c0, data, cntx
		);

		return;
#else
		if ( 5 <= m_left )
		{
			const dim_t mr_cur = 5;

			bli_dgemmsup_rv_haswell_asm_5x2
			(
			  conja, conjb, mr_cur, nr_cur, k0,
			  alpha, ai, rs_a0, cs_a0, bj, rs_b0, cs_b0,
			  beta, cij, rs_c0, cs_c0, data, cntx
			);
			cij += mr_cur*rs_c0; ai += mr_cur*rs_a0; m_left -= mr_cur;
		}
		if ( 4 <= m_left )
		{
			const dim_t mr_cur = 4;

			bli_dgemmsup_rv_haswell_asm_4x2
			(
			  conja, conjb, mr_cur, nr_cur, k0,
			  alpha, ai, rs_a0, cs_a0, bj, rs_b0, cs_b0,
			  beta, cij, rs_c0, cs_c0, data, cntx
			);
			cij += mr_cur*rs_c0; ai += mr_cur*rs_a0; m_left -= mr_cur;
		}
		if ( 3 <= m_left )
		{
			const dim_t mr_cur = 3;

			bli_dgemmsup_rv_haswell_asm_3x2
			(
			  conja, conjb, mr_cur, nr_cur, k0,
			  alpha, ai, rs_a0, cs_a0, bj, rs_b0, cs_b0,
			  beta, cij, rs_c0, cs_c0, data, cntx
			);
			cij += mr_cur*rs_c0; ai += mr_cur*rs_a0; m_left -= mr_cur;
		}
		if ( 2 <= m_left )
		{
			const dim_t mr_cur = 2;

			bli_dgemmsup_rv_haswell_asm_2x2
			(
			  conja, conjb, mr_cur, nr_cur, k0,
			  alpha, ai, rs_a0, cs_a0, bj, rs_b0, cs_b0,
			  beta, cij, rs_c0, cs_c0, data, cntx
			);
			cij += mr_cur*rs_c0; ai += mr_cur*rs_a0; m_left -= mr_cur;
		}
		if ( 1 == m_left )
		{
			const dim_t mr_cur = 1;

			bli_dgemmsup_rv_haswell_asm_1x2
			(
			  conja, conjb, mr_cur, nr_cur, k0,
			  alpha, ai, rs_a0, cs_a0, bj, rs_b0, cs_b0,
			  beta, cij, rs_c0, cs_c0, data, cntx
			);
		}
#endif
	}
}

