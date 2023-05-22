/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.

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

/**
 * Shuffle 2 double-precision elements selected by imm8 from S1 and S2,
 * and store the results in D1
 * S1 : 1  9 3 11 5 13 7 15
 * S2 : 2 10 4 12 6 14 8 16
 * D1 : 1  9  5  13  2  10  6  14
 * D2 : 3 11  7  15  4  12  8  16
*/
#define SHUFFLE_DATA(S1, S2, D1, D2, S3, S4, D3, D4) \
\
    VSHUFF64X2(IMM(0x88), ZMM(S1), ZMM(S2), ZMM(D1)) \
    VSHUFF64X2(IMM(0xDD), ZMM(S1), ZMM(S2), ZMM(D2)) \
    VSHUFF64X2(IMM(0x88), ZMM(S3), ZMM(S4), ZMM(D3)) \
    VSHUFF64X2(IMM(0xDD), ZMM(S3), ZMM(S4), ZMM(D4)) \

/**
 * Unpacks and interleave low half and high half of each
 * 128-bit lane in S1 and S2 and store into D1 and D2
 * respectively.
 * S1 : 1  2  3  4  5  6  7  8
 * S2 : 9 10 11 12 13 14 15 16
 * D1 : 1  9 3 11 5 13 7 15
 * D2 : 2 10 4 12 6 14 8 16
*/
#define UNPACK_LO_HIGH(S1, S2, D1, D2, S3, S4, D3, D4) \
\
    vunpcklpd( zmm(S1),  zmm(S2),  zmm(D1)) \
    vunpckhpd( zmm(S1),  zmm(S2),  zmm(D2)) \
    vunpcklpd( zmm(S3),  zmm(S4),  zmm(D3)) \
    vunpckhpd( zmm(S3),  zmm(S4),  zmm(D4))

void bli_dpackm_zen4_asm_8xk
     (
       conj_t              conja,
       pack_t              schema,
       dim_t               cdim0,
       dim_t               k0,
       dim_t               k0_max,
       double*    restrict kappa,
       double*    restrict a, inc_t inca0, inc_t lda0,
       double*    restrict p,              inc_t ldp0,
       cntx_t*    restrict cntx
     )
{
	// This is the panel dimension assumed by the packm kernel.
	const dim_t      mnr   = 8;

	// This is the "packing" dimension assumed by the packm kernel.
	// This should be equal to ldp.
	//const dim_t    packmnr = 8;

	// Typecast local copies of integers in case dim_t and inc_t are a
	// different size than is expected by load instructions.
	const uint64_t k_iter = k0 /8;
	const uint64_t k_left = k0 % 8;
	/**
	 * prepares mask for k_left, since we are computing in multiple of 8,
	 * for edge cases mask is initialized for loading and storing only
	 * left over elements.
	 */
	uint8_t mask = 0xff >> (0x8 - (k_left));
	if (mask == 0) mask = 0xff;

	// NOTE: For the purposes of the comments in this packm kernel, we
	// interpret inca and lda as rs_a and cs_a, respectively, and similarly
	// interpret ldp as cs_p (with rs_p implicitly unit). Thus, when reading
	// this packm kernel, you should think of the operation as packing an
	// m x n micropanel, where m and n are tiny and large, respectively, and
	// where elements of each column of the packed matrix P are contiguous.
	// (This packm kernel can still be used to pack micropanels of matrix B
	// in a gemm operation.)
	const uint64_t inca   = inca0;
	const uint64_t lda    = lda0;
	const uint64_t ldp    = ldp0;

	const bool     gs     = ( inca0 != 1 && lda0 != 1 );

	// NOTE: If/when this kernel ever supports scaling by kappa within the
	// assembly region, this constraint should be lifted.
	const bool     unitk  = bli_deq1( *kappa );

	double* restrict a_next = a + cdim0;

	// -------------------------------------------------------------------------

	if ( cdim0 == mnr && !gs && unitk )
	{
		begin_asm()
		mov(var(mask), rdx)                // load mask
		kmovw(edx, k(2))                   // move mask to k2 register
		mov(var(a), rax)                   // load address of a.
		mov(var(a), r13)                   // load address of a.
		mov(var(inca), r8)                 // load inca
		mov(var(lda), r10)                 // load lda
		lea(mem(, r8,  8), r8)             // inca *= sizeof(double)
		lea(mem(, r10, 8), r10)            // lda *= sizeof(double)
		mov(var(p), rbx)                   // load address of p.
		cmp(imm(8), r8)                    // set ZF if (8*inca) == 8.
		jz(.DCOLUNIT)                      // jump to column storage case

		// -- kappa unit, row storage on A -----------------------------------------

		label(.DROWUNIT)

		lea(mem(r8,  r8,  2), r12)         // r12 = 3*inca
		lea(mem(r12, r8,  2), rcx)         // rcx = 5*inca
		lea(mem(r12, r8,  4), rdx)         // rdx = 7*inca

		mov(var(k_iter), rsi)              // i = k_iter;
		test(rsi, rsi)                     // check i via logical AND.
		je(.DCONKLEFTROWU)                 // if i == 0, jump to code that
		                                   // contains the k_left loop.

		label(.DKITERROWU)                 // MAIN LOOP (k_iter)

		/**
		 * Load first 8 rows of matrix.
		 * Transpose 8x8 tile and store it back to destination buffer.
		 */
		vmovupd(mem(rax,         0), zmm6)
		vmovupd(mem(rax,  r8, 1, 0), zmm8)
		vmovupd(mem(rax,  r8, 2, 0), zmm10)
		vmovupd(mem(rax, r12, 1, 0), zmm12)
		vmovupd(mem(rax,  r8, 4, 0), zmm14)
		vmovupd(mem(rax, rcx, 1, 0), zmm16)
		vmovupd(mem(rax, r12, 2, 0), zmm18)
		vmovupd(mem(rax, rdx, 1, 0), zmm20)

		UNPACK_LO_HIGH(8, 6, 0, 1, 12, 10, 2, 3)
		SHUFFLE_DATA(2, 0, 4, 5, 3, 1, 30, 31)
		UNPACK_LO_HIGH(16, 14, 0, 1, 20, 18, 2, 3)
		SHUFFLE_DATA(2, 0, 6, 8, 3, 1, 10, 12)
		SHUFFLE_DATA(6, 4, 0, 1, 8, 5, 2, 3)
		SHUFFLE_DATA(10, 30, 4, 5, 12, 31, 6, 8)

		vmovupd(zmm0, mem(rbx, 0*64))
		vmovupd(zmm4, mem(rbx, 1*64))
		vmovupd(zmm2, mem(rbx, 2*64))
		vmovupd(zmm6, mem(rbx, 3*64))
		vmovupd(zmm1, mem(rbx, 4*64))
		vmovupd(zmm5, mem(rbx, 5*64))
		vmovupd(zmm3, mem(rbx, 6*64))
		vmovupd(zmm8, mem(rbx, 7*64))

		add(imm(8*8), r13)
		mov(r13, rax)
		add(imm(8*8*8), rbx)               // p += 8*ldp

		dec(rsi)                           // i -= 1;
		jne(.DKITERROWU)                   // iterate again if i != 0.

		label(.DCONKLEFTROWU)

		mov(var(k_left), rsi)              // i = k_left;
		test(rsi, rsi)                     // check i via logical AND.
		je(.DDONE)                         // if i == 0, we're done; jump to end.
		                                   // else, we prepare to enter k_left loop.

		label(.DKLEFTROWU)                 // EDGE LOOP (k_left)

		vmovupd(mem(rax,         0), zmm6 MASK_KZ(2))
		vmovupd(mem(rax,  r8, 1, 0), zmm8 MASK_KZ(2))
		vmovupd(mem(rax,  r8, 2, 0), zmm10 MASK_KZ(2))
		vmovupd(mem(rax, r12, 1, 0), zmm12 MASK_KZ(2))
		vmovupd(mem(rax,  r8, 4, 0), zmm14 MASK_KZ(2))
		vmovupd(mem(rax, rcx, 1, 0), zmm16 MASK_KZ(2))
		vmovupd(mem(rax, r12, 2, 0), zmm18 MASK_KZ(2))
		vmovupd(mem(rax, rdx, 1, 0), zmm20 MASK_KZ(2))

		UNPACK_LO_HIGH(8, 6, 0, 1, 12, 10, 2, 3)
		SHUFFLE_DATA(2, 0, 4, 5, 3, 1, 30, 31)
		UNPACK_LO_HIGH(16, 14, 0, 1, 20, 18, 2, 3)
		SHUFFLE_DATA(2, 0, 6, 8, 3, 1, 10, 12)
		SHUFFLE_DATA(6, 4, 0, 1, 8, 5, 2, 3)
		SHUFFLE_DATA(10, 30, 4, 5, 12, 31, 6, 8)

		cmp(imm(7), rsi)
		JZ(.UPDATE7L1)
		cmp(imm(6), rsi)
		JZ(.UPDATE6L1)
		cmp(imm(5), rsi)
		JZ(.UPDATE5L1)
		cmp(imm(4), rsi)
		JZ(.UPDATE4L1)
		cmp(imm(3), rsi)
		JZ(.UPDATE3L1)
		cmp(imm(2), rsi)
		JZ(.UPDATE2L1)
		cmp(imm(1), rsi)
		JZ(.UPDATE1L1)
		cmp(imm(0), rsi)
		JZ(.UPDATEDONE)

		LABEL(.UPDATE7L1)
		vmovupd(zmm0, mem(rbx, 0*64))
		vmovupd(zmm4, mem(rbx, 1*64))
		vmovupd(zmm2, mem(rbx, 2*64))
		vmovupd(zmm6, mem(rbx, 3*64))
		vmovupd(zmm1, mem(rbx, 4*64))
		vmovupd(zmm5, mem(rbx, 5*64))
		vmovupd(zmm3, mem(rbx, 6*64))
		jmp(.UPDATEDONE)

		LABEL(.UPDATE6L1)
		vmovupd(zmm0, mem(rbx, 0*64))
		vmovupd(zmm4, mem(rbx, 1*64))
		vmovupd(zmm2, mem(rbx, 2*64))
		vmovupd(zmm6, mem(rbx, 3*64))
		vmovupd(zmm1, mem(rbx, 4*64))
		vmovupd(zmm5, mem(rbx, 5*64))
		jmp(.UPDATEDONE)                                              // jump to end.

		LABEL(.UPDATE5L1)
		vmovupd(zmm0, mem(rbx, 0*64))
		vmovupd(zmm4, mem(rbx, 1*64))
		vmovupd(zmm2, mem(rbx, 2*64))
		vmovupd(zmm6, mem(rbx, 3*64))
		vmovupd(zmm1, mem(rbx, 4*64))
		jmp(.UPDATEDONE)

		LABEL(.UPDATE4L1)
		vmovupd(zmm0, mem(rbx, 0*64))
		vmovupd(zmm4, mem(rbx, 1*64))
		vmovupd(zmm2, mem(rbx, 2*64))
		vmovupd(zmm6, mem(rbx, 3*64))
		jmp(.UPDATEDONE)

		LABEL(.UPDATE3L1)
		vmovupd(zmm0, mem(rbx, 0*64))
		vmovupd(zmm4, mem(rbx, 1*64))
		vmovupd(zmm2, mem(rbx, 2*64))
		jmp(.UPDATEDONE)

		LABEL(.UPDATE2L1)
		vmovupd(zmm0, mem(rbx, 0*64))
		vmovupd(zmm4, mem(rbx, 1*64))
		jmp(.UPDATEDONE)

		LABEL(.UPDATE1L1)
		vmovupd(zmm0, mem(rbx, 0*64))
		jmp(.UPDATEDONE)

		LABEL(.UPDATEDONE)

		jmp(.DDONE)                        // jump to end.

		// -- kappa unit, column storage on A -----------------------------------------
		label(.DCOLUNIT)
		mov(var(a_next), rcx)
		mov(var(ldp), r8)                  // load lda
		lea(mem(, r8,  8), r8)             // inca *= sizeof(double)
		mov(var(k_iter), rsi)              // i = k_iter;
		test(rsi, rsi)                     // check i via logical AND.
		je(.DCONKLEFTCOLU)                 // if i == 0, jump to code that
		                                   // contains the k_left loop.


		label(.DKITERCOLU)                 // MAIN LOOP (k_iter)

		vmovupd(mem(rax,         0), zmm6)
		vmovupd(zmm6, mem(rbx))
		add(r10, rax)
		add(r8, rbx)
		vmovupd(mem(rax,         0), zmm6)
		vmovupd(zmm6, mem(rbx,))

		add(r10, rax)
		add(r8, rbx)

		vmovupd(mem(rax,         0), zmm6)
		vmovupd(zmm6, mem(rbx))
		add(r10, rax)
		add(r8, rbx)

		vmovupd(mem(rax,         0), zmm6)
		vmovupd(zmm6, mem(rbx))

		add(r10, rax)
		add(r8, rbx)

		vmovupd(mem(rax,         0), zmm6)
		vmovupd(zmm6, mem(rbx))

		add(r10, rax)
		add(r8, rbx)

		vmovupd(mem(rax,         0), zmm6)
		vmovupd(zmm6, mem(rbx))

		add(r10, rax)
		add(r8, rbx)

		vmovupd(mem(rax,         0), zmm6)
		vmovupd(zmm6, mem(rbx))

		add(r10, rax)
		add(r8, rbx)

		vmovupd(mem(rax,         0), zmm6)
		vmovupd(zmm6, mem(rbx))

		add(r10, rax)
		add(r8, rbx)

		dec(rsi)                           // i -= 1;
		jne(.DKITERCOLU)                   // iterate again if i != 0.

		label(.DCONKLEFTCOLU)

		mov(var(k_left), rsi)              // i = k_left;
		test(rsi, rsi)                     // check i via logical AND.
		je(.DDONE)                         // if i == 0, we're done; jump to end.
		                                   // else, we prepare to enter k_left loop.
		label(.DKLEFTCOLU)                 // EDGE LOOP (k_left)

		vmovupd(mem(rax,         0), zmm6)
		vmovupd(zmm6, mem(rbx))

		add(r10, rax)
		add(r8, rbx)

		dec(rsi)                           // i -= 1;
		jne(.DKLEFTCOLU)                   // iterate again if i != 0.

		label(.DDONE)

		end_asm(
		: // output operands (none)
		: // input operands
		  [mask] "m" (mask),
		  [k_iter] "m" (k_iter),
		  [k_left] "m" (k_left),
		  [a]      "m" (a),
		  [inca]   "m" (inca),
		  [lda]    "m" (lda),
		  [p]      "m" (p),
		  [ldp]    "m" (ldp),
		  [kappa]  "m" (kappa),
		  [a_next]    "m" (a_next)
		: // register clobber list
		  "rax", "rbx", "rcx", "rdx", "rsi",
		  "r8", "r10", "r12", "r13",
		  "zmm0", "zmm1", "zmm2", "zmm3",
		  "zmm4", "zmm5", "zmm6", "zmm7",
		  "zmm8", "zmm9", "zmm10", "zmm11",
		  "zmm12", "zmm13", "zmm14", "zmm15",
		  "zmm16", "zmm18", "zmm20", "zmm30", "zmm31", "memory"
		)
	}
	else // if ( cdim0 < mnr || gs || !unitk )
	{
		PASTEMAC(dscal2m,BLIS_TAPI_EX_SUF)
		(
		  0,
		  BLIS_NONUNIT_DIAG,
		  BLIS_DENSE,
		  ( trans_t )conja,
		  cdim0,
		  k0,
		  kappa,
		  a, inca0, lda0,
		  p,     1, ldp0,
		  cntx,
		  NULL
		);

		if ( cdim0 < mnr )
		{
			// Handle zero-filling along the "long" edge of the micropanel.

			const dim_t      i      = cdim0;
			const dim_t      m_edge = mnr - cdim0;
			const dim_t      n_edge = k0_max;
			double* restrict p_edge = p + (i  )*1;

			bli_dset0s_mxn
			(
			  m_edge,
			  n_edge,
			  p_edge, 1, ldp
			);
		}
	}
	if ( k0 < k0_max )
	{
		// Handle zero-filling along the "short" (far) edge of the micropanel.

		const dim_t      j      = k0;
		const dim_t      m_edge = mnr;
		const dim_t      n_edge = k0_max - k0;
		double* restrict p_edge = p + (j  )*ldp;

		bli_dset0s_mxn
		(
		  m_edge,
		  n_edge,
		  p_edge, 1, ldp
		);
	}
}
