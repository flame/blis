/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2014, The University of Texas at Austin
   Copyright (C) 2019 - 2020, Advanced Micro Devices, Inc.

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

// Prototype reference packm kernels.
PACKM_KER_PROT( double,   d, packm_8xk_haswell_ref )

void bli_dpackm_haswell_asm_8xk
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
#if 0
	bli_dpackm_8xk_haswell_ref
	(
	  conja, schema, cdim0, k0, k0_max,
	  kappa, a, inca0, lda0, p, ldp0, cntx
	);
	return;
#endif

	// This is the panel dimension assumed by the packm kernel.
	const dim_t      mnr   = 8;

	// This is the "packing" dimension assumed by the packm kernel.
	// This should be equal to ldp.
	//const dim_t    packmnr = 8;

	// Define a local copy of 1.0 so we can test for unit kappa.
	double           one_l = 1.0;
	double* restrict one   = &one_l;

	// Typecast local copies of integers in case dim_t and inc_t are a
	// different size than is expected by load instructions.
	const uint64_t k_iter = k0 / 4;
#if 1
	const uint64_t k_left = k0 % 4;
#else
	const uint64_t k_left = k0;
#endif

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


	// -------------------------------------------------------------------------

	if ( cdim0 == mnr && !gs && unitk )
	{
		begin_asm()
		
		mov(var(a), rax)                   // load address of a.

		mov(var(inca), r8)                 // load inca
		mov(var(lda), r10)                 // load lda
		lea(mem(, r8,  8), r8)             // inca *= sizeof(double)
		lea(mem(, r10, 8), r10)            // lda *= sizeof(double)

		mov(var(p), rbx)                   // load address of p.

		lea(mem(   , r10, 4), r14)         // r14 = 4*lda

		mov(var(one), rdx)                 // load address of 1.0 constant
		vmovsd(mem(rdx), xmm1)             // load 1.0
		
		mov(var(kappa), rcx)               // load address of kappa
		vmovsd(mem(rcx), xmm0)             // load kappa
		

										   // now branch on kappa == 1.0
		
		vucomisd(xmm0, xmm1)               // set ZF if kappa == 1.0
		je(.DKAPPAUNIT)                    // if ZF = 1, jump to beta == 0 case



		label(.DKAPPANONU)

		cmp(imm(8), r8)                    // set ZF if (8*inca) == 8.
		jz(.DCOLNONU)                      // jump to column storage case
		
		// -- kappa non-unit, row storage on A -------------------------------------

		label(.DROWNONU)

		jmp(.DDONE)                        // jump to end.


		// -- kappa non-unit, column storage on A ----------------------------------

		label(.DCOLNONU)

		jmp(.DDONE)                        // jump to end.
		



		label(.DKAPPAUNIT)

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

		vmovupd(mem(rax,         0), ymm0)
		vmovupd(mem(rax,  r8, 1, 0), ymm2)
		vmovupd(mem(rax,  r8, 2, 0), ymm4)
		vmovupd(mem(rax, r12, 1, 0), ymm6)

		vunpcklpd(ymm2, ymm0, ymm10)
		vunpckhpd(ymm2, ymm0, ymm11)
		vunpcklpd(ymm6, ymm4, ymm12)
		vunpckhpd(ymm6, ymm4, ymm13)
		vinsertf128(imm(0x1), xmm12, ymm10, ymm0)
		vinsertf128(imm(0x1), xmm13, ymm11, ymm2)
		vperm2f128(imm(0x31), ymm12, ymm10, ymm4)
		vperm2f128(imm(0x31), ymm13, ymm11, ymm6)

		vmovupd(ymm0, mem(rbx, 0*64))
		vmovupd(ymm2, mem(rbx, 1*64))
		vmovupd(ymm4, mem(rbx, 2*64))
		vmovupd(ymm6, mem(rbx, 3*64))

		vmovupd(mem(rax,  r8, 4, 0), ymm1)
		vmovupd(mem(rax, rcx, 1, 0), ymm3)
		vmovupd(mem(rax, r12, 2, 0), ymm5)
		vmovupd(mem(rax, rdx, 1, 0), ymm7)

		add(r14, rax)                      // a += 4*lda;

		vunpcklpd(ymm3, ymm1, ymm10)
		vunpckhpd(ymm3, ymm1, ymm11)
		vunpcklpd(ymm7, ymm5, ymm12)
		vunpckhpd(ymm7, ymm5, ymm13)
		vinsertf128(imm(0x1), xmm12, ymm10, ymm1)
		vinsertf128(imm(0x1), xmm13, ymm11, ymm3)
		vperm2f128(imm(0x31), ymm12, ymm10, ymm5)
		vperm2f128(imm(0x31), ymm13, ymm11, ymm7)

		vmovupd(ymm1, mem(rbx, 0*64+32))
		vmovupd(ymm3, mem(rbx, 1*64+32))
		vmovupd(ymm5, mem(rbx, 2*64+32))
		vmovupd(ymm7, mem(rbx, 3*64+32))

		add(imm(4*8*8), rbx)               // p += 4*ldp = 4*8;

		dec(rsi)                           // i -= 1;
		jne(.DKITERROWU)                   // iterate again if i != 0.



		label(.DCONKLEFTROWU)

		mov(var(k_left), rsi)              // i = k_left;
		test(rsi, rsi)                     // check i via logical AND.
		je(.DDONE)                         // if i == 0, we're done; jump to end.
		                                   // else, we prepare to enter k_left loop.


		label(.DKLEFTROWU)                 // EDGE LOOP (k_left)

		vmovsd(mem(rax,         0), xmm0)
		vmovsd(mem(rax,  r8, 1, 0), xmm2)
		vmovsd(mem(rax,  r8, 2, 0), xmm4)
		vmovsd(mem(rax, r12, 1, 0), xmm6)
		vmovsd(mem(rax,  r8, 4, 0), xmm1)
		vmovsd(mem(rax, rcx, 1, 0), xmm3)
		vmovsd(mem(rax, r12, 2, 0), xmm5)
		vmovsd(mem(rax, rdx, 1, 0), xmm7)

		add(r10, rax)                      // a += lda;

		vmovsd(xmm0, mem(rbx, 0*8))
		vmovsd(xmm2, mem(rbx, 1*8))
		vmovsd(xmm4, mem(rbx, 2*8))
		vmovsd(xmm6, mem(rbx, 3*8))
		vmovsd(xmm1, mem(rbx, 4*8))
		vmovsd(xmm3, mem(rbx, 5*8))
		vmovsd(xmm5, mem(rbx, 6*8))
		vmovsd(xmm7, mem(rbx, 7*8))

		add(imm(8*8), rbx)                 // p += ldp = 8;

		dec(rsi)                           // i -= 1;
		jne(.DKLEFTROWU)                   // iterate again if i != 0.


		jmp(.DDONE)                        // jump to end.


		// -- kappa unit, column storage on A --------------------------------------

		label(.DCOLUNIT)
		
		lea(mem(r10, r10, 2), r13)         // r13 = 3*lda

		mov(var(k_iter), rsi)              // i = k_iter;
		test(rsi, rsi)                     // check i via logical AND.
		je(.DCONKLEFTCOLU)                 // if i == 0, jump to code that
		                                   // contains the k_left loop.


		label(.DKITERCOLU)                 // MAIN LOOP (k_iter)

		vmovupd(mem(rax,          0), ymm0)
		vmovupd(mem(rax,         32), ymm1)
		vmovupd(ymm0, mem(rbx, 0*64+ 0))
		vmovupd(ymm1, mem(rbx, 0*64+32))

		vmovupd(mem(rax, r10, 1,  0), ymm2)
		vmovupd(mem(rax, r10, 1, 32), ymm3)
		vmovupd(ymm2, mem(rbx, 1*64+ 0))
		vmovupd(ymm3, mem(rbx, 1*64+32))

		vmovupd(mem(rax, r10, 2,  0), ymm4)
		vmovupd(mem(rax, r10, 2, 32), ymm5)
		vmovupd(ymm4, mem(rbx, 2*64+ 0))
		vmovupd(ymm5, mem(rbx, 2*64+32))

		vmovupd(mem(rax, r13, 1,  0), ymm6)
		vmovupd(mem(rax, r13, 1, 32), ymm7)
		add(r14, rax)                      // a += 4*lda;
		vmovupd(ymm6, mem(rbx, 3*64+ 0))
		vmovupd(ymm7, mem(rbx, 3*64+32))
		add(imm(4*8*8), rbx)               // p += 4*ldp = 4*8;

		dec(rsi)                           // i -= 1;
		jne(.DKITERCOLU)                   // iterate again if i != 0.



		label(.DCONKLEFTCOLU)

		mov(var(k_left), rsi)              // i = k_left;
		test(rsi, rsi)                     // check i via logical AND.
		je(.DDONE)                         // if i == 0, we're done; jump to end.
		                                   // else, we prepare to enter k_left loop.


		label(.DKLEFTCOLU)                 // EDGE LOOP (k_left)

		vmovupd(mem(rax,          0), ymm0)
		vmovupd(mem(rax,         32), ymm1)
		add(r10, rax)                      // a += lda;
		vmovupd(ymm0, mem(rbx, 0*64+ 0))
		vmovupd(ymm1, mem(rbx, 0*64+32))
		add(imm(8*8), rbx)                 // p += ldp = 8;

		dec(rsi)                           // i -= 1;
		jne(.DKLEFTCOLU)                   // iterate again if i != 0.


		//jmp(.DDONE)                        // jump to end.



		label(.DDONE)
		
		

		end_asm(
		: // output operands (none)
		: // input operands
		  [k_iter] "m" (k_iter),
		  [k_left] "m" (k_left),
		  [a]      "m" (a),
		  [inca]   "m" (inca),
		  [lda]    "m" (lda),
		  [p]      "m" (p),
		  [ldp]    "m" (ldp),
		  [kappa]  "m" (kappa),
		  [one]    "m" (one)
		: // register clobber list
		  "rax", "rbx", "rcx", "rdx", "rsi", "rdi",
		  "r8", /*"r9",*/ "r10", /*"r11",*/ "r12", "r13", "r14", "r15",
		  "xmm0", "xmm1", "xmm2", "xmm3",
		  "xmm4", "xmm5", "xmm6", "xmm7",
		  "xmm8", "xmm9", "xmm10", "xmm11",
		  "xmm12", "xmm13", "xmm14", "xmm15",
		  "memory"
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

