/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2014, The University of Texas at Austin
   Copyright (C) 2018 - 2019, Advanced Micro Devices, Inc.

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

#define SGEMM_INPUT_GS_BETA_NZ \
	vmovlps(mem(rcx), xmm0, xmm0) \
	vmovhps(mem(rcx, rsi, 1), xmm0, xmm0) \
	vmovlps(mem(rcx, rsi, 2), xmm1, xmm1) \
	vmovhps(mem(rcx, r13, 1), xmm1, xmm1) \
	vshufps(imm(0x88), xmm1, xmm0, xmm0) \
	vmovlps(mem(rcx, rsi, 4), xmm2, xmm2) \
	vmovhps(mem(rcx, r15, 1), xmm2, xmm2) \
	/* We can't use vmovhps for loading the last element becauase that
	   might result in reading beyond valid memory. (vmov[lh]psd load
	   pairs of adjacent floats at a time.) So we need to use vmovss
	   instead. But since we're limited to using ymm0 through ymm2
	   (ymm3 contains beta and ymm4 through ymm15 contain the microtile)
	   and due to the way vmovss zeros out all bits above 31, we have to
	   load element 7 before element 6. */ \
	vmovss(mem(rcx, r10, 1), xmm1) \
	vpermilps(imm(0xcf), xmm1, xmm1) \
	vmovlps(mem(rcx, r13, 2), xmm1, xmm1) \
	/*vmovhps(mem(rcx, r10, 1), xmm1, xmm1)*/ \
	vshufps(imm(0x88), xmm1, xmm2, xmm2) \
	vperm2f128(imm(0x20), ymm2, ymm0, ymm0)

#define SGEMM_OUTPUT_GS_BETA_NZ \
	vextractf128(imm(1), ymm0, xmm2) \
	vmovss(xmm0, mem(rcx)) \
	vpermilps(imm(0x39), xmm0, xmm1) \
	vmovss(xmm1, mem(rcx, rsi, 1)) \
	vpermilps(imm(0x39), xmm1, xmm0) \
	vmovss(xmm0, mem(rcx, rsi, 2)) \
	vpermilps(imm(0x39), xmm0, xmm1) \
	vmovss(xmm1, mem(rcx, r13, 1)) \
	vmovss(xmm2, mem(rcx, rsi, 4)) \
	vpermilps(imm(0x39), xmm2, xmm1) \
	vmovss(xmm1, mem(rcx, r15, 1)) \
	vpermilps(imm(0x39), xmm1, xmm2) \
	vmovss(xmm2, mem(rcx, r13, 2)) \
	vpermilps(imm(0x39), xmm2, xmm1) \
	vmovss(xmm1, mem(rcx, r10, 1))

void bli_sgemm_haswell_asm_6x16
     (
       dim_t               m,
       dim_t               n,
       dim_t               k,
       float*     restrict alpha,
       float*     restrict a,
       float*     restrict b,
       float*     restrict beta,
       float*     restrict c, inc_t rs_c0, inc_t cs_c0,
       auxinfo_t* restrict data,
       cntx_t*    restrict cntx
     )
{
	//void*   a_next = bli_auxinfo_next_a( data );
	//void*   b_next = bli_auxinfo_next_b( data );

	// Typecast local copies of integers in case dim_t and inc_t are a
	// different size than is expected by load instructions.
	uint64_t k_iter = k / 4;
	uint64_t k_left = k % 4;
	uint64_t rs_c   = rs_c0;
	uint64_t cs_c   = cs_c0;

	GEMM_UKR_SETUP_CT_AMBI( s, 6, 16, true );

	begin_asm()

	vzeroall() // zero all xmm/ymm registers.


	mov(var(a), rax) // load address of a.
	mov(var(b), rbx) // load address of b.
	//mov(%9, r15) // load address of b_next.

	add(imm(32*4), rbx)
	// initialize loop by pre-loading
	vmovaps(mem(rbx, -4*32), ymm0)
	vmovaps(mem(rbx, -3*32), ymm1)

	mov(var(c), rcx) // load address of c
	mov(var(rs_c), rdi) // load rs_c
	lea(mem(, rdi, 4), rdi) // rs_c *= sizeof(float)
	mov(var(cs_c), rsi) // load cs_c
	lea(mem(, rsi, 4), rsi) // cs_c *= sizeof(float)

	cmp(imm(4), rdi) // set ZF if (4*rs_c) == 4.
	jz(.SCOLPREFETCH) // jump to column prefetch case

		lea(mem(rdi, rdi, 2), r13) // r13 = 3*rs_c;
		lea(mem(rcx, r13, 1), rdx) // rdx = c + 3*rs_c;
		prefetch(0, mem(rcx, 7*8)) // prefetch c + 0*rs_c
		prefetch(0, mem(rcx, rdi, 1, 7*8)) // prefetch c + 1*rs_c
		prefetch(0, mem(rcx, rdi, 2, 7*8)) // prefetch c + 2*rs_c
		prefetch(0, mem(rdx, 7*8)) // prefetch c + 3*rs_c
		prefetch(0, mem(rdx, rdi, 1, 7*8)) // prefetch c + 4*rs_c
		prefetch(0, mem(rdx, rdi, 2, 7*8)) // prefetch c + 5*rs_c

		jmp(.SPREFETCHDONE)

	label(.SCOLPREFETCH)

		lea(mem(rsi, rsi, 2), r13) // r13 = 3*cs_c;
		lea(mem(rcx, r13, 1), rdx) // rdx = c + 3*cs_c;
		prefetch(0, mem(rcx, 7*8)) // prefetch c + 0*cs_c
		prefetch(0, mem(rcx, rsi, 1, 7*8)) // prefetch c + 1*cs_c
		prefetch(0, mem(rcx, rsi, 2, 7*8)) // prefetch c + 2*cs_c
		prefetch(0, mem(rdx, 7*8)) // prefetch c + 3*cs_c
		prefetch(0, mem(rdx, rsi, 1, 7*8)) // prefetch c + 4*cs_c
		prefetch(0, mem(rdx, rsi, 2, 7*8)) // prefetch c + 5*cs_c
		prefetch(0, mem(rdx, r13, 1, 7*8)) // prefetch c + 6*cs_c
		prefetch(0, mem(rdx, rsi, 4, 7*8)) // prefetch c + 7*cs_c
		lea(mem(rcx, rsi, 8), r14) // r14 = c + 8*cs_c;
		lea(mem(r14, r13, 1), rdx) // rdx = c + 11*cs_c;
		prefetch(0, mem(r14, 7*8)) // prefetch c + 8*cs_c
		prefetch(0, mem(r14, rsi, 1, 7*8)) // prefetch c + 9*cs_c
		prefetch(0, mem(r14, rsi, 2, 7*8)) // prefetch c + 10*cs_c
		prefetch(0, mem(rdx, 7*8)) // prefetch c + 11*cs_c
		prefetch(0, mem(rdx, rsi, 1, 7*8)) // prefetch c + 12*cs_c
		prefetch(0, mem(rdx, rsi, 2, 7*8)) // prefetch c + 13*cs_c
		prefetch(0, mem(rdx, r13, 1, 7*8)) // prefetch c + 14*cs_c
		prefetch(0, mem(rdx, rsi, 4, 7*8)) // prefetch c + 15*cs_c

	label(.SPREFETCHDONE)

	mov(var(k_iter), rsi) // i = k_iter;
	test(rsi, rsi) // check i via logical AND.
	je(.SCONSIDKLEFT) // if i == 0, jump to code that
	// contains the k_left loop.


	label(.SLOOPKITER) // MAIN LOOP


	// iteration 0
	prefetch(0, mem(rax, 64*4))

	vbroadcastss(mem(rax, 0*4), ymm2)
	vbroadcastss(mem(rax, 1*4), ymm3)
	vfmadd231ps(ymm0, ymm2, ymm4)
	vfmadd231ps(ymm1, ymm2, ymm5)
	vfmadd231ps(ymm0, ymm3, ymm6)
	vfmadd231ps(ymm1, ymm3, ymm7)

	vbroadcastss(mem(rax, 2*4), ymm2)
	vbroadcastss(mem(rax, 3*4), ymm3)
	vfmadd231ps(ymm0, ymm2, ymm8)
	vfmadd231ps(ymm1, ymm2, ymm9)
	vfmadd231ps(ymm0, ymm3, ymm10)
	vfmadd231ps(ymm1, ymm3, ymm11)

	vbroadcastss(mem(rax, 4*4), ymm2)
	vbroadcastss(mem(rax, 5*4), ymm3)
	vfmadd231ps(ymm0, ymm2, ymm12)
	vfmadd231ps(ymm1, ymm2, ymm13)
	vfmadd231ps(ymm0, ymm3, ymm14)
	vfmadd231ps(ymm1, ymm3, ymm15)

	vmovaps(mem(rbx, -2*32), ymm0)
	vmovaps(mem(rbx, -1*32), ymm1)

	// iteration 1
	vbroadcastss(mem(rax, 6*4), ymm2)
	vbroadcastss(mem(rax, 7*4), ymm3)
	vfmadd231ps(ymm0, ymm2, ymm4)
	vfmadd231ps(ymm1, ymm2, ymm5)
	vfmadd231ps(ymm0, ymm3, ymm6)
	vfmadd231ps(ymm1, ymm3, ymm7)

	vbroadcastss(mem(rax, 8*4), ymm2)
	vbroadcastss(mem(rax, 9*4), ymm3)
	vfmadd231ps(ymm0, ymm2, ymm8)
	vfmadd231ps(ymm1, ymm2, ymm9)
	vfmadd231ps(ymm0, ymm3, ymm10)
	vfmadd231ps(ymm1, ymm3, ymm11)

	vbroadcastss(mem(rax, 10*4), ymm2)
	vbroadcastss(mem(rax, 11*4), ymm3)
	vfmadd231ps(ymm0, ymm2, ymm12)
	vfmadd231ps(ymm1, ymm2, ymm13)
	vfmadd231ps(ymm0, ymm3, ymm14)
	vfmadd231ps(ymm1, ymm3, ymm15)

	vmovaps(mem(rbx, 0*32), ymm0)
	vmovaps(mem(rbx, 1*32), ymm1)

	// iteration 2
	prefetch(0, mem(rax, 76*4))

	vbroadcastss(mem(rax, 12*4), ymm2)
	vbroadcastss(mem(rax, 13*4), ymm3)
	vfmadd231ps(ymm0, ymm2, ymm4)
	vfmadd231ps(ymm1, ymm2, ymm5)
	vfmadd231ps(ymm0, ymm3, ymm6)
	vfmadd231ps(ymm1, ymm3, ymm7)

	vbroadcastss(mem(rax, 14*4), ymm2)
	vbroadcastss(mem(rax, 15*4), ymm3)
	vfmadd231ps(ymm0, ymm2, ymm8)
	vfmadd231ps(ymm1, ymm2, ymm9)
	vfmadd231ps(ymm0, ymm3, ymm10)
	vfmadd231ps(ymm1, ymm3, ymm11)

	vbroadcastss(mem(rax, 16*4), ymm2)
	vbroadcastss(mem(rax, 17*4), ymm3)
	vfmadd231ps(ymm0, ymm2, ymm12)
	vfmadd231ps(ymm1, ymm2, ymm13)
	vfmadd231ps(ymm0, ymm3, ymm14)
	vfmadd231ps(ymm1, ymm3, ymm15)

	vmovaps(mem(rbx, 2*32), ymm0)
	vmovaps(mem(rbx, 3*32), ymm1)

	// iteration 3
	vbroadcastss(mem(rax, 18*4), ymm2)
	vbroadcastss(mem(rax, 19*4), ymm3)
	vfmadd231ps(ymm0, ymm2, ymm4)
	vfmadd231ps(ymm1, ymm2, ymm5)
	vfmadd231ps(ymm0, ymm3, ymm6)
	vfmadd231ps(ymm1, ymm3, ymm7)

	vbroadcastss(mem(rax, 20*4), ymm2)
	vbroadcastss(mem(rax, 21*4), ymm3)
	vfmadd231ps(ymm0, ymm2, ymm8)
	vfmadd231ps(ymm1, ymm2, ymm9)
	vfmadd231ps(ymm0, ymm3, ymm10)
	vfmadd231ps(ymm1, ymm3, ymm11)

	vbroadcastss(mem(rax, 22*4), ymm2)
	vbroadcastss(mem(rax, 23*4), ymm3)
	vfmadd231ps(ymm0, ymm2, ymm12)
	vfmadd231ps(ymm1, ymm2, ymm13)
	vfmadd231ps(ymm0, ymm3, ymm14)
	vfmadd231ps(ymm1, ymm3, ymm15)

	add(imm(4*6*4), rax) // a += 4*6  (unroll x mr)
	add(imm(4*16*4), rbx) // b += 4*16 (unroll x nr)

	vmovaps(mem(rbx, -4*32), ymm0)
	vmovaps(mem(rbx, -3*32), ymm1)


	dec(rsi) // i -= 1;
	jne(.SLOOPKITER) // iterate again if i != 0.






	label(.SCONSIDKLEFT)

	mov(var(k_left), rsi) // i = k_left;
	test(rsi, rsi) // check i via logical AND.
	je(.SPOSTACCUM) // if i == 0, we're done; jump to end.
	// else, we prepare to enter k_left loop.


	label(.SLOOPKLEFT) // EDGE LOOP

	prefetch(0, mem(rax, 64*4))

	vbroadcastss(mem(rax, 0*4), ymm2)
	vbroadcastss(mem(rax, 1*4), ymm3)
	vfmadd231ps(ymm0, ymm2, ymm4)
	vfmadd231ps(ymm1, ymm2, ymm5)
	vfmadd231ps(ymm0, ymm3, ymm6)
	vfmadd231ps(ymm1, ymm3, ymm7)

	vbroadcastss(mem(rax, 2*4), ymm2)
	vbroadcastss(mem(rax, 3*4), ymm3)
	vfmadd231ps(ymm0, ymm2, ymm8)
	vfmadd231ps(ymm1, ymm2, ymm9)
	vfmadd231ps(ymm0, ymm3, ymm10)
	vfmadd231ps(ymm1, ymm3, ymm11)

	vbroadcastss(mem(rax, 4*4), ymm2)
	vbroadcastss(mem(rax, 5*4), ymm3)
	vfmadd231ps(ymm0, ymm2, ymm12)
	vfmadd231ps(ymm1, ymm2, ymm13)
	vfmadd231ps(ymm0, ymm3, ymm14)
	vfmadd231ps(ymm1, ymm3, ymm15)

	add(imm(1*6*4), rax) // a += 1*6  (unroll x mr)
	add(imm(1*16*4), rbx) // b += 1*16 (unroll x nr)

	vmovaps(mem(rbx, -4*32), ymm0)
	vmovaps(mem(rbx, -3*32), ymm1)


	dec(rsi) // i -= 1;
	jne(.SLOOPKLEFT) // iterate again if i != 0.



	label(.SPOSTACCUM)




	mov(var(alpha), rax) // load address of alpha
	mov(var(beta), rbx) // load address of beta
	vbroadcastss(mem(rax), ymm0) // load alpha and duplicate
	vbroadcastss(mem(rbx), ymm3) // load beta and duplicate

	vmulps(ymm0, ymm4, ymm4) // scale by alpha
	vmulps(ymm0, ymm5, ymm5)
	vmulps(ymm0, ymm6, ymm6)
	vmulps(ymm0, ymm7, ymm7)
	vmulps(ymm0, ymm8, ymm8)
	vmulps(ymm0, ymm9, ymm9)
	vmulps(ymm0, ymm10, ymm10)
	vmulps(ymm0, ymm11, ymm11)
	vmulps(ymm0, ymm12, ymm12)
	vmulps(ymm0, ymm13, ymm13)
	vmulps(ymm0, ymm14, ymm14)
	vmulps(ymm0, ymm15, ymm15)






	mov(var(cs_c), rsi) // load cs_c
	lea(mem(, rsi, 4), rsi) // rsi = cs_c * sizeof(float)

	lea(mem(rcx, rsi, 8), rdx) // load address of c +  8*cs_c;
	lea(mem(rcx, rdi, 4), r14) // load address of c +  4*rs_c;

	lea(mem(rsi, rsi, 2), r13) // r13 = 3*cs_c;
	lea(mem(rsi, rsi, 4), r15) // r15 = 5*cs_c;
	lea(mem(r13, rsi, 4), r10) // r10 = 7*cs_c;


	// now avoid loading C if beta == 0

	vxorps(ymm0, ymm0, ymm0) // set ymm0 to zero.
	vucomiss(xmm0, xmm3) // set ZF if beta == 0.
	je(.SBETAZERO) // if ZF = 1, jump to beta == 0 case

		cmp(imm(4), rdi) // set ZF if (4*cs_c) == 4.
		jz(.SCOLSTORED) // jump to column storage case

			vfmadd231ps(mem(rcx), ymm3, ymm4)
			vmovups(ymm4, mem(rcx))
			vfmadd231ps(mem(rcx,32), ymm3, ymm5)
			vmovups(ymm5, mem(rcx,32))
			add(rdi, rcx)


			vfmadd231ps(mem(rcx), ymm3, ymm6)
			vmovups(ymm6, mem(rcx))
			vfmadd231ps(mem(rcx,32), ymm3, ymm7)
			vmovups(ymm7, mem(rcx,32))
			add(rdi, rcx)


			vfmadd231ps(mem(rcx), ymm3, ymm8)
			vmovups(ymm8, mem(rcx))
			vfmadd231ps(mem(rcx,32), ymm3, ymm9)
			vmovups(ymm9, mem(rcx,32))
			add(rdi, rcx)


			vfmadd231ps(mem(rcx), ymm3, ymm10)
			vmovups(ymm10, mem(rcx))
			vfmadd231ps(mem(rcx,32), ymm3, ymm11)
			vmovups(ymm11, mem(rcx,32))
			add(rdi, rcx)


			vfmadd231ps(mem(rcx), ymm3, ymm12)
			vmovups(ymm12, mem(rcx))
			vfmadd231ps(mem(rcx,32), ymm3, ymm13)
			vmovups(ymm13, mem(rcx,32))
			add(rdi, rcx)


			vfmadd231ps(mem(rcx), ymm3, ymm14)
			vmovups(ymm14, mem(rcx))
			vfmadd231ps(mem(rcx,32), ymm3, ymm15)
			vmovups(ymm15, mem(rcx,32))
			//add(rdi, rcx)

			jmp(.SDONE) // jump to end.

		label(.SCOLSTORED)

			vunpcklps(ymm6, ymm4, ymm0)
			vunpcklps(ymm10, ymm8, ymm1)
			vshufps(imm(0x4e), ymm1, ymm0, ymm2)
			vblendps(imm(0xcc), ymm2, ymm0, ymm0)
			vblendps(imm(0x33), ymm2, ymm1, ymm1)

			vextractf128(imm(0x1), ymm0, xmm2)
			vfmadd231ps(mem(rcx), xmm3, xmm0)
			vfmadd231ps(mem(rcx, rsi, 4), xmm3, xmm2)
			vmovups(xmm0, mem(rcx)) // store ( gamma00..gamma30 )
			vmovups(xmm2, mem(rcx, rsi, 4)) // store ( gamma04..gamma34 )

			vextractf128(imm(0x1), ymm1, xmm2)
			vfmadd231ps(mem(rcx, rsi, 1), xmm3, xmm1)
			vfmadd231ps(mem(rcx, r15, 1), xmm3, xmm2)
			vmovups(xmm1, mem(rcx, rsi, 1)) // store ( gamma01..gamma31 )
			vmovups(xmm2, mem(rcx, r15, 1)) // store ( gamma05..gamma35 )


			vunpckhps(ymm6, ymm4, ymm0)
			vunpckhps(ymm10, ymm8, ymm1)
			vshufps(imm(0x4e), ymm1, ymm0, ymm2)
			vblendps(imm(0xcc), ymm2, ymm0, ymm0)
			vblendps(imm(0x33), ymm2, ymm1, ymm1)

			vextractf128(imm(0x1), ymm0, xmm2)
			vfmadd231ps(mem(rcx, rsi, 2), xmm3, xmm0)
			vfmadd231ps(mem(rcx, r13, 2), xmm3, xmm2)
			vmovups(xmm0, mem(rcx, rsi, 2)) // store ( gamma02..gamma32 )
			vmovups(xmm2, mem(rcx, r13, 2)) // store ( gamma06..gamma36 )

			vextractf128(imm(0x1), ymm1, xmm2)
			vfmadd231ps(mem(rcx, r13, 1), xmm3, xmm1)
			vfmadd231ps(mem(rcx, r10, 1), xmm3, xmm2)
			vmovups(xmm1, mem(rcx, r13, 1)) // store ( gamma03..gamma33 )
			vmovups(xmm2, mem(rcx, r10, 1)) // store ( gamma07..gamma37 )

			lea(mem(rcx, rsi, 8), rcx) // rcx += 8*cs_c

			vunpcklps(ymm14, ymm12, ymm0)
			vextractf128(imm(0x1), ymm0, xmm2)
			vmovlpd(mem(r14), xmm1, xmm1)
			vmovhpd(mem(r14, rsi, 1), xmm1, xmm1)
			vfmadd231ps(xmm1, xmm3, xmm0)
			vmovlpd(xmm0, mem(r14)) // store ( gamma40..gamma50 )
			vmovhpd(xmm0, mem(r14, rsi, 1)) // store ( gamma41..gamma51 )
			vmovlpd(mem(r14, rsi, 4), xmm1, xmm1)
			vmovhpd(mem(r14, r15, 1), xmm1, xmm1)
			vfmadd231ps(xmm1, xmm3, xmm2)
			vmovlpd(xmm2, mem(r14, rsi, 4)) // store ( gamma44..gamma54 )
			vmovhpd(xmm2, mem(r14, r15, 1)) // store ( gamma45..gamma55 )

			vunpckhps(ymm14, ymm12, ymm0)
			vextractf128(imm(0x1), ymm0, xmm2)
			vmovlpd(mem(r14, rsi, 2), xmm1, xmm1)
			vmovhpd(mem(r14, r13, 1), xmm1, xmm1)
			vfmadd231ps(xmm1, xmm3, xmm0)
			vmovlpd(xmm0, mem(r14, rsi, 2)) // store ( gamma42..gamma52 )
			vmovhpd(xmm0, mem(r14, r13, 1)) // store ( gamma43..gamma53 )
			vmovlpd(mem(r14, r13, 2), xmm1, xmm1)
			vmovhpd(mem(r14, r10, 1), xmm1, xmm1)
			vfmadd231ps(xmm1, xmm3, xmm2)
			vmovlpd(xmm2, mem(r14, r13, 2)) // store ( gamma46..gamma56 )
			vmovhpd(xmm2, mem(r14, r10, 1)) // store ( gamma47..gamma57 )

			lea(mem(r14, rsi, 8), r14) // r14 += 8*cs_c



			vunpcklps(ymm7, ymm5, ymm0)
			vunpcklps(ymm11, ymm9, ymm1)
			vshufps(imm(0x4e), ymm1, ymm0, ymm2)
			vblendps(imm(0xcc), ymm2, ymm0, ymm0)
			vblendps(imm(0x33), ymm2, ymm1, ymm1)

			vextractf128(imm(0x1), ymm0, xmm2)
			vfmadd231ps(mem(rcx), xmm3, xmm0)
			vfmadd231ps(mem(rcx, rsi, 4), xmm3, xmm2)
			vmovups(xmm0, mem(rcx)) // store ( gamma00..gamma30 )
			vmovups(xmm2, mem(rcx, rsi, 4)) // store ( gamma04..gamma34 )

			vextractf128(imm(0x1), ymm1, xmm2)
			vfmadd231ps(mem(rcx, rsi, 1), xmm3, xmm1)
			vfmadd231ps(mem(rcx, r15, 1), xmm3, xmm2)
			vmovups(xmm1, mem(rcx, rsi, 1)) // store ( gamma01..gamma31 )
			vmovups(xmm2, mem(rcx, r15, 1)) // store ( gamma05..gamma35 )


			vunpckhps(ymm7, ymm5, ymm0)
			vunpckhps(ymm11, ymm9, ymm1)
			vshufps(imm(0x4e), ymm1, ymm0, ymm2)
			vblendps(imm(0xcc), ymm2, ymm0, ymm0)
			vblendps(imm(0x33), ymm2, ymm1, ymm1)

			vextractf128(imm(0x1), ymm0, xmm2)
			vfmadd231ps(mem(rcx, rsi, 2), xmm3, xmm0)
			vfmadd231ps(mem(rcx, r13, 2), xmm3, xmm2)
			vmovups(xmm0, mem(rcx, rsi, 2)) // store ( gamma02..gamma32 )
			vmovups(xmm2, mem(rcx, r13, 2)) // store ( gamma06..gamma36 )

			vextractf128(imm(0x1), ymm1, xmm2)
			vfmadd231ps(mem(rcx, r13, 1), xmm3, xmm1)
			vfmadd231ps(mem(rcx, r10, 1), xmm3, xmm2)
			vmovups(xmm1, mem(rcx, r13, 1)) // store ( gamma03..gamma33 )
			vmovups(xmm2, mem(rcx, r10, 1)) // store ( gamma07..gamma37 )

			//lea(mem(rcx, rsi, 8), rcx) // rcx += 8*cs_c

			vunpcklps(ymm15, ymm13, ymm0)
			vextractf128(imm(0x1), ymm0, xmm2)
			vmovlpd(mem(r14), xmm1, xmm1)
			vmovhpd(mem(r14, rsi, 1), xmm1, xmm1)
			vfmadd231ps(xmm1, xmm3, xmm0)
			vmovlpd(xmm0, mem(r14)) // store ( gamma40..gamma50 )
			vmovhpd(xmm0, mem(r14, rsi, 1)) // store ( gamma41..gamma51 )
			vmovlpd(mem(r14, rsi, 4), xmm1, xmm1)
			vmovhpd(mem(r14, r15, 1), xmm1, xmm1)
			vfmadd231ps(xmm1, xmm3, xmm2)
			vmovlpd(xmm2, mem(r14, rsi, 4)) // store ( gamma44..gamma54 )
			vmovhpd(xmm2, mem(r14, r15, 1)) // store ( gamma45..gamma55 )

			vunpckhps(ymm15, ymm13, ymm0)
			vextractf128(imm(0x1), ymm0, xmm2)
			vmovlpd(mem(r14, rsi, 2), xmm1, xmm1)
			vmovhpd(mem(r14, r13, 1), xmm1, xmm1)
			vfmadd231ps(xmm1, xmm3, xmm0)
			vmovlpd(xmm0, mem(r14, rsi, 2)) // store ( gamma42..gamma52 )
			vmovhpd(xmm0, mem(r14, r13, 1)) // store ( gamma43..gamma53 )
			vmovlpd(mem(r14, r13, 2), xmm1, xmm1)
			vmovhpd(mem(r14, r10, 1), xmm1, xmm1)
			vfmadd231ps(xmm1, xmm3, xmm2)
			vmovlpd(xmm2, mem(r14, r13, 2)) // store ( gamma46..gamma56 )
			vmovhpd(xmm2, mem(r14, r10, 1)) // store ( gamma47..gamma57 )

			//lea(mem(r14, rsi, 8), r14) // r14 += 8*cs_c

			jmp(.SDONE) // jump to end.

	label(.SBETAZERO)

		cmp(imm(4), rdi) // set ZF if (4*cs_c) == 4.
		jz(.SCOLSTORBZ) // jump to column storage case

			vmovups(ymm4, mem(rcx))
			vmovups(ymm5, mem(rcx,32))
			add(rdi, rcx)

			vmovups(ymm6, mem(rcx))
			vmovups(ymm7, mem(rcx,32))
			add(rdi, rcx)


			vmovups(ymm8, mem(rcx))
			vmovups(ymm9, mem(rcx,32))
			add(rdi, rcx)


			vmovups(ymm10, mem(rcx))
			vmovups(ymm11, mem(rcx,32))
			add(rdi, rcx)


			vmovups(ymm12, mem(rcx))
			vmovups(ymm13, mem(rcx,32))
			add(rdi, rcx)


			vmovups(ymm14, mem(rcx))
			vmovups(ymm15, mem(rcx,32))
			//add(rdi, rcx)

			jmp(.SDONE) // jump to end.

		label(.SCOLSTORBZ)

			vunpcklps(ymm6, ymm4, ymm0)
			vunpcklps(ymm10, ymm8, ymm1)
			vshufps(imm(0x4e), ymm1, ymm0, ymm2)
			vblendps(imm(0xcc), ymm2, ymm0, ymm0)
			vblendps(imm(0x33), ymm2, ymm1, ymm1)

			vextractf128(imm(0x1), ymm0, xmm2)
			vmovups(xmm0, mem(rcx)) // store ( gamma00..gamma30 )
			vmovups(xmm2, mem(rcx, rsi, 4)) // store ( gamma04..gamma34 )

			vextractf128(imm(0x1), ymm1, xmm2)
			vmovups(xmm1, mem(rcx, rsi, 1)) // store ( gamma01..gamma31 )
			vmovups(xmm2, mem(rcx, r15, 1)) // store ( gamma05..gamma35 )


			vunpckhps(ymm6, ymm4, ymm0)
			vunpckhps(ymm10, ymm8, ymm1)
			vshufps(imm(0x4e), ymm1, ymm0, ymm2)
			vblendps(imm(0xcc), ymm2, ymm0, ymm0)
			vblendps(imm(0x33), ymm2, ymm1, ymm1)

			vextractf128(imm(0x1), ymm0, xmm2)
			vmovups(xmm0, mem(rcx, rsi, 2)) // store ( gamma02..gamma32 )
			vmovups(xmm2, mem(rcx, r13, 2)) // store ( gamma06..gamma36 )

			vextractf128(imm(0x1), ymm1, xmm2)
			vmovups(xmm1, mem(rcx, r13, 1)) // store ( gamma03..gamma33 )
			vmovups(xmm2, mem(rcx, r10, 1)) // store ( gamma07..gamma37 )

			lea(mem(rcx, rsi, 8), rcx) // rcx += 8*cs_c

			vunpcklps(ymm14, ymm12, ymm0)
			vextractf128(imm(0x1), ymm0, xmm2)
			vmovlpd(xmm0, mem(r14)) // store ( gamma40..gamma50 )
			vmovhpd(xmm0, mem(r14, rsi, 1)) // store ( gamma41..gamma51 )
			vmovlpd(xmm2, mem(r14, rsi, 4)) // store ( gamma44..gamma54 )
			vmovhpd(xmm2, mem(r14, r15, 1)) // store ( gamma45..gamma55 )

			vunpckhps(ymm14, ymm12, ymm0)
			vextractf128(imm(0x1), ymm0, xmm2)
			vmovlpd(xmm0, mem(r14, rsi, 2)) // store ( gamma42..gamma52 )
			vmovhpd(xmm0, mem(r14, r13, 1)) // store ( gamma43..gamma53 )
			vmovlpd(xmm2, mem(r14, r13, 2)) // store ( gamma46..gamma56 )
			vmovhpd(xmm2, mem(r14, r10, 1)) // store ( gamma47..gamma57 )

			lea(mem(r14, rsi, 8), r14) // r14 += 8*cs_c



			vunpcklps(ymm7, ymm5, ymm0)
			vunpcklps(ymm11, ymm9, ymm1)
			vshufps(imm(0x4e), ymm1, ymm0, ymm2)
			vblendps(imm(0xcc), ymm2, ymm0, ymm0)
			vblendps(imm(0x33), ymm2, ymm1, ymm1)

			vextractf128(imm(0x1), ymm0, xmm2)
			vmovups(xmm0, mem(rcx)) // store ( gamma00..gamma30 )
			vmovups(xmm2, mem(rcx, rsi, 4)) // store ( gamma04..gamma34 )

			vextractf128(imm(0x1), ymm1, xmm2)
			vmovups(xmm1, mem(rcx, rsi, 1)) // store ( gamma01..gamma31 )
			vmovups(xmm2, mem(rcx, r15, 1)) // store ( gamma05..gamma35 )


			vunpckhps(ymm7, ymm5, ymm0)
			vunpckhps(ymm11, ymm9, ymm1)
			vshufps(imm(0x4e), ymm1, ymm0, ymm2)
			vblendps(imm(0xcc), ymm2, ymm0, ymm0)
			vblendps(imm(0x33), ymm2, ymm1, ymm1)

			vextractf128(imm(0x1), ymm0, xmm2)
			vmovups(xmm0, mem(rcx, rsi, 2)) // store ( gamma02..gamma32 )
			vmovups(xmm2, mem(rcx, r13, 2)) // store ( gamma06..gamma36 )

			vextractf128(imm(0x1), ymm1, xmm2)
			vmovups(xmm1, mem(rcx, r13, 1)) // store ( gamma03..gamma33 )
			vmovups(xmm2, mem(rcx, r10, 1)) // store ( gamma07..gamma37 )

			//lea(mem(rcx, rsi, 8), rcx) // rcx += 8*cs_c

			vunpcklps(ymm15, ymm13, ymm0)
			vextractf128(imm(0x1), ymm0, xmm2)
			vmovlpd(xmm0, mem(r14)) // store ( gamma40..gamma50 )
			vmovhpd(xmm0, mem(r14, rsi, 1)) // store ( gamma41..gamma51 )
			vmovlpd(xmm2, mem(r14, rsi, 4)) // store ( gamma44..gamma54 )
			vmovhpd(xmm2, mem(r14, r15, 1)) // store ( gamma45..gamma55 )

			vunpckhps(ymm15, ymm13, ymm0)
			vextractf128(imm(0x1), ymm0, xmm2)
			vmovlpd(xmm0, mem(r14, rsi, 2)) // store ( gamma42..gamma52 )
			vmovhpd(xmm0, mem(r14, r13, 1)) // store ( gamma43..gamma53 )
			vmovlpd(xmm2, mem(r14, r13, 2)) // store ( gamma46..gamma56 )
			vmovhpd(xmm2, mem(r14, r10, 1)) // store ( gamma47..gamma57 )

			//lea(mem(r14, rsi, 8), r14) // r14 += 8*cs_

	label(.SDONE)


	vzeroupper()



	end_asm(
	: // output operands (none)
	: // input operands
	  [k_iter] "m" (k_iter), // 0
	  [k_left] "m" (k_left), // 1
	  [a]      "m" (a),      // 2
	  [b]      "m" (b),      // 3
	  [alpha]  "m" (alpha),  // 4
	  [beta]   "m" (beta),   // 5
	  [c]      "m" (c),      // 6
	  [rs_c]   "m" (rs_c),   // 7
	  [cs_c]   "m" (cs_c)/*,   // 8
	  [b_next] "m" (b_next), // 9
	  [a_next] "m" (a_next)*/  // 10
	: // register clobber list
	  "rax", "rbx", "rcx", "rdx", "rsi", "rdi",
	  "r8", "r9", "r10", "r11", "r12", "r13", "r14", "r15",
	  "xmm0", "xmm1", "xmm2", "xmm3",
	  "xmm4", "xmm5", "xmm6", "xmm7",
	  "xmm8", "xmm9", "xmm10", "xmm11",
	  "xmm12", "xmm13", "xmm14", "xmm15",
	  "memory"
	)

	GEMM_UKR_FLUSH_CT( s );
}




#define DGEMM_INPUT_GS_BETA_NZ \
	vmovlpd(mem(rcx), xmm0, xmm0) \
	vmovhpd(mem(rcx, rsi, 1), xmm0, xmm0) \
	vmovlpd(mem(rcx, rsi, 2), xmm1, xmm1) \
	vmovhpd(mem(rcx, r13, 1), xmm1, xmm1) \
	vperm2f128(imm(0x20), ymm1, ymm0, ymm0) /*\
	vmovlpd(mem(rcx, rsi, 4), xmm2, xmm2) \
	vmovhpd(mem(rcx, r15, 1), xmm2, xmm2) \
	vmovlpd(mem(rcx, r13, 2), xmm1, xmm1) \
	vmovhpd(mem(rcx, r10, 1), xmm1, xmm1) \
	vperm2f128(imm(0x20), ymm1, ymm2, ymm2)*/

#define DGEMM_OUTPUT_GS_BETA_NZ \
	vextractf128(imm(1), ymm0, xmm1) \
	vmovlpd(xmm0, mem(rcx)) \
	vmovhpd(xmm0, mem(rcx, rsi, 1)) \
	vmovlpd(xmm1, mem(rcx, rsi, 2)) \
	vmovhpd(xmm1, mem(rcx, r13, 1)) /*\
	vextractf128(imm(1), ymm2, xmm1) \
	vmovlpd(xmm2, mem(rcx, rsi, 4)) \
	vmovhpd(xmm2, mem(rcx, r15, 1)) \
	vmovlpd(xmm1, mem(rcx, r13, 2)) \
	vmovhpd(xmm1, mem(rcx, r10, 1))*/

void bli_dgemm_haswell_asm_6x8
     (
       dim_t               m,
       dim_t               n,
       dim_t               k,
       double*    restrict alpha,
       double*    restrict a,
       double*    restrict b,
       double*    restrict beta,
       double*    restrict c, inc_t rs_c0, inc_t cs_c0,
       auxinfo_t* restrict data,
       cntx_t*    restrict cntx
     )
{
	//void*   a_next = bli_auxinfo_next_a( data );
	//void*   b_next = bli_auxinfo_next_b( data );

	// Typecast local copies of integers in case dim_t and inc_t are a
	// different size than is expected by load instructions.
	uint64_t k_iter = k / 4;
	uint64_t k_left = k % 4;
	uint64_t rs_c   = rs_c0;
	uint64_t cs_c   = cs_c0;

	GEMM_UKR_SETUP_CT_AMBI( d, 6, 8, true );

	begin_asm()

	vzeroall() // zero all xmm/ymm registers.


	mov(var(a), rax) // load address of a.
	mov(var(b), rbx) // load address of b.
	//mov(%9, r15) // load address of b_next.

	add(imm(32*4), rbx)
	// initialize loop by pre-loading
	vmovapd(mem(rbx, -4*32), ymm0)
	vmovapd(mem(rbx, -3*32), ymm1)

	mov(var(c), rcx) // load address of c
	mov(var(rs_c), rdi) // load rs_c
	lea(mem(, rdi, 8), rdi) // rs_c *= sizeof(double)
	mov(var(cs_c), rsi) // load cs_c
	lea(mem(, rsi, 8), rsi) // cs_c *= sizeof(double)

	cmp(imm(8), rdi) // set ZF if (8*rs_c) == 8.
	jz(.DCOLPREFETCH) // jump to column prefetch case

		lea(mem(rdi, rdi, 2), r13) // r13 = 3*rs_c;
		lea(mem(rcx, r13, 1), rdx) // rdx = c + 3*rs_c;
		prefetch(0, mem(rcx, 7*8)) // prefetch c + 0*rs_c
		prefetch(0, mem(rcx, rdi, 1, 7*8)) // prefetch c + 1*rs_c
		prefetch(0, mem(rcx, rdi, 2, 7*8)) // prefetch c + 2*rs_c
		prefetch(0, mem(rdx, 7*8)) // prefetch c + 3*rs_c
		prefetch(0, mem(rdx, rdi, 1, 7*8)) // prefetch c + 4*rs_c
		prefetch(0, mem(rdx, rdi, 2, 7*8)) // prefetch c + 5*rs_c

		jmp(.DPREFETCHDONE)

	label(.DCOLPREFETCH)

		lea(mem(rsi, rsi, 2), r13) // r13 = 3*cs_c;
		lea(mem(rcx, r13, 1), rdx) // rdx = c + 3*cs_c;
		prefetch(0, mem(rcx, 7*8)) // prefetch c + 0*cs_c
		prefetch(0, mem(rcx, rsi, 1, 7*8)) // prefetch c + 1*cs_c
		prefetch(0, mem(rcx, rsi, 2, 7*8)) // prefetch c + 2*cs_c
		prefetch(0, mem(rdx, 7*8)) // prefetch c + 3*cs_c
		prefetch(0, mem(rdx, rsi, 1, 7*8)) // prefetch c + 4*cs_c
		prefetch(0, mem(rdx, rsi, 2, 7*8)) // prefetch c + 5*cs_c
		prefetch(0, mem(rdx, r13, 1, 7*8)) // prefetch c + 6*cs_c
		prefetch(0, mem(rdx, rsi, 4, 7*8)) // prefetch c + 7*cs_c

	label(.DPREFETCHDONE)


	mov(var(k_iter), rsi) // i = k_iter;
	test(rsi, rsi) // check i via logical AND.
	je(.DCONSIDKLEFT) // if i == 0, jump to code that
	// contains the k_left loop.


	label(.DLOOPKITER) // MAIN LOOP


	// iteration 0
	prefetch(0, mem(rax, 64*8))

	vbroadcastsd(mem(rax, 0*8), ymm2)
	vbroadcastsd(mem(rax, 1*8), ymm3)
	vfmadd231pd(ymm0, ymm2, ymm4)
	vfmadd231pd(ymm1, ymm2, ymm5)
	vfmadd231pd(ymm0, ymm3, ymm6)
	vfmadd231pd(ymm1, ymm3, ymm7)

	vbroadcastsd(mem(rax, 2*8), ymm2)
	vbroadcastsd(mem(rax, 3*8), ymm3)
	vfmadd231pd(ymm0, ymm2, ymm8)
	vfmadd231pd(ymm1, ymm2, ymm9)
	vfmadd231pd(ymm0, ymm3, ymm10)
	vfmadd231pd(ymm1, ymm3, ymm11)

	vbroadcastsd(mem(rax, 4*8), ymm2)
	vbroadcastsd(mem(rax, 5*8), ymm3)
	vfmadd231pd(ymm0, ymm2, ymm12)
	vfmadd231pd(ymm1, ymm2, ymm13)
	vfmadd231pd(ymm0, ymm3, ymm14)
	vfmadd231pd(ymm1, ymm3, ymm15)

	vmovapd(mem(rbx, -2*32), ymm0)
	vmovapd(mem(rbx, -1*32), ymm1)

	// iteration 1
	prefetch(0, mem(rax, 72*8))

	vbroadcastsd(mem(rax, 6*8), ymm2)
	vbroadcastsd(mem(rax, 7*8), ymm3)
	vfmadd231pd(ymm0, ymm2, ymm4)
	vfmadd231pd(ymm1, ymm2, ymm5)
	vfmadd231pd(ymm0, ymm3, ymm6)
	vfmadd231pd(ymm1, ymm3, ymm7)

	vbroadcastsd(mem(rax, 8*8), ymm2)
	vbroadcastsd(mem(rax, 9*8), ymm3)
	vfmadd231pd(ymm0, ymm2, ymm8)
	vfmadd231pd(ymm1, ymm2, ymm9)
	vfmadd231pd(ymm0, ymm3, ymm10)
	vfmadd231pd(ymm1, ymm3, ymm11)

	vbroadcastsd(mem(rax, 10*8), ymm2)
	vbroadcastsd(mem(rax, 11*8), ymm3)
	vfmadd231pd(ymm0, ymm2, ymm12)
	vfmadd231pd(ymm1, ymm2, ymm13)
	vfmadd231pd(ymm0, ymm3, ymm14)
	vfmadd231pd(ymm1, ymm3, ymm15)

	vmovapd(mem(rbx, 0*32), ymm0)
	vmovapd(mem(rbx, 1*32), ymm1)

	// iteration 2
	prefetch(0, mem(rax, 80*8))

	vbroadcastsd(mem(rax, 12*8), ymm2)
	vbroadcastsd(mem(rax, 13*8), ymm3)
	vfmadd231pd(ymm0, ymm2, ymm4)
	vfmadd231pd(ymm1, ymm2, ymm5)
	vfmadd231pd(ymm0, ymm3, ymm6)
	vfmadd231pd(ymm1, ymm3, ymm7)

	vbroadcastsd(mem(rax, 14*8), ymm2)
	vbroadcastsd(mem(rax, 15*8), ymm3)
	vfmadd231pd(ymm0, ymm2, ymm8)
	vfmadd231pd(ymm1, ymm2, ymm9)
	vfmadd231pd(ymm0, ymm3, ymm10)
	vfmadd231pd(ymm1, ymm3, ymm11)

	vbroadcastsd(mem(rax, 16*8), ymm2)
	vbroadcastsd(mem(rax, 17*8), ymm3)
	vfmadd231pd(ymm0, ymm2, ymm12)
	vfmadd231pd(ymm1, ymm2, ymm13)
	vfmadd231pd(ymm0, ymm3, ymm14)
	vfmadd231pd(ymm1, ymm3, ymm15)

	vmovapd(mem(rbx, 2*32), ymm0)
	vmovapd(mem(rbx, 3*32), ymm1)

	// iteration 3
	vbroadcastsd(mem(rax, 18*8), ymm2)
	vbroadcastsd(mem(rax, 19*8), ymm3)
	vfmadd231pd(ymm0, ymm2, ymm4)
	vfmadd231pd(ymm1, ymm2, ymm5)
	vfmadd231pd(ymm0, ymm3, ymm6)
	vfmadd231pd(ymm1, ymm3, ymm7)

	vbroadcastsd(mem(rax, 20*8), ymm2)
	vbroadcastsd(mem(rax, 21*8), ymm3)
	vfmadd231pd(ymm0, ymm2, ymm8)
	vfmadd231pd(ymm1, ymm2, ymm9)
	vfmadd231pd(ymm0, ymm3, ymm10)
	vfmadd231pd(ymm1, ymm3, ymm11)

	vbroadcastsd(mem(rax, 22*8), ymm2)
	vbroadcastsd(mem(rax, 23*8), ymm3)
	vfmadd231pd(ymm0, ymm2, ymm12)
	vfmadd231pd(ymm1, ymm2, ymm13)
	vfmadd231pd(ymm0, ymm3, ymm14)
	vfmadd231pd(ymm1, ymm3, ymm15)

	add(imm(4*6*8), rax) // a += 4*6 (unroll x mr)
	add(imm(4*8*8), rbx) // b += 4*8 (unroll x nr)

	vmovapd(mem(rbx, -4*32), ymm0)
	vmovapd(mem(rbx, -3*32), ymm1)


	dec(rsi) // i -= 1;
	jne(.DLOOPKITER) // iterate again if i != 0.






	label(.DCONSIDKLEFT)

	mov(var(k_left), rsi) // i = k_left;
	test(rsi, rsi) // check i via logical AND.
	je(.DPOSTACCUM) // if i == 0, we're done; jump to end.
	// else, we prepare to enter k_left loop.


	label(.DLOOPKLEFT) // EDGE LOOP

	prefetch(0, mem(rax, 64*8))

	vbroadcastsd(mem(rax, 0*8), ymm2)
	vbroadcastsd(mem(rax, 1*8), ymm3)
	vfmadd231pd(ymm0, ymm2, ymm4)
	vfmadd231pd(ymm1, ymm2, ymm5)
	vfmadd231pd(ymm0, ymm3, ymm6)
	vfmadd231pd(ymm1, ymm3, ymm7)

	vbroadcastsd(mem(rax, 2*8), ymm2)
	vbroadcastsd(mem(rax, 3*8), ymm3)
	vfmadd231pd(ymm0, ymm2, ymm8)
	vfmadd231pd(ymm1, ymm2, ymm9)
	vfmadd231pd(ymm0, ymm3, ymm10)
	vfmadd231pd(ymm1, ymm3, ymm11)

	vbroadcastsd(mem(rax, 4*8), ymm2)
	vbroadcastsd(mem(rax, 5*8), ymm3)
	vfmadd231pd(ymm0, ymm2, ymm12)
	vfmadd231pd(ymm1, ymm2, ymm13)
	vfmadd231pd(ymm0, ymm3, ymm14)
	vfmadd231pd(ymm1, ymm3, ymm15)

	add(imm(1*6*8), rax) // a += 1*6 (unroll x mr)
	add(imm(1*8*8), rbx) // b += 1*8 (unroll x nr)

	vmovapd(mem(rbx, -4*32), ymm0)
	vmovapd(mem(rbx, -3*32), ymm1)


	dec(rsi) // i -= 1;
	jne(.DLOOPKLEFT) // iterate again if i != 0.



	label(.DPOSTACCUM)




	mov(var(alpha), rax) // load address of alpha
	mov(var(beta), rbx) // load address of beta
	vbroadcastsd(mem(rax), ymm0) // load alpha and duplicate
	vbroadcastsd(mem(rbx), ymm3) // load beta and duplicate

	vmulpd(ymm0, ymm4, ymm4) // scale by alpha
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






	mov(var(cs_c), rsi) // load cs_c
	lea(mem(, rsi, 8), rsi) // rsi = cs_c * sizeof(double)

	lea(mem(rcx, rsi, 4), rdx) // load address of c +  4*cs_c;
	lea(mem(rcx, rdi, 4), r14) // load address of c +  4*rs_c;

	lea(mem(rsi, rsi, 2), r13) // r13 = 3*cs_c;
	//lea(mem(rsi, rsi, 4), r15) // r15 = 5*cs_c;
	//lea(mem(r13, rsi, 4), r10) // r10 = 7*cs_c;


	// now avoid loading C if beta == 0

	vxorpd(ymm0, ymm0, ymm0) // set ymm0 to zero.
	vucomisd(xmm0, xmm3) // set ZF if beta == 0.
	je(.DBETAZERO) // if ZF = 1, jump to beta == 0 case

		cmp(imm(8), rdi) // set ZF if (8*rs_c) == 8.
		jz(.DCOLSTORED) // jump to column storage case

			vfmadd231pd(mem(rcx), ymm3, ymm4)
			vmovupd(ymm4, mem(rcx))
			vfmadd231pd(mem(rcx,32), ymm3, ymm5)
			vmovupd(ymm5, mem(rcx,32))
			add(rdi, rcx)


			vfmadd231pd(mem(rcx), ymm3, ymm6)
			vmovupd(ymm6, mem(rcx))
			vfmadd231pd(mem(rcx,32), ymm3, ymm7)
			vmovupd(ymm7, mem(rcx,32))
			add(rdi, rcx)


			vfmadd231pd(mem(rcx), ymm3, ymm8)
			vmovupd(ymm8, mem(rcx))
			vfmadd231pd(mem(rcx,32), ymm3, ymm9)
			vmovupd(ymm9, mem(rcx,32))
			add(rdi, rcx)


			vfmadd231pd(mem(rcx), ymm3, ymm10)
			vmovupd(ymm10, mem(rcx))
			vfmadd231pd(mem(rcx,32), ymm3, ymm11)
			vmovupd(ymm11, mem(rcx,32))
			add(rdi, rcx)


			vfmadd231pd(mem(rcx), ymm3, ymm12)
			vmovupd(ymm12, mem(rcx))
			vfmadd231pd(mem(rcx,32), ymm3, ymm13)
			vmovupd(ymm13, mem(rcx,32))
			add(rdi, rcx)


			vfmadd231pd(mem(rcx), ymm3, ymm14)
			vmovupd(ymm14, mem(rcx))
			vfmadd231pd(mem(rcx,32), ymm3, ymm15)
			vmovupd(ymm15, mem(rcx,32))
			//add(rdi, rcx)

			jmp(.DDONE) // jump to end.

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
			vfmadd231pd(mem(rcx, r13, 1), ymm3, ymm10)
			vmovupd(ymm4, mem(rcx))
			vmovupd(ymm6, mem(rcx, rsi, 1))
			vmovupd(ymm8, mem(rcx, rsi, 2))
			vmovupd(ymm10, mem(rcx, r13, 1))

			lea(mem(rcx, rsi, 4), rcx)

			vunpcklpd(ymm14, ymm12, ymm0)
			vunpckhpd(ymm14, ymm12, ymm1)
			vextractf128(imm(0x1), ymm0, xmm2)
			vextractf128(imm(0x1), ymm1, xmm4)

			vfmadd231pd(mem(r14), xmm3, xmm0)
			vfmadd231pd(mem(r14, rsi, 1), xmm3, xmm1)
			vfmadd231pd(mem(r14, rsi, 2), xmm3, xmm2)
			vfmadd231pd(mem(r14, r13, 1), xmm3, xmm4)
			vmovupd(xmm0, mem(r14))
			vmovupd(xmm1, mem(r14, rsi, 1))
			vmovupd(xmm2, mem(r14, rsi, 2))
			vmovupd(xmm4, mem(r14, r13, 1))

			lea(mem(r14, rsi, 4), r14)


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
			vfmadd231pd(mem(rcx, r13, 1), ymm3, ymm11)
			vmovupd(ymm5, mem(rcx))
			vmovupd(ymm7, mem(rcx, rsi, 1))
			vmovupd(ymm9, mem(rcx, rsi, 2))
			vmovupd(ymm11, mem(rcx, r13, 1))

			//lea(mem(rcx, rsi, 4), rcx)

			vunpcklpd(ymm15, ymm13, ymm0)
			vunpckhpd(ymm15, ymm13, ymm1)
			vextractf128(imm(0x1), ymm0, xmm2)
			vextractf128(imm(0x1), ymm1, xmm4)

			vfmadd231pd(mem(r14), xmm3, xmm0)
			vfmadd231pd(mem(r14, rsi, 1), xmm3, xmm1)
			vfmadd231pd(mem(r14, rsi, 2), xmm3, xmm2)
			vfmadd231pd(mem(r14, r13, 1), xmm3, xmm4)
			vmovupd(xmm0, mem(r14))
			vmovupd(xmm1, mem(r14, rsi, 1))
			vmovupd(xmm2, mem(r14, rsi, 2))
			vmovupd(xmm4, mem(r14, r13, 1))

			//lea(mem(r14, rsi, 4), r14)

			jmp(.DDONE) // jump to end.

	label(.DBETAZERO)

		cmp(imm(8), rdi) // set ZF if (8*rs_c) == 8.
		jz(.DCOLSTORBZ) // jump to column storage case

			vmovupd(ymm4, mem(rcx))
			vmovupd(ymm5, mem(rcx,32))
			add(rdi, rcx)

			vmovupd(ymm6, mem(rcx))
			vmovupd(ymm7, mem(rcx,32))
			add(rdi, rcx)


			vmovupd(ymm8, mem(rcx))
			vmovupd(ymm9, mem(rcx,32))
			add(rdi, rcx)


			vmovupd(ymm10, mem(rcx))
			vmovupd(ymm11, mem(rcx,32))
			add(rdi, rcx)


			vmovupd(ymm12, mem(rcx))
			vmovupd(ymm13, mem(rcx,32))
			add(rdi, rcx)


			vmovupd(ymm14, mem(rcx))
			vmovupd(ymm15, mem(rcx,32))
			//add(rdi, rcx)

			jmp(.DDONE) // jump to end.

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
			vmovupd(ymm10, mem(rcx, r13, 1))

			lea(mem(rcx, rsi, 4), rcx)

			vunpcklpd(ymm14, ymm12, ymm0)
			vunpckhpd(ymm14, ymm12, ymm1)
			vextractf128(imm(0x1), ymm0, xmm2)
			vextractf128(imm(0x1), ymm1, xmm4)

			vmovupd(xmm0, mem(r14))
			vmovupd(xmm1, mem(r14, rsi, 1))
			vmovupd(xmm2, mem(r14, rsi, 2))
			vmovupd(xmm4, mem(r14, r13, 1))

			lea(mem(r14, rsi, 4), r14)


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
			vmovupd(ymm11, mem(rcx, r13, 1))

			//lea(mem(rcx, rsi, 4), rcx)

			vunpcklpd(ymm15, ymm13, ymm0)
			vunpckhpd(ymm15, ymm13, ymm1)
			vextractf128(imm(0x1), ymm0, xmm2)
			vextractf128(imm(0x1), ymm1, xmm4)

			vmovupd(xmm0, mem(r14))
			vmovupd(xmm1, mem(r14, rsi, 1))
			vmovupd(xmm2, mem(r14, rsi, 2))
			vmovupd(xmm4, mem(r14, r13, 1))

			//lea(mem(r14, rsi, 4), r14)

	label(.DDONE)


	vzeroupper()



	end_asm(
	: // output operands (none)
	: // input operands
	  [k_iter] "m" (k_iter), // 0
	  [k_left] "m" (k_left), // 1
	  [a]      "m" (a),      // 2
	  [b]      "m" (b),      // 3
	  [alpha]  "m" (alpha),  // 4
	  [beta]   "m" (beta),   // 5
	  [c]      "m" (c),      // 6
	  [rs_c]   "m" (rs_c),   // 7
	  [cs_c]   "m" (cs_c)/*,   // 8
	  [b_next] "m" (b_next), // 9
	  [a_next] "m" (a_next)*/  // 10
	: // register clobber list
	  "rax", "rbx", "rcx", "rdx", "rsi", "rdi",
	  "r8", "r9", "r10", "r11", "r12", "r13", "r14", "r15",
	  "xmm0", "xmm1", "xmm2", "xmm3",
	  "xmm4", "xmm5", "xmm6", "xmm7",
	  "xmm8", "xmm9", "xmm10", "xmm11",
	  "xmm12", "xmm13", "xmm14", "xmm15",
	  "memory"
	)

	GEMM_UKR_FLUSH_CT( d );
}





#define CGEMM_INPUT_SCALE_RS_BETA_NZ(where) \
	vmovups(where, ymm0) \
	vpermilps(imm(0xb1), ymm0, ymm3) \
	vmulps(ymm1, ymm0, ymm0) \
	vmulps(ymm2, ymm3, ymm3) \
	vaddsubps(ymm3, ymm0, ymm0)

void bli_cgemm_haswell_asm_3x8
     (
       dim_t               m,
       dim_t               n,
       dim_t               k,
       scomplex*  restrict alpha,
       scomplex*  restrict a,
       scomplex*  restrict b,
       scomplex*  restrict beta,
       scomplex*  restrict c, inc_t rs_c0, inc_t cs_c0,
       auxinfo_t* restrict data,
       cntx_t*    restrict cntx
     )
{
	//void*   a_next = bli_auxinfo_next_a( data );
	//void*   b_next = bli_auxinfo_next_b( data );

	// Typecast local copies of integers in case dim_t and inc_t are a
	// different size than is expected by load instructions.
	uint64_t k_iter = k / 4;
	uint64_t k_left = k % 4;
	uint64_t rs_c   = rs_c0;
	uint64_t cs_c   = cs_c0;

	GEMM_UKR_SETUP_CT( c, 3, 8, true );

	begin_asm()

	vzeroall() // zero all xmm/ymm registers.


	mov(var(a), rax) // load address of a.
	mov(var(b), rbx) // load address of b.
	//mov(%9, r15) // load address of b_next.

	add(imm(32*4), rbx)
	// initialize loop by pre-loading
	vmovaps(mem(rbx, -4*32), ymm0)
	vmovaps(mem(rbx, -3*32), ymm1)

	mov(var(c), rcx) // load address of c
	mov(var(rs_c), rdi) // load rs_c
	lea(mem(, rdi, 8), rdi) // rs_c *= sizeof(scomplex)

	lea(mem(rcx, rdi, 1), r11) // r11 = c + 1*rs_c;
	lea(mem(rcx, rdi, 2), r12) // r12 = c + 2*rs_c;

	prefetch(0, mem(rcx, 7*8)) // prefetch c + 0*rs_c
	prefetch(0, mem(r11, 7*8)) // prefetch c + 1*rs_c
	prefetch(0, mem(r12, 7*8)) // prefetch c + 2*rs_c




	mov(var(k_iter), rsi) // i = k_iter;
	test(rsi, rsi) // check i via logical AND.
	je(.CCONSIDKLEFT) // if i == 0, jump to code that
	// contains the k_left loop.


	label(.CLOOPKITER) // MAIN LOOP


	// iteration 0
	prefetch(0, mem(rax, 32*8))

	vbroadcastss(mem(rax, 0*4), ymm2)
	vbroadcastss(mem(rax, 1*4), ymm3)
	vfmadd231ps(ymm0, ymm2, ymm4)
	vfmadd231ps(ymm1, ymm2, ymm5)
	vfmadd231ps(ymm0, ymm3, ymm6)
	vfmadd231ps(ymm1, ymm3, ymm7)

	vbroadcastss(mem(rax, 2*4), ymm2)
	vbroadcastss(mem(rax, 3*4), ymm3)
	vfmadd231ps(ymm0, ymm2, ymm8)
	vfmadd231ps(ymm1, ymm2, ymm9)
	vfmadd231ps(ymm0, ymm3, ymm10)
	vfmadd231ps(ymm1, ymm3, ymm11)

	vbroadcastss(mem(rax, 4*4), ymm2)
	vbroadcastss(mem(rax, 5*4), ymm3)
	vfmadd231ps(ymm0, ymm2, ymm12)
	vfmadd231ps(ymm1, ymm2, ymm13)
	vfmadd231ps(ymm0, ymm3, ymm14)
	vfmadd231ps(ymm1, ymm3, ymm15)

	vmovaps(mem(rbx, -2*32), ymm0)
	vmovaps(mem(rbx, -1*32), ymm1)

	// iteration 1
	vbroadcastss(mem(rax, 6*4), ymm2)
	vbroadcastss(mem(rax, 7*4), ymm3)
	vfmadd231ps(ymm0, ymm2, ymm4)
	vfmadd231ps(ymm1, ymm2, ymm5)
	vfmadd231ps(ymm0, ymm3, ymm6)
	vfmadd231ps(ymm1, ymm3, ymm7)

	vbroadcastss(mem(rax, 8*4), ymm2)
	vbroadcastss(mem(rax, 9*4), ymm3)
	vfmadd231ps(ymm0, ymm2, ymm8)
	vfmadd231ps(ymm1, ymm2, ymm9)
	vfmadd231ps(ymm0, ymm3, ymm10)
	vfmadd231ps(ymm1, ymm3, ymm11)

	vbroadcastss(mem(rax, 10*4), ymm2)
	vbroadcastss(mem(rax, 11*4), ymm3)
	vfmadd231ps(ymm0, ymm2, ymm12)
	vfmadd231ps(ymm1, ymm2, ymm13)
	vfmadd231ps(ymm0, ymm3, ymm14)
	vfmadd231ps(ymm1, ymm3, ymm15)

	vmovaps(mem(rbx, 0*32), ymm0)
	vmovaps(mem(rbx, 1*32), ymm1)

	// iteration 2
	prefetch(0, mem(rax, 38*8))

	vbroadcastss(mem(rax, 12*4), ymm2)
	vbroadcastss(mem(rax, 13*4), ymm3)
	vfmadd231ps(ymm0, ymm2, ymm4)
	vfmadd231ps(ymm1, ymm2, ymm5)
	vfmadd231ps(ymm0, ymm3, ymm6)
	vfmadd231ps(ymm1, ymm3, ymm7)

	vbroadcastss(mem(rax, 14*4), ymm2)
	vbroadcastss(mem(rax, 15*4), ymm3)
	vfmadd231ps(ymm0, ymm2, ymm8)
	vfmadd231ps(ymm1, ymm2, ymm9)
	vfmadd231ps(ymm0, ymm3, ymm10)
	vfmadd231ps(ymm1, ymm3, ymm11)

	vbroadcastss(mem(rax, 16*4), ymm2)
	vbroadcastss(mem(rax, 17*4), ymm3)
	vfmadd231ps(ymm0, ymm2, ymm12)
	vfmadd231ps(ymm1, ymm2, ymm13)
	vfmadd231ps(ymm0, ymm3, ymm14)
	vfmadd231ps(ymm1, ymm3, ymm15)

	vmovaps(mem(rbx, 2*32), ymm0)
	vmovaps(mem(rbx, 3*32), ymm1)

	// iteration 3
	vbroadcastss(mem(rax, 18*4), ymm2)
	vbroadcastss(mem(rax, 19*4), ymm3)
	vfmadd231ps(ymm0, ymm2, ymm4)
	vfmadd231ps(ymm1, ymm2, ymm5)
	vfmadd231ps(ymm0, ymm3, ymm6)
	vfmadd231ps(ymm1, ymm3, ymm7)

	vbroadcastss(mem(rax, 20*4), ymm2)
	vbroadcastss(mem(rax, 21*4), ymm3)
	vfmadd231ps(ymm0, ymm2, ymm8)
	vfmadd231ps(ymm1, ymm2, ymm9)
	vfmadd231ps(ymm0, ymm3, ymm10)
	vfmadd231ps(ymm1, ymm3, ymm11)

	vbroadcastss(mem(rax, 22*4), ymm2)
	vbroadcastss(mem(rax, 23*4), ymm3)
	vfmadd231ps(ymm0, ymm2, ymm12)
	vfmadd231ps(ymm1, ymm2, ymm13)
	vfmadd231ps(ymm0, ymm3, ymm14)
	vfmadd231ps(ymm1, ymm3, ymm15)

	add(imm(4*3*8), rax) // a += 4*3  (unroll x mr)
	add(imm(4*8*8), rbx) // b += 4*8  (unroll x nr)

	vmovaps(mem(rbx, -4*32), ymm0)
	vmovaps(mem(rbx, -3*32), ymm1)


	dec(rsi) // i -= 1;
	jne(.CLOOPKITER) // iterate again if i != 0.






	label(.CCONSIDKLEFT)

	mov(var(k_left), rsi) // i = k_left;
	test(rsi, rsi) // check i via logical AND.
	je(.CPOSTACCUM) // if i == 0, we're done; jump to end.
	// else, we prepare to enter k_left loop.


	label(.CLOOPKLEFT) // EDGE LOOP

	prefetch(0, mem(rax, 32*8))

	vbroadcastss(mem(rax, 0*4), ymm2)
	vbroadcastss(mem(rax, 1*4), ymm3)
	vfmadd231ps(ymm0, ymm2, ymm4)
	vfmadd231ps(ymm1, ymm2, ymm5)
	vfmadd231ps(ymm0, ymm3, ymm6)
	vfmadd231ps(ymm1, ymm3, ymm7)

	vbroadcastss(mem(rax, 2*4), ymm2)
	vbroadcastss(mem(rax, 3*4), ymm3)
	vfmadd231ps(ymm0, ymm2, ymm8)
	vfmadd231ps(ymm1, ymm2, ymm9)
	vfmadd231ps(ymm0, ymm3, ymm10)
	vfmadd231ps(ymm1, ymm3, ymm11)

	vbroadcastss(mem(rax, 4*4), ymm2)
	vbroadcastss(mem(rax, 5*4), ymm3)
	vfmadd231ps(ymm0, ymm2, ymm12)
	vfmadd231ps(ymm1, ymm2, ymm13)
	vfmadd231ps(ymm0, ymm3, ymm14)
	vfmadd231ps(ymm1, ymm3, ymm15)

	add(imm(1*3*8), rax) // a += 1*3  (unroll x mr)
	add(imm(1*8*8), rbx) // b += 1*8  (unroll x nr)

	vmovaps(mem(rbx, -4*32), ymm0)
	vmovaps(mem(rbx, -3*32), ymm1)


	dec(rsi) // i -= 1;
	jne(.CLOOPKLEFT) // iterate again if i != 0.



	label(.CPOSTACCUM)


	// permute even and odd elements
	// of ymm6/7, ymm10/11, ymm/14/15
	vpermilps(imm(0xb1), ymm6, ymm6)
	vpermilps(imm(0xb1), ymm7, ymm7)
	vpermilps(imm(0xb1), ymm10, ymm10)
	vpermilps(imm(0xb1), ymm11, ymm11)
	vpermilps(imm(0xb1), ymm14, ymm14)
	vpermilps(imm(0xb1), ymm15, ymm15)


	// subtract/add even/odd elements
	vaddsubps(ymm6, ymm4, ymm4)
	vaddsubps(ymm7, ymm5, ymm5)

	vaddsubps(ymm10, ymm8, ymm8)
	vaddsubps(ymm11, ymm9, ymm9)

	vaddsubps(ymm14, ymm12, ymm12)
	vaddsubps(ymm15, ymm13, ymm13)




	mov(var(alpha), rax) // load address of alpha
	vbroadcastss(mem(rax), ymm0) // load alpha_r and duplicate
	vbroadcastss(mem(rax, 4), ymm1) // load alpha_i and duplicate


	vpermilps(imm(0xb1), ymm4, ymm3)
	vmulps(ymm0, ymm4, ymm4)
	vmulps(ymm1, ymm3, ymm3)
	vaddsubps(ymm3, ymm4, ymm4)

	vpermilps(imm(0xb1), ymm5, ymm3)
	vmulps(ymm0, ymm5, ymm5)
	vmulps(ymm1, ymm3, ymm3)
	vaddsubps(ymm3, ymm5, ymm5)


	vpermilps(imm(0xb1), ymm8, ymm3)
	vmulps(ymm0, ymm8, ymm8)
	vmulps(ymm1, ymm3, ymm3)
	vaddsubps(ymm3, ymm8, ymm8)

	vpermilps(imm(0xb1), ymm9, ymm3)
	vmulps(ymm0, ymm9, ymm9)
	vmulps(ymm1, ymm3, ymm3)
	vaddsubps(ymm3, ymm9, ymm9)


	vpermilps(imm(0xb1), ymm12, ymm3)
	vmulps(ymm0, ymm12, ymm12)
	vmulps(ymm1, ymm3, ymm3)
	vaddsubps(ymm3, ymm12, ymm12)

	vpermilps(imm(0xb1), ymm13, ymm3)
	vmulps(ymm0, ymm13, ymm13)
	vmulps(ymm1, ymm3, ymm3)
	vaddsubps(ymm3, ymm13, ymm13)





	mov(var(beta), rbx) // load address of beta
	vbroadcastss(mem(rbx), ymm1) // load beta_r and duplicate
	vbroadcastss(mem(rbx, 4), ymm2) // load beta_i and duplicate


	// now avoid loading C if beta == 0
	vxorps(ymm0, ymm0, ymm0) // set ymm0 to zero.
	vucomiss(xmm0, xmm1) // set ZF if beta_r == 0.
	sete(r8b) // r8b = ( ZF == 1 ? 1 : 0 );
	vucomiss(xmm0, xmm2) // set ZF if beta_i == 0.
	sete(r9b) // r9b = ( ZF == 1 ? 1 : 0 );
	and(r8b, r9b) // set ZF if r8b & r9b == 1.
	jne(.CBETAZERO) // if ZF = 1, jump to beta == 0 case

		CGEMM_INPUT_SCALE_RS_BETA_NZ(mem(rcx))
		vaddps(ymm4, ymm0, ymm0)
		vmovups(ymm0, mem(rcx))


		CGEMM_INPUT_SCALE_RS_BETA_NZ(mem(rcx,32))
		vaddps(ymm5, ymm0, ymm0)
		vmovups(ymm0, mem(rcx,32))



		CGEMM_INPUT_SCALE_RS_BETA_NZ(mem(r11))
		vaddps(ymm8, ymm0, ymm0)
		vmovups(ymm0, mem(r11))


		CGEMM_INPUT_SCALE_RS_BETA_NZ(mem(r11,32))
		vaddps(ymm9, ymm0, ymm0)
		vmovups(ymm0, mem(r11,32))



		CGEMM_INPUT_SCALE_RS_BETA_NZ(mem(r12))
		vaddps(ymm12, ymm0, ymm0)
		vmovups(ymm0, mem(r12))


		CGEMM_INPUT_SCALE_RS_BETA_NZ(mem(r12,32))
		vaddps(ymm13, ymm0, ymm0)
		vmovups(ymm0, mem(r12,32))

		jmp(.CDONE) // jump to end.

	label(.CBETAZERO)

		vmovups(ymm4, mem(rcx))
		vmovups(ymm5, mem(rcx,32))

		vmovups(ymm8, mem(r11))
		vmovups(ymm9, mem(r11,32))

		vmovups(ymm12, mem(r12))
		vmovups(ymm13, mem(r12,32))

	label(.CDONE)


	vzeroupper()



	end_asm(
	: // output operands (none)
	: // input operands
	  [k_iter] "m" (k_iter), // 0
	  [k_left] "m" (k_left), // 1
	  [a]      "m" (a),      // 2
	  [b]      "m" (b),      // 3
	  [alpha]  "m" (alpha),  // 4
	  [beta]   "m" (beta),   // 5
	  [c]      "m" (c),      // 6
	  [rs_c]   "m" (rs_c),   // 7
	  [cs_c]   "m" (cs_c)/*,   // 8
	  [b_next] "m" (b_next), // 9
	  [a_next] "m" (a_next)*/  // 10
	: // register clobber list
	  "rax", "rbx", "rcx", "rdx", "rsi", "rdi",
	  "r8", "r9", "r10", "r11", "r12", "r13", "r14", "r15",
	  "xmm0", "xmm1", "xmm2", "xmm3",
	  "xmm4", "xmm5", "xmm6", "xmm7",
	  "xmm8", "xmm9", "xmm10", "xmm11",
	  "xmm12", "xmm13", "xmm14", "xmm15",
	  "memory"
	)

	GEMM_UKR_FLUSH_CT( c );
}




#define ZGEMM_INPUT_SCALE_RS_BETA_NZ(where) \
	vmovupd(where, ymm0) \
	vpermilpd(imm(0x5), ymm0, ymm3) \
	vmulpd(ymm1, ymm0, ymm0) \
	vmulpd(ymm2, ymm3, ymm3) \
	vaddsubpd(ymm3, ymm0, ymm0)

void bli_zgemm_haswell_asm_3x4
     (
       dim_t               m,
       dim_t               n,
       dim_t               k,
       dcomplex*  restrict alpha,
       dcomplex*  restrict a,
       dcomplex*  restrict b,
       dcomplex*  restrict beta,
       dcomplex*  restrict c, inc_t rs_c0, inc_t cs_c0,
       auxinfo_t* restrict data,
       cntx_t*    restrict cntx
     )
{
	//void*   a_next = bli_auxinfo_next_a( data );
	//void*   b_next = bli_auxinfo_next_b( data );

	// Typecast local copies of integers in case dim_t and inc_t are a
	// different size than is expected by load instructions.
	uint64_t k_iter = k / 4;
	uint64_t k_left = k % 4;
	uint64_t rs_c   = rs_c0;
	uint64_t cs_c   = cs_c0;

	GEMM_UKR_SETUP_CT( z, 3, 4, true );

	begin_asm()

	vzeroall() // zero all xmm/ymm registers.


	mov(var(a), rax) // load address of a.
	mov(var(b), rbx) // load address of b.
	//mov(%9, r15) // load address of b_next.

	add(imm(32*4), rbx)
	// initialize loop by pre-loading
	vmovapd(mem(rbx, -4*32), ymm0)
	vmovapd(mem(rbx, -3*32), ymm1)

	mov(var(c), rcx) // load address of c
	mov(var(rs_c), rdi) // load rs_c
	lea(mem(, rdi, 8), rdi) // rs_c *= sizeof(dcomplex)
	lea(mem(, rdi, 2), rdi)

	lea(mem(rcx, rdi, 1), r11) // r11 = c + 1*rs_c;
	lea(mem(rcx, rdi, 2), r12) // r12 = c + 2*rs_c;

	prefetch(0, mem(rcx, 7*8)) // prefetch c + 0*rs_c
	prefetch(0, mem(r11, 7*8)) // prefetch c + 1*rs_c
	prefetch(0, mem(r12, 7*8)) // prefetch c + 2*rs_c




	mov(var(k_iter), rsi) // i = k_iter;
	test(rsi, rsi) // check i via logical AND.
	je(.ZCONSIDKLEFT) // if i == 0, jump to code that
	// contains the k_left loop.


	label(.ZLOOPKITER) // MAIN LOOP


	// iteration 0
	prefetch(0, mem(rax, 32*16))

	vbroadcastsd(mem(rax, 0*8), ymm2)
	vbroadcastsd(mem(rax, 1*8), ymm3)
	vfmadd231pd(ymm0, ymm2, ymm4)
	vfmadd231pd(ymm1, ymm2, ymm5)
	vfmadd231pd(ymm0, ymm3, ymm6)
	vfmadd231pd(ymm1, ymm3, ymm7)

	vbroadcastsd(mem(rax, 2*8), ymm2)
	vbroadcastsd(mem(rax, 3*8), ymm3)
	vfmadd231pd(ymm0, ymm2, ymm8)
	vfmadd231pd(ymm1, ymm2, ymm9)
	vfmadd231pd(ymm0, ymm3, ymm10)
	vfmadd231pd(ymm1, ymm3, ymm11)

	vbroadcastsd(mem(rax, 4*8), ymm2)
	vbroadcastsd(mem(rax, 5*8), ymm3)
	vfmadd231pd(ymm0, ymm2, ymm12)
	vfmadd231pd(ymm1, ymm2, ymm13)
	vfmadd231pd(ymm0, ymm3, ymm14)
	vfmadd231pd(ymm1, ymm3, ymm15)

	vmovapd(mem(rbx, -2*32), ymm0)
	vmovapd(mem(rbx, -1*32), ymm1)

	// iteration 1
	prefetch(0, mem(rax, 36*16))

	vbroadcastsd(mem(rax, 6*8), ymm2)
	vbroadcastsd(mem(rax, 7*8), ymm3)
	vfmadd231pd(ymm0, ymm2, ymm4)
	vfmadd231pd(ymm1, ymm2, ymm5)
	vfmadd231pd(ymm0, ymm3, ymm6)
	vfmadd231pd(ymm1, ymm3, ymm7)

	vbroadcastsd(mem(rax, 8*8), ymm2)
	vbroadcastsd(mem(rax, 9*8), ymm3)
	vfmadd231pd(ymm0, ymm2, ymm8)
	vfmadd231pd(ymm1, ymm2, ymm9)
	vfmadd231pd(ymm0, ymm3, ymm10)
	vfmadd231pd(ymm1, ymm3, ymm11)

	vbroadcastsd(mem(rax, 10*8), ymm2)
	vbroadcastsd(mem(rax, 11*8), ymm3)
	vfmadd231pd(ymm0, ymm2, ymm12)
	vfmadd231pd(ymm1, ymm2, ymm13)
	vfmadd231pd(ymm0, ymm3, ymm14)
	vfmadd231pd(ymm1, ymm3, ymm15)

	vmovapd(mem(rbx, 0*32), ymm0)
	vmovapd(mem(rbx, 1*32), ymm1)

	// iteration 2
	prefetch(0, mem(rax, 40*16))

	vbroadcastsd(mem(rax, 12*8), ymm2)
	vbroadcastsd(mem(rax, 13*8), ymm3)
	vfmadd231pd(ymm0, ymm2, ymm4)
	vfmadd231pd(ymm1, ymm2, ymm5)
	vfmadd231pd(ymm0, ymm3, ymm6)
	vfmadd231pd(ymm1, ymm3, ymm7)

	vbroadcastsd(mem(rax, 14*8), ymm2)
	vbroadcastsd(mem(rax, 15*8), ymm3)
	vfmadd231pd(ymm0, ymm2, ymm8)
	vfmadd231pd(ymm1, ymm2, ymm9)
	vfmadd231pd(ymm0, ymm3, ymm10)
	vfmadd231pd(ymm1, ymm3, ymm11)

	vbroadcastsd(mem(rax, 16*8), ymm2)
	vbroadcastsd(mem(rax, 17*8), ymm3)
	vfmadd231pd(ymm0, ymm2, ymm12)
	vfmadd231pd(ymm1, ymm2, ymm13)
	vfmadd231pd(ymm0, ymm3, ymm14)
	vfmadd231pd(ymm1, ymm3, ymm15)

	vmovapd(mem(rbx, 2*32), ymm0)
	vmovapd(mem(rbx, 3*32), ymm1)

	// iteration 3
	vbroadcastsd(mem(rax, 18*8), ymm2)
	vbroadcastsd(mem(rax, 19*8), ymm3)
	vfmadd231pd(ymm0, ymm2, ymm4)
	vfmadd231pd(ymm1, ymm2, ymm5)
	vfmadd231pd(ymm0, ymm3, ymm6)
	vfmadd231pd(ymm1, ymm3, ymm7)

	vbroadcastsd(mem(rax, 20*8), ymm2)
	vbroadcastsd(mem(rax, 21*8), ymm3)
	vfmadd231pd(ymm0, ymm2, ymm8)
	vfmadd231pd(ymm1, ymm2, ymm9)
	vfmadd231pd(ymm0, ymm3, ymm10)
	vfmadd231pd(ymm1, ymm3, ymm11)

	vbroadcastsd(mem(rax, 22*8), ymm2)
	vbroadcastsd(mem(rax, 23*8), ymm3)
	vfmadd231pd(ymm0, ymm2, ymm12)
	vfmadd231pd(ymm1, ymm2, ymm13)
	vfmadd231pd(ymm0, ymm3, ymm14)
	vfmadd231pd(ymm1, ymm3, ymm15)

	add(imm(4*3*16), rax) // a += 4*3 (unroll x mr)
	add(imm(4*4*16), rbx) // b += 4*4 (unroll x nr)

	vmovapd(mem(rbx, -4*32), ymm0)
	vmovapd(mem(rbx, -3*32), ymm1)


	dec(rsi) // i -= 1;
	jne(.ZLOOPKITER) // iterate again if i != 0.






	label(.ZCONSIDKLEFT)

	mov(var(k_left), rsi) // i = k_left;
	test(rsi, rsi) // check i via logical AND.
	je(.ZPOSTACCUM) // if i == 0, we're done; jump to end.
	// else, we prepare to enter k_left loop.


	label(.ZLOOPKLEFT) // EDGE LOOP

	prefetch(0, mem(rax, 32*16))

	vbroadcastsd(mem(rax, 0*8), ymm2)
	vbroadcastsd(mem(rax, 1*8), ymm3)
	vfmadd231pd(ymm0, ymm2, ymm4)
	vfmadd231pd(ymm1, ymm2, ymm5)
	vfmadd231pd(ymm0, ymm3, ymm6)
	vfmadd231pd(ymm1, ymm3, ymm7)

	vbroadcastsd(mem(rax, 2*8), ymm2)
	vbroadcastsd(mem(rax, 3*8), ymm3)
	vfmadd231pd(ymm0, ymm2, ymm8)
	vfmadd231pd(ymm1, ymm2, ymm9)
	vfmadd231pd(ymm0, ymm3, ymm10)
	vfmadd231pd(ymm1, ymm3, ymm11)

	vbroadcastsd(mem(rax, 4*8), ymm2)
	vbroadcastsd(mem(rax, 5*8), ymm3)
	vfmadd231pd(ymm0, ymm2, ymm12)
	vfmadd231pd(ymm1, ymm2, ymm13)
	vfmadd231pd(ymm0, ymm3, ymm14)
	vfmadd231pd(ymm1, ymm3, ymm15)

	add(imm(1*3*16), rax) // a += 1*3 (unroll x mr)
	add(imm(1*4*16), rbx) // b += 1*4 (unroll x nr)

	vmovapd(mem(rbx, -4*32), ymm0)
	vmovapd(mem(rbx, -3*32), ymm1)


	dec(rsi) // i -= 1;
	jne(.ZLOOPKLEFT) // iterate again if i != 0.



	label(.ZPOSTACCUM)

	// permute even and odd elements
	// of ymm6/7, ymm10/11, ymm/14/15
	vpermilpd(imm(0x5), ymm6, ymm6)
	vpermilpd(imm(0x5), ymm7, ymm7)
	vpermilpd(imm(0x5), ymm10, ymm10)
	vpermilpd(imm(0x5), ymm11, ymm11)
	vpermilpd(imm(0x5), ymm14, ymm14)
	vpermilpd(imm(0x5), ymm15, ymm15)


	// subtract/add even/odd elements
	vaddsubpd(ymm6, ymm4, ymm4)
	vaddsubpd(ymm7, ymm5, ymm5)

	vaddsubpd(ymm10, ymm8, ymm8)
	vaddsubpd(ymm11, ymm9, ymm9)

	vaddsubpd(ymm14, ymm12, ymm12)
	vaddsubpd(ymm15, ymm13, ymm13)




	mov(var(alpha), rax) // load address of alpha
	vbroadcastsd(mem(rax), ymm0) // load alpha_r and duplicate
	vbroadcastsd(mem(rax, 8), ymm1) // load alpha_i and duplicate


	vpermilpd(imm(0x5), ymm4, ymm3)
	vmulpd(ymm0, ymm4, ymm4)
	vmulpd(ymm1, ymm3, ymm3)
	vaddsubpd(ymm3, ymm4, ymm4)

	vpermilpd(imm(0x5), ymm5, ymm3)
	vmulpd(ymm0, ymm5, ymm5)
	vmulpd(ymm1, ymm3, ymm3)
	vaddsubpd(ymm3, ymm5, ymm5)


	vpermilpd(imm(0x5), ymm8, ymm3)
	vmulpd(ymm0, ymm8, ymm8)
	vmulpd(ymm1, ymm3, ymm3)
	vaddsubpd(ymm3, ymm8, ymm8)

	vpermilpd(imm(0x5), ymm9, ymm3)
	vmulpd(ymm0, ymm9, ymm9)
	vmulpd(ymm1, ymm3, ymm3)
	vaddsubpd(ymm3, ymm9, ymm9)


	vpermilpd(imm(0x5), ymm12, ymm3)
	vmulpd(ymm0, ymm12, ymm12)
	vmulpd(ymm1, ymm3, ymm3)
	vaddsubpd(ymm3, ymm12, ymm12)

	vpermilpd(imm(0x5), ymm13, ymm3)
	vmulpd(ymm0, ymm13, ymm13)
	vmulpd(ymm1, ymm3, ymm3)
	vaddsubpd(ymm3, ymm13, ymm13)





	mov(var(beta), rbx) // load address of beta
	vbroadcastsd(mem(rbx), ymm1) // load beta_r and duplicate
	vbroadcastsd(mem(rbx, 8), ymm2) // load beta_i and duplicate



	// now avoid loading C if beta == 0
	vxorpd(ymm0, ymm0, ymm0) // set ymm0 to zero.
	vucomisd(xmm0, xmm1) // set ZF if beta_r == 0.
	sete(r8b) // r8b = ( ZF == 1 ? 1 : 0 );
	vucomisd(xmm0, xmm2) // set ZF if beta_i == 0.
	sete(r9b) // r9b = ( ZF == 1 ? 1 : 0 );
	and(r8b, r9b) // set ZF if r8b & r9b == 1.
	jne(.ZBETAZERO) // if ZF = 1, jump to beta == 0 case

		ZGEMM_INPUT_SCALE_RS_BETA_NZ(mem(rcx))
		vaddpd(ymm4, ymm0, ymm0)
		vmovupd(ymm0, mem(rcx))


		ZGEMM_INPUT_SCALE_RS_BETA_NZ(mem(rcx,32))
		vaddpd(ymm5, ymm0, ymm0)
		vmovupd(ymm0, mem(rcx,32))



		ZGEMM_INPUT_SCALE_RS_BETA_NZ(mem(r11))
		vaddpd(ymm8, ymm0, ymm0)
		vmovupd(ymm0, mem(r11))


		ZGEMM_INPUT_SCALE_RS_BETA_NZ(mem(r11,32))
		vaddpd(ymm9, ymm0, ymm0)
		vmovupd(ymm0, mem(r11,32))



		ZGEMM_INPUT_SCALE_RS_BETA_NZ(mem(r12))
		vaddpd(ymm12, ymm0, ymm0)
		vmovupd(ymm0, mem(r12))


		ZGEMM_INPUT_SCALE_RS_BETA_NZ(mem(r12,32))
		vaddpd(ymm13, ymm0, ymm0)
		vmovupd(ymm0, mem(r12,32))

		jmp(.ZDONE) // jump to end.

	label(.ZBETAZERO)

		vmovupd(ymm4, mem(rcx))
		vmovupd(ymm5, mem(rcx,32))

		vmovupd(ymm8, mem(r11))
		vmovupd(ymm9, mem(r11,32))

		vmovupd(ymm12, mem(r12))
		vmovupd(ymm13, mem(r12,32))

	label(.ZDONE)


	vzeroupper()



	end_asm(
	: // output operands (none)
	: // input operands
	  [k_iter] "m" (k_iter), // 0
	  [k_left] "m" (k_left), // 1
	  [a]      "m" (a),      // 2
	  [b]      "m" (b),      // 3
	  [alpha]  "m" (alpha),  // 4
	  [beta]   "m" (beta),   // 5
	  [c]      "m" (c),      // 6
	  [rs_c]   "m" (rs_c),   // 7
	  [cs_c]   "m" (cs_c)/*,   // 8
	  [b_next] "m" (b_next), // 9
	  [a_next] "m" (a_next)*/  // 10
	: // register clobber list
	  "rax", "rbx", "rcx", "rdx", "rsi", "rdi",
	  "r8", "r9", "r10", "r11", "r12", "r13", "r14", "r15",
	  "xmm0", "xmm1", "xmm2", "xmm3",
	  "xmm4", "xmm5", "xmm6", "xmm7",
	  "xmm8", "xmm9", "xmm10", "xmm11",
	  "xmm12", "xmm13", "xmm14", "xmm15",
	  "memory"
	)

	GEMM_UKR_FLUSH_CT( z );
}


