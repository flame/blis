/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2026, Yicheng Kang

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


// VLEN-scaled sgemm micro-kernel (rv64iv selects it for VLEN = 256): MR = 2*v (one LMUL-2 vector group of A),
// NR = 12. Twelve LMUL-2 accumulators (v0, v2, ..., v22) hold the MR x 12 tile;
// A is double-buffered in v24/v26 and C is updated through v28. Per k-step: 12
// scalar loads of B, one vector load of A and 12 vfmacc.vf, i.e. 12 FMAs per
// vector load. Only caller-saved registers are used, and only RV32/RV64 common
// instructions, so the file assembles for rv32iv as well as rv64iv.
//
// void bli_sgemm_rviv_asm_2vx12(intptr_t k, const void* alpha, const void* a,
//                               const void* b, const void* beta, void* c,
//                               intptr_t cs_c)   // cs_c in bytes; rs_c == 1
//
// a0 k   a1 alpha   a2 a   a3 b   a4 beta   a5 c   a6 cs_c

	.text
	.align      2
	.global     bli_sgemm_rviv_asm_2vx12

#define k_iter   a0
#define A_ptr    a2
#define B_ptr    a3
#define C_ptr    a5
#define cs_c     a6
#define A_step   t0
#define k_left   t1
#define tmp      t2

#define ALPHA    fa0
#define BETA     fa1
#define FZERO    fa2
#define B0     ft0
#define B1     ft1
#define B2     ft2
#define B3     ft3
#define B4     ft4
#define B5     ft5
#define B6     ft6
#define B7     ft7
#define B8     ft8
#define B9     ft9
#define B10    ft10
#define B11    ft11

#define A0       v24
#define A1       v26
#define CT       v28

bli_sgemm_rviv_asm_2vx12:
	vsetvli tmp, zero, e32, m2, ta, ma
	csrr A_step, vlenb
	slli A_step, A_step, 1          // bytes per packed A column: 2*VLENB
	vmv.v.i v0, 0
	vmv.v.i v2, 0
	vmv.v.i v4, 0
	vmv.v.i v6, 0
	vmv.v.i v8, 0
	vmv.v.i v10, 0
	vmv.v.i v12, 0
	vmv.v.i v14, 0
	vmv.v.i v16, 0
	vmv.v.i v18, 0
	vmv.v.i v20, 0
	vmv.v.i v22, 0

	andi k_left, k_iter, 1
	srli k_iter, k_iter, 1
	beqz k_iter, .LTAIL

.LLOOP:
	flw B0, 0(B_ptr)
	flw B1, 4(B_ptr)
	flw B2, 8(B_ptr)
	flw B3, 12(B_ptr)
	flw B4, 16(B_ptr)
	flw B5, 20(B_ptr)
	flw B6, 24(B_ptr)
	flw B7, 28(B_ptr)
	flw B8, 32(B_ptr)
	flw B9, 36(B_ptr)
	flw B10, 40(B_ptr)
	flw B11, 44(B_ptr)
	addi B_ptr, B_ptr, 48
	vle32.v A0, (A_ptr)
	add A_ptr, A_ptr, A_step
	vfmacc.vf v0, B0, A0
	vfmacc.vf v2, B1, A0
	vfmacc.vf v4, B2, A0
	vfmacc.vf v6, B3, A0
	vfmacc.vf v8, B4, A0
	vfmacc.vf v10, B5, A0
	vfmacc.vf v12, B6, A0
	vfmacc.vf v14, B7, A0
	vfmacc.vf v16, B8, A0
	vfmacc.vf v18, B9, A0
	vfmacc.vf v20, B10, A0
	vfmacc.vf v22, B11, A0
	flw B0, 0(B_ptr)
	flw B1, 4(B_ptr)
	flw B2, 8(B_ptr)
	flw B3, 12(B_ptr)
	flw B4, 16(B_ptr)
	flw B5, 20(B_ptr)
	flw B6, 24(B_ptr)
	flw B7, 28(B_ptr)
	flw B8, 32(B_ptr)
	flw B9, 36(B_ptr)
	flw B10, 40(B_ptr)
	flw B11, 44(B_ptr)
	addi B_ptr, B_ptr, 48
	vle32.v A1, (A_ptr)
	add A_ptr, A_ptr, A_step
	vfmacc.vf v0, B0, A1
	vfmacc.vf v2, B1, A1
	vfmacc.vf v4, B2, A1
	vfmacc.vf v6, B3, A1
	vfmacc.vf v8, B4, A1
	vfmacc.vf v10, B5, A1
	vfmacc.vf v12, B6, A1
	vfmacc.vf v14, B7, A1
	vfmacc.vf v16, B8, A1
	vfmacc.vf v18, B9, A1
	vfmacc.vf v20, B10, A1
	vfmacc.vf v22, B11, A1
	addi k_iter, k_iter, -1
	bnez k_iter, .LLOOP

.LTAIL:
	beqz k_left, .LSCALE
	flw B0, 0(B_ptr)
	flw B1, 4(B_ptr)
	flw B2, 8(B_ptr)
	flw B3, 12(B_ptr)
	flw B4, 16(B_ptr)
	flw B5, 20(B_ptr)
	flw B6, 24(B_ptr)
	flw B7, 28(B_ptr)
	flw B8, 32(B_ptr)
	flw B9, 36(B_ptr)
	flw B10, 40(B_ptr)
	flw B11, 44(B_ptr)
	vle32.v A0, (A_ptr)
	vfmacc.vf v0, B0, A0
	vfmacc.vf v2, B1, A0
	vfmacc.vf v4, B2, A0
	vfmacc.vf v6, B3, A0
	vfmacc.vf v8, B4, A0
	vfmacc.vf v10, B5, A0
	vfmacc.vf v12, B6, A0
	vfmacc.vf v14, B7, A0
	vfmacc.vf v16, B8, A0
	vfmacc.vf v18, B9, A0
	vfmacc.vf v20, B10, A0
	vfmacc.vf v22, B11, A0

.LSCALE:
	flw ALPHA, 0(a1)
	flw BETA, 0(a4)
	fmv.w.x FZERO, zero
	vfmul.vf v0, v0, ALPHA
	vfmul.vf v2, v2, ALPHA
	vfmul.vf v4, v4, ALPHA
	vfmul.vf v6, v6, ALPHA
	vfmul.vf v8, v8, ALPHA
	vfmul.vf v10, v10, ALPHA
	vfmul.vf v12, v12, ALPHA
	vfmul.vf v14, v14, ALPHA
	vfmul.vf v16, v16, ALPHA
	vfmul.vf v18, v18, ALPHA
	vfmul.vf v20, v20, ALPHA
	vfmul.vf v22, v22, ALPHA
	feq.s tmp, BETA, FZERO
	beqz tmp, .LBETA

	// beta == 0: C is only written, never read.
	vse32.v v0, (C_ptr)
	add C_ptr, C_ptr, cs_c
	vse32.v v2, (C_ptr)
	add C_ptr, C_ptr, cs_c
	vse32.v v4, (C_ptr)
	add C_ptr, C_ptr, cs_c
	vse32.v v6, (C_ptr)
	add C_ptr, C_ptr, cs_c
	vse32.v v8, (C_ptr)
	add C_ptr, C_ptr, cs_c
	vse32.v v10, (C_ptr)
	add C_ptr, C_ptr, cs_c
	vse32.v v12, (C_ptr)
	add C_ptr, C_ptr, cs_c
	vse32.v v14, (C_ptr)
	add C_ptr, C_ptr, cs_c
	vse32.v v16, (C_ptr)
	add C_ptr, C_ptr, cs_c
	vse32.v v18, (C_ptr)
	add C_ptr, C_ptr, cs_c
	vse32.v v20, (C_ptr)
	add C_ptr, C_ptr, cs_c
	vse32.v v22, (C_ptr)
	add C_ptr, C_ptr, cs_c
	ret

.LBETA:
	// C := beta * C + alpha * A * B
	vle32.v CT, (C_ptr)
	vfmacc.vf v0, BETA, CT
	vse32.v v0, (C_ptr)
	add C_ptr, C_ptr, cs_c
	vle32.v CT, (C_ptr)
	vfmacc.vf v2, BETA, CT
	vse32.v v2, (C_ptr)
	add C_ptr, C_ptr, cs_c
	vle32.v CT, (C_ptr)
	vfmacc.vf v4, BETA, CT
	vse32.v v4, (C_ptr)
	add C_ptr, C_ptr, cs_c
	vle32.v CT, (C_ptr)
	vfmacc.vf v6, BETA, CT
	vse32.v v6, (C_ptr)
	add C_ptr, C_ptr, cs_c
	vle32.v CT, (C_ptr)
	vfmacc.vf v8, BETA, CT
	vse32.v v8, (C_ptr)
	add C_ptr, C_ptr, cs_c
	vle32.v CT, (C_ptr)
	vfmacc.vf v10, BETA, CT
	vse32.v v10, (C_ptr)
	add C_ptr, C_ptr, cs_c
	vle32.v CT, (C_ptr)
	vfmacc.vf v12, BETA, CT
	vse32.v v12, (C_ptr)
	add C_ptr, C_ptr, cs_c
	vle32.v CT, (C_ptr)
	vfmacc.vf v14, BETA, CT
	vse32.v v14, (C_ptr)
	add C_ptr, C_ptr, cs_c
	vle32.v CT, (C_ptr)
	vfmacc.vf v16, BETA, CT
	vse32.v v16, (C_ptr)
	add C_ptr, C_ptr, cs_c
	vle32.v CT, (C_ptr)
	vfmacc.vf v18, BETA, CT
	vse32.v v18, (C_ptr)
	add C_ptr, C_ptr, cs_c
	vle32.v CT, (C_ptr)
	vfmacc.vf v20, BETA, CT
	vse32.v v20, (C_ptr)
	add C_ptr, C_ptr, cs_c
	vle32.v CT, (C_ptr)
	vfmacc.vf v22, BETA, CT
	vse32.v v22, (C_ptr)
	add C_ptr, C_ptr, cs_c
	ret
