/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2023, The University of Texas at Austin
   Copyright (C) 2026, Hugo Meiland

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
// RVV gemm microkernel 2vx8 for VLEN >= 256.
// Shape: 2 vector-rows of A x 8 columns of B.
// At VLEN=256: dgemm mr=8/nr=8, sgemm mr=16/nr=8.
// k-unrolled by 4 with A/B software pipeline; skip alpha scale when alpha==1.

	.text
	.align      2
	.global     REALNAME

REALNAME:
	#include "rviv_save_registers.h"

	addi sp, sp, -48
	# 0 alpha_ptr, 8 beta_ptr, 16 a_base, 24 vlenb, 32 colstride
	sd a1, 0(sp)
	sd a4, 8(sp)
	sd a2, 16(sp)

	vsetvli t0, zero, VTYPE, m1, ta, ma
	csrr t0, vlenb
	sd t0, 24(sp)
	FZERO(ft11)

	# C row0: a5,t1-t6,a4
	add t1, a5, a7
	add t2, t1, a7
	add t3, t2, a7
	add t4, t3, a7
	add t5, t4, a7
	add t6, t5, a7
	add a4, t6, a7

	# C row1: s0-s5, s6=C16, s7=C17  (s6/s7 live across kernel — OK, restored later)
	add s0, a5, a6
	add s1, t1, a6
	add s2, t2, a6
	add s3, t3, a6
	add s4, t4, a6
	add s5, t5, a6
	add s6, t6, a6
	add s7, a4, a6

	vxor.vv v0, v0, v0
	vxor.vv v1, v1, v1
	vxor.vv v2, v2, v2
	vxor.vv v3, v3, v3
	vxor.vv v4, v4, v4
	vxor.vv v5, v5, v5
	vxor.vv v6, v6, v6
	vxor.vv v7, v7, v7
	vxor.vv v8, v8, v8
	vxor.vv v9, v9, v9
	vxor.vv v10, v10, v10
	vxor.vv v11, v11, v11
	vxor.vv v12, v12, v12
	vxor.vv v13, v13, v13
	vxor.vv v14, v14, v14
	vxor.vv v15, v15, v15

	beqz a0, MULTIPLYBETA

	ld a2, 16(sp)
	ld t0, 24(sp)
	# a1 = A10 = A00 + vlenb; keep colstride in stack slot 32
	add a1, a2, t0
	slli t0, t0, 1
	sd t0, 32(sp)              # colstride = 2*vlenb

	li t0, 3
	ble a0, t0, TAIL_LE3

	# Prefetch first A(:,0) and B(0,:)
	VLE v24, (a2)
	VLE v25, (a1)
	FLOAD fa0, 0*DATASIZE(a3)
	FLOAD fa1, 1*DATASIZE(a3)
	FLOAD fa2, 2*DATASIZE(a3)
	FLOAD fa3, 3*DATASIZE(a3)
	FLOAD fa4, 4*DATASIZE(a3)
	FLOAD fa5, 5*DATASIZE(a3)
	FLOAD fa6, 6*DATASIZE(a3)
	FLOAD fa7, 7*DATASIZE(a3)

LOOP4:
	addi a0, a0, -4
	ld t0, 32(sp)              # colstride

	# --- k+0 ---
	vfmacc.vf v0,  fa0, v24
	vfmacc.vf v1,  fa1, v24
	vfmacc.vf v2,  fa2, v24
	vfmacc.vf v3,  fa3, v24
	# point A(:,1)
	add a2, a2, t0
	add a1, a1, t0
	vfmacc.vf v4,  fa4, v24
	vfmacc.vf v5,  fa5, v24
	vfmacc.vf v6,  fa6, v24
	vfmacc.vf v7,  fa7, v24
	FLOAD ft0,  8*DATASIZE(a3)
	FLOAD ft1,  9*DATASIZE(a3)
	vfmacc.vf v8,  fa0, v25
	vfmacc.vf v9,  fa1, v25
	FLOAD ft2, 10*DATASIZE(a3)
	FLOAD ft3, 11*DATASIZE(a3)
	vfmacc.vf v10, fa2, v25
	vfmacc.vf v11, fa3, v25
	FLOAD ft4, 12*DATASIZE(a3)
	FLOAD ft5, 13*DATASIZE(a3)
	vfmacc.vf v12, fa4, v25
	vfmacc.vf v13, fa5, v25
	FLOAD ft6, 14*DATASIZE(a3)
	FLOAD ft7, 15*DATASIZE(a3)
	vfmacc.vf v14, fa6, v25
	vfmacc.vf v15, fa7, v25
	VLE v26, (a2)              # A(:,1)
	VLE v27, (a1)

	# --- k+1 ---
	vfmacc.vf v0,  ft0, v26
	vfmacc.vf v1,  ft1, v26
	vfmacc.vf v2,  ft2, v26
	vfmacc.vf v3,  ft3, v26
	add a2, a2, t0
	add a1, a1, t0
	vfmacc.vf v4,  ft4, v26
	vfmacc.vf v5,  ft5, v26
	vfmacc.vf v6,  ft6, v26
	vfmacc.vf v7,  ft7, v26
	FLOAD fa0, 16*DATASIZE(a3)
	FLOAD fa1, 17*DATASIZE(a3)
	vfmacc.vf v8,  ft0, v27
	vfmacc.vf v9,  ft1, v27
	FLOAD fa2, 18*DATASIZE(a3)
	FLOAD fa3, 19*DATASIZE(a3)
	vfmacc.vf v10, ft2, v27
	vfmacc.vf v11, ft3, v27
	FLOAD fa4, 20*DATASIZE(a3)
	FLOAD fa5, 21*DATASIZE(a3)
	vfmacc.vf v12, ft4, v27
	vfmacc.vf v13, ft5, v27
	FLOAD fa6, 22*DATASIZE(a3)
	FLOAD fa7, 23*DATASIZE(a3)
	vfmacc.vf v14, ft6, v27
	vfmacc.vf v15, ft7, v27
	VLE v24, (a2)              # A(:,2)
	VLE v25, (a1)

	# --- k+2 ---
	vfmacc.vf v0,  fa0, v24
	vfmacc.vf v1,  fa1, v24
	vfmacc.vf v2,  fa2, v24
	vfmacc.vf v3,  fa3, v24
	add a2, a2, t0
	add a1, a1, t0
	vfmacc.vf v4,  fa4, v24
	vfmacc.vf v5,  fa5, v24
	vfmacc.vf v6,  fa6, v24
	vfmacc.vf v7,  fa7, v24
	FLOAD ft0, 24*DATASIZE(a3)
	FLOAD ft1, 25*DATASIZE(a3)
	vfmacc.vf v8,  fa0, v25
	vfmacc.vf v9,  fa1, v25
	FLOAD ft2, 26*DATASIZE(a3)
	FLOAD ft3, 27*DATASIZE(a3)
	vfmacc.vf v10, fa2, v25
	vfmacc.vf v11, fa3, v25
	FLOAD ft4, 28*DATASIZE(a3)
	FLOAD ft5, 29*DATASIZE(a3)
	vfmacc.vf v12, fa4, v25
	vfmacc.vf v13, fa5, v25
	FLOAD ft6, 30*DATASIZE(a3)
	FLOAD ft7, 31*DATASIZE(a3)
	addi a3, a3, 32*DATASIZE
	vfmacc.vf v14, fa6, v25
	vfmacc.vf v15, fa7, v25
	VLE v26, (a2)              # A(:,3)
	VLE v27, (a1)

	# --- k+3 ---
	vfmacc.vf v0,  ft0, v26
	vfmacc.vf v1,  ft1, v26
	vfmacc.vf v2,  ft2, v26
	vfmacc.vf v3,  ft3, v26
	add a2, a2, t0
	add a1, a1, t0
	vfmacc.vf v4,  ft4, v26
	vfmacc.vf v5,  ft5, v26
	vfmacc.vf v6,  ft6, v26
	vfmacc.vf v7,  ft7, v26
	vfmacc.vf v8,  ft0, v27
	vfmacc.vf v9,  ft1, v27
	vfmacc.vf v10, ft2, v27
	vfmacc.vf v11, ft3, v27
	vfmacc.vf v12, ft4, v27
	vfmacc.vf v13, ft5, v27
	vfmacc.vf v14, ft6, v27
	vfmacc.vf v15, ft7, v27

	li t0, 3
	ble a0, t0, AFTER4

	# Prefetch next quartet A(:,0), B(0,:)
	VLE v24, (a2)
	VLE v25, (a1)
	FLOAD fa0, 0*DATASIZE(a3)
	FLOAD fa1, 1*DATASIZE(a3)
	FLOAD fa2, 2*DATASIZE(a3)
	FLOAD fa3, 3*DATASIZE(a3)
	FLOAD fa4, 4*DATASIZE(a3)
	FLOAD fa5, 5*DATASIZE(a3)
	FLOAD fa6, 6*DATASIZE(a3)
	FLOAD fa7, 7*DATASIZE(a3)
	j LOOP4

AFTER4:
	beqz a0, MULTIPLYALPHA
	# fall through to tail with a2/a1 already at next column

TAIL_LE3:
	li t0, 1
	ble a0, t0, TAIL1_CHECK

TAIL2:
	# process 2 if a0 >= 2
	addi a0, a0, -2
	ld t0, 32(sp)
	VLE v24, (a2)
	VLE v25, (a1)
	FLOAD fa0, 0*DATASIZE(a3)
	FLOAD fa1, 1*DATASIZE(a3)
	FLOAD fa2, 2*DATASIZE(a3)
	FLOAD fa3, 3*DATASIZE(a3)
	FLOAD fa4, 4*DATASIZE(a3)
	FLOAD fa5, 5*DATASIZE(a3)
	FLOAD fa6, 6*DATASIZE(a3)
	FLOAD fa7, 7*DATASIZE(a3)
	vfmacc.vf v0,  fa0, v24
	vfmacc.vf v1,  fa1, v24
	vfmacc.vf v2,  fa2, v24
	vfmacc.vf v3,  fa3, v24
	vfmacc.vf v4,  fa4, v24
	vfmacc.vf v5,  fa5, v24
	vfmacc.vf v6,  fa6, v24
	vfmacc.vf v7,  fa7, v24
	add a2, a2, t0
	add a1, a1, t0
	vfmacc.vf v8,  fa0, v25
	vfmacc.vf v9,  fa1, v25
	vfmacc.vf v10, fa2, v25
	vfmacc.vf v11, fa3, v25
	FLOAD ft0,  8*DATASIZE(a3)
	FLOAD ft1,  9*DATASIZE(a3)
	FLOAD ft2, 10*DATASIZE(a3)
	FLOAD ft3, 11*DATASIZE(a3)
	vfmacc.vf v12, fa4, v25
	vfmacc.vf v13, fa5, v25
	FLOAD ft4, 12*DATASIZE(a3)
	FLOAD ft5, 13*DATASIZE(a3)
	FLOAD ft6, 14*DATASIZE(a3)
	FLOAD ft7, 15*DATASIZE(a3)
	addi a3, a3, 16*DATASIZE
	vfmacc.vf v14, fa6, v25
	vfmacc.vf v15, fa7, v25
	VLE v26, (a2)
	VLE v27, (a1)
	vfmacc.vf v0,  ft0, v26
	vfmacc.vf v1,  ft1, v26
	vfmacc.vf v2,  ft2, v26
	vfmacc.vf v3,  ft3, v26
	vfmacc.vf v4,  ft4, v26
	vfmacc.vf v5,  ft5, v26
	vfmacc.vf v6,  ft6, v26
	vfmacc.vf v7,  ft7, v26
	add a2, a2, t0
	add a1, a1, t0
	vfmacc.vf v8,  ft0, v27
	vfmacc.vf v9,  ft1, v27
	vfmacc.vf v10, ft2, v27
	vfmacc.vf v11, ft3, v27
	vfmacc.vf v12, ft4, v27
	vfmacc.vf v13, ft5, v27
	vfmacc.vf v14, ft6, v27
	vfmacc.vf v15, ft7, v27

TAIL1_CHECK:
	beqz a0, MULTIPLYALPHA

TAIL1:
	VLE v24, (a2)
	VLE v25, (a1)
	FLOAD fa0, 0*DATASIZE(a3)
	FLOAD fa1, 1*DATASIZE(a3)
	FLOAD fa2, 2*DATASIZE(a3)
	FLOAD fa3, 3*DATASIZE(a3)
	FLOAD fa4, 4*DATASIZE(a3)
	FLOAD fa5, 5*DATASIZE(a3)
	FLOAD fa6, 6*DATASIZE(a3)
	FLOAD fa7, 7*DATASIZE(a3)
	vfmacc.vf v0,  fa0, v24
	vfmacc.vf v1,  fa1, v24
	vfmacc.vf v2,  fa2, v24
	vfmacc.vf v3,  fa3, v24
	vfmacc.vf v4,  fa4, v24
	vfmacc.vf v5,  fa5, v24
	vfmacc.vf v6,  fa6, v24
	vfmacc.vf v7,  fa7, v24
	vfmacc.vf v8,  fa0, v25
	vfmacc.vf v9,  fa1, v25
	vfmacc.vf v10, fa2, v25
	vfmacc.vf v11, fa3, v25
	vfmacc.vf v12, fa4, v25
	vfmacc.vf v13, fa5, v25
	vfmacc.vf v14, fa6, v25
	vfmacc.vf v15, fa7, v25

MULTIPLYALPHA:
	ld a1, 0(sp)
	FLOAD fa1, (a1)
#if DATASIZE == 8
	li a2, 0x3FF0000000000000
	fmv.d.x ft10, a2
	feq.d a2, fa1, ft10
#else
	li a2, 0x3F800000
	fmv.w.x ft10, a2
	feq.s a2, fa1, ft10
#endif
	bnez a2, MULTIPLYBETA
	vfmul.vf v0,  v0,  fa1
	vfmul.vf v1,  v1,  fa1
	vfmul.vf v2,  v2,  fa1
	vfmul.vf v3,  v3,  fa1
	vfmul.vf v4,  v4,  fa1
	vfmul.vf v5,  v5,  fa1
	vfmul.vf v6,  v6,  fa1
	vfmul.vf v7,  v7,  fa1
	vfmul.vf v8,  v8,  fa1
	vfmul.vf v9,  v9,  fa1
	vfmul.vf v10, v10, fa1
	vfmul.vf v11, v11, fa1
	vfmul.vf v12, v12, fa1
	vfmul.vf v13, v13, fa1
	vfmul.vf v14, v14, fa1
	vfmul.vf v15, v15, fa1

MULTIPLYBETA:
	ld a1, 8(sp)
	FLOAD fa2, (a1)
	FEQ a1, fa2, ft11
	beq a1, zero, BETANOTZERO

BETAZERO:
	VSE v0, (a5)
	VSE v1, (t1)
	VSE v2, (t2)
	VSE v3, (t3)
	VSE v4, (t4)
	VSE v5, (t5)
	VSE v6, (t6)
	VSE v7, (a4)
	VSE v8, (s0)
	VSE v9, (s1)
	VSE v10, (s2)
	VSE v11, (s3)
	VSE v12, (s4)
	VSE v13, (s5)
	VSE v14, (s6)
	VSE v15, (s7)
	j END

BETANOTZERO:
	VLE v16, (a5)
	VLE v17, (t1)
	VLE v18, (t2)
	VLE v19, (t3)
	VLE v20, (t4)
	VLE v21, (t5)
	VLE v22, (t6)
	VLE v23, (a4)
	vfmacc.vf v0, fa2, v16
	vfmacc.vf v1, fa2, v17
	vfmacc.vf v2, fa2, v18
	vfmacc.vf v3, fa2, v19
	vfmacc.vf v4, fa2, v20
	vfmacc.vf v5, fa2, v21
	vfmacc.vf v6, fa2, v22
	vfmacc.vf v7, fa2, v23
	VSE v0, (a5)
	VSE v1, (t1)
	VSE v2, (t2)
	VSE v3, (t3)
	VSE v4, (t4)
	VSE v5, (t5)
	VSE v6, (t6)
	VSE v7, (a4)
	VLE v16, (s0)
	VLE v17, (s1)
	VLE v18, (s2)
	VLE v19, (s3)
	VLE v20, (s4)
	VLE v21, (s5)
	VLE v22, (s6)
	VLE v23, (s7)
	vfmacc.vf v8,  fa2, v16
	vfmacc.vf v9,  fa2, v17
	vfmacc.vf v10, fa2, v18
	vfmacc.vf v11, fa2, v19
	vfmacc.vf v12, fa2, v20
	vfmacc.vf v13, fa2, v21
	vfmacc.vf v14, fa2, v22
	vfmacc.vf v15, fa2, v23
	VSE v8,  (s0)
	VSE v9,  (s1)
	VSE v10, (s2)
	VSE v11, (s3)
	VSE v12, (s4)
	VSE v13, (s5)
	VSE v14, (s6)
	VSE v15, (s7)

END:
	addi sp, sp, 48
	#include "rviv_restore_registers.h"
	ret
