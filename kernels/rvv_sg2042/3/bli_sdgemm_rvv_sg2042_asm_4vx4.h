/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2023, The University of Texas at Austin

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

	.text
	.align      2
	.global     REALNAME

// void REALNAME(intptr_t k, void* alpha, void* a, void* b,
//               void* beta, void* c, intptr_t rs_c, intptr_t cs_c)
//
// register arguments:
// a0   k
// a1   alpha
// a2   a
// a3   b
// a4   beta
// a5   c
// a6   rs_c
// a7   cs_c
//

#define loop_counter a0

#define A00_ptr   a2
#define A10_ptr   t0
#define A20_ptr   t1
#define A30_ptr   t2
#define A01_ptr   s5
#define A11_ptr   s6
#define A21_ptr   s7
#define A31_ptr   t6

#define B_row_ptr a3

#define C00_ptr   a5
#define C01_ptr   t3
#define C02_ptr   t4
#define C03_ptr   t5
#define C10_ptr   s1
#define C11_ptr   s2
#define C12_ptr   s3
#define C13_ptr   s4

#define tmp    t6

#define ALPHA  fa1
#define BETA   fa2

#define B00    fa4
#define B01    fa5
#define B02    fa6
#define B03    fa7

#define B10    fa0
#define B11    fa1
#define B12    fa2
#define B13    fa3

#define fzero  ft8

// #define A00    v24
// #define A10    v25
// #define A20    v26
// #define A30    v27

// #define A01    v28
// #define A11    v29
// #define A21    v30
// #define A31    v31

#define A00    v24
#define A10    v25
#define A20    v26
#define A30    v27

#define A01    v28
#define A11    v29
#define A21    v30
#define A31    v31

// #define C00    v16
// #define C01    v18
// #define C02    v20
// #define C03    v22
// #define C10    v17
// #define C11    v19
// #define C12    v21
// #define C13    v23
// #define C20    v0
// #define C21    v2
// #define C22    v4
// #define C23    v6
// #define C30    v1
// #define C31    v3
// #define C32    v5
// #define C33    v7

#define C00    v16
#define C01    v20
#define C02    v24
#define C03    v28
#define C10    v17
#define C11    v21
#define C12    v25
#define C13    v29
#define C20    v18
#define C21    v22
#define C22    v26
#define C23    v30
#define C30    v19
#define C31    v23
#define C32    v27
#define C33    v31

// #define AB00   v0
// #define AB01   v2
// #define AB02   v4
// #define AB03   v6
// #define AB10   v1
// #define AB11   v3
// #define AB12   v5
// #define AB13   v7
// #define AB20   v8
// #define AB21   v10
// #define AB22   v12
// #define AB23   v14
// #define AB30   v9
// #define AB31   v11
// #define AB32   v13
// #define AB33   v15

#define AB00   v0
#define AB01   v4
#define AB02   v8
#define AB03   v12
#define AB10   v1
#define AB11   v5
#define AB12   v9
#define AB13   v13
#define AB20   v2
#define AB21   v6
#define AB22   v10
#define AB23   v14
#define AB30   v3
#define AB31   v7
#define AB32   v11
#define AB33   v15

#define rs_c   a6
#define cs_c   a7

REALNAME:
	#include "rvv_sg2042_save_registers.h"

	th.vsetvli s0, zero, VTYPE, m4
	csrr s0, vlenb
	//li s0, 16
	FZERO(fzero)

	// Set up pointers
	add C01_ptr, C00_ptr, cs_c
	add C02_ptr, C01_ptr, cs_c
	add C03_ptr, C02_ptr, cs_c
	// add C10_ptr, C00_ptr, rs_c
	// add C11_ptr, C01_ptr, rs_c
	// add C12_ptr, C02_ptr, rs_c
	// add C13_ptr, C03_ptr, rs_c

	// Zero-initialize accumulators
	th.vxor.vv AB00, AB00, AB00
	th.vxor.vv AB01, AB01, AB01
	th.vxor.vv AB02, AB02, AB02
	th.vxor.vv AB03, AB03, AB03
	//th.vxor.vv AB10, AB10, AB10
	//th.vxor.vv AB11, AB11, AB11
	//th.vxor.vv AB12, AB12, AB12
	//th.vxor.vv AB13, AB13, AB13
	// th.vxor.vv AB20, AB20, AB20
	// th.vxor.vv AB21, AB21, AB21
	// th.vxor.vv AB22, AB22, AB22
	// th.vxor.vv AB23, AB23, AB23
	//th.vxor.vv AB30, AB30, AB30
	//th.vxor.vv AB31, AB31, AB31
	//th.vxor.vv AB32, AB32, AB32
	//th.vxor.vv AB33, AB33, AB33

	// Handle k == 0
	beqz loop_counter, MULTIPLYBETA

	// Set up pointers to rows of A
	// add A10_ptr, A00_ptr, s0
	// add A20_ptr, A10_ptr, s0
	// add A30_ptr, A20_ptr, s0

	slli s0, s0, 2 // length of a column of A in bytes

	li tmp, 3
	ble loop_counter, tmp, TAIL_UNROLL_2

	// Preload A and B
	// Load A(:,l)
	VLE A00, (A00_ptr)
	//VLE A10, (A10_ptr)
	// VLE A20, (A20_ptr)
	//VLE A30, (A30_ptr)

	// Load B(l,0:3)
	FLOAD B00, 0*DATASIZE(B_row_ptr)
	FLOAD B01, 1*DATASIZE(B_row_ptr)
	FLOAD B02, 2*DATASIZE(B_row_ptr)
	FLOAD B03, 3*DATASIZE(B_row_ptr)

	// Set up pointers to A(:,l+1)
	add A01_ptr, A00_ptr, s0
	// add A11_ptr, A10_ptr, s0
	// add A21_ptr, A20_ptr, s0
	// add A31_ptr, A30_ptr, s0

LOOP_UNROLL_4:
	addi loop_counter, loop_counter, -4

	//th.vsetvli s0, zero, e64, m2
	//csrr s0, vlenb
	//VLE A00, (A00_ptr)
	th.vfmacc.vf AB00, B00, A00   // AB(0,:) += A(0,0) * B(0,:)
	//th.vsetvli s0, zero, e64, m1
	//csrr s0, vlenb
	//slli s0, s0, 2
	//VLE A00, (A00_ptr)
	th.vfmacc.vf AB01, B01, A00
	th.vfmacc.vf AB02, B02, A00
	th.vfmacc.vf AB03, B03, A00

	//th.vfmacc.vf AB10, B00, A10   // AB(1,:) += A(1,0) * B(0,:) commento questa perchè la faccio con LMUL=2 alla riga 212
	//th.vfmacc.vf AB11, B01, A10
	//th.vfmacc.vf AB12, B02, A10
	//th.vfmacc.vf AB13, B03, A10

	// Load B(l+1,0:3)
	FLOAD B10, 4*DATASIZE(B_row_ptr)
	FLOAD B11, 5*DATASIZE(B_row_ptr)
	FLOAD B12, 6*DATASIZE(B_row_ptr)
	FLOAD B13, 7*DATASIZE(B_row_ptr)
	addi B_row_ptr, B_row_ptr, 8*DATASIZE

	// th.vfmacc.vf AB20, B00, A20   // AB(2,:) += A(2,0) * B(0,:)
	// th.vfmacc.vf AB21, B01, A20
	// th.vfmacc.vf AB22, B02, A20
	// th.vfmacc.vf AB23, B03, A20

	// Load A(:,l+1)
	VLE A01, (A01_ptr)
	//VLE A11, (A11_ptr)
	// VLE A21, (A21_ptr)
	//VLE A31, (A31_ptr)

	// Point to A(:,l+2)
	add A00_ptr, A01_ptr, s0
	// add A10_ptr, A11_ptr, s0
	// add A20_ptr, A21_ptr, s0
	// add A30_ptr, A31_ptr, s0

	//th.vfmacc.vf AB30, B00, A30   // AB(3,:) += A(3,0) * B(0,:)
	//th.vfmacc.vf AB31, B01, A30
	//th.vfmacc.vf AB32, B02, A30
	//th.vfmacc.vf AB33, B03, A30

	th.vfmacc.vf AB00, B10, A01   // AB(0,:) += A(0,1) * B(1,:)
	th.vfmacc.vf AB01, B11, A01
	th.vfmacc.vf AB02, B12, A01
	th.vfmacc.vf AB03, B13, A01

	// Load B(l+2,0:3)
	FLOAD B00, 0*DATASIZE(B_row_ptr)
	FLOAD B01, 1*DATASIZE(B_row_ptr)
	FLOAD B02, 2*DATASIZE(B_row_ptr)
	FLOAD B03, 3*DATASIZE(B_row_ptr)

	//th.vfmacc.vf AB10, B10, A11   // AB(1,:) += A(1,1) * B(1,:)
	//th.vfmacc.vf AB11, B11, A11
	//th.vfmacc.vf AB12, B12, A11
	//th.vfmacc.vf AB13, B13, A11

	// Load A(:,l+2)
	VLE A00, (A00_ptr)
	//VLE A10, (A10_ptr)
	// VLE A20, (A20_ptr)
	//VLE A30, (A30_ptr)

	// Point to A(:,l+3)
	add A01_ptr, A00_ptr, s0
	// add A11_ptr, A10_ptr, s0
	// add A21_ptr, A20_ptr, s0
	// add A31_ptr, A30_ptr, s0

	// th.vfmacc.vf AB20, B10, A21   // AB(2,:) += A(2,1) * B(1,:)
	// th.vfmacc.vf AB21, B11, A21
	// th.vfmacc.vf AB22, B12, A21
	// th.vfmacc.vf AB23, B13, A21

	//th.vfmacc.vf AB30, B10, A31   // AB(3,:) += A(3,1) * B(1,:)
	//th.vfmacc.vf AB31, B11, A31
	//th.vfmacc.vf AB32, B12, A31
	//th.vfmacc.vf AB33, B13, A31

	// Load A(:,l+3)
	VLE A01, (A01_ptr)
	//VLE A11, (A11_ptr)
	// VLE A21, (A21_ptr)
	//VLE A31, (A31_ptr)

	// Point to A(:,l+4)
	add A00_ptr, A01_ptr, s0
	// add A10_ptr, A11_ptr, s0
	// add A20_ptr, A21_ptr, s0
	// add A30_ptr, A31_ptr, s0

	th.vfmacc.vf AB00, B00, A00   // AB(0,:) += A(0,2) * B(2,:)
	th.vfmacc.vf AB01, B01, A00
	th.vfmacc.vf AB02, B02, A00
	th.vfmacc.vf AB03, B03, A00

	// Load B(l+3,0:3)
	FLOAD B10, 4*DATASIZE(B_row_ptr)
	FLOAD B11, 5*DATASIZE(B_row_ptr)
	FLOAD B12, 6*DATASIZE(B_row_ptr)
	FLOAD B13, 7*DATASIZE(B_row_ptr)
	addi B_row_ptr, B_row_ptr, 8*DATASIZE

	//th.vfmacc.vf AB10, B00, A10   // AB(1,:) += A(1,2) * B(2,:)
	//th.vfmacc.vf AB11, B01, A10
	//th.vfmacc.vf AB12, B02, A10
	//th.vfmacc.vf AB13, B03, A10

	// th.vfmacc.vf AB20, B00, A20   // AB(2,:) += A(2,2) * B(2,:)
	// th.vfmacc.vf AB21, B01, A20
	// th.vfmacc.vf AB22, B02, A20
	// th.vfmacc.vf AB23, B03, A20

	//th.vfmacc.vf AB30, B00, A30   // AB(3,:) += A(3,2) * B(3,:)
	//th.vfmacc.vf AB31, B01, A30
	//th.vfmacc.vf AB32, B02, A30
	//th.vfmacc.vf AB33, B03, A30

	th.vfmacc.vf AB00, B10, A01   // AB(0,:) += A(0,3) * B(3,:)
	th.vfmacc.vf AB01, B11, A01
	th.vfmacc.vf AB02, B12, A01
	th.vfmacc.vf AB03, B13, A01

	//th.vfmacc.vf AB10, B10, A11   // AB(1,:) += A(1,3) * B(3,:)
	//th.vfmacc.vf AB11, B11, A11
	//th.vfmacc.vf AB12, B12, A11
	//th.vfmacc.vf AB13, B13, A11

	// th.vfmacc.vf AB20, B10, A21   // AB(2,:) += A(2,3) * B(3,:)
	// th.vfmacc.vf AB21, B11, A21
	// th.vfmacc.vf AB22, B12, A21
	// th.vfmacc.vf AB23, B13, A21

	//th.vfmacc.vf AB30, B10, A31   // AB(3,:) += A(3,3) * B(3,:)
	//th.vfmacc.vf AB31, B11, A31
	//th.vfmacc.vf AB32, B12, A31
	//th.vfmacc.vf AB33, B13, A31

	li tmp, 3
	ble loop_counter, tmp, TAIL_UNROLL_2

	// Load A and B for the next iteration
	// Load B(l,0:3)
	FLOAD B00, 0*DATASIZE(B_row_ptr)
	FLOAD B01, 1*DATASIZE(B_row_ptr)
	FLOAD B02, 2*DATASIZE(B_row_ptr)
	FLOAD B03, 3*DATASIZE(B_row_ptr)

	// Load A(:,l)
	VLE A00, (A00_ptr)
	//VLE A10, (A10_ptr)
	// VLE A20, (A20_ptr)
	//VLE A30, (A30_ptr)

	// Set up pointers to A(:,l+1)
	add A01_ptr, A00_ptr, s0
	// add A11_ptr, A10_ptr, s0
	// add A21_ptr, A20_ptr, s0
	// add A31_ptr, A30_ptr, s0

	j LOOP_UNROLL_4

TAIL_UNROLL_2: // loop_counter <= 3
	li tmp, 1
	ble loop_counter, tmp, TAIL_UNROLL_1

	addi loop_counter, loop_counter, -2

	// Load B(l,0:3)
	FLOAD B00, 0*DATASIZE(B_row_ptr)
	FLOAD B01, 1*DATASIZE(B_row_ptr)
	FLOAD B02, 2*DATASIZE(B_row_ptr)
	FLOAD B03, 3*DATASIZE(B_row_ptr)

	// Load A(0:1,l)
	VLE A00, (A00_ptr)
	//VLE A10, (A10_ptr)

	// Point to A(:,l+1)
	add A01_ptr, A00_ptr, s0
	// add A11_ptr, A10_ptr, s0
	// add A21_ptr, A20_ptr, s0
	// add A31_ptr, A30_ptr, s0

	th.vfmacc.vf AB00, B00, A00   // AB(0,:) += A(0,0) * B(0,:)
	th.vfmacc.vf AB01, B01, A00
	th.vfmacc.vf AB02, B02, A00
	th.vfmacc.vf AB03, B03, A00

	// Load A(2:3,l)
	// VLE A20, (A20_ptr)
	//VLE A30, (A30_ptr)

	//th.vfmacc.vf AB10, B00, A10   // AB(1,:) += A(1,0) * B(0,:)
	//th.vfmacc.vf AB11, B01, A10
	//vth.vfmacc.vf AB12, B02, A10
	//th.vfmacc.vf AB13, B03, A10

	// Load B(l+1,0:3)
	FLOAD B10, 4*DATASIZE(B_row_ptr)
	FLOAD B11, 5*DATASIZE(B_row_ptr)
	FLOAD B12, 6*DATASIZE(B_row_ptr)
	FLOAD B13, 7*DATASIZE(B_row_ptr)
	addi B_row_ptr, B_row_ptr, 8*DATASIZE

	// Load A(:,l+1)
	VLE A01, (A01_ptr)
	//VLE A11, (A11_ptr)
	// VLE A21, (A21_ptr)
	//VLE A31, (A31_ptr)

	// th.vfmacc.vf AB20, B00, A20   // AB(2,:) += A(2,0) * B(0,:)
	// th.vfmacc.vf AB21, B01, A20
	// th.vfmacc.vf AB22, B02, A20
	// th.vfmacc.vf AB23, B03, A20

	//th.vfmacc.vf AB30, B00, A30   // AB(3,:) += A(3,0) * B(0,:)
	//th.vfmacc.vf AB31, B01, A30
	//th.vfmacc.vf AB32, B02, A30
	//th.vfmacc.vf AB33, B03, A30

	// Point to A(:,l+2)
	add A00_ptr, A01_ptr, s0
	// add A10_ptr, A11_ptr, s0
	// add A20_ptr, A21_ptr, s0
	// add A30_ptr, A31_ptr, s0

	th.vfmacc.vf AB00, B10, A01   // AB(0,:) += A(0,1) * B(1,:)
	th.vfmacc.vf AB01, B11, A01
	th.vfmacc.vf AB02, B12, A01
	th.vfmacc.vf AB03, B13, A01

	//th.vfmacc.vf AB10, B10, A11   // AB(1,:) += A(1,1) * B(1,:)
	//th.vfmacc.vf AB11, B11, A11
	//th.vfmacc.vf AB12, B12, A11
	//th.vfmacc.vf AB13, B13, A11

	// th.vfmacc.vf AB20, B10, A21   // AB(2,:) += A(2,1) * B(1,:)
	// th.vfmacc.vf AB21, B11, A21
	// th.vfmacc.vf AB22, B12, A21
	// th.vfmacc.vf AB23, B13, A21

	//th.vfmacc.vf AB30, B10, A31   // AB(3,:) += A(3,1) * B(1,:)
	//th.vfmacc.vf AB31, B11, A31
	//th.vfmacc.vf AB32, B12, A31
	//th.vfmacc.vf AB33, B13, A31

	li tmp, 1
	ble loop_counter, tmp, TAIL_UNROLL_1

TAIL_UNROLL_1: // loop_counter <= 1
	beqz loop_counter, MULTIPLYALPHA

	// Load row of B
	FLOAD B00, 0*DATASIZE(B_row_ptr)
	FLOAD B01, 1*DATASIZE(B_row_ptr)
	FLOAD B02, 2*DATASIZE(B_row_ptr)
	FLOAD B03, 3*DATASIZE(B_row_ptr)

	// Load A(:,l)
	VLE A00, (A00_ptr)
	//VLE A10, (A10_ptr)
	// VLE A20, (A20_ptr)
	//VLE A30, (A30_ptr)

	th.vfmacc.vf AB00, B00, A00   // AB(0,:) += A(0,0) * B(0,:)
	th.vfmacc.vf AB01, B01, A00
	th.vfmacc.vf AB02, B02, A00
	th.vfmacc.vf AB03, B03, A00

	//th.vfmacc.vf AB10, B00, A10   // AB(1,:) += A(1,0) * B(0,:)
	//th.vfmacc.vf AB11, B01, A10
	//th.vfmacc.vf AB12, B02, A10
	//th.vfmacc.vf AB13, B03, A10

	// th.vfmacc.vf AB20, B00, A20   // AB(2,:) += A(2,0) * B(0,:)
	// th.vfmacc.vf AB21, B01, A20
	// th.vfmacc.vf AB22, B02, A20
	// th.vfmacc.vf AB23, B03, A20

	//th.vfmacc.vf AB30, B00, A30   // AB(3,:) += A(3,0) * B(0,:)
	//th.vfmacc.vf AB31, B01, A30
	//th.vfmacc.vf AB32, B02, A30
	//th.vfmacc.vf AB33, B03, A30

MULTIPLYALPHA:
	FLOAD ALPHA, (a1)

	// Multiply with alpha
	th.vfmul.vf AB00, AB00, ALPHA
	th.vfmul.vf AB01, AB01, ALPHA
	th.vfmul.vf AB02, AB02, ALPHA
	th.vfmul.vf AB03, AB03, ALPHA

	//th.vfmul.vf AB10, AB10, ALPHA
	//th.vfmul.vf AB11, AB11, ALPHA
	//th.vfmul.vf AB12, AB12, ALPHA
	//th.vfmul.vf AB13, AB13, ALPHA

	// th.vfmul.vf AB20, AB20, ALPHA
	// th.vfmul.vf AB21, AB21, ALPHA
	// th.vfmul.vf AB22, AB22, ALPHA
	// th.vfmul.vf AB23, AB23, ALPHA

	//th.vfmul.vf AB30, AB30, ALPHA
	//th.vfmul.vf AB31, AB31, ALPHA
	//th.vfmul.vf AB32, AB32, ALPHA
	//th.vfmul.vf AB33, AB33, ALPHA

MULTIPLYBETA:
	FLOAD BETA,  (a4)
	FEQ tmp, BETA, fzero
	beq tmp, zero, BETANOTZERO

BETAZERO:

	VSE AB00, (C00_ptr)
	VSE AB01, (C01_ptr)
	VSE AB02, (C02_ptr)
	VSE AB03, (C03_ptr)

	// add C00_ptr, C10_ptr, rs_c  // advance pointers to row 2*VLEN
	// add C01_ptr, C11_ptr, rs_c
	// add C02_ptr, C12_ptr, rs_c
	// add C03_ptr, C13_ptr, rs_c

	// VSE AB10, (C10_ptr)
	// VSE AB11, (C11_ptr)
	// VSE AB12, (C12_ptr)
	// VSE AB13, (C13_ptr)

	// add C10_ptr, C00_ptr, rs_c  // advance pointers to row 3*VLEN
	// add C11_ptr, C01_ptr, rs_c
	// add C12_ptr, C02_ptr, rs_c
	// add C13_ptr, C03_ptr, rs_c

	// VSE AB20, (C00_ptr)
	// VSE AB21, (C01_ptr)
	// VSE AB22, (C02_ptr)
	// VSE AB23, (C03_ptr)

	// VSE AB30, (C10_ptr)
	// VSE AB31, (C11_ptr)
	// VSE AB32, (C12_ptr)
	// VSE AB33, (C13_ptr)

	j END

BETANOTZERO:
	VLE C00, (C00_ptr)  // Load C(0:VLEN-1, 0:3)
	VLE C01, (C01_ptr)
	VLE C02, (C02_ptr)
	VLE C03, (C03_ptr)

	th.vfmacc.vf AB00, BETA, C00
	th.vfmacc.vf AB01, BETA, C01
	th.vfmacc.vf AB02, BETA, C02
	th.vfmacc.vf AB03, BETA, C03

	VSE AB00, (C00_ptr)  // Store C(0:VLEN-1, 0:3)
	VSE AB01, (C01_ptr)
	VSE AB02, (C02_ptr)
	VSE AB03, (C03_ptr)

	// add C00_ptr, C10_ptr, rs_c  // advance pointers to row 2*VLEN
	// add C01_ptr, C11_ptr, rs_c
	// add C02_ptr, C12_ptr, rs_c
	// add C03_ptr, C13_ptr, rs_c

	// VLE C10, (C10_ptr)  // Load C(VLEN:2*VLEN-1, 0:3)
	// VLE C11, (C11_ptr)
	// VLE C12, (C12_ptr)
	// VLE C13, (C13_ptr)

	// th.vfmacc.vf AB10, BETA, C10
	// th.vfmacc.vf AB11, BETA, C11
	// th.vfmacc.vf AB12, BETA, C12
	// th.vfmacc.vf AB13, BETA, C13

	// VSE AB10, (C10_ptr)  // Store C(VLEN:2*VLEN-1, 0:3)
	// VSE AB11, (C11_ptr)
	// VSE AB12, (C12_ptr)
	// VSE AB13, (C13_ptr)

	// add C10_ptr, C00_ptr, rs_c  // advance pointers to row 3*VLEN
	// add C11_ptr, C01_ptr, rs_c
	// add C12_ptr, C02_ptr, rs_c
	// add C13_ptr, C03_ptr, rs_c

	// VLE C20, (C00_ptr)  // Load C(2*VLEN:3*VLEN-1, 0:3)
	// VLE C21, (C01_ptr)
	// VLE C22, (C02_ptr)
	// VLE C23, (C03_ptr)

	// th.vfmacc.vf AB20, BETA, C20
	// th.vfmacc.vf AB21, BETA, C21
	// th.vfmacc.vf AB22, BETA, C22
	// th.vfmacc.vf AB23, BETA, C23

	// VSE AB20, (C00_ptr)  // Store C(2*VLEN:3*VLEN-1, 0:3)
	// VSE AB21, (C01_ptr)
	// VSE AB22, (C02_ptr)
	// VSE AB23, (C03_ptr)

	// VLE C30, (C10_ptr)  // Load C(3*VLEN:4*VLEN-1, 0:3)
	// VLE C31, (C11_ptr)
	// VLE C32, (C12_ptr)
	// VLE C33, (C13_ptr)

	// th.vfmacc.vf AB30, BETA, C30
	// th.vfmacc.vf AB31, BETA, C31
	// th.vfmacc.vf AB32, BETA, C32
	// th.vfmacc.vf AB33, BETA, C33

	// VSE AB30, (C10_ptr)  // Store C(3*VLEN:4*VLEN-1, 0:3)
	// VSE AB31, (C11_ptr)
	// VSE AB32, (C12_ptr)
	// VSE AB33, (C13_ptr)

END:
	#include "rvv_sg2042_restore_registers.h"
	ret
