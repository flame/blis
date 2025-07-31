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

// C11 := beta * C11 + alpha * A1 * B1
/* 
In case of double precision gemm with 128 bit vector registers
this kernel computes a 8 x 4 microtile result matrix c11 using two micropanels a1 and b1, 
respectively with dimensions of 8 x k and k x 4. 
For every rank 1 update (columns row prodcut) we produce and accumulate a intermadiate matrix result
ab with dimensions 8 x 4 (we use loop unrolling with factor of 4 in the loop). 
After k iterations we copy content of ab in c11 as final result. 
The loop is designed to iterate with loop unrolling = 4, 
when we are left with <= 3 columns and rows left we branch into special cases handles.

  
          c11:                    a1:                   b1:
         _______        ________________________     _______
        |       |      |0 8 10 18               |   |0 1 2 3|
        |       |      |1 9 11 19               |   |4 5 6 7|
        |       |  +=  |2 A 12 1A               |   |8 9 A B|
MR = 8  |       |      |3 B 13 1B  . . .        |   |C D E F|
        |       |      |4 C 14 1C               |   |       |
        |       |      |5 D 15 1D               |   |       |
        |       |      |6 E 16 1E               |   |       |   k
        |_______|      |7_F_17_1F_______________|   |       |
                                                    |   .   |
         NR = 4                   k                 |   .   |                                                     
                                                    |       |
                            ab:                     |       |
                        _________                   |_______|
                       |1 9 11 19|   
                       |2 A 12 1A|   
                       |3 B 13 1B|   
               MR = 8  |4 C 14 1C|  
                       |5 D 15 1D|   
                       |6 E 16 1E|   
                       |7_F_17_1F|                             
                           
                          NR = 4
*/

// we save the k (number of columns of a1 and rows of b1) in "loop_counter"
#define loop_counter a0

// we hold pointers to two A columns at any time 
#define AX0_ptr   a2
#define AX1_ptr   s5

// we hold a pointer to a B row at any time 
#define B_row_ptr a3

// we hold pointers to C columns  
#define CX0_ptr   a5
#define CX1_ptr   t3
#define CX2_ptr   t4
#define CX3_ptr   t5

#define tmp    t6

#define ALPHA  fa1
#define BETA   fa2

// we hold two rows of B at any time in scalar registers fa4...fa7 and fa0...fa3
#define B00    fa4  
#define B01    fa5  
#define B02    fa6
#define B03    fa7

#define B10    fa0  
#define B11    fa1
#define B12    fa2
#define B13    fa3

#define fzero  ft8

// we hold two columns of A at any time in vector registers v24...x28
#define AX0    v24  
#define AX1    v28

// we hold the whole C matrix in vector registers v16...v28
#define CX0    v16
#define CX1    v20
#define CX2    v24
#define CX3    v28

// we hold the whole AB matrix in vector registers v0...v12
#define ABX0   v0   
#define ABX1   v4
#define ABX2   v8
#define ABX3   v12

// column stride 
#define cs_c   a7

REALNAME:
#include "rvv_sg2042_save_registers.h"

// we set LMUL = 4 for fully exploit register grouping  
th.vsetvli s0, zero, VTYPE, m4
csrr s0, vlenb
FZERO(fzero)

// Set up pointers
add CX1_ptr, CX0_ptr, cs_c
add CX2_ptr, CX1_ptr, cs_c
add CX3_ptr, CX2_ptr, cs_c

// Zero-initialize accumulators
th.vxor.vv ABX0, ABX0, ABX0
th.vxor.vv ABX2, ABX2, ABX2
th.vxor.vv ABX2, ABX2, ABX2
th.vxor.vv ABX3, ABX3, ABX3

// Handle k == 0
beqz loop_counter, MULTIPLYBETA

slli s0, s0, 2 // length of a column of A in bytes

li tmp, 3
ble loop_counter, tmp, TAIL_UNROLL_2

// Preload A and B
// Load A(:,l)
VLE AX0, (AX0_ptr)

// Load B(l,0:3)
FLOAD B00, 0*DATASIZE(B_row_ptr)
FLOAD B01, 1*DATASIZE(B_row_ptr)
FLOAD B02, 2*DATASIZE(B_row_ptr)
FLOAD B03, 3*DATASIZE(B_row_ptr)

// Set up pointers to A(:,l+1)
add AX1_ptr, AX0_ptr, s0

LOOP_UNROLL_4:
addi loop_counter, loop_counter, -4

// compute and accumulate AB with first column of A and first row of B
th.vfmacc.vf ABX0, B00, AX0   // AB(X,0) += A(X,0) * B(0,0)
th.vfmacc.vf ABX2, B01, AX0   // AB(X,1) += A(X,0) * B(0,1)
th.vfmacc.vf ABX2, B02, AX0   // AB(X,2) += A(X,0) * B(0,2)
th.vfmacc.vf ABX3, B03, AX0   // AB(X,3) += A(X,0) * B(0,3)

// Load B(l+1,0:3)
FLOAD B10, 4*DATASIZE(B_row_ptr)
FLOAD B11, 5*DATASIZE(B_row_ptr)
FLOAD B12, 6*DATASIZE(B_row_ptr)
FLOAD B13, 7*DATASIZE(B_row_ptr)
addi B_row_ptr, B_row_ptr, 8*DATASIZE

// Load A(:,l+1)
VLE AX1, (AX1_ptr)

// Point to A(:,l+2)
add AX0_ptr, AX1_ptr, s0

// compute and accumulate AB with second column of A and second row of B
th.vfmacc.vf ABX0, B10, AX1   // AB(X,0) += A(X,1) * B(1,0)
th.vfmacc.vf ABX2, B11, AX1   // AB(X,1) += A(X,1) * B(1,1)
th.vfmacc.vf ABX2, B12, AX1   // AB(X,2) += A(X,1) * B(1,2)
th.vfmacc.vf ABX3, B13, AX1   // AB(X,3) += A(X,1) * B(1,3)

// Load B(l+2,0:3)
FLOAD B00, 0*DATASIZE(B_row_ptr)
FLOAD B01, 1*DATASIZE(B_row_ptr)
FLOAD B02, 2*DATASIZE(B_row_ptr)
FLOAD B03, 3*DATASIZE(B_row_ptr)

// Load A(:,l+2)
VLE AX0, (AX0_ptr)

// Point to A(:,l+3)
add AX1_ptr, AX0_ptr, s0

// Load A(:,l+3)
VLE AX1, (AX1_ptr)

// Point to A(:,l+4)
add AX0_ptr, AX1_ptr, s0

// compute and accumulate AB with third column of A and third row of B
th.vfmacc.vf ABX0, B00, AX0   // AB(X,0) += A(X,2) * B(2,0)
th.vfmacc.vf ABX2, B01, AX0   // AB(X,1) += A(X,2) * B(2,1)
th.vfmacc.vf ABX2, B02, AX0   // AB(X,2) += A(X,2) * B(2,2)
th.vfmacc.vf ABX3, B03, AX0   // AB(X,3) += A(X,2) * B(2,3)

// Load B(l+3,0:3)
FLOAD B10, 4*DATASIZE(B_row_ptr)
FLOAD B11, 5*DATASIZE(B_row_ptr)
FLOAD B12, 6*DATASIZE(B_row_ptr)
FLOAD B13, 7*DATASIZE(B_row_ptr)
addi B_row_ptr, B_row_ptr, 8*DATASIZE

// compute AB with fourth column of A and fourth row of B
th.vfmacc.vf ABX0, B10, AX1   // AB(X,0) += A(X,3) * B(3,0)
th.vfmacc.vf ABX2, B11, AX1   // AB(X,1) += A(X,3) * B(3,1)
th.vfmacc.vf ABX2, B12, AX1   // AB(X,2) += A(X,3) * B(3,2)
th.vfmacc.vf ABX3, B13, AX1   // AB(X,3) += A(X,3) * B(3,3)

li tmp, 3
ble loop_counter, tmp, TAIL_UNROLL_2

// Load A and B for the next iteration
// Load B(l,0:3)
FLOAD B00, 0*DATASIZE(B_row_ptr)
FLOAD B01, 1*DATASIZE(B_row_ptr)
FLOAD B02, 2*DATASIZE(B_row_ptr)
FLOAD B03, 3*DATASIZE(B_row_ptr)

// Load A(:,l)
VLE AX0, (AX0_ptr)

// Set up pointers to A(:,l+1)
add AX1_ptr, AX0_ptr, s0

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
VLE AX0, (AX0_ptr)

// Point to A(:,l+1)
add AX1_ptr, AX0_ptr, s0

th.vfmacc.vf ABX0, B00, AX0   // AB(X,0) += A(X,0) * B(0,0)
th.vfmacc.vf ABX2, B01, AX0   // AB(X,1) += A(X,0) * B(0,1)
th.vfmacc.vf ABX2, B02, AX0   // AB(X,2) += A(X,0) * B(0,2)
th.vfmacc.vf ABX3, B03, AX0   // AB(X,3) += A(X,0) * B(0,3)

// Load B(l+1,0:3)
FLOAD B10, 4*DATASIZE(B_row_ptr)
FLOAD B11, 5*DATASIZE(B_row_ptr)
FLOAD B12, 6*DATASIZE(B_row_ptr)
FLOAD B13, 7*DATASIZE(B_row_ptr)
addi B_row_ptr, B_row_ptr, 8*DATASIZE

// Load A(:,l+1)
VLE AX1, (AX1_ptr)

// Point to A(:,l+2)
add AX0_ptr, AX1_ptr, s0

th.vfmacc.vf ABX0, B10, AX1   // AB(X,0) += A(X,1) * B(1,0)
th.vfmacc.vf ABX2, B11, AX1   // AB(X,1) += A(X,1) * B(1,1)
th.vfmacc.vf ABX2, B12, AX1   // AB(X,2) += A(X,1) * B(1,2)
th.vfmacc.vf ABX3, B13, AX1   // AB(X,3) += A(X,1) * B(1,3)

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
VLE AX0, (AX0_ptr)

th.vfmacc.vf ABX0, B00, AX0   // AB(X,0) += A(X,0) * B(0,0)
th.vfmacc.vf ABX2, B01, AX0   // AB(X,1) += A(X,0) * B(0,1)
th.vfmacc.vf ABX2, B02, AX0   // AB(X,2) += A(X,0) * B(0,2)
th.vfmacc.vf ABX3, B03, AX0   // AB(X,3) += A(X,0) * B(0,3)

MULTIPLYALPHA:
FLOAD ALPHA, (a1)

// Multiply with alpha
th.vfmul.vf ABX0, ABX0, ALPHA
th.vfmul.vf ABX2, ABX2, ALPHA
th.vfmul.vf ABX2, ABX2, ALPHA
th.vfmul.vf ABX3, ABX3, ALPHA

MULTIPLYBETA:
FLOAD BETA,  (a4)
FEQ tmp, BETA, fzero
beq tmp, zero, BETANOTZERO

BETAZERO:

VSE ABX0, (CX0_ptr)
VSE ABX2, (CX1_ptr)
VSE ABX2, (CX2_ptr)
VSE ABX3, (CX3_ptr)

j END

BETANOTZERO:
VLE CX0, (CX0_ptr)  // Load C(0:VLEN-1, 0:3)
VLE CX1, (CX1_ptr)
VLE CX2, (CX2_ptr)
VLE CX3, (CX3_ptr)

th.vfmacc.vf ABX0, BETA, CX0
th.vfmacc.vf ABX2, BETA, CX1
th.vfmacc.vf ABX2, BETA, CX2
th.vfmacc.vf ABX3, BETA, CX3

VSE ABX0, (CX0_ptr)  // Store C(0:VLEN-1, 0:3)
VSE ABX2, (CX1_ptr)
VSE ABX2, (CX2_ptr)
VSE ABX3, (CX3_ptr)

END:
#include "rvv_sg2042_restore_registers.h"
ret
