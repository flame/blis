/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2024 - 2025, Advanced Micro Devices, Inc. All rights reserved.

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
#include "immintrin.h"
#define BLIS_ASM_SYNTAX_ATT
#include "bli_x86_asm_macros.h"

// --------------------------------------------------------------------------------------

/*
    Functionality
    -------------

    This function copies a vector x to a vector y for
    type float.

    y := conj?(x)

    Function Signature
    -------------------

    * 'conjx' - Variable specified if x needs to be conjugated
    * 'n' - Length of the array passed
    * 'x' - Float pointer pointing to an array
    * 'y' - Float pointer pointing to an array
    * 'incx' - Stride to point to the next element in x array
    * 'incy' - Stride to point to jthe next element in y array
    * 'cntx' - BLIS context object

    Exception
    ----------

    None

    Deviation from BLAS
    --------------------

    None

    Undefined behaviour
    -------------------

    1. The kernel results in undefined behaviour when n < 0, incx < 1 and incy < 1.
       The expectation is that these are standard BLAS exceptions and should be handled in
       a higher layer
*/

void bli_scopyv_zen4_asm_avx512
(
    conj_t           conjx,
    dim_t            n,
    float*  restrict x, inc_t incx,
    float*  restrict y, inc_t incy,
    cntx_t* restrict cntx
)
{
    AOCL_DTL_TRACE_ENTRY(AOCL_DTL_LEVEL_TRACE_2)

    // Initialize local pointers.
    float *x0 = x;
    float *y0 = y;

    // Typecast int to 64 bit
    uint64_t n0 = (uint64_t)n;
    int64_t incy0 = (int64_t)incy;
    int64_t incx0 = (int64_t)incx;

    // If the vector dimension is zero return early.
    if (bli_zero_dim1(n))
    {
        AOCL_DTL_TRACE_EXIT(AOCL_DTL_LEVEL_TRACE_2)
        return;
    }

    // Assembly Code
    begin_asm()

     /*
        rsi - > n
        rdx - > x
        rcx - > incx
        r8  - > y
        r9  - > incy
    */

    // Loading the source memory address to the respective registers
    mov(var(x0), rdx)
    mov(var(y0),  r8)

    // Loading the values in 'n', 'incx' and 'incy' to the respective registers
    mov(var(n0), rsi)
    mov(var(incx0), rcx)
    mov(var(incy0), r9)

    // Checking if incx == 1 and incy == 1, incase the condition fails then SCALAR code section is executed
    cmp(imm(1),rcx)
    jne(.SCALAR)
    cmp(imm(1),r9)
    jne(.SCALAR)

    // ========================================================================================================================

    // Section of code to move the data as blocks of 256 elements
    label(.BLOCK256)

    cmp(imm(16*16), rsi)               // check if the number of remaining elements greater than or equal to 256
    jl(.BLOCK128)                      // else, goto to the section of code for block of size 128

    label(.MAINLOOP)

    // Interleaved SIMD load and store operations to copy data from source to the destination
    // Each vector register can hold 16 elements and is used twice before next jump operation 
    // 1 for loading the element from source and 1 for store it into the destination

    vmovups(mem(rdx, 0*64), zmm0)       // zmm0 = x[i+0] - x[i+15]
    vmovups(zmm0,  mem(r8, 0*64))       // y[i+0] - y[i+15] = zmm0
    vmovups(mem(rdx, 1*64), zmm1)       // zmm1 = x[i+16] - x[i+31]
    vmovups(zmm1,  mem(r8, 1*64))       // y[i+16] - y[i+31] = zmm1
    vmovups(mem(rdx, 2*64), zmm2)       // zmm2 = x[i+32] - x[i+47]
    vmovups(zmm2,  mem(r8, 2*64))       // y[i+32] - y[i+47] = zmm2
    vmovups(mem(rdx, 3*64), zmm3)       // zmm3 = x[i+48] - x[i+63]
    vmovups(zmm3,  mem(r8, 3*64))       // y[i+48] - y[i+63] = zmm3

    vmovups(mem(rdx, 4*64), zmm4)       // zmm4 = x[i+64] - x[i+79]
    vmovups(zmm4,  mem(r8, 4*64))       // y[i+64] - y[i+79] = zmm4
    vmovups(mem(rdx, 5*64), zmm5)       // zmm5 = x[i+80] - x[i+95]
    vmovups(zmm5,  mem(r8, 5*64))       // y[i+80] - y[i+95] = zmm5
    vmovups(mem(rdx, 6*64), zmm6)       // zmm6 = x[i+96] - x[i+111]
    vmovups(zmm6,  mem(r8, 6*64))       // y[i+96] - y[i+111] = zmm6
    vmovups(mem(rdx, 7*64), zmm7)       // zmm7 = x[i+112] - x[i+127]
    vmovups(zmm7,  mem(r8, 7*64))       // y[i+112] - y[i+127] = zmm7

    vmovups(mem(rdx, 8*64), zmm8)       // zmm8 = x[i+128] - x[i+143]
    vmovups(zmm8,  mem(r8, 8*64))       // y[i+128] - y[i+143] = zmm8
    vmovups(mem(rdx, 9*64), zmm9)       // zmm9 = x[i+144] - x[i+159]
    vmovups(zmm9,  mem(r8, 9*64))       // y[i+144] - y[i+159] = zmm9
    vmovups(mem(rdx, 10*64), zmm10)     // zmm10 = x[i+160] - x[i+175]
    vmovups(zmm10,  mem(r8, 10*64))     // y[i+160] - y[i+175] = zmm10
    vmovups(mem(rdx, 11*64), zmm11)     // zmm11 = x[i+176] - x[i+191]
    vmovups(zmm11,  mem(r8, 11*64))     // y[i+176] - y[i+191] = zmm11

    vmovups(mem(rdx, 12*64), zmm12)     // zmm12 = x[i+192] - x[i+207]
    vmovups(zmm12,  mem(r8, 12*64))     // y[i+192] - y[i+207] = zmm12
    vmovups(mem(rdx, 13*64), zmm13)     // zmm13 = x[i+208] - x[i+223]
    vmovups(zmm13,  mem(r8, 13*64))     // y[i+208] - y[i+223] = zmm13
    vmovups(mem(rdx, 14*64), zmm14)     // zmm14 = x[i+224] - x[i+239]
    vmovups(zmm14,  mem(r8, 14*64))     // y[i+224] - y[i+239] = zmm14
    vmovups(mem(rdx, 15*64), zmm15)     // zmm15 = x[i+240] - x[i+255]
    vmovups(zmm15,  mem(r8, 15*64))     // y[i+240] - y[i+255] = zmm15

    // Increment the pointer
    add(imm(16*4*16), rdx)
    add(imm(16*4*16),  r8)
    sub(imm(16*16),   rsi)             // reduce the number of remaining elements by 256

    cmp(imm(16*16), rsi)
    jge(.MAINLOOP)

    // -----------------------------------------------------------

    // Section of code to move the data as blocks of 128 elements
    label(.BLOCK128)

    cmp(imm(16*8), rsi)                // check if the number of remaining elements greater than or equal to 128
    jl(.BLOCK64)                       // else, goto to the section of code for block of size 64

    // Interleaved SIMD load and store operations to copy data from source to the destination

    vmovups(mem(rdx, 0*64), zmm0)       // zmm0 = x[i+0] - x[i+15]
    vmovups(zmm0,  mem(r8, 0*64))       // y[i+0] - y[i+15] = zmm0
    vmovups(mem(rdx, 1*64), zmm1)       // zmm1 = x[i+16] - x[i+31]
    vmovups(zmm1,  mem(r8, 1*64))       // y[i+16] - y[i+31] = zmm1
    vmovups(mem(rdx, 2*64), zmm2)       // zmm2 = x[i+32] - x[i+47]
    vmovups(zmm2,  mem(r8, 2*64))       // y[i+32] - y[i+47] = zmm2
    vmovups(mem(rdx, 3*64), zmm3)       // zmm3 = x[i+48] - x[i+63]
    vmovups(zmm3,  mem(r8, 3*64))       // y[i+48] - y[i+63] = zmm3

    vmovups(mem(rdx, 4*64), zmm4)       // zmm4 = x[i+64] - x[i+79]
    vmovups(zmm4,  mem(r8, 4*64))       // y[i+64] - y[i+79] = zmm4
    vmovups(mem(rdx, 5*64), zmm5)       // zmm5 = x[i+80] - x[i+95]
    vmovups(zmm5,  mem(r8, 5*64))       // y[i+80] - y[i+95] = zmm5
    vmovups(mem(rdx, 6*64), zmm6)       // zmm6 = x[i+96] - x[i+111]
    vmovups(zmm6,  mem(r8, 6*64))       // y[i+96] - y[i+111] = zmm6
    vmovups(mem(rdx, 7*64), zmm7)       // zmm7 = x[i+112] - x[i+127]
    vmovups(zmm7,  mem(r8, 7*64))       // y[i+112] - y[i+127] = zmm7

    // Increment the pointer
    add(imm(16*4*8), rdx)
    add(imm(16*4*8),  r8)
    sub(imm(16*8),   rsi)              // reduce the number of remaining elements by 128

    // -----------------------------------------------------------

    // Section of code to move the data as blocks of 64 elements
    label(.BLOCK64)

    cmp(imm(16*4), rsi)                // check if the number of remaining elements greater than or equal to 64
    jl(.BLOCK32)                       // else, goto to the section of code for block of size 32

    // Interleaved SIMD load and store operations to copy data from source to the destination

    vmovups(mem(rdx, 0*64), zmm0)       // zmm0 = x[i+0] - x[i+15]
    vmovups(zmm0,  mem(r8, 0*64))       // y[i+0] - y[i+15] = zmm0
    vmovups(mem(rdx, 1*64), zmm1)       // zmm1 = x[i+16] - x[i+31]
    vmovups(zmm1,  mem(r8, 1*64))       // y[i+16] - y[i+31] = zmm1
    vmovups(mem(rdx, 2*64), zmm2)       // zmm2 = x[i+32] - x[i+47]
    vmovups(zmm2,  mem(r8, 2*64))       // y[i+32] - y[i+47] = zmm2
    vmovups(mem(rdx, 3*64), zmm3)       // zmm3 = x[i+48] - x[i+63]
    vmovups(zmm3,  mem(r8, 3*64))       // y[i+48] - y[i+63] = zmm3

    // Increment the pointer
    add(imm(16*4*4), rdx)
    add(imm(16*4*4),  r8)
    sub(imm(16*4),   rsi)              // reduce the number of remaining elements by 64

    // -----------------------------------------------------------

    // Section of code to move the data as blocks of 32 elements
    label(.BLOCK32)

    cmp(imm(16*2), rsi)                // check if the number of remaining elements greater than or equal to 32
    jl(.BLOCK16)                       // else, goto to the section of code for block of size 16

    // Interleaved SIMD load and store operations to copy data from source to the destination

    vmovups(mem(rdx, 0*64), zmm0)       // zmm0 = x[i+0] - x[i+15]
    vmovups(zmm0,  mem(r8, 0*64))       // y[i+0] - y[i+15] = zmm0
    vmovups(mem(rdx, 1*64), zmm1)       // zmm1 = x[i+16] - x[i+31]
    vmovups(zmm1,  mem(r8, 1*64))       // y[i+16] - y[i+31] = zmm1

    add(imm(16*4*2), rdx)
    add(imm(16*4*2),  r8)
    sub(imm(16*2),   rsi)              // reduce the number of remaining elements by 32

    // -----------------------------------------------------------

    // Section of code to move the data as blocks of 16 elements
    label(.BLOCK16)

    cmp(imm(16), rsi)                  // check if the number of remaining elements greater than or equal to 16
    jl(.FRINGE)                        // else, goto to the section of code for fringe cases

    // Loading and storing the values to destination

    vmovups(mem(rdx, 0*64), zmm0)       // zmm0 = x[i+0] - x[i+15]
    vmovups(zmm0,  mem(r8, 0*64))       // y[i+0] - y[i+15] = zmm0

    // Increment the pointer
    add(imm(16*4), rdx)
    add(imm(16*4),  r8)
    sub(imm(16),   rsi)                // reduce the number of remaining elements by 16

    // -----------------------------------------------------------

    // Section of code to deal with fringe cases
    label(.FRINGE)

    cmp(imm(0), rsi)                   // check if there is any fringe cases
    je(.END)

    // Creating a 8-bit mask
    mov(imm(65535), rcx)               // (65535)BASE_10 -> (1111 1111 1111 1111)BASE_2
    shlx(rsi,rcx,rcx)                  // shifting the bits in the register to the left depending on the number of fringe elements remaining
    xor(imm(65535),rcx)                // taking compliment of the register
    kmovq(rcx, k(2))                   // copying the value in the register to mask register

    /*
        Creating mask: Example - fringe case = 2
            step 1 : rdx = (1111 1111 1111 1111)BASE_2  or  (65535)BASE_10
            step 2 : rdx = (1111 1111 1111 1100)BASE_2  or  (65532)BASE_10
            step 3 : rdx = (0000 0000 0000 0011)BASE_2  or  (3)BASE_10
    */

    // Loading the input values using masked load
    vmovups(mem(rdx, 0*64), zmm0 MASK_(K(2)))

    // Storing the values to destination using masked store
    vmovups(zmm0,  mem(r8) MASK_(K(2)))

    // After the above instructions are executed, the remaining part are not executed
    jmp(.END)

    // ========================================================================================================================

    // Code section used to deal with situations where incx or incy is not 1
    label(.SCALAR)

    // incx and incy are multipled by 8 (shift left by 2 bits) and stored back into their respective registers
    mov(imm(2), r11)
    shlx(r11, rcx, rcx)
    shlx(r11, r9, r9)

    // A loop is used to move one element at a time to the destination
    label(.SCALARLOOP)

    // checking if all the elements are moved, then the loop will be terminated
    cmp(imm(0), rsi)
    je(.END)

    // Using vector register to mov one element at a time
    vmovss(mem(rdx, 0), xmm0)
    vmovss(xmm0,  mem(r8, 0))

    // Moving the address pointer of x and y array by incx*8 and incy*8 bytes
    add(rcx, rdx)
    add(r9,   r8)

    dec(rsi)
    jmp(.SCALARLOOP)

    label(.END)
    end_asm
    (
        : // output operands
        : // input operands
          [n0]     "m"     (n0),
          [x0]     "m"     (x0),
          [incx0]  "m"     (incx0),
          [y0]     "m"     (y0),
          [incy0]  "m"     (incy0)
        : // register clobber list
          "zmm0",  "zmm1",  "zmm2",  "zmm3",
          "zmm4",  "zmm5",  "zmm6",  "zmm7",
          "zmm8",  "zmm9",  "zmm10", "zmm11",
          "zmm12", "zmm13", "zmm14", "zmm15",
          "xmm0",  "rsi",   "rdx",   "rcx",
          "r8",    "r9",    "r11",   "k2",
          "memory"
    )

    AOCL_DTL_TRACE_EXIT(AOCL_DTL_LEVEL_TRACE_2)
}


// --------------------------------------------------------------------------------------

/*
    Functionality
    -------------

    This function copies a vector x to a vector y for
    type double.

    y := conj?(x)

    Function Signature
    -------------------

    * 'conjx' - Variable specified if x needs to be conjugated
    * 'n' - Length of the array passed
    * 'x' - Double pointer pointing to an array
    * 'y' - Double pointer pointing to an array
    * 'incx' - Stride to point to the next element in x array
    * 'incy' - Stride to point to the next element in y array
    * 'cntx' - BLIS context object

    Exception
    ----------

    None

    Deviation from BLAS
    --------------------

    None

    Undefined behaviour
    -------------------

    1. The kernel results in undefined behaviour when n < 0, incx < 1 and incy < 1.
       The expectation is that these are standard BLAS exceptions and should be handled in
       a higher layer
*/

void bli_dcopyv_zen4_asm_avx512
(
    conj_t           conjx,
    dim_t            n,
    double*  restrict x, dim_t incx,
    double*  restrict y, dim_t incy,
    cntx_t* restrict cntx
)
{
    AOCL_DTL_TRACE_ENTRY(AOCL_DTL_LEVEL_TRACE_2)

    // Initialize local pointers.
    double *x0 = x;
    double *y0 = y;

    // Typecast int to 64 bit
    uint64_t n0 = (uint64_t)n;
    int64_t incy0 = (int64_t)incy;
    int64_t incx0 = (int64_t)incx;

    // If the vector dimension is zero return early.
    if (bli_zero_dim1(n))
    {
        AOCL_DTL_TRACE_EXIT(AOCL_DTL_LEVEL_TRACE_2)
        return;
    }

    // assembly code
    begin_asm()

    /*
        rcx  - > n
        rsi  - > x
        r8   - > incx
        rdi  - > y
        r9   - > incy
    */

    // Loading the source and destination memory addresses into the respective registers
    mov(var(x0), rsi)
    mov(var(y0), rdi)

    // Loading the values in n, incx and inxy into the respective registers
    mov(var(n0),    rcx)
    mov(var(incx0), r8 )
    mov(var(incy0), r9 )

    // Checking if incx == 1 and incy == 1, incase the condition fails then SCALAR code section is executed
    cmp(imm(1), r8)
    jne(.SCALAR)
    cmp(imm(1),r9)
    jne(.SCALAR)

// ==========================================================================================================================

    // Section of code to move the data as blocks of 128 elements
    label(.BLOCK128)

    cmp(imm(8*16), rcx)                 // Check if the number of remaining elements greater than or equal to 128 -> (NUMBER OF ELEMENTS PER REGISTER) * (NUMBER OF REGISTERS USED IN THE BLOCK)
    jl(.BLOCK64)                        // Else, skip the BLOCK128 section and goto to BLOCK64 section of the code

    label(.MAINLOOP)
    
    // Interleaved SIMD load and store operations to copy data from source to the destination
    // Each vector register can hold 8 elements and is used twice before next jump operation 
    // 1 vmovupd for loading the element from source and 1 vmovupd for store it into the destination

    vmovupd(mem(rsi, 0*64), zmm0)       // zmm0 = x[i+0] - x[i+7]
    vmovupd(zmm0, mem(rdi, 0*64))       // y[i+0] - y[i+7] = zmm0
    vmovupd(mem(rsi, 1*64), zmm1)       // zmm1 = x[i+8] - x[i+15]
    vmovupd(zmm1, mem(rdi, 1*64))       // y[i+8] - y[i+15] = zmm1
    vmovupd(mem(rsi, 2*64), zmm2)       // zmm2 = x[i+16] - x[i+23]
    vmovupd(zmm2, mem(rdi, 2*64))       // y[i+16] - y[i+23] = zmm2
    vmovupd(mem(rsi, 3*64), zmm3)       // zmm3 = x[i+24] - x[i+31]
    vmovupd(zmm3, mem(rdi, 3*64))       // y[i+24] - y[i+31] = zmm3

    vmovupd(mem(rsi, 4*64), zmm4)       // zmm4 = x[i+32] - x[i+39]
    vmovupd(zmm4, mem(rdi, 4*64))       // y[i+32] - y[i+39] = zmm4
    vmovupd(mem(rsi, 5*64), zmm5)       // zmm5 = x[i+40] - x[i+47]
    vmovupd(zmm5, mem(rdi, 5*64))       // y[i+40] - y[i+47] = zmm5
    vmovupd(mem(rsi, 6*64), zmm6)       // zmm6 = x[i+48] - x[i+55]
    vmovupd(zmm6, mem(rdi, 6*64))       // y[i+48] - y[i+55] = zmm6
    vmovupd(mem(rsi, 7*64), zmm7)       // zmm7 = x[i+56] - x[i+63]
    vmovupd(zmm7, mem(rdi, 7*64))       // y[i+56] - y[i+63] = zmm7

    vmovupd(mem(rsi, 8*64), zmm8)       // zmm8 = x[i+64] - x[i+71]
    vmovupd(zmm8, mem(rdi, 8*64))       // y[i+64] - y[i+71] = zmm8
    vmovupd(mem(rsi, 9*64), zmm9)       // zmm9 = x[i+72] - x[i+79]
    vmovupd(zmm9, mem(rdi, 9*64))       // y[i+72] - y[i+79] = zmm9
    vmovupd(mem(rsi, 10*64), zmm10)     // zmm10 = x[i+80] - x[i+87]
    vmovupd(zmm10, mem(rdi, 10*64))     // y[i+80] - y[i+87] = zmm10
    vmovupd(mem(rsi, 11*64), zmm11)     // zmm11 = x[i+88] - x[i+95]
    vmovupd(zmm11, mem(rdi, 11*64))     // y[i+88] - y[i+95] = zmm11

    vmovupd(mem(rsi, 12*64), zmm12)     // zmm12 = x[i+96] - x[i+103]
    vmovupd(zmm12, mem(rdi, 12*64))     // y[i+96] - y[i+103] = zmm12
    vmovupd(mem(rsi, 13*64), zmm13)     // zmm13 = x[i+104] - x[i+111]
    vmovupd(zmm13, mem(rdi, 13*64))     // y[i+104] - y[i+111] = zmm13
    vmovupd(mem(rsi, 14*64), zmm14)     // zmm14 = x[i+112] - x[i+119]
    vmovupd(zmm14, mem(rdi, 14*64))     // y[i+112] - y[i+119] = zmm14
    vmovupd(mem(rsi, 15*64), zmm15)     // zmm15 = x[i+120] - x[i+127]
    vmovupd(zmm15, mem(rdi, 15*64))     // y[i+120] - y[i+127] = zmm15

    // Increment the pointer
    add(imm(8*8*16), rsi)               // Increment the x0 pointer by 1024 -> ( Size of double datatype ) * ( Number of elements per register ) * ( Number of zmm registers used in the section of code )
    add(imm(8*8*16), rdi)               // Increment the y0 pointer by 1024
    sub(imm(8*16),   rcx)               // reduce the number of remaining elements by 128 ->  ( Number of elements per register ) * ( Number of zmm registers used in the section of code )

    // Jump back to the Main loop if the number of remaning elements are still greater than 128
    cmp(imm(8*16), rcx)
    jge(.MAINLOOP)

    // -----------------------------------------------------------

    // Section of code to move the data as blocks of 64 elements
    label(.BLOCK64)

    cmp(imm(8*8), rcx)                  // Check if the number of remaining elements greater than or equal to 64
    jl(.BLOCK32)                        // Else, skip the BLOCK64 section and goto to BLOCK32 section of the code

    // Interleaved SIMD load and store operations to copy data from source to the destination

    vmovupd(mem(rsi, 0*64), zmm0)       // zmm0 = x[i+0] - x[i+7]
    vmovupd(zmm0, mem(rdi, 0*64))       // y[i+0] - y[i+7] = zmm0
    vmovupd(mem(rsi, 1*64), zmm1)       // zmm1 = x[i+8] - x[i+15]
    vmovupd(zmm1, mem(rdi, 1*64))       // y[i+8] - y[i+15] = zmm1
    vmovupd(mem(rsi, 2*64), zmm2)       // zmm2 = x[i+16] - x[i+23]
    vmovupd(zmm2, mem(rdi, 2*64))       // y[i+16] - y[i+23] = zmm2
    vmovupd(mem(rsi, 3*64), zmm3)       // zmm3 = x[i+24] - x[i+31]
    vmovupd(zmm3, mem(rdi, 3*64))       // y[i+24] - y[i+31] = zmm3

    vmovupd(mem(rsi, 4*64), zmm4)       // zmm4 = x[i+32] - x[i+39]
    vmovupd(zmm4, mem(rdi, 4*64))       // y[i+32] - y[i+39] = zmm4
    vmovupd(mem(rsi, 5*64), zmm5)       // zmm5 = x[i+40] - x[i+47]
    vmovupd(zmm5, mem(rdi, 5*64))       // y[i+40] - y[i+47] = zmm5
    vmovupd(mem(rsi, 6*64), zmm6)       // zmm6 = x[i+48] - x[i+55]
    vmovupd(zmm6, mem(rdi, 6*64))       // y[i+48] - y[i+55] = zmm6
    vmovupd(mem(rsi, 7*64), zmm7)       // zmm7 = x[i+56] - x[i+63]
    vmovupd(zmm7, mem(rdi, 7*64))       // y[i+56] - y[i+63] = zmm7

    // Increment the pointer
    add(imm(8*8*8), rsi)                // Increment the x0 pointer by 512
    add(imm(8*8*8), rdi)                // Increment the y0 pointer by 512
    sub(imm(8*8),   rcx)                // reduce the number of remaining elements by 64

    // -----------------------------------------------------------

    // Section of code to move the data as blocks of 32 elements
    label(.BLOCK32)

    cmp(imm(8*4), rcx)                  // check if the number of remaining elements greater than or equal to 32
    jl(.BLOCK16)                        // Else, skip the BLOCK32 section and goto to BLOCK16 section of the code

    // Interleaved SIMD load and store operations to copy data from source to the destination

    vmovupd(mem(rsi, 0*64), zmm0)       // zmm0 = x[i+0] - x[i+7]
    vmovupd(zmm0, mem(rdi, 0*64))       // y[i+0] - y[i+7] = zmm0
    vmovupd(mem(rsi, 1*64), zmm1)       // zmm1 = x[i+8] - x[i+15]
    vmovupd(zmm1, mem(rdi, 1*64))       // y[i+8] - y[i+15] = zmm1
    vmovupd(mem(rsi, 2*64), zmm2)       // zmm2 = x[i+16] - x[i+23]
    vmovupd(zmm2, mem(rdi, 2*64))       // y[i+16] - y[i+23] = zmm2
    vmovupd(mem(rsi, 3*64), zmm3)       // zmm3 = x[i+24] - x[i+31]
    vmovupd(zmm3, mem(rdi, 3*64))       // y[i+24] - y[i+31] = zmm3

    // Increment the pointer
    add(imm(8*8*4), rsi)                // Increment the x0 pointer by 256
    add(imm(8*8*4), rdi)                // Increment the y0 pointer by 256
    sub(imm(8*4),   rcx)                // reduce the number of remaining elements by 32

    // -----------------------------------------------------------

    // Section of code to move the data as blocks of 16 elements
    label(.BLOCK16)

    cmp(imm(8*2), rcx)                  // check if the number of remaining elements greater than or equal to 16
    jl(.BLOCK8)                         // else, skip the BLOCK16 section and goto to BLOCK8 section of the code

    // Interleaved SIMD load and store operations to copy data from source to the destination

    vmovupd(mem(rsi, 0*64), zmm0)       // zmm0 = x[i+0] - x[i+7]
    vmovupd(zmm0, mem(rdi, 0*64))       // y[i+0] - y[i+7] = zmm0
    vmovupd(mem(rsi, 1*64), zmm1)       // zmm1 = x[i+8] - x[i+15]
    vmovupd(zmm1, mem(rdi, 1*64))       // y[i+8] - y[i+15] = zmm1

    // Increment the pointer
    add(imm(8*8*2), rsi)                // Increment the x0 pointer by 128
    add(imm(8*8*2), rdi)                // Increment the y0 pointer by 128
    sub(imm(8*2),   rcx)                // reduce the number of remaining elements by 16

    // -----------------------------------------------------------

    // Section of code to move the data as blocks of 8 elements
    label(.BLOCK8)

    cmp(imm(8), rcx)                    // check if the number of remaining elements greater than or equal to 8
    jl(.FRINGE)                         // else, skip the BLOCK8 section and goto to FRINGE section of the code

    // Load and store operations to copy data from source to the destination

    vmovupd(mem(rsi, 0*64), zmm0)       // zmm0 = x[i+0] - x[i+7]
    vmovupd(zmm0, mem(rdi, 0*64))       // y[i+0] - y[i+7] = zmm0

    // Increment the pointer
    add(imm(8*8), rsi)                  // Increment the x0 pointer by 64
    add(imm(8*8), rdi)                  // Increment the y0 pointer by 64
    sub(imm(8),   rcx)                  // reduce the number of remaining elements by 8

    // -----------------------------------------------------------

    // Section of code to deal with fringe cases
    label(.FRINGE)

    cmp(imm(0), rcx)                    // Check if there are any fringe cases
    je(.END)                            // Else, skip rest of the code

    // Creating a 8-bit mask
    mov(imm(255), r8)                   // (255)10 -> (1111 1111)2
    shlx(rcx, r8, r8)                   // shifting the bits in the register to the left depending on the number of fringe elements remaining
    xor(imm(255), r8)                   // taking compliment of the register
    
    // Copying the 8-bit mask in the register to mask register
    kmovq(r8, k(2))                     

    /*
        Creating mask: Example - fringe case = 2
            step 1 : r8 = (1111 1111)2  or  (255)10
            step 2 : r8 = (1111 1100)2  or  (252)10
            step 3 : r8 = (0000 0011)2  or  (3)10
    */

    // Loading the input values using masked load
    vmovupd(mem(rsi), zmm0 MASK_(K(2)))

    // Storing the values to destination using masked store
    vmovupd(zmm0, mem(rdi) MASK_(K(2)))
    
    // Multiple the value of remaining elements by 8
    mov(imm(3), r11)                    // Load the value 3 to r11 register
    shlx(r11, rcx, r11)                 // Left-Shift the value in rcx by 8

    // Increment the pointer
    add(r11, rsi)                       // Increment the x0 pointer by (Number of remaining elements * 8)
    add(r11, rdi)                       // Increment the y0 pointer by (Number of remaining elements * 8)
    xor(rcx, rcx)                       // Set the value of remaining elements to 0

    // After the above instructions are executed, the remaining part are skipped
    jmp(.END)

    // ========================================================================================================================

    // Code section used to deal with situations where incx or incy is not 1
    label(.SCALAR)

    // incx and incy are multipled by 8 (shift left by 3 bits) and stored back into their respective registers
    mov(imm(3), r11)
    shlx(r11, r8, r8)
    shlx(r11, r9, r9)

    // A loop is used to move one element at a time to the destination
    label(.SCALARLOOP)

    // Checking if all the elements are moved, then the loop will be terminated
    cmp(imm(0), rcx)
    je(.END)

    // Using vector register to mov one element at a time
    vmovsd(mem(rsi, 0), xmm0)
    vmovsd(xmm0, mem(rdi, 0))

    // Moving the address pointer of x and y array by incx*8 and incy*8 bytes
    add(r8, rsi)
    add(r9, rdi)

    // Decrease the count for number of remaining elements
    dec(rcx)

    // Jump back to SCALARLOOP
    jmp(.SCALARLOOP)

    label(.END)
    end_asm
    (
        : // output operands
        : // input operands
          [n0]     "m"     (n0),
          [x0]     "m"     (x0),
          [incx0]  "m"     (incx0),
          [y0]     "m"     (y0),
          [incy0]  "m"     (incy0)
        : // register clobber list
          "zmm0",  "zmm1",  "zmm2",  "zmm3",
          "zmm4",  "zmm5",  "zmm6",  "zmm7",
          "zmm8",  "zmm9",  "zmm10", "zmm11",
          "zmm12", "zmm13", "zmm14", "zmm15",
          "rsi",   "rdi",   "rcx",   "r8",
          "r9",    "r11",   "k2",    "xmm0",
          "memory"
    )

    AOCL_DTL_TRACE_EXIT(AOCL_DTL_LEVEL_TRACE_2)
}

// -----------------------------------------------------------------------------

/*
    Functionality
    -------------

    This function copies a double complex vector x to a double complex vector y.

    y := conj?(x)

    Function Signature
    -------------------

    * 'conjx' - Variable specified if x needs to be conjugated
    * 'n' - Length of the array passed
    * 'x' - Double pointer pointing to an array
    * 'y' - Double pointer pointing to an array
    * 'incx' - Stride to point to the next element in x array
    * 'incy' - Stride to point to the next element in y array
    * 'cntx' - BLIS context object

    Exception
    ----------

    None

    Deviation from BLAS
    --------------------

    None

    Undefined behaviour
    -------------------

    1. The kernel results in undefined behaviour when n < 0, incx < 1 and incy < 1.
       The expectation is that these are standard BLAS exceptions and should be handled in
       a higher layer
*/

void bli_zcopyv_zen4_asm_avx512
(
    conj_t           conjx,
    dim_t            n,
    dcomplex*  restrict x, inc_t incx,
    dcomplex*  restrict y, inc_t incy,
    cntx_t* restrict cntx
)
{
    AOCL_DTL_TRACE_ENTRY(AOCL_DTL_LEVEL_TRACE_2)

    // Initialize local pointers.
    dcomplex *x0 = x;
    dcomplex *y0 = y;

    // Typecast int to 64 bit
    uint64_t n0 = (uint64_t)n;

    // If the vector dimension is zero return early.
    if (bli_zero_dim1(n))
    {
        AOCL_DTL_TRACE_EXIT(AOCL_DTL_LEVEL_TRACE_2)
        return;
    }

    if (bli_is_conj(conjx))
    {
        if (incx == 1 && incy == 1)
        {
            // assembly code
            begin_asm()

            /*
                rdi - > conjx
                rsi - > n
                rdx - > x
                rcx - > incx
                r8  - > y
                r9  - > incy
            */

            // Loading the source memory address to respective registers
            mov(var(x0), rdx)
            mov(var(y0),  r8)

            // Loading the value of 'n' into rsi register
            mov(var(n0), rsi)

            // Setting the value of zmm16 to zero
            vxorpd(zmm16, zmm16, zmm16)

            // ===========================================================

            // Section of code to move the data as blocks of 64 elements
            label(.BLOCK64)

            cmp(imm(4*16), rsi)        // check if the number of remaining elements greater than or equal to 64
            jl(.BLOCK32)               // else, goto to the section of code for block of size 32

            label(.MAINLOOP)
            // Interleaved SIMD load, conjugate and store operations to copy data from source to the destination

            vmovupd(mem(rdx, 0*64), zmm0)       // zmm0 = x[i+0] - x[i+3]
            vfmsubadd231pd(zmm16, zmm16, zmm0)  // zmm0 = conj(zmm0)
            vmovupd(zmm0,  mem(r8, 0*64))       // y[i+0] - y[i+3] = zmm0
            vmovupd(mem(rdx, 1*64), zmm1)       // zmm1 = x[i+4] - x[i+7]
            vfmsubadd231pd(zmm16, zmm16, zmm1)  // zmm1 = conj(zmm1)
            vmovupd(zmm1,  mem(r8, 1*64))       // y[i+4] - y[i+7] = zmm1
            vmovupd(mem(rdx, 2*64), zmm2)       // zmm2 = x[i+8] - x[i+11]
            vfmsubadd231pd(zmm16, zmm16, zmm2)  // zmm2 = conj(zmm2)
            vmovupd(zmm2,  mem(r8, 2*64))       // y[i+8] - y[i+11] = zmm2
            vmovupd(mem(rdx, 3*64), zmm3)       // zmm3 = x[i+12] - x[i+15]
            vfmsubadd231pd(zmm16, zmm16, zmm3)  // zmm3 = conj(zmm3)
            vmovupd(zmm3,  mem(r8, 3*64))       // y[i+12] - y[i+15] = zmm3

            vmovupd(mem(rdx, 4*64), zmm4)       // zmm4 = x[i+16] - x[i+19]
            vfmsubadd231pd(zmm16, zmm16, zmm4)  // zmm4 = conj(zmm4)
            vmovupd(zmm4,  mem(r8, 4*64))       // y[i+16] - y[i+19] = zmm4
            vmovupd(mem(rdx, 5*64), zmm5)       // zmm5 = x[i+20] - x[i+23]
            vfmsubadd231pd(zmm16, zmm16, zmm5)  // zmm5 = conj(zmm5)
            vmovupd(zmm5,  mem(r8, 5*64))       // y[i+20] - y[i+23] = zmm5
            vmovupd(mem(rdx, 6*64), zmm6)       // zmm6 = x[i+24] - x[i+27]
            vfmsubadd231pd(zmm16, zmm16, zmm6)  // zmm6 = conj(zmm6)
            vmovupd(zmm6,  mem(r8, 6*64))       // y[i+24] - y[i+27] = zmm6
            vmovupd(mem(rdx, 7*64), zmm7)       // zmm7 = x[i+28] - x[i+31]
            vfmsubadd231pd(zmm16, zmm16, zmm7)  // zmm7 = conj(zmm7)
            vmovupd(zmm7,  mem(r8, 7*64))       // y[i+28] - y[i+31] = zmm7

            vmovupd(mem(rdx, 8*64), zmm8)       // zmm8 = x[i+32] - x[i+35]
            vfmsubadd231pd(zmm16, zmm16, zmm8)  // zmm8 = conj(zmm8)
            vmovupd(zmm8,  mem(r8, 8*64))       // y[i+32] - y[i+35] = zmm8
            vmovupd(mem(rdx, 9*64), zmm9)       // zmm9 = x[i+36] - x[i+39]
            vfmsubadd231pd(zmm16, zmm16, zmm9)  // zmm9 = conj(zmm9)
            vmovupd(zmm9,  mem(r8, 9*64))       // y[i+36] - y[i+39] = zmm9
            vmovupd(mem(rdx, 10*64), zmm10)     // zmm10 = x[i+40] - x[i+43]
            vfmsubadd231pd(zmm16, zmm16, zmm10) // zmm10 = conj(zmm10)
            vmovupd(zmm10,  mem(r8, 10*64))     // y[i+40] - y[i+43] = zmm10
            vmovupd(mem(rdx, 11*64), zmm11)     // zmm11 = x[i+44] - x[i+47]
            vfmsubadd231pd(zmm16, zmm16, zmm11) // zmm11 = conj(zmm11)
            vmovupd(zmm11,  mem(r8, 11*64))     // y[i+44] - y[i+47] = zmm11

            vmovupd(mem(rdx, 12*64), zmm12)     // zmm12 = x[i+48] - x[i+51]
            vfmsubadd231pd(zmm16, zmm16, zmm12) // zmm12 = conj(zmm12)
            vmovupd(zmm12,  mem(r8, 12*64))     // y[i+48] - y[i+51] = zmm12
            vmovupd(mem(rdx, 13*64), zmm13)     // zmm13 = x[i+52] - x[i+55]
            vfmsubadd231pd(zmm16, zmm16, zmm13) // zmm13 = conj(zmm13)
            vmovupd(zmm13,  mem(r8, 13*64))     // y[i+52] - y[i+55] = zmm13
            vmovupd(mem(rdx, 14*64), zmm14)     // zmm14 = x[i+56] - x[i+59]
            vfmsubadd231pd(zmm16, zmm16, zmm14) // zmm14 = conj(zmm14)
            vmovupd(zmm14,  mem(r8, 14*64))     // y[i+56] - y[i+59] = zmm14
            vmovupd(mem(rdx, 15*64), zmm15)     // zmm15 = x[i+60] - x[i+63]
            vfmsubadd231pd(zmm16, zmm16, zmm15) // zmm15 = conj(zmm15)
            vmovupd(zmm15,  mem(r8, 15*64))     // y[i+60] - y[i+63] = zmm15

            // Increment the pointer
            add(imm(16*4*16), rdx)     // ( Size of double datatype ) * ( Number of elements per register ) * ( Number of zmm registers used in the section of code )
            add(imm(16*4*16),  r8)
            sub(imm(4*16),    rsi)     // reduce the number of remaining elements by 64  ->  ( Number of elements per register ) * ( Number of zmm registers used in the section of code )

            cmp(imm(4*16), rsi)
            jge(.MAINLOOP)

            // -----------------------------------------------------------

            // Section of code to move the data as blocks of 32 elements
            label(.BLOCK32)

            cmp(imm(4*8), rsi)         // check if the number of remaining elements greater than or equal to 32
            jl(.BLOCK16)               // else, goto to the section of code for block of size 16

            // Interleaved SIMD load, conjugate and store operations to copy data from source to the destination

            vmovupd(mem(rdx, 0*64), zmm0)       // zmm0 = x[i+0] - x[i+3]
            vfmsubadd231pd(zmm16, zmm16, zmm0)  // zmm0 = conj(zmm0)
            vmovupd(zmm0,  mem(r8, 0*64))       // y[i+0] - y[i+3] = zmm0
            vmovupd(mem(rdx, 1*64), zmm1)       // zmm1 = x[i+4] - x[i+7]
            vfmsubadd231pd(zmm16, zmm16, zmm1)  // zmm1 = conj(zmm1)
            vmovupd(zmm1,  mem(r8, 1*64))       // y[i+4] - y[i+7] = zmm1
            vmovupd(mem(rdx, 2*64), zmm2)       // zmm2 = x[i+8] - x[i+11]
            vfmsubadd231pd(zmm16, zmm16, zmm2)  // zmm2 = conj(zmm2)
            vmovupd(zmm2,  mem(r8, 2*64))       // y[i+8] - y[i+11] = zmm2
            vmovupd(mem(rdx, 3*64), zmm3)       // zmm3 = x[i+12] - x[i+15]
            vfmsubadd231pd(zmm16, zmm16, zmm3)  // zmm3 = conj(zmm3)
            vmovupd(zmm3,  mem(r8, 3*64))       // y[i+12] - y[i+15] = zmm3

            vmovupd(mem(rdx, 4*64), zmm4)       // zmm4 = x[i+16] - x[i+19]
            vfmsubadd231pd(zmm16, zmm16, zmm4)  // zmm4 = conj(zmm4)
            vmovupd(zmm4,  mem(r8, 4*64))       // y[i+16] - y[i+19] = zmm4
            vmovupd(mem(rdx, 5*64), zmm5)       // zmm5 = x[i+20] - x[i+23]
            vfmsubadd231pd(zmm16, zmm16, zmm5)  // zmm5 = conj(zmm5)
            vmovupd(zmm5,  mem(r8, 5*64))       // y[i+20] - y[i+23] = zmm5
            vmovupd(mem(rdx, 6*64), zmm6)       // zmm6 = x[i+24] - x[i+27]
            vfmsubadd231pd(zmm16, zmm16, zmm6)  // zmm6 = conj(zmm6)
            vmovupd(zmm6,  mem(r8, 6*64))       // y[i+24] - y[i+27] = zmm6
            vmovupd(mem(rdx, 7*64), zmm7)       // zmm7 = x[i+28] - x[i+31]
            vfmsubadd231pd(zmm16, zmm16, zmm7)  // zmm7 = conj(zmm7)
            vmovupd(zmm7,  mem(r8, 7*64))       // y[i+28] - y[i+31] = zmm7

            // Increment the pointer
            add(imm(16*4*8), rdx)
            add(imm(16*4*8),  r8)
            sub(imm(4*8),    rsi)      // reduce the number of remaining elements by 32

            // -----------------------------------------------------------

            // Section of code to move the data as blocks of 16 elements
            label(.BLOCK16)

            cmp(imm(4*4), rsi)         // check if the number of remaining elements greater than or equal to 16
            jl(.BLOCK8)                // else, goto to the section of code for block of size 8

            // Interleaved SIMD load, conjugate and store operations to copy data from source to the destination

            vmovupd(mem(rdx, 0*64), zmm0)       // zmm0 = x[i+0] - x[i+3]
            vfmsubadd231pd(zmm16, zmm16, zmm0)  // zmm0 = conj(zmm0)
            vmovupd(zmm0,  mem(r8, 0*64))       // y[i+0] - y[i+3] = zmm0
            vmovupd(mem(rdx, 1*64), zmm1)       // zmm1 = x[i+4] - x[i+7]
            vfmsubadd231pd(zmm16, zmm16, zmm1)  // zmm1 = conj(zmm1)
            vmovupd(zmm1,  mem(r8, 1*64))       // y[i+4] - y[i+7] = zmm1
            vmovupd(mem(rdx, 2*64), zmm2)       // zmm2 = x[i+8] - x[i+11]
            vfmsubadd231pd(zmm16, zmm16, zmm2)  // zmm2 = conj(zmm2)
            vmovupd(zmm2,  mem(r8, 2*64))       // y[i+8] - y[i+11] = zmm2
            vmovupd(mem(rdx, 3*64), zmm3)       // zmm3 = x[i+12] - x[i+15]
            vfmsubadd231pd(zmm16, zmm16, zmm3)  // zmm3 = conj(zmm3)
            vmovupd(zmm3,  mem(r8, 3*64))       // y[i+12] - y[i+15] = zmm3

            // Increment the pointer
            add(imm(16*4*4), rdx)
            add(imm(16*4*4),  r8)
            sub(imm(4*4),    rsi)      // reduce the number of remaining elements by 16

            // -----------------------------------------------------------

            // Section of code to move the data as blocks of 8 elements
            label(.BLOCK8)

            cmp(imm(4*2), rsi)         // check if the number of remaining elements greater than or equal to 8
            jl(.BLOCK4)                // else, goto to the section of code for block of size 4

            // Interleaved SIMD load, conjugate and store operations to copy data from source to the destination

            vmovupd(mem(rdx, 0*64), zmm0)       // zmm0 = x[i+0] - x[i+3]
            vfmsubadd231pd(zmm16, zmm16, zmm0)  // zmm0 = conj(zmm0)
            vmovupd(zmm0,  mem(r8, 0*64))       // y[i+0] - y[i+3] = zmm0
            vmovupd(mem(rdx, 1*64), zmm1)       // zmm1 = x[i+4] - x[i+7]
            vfmsubadd231pd(zmm16, zmm16, zmm1)  // zmm1 = conj(zmm1)
            vmovupd(zmm1,  mem(r8, 1*64))       // y[i+4] - y[i+7] = zmm1

            // Increment the pointer
            add(imm(16*4*2), rdx)
            add(imm(16*4*2),  r8)
            sub(imm(4*2),    rsi)      // reduce the number of remaining elements by 8

            // -----------------------------------------------------------

            // Section of code to move the data as blocks of 4 elements
            label(.BLOCK4)

            cmp(imm(4), rsi)           // check if the number of remaining elements greater than or equal to 4
            jl(.FRINGE)                // else, goto to the section of code that deals with fringe cases

            // Load, conjugate and store operations to copy data from source to the destination

            vmovupd(mem(rdx, 0*64), zmm0)       // zmm0 = x[i+0] - x[i+3]
            vfmsubadd231pd(zmm16, zmm16, zmm0)  // zmm0 = conj(zmm0)
            vmovupd(zmm0,  mem(r8, 0*64))       // y[i+0] - y[i+3] = zmm0

            // Increment the pointer
            add(imm(16*4), rdx)
            add(imm(16*4),  r8)
            sub(imm(4),    rsi)        // reduce the number of remaining elements by 4

            // -----------------------------------------------------------

            // Section of code to deal with fringe cases
            label(.FRINGE)

            cmp(imm(0), rsi)           // check if there is any fringe cases
            je(.END)

            // Creating a 8-bit mask
            mov(imm(255), rcx)         // (255)10 -> (1111 1111)2
            shlx(rsi, rcx, rcx)        // shifting the bits in the register to the left depending on the number of fringe elements remaining
            shlx(rsi, rcx, rcx)
            xor(imm(255),rcx)          // taking compliment of the register
            kmovq(rcx, k(2))           // copying the value in the register to mask register

            /*
                Creating mask: Example - fringe case = 1
                    step 1 : rcx = (1111 1111)2  or  (255)10
                    step 2 : rcx = (1111 1110)2  or  (254)10
                    step 3 : rcx = (1111 1100)2  or  (252)10
                    step 4 : rcx = (0000 0011)2  or  (3)10
            */
            // Loading the input values using masked load
            vmovupd(mem(rdx, 0*64), zmm0 MASK_(K(2)))

            // Using Fused Multiply-AlternatingAdd/Subtract operation to get conjugate of the input
            vfmsubadd231pd(zmm16, zmm16, zmm0)

            // Storing the values to destination using masked store
            vmovupd(zmm0,  mem(r8) MASK_(K(2)))

            // Increment the pointer
            add(rsi,    rdx)
            add(rsi,     r8)
            and(imm(0), rsi)

            label(.END)
            end_asm
            (
                : // output operands
                : // input operands
                  [n0]     "m"     (n0),
                  [x0]     "m"     (x0),
                  [y0]     "m"     (y0)
                : // register clobber list
                  "zmm0",  "zmm1",  "zmm2",  "zmm3",
                  "zmm4",  "zmm5",  "zmm6",  "zmm7",
                  "zmm8",  "zmm9",  "zmm10", "zmm11",
                  "zmm12", "zmm13", "zmm14", "zmm15",
                  "zmm16", "rsi",   "rdx",   "rcx",
                  "r8",    "r9",    "k2",    "memory"
            )
        }
        else
        {
            // Since double complex elements are of size 128 bits,
            // vectorization can be done using XMM registers when incx and incy are not 1.
            // This is done in the else condition.
            dim_t i = 0;
            __m128d  xv[16];
            __m128d zero_reg = _mm_setzero_pd();

            // n & (~0x0F) = n & 0xFFFFFFF0 -> this masks the numbers less than 16,
            // if value of n < 16, then (n & (~0x0F)) = 0
            // the copy operation will be done for the multiples of 16
            for ( i = 0; i < (n & (~0x0F)); i += 16)
            {
                // Loading the input values
                xv[0] = _mm_loadu_pd((double *)(x0 + 0 * incx));
                xv[1] = _mm_loadu_pd((double *)(x0 + 1 * incx));
                xv[2] = _mm_loadu_pd((double *)(x0 + 2 * incx));
                xv[3] = _mm_loadu_pd((double *)(x0 + 3 * incx));

                xv[4] = _mm_loadu_pd((double *)(x0 + 4 * incx));
                xv[5] = _mm_loadu_pd((double *)(x0 + 5 * incx));
                xv[6] = _mm_loadu_pd((double *)(x0 + 6 * incx));
                xv[7] = _mm_loadu_pd((double *)(x0 + 7 * incx));

                xv[8] = _mm_loadu_pd((double *)(x0 + 8 * incx));
                xv[9] = _mm_loadu_pd((double *)(x0 + 9 * incx));
                xv[10] = _mm_loadu_pd((double *)(x0 + 10 * incx));
                xv[11] = _mm_loadu_pd((double *)(x0 + 11 * incx));

                xv[12] = _mm_loadu_pd((double *)(x0 + 12 * incx));
                xv[13] = _mm_loadu_pd((double *)(x0 + 13 * incx));
                xv[14] = _mm_loadu_pd((double *)(x0 + 14 * incx));
                xv[15] = _mm_loadu_pd((double *)(x0 + 15 * incx));

                // Perform conjugation by multiplying the imaginary part with -1 and real part with 1
                xv[0] = _mm_fmsubadd_pd(zero_reg, zero_reg, xv[0]);
                xv[1] = _mm_fmsubadd_pd(zero_reg, zero_reg, xv[1]);
                xv[2] = _mm_fmsubadd_pd(zero_reg, zero_reg, xv[2]);
                xv[3] = _mm_fmsubadd_pd(zero_reg, zero_reg, xv[3]);

                xv[4] = _mm_fmsubadd_pd(zero_reg, zero_reg, xv[4]);
                xv[5] = _mm_fmsubadd_pd(zero_reg, zero_reg, xv[5]);
                xv[6] = _mm_fmsubadd_pd(zero_reg, zero_reg, xv[6]);
                xv[7] = _mm_fmsubadd_pd(zero_reg, zero_reg, xv[7]);

                xv[8] = _mm_fmsubadd_pd(zero_reg, zero_reg, xv[8]);
                xv[9] = _mm_fmsubadd_pd(zero_reg, zero_reg, xv[9]);
                xv[10] = _mm_fmsubadd_pd(zero_reg, zero_reg, xv[10]);
                xv[11] = _mm_fmsubadd_pd(zero_reg, zero_reg, xv[11]);

                xv[12] = _mm_fmsubadd_pd(zero_reg, zero_reg, xv[12]);
                xv[13] = _mm_fmsubadd_pd(zero_reg, zero_reg, xv[13]);
                xv[14] = _mm_fmsubadd_pd(zero_reg, zero_reg, xv[14]);
                xv[15] = _mm_fmsubadd_pd(zero_reg, zero_reg, xv[15]);

                // Storing the values to destination
                _mm_storeu_pd((double *)(y0 + incy * 0), xv[0]);
                _mm_storeu_pd((double *)(y0 + incy * 1), xv[1]);
                _mm_storeu_pd((double *)(y0 + incy * 2), xv[2]);
                _mm_storeu_pd((double *)(y0 + incy * 3), xv[3]);

                _mm_storeu_pd((double *)(y0 + incy * 4), xv[4]);
                _mm_storeu_pd((double *)(y0 + incy * 5), xv[5]);
                _mm_storeu_pd((double *)(y0 + incy * 6), xv[6]);
                _mm_storeu_pd((double *)(y0 + incy * 7), xv[7]);

                _mm_storeu_pd((double *)(y0 + incy * 8), xv[8]);
                _mm_storeu_pd((double *)(y0 + incy * 9 ), xv[9]);
                _mm_storeu_pd((double *)(y0 + incy * 10), xv[10]);
                _mm_storeu_pd((double *)(y0 + incy * 11), xv[11]);

                _mm_storeu_pd((double *)(y0 + incy * 12), xv[12]);
                _mm_storeu_pd((double *)(y0 + incy * 13), xv[13]);
                _mm_storeu_pd((double *)(y0 + incy * 14), xv[14]);
                _mm_storeu_pd((double *)(y0 + incy * 15), xv[15]);

                // Increment the pointer
                x0 += 16 * incx;
                y0 += 16 * incy;
            }

            for ( ; i < (n & (~0x07)); i += 8)
            {
                // Loading the input values
                xv[0] = _mm_loadu_pd((double *)(x0 + 0 * incx));
                xv[1] = _mm_loadu_pd((double *)(x0 + 1 * incx));
                xv[2] = _mm_loadu_pd((double *)(x0 + 2 * incx));
                xv[3] = _mm_loadu_pd((double *)(x0 + 3 * incx));

                xv[4] = _mm_loadu_pd((double *)(x0 + 4 * incx));
                xv[5] = _mm_loadu_pd((double *)(x0 + 5 * incx));
                xv[6] = _mm_loadu_pd((double *)(x0 + 6 * incx));
                xv[7] = _mm_loadu_pd((double *)(x0 + 7 * incx));

                // Perform conjugation by multiplying the imaginary part with -1 and real part with 1
                xv[0] = _mm_fmsubadd_pd(zero_reg, zero_reg, xv[0]);
                xv[1] = _mm_fmsubadd_pd(zero_reg, zero_reg, xv[1]);
                xv[2] = _mm_fmsubadd_pd(zero_reg, zero_reg, xv[2]);
                xv[3] = _mm_fmsubadd_pd(zero_reg, zero_reg, xv[3]);

                xv[4] = _mm_fmsubadd_pd(zero_reg, zero_reg, xv[4]);
                xv[5] = _mm_fmsubadd_pd(zero_reg, zero_reg, xv[5]);
                xv[6] = _mm_fmsubadd_pd(zero_reg, zero_reg, xv[6]);
                xv[7] = _mm_fmsubadd_pd(zero_reg, zero_reg, xv[7]);

                // Storing the values to destination
                _mm_storeu_pd((double *)(y0 + incy * 0), xv[0]);
                _mm_storeu_pd((double *)(y0 + incy * 1), xv[1]);
                _mm_storeu_pd((double *)(y0 + incy * 2), xv[2]);
                _mm_storeu_pd((double *)(y0 + incy * 3), xv[3]);

                _mm_storeu_pd((double *)(y0 + incy * 4), xv[4]);
                _mm_storeu_pd((double *)(y0 + incy * 5), xv[5]);
                _mm_storeu_pd((double *)(y0 + incy * 6), xv[6]);
                _mm_storeu_pd((double *)(y0 + incy * 7), xv[7]);

                // Increment the pointer
                x0 += 8 * incx;
                y0 += 8 * incy;
            }

            for ( ; i < (n & (~0x03)); i += 4)
            {
                // Loading the input values
                xv[0] = _mm_loadu_pd((double *)(x0 + 0 * incx));
                xv[1] = _mm_loadu_pd((double *)(x0 + 1 * incx));
                xv[2] = _mm_loadu_pd((double *)(x0 + 2 * incx));
                xv[3] = _mm_loadu_pd((double *)(x0 + 3 * incx));

                // Perform conjugation by multiplying the imaginary part with -1 and real part with 1
                xv[0] = _mm_fmsubadd_pd(zero_reg, zero_reg, xv[0]);
                xv[1] = _mm_fmsubadd_pd(zero_reg, zero_reg, xv[1]);
                xv[2] = _mm_fmsubadd_pd(zero_reg, zero_reg, xv[2]);
                xv[3] = _mm_fmsubadd_pd(zero_reg, zero_reg, xv[3]);

                // Storing the values to destination
                _mm_storeu_pd((double *)(y0 + incy * 0), xv[0]);
                _mm_storeu_pd((double *)(y0 + incy * 1), xv[1]);
                _mm_storeu_pd((double *)(y0 + incy * 2), xv[2]);
                _mm_storeu_pd((double *)(y0 + incy * 3), xv[3]);

                // Increment the pointer
                x0 += 4 * incx;
                y0 += 4 * incy;
            }

            for ( ; i < (n & (~0x01)); i += 2)
            {
                // Loading the input values
                xv[0] = _mm_loadu_pd((double *)(x0 + 0 * incx));
                xv[1] = _mm_loadu_pd((double *)(x0 + 1 * incx));

                // Perform conjugation by multiplying the imaginary part with -1 and real part with 1
                xv[0] = _mm_fmsubadd_pd(zero_reg, zero_reg, xv[0]);
                xv[1] = _mm_fmsubadd_pd(zero_reg, zero_reg, xv[1]);

                // Storing the values to destination
                _mm_storeu_pd((double *)(y0 + incy * 0), xv[0]);
                _mm_storeu_pd((double *)(y0 + incy * 1), xv[1]);

                // Increment the pointer
                x0 += 2 * incx;
                y0 += 2 * incy;
            }

            for ( ; i < n; i += 1)
            {
                // Loading the input values
                xv[0] = _mm_loadu_pd((double *)(x0 + 0 * incx));

                // Perform conjugation by multiplying the imaginary part with -1 and real part with 1
                xv[0] = _mm_fmsubadd_pd(zero_reg, zero_reg, xv[0]);

                // Storing the values to destination
                _mm_storeu_pd((double *)(y0 + incy * 0), xv[0]);

                // Increment the pointer
                x0 += 1 * incx;
                y0 += 1 * incy;
            }
        }
    }
    else
    {
        if (incx == 1 && incy == 1)
        {
            // assembly code
            begin_asm()

            /*
                rdi - > conjx
                rsi - > n
                rdx - > x
                rcx - > incx
                r8  - > y
                r9  - > incy
            */

            // Loading the source memory address to respective registers
            mov(var(x0), rdx)
            mov(var(y0),  r8)

            // Loading the value of 'n' to respective register
            mov(var(n0), rsi)

            // ===========================================================

            // Section of code to move the data as blocks of 128 elements
            label(.BLOCK128)

            cmp(imm(4*32), rsi)        // check if the number of remaining elements greater than or equal to 128 -> (NUMBER OF ELEMENTS PER REGISTER) * (NUMBER OF REGISTERS USED IN THE BLOCK)
            jl(.BLOCK64)               // else, goto block of size 64

            label(.MAINLOOP)
            // Interleaved SIMD load and store operations to copy data from source to the destination
            // Each vector register can hold 4 elements and is used twice before next jump operation 
            // 1 for loading the element from source and 1 for store it into the destination

            vmovupd(mem(rdx, 0*64), zmm0)       // zmm0 = x[i+0] - x[i+3]
            vmovupd(zmm0,  mem(r8, 0*64))       // y[i+0] - y[i+3] = zmm0
            vmovupd(mem(rdx, 1*64), zmm1)       // zmm1 = x[i+4] - x[i+7]
            vmovupd(zmm1,  mem(r8, 1*64))       // y[i+4] - y[i+7] = zmm1
            vmovupd(mem(rdx, 2*64), zmm2)       // zmm2 = x[i+8] - x[i+11]
            vmovupd(zmm2,  mem(r8, 2*64))       // y[i+8] - y[i+11] = zmm2
            vmovupd(mem(rdx, 3*64), zmm3)       // zmm3 = x[i+12] - x[i+15]
            vmovupd(zmm3,  mem(r8, 3*64))       // y[i+12] - y[i+15] = zmm3

            vmovupd(mem(rdx, 4*64), zmm4)       // zmm4 = x[i+16] - x[i+19]
            vmovupd(zmm4,  mem(r8, 4*64))       // y[i+16] - y[i+19] = zmm4
            vmovupd(mem(rdx, 5*64), zmm5)       // zmm5 = x[i+20] - x[i+23]
            vmovupd(zmm5,  mem(r8, 5*64))       // y[i+20] - y[i+23] = zmm5
            vmovupd(mem(rdx, 6*64), zmm6)       // zmm6 = x[i+24] - x[i+27]
            vmovupd(zmm6,  mem(r8, 6*64))       // y[i+24] - y[i+27] = zmm6
            vmovupd(mem(rdx, 7*64), zmm7)       // zmm7 = x[i+28] - x[i+31]
            vmovupd(zmm7,  mem(r8, 7*64))       // y[i+28] - y[i+31] = zmm7

            vmovupd(mem(rdx, 8*64), zmm8)       // zmm8 = x[i+32] - x[i+35]
            vmovupd(zmm8,  mem(r8, 8*64))       // y[i+32] - y[i+35] = zmm8
            vmovupd(mem(rdx, 9*64), zmm9)       // zmm9 = x[i+36] - x[i+39]
            vmovupd(zmm9,  mem(r8, 9*64))       // y[i+36] - y[i+39] = zmm9
            vmovupd(mem(rdx, 10*64), zmm10)     // zmm10 = x[i+40] - x[i+43]
            vmovupd(zmm10,  mem(r8, 10*64))     // y[i+40] - y[i+43] = zmm10
            vmovupd(mem(rdx, 11*64), zmm11)     // zmm11 = x[i+44] - x[i+47]
            vmovupd(zmm11,  mem(r8, 11*64))     // y[i+44] - y[i+47] = zmm11

            vmovupd(mem(rdx, 12*64), zmm12)     // zmm12 = x[i+48] - x[i+51]
            vmovupd(zmm12,  mem(r8, 12*64))     // y[i+48] - y[i+51] = zmm12
            vmovupd(mem(rdx, 13*64), zmm13)     // zmm13 = x[i+52] - x[i+55]
            vmovupd(zmm13,  mem(r8, 13*64))     // y[i+52] - y[i+55] = zmm13
            vmovupd(mem(rdx, 14*64), zmm14)     // zmm14 = x[i+56] - x[i+59]
            vmovupd(zmm14,  mem(r8, 14*64))     // y[i+56] - y[i+59] = zmm14
            vmovupd(mem(rdx, 15*64), zmm15)     // zmm15 = x[i+60] - x[i+63]
            vmovupd(zmm15,  mem(r8, 15*64))     // y[i+60] - y[i+63] = zmm15

            vmovupd(mem(rdx, 16*64), zmm16)     // zmm16 = x[i+64] - x[i+67]
            vmovupd(zmm16,  mem(r8, 16*64))     // y[i+64] - y[i+67] = zmm16
            vmovupd(mem(rdx, 17*64), zmm17)     // zmm17 = x[i+68] - x[i+71]
            vmovupd(zmm17,  mem(r8, 17*64))     // y[i+68] - y[i+71] = zmm17
            vmovupd(mem(rdx, 18*64), zmm18)     // zmm18 = x[i+72] - x[i+75]
            vmovupd(zmm18,  mem(r8, 18*64))     // y[i+72] - y[i+75] = zmm18
            vmovupd(mem(rdx, 19*64), zmm19)     // zmm19 = x[i+76] - x[i+79]
            vmovupd(zmm19,  mem(r8, 19*64))     // y[i+76] - y[i+79] = zmm19

            vmovupd(mem(rdx, 20*64), zmm20)     // zmm20 = x[i+80] - x[i+83]
            vmovupd(zmm20,  mem(r8, 20*64))     // y[i+80] - y[i+83] = zmm20
            vmovupd(mem(rdx, 21*64), zmm21)     // zmm21 = x[i+84] - x[i+87]
            vmovupd(zmm21,  mem(r8, 21*64))     // y[i+84] - y[i+87] = zmm21
            vmovupd(mem(rdx, 22*64), zmm22)     // zmm22 = x[i+88] - x[i+91]
            vmovupd(zmm22,  mem(r8, 22*64))     // y[i+88] - y[i+91] = zmm22
            vmovupd(mem(rdx, 23*64), zmm23)     // zmm23 = x[i+92] - x[i+95]
            vmovupd(zmm23,  mem(r8, 23*64))     // y[i+92] - y[i+95] = zmm23

            vmovupd(mem(rdx, 24*64), zmm24)     // zmm24 = x[i+96] - x[i+99]
            vmovupd(zmm24,  mem(r8, 24*64))     // y[i+96] - y[i+99] = zmm24
            vmovupd(mem(rdx, 25*64), zmm25)     // zmm25 = x[i+100] - x[i+103]
            vmovupd(zmm25,  mem(r8, 25*64))     // y[i+100] - y[i+103] = zmm25
            vmovupd(mem(rdx, 26*64), zmm26)     // zmm26 = x[i+104] - x[i+107]
            vmovupd(zmm26,  mem(r8, 26*64))     // y[i+104] - y[i+107] = zmm26
            vmovupd(mem(rdx, 27*64), zmm27)     // zmm27 = x[i+108] - x[i+111]
            vmovupd(zmm27,  mem(r8, 27*64))     // y[i+108] - y[i+111] = zmm27

            vmovupd(mem(rdx, 28*64), zmm28)     // zmm28 = x[i+112] - x[i+115]
            vmovupd(zmm28,  mem(r8, 28*64))     // y[i+112] - y[i+115] = zmm28
            vmovupd(mem(rdx, 29*64), zmm29)     // zmm29 = x[i+116] - x[i+119]
            vmovupd(zmm29,  mem(r8, 29*64))     // y[i+116] - y[i+119] = zmm29
            vmovupd(mem(rdx, 30*64), zmm30)     // zmm30 = x[i+120] - x[i+123]
            vmovupd(zmm30,  mem(r8, 30*64))     // y[i+120] - y[i+123] = zmm30
            vmovupd(mem(rdx, 31*64), zmm31)     // zmm31 = x[i+124] - x[i+127]
            vmovupd(zmm31,  mem(r8, 31*64))     // y[i+124] - y[i+127] = zmm31

            // Increment the pointer
            add(imm(16*4*32), rdx)     // ( Size of double datatype ) * ( Number of elements per register ) * ( Number of zmm registers used in the section of code )
            add(imm(16*4*32),  r8)

            // reduce the number of remaining elements by 128
            sub(imm(4*32), rsi)        // ( Number of elements per register ) * ( Number of zmm registers used in the section of code )

            cmp(imm(4*32), rsi)
            jge(.MAINLOOP)

            // -----------------------------------------------------------

            // Section of code to move the data as blocks of 64 elements
            label(.BLOCK64)

            cmp(imm(4*16), rsi)        // check if the number of remaining elements greater than or equal to 64
            jl(.BLOCK32)               // else, goto to the section of code for block of size 32

            // Interleaved SIMD load and store operations to copy data from source to the destination

            vmovupd(mem(rdx, 0*64), zmm0)       // zmm0 = x[i+0] - x[i+3]
            vmovupd(zmm0,  mem(r8, 0*64))       // y[i+0] - y[i+3] = zmm0
            vmovupd(mem(rdx, 1*64), zmm1)       // zmm1 = x[i+4] - x[i+7]
            vmovupd(zmm1,  mem(r8, 1*64))       // y[i+4] - y[i+7] = zmm1
            vmovupd(mem(rdx, 2*64), zmm2)       // zmm2 = x[i+8] - x[i+11]
            vmovupd(zmm2,  mem(r8, 2*64))       // y[i+8] - y[i+11] = zmm2
            vmovupd(mem(rdx, 3*64), zmm3)       // zmm3 = x[i+12] - x[i+15]
            vmovupd(zmm3,  mem(r8, 3*64))       // y[i+12] - y[i+15] = zmm3

            vmovupd(mem(rdx, 4*64), zmm4)       // zmm4 = x[i+16] - x[i+19]
            vmovupd(zmm4,  mem(r8, 4*64))       // y[i+16] - y[i+19] = zmm4
            vmovupd(mem(rdx, 5*64), zmm5)       // zmm5 = x[i+20] - x[i+23]
            vmovupd(zmm5,  mem(r8, 5*64))       // y[i+20] - y[i+23] = zmm5
            vmovupd(mem(rdx, 6*64), zmm6)       // zmm6 = x[i+24] - x[i+27]
            vmovupd(zmm6,  mem(r8, 6*64))       // y[i+24] - y[i+27] = zmm6
            vmovupd(mem(rdx, 7*64), zmm7)       // zmm7 = x[i+28] - x[i+31]
            vmovupd(zmm7,  mem(r8, 7*64))       // y[i+28] - y[i+31] = zmm7

            vmovupd(mem(rdx, 8*64), zmm8)       // zmm8 = x[i+32] - x[i+35]
            vmovupd(zmm8,  mem(r8, 8*64))       // y[i+32] - y[i+35] = zmm8
            vmovupd(mem(rdx, 9*64), zmm9)       // zmm9 = x[i+36] - x[i+39]
            vmovupd(zmm9,  mem(r8, 9*64))       // y[i+36] - y[i+39] = zmm9
            vmovupd(mem(rdx, 10*64), zmm10)     // zmm10 = x[i+40] - x[i+43]
            vmovupd(zmm10,  mem(r8, 10*64))     // y[i+40] - y[i+43] = zmm10
            vmovupd(mem(rdx, 11*64), zmm11)     // zmm11 = x[i+44] - x[i+47]
            vmovupd(zmm11,  mem(r8, 11*64))     // y[i+44] - y[i+47] = zmm11

            vmovupd(mem(rdx, 12*64), zmm12)     // zmm12 = x[i+48] - x[i+51]
            vmovupd(zmm12,  mem(r8, 12*64))     // y[i+48] - y[i+51] = zmm12
            vmovupd(mem(rdx, 13*64), zmm13)     // zmm13 = x[i+52] - x[i+55]
            vmovupd(zmm13,  mem(r8, 13*64))     // y[i+52] - y[i+55] = zmm13
            vmovupd(mem(rdx, 14*64), zmm14)     // zmm14 = x[i+56] - x[i+59]
            vmovupd(zmm14,  mem(r8, 14*64))     // y[i+56] - y[i+59] = zmm14
            vmovupd(mem(rdx, 15*64), zmm15)     // zmm15 = x[i+60] - x[i+63]
            vmovupd(zmm15,  mem(r8, 15*64))     // y[i+60] - y[i+63] = zmm15

            // Increment the pointer
            add(imm(16*4*16), rdx)
            add(imm(16*4*16),  r8)

            // reduce the number of remaining elements by 64
            sub(imm(4*16), rsi)

            // -----------------------------------------------------------

            // Section of code to move the data as blocks of 32 elements
            label(.BLOCK32)

            cmp(imm(4*8), rsi)         // check if the number of remaining elements greater than or equal to 32
            jl(.BLOCK16)               // else, goto to the section of code for block of size 16

            // Interleaved SIMD load and store operations to copy data from source to the destination

            vmovupd(mem(rdx, 0*64), zmm0)       // zmm0 = x[i+0] - x[i+3]
            vmovupd(zmm0,  mem(r8, 0*64))       // y[i+0] - y[i+3] = zmm0
            vmovupd(mem(rdx, 1*64), zmm1)       // zmm1 = x[i+4] - x[i+7]
            vmovupd(zmm1,  mem(r8, 1*64))       // y[i+4] - y[i+7] = zmm1
            vmovupd(mem(rdx, 2*64), zmm2)       // zmm2 = x[i+8] - x[i+11]
            vmovupd(zmm2,  mem(r8, 2*64))       // y[i+8] - y[i+11] = zmm2
            vmovupd(mem(rdx, 3*64), zmm3)       // zmm3 = x[i+12] - x[i+15]
            vmovupd(zmm3,  mem(r8, 3*64))       // y[i+12] - y[i+15] = zmm3

            vmovupd(mem(rdx, 4*64), zmm4)       // zmm4 = x[i+16] - x[i+19]
            vmovupd(zmm4,  mem(r8, 4*64))       // y[i+16] - y[i+19] = zmm4
            vmovupd(mem(rdx, 5*64), zmm5)       // zmm5 = x[i+20] - x[i+23]
            vmovupd(zmm5,  mem(r8, 5*64))       // y[i+20] - y[i+23] = zmm5
            vmovupd(mem(rdx, 6*64), zmm6)       // zmm6 = x[i+24] - x[i+27]
            vmovupd(zmm6,  mem(r8, 6*64))       // y[i+24] - y[i+27] = zmm6
            vmovupd(mem(rdx, 7*64), zmm7)       // zmm7 = x[i+28] - x[i+31]
            vmovupd(zmm7,  mem(r8, 7*64))       // y[i+28] - y[i+31] = zmm7

            // Increment the pointer
            add(imm(16*4*8), rdx)
            add(imm(16*4*8),  r8)

            // reduce the number of remaining elements by 32
            sub(imm(4*8), rsi)

            // -----------------------------------------------------------

            // Section of code to move the data as blocks of 16 elements
            label(.BLOCK16)

            cmp(imm(4*4), rsi)         // check if the number of remaining elements greater than or equal to 16
            jl(.BLOCK8)                // else, goto to the section of code for block of size 8

            // Interleaved SIMD load and store operations to copy data from source to the destination

            vmovupd(mem(rdx, 0*64), zmm0)       // zmm0 = x[i+0] - x[i+3]
            vmovupd(zmm0,  mem(r8, 0*64))       // y[i+0] - y[i+3] = zmm0
            vmovupd(mem(rdx, 1*64), zmm1)       // zmm1 = x[i+4] - x[i+7]
            vmovupd(zmm1,  mem(r8, 1*64))       // y[i+4] - y[i+7] = zmm1
            vmovupd(mem(rdx, 2*64), zmm2)       // zmm2 = x[i+8] - x[i+11]
            vmovupd(zmm2,  mem(r8, 2*64))       // y[i+8] - y[i+11] = zmm2
            vmovupd(mem(rdx, 3*64), zmm3)       // zmm3 = x[i+12] - x[i+15]
            vmovupd(zmm3,  mem(r8, 3*64))       // y[i+12] - y[i+15] = zmm3

            // Increment the pointer
            add(imm(16*4*4), rdx)
            add(imm(16*4*4),  r8)

            // reduce the number of remaining elements by 16
            sub(imm(4*4), rsi)

            // -----------------------------------------------------------

            // Section of code to move the data as blocks of 8 elements
            label(.BLOCK8)

            cmp(imm(4*2), rsi)         // check if the number of remaining elements greater than or equal to 8
            jl(.BLOCK4)                // else, goto to the section of code for block of size 4

            // Interleaved SIMD load and store operations to copy data from source to the destination

            vmovupd(mem(rdx, 0*64), zmm0)       // zmm0 = x[i+0] - x[i+3]
            vmovupd(zmm0,  mem(r8, 0*64))       // y[i+0] - y[i+3] = zmm0
            vmovupd(mem(rdx, 1*64), zmm1)       // zmm1 = x[i+4] - x[i+7]
            vmovupd(zmm1,  mem(r8, 1*64))       // y[i+4] - y[i+7] = zmm1

            // Increment the pointer
            add(imm(16*4*2), rdx)
            add(imm(16*4*2),  r8)

            // reduce the number of remaining elements by 8
            sub(imm(4*2), rsi)

            // -----------------------------------------------------------

            // Section of code to move the data as blocks of 4 elements
            label(.BLOCK4)

            cmp(imm(4), rsi)           // check if the number of remaining elements greater than or equal to 4
            jl(.FRINGE)                // else, goto to the section of code that deals with fringe cases

            // Loading and storing the values to destination

            vmovupd(mem(rdx, 0*64), zmm0)       // zmm0 = x[i+0] - x[i+3]
            vmovupd(zmm0,  mem(r8, 0*64))       // y[i+0] - y[i+3] = zmm0

            // Increment the pointer
            add(imm(16*4), rdx)
            add(imm(16*4),  r8)

            // reduce the number of remaining elements by 4
            sub(imm(4), rsi)

            // -----------------------------------------------------------

            // Section of code to deal with fringe cases
            label(.FRINGE)

            cmp(imm(0), rsi)           // check if there is any fringe cases
            je(.END)

            // Creating a 8-bit mask
            mov(imm(255), rcx)         // (255)10 -> (1111 1111)2
            shlx(rsi,rcx,rcx)          // shifting the bits in the register to the left depending on the number of fringe elements remaining
            shlx(rsi,rcx,rcx)
            xor(imm(255),rcx)          // taking compliment of the register
            kmovq(rcx, k(2))           // copying the value in the register to mask register

            /*
                Creating mask: Example - fringe case = 1
                    step 1 : rcx = (1111 1111)2  or  (255)10
                    step 2 : rcx = (1111 1110)2  or  (254)10
                    step 3 : rcx = (1111 1100)2  or  (252)10
                    step 4 : rcx = (0000 0011)2  or  (3)10
            */
            // Loading the input values using masked load
            vmovupd(mem(rdx, 0*64), zmm0 MASK_(K(2)))

            // Storing the values to destination using masked store
            vmovupd(zmm0,  mem(r8) MASK_(K(2)))

            // Increment the pointer
            add(rsi, rdx)
            add(rsi, r8)
            and(imm(0), rsi)

            label(.END)
            end_asm
            (
                : // output operands
                : // input operands
                  [n0]     "m"     (n0),
                  [x0]     "m"     (x0),
                  [y0]     "m"     (y0)
                : // register clobber list
                  "zmm0",  "zmm1",  "zmm2",  "zmm3",
                  "zmm4",  "zmm5",  "zmm6",  "zmm7",
                  "zmm8",  "zmm9",  "zmm10", "zmm11",
                  "zmm12", "zmm13", "zmm14", "zmm15",
                  "zmm16", "zmm17", "zmm18", "zmm19",
                  "zmm20", "zmm21", "zmm22", "zmm23",
                  "zmm24", "zmm25", "zmm26", "zmm27",
                  "zmm28", "zmm29", "zmm30", "zmm31",
                  "rsi",   "rdx",   "rcx",   "r8",
                  "r9",    "k2",    "memory"
            )
        }
        else
        {
            // Since double complex elements are of size 128 bits,
            // vectorization can be done using XMM registers when incx and incy are not 1.
            // This is done in the else condition.
            __m128d  xv[32];
            dim_t i = 0;

            // n & (~0x1F) = n & 0xFFFFFFE0 -> this masks the numbers less than 32,
            // if value of n < 32, then (n & (~0x1F)) = 0
            // the copy operation will be done for the multiples of 32
            for ( i = 0; i < (n & (~0x1F)); i += 32)
            {
                // Loading the input values
                xv[0] = _mm_loadu_pd((double *)(x0 + 0 * incx));
                xv[1] = _mm_loadu_pd((double *)(x0 + 1 * incx));
                xv[2] = _mm_loadu_pd((double *)(x0 + 2 * incx));
                xv[3] = _mm_loadu_pd((double *)(x0 + 3 * incx));

                xv[4] = _mm_loadu_pd((double *)(x0 + 4 * incx));
                xv[5] = _mm_loadu_pd((double *)(x0 + 5 * incx));
                xv[6] = _mm_loadu_pd((double *)(x0 + 6 * incx));
                xv[7] = _mm_loadu_pd((double *)(x0 + 7 * incx));

                xv[8] = _mm_loadu_pd((double *)(x0 + 8 * incx));
                xv[9] = _mm_loadu_pd((double *)(x0 + 9 * incx));
                xv[10] = _mm_loadu_pd((double *)(x0 + 10 * incx));
                xv[11] = _mm_loadu_pd((double *)(x0 + 11 * incx));

                xv[12] = _mm_loadu_pd((double *)(x0 + 12 * incx));
                xv[13] = _mm_loadu_pd((double *)(x0 + 13 * incx));
                xv[14] = _mm_loadu_pd((double *)(x0 + 14 * incx));
                xv[15] = _mm_loadu_pd((double *)(x0 + 15 * incx));

                xv[16] = _mm_loadu_pd((double *)(x0 + 16 * incx));
                xv[17] = _mm_loadu_pd((double *)(x0 + 17 * incx));
                xv[18] = _mm_loadu_pd((double *)(x0 + 18 * incx));
                xv[19] = _mm_loadu_pd((double *)(x0 + 19 * incx));

                xv[20] = _mm_loadu_pd((double *)(x0 + 20 * incx));
                xv[21] = _mm_loadu_pd((double *)(x0 + 21 * incx));
                xv[22] = _mm_loadu_pd((double *)(x0 + 22 * incx));
                xv[23] = _mm_loadu_pd((double *)(x0 + 23 * incx));

                xv[24] = _mm_loadu_pd((double *)(x0 + 24 * incx));
                xv[25] = _mm_loadu_pd((double *)(x0 + 25 * incx));
                xv[26] = _mm_loadu_pd((double *)(x0 + 26 * incx));
                xv[27] = _mm_loadu_pd((double *)(x0 + 27 * incx));

                xv[28] = _mm_loadu_pd((double *)(x0 + 28 * incx));
                xv[29] = _mm_loadu_pd((double *)(x0 + 29 * incx));
                xv[30] = _mm_loadu_pd((double *)(x0 + 30 * incx));
                xv[31] = _mm_loadu_pd((double *)(x0 + 31 * incx));

                // Storing the values to destination
                _mm_storeu_pd((double *)(y0 + incy * 0), xv[0]);
                _mm_storeu_pd((double *)(y0 + incy * 1), xv[1]);
                _mm_storeu_pd((double *)(y0 + incy * 2), xv[2]);
                _mm_storeu_pd((double *)(y0 + incy * 3), xv[3]);

                _mm_storeu_pd((double *)(y0 + incy * 4), xv[4]);
                _mm_storeu_pd((double *)(y0 + incy * 5), xv[5]);
                _mm_storeu_pd((double *)(y0 + incy * 6), xv[6]);
                _mm_storeu_pd((double *)(y0 + incy * 7), xv[7]);

                _mm_storeu_pd((double *)(y0 + incy * 8), xv[8]);
                _mm_storeu_pd((double *)(y0 + incy * 9), xv[9]);
                _mm_storeu_pd((double *)(y0 + incy * 10), xv[10]);
                _mm_storeu_pd((double *)(y0 + incy * 11), xv[11]);

                _mm_storeu_pd((double *)(y0 + incy * 12), xv[12]);
                _mm_storeu_pd((double *)(y0 + incy * 13), xv[13]);
                _mm_storeu_pd((double *)(y0 + incy * 14), xv[14]);
                _mm_storeu_pd((double *)(y0 + incy * 15), xv[15]);

                _mm_storeu_pd((double *)(y0 + incy * 16), xv[16]);
                _mm_storeu_pd((double *)(y0 + incy * 17), xv[17]);
                _mm_storeu_pd((double *)(y0 + incy * 18), xv[18]);
                _mm_storeu_pd((double *)(y0 + incy * 19), xv[19]);

                _mm_storeu_pd((double *)(y0 + incy * 20), xv[20]);
                _mm_storeu_pd((double *)(y0 + incy * 21), xv[21]);
                _mm_storeu_pd((double *)(y0 + incy * 22), xv[22]);
                _mm_storeu_pd((double *)(y0 + incy * 23), xv[23]);

                _mm_storeu_pd((double *)(y0 + incy * 24), xv[24]);
                _mm_storeu_pd((double *)(y0 + incy * 25), xv[25]);
                _mm_storeu_pd((double *)(y0 + incy * 26), xv[26]);
                _mm_storeu_pd((double *)(y0 + incy * 27), xv[27]);

                _mm_storeu_pd((double *)(y0 + incy * 28), xv[28]);
                _mm_storeu_pd((double *)(y0 + incy * 29), xv[29]);
                _mm_storeu_pd((double *)(y0 + incy * 30), xv[30]);
                _mm_storeu_pd((double *)(y0 + incy * 31), xv[31]);

                // Increment the pointer
                x0 += 32 * incx;
                y0 += 32 * incy;
            }

            for ( ; i < (n & (~0x0F)); i += 16)
            {
                // Loading the input values
                xv[0] = _mm_loadu_pd((double *)(x0 + 0 * incx));
                xv[1] = _mm_loadu_pd((double *)(x0 + 1 * incx));
                xv[2] = _mm_loadu_pd((double *)(x0 + 2 * incx));
                xv[3] = _mm_loadu_pd((double *)(x0 + 3 * incx));

                xv[4] = _mm_loadu_pd((double *)(x0 + 4 * incx));
                xv[5] = _mm_loadu_pd((double *)(x0 + 5 * incx));
                xv[6] = _mm_loadu_pd((double *)(x0 + 6 * incx));
                xv[7] = _mm_loadu_pd((double *)(x0 + 7 * incx));

                xv[8] = _mm_loadu_pd((double *)(x0 + 8 * incx));
                xv[9] = _mm_loadu_pd((double *)(x0 + 9 * incx));
                xv[10] = _mm_loadu_pd((double *)(x0 + 10 * incx));
                xv[11] = _mm_loadu_pd((double *)(x0 + 11 * incx));

                xv[12] = _mm_loadu_pd((double *)(x0 + 12 * incx));
                xv[13] = _mm_loadu_pd((double *)(x0 + 13 * incx));
                xv[14] = _mm_loadu_pd((double *)(x0 + 14 * incx));
                xv[15] = _mm_loadu_pd((double *)(x0 + 15 * incx));

                // Storing the values to destination
                _mm_storeu_pd((double *)(y0 + incy * 0), xv[0]);
                _mm_storeu_pd((double *)(y0 + incy * 1), xv[1]);
                _mm_storeu_pd((double *)(y0 + incy * 2), xv[2]);
                _mm_storeu_pd((double *)(y0 + incy * 3), xv[3]);

                _mm_storeu_pd((double *)(y0 + incy * 4), xv[4]);
                _mm_storeu_pd((double *)(y0 + incy * 5), xv[5]);
                _mm_storeu_pd((double *)(y0 + incy * 6), xv[6]);
                _mm_storeu_pd((double *)(y0 + incy * 7), xv[7]);

                _mm_storeu_pd((double *)(y0 + incy * 8), xv[8]);
                _mm_storeu_pd((double *)(y0 + incy * 9), xv[9]);
                _mm_storeu_pd((double *)(y0 + incy * 10), xv[10]);
                _mm_storeu_pd((double *)(y0 + incy * 11), xv[11]);

                _mm_storeu_pd((double *)(y0 + incy * 12), xv[12]);
                _mm_storeu_pd((double *)(y0 + incy * 13), xv[13]);
                _mm_storeu_pd((double *)(y0 + incy * 14), xv[14]);
                _mm_storeu_pd((double *)(y0 + incy * 15), xv[15]);

                // Increment the pointer
                x0 += 16 * incx;
                y0 += 16 * incy;
            }

            for ( ; i < (n & (~0x07)); i += 8)
            {
                // Loading the input values
                xv[0] = _mm_loadu_pd((double *)(x0 + 0 * incx));
                xv[1] = _mm_loadu_pd((double *)(x0 + 1 * incx));
                xv[2] = _mm_loadu_pd((double *)(x0 + 2 * incx));
                xv[3] = _mm_loadu_pd((double *)(x0 + 3 * incx));

                xv[4] = _mm_loadu_pd((double *)(x0 + 4 * incx));
                xv[5] = _mm_loadu_pd((double *)(x0 + 5 * incx));
                xv[6] = _mm_loadu_pd((double *)(x0 + 6 * incx));
                xv[7] = _mm_loadu_pd((double *)(x0 + 7 * incx));

                // Storing the values to destination
                _mm_storeu_pd((double *)(y0 + incy * 0), xv[0]);
                _mm_storeu_pd((double *)(y0 + incy * 1), xv[1]);
                _mm_storeu_pd((double *)(y0 + incy * 2), xv[2]);
                _mm_storeu_pd((double *)(y0 + incy * 3), xv[3]);

                _mm_storeu_pd((double *)(y0 + incy * 4), xv[4]);
                _mm_storeu_pd((double *)(y0 + incy * 5), xv[5]);
                _mm_storeu_pd((double *)(y0 + incy * 6), xv[6]);
                _mm_storeu_pd((double *)(y0 + incy * 7), xv[7]);

                // Increment the pointer
                x0 += 8 * incx;
                y0 += 8 * incy;
            }

            for ( ; i < (n & (~0x03)); i += 4)
            {
                // Loading the input values
                xv[0] = _mm_loadu_pd((double *)(x0 + 0 * incx));
                xv[1] = _mm_loadu_pd((double *)(x0 + 1 * incx));
                xv[2] = _mm_loadu_pd((double *)(x0 + 2 * incx));
                xv[3] = _mm_loadu_pd((double *)(x0 + 3 * incx));

                // Storing the values to destination
                _mm_storeu_pd((double *)(y0 + incy * 0), xv[0]);
                _mm_storeu_pd((double *)(y0 + incy * 1), xv[1]);
                _mm_storeu_pd((double *)(y0 + incy * 2), xv[2]);
                _mm_storeu_pd((double *)(y0 + incy * 3), xv[3]);

                // Increment the pointer
                x0 += 4 * incx;
                y0 += 4 * incy;
            }

            for ( ; i < (n & (~0x01)); i += 2)
            {
                // Loading the input values
                xv[0] = _mm_loadu_pd((double *)(x0 + 0 * incx));
                xv[1] = _mm_loadu_pd((double *)(x0 + 1 * incx));

                // Storing the values to destination
                _mm_storeu_pd((double *)(y0 + incy * 0), xv[0]);
                _mm_storeu_pd((double *)(y0 + incy * 1), xv[1]);

                // Increment the pointer
                x0 += 2 * incx;
                y0 += 2 * incy;
            }

            for ( ; i < n; i += 1)
            {
                // Loading the input values
                xv[0] = _mm_loadu_pd((double *)(x0 + 0 * incx));

                // Storing the values to destination
                _mm_storeu_pd((double *)(y0 + incy * 0), xv[0]);

                // Increment the pointer
                x0 += 1 * incx;
                y0 += 1 * incy;
            }
        }
    }

    AOCL_DTL_TRACE_EXIT(AOCL_DTL_LEVEL_TRACE_2)
}
