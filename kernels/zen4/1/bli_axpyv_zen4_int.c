/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2023 - 2026, Advanced Micro Devices, Inc. All rights reserved.

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

// -----------------------------------------------------------------------------

/*
    Functionality
    -------------

    This function calculates y := y + alpha * x where all three variables are of type
    float.

    Function Signature
    -------------------

    This function takes three float pointer as input, the correspending vector's stride
    and length. It uses the function parameters to return the output.

    * 'conjx' - Info about conjugation of x (This variable is not used in the kernel)
    * 'n' - Length of the array passed
    * 'alpha' - Float pointer to a scalar value
    * 'x' - Float pointer to an array
    * 'incx' - Stride to point to the next element in the array
    * 'y' - Float pointer to an array
    * 'incy' - Stride to point to the next element in the array
    * 'cntx' - BLIS context object

    Exception
    ----------

    None

    Deviation from BLAS
    --------------------

    None

    Undefined behaviour
    -------------------

    1. The kernel results in undefined behaviour when n <= 0, incx <= 0 and incy <= 0.
       The expectation is that these are standard BLAS exceptions and should be handled in
       a higher layer
*/
void bli_saxpyv_zen4_int
     (
       conj_t           conjx,
       dim_t            n,
       float*  restrict alpha,
       float*  restrict x, inc_t incx,
       float*  restrict y, inc_t incy,
       cntx_t* restrict cntx
     )
{
    AOCL_DTL_TRACE_ENTRY(AOCL_DTL_LEVEL_TRACE_4)

    // Initialize local pointers.
    float *restrict x0 = x;
    float *restrict y0 = y;
    float *restrict alpha0 = alpha;

    // If the vector dimension is zero return early.
    if (bli_zero_dim1(n) || bli_seq0(*alpha0))
    {
        AOCL_DTL_TRACE_EXIT(AOCL_DTL_LEVEL_TRACE_4)
        return;
    }

    // Check for unit stride - use optimized assembly code
    if (incx == 1 && incy == 1)
    {
        // Typecast to 64 bit
        uint64_t n0 = (uint64_t)n;

        // Assembly Code for unit stride case
        begin_asm()

        /*
            rsi - > n
            rax - > alpha
            rdx - > x
            r8  - > y
        */

        // Loading the source memory address to the respective registers
        mov(var(alpha0), rax)
        mov(var(x0), rdx)
        mov(var(y0), r8)

        // Loading the value in 'n' to register
        mov(var(n0), rsi)

        // Broadcast alpha to all elements of zmm0
        vbroadcastss(mem(rax), zmm0)

        // ========================================================================================================================

        // Section of code to process blocks of 128 elements
        label(.BLOCK128)

        cmp(imm(16*8), rsi)                // check if the number of remaining elements >= 128
        jl(.BLOCK64)                       // else, goto to the section of code for block of size 64

        label(.MAINLOOP)

        // First 64 elements (0-63)
        vmovups(mem(rdx, 0*64), zmm1)      // zmm1 = x[i+0] - x[i+15]
        vmovups(mem(rdx, 1*64), zmm2)      // zmm2 = x[i+16] - x[i+31]
        vmovups(mem(rdx, 2*64), zmm3)      // zmm3 = x[i+32] - x[i+47]
        vmovups(mem(rdx, 3*64), zmm4)      // zmm4 = x[i+48] - x[i+63]

        vfmadd213ps(mem(r8, 0*64), zmm0, zmm1)  // zmm1 = alpha*x + y[i+0]
        vfmadd213ps(mem(r8, 1*64), zmm0, zmm2)  // zmm2 = alpha*x + y[i+16]
        vfmadd213ps(mem(r8, 2*64), zmm0, zmm3)  // zmm3 = alpha*x + y[i+32]
        vfmadd213ps(mem(r8, 3*64), zmm0, zmm4)  // zmm4 = alpha*x + y[i+48]

        vmovups(zmm1, mem(r8, 0*64))       // y[i+0] - y[i+15] = zmm1
        vmovups(zmm2, mem(r8, 1*64))       // y[i+16] - y[i+31] = zmm2
        vmovups(zmm3, mem(r8, 2*64))       // y[i+32] - y[i+47] = zmm3
        vmovups(zmm4, mem(r8, 3*64))       // y[i+48] - y[i+63] = zmm4

        // Second 64 elements (64-127)
        vmovups(mem(rdx, 4*64), zmm5)      // zmm5 = x[i+64] - x[i+79]
        vmovups(mem(rdx, 5*64), zmm6)      // zmm6 = x[i+80] - x[i+95]
        vmovups(mem(rdx, 6*64), zmm7)      // zmm7 = x[i+96] - x[i+111]
        vmovups(mem(rdx, 7*64), zmm8)      // zmm8 = x[i+112] - x[i+127]

        vfmadd213ps(mem(r8, 4*64), zmm0, zmm5)  // zmm5 = alpha*x + y[i+64]
        vfmadd213ps(mem(r8, 5*64), zmm0, zmm6)  // zmm6 = alpha*x + y[i+80]
        vfmadd213ps(mem(r8, 6*64), zmm0, zmm7)  // zmm7 = alpha*x + y[i+96]
        vfmadd213ps(mem(r8, 7*64), zmm0, zmm8)  // zmm8 = alpha*x + y[i+112]

        vmovups(zmm5, mem(r8, 4*64))       // y[i+64] - y[i+79] = zmm5
        vmovups(zmm6, mem(r8, 5*64))       // y[i+80] - y[i+95] = zmm6
        vmovups(zmm7, mem(r8, 6*64))       // y[i+96] - y[i+111] = zmm7
        vmovups(zmm8, mem(r8, 7*64))       // y[i+112] - y[i+127] = zmm8

        // Increment the pointers
        add(imm(16*4*8), rdx)
        add(imm(16*4*8), r8)
        sub(imm(16*8), rsi)                // reduce the number of remaining elements by 128

        cmp(imm(16*8), rsi)
        jge(.MAINLOOP)

        // -----------------------------------------------------------

        // Section of code to process blocks of 64 elements
        label(.BLOCK64)

        cmp(imm(16*4), rsi)                // check if the number of remaining elements >= 64
        jl(.BLOCK32)                       // else, goto to the section of code for block of size 32

        vmovups(mem(rdx, 0*64), zmm1)      // zmm1 = x[i+0] - x[i+15]
        vmovups(mem(rdx, 1*64), zmm2)      // zmm2 = x[i+16] - x[i+31]
        vmovups(mem(rdx, 2*64), zmm3)      // zmm3 = x[i+32] - x[i+47]
        vmovups(mem(rdx, 3*64), zmm4)      // zmm4 = x[i+48] - x[i+63]

        vfmadd213ps(mem(r8, 0*64), zmm0, zmm1)  // zmm1 = alpha*x + y[i+0]
        vfmadd213ps(mem(r8, 1*64), zmm0, zmm2)  // zmm2 = alpha*x + y[i+16]
        vfmadd213ps(mem(r8, 2*64), zmm0, zmm3)  // zmm3 = alpha*x + y[i+32]
        vfmadd213ps(mem(r8, 3*64), zmm0, zmm4)  // zmm4 = alpha*x + y[i+48]

        vmovups(zmm1, mem(r8, 0*64))       // y[i+0] - y[i+15] = zmm1
        vmovups(zmm2, mem(r8, 1*64))       // y[i+16] - y[i+31] = zmm2
        vmovups(zmm3, mem(r8, 2*64))       // y[i+32] - y[i+47] = zmm3
        vmovups(zmm4, mem(r8, 3*64))       // y[i+48] - y[i+63] = zmm4

        // Increment the pointers
        add(imm(16*4*4), rdx)
        add(imm(16*4*4), r8)
        sub(imm(16*4), rsi)                // reduce the number of remaining elements by 64

        // -----------------------------------------------------------

        // Section of code to process blocks of 32 elements
        label(.BLOCK32)

        cmp(imm(16*2), rsi)                // check if the number of remaining elements >= 32
        jl(.BLOCK16)                       // else, goto to the section of code for block of size 16

        vmovups(mem(rdx, 0*64), zmm1)      // zmm1 = x[i+0] - x[i+15]
        vmovups(mem(rdx, 1*64), zmm2)      // zmm2 = x[i+16] - x[i+31]

        vfmadd213ps(mem(r8, 0*64), zmm0, zmm1)  // zmm1 = alpha*x + y[i+0]
        vfmadd213ps(mem(r8, 1*64), zmm0, zmm2)  // zmm2 = alpha*x + y[i+16]

        vmovups(zmm1, mem(r8, 0*64))       // y[i+0] - y[i+15] = zmm1
        vmovups(zmm2, mem(r8, 1*64))       // y[i+16] - y[i+31] = zmm2

        add(imm(16*4*2), rdx)
        add(imm(16*4*2), r8)
        sub(imm(16*2), rsi)                // reduce the number of remaining elements by 32

        // -----------------------------------------------------------

        // Section of code to process blocks of 16 elements
        label(.BLOCK16)

        cmp(imm(16), rsi)                  // check if the number of remaining elements >= 16
        jl(.FRINGE)                        // else, goto to the section of code for fringe cases

        vmovups(mem(rdx, 0*64), zmm1)      // zmm1 = x[i+0] - x[i+15]
        vfmadd213ps(mem(r8, 0*64), zmm0, zmm1)  // zmm1 = alpha*x + y[i+0]
        vmovups(zmm1, mem(r8, 0*64))       // y[i+0] - y[i+15] = zmm1

        // Increment the pointers
        add(imm(16*4), rdx)
        add(imm(16*4), r8)
        sub(imm(16), rsi)                  // reduce the number of remaining elements by 16

        // -----------------------------------------------------------

        // Section of code to process remaining elements (up to 16) using masked operations
        label(.FRINGE)

        test(rsi, rsi)                     // check if there are any remaining elements
        je(.END)

        // Creating a 16-bit mask for remaining elements (1-16)
        mov(imm(65535), rcx)               // (65535)BASE_10 -> (1111 1111 1111 1111)BASE_2
        shlx(rsi, rcx, rcx)                // shifting the bits to the left depending on remaining elements
        xor(imm(65535), rcx)               // taking complement of the register
        kmovq(rcx, k(2))                   // copying the value to mask register

        // Loading the input values using masked load
        vmovups(mem(rdx, 0*64), zmm1 MASK_(K(2)))

        // Perform FMA with masked operation
        vfmadd213ps(mem(r8, 0*64), zmm0, zmm1 MASK_(K(2)))

        // Storing the values to destination using masked store
        vmovups(zmm1, mem(r8) MASK_(K(2)))

        label(.END)
        end_asm(
            : // output operands
            : // input operands
            [n0]     "m"     (n0),
            [alpha0] "m"     (alpha0),
            [x0]     "m"     (x0),
            [y0]     "m"     (y0)
            : // register clobber list
            "zmm0",  "zmm1",  "zmm2",  "zmm3",
            "zmm4",  "zmm5",  "zmm6",  "zmm7",
            "zmm8",  "ymm0",  "ymm1",  "xmm0",
            "xmm1",  "rsi",   "rax",   "rdx",
            "rcx",   "r8",    "k2",    "memory"
        )
    }
    else
    {
        /**
        * SAXPY implementation for non-unit strides (incx/incy != 1).
        * * While the loop remains scalar due to non-contiguous memory access, we use
        * SSE scalar intrinsics (_mm_fmadd_ss) to leverage the hardware's Fused
        * Multiply-Add (FMA) unit. This reduces instruction count and maintains
        * higher precision by performing the multiply-add in a single step.
        */

        // // Scalar code for non-unit stride
        // for (dim_t i = 0; i < n; ++i)
        // {
        //     *y0 += (*alpha0) * (*x0);

        //     x0 += incx;
        //     y0 += incy;
        // }

        // Use scalar SSE FMA for non-unit stride
        __m128 alphas = _mm_set_ss(*alpha0);
        for (dim_t i = 0; i < n; ++i)
        {
            // Load single float from x0 and y0
            __m128 xv = _mm_load_ss(x0);
            __m128 yv = _mm_load_ss(y0);
            // FMA: yv = (xv * alphas) + yv
            yv = _mm_fmadd_ss(xv, alphas, yv);
            // Store single float back to y0
            _mm_store_ss(y0, yv);
            x0 += incx;
            y0 += incy;
        }
    }

    AOCL_DTL_TRACE_EXIT(AOCL_DTL_LEVEL_TRACE_4)
}

// -----------------------------------------------------------------------------

/*
    Functionality
    -------------

    This function calculates y := y + alpha * x where all three variables are of type
    double.

    Function Signature
    -------------------

    This function takes three float pointer as input, the correspending vector's stride
    and length. It uses the function parameters to return the output.

    * 'conjx' - Info about conjugation of x (This variable is not used in the kernel)
    * 'n' - Length of the array passed
    * 'alpha' - Double pointer to a scalar value
    * 'x' - Double pointer to an array
    * 'incx' - Stride to point to the next element in the array
    * 'y' - Double pointer to an array
    * 'incy' - Stride to point to the next element in the array
    * 'cntx' - BLIS context object

    Exception
    ----------

    None

    Deviation from BLAS
    --------------------

    None

    Undefined behaviour
    -------------------

    1. The kernel results in undefined behaviour when n <= 0, incx <= 0 and incy <= 0.
       The expectation is that these are standard BLAS exceptions and should be handled in
       a higher layer
*/
BLIS_EXPORT_BLIS void bli_daxpyv_zen4_int
     (
       conj_t           conjx,
       dim_t            n,
       double*  restrict alpha,
       double*  restrict x, inc_t incx,
       double*  restrict y, inc_t incy,
       cntx_t* restrict cntx
     )
{
    const int n_elem_per_reg = 8;

    dim_t i = 0;

    // Initialize local pointers.
    double *restrict x0 = x;
    double *restrict y0 = y;

    if (incx == 1 && incy == 1)
    {
        if ( n < 8 )
        {
            // At this point we are sure that N is not more than 7
            // If N is at least 4, use a full AVX2 FMA
            if ( ( i + 3 ) < n )
            {
                __m256d y_vec = _mm256_loadu_pd(y0);
                y_vec = _mm256_fmadd_pd(_mm256_loadu_pd(x0), _mm256_set1_pd(*alpha), y_vec);
                _mm256_storeu_pd(y0, y_vec);

                x0 += 4;
                y0 += 4;
                i += 4;
            }

            // At this point remainder is not more than 3
            // If remainder is at least 2, we use full SSE FMA
            if ( ( i + 1 ) < n )
            {
                __m128d y_vec = _mm_loadu_pd(y0);
                y_vec = _mm_fmadd_pd(_mm_loadu_pd(x0), _mm_set1_pd(*alpha), y_vec);
                _mm_storeu_pd(y0, y_vec);

                x0 += 2;
                y0 += 2;
                i += 2;
            }

            // At this point remainder is either 0 or 1
            // If remainder is 1, we use SSE FMA
            // Note: Using C code instead of SSE results in MUL+ADD which is
            // bad for accuracy.
            if ( i < n )
            {
                __m128d y_vec = _mm_load1_pd(y0);
                y_vec = _mm_fmadd_pd(_mm_load1_pd(x0), _mm_set1_pd(*alpha), y_vec);
                _mm_storel_pd(y0, y_vec);
            }

            AOCL_DTL_TRACE_EXIT(AOCL_DTL_LEVEL_TRACE_4);
            return;
        }

        __m512d xv[8], yv[8], alphav;

        // Broadcast the alpha scalar to all elements of a vector register.
        alphav = _mm512_set1_pd(*alpha);

        for (i = 0; (i + 63) < n; i += 64)
        {
            // Loading the input values
            xv[0] = _mm512_loadu_pd(x0 + 0 * n_elem_per_reg);
            xv[1] = _mm512_loadu_pd(x0 + 1 * n_elem_per_reg);
            xv[2] = _mm512_loadu_pd(x0 + 2 * n_elem_per_reg);
            xv[3] = _mm512_loadu_pd(x0 + 3 * n_elem_per_reg);

            yv[0] = _mm512_loadu_pd(y0 + 0 * n_elem_per_reg);
            yv[1] = _mm512_loadu_pd(y0 + 1 * n_elem_per_reg);
            yv[2] = _mm512_loadu_pd(y0 + 2 * n_elem_per_reg);
            yv[3] = _mm512_loadu_pd(y0 + 3 * n_elem_per_reg);

            // Perform y += alpha * x
            yv[0] = _mm512_fmadd_pd(xv[0], alphav, yv[0]);
            yv[1] = _mm512_fmadd_pd(xv[1], alphav, yv[1]);
            yv[2] = _mm512_fmadd_pd(xv[2], alphav, yv[2]);
            yv[3] = _mm512_fmadd_pd(xv[3], alphav, yv[3]);

            // Store updated y
            _mm512_storeu_pd((y0 + 0 * n_elem_per_reg), yv[0]);
            _mm512_storeu_pd((y0 + 1 * n_elem_per_reg), yv[1]);
            _mm512_storeu_pd((y0 + 2 * n_elem_per_reg), yv[2]);
            _mm512_storeu_pd((y0 + 3 * n_elem_per_reg), yv[3]);

            xv[4] = _mm512_loadu_pd(x0 + 4 * n_elem_per_reg);
            xv[5] = _mm512_loadu_pd(x0 + 5 * n_elem_per_reg);
            xv[6] = _mm512_loadu_pd(x0 + 6 * n_elem_per_reg);
            xv[7] = _mm512_loadu_pd(x0 + 7 * n_elem_per_reg);

            yv[4] = _mm512_loadu_pd(y0 + 4 * n_elem_per_reg);
            yv[5] = _mm512_loadu_pd(y0 + 5 * n_elem_per_reg);
            yv[6] = _mm512_loadu_pd(y0 + 6 * n_elem_per_reg);
            yv[7] = _mm512_loadu_pd(y0 + 7 * n_elem_per_reg);

            yv[4] = _mm512_fmadd_pd(xv[4], alphav, yv[4]);
            yv[5] = _mm512_fmadd_pd(xv[5], alphav, yv[5]);
            yv[6] = _mm512_fmadd_pd(xv[6], alphav, yv[6]);
            yv[7] = _mm512_fmadd_pd(xv[7], alphav, yv[7]);

            _mm512_storeu_pd((y0 + 7 * n_elem_per_reg), yv[7]);
            _mm512_storeu_pd((y0 + 6 * n_elem_per_reg), yv[6]);
            _mm512_storeu_pd((y0 + 5 * n_elem_per_reg), yv[5]);
            _mm512_storeu_pd((y0 + 4 * n_elem_per_reg), yv[4]);

            x0 += 8 * n_elem_per_reg;
            y0 += 8 * n_elem_per_reg;
        }

        for (; (i + 31) < n; i += 32)
        {
            xv[0] = _mm512_loadu_pd(x0 + 0 * n_elem_per_reg);
            xv[1] = _mm512_loadu_pd(x0 + 1 * n_elem_per_reg);
            xv[2] = _mm512_loadu_pd(x0 + 2 * n_elem_per_reg);
            xv[3] = _mm512_loadu_pd(x0 + 3 * n_elem_per_reg);

            yv[0] = _mm512_loadu_pd(y0 + 0 * n_elem_per_reg);
            yv[1] = _mm512_loadu_pd(y0 + 1 * n_elem_per_reg);
            yv[2] = _mm512_loadu_pd(y0 + 2 * n_elem_per_reg);
            yv[3] = _mm512_loadu_pd(y0 + 3 * n_elem_per_reg);

            yv[0] = _mm512_fmadd_pd(xv[0], alphav, yv[0]);
            yv[1] = _mm512_fmadd_pd(xv[1], alphav, yv[1]);
            yv[2] = _mm512_fmadd_pd(xv[2], alphav, yv[2]);
            yv[3] = _mm512_fmadd_pd(xv[3], alphav, yv[3]);

            _mm512_storeu_pd((y0 + 0 * n_elem_per_reg), yv[0]);
            _mm512_storeu_pd((y0 + 1 * n_elem_per_reg), yv[1]);
            _mm512_storeu_pd((y0 + 2 * n_elem_per_reg), yv[2]);
            _mm512_storeu_pd((y0 + 3 * n_elem_per_reg), yv[3]);

            x0 += 4 * n_elem_per_reg;
            y0 += 4 * n_elem_per_reg;
        }

        for (; (i + 15) < n; i += 16)
        {
            xv[0] = _mm512_loadu_pd(x0 + 0 * n_elem_per_reg);
            xv[1] = _mm512_loadu_pd(x0 + 1 * n_elem_per_reg);

            yv[0] = _mm512_loadu_pd(y0 + 0 * n_elem_per_reg);
            yv[1] = _mm512_loadu_pd(y0 + 1 * n_elem_per_reg);

            yv[0] = _mm512_fmadd_pd(xv[0], alphav, yv[0]);
            yv[1] = _mm512_fmadd_pd(xv[1], alphav, yv[1]);

            _mm512_storeu_pd((y0 + 0 * n_elem_per_reg), yv[0]);
            _mm512_storeu_pd((y0 + 1 * n_elem_per_reg), yv[1]);

            x0 += 2 * n_elem_per_reg;
            y0 += 2 * n_elem_per_reg;
        }

        for (; (i + 7) < n; i += 8)
        {
            xv[0] = _mm512_loadu_pd(x0);

            yv[0] = _mm512_loadu_pd(y0);

            yv[0] = _mm512_fmadd_pd(xv[0], alphav, yv[0]);

            _mm512_storeu_pd(y0, yv[0]);

            x0 += n_elem_per_reg;
            y0 += n_elem_per_reg;
        }

        // compute the remainder with masked operations
        if ( i < n )
        {
            dim_t n_remainder = ( n - i );
            __mmask8 mask_ = ( 1 <<  n_remainder ) - 1;

            xv[0] = _mm512_maskz_loadu_pd( mask_, x0 );

            yv[0] = _mm512_maskz_loadu_pd( mask_, y0 );

            yv[0] = _mm512_fmadd_pd(xv[0], alphav, yv[0]);

            _mm512_mask_storeu_pd( y0, mask_, yv[0] );

            x0 += n_remainder;
            y0 += n_remainder;
            i  += n_remainder;
        }
    }

    /*
        This loop has two functions:
        1. Handles the remainder of n / 4 when incx and incy are 1.
        2. Performs the complete compute when incx or incy != 1
    */
    for (; i < n; i += 1)
    {
        *y0 += (*alpha) * (*x0);

        x0 += incx;
        y0 += incy;
    }

    AOCL_DTL_TRACE_EXIT(AOCL_DTL_LEVEL_TRACE_4)
}

// -----------------------------------------------------------------------------

/*
    Functionality
    -------------

    This function calculates y := y + alpha * x where all three variables are of type
    double.

    Function Signature
    -------------------

    This function takes three float pointer as input, the correspending vector's stride
    and length. It uses the function parameters to return the output.

    * 'conjx' - Info about conjugation of x (This variable is not used in the kernel)
    * 'n' - Length of the array passed
    * 'alpha' - Double pointer to a scalar value
    * 'x' - Double pointer to an array
    * 'incx' - Stride to point to the next element in the array
    * 'y' - Double pointer to an array
    * 'incy' - Stride to point to the next element in the array
    * 'cntx' - BLIS context object

    Exception
    ----------

    None

    Deviation from BLAS
    --------------------

    None

    Undefined behaviour
    -------------------

    1. The kernel results in undefined behaviour when n <= 0, incx <= 0 and incy <= 0.
       The expectation is that these are standard BLAS exceptions and should be handled in
       a higher layer
*/
void bli_zaxpyv_zen4_int
     (
       conj_t           conjx,
       dim_t            n,
       dcomplex*  restrict alpha,
       dcomplex*  restrict x, inc_t incx,
       dcomplex*  restrict y, inc_t incy,
       cntx_t* restrict cntx
     )
{
    const int n_elem_per_reg = 8;

    dim_t i = 0;

    // Initialize local pointers.
    double *restrict x0 = (double *)x;
    double *restrict y0 = (double *)y;

    if (incx == 1 && incy == 1)
    {
        __m512d xv[8], yv[8], alphaRv, alphaIv;

        // Broadcast real and imag parts of alpha to separate registers
        alphaRv = _mm512_set1_pd(alpha->real);
        alphaIv = _mm512_set1_pd(alpha->imag);

        xv[0] = _mm512_setzero_pd();

        // Handle X conjugate by negating some elements of alphaRv/alphaIv
        if ( bli_is_noconj( conjx ) )
            alphaIv = _mm512_fmaddsub_pd(xv[0], xv[0], alphaIv);
        else
            alphaRv = _mm512_fmsubadd_pd(xv[0], xv[0], alphaRv);

        // To check if code has to go to masked load/store directly
        if ( n >= 4 )
        {
            for (; (i + 31) < n; i += 32)
            {
                // Loading elements from X
                xv[0] = _mm512_loadu_pd(x0 + 0 * n_elem_per_reg);
                xv[1] = _mm512_loadu_pd(x0 + 1 * n_elem_per_reg);
                xv[2] = _mm512_loadu_pd(x0 + 2 * n_elem_per_reg);
                xv[3] = _mm512_loadu_pd(x0 + 3 * n_elem_per_reg);

                // Loading elements from Y
                yv[0] = _mm512_loadu_pd(y0 + 0 * n_elem_per_reg);
                yv[1] = _mm512_loadu_pd(y0 + 1 * n_elem_per_reg);
                yv[2] = _mm512_loadu_pd(y0 + 2 * n_elem_per_reg);
                yv[3] = _mm512_loadu_pd(y0 + 3 * n_elem_per_reg);

                // Scale X with real-part of alpha and add to Y
                yv[0] = _mm512_fmadd_pd(alphaRv, xv[0], yv[0]);
                yv[1] = _mm512_fmadd_pd(alphaRv, xv[1], yv[1]);
                yv[2] = _mm512_fmadd_pd(alphaRv, xv[2], yv[2]);
                yv[3] = _mm512_fmadd_pd(alphaRv, xv[3], yv[3]);

                // Swapping real and imag parts of every element in X
                xv[0] = _mm512_permute_pd(xv[0], 0x55);
                xv[1] = _mm512_permute_pd(xv[1], 0x55);
                xv[2] = _mm512_permute_pd(xv[2], 0x55);
                xv[3] = _mm512_permute_pd(xv[3], 0x55);

                // Scale X with imag-part of alpha and add to Y
                yv[0] = _mm512_fmadd_pd(alphaIv, xv[0], yv[0]);
                yv[1] = _mm512_fmadd_pd(alphaIv, xv[1], yv[1]);
                yv[2] = _mm512_fmadd_pd(alphaIv, xv[2], yv[2]);
                yv[3] = _mm512_fmadd_pd(alphaIv, xv[3], yv[3]);

                // Store updated Y
                _mm512_storeu_pd((y0 + 0 * n_elem_per_reg), yv[0]);
                _mm512_storeu_pd((y0 + 1 * n_elem_per_reg), yv[1]);
                _mm512_storeu_pd((y0 + 2 * n_elem_per_reg), yv[2]);
                _mm512_storeu_pd((y0 + 3 * n_elem_per_reg), yv[3]);

                // Loading elements from X
                xv[4] = _mm512_loadu_pd(x0 + 4 * n_elem_per_reg);
                xv[5] = _mm512_loadu_pd(x0 + 5 * n_elem_per_reg);
                xv[6] = _mm512_loadu_pd(x0 + 6 * n_elem_per_reg);
                xv[7] = _mm512_loadu_pd(x0 + 7 * n_elem_per_reg);

                // Loading elements from Y
                yv[4] = _mm512_loadu_pd(y0 + 4 * n_elem_per_reg);
                yv[5] = _mm512_loadu_pd(y0 + 5 * n_elem_per_reg);
                yv[6] = _mm512_loadu_pd(y0 + 6 * n_elem_per_reg);
                yv[7] = _mm512_loadu_pd(y0 + 7 * n_elem_per_reg);

                // Scale X with real-part of alpha and add to Y
                yv[4] = _mm512_fmadd_pd(alphaRv, xv[4], yv[4]);
                yv[5] = _mm512_fmadd_pd(alphaRv, xv[5], yv[5]);
                yv[6] = _mm512_fmadd_pd(alphaRv, xv[6], yv[6]);
                yv[7] = _mm512_fmadd_pd(alphaRv, xv[7], yv[7]);

                // Swapping real and imag parts of every element in X
                xv[4] = _mm512_permute_pd(xv[4], 0x55);
                xv[5] = _mm512_permute_pd(xv[5], 0x55);
                xv[6] = _mm512_permute_pd(xv[6], 0x55);
                xv[7] = _mm512_permute_pd(xv[7], 0x55);

                // Scale X with imag-part of alpha and add to Y
                yv[4] = _mm512_fmadd_pd(alphaIv, xv[4], yv[4]);
                yv[5] = _mm512_fmadd_pd(alphaIv, xv[5], yv[5]);
                yv[6] = _mm512_fmadd_pd(alphaIv, xv[6], yv[6]);
                yv[7] = _mm512_fmadd_pd(alphaIv, xv[7], yv[7]);

                // Store updated Y
                _mm512_storeu_pd((y0 + 4 * n_elem_per_reg), yv[4]);
                _mm512_storeu_pd((y0 + 5 * n_elem_per_reg), yv[5]);
                _mm512_storeu_pd((y0 + 6 * n_elem_per_reg), yv[6]);
                _mm512_storeu_pd((y0 + 7 * n_elem_per_reg), yv[7]);

                x0 += 8 * n_elem_per_reg;
                y0 += 8 * n_elem_per_reg;
            }

            for (; (i + 15) < n; i += 16)
            {
                // Loading elements from X
                xv[0] = _mm512_loadu_pd(x0 + 0 * n_elem_per_reg);
                xv[1] = _mm512_loadu_pd(x0 + 1 * n_elem_per_reg);
                xv[2] = _mm512_loadu_pd(x0 + 2 * n_elem_per_reg);
                xv[3] = _mm512_loadu_pd(x0 + 3 * n_elem_per_reg);

                // Loading elements from Y
                yv[0] = _mm512_loadu_pd(y0 + 0 * n_elem_per_reg);
                yv[1] = _mm512_loadu_pd(y0 + 1 * n_elem_per_reg);
                yv[2] = _mm512_loadu_pd(y0 + 2 * n_elem_per_reg);
                yv[3] = _mm512_loadu_pd(y0 + 3 * n_elem_per_reg);

                // Scale X with real-part of alpha and add to Y
                yv[0] = _mm512_fmadd_pd(alphaRv, xv[0], yv[0]);
                yv[1] = _mm512_fmadd_pd(alphaRv, xv[1], yv[1]);
                yv[2] = _mm512_fmadd_pd(alphaRv, xv[2], yv[2]);
                yv[3] = _mm512_fmadd_pd(alphaRv, xv[3], yv[3]);

                // Swapping real and imag parts of every element in X
                xv[0] = _mm512_permute_pd(xv[0], 0x55);
                xv[1] = _mm512_permute_pd(xv[1], 0x55);
                xv[2] = _mm512_permute_pd(xv[2], 0x55);
                xv[3] = _mm512_permute_pd(xv[3], 0x55);

                // Scale X with imag-part of alpha and add to Y
                yv[0] = _mm512_fmadd_pd(alphaIv, xv[0], yv[0]);
                yv[1] = _mm512_fmadd_pd(alphaIv, xv[1], yv[1]);
                yv[2] = _mm512_fmadd_pd(alphaIv, xv[2], yv[2]);
                yv[3] = _mm512_fmadd_pd(alphaIv, xv[3], yv[3]);

                // Store updated Y
                _mm512_storeu_pd((y0 + 0 * n_elem_per_reg), yv[0]);
                _mm512_storeu_pd((y0 + 1 * n_elem_per_reg), yv[1]);
                _mm512_storeu_pd((y0 + 2 * n_elem_per_reg), yv[2]);
                _mm512_storeu_pd((y0 + 3 * n_elem_per_reg), yv[3]);

                x0 += 4 * n_elem_per_reg;
                y0 += 4 * n_elem_per_reg;
            }

            for (; (i + 7) < n; i += 8)
            {
                // Loading elements from X
                xv[0] = _mm512_loadu_pd(x0 + 0 * n_elem_per_reg);
                xv[1] = _mm512_loadu_pd(x0 + 1 * n_elem_per_reg);

                // Loading elements from Y
                yv[0] = _mm512_loadu_pd(y0 + 0 * n_elem_per_reg);
                yv[1] = _mm512_loadu_pd(y0 + 1 * n_elem_per_reg);

                // Scale X with real-part of alpha and add to Y
                yv[0] = _mm512_fmadd_pd(alphaRv, xv[0], yv[0]);
                yv[1] = _mm512_fmadd_pd(alphaRv, xv[1], yv[1]);

                // Swapping real and imag parts of every element in X
                xv[0] = _mm512_permute_pd(xv[0], 0x55);
                xv[1] = _mm512_permute_pd(xv[1], 0x55);

                // Scale X with imag-part of alpha and add to Y
                yv[0] = _mm512_fmadd_pd(alphaIv, xv[0], yv[0]);
                yv[1] = _mm512_fmadd_pd(alphaIv, xv[1], yv[1]);

                // Store updated Y
                _mm512_storeu_pd((y0 + 0 * n_elem_per_reg), yv[0]);
                _mm512_storeu_pd((y0 + 1 * n_elem_per_reg), yv[1]);

                x0 += 2 * n_elem_per_reg;
                y0 += 2 * n_elem_per_reg;
            }

            for (; (i + 3) < n; i += 4)
            {
                // Loading elements from X
                xv[0] = _mm512_loadu_pd(x0 + 0 * n_elem_per_reg);

                // Loading elements from Y
                yv[0] = _mm512_loadu_pd(y0 + 0 * n_elem_per_reg);

                // Scale X with real-part of alpha and add to Y
                yv[0] = _mm512_fmadd_pd(alphaRv, xv[0], yv[0]);

                                // Swapping real and imag parts of every element in X
                xv[0] = _mm512_permute_pd(xv[0], 0x55);

                // Scale X with imag-part of alpha and add to Y
                yv[0] = _mm512_fmadd_pd(alphaIv, xv[0], yv[0]);

                // Store updated Y
                _mm512_storeu_pd((y0 + 0 * n_elem_per_reg), yv[0]);

                x0 += n_elem_per_reg;
                y0 += n_elem_per_reg;

            }
        }

        if ( i < n )
        {
            // Setting the mask bit based on remaining elements
            // Since each dcomplex elements corresponds to 2 doubles
            // we need to load and store 2*(n-i) elements.
            __mmask8 n_mask = (1 << 2*(n - i)) - 1;

            // Loading elements from X
            xv[0] = _mm512_maskz_loadu_pd(n_mask, x0);

            // Loading elements from Y
            yv[0] = _mm512_maskz_loadu_pd(n_mask, y0);

            // Scale X with real-part of alpha and add to Y
            yv[0] = _mm512_fmadd_pd(alphaRv, xv[0], yv[0]);

            // Swapping real and imag parts of every element in X
            xv[0] = _mm512_permute_pd(xv[0], 0x55);

            // Scale X with imag-part of alpha and add to Y
            yv[0] = _mm512_fmadd_pd(alphaIv, xv[0], yv[0]);

            // Store updated Y
            _mm512_mask_storeu_pd(y0, n_mask, yv[0]);
        }
    }
    else
    {
        __m128d xv, yv, temp, alphaRv, alphaIv;

        alphaRv = _mm_loaddup_pd((double *)alpha);
        alphaIv = _mm_loaddup_pd((double *)alpha + 1);

        xv = _mm_setzero_pd();

        if (bli_is_noconj(conjx))
            alphaIv = _mm_addsub_pd(xv, alphaIv);
        else
        {
            alphaRv = _mm_addsub_pd(xv, alphaRv);
            alphaRv = _mm_shuffle_pd(alphaRv, alphaRv, 0x01);
        }

        for (; i < n; i += 1)
        {
            xv = _mm_loadu_pd(x0);
            yv = _mm_loadu_pd(y0);

            temp = _mm_shuffle_pd(xv, xv, 0x01);

            temp = _mm_mul_pd(alphaIv, temp);
            xv = _mm_mul_pd(alphaRv, xv);

            xv = _mm_add_pd(xv, temp);
            yv = _mm_add_pd(yv, xv);

            _mm_storeu_pd(y0, yv);

            x0 += 2 * incx;
            y0 += 2 * incy;
        }
    }

    AOCL_DTL_TRACE_EXIT(AOCL_DTL_LEVEL_TRACE_4)
}
