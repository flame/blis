/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.

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

#define KER_THRESHOLD 3368

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

// This function is a wrapper function used to select the appropriate kernel based on the vector length and vector strides
void bli_dcopyv_zen5_asm_avx512
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
    uint64_t n0 = (uint64_t)n;

    /* Function pointer declaration for the function
	   that will be used by this API. */
	dcopyv_ker_ft copyv_ker_ptr;    // DCOPYV kernel function pointer

    // Selecting the kernel based on the vector length
    if ( n0 <= KER_THRESHOLD )
        copyv_ker_ptr = bli_dcopyv_zen4_asm_avx512;
    else
        copyv_ker_ptr = bli_dcopyv_zen4_asm_avx512_biway;

    // Invoke the function pointer.
    copyv_ker_ptr
    (
        conjx,
        n0,
        x0, incx,
        y0, incy,
        cntx
    );

    AOCL_DTL_TRACE_EXIT(AOCL_DTL_LEVEL_TRACE_2)
    return;
}