/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2014, The University of Texas at Austin
   Copyright (C) 2020-22, Advanced Micro Devices, Inc. All rights reserved.

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


//
// Define BLAS-to-BLIS interfaces.
//
#undef  GENTFUNC
#define GENTFUNC( ftype, ch, blasname, blisname ) \
\
void PASTEF77(ch,blasname) \
     ( \
       const f77_int* n, \
       ftype*   x, const f77_int* incx, \
       ftype*   y, const f77_int* incy  \
     ) \
{ \
	AOCL_DTL_TRACE_ENTRY(AOCL_DTL_LEVEL_TRACE_1) \
    dim_t  n0; \
    ftype* x0; \
    ftype* y0; \
    inc_t  incx0; \
    inc_t  incy0; \
\
    /* Initialize BLIS. */ \
    bli_init_auto(); \
\
    /* Convert/typecast negative values of n to zero. */ \
    bli_convert_blas_dim1( *n, n0 ); \
\
    /* If the input increments are negative, adjust the pointers so we can
       use positive increments instead. */ \
    bli_convert_blas_incv( n0, (ftype*)x, *incx, x0, incx0 ); \
    bli_convert_blas_incv( n0, (ftype*)y, *incy, y0, incy0 ); \
\
    /* Call BLIS interface. */ \
    PASTEMAC2(ch,blisname,BLIS_TAPI_EX_SUF) \
    ( \
      n0, \
      x0, incx0, \
      y0, incy0, \
      NULL, \
      NULL  \
    ); \
\
    AOCL_DTL_TRACE_EXIT(AOCL_DTL_LEVEL_TRACE_1) \
    /* Finalize BLIS. */ \
    bli_finalize_auto(); \
}

#ifdef BLIS_ENABLE_BLAS

void sswap_
     (
       const f77_int* n,
       float*   x, const f77_int* incx,
       float*   y, const f77_int* incy
     )
{
    AOCL_DTL_TRACE_ENTRY(AOCL_DTL_LEVEL_TRACE_1)
    AOCL_DTL_LOG_SWAP_INPUTS(AOCL_DTL_LEVEL_TRACE_1, 'S', *n, *incx, *incy);
    dim_t  n0;
    float* x0;
    float* y0;
    inc_t  incx0;
    inc_t  incy0;

    /* Initialize BLIS. */
//  bli_init_auto();

    /* Convert/typecast negative values of n to zero. */
    if ( *n < 0 ) n0 = ( dim_t )0;
    else              n0 = ( dim_t )(*n);

    /* If the input increments are negative, adjust the pointers so we can
       use positive increments instead. */
    if ( *incx < 0 )
    {
        /* The semantics of negative stride in BLAS are that the vector
        operand be traversed in reverse order. (Another way to think
        of this is that negative strides effectively reverse the order
        of the vector, but without any explicit data movements.) This
        is also how BLIS interprets negative strides. The differences
        is that with BLAS, the caller *always* passes in the 0th (i.e.,
        top-most or left-most) element of the vector, even when the
        stride is negative. By contrast, in BLIS, negative strides are
        used *relative* to the vector address as it is given. Thus, in
        BLIS, if this backwards traversal is desired, the caller *must*
        pass in the address to the (n-1)th (i.e., the bottom-most or
        right-most) element along with a negative stride. */

        x0    = (x) + (n0-1)*(-*incx);
        incx0 = ( inc_t )(*incx);

    }
    else
    {
        x0    = (x);
        incx0 = ( inc_t )(*incx);
    }

    if ( *incy < 0 )
    {
        y0    = (y) + (n0-1)*(-*incy);
        incy0 = ( inc_t )(*incy);

    }
    else
    {
        y0    = (y);
        incy0 = ( inc_t )(*incy);
    }

    // This function is invoked on all architectures including ‘generic’.
    // Non-AVX platforms will use the kernels derived from the context.
    if (bli_cpuid_is_avx_supported() == TRUE) {
        /* Call BLIS kernel */
	    bli_sswapv_zen_int8
		    (
		     n0,
		     x0, incx0,
		     y0, incy0,
		     NULL
		    );
    }
    else{
	    PASTEMAC2(s,swapv,BLIS_TAPI_EX_SUF) \
		    ( \
		      n0, \
		      x0, incx0, \
		      y0, incy0, \
		      NULL, \
		      NULL  \
		    ); \
    }

    /* Finalize BLIS. */
//    bli_finalize_auto();
    AOCL_DTL_TRACE_EXIT(AOCL_DTL_LEVEL_TRACE_1)
}

void dswap_
     (
       const f77_int* n,
       double*   x, const f77_int* incx,
       double*   y, const f77_int* incy
     )
{
    AOCL_DTL_TRACE_ENTRY(AOCL_DTL_LEVEL_TRACE_1)
    AOCL_DTL_LOG_SWAP_INPUTS(AOCL_DTL_LEVEL_TRACE_1, 'D', *n, *incx, *incy);
    dim_t  n0;
    double* x0;
    double* y0;
    inc_t  incx0;
    inc_t  incy0;

    /* Initialize BLIS. */
//  bli_init_auto();

    /* Convert/typecast negative values of n to zero. */
    if ( *n < 0 ) n0 = ( dim_t )0;
    else              n0 = ( dim_t )(*n);

    /* If the input increments are negative, adjust the pointers so we can
       use positive increments instead. */
    if ( *incx < 0 )
    {
        /* The semantics of negative stride in BLAS are that the vector
        operand be traversed in reverse order. (Another way to think
        of this is that negative strides effectively reverse the order
        of the vector, but without any explicit data movements.) This
        is also how BLIS interprets negative strides. The differences
        is that with BLAS, the caller *always* passes in the 0th (i.e.,
        top-most or left-most) element of the vector, even when the
        stride is negative. By contrast, in BLIS, negative strides are
        used *relative* to the vector address as it is given. Thus, in
        BLIS, if this backwards traversal is desired, the caller *must*
        pass in the address to the (n-1)th (i.e., the bottom-most or
        right-most) element along with a negative stride. */

        x0    = (x) + (n0-1)*(-*incx);
        incx0 = ( inc_t )(*incx);

    }
    else
    {
        x0    = (x);
        incx0 = ( inc_t )(*incx);
    }

    if ( *incy < 0 )
    {
        y0    = (y) + (n0-1)*(-*incy);
        incy0 = ( inc_t )(*incy);

    }
    else
    {
        y0    = (y);
        incy0 = ( inc_t )(*incy);
    }


    // This function is invoked on all architectures including ‘generic’.
    // Non-AVX platforms will use the kernels derived from the context.
    if (bli_cpuid_is_avx_supported() == TRUE) {
	    bli_dswapv_zen_int8
		    (
		     n0,
		     x0, incx0,
		     y0, incy0,
		     NULL
		    );
    }
    else{
	    PASTEMAC2(d,swapv,BLIS_TAPI_EX_SUF) \
		    ( \
		      n0, \
		      x0, incx0, \
		      y0, incy0, \
		      NULL, \
		      NULL  \
		    ); \
    }

    /* Finalize BLIS. */
//    bli_finalize_auto();
    AOCL_DTL_TRACE_EXIT(AOCL_DTL_LEVEL_TRACE_1)
}

INSERT_GENTFUNC_BLAS_CZ( swap, swapv )


#endif
