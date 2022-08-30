/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2014, The University of Texas at Austin
   Copyright (C) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

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
#undef  GENTFUNCDOT
#define GENTFUNCDOT( ftype, ch, chc, blis_conjx, blasname, blisname ) \
\
ftype PASTEF772(ch,blasname,chc) \
     ( \
       const f77_int* n, \
       const ftype*   x, const f77_int* incx, \
       const ftype*   y, const f77_int* incy  \
     ) \
{ \
    AOCL_DTL_TRACE_ENTRY(AOCL_DTL_LEVEL_TRACE_1); \
    AOCL_DTL_LOG_DOTV_INPUTS(AOCL_DTL_LEVEL_TRACE_1, *MKSTR(ch), *n, *incx, *incy); \
    dim_t  n0; \
    ftype* x0; \
    ftype* y0; \
    inc_t  incx0; \
    inc_t  incy0; \
    ftype  rho; \
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
      blis_conjx, \
      BLIS_NO_CONJUGATE, \
      n0, \
      x0, incx0, \
      y0, incy0, \
      &rho, \
      NULL, \
      NULL  \
    ); \
\
    AOCL_DTL_TRACE_EXIT(AOCL_DTL_LEVEL_TRACE_1); \
    /* Finalize BLIS. */ \
    bli_finalize_auto(); \
\
    return rho; \
}

#ifdef BLIS_ENABLE_BLAS
float sdot_
     (
       const f77_int* n,
       const float*   x, const f77_int* incx,
       const float*   y, const f77_int* incy
     )
{
    AOCL_DTL_TRACE_ENTRY(AOCL_DTL_LEVEL_TRACE_1);
    AOCL_DTL_LOG_DOTV_INPUTS(AOCL_DTL_LEVEL_TRACE_1, 'S', *n, *incx, *incy);
    dim_t  n0;
    float* x0;
    float* y0;
    inc_t  incx0;
    inc_t  incy0;
    float  rho;

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

        x0    = ((float*)x) + (n0-1)*(-*incx);
        incx0 = ( inc_t )(*incx);

    }
    else
    {
        x0    = ((float*)x);
        incx0 = ( inc_t )(*incx);
    }

    if ( *incy < 0 )
    {
        y0    = ((float*)y) + (n0-1)*(-*incy);
        incy0 = ( inc_t )(*incy);

    }
    else
    {
        y0    = ((float*)y);
        incy0 = ( inc_t )(*incy);
    }

    // This function is invoked on all architectures including ‘generic’.
    // Non-AVX platforms will use the kernels derived from the context.
    if (bli_cpuid_is_avx_supported() == TRUE)
    {
        /* Call BLIS kernel. */
        bli_sdotv_zen_int10
        (
        BLIS_NO_CONJUGATE,
        BLIS_NO_CONJUGATE,
        n0,
        x0, incx0,
        y0, incy0,
        &rho,
        NULL
        );
    }
    else
    {
        /* Call BLIS interface. */
        PASTEMAC2(s,dotv,BLIS_TAPI_EX_SUF)
        (
        BLIS_NO_CONJUGATE,
        BLIS_NO_CONJUGATE,
        n0,
        x0, incx0,
        y0, incy0,
        &rho,
        NULL,
        NULL
        );
    }

    /* Finalize BLIS. */
//  bli_finalize_auto();
    AOCL_DTL_TRACE_EXIT(AOCL_DTL_LEVEL_TRACE_1);
    return rho;
}

double ddot_
     (
       const f77_int* n,
       const double*   x, const f77_int* incx,
       const double*   y, const f77_int* incy
     )
{
    AOCL_DTL_TRACE_ENTRY(AOCL_DTL_LEVEL_TRACE_1);
    AOCL_DTL_LOG_DOTV_INPUTS(AOCL_DTL_LEVEL_TRACE_1, 'D', *n, *incx, *incy);
    dim_t  n0;
    double* x0;
    double* y0;
    inc_t  incx0;
    inc_t  incy0;
    double  rho;

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

        x0    = ((double*)x) + (n0-1)*(-*incx);
        incx0 = ( inc_t )(*incx);

    }
    else
    {
        x0    = ((double*)x);
        incx0 = ( inc_t )(*incx);
    }

    if ( *incy < 0 )
    {
        y0    = ((double*)y) + (n0-1)*(-*incy);
        incy0 = ( inc_t )(*incy);

    }
    else
    {
        y0    = ((double*)y);
        incy0 = ( inc_t )(*incy);
    }

    // This function is invoked on all architectures including ‘generic’.
    // Non-AVX platforms will use the kernels derived from the context.
    if (bli_cpuid_is_avx_supported() == TRUE)
    {
        /* Call BLIS kernel. */
        bli_ddotv_zen_int10
        (
        BLIS_NO_CONJUGATE,
        BLIS_NO_CONJUGATE,
        n0,
        x0, incx0,
        y0, incy0,
        &rho,
        NULL
        );
    }
    else
    {
        /* Call BLIS interface. */
        PASTEMAC2(d,dotv,BLIS_TAPI_EX_SUF)
        (
        BLIS_NO_CONJUGATE,
        BLIS_NO_CONJUGATE,
        n0,
        x0, incx0,
        y0, incy0,
        &rho,
        NULL,
        NULL
        );
    }

    /* Finalize BLIS. */
//  bli_finalize_auto();
    AOCL_DTL_TRACE_EXIT(AOCL_DTL_LEVEL_TRACE_1);
    return rho;
}

#ifdef BLIS_DISABLE_COMPLEX_RETURN_INTEL
scomplex cdotu_
     (
       const f77_int* n,
       const scomplex*   x, const f77_int* incx,
       const scomplex*   y, const f77_int* incy
     )
{
    AOCL_DTL_TRACE_ENTRY(AOCL_DTL_LEVEL_TRACE_1);
    AOCL_DTL_LOG_DOTV_INPUTS(AOCL_DTL_LEVEL_TRACE_1, 'C', *n, *incx, *incy);
    dim_t  n0;
    scomplex* x0;
    scomplex* y0;
    inc_t  incx0;
    inc_t  incy0;
    scomplex  rho;

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

        x0    = ((scomplex*)x) + (n0-1)*(-*incx);
        incx0 = ( inc_t )(*incx);

    }
    else
    {
        x0    = ((scomplex*)x);
        incx0 = ( inc_t )(*incx);
    }

    if ( *incy < 0 )
    {
        y0    = ((scomplex*)y) + (n0-1)*(-*incy);
        incy0 = ( inc_t )(*incy);

    }
    else
    {
        y0    = ((scomplex*)y);
        incy0 = ( inc_t )(*incy);
    }

    // This function is invoked on all architectures including ‘generic’.
    // Non-AVX platforms will use the kernels derived from the context.
    if (bli_cpuid_is_avx_supported() == TRUE)
    {
        /* Call BLIS kernel. */
        bli_cdotv_zen_int5
        (
        BLIS_NO_CONJUGATE,
        BLIS_NO_CONJUGATE,
        n0,
        x0, incx0,
        y0, incy0,
        &rho,
        NULL
        );
    }
    else
    {
        /* Call BLIS interface. */
        PASTEMAC2(c,dotv,BLIS_TAPI_EX_SUF)
        (
        BLIS_NO_CONJUGATE,
        BLIS_NO_CONJUGATE,
        n0,
        x0, incx0,
        y0, incy0,
        &rho,
        NULL,
        NULL
        );
    }

    /* Finalize BLIS. */
//  bli_finalize_auto();
    AOCL_DTL_TRACE_EXIT(AOCL_DTL_LEVEL_TRACE_1);
    return rho;
}

dcomplex zdotu_
     (
       const f77_int* n,
       const dcomplex*   x, const f77_int* incx,
       const dcomplex*   y, const f77_int* incy
     )
{
    dim_t  n0;
    dcomplex* x0;
    dcomplex* y0;
    inc_t  incx0;
    inc_t  incy0;
    dcomplex  rho;

    AOCL_DTL_TRACE_ENTRY(AOCL_DTL_LEVEL_TRACE_1);
    AOCL_DTL_LOG_DOTV_INPUTS(AOCL_DTL_LEVEL_TRACE_1, 'Z', *n, *incx, *incy);

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

        x0    = ((dcomplex*)x) + (n0-1)*(-*incx);
        incx0 = ( inc_t )(*incx);

    }
    else
    {
        x0    = ((dcomplex*)x);
        incx0 = ( inc_t )(*incx);
    }

    if ( *incy < 0 )
    {
        y0    = ((dcomplex*)y) + (n0-1)*(-*incy);
        incy0 = ( inc_t )(*incy);

    }
    else
    {
        y0    = ((dcomplex*)y);
        incy0 = ( inc_t )(*incy);
    }

    // This function is invoked on all architectures including ‘generic’.
    // Non-AVX platforms will use the kernels derived from the context.
    if (bli_cpuid_is_avx_supported() == TRUE)
    {
        /* Call BLIS kernel. */
        bli_zdotv_zen_int5
        (
        BLIS_NO_CONJUGATE,
        BLIS_NO_CONJUGATE,
        n0,
        x0, incx0,
        y0, incy0,
        &rho,
        NULL
        );
    }
    else
    {
        /* Call BLIS interface. */
        PASTEMAC2(z,dotv,BLIS_TAPI_EX_SUF)
        (
        BLIS_NO_CONJUGATE,
        BLIS_NO_CONJUGATE,
        n0,
        x0, incx0,
        y0, incy0,
        &rho,
        NULL,
        NULL
        );
    }

    /* Finalize BLIS. */
//  bli_finalize_auto();

    AOCL_DTL_TRACE_EXIT(AOCL_DTL_LEVEL_TRACE_1);

    return rho;
}


scomplex cdotc_
     (
       const f77_int* n,
       const scomplex*   x, const f77_int* incx,
       const scomplex*   y, const f77_int* incy
     )
{
    dim_t  n0;
    scomplex* x0;
    scomplex* y0;
    inc_t  incx0;
    inc_t  incy0;
    scomplex  rho;

    AOCL_DTL_TRACE_ENTRY(AOCL_DTL_LEVEL_TRACE_1);
    AOCL_DTL_LOG_DOTV_INPUTS(AOCL_DTL_LEVEL_TRACE_1, 'C', *n, *incx, *incy);

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

        x0    = ((scomplex*)x) + (n0-1)*(-*incx);
        incx0 = ( inc_t )(*incx);

    }
    else
    {
        x0    = ((scomplex*)x);
        incx0 = ( inc_t )(*incx);
    }

    if ( *incy < 0 )
    {
        y0    = ((scomplex*)y) + (n0-1)*(-*incy);
        incy0 = ( inc_t )(*incy);

    }
    else
    {
        y0    = ((scomplex*)y);
        incy0 = ( inc_t )(*incy);
    }

    // This function is invoked on all architectures including ‘generic’.
    // Non-AVX platforms will use the kernels derived from the context.
    if (bli_cpuid_is_avx_supported() == TRUE)
    {
        /* Call BLIS kernel. */
        bli_cdotv_zen_int5
        (
        BLIS_CONJUGATE,
        BLIS_NO_CONJUGATE,
        n0,
        x0, incx0,
        y0, incy0,
        &rho,
        NULL
        );
    }
    else
    {
        /* Call BLIS interface. */
        PASTEMAC2(c,dotv,BLIS_TAPI_EX_SUF)
        (
        BLIS_CONJUGATE,
        BLIS_NO_CONJUGATE,
        n0,
        x0, incx0,
        y0, incy0,
        &rho,
        NULL,
        NULL
        );
    }

    /* Finalize BLIS. */
//  bli_finalize_auto();
    AOCL_DTL_TRACE_EXIT(AOCL_DTL_LEVEL_TRACE_1);

    return rho;
}

dcomplex zdotc_
     (
       const f77_int* n,
       const dcomplex*   x, const f77_int* incx,
       const dcomplex*   y, const f77_int* incy
     )
{
    AOCL_DTL_TRACE_ENTRY(AOCL_DTL_LEVEL_TRACE_1);
    AOCL_DTL_LOG_DOTV_INPUTS(AOCL_DTL_LEVEL_TRACE_1, 'Z', *n, *incx, *incy);
    dim_t  n0;
    dcomplex* x0;
    dcomplex* y0;
    inc_t  incx0;
    inc_t  incy0;
    dcomplex  rho;

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

        x0    = ((dcomplex*)x) + (n0-1)*(-*incx);
        incx0 = ( inc_t )(*incx);

    }
    else
    {
        x0    = ((dcomplex*)x);
        incx0 = ( inc_t )(*incx);
    }

    if ( *incy < 0 )
    {
        y0    = ((dcomplex*)y) + (n0-1)*(-*incy);
        incy0 = ( inc_t )(*incy);

    }
    else
    {
        y0    = ((dcomplex*)y);
        incy0 = ( inc_t )(*incy);
    }

    // This function is invoked on all architectures including ‘generic’.
    // Non-AVX platforms will use the kernels derived from the context.
    if (bli_cpuid_is_avx_supported() == TRUE)
    {
        /* Call BLIS kernel. */
        bli_zdotv_zen_int5
        (
        BLIS_CONJUGATE,
        BLIS_NO_CONJUGATE,
        n0,
        x0, incx0,
        y0, incy0,
        &rho,
        NULL
        );
    }
    else
    {
        /* Call BLIS interface. */
        PASTEMAC2(z,dotv,BLIS_TAPI_EX_SUF)
        (
        BLIS_CONJUGATE,
        BLIS_NO_CONJUGATE,
        n0,
        x0, incx0,
        y0, incy0,
        &rho,
        NULL,
        NULL
        );
    }





    /* Finalize BLIS. */
//  bli_finalize_auto();

    AOCL_DTL_TRACE_EXIT(AOCL_DTL_LEVEL_TRACE_1);

    return rho;
}

#else // BLIS_DISABLE_COMPLEX_RETURN_INTEL
// For the "intel" complex return type, use a hidden parameter to return the result
#undef  GENTFUNCDOT
#define GENTFUNCDOT( ftype, ch, chc, blis_conjx, blasname, blisname ) \
\
void PASTEF772(ch,blasname,chc) \
     ( \
       ftype*         rhop, \
       const f77_int* n, \
       const ftype*   x, const f77_int* incx, \
       const ftype*   y, const f77_int* incy  \
     ) \
{ \
  AOCL_DTL_TRACE_ENTRY(AOCL_DTL_LEVEL_TRACE_1); \
  AOCL_DTL_LOG_DOTV_INPUTS(AOCL_DTL_LEVEL_TRACE_1, *MKSTR(ch), *n, *incx, *incy); \
        dim_t  n0; \
        ftype* x0; \
        ftype* y0; \
        inc_t  incx0; \
        inc_t  incy0; \
        ftype  rho; \
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
          blis_conjx, \
          BLIS_NO_CONJUGATE, \
          n0, \
          x0, incx0, \
          y0, incy0, \
          &rho, \
          NULL, \
          NULL  \
        ); \
\
        /* Finalize BLIS. */ \
        AOCL_DTL_TRACE_EXIT(AOCL_DTL_LEVEL_TRACE_1); \
        bli_finalize_auto(); \
\
        *rhop = rho; \
}

INSERT_GENTFUNCDOTC_BLAS( dot, dotv )
#endif // BLIS_DISABLE_COMPLEX_RETURN_INTEL



// -- "Black sheep" dot product function definitions --

// Input vectors stored in single precision, computed in double precision,
// with result returned in single precision.
float PASTEF77(sd,sdot)
     (
       const f77_int* n,
       const float*   sb,
       const float*   x, const f77_int* incx,
       const float*   y, const f77_int* incy
     )
{
    return ( float )
           (
             ( double )(*sb) +
             PASTEF77(d,sdot)
             (
               n,
               x, incx,
               y, incy
             )
           );
}

// Input vectors stored in single precision, computed in double precision,
// with result returned in double precision.
double PASTEF77(d,sdot)
     (
       const f77_int* n,
       const float*   x, const f77_int* incx,
       const float*   y, const f77_int* incy
     )
{
    dim_t   n0;
    float*  x0;
    float*  y0;
    inc_t   incx0;
    inc_t   incy0;
    double  rho;
    dim_t   i;

    AOCL_DTL_TRACE_ENTRY(AOCL_DTL_LEVEL_TRACE_1);
    AOCL_DTL_LOG_DOTV_INPUTS(AOCL_DTL_LEVEL_TRACE_1, 'D', *n, *incx, *incy);
    /* Initialization of BLIS is not required. */

    /* Convert/typecast negative values of n to zero. */
    bli_convert_blas_dim1( *n, n0 );

    /* If the input increments are negative, adjust the pointers so we can
       use positive increments instead. */
    bli_convert_blas_incv( n0, (float*)x, *incx, x0, incx0 );
    bli_convert_blas_incv( n0, (float*)y, *incy, y0, incy0 );

    rho = 0.0;

    for ( i = 0; i < n0; i++ )
    {
        float* chi1 = x0 + (i  )*incx0;
        float* psi1 = y0 + (i  )*incy0;

        bli_ddots( (( double )(*chi1)),
                   (( double )(*psi1)), rho );
    }

    /* Finalization of BLIS is not required, because initialization was
       not required. */
    AOCL_DTL_TRACE_EXIT(AOCL_DTL_LEVEL_TRACE_1);

    return rho;
}

#endif
