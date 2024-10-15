/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2024, Advanced Micro Devices, Inc. All rights reserved.

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
void PASTEF77S(ch,blasname) \
     ( \
       const f77_int* n, \
       const ftype*   alpha, \
       const ftype*   x, const f77_int* incx, \
       const ftype*   beta, \
             ftype*   y, const f77_int* incy  \
     ) \
{ \
    AOCL_DTL_TRACE_ENTRY(AOCL_DTL_LEVEL_TRACE_1) \
    AOCL_DTL_LOG_AXPBY_INPUTS(AOCL_DTL_LEVEL_TRACE_1, *MKSTR(ch), *n, (void*)alpha, *incx, (void*)beta, *incy) \
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
      BLIS_NO_CONJUGATE, \
      n0, \
      (ftype*)alpha, \
      x0, incx0, \
      (ftype*)beta,  \
      y0, incy0, \
      NULL, \
      NULL  \
    ); \
\
    AOCL_DTL_TRACE_EXIT(AOCL_DTL_LEVEL_TRACE_1) \
    /* Finalize BLIS. */ \
    bli_finalize_auto(); \
}\
\
IF_BLIS_ENABLE_BLAS(\
void PASTEF77(ch,blasname) \
     ( \
       const f77_int* n, \
       const ftype*   alpha, \
       const ftype*   x, const f77_int* incx, \
       const ftype*   beta, \
             ftype*   y, const f77_int* incy  \
     ) \
{ \
  PASTEF77S(ch,blasname) \
     ( n, alpha, x, incx, beta, y, incy ); \
} \
)

void saxpby_blis_impl
(
 const f77_int* n,
 const float*   alpha,
 const float*   x, const f77_int* incx,
 const float*   beta,
 float*   y, const f77_int* incy
)
{
  AOCL_DTL_TRACE_ENTRY(AOCL_DTL_LEVEL_TRACE_1)
  AOCL_DTL_LOG_AXPY_INPUTS(AOCL_DTL_LEVEL_TRACE_1, 'S', *n, (float *)alpha, *incx, *incy)

  /* Early exit in case n is 0, or alpha is 0 and beta is 1 */
  if ( ( *n <= 0 ) ||
     ( PASTEMAC( s, eq0 )( *alpha ) && PASTEMAC( s, eq1 )( *beta ) ) )
  {
    AOCL_DTL_TRACE_EXIT(AOCL_DTL_LEVEL_TRACE_1)
    return;
  }

    dim_t n0;
    float *x0;
    float *y0;
    inc_t incx0;
    inc_t incy0;

    /* Initialize BLIS. */
    //    bli_init_auto();

    n0 = ( dim_t )( *n );

    /*
      If the input increments are negative, adjust the pointers so we can
      use positive increments instead.
    */
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

      x0 = ( ( float * )x ) + ( n0 - 1 ) * ( -( *incx ) );
      incx0 = ( inc_t )( *incx );
    }
    else
    {
      x0    = ( ( float* )x );
      incx0 = ( inc_t )( *incx );
    }
    if ( *incy < 0 )
    {
      y0    = ( ( float* )y ) + ( n0 - 1 ) * ( -( *incy ) );
      incy0 = ( inc_t )(*incy);
    }
    else
    {
      y0    = ( ( float* )y );
      incy0 = ( inc_t )( *incy );
    }

    cntx_t *cntx = NULL;

    // Query the architecture ID
    arch_t id = bli_arch_query_id();

    /*
      Function pointer declaration for the function
      that will be used by this API
    */
    saxpbyv_ker_ft axpbyv_ker_ptr; // DAXPBYV

    // Pick the kernel based on the architecture ID
    switch (id)
    {
      case BLIS_ARCH_ZEN5:
      case BLIS_ARCH_ZEN4:
      case BLIS_ARCH_ZEN:
      case BLIS_ARCH_ZEN2:
      case BLIS_ARCH_ZEN3:
        axpbyv_ker_ptr = bli_saxpbyv_zen_int10;

        break;
      default:

        // For non-Zen architectures, query the context
        cntx = bli_gks_query_cntx();

        // Query the context for the kernel function pointers for saxpbyv
        axpbyv_ker_ptr = bli_cntx_get_l1v_ker_dt(BLIS_FLOAT, BLIS_AXPBYV_KER, cntx);
    }

    // Call the function based on the function pointer assigned above
    axpbyv_ker_ptr
    (
      BLIS_NO_CONJUGATE,
      n0,
      (float *)alpha,
      x0, incx0,
      (float *)beta,
      y0, incy0,
      cntx
    );

    /* Finalize BLIS. */
    //    bli_finalize_auto();
    AOCL_DTL_TRACE_EXIT(AOCL_DTL_LEVEL_TRACE_1);
}

#ifdef BLIS_ENABLE_BLAS
void saxpby_
(
 const f77_int* n,
 const float*   alpha,
 const float*   x, const f77_int* incx,
 const float*   beta,
 float*   y, const f77_int* incy
)
{
  saxpby_blis_impl( n, alpha, x, incx, beta, y, incy ) ;
}
#endif

//-------------------------------------------------------------------------

void daxpby_blis_impl
(
 const f77_int* n,
 const double*   alpha,
 const double*   x, const f77_int* incx,
 const double*   beta,
 double*   y, const f77_int* incy
)
{
  AOCL_DTL_TRACE_ENTRY(AOCL_DTL_LEVEL_TRACE_1)
  AOCL_DTL_LOG_AXPY_INPUTS(AOCL_DTL_LEVEL_TRACE_1, 'D', *n, (double *)alpha, *incx, *incy)

  /* Early exit in case n is 0, or alpha is 0 and beta is 1 */
  if ( ( *n <= 0 ) ||
     ( PASTEMAC( d, eq0 )( *alpha ) && PASTEMAC( d, eq1 )( *beta ) ) )
  {
    AOCL_DTL_TRACE_EXIT(AOCL_DTL_LEVEL_TRACE_1)
    return;
  }

    dim_t n0;
    double *x0;
    double *y0;
    inc_t incx0;
    inc_t incy0;

    /* Initialize BLIS. */
    //    bli_init_auto();

    n0 = ( dim_t )( *n );

    /*
      If the input increments are negative, adjust the pointers so we can
      use positive increments instead.
    */
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

      x0 = ( ( double * )x ) + ( n0 - 1 ) * ( -( *incx ) );
      incx0 = ( inc_t )( *incx );
    }
    else
    {
      x0    = ( ( double* )x );
      incx0 = ( inc_t )( *incx );
    }
    if ( *incy < 0 )
    {
      y0    = ( ( double* )y ) + ( n0 - 1 ) * ( -( *incy ) );
      incy0 = ( inc_t )(*incy);
    }
    else
    {
      y0    = ( ( double* )y );
      incy0 = ( inc_t )( *incy );
    }

    cntx_t *cntx = NULL;

    // Query the architecture ID
    arch_t id = bli_arch_query_id();

    /*
      Function pointer declarations for the function
      that will be used by this API
    */
    daxpbyv_ker_ft axpbyv_ker_ptr; // DAXPBYV

    // Pick the kernel based on the architecture ID
    switch (id)
    {
      case BLIS_ARCH_ZEN5:
      case BLIS_ARCH_ZEN4:
#if defined(BLIS_KERNELS_ZEN4)
        axpbyv_ker_ptr = bli_daxpbyv_zen_int_avx512;

        break;
#endif
      case BLIS_ARCH_ZEN:
      case BLIS_ARCH_ZEN2:
      case BLIS_ARCH_ZEN3:
        axpbyv_ker_ptr = bli_daxpbyv_zen_int10;

        break;
      default:

        // For non-Zen architectures, query the context
        cntx = bli_gks_query_cntx();

        // Query the context for the kernel function pointers for daxpbyv
        axpbyv_ker_ptr = bli_cntx_get_l1v_ker_dt(BLIS_DOUBLE, BLIS_AXPBYV_KER, cntx);
    }

    // Call the function based on the function pointer assigned above
    axpbyv_ker_ptr
    (
      BLIS_NO_CONJUGATE,
      n0,
      (double *)alpha,
      x0, incx0,
      (double *)beta,
      y0, incy0,
      cntx
    );

    /* Finalize BLIS. */
    //    bli_finalize_auto();
    AOCL_DTL_TRACE_EXIT(AOCL_DTL_LEVEL_TRACE_1);
}

#ifdef BLIS_ENABLE_BLAS
void daxpby_
(
 const f77_int* n,
 const double*   alpha,
 const double*   x, const f77_int* incx,
 const double*   beta,
 double*   y, const f77_int* incy
)
{
  daxpby_blis_impl( n, alpha, x, incx, beta, y, incy ) ;
}
#endif

//-------------------------------------------------------------------------

void caxpby_blis_impl
(
 const f77_int*    n,
 const scomplex*   alpha,
 const scomplex*   x, const f77_int* incx,
 const scomplex*   beta,
 scomplex*         y, const f77_int* incy
)
{
  AOCL_DTL_TRACE_ENTRY(AOCL_DTL_LEVEL_TRACE_1)
  AOCL_DTL_LOG_AXPY_INPUTS(AOCL_DTL_LEVEL_TRACE_1, 'C', *n, (scomplex *)alpha, *incx, *incy)

  /* Early exit in case n is 0, or alpha is 0 and beta is 1 */
  if ( ( *n <= 0 ) ||
     ( PASTEMAC( c, eq0 )( *alpha ) && PASTEMAC( c, eq1 )( *beta ) ) )
  {
    AOCL_DTL_TRACE_EXIT(AOCL_DTL_LEVEL_TRACE_1)
    return;
  }

    dim_t n0;
    scomplex *x0;
    scomplex *y0;
    inc_t incx0;
    inc_t incy0;

    /* Initialize BLIS. */
    //    bli_init_auto();

    n0 = ( dim_t )( *n );

    /*
      If the input increments are negative, adjust the pointers so we can
      use positive increments instead.
    */
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

      x0 = ( ( scomplex * )x ) + ( n0 - 1 ) * ( -( *incx ) );
      incx0 = ( inc_t )( *incx );
    }
    else
    {
      x0    = ( ( scomplex* )x );
      incx0 = ( inc_t )( *incx );
    }
    if ( *incy < 0 )
    {
      y0    = ( ( scomplex* )y ) + ( n0 - 1 ) * ( -( *incy ) );
      incy0 = ( inc_t )(*incy);
    }
    else
    {
      y0    = ( ( scomplex* )y );
      incy0 = ( inc_t )( *incy );
    }

    cntx_t *cntx = NULL;

    // Query the architecture ID
    arch_t id = bli_arch_query_id();

    /*
      Function pointer declarations for the function
      that will be used by this API
    */
    caxpbyv_ker_ft axpbyv_ker_ptr; // caxpbyV

    // Pick the kernel based on the architecture ID
    switch (id)
    {
      case BLIS_ARCH_ZEN5:
      case BLIS_ARCH_ZEN4:
      case BLIS_ARCH_ZEN:
      case BLIS_ARCH_ZEN2:
      case BLIS_ARCH_ZEN3:
        axpbyv_ker_ptr = bli_caxpbyv_zen_int;

        break;
      default:

        // For non-Zen architectures, query the context
        cntx = bli_gks_query_cntx();

        // Query the context for the kernel function pointers for caxpbyv
        axpbyv_ker_ptr = bli_cntx_get_l1v_ker_dt(BLIS_SCOMPLEX, BLIS_AXPBYV_KER, cntx);
    }

    // Call the function based on the function pointer assigned above
    axpbyv_ker_ptr
    (
      BLIS_NO_CONJUGATE,
      n0,
      (scomplex *)alpha,
      x0, incx0,
      (scomplex *)beta,
      y0, incy0,
      cntx
    );

    /* Finalize BLIS. */
    //    bli_finalize_auto();
    AOCL_DTL_TRACE_EXIT(AOCL_DTL_LEVEL_TRACE_1);
}

#ifdef BLIS_ENABLE_BLAS
void caxpby_
(
 const f77_int* n,
 const scomplex*   alpha,
 const scomplex*   x, const f77_int* incx,
 const scomplex*   beta,
 scomplex*   y, const f77_int* incy
)
{
  caxpby_blis_impl( n, alpha, x, incx, beta, y, incy ) ;
}
#endif

//-------------------------------------------------------------------------

void zaxpby_blis_impl
(
 const f77_int*    n,
 const dcomplex*   alpha,
 const dcomplex*   x, const f77_int* incx,
 const dcomplex*   beta,
 dcomplex*         y, const f77_int* incy
)
{
  AOCL_DTL_TRACE_ENTRY(AOCL_DTL_LEVEL_TRACE_1)
  AOCL_DTL_LOG_AXPY_INPUTS(AOCL_DTL_LEVEL_TRACE_1, 'Z', *n, (dcomplex *)alpha, *incx, *incy)

  /* Early exit in case n is 0, or alpha is 0 and beta is 1 */
  if ( ( *n <= 0 ) ||
     ( PASTEMAC( c, eq0 )( *alpha ) && PASTEMAC( c, eq1 )( *beta ) ) )
  {
    AOCL_DTL_TRACE_EXIT(AOCL_DTL_LEVEL_TRACE_1)
    return;
  }

    dim_t n0;
    dcomplex *x0;
    dcomplex *y0;
    inc_t incx0;
    inc_t incy0;

    /* Initialize BLIS. */
    //    bli_init_auto();

    n0 = ( dim_t )( *n );

    /*
      If the input increments are negative, adjust the pointers so we can
      use positive increments instead.
    */
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

      x0 = ( ( dcomplex * )x ) + ( n0 - 1 ) * ( -( *incx ) );
      incx0 = ( inc_t )( *incx );
    }
    else
    {
      x0    = ( ( dcomplex* )x );
      incx0 = ( inc_t )( *incx );
    }
    if ( *incy < 0 )
    {
      y0    = ( ( dcomplex* )y ) + ( n0 - 1 ) * ( -( *incy ) );
      incy0 = ( inc_t )(*incy);
    }
    else
    {
      y0    = ( ( dcomplex* )y );
      incy0 = ( inc_t )( *incy );
    }

    cntx_t *cntx = NULL;

    // Query the architecture ID
    arch_t id = bli_arch_query_id();

    /*
      Function pointer declarations for the function
      that will be used by this API
    */
    zaxpbyv_ker_ft axpbyv_ker_ptr; // zaxpbyV

    // Pick the kernel based on the architecture ID
    switch (id)
    {
      case BLIS_ARCH_ZEN5:
      case BLIS_ARCH_ZEN4:
      case BLIS_ARCH_ZEN:
      case BLIS_ARCH_ZEN2:
      case BLIS_ARCH_ZEN3:
        axpbyv_ker_ptr = bli_zaxpbyv_zen_int;

        break;
      default:

        // For non-Zen architectures, query the context
        cntx = bli_gks_query_cntx();

        // Query the context for the kernel function pointers for zaxpbyv
        axpbyv_ker_ptr = bli_cntx_get_l1v_ker_dt(BLIS_DCOMPLEX, BLIS_AXPBYV_KER, cntx);
    }

    // Call the function based on the function pointer assigned above
    axpbyv_ker_ptr
    (
      BLIS_NO_CONJUGATE,
      n0,
      (dcomplex *)alpha,
      x0, incx0,
      (dcomplex *)beta,
      y0, incy0,
      cntx
    );

    /* Finalize BLIS. */
    //    bli_finalize_auto();
    AOCL_DTL_TRACE_EXIT(AOCL_DTL_LEVEL_TRACE_1);
}

#ifdef BLIS_ENABLE_BLAS
void zaxpby_
(
 const f77_int* n,
 const dcomplex*   alpha,
 const dcomplex*   x, const f77_int* incx,
 const dcomplex*   beta,
 dcomplex*   y, const f77_int* incy
)
{
  zaxpby_blis_impl( n, alpha, x, incx, beta, y, incy ) ;
}
#endif
