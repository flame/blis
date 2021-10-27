/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2014, The University of Texas at Austin
   Copyright (C) 2020 - 21, Advanced Micro Devices, Inc. All rights reserved.

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
       const ftype*   alpha, \
       const ftype*   x, const f77_int* incx, \
             ftype*   y, const f77_int* incy  \
     ) \
{ \
    dim_t  n0; \
    ftype* x0; \
    ftype* y0; \
    inc_t  incx0; \
    inc_t  incy0; \
\
    /* Initialize BLIS. */ \
    bli_init_auto(); \
    AOCL_DTL_TRACE_ENTRY(AOCL_DTL_LEVEL_TRACE_1) \
    AOCL_DTL_LOG_AXPY_INPUTS(AOCL_DTL_LEVEL_TRACE_1, *MKSTR(ch), *n, (void*)alpha, *incx, *incy) \
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
      y0, incy0, \
      NULL, \
      NULL  \
    ); \
\
     AOCL_DTL_TRACE_EXIT(AOCL_DTL_LEVEL_TRACE_1) \
        /* Finalize BLIS. */ \
     bli_finalize_auto();  \
}

#ifdef BLIS_ENABLE_BLAS

#ifdef BLIS_CONFIG_EPYC
void saxpy_
(
 const f77_int* n,
 const float*   alpha,
 const float*   x, const f77_int* incx,
 float*   y, const f77_int* incy
 )
{
  AOCL_DTL_TRACE_ENTRY(AOCL_DTL_LEVEL_TRACE_1)
  AOCL_DTL_LOG_AXPY_INPUTS(AOCL_DTL_LEVEL_TRACE_1, 'S', *n, (float*)alpha, *incx, *incy)
  dim_t  n0;
  float* x0;
  float* y0;
  inc_t  incx0;
  inc_t  incy0;

  /* Initialize BLIS. */
  //    bli_init_auto();

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

  // When dynamic dispatch is enabled i.e. library is built for ‘amdzen’ configuration.
  // This function is invoked on all architectures including ‘generic’.
  // Invoke architecture specific kernels only if we are sure that we are running on zen,
  // zen2 or zen3 otherwise fall back to reference kernels (via framework and context).
  arch_t id = bli_arch_query_id();
  bool bamdzen = (id == BLIS_ARCH_ZEN3) || (id == BLIS_ARCH_ZEN2) || (id == BLIS_ARCH_ZEN);

  if (bamdzen)
  {
      bli_saxpyv_zen_int10
      (
        BLIS_NO_CONJUGATE,
        n0,
        (float*)alpha,
        x0, incx0,
        y0, incy0,
        NULL
      );

  }
  else
  {
      PASTEMAC2(s,axpyv,BLIS_TAPI_EX_SUF)
      (
        BLIS_NO_CONJUGATE,
        n0,
        (float*)alpha,
        x0, incx0,
        y0, incy0,
        NULL,
        NULL
      );

  }
  /* Finalize BLIS. */
  //    bli_finalize_auto();
  AOCL_DTL_TRACE_EXIT(AOCL_DTL_LEVEL_TRACE_1);
}

void daxpy_
(
 const f77_int* n,
 const double*   alpha,
 const double*   x, const f77_int* incx,
 double*   y, const f77_int* incy
 )
{
  dim_t  n0;
  double* x0;
  double* y0;
  inc_t  incx0;
  inc_t  incy0;

  AOCL_DTL_TRACE_ENTRY(AOCL_DTL_LEVEL_TRACE_1)
  AOCL_DTL_LOG_AXPY_INPUTS(AOCL_DTL_LEVEL_TRACE_1, 'D', *n, (double*)alpha, *incx, *incy)
  /* Initialize BLIS. */
  //    bli_init_auto();

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

  // When dynamic dispatch is enabled i.e. library is built for ‘amdzen’ configuration.
  // This function is invoked on all architectures including ‘generic’.
  // Invoke architecture specific kernels only if we are sure that we are running on zen,
  // zen2 or zen3 otherwise fall back to reference kernels (via framework and context).
  arch_t id = bli_arch_query_id();
  bool bamdzen = (id == BLIS_ARCH_ZEN3) || (id == BLIS_ARCH_ZEN2) || (id == BLIS_ARCH_ZEN);

  if (bamdzen)
  {
      bli_daxpyv_zen_int10
      (
        BLIS_NO_CONJUGATE,
        n0,
        (double*)alpha,
        x0, incx0,
        y0, incy0,
        NULL
      );

  }
  else
  {
      PASTEMAC2(d,axpyv,BLIS_TAPI_EX_SUF)
      (
        BLIS_NO_CONJUGATE,
        n0,
        (double*)alpha,
        x0, incx0,
        y0, incy0,
        NULL,
        NULL
      );

  }

  AOCL_DTL_TRACE_EXIT(AOCL_DTL_LEVEL_TRACE_1);
  /* Finalize BLIS. */
  //    bli_finalize_auto();
}

void caxpy_
(
 const f77_int* n,
 const scomplex*   alpha,
 const scomplex*   x, const f77_int* incx,
 scomplex*   y, const f77_int* incy
 )
{
  dim_t     n0;
  scomplex* x0;
  scomplex* y0;
  inc_t  incx0;
  inc_t  incy0;

  AOCL_DTL_TRACE_ENTRY(AOCL_DTL_LEVEL_TRACE_1)
  AOCL_DTL_LOG_AXPY_INPUTS(AOCL_DTL_LEVEL_TRACE_1, 'C', *n, (scomplex*)alpha, *incx, *incy)

  /* Initialize BLIS. */
  //    bli_init_auto();
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

  // When dynamic dispatch is enabled i.e. library is built for ‘amdzen’ configuration.
  // This function is invoked on all architectures including ‘generic’.
  // Invoke architecture specific kernels only if we are sure that we are running on zen,
  // zen2 or zen3 otherwise fall back to reference kernels (via framework and context).
  arch_t id = bli_arch_query_id();
  bool bamdzen = (id == BLIS_ARCH_ZEN3) || (id == BLIS_ARCH_ZEN2) || (id == BLIS_ARCH_ZEN);

  if (bamdzen)
  {
      bli_caxpyv_zen_int5
      (
        BLIS_NO_CONJUGATE,
        n0,
        (scomplex*)alpha,
        x0, incx0,
        y0, incy0,
        NULL
      );

  }
  else
  {
      PASTEMAC2(c,axpyv,BLIS_TAPI_EX_SUF)
      (
        BLIS_NO_CONJUGATE,
        n0,
        (scomplex*)alpha,
        x0, incx0,
        y0, incy0,
        NULL,
        NULL
      );
  }

  AOCL_DTL_TRACE_EXIT(AOCL_DTL_LEVEL_TRACE_1);
  /* Finalize BLIS. */
  //    bli_finalize_auto();
}

void zaxpy_
(
 const f77_int* n,
 const dcomplex*   alpha,
 const dcomplex*   x, const f77_int* incx,
 dcomplex*   y, const f77_int* incy
 )
{
  dim_t  n0;
  dcomplex* x0;
  dcomplex* y0;
  inc_t  incx0;
  inc_t  incy0;

  AOCL_DTL_TRACE_ENTRY(AOCL_DTL_LEVEL_TRACE_1)
  AOCL_DTL_LOG_AXPY_INPUTS(AOCL_DTL_LEVEL_TRACE_1, 'Z', *n, (dcomplex*)alpha, *incx, *incy)

  /* Initialize BLIS. */
  //    bli_init_auto();

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

  // When dynamic dispatch is enabled i.e. library is built for ‘amdzen’ configuration.
  // This function is invoked on all architectures including ‘generic’.
  // Invoke architecture specific kernels only if we are sure that we are running on zen,
  // zen2 or zen3 otherwise fall back to reference kernels (via framework and context).
  arch_t id = bli_arch_query_id();
  bool bamdzen = (id == BLIS_ARCH_ZEN3) || (id == BLIS_ARCH_ZEN2) || (id == BLIS_ARCH_ZEN);

  if (bamdzen)
  {
      bli_zaxpyv_zen_int5
      (
        BLIS_NO_CONJUGATE,
        n0,
        (dcomplex*)alpha,
        x0, incx0,
        y0, incy0,
        NULL
      );

  }
  else
  {
      PASTEMAC2(z,axpyv,BLIS_TAPI_EX_SUF)
      (
        BLIS_NO_CONJUGATE,
        n0,
        (dcomplex*)alpha,
        x0, incx0,
        y0, incy0,
        NULL,
        NULL
      );
  }

  AOCL_DTL_TRACE_EXIT(AOCL_DTL_LEVEL_TRACE_1);
  /* Finalize BLIS. */
  //    bli_finalize_auto();
}

#else
INSERT_GENTFUNC_BLAS( axpy, axpyv )
#endif

#endif
