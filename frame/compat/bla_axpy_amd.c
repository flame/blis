/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2014, The University of Texas at Austin
   Copyright (C) 2020-2023, Advanced Micro Devices, Inc. All rights reserved.

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

/*
  Early return conditions
  ------------------------

  1. When n <= 0 where n is the length of the vector passed
  2. When alpha == 0 where alpha is the scalar value by which the vector is
     to be scaled

  NaN propagation expectation
  --------------------------

  1. When alpha == NaN - Propogate the NaN to the vector
*/

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
}\
\
IF_BLIS_ENABLE_BLAS(\
void PASTEF77(ch,blasname) \
     ( \
       const f77_int* n, \
       const ftype*   alpha, \
       const ftype*   x, const f77_int* incx, \
             ftype*   y, const f77_int* incy  \
     ) \
{ \
  PASTEF77S(ch,blasname)( n, alpha, x, incx, y, incy ) ; \
} \
)



void saxpy_blis_impl
(
 const f77_int* n,
 const float*   alpha,
 const float*   x, const f77_int* incx,
 float*   y, const f77_int* incy
 )
{
    AOCL_DTL_TRACE_ENTRY(AOCL_DTL_LEVEL_TRACE_1)
    AOCL_DTL_LOG_AXPY_INPUTS(AOCL_DTL_LEVEL_TRACE_1, 'S', *n, (float *)alpha, *incx, *incy)

    /*
      BLAS exception: If the vector dimension is zero, or if alpha is zero, return early.
    */
    if ((*n) <= 0 || PASTEMAC(s, eq0)(*alpha))
    {
      AOCL_DTL_TRACE_EXIT(AOCL_DTL_LEVEL_TRACE_1);

      return;
    }

    dim_t n_elem;
    float *x0;
    float *y0;
    inc_t incx0;
    inc_t incy0;

    /* Initialize BLIS. */
    //    bli_init_auto();

    /* Convert/typecast negative values of n to zero. */
    if (*n < 0)
      n_elem = (dim_t)0;
    else
      n_elem = (dim_t)(*n);

    /*
      If the input increments are negative, adjust the pointers so we can
      use positive increments instead.
    */
    if (*incx < 0)
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

      x0 = ((float *)x) + (n_elem - 1) * (-*incx);
      incx0 = (inc_t)(*incx);
    }
    else
    {
      x0    = ((float*)x);
      incx0 = ( inc_t )(*incx);
    }
    if ( *incy < 0 )
    {
      y0    = ((float*)y) + (n_elem-1)*(-*incy);
      incy0 = ( inc_t )(*incy);
    }
    else
    {
      y0    = ((float*)y);
      incy0 = ( inc_t )(*incy);
    }

    cntx_t *cntx = NULL;

    // Query the architecture ID
    arch_t id = bli_arch_query_id();

    /*
      Function pointer declaration for the function
      that will be used by this API
    */
    saxpyv_ker_ft axpyv_ker_ptr; // DAXPYV

    // Pick the kernel based on the architecture ID
    switch (id)
    {
      case BLIS_ARCH_ZEN4:
#if defined(BLIS_KERNELS_ZEN4)
        axpyv_ker_ptr = bli_saxpyv_zen_int_avx512;

        break;
#endif
      case BLIS_ARCH_ZEN:
      case BLIS_ARCH_ZEN2:
      case BLIS_ARCH_ZEN3:
        axpyv_ker_ptr = bli_saxpyv_zen_int10;

        break;
      default:

        // For non-Zen architectures, query the context
        cntx = bli_gks_query_cntx();

        // Query the context for the kernel function pointers for saxpyv
        axpyv_ker_ptr = bli_cntx_get_l1v_ker_dt(BLIS_FLOAT, BLIS_AXPYV_KER, cntx);
    }

    // Call the function based on the function pointer assigned above
    axpyv_ker_ptr
    (
      BLIS_NO_CONJUGATE,
      n_elem,
      (float *)alpha,
      x0, incx0,
      y0, incy0,
      cntx
    );

    /* Finalize BLIS. */
    //    bli_finalize_auto();
    AOCL_DTL_TRACE_EXIT(AOCL_DTL_LEVEL_TRACE_1);
}

#ifdef BLIS_ENABLE_BLAS
void saxpy_
(
 const f77_int* n,
 const float*   alpha,
 const float*   x, const f77_int* incx,
 float*   y, const f77_int* incy
 )
{
  saxpy_blis_impl( n, alpha, x, incx, y, incy ) ; 
}
#endif

void daxpy_blis_impl
(
 const f77_int* n,
 const double*   alpha,
 const double*   x, const f77_int* incx,
 double*   y, const f77_int* incy
 )
{
    dim_t  n_elem;
    double* x0;
    double* y0;
    inc_t  incx0;
    inc_t  incy0;

    AOCL_DTL_TRACE_ENTRY(AOCL_DTL_LEVEL_TRACE_1)
    AOCL_DTL_LOG_AXPY_INPUTS(AOCL_DTL_LEVEL_TRACE_1, 'D', *n, (double*)alpha, *incx, *incy)
    /* Initialize BLIS. */
    // bli_init_auto();

    /* Convert/typecast negative values of n to zero. */
    if ( *n < 0 ) n_elem = ( dim_t )0;
    else          n_elem = ( dim_t )(*n);

    // BLAS exception to return early when n <= 0 or alpha is 0.0
    if(*n <= 0 || bli_deq0(*alpha))
    {
      AOCL_DTL_TRACE_EXIT(AOCL_DTL_LEVEL_TRACE_1);
      return;
    }

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
      x0    = ( (double*)x ) + ( n_elem - 1 ) * ( - (*incx) );
      incx0 = ( inc_t )(*incx);
    }
    else
    {
      x0    = ((double*)x);
      incx0 = ( inc_t )(*incx);
    }
    if ( *incy < 0 )
    {
      y0    = ( (double*) y ) + ( n_elem - 1 )*( - (*incy) );
      incy0 = ( inc_t )(*incy);
    }
    else
    {
      y0    = ((double*)y);
      incy0 = ( inc_t )(*incy);
    }

    // Definition of function pointer
    daxpyv_ker_ft axpyv_ker_ptr;

    cntx_t *cntx = NULL;

    // Query the architecture ID
    arch_t arch_id_local = bli_arch_query_id();

    // Pick the kernel based on the architecture ID
    switch (arch_id_local)
    {
      case BLIS_ARCH_ZEN4:
#if defined(BLIS_KERNELS_ZEN4)
        axpyv_ker_ptr = bli_daxpyv_zen_int_avx512;

        break;
#endif
      case BLIS_ARCH_ZEN:
      case BLIS_ARCH_ZEN2:
      case BLIS_ARCH_ZEN3:

          // AVX2 Kernel
          axpyv_ker_ptr = bli_daxpyv_zen_int10;
          break;

      default:

          // Query the context
          cntx = bli_gks_query_cntx();

          // Query the function pointer using the context
          axpyv_ker_ptr = bli_cntx_get_l1v_ker_dt(BLIS_DOUBLE, BLIS_AXPYV_KER, cntx);
    }

#ifdef BLIS_ENABLE_OPENMP
    /*
      Initializing the number of thread to one
      to avoid compiler warnings
    */
    dim_t nt = 1;

    /*
      For the given problem size and architecture, the function
      returns the optimum number of threads with AOCL dynamic enabled
      else it returns the number of threads requested by the user.
    */
    bli_nthreads_l1
    (
      BLIS_AXPYV_KER,
      BLIS_DOUBLE,
      BLIS_DOUBLE,
      arch_id_local,
      n_elem,
      &nt
    );

    if (nt == 1)
    {
#endif
        axpyv_ker_ptr
        (
          BLIS_NO_CONJUGATE,
          n_elem,
          (double *)alpha,
          x0, incx0,
          y0, incy0,
          cntx
        );

        AOCL_DTL_TRACE_EXIT(AOCL_DTL_LEVEL_TRACE_1)

        return;
#ifdef BLIS_ENABLE_OPENMP
    }

    _Pragma("omp parallel num_threads(nt)")
    {
        dim_t start, length;

        // Get the thread ID
        dim_t thread_id = omp_get_thread_num();

        // Get the actual number of threads spawned
        dim_t nt_use = omp_get_num_threads();

        /*
          Calculate the compute range for the current thread
          based on the actual number of threads spawned
        */
        bli_thread_vector_partition
        (
          n_elem,
          nt_use,
          &start, &length,
          thread_id
        );

        // Adjust the local pointer for computation
        double *x_thread_local = x0 + (start * incx0);
        double *y_thread_local = y0 + (start * incy0);

        // Invoke the function based on the kernel function pointer
        axpyv_ker_ptr
        (
          BLIS_NO_CONJUGATE,
          length,
          (double *)alpha,
          x_thread_local, incx0,
          y_thread_local, incy0,
          cntx
        );
    }
#endif // BLIS_ENABLE_OPENMP

    AOCL_DTL_TRACE_EXIT(AOCL_DTL_LEVEL_TRACE_1);
    /* Finalize BLIS. */
    // bli_finalize_auto();
}

#ifdef BLIS_ENABLE_BLAS
void daxpy_
(
 const f77_int* n,
 const double*   alpha,
 const double*   x, const f77_int* incx,
 double*   y, const f77_int* incy
 )
{
  daxpy_blis_impl( n, alpha, x, incx, y, incy ) ; 
}
#endif
void caxpy_blis_impl
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

  // This function is invoked on all architectures including 'generic'.
  // Non-AVX2+FMA3 platforms will use the kernels derived from the context.
  if (bli_cpuid_is_avx2fma3_supported() == TRUE)
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
#ifdef BLIS_ENABLE_BLAS
void caxpy_
(
 const f77_int* n,
 const scomplex*   alpha,
 const scomplex*   x, const f77_int* incx,
 scomplex*   y, const f77_int* incy
 )
{
  caxpy_blis_impl( n, alpha, x, incx, y, incy ) ; 
}
#endif
void zaxpy_blis_impl
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

  // This function is invoked on all architectures including 'generic'.
  // Non-AVX2+FMA3 platforms will use the kernels derived from the context.
  if (bli_cpuid_is_avx2fma3_supported() == TRUE)
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
#ifdef BLIS_ENABLE_BLAS
void zaxpy_
(
 const f77_int* n,
 const dcomplex*   alpha,
 const dcomplex*   x, const f77_int* incx,
 dcomplex*   y, const f77_int* incy
 )
{
  zaxpy_blis_impl( n, alpha, x, incx, y, incy ) ; 
}


#endif
