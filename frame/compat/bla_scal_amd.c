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
  2. When incx <= 0 where incx is the storage spacing between elements of
     the vector passed
  3. When alpha == 1 where alpha is the scalar value by which the vector is
     to be scaled

  NaN propagation expectation
  --------------------------

  1. When alpha == NaN - Propogate the NaN to the vector
  2. When alpha == 0 - Perform the SCALV operation completely and don't use setv.
*/

//
// Define BLAS-to-BLIS interfaces.
//
#undef  GENTFUNCSCAL
#define GENTFUNCSCAL( ftype_x, ftype_a, chx, cha, blasname, blisname ) \
\
void PASTEF772S(chx,cha,blasname) \
     ( \
       const f77_int* n, \
       const ftype_a* alpha, \
       ftype_x* x, const f77_int* incx  \
     ) \
{ \
	AOCL_DTL_TRACE_ENTRY(AOCL_DTL_LEVEL_TRACE_1) \
	dim_t    n0; \
	ftype_x* x0; \
	inc_t    incx0; \
	ftype_x  alpha_cast; \
\
	/* Initialize BLIS. */ \
	bli_init_auto(); \
\
	if (*n == 0 || alpha == NULL) { \
		AOCL_DTL_TRACE_EXIT(AOCL_DTL_LEVEL_TRACE_1); \
		return ; \
	} \
\
	/* Convert/typecast negative values of n to zero. */ \
	bli_convert_blas_dim1( *n, n0 ); \
\
	/* If the input increments are negative, adjust the pointers so we can
	   use positive increments instead. */ \
	bli_convert_blas_incv( n0, (ftype_x*)x, *incx, x0, incx0 ); \
\
	/* NOTE: We do not natively implement BLAS's csscal/zdscal in BLIS.
	   that is, we just always sub-optimally implement those cases
	   by casting alpha to ctype_x (potentially the complex domain) and
	   using the homogeneous datatype instance according to that type. */ \
	PASTEMAC2(cha,chx,copys)( *alpha, alpha_cast ); \
\
	/* Call BLIS interface. */ \
	PASTEMAC2(chx,blisname,BLIS_TAPI_EX_SUF) \
	( \
	  BLIS_NO_CONJUGATE, \
	  n0, \
	  &alpha_cast, \
	  x0, incx0, \
	  NULL, \
	  NULL  \
	); \
\
  AOCL_DTL_TRACE_EXIT(AOCL_DTL_LEVEL_TRACE_1) \
	/* Finalize BLIS. */ \
	bli_finalize_auto(); \
}\
IF_BLIS_ENABLE_BLAS(\
void PASTEF772(chx,cha,blasname) \
     ( \
       const f77_int* n, \
       const ftype_a* alpha, \
       ftype_x* x, const f77_int* incx  \
     ) \
{ \
  PASTEF772S(chx,cha,blasname)( n, alpha, x, incx ); \
} \
)

void sscal_blis_impl
     (
       const f77_int* n,
       const float* alpha,
       float*   x, const f77_int* incx
     )
{
    AOCL_DTL_TRACE_ENTRY(AOCL_DTL_LEVEL_TRACE_1)
    AOCL_DTL_LOG_SCAL_INPUTS(AOCL_DTL_LEVEL_TRACE_1, 'S', (void *) alpha, *n, *incx );
    dim_t  n0;
    float* x0;
    inc_t  incx0;
    /* Initialize BLIS. */
    //bli_init_auto();

    if ((*n) <= 0 || alpha == NULL || bli_seq1(*alpha))
    {
      AOCL_DTL_TRACE_EXIT(AOCL_DTL_LEVEL_TRACE_1);
      return;
    }

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

    /*
      According to the BLAS definition, return early when incx <= 0
    */
    if (incx0 <= 0)
    {
        AOCL_DTL_TRACE_EXIT(AOCL_DTL_LEVEL_TRACE_1);
        return;
    }

    cntx_t *cntx = NULL;

    // Query the architecture ID
    arch_t id = bli_arch_query_id();

    /*
      Function pointer declaration for the function
      that will be used by this API
    */
    sscalv_ker_ft scalv_ker_ptr; // DSCALV

    // Pick the kernel based on the architecture ID
    switch (id)
    {
      case BLIS_ARCH_ZEN4:
#if defined(BLIS_KERNELS_ZEN4)
        scalv_ker_ptr = bli_sscalv_zen_int_avx512;

        break;
#endif
      case BLIS_ARCH_ZEN:
      case BLIS_ARCH_ZEN2:
      case BLIS_ARCH_ZEN3:
        scalv_ker_ptr = bli_sscalv_zen_int10;

        break;
      default:

        // For non-Zen architectures, query the context
        cntx = bli_gks_query_cntx();

        // Query the context for the kernel function pointers for sscalv
        scalv_ker_ptr = bli_cntx_get_l1v_ker_dt(BLIS_FLOAT, BLIS_SCALV_KER, cntx);
    }

    scalv_ker_ptr
    (
      BLIS_NO_CONJUGATE,
      n0,
      (float *)alpha,
      x0, incx0,
      cntx
    );

    /* Finalize BLIS. */
    //    bli_finalize_auto();
    AOCL_DTL_TRACE_EXIT(AOCL_DTL_LEVEL_TRACE_1)
}
#ifdef BLIS_ENABLE_BLAS
void sscal_
     (
       const f77_int* n,
       const float* alpha,
       float*   x, const f77_int* incx
     )
{
  sscal_blis_impl( n, alpha, x, incx );
}
#endif
void dscal_blis_impl
     (
       const f77_int* n,
       const double* alpha,
       double*   x, const f77_int* incx
     )
{
    AOCL_DTL_TRACE_ENTRY(AOCL_DTL_LEVEL_TRACE_1)
    AOCL_DTL_LOG_SCAL_INPUTS(AOCL_DTL_LEVEL_TRACE_1, 'D', (void *)alpha, *n, *incx );
    dim_t  n_elem;
    double* x0;
    inc_t  incx0;

    /* Initialize BLIS  */
    //bli_init_auto();

    /* Convert typecast negative values of n to zero. */
    if ( *n < 0 ) n_elem = ( dim_t )0;
    else          n_elem = ( dim_t )(*n);

    /*
      Return early when n <= 0 or incx <= 0 or alpha == 1.0 - BLAS exception
      Return early when alpha pointer is NULL - BLIS exception
    */
    if ((*n) <= 0 || alpha == NULL || bli_deq1(*alpha) || (*incx) <= 0)
    {
        AOCL_DTL_TRACE_EXIT(AOCL_DTL_LEVEL_TRACE_1);
        return;
    }

    /* If the input increments are negative, adjust the pointers so we can
       use positive increments instead.
       * This check is redundant and can be safely removed
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

        x0    = (x) + (n_elem-1)*(-*incx);
        incx0 = ( inc_t )(*incx);

    }
    else
    {
        x0    = (x);
        incx0 = ( inc_t )(*incx);
    }

     // Definition of function pointer
    dscalv_ker_ft scalv_ker_ptr;

    cntx_t *cntx = NULL;

    // Query the architecture ID
    arch_t arch_id_local = bli_arch_query_id();

    // Pick the kernel based on the architecture ID
    switch (arch_id_local)
    {
      case BLIS_ARCH_ZEN4:
#if defined(BLIS_KERNELS_ZEN4)
        scalv_ker_ptr = bli_dscalv_zen_int_avx512;

        break;
#endif
      case BLIS_ARCH_ZEN:
      case BLIS_ARCH_ZEN2:
      case BLIS_ARCH_ZEN3:

          // AVX2 Kernel
          scalv_ker_ptr = bli_dscalv_zen_int10;
          break;

      default:

          // Query the context
          cntx = bli_gks_query_cntx();

          // Query the function pointer using the context
          scalv_ker_ptr = bli_cntx_get_l1v_ker_dt(BLIS_DOUBLE, BLIS_SCALV_KER, cntx);
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
      BLIS_SCALV_KER,
      BLIS_DOUBLE,
      BLIS_DOUBLE,
      arch_id_local,
      n_elem,
      &nt
    );

    /*
      If the number of optimum threads is 1, the OpenMP overhead
      is avoided by calling the function directly
    */
    if (nt == 1)
    {
#endif
        scalv_ker_ptr
        (
          BLIS_NO_CONJUGATE,
          n_elem,
          (double *)alpha,
          x0, incx0,
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

        // Invoke the function based on the kernel function pointer
        scalv_ker_ptr
        (
          BLIS_NO_CONJUGATE,
          length,
          (double *)alpha,
          x_thread_local, incx0,
          cntx
        );
    }
#endif


    /* Finalize BLIS. */
    // bli_finalize_auto();
    AOCL_DTL_TRACE_EXIT(AOCL_DTL_LEVEL_TRACE_1)
}
#ifdef BLIS_ENABLE_BLAS
void dscal_
     (
       const f77_int* n,
       const double* alpha,
       double*   x, const f77_int* incx
     )
{
  dscal_blis_impl( n, alpha, x, incx );
}
#endif
void zdscal_blis_impl
     (
       const f77_int* n,
       const double* alpha,
       dcomplex*   x, const f77_int* incx
     )
{
    AOCL_DTL_TRACE_ENTRY(AOCL_DTL_LEVEL_TRACE_1)
    AOCL_DTL_LOG_SCAL_INPUTS(AOCL_DTL_LEVEL_TRACE_1, 'Z', (void *) alpha, *n, *incx );
    dim_t  n_elem;
    dcomplex* x0;
    inc_t  incx0;
    /* Initialize BLIS. */
    //bli_init_auto();

    /* Convert/typecast negative values of n to zero. */
    if ( *n < 0 ) n_elem = ( dim_t )0;
    else          n_elem = ( dim_t )(*n);

    /*
      Return early when n <= 0 or incx <= 0 or alpha == 1.0 - BLAS exception
      Return early when alpha pointer is NULL - BLIS exception
    */
    if (*n <= 0 || alpha == NULL || bli_deq1(*alpha) || incx <= 0)
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

        x0    = (x) + (n_elem-1)*(-*incx);
        incx0 = ( inc_t )(*incx);
    }
    else
    {
        x0    = (x);
        incx0 = ( inc_t )(*incx);
    }

    dcomplex  alpha_cast;
    alpha_cast.real = *alpha;
    alpha_cast.imag = 0.0;

    // Definition of function pointer
    zscalv_ker_ft scalv_ker_ptr;

    cntx_t *cntx = NULL;

    // Query the architecture ID
    arch_t arch_id_local = bli_arch_query_id();

    // Pick the kernel based on the architecture ID
    switch (arch_id_local)
    {
      case BLIS_ARCH_ZEN4:
      case BLIS_ARCH_ZEN:
      case BLIS_ARCH_ZEN2:
      case BLIS_ARCH_ZEN3:

          // AVX2 Kernel
          scalv_ker_ptr = bli_zdscalv_zen_int10;
          break;

      default:

          // Query the context
          cntx = bli_gks_query_cntx();

          // Query the function pointer using the context
          scalv_ker_ptr = bli_cntx_get_l1v_ker_dt(BLIS_DCOMPLEX, BLIS_SCALV_KER, cntx);
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
      BLIS_SCALV_KER,
      BLIS_DCOMPLEX,
      BLIS_DOUBLE,
      arch_id_local,
      n_elem,
      &nt
    );

    /*
      If the number of optimum threads is 1, the OpenMP overhead
      is avoided by calling the function directly
    */
    if (nt == 1)
    {
#endif
        scalv_ker_ptr
        (
          BLIS_NO_CONJUGATE,
          n_elem,
          (dcomplex *)&alpha_cast,
          x0, incx0,
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
        dcomplex *x_thread_local = x0 + (start * incx0);

        // Invoke the function based on the kernel function pointer
        scalv_ker_ptr
        (
          BLIS_NO_CONJUGATE,
          length,
          (dcomplex *)&alpha_cast,
          x_thread_local, incx0,
          cntx
        );
    }
#endif

    AOCL_DTL_TRACE_EXIT(AOCL_DTL_LEVEL_TRACE_1)
}
#ifdef BLIS_ENABLE_BLAS
void zdscal_
     (
       const f77_int* n,
       const double* alpha,
       dcomplex*   x, const f77_int* incx
     )
{
    zdscal_blis_impl( n, alpha, x, incx );
}
#endif

void zscal_blis_impl
     (
       const f77_int* n,
       const dcomplex* alpha,
       dcomplex*   x, const f77_int* incx
     )
{
  AOCL_DTL_TRACE_ENTRY(AOCL_DTL_LEVEL_TRACE_1)
  AOCL_DTL_LOG_SCAL_INPUTS(AOCL_DTL_LEVEL_TRACE_1, 'Z', (void *)alpha, *n, *incx);
  dim_t n0;
  dcomplex *x0;
  inc_t incx0;

  // When n is zero or the alpha pointer passed is null, return early
  if ((*n == 0) || (alpha == NULL))
  {
      AOCL_DTL_TRACE_EXIT(AOCL_DTL_LEVEL_TRACE_1);
      return;
  }

  /* Convert/typecast negative values of n to zero. */
  if (*n < 0)
    n0 = (dim_t)0;
  else
    n0 = (dim_t)(*n);

  /* If the input increments are negative, adjust the pointers so we can
    use positive increments instead. */
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

    x0 = (x) + (n0 - 1) * (-*incx);
    incx0 = (inc_t)(*incx);
  }
  else
  {
    x0 = (x);
    incx0 = (inc_t)(*incx);
  }

  /* If the incx is zero, return early. */
  if (bli_zero_dim1(incx0))
    return;

  // Definition of function pointer
  zscalv_ker_ft scalv_fun_ptr;

  cntx_t* cntx = NULL;

  // Query the architecture ID
  arch_t id = bli_arch_query_id();

  // Pick the kernel based on the architecture ID
  switch (id)
  {
  case BLIS_ARCH_ZEN4:
  case BLIS_ARCH_ZEN:
  case BLIS_ARCH_ZEN2:
  case BLIS_ARCH_ZEN3:

    // AVX2 Kernel
    scalv_fun_ptr = bli_zscalv_zen_int;
    break;

  default:

    // Query the context
    cntx = bli_gks_query_cntx();

    // Query the function pointer using the context
    scalv_fun_ptr = bli_cntx_get_l1v_ker_dt(BLIS_DCOMPLEX, BLIS_SCALV_KER, cntx);
  }

  /* The expectation is that the condition to return early for vector dimension is zero
  or the real part of alpha is 1 and imaginary part 0 is inside the compute kernel called */

  // Call the function based on the function pointer assigned above
  scalv_fun_ptr
  (
    BLIS_NO_CONJUGATE,
    n0,
    (dcomplex*) alpha,
    x0, incx0,
    cntx
  );

  AOCL_DTL_TRACE_EXIT(AOCL_DTL_LEVEL_TRACE_1)
}
#ifdef BLIS_ENABLE_BLAS
void zscal_
     (
       const f77_int* n,
       const dcomplex* alpha,
       dcomplex*   x, const f77_int* incx
     )
{
    zscal_blis_impl(n, alpha, x, incx);
}
#endif

INSERT_GENTFUNCSCAL_BLAS_C( scal, scalv )

