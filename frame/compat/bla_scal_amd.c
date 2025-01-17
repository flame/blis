/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2014, The University of Texas at Austin
   Copyright (C) 2020 - 2025, Advanced Micro Devices, Inc. All rights reserved.

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
     As SCALV kernels are used in many other BLAS APIs where we want setv to be
     used in this scenario, here we call the kernels with n=-n to signify that
     setv should not be used.
*/

//
// Define BLAS-to-BLIS interfaces.
//
#undef  GENTFUNCSCAL
#define GENTFUNCSCAL( ftype_x, ftype_a, chx, cha, chau, blasname, blisname ) \
\
void PASTEF772S(chx,cha,blasname) \
     ( \
       const f77_int* n, \
       const ftype_a* alpha, \
       ftype_x* x, const f77_int* incx  \
     ) \
{ \
	AOCL_DTL_TRACE_ENTRY(AOCL_DTL_LEVEL_TRACE_1) \
\
	/* Initialize BLIS. */ \
	bli_init_auto(); \
\
	dim_t n0 = (dim_t)(*n); \
	ftype_x *x0 = x; \
	inc_t incx0 = (inc_t)(*incx); \
\
	if ((n0 <= 0) || (alpha == NULL) || (incx0 <= 0) || PASTEMAC(chau, eq1)(*alpha)) \
	{ \
		AOCL_DTL_TRACE_EXIT(AOCL_DTL_LEVEL_TRACE_1); \
		/* Finalize BLIS. */ \
		bli_finalize_auto(); \
		return ; \
	} \
\
	/* NOTE: We do not natively implement BLAS's csscal/zdscal in BLIS.
	   that is, we just always sub-optimally implement those cases
	   by casting alpha to ctype_x (potentially the complex domain) and
	   using the homogeneous datatype instance according to that type. */ \
	ftype_x  alpha_cast; \
	PASTEMAC2(cha,chx,copys)( *alpha, alpha_cast ); \
\
	/* Call BLIS interface. */ \
	/* Pass size as negative to stipulate don't use SETV when alpha=0 */ \
	PASTEMAC2(chx,blisname,BLIS_TAPI_EX_SUF) \
	( \
	  BLIS_NO_CONJUGATE, \
	  -n0, \
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

    /* Initialize BLIS. */
    //bli_init_auto();

    dim_t n0 = (dim_t)(*n);
    float *x0 = x;
    inc_t incx0 = (inc_t)(*incx);

    /*
      Return early when n <= 0 or incx <= 0 or alpha == 1.0 - BLAS exception
      Return early when alpha pointer is NULL - BLIS exception
    */
    if ((n0 <= 0) || (alpha == NULL) || (incx0 <= 0) || PASTEMAC(s, eq1)(*alpha))
    {
        AOCL_DTL_TRACE_EXIT(AOCL_DTL_LEVEL_TRACE_1);
        /* Finalize BLIS. */
        //bli_finalize_auto();
        return;
    }

    // Definition of function pointer
    sscalv_ker_ft scalv_ker_ptr;

    cntx_t *cntx = NULL;

    // Query the architecture ID
    arch_t id = bli_arch_query_id();

    // Pick the kernel based on the architecture ID
    switch (id)
    {
        case BLIS_ARCH_ZEN5:
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

    // Invoke the function based on the kernel function pointer
    // Pass size as negative to stipulate don't use SETV when alpha=0
    scalv_ker_ptr
    (
      BLIS_NO_CONJUGATE,
      -n0,
      (float *)alpha,
      x0, incx0,
      cntx
    );

    AOCL_DTL_TRACE_EXIT(AOCL_DTL_LEVEL_TRACE_1)
    /* Finalize BLIS. */
    //bli_finalize_auto();
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

    /* Initialize BLIS. */
    //bli_init_auto();

    dim_t n0 = (dim_t)(*n);
    double *x0 = x;
    inc_t incx0 = (inc_t)(*incx);

    /*
      Return early when n <= 0 or incx <= 0 or alpha == 1.0 - BLAS exception
      Return early when alpha pointer is NULL - BLIS exception
    */
    if ((n0 <= 0) || (alpha == NULL) || (incx0 <= 0) || PASTEMAC(d, eq1)(*alpha))
    {
        AOCL_DTL_TRACE_EXIT(AOCL_DTL_LEVEL_TRACE_1);
        /* Finalize BLIS. */
        //bli_finalize_auto();
        return;
    }

    // Definition of function pointer
    dscalv_ker_ft scalv_ker_ptr;

    cntx_t *cntx = NULL;

#ifdef BLIS_ENABLE_OPENMP
    dim_t ST_THRESH = 30000;
#endif

    // Query the architecture ID
    arch_t id = bli_arch_query_id();

    // Pick the kernel based on the architecture ID
    switch (id)
    {
        case BLIS_ARCH_ZEN5:
#if defined(BLIS_KERNELS_ZEN5)
          // AVX512 Kernel
          scalv_ker_ptr = bli_dscalv_zen_int_avx512;
  #ifdef BLIS_ENABLE_OPENMP
          ST_THRESH = 63894;
  #endif
          break;
#endif
        case BLIS_ARCH_ZEN4:
#if defined(BLIS_KERNELS_ZEN4)
          // AVX512 Kernel
          scalv_ker_ptr = bli_dscalv_zen_int_avx512;
  #ifdef BLIS_ENABLE_OPENMP
          ST_THRESH = 27500;
  #endif
          break;
#endif
        case BLIS_ARCH_ZEN:
        case BLIS_ARCH_ZEN2:
        case BLIS_ARCH_ZEN3:

          // AVX2 Kernel
          scalv_ker_ptr = bli_dscalv_zen_int10;
#ifdef BLIS_ENABLE_OPENMP
          ST_THRESH = 30000;
#endif
          break;

        default:

          // For non-Zen architectures, query the context
          cntx = bli_gks_query_cntx();

          // Query the function pointer using the context
          scalv_ker_ptr = bli_cntx_get_l1v_ker_dt(BLIS_DOUBLE, BLIS_SCALV_KER, cntx);

    }

#ifdef BLIS_ENABLE_OPENMP
    /*
      If the optimal number of threads is 1, the OpenMP and
      'bli_nthreads_l1' overheads are avoided by calling the
      function directly. This ensures that performance of dscalv
      does not drop for single  thread when OpenMP is enabled.
    */
    if (n0 <= ST_THRESH)
    {
#endif
        // Invoke the function based on the kernel function pointer
        // Pass size as negative to stipulate don't use SETV when alpha=0
        scalv_ker_ptr
        (
          BLIS_NO_CONJUGATE,
          -n0,
          (double *)alpha,
          x0, incx0,
          cntx
        );

        AOCL_DTL_TRACE_EXIT(AOCL_DTL_LEVEL_TRACE_1)
        /* Finalize BLIS. */
        //bli_finalize_auto();
        return;
#ifdef BLIS_ENABLE_OPENMP
    }

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
      id,
      n0,
      &nt
    );

    _Pragma("omp parallel num_threads(nt)")
    {
        dim_t start, end, length;
        thrinfo_t thrinfo_vec;

        // The block size is the minimum factor, whose multiple will ensure that only
        // the vector code section is executed. Furthermore, for double datatype it corresponds
        // to one cacheline size.
        dim_t block_size = 8;

        // Get the actual number of threads spawned
        thrinfo_vec.n_way = omp_get_num_threads();

        // Get the thread ID
        thrinfo_vec.work_id = omp_get_thread_num();

        /*
          Calculate the compute range for the current thread
          based on the actual number of threads spawned
        */

        bli_thread_range_sub
        (
          &thrinfo_vec,
          n0,
          block_size,
          FALSE,
          &start,
          &end
        );

        length = end - start;

        // Adjust the local pointer for computation
        double *x_thread_local = x0 + (start * incx0);

        // Invoke the function based on the kernel function pointer
        // Pass size as negative to stipulate don't use SETV when alpha=0
        scalv_ker_ptr
        (
          BLIS_NO_CONJUGATE,
          -length,
          (double *)alpha,
          x_thread_local, incx0,
          cntx
        );
    }

    AOCL_DTL_TRACE_EXIT(AOCL_DTL_LEVEL_TRACE_1)
    /* Finalize BLIS. */
    //bli_finalize_auto();
#endif
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

    /* Initialize BLIS. */
    //bli_init_auto();

    dim_t  n0 = (dim_t)(*n);
    dcomplex* x0 = x;
    inc_t  incx0 = (inc_t)(*incx);

    /*
      Return early when n <= 0 or incx <= 0 or alpha == 1.0 - BLAS exception
      Return early when alpha pointer is NULL - BLIS exception
    */
    if ((n0 <= 0) || (alpha == NULL) || (incx0 <= 0) || PASTEMAC(d, eq1)(*alpha))
    {
        AOCL_DTL_TRACE_EXIT(AOCL_DTL_LEVEL_TRACE_1);
        /* Finalize BLIS. */
        //bli_finalize_auto();
        return;
    }

    dcomplex  alpha_cast;
    alpha_cast.real = *alpha;
    alpha_cast.imag = 0.0;

    // Definition of function pointer
    zscalv_ker_ft scalv_ker_ptr;

    cntx_t *cntx = NULL;

#ifdef BLIS_ENABLE_OPENMP
    dim_t ST_THRESH = 10000;
#endif

    // Query the architecture ID
    arch_t id = bli_arch_query_id();

    // Pick the kernel based on the architecture ID
    switch (id)
    {
        case BLIS_ARCH_ZEN5:
        case BLIS_ARCH_ZEN4:
#if defined(BLIS_KERNELS_ZEN4)
          // AVX512 Kernel
          scalv_ker_ptr = bli_zdscalv_zen_int_avx512;
          break;
#endif
        case BLIS_ARCH_ZEN:
        case BLIS_ARCH_ZEN2:
        case BLIS_ARCH_ZEN3:

          // AVX2 Kernel
          scalv_ker_ptr = bli_zdscalv_zen_int10;
          break;

        default:

          // For non-Zen architectures, query the context
          cntx = bli_gks_query_cntx();

          // Query the function pointer using the context
          scalv_ker_ptr = bli_cntx_get_l1v_ker_dt(BLIS_DCOMPLEX, BLIS_SCALV_KER, cntx);
    }

#ifdef BLIS_ENABLE_OPENMP
    /*
      If the optimal number of threads is 1, the OpenMP and
      'bli_nthreads_l1' overheads are avoided by calling the
      function directly. This ensures that performance of dscalv
      does not drop for single  thread when OpenMP is enabled.
    */
    if (n0 <= ST_THRESH)
    {
#endif
        // Invoke the function based on the kernel function pointer
        // Pass size as negative to stipulate don't use SETV when alpha=0
        scalv_ker_ptr
        (
          BLIS_NO_CONJUGATE,
          -n0,
          (dcomplex *)&alpha_cast,
          x0, incx0,
          cntx
        );

        AOCL_DTL_TRACE_EXIT(AOCL_DTL_LEVEL_TRACE_1)
        /* Finalize BLIS. */
        //bli_finalize_auto();
        return;
#ifdef BLIS_ENABLE_OPENMP
    }

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
      id,
      n0,
      &nt
    );

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
          n0,
          nt_use,
          &start, &length,
          thread_id
        );

        // Adjust the local pointer for computation
        dcomplex *x_thread_local = x0 + (start * incx0);

        // Invoke the function based on the kernel function pointer
        // Pass size as negative to stipulate don't use SETV when alpha=0
        scalv_ker_ptr
        (
          BLIS_NO_CONJUGATE,
          -length,
          (dcomplex *)&alpha_cast,
          x_thread_local, incx0,
          cntx
        );
    }

    AOCL_DTL_TRACE_EXIT(AOCL_DTL_LEVEL_TRACE_1)
    /* Finalize BLIS. */
    //bli_finalize_auto();
#endif
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

void cscal_blis_impl
     (
       const f77_int*  n,
       const scomplex* alpha,
             scomplex* x, const f77_int* incx
     )
{
    AOCL_DTL_TRACE_ENTRY(AOCL_DTL_LEVEL_TRACE_1)
    AOCL_DTL_LOG_SCAL_INPUTS(AOCL_DTL_LEVEL_TRACE_1, 'C', (void *)alpha, *n, *incx);

    /* Initialize BLIS. */
    //bli_init_auto();

    dim_t n0 = (dim_t)(*n);
    scomplex *x0 = x;
    inc_t incx0 = (inc_t)(*incx);

    /*
      Return early when n <= 0 or incx <= 0 or alpha == 1.0 - BLAS exception
      Return early when alpha pointer is NULL - BLIS exception
    */
    if ((n0 <= 0) || (alpha == NULL) || (incx0 <= 0) || PASTEMAC(c, eq1)(*alpha))
    {
        AOCL_DTL_TRACE_EXIT(AOCL_DTL_LEVEL_TRACE_1);
        /* Finalize BLIS. */
        //bli_finalize_auto();
        return;
    }

    // Definition of function pointer
    cscalv_ker_ft scalv_ker_ptr;

    cntx_t* cntx = NULL;

    // Query the architecture ID
    arch_t id = bli_arch_query_id();

    // Pick the kernel based on the architecture ID
    switch (id)
    {
        case BLIS_ARCH_ZEN5:
        case BLIS_ARCH_ZEN4:
#if defined(BLIS_KERNELS_ZEN4)
          // AVX512 Kernel
          scalv_ker_ptr = bli_cscalv_zen_int_avx512;
          break;
#endif
        case BLIS_ARCH_ZEN:
        case BLIS_ARCH_ZEN2:
        case BLIS_ARCH_ZEN3:

          // AVX2 Kernel
          scalv_ker_ptr = bli_cscalv_zen_int;
          break;

        default:

          // For non-Zen architectures, query the context
          cntx = bli_gks_query_cntx();

          // Query the function pointer using the context
          scalv_ker_ptr = bli_cntx_get_l1v_ker_dt(BLIS_SCOMPLEX, BLIS_SCALV_KER, cntx);
    }

    // Invoke the function based on the kernel function pointer
    // Pass size as negative to stipulate don't use SETV when alpha=0
    scalv_ker_ptr
    (
      BLIS_NO_CONJUGATE,
      -n0,
      (scomplex*) alpha,
      x0, incx0,
      cntx
    );

    AOCL_DTL_TRACE_EXIT(AOCL_DTL_LEVEL_TRACE_1)
    /* Finalize BLIS. */
    //bli_finalize_auto();
}
#ifdef BLIS_ENABLE_BLAS
void cscal_
     (
        const f77_int*  n,
        const scomplex* alpha,
              scomplex* x, const f77_int* incx
     )
{
    cscal_blis_impl( n, alpha, x, incx );
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

    /* Initialize BLIS. */
    //bli_init_auto();

    dim_t n0 = (dim_t)(*n);
    dcomplex *x0 = x;
    inc_t incx0 = (inc_t)(*incx);

    /*
      Return early when n <= 0 or incx <= 0 or alpha == 1.0 - BLAS exception
      Return early when alpha pointer is NULL - BLIS exception
    */
    if ((n0 <= 0) || (alpha == NULL) || (incx0 <= 0) || PASTEMAC(z, eq1)(*alpha))
    {
        AOCL_DTL_TRACE_EXIT(AOCL_DTL_LEVEL_TRACE_1);
        /* Finalize BLIS. */
        //bli_finalize_auto();
        return;
    }

    // Definition of function pointer
    zscalv_ker_ft scalv_ker_ptr;

    cntx_t* cntx = NULL;

    // Query the architecture ID
    arch_t id = bli_arch_query_id();

    // Pick the kernel based on the architecture ID
    switch (id)
    {
        case BLIS_ARCH_ZEN5:
        case BLIS_ARCH_ZEN4:
#if defined(BLIS_KERNELS_ZEN4)
          // AVX512 Kernel
          scalv_ker_ptr = bli_zscalv_zen_int_avx512;
          break;
#endif
        case BLIS_ARCH_ZEN:
        case BLIS_ARCH_ZEN2:
        case BLIS_ARCH_ZEN3:

          // AVX2 Kernel
          scalv_ker_ptr = bli_zscalv_zen_int;
          break;

        default:

          // For non-Zen architectures, query the context
          cntx = bli_gks_query_cntx();

          // Query the function pointer using the context
          scalv_ker_ptr = bli_cntx_get_l1v_ker_dt(BLIS_DCOMPLEX, BLIS_SCALV_KER, cntx);
    }

    // Invoke the function based on the kernel function pointer
    // Pass size as negative to stipulate don't use SETV when alpha=0
    scalv_ker_ptr
    (
      BLIS_NO_CONJUGATE,
      -n0,
      (dcomplex*) alpha,
      x0, incx0,
      cntx
    );

    AOCL_DTL_TRACE_EXIT(AOCL_DTL_LEVEL_TRACE_1)
    /* Finalize BLIS. */
    //bli_finalize_auto();
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

GENTFUNCSCAL( scomplex, float, c, s, s, scal, scalv )
