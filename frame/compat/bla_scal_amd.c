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

	if (*n == 0 || alpha == NULL) {
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

    // This function is invoked on all architectures including ‘generic’.
    // Non-AVX2+FMA3 platforms will use the kernels derived from the context.
    if (bli_cpuid_is_avx2fma3_supported() == TRUE) {
	    bli_sscalv_zen_int10
		    (
		     BLIS_NO_CONJUGATE,
		     n0,
		     (float *)alpha,
		     x0, incx0,
		     NULL
		    );
    }
    else{
	    PASTEMAC2(s,scalv,BLIS_TAPI_EX_SUF) \
		    ( \
		      BLIS_NO_CONJUGATE,\
		      n0, \
		      (float *)alpha,\
		      x0, incx0,\
		      NULL, \
		      NULL  \
		    );\
    }

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
    dim_t  n0;
    double* x0;
    inc_t  incx0;

    /* Initialize BLIS  */
    //bli_init_auto();

    /* Convert typecast negative values of n to zero. */
    if ( *n < 0 ) n0 = ( dim_t )0;
    else          n0 = ( dim_t )(*n);

    if (*n == 0 || alpha == NULL) {
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

        x0    = (x) + (n0-1)*(-*incx);
        incx0 = ( inc_t )(*incx);

    }
    else
    {
        x0    = (x);
        incx0 = ( inc_t )(*incx);
    }

    // This function is invoked on all architectures including ‘generic’.
    // Non-AVX2+FMA3 platforms will use the kernels derived from the context.
    if (bli_cpuid_is_avx2fma3_supported() == TRUE){
#ifdef BLIS_ENABLE_OPENMP
        // For sizes less than 10000, optimal number of threads is 1, but
        // due to the overhead of calling omp functions it is being done outside
        // by directly calling dscalv so as to get maximum performance.
        if ( n0 <= 10000 )
        {
            bli_dscalv_zen_int10
            (
              BLIS_NO_CONJUGATE,
              n0,
              (double*) alpha,
              x0, incx0,
              NULL
            );
        }
        else
        {
            rntm_t rntm_local;
            bli_rntm_init_from_global( &rntm_local );
            dim_t nt = bli_rntm_num_threads( &rntm_local );

            if (nt<=0)
            {
                // nt is less than one if BLIS manual setting of parallelism
                // has been used. Parallelism here will be product of values.
                dim_t jc, pc, ic, jr, ir;
	        jc = bli_rntm_jc_ways( &rntm_local );
	        pc = bli_rntm_pc_ways( &rntm_local );
	        ic = bli_rntm_ic_ways( &rntm_local );
	        jr = bli_rntm_jr_ways( &rntm_local );
	        ir = bli_rntm_ir_ways( &rntm_local );
                nt = jc*pc*ic*jr*ir;
            }

#ifdef AOCL_DYNAMIC
            dim_t nt_ideal;

            if      ( n0 <= 20000 ) nt_ideal = 2;
            else if ( n0 <= 50000 ) nt_ideal = 4;
            else                    nt_ideal = 8;

            nt = bli_min( nt_ideal, nt );
#endif

            dim_t n_elem_per_thrd = n0 / nt;
            dim_t n_elem_rem = n0 % nt;

            _Pragma( "omp parallel num_threads(nt)" )
            {
                // Getting the actual number of threads that are spawned.
                dim_t nt_real = omp_get_num_threads();
                dim_t t_id = omp_get_thread_num();

                // The actual number of threads spawned might be different
                // from the predicted number of threads for which this parallel
                // region is being generated. Thus, in such a case we are
                // falling back to the Single-Threaded call.
                if ( nt_real != nt )
                {
                    // More than one thread can still be spawned but since we
                    // are falling back to the ST call, we are
                    // calling the kernel from thread 0 only.
                    if ( t_id == 0 )
                    {
                        bli_dscalv_zen_int10
                        (
                          BLIS_NO_CONJUGATE,
                          n0,
                          (double*) alpha,
                          x0, incx0,
                          NULL
                        );
                    }
                }
                else
                {
                    // The following conditions handle the optimal distribution of
                    // load among the threads.
                    // Say we have n0 = 50 & nt = 4.
                    // So we get 12 ( n0 / nt ) elements per thread along with 2
                    // remaining elements. Each of these remaining elements is given
                    // to the last threads, respectively.
                    // So, t0, t1, t2 and t3 gets 12, 12, 13 and 13 elements,
                    // respectively.
                    dim_t npt, offset;

                    if ( t_id < ( nt - n_elem_rem ) )
                    {
                        npt = n_elem_per_thrd;
                        offset = t_id * npt * incx0;
                    }
                    else
                    {
                        npt = n_elem_per_thrd + 1;
                        offset = ( ( t_id * n_elem_per_thrd ) +
                                  ( t_id - ( nt - n_elem_rem ) ) ) * incx0;
                    }

                    bli_dscalv_zen_int10
                    (
                      BLIS_NO_CONJUGATE,
                      npt,
                      (double*) alpha,
                      x0 + offset, incx0,
                      NULL
                    );
                }
            }
        }
#else
        // Default call to dscalv for single-threaded work
        bli_dscalv_zen_int10
        (
          BLIS_NO_CONJUGATE,
          n0,
          (double*) alpha,
          x0, incx0,
          NULL
        );
#endif
    }
    else
    {
        PASTEMAC2(d,scalv,BLIS_TAPI_EX_SUF) \
          ( \
            BLIS_NO_CONJUGATE,\
            n0, \
            (double *)alpha,\
            x0, incx0,\
            NULL, \
            NULL  \
          );\
    }

    /* Finalize BLIS. */
//    bli_finalize_auto();
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
    dim_t  n0;
    dcomplex* x0;
    inc_t  incx0;
    /* Initialize BLIS. */
    //bli_init_auto();

    /* Convert/typecast negative values of n to zero. */
    if ( *n < 0 ) n0 = ( dim_t )0;
    else          n0 = ( dim_t )(*n);

    if (*n == 0 || alpha == NULL) {
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

        x0    = (x) + (n0-1)*(-*incx);
        incx0 = ( inc_t )(*incx);
    }
    else
    {
        x0    = (x);
        incx0 = ( inc_t )(*incx);
    }

    // This function is invoked on all architectures including ‘generic’.
    // Non-AVX2+FMA3 platforms will use the kernels derived from the context.
    if ( bli_cpuid_is_avx2fma3_supported() == TRUE )
    {
#ifdef BLIS_ENABLE_OPENMP
        // For sizes less than 10000, optimal number of threads is 1, but
        // due to the overhead of calling omp functions it is being done outside
        // by directly calling dscalv so as to get maximum performance.
        if ( n0 <= 10000 )
        {
            bli_zdscalv_zen_int10
            (
              BLIS_NO_CONJUGATE,
              n0,
              (double*) alpha,
              x0, incx0,
              NULL
            );
        }
        else
        {
            rntm_t rntm_local;
            bli_rntm_init_from_global( &rntm_local );
            dim_t nt = bli_rntm_num_threads( &rntm_local );

            if (nt<=0)
            {
                // nt is less than one if BLIS manual setting of parallelism
                // has been used. Parallelism here will be product of values.
                dim_t jc, pc, ic, jr, ir;
	        jc = bli_rntm_jc_ways( &rntm_local );
	        pc = bli_rntm_pc_ways( &rntm_local );
	        ic = bli_rntm_ic_ways( &rntm_local );
	        jr = bli_rntm_jr_ways( &rntm_local );
	        ir = bli_rntm_ir_ways( &rntm_local );
                nt = jc*pc*ic*jr*ir;
            }

#ifdef AOCL_DYNAMIC
            dim_t nt_ideal;

            if      ( n0 <= 20000 )   nt_ideal = 4;
            else if ( n0 <= 1000000 ) nt_ideal = 8;
            else if ( n0 <= 2500000 ) nt_ideal = 12;
            else if ( n0 <= 5000000 ) nt_ideal = 32;
            else                      nt_ideal = 64;

            nt = bli_min( nt_ideal, nt );
#endif
            dim_t n_elem_per_thread = n0 / nt;
            dim_t n_elem_rem = n0 % nt;

            _Pragma( "omp parallel num_threads(nt)" )
            {
                // Getting the actual number of threads that are spawned.
                dim_t nt_real = omp_get_num_threads();
                dim_t t_id = omp_get_thread_num();

                // The actual number of threads spawned might be different
                // from the predicted number of threads for which this parallel
                // region is being generated. Thus, in such a case we are
                // falling back to the Single-Threaded call.
                if ( nt_real != nt )
                {
                    // More than one thread can still be spawned but since we
                    // are falling back to the ST call, we are
                    // calling the kernel from thread 0 only.
                    if ( t_id == 0 )
                    {
                        bli_zdscalv_zen_int10
                        (
                          BLIS_NO_CONJUGATE,
                          n0,
                          (double*) alpha,
                          x0, incx0,
                          NULL
                        );
                    }
                }
                else
                {
                    // The following conditions handle the optimal distribution of
                    // load among the threads.
                    // Say we have n0 = 50 & nt = 4.
                    // So we get 12 ( n0 / nt ) elements per thread along with 2
                    // remaining elements. Each of these remaining elements is given
                    // to the last threads, respectively.
                    // So, t0, t1, t2 and t3 gets 12, 12, 13 and 13 elements,
                    // respectively.
                    dim_t npt, offset;

                    if ( t_id < ( nt - n_elem_rem ) )
                    {
                        npt = n_elem_per_thread;
                        offset = t_id * npt * incx0;
                    }
                    else
                    {
                        npt = n_elem_per_thread + 1;
                        offset = ( ( t_id * n_elem_per_thread ) +
                                  ( t_id - ( nt - n_elem_rem ) ) ) * incx0;
                    }

                    bli_zdscalv_zen_int10
                    (
                      BLIS_NO_CONJUGATE,
                      npt,
                      (double *) alpha,
                      x0 + offset, incx0,
                      NULL
                    );
                }
            }
        }
#else
        // Default call to zdscalv for single-threaded work
        bli_zdscalv_zen_int10
        (
          BLIS_NO_CONJUGATE,
          n0,
          (double *) alpha,
          x0, incx0,
          NULL
        );
#endif
    }
    else
    {
        // Sub-optimal implementation for zdscal
        // by casting alpha to the double complex domain and
        // calling the zscal
        dcomplex  alpha_cast;
        PASTEMAC2(d,z,copys)( *alpha, alpha_cast );

        /* Call BLIS interface. */ \
        PASTEMAC2(z,scalv,BLIS_TAPI_EX_SUF) \
        ( \
          BLIS_NO_CONJUGATE, \
          n0, \
          &alpha_cast, \
          x0, incx0, \
          NULL, \
          NULL  \
        ); \
    }

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

  cntx_t* cntx;

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
    alpha,
    x0, incx0,
    NULL
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

