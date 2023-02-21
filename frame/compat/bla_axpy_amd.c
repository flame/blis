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

  // This function is invoked on all architectures including ‘generic’.
  // Non-AVX platforms will use the kernels derived from the context.
  if (bli_cpuid_is_avx_supported() == TRUE)
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

  // This function is invoked on all architectures including ‘generic’.
  // Non-AVX platforms will use the kernels derived from the context.
  if (bli_cpuid_is_avx_supported() == TRUE)
  {
#ifdef BLIS_ENABLE_OPENMP
        // For sizes less than 100, optimal number of threads is 1, but
        // due to the overhead of calling omp functions it is being done outside
        // by directly calling daxpyv so as to get maximum performance.
        if ( n0 <= 100 )
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
		rntm_t rntm_local;
		bli_rntm_init_from_global( &rntm_local );
            	dim_t nt = bli_rntm_num_threads( &rntm_local );
            	if (nt <= 0)
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

	    // Below tuning is based on Empirical Data collected on Genoa.
	    // On Milan, perf. vs number of threads is similar, but boundaries might vary slightly.

            if      ( n0 <= 10000 ) nt_ideal = 2;
            else if ( n0 <= 250000 ) nt_ideal = 8;
            else if ( n0 <= 750000 ) nt_ideal = 16;
            else if ( n0 <= 2000000 ) nt_ideal = 32;
            else                    nt_ideal = nt;

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
                    dim_t npt, offsetx0, offsety0;
                    if ( t_id < ( nt - n_elem_rem ) )
                    {
                        npt = n_elem_per_thrd;
			// Stride can be non-unit for both x and y vectors.
                        offsetx0 = t_id * npt * incx0;
			offsety0 = t_id * npt * incy0;
                    }
                    else
                    {
                        npt = n_elem_per_thrd + 1;
			// Stride can be non-unit for both x and y vectors.
                        offsetx0 = ( ( t_id * n_elem_per_thrd ) +
                                  ( t_id - ( nt - n_elem_rem ) ) ) * incx0;
			offsety0 = ( ( t_id * n_elem_per_thrd ) +
                                  ( t_id - ( nt - n_elem_rem ) ) ) * incy0;
                    }
		    bli_daxpyv_zen_int10
		    (
			BLIS_NO_CONJUGATE,
			npt,
			(double*)alpha,
			x0 + offsetx0, incx0,
			y0 + offsety0, incy0,
			NULL
		    );
                }
            }
	}
#else //BLIS_ENABLE_OPENMP

			bli_daxpyv_zen_int10
			  (
				BLIS_NO_CONJUGATE,
				n0,
				(double*)alpha,
				x0, incx0,
				y0, incy0,
				NULL
			  );

#endif //BLIS_ENABLE_OPENMP
  }
  else //if (bli_cpuid_is_avx_supported() == TRUE)
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

  } // if (bli_cpuid_is_avx_supported() == TRUE)

  AOCL_DTL_TRACE_EXIT(AOCL_DTL_LEVEL_TRACE_1);
  /* Finalize BLIS. */
  //    bli_finalize_auto();
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

  // This function is invoked on all architectures including ‘generic’.
  // Non-AVX platforms will use the kernels derived from the context.
  if (bli_cpuid_is_avx_supported() == TRUE)
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

  // This function is invoked on all architectures including ‘generic’.
  // Non-AVX platforms will use the kernels derived from the context.
  if (bli_cpuid_is_avx_supported() == TRUE)
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
