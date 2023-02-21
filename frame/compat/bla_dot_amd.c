/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2014, The University of Texas at Austin
   Copyright (C) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

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
ftype PASTEF772S(ch,blasname,chc) \
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
}\
\
IF_BLIS_ENABLE_BLAS(\
ftype PASTEF772(ch,blasname,chc) \
     ( \
       const f77_int* n, \
       const ftype*   x, const f77_int* incx, \
       const ftype*   y, const f77_int* incy  \
     ) \
{ \
  return PASTEF772S(ch,blasname,chc)( n, x, incx, y, incy );\
} \
)

float sdot_blis_impl
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
#ifdef BLIS_ENABLE_BLAS
float sdot_
     (
       const f77_int* n,
       const float*   x, const f77_int* incx,
       const float*   y, const f77_int* incy
     )
{
  return sdot_blis_impl( n, x, incx, y, incy );
}
#endif
double ddot_blis_impl
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
    double  rho = 0.0;

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
#ifdef BLIS_ENABLE_OPENMP
        // For sizes less than or equal to 2500, optimal number of threads is 1,
        // but due to the overhead of calling omp functions it is being done
        // outside by directly calling ddotv so as to get maximum performance.
        if ( n0 <= 2500 )
        {
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
            rntm_t rntm;
            double* rho_temp = NULL;
            dim_t nt, n0_per_thread, n0_rem, nt_pred;
            dim_t i;
            rho = 0;
            // Initialize a local runtime with global settings.
            bli_rntm_init_from_global(&rntm);

            // Query the total number of threads from the rntm_t object.
            nt = bli_rntm_num_threads(&rntm);

            if (nt<=0)
            {
                // nt is less than one if BLIS manual setting of parallelism
                // has been used. Parallelism here will be product of values.
                dim_t jc, pc, ic, jr, ir;
	        jc = bli_rntm_jc_ways( &rntm );
	        pc = bli_rntm_pc_ways( &rntm );
	        ic = bli_rntm_ic_ways( &rntm );
	        jr = bli_rntm_jr_ways( &rntm );
	        ir = bli_rntm_ir_ways( &rntm );
                nt = jc*pc*ic*jr*ir;
            }

            mem_t local_mem_buf = { 0 };

            bli_membrk_rntm_set_membrk(&rntm);
            siz_t buffer_size = bli_pool_block_size(bli_membrk_pool(
                bli_packbuf_index(BLIS_BITVAL_BUFFER_FOR_A_BLOCK),
                bli_rntm_membrk(&rntm)));

            if ( (nt * sizeof(double)) > buffer_size )
                return BLIS_NOT_YET_IMPLEMENTED;

            bli_membrk_acquire_m(&rntm,
                buffer_size,
                BLIS_BITVAL_BUFFER_FOR_A_BLOCK,
                &local_mem_buf);
            if ( FALSE == bli_mem_is_alloc(&local_mem_buf) )
                return BLIS_NULL_POINTER;
            rho_temp = bli_mem_buffer(&local_mem_buf);
            if ( NULL == rho_temp ) return BLIS_NULL_POINTER;

            // Initializing rho_temp array to 0
            for ( i = 0; i < nt; i++ )
            {
                rho_temp[i] = 0;
            }

#ifdef AOCL_DYNAMIC
            // Calculate the optimal number of threads required
            // based on input dimension. These conditions are taken
            // after cheking the performance for range of dimensions
            // and number of threads
            if ( n0 <= 5000 ) nt_pred = 4;
            else if ( n0 <= 15000 ) nt_pred = 8;
            else if ( n0 <= 40000 ) nt_pred = 16;
            else if ( n0 <= 200000 ) nt_pred = 32;
            else nt_pred = nt;
            nt = bli_min(nt_pred, nt);
#endif
            // Calculating the input sizes per thread
            n0_per_thread = n0 / nt;
            n0_rem = n0 % nt;

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
                        bli_ddotv_zen_int10
                        (
                          BLIS_NO_CONJUGATE,
                          BLIS_NO_CONJUGATE,
                          n0,
                          x0, incx0,
                          y0, incy0,
                          rho_temp,
                          NULL
                        );
                    }
                }
                else
                {
                    // The following conditions handle the optimal distribution
                    // of load among the threads.
                    // Say we have n0 = 50 & nt = 4.
                    // So we get 12 ( n0 / nt ) elements per thread along with 2
                    // remaining elements. Each of these remaining elements is
                    // given to the last threads, respectively.
                    // So, t0, t1, t2 and t3 gets 12, 12, 13 and 13 elements,
                    // respectively.
                    dim_t npt, offset;
                    if ( t_id < ( nt - n0_rem ) )
                    {
                        npt = n0_per_thread;
                        offset = t_id * npt;
                    }
                    else
                    {
                        npt = n0_per_thread + 1;
                        offset = ( ( t_id * n0_per_thread ) +
                                ( t_id - ( nt - n0_rem ) ) );
                    }
                    bli_ddotv_zen_int10
                    (
                      BLIS_NO_CONJUGATE,
                      BLIS_NO_CONJUGATE,
                      npt,
                      x0 + ( offset * incx0 ), incx0,
                      y0 + ( offset * incy0 ), incy0,
                      rho_temp + t_id,
                      NULL
                    );
                }
            }

            // Accumulating the nt thread outputs to rho
            for ( i = 0; i < nt; i++ )
                rho += rho_temp[i];

            // Releasing the allocated memory
            bli_membrk_release(&rntm, &local_mem_buf);
        }
#else
        // Default call to ddotv for single-threaded work
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
#endif
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
#ifdef BLIS_ENABLE_BLAS
double ddot_
     (
       const f77_int* n,
       const double*   x, const f77_int* incx,
       const double*   y, const f77_int* incy
     )
{
  return ddot_blis_impl( n, x, incx, y, incy );
}
#endif

#ifdef BLIS_DISABLE_COMPLEX_RETURN_INTEL
scomplex cdotu_blis_impl
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
#ifdef BLIS_ENABLE_BLAS
scomplex cdotu_
     (
       const f77_int* n,
       const scomplex*   x, const f77_int* incx,
       const scomplex*   y, const f77_int* incy
     )
{
  return cdotu_blis_impl( n, x, incx, y, incy );
}
#endif
dcomplex zdotu_blis_impl
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
#ifdef BLIS_ENABLE_BLAS
dcomplex zdotu_
     (
       const f77_int* n,
       const dcomplex*   x, const f77_int* incx,
       const dcomplex*   y, const f77_int* incy
     )
{
  return zdotu_blis_impl( n, x, incx, y, incy );
}
#endif
scomplex cdotc_blis_impl
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
#ifdef BLIS_ENABLE_BLAS
scomplex cdotc_
     (
       const f77_int* n,
       const scomplex*   x, const f77_int* incx,
       const scomplex*   y, const f77_int* incy
     )
{
  return cdotc_blis_impl( n, x, incx, y, incy );
}
#endif
dcomplex zdotc_blis_impl
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
#ifdef BLIS_ENABLE_BLAS
dcomplex zdotc_
     (
       const f77_int* n,
       const dcomplex*   x, const f77_int* incx,
       const dcomplex*   y, const f77_int* incy
     )
{
  return zdotc_blis_impl( n, x, incx, y, incy );
}
#endif

#else // BLIS_DISABLE_COMPLEX_RETURN_INTEL
// For the "intel" complex return type, use a hidden parameter to return the result
#undef  GENTFUNCDOT
#define GENTFUNCDOT( ftype, ch, chc, blis_conjx, blasname, blisname ) \
\
void PASTEF772S(ch,blasname,chc) \
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
}\
\
IF_BLIS_ENABLE_BLAS(\
void PASTEF772(ch,blasname,chc) \
     ( \
       ftype*         rhop, \
       const f77_int* n, \
       const ftype*   x, const f77_int* incx, \
       const ftype*   y, const f77_int* incy  \
     ) \
{ \
  PASTEF772S(ch,blasname,chc)( rhop, n, x, incx, y, incy );\
} \
)

INSERT_GENTFUNCDOTC_BLAS( dot, dotv )
#endif // BLIS_DISABLE_COMPLEX_RETURN_INTEL

// -- "Black sheep" dot product function definitions --

// Input vectors stored in single precision, computed in double precision,
// with result returned in single precision.
float PASTEF77S(sd,sdot)
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
             PASTEF77S(d,sdot)
             (
               n,
               x, incx,
               y, incy
             )
           );
}
#ifdef BLIS_ENABLE_BLAS
float PASTEF77(sd,sdot)
     (
       const f77_int* n,
       const float*   sb,
       const float*   x, const f77_int* incx,
       const float*   y, const f77_int* incy
     )
{
  return PASTEF77S(sd,sdot)( n,sb, x, incx, y, incy );
}
#endif

// Input vectors stored in single precision, computed in double precision,
// with result returned in double precision.
double PASTEF77S(d,sdot)
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

#ifdef BLIS_ENABLE_BLAS
double PASTEF77(d,sdot)
     (
       const f77_int* n,
       const float*   x, const f77_int* incx,
       const float*   y, const f77_int* incy
     )
{
  return PASTEF77S(d,sdot)( n, x, incx, y, incy );
}
#endif // BLIS_ENABLE_BLAS
