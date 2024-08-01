/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2014, The University of Texas at Austin
   Copyright (C) 2018 - 2024, Advanced Micro Devices, Inc. All rights reserved.

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

  NaN propagation expectation
  --------------------------

  1. Always propagate
*/

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
    AOCL_DTL_LOG_DOTV_INPUTS(AOCL_DTL_LEVEL_TRACE_1, *MKSTR(ch), *MKSTR(blis_conjx), *n, *incx, *incy); \
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
    AOCL_DTL_LOG_DOTV_INPUTS(AOCL_DTL_LEVEL_TRACE_1, 'S', 'N', *n, *incx, *incy);
    dim_t  n0;
    float* x0;
    float* y0;
    inc_t  incx0;
    inc_t  incy0;
    float  rho;

    /* Initialize BLIS. */
    //  bli_init_auto();

    // If the vector dimension is less than or equal to zero, return.
    if (*n <= 0)
    {
      rho = 0.0f;

      AOCL_DTL_TRACE_EXIT(AOCL_DTL_LEVEL_TRACE_1);
      return rho;
    }
    else
    {
      n0 = ( dim_t )(*n);
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

    cntx_t *cntx = NULL;

    // Query the architecture ID
    arch_t arch_id = bli_arch_query_id();

    /*
      Function pointer declaration for the function
      that will be used by this API
    */
    sdotv_ker_ft dotv_ker_ptr; // SDOTV

    // Pick the kernel based on the architecture ID
    switch (arch_id)
    {
        case BLIS_ARCH_ZEN5:
        case BLIS_ARCH_ZEN4:
#if defined(BLIS_KERNELS_ZEN4)

            // AVX-512 Kernel
            dotv_ker_ptr = bli_sdotv_zen_int_avx512;

        break;
#endif
        case BLIS_ARCH_ZEN:
        case BLIS_ARCH_ZEN2:
        case BLIS_ARCH_ZEN3:

            // AVX-2 Kernel
            dotv_ker_ptr = bli_sdotv_zen_int10;

            break;
        default:

            // For non-Zen architectures, query the context
            cntx = bli_gks_query_cntx();

            // Query the context for the kernel function pointers for sdotv
            dotv_ker_ptr = bli_cntx_get_l1v_ker_dt(BLIS_FLOAT, BLIS_DOTV_KER, cntx);
    }

    dotv_ker_ptr
    (
      BLIS_NO_CONJUGATE,
      BLIS_NO_CONJUGATE,
      n0,
      x0, incx0,
      y0, incy0,
      &rho,
      cntx
    );

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
    AOCL_DTL_LOG_DOTV_INPUTS(AOCL_DTL_LEVEL_TRACE_1, 'D', 'N', *n, *incx, *incy);
    dim_t  n_elem;
    double* x0;
    double* y0;
    inc_t  incx0;
    inc_t  incy0;
    double  rho = 0.0;

    // BLAS Exception: Return early when n <= 0.
    if((*n) <= 0)
    {
        AOCL_DTL_TRACE_EXIT(AOCL_DTL_LEVEL_TRACE_1)
        return 0.0;
    }
    else
    {
        n_elem = ( dim_t )(*n);
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

        x0    = ((double*)x) + (n_elem-1)*(-*incx);
        incx0 = ( inc_t )(*incx);

    }
    else
    {
        x0    = ((double*)x);
        incx0 = ( inc_t )(*incx);
    }

    if ( *incy < 0 )
    {
        y0    = ((double*)y) + (n_elem-1)*(-*incy);
        incy0 = ( inc_t )(*incy);

    }
    else
    {
        y0    = ((double*)y);
        incy0 = ( inc_t )(*incy);
    }

     // Definition of function pointer
    ddotv_ker_ft dotv_ker_ptr;

    cntx_t *cntx = NULL;

    // Query the architecture ID
    arch_t arch_id_local = bli_arch_query_id();

    // Pick the kernel based on the architecture ID
    switch (arch_id_local)
    {
      case BLIS_ARCH_ZEN5:
      case BLIS_ARCH_ZEN4:
#if defined(BLIS_KERNELS_ZEN4)

        // AVX-512 Kernel
        dotv_ker_ptr = bli_ddotv_zen_int_avx512;
        break;
#endif
      case BLIS_ARCH_ZEN:
      case BLIS_ARCH_ZEN2:
      case BLIS_ARCH_ZEN3:

          // AVX2 Kernel
          dotv_ker_ptr = bli_ddotv_zen_int10;
          break;

      default:

          // Query the context
          cntx = bli_gks_query_cntx();

          // Query the function pointer using the context
          dotv_ker_ptr = bli_cntx_get_l1v_ker_dt(BLIS_DOUBLE, BLIS_DOTV_KER, cntx);
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
      BLIS_DOTV_KER,
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
        dotv_ker_ptr
        (
          BLIS_NO_CONJUGATE,
          BLIS_NO_CONJUGATE,
          n_elem,
          x0, incx0,
          y0, incy0,
          &rho,
          cntx
        );

        AOCL_DTL_TRACE_EXIT(AOCL_DTL_LEVEL_TRACE_1)

        return rho;
#ifdef BLIS_ENABLE_OPENMP
    }

    /*
      Here we know that more than one thread needs to be spawned.

      In such a case, each thread will need its own rho value to
      do the accumulation. These temporary rho's will be accumulated
      in the end.
    */
    rntm_t rntm;
    mem_t mem_buf_rho;
    double *rho_temp = NULL;
    rho = 0.0;

    /*
      Initialize mem pool buffer to NULL and size to 0
      "buf" and "size" fields are assigned once memory
      is allocated from the pool in bli_pba_acquire_m().
      This will ensure bli_mem_is_alloc() will be passed on
      an allocated memory if created or a NULL .
    */
    mem_buf_rho.pblk.buf = NULL;
    mem_buf_rho.pblk.block_size = 0;
    mem_buf_rho.buf_type = 0;
    mem_buf_rho.size = 0;
    mem_buf_rho.pool = NULL;

    /*
        In order to get the buffer from pool via rntm access to
        memory broker is needed.Following are initializations
        for rntm
    */
    bli_rntm_init_from_global(&rntm);
    bli_rntm_set_num_threads_only(1, &rntm);
    bli_pba_rntm_set_pba(&rntm);

    // Calculate the size required for rho buffer.
    size_t buffer_size = nt * sizeof(double);

#ifdef BLIS_ENABLE_MEM_TRACING
    printf("bli_ddotv_unf_var1(): get mem pool block\n");
#endif

    /*
      Acquire a buffer (nt * size(double)) from the memory broker
      and save the associated mem_t entry to mem_buf_rho.
    */
    bli_pba_acquire_m(&rntm,
                         buffer_size,
                         BLIS_BITVAL_BUFFER_FOR_A_BLOCK,
                         &mem_buf_rho);

    /* Continue if rho buffer memory is allocated*/
    if ((bli_mem_is_alloc(&mem_buf_rho)))
    {
        rho_temp = bli_mem_buffer(&mem_buf_rho);

        /*
          This is done to handle cases when the
          number of threads launched is not equal
          to the number of threads requested. In
          such cases, the garbage value in the created
          buffer will not be overwritten by valid values.

          This will ensure that garbage value will
          not get accumulated with the final result.
        */
        for (dim_t i = 0; i < nt; i++)
          rho_temp[i] = 0.0;
    }
    else
    {
      dotv_ker_ptr
        (
          BLIS_NO_CONJUGATE,
          BLIS_NO_CONJUGATE,
          n_elem,
          x0, incx0,
          y0, incy0,
          &rho,
          cntx
        );

        AOCL_DTL_TRACE_EXIT(AOCL_DTL_LEVEL_TRACE_1)
        return rho;
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
        dotv_ker_ptr
        (
          BLIS_NO_CONJUGATE,
          BLIS_NO_CONJUGATE,
          length,
          x_thread_local, incx0,
          y_thread_local, incy0,
          rho_temp + thread_id,
          cntx
        );
    }

    /*
      Accumulate the values in rho_temp only when mem is allocated.
      When the memory cannot be allocated rho_temp will point to
      rho
    */
    if (bli_mem_is_alloc(&mem_buf_rho))
    {
        // Accumulating the nt thread outputs to rho
        for (dim_t i = 0; i < nt; i++)
          rho += rho_temp[i];

        // Releasing the allocated memory if it was allocated
        bli_pba_release(&rntm, &mem_buf_rho);
    }
#endif

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
    AOCL_DTL_LOG_DOTV_INPUTS(AOCL_DTL_LEVEL_TRACE_1, 'C', 'N', *n, *incx, *incy);
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
    else          n0 = ( dim_t )(*n);

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

    PASTEMAC(z,set0s)( rho );   // Initializing rho to 0.

    AOCL_DTL_TRACE_ENTRY(AOCL_DTL_LEVEL_TRACE_1);
    AOCL_DTL_LOG_DOTV_INPUTS(AOCL_DTL_LEVEL_TRACE_1, 'Z', 'N', *n, *incx, *incy);

    /* Initialize BLIS. */
    //  bli_init_auto();

    /* Convert/typecast negative values of n to zero. */
    if ( *n < 0 ) n0 = ( dim_t )0;
    else          n0 = ( dim_t )(*n);

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

    cntx_t *cntx = NULL;

    // Query the architecture ID
    arch_t arch_id_local = bli_arch_query_id();
    zdotv_ker_ft zdotv_ker_ptr;

    switch ( arch_id_local )
    {
        case BLIS_ARCH_ZEN5:
        case BLIS_ARCH_ZEN4:
#if defined(BLIS_KERNELS_ZEN4)
            zdotv_ker_ptr = bli_zdotv_zen_int_avx512;
            break;
#endif

        case BLIS_ARCH_ZEN3:
        case BLIS_ARCH_ZEN2:
        case BLIS_ARCH_ZEN:
            zdotv_ker_ptr = bli_zdotv_zen_int5;
            break;

        default:
            // For non-Zen architectures, query the context
            cntx = bli_gks_query_cntx();

            // Query the context for the kernel function pointers for zdotv
            zdotv_ker_ptr = bli_cntx_get_l1v_ker_dt(BLIS_DCOMPLEX, BLIS_DOTV_KER, cntx);
            break;
    }

#ifdef BLIS_ENABLE_OPENMP
    // Initialize number of threads to one.
    dim_t nt = 1;

    bli_nthreads_l1
    (
      BLIS_DOTV_KER,
      BLIS_DCOMPLEX,
      BLIS_DCOMPLEX,
      arch_id_local,
      n0,
      &nt
    );

    /*
      If the number of optimum threads is 1, the OpenMP overhead
      is avoided by calling the function directly
    */
    if (nt == 1)
    {
#endif
        zdotv_ker_ptr
        (
          BLIS_NO_CONJUGATE,
          BLIS_NO_CONJUGATE,
          n0,
          x0, incx0,
          y0, incy0,
          &rho,
          cntx
        );

        AOCL_DTL_TRACE_EXIT(AOCL_DTL_LEVEL_TRACE_1)

        return rho;
#ifdef BLIS_ENABLE_OPENMP
    }

    /*
      Here we know that more than one thread needs to be spawned.

      In such a case, each thread will need its own rho value to
      do the accumulation. These temporary rho's will be accumulated
      in the end.
    */
    rntm_t rntm_l;
    bli_rntm_init_from_global( &rntm_l );

    dcomplex *rho_temp = NULL;

    /*
      Initialize mem pool buffer to NULL and size to 0
      "buf" and "size" fields are assigned once memory
      is allocated from the pool in bli_pba_acquire_m().
      This will ensure bli_mem_is_alloc() will be passed on
      an allocated memory if created or a NULL .
    */
    mem_t mem_buf_rho;
    mem_buf_rho.pblk.buf = NULL;
    mem_buf_rho.pblk.block_size = 0;
    mem_buf_rho.buf_type = 0;
    mem_buf_rho.size = 0;
    mem_buf_rho.pool = NULL;

    /*
        In order to get the buffer from pool via rntm access to
        memory broker is needed.Following are initializations
        for rntm.
    */
    bli_rntm_set_num_threads_only(1, &rntm_l);
    bli_pba_rntm_set_pba(&rntm_l);

    // Calculate the size required for rho buffer.
    size_t buffer_size = nt * sizeof(dcomplex);

#ifdef BLIS_ENABLE_MEM_TRACING
    printf("bli_zdotu(): get mem pool block\n");
#endif

    /*
      Acquire a buffer (nt * size(dcomplex)) from the memory broker
      and save the associated mem_t entry to mem_buf_rho.
    */
    bli_pba_acquire_m
    (
      &rntm_l,
      buffer_size,
      BLIS_BITVAL_BUFFER_FOR_A_BLOCK,
      &mem_buf_rho
    );

    /* Continue if rho buffer memory is allocated*/
    if ( bli_mem_is_alloc( &mem_buf_rho ) )
    {
        rho_temp = bli_mem_buffer( &mem_buf_rho );

        /*
          Initializing rho_temp buffer to zeros.

          This is done to handle cases when the
          number of threads launched is not equal
          to the number of threads requested. In
          such cases, the garbage value in the created
          buffer will not be overwritten by valid values.

          This will ensure that garbage values will
          not get accumulated with the final result.
        */
        for ( dim_t i = 0; i < nt; ++i )
            PASTEMAC(z,set0s)( *(rho_temp + i) );
    }
    else
    {
      zdotv_ker_ptr
        (
          BLIS_NO_CONJUGATE,
          BLIS_NO_CONJUGATE,
          n0,
          x0, incx0,
          y0, incy0,
          &rho,
          cntx
        );

        AOCL_DTL_TRACE_EXIT(AOCL_DTL_LEVEL_TRACE_1)
        return rho;
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
          n0,
          nt_use,
          &start, &length,
          thread_id
        );

        // Adjust the local pointer for computation
        dcomplex *x_thread_local = x0 + (start * incx0);
        dcomplex *y_thread_local = y0 + (start * incy0);

        // Invoke the function based on the kernel function pointer
        zdotv_ker_ptr
        (
          BLIS_NO_CONJUGATE,
          BLIS_NO_CONJUGATE,
          length,
          x_thread_local, incx0,
          y_thread_local, incy0,
          rho_temp + thread_id,
          cntx
        );
    }

    /*
      Accumulate the values in rho_temp only when mem is allocated.
      When the memory cannot be allocated rho_temp will point to
      rho
    */
    if ( bli_mem_is_alloc( &mem_buf_rho ) )
    {
        // Accumulating the nt thread outputs to rho
        for ( dim_t i = 0; i < nt; ++i )
            PASTEMAC(z,adds)( *(rho_temp + i), rho );

        // Releasing the allocated memory if it was allocated
        bli_pba_release( &rntm_l, &mem_buf_rho );
    }
#endif

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
    AOCL_DTL_LOG_DOTV_INPUTS(AOCL_DTL_LEVEL_TRACE_1, 'C', 'C', *n, *incx, *incy);

    /* Initialize BLIS. */
    //  bli_init_auto();

    /* Convert/typecast negative values of n to zero. */
    if ( *n < 0 ) n0 = ( dim_t )0;
    else          n0 = ( dim_t )(*n);

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
    AOCL_DTL_LOG_DOTV_INPUTS(AOCL_DTL_LEVEL_TRACE_1, 'Z', 'C', *n, *incx, *incy);
    dim_t  n0;
    dcomplex* x0;
    dcomplex* y0;
    inc_t  incx0;
    inc_t  incy0;
    dcomplex  rho;

    PASTEMAC(z,set0s)( rho );   // Initializing rho to 0.

    /* Initialize BLIS. */
    //  bli_init_auto();

    /* Convert/typecast negative values of n to zero. */
    if ( *n < 0 ) n0 = ( dim_t )0;
    else          n0 = ( dim_t )(*n);

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

    cntx_t *cntx = NULL;

    // Query the architecture ID
    arch_t arch_id_local = bli_arch_query_id();
    zdotv_ker_ft zdotv_ker_ptr;

    switch ( arch_id_local )
    {
        case BLIS_ARCH_ZEN5:
        case BLIS_ARCH_ZEN4:
#if defined(BLIS_KERNELS_ZEN4)
            // Currently only the AVX512 intrinsic kernel is enabled.
            zdotv_ker_ptr = bli_zdotv_zen_int_avx512;
            // zdotv_ker_ptr = bli_zdotv_zen4_asm_avx512;
            break;
#endif

        case BLIS_ARCH_ZEN3:
        case BLIS_ARCH_ZEN2:
        case BLIS_ARCH_ZEN:
            zdotv_ker_ptr = bli_zdotv_zen_int5;
            break;
        
        default:
            // For non-Zen architectures, query the context
            cntx = bli_gks_query_cntx();

            // Query the context for the kernel function pointers for zdotv
            zdotv_ker_ptr = bli_cntx_get_l1v_ker_dt(BLIS_DCOMPLEX, BLIS_DOTV_KER, cntx);
            break;
    }

#ifdef BLIS_ENABLE_OPENMP
    // Initialize number of threads to one.
    dim_t nt = 1;

    bli_nthreads_l1
    (
      BLIS_DOTV_KER,
      BLIS_DCOMPLEX,
      BLIS_DCOMPLEX,
      arch_id_local,
      n0,
      &nt
    );

    /*
      If the number of optimum threads is 1, the OpenMP overhead
      is avoided by calling the function directly
    */
    if (nt == 1)
    {
#endif
        zdotv_ker_ptr
        (
          BLIS_CONJUGATE,
          BLIS_NO_CONJUGATE,
          n0,
          x0, incx0,
          y0, incy0,
          &rho,
          cntx
        );

        AOCL_DTL_TRACE_EXIT(AOCL_DTL_LEVEL_TRACE_1)

        return rho;
#ifdef BLIS_ENABLE_OPENMP
    }

    /*
      Here we know that more than one thread needs to be spawned.

      In such a case, each thread will need its own rho value to
      do the accumulation. These temporary rho's will be accumulated
      in the end.
    */
    rntm_t rntm_l;
    bli_rntm_init_from_global( &rntm_l );

    dcomplex *rho_temp = NULL;

    /*
      Initialize mem pool buffer to NULL and size to 0
      "buf" and "size" fields are assigned once memory
      is allocated from the pool in bli_pba_acquire_m().
      This will ensure bli_mem_is_alloc() will be passed on
      an allocated memory if created or a NULL .
    */
    mem_t mem_buf_rho;
    mem_buf_rho.pblk.buf = NULL;
    mem_buf_rho.pblk.block_size = 0;
    mem_buf_rho.buf_type = 0;
    mem_buf_rho.size = 0;
    mem_buf_rho.pool = NULL;

    /*
        In order to get the buffer from pool via rntm access to
        memory broker is needed.Following are initializations
        for rntm.
    */
    bli_rntm_set_num_threads_only(1, &rntm_l);
    bli_pba_rntm_set_pba(&rntm_l);

    // Calculate the size required for rho buffer.
    size_t buffer_size = nt * sizeof(dcomplex);

#ifdef BLIS_ENABLE_MEM_TRACING
    printf("bli_zdotc(): get mem pool block\n");
#endif

    /*
      Acquire a buffer (nt * size(dcomplex)) from the memory broker
      and save the associated mem_t entry to mem_buf_rho.
    */
    bli_pba_acquire_m
    (
      &rntm_l,
      buffer_size,
      BLIS_BITVAL_BUFFER_FOR_A_BLOCK,
      &mem_buf_rho
    );

    /* Continue if rho buffer memory is allocated*/
    if ( bli_mem_is_alloc( &mem_buf_rho ) )
    {
        rho_temp = bli_mem_buffer( &mem_buf_rho );

        /*
          Initializing rho_temp buffer to zeros.

          This is done to handle cases when the
          number of threads launched is not equal
          to the number of threads requested. In
          such cases, the garbage value in the created
          buffer will not be overwritten by valid values.

          This will ensure that garbage values will
          not get accumulated with the final result.
        */
        for ( dim_t i = 0; i < nt; ++i )
            PASTEMAC(z,set0s)( *(rho_temp + i) );
    }
    else
    {
      zdotv_ker_ptr
        (
          BLIS_CONJUGATE,
          BLIS_NO_CONJUGATE,
          n0,
          x0, incx0,
          y0, incy0,
          &rho,
          cntx
        );

        AOCL_DTL_TRACE_EXIT(AOCL_DTL_LEVEL_TRACE_1)
        return rho;
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
          n0,
          nt_use,
          &start, &length,
          thread_id
        );

        // Adjust the local pointer for computation
        dcomplex *x_thread_local = x0 + (start * incx0);
        dcomplex *y_thread_local = y0 + (start * incy0);

        // Invoke the function based on the kernel function pointer
        zdotv_ker_ptr
        (
          BLIS_CONJUGATE,
          BLIS_NO_CONJUGATE,
          length,
          x_thread_local, incx0,
          y_thread_local, incy0,
          rho_temp + thread_id,
          cntx
        );
    }

    /*
      Accumulate the values in rho_temp only when mem is allocated.
      When the memory cannot be allocated rho_temp will point to
      rho
    */
    if ( bli_mem_is_alloc( &mem_buf_rho ) )
    {
        // Accumulating the nt thread outputs to rho
        for ( dim_t i = 0; i < nt; ++i )
            PASTEMAC(z,adds)( *(rho_temp + i), rho );

        // Releasing the allocated memory if it was allocated
        bli_pba_release( &rntm_l, &mem_buf_rho );
    }
#endif

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
  AOCL_DTL_LOG_DOTV_INPUTS(AOCL_DTL_LEVEL_TRACE_1, *MKSTR(ch), *MKSTR(blis_conjx), *n, *incx, *incy); \
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
    AOCL_DTL_LOG_DOTV_INPUTS(AOCL_DTL_LEVEL_TRACE_1, 'D', 'N', *n, *incx, *incy);
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
