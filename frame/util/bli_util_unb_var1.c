/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2014, The University of Texas at Austin
   Copyright (C) 2018 - 2023, Advanced Micro Devices, Inc. All rights reserved.

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
#include <fenv.h>

//
// Define BLAS-like interfaces with typed operands.
//

#undef  GENTFUNCR
#define GENTFUNCR( ctype, ctype_r, ch, chr, varname ) \
\
void PASTEMAC(ch,varname) \
     ( \
       dim_t    n, \
       ctype*   x, inc_t incx, \
       ctype_r* asum, \
       cntx_t*  cntx, \
       rntm_t*  rntm  \
     ) \
{ \
    ctype*   chi1; \
    ctype_r  chi1_r; \
    ctype_r  chi1_i; \
    ctype_r  absum; \
    dim_t    i; \
\
    /* Initialize the absolute sum accumulator to zero. */ \
    PASTEMAC(chr,set0s)( absum ); \
\
    for ( i = 0; i < n; ++i ) \
    { \
        chi1 = x + (i  )*incx; \
\
        /* Get the real and imaginary components of chi1. */ \
        PASTEMAC2(ch,chr,gets)( *chi1, chi1_r, chi1_i ); \
\
        /* Replace chi1_r and chi1_i with their absolute values. */ \
        chi1_r = bli_fabs( chi1_r ); \
        chi1_i = bli_fabs( chi1_i ); \
\
        /* Accumulate the real and imaginary components into absum. */ \
        PASTEMAC(chr,adds)( chi1_r, absum ); \
        PASTEMAC(chr,adds)( chi1_i, absum ); \
    } \
\
    /* Store the final value of absum to the output variable. */ \
    PASTEMAC(chr,copys)( absum, *asum ); \
}

INSERT_GENTFUNCR_BASIC0( asumv_unb_var1 )


#undef  GENTFUNCR
#define GENTFUNCR( ctype, ctype_r, ch, chr, varname ) \
\
void PASTEMAC(ch,varname) \
     ( \
       uplo_t  uploa, \
       dim_t   m, \
       ctype*  a, inc_t rs_a, inc_t cs_a, \
       cntx_t* cntx, \
       rntm_t* rntm  \
     ) \
{ \
    ctype_r* zeror = PASTEMAC(chr,0); \
    doff_t   diagoffa; \
\
    /* If the dimension is zero, return early. */ \
    if ( bli_zero_dim1( m ) ) return; \
\
    /* In order to avoid the main diagonal, we must nudge the diagonal either
       up or down by one, depending on which triangle is currently stored. */ \
    if        ( bli_is_upper( uploa ) )   diagoffa =  1; \
    else /*if ( bli_is_lower( uploa ) )*/ diagoffa = -1; \
\
    /* We will be reflecting the stored region over the diagonal into the
       unstored region, so a transposition is necessary. Furthermore, since
       we are creating a Hermitian matrix, we must also conjugate. */ \
    PASTEMAC2(ch,copym,BLIS_TAPI_EX_SUF) \
    ( \
      diagoffa, \
      BLIS_NONUNIT_DIAG, \
      uploa, \
      BLIS_CONJ_TRANSPOSE, \
      m, \
      m, \
      a, rs_a, cs_a, \
      a, rs_a, cs_a, \
      cntx, \
      rntm  \
    ); \
\
    /* Set the imaginary parts of the diagonal elements to zero. */ \
    PASTEMAC2(ch,setid,BLIS_TAPI_EX_SUF) \
    ( \
      0, \
      m, \
      m, \
      zeror, \
      a, rs_a, cs_a, \
      cntx, \
      rntm  \
    ); \
}

INSERT_GENTFUNCR_BASIC0( mkherm_unb_var1 )


#undef  GENTFUNC
#define GENTFUNC( ctype, ch, varname ) \
\
void PASTEMAC(ch,varname) \
     ( \
       uplo_t  uploa, \
       dim_t   m, \
       ctype*  a, inc_t rs_a, inc_t cs_a, \
       cntx_t* cntx, \
       rntm_t* rntm  \
     ) \
{ \
    doff_t  diagoffa; \
\
    /* If the dimension is zero, return early. */ \
    if ( bli_zero_dim1( m ) ) return; \
\
    /* In order to avoid the main diagonal, we must nudge the diagonal either
       up or down by one, depending on which triangle is currently stored. */ \
    if        ( bli_is_upper( uploa ) )   diagoffa =  1; \
    else /*if ( bli_is_lower( uploa ) )*/ diagoffa = -1; \
\
    /* We will be reflecting the stored region over the diagonal into the
       unstored region, so a transposition is necessary. */ \
    PASTEMAC2(ch,copym,BLIS_TAPI_EX_SUF) \
    ( \
      diagoffa, \
      BLIS_NONUNIT_DIAG, \
      uploa, \
      BLIS_TRANSPOSE, \
      m, \
      m, \
      a, rs_a, cs_a, \
      a, rs_a, cs_a, \
      cntx, \
      rntm  \
    ); \
}

INSERT_GENTFUNC_BASIC0( mksymm_unb_var1 )


#undef  GENTFUNC
#define GENTFUNC( ctype, ch, varname ) \
\
void PASTEMAC(ch,varname) \
     ( \
       uplo_t  uploa, \
       dim_t   m, \
       ctype*  a, inc_t rs_a, inc_t cs_a, \
       cntx_t* cntx, \
       rntm_t* rntm  \
     ) \
{ \
    ctype*  zero = PASTEMAC(ch,0); \
    doff_t  diagoffa; \
\
    /* If the dimension is zero, return early. */ \
    if ( bli_zero_dim1( m ) ) return; \
\
    /* Toggle uplo so that it refers to the unstored triangle. */ \
    bli_toggle_uplo( &uploa ); \
\
    /* In order to avoid the main diagonal, we must nudge the diagonal either
       up or down by one, depending on which triangle is to be zeroed. */ \
    if        ( bli_is_upper( uploa ) )   diagoffa =  1; \
    else /*if ( bli_is_lower( uploa ) )*/ diagoffa = -1; \
\
    /* Set the unstored triangle to zero. */ \
    PASTEMAC2(ch,setm,BLIS_TAPI_EX_SUF) \
    ( \
      BLIS_NO_CONJUGATE, \
      diagoffa, \
      BLIS_NONUNIT_DIAG, \
      uploa, \
      m, \
      m, \
      zero, \
      a, rs_a, cs_a, \
      cntx, \
      rntm  \
    ); \
}

INSERT_GENTFUNC_BASIC0( mktrim_unb_var1 )


#undef  GENTFUNCR
#define GENTFUNCR( ctype, ctype_r, ch, chr, varname ) \
\
void PASTEMAC(ch,varname) \
     ( \
       dim_t    n, \
       ctype*   x, inc_t incx, \
       ctype_r* norm, \
       cntx_t*  cntx, \
       rntm_t*  rntm  \
     ) \
{ \
    ctype*   chi1; \
    ctype_r  abs_chi1; \
    ctype_r  absum; \
    dim_t    i; \
\
    /* Initialize the absolute sum accumulator to zero. */ \
    PASTEMAC(chr,set0s)( absum ); \
\
    for ( i = 0; i < n; ++i ) \
    { \
        chi1 = x + (i  )*incx; \
\
        /* Compute the absolute value (or complex magnitude) of chi1. */ \
        PASTEMAC2(ch,chr,abval2s)( *chi1, abs_chi1 ); \
\
        /* Accumulate the absolute value of chi1 into absum. */ \
        PASTEMAC(chr,adds)( abs_chi1, absum ); \
    } \
\
    /* Store final value of absum to the output variable. */ \
    PASTEMAC(chr,copys)( absum, *norm ); \
}

INSERT_GENTFUNCR_BASIC0( norm1v_unb_var1 )


#undef  GENTFUNCR
#define GENTFUNCR( ctype, ctype_r, ch, chr, varname, kername ) \
\
void PASTEMAC(ch,varname) \
     ( \
       dim_t    n, \
       ctype*   x, inc_t incx, \
       ctype_r* norm, \
       cntx_t*  cntx, \
       rntm_t*  rntm  \
     ) \
{ \
    ctype_r* zero       = PASTEMAC(chr,0); \
    ctype_r* one        = PASTEMAC(chr,1); \
    ctype_r  scale; \
    ctype_r  sumsq; \
    ctype_r  sqrt_sumsq; \
\
    /* Initialize scale and sumsq to begin the summation. */ \
    PASTEMAC(chr,copys)( *zero, scale ); \
    PASTEMAC(chr,copys)( *one,  sumsq ); \
\
    /* Compute the sum of the squares of the vector. */ \
    PASTEMAC(ch,kername) \
    ( \
      n, \
      x, incx, \
      &scale, \
      &sumsq, \
      cntx, \
      rntm  \
    ); \
\
    /* Compute: norm = scale * sqrt( sumsq ) */ \
    PASTEMAC(chr,sqrt2s)( sumsq, sqrt_sumsq ); \
    PASTEMAC(chr,scals)( scale, sqrt_sumsq ); \
\
    /* Store the final value to the output variable. */ \
    PASTEMAC(chr,copys)( sqrt_sumsq, *norm ); \
}

//INSERT_GENTFUNCR_BASIC( normfv_unb_var1, sumsqv_unb_var1 )
//GENTFUNCR( scomplex, float,  c, s, normfv_unb_var1, sumsqv_unb_var1 )
void bli_cnormfv_unb_var1
    (
        dim_t    n,
        scomplex*   x,
        inc_t incx,
        float* norm,
        cntx_t*  cntx,
        rntm_t*  rntm
    )
{
    scomplex *x_buf = x;
    inc_t incx_buf = incx;

    // Querying the architecture ID to deploy the appropriate kernel
    arch_t id = bli_arch_query_id();
    switch ( id )
    {
        case BLIS_ARCH_ZEN4:
        case BLIS_ARCH_ZEN3:
        case BLIS_ARCH_ZEN2:
        case BLIS_ARCH_ZEN:;
#ifdef BLIS_KERNELS_ZEN
            // Memory pool declarations for packing vector X.
            // Initialize mem pool buffer to NULL and size to 0.
            // "buf" and "size" fields are assigned once memory
            // is allocated from the pool in bli_pba_acquire_m().
            // This will ensure bli_mem_is_alloc() will be passed on
            // an allocated memory if created or a NULL.
            mem_t   mem_buf_X = { 0 };
            rntm_t  rntm_l;
            // Packing for non-unit strided vector x.
            if ( incx != 1 )
            {
                // In order to get the buffer from pool via rntm access to memory broker
                // is needed. Following are initializations for rntm.
                if ( rntm == NULL ) { bli_rntm_init_from_global( &rntm_l ); }
                else                { rntm_l = *rntm; }
                bli_rntm_set_num_threads_only( 1, &rntm_l );
                bli_pba_rntm_set_pba( &rntm_l );

                // Calculate the size required for "n" scomplex elements in vector x.
                size_t buffer_size = n * sizeof( scomplex );

                #ifdef BLIS_ENABLE_MEM_TRACING
                    printf( "bli_scnorm2fv_unb_var1_avx2(): get mem pool block\n" );
                #endif

                // Acquire a Buffer(n*size(scomplex)) from the memory broker
                // and save the associated mem_t entry to mem_buf_X.
                bli_pba_acquire_m
                (
                    &rntm_l,
                    buffer_size,
                    BLIS_BUFFER_FOR_B_PANEL,
                    &mem_buf_X
                );

                // Continue packing X if buffer memory is allocated.
                if ( bli_mem_is_alloc( &mem_buf_X ) )
                {
                    x_buf = bli_mem_buffer( &mem_buf_X );
                    // Pack vector x with non-unit stride to a temp buffer x_buf with unit stride.
                    for ( dim_t x_index = 0; x_index < n; x_index++ )
                    {
                        *( x_buf + x_index ) = *( x + ( x_index * incx ) );
                    }
                    incx_buf = 1;
                }
            }

            bli_scnorm2fv_unb_var1_avx2( n, x_buf, incx_buf, norm, cntx );

            if ( bli_mem_is_alloc( &mem_buf_X ) )
            {
                #ifdef BLIS_ENABLE_MEM_TRACING
                    printf( "bli_scnorm2fv_unb_var1_avx2(): releasing mem pool block\n" );
                #endif
                // Return the buffer to pool.
                bli_pba_release( &rntm_l , &mem_buf_X );
            }
            break;
#endif
        default:;
            float* zero       = bli_s0;
            float* one        = bli_s1;
            float  scale;
            float  sumsq;
            float  sqrt_sumsq;

            // Initialize scale and sumsq to begin the summation.
            bli_scopys( *zero, scale );
            bli_scopys( *one,  sumsq );

            // Compute the sum of the squares of the vector.

            bli_csumsqv_unb_var1
            (
                n,
                x_buf,
                incx_buf,
                &scale,
                &sumsq,
                cntx,
                rntm
            );

            // Compute: norm = scale * sqrt( sumsq )
            bli_ssqrt2s( sumsq, sqrt_sumsq );
            bli_sscals( scale, sqrt_sumsq );

            // Store the final value to the output variable.
            bli_scopys( sqrt_sumsq, *norm );
    }
}

void bli_znormfv_unb_var1
    (
        dim_t    n,
        dcomplex*   x,
        inc_t incx,
        double* norm,
        cntx_t*  cntx,
        rntm_t*  rntm
    )
{
    /*
        Declaring a function pointer to point to the supported vectorized kernels.
        Based on the arch_id support, the appropriate function is set to the function
        pointer. Deployment happens post the switch cases.

        NOTE : A separate function pointer type is set to NULL, which will be used
               only for reduction purpose. This is because the norm(per thread)
               is of type double, and thus requires call to the vectorized
               kernel for dnormfv operation.
    */
    void ( *norm_fp )( dim_t, dcomplex*, inc_t, double*, cntx_t* ) = NULL;
    void ( *reduce_fp )( dim_t, double*, inc_t, double*, cntx_t* ) = NULL;

    dcomplex *x_buf = x;
    dim_t nt_ideal = -1;
    arch_t id = bli_arch_query_id();
    switch ( id )
    {
        case BLIS_ARCH_ZEN4:
        case BLIS_ARCH_ZEN3:
        case BLIS_ARCH_ZEN2:
        case BLIS_ARCH_ZEN:
#ifdef BLIS_KERNELS_ZEN

            norm_fp   = bli_dznorm2fv_unb_var1_avx2;
            reduce_fp = bli_dnorm2fv_unb_var1_avx2;

            break;
#endif
        default:;
            double* zero       = bli_d0;
            double* one        = bli_d1;
            double  scale;
            double  sumsq;
            double  sqrt_sumsq;

            // Initialize scale and sumsq to begin the summation.
            bli_dcopys( *zero, scale );
            bli_dcopys( *one,  sumsq );

            // Compute the sum of the squares of the vector.

            bli_zsumsqv_unb_var1
            (
                n,
                x,
                incx,
                &scale,
                &sumsq,
                cntx,
                rntm
            );

            // Compute: norm = scale * sqrt( sumsq )
            bli_dsqrt2s( sumsq, sqrt_sumsq );
            bli_dscals( scale, sqrt_sumsq );

            // Store the final value to the output variable.
            bli_dcopys( sqrt_sumsq, *norm );
    }

    /*
        If the function signature to vectorized kernel was not set,
        the default case would have been performed. Thus exit early.

        NOTE : Both the pointers are used here to avoid compilation warning.
    */
    if ( norm_fp == NULL && reduce_fp == NULL )
        return;
    
    /*
        When the size is such that nt_ideal is 1, and packing is not
        required( incx == 1 ), we can directly call the kernel to
        avoid framework overheads( fast-path ).
    */
    else if ( ( incx == 1 ) && ( n < 2000 ) )
    {
        norm_fp( n, x, incx, norm, cntx );
        return;
    }

    // Setting the ideal number of threads if support is enabled
    #if defined( BLIS_ENABLE_OPENMP ) && defined( AOCL_DYNAMIC )
        if ( n < 2000 )
            nt_ideal = 1;
        else if ( n < 6500 )
            nt_ideal = 4;
        else if ( n < 71000 )
            nt_ideal = 8;
        else if ( n < 200000 )
            nt_ideal = 16;
        else if ( n < 1530000 )
            nt_ideal = 32;
        
    #endif

    // Initialize a local runtime with global settings if necessary. Note
    // that in the case that a runtime is passed in, we make a local copy.
    rntm_t rntm_l;
    if ( rntm == NULL ) { bli_rntm_init_from_global( &rntm_l ); }
    else                { rntm_l = *rntm; }

    /*
        Initialize mem pool buffer to NULL and size to 0
        "buf" and "size" fields are assigned once memory
        is allocated from the pool in bli_pba_acquire_m().
        This will ensure bli_mem_is_alloc() will be passed on
        an allocated memory if created or a NULL .
    */

    mem_t mem_buf_X = { 0 };
    inc_t incx_buf = incx;
    dim_t nt;

    nt = bli_rntm_num_threads( &rntm_l );

    // nt is less than 1 if BLIS was configured with default settings for parallelism
    nt = ( nt < 1 )? 1 : nt;

    // Altering the ideal thread count if it was not set or if it is greater than nt
    if ( ( nt_ideal == -1 ) ||  ( nt_ideal > nt ) )
        nt_ideal = nt;

    // Packing for non-unit strided vector x.
    // In order to get the buffer from pool via rntm access to memory broker
    // is needed. Following are initializations for rntm.
    bli_rntm_set_num_threads_only( 1, &rntm_l );
    bli_pba_rntm_set_pba( &rntm_l );

    if ( incx == 0 )    nt_ideal = 1;
    else if ( incx != 1 )
    {
        // Calculate the size required for "n" double elements in vector x.
        size_t buffer_size = n * sizeof( dcomplex );

        #ifdef BLIS_ENABLE_MEM_TRACING
            printf( "bli_znorm2fv_unb_var1(): get mem pool block\n" );
        #endif

        // Acquire a buffer of the required size from the memory broker
        // and save the associated mem_t entry to mem_buf_X.
        bli_pba_acquire_m(
                            &rntm_l,
                            buffer_size,
                            BLIS_BITVAL_BUFFER_FOR_A_BLOCK,
                            &mem_buf_X
                         );


        // Continue packing X if buffer memory is allocated.
        if ( bli_mem_is_alloc( &mem_buf_X ) )
        {
            x_buf = bli_mem_buffer( &mem_buf_X );
            // Pack vector x with non-unit stride to a temp buffer x_buf with unit stride.
            for ( dim_t x_index = 0; x_index < n; x_index++ )
            {
                *( x_buf + x_index ) = *( x + ( x_index * incx ) );
            }
            incx_buf = 1;
        }
        else
        {
            nt_ideal = 1;
        }
    }

    #ifdef BLIS_ENABLE_OPENMP

        if( nt_ideal == 1 )
        {
    #endif
            /*
                The overhead cost with OpenMP is avoided in case
                the ideal number of threads needed is 1.
            */

            norm_fp( n, x_buf, incx_buf, norm, cntx );

            if ( bli_mem_is_alloc( &mem_buf_X ) )
            {
                #ifdef BLIS_ENABLE_MEM_TRACING
                    printf( "bli_znorm2fv_unb_var1(): releasing mem pool block\n" );
                #endif
                // Return the buffer to pool.
                bli_pba_release( &rntm_l , &mem_buf_X );
            }
            return;

    #ifdef BLIS_ENABLE_OPENMP
        }

        /*
            The following code-section is touched only in the case of
            requiring multiple threads for the computation.

            Every thread will calculate its own local norm, and all
            the local results will finally be reduced as per the mandate.
        */

        mem_t mem_buf_norm = { 0 };

        double *norm_per_thread = NULL;

        // Calculate the size required for buffer.
        size_t buffer_size = nt_ideal * sizeof(double);

        /*
            Acquire a buffer (nt_ideal * size(double)) from the memory broker
            and save the associated mem_t entry to mem_buf_norm.
        */

        bli_pba_acquire_m(
                            &rntm_l,
                            buffer_size,
                            BLIS_BITVAL_BUFFER_FOR_A_BLOCK,
                            &mem_buf_norm
                         );

        /* Continue if norm buffer memory is allocated*/
        if ( bli_mem_is_alloc( &mem_buf_norm ) )
        {
            norm_per_thread = bli_mem_buffer( &mem_buf_norm );

            /*
                In case the number of threads launched is not
                equal to the number of threads required, we will
                need to ensure that the garbage values are not part
                of the reduction step.

                Every local norm is initialized to 0.0 to avoid this.
            */

            for ( dim_t i = 0; i < nt_ideal; i++ )
                norm_per_thread[i] = 0.0;

            // Parallel code-section
            _Pragma("omp parallel num_threads(nt_ideal)")
            {
                /*
                    The number of actual threads spawned is
                    obtained here, so as to distribute the
                    job precisely.
                */

                dim_t n_threads = omp_get_num_threads();
                dim_t thread_id = omp_get_thread_num();
                dcomplex *x_start;

                // Obtain the job-size and region for compute
                dim_t job_per_thread, offset;

                bli_normfv_thread_partition( n, n_threads, &offset, &job_per_thread, 2, incx_buf, thread_id );
                x_start = x_buf + offset;

                // Call to the kernel with the appropriate starting address
                norm_fp( job_per_thread, x_start, incx_buf, ( norm_per_thread + thread_id ), cntx );
            }

            /*
                Reduce the partial results onto a final scalar, based
                on the mandate.

                Every partial result needs to be subjected to overflow or
                underflow handling if needed. Thus this reduction step involves
                the same logic as the one present in the kernel. The kernel is
                therefore reused for the reduction step.
            */

            reduce_fp( nt_ideal, norm_per_thread, 1, norm, cntx );

            // Releasing the allocated memory if it was allocated
            bli_pba_release( &rntm_l, &mem_buf_norm );
        }

        /*
            In case of failing to acquire the buffer from the memory
            pool, call the single-threaded kernel and return.
        */
        else
        {
            norm_fp( n, x_buf, incx_buf, norm, cntx );
        }

        /*
            By this point, the norm value would have been set by the appropriate
            code-section that was touched. The assignment is not abstracted outside
            in order to avoid unnecessary conditionals.
        */

    #endif

    if ( bli_mem_is_alloc( &mem_buf_X ) )
    {
        #ifdef BLIS_ENABLE_MEM_TRACING
            printf( "bli_znorm2fv_unb_var1(): releasing mem pool block\n" );
        #endif
        // Return the buffer to pool.
        bli_pba_release( &rntm_l , &mem_buf_X );
    }
}

#undef  GENTFUNCR
// We've disabled the dotv-based implementation because that method of
// computing the sum of the squares of x inherently does not check for
// overflow. Instead, we use the fallback method based on sumsqv, which
// takes care to not overflow unnecessarily (ie: takes care for the
// sqrt( sum of the squares of x ) to not overflow if the sum of the
// squares of x would normally overflow. See GitHub issue #332 for
// discussion.
#if 0 //defined(FE_OVERFLOW) && !defined(__APPLE__)
#define GENTFUNCR( ctype, ctype_r, ch, chr, varname, kername ) \
\
void PASTEMAC(ch,varname) \
     ( \
       dim_t    n, \
       ctype*   x, inc_t incx, \
       ctype_r* norm, \
       cntx_t*  cntx, \
       rntm_t*  rntm  \
     ) \
{ \
    ctype_r* zero       = PASTEMAC(chr,0); \
    ctype_r* one        = PASTEMAC(chr,1); \
    ctype_r  scale; \
    ctype_r  sumsq; \
    ctype_r  sqrt_sumsq; \
\
    /* Initialize scale and sumsq to begin the summation. */ \
    PASTEMAC(chr,copys)( *zero, scale ); \
    PASTEMAC(chr,copys)( *one,  sumsq ); \
\
    /* An optimization: first try to use dotv to compute the sum of
       the squares of the vector. If no floating-point exceptions
       (specifically, overflow and invalid exceptions) were produced,
       then we accept the computed value and returne early. The cost
       of this optimization is the "sunk" cost of the initial dotv
       when sumsqv must be used instead. However, we expect that the
       vast majority of use cases will not produce exceptions, and
       therefore only one pass through the data, via dotv, will be
       required. */ \
    if ( TRUE ) \
    { \
        int      f_exp_raised;\
        ctype    sumsqc; \
\
        feclearexcept( FE_ALL_EXCEPT );\
\
        PASTEMAC2(ch,dotv,BLIS_TAPI_EX_SUF) \
        ( \
          BLIS_NO_CONJUGATE, \
          BLIS_NO_CONJUGATE, \
          n,\
          x, incx, \
          x, incx, \
          &sumsqc, \
          cntx, \
          rntm  \
        ); \
\
        PASTEMAC2(ch,chr,copys)( sumsqc, sumsq ); \
\
        f_exp_raised = fetestexcept( FE_OVERFLOW | FE_INVALID );\
\
        if ( !f_exp_raised ) \
        { \
            PASTEMAC(chr,sqrt2s)( sumsq, *norm ); \
            return; \
        } \
    } \
\
    /* Compute the sum of the squares of the vector. */ \
    PASTEMAC(ch,kername) \
    ( \
      n, \
      x, incx, \
      &scale, \
      &sumsq, \
      cntx, \
      rntm  \
    ); \
\
    /* Compute: norm = scale * sqrt( sumsq ) */ \
    PASTEMAC(chr,sqrt2s)( sumsq, sqrt_sumsq ); \
    PASTEMAC(chr,scals)( scale, sqrt_sumsq ); \
\
    /* Store the final value to the output variable. */ \
    PASTEMAC(chr,copys)( sqrt_sumsq, *norm ); \
}
#else
#define GENTFUNCR( ctype, ctype_r, ch, chr, varname, kername ) \
\
void PASTEMAC(ch,varname) \
     ( \
       dim_t    n, \
       ctype*   x, inc_t incx, \
       ctype_r* norm, \
       cntx_t*  cntx, \
       rntm_t*  rntm  \
     ) \
{ \
    ctype_r* zero       = PASTEMAC(chr,0); \
    ctype_r* one        = PASTEMAC(chr,1); \
    ctype_r  scale; \
    ctype_r  sumsq; \
    ctype_r  sqrt_sumsq; \
\
    /* Initialize scale and sumsq to begin the summation. */ \
    PASTEMAC(chr,copys)( *zero, scale ); \
    PASTEMAC(chr,copys)( *one,  sumsq ); \
\
    /* Compute the sum of the squares of the vector. */ \
\
    PASTEMAC(ch,kername) \
    ( \
      n, \
      x, incx, \
      &scale, \
      &sumsq, \
      cntx, \
      rntm  \
    ); \
\
    /* Compute: norm = scale * sqrt( sumsq ) */ \
    PASTEMAC(chr,sqrt2s)( sumsq, sqrt_sumsq ); \
    PASTEMAC(chr,scals)( scale, sqrt_sumsq ); \
\
    /* Store the final value to the output variable. */ \
    PASTEMAC(chr,copys)( sqrt_sumsq, *norm ); \
}
#endif
//GENTFUNCR( float,   float,  s, s, normfv_unb_var1, sumsqv_unb_var1 )

void bli_snormfv_unb_var1
    (
        dim_t    n,
        float*   x,
        inc_t incx,
        float* norm,
        cntx_t*  cntx,
        rntm_t*  rntm
    )
{
    // Early return if n = 1.
    if ( n == 1 )
    {
        *norm = bli_fabs( *x );

        // If the value in x is 0.0, the sign bit gets inverted
        // Reinvert the sign bit in this case.
        if ( ( *norm ) == -0.0 ) ( *norm ) = 0.0;
        return;
    }

    float *x_buf = x;
    inc_t incx_buf = incx;

    // Querying the architecture ID to deploy the appropriate kernel
    arch_t id = bli_arch_query_id();
    switch ( id )
    {
        case BLIS_ARCH_ZEN4:
        case BLIS_ARCH_ZEN3:
        case BLIS_ARCH_ZEN2:
        case BLIS_ARCH_ZEN:;
#ifdef BLIS_KERNELS_ZEN
            // Memory pool declarations for packing vector X.
            // Initialize mem pool buffer to NULL and size to 0.
            // "buf" and "size" fields are assigned once memory
            // is allocated from the pool in bli_pba_acquire_m().
            // This will ensure bli_mem_is_alloc() will be passed on
            // an allocated memory if created or a NULL.
            mem_t   mem_buf_X = { 0 };
            rntm_t  rntm_l;
            // Packing for non-unit strided vector x.
            if ( incx != 1 )
            {
                // Initialize a local runtime with global settings if necessary. Note
                // that in the case that a runtime is passed in, we make a local copy.
                if ( rntm == NULL ) { bli_rntm_init_from_global( &rntm_l ); }
                else                { rntm_l = *rntm; }

                // In order to get the buffer from pool via rntm access to memory broker
                // is needed. Following are initializations for rntm.
                bli_rntm_set_num_threads_only( 1, &rntm_l );
                bli_pba_rntm_set_pba( &rntm_l );

                // Calculate the size required for "n" float elements in vector x.
                size_t buffer_size = n * sizeof( float );

                #ifdef BLIS_ENABLE_MEM_TRACING
                    printf( "bli_snorm2fv_unb_var1_avx2(): get mem pool block\n" );
                #endif

                // Acquire a Buffer(n*size(float)) from the memory broker
                // and save the associated mem_t entry to mem_buf_X.
                bli_pba_acquire_m
                (
                    &rntm_l,
                    buffer_size,
                    BLIS_BUFFER_FOR_B_PANEL,
                    &mem_buf_X
                );

                // Continue packing X if buffer memory is allocated.
                if ( bli_mem_is_alloc( &mem_buf_X ) )
                {
                    x_buf = bli_mem_buffer( &mem_buf_X );
                    // Pack vector x with non-unit stride to a temp buffer x_buf with unit stride.
                    for ( dim_t x_index = 0; x_index < n; x_index++ )
                    {
                        *( x_buf + x_index ) = *( x + ( x_index * incx ) );
                    }
                    incx_buf = 1;
                }
            }

            bli_snorm2fv_unb_var1_avx2( n, x_buf, incx_buf, norm, cntx );

            if ( bli_mem_is_alloc( &mem_buf_X ) )
            {
                #ifdef BLIS_ENABLE_MEM_TRACING
                    printf( "bli_snorm2fv_unb_var1_avx2(): releasing mem pool block\n" );
                #endif
                // Return the buffer to pool.
                bli_pba_release( &rntm_l , &mem_buf_X );
            }
            break;
#endif
        default:;
            float* zero       = bli_s0;
            float* one        = bli_s1;
            float  scale;
            float  sumsq;
            float  sqrt_sumsq;

            // Initialize scale and sumsq to begin the summation.
            bli_sscopys( *zero, scale );
            bli_sscopys( *one,  sumsq );

            // Compute the sum of the squares of the vector.
            bli_ssumsqv_unb_var1
            (
                n,
                x_buf,
                incx_buf,
                &scale,
                &sumsq,
                cntx,
                rntm
            );

            // Compute: norm = scale * sqrt( sumsq )
            bli_ssqrt2s( sumsq, sqrt_sumsq );
            bli_sscals( scale, sqrt_sumsq );

            // Store the final value to the output variable.
            bli_scopys( sqrt_sumsq, *norm );
    }
}

void bli_dnormfv_unb_var1
    (
        dim_t    n,
        double*   x,
        inc_t incx,
        double* norm,
        cntx_t*  cntx,
        rntm_t*  rntm
    )
{
    // Early return if n = 1.
    if ( n == 1 )
    {
        *norm = bli_fabs( *x );

        // If the value in x is 0.0, the sign bit gets inverted
        // Reinvert the sign bit in this case.
        if ( ( *norm ) == -0.0 ) ( *norm ) = 0.0;
        return;
    }
    /*
        Declaring a function pointer to point to the supported vectorized kernels.
        Based on the arch_id support, the appropriate function is set to the function
        pointer. Deployment happens post the switch cases. In case of adding any
        AVX-512 kernel, the code for deployment remains the same.
    */
    void ( *norm_fp )( dim_t, double*, inc_t, double*, cntx_t* ) = NULL;

    double *x_buf = x;
    dim_t nt_ideal = -1;
    arch_t id = bli_arch_query_id();
    switch ( id )
    {
        case BLIS_ARCH_ZEN4:
        case BLIS_ARCH_ZEN3:
        case BLIS_ARCH_ZEN2:
        case BLIS_ARCH_ZEN:
#ifdef BLIS_KERNELS_ZEN

        norm_fp = bli_dnorm2fv_unb_var1_avx2;

        break;
#endif
        default:;
            // The following call to the kernel is
            // single threaded in this case.
            double* zero       = bli_d0;
            double* one        = bli_d1;
            double  scale;
            double  sumsq;
            double  sqrt_sumsq;

            // Initialize scale and sumsq to begin the summation.
            bli_ddcopys( *zero, scale );
            bli_ddcopys( *one,  sumsq );

            // Compute the sum of the squares of the vector.
            bli_dsumsqv_unb_var1
            (
                n,
                x,
                incx,
                &scale,
                &sumsq,
                cntx,
                rntm
            );

            // Compute: norm = scale * sqrt( sumsq )
            bli_dsqrt2s( sumsq, sqrt_sumsq );
            bli_dscals( scale, sqrt_sumsq );

            // Store the final value to the output variable.
            bli_dcopys( sqrt_sumsq, *norm );
    }

    /*
        If the function signature to vectorized kernel was not set,
        the default case would have been performed. Thus exit early.
    */
    if ( norm_fp == NULL )
        return;
    
    /*
        When the size is such that nt_ideal is 1, and packing is not
        required( incx == 1 ), we can directly call the kernel to
        avoid framework overheads( fast-path ).
    */
    else if ( ( incx == 1 ) && ( n < 4000 ) )
    {
        norm_fp( n, x, incx, norm, cntx );
        return;
    }

    // Setting the ideal number of threads if support is enabled
    #if defined( BLIS_ENABLE_OPENMP ) && defined( AOCL_DYNAMIC )

        if ( n < 4000 )
            nt_ideal = 1;
        else if ( n < 17000 )
            nt_ideal = 4;
        else if ( n < 136000 )
            nt_ideal = 8;
        else if ( n < 365000 )
            nt_ideal = 16;
        else if ( n < 2950000 )
            nt_ideal = 32;

    #endif

    // Initialize a local runtime with global settings if necessary. Note
    // that in the case that a runtime is passed in, we make a local copy.
    rntm_t rntm_l;
    if ( rntm == NULL ) { bli_rntm_init_from_global( &rntm_l ); }
    else                { rntm_l = *rntm; }

    /*
        Initialize mem pool buffer to NULL and size to 0
        "buf" and "size" fields are assigned once memory
        is allocated from the pool in bli_pba_acquire_m().
        This will ensure bli_mem_is_alloc() will be passed on
        an allocated memory if created or a NULL .
    */

    mem_t mem_buf_X = { 0 };
    inc_t incx_buf = incx;
    dim_t nt;

    nt = bli_rntm_num_threads( &rntm_l );

    // nt is less than 1 if BLIS was configured with default settings for parallelism
    nt = ( nt < 1 )? 1 : nt;

    if ( ( nt_ideal == -1 ) ||  ( nt_ideal > nt ) )
        nt_ideal = nt;

    // Packing for non-unit strided vector x.
    // In order to get the buffer from pool via rntm access to memory broker
    // is needed. Following are initializations for rntm.
    bli_rntm_set_num_threads_only( 1, &rntm_l );
    bli_pba_rntm_set_pba( &rntm_l );

    if ( incx == 0 )    nt_ideal = 1;
    else if ( incx != 1 )
    {
        // Calculate the size required for "n" double elements in vector x.
        size_t buffer_size = n * sizeof( double );

        #ifdef BLIS_ENABLE_MEM_TRACING
            printf( "bli_dnorm2fv_unb_var1(): get mem pool block\n" );
        #endif

        // Acquire a buffer of the required size from the memory broker
        // and save the associated mem_t entry to mem_buf_X.
        bli_pba_acquire_m(
                            &rntm_l,
                            buffer_size,
                            BLIS_BITVAL_BUFFER_FOR_A_BLOCK,
                            &mem_buf_X
                         );


        // Continue packing X if buffer memory is allocated.
        if ( bli_mem_is_alloc( &mem_buf_X ) )
        {
            x_buf = bli_mem_buffer( &mem_buf_X );
            // Pack vector x with non-unit stride to a temp buffer x_buf with unit stride.
            for ( dim_t x_index = 0; x_index < n; x_index++ )
            {
                *( x_buf + x_index ) = *( x + ( x_index * incx ) );
            }
            incx_buf = 1;
        }
        else
        {
            nt_ideal = 1;
        }
    }

    #ifdef BLIS_ENABLE_OPENMP

        if( nt_ideal == 1 )
        {
    #endif
            /*
                The overhead cost with OpenMP is avoided in case
                the ideal number of threads needed is 1.
            */

            norm_fp( n, x_buf, incx_buf, norm, cntx );

            if ( bli_mem_is_alloc( &mem_buf_X ) )
            {
                #ifdef BLIS_ENABLE_MEM_TRACING
                    printf( "bli_dnorm2fv_unb_var1(): releasing mem pool block\n" );
                #endif
                // Return the buffer to pool.
                bli_pba_release( &rntm_l , &mem_buf_X );
            }
            return;

    #ifdef BLIS_ENABLE_OPENMP
        }

        /*
            The following code-section is touched only in the case of
            requiring multiple threads for the computation.

            Every thread will calculate its own local norm, and all
            the local results will finally be reduced as per the mandate.
        */

        mem_t mem_buf_norm = { 0 };

        double *norm_per_thread = NULL;

        // Calculate the size required for buffer.
        size_t buffer_size = nt_ideal * sizeof(double);

        /*
            Acquire a buffer (nt_ideal * size(double)) from the memory broker
            and save the associated mem_t entry to mem_buf_norm.
        */

        bli_pba_acquire_m(
                            &rntm_l,
                            buffer_size,
                            BLIS_BITVAL_BUFFER_FOR_A_BLOCK,
                            &mem_buf_norm
                         );

        /* Continue if norm buffer memory is allocated*/
        if ( bli_mem_is_alloc( &mem_buf_norm ) )
        {
            norm_per_thread = bli_mem_buffer( &mem_buf_norm );

            /*
                In case the number of threads launched is not
                equal to the number of threads required, we will
                need to ensure that the garbage values are not part
                of the reduction step.

                Every local norm is initialized to 0.0 to avoid this.
            */

            for ( dim_t i = 0; i < nt_ideal; i++ )
                norm_per_thread[i] = 0.0;

            // Parallel code-section
            _Pragma("omp parallel num_threads(nt_ideal)")
            {
                /*
                    The number of actual threads spawned is
                    obtained here, so as to distribute the
                    job precisely.
                */

                dim_t n_threads = omp_get_num_threads();

                dim_t thread_id = omp_get_thread_num();
                double *x_start;

                // Obtain the job-size and region for compute
                dim_t job_per_thread, offset;
                bli_normfv_thread_partition( n, n_threads, &offset, &job_per_thread, 4, incx_buf, thread_id );

                x_start = x_buf + offset;

                // Call to the kernel with the appropriate starting address
                norm_fp( job_per_thread, x_start, incx_buf, ( norm_per_thread + thread_id ), cntx );
            }

            /*
                Reduce the partial results onto a final scalar, based
                on the mandate.

                Every partial result needs to be subjected to overflow or
                underflow handling if needed. Thus this reduction step involves
                the same logic as the one present in the kernel. The kernel is
                therefore reused for the reduction step.
            */

            norm_fp( nt_ideal, norm_per_thread, 1, norm, cntx );

            // Releasing the allocated memory if it was allocated
            bli_pba_release( &rntm_l, &mem_buf_norm );
        }

        /*
            In case of failing to acquire the buffer from the memory
            pool, call the single-threaded kernel and return.
        */
        else
        {
            norm_fp( n, x_buf, incx_buf, norm, cntx );
        }

        /*
            By this point, the norm value would have been set by the appropriate
            code-section that was touched. The assignment is not abstracted outside
            in order to avoid unnecessary conditionals.
        */

    #endif

    if ( bli_mem_is_alloc( &mem_buf_X ) )
    {
        #ifdef BLIS_ENABLE_MEM_TRACING
            printf( "bli_dnorm2fv_unb_var1(): releasing mem pool block\n" );
        #endif
        // Return the buffer to pool.
        bli_pba_release( &rntm_l , &mem_buf_X );
    }
}

#undef  GENTFUNCR
#define GENTFUNCR( ctype, ctype_r, ch, chr, varname ) \
\
void PASTEMAC(ch,varname) \
     ( \
       dim_t    n, \
       ctype*   x, inc_t incx, \
       ctype_r* norm, \
       cntx_t*  cntx, \
       rntm_t*  rntm  \
     ) \
{ \
    ctype*   chi1; \
    ctype_r  abs_chi1; \
    ctype_r  abs_chi1_max; \
    dim_t    i; \
\
    /* Initialize the maximum absolute value to zero. */ \
    PASTEMAC(chr,set0s)( abs_chi1_max ); \
\
    for ( i = 0; i < n; ++i ) \
    { \
        chi1 = x + (i  )*incx; \
\
        /* Compute the absolute value (or complex magnitude) of chi1. */ \
        PASTEMAC2(ch,chr,abval2s)( *chi1, abs_chi1 ); \
\
        /* If the absolute value of the current element exceeds that of
           the previous largest, save it and its index. If NaN is
           encountered, then treat it the same as if it were a valid
           value that was larger than any previously seen. This
           behavior mimics that of LAPACK's ?lange(). */ \
        if ( abs_chi1_max < abs_chi1 || bli_isnan( abs_chi1 ) ) \
        { \
            PASTEMAC(chr,copys)( abs_chi1, abs_chi1_max ); \
        } \
    } \
\
    /* Store the final value to the output variable. */ \
    PASTEMAC(chr,copys)( abs_chi1_max, *norm ); \
}

INSERT_GENTFUNCR_BASIC0( normiv_unb_var1 )



#undef  GENTFUNCR
#define GENTFUNCR( ctype, ctype_r, ch, chr, varname, kername ) \
\
void PASTEMAC(ch,varname) \
     ( \
       doff_t   diagoffx, \
       diag_t   diagx, \
       uplo_t   uplox, \
       dim_t    m, \
       dim_t    n, \
       ctype*   x, inc_t rs_x, inc_t cs_x, \
       ctype_r* norm, \
       cntx_t*  cntx, \
       rntm_t*  rntm  \
     ) \
{ \
    ctype*   one       = PASTEMAC(ch,1); \
    ctype*   x0; \
    ctype*   chi1; \
    ctype*   x2; \
    ctype_r  absum_max; \
    ctype_r  absum_j; \
    ctype_r  abval_chi1; \
    uplo_t   uplox_eff; \
    dim_t    n_iter; \
    dim_t    n_elem, n_elem_max; \
    inc_t    ldx, incx; \
    dim_t    j, i; \
    dim_t    ij0, n_shift; \
\
    /* Initialize the maximum absolute column sum to zero. */ \
    PASTEMAC(chr,set0s)( absum_max ); \
\
    /* If either dimension is zero, return with absum_max equal to zero. */ \
    if ( bli_zero_dim2( m, n ) ) \
    { \
        PASTEMAC(chr,copys)( absum_max, *norm ); \
        return; \
    } \
\
    /* Set various loop parameters. */ \
    bli_set_dims_incs_uplo_1m_noswap \
    ( \
      diagoffx, BLIS_NONUNIT_DIAG, \
      uplox, m, n, rs_x, cs_x, \
      &uplox_eff, &n_elem_max, &n_iter, &incx, &ldx, \
      &ij0, &n_shift \
    ); \
\
    /* If the matrix is zeros, return with absum_max equal to zero. */ \
    if ( bli_is_zeros( uplox_eff ) ) \
    { \
        PASTEMAC(chr,copys)( absum_max, *norm ); \
        return; \
    } \
\
\
    /* Handle dense and upper/lower storage cases separately. */ \
    if ( bli_is_dense( uplox_eff ) ) \
    { \
        for ( j = 0; j < n_iter; ++j ) \
        { \
            n_elem = n_elem_max; \
\
            x0     = x + (j  )*ldx + (0  )*incx; \
\
            /* Compute the norm of the current column. */ \
            PASTEMAC(ch,kername) \
            ( \
              n_elem, \
              x0, incx, \
              &absum_j, \
              cntx, \
              rntm  \
            ); \
\
            /* If absum_j is greater than the previous maximum value,
               then save it. */ \
            if ( absum_max < absum_j || bli_isnan( absum_j ) ) \
            { \
                PASTEMAC(chr,copys)( absum_j, absum_max ); \
            } \
        } \
    } \
    else \
    { \
        if ( bli_is_upper( uplox_eff ) ) \
        { \
            for ( j = 0; j < n_iter; ++j ) \
            { \
                n_elem = bli_min( n_shift + j + 1, n_elem_max ); \
\
                x0     = x + (ij0+j  )*ldx + (0       )*incx; \
                chi1   = x + (ij0+j  )*ldx + (n_elem-1)*incx; \
\
                /* Compute the norm of the super-diagonal elements. */ \
                PASTEMAC(ch,kername) \
                ( \
                  n_elem - 1, \
                  x0, incx, \
                  &absum_j, \
                  cntx, \
                  rntm  \
                ); \
\
                if ( bli_is_unit_diag( diagx ) ) chi1 = one; \
\
                /* Handle the diagonal element separately in case it's
                   unit. */ \
                PASTEMAC2(ch,chr,abval2s)( *chi1, abval_chi1 ); \
                PASTEMAC(chr,adds)( abval_chi1, absum_j ); \
\
                /* If absum_j is greater than the previous maximum value,
                   then save it. */ \
                if ( absum_max < absum_j || bli_isnan( absum_j ) ) \
                { \
                    PASTEMAC(chr,copys)( absum_j, absum_max ); \
                } \
            } \
        } \
        else if ( bli_is_lower( uplox_eff ) ) \
        { \
            for ( j = 0; j < n_iter; ++j ) \
            { \
                i      = bli_max( 0, ( doff_t )j - ( doff_t )n_shift ); \
                n_elem = n_elem_max - i; \
\
                chi1   = x + (j  )*ldx + (ij0+i  )*incx; \
                x2     = x + (j  )*ldx + (ij0+i+1)*incx; \
\
                /* Compute the norm of the sub-diagonal elements. */ \
                PASTEMAC(ch,kername) \
                ( \
                  n_elem - 1, \
                  x2, incx, \
                  &absum_j, \
                  cntx, \
                  rntm  \
                ); \
\
                if ( bli_is_unit_diag( diagx ) ) chi1 = one; \
\
                /* Handle the diagonal element separately in case it's
                   unit. */ \
                PASTEMAC2(ch,chr,abval2s)( *chi1, abval_chi1 ); \
                PASTEMAC(chr,adds)( abval_chi1, absum_j ); \
\
                /* If absum_j is greater than the previous maximum value,
                   then save it. */ \
                if ( absum_max < absum_j || bli_isnan( absum_j ) ) \
                { \
                    PASTEMAC(chr,copys)( absum_j, absum_max ); \
                } \
            } \
        } \
    } \
\
    /* Store final value of absum_max to the output variable. */ \
    PASTEMAC(chr,copys)( absum_max, *norm ); \
}

INSERT_GENTFUNCR_BASIC( norm1m_unb_var1, norm1v_unb_var1 )


#undef  GENTFUNCR
#define GENTFUNCR( ctype, ctype_r, ch, chr, varname, kername ) \
\
void PASTEMAC(ch,varname) \
     ( \
       doff_t   diagoffx, \
       diag_t   diagx, \
       uplo_t   uplox, \
       dim_t    m, \
       dim_t    n, \
       ctype*   x, inc_t rs_x, inc_t cs_x, \
       ctype_r* norm, \
       cntx_t*  cntx, \
       rntm_t*  rntm  \
     ) \
{ \
    ctype*   one    = PASTEMAC(ch,1); \
    ctype_r* one_r  = PASTEMAC(chr,1); \
    ctype_r* zero_r = PASTEMAC(chr,0); \
    ctype*   x0; \
    ctype*   chi1; \
    ctype*   x2; \
    ctype_r  scale; \
    ctype_r  sumsq; \
    ctype_r  sqrt_sumsq; \
    uplo_t   uplox_eff; \
    dim_t    n_iter; \
    dim_t    n_elem, n_elem_max; \
    inc_t    ldx, incx; \
    dim_t    j, i; \
    dim_t    ij0, n_shift; \
\
    /* Return a norm of zero if either dimension is zero. */ \
    if ( bli_zero_dim2( m, n ) ) \
    { \
        PASTEMAC(chr,set0s)( *norm ); \
        return; \
    } \
\
    /* Set various loop parameters. Here, we pretend that diagx is equal to
       BLIS_NONUNIT_DIAG because we handle the unit diagonal case manually. */ \
    bli_set_dims_incs_uplo_1m \
    ( \
      diagoffx, BLIS_NONUNIT_DIAG, \
      uplox, m, n, rs_x, cs_x, \
      &uplox_eff, &n_elem_max, &n_iter, &incx, &ldx, \
      &ij0, &n_shift \
    ); \
\
    /* Check the effective uplo; if it's zeros, then our norm is zero. */ \
    if ( bli_is_zeros( uplox_eff ) ) \
    { \
        PASTEMAC(chr,set0s)( *norm ); \
        return; \
    } \
\
    /* Initialize scale and sumsq to begin the summation. */ \
    PASTEMAC(chr,copys)( *zero_r, scale ); \
    PASTEMAC(chr,copys)( *one_r,  sumsq ); \
\
    /* Handle dense and upper/lower storage cases separately. */ \
    if ( bli_is_dense( uplox_eff ) ) \
    { \
        for ( j = 0; j < n_iter; ++j ) \
        { \
            n_elem = n_elem_max; \
\
            x0     = x + (j  )*ldx + (0  )*incx; \
\
            /* Compute the norm of the current column. */ \
            PASTEMAC(ch,kername) \
            ( \
              n_elem, \
              x0, incx, \
              &scale, \
              &sumsq, \
              cntx, \
              rntm  \
            ); \
        } \
    } \
    else \
    { \
        if ( bli_is_upper( uplox_eff ) ) \
        { \
            for ( j = 0; j < n_iter; ++j ) \
            { \
                n_elem = bli_min( n_shift + j + 1, n_elem_max ); \
\
                x0     = x + (ij0+j  )*ldx + (0       )*incx; \
                chi1   = x + (ij0+j  )*ldx + (n_elem-1)*incx; \
\
                /* Sum the squares of the super-diagonal elements. */ \
                PASTEMAC(ch,kername) \
                ( \
                  n_elem - 1, \
                  x0, incx, \
                  &scale, \
                  &sumsq, \
                  cntx, \
                  rntm  \
                ); \
\
                if ( bli_is_unit_diag( diagx ) ) chi1 = one; \
\
                /* Handle the diagonal element separately in case it's
                   unit. */ \
                PASTEMAC(ch,kername) \
                ( \
                  1, \
                  chi1, incx, \
                  &scale, \
                  &sumsq, \
                  cntx, \
                  rntm  \
                ); \
            } \
        } \
        else if ( bli_is_lower( uplox_eff ) ) \
        { \
            for ( j = 0; j < n_iter; ++j ) \
            { \
                i      = bli_max( 0, ( doff_t )j - ( doff_t )n_shift ); \
                n_elem = n_elem_max - i; \
\
                chi1   = x + (j  )*ldx + (ij0+i  )*incx; \
                x2     = x + (j  )*ldx + (ij0+i+1)*incx; \
\
                /* Sum the squares of the sub-diagonal elements. */ \
                PASTEMAC(ch,kername) \
                ( \
                  n_elem - 1, \
                  x2, incx, \
                  &scale, \
                  &sumsq, \
                  cntx, \
                  rntm  \
                ); \
\
                if ( bli_is_unit_diag( diagx ) ) chi1 = one; \
\
                /* Handle the diagonal element separately in case it's
                   unit. */ \
                PASTEMAC(ch,kername) \
                ( \
                  1, \
                  chi1, incx, \
                  &scale, \
                  &sumsq, \
                  cntx, \
                  rntm  \
                ); \
            } \
        } \
    } \
\
    /* Compute: norm = scale * sqrt( sumsq ) */ \
    PASTEMAC(chr,sqrt2s)( sumsq, sqrt_sumsq ); \
    PASTEMAC(chr,scals)( scale, sqrt_sumsq ); \
\
    /* Store the final value to the output variable. */ \
    PASTEMAC(chr,copys)( sqrt_sumsq, *norm ); \
}

INSERT_GENTFUNCR_BASIC( normfm_unb_var1, sumsqv_unb_var1 )


#undef  GENTFUNCR
#define GENTFUNCR( ctype, ctype_r, ch, chr, varname, kername ) \
\
void PASTEMAC(ch,varname) \
     ( \
       doff_t   diagoffx, \
       diag_t   diagx, \
       uplo_t   uplox, \
       dim_t    m, \
       dim_t    n, \
       ctype*   x, inc_t rs_x, inc_t cs_x, \
       ctype_r* norm, \
       cntx_t*  cntx, \
       rntm_t*  rntm  \
     ) \
{ \
    /* Induce a transposition so that rows become columns. */ \
    bli_swap_dims( &m, &n ); \
    bli_swap_incs( &rs_x, &cs_x ); \
    bli_toggle_uplo( &uplox ); \
    bli_negate_diag_offset( &diagoffx ); \
\
    /* Now we can simply compute the 1-norm of this transposed matrix,
       which will be equivalent to the infinity-norm of the original
       matrix. */ \
    PASTEMAC(ch,kername) \
    ( \
      diagoffx, \
      diagx, \
      uplox, \
      m, \
      n, \
      x, rs_x, cs_x, \
      norm, \
      cntx, \
      rntm  \
    ); \
}

INSERT_GENTFUNCR_BASIC( normim_unb_var1, norm1m_unb_var1 )


#undef  GENTFUNC
#define GENTFUNC( ctype, ch, varname, randmac ) \
\
void PASTEMAC(ch,varname) \
     ( \
       dim_t   n, \
       ctype*  x, inc_t incx, \
       cntx_t* cntx, \
       rntm_t* rntm  \
     ) \
{ \
    ctype* chi1; \
    dim_t  i; \
\
    chi1 = x; \
\
    for ( i = 0; i < n; ++i ) \
    { \
        PASTEMAC(ch,randmac)( *chi1 ); \
\
        chi1 += incx; \
    } \
}

INSERT_GENTFUNC_BASIC( randv_unb_var1,  rands )
INSERT_GENTFUNC_BASIC( randnv_unb_var1, randnp2s )


#undef  GENTFUNC
#define GENTFUNC( ctype, ch, varname, kername ) \
\
void PASTEMAC(ch,varname) \
     ( \
       doff_t  diagoffx, \
       uplo_t  uplox, \
       dim_t   m, \
       dim_t   n, \
       ctype*  x, inc_t rs_x, inc_t cs_x, \
       cntx_t* cntx, \
       rntm_t* rntm  \
     ) \
{ \
    ctype*  one = PASTEMAC(ch,1); \
    ctype*  x0; \
    ctype*  x1; \
    ctype*  x2; \
    ctype*  chi1; \
    ctype   beta; \
    ctype   omega; \
    double  max_m_n; \
    uplo_t  uplox_eff; \
    dim_t   n_iter; \
    dim_t   n_elem, n_elem_max; \
    inc_t   ldx, incx; \
    dim_t   j, i; \
    dim_t   ij0, n_shift; \
\
    /* Set various loop parameters. Here, we pretend that diagx is equal to
       BLIS_NONUNIT_DIAG because we handle the unit diagonal case manually. */ \
    bli_set_dims_incs_uplo_1m \
    ( \
      diagoffx, BLIS_NONUNIT_DIAG, \
      uplox, m, n, rs_x, cs_x, \
      &uplox_eff, &n_elem_max, &n_iter, &incx, &ldx, \
      &ij0, &n_shift \
    ); \
\
    if ( bli_is_zeros( uplox_eff ) ) return; \
\
    /* Handle dense and upper/lower storage cases separately. */ \
    if ( bli_is_dense( uplox_eff ) ) \
    { \
        for ( j = 0; j < n_iter; ++j ) \
        { \
            n_elem = n_elem_max; \
\
            x1     = x + (j  )*ldx + (0  )*incx; \
\
            /*PASTEMAC2(ch,kername,BLIS_TAPI_EX_SUF)*/ \
            PASTEMAC(ch,kername) \
            ( \
              n_elem, \
              x1, incx, \
              cntx, \
              rntm  \
            ); \
        } \
    } \
    else \
    { \
        max_m_n = bli_max( m, n ); \
\
        PASTEMAC2(d,ch,sets)( max_m_n, 0.0, omega ); \
        PASTEMAC(ch,copys)( *one, beta ); \
        PASTEMAC(ch,invscals)( omega, beta ); \
\
        if ( bli_is_upper( uplox_eff ) ) \
        { \
            for ( j = 0; j < n_iter; ++j ) \
            { \
                n_elem = bli_min( n_shift + j + 1, n_elem_max ); \
\
                x1     = x + (ij0+j  )*ldx + (0  )*incx; \
                x0     = x1; \
                chi1   = x1 + (n_elem-1)*incx; \
\
                /*PASTEMAC2(ch,kername,BLIS_TAPI_EX_SUF)*/ \
                PASTEMAC(ch,kername) \
                ( \
                  n_elem, \
                  x1, incx, \
                  cntx, \
                  rntm  \
                ); \
\
                ( void )x0; \
                ( void )chi1; \
                /* We want positive diagonal elements between 1 and 2. */ \
/*
                PASTEMAC(ch,abval2s)( *chi1, *chi1 ); \
                PASTEMAC(ch,adds)( *one, *chi1 ); \
*/ \
\
                /* Scale the super-diagonal elements by 1/max(m,n). */ \
/*
                PASTEMAC(ch,scalv) \
                ( \
                  BLIS_NO_CONJUGATE, \
                  n_elem - 1, \
                  &beta, \
                  x0, incx, \
                  cntx  \
                ); \
*/ \
            } \
        } \
        else if ( bli_is_lower( uplox_eff ) ) \
        { \
            for ( j = 0; j < n_iter; ++j ) \
            { \
                i      = bli_max( 0, ( doff_t )j - ( doff_t )n_shift ); \
                n_elem = n_elem_max - i; \
\
                x1     = x + (j  )*ldx + (ij0+i  )*incx; \
                x2     = x1 + incx; \
                chi1   = x1; \
\
                /*PASTEMAC2(ch,kername,BLIS_TAPI_EX_SUF)*/ \
                PASTEMAC(ch,kername) \
                ( \
                  n_elem, \
                  x1, incx, \
                  cntx, \
                  rntm  \
                ); \
\
                ( void )x2; \
                ( void )chi1; \
                /* We want positive diagonal elements between 1 and 2. */ \
/*
                PASTEMAC(ch,abval2s)( *chi1, *chi1 ); \
                PASTEMAC(ch,adds)( *one, *chi1 ); \
*/ \
\
                /* Scale the sub-diagonal elements by 1/max(m,n). */ \
/*
                PASTEMAC(ch,scalv) \
                ( \
                  BLIS_NO_CONJUGATE, \
                  n_elem - 1, \
                  &beta, \
                  x2, incx, \
                  cntx  \
                ); \
*/ \
            } \
        } \
    } \
}

INSERT_GENTFUNC_BASIC( randm_unb_var1,  randv_unb_var1 )
INSERT_GENTFUNC_BASIC( randnm_unb_var1, randnv_unb_var1 )


#undef  GENTFUNCR
#define GENTFUNCR( ctype, ctype_r, ch, chr, varname ) \
\
void PASTEMAC(ch,varname) \
     ( \
       dim_t    n, \
       ctype*   x, inc_t incx, \
       ctype_r* scale, \
       ctype_r* sumsq, \
       cntx_t*  cntx, \
       rntm_t*  rntm  \
     ) \
{ \
	ctype_r zero_r = *PASTEMAC(chr,0); \
	ctype_r one_r  = *PASTEMAC(chr,1); \
\
	ctype*  chi1; \
	ctype_r chi1_r; \
	ctype_r chi1_i; \
	ctype_r scale_r; \
	ctype_r sumsq_r; \
	ctype_r abs_chi1_r; \
	ctype_r abs_chi1_i; \
	dim_t   i; \
\
	/* NOTE: This function attempts to mimic the algorithm for computing
	   the Frobenius norm in netlib LAPACK's ?lassq(). */ \
\
	/* Copy scale and sumsq to local variables. */ \
	PASTEMAC(chr,copys)( *scale, scale_r ); \
	PASTEMAC(chr,copys)( *sumsq, sumsq_r ); \
\
	chi1 = x; \
\
	for ( i = 0; i < n; ++i ) \
	{ \
		/* Get the real and imaginary components of chi1. */ \
		PASTEMAC2(ch,chr,gets)( *chi1, chi1_r, chi1_i ); \
\
		abs_chi1_r = bli_fabs( chi1_r ); \
		abs_chi1_i = bli_fabs( chi1_i ); \
\
		if ( bli_isnan( abs_chi1_r ) ) \
		{ \
			sumsq_r = abs_chi1_r; \
			scale_r = one_r; \
		} \
\
		if ( bli_isnan( abs_chi1_i ) ) \
		{ \
			sumsq_r = abs_chi1_i; \
			scale_r = one_r; \
		} \
\
		if ( bli_isnan( sumsq_r ) ) \
		{ \
			chi1 += incx; \
			continue; \
		} \
\
		if ( bli_isinf( abs_chi1_r ) ) \
		{ \
			sumsq_r = abs_chi1_r; \
			scale_r = one_r; \
		} \
\
		if ( bli_isinf( abs_chi1_i ) ) \
		{ \
			sumsq_r = abs_chi1_i; \
			scale_r = one_r; \
		} \
\
		if ( bli_isinf( sumsq_r ) ) \
		{ \
			chi1 += incx; \
			continue; \
		} \
\
		/* Accumulate real component into sumsq, adjusting scale if
		   needed. */ \
		if ( abs_chi1_r > zero_r ) \
		{ \
			if ( scale_r < abs_chi1_r ) \
			{ \
				sumsq_r = one_r + \
				          sumsq_r * ( scale_r / abs_chi1_r ) * \
				                    ( scale_r / abs_chi1_r );  \
\
				PASTEMAC(chr,copys)( abs_chi1_r, scale_r ); \
			} \
			else \
			{ \
				sumsq_r = sumsq_r + ( abs_chi1_r / scale_r ) * \
				                    ( abs_chi1_r / scale_r );  \
			} \
		} \
\
		/* Accumulate imaginary component into sumsq, adjusting scale if
		   needed. */ \
		if ( abs_chi1_i > zero_r ) \
		{ \
			if ( scale_r < abs_chi1_i ) \
			{ \
				sumsq_r = one_r + \
				          sumsq_r * ( scale_r / abs_chi1_i ) * \
				                    ( scale_r / abs_chi1_i );  \
\
				PASTEMAC(chr,copys)( abs_chi1_i, scale_r ); \
			} \
			else \
			{ \
				sumsq_r = sumsq_r + ( abs_chi1_i / scale_r ) * \
				                    ( abs_chi1_i / scale_r );  \
			} \
		} \
\
		chi1 += incx; \
	} \
\
	/* Store final values of scale and sumsq to output variables. */ \
	PASTEMAC(chr,copys)( scale_r, *scale ); \
	PASTEMAC(chr,copys)( sumsq_r, *sumsq ); \
}

INSERT_GENTFUNCR_BASIC0( sumsqv_unb_var1 )

// -----------------------------------------------------------------------------

#undef  GENTFUNC
#define GENTFUNC( ctype, ch, opname ) \
\
bool PASTEMAC(ch,opname) \
     ( \
       conj_t  conjx, \
       dim_t   n, \
       ctype*  x, inc_t incx, \
       ctype*  y, inc_t incy  \
     ) \
{ \
	for ( dim_t i = 0; i < n; ++i ) \
	{ \
		ctype* chi1 = x + (i  )*incx; \
		ctype* psi1 = y + (i  )*incy; \
\
		ctype chi1c; \
\
		if ( bli_is_conj( conjx ) ) { PASTEMAC(ch,copyjs)( *chi1, chi1c ); } \
		else                        { PASTEMAC(ch,copys)( *chi1, chi1c ); } \
\
		if ( !PASTEMAC(ch,eq)( chi1c, *psi1 ) ) \
			return FALSE; \
	} \
\
	return TRUE; \
}

INSERT_GENTFUNC_BASIC0( eqv_unb_var1 )


#undef  GENTFUNC
#define GENTFUNC( ctype, ch, opname ) \
\
bool PASTEMAC(ch,opname) \
     ( \
       doff_t  diagoffx, \
       diag_t  diagx, \
       uplo_t  uplox, \
       trans_t transx, \
       dim_t   m, \
       dim_t   n, \
       ctype*  x, inc_t rs_x, inc_t cs_x, \
       ctype*  y, inc_t rs_y, inc_t cs_y  \
     ) \
{ \
	uplo_t   uplox_eff; \
	conj_t   conjx; \
	dim_t    n_iter; \
	dim_t    n_elem_max; \
	inc_t    ldx, incx; \
	inc_t    ldy, incy; \
	dim_t    ij0, n_shift; \
\
	/* Set various loop parameters. */ \
	bli_set_dims_incs_uplo_2m \
	( \
	  diagoffx, diagx, transx, \
	  uplox, m, n, rs_x, cs_x, rs_y, cs_y, \
	  &uplox_eff, &n_elem_max, &n_iter, &incx, &ldx, &incy, &ldy, \
	  &ij0, &n_shift \
	); \
\
	/* In the odd case where we are comparing against a complete unstored
	   matrix, we assert equality. Why? We assume the matrices are equal
	   unless we can find two corresponding elements that are unequal. So
	   if there are no elements, there is no inequality. Granted, this logic
	   is strange to think about no matter what, and thankfully it should
	   never be used under normal usage. */ \
	if ( bli_is_zeros( uplox_eff ) ) return TRUE; \
\
	/* Extract the conjugation component from the transx parameter. */ \
	conjx = bli_extract_conj( transx ); \
\
	/* Handle dense and upper/lower storage cases separately. */ \
	if ( bli_is_dense( uplox_eff ) ) \
	{ \
		for ( dim_t j = 0; j < n_iter; ++j ) \
		{ \
			const dim_t n_elem = n_elem_max; \
\
			ctype* x1 = x + (j  )*ldx + (0  )*incx; \
			ctype* y1 = y + (j  )*ldy + (0  )*incy; \
\
			for ( dim_t i = 0; i < n_elem; ++i ) \
			{ \
				ctype* x11 = x1 + (i  )*incx; \
				ctype* y11 = y1 + (i  )*incy; \
				ctype  x11c; \
\
				if ( bli_is_conj( conjx ) ) { PASTEMAC(ch,copyjs)( *x11, x11c ); } \
				else                        { PASTEMAC(ch,copys)( *x11, x11c ); } \
\
				if ( !PASTEMAC(ch,eq)( x11c, *y11 ) ) \
					return FALSE; \
			} \
		} \
	} \
	else \
	{ \
		if ( bli_is_upper( uplox_eff ) ) \
		{ \
			for ( dim_t j = 0; j < n_iter; ++j ) \
			{ \
				const dim_t n_elem = bli_min( n_shift + j + 1, n_elem_max ); \
\
				ctype* x1 = x + (ij0+j  )*ldx + (0  )*incx; \
				ctype* y1 = y + (ij0+j  )*ldy + (0  )*incy; \
\
				for ( dim_t i = 0; i < n_elem; ++i ) \
				{ \
					ctype* x11 = x1 + (i  )*incx; \
					ctype* y11 = y1 + (i  )*incy; \
					ctype  x11c; \
\
					if ( bli_is_conj( conjx ) ) { PASTEMAC(ch,copyjs)( *x11, x11c ); } \
					else                        { PASTEMAC(ch,copys)( *x11, x11c ); } \
\
					if ( !PASTEMAC(ch,eq)( x11c, *y11 ) ) \
						return FALSE; \
				} \
			} \
		} \
		else if ( bli_is_lower( uplox_eff ) ) \
		{ \
			for ( dim_t j = 0; j < n_iter; ++j ) \
			{ \
				const dim_t offi   = bli_max( 0, ( doff_t )j - ( doff_t )n_shift ); \
				const dim_t n_elem = n_elem_max - offi; \
\
				ctype* x1 = x + (j  )*ldx + (ij0+offi  )*incx; \
				ctype* y1 = y + (j  )*ldy + (ij0+offi  )*incy; \
\
				for ( dim_t i = 0; i < n_elem; ++i ) \
				{ \
					ctype* x11 = x1 + (i  )*incx; \
					ctype* y11 = y1 + (i  )*incy; \
					ctype  x11c; \
\
					if ( bli_is_conj( conjx ) ) { PASTEMAC(ch,copyjs)( *x11, x11c ); } \
					else                        { PASTEMAC(ch,copys)( *x11, x11c ); } \
\
					if ( !PASTEMAC(ch,eq)( x11c, *y11 ) ) \
						return FALSE; \
				} \
			} \
		} \
	} \
\
	return TRUE; \
}

INSERT_GENTFUNC_BASIC0( eqm_unb_var1 )


#undef  GENTFUNC
#define GENTFUNC( ctype, ch, opname ) \
\
void PASTEMAC(ch,opname) \
     ( \
       FILE*  file, \
       char*  s1, \
       dim_t  n, \
       ctype* x, inc_t incx, \
       char*  format, \
       char*  s2  \
     ) \
{ \
	dim_t  i; \
	ctype* chi1; \
	char   default_spec[32] = PASTEMAC(ch,formatspec)(); \
\
	if ( format == NULL ) format = default_spec; \
\
	chi1 = x; \
\
	fprintf( file, "%s\n", s1 ); \
\
	for ( i = 0; i < n; ++i ) \
	{ \
		PASTEMAC(ch,fprints)( file, format, *chi1 ); \
		fprintf( file, "\n" ); \
\
		chi1 += incx; \
	} \
\
	fprintf( file, "%s\n", s2 ); \
}

INSERT_GENTFUNC_BASIC0_I( fprintv )


#undef  GENTFUNC
#define GENTFUNC( ctype, ch, opname ) \
\
void PASTEMAC(ch,opname) \
     ( \
       FILE*  file, \
       char*  s1, \
       dim_t  m, \
       dim_t  n, \
       ctype* x, inc_t rs_x, inc_t cs_x, \
       char*  format, \
       char*  s2  \
     ) \
{ \
	dim_t  i, j; \
	ctype* chi1; \
	char   default_spec[32] = PASTEMAC(ch,formatspec)(); \
\
	if ( format == NULL ) format = default_spec; \
\
	fprintf( file, "%s\n", s1 ); \
\
	for ( i = 0; i < m; ++i ) \
	{ \
		for ( j = 0; j < n; ++j ) \
		{ \
			chi1 = (( ctype* ) x) + i*rs_x + j*cs_x; \
\
			PASTEMAC(ch,fprints)( file, format, *chi1 ); \
			fprintf( file, " " ); \
		} \
\
		fprintf( file, "\n" ); \
	} \
\
	fprintf( file, "%s\n", s2 ); \
	fflush( file ); \
}

INSERT_GENTFUNC_BASIC0_I( fprintm )

