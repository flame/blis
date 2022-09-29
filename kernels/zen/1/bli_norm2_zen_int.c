/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2021 - 2022, Advanced Micro Devices, Inc. All rights reserved.

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
#include "immintrin.h"
#include "blis.h"

// Union data structure to access AVX registers
// One 256-bit AVX register holds 8 SP elements. 
typedef union
{
    __m256  v;
    float   f[8] __attribute__( ( aligned( 64 ) ) );
} v8sf_t;

// Union data structure to access AVX registers
// One 256-bit AVX register holds 4 DP elements. 
typedef union
{
    __m256d v;
    double  d[4] __attribute__( ( aligned( 64 ) ) );
} v4df_t;

// Return a mask which indicates either:
// v <= t or v >= T
#define CMP256( v, t, T ) \
	_mm256_or_pd( _mm256_cmp_pd( v, t, _CMP_LE_OS ), _mm256_cmp_pd( v, T, _CMP_GE_OS ) );

// Returns true if any of the values in the mask vector is true, 
// and false, otherwise.
static inline bool bli_horizontal_or( __m256d a ) { return ! _mm256_testz_pd( a, a ); }

// Optimized function that computes the Frobenius norm using AVX2 intrinsics.
void bli_dnorm2fv_unb_var1_avx2
    (
       dim_t    n,
       double*   x, inc_t incx,
       double* norm,
       cntx_t*  cntx
    )
{
    AOCL_DTL_TRACE_ENTRY( AOCL_DTL_LEVEL_TRACE_3 );

    double sumsq = 0;
    dim_t i = 0;
    dim_t n_remainder = 0;
    double  *x_buf = x;

    // Early return if n<=0 or incx=0
    if ( ( n <= 0) || ( incx == 0 ) )
    {
        return;
    }

    // Memory pool declarations for packing vector X.
    // Initialize mem pool buffer to NULL and size to 0.
    // "buf" and "size" fields are assigned once memory
    // is allocated from the pool in bli_membrk_acquire_m().
    // This will ensure bli_mem_is_alloc() will be passed on
    // an allocated memory if created or a NULL.
    mem_t   mem_bufX = {0};
    rntm_t  rntm;

    // Packing for non-unit strided vector x.
    if ( incx != 1 )
    {
        // In order to get the buffer from pool via rntm access to memory broker
        //is needed. Following are initializations for rntm.
        bli_rntm_init_from_global( &rntm );
        bli_rntm_set_num_threads_only( 1, &rntm );
        bli_membrk_rntm_set_membrk( &rntm );

        // Calculate the size required for "n" double elements in vector x.
        size_t buffer_size = n * sizeof( double );

        #ifdef BLIS_ENABLE_MEM_TRACING
            printf( "bli_dnorm2fv_unb_var1(): get mem pool block\n" );
        #endif

        // Acquire a Buffer(n*size(double)) from the memory broker
        // and save the associated mem_t entry to mem_bufX.
        bli_membrk_acquire_m
        (
            &rntm,
            buffer_size,
            BLIS_BUFFER_FOR_B_PANEL,
            &mem_bufX
        );

        // Continue packing X if buffer memory is allocated.
        if ( ( bli_mem_is_alloc( &mem_bufX ) ) )
        {
            x_buf = bli_mem_buffer( &mem_bufX );
            // Pack vector x with non-unit stride to a temp buffer x_buf with unit stride.
            for ( dim_t x_index = 0; x_index < n; x_index++ )
            {
                if ( incx > 0 )
                {
                    *( x_buf + x_index ) = *( x + ( x_index * incx ) );
                }
                else
                {
                    *( x_buf + x_index ) =  *( x + ( - ( n - x_index - 1 ) * incx ) );
                }
            }
        }
    }

    double *xt = x_buf;

    // Compute the sum of squares on 3 accumulators to avoid overflow
    // and underflow, depending on the vector element value.
    // Accumulator for small values; using scaling to avoid underflow.
    double sum_sml = 0;
   // Accumulator for medium values; no scaling required.
    double sum_med = 0;
    // Accumulator for big values; using scaling to avoid overflow.
    double sum_big = 0;

    // Constants chosen to minimize roundoff, according to Blue's algorithm.
    const double thres_sml = pow( ( double )FLT_RADIX,    ceil( ( DBL_MIN_EXP - 1 )  * 0.5 ) );
    const double thres_big = pow( ( double )FLT_RADIX,   floor( ( DBL_MAX_EXP - 52)  * 0.5 ) );
    const double scale_sml = pow( ( double )FLT_RADIX, - floor( ( DBL_MIN_EXP - 53 ) * 0.5 ) );
    const double scale_big = pow( ( double )FLT_RADIX,  - ceil( ( DBL_MAX_EXP - 52 ) * 0.5 ) );

    double scale;
    double abs_chi;
    bool isbig = false;

    if ( n > 4 )
    {
        // Constants used for comparisons.
        v4df_t temp, thres_sml_vec, thres_big_vec, zerov, ymm0, ymm1;
        temp.v = _mm256_set1_pd( -0.0 );
        thres_sml_vec.v = _mm256_set1_pd( thres_sml );
        thres_big_vec.v = _mm256_set1_pd( thres_big );
        v4df_t x0v, x1v, mask_vec0, mask_vec1;
        zerov.v  = _mm256_setzero_pd();

        // Partial sums used for scaling.
        v4df_t sum_med_vec0, sum_big_vec0, sum_sml_vec0, sum_med_vec1, sum_big_vec1, sum_sml_vec1;
        sum_med_vec0.v = _mm256_setzero_pd();
        sum_big_vec0.v = _mm256_setzero_pd();
        sum_sml_vec0.v = _mm256_setzero_pd();
        sum_med_vec1.v = _mm256_setzero_pd();
        sum_big_vec1.v = _mm256_setzero_pd();
        sum_sml_vec1.v = _mm256_setzero_pd();

        for (; ( i + 8 ) <= n; i = i + 8)
        {
            x0v.v = _mm256_loadu_pd( xt );
            x1v.v = _mm256_loadu_pd( xt + 4 );

            // Getting the abs of the vector elements.
            x0v.v = _mm256_andnot_pd( temp.v, x0v.v );
            x1v.v = _mm256_andnot_pd( temp.v, x1v.v );

            // Check if any of the values is a NaN and if so, return.
            mask_vec0.v = _mm256_cmp_pd(x0v.v, x0v.v, _CMP_UNORD_Q);
            mask_vec1.v = _mm256_cmp_pd(x1v.v, x1v.v, _CMP_UNORD_Q);
            if ( bli_horizontal_or( mask_vec0.v ) )
            {
                *norm = NAN;
                return;
            }
            if ( bli_horizontal_or( mask_vec1.v ) )
            {
                *norm = NAN;
                return;
            }

            // Mask vectors which indicate whether
            // xi<=thres_sml or xi>=thres_big.
            mask_vec0.v = CMP256( x0v.v, thres_sml_vec.v, thres_big_vec.v );
            mask_vec1.v = CMP256( x1v.v, thres_sml_vec.v, thres_big_vec.v );

            if ( !bli_horizontal_or( mask_vec0.v ) )
            {
                // Scaling is not necessary; only medium values.
                sum_med_vec0.v = _mm256_fmadd_pd( x0v.v, x0v.v, sum_med_vec0.v );
            }
            else
            {
                // Mask vector which indicate whether xi > thres_big.
                mask_vec0.v = _mm256_cmp_pd( x0v.v, thres_big_vec.v, _CMP_GT_OQ );

                if ( bli_horizontal_or( mask_vec0.v ) )
                {
                    isbig = true;

                    // Fill sum_med vector without scaling.
                    ymm0.v = _mm256_blendv_pd( x0v.v, zerov.v, mask_vec0.v );
                    sum_med_vec0.v = _mm256_fmadd_pd( ymm0.v, ymm0.v, sum_med_vec0.v );

                    // Fill sum_big vector using scaling.
                    temp.v = _mm256_set1_pd( scale_big );
                    ymm0.v = _mm256_blendv_pd( zerov.v, temp.v, mask_vec0.v ); 
                    ymm0.v = _mm256_mul_pd( x0v.v, ymm0.v );
                    sum_big_vec0.v = _mm256_fmadd_pd( ymm0.v, ymm0.v, sum_big_vec0.v );
                    temp.v = _mm256_set1_pd( -0.0 );
                }
                else
                {
                    // Mask vector which indicates whether xi > thres_small.
                    mask_vec0.v = _mm256_cmp_pd( x0v.v, thres_sml_vec.v, _CMP_LT_OQ );
                    // Fill sum_med vector without scaling.
                    ymm0.v = _mm256_blendv_pd( x0v.v, zerov.v, mask_vec0.v );
                    sum_med_vec0.v = _mm256_fmadd_pd( ymm0.v, ymm0.v, sum_med_vec0.v );

                    // Accumulate small values only if there have not been any big values so far.
                    if ( !isbig )
                    {
                        // Fill sum_sml vector using scaling.
                        temp.v = _mm256_set1_pd( scale_sml );
                        ymm0.v = _mm256_blendv_pd( zerov.v, temp.v, mask_vec0.v );
                        ymm0.v = _mm256_mul_pd( x0v.v, ymm0.v );
                        sum_sml_vec0.v = _mm256_fmadd_pd( ymm0.v, ymm0.v, sum_sml_vec0.v );
                        temp.v = _mm256_set1_pd( -0.0 );
                    }
                }
            }

            if ( !bli_horizontal_or( mask_vec1.v ) )
            {
                // Scaling is not necessary; only medium values.
                sum_med_vec1.v = _mm256_fmadd_pd( x1v.v, x1v.v, sum_med_vec1.v );
            }
            else
            {
                // Mask vector which indicate whether xi > thres_big.
                mask_vec1.v = _mm256_cmp_pd( x1v.v, thres_big_vec.v, _CMP_GT_OQ );

                if ( bli_horizontal_or( mask_vec1.v ) )
                {
                    isbig = true;

                    // Fill sum_med vector without scaling.
                    ymm1.v = _mm256_blendv_pd( x1v.v, zerov.v, mask_vec1.v );
                    sum_med_vec1.v = _mm256_fmadd_pd( ymm1.v, ymm1.v, sum_med_vec1.v );

                    // Fill sum_big vector using scaling.
                    temp.v = _mm256_set1_pd( scale_big );
                    ymm1.v = _mm256_blendv_pd( zerov.v, temp.v, mask_vec1.v ); 
                    ymm1.v = _mm256_mul_pd( x1v.v, ymm1.v );
                    sum_big_vec1.v = _mm256_fmadd_pd( ymm1.v, ymm1.v, sum_big_vec1.v ); 
                    temp.v = _mm256_set1_pd( -0.0 );
                }
                else
                {
                    // Mask vector which indicates whether xi > thres_small.
                    mask_vec1.v = _mm256_cmp_pd( x1v.v, thres_sml_vec.v, _CMP_LT_OQ );
                    // Fill sum_med vector without scaling.
                    ymm1.v = _mm256_blendv_pd( x1v.v, zerov.v, mask_vec1.v );
                    sum_med_vec1.v = _mm256_fmadd_pd( ymm1.v, ymm1.v, sum_med_vec1.v );

                    // Accumulate small values only if there have not been any big values so far.
                    if ( !isbig )
                    {
                        // Fill sum_sml vector using scaling.
                        temp.v = _mm256_set1_pd( scale_sml );
                        ymm1.v = _mm256_blendv_pd( zerov.v, temp.v, mask_vec1.v );
                        ymm1.v = _mm256_mul_pd( x1v.v, ymm1.v );
                        sum_sml_vec1.v = _mm256_fmadd_pd( ymm1.v, ymm1.v, sum_sml_vec1.v );
                        temp.v = _mm256_set1_pd( -0.0 );
                    }
                }
            }

            xt += 8;
        }

        for ( ; ( i + 4 ) <= n; i = i + 4 )
        {
            x0v.v = _mm256_loadu_pd( xt );

            // Getting the abs of the vector elements.
            x0v.v = _mm256_andnot_pd( temp.v, x0v.v );

            // Check if any of the values is a NaN and if so, return.
            mask_vec0.v = _mm256_cmp_pd(x0v.v, x0v.v, _CMP_UNORD_Q);
            if ( bli_horizontal_or( mask_vec0.v ) )
            {
                *norm = NAN;
                return;
            }

            // Mask vectors which indicate whether
            // xi<=thres_sml or xi>=thres_big.
            mask_vec0.v = CMP256( x0v.v, thres_sml_vec.v, thres_big_vec.v );

            if ( !bli_horizontal_or( mask_vec0.v ) )
            {
                // Scaling is not necessary; only medium values.
                sum_med_vec0.v = _mm256_fmadd_pd( x0v.v, x0v.v, sum_med_vec0.v );
            }
            else
            {
                // Mask vector which indicate whether xi > thres_big.
                mask_vec0.v = _mm256_cmp_pd( x0v.v, thres_big_vec.v, _CMP_GT_OQ );

                if ( bli_horizontal_or( mask_vec0.v ) )
                {
                    isbig = true;

                    // Fill sum_med vector without scaling.
                    ymm0.v = _mm256_blendv_pd( x0v.v, zerov.v, mask_vec0.v );
                    sum_med_vec0.v = _mm256_fmadd_pd( ymm0.v, ymm0.v, sum_med_vec0.v );

                    // Fill sum_big vector using scaling.
                    temp.v = _mm256_set1_pd( scale_big );
                    ymm0.v = _mm256_blendv_pd( zerov.v, temp.v, mask_vec0.v );
                    ymm0.v = _mm256_mul_pd( x0v.v, ymm0.v );
                    sum_big_vec0.v = _mm256_fmadd_pd( ymm0.v, ymm0.v, sum_big_vec0.v );
                    temp.v = _mm256_set1_pd( -0.0 );
                }
                else
                {
                    // Mask vector which indicates whether xi > thres_small.
                    mask_vec0.v = _mm256_cmp_pd( x0v.v, thres_sml_vec.v, _CMP_LT_OQ );
                    // Fill sum_med vector without scaling.
                    ymm0.v = _mm256_blendv_pd( x0v.v, zerov.v, mask_vec0.v );
                    sum_med_vec0.v = _mm256_fmadd_pd( ymm0.v, ymm0.v, sum_med_vec0.v );

                    // Accumulate small values only if there have not been any big values so far.
                    if ( !isbig )
                    {
                        // Fill sum_sml vector using scaling.
                        temp.v = _mm256_set1_pd( scale_sml );
                        ymm0.v = _mm256_blendv_pd( zerov.v, temp.v, mask_vec0.v );
                        ymm0.v = _mm256_mul_pd( x0v.v, ymm0.v );
                        sum_sml_vec0.v = _mm256_fmadd_pd( ymm0.v, ymm0.v, sum_sml_vec0.v );
                        temp.v = _mm256_set1_pd( -0.0 );
                    }
                }
            }
            xt += 4;
        }

        sum_sml_vec0.v = _mm256_add_pd( sum_sml_vec0.v, sum_sml_vec1.v );
        sum_med_vec0.v = _mm256_add_pd( sum_med_vec0.v, sum_med_vec1.v );
        sum_big_vec0.v = _mm256_add_pd( sum_big_vec0.v, sum_big_vec1.v );

        sum_sml += sum_sml_vec0.v[0] + sum_sml_vec0.v[1]
                + sum_sml_vec0.v[2] + sum_sml_vec0.v[3];
        sum_med += sum_med_vec0.v[0] + sum_med_vec0.v[1]
                + sum_med_vec0.v[2] + sum_med_vec0.v[3];
        sum_big += sum_big_vec0.v[0] + sum_big_vec0.v[1]
                + sum_big_vec0.v[2] + sum_big_vec0.v[3];
    }

    n_remainder = n - i;
    bool hasInf = false;
    if ( ( n_remainder > 0 ) )
    {
        // Put first the most likely to happen to avoid evaluations on if statements.
        for (i = 0; i < n_remainder; i++)
        {
            abs_chi = bli_fabs( *xt );
            // If any of the elements is NaN, then return NaN as a result.
            if ( bli_isnan( abs_chi ) )
            {
                *norm = abs_chi;
                return;
            }
            // Else, if any of the elements is an Inf, then return +Inf as a result.
            if ( bli_isinf( abs_chi ) )
            {
                *norm = abs_chi;
                // Instead of returning immediately, use this flag
                // to denote that there is an Inf element in the vector.
                // That is used to avoid cases where there is a NaN which comes
                // after an Inf.
                hasInf = true;
            }
            // Most likely case: medium values, not over/under-flow.
            if ( ( abs_chi <= thres_big ) && ( abs_chi >= thres_sml ) )
            {
                sum_med += abs_chi * abs_chi;
            }
            // Case where there could be an overflow. Scaling is required.
            else if ( abs_chi > thres_big )
            {
                sum_big += ( abs_chi * scale_big ) * ( abs_chi * scale_big );
                isbig = true;
            }
            // Case where there could be an underflow. Scaling is required.
            else if (  ( !isbig ) && ( abs_chi < thres_sml ) )
            {
                sum_sml += ( abs_chi * scale_sml ) * ( abs_chi * scale_sml );
            }
            xt++;
        }
    }

    // Early return if there is an Inf.
    if ( hasInf ) return;

    // Combine accumulators.
    if ( isbig )
    {
        // Combine sum_big and sum_med if sum_med > 0.
        if ( sum_med > 0.0 )
        {
            sum_big += ( sum_med * scale_big ) * scale_big;
        }
        scale = 1.0 / scale_big;
        sumsq = sum_big;
    }

    else if ( sum_sml > 0.0 )
    {
        // Combine sum_med and sum_sml if sum_sml>0.
        if ( sum_med > 0.0 )
        {
            sum_med = sqrt( sum_med );
            sum_sml = sqrt( sum_sml ) / scale_sml;
            double ymin, ymax;
            if ( sum_sml > sum_med )
            {
                ymin = sum_med;
                ymax = sum_sml;
            }
            else
            {
                ymin = sum_sml;
                ymax = sum_med;
            }
            scale = 1.0;
            sumsq = ymax * ymax * ( 1.0 + ( ymin / ymax ) * ( ymin / ymax ) );
        }
        else
        {
            scale = 1.0 / scale_sml;
            sumsq = sum_sml;
        }
    }
    else
    {
        // If all values are mid-range:
        scale = 1.0;
        sumsq = sum_med;
    }

    *norm = scale * sqrt( sumsq );

    if ( ( incx != 1 ) && bli_mem_is_alloc( &mem_bufX ) )
    {
        #ifdef BLIS_ENABLE_MEM_TRACING
            printf( "bli_dnorm2fv_unb_var1(): releasing mem pool block\n" );
        #endif
        // Return the buffer to pool.
        bli_membrk_release( &rntm , &mem_bufX );
    }

    AOCL_DTL_TRACE_EXIT( AOCL_DTL_LEVEL_TRACE_3 );

    return;
}

// Optimized function that computes the Frobenius norm using AVX2 intrinsics.
void bli_dznorm2fv_unb_var1_avx2
    (
       dim_t    n,
       dcomplex*   x, inc_t incx,
       double* norm,
       cntx_t*  cntx
    )
{
    AOCL_DTL_TRACE_ENTRY( AOCL_DTL_LEVEL_TRACE_3 );

    double sumsq = 0;
    dim_t i = 0;
    dim_t n_remainder = 0;
    dcomplex  *x_buf = x;

    // Early return if n<=0 or incx=0
    if ( ( n <= 0) || ( incx == 0 ) )
    {
        return;
    }

    // Memory pool declarations for packing vector X.
    // Initialize mem pool buffer to NULL and size to 0.
    // "buf" and "size" fields are assigned once memory
    // is allocated from the pool in bli_membrk_acquire_m().
    // This will ensure bli_mem_is_alloc() will be passed on
    // an allocated memory if created or a NULL.
    mem_t   mem_bufX = {0};
    rntm_t  rntm;

    // Packing for non-unit strided vector x.
    if ( incx != 1 )
    {
        // In order to get the buffer from pool via rntm access to memory broker
        //is needed. Following are initializations for rntm.
        bli_rntm_init_from_global( &rntm );
        bli_rntm_set_num_threads_only( 1, &rntm );
        bli_membrk_rntm_set_membrk( &rntm );

        // Calculate the size required for "n" dcomplex elements in vector x.
        size_t buffer_size = n * sizeof( dcomplex );

        #ifdef BLIS_ENABLE_MEM_TRACING
            printf( "bli_dznorm2fv_unb_var1(): get mem pool block\n" );
        #endif

        // Acquire a Buffer(n*size(dcomplex)) from the memory broker
        // and save the associated mem_t entry to mem_bufX.
        bli_membrk_acquire_m
        (
            &rntm,
            buffer_size,
            BLIS_BUFFER_FOR_B_PANEL,
            &mem_bufX
        );

        // Continue packing X if buffer memory is allocated.
        if ( ( bli_mem_is_alloc( &mem_bufX ) ) )
        {
            x_buf = bli_mem_buffer( &mem_bufX );
            // Pack vector x with non-unit stride to a temp buffer x_buf with unit stride.
            for ( dim_t x_index = 0; x_index < n; x_index++ )
            {
                if ( incx > 0 )
                {
                    *( x_buf + x_index ) = *( x + ( x_index * incx ) );
                }
                else
                {
                    *( x_buf + x_index ) =  *( x + ( - ( n - x_index - 1 ) * incx ) );
                }
            }
        }
    }

    dcomplex *xt = x_buf;

    // Compute the sum of squares on 3 accumulators to avoid overflow
    // and underflow, depending on the vector element value.
    // Accumulator for small values; using scaling to avoid underflow.
    double sum_sml = 0;
   // Accumulator for medium values; no scaling required.
    double sum_med = 0;
    // Accumulator for big values; using scaling to avoid overflow.
    double sum_big = 0;

    // Constants chosen to minimize roundoff, according to Blue's algorithm.
    const double thres_sml = pow( ( double )FLT_RADIX,    ceil( ( DBL_MIN_EXP - 1 )  * 0.5 ) );
    const double thres_big = pow( ( double )FLT_RADIX,   floor( ( DBL_MAX_EXP - 52)  * 0.5 ) );
    const double scale_sml = pow( ( double )FLT_RADIX, - floor( ( DBL_MIN_EXP - 53 ) * 0.5 ) );
    const double scale_big = pow( ( double )FLT_RADIX,  - ceil( ( DBL_MAX_EXP - 52 ) * 0.5 ) );

    double scale;
    double abs_chi;
    bool isbig = false;

    if ( n > 2 )
    {
        // Constants used for comparisons.
        v4df_t temp, thres_sml_vec, thres_big_vec, zerov, ymm0, ymm1;
        temp.v = _mm256_set1_pd( -0.0 );
        thres_sml_vec.v = _mm256_set1_pd( thres_sml );
        thres_big_vec.v = _mm256_set1_pd( thres_big );
        v4df_t x0v, x1v, mask_vec0, mask_vec1;
        zerov.v  = _mm256_setzero_pd();

        // Partial sums used for scaling.
        v4df_t sum_med_vec0, sum_big_vec0, sum_sml_vec0, sum_med_vec1, sum_big_vec1, sum_sml_vec1;
        sum_med_vec0.v = _mm256_setzero_pd();
        sum_big_vec0.v = _mm256_setzero_pd();
        sum_sml_vec0.v = _mm256_setzero_pd();
        sum_med_vec1.v = _mm256_setzero_pd();
        sum_big_vec1.v = _mm256_setzero_pd();
        sum_sml_vec1.v = _mm256_setzero_pd();

        for (; ( i + 4 ) <= n; i = i + 4)
        {
            x0v.v = _mm256_loadu_pd( xt );
            x1v.v = _mm256_loadu_pd( xt + 2 );

            // Getting the abs of the vector elements.
            x0v.v = _mm256_andnot_pd( temp.v, x0v.v );
            x1v.v = _mm256_andnot_pd( temp.v, x1v.v );

            // Check if any of the values is a NaN and if so, return.
            mask_vec0.v = _mm256_cmp_pd(x0v.v, x0v.v, _CMP_UNORD_Q);
            mask_vec1.v = _mm256_cmp_pd(x1v.v, x1v.v, _CMP_UNORD_Q);
            if ( bli_horizontal_or( mask_vec0.v ) )
            {
                *norm = NAN;
                return;
            }
            if ( bli_horizontal_or( mask_vec1.v ) )
            {
                *norm = NAN;
                return;
            }

            // Mask vectors which indicate whether
            // xi<=thres_sml or xi>=thres_big.
            mask_vec0.v = CMP256( x0v.v, thres_sml_vec.v, thres_big_vec.v );
            mask_vec1.v = CMP256( x1v.v, thres_sml_vec.v, thres_big_vec.v );

            if ( !bli_horizontal_or( mask_vec0.v ) )
            {
                // Scaling is not necessary; only medium values.
                sum_med_vec0.v = _mm256_fmadd_pd( x0v.v, x0v.v, sum_med_vec0.v );
            }
            else
            {
                // Mask vector which indicate whether xi > thres_big.
                mask_vec0.v = _mm256_cmp_pd( x0v.v, thres_big_vec.v, _CMP_GT_OQ );

                if ( bli_horizontal_or( mask_vec0.v ) )
                {
                    isbig = true;

                    // Fill sum_med vector without scaling.
                    ymm0.v = _mm256_blendv_pd( x0v.v, zerov.v, mask_vec0.v );
                    sum_med_vec0.v = _mm256_fmadd_pd( ymm0.v, ymm0.v, sum_med_vec0.v );

                    // Fill sum_big vector using scaling.
                    temp.v = _mm256_set1_pd( scale_big );
                    ymm0.v = _mm256_blendv_pd( zerov.v, temp.v, mask_vec0.v );
                    ymm0.v = _mm256_mul_pd( x0v.v, ymm0.v );
                    sum_big_vec0.v = _mm256_fmadd_pd( ymm0.v, ymm0.v, sum_big_vec0.v );
                    temp.v = _mm256_set1_pd( -0.0 );
                }
                else
                {
                    // Mask vector which indicates whether xi > thres_small.
                    mask_vec0.v = _mm256_cmp_pd( x0v.v, thres_sml_vec.v, _CMP_LT_OQ );
                    // Fill sum_med vector without scaling.
                    ymm0.v = _mm256_blendv_pd( x0v.v, zerov.v, mask_vec0.v );
                    sum_med_vec0.v = _mm256_fmadd_pd( ymm0.v, ymm0.v, sum_med_vec0.v );

                    // Accumulate small values only if there have not been any big values so far.
                    if ( !isbig )
                    {
                        // Fill sum_sml vector using scaling.
                        temp.v = _mm256_set1_pd( scale_sml );
                        ymm0.v = _mm256_blendv_pd( zerov.v, temp.v, mask_vec0.v );
                        ymm0.v = _mm256_mul_pd( x0v.v, ymm0.v );
                        sum_sml_vec0.v = _mm256_fmadd_pd( ymm0.v, ymm0.v, sum_sml_vec0.v );
                        temp.v = _mm256_set1_pd( -0.0 );
                    }
                }
            }

            if ( !bli_horizontal_or( mask_vec1.v ) )
            {
                // Scaling is not necessary; only medium values.
                sum_med_vec1.v = _mm256_fmadd_pd( x1v.v, x1v.v, sum_med_vec1.v );
            }
            else
            {
                // Mask vector which indicate whether xi > thres_big.
                mask_vec1.v = _mm256_cmp_pd( x1v.v, thres_big_vec.v, _CMP_GT_OQ );

                if ( bli_horizontal_or( mask_vec1.v ) )
                {
                    isbig = true;

                    // Fill sum_med vector without scaling.
                    ymm1.v = _mm256_blendv_pd( x1v.v, zerov.v, mask_vec1.v );
                    sum_med_vec1.v = _mm256_fmadd_pd( ymm1.v, ymm1.v, sum_med_vec1.v );

                    // Fill sum_big vector using scaling.
                    temp.v = _mm256_set1_pd( scale_big );
                    ymm1.v = _mm256_blendv_pd( zerov.v, temp.v, mask_vec1.v );
                    ymm1.v = _mm256_mul_pd( x1v.v, ymm1.v );
                    sum_big_vec1.v = _mm256_fmadd_pd( ymm1.v, ymm1.v, sum_big_vec1.v ); 
                    temp.v = _mm256_set1_pd( -0.0 );
                }
                else
                {
                    // Mask vector which indicates whether xi > thres_small.
                    mask_vec1.v = _mm256_cmp_pd( x1v.v, thres_sml_vec.v, _CMP_LT_OQ );
                    // Fill sum_med vector without scaling.
                    ymm1.v = _mm256_blendv_pd( x1v.v, zerov.v, mask_vec1.v );
                    sum_med_vec1.v = _mm256_fmadd_pd( ymm1.v, ymm1.v, sum_med_vec1.v );

                    // Accumulate small values only if there have not been any big values so far.
                    if ( !isbig )
                    {
                        // Fill sum_sml vector using scaling.
                        temp.v = _mm256_set1_pd( scale_sml );
                        ymm1.v = _mm256_blendv_pd( zerov.v, temp.v, mask_vec1.v );
                        ymm1.v = _mm256_mul_pd( x1v.v, ymm1.v );
                        sum_sml_vec1.v = _mm256_fmadd_pd( ymm1.v, ymm1.v, sum_sml_vec1.v );
                        temp.v = _mm256_set1_pd( -0.0 );
                    }
                }
            }

            xt += 4;
        }

        for ( ; ( i + 2 ) <= n; i = i + 2 )
        {
            x0v.v = _mm256_loadu_pd( xt );

            // Getting the abs of the vector elements.
            x0v.v = _mm256_andnot_pd( temp.v, x0v.v );

            // Check if any of the values is a NaN and if so, return.
            mask_vec0.v = _mm256_cmp_pd(x0v.v, x0v.v, _CMP_UNORD_Q);
            if ( bli_horizontal_or( mask_vec0.v ) )
            {
                *norm = NAN;
                return;
            }

            // Mask vectors which indicate whether
            // xi<=thres_sml or xi>=thres_big.
            mask_vec0.v = CMP256( x0v.v, thres_sml_vec.v, thres_big_vec.v );

            if ( !bli_horizontal_or( mask_vec0.v ) )
            {
                // Scaling is not necessary; only medium values.
                sum_med_vec0.v = _mm256_fmadd_pd( x0v.v, x0v.v, sum_med_vec0.v );
            }
            else
            {
                // Mask vector which indicate whether xi > thres_big.
                mask_vec0.v = _mm256_cmp_pd( x0v.v, thres_big_vec.v, _CMP_GT_OQ );

                if ( bli_horizontal_or( mask_vec0.v ) )
                {
                    isbig = true;

                    // Fill sum_med vector without scaling.
                    ymm0.v = _mm256_blendv_pd( x0v.v, zerov.v, mask_vec0.v );
                    sum_med_vec0.v = _mm256_fmadd_pd( ymm0.v, ymm0.v, sum_med_vec0.v );

                    // Fill sum_big vector using scaling.
                    temp.v = _mm256_set1_pd( scale_big );
                    ymm0.v = _mm256_blendv_pd( zerov.v, temp.v, mask_vec0.v );
                    ymm0.v = _mm256_mul_pd( x0v.v, ymm0.v );
                    sum_big_vec0.v = _mm256_fmadd_pd( ymm0.v, ymm0.v, sum_big_vec0.v );
                    temp.v = _mm256_set1_pd( -0.0 );
                }
                else
                {
                    // Mask vector which indicates whether xi > thres_small.
                    mask_vec0.v = _mm256_cmp_pd( x0v.v, thres_sml_vec.v, _CMP_LT_OQ );
                    // Fill sum_med vector without scaling.
                    ymm0.v = _mm256_blendv_pd( x0v.v, zerov.v, mask_vec0.v );
                    sum_med_vec0.v = _mm256_fmadd_pd( ymm0.v, ymm0.v, sum_med_vec0.v );

                    // Accumulate small values only if there have not been any big values so far.
                    if ( !isbig )
                    {
                        // Fill sum_sml vector using scaling.
                        temp.v = _mm256_set1_pd( scale_sml );
                        ymm0.v = _mm256_blendv_pd( zerov.v, temp.v, mask_vec0.v );
                        ymm0.v = _mm256_mul_pd( x0v.v, ymm0.v );
                        sum_sml_vec0.v = _mm256_fmadd_pd( ymm0.v, ymm0.v, sum_sml_vec0.v );
                        temp.v = _mm256_set1_pd( -0.0 );
                    }
                }
            }
            xt += 2;
        }

        sum_sml_vec0.v = _mm256_add_pd( sum_sml_vec0.v, sum_sml_vec1.v );
        sum_med_vec0.v = _mm256_add_pd( sum_med_vec0.v, sum_med_vec1.v );
        sum_big_vec0.v = _mm256_add_pd( sum_big_vec0.v, sum_big_vec1.v );

        sum_sml += sum_sml_vec0.v[0] + sum_sml_vec0.v[1]
                + sum_sml_vec0.v[2] + sum_sml_vec0.v[3];
        sum_med += sum_med_vec0.v[0] + sum_med_vec0.v[1]
                + sum_med_vec0.v[2] + sum_med_vec0.v[3];
        sum_big += sum_big_vec0.v[0] + sum_big_vec0.v[1]
                + sum_big_vec0.v[2] + sum_big_vec0.v[3];
    }

    n_remainder = n - i;
    bool hasInf = false;
    if ( ( n_remainder > 0 ) )
    {
        // Put first the most likely to happen to avoid evaluations on if statements.
        for (i = 0; i < n_remainder; i++)
        {
            // Get real and imaginary component of the vector element.
            double chi_r, chi_i;
            bli_zdgets(*xt, chi_r, chi_i);

            // Start with accumulating the real component of the vector element.
            abs_chi = bli_fabs( chi_r );
            // If any of the elements is NaN, then return NaN as a result.
            if ( bli_isnan( abs_chi ) )
            {
                *norm = abs_chi;
                return;
            }
            // Else, if any of the elements is an Inf, then return +Inf as a result.
            if ( bli_isinf( abs_chi ) )
            {
                *norm = abs_chi;
                // Instead of returning immediately, use this flag
                // to denote that there is an Inf element in the vector.
                // That is used to avoid cases where there is a NaN which comes
                // after an Inf.
                hasInf = true;
            }
            // Most likely case: medium values, not over/under-flow.
            if ( ( abs_chi <= thres_big ) && ( abs_chi >= thres_sml ) )
            {
                sum_med += abs_chi * abs_chi;
            }
            // Case where there could be an overflow. Scaling is required.
            else if ( abs_chi > thres_big )
            {
                sum_big += ( abs_chi * scale_big ) * ( abs_chi * scale_big );
                isbig = true;
            }
            // Case where there could be an underflow. Scaling is required.
            else if ( ( !isbig ) && ( abs_chi < thres_sml ) )
            {
                sum_sml += ( abs_chi * scale_sml ) * ( abs_chi * scale_sml );
            }

            // Accumulate the imaginary component of the vector element.
            abs_chi = bli_fabs( chi_i );
            // If any of the elements is NaN, then return NaN as a result.
            if ( bli_isnan( abs_chi ) )
            {
                *norm = abs_chi;
                return;
            }
            // Else, if any of the elements is an Inf, then return +Inf as a result.
            if ( bli_isinf( abs_chi ) )
            {
                *norm = abs_chi;
                // Instead of returning immediately, use this flag
                // to denote that there is an Inf element in the vector.
                // That is used to avoid cases where there is a NaN which comes
                // after an Inf.
                hasInf = true;
            }
            // Most likely case: medium values, not over/under-flow.
            if ( ( abs_chi <= thres_big ) && ( abs_chi >= thres_sml ) )
            {
                sum_med += abs_chi * abs_chi;
            }
            // Case where there could be an overflow. Scaling is required.
            else if ( abs_chi > thres_big )
            {
                sum_big += ( abs_chi * scale_big ) * ( abs_chi * scale_big );
                isbig = true;
            }
            // Case where there could be an underflow. Scaling is required.
            else if ( ( !isbig ) && ( abs_chi < thres_sml ) )
            {
                sum_sml += ( abs_chi * scale_sml ) * ( abs_chi * scale_sml );
            }

            xt++;
        }
    }

    // Early return if there is an Inf.
    if ( hasInf ) return;

    // Combine accumulators.
    if ( isbig )
    {
        // Combine sum_big and sum_med if sum_med > 0.
        if ( sum_med > 0.0 )
        {
            sum_big += ( sum_med * scale_big ) * scale_big;
        }
        scale = 1.0 / scale_big;
        sumsq = sum_big;
    }

    else if ( sum_sml > 0.0 )
    {
        // Combine sum_med and sum_sml if sum_sml>0.
        if ( sum_med > 0.0 )
        {
            sum_med = sqrt( sum_med );
            sum_sml = sqrt( sum_sml ) / scale_sml;
            double ymin, ymax;
            if ( sum_sml > sum_med )
            {
                ymin = sum_med;
                ymax = sum_sml;
            }
            else
            {
                ymin = sum_sml;
                ymax = sum_med;
            }
            scale = 1.0;
            sumsq = ymax * ymax * ( 1.0 + ( ymin / ymax ) * ( ymin / ymax ) );
        }
        else
        {
            scale = 1.0 / scale_sml;
            sumsq = sum_sml;
        }
    }
    else
    {
        // If all values are mid-range:
        scale = 1.0;
        sumsq = sum_med;
    }

    *norm = scale * sqrt( sumsq );

    if ( ( incx != 1 ) && bli_mem_is_alloc( &mem_bufX ) )
    {
        #ifdef BLIS_ENABLE_MEM_TRACING
            printf( "bli_dznorm2fv_unb_var1(): releasing mem pool block\n" );
        #endif
        // Return the buffer to pool.
        bli_membrk_release( &rntm , &mem_bufX );
    }

    AOCL_DTL_TRACE_EXIT( AOCL_DTL_LEVEL_TRACE_3 );

    return;
}