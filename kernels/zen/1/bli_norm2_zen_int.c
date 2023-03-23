/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2021 - 2023, Advanced Micro Devices, Inc. All rights reserved.

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
#define CMP256_sf( v, t, T ) \
	_mm256_or_ps( _mm256_cmp_ps( v, t, _CMP_LE_OS ), _mm256_cmp_ps( v, T, _CMP_GE_OS ) );

#define CMP256_df( v, t, T ) \
	_mm256_or_pd( _mm256_cmp_pd( v, t, _CMP_LE_OS ), _mm256_cmp_pd( v, T, _CMP_GE_OS ) );

// Returns true if any of the values in the mask vector a is true, 
// and false, otherwise.
// In more detail, __mm256_testz_ps() performs the bitwise (a AND b) operation and returns:
//    1 if the sign bit of all bitwise operations is 0,
//    0 if at least one of the sign bits of each bitwise operation is 1.
// The sign bit of (a AND a) will be 1 iff the sign bit of a is 1, and 0 otherwise.
// That means that __mm256_testz_ps(a,a) returns:
//    1 if the sign bit of all elements in a is 0,
//    0 if at least one of the sign bits of a is 1.
// Because of the negation, bli_horizontal_or_sf() returns:
//    0 if the sign bit of all elements in a is 0,
//    1 if at least one of the sign bits of a is 1. 
// Since a is the result of a masking operation, bli_horizontal_or_sf() returns:
//    0 (false) if the mask is false for all elements in a,
//    1 (true)  if the mask is true for at least one element in a.
static inline bool bli_horizontal_or_sf( __m256  a ) { return ! _mm256_testz_ps( a, a ); }
static inline bool bli_horizontal_or_df( __m256d a ) { return ! _mm256_testz_pd( a, a ); }

float horizontal_add_sf(__m256 const a) {
    __m256 t1 = _mm256_hadd_ps(a, a);
    __m256 t2 = _mm256_hadd_ps(t1,t1);
    __m128 t3 = _mm256_extractf128_ps(t2,1);
    __m128 t4 = _mm_add_ss(_mm256_castps256_ps128(t2),t3);
    return _mm_cvtss_f32(t4); // sign extend to 32 bits
}

// Optimized function that computes the Frobenius norm using AVX2 intrinsics.
void bli_snorm2fv_unb_var1_avx2
    (
       dim_t    n,
       float*   x, inc_t incx,
       float* norm,
       cntx_t*  cntx
    )
{
    AOCL_DTL_TRACE_ENTRY( AOCL_DTL_LEVEL_TRACE_3 );
    
    float sumsq = 0.0f;
    dim_t i = 0;
    dim_t n_remainder = 0;
    float  *x_buf = x;

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

        // Calculate the size required for "n" float elements in vector x.
        size_t buffer_size = n * sizeof( float );

        #ifdef BLIS_ENABLE_MEM_TRACING
            printf( "bli_snorm2fv_unb_var1_avx2(): get mem pool block\n" );
        #endif

        // Acquire a Buffer(n*size(float)) from the memory broker
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
                *( x_buf + x_index ) = *( x + ( x_index * incx ) );
            }
        }
    }

    float *xt = x_buf;

    // Compute the sum of squares on 3 accumulators to avoid overflow
    // and underflow, depending on the vector element value.
    // Accumulator for small values; using scaling to avoid underflow.
    float sum_sml = 0.0f;
    // Accumulator for medium values; no scaling required.
    float sum_med = 0.0f;
    // Accumulator for big values; using scaling to avoid overflow.
    float sum_big = 0.0f;

    // Constants chosen to minimize roundoff, according to Blue's algorithm.
    const float thres_sml = powf( ( float )FLT_RADIX,    ceilf( ( FLT_MIN_EXP - 1 )  * 0.5f ) );
    const float thres_big = powf( ( float )FLT_RADIX,   floorf( ( FLT_MAX_EXP - 23)  * 0.5f ) );
    const float scale_sml = powf( ( float )FLT_RADIX, - floorf( ( FLT_MIN_EXP - 24 ) * 0.5f ) );
    const float scale_big = powf( ( float )FLT_RADIX,  - ceilf( ( FLT_MAX_EXP + 23 ) * 0.5f ) );

    float scale = 1.0f;
    float abs_chi;
    bool isbig = false;

    if ( n >= 64 )
    {
        // Constants used for comparisons.
        v8sf_t temp, thres_sml_vec, thres_big_vec, zerov;
        temp.v = _mm256_set1_ps( -0.0f );
        thres_sml_vec.v = _mm256_set1_ps( thres_sml );
        thres_big_vec.v = _mm256_set1_ps( thres_big );
        v8sf_t x0v, x1v, x2v, x3v;
        v8sf_t y0v, y1v, y2v, y3v;
        v8sf_t mask_vec0, mask_vec1, mask_vec2, mask_vec3;
        zerov.v  = _mm256_setzero_ps();

        // Partial sums used for scaling.
        v8sf_t sum_sml_vec0, sum_sml_vec1, sum_sml_vec2, sum_sml_vec3;
        sum_sml_vec0.v = _mm256_setzero_ps();
        sum_sml_vec1.v = _mm256_setzero_ps();
        sum_sml_vec2.v = _mm256_setzero_ps();
        sum_sml_vec3.v = _mm256_setzero_ps();

        v8sf_t sum_med_vec0, sum_med_vec1, sum_med_vec2, sum_med_vec3;
        sum_med_vec0.v = _mm256_setzero_ps();
        sum_med_vec1.v = _mm256_setzero_ps();
        sum_med_vec2.v = _mm256_setzero_ps();
        sum_med_vec3.v = _mm256_setzero_ps();

        v8sf_t sum_big_vec0, sum_big_vec1, sum_big_vec2, sum_big_vec3;
        sum_big_vec0.v = _mm256_setzero_ps();
        sum_big_vec1.v = _mm256_setzero_ps();
        sum_big_vec2.v = _mm256_setzero_ps();
        sum_big_vec3.v = _mm256_setzero_ps();

        for (; ( i + 32 ) <= n; i = i + 32)
        {
            x0v.v = _mm256_loadu_ps( xt );
            x1v.v = _mm256_loadu_ps( xt + 8 );
            x2v.v = _mm256_loadu_ps( xt + 16 );
            x3v.v = _mm256_loadu_ps( xt + 24 );

            // Getting the abs of the vector elements.
            x0v.v = _mm256_andnot_ps( temp.v, x0v.v );
            x1v.v = _mm256_andnot_ps( temp.v, x1v.v );
            x2v.v = _mm256_andnot_ps( temp.v, x2v.v );
            x3v.v = _mm256_andnot_ps( temp.v, x3v.v );

            // Check if any of the values is a NaN and if so, return.
            mask_vec0.v = _mm256_cmp_ps(x0v.v, x0v.v, _CMP_UNORD_Q);
            mask_vec1.v = _mm256_cmp_ps(x1v.v, x1v.v, _CMP_UNORD_Q);
            mask_vec2.v = _mm256_cmp_ps(x2v.v, x2v.v, _CMP_UNORD_Q);
            mask_vec3.v = _mm256_cmp_ps(x3v.v, x3v.v, _CMP_UNORD_Q);
            if ( bli_horizontal_or_sf( mask_vec0.v ) )
            {
                *norm = NAN;
                if ( ( incx != 1 ) && bli_mem_is_alloc( &mem_bufX ) )
                {
                    #ifdef BLIS_ENABLE_MEM_TRACING
                        printf( "bli_snorm2fv_unb_var1_avx2(): releasing mem pool block\n" );
                    #endif
                    // Return the buffer to pool.
                    bli_membrk_release( &rntm , &mem_bufX );
                }

                AOCL_DTL_TRACE_EXIT( AOCL_DTL_LEVEL_TRACE_3 );
                return;
            }
            if ( bli_horizontal_or_sf( mask_vec1.v ) )
            {
                *norm = NAN;
                if ( ( incx != 1 ) && bli_mem_is_alloc( &mem_bufX ) )
                {
                    #ifdef BLIS_ENABLE_MEM_TRACING
                        printf( "bli_snorm2fv_unb_var1_avx2(): releasing mem pool block\n" );
                    #endif
                    // Return the buffer to pool.
                    bli_membrk_release( &rntm , &mem_bufX );
                }

                AOCL_DTL_TRACE_EXIT( AOCL_DTL_LEVEL_TRACE_3 );
                return;
            }
            if ( bli_horizontal_or_sf( mask_vec2.v ) )
            {
                *norm = NAN;
                if ( ( incx != 1 ) && bli_mem_is_alloc( &mem_bufX ) )
                {
                    #ifdef BLIS_ENABLE_MEM_TRACING
                        printf( "bli_snorm2fv_unb_var1_avx2(): releasing mem pool block\n" );
                    #endif
                    // Return the buffer to pool.
                    bli_membrk_release( &rntm , &mem_bufX );
                }

                AOCL_DTL_TRACE_EXIT( AOCL_DTL_LEVEL_TRACE_3 );
                return;
            }
            if ( bli_horizontal_or_sf( mask_vec3.v ) )
            {
                *norm = NAN;
                if ( ( incx != 1 ) && bli_mem_is_alloc( &mem_bufX ) )
                {
                    #ifdef BLIS_ENABLE_MEM_TRACING
                        printf( "bli_snorm2fv_unb_var1_avx2(): releasing mem pool block\n" );
                    #endif
                    // Return the buffer to pool.
                    bli_membrk_release( &rntm , &mem_bufX );
                }

                AOCL_DTL_TRACE_EXIT( AOCL_DTL_LEVEL_TRACE_3 );
                return;
            }

            // Mask vectors which indicate whether
            // xi<=thres_sml or xi>=thres_big.
            mask_vec0.v = CMP256_sf( x0v.v, thres_sml_vec.v, thres_big_vec.v );
            mask_vec1.v = CMP256_sf( x1v.v, thres_sml_vec.v, thres_big_vec.v );
            mask_vec2.v = CMP256_sf( x2v.v, thres_sml_vec.v, thres_big_vec.v );
            mask_vec3.v = CMP256_sf( x3v.v, thres_sml_vec.v, thres_big_vec.v );

            if ( !bli_horizontal_or_sf( mask_vec0.v ) )
            {
                // Scaling is not necessary; only medium values.
                sum_med_vec0.v = _mm256_fmadd_ps( x0v.v, x0v.v, sum_med_vec0.v );
            }
            else
            {
                // Mask vector which indicate whether xi > thres_big.
                mask_vec0.v = _mm256_cmp_ps( x0v.v, thres_big_vec.v, _CMP_GT_OQ );

                if ( bli_horizontal_or_sf( mask_vec0.v ) )
                {
                    isbig = true;
                    // Fill sum_med vector without scaling.
                    y0v.v = _mm256_blendv_ps( x0v.v, zerov.v, mask_vec0.v );
                    sum_med_vec0.v = _mm256_fmadd_ps( y0v.v, y0v.v, sum_med_vec0.v );

                    // Fill sum_big vector using scaling.
                    temp.v = _mm256_set1_ps( scale_big );
                    y0v.v = _mm256_blendv_ps( zerov.v, temp.v, mask_vec0.v );
                    y0v.v = _mm256_mul_ps( x0v.v, y0v.v );
                    sum_big_vec0.v = _mm256_fmadd_ps( y0v.v, y0v.v, sum_big_vec0.v );
                    temp.v = _mm256_set1_ps( -0.0 );
                }
                else
                {
                    // Mask vector which indicates whether xi > thres_small.
                    mask_vec0.v = _mm256_cmp_ps( x0v.v, thres_sml_vec.v, _CMP_LT_OQ );
                    // Fill sum_med vector without scaling.
                    y0v.v = _mm256_blendv_ps( x0v.v, zerov.v, mask_vec0.v );
                    sum_med_vec0.v = _mm256_fmadd_ps( y0v.v, y0v.v, sum_med_vec0.v );

                    // Accumulate small values only if there have not been any big values so far.
                    if ( !isbig )
                    {
                        // Fill sum_sml vector using scaling.
                        temp.v = _mm256_set1_ps( scale_sml );
                        y0v.v = _mm256_blendv_ps( zerov.v, temp.v, mask_vec0.v );
                        y0v.v = _mm256_mul_ps( x0v.v, y0v.v );
                        sum_sml_vec0.v = _mm256_fmadd_ps( y0v.v, y0v.v, sum_sml_vec0.v );
                        temp.v = _mm256_set1_ps( -0.0 );
                    }
                }
                
            }
            if ( !bli_horizontal_or_sf( mask_vec1.v ) )
            {
                // Scaling is not necessary; only medium values.
                sum_med_vec1.v = _mm256_fmadd_ps( x1v.v, x1v.v, sum_med_vec1.v );
            }
            else
            {
                // Mask vector which indicate whether xi > thres_big.
                mask_vec1.v = _mm256_cmp_ps( x1v.v, thres_big_vec.v, _CMP_GT_OQ );

                if ( bli_horizontal_or_sf( mask_vec1.v ) )
                {
                    isbig = true;
                    // Fill sum_med vector without scaling.
                    y1v.v = _mm256_blendv_ps( x1v.v, zerov.v, mask_vec1.v );
                    sum_med_vec1.v = _mm256_fmadd_ps( y1v.v, y1v.v, sum_med_vec1.v );

                    // Fill sum_big vector using scaling.
                    temp.v = _mm256_set1_ps( scale_big );
                    y1v.v = _mm256_blendv_ps( zerov.v, temp.v, mask_vec1.v );
                    y1v.v = _mm256_mul_ps( x1v.v, y1v.v );
                    sum_big_vec1.v = _mm256_fmadd_ps( y1v.v, y1v.v, sum_big_vec1.v );
                    temp.v = _mm256_set1_ps( -0.0 );
                }
                else
                {
                    // Mask vector which indicates whether xi > thres_small.
                    mask_vec1.v = _mm256_cmp_ps( x1v.v, thres_sml_vec.v, _CMP_LT_OQ );
                    // Fill sum_med vector without scaling.
                    y1v.v = _mm256_blendv_ps( x1v.v, zerov.v, mask_vec1.v );
                    sum_med_vec1.v = _mm256_fmadd_ps( y1v.v, y1v.v, sum_med_vec1.v );

                    // Accumulate small values only if there have not been any big values so far.
                    if ( !isbig )
                    {
                        // Fill sum_sml vector using scaling.
                        temp.v = _mm256_set1_ps( scale_sml );
                        y1v.v = _mm256_blendv_ps( zerov.v, temp.v, mask_vec1.v );
                        y1v.v = _mm256_mul_ps( x1v.v, y1v.v );
                        sum_sml_vec1.v = _mm256_fmadd_ps( y1v.v, y1v.v, sum_sml_vec1.v );
                        temp.v = _mm256_set1_ps( -0.0 );
                    }
                }
            }
            if ( !bli_horizontal_or_sf( mask_vec2.v ) )
            {
                // Scaling is not necessary; only medium values.
                sum_med_vec2.v = _mm256_fmadd_ps( x2v.v, x2v.v, sum_med_vec2.v );
            }
            else
            {
                // Mask vector which indicate whether xi > thres_big.
                mask_vec2.v = _mm256_cmp_ps( x2v.v, thres_big_vec.v, _CMP_GT_OQ );

                if ( bli_horizontal_or_sf( mask_vec2.v ) )
                {
                    isbig = true;
                    // Fill sum_med vector without scaling.
                    y2v.v = _mm256_blendv_ps( x2v.v, zerov.v, mask_vec2.v );
                    sum_med_vec2.v = _mm256_fmadd_ps( y2v.v, y2v.v, sum_med_vec2.v );

                    // Fill sum_big vector using scaling.
                    temp.v = _mm256_set1_ps( scale_big );
                    y2v.v = _mm256_blendv_ps( zerov.v, temp.v, mask_vec2.v );
                    y2v.v = _mm256_mul_ps( x2v.v, y2v.v );
                    sum_big_vec2.v = _mm256_fmadd_ps( y2v.v, y2v.v, sum_big_vec2.v );
                    temp.v = _mm256_set1_ps( -0.0 );
                }
                else
                {                    
                    // Mask vector which indicates whether xi > thres_small.
                    mask_vec2.v = _mm256_cmp_ps( x2v.v, thres_sml_vec.v, _CMP_LT_OQ );
                    // Fill sum_med vector without scaling.
                    y2v.v = _mm256_blendv_ps( x2v.v, zerov.v, mask_vec2.v );
                    sum_med_vec2.v = _mm256_fmadd_ps( y2v.v, y2v.v, sum_med_vec2.v );

                    // Accumulate small values only if there have not been any big values so far.
                    if ( !isbig )
                    {
                        // Fill sum_sml vector using scaling.
                        temp.v = _mm256_set1_ps( scale_sml );
                        y2v.v = _mm256_blendv_ps( zerov.v, temp.v, mask_vec2.v );
                        y2v.v = _mm256_mul_ps( x2v.v, y2v.v );
                        sum_sml_vec2.v = _mm256_fmadd_ps( y2v.v, y2v.v, sum_sml_vec2.v );
                        temp.v = _mm256_set1_ps( -0.0 );
                    }
                }
            }
            if ( !bli_horizontal_or_sf( mask_vec3.v ) )
            {
                // Scaling is not necessary; only medium values.
                sum_med_vec3.v = _mm256_fmadd_ps( x3v.v, x3v.v, sum_med_vec3.v );
            }
            else
            {
                // Mask vector which indicate whether xi > thres_big.
                mask_vec3.v = _mm256_cmp_ps( x3v.v, thres_big_vec.v, _CMP_GT_OQ );

                if ( bli_horizontal_or_sf( mask_vec3.v ) )
                {
                    isbig = true;
                    // Fill sum_med vector without scaling.
                    y3v.v = _mm256_blendv_ps( x3v.v, zerov.v, mask_vec3.v );
                    sum_med_vec3.v = _mm256_fmadd_ps( y3v.v, y3v.v, sum_med_vec3.v );

                    // Fill sum_big vector using scaling.
                    temp.v = _mm256_set1_ps( scale_big );
                    y3v.v = _mm256_blendv_ps( zerov.v, temp.v, mask_vec3.v );
                    y3v.v = _mm256_mul_ps( x3v.v, y3v.v );
                    sum_big_vec3.v = _mm256_fmadd_ps( y3v.v, y3v.v, sum_big_vec3.v );
                    temp.v = _mm256_set1_ps( -0.0 );
                }
                else
                {
                    // Mask vector which indicates whether xi > thres_small.
                    mask_vec3.v = _mm256_cmp_ps( x3v.v, thres_sml_vec.v, _CMP_LT_OQ );
                    // Fill sum_med vector without scaling.
                    y3v.v = _mm256_blendv_ps( x3v.v, zerov.v, mask_vec3.v );
                    sum_med_vec3.v = _mm256_fmadd_ps( y3v.v, y3v.v, sum_med_vec3.v );

                    // Accumulate small values only if there have not been any big values so far.
                    if ( !isbig )
                    {
                        // Fill sum_sml vector using scaling.
                        temp.v = _mm256_set1_ps( scale_sml );
                        y3v.v = _mm256_blendv_ps( zerov.v, temp.v, mask_vec3.v );
                        y3v.v = _mm256_mul_ps( x3v.v, y3v.v );
                        sum_sml_vec3.v = _mm256_fmadd_ps( y3v.v, y3v.v, sum_sml_vec3.v );
                        temp.v = _mm256_set1_ps( -0.0 );
                    }
                }
            }
            xt += 32;
        }

        for (; ( i + 24 ) <= n; i = i + 24)
        {
            x0v.v = _mm256_loadu_ps( xt );
            x1v.v = _mm256_loadu_ps( xt + 8 );
            x2v.v = _mm256_loadu_ps( xt + 16 );

            // Getting the abs of the vector elements.
            x0v.v = _mm256_andnot_ps( temp.v, x0v.v );
            x1v.v = _mm256_andnot_ps( temp.v, x1v.v );
            x2v.v = _mm256_andnot_ps( temp.v, x2v.v );

            // Check if any of the values is a NaN and if so, return.
            mask_vec0.v = _mm256_cmp_ps(x0v.v, x0v.v, _CMP_UNORD_Q);
            mask_vec1.v = _mm256_cmp_ps(x1v.v, x1v.v, _CMP_UNORD_Q);
            mask_vec2.v = _mm256_cmp_ps(x2v.v, x2v.v, _CMP_UNORD_Q);
            if ( bli_horizontal_or_sf( mask_vec0.v ) )
            {
                *norm = NAN;
                if ( ( incx != 1 ) && bli_mem_is_alloc( &mem_bufX ) )
                {
                    #ifdef BLIS_ENABLE_MEM_TRACING
                        printf( "bli_snorm2fv_unb_var1_avx2(): releasing mem pool block\n" );
                    #endif
                    // Return the buffer to pool.
                    bli_membrk_release( &rntm , &mem_bufX );
                }

                AOCL_DTL_TRACE_EXIT( AOCL_DTL_LEVEL_TRACE_3 );
                return;
            }
            if ( bli_horizontal_or_sf( mask_vec1.v ) )
            {
                *norm = NAN;
                if ( ( incx != 1 ) && bli_mem_is_alloc( &mem_bufX ) )
                {
                    #ifdef BLIS_ENABLE_MEM_TRACING
                        printf( "bli_snorm2fv_unb_var1_avx2(): releasing mem pool block\n" );
                    #endif
                    // Return the buffer to pool.
                    bli_membrk_release( &rntm , &mem_bufX );
                }

                AOCL_DTL_TRACE_EXIT( AOCL_DTL_LEVEL_TRACE_3 );
                return;
            }
            if ( bli_horizontal_or_sf( mask_vec2.v ) )
            {
                *norm = NAN;
                if ( ( incx != 1 ) && bli_mem_is_alloc( &mem_bufX ) )
                {
                    #ifdef BLIS_ENABLE_MEM_TRACING
                        printf( "bli_snorm2fv_unb_var1_avx2(): releasing mem pool block\n" );
                    #endif
                    // Return the buffer to pool.
                    bli_membrk_release( &rntm , &mem_bufX );
                }

                AOCL_DTL_TRACE_EXIT( AOCL_DTL_LEVEL_TRACE_3 );
                return;
            }

            // Mask vectors which indicate whether
            // xi<=thres_sml or xi>=thres_big.
            mask_vec0.v = CMP256_sf( x0v.v, thres_sml_vec.v, thres_big_vec.v );
            mask_vec1.v = CMP256_sf( x1v.v, thres_sml_vec.v, thres_big_vec.v );
            mask_vec2.v = CMP256_sf( x2v.v, thres_sml_vec.v, thres_big_vec.v );

            if ( !bli_horizontal_or_sf( mask_vec0.v ) )
            {
                // Scaling is not necessary; only medium values.
                sum_med_vec0.v = _mm256_fmadd_ps( x0v.v, x0v.v, sum_med_vec0.v );
            }
            else
            {
                // Mask vector which indicate whether xi > thres_big.
                mask_vec0.v = _mm256_cmp_ps( x0v.v, thres_big_vec.v, _CMP_GT_OQ );

                if ( bli_horizontal_or_sf( mask_vec0.v ) )
                {
                    isbig = true;

                    // Fill sum_med vector without scaling.
                    y0v.v = _mm256_blendv_ps( x0v.v, zerov.v, mask_vec0.v );
                    sum_med_vec0.v = _mm256_fmadd_ps( y0v.v, y0v.v, sum_med_vec0.v );

                    // Fill sum_big vector using scaling.
                    temp.v = _mm256_set1_ps( scale_big );
                    y0v.v = _mm256_blendv_ps( zerov.v, temp.v, mask_vec0.v );
                    y0v.v = _mm256_mul_ps( x0v.v, y0v.v );
                    sum_big_vec0.v = _mm256_fmadd_ps( y0v.v, y0v.v, sum_big_vec0.v );
                    temp.v = _mm256_set1_ps( -0.0 );
                }
                else
                {
                    // Mask vector which indicates whether xi > thres_small.
                    mask_vec0.v = _mm256_cmp_ps( x0v.v, thres_sml_vec.v, _CMP_LT_OQ );
                    // Fill sum_med vector without scaling.
                    y0v.v = _mm256_blendv_ps( x0v.v, zerov.v, mask_vec0.v );
                    sum_med_vec0.v = _mm256_fmadd_ps( y0v.v, y0v.v, sum_med_vec0.v );

                    // Accumulate small values only if there have not been any big values so far.
                    if ( !isbig )
                    {
                        // Fill sum_sml vector using scaling.
                        temp.v = _mm256_set1_ps( scale_sml );
                        y0v.v = _mm256_blendv_ps( zerov.v, temp.v, mask_vec0.v );
                        y0v.v = _mm256_mul_ps( x0v.v, y0v.v );
                        sum_sml_vec0.v = _mm256_fmadd_ps( y0v.v, y0v.v, sum_sml_vec0.v );
                        temp.v = _mm256_set1_ps( -0.0 );
                    }
                }
            }
            if ( !bli_horizontal_or_sf( mask_vec1.v ) )
            {
                // Scaling is not necessary; only medium values.
                sum_med_vec1.v = _mm256_fmadd_ps( x1v.v, x1v.v, sum_med_vec1.v );
            }
            else
            {
                // Mask vector which indicate whether xi > thres_big.
                mask_vec1.v = _mm256_cmp_ps( x1v.v, thres_big_vec.v, _CMP_GT_OQ );

                if ( bli_horizontal_or_sf( mask_vec1.v ) )
                {
                    isbig = true;

                    // Fill sum_med vector without scaling.
                    y1v.v = _mm256_blendv_ps( x1v.v, zerov.v, mask_vec1.v );
                    sum_med_vec1.v = _mm256_fmadd_ps( y1v.v, y1v.v, sum_med_vec1.v );

                    // Fill sum_big vector using scaling.
                    temp.v = _mm256_set1_ps( scale_big );
                    y1v.v = _mm256_blendv_ps( zerov.v, temp.v, mask_vec1.v );
                    y1v.v = _mm256_mul_ps( x1v.v, y1v.v );
                    sum_big_vec1.v = _mm256_fmadd_ps( y1v.v, y1v.v, sum_big_vec1.v );
                    temp.v = _mm256_set1_ps( -0.0 );
                }
                else
                {
                    // Mask vector which indicates whether xi > thres_small.
                    mask_vec1.v = _mm256_cmp_ps( x1v.v, thres_sml_vec.v, _CMP_LT_OQ );
                    // Fill sum_med vector without scaling.
                    y1v.v = _mm256_blendv_ps( x1v.v, zerov.v, mask_vec1.v );
                    sum_med_vec1.v = _mm256_fmadd_ps( y1v.v, y1v.v, sum_med_vec1.v );

                    // Accumulate small values only if there have not been any big values so far.
                    if ( !isbig )
                    {
                        // Fill sum_sml vector using scaling.
                        temp.v = _mm256_set1_ps( scale_sml );
                        y1v.v = _mm256_blendv_ps( zerov.v, temp.v, mask_vec1.v );
                        y1v.v = _mm256_mul_ps( x1v.v, y1v.v );
                        sum_sml_vec1.v = _mm256_fmadd_ps( y1v.v, y1v.v, sum_sml_vec1.v );
                        temp.v = _mm256_set1_ps( -0.0 );
                    }
                }
            }
            if ( !bli_horizontal_or_sf( mask_vec2.v ) )
            {
                // Scaling is not necessary; only medium values.
                sum_med_vec2.v = _mm256_fmadd_ps( x2v.v, x2v.v, sum_med_vec2.v );
            }
            else
            {
                // Mask vector which indicate whether xi > thres_big.
                mask_vec2.v = _mm256_cmp_ps( x2v.v, thres_big_vec.v, _CMP_GT_OQ );

                if ( bli_horizontal_or_sf( mask_vec2.v ) )
                {
                    isbig = true;

                    // Fill sum_med vector without scaling.
                    y2v.v = _mm256_blendv_ps( x2v.v, zerov.v, mask_vec2.v );
                    sum_med_vec2.v = _mm256_fmadd_ps( y2v.v, y2v.v, sum_med_vec2.v );

                    // Fill sum_big vector using scaling.
                    temp.v = _mm256_set1_ps( scale_big );
                    y2v.v = _mm256_blendv_ps( zerov.v, temp.v, mask_vec2.v );
                    y2v.v = _mm256_mul_ps( x2v.v, y2v.v );
                    sum_big_vec2.v = _mm256_fmadd_ps( y2v.v, y2v.v, sum_big_vec2.v );
                    temp.v = _mm256_set1_ps( -0.0 );
                }
                else
                {
                    // Mask vector which indicates whether xi > thres_small.
                    mask_vec2.v = _mm256_cmp_ps( x2v.v, thres_sml_vec.v, _CMP_LT_OQ );
                    // Fill sum_med vector without scaling.
                    y2v.v = _mm256_blendv_ps( x2v.v, zerov.v, mask_vec2.v );
                    sum_med_vec2.v = _mm256_fmadd_ps( y2v.v, y2v.v, sum_med_vec2.v );

                    // Accumulate small values only if there have not been any big values so far.
                    if ( !isbig )
                    {
                        // Fill sum_sml vector using scaling.
                        temp.v = _mm256_set1_ps( scale_sml );
                        y2v.v = _mm256_blendv_ps( zerov.v, temp.v, mask_vec2.v );
                        y2v.v = _mm256_mul_ps( x2v.v, y2v.v );
                        sum_sml_vec2.v = _mm256_fmadd_ps( y2v.v, y2v.v, sum_sml_vec2.v );
                        temp.v = _mm256_set1_ps( -0.0 );
                    }
                }
            }
            
            xt += 24;
        }

        for (; ( i + 16 ) <= n; i = i + 16)
        {
            x0v.v = _mm256_loadu_ps( xt );
            x1v.v = _mm256_loadu_ps( xt + 8 );

            // Getting the abs of the vector elements.
            x0v.v = _mm256_andnot_ps( temp.v, x0v.v );
            x1v.v = _mm256_andnot_ps( temp.v, x1v.v );

            // Check if any of the values is a NaN and if so, return.
            mask_vec0.v = _mm256_cmp_ps(x0v.v, x0v.v, _CMP_UNORD_Q);
            mask_vec1.v = _mm256_cmp_ps(x1v.v, x1v.v, _CMP_UNORD_Q);
            if ( bli_horizontal_or_sf( mask_vec0.v ) )
            {
                *norm = NAN;
                if ( ( incx != 1 ) && bli_mem_is_alloc( &mem_bufX ) )
                {
                    #ifdef BLIS_ENABLE_MEM_TRACING
                        printf( "bli_snorm2fv_unb_var1_avx2(): releasing mem pool block\n" );
                    #endif
                    // Return the buffer to pool.
                    bli_membrk_release( &rntm , &mem_bufX );
                }

                AOCL_DTL_TRACE_EXIT( AOCL_DTL_LEVEL_TRACE_3 );
                return;
            }
            if ( bli_horizontal_or_sf( mask_vec1.v ) )
            {
                *norm = NAN;
                if ( ( incx != 1 ) && bli_mem_is_alloc( &mem_bufX ) )
                {
                    #ifdef BLIS_ENABLE_MEM_TRACING
                        printf( "bli_snorm2fv_unb_var1_avx2(): releasing mem pool block\n" );
                    #endif
                    // Return the buffer to pool.
                    bli_membrk_release( &rntm , &mem_bufX );
                }

                AOCL_DTL_TRACE_EXIT( AOCL_DTL_LEVEL_TRACE_3 );
                return;
            }
            // Mask vectors which indicate whether
            // xi<=thres_sml or xi>=thres_big.
            mask_vec0.v = CMP256_sf( x0v.v, thres_sml_vec.v, thres_big_vec.v );
            mask_vec1.v = CMP256_sf( x1v.v, thres_sml_vec.v, thres_big_vec.v );

            if ( !bli_horizontal_or_sf( mask_vec0.v ) )
            {
                // Scaling is not necessary; only medium values.
                sum_med_vec0.v = _mm256_fmadd_ps( x0v.v, x0v.v, sum_med_vec0.v );
            }
            else
            {
                // Mask vector which indicate whether xi > thres_big.
                mask_vec0.v = _mm256_cmp_ps( x0v.v, thres_big_vec.v, _CMP_GT_OQ );

                if ( bli_horizontal_or_sf( mask_vec0.v ) )
                {
                    isbig = true;

                    // Fill sum_med vector without scaling.
                    y0v.v = _mm256_blendv_ps( x0v.v, zerov.v, mask_vec0.v );
                    sum_med_vec0.v = _mm256_fmadd_ps( y0v.v, y0v.v, sum_med_vec0.v );

                    // Fill sum_big vector using scaling.
                    temp.v = _mm256_set1_ps( scale_big );
                    y0v.v = _mm256_blendv_ps( zerov.v, temp.v, mask_vec0.v );
                    y0v.v = _mm256_mul_ps( x0v.v, y0v.v );
                    sum_big_vec0.v = _mm256_fmadd_ps( y0v.v, y0v.v, sum_big_vec0.v );
                    temp.v = _mm256_set1_ps( -0.0 );
                }
                else
                {
                    // Mask vector which indicates whether xi > thres_small.
                    mask_vec0.v = _mm256_cmp_ps( x0v.v, thres_sml_vec.v, _CMP_LT_OQ );
                    // Fill sum_med vector without scaling.
                    y0v.v = _mm256_blendv_ps( x0v.v, zerov.v, mask_vec0.v );
                    sum_med_vec0.v = _mm256_fmadd_ps( y0v.v, y0v.v, sum_med_vec0.v );

                    // Accumulate small values only if there have not been any big values so far.
                    if ( !isbig )
                    {
                        // Fill sum_sml vector using scaling.
                        temp.v = _mm256_set1_ps( scale_sml );
                        y0v.v = _mm256_blendv_ps( zerov.v, temp.v, mask_vec0.v );
                        y0v.v = _mm256_mul_ps( x0v.v, y0v.v );
                        sum_sml_vec0.v = _mm256_fmadd_ps( y0v.v, y0v.v, sum_sml_vec0.v );
                        temp.v = _mm256_set1_ps( -0.0 );
                    }
                }
            }
            if ( !bli_horizontal_or_sf( mask_vec1.v ) )
            {
                // Scaling is not necessary; only medium values.
                sum_med_vec1.v = _mm256_fmadd_ps( x1v.v, x1v.v, sum_med_vec1.v );
            }
            else
            {
                // Mask vector which indicate whether xi > thres_big.
                mask_vec1.v = _mm256_cmp_ps( x1v.v, thres_big_vec.v, _CMP_GT_OQ );

                if ( bli_horizontal_or_sf( mask_vec1.v ) )
                {
                    isbig = true;

                    // Fill sum_med vector without scaling.
                    y1v.v = _mm256_blendv_ps( x1v.v, zerov.v, mask_vec1.v );
                    sum_med_vec1.v = _mm256_fmadd_ps( y1v.v, y1v.v, sum_med_vec1.v );

                    // Fill sum_big vector using scaling.
                    temp.v = _mm256_set1_ps( scale_big );
                    y1v.v = _mm256_blendv_ps( zerov.v, temp.v, mask_vec1.v );
                    y1v.v = _mm256_mul_ps( x1v.v, y1v.v );
                    sum_big_vec1.v = _mm256_fmadd_ps( y1v.v, y1v.v, sum_big_vec1.v );
                    temp.v = _mm256_set1_ps( -0.0 );
                }
                else
                {
                    // Mask vector which indicates whether xi > thres_small.
                    mask_vec1.v = _mm256_cmp_ps( x1v.v, thres_sml_vec.v, _CMP_LT_OQ );
                    // Fill sum_med vector without scaling.
                    y1v.v = _mm256_blendv_ps( x1v.v, zerov.v, mask_vec1.v );
                    sum_med_vec1.v = _mm256_fmadd_ps( y1v.v, y1v.v, sum_med_vec1.v );

                    // Accumulate small values only if there have not been any big values so far.
                    if ( !isbig )
                    {
                        // Fill sum_sml vector using scaling.
                        temp.v = _mm256_set1_ps( scale_sml );
                        y1v.v = _mm256_blendv_ps( zerov.v, temp.v, mask_vec1.v );
                        y1v.v = _mm256_mul_ps( x1v.v, y1v.v );
                        sum_sml_vec1.v = _mm256_fmadd_ps( y1v.v, y1v.v, sum_sml_vec1.v );
                        temp.v = _mm256_set1_ps( -0.0 );
                    }
                }
            }
            xt += 16;
        }
        
        // This seems to not be improving performance.
        #if 0
        for (; ( i + 8 ) <= n; i = i + 8)
        {
            x0v.v = _mm256_loadu_ps( xt );

            // Getting the abs of the vector elements.
            x0v.v = _mm256_andnot_ps( temp.v, x0v.v );

            // Check if any of the values is a NaN and if so, return.
            mask_vec0.v = _mm256_cmp_ps(x0v.v, x0v.v, _CMP_UNORD_Q);
            if ( bli_horizontal_or_sf( mask_vec0.v ) )
            {
                *norm = NAN;
                if ( ( incx != 1 ) && bli_mem_is_alloc( &mem_bufX ) )
                {
                    #ifdef BLIS_ENABLE_MEM_TRACING
                        printf( "bli_snorm2fv_unb_var1_avx2(): releasing mem pool block\n" );
                    #endif
                    // Return the buffer to pool.
                    bli_membrk_release( &rntm , &mem_bufX );
                }

                AOCL_DTL_TRACE_EXIT( AOCL_DTL_LEVEL_TRACE_3 );
                return;
            }
            // Mask vectors which indicate whether
            // xi<=thres_sml or xi>=thres_big.
            mask_vec0.v = CMP256_sf( x0v.v, thres_sml_vec.v, thres_big_vec.v );

            if ( !bli_horizontal_or_sf( mask_vec0.v ) )
            {
                // Scaling is not necessary; only medium values.
                sum_med_vec0.v = _mm256_fmadd_ps( x0v.v, x0v.v, sum_med_vec0.v );
            }
            else
            {
                // Mask vector which indicate whether xi > thres_big.
                mask_vec0.v = _mm256_cmp_ps( x0v.v, thres_big_vec.v, _CMP_GT_OQ );

                if ( bli_horizontal_or_sf( mask_vec0.v ) )
                {
                    isbig = true;

                    // Fill sum_med vector without scaling.
                    y0v.v = _mm256_blendv_ps( x0v.v, zerov.v, mask_vec0.v );
                    sum_med_vec0.v = _mm256_fmadd_ps( y0v.v, y0v.v, sum_med_vec0.v );

                    // Fill sum_big vector using scaling.
                    temp.v = _mm256_set1_ps( scale_big );
                    y0v.v = _mm256_blendv_ps( zerov.v, temp.v, mask_vec0.v );
                    y0v.v = _mm256_mul_ps( x0v.v, y0v.v );
                    sum_big_vec0.v = _mm256_fmadd_ps( y0v.v, y0v.v, sum_big_vec0.v );
                    temp.v = _mm256_set1_ps( -0.0 );
                }
                else
                {
                    // Mask vector which indicates whether xi > thres_small.
                    mask_vec0.v = _mm256_cmp_ps( x0v.v, thres_sml_vec.v, _CMP_LT_OQ );
                    // Fill sum_med vector without scaling.
                    y0v.v = _mm256_blendv_ps( x0v.v, zerov.v, mask_vec0.v );
                    sum_med_vec0.v = _mm256_fmadd_ps( y0v.v, y0v.v, sum_med_vec0.v );

                    // Accumulate small values only if there have not been any big values so far.
                    if ( !isbig )
                    {
                        // Fill sum_sml vector using scaling.
                        temp.v = _mm256_set1_ps( scale_sml );
                        y0v.v = _mm256_blendv_ps( zerov.v, temp.v, mask_vec0.v );
                        y0v.v = _mm256_mul_ps( x0v.v, y0v.v );
                        sum_sml_vec0.v = _mm256_fmadd_ps( y0v.v, y0v.v, sum_sml_vec0.v );
                        temp.v = _mm256_set1_ps( -0.0 );
                    }
                }
            }
            xt += 8;
        }
        #endif

        sum_sml_vec0.v = _mm256_add_ps( sum_sml_vec0.v, sum_sml_vec1.v );
        sum_sml_vec2.v = _mm256_add_ps( sum_sml_vec2.v, sum_sml_vec3.v );
        sum_sml_vec0.v = _mm256_add_ps( sum_sml_vec0.v, sum_sml_vec2.v ); 
        sum_sml = horizontal_add_sf(sum_sml_vec0.v);

        sum_med_vec0.v = _mm256_add_ps( sum_med_vec0.v, sum_med_vec1.v );
        sum_med_vec2.v = _mm256_add_ps( sum_med_vec2.v, sum_med_vec3.v );
        sum_med_vec0.v = _mm256_add_ps( sum_med_vec0.v, sum_med_vec2.v );
        sum_med = horizontal_add_sf(sum_med_vec0.v);

        sum_big_vec0.v = _mm256_add_ps( sum_big_vec0.v, sum_big_vec1.v );
        sum_big_vec2.v = _mm256_add_ps( sum_big_vec2.v, sum_big_vec3.v );
        sum_big_vec0.v = _mm256_add_ps( sum_big_vec0.v, sum_big_vec2.v );
        sum_big = horizontal_add_sf(sum_big_vec0.v);
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
                if ( ( incx != 1 ) && bli_mem_is_alloc( &mem_bufX ) )
                {
                    #ifdef BLIS_ENABLE_MEM_TRACING
                        printf( "bli_snorm2fv_unb_var1_avx2(): releasing mem pool block\n" );
                    #endif
                    // Return the buffer to pool.
                    bli_membrk_release( &rntm , &mem_bufX );
                }

                AOCL_DTL_TRACE_EXIT( AOCL_DTL_LEVEL_TRACE_3 );
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
    if ( hasInf ) 
    {
        
        if ( ( incx != 1 ) && bli_mem_is_alloc( &mem_bufX ) )
        {
            #ifdef BLIS_ENABLE_MEM_TRACING
                printf( "bli_snorm2fv_unb_var1_avx2(): releasing mem pool block\n" );
            #endif
            // Return the buffer to pool.
            bli_membrk_release( &rntm , &mem_bufX );
        }

        AOCL_DTL_TRACE_EXIT( AOCL_DTL_LEVEL_TRACE_3 );
        return;
    }
    
    // Combine accumulators.
    if ( isbig )
    {
        // Combine sum_big and sum_med if sum_med > 0.
        if ( sum_med > 0.0f )
        {
            sum_big += ( sum_med * scale_big ) * scale_big;
        }
        scale = 1.0f / scale_big;
        sumsq = sum_big;
    }
    else if ( sum_sml > 0.0f )
    {
        // Combine sum_med and sum_sml if sum_sml>0.
        if ( sum_med > 0.0f )
        {
            sum_med = sqrtf( sum_med );
            sum_sml = sqrtf( sum_sml ) / scale_sml;
            float ymin, ymax;
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
            scale = 1.0f;
            sumsq = ymax * ymax * ( 1.0f + ( ymin / ymax ) * ( ymin / ymax ) );
        }
        else
        {
            scale = 1.0f / scale_sml;
            sumsq = sum_sml;
        }
    }
    else
    {
        // If all values are mid-range:
        scale = 1.0f;
        sumsq = sum_med;
    }

    *norm = scale * sqrtf( sumsq );

    if ( ( incx != 1 ) && bli_mem_is_alloc( &mem_bufX ) )
    {
        #ifdef BLIS_ENABLE_MEM_TRACING
            printf( "bli_snorm2fv_unb_var1_avx2(): releasing mem pool block\n" );
        #endif
        // Return the buffer to pool.
        bli_membrk_release( &rntm , &mem_bufX );
    }

    AOCL_DTL_TRACE_EXIT( AOCL_DTL_LEVEL_TRACE_3 );

    return;
}

// Optimized function that computes the Frobenius norm using AVX2 intrinsics.
void bli_scnorm2fv_unb_var1_avx2
    (
       dim_t    n,
       scomplex*   x, inc_t incx,
       float* norm,
       cntx_t*  cntx
    )
{
    AOCL_DTL_TRACE_ENTRY( AOCL_DTL_LEVEL_TRACE_3 );

    float sumsq = 0.0f;
    dim_t i = 0;
    dim_t n_remainder = 0;
    scomplex  *x_buf = x;

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

        // Calculate the size required for "n" scomplex elements in vector x.
        size_t buffer_size = n * sizeof( scomplex );

        #ifdef BLIS_ENABLE_MEM_TRACING
            printf( "bli_scnorm2fv_unb_var1_avx2(): get mem pool block\n" );
        #endif

        // Acquire a Buffer(n*size(scomplex)) from the memory broker
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
                *( x_buf + x_index ) = *( x + ( x_index * incx ) );
            }
        }
    }

    scomplex *xt = x_buf;

    // Compute the sum of squares on 3 accumulators to avoid overflow
    // and underflow, depending on the vector element value.
    // Accumulator for small values; using scaling to avoid underflow.
    float sum_sml = 0.0f;
   // Accumulator for medium values; no scaling required.
    float sum_med = 0.0f;
    // Accumulator for big values; using scaling to avoid overflow.
    float sum_big = 0.0f;

    // Constants chosen to minimize roundoff, according to Blue's algorithm.
    const float thres_sml = powf( ( float )FLT_RADIX,    ceilf( ( FLT_MIN_EXP - 1 )  * 0.5f ) );
    const float thres_big = powf( ( float )FLT_RADIX,   floorf( ( FLT_MAX_EXP - 23)  * 0.5f ) );
    const float scale_sml = powf( ( float )FLT_RADIX, - floorf( ( FLT_MIN_EXP - 24 ) * 0.5f ) );
    const float scale_big = powf( ( float )FLT_RADIX,  - ceilf( ( FLT_MAX_EXP + 23 ) * 0.5f ) );

    float scale = 1.0f;
    float abs_chi;
    bool isbig = false;

    if ( n >= 64 )
    {
        // Constants used for comparisons.
        v8sf_t temp, thres_sml_vec, thres_big_vec, zerov;
        temp.v = _mm256_set1_ps( -0.0f );
        thres_sml_vec.v = _mm256_set1_ps( thres_sml );
        thres_big_vec.v = _mm256_set1_ps( thres_big );
        v8sf_t x0v, x1v, x2v, x3v;
        v8sf_t y0v, y1v, y2v, y3v;
        v8sf_t mask_vec0, mask_vec1, mask_vec2, mask_vec3;
        zerov.v  = _mm256_setzero_ps();

        // Partial sums used for scaling.
        v8sf_t sum_sml_vec0, sum_sml_vec1, sum_sml_vec2, sum_sml_vec3;
        sum_sml_vec0.v = _mm256_setzero_ps();
        sum_sml_vec1.v = _mm256_setzero_ps();
        sum_sml_vec2.v = _mm256_setzero_ps();
        sum_sml_vec3.v = _mm256_setzero_ps();

        v8sf_t sum_med_vec0, sum_med_vec1, sum_med_vec2, sum_med_vec3;
        sum_med_vec0.v = _mm256_setzero_ps();
        sum_med_vec1.v = _mm256_setzero_ps();
        sum_med_vec2.v = _mm256_setzero_ps();
        sum_med_vec3.v = _mm256_setzero_ps();

        v8sf_t sum_big_vec0, sum_big_vec1, sum_big_vec2, sum_big_vec3;
        sum_big_vec0.v = _mm256_setzero_ps();
        sum_big_vec1.v = _mm256_setzero_ps();
        sum_big_vec2.v = _mm256_setzero_ps();
        sum_big_vec3.v = _mm256_setzero_ps();

        for (; ( i + 16 ) <= n; i = i + 16)
        {
            x0v.v = _mm256_loadu_ps( (float*) xt );
            x1v.v = _mm256_loadu_ps( (float*) (xt + 4) );
            x2v.v = _mm256_loadu_ps( (float*) (xt + 8) );
            x3v.v = _mm256_loadu_ps( (float*) (xt + 12) );

            // Getting the abs of the vector elements.
            x0v.v = _mm256_andnot_ps( temp.v, x0v.v );
            x1v.v = _mm256_andnot_ps( temp.v, x1v.v );
            x2v.v = _mm256_andnot_ps( temp.v, x2v.v );
            x3v.v = _mm256_andnot_ps( temp.v, x3v.v );

            // Check if any of the values is a NaN and if so, return.
            mask_vec0.v = _mm256_cmp_ps(x0v.v, x0v.v, _CMP_UNORD_Q);
            mask_vec1.v = _mm256_cmp_ps(x1v.v, x1v.v, _CMP_UNORD_Q);
            mask_vec2.v = _mm256_cmp_ps(x2v.v, x2v.v, _CMP_UNORD_Q);
            mask_vec3.v = _mm256_cmp_ps(x3v.v, x3v.v, _CMP_UNORD_Q);
            if ( bli_horizontal_or_sf( mask_vec0.v ) )
            {
                *norm = NAN;
                if ( ( incx != 1 ) && bli_mem_is_alloc( &mem_bufX ) )
                {
                    #ifdef BLIS_ENABLE_MEM_TRACING
                        printf( "bli_scnorm2fv_unb_var1_avx2(): releasing mem pool block\n" );
                    #endif
                    // Return the buffer to pool.
                    bli_membrk_release( &rntm , &mem_bufX );
                }

                AOCL_DTL_TRACE_EXIT( AOCL_DTL_LEVEL_TRACE_3 );
                return;
            }
            if ( bli_horizontal_or_sf( mask_vec1.v ) )
            {
                *norm = NAN;
                if ( ( incx != 1 ) && bli_mem_is_alloc( &mem_bufX ) )
                {
                    #ifdef BLIS_ENABLE_MEM_TRACING
                        printf( "bli_scnorm2fv_unb_var1_avx2(): releasing mem pool block\n" );
                    #endif
                    // Return the buffer to pool.
                    bli_membrk_release( &rntm , &mem_bufX );
                }

                AOCL_DTL_TRACE_EXIT( AOCL_DTL_LEVEL_TRACE_3 );
                return;
            }
            if ( bli_horizontal_or_sf( mask_vec2.v ) )
            {
                *norm = NAN;
                if ( ( incx != 1 ) && bli_mem_is_alloc( &mem_bufX ) )
                {
                    #ifdef BLIS_ENABLE_MEM_TRACING
                        printf( "bli_scnorm2fv_unb_var1_avx2(): releasing mem pool block\n" );
                    #endif
                    // Return the buffer to pool.
                    bli_membrk_release( &rntm , &mem_bufX );
                }

                AOCL_DTL_TRACE_EXIT( AOCL_DTL_LEVEL_TRACE_3 );
                return;
            }
            if ( bli_horizontal_or_sf( mask_vec3.v ) )
            {
                *norm = NAN;
                if ( ( incx != 1 ) && bli_mem_is_alloc( &mem_bufX ) )
                {
                    #ifdef BLIS_ENABLE_MEM_TRACING
                        printf( "bli_scnorm2fv_unb_var1_avx2(): releasing mem pool block\n" );
                    #endif
                    // Return the buffer to pool.
                    bli_membrk_release( &rntm , &mem_bufX );
                }

                AOCL_DTL_TRACE_EXIT( AOCL_DTL_LEVEL_TRACE_3 );
                return;
            }

            // Mask vectors which indicate whether
            // xi<=thres_sml or xi>=thres_big.
            mask_vec0.v = CMP256_sf( x0v.v, thres_sml_vec.v, thres_big_vec.v );
            mask_vec1.v = CMP256_sf( x1v.v, thres_sml_vec.v, thres_big_vec.v );
            mask_vec2.v = CMP256_sf( x2v.v, thres_sml_vec.v, thres_big_vec.v );
            mask_vec3.v = CMP256_sf( x3v.v, thres_sml_vec.v, thres_big_vec.v );

            if ( !bli_horizontal_or_sf( mask_vec0.v ) )
            {
                // Scaling is not necessary; only medium values.
                sum_med_vec0.v = _mm256_fmadd_ps( x0v.v, x0v.v, sum_med_vec0.v );
            }
            else
            {
                // Mask vector which indicate whether xi > thres_big.
                mask_vec0.v = _mm256_cmp_ps( x0v.v, thres_big_vec.v, _CMP_GT_OQ );

                if ( bli_horizontal_or_sf( mask_vec0.v ) )
                {
                    isbig = true;
                    // Fill sum_med vector without scaling.
                    y0v.v = _mm256_blendv_ps( x0v.v, zerov.v, mask_vec0.v );
                    sum_med_vec0.v = _mm256_fmadd_ps( y0v.v, y0v.v, sum_med_vec0.v );

                    // Fill sum_big vector using scaling.
                    temp.v = _mm256_set1_ps( scale_big );
                    y0v.v = _mm256_blendv_ps( zerov.v, temp.v, mask_vec0.v );
                    y0v.v = _mm256_mul_ps( x0v.v, y0v.v );
                    sum_big_vec0.v = _mm256_fmadd_ps( y0v.v, y0v.v, sum_big_vec0.v );
                    temp.v = _mm256_set1_ps( -0.0 );
                }
                else
                {
                    // Mask vector which indicates whether xi > thres_small.
                    mask_vec0.v = _mm256_cmp_ps( x0v.v, thres_sml_vec.v, _CMP_LT_OQ );
                    // Fill sum_med vector without scaling.
                    y0v.v = _mm256_blendv_ps( x0v.v, zerov.v, mask_vec0.v );
                    sum_med_vec0.v = _mm256_fmadd_ps( y0v.v, y0v.v, sum_med_vec0.v );

                    // Accumulate small values only if there have not been any big values so far.
                    if ( !isbig )
                    {
                        // Fill sum_sml vector using scaling.
                        temp.v = _mm256_set1_ps( scale_sml );
                        y0v.v = _mm256_blendv_ps( zerov.v, temp.v, mask_vec0.v );
                        y0v.v = _mm256_mul_ps( x0v.v, y0v.v );
                        sum_sml_vec0.v = _mm256_fmadd_ps( y0v.v, y0v.v, sum_sml_vec0.v );
                        temp.v = _mm256_set1_ps( -0.0 );
                    }
                }
                
            }
            if ( !bli_horizontal_or_sf( mask_vec1.v ) )
            {
                // Scaling is not necessary; only medium values.
                sum_med_vec1.v = _mm256_fmadd_ps( x1v.v, x1v.v, sum_med_vec1.v );
            }
            else
            {
                // Mask vector which indicate whether xi > thres_big.
                mask_vec1.v = _mm256_cmp_ps( x1v.v, thres_big_vec.v, _CMP_GT_OQ );

                if ( bli_horizontal_or_sf( mask_vec1.v ) )
                {
                    isbig = true;
                    // Fill sum_med vector without scaling.
                    y1v.v = _mm256_blendv_ps( x1v.v, zerov.v, mask_vec1.v );
                    sum_med_vec1.v = _mm256_fmadd_ps( y1v.v, y1v.v, sum_med_vec1.v );

                    // Fill sum_big vector using scaling.
                    temp.v = _mm256_set1_ps( scale_big );
                    y1v.v = _mm256_blendv_ps( zerov.v, temp.v, mask_vec1.v );
                    y1v.v = _mm256_mul_ps( x1v.v, y1v.v );
                    sum_big_vec1.v = _mm256_fmadd_ps( y1v.v, y1v.v, sum_big_vec1.v );
                    temp.v = _mm256_set1_ps( -0.0 );
                }
                else
                {
                    // Mask vector which indicates whether xi > thres_small.
                    mask_vec1.v = _mm256_cmp_ps( x1v.v, thres_sml_vec.v, _CMP_LT_OQ );
                    // Fill sum_med vector without scaling.
                    y1v.v = _mm256_blendv_ps( x1v.v, zerov.v, mask_vec1.v );
                    sum_med_vec1.v = _mm256_fmadd_ps( y1v.v, y1v.v, sum_med_vec1.v );

                    // Accumulate small values only if there have not been any big values so far.
                    if ( !isbig )
                    {
                        // Fill sum_sml vector using scaling.
                        temp.v = _mm256_set1_ps( scale_sml );
                        y1v.v = _mm256_blendv_ps( zerov.v, temp.v, mask_vec1.v );
                        y1v.v = _mm256_mul_ps( x1v.v, y1v.v );
                        sum_sml_vec1.v = _mm256_fmadd_ps( y1v.v, y1v.v, sum_sml_vec1.v );
                        temp.v = _mm256_set1_ps( -0.0 );
                    }
                }
            }
            if ( !bli_horizontal_or_sf( mask_vec2.v ) )
            {
                // Scaling is not necessary; only medium values.
                sum_med_vec2.v = _mm256_fmadd_ps( x2v.v, x2v.v, sum_med_vec2.v );
            }
            else
            {
                // Mask vector which indicate whether xi > thres_big.
                mask_vec2.v = _mm256_cmp_ps( x2v.v, thres_big_vec.v, _CMP_GT_OQ );

                if ( bli_horizontal_or_sf( mask_vec2.v ) )
                {
                    isbig = true;
                    // Fill sum_med vector without scaling.
                    y2v.v = _mm256_blendv_ps( x2v.v, zerov.v, mask_vec2.v );
                    sum_med_vec2.v = _mm256_fmadd_ps( y2v.v, y2v.v, sum_med_vec2.v );

                    // Fill sum_big vector using scaling.
                    temp.v = _mm256_set1_ps( scale_big );
                    y2v.v = _mm256_blendv_ps( zerov.v, temp.v, mask_vec2.v );
                    y2v.v = _mm256_mul_ps( x2v.v, y2v.v );
                    sum_big_vec2.v = _mm256_fmadd_ps( y2v.v, y2v.v, sum_big_vec2.v );
                    temp.v = _mm256_set1_ps( -0.0 );
                }
                else
                {                    
                    // Mask vector which indicates whether xi > thres_small.
                    mask_vec2.v = _mm256_cmp_ps( x2v.v, thres_sml_vec.v, _CMP_LT_OQ );
                    // Fill sum_med vector without scaling.
                    y2v.v = _mm256_blendv_ps( x2v.v, zerov.v, mask_vec2.v );
                    sum_med_vec2.v = _mm256_fmadd_ps( y2v.v, y2v.v, sum_med_vec2.v );

                    // Accumulate small values only if there have not been any big values so far.
                    if ( !isbig )
                    {
                        // Fill sum_sml vector using scaling.
                        temp.v = _mm256_set1_ps( scale_sml );
                        y2v.v = _mm256_blendv_ps( zerov.v, temp.v, mask_vec2.v );
                        y2v.v = _mm256_mul_ps( x2v.v, y2v.v );
                        sum_sml_vec2.v = _mm256_fmadd_ps( y2v.v, y2v.v, sum_sml_vec2.v );
                        temp.v = _mm256_set1_ps( -0.0 );
                    }
                }
            }

            if ( !bli_horizontal_or_sf( mask_vec3.v ) )
            {
                // Scaling is not necessary; only medium values.
                sum_med_vec3.v = _mm256_fmadd_ps( x3v.v, x3v.v, sum_med_vec3.v );
            }
            else
            {
                // Mask vector which indicate whether xi > thres_big.
                mask_vec3.v = _mm256_cmp_ps( x3v.v, thres_big_vec.v, _CMP_GT_OQ );

                if ( bli_horizontal_or_sf( mask_vec3.v ) )
                {
                    isbig = true;
                    // Fill sum_med vector without scaling.
                    y3v.v = _mm256_blendv_ps( x3v.v, zerov.v, mask_vec3.v );
                    sum_med_vec3.v = _mm256_fmadd_ps( y3v.v, y3v.v, sum_med_vec3.v );

                    // Fill sum_big vector using scaling.
                    temp.v = _mm256_set1_ps( scale_big );
                    y3v.v = _mm256_blendv_ps( zerov.v, temp.v, mask_vec3.v );
                    y3v.v = _mm256_mul_ps( x3v.v, y3v.v );
                    sum_big_vec3.v = _mm256_fmadd_ps( y3v.v, y3v.v, sum_big_vec3.v );
                    temp.v = _mm256_set1_ps( -0.0 );
                }
                else
                {
                    // Mask vector which indicates whether xi > thres_small.
                    mask_vec3.v = _mm256_cmp_ps( x3v.v, thres_sml_vec.v, _CMP_LT_OQ );
                    // Fill sum_med vector without scaling.
                    y3v.v = _mm256_blendv_ps( x3v.v, zerov.v, mask_vec3.v );
                    sum_med_vec3.v = _mm256_fmadd_ps( y3v.v, y3v.v, sum_med_vec3.v );

                    // Accumulate small values only if there have not been any big values so far.
                    if ( !isbig )
                    {
                        // Fill sum_sml vector using scaling.
                        temp.v = _mm256_set1_ps( scale_sml );
                        y3v.v = _mm256_blendv_ps( zerov.v, temp.v, mask_vec3.v );
                        y3v.v = _mm256_mul_ps( x3v.v, y3v.v );
                        sum_sml_vec3.v = _mm256_fmadd_ps( y3v.v, y3v.v, sum_sml_vec3.v );
                        temp.v = _mm256_set1_ps( -0.0 );
                    }
                }
            }
            xt += 16;
        }

        for (; ( i + 12 ) <= n; i = i + 12)
        {
            x0v.v = _mm256_loadu_ps( (float*)xt );
            x1v.v = _mm256_loadu_ps( (float*) (xt + 4) );
            x2v.v = _mm256_loadu_ps( (float*) (xt + 8) );

            // Getting the abs of the vector elements.
            x0v.v = _mm256_andnot_ps( temp.v, x0v.v );
            x1v.v = _mm256_andnot_ps( temp.v, x1v.v );
            x2v.v = _mm256_andnot_ps( temp.v, x2v.v );

            // Check if any of the values is a NaN and if so, return.
            mask_vec0.v = _mm256_cmp_ps(x0v.v, x0v.v, _CMP_UNORD_Q);
            mask_vec1.v = _mm256_cmp_ps(x1v.v, x1v.v, _CMP_UNORD_Q);
            mask_vec2.v = _mm256_cmp_ps(x2v.v, x2v.v, _CMP_UNORD_Q);
            if ( bli_horizontal_or_sf( mask_vec0.v ) )
            {
                *norm = NAN;
                if ( ( incx != 1 ) && bli_mem_is_alloc( &mem_bufX ) )
                {
                    #ifdef BLIS_ENABLE_MEM_TRACING
                        printf( "bli_scnorm2fv_unb_var1_avx2(): releasing mem pool block\n" );
                    #endif
                    // Return the buffer to pool.
                    bli_membrk_release( &rntm , &mem_bufX );
                }

                AOCL_DTL_TRACE_EXIT( AOCL_DTL_LEVEL_TRACE_3 );
                return;
            }
            if ( bli_horizontal_or_sf( mask_vec1.v ) )
            {
                *norm = NAN;
                if ( ( incx != 1 ) && bli_mem_is_alloc( &mem_bufX ) )
                {
                    #ifdef BLIS_ENABLE_MEM_TRACING
                        printf( "bli_scnorm2fv_unb_var1_avx2(): releasing mem pool block\n" );
                    #endif
                    // Return the buffer to pool.
                    bli_membrk_release( &rntm , &mem_bufX );
                }

                AOCL_DTL_TRACE_EXIT( AOCL_DTL_LEVEL_TRACE_3 );
                return;
            }
            if ( bli_horizontal_or_sf( mask_vec2.v ) )
            {
                *norm = NAN;
                if ( ( incx != 1 ) && bli_mem_is_alloc( &mem_bufX ) )
                {
                    #ifdef BLIS_ENABLE_MEM_TRACING
                        printf( "bli_scnorm2fv_unb_var1_avx2(): releasing mem pool block\n" );
                    #endif
                    // Return the buffer to pool.
                    bli_membrk_release( &rntm , &mem_bufX );
                }

                AOCL_DTL_TRACE_EXIT( AOCL_DTL_LEVEL_TRACE_3 );
                return;
            }

            // Mask vectors which indicate whether
            // xi<=thres_sml or xi>=thres_big.
            mask_vec0.v = CMP256_sf( x0v.v, thres_sml_vec.v, thres_big_vec.v );
            mask_vec1.v = CMP256_sf( x1v.v, thres_sml_vec.v, thres_big_vec.v );
            mask_vec2.v = CMP256_sf( x2v.v, thres_sml_vec.v, thres_big_vec.v );

            if ( !bli_horizontal_or_sf( mask_vec0.v ) )
            {
                // Scaling is not necessary; only medium values.
                sum_med_vec0.v = _mm256_fmadd_ps( x0v.v, x0v.v, sum_med_vec0.v );
            }
            else
            {
                // Mask vector which indicate whether xi > thres_big.
                mask_vec0.v = _mm256_cmp_ps( x0v.v, thres_big_vec.v, _CMP_GT_OQ );

                if ( bli_horizontal_or_sf( mask_vec0.v ) )
                {
                    isbig = true;

                    // Fill sum_med vector without scaling.
                    y0v.v = _mm256_blendv_ps( x0v.v, zerov.v, mask_vec0.v );
                    sum_med_vec0.v = _mm256_fmadd_ps( y0v.v, y0v.v, sum_med_vec0.v );

                    // Fill sum_big vector using scaling.
                    temp.v = _mm256_set1_ps( scale_big );
                    y0v.v = _mm256_blendv_ps( zerov.v, temp.v, mask_vec0.v );
                    y0v.v = _mm256_mul_ps( x0v.v, y0v.v );
                    sum_big_vec0.v = _mm256_fmadd_ps( y0v.v, y0v.v, sum_big_vec0.v );
                    temp.v = _mm256_set1_ps( -0.0 );
                }
                else
                {
                    // Mask vector which indicates whether xi > thres_small.
                    mask_vec0.v = _mm256_cmp_ps( x0v.v, thres_sml_vec.v, _CMP_LT_OQ );
                    // Fill sum_med vector without scaling.
                    y0v.v = _mm256_blendv_ps( x0v.v, zerov.v, mask_vec0.v );
                    sum_med_vec0.v = _mm256_fmadd_ps( y0v.v, y0v.v, sum_med_vec0.v );

                    // Accumulate small values only if there have not been any big values so far.
                    if ( !isbig )
                    {
                        // Fill sum_sml vector using scaling.
                        temp.v = _mm256_set1_ps( scale_sml );
                        y0v.v = _mm256_blendv_ps( zerov.v, temp.v, mask_vec0.v );
                        y0v.v = _mm256_mul_ps( x0v.v, y0v.v );
                        sum_sml_vec0.v = _mm256_fmadd_ps( y0v.v, y0v.v, sum_sml_vec0.v );
                        temp.v = _mm256_set1_ps( -0.0 );
                    }
                }
            }
            if ( !bli_horizontal_or_sf( mask_vec1.v ) )
            {
                // Scaling is not necessary; only medium values.
                sum_med_vec1.v = _mm256_fmadd_ps( x1v.v, x1v.v, sum_med_vec1.v );
            }
            else
            {
                // Mask vector which indicate whether xi > thres_big.
                mask_vec1.v = _mm256_cmp_ps( x1v.v, thres_big_vec.v, _CMP_GT_OQ );

                if ( bli_horizontal_or_sf( mask_vec1.v ) )
                {
                    isbig = true;

                    // Fill sum_med vector without scaling.
                    y1v.v = _mm256_blendv_ps( x1v.v, zerov.v, mask_vec1.v );
                    sum_med_vec1.v = _mm256_fmadd_ps( y1v.v, y1v.v, sum_med_vec1.v );

                    // Fill sum_big vector using scaling.
                    temp.v = _mm256_set1_ps( scale_big );
                    y1v.v = _mm256_blendv_ps( zerov.v, temp.v, mask_vec1.v );
                    y1v.v = _mm256_mul_ps( x1v.v, y1v.v );
                    sum_big_vec1.v = _mm256_fmadd_ps( y1v.v, y1v.v, sum_big_vec1.v );
                    temp.v = _mm256_set1_ps( -0.0 );
                }
                else
                {
                    // Mask vector which indicates whether xi > thres_small.
                    mask_vec1.v = _mm256_cmp_ps( x1v.v, thres_sml_vec.v, _CMP_LT_OQ );
                    // Fill sum_med vector without scaling.
                    y1v.v = _mm256_blendv_ps( x1v.v, zerov.v, mask_vec1.v );
                    sum_med_vec1.v = _mm256_fmadd_ps( y1v.v, y1v.v, sum_med_vec1.v );

                    // Accumulate small values only if there have not been any big values so far.
                    if ( !isbig )
                    {
                        // Fill sum_sml vector using scaling.
                        temp.v = _mm256_set1_ps( scale_sml );
                        y1v.v = _mm256_blendv_ps( zerov.v, temp.v, mask_vec1.v );
                        y1v.v = _mm256_mul_ps( x1v.v, y1v.v );
                        sum_sml_vec1.v = _mm256_fmadd_ps( y1v.v, y1v.v, sum_sml_vec1.v );
                        temp.v = _mm256_set1_ps( -0.0 );
                    }
                }
            }
            if ( !bli_horizontal_or_sf( mask_vec2.v ) )
            {
                // Scaling is not necessary; only medium values.
                sum_med_vec2.v = _mm256_fmadd_ps( x2v.v, x2v.v, sum_med_vec2.v );
            }
            else
            {
                // Mask vector which indicate whether xi > thres_big.
                mask_vec2.v = _mm256_cmp_ps( x2v.v, thres_big_vec.v, _CMP_GT_OQ );

                if ( bli_horizontal_or_sf( mask_vec2.v ) )
                {
                    isbig = true;

                    // Fill sum_med vector without scaling.
                    y2v.v = _mm256_blendv_ps( x2v.v, zerov.v, mask_vec2.v );
                    sum_med_vec2.v = _mm256_fmadd_ps( y2v.v, y2v.v, sum_med_vec2.v );

                    // Fill sum_big vector using scaling.
                    temp.v = _mm256_set1_ps( scale_big );
                    y2v.v = _mm256_blendv_ps( zerov.v, temp.v, mask_vec2.v );
                    y2v.v = _mm256_mul_ps( x2v.v, y2v.v );
                    sum_big_vec2.v = _mm256_fmadd_ps( y2v.v, y2v.v, sum_big_vec2.v );
                    temp.v = _mm256_set1_ps( -0.0 );
                }
                else
                {
                    // Mask vector which indicates whether xi > thres_small.
                    mask_vec2.v = _mm256_cmp_ps( x2v.v, thres_sml_vec.v, _CMP_LT_OQ );
                    // Fill sum_med vector without scaling.
                    y2v.v = _mm256_blendv_ps( x2v.v, zerov.v, mask_vec2.v );
                    sum_med_vec2.v = _mm256_fmadd_ps( y2v.v, y2v.v, sum_med_vec2.v );

                    // Accumulate small values only if there have not been any big values so far.
                    if ( !isbig )
                    {
                        // Fill sum_sml vector using scaling.
                        temp.v = _mm256_set1_ps( scale_sml );
                        y2v.v = _mm256_blendv_ps( zerov.v, temp.v, mask_vec2.v );
                        y2v.v = _mm256_mul_ps( x2v.v, y2v.v );
                        sum_sml_vec2.v = _mm256_fmadd_ps( y2v.v, y2v.v, sum_sml_vec2.v );
                        temp.v = _mm256_set1_ps( -0.0 );
                    }
                }
            }
            
            xt += 12;
        }
        
        for (; ( i + 8 ) <= n; i = i + 8)
        {
            x0v.v = _mm256_loadu_ps( (float*)xt );
            x1v.v = _mm256_loadu_ps( (float*) (xt + 4) );

            // Getting the abs of the vector elements.
            x0v.v = _mm256_andnot_ps( temp.v, x0v.v );
            x1v.v = _mm256_andnot_ps( temp.v, x1v.v );

            // Check if any of the values is a NaN and if so, return.
            mask_vec0.v = _mm256_cmp_ps(x0v.v, x0v.v, _CMP_UNORD_Q);
            mask_vec1.v = _mm256_cmp_ps(x1v.v, x1v.v, _CMP_UNORD_Q);
            if ( bli_horizontal_or_sf( mask_vec0.v ) )
            {
                *norm = NAN;
                if ( ( incx != 1 ) && bli_mem_is_alloc( &mem_bufX ) )
                {
                    #ifdef BLIS_ENABLE_MEM_TRACING
                        printf( "bli_scnorm2fv_unb_var1_avx2(): releasing mem pool block\n" );
                    #endif
                    // Return the buffer to pool.
                    bli_membrk_release( &rntm , &mem_bufX );
                }

                AOCL_DTL_TRACE_EXIT( AOCL_DTL_LEVEL_TRACE_3 );
                return;
            }
            if ( bli_horizontal_or_sf( mask_vec1.v ) )
            {
                *norm = NAN;
                if ( ( incx != 1 ) && bli_mem_is_alloc( &mem_bufX ) )
                {
                    #ifdef BLIS_ENABLE_MEM_TRACING
                        printf( "bli_scnorm2fv_unb_var1_avx2(): releasing mem pool block\n" );
                    #endif
                    // Return the buffer to pool.
                    bli_membrk_release( &rntm , &mem_bufX );
                }

                AOCL_DTL_TRACE_EXIT( AOCL_DTL_LEVEL_TRACE_3 );
                return;
            }
            // Mask vectors which indicate whether
            // xi<=thres_sml or xi>=thres_big.
            mask_vec0.v = CMP256_sf( x0v.v, thres_sml_vec.v, thres_big_vec.v );
            mask_vec1.v = CMP256_sf( x1v.v, thres_sml_vec.v, thres_big_vec.v );

            if ( !bli_horizontal_or_sf( mask_vec0.v ) )
            {
                // Scaling is not necessary; only medium values.
                sum_med_vec0.v = _mm256_fmadd_ps( x0v.v, x0v.v, sum_med_vec0.v );
            }
            else
            {
                // Mask vector which indicate whether xi > thres_big.
                mask_vec0.v = _mm256_cmp_ps( x0v.v, thres_big_vec.v, _CMP_GT_OQ );

                if ( bli_horizontal_or_sf( mask_vec0.v ) )
                {
                    isbig = true;

                    // Fill sum_med vector without scaling.
                    y0v.v = _mm256_blendv_ps( x0v.v, zerov.v, mask_vec0.v );
                    sum_med_vec0.v = _mm256_fmadd_ps( y0v.v, y0v.v, sum_med_vec0.v );

                    // Fill sum_big vector using scaling.
                    temp.v = _mm256_set1_ps( scale_big );
                    y0v.v = _mm256_blendv_ps( zerov.v, temp.v, mask_vec0.v );
                    y0v.v = _mm256_mul_ps( x0v.v, y0v.v );
                    sum_big_vec0.v = _mm256_fmadd_ps( y0v.v, y0v.v, sum_big_vec0.v );
                    temp.v = _mm256_set1_ps( -0.0 );
                }
                else
                {
                    // Mask vector which indicates whether xi > thres_small.
                    mask_vec0.v = _mm256_cmp_ps( x0v.v, thres_sml_vec.v, _CMP_LT_OQ );
                    // Fill sum_med vector without scaling.
                    y0v.v = _mm256_blendv_ps( x0v.v, zerov.v, mask_vec0.v );
                    sum_med_vec0.v = _mm256_fmadd_ps( y0v.v, y0v.v, sum_med_vec0.v );

                    // Accumulate small values only if there have not been any big values so far.
                    if ( !isbig )
                    {
                        // Fill sum_sml vector using scaling.
                        temp.v = _mm256_set1_ps( scale_sml );
                        y0v.v = _mm256_blendv_ps( zerov.v, temp.v, mask_vec0.v );
                        y0v.v = _mm256_mul_ps( x0v.v, y0v.v );
                        sum_sml_vec0.v = _mm256_fmadd_ps( y0v.v, y0v.v, sum_sml_vec0.v );
                        temp.v = _mm256_set1_ps( -0.0 );
                    }
                }
            }
            if ( !bli_horizontal_or_sf( mask_vec1.v ) )
            {
                // Scaling is not necessary; only medium values.
                sum_med_vec1.v = _mm256_fmadd_ps( x1v.v, x1v.v, sum_med_vec1.v );
            }
            else
            {
                // Mask vector which indicate whether xi > thres_big.
                mask_vec1.v = _mm256_cmp_ps( x1v.v, thres_big_vec.v, _CMP_GT_OQ );

                if ( bli_horizontal_or_sf( mask_vec1.v ) )
                {
                    isbig = true;

                    // Fill sum_med vector without scaling.
                    y1v.v = _mm256_blendv_ps( x1v.v, zerov.v, mask_vec1.v );
                    sum_med_vec1.v = _mm256_fmadd_ps( y1v.v, y1v.v, sum_med_vec1.v );

                    // Fill sum_big vector using scaling.
                    temp.v = _mm256_set1_ps( scale_big );
                    y1v.v = _mm256_blendv_ps( zerov.v, temp.v, mask_vec1.v );
                    y1v.v = _mm256_mul_ps( x1v.v, y1v.v );
                    sum_big_vec1.v = _mm256_fmadd_ps( y1v.v, y1v.v, sum_big_vec1.v );
                    temp.v = _mm256_set1_ps( -0.0 );
                }
                else
                {
                    // Mask vector which indicates whether xi > thres_small.
                    mask_vec1.v = _mm256_cmp_ps( x1v.v, thres_sml_vec.v, _CMP_LT_OQ );
                    // Fill sum_med vector without scaling.
                    y1v.v = _mm256_blendv_ps( x1v.v, zerov.v, mask_vec1.v );
                    sum_med_vec1.v = _mm256_fmadd_ps( y1v.v, y1v.v, sum_med_vec1.v );

                    // Accumulate small values only if there have not been any big values so far.
                    if ( !isbig )
                    {
                        // Fill sum_sml vector using scaling.
                        temp.v = _mm256_set1_ps( scale_sml );
                        y1v.v = _mm256_blendv_ps( zerov.v, temp.v, mask_vec1.v );
                        y1v.v = _mm256_mul_ps( x1v.v, y1v.v );
                        sum_sml_vec1.v = _mm256_fmadd_ps( y1v.v, y1v.v, sum_sml_vec1.v );
                        temp.v = _mm256_set1_ps( -0.0 );
                    }
                }
            }
            xt += 8;
        }
        // This seems to not be improving performance.
        #if 0
        for (; ( i + 4 ) <= n; i = i + 4)
        {
            x0v.v = _mm256_loadu_ps( (float*)xt );

            // Getting the abs of the vector elements.
            x0v.v = _mm256_andnot_ps( temp.v, x0v.v );

            // Check if any of the values is a NaN and if so, return.
            mask_vec0.v = _mm256_cmp_ps(x0v.v, x0v.v, _CMP_UNORD_Q);
            if ( bli_horizontal_or_sf( mask_vec0.v ) )
            {
                *norm = NAN;
                if ( ( incx != 1 ) && bli_mem_is_alloc( &mem_bufX ) )
                {
                    #ifdef BLIS_ENABLE_MEM_TRACING
                        printf( "bli_scnorm2fv_unb_var1_avx2(): releasing mem pool block\n" );
                    #endif
                    // Return the buffer to pool.
                    bli_membrk_release( &rntm , &mem_bufX );
                }

                AOCL_DTL_TRACE_EXIT( AOCL_DTL_LEVEL_TRACE_3 );
                return;
            }
            // Mask vectors which indicate whether
            // xi<=thres_sml or xi>=thres_big.
            mask_vec0.v = CMP256_sf( x0v.v, thres_sml_vec.v, thres_big_vec.v );

            if ( !bli_horizontal_or_sf( mask_vec0.v ) )
            {
                // Scaling is not necessary; only medium values.
                sum_med_vec0.v = _mm256_fmadd_ps( x0v.v, x0v.v, sum_med_vec0.v );
            }
            else
            {
                // Mask vector which indicate whether xi > thres_big.
                mask_vec0.v = _mm256_cmp_ps( x0v.v, thres_big_vec.v, _CMP_GT_OQ );

                if ( bli_horizontal_or_sf( mask_vec0.v ) )
                {
                    isbig = true;

                    // Fill sum_med vector without scaling.
                    y0v.v = _mm256_blendv_ps( x0v.v, zerov.v, mask_vec0.v );
                    sum_med_vec0.v = _mm256_fmadd_ps( y0v.v, y0v.v, sum_med_vec0.v );

                    // Fill sum_big vector using scaling.
                    temp.v = _mm256_set1_ps( scale_big );
                    y0v.v = _mm256_blendv_ps( zerov.v, temp.v, mask_vec0.v );
                    y0v.v = _mm256_mul_ps( x0v.v, y0v.v );
                    sum_big_vec0.v = _mm256_fmadd_ps( y0v.v, y0v.v, sum_big_vec0.v );
                    temp.v = _mm256_set1_ps( -0.0 );
                }
                else
                {
                    // Mask vector which indicates whether xi > thres_small.
                    mask_vec0.v = _mm256_cmp_ps( x0v.v, thres_sml_vec.v, _CMP_LT_OQ );
                    // Fill sum_med vector without scaling.
                    y0v.v = _mm256_blendv_ps( x0v.v, zerov.v, mask_vec0.v );
                    sum_med_vec0.v = _mm256_fmadd_ps( y0v.v, y0v.v, sum_med_vec0.v );

                    // Accumulate small values only if there have not been any big values so far.
                    if ( !isbig )
                    {
                        // Fill sum_sml vector using scaling.
                        temp.v = _mm256_set1_ps( scale_sml );
                        y0v.v = _mm256_blendv_ps( zerov.v, temp.v, mask_vec0.v );
                        y0v.v = _mm256_mul_ps( x0v.v, y0v.v );
                        sum_sml_vec0.v = _mm256_fmadd_ps( y0v.v, y0v.v, sum_sml_vec0.v );
                        temp.v = _mm256_set1_ps( -0.0 );
                    }
                }
            }
            xt += 4;
        }
        #endif

        sum_sml_vec0.v = _mm256_add_ps( sum_sml_vec0.v, sum_sml_vec1.v );
        sum_sml_vec2.v = _mm256_add_ps( sum_sml_vec2.v, sum_sml_vec3.v );
        sum_sml_vec0.v = _mm256_add_ps( sum_sml_vec0.v, sum_sml_vec2.v ); 
        sum_sml = horizontal_add_sf(sum_sml_vec0.v);

        sum_med_vec0.v = _mm256_add_ps( sum_med_vec0.v, sum_med_vec1.v );
        sum_med_vec2.v = _mm256_add_ps( sum_med_vec2.v, sum_med_vec3.v );
        sum_med_vec0.v = _mm256_add_ps( sum_med_vec0.v, sum_med_vec2.v );
        sum_med = horizontal_add_sf(sum_med_vec0.v);

        sum_big_vec0.v = _mm256_add_ps( sum_big_vec0.v, sum_big_vec1.v );
        sum_big_vec2.v = _mm256_add_ps( sum_big_vec2.v, sum_big_vec3.v );
        sum_big_vec0.v = _mm256_add_ps( sum_big_vec0.v, sum_big_vec2.v );
        sum_big = horizontal_add_sf(sum_big_vec0.v);
    }

    n_remainder = n - i;
    bool hasInf = false;
    double chi_r, chi_i;
    if ( ( n_remainder > 0 ) )
    {
        // Put first the most likely to happen to avoid evaluations on if statements.
        for (i = 0; i < n_remainder; i++)
        {
            // Get real and imaginary component of the vector element.            
            bli_csgets(*xt, chi_r, chi_i);
            // Start with accumulating the real component of the vector element.
            abs_chi = bli_fabs( chi_r );
            // If any of the elements is NaN, then return NaN as a result.
            if ( bli_isnan( abs_chi ) )
            {
                *norm = abs_chi;
                if ( ( incx != 1 ) && bli_mem_is_alloc( &mem_bufX ) )
                {
                    #ifdef BLIS_ENABLE_MEM_TRACING
                        printf( "bli_scnorm2fv_unb_var1_avx2(): releasing mem pool block\n" );
                    #endif
                    // Return the buffer to pool.
                    bli_membrk_release( &rntm , &mem_bufX );
                }

                AOCL_DTL_TRACE_EXIT( AOCL_DTL_LEVEL_TRACE_3 );
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
            // Accumulate the imaginary component of the vector element.
            abs_chi = bli_fabs( chi_i );
            // If any of the elements is NaN, then return NaN as a result.
            if ( bli_isnan( abs_chi ) )
            {
                *norm = abs_chi;
                if ( ( incx != 1 ) && bli_mem_is_alloc( &mem_bufX ) )
                {
                    #ifdef BLIS_ENABLE_MEM_TRACING
                        printf( "bli_scnorm2fv_unb_var1_avx2(): releasing mem pool block\n" );
                    #endif
                    // Return the buffer to pool.
                    bli_membrk_release( &rntm , &mem_bufX );
                }

                AOCL_DTL_TRACE_EXIT( AOCL_DTL_LEVEL_TRACE_3 );
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
    if ( hasInf ) 
    {
        if ( ( incx != 1 ) && bli_mem_is_alloc( &mem_bufX ) )
        {
            #ifdef BLIS_ENABLE_MEM_TRACING
                printf( "bli_scnorm2fv_unb_var1_avx2(): releasing mem pool block\n" );
            #endif
            // Return the buffer to pool.
            bli_membrk_release( &rntm , &mem_bufX );
        }

        AOCL_DTL_TRACE_EXIT( AOCL_DTL_LEVEL_TRACE_3 );
        return;
    }
    
    // Combine accumulators.
    if ( isbig )
    {
        // Combine sum_big and sum_med if sum_med > 0.
        if ( sum_med > 0.0f )
        {
            sum_big += ( sum_med * scale_big ) * scale_big;
        }
        scale = 1.0f / scale_big;
        sumsq = sum_big;
    }
    else if ( sum_sml > 0.0f )
    {
        // Combine sum_med and sum_sml if sum_sml>0.
        if ( sum_med > 0.0f )
        {
            sum_med = sqrtf( sum_med );
            sum_sml = sqrtf( sum_sml ) / scale_sml;
            float ymin, ymax;
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
            scale = 1.0f;
            sumsq = ymax * ymax * ( 1.0f + ( ymin / ymax ) * ( ymin / ymax ) );
        }
        else
        {
            scale = 1.0f / scale_sml;
            sumsq = sum_sml;
        }
    }
    else
    {
        // If all values are mid-range:
        scale = 1.0f;
        sumsq = sum_med;
    }

    *norm = scale * sqrtf( sumsq );

    if ( ( incx != 1 ) && bli_mem_is_alloc( &mem_bufX ) )
    {
        #ifdef BLIS_ENABLE_MEM_TRACING
            printf( "bli_scnorm2fv_unb_var1_avx2(): releasing mem pool block\n" );
        #endif
        // Return the buffer to pool.
        bli_membrk_release( &rntm , &mem_bufX );
    }

    AOCL_DTL_TRACE_EXIT( AOCL_DTL_LEVEL_TRACE_3 );

    return;
}

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
                *( x_buf + x_index ) = *( x + ( x_index * incx ) );
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
    const double scale_big = pow( ( double )FLT_RADIX,  - ceil( ( DBL_MAX_EXP + 52 ) * 0.5 ) );

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
            if ( bli_horizontal_or_df( mask_vec0.v ) )
            {
                *norm = NAN;
                if ( ( incx != 1 ) && bli_mem_is_alloc( &mem_bufX ) )
                {
                    #ifdef BLIS_ENABLE_MEM_TRACING
                        printf( "bli_dnorm2fv_unb_var1_avx2(): releasing mem pool block\n" );
                    #endif
                    // Return the buffer to pool.
                    bli_membrk_release( &rntm , &mem_bufX );
                }

                AOCL_DTL_TRACE_EXIT( AOCL_DTL_LEVEL_TRACE_3 );
                return;
            }
            if ( bli_horizontal_or_df( mask_vec1.v ) )
            {
                *norm = NAN;
                if ( ( incx != 1 ) && bli_mem_is_alloc( &mem_bufX ) )
                {
                    #ifdef BLIS_ENABLE_MEM_TRACING
                        printf( "bli_dnorm2fv_unb_var1_avx2(): releasing mem pool block\n" );
                    #endif
                    // Return the buffer to pool.
                    bli_membrk_release( &rntm , &mem_bufX );
                }

                AOCL_DTL_TRACE_EXIT( AOCL_DTL_LEVEL_TRACE_3 );
                return;
            }

            // Mask vectors which indicate whether
            // xi<=thres_sml or xi>=thres_big.
            mask_vec0.v = CMP256_df( x0v.v, thres_sml_vec.v, thres_big_vec.v );
            mask_vec1.v = CMP256_df( x1v.v, thres_sml_vec.v, thres_big_vec.v );

            if ( !bli_horizontal_or_df( mask_vec0.v ) )
            {
                // Scaling is not necessary; only medium values.
                sum_med_vec0.v = _mm256_fmadd_pd( x0v.v, x0v.v, sum_med_vec0.v );
            }
            else
            {
                // Mask vector which indicate whether xi > thres_big.
                mask_vec0.v = _mm256_cmp_pd( x0v.v, thres_big_vec.v, _CMP_GT_OQ );

                if ( bli_horizontal_or_df( mask_vec0.v ) )
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

            if ( !bli_horizontal_or_df( mask_vec1.v ) )
            {
                // Scaling is not necessary; only medium values.
                sum_med_vec1.v = _mm256_fmadd_pd( x1v.v, x1v.v, sum_med_vec1.v );
            }
            else
            {
                // Mask vector which indicate whether xi > thres_big.
                mask_vec1.v = _mm256_cmp_pd( x1v.v, thres_big_vec.v, _CMP_GT_OQ );

                if ( bli_horizontal_or_df( mask_vec1.v ) )
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
            if ( bli_horizontal_or_df( mask_vec0.v ) )
            {
                *norm = NAN;
                if ( ( incx != 1 ) && bli_mem_is_alloc( &mem_bufX ) )
                {
                    #ifdef BLIS_ENABLE_MEM_TRACING
                        printf( "bli_dnorm2fv_unb_var1_avx2(): releasing mem pool block\n" );
                    #endif
                    // Return the buffer to pool.
                    bli_membrk_release( &rntm , &mem_bufX );
                }

                AOCL_DTL_TRACE_EXIT( AOCL_DTL_LEVEL_TRACE_3 );
                return;
            }

            // Mask vectors which indicate whether
            // xi<=thres_sml or xi>=thres_big.
            mask_vec0.v = CMP256_df( x0v.v, thres_sml_vec.v, thres_big_vec.v );

            if ( !bli_horizontal_or_df( mask_vec0.v ) )
            {
                // Scaling is not necessary; only medium values.
                sum_med_vec0.v = _mm256_fmadd_pd( x0v.v, x0v.v, sum_med_vec0.v );
            }
            else
            {
                // Mask vector which indicate whether xi > thres_big.
                mask_vec0.v = _mm256_cmp_pd( x0v.v, thres_big_vec.v, _CMP_GT_OQ );

                if ( bli_horizontal_or_df( mask_vec0.v ) )
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
                if ( ( incx != 1 ) && bli_mem_is_alloc( &mem_bufX ) )
                {
                    #ifdef BLIS_ENABLE_MEM_TRACING
                        printf( "bli_dnorm2fv_unb_var1_avx2(): releasing mem pool block\n" );
                    #endif
                    // Return the buffer to pool.
                    bli_membrk_release( &rntm , &mem_bufX );
                }

                AOCL_DTL_TRACE_EXIT( AOCL_DTL_LEVEL_TRACE_3 );
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
    if ( hasInf ) 
    {        
        if ( ( incx != 1 ) && bli_mem_is_alloc( &mem_bufX ) )
        {
            #ifdef BLIS_ENABLE_MEM_TRACING
                printf( "bli_dnorm2fv_unb_var1_avx2(): releasing mem pool block\n" );
            #endif
            // Return the buffer to pool.
            bli_membrk_release( &rntm , &mem_bufX );
        }

        AOCL_DTL_TRACE_EXIT( AOCL_DTL_LEVEL_TRACE_3 );
        return;
    }

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
                *( x_buf + x_index ) = *( x + ( x_index * incx ) );
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
    const double scale_big = pow( ( double )FLT_RADIX,  - ceil( ( DBL_MAX_EXP + 52 ) * 0.5 ) );

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
            x0v.v = _mm256_loadu_pd( (double*) xt );
            x1v.v = _mm256_loadu_pd( (double*) (xt + 2) );

            // Getting the abs of the vector elements.
            x0v.v = _mm256_andnot_pd( temp.v, x0v.v );
            x1v.v = _mm256_andnot_pd( temp.v, x1v.v );

            // Check if any of the values is a NaN and if so, return.
            mask_vec0.v = _mm256_cmp_pd(x0v.v, x0v.v, _CMP_UNORD_Q);
            mask_vec1.v = _mm256_cmp_pd(x1v.v, x1v.v, _CMP_UNORD_Q);
            if ( bli_horizontal_or_df( mask_vec0.v ) )
            {
                *norm = NAN;
                if ( ( incx != 1 ) && bli_mem_is_alloc( &mem_bufX ) )
                {
                    #ifdef BLIS_ENABLE_MEM_TRACING
                        printf( "bli_dznorm2fv_unb_var1_avx2(): releasing mem pool block\n" );
                    #endif
                    // Return the buffer to pool.
                    bli_membrk_release( &rntm , &mem_bufX );
                }

                AOCL_DTL_TRACE_EXIT( AOCL_DTL_LEVEL_TRACE_3 );
                return;
            }
            if ( bli_horizontal_or_df( mask_vec1.v ) )
            {
                *norm = NAN;
                if ( ( incx != 1 ) && bli_mem_is_alloc( &mem_bufX ) )
                {
                    #ifdef BLIS_ENABLE_MEM_TRACING
                        printf( "bli_dznorm2fv_unb_var1_avx2(): releasing mem pool block\n" );
                    #endif
                    // Return the buffer to pool.
                    bli_membrk_release( &rntm , &mem_bufX );
                }

                AOCL_DTL_TRACE_EXIT( AOCL_DTL_LEVEL_TRACE_3 );
                return;
            }

            // Mask vectors which indicate whether
            // xi<=thres_sml or xi>=thres_big.
            mask_vec0.v = CMP256_df( x0v.v, thres_sml_vec.v, thres_big_vec.v );
            mask_vec1.v = CMP256_df( x1v.v, thres_sml_vec.v, thres_big_vec.v );

            if ( !bli_horizontal_or_df( mask_vec0.v ) )
            {
                // Scaling is not necessary; only medium values.
                sum_med_vec0.v = _mm256_fmadd_pd( x0v.v, x0v.v, sum_med_vec0.v );
            }
            else
            {
                // Mask vector which indicate whether xi > thres_big.
                mask_vec0.v = _mm256_cmp_pd( x0v.v, thres_big_vec.v, _CMP_GT_OQ );

                if ( bli_horizontal_or_df( mask_vec0.v ) )
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

            if ( !bli_horizontal_or_df( mask_vec1.v ) )
            {
                // Scaling is not necessary; only medium values.
                sum_med_vec1.v = _mm256_fmadd_pd( x1v.v, x1v.v, sum_med_vec1.v );
            }
            else
            {
                // Mask vector which indicate whether xi > thres_big.
                mask_vec1.v = _mm256_cmp_pd( x1v.v, thres_big_vec.v, _CMP_GT_OQ );

                if ( bli_horizontal_or_df( mask_vec1.v ) )
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
            x0v.v = _mm256_loadu_pd( (double*) xt );

            // Getting the abs of the vector elements.
            x0v.v = _mm256_andnot_pd( temp.v, x0v.v );

            // Check if any of the values is a NaN and if so, return.
            mask_vec0.v = _mm256_cmp_pd(x0v.v, x0v.v, _CMP_UNORD_Q);
            if ( bli_horizontal_or_df( mask_vec0.v ) )
            {
                *norm = NAN;
                if ( ( incx != 1 ) && bli_mem_is_alloc( &mem_bufX ) )
                {
                    #ifdef BLIS_ENABLE_MEM_TRACING
                        printf( "bli_dznorm2fv_unb_var1_avx2(): releasing mem pool block\n" );
                    #endif
                    // Return the buffer to pool.
                    bli_membrk_release( &rntm , &mem_bufX );
                }

                AOCL_DTL_TRACE_EXIT( AOCL_DTL_LEVEL_TRACE_3 );
                return;
            }

            // Mask vectors which indicate whether
            // xi<=thres_sml or xi>=thres_big.
            mask_vec0.v = CMP256_df( x0v.v, thres_sml_vec.v, thres_big_vec.v );

            if ( !bli_horizontal_or_df( mask_vec0.v ) )
            {
                // Scaling is not necessary; only medium values.
                sum_med_vec0.v = _mm256_fmadd_pd( x0v.v, x0v.v, sum_med_vec0.v );
            }
            else
            {
                // Mask vector which indicate whether xi > thres_big.
                mask_vec0.v = _mm256_cmp_pd( x0v.v, thres_big_vec.v, _CMP_GT_OQ );

                if ( bli_horizontal_or_df( mask_vec0.v ) )
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
    double chi_r, chi_i;
    if ( ( n_remainder > 0 ) )
    {
        // Put first the most likely to happen to avoid evaluations on if statements.
        for (i = 0; i < n_remainder; i++)
        {
            // Get real and imaginary component of the vector element.
            bli_zdgets(*xt, chi_r, chi_i);

            // Start with accumulating the real component of the vector element.
            abs_chi = bli_fabs( chi_r );
            // If any of the elements is NaN, then return NaN as a result.
            if ( bli_isnan( abs_chi ) )
            {
                *norm = abs_chi;
                if ( ( incx != 1 ) && bli_mem_is_alloc( &mem_bufX ) )
                {
                    #ifdef BLIS_ENABLE_MEM_TRACING
                        printf( "bli_dznorm2fv_unb_var1_avx2(): releasing mem pool block\n" );
                    #endif
                    // Return the buffer to pool.
                    bli_membrk_release( &rntm , &mem_bufX );
                }

                AOCL_DTL_TRACE_EXIT( AOCL_DTL_LEVEL_TRACE_3 );
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
                if ( ( incx != 1 ) && bli_mem_is_alloc( &mem_bufX ) )
                {
                    #ifdef BLIS_ENABLE_MEM_TRACING
                        printf( "bli_dznorm2fv_unb_var1_avx2(): releasing mem pool block\n" );
                    #endif
                    // Return the buffer to pool.
                    bli_membrk_release( &rntm , &mem_bufX );
                }

                AOCL_DTL_TRACE_EXIT( AOCL_DTL_LEVEL_TRACE_3 );
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
    if ( hasInf ) 
    {        
        if ( ( incx != 1 ) && bli_mem_is_alloc( &mem_bufX ) )
        {
            #ifdef BLIS_ENABLE_MEM_TRACING
                printf( "bli_dnorm2fv_unb_var1_avx2(): releasing mem pool block\n" );
            #endif
            // Return the buffer to pool.
            bli_membrk_release( &rntm , &mem_bufX );
        }

        AOCL_DTL_TRACE_EXIT( AOCL_DTL_LEVEL_TRACE_3 );
        return;
    }

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
