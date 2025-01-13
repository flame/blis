/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2022 - 2025, Advanced Micro Devices, Inc. All rights reserved.

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

/* Union data structure to access AVX registers
   One 512-bit AVX register holds 16 SP elements. */
typedef union
{
    __m512  v;
    float   f[16] __attribute__((aligned(64)));
} v16sf_t;

/* Union data structure to access AVX registers
   One 256-bit AVX register holds 8 SP elements. */
typedef union
{
    __m256  v;
    float   f[8] __attribute__((aligned(64)));
} v8sf_t;

/* Union data structure to access SSE registers
   One 128-bit SSE register holds 4 SP elements. */
typedef union
{
    __m128  v;
    float   f[4];
} v4sf_t;

/* Union data structure to access AVX512 registers
   One 512-bit AVX512 register holds 8 DP elements. */
typedef union
{
    __m512d v;
    double d[8] __attribute__((aligned(64)));
} v8df_t;

/* Union data structure to access AVX registers
   One 256-bit AVX register holds 4 DP elements. */
typedef union
{
    __m256d v;
    double  d[4] __attribute__((aligned(64)));
}v4df_t;

/* Union data structure to access SSE registers
   One 128-bit SSE register holds 2 DP elements. */
typedef union
{
    __m128d v;
    double  d[2];
}v2df_t;

/* Convert the nan to -ve numbers decrementing with
   the times the function is called to ensure that
   bigger numbers are assigned for nan which showed
   up first.*/
#define REMOVE_NAN_512S(reg_512) \
    { \
        /*Sign is -0.f in IEEE754 is just signbit set, all others 0*/ \
        __m512 sign_mask = _mm512_set1_ps( -0.0f ); \
 \
        /* Numbers other than NAN will become 0. */ \
        __m512 vec_mask = _mm512_mul_ps( reg_512, sign_mask ); \
 \
        /* Typecast mask into int type no clock cycle is taken just to
         * convince compiler. */ \
        __m512i int_mask_vec = _mm512_castps_si512( vec_mask ); \
        /* Extract the signbits and put it in a 16bit mask register. */ \
        __mmask16 vec_mask16 = _mm512_movepi32_mask( int_mask_vec ); \
 \
        /* Swap NAN with -ve number. */ \
        reg_512 = _mm512_mask_blend_ps( vec_mask16, _mm512_set1_ps( nan_repl ), reg_512 ); \
        nan_repl = nan_repl - 1; \
    }

// return a mask which indicates either:
// - v1 > v2
// - v1 is NaN and v2 is not
// assumes that idx(v1) > idx(v2)
// all "OQ" comparisons false if either operand NaN
#define CMP256( dt, v1, v2 ) \
        _mm256_or_p##dt( _mm256_cmp_p##dt( v1, v2, _CMP_GT_OQ ),       /* v1 > v2  ||     */ \
        _mm256_andnot_p##dt( _mm256_cmp_p##dt( v2, v2, _CMP_UNORD_Q ), /* ( !isnan(v2) && */ \
        _mm256_cmp_p##dt( v1, v1, _CMP_UNORD_Q )                       /*    isnan(v1) )  */ \
        ) \
    );

// return a mask which indicates either:
// - v1 > v2
// - v1 is NaN and v2 is not
// - v1 == v2 (maybe == NaN) and i1 < i2
// all "OQ" comparisons false if either operand NaN
#define CMP128( dt, v1, v2, i1, i2 ) \
        _mm_or_p##dt( _mm_or_p##dt( _mm_cmp_p##dt( v1, v2, _CMP_GT_OQ ), /* ( v1 > v2 ||           */ \
        _mm_andnot_p##dt( _mm_cmp_p##dt( v2, v2, _CMP_UNORD_Q ),         /*   ( !isnan(v2) &&      */ \
        _mm_cmp_p##dt( v1, v1, _CMP_UNORD_Q )                            /*      isnan(v1) ) ) ||  */ \
        ) \
        ), \
        _mm_and_p##dt( _mm_or_p##dt( _mm_cmp_p##dt( v1, v2, _CMP_EQ_OQ ), /* ( ( v1 == v2 ||        */ \
        _mm_and_p##dt( _mm_cmp_p##dt( v1, v1, _CMP_UNORD_Q ),             /*     ( isnan(v1) &&     */ \
        _mm_cmp_p##dt( v2, v2, _CMP_UNORD_Q )                             /*       isnan(v2) ) ) && */ \
        ) \
        ), \
    _mm_cmp_p##dt( i1, i2, _CMP_LT_OQ )                                  /*   i1 < i2 )            */ \
    ) \
    );

// ----------------------------------------------------------------------------
void bli_samaxv_zen_int_avx512(
    dim_t n,
    float *restrict x, inc_t incx,
    dim_t *restrict i_max,
    cntx_t *restrict cntx)
{
    AOCL_DTL_TRACE_ENTRY(AOCL_DTL_LEVEL_TRACE_3)
    /* Local pointer for accessing the input */
    float *x0 = x;

    /* Handling when vector is unit-strided */
    if( incx == 1 )
    {
        /*
          x_vec[0 ... 8]     - To load elements from x vector
          abs_x_vec[0 ... 8] - To store absolute value of elements from x vector
        */
        v16sf_t x_vec[8], abs_x_vec[8];
        /*
          sign_mask             - To compute absolute value of elements
          iter_max_vec[0 ... 4] - To compute 64-bit lane wise maximum elements
                                  from every load, and also compute the maximum
                                  scalar in every loop(reduction).
        */
        v16sf_t sign_mask, iter_max_vec[4];
        /*
          base_max_scalar - To store the absoulte maximum value from elements
                            across all iterations.
          iter_max_scalar - To store the absolute maximum value from elements
                            in the current iteration.
          start           - To store the start index of the range in which
                            base_max_scalar occurs.
          end             - To store the end index of the range in which
                            base_max_scalar occurs.
        */
        float base_max_scalar, iter_max_scalar;
        dim_t start, end;
        base_max_scalar = iter_max_scalar = 0.0f;

        /* Initializing the search space with index as 0 */
        start = end = 0;

        /* Initializing iter_max_vec to zeroes */
        iter_max_vec[0].v = _mm512_setzero_ps();
        iter_max_vec[1].v = _mm512_setzero_ps();
        iter_max_vec[2].v = _mm512_setzero_ps();
        iter_max_vec[3].v = _mm512_setzero_ps();

        /* Signed -0.0 helps in abs computation in the following way : */
        /*
            Considering IEEE 32-bit floating point numbers, here's the logic
            ~(-0.0) = ~(| 1 | 00000000 | 000...0 |) = | 0 | 11111111 | 111...1 |
            If num = -2.0 = | 1 | 10000000 | 000...0 |
            Then ((~(-0.0)) & num) = | 0 | 10000000 | 000...0 | = 2.0
        */
        sign_mask.v = _mm512_set1_ps( -0.0f );

        /* Hot-loop computation */
        dim_t i = 0;
        for( ; i + 127 < n; i += 128 )
        {
            /* Loading vector */
            x_vec[0].v = _mm512_loadu_ps(( const float* )( x0 ));
            x_vec[1].v = _mm512_loadu_ps(( const float* )( x0 + 16 ));
            x_vec[2].v = _mm512_loadu_ps(( const float* )( x0 + 32 ));
            x_vec[3].v = _mm512_loadu_ps(( const float* )( x0 + 48 ));
            x_vec[4].v = _mm512_loadu_ps(( const float* )( x0 + 64 ));
            x_vec[5].v = _mm512_loadu_ps(( const float* )( x0 + 80 ));
            x_vec[6].v = _mm512_loadu_ps(( const float* )( x0 + 96 ));
            x_vec[7].v = _mm512_loadu_ps(( const float* )( x0 + 112 ));

            /* Getting the abs value, using sign_mask */
            abs_x_vec[0].v = _mm512_andnot_ps( sign_mask.v, x_vec[0].v );
            abs_x_vec[1].v = _mm512_andnot_ps( sign_mask.v, x_vec[1].v );
            abs_x_vec[2].v = _mm512_andnot_ps( sign_mask.v, x_vec[2].v );
            abs_x_vec[3].v = _mm512_andnot_ps( sign_mask.v, x_vec[3].v );
            abs_x_vec[4].v = _mm512_andnot_ps( sign_mask.v, x_vec[4].v );
            abs_x_vec[5].v = _mm512_andnot_ps( sign_mask.v, x_vec[5].v );
            abs_x_vec[6].v = _mm512_andnot_ps( sign_mask.v, x_vec[6].v );
            abs_x_vec[7].v = _mm512_andnot_ps( sign_mask.v, x_vec[7].v );

            /* Calculating the maximum value amongst the loaded vectors */
            /*
                The behaviour of vmaxps instruction is as follows :
                Assuming v1, v2 are AVX512 registers and s1, s2 are IEEE 64-bit floating point
                numbers, we have :
                vmaxps( v1, v2 ) = max( v1[i:i+31], v2[i:i+31] ), where i = {0, 1, ... 15}
                max( s1, s2 ) = s2, if s1 = NaN or s2 = NaN
                                s1, if s1 > s2
                With the loads as first operand(abs_x_vec) and vectors with non NaN values as
                second operand(iter_max_vec), we avoid the propagation of NaNs in the computation.
            */
            iter_max_vec[0].v = _mm512_max_ps( abs_x_vec[0].v, iter_max_vec[0].v );
            iter_max_vec[1].v = _mm512_max_ps( abs_x_vec[1].v, iter_max_vec[1].v );
            iter_max_vec[2].v = _mm512_max_ps( abs_x_vec[2].v, iter_max_vec[2].v );
            iter_max_vec[3].v = _mm512_max_ps( abs_x_vec[3].v, iter_max_vec[3].v );
            iter_max_vec[0].v = _mm512_max_ps( abs_x_vec[4].v, iter_max_vec[0].v );
            iter_max_vec[1].v = _mm512_max_ps( abs_x_vec[5].v, iter_max_vec[1].v );
            iter_max_vec[2].v = _mm512_max_ps( abs_x_vec[6].v, iter_max_vec[2].v );
            iter_max_vec[3].v = _mm512_max_ps( abs_x_vec[7].v, iter_max_vec[3].v );

            /* Further computing the maximum(in order) based on the previous results */
            iter_max_vec[0].v = _mm512_max_ps( iter_max_vec[0].v, iter_max_vec[1].v );
            iter_max_vec[2].v = _mm512_max_ps( iter_max_vec[2].v, iter_max_vec[3].v );

            iter_max_vec[0].v = _mm512_max_ps( iter_max_vec[0].v, iter_max_vec[2].v );

            /* Obtaining the maximum value as a scalar, using reduction */
            iter_max_scalar = _mm512_reduce_max_ps( iter_max_vec[0].v );

            /* Updating the indices and base_max_scalar, if needed */
            if( iter_max_scalar > base_max_scalar )
            {
                base_max_scalar = iter_max_scalar;
                start = i;
                end = i + 127;
            }

            /* Incrementing the pointer based on unrolling */
            x0 += 128;
        }
        for( ; i + 63 < n; i += 64 )
        {
            /* Loading vector */
            x_vec[0].v = _mm512_loadu_ps(( const float* )( x0 ));
            x_vec[1].v = _mm512_loadu_ps(( const float* )( x0 + 16 ));
            x_vec[2].v = _mm512_loadu_ps(( const float* )( x0 + 32 ));
            x_vec[3].v = _mm512_loadu_ps(( const float* )( x0 + 48 ));

            /* Getting the abs value, using sign_mask */
            abs_x_vec[0].v = _mm512_andnot_ps( sign_mask.v, x_vec[0].v );
            abs_x_vec[1].v = _mm512_andnot_ps( sign_mask.v, x_vec[1].v );
            abs_x_vec[2].v = _mm512_andnot_ps( sign_mask.v, x_vec[2].v );
            abs_x_vec[3].v = _mm512_andnot_ps( sign_mask.v, x_vec[3].v );

            /* Calculating the maximum value amongst the loaded vectors */
            iter_max_vec[0].v = _mm512_max_ps( abs_x_vec[0].v, iter_max_vec[0].v );
            iter_max_vec[1].v = _mm512_max_ps( abs_x_vec[1].v, iter_max_vec[1].v );
            iter_max_vec[0].v = _mm512_max_ps( abs_x_vec[2].v, iter_max_vec[0].v );
            iter_max_vec[1].v = _mm512_max_ps( abs_x_vec[3].v, iter_max_vec[1].v );

            iter_max_vec[0].v = _mm512_max_ps( iter_max_vec[0].v, iter_max_vec[1].v );

            // Obtaining the maximum value as a scalar, using reduction
            iter_max_scalar = _mm512_reduce_max_ps( iter_max_vec[0].v );

            // Updating the indices and base_max_scalar, if needed
            if( iter_max_scalar > base_max_scalar )
            {
                base_max_scalar = iter_max_scalar;
                start = i;
                end = i + 63;
            }

            /* Incrementing the pointer based on unrolling */
            x0 += 64;
        }
        for( ; i + 31 < n; i += 32 )
        {
            /* Loading vector */
            x_vec[0].v = _mm512_loadu_ps(( const float* )( x0 ));
            x_vec[1].v = _mm512_loadu_ps(( const float* )( x0 + 16 ));

            /* Getting the abs value, using sign_mask */
            abs_x_vec[0].v = _mm512_andnot_ps( sign_mask.v, x_vec[0].v );
            abs_x_vec[1].v = _mm512_andnot_ps( sign_mask.v, x_vec[1].v );

            /* Calculating the maximum value amongst the loaded vectors */
            iter_max_vec[0].v = _mm512_max_ps( abs_x_vec[0].v, iter_max_vec[0].v );
            iter_max_vec[0].v = _mm512_max_ps( abs_x_vec[1].v, iter_max_vec[0].v );

            /* Obtaining the maximum value as a scalar, using reduction */
            iter_max_scalar = _mm512_reduce_max_ps( iter_max_vec[0].v );

            /* Obtaining the maximum value as a scalar, using reduction */
            iter_max_scalar = _mm512_reduce_max_ps( iter_max_vec[0].v );

            /* Updating the indices and base_max_scalar, if needed */
            if( iter_max_scalar > base_max_scalar )
            {
                base_max_scalar = iter_max_scalar;
                start = i;
                end = i + 31;
            }

            /* Incrementing the pointer based on unrolling */
            x0 += 32;
        }
        for( ; i + 15 < n; i += 16 )
        {
            /* Loading vector */
            x_vec[0].v = _mm512_loadu_ps(( const float* )( x0 ));

            /* Getting abs value */
            abs_x_vec[0].v = _mm512_andnot_ps( sign_mask.v, x_vec[0].v );

            /* Calculating the maximum value amongst the loaded vectors */
            iter_max_vec[0].v = _mm512_max_ps( abs_x_vec[0].v, iter_max_vec[0].v );

            /* Obtaining the maximum value as a scalar, using reduction */
            iter_max_scalar = _mm512_reduce_max_ps( iter_max_vec[0].v );

            /* Updating the indices and base_max_scalar, if needed */
            if( iter_max_scalar > base_max_scalar )
            {
                base_max_scalar = iter_max_scalar;
                start = i;
                end = i + 15;
            }
            /* Incrementing the pointer based on unrolling */
            x0 += 16;
        }
        if( i < n )
        {
            /* Setting the mask for the fringe case */
            __mmask16 n_mask = ( 1 << ( n - i ) ) - 1;

            /* Loading vector */
            x_vec[0].v = _mm512_maskz_loadu_ps( n_mask, ( const float* )( x0 ) );

            /* Getting the abs value */
            abs_x_vec[0].v = _mm512_andnot_ps( sign_mask.v, x_vec[0].v );

            /* Calculating the maximum value amongst the loaded vectors */
            iter_max_vec[0].v = _mm512_max_ps( abs_x_vec[0].v, iter_max_vec[0].v );

            /* Obtaining the maximum value as a scalar, using reduction */
            iter_max_scalar = _mm512_reduce_max_ps( iter_max_vec[0].v );

            /* Updating the indices and base_max_scalar, if needed */
            if( iter_max_scalar > base_max_scalar )
            {
                base_max_scalar = iter_max_scalar;
                start = i;
                end = n - 1;
            }
        }

        /* Reduction to find the first occurence of absolute maximum */
        /* Update the pointer based on new 'start' value */
        x0 = x + start;
        /* Search for the index in the final search space */
        while( start < end )
        {
            /* In case we encounter the abs max value, terminate the loop */
            if( fabs( *x0 ) == base_max_scalar )
                break;

            /* Update the pointer and the offset */
            x0 += 1;
            start += 1;
        }

        /* Set the actual parameter with the index */
        *i_max = start;
    }

    else
    {
        float chi1, abs_chi1, abs_chi1_max;
        dim_t i_max_l = 0;
        abs_chi1_max = -1.0f;
        for (dim_t i = 0; i < n; ++i)
        {
            chi1 = *( x + ( i ) * incx );

            /* Assign abs_chi1 with its absolute value. */
            abs_chi1 = fabs(chi1);

            /* If the absolute value of the current element exceeds that of
               the previous largest, save it and its index. */
            if ( abs_chi1_max < abs_chi1 )
            {
                abs_chi1_max = abs_chi1;
                i_max_l = i;
            }
        }

        /* Set the actual parameter with the index */
        *i_max = i_max_l;
    }

    AOCL_DTL_TRACE_EXIT(AOCL_DTL_LEVEL_TRACE_3)
}

// -----------------------------------------------------------------------------
/* Converts all the NAN to a negative number less than previously encountered NANs*/
#define REMOVE_NAN_512D(reg_512) \
    { \
        __m512d sign_mask = _mm512_set1_pd( -0.0f ); \
 \
        /* Numbers other than NAN will become 0. */ \
        __m512d vec_mask = _mm512_mul_pd( reg_512, sign_mask ); \
 \
        /* Producing an 8-bit mask. */ \
        __m512i int_mask_vec = _mm512_castpd_si512( vec_mask ); \
        __mmask8 vec_mask8 = _mm512_movepi64_mask( int_mask_vec ); \
 \
        /* Replacing all the NAN with negative numbers. */ \
        reg_512 = _mm512_mask_blend_pd( vec_mask8, _mm512_set1_pd( nan_repl ), reg_512 ); \
        nan_repl = nan_repl - 1; \
    }

/*----------------------------------------------------------------------------------------------------*/
BLIS_EXPORT_BLIS void bli_damaxv_zen_int_avx512
(
    dim_t n,
    double *restrict x, inc_t incx,
    dim_t *restrict i_max,
    cntx_t *restrict cntx
)
{
    AOCL_DTL_TRACE_ENTRY(AOCL_DTL_LEVEL_TRACE_3)
    /* Local pointer for accessing the input */
    double *x0 = x;

    /* Handling when vector is unit-strided */
    if( incx == 1 )
    {
        /*
          x_vec[0 ... 8]     - To load elements from x vector
          abs_x_vec[0 ... 8] - To store absolute value of elements from x vector
        */
        v8df_t x_vec[8], abs_x_vec[8];
        /*
          sign_mask             - To compute absolute value of elements
          iter_max_vec[0 ... 4] - To compute 64-bit lane wise maximum elements
                                  from every load, and also compute the maximum
                                  scalar in every loop(reduction).
        */
        v8df_t sign_mask, iter_max_vec[4];
        /*
          base_max_scalar - To store the absoulte maximum value from elements
                            across all iterations.
          iter_max_scalar - To store the absolute maximum value from elements
                            in the current iteration.
          start           - To store the start index of the range in which
                            base_max_scalar occurs.
          end             - To store the end index of the range in which
                            base_max_scalar occurs.
        */
        double base_max_scalar, iter_max_scalar;
        dim_t start, end;
        base_max_scalar = iter_max_scalar = 0.0;

        /* Initializing the search space with index as 0 */
        start = end = 0;

        /* Initializing iter_max_vec to zeroes */
        iter_max_vec[0].v = _mm512_setzero_pd();
        iter_max_vec[1].v = _mm512_setzero_pd();
        iter_max_vec[2].v = _mm512_setzero_pd();
        iter_max_vec[3].v = _mm512_setzero_pd();

        /* Signed -0.0 helps in abs computation in the following way : */
        /*
            Considering IEEE 64-bit floating point numbers, here's the logic
            ~(-0.0) = ~(| 1 | 00000000 | 000...0 |) = | 0 | 11111111 | 111...1 |
            If num = -2.0 = | 1 | 10000000 | 000...0 |
            Then ((~(-0.0)) & num) = | 0 | 10000000 | 000...0 | = 2.0
        */
        sign_mask.v = _mm512_set1_pd( -0.0 );

        /* Hot-loop computation */
        dim_t i = 0;
        for( ; i + 63 < n; i += 64 )
        {
            /* Loading vector */
            x_vec[0].v = _mm512_loadu_pd(( const double* )( x0 ));
            x_vec[1].v = _mm512_loadu_pd(( const double* )( x0 + 8 ));
            x_vec[2].v = _mm512_loadu_pd(( const double* )( x0 + 16 ));
            x_vec[3].v = _mm512_loadu_pd(( const double* )( x0 + 24 ));
            x_vec[4].v = _mm512_loadu_pd(( const double* )( x0 + 32 ));
            x_vec[5].v = _mm512_loadu_pd(( const double* )( x0 + 40 ));
            x_vec[6].v = _mm512_loadu_pd(( const double* )( x0 + 48 ));
            x_vec[7].v = _mm512_loadu_pd(( const double* )( x0 + 56 ));

            /* Getting the abs value, using sign_mask */
            abs_x_vec[0].v = _mm512_andnot_pd( sign_mask.v, x_vec[0].v );
            abs_x_vec[1].v = _mm512_andnot_pd( sign_mask.v, x_vec[1].v );
            abs_x_vec[2].v = _mm512_andnot_pd( sign_mask.v, x_vec[2].v );
            abs_x_vec[3].v = _mm512_andnot_pd( sign_mask.v, x_vec[3].v );
            abs_x_vec[4].v = _mm512_andnot_pd( sign_mask.v, x_vec[4].v );
            abs_x_vec[5].v = _mm512_andnot_pd( sign_mask.v, x_vec[5].v );
            abs_x_vec[6].v = _mm512_andnot_pd( sign_mask.v, x_vec[6].v );
            abs_x_vec[7].v = _mm512_andnot_pd( sign_mask.v, x_vec[7].v );

            /* Calculating the maximum value amongst the loaded vectors */
            /*
                The behaviour of vmaxpd instruction is as follows :
                Assuming v1, v2 are AVX512 registers and s1, s2 are IEEE 64-bit floating point
                numbers, we have :
                vmaxpd( v1, v2 ) = max( v1[i:i+63], v2[i:i+63] ), where i = {0, 1, ... 7}
                max( s1, s2 ) = s2, if s1 = NaN or s2 = NaN
                                s1, if s1 > s2
                With the loads as first operand(abs_x_vec) and vectors with non NaN values as
                second operand(iter_max_vec), we avoid the propagation of NaNs in the computation.
            */
            iter_max_vec[0].v = _mm512_max_pd( abs_x_vec[0].v, iter_max_vec[0].v );
            iter_max_vec[1].v = _mm512_max_pd( abs_x_vec[1].v, iter_max_vec[1].v );
            iter_max_vec[2].v = _mm512_max_pd( abs_x_vec[2].v, iter_max_vec[2].v );
            iter_max_vec[3].v = _mm512_max_pd( abs_x_vec[3].v, iter_max_vec[3].v );
            iter_max_vec[0].v = _mm512_max_pd( abs_x_vec[4].v, iter_max_vec[0].v );
            iter_max_vec[1].v = _mm512_max_pd( abs_x_vec[5].v, iter_max_vec[1].v );
            iter_max_vec[2].v = _mm512_max_pd( abs_x_vec[6].v, iter_max_vec[2].v );
            iter_max_vec[3].v = _mm512_max_pd( abs_x_vec[7].v, iter_max_vec[3].v );

            /* Further computing the maximum(in order) based on the previous results */
            iter_max_vec[0].v = _mm512_max_pd( iter_max_vec[0].v, iter_max_vec[1].v );
            iter_max_vec[2].v = _mm512_max_pd( iter_max_vec[2].v, iter_max_vec[3].v );

            iter_max_vec[0].v = _mm512_max_pd( iter_max_vec[0].v, iter_max_vec[2].v );

            /* Obtaining the maximum value as a scalar, using reduction */
            iter_max_scalar = _mm512_reduce_max_pd( iter_max_vec[0].v );

            /* Updating the indices and base_max_scalar, if needed */
            if( iter_max_scalar > base_max_scalar )
            {
                base_max_scalar = iter_max_scalar;
                start = i;
                end = i + 63;
            }

            /* Incrementing the pointer based on unrolling */
            x0 += 64;
        }
        for( ; i + 31 < n; i += 32 )
        {
            /* Loading vector */
            x_vec[0].v = _mm512_loadu_pd(( const double* )( x0 ));
            x_vec[1].v = _mm512_loadu_pd(( const double* )( x0 + 8 ));
            x_vec[2].v = _mm512_loadu_pd(( const double* )( x0 + 16 ));
            x_vec[3].v = _mm512_loadu_pd(( const double* )( x0 + 24 ));

            /* Getting the abs value, using sign_mask */
            abs_x_vec[0].v = _mm512_andnot_pd( sign_mask.v, x_vec[0].v );
            abs_x_vec[1].v = _mm512_andnot_pd( sign_mask.v, x_vec[1].v );
            abs_x_vec[2].v = _mm512_andnot_pd( sign_mask.v, x_vec[2].v );
            abs_x_vec[3].v = _mm512_andnot_pd( sign_mask.v, x_vec[3].v );

            /* Calculating the maximum value amongst the loaded vectors */
            iter_max_vec[0].v = _mm512_max_pd( abs_x_vec[0].v, iter_max_vec[0].v );
            iter_max_vec[1].v = _mm512_max_pd( abs_x_vec[1].v, iter_max_vec[1].v );
            iter_max_vec[0].v = _mm512_max_pd( abs_x_vec[2].v, iter_max_vec[0].v );
            iter_max_vec[1].v = _mm512_max_pd( abs_x_vec[3].v, iter_max_vec[1].v );

            iter_max_vec[0].v = _mm512_max_pd( iter_max_vec[0].v, iter_max_vec[1].v );

            /* Obtaining the maximum value as a scalar, using reduction */
            iter_max_scalar = _mm512_reduce_max_pd( iter_max_vec[0].v );

            /* Updating the indices and base_max_scalar, if needed */
            if( iter_max_scalar > base_max_scalar )
            {
                base_max_scalar = iter_max_scalar;
                start = i;
                end = i + 31;
            }

            /* Incrementing the pointer based on unrolling */
            x0 += 32;
        }
        for( ; i + 15 < n; i += 16 )
        {
            /* Loading vector */
            x_vec[0].v = _mm512_loadu_pd(( const double* )( x0 ));
            x_vec[1].v = _mm512_loadu_pd(( const double* )( x0 + 8 ));

            /* Getting the abs value, using sign_mask */
            abs_x_vec[0].v = _mm512_andnot_pd( sign_mask.v, x_vec[0].v );
            abs_x_vec[1].v = _mm512_andnot_pd( sign_mask.v, x_vec[1].v );

            /* Calculating the maximum value amongst the loaded vectors */
            iter_max_vec[0].v = _mm512_max_pd( abs_x_vec[0].v, iter_max_vec[0].v );
            iter_max_vec[0].v = _mm512_max_pd( abs_x_vec[1].v, iter_max_vec[0].v );

            /* Obtaining the maximum value as a scalar, using reduction */
            iter_max_scalar = _mm512_reduce_max_pd( iter_max_vec[0].v );

            /* Updating the indices and base_max_scalar, if needed */
            if( iter_max_scalar > base_max_scalar )
            {
                base_max_scalar = iter_max_scalar;
                start = i;
                end = i + 15;
            }

            /* Incrementing the pointer based on unrolling */
            x0 += 16;
        }
        for( ; i + 7 < n; i += 8 )
        {
            /* Loading vector */
            x_vec[0].v = _mm512_loadu_pd(( const double* )( x0 ));

            /* Getting abs value */
            abs_x_vec[0].v = _mm512_andnot_pd( sign_mask.v, x_vec[0].v );

            /* Calculating the maximum value amongst the loaded vectors */
            iter_max_vec[0].v = _mm512_max_pd( abs_x_vec[0].v, iter_max_vec[0].v );

            /* Obtaining the maximum value as a scalar, using reduction */
            iter_max_scalar = _mm512_reduce_max_pd( iter_max_vec[0].v );

            /* Updating the indices and base_max_scalar, if needed */
            if( iter_max_scalar > base_max_scalar )
            {
                base_max_scalar = iter_max_scalar;
                start = i;
                end = i + 7;
            }
            /* Incrementing the pointer based on unrolling */
            x0 += 8;
        }
        if( i < n )
        {
            /* Setting the mask for the fringe case */
            __mmask8 n_mask = ( 1 << ( n - i ) ) - 1;

            /* Loading vector */
            x_vec[0].v = _mm512_maskz_loadu_pd( n_mask, ( const double* )( x0 ) );

            /* Getting the abs value */
            abs_x_vec[0].v = _mm512_andnot_pd( sign_mask.v, x_vec[0].v );

            /* Calculating the maximum value amongst the loaded vectors */
            iter_max_vec[0].v = _mm512_max_pd( abs_x_vec[0].v, iter_max_vec[0].v );

            /* Obtaining the maximum value as a scalar, using reduction */
            iter_max_scalar = _mm512_reduce_max_pd( iter_max_vec[0].v );

            /* Updating the indices and base_max_scalar, if needed */
            if( iter_max_scalar > base_max_scalar )
            {
                base_max_scalar = iter_max_scalar;
                start = i;
                end = n - 1;
            }
        }

        /* Reduction to find the first occurence of absolute maximum */
        /* Update the pointer based on new 'start' value */
        x0 = x + start;
        /* Search for the index in the final search space */
        while( start < end )
        {
            /* In case we encounter the abs max value, terminate the loop */
            if( fabs( *x0 ) == base_max_scalar )
                break;

            /* Update the pointer and the offset */
            x0 += 1;
            start += 1;
        }

        /* Set the actual parameter with the index */
        *i_max = start;
    }

    else
    {
        double chi1, abs_chi1, abs_chi1_max;
        dim_t i_max_l = 0;
        abs_chi1_max = -1.0;
        for (dim_t i = 0; i < n; ++i)
        {
            chi1 = *( x + ( i ) * incx );

            /* Assign abs_chi1 with its absolute value. */
            abs_chi1 = fabs(chi1);

            /* If the absolute value of the current element exceeds that of
               the previous largest, save it and its index. */
            if ( abs_chi1_max < abs_chi1 )
            {
                abs_chi1_max = abs_chi1;
                i_max_l = i;
            }
        }

        /* Set the actual parameter with the index */
        *i_max = i_max_l;
    }

    AOCL_DTL_TRACE_EXIT(AOCL_DTL_LEVEL_TRACE_3)
}
