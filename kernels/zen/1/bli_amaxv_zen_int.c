/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2016 - 2021, Advanced Micro Devices, Inc.
   Copyright (C) 2018, The University of Texas at Austin

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
   One 512-bit AVX register holds 8 DP elements. */
typedef union
{
	__m512d v;
	double d[8] __attribute__((aligned(64)));
} v8df_t;


/* Union data structure to access AVX registers
   One 256-bit AVX register holds 8 SP elements. */
typedef union
{
	__m256  v;
	float   f[8] __attribute__((aligned(64)));
} v8sf_t;

typedef union
{
	__m128  v;
	float   f[4];
} v4sf_t;

/* Union data structure to access AVX registers
   One 256-bit AVX register holds 4 DP elements. */
typedef union
{
	__m256d v;
	double  d[4] __attribute__((aligned(64)));
}v4df_t;

typedef union
{
	__m128d v;
	double  d[2];
}v2dd_t;

// return a mask which indicates either:
// - v1 > v2
// - v1 is NaN and v2 is not
// assumes that idx(v1) > idx(v2)
// all "OQ" comparisons false if either operand NaN
#define CMP256( dt, v1, v2 ) \
	_mm256_or_p##dt( _mm256_cmp_p##dt( v1, v2, _CMP_GT_OQ ),                        /* v1 > v2  ||     */ \
	                 _mm256_andnot_p##dt( _mm256_cmp_p##dt( v2, v2, _CMP_UNORD_Q ), /* ( !isnan(v2) && */ \
	                                      _mm256_cmp_p##dt( v1, v1, _CMP_UNORD_Q )  /*    isnan(v1) )  */ \
	                                    ) \
	               );

// return a mask which indicates either:
// - v1 > v2
// - v1 is NaN and v2 is not
// - v1 == v2 (maybe == NaN) and i1 < i2
// all "OQ" comparisons false if either operand NaN
#define CMP128( dt, v1, v2, i1, i2 ) \
	_mm_or_p##dt( _mm_or_p##dt( _mm_cmp_p##dt( v1, v2, _CMP_GT_OQ ),                      /* ( v1 > v2 ||           */ \
	                            _mm_andnot_p##dt( _mm_cmp_p##dt( v2, v2, _CMP_UNORD_Q ),  /*   ( !isnan(v2) &&      */ \
	                                              _mm_cmp_p##dt( v1, v1, _CMP_UNORD_Q )   /*      isnan(v1) ) ) ||  */ \
	                                            ) \
	                          ), \
	              _mm_and_p##dt( _mm_or_p##dt( _mm_cmp_p##dt( v1, v2, _CMP_EQ_OQ ),                  /* ( ( v1 == v2 ||        */ \
	                                           _mm_and_p##dt( _mm_cmp_p##dt( v1, v1, _CMP_UNORD_Q ), /*     ( isnan(v1) &&     */ \
	                                                          _mm_cmp_p##dt( v2, v2, _CMP_UNORD_Q )  /*       isnan(v2) ) ) && */ \
	                                                        ) \
	                                         ), \
	                             _mm_cmp_p##dt( i1, i2, _CMP_LT_OQ )                                 /*   i1 < i2 )            */ \
	                           ) \
	            );

// -----------------------------------------------------------------------------

void bli_samaxv_zen_int
     (
       dim_t            n,
       float*  restrict x, inc_t incx,
       dim_t*  restrict i_max,
       cntx_t* restrict cntx
     )
{
    AOCL_DTL_TRACE_ENTRY(AOCL_DTL_LEVEL_TRACE_3)
	float*  minus_one = PASTEMAC(s,m1);
	dim_t*  zero_i    = PASTEMAC(i,0);

	float   chi1_r;
	//float   chi1_i;
	float   abs_chi1;
	float   abs_chi1_max;
	dim_t   i_max_l;
	dim_t   i;

	/* If the vector length is zero, return early. This directly emulates
	   the behavior of netlib BLAS's i?amax() routines. */
	if ( bli_zero_dim1( n ) )
	{
		PASTEMAC(i,copys)( *zero_i, *i_max );
        AOCL_DTL_TRACE_EXIT(AOCL_DTL_LEVEL_TRACE_3)
		return;
	}

	/* Initialize the index of the maximum absolute value to zero. */
	PASTEMAC(i,copys)( *zero_i, i_max_l );

	/* Initialize the maximum absolute value search candidate with
	   -1, which is guaranteed to be less than all values we will
	   compute. */
	PASTEMAC(s,copys)( *minus_one, abs_chi1_max );

	// For non-unit strides, or very small vector lengths, compute with
	// scalar code.
	if ( incx != 1 || n < 8 )
	{
		for ( i = 0; i < n; ++i )
		{
			float* chi1 = x + (i  )*incx;

			/* Get the real and imaginary components of chi1. */
			chi1_r = *chi1;

			/* Replace chi1_r and chi1_i with their absolute values. */
			chi1_r = fabsf( chi1_r );

			/* Add the real and imaginary absolute values together. */
			abs_chi1 = chi1_r;

			/* If the absolute value of the current element exceeds that of
			   the previous largest, save it and its index. If NaN is
			   encountered, then treat it the same as if it were a valid
			   value that was smaller than any previously seen. This
			   behavior mimics that of LAPACK's i?amax(). */
			if ( abs_chi1_max < abs_chi1 || ( isnan( abs_chi1 ) && !isnan( abs_chi1_max ) ) )
			{
				abs_chi1_max = abs_chi1;
				i_max_l      = i;
			}
		}
	}
	else
	{
		dim_t  n_iter, n_left;
		dim_t  num_vec_elements = 8;
		v8sf_t x_vec, max_vec, maxInx_vec, mask_vec;
		v8sf_t idx_vec, inc_vec;
		v8sf_t sign_mask;

		v4sf_t max_vec_lo, max_vec_hi, mask_vec_lo;
		v4sf_t maxInx_vec_lo, maxInx_vec_hi;

		n_iter = n / num_vec_elements;
		n_left = n % num_vec_elements;

		idx_vec.v    = _mm256_set_ps( 7, 6, 5, 4, 3, 2, 1, 0 );
		inc_vec.v    = _mm256_set1_ps( 8 );
		max_vec.v    = _mm256_set1_ps( -1 );
		maxInx_vec.v = _mm256_setzero_ps();
		sign_mask.v  = _mm256_set1_ps( -0.f );

		for ( i = 0; i < n_iter; ++i )
		{
			x_vec.v      = _mm256_loadu_ps( x );

			// Get the absolute value of the vector element.
			x_vec.v      = _mm256_andnot_ps( sign_mask.v, x_vec.v );

			mask_vec.v   = CMP256( s, x_vec.v, max_vec.v );

			max_vec.v    = _mm256_blendv_ps( max_vec.v, x_vec.v, mask_vec.v );
			maxInx_vec.v = _mm256_blendv_ps( maxInx_vec.v, idx_vec.v, mask_vec.v );

			idx_vec.v += inc_vec.v;
			x         += num_vec_elements;
		}

		max_vec_lo.v    = _mm256_extractf128_ps( max_vec.v, 0 );
		max_vec_hi.v    = _mm256_extractf128_ps( max_vec.v, 1 );
		maxInx_vec_lo.v = _mm256_extractf128_ps( maxInx_vec.v, 0 );
		maxInx_vec_hi.v = _mm256_extractf128_ps( maxInx_vec.v, 1 );
		
		mask_vec_lo.v = CMP128( s, max_vec_hi.v, max_vec_lo.v, maxInx_vec_hi.v, maxInx_vec_lo.v );

		max_vec_lo.v    = _mm_blendv_ps( max_vec_lo.v, max_vec_hi.v, mask_vec_lo.v );
		maxInx_vec_lo.v = _mm_blendv_ps( maxInx_vec_lo.v, maxInx_vec_hi.v, mask_vec_lo.v );

		max_vec_hi.v    = _mm_permute_ps( max_vec_lo.v, 14 );
		maxInx_vec_hi.v = _mm_permute_ps( maxInx_vec_lo.v, 14 );
		
		mask_vec_lo.v = CMP128( s, max_vec_hi.v, max_vec_lo.v, maxInx_vec_hi.v, maxInx_vec_lo.v );

		max_vec_lo.v    = _mm_blendv_ps( max_vec_lo.v, max_vec_hi.v, mask_vec_lo.v );
		maxInx_vec_lo.v = _mm_blendv_ps( maxInx_vec_lo.v, maxInx_vec_hi.v, mask_vec_lo.v );

		max_vec_hi.v    = _mm_permute_ps( max_vec_lo.v, 1 );
		maxInx_vec_hi.v = _mm_permute_ps( maxInx_vec_lo.v, 1 );
		
		mask_vec_lo.v = CMP128( s, max_vec_hi.v, max_vec_lo.v, maxInx_vec_hi.v, maxInx_vec_lo.v );

		max_vec_lo.v    = _mm_blendv_ps( max_vec_lo.v, max_vec_hi.v, mask_vec_lo.v );
		maxInx_vec_lo.v = _mm_blendv_ps( maxInx_vec_lo.v, maxInx_vec_hi.v, mask_vec_lo.v );

		abs_chi1_max = max_vec_lo.f[0];
		i_max_l      = maxInx_vec_lo.f[0];

		for ( i = n - n_left; i < n; i++ )
		{
			float* chi1 = x;

			/* Get the real and imaginary components of chi1. */
			chi1_r = *chi1;

			/* Replace chi1_r and chi1_i with their absolute values. */
			abs_chi1 = fabsf( chi1_r );

			/* If the absolute value of the current element exceeds that of
			   the previous largest, save it and its index. If NaN is
			   encountered, then treat it the same as if it were a valid
			   value that was smaller than any previously seen. This
			   behavior mimics that of LAPACK's i?amax(). */
			if ( abs_chi1_max < abs_chi1 || ( isnan( abs_chi1 ) && !isnan( abs_chi1_max ) ) )
			{
				abs_chi1_max = abs_chi1;
				i_max_l      = i;
			}

			x += 1;
		}
	}

	// Issue vzeroupper instruction to clear upper lanes of ymm registers.
	// This avoids a performance penalty caused by false dependencies when
	// transitioning from from AVX to SSE instructions (which may occur
	// later, especially if BLIS is compiled with -mfpmath=sse).
	_mm256_zeroupper();

	/* Store final index to output variable. */
	*i_max = i_max_l;
    AOCL_DTL_TRACE_EXIT(AOCL_DTL_LEVEL_TRACE_3)
}

// -----------------------------------------------------------------------------

void bli_damaxv_zen_int
     (
       dim_t            n,
       double* restrict x, inc_t incx,
       dim_t*  restrict i_max,
       cntx_t* restrict cntx
     )
{
    AOCL_DTL_TRACE_EXIT(AOCL_DTL_LEVEL_TRACE_3)
	double* minus_one = PASTEMAC(d,m1);
	dim_t*  zero_i    = PASTEMAC(i,0);

	double  chi1_r;
	//double  chi1_i;
	double  abs_chi1;
	double  abs_chi1_max;
	dim_t   i_max_l;
	dim_t   i;

	/* If the vector length is zero, return early. This directly emulates
	   the behavior of netlib BLAS's i?amax() routines. */
	if ( bli_zero_dim1( n ) )
	{
		PASTEMAC(i,copys)( *zero_i, *i_max );
        AOCL_DTL_TRACE_EXIT(AOCL_DTL_LEVEL_TRACE_3)
		return;
	}

	/* Initialize the index of the maximum absolute value to zero. */ \
	PASTEMAC(i,copys)( *zero_i, i_max_l );

	/* Initialize the maximum absolute value search candidate with
	   -1, which is guaranteed to be less than all values we will
	   compute. */
	PASTEMAC(d,copys)( *minus_one, abs_chi1_max );

	// For non-unit strides, or very small vector lengths, compute with
	// scalar code.
	if ( incx != 1 || n < 4 )
	{
		for ( i = 0; i < n; ++i )
		{
			double* chi1 = x + (i  )*incx;

			/* Get the real and imaginary components of chi1. */
			chi1_r = *chi1;

			/* Replace chi1_r and chi1_i with their absolute values. */
			chi1_r = fabs( chi1_r );

			/* Add the real and imaginary absolute values together. */
			abs_chi1 = chi1_r;

			/* If the absolute value of the current element exceeds that of
			   the previous largest, save it and its index. If NaN is
			   encountered, then treat it the same as if it were a valid
			   value that was smaller than any previously seen. This
			   behavior mimics that of LAPACK's i?amax(). */
			if ( abs_chi1_max < abs_chi1 || ( isnan( abs_chi1 ) && !isnan( abs_chi1_max ) ) )
			{
				abs_chi1_max = abs_chi1;
				i_max_l      = i;
			}
		}
	}
	else
	{
		dim_t  n_iter, n_left;
		dim_t  num_vec_elements = 4;
		v4df_t x_vec, max_vec, maxInx_vec, mask_vec;
		v4df_t idx_vec, inc_vec;
		v4df_t sign_mask;

		v2dd_t max_vec_lo, max_vec_hi, mask_vec_lo;
		v2dd_t maxInx_vec_lo, maxInx_vec_hi;

		n_iter = n / num_vec_elements;
		n_left = n % num_vec_elements;

		idx_vec.v    = _mm256_set_pd( 3, 2, 1, 0 );
		inc_vec.v    = _mm256_set1_pd( 4 );
		max_vec.v    = _mm256_set1_pd( -1 );
		maxInx_vec.v = _mm256_setzero_pd();
		sign_mask.v  = _mm256_set1_pd( -0.f );

		for ( i = 0; i < n_iter; ++i )
		{
			x_vec.v      = _mm256_loadu_pd( x );

			// Get the absolute value of the vector element.
			x_vec.v      = _mm256_andnot_pd( sign_mask.v, x_vec.v );

			mask_vec.v   = CMP256( d, x_vec.v, max_vec.v );

			max_vec.v    = _mm256_blendv_pd( max_vec.v, x_vec.v, mask_vec.v );
			maxInx_vec.v = _mm256_blendv_pd( maxInx_vec.v, idx_vec.v, mask_vec.v );

			idx_vec.v += inc_vec.v;
			x         += num_vec_elements;
		}

		max_vec_lo.v    = _mm256_extractf128_pd( max_vec.v, 0 );
		max_vec_hi.v    = _mm256_extractf128_pd( max_vec.v, 1 );
		maxInx_vec_lo.v = _mm256_extractf128_pd( maxInx_vec.v, 0 );
		maxInx_vec_hi.v = _mm256_extractf128_pd( maxInx_vec.v, 1 );
		
		mask_vec_lo.v = CMP128( d, max_vec_hi.v, max_vec_lo.v, maxInx_vec_hi.v, maxInx_vec_lo.v );

		max_vec_lo.v    = _mm_blendv_pd( max_vec_lo.v, max_vec_hi.v, mask_vec_lo.v );
		maxInx_vec_lo.v = _mm_blendv_pd( maxInx_vec_lo.v, maxInx_vec_hi.v, mask_vec_lo.v );
		
		max_vec_hi.v    = _mm_permute_pd( max_vec_lo.v, 1 );
		maxInx_vec_hi.v = _mm_permute_pd( maxInx_vec_lo.v, 1 );
		
		mask_vec_lo.v = CMP128( d, max_vec_hi.v, max_vec_lo.v, maxInx_vec_hi.v, maxInx_vec_lo.v );

		max_vec_lo.v    = _mm_blendv_pd( max_vec_lo.v, max_vec_hi.v, mask_vec_lo.v );
		maxInx_vec_lo.v = _mm_blendv_pd( maxInx_vec_lo.v, maxInx_vec_hi.v, mask_vec_lo.v );

		abs_chi1_max = max_vec_lo.d[0];
		i_max_l      = maxInx_vec_lo.d[0];

		for ( i = n - n_left; i < n; i++ )
		{
			double* chi1 = x;

			/* Get the real and imaginary components of chi1. */
			chi1_r = *chi1;

			/* Replace chi1_r and chi1_i with their absolute values. */
			abs_chi1 = fabs( chi1_r );

			/* If the absolute value of the current element exceeds that of
			   the previous largest, save it and its index. If NaN is
			   encountered, return the index of the first NaN. This
			   behavior mimics that of LAPACK's i?amax(). */
			if ( abs_chi1_max < abs_chi1 || ( isnan( abs_chi1 ) && !isnan( abs_chi1_max ) ) )
			{
				abs_chi1_max = abs_chi1;
				i_max_l      = i;
			}

			x += 1;
		}
	}

	// Issue vzeroupper instruction to clear upper lanes of ymm registers.
	// This avoids a performance penalty caused by false dependencies when
	// transitioning from from AVX to SSE instructions (which may occur
	// later, especially if BLIS is compiled with -mfpmath=sse).
	_mm256_zeroupper();

	/* Store final index to output variable. */
	*i_max = i_max_l;
    AOCL_DTL_TRACE_EXIT(AOCL_DTL_LEVEL_TRACE_3)
}

// -----------------------------------------------------------------------------

#if 0
#undef  GENTFUNCR
#define GENTFUNCR( ctype, ctype_r, ch, chr, varname ) \
\
void PASTEMAC(ch,varname) \
     ( \
       dim_t    n, \
       ctype*   x, inc_t incx, \
       dim_t*   i_max, \
       cntx_t*  cntx  \
     ) \
{ \
	ctype_r* minus_one = PASTEMAC(chr,m1); \
	dim_t*   zero_i    = PASTEMAC(i,0); \
\
	ctype_r  chi1_r; \
	ctype_r  chi1_i; \
	ctype_r  abs_chi1; \
	ctype_r  abs_chi1_max; \
	dim_t    i; \
\
	/* Initialize the index of the maximum absolute value to zero. */ \
	PASTEMAC(i,copys)( zero_i, *i_max ); \
\
	/* If the vector length is zero, return early. This directly emulates
	   the behavior of netlib BLAS's i?amax() routines. */ \
	if ( bli_zero_dim1( n ) ) return; \
\
	/* Initialize the maximum absolute value search candidate with
	   -1, which is guaranteed to be less than all values we will
	   compute. */ \
	PASTEMAC(chr,copys)( *minus_one, abs_chi1_max ); \
\
	if ( incx == 1 ) \
	{ \
		for ( i = 0; i < n; ++i ) \
		{ \
			/* Get the real and imaginary components of chi1. */ \
			PASTEMAC2(ch,chr,gets)( x[i], chi1_r, chi1_i ); \
\
			/* Replace chi1_r and chi1_i with their absolute values. */ \
			PASTEMAC(chr,abval2s)( chi1_r, chi1_r ); \
			PASTEMAC(chr,abval2s)( chi1_i, chi1_i ); \
\
			/* Add the real and imaginary absolute values together. */ \
			PASTEMAC(chr,set0s)( abs_chi1 ); \
			PASTEMAC(chr,adds)( chi1_r, abs_chi1 ); \
			PASTEMAC(chr,adds)( chi1_i, abs_chi1 ); \
\
			/* If the absolute value of the current element exceeds that of
			   the previous largest, save it and its index. If NaN is
			   encountered, then treat it the same as if it were a valid
			   value that was smaller than any previously seen. This
			   behavior mimics that of LAPACK's ?lange(). */ \
			if ( abs_chi1_max < abs_chi1 || bli_isnan( abs_chi1 ) ) \
			{ \
				abs_chi1_max = abs_chi1; \
				*i_max       = i; \
			} \
		} \
	} \
	else \
	{ \
		for ( i = 0; i < n; ++i ) \
		{ \
			ctype* chi1 = x + (i  )*incx; \
\
			/* Get the real and imaginary components of chi1. */ \
			PASTEMAC2(ch,chr,gets)( *chi1, chi1_r, chi1_i ); \
\
			/* Replace chi1_r and chi1_i with their absolute values. */ \
			PASTEMAC(chr,abval2s)( chi1_r, chi1_r ); \
			PASTEMAC(chr,abval2s)( chi1_i, chi1_i ); \
\
			/* Add the real and imaginary absolute values together. */ \
			PASTEMAC(chr,set0s)( abs_chi1 ); \
			PASTEMAC(chr,adds)( chi1_r, abs_chi1 ); \
			PASTEMAC(chr,adds)( chi1_i, abs_chi1 ); \
\
			/* If the absolute value of the current element exceeds that of
			   the previous largest, save it and its index. If NaN is
			   encountered, then treat it the same as if it were a valid
			   value that was smaller than any previously seen. This
			   behavior mimics that of LAPACK's ?lange(). */ \
			if ( abs_chi1_max < abs_chi1 || bli_isnan( abs_chi1 ) ) \
			{ \
				abs_chi1_max = abs_chi1; \
				*i_max       = i; \
			} \
		} \
	} \
}
GENTFUNCR( scomplex, float,  c, s, amaxv_zen_int )
GENTFUNCR( dcomplex, double, z, d, amaxv_zen_int )
#endif


/* Converts all the NAN to a negative number less than previously encountered NANs*/
__m512d remove_NAN_512d(__m512d vec)
{

	static int iter;
	static __m512d sign_mask;

	__m512d vec_mask;
	__m512i int_mask_vec;
	__mmask8 vec_mask8;

	iter = iter - 1;

	sign_mask = _mm512_set1_pd(-0.f);

	//numbers other than NAN will become 0
	vec_mask = _mm512_mul_pd(vec, sign_mask);

	//producing an 8-bit mask
	int_mask_vec = _mm512_castpd_si512(vec_mask);
	vec_mask8 = _mm512_movepi64_mask(int_mask_vec);

	//replacing all the NAN with negative numbers
	vec = _mm512_mask_blend_pd(vec_mask8, _mm512_set1_pd(-1 + iter), vec);

	return vec;
}

//----------------------------------------------------------------------------------------------------

void bli_damaxv_zen_int_avx512(
	dim_t n,
	double *restrict x, inc_t incx,
	dim_t *restrict i_max,
	cntx_t *restrict cntx)
{
	AOCL_DTL_TRACE_EXIT(AOCL_DTL_LEVEL_TRACE_3)
	double *minus_one = PASTEMAC(d, m1);
	dim_t *zero_i = PASTEMAC(i, 0);

	double chi1_r;
	//double  chi1_i;
	double abs_chi1;
	double abs_chi1_max;
	dim_t i_max_l;
	dim_t i;

	/* If the vector length is zero, return early. This directly emulates
	   the behavior of netlib BLAS's i?amax() routines. */
	if (bli_zero_dim1(n))
	{
		PASTEMAC(i, copys)
		(*zero_i, *i_max);
		AOCL_DTL_TRACE_EXIT(AOCL_DTL_LEVEL_TRACE_3)
		return;
	}

	/* Initialize the index of the maximum absolute value to zero. */
	PASTEMAC(i, copys)
	(*zero_i, i_max_l);

	/* Initialize the maximum absolute value search candidate with
	   -1, which is guaranteed to be less than all values we will
	   compute. */
	PASTEMAC(d, copys)
	(*minus_one, abs_chi1_max);

	// For non-unit strides, or very small vector lengths, compute with
	// scalar code.
	if (incx != 1 || n < 8)
	{
		for (i = 0; i < n; ++i)
		{
			double *chi1 = x + (i)*incx;

			/* Get the real and imaginary components of chi1. */
			chi1_r = *chi1;

			/* Replace chi1_r and chi1_i with their absolute values. */
			chi1_r = fabs(chi1_r);

			/* Add the real and imaginary absolute values together. */
			abs_chi1 = chi1_r;

			/* If the absolute value of the current element exceeds that of
			   the previous largest, save it and its index. If NaN is
			   encountered, then treat it the same as if it were a valid
			   value that was smaller than any previously seen. This
			   behavior mimics that of LAPACK's i?amax(). */
			if (abs_chi1_max < abs_chi1 || (isnan(abs_chi1) && !isnan(abs_chi1_max)))
			{
				abs_chi1_max = abs_chi1;
				i_max_l = i;
			}
		}
	}
	else
	{

		dim_t iterations, n_left, vector_length = 8, unrollCount = 0;

		//mask bits
		__mmask8 mask_got_01, mask_got_23;

		//YMM0 - YMM6 registers
		v4df_t max_hi, max_lo, max_ind_hi, max_ind_lo,
				mask_final, inter_result, inter_ind;

		//XMM0 to XMM4 registers
		v2dd_t max_vec_hi, max_vec_lo, max_ind_hi_128,
				max_ind_lo_128, mask_vec_lo;

		//ZMM0 to ZMM13 registers
		v8df_t zmm0, zmm1, zmm2, zmm3, zmm4_Ind,
			zmm5_Ind, zmm6_Ind, zmm7_Ind, max_01,
			max_23, final_max, max_array, max_ind, inc_vec;

		//ZMM14 to ZMM16 registers
		__m512d mask_01, mask_23, sign_mask;

		//Intermediate int mask values
		__m512i int_mask_01, int_mask_23;

		// Initialize sign mask
		sign_mask = _mm512_set1_pd(-0.f);

		//Initializing the indexes of the base case of max vector
		zmm4_Ind.v = _mm512_set_pd(7, 6, 5, 4, 3, 2, 1, 0);
		inc_vec.v = _mm512_set1_pd(8); //Vector for incrementing

		// Initializing the max array as vec [ 0 : 512 ]
		max_array.v = _mm512_loadu_pd(x);

		// Taking the absolute value and removing the NAN
		max_array.v = _mm512_andnot_pd(sign_mask, max_array.v);
		max_array.v = remove_NAN_512d(max_array.v);

		// Initializing the maximumum index
		max_ind.v = _mm512_set_pd(7, 6, 5, 4, 3, 2, 1, 0);
		x += vector_length;

		//Incrementing to make the vector
		//to point to the next 8 elements
		zmm4_Ind.v = _mm512_add_pd(zmm4_Ind.v, inc_vec.v);

		/*	Loop unrolled by a factor of 4
			At the end of the loop max_array holds the largest element
			in each corresponding vector index */
		for (unrollCount = 8; (unrollCount + 31) < n; unrollCount += 32)
		{
			// Taking 32 elements
			// Taking only the absolute values of the registers
			// Removing the NAN values and replacing it
			// with negative numbers
			zmm0.v = _mm512_loadu_pd(x);
			zmm0.v = _mm512_andnot_pd(sign_mask, zmm0.v);
			zmm0.v = remove_NAN_512d(zmm0.v);
			x += vector_length;

			zmm1.v = _mm512_loadu_pd(x);
			zmm5_Ind.v = _mm512_add_pd(zmm4_Ind.v, inc_vec.v);
			zmm1.v = _mm512_andnot_pd(sign_mask, zmm1.v);
			zmm1.v = remove_NAN_512d(zmm1.v);
			x += vector_length;

			zmm2.v = _mm512_loadu_pd(x);
			zmm6_Ind.v = _mm512_add_pd(zmm5_Ind.v, inc_vec.v);
			zmm2.v = _mm512_andnot_pd(sign_mask, zmm2.v);
			zmm2.v = remove_NAN_512d(zmm2.v);
			x += vector_length;

			zmm3.v = _mm512_loadu_pd(x);
			zmm7_Ind.v = _mm512_add_pd(zmm6_Ind.v, inc_vec.v);
			zmm3.v = _mm512_andnot_pd(sign_mask, zmm3.v);
			zmm3.v = remove_NAN_512d(zmm3.v);
			x += vector_length;

			/*Using sub function to generating the mask
				as a 512d type*/
			mask_01 = _mm512_sub_pd(zmm0.v, zmm1.v);
			mask_23 = _mm512_sub_pd(zmm2.v, zmm3.v);

			//Converting the 512d mask to a 512i mask
			int_mask_01 = _mm512_castpd_si512(mask_01);
			int_mask_23 = _mm512_castpd_si512(mask_23);

			/*Converting the 512i mask
			to mmask type to use the mask bits*/
			mask_got_01 = _mm512_movepi64_mask(int_mask_01);
			mask_got_23 = _mm512_movepi64_mask(int_mask_23);

			//Storing the largest elements in index % 8 position for
			//vector 1 and 2, and the index of the corresponding element
			max_01.v = _mm512_mask_blend_pd(mask_got_01, zmm0.v, zmm1.v);
			zmm5_Ind.v = _mm512_mask_blend_pd(mask_got_01, zmm4_Ind.v, zmm5_Ind.v);

			//Storing the largest elements in index % 8 position for
			//vector 3 and 4, and the index of the corresponding element
			max_23.v = _mm512_mask_blend_pd(mask_got_23, zmm2.v, zmm3.v);
			zmm6_Ind.v = _mm512_mask_blend_pd(mask_got_23, zmm6_Ind.v, zmm7_Ind.v);

			//Generating mask for the intermediate max vector
			mask_01 = _mm512_sub_pd(max_01.v, max_23.v);
			int_mask_01 = _mm512_castpd_si512(mask_01);
			mask_got_01 = _mm512_movepi64_mask(int_mask_01);

			/*Storing the largest elements in index % 8 position for
			the intermediate max vectors,
			and the index of the corresponding element*/
			final_max.v = _mm512_mask_blend_pd(mask_got_01, max_01.v, max_23.v);
			zmm5_Ind.v = _mm512_mask_blend_pd(mask_got_01, zmm5_Ind.v, zmm6_Ind.v);

			//Generating the mask for final max vector and base max vector
			mask_01 = _mm512_sub_pd(max_array.v, final_max.v);
			int_mask_01 = _mm512_castpd_si512(mask_01);
			mask_got_01 = _mm512_movepi64_mask(int_mask_01);

			// Result is the maximum of all index % 8 locations
			max_array.v = _mm512_mask_blend_pd(mask_got_01, max_array.v, final_max.v);
			max_ind.v = _mm512_mask_blend_pd(mask_got_01, max_ind.v, zmm5_Ind.v);

			// Incrementing the index to point to the next 8 locations
			zmm4_Ind.v = _mm512_add_pd(zmm7_Ind.v, inc_vec.v);
		}

		// Calculating the number of iterations left
		iterations = (n - unrollCount) / vector_length;
		n_left = (n - unrollCount) % vector_length;

		/* At the end of the loop max_array holds the largest element
		 in each corresponding vector index */
		for (int i = 1; i < iterations; ++i)
		{
			// Taking 32 elements
			// Taking only the absolute values of the registers
			// Removing the NAN values and replacing it
			// with negative numbers
			zmm0.v = _mm512_loadu_pd(x);
			zmm0.v = _mm512_abs_pd(zmm0.v);
			zmm0.v = remove_NAN_512d(zmm0.v);

			//Generating mask for the intermediate max vector
			mask_01 = _mm512_sub_pd(max_array.v, zmm0.v);
			int_mask_01 = _mm512_castpd_si512(mask_01);
			mask_got_01 = _mm512_movepi64_mask(int_mask_01);

			// Result is the maximum of all index % 8 locations
			max_array.v = _mm512_mask_blend_pd(mask_got_01, max_array.v, zmm0.v);

			//Storing the index of the corresponding max array elemets
			max_ind.v = _mm512_mask_blend_pd(mask_got_01, max_ind.v, zmm4_Ind.v);

			//Incrementing the vector the point to the next location
			//Incrementing the vector indexes
			x += vector_length;
			zmm4_Ind.v = _mm512_add_pd(zmm4_Ind.v, inc_vec.v);
		}

		//Breaking max array into vectors of length 4
		//Taking upper and lower halves
		max_hi.v = _mm512_extractf64x4_pd(max_array.v, 1);
		max_ind_hi.v = _mm512_extractf64x4_pd(max_ind.v, 1);
		max_lo.v = _mm512_extractf64x4_pd(max_array.v, 0);
		max_ind_lo.v = _mm512_extractf64x4_pd(max_ind.v, 0);

		//Generating the mask for blending
		mask_final.v = _mm256_sub_pd(max_hi.v, max_lo.v);

		// Storing the max of max array index % 4
		inter_result.v = _mm256_blendv_pd(max_hi.v, max_lo.v, mask_final.v);
		inter_ind.v = _mm256_blendv_pd(max_ind_hi.v, max_ind_lo.v, mask_final.v);

		//Breaking max array into vectors of length 2
		max_vec_lo.v = _mm256_extractf128_pd(inter_result.v, 0);
		max_vec_hi.v = _mm256_extractf128_pd(inter_result.v, 1);
		max_ind_hi_128.v = _mm256_extractf128_pd(inter_ind.v, 1);
		max_ind_lo_128.v = _mm256_extractf128_pd(inter_ind.v, 0);

		//Generating the mask for blending
		mask_vec_lo.v = _mm_sub_pd(max_vec_lo.v, max_vec_hi.v);

		// Storing the max of max array index % 2
		max_vec_lo.v = _mm_blendv_pd(max_vec_lo.v, max_vec_hi.v, mask_vec_lo.v);
		max_ind_lo_128.v = _mm_blendv_pd(max_ind_lo_128.v, max_ind_hi_128.v, mask_vec_lo.v);

		max_vec_hi.v = _mm_permute_pd(max_vec_lo.v, 1);
		max_ind_hi_128.v = _mm_permute_pd(max_ind_lo_128.v, 1);

		//Performing work of CMP128 i.e generating mask
		mask_vec_lo.v = _mm_sub_pd(max_vec_lo.v, max_vec_hi.v);

		//Finding the maximum element
		max_vec_lo.v = _mm_blendv_pd(max_vec_lo.v, max_vec_hi.v, mask_vec_lo.v);
		max_ind_lo_128.v = _mm_blendv_pd(max_ind_lo_128.v, max_ind_hi_128.v, mask_vec_lo.v);

		abs_chi1_max = max_vec_lo.d[0];

		//If the largest number is negative it is NAN
		if (abs_chi1_max < 0)
			abs_chi1_max = NAN;

		i_max_l = max_ind_lo_128.d[0];

		for (i = n - n_left; i < n; i++)
		{
			double *chi1 = x;

			/* Get the real and imaginary components of chi1. */
			chi1_r = *chi1;

			/* Replace chi1_r and chi1_i with their absolute values. */
			abs_chi1 = fabs(chi1_r);

			/* If the absolute value of the current element exceeds that of
			   the previous largest, save it and its index. If NaN is
			   encountered, return the index of the first NaN. This
			   behavior mimics that of LAPACK's i?amax(). */
			if (abs_chi1_max < abs_chi1 || (isnan(abs_chi1) && !isnan(abs_chi1_max)))
			{
				abs_chi1_max = abs_chi1;
				i_max_l = i;
			}

			x += 1;
		}
	}

	// Return value
	*i_max = i_max_l;

	AOCL_DTL_TRACE_EXIT(AOCL_DTL_LEVEL_TRACE_3)
}

// ---------------------------------------------------------------------------------
