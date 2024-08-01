/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2016 - 2024, Advanced Micro Devices, Inc. All rights reserved.
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
	// transitioning from AVX to SSE instructions (which may occur later,
	// especially if BLIS is compiled with -mfpmath=sse).
	_mm256_zeroupper();

	/* Store final index to output variable. */
	*i_max = i_max_l;
    AOCL_DTL_TRACE_EXIT(AOCL_DTL_LEVEL_TRACE_3)
}

// -----------------------------------------------------------------------------


/*
	This macro takes a __m128d vector and a double pointer as inputs. It
	stores the largest element in the __m128d in the address pointed by the
	double pointer.

	Signature
	----------

	* 'max_res' - __m128d
	* 'max_val' - Double pointer
*/
#define _mm_vec_max_pd(max_res, max_val)\
	*(max_val) = (max_res[0] >= max_res[1]) ? max_res[0] : max_res[1];

/*
	Functionality
	--------------

	This function finds the first occurence of the absolute largest element in a double
	array and the range (start and end index) in which that element can be found.

	Function signature
	-------------------

	This function takes a void pointer as input (internally casted to a double pointer)
	which points to an array of type double, the correspending array's stride and length.
	It uses the function parameters to return the output.

	* 'x' - Void pointer pointing to an array of type double
	* 'incx' - Stride to point to the next element in the array
	* 'n' - Length of the array passed
	* 'max_num' - Double pointer to the memory of the absolute largest element
	* 'start_index', 'end_index' - Range in which the largest element can be found

	The function has been made static to restrict its scope.

	Exception
	----------

	1. When the length of the array or the increment is zero set the absolute maximum, start
	   index and end index to -1.
*/
static void bli_vec_absmax_double
(
	const void *x, dim_t incx, dim_t n,
	double *abs_max_num,
	dim_t *start_index, dim_t *end_index
)
{
	/*
		When the length of the array or the increment
		is zero set the absolute maximum, start index and
		end index to -1.
	*/
	if ( bli_zero_dim1( n ) || bli_zero_dim1( incx ) )
	{
		*abs_max_num = -1;
		*start_index = *end_index = -1;

		return;
	}

	// Cast the void pointer to double
	double *temp_ptr = (double *)x;
	double temp_max_val, curr_max_val = -1;

	dim_t window_start, window_end, i = 0;
	window_start = window_end = 0;

	/*
		When incx == 1 and n >= 2 the compute can be
		vectorized using AVX-2 or SSE instructions
	*/
	if (incx == 1 && n >= 2)
	{
		dim_t const n_elem_per_reg = 4;

		__m256d x_vec[12], max_array, sign_mask;
		v2dd_t max_hi, max_lo, sign_mask_128;

		// Initializing the mask to minus zero (-0.0)
		sign_mask = _mm256_set1_pd(-0.f);
		sign_mask_128.v = _mm_set1_pd(-0.f);

		for (i = 0; (i + 47) < n; i += 48)
		{
			// Load the elements into YMM registers
			x_vec[0] = _mm256_loadu_pd(temp_ptr);
			x_vec[1] = _mm256_loadu_pd(temp_ptr + n_elem_per_reg);
			x_vec[2] = _mm256_loadu_pd(temp_ptr + 2 * n_elem_per_reg);
			x_vec[3] = _mm256_loadu_pd(temp_ptr + 3 * n_elem_per_reg);

			/*
				Calculate the absolute value of the elements
				and store it in the same vectors
			*/
			x_vec[0] = _mm256_andnot_pd(sign_mask, x_vec[0]);
			x_vec[1] = _mm256_andnot_pd(sign_mask, x_vec[1]);
			x_vec[2] = _mm256_andnot_pd(sign_mask, x_vec[2]);
			x_vec[3] = _mm256_andnot_pd(sign_mask, x_vec[3]);

			/*
				Find the largest element in the corresponding
				vector indices for the given set of 256-bit vectors
			*/
			x_vec[0] = _mm256_max_pd(x_vec[0], x_vec[1]);
			x_vec[2] = _mm256_max_pd(x_vec[2], x_vec[3]);
			x_vec[0] = _mm256_max_pd(x_vec[0], x_vec[2]);

			x_vec[4] = _mm256_loadu_pd(temp_ptr + 4 * n_elem_per_reg);
			x_vec[5] = _mm256_loadu_pd(temp_ptr + 5 * n_elem_per_reg);
			x_vec[6] = _mm256_loadu_pd(temp_ptr + 6 * n_elem_per_reg);
			x_vec[7] = _mm256_loadu_pd(temp_ptr + 7 * n_elem_per_reg);

			x_vec[4] = _mm256_andnot_pd(sign_mask, x_vec[4]);
			x_vec[5] = _mm256_andnot_pd(sign_mask, x_vec[5]);
			x_vec[6] = _mm256_andnot_pd(sign_mask, x_vec[6]);
			x_vec[7] = _mm256_andnot_pd(sign_mask, x_vec[7]);

			x_vec[4] = _mm256_max_pd(x_vec[4], x_vec[5]);
			x_vec[6] = _mm256_max_pd(x_vec[6], x_vec[7]);
			x_vec[4] = _mm256_max_pd(x_vec[4], x_vec[6]);

			x_vec[8] = _mm256_loadu_pd(temp_ptr + 8 * n_elem_per_reg);
			x_vec[9] = _mm256_loadu_pd(temp_ptr + 9 * n_elem_per_reg);
			x_vec[10] = _mm256_loadu_pd(temp_ptr + 10 * n_elem_per_reg);
			x_vec[11] = _mm256_loadu_pd(temp_ptr + 11 * n_elem_per_reg);

			x_vec[8] = _mm256_andnot_pd(sign_mask, x_vec[8]);
			x_vec[9] = _mm256_andnot_pd(sign_mask, x_vec[9]);
			x_vec[10] = _mm256_andnot_pd(sign_mask, x_vec[10]);
			x_vec[11] = _mm256_andnot_pd(sign_mask, x_vec[11]);

			x_vec[8] = _mm256_max_pd(x_vec[8], x_vec[9]);
			x_vec[10] = _mm256_max_pd(x_vec[10], x_vec[11]);
			x_vec[8] = _mm256_max_pd(x_vec[10], x_vec[8]);

			max_array = _mm256_max_pd(x_vec[0], x_vec[4]);

			/*
				max_array holds the largest element in
				the corresponding vector indices
			*/
			max_array = _mm256_max_pd(max_array, x_vec[8]);

			// Extract the higher and lower 128-bit from the max_array
			max_hi.v = _mm256_extractf128_pd(max_array, 1);
			max_lo.v = _mm256_extractf128_pd(max_array, 0);

			/*
				Find the largest element in the corresponding
				vector indices for the given set of 128-bit vectors
			*/
			max_hi.v = _mm_max_pd(max_hi.v, max_lo.v);

			/*
				Find the largest element in the 128-bit vector
				and store it in temp_max_val
			*/
			_mm_vec_max_pd(max_hi.d, &temp_max_val);

			/*
				If the new max value found is greater than the previous
				max value, update the range and the largest value.
			*/
			if (curr_max_val < temp_max_val)
			{
				window_start = i;
				window_end = window_start + (12 * n_elem_per_reg);

				curr_max_val = temp_max_val;
			}

			// Increment the pointer
			temp_ptr += 12 * n_elem_per_reg;
		}

		for (; (i + 31) < n; i += 32)
		{
			x_vec[0] = _mm256_loadu_pd(temp_ptr);
			x_vec[1] = _mm256_loadu_pd(temp_ptr + n_elem_per_reg);
			x_vec[2] = _mm256_loadu_pd(temp_ptr + 2 * n_elem_per_reg);
			x_vec[3] = _mm256_loadu_pd(temp_ptr + 3 * n_elem_per_reg);

			x_vec[0] = _mm256_andnot_pd(sign_mask, x_vec[0]);
			x_vec[1] = _mm256_andnot_pd(sign_mask, x_vec[1]);
			x_vec[2] = _mm256_andnot_pd(sign_mask, x_vec[2]);
			x_vec[3] = _mm256_andnot_pd(sign_mask, x_vec[3]);

			x_vec[0] = _mm256_max_pd(x_vec[0], x_vec[1]);
			x_vec[2] = _mm256_max_pd(x_vec[2], x_vec[3]);
			x_vec[0] = _mm256_max_pd(x_vec[0], x_vec[2]);

			x_vec[4] = _mm256_loadu_pd(temp_ptr + 4 * n_elem_per_reg);
			x_vec[5] = _mm256_loadu_pd(temp_ptr + 5 * n_elem_per_reg);
			x_vec[6] = _mm256_loadu_pd(temp_ptr + 6 * n_elem_per_reg);
			x_vec[7] = _mm256_loadu_pd(temp_ptr + 7 * n_elem_per_reg);

			x_vec[4] = _mm256_andnot_pd(sign_mask, x_vec[4]);
			x_vec[5] = _mm256_andnot_pd(sign_mask, x_vec[5]);
			x_vec[6] = _mm256_andnot_pd(sign_mask, x_vec[6]);
			x_vec[7] = _mm256_andnot_pd(sign_mask, x_vec[7]);

			x_vec[4] = _mm256_max_pd(x_vec[4], x_vec[5]);
			x_vec[6] = _mm256_max_pd(x_vec[6], x_vec[7]);
			x_vec[4] = _mm256_max_pd(x_vec[4], x_vec[6]);

			max_array = _mm256_max_pd(x_vec[0], x_vec[4]);

			max_hi.v = _mm256_extractf128_pd(max_array, 1);
			max_lo.v = _mm256_extractf128_pd(max_array, 0);

			max_hi.v = _mm_max_pd(max_hi.v, max_lo.v);

			_mm_vec_max_pd(max_hi.d, &temp_max_val);

			if (curr_max_val < temp_max_val)
			{
				window_start = i;
				window_end = window_start + (8 * n_elem_per_reg);

				curr_max_val = temp_max_val;
			}

			temp_ptr += 8 * n_elem_per_reg;
		}

		for (; (i + 15) < n; i += 16)
		{
			x_vec[0] = _mm256_loadu_pd(temp_ptr);
			x_vec[1] = _mm256_loadu_pd(temp_ptr + n_elem_per_reg);
			x_vec[2] = _mm256_loadu_pd(temp_ptr + 2 * n_elem_per_reg);
			x_vec[3] = _mm256_loadu_pd(temp_ptr + 3 * n_elem_per_reg);

			x_vec[0] = _mm256_andnot_pd(sign_mask, x_vec[0]);
			x_vec[1] = _mm256_andnot_pd(sign_mask, x_vec[1]);
			x_vec[2] = _mm256_andnot_pd(sign_mask, x_vec[2]);
			x_vec[3] = _mm256_andnot_pd(sign_mask, x_vec[3]);

			x_vec[0] = _mm256_max_pd(x_vec[0], x_vec[1]);
			x_vec[2] = _mm256_max_pd(x_vec[2], x_vec[3]);

			max_array = _mm256_max_pd(x_vec[0], x_vec[2]);

			max_hi.v = _mm256_extractf128_pd(max_array, 1);
			max_lo.v = _mm256_extractf128_pd(max_array, 0);

			max_hi.v = _mm_max_pd(max_hi.v, max_lo.v);

			_mm_vec_max_pd(max_hi.d, &temp_max_val);

			if (curr_max_val < temp_max_val)
			{
				window_start = i;
				window_end = window_start + (4 * n_elem_per_reg);

				curr_max_val = temp_max_val;
			}

			temp_ptr += 4 * n_elem_per_reg;
		}

		for (; (i + 7) < n; i += 8)
		{
			x_vec[0] = _mm256_loadu_pd(temp_ptr);
			x_vec[1] = _mm256_loadu_pd(temp_ptr + n_elem_per_reg);

			x_vec[0] = _mm256_andnot_pd(sign_mask, x_vec[0]);
			x_vec[1] = _mm256_andnot_pd(sign_mask, x_vec[1]);

			max_array = _mm256_max_pd(x_vec[0], x_vec[1]);

			max_hi.v = _mm256_extractf128_pd(max_array, 1);
			max_lo.v = _mm256_extractf128_pd(max_array, 0);

			max_hi.v = _mm_max_pd(max_hi.v, max_lo.v);

			_mm_vec_max_pd(max_hi.d, &temp_max_val);

			if (curr_max_val < temp_max_val)
			{
				window_start = i;
				window_end = window_start + (2 * n_elem_per_reg);

				curr_max_val = temp_max_val;
			}

			temp_ptr += 2 * n_elem_per_reg;
		}

		for (; (i + 3) < n; i += 4)
		{
			max_array = _mm256_loadu_pd(temp_ptr);
			max_array = _mm256_andnot_pd(sign_mask, max_array);

			max_hi.v = _mm256_extractf128_pd(max_array, 1);
			max_lo.v = _mm256_extractf128_pd(max_array, 0);

			max_hi.v = _mm_max_pd(max_hi.v, max_lo.v);

			_mm_vec_max_pd(max_hi.d, &temp_max_val);

			if (curr_max_val < temp_max_val)
			{
				window_start = i;
				window_end = window_start + n_elem_per_reg;

				curr_max_val = temp_max_val;
			}

			temp_ptr += n_elem_per_reg;
		}

		for (; (i + 1) < n; i += 2)
		{
			max_hi.v = _mm_loadu_pd(temp_ptr);
			max_hi.v = _mm_andnot_pd(sign_mask_128.v, max_hi.v);

			_mm_vec_max_pd(max_hi.d, &temp_max_val);

			if (curr_max_val < temp_max_val)
			{
				window_start = i;
				window_end = window_start + 2;

				curr_max_val = temp_max_val;
			}

			temp_ptr += 2;
		}
	}

	/*
		This loops performs the compute in two cases:

		1. The complete compute when incx != 1
		2. It process the last element when 'n' is not a
		   multiple of 2.
	*/
	for (; i < n; ++i)
	{
		temp_max_val = fabs(*temp_ptr);

		if (temp_max_val > curr_max_val)
		{
			curr_max_val = temp_max_val;
			window_end = i;
			window_start = i;
		}

		temp_ptr += incx;
	}

	/*
		Store the index range  in which the largest element
		can be found in the address passed
	*/
	*start_index = window_start;
	*end_index = window_end;

	// Store the value of the largest element in the address passed
	*abs_max_num = curr_max_val;
}


/*
	Functionality
	-------------

	This function locates the index at which a given absolute value of an element
	first occurs.

	Function Signature
	-------------------

	This function takes a void pointer as input (internally casted to a double pointer)
	which points to an array of type double, the correspending array's stride and length.
	It uses the function parameters to return the output.

	* 'x' - Void pointer pointing to an array of type double
	* 'incx' - Stride to point to the next element in the array
	* 'n' - Length of the array passed
	* 'element' - Double pointer to the memory of the element to be searched
	* 'index' - Range in which the largest element can be found

	The function has been made static to restrict its scope.

	Exception
	----------

	1. When the length of the array or the increment is zero set the index to -1.
*/
static void bli_vec_search_double
(
    const void* x, dim_t incx,
    dim_t n,
    double* element,
    dim_t* index
)
{
	/*
		When the length of the array or the
		increment is zero set the index to -1.
	*/
	if (bli_zero_dim1(n) || bli_zero_dim1(incx))
	{
		*index = -1;

		return;
	}

	double *temp_ptr = (double *)x;

	dim_t i = 0;

	/*
		When incx == 1 and n >= 2 the compute can be
		vectorized using AVX-2 or SSE instructions.

		This vectorization does not reduce the total
		number of comparisons performed but vectorizes the
		calculation of the absolute values.
	*/
	if (incx == 1 && n >= 2)
	{
		const dim_t n_elem_per_reg = 4;

		__m256d x_vec, max_reg, mask_gen;

		// Initializing the mask to minus zero (-0.0)
		__m256d sign_mask = _mm256_set1_pd(-0.f);

		/*
			Set the register to the absolute
			value of the element passed
		*/
		max_reg = _mm256_set1_pd(fabs(*element));

		for (i = 0; (i + 3)< n; i += n_elem_per_reg)
		{
			// Load the array elements into the register
			x_vec = _mm256_loadu_pd(temp_ptr);

			// Calculate the absolute values of the elements
			x_vec = _mm256_andnot_pd(sign_mask, x_vec);

			/*
				Check for equality with the absolute value
				of the element to be searched for
			*/
			mask_gen = _mm256_cmp_pd(x_vec, max_reg, _CMP_EQ_OQ);

			/*
				Check if the element exists in the loaded vector
				using the mask generated.

				The mask is generated because comparison to zero is
				a cheaper operation.
			*/
			for (dim_t j = 0; j < n_elem_per_reg; ++j)
			{
				double mask_val = mask_gen[j];

				if (mask_val != 0)
				{
					*index = i + j;
					return;
				}
			}

			temp_ptr += n_elem_per_reg;
		}

		/*
			Issue vzeroupper instruction to clear upper lanes of ymm registers.
			This avoids a performance penalty caused by false dependencies when
			transitioning from AVX to SSE instructions (which may occur as soon
			as the n_left cleanup loop below if BLIS is compiled with
			-mfpmath=sse).
		*/
		_mm256_zeroupper();

		// Perform the above compute using SSE instructions
		__m128d x_vec_128, mask_gen_128, max_reg_128;

		max_reg_128 = _mm_set1_pd(*element);

		for (; (i + 1) < n; i += 2)
		{
			x_vec_128 = _mm_loadu_pd(temp_ptr);
			x_vec_128 = _mm_andnot_pd(_mm_set1_pd(-0.f), x_vec_128);

			mask_gen_128 = _mm_cmp_pd(x_vec_128, max_reg_128, _CMP_EQ_OQ);

			for (dim_t j = 0; j < 2; ++j)
			{
				double mask_val = mask_gen_128[j];

				if (mask_val != 0)
				{
					*index = i + j;
					return;
				}
			}

			temp_ptr += 2;
		}
	}

	/*
		This loops performs the compute in two cases:

		1. The complete compute when incx != 1
		2. It process the last element when 'n' is not a
		   multiple of 2.
	*/
	for (; i < n; i += 1)
	{
		double value = fabs(*temp_ptr);

		if (value == *element)
		{
			*index = i;
			return;
		}

		temp_ptr += incx;
	}

	// When the element is not found in the
	*index = -2;
}

/*
	Functionality
	-------------

	This function finds the index of the first element having maximum absolute value
	with the array index starting from 0.

	Function Signature
	-------------------

	This function takes a double pointer as input, the correspending vector's stride
	and length. It uses the function parameters to return the output.

	* 'x' - Double pointer pointing to an array
	* 'incx' - Stride to point to the next element in the array
	* 'n' - Length of the array passed
	* 'i_max' - Index at which the absolute largest element can be found
	* 'cntx' - BLIS context object

	Exception
	----------

	1. When the vector length is zero, return early. This directly emulates the behavior
	   of netlib BLAS's i?amax() routines.

	Deviation from BLAS
	--------------------

	1. In this function, the array index starts from 0 while in BLAS the indexing
	   starts from 1. The deviation is expected to be handled in the BLAS layer of
	   the library.

	Undefined behaviour
	-------------------

	1. The function results in undefined behaviour when NaN elements are present in the
	   array. This behaviour is BLAS complaint.
*/
BLIS_EXPORT_BLIS void bli_damaxv_zen_int
     (
       dim_t            n,
       double* restrict x, inc_t incx,
       dim_t*  restrict i_max,
       cntx_t* restrict cntx
     )
{
	// Temporary pointer used inside the function
	double *x_temp = x;

	// Will hold the absolute largest element in the array
	double max_val;

	/*
		Holds the index range in which the absolute
		largest element first occurs
	*/
	dim_t start_index, end_index;

	/*
		Length of the search space where the absolute
		largest element first occurs
	*/
	dim_t search_len;

	/*
		This function find the first occurence of the absolute largest element in a double
		array and the range (start and end index) in which that element can be found.
	*/
	bli_vec_absmax_double
	(
		(void *)x_temp, incx, n,
		&max_val,
		&start_index, &end_index
	);

	// Calculate the length of the search space
	search_len = end_index - start_index;

	dim_t element_index;

	if (start_index != end_index)
	{
		// Adjust the pointer based on the search range
		x_temp = x + (start_index * incx);

		/*
			This function locates the index at which a given absolute
			value of a element first occurs.
		*/
		bli_vec_search_double
		(
			(void *)x_temp, incx,
			search_len,
			&max_val,
			&element_index
		);

		// Calculate the index the of the absolute largest element and store it
		element_index = start_index + element_index;
	}
	else
	{
		// Store the index the of the absolute largest element
		element_index = start_index;
	}

	// Store final index to output variable.
	*i_max = element_index;
	AOCL_DTL_TRACE_EXIT(AOCL_DTL_LEVEL_TRACE_3)
}
