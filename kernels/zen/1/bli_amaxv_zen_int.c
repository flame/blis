/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2016 - 2025, Advanced Micro Devices, Inc. All rights reserved.
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

/*
	This macro takes a __m128 vector and a double pointer as inputs. It
	stores the largest element in the __m128 in the address pointed by the
	float pointer. In case of the vector having NaNs, it stores the non-NaN
	element(unless all are NaNs, in which case it stores NaN).

	Signature
	----------

	* 'max_res' - __m128
	* 'max_val' - Double pointer
*/
#define _mm_vec_max_ps(max_res, max_val) \
		bool first_nan = isnan(max_res[0]); \
		bool second_nan = isnan(max_res[1]); \
		bool third_nan = isnan(max_res[2]); \
		bool fourth_nan = isnan(max_res[3]); \
		float max_val_0, max_val_1; \
		/* Acquiring the abs max from the first two elements */ \
		/* In case both are NaNs or none of them is, we compare and
		   store the greater element */ \
		if(!(first_nan ^ second_nan)) \
		{ \
			max_val_0 = bli_max(max_res[0], max_res[1]); \
		} \
		/* If first element is NaN, store the second element */ \
		else if(first_nan) max_val_0 = max_res[1]; \
		/* If second element is NaN, store the first element */ \
		else 			   max_val_0 = max_res[0]; \
\
		/* Acquiring the abs max from the last two elements */ \
		/* In case both are NaNs or none of them is, we compare and
		   store the greater element */ \
		if(!(third_nan ^ fourth_nan)) \
		{ \
			max_val_1 = bli_max(max_res[2], max_res[3]); \
		} \
		/* If third element is NaN, store the fourth element */ \
		else if(third_nan) max_val_1 = max_res[3]; \
		/* If fourth element is NaN, store the third element */ \
		else 			   max_val_1 = max_res[2]; \
		/* Final conparision */ \
		if(!(isnan(max_val_0) ^ isnan(max_val_1))) \
		{ \
			*max_val = bli_max(max_val_0, max_val_1); \
		} \
		/* If third element is NaN, store the fourth element */ \
		else if(isnan(max_val_0)) *max_val = max_val_1; \
		/* If fourth element is NaN, store the third element */ \
		else 			   *max_val = max_val_0; \


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

	/* Declare the local variables for usage */
	float   abs_chi1_max;
	dim_t   i_max_l;
	dim_t   i;

	/* Initialize the index of the maximum absolute value to zero. */
	PASTEMAC(i,copys)( *zero_i, i_max_l );

	/* Initialize the index for the iterations. */
	PASTEMAC(i,copys)( *zero_i, i );

	/* Initialize the maximum absolute value search candidate with
	   -1, which is guaranteed to be less than all values we will
	   compute. */
	PASTEMAC(s,copys)( *minus_one, abs_chi1_max );

	// For non-unit strides, or very small vector lengths, compute with
	// scalar code.
	if ( incx != 1 || n < 8 )
	{
		// Variable to hold the abs value of the current element
		float   abs_chi1;
		for ( ; i < n; ++i )
		{
			float* chi1 = x + (i  )*incx;

			/* Replace chi1_r with the absolute values. */
			abs_chi1 = fabsf( *chi1 );

			/* If the absolute value of the current element exceeds that of
			   the previous largest, save it and its index. If NaN is
			   encountered, then treat it the same as if it were a valid
			   value that was smaller than any previously seen. The comparision
			   operation does this, since with NaN it always fails. This
			   behavior mimics that of LAPACK's i?amax(). */
			if ( abs_chi1_max < abs_chi1 )
			{
				abs_chi1_max = abs_chi1;
				i_max_l      = i;
			}
		}
	}
	else
	{
		dim_t const n_elem_per_reg = 8;

		__m256 x_vec[8], temp[4], max_array, sign_mask;
		v4sf_t max_hi, max_lo, sign_mask_128;

		// Initializing a local pointer for iterating
		float *temp_ptr = x;

		// Initializing variables to keep track of the
		// absolute maximum
		float abs_max_val = -1.0f, temp_max_val = -1.0f;

		// Initializing the start and end of the search space
		dim_t window_start = -1, window_end = -1;

		// Initializing the mask to minus zero (-0.0)
		sign_mask = _mm256_set1_ps(-0.f);
		sign_mask_128.v = _mm_set1_ps(-0.f);

		// Setting the temporary registers to 0.0
		temp[0] = _mm256_setzero_ps();
		temp[1] = _mm256_setzero_ps();
		temp[2] = _mm256_setzero_ps();
		temp[3] = _mm256_setzero_ps();

		/*
			The logic for this kernels can be broken down into two
			phases :
			- Finding the absolute maximum value and the search space
			  for its first occurence.
			- Finding the index of the first occurence of absolute maximum
			  value.
		*/
		for (; (i + 63) < n; i += 64)
		{
			// Loading the first 4 vectors
			x_vec[0] = _mm256_loadu_ps(temp_ptr);
			x_vec[1] = _mm256_loadu_ps(temp_ptr + n_elem_per_reg);
			x_vec[2] = _mm256_loadu_ps(temp_ptr + 2 * n_elem_per_reg);
			x_vec[3] = _mm256_loadu_ps(temp_ptr + 3 * n_elem_per_reg);

			// Obtaining the absolute values
			x_vec[0] = _mm256_andnot_ps(sign_mask, x_vec[0]);
			x_vec[1] = _mm256_andnot_ps(sign_mask, x_vec[1]);
			x_vec[2] = _mm256_andnot_ps(sign_mask, x_vec[2]);
			x_vec[3] = _mm256_andnot_ps(sign_mask, x_vec[3]);

			// Propagating the absolute maximums using temp.
			// NOTE : NaNs as part of the load are not propagated
			//        if they are the first operand, due to the nature
			//		  of vmaxpd instruction.
			temp[0] = _mm256_max_ps(x_vec[0], temp[0]);
			temp[1] = _mm256_max_ps(x_vec[1], temp[1]);
			temp[2] = _mm256_max_ps(x_vec[2], temp[2]);
			temp[3] = _mm256_max_ps(x_vec[3], temp[3]);

			// Loading the next 4 vectors
			x_vec[4] = _mm256_loadu_ps(temp_ptr + 4 * n_elem_per_reg);
			x_vec[5] = _mm256_loadu_ps(temp_ptr + 5 * n_elem_per_reg);
			x_vec[6] = _mm256_loadu_ps(temp_ptr + 6 * n_elem_per_reg);
			x_vec[7] = _mm256_loadu_ps(temp_ptr + 7 * n_elem_per_reg);

			// Obtaining the absolute values
			x_vec[4] = _mm256_andnot_ps(sign_mask, x_vec[4]);
			x_vec[5] = _mm256_andnot_ps(sign_mask, x_vec[5]);
			x_vec[6] = _mm256_andnot_ps(sign_mask, x_vec[6]);
			x_vec[7] = _mm256_andnot_ps(sign_mask, x_vec[7]);

			// Propagating the absolute maximums using temp.
			// NOTE : NaNs as part of the load are not propagated
			//        if they are the first operand, due to the nature
			//		  of vmaxpd instruction.
			temp[0] = _mm256_max_ps(x_vec[4], temp[0]);
			temp[1] = _mm256_max_ps(x_vec[5], temp[1]);
			temp[2] = _mm256_max_ps(x_vec[6], temp[2]);
			temp[3] = _mm256_max_ps(x_vec[7], temp[3]);

			// Obtaining the final abs max values from the 8 loads
			temp[0] = _mm256_max_ps(temp[0], temp[1]);
			temp[2] = _mm256_max_ps(temp[2], temp[3]);

			max_array = _mm256_max_ps(temp[0], temp[2]);

			// Reduction to obtain the abs max as a scalar
			max_hi.v = _mm256_extractf128_ps(max_array, 1);
			max_lo.v = _mm256_extractf128_ps(max_array, 0);

			max_hi.v = _mm_max_ps(max_hi.v, max_lo.v);

			_mm_vec_max_ps(max_hi.f, &temp_max_val);

			// Updating the search space if the current maximum is
			// greater than the previous maximum
			if (abs_max_val < temp_max_val)
			{
				window_start = i;
				window_end = window_start + (8 * n_elem_per_reg);

				abs_max_val = temp_max_val;
			}

			// Increment the pointer for the next iteration
			temp_ptr += 8 * n_elem_per_reg;
		}

		for (; (i + 31) < n; i += 32)
		{
			// Loading the first 4 vectors
			x_vec[0] = _mm256_loadu_ps(temp_ptr);
			x_vec[1] = _mm256_loadu_ps(temp_ptr + n_elem_per_reg);
			x_vec[2] = _mm256_loadu_ps(temp_ptr + 2 * n_elem_per_reg);
			x_vec[3] = _mm256_loadu_ps(temp_ptr + 3 * n_elem_per_reg);

			// Obtaining the absolute values
			x_vec[0] = _mm256_andnot_ps(sign_mask, x_vec[0]);
			x_vec[1] = _mm256_andnot_ps(sign_mask, x_vec[1]);
			x_vec[2] = _mm256_andnot_ps(sign_mask, x_vec[2]);
			x_vec[3] = _mm256_andnot_ps(sign_mask, x_vec[3]);

			// Propagating the absolute maximums using temp.
			// NOTE : NaNs as part of the load are not propagated
			//        if they are the first operand, due to the nature
			//		  of vmaxpd instruction.
			temp[0] = _mm256_max_ps(x_vec[0], temp[0]);
			temp[1] = _mm256_max_ps(x_vec[1], temp[1]);
			temp[0] = _mm256_max_ps(x_vec[2], temp[0]);
			temp[1] = _mm256_max_ps(x_vec[3], temp[1]);

			// Obtaining the final abs max values from the 4 loads
			max_array = _mm256_max_ps(temp[0], temp[1]);

			// Reduction to obtain the abs max as a scalar
			max_hi.v = _mm256_extractf128_ps(max_array, 1);
			max_lo.v = _mm256_extractf128_ps(max_array, 0);

			max_hi.v = _mm_max_ps(max_hi.v, max_lo.v);

			_mm_vec_max_ps(max_hi.f, &temp_max_val);

			// Updating the search space if the current maximum is
			// greater than the previous maximum
			if (abs_max_val < temp_max_val)
			{
				window_start = i;
				window_end = window_start + (4 * n_elem_per_reg);

				abs_max_val = temp_max_val;
			}

			// Increment the pointer for the next iteration
			temp_ptr += 4 * n_elem_per_reg;
		}

		for (; (i + 15) < n; i += 16)
		{
			// Loading the first 4 vectors
			x_vec[0] = _mm256_loadu_ps(temp_ptr);
			x_vec[1] = _mm256_loadu_ps(temp_ptr + n_elem_per_reg);

			// Obtaining the absolute values
			x_vec[0] = _mm256_andnot_ps(sign_mask, x_vec[0]);
			x_vec[1] = _mm256_andnot_ps(sign_mask, x_vec[1]);

			// Propagating the absolute maximums using temp.
			// NOTE : NaNs as part of the load are not propagated
			//        if they are the first operand, due to the nature
			//		  of vmaxpd instruction.
			temp[0] = _mm256_max_ps(x_vec[0], temp[0]);
			temp[1] = _mm256_max_ps(x_vec[1], temp[1]);

			// Obtaining the final abs max values from the 4 loads
			max_array = _mm256_max_ps(temp[0], temp[1]);

			// Reduction to obtain the abs max as a scalar
			max_hi.v = _mm256_extractf128_ps(max_array, 1);
			max_lo.v = _mm256_extractf128_ps(max_array, 0);

			max_hi.v = _mm_max_ps(max_hi.v, max_lo.v);

			_mm_vec_max_ps(max_hi.f, &temp_max_val);

			// Updating the search space if the current maximum is
			// greater than the previous maximum
			if (abs_max_val < temp_max_val)
			{
				window_start = i;
				window_end = window_start + (2 * n_elem_per_reg);

				abs_max_val = temp_max_val;
			}

			// Increment the pointer for the next iteration
			temp_ptr += 2 * n_elem_per_reg;
		}

		for (; (i + 7) < n; i += 8)
		{
			// Loading the vector
			x_vec[0] = _mm256_loadu_ps(temp_ptr);

			// Obtaining the absolute values
			x_vec[0] = _mm256_andnot_ps(sign_mask, x_vec[0]);

			// Propagating the absolute maximums using temp.
			// NOTE : NaNs as part of the load are not propagated
			//        if they are the first operand, due to the nature
			//		  of vmaxpd instruction.
			max_array = _mm256_max_ps(x_vec[0], temp[0]);

			max_hi.v = _mm256_extractf128_ps(max_array, 1);
			max_lo.v = _mm256_extractf128_ps(max_array, 0);

			// Reduction to obtain the abs max as a scalar
			max_hi.v = _mm_max_ps(max_hi.v, max_lo.v);

			_mm_vec_max_ps(max_hi.f, &temp_max_val);

			// Updating the search space if the current maximum is
			// greater than the previous maximum
			if (abs_max_val < temp_max_val)
			{
				window_start = i;
				window_end = window_start + n_elem_per_reg;

				abs_max_val = temp_max_val;
			}

			// Increment the pointer for the next iteration
			temp_ptr += n_elem_per_reg;
		}

		for (; (i + 3) < n; i += 4)
		{
			// Loading the 4 elements
			max_hi.v = _mm_loadu_ps(temp_ptr);
			max_hi.v = _mm_andnot_ps(sign_mask_128.v, max_hi.v);

			// Finding the maximum without NaN propagation
			_mm_vec_max_ps(max_hi.f, &temp_max_val);

			// Updating the search space if the current maximum is
			// greater than the previous maximum
			if (abs_max_val < temp_max_val)
			{
				window_start = i;
				window_end = window_start + 4;

				abs_max_val = temp_max_val;
			}

			// Increment the pointer for the next iteration
			temp_ptr += 4;
		}

		for (; i < n; ++i)
		{
			temp_max_val = fabsf(*temp_ptr);

			if (temp_max_val > abs_max_val)
			{
				abs_max_val = temp_max_val;
				window_end = i;
				window_start = i;
			}

			temp_ptr += incx;
		}

		// Setting the variables for finding the first occurence
		// of abs max value(max search space length is 64)
		temp_ptr = x + window_start;
		i_max_l = window_start;

		// Searching for the first occurence of the element
		while( i_max_l < window_end )
		{
			float value = fabsf(*temp_ptr);

			if ( value == abs_max_val )
				break;

			i_max_l += 1;
			temp_ptr += 1;
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
	double pointer. In case of the vector having NaNs, it stores the non-NaN
	element(unless both are NaNs, in which case it stores NaN).

	Signature
	----------

	* 'max_res' - __m128d
	* 'max_val' - Double pointer
*/
#define _mm_vec_max_pd(max_res, max_val) \
		/* Checking if first and second elements are NaNs */ \
		bool first_nan = isnan(max_res[0]); \
		bool second_nan = isnan(max_res[1]); \
		/* In case both are NaNs or none of them is, we compare and
		   store the greater element */ \
		if(!(first_nan ^ second_nan)) \
		{ \
			*max_val = bli_max(max_res[0], max_res[1]);; \
		} \
		/* If first element is NaN, store the second element */ \
		else if(first_nan) *max_val = max_res[1]; \
		/* If second element is NaN, store the first element */ \
		else 			   *max_val = max_res[0];

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

	The function has been made static(BLIS_INLINE) to restrict its scope.

	Exception
	----------

	1. When the length of the array or the increment is zero set the absolute maximum, start
	   index and end index to -1.
*/
BLIS_INLINE void bli_vec_absmax_double
(
	const void *x, dim_t incx, dim_t n,
	double *abs_max_num,
	dim_t *start_index, dim_t *end_index
)
{
	// Local variables/pointers to hold the relevant info
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

		__m256d x_vec[8], temp[4], max_array, sign_mask;
		v2dd_t max_hi, max_lo, sign_mask_128;

		// Initializing the mask to minus zero (-0.0)
		sign_mask = _mm256_set1_pd(-0.f);
		sign_mask_128.v = _mm_set1_pd(-0.f);

		// Setting the temporary registers to 0.0
		temp[0] = _mm256_setzero_pd();
		temp[1] = _mm256_setzero_pd();
		temp[2] = _mm256_setzero_pd();
		temp[3] = _mm256_setzero_pd();

		for (i = 0; (i + 47) < n; i += 48)
		{
			// Loading the first 4 vectors
			x_vec[0] = _mm256_loadu_pd(temp_ptr);
			x_vec[1] = _mm256_loadu_pd(temp_ptr + n_elem_per_reg);
			x_vec[2] = _mm256_loadu_pd(temp_ptr + 2 * n_elem_per_reg);
			x_vec[3] = _mm256_loadu_pd(temp_ptr + 3 * n_elem_per_reg);

			// Obtaining the absolute values
			x_vec[0] = _mm256_andnot_pd(sign_mask, x_vec[0]);
			x_vec[1] = _mm256_andnot_pd(sign_mask, x_vec[1]);
			x_vec[2] = _mm256_andnot_pd(sign_mask, x_vec[2]);
			x_vec[3] = _mm256_andnot_pd(sign_mask, x_vec[3]);

			// Propagating the absolute maximums using temp.
			// NOTE : NaNs as part of the load are not propagated
			//        if they are the first operand, due to the nature
			//		  of vmaxpd instruction.
			temp[0] = _mm256_max_pd(x_vec[0], temp[0]);
			temp[1] = _mm256_max_pd(x_vec[1], temp[1]);
			temp[2] = _mm256_max_pd(x_vec[2], temp[2]);
			temp[3] = _mm256_max_pd(x_vec[3], temp[3]);

			// Loading the next 4 vectors
			x_vec[4] = _mm256_loadu_pd(temp_ptr + 4 * n_elem_per_reg);
			x_vec[5] = _mm256_loadu_pd(temp_ptr + 5 * n_elem_per_reg);
			x_vec[6] = _mm256_loadu_pd(temp_ptr + 6 * n_elem_per_reg);
			x_vec[7] = _mm256_loadu_pd(temp_ptr + 7 * n_elem_per_reg);

			// Obtaining the absolute values
			x_vec[4] = _mm256_andnot_pd(sign_mask, x_vec[4]);
			x_vec[5] = _mm256_andnot_pd(sign_mask, x_vec[5]);
			x_vec[6] = _mm256_andnot_pd(sign_mask, x_vec[6]);
			x_vec[7] = _mm256_andnot_pd(sign_mask, x_vec[7]);

			// Propagating the absolute maximums using temp.
			// NOTE : NaNs as part of the load are not propagated
			//        if they are the first operand, due to the nature
			//		  of vmaxpd instruction.
			temp[0] = _mm256_max_pd(x_vec[4], temp[0]);
			temp[1] = _mm256_max_pd(x_vec[5], temp[1]);
			temp[2] = _mm256_max_pd(x_vec[6], temp[2]);
			temp[3] = _mm256_max_pd(x_vec[7], temp[3]);

			// Loading the last 4 vectors
			x_vec[0] = _mm256_loadu_pd(temp_ptr + 8 * n_elem_per_reg);
			x_vec[1] = _mm256_loadu_pd(temp_ptr + 9 * n_elem_per_reg);
			x_vec[2] = _mm256_loadu_pd(temp_ptr + 10 * n_elem_per_reg);
			x_vec[3] = _mm256_loadu_pd(temp_ptr + 11 * n_elem_per_reg);

			// Obtaining the absolute values
			x_vec[0] = _mm256_andnot_pd(sign_mask, x_vec[0]);
			x_vec[1] = _mm256_andnot_pd(sign_mask, x_vec[1]);
			x_vec[2] = _mm256_andnot_pd(sign_mask, x_vec[2]);
			x_vec[3] = _mm256_andnot_pd(sign_mask, x_vec[3]);

			// Propagating the absolute maximums using temp.
			// NOTE : NaNs as part of the load are not propagated
			//        if they are the first operand, due to the nature
			//		  of vmaxpd instruction.
			temp[0] = _mm256_max_pd(x_vec[0], temp[0]);
			temp[1] = _mm256_max_pd(x_vec[1], temp[1]);
			temp[2] = _mm256_max_pd(x_vec[2], temp[2]);
			temp[3] = _mm256_max_pd(x_vec[3], temp[3]);

			// Obtaining the final abs max values from the 8 loads
			temp[0] = _mm256_max_pd(temp[0], temp[1]);
			temp[2] = _mm256_max_pd(temp[2], temp[3]);

			max_array = _mm256_max_pd(temp[0], temp[2]);

			// Reduction to obtain the abs max as a scalar
			max_hi.v = _mm256_extractf128_pd(max_array, 1);
			max_lo.v = _mm256_extractf128_pd(max_array, 0);

			max_hi.v = _mm_max_pd(max_hi.v, max_lo.v);

			_mm_vec_max_pd(max_hi.d, &temp_max_val);

			// Updating the search space if the current maximum is
			// greater than the previous maximum
			if (curr_max_val < temp_max_val)
			{
				window_start = i;
				window_end = window_start + (12 * n_elem_per_reg);

				curr_max_val = temp_max_val;
			}

			// Increment the pointer for the next iteration
			temp_ptr += 12 * n_elem_per_reg;
		}

		for (; (i + 31) < n; i += 32)
		{
			// Loading the first 4 vectors
			x_vec[0] = _mm256_loadu_pd(temp_ptr);
			x_vec[1] = _mm256_loadu_pd(temp_ptr + n_elem_per_reg);
			x_vec[2] = _mm256_loadu_pd(temp_ptr + 2 * n_elem_per_reg);
			x_vec[3] = _mm256_loadu_pd(temp_ptr + 3 * n_elem_per_reg);

			// Obtaining the absolute values
			x_vec[0] = _mm256_andnot_pd(sign_mask, x_vec[0]);
			x_vec[1] = _mm256_andnot_pd(sign_mask, x_vec[1]);
			x_vec[2] = _mm256_andnot_pd(sign_mask, x_vec[2]);
			x_vec[3] = _mm256_andnot_pd(sign_mask, x_vec[3]);

			// Propagating the absolute maximums using temp.
			// NOTE : NaNs as part of the load are not propagated
			//        if they are the first operand, due to the nature
			//		  of vmaxpd instruction.
			temp[0] = _mm256_max_pd(x_vec[0], temp[0]);
			temp[1] = _mm256_max_pd(x_vec[1], temp[1]);
			temp[2] = _mm256_max_pd(x_vec[2], temp[2]);
			temp[3] = _mm256_max_pd(x_vec[3], temp[3]);

			// Loading the next 4 vectors
			x_vec[4] = _mm256_loadu_pd(temp_ptr + 4 * n_elem_per_reg);
			x_vec[5] = _mm256_loadu_pd(temp_ptr + 5 * n_elem_per_reg);
			x_vec[6] = _mm256_loadu_pd(temp_ptr + 6 * n_elem_per_reg);
			x_vec[7] = _mm256_loadu_pd(temp_ptr + 7 * n_elem_per_reg);

			// Obtaining the absolute values
			x_vec[4] = _mm256_andnot_pd(sign_mask, x_vec[4]);
			x_vec[5] = _mm256_andnot_pd(sign_mask, x_vec[5]);
			x_vec[6] = _mm256_andnot_pd(sign_mask, x_vec[6]);
			x_vec[7] = _mm256_andnot_pd(sign_mask, x_vec[7]);

			// Propagating the absolute maximums using temp.
			// NOTE : NaNs as part of the load are not propagated
			//        if they are the first operand, due to the nature
			//		  of vmaxpd instruction.
			temp[0] = _mm256_max_pd(x_vec[4], temp[0]);
			temp[1] = _mm256_max_pd(x_vec[5], temp[1]);
			temp[2] = _mm256_max_pd(x_vec[6], temp[2]);
			temp[3] = _mm256_max_pd(x_vec[7], temp[3]);

			// Obtaining the final abs max values from the 8 loads
			temp[0] = _mm256_max_pd(temp[0], temp[1]);
			temp[2] = _mm256_max_pd(temp[2], temp[3]);

			max_array = _mm256_max_pd(temp[0], temp[2]);

			// Reduction to obtain the abs max as a scalar
			max_hi.v = _mm256_extractf128_pd(max_array, 1);
			max_lo.v = _mm256_extractf128_pd(max_array, 0);

			max_hi.v = _mm_max_pd(max_hi.v, max_lo.v);

			_mm_vec_max_pd(max_hi.d, &temp_max_val);

			// Updating the search space if the current maximum is
			// greater than the previous maximum
			if (curr_max_val < temp_max_val)
			{
				window_start = i;
				window_end = window_start + (8 * n_elem_per_reg);

				curr_max_val = temp_max_val;
			}

			// Increment the pointer for the next iteration
			temp_ptr += 8 * n_elem_per_reg;
		}

		for (; (i + 15) < n; i += 16)
		{
			// Loading the first 4 vectors
			x_vec[0] = _mm256_loadu_pd(temp_ptr);
			x_vec[1] = _mm256_loadu_pd(temp_ptr + n_elem_per_reg);
			x_vec[2] = _mm256_loadu_pd(temp_ptr + 2 * n_elem_per_reg);
			x_vec[3] = _mm256_loadu_pd(temp_ptr + 3 * n_elem_per_reg);

			// Obtaining the absolute values
			x_vec[0] = _mm256_andnot_pd(sign_mask, x_vec[0]);
			x_vec[1] = _mm256_andnot_pd(sign_mask, x_vec[1]);
			x_vec[2] = _mm256_andnot_pd(sign_mask, x_vec[2]);
			x_vec[3] = _mm256_andnot_pd(sign_mask, x_vec[3]);

			// Propagating the absolute maximums using temp.
			// NOTE : NaNs as part of the load are not propagated
			//        if they are the first operand, due to the nature
			//		  of vmaxpd instruction.
			temp[0] = _mm256_max_pd(x_vec[0], temp[0]);
			temp[1] = _mm256_max_pd(x_vec[1], temp[1]);
			temp[0] = _mm256_max_pd(x_vec[2], temp[0]);
			temp[1] = _mm256_max_pd(x_vec[3], temp[1]);

			// Obtaining the final abs max values from the 4 loads
			max_array = _mm256_max_pd(temp[0], temp[1]);

			// Reduction to obtain the abs max as a scalar
			max_hi.v = _mm256_extractf128_pd(max_array, 1);
			max_lo.v = _mm256_extractf128_pd(max_array, 0);

			max_hi.v = _mm_max_pd(max_hi.v, max_lo.v);

			_mm_vec_max_pd(max_hi.d, &temp_max_val);

			// Updating the search space if the current maximum is
			// greater than the previous maximum
			if (curr_max_val < temp_max_val)
			{
				window_start = i;
				window_end = window_start + (4 * n_elem_per_reg);

				curr_max_val = temp_max_val;
			}

			// Increment the pointer for the next iteration
			temp_ptr += 4 * n_elem_per_reg;
		}

		for (; (i + 7) < n; i += 8)
		{
			// Loading the first 2 vectors
			x_vec[0] = _mm256_loadu_pd(temp_ptr);
			x_vec[1] = _mm256_loadu_pd(temp_ptr + n_elem_per_reg);

			// Obtaining the absolute values
			x_vec[0] = _mm256_andnot_pd(sign_mask, x_vec[0]);
			x_vec[1] = _mm256_andnot_pd(sign_mask, x_vec[1]);

			// Propagating the absolute maximums using temp.
			// NOTE : NaNs as part of the load are not propagated
			//        if they are the first operand, due to the nature
			//		  of vmaxpd instruction.
			temp[0] = _mm256_max_pd(x_vec[0], temp[0]);
			max_array = _mm256_max_pd(x_vec[1], temp[0]);

			// Reduction to obtain the abs max as a scalar
			max_hi.v = _mm256_extractf128_pd(max_array, 1);
			max_lo.v = _mm256_extractf128_pd(max_array, 0);

			max_hi.v = _mm_max_pd(max_hi.v, max_lo.v);

			_mm_vec_max_pd(max_hi.d, &temp_max_val);

			// Updating the search space if the current maximum is
			// greater than the previous maximum
			if (curr_max_val < temp_max_val)
			{
				window_start = i;
				window_end = window_start + (2 * n_elem_per_reg);

				curr_max_val = temp_max_val;
			}

			// Increment the pointer for the next iteration
			temp_ptr += 2 * n_elem_per_reg;
		}

		for (; (i + 3) < n; i += 4)
		{
			// Loading the vector
			x_vec[0] = _mm256_loadu_pd(temp_ptr);

			// Obtaining the absolute values
			x_vec[0] = _mm256_andnot_pd(sign_mask, x_vec[0]);

			// Propagating the absolute maximums using temp.
			// NOTE : NaNs as part of the load are not propagated
			//        if they are the first operand, due to the nature
			//		  of vmaxpd instruction.
			max_array = _mm256_max_pd(x_vec[0], temp[0]);

			max_hi.v = _mm256_extractf128_pd(max_array, 1);
			max_lo.v = _mm256_extractf128_pd(max_array, 0);

			// Reduction to obtain the abs max as a scalar
			max_hi.v = _mm_max_pd(max_hi.v, max_lo.v);

			_mm_vec_max_pd(max_hi.d, &temp_max_val);

			// Updating the search space if the current maximum is
			// greater than the previous maximum
			if (curr_max_val < temp_max_val)
			{
				window_start = i;
				window_end = window_start + n_elem_per_reg;

				curr_max_val = temp_max_val;
			}

			// Increment the pointer for the next iteration
			temp_ptr += n_elem_per_reg;
		}

		for (; (i + 1) < n; i += 2)
		{
			// Loading the 2 elements
			max_hi.v = _mm_loadu_pd(temp_ptr);
			max_hi.v = _mm_andnot_pd(sign_mask_128.v, max_hi.v);

			// Finding the maximum without NaN propagation
			_mm_vec_max_pd(max_hi.d, &temp_max_val);

			// Updating the search space if the current maximum is
			// greater than the previous maximum
			if (curr_max_val < temp_max_val)
			{
				window_start = i;
				window_end = window_start + 2;

				curr_max_val = temp_max_val;
			}

			// Increment the pointer for the next iteration
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

	The function has been made static(BLIS_INLINE) to restrict its scope.

	Exception
	----------

	1. When the length of the array or the increment is zero set the index to -1.
*/
BLIS_INLINE void bli_vec_search_double
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
