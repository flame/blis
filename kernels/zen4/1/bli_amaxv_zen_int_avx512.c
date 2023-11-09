/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2022 - 2023, Advanced Micro Devices, Inc. All rights reserved.

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
typedef union
{
    __m512d v;
    double d[8] __attribute__((aligned(64)));
} v8df_t;

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
    // *minus_one = -1
    float *minus_one = PASTEMAC(s, m1); // bli_sm1()
                                        // *zero_i = 0
    dim_t *zero_i = PASTEMAC(i, 0);        // bli_i0()

    // Used to replace NAN in registers. This value is decremented each time
    // remove NAN is applied so as to keep the NAN value replacements unique.
    float nan_repl = -1.0;

    float fndMaxVal; // Max value will be stored in this
    dim_t fndInd;       // Max value's index will be stored in this
    // Iterator for loops to keep continuity throughout the loops
    dim_t i;

    /* If the vector length is zero, return early. This directly emulates
    the behavior of netlib BLAS's i?amax() routines. */
    if (bli_zero_dim1(n))
    {
        /* Set i_max to zero if dimension is 0, no need to compute */
        // Copy zero_i, that is 0 to i_max (i_max = 0)
        PASTEMAC(i, copys) // bli_icopys
        (*zero_i, *i_max);
        AOCL_DTL_TRACE_EXIT(AOCL_DTL_LEVEL_TRACE_3)
        return;
    }

    /* Initialize the index of the maximum absolute value to zero. */
    // Copy zero_i, that is 0 to fndInd (fndInd = 0)
    PASTEMAC(i, copys) // bli_icopys
    (*zero_i, fndInd);

    /* Initialize the maximum absolute value search candidate with
    -1, which is guaranteed to be less than all values we will
    compute. */
    // Copy minus_one to fndMaxVal real and imaginary.
    PASTEMAC(s, copys) // bli_scopys
    (*minus_one, fndMaxVal);

    // For non-unit strides, or very small vector lengths, compute with
    // scalar code.
    // n is less than the single vector length or non unit stride.
    if (incx != 1 || n < 16)
    {
        for (i = 0; i < n; ++i)
        {
            // Call math.h fabsf to take absolute value of *(x +(i)*incx)
            float absval = fabsf(*(x + (i)*incx));
            if (fndMaxVal < absval || (isnan(absval) && !isnan(fndMaxVal)))
            {
                // If max value is found, set the value and index
                fndMaxVal = absval;
                fndInd = i;
            }
        }
    }
    else
    {
        dim_t num_iter, num_remain;
        dim_t num_vector_elements = 16;
        /* Total Registers used is
        * xmm0-xmm4
        * ymm5-ymm9
        * zmm10-zmm26
        * There are 6 free registers to use
        */
        // zmm register 15x
        v16sf_t x_vec_1, x_vec_2, x_vec_3, max_vec_1, max_vec_2,
                max_vec_3, maxInd_vec_1, maxInd_vec_2,
                maxInd_vec_3, index_vec_1, ind_vec_2,
                ind_vec_3, inc_vec, mask,
                abs_mask;
        // ymm register 5x
        v8sf_t max_vec_lo, max_vec_hi,
               maxInd_vec_lo, maxInd_vec_hi,
               mask_vec_lo;
        // xmm register 5x
        v4sf_t max_vec_lo_lo, max_vec_lo_hi,
               maxInd_vec_lo_lo, maxInd_vec_lo_hi,
               mask_vec_lo_lo;
        // zmm register 1x
        __m512i intMask;
        // k register 3x
        __mmask16 mask_vec_1, mask_vec_2,
                  mask_vec_3;

        // Number of iterations for main loop.
        num_iter = n / num_vector_elements;
        // Number of iterations remaining for residual non vector loop
        num_remain = n % num_vector_elements;
        // A number with signbit one and others 0 IEEE-754
        abs_mask.v = _mm512_set1_ps(-0.f);
        // index_vector after loading max_vector with initial values.
        index_vec_1.v = _mm512_setr_ps(16, 17, 18, 19, 20, 21,
                                        22, 23, 24, 25, 26, 27,
                                        28, 29, 30, 31);
        // Broadcast 16. This is to increment the vector easily
        inc_vec.v = _mm512_set1_ps(16);
        // Load 16 float values from memory
        max_vec_1.v = _mm512_loadu_ps(x);
        // max_vector = abs(max_vector)
        max_vec_1.v = _mm512_andnot_ps(abs_mask.v, max_vec_1.v);
        // Remove nan and replace with -ve values
        REMOVE_NAN_512S(max_vec_1.v);

        // Increment x vector as we have loaded 16 values
        x += num_vector_elements;
        // indexes for values present in max vector.
        maxInd_vec_1.v = _mm512_setr_ps(0, 1, 2, 3, 4, 5, 6, 7, 8,
                                            9, 10, 11, 12, 13, 14, 15);

        dim_t i = 1;
        for (; (i + 4) < num_iter; i += 5)
        {
            /*
                Unrolled to process 5 at a time. It basically works
                by taking a master max_vec_1 and a maxInd_vec_1
                holding indexes. Elements are taken from the RAM on a batch
                of 5 (1 master max_vec_1 already exists to compare so
                6 elements). Now each 2 of them is compared with each other
                and an intermediate result is obtained. This intermediate
                result is again with each other and combined until we reach
                one vector in max_vector and maxIndex_vector.
            */

            // Load the vector and subs NAN
            // Load Value x values
            x_vec_1.v = _mm512_loadu_ps(x);
            // x_vec_1 = abs(x_vec_1)
            x_vec_1.v = _mm512_andnot_ps(abs_mask.v, x_vec_1.v);
            // Increment x vector as we have loaded 16 values
            x += num_vector_elements;
            // Remove nan and replace with -ve values
            REMOVE_NAN_512S(x_vec_1.v);

            // Mask Generation of 1st(can be previous max) and 2nd element
            // mask = max_vector - x_vec_1
            mask.v = _mm512_sub_ps(max_vec_1.v, x_vec_1.v);
            // Type cast mask from IEEE754 (float) to integer type
            // This operation will not need a new register, its just to convince
            // the compiler. But its accounted as separate register in the
            // above calculations
            intMask = _mm512_castps_si512(mask.v);
            // Extract the signbit and build the mask.
            mask_vec_1 = _mm512_movepi32_mask(intMask);

            // Load 2 elements to 2nd max and x vector, set indexes
            // Load Value x values
            max_vec_2.v = _mm512_loadu_ps(x);
            // max_vec_2 = abs(max_vec_2)
            max_vec_2.v = _mm512_andnot_ps(abs_mask.v, max_vec_2.v);
            // Remove nan and replace with -ve values
            REMOVE_NAN_512S(max_vec_2.v);
            // Increment x vector as we have loaded 16 values
            x += num_vector_elements;
            // Increment the index vector to point to next indexes.
            maxInd_vec_2.v = _mm512_add_ps(index_vec_1.v, inc_vec.v);

            // Load Value x values
            x_vec_2.v = _mm512_loadu_ps(x);
            // x_vec_2 = abs(x_vec_2)
            x_vec_2.v = _mm512_andnot_ps(abs_mask.v, x_vec_2.v);
            // Remove nan and replace with -ve values
            REMOVE_NAN_512S(x_vec_2.v);
            // Increment x vector as we have loaded 16 values
            x += num_vector_elements;
            // Increment the index vector to point to next indexes.
            ind_vec_2.v = _mm512_add_ps(maxInd_vec_2.v, inc_vec.v);

            // Mask generation for last loaded 2 elements into x and max vectors.
            // mask = max_vec_2 - x_vec_2
            mask.v = _mm512_sub_ps(max_vec_2.v, x_vec_2.v);
            // Type cast mask from IEEE754 (float) to integer type
            // This operation will not need a new register, its just to convince
            // the compiler. But its accounted as separate register in the
            // above calculations
            intMask = _mm512_castps_si512(mask.v);
            // Extract the signbit and build the mask.
            mask_vec_2 = _mm512_movepi32_mask(intMask);

            // Load 2 more elements to 3rd max and x vector, set indexes
            // Load Value x values
            max_vec_3.v = _mm512_loadu_ps(x);
            // max_vec_3 = abs(max_vec_3)
            max_vec_3.v = _mm512_andnot_ps(abs_mask.v, max_vec_3.v);
            // Remove nan and replace with -ve values
            REMOVE_NAN_512S(max_vec_3.v);
            // Increment x vector as we have loaded 16 values
            x += num_vector_elements;
            // Increment the index vector to point to next indexes.
            maxInd_vec_3.v = _mm512_add_ps(ind_vec_2.v, inc_vec.v);
            // Load Value x values
            x_vec_3.v = _mm512_loadu_ps(x);
            // x_vec_3 = abs(x_vec_3)
            x_vec_3.v = _mm512_andnot_ps(abs_mask.v, x_vec_3.v);
            // Remove nan and replace with -ve values
            REMOVE_NAN_512S(x_vec_3.v);
            // Increment x vector as we have loaded 16 values
            x += num_vector_elements;
            // Increment the index vector to point to next indexes.
            ind_vec_3.v = _mm512_add_ps(maxInd_vec_3.v, inc_vec.v);

            // Mask generation for last 2 elements loaded into x and max vectors.
            // mask = max_vec_3 - x_vec_3
            mask.v = _mm512_sub_ps(max_vec_3.v, x_vec_3.v);
            // Type cast mask from IEEE754 (float) to integer type
            // This operation will not need a new register, its just to convince
            // the compiler. But its accounted as separate register in the
            // above calculations
            intMask = _mm512_castps_si512(mask.v);
            // Extract the signbit and build the mask.
            mask_vec_3 = _mm512_movepi32_mask(intMask);

            // Blend max vector and index vector (3 pairs of elements needs to be blended).
            /* Take values from max_vector if corresponding bit in mask_vector is 0
            * otherwise take value from x_vector, this is accumulated maximum value
            * from max_vector and x_vector to mask_vector */
            max_vec_1.v = _mm512_mask_blend_ps(mask_vec_1,
                                               max_vec_1.v,
                                               x_vec_1.v);
            /* Take values from max_vector if corresponding bit in mask_vector is 0
            * otherwise take value from x_vector, this is accumulated maximum value
            * from max_vector and x_vector to mask_vector */
            max_vec_2.v = _mm512_mask_blend_ps(mask_vec_2,
                                               max_vec_2.v,
                                               x_vec_2.v);
            /* Take values from max_vector if corresponding bit in mask_vector is 0
            * otherwise take value from x_vector, this is accumulated maximum value
            * from max_vector and x_vector to mask_vector */
            max_vec_3.v = _mm512_mask_blend_ps(mask_vec_3,
                                               max_vec_3.v,
                                               x_vec_3.v);
            /* Take values from maxIndex_vector if corresponding bit in mask_vector
            * is 0 otherwise take value from index_vec_1, this is accumulated
            * maximum value index from maxIndex_vector and index_vec_1
            * to maxIndex_vector */
            maxInd_vec_1.v = _mm512_mask_blend_ps(mask_vec_1,
                                                  maxInd_vec_1.v,
                                                  index_vec_1.v);
            /* Take values from maxIndex_vector if corresponding bit in mask_vector
            * is 0 otherwise take value from index_vec_1, this is accumulated
            * maximum value index from maxIndex_vector and index_vec_1
            * to maxIndex_vector */
            maxInd_vec_2.v = _mm512_mask_blend_ps(mask_vec_2,
                                                  maxInd_vec_2.v,
                                                  ind_vec_2.v);
            /* Take values from maxIndex_vector if corresponding bit in mask_vector
            * is 0 otherwise take value from index_vec_1, this is accumulated
            * maximum value index from maxIndex_vector and index_vec_1
            * to maxIndex_vector */
            maxInd_vec_3.v = _mm512_mask_blend_ps(mask_vec_3,
                                                  maxInd_vec_3.v,
                                                  ind_vec_3.v);

            // Mask generation for blending max_vec_2 and max_vec_3 to max_vec_2.
            // mask = max_vec_2 - max_vec_3
            mask.v = _mm512_sub_ps(max_vec_2.v, max_vec_3.v);
            // Type cast mask from IEEE754 (float) to integer type
            // This operation will not need a new register, its just to convince
            // the compiler. But its accounted as separate register in the
            // above calculations
            intMask = _mm512_castps_si512(mask.v);
            // Extract the signbit and build the mask.
            mask_vec_2 = _mm512_movepi32_mask(intMask);

            // Blend to obtain 1 vector each of max values and index.
            /* Take values from max_vec_2 if corresponding bit in mask_vec_2
            * is 0 otherwise take value from max_vec_3, this is accumulated
            * maximum value from max_vec_2 and max_vec_3 to mask_vec_2 */
            max_vec_2.v = _mm512_mask_blend_ps(mask_vec_2,
                                               max_vec_2.v,
                                               max_vec_3.v);
            /* Take values from maxInd_vec_2 if corresponding bit in mask_vector
            * is 0 otherwise take value from maxInd_vec_3, this is accumulated
            * maximum value index from maxInd_vec_2 and maxInd_vec_3
            * to maxInd_vec_2 */
            maxInd_vec_2.v = _mm512_mask_blend_ps(mask_vec_2,
                                                  maxInd_vec_2.v,
                                                  maxInd_vec_3.v);

            // Mask generation for blending max_vec_1 and max_vec_2 into max_vec_1.
            // mask = max_vec_1 - max_vec_2
            mask.v = _mm512_sub_ps(max_vec_1.v, max_vec_2.v);
            // Type cast mask from IEEE754 (float) to integer type
            // This operation will not need a new register, its just to convince
            // the compiler. But its accounted as separate register in the
            // above calculations
            intMask = _mm512_castps_si512(mask.v);
            // Extract the signbit and build the mask.
            mask_vec_1 = _mm512_movepi32_mask(intMask);

            // Final blend to the master max_vec_1 and maxInd_vec_1
            /* Take values from max_vec_1 if corresponding bit in mask_vec_1
            * is 0 otherwise take value from max_vec_2, this is accumulated
            * maximum value from max_vec_1 and max_vec_2 to mask_vec_1 */
            max_vec_1.v = _mm512_mask_blend_ps(mask_vec_1, max_vec_1.v, max_vec_2.v);
            /* Take values from maxInd_vec_1 if corresponding bit in mask_vector
            * is 0 otherwise take value from maxInd_vec_2, this is accumulated
            * maximum value index from maxInd_vec_1 and maxInd_vec_2
            * to maxInd_vec_1 */
            maxInd_vec_1.v = _mm512_mask_blend_ps(mask_vec_1,
                                                  maxInd_vec_1.v,
                                                  maxInd_vec_2.v);

            // Increment the index vector to point to next indexes.
            index_vec_1.v = _mm512_add_ps(ind_vec_3.v, inc_vec.v);
        }

        for (; i < num_iter; i++)
        {
            /*
                Take vector one by one, above code makes max_vec_1
                contain the first 16 elements, now with the max vector
                as first 16 elements (abs), we need to load next 16 elements
                into x_vec_1 (abs). Now with those we can safely removeNan
                which will put -ve values as NAN.

                These -ve values of NAN decreases by 1 in each iteration,
                this helps us find the first NAN value.
            */
            // Load Value x values
            x_vec_1.v = _mm512_loadu_ps(x);
            // x_vec_1 = abs(x_vec_1)
            x_vec_1.v = _mm512_andnot_ps(abs_mask.v, x_vec_1.v);
            // Remove nan and replace with -ve values
            REMOVE_NAN_512S(x_vec_1.v);

            // Mask Generation
            // mask = max_vec_1 - x_vec_1
            mask.v = _mm512_sub_ps(max_vec_1.v, x_vec_1.v);
            // Extract the signbit and build the mask.
            mask_vec_1 = _mm512_movepi32_mask(_mm512_castps_si512(mask.v));
            /* Take values from max_vec_1 if corresponding bit in
            * mask_vec_1 is 0 otherwise take value from x_vec_1,
            * this is accumulated maximum value from max_vec_1 and
            * x_vec_1 to mask_vec_1 */
            max_vec_1.v = _mm512_mask_blend_ps(mask_vec_1,
                                               max_vec_1.v,
                                               x_vec_1.v);
            /* Take values from maxInd_vec_1 if corresponding bit in
            * mask_vector is 0 otherwise take value from index_vec_1,
            * this is accumulated maximum value index from maxInd_vec_1
            * and index_vec_1 to maxInd_vec_1 */
            maxInd_vec_1.v = _mm512_mask_blend_ps(mask_vec_1,
                                                  maxInd_vec_1.v,
                                                  index_vec_1.v);

            // Increment the index vector to point to next indexes.
            index_vec_1.v = _mm512_add_ps(index_vec_1.v, inc_vec.v);

            // Increment x vector as we have loaded 16 values
            x += num_vector_elements;
        }

        num_remain = (n - ((i)*16));

        /*
            Now take the max vector and produce the max value from
            the max vector by slicing and comparing with itself,
            until we are left with just one index position and max value.
        */
        // Split max to hi and lo
        max_vec_hi.v = _mm512_extractf32x8_ps(max_vec_1.v, 1);
        max_vec_lo.v = _mm512_extractf32x8_ps(max_vec_1.v, 0);

        // Split maxIndex to hi and lo
        maxInd_vec_hi.v = _mm512_extractf32x8_ps(maxInd_vec_1.v, 1);
        maxInd_vec_lo.v = _mm512_extractf32x8_ps(maxInd_vec_1.v, 0);

        // Compare max_vec_hi > max_vec_1
        // mask_vec_lo = max_vec_lo - max_vec_hi
        mask_vec_lo.v = _mm256_sub_ps(max_vec_lo.v, max_vec_hi.v);

        /* Take values from max_vec_lo if corresponding bit in mask_vec_lo
        * is 0 otherwise take value from max_vec_hi, this is accumulated
        * maximum value from max_vec_lo and max_vec_hi to max_vec_lo */
        max_vec_lo.v = _mm256_blendv_ps(max_vec_lo.v,
                                        max_vec_hi.v,
                                        mask_vec_lo.v);
        /* Take values from maxInd_vec_lo if corresponding bit
        * in mask_vec_lo is 0 otherwise take value from maxInd_vec_hi,
        * this is accumulated maximum value from maxInd_vec_lo and
        * maxInd_vec_hi to maxInd_vec_lo */
        maxInd_vec_lo.v = _mm256_blendv_ps(maxInd_vec_lo.v,
                                           maxInd_vec_hi.v,
                                           mask_vec_lo.v);

        // Split max_lo to hi and lo
        max_vec_lo_hi.v = _mm256_extractf128_ps(max_vec_lo.v, 1);
        max_vec_lo_lo.v = _mm256_extractf128_ps(max_vec_lo.v, 0);

        // Split maxIndex_lo to hi and lo
        maxInd_vec_lo_hi.v = _mm256_extractf128_ps(maxInd_vec_lo.v, 1);
        maxInd_vec_lo_lo.v = _mm256_extractf128_ps(maxInd_vec_lo.v, 0);

        // mask_vec_lo_lo = max_vec_lo_lo - max_vec_lo_hi
        mask_vec_lo_lo.v = _mm_sub_ps(max_vec_lo_lo.v, max_vec_lo_hi.v);
        /* Take values from max_vec_lo_lo if corresponding bit in
        * mask_vec_lo_lo is 0 otherwise take value from max_vec_lo_hi,
        * this is accumulated maximum value from max_vec_lo_lo and
        * max_vec_lo_hi to max_vec_lo_lo */
        max_vec_lo_lo.v = _mm_blendv_ps(max_vec_lo_lo.v,
                                        max_vec_lo_hi.v,
                                        mask_vec_lo_lo.v);
        /* Take values from maxInd_vec_lo if corresponding bit
        * in mask_vec_lo_lo is 0 otherwise take value from maxInd_vec_hi,
        * this is accumulated maximum value from maxInd_vec_lo and
        * maxInd_vec_hi to maxInd_vec_lo */
        maxInd_vec_lo_lo.v = _mm_blendv_ps(maxInd_vec_lo_lo.v,
                                           maxInd_vec_lo_hi.v,
                                           mask_vec_lo_lo.v);

        // Take 64 high bits of max_lo_lo and put it to 64 low bits, rest 1st value
        /* Example max_vec_lo_lo is {a, b, x, y}
        * After max_vec_lo_hi.v = _mm_permute_ps(max_vec_lo_lo.v, 14);
        * max_vec_lo_hi is {x, y, a, a} (essentially folding the vector)
        */
        max_vec_lo_hi.v = _mm_permute_ps(max_vec_lo_lo.v, 14);
        // Fold the vector same as max_vector
        maxInd_vec_lo_hi.v = _mm_permute_ps(maxInd_vec_lo_lo.v, 14);

        // mask_vec_lo_lo = max_vec_lo_lo - max_vec_lo_hi
        mask_vec_lo_lo.v = _mm_sub_ps(max_vec_lo_lo.v, max_vec_lo_hi.v);
        /* Take values from max_vec_lo_lo if corresponding bit in
        * mask_vec_lo_lo is 0 otherwise take value from max_vec_lo_hi,
        * this is accumulated maximum value from max_vec_lo_lo and
        * max_vec_lo_hi to max_vec_lo_lo */
        max_vec_lo_lo.v = _mm_blendv_ps(max_vec_lo_lo.v,
                                        max_vec_lo_hi.v,
                                        mask_vec_lo_lo.v);
        /* Take values from maxInd_vec_lo if corresponding bit
        * in mask_vec_lo_lo is 0 otherwise take value from maxInd_vec_hi,
        * this is accumulated maximum value from maxInd_vec_lo and
        * maxInd_vec_hi to maxInd_vec_lo */
        maxInd_vec_lo_lo.v = _mm_blendv_ps(maxInd_vec_lo_lo.v,
                                           maxInd_vec_lo_hi.v,
                                           mask_vec_lo_lo.v);

        // Take max_vec_lo_lo.f[1] and put it to max_vec_lo_hi.f[0]
        /* Example max_vec_lo_lo is {a, b, x, y}
        * After max_vec_lo_hi.v = _mm_permute_ps(max_vec_lo_lo.v, 1);
        * max_vec_lo_hi is {b, a, a, a} (essentially folding the vector)
        */
        max_vec_lo_hi.v = _mm_permute_ps(max_vec_lo_lo.v, 1);
        // Do the same operation.
        maxInd_vec_lo_hi.v = _mm_permute_ps(maxInd_vec_lo_lo.v, 1);

        // mask_vec_lo_lo = max_vec_lo_lo - max_vec_lo_hi
        mask_vec_lo_lo.v = _mm_sub_ps(max_vec_lo_lo.v, max_vec_lo_hi.v);
        /* Take values from max_vec_lo_lo if corresponding bit in
        * mask_vec_lo_lo is 0 otherwise take value from max_vec_lo_hi,
        * this is accumulated maximum value from max_vec_lo_lo and
        * max_vec_lo_hi to max_vec_lo_lo */
        max_vec_lo_lo.v = _mm_blendv_ps(max_vec_lo_lo.v,
                                        max_vec_lo_hi.v,
                                        mask_vec_lo_lo.v);
        /* Take values from maxInd_vec_lo if corresponding bit
        * in mask_vec_lo_lo is 0 otherwise take value from maxInd_vec_hi,
        * this is accumulated maximum value from maxInd_vec_lo and
        * maxInd_vec_hi to maxInd_vec_lo */
        maxInd_vec_lo_lo.v = _mm_blendv_ps(maxInd_vec_lo_lo.v,
                                           maxInd_vec_lo_hi.v,
                                           mask_vec_lo_lo.v);
        /* We have kept on folding and comparing until we got one single index
        * and max value so that is the final answer so set it as the final
        * answer.*/
        fndInd = maxInd_vec_lo_lo.f[0];
        fndMaxVal = max_vec_lo_lo.f[0];
        // Found value is < 0 means it was the max NAN which was accumulated.
        if (fndMaxVal < 0)
        {
            // So just set it as NAN
            fndMaxVal = NAN;
        }
        // Finish off the remaining values using normal instructions
        for (dim_t i = n - num_remain; i < n; i++)
        {
            float absval = fabsf(*(x));
            if (fndMaxVal < absval || (isnan(absval) && !isnan(fndMaxVal)))
            {
                fndMaxVal = absval;
                fndInd = i;
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
    *i_max = fndInd;
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

//----------------------------------------------------------------------------------------------------
void bli_damaxv_zen_int_avx512(
    dim_t n,
    double *restrict x, inc_t incx,
    dim_t *restrict i_max,
    cntx_t *restrict cntx)
{
    AOCL_DTL_TRACE_EXIT(AOCL_DTL_LEVEL_TRACE_3)
    double *minus_one = PASTEMAC(d, m1);

    // Used to replace NAN in registers. This value is decremented each time
    // remove NAN is applied so as to keep the NAN value replacements unique.
    double nan_repl = -1.0;

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
        REMOVE_NAN_512D(max_array.v);

        // Initializing the maximumum index
        max_ind.v = _mm512_set_pd(7, 6, 5, 4, 3, 2, 1, 0);
        x += vector_length;

        //Incrementing to make the vector
        //to point to the next 8 elements
        zmm4_Ind.v = _mm512_add_pd(zmm4_Ind.v, inc_vec.v);

        /*    Loop unrolled by a factor of 4
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
            REMOVE_NAN_512D(zmm0.v);
            x += vector_length;

            zmm1.v = _mm512_loadu_pd(x);
            zmm5_Ind.v = _mm512_add_pd(zmm4_Ind.v, inc_vec.v);
            zmm1.v = _mm512_andnot_pd(sign_mask, zmm1.v);
            REMOVE_NAN_512D(zmm1.v);
            x += vector_length;

            zmm2.v = _mm512_loadu_pd(x);
            zmm6_Ind.v = _mm512_add_pd(zmm5_Ind.v, inc_vec.v);
            zmm2.v = _mm512_andnot_pd(sign_mask, zmm2.v);
            REMOVE_NAN_512D(zmm2.v);
            x += vector_length;

            zmm3.v = _mm512_loadu_pd(x);
            zmm7_Ind.v = _mm512_add_pd(zmm6_Ind.v, inc_vec.v);
            zmm3.v = _mm512_andnot_pd(sign_mask, zmm3.v);
            REMOVE_NAN_512D(zmm3.v);
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
        for (dim_t i = 0; i < iterations; ++i)
        {
            // Taking 32 elements
            // Taking only the absolute values of the registers
            // Removing the NAN values and replacing it
            // with negative numbers
            zmm0.v = _mm512_loadu_pd(x);
            zmm0.v = _mm512_abs_pd(zmm0.v);
            REMOVE_NAN_512D(zmm0.v);

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

    // Issue vzeroupper instruction to clear upper lanes of ymm registers.
    // This avoids a performance penalty caused by false dependencies when
    // transitioning from AVX to SSE instructions (which may occur later,
    // especially if BLIS is compiled with -mfpmath=sse).
    _mm256_zeroupper();

    // Return value
    *i_max = i_max_l;

    AOCL_DTL_TRACE_EXIT(AOCL_DTL_LEVEL_TRACE_3)
}
