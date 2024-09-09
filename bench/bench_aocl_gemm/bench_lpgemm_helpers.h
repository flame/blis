/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2024, Advanced Micro Devices, Inc. All rights reserved.

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

#ifndef LPGEMM_BENCH_UTILS_H
#define LPGEMM_BENCH_UTILS_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <immintrin.h>
#include <time.h>
#include <float.h>
#include <math.h>

#include "blis.h"

// Used to clip downscaled output, will be set in the main loop based
// on the accumulation and C data type.
int64_t DSCALE_CLIP_MIN = 0;
int64_t DSCALE_CLIP_MAX = 0;

// Mode can be one of the follwoing:
// 1. p - performance, used for benchmarks.
// 2. a - accuracy, used to test accuracy/correctness.
// Default value is p, can be modified by passing command line arg.
char bench_mode = 'p';

int32_t global_n_repeat = 0;

char global_dscale_out = 'n';

dim_t num_eltwise = 0; // To keep track of eltwise operations.

#define _XSTR(str) #str
#define XSTR(str) _XSTR(str)

#define GEN_FUNC_NAME(prototype,ctype) prototype ## ctype

// Inplace to lower func.
static inline void str_tolower( char* str )
{
    for ( char* c = str; ( *c ) != '\0'; ++c )
    { *( c ) = tolower( *( c ) ); }
}

#define CONVERT_TO_FLOAT(ctype) \
static inline void GEN_FUNC_NAME(ctype,_to_float) ( ctype val, float* float_val ) \
{ \
    *float_val = (float) val; \
} \

static inline void float_to_bf16( float* float_value, bfloat16* bf16_val )
{
    /*Set offset 2 to copy most significant 2 bytes of float
    to convert float values to bf16 values*/
    memcpy( ( bf16_val ), (char *)( float_value ) + 2, sizeof ( bfloat16 ) );
}

// Only works for little endian systems.
static inline void bfloat16_to_float( bfloat16 bf16_val, float*  float_val )
{
    int32_t inter_temp = *( ( int16_t* ) &bf16_val );
    inter_temp = inter_temp << 16;
    memcpy( float_val, &inter_temp, sizeof( int32_t ) );
}

static inline void convert_float_arr_to_bf16( float* array, bfloat16* array_bf16, dim_t size )
{
    for (dim_t i=0; i< size; i++)
    {
        float_to_bf16( ( array + i ), ( array_bf16 + i ) );
    }
}

static inline void* lpgemm_malloc( dim_t size )
{
    void* p;
    // creating a dummy buffer of size 4 bytes in case
    // size of the matrix is negative.
    if( size <= 0 )
    {
        p = malloc( 4 );
        return p;
    }

    if( bench_mode == 'a' )
    {
        p = malloc(size);
    }
    else
    {
        err_t err = BLIS_SUCCESS;
        p = bli_malloc_user(size, &err);
    }
    if ( p == NULL )
    {
        printf("Unable to allocate memory.\n");
        exit(1);
    }
    return p;
}

static inline void lpgemm_free( void* p )
{
    if( p == NULL)
    {
        printf("Attempt to free null pointer\n");
        return;
    }

    if( bench_mode == 'a' )
    {
        free(p);
    }
    else
    {
        bli_free_user(p);
    }
}

/* Matrix fill helper macros. */
#define GEN_FILL_ARRAY_FUNC(ctype) \
static inline void fill_array_ ## ctype ( void* arr, dim_t size ) \
{ \
    if( size < 0 ) return; \
    ctype* temp_arr = ( ctype* ) arr; \
    for ( dim_t i = 0; i < size; ++i ) \
    { \
        temp_arr[i] = ( ctype )( ( rand() % 11 ) - 5 ); \
    } \
} \

static inline void fill_array_bfloat16( void* arr, dim_t size )
{
    err_t bli_errors = BLIS_SUCCESS;
    if( size < 0 ) return;
    float* c_float = ( float* ) bli_malloc_user( sizeof( float ) * size, &bli_errors );
    for ( dim_t i = 0; i < size; ++i )
    {
        c_float[i] = (rand() % 5 );
    }
    convert_float_arr_to_bf16( c_float, arr, size );
    if ( c_float != NULL )
    {
        bli_free_user( c_float );
    }
}

#define GEN_FILL_ARRAY_POST_OPS_FUNC(ctype) \
static inline void fill_array_post_ops_ ## ctype ( void* arr, dim_t size ) \
{ \
    ctype* temp_arr = ( ctype* ) arr; \
    for ( dim_t i = 0; i < size; ++i ) \
    { \
        temp_arr[i] = ( ctype )( rand() % 5 ); \
    } \
} \

static inline void fill_array_post_ops_bfloat16( void* arr, dim_t size )
{
    fill_array_bfloat16( arr, size );
}

/* POST-OPS Helper macros. */

/* Bias. */
#define GEN_GET_BIAS_POST_OP_VAL_BF16(BLAS_SFX) \
static inline float get_bias_post_op_val_ ## BLAS_SFX \
     ( \
       void* post_op_bias_ptr, \
       dim_t j \
     ) \
{ \
    float ret_val = 0.0; \
    bfloat16_to_float( *( ( bfloat16* )post_op_bias_ptr + j ), &ret_val ); \
    return ret_val; \
} \

#define GEN_GET_BIAS_POST_OP_VAL(ACCUM_type,BLAS_SFX) \
static inline ACCUM_type get_bias_post_op_val_ ## BLAS_SFX \
     ( \
       void* post_op_bias_ptr, \
       dim_t j \
     ) \
{ \
    return *( ( ACCUM_type* )post_op_bias_ptr + j ); \
} \

/* GELU Tanh. */
#define GEN_GELU_TANH_POSTOP_INT(ACCUM_type,BLAS_SFX) \
static inline ACCUM_type GELU_TANH_post_op_ ## BLAS_SFX \
     ( \
       ACCUM_type temp_accum \
     ) \
{ \
    float gelu_reference = 0.5 *(double)temp_accum * (1 + tanhf( 0.797884 * ( (double)temp_accum + \
                    ( 0.044715 * ((double)temp_accum * (double)temp_accum * \
                    (double)temp_accum ) ) ) ) ); \
    temp_accum = round (gelu_reference); \
    return temp_accum; \
} \

#define GEN_GELU_TANH_POSTOP_FLOAT(BLAS_SFX) \
static inline float GELU_TANH_post_op_ ## BLAS_SFX \
     ( \
       float temp_accum \
     ) \
{ \
    temp_accum = 0.5 *(double)temp_accum * (1 + tanhf( 0.797884 * ( (double)temp_accum + \
                  ( 0.044715 * ((double)temp_accum * (double)temp_accum * \
                  (double)temp_accum ) ) ) ) ); \
    return temp_accum; \
} \

/* GELU Erf. */
#define GEN_GELU_ERF_POSTOP_INT(ACCUM_type,BLAS_SFX) \
static inline ACCUM_type GELU_ERF_post_op_ ## BLAS_SFX \
     ( \
       ACCUM_type temp_accum \
     ) \
{ \
    float gelu_reference = 0.5 *(double)temp_accum * (1 + erff( (double)temp_accum * 0.707107 )); \
    temp_accum = round (gelu_reference); \
    return temp_accum; \
} \

#define GEN_GELU_ERF_POSTOP_FLOAT(BLAS_SFX) \
static inline float GELU_ERF_post_op_ ## BLAS_SFX \
     ( \
       float temp_accum \
     ) \
{ \
    temp_accum = 0.5 *(double)temp_accum * (1 + erff( (double)temp_accum * 0.707107 )); \
    return temp_accum; \
} \

/* SWISH. */
#define GEN_SWISH_POSTOP_INT(ACCUM_type,BLAS_SFX) \
static inline ACCUM_type SWISH_post_op_ ## BLAS_SFX \
     ( \
       ACCUM_type temp_accum, \
       ACCUM_type alpha \
     ) \
{ \
    float swish_reference = ( temp_accum / ( 1 + \
                            expf( ( double )alpha * temp_accum * -1 ) ) ); \
    temp_accum = round (swish_reference); \
    return temp_accum; \
} \

#define GEN_SWISH_POSTOP_FLOAT(BLAS_SFX) \
static inline float SWISH_post_op_ ## BLAS_SFX \
     ( \
       float temp_accum, \
       float alpha \
     ) \
{ \
    temp_accum = ( temp_accum / ( 1 + \
                  expf( ( double )alpha * temp_accum * -1 ) ) ); \
    return temp_accum; \
} \

/* Matrix Add. */
#define GEN_GET_MATRIX_ADD_POST_OP_VAL_BF16(C_type,BLAS_SFX) \
static inline float get_matrix_add_post_op_val_ ## BLAS_SFX \
     ( \
       C_type val \
     ) \
{ \
    float ret_val = 0.0; \
    bfloat16_to_float( val, &ret_val ); \
    return ret_val; \
} \

#define GEN_GET_MATRIX_ADD_POST_OP_VAL(C_type,ACCUM_type,BLAS_SFX) \
static inline ACCUM_type get_matrix_add_post_op_val_ ## BLAS_SFX \
     ( \
       C_type val \
     ) \
{ \
    return (ACCUM_type) val; \
} \

#define GEN_GET_MATRIX_MUL_POST_OP_VAL_BF16(C_type,BLAS_SFX) \
static inline float get_matrix_mul_post_op_val_ ## BLAS_SFX \
     ( \
       C_type val \
     ) \
{ \
    float ret_val = 0.0; \
    bfloat16_to_float( val, &ret_val ); \
    return ret_val; \
} \

#define GEN_GET_MATRIX_MUL_POST_OP_VAL(C_type,ACCUM_type,BLAS_SFX) \
static inline ACCUM_type get_matrix_mul_post_op_val_ ## BLAS_SFX \
     ( \
       C_type val \
     ) \
{ \
    return (ACCUM_type) val; \
} \

/* Final output type value getter. */
#define GEN_MAT_MUL_GET_OUTPUT_TYPE_VALUE(C_type, ACCUM_type) \
static inline void mat_mul_get_output_type_val ## ACCUM_type ## C_type \
     ( \
       C_type* out_temp_accum, \
       ACCUM_type* temp_accum \
     ) \
{ \
    ( *out_temp_accum ) = ( C_type )( *temp_accum ); \
} \

static inline void mat_mul_get_output_type_valfloatbfloat16
     (
       bfloat16* out_temp_accum,
       float* temp_accum
     )
{
	/* Fix for rounding bias. */
	uint32_t inter_temp;
	memcpy( &inter_temp, temp_accum, sizeof( float ) );

	/* Check if 16th bit is set */
	uint32_t tlsb = ( inter_temp & ( uint32_t )0x00010000 ) > 16;

	/* Adding rounding bias. */
	uint32_t rounded = inter_temp + ( uint32_t )0x00007FFF + tlsb;
	memcpy( temp_accum, &rounded, sizeof( float ) );

    float_to_bf16( temp_accum, out_temp_accum );
}

#ifndef WIN32
static inline int max (int a, int b)
{
    return ( a > b ? a : b );
}

static inline int min (int a, int b)
{
    return ( a < b ? a : b );
}
#endif

static inline void lpgemm_destroy_post_ops_struct( aocl_post_op* post_ops )
{
    if ( post_ops == NULL )
    {
        return;
    }

    if ( post_ops->eltwise != NULL )
    {
        for ( dim_t i = 0; i < num_eltwise; ++i )
        {
            free( ( post_ops->eltwise + i )->algo.alpha );
            free( ( post_ops->eltwise + i )->algo.beta );
        }
        free( post_ops->eltwise );
    }

    if ( post_ops->matrix_add != NULL )
    {
        free( ( post_ops->matrix_add )->matrix );
        free( post_ops->matrix_add );
    }

    if ( post_ops->sum != NULL )
    {
        free( ( post_ops->sum )->scale_factor );
        free( ( post_ops->sum )->zero_point );
        free( post_ops->sum );
    }

    if ( post_ops->matrix_mul != NULL )
    {
        free( ( post_ops->matrix_mul )->matrix );
        free( post_ops->matrix_mul );
    }

    if ( post_ops->bias != NULL )
    {
        free( ( post_ops->bias )->bias );
        free( post_ops->bias );
    }

    if ( post_ops->pre_ops != NULL )
    {
        if ( ( post_ops->pre_ops )->b_zp != NULL )
        {
            free( ( ( post_ops->pre_ops )->b_zp )->zero_point );
            free( ( post_ops->pre_ops )->b_zp );
        }
        if ( ( post_ops->pre_ops )->b_scl != NULL )
        {
            free( ( ( post_ops->pre_ops )->b_scl )->scale_factor );
            free( ( post_ops->pre_ops )->b_scl );
        }
        free( post_ops->pre_ops );
    }

    free( post_ops->seq_vector );
    free( post_ops );
}

#endif //LPGEMM_BENCH_UTILS_H
