/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2024 - 2025, Advanced Micro Devices, Inc. All rights reserved.

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
#include <omp.h>
#include <limits.h>

#include "blis.h"

// Used to clip downscaled output, will be set in the main loop based
// on the accumulation and C data type.
int64_t DSCALE_CLIP_MIN = INT_MIN;
int64_t DSCALE_CLIP_MAX = INT_MAX;

// Mode can be one of the follwoing:
// 1. p - performance, used for benchmarks.
// 2. a - accuracy, used to test accuracy/correctness.
// Default value is p, can be modified by passing command line arg.
char bench_mode = 'p';

int32_t global_n_repeat = 0;

char global_dscale_out = 'n';
char global_can_dscale = 'n';
char global_pre_op = 'n';


#define _XSTR(str) #str
#define XSTR(str) _XSTR(str)

#define GEN_FUNC_NAME(prototype,ctype) prototype ## ctype
#define CVT_FUNC_NAME(stype, dtype) stype ## _to_ ## dtype
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

CONVERT_TO_FLOAT(uint8_t)
CONVERT_TO_FLOAT(int8_t)
CONVERT_TO_FLOAT(int16_t)
CONVERT_TO_FLOAT(float)
CONVERT_TO_FLOAT(int32_t)


#define CONVERT_ITSELF(ctype) \
static inline void GEN_FUNC_NAME(ctype,_to_ ## ctype) ( ctype val, ctype* ctype_val ) \
{ \
    *ctype_val = val; \
}

CONVERT_ITSELF(int16_t)
CONVERT_ITSELF(int32_t)

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

static inline void bfloat16_to_int32_t( bfloat16 bf16_val, int32_t*  int_val )
{
    int32_t inter_temp = *( ( int16_t* ) &bf16_val );
    inter_temp = inter_temp << 16;
    float float_val = 0.0;
    memcpy(&float_val, &inter_temp, sizeof(float));
    *int_val = ( int32_t )float_val;
}

static inline void int8_t_to_int32_t( int8_t int8_t_val, int32_t*  int_val )
{
   *int_val = ( int32_t )int8_t_val;
}

static inline void convert_float_arr_to_bf16( float* array, bfloat16* array_bf16, dim_t size )
{
#ifdef BLIS_ENABLE_OPENMP
    #pragma omp parallel for
#endif
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

bool is_integerAPI_avx512( char* api_name )
{
    if ( ( strcmp( api_name, "u8s8s32of32" ) == 0) || ( strcmp( api_name, "u8s8s32os8" ) == 0) \
    || ( strcmp( api_name, "u8s8s32obf16" ) == 0) || ( strcmp( api_name, "u8s8s32os32" ) == 0) \
    || ( strcmp( api_name, "u8s8s32ou8" ) == 0) || ( strcmp( api_name, "s8s8s32of32" ) == 0) \
    || ( strcmp( api_name, "s8s8s32os8" ) == 0) || ( strcmp( api_name, "s8s8s32obf16" ) == 0) \
    || ( strcmp( api_name, "s8s8s32os32" ) == 0) || ( strcmp( api_name, "s8s8s32ou8" ) == 0)) \
    { \
        return TRUE; \
    } \
    else \
    { \
        return FALSE; \
    }
}
bool is_integer( char* type )
{
    if ( ( strcmp( type, "int8_t" ) == 0 ) || ( strcmp( type, "int16_t" ) == 0 ) \
    || ( strcmp( type, "int32_t" ) == 0 ) || ( strcmp( type, "uint8_t" ) == 0 ) ) \
    { \
        return TRUE; \
    } \
    else \
    { \
        return FALSE; \
    }
}
bool is_bf16API_avx512( char* api_name )
{
    if ( ( strcmp( api_name, "bf16bf16f32of32" ) == 0) || ( strcmp( api_name, "bf16bf16f32obf16" ) == 0) \
    || ( strcmp( api_name, "bf16s4f32of32" ) == 0) || strcmp( api_name, "bf16s4f32obf16")) \
    { \
        return TRUE; \
    } \
    else \
    { \
        return FALSE; \
    }
}
#ifdef BLIS_ENABLE_OPENMP
/* Matrix fill helper macros. */
#define GEN_FILL_ARRAY_FUNC(ctype) \
static inline void fill_array_ ## ctype ( void* arr, dim_t size ) \
{ \
    if( size < 0 ) return; \
    ctype* temp_arr = ( ctype* ) arr; \
    _Pragma( "omp parallel for" ) \
    for ( dim_t i = 0; i < size; ++i ) \
    { \
        temp_arr[i] = ( ctype )( ( i % 11 ) - 5 ); \
    } \
}
#else
/* Matrix fill helper macros. */
#define GEN_FILL_ARRAY_FUNC(ctype) \
static inline void fill_array_ ## ctype ( void* arr, dim_t size ) \
{ \
    if( size < 0 ) return; \
    ctype* temp_arr = ( ctype* ) arr; \
    for ( dim_t i = 0; i < size; ++i ) \
    { \
        temp_arr[i] = ( ctype )( ( i % 11 ) - 5 ); \
    } \
}

#endif
static inline void fill_array_bfloat16( void* arr, dim_t size )
{
    err_t bli_errors = BLIS_SUCCESS;
    if( size < 0 ) return;
    float* c_float = ( float* ) bli_malloc_user( sizeof( float ) * size, &bli_errors );
#ifdef BLIS_ENABLE_OPENMP
    #pragma omp parallel for
#endif
    for ( dim_t i = 0; i < size; ++i )
    {
        c_float[i] = (float)(i % 5);
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
        temp_arr[i] = ( ctype )( i % 5 ); \
    } \
} \

GEN_FILL_ARRAY_POST_OPS_FUNC(int8_t)
GEN_FILL_ARRAY_POST_OPS_FUNC(int16_t)
GEN_FILL_ARRAY_POST_OPS_FUNC(int32_t)
GEN_FILL_ARRAY_POST_OPS_FUNC(float)
GEN_FILL_ARRAY_POST_OPS_FUNC(uint8_t)

static inline void fill_array_post_ops_bfloat16( void* arr, dim_t size )
{
    fill_array_bfloat16( arr, size );
}

/* POST-OPS Helper macros. */
/* CLIP */
#define GEN_CLIP_POST_OP_VAL_INT(BLAS_SFX) \
static inline float get_clip_post_op_val_ ## BLAS_SFX \
     ( \
       float post_temp_accum, \
       void* post_op_alpha_ptr, \
       void* post_op_beta_ptr \
     ) \
{ \
    float alpha, beta; \
    int32_t_to_float(*( ( int32_t* )post_op_alpha_ptr), &alpha); \
    int32_t_to_float(*( ( int32_t* )post_op_beta_ptr), &beta); \
    return min( max( post_temp_accum, alpha),beta); \
}

#define GEN_CLIP_POST_OP_VAL_FLOAT(BLAS_SFX) \
static inline float get_clip_post_op_val_ ## BLAS_SFX \
     ( \
       float post_temp_accum, \
       void* post_op_alpha_ptr, \
       void* post_op_beta_ptr \
     ) \
{ \
    return min( max( post_temp_accum, *( ( float* )post_op_alpha_ptr ) ), \
                *( ( float* )post_op_beta_ptr ) ); \
}

/* PRELU */
#define GEN_PRELU_POST_OP_VAL_FLOAT(BLAS_SFX) \
static inline float get_prelu_post_op_val_ ## BLAS_SFX \
     ( \
       float post_temp_accum, \
       void* post_op_alpha_ptr \
     ) \
{ \
    return (( post_temp_accum > 0 ) ? \
                                post_temp_accum : \
                                ( post_temp_accum * \
                                (*( float* )post_op_alpha_ptr) )); \
}

#define GEN_PRELU_POST_OP_VAL_INT(BLAS_SFX) \
static inline float get_prelu_post_op_val_ ## BLAS_SFX \
     ( \
       float post_temp_accum, \
       void* post_op_alpha_ptr \
     ) \
{ \
    float ret_val; \
    int32_t_to_float(*( ( int32_t* )post_op_alpha_ptr), &ret_val); \
 \
    return ( post_temp_accum > 0 ) ? \
                                post_temp_accum : \
                                ( post_temp_accum * ret_val ); \
}

/* Bias. */
#define GEN_GET_BIAS_POST_OP_VAL_BF16(BLAS_SFX) \
static inline float get_bias_post_op_val_ ## BLAS_SFX \
     ( \
       void* post_op_bias_ptr, \
       dim_t j, \
       AOCL_PARAMS_STORAGE_TYPES bias_stor_type \
     ) \
{ \
    float ret_val = 0.0; \
    if(bias_stor_type == AOCL_GEMM_F32) \
    { \
        return *( ( float* )post_op_bias_ptr + j ); \
    } \
    bfloat16_to_float( *( ( bfloat16* )post_op_bias_ptr + j ), &ret_val ); \
    return ret_val; \
} \

#define GEN_GET_BIAS_POST_OP_VAL_f32(BLAS_SFX) \
static inline float get_bias_post_op_val_ ## BLAS_SFX \
     ( \
       void* post_op_bias_ptr, \
       dim_t j, \
       AOCL_PARAMS_STORAGE_TYPES bias_stor_type \
     ) \
{ \
    float ret_val = 0.0; \
    if(bias_stor_type == AOCL_GEMM_BF16) \
    { \
        bfloat16_to_float( *( ( bfloat16* )post_op_bias_ptr + j ), &ret_val ); \
        return ret_val; \
    } \
    return *( ( float* )post_op_bias_ptr + j ); \
} \

#define GEN_GET_BIAS_POST_OP_VAL(ACCUM_type,BLAS_SFX) \
static inline ACCUM_type get_bias_post_op_val_ ## BLAS_SFX \
     ( \
       void* post_op_bias_ptr, \
       dim_t j, \
       AOCL_PARAMS_STORAGE_TYPES bias_stor_type \
     ) \
{ \
    if(bias_stor_type == AOCL_GEMM_BF16) \
    { \
        float ret_val = 0.0; \
        bfloat16_to_float( *( ( bfloat16* )post_op_bias_ptr + j ), &ret_val ); \
        return ret_val; \
    } \
    if(bias_stor_type == AOCL_GEMM_INT8) \
    { \
        float ret_val = 0.0; \
        int8_t_to_float( *( ( int8_t* )post_op_bias_ptr + j ), &ret_val ); \
        return ret_val; \
    } \
    if(bias_stor_type == AOCL_GEMM_INT32) \
    { \
        float ret_val = 0.0; \
        int32_t_to_float( *( ( int32_t* )post_op_bias_ptr + j ), &ret_val ); \
        return ret_val; \
    } \
    return *( ( ACCUM_type* )post_op_bias_ptr + j ); \
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
#define GEN_GELU_ERF_POSTOP_FLOAT(BLAS_SFX) \
static inline float GELU_ERF_post_op_ ## BLAS_SFX \
     ( \
       float temp_accum \
     ) \
{ \
    temp_accum = 0.5 *(double)temp_accum * (1 + erff( (double)temp_accum * 0.707107 )); \
    return temp_accum; \
} \

/* TANH. */
#define GEN_TANH_POSTOP_FLOAT(BLAS_SFX) \
static inline float TANH_post_op_ ## BLAS_SFX \
     ( \
       float temp_accum \
     ) \
{ \
    temp_accum = tanhf( ( double )temp_accum ); \
    return temp_accum; \
} \

/* SIGMOID. */
#define GEN_SIGMOID_POSTOP_FLOAT(BLAS_SFX) \
static inline float SIGMOID_post_op_ ## BLAS_SFX \
     ( \
       float temp_accum \
     ) \
{ \
    temp_accum = ( 1 / ( 1 + \
                  expf( ( double )temp_accum * -1 ) ) ); \
    return temp_accum; \
} \

/* SWISH. */
#define GEN_SWISH_POSTOP_INT(ACCUM_type,BLAS_SFX) \
static inline ACCUM_type SWISH_post_op_ ## BLAS_SFX \
     ( \
       ACCUM_type temp_accum, \
       void* alpha \
     ) \
{ \
    float alpha_val; \
    int32_t_to_float(*( ( int32_t* )alpha), &alpha_val); \
    float swish_reference = ( temp_accum / ( 1 + \
                            expf( ( double )((alpha_val) * temp_accum * -1 )) ) ); \
    return swish_reference; \
} \

#define GEN_SWISH_POSTOP_FLOAT(BLAS_SFX) \
static inline float SWISH_post_op_ ## BLAS_SFX \
     ( \
       float temp_accum, \
       void* alpha \
     ) \
{ \
    temp_accum = ( temp_accum / ( 1 + \
                  expf( ( double )(*((float*)alpha) * temp_accum * -1 ) ) )); \
    return temp_accum; \
} \

#define GEN_GET_MATRIX_ADD_POST_OP_VAL(ACCUM_type,BLAS_SFX) \
static inline ACCUM_type get_matrix_add_post_op_val_ ## BLAS_SFX \
     ( \
       void* mat_add_ptr, \
       dim_t i, \
       dim_t j, \
       dim_t rs_m, \
       dim_t cs_m, \
       float* scl_fctr, \
       dim_t scl_fctr_len, \
       AOCL_PARAMS_STORAGE_TYPES matadd_stor_type \
     ) \
{ \
    dim_t j_scale = j; \
    if ( scl_fctr_len == 1 ) \
    { \
       j_scale = 0; \
    } \
    if( matadd_stor_type == AOCL_GEMM_BF16 ) \
    { \
        float ret_val = 0.0; \
        bfloat16 val = *( ( bfloat16* )mat_add_ptr + ( i * rs_m ) + ( j * cs_m ) ); \
        bfloat16_to_float( val, &ret_val ); \
        return ( ( float )ret_val * *( scl_fctr + j_scale ) ); \
    } \
    if( matadd_stor_type == AOCL_GEMM_INT8 ) \
    { \
        float ret_val = 0.0; \
        int8_t_to_float( *( ( int8_t* )mat_add_ptr + ( i * rs_m ) + ( j * cs_m ) ), &ret_val ); \
        return ( ( float )ret_val * *( scl_fctr + j_scale ) ); \
    } \
    if( matadd_stor_type == AOCL_GEMM_INT32 ) \
    { \
        float ret_val = 0.0; \
        int32_t_to_float( *( ( int32_t* )mat_add_ptr + ( i * rs_m ) + ( j * cs_m ) ), &ret_val ); \
        return ( ( float )ret_val * *( scl_fctr + j_scale ) ); \
    } \
    if( matadd_stor_type == AOCL_GEMM_F32 ) \
    { \
        float ret_val = 0.0; \
        ret_val = *( ( float* )mat_add_ptr + ( i * rs_m ) + ( j * cs_m ) ); \
        return ( ( float )ret_val * *( scl_fctr + j_scale ) ); \
    } \
    /* default case */ \
    if( is_integerAPI_avx512(#BLAS_SFX) ) \
    { \
        if( strcmp( #BLAS_SFX, "u8s8s32os8" ) == 0 ) \
        { \
            float ret_val = 0.0; \
            int8_t_to_float( *( ( int8_t* )mat_add_ptr + ( i * rs_m ) + ( j * cs_m ) ), &ret_val ); \
            return ( ( float )ret_val * *( scl_fctr + j_scale ) ); \
        } \
        float ret_val = 0.0; \
        int32_t_to_float( *( ( int32_t* )mat_add_ptr + ( i * rs_m ) + ( j * cs_m ) ), &ret_val ); \
        return ( ( float )ret_val * *( scl_fctr + j_scale ) ); \
    } \
    else \
    { \
        if( global_dscale_out == 'y' ) \
        { \
            float ret_val = 0.0; \
            bfloat16 val = *( ( bfloat16* )mat_add_ptr + ( i * rs_m ) + ( j * cs_m ) ); \
            bfloat16_to_float( val, &ret_val ); \
            return ( ( float )ret_val * *( scl_fctr + j_scale ) ); \
        } \
        float ret_val = 0.0; \
        ret_val = *( ( float* )mat_add_ptr + ( i * rs_m ) + ( j * cs_m ) ); \
        return ( ( float )ret_val * *( scl_fctr + j_scale ) ); \
    } \
} \

#define GEN_GET_MATRIX_MUL_POST_OP_VAL_BF16(BLAS_SFX) \
static inline float get_matrix_mul_post_op_val_ ## BLAS_SFX \
     ( \
       void* mat_add_ptr, \
       dim_t i, \
       dim_t j, \
       dim_t rs_m, \
       dim_t cs_m, \
       float* scl_fctr, \
       dim_t scl_fctr_len, \
       AOCL_PARAMS_STORAGE_TYPES matadd_stor_type \
     ) \
{ \
    return GEN_FUNC_NAME(get_matrix_add_post_op_val_,BLAS_SFX) \
            ( \
              mat_add_ptr, i, j, rs_m, cs_m, scl_fctr, scl_fctr_len, matadd_stor_type \
            ); \
} \

#define GEN_GET_MATRIX_MUL_POST_OP_VAL(ACCUM_type,BLAS_SFX) \
static inline ACCUM_type get_matrix_mul_post_op_val_ ## BLAS_SFX \
     ( \
       void* mat_add_ptr, \
       dim_t i, \
       dim_t j, \
       dim_t rs_m, \
       dim_t cs_m, \
       float* scl_fctr, \
       dim_t scl_fctr_len, \
       AOCL_PARAMS_STORAGE_TYPES matadd_stor_type \
     ) \
{ \
    return GEN_FUNC_NAME(get_matrix_add_post_op_val_,BLAS_SFX) \
            ( \
              mat_add_ptr, i, j, rs_m, cs_m, scl_fctr, scl_fctr_len, matadd_stor_type \
            ); \
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

static inline void mat_mul_get_output_type_valint32_tfloat
     (
       float* out_temp_accum,
       int32_t* temp_accum
     )
{
    ( *out_temp_accum ) = ( float )( *temp_accum );
}

static inline void mat_mul_get_output_type_valint32_tbfloat16
     (
       bfloat16* out_temp_accum,
       int32_t* temp_accum
     )
{

    float float_temp_accum = ( float )( *temp_accum );

     /* Fix for rounding bias. */
    uint32_t inter_temp;
    memcpy( &inter_temp, &float_temp_accum, sizeof( float ) );

    /* Check if 16th bit is set */
    uint32_t tlsb = ( inter_temp & ( uint32_t )0x00010000 ) > 16;

    /* Adding rounding bias. */
    uint32_t rounded = inter_temp + ( uint32_t )0x00007FFF + tlsb;
    memcpy( &float_temp_accum, &rounded, sizeof( float ) );

    float_to_bf16( &float_temp_accum, out_temp_accum );
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
        dim_t num_eltwise = post_ops->num_eltwise;
        for ( dim_t i = 0; i < num_eltwise; ++i )
        {
            free( ( post_ops->eltwise + i )->algo.alpha );
            free( ( post_ops->eltwise + i )->algo.beta );
        }
    }

    if ( post_ops->matrix_add != NULL )
    {
        free( ( post_ops->matrix_add )->matrix );
        free( ( post_ops->matrix_add )->scale_factor );
    }

    if ( post_ops->sum != NULL )
    {
        free( ( post_ops->sum )->scale_factor );
        free( ( post_ops->sum )->zero_point );
    }

    if ( post_ops->matrix_mul != NULL )
    {
        free( ( post_ops->matrix_mul )->matrix );
        free( ( post_ops->matrix_mul )->scale_factor );
    }

    if ( post_ops->bias != NULL )
    {
        free( ( post_ops->bias )->bias );
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
    }

    if( ( post_ops->post_op_grp != NULL ) )
    {
        if( ( post_ops->post_op_grp ) -> a_scl != NULL )
        {
            free( ( ( post_ops->post_op_grp ) -> a_scl )->scale_factor );
            free( ( post_ops->post_op_grp ) -> a_scl );
        }

        if( ( post_ops->post_op_grp ) -> a_zp != NULL )
        {
            free( ( ( post_ops->post_op_grp ) -> a_zp )->zero_point );
            free( ( post_ops->post_op_grp ) -> a_zp );
        }

        if( ( post_ops->post_op_grp ) -> b_scl != NULL )
        {
            free( ( ( post_ops->post_op_grp ) -> b_scl )->scale_factor );
            free( ( post_ops->post_op_grp ) -> b_scl );
        }

        if( ( post_ops->post_op_grp ) -> b_zp != NULL )
        {
            free( ( ( post_ops->post_op_grp ) -> b_zp )->zero_point );
            free( ( post_ops->post_op_grp ) -> b_zp );
        }
    }

    /* Freeing all the structs */
    free( post_ops->eltwise );
    free( post_ops->pre_ops );
    free( post_ops->matrix_add );
    free( post_ops->sum );
    free( post_ops->matrix_mul );
    free( post_ops->bias );
    free( post_ops->post_op_grp );

    free( post_ops->seq_vector );
    free( post_ops );
}


#define PRINT_MATRIX(ctype) \
void print_matrix_## ctype ( ctype* a, dim_t m, dim_t n, dim_t rs, dim_t cs) \
{ \
    for(dim_t i = 0; i < m; i++) \
    { \
        for(dim_t j = 0; j < n; j++) \
        { \
            printf("%d ", (int) (*(a + i * ( rs ) + j * cs ) ) ); \
        } \
        printf("\n"); \
    } \
} \

/* Helper functions to print matrices when debugging */
void print_matrix_bfloat16
     (
       bfloat16* a,
       dim_t m,
       dim_t n,
       dim_t rs_a,
       dim_t cs_a
     )
{
    for(dim_t i = 0; i < m; i++)
    {
        for(dim_t j = 0; j < n; j++)
        {
            float temp;
            bfloat16_to_float(*(a + i*(rs_a) + j *cs_a), &temp);
            printf("%d ", (int)temp);
        }
        printf("\n");
    }
}

#define GEN_MAT_MUL_ACC_CHK_DOWNSCALE(ZP_type,C_type,ACCUM_type,SCALE_type,BLAS_DOWNSCALE_SFX) \
static inline ACCUM_type mat_mul_accuracy_check_downscale_ ## BLAS_DOWNSCALE_SFX \
     (\
       ACCUM_type temp_accum,\
       aocl_post_op*  post_op, \
       dim_t j \
     )\
{ \
    dim_t j_scale = j; \
    if ( ( post_op->sum )->scale_factor_len == 1 ) \
    { \
       j_scale = 0; \
    } \
 \
    dim_t j_zp = j; \
    if ( ( post_op->sum )->zero_point_len == 1 ) \
    { \
       j_zp = 0; \
    } \
 \
    ACCUM_type out_temp_accum = \
        ( ACCUM_type )min( \
                        max( nearbyintf( ( SCALE_type )( temp_accum ) * \
                            ( *( ( SCALE_type* )( post_op->sum )->scale_factor + j_scale ) ) ) + \
                            *( ( ZP_type* )( post_op->sum )->zero_point + j_zp ), \
                            DSCALE_CLIP_MIN ), \
                        DSCALE_CLIP_MAX ); \
    return out_temp_accum; \
}\


static inline float mat_mul_accuracy_check_downscale_bf16bf16f32obf16
     (
       float temp_accum,
       aocl_post_op*  post_op,
       dim_t j
     )
{
    dim_t j_scale = j;
    if ( ( post_op->sum )->scale_factor_len == 1 )
    {
       j_scale = 0;
    }

    dim_t j_zp = j;
    if ( ( post_op->sum )->zero_point_len == 1 )
    {
       j_zp = 0;
    }

    float zp_float = 0.0;
    bfloat16_to_float( *( ( bfloat16* )( post_op->sum )->zero_point + j_zp ),
                       &zp_float );
    float out_temp_accum = ( temp_accum *
                ( *( ( float* )( post_op->sum )->scale_factor + j_scale ) ) +
                zp_float );
    return out_temp_accum;
}
static inline float mat_mul_accuracy_check_downscale_f32f32f32of32
     (
       float temp_accum,
       aocl_post_op*  post_op,
       dim_t j
     )
{
    dim_t j_scale = j;
    if ( ( post_op->sum )->scale_factor_len == 1 )
    {
       j_scale = 0;
    }

    dim_t j_zp = j;
    if ( ( post_op->sum )->zero_point_len == 1 )
    {
       j_zp = 0;
    }
    float out_temp_accum = ( temp_accum *
                ( *( ( float* )( post_op->sum )->scale_factor + j_scale ) ) +
                 *( ( float* )( post_op->sum )->zero_point + j_zp ) );
    return out_temp_accum;
}

#define GEN_MAT_MUL_ACC_CHK_ACCUM(A_type, B_type, C_type,ACCUM_type,BLAS_SFX) \
static inline ACCUM_type mat_mul_accuracy_check_accum_ ## BLAS_SFX \
     (\
       A_type* a, \
       B_type* b, \
       C_type* c_ref, \
       ACCUM_type temp_accum,\
       ACCUM_type  alpha, \
       ACCUM_type beta, \
       dim_t rs_a, \
       dim_t rs_b, \
       dim_t cs_a, \
       dim_t cs_b, \
       dim_t rs_c_ref, \
       dim_t cs_c_ref, \
       dim_t i, \
       dim_t j, \
       dim_t k, \
       dim_t pre_op_ld, /* Ignored */ \
       aocl_pre_op* pre_op /* Workaround to enable B pre-ops. */ \
     ) \
{ \
    ( void ) pre_op; \
    temp_accum = (ACCUM_type) 0; \
    for ( dim_t p = 0; p < k; ++p) \
    { \
        temp_accum += ( *( a + ( i * rs_a ) + ( cs_a * p ) ) * \
                         *( b + ( rs_b * p ) + ( cs_b * j ) ) ); \
    } \
    \
    temp_accum = ( beta * ( * (c_ref + ( rs_c_ref * i ) + ( cs_c_ref * j ) ) ) ) \
                  + ( alpha * temp_accum ); \
    return temp_accum; \
 } \

 #define GEN_MAT_MUL_ACC_CHK_ACCUM_SYM_QUANT(A_type, B_type, C_type,ACCUM_type,BLAS_SFX) \
static inline float mat_mul_accuracy_check_accum_ ## BLAS_SFX \
     (\
       A_type* a, \
       B_type* b, \
       C_type* c_ref, \
       ACCUM_type temp_accum,\
       ACCUM_type  alpha, \
       ACCUM_type beta, \
       dim_t rs_a, \
       dim_t rs_b, \
       dim_t cs_a, \
       dim_t cs_b, \
       dim_t rs_c_ref, \
       dim_t cs_c_ref, \
       dim_t i, \
       dim_t j, \
       dim_t k, \
       dim_t n, \
       aocl_group_post_op* grp_post_op /* Workaround to enable B pre-ops. */ \
     ) \
{ \
    float temp_accum_float = (float) 0; \
    float float_accum = (float) 0; \
    dim_t group_size = grp_post_op->group_size; \
    dim_t num_groups = ( k + group_size - 1 ) / group_size; \
    for( dim_t p_g = 0; p_g < k; p_g += group_size ) \
    { \
        dim_t gs = bli_min( group_size, k - p_g ); \
        dim_t group_num = p_g / group_size; \
        float a_scl, b_scl; \
        if( (grp_post_op->a_scl)->scale_factor_type == AOCL_GEMM_BF16 ) \
        { \
            bfloat16_to_float( *( ( bfloat16* )( ( grp_post_op->a_scl )->scale_factor ) + ( i * num_groups ) + group_num ), &a_scl ); \
        } \
        else \
        { \
            a_scl = *( ( float* )( ( grp_post_op->a_scl )->scale_factor ) + ( i * num_groups ) + group_num ); \
        } \
        if( ( grp_post_op->b_scl )->scale_factor_type == AOCL_GEMM_BF16 ) \
        { \
            bfloat16_to_float( *( ( bfloat16* )( ( grp_post_op->b_scl )->scale_factor ) + ( group_num * n ) + j ), &b_scl ); \
        } \
        else \
        { \
            b_scl = *( ( float* )( ( grp_post_op->b_scl )->scale_factor ) + ( group_num * n ) + j ); \
        } \
        temp_accum = (ACCUM_type) 0; \
        for( dim_t p = p_g; p < p_g + gs; ++p) \
        { \
            temp_accum += ( *( a + ( i * rs_a ) + ( cs_a * p ) ) * \
                             *( b + ( rs_b * p ) + ( cs_b * j ) )); \
        } \
        temp_accum_float = (float)temp_accum; \
        temp_accum_float *= a_scl; \
        temp_accum_float *= b_scl; \
        float_accum += temp_accum_float; \
    } \
    float c_ref_float = 0.0; \
    GEN_FUNC_NAME(C_type,_to_float)( *( c_ref + ( rs_c_ref * i ) + ( cs_c_ref * j ) ), &c_ref_float ); \
    float_accum = ( (float)beta * c_ref_float ) \
                  + ( (float)alpha * float_accum ); \
    return float_accum; \
} \

static inline int32_t mat_mul_accuracy_check_accum_u8s8s32obf16
     (
       uint8_t* a,
       int8_t* b,
       bfloat16* c_ref,
       int32_t temp_accum,
       int32_t  alpha,
       int32_t beta,
       dim_t rs_a,
       dim_t rs_b,
       dim_t cs_a,
       dim_t cs_b,
       dim_t rs_c_ref,
       dim_t cs_c_ref,
       dim_t i,
       dim_t j,
       dim_t k,
       dim_t pre_op_ld, /* Ignored */
       aocl_pre_op* pre_op /* Workaround to enable B pre-ops. */
     )
{
    ( void ) pre_op;
    for ( dim_t p = 0; p < k; ++p)
    {
        temp_accum += ( *( a + ( i * rs_a ) + ( cs_a * p ) ) *
                         *( b + ( rs_b * p ) + ( cs_b * j ) ) );
    }

    float c_ref_float;
    bfloat16_to_float(( * (c_ref + ( rs_c_ref * i ) + ( cs_c_ref * j ) ) ), &c_ref_float);
    temp_accum = ( beta * c_ref_float ) + ( alpha * temp_accum );
    return temp_accum;
  }
static inline int32_t mat_mul_accuracy_check_accum_s8s8s32obf16
     (
       int8_t* a,
       int8_t* b,
       bfloat16* c_ref,
       int32_t temp_accum,
       int32_t  alpha,
       int32_t beta,
       dim_t rs_a,
       dim_t rs_b,
       dim_t cs_a,
       dim_t cs_b,
       dim_t rs_c_ref,
       dim_t cs_c_ref,
       dim_t i,
       dim_t j,
       dim_t k,
       dim_t pre_op_ld, /* Ignored */
       aocl_pre_op* pre_op /* Workaround to enable B pre-ops. */
     )
{
    ( void ) pre_op;
    for ( dim_t p = 0; p < k; ++p)
    {
        temp_accum += ( *( a + ( i * rs_a ) + ( cs_a * p ) ) *
                         *( b + ( rs_b * p ) + ( cs_b * j ) ) );
    }

    float c_ref_float;
    bfloat16_to_float(( * (c_ref + ( rs_c_ref * i ) + ( cs_c_ref * j ) ) ), &c_ref_float);
    temp_accum = ( beta * c_ref_float ) + ( alpha * temp_accum );
    return temp_accum;
  }

static inline int32_t mat_mul_accuracy_check_accum_u8s8s32of32
     (
       uint8_t* a,
       int8_t* b,
       float* c_ref,
       int32_t temp_accum,
       int32_t  alpha,
       int32_t beta,
       dim_t rs_a,
       dim_t rs_b,
       dim_t cs_a,
       dim_t cs_b,
       dim_t rs_c_ref,
       dim_t cs_c_ref,
       dim_t i,
       dim_t j,
       dim_t k,
       dim_t pre_op_ld, /* Ignored */
       aocl_pre_op* pre_op /* Workaround to enable B pre-ops. */
     )
{
    ( void ) pre_op;
    for ( dim_t p = 0; p < k; ++p)
    {
        temp_accum += ( *( a + ( i * rs_a ) + ( cs_a * p ) ) *
                         *( b + ( rs_b * p ) + ( cs_b * j ) ) );
    }

    float c_ref_float =  *(c_ref + ( rs_c_ref * i ) + ( cs_c_ref * j ) );
    temp_accum = ( beta * c_ref_float ) + ( alpha * temp_accum );
    return temp_accum;
  }

static inline int32_t mat_mul_accuracy_check_accum_s8s8s32of32
     (
       int8_t* a,
       int8_t* b,
       float* c_ref,
       int32_t temp_accum,
       int32_t  alpha,
       int32_t beta,
       dim_t rs_a,
       dim_t rs_b,
       dim_t cs_a,
       dim_t cs_b,
       dim_t rs_c_ref,
       dim_t cs_c_ref,
       dim_t i,
       dim_t j,
       dim_t k,
       dim_t pre_op_ld, /* Ignored */
       aocl_pre_op* pre_op /* Workaround to enable B pre-ops. */
     )
{
    ( void ) pre_op;
    for ( dim_t p = 0; p < k; ++p)
    {
        temp_accum += ( *( a + ( i * rs_a ) + ( cs_a * p ) ) *
                         *( b + ( rs_b * p ) + ( cs_b * j ) ) );
    }

    float c_ref_float =  *(c_ref + ( rs_c_ref * i ) + ( cs_c_ref * j ) );
    temp_accum = ( beta * ((int32_t)c_ref_float) ) + ( alpha * temp_accum );
    return temp_accum;
  }

static inline float mat_mul_accuracy_check_accum_bf16bf16f32of32
     (
       bfloat16* a,
       bfloat16* b,
       float* c_ref,
       float temp_accum,
       float  alpha,
       float beta,
       dim_t rs_a,
       dim_t rs_b,
       dim_t cs_a,
       dim_t cs_b,
       dim_t rs_c_ref,
       dim_t cs_c_ref,
       dim_t i,
       dim_t j,
       dim_t k,
       dim_t pre_op_ld, /* Ignored */ \
       aocl_pre_op* pre_op /* Workaround to enable B pre-ops. */ \
     )
{
    ( void ) pre_op;
    for ( dim_t p = 0; p < k; ++p)
    {
        float a_float, b_float;
        bfloat16_to_float( *( a + i * rs_a + p * cs_a ) , &a_float);
        bfloat16_to_float( *( b + p * rs_b + j * cs_b ) , &b_float);
        temp_accum += ( ( a_float ) * ( b_float ) );
    }
    temp_accum = ( beta * ( * (c_ref + ( rs_c_ref * i ) + ( cs_c_ref * j ) ) ) )
                         + ( alpha * temp_accum );
    return temp_accum;
}

static inline float mat_mul_accuracy_check_accum_bf16bf16f32obf16
     (
       bfloat16* a,
       bfloat16* b,
       bfloat16* c_ref,
       float temp_accum,
       float  alpha,
       float beta,
       dim_t rs_a,
       dim_t rs_b,
       dim_t cs_a,
       dim_t cs_b,
       dim_t rs_c_ref,
       dim_t cs_c_ref,
       dim_t i,
       dim_t j,
       dim_t k,
       dim_t pre_op_ld, /* Ignored */
       aocl_pre_op* pre_op /* Workaround to enable B pre-ops. */ \
     )
{
    ( void ) pre_op;
    for ( dim_t p = 0; p < k; ++p)
    {
        float a_float, b_float;
        bfloat16_to_float( *( a + i*rs_a + p*cs_a ), &a_float );
        bfloat16_to_float( *( b + p*rs_b + j*cs_b ), &b_float );
        temp_accum += ( ( a_float ) * ( b_float ) );
    }
    float c_ref_float;
    bfloat16_to_float( *( c_ref + i*rs_c_ref + j*cs_c_ref ), &c_ref_float );
    temp_accum = ( beta * ( c_ref_float ) ) + ( alpha * temp_accum );

    return temp_accum;
}

static inline float get_s4_to_f32_scale_val
     (
       int8_t* b,
       dim_t p,
       dim_t j,
       dim_t n,
       dim_t b_inc,
       aocl_pre_op* pre_op
     )
{
    float b_float = 0.0;
    int8_t b_val = 0;

    dim_t group_size = pre_op->group_size;

    /* Even index will have data at low 4 bits, and odd at hi 4 bits.
     * B matrix increments has to be halved to account for 4 bit
     * traversal. */
    if ( ( b_inc % 2 ) != 0 )
    {
        b_val = ( ( *( b + ( b_inc / 2 ) ) ) >> 4 ) & 0x0F;
    }
    else
    {
        b_val = ( *( b + ( b_inc / 2 ) ) ) & 0x0F;
    }

    /* Signed scale. */
    if ( b_val & 0x08 )
    {
         b_val = b_val | 0xF0;
    }

    if ( ( pre_op != NULL ) && ( pre_op->seq_length > 0 ) )
    {
        dim_t j_zp=0, j_scale=0;
        if(group_size!=0)
        {
            j_zp = ( ( p / group_size ) * n ) + j;
            if ( ( pre_op->b_zp != NULL ) &&
                ( ( pre_op->b_zp )->zero_point_len == 1 ) )
            {
                j_zp = p / group_size;
            }

            j_scale = ( ( p / group_size ) * n ) + j;
            if ( ( pre_op->b_scl != NULL ) &&
                ( ( pre_op->b_scl )->scale_factor_len == 1 ) )
            {
                j_scale = (p / group_size);
            }
        }

        // Assuming only 1 scale and zp.
        int8_t zp = 0;
        if ( ( pre_op->b_zp != NULL ) &&
             ( ( pre_op->b_zp )->zero_point != NULL ) )
        {
            zp = *( ( int8_t* )( pre_op->b_zp )->zero_point + j_zp );
        }

        float scale_factor = 1.0;
        if ( ( pre_op->b_scl != NULL ) &&
             ( ( pre_op->b_scl )->scale_factor != NULL ) )
        {
            if( pre_op->b_scl->scale_factor_type == AOCL_GEMM_F32 )
            {
                scale_factor = *( ( float* )( pre_op->b_scl )->scale_factor + j_scale );
            }
            else
            {
                bfloat16_to_float( *( ( bfloat16* )( pre_op->b_scl )->scale_factor + j_scale ) , &scale_factor);
            }

        }
        b_float = (float)( b_val - zp ) * scale_factor;
    }
    else
    {
        b_float = (float)( b_val);
    }

    return b_float;
}

static inline float mat_mul_accuracy_check_accum_bf16s4f32of32
     (
       bfloat16* a,
       int8_t* b,
       float* c_ref,
       float temp_accum,
       float  alpha,
       float beta,
       dim_t rs_a,
       dim_t rs_b,
       dim_t cs_a,
       dim_t cs_b,
       dim_t rs_c_ref,
       dim_t cs_c_ref,
       dim_t i,
       dim_t j,
       dim_t k,
       dim_t pre_op_ld,
       aocl_pre_op* pre_op /* Workaround to enable B pre-ops. */ \
     )
{
    for ( dim_t p = 0; p < k; ++p)
    {
        float a_float, b_float;
        bfloat16_to_float( *( a + i * rs_a + p * cs_a ) , &a_float);

        /* Get B matrix int4_t value and upscale it to float. */
        dim_t b_inc = ( rs_b * p ) + ( cs_b * j );
        b_float = get_s4_to_f32_scale_val( b, p, j, pre_op_ld, b_inc, pre_op );

        temp_accum += ( ( a_float ) * ( b_float ) );
    }
    temp_accum = ( beta * ( * (c_ref + ( rs_c_ref * i ) + ( cs_c_ref * j ) ) ) )
                         + ( alpha * temp_accum );
    return temp_accum;
}


static inline float mat_mul_accuracy_check_accum_bf16s4f32obf16
     (
       bfloat16* a,
       int8_t* b,
       bfloat16* c_ref,
       float temp_accum,
       float  alpha,
       float beta,
       dim_t rs_a,
       dim_t rs_b,
       dim_t cs_a,
       dim_t cs_b,
       dim_t rs_c_ref,
       dim_t cs_c_ref,
       dim_t i,
       dim_t j,
       dim_t k,
       dim_t pre_op_ld,
       aocl_pre_op* pre_op /* Workaround to enable B pre-ops. */ \
     )
{
    for ( dim_t p = 0; p < k; ++p)
    {
        float a_float, b_float;
        bfloat16_to_float( *( a + i*rs_a + p*cs_a ), &a_float );

        /* Get B matrix int4_t value and upscale it to float. */
        dim_t b_inc = ( rs_b * p ) + ( cs_b * j );
        b_float = get_s4_to_f32_scale_val( b, p, j, pre_op_ld, b_inc, pre_op );

        temp_accum += ( ( a_float ) * ( b_float ) );
    }
    float c_ref_float;
    bfloat16_to_float( *( c_ref + i*rs_c_ref + j*cs_c_ref ), &c_ref_float );
    temp_accum = ( beta * ( c_ref_float ) ) + ( alpha * temp_accum );

    return temp_accum;
}


#define GEN_MAT_MUL_POST_OPS_CREATOR(C_DSCALE_type,C_type,DSCALE_type,BIAS_type,BLAS_SFX) \
static inline aocl_post_op* lpgemm_create_post_ops_struct_ ## BLAS_SFX \
     ( \
       dim_t m, \
       dim_t n, \
       dim_t k, \
       char* post_ops_str, \
       char  stor_order \
     ) \
{ \
    if ( ( ( post_ops_str == NULL ) || \
           ( strcmp( post_ops_str, "none" ) == 0 ) ) && \
         ( global_dscale_out == 'n' ) && ( global_pre_op == 'n' ) ) \
    { \
        return NULL; \
    } \
 \
    aocl_post_op* post_ops = NULL; \
    post_ops = ( aocl_post_op* ) malloc( sizeof( aocl_post_op ) ); \
 \
    if ( post_ops == NULL ) \
    { \
        return NULL; \
    } \
    post_ops->eltwise = NULL; \
    post_ops->bias = NULL; \
    post_ops->sum = NULL; \
    post_ops->matrix_add = NULL; \
    post_ops->matrix_mul = NULL; \
    post_ops->pre_ops = NULL; \
 \
    /* Only supporting 8 post ops at max for now.*/ \
    dim_t max_post_ops_seq_length = 8; \
    post_ops->seq_vector = ( AOCL_POST_OP_TYPE* ) \
                            malloc \
                            ( \
                              max_post_ops_seq_length * \
                              sizeof( AOCL_POST_OP_TYPE ) \
                            ); \
 \
    if ( post_ops->seq_vector == NULL ) \
    { \
        goto err_handler; \
    } \
 \
    /* Parse post ops list.*/ \
    dim_t cur_op_index = 0; \
 \
    bool is_bias = FALSE; \
    bool is_relu = FALSE; \
    bool is_param_relu = FALSE; \
    bool is_gelu_tanh = FALSE; \
    bool is_gelu_erf = FALSE; \
    bool is_swish = FALSE; \
    bool is_clip = FALSE; \
    bool is_scalar_scale = FALSE; \
    bool is_scalar_zp = FALSE; \
    bool is_matrix_add = FALSE; \
    bool is_matrix_mul = FALSE; \
    bool is_tanh = FALSE; \
    bool is_sigmoid = FALSE; \
    bool is_bias_stor_type = FALSE; \
    dim_t activator_idx = 0; \
    dim_t clip_idx = 0; \
    char * bias_stor_type = ""; \
    bool is_zp_stor_type = FALSE; \
    char* zp_stor_type = ""; \
    bool is_matadd_stor_type = FALSE; \
    char* matadd_stor_type = ""; \
    bool is_matmul_stor_type = FALSE; \
    char* matmul_stor_type = ""; \
    bool is_group_quant = FALSE; \
    bool is_pre_op_scale_scalar = FALSE; \
    bool is_pre_op_scale_f32 = TRUE; \
    bool is_sym_quant = FALSE; \
    char* sym_quant_sf_type = ""; \
    dim_t zp_vec_length = 0; \
    dim_t quant_group_size = 0; \
 \
    /* Post-Ops string parser. */ \
    dim_t num_eltwise = 0; \
    if ( strcmp( post_ops_str, "none" ) != 0 ) \
    { \
        char* ops_tok = strtok(post_ops_str, ", =" ); \
 \
        /* Ensure only one activator is used as an eltwise post-op.*/ \
        bool is_activator_set = FALSE; \
        while ( ops_tok ) \
        { \
            str_tolower( ops_tok ); \
            if ( strcmp( ops_tok, "bias" ) == 0 ) \
            { \
                post_ops->seq_vector[cur_op_index] = BIAS; \
                ops_tok = strtok( NULL, ", " ); \
                if( ( strcmp( ops_tok, "na" ) == 0 ) ) \
                { \
                    is_bias_stor_type = FALSE; \
                } \
                else if ( ( strcmp( ops_tok, "f32" ) == 0 ) ) \
                { \
                    is_bias_stor_type = TRUE; \
                    bias_stor_type = "F32"; \
                } \
                else if ( ( strcmp( ops_tok, "bf16" ) == 0 ) ) \
                { \
                    is_bias_stor_type = TRUE; \
                    bias_stor_type = "BF16"; \
                } \
                else if ( ( strcmp( ops_tok, "s32" ) == 0 ) ) \
                { \
                    is_bias_stor_type = TRUE; \
                    bias_stor_type = "S32"; \
                } \
                else if ( ( strcmp( ops_tok, "s8" ) == 0 ) ) \
                { \
                    is_bias_stor_type = TRUE; \
                    bias_stor_type = "S8"; \
                } \
                is_bias = TRUE; \
                cur_op_index++; \
            } \
            else if ( ( strcmp( ops_tok, "relu" ) == 0 ) && \
                      ( is_activator_set == FALSE ) ) \
            { \
                post_ops->seq_vector[cur_op_index] = ELTWISE; \
                is_relu = TRUE; \
                is_activator_set = TRUE; \
                num_eltwise += 1; \
                activator_idx = cur_op_index; \
                cur_op_index++; \
            } \
            else if ( ( strcmp( ops_tok, "prelu" ) == 0 ) && \
                      ( is_activator_set == FALSE ) ) \
            { \
                post_ops->seq_vector[cur_op_index] = ELTWISE; \
                is_param_relu = TRUE; \
                is_activator_set = TRUE; \
                num_eltwise += 1; \
                activator_idx = cur_op_index; \
                cur_op_index++; \
            } \
            else if ( ( strcmp( ops_tok, "swish" ) == 0 ) && \
                      ( is_activator_set == FALSE ) ) \
            { \
                post_ops->seq_vector[cur_op_index] = ELTWISE; \
                is_swish = TRUE; \
                is_activator_set = TRUE; \
                num_eltwise += 1; \
                activator_idx = cur_op_index; \
                cur_op_index++; \
            } \
            else if ( ( strcmp( ops_tok, "gelu_tanh" ) == 0 ) && \
                      ( is_activator_set == FALSE ) ) \
            { \
                post_ops->seq_vector[cur_op_index] = ELTWISE; \
                is_gelu_tanh = TRUE; \
                is_activator_set = TRUE; \
                num_eltwise += 1; \
                activator_idx = cur_op_index; \
                cur_op_index++; \
            } \
            else if ( ( strcmp( ops_tok, "gelu_erf" ) == 0 ) && \
                      ( is_activator_set == FALSE ) ) \
            { \
                post_ops->seq_vector[cur_op_index] = ELTWISE; \
                is_gelu_erf = TRUE; \
                is_activator_set = TRUE; \
                num_eltwise += 1; \
                activator_idx = cur_op_index; \
                cur_op_index++; \
            } \
            else if ( strcmp( ops_tok, "clip" ) == 0 ) \
            { \
                post_ops->seq_vector[cur_op_index] = ELTWISE; \
                is_clip = TRUE; \
                num_eltwise += 1; \
                clip_idx = cur_op_index; \
                cur_op_index++; \
            } \
            else if ( strcmp( ops_tok, "scale" ) == 0 ) \
            { \
                ops_tok = strtok( NULL, ", " ); \
                str_tolower( ops_tok ); \
                if ( ( strcmp( ops_tok, "scalar" ) == 0 ) || \
                     ( strcmp( ops_tok, "s" ) == 0 ) ) \
                { \
                    is_scalar_scale = TRUE; \
                } \
            } \
            else if ( strcmp( ops_tok, "zp" ) == 0 ) \
            { \
                ops_tok = strtok( NULL, ", " ); \
                str_tolower( ops_tok ); \
                if ( ( strcmp( ops_tok, "scalar" ) == 0 ) || \
                     ( strcmp( ops_tok, "s" ) == 0 ) ) \
                { \
                    is_scalar_zp = TRUE; \
                } \
            } \
            else if ( strcmp( ops_tok, "zp_stor_type" ) == 0) \
            { \
                 ops_tok = strtok( NULL, ", " ); \
                if( ( strcmp( ops_tok, "na" ) == 0 ) ) \
                { \
                    is_zp_stor_type = FALSE; \
                } \
                else if ( ( strcmp( ops_tok, "f32" ) == 0 ) ) \
                { \
                    is_zp_stor_type = TRUE; \
                    zp_stor_type = "F32"; \
                } \
                else if ( ( strcmp( ops_tok, "bf16" ) == 0 ) ) \
                { \
                    is_zp_stor_type = TRUE; \
                    zp_stor_type = "BF16"; \
                } \
                else if ( ( strcmp( ops_tok, "s32" ) == 0 ) ) \
                { \
                    is_zp_stor_type = TRUE; \
                    zp_stor_type = "S32"; \
                } \
                else if ( ( strcmp( ops_tok, "s8" ) == 0 ) ) \
                { \
                    is_zp_stor_type = TRUE; \
                    zp_stor_type = "S8"; \
                } \
                else if ( ( strcmp( ops_tok, "u8" ) == 0 ) ) \
                { \
                    is_zp_stor_type = TRUE; \
                    zp_stor_type = "U8"; \
                } \
            } \
            else if ( strcmp( ops_tok, "matrix_add" ) == 0 ) \
            { \
                post_ops->seq_vector[cur_op_index] = MATRIX_ADD; \
                ops_tok = strtok( NULL, ", " ); \
                if( ( strcmp( ops_tok, "na" ) == 0 ) ) \
                { \
                    is_matadd_stor_type = FALSE; \
                } \
                else if ( ( strcmp( ops_tok, "f32" ) == 0 ) ) \
                { \
                    is_matadd_stor_type = TRUE; \
                    matadd_stor_type = "F32"; \
                } \
                else if ( ( strcmp( ops_tok, "bf16" ) == 0 ) ) \
                { \
                    is_matadd_stor_type = TRUE; \
                    matadd_stor_type = "BF16"; \
                } \
                else if ( ( strcmp( ops_tok, "s32" ) == 0 ) ) \
                { \
                    is_matadd_stor_type = TRUE; \
                    matadd_stor_type = "S32"; \
                } \
                else if ( ( strcmp( ops_tok, "s8" ) == 0 ) ) \
                { \
                    is_matadd_stor_type = TRUE; \
                    matadd_stor_type = "S8"; \
                } \
                is_matrix_add = TRUE; \
                cur_op_index++; \
            } \
            else if ( strcmp( ops_tok, "matrix_mul" ) == 0 ) \
            { \
                post_ops->seq_vector[cur_op_index] = MATRIX_MUL; \
                ops_tok = strtok( NULL, ", " ); \
                if( ( strcmp( ops_tok, "na" ) == 0 ) ) \
                { \
                    is_matmul_stor_type = FALSE; \
                } \
                else if ( ( strcmp( ops_tok, "f32" ) == 0 ) ) \
                { \
                    is_matmul_stor_type = TRUE; \
                    matmul_stor_type = "F32"; \
                } \
                else if ( ( strcmp( ops_tok, "bf16" ) == 0 ) ) \
                { \
                    is_matmul_stor_type = TRUE; \
                    matmul_stor_type = "BF16"; \
                } \
                else if ( ( strcmp( ops_tok, "s32" ) == 0 ) ) \
                { \
                    is_matmul_stor_type = TRUE; \
                    matmul_stor_type = "S32"; \
                } \
                else if ( ( strcmp( ops_tok, "s8" ) == 0 ) ) \
                { \
                    is_matmul_stor_type = TRUE; \
                    matmul_stor_type = "S8"; \
                } \
                is_matrix_mul = TRUE; \
                cur_op_index++; \
            } \
            else if ( ( strcmp( ops_tok, "tanh" ) == 0 ) && \
                      ( is_activator_set == FALSE ) ) \
            { \
                post_ops->seq_vector[cur_op_index] = ELTWISE; \
                is_tanh = TRUE; \
                is_activator_set = TRUE; \
                num_eltwise += 1; \
                activator_idx = cur_op_index; \
                cur_op_index++; \
            } \
             else if ( ( strcmp( ops_tok, "sigmoid" ) == 0 ) && \
                      ( is_activator_set == FALSE ) ) \
            { \
                post_ops->seq_vector[cur_op_index] = ELTWISE; \
                is_sigmoid = TRUE; \
                is_activator_set = TRUE; \
                num_eltwise += 1; \
                activator_idx = cur_op_index; \
                cur_op_index++; \
            } \
            else if ( strcmp( ops_tok, "group_size" ) == 0 ) \
            { \
                ops_tok = strtok( NULL, ", " ); \
                quant_group_size = atoi(ops_tok); \
                is_group_quant = TRUE; \
            } \
            else if ( strcmp( ops_tok, "pre_op_zp" ) == 0 ) \
            { \
                ops_tok = strtok( NULL, ", " ); \
                str_tolower( ops_tok ); \
                if ( ( strcmp( ops_tok, "scalar" ) == 0 ) || \
                     ( strcmp( ops_tok, "s" ) == 0 ) ) \
                { \
                    /* set scalar zp */\
                    zp_vec_length = 1; \
                } \
                else \
                { \
                    /* set vector zp */\
                    zp_vec_length = n; \
                } \
            } \
            else if ( strcmp( ops_tok, "pre_op_scale" ) == 0 ) \
            { \
               ops_tok = strtok( NULL, ", " ); \
                str_tolower( ops_tok ); \
                if ( ( strcmp( ops_tok, "scalar" ) == 0 ) || \
                     ( strcmp( ops_tok, "s" ) == 0 ) ) \
                { \
                    /* set scalar scale */\
                    is_pre_op_scale_scalar = TRUE; \
                } \
                else \
                { \
                    /* set vector scale */\
                    is_pre_op_scale_scalar = FALSE; \
                } \
            } \
            else if ( strcmp( ops_tok, "pre_op_scale_type" ) == 0 ) \
            { \
               ops_tok = strtok( NULL, ", " ); \
                str_tolower( ops_tok ); \
                if ( ( strcmp( ops_tok, "bf16" ) == 0 ) ) \
                { \
                    /* set scalar scale */\
                    is_pre_op_scale_f32 = FALSE; \
                }else \
                { \
                    /* set vector scale */\
                    is_pre_op_scale_f32 = TRUE; \
                } \
            } \
            else if ( strcmp( ops_tok, "sym_quant_sf" ) == 0 ) \
            { \
                ops_tok = strtok( NULL, ", " ); \
                str_tolower( ops_tok ); \
                if ( ( strcmp( ops_tok, "bf16" ) == 0 ) ) \
                { \
                    /* set bf16 scale */\
                    is_sym_quant = TRUE; \
                    sym_quant_sf_type = "BF16"; \
                } \
                else if ( ( strcmp( ops_tok, "f32" ) == 0 ) ) \
                { \
                    /* set f32 scale */\
                    is_sym_quant = TRUE; \
                    sym_quant_sf_type = "F32"; \
                } \
            } \
            else{} \
 \
            ops_tok = strtok( NULL, ", =" ); \
        } \
    } \
 \
        if ( is_bias == TRUE ) \
        { \
            /* Bench limitation: can only support 1 bias, but LPGEMM can support
            * multiple bias post-ops. */ \
            post_ops->bias = malloc( sizeof( aocl_post_op_bias ) ); \
            if ( post_ops->bias == NULL ) \
            { \
                goto err_handler; \
            } \
 \
            if(is_bias_stor_type == TRUE) \
            { \
                if( ( strcmp( bias_stor_type, "BF16" ) == 0 ) ) \
                { \
                    ( post_ops->bias )->stor_type = AOCL_GEMM_BF16; \
                    /* Allocate bias buffer, return early if alloc fails.*/ \
                    ( post_ops->bias )->bias = malloc( n * sizeof( bfloat16 ) ); \
                    if ( ( post_ops->bias )->bias == NULL ) \
                    { \
                        goto err_handler; \
                    } \
                    GEN_FUNC_NAME(fill_array_post_ops_,bfloat16)( ( post_ops->bias )->bias, n ); \
                } \
                else if( ( strcmp( bias_stor_type, "F32" ) == 0 ) ) \
                { \
                    ( post_ops->bias )->stor_type = AOCL_GEMM_F32; \
                    /* Allocate bias buffer, return early if alloc fails.*/ \
                    ( post_ops->bias )->bias = malloc( n * sizeof( float ) ); \
                    if ( ( post_ops->bias )->bias == NULL ) \
                    { \
                        goto err_handler; \
                    } \
                    GEN_FUNC_NAME(fill_array_post_ops_,float)( ( post_ops->bias )->bias, n ); \
                } \
                else if( ( strcmp( bias_stor_type, "S8" ) == 0 ) ) \
                { \
                    ( post_ops->bias )->stor_type = AOCL_GEMM_INT8; \
                    /* Allocate bias buffer, return early if alloc fails.*/ \
                    ( post_ops->bias )->bias = malloc( n * sizeof( int8_t ) ); \
                    if ( ( post_ops->bias )->bias == NULL ) \
                    { \
                        goto err_handler; \
                    } \
                    GEN_FUNC_NAME(fill_array_post_ops_,int8_t)( ( post_ops->bias )->bias, n ); \
                } \
                else if( ( strcmp( bias_stor_type, "S32" ) == 0 ) ) \
                { \
                    ( post_ops->bias )->stor_type = AOCL_GEMM_INT32; \
                    /* Allocate bias buffer, return early if alloc fails.*/ \
                    ( post_ops->bias )->bias = malloc( n * sizeof( int32_t ) ); \
                    if ( ( post_ops->bias )->bias == NULL ) \
                    { \
                        goto err_handler; \
                    } \
                    GEN_FUNC_NAME(fill_array_post_ops_,int32_t)( ( post_ops->bias )->bias, n ); \
                } \
                else {} \
            } \
            else \
            { \
                ( post_ops->bias )->stor_type = NULLTYPE; \
                /* Allocate bias buffer, return early if alloc fails.*/ \
                ( post_ops->bias )->bias = malloc( n * sizeof( BIAS_type ) ); \
                if ( ( post_ops->bias )->bias == NULL ) \
                { \
                    goto err_handler; \
                } \
                GEN_FUNC_NAME(fill_array_post_ops_,BIAS_type)( ( post_ops->bias )->bias, n ); \
            } \
        } \
 \
        if ( num_eltwise > 0 ) \
        { \
            if ( num_eltwise > 1 ) \
            { \
                if ( activator_idx < clip_idx ) \
                { \
                    activator_idx = 0; \
                    clip_idx = 1; \
                } \
                else \
                { \
                    activator_idx = 1; \
                    clip_idx = 0; \
                } \
            } \
            else \
            { \
            activator_idx = 0; \
            clip_idx = 0; \
        } \
 \
        post_ops->num_eltwise = num_eltwise; \
        post_ops->eltwise = malloc( num_eltwise * sizeof( aocl_post_op_eltwise ) ); \
        if ( post_ops->eltwise == NULL ) \
        { \
            goto err_handler; \
        } \
 \
        /* Only one of relu, prelu, swish, gelu_tanh, gelu_erf allowed as
         * an activator. */ \
        if ( is_relu == TRUE ) \
        { \
            ( post_ops->eltwise + activator_idx )->is_power_of_2 = FALSE; \
            ( post_ops->eltwise + activator_idx )->scale_factor = NULL; \
            ( post_ops->eltwise + activator_idx )->algo.alpha = NULL; \
            ( post_ops->eltwise + activator_idx )->algo.beta = NULL; \
            ( post_ops->eltwise + activator_idx )->algo.algo_type = RELU; \
        } \
        else if ( is_param_relu == TRUE ) \
        { \
            ( post_ops->eltwise + activator_idx )->is_power_of_2 = FALSE; \
            ( post_ops->eltwise + activator_idx )->scale_factor = NULL; \
            ( post_ops->eltwise + activator_idx )->algo.alpha = NULL; \
            /* If output is float/bfloat16, param type will be float otherwise s32 */ \
            if( is_integer(#C_type) ) \
            { \
                ( post_ops->eltwise + activator_idx )->algo.alpha = malloc( sizeof( int32_t ) ); \
                if ( ( post_ops->eltwise + activator_idx )->algo.alpha == NULL ) \
                { \
                    goto err_handler; \
                } \
                *( ( int32_t* ) ( post_ops->eltwise + activator_idx )->algo.alpha ) = ( int32_t )6; \
            } \
            else \
            { \
                ( post_ops->eltwise + activator_idx )->algo.alpha = malloc( sizeof( float ) ); \
                if ( ( post_ops->eltwise + activator_idx )->algo.alpha == NULL ) \
                { \
                    goto err_handler; \
                } \
                *( ( float* ) ( post_ops->eltwise + activator_idx )->algo.alpha ) = ( float )6; \
            } \
            ( post_ops->eltwise + activator_idx )->algo.beta = NULL; \
            ( post_ops->eltwise + activator_idx )->algo.algo_type = PRELU; \
            ( post_ops->eltwise + activator_idx )->algo.beta = NULL; \
            ( post_ops->eltwise + activator_idx )->algo.algo_type = PRELU; \
        } \
        if ( is_swish == TRUE ) \
        { \
            ( post_ops->eltwise + activator_idx )->is_power_of_2 = FALSE; \
            ( post_ops->eltwise + activator_idx )->scale_factor = NULL; \
            ( post_ops->eltwise + activator_idx )->algo.alpha = NULL; \
            /* If output is float/bfloat16, params type will be float otherwise s32 */ \
            if( is_integer(#C_type) ) \
            { \
                ( post_ops->eltwise + activator_idx )->algo.alpha = malloc( sizeof( int32_t ) ); \
                if ( ( post_ops->eltwise + activator_idx )->algo.alpha == NULL ) \
                { \
                    goto err_handler; \
                } \
                *( ( int32_t* ) ( post_ops->eltwise + activator_idx )->algo.alpha ) = ( int32_t )2; \
            } \
            else \
            { \
                ( post_ops->eltwise + activator_idx )->algo.alpha = malloc( sizeof( float ) ); \
                if ( ( post_ops->eltwise + activator_idx )->algo.alpha == NULL ) \
                { \
                    goto err_handler; \
                } \
                *( ( float* ) ( post_ops->eltwise + activator_idx )->algo.alpha ) = ( float )2; \
            } \
            ( post_ops->eltwise + activator_idx )->algo.beta = NULL; \
            ( post_ops->eltwise + activator_idx )->algo.algo_type = SWISH; \
        } \
        else if ( is_gelu_tanh == TRUE ) \
        { \
            ( post_ops->eltwise + activator_idx )->is_power_of_2 = FALSE; \
            ( post_ops->eltwise + activator_idx )->scale_factor = NULL; \
            ( post_ops->eltwise + activator_idx )->algo.alpha = NULL; \
            ( post_ops->eltwise + activator_idx )->algo.beta = NULL; \
            ( post_ops->eltwise + activator_idx )->algo.algo_type = GELU_TANH; \
        } \
        else if ( is_gelu_erf == TRUE ) \
        { \
            ( post_ops->eltwise + activator_idx )->is_power_of_2 = FALSE; \
            ( post_ops->eltwise + activator_idx )->scale_factor = NULL; \
            ( post_ops->eltwise + activator_idx )->algo.alpha = NULL; \
            ( post_ops->eltwise + activator_idx )->algo.beta = NULL; \
            ( post_ops->eltwise + activator_idx )->algo.algo_type = GELU_ERF; \
        } \
        if ( is_clip == TRUE ) \
        { \
            ( post_ops->eltwise + clip_idx )->is_power_of_2 = FALSE; \
            ( post_ops->eltwise + clip_idx )->scale_factor = NULL; \
            ( post_ops->eltwise + clip_idx )->algo.alpha = NULL; \
            ( post_ops->eltwise + clip_idx )->algo.beta = NULL; \
            /* If output is float/bfloat16, params type will be float otherwise s32 */ \
            if( is_integer(#C_type) ) \
            { \
                ( post_ops->eltwise + clip_idx )->algo.alpha = malloc( sizeof( int32_t ) ); \
                if ( ( post_ops->eltwise + clip_idx )->algo.alpha == NULL ) \
                { \
                    goto err_handler; \
                } \
                ( post_ops->eltwise + clip_idx )->algo.beta = malloc( sizeof( int32_t ) ); \
                if ( ( post_ops->eltwise + clip_idx )->algo.beta == NULL ) \
                { \
                    goto err_handler; \
                } \
                *( ( int32_t* ) ( post_ops->eltwise + clip_idx )->algo.alpha ) = ( int32_t ) ( -64 ); \
                *( ( int32_t* ) ( post_ops->eltwise + clip_idx )->algo.beta ) = ( int32_t ) ( 23 ); \
            } \
            else \
            { \
                ( post_ops->eltwise + clip_idx )->algo.alpha = malloc( sizeof( float ) ); \
                if ( ( post_ops->eltwise + clip_idx )->algo.alpha == NULL ) \
                { \
                    goto err_handler; \
                } \
                ( post_ops->eltwise + clip_idx )->algo.beta = malloc( sizeof( float ) ); \
                if ( ( post_ops->eltwise + clip_idx )->algo.beta == NULL ) \
                { \
                    goto err_handler; \
                } \
                *( ( float* ) ( post_ops->eltwise + clip_idx )->algo.alpha ) = ( float ) ( -64 ); \
                *( ( float* ) ( post_ops->eltwise + clip_idx )->algo.beta ) = ( float ) ( 23 ); \
            } \
            ( post_ops->eltwise + clip_idx )->algo.algo_type = CLIP; \
        } \
        else if ( is_tanh == TRUE ) \
        { \
            ( post_ops->eltwise + activator_idx )->is_power_of_2 = FALSE; \
            ( post_ops->eltwise + activator_idx )->scale_factor = NULL; \
            ( post_ops->eltwise + activator_idx )->algo.alpha = NULL; \
            ( post_ops->eltwise + activator_idx )->algo.beta = NULL; \
            ( post_ops->eltwise + activator_idx )->algo.algo_type = TANH; \
        } \
        if ( is_sigmoid == TRUE ) \
        { \
            ( post_ops->eltwise + activator_idx )->is_power_of_2 = FALSE; \
            ( post_ops->eltwise + activator_idx )->scale_factor = NULL; \
            ( post_ops->eltwise + activator_idx )->algo.alpha = NULL; \
            ( post_ops->eltwise + activator_idx )->algo.beta = NULL; \
            ( post_ops->eltwise + activator_idx )->algo.algo_type = SIGMOID; \
        } \
    } \
 \
    if ( ( global_dscale_out == 'y' ) || ( global_can_dscale == 'y' ) ) \
    { \
        /* Bench limitation: can only support 1 scale, but LPGEMM can support
        * multiple scale post-ops. */ \
        post_ops->sum = malloc( sizeof( aocl_post_op_sum ) ); \
        if ( post_ops->sum == NULL ) \
        { \
            goto err_handler; \
        } \
        ( post_ops->sum )->scale_factor = NULL; \
        ( post_ops->sum )->buff = NULL; \
        ( post_ops->sum )->zero_point = NULL; \
        ( post_ops->sum )->scale_factor_len = 0; \
        ( post_ops->sum )->zero_point_len = 0; \
 \
        post_ops->seq_vector[cur_op_index] = SCALE; \
        cur_op_index++; \
 \
        ( post_ops->sum )->is_power_of_2 = FALSE; \
        dim_t n_scale = n; \
        if ( is_scalar_scale == TRUE ) \
        { \
            n_scale = 1; \
        } \
\
        dim_t n_zp = n; \
        if ( is_scalar_zp == TRUE ) \
        { \
            n_zp = 1; \
        } \
\
        /* Allocate scale buffer, return early if alloc fails.*/ \
        ( post_ops->sum )->scale_factor = malloc( n_scale * sizeof( DSCALE_type ) ); \
        if ( ( post_ops->sum )->scale_factor == NULL ) \
        { \
            goto err_handler; \
        } \
        /* Fill scale factor */ \
        DSCALE_type* temp_dscale_ptr = ( DSCALE_type* )( post_ops->sum )->scale_factor; \
        GEN_FUNC_NAME(fill_array_,DSCALE_type)(temp_dscale_ptr, n_scale);   \
        ( post_ops->sum )->scale_factor_len = n_scale; \
        if(strcmp(#BLAS_SFX, "u8s8s32ou8")) for(dim_t i=0;i<n_scale;i++) temp_dscale_ptr[i] = abs(temp_dscale_ptr[i]);\
\
        if(is_zp_stor_type == TRUE) \
        { \
            if( ( strcmp( zp_stor_type, "BF16" ) == 0 ) ) \
            { \
                ( post_ops->sum )->zp_stor_type = AOCL_GEMM_BF16; \
                ( post_ops->sum )->zero_point = malloc( n_zp * sizeof( bfloat16 ) ); \
                if ( ( post_ops->sum )->zero_point == NULL ) \
                { \
                    goto err_handler; \
                } \
                GEN_FUNC_NAME(fill_array_,bfloat16)( ( post_ops->sum )->zero_point, n_zp ); \
                ( post_ops->sum )->zero_point_len = n_zp; \
            } \
            else if( ( strcmp( zp_stor_type, "F32" ) == 0 ) ) \
            { \
                ( post_ops->sum )->zp_stor_type = AOCL_GEMM_F32; \
                ( post_ops->sum )->zero_point = malloc( n_zp * sizeof( float ) ); \
                if ( ( post_ops->sum )->zero_point == NULL ) \
                { \
                    goto err_handler; \
                } \
                GEN_FUNC_NAME(fill_array_,float)( ( post_ops->sum )->zero_point, n_zp ); \
                ( post_ops->sum )->zero_point_len = n_zp; \
            } \
            else if( ( strcmp( zp_stor_type, "S32" ) == 0 ) ) \
            { \
                ( post_ops->sum )->zp_stor_type = AOCL_GEMM_INT32; \
                ( post_ops->sum )->zero_point = malloc( n_zp * sizeof( int32_t ) ); \
                if ( ( post_ops->sum )->zero_point == NULL ) \
                { \
                    goto err_handler; \
                } \
                GEN_FUNC_NAME(fill_array_,int32_t)( ( post_ops->sum )->zero_point, n_zp ); \
                ( post_ops->sum )->zero_point_len = n_zp; \
            } \
            else if( ( strcmp( zp_stor_type, "S8" ) == 0 ) ) \
            { \
                ( post_ops->sum )->zp_stor_type = AOCL_GEMM_INT8; \
                ( post_ops->sum )->zero_point = malloc( n_zp * sizeof( int8_t ) ); \
                if ( ( post_ops->sum )->zero_point == NULL ) \
                { \
                    goto err_handler; \
                } \
                GEN_FUNC_NAME(fill_array_,int8_t)( ( post_ops->sum )->zero_point, n_zp ); \
                ( post_ops->sum )->zero_point_len = n_zp; \
            } \
            else if( ( strcmp( zp_stor_type, "U8" ) == 0 ) ) \
            { \
                ( post_ops->sum )->zp_stor_type = AOCL_GEMM_UINT8; \
                ( post_ops->sum )->zero_point = malloc( n_zp * sizeof( uint8_t ) ); \
                if ( ( post_ops->sum )->zero_point == NULL ) \
                { \
                    goto err_handler; \
                } \
                GEN_FUNC_NAME(fill_array_,uint8_t)( ( post_ops->sum )->zero_point, n_zp ); \
                ( post_ops->sum )->zero_point_len = n_zp; \
            } \
            else {} \
        } \
        else \
        { \
            ( post_ops->sum )->zp_stor_type = NULLTYPE; \
            ( post_ops->sum )->zero_point = malloc( n_zp * sizeof( C_DSCALE_type ) ); \
            if ( ( post_ops->sum )->zero_point == NULL ) \
            { \
                goto err_handler; \
            } \
            C_DSCALE_type* temp_dzero_point_ptr = ( C_DSCALE_type* )( post_ops->sum )->zero_point; \
            GEN_FUNC_NAME(fill_array_,C_DSCALE_type)(temp_dzero_point_ptr, n_zp);   \
            ( post_ops->sum )->zero_point_len = n_zp; \
            if(strcmp(#BLAS_SFX, "u8s8s32ou8")) for(dim_t i=0;i<n_zp;i++) temp_dzero_point_ptr[i] = abs(temp_dzero_point_ptr[i]);\
        } \
    } \
 \
    if ( is_matrix_add == TRUE ) \
    { \
        /* Bench limitation: can only support 1 matrix add, but LPGEMM can support
        * multiple scale post-ops. */ \
        post_ops->matrix_add = malloc( sizeof( aocl_post_op_matrix_add ) ); \
        if ( post_ops->matrix_add == NULL ) \
        { \
            goto err_handler; \
        } \
 \
        if( is_matadd_stor_type == TRUE) \
        { \
            if( ( strcmp( matadd_stor_type, "BF16" ) == 0 ) ) \
            { \
                ( post_ops->matrix_add )->stor_type = AOCL_GEMM_BF16; \
                ( post_ops->matrix_add )->matrix = malloc( m * n * sizeof(bfloat16) ); \
                if ( ( post_ops->matrix_add )->matrix == NULL ) \
                { \
                    goto err_handler; \
                } \
                GEN_FUNC_NAME(fill_array_,bfloat16)( ( post_ops->matrix_add )->matrix, ( m * n ) ); \
            } \
            else if( ( strcmp( matadd_stor_type, "F32" ) == 0 ) ) \
            { \
                ( post_ops->matrix_add )->stor_type = AOCL_GEMM_F32; \
                ( post_ops->matrix_add )->matrix = malloc( m * n * sizeof(float) ); \
                if ( ( post_ops->matrix_add )->matrix == NULL ) \
                { \
                    goto err_handler; \
                } \
                GEN_FUNC_NAME(fill_array_,float)( ( post_ops->matrix_add )->matrix, ( m * n ) ); \
            } \
            else if( ( strcmp( matadd_stor_type, "S32" ) == 0 ) ) \
            { \
                ( post_ops->matrix_add )->stor_type = AOCL_GEMM_INT32; \
                ( post_ops->matrix_add )->matrix = malloc( m * n * sizeof(int32_t) ); \
                if ( ( post_ops->matrix_add )->matrix == NULL ) \
                { \
                    goto err_handler; \
                } \
                GEN_FUNC_NAME(fill_array_,int32_t)( ( post_ops->matrix_add )->matrix, ( m * n ) ); \
            } \
            else if( ( strcmp( matadd_stor_type, "S8" ) == 0 ) ) \
            { \
                ( post_ops->matrix_add )->stor_type = AOCL_GEMM_INT8; \
                ( post_ops->matrix_add )->matrix = malloc( m * n * sizeof(int8_t) ); \
                if ( ( post_ops->matrix_add )->matrix == NULL ) \
                { \
                    goto err_handler; \
                } \
                GEN_FUNC_NAME(fill_array_,int8_t)( ( post_ops->matrix_add )->matrix, ( m * n ) ); \
            } \
            else {} \
        } \
        else \
        { \
            /*  default is int32_t for integer APIs and float for others */ \
            if( is_integerAPI_avx512(#BLAS_SFX)) \
            { \
                if( strcmp(#C_type, "int8_t") == 0 ) \
                { \
                    ( post_ops->matrix_add )->stor_type = NULLTYPE; \
                    ( post_ops->matrix_add )->matrix = malloc( m * n * sizeof(int8_t) ); \
                    if ( ( post_ops->matrix_add )->matrix == NULL ) \
                    { \
                        goto err_handler; \
                    } \
                    GEN_FUNC_NAME(fill_array_,int8_t)( ( post_ops->matrix_add )->matrix, ( m * n ) ); \
                } \
                else \
                { \
                    ( post_ops->matrix_add )->stor_type = NULLTYPE; \
                    ( post_ops->matrix_add )->matrix = malloc( m * n * sizeof(int32_t) ); \
                    if ( ( post_ops->matrix_add )->matrix == NULL ) \
                    { \
                        goto err_handler; \
                    } \
                    GEN_FUNC_NAME(fill_array_,int32_t)( ( post_ops->matrix_add )->matrix, ( m * n ) ); \
                } \
            } \
            else \
            { \
                if( global_dscale_out == 'y' ) \
                { \
                    ( post_ops->matrix_add )->stor_type = NULLTYPE; \
                    ( post_ops->matrix_add )->matrix = malloc( m * n * sizeof(C_DSCALE_type) ); \
                    if ( ( post_ops->matrix_add )->matrix == NULL ) \
                    { \
                        goto err_handler; \
                    } \
                    GEN_FUNC_NAME(fill_array_,C_DSCALE_type)( ( post_ops->matrix_add )->matrix, ( m * n ) ); \
                } \
                else \
                { \
                    ( post_ops->matrix_add )->stor_type = NULLTYPE; \
                    ( post_ops->matrix_add )->matrix = malloc( m * n * sizeof(float) ); \
                    if ( ( post_ops->matrix_add )->matrix == NULL ) \
                    { \
                        goto err_handler; \
                    } \
                    GEN_FUNC_NAME(fill_array_,float)( ( post_ops->matrix_add )->matrix, ( m * n ) ); \
                } \
            } \
        } \
        if ( ( stor_order == 'C' ) || ( stor_order == 'c' ) ) \
        { \
            ( post_ops->matrix_add )->ldm = m; \
        } \
        else \
        { \
            ( post_ops->matrix_add )->ldm = n; \
        } \
 \
        /* Allocate scale buffer. */ \
        dim_t n_scale = 1; /* Support for vector will be added. */ \
        ( post_ops->matrix_add )->scale_factor = malloc( n_scale * sizeof( DSCALE_type ) ); \
        if ( ( post_ops->matrix_add )->scale_factor == NULL ) \
        { \
            goto err_handler; \
        } \
        /* Fill scale factor. */ \
        DSCALE_type* temp_dscale_ptr = ( DSCALE_type* )( post_ops->matrix_add )->scale_factor; \
        for ( dim_t i = 0; i < n_scale; ++i ) \
        { \
            temp_dscale_ptr[i] = ( ( DSCALE_type )2 ); \
        } \
        ( post_ops->matrix_add )->scale_factor_len = n_scale; \
    } \
 \
    if ( is_matrix_mul == TRUE ) \
    { \
        /* Bench limitation: can only support 1 matrix mul, but LPGEMM can support
        * multiple scale post-ops. */ \
        post_ops->matrix_mul = malloc( sizeof( aocl_post_op_matrix_mul ) ); \
        if ( post_ops->matrix_mul == NULL ) \
        { \
            goto err_handler; \
        } \
        if( is_matmul_stor_type == TRUE) \
        { \
            if( ( strcmp( matmul_stor_type, "BF16" ) == 0 ) ) \
            { \
                ( post_ops->matrix_mul )->stor_type = AOCL_GEMM_BF16; \
                ( post_ops->matrix_mul )->matrix = malloc( m * n * sizeof(bfloat16) ); \
                if ( ( post_ops->matrix_mul )->matrix == NULL ) \
                { \
                    goto err_handler; \
                } \
                GEN_FUNC_NAME(fill_array_,bfloat16)( ( post_ops->matrix_mul )->matrix, ( m * n ) ); \
            } \
            else if( ( strcmp( matmul_stor_type, "F32" ) == 0 ) ) \
            { \
                ( post_ops->matrix_mul )->stor_type = AOCL_GEMM_F32; \
                ( post_ops->matrix_mul )->matrix = malloc( m * n * sizeof(float) ); \
                if ( ( post_ops->matrix_mul )->matrix == NULL ) \
                { \
                    goto err_handler; \
                } \
                GEN_FUNC_NAME(fill_array_,float)( ( post_ops->matrix_mul )->matrix, ( m * n ) ); \
            } \
            else if( ( strcmp( matmul_stor_type, "S32" ) == 0 ) ) \
            { \
                ( post_ops->matrix_mul )->stor_type = AOCL_GEMM_INT32; \
                ( post_ops->matrix_mul )->matrix = malloc( m * n * sizeof(int32_t) ); \
                if ( ( post_ops->matrix_mul )->matrix == NULL ) \
                { \
                    goto err_handler; \
                } \
                GEN_FUNC_NAME(fill_array_,int32_t)( ( post_ops->matrix_mul )->matrix, ( m * n ) ); \
            } \
            else if( ( strcmp( matmul_stor_type, "S8" ) == 0 ) ) \
            { \
                ( post_ops->matrix_mul )->stor_type = AOCL_GEMM_INT8; \
                ( post_ops->matrix_mul )->matrix = malloc( m * n * sizeof(int8_t) ); \
                if ( ( post_ops->matrix_mul )->matrix == NULL ) \
                { \
                    goto err_handler; \
                } \
                GEN_FUNC_NAME(fill_array_,int8_t)( ( post_ops->matrix_mul )->matrix, ( m * n ) ); \
            } \
            else {} \
        } \
        else \
        { \
            /*  default is int32_t for integer APIs and float for others */ \
            if( is_integerAPI_avx512(#BLAS_SFX)) \
            { \
                if( strcmp(#C_type, "int8_t") == 0 ) \
                { \
                    ( post_ops->matrix_mul )->stor_type = NULLTYPE; \
                    ( post_ops->matrix_mul )->matrix = malloc( m * n * sizeof(int8_t) ); \
                    if ( ( post_ops->matrix_mul )->matrix == NULL ) \
                    { \
                        goto err_handler; \
                    } \
                    GEN_FUNC_NAME(fill_array_,int8_t)( ( post_ops->matrix_mul )->matrix, ( m * n ) ); \
                } \
                else \
                { \
                    ( post_ops->matrix_mul )->stor_type = NULLTYPE; \
                    ( post_ops->matrix_mul )->matrix = malloc( m * n * sizeof(int32_t) ); \
                    if ( ( post_ops->matrix_mul )->matrix == NULL ) \
                    { \
                        goto err_handler; \
                    } \
                    GEN_FUNC_NAME(fill_array_,int32_t)( ( post_ops->matrix_mul )->matrix, ( m * n ) ); \
                } \
            } \
            else \
            { \
                if( global_dscale_out == 'y' ) \
                { \
                    ( post_ops->matrix_mul )->stor_type = NULLTYPE; \
                    ( post_ops->matrix_mul )->matrix = malloc( m * n * sizeof(C_DSCALE_type) ); \
                    if ( ( post_ops->matrix_mul )->matrix == NULL ) \
                    { \
                        goto err_handler; \
                    } \
                    GEN_FUNC_NAME(fill_array_,C_DSCALE_type)( ( post_ops->matrix_mul )->matrix, ( m * n ) ); \
                } \
                else \
                { \
                    ( post_ops->matrix_mul )->stor_type = NULLTYPE; \
                    ( post_ops->matrix_mul )->matrix = malloc( m * n * sizeof(float) ); \
                    if ( ( post_ops->matrix_mul )->matrix == NULL ) \
                    { \
                        goto err_handler; \
                    } \
                    GEN_FUNC_NAME(fill_array_,float)( ( post_ops->matrix_mul )->matrix, ( m * n ) ); \
                } \
            } \
        } \
        if ( ( stor_order == 'C' ) || ( stor_order == 'c' ) ) \
        { \
            ( post_ops->matrix_mul )->ldm = m; \
        } \
        else \
        { \
            ( post_ops->matrix_mul )->ldm = n; \
        } \
 \
        /* Allocate scale buffer. */ \
        dim_t n_scale = 1; /* Support for vector will be added. */ \
        ( post_ops->matrix_mul )->scale_factor = malloc( n_scale * sizeof( DSCALE_type ) ); \
        if ( ( post_ops->matrix_mul )->scale_factor == NULL ) \
        { \
            goto err_handler; \
        } \
        /* Fill scale factor. */ \
        DSCALE_type* temp_dscale_ptr = ( DSCALE_type* )( post_ops->matrix_mul )->scale_factor; \
        for ( dim_t i = 0; i < n_scale; ++i ) \
        { \
            temp_dscale_ptr[i] = ( ( DSCALE_type )2 ); \
        } \
        ( post_ops->matrix_mul )->scale_factor_len = n_scale; \
    } \
 \
    post_ops->seq_length = cur_op_index; \
 \
     /* ---------------- Setup the pre_ops struct ----------------------------- */ \
    post_ops->pre_ops = NULL; \
    if ( global_pre_op == 'y' ) \
    { \
        post_ops->pre_ops = malloc( sizeof( aocl_pre_op ) ); \
        if ( post_ops->pre_ops == NULL ) { goto err_handler; } \
\
        dim_t num_groups = 1; \
        if (quant_group_size == 0) \
        { \
            post_ops->pre_ops->group_size = k; \
        } \
        else \
        { \
            post_ops->pre_ops->group_size = quant_group_size; \
            if(is_group_quant) \
            { \
                num_groups = ( k + post_ops->pre_ops->group_size - 1 ) / post_ops->pre_ops->group_size; \
            } \
        } \
\
        ( post_ops->pre_ops )->b_zp = NULL; \
        if ( zp_vec_length == 0 ) \
        { \
            zp_vec_length = n; \
        } \
        if( zp_vec_length != 0 ) \
        { \
            ( post_ops->pre_ops )->b_zp = malloc( sizeof( aocl_pre_op_zp ) ); \
            if ( ( post_ops->pre_ops )->b_zp == NULL ) { goto err_handler; } \
            ( ( post_ops->pre_ops )->b_zp )->zero_point = malloc( num_groups * zp_vec_length * sizeof( int8_t ) ); \
            if ( ( ( post_ops->pre_ops )->b_zp )->zero_point == NULL ) { goto err_handler; } \
            for ( dim_t i = 0; i < num_groups * zp_vec_length; ++i ) \
            { \
                ( ( int8_t* )( ( post_ops->pre_ops )->b_zp )->zero_point )[i] = ( int8_t )( ( i + 1 ) % 5 ); \
            } \
            ( ( post_ops->pre_ops )->b_zp )->zero_point_len = zp_vec_length; \
        } \
\
        ( post_ops->pre_ops )->b_scl = malloc( sizeof( aocl_pre_op_sf ) ); \
        if ( ( post_ops->pre_ops )->b_scl == NULL ) { goto err_handler; } \
        dim_t scale_factor_len  = (is_pre_op_scale_scalar == TRUE)? 1 : n; \
        ( ( post_ops->pre_ops )->b_scl )->scale_factor_len = scale_factor_len; \
        if( is_pre_op_scale_f32 ) \
        { \
            ( ( post_ops->pre_ops )->b_scl )->scale_factor = malloc( num_groups * scale_factor_len * sizeof( float ) ); \
            if ( ( ( post_ops->pre_ops )->b_scl )->scale_factor == NULL ) { goto err_handler; } \
            GEN_FUNC_NAME(fill_array_,float)( ( ( post_ops->pre_ops )->b_scl )->scale_factor, num_groups * scale_factor_len ); \
            ((post_ops->pre_ops)->b_scl)->scale_factor_type = AOCL_GEMM_F32; \
        } \
        else \
        { \
            ( ( post_ops->pre_ops )->b_scl )->scale_factor = malloc( num_groups * scale_factor_len * sizeof( bfloat16 ) ); \
            if ( ( ( post_ops->pre_ops )->b_scl )->scale_factor == NULL ) { goto err_handler; } \
            GEN_FUNC_NAME(fill_array_,bfloat16)( ( ( post_ops->pre_ops )->b_scl )->scale_factor, num_groups * scale_factor_len ); \
            ((post_ops->pre_ops)->b_scl)->scale_factor_type = AOCL_GEMM_BF16; \
        } \
 \
         ( post_ops->pre_ops )->seq_length = 1; \
    } \
 \
    /* ------------------------ Setup group level post ops ---------------------*/ \
    bool is_symm = ( strcmp(#BLAS_SFX, "s8s8s32of32_sym_quant") == 0 ) || ( strcmp(#BLAS_SFX, "s8s8s32obf16_sym_quant") == 0); \
    if( is_symm && is_sym_quant ) \
    { \
        post_ops->post_op_grp = malloc(sizeof(aocl_group_post_op)); \
        if( post_ops->post_op_grp == NULL ) { goto err_handler; } \
        /* Initializing both zero_points to NULL */ \
        /* They'll be initialized  in asymm section */ \
        post_ops->post_op_grp->a_zp = NULL; \
        post_ops->post_op_grp->b_zp = NULL; \
        dim_t num_groups = 1; \
        if( quant_group_size == 0 ) \
        { \
            post_ops->post_op_grp->group_size = k; \
        } \
        else \
        { \
            post_ops->post_op_grp->group_size = quant_group_size; \
            if(is_group_quant) \
            { \
                num_groups = ( k + post_ops->post_op_grp->group_size - 1 ) / post_ops->post_op_grp->group_size; \
            } \
        } \
 \
        /* Assuming that scale factors and zero points are always vectors */ \
        /* A sf & zp are of size m x num_groups */ \
        /* B sf and zp are of size num_groups * n */ \
        /* All the above as assumed to be row-major matrices of respective sizes */ \
        /* All the above are of bfloat16 type by default */ \
 \
        /* Filling scale_factor for A & B matrices */ \
        post_ops->post_op_grp->a_scl = malloc(sizeof(aocl_pre_op_sf)); \
        if( post_ops->post_op_grp->a_scl == NULL ) { goto err_handler; } \
        post_ops->post_op_grp->b_scl = malloc(sizeof(aocl_pre_op_sf)); \
        if( post_ops->post_op_grp->b_scl == NULL ) { goto err_handler; } \
        if( strcmp( sym_quant_sf_type, "BF16" ) == 0 ) \
        { \
            post_ops->post_op_grp->a_scl->scale_factor = malloc(m * num_groups * sizeof(bfloat16)); \
            if( post_ops->post_op_grp->a_scl->scale_factor == NULL ) { goto err_handler; } \
            GEN_FUNC_NAME(fill_array_,bfloat16)(post_ops->post_op_grp->a_scl->scale_factor, m * num_groups); \
            post_ops->post_op_grp->a_scl->scale_factor_type = AOCL_GEMM_BF16; \
            post_ops->post_op_grp->b_scl->scale_factor = malloc(n * num_groups * sizeof(bfloat16)); \
            if( post_ops->post_op_grp->b_scl->scale_factor == NULL ) { goto err_handler; } \
            GEN_FUNC_NAME(fill_array_,bfloat16)(post_ops->post_op_grp->b_scl->scale_factor, num_groups * n); \
            post_ops->post_op_grp->b_scl->scale_factor_type = AOCL_GEMM_BF16; \
        } \
        else \
        { \
            post_ops->post_op_grp->a_scl->scale_factor = malloc(m * num_groups * sizeof(float)); \
            if( post_ops->post_op_grp->a_scl->scale_factor == NULL ) { goto err_handler; } \
            GEN_FUNC_NAME(fill_array_,float)(post_ops->post_op_grp->a_scl->scale_factor, m * num_groups); \
            /* for( dim_t i = 0; i < num_groups * m; i++) { \
                ((float*)post_ops->post_op_grp->a_scl->scale_factor)[i] = 1.0; \
            }  */\
            post_ops->post_op_grp->a_scl->scale_factor_type = AOCL_GEMM_F32; \
            post_ops->post_op_grp->b_scl->scale_factor = malloc(n * num_groups * sizeof(float)); \
            if( post_ops->post_op_grp->b_scl->scale_factor == NULL ) { goto err_handler; } \
            GEN_FUNC_NAME(fill_array_,float)(post_ops->post_op_grp->b_scl->scale_factor, num_groups * n); \
            /*for( dim_t i = 0; i < num_groups * n; i++) { \
                ((float*)post_ops->post_op_grp->b_scl->scale_factor)[i] = 1.0; \
            } */\
            post_ops->post_op_grp->b_scl->scale_factor_type = AOCL_GEMM_F32; \
        } \
        post_ops->post_op_grp->a_scl->scale_factor_len = m; \
        post_ops->post_op_grp->b_scl->scale_factor_len = n; \
        ( post_ops->post_op_grp )->seq_length = 1; \
    } \
    return post_ops; \
 \
    err_handler: \
    lpgemm_destroy_post_ops_struct( post_ops ); \
    return NULL; \
} \

#endif //LPGEMM_BENCH_UTILS_H
