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

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <immintrin.h>
#include <time.h>
#include <float.h>
#include <unistd.h>
#include <math.h>

#include "blis.h"


// Used to clip downscaled output, will be set in the main loop based
// on the accumulation and C data type.
int64_t DSCALE_CLIP_MIN = 0;
int64_t DSCALE_CLIP_MAX = 0;

// Mode can be one of the follwoing:
// 	1. p - performance, used for benchmarks.
// 	2. a - accuracy, used to test accuracy/correctness.
// Default value is p, can be modified by passing command line arg.
char bench_mode = 'p';

int32_t global_n_repeat = 0;

char global_dscale_out = 'n';

dim_t num_eltwise = 0; // To keep track of eltwise operations.

#define _XSTR(str) #str
#define XSTR(str) _XSTR(str)

#define GEN_FUNC_NAME(prototype,ctype) prototype ## ctype

static inline void float_to_bf16( float* float_value, bfloat16* bf16_val )
{
    /*Set offset 2 to copy most significant 2 bytes of float
    to convert float values to bf16 values*/
    memcpy( ( bf16_val ), (char *)( float_value ) + 2, sizeof ( bfloat16 ) );
}

static inline void convert_float_arr_to_bf16( float* array, bfloat16* array_bf16, int size )
{
    for (int i=0; i< size; i++)
    {
        float_to_bf16( ( array + i ), ( array_bf16 + i ) );
    }
}


static inline void bfloat16_to_float( bfloat16 bf16_val, float*  float_val )
{
    int32_t inter_temp = *( ( int16_t* ) &bf16_val );
    inter_temp = inter_temp << 16;
    memcpy( float_val, &inter_temp, sizeof( int32_t ) );
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
            printf("%f ",  temp);
        }
        printf("\n");
    }
}

#define PRINT_MATRIX(ctype) \
void print_matrix_## ctype ( ctype* a, int32_t m, int32_t n, int32_t rs, int32_t cs) \
{ \
    for(int32_t i = 0; i < m; i++) \
    { \
        for(int32_t j = 0; j < n; j++) \
        { \
            printf("%f ", (float) (*(a + i * ( rs ) + j * cs ) ) ); \
        } \
        printf("\n"); \
    } \
} \

PRINT_MATRIX(uint8_t)
PRINT_MATRIX(int8_t)
PRINT_MATRIX(int16_t)
PRINT_MATRIX(float)
PRINT_MATRIX(int32_t)

void* lpgemm_malloc( int32_t size )
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

void lpgemm_free( void* p )
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

#define GEN_FILL_ARRAY_FUNC(ctype) \
void fill_array_ ## ctype ( void* arr, dim_t size ) \
{ \
    if( size < 0 ) return; \
    ctype* temp_arr = ( ctype* ) arr; \
    for ( dim_t i = 0; i < size; ++i ) \
    { \
        temp_arr[i] = ( ctype )( i % 5 ); \
    } \
} \

GEN_FILL_ARRAY_FUNC(uint8_t)
GEN_FILL_ARRAY_FUNC(int8_t)
GEN_FILL_ARRAY_FUNC(int16_t)
GEN_FILL_ARRAY_FUNC(float)
GEN_FILL_ARRAY_FUNC(int32_t)

void fill_array_bfloat16( void* arr, dim_t size )
{
    err_t bli_errors = BLIS_SUCCESS;
    if( size < 0 ) return;
    float* c_float = ( float* ) bli_malloc_user( sizeof( float ) * size, &bli_errors );
    for ( dim_t i = 0; i < size; ++i )
    {
        c_float[i] = i % 5;
    }
    convert_float_arr_to_bf16( c_float, arr, size );
    if ( c_float != NULL )
    {
        bli_free_user( c_float );
    }
}

#define GEN_FILL_ARRAY_POST_OPS_FUNC(ctype) \
void fill_array_post_ops_ ## ctype ( void* arr, dim_t size ) \
{ \
    ctype* temp_arr = ( ctype* ) arr; \
    for ( dim_t i = 0; i < size; ++i ) \
    { \
        temp_arr[i] = ( ctype )( i % 20 ); \
    } \
} \

GEN_FILL_ARRAY_POST_OPS_FUNC(int16_t)
GEN_FILL_ARRAY_POST_OPS_FUNC(int32_t)
GEN_FILL_ARRAY_POST_OPS_FUNC(float)

#define GEN_BLIS_MAT_MUL_FUNC(A_type,B_type,C_type,ACCUM_type,BLAS_SFX) \
void mat_mul_ ## BLAS_SFX \
     ( \
       char    stor_order, \
       char    transa, \
       char    transb,  \
       char    op_a, \
       char    op_b, \
       dim_t   m, \
       dim_t   n, \
       dim_t   k, \
       ACCUM_type  alpha, \
       A_type* a, \
       dim_t   lda, \
       B_type* b, \
       dim_t   ldb, \
       ACCUM_type  beta, \
       C_type* c, \
       dim_t   ldc, \
       aocl_post_op*  post_op\
     ) \
{ \
    aocl_gemm_ ## BLAS_SFX( stor_order, transa, transb, m, n, k, \
                    alpha, \
                    a, lda, op_a, \
                    b, ldb, op_b, \
                    beta, \
                    c, ldc, post_op ); \
 \
    /*dim_t MR = 6; \
    dim_t NR = 16; \
 \
    __m512i selector1; \
    __m512i all_zero = _mm512_setzero_epi32(); \
    __m512i c0; \
    __m512i c1; \
    __m512i c2; \
    __m512i c3; \
    __m512i c4; \
    __m512i c5; \
 \
    for ( dim_t i = 0; i < m; i += MR ) \
    { \
        if ( ( i + MR ) > m ) \
        { \
            break; \
        } \
        for ( dim_t j = 0; j < n; j += NR ) \
        { \
            if ( ( j + NR ) > n ) \
            { \
                break; \
            } \
            selector1 = _mm512_loadu_epi32( (int32_t*)post_op->bias.bias + j ); \
            c0 = _mm512_loadu_epi32( c + ( ( i + 0 ) * ldc ) + j ); \
            c1 = _mm512_loadu_epi32( c + ( ( i + 1 ) * ldc ) + j ); \
            c2 = _mm512_loadu_epi32( c + ( ( i + 2 ) * ldc ) + j ); \
            c3 = _mm512_loadu_epi32( c + ( ( i + 3 ) * ldc ) + j ); \
            c4 = _mm512_loadu_epi32( c + ( ( i + 4 ) * ldc ) + j ); \
            c5 = _mm512_loadu_epi32( c + ( ( i + 5 ) * ldc ) + j ); \
 \
            c0 = _mm512_add_epi32( selector1, c0 ); \
            c1 = _mm512_add_epi32( selector1, c1 ); \
            c2 = _mm512_add_epi32( selector1, c2 ); \
            c3 = _mm512_add_epi32( selector1, c3 ); \
            c4 = _mm512_add_epi32( selector1, c4 ); \
            c5 = _mm512_add_epi32( selector1, c5 ); \
 \
            c0 = _mm512_max_epi32( all_zero, c0 ); \
            c1 = _mm512_max_epi32( all_zero, c1 ); \
            c2 = _mm512_max_epi32( all_zero, c2 ); \
            c3 = _mm512_max_epi32( all_zero, c3 ); \
            c4 = _mm512_max_epi32( all_zero, c4 ); \
            c5 = _mm512_max_epi32( all_zero, c5 ); \
 \
            _mm512_storeu_epi32( c + ( ( i + 0 ) * ldc ) + j, c0 ); \
            _mm512_storeu_epi32( c + ( ( i + 1 ) * ldc ) + j, c1 ); \
            _mm512_storeu_epi32( c + ( ( i + 2 ) * ldc ) + j, c2 ); \
            _mm512_storeu_epi32( c + ( ( i + 3 ) * ldc ) + j, c3 ); \
            _mm512_storeu_epi32( c + ( ( i + 4 ) * ldc ) + j, c4 ); \
            _mm512_storeu_epi32( c + ( ( i + 5 ) * ldc ) + j, c5 ); \
        } \
    } */\
} \

GEN_BLIS_MAT_MUL_FUNC(uint8_t,int8_t,int16_t,int16_t,u8s8s16os16)
GEN_BLIS_MAT_MUL_FUNC(uint8_t,int8_t,int8_t,int16_t,u8s8s16os8)
GEN_BLIS_MAT_MUL_FUNC(uint8_t,int8_t,uint8_t,int16_t,u8s8s16ou8)
GEN_BLIS_MAT_MUL_FUNC(uint8_t,int8_t,int32_t,int32_t,u8s8s32os32)
GEN_BLIS_MAT_MUL_FUNC(uint8_t,int8_t,int8_t,int32_t,u8s8s32os8)
GEN_BLIS_MAT_MUL_FUNC(bfloat16,bfloat16,float,float,bf16bf16f32of32)
GEN_BLIS_MAT_MUL_FUNC(bfloat16,bfloat16,bfloat16,float,bf16bf16f32obf16)
GEN_BLIS_MAT_MUL_FUNC(float,float,float,float,f32f32f32of32)
GEN_BLIS_MAT_MUL_FUNC(int8_t,int8_t,int32_t,int32_t,s8s8s32os32)
GEN_BLIS_MAT_MUL_FUNC(int8_t,int8_t,int8_t,int32_t,s8s8s32os8)
GEN_BLIS_MAT_MUL_FUNC(int8_t,int8_t,int16_t,int16_t,s8s8s16os16)
GEN_BLIS_MAT_MUL_FUNC(int8_t,int8_t,int8_t,int16_t,s8s8s16os8)

double get_gflops
     (
       dim_t  m,
       dim_t  n,
       dim_t  k,
       double runtime
     )
{
    return ( ( 2.0 * m * n * k ) / ( runtime * 1.0e9 ) );
}

void print_result
     (
       const char* msg,
       int32_t     n_repeats,
       char        transa,
       char        transb,
       dim_t       m,
       dim_t       n,
       dim_t       k,
       dim_t       lda,
       dim_t       ldb,
       dim_t       ldc,
       double      runtime
     )
{
    double gflops = get_gflops( m, n, k, runtime );
    printf("%s transa:%c, transb:%c, m: %ld, n: %ld, k: %ld, lda: %ld, ldb: %ld, ldc: %ld," \
                    " Gops: %f, n_repeats: %d\n",
            msg, transa, transb, m, n, k, lda, ldb, ldc, gflops, n_repeats);
}

#define GEN_MAT_MUL_BENCH_DRV_FUNC(A_type,B_type,C_type,ACCUM_type,BLAS_SFX) \
void mat_mul_bench_driver_ ## BLAS_SFX \
     ( \
       char    stor_order, \
       char    transa, \
       char    transb, \
       char    op_a, \
       char    op_b, \
       int32_t n_repeats, \
       dim_t   m, \
       dim_t   n, \
       dim_t   k, \
       ACCUM_type  alpha, \
       A_type* a, \
       dim_t   lda, \
       B_type* b, \
       dim_t   ldb, \
       ACCUM_type  beta, \
       C_type* c, \
       dim_t   ldc, \
       aocl_post_op*  post_op\
     ) \
{ \
    double min_time_diff = DBL_MAX; \
    for ( int32_t nr = 0; nr < n_repeats; ++nr ) \
    { \
        if ( bench_mode == 'a' ) \
        { \
            int32_t size_C = ( ( stor_order == 'r') || ( stor_order == 'R' ) )? m * ldc : n * ldc; \
            GEN_FUNC_NAME(fill_array_,C_type)( c, ( size_C ) ); \
        } \
 \
        struct timespec tstart={0,0}, tend={0,0}; \
        clock_gettime(CLOCK_MONOTONIC, &tstart); \
 \
        GEN_FUNC_NAME(mat_mul_,BLAS_SFX) \
        ( \
          stor_order, transa, transb, op_a, op_b, m, n, k, \
          alpha, \
          a, lda, \
          b, ldb, \
          beta, \
          c, ldc, \
          post_op \
        ); \
 \
        clock_gettime(CLOCK_MONOTONIC, &tend); \
 \
        double diff = \
            ( ( double ) tend.tv_sec + ( 1.0e-9 * tend.tv_nsec ) ) - \
            ( ( double ) tstart.tv_sec + ( 1.0e-9 * tstart.tv_nsec ) ); \
        min_time_diff = ( diff < min_time_diff ) ? diff : min_time_diff; \
    } \
 \
    print_result( XSTR(BLAS_SFX), n_repeats, transa, transb, m, n, k, lda, ldb, ldc, min_time_diff); \
} \

GEN_MAT_MUL_BENCH_DRV_FUNC(uint8_t,int8_t,int16_t,int16_t,u8s8s16os16)
GEN_MAT_MUL_BENCH_DRV_FUNC(uint8_t,int8_t,int8_t,int16_t,u8s8s16os8)
GEN_MAT_MUL_BENCH_DRV_FUNC(uint8_t,int8_t,uint8_t,int16_t,u8s8s16ou8)
GEN_MAT_MUL_BENCH_DRV_FUNC(uint8_t,int8_t,int32_t,int32_t,u8s8s32os32)
GEN_MAT_MUL_BENCH_DRV_FUNC(uint8_t,int8_t,int8_t,int32_t,u8s8s32os8)
GEN_MAT_MUL_BENCH_DRV_FUNC(bfloat16,bfloat16,float,float,bf16bf16f32of32)
GEN_MAT_MUL_BENCH_DRV_FUNC(bfloat16,bfloat16,bfloat16,float,bf16bf16f32obf16)
GEN_MAT_MUL_BENCH_DRV_FUNC(float,float,float,float,f32f32f32of32)
GEN_MAT_MUL_BENCH_DRV_FUNC(int8_t,int8_t,int32_t,int32_t,s8s8s32os32)
GEN_MAT_MUL_BENCH_DRV_FUNC(int8_t,int8_t,int8_t,int32_t,s8s8s32os8)
GEN_MAT_MUL_BENCH_DRV_FUNC(int8_t,int8_t,int16_t,int16_t,s8s8s16os16)
GEN_MAT_MUL_BENCH_DRV_FUNC(int8_t,int8_t,int8_t,int16_t,s8s8s16os8)

int max (int a, int b)
{
    return ( a > b ? a : b );
}

int min (int a, int b)
{
    return ( a < b ? a : b );
}

#define GEN_MAT_MUL_ACC_CHK_DOWNSCALE(C_type,ACCUM_type,SCALE_type,BLAS_DOWNSCALE_SFX) \
static inline ACCUM_type mat_mul_accuracy_check_downscale_ ## BLAS_DOWNSCALE_SFX \
     (\
       ACCUM_type temp_accum,\
       aocl_post_op*  post_op, \
       dim_t j \
     )\
{ \
    ACCUM_type out_temp_accum = \
        ( ACCUM_type )min( \
                        max( nearbyintf( ( SCALE_type )( temp_accum ) * \
                            ( *( ( SCALE_type* )post_op->sum.scale_factor + j ) ) ) + \
                            *( ( C_type* )post_op->sum.zero_point + j ), \
                            DSCALE_CLIP_MIN ), \
                        DSCALE_CLIP_MAX ); \
    return 	out_temp_accum; \
}\

GEN_MAT_MUL_ACC_CHK_DOWNSCALE(int8_t,int16_t,float,u8s8s16os8)
GEN_MAT_MUL_ACC_CHK_DOWNSCALE(uint8_t,int16_t,float,u8s8s16ou8)
GEN_MAT_MUL_ACC_CHK_DOWNSCALE(int8_t,int32_t,float,u8s8s32os8)
GEN_MAT_MUL_ACC_CHK_DOWNSCALE(int8_t,int32_t,float,s8s8s32os8)
GEN_MAT_MUL_ACC_CHK_DOWNSCALE(int8_t,int16_t,float,s8s8s16os8)

static inline float mat_mul_accuracy_check_downscale_bf16bf16f32obf16
     (
       float temp_accum,
       aocl_post_op*  post_op,
       dim_t j
     )
{
    return temp_accum;
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
       dim_t k \
     )\
{\
    for ( dim_t p = 0; p < k; ++p) \
    { \
        temp_accum += ( *( a + ( i * rs_a ) + ( cs_a * p ) ) * \
                         *( b + ( rs_b * p ) + ( cs_b * j ) ) ); \
    } \
\
    temp_accum = ( beta * ( * (c_ref + ( rs_c_ref * i ) + ( cs_c_ref * j ) ) ) ) \
                         + ( alpha * temp_accum ); \
    return temp_accum; \
}\

GEN_MAT_MUL_ACC_CHK_ACCUM(uint8_t,int8_t,int8_t,int16_t,u8s8s16os8)
GEN_MAT_MUL_ACC_CHK_ACCUM(uint8_t,int8_t,uint8_t,int16_t,u8s8s16ou8)
GEN_MAT_MUL_ACC_CHK_ACCUM(uint8_t,int8_t,int16_t,int16_t,u8s8s16os16)
GEN_MAT_MUL_ACC_CHK_ACCUM(uint8_t,int8_t,int8_t,int32_t,u8s8s32os8)
GEN_MAT_MUL_ACC_CHK_ACCUM(uint8_t,int8_t,int32_t,int32_t,u8s8s32os32)
GEN_MAT_MUL_ACC_CHK_ACCUM(float,float,float,float,f32f32f32of32)
GEN_MAT_MUL_ACC_CHK_ACCUM(int8_t,int8_t,int8_t,int32_t,s8s8s32os8)
GEN_MAT_MUL_ACC_CHK_ACCUM(int8_t,int8_t,int32_t,int32_t,s8s8s32os32)
GEN_MAT_MUL_ACC_CHK_ACCUM(int8_t,int8_t,int8_t,int16_t,s8s8s16os8)
GEN_MAT_MUL_ACC_CHK_ACCUM(int8_t,int8_t,int16_t,int16_t,s8s8s16os16)

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
       dim_t k
     )
{
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
       dim_t k
     )
{
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

#define GEN_GELU_TANH_POSTOP_INT(ACCUM_type,BLAS_SFX) \
static inline ACCUM_type GELU_TANH_post_op_ ## BLAS_SFX \
     (\
       ACCUM_type temp_accum \
     )\
{\
    float gelu_reference = 0.5 *(double)temp_accum * (1 + tanhf( 0.797884 * ( (double)temp_accum + \
                    ( 0.044715 * ((double)temp_accum * (double)temp_accum * \
                    (double)temp_accum ) ) ) ) ); \
    temp_accum = round (gelu_reference); \
    return temp_accum; \
}\

GEN_GELU_TANH_POSTOP_INT(int16_t,u8s8s16os8)
GEN_GELU_TANH_POSTOP_INT(int16_t,u8s8s16ou8)
GEN_GELU_TANH_POSTOP_INT(int16_t,u8s8s16os16)
GEN_GELU_TANH_POSTOP_INT(int32_t,u8s8s32os8)
GEN_GELU_TANH_POSTOP_INT(int32_t,u8s8s32os32)
GEN_GELU_TANH_POSTOP_INT(int32_t,s8s8s32os8)
GEN_GELU_TANH_POSTOP_INT(int32_t,s8s8s32os32)
GEN_GELU_TANH_POSTOP_INT(int16_t,s8s8s16os8)
GEN_GELU_TANH_POSTOP_INT(int16_t,s8s8s16os16)

#define GEN_GELU_TANH_POSTOP_FLOAT(BLAS_SFX) \
static inline float GELU_TANH_post_op_ ## BLAS_SFX \
     (\
       float temp_accum \
     )\
{\
    temp_accum = 0.5 *(double)temp_accum * (1 + tanhf( 0.797884 * ( (double)temp_accum + \
                  ( 0.044715 * ((double)temp_accum * (double)temp_accum * \
                  (double)temp_accum ) ) ) ) ); \
    return temp_accum; \
}\

GEN_GELU_TANH_POSTOP_FLOAT(f32f32f32of32)
GEN_GELU_TANH_POSTOP_FLOAT(bf16bf16f32of32)
GEN_GELU_TANH_POSTOP_FLOAT(bf16bf16f32obf16)

#define GEN_GELU_ERF_POSTOP_INT(ACCUM_type,BLAS_SFX) \
static inline ACCUM_type GELU_ERF_post_op_ ## BLAS_SFX \
     (\
       ACCUM_type temp_accum \
     )\
{\
    float gelu_reference = 0.5 *(double)temp_accum * (1 + erff( (double)temp_accum * 0.707107 )); \
    temp_accum = round (gelu_reference); \
    return temp_accum; \
}\

GEN_GELU_ERF_POSTOP_INT(int16_t,u8s8s16os8)
GEN_GELU_ERF_POSTOP_INT(int16_t,u8s8s16ou8)
GEN_GELU_ERF_POSTOP_INT(int16_t,u8s8s16os16)
GEN_GELU_ERF_POSTOP_INT(int32_t,u8s8s32os8)
GEN_GELU_ERF_POSTOP_INT(int32_t,u8s8s32os32)
GEN_GELU_ERF_POSTOP_INT(int32_t,s8s8s32os8)
GEN_GELU_ERF_POSTOP_INT(int32_t,s8s8s32os32)
GEN_GELU_ERF_POSTOP_INT(int16_t,s8s8s16os8)
GEN_GELU_ERF_POSTOP_INT(int16_t,s8s8s16os16)

#define GEN_GELU_ERF_POSTOP_FLOAT(BLAS_SFX) \
static inline float GELU_ERF_post_op_ ## BLAS_SFX \
     (\
       float temp_accum \
     )\
{\
    temp_accum = 0.5 *(double)temp_accum * (1 + erff( (double)temp_accum * 0.707107 )); \
    return temp_accum; \
}\

GEN_GELU_ERF_POSTOP_FLOAT(f32f32f32of32)
GEN_GELU_ERF_POSTOP_FLOAT(bf16bf16f32of32)
GEN_GELU_ERF_POSTOP_FLOAT(bf16bf16f32obf16)

#define GEN_MAT_MUL_GET_OUTPUT_TYPE_VALUE(C_type, ACCUM_type) \
void mat_mul_get_output_type_val ## ACCUM_type ## C_type \
     ( \
       C_type* out_temp_accum, \
       ACCUM_type* temp_accum \
     ) \
{ \
    ( *out_temp_accum ) = ( C_type )( *temp_accum ); \
} \

GEN_MAT_MUL_GET_OUTPUT_TYPE_VALUE(int32_t,int32_t)
GEN_MAT_MUL_GET_OUTPUT_TYPE_VALUE(int8_t,int32_t)
GEN_MAT_MUL_GET_OUTPUT_TYPE_VALUE(int16_t,int16_t)
GEN_MAT_MUL_GET_OUTPUT_TYPE_VALUE(int8_t,int16_t)
GEN_MAT_MUL_GET_OUTPUT_TYPE_VALUE(uint8_t,int16_t)
GEN_MAT_MUL_GET_OUTPUT_TYPE_VALUE(float,float)

void mat_mul_get_output_type_valfloatbfloat16
     (
       bfloat16* out_temp_accum,
       float* temp_accum
     )
{
    float_to_bf16( temp_accum, out_temp_accum );
}

#define GEN_MAT_MUL_ACC_CHK_DRV_FUNC(A_type,B_type,C_type,ACCUM_type,SCALE_type,BLAS_SFX,BLAS_DOWNSCALE_SFX) \
void mat_mul_accuracy_check_driver_ ## BLAS_SFX \
     ( \
       FILE*   fout, \
       const char stor_order, \
       char    transa, \
       char    transb, \
       dim_t   m, \
       dim_t   n, \
       dim_t   k, \
       ACCUM_type  alpha, \
       A_type* a, \
       dim_t   lda, \
       B_type* b, \
       dim_t   ldb, \
       ACCUM_type  beta, \
       C_type* c, \
       dim_t   ldc, \
       C_type* c_ref, \
       dim_t   ldc_ref, \
       aocl_post_op*  post_op\
     ) \
{ \
    dim_t rs_a, cs_a; \
    if( ( transa == 'n' ) || ( transa == 'N' ) ) \
    { \
        rs_a = lda; \
        cs_a = 1; \
    } \
    else \
    { \
        rs_a = 1; \
        cs_a = lda; \
    } \
    dim_t rs_b, cs_b; \
    if( ( transb == 'n' ) || ( transb == 'N' ) ) \
    { \
        rs_b = ldb; \
        cs_b = 1; \
    } \
    else \
    { \
        rs_b = 1; \
        cs_b = ldb; \
    } \
    dim_t rs_c = ldc; \
    dim_t cs_c = 1; \
    dim_t rs_c_ref = ldc_ref; \
    dim_t cs_c_ref = 1; \
 \
    if ( ( stor_order == 'C' ) || ( stor_order == 'c' ) ) \
    { \
        if( transa == 'n' || transa == 'N') \
        { \
            rs_a = 1; \
            cs_a = lda; \
        } \
        else \
        { \
            rs_a = lda; \
            cs_a = 1; \
        } \
        if( ( transb == 'n' ) || ( transb == 'N' ) ) \
        { \
            rs_b = 1; \
            cs_b = ldb; \
        } \
        else \
        { \
            rs_b = ldb; \
            cs_b = 1; \
        } \
        rs_c = 1; \
        cs_c = ldc; \
        rs_c_ref = 1; \
        cs_c_ref = ldc_ref; \
    } \
 \
    for ( dim_t i = 0; i < m; ++i ) \
    { \
        for ( dim_t j = 0; j < n; ++j ) \
        { \
            ACCUM_type temp_accum = 0; \
            C_type out_temp_accum = 0; \
 \
            temp_accum = GEN_FUNC_NAME(mat_mul_accuracy_check_accum_,BLAS_SFX) \
                (a,b,c_ref,temp_accum,alpha,beta,rs_a,rs_b,cs_a,cs_b,rs_c_ref,cs_c_ref,i,j,k); \
\
            if ( post_op != NULL ) \
            { \
                dim_t ele_i = 0; \
                for ( dim_t op_id = 0; op_id < post_op->seq_length; ++op_id ) \
                { \
                    if ( post_op->seq_vector[op_id] == BIAS ) \
                    { \
                        temp_accum += ( *( ( ACCUM_type* )post_op->bias.bias + j ) ); \
                    } \
                    else if ( post_op->seq_vector[op_id] == ELTWISE ) \
                    { \
                        if ( ( post_op->eltwise + ele_i )->algo.algo_type == \
                                PRELU ) /* PReLU*/ \
                        { \
                            temp_accum = ( temp_accum > 0 ) ? \
                                temp_accum : \
                                ( temp_accum * \
                                *( ( ACCUM_type* ) ( post_op->eltwise + ele_i )->algo.alpha ) ); \
                            ele_i += 1; \
                        } \
                        else if ( ( post_op->eltwise + ele_i )->algo.algo_type == \
                                GELU_TANH ) /* TANH GeLU*/ \
                        { \
                            temp_accum = GEN_FUNC_NAME(GELU_TANH_post_op_,BLAS_SFX) (temp_accum);\
                            ele_i += 1; \
                        } \
                        else if ( ( post_op->eltwise + ele_i )->algo.algo_type == \
                                GELU_ERF ) /* ERF GeLU*/ \
                        { \
                            temp_accum = GEN_FUNC_NAME(GELU_ERF_post_op_,BLAS_SFX) (temp_accum);\
                            ele_i += 1; \
                        } \
                        else if ( ( post_op->eltwise + ele_i )->algo.algo_type == \
                                RELU ) /* ReLU*/ \
                        { \
                            temp_accum = ( temp_accum > 0 ) ? temp_accum : 0 ; \
                            ele_i += 1; \
                        } \
                        else if ( ( post_op->eltwise + ele_i )->algo.algo_type == \
                                CLIP ) /* CLIP*/ \
                        { \
                            temp_accum = \
                                min \
                                ( \
                                  max \
                                  ( \
                                    temp_accum, \
                                    *( ( ACCUM_type* ) \
                                       ( post_op->eltwise + ele_i )->algo.alpha ) \
                                  ), \
                                  *( ( ACCUM_type* ) \
                                     ( post_op->eltwise + ele_i )->algo.beta) \
                                ); \
                            ele_i += 1; \
                        } \
                        else \
                        {} \
                    } \
                    else if ( post_op->seq_vector[op_id] == SCALE ) \
                    { \
                        temp_accum = GEN_FUNC_NAME(mat_mul_accuracy_check_downscale_,BLAS_DOWNSCALE_SFX) \
                            (temp_accum, post_op, j); \
                    } \
                    else \
                    {} \
                } \
            } \
            /* Need to convert to downscaled type if required.*/ \
            mat_mul_get_output_type_val ## ACCUM_type ## C_type \
            ( \
              &out_temp_accum, &temp_accum \
            ); \
 \
            if ( *( c + ( rs_c * i ) + ( cs_c * j ) ) != out_temp_accum ) \
            { \
                float comp_float, ref_float; \
                GEN_FUNC_NAME(C_type,_to_float)(*( c + ( rs_c * i ) + ( cs_c * j ) ), &comp_float); \
                GEN_FUNC_NAME(C_type,_to_float)(out_temp_accum, &ref_float); \
                if ( fout ) \
                { \
                    fprintf( fout, "%s Failure input m: %ld, n: %ld, k: %ld," \
                                    " lda: %ld, ldb: %ld, ldc: %ld, computed:%f, ref:%f, diff:%f\n", \
                                    XSTR(BLAS_SFX), m, n, k, lda, ldb, ldc, comp_float, \
                                    ref_float, comp_float - ref_float); \
                    fflush( fout ); \
                } \
                    printf("failure, m: %ld, n: %ld, k: %ld, computed:%f, ref:%f, diff:%f\n", i, j, k, \
                            comp_float, ref_float, comp_float-ref_float); \
                goto cleanup_acc; \
            } \
        } \
    } \
cleanup_acc: \
    return; \
} \

GEN_MAT_MUL_ACC_CHK_DRV_FUNC(uint8_t,int8_t,int16_t,int16_t,float,u8s8s16os16,u8s8s16os8)
GEN_MAT_MUL_ACC_CHK_DRV_FUNC(uint8_t,int8_t,int8_t,int16_t,float,u8s8s16os8,u8s8s16os8)
GEN_MAT_MUL_ACC_CHK_DRV_FUNC(uint8_t,int8_t,uint8_t,int16_t,float,u8s8s16ou8,u8s8s16ou8)
GEN_MAT_MUL_ACC_CHK_DRV_FUNC(uint8_t,int8_t,int32_t,int32_t,float,u8s8s32os32,u8s8s32os8)
GEN_MAT_MUL_ACC_CHK_DRV_FUNC(uint8_t,int8_t,int8_t,int32_t,float,u8s8s32os8,u8s8s32os8)
GEN_MAT_MUL_ACC_CHK_DRV_FUNC(bfloat16,bfloat16,float,float,float,bf16bf16f32of32,bf16bf16f32obf16)
GEN_MAT_MUL_ACC_CHK_DRV_FUNC(bfloat16,bfloat16,bfloat16,float,float,bf16bf16f32obf16,bf16bf16f32obf16)
GEN_MAT_MUL_ACC_CHK_DRV_FUNC(float,float,float,float,float,f32f32f32of32,bf16bf16f32obf16)
GEN_MAT_MUL_ACC_CHK_DRV_FUNC(int8_t,int8_t,int32_t,int32_t,float,s8s8s32os32,s8s8s32os8)
GEN_MAT_MUL_ACC_CHK_DRV_FUNC(int8_t,int8_t,int8_t,int32_t,float,s8s8s32os8,s8s8s32os8)
GEN_MAT_MUL_ACC_CHK_DRV_FUNC(int8_t,int8_t,int16_t,int16_t,float,s8s8s16os16,s8s8s16os8)
GEN_MAT_MUL_ACC_CHK_DRV_FUNC(int8_t,int8_t,int8_t,int16_t,float,s8s8s16os8,s8s8s16os8)

#define GEN_MAT_MUL_POST_OPS_CREATOR(C_DSCALE_type,C_type,DSCALE_type,BLAS_SFX) \
aocl_post_op* lpgemm_create_post_ops_struct_ ## BLAS_SFX \
     ( \
       dim_t m, \
       dim_t n, \
       char* post_ops_str \
     ) \
{ \
    aocl_post_op* post_ops = NULL; \
    post_ops = ( aocl_post_op* ) malloc( sizeof( aocl_post_op ) ); \
 \
    if ( ( post_ops == NULL ) && ( global_dscale_out == 'n' ) ) \
    { \
        return NULL; \
    } \
 \
    /* Only supporting 5 post ops at max for now.*/ \
    dim_t max_post_ops_seq_length = 5; \
    post_ops->seq_vector = ( AOCL_POST_OP_TYPE* ) \
                            malloc \
                            ( \
                              max_post_ops_seq_length * \
                              sizeof( AOCL_POST_OP_TYPE ) \
                            ); \
 \
    if ( post_ops->seq_vector == NULL ) \
    { \
        free( post_ops ); \
        return NULL; \
    } \
 \
    /* Parse post ops list.*/ \
    dim_t cur_op_index = 0; \
    /* Ensure the buffers that use NULL check in deinit code is properly set to NULL.*/ \
    post_ops->eltwise = NULL; \
    post_ops->bias.bias = NULL; \
    post_ops->sum.scale_factor = NULL; \
    post_ops->sum.buff = NULL; \
    post_ops->sum.zero_point = NULL; \
    if ( post_ops_str != NULL ) \
    { \
        char* ops_tok = strtok(post_ops_str, ", " ); \
        bool is_relu = FALSE; \
        bool is_param_relu = FALSE; \
        bool is_gelu_tanh = FALSE; \
        bool is_gelu_erf = FALSE; \
        bool is_clip = FALSE; \
        dim_t activator_idx = 0; \
        dim_t clip_idx = 0; \
 \
        /* Ensure only one activator is used as an eltwise post-op.*/ \
        bool is_activator_set = FALSE; \
        num_eltwise = 0; \
        while ( ops_tok ) \
        { \
            if ( strcmp( ops_tok, "bias" ) == 0 ) \
            { \
                post_ops->seq_vector[cur_op_index] = BIAS; \
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
            ops_tok = strtok( NULL, ", " ); \
        } \
 \
        /* Allocate bias buffer, return early if alloc fails.*/ \
        post_ops->bias.bias = malloc( n * sizeof( C_type ) ); \
        if ( post_ops->bias.bias == NULL ) \
        { \
            free( post_ops->seq_vector ); \
            free( post_ops ); \
            return NULL; \
        } \
        GEN_FUNC_NAME(fill_array_post_ops_,C_type)( post_ops->bias.bias, n ); \
 \
        post_ops->eltwise = malloc( num_eltwise * sizeof( aocl_post_op_eltwise ) ); \
        if ( post_ops->eltwise == NULL ) \
        { \
            free( post_ops->bias.bias ); \
            free( post_ops->seq_vector ); \
            free( post_ops ); \
            return NULL; \
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
        } \
        /* Only one of relu,prelu,gelu_tanh,gelu_erf allowed as an activator.*/ \
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
            ( post_ops->eltwise + activator_idx )->algo.beta = NULL; \
            ( post_ops->eltwise + activator_idx )->algo.alpha = malloc( sizeof( C_type ) ); \
            *( ( C_type* ) ( post_ops->eltwise + activator_idx )->algo.alpha ) = ( C_type )6; \
            ( post_ops->eltwise + activator_idx )->algo.algo_type = PRELU; \
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
            ( post_ops->eltwise + clip_idx )->algo.alpha = malloc( sizeof( C_type ) ); \
            ( post_ops->eltwise + clip_idx )->algo.beta = malloc( sizeof( C_type ) ); \
            *( ( C_type* ) ( post_ops->eltwise + clip_idx )->algo.alpha ) = ( C_type ) ( -64 ); \
            *( ( C_type* ) ( post_ops->eltwise + clip_idx )->algo.beta ) = ( C_type ) ( 23 ); \
            ( post_ops->eltwise + clip_idx )->algo.algo_type = CLIP; \
        } \
    } \
 \
    if ( global_dscale_out == 'y' ) \
    { \
        post_ops->seq_vector[cur_op_index] = SCALE; \
        cur_op_index++; \
 \
        post_ops->sum.is_power_of_2 = FALSE; \
        if ( global_dscale_out == 'y' ) \
        { \
            /* Allocate scale buffer, return early if alloc fails.*/ \
            post_ops->sum.scale_factor = malloc( n * sizeof( DSCALE_type ) ); \
            post_ops->sum.zero_point = malloc( n * sizeof( C_DSCALE_type ) ); \
            if ( ( post_ops->sum.scale_factor == NULL ) || \
                 ( post_ops->sum.zero_point == NULL ) ) \
            { \
                free ( post_ops->eltwise ); \
                free ( post_ops->bias.bias ); \
                free( post_ops->seq_vector ); \
                if ( post_ops->sum.zero_point != NULL ) \
                { \
                    free( post_ops->sum.zero_point ); \
                } \
                if ( post_ops->sum.scale_factor != NULL ) \
                { \
                    free( post_ops->sum.scale_factor ); \
                } \
                free( post_ops ); \
                return NULL; \
            } \
            /* Fill scale factor and zero points.*/ \
            DSCALE_type* temp_dscale_ptr = ( DSCALE_type* )post_ops->sum.scale_factor; \
            C_DSCALE_type* temp_dzero_point_ptr = ( C_DSCALE_type* )post_ops->sum.zero_point; \
            for ( dim_t i = 0; i < n; ++i ) \
            { \
                temp_dscale_ptr[i] = ( ( DSCALE_type )1 )/ ( ( DSCALE_type )1000 ); \
                temp_dzero_point_ptr[i] = (C_DSCALE_type)( i % 126 ); \
            } \
        } \
    } \
 \
    post_ops->seq_length = cur_op_index; \
 \
    return post_ops; \
} \

GEN_MAT_MUL_POST_OPS_CREATOR(int8_t,int16_t,float,u8s8s16os16)
GEN_MAT_MUL_POST_OPS_CREATOR(int8_t,int32_t,float,u8s8s32os32)
GEN_MAT_MUL_POST_OPS_CREATOR(bfloat16,float,float,bf16bf16f32of32)
GEN_MAT_MUL_POST_OPS_CREATOR(bfloat16,float,float,f32f32f32of32)
GEN_MAT_MUL_POST_OPS_CREATOR(int8_t,int32_t,float,s8s8s32os32)
GEN_MAT_MUL_POST_OPS_CREATOR(int8_t,int16_t,float,s8s8s16os16)

void lpgemm_destroy_post_ops_struct( aocl_post_op* post_ops )
{
    if ( post_ops == NULL )
    {
        return;
    }

    if ( post_ops->eltwise != NULL )
    {
        for ( dim_t i = 0; i < num_eltwise; ++i )
        {
            if ( ( post_ops->eltwise + i )->algo.alpha != NULL )
            {
                free( ( post_ops->eltwise + i )->algo.alpha );
            }
            if ( ( post_ops->eltwise + i )->algo.beta != NULL )
            {
                free( ( post_ops->eltwise + i )->algo.beta );
            }
        }
        free( post_ops->eltwise );
    }
    if ( post_ops->sum.scale_factor != NULL )
    {
        free( post_ops->sum.scale_factor );
    }
    if ( post_ops->sum.zero_point != NULL )
    {
        free( post_ops->sum.zero_point );
    }
    if ( post_ops->bias.bias != NULL )
    {
        free( post_ops->bias.bias );
    }
    if( post_ops->seq_vector != NULL )
    {
        free( post_ops->seq_vector );
    }

    free( post_ops );
}

#define GEN_MAT_MUL_BENCH_MAIN_FUNC(A_type, B_type, C_type, Sum_type, BLAS_SFX, REORDER_SFX) \
void mat_mul_bench_main_ ## BLAS_SFX \
     ( \
       FILE*   fin, \
       FILE*   fout, \
       char    stor_order, \
       char    transa, \
       char    transb, \
       char    op_a, \
       char    op_b, \
       int32_t m, \
       int32_t n, \
       int32_t k, \
       int32_t stride_a, \
       int32_t stride_b, \
       int32_t stride_c, \
       char*   post_ops_str \
     ) \
{ \
    int32_t n_repeats = bli_max( 30, bli_min(( 3e10 / ( ( int64_t )m * n * k )), 1000 )); \
    if ( global_n_repeat > 0 ) \
    { \
        n_repeats = global_n_repeat; \
    } \
 \
    int32_t size_A = 0; \
    int32_t size_B = 0; \
    int32_t size_C = 0; \
    if( ( stor_order == 'r' ) || ( stor_order == 'R' ) ) \
    { \
        size_A = ( ( transa == 'n' ) || ( transa == 'N' ) ) ? m * stride_a : k * stride_a; \
        size_B = ( ( transb == 'n' ) || ( transb == 'N' ) ) ? k * stride_b : n * stride_b; \
        size_C = m * stride_c; \
    } \
    else \
    { \
        size_A = ( ( transa == 'n' ) || ( transa == 'N' ) ) ? k * stride_a : m * stride_a; \
        size_B = ( ( transb == 'n' ) || ( transb == 'N' ) ) ? n * stride_b : k * stride_b; \
        size_C = n * stride_c; \
    } \
    A_type* a = ( A_type* ) lpgemm_malloc( sizeof( A_type ) * size_A ); \
    GEN_FUNC_NAME(fill_array_,A_type)(a, size_A ); \
 \
    B_type* b = ( B_type* ) lpgemm_malloc( sizeof( B_type ) * size_B ); \
    GEN_FUNC_NAME(fill_array_,B_type)(b, size_B ); \
 \
    C_type* c = ( C_type* ) lpgemm_malloc( sizeof( C_type ) * size_C ); \
 \
    C_type* c_ref = ( C_type* ) lpgemm_malloc( sizeof( C_type ) * size_C ); \
 \
    if ( bench_mode == 'a' ) \
    { \
        GEN_FUNC_NAME(fill_array_,C_type)( c, ( size_C ) ); \
        GEN_FUNC_NAME(fill_array_,C_type)( c_ref, ( size_C ) ); \
    } \
    else \
    { \
        memset( ( void* ) c, 0, sizeof( C_type ) * size_C ); \
        memset( ( void* ) c_ref, 0, sizeof( C_type ) * size_C ); \
    } \
 \
    Sum_type alpha = 0; \
    Sum_type beta = 0; \
    if ( bench_mode == 'p' ) \
    { \
        alpha = 2; \
        beta = 9; \
    } \
    else if ( bench_mode == 'a' ) \
    { \
        n_repeats = 1; \
        alpha = 2; \
        beta = 9; \
    } 	\
 \
    aocl_post_op* post_op = NULL; \
    if ( ( post_ops_str != NULL ) || ( global_dscale_out == 'y' ) ) \
    { \
        post_op = GEN_FUNC_NAME(lpgemm_create_post_ops_struct_,REORDER_SFX)( m, n, post_ops_str ); \
        if ( post_op == NULL ) \
        { \
            printf(" post op struct allocation failure, returning.\n"); \
            return; \
        } \
    } \
 \
    if ( ( op_b == 'p' ) || ( op_b == 'P' ) || ( op_b == 'n' ) || ( op_b == 'N' ) )  \
    { \
        /* No reordering of B.*/ \
        GEN_FUNC_NAME(mat_mul_bench_driver_,BLAS_SFX) \
        ( \
          stor_order, transa, transb, op_a, op_b, n_repeats, m, n, k, \
          alpha, \
          a, stride_a, \
          b, stride_b, \
          beta, \
          c, stride_c, \
          post_op \
        ); \
    } \
    else if ( ( op_b == 'r' ) || ( op_b == 'R' ) ) \
    { \
        /* Reorder B.*/ \
        siz_t b_reorder_buf_siz_req = \
            GEN_FUNC_NAME(aocl_get_reorder_buf_size_,REORDER_SFX)( stor_order, transb, 'B', k, n ); \
 \
        B_type* b_reorder = ( B_type* ) lpgemm_malloc( b_reorder_buf_siz_req ); \
        GEN_FUNC_NAME(aocl_reorder_,REORDER_SFX)( stor_order, transb, 'B', b, b_reorder, k, n, stride_b ); \
 \
         GEN_FUNC_NAME(mat_mul_bench_driver_,BLAS_SFX) \
        ( \
          stor_order, transa, transb, op_a, op_b, n_repeats, m, n, k, \
          alpha, \
          a, stride_a, \
          b_reorder, stride_b, \
          beta, \
          c, stride_c, \
          post_op \
        ); \
    } \
 \
    if ( bench_mode == 'a' ) \
    { \
        printf(" Running accuracy check.\n"); \
        GEN_FUNC_NAME(mat_mul_accuracy_check_driver_,BLAS_SFX) \
        ( \
          fout, stor_order, transa, transb, m, n, k, \
          alpha, \
          a, stride_a, \
          b, stride_b, \
          beta, \
          c, stride_c, \
          c_ref, stride_c, \
          post_op \
        ); \
    } \
 \
    lpgemm_destroy_post_ops_struct( post_op ); \
 \
    lpgemm_free( a ); \
    lpgemm_free( b ); \
    lpgemm_free( c ); \
    lpgemm_free( c_ref ); \
} \

GEN_MAT_MUL_BENCH_MAIN_FUNC(bfloat16,bfloat16,float,float,bf16bf16f32of32,bf16bf16f32of32)
GEN_MAT_MUL_BENCH_MAIN_FUNC(bfloat16,bfloat16,bfloat16,float,bf16bf16f32obf16,bf16bf16f32of32)
GEN_MAT_MUL_BENCH_MAIN_FUNC(uint8_t,int8_t,int16_t,int16_t,u8s8s16os16,u8s8s16os16)
GEN_MAT_MUL_BENCH_MAIN_FUNC(uint8_t,int8_t,int8_t,int16_t,u8s8s16os8,u8s8s16os16)
GEN_MAT_MUL_BENCH_MAIN_FUNC(uint8_t,int8_t,uint8_t,int16_t,u8s8s16ou8,u8s8s16os16)
GEN_MAT_MUL_BENCH_MAIN_FUNC(uint8_t,int8_t,int32_t,int32_t,u8s8s32os32,u8s8s32os32)
GEN_MAT_MUL_BENCH_MAIN_FUNC(uint8_t,int8_t,int8_t,int32_t,u8s8s32os8,u8s8s32os32)
GEN_MAT_MUL_BENCH_MAIN_FUNC(float,float,float,float,f32f32f32of32,f32f32f32of32)
GEN_MAT_MUL_BENCH_MAIN_FUNC(int8_t,int8_t,int32_t,int32_t,s8s8s32os32,s8s8s32os32)
GEN_MAT_MUL_BENCH_MAIN_FUNC(int8_t,int8_t,int8_t,int32_t,s8s8s32os8,s8s8s32os32)
GEN_MAT_MUL_BENCH_MAIN_FUNC(int8_t,int8_t,int16_t,int16_t,s8s8s16os16,s8s8s16os16)
GEN_MAT_MUL_BENCH_MAIN_FUNC(int8_t,int8_t,int8_t,int16_t,s8s8s16os8,s8s8s16os16)
int main( int argc, char** argv )
{
    FILE* fin  = NULL;
    if ( argc < 5 )
    {
        printf
        (
          "Usage: ./bench_lpgemm -i input.txt -m mode < -n 100 -o op1,op2 >\n" \
          "--Mode is either a or p.\n" \
          "\ta is used for accuracy testing.\n" \
          "\tp is used for performance benchmarking.\n" \
          "--n_repeats can be set optionally using -n arg.\n" \
          "--Post ops can be executed optionaly by providing a coma separated\n" \
          "  list of post-ops after -o arg. Following post-ops are supported:\n" \
          "    1. bias\n" \
          "    2. 4 activators\n" \
          "      a. relu\n" \
          "      b. prelu\n" \
          "      c. gelu_tanh\n" \
          "      d. gelu_erf\n" \
          "    3.clip\n" \
          "  Atleast one post-op needs to be specified if the -o arg is used.\n" \
          "  eg: -o gelu_tanh; -o bias,relu ; -o clip,prelu,bias.\n" \
          "  It is to be noted only one activator can be used at a time.\n" \
          "  If more than one activator is used, only the first activator is\n" \
          "  applied and the other activators are ignored.\n" \
          "--Downscaled version of an API is enabled by using -d arg followed\n" \
          "  by the datatype that needs to be downscaled to"
          "  Downscaled api's are used to enable quantization workflows.\n" \
          "  Following downscaled api's are supported:\n" \
          "    1. u8s8s32os32 -d s8 = u8s8s32os8.\n" \
          "    2. u8s8s16os16 -d s8 = u8s8s16os8.\n" \
          "    3. u8s8s16os16 -d u8 = u8s8s16ou8.\n" \
          "    4. bf16bf16f32obf32 -d bf16 = bf16bf16f32obf16.\n" \
          "    5. s8s8s32os32 -d s8 = s8s8s32os8.\n" \
          "    6. s8s8s16os16 -d s8 = s8s8s16os8.\n" \
          "  Example: ./bench_lpgemm -m a -n 2 -o bias,relu -d bf16 -i input.txt\n" \
        );
        exit( 1 );
    }

    char* file_name = NULL;
    char post_ops_str[50];
    char* post_ops_str_dest = NULL; //Strtok is used to parse, need to maintain a copy.
    char dscale_type_str[10];

    // Parse CLI arguments.
    opterr = 0;
    int opt_val;
    while ( ( opt_val = getopt( argc, argv, "i:m:n:" ) ) != -1 )
    {
        switch ( opt_val )
        {
            case 'i':
                    file_name = optarg;
                    break;
            case 'm':
                    bench_mode = ( ( ( *optarg ) == 'a' ) || ( ( *optarg ) == 'p' ) ) ? ( *optarg ) : 'p';
                    break;
            case 'n':
                    global_n_repeat = ( atoi( optarg ) > 0 ) ? atoi( optarg ) : 0;
                    break;
            default:
                    break;
        }
    }

    if ( bench_mode == 'p' )
    {
        printf( "Running bench in performance benchmarking mode.\n" );
    }
    else if ( bench_mode == 'a' )
    {
        printf( "Running bench in accuracy/correctness testing mode.\n" );
    }

    if ( file_name == NULL )
    {
        printf( " File name provided is invalid.\n" );
        exit( 1 );
    }

    fin = fopen( file_name, "r" );
    if (fin == NULL)
    {
        printf( "Error opening the file %s\n", argv[1] );
        exit( 1 );
    }

    FILE* fout = NULL;

    fout = fopen( "lpgemm_accuracy_test_failures.txt", "w" );

    char op_type_char;
    char op_a, op_b;
    char stor_order;
    char transa, transb;
    int32_t m, n, k;
    int32_t stride_a, stride_b, stride_c;

    const dim_t len_list_omp_cores_for_testing = 2;
    const dim_t list_omp_cores_for_testing[2] = { 80, 1 };

    dim_t core_index = 0;
    bool can_run = TRUE;
    while ( ( can_run == TRUE ) && ( fseek( fin, 0L, SEEK_SET ) == 0 ) )
    {
        if ( bench_mode == 'p' )
        {
            can_run = FALSE;
        }
        else if ( bench_mode == 'a' )
        {
            // For accuracy testing, we test accuracy using multiple different
            // number of cores. This helps uncover any bugs related to over
            // subscription or varying thread factorizations.
            // Set current number of cores.
#ifdef BLIS_ENABLE_OPENMP
            omp_set_num_threads( list_omp_cores_for_testing[core_index] );
#endif
            printf( "Accuracy test using %ld threads.\n",
                            list_omp_cores_for_testing[core_index] );

            core_index++;
            if ( core_index < len_list_omp_cores_for_testing )
            {
                can_run = TRUE;
            }
            else
            {
                can_run = FALSE;
            }
        }

        // Input format: data_type stor_type pack/reorder m n k lda ldb ldc
        while ( fscanf( fin, "%c %s %c %c %c %c %c %d %d %d %d %d %d %s\n",
                &op_type_char, dscale_type_str, &stor_order, &transa, &transb, &op_a, &op_b, &m, &n, &k,
                &stride_a, &stride_b, &stride_c, post_ops_str ) == 14 )
        {
            stor_order = ( ( stor_order == 'r' ) || ( stor_order == 'R' ) ||
                            ( stor_order == 'c' ) || ( stor_order == 'C' ) ) ?
                            stor_order : 'r';

            if ( strcmp( post_ops_str, "none" ) != 0 )
            {
                post_ops_str_dest = ( char* )malloc \
                        ( ( strlen( post_ops_str) + 1 )* sizeof( char ) );
                strcpy( post_ops_str_dest, post_ops_str );
            }

            if ( ( op_type_char == 'i' ) || ( op_type_char == 'I' ) )
            {
                if ( ( strcmp( dscale_type_str, "S32" ) == 0 ) ||
                     ( strcmp( dscale_type_str, "s32" ) == 0 ) )
                {
                    global_dscale_out = 'n';
                    GEN_FUNC_NAME(mat_mul_bench_main_,u8s8s32os32)
                    (
                      fin, fout, stor_order, transa, transb, op_a, op_b,
                      m, n, k, stride_a, stride_b, stride_c,
                      post_ops_str_dest
                    );
                }
                else
                {
                    if ( ( strcmp( dscale_type_str, "S8" ) == 0 ) ||
                         ( strcmp( dscale_type_str, "s8" ) == 0 ) )
                    {
                        global_dscale_out = 'y';
                        DSCALE_CLIP_MIN = -128;
                        DSCALE_CLIP_MAX = +127;
                        GEN_FUNC_NAME(mat_mul_bench_main_,u8s8s32os8)
                        (
                          fin, fout, stor_order, transa, transb, op_a, op_b,
                          m, n, k, stride_a, stride_b, stride_c,
                          post_ops_str_dest
                        );
                    }
                    else
                    {
                        printf("Downscale type not supported.\n");
                    }
                }
            }
            else if ( ( op_type_char == 'f' ) || ( op_type_char == 'F' ) )
            {
                global_dscale_out = 'n';
                GEN_FUNC_NAME(mat_mul_bench_main_,f32f32f32of32)
                (
                  fin, fout, stor_order, transa, transb, op_a, op_b,
                  m, n, k, stride_a, stride_b, stride_c,
                  post_ops_str_dest
                );
            }
            else if ((op_type_char == 's') || (op_type_char == 'S'))
            {
                if ( ( strcmp( dscale_type_str, "S16" ) == 0 ) ||
                     ( strcmp( dscale_type_str, "s16" ) == 0 ) )
                {
                    global_dscale_out = 'n';
                    GEN_FUNC_NAME(mat_mul_bench_main_,u8s8s16os16)
                    (
                        fin, fout, stor_order, transa, transb, op_a, op_b,
                        m, n, k, stride_a, stride_b, stride_c,
                        post_ops_str_dest
                    );
                }
                else
                {
                    if ( ( strcmp( dscale_type_str, "S8" ) == 0 ) ||
                         ( strcmp( dscale_type_str, "s8" ) == 0 ) )
                    {
                        global_dscale_out = 'y';
                        DSCALE_CLIP_MIN = -128;
                        DSCALE_CLIP_MAX = +127;
                        GEN_FUNC_NAME(mat_mul_bench_main_,u8s8s16os8)
                        (
                            fin, fout, stor_order, transa, transb, op_a, op_b,
                            m, n, k, stride_a, stride_b, stride_c,
                            post_ops_str_dest
                        );
                    }
                    else if ( ( strcmp( dscale_type_str, "U8" ) == 0 ) ||
                              ( strcmp( dscale_type_str, "u8" ) == 0 ) )
                    {
                        global_dscale_out = 'y';
                        DSCALE_CLIP_MIN = 0;
                        DSCALE_CLIP_MAX = +255;
                        GEN_FUNC_NAME(mat_mul_bench_main_,u8s8s16ou8)
                        (
                            fin, fout, stor_order, transa, transb, op_a, op_b,
                            m, n, k, stride_a, stride_b, stride_c,
                            post_ops_str_dest
                        );
                    }
                    else
                    {
                        printf("Downscale type not supported.\n");
                    }
                }
            }
            else if ((op_type_char == 'b') || (op_type_char == 'B'))
            {
                if ( ( strcmp( dscale_type_str, "F32" ) == 0 ) ||
                     ( strcmp( dscale_type_str, "f32" ) == 0 ) )
                {
                    global_dscale_out = 'n';
                    GEN_FUNC_NAME(mat_mul_bench_main_, bf16bf16f32of32)
                    (
                        fin, fout, stor_order, transa, transb, op_a, op_b,
                        m, n, k, stride_a, stride_b, stride_c,
                        post_ops_str_dest
                    );
                }
                else if ( ( strcmp( dscale_type_str, "BF16" ) == 0 ) ||
                          ( strcmp( dscale_type_str, "bf16" ) == 0 ) )
                {
                    global_dscale_out = 'y';
                    GEN_FUNC_NAME(mat_mul_bench_main_, bf16bf16f32obf16)
                    (
                        fin, fout, stor_order, transa, transb, op_a, op_b,
                        m, n, k, stride_a, stride_b, stride_c,
                        post_ops_str_dest
                    );
                }
                else
                {
                    printf("Downscale type not supported.\n");
                }
            }
            else if ( ( op_type_char == 'u' ) || ( op_type_char == 'U' ) )
            {
                if ( ( strcmp( dscale_type_str, "S32" ) == 0 ) ||
                     ( strcmp( dscale_type_str, "s32" ) == 0 ) )
                {
                    global_dscale_out = 'n';
                    GEN_FUNC_NAME(mat_mul_bench_main_,s8s8s32os32)
                    (
                      fin, fout, stor_order, transa, transb, op_a, op_b,
                      m, n, k, stride_a, stride_b, stride_c,
                      post_ops_str_dest
                    );
                }
                else
                {
                    if ( ( strcmp( dscale_type_str, "S8" ) == 0 ) ||
                         ( strcmp( dscale_type_str, "s8" ) == 0 ) )
                    {
                        global_dscale_out = 'y';
                        DSCALE_CLIP_MIN = -128;
                        DSCALE_CLIP_MAX = +127;
                        GEN_FUNC_NAME(mat_mul_bench_main_,s8s8s32os8)
                        (
                          fin, fout, stor_order, transa, transb, op_a, op_b,
                          m, n, k, stride_a, stride_b, stride_c,
                          post_ops_str_dest
                        );
                    }
                    else
                    {
                        printf("Downscale type not supported.\n");
                    }
                }
            }
            else if ( ( op_type_char == 'v' ) || ( op_type_char == 'V' ) )
            {
                if ( ( strcmp( dscale_type_str, "S16" ) == 0 ) ||
                     ( strcmp( dscale_type_str, "s16" ) == 0 ) )
                {
                    global_dscale_out = 'n';
                    GEN_FUNC_NAME(mat_mul_bench_main_,s8s8s16os16)
                    (
                      fin, fout, stor_order, transa, transb, op_a, op_b,
                      m, n, k, stride_a, stride_b, stride_c,
                      post_ops_str_dest
                    );
                }
                else
                {
                    if ( ( strcmp( dscale_type_str, "S8" ) == 0 ) ||
                         ( strcmp( dscale_type_str, "s8" ) == 0 ) )
                    {
                        global_dscale_out = 'y';
                        DSCALE_CLIP_MIN = -128;
                        DSCALE_CLIP_MAX = +127;
                        GEN_FUNC_NAME(mat_mul_bench_main_,s8s8s16os8)
                        (
                          fin, fout, stor_order, transa, transb, op_a, op_b,
                          m, n, k, stride_a, stride_b, stride_c,
                          post_ops_str_dest
                        );
                    }
                    else
                    {
                        printf("Downscale type not supported.\n");
                    }
                }
            }
            if ( strcmp( post_ops_str, "none" ) != 0 )
            {
                strcpy( post_ops_str_dest, post_ops_str );
            }
        }
    }

    if ( post_ops_str_dest != NULL )
    {
        free( post_ops_str_dest );
    }
    if ( fin )
    {
        fclose( fin );
    }
    if ( fout )
    {
        fclose( fout );
    }
    return 0;
}
