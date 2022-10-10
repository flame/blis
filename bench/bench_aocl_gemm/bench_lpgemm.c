#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <immintrin.h>
#include <time.h>
#include <float.h>
#include <unistd.h>

#include "blis.h"

#define S8_MIN  (-128)
#define S8_MAX  (+127)

// Mode can be one of the follwoing:
// 	1. p - performance, used for benchmarks.
// 	2. a - accuracy, used to test accuracy/correctness.
// Default value is p, can be modified by passing command line arg.
char bench_mode = 'p';

int32_t global_n_repeat = 0;

char global_dscale_out = 'n';

#define _XSTR(str) #str
#define XSTR(str) _XSTR(str)

#define GEN_FUNC_NAME(prototype,ctype) prototype ## ctype

#define GEN_FILL_ARRAY_FUNC(ctype) \
void fill_array_ ## ctype ( void* arr, dim_t size ) \
{ \
	ctype* temp_arr = ( ctype* ) arr; \
	for ( dim_t i = 0; i < size; ++i ) \
	{ \
		temp_arr[i] = ( ctype )( i % 10 ); \
	} \
} \

GEN_FILL_ARRAY_FUNC(uint8_t)
GEN_FILL_ARRAY_FUNC(int8_t)
GEN_FILL_ARRAY_FUNC(float)
GEN_FILL_ARRAY_FUNC(int32_t)

inline void float_to_bf16( float* float_value, bfloat16* bf16_val )
{
	/*Set offset 2 to copy most significant 2 bytes of float 
	to convert float values to bf16 values*/
	memcpy( ( bf16_val ), (char *)( float_value ) + 2, sizeof ( bfloat16 ) );
}

inline void convert_float_arr_to_bf16( float* array, bfloat16* array_bf16, int size )
{
	for (int i=0; i< size; i++)
	{
		float_to_bf16( ( array + i ), ( array_bf16 + i ) );
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
       char    op_t, \
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
	char storage = stor_order; \
	char transa = 'n'; \
	char transb = 'n'; \
	char reordera = 'n'; \
	char reorderb = 'n'; \
 \
 	if ( ( op_t == 'p' ) || ( op_t == 'P' ) ) \
	{ \
		/* No reordering of B.*/ \
		reordera = 'n'; \
		reorderb = 'n'; \
	} \
	else if ( ( op_t == 'r' ) || ( op_t == 'R' ) ) \
	{ \
		/* Reordered B.*/ \
		reordera = 'n'; \
		reorderb = 'r'; \
	} \
 \
	aocl_gemm_ ## BLAS_SFX( storage, transa, transb, m, n, k, \
					alpha, \
					a, lda, reordera, \
					b, ldb, reorderb, \
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
GEN_BLIS_MAT_MUL_FUNC(uint8_t,int8_t,int32_t,int32_t,u8s8s32os32)
GEN_BLIS_MAT_MUL_FUNC(uint8_t,int8_t,int8_t,int32_t,u8s8s32os8)
GEN_BLIS_MAT_MUL_FUNC(bfloat16,bfloat16,float,float,bf16bf16f32of32)
GEN_BLIS_MAT_MUL_FUNC(bfloat16,bfloat16,bfloat16,float,bf16bf16f32obf16)
GEN_BLIS_MAT_MUL_FUNC(float,float,float,float,f32f32f32of32)

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
	printf("%s m: %ld, n: %ld, k: %ld, lda: %ld, ldb: %ld, ldc: %ld," \
					" Gops: %f, n_repeats: %d\n", 
			msg, m, n, k, lda, ldb, ldc, gflops, n_repeats);
}

#define GEN_MAT_MUL_BENCH_DRV_FUNC(A_type,B_type,C_type,ACCUM_type,BLAS_SFX) \
void mat_mul_bench_driver_ ## BLAS_SFX \
     ( \
       char    stor_order, \
       char    op_t, \
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
			memset( ( void* ) c, 0, sizeof( C_type ) * m * n ); \
		} \
 \
		struct timespec tstart={0,0}, tend={0,0}; \
		clock_gettime(CLOCK_MONOTONIC, &tstart); \
 \
		GEN_FUNC_NAME(mat_mul_,BLAS_SFX) \
		( \
		  stor_order, op_t, m, n, k, \
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
	print_result( XSTR(BLAS_SFX), n_repeats, m, n, k, lda, ldb, ldc, min_time_diff); \
} \

GEN_MAT_MUL_BENCH_DRV_FUNC(uint8_t,int8_t,int16_t,int16_t,u8s8s16os16)
GEN_MAT_MUL_BENCH_DRV_FUNC(uint8_t,int8_t,int8_t,int16_t,u8s8s16os8)
GEN_MAT_MUL_BENCH_DRV_FUNC(uint8_t,int8_t,int32_t,int32_t,u8s8s32os32)
GEN_MAT_MUL_BENCH_DRV_FUNC(uint8_t,int8_t,int8_t,int32_t,u8s8s32os8)
GEN_MAT_MUL_BENCH_DRV_FUNC(bfloat16,bfloat16,float,float,bf16bf16f32of32)
GEN_MAT_MUL_BENCH_DRV_FUNC(bfloat16,bfloat16,bfloat16,float,bf16bf16f32obf16)
GEN_MAT_MUL_BENCH_DRV_FUNC(float,float,float,float,f32f32f32of32)

int max (int a, int b)
{
	return ( a > b ? a : b );
}

int min (int a, int b)
{
	return ( a < b ? a : b );
}

#define GEN_MAT_MUL_ACC_CHK_DOWNSCALE(C_type,ACCUM_type,SCALE_type,BLAS_DOWNSCALE_SFX) \
inline C_type mat_mul_accuracy_check_downscale_ ## BLAS_DOWNSCALE_SFX \
     (\
       ACCUM_type temp_accum,\
       C_type out_temp_accum, \
       aocl_post_op*  post_op, \
       dim_t j \
     )\
{\
	out_temp_accum = ( C_type ) min ( max ( nearbyintf( ( SCALE_type )temp_accum * \
		( *( ( SCALE_type* )post_op->sum.scale_factor + j ) ) ), S8_MIN ), S8_MAX ) ; \
	return 	out_temp_accum; \
}\

GEN_MAT_MUL_ACC_CHK_DOWNSCALE(int8_t,int16_t,float,u8s8s16os8)
GEN_MAT_MUL_ACC_CHK_DOWNSCALE(int8_t,int32_t,float,u8s8s32os8)

inline bfloat16 mat_mul_accuracy_check_downscale_bf16bf16f32obf16
     (
       float temp_accum, 
       bfloat16 out_temp_accum, 
       aocl_post_op*  post_op, 
       dim_t j
     )
{
	float_to_bf16( ( &temp_accum ), ( &out_temp_accum ) );
	return out_temp_accum;
}

#define GEN_MAT_MUL_ACC_CHK_ACCUM(A_type, B_type, C_type,ACCUM_type,BLAS_SFX) \
inline ACCUM_type mat_mul_accuracy_check_accum_ ## BLAS_SFX \
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
GEN_MAT_MUL_ACC_CHK_ACCUM(uint8_t,int8_t,int16_t,int16_t,u8s8s16os16)
GEN_MAT_MUL_ACC_CHK_ACCUM(uint8_t,int8_t,int8_t,int32_t,u8s8s32os8)
GEN_MAT_MUL_ACC_CHK_ACCUM(uint8_t,int8_t,int32_t,int32_t,u8s8s32os32)
GEN_MAT_MUL_ACC_CHK_ACCUM(float,float,float,float,f32f32f32of32)

inline float bf16_to_float 
     (
       bfloat16 bf16_val
     )
{
	int32_t inter_temp = *( ( int16_t* ) &bf16_val );
	inter_temp = inter_temp << 16; 
	float float_value = *( float* ) ( &inter_temp );
	return float_value;
}
	
inline float mat_mul_accuracy_check_accum_bf16bf16f32of32
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
		float a_float = bf16_to_float( *( a + i * rs_a + p * cs_a ) ); 
		float b_float = bf16_to_float( *( b + p * rs_b + j * cs_b ) ); 
		temp_accum += ( ( a_float ) * ( b_float ) ); 
	} 
	temp_accum = ( beta * ( * (c_ref + ( rs_c_ref * i ) + ( cs_c_ref * j ) ) ) ) 
			             + ( alpha * temp_accum ); 
	return temp_accum; 					 
}

inline float mat_mul_accuracy_check_accum_bf16bf16f32obf16
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
		float a_float = bf16_to_float( *( a + i*rs_a + p*cs_a ) ); 
		float b_float = bf16_to_float( *( b + p*rs_b + j*cs_b ) ); 
		temp_accum += ( ( a_float ) * ( b_float ) ); 
	}  
	float c_ref_float = bf16_to_float( *( c_ref + i*rs_c_ref + j*cs_c_ref ) ); 
	temp_accum = ( beta * ( c_ref_float ) ) + ( alpha * temp_accum ); 

	return temp_accum; 
}

#define GEN_MAT_MUL_ACC_CHK_DRV_FUNC(A_type,B_type,C_type,ACCUM_type,SCALE_type,BLAS_SFX,BLAS_DOWNSCALE_SFX) \
void mat_mul_accuracy_check_driver_ ## BLAS_SFX \
     ( \
       FILE*   fout, \
       const char stor_order, \
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
	dim_t rs_a = lda; \
	dim_t cs_a = 1; \
	dim_t rs_b = ldb; \
	dim_t cs_b = 1; \
	dim_t rs_c = ldc; \
	dim_t cs_c = 1; \
	dim_t rs_c_ref = ldc_ref; \
	dim_t cs_c_ref = 1; \
 \
	if ( ( stor_order == 'C' ) || ( stor_order == 'c' ) ) \
	{ \
		rs_a = 1; \
		cs_a = lda; \
		rs_b = 1; \
		cs_b = ldb; \
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
				/* Apply bias followed by relu. */ \
				if ( post_op->seq_vector[0] == BIAS ) \
				{ \
					if ( post_op->seq_length >= 1 ) \
					{ \
						temp_accum += ( *( ( ACCUM_type* )post_op->bias.bias + j ) ); \
					} \
					if ( ( post_op->seq_length > 1 ) && \
						 ( post_op->seq_vector[1] == ELTWISE ) ) \
					{ \
						if ( post_op->eltwise.algo.alpha != NULL ) /* PReLU*/ \
						{ \
							temp_accum = ( temp_accum > 0 ) ? \
								temp_accum : \
								( temp_accum * \
								*( ( ACCUM_type* ) post_op->eltwise.algo.alpha ) ); \
						} \
						else \
						{ \
							temp_accum = ( temp_accum > 0 ) ? temp_accum : 0 ; \
						} \
					} \
				} \
				else if ( post_op->seq_vector[0] == ELTWISE ) \
				{ \
					if ( post_op->seq_length >= 1 ) \
					{ \
						if ( post_op->eltwise.algo.alpha != NULL ) /* PReLU*/ \
						{ \
							temp_accum = ( temp_accum > 0 ) ? \
									temp_accum : \
									( temp_accum * *( ( ACCUM_type* ) post_op->eltwise.algo.alpha ) ); \
						} \
						else \
						{ \
							temp_accum = ( temp_accum > 0 ) ? temp_accum : 0 ; \
						} \
					} \
					if ( ( post_op->seq_length > 1 ) && ( post_op->seq_vector[1] == BIAS ) ) \
					{ \
						temp_accum += ( *( ( ACCUM_type* )post_op->bias.bias + j ) ); \
					} \
				} \
			} \
			if ( global_dscale_out == 'y' ) \
			{ \
				out_temp_accum = GEN_FUNC_NAME(mat_mul_accuracy_check_downscale_,BLAS_DOWNSCALE_SFX) \
				        (temp_accum, out_temp_accum, post_op, j); \
			} \
			else \
			{ \
				out_temp_accum = ( C_type )temp_accum; \
			} \
 \
			if ( *( c + ( rs_c * i ) + ( cs_c * j ) ) != out_temp_accum ) \
			{ \
				if ( fout ) \
				{ \
					fprintf( fout, "%s Failure input m: %ld, n: %ld, k: %ld," \
									" lda: %ld, ldb: %ld, ldc: %ld\n", \
									XSTR(BLAS_SFX), m, n, k, lda, ldb, ldc ); \
					fflush( fout ); \
				} \
				printf("failure, m: %ld, n: %ld, k: %ld\n", i, j, k ); \
				goto cleanup_acc; \
			} \
		} \
	} \
cleanup_acc: \
	return; \
} \

GEN_MAT_MUL_ACC_CHK_DRV_FUNC(uint8_t,int8_t,int16_t,int16_t,float,u8s8s16os16,u8s8s16os8)
GEN_MAT_MUL_ACC_CHK_DRV_FUNC(uint8_t,int8_t,int8_t,int16_t,float,u8s8s16os8,u8s8s16os8)
GEN_MAT_MUL_ACC_CHK_DRV_FUNC(uint8_t,int8_t,int32_t,int32_t,float,u8s8s32os32,u8s8s32os8)
GEN_MAT_MUL_ACC_CHK_DRV_FUNC(uint8_t,int8_t,int8_t,int32_t,float,u8s8s32os8,u8s8s32os8)
GEN_MAT_MUL_ACC_CHK_DRV_FUNC(bfloat16,bfloat16,float,float,float,bf16bf16f32of32,bf16bf16f32obf16)
GEN_MAT_MUL_ACC_CHK_DRV_FUNC(bfloat16,bfloat16,bfloat16,float,float,bf16bf16f32obf16,bf16bf16f32obf16)
GEN_MAT_MUL_ACC_CHK_DRV_FUNC(float,float,float,float,float,f32f32f32of32,bf16bf16f32obf16) 

/* Only supports bias followed by RELU and vice versa for now.*/ \
#define GEN_MAT_MUL_POST_OPS_CREATOR(C_type,DSCALE_type,BLAS_SFX) \
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
	/* Only supporting 3 post ops at max for now.*/ \
	dim_t max_post_ops_seq_length = 3; \
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
	post_ops->eltwise.algo.alpha = NULL; \
	post_ops->bias.bias = NULL; \
	post_ops->sum.scale_factor = NULL; \
	if ( post_ops_str != NULL ) \
	{ \
		char* ops_tok = strtok(post_ops_str, ", " ); \
		bool is_param_relu = FALSE; \
		while ( ops_tok ) \
		{ \
			if ( strcmp( ops_tok, "bias") == 0 ) \
			{ \
				post_ops->seq_vector[cur_op_index] = BIAS; \
			} \
			else if ( strcmp( ops_tok, "relu") == 0 ) \
			{ \
				post_ops->seq_vector[cur_op_index] = ELTWISE; \
			} \
			else if ( strcmp( ops_tok, "prelu") == 0 ) \
			{ \
				post_ops->seq_vector[cur_op_index] = ELTWISE; \
				is_param_relu = TRUE; \
			} \
			ops_tok = strtok( NULL, ", " ); \
			cur_op_index++; \
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
		post_ops->eltwise.is_power_of_2 = FALSE; \
		post_ops->eltwise.scale_factor = NULL; \
		post_ops->eltwise.algo.alpha = NULL; \
		post_ops->eltwise.algo.algo_type = RELU; \
		if ( is_param_relu == TRUE ) \
		{ \
			post_ops->eltwise.algo.alpha = malloc( sizeof( C_type ) ); \
			*( ( C_type* ) post_ops->eltwise.algo.alpha ) = ( C_type )6; \
			post_ops->eltwise.algo.algo_type = PRELU; \
		} \
		post_ops->eltwise.algo.beta = NULL; \
	} \
 \
	if ( global_dscale_out == 'y' ) \
	{ \
		post_ops->seq_vector[cur_op_index] = SCALE; \
		cur_op_index++; \
 \
		post_ops->sum.is_power_of_2 = FALSE; \
		post_ops->sum.scale_factor = NULL; \
		post_ops->sum.buff = NULL; \
		post_ops->sum.zero_point = NULL; \
		if ( global_dscale_out == 'y' ) \
		{ \
			/* Allocate scale buffer, return early if alloc fails.*/ \
			post_ops->sum.scale_factor = malloc( n * sizeof( DSCALE_type ) ); \
			if ( post_ops->sum.scale_factor == NULL ) \
			{ \
				free ( post_ops->bias.bias ); \
				free( post_ops->seq_vector ); \
				free( post_ops ); \
				return NULL; \
			} \
			/* Fill scale factor.*/ \
			DSCALE_type* temp_dscale_ptr = ( DSCALE_type* )post_ops->sum.scale_factor; \
			for ( dim_t i = 0; i < n; ++i ) \
			{ \
				temp_dscale_ptr[i] = ( ( DSCALE_type )1 )/ ( ( DSCALE_type )1000 ); \
			} \
		} \
	} \
 \
	post_ops->seq_length = cur_op_index; \
 \
	return post_ops; \
} \

GEN_MAT_MUL_POST_OPS_CREATOR(int16_t,float,u8s8s16os16)
GEN_MAT_MUL_POST_OPS_CREATOR(int32_t,float,u8s8s32os32)
GEN_MAT_MUL_POST_OPS_CREATOR(float,float,bf16bf16f32of32) 
GEN_MAT_MUL_POST_OPS_CREATOR(float,float,f32f32f32of32)

void lpgemm_destroy_post_ops_struct( aocl_post_op* post_ops )
{
	if ( post_ops == NULL )
	{
		return;
	}

	if ( post_ops->eltwise.algo.alpha != NULL )
	{
		free( post_ops->eltwise.algo.alpha );
	}
	if ( post_ops->sum.scale_factor != NULL )
	{
		free( post_ops->sum.scale_factor );
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

#define GEN_MAT_MUL_BENCH_MAIN_FUNC(A_type,B_type,C_type,BLAS_SFX,REORDER_SFX) \
void mat_mul_bench_main_ ## BLAS_SFX \
     ( \
       FILE*   fin, \
       FILE*   fout, \
       char    stor_order, \
       char    op_t, \
       int32_t m, \
       int32_t n, \
       int32_t k, \
       int32_t stride_a, \
       int32_t stride_b, \
       int32_t stride_c, \
       char*   post_ops_str \
     ) \
{ \
	if ( ( op_t != 'p' ) && ( op_t != 'P' ) && ( op_t != 'r' ) && ( op_t != 'R' ) ) \
	{ \
		printf("The op_t ( 2nd arg in input.txt) is not valid\n"); \
		return; \
	} \
 \
	int32_t n_repeats = bli_max( 30, bli_min(( 3e10 / ( ( int64_t )m * n * k )), 100 )); \
	if ( global_n_repeat > 0 ) \
	{ \
		n_repeats = global_n_repeat; \
	} \
 \
	/* Get 64 byte aligned memory.*/ \
	A_type* a = ( A_type* ) bli_malloc_user( sizeof( A_type ) * m * k ); \
 \
	B_type* b = ( B_type* ) bli_malloc_user( sizeof( B_type ) * n * k ); \
 \
	C_type* c = ( C_type* ) bli_malloc_user( sizeof( C_type ) * m * n ); \
	memset( ( void* ) c, 0, sizeof( C_type ) * m * n ); \
 \
	C_type* c_ref = ( C_type* ) bli_malloc_user( sizeof( C_type ) * m * n ); \
	memset( ( void* ) c_ref, 0, sizeof( C_type ) * m * n ); \
 \
	C_type alpha; \
	C_type beta; \
	if ( bench_mode == 'p' ) \
	{ \
		alpha = 1; \
		beta = 0; \
	} \
	else if ( bench_mode == 'a' ) \
	{ \
		alpha = 2; \
		beta = 9; \
	} \
 \
	GEN_FUNC_NAME(fill_array_,A_type)( a, ( m * k ) ); \
	GEN_FUNC_NAME(fill_array_,B_type)( b, ( k * n ) ); \
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
	if ( ( op_t == 'p' ) || ( op_t == 'P' ) ) \
	{ \
		/* No reordering of B.*/ \
		GEN_FUNC_NAME(mat_mul_bench_driver_,BLAS_SFX) \
		( \
		  stor_order, op_t, n_repeats, m, n, k, \
		  alpha, \
		  a, stride_a, \
		  b, stride_b, \
		  beta, \
		  c, stride_c, \
		  post_op \
		); \
	} \
	else if ( ( op_t == 'r' ) || ( op_t == 'R' ) ) \
	{ \
		/* Reorder B.*/ \
		siz_t b_reorder_buf_siz_req = \
			GEN_FUNC_NAME(aocl_get_reorder_buf_size_,REORDER_SFX)( 'B', k, n ); \
 \
		B_type* b_reorder = ( B_type* ) bli_malloc_user( b_reorder_buf_siz_req ); \
		GEN_FUNC_NAME(aocl_reorder_,REORDER_SFX)( 'B', b, b_reorder, k, n, stride_b ); \
 \
		GEN_FUNC_NAME(mat_mul_bench_driver_,BLAS_SFX) \
		( \
		  stor_order, op_t, n_repeats, m, n, k, \
		  alpha, \
		  a, stride_a, \
		  b_reorder, stride_b, \
		  beta, \
		  c, stride_c, \
		  post_op \
		); \
 \
		bli_free_user( b_reorder ); \
	} \
 \
	if ( bench_mode == 'a' ) \
	{ \
		printf("Running accuracy check.\n"); \
		GEN_FUNC_NAME(mat_mul_accuracy_check_driver_,BLAS_SFX) \
		( \
		  fout, stor_order, m, n, k, \
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
	if ( a != NULL ) \
	{ \
		bli_free_user( a ); \
	} \
	if ( b != NULL ) \
	{ \
		bli_free_user( b ); \
	} \
	if ( c != NULL ) \
	{ \
		bli_free_user( c ); \
	} \
	if ( c_ref != NULL ) \
	{ \
		bli_free_user( c_ref ); \
	} \
} \

GEN_MAT_MUL_BENCH_MAIN_FUNC(uint8_t,int8_t,int16_t,u8s8s16os16,u8s8s16os16)
GEN_MAT_MUL_BENCH_MAIN_FUNC(uint8_t,int8_t,int8_t,u8s8s16os8,u8s8s16os16)
GEN_MAT_MUL_BENCH_MAIN_FUNC(uint8_t,int8_t,int32_t,u8s8s32os32,u8s8s32os32)
GEN_MAT_MUL_BENCH_MAIN_FUNC(uint8_t,int8_t,int8_t,u8s8s32os8,u8s8s32os32)
GEN_MAT_MUL_BENCH_MAIN_FUNC(float,float,float,f32f32f32of32,f32f32f32of32)

#define GEN_MAT_MUL_BENCH_MAIN_FUNC_BF16(C_type, BLAS_SFX) \
void mat_mul_bench_main_ ## BLAS_SFX \
     ( \
       FILE*   fin, \
       FILE*   fout, \
	   char    stor_order, \
       char    op_t, \
       int32_t m, \
       int32_t n, \
       int32_t k, \
       int32_t stride_a, \
       int32_t stride_b, \
       int32_t stride_c, \
	   char*   post_ops_str \
     ) \
{ \
	if ( ( op_t != 'p' ) && ( op_t != 'P' ) && ( op_t != 'r' ) && ( op_t != 'R' ) ) \
	{ \
		printf("The op_t ( 2nd arg in input.txt) is not valid\n");\
		return; \
	} \
 \
	int32_t n_repeats = bli_max( 30, bli_min(( 3e10 / ( ( int64_t )m * n * k )), 1000 )); \
	if ( global_n_repeat > 0 ) \
	{ \
		n_repeats = global_n_repeat; \
	} \
 \
	/* Get 64 byte aligned memory.*/ \
	bfloat16* a = ( bfloat16* ) bli_malloc_user( sizeof( bfloat16 ) * m * k ); \
	float *a_float = bli_malloc_user( m * k * sizeof( float )); \
	for ( int32_t i = 0; i < m*k; ++i ) \
    { \
        a_float[i] = ( float ) ( i % 5 ); \
    } \
	convert_float_arr_to_bf16( a_float, a, m * k ); \
 \
	bfloat16* b = ( bfloat16* ) bli_malloc_user( sizeof( bfloat16 ) * n * k ); \
	float *b_float = bli_malloc_user( k * n * sizeof( float ));  \
	for ( int32_t i = 0; i < k*n; ++i ) \
	{ \
		b_float[i] = ( float ) ( i % 5 );\
	} \
	convert_float_arr_to_bf16( b_float, b, k * n ); \
 \
	C_type* c = ( C_type* ) bli_malloc_user( sizeof( C_type ) * m * n ); \
	memset( ( void* ) c, 0, sizeof( C_type ) * m * n ); \
 \
	C_type* c_ref = ( C_type* ) bli_malloc_user( sizeof( C_type ) * m * n ); \
	memset( ( void* ) c_ref, 0, sizeof( C_type ) * m * n ); \
 \
	float alpha; \
	float beta; \
	if ( bench_mode == 'p' ) \
	{ \
		alpha = 1; \
		beta = 0; \
	} \
	else if ( bench_mode == 'a' ) \
	{ \
		alpha = 2; \
		beta = 9; \
	} 	\
 \
	aocl_post_op* post_op = NULL; \
	if ( ( post_ops_str != NULL ) || ( global_dscale_out == 'y' ) ) \
	{ \
		post_op = lpgemm_create_post_ops_struct_bf16bf16f32of32( m, n, post_ops_str ); \
		if ( post_op == NULL ) \
		{ \
			printf(" post op struct allocation failure, returning.\n"); \
			return; \
		} \
	} \
 \
	if ( ( op_t == 'p' ) || ( op_t == 'P' ) ) \
	{ \
		/* No reordering of B.*/ \
		GEN_FUNC_NAME(mat_mul_bench_driver_,BLAS_SFX) \
		( \
		  stor_order, op_t, n_repeats, m, n, k, \
		  alpha, \
		  a, stride_a, \
		  b, stride_b, \
		  beta, \
		  c, stride_c, \
		  post_op \
		); \
	} \
	else if ( ( op_t == 'r' ) || ( op_t == 'R' ) ) \
	{ \
		/* Reorder B.*/ \
		siz_t b_reorder_buf_siz_req = \
			aocl_get_reorder_buf_size_bf16bf16f32of32( 'B', k, n ); \
 \
		bfloat16* b_reorder = ( bfloat16* ) bli_malloc_user( b_reorder_buf_siz_req ); \
			aocl_reorder_bf16bf16f32of32( 'B', b, b_reorder, k, n, stride_b ); \
 \
		GEN_FUNC_NAME(mat_mul_bench_driver_,BLAS_SFX) \
		( \
		  stor_order, op_t, n_repeats, m, n, k, \
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
		  fout, stor_order, m, n, k, \
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
	if ( a != NULL ) \
	{ \
		bli_free_user( a ); \
	} \
	if ( b != NULL ) \
	{ \
		bli_free_user( b ); \
	} \
	if ( a_float != NULL ) \
	{ \
		bli_free_user( a_float ); \
	} \
	if ( b_float != NULL ) \
	{ \
		bli_free_user( b_float ); \
	} \
	if ( c != NULL ) \
	{ \
		bli_free_user( c ); \
	} \
	if ( c_ref != NULL ) \
 	{ \
 		bli_free_user( c_ref ); \
 	} \
} \

GEN_MAT_MUL_BENCH_MAIN_FUNC_BF16(float,bf16bf16f32of32)
GEN_MAT_MUL_BENCH_MAIN_FUNC_BF16(bfloat16,bf16bf16f32obf16)

int main( int argc, char** argv )
{
	FILE* fin  = NULL;
	if ( argc < 5 )
	{
		printf( "Usage: ./mat_mul -i input.txt -m mode < -n 1000 -o op1,op2.. >" \
						"\nMode is either a or p. a is used for accuracy test, " \
						"whereas p is used for performance benchmarking." \
						"\nn_repeats can be set optionally using -n arg." \
						"\nPost ops can be executed optionaly by providing a " \
						"coma separated list of ops after -o arg.\nCurrently " \
						"bias and relu/prelu is supported and can be specified " \
			 			"as a single post op or combination of the same. eg: -o bias,relu ; -o prelu." \
						"\nDownscaled version of an API can be enabled by using -d arg. " \
						"downscale is used to enable- u8s8s32os8, u8s8s16os8 or bf16bf16f32obf16 \n" );
		exit( 1 );
	}

	char* file_name = NULL;
	char* post_ops_str = NULL;
	char* post_ops_str_dest = NULL; //Strtok is used to parse, need to maintain a copy.

	// Parse CLI arguments.
	opterr = 0;
	int opt_val;
	while ( ( opt_val = getopt( argc, argv, "i:m:n:o:d" ) ) != -1 )
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
			case 'o':
					post_ops_str = optarg;
					break;
			case 'd':
					global_dscale_out = 'y';
					break;
			default:
					break;
		}
	}

	if ( post_ops_str != NULL )
	{
		post_ops_str_dest = strdup( post_ops_str );
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
	char op_t;
	char stor_order;
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
		while ( fscanf( fin, "%c %c %c %d %d %d %d %d %d\n",
				&op_type_char, &stor_order, &op_t, &m, &n, &k,
				&stride_a, &stride_b, &stride_c ) == 9 )
		{
			stor_order = ( ( stor_order == 'r' ) || ( stor_order == 'R' ) ||
							( stor_order == 'c' ) || ( stor_order == 'C' ) ) ?
							stor_order : 'r';

			if ( ( op_type_char == 'i' ) || ( op_type_char == 'I' ) )
			{
				if ( global_dscale_out == 'n' )
				{
					GEN_FUNC_NAME(mat_mul_bench_main_,u8s8s32os32)
					(
					  fin, fout, stor_order, op_t,
					  m, n, k, stride_a, stride_b, stride_c,
					  post_ops_str_dest
					);
				}
				else
				{
					GEN_FUNC_NAME(mat_mul_bench_main_,u8s8s32os8)
					(
					  fin, fout, stor_order, op_t,
					  m, n, k, stride_a, stride_b, stride_c,
					  post_ops_str_dest
					);
				}
			}
			else if ( ( op_type_char == 'f' ) || ( op_type_char == 'F' ) )
			{
				GEN_FUNC_NAME(mat_mul_bench_main_,f32f32f32of32)
				(
				  fin, fout, stor_order, op_t,
				  m, n, k, stride_a, stride_b, stride_c,
				  NULL
				);
			}
			else if ((op_type_char == 's') || (op_type_char == 'S'))
			{
				if ( global_dscale_out == 'n' )
				{
					GEN_FUNC_NAME(mat_mul_bench_main_,u8s8s16os16)
					(
						fin, fout, stor_order, op_t,
						m, n, k, stride_a, stride_b, stride_c,
						post_ops_str_dest
					);
				}
				else
				{
					GEN_FUNC_NAME(mat_mul_bench_main_,u8s8s16os8)
					(
						fin, fout, stor_order, op_t,
						m, n, k, stride_a, stride_b, stride_c,
						post_ops_str_dest
					);
				}
			}
			if ((op_type_char == 'b') || (op_type_char == 'B'))
			{
				if ( global_dscale_out == 'n' )
				{
					GEN_FUNC_NAME(mat_mul_bench_main_, bf16bf16f32of32)
					(
						fin, fout, stor_order, op_t,
						m, n, k, stride_a, stride_b, stride_c,
						post_ops_str_dest
					);
				}
				else
				{
					GEN_FUNC_NAME(mat_mul_bench_main_, bf16bf16f32obf16)
					(
						fin, fout, stor_order, op_t,
						m, n, k, stride_a, stride_b, stride_c,
						post_ops_str_dest
					);
				}	
			}
			if ( post_ops_str != NULL )
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
