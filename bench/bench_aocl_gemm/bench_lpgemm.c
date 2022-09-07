#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <immintrin.h>
#include <time.h>
#include <float.h>
#include <unistd.h>

#include "blis.h"

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

#define GEN_BLIS_MAT_MUL_FUNC(A_type,B_type,C_type,BLAS_SFX) \
void mat_mul_ ## BLAS_SFX \
     ( \
       char    op_t, \
       dim_t   m, \
       dim_t   n, \
       dim_t   k, \
       C_type  alpha, \
       A_type* a, \
       dim_t   lda, \
       B_type* b, \
       dim_t   ldb, \
       C_type  beta, \
       C_type* c, \
       dim_t   ldc, \
       aocl_post_op*  post_op\
     ) \
{ \
	char storage = 'r'; \
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

GEN_BLIS_MAT_MUL_FUNC(uint8_t,int8_t,int16_t,u8s8s16os16)
GEN_BLIS_MAT_MUL_FUNC(uint8_t,int8_t,int8_t,u8s8s16os8)
GEN_BLIS_MAT_MUL_FUNC(uint8_t,int8_t,int32_t,u8s8s32os32)
GEN_BLIS_MAT_MUL_FUNC(uint8_t,int8_t,int8_t,u8s8s32os8)
GEN_BLIS_MAT_MUL_FUNC(float,float,float,f32f32f32of32)

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

#define GEN_MAT_MUL_BENCH_DRV_FUNC(A_type,B_type,C_type,BLAS_SFX) \
void mat_mul_bench_driver_ ## BLAS_SFX \
     ( \
       char    op_t, \
       int32_t n_repeats, \
       dim_t   m, \
       dim_t   n, \
       dim_t   k, \
       C_type  alpha, \
       A_type* a, \
       dim_t   lda, \
       B_type* b, \
       dim_t   ldb, \
       C_type  beta, \
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
		  op_t, m, n, k, \
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

GEN_MAT_MUL_BENCH_DRV_FUNC(uint8_t,int8_t,int16_t,u8s8s16os16)
GEN_MAT_MUL_BENCH_DRV_FUNC(uint8_t,int8_t,int8_t,u8s8s16os8)
GEN_MAT_MUL_BENCH_DRV_FUNC(uint8_t,int8_t,int32_t,u8s8s32os32)
GEN_MAT_MUL_BENCH_DRV_FUNC(uint8_t,int8_t,int8_t,u8s8s32os8)
GEN_MAT_MUL_BENCH_DRV_FUNC(float,float,float,f32f32f32of32)

#define GEN_MAT_MUL_ACC_CHK_DRV_FUNC(A_type,B_type,C_type,DSCALE_type,SCALE_type,BLAS_SFX) \
void mat_mul_accuracy_check_driver_ ## BLAS_SFX \
     ( \
       FILE*   fout, \
       dim_t   m, \
       dim_t   n, \
       dim_t   k, \
       C_type  alpha, \
       A_type* a, \
       dim_t   lda, \
       B_type* b, \
       dim_t   ldb, \
       C_type  beta, \
       C_type* c, \
       dim_t   ldc, \
       C_type* c_ref, \
       dim_t   ldc_ref, \
       aocl_post_op*  post_op\
     ) \
{ \
	for ( dim_t i = 0; i < m; ++i ) \
	{ \
		for ( dim_t j = 0; j < n; ++j ) \
		{ \
			DSCALE_type temp_accum = 0; \
			C_type out_temp_accum = 0; \
 \
			for ( dim_t p = 0; p < k; ++p) \
			{ \
				temp_accum += ( *( a + ( i * lda ) + p ) * *( b + ( p * ldb ) + j ) ); \
			} \
 \
			temp_accum = ( beta * ( * (c_ref + ( ldc_ref * i ) + j ) ) ) \
			             + ( alpha * temp_accum ); \
 \
			if ( post_op != NULL ) \
			{ \
				/* Apply bias followed by relu. */ \
				if ( post_op->seq_vector[0] == BIAS ) \
				{ \
					if ( post_op->seq_length >= 1 ) \
					{ \
						temp_accum += ( *( ( DSCALE_type* )post_op->bias.bias + j ) ); \
					} \
					if ( ( post_op->seq_length > 1 ) && \
						 ( post_op->seq_vector[1] == ELTWISE ) ) \
					{ \
						if ( post_op->eltwise.algo.alpha != NULL ) /* PReLU*/ \
						{ \
							temp_accum = ( temp_accum > 0 ) ? \
								temp_accum : \
								( temp_accum * \
								*( ( DSCALE_type* ) post_op->eltwise.algo.alpha ) ); \
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
									( temp_accum * *( ( DSCALE_type* ) post_op->eltwise.algo.alpha ) ); \
						} \
						else \
						{ \
							temp_accum = ( temp_accum > 0 ) ? temp_accum : 0 ; \
						} \
					} \
					if ( ( post_op->seq_length > 1 ) && ( post_op->seq_vector[1] == BIAS ) ) \
					{ \
						temp_accum += ( *( ( DSCALE_type* )post_op->bias.bias + j ) ); \
					} \
				} \
			} \
			if ( global_dscale_out == 'y' ) \
			{ \
				out_temp_accum = ( C_type )lroundf( ( SCALE_type )temp_accum * \
								( *( ( SCALE_type* )post_op->sum.scale_factor + j ) ) ); \
			} \
			else \
			{ \
				out_temp_accum = ( C_type )temp_accum; \
			} \
 \
			if ( *( c + ( ldc * i ) + j ) != out_temp_accum ) \
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

GEN_MAT_MUL_ACC_CHK_DRV_FUNC(uint8_t,int8_t,int16_t,int16_t,float,u8s8s16os16)
GEN_MAT_MUL_ACC_CHK_DRV_FUNC(uint8_t,int8_t,int8_t,int16_t,float,u8s8s16os8)
GEN_MAT_MUL_ACC_CHK_DRV_FUNC(uint8_t,int8_t,int32_t,int32_t,float,u8s8s32os32)
GEN_MAT_MUL_ACC_CHK_DRV_FUNC(uint8_t,int8_t,int8_t,int32_t,float,u8s8s32os8)
GEN_MAT_MUL_ACC_CHK_DRV_FUNC(float,float,float,float,float,f32f32f32of32)

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
		  op_t, n_repeats, m, n, k, \
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
		  op_t, n_repeats, m, n, k, \
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
		  fout, m, n, k, \
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
			 			"as a single post op or combination of the same. eg: -o bias,relu ; -o prelu.\n" );
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

		while ( fscanf( fin, "%c %c %d %d %d %d %d %d\n",
				&op_type_char, &op_t, &m, &n, &k,
				&stride_a, &stride_b, &stride_c ) == 8 )
		{
			if ( ( op_type_char == 'i' ) || ( op_type_char == 'I' ) )
			{
				if ( global_dscale_out == 'n' )
				{
					GEN_FUNC_NAME(mat_mul_bench_main_,u8s8s32os32)
					(
					  fin, fout, op_t,
					  m, n, k, stride_a, stride_b, stride_c,
					  post_ops_str_dest
					);
				}
				else
				{
					GEN_FUNC_NAME(mat_mul_bench_main_,u8s8s32os8)
					(
					  fin, fout, op_t,
					  m, n, k, stride_a, stride_b, stride_c,
					  post_ops_str_dest
					);
				}
			}
			else if ( ( op_type_char == 'f' ) || ( op_type_char == 'F' ) )
			{
				GEN_FUNC_NAME(mat_mul_bench_main_,f32f32f32of32)
				(
				  fin, fout, op_t,
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
						fin, fout, op_t,
						m, n, k, stride_a, stride_b, stride_c,
						post_ops_str_dest
					);
				}
				else
				{
					GEN_FUNC_NAME(mat_mul_bench_main_,u8s8s16os8)
					(
						fin, fout, op_t,
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
