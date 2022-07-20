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

#define _XSTR(str) #str
#define XSTR(str) _XSTR(str)

#define GEN_FUNC_NAME(prototype,ctype) prototype ## ctype

#define GEN_FILL_ARRAY_FUNC(ctype) \
void fill_array_ ## ctype ( void* arr, dim_t size ) \
{ \
	ctype* temp_arr = ( ctype* ) arr; \
	for ( dim_t i = 0; i < size; ++i ) \
	{ \
		temp_arr[i] = ( ctype )( i % 100 ); \
	} \
} \

GEN_FILL_ARRAY_FUNC(uint8_t)
GEN_FILL_ARRAY_FUNC(int8_t)
GEN_FILL_ARRAY_FUNC(float)

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
       dim_t   ldc \
     ) \
{ \
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
	aocl_gemm_ ## BLAS_SFX( transa, transb, m, n, k, \
					alpha, \
					a, lda, reordera, \
					b, ldb, reorderb, \
					beta, \
					c, ldc ); \
} \

GEN_BLIS_MAT_MUL_FUNC(uint8_t,int8_t,int32_t,u8s8s32os32)
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
					" GFlops: %f, n_repeats: %d\n", 
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
       dim_t   ldc \
     ) \
{ \
	double min_time_diff = DBL_MAX; \
	for ( int32_t nr = 0; nr < n_repeats; ++nr ) \
	{ \
		if ( bench_mode == 'a' ) \
		{ \
			memset( ( void* ) c, 0, sizeof( float ) * m * n ); \
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
		  c, ldc \
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

GEN_MAT_MUL_BENCH_DRV_FUNC(uint8_t,int8_t,int32_t,u8s8s32os32)
GEN_MAT_MUL_BENCH_DRV_FUNC(float,float,float,f32f32f32of32)

#define GEN_MAT_MUL_ACC_CHK_DRV_FUNC(A_type,B_type,C_type,BLAS_SFX) \
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
       dim_t   ldc_ref \
     ) \
{ \
	for ( dim_t i = 0; i < m; ++i ) \
	{ \
		for ( dim_t j = 0; j < n; ++j ) \
		{ \
			C_type temp_accum = 0; \
 \
			for ( dim_t p = 0; p < k; ++p) \
			{ \
				temp_accum += ( *( a + ( i * lda ) + p ) * *( b + ( p * ldb ) + j ) ); \
			} \
 \
			temp_accum = ( beta * ( * (c_ref + ( ldc_ref * i ) + j ) ) ) \
			             + ( alpha * temp_accum ); \
			if ( *( c + ( ldc * i ) + j ) != temp_accum ) \
			{ \
				if ( fout ) \
				{ \
					fprintf( fout, "%s Failure input m: %ld, n: %ld, k: %ld," \
									" lda: %ld, ldb: %ld, ldc: %ld\n", \
									XSTR(BLAS_SFX), m, n, k, lda, ldb, ldc ); \
					fflush( fout ); \
				} \
				printf("failure, m: %ld, n: %ld, k: %ld\n", i, j, k); \
				goto cleanup_acc; \
			} \
		} \
	} \
cleanup_acc: \
	return; \
} \

GEN_MAT_MUL_ACC_CHK_DRV_FUNC(uint8_t,int8_t,int32_t,u8s8s32os32)
GEN_MAT_MUL_ACC_CHK_DRV_FUNC(float,float,float,f32f32f32of32)

#define GEN_MAT_MUL_BENCH_MAIN_FUNC(A_type,B_type,C_type,BLAS_SFX) \
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
       int32_t stride_c \
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
		  c, stride_c \
		); \
	} \
	else if ( ( op_t == 'r' ) || ( op_t == 'R' ) ) \
	{ \
		/* Reorder B.*/ \
		siz_t b_reorder_buf_siz_req = \
			GEN_FUNC_NAME(aocl_get_reorder_buf_size_,BLAS_SFX)( 'B', k, n ); \
 \
		B_type* b_reorder = ( B_type* ) bli_malloc_user( b_reorder_buf_siz_req ); \
		GEN_FUNC_NAME(aocl_reorder_,BLAS_SFX)( 'B', b, b_reorder, k, n, stride_b ); \
 \
		GEN_FUNC_NAME(mat_mul_bench_driver_,BLAS_SFX) \
		( \
		  op_t, n_repeats, m, n, k, \
		  alpha, \
		  a, stride_a, \
		  b_reorder, stride_b, \
		  beta, \
		  c, stride_c \
		); \
 \
		bli_free_user( b_reorder ); \
	} \
 \
	if ( bench_mode == 'a' ) \
	{ \
		printf(" Running accuracy check.\n"); \
		GEN_FUNC_NAME(mat_mul_accuracy_check_driver_,BLAS_SFX) \
		( \
		  fout, m, n, k, \
		  alpha, \
		  a, stride_a, \
		  b, stride_b, \
		  beta, \
		  c, stride_c, \
		  c_ref, stride_c \
		); \
	} \
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

GEN_MAT_MUL_BENCH_MAIN_FUNC(uint8_t,int8_t,int32_t,u8s8s32os32)
GEN_MAT_MUL_BENCH_MAIN_FUNC(float,float,float,f32f32f32of32)

int main( int argc, char** argv )
{
	FILE* fin  = NULL;
	if ( argc < 5 )
	{
		printf( "Usage: ./mat_mul -i input.txt -m mode < -n 1000 >\nMode is either a or p." \
				" a is used for accuracy test, whereas p is used for" \
				" performance benchmarking.\nn_repeats can be set" \
				" optionally using -n argument.\n" );
		exit( 1 );
	}

	char* file_name = NULL;

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
	char op_t;
	int32_t m, n, k;
	int32_t stride_a, stride_b, stride_c;

	const dim_t len_list_omp_cores_for_testing = 6;
	const dim_t list_omp_cores_for_testing[6] = { 100, 80, 64, 24, 8, 1 };

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
				GEN_FUNC_NAME(mat_mul_bench_main_,u8s8s32os32)
				(
				  fin, fout, op_t,
				  m, n, k, stride_a, stride_b, stride_c
				);
			}
			else if ( ( op_type_char == 'f' ) || ( op_type_char == 'F' ) )
			{
				GEN_FUNC_NAME(mat_mul_bench_main_,f32f32f32of32)
				(
				  fin, fout, op_t,
				  m, n, k, stride_a, stride_b, stride_c
				);
			}
		}
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
