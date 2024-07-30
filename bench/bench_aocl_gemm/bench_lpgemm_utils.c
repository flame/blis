/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2022 - 2024, Advanced Micro Devices, Inc. All rights reserved.

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
#include <time.h>
#include <float.h>
#include <math.h>

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
		temp_arr[i] = ( ctype )( i % 10 ); \
	} \
} \

GEN_FILL_ARRAY_FUNC(float)

void print_result
     (
       const char* msg,
       int32_t     n_repeats,
       dim_t       n,
       dim_t       incx,
       double      runtime
     )
{
	printf("%s n: %ld, incx: %ld, runtime: %f s, n_repeats: %d\n", \
			msg, n, incx, runtime, n_repeats);
}

#define GEN_GELU_BENCH_DRV_FN(V_type,GELU_SFX) \
void gelu_bench_driver_ ## GELU_SFX \
     ( \
       int32_t n_repeats, \
       dim_t   n, \
       V_type* x, \
       inc_t   incx \
     ) \
{ \
	double   dtime;                 \
	double   dtime_save = DBL_MAX;  \
	for ( int32_t nr = 0; nr < n_repeats; ++nr ) \
	{ \
		dtime = bli_clock();  \
 \
		if ( bench_mode == 'a' ) \
		{ \
			GEN_FUNC_NAME(fill_array_,V_type)( x, ( n * incx) ); \
		} \
 \
		GEN_FUNC_NAME(aocl_gemm_,GELU_SFX) \
		( \
		  n, x, incx \
		); \
 \
		dtime_save = bli_clock_min_diff( dtime_save, dtime ); \
 \
	} \
 \
	print_result( XSTR(GELU_SFX), n_repeats, n, incx, dtime_save); \
} \

GEN_GELU_BENCH_DRV_FN(float,gelu_tanh_f32)
GEN_GELU_BENCH_DRV_FN(float,gelu_erf_f32)

#define GEN_SOFTMAX_BENCH_DRV_FN(V_type,SOFTMAX_SFX) \
void softmax_bench_driver_ ## SOFTMAX_SFX \
     ( \
       int32_t n_repeats, \
       dim_t   n, \
       V_type* x, \
       inc_t   incx \
     ) \
{ \
	double   dtime;                 \
	double   dtime_save = DBL_MAX;  \
	for ( int32_t nr = 0; nr < n_repeats; ++nr ) \
	{ \
		dtime = bli_clock();  \
 \
		if ( bench_mode == 'a' ) \
		{ \
			GEN_FUNC_NAME(fill_array_,V_type)( x, ( n * incx) ); \
		} \
 \
		GEN_FUNC_NAME(aocl_gemm_,SOFTMAX_SFX) \
		( \
		  n, x, incx \
		); \
 \
		dtime_save = bli_clock_min_diff( dtime_save, dtime ); \
	} \
 \
	print_result( XSTR(SOFTMAX_SFX), n_repeats, n, incx, dtime_save); \
} \

GEN_SOFTMAX_BENCH_DRV_FN(float,softmax_f32)

static inline float gelu_tanh_f32
     (
       float temp_accum
     )
{
	temp_accum = 0.5 *(double)temp_accum * (1 + tanhf( 0.797884 * ( (double)temp_accum + \
	              ( 0.044715 * ((double)temp_accum * (double)temp_accum * \
				  (double)temp_accum ) ) ) ) );
	return temp_accum;
}\

static inline float gelu_erf_f32
     (
       float temp_accum
     )
{
	temp_accum = 0.5 *(double)temp_accum * (1 + erff( (double)temp_accum * 0.707107 ));
	return temp_accum;
}

#define GEN_GELU_ACC_CHK_FN(V_type,GELU_SFX) \
void gelu_acc_check_ ## GELU_SFX \
     ( \
       FILE*   fout, \
       dim_t n, \
       V_type* x, \
       V_type* ref_x, \
       inc_t incx \
     ) \
{ \
	for ( dim_t idx = 0; idx < ( n * incx ); idx += incx ) \
	{ \
		V_type temp_acc = GELU_SFX( *( ref_x + idx ) ); \
		if ( temp_acc != *( x + idx ) ) \
		{ \
			if ( fout ) \
			{ \
				fprintf( fout, "%s Failure input n: %ld, incx: %ld, idx: %ld \n", \
								XSTR(GELU_SFX), n, incx, ( idx / incx ) ); \
				fflush( fout ); \
			} \
			printf("%s failure, n: %ld, incx: %ld, idx: %ld, ref: %f, calc: %f\n", \
						XSTR(GELU_SFX), n, incx, ( idx / incx ), temp_acc, *(x + idx)); \
			goto cleanup_acc; \
		} \
	} \
cleanup_acc: \
	return; \
} \

GEN_GELU_ACC_CHK_FN(float,gelu_tanh_f32)
GEN_GELU_ACC_CHK_FN(float,gelu_erf_f32)

#define GEN_SOFTMAX_ACC_CHK_FN(V_type,SOFTMAX_SFX) \
void softmax_acc_check_ ## SOFTMAX_SFX \
     ( \
       FILE*   fout, \
       dim_t n, \
       V_type* x, \
       V_type* ref_x, \
       inc_t incx \
     ) \
{ \
	double exp_sum = 0.0; \
	for ( dim_t idx = 0; idx < ( n * incx ); idx += incx )\
	{ \
		exp_sum += ( double )expf( *(ref_x + idx ) ); \
	} \
	for ( dim_t idx = 0; idx < ( n * incx ); idx += incx ) \
	{ \
		V_type temp_acc = ( V_type )( ( ( double )*( ref_x + idx ) ) / exp_sum ); \
		if ( temp_acc != *( x + idx ) ) \
		{ \
			if ( fout ) \
			{ \
				fprintf( fout, "%s Failure input n: %ld, incx: %ld, idx: %ld \n", \
								XSTR(SOFTMAX_SFX), n, incx, ( idx / incx ) ); \
				fflush( fout ); \
			} \
			printf("%s failure, n: %ld, incx: %ld, idx: %ld, ref: %.10f, calc: %.10f\n", \
						XSTR(SOFTMAX_SFX), n, incx, ( idx / incx ), temp_acc, *(x + idx)); \
			goto cleanup_acc; \
		} \
	} \
cleanup_acc: \
	return; \
} \

GEN_SOFTMAX_ACC_CHK_FN(float,softmax_f32)

#define GEN_GELU_BENCH_MAIN_FN(V_type,GELU_SFX) \
void gelu_bench_main_ ## GELU_SFX \
    ( \
       FILE*   fout, \
       dim_t n, \
       inc_t incx \
     ) \
{ \
	int32_t n_repeats = 1000; \
	if ( global_n_repeat > 0 ) \
	{ \
		n_repeats = global_n_repeat; \
	} \
 \
	err_t bli_errors = BLIS_SUCCESS; \
	V_type* x = ( V_type* ) bli_malloc_user( sizeof( V_type ) * n * incx, &bli_errors ); \
	GEN_FUNC_NAME(fill_array_,V_type)( x, ( n * incx ) ); \
 \
	V_type* ref_x = ( V_type* ) bli_malloc_user( sizeof( V_type ) * n * incx, &bli_errors ); \
	GEN_FUNC_NAME(fill_array_,V_type)( ref_x, ( n * incx ) ); \
 \
	GEN_FUNC_NAME(gelu_bench_driver_,GELU_SFX)(n_repeats,n,x,incx); \
 \
	if ( bench_mode == 'a' ) \
	{ \
		GEN_FUNC_NAME(gelu_acc_check_,GELU_SFX)(fout,n,x,ref_x,incx); \
	} \
} \

GEN_GELU_BENCH_MAIN_FN(float,gelu_tanh_f32)
GEN_GELU_BENCH_MAIN_FN(float,gelu_erf_f32)

#define GEN_SOFTMAX_BENCH_MAIN_FN(V_type,SOFTMAX_SFX) \
void softmax_bench_main_ ## SOFTMAX_SFX \
    ( \
       FILE*   fout, \
       dim_t n, \
       inc_t incx \
     ) \
{ \
	int32_t n_repeats = 1000; \
	if ( global_n_repeat > 0 ) \
	{ \
		n_repeats = global_n_repeat; \
	} \
 \
	err_t bli_errors = BLIS_SUCCESS; \
	V_type* x = ( V_type* ) bli_malloc_user( sizeof( V_type ) * n * incx, &bli_errors ); \
	GEN_FUNC_NAME(fill_array_,V_type)( x, ( n * incx ) ); \
 \
	V_type* ref_x = ( V_type* ) bli_malloc_user( sizeof( V_type ) * n * incx, &bli_errors ); \
	GEN_FUNC_NAME(fill_array_,V_type)( ref_x, ( n * incx ) ); \
 \
	GEN_FUNC_NAME(softmax_bench_driver_,SOFTMAX_SFX)(n_repeats,n,x,incx); \
 \
	if ( bench_mode == 'a' ) \
	{ \
		GEN_FUNC_NAME(softmax_acc_check_,SOFTMAX_SFX)(fout,n,x,ref_x,incx); \
	} \
} \

GEN_SOFTMAX_BENCH_MAIN_FN(float,softmax_f32)

int main( int argc, char** argv )
{
	FILE* fin  = NULL;
	if ( argc < 5 )
	{
		printf( "Usage: ./bench_lpgemm_utils -i input.txt -m mode < -n 1000 >" \
						"\nMode is either a or p. a is used for accuracy test, " \
						"whereas p is used for performance benchmarking." \
						"\nn_repeats can be set optionally using -n arg.\n" );
		exit( 1 );
	}

	char* file_name = NULL;
	getopt_t state;
	// Initialize the state for running bli_getopt(). Here, 0 is the
	// initial value for opterr, which suppresses error messages.
	bli_getopt_init_state( 0, &state );

	int opt;
	// Process all option arguments until we get a -1, which means we're done.
	while( (opt = bli_getopt( argc, argv, "i:m:n:", &state )) != -1 )
	{
		char opt_ch = ( char )opt;
		switch( opt_ch )
		{
			case 'i':
					file_name = state.optarg;
					break;
			case 'm':
					bench_mode = ( ( ( *state.optarg ) == 'a' ) || ( ( *state.optarg ) == 'p' ) ) ? ( *state.optarg ) : 'p';
					break;
			case 'n':
					global_n_repeat = ( atoi( state.optarg ) > 0 ) ? atoi( state.optarg ) : 0;
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

	char l1_op_type[128];
	dim_t n;
	inc_t incx;
	while ( fscanf( fin, "%s %ld %ld\n", l1_op_type, &n, &incx )  == 3 )
	{
		if ( strcmp( l1_op_type, "f32_gelu_tanh" ) == 0 )
		{
			gelu_bench_main_gelu_tanh_f32( fout, n, incx );
		}
		else if ( strcmp( l1_op_type, "f32_gelu_erf" ) == 0 )
		{
			gelu_bench_main_gelu_erf_f32( fout, n, incx );
		}
		else if ( strcmp( l1_op_type, "f32_softmax" ) == 0 )
		{
			softmax_bench_main_softmax_f32( fout, n, incx );
		}
	}

	return 0;
}
