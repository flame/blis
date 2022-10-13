/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2014, The University of Texas at Austin
   Copyright (C) 2018 - 2022, Advanced Micro Devices, Inc.

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

#include "blis.h"
#include "test_libblis.h"

// Global variables.
char libblis_test_binary_name[ MAX_BINARY_NAME_LENGTH + 1 ];

char libblis_test_parameters_filename[ MAX_FILENAME_LENGTH + 1 ];
char libblis_test_operations_filename[ MAX_FILENAME_LENGTH + 1 ];

char libblis_test_pass_string[ MAX_PASS_STRING_LENGTH + 1 ];
char libblis_test_warn_string[ MAX_PASS_STRING_LENGTH + 1 ];
char libblis_test_fail_string[ MAX_PASS_STRING_LENGTH + 1 ];

char libblis_test_store_chars[ NUM_OPERAND_TYPES ][ MAX_STORE_VALS_PER_TYPE + 1 ];

char libblis_test_param_chars[ NUM_PARAM_TYPES ][ MAX_PARAM_VALS_PER_TYPE + 1 ];

char libblis_test_sp_chars[ 2 + 1 ] = "sc";
char libblis_test_dp_chars[ 2 + 1 ] = "dz";

char libblis_test_rd_chars[ 2 + 1 ] = "sd";
char libblis_test_cd_chars[ 2 + 1 ] = "cz";

char libblis_test_dt_chars[ 4 + 1 ] = "sdcz";


int main( int argc, char** argv )
{
	test_params_t params;
	test_ops_t    ops;

	// Initialize libblis.
	//bli_init();

	// Initialize some strings.
	libblis_test_init_strings();

	// Parse the command line parameters.
	libblis_test_parse_command_line( argc, argv );

	// Read the global parameters file.
	libblis_test_read_params_file( libblis_test_parameters_filename, &params );

	// Read the operations parameter file.
	libblis_test_read_ops_file( libblis_test_operations_filename, &ops );

	// Walk through all test modules.
	//libblis_test_all_ops( &params, &ops );
	libblis_test_thread_decorator( &params, &ops );

	// Finalize libblis.
	bli_finalize();

	// Return peacefully.
	return 0;
}


#if 0
typedef struct thread_data
{
	test_params_t*     params;
	test_ops_t*        ops;
	unsigned int       nt;
	unsigned int       id;
	unsigned int       xc;
	//pthread_mutex_t*   mutex;
	pthread_barrier_t* barrier;
} thread_data_t;
#endif

void* libblis_test_thread_entry( void* tdata_void )
{
	thread_data_t* tdata  = tdata_void;

	test_params_t* params = tdata->params;
	test_ops_t*    ops    = tdata->ops;

	// Walk through all test modules.
	libblis_test_all_ops( tdata, params, ops );

	return NULL;
}



void libblis_test_thread_decorator( test_params_t* params, test_ops_t* ops )
{
	// Query the total number of threads to simulate.
	size_t nt = ( size_t )params->n_app_threads;

	// Allocate an array of pthread objects and auxiliary data structs to pass
	// to the thread entry functions.

	#ifdef BLIS_ENABLE_MEM_TRACING
	printf( "libblis_test_thread_decorator(): " );
	#endif
	bli_pthread_t* pthread = bli_malloc_user( sizeof( bli_pthread_t ) * nt );

	#ifdef BLIS_ENABLE_MEM_TRACING
	printf( "libblis_test_thread_decorator(): " );
	#endif
	thread_data_t* tdata   = bli_malloc_user( sizeof( thread_data_t ) * nt );

	// Allocate a mutex for the threads to share.
	//bli_pthread_mutex_t* mutex   = bli_malloc_user( sizeof( bli_pthread_mutex_t ) );

	// Allocate a barrier for the threads to share.

	#ifdef BLIS_ENABLE_MEM_TRACING
	printf( "libblis_test_thread_decorator(): " );
	#endif
	bli_pthread_barrier_t* barrier = bli_malloc_user( sizeof( bli_pthread_barrier_t ) );

	// Initialize the mutex.
	//bli_pthread_mutex_init( mutex, NULL );

	// Initialize the barrier for nt threads.
	bli_pthread_barrier_init( barrier, NULL, nt );

	// NOTE: We must iterate backwards so that the chief thread (thread id 0)
	// can spawn all other threads before proceeding with its own computation.
	// ALSO: Since we need to let the counter go negative, id must be a signed
	// integer here.
	for ( signed int id = nt - 1; 0 <= id; id-- )
	{
		tdata[id].params  = params;
		tdata[id].ops     = ops;
		tdata[id].nt      = nt;
		tdata[id].id      = id;
		tdata[id].xc      = 0;
		//tdata[id].mutex   = mutex;
		tdata[id].barrier = barrier;

		// Spawn additional threads for ids greater than 1.
		if ( id != 0 )
			bli_pthread_create( &pthread[id], NULL, libblis_test_thread_entry, &tdata[id] );
		else
			libblis_test_thread_entry( ( void* )(&tdata[0]) );
	}

	// Thread 0 waits for additional threads to finish.
	for ( unsigned int id = 1; id < nt; id++ )
	{
		bli_pthread_join( pthread[id], NULL );
	}

	// Destroy the mutex.
	//bli_pthread_mutex_destroy( mutex );

	// Destroy the barrier.
	bli_pthread_barrier_destroy( barrier );

	// Free the pthread-related memory.

	#ifdef BLIS_ENABLE_MEM_TRACING
	printf( "libblis_test_thread_decorator(): " );
	#endif
	bli_free_user( pthread );

	#ifdef BLIS_ENABLE_MEM_TRACING
	printf( "libblis_test_thread_decorator(): " );
	#endif
	bli_free_user( tdata );

	#ifdef BLIS_ENABLE_MEM_TRACING
	printf( "libblis_test_thread_decorator(): " );
	#endif
	//bli_free_user( mutex );
	bli_free_user( barrier );
}



void libblis_test_all_ops( thread_data_t* tdata, test_params_t* params, test_ops_t* ops )
{
	// Test the utility operations.
	libblis_test_utility_ops( tdata, params, ops );

	// Test the level-1v operations.
	libblis_test_level1v_ops( tdata, params, ops );

	// Test the level-1m operations.
	libblis_test_level1m_ops( tdata, params, ops );

	// Test the level-1f operations.
	libblis_test_level1f_ops( tdata, params, ops );

	// Test the level-2 operations.
	libblis_test_level2_ops( tdata, params, ops );

	// Test the level-3 micro-kernels.
	libblis_test_level3_ukrs( tdata, params, ops );

	// Test the level-3 operations.
	libblis_test_level3_ops( tdata, params, ops );
}



void libblis_test_utility_ops( thread_data_t* tdata, test_params_t* params, test_ops_t* ops )
{
	libblis_test_randv( tdata, params, &(ops->randv) );
	libblis_test_randm( tdata, params, &(ops->randm) );
}



void libblis_test_level1v_ops( thread_data_t* tdata, test_params_t* params, test_ops_t* ops )
{
	libblis_test_addv( tdata, params, &(ops->addv) );
	libblis_test_amaxv( tdata, params, &(ops->amaxv) );
	libblis_test_axpbyv( tdata, params, &(ops->axpbyv) );
	libblis_test_axpyv( tdata, params, &(ops->axpyv) );
	libblis_test_copyv( tdata, params, &(ops->copyv) );
	libblis_test_dotv( tdata, params, &(ops->dotv) );
	libblis_test_dotxv( tdata, params, &(ops->dotxv) );
	libblis_test_normfv( tdata, params, &(ops->normfv) );
	libblis_test_scalv( tdata, params, &(ops->scalv) );
	libblis_test_scal2v( tdata, params, &(ops->scal2v) );
	libblis_test_setv( tdata, params, &(ops->setv) );
	libblis_test_subv( tdata, params, &(ops->subv) );
	libblis_test_xpbyv( tdata, params, &(ops->xpbyv) );
}



void libblis_test_level1m_ops( thread_data_t* tdata, test_params_t* params, test_ops_t* ops )
{
	libblis_test_addm( tdata, params, &(ops->addm) );
	libblis_test_axpym( tdata, params, &(ops->axpym) );
	libblis_test_copym( tdata, params, &(ops->copym) );
	libblis_test_normfm( tdata, params, &(ops->normfm) );
	libblis_test_scalm( tdata, params, &(ops->scalm) );
	libblis_test_scal2m( tdata, params, &(ops->scal2m) );
	libblis_test_setm( tdata, params, &(ops->setm) );
	libblis_test_subm( tdata, params, &(ops->subm) );
	libblis_test_xpbym( tdata, params, &(ops->xpbym) );
}



void libblis_test_level1f_ops( thread_data_t* tdata, test_params_t* params, test_ops_t* ops )
{
	libblis_test_axpy2v( tdata, params, &(ops->axpy2v) );
	libblis_test_dotaxpyv( tdata, params, &(ops->dotaxpyv) );
	libblis_test_axpyf( tdata, params, &(ops->axpyf) );
	libblis_test_dotxf( tdata, params, &(ops->dotxf) );
	libblis_test_dotxaxpyf( tdata, params, &(ops->dotxaxpyf) );
}



void libblis_test_level2_ops( thread_data_t* tdata, test_params_t* params, test_ops_t* ops )
{
	libblis_test_gemv( tdata, params, &(ops->gemv) );
	libblis_test_ger( tdata, params, &(ops->ger) );
	libblis_test_hemv( tdata, params, &(ops->hemv) );
	libblis_test_her( tdata, params, &(ops->her) );
	libblis_test_her2( tdata, params, &(ops->her2) );
	libblis_test_symv( tdata, params, &(ops->symv) );
	libblis_test_syr( tdata, params, &(ops->syr) );
	libblis_test_syr2( tdata, params, &(ops->syr2) );
	libblis_test_trmv( tdata, params, &(ops->trmv) );
	libblis_test_trsv( tdata, params, &(ops->trsv) );
}



void libblis_test_level3_ukrs( thread_data_t* tdata, test_params_t* params, test_ops_t* ops )
{
	libblis_test_gemm_ukr( tdata, params, &(ops->gemm_ukr) );
	libblis_test_trsm_ukr( tdata, params, &(ops->trsm_ukr) );
	libblis_test_gemmtrsm_ukr( tdata, params, &(ops->gemmtrsm_ukr) );
}



void libblis_test_level3_ops( thread_data_t* tdata, test_params_t* params, test_ops_t* ops )
{
	libblis_test_gemm( tdata, params, &(ops->gemm) );
	libblis_test_gemmt( tdata, params, &(ops->gemmt) );
	libblis_test_hemm( tdata, params, &(ops->hemm) );
	libblis_test_herk( tdata, params, &(ops->herk) );
	libblis_test_her2k( tdata, params, &(ops->her2k) );
	libblis_test_symm( tdata, params, &(ops->symm) );
	libblis_test_syrk( tdata, params, &(ops->syrk) );
	libblis_test_syr2k( tdata, params, &(ops->syr2k) );
	libblis_test_trmm( tdata, params, &(ops->trmm) );
	libblis_test_trmm3( tdata, params, &(ops->trmm3) );
	libblis_test_trsm( tdata, params, &(ops->trsm) );
}



void libblis_test_read_ops_file( char* input_filename, test_ops_t* ops )
{
	FILE* input_stream;

	// Attempt to open input file corresponding to input_filename as
	// read-only/binary.
	input_stream = fopen( input_filename, "r" );
	libblis_test_fopen_check_stream( input_filename, input_stream );

	// Initialize the individual override field to FALSE.
	ops->indiv_over = FALSE;

	// Begin reading operations input file.

	// Section overrides
	libblis_test_read_section_override( ops, input_stream, &(ops->util_over) );
	libblis_test_read_section_override( ops, input_stream, &(ops->l1v_over) );
	libblis_test_read_section_override( ops, input_stream, &(ops->l1m_over) );
	libblis_test_read_section_override( ops, input_stream, &(ops->l1f_over) );
	libblis_test_read_section_override( ops, input_stream, &(ops->l2_over) );
	libblis_test_read_section_override( ops, input_stream, &(ops->l3ukr_over) );
	libblis_test_read_section_override( ops, input_stream, &(ops->l3_over) );

	//                                            dimensions          n_param   operation

	// Utility operations
	libblis_test_read_op_info( ops, input_stream, BLIS_NOID, BLIS_TEST_DIMS_M,   0, &(ops->randv) );
	libblis_test_read_op_info( ops, input_stream, BLIS_NOID, BLIS_TEST_DIMS_MN,  0, &(ops->randm) );

	// Level-1v
	libblis_test_read_op_info( ops, input_stream, BLIS_NOID, BLIS_TEST_DIMS_M,   1, &(ops->addv) );
	libblis_test_read_op_info( ops, input_stream, BLIS_NOID, BLIS_TEST_DIMS_M,   0, &(ops->amaxv) );
	libblis_test_read_op_info( ops, input_stream, BLIS_NOID, BLIS_TEST_DIMS_M,   1, &(ops->axpbyv) );
	libblis_test_read_op_info( ops, input_stream, BLIS_NOID, BLIS_TEST_DIMS_M,   1, &(ops->axpyv) );
	libblis_test_read_op_info( ops, input_stream, BLIS_NOID, BLIS_TEST_DIMS_M,   1, &(ops->copyv) );
	libblis_test_read_op_info( ops, input_stream, BLIS_NOID, BLIS_TEST_DIMS_M,   2, &(ops->dotv) );
	libblis_test_read_op_info( ops, input_stream, BLIS_NOID, BLIS_TEST_DIMS_M,   2, &(ops->dotxv) );
	libblis_test_read_op_info( ops, input_stream, BLIS_NOID, BLIS_TEST_DIMS_M,   0, &(ops->normfv) );
	libblis_test_read_op_info( ops, input_stream, BLIS_NOID, BLIS_TEST_DIMS_M,   1, &(ops->scalv) );
	libblis_test_read_op_info( ops, input_stream, BLIS_NOID, BLIS_TEST_DIMS_M,   1, &(ops->scal2v) );
	libblis_test_read_op_info( ops, input_stream, BLIS_NOID, BLIS_TEST_DIMS_M,   0, &(ops->setv) );
	libblis_test_read_op_info( ops, input_stream, BLIS_NOID, BLIS_TEST_DIMS_M,   1, &(ops->subv) );
	libblis_test_read_op_info( ops, input_stream, BLIS_NOID, BLIS_TEST_DIMS_M,   1, &(ops->xpbyv) );

	// Level-1m
	libblis_test_read_op_info( ops, input_stream, BLIS_NOID, BLIS_TEST_DIMS_MN,  1, &(ops->addm) );
	libblis_test_read_op_info( ops, input_stream, BLIS_NOID, BLIS_TEST_DIMS_MN,  1, &(ops->axpym) );
	libblis_test_read_op_info( ops, input_stream, BLIS_NOID, BLIS_TEST_DIMS_MN,  1, &(ops->copym) );
	libblis_test_read_op_info( ops, input_stream, BLIS_NOID, BLIS_TEST_DIMS_MN,  0, &(ops->normfm) );
	libblis_test_read_op_info( ops, input_stream, BLIS_NOID, BLIS_TEST_DIMS_MN,  1, &(ops->scalm) );
	libblis_test_read_op_info( ops, input_stream, BLIS_NOID, BLIS_TEST_DIMS_MN,  1, &(ops->scal2m) );
	libblis_test_read_op_info( ops, input_stream, BLIS_NOID, BLIS_TEST_DIMS_MN,  0, &(ops->setm) );
	libblis_test_read_op_info( ops, input_stream, BLIS_NOID, BLIS_TEST_DIMS_MN,  1, &(ops->subm) );
	libblis_test_read_op_info( ops, input_stream, BLIS_NOID, BLIS_TEST_DIMS_MN,  1, &(ops->xpbym) );

	// Level-1f
	libblis_test_read_op_info( ops, input_stream, BLIS_NOID, BLIS_TEST_DIMS_M,   2, &(ops->axpy2v) );
	libblis_test_read_op_info( ops, input_stream, BLIS_NOID, BLIS_TEST_DIMS_M,   3, &(ops->dotaxpyv) );
	libblis_test_read_op_info( ops, input_stream, BLIS_NOID, BLIS_TEST_DIMS_MF,  2, &(ops->axpyf) );
	libblis_test_read_op_info( ops, input_stream, BLIS_NOID, BLIS_TEST_DIMS_MF,  2, &(ops->dotxf) );
	libblis_test_read_op_info( ops, input_stream, BLIS_NOID, BLIS_TEST_DIMS_MF,  4, &(ops->dotxaxpyf) );

	// Level-2
	libblis_test_read_op_info( ops, input_stream, BLIS_NOID, BLIS_TEST_DIMS_MN,  2, &(ops->gemv) );
	libblis_test_read_op_info( ops, input_stream, BLIS_NOID, BLIS_TEST_DIMS_MN,  2, &(ops->ger) );
	libblis_test_read_op_info( ops, input_stream, BLIS_NOID, BLIS_TEST_DIMS_M,   3, &(ops->hemv) );
	libblis_test_read_op_info( ops, input_stream, BLIS_NOID, BLIS_TEST_DIMS_M,   2, &(ops->her) );
	libblis_test_read_op_info( ops, input_stream, BLIS_NOID, BLIS_TEST_DIMS_M,   3, &(ops->her2) );
	libblis_test_read_op_info( ops, input_stream, BLIS_NOID, BLIS_TEST_DIMS_M,   3, &(ops->symv) );
	libblis_test_read_op_info( ops, input_stream, BLIS_NOID, BLIS_TEST_DIMS_M,   2, &(ops->syr) );
	libblis_test_read_op_info( ops, input_stream, BLIS_NOID, BLIS_TEST_DIMS_M,   3, &(ops->syr2) );
	libblis_test_read_op_info( ops, input_stream, BLIS_NOID, BLIS_TEST_DIMS_M,   3, &(ops->trmv) );
	libblis_test_read_op_info( ops, input_stream, BLIS_NOID, BLIS_TEST_DIMS_M,   3, &(ops->trsv) );

	// Level-3 micro-kernels
	libblis_test_read_op_info( ops, input_stream, BLIS_NOID, BLIS_TEST_DIMS_K,   0, &(ops->gemm_ukr) );
	libblis_test_read_op_info( ops, input_stream, BLIS_NOID, BLIS_TEST_NO_DIMS,  1, &(ops->trsm_ukr) );
	libblis_test_read_op_info( ops, input_stream, BLIS_NOID, BLIS_TEST_DIMS_K,   1, &(ops->gemmtrsm_ukr) );

	// Level-3
	libblis_test_read_op_info( ops, input_stream, BLIS_GEMM,  BLIS_TEST_DIMS_MNK, 2, &(ops->gemm) );
	libblis_test_read_op_info( ops, input_stream, BLIS_GEMMT, BLIS_TEST_DIMS_MK,  3, &(ops->gemmt) );
	libblis_test_read_op_info( ops, input_stream, BLIS_HEMM,  BLIS_TEST_DIMS_MN,  4, &(ops->hemm) );
	libblis_test_read_op_info( ops, input_stream, BLIS_HERK,  BLIS_TEST_DIMS_MK,  2, &(ops->herk) );
	libblis_test_read_op_info( ops, input_stream, BLIS_HER2K, BLIS_TEST_DIMS_MK,  3, &(ops->her2k) );
	libblis_test_read_op_info( ops, input_stream, BLIS_SYMM,  BLIS_TEST_DIMS_MN,  4, &(ops->symm) );
	libblis_test_read_op_info( ops, input_stream, BLIS_SYRK,  BLIS_TEST_DIMS_MK,  2, &(ops->syrk) );
	libblis_test_read_op_info( ops, input_stream, BLIS_SYR2K, BLIS_TEST_DIMS_MK,  3, &(ops->syr2k) );
	libblis_test_read_op_info( ops, input_stream, BLIS_TRMM,  BLIS_TEST_DIMS_MN,  4, &(ops->trmm) );
	libblis_test_read_op_info( ops, input_stream, BLIS_TRMM3, BLIS_TEST_DIMS_MN,  5, &(ops->trmm3) );
	libblis_test_read_op_info( ops, input_stream, BLIS_TRSM,  BLIS_TEST_DIMS_MN,  4, &(ops->trsm) );

	// Output the section overrides.
	libblis_test_output_section_overrides( stdout, ops );

	// Close the file.
	fclose( input_stream );

}



void libblis_test_read_params_file( char* input_filename, test_params_t* params )
{
	FILE* input_stream;
	char  buffer[ INPUT_BUFFER_SIZE ];
	char  temp[ INPUT_BUFFER_SIZE ];
	int   i;

	// Attempt to open input file corresponding to input_filename as
	// read-only/binary.
	input_stream = fopen( input_filename, "r" );
	libblis_test_fopen_check_stream( input_filename, input_stream );

	// Read the number of repeats.
	libblis_test_read_next_line( buffer, input_stream );
	sscanf( buffer, "%u ", &(params->n_repeats) );

	// Read the matrix storage schemes to test. We should have at most three:
	// 'r' for row-major, 'c' for column-major, and 'g' for general strides.
	libblis_test_read_next_line( buffer, input_stream );
	sscanf( buffer, "%s ", temp );

	params->n_mstorage = strlen( temp );
	if ( params->n_mstorage > MAX_NUM_MSTORAGE )
	{
		libblis_test_printf_error( "Detected too many matrix storage schemes (%u) in input file.\n",
		                           params->n_mstorage );
	}
	strcpy( params->storage[ BLIS_TEST_MATRIX_OPERAND ], temp );

	// Read the vector storage schemes to test. We should have at most four:
	// 'r' for row vectors with unit stride, 'c' for column vectors with unit
	// stride, 'i' for row vectors with non-unit stride, and 'j' for column
	// vectors with non-unit stride.
	libblis_test_read_next_line( buffer, input_stream );
	sscanf( buffer, "%s ", temp );

	params->n_vstorage = strlen( temp );
	if ( params->n_vstorage > MAX_NUM_VSTORAGE )
	{
		libblis_test_printf_error( "Detected too many vector storage schemes (%u) in input file.\n",
		                           params->n_vstorage );
	}
	strcpy( params->storage[ BLIS_TEST_VECTOR_OPERAND ], temp );

	// Read whether to mix all storage combinations.
	libblis_test_read_next_line( buffer, input_stream );
	sscanf( buffer, "%u ", &(params->mix_all_storage) );

	// Read whether to perform all tests with aligned addresses and ldims.
	libblis_test_read_next_line( buffer, input_stream );
	sscanf( buffer, "%u ", &(params->alignment) );

	// Read the randomization method.
	libblis_test_read_next_line( buffer, input_stream );
	sscanf( buffer, "%u ", &(params->rand_method) );

	if ( params->rand_method != BLIS_TEST_RAND_REAL_VALUES &&
	     params->rand_method != BLIS_TEST_RAND_NARROW_POW2 )
	{
		libblis_test_printf_error( "Invalid randomization method (%u) in input file.\n",
		                           params->rand_method );
	}

	// Read the general stride "spacing".
	libblis_test_read_next_line( buffer, input_stream );
	sscanf( buffer, "%u ", &(params->gs_spacing) );

	// Overwrite the existing storage character arrays with the sets provided.
	strcpy( libblis_test_store_chars[BLIS_TEST_MATRIX_OPERAND],
	                 params->storage[BLIS_TEST_MATRIX_OPERAND] );
	strcpy( libblis_test_store_chars[BLIS_TEST_VECTOR_OPERAND],
	                 params->storage[BLIS_TEST_VECTOR_OPERAND] );

	// Read the datatypes to test. We should have at most four: 's', 'd', 'c',
	// and 'z'.
	libblis_test_read_next_line( buffer, input_stream );
	sscanf( buffer, "%s ", temp );

	params->n_datatypes = strlen( temp );
	if ( params->n_datatypes > MAX_NUM_DATATYPES )
	{
		libblis_test_printf_error( "Detected too many datatype requests (%u) in input file.\n",
		                           params->n_datatypes );
	}

	for( i = 0; i < params->n_datatypes; ++i )
	{
		//if      ( temp[i] == 's' ) params->datatype[i] = BLIS_FLOAT;
		//else if ( temp[i] == 'd' ) params->datatype[i] = BLIS_DOUBLE;
		//else if ( temp[i] == 'c' ) params->datatype[i] = BLIS_SCOMPLEX;
		//else if ( temp[i] == 'z' ) params->datatype[i] = BLIS_DCOMPLEX;

		// Map the char in temp[i] to the corresponding num_t value.
		bli_param_map_char_to_blis_dt( temp[i], &(params->datatype[i]) );

		params->datatype_char[i] = temp[i];
	}

	// Read whether to test gemm with mixed-domain operands.
	libblis_test_read_next_line( buffer, input_stream );
	sscanf( buffer, "%u ", &(params->mixed_domain) );

	// Read whether to test gemm with mixed-precision operands.
	libblis_test_read_next_line( buffer, input_stream );
	sscanf( buffer, "%u ", &(params->mixed_precision) );

	// Read the initial problem size to test.
	libblis_test_read_next_line( buffer, input_stream );
	sscanf( buffer, "%u ", &(params->p_first) );

	// Read the maximum problem size to test.
	libblis_test_read_next_line( buffer, input_stream );
	sscanf( buffer, "%u ", &(params->p_max) );

	// Read the problem size increment to test.
	libblis_test_read_next_line( buffer, input_stream );
	sscanf( buffer, "%u ", &(params->p_inc) );

	// Read whether to enable 3mh.
	libblis_test_read_next_line( buffer, input_stream );
	sscanf( buffer, "%u ", &(params->ind_enable[ BLIS_3MH ]) );

	// Read whether to enable 3m1.
	libblis_test_read_next_line( buffer, input_stream );
	sscanf( buffer, "%u ", &(params->ind_enable[ BLIS_3M1 ]) );

	// Read whether to enable 4mh.
	libblis_test_read_next_line( buffer, input_stream );
	sscanf( buffer, "%u ", &(params->ind_enable[ BLIS_4MH ]) );

	// Read whether to enable 4m1b (4mb).
	libblis_test_read_next_line( buffer, input_stream );
	sscanf( buffer, "%u ", &(params->ind_enable[ BLIS_4M1B ]) );

	// Read whether to enable 4m1a (4m1).
	libblis_test_read_next_line( buffer, input_stream );
	sscanf( buffer, "%u ", &(params->ind_enable[ BLIS_4M1A ]) );

	// Read whether to enable 1m.
	libblis_test_read_next_line( buffer, input_stream );
	sscanf( buffer, "%u ", &(params->ind_enable[ BLIS_1M ]) );

	// Read whether to native (complex) execution.
	libblis_test_read_next_line( buffer, input_stream );
	sscanf( buffer, "%u ", &(params->ind_enable[ BLIS_NAT ]) );

	// Read whether to simulate application-level threading.
	libblis_test_read_next_line( buffer, input_stream );
	sscanf( buffer, "%u ", &(params->n_app_threads) );

	// Silently interpret non-positive numbers the same as 1.
	if ( params->n_app_threads < 1 ) params->n_app_threads = 1;

	// Disable induced methods when simulating more than one application
	// threads.
	if ( params->n_app_threads > 1 )
	{
		if ( params->ind_enable[ BLIS_3MH  ] ||
		     params->ind_enable[ BLIS_3M1  ] ||
		     params->ind_enable[ BLIS_4MH  ] ||
		     params->ind_enable[ BLIS_4M1B ] ||
		     params->ind_enable[ BLIS_4M1A ] ||
		     params->ind_enable[ BLIS_1M   ]
		   )
		{
			// Due to an inherent race condition in the way induced methods
			// are enabled and disabled at runtime, all induced methods must be
			// disabled when simulating multiple application threads.
			libblis_test_printf_infoc( "simulating multiple application threads; disabling induced methods.\n" );

			params->ind_enable[ BLIS_3MH  ] = 0;
			params->ind_enable[ BLIS_3M1  ] = 0;
			params->ind_enable[ BLIS_4MH  ] = 0;
			params->ind_enable[ BLIS_4M1B ] = 0;
			params->ind_enable[ BLIS_4M1A ] = 0;
			params->ind_enable[ BLIS_1M   ] = 0;
		}
	}

	// Read the requested error-checking level.
	libblis_test_read_next_line( buffer, input_stream );
	sscanf( buffer, "%u ", &(params->error_checking_level) );

	// Read the requested course of action if a test fails.
	libblis_test_read_next_line( buffer, input_stream );
	sscanf( buffer, "%c ", &(params->reaction_to_failure) );

	if ( params->reaction_to_failure != ON_FAILURE_IGNORE_CHAR &&
	     params->reaction_to_failure != ON_FAILURE_SLEEP_CHAR  &&
	     params->reaction_to_failure != ON_FAILURE_ABORT_CHAR  )
	{
		libblis_test_printf_error( "Invalid reaction-to-failure character code (%c) in input file.\n",
		                           params->reaction_to_failure );
	}

	// Read whether to output in matlab format.
	libblis_test_read_next_line( buffer, input_stream );
	sscanf( buffer, "%u ", &(params->output_matlab_format) );

	// Read whether to output to files in addition to stdout.
	libblis_test_read_next_line( buffer, input_stream );
	sscanf( buffer, "%u ", &(params->output_files) );

	// Close the file.
	fclose( input_stream );

	// Output the parameter struct.
	libblis_test_output_params_struct( stdout, params );
}


void libblis_test_read_section_override( test_ops_t*  ops,
                                         FILE*        input_stream,
                                         int*         override )
{
	char  buffer[ INPUT_BUFFER_SIZE ];

	// Read the line for the section override switch.
	libblis_test_read_next_line( buffer, input_stream );
	sscanf( buffer, "%d ", override );
}


void libblis_test_read_op_info( test_ops_t*  ops,
                                FILE*        input_stream,
                                opid_t       opid,
                                dimset_t     dimset,
                                unsigned int n_params,
                                test_op_t*   op )
{
	char  buffer[ INPUT_BUFFER_SIZE ];
	char  temp[ INPUT_BUFFER_SIZE ];
	int   i, p;

	// Initialize the operation type field.
	op->opid = opid; 

	// Read the line for the overall operation switch.
	libblis_test_read_next_line( buffer, input_stream );
	sscanf( buffer, "%d ", &(op->op_switch) );

	// Check the op_switch for the individual override value.
	if ( op->op_switch == ENABLE_ONLY )
	{
		ops->indiv_over = TRUE;
	}

	op->n_dims = libblis_test_get_n_dims_from_dimset( dimset );
	op->dimset = dimset;

	if ( op->n_dims > MAX_NUM_DIMENSIONS )
	{
		libblis_test_printf_error( "Detected too many dimensions (%u) in input file to store.\n",
		                           op->n_dims );
	}

	//printf( "n_dims = %u\n", op->n_dims );

	// If there is at least one dimension for the current operation, read the
	// dimension specifications, which encode the actual dimensions or the
	// dimension ratios for each dimension.
	if ( op->n_dims > 0 )
	{
		libblis_test_read_next_line( buffer, input_stream );

		for ( i = 0, p = 0; i < op->n_dims; ++i )
		{
			//printf( "buffer[p]:       %s\n", &buffer[p] );

			// Advance until we hit non-whitespace (ie: the next number).
			for ( ; isspace( buffer[p] ); ++p ) ; 

			//printf( "buffer[p] after: %s\n", &buffer[p] );

			sscanf( &buffer[p], "%d", &(op->dim_spec[i]) );

			//printf( "dim[%d] = %d\n", i, op->dim_spec[i] );

			// Advance until we hit whitespace (ie: the space before the next number).
			for ( ; !isspace( buffer[p] ); ++p ) ; 
		}
	}

	// If there is at least one parameter for the current operation, read the
	// parameter chars, which encode which parameter combinations to test.
	if ( n_params > 0 )
	{
		libblis_test_read_next_line( buffer, input_stream );
		sscanf( buffer, "%s ", temp );

		op->n_params = strlen( temp );
		if ( op->n_params > MAX_NUM_PARAMETERS )
		{
			libblis_test_printf_error( "Detected too many parameters (%u) in input file.\n",
			                           op->n_params );
		}
		if ( op->n_params != n_params )
		{
			libblis_test_printf_error( "Number of parameters specified by caller does not match length of parameter string in input file. strlen( temp ) = %u; n_params = %u\n", op->n_params, n_params );
		}

		strcpy( op->params, temp );
	}
	else
	{
		op->n_params = 0;
		strcpy( op->params, "" );
	}

	// Initialize the "test done" switch.
	op->test_done = FALSE;

	// Initialize the parent pointer.
	op->ops = ops;
}


void libblis_test_output_section_overrides( FILE* os, test_ops_t* ops )
{
	libblis_test_fprintf_c( os, "\n" );
	libblis_test_fprintf_c( os, "--- Section overrides ---\n" );
	libblis_test_fprintf_c( os, "\n" );
	libblis_test_fprintf_c( os, "Utility operations           %d\n", ops->util_over );
	libblis_test_fprintf_c( os, "Level-1v operations          %d\n", ops->l1v_over );
	libblis_test_fprintf_c( os, "Level-1m operations          %d\n", ops->l1m_over );
	libblis_test_fprintf_c( os, "Level-1f operations          %d\n", ops->l1f_over );
	libblis_test_fprintf_c( os, "Level-2 operations           %d\n", ops->l2_over );
	libblis_test_fprintf_c( os, "Level-3 micro-kernels        %d\n", ops->l3ukr_over );
	libblis_test_fprintf_c( os, "Level-3 operations           %d\n", ops->l3_over );
	libblis_test_fprintf_c( os, "\n" );
	libblis_test_fprintf( os, "\n" );
}



void libblis_test_output_params_struct( FILE* os, test_params_t* params )
{
	int     i;
	//char   int_type_size_str[8];
	gint_t  int_type_size;
	ind_t   im;
	cntx_t* cntx;
	cntx_t* cntx_c;
	cntx_t* cntx_z;

	// If bli_info_get_int_type_size() returns 32 or 64, the size is forced.
	// Otherwise, the size is chosen automatically. We query the result of
	// that automatic choice via sizeof(gint_t).
	if ( bli_info_get_int_type_size() == 32 ||
	     bli_info_get_int_type_size() == 64 )
		int_type_size = bli_info_get_int_type_size();
	else
		int_type_size = sizeof(gint_t) * 8;

	char impl_str[16];
	char jrir_str[16];

	// Describe the threading implementation.
	if      ( bli_info_get_enable_openmp()   ) sprintf( impl_str, "openmp" );
	else if ( bli_info_get_enable_pthreads() ) sprintf( impl_str, "pthreads" );
	else    /* threading disabled */           sprintf( impl_str, "disabled" );

	// Describe the status of jrir thread partitioning.
	if   ( bli_info_get_thread_part_jrir_slab() ) sprintf( jrir_str, "slab" );
	else /*bli_info_get_thread_part_jrir_rr()*/   sprintf( jrir_str, "round-robin" );

	char nt_str[16];
	char jc_nt_str[16];
	char pc_nt_str[16];
	char ic_nt_str[16];
	char jr_nt_str[16];
	char ir_nt_str[16];

	// Query the number of ways of parallelism per loop (and overall) and
	// convert these values into strings, with "unset" being used if the
	// value returned was -1 (indicating the environment variable was unset).
	dim_t nt    = bli_thread_get_num_threads();
	dim_t jc_nt = bli_thread_get_jc_nt(); 
	dim_t pc_nt = bli_thread_get_pc_nt(); 
	dim_t ic_nt = bli_thread_get_ic_nt(); 
	dim_t jr_nt = bli_thread_get_jr_nt(); 
	dim_t ir_nt = bli_thread_get_ir_nt(); 

	if (    nt == -1 ) sprintf(    nt_str, "unset" );
	else               sprintf(    nt_str, "%d", ( int )   nt );
	if ( jc_nt == -1 ) sprintf( jc_nt_str, "unset" );
	else               sprintf( jc_nt_str, "%d", ( int )jc_nt );
	if ( pc_nt == -1 ) sprintf( pc_nt_str, "unset" );
	else               sprintf( pc_nt_str, "%d", ( int )pc_nt );
	if ( ic_nt == -1 ) sprintf( ic_nt_str, "unset" );
	else               sprintf( ic_nt_str, "%d", ( int )ic_nt );
	if ( jr_nt == -1 ) sprintf( jr_nt_str, "unset" );
	else               sprintf( jr_nt_str, "%d", ( int )jr_nt );
	if ( ir_nt == -1 ) sprintf( ir_nt_str, "unset" );
	else               sprintf( ir_nt_str, "%d", ( int )ir_nt );

	// Set up rntm_t objects for each of the four families:
	// gemm, herk, trmm, trsm.
	rntm_t gemm, herk, trmm_l, trmm_r, trsm_l, trsm_r;
	dim_t  m = 1000, n = 1000, k = 1000;

	bli_rntm_init_from_global( &gemm   );
	bli_rntm_init_from_global( &herk   );
	bli_rntm_init_from_global( &trmm_l );
	bli_rntm_init_from_global( &trmm_r );
	bli_rntm_init_from_global( &trsm_l );
	bli_rntm_init_from_global( &trsm_r );

	bli_rntm_set_ways_for_op( BLIS_GEMM, BLIS_LEFT,  m, n, k, &gemm );
	bli_rntm_set_ways_for_op( BLIS_HERK, BLIS_LEFT,  m, n, k, &herk );
	bli_rntm_set_ways_for_op( BLIS_TRMM, BLIS_LEFT,  m, n, k, &trmm_l );
	bli_rntm_set_ways_for_op( BLIS_TRMM, BLIS_RIGHT, m, n, k, &trmm_r );
	bli_rntm_set_ways_for_op( BLIS_TRSM, BLIS_LEFT,  m, n, k, &trsm_l );
	bli_rntm_set_ways_for_op( BLIS_TRSM, BLIS_RIGHT, m, n, k, &trsm_r );

	// Output some system parameters.
	libblis_test_fprintf_c( os, "\n" );
	libblis_test_fprintf_c( os, "--- BLIS library info -------------------------------------\n" );
	libblis_test_fprintf_c( os, "\n" );
	libblis_test_fprintf_c( os, "version string                 %s\n", bli_info_get_version_str() );
	libblis_test_fprintf_c( os, "\n" );
	libblis_test_fprintf_c( os, "--- BLIS configuration info ---\n" );
	libblis_test_fprintf_c( os, "\n" );
	libblis_test_fprintf_c( os, "active sub-configuration       %s\n", bli_arch_string( bli_arch_query_id() ) );
	libblis_test_fprintf_c( os, "\n" );
	libblis_test_fprintf_c( os, "BLIS integer type size (bits)  %d\n", ( int )int_type_size );
	libblis_test_fprintf_c( os, "\n" );
	libblis_test_fprintf_c( os, "Assumed max # of SIMD regs     %d\n", ( int )bli_info_get_simd_num_registers() );
	libblis_test_fprintf_c( os, "SIMD size (bytes)              %d\n", ( int )bli_info_get_simd_size() );
	libblis_test_fprintf_c( os, "SIMD alignment (bytes)         %d\n", ( int )bli_info_get_simd_align_size() );
	libblis_test_fprintf_c( os, "Max stack buffer size (bytes)  %d\n", ( int )bli_info_get_stack_buf_max_size() );
	libblis_test_fprintf_c( os, "Page size (bytes)              %d\n", ( int )bli_info_get_page_size() );
	libblis_test_fprintf_c( os, "\n" );
	libblis_test_fprintf_c( os, "memory pools\n" );
	libblis_test_fprintf_c( os, "  enabled for packing blocks?  %d\n", ( int )bli_info_get_enable_pba_pools() );
	libblis_test_fprintf_c( os, "  enabled for small blocks?    %d\n", ( int )bli_info_get_enable_sba_pools() );
	libblis_test_fprintf_c( os, "\n" );
	libblis_test_fprintf_c( os, "memory alignment (bytes)         \n" );
	libblis_test_fprintf_c( os, "  stack address                %d\n", ( int )bli_info_get_stack_buf_align_size() );
	libblis_test_fprintf_c( os, "  obj_t address                %d\n", ( int )bli_info_get_heap_addr_align_size() );
	libblis_test_fprintf_c( os, "  obj_t stride                 %d\n", ( int )bli_info_get_heap_stride_align_size() );
	libblis_test_fprintf_c( os, "  pool block addr A (+offset)  %d (+%d)\n", ( int )bli_info_get_pool_addr_align_size_a(), ( int )bli_info_get_pool_addr_offset_size_a() );
	libblis_test_fprintf_c( os, "  pool block addr B (+offset)  %d (+%d)\n", ( int )bli_info_get_pool_addr_align_size_b(), ( int )bli_info_get_pool_addr_offset_size_b() );
	libblis_test_fprintf_c( os, "\n" );
	libblis_test_fprintf_c( os, "BLAS/CBLAS compatibility layers  \n" );
	libblis_test_fprintf_c( os, "  BLAS API enabled?            %d\n", ( int )bli_info_get_enable_blas() );
	libblis_test_fprintf_c( os, "  CBLAS API enabled?           %d\n", ( int )bli_info_get_enable_cblas() );
	libblis_test_fprintf_c( os, "  integer type size (bits)     %d\n", ( int )bli_info_get_blas_int_type_size() );
	libblis_test_fprintf_c( os, "\n" );
	libblis_test_fprintf_c( os, "libmemkind                       \n" );
	libblis_test_fprintf_c( os, "  enabled?                     %d\n", ( int )bli_info_get_enable_memkind() );
	libblis_test_fprintf_c( os, "\n" );
	libblis_test_fprintf_c( os, "gemm sandbox                     \n" );
	libblis_test_fprintf_c( os, "  enabled?                     %d\n", ( int )bli_info_get_enable_sandbox() );
	libblis_test_fprintf_c( os, "\n" );
	libblis_test_fprintf_c( os, "floating-point types           s       d       c       z \n" );
	libblis_test_fprintf_c( os, "  sizes (bytes)          %7u %7u %7u %7u\n", sizeof(float),
	                                                                          sizeof(double),
	                                                                          sizeof(scomplex),
	                                                                          sizeof(dcomplex) );
	libblis_test_fprintf_c( os, "\n" );
	libblis_test_fprintf_c( os, "\n" );
	libblis_test_fprintf_c( os, "--- BLIS parallelization info ---\n" );
	libblis_test_fprintf_c( os, "\n" );
	libblis_test_fprintf_c( os, "multithreading                 %s\n", impl_str );
	libblis_test_fprintf_c( os, "\n" );
	libblis_test_fprintf_c( os, "thread auto-factorization        \n" );
	libblis_test_fprintf_c( os, "  m dim thread ratio           %d\n", ( int )BLIS_THREAD_RATIO_M );
	libblis_test_fprintf_c( os, "  n dim thread ratio           %d\n", ( int )BLIS_THREAD_RATIO_N );
	libblis_test_fprintf_c( os, "  jr max threads               %d\n", ( int )BLIS_THREAD_MAX_JR );
	libblis_test_fprintf_c( os, "  ir max threads               %d\n", ( int )BLIS_THREAD_MAX_IR );
	libblis_test_fprintf_c( os, "\n" );
	libblis_test_fprintf_c( os, "ways of parallelism     nt    jc    pc    ic    jr    ir\n" );
	libblis_test_fprintf_c( os, "  environment        %5s %5s %5s %5s %5s %5s\n",
	                                                               nt_str, jc_nt_str, pc_nt_str,
	                                                            ic_nt_str, jr_nt_str, ir_nt_str );
	libblis_test_fprintf_c( os, "  gemm   (m,n,k=1000)      %5d %5d %5d %5d %5d\n",
	                                ( int )bli_rntm_jc_ways( &gemm ), ( int )bli_rntm_pc_ways( &gemm ),
	                                ( int )bli_rntm_ic_ways( &gemm ),
	                                ( int )bli_rntm_jr_ways( &gemm ), ( int )bli_rntm_ir_ways( &gemm ) );
	libblis_test_fprintf_c( os, "  herk   (m,k=1000)        %5d %5d %5d %5d %5d\n",
	                                ( int )bli_rntm_jc_ways( &herk ), ( int )bli_rntm_pc_ways( &herk ),
	                                ( int )bli_rntm_ic_ways( &herk ),
	                                ( int )bli_rntm_jr_ways( &herk ), ( int )bli_rntm_ir_ways( &herk ) );
	libblis_test_fprintf_c( os, "  trmm_l (m,n=1000)        %5d %5d %5d %5d %5d\n",
	                                ( int )bli_rntm_jc_ways( &trmm_l ), ( int )bli_rntm_pc_ways( &trmm_l ),
	                                ( int )bli_rntm_ic_ways( &trmm_l ),
	                                ( int )bli_rntm_jr_ways( &trmm_l ), ( int )bli_rntm_ir_ways( &trmm_l ) );
	libblis_test_fprintf_c( os, "  trmm_r (m,n=1000)        %5d %5d %5d %5d %5d\n",
	                                ( int )bli_rntm_jc_ways( &trmm_r ), ( int )bli_rntm_pc_ways( &trmm_r ),
	                                ( int )bli_rntm_ic_ways( &trmm_r ),
	                                ( int )bli_rntm_jr_ways( &trmm_r ), ( int )bli_rntm_ir_ways( &trmm_r ) );
	libblis_test_fprintf_c( os, "  trsm_l (m,n=1000)        %5d %5d %5d %5d %5d\n",
	                                ( int )bli_rntm_jc_ways( &trsm_l ), ( int )bli_rntm_pc_ways( &trsm_l ),
	                                ( int )bli_rntm_ic_ways( &trsm_l ),
	                                ( int )bli_rntm_jr_ways( &trsm_l ), ( int )bli_rntm_ir_ways( &trsm_l ) );
	libblis_test_fprintf_c( os, "  trsm_r (m,n=1000)        %5d %5d %5d %5d %5d\n",
	                                ( int )bli_rntm_jc_ways( &trsm_r ), ( int )bli_rntm_pc_ways( &trsm_r ),
	                                ( int )bli_rntm_ic_ways( &trsm_r ),
	                                ( int )bli_rntm_jr_ways( &trsm_r ), ( int )bli_rntm_ir_ways( &trsm_r ) );
	libblis_test_fprintf_c( os, "\n" );
	libblis_test_fprintf_c( os, "thread partitioning              \n" );
	//libblis_test_fprintf_c( os, "  jc/ic loops                  %s\n", "slab" );
	libblis_test_fprintf_c( os, "  jr/ir loops                  %s\n", jrir_str );
	libblis_test_fprintf_c( os, "\n" );

	libblis_test_fprintf_c( os, "\n" );
	libblis_test_fprintf_c( os, "--- BLIS default implementations ---\n" );
	libblis_test_fprintf_c( os, "\n" );
	libblis_test_fprintf_c( os, "level-3 implementations        s       d       c       z\n" );
	libblis_test_fprintf_c( os, "  gemm                   %7s %7s %7s %7s\n",
	                        bli_info_get_gemm_impl_string( BLIS_FLOAT ),
	                        bli_info_get_gemm_impl_string( BLIS_DOUBLE ),
	                        bli_info_get_gemm_impl_string( BLIS_SCOMPLEX ),
	                        bli_info_get_gemm_impl_string( BLIS_DCOMPLEX ) );
	libblis_test_fprintf_c( os, "  hemm                   %7s %7s %7s %7s\n",
	                        bli_info_get_hemm_impl_string( BLIS_FLOAT ),
	                        bli_info_get_hemm_impl_string( BLIS_DOUBLE ),
	                        bli_info_get_hemm_impl_string( BLIS_SCOMPLEX ),
	                        bli_info_get_hemm_impl_string( BLIS_DCOMPLEX ) );
	libblis_test_fprintf_c( os, "  herk                   %7s %7s %7s %7s\n",
	                        bli_info_get_herk_impl_string( BLIS_FLOAT ),
	                        bli_info_get_herk_impl_string( BLIS_DOUBLE ),
	                        bli_info_get_herk_impl_string( BLIS_SCOMPLEX ),
	                        bli_info_get_herk_impl_string( BLIS_DCOMPLEX ) );
	libblis_test_fprintf_c( os, "  her2k                  %7s %7s %7s %7s\n",
	                        bli_info_get_her2k_impl_string( BLIS_FLOAT ),
	                        bli_info_get_her2k_impl_string( BLIS_DOUBLE ),
	                        bli_info_get_her2k_impl_string( BLIS_SCOMPLEX ),
	                        bli_info_get_her2k_impl_string( BLIS_DCOMPLEX ) );
	libblis_test_fprintf_c( os, "  symm                   %7s %7s %7s %7s\n",
	                        bli_info_get_symm_impl_string( BLIS_FLOAT ),
	                        bli_info_get_symm_impl_string( BLIS_DOUBLE ),
	                        bli_info_get_symm_impl_string( BLIS_SCOMPLEX ),
	                        bli_info_get_symm_impl_string( BLIS_DCOMPLEX ) );
	libblis_test_fprintf_c( os, "  syrk                   %7s %7s %7s %7s\n",
	                        bli_info_get_syrk_impl_string( BLIS_FLOAT ),
	                        bli_info_get_syrk_impl_string( BLIS_DOUBLE ),
	                        bli_info_get_syrk_impl_string( BLIS_SCOMPLEX ),
	                        bli_info_get_syrk_impl_string( BLIS_DCOMPLEX ) );
	libblis_test_fprintf_c( os, "  syr2k                  %7s %7s %7s %7s\n",
	                        bli_info_get_syr2k_impl_string( BLIS_FLOAT ),
	                        bli_info_get_syr2k_impl_string( BLIS_DOUBLE ),
	                        bli_info_get_syr2k_impl_string( BLIS_SCOMPLEX ),
	                        bli_info_get_syr2k_impl_string( BLIS_DCOMPLEX ) );
	libblis_test_fprintf_c( os, "  trmm                   %7s %7s %7s %7s\n",
	                        bli_info_get_trmm_impl_string( BLIS_FLOAT ),
	                        bli_info_get_trmm_impl_string( BLIS_DOUBLE ),
	                        bli_info_get_trmm_impl_string( BLIS_SCOMPLEX ),
	                        bli_info_get_trmm_impl_string( BLIS_DCOMPLEX ) );
	libblis_test_fprintf_c( os, "  trmm3                  %7s %7s %7s %7s\n",
	                        bli_info_get_trmm3_impl_string( BLIS_FLOAT ),
	                        bli_info_get_trmm3_impl_string( BLIS_DOUBLE ),
	                        bli_info_get_trmm3_impl_string( BLIS_SCOMPLEX ),
	                        bli_info_get_trmm3_impl_string( BLIS_DCOMPLEX ) );
	libblis_test_fprintf_c( os, "  trsm                   %7s %7s %7s %7s\n",
	                        bli_info_get_trsm_impl_string( BLIS_FLOAT ),
	                        bli_info_get_trsm_impl_string( BLIS_DOUBLE ),
	                        bli_info_get_trsm_impl_string( BLIS_SCOMPLEX ),
	                        bli_info_get_trsm_impl_string( BLIS_DCOMPLEX ) );
	libblis_test_fprintf_c( os, "\n" );

	//bli_ind_disable_all();

	bli_ind_oper_enable_only( BLIS_GEMM, BLIS_NAT, BLIS_SCOMPLEX );
	bli_ind_oper_enable_only( BLIS_GEMM, BLIS_NAT, BLIS_DCOMPLEX );

	libblis_test_fprintf_c( os, "--- BLIS native implementation info ---\n" );
	libblis_test_fprintf_c( os, "\n" );
	libblis_test_fprintf_c( os, "                                               c       z \n" );
	libblis_test_fprintf_c( os, "complex implementation                   %7s %7s\n",
	                        bli_ind_oper_get_avail_impl_string( BLIS_GEMM, BLIS_SCOMPLEX ),
	                        bli_ind_oper_get_avail_impl_string( BLIS_GEMM, BLIS_DCOMPLEX ) );
	libblis_test_fprintf_c( os, "\n" );

	// Query a native context.
	cntx = bli_gks_query_nat_cntx();

	libblis_test_fprintf_c( os, "level-3 blocksizes             s       d       c       z \n" );
	libblis_test_fprintf_c( os, "  mc                     %7d %7d %7d %7d\n",
	                        ( int )bli_cntx_get_blksz_def_dt( BLIS_FLOAT,    BLIS_MC, cntx ),
	                        ( int )bli_cntx_get_blksz_def_dt( BLIS_DOUBLE,   BLIS_MC, cntx ),
	                        ( int )bli_cntx_get_blksz_def_dt( BLIS_SCOMPLEX, BLIS_MC, cntx ),
	                        ( int )bli_cntx_get_blksz_def_dt( BLIS_DCOMPLEX, BLIS_MC, cntx ) );
	libblis_test_fprintf_c( os, "  kc                     %7d %7d %7d %7d\n",
	                        ( int )bli_cntx_get_blksz_def_dt( BLIS_FLOAT,    BLIS_KC, cntx ),
	                        ( int )bli_cntx_get_blksz_def_dt( BLIS_DOUBLE,   BLIS_KC, cntx ),
	                        ( int )bli_cntx_get_blksz_def_dt( BLIS_SCOMPLEX, BLIS_KC, cntx ),
	                        ( int )bli_cntx_get_blksz_def_dt( BLIS_DCOMPLEX, BLIS_KC, cntx ) );
	libblis_test_fprintf_c( os, "  nc                     %7d %7d %7d %7d\n",
	                        ( int )bli_cntx_get_blksz_def_dt( BLIS_FLOAT,    BLIS_NC, cntx ),
	                        ( int )bli_cntx_get_blksz_def_dt( BLIS_DOUBLE,   BLIS_NC, cntx ),
	                        ( int )bli_cntx_get_blksz_def_dt( BLIS_SCOMPLEX, BLIS_NC, cntx ),
	                        ( int )bli_cntx_get_blksz_def_dt( BLIS_DCOMPLEX, BLIS_NC, cntx ) );
	libblis_test_fprintf_c( os, "\n" );
	libblis_test_fprintf_c( os, "  mc maximum             %7d %7d %7d %7d\n",
	                        ( int )bli_cntx_get_blksz_max_dt( BLIS_FLOAT,    BLIS_MC, cntx ),
	                        ( int )bli_cntx_get_blksz_max_dt( BLIS_DOUBLE,   BLIS_MC, cntx ),
	                        ( int )bli_cntx_get_blksz_max_dt( BLIS_SCOMPLEX, BLIS_MC, cntx ),
	                        ( int )bli_cntx_get_blksz_max_dt( BLIS_DCOMPLEX, BLIS_MC, cntx ) );
	libblis_test_fprintf_c( os, "  kc maximum             %7d %7d %7d %7d\n",
	                        ( int )bli_cntx_get_blksz_max_dt( BLIS_FLOAT,    BLIS_KC, cntx ),
	                        ( int )bli_cntx_get_blksz_max_dt( BLIS_DOUBLE,   BLIS_KC, cntx ),
	                        ( int )bli_cntx_get_blksz_max_dt( BLIS_SCOMPLEX, BLIS_KC, cntx ),
	                        ( int )bli_cntx_get_blksz_max_dt( BLIS_DCOMPLEX, BLIS_KC, cntx ) );
	libblis_test_fprintf_c( os, "  nc maximum             %7d %7d %7d %7d\n",
	                        ( int )bli_cntx_get_blksz_max_dt( BLIS_FLOAT,    BLIS_NC, cntx ),
	                        ( int )bli_cntx_get_blksz_max_dt( BLIS_DOUBLE,   BLIS_NC, cntx ),
	                        ( int )bli_cntx_get_blksz_max_dt( BLIS_SCOMPLEX, BLIS_NC, cntx ),
	                        ( int )bli_cntx_get_blksz_max_dt( BLIS_DCOMPLEX, BLIS_NC, cntx ) );
	libblis_test_fprintf_c( os, "\n" );
	libblis_test_fprintf_c( os, "  mr                     %7d %7d %7d %7d\n",
	                        ( int )bli_cntx_get_blksz_def_dt( BLIS_FLOAT,    BLIS_MR, cntx ),
	                        ( int )bli_cntx_get_blksz_def_dt( BLIS_DOUBLE,   BLIS_MR, cntx ),
	                        ( int )bli_cntx_get_blksz_def_dt( BLIS_SCOMPLEX, BLIS_MR, cntx ),
	                        ( int )bli_cntx_get_blksz_def_dt( BLIS_DCOMPLEX, BLIS_MR, cntx ) );
	libblis_test_fprintf_c( os, "  nr                     %7d %7d %7d %7d\n",
	                        ( int )bli_cntx_get_blksz_def_dt( BLIS_FLOAT,    BLIS_NR, cntx ),
	                        ( int )bli_cntx_get_blksz_def_dt( BLIS_DOUBLE,   BLIS_NR, cntx ),
	                        ( int )bli_cntx_get_blksz_def_dt( BLIS_SCOMPLEX, BLIS_NR, cntx ),
	                        ( int )bli_cntx_get_blksz_def_dt( BLIS_DCOMPLEX, BLIS_NR, cntx ) );
	libblis_test_fprintf_c( os, "\n" );
	libblis_test_fprintf_c( os, "  mr packdim             %7d %7d %7d %7d\n",
	                        ( int )bli_cntx_get_blksz_max_dt( BLIS_FLOAT,    BLIS_MR, cntx ),
	                        ( int )bli_cntx_get_blksz_max_dt( BLIS_DOUBLE,   BLIS_MR, cntx ),
	                        ( int )bli_cntx_get_blksz_max_dt( BLIS_SCOMPLEX, BLIS_MR, cntx ),
	                        ( int )bli_cntx_get_blksz_max_dt( BLIS_DCOMPLEX, BLIS_MR, cntx ) );
	libblis_test_fprintf_c( os, "  nr packdim             %7d %7d %7d %7d\n",
	                        ( int )bli_cntx_get_blksz_max_dt( BLIS_FLOAT,    BLIS_NR, cntx ),
	                        ( int )bli_cntx_get_blksz_max_dt( BLIS_DOUBLE,   BLIS_NR, cntx ),
	                        ( int )bli_cntx_get_blksz_max_dt( BLIS_SCOMPLEX, BLIS_NR, cntx ),
	                        ( int )bli_cntx_get_blksz_max_dt( BLIS_DCOMPLEX, BLIS_NR, cntx ) );
	libblis_test_fprintf_c( os, "\n" );
	libblis_test_fprintf_c( os, "micro-kernel types             s       d       c       z\n" );
	libblis_test_fprintf_c( os, "  gemm                   %7s %7s %7s %7s\n",
	                        bli_info_get_gemm_ukr_impl_string( BLIS_NAT, BLIS_FLOAT ),
	                        bli_info_get_gemm_ukr_impl_string( BLIS_NAT, BLIS_DOUBLE ),
	                        bli_info_get_gemm_ukr_impl_string( BLIS_NAT, BLIS_SCOMPLEX ),
	                        bli_info_get_gemm_ukr_impl_string( BLIS_NAT, BLIS_DCOMPLEX ) );
	libblis_test_fprintf_c( os, "  gemmtrsm_l             %7s %7s %7s %7s\n",
	                        bli_info_get_gemmtrsm_l_ukr_impl_string( BLIS_NAT, BLIS_FLOAT ),
	                        bli_info_get_gemmtrsm_l_ukr_impl_string( BLIS_NAT, BLIS_DOUBLE ),
	                        bli_info_get_gemmtrsm_l_ukr_impl_string( BLIS_NAT, BLIS_SCOMPLEX ),
	                        bli_info_get_gemmtrsm_l_ukr_impl_string( BLIS_NAT, BLIS_DCOMPLEX ) );
	libblis_test_fprintf_c( os, "  gemmtrsm_u             %7s %7s %7s %7s\n",
	                        bli_info_get_gemmtrsm_u_ukr_impl_string( BLIS_NAT, BLIS_FLOAT ),
	                        bli_info_get_gemmtrsm_u_ukr_impl_string( BLIS_NAT, BLIS_DOUBLE ),
	                        bli_info_get_gemmtrsm_u_ukr_impl_string( BLIS_NAT, BLIS_SCOMPLEX ),
	                        bli_info_get_gemmtrsm_u_ukr_impl_string( BLIS_NAT, BLIS_DCOMPLEX ) );
	libblis_test_fprintf_c( os, "  trsm_l                 %7s %7s %7s %7s\n",
	                        bli_info_get_trsm_l_ukr_impl_string( BLIS_NAT, BLIS_FLOAT ),
	                        bli_info_get_trsm_l_ukr_impl_string( BLIS_NAT, BLIS_DOUBLE ),
	                        bli_info_get_trsm_l_ukr_impl_string( BLIS_NAT, BLIS_SCOMPLEX ),
	                        bli_info_get_trsm_l_ukr_impl_string( BLIS_NAT, BLIS_DCOMPLEX ) );
	libblis_test_fprintf_c( os, "  trsm_u                 %7s %7s %7s %7s\n",
	                        bli_info_get_trsm_u_ukr_impl_string( BLIS_NAT, BLIS_FLOAT ),
	                        bli_info_get_trsm_u_ukr_impl_string( BLIS_NAT, BLIS_DOUBLE ),
	                        bli_info_get_trsm_u_ukr_impl_string( BLIS_NAT, BLIS_SCOMPLEX ),
	                        bli_info_get_trsm_u_ukr_impl_string( BLIS_NAT, BLIS_DCOMPLEX ) );
	libblis_test_fprintf_c( os, "\n" );
	libblis_test_fprintf_c( os, "\n" );

	libblis_test_fprintf_c( os, "--- BLIS induced implementation info ---\n" );
	libblis_test_fprintf_c( os, "\n" );

	for ( im = 0; im < BLIS_NAT; ++im )
	{
	if ( params->ind_enable[ im ] == 0 ) continue;

	bli_ind_oper_enable_only( BLIS_GEMM, im, BLIS_SCOMPLEX );
	bli_ind_oper_enable_only( BLIS_GEMM, im, BLIS_DCOMPLEX );

	//libblis_test_fprintf_c( os, "                               c       z \n" );
	libblis_test_fprintf_c( os, "                                               c       z \n" );
	libblis_test_fprintf_c( os, "complex implementation                   %7s %7s\n",
	                        bli_ind_oper_get_avail_impl_string( BLIS_GEMM, BLIS_SCOMPLEX ),
	                        bli_ind_oper_get_avail_impl_string( BLIS_GEMM, BLIS_DCOMPLEX ) );
	libblis_test_fprintf_c( os, "\n" );

	// Query a native context.
	cntx_c = bli_gks_query_ind_cntx( im, BLIS_SCOMPLEX );
	cntx_z = bli_gks_query_ind_cntx( im, BLIS_DCOMPLEX );

	libblis_test_fprintf_c( os, "level-3 blocksizes                             c       z \n" );
	libblis_test_fprintf_c( os, "  mc                                     %7d %7d\n",
	                        ( int )bli_cntx_get_blksz_def_dt( BLIS_SCOMPLEX, BLIS_MC, cntx_c ),
	                        ( int )bli_cntx_get_blksz_def_dt( BLIS_DCOMPLEX, BLIS_MC, cntx_z ) );
	libblis_test_fprintf_c( os, "  kc                                     %7d %7d\n",
	                        ( int )bli_cntx_get_blksz_def_dt( BLIS_SCOMPLEX, BLIS_KC, cntx_c ),
	                        ( int )bli_cntx_get_blksz_def_dt( BLIS_DCOMPLEX, BLIS_KC, cntx_z ) );
	libblis_test_fprintf_c( os, "  nc                                     %7d %7d\n",
	                        ( int )bli_cntx_get_blksz_def_dt( BLIS_SCOMPLEX, BLIS_NC, cntx_c ),
	                        ( int )bli_cntx_get_blksz_def_dt( BLIS_DCOMPLEX, BLIS_NC, cntx_z ) );
	libblis_test_fprintf_c( os, "\n" );
	libblis_test_fprintf_c( os, "  mc maximum                             %7d %7d\n",
	                        ( int )bli_cntx_get_blksz_max_dt( BLIS_SCOMPLEX, BLIS_MC, cntx_c ),
	                        ( int )bli_cntx_get_blksz_max_dt( BLIS_DCOMPLEX, BLIS_MC, cntx_z ) );
	libblis_test_fprintf_c( os, "  kc maximum                             %7d %7d\n",
	                        ( int )bli_cntx_get_blksz_max_dt( BLIS_SCOMPLEX, BLIS_KC, cntx_c ),
	                        ( int )bli_cntx_get_blksz_max_dt( BLIS_DCOMPLEX, BLIS_KC, cntx_z ) );
	libblis_test_fprintf_c( os, "  nc maximum                             %7d %7d\n",
	                        ( int )bli_cntx_get_blksz_max_dt( BLIS_SCOMPLEX, BLIS_NC, cntx_c ),
	                        ( int )bli_cntx_get_blksz_max_dt( BLIS_DCOMPLEX, BLIS_NC, cntx_z ) );
	libblis_test_fprintf_c( os, "\n" );
	libblis_test_fprintf_c( os, "  mr                                     %7d %7d\n",
	                        ( int )bli_cntx_get_blksz_def_dt( BLIS_SCOMPLEX, BLIS_MR, cntx_c ),
	                        ( int )bli_cntx_get_blksz_def_dt( BLIS_DCOMPLEX, BLIS_MR, cntx_z ) );
	libblis_test_fprintf_c( os, "  nr                                     %7d %7d\n",
	                        ( int )bli_cntx_get_blksz_def_dt( BLIS_SCOMPLEX, BLIS_NR, cntx_c ),
	                        ( int )bli_cntx_get_blksz_def_dt( BLIS_DCOMPLEX, BLIS_NR, cntx_z ) );
	libblis_test_fprintf_c( os, "\n" );
	libblis_test_fprintf_c( os, "  mr packdim                             %7d %7d\n",
	                        ( int )bli_cntx_get_blksz_max_dt( BLIS_SCOMPLEX, BLIS_MR, cntx_c ),
	                        ( int )bli_cntx_get_blksz_max_dt( BLIS_DCOMPLEX, BLIS_MR, cntx_z ) );
	libblis_test_fprintf_c( os, "  nr packdim                             %7d %7d\n",
	                        ( int )bli_cntx_get_blksz_max_dt( BLIS_SCOMPLEX, BLIS_NR, cntx_c ),
	                        ( int )bli_cntx_get_blksz_max_dt( BLIS_DCOMPLEX, BLIS_NR, cntx_z ) );
	libblis_test_fprintf_c( os, "\n" );
	libblis_test_fprintf_c( os, "micro-kernel types                             c       z\n" );
	libblis_test_fprintf_c( os, "  gemm                                   %7s %7s\n",
	                        bli_info_get_gemm_ukr_impl_string( im, BLIS_SCOMPLEX ),
	                        bli_info_get_gemm_ukr_impl_string( im, BLIS_DCOMPLEX ) );
	libblis_test_fprintf_c( os, "  gemmtrsm_l                             %7s %7s\n",
	                        bli_info_get_gemmtrsm_l_ukr_impl_string( im, BLIS_SCOMPLEX ),
	                        bli_info_get_gemmtrsm_l_ukr_impl_string( im, BLIS_DCOMPLEX ) );
	libblis_test_fprintf_c( os, "  gemmtrsm_u                             %7s %7s\n",
	                        bli_info_get_gemmtrsm_u_ukr_impl_string( im, BLIS_SCOMPLEX ),
	                        bli_info_get_gemmtrsm_u_ukr_impl_string( im, BLIS_DCOMPLEX ) );
	libblis_test_fprintf_c( os, "  trsm_l                                 %7s %7s\n",
	                        bli_info_get_trsm_l_ukr_impl_string( im, BLIS_SCOMPLEX ),
	                        bli_info_get_trsm_l_ukr_impl_string( im, BLIS_DCOMPLEX ) );
	libblis_test_fprintf_c( os, "  trsm_u                                 %7s %7s\n",
	                        bli_info_get_trsm_u_ukr_impl_string( im, BLIS_SCOMPLEX ),
	                        bli_info_get_trsm_u_ukr_impl_string( im, BLIS_DCOMPLEX ) );
	libblis_test_fprintf_c( os, "\n" );

	}

	bli_ind_disable_all();

	libblis_test_fprintf_c( os, "\n" );
	libblis_test_fprintf_c( os, "--- BLIS misc. other info ---\n" );
	libblis_test_fprintf_c( os, "\n" );
	libblis_test_fprintf_c( os, "level-2 cache blocksizes       s       d       c       z \n" );
	libblis_test_fprintf_c( os, "  m dimension            %7d %7d %7d %7d\n",
	                        ( int )bli_cntx_get_blksz_def_dt( BLIS_FLOAT,    BLIS_M2, cntx ),
	                        ( int )bli_cntx_get_blksz_def_dt( BLIS_DOUBLE,   BLIS_M2, cntx ),
	                        ( int )bli_cntx_get_blksz_def_dt( BLIS_SCOMPLEX, BLIS_M2, cntx ),
	                        ( int )bli_cntx_get_blksz_def_dt( BLIS_DCOMPLEX, BLIS_M2, cntx ) );
	libblis_test_fprintf_c( os, "  n dimension            %7d %7d %7d %7d\n",
	                        ( int )bli_cntx_get_blksz_def_dt( BLIS_FLOAT,    BLIS_N2, cntx ),
	                        ( int )bli_cntx_get_blksz_def_dt( BLIS_DOUBLE,   BLIS_N2, cntx ),
	                        ( int )bli_cntx_get_blksz_def_dt( BLIS_SCOMPLEX, BLIS_N2, cntx ),
	                        ( int )bli_cntx_get_blksz_def_dt( BLIS_DCOMPLEX, BLIS_N2, cntx ) );
	libblis_test_fprintf_c( os, "\n" );
	libblis_test_fprintf_c( os, "level-1f fusing factors        s       d       c       z \n" );
	libblis_test_fprintf_c( os, "  axpyf                  %7d %7d %7d %7d\n",
	                        ( int )bli_cntx_get_blksz_def_dt( BLIS_FLOAT,    BLIS_AF, cntx ),
	                        ( int )bli_cntx_get_blksz_def_dt( BLIS_DOUBLE,   BLIS_AF, cntx ),
	                        ( int )bli_cntx_get_blksz_def_dt( BLIS_SCOMPLEX, BLIS_AF, cntx ),
	                        ( int )bli_cntx_get_blksz_def_dt( BLIS_DCOMPLEX, BLIS_AF, cntx ) );
	libblis_test_fprintf_c( os, "  dotxf                  %7d %7d %7d %7d\n",
	                        ( int )bli_cntx_get_blksz_def_dt( BLIS_FLOAT,    BLIS_DF, cntx ),
	                        ( int )bli_cntx_get_blksz_def_dt( BLIS_DOUBLE,   BLIS_DF, cntx ),
	                        ( int )bli_cntx_get_blksz_def_dt( BLIS_SCOMPLEX, BLIS_DF, cntx ),
	                        ( int )bli_cntx_get_blksz_def_dt( BLIS_DCOMPLEX, BLIS_DF, cntx ) );
	libblis_test_fprintf_c( os, "  dotxaxpyf              %7d %7d %7d %7d\n",
	                        ( int )bli_cntx_get_blksz_def_dt( BLIS_FLOAT,    BLIS_XF, cntx ),
	                        ( int )bli_cntx_get_blksz_def_dt( BLIS_DOUBLE,   BLIS_XF, cntx ),
	                        ( int )bli_cntx_get_blksz_def_dt( BLIS_SCOMPLEX, BLIS_XF, cntx ),
	                        ( int )bli_cntx_get_blksz_def_dt( BLIS_DCOMPLEX, BLIS_XF, cntx ) );
	libblis_test_fprintf_c( os, "\n" );
	libblis_test_fprintf( os, "\n" );

	// Output the contents of the param struct.
	libblis_test_fprintf_c( os, "\n" );
	libblis_test_fprintf_c( os, "--- BLIS test suite parameters ----------------------------\n" );
	libblis_test_fprintf_c( os, "\n" );
	libblis_test_fprintf_c( os, "num repeats per experiment   %u\n", params->n_repeats );
	libblis_test_fprintf_c( os, "num matrix storage schemes   %u\n", params->n_mstorage );
	libblis_test_fprintf_c( os, "storage[ matrix ]            %s\n", params->storage[ BLIS_TEST_MATRIX_OPERAND ] );
	libblis_test_fprintf_c( os, "num vector storage schemes   %u\n", params->n_vstorage );
	libblis_test_fprintf_c( os, "storage[ vector ]            %s\n", params->storage[ BLIS_TEST_VECTOR_OPERAND ] );
	libblis_test_fprintf_c( os, "mix all storage schemes?     %u\n", params->mix_all_storage );
	libblis_test_fprintf_c( os, "test with aligned memory?    %u\n", params->alignment );
	libblis_test_fprintf_c( os, "randomization method         %u\n", params->rand_method );
	libblis_test_fprintf_c( os, "general stride spacing       %u\n", params->gs_spacing );
	libblis_test_fprintf_c( os, "num datatypes                %u\n", params->n_datatypes );
	libblis_test_fprintf_c( os, "datatype[0]                  %d (%c)\n", params->datatype[0],
	                                                                params->datatype_char[0] );
	for( i = 1; i < params->n_datatypes; ++i )
	libblis_test_fprintf_c( os, "        [%d]                  %d (%c)\n", i, params->datatype[i],
	                                                                    params->datatype_char[i] );
	libblis_test_fprintf_c( os, "mix domains for gemm?        %u\n", params->mixed_domain );
	libblis_test_fprintf_c( os, "mix precisions for gemm?     %u\n", params->mixed_precision );
	libblis_test_fprintf_c( os, "problem size: first to test  %u\n", params->p_first );
	libblis_test_fprintf_c( os, "problem size: max to test    %u\n", params->p_max );
	libblis_test_fprintf_c( os, "problem size increment       %u\n", params->p_inc );
	libblis_test_fprintf_c( os, "complex implementations        \n" );
	libblis_test_fprintf_c( os, "  3mh?                       %u\n", params->ind_enable[ BLIS_3MH ] );
	libblis_test_fprintf_c( os, "  3m1?                       %u\n", params->ind_enable[ BLIS_3M1 ] );
	libblis_test_fprintf_c( os, "  4mh?                       %u\n", params->ind_enable[ BLIS_4MH ] );
	libblis_test_fprintf_c( os, "  4m1b (4mb)?                %u\n", params->ind_enable[ BLIS_4M1B ] );
	libblis_test_fprintf_c( os, "  4m1a (4m1)?                %u\n", params->ind_enable[ BLIS_4M1A ] );
	libblis_test_fprintf_c( os, "  1m?                        %u\n", params->ind_enable[ BLIS_1M ] );
	libblis_test_fprintf_c( os, "  native?                    %u\n", params->ind_enable[ BLIS_NAT ] );
	libblis_test_fprintf_c( os, "simulated app-level threads  %u\n", params->n_app_threads );
	libblis_test_fprintf_c( os, "error-checking level         %u\n", params->error_checking_level );
	libblis_test_fprintf_c( os, "reaction to failure          %c\n", params->reaction_to_failure );
	libblis_test_fprintf_c( os, "output in matlab format?     %u\n", params->output_matlab_format );
	libblis_test_fprintf_c( os, "output to stdout AND files?  %u\n", params->output_files );
	libblis_test_fprintf_c( os, "\n" );
	libblis_test_fprintf( os, "\n" );

#ifndef BLIS_ENABLE_GEMM_MD
	// Notify the user if mixed domain or mixed precision was requested.
	if ( params->mixed_domain || params->mixed_precision )
	{
		libblis_test_printf_error( "mixed domain and/or mixed precision testing requested, but building against BLIS without mixed datatype support.\n" );
	}
#endif

	// If mixed domain or mixed precision was requested, we disable all
	// induced methods except 1m and native execution.
	if ( params->mixed_domain || params->mixed_precision )
	{
		ind_t im;

		for ( im = BLIS_IND_FIRST; im < BLIS_IND_LAST+1; ++im )
		{
			if ( im != BLIS_1M && im != BLIS_NAT )
				params->ind_enable[ im ] = 0;
		}
	}
}



void libblis_test_output_op_struct( FILE* os, test_op_t* op, char* op_str )
{
	dimset_t dimset = op->dimset;

	if      ( dimset == BLIS_TEST_DIMS_MNK )
	{
		libblis_test_fprintf_c( os, "%s m n k                  %d %d %d\n", op_str,
		                                op->dim_spec[0], op->dim_spec[1], op->dim_spec[2] );
	}
	else if ( dimset == BLIS_TEST_DIMS_MN )
	{
		libblis_test_fprintf_c( os, "%s m n                    %d %d\n", op_str,
		                                op->dim_spec[0], op->dim_spec[1] );
	}
	else if ( dimset == BLIS_TEST_DIMS_MK )
	{
		libblis_test_fprintf_c( os, "%s m k                    %d %d\n", op_str,
		                                op->dim_spec[0], op->dim_spec[1] );
	}
	else if ( dimset == BLIS_TEST_DIMS_M ||
	          dimset == BLIS_TEST_DIMS_MF )
	{
		libblis_test_fprintf_c( os, "%s m                      %d\n", op_str,
		                                op->dim_spec[0] );
	}
	else if ( dimset == BLIS_TEST_DIMS_K )
	{
		libblis_test_fprintf_c( os, "%s k                      %d\n", op_str,
		                                op->dim_spec[0] );
	}
	else if ( dimset == BLIS_TEST_NO_DIMS )
	{
		// Do nothing.
	}
	else
	{
		libblis_test_printf_error( "Invalid dimension combination.\n" );
	}

	if ( op->n_params > 0 )
		libblis_test_fprintf_c( os, "%s operand params         %s\n", op_str, op->params );
	else
		libblis_test_fprintf_c( os, "%s operand params         %s\n", op_str, "(none)" );

	libblis_test_fprintf_c( os, "\n" );
	libblis_test_fprintf( os, "\n" );
}



char* libblis_test_get_string_for_result( double    resid,
                                          num_t     dt,
                                          thresh_t* thresh )
{
	char* r_val;

	// Before checking against the thresholds, make sure the residual is
	// neither NaN nor Inf. (Note that bli_isnan() and bli_isinf() are
	// both simply wrappers to the isnan() and isinf() macros defined
	// defined in math.h.)
	if ( bli_isnan( resid ) || bli_isinf( resid ) )
	{
		r_val = libblis_test_fail_string;
	}
	else
	{
		// Check the result against the thresholds.
		if      ( resid > thresh[dt].failwarn ) r_val = libblis_test_fail_string;
		else if ( resid > thresh[dt].warnpass ) r_val = libblis_test_warn_string;
		else                                    r_val = libblis_test_pass_string;
	}

	return r_val;
}



param_t libblis_test_get_param_type_for_char( char p_type )
{
	param_t r_val;

	if      ( p_type == 's' ) r_val = BLIS_TEST_PARAM_SIDE;
	else if ( p_type == 'u' ) r_val = BLIS_TEST_PARAM_UPLO;
	else if ( p_type == 'e' ) r_val = BLIS_TEST_PARAM_UPLODE;
	else if ( p_type == 'h' ) r_val = BLIS_TEST_PARAM_TRANS;
	else if ( p_type == 'c' ) r_val = BLIS_TEST_PARAM_CONJ;
	else if ( p_type == 'd' ) r_val = BLIS_TEST_PARAM_DIAG;
	else
	{
		r_val = 0;
		libblis_test_printf_error( "Invalid parameter character.\n" );
	}

	return r_val;
}



operand_t libblis_test_get_operand_type_for_char( char o_type )
{
	operand_t r_val;

	if      ( o_type == 'm' ) r_val = BLIS_TEST_MATRIX_OPERAND;
	else if ( o_type == 'v' ) r_val = BLIS_TEST_VECTOR_OPERAND;
	else
	{
		r_val = 0;
		libblis_test_printf_error( "Invalid operand character.\n" );
	}

	return r_val;
}



unsigned int libblis_test_get_n_dims_from_dimset( dimset_t dimset )
{
	unsigned int n_dims;

	if      ( dimset == BLIS_TEST_DIMS_MNK ) n_dims = 3;
	else if ( dimset == BLIS_TEST_DIMS_MN  ) n_dims = 2;
	else if ( dimset == BLIS_TEST_DIMS_MK  ) n_dims = 2;
	else if ( dimset == BLIS_TEST_DIMS_M   ) n_dims = 1;
	else if ( dimset == BLIS_TEST_DIMS_MF  ) n_dims = 1;
	else if ( dimset == BLIS_TEST_DIMS_K   ) n_dims = 1;
	else if ( dimset == BLIS_TEST_NO_DIMS  ) n_dims = 0;
	else
	{
		n_dims = 0;
		libblis_test_printf_error( "Invalid dimension combination.\n" );
	}

	return n_dims;
}



unsigned int libblis_test_get_n_dims_from_string( char* dims_str )
{
	unsigned int n_dims;
	char*        cp;

	cp = dims_str;

	for ( n_dims = 0; *cp != '\0'; ++n_dims )
	{
		//printf( "n_dims = %u\n", n_dims );
		while ( isspace( *cp ) )
		{
			//printf( "current char: _%c_", *cp );
			 ++cp;
		}

		while ( isdigit( *cp ) )
		{
			//printf( "current char: _%c_", *cp );
			++cp;
		}
	}
	//printf( "n_dims finally = %u\n", n_dims );

	return n_dims;
}



dim_t libblis_test_get_dim_from_prob_size( int          dim_spec,
                                           unsigned int p_size )
{
	dim_t dim;

	if ( dim_spec < 0 ) dim = p_size / bli_abs(dim_spec);
	else                dim = dim_spec;

	return dim;
}



void libblis_test_fill_param_strings( char*         p_spec_str,
                                      char**        chars_for_param,
                                      unsigned int  n_params,
                                      unsigned int  n_param_combos,
                                      char**        pc_str )
{
	unsigned int  pci, pi, i;
	unsigned int* counter;
	unsigned int* n_vals_for_param;

	// Allocate an array that will store the number of parameter values
	// for each parameter.
	n_vals_for_param = ( unsigned int* ) malloc( n_params * sizeof( unsigned int ) );

	// Fill n_vals_for_param[i] with the number of parameter values (chars)
	// in chars_for_param[i] (this is simply the string length).
	for ( i = 0; i < n_params; ++i )
	{
		if ( p_spec_str[i] == '?' ) n_vals_for_param[i] = strlen( chars_for_param[i] );
		else                        n_vals_for_param[i] = 1;
	}

	// Allocate an array with one digit per parameter. We will use
	// this array to keep track of our progress as we canonically move
	// though all possible parameter combinations.
	counter = ( unsigned int* ) malloc( n_params * sizeof( unsigned int ) );

	// Initialize all values in c to zero.
	for ( i = 0; i < n_params; ++i ) counter[i] = 0;

	for ( pci = 0; pci < n_param_combos; ++pci )
	{
		// Iterate backwards through each parameter string we create, since we
		// want to form (for example, if the parameters are transa and conjx:
		// (1) nn, (2) nc, (3) cn, (4) cc, (5) tn, (6) tc, (7) hn, (8) hc.
		for ( i = 0, pi = n_params - 1; i < n_params; --pi, ++i )
		{
			// If the current parameter character, p_spec_str[pi] is fixed (ie: if
			// it is not '?'), then just copy it into the parameter combination
			// string. Otherwise, map the current integer value in c to the
			// corresponding character in char_for_param[pi].
			if ( p_spec_str[pi] != '?' )
				pc_str[pci][pi] = p_spec_str[pi];
			else
				pc_str[pci][pi] = chars_for_param[ pi ][ counter[pi] ];
		}

		// Terminate the current parameter combination string.
		pc_str[pci][n_params] = '\0';

		// Only try to increment/carryover if this is NOT the last param
		// combo.
		if ( pci < n_param_combos - 1 )
		{
			// Increment the least-most significant counter.
			counter[ n_params - 1 ]++;

			// Perform "carryover" if needed.
			carryover( &counter[ n_params - 1 ],
			           &n_vals_for_param[ n_params - 1 ],
			           n_params );
		}
	}

	// Free the temporary arrays.
	free( counter );

	// Free the array holding the number of parameter values for each
	// parameter.
	free( n_vals_for_param );
}



void carryover( unsigned int* c,
                unsigned int* n_vals_for_param,
                unsigned int  n_params )
{
	if ( n_params == 1 ) return;
	else
	{
		if ( *c == *n_vals_for_param )
		{
			*c = 0;
			*(c-1) += 1;
			carryover( c-1, n_vals_for_param-1, n_params-1 );
		}
	}
}



void libblis_test_op_driver
     (
       thread_data_t* tdata,
       test_params_t* params,
       test_op_t*     op,
       iface_t        iface,
       char*          op_str,
       char*          p_types,
       char*          o_types,
       thresh_t*      thresh,
       void (*f_exp)  (test_params_t*, // params struct
                       test_op_t*,     // op struct
                       iface_t,        // iface
                       char*,          // datatype (current datatype)
                       char*,          // pc_str (current param string)
                       char*,          // sc_str (current storage string)
                       unsigned int,   // p_cur (current problem size)
                       double*,        // perf
                       double* )       // residual
     )
{
	unsigned int  n_mstorage          = params->n_mstorage;
	unsigned int  n_vstorage          = params->n_vstorage;
	unsigned int  n_datatypes         = params->n_datatypes;
	unsigned int  p_first             = params->p_first;
	unsigned int  p_max               = params->p_max;
	unsigned int  p_inc               = params->p_inc;
	unsigned int  mix_all_storage     = params->mix_all_storage;
	unsigned int  mixed_domain        = params->mixed_domain;
	unsigned int  mixed_precision     = params->mixed_precision;
	unsigned int  reaction_to_failure = params->reaction_to_failure;

	num_t         datatype;
	num_t         dt_check;
	char          dt_char;

	char*         p_spec_str;
	unsigned int  n_params;
	char**        chars_for_param;
	unsigned int  n_param_combos;
	char**        pc_str;

	char          s_spec_str[ MAX_NUM_OPERANDS + 1 ];
	unsigned int  n_operands;
	unsigned int  n_operandsp1;
	char**        chars_for_storage;
	unsigned int  n_store_combos;
	char**        sc_str;

	char          d_spec_str[ MAX_NUM_OPERANDS + 1 ];
	char**        chars_for_spdt;
	char**        chars_for_dpdt;
	unsigned int  n_spdt_combos;
	unsigned int  n_dpdt_combos;
	unsigned int  n_dt_combos;
	char**        dc_str;

	char**        chars_for_dt;
	char**        chars_for_rddt;
	char**        chars_for_cddt;
	unsigned int  n_rddt_combos;
	unsigned int  n_cddt_combos;

	unsigned int  p_cur, pi;
	unsigned int  indi, pci, sci, dci, i, j, o;
	unsigned int  is_mixed_dt;

	double        perf, resid;
	char*         pass_str;
	char*         ind_str;
	char          blank_str[32];
	char          funcname_str[64];
	char          dims_str[64];
	char          label_str[128];
	unsigned int  n_spaces;
	unsigned int  n_dims_print;

	FILE*         output_stream = NULL;

	// These arrays are malloc()'ed in select branches. Here, we set
	// them to NULL so they can be unconditionally free()'ed at the
	// end of the function.
	chars_for_rddt = NULL;
	chars_for_cddt = NULL;
	chars_for_spdt = NULL;
	chars_for_dpdt = NULL;

	// If output to files was requested, attempt to open a file stream.
	if ( params->output_files )
		libblis_test_fopen_ofile( op_str, iface, &output_stream );

	// Set the error-checking level according to what was specified in the
	// input file.
	if ( params->error_checking_level == 0 )
		bli_error_checking_level_set( BLIS_NO_ERROR_CHECKING );
	else
		bli_error_checking_level_set( BLIS_FULL_ERROR_CHECKING );

	// Obtain the parameter specification (filter) string.
	p_spec_str = op->params;

	// Figure out how many parameters we have.
	n_params = strlen( p_types );

	if ( strlen( p_types ) != strlen( p_spec_str) )
	{
		libblis_test_printf_error( "Parameter specification string from input file does not match length of p_types string.\n" );
	}

	// Allocate an array that stores pointers to the sets of possible parameter
	// chars for each parameter.
	chars_for_param = ( char** ) malloc( n_params * sizeof( char* ) );

	// Set the values in chars_for_param to the appropriate string addresses.
	for ( i = 0; i < n_params; ++i )
	{
		param_t param_type = libblis_test_get_param_type_for_char( p_types[i] );
		chars_for_param[i] = libblis_test_param_chars[ param_type ];
	}

	// Compute the total number of parameter combinations to test (which is
	// simply the product of the string lengths of chars_for_param[i].
	n_param_combos = libblis_test_count_combos( n_params, p_spec_str,
	                                            chars_for_param );

	// Allocate an array of parameter combination strings, one for each
	// parameter combination that needs to be tested.
	pc_str = ( char** ) malloc( n_param_combos * sizeof( char* ) );
	for ( i = 0; i < n_param_combos; ++i )
		pc_str[i] = ( char* ) malloc( ( n_params + 1 ) * sizeof( char ) );

	// Fill the parameter combination strings in pc_str with the parameter
	// combinations called for by the parameter string p_spec_str.
	libblis_test_fill_param_strings( p_spec_str,
	                                 chars_for_param,
	                                 n_params,
	                                 n_param_combos,
	                                 pc_str );



	// Figure out how many operands we have.
	n_operands = strlen( o_types );

	// If we are testing a micro-kernel, unconditionally disable the
	// "mix all storage" option.
	if ( iface == BLIS_TEST_SEQ_UKERNEL )
		mix_all_storage = DISABLE;

	// Enumerate all combinations of storage schemes requested.
	if ( mix_all_storage )
	{
		// Fill storage specification string with wildcard chars.
		for ( i = 0; i < n_operands; ++i ) s_spec_str[i] = '?';
		s_spec_str[i] = '\0';

		// Allocate an array that stores pointers to the sets of possible
		// storage chars for each operand.
		chars_for_storage = ( char** ) malloc( n_operands * sizeof( char* ) );

		// Set the values in chars_for_storage to the address of the string
		// that holds the storage chars.
		for ( i = 0; i < n_operands; ++i )
		{
			operand_t operand_type = libblis_test_get_operand_type_for_char( o_types[i] );
			chars_for_storage[i] = libblis_test_store_chars[ operand_type ];
		}

		// Compute the total number of storage combinations to test (which is
		// simply the product of the string lengths of chars_for_storage[i].
		n_store_combos = libblis_test_count_combos( n_operands, s_spec_str,
		                                            chars_for_storage );

		// Allocate an array of storage combination strings, one for each
		// storage combination that needs to be tested.
		sc_str = ( char** ) malloc( n_store_combos * sizeof( char* ) );
		for ( sci = 0; sci < n_store_combos; ++sci )
			sc_str[sci] = ( char* ) malloc( ( n_operands + 1 ) * sizeof( char ) );


		// Fill the storage combination strings in sc_str with the storage
		// combinations called for by the storage string p_spec_str.
		libblis_test_fill_param_strings( s_spec_str,
	                                     chars_for_storage,
	                                     n_operands,
	                                     n_store_combos,
	                                     sc_str );
	}
	else // if ( !mix_all_storage )
	{
		// Only run combinations where all operands of either type (matrices
		// or vectors) are stored in one storage scheme or another (no mixing
		// of schemes within the same operand type).
		unsigned int n_mat_operands = 0;
		unsigned int n_vec_operands = 0;

		for ( o = 0; o < n_operands; ++o )
		{
			operand_t operand_type
			          = libblis_test_get_operand_type_for_char( o_types[o] );
			if      ( operand_type == BLIS_TEST_MATRIX_OPERAND ) ++n_mat_operands;
			else if ( operand_type == BLIS_TEST_VECTOR_OPERAND ) ++n_vec_operands;
		}

		// We compute the total number of storage combinations based on whether
		// the current operation has only matrix operands, only vector operands,
		// or both.
		if      ( n_vec_operands == 0 )
		{
			n_store_combos = n_mstorage;
			n_vstorage = 1;
		}
		else if ( n_mat_operands == 0 )
		{
			n_store_combos = n_vstorage;
			n_mstorage = 1;
		}
		else
		{
			n_store_combos = n_mstorage * n_vstorage;
		}

		sc_str = ( char** ) malloc( n_store_combos * sizeof( char* ) );

		for ( j = 0; j < n_mstorage; ++j )
		{
			for ( i = 0; i < n_vstorage; ++i )
			{
				sci = j*n_vstorage + i;

				sc_str[ sci ]
				= ( char* ) malloc( ( n_operands + 1 ) * sizeof( char ) );

				for ( o = 0; o < n_operands; ++o )
				{ 
					unsigned int ij;
					operand_t    operand_type
					= libblis_test_get_operand_type_for_char( o_types[o] );

					if ( operand_type == BLIS_TEST_MATRIX_OPERAND ) ij = j;
					else                                            ij = i;

					sc_str[sci][o] = params->storage[ operand_type ][ij];
				}
				sc_str[sci][n_operands] = '\0';
			}
		}
	}

	// Enumerate all combinations of datatype domains requested, but only
	// for the gemm operation.

	if      ( !mixed_domain &&  mixed_precision && op->opid == BLIS_GEMM )
	{
		is_mixed_dt = TRUE;

		// Increment the number of operands by one to account for the
		// computation precision (or computation datatype, as we will encode
		// it in the char string).
		n_operandsp1 = n_operands + 1;

		unsigned int has_rd = libblis_test_dt_str_has_rd_char( params );
		unsigned int has_cd = libblis_test_dt_str_has_cd_char( params );

		// Fill datatype specification string with wildcard chars.
		for ( i = 0; i < n_operandsp1; ++i ) d_spec_str[i] = '?';
		d_spec_str[i] = '\0';

		// Allocate an array that stores pointers to the sets of possible
		// datatype chars for each operand.
		chars_for_rddt = ( char** ) malloc( n_operandsp1 * sizeof( char* ) );
		chars_for_cddt = ( char** ) malloc( n_operandsp1 * sizeof( char* ) );

		// Set the values in chars_for_rddt/cddt to the address of the string
		// that holds the datatype chars.
		for ( i = 0; i < n_operandsp1; ++i )
		{
			chars_for_rddt[i] = libblis_test_rd_chars;
			chars_for_cddt[i] = libblis_test_cd_chars;
		}

		// Set the last set of chars in chars_for_cddt to the real domain
		// charset. This is because the last char will be the computation
		// precision.
		chars_for_cddt[i-1] = libblis_test_rd_chars;

		// Compute the total number of datatype combinations to test (which is
		// simply the product of the string lengths of chars_for_spdt/dpdt[i]).
		// NOTE: We skip inspecting/branching off of the d_spec_str chars since
		// we know they are all '?'.
		n_rddt_combos = 0; n_cddt_combos = 0;

		if ( has_rd )
			n_rddt_combos = libblis_test_count_combos( n_operandsp1, d_spec_str,
			                                           chars_for_rddt );

		if ( has_cd )
			n_cddt_combos = libblis_test_count_combos( n_operandsp1, d_spec_str,
			                                           chars_for_cddt );

		// Add real and complex domain combinations.
		n_dt_combos = n_rddt_combos + n_cddt_combos;

		// Allocate an array of datatype combination strings, one for each
		// datatype combination that needs to be tested.
		dc_str = ( char** ) malloc( n_dt_combos * sizeof( char* ) );
		for ( dci = 0; dci < n_dt_combos; ++dci )
			dc_str[dci] = ( char* ) malloc( ( n_operandsp1 + 1 ) * sizeof( char ) );

		char** dc_str_p = dc_str;

		// Fill the datatype combination strings in dc_str with the datatype
		// combinations implied by chars_for_rddt/cddt.
		if ( has_rd )
		{
			libblis_test_fill_param_strings( d_spec_str,
			                                 chars_for_rddt,
			                                 n_operandsp1,
			                                 n_rddt_combos,
			                                 dc_str_p );
			dc_str_p += n_rddt_combos;
		}
		if ( has_cd )
		{
			libblis_test_fill_param_strings( d_spec_str,
			                                 chars_for_cddt,
			                                 n_operandsp1,
			                                 n_cddt_combos,
			                                 dc_str_p );
			dc_str_p += n_cddt_combos;
		}

#if 0
		printf( "n_rddt_combos = %d\n", n_rddt_combos );
		printf( "n_cddt_combos = %d\n", n_cddt_combos );
		printf( "n_dt_combos   = %d\n\n", n_dt_combos );

		for ( dci = 0; dci < n_dt_combos; ++dci )
			printf( "dc_str[%2d] = %s\n", dci, dc_str[dci] );

		bli_abort();
#endif
	}
	else if (  mixed_domain && !mixed_precision && op->opid == BLIS_GEMM )
	{
		is_mixed_dt = TRUE;

		// Increment the number of operands by one to account for the
		// computation precision (or computation datatype, as we will encode
		// it in the char string).
		n_operandsp1 = n_operands + 1;

		unsigned int has_sp = libblis_test_dt_str_has_sp_char( params );
		unsigned int has_dp = libblis_test_dt_str_has_dp_char( params );

		// Fill datatype specification string with wildcard chars.
		for ( i = 0; i < n_operands; ++i ) d_spec_str[i] = '?';
		d_spec_str[i] = '\0';

		// Allocate an array that stores pointers to the sets of possible
		// datatype chars for each operand (plus the computation precision
		// char).
		chars_for_spdt = ( char** ) malloc( n_operands * sizeof( char* ) );
		chars_for_dpdt = ( char** ) malloc( n_operands * sizeof( char* ) );

		// Set the values in chars_for_spdt/dpdt to the address of the string
		// that holds the datatype chars.
		for ( i = 0; i < n_operands; ++i )
		{
			chars_for_spdt[i] = libblis_test_sp_chars;
			chars_for_dpdt[i] = libblis_test_dp_chars;
		}

		// Compute the total number of datatype combinations to test (which is
		// simply the product of the string lengths of chars_for_spdt/dpdt[i]).
		// NOTE: We skip inspecting/branching off of the d_spec_str chars since
		// we know they are all '?'.
		n_spdt_combos = 0; n_dpdt_combos = 0;

		if ( has_sp )
			n_spdt_combos = libblis_test_count_combos( n_operands, d_spec_str,
			                                           chars_for_spdt );

		if ( has_dp )
			n_dpdt_combos = libblis_test_count_combos( n_operands, d_spec_str,
			                                           chars_for_dpdt );

		// Add single- and double-precision combinations.
		n_dt_combos = n_spdt_combos + n_dpdt_combos;

		// Allocate an array of datatype combination strings, one for each
		// datatype combination that needs to be tested.
		dc_str = ( char** ) malloc( n_dt_combos * sizeof( char* ) );
		for ( dci = 0; dci < n_dt_combos; ++dci )
			dc_str[dci] = ( char* ) malloc( ( n_operandsp1 + 1 ) * sizeof( char ) );

		char** dc_str_p = dc_str;

		// Fill the datatype combination strings in dc_str with the datatype
		// combinations implied by chars_for_spdt/dpdt.
		if ( has_sp )
		{
			libblis_test_fill_param_strings( d_spec_str,
			                                 chars_for_spdt,
			                                 n_operands,
			                                 n_spdt_combos,
			                                 dc_str_p );
			dc_str_p += n_spdt_combos;
		}
		if ( has_dp )
		{
			libblis_test_fill_param_strings( d_spec_str,
			                                 chars_for_dpdt,
			                                 n_operands,
			                                 n_dpdt_combos,
			                                 dc_str_p );
			dc_str_p += n_dpdt_combos;
		}

		// Manually set the computation char to the real projection of the
		// first char of each combination.
		int prec_i = n_operands;
		for ( i = 0; i < n_dt_combos; ++i )
		{
			dc_str[i][prec_i]   = libblis_test_proj_dtchar_to_precchar( dc_str[i][0] );
			dc_str[i][prec_i+1] = '\0';
		}

#if 0
		printf( "n_spdt_combos = %d\n", n_spdt_combos );
		printf( "n_dpdt_combos = %d\n", n_dpdt_combos );
		printf( "n_dt_combos   = %d\n\n", n_dt_combos );

		for ( dci = 0; dci < n_dt_combos; ++dci )
			printf( "dc_str[%2d] = %s\n", dci, dc_str[dci] );

		bli_abort();
#endif
	}
	else if (  mixed_domain &&  mixed_precision && op->opid == BLIS_GEMM )
	{
		is_mixed_dt = TRUE;

		// Increment the number of operands by one to account for the
		// computation precision (or computation datatype, as we will encode
		// it in the char string).
		n_operandsp1 = n_operands + 1;

		// Fill datatype specification string with wildcard chars.
		for ( i = 0; i < n_operandsp1; ++i ) d_spec_str[i] = '?';
		d_spec_str[i] = '\0';

		// Allocate an array that stores pointers to the sets of possible
		// datatype chars for each operand.
		chars_for_dt = ( char** ) malloc( n_operandsp1 * sizeof( char* ) );

		// Set the values in chars_for_rddt/cddt to the address of the string
		// that holds the datatype chars.
		for ( i = 0; i < n_operandsp1; ++i )
		{
			chars_for_dt[i] = libblis_test_dt_chars;
		}

		// Set the last set of chars in chars_for_dt to the real domain
		// charset. This is because the last char will be the computation
		// precision, with the computation domain implied by the operands'
		// storage datatypes.
		chars_for_dt[i-1] = libblis_test_rd_chars;

		// Compute the total number of datatype combinations to test (which is
		// simply the product of the string lengths of chars_for_dt[i]).
		// NOTE: We skip inspecting/branching off of the d_spec_str chars since
		// we know they are all '?'.
		n_dt_combos = libblis_test_count_combos( n_operandsp1, d_spec_str,
		                                         chars_for_dt );

		// Allocate an array of datatype combination strings, one for each
		// datatype combination that needs to be tested.
		dc_str = ( char** ) malloc( n_dt_combos * sizeof( char* ) );
		for ( dci = 0; dci < n_dt_combos; ++dci )
			dc_str[dci] = ( char* ) malloc( ( n_operandsp1 + 1 ) * sizeof( char ) );

		// Fill the datatype combination strings in dc_str with the datatype
		// combinations implied by chars_for_rddt/cddt.
		libblis_test_fill_param_strings( d_spec_str,
		                                 chars_for_dt,
		                                 n_operandsp1,
		                                 n_dt_combos,
		                                 dc_str );

#if 0
		printf( "n_dt_combos   = %d\n\n", n_dt_combos );

		for ( dci = 0; dci < n_dt_combos; ++dci )
			printf( "dc_str[%3d] = %s\n", dci, dc_str[dci] );

		bli_abort();
#endif
	}
	else // ( ( !mixed_domain && !mixed_precision ) || op->opid != BLIS_GEMM )
	{
		is_mixed_dt = FALSE;

		// Increment the number of operands by one to account for the
		// computation precision (or computation datatype, as we will encode
		// it in the char string).
		n_operandsp1 = n_operands + 1;

		// Since we are not mixing domains, we only consider n_datatype
		// datatype combinations, where each combination is actually
		// homogeneous (e.g. "sss", "ddd", etc., if n_operands == 3).
		n_dt_combos = n_datatypes;

		// Allocate an array of datatype combination strings, one for each
		// datatype specified.
		dc_str = ( char** ) malloc( n_dt_combos * sizeof( char* ) );
		for ( dci = 0; dci < n_dt_combos; ++dci )
			dc_str[dci] = ( char* ) malloc( ( n_operandsp1 + 1 ) * sizeof( char ) );

		// Fill each datatype combination string with the same dt char for
		// each operand in the current operation.
		for ( dci = 0; dci < n_dt_combos; ++dci )
		{
			dt_char = params->datatype_char[dci];

			for ( i = 0; i < n_operands; ++i )
				dc_str[dci][i] = dt_char;

			// Encode the computation precision as the last char.
			dc_str[dci][i] = libblis_test_proj_dtchar_to_precchar( dc_str[dci][0] );

			dc_str[dci][i+1] = '\0';
		}

#if 0
		printf( "n_dt_combos   = %d\n\n", n_dt_combos );

		for ( dci = 0; dci < n_dt_combos; ++dci )
			printf( "dc_str[%3d] = %s\n", dci, dc_str[dci] );

		bli_abort();
#endif
	}



	// These statements should only be executed by one thread.
	if ( tdata->id == 0 )
	{
		// Output a heading and the contents of the op struct.
		libblis_test_fprintf_c( stdout, "--- %s ---\n", op_str );
		libblis_test_fprintf_c( stdout, "\n" );
		libblis_test_output_op_struct( stdout, op, op_str );

		// Also output to a matlab file if requested (and successfully opened).
		if ( output_stream )
		{
			// For file output, we also include the contents of the global
			// parameter struct. We do this per operation so that the parameters
			// are included in each file, whereas we only output it once to
			// stdout (at the end of libblis_test_read_parameter_file()).
			libblis_test_output_params_struct( output_stream, params );

			libblis_test_fprintf_c( output_stream, "--- %s ---\n", op_str );
			libblis_test_fprintf_c( output_stream, "\n" );
			libblis_test_output_op_struct( output_stream, op, op_str );
		}
	}


	// Loop over the requested storage schemes.
	for ( sci = 0; sci < n_store_combos; ++sci )
	//for ( sci = 0; sci < 5; ( sci == 0 || sci == 2 ? sci+=2 : ++sci ) )
	//for ( sci = 0; sci < 5; ( sci == 2 ? sci+=2 : ++sci ) )
	//for ( sci = 3; sci < 8; ( sci == 3 ? sci+=2 : ++sci ) )
	//for ( sci = 0; sci < 1; ++sci )
	//for ( sci = 7; sci < 8; ++sci )
	{
		// Loop over the requested datatypes.
		for ( dci = 0; dci < n_dt_combos; ++dci )
		//for ( dci = 14; dci < 15; ++dci )
		//for ( dci = 6; dci < 7; dci += 1 )
		//for ( dci = 12; dci < 13; ++dci )
		//for ( dci = 4; dci < 5; ++dci )
		//for ( dci = 8; dci < 9; ++dci )
		//for ( dci = 0; dci < 1; ++dci )
		{
			// We need a datatype to use for induced method related things
			// as well as to decide which set of residual thresholds to use.
			// We must choose the first operand's dt char since that's the
			// only operand we know is guaranteed to exist.
			bli_param_map_char_to_blis_dt( dc_str[dci][0], &datatype );
			dt_check = datatype;

			int has_sp = libblis_test_dt_str_has_sp_char_str( n_operandsp1,
			                                                  dc_str[dci] );
			int has_dp = libblis_test_dt_str_has_dp_char_str( n_operandsp1,
			                                                  dc_str[dci] );
			int has_samep = (has_sp && !has_dp ) ||
			                (has_dp && !has_sp );

			// Notice that we use n_operands here instead of
			// n_operandsp1 since we only want to chars for the
			// storage datatypes of the matrix operands, not the
			// computation precision char.
			int has_cd_only =
			!libblis_test_dt_str_has_rd_char_str( n_operands,
			                                      dc_str[dci] );

			if ( has_sp )
			{
				// If any of the operands are single precision, ensure that
				// dt_check is also single precision so that the residual is
				// compared to datatype-appropriate thresholds.
				dt_check = bli_dt_proj_to_single_prec( datatype );
			}

			// Build a commented column label string.
			libblis_test_build_col_labels_string( params, op, label_str );

			// These statements should only be executed by one thread.
			if ( tdata->id == 0 )
			{
				// Output the column label string.
				libblis_test_fprintf( stdout, "%s\n", label_str );

				// Also output to a matlab file if requested (and successfully
				// opened).
				if ( output_stream )
					libblis_test_fprintf( output_stream, "%s\n", label_str );
			}

			// Start by assuming we will only test native execution.
			ind_t ind_first = BLIS_NAT;
			dim_t ind_last  = BLIS_NAT;

			// If the operation is level-3, and all operand domains are complex,
			// then we iterate over all induced methods.
			if ( bli_opid_is_level3( op->opid ) && has_cd_only )
				ind_first = 0;

			// Loop over induced methods (or just BLIS_NAT).
			for ( indi = ind_first; indi <= ind_last; ++indi )
			{
				// If the current datatype is real, OR if the current
				// induced method is implemented (for the operation
				// being tested) AND it was requested, then we enable
				// ONLY that method and proceed. Otherwise, we skip the
				// current method and go to the next method.
				if ( bli_is_real( datatype ) ) { ; }
				else if ( bli_ind_oper_is_impl( op->opid, indi ) &&
				          params->ind_enable[ indi ] == 1 )
				{
					// If the current induced method is 1m, make sure that
					// we only proceed for gemm where all operands are stored
					// in the complex domain. (This prevents 1m from being
					// executed on mixed-datatype combinations that contain
					// real domain datatypes.)
					if ( indi == BLIS_1M )
					{
						if      ( op->opid == BLIS_GEMM && has_cd_only ) { ; }
						else if ( has_samep && has_cd_only ) { ; }
						else { continue; }
					}
					else { ; }
				}
				else { continue; }

				bli_ind_oper_enable_only( op->opid, indi, datatype );

				// Query the implementation string associated with the
				// current operation and datatype. If the operation is
				// not level-3, we will always get back the native string.
				ind_str = bli_ind_oper_get_avail_impl_string( op->opid, datatype );

				// Loop over the requested parameter combinations.
				for ( pci = 0; pci < n_param_combos; ++pci )	
				{
					// Loop over the requested problem sizes.
					for ( p_cur = p_first, pi = 1; p_cur <= p_max; p_cur += p_inc, ++pi )
					{
						// Skip this experiment (for this problem size) according to
						// to the counter, number of threads, and thread id.
						if ( tdata->xc % tdata->nt != tdata->id )
						{
							tdata->xc++;
							continue;
						}

						// Call the given experiment function. perf and resid will
						// contain the resulting performance and residual values,
						// respectively.
						f_exp( params,
						       op,
						       iface,
						       dc_str[dci],
						       pc_str[pci],
						       sc_str[sci],
						       p_cur,
						       &perf, &resid );

						// Remove the sign of the residual, if there is one.
						resid = bli_fabs( resid );
						if ( resid == -0.0 ) resid = 0.0;

						// Query the string corresponding to the residual's
						// position relative to the thresholds.
						pass_str = libblis_test_get_string_for_result( resid,
						                                               dt_check,
						                                               thresh );

						// Build a string unique to the operation, datatype combo,
						// parameter combo, and storage combo being tested.
						libblis_test_build_function_string( BLIS_FILEDATA_PREFIX_STR,
						                                    op->opid,
						                                    indi,
						                                    ind_str,
						                                    op_str,
						                                    is_mixed_dt,
						                                    dc_str[dci],
						                                    n_param_combos,
						                                    pc_str[pci],
						                                    sc_str[sci],
						                                    funcname_str );

						// Compute the number of spaces we have left to fill given
						// length of our operation's name.
						n_spaces = MAX_FUNC_STRING_LENGTH - strlen( funcname_str );
						fill_string_with_n_spaces( blank_str, n_spaces );

						// Print all dimensions to a single string.
						libblis_test_build_dims_string( op, p_cur, dims_str );

						// Count the number of dimensions that were printed to the string.
						n_dims_print = libblis_test_get_n_dims_from_string( dims_str );

						// Output the results of the test. Use matlab format if requested.
						// NOTE: Here we use fprintf() over libblis_test_fprintf() so
						// that on POSIX systems the output is not intermingled. If we
						// used libblis_test_fprintf(), we would need to enclose this
						// conditional with the acquisition of a mutex shared among all
						// threads to prevent intermingled output.
						if ( params->output_matlab_format )
						{
							fprintf( stdout,
							         "%s%s( %3u, 1:%u ) = [%s  %7.2lf  %8.2le ]; %c %s\n",
							         funcname_str, blank_str, pi, n_dims_print + 2,
							         dims_str, perf, resid,
							         OUTPUT_COMMENT_CHAR,
							         pass_str );

							// Also output to a file if requested (and successfully
							// opened).
							if ( output_stream )
							fprintf( output_stream,
							         "%s%s( %3u, 1:%u ) = [%s  %7.2lf  %8.2le ]; %c %s\n",
							         funcname_str, blank_str, pi, n_dims_print + 2,
							         dims_str, perf, resid,
							         OUTPUT_COMMENT_CHAR,
							         pass_str );
						}
						else
						{
							fprintf( stdout,
							         "%s%s      %s  %7.2lf   %8.2le   %s\n",
							         funcname_str, blank_str,
							         dims_str, perf, resid,
							         pass_str );

							// Also output to a file if requested (and successfully
							// opened).
							if ( output_stream )
							fprintf( output_stream,
							         "%s%s      %s  %7.2lf   %8.2le   %s\n",
							         funcname_str, blank_str,
							         dims_str, perf, resid,
							         pass_str );
						}

						// If we need to check whether to do something on failure,
						// do so now.
						if ( reaction_to_failure == ON_FAILURE_SLEEP_CHAR )
						{
							if ( strstr( pass_str, BLIS_TEST_FAIL_STRING ) == pass_str )
								libblis_test_sleep();
						}
						else if ( reaction_to_failure == ON_FAILURE_ABORT_CHAR )
						{
							if ( strstr( pass_str, BLIS_TEST_FAIL_STRING ) == pass_str )
								libblis_test_abort();
						}

						// Increment the experiment counter (regardless of whether
						// the thread executed or skipped the current experiment).
						tdata->xc += 1;
					}
				}
			}

			// Wait for all other threads so that the output stays organized.
			bli_pthread_barrier_wait( tdata->barrier );

			// These statements should only be executed by one thread.
			if ( tdata->id == 0 )
			{
				libblis_test_fprintf( stdout, "\n" );

				if ( output_stream )
					libblis_test_fprintf( output_stream, "\n" );
			}
		}
	}


	// Free the array that stored pointers to the sets of possible parameter
	// chars for each parameter.
	free( chars_for_param );

	// Free the parameter combination strings and then the master pointer.
	for ( pci = 0; pci < n_param_combos; ++pci )
		free( pc_str[pci] );
	free( pc_str );

	// Free the storage combination strings and then the master pointer.
	for ( sci = 0; sci < n_store_combos; ++sci )
		free( sc_str[sci] );
	free( sc_str );

	// Free some auxiliary arrays used by the mixed-domain/mixed-precision
	// datatype-handling logic.
	free( chars_for_rddt );
	free( chars_for_cddt );
	free( chars_for_spdt );
	free( chars_for_dpdt );

	// Free the datatype combination strings and then the master pointer.
	for ( dci = 0; dci < n_dt_combos; ++dci )
		free( dc_str[dci] );
	free( dc_str );


	// If the file was opened (successfully), close the output stream.
	if ( output_stream )
		libblis_test_fclose_ofile( output_stream );


	// Mark this operation as done.
	if ( tdata->id == 0 )
		op->test_done = TRUE;

	// Wait here so that all threads know we are done
	bli_pthread_barrier_wait( tdata->barrier );
}



void libblis_test_build_function_string
     (
       char*        prefix_str,
       opid_t       opid,
       ind_t        method,
       char*        ind_str,
       char*        op_str,
       unsigned int is_mixed_dt,
       char*        dc_str,
       unsigned int n_param_combos,
       char*        pc_str,
       char*        sc_str,
       char*        funcname_str
     )
{
	// We only print the full datatype combination string if is_mixed_dt
	// is set and the operation is gemm. Otherwise, we print only
	// the first char (since they are all the same).
	if ( is_mixed_dt == TRUE && opid == BLIS_GEMM )
		sprintf( funcname_str, "%s_%s%s", prefix_str, dc_str, op_str );
	else
		sprintf( funcname_str, "%s_%c%s", prefix_str, dc_str[0], op_str );

	// If the method is non-native (ie: induced), append a string
	// identifying the induced method.
	if ( method != BLIS_NAT )
		sprintf( &funcname_str[strlen(funcname_str)], "%s", ind_str );

	// We check the string length of pc_str in case the user is running an
	// operation that has parameters (and thus generally more than one
	// parameter combination), but has fixed all parameters in the input
	// file, which would result in n_param_combos to equal one. This way,
	// the function string contains the parameter string associated with
	// the parameters that were fixed.
	if ( n_param_combos > 1 || strlen(pc_str) > 0 )
		sprintf( &funcname_str[strlen(funcname_str)], "_%s_%s", pc_str, sc_str );
	else
		sprintf( &funcname_str[strlen(funcname_str)], "_%s", sc_str );

	if ( strlen( funcname_str ) > MAX_FUNC_STRING_LENGTH )
		libblis_test_printf_error( "Function name string length (%d) exceeds maximum (%d).\n",
		                           strlen( funcname_str ), MAX_FUNC_STRING_LENGTH );
		
}


void libblis_test_build_dims_string( test_op_t* op,
                                     dim_t      p_cur,
                                     char*      dims_str )
{
	unsigned int i;

	// For level-1f experiments with fusing factors, we grab the fusing
	// factor from the op struct. We do something similar for micro-kernel
	// calls.
	if      ( op->dimset == BLIS_TEST_DIMS_MF )
	{
		//sprintf( &dims_str[strlen(dims_str)], " %5u %5u",
		sprintf( dims_str, " %5u %5u",
		         ( unsigned int )
		         libblis_test_get_dim_from_prob_size( op->dim_spec[0],
		                                              p_cur ),
		         ( unsigned int ) op->dim_aux[0] );
	}
	else if ( op->dimset == BLIS_TEST_DIMS_K )
	{
		//sprintf( &dims_str[strlen(dims_str)], " %5u %5u %5u",
		sprintf( dims_str, " %5u %5u %5u",
		         ( unsigned int ) op->dim_aux[0],
		         ( unsigned int ) op->dim_aux[1],
	             ( unsigned int )
		         libblis_test_get_dim_from_prob_size( op->dim_spec[0],
		                                              p_cur ) );
	}
	else if ( op->dimset == BLIS_TEST_NO_DIMS )
	{
		//sprintf( &dims_str[strlen(dims_str)], " %5u %5u",
		sprintf( dims_str, " %5u %5u",
		         ( unsigned int ) op->dim_aux[0],
		         ( unsigned int ) op->dim_aux[1] );
	}
	else // For all other operations, we just use the dim_spec[] values
	     // and the current problem size.
	{
		// Initialize the string as empty.
		sprintf( dims_str, "%s", "" );

		// Print all dimensions to a single string.
		for ( i = 0; i < op->n_dims; ++i )
		{
			sprintf( &dims_str[strlen(dims_str)], " %5u",
			         ( unsigned int )
			         libblis_test_get_dim_from_prob_size( op->dim_spec[i],
			                                              p_cur ) );
		}
	}
}


// % dtoper_params_storage                       m     n     k   gflops  resid       result
void libblis_test_build_col_labels_string( test_params_t* params, test_op_t* op, char* l_str )
{
	unsigned int n_spaces;
	char         blank_str[64];

	strcpy( l_str, "" );

	if ( op->n_params > 0 )
	{
		sprintf( &l_str[strlen(l_str)], "%c %s_%s", OUTPUT_COMMENT_CHAR,
		                                            BLIS_FILEDATA_PREFIX_STR,
		                                            "<dt><op>_<params>_<stor>" );
	}
	else // ( n_params == 0 )
	{
		sprintf( &l_str[strlen(l_str)], "%c %s_%s", OUTPUT_COMMENT_CHAR,
		                                            BLIS_FILEDATA_PREFIX_STR,
		                                            "<dt><op>_<stor>         " );
	}

	if ( params->output_matlab_format ) n_spaces = 11;
	else                                n_spaces = 1;

	fill_string_with_n_spaces( blank_str, n_spaces );

	sprintf( &l_str[strlen(l_str)], "%s", blank_str );

	if ( op->dimset == BLIS_TEST_DIMS_MNK ||
	     op->dimset == BLIS_TEST_DIMS_MN  ||
	     op->dimset == BLIS_TEST_DIMS_MK  ||
	     op->dimset == BLIS_TEST_DIMS_M   ||
	     op->dimset == BLIS_TEST_DIMS_K   ||
	     op->dimset == BLIS_TEST_DIMS_MF  ||
	     op->dimset == BLIS_TEST_NO_DIMS  )
		sprintf( &l_str[strlen(l_str)], " %5s", "m" );

	if ( op->dimset == BLIS_TEST_DIMS_MNK ||
	     op->dimset == BLIS_TEST_DIMS_MN  ||
	     op->dimset == BLIS_TEST_DIMS_K   ||
	     op->dimset == BLIS_TEST_DIMS_MF  ||
	     op->dimset == BLIS_TEST_NO_DIMS  )
		sprintf( &l_str[strlen(l_str)], " %5s", "n" );

	if ( op->dimset == BLIS_TEST_DIMS_MNK ||
	     op->dimset == BLIS_TEST_DIMS_MK  ||
	     op->dimset == BLIS_TEST_DIMS_K   )
		sprintf( &l_str[strlen(l_str)], " %5s", "k" );

	sprintf( &l_str[strlen(l_str)], "%s", "   gflops   resid      result" );
}



void libblis_test_build_filename_string( char*        prefix_str,
                                         char*        op_str,
                                         char*        funcname_str )
{
	sprintf( funcname_str, "%s_%s.m", prefix_str, op_str );
}



void fill_string_with_n_spaces( char* str, unsigned int n_spaces )
{
	unsigned int i;

	// Initialze to empty string in case n_spaces == 0.
	sprintf( str, "%s", "" );

	for ( i = 0; i < n_spaces; ++i )
		sprintf( &str[i], " " );
}



void libblis_test_mobj_create( test_params_t* params, num_t dt, trans_t trans, char storage, dim_t m, dim_t n, obj_t* a )
{
	dim_t  gs        = params->gs_spacing;
	bool   alignment = params->alignment;
	siz_t  elem_size = bli_dt_size( dt );
	dim_t  m_trans   = m;
	dim_t  n_trans   = n;
	dim_t  rs        = 1; // Initialization avoids a compiler warning.
	dim_t  cs        = 1; // Initialization avoids a compiler warning.
	
	// Apply the trans parameter to the dimensions (if needed).
	bli_set_dims_with_trans( trans, m, n, &m_trans, &n_trans );

	// Compute unaligned strides according to the storage case encoded in
	// the storage char, and then align the leading dimension if alignment
	// was requested.
	if      ( storage == 'c' )
	{
		rs = 1;
		cs = m_trans;

		if ( alignment )
			cs = bli_align_dim_to_size( cs, elem_size,
			                            BLIS_HEAP_STRIDE_ALIGN_SIZE );
	}
	else if ( storage == 'r' )
	{
		rs = n_trans;
		cs = 1;

		if ( alignment )
			rs = bli_align_dim_to_size( rs, elem_size,
			                            BLIS_HEAP_STRIDE_ALIGN_SIZE );
	}
	else if ( storage == 'g' )
	{
		// We apply (arbitrarily) a column tilt, instead of a row tilt, to
		// all general stride cases.
		rs = gs;
		cs = gs * m_trans;

		if ( alignment )
			cs = bli_align_dim_to_size( cs, elem_size,
			                            BLIS_HEAP_STRIDE_ALIGN_SIZE );
	}
	else
	{
		libblis_test_printf_error( "Invalid storage character: %c\n", storage );
	}

	// Create the object using the dimensions and strides computed above.
	bli_obj_create( dt, m_trans, n_trans, rs, cs, a );
}



#if 0
cntl_t* libblis_test_pobj_create( bszid_t bmult_id_m, bszid_t bmult_id_n, invdiag_t inv_diag, pack_t pack_schema, packbuf_t pack_buf, obj_t* a, obj_t* p, cntx_t* cntx )
{
	bool   does_inv_diag;
	rntm_t rntm;

	if ( inv_diag == BLIS_NO_INVERT_DIAG ) does_inv_diag = FALSE;
	else                                   does_inv_diag = TRUE;

	// Create a control tree node for the packing operation.
	cntl_t* cntl = bli_packm_cntl_create_node
	(
	  NULL, // we don't need the small block allocator from the runtime.
	  NULL, // func ptr is not referenced b/c we don't call via l3 _int().
	  bli_packm_blk_var1,
	  bmult_id_m,
	  bmult_id_n,
	  does_inv_diag,
	  FALSE,
	  FALSE,
	  pack_schema,
	  pack_buf,
	  NULL  // no child node needed
	);

	// Initialize a local-to-BLIS rntm_t. This is simply so we have something
	// to pass into bli_l3_packm(). The function doesn't (currently) use the
	// runtime object, and even if it did, one with default values would work
	// fine here.
	bli_rntm_init( &rntm );

	// Pack the contents of A to P.
	bli_l3_packm( a, p, cntx, &rntm, cntl, &BLIS_PACKM_SINGLE_THREADED );

	// Return the control tree pointer so the caller can free the cntl_t and its
	// mem_t entry later on.
	return cntl;
}
#endif


void libblis_test_vobj_create( test_params_t* params, num_t dt, char storage, dim_t m, obj_t* x )
{
	dim_t gs = params->gs_spacing;

	// Column vector (unit stride)
	if      ( storage == 'c' ) bli_obj_create( dt, m, 1,  1,  m,    x );
	// Row vector (unit stride)
	else if ( storage == 'r' ) bli_obj_create( dt, 1, m,  m,  1,    x );
	// Column vector (non-unit stride)
	else if ( storage == 'j' ) bli_obj_create( dt, m, 1,  gs, gs*m, x );
	// Row vector (non-unit stride)
	else if ( storage == 'i' ) bli_obj_create( dt, 1, m,  gs*m, gs, x );
	else
	{
		libblis_test_printf_error( "Invalid storage character: %c\n", storage );
	}
}



void libblis_test_vobj_randomize( test_params_t* params, bool normalize, obj_t* x )
{
	if ( params->rand_method == BLIS_TEST_RAND_REAL_VALUES )
		bli_randv( x );
	else // if ( params->rand_method == BLIS_TEST_RAND_NARROW_POW2 )
		bli_randnv( x );

	if ( normalize )
	{
		num_t dt   = bli_obj_dt( x );
		num_t dt_r = bli_obj_dt_proj_to_real( x );
		obj_t kappa;
		obj_t kappa_r;

		bli_obj_scalar_init_detached( dt,   &kappa );
		bli_obj_scalar_init_detached( dt_r, &kappa_r );

		// Normalize vector elements. The following code ensures that we
		// always invert-scale by whole power of two.
		bli_normfv( x, &kappa_r );
		libblis_test_ceil_pow2( &kappa_r );
		bli_copysc( &kappa_r, &kappa );
		bli_invertsc( &kappa );
		bli_scalv( &kappa, x );
	}
}



void libblis_test_mobj_randomize( test_params_t* params, bool normalize, obj_t* a )
{
	if ( params->rand_method == BLIS_TEST_RAND_REAL_VALUES )
		bli_randm( a );
	else // if ( params->rand_method == BLIS_TEST_RAND_NARROW_POW2 )
		bli_randnm( a );

	if ( normalize )
	{
#if 0
		num_t dt      = bli_obj_dt( a );
		dim_t max_m_n = bli_obj_max_dim( a );
		obj_t kappa;

		bli_obj_scalar_init_detached( dt, &kappa );

		// Normalize vector elements by maximum matrix dimension.
		bli_setsc( 1.0/( double )max_m_n, 0.0, &kappa );
		bli_scalm( &kappa, a );
#endif
		num_t dt   = bli_obj_dt( a );
		num_t dt_r = bli_obj_dt_proj_to_real( a );
		obj_t kappa;
		obj_t kappa_r;

		bli_obj_scalar_init_detached( dt,   &kappa );
		bli_obj_scalar_init_detached( dt_r, &kappa_r );

		// Normalize matrix elements.
		bli_norm1m( a, &kappa_r );
		libblis_test_ceil_pow2( &kappa_r );
		bli_copysc( &kappa_r, &kappa );
		bli_invertsc( &kappa );
		bli_scalm( &kappa, a );
	}
}



void libblis_test_ceil_pow2( obj_t* alpha )
{
	double alpha_r;
	double alpha_i;

	bli_getsc( alpha, &alpha_r, &alpha_i );

	alpha_r = pow( 2.0, ceil( log2( alpha_r ) ) );

	bli_setsc( alpha_r, alpha_i, alpha );
}



void libblis_test_mobj_load_diag( test_params_t* params, obj_t* a )
{
	// We assume that all elements of a were intialized on interval [-1,1].

	// Load the diagonal by 2.0.
	bli_shiftd( &BLIS_TWO, a );
}



void libblis_test_init_strings( void )
{
	strcpy( libblis_test_pass_string, BLIS_TEST_PASS_STRING );
	strcpy( libblis_test_warn_string, BLIS_TEST_WARN_STRING );
	strcpy( libblis_test_fail_string, BLIS_TEST_FAIL_STRING );

	strcpy( libblis_test_param_chars[BLIS_TEST_PARAM_SIDE],   BLIS_TEST_PARAM_SIDE_CHARS );
	strcpy( libblis_test_param_chars[BLIS_TEST_PARAM_UPLO],   BLIS_TEST_PARAM_UPLO_CHARS );
	strcpy( libblis_test_param_chars[BLIS_TEST_PARAM_UPLODE], BLIS_TEST_PARAM_UPLODE_CHARS );
	strcpy( libblis_test_param_chars[BLIS_TEST_PARAM_TRANS],  BLIS_TEST_PARAM_TRANS_CHARS );
	strcpy( libblis_test_param_chars[BLIS_TEST_PARAM_CONJ],   BLIS_TEST_PARAM_CONJ_CHARS );
	strcpy( libblis_test_param_chars[BLIS_TEST_PARAM_DIAG],   BLIS_TEST_PARAM_DIAG_CHARS );

	strcpy( libblis_test_store_chars[BLIS_TEST_MATRIX_OPERAND], BLIS_TEST_MSTORE_CHARS );
	strcpy( libblis_test_store_chars[BLIS_TEST_VECTOR_OPERAND], BLIS_TEST_VSTORE_CHARS );
}



void libblis_test_sleep( void )
{
	int i;

	libblis_test_printf_infoc( "Resuming in " );
	for ( i = SECONDS_TO_SLEEP; i > 0; --i )
	{
		libblis_test_printf_info( "%d ", i );
		bli_sleep(1);
	}
	libblis_test_printf_info( "\n" );
}



void libblis_test_abort( void )
{
	abort();
}



void libblis_test_fopen_ofile( char* op_str, iface_t iface, FILE** output_stream )
{
	char filename_str[ MAX_FILENAME_LENGTH ];

	if ( iface == BLIS_TEST_MT_FRONT_END )
		bli_check_error_code( BLIS_NOT_YET_IMPLEMENTED );

	// Construct a filename string for the current operation.
	libblis_test_build_filename_string( BLIS_FILE_PREFIX_STR,
	                                    op_str,
	                                    filename_str );

	// Open the output file (overwriting a previous instance, if it exists)
	// for writing (in binary mode).
	*output_stream = fopen( filename_str, "wb" );

	// Check the output stream and report an error if something went wrong.
	libblis_test_fopen_check_stream( filename_str, *output_stream );
}



void libblis_test_fclose_ofile( FILE* output_stream )
{
	fclose( output_stream );
}



void libblis_test_fopen_check_stream( char* filename_str,
                                      FILE* stream )
{
	// Check for success.
	if ( stream == NULL )
	{
		libblis_test_printf_error( "Failed to open file %s. Check existence (if file is being read), permissions (if file is being overwritten), and/or storage limit.\n",
		                           filename_str );
	}
}



void libblis_test_read_next_line( char* buffer, FILE* input_stream )
{
	char temp[ INPUT_BUFFER_SIZE ];

	// We want to read at least one line, so we use a do-while loop.
	do
	{
		// Read the next line into a temporary buffer and check success.
		if ( fgets( temp, INPUT_BUFFER_SIZE-1, input_stream ) == NULL )
		{
			if ( feof( input_stream ) )
				libblis_test_printf_error( "Error reading input file: encountered unexpected EOF." );
			else
				libblis_test_printf_error( "Error (non-EOF) reading input file." );
		}
	}
    // We continue to read lines into buffer until the line is neither
	// commented nor blank.
	while ( temp[0] == INPUT_COMMENT_CHAR || temp[0] == '\n' ||
	        temp[0] == ' '                || temp[0] == '\t' );

	// Save the string in temp, up to first white space character, into buffer.
	//sscanf( temp, "%s ", buffer );
	strcpy( buffer, temp );

	//printf( "libblis_test_read_next_line() read: %s\n", buffer );
}



void libblis_test_fprintf( FILE* output_stream, char* message, ... )
{
    va_list args;

    // Initialize variable argument environment.
    va_start( args, message );

    // Parse the received message and print its components.
    libblis_test_parse_message( output_stream, message, args );

    // Shutdown variable argument environment and clean up stack.
    va_end( args );

	// Flush the output stream.
    fflush( output_stream );
}



void libblis_test_fprintf_c( FILE* output_stream, char* message, ... )
{
    va_list args;

	fprintf( output_stream, "%c ", OUTPUT_COMMENT_CHAR );

    // Initialize variable argument environment.
    va_start( args, message );

    // Parse the received message and print its components.
    libblis_test_parse_message( output_stream, message, args );

    // Shutdown variable argument environment and clean up stack.
    va_end( args );

	// Flush the output stream.
    fflush( output_stream );
}



void libblis_test_printf_info( char* message, ... )
{
	FILE*   output_stream = stdout;
    va_list args;

    // Initialize variable argument environment.
    va_start( args, message );

    // Parse the received message and print its components.
    libblis_test_parse_message( output_stream, message, args );

    // Shutdown variable argument environment and clean up stack.
    va_end( args );

	// Flush the output stream.
    fflush( output_stream );
}



void libblis_test_printf_infoc( char* message, ... )
{
	FILE*   output_stream = stdout;
    va_list args;

	fprintf( output_stream, "%c ", OUTPUT_COMMENT_CHAR );

    // Initialize variable argument environment.
    va_start( args, message );

    // Parse the received message and print its components.
    libblis_test_parse_message( output_stream, message, args );

    // Shutdown variable argument environment and clean up stack.
    va_end( args );

	// Flush the output stream.
    fflush( output_stream );
}



void libblis_test_printf_error( char* message, ... )
{
	FILE*   output_stream = stderr;
    va_list args;

    fprintf( output_stream, "%s: *** error ***: ", libblis_test_binary_name );

    // Initialize variable argument environment.
    va_start( args, message );

    // Parse the received message and print its components.
    libblis_test_parse_message( output_stream, message, args );

    // Shutdown variable argument environment and clean up stack.
    va_end( args );

	// Flush the output stream.
    fflush( output_stream );

	// Exit.
	exit(1);
}



void libblis_test_parse_message( FILE* output_stream, char* message, va_list args )
{
	int           c, cf;
	char          format_spec[8];
	unsigned int  the_uint;
	int           the_int;
	double        the_double;
	char*         the_string;
	char          the_char;

	// Begin looping over message to insert variables wherever there are 
	// format specifiers.
	for ( c = 0; message[c] != '\0'; )
	{
		if ( message[c] != '%' )
		{
			fprintf( output_stream, "%c", message[c] );
			c += 1;
		}
		else if ( message[c] == '%' && message[c+1] == '%' ) // handle escaped '%' chars.
		{
			fprintf( output_stream, "%c", message[c] );
			c += 2;
		}
		else
		{
			// Save the format string if there is one.
			format_spec[0] = '%';
			for ( c += 1, cf = 1; strchr( "udefsc", message[c] ) == NULL; ++c, ++cf )
			{
				format_spec[cf] = message[c];
			}

			// Add the final type specifier, and null-terminate the string.
			format_spec[cf] = message[c];
			format_spec[cf+1] = '\0';

			// Switch based on type, since we can't predict what will
			// va_args() will return.
			switch ( message[c] )
			{
				case 'u':
				the_uint = va_arg( args, unsigned int );
				fprintf( output_stream, format_spec, the_uint );
				break;

				case 'd':
				the_int = va_arg( args, int );
				fprintf( output_stream, format_spec, the_int );
				break;

				case 'e':
				the_double = va_arg( args, double );
				fprintf( output_stream, format_spec, the_double );
				break;

				case 'f':
				the_double = va_arg( args, double );
				fprintf( output_stream, format_spec, the_double );
				break;

				case 's':
				the_string = va_arg( args, char* );
				//fprintf( output_stream, "%s", the_string );
				fprintf( output_stream, format_spec, the_string );
				break;

				case 'c':
				the_char = va_arg( args, int );
				fprintf( output_stream, "%c", the_char );
				break;
			}

			// Move to next character past type specifier.
			c += 1;
		}
	}
}



void libblis_test_parse_command_line( int argc, char** argv )
{
	bool     gave_option_g = FALSE;
	bool     gave_option_o = FALSE;
	int      opt;
	char     opt_ch;
	getopt_t state;

	// Copy the binary name to a global string so we can use it later.
	strncpy( libblis_test_binary_name, argv[0], MAX_BINARY_NAME_LENGTH );

	// Initialize the state for running bli_getopt(). Here, 0 is the
	// initial value for opterr, which suppresses error messages.
	bli_getopt_init_state( 0, &state );

	// Process all option arguments until we get a -1, which means we're done.
	while( (opt = bli_getopt( argc, argv, "g:o:", &state )) != -1 )
	{
		// Explicitly typecast opt, which is an int, to a char. (Failing to
		// typecast resulted in at least one user-reported problem whereby
		// opt was being filled with garbage.)
		opt_ch = ( char )opt;

		switch( opt_ch )
		{
			case 'g':
			libblis_test_printf_infoc( "detected -g option; using \"%s\" for parameters filename.\n", state.optarg );
			strncpy( libblis_test_parameters_filename,
			         state.optarg, MAX_FILENAME_LENGTH );
			gave_option_g = TRUE;
			break;

			case 'o':
			libblis_test_printf_infoc( "detected -o option; using \"%s\" for operations filename.\n", state.optarg );
			strncpy( libblis_test_operations_filename,
			         state.optarg, MAX_FILENAME_LENGTH );
			gave_option_o = TRUE;
			break;

			case '?':
			libblis_test_printf_error( "unexpected option '%c' given or missing option argument\n", state.optopt );
			break;

			default:
			libblis_test_printf_error( "unexpected option chararcter returned from getopt: %c\n", opt_ch );
		}
	}

	if ( gave_option_g == FALSE )
	{
		libblis_test_printf_infoc( "no -g option given; defaulting to \"%s\" for parameters filename.\n", PARAMETERS_FILENAME );

		// Copy default parameters filename into its global string.
		strncpy( libblis_test_parameters_filename,
		         PARAMETERS_FILENAME, MAX_FILENAME_LENGTH );
	}

	if ( gave_option_o == FALSE )
	{
		libblis_test_printf_infoc( "no -o option given; defaulting to \"%s\" for operations filename.\n", OPERATIONS_FILENAME );

		// Copy default operations filename into its global string.
		strncpy( libblis_test_operations_filename,
		         OPERATIONS_FILENAME, MAX_FILENAME_LENGTH );
	}

	// If there are still arguments remaining after getopt() processing is
	// complete, print an error.
	if ( state.optind < argc )
	{
		libblis_test_printf_error( "Encountered unexpected non-option argument: %s\n", argv[ state.optind ] );
	}
}



void libblis_test_check_empty_problem( obj_t* c, double* perf, double* resid )
{
	if ( bli_obj_has_zero_dim( c ) )
	{
		*perf  = 0.0;
		*resid = 0.0;
	}
}



int libblis_test_op_is_disabled( test_op_t* op )
{
	int r_val;

	// If there was at least one individual override, then an op test is
	// disabled if it is NOT equal to ENABLE_ONLY. If there were no
	// individual overrides, then an op test is disabled if it is equal
	// to DISABLE.
	if ( op->ops->indiv_over == TRUE )
	{
		if ( op->op_switch != ENABLE_ONLY ) r_val = TRUE;
		else                                r_val = FALSE;
	}
	else // if ( op->ops->indiv_over == FALSE )
	{
		if ( op->op_switch == DISABLE ) r_val = TRUE;
		else                            r_val = FALSE;
	}

	return r_val;
}

bool libblis_test_op_is_done( test_op_t* op )
{
	return op->test_done;
}

int libblis_test_util_is_disabled( test_op_t* op )
{
	if ( op->ops->util_over == DISABLE ) return TRUE;
	else                                 return FALSE;
}

int libblis_test_l1v_is_disabled( test_op_t* op )
{
	if ( op->ops->l1v_over == DISABLE ) return TRUE;
	else                                return FALSE;
}

int libblis_test_l1m_is_disabled( test_op_t* op )
{
	if ( op->ops->l1m_over == DISABLE ) return TRUE;
	else                                return FALSE;
}

int libblis_test_l1f_is_disabled( test_op_t* op )
{
	if ( op->ops->l1f_over == DISABLE ) return TRUE;
	else                                return FALSE;
}

int libblis_test_l2_is_disabled( test_op_t* op )
{
	if ( op->ops->l2_over == DISABLE ) return TRUE;
	else                               return FALSE;
}

int libblis_test_l3ukr_is_disabled( test_op_t* op )
{
	if ( op->ops->l3ukr_over == DISABLE ) return TRUE;
	else                                  return FALSE;
}

int libblis_test_l3_is_disabled( test_op_t* op )
{
	if ( op->ops->l3_over == DISABLE ) return TRUE;
	else                               return FALSE;
}

// ---

int libblis_test_dt_str_has_sp_char( test_params_t* params )
{
	return libblis_test_dt_str_has_sp_char_str( params->n_datatypes,
	                                            params->datatype_char );
}

int libblis_test_dt_str_has_sp_char_str( int n, char* str )
{
	for ( int i = 0; i < n; ++i )
	{
		if ( str[i] == 's' ||
		     str[i] == 'c' ) return TRUE;
	}

	return FALSE;
}

int libblis_test_dt_str_has_dp_char( test_params_t* params )
{
	return libblis_test_dt_str_has_dp_char_str( params->n_datatypes,
	                                            params->datatype_char );
}

int libblis_test_dt_str_has_dp_char_str( int n, char* str )
{
	for ( int i = 0; i < n; ++i )
	{
		if ( str[i] == 'd' ||
		     str[i] == 'z' ) return TRUE;
	}

	return FALSE;
}

// ---

int libblis_test_dt_str_has_rd_char( test_params_t* params )
{
	return libblis_test_dt_str_has_rd_char_str( params->n_datatypes,
	                                            params->datatype_char );
}

int libblis_test_dt_str_has_rd_char_str( int n, char* str )
{
	for ( int i = 0; i < n; ++i )
	{
		if ( str[i] == 's' ||
		     str[i] == 'd' ) return TRUE;
	}

	return FALSE;
}

int libblis_test_dt_str_has_cd_char( test_params_t* params )
{
	return libblis_test_dt_str_has_cd_char_str( params->n_datatypes,
	                                            params->datatype_char );
}

int libblis_test_dt_str_has_cd_char_str( int n, char* str )
{
	for ( int i = 0; i < n; ++i )
	{
		if ( str[i] == 'c' ||
		     str[i] == 'z' ) return TRUE;
	}

	return FALSE;
}

// ---

unsigned int libblis_test_count_combos
     (
       unsigned int n_operands,
       char*        spec_str,
       char**       char_sets
     )
{
	unsigned int n_combos = 1;

	for ( int i = 0; i < n_operands; ++i )
	{
		if ( spec_str[i] == '?' )
			n_combos *= strlen( char_sets[i] );
	}

	return n_combos;
}

char libblis_test_proj_dtchar_to_precchar( char dt_char )
{
	char r_val = dt_char;

	if      ( r_val == 'c' ) r_val = 's';
	else if ( r_val == 'z' ) r_val = 'd';

	return r_val;
}

