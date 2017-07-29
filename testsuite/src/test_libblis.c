/*

   BLIS    
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2014, The University of Texas at Austin

   Redistribution and use in source and binary forms, with or without
   modification, are permitted provided that the following conditions are
   met:
    - Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.
    - Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.
    - Neither the name of The University of Texas at Austin nor the names
      of its contributors may be used to endorse or promote products
      derived from this software without specific prior written permission.

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



int main( int argc, char** argv )
{
	test_params_t params;
	test_ops_t    ops;

	// Initialize libblis.
	bli_init();

	// Initialize some strings.
	libblis_test_init_strings();

	// Parse the command line parameters.
	libblis_test_parse_command_line( argc, argv );

	// Read the global parameters file.
	libblis_test_read_params_file( libblis_test_parameters_filename, &params );

	// Read the operations parameter file.
	libblis_test_read_ops_file( libblis_test_operations_filename, &ops );
	
	// Test the utility operations.
	libblis_test_utility_ops( &params, &ops );

	// Test the level-1v operations.
	libblis_test_level1v_ops( &params, &ops );

	// Test the level-1m operations.
	libblis_test_level1m_ops( &params, &ops );

	// Test the level-1f operations.
	libblis_test_level1f_ops( &params, &ops );

	// Test the level-2 operations.
	libblis_test_level2_ops( &params, &ops );

	// Test the level-3 micro-kernels.
	libblis_test_level3_ukrs( &params, &ops );

	// Test the level-3 operations.
	libblis_test_level3_ops( &params, &ops );

	// Finalize libblis.
	bli_finalize();

	// Return peacefully.
	return 0;
}



void libblis_test_utility_ops( test_params_t* params, test_ops_t* ops )
{
	libblis_test_randv( params, &(ops->randv) );
	libblis_test_randm( params, &(ops->randm) );
}



void libblis_test_level1v_ops( test_params_t* params, test_ops_t* ops )
{
	libblis_test_addv( params, &(ops->addv) );
	libblis_test_amaxv( params, &(ops->amaxv) );
	libblis_test_axpbyv( params, &(ops->axpbyv) );
	libblis_test_axpyv( params, &(ops->axpyv) );
	libblis_test_copyv( params, &(ops->copyv) );
	libblis_test_dotv( params, &(ops->dotv) );
	libblis_test_dotxv( params, &(ops->dotxv) );
	libblis_test_normfv( params, &(ops->normfv) );
	libblis_test_scalv( params, &(ops->scalv) );
	libblis_test_scal2v( params, &(ops->scal2v) );
	libblis_test_setv( params, &(ops->setv) );
	libblis_test_subv( params, &(ops->subv) );
	libblis_test_xpbyv( params, &(ops->xpbyv) );
}



void libblis_test_level1m_ops( test_params_t* params, test_ops_t* ops )
{
	libblis_test_addm( params, &(ops->addm) );
	libblis_test_axpym( params, &(ops->axpym) );
	libblis_test_copym( params, &(ops->copym) );
	libblis_test_normfm( params, &(ops->normfm) );
	libblis_test_scalm( params, &(ops->scalm) );
	libblis_test_scal2m( params, &(ops->scal2m) );
	libblis_test_setm( params, &(ops->setm) );
	libblis_test_subm( params, &(ops->subm) );
}



void libblis_test_level1f_ops( test_params_t* params, test_ops_t* ops )
{
	libblis_test_axpy2v( params, &(ops->axpy2v) );
	libblis_test_dotaxpyv( params, &(ops->dotaxpyv) );
	libblis_test_axpyf( params, &(ops->axpyf) );
	libblis_test_dotxf( params, &(ops->dotxf) );
	libblis_test_dotxaxpyf( params, &(ops->dotxaxpyf) );
}



void libblis_test_level2_ops( test_params_t* params, test_ops_t* ops )
{
	libblis_test_gemv( params, &(ops->gemv) );
	libblis_test_ger( params, &(ops->ger) );
	libblis_test_hemv( params, &(ops->hemv) );
	libblis_test_her( params, &(ops->her) );
	libblis_test_her2( params, &(ops->her2) );
	libblis_test_symv( params, &(ops->symv) );
	libblis_test_syr( params, &(ops->syr) );
	libblis_test_syr2( params, &(ops->syr2) );
	libblis_test_trmv( params, &(ops->trmv) );
	libblis_test_trsv( params, &(ops->trsv) );
}



void libblis_test_level3_ukrs( test_params_t* params, test_ops_t* ops )
{
	libblis_test_gemm_ukr( params, &(ops->gemm_ukr) );
	libblis_test_trsm_ukr( params, &(ops->trsm_ukr) );
	libblis_test_gemmtrsm_ukr( params, &(ops->gemmtrsm_ukr) );
}



void libblis_test_level3_ops( test_params_t* params, test_ops_t* ops )
{
	libblis_test_gemm( params, &(ops->gemm) );
	libblis_test_hemm( params, &(ops->hemm) );
	libblis_test_herk( params, &(ops->herk) );
	libblis_test_her2k( params, &(ops->her2k) );
	libblis_test_symm( params, &(ops->symm) );
	libblis_test_syrk( params, &(ops->syrk) );
	libblis_test_syr2k( params, &(ops->syr2k) );
	libblis_test_trmm( params, &(ops->trmm) );
	libblis_test_trmm3( params, &(ops->trmm3) );
	libblis_test_trsm( params, &(ops->trsm) );
}



void libblis_test_read_ops_file( char* input_filename, test_ops_t* ops )
{
	FILE* input_stream;

	// Attempt to open input file corresponding to input_filename as
	// read-only/binary.
	input_stream = fopen( input_filename, "rb" );
	libblis_test_fopen_check_stream( input_filename, input_stream );

	// Begin reading operations input file.

	//                                            dimensions          n_param   operation

	// Section overrides
	libblis_test_read_section_override( ops, input_stream, &(ops->util_over) );
	libblis_test_read_section_override( ops, input_stream, &(ops->l1v_over) );
	libblis_test_read_section_override( ops, input_stream, &(ops->l1m_over) );
	libblis_test_read_section_override( ops, input_stream, &(ops->l1f_over) );
	libblis_test_read_section_override( ops, input_stream, &(ops->l2_over) );
	libblis_test_read_section_override( ops, input_stream, &(ops->l3ukr_over) );
	libblis_test_read_section_override( ops, input_stream, &(ops->l3_over) );

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
	input_stream = fopen( input_filename, "rb" );
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
		if      ( temp[i] == 's' ) params->datatype[i] = BLIS_FLOAT;
		else if ( temp[i] == 'd' ) params->datatype[i] = BLIS_DOUBLE;
		else if ( temp[i] == 'c' ) params->datatype[i] = BLIS_SCOMPLEX;
		else if ( temp[i] == 'z' ) params->datatype[i] = BLIS_DCOMPLEX;

		params->datatype_char[i] = temp[i];
	}

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

	// Read whether to enable 3m3.
	libblis_test_read_next_line( buffer, input_stream );
	sscanf( buffer, "%u ", &(params->ind_enable[ BLIS_3M3 ]) );

	// Read whether to enable 3m2.
	libblis_test_read_next_line( buffer, input_stream );
	sscanf( buffer, "%u ", &(params->ind_enable[ BLIS_3M2 ]) );

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

	// Read the line for the sequential front-end/micro-kernel interface.
	libblis_test_read_next_line( buffer, input_stream );
	sscanf( buffer, "%d ", &(op->front_seq) );

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

	// Disable operation if requested.
	if ( op->op_switch == DISABLE_ALL )
	{
		op->front_seq = DISABLE;
	}
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
	cntx_t  cntx_local;
	cntx_t  cntx_local_c;
	cntx_t  cntx_local_z;
	cntx_t* cntx   = &cntx_local;
	cntx_t* cntx_c = &cntx_local_c;
	cntx_t* cntx_z = &cntx_local_z;

	// If bli_info_get_int_type_size() returns 32 or 64, the size is forced.
	// Otherwise, the size is chosen automatically. We query the result of
	// that automatic choice via sizeof(gint_t).
/*	
	if ( bli_info_get_int_type_size() == 32 ||
	     bli_info_get_int_type_size() == 64 )
		sprintf( int_type_size_str, "%d", ( int )bli_info_get_int_type_size() );
	else
		sprintf( int_type_size_str, "%d", ( int )sizeof(gint_t) * 8 );
*/
	if ( bli_info_get_int_type_size() == 32 ||
	     bli_info_get_int_type_size() == 64 )
		int_type_size = bli_info_get_int_type_size();
	else
		int_type_size = sizeof(gint_t) * 8;

	// Output some system parameters.
	libblis_test_fprintf_c( os, "\n" );
	libblis_test_fprintf_c( os, "--- BLIS library info -------------------------------------\n" );
	libblis_test_fprintf_c( os, "\n" );
	libblis_test_fprintf_c( os, "version string                 %s\n", bli_info_get_version_str() );
	libblis_test_fprintf_c( os, "\n" );
	libblis_test_fprintf_c( os, "--- BLIS configuration info ---\n" );
	libblis_test_fprintf_c( os, "\n" );
	libblis_test_fprintf_c( os, "integer type size (bits)       %d\n", ( int )int_type_size );
	libblis_test_fprintf_c( os, "\n" );
	libblis_test_fprintf_c( os, "SIMD number of registers       %d\n", ( int )bli_info_get_simd_num_registers() );
	libblis_test_fprintf_c( os, "SIMD size (bytes)              %d\n", ( int )bli_info_get_simd_size() );
	libblis_test_fprintf_c( os, "SIMD alignment (bytes)         %d\n", ( int )bli_info_get_simd_align_size() );
	libblis_test_fprintf_c( os, "Max stack buffer size (bytes)  %d\n", ( int )bli_info_get_stack_buf_max_size() );
	libblis_test_fprintf_c( os, "Page size (bytes)              %d\n", ( int )bli_info_get_page_size() );
	libblis_test_fprintf_c( os, "\n" );
	libblis_test_fprintf_c( os, "memory alignment (bytes)         \n" );
	libblis_test_fprintf_c( os, "  stack address                %d\n", ( int )bli_info_get_stack_buf_align_size() );
	libblis_test_fprintf_c( os, "  obj_t address                %d\n", ( int )bli_info_get_heap_addr_align_size() );
	libblis_test_fprintf_c( os, "  obj_t stride                 %d\n", ( int )bli_info_get_heap_stride_align_size() );
	libblis_test_fprintf_c( os, "  pool block addr              %d\n", ( int )bli_info_get_pool_addr_align_size() );
	libblis_test_fprintf_c( os, "\n" );
	libblis_test_fprintf_c( os, "BLAS compatibility layer         \n" );
	libblis_test_fprintf_c( os, "  enabled?                     %d\n", ( int )bli_info_get_enable_blas2blis() );
	libblis_test_fprintf_c( os, "  integer type size (bits)     %d\n", ( int )bli_info_get_blas2blis_int_type_size() );
	libblis_test_fprintf_c( os, "\n" );
	libblis_test_fprintf_c( os, "CBLAS compatibility layer        \n" );
	libblis_test_fprintf_c( os, "  enabled?                     %d\n", ( int )bli_info_get_enable_cblas() );
	libblis_test_fprintf_c( os, "\n" );
	libblis_test_fprintf_c( os, "floating-point types           s       d       c       z \n" );
	libblis_test_fprintf_c( os, "  sizes (bytes)          %7u %7u %7u %7u\n", sizeof(float),
	                                                                          sizeof(double),
	                                                                          sizeof(scomplex),
	                                                                          sizeof(dcomplex) );
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

	// Initialize a context for the gemm family, assuming native execution.
	// We use BLIS_DOUBLE for the datatype, but the dt argument is actually
	// only used when initializing contexts for induced methods.
	bli_gemmnat_cntx_init( BLIS_DOUBLE, cntx );

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

	bli_gemmnat_cntx_finalize( cntx );

	libblis_test_fprintf_c( os, "--- BLIS induced implementation info ---\n" );
	libblis_test_fprintf_c( os, "\n" );

	for ( im = 0; im < BLIS_NAT; ++im )
	{
	if ( params->ind_enable[ im ] == 0 ) continue;

	bli_ind_oper_enable_only( BLIS_GEMM, im, BLIS_SCOMPLEX );
	bli_ind_oper_enable_only( BLIS_GEMM, im, BLIS_DCOMPLEX );

	libblis_test_fprintf_c( os, "                               c       z \n" );
	libblis_test_fprintf_c( os, "complex implementation   %7s %7s\n",
	                        bli_ind_oper_get_avail_impl_string( BLIS_GEMM, BLIS_SCOMPLEX ),
	                        bli_ind_oper_get_avail_impl_string( BLIS_GEMM, BLIS_DCOMPLEX ) );
	libblis_test_fprintf_c( os, "\n" );

	bli_gemmind_cntx_init( im, BLIS_SCOMPLEX, cntx_c );
	bli_gemmind_cntx_init( im, BLIS_DCOMPLEX, cntx_z );

	libblis_test_fprintf_c( os, "level-3 blocksizes             c       z \n" );
	libblis_test_fprintf_c( os, "  mc                     %7d %7d\n",
	                        ( int )bli_cntx_get_blksz_def_dt( BLIS_SCOMPLEX, BLIS_MC, cntx_c ),
	                        ( int )bli_cntx_get_blksz_def_dt( BLIS_DCOMPLEX, BLIS_MC, cntx_z ) );
	libblis_test_fprintf_c( os, "  kc                     %7d %7d\n",
	                        ( int )bli_cntx_get_blksz_def_dt( BLIS_SCOMPLEX, BLIS_KC, cntx_c ),
	                        ( int )bli_cntx_get_blksz_def_dt( BLIS_DCOMPLEX, BLIS_KC, cntx_z ) );
	libblis_test_fprintf_c( os, "  nc                     %7d %7d\n",
	                        ( int )bli_cntx_get_blksz_def_dt( BLIS_SCOMPLEX, BLIS_NC, cntx_c ),
	                        ( int )bli_cntx_get_blksz_def_dt( BLIS_DCOMPLEX, BLIS_NC, cntx_z ) );
	libblis_test_fprintf_c( os, "\n" );
	libblis_test_fprintf_c( os, "  mc maximum             %7d %7d\n",
	                        ( int )bli_cntx_get_blksz_max_dt( BLIS_SCOMPLEX, BLIS_MC, cntx_c ),
	                        ( int )bli_cntx_get_blksz_max_dt( BLIS_DCOMPLEX, BLIS_MC, cntx_z ) );
	libblis_test_fprintf_c( os, "  kc maximum             %7d %7d\n",
	                        ( int )bli_cntx_get_blksz_max_dt( BLIS_SCOMPLEX, BLIS_KC, cntx_c ),
	                        ( int )bli_cntx_get_blksz_max_dt( BLIS_DCOMPLEX, BLIS_KC, cntx_z ) );
	libblis_test_fprintf_c( os, "  nc maximum             %7d %7d\n",
	                        ( int )bli_cntx_get_blksz_max_dt( BLIS_SCOMPLEX, BLIS_NC, cntx_c ),
	                        ( int )bli_cntx_get_blksz_max_dt( BLIS_DCOMPLEX, BLIS_NC, cntx_z ) );
	libblis_test_fprintf_c( os, "\n" );
	libblis_test_fprintf_c( os, "  mr                     %7d %7d\n",
	                        ( int )bli_cntx_get_blksz_def_dt( BLIS_SCOMPLEX, BLIS_MR, cntx_c ),
	                        ( int )bli_cntx_get_blksz_def_dt( BLIS_DCOMPLEX, BLIS_MR, cntx_z ) );
	libblis_test_fprintf_c( os, "  nr                     %7d %7d\n",
	                        ( int )bli_cntx_get_blksz_def_dt( BLIS_SCOMPLEX, BLIS_NR, cntx_c ),
	                        ( int )bli_cntx_get_blksz_def_dt( BLIS_DCOMPLEX, BLIS_NR, cntx_z ) );
	libblis_test_fprintf_c( os, "\n" );
	libblis_test_fprintf_c( os, "  mr packdim             %7d %7d\n",
	                        ( int )bli_cntx_get_blksz_max_dt( BLIS_SCOMPLEX, BLIS_MR, cntx_c ),
	                        ( int )bli_cntx_get_blksz_max_dt( BLIS_DCOMPLEX, BLIS_MR, cntx_z ) );
	libblis_test_fprintf_c( os, "  nr packdim             %7d %7d\n",
	                        ( int )bli_cntx_get_blksz_max_dt( BLIS_SCOMPLEX, BLIS_NR, cntx_c ),
	                        ( int )bli_cntx_get_blksz_max_dt( BLIS_DCOMPLEX, BLIS_NR, cntx_z ) );
	libblis_test_fprintf_c( os, "\n" );
	libblis_test_fprintf_c( os, "micro-kernel types             c       z\n" );
	libblis_test_fprintf_c( os, "  gemm                   %7s %7s\n",
	                        bli_info_get_gemm_ukr_impl_string( im, BLIS_SCOMPLEX ),
	                        bli_info_get_gemm_ukr_impl_string( im, BLIS_DCOMPLEX ) );
	libblis_test_fprintf_c( os, "  gemmtrsm_l             %7s %7s\n",
	                        bli_info_get_gemmtrsm_l_ukr_impl_string( im, BLIS_SCOMPLEX ),
	                        bli_info_get_gemmtrsm_l_ukr_impl_string( im, BLIS_DCOMPLEX ) );
	libblis_test_fprintf_c( os, "  gemmtrsm_u             %7s %7s\n",
	                        bli_info_get_gemmtrsm_u_ukr_impl_string( im, BLIS_SCOMPLEX ),
	                        bli_info_get_gemmtrsm_u_ukr_impl_string( im, BLIS_DCOMPLEX ) );
	libblis_test_fprintf_c( os, "  trsm_l                 %7s %7s\n",
	                        bli_info_get_trsm_l_ukr_impl_string( im, BLIS_SCOMPLEX ),
	                        bli_info_get_trsm_l_ukr_impl_string( im, BLIS_DCOMPLEX ) );
	libblis_test_fprintf_c( os, "  trsm_u                 %7s %7s\n",
	                        bli_info_get_trsm_u_ukr_impl_string( im, BLIS_SCOMPLEX ),
	                        bli_info_get_trsm_u_ukr_impl_string( im, BLIS_DCOMPLEX ) );
	libblis_test_fprintf_c( os, "\n" );

	bli_gemmind_cntx_finalize( im, cntx_c );
	bli_gemmind_cntx_finalize( im, cntx_z );
	}

	bli_ind_disable_all();

	// We use hemv's context because we know it is initialized with all of the fields
	// we will be outputing.
	// We use BLIS_DOUBLE for the datatype, but the dt argument is actually
	// only used when initializing contexts for induced methods.
	bli_hemv_cntx_init( BLIS_DOUBLE, cntx );

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

	bli_hemv_cntx_finalize( cntx );

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
	libblis_test_fprintf_c( os, "problem size: first to test  %u\n", params->p_first );
	libblis_test_fprintf_c( os, "problem size: max to test    %u\n", params->p_max );
	libblis_test_fprintf_c( os, "problem size increment       %u\n", params->p_inc );
	libblis_test_fprintf_c( os, "complex implementations        \n" );
	libblis_test_fprintf_c( os, "  3mh?                       %u\n", params->ind_enable[ BLIS_3MH ] );
	libblis_test_fprintf_c( os, "  3m3?                       %u\n", params->ind_enable[ BLIS_3M3 ] );
	libblis_test_fprintf_c( os, "  3m2?                       %u\n", params->ind_enable[ BLIS_3M2 ] );
	libblis_test_fprintf_c( os, "  3m1?                       %u\n", params->ind_enable[ BLIS_3M1 ] );
	libblis_test_fprintf_c( os, "  4mh?                       %u\n", params->ind_enable[ BLIS_4MH ] );
	libblis_test_fprintf_c( os, "  4m1b (4mb)?                %u\n", params->ind_enable[ BLIS_4M1B ] );
	libblis_test_fprintf_c( os, "  4m1a (4m1)?                %u\n", params->ind_enable[ BLIS_4M1A ] );
	libblis_test_fprintf_c( os, "  1m?                        %u\n", params->ind_enable[ BLIS_1M ] );
	libblis_test_fprintf_c( os, "  native?                    %u\n", params->ind_enable[ BLIS_NAT ] );
	libblis_test_fprintf_c( os, "error-checking level         %u\n", params->error_checking_level );
	libblis_test_fprintf_c( os, "reaction to failure          %c\n", params->reaction_to_failure );
	libblis_test_fprintf_c( os, "output in matlab format?     %u\n", params->output_matlab_format );
	libblis_test_fprintf_c( os, "output to stdout AND files?  %u\n", params->output_files );
	libblis_test_fprintf_c( os, "\n" );
	libblis_test_fprintf( os, "\n" );
}



void libblis_test_output_op_struct( FILE* os, test_op_t* op, char* op_str )
{
	dimset_t dimset = op->dimset;

	libblis_test_fprintf_c( os, "test %s seq front-end?    %d\n", op_str, op->front_seq );

	if (      dimset == BLIS_TEST_DIMS_MNK )
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

	if      ( resid > thresh[dt].failwarn ) r_val = libblis_test_fail_string;
	else if ( resid > thresh[dt].warnpass ) r_val = libblis_test_warn_string;
	else                                    r_val = libblis_test_pass_string;

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



void libblis_test_op_driver( test_params_t* params,
                             test_op_t*     op,
                             iface_t        iface,
                             char*          op_str,
                             char*          p_types,
                             char*          o_types,
                             thresh_t*      thresh,
                             void (*f_exp)  (test_params_t*, // params struct
                                             test_op_t*,     // op struct
                                             iface_t,        // iface
                                             num_t,          // datatype (current datatype)
                                             char*,          // pc_str (current param string)
                                             char*,          // sc_str (current storage string)
                                             unsigned int,   // p_cur (current problem size)
                                             double*,        // perf
                                             double* ) )     // residual
{
	unsigned int  n_mstorage          = params->n_mstorage;
	unsigned int  n_vstorage          = params->n_vstorage;
	unsigned int  n_datatypes         = params->n_datatypes;
	unsigned int  p_first             = params->p_first;
	unsigned int  p_max               = params->p_max;
	unsigned int  p_inc               = params->p_inc;
	unsigned int  mix_all_storage     = params->mix_all_storage;
	unsigned int  reaction_to_failure = params->reaction_to_failure;

	num_t         datatype;
	char          dt_char;

	char*         p_spec_str;
	unsigned int  n_params;
	char**        chars_for_param;
	unsigned int  n_param_combos;
	char**        pc_str;

	char          s_spec_str[ MAX_NUM_OPERANDS + 1 ];
	unsigned int  n_operands;
	char**        chars_for_storage;
	unsigned int  n_store_combos;
	char**        sc_str;

	unsigned int  p_cur, pi;
	unsigned int  dt, indi, pci, sci, i, j, o;

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
	for ( i = 0, n_param_combos = 1; i < n_params; ++i )
	{
		if ( p_spec_str[i] == '?' )
			n_param_combos *= strlen( chars_for_param[i] );
	}

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

	// Determine the total number of storage schemes.
	if ( mix_all_storage )
	{
		// Fill storage specification string with wildcard chars.
		for ( i = 0; i < n_operands; ++i )
			s_spec_str[i] = '?';
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
		for ( i = 0, n_store_combos = 1; i < n_operands; ++i )
		{
			if ( s_spec_str[i] == '?' )
				n_store_combos *= strlen( chars_for_storage[i] );
		}

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


	// Loop over the requested storage schemes.
	for ( sci = 0; sci < n_store_combos; ++sci )
	{
		// Loop over the requested datatypes.
		for ( dt = 0; dt < n_datatypes; ++dt )
		{
			datatype = params->datatype[dt];
			dt_char  = params->datatype_char[dt];

			// Build a commented column label string.
			libblis_test_build_col_labels_string( params, op, label_str );

			// Output the column label string.
			libblis_test_fprintf( stdout, "%s\n", label_str );

			// Also output to a matlab file if requested (and successfully
			// opened).
			if ( output_stream )
				libblis_test_fprintf( output_stream, "%s\n", label_str );

			// Start by assuming we will only test native execution.
			ind_t ind_first = BLIS_NAT;
			dim_t ind_last  = BLIS_NAT;

			// If the operation is level-3, and the datatype is complex,
			// then we iterate over all induced methods.
			if ( bli_opid_is_level3( op->opid ) &&
			     bli_is_complex( datatype ) ) ind_first = 0;

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
				          params->ind_enable[ indi ] == 1 ) { ; }
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
						f_exp( params,
						       op,
						       iface,
						       datatype,
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
						                                               datatype,
						                                               thresh );

						// Build a string unique to the operation, datatype,
						// parameter combination, and storage combination being
						// tested.
						libblis_test_build_function_string( BLIS_FILEDATA_PREFIX_STR,
						                                    indi,
						                                    ind_str,
						                                    op_str,
						                                    dt_char,
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
						if ( params->output_matlab_format )
						{
							libblis_test_fprintf( stdout,
							                      "%s%s( %3u, 1:%u ) = [%s  %7.3lf  %8.2le ]; %c %s\n",
							                      funcname_str, blank_str, pi, n_dims_print + 2,
							                      dims_str, perf, resid,
							                      OUTPUT_COMMENT_CHAR,
							                      pass_str );

							// Also output to a file if requested (and successfully opened).
							if ( output_stream )
							libblis_test_fprintf( output_stream,
							                      "%s%s( %3u, 1:%u ) = [%s  %7.3lf  %8.2le ]; %c %s\n",
							                      funcname_str, blank_str, pi, n_dims_print + 2,
							                      dims_str, perf, resid,
							                      OUTPUT_COMMENT_CHAR,
							                      pass_str );
						}
						else
						{
							libblis_test_fprintf( stdout,
							                      "%s%s      %s  %7.3lf   %8.2le   %s\n",
							                      funcname_str, blank_str,
							                      dims_str, perf, resid,
							                      pass_str );

							// Also output to a file if requested (and successfully opened).
							if ( output_stream )
							libblis_test_fprintf( output_stream,
							                      "%s%s      %s  %7.3lf   %8.2le   %s\n",
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
					}
				}
			}
	
			libblis_test_fprintf( stdout, "\n" );

			if ( output_stream )
				libblis_test_fprintf( output_stream, "\n" );
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


	// If the file was opened (successfully), close the output stream.
	if ( output_stream )
		libblis_test_fclose_ofile( output_stream );


	// Mark this operation as done.
	op->test_done = TRUE;
}



void libblis_test_build_function_string( char*        prefix_str,
                                         ind_t        method,
                                         char*        ind_str,
                                         char*        op_str,
                                         char         dt_char,
                                         unsigned int n_param_combos,
                                         char*        pc_str,
                                         char*        sc_str,
                                         char*        funcname_str )
{
	sprintf( funcname_str, "%s_%c%s", prefix_str, dt_char, op_str );

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
	bool_t alignment = params->alignment;
	siz_t  elem_size = bli_datatype_size( dt );
	dim_t  m_trans   = m;
	dim_t  n_trans   = n;
	dim_t  rs        = 1; // Initialization avoids a compiler warning.
	dim_t  cs        = 1; // Initialization avoids a compiler warning.
	
	// Apply the trans parameter to the dimensions (if needed).
	bli_set_dims_with_trans( trans, m, n, m_trans, n_trans );

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



cntl_t* libblis_test_pobj_create( bszid_t bmult_id_m, bszid_t bmult_id_n, invdiag_t inv_diag, pack_t pack_schema, packbuf_t pack_buf, obj_t* a, obj_t* p, cntx_t* cntx )
{
	bool_t does_inv_diag;

	if ( inv_diag == BLIS_NO_INVERT_DIAG ) does_inv_diag = FALSE;
	else                                   does_inv_diag = TRUE;

	// Create a control tree node for the packing operation.
	cntl_t* cntl = bli_packm_cntl_create_node
	(
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

	// Pack the contents of A to P.
	bli_l3_packm( a, p, cntx, cntl, &BLIS_PACKM_SINGLE_THREADED );

	// Return the control tree pointer so the caller can free the cntl_t and its
	// mem_t entry later on.
	return cntl;
}



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



void libblis_test_vobj_randomize( test_params_t* params, bool_t normalize, obj_t* x )
{
	if ( params->rand_method == BLIS_TEST_RAND_REAL_VALUES )
		bli_randv( x );
	else // if ( params->rand_method == BLIS_TEST_RAND_NARROW_POW2 )
		bli_randnv( x );

	if ( normalize )
	{
		num_t dt   = bli_obj_datatype( *x );
		num_t dt_r = bli_obj_datatype_proj_to_real( *x );
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



void libblis_test_mobj_randomize( test_params_t* params, bool_t normalize, obj_t* a )
{
	if ( params->rand_method == BLIS_TEST_RAND_REAL_VALUES )
		bli_randm( a );
	else // if ( params->rand_method == BLIS_TEST_RAND_NARROW_POW2 )
		bli_randnm( a );

	if ( normalize )
	{
#if 0
		num_t dt      = bli_obj_datatype( *a );
		dim_t max_m_n = bli_obj_max_dim( *a );
		obj_t kappa;

		bli_obj_scalar_init_detached( dt, &kappa );

		// Normalize vector elements by maximum matrix dimension.
		bli_setsc( 1.0/( double )max_m_n, 0.0, &kappa );
		bli_scalm( &kappa, a );
#endif
		num_t dt   = bli_obj_datatype( *a );
		num_t dt_r = bli_obj_datatype_proj_to_real( *a );
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
	num_t dt = bli_obj_datatype( *a );
	dim_t m  = bli_obj_length( *a );
	dim_t n  = bli_obj_width( *a );

	obj_t d;

	// We assume that all elements of a were intialized on interval [-1,1].

	bli_obj_create( dt, m, n, 0, 0, &d );

	// Initialize the diagonal of d to 2.0 and then add the diagonal of a.
	bli_setd( &BLIS_TWO, &d );
	bli_addd( &d, a );

	bli_obj_free( &d );
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
		sleep(1);
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
	bool_t gave_option_g = FALSE;
	bool_t gave_option_o = FALSE;
	int    opt;
	char   opt_ch;

	// Copy the binary name to a global string so we can use it later.
	strncpy( libblis_test_binary_name, argv[0], MAX_BINARY_NAME_LENGTH );

	// Process all option arguments until we get a -1, which means we're done.
	while( (opt = bli_getopt( argc, argv, "g:o:" )) != -1 )
	{
		// Explicitly typecast opt, which is an int, to a char. (Failing to
		// typecast resulted in at least one user-reported problem whereby
		// opt was being filled with garbage.)
		opt_ch = ( char )opt;

		switch( opt_ch )
		{
			case 'g':
			libblis_test_printf_infoc( "detected -g option; using \"%s\" for parameters filename.\n", bli_optarg );
			strncpy( libblis_test_parameters_filename,
			         bli_optarg, MAX_FILENAME_LENGTH );
			gave_option_g = TRUE;
			break;

			case 'o':
			libblis_test_printf_infoc( "detected -o option; using \"%s\" for operations filename.\n", bli_optarg );
			strncpy( libblis_test_operations_filename,
			         bli_optarg, MAX_FILENAME_LENGTH );
			gave_option_o = TRUE;
			break;

			case '?':
			libblis_test_printf_error( "unexpected option '%c' given or missing option argument\n", bli_optopt );
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
	if ( bli_optind < argc )
	{
		libblis_test_printf_error( "Encountered unexpected non-option argument: %s\n", argv[ bli_optind ] );
	}
}



void libblis_test_check_empty_problem( obj_t* c, double* perf, double* resid )
{
	if ( bli_obj_has_zero_dim( *c ) )
	{
		*perf  = 0.0;
		*resid = 0.0;
	}
}

