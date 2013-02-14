/*

   BLIS    
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2013, The University of Texas

   Redistribution and use in source and binary forms, with or without
   modification, are permitted provided that the following conditions are
   met:
    - Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.
    - Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.
    - Neither the name of The University of Texas nor the names of its
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

#include "blis2.h"
#include "test_libblis.h"


// Global variables.
char libblis_test_binary_name[ MAX_BINARY_NAME_LENGTH + 1 ];

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
	bl2_init();

	// Initialize some strings.
	libblis_test_init_strings();

	// Parse the command line parameters.
	libblis_test_parse_command_line( argc, argv );

	// Read the global parameters file.
	libblis_test_read_params_file( PARAMETERS_FILENAME, &params );

	// Read the operations parameter file.
	libblis_test_read_ops_file( OPERATIONS_FILENAME, &ops );
	
	// Test the utility operations.
	libblis_test_utility_ops( &params, &ops );

	// Test the level-1v operations.
	libblis_test_level1v_ops( &params, &ops );

	// Test the level-1m operations.
	libblis_test_level1m_ops( &params, &ops );

	// Test the level-2 operations.
	libblis_test_level2_ops( &params, &ops );

	// Test the level-3 operations.
	libblis_test_level3_ops( &params, &ops );

	// Finalize libblis.
	bl2_finalize();

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
	libblis_test_axpyv( params, &(ops->axpyv) );
	libblis_test_copyv( params, &(ops->copyv) );
	libblis_test_dotv( params, &(ops->dotv) );
	libblis_test_dotxv( params, &(ops->dotxv) );
	libblis_test_fnormv( params, &(ops->fnormv) );
	libblis_test_scalv( params, &(ops->scalv) );
	libblis_test_scal2v( params, &(ops->scal2v) );
	libblis_test_setv( params, &(ops->setv) );
	libblis_test_subv( params, &(ops->subv) );

}



void libblis_test_level1m_ops( test_params_t* params, test_ops_t* ops )
{
	libblis_test_addm( params, &(ops->addm) );
	libblis_test_axpym( params, &(ops->axpym) );
	libblis_test_copym( params, &(ops->copym) );
	libblis_test_fnormm( params, &(ops->fnormm) );
	libblis_test_scalm( params, &(ops->scalm) );
	libblis_test_scal2m( params, &(ops->scal2m) );
	libblis_test_setm( params, &(ops->setm) );
	libblis_test_subm( params, &(ops->subm) );

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

	// Utility operations
	libblis_test_read_op_info( ops, input_stream, BLIS_TEST_DIMS_M,   0, &(ops->randv) );
	libblis_test_read_op_info( ops, input_stream, BLIS_TEST_DIMS_MN,  0, &(ops->randm) );

	// Level-1v
	libblis_test_read_op_info( ops, input_stream, BLIS_TEST_DIMS_M,   1, &(ops->addv) );
	libblis_test_read_op_info( ops, input_stream, BLIS_TEST_DIMS_M,   1, &(ops->axpyv) );
	libblis_test_read_op_info( ops, input_stream, BLIS_TEST_DIMS_M,   1, &(ops->copyv) );
	libblis_test_read_op_info( ops, input_stream, BLIS_TEST_DIMS_M,   2, &(ops->dotv) );
	libblis_test_read_op_info( ops, input_stream, BLIS_TEST_DIMS_M,   2, &(ops->dotxv) );
	libblis_test_read_op_info( ops, input_stream, BLIS_TEST_DIMS_M,   0, &(ops->fnormv) );
	libblis_test_read_op_info( ops, input_stream, BLIS_TEST_DIMS_M,   1, &(ops->scalv) );
	libblis_test_read_op_info( ops, input_stream, BLIS_TEST_DIMS_M,   1, &(ops->scal2v) );
	libblis_test_read_op_info( ops, input_stream, BLIS_TEST_DIMS_M,   0, &(ops->setv) );
	libblis_test_read_op_info( ops, input_stream, BLIS_TEST_DIMS_M,   1, &(ops->subv) );

	// Level-1m
	libblis_test_read_op_info( ops, input_stream, BLIS_TEST_DIMS_MN,  1, &(ops->addm) );
	libblis_test_read_op_info( ops, input_stream, BLIS_TEST_DIMS_MN,  1, &(ops->axpym) );
	libblis_test_read_op_info( ops, input_stream, BLIS_TEST_DIMS_MN,  1, &(ops->copym) );
	libblis_test_read_op_info( ops, input_stream, BLIS_TEST_DIMS_MN,  0, &(ops->fnormm) );
	libblis_test_read_op_info( ops, input_stream, BLIS_TEST_DIMS_MN,  1, &(ops->scalm) );
	libblis_test_read_op_info( ops, input_stream, BLIS_TEST_DIMS_MN,  1, &(ops->scal2m) );
	libblis_test_read_op_info( ops, input_stream, BLIS_TEST_DIMS_MN,  0, &(ops->setm) );
	libblis_test_read_op_info( ops, input_stream, BLIS_TEST_DIMS_MN,  1, &(ops->subm) );

	// Level-2
	libblis_test_read_op_info( ops, input_stream, BLIS_TEST_DIMS_MN,  2, &(ops->gemv) );
	libblis_test_read_op_info( ops, input_stream, BLIS_TEST_DIMS_MN,  2, &(ops->ger) );
	libblis_test_read_op_info( ops, input_stream, BLIS_TEST_DIMS_M,   3, &(ops->hemv) );
	libblis_test_read_op_info( ops, input_stream, BLIS_TEST_DIMS_M,   2, &(ops->her) );
	libblis_test_read_op_info( ops, input_stream, BLIS_TEST_DIMS_M,   3, &(ops->her2) );
	libblis_test_read_op_info( ops, input_stream, BLIS_TEST_DIMS_M,   3, &(ops->symv) );
	libblis_test_read_op_info( ops, input_stream, BLIS_TEST_DIMS_M,   2, &(ops->syr) );
	libblis_test_read_op_info( ops, input_stream, BLIS_TEST_DIMS_M,   3, &(ops->syr2) );
	libblis_test_read_op_info( ops, input_stream, BLIS_TEST_DIMS_M,   3, &(ops->trmv) );
	libblis_test_read_op_info( ops, input_stream, BLIS_TEST_DIMS_M,   3, &(ops->trsv) );

	// Level-3
	libblis_test_read_op_info( ops, input_stream, BLIS_TEST_DIMS_MNK, 2, &(ops->gemm) );
	libblis_test_read_op_info( ops, input_stream, BLIS_TEST_DIMS_MN,  4, &(ops->hemm) );
	libblis_test_read_op_info( ops, input_stream, BLIS_TEST_DIMS_MK,  2, &(ops->herk) );
	libblis_test_read_op_info( ops, input_stream, BLIS_TEST_DIMS_MK,  3, &(ops->her2k) );
	libblis_test_read_op_info( ops, input_stream, BLIS_TEST_DIMS_MN,  4, &(ops->symm) );
	libblis_test_read_op_info( ops, input_stream, BLIS_TEST_DIMS_MK,  2, &(ops->syrk) );
	libblis_test_read_op_info( ops, input_stream, BLIS_TEST_DIMS_MK,  3, &(ops->syr2k) );
	libblis_test_read_op_info( ops, input_stream, BLIS_TEST_DIMS_MN,  4, &(ops->trmm) );
	libblis_test_read_op_info( ops, input_stream, BLIS_TEST_DIMS_MN,  5, &(ops->trmm3) );
	libblis_test_read_op_info( ops, input_stream, BLIS_TEST_DIMS_MN,  4, &(ops->trsm) );


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

	// Check for success.
	if ( input_stream == NULL )
	{
		libblis_test_printf_error( "Failed to open input file %s. Check existence and permissions.\n",
		                           input_filename );
	}
	
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

	// Read the general stride "spacing".
	libblis_test_read_next_line( buffer, input_stream );
	sscanf( buffer, "%d ", &(params->gs_spacing) );

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
	sscanf( buffer, "%lu ", &(params->p_first) );

	// Read the maximum problem size to test.
	libblis_test_read_next_line( buffer, input_stream );
	sscanf( buffer, "%lu ", &(params->p_max) );

	// Read the problem size increment to test.
	libblis_test_read_next_line( buffer, input_stream );
	sscanf( buffer, "%lu ", &(params->p_inc) );

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

	// Read whether to output to matlab files.
	libblis_test_read_next_line( buffer, input_stream );
	sscanf( buffer, "%u ", &(params->output_matlab_files) );

	// Close the file.
	fclose( input_stream );

	// Output the parameter struct.
	libblis_test_output_params_struct( stdout, params );
}



void libblis_test_read_op_info( test_ops_t*  ops,
                                FILE*        input_stream,
                                dimset_t     dimset,
                                unsigned int n_params,
                                test_op_t*   op )
{
	char         buffer[ INPUT_BUFFER_SIZE ];
	char         temp[ INPUT_BUFFER_SIZE ];
	int          op_switch;
	int          i, p;

	// Read the line for the overall operation switch.
	libblis_test_read_next_line( buffer, input_stream );
	sscanf( buffer, "%d ", &op_switch );

	// Read the line for the sequential front-end.
	libblis_test_read_next_line( buffer, input_stream );
	sscanf( buffer, "%d ", &(op->front_seq) );

	op->n_dims = libblis_test_get_n_dims_from_dimset( dimset );
	op->dimset = dimset;

	if ( op->n_dims > MAX_NUM_DIMENSIONS )
	{
		libblis_test_printf_error( "Detected too many dimensions (%u) in input file to store.\n",
		                           op->n_dims );
	}

	// Read the dimension specifications.
	libblis_test_read_next_line( buffer, input_stream );
	for ( i = 0, p = 0; i < op->n_dims; ++i )
	{
		// Advance until we hit non-whitespace (ie: the next number).
		for ( ; isspace( buffer[p] ); ++p ) ; 

		sscanf( &buffer[p], "%d", &(op->dim_spec[i]) );

		//printf( "dim[%d] = %d\n", i, op->dim_spec[i] );

		// Advance until we hit whitespace (ie: the space before the next number).
		for ( ; !isspace( buffer[p] ); ++p ) ; 
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
			libblis_test_printf_error( "Number of parameters specified by caller does not match length of parameter string in input file.\n" );
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
	if ( op_switch == DISABLE_ALL )
	{
		op->front_seq = DISABLE;
	}
}



void libblis_test_output_params_struct( FILE* os, test_params_t* params )
{
	int i;

	// Output some system parameters.
	libblis_test_fprintf_c( os, "\n" );
	libblis_test_fprintf_c( os, "--- BLIS library info -------------------------------------\n" );
	libblis_test_fprintf_c( os, "\n" );
	libblis_test_fprintf_c( os, "version string              %s\n", bl2_version() );
	libblis_test_fprintf_c( os, "\n" );
	libblis_test_fprintf_c( os, "--- BLIS config header ---\n" );
	libblis_test_fprintf_c( os, "\n" );
	libblis_test_fprintf_c( os, "enable memory alignment?    %u\n", BLIS_ENABLE_MEMORY_ALIGNMENT );
	libblis_test_fprintf_c( os, "  alignment boundary        %u\n", BLIS_MEMORY_ALIGNMENT_BOUNDARY );
	libblis_test_fprintf_c( os, "static memory pool            \n" );
	libblis_test_fprintf_c( os, "  # of mc x kc blocks       %u\n", BLIS_NUM_MC_X_KC_BLOCKS );
	libblis_test_fprintf_c( os, "  # of kc x nc blocks       %u\n", BLIS_NUM_KC_X_NC_BLOCKS );
	libblis_test_fprintf_c( os, "  page size                 %u\n", BLIS_PAGE_SIZE );
	libblis_test_fprintf_c( os, "  max prefetch byte offset  %u\n", BLIS_MAX_PREFETCH_BYTE_OFFSET );
	libblis_test_fprintf_c( os, "\n" );
	libblis_test_fprintf_c( os, "--- BLIS kernel header ---\n" );
	libblis_test_fprintf_c( os, "\n" );
	libblis_test_fprintf_c( os, "cache blocksizes            s     d     c     z \n" );
	libblis_test_fprintf_c( os, "  m dimension           %5u %5u %5u %5u\n",
	                        BLIS_DEFAULT_MC_S,
	                        BLIS_DEFAULT_MC_D,
	                        BLIS_DEFAULT_MC_C,
	                        BLIS_DEFAULT_MC_Z );
	libblis_test_fprintf_c( os, "  k dimension           %5u %5u %5u %5u\n",
	                        BLIS_DEFAULT_KC_S,
	                        BLIS_DEFAULT_KC_D,
	                        BLIS_DEFAULT_KC_C,
	                        BLIS_DEFAULT_KC_Z );
	libblis_test_fprintf_c( os, "  n dimension           %5u %5u %5u %5u\n",
	                        BLIS_DEFAULT_NC_S,
	                        BLIS_DEFAULT_NC_D,
	                        BLIS_DEFAULT_NC_C,
	                        BLIS_DEFAULT_NC_Z );
	libblis_test_fprintf_c( os, "\n" );
	libblis_test_fprintf_c( os, "register blocksizes         s     d     c     z \n" );
	libblis_test_fprintf_c( os, "  m dimension           %5u %5u %5u %5u\n",
	                        BLIS_DEFAULT_MR_S,
	                        BLIS_DEFAULT_MR_D,
	                        BLIS_DEFAULT_MR_C,
	                        BLIS_DEFAULT_MR_Z );
	libblis_test_fprintf_c( os, "  n dimension           %5u %5u %5u %5u\n",
	                        BLIS_DEFAULT_NR_S,
	                        BLIS_DEFAULT_NR_D,
	                        BLIS_DEFAULT_NR_C,
	                        BLIS_DEFAULT_NR_Z );
	libblis_test_fprintf_c( os, "\n" );
	libblis_test_fprintf_c( os, "level-1f fusing factors     s     d     c     z \n" );
	libblis_test_fprintf_c( os, "                        %5u %5u %5u %5u\n",
	                        BLIS_DEFAULT_FUSING_FACTOR_S,
	                        BLIS_DEFAULT_FUSING_FACTOR_D,
	                        BLIS_DEFAULT_FUSING_FACTOR_C,
	                        BLIS_DEFAULT_FUSING_FACTOR_Z );
	libblis_test_fprintf_c( os, "\n" );
	libblis_test_fprintf( os, "\n" );

	// Output the contents of the param struct.
	libblis_test_fprintf_c( os, "\n" );
	libblis_test_fprintf_c( os, "--- BLIS test suite parameters ----------------------------\n" );
	libblis_test_fprintf_c( os, "\n" );
	libblis_test_fprintf_c( os, "n_repeats                   %u\n", params->n_repeats );
	libblis_test_fprintf_c( os, "n_mstorage                  %u\n", params->n_mstorage );
	libblis_test_fprintf_c( os, "storage[ matrix ]           %s\n", params->storage[ BLIS_TEST_MATRIX_OPERAND ] );
	libblis_test_fprintf_c( os, "n_vstorage                  %u\n", params->n_vstorage );
	libblis_test_fprintf_c( os, "storage[ vector ]           %s\n", params->storage[ BLIS_TEST_VECTOR_OPERAND ] );
	libblis_test_fprintf_c( os, "mix_all_storage             %u\n", params->mix_all_storage );
	libblis_test_fprintf_c( os, "gs_spacing                  %d\n", params->gs_spacing );
	libblis_test_fprintf_c( os, "n_datatypes                 %u\n", params->n_datatypes );
	libblis_test_fprintf_c( os, "datatype[0]                 %d (%c)\n", params->datatype[0],
	                                                                params->datatype_char[0] );
	for( i = 1; i < params->n_datatypes; ++i )
	libblis_test_fprintf_c( os, "        [%d]                 %d (%c)\n", i, params->datatype[i],
	                                                                    params->datatype_char[i] );
	libblis_test_fprintf_c( os, "p_first                     %u\n", params->p_first );
	libblis_test_fprintf_c( os, "p_max                       %u\n", params->p_max );
	libblis_test_fprintf_c( os, "p_inc                       %u\n", params->p_inc );
	libblis_test_fprintf_c( os, "reaction_to_failure         %c\n", params->reaction_to_failure );
	libblis_test_fprintf_c( os, "output_matlab_files         %u\n", params->output_matlab_files );
	libblis_test_fprintf_c( os, "\n" );
	libblis_test_fprintf( os, "\n" );
}



void libblis_test_output_op_struct( FILE* os, test_op_t* op, char* op_str )
{
	dimset_t dimset = op->dimset;

	libblis_test_fprintf_c( os, "%s seq_front              %d\n", op_str, op->front_seq );

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
	else if ( dimset == BLIS_TEST_DIMS_M )
	{
		libblis_test_fprintf_c( os, "%s m                      %d\n", op_str,
		                                op->dim_spec[0] );
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



dim_t libblis_test_get_n_dims_from_dimset( dimset_t dimset )
{
	dim_t n_dims;

	if      ( dimset == BLIS_TEST_DIMS_MNK ) n_dims = 3;
	else if ( dimset == BLIS_TEST_DIMS_MN  ) n_dims = 2;
	else if ( dimset == BLIS_TEST_DIMS_MK  ) n_dims = 2;
	else if ( dimset == BLIS_TEST_DIMS_M   ) n_dims = 1;
	else
	{
		n_dims = 0;
		libblis_test_printf_error( "Invalid dimension combination.\n" );
	}

	return n_dims;
}



dim_t libblis_test_get_dim_from_prob_size( int   dim_spec,
                                           dim_t p_size )
{
	dim_t dim;

	if ( dim_spec < 0 ) dim = p_size / bl2_abs(dim_spec);
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
                             mt_impl_t      impl,
                             char*          op_str,
                             char*          p_types,
                             char*          o_types,
                             thresh_t*      thresh,
                             void (*f_exp)  (test_params_t*, // params struct
                                             test_op_t*,     // op struct
                                             mt_impl_t,      // impl
                                             num_t,          // datatype (current datatype)
                                             char*,          // pc_str (current param string)
                                             char*,          // sc_str (current storage string)
                                             dim_t,          // p_cur (current problem size)
                                             double*,        // perf
                                             double* ) )     // residual
{
	unsigned int  n_mstorage          = params->n_mstorage;
	unsigned int  n_vstorage          = params->n_vstorage;
	unsigned int  n_datatypes         = params->n_datatypes;
	dim_t         p_first             = params->p_first;
	dim_t         p_max               = params->p_max;
	dim_t         p_inc               = params->p_inc;
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

	dim_t         p_cur, pi;
	unsigned int  dt, pci, sci, i, j, o;

	double        perf, resid;
	char*         pass_str;
	char          blank_str[32];
	char          funcname_str[64];
	char          dims_str[64];
	char          label_str[128];
	unsigned int  n_spaces;

	FILE*         output_stream = NULL;


	// If output to matlab files was requested, attempt to open a file stream.
	if ( params->output_matlab_files )
		libblis_test_fopen_mfile( op_str, impl, &output_stream );


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

	// Determine the total number of storage schemes.
	if ( params->mix_all_storage )
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
		//libblis_test_fill_storage_strings( sc_str, n_store_combos, n_operands );
		libblis_test_fill_param_strings( s_spec_str,
	                                     chars_for_storage,
	                                     n_operands,
	                                     n_store_combos,
	                                     sc_str );
	}
	else // if ( !params->mix_all_storage )
	{
		// Only run combinations where all operands of either type (matrices
		// or vectors) are stored in one storage scheme or another (no mixing
		// of schemes within the same operand type).
		n_store_combos = n_mstorage * n_vstorage;

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
			libblis_test_build_col_labels_string( op, label_str );

			// Output the column label string.
			libblis_test_fprintf( stdout, "%s\n", label_str );

			// Also output to a matlab file if requested (and successfully
			// opened).
			if ( output_stream )
				libblis_test_fprintf( output_stream, "%s\n", label_str );

			// Loop over the requested parameter combinations.
			for ( pci = 0; pci < n_param_combos; ++pci )	
			{
				// Loop over the requested problem sizes.
				for ( p_cur = p_first, pi = 1; p_cur <= p_max; p_cur += p_inc, ++pi )
				{
					f_exp( params,
					       op,
					       impl,
					       datatype,
					       pc_str[pci],
					       sc_str[sci],
					       p_cur,
					       &perf, &resid );

					// Remove the sign of the residual, if there is one.
					resid = bl2_fabs( resid );
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
					                                    op_str,
					                                    dt_char,
					                                    n_param_combos,
					                                    pc_str[pci],
					                                    sc_str[sci],
					                                    funcname_str );


					n_spaces = MAX_FUNC_STRING_LENGTH - strlen( funcname_str );
					fill_string_with_n_spaces( blank_str, n_spaces );

					// Print all dimensions to a single string.
					strcpy( dims_str, "" );
					for ( i = 0; i < op->n_dims; ++i )
					{
						sprintf( &dims_str[strlen(dims_str)], " %5lu",
						         libblis_test_get_dim_from_prob_size( op->dim_spec[i],
						                                              p_cur ) );
					}

					// Output the results of the test.
					libblis_test_fprintf( stdout,
					                      "%s%s                 %s  %6.3lf  %9.2le    %s\n",
					                      funcname_str, blank_str,
					                      dims_str, perf, resid,
					                      pass_str );

					// Also output to a matlab file if requested (and successfully
					// opened).
					if ( output_stream )
					libblis_test_fprintf( output_stream,
					                      "%s%s( %3lu, 1:%lu ) = [ %s  %6.3lf  %9.2le ]; %c %s\n",
					                      funcname_str, blank_str, pi, op->n_dims + 2,
					                      dims_str, perf, resid,
					                      OUTPUT_COMMENT_CHAR,
					                      pass_str );

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
		libblis_test_fclose_mfile( output_stream );


	// Mark this operation as done.
	op->test_done = TRUE;
}



void libblis_test_build_function_string( char*        prefix_str,
                                         char*        op_str,
                                         char         dt_char,
                                         unsigned int n_param_combos,
                                         char*        pc_str,
                                         char*        sc_str,
                                         char*        funcname_str )
{
	sprintf( funcname_str, "%s_%c%s", prefix_str, dt_char, op_str );

	if ( n_param_combos > 1 )
		sprintf( &funcname_str[strlen(funcname_str)], "_%s_%s", pc_str, sc_str );
	else
		sprintf( &funcname_str[strlen(funcname_str)], "_%s", sc_str );
}


// % dtoper_params_storage                       m     n     k   gflops  resid       result
void libblis_test_build_col_labels_string( test_op_t* op, char* l_str )
{
	unsigned int n_spaces;
	char         blank_str[64];

	strcpy( l_str, "" );

	if ( op->n_params > 0 )
	{
		sprintf( &l_str[strlen(l_str)], "%c %s_%s", OUTPUT_COMMENT_CHAR,
		                                            BLIS_FILEDATA_PREFIX_STR,
		                                            "<dt><oper>_<params>_<storage>" );
	}
	else // ( n_params == 0 )
	{
		sprintf( &l_str[strlen(l_str)], "%c %s_%s", OUTPUT_COMMENT_CHAR,
		                                            BLIS_FILEDATA_PREFIX_STR,
		                                            "<dt><oper>_<storage>         " );
	}

	n_spaces = 5;
	fill_string_with_n_spaces( blank_str, n_spaces );

	sprintf( &l_str[strlen(l_str)], "%s", blank_str );

	if ( op->dimset == BLIS_TEST_DIMS_MNK ||
	     op->dimset == BLIS_TEST_DIMS_MN  ||
	     op->dimset == BLIS_TEST_DIMS_MK  ||
	     op->dimset == BLIS_TEST_DIMS_M   )
		sprintf( &l_str[strlen(l_str)], " %5s", "m" );

	if ( op->dimset == BLIS_TEST_DIMS_MNK ||
	     op->dimset == BLIS_TEST_DIMS_MN  )
		sprintf( &l_str[strlen(l_str)], " %5s", "n" );

	if ( op->dimset == BLIS_TEST_DIMS_MNK ||
	     op->dimset == BLIS_TEST_DIMS_MK  )
		sprintf( &l_str[strlen(l_str)], " %5s", "k" );

	sprintf( &l_str[strlen(l_str)], "%s", "   gflops  resid       result" );
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

	for ( i = 0; i < n_spaces; ++i )
		sprintf( &str[i], " " );
}



void libblis_test_mobj_create( test_params_t* params, num_t dt, trans_t trans, char storage, dim_t m, dim_t n, obj_t* a )
{
	dim_t gs      = params->gs_spacing;
	dim_t m_trans = m;
	dim_t n_trans = n;
	dim_t rs_g;
	dim_t cs_g;
	
	// Apply the trans parameter to the dimensions (if needed).
	bl2_set_dims_with_trans( trans, m, n, m_trans, n_trans );

	// In case of general strides, use the general stride spacing specified by the
	// user to create strides with a column-major tilt.
	rs_g = gs * 1;
	cs_g = gs * m_trans;

	if      ( storage == 'c' ) bl2_obj_create( dt, m_trans, n_trans, 0,       0, a );
	else if ( storage == 'r' ) bl2_obj_create( dt, m_trans, n_trans, n_trans, 1, a );
	else if ( storage == 'g' ) bl2_obj_create( dt, m_trans, n_trans, rs_g, cs_g, a );
	else
	{
		libblis_test_printf_error( "Invalid storage character: %c\n", storage );
	}
}



void libblis_test_vobj_create( test_params_t* params, num_t dt, char storage, dim_t m, obj_t* x )
{
	dim_t gs = params->gs_spacing;

	// Column vector (unit stride)
	if      ( storage == 'c' ) bl2_obj_create( dt, m, 1,  1,  m,    x );
	// Row vector (unit stride)
	else if ( storage == 'r' ) bl2_obj_create( dt, 1, m,  m,  1,    x );
	// Column vector (non-unit stride)
	else if ( storage == 'j' ) bl2_obj_create( dt, m, 1,  gs, gs*m, x );
	// Row vector (non-unit stride)
	else if ( storage == 'i' ) bl2_obj_create( dt, 1, m,  gs*m, gs, x );
	else
	{
		libblis_test_printf_error( "Invalid storage character: %c\n", storage );
	}
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



void libblis_test_fopen_mfile( char* op_str, mt_impl_t impl, FILE** output_stream )
{
	char filename_str[ MAX_FILENAME_LENGTH ];

	if ( impl == BLIS_TEST_MT_FRONT_END )
		bl2_check_error_code( BLIS_NOT_YET_IMPLEMENTED );

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



void libblis_test_fclose_mfile( FILE* output_stream )
{
	fclose( output_stream );
}



void libblis_test_fopen_check_stream( char* filename_str,
                                      FILE* stream )
{
	// Check for success.
	if ( stream == NULL )
	{
		libblis_test_printf_error( "Failed to open output file %s. Check existence (if file is being read), permissions (if file is being overwritten), and/or storage limit.\n",
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
	if ( argc > 1 )
	{
		fprintf( stderr, "Too many command line arguments.\n" );
		exit(1);
	}
	
	// Copy the binary name to a global string so we can use it later.
	strncpy( libblis_test_binary_name, argv[0], MAX_BINARY_NAME_LENGTH );
}



