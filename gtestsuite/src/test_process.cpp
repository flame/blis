#include <fstream>

#include "blis_utils.h"
#include "blis_api.h"

using namespace std;

char libblis_test_sp_chars[ 2 + 1 ] = "sc";
char libblis_test_dp_chars[ 2 + 1 ] = "dz";

char libblis_test_rd_chars[ 2 + 1 ] = "sd";
char libblis_test_cd_chars[ 2 + 1 ] = "cz";

char libblis_test_dt_chars[ 4 + 1 ] = "sdcz";

char libblis_test_store_chars[ NUM_OPERAND_TYPES ][ MAX_STORE_VALS_PER_TYPE + 1 ];
char libblis_test_param_chars[ NUM_PARAM_TYPES ][ MAX_PARAM_VALS_PER_TYPE + 1 ];

char libblis_test_pass_string[ MAX_PASS_STRING_LENGTH + 1 ];
char libblis_test_warn_string[ MAX_PASS_STRING_LENGTH + 1 ];
char libblis_test_fail_string[ MAX_PASS_STRING_LENGTH + 1 ];
char libblis_test_overflow_string[ MAX_PASS_STRING_LENGTH + 1 ];
char libblis_test_underflow_string[ MAX_PASS_STRING_LENGTH + 1 ];

#define SIGN(number) (number > 0 ? 1 : 0)
int BlisTestSuite::libblis_test_init_strings( blis_string_t *test_data )
{
    // Copy default parameters filename into its global string.
    strncpy( test_data->libblis_test_parameters_filename,
       PARAMETERS_FILENAME, MAX_FILENAME_LENGTH );

    // Copy default operations filename into its global string.
    strncpy( test_data->libblis_test_operations_filename,
       OPERATIONS_FILENAME, MAX_FILENAME_LENGTH );

    // Copy default alpha-beta parameter filename .
    strncpy( test_data->libblis_test_alphabeta_parameter,
       ALPHA_BETA_FILENAME, MAX_FILENAME_LENGTH );

    strcpy( libblis_test_pass_string, BLIS_TEST_PASS_STRING );
    strcpy( libblis_test_warn_string, BLIS_TEST_WARN_STRING );
    strcpy( libblis_test_fail_string, BLIS_TEST_FAIL_STRING );
    strcpy( libblis_test_overflow_string, BLIS_TEST_OVERFLOW_STRING );
    strcpy( libblis_test_underflow_string, BLIS_TEST_UNDERFLOW_STRING );

    strcpy( libblis_test_param_chars[BLIS_TEST_PARAM_SIDE],   BLIS_TEST_PARAM_SIDE_CHARS );
    strcpy( libblis_test_param_chars[BLIS_TEST_PARAM_UPLO],   BLIS_TEST_PARAM_UPLO_CHARS );
    strcpy( libblis_test_param_chars[BLIS_TEST_PARAM_UPLODE], BLIS_TEST_PARAM_UPLODE_CHARS );
    strcpy( libblis_test_param_chars[BLIS_TEST_PARAM_TRANS],  BLIS_TEST_PARAM_TRANS_CHARS );
    strcpy( libblis_test_param_chars[BLIS_TEST_PARAM_CONJ],   BLIS_TEST_PARAM_CONJ_CHARS );
    strcpy( libblis_test_param_chars[BLIS_TEST_PARAM_DIAG],   BLIS_TEST_PARAM_DIAG_CHARS );

    strcpy( libblis_test_store_chars[BLIS_TEST_MATRIX_OPERAND], BLIS_TEST_MSTORE_CHARS );
    strcpy( libblis_test_store_chars[BLIS_TEST_VECTOR_OPERAND], BLIS_TEST_VSTORE_CHARS );

    return 0;
}

int BlisTestSuite::libblis_test_read_params_file
     (
       char*          input_filename,
       test_params_t* params,
       char*          abpf
     )
{
    FILE* input_stream;
    char  buffer[ INPUT_BUFFER_SIZE ];
    char  temp[ INPUT_BUFFER_SIZE ];
    unsigned int i;

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
    if( params->n_mstorage > MAX_NUM_MSTORAGE )
    {
        printf( "Detected too many matrix storage schemes (%u) in input file.\n",
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
    if( params->n_vstorage > MAX_NUM_VSTORAGE )
    {
        printf( "Detected too many vector storage schemes (%u) in input file.\n",
                                 params->n_vstorage );
    }
    strcpy( params->storage[ BLIS_TEST_VECTOR_OPERAND ], temp );

    // Read whether to mix all storage combinations.
    libblis_test_read_next_line( buffer, input_stream );
    sscanf( buffer, "%u ", &(params->mix_all_storage) );

    // Read whether to perform all tests with aligned addresses and ldims.
    libblis_test_read_next_line( buffer, input_stream );
    sscanf( buffer, "%u ", &(params->alignment) );

#ifdef __GTEST_VALGRIND_TEST__
    params->alignment = 0;
#endif

    // Read the randomization method.
    libblis_test_read_next_line( buffer, input_stream );
    sscanf( buffer, "%u ", &(params->rand_method) );

    if( params->rand_method != BLIS_TEST_RAND_REAL_VALUES &&
         params->rand_method != BLIS_TEST_RAND_NARROW_POW2 )
    {
        printf( "Invalid randomization method (%u) in input file.\n",
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
    if( params->n_datatypes > MAX_NUM_DATATYPES )
    {
        printf( "Detected too many datatype requests (%u) in input file.\n",
                                 params->n_datatypes );
    }

    for( i = 0 ; i < params->n_datatypes ; ++i )
    {
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
    if( params->n_app_threads < 1 ) params->n_app_threads = 1;

    // Disable induced methods when simulating more than one application
    // threads.
    if( params->n_app_threads > 1 )
    {
        if( params->ind_enable[ BLIS_3MH  ] ||
            params->ind_enable[ BLIS_3M1  ] ||
            params->ind_enable[ BLIS_4MH  ] ||
            params->ind_enable[ BLIS_4M1B ] ||
            params->ind_enable[ BLIS_4M1A ] ||
            params->ind_enable[ BLIS_1M   ]  )
        {
            // Due to an inherent race condition in the way induced methods
            // are enabled and disabled at runtime, all induced methods must be
            // disabled when simulating multiple application threads.
            printf( "simulating multiple application threads; disabling induced methods.\n" );

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

    if( params->reaction_to_failure != ON_FAILURE_IGNORE_CHAR &&
         params->reaction_to_failure != ON_FAILURE_SLEEP_CHAR  &&
         params->reaction_to_failure != ON_FAILURE_ABORT_CHAR  )
    {
           printf( "Invalid reaction-to-failure character code (%c) in input file.\n",
                                 params->reaction_to_failure );
    }

    // Read whether to output in matlab format.
    libblis_test_read_next_line( buffer, input_stream );
    sscanf( buffer, "%u ", &(params->output_matlab_format) );

    // Read whether to output to files in addition to stdout.
    libblis_test_read_next_line( buffer, input_stream );
    sscanf( buffer, "%u ", &(params->output_files) );

    // Read the requested api path.
    libblis_test_read_next_line( buffer, input_stream );
    sscanf( buffer, "%u ", &i );
    params->api = (api_t)i;

    libblis_test_read_next_line( buffer, input_stream );
    sscanf( buffer, "%u ", &params->dimf );

    libblis_test_read_next_line( buffer, input_stream );
    sscanf( buffer, "%u ", &params->abf );

    {
      FILE *input_abpf = fopen( abpf, "r" );
      libblis_test_fopen_check_stream( abpf, input_abpf );

      libblis_test_read_next_line( buffer, input_abpf );
      sscanf( buffer, "%u ", &params->nab );

      params->alpha = ( atom_t * ) malloc( params->nab * sizeof( atom_t ) );
      params->beta  = ( atom_t * ) malloc( params->nab * sizeof( atom_t ) );

      for( i = 0 ; i < params->nab ; i++)
      {
          libblis_test_read_next_line( buffer, input_abpf );
          sscanf( buffer, "%lf   %lf", &params->alpha[i].real, &params->alpha[i].imag );
      }

      for( i = 0 ; i < params->nab ; i++)
      {
          libblis_test_read_next_line( buffer, input_abpf );
          sscanf( buffer, "%lf   %lf", &params->beta[i].real,  &params->beta[i].imag);
      }
      fclose(input_abpf);
    }

    libblis_test_read_next_line( buffer, input_stream );
    sscanf( buffer, "%u ", &params->bitextf );

    libblis_test_read_next_line( buffer, input_stream );
    sscanf( buffer, "%u ", &params->passflag );

    libblis_test_read_next_line( buffer, input_stream );
    sscanf( buffer, "%u ", &params->bitrp );

    libblis_test_read_next_line( buffer, input_stream );
    sscanf( buffer, "%c ", &(params->op_t) );

    // Close the file.
    fclose( input_stream );

    if(params->oruflw != 0)
      params->bitextf = 1;

    // Output the parameter struct.
    libblis_test_output_params_struct( stdout, params );
    return 0;
}

int BlisTestSuite::libblis_test_read_ops_file( char* input_filename, test_ops_t* ops )
{
    FILE* input_stream;

    // Attempt to open input file corresponding to input_filename as
    // read-only/binary.
    input_stream = fopen( input_filename, "r" );
    libblis_test_fopen_check_stream( input_filename, input_stream );

    // Initialize the individual override field to FALSE.
    ops->indiv_over = FALSE;

    // Begin reading operations input file.
#if 0
    // Section overrides
    libblis_test_read_section_override( ops, input_stream, &(ops->util_over) );
    libblis_test_read_section_override( ops, input_stream, &(ops->l1v_over) );
    libblis_test_read_section_override( ops, input_stream, &(ops->l1m_over) );
    libblis_test_read_section_override( ops, input_stream, &(ops->l1f_over) );
    libblis_test_read_section_override( ops, input_stream, &(ops->l2_over) );
    libblis_test_read_section_override( ops, input_stream, &(ops->l3ukr_over) );
    libblis_test_read_section_override( ops, input_stream, &(ops->l3_over) );
#endif
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
    libblis_test_read_op_info( ops, input_stream, BLIS_HEMM,  BLIS_TEST_DIMS_MN,  2, &(ops->hemm) );
    libblis_test_read_op_info( ops, input_stream, BLIS_HERK,  BLIS_TEST_DIMS_MK,  2, &(ops->herk) );
    libblis_test_read_op_info( ops, input_stream, BLIS_HER2K, BLIS_TEST_DIMS_MK,  2, &(ops->her2k) );
    libblis_test_read_op_info( ops, input_stream, BLIS_SYMM,  BLIS_TEST_DIMS_MN,  2, &(ops->symm) );
    libblis_test_read_op_info( ops, input_stream, BLIS_SYRK,  BLIS_TEST_DIMS_MK,  2, &(ops->syrk) );
    libblis_test_read_op_info( ops, input_stream, BLIS_SYR2K, BLIS_TEST_DIMS_MK,  2, &(ops->syr2k) );
    libblis_test_read_op_info( ops, input_stream, BLIS_TRMM,  BLIS_TEST_DIMS_MN,  4, &(ops->trmm) );
    libblis_test_read_op_info( ops, input_stream, BLIS_TRMM3, BLIS_TEST_DIMS_MN,  5, &(ops->trmm3) );
    libblis_test_read_op_info( ops, input_stream, BLIS_TRSM,  BLIS_TEST_DIMS_MN,  4, &(ops->trsm) );

    libblis_test_read_op_info( ops, input_stream, BLIS_NOID,  BLIS_TEST_DIMS_MNK, 2, &(ops->gemm_u8s8s32os32) );
    libblis_test_read_op_info( ops, input_stream, BLIS_NOID,  BLIS_TEST_DIMS_MNK, 2, &(ops->gemm_u8s8s32os8) );
    libblis_test_read_op_info( ops, input_stream, BLIS_NOID,  BLIS_TEST_DIMS_MNK, 2, &(ops->gemm_f32f32f32of32) );
    libblis_test_read_op_info( ops, input_stream, BLIS_NOID,  BLIS_TEST_DIMS_MNK, 2, &(ops->gemm_u8s8s16os16) );
    libblis_test_read_op_info( ops, input_stream, BLIS_NOID,  BLIS_TEST_DIMS_MNK, 2, &(ops->gemm_u8s8s16os8) );
    libblis_test_read_op_info( ops, input_stream, BLIS_NOID,  BLIS_TEST_DIMS_MNK, 2, &(ops->gemm_bf16bf16f32of32) );
    libblis_test_read_op_info( ops, input_stream, BLIS_NOID,  BLIS_TEST_DIMS_MNK, 2, &(ops->gemm_bf16bf16f32obf16) );

    // Output the section overrides.
    libblis_test_output_section_overrides( stdout, ops );

    // Close the file.
    fclose( input_stream );

    return 0;
}

ind_t ind_enable_get_str( test_params_t* params, unsigned int d,
                    unsigned int x, test_op_t* op )
{
    ind_t indi = (ind_t)params->indim[d][x];
    num_t datatype = params->dt[d];

    bli_ind_oper_enable_only( op->opid, indi, datatype );

    return bli_ind_oper_find_avail( op->opid, datatype );
}

void libblis_test_build_function_string
     (
       char*        prefix_str,
       opid_t       opid,
       ind_t        method,
       char*        ind_str,
       const char*  op_str,
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
    if( is_mixed_dt == TRUE ) //&& opid == BLIS_GEMM )
        sprintf( funcname_str, "%s_%s%s", prefix_str, dc_str, op_str );
    else
        sprintf( funcname_str, "%s_%c%s", prefix_str, dc_str[0], op_str );

    // If the method is non-native (ie: induced), append a string
    // identifying the induced method.
    if( method != BLIS_NAT )
        sprintf( &funcname_str[strlen(funcname_str)], "%s", ind_str );

    // We check the string length of pc_str in case the user is running an
    // operation that has parameters (and thus generally more than one
    // parameter combination), but has fixed all parameters in the input
    // file, which would result in n_param_combos to equal one. This way,
    // the function string contains the parameter string associated with
    // the parameters that were fixed.
    if( n_param_combos > 1 || strlen( pc_str ) > 0 )
      sprintf( &funcname_str[strlen(funcname_str)], "_%s_%s", pc_str, sc_str );
    else
      sprintf( &funcname_str[strlen(funcname_str)], "_%s", sc_str );

    if( strlen( funcname_str ) > MAX_FUNC_STRING_LENGTH )
      libblis_test_printf_error( "Function name string length (%d) exceeds maximum (%d).\n",
                                strlen( funcname_str ), MAX_FUNC_STRING_LENGTH );
    return;
}

// % dtoper_params_storage         m     n     k   gflops  resid       result
void libblis_test_build_col_labels_string
     (
       test_params_t* params,
       test_op_t* op,
       char* l_str
     )
{
    unsigned int n_spaces;
    char         blank_str[64];

    strcpy( l_str, "" );

    if( op->n_params > 0 )
    {
      sprintf( &l_str[strlen(l_str)], "%c %s_%s", OUTPUT_COMMENT_CHAR,
                                                  BLIS_FILEDATA_PREFIX_STR,
                                                  "<dt><op>_<params>_<stor>" );
    }
    else  // ( n_params == 0 )
    {
     sprintf( &l_str[strlen(l_str)], "%c %s_%s", OUTPUT_COMMENT_CHAR,
                                                 BLIS_FILEDATA_PREFIX_STR,
                                                 "<dt><op>_<stor>         " );
    }

    if( params->output_matlab_format ) n_spaces = 11;
    else                               n_spaces = 1;

    fill_string_with_n_spaces( blank_str, n_spaces );

    sprintf( &l_str[strlen(l_str)], "%s", blank_str );

    if( op->dimset == BLIS_TEST_DIMS_MNK ||
        op->dimset == BLIS_TEST_DIMS_MN  ||
        op->dimset == BLIS_TEST_DIMS_MK  ||
        op->dimset == BLIS_TEST_DIMS_M   ||
        op->dimset == BLIS_TEST_DIMS_K   ||
        op->dimset == BLIS_TEST_DIMS_MF  ||
        op->dimset == BLIS_TEST_NO_DIMS  )
     sprintf( &l_str[strlen(l_str)], " %5s", "m" );

    if( op->dimset == BLIS_TEST_DIMS_MNK ||
        op->dimset == BLIS_TEST_DIMS_MN  ||
        op->dimset == BLIS_TEST_DIMS_K   ||
        op->dimset == BLIS_TEST_DIMS_MF  ||
        op->dimset == BLIS_TEST_NO_DIMS  )
     sprintf( &l_str[strlen(l_str)], " %5s", "n" );

    if( op->dimset == BLIS_TEST_DIMS_MNK ||
        op->dimset == BLIS_TEST_DIMS_MK  ||
        op->dimset == BLIS_TEST_DIMS_K   )
     sprintf( &l_str[strlen(l_str)], " %5s", "k" );

    sprintf( &l_str[strlen(l_str)], "%s", "   resid    result" );
}

BlisTestSuite::~BlisTestSuite( )
{
    free( params.alpha );
    free( params.beta );
    bli_finalize();
    cout << "BlisTestSuite destructor completed" << endl;
}

bool AoclBlisTestFixture::destroy_params( test_params_t *params )
{
    unsigned int  pci, sci, dci;
    char**  pc_str = params->pc_str;
    char**  sc_str = params->sc_str;
    char**  dc_str = params->dc_str;

    // Free the parameter combination strings and then the master pointer.
    for ( pci = 0 ; pci < params->n_param_combos ; ++pci )
        free( pc_str[pci] );
    free( pc_str );

    // Free the storage combination strings and then the master pointer.
    for( sci = 0 ; sci < params->n_store_combos ; ++sci )
        free( sc_str[sci] );
    free( sc_str );

    // Free the datatype combination strings and then the master pointer.
    for( dci = 0 ; dci < params->n_dt_combos ; ++dci )
        free( dc_str[dci] );
    free( dc_str );

    free( params->dim );

    return true;
}

bool AoclBlisTestFixture::libblis_test_preprocess_params
     (
       test_params_t* params,
       test_op_t*     op,
       iface_t        iface,
       const char*    p_types,
       const char*    o_types
     )
{
    unsigned int  n_mstorage     = params->n_mstorage;
    unsigned int  n_vstorage     = params->n_vstorage;
    unsigned int  n_datatypes    = params->n_datatypes;

    unsigned int  mix_all_storage = params->mix_all_storage;
    unsigned int  mixed_domain    = params->mixed_domain;
    unsigned int  mixed_precision = params->mixed_precision;
    char          dt_char;

    char*         p_spec_str;
    unsigned int  n_params;
    char**        chars_for_param;

    unsigned int  n_param_combos;
    unsigned int  n_store_combos;
    unsigned int  n_dt_combos;
    char**        pc_str;
    char**        sc_str;
    char**        dc_str;

    char          s_spec_str[ MAX_NUM_OPERANDS + 1 ];
    unsigned int  n_operands;
    unsigned int  n_operandsp1;
    char**        chars_for_storage;

    char          d_spec_str[ MAX_NUM_OPERANDS + 1 ];
    char**        chars_for_spdt;
    char**        chars_for_dpdt;
    unsigned int  n_spdt_combos;
    unsigned int  n_dpdt_combos;

    char**        chars_for_dt;
    char**        chars_for_rddt;
    char**        chars_for_cddt;
    unsigned int  n_rddt_combos;
    unsigned int  n_cddt_combos;

    unsigned int  sci, dci, i, j, o;

    // These arrays are malloc()'ed in select branches. Here, we set
    // them to NULL so they can be unconditionally free()'ed at the
    // end of the function.
    chars_for_rddt = NULL;
    chars_for_cddt = NULL;
    chars_for_spdt = NULL;
    chars_for_dpdt = NULL;

    // Set the error-checking level according to what was specified in the
    // input file.
    if( params->error_checking_level == 0 )
        bli_error_checking_level_set( BLIS_NO_ERROR_CHECKING );
    else
        bli_error_checking_level_set( BLIS_FULL_ERROR_CHECKING );

    // Obtain the parameter specification (filter) string.
    p_spec_str = op->params;

    // Figure out how many parameters we have.
    n_params = strlen( p_types );

    if( strlen( p_types ) != strlen( p_spec_str) ) {
        libblis_test_printf_error( "Parameter specification string from input file does not match length of p_types string.\n" );
    }

    // Allocate an array that stores pointers to the sets of possible parameter
    // chars for each parameter.
    chars_for_param = ( char** ) malloc( n_params * sizeof( char* ) );

    // Set the values in chars_for_param to the appropriate string addresses.
    for( i = 0 ; i < n_params ; ++i )
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
    for( i = 0 ; i < n_param_combos ; ++i )
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
    if( iface == BLIS_TEST_SEQ_UKERNEL )
      mix_all_storage = DISABLE;

    // Enumerate all combinations of storage schemes requested.
    if( mix_all_storage )
    {
        // Fill storage specification string with wildcard chars.
        for( i = 0 ; i < n_operands ; ++i )
            s_spec_str[i] = '?';
        s_spec_str[i] = '\0';

        // Allocate an array that stores pointers to the sets of possible
        // storage chars for each operand.
        chars_for_storage = ( char** ) malloc( n_operands * sizeof( char* ) );

        // Set the values in chars_for_storage to the address of the string
        // that holds the storage chars.
        for( i = 0 ; i < n_operands ; ++i )
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
        for( sci = 0 ; sci < n_store_combos ; ++sci )
          sc_str[sci] = ( char* ) malloc( ( n_operands + 1 ) * sizeof( char ) );


        // Fill the storage combination strings in sc_str with the storage
        // combinations called for by the storage string p_spec_str.
        libblis_test_fill_param_strings( s_spec_str,
                                            chars_for_storage,
                                            n_operands,
                                            n_store_combos,
                                            sc_str );
        free( chars_for_storage );
    }
    else // if ( !mix_all_storage )
    {
        // Only run combinations where all operands of either type (matrices
        // or vectors) are stored in one storage scheme or another (no mixing
        // of schemes within the same operand type).
        unsigned int n_mat_operands = 0;
        unsigned int n_vec_operands = 0;

        for( o = 0 ; o < n_operands ; ++o ) {
         operand_t operand_type
                  = libblis_test_get_operand_type_for_char( o_types[o] );
         if( operand_type == BLIS_TEST_MATRIX_OPERAND )
             ++n_mat_operands;
         else if( operand_type == BLIS_TEST_VECTOR_OPERAND )
             ++n_vec_operands;
        }

        // We compute the total number of storage combinations based on whether
        // the current operation has only matrix operands, only vector operands,
        // or both.
        if( n_vec_operands == 0 )
        {
            n_store_combos = n_mstorage;
            n_vstorage = 1;
        }
        else if( n_mat_operands == 0 )
        {
            n_store_combos = n_vstorage;
            n_mstorage = 1;
        }
        else {
           n_store_combos = n_mstorage * n_vstorage;
        }

        sc_str = ( char** ) malloc( n_store_combos * sizeof( char* ) );

        for( j = 0 ; j < n_mstorage ; ++j )
        {
            for( i = 0 ; i < n_vstorage ; ++i )
            {
                sci = j*n_vstorage + i;
                sc_str[ sci ] = ( char* ) malloc( ( n_operands + 1 ) * sizeof( char ) );

                for( o = 0 ; o < n_operands ; ++o )
                {
                    unsigned int ij;
                    operand_t  operand_type
                           = libblis_test_get_operand_type_for_char( o_types[o] );

                    if ( operand_type == BLIS_TEST_MATRIX_OPERAND )
                      ij = j;
                    else
                      ij = i;

                    sc_str[sci][o] = params->storage[ operand_type ][ij];
                }
                sc_str[sci][n_operands] = '\0';
            }
        }
    }

    // Enumerate all combinations of datatype domains requested, but only
    // for the gemm operation.
    if( !mixed_domain &&  mixed_precision && op->opid == BLIS_GEMM )
    {
        params->is_mixed_dt = TRUE;

        // Increment the number of operands by one to account for the
        // computation precision (or computation datatype, as we will encode
        // it in the char string).
        n_operandsp1 = n_operands + 1;

        unsigned int has_rd = libblis_test_dt_str_has_rd_char( params );
        unsigned int has_cd = libblis_test_dt_str_has_cd_char( params );

        // Fill datatype specification string with wildcard chars.
        for( i = 0 ; i < n_operandsp1 ; ++i )
            d_spec_str[i] = '?';
        d_spec_str[i] = '\0';

        // Allocate an array that stores pointers to the sets of possible
        // datatype chars for each operand.
        chars_for_rddt = ( char** ) malloc( n_operandsp1 * sizeof( char* ) );
        chars_for_cddt = ( char** ) malloc( n_operandsp1 * sizeof( char* ) );

        // Set the values in chars_for_rddt/cddt to the address of the string
        // that holds the datatype chars.
        for( i = 0 ; i < n_operandsp1 ; ++i )
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

        if( has_rd )
          n_rddt_combos = libblis_test_count_combos( n_operandsp1, d_spec_str,
                                                    chars_for_rddt );

        if( has_cd )
          n_cddt_combos = libblis_test_count_combos( n_operandsp1, d_spec_str,
                                                    chars_for_cddt );

        // Add real and complex domain combinations.
        n_dt_combos = n_rddt_combos + n_cddt_combos;

        // Allocate an array of datatype combination strings, one for each
        // datatype combination that needs to be tested.
        dc_str = ( char** ) malloc( n_dt_combos * sizeof( char* ) );
        for( dci = 0 ; dci < n_dt_combos ; ++dci )
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
    }
    else if( mixed_domain && !mixed_precision && op->opid == BLIS_GEMM )
    {
        params->is_mixed_dt = TRUE;

        // Increment the number of operands by one to account for the
        // computation precision (or computation datatype, as we will encode
        // it in the char string).
        n_operandsp1 = n_operands + 1;

        unsigned int has_sp = libblis_test_dt_str_has_sp_char( params );
        unsigned int has_dp = libblis_test_dt_str_has_dp_char( params );

        // Fill datatype specification string with wildcard chars.
        for( i = 0 ; i < n_operands ; ++i )
            d_spec_str[i] = '?';
        d_spec_str[i] = '\0';

        // Allocate an array that stores pointers to the sets of possible
        // datatype chars for each operand (plus the computation precision
        // char).
        chars_for_spdt = ( char** ) malloc( n_operands * sizeof( char* ) );
        chars_for_dpdt = ( char** ) malloc( n_operands * sizeof( char* ) );

        // Set the values in chars_for_spdt/dpdt to the address of the string
        // that holds the datatype chars.
        for( i = 0 ; i < n_operands ; ++i )
        {
          chars_for_spdt[i] = libblis_test_sp_chars;
          chars_for_dpdt[i] = libblis_test_dp_chars;
        }

        // Compute the total number of datatype combinations to test (which is
        // simply the product of the string lengths of chars_for_spdt/dpdt[i]).
        // NOTE: We skip inspecting/branching off of the d_spec_str chars since
        // we know they are all '?'.
        n_spdt_combos = 0; n_dpdt_combos = 0;

        if( has_sp )
          n_spdt_combos = libblis_test_count_combos( n_operands, d_spec_str,
                                                    chars_for_spdt );

        if( has_dp )
          n_dpdt_combos = libblis_test_count_combos( n_operands, d_spec_str,
                                                    chars_for_dpdt );

        // Add single- and double-precision combinations.
        n_dt_combos = n_spdt_combos + n_dpdt_combos;

        // Allocate an array of datatype combination strings, one for each
        // datatype combination that needs to be tested.
        dc_str = ( char** ) malloc( n_dt_combos * sizeof( char* ) );
        for( dci = 0 ; dci < n_dt_combos ; ++dci )
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
        for( i = 0 ; i < n_dt_combos ; ++i )
        {
            dc_str[i][prec_i]   = libblis_test_proj_dtchar_to_precchar( dc_str[i][0] );
            dc_str[i][prec_i+1] = '\0';
        }
    }
    else if( mixed_domain && mixed_precision && op->opid == BLIS_GEMM )
    {
        params->is_mixed_dt = TRUE;

        // Increment the number of operands by one to account for the
        // computation precision (or computation datatype, as we will encode
        // it in the char string).
        n_operandsp1 = n_operands + 1;

        // Fill datatype specification string with wildcard chars.
        for ( i = 0 ; i < n_operandsp1 ; ++i )
            d_spec_str[i] = '?';
        d_spec_str[i] = '\0';

        // Allocate an array that stores pointers to the sets of possible
        // datatype chars for each operand.
        chars_for_dt = ( char** ) malloc( n_operandsp1 * sizeof( char* ) );

        // Set the values in chars_for_rddt/cddt to the address of the string
        // that holds the datatype chars.
        for( i = 0; i < n_operandsp1; ++i )
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
        for( dci = 0 ; dci < n_dt_combos ; ++dci )
          dc_str[dci] = ( char* ) malloc( ( n_operandsp1 + 1 ) * sizeof( char ) );

        // Fill the datatype combination strings in dc_str with the datatype
        // combinations implied by chars_for_rddt/cddt.
        libblis_test_fill_param_strings( d_spec_str,
                                         chars_for_dt,
                                         n_operandsp1,
                                         n_dt_combos,
                                         dc_str );
    }
    else  // ( ( !mixed_domain && !mixed_precision ) || op->opid != BLIS_GEMM )
    {
        params->is_mixed_dt = FALSE;

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
        for ( dci = 0 ; dci < n_dt_combos ; ++dci )
          dc_str[dci] = ( char* ) malloc( ( n_operandsp1 + 1 ) * sizeof( char ) );

        // Fill each datatype combination string with the same dt char for
        // each operand in the current operation.
        for( dci = 0 ; dci < n_dt_combos ; ++dci )
        {
            dt_char = params->datatype_char[dci];

            for ( i = 0; i < n_operands; ++i )
              dc_str[dci][i] = dt_char;

            // Encode the computation precision as the last char.
            dc_str[dci][i] = libblis_test_proj_dtchar_to_precchar( dc_str[dci][0] );
            dc_str[dci][i+1] = '\0';
        }
    }

    if( (mixed_domain) || (mixed_precision) )
    {
        params->is_mixed_dt = TRUE;
    }

  {
    unsigned int ind, indi;
	   num_t         datatype;
	   num_t         dt_check;
    // Loop over the requested datatypes.
    for( dci = 0 ; dci < n_dt_combos ; ++dci )
    {
        // We need a datatype to use for induced method related things
        // as well as to decide which set of residual thresholds to use.
        // We must choose the first operand's dt char since that's the
        // only operand we know is guaranteed to exist.
        bli_param_map_char_to_blis_dt( dc_str[dci][0], &datatype );
        dt_check = datatype;

        int has_sp =
            libblis_test_dt_str_has_sp_char_str( n_operandsp1, dc_str[dci] );
        int has_dp =
            libblis_test_dt_str_has_dp_char_str( n_operandsp1, dc_str[dci] );

        int has_samep = ( has_sp && !has_dp ) || ( has_dp && !has_sp );

        // Notice that we use n_operands here instead of
        // n_operandsp1 since we only want to chars for the
        // storage datatypes of the matrix operands, not the
        // computation precision char.
        int has_cd_only =
            !libblis_test_dt_str_has_rd_char_str( n_operands, dc_str[dci] );

        if( has_sp )
        {
            // If any of the operands are single precision, ensure that
            // dt_check is also single precision so that the residual is
            // compared to datatype-appropriate thresholds.
            dt_check = bli_dt_proj_to_single_prec( datatype );
        }

        // Start by assuming we will only test native execution.
        ind_t ind_first = BLIS_NAT;
        dim_t ind_last  = BLIS_NAT;

        // If the operation is level-3, and all operand domains are complex,
        // then we iterate over all induced methods.
        if( bli_opid_is_level3( op->opid ) && has_cd_only )
          ind_first = BLIS_IND_FIRST;

        // Loop over induced methods (or just BLIS_NAT).
        ind = 0;
        for( indi = ind_first ; indi <= ind_last ; ++indi )
        {
            // If the current datatype is real, OR if the current
            // induced method is implemented (for the operation
            // being tested) AND it was requested, then we enable
            // ONLY that method and proceed. Otherwise, we skip the
            // current method and go to the next method.
            if( bli_is_real( datatype ) ) { ; }
            else if( bli_ind_oper_is_impl( op->opid, (ind_t)indi ) &&
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
            params->indim[dci][ind] = indi;
            ind++;
        }
        params->indn[dci] = ind;
        params->dt[dci] = dt_check;
    }
  }
    // Free the array that stored pointers to the sets of possible parameter
    // chars for each parameter.
    free( chars_for_param );

    // Free some auxiliary arrays used by the mixed-domain/mixed-precision
    // datatype-handling logic.
    free( chars_for_rddt );
    free( chars_for_cddt );
    free( chars_for_spdt );
    free( chars_for_dpdt );
  /*
    if ( params->bitextf ) {
      params->dimf    = 0;
    }
  */

    if ( params->nanf )
    {
        params->bitextf = 0;
    }

    params->n_param_combos = n_param_combos;
    params->n_store_combos = n_store_combos;
    params->n_dt_combos    = n_dt_combos;

    params->pc_str = pc_str;
    params->sc_str = sc_str;
    params->dc_str = dc_str;

    if (params->abf != 1)
        params->nab = 1;

    if(params->ldf == 1)
    {
        params->ld[0] = (int)rand()%100;
        params->ld[1] = (int)rand()%100;
        params->ld[2] = (int)rand()%100;
    }
    else
    {
        params->ld[0] = 0;
        params->ld[1] = 0;
        params->ld[2] = 0;
    }

    tensor_t *v;
    unsigned int d = ( ( ( params->p_max - params->p_first ) + params->p_inc ) / params->p_inc );
    j = 0;
    if( params->dimf == 1 )
    {
        if( op->n_dims == 3)
        {
            unsigned int nc = 1;
            for( i = 0 ; i < op->n_dims ; i++ )
            {
                if( op->dim_spec[i] < 0 )
                    nc *= d;
            }
            params->ndim = nc;
            v = ( tensor_t* ) malloc( nc * sizeof( tensor_t ) );
            dim_t mm = abs( op->dim_spec[0] );
            dim_t nn = abs( op->dim_spec[1] );
            dim_t kk = abs( op->dim_spec[2] );
            dim_t m,n,k;
            dim_t mmax = ( op->dim_spec[0] > 0 ) ? params->p_first : params->p_max ;
            dim_t nmax = ( op->dim_spec[1] > 0 ) ? params->p_first : params->p_max ;
            dim_t kmax = ( op->dim_spec[2] > 0 ) ? params->p_first : params->p_max ;
            for( k = params->p_first ; k <= kmax ; k += params->p_inc )
            {
                for( n = params->p_first ; n <= nmax ; n += params->p_inc )
                {
                    for( m = params->p_first ; m <= mmax ; m += params->p_inc )
                    {
                        v[j].m = ( op->dim_spec[0] > 0 ) ? op->dim_spec[0] : (m/mm) ;
                        v[j].n = ( op->dim_spec[1] > 0 ) ? op->dim_spec[1] : (n/nn) ;
                        v[j].k = ( op->dim_spec[2] > 0 ) ? op->dim_spec[2] : (k/kk) ;
                        j++;
                    }
                }
            }
            params->ndim = j;
        }
        else if(op->n_dims == 2)
        {
            unsigned int nc = 1;
            for( i = 0 ; i < op->n_dims ; i++ )
            {
                if( op->dim_spec[i] < 0 )
                    nc *= d;
            }
            params->ndim = nc;
            v = ( tensor_t* ) malloc( nc * sizeof( tensor_t ) );
            dim_t mm = abs( op->dim_spec[0] );
            dim_t nn = abs( op->dim_spec[1] );
            dim_t m,n;
            dim_t mmax = ( op->dim_spec[0] > 0 ) ? params->p_first : params->p_max ;
            dim_t nmax = ( op->dim_spec[1] > 0 ) ? params->p_first : params->p_max ;
            for( n = params->p_first ; n <= nmax ; n += params->p_inc )
            {
                for( m = params->p_first ; m <= mmax ; m += params->p_inc )
                {
                    v[j].m = ( op->dim_spec[0] > 0 ) ? op->dim_spec[0] : (m/mm) ;
                    v[j].n = ( op->dim_spec[1] > 0 ) ? op->dim_spec[1] : (n/nn) ;
                    v[j].k = 0;
                    j++;
                }
            }
            params->ndim = j;
        }
        else
        {
            unsigned int nc = 1;
            for( i = 0 ; i < op->n_dims ; i++ )
            {
                if( op->dim_spec[i] < 0 )
                    nc *= d;
            }
            params->ndim = nc;
            v = ( tensor_t* ) malloc( nc * sizeof( tensor_t ) );
            dim_t mm = abs( op->dim_spec[0] );
            dim_t m;
            dim_t mmax = ( op->dim_spec[0] > 0 ) ? params->p_first : params->p_max ;
            for( m = params->p_first ; m <= mmax; m += params->p_inc )  {
                v[j].m = ( op->dim_spec[0] > 0 ) ? op->dim_spec[0] : (m/mm) ;
                v[j].n = 0;
                v[j].k = 0;
                j++;
            }
            params->ndim = j;
        }
    }
    else
    {
        if( op->n_dims == 3 )
        {
            if( SIGN( op->dim_spec[0] ) &&
                SIGN( op->dim_spec[1] ) &&
                SIGN( op->dim_spec[2] ) )
            {
                d = 1 ;
            }
            params->ndim = d;
            v = ( tensor_t* ) malloc( d * sizeof( tensor_t ) );
            dim_t mm = abs( op->dim_spec[0] );
            dim_t nn = abs( op->dim_spec[1] );
            dim_t kk = abs( op->dim_spec[2] );
            dim_t m;
            dim_t mmax = d == 1 ? params->p_first : params->p_max ;
            for( m = params->p_first ; m <= mmax ; m += params->p_inc )
            {
                v[j].m = ( op->dim_spec[0] > 0 ) ? op->dim_spec[0] : (m/mm) ;
                v[j].n = ( op->dim_spec[1] > 0 ) ? op->dim_spec[1] : (m/nn) ;
                v[j].k = ( op->dim_spec[2] > 0 ) ? op->dim_spec[2] : (m/kk) ;
                j++;
            }
            params->ndim = j;
        }
        else if( op->n_dims == 2 )
        {
            if( SIGN( op->dim_spec[0] ) && SIGN( op->dim_spec[1] ) )
            {
                d = 1 ;
            }
            params->ndim = d;
            v = ( tensor_t* ) malloc( d * sizeof( tensor_t ) );
            dim_t mm = abs( op->dim_spec[0] );
            dim_t nn = abs( op->dim_spec[1] );
            dim_t m;
            dim_t mmax = d == 1 ? params->p_first : params->p_max ;
            for( m = params->p_first ; m <= mmax; m += params->p_inc )
            {
                v[j].m = ( op->dim_spec[0] > 0 ) ? op->dim_spec[0] : (m/mm) ;
                v[j].n = ( op->dim_spec[1] > 0 ) ? op->dim_spec[1] : (m/nn) ;
                v[j].k = 0;
                j++;
            }
            params->ndim = j;
        }
        else
        {
            if( SIGN( op->dim_spec[0] ) )
            {
                d = 1 ;
            }
            params->ndim = d;
            v = ( tensor_t* ) malloc( d * sizeof( tensor_t ) );
            dim_t mm = abs(op->dim_spec[0]);
            dim_t m;
            dim_t mmax = d == 1 ? params->p_first : params->p_max ;
            for ( m = params->p_first ; m <= mmax; m += params->p_inc)
            {
                v[j].m = ( op->dim_spec[0] > 0 ) ? op->dim_spec[0] : (m/mm) ;
                v[j].n = 0 ;
                v[j].k = 0;
                j++;
            }
            params->ndim = j;
        }
    }
    params->dim = v;
    return true;
}

static int test_check_func( test_op_t* op )
{
  return op->op_switch;
}

void BlisTestSuite::CreateGtestFilters( test_ops_t* ops, string &str )
{
    if( test_check_func( &(ops->randv) ) )
    {
        str = str + "*AOCL_BLIS_RANDV";
        str = str + "*:";
    }
    if( test_check_func( &(ops->randm) ) )
     {
        str = str + "*AOCL_BLIS_RANDM";
        str = str + "*:";
    }
    if( test_check_func( &(ops->addv) ) )
    {
        str = str + "*AOCL_BLIS_ADDV";
        str = str + "*:";
    }
    if( test_check_func( &(ops->amaxv) ) )
    {
        str = str + "*AOCL_BLIS_AMAXV";
        str = str + "*:";
    }
    if( test_check_func( &(ops->axpbyv) ) )
    {
        str = str + "*AOCL_BLIS_AXPBYV";
        str = str + "*:";
    }
    if( test_check_func( &(ops->axpyv) ) )
    {
        str = str + "*AOCL_BLIS_AXPYV";
        str = str + "*:";
    }
    if( test_check_func( &(ops->copyv) ) )
    {
        str = str + "*AOCL_BLIS_COPYV";
        str = str + "*:";
    }
    if( test_check_func( &(ops->dotv) ) )
    {
        str = str + "*AOCL_BLIS_DOTV";
        str = str + "*:";
    }
    if( test_check_func( &(ops->dotxv) ) )
    {
        str = str + "*AOCL_BLIS_DOTXV";
        str = str + "*:";
    }
    if( test_check_func( &(ops->normfv) ) )
    {
        str = str + "*AOCL_BLIS_NORMFV";
        str = str + "*:";
    }
    if( test_check_func( &(ops->scal2v) ) )
    {
        str = str + "*AOCL_BLIS_SCAL2V";
        str = str + "*:";
    }
    if( test_check_func( &(ops->scalv) ) )
    {
        str = str + "*AOCL_BLIS_SCALV";
        str = str + "*:";
    }
    if( test_check_func( &(ops->setv) ) )
    {
        str = str + "*AOCL_BLIS_SETV";
        str = str + "*:";
    }
    if( test_check_func( &(ops->xpbyv) ) )
    {
        str = str + "*AOCL_BLIS_XPBYV";
        str = str + "*:";
    }
    if( test_check_func( &(ops->subv) ) )
    {
        str = str + "*AOCL_BLIS_SUBV";
        str = str + "*:";
    }
    if( test_check_func( &(ops->axpyf) ) )
    {
        str = str + "*AOCL_BLIS_AXPYF";
        str = str + "*:";
    }
    if( test_check_func( &(ops->axpy2v) ) )
    {
        str = str + "*AOCL_BLIS_AXPY2V";
        str = str + "*:";
    }
    if( test_check_func( &(ops->dotxf) ) )
    {
        str = str + "*AOCL_BLIS_DOTXF";
        str = str + "*:";
    }
    if( test_check_func( &(ops->dotaxpyv) ) )
    {
        str = str + "*AOCL_BLIS_DOTAXPYV";
        str = str + "*:";
    }
    if( test_check_func( &(ops->dotxaxpyf) ) )
    {
        str = str + "*AOCL_BLIS_DOTXAXPYF";
        str = str + "*:";
    }
    if( test_check_func( &(ops->addm) ) )
    {
        str = str + "*AOCL_BLIS_ADDM";
        str = str + "*:";
    }
    if( test_check_func( &(ops->axpym) ) )
    {
        str = str + "*AOCL_BLIS_AXPYM";
        str = str + "*:";
    }
    if( test_check_func( &(ops->copym) ) )
    {
        str = str + "*AOCL_BLIS_COPYM";
        str = str + "*:";
    }
    if( test_check_func( &(ops->normfm) ) )
    {
        str = str + "*AOCL_BLIS_NORMFM";
        str = str + "*:";
    }
    if( test_check_func( &(ops->scalm) ) )
    {
        str = str + "*AOCL_BLIS_SCALM";
        str = str + "*:";
    }
    if( test_check_func( &(ops->scal2m) ) )
    {
        str = str + "*AOCL_BLIS_SCAL2M";
        str = str + "*:";
    }
    if( test_check_func( &(ops->setm) ) )
    {
        str = str + "*AOCL_BLIS_SETM";
        str = str + "*:";
    }
    if( test_check_func( &(ops->subm) ) )
    {
        str = str + "*AOCL_BLIS_SUBM";
        str = str + "*:";
    }
    if( test_check_func( &(ops->xpbym) ) )
    {
        str = str + "*AOCL_BLIS_XPBYM";
        str = str + "*:";
    }
    if( test_check_func( &(ops->gemv) ) )
    {
        str = str + "*AOCL_BLIS_GEMV";
        str = str + "*:";
    }
    if( test_check_func( &(ops->ger) ) )
    {
        str = str + "*AOCL_BLIS_GER";
        str = str + "*:";
    }
    if( test_check_func( &(ops->hemv) ) )
    {
        str = str + "*AOCL_BLIS_HEMV";
        str = str + "*:";
    }
    if( test_check_func( &(ops->her) ) )
    {
        str = str + "*AOCL_BLIS_L2HER";
        str = str + "*:";
    }
    if( test_check_func( &(ops->her2) ) )
    {
        str = str + "*AOCL_BLIS_L2HER2";
        str = str + "*:";
    }
    if( test_check_func( &(ops->symv) ) )
    {
        str = str + "*AOCL_BLIS_SYMV";
        str = str + "*:";
    }
    if( test_check_func( &(ops->syr) ) )
    {
        str = str + "*AOCL_BLIS_L2SYR";
        str = str + "*:";
    }
    if( test_check_func( &(ops->syr2) ) )
    {
        str = str + "*AOCL_BLIS_L2SYR2";
        str = str + "*:";
    }
    if( test_check_func( &(ops->trmv) ) )
    {
        str = str + "*AOCL_BLIS_TRMV";
        str = str + "*:";
    }
    if( test_check_func( &(ops->trsv) ) )
    {
        str = str + "*AOCL_BLIS_TRSV";
        str = str + "*:";
    }
    if( test_check_func( &(ops->gemm) ) )
    {
        str = str + "*AOCL_BLIS_GEMM";
        str = str + "*:";
    }
    if( test_check_func( &(ops->gemmt) ) )
    {
        str = str + "*AOCL_BLIS_L3GEMMT";
        str = str + "*:";
    }
    if( test_check_func( &(ops->hemm) ) )
    {
        str = str + "*AOCL_BLIS_HEMM";
        str = str + "*:";
    }
    if( test_check_func( &(ops->herk) ) )
    {
        str = str + "*AOCL_BLIS_HERK";
        str = str + "*:";
    }
    if( test_check_func( &(ops->her2k) ) )
    {
        str = str + "*AOCL_BLIS_HER2K";
        str = str + "*:";
    }
    if( test_check_func( &(ops->symm) ) )
    {
        str = str + "*AOCL_BLIS_SYMM";
        str = str + "*:";
    }
    if( test_check_func( &(ops->syrk) ) )
    {
        str = str + "*AOCL_BLIS_SYRK";
        str = str + "*:";
    }
    if( test_check_func( &(ops->syr2k) ) )
    {
        str = str + "*AOCL_BLIS_SYR2K";
        str = str + "*:";
    }
    if( test_check_func( &(ops->trmm) ) )
    {
        str = str + "*AOCL_BLIS_L3TRMM";
        str = str + "*:";
    }
    if( test_check_func( &(ops->trmm3) ) )
    {
        str = str + "*AOCL_BLIS_TRMM3";
        str = str + "*:";
    }
    if( test_check_func( &(ops->trsm) ) )
    {
        str = str + "*AOCL_BLIS_TRSM";
        str = str + "*:";
    }
    if( test_check_func( &(ops->gemm_u8s8s32os32) ) )
    {
        str = str + "*AOCL_GEMM_S32S32";
        str = str + "*:";
    }
    if( test_check_func( &(ops->gemm_u8s8s32os8) ) )
    {
        str = str + "*AOCL_GEMM_S8S32";
        str = str + "*:";
    }
    if( test_check_func( &(ops->gemm_f32f32f32of32) ) )
    {
        str = str + "*AOCL_GEMM_F32F32";
        str = str + "*:";
    }
    if( test_check_func( &(ops->gemm_u8s8s16os16) ) )
    {
        str = str + "*AOCL_GEMM_S16S16";
        str = str + "*:";
    }
    if( test_check_func( &(ops->gemm_u8s8s16os8) ) )
    {
        str = str + "*AOCL_GEMM_S8S16";
        str = str + "*:";
    }
    if( test_check_func( &(ops->gemm_bf16bf16f32of32) ) )
    {
        str = str + "*AOCL_GEMM_F32BF16";
        str = str + "*:";
    }
    if( test_check_func( &(ops->gemm_bf16bf16f32obf16) ) )
    {
        str = str + "*AOCL_GEMM_BF16BF16";
        str = str + "*:";
    }
    cout << "Filter_data :" << str.c_str() << endl;
}

int BlisTestSuite::libblis_test_inpfile( char* filename, input_file_t* pfile )
{
  ifstream input_filename( filename );
  if( !input_filename.is_open() )
  {
      cerr << "Could not open the input file - '" << filename << "  " << endl;
      return EXIT_FAILURE;
  }
  strncpy( pfile->inputfile, filename, MAX_FILENAME_LENGTH );
  pfile->fileread = 1;

  input_filename.close();
  return 0;
}

int AoclBlisTestFixture::libblis_test_read_params_inpfile( char* filename,
  test_params_t* params, test_ops_t* ops, printres_t* pfr )
{
    string line;
    ifstream input_filename( filename );

    if( !input_filename.is_open() )
    {
        cerr << "Could not open the input file - '" << filename << "  " << endl;
        return EXIT_FAILURE;
    }

    while( getline(input_filename, line) )
    {
        libblis_read_inpprms( line, params, ops, pfr );
    }

    input_filename.close();
    return 0;
}

bool AoclBlisTestFixture::create_params( test_params_t *params )
{
    char** pc_str;
    char** sc_str;
    char** dc_str;
    unsigned int i;
    unsigned int n_params  = 4;  //max n_params = 4
    unsigned int n_dims    = 3;  //max n_dims = 3

    params->n_param_combos = n_params;
    params->n_store_combos = n_dims;
    params->n_dt_combos    = 1;

    // Free the parameter combination strings and then the master pointer.
    pc_str = ( char** ) malloc( params->n_param_combos * sizeof( char* ) );
    for ( i = 0 ; i < params->n_param_combos ; ++i )
    {
        pc_str[i] = ( char* ) malloc( ( n_params + 1 ) * sizeof( char ) );
        memset( pc_str[i], 0, ( n_params + 1 ) );
    }

    // Free the storage combination strings and then the master pointer.
    sc_str = ( char** ) malloc( params->n_store_combos * sizeof( char* ) );
    for ( i = 0 ; i < params->n_store_combos ; ++i )
    {
        sc_str[i] = ( char* ) malloc( ( n_dims + 1 ) * sizeof( char ) );
        memset( sc_str[i], 0, ( n_dims + 1 ) );
    }

    // Free the datatype combination strings and then the master pointer.
    dc_str = ( char** ) malloc( params->n_dt_combos * sizeof( char* ) );
    for ( i = 0 ; i < params->n_dt_combos ; ++i )
    {
        dc_str[i] = ( char* ) malloc( ( params->n_dt_combos + 1 ) * sizeof( char ) );
        memset( dc_str[i], 0, ( params->n_dt_combos + 1 ) );
    }

    params->n_repeats     = 1 ;

    params->rand_method   = 0 ;
    params->gs_spacing    = 32;
    params->alignment     = 0 ;
    params->n_app_threads = 1 ;
    params->error_checking_level = 1;

    params->bitextf      =  0 ;
    params->nab          =  1 ;
    params->passflag     =  1 ;
    params->api          = API_CBLAS;

    params->pc_str       =  pc_str;
    params->sc_str       =  sc_str;
    params->dc_str       =  dc_str;

    params->alpha = ( atom_t * ) malloc( params->nab * sizeof( atom_t ) );
    params->beta  = ( atom_t * ) malloc( params->nab * sizeof( atom_t ) );

    params->dim = ( tensor_t* ) malloc(3 * sizeof( tensor_t ) );
    memset( params->dim, 0, (3 * sizeof( tensor_t )) );

    return true;
}

void BlisTestSuite::CreateGtestFilters_api(input_file_t* pfile, string &str)
{
    if(pfile->fileread == 1)
    {
        str = str + "*AOCL_BLIS_READ_INPUTFILE";
        str = str + "*:";
    }
    cout << "Filter_data :" << str.c_str() << endl;
}
