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


//
// --- System headers ----------------------------------------------------------
//

// For va_* functions.
#include <stdarg.h>

// For string manipulation functions.
#include <string.h>

// For other string manipulation functions (e.g. isspace()).
#include <ctype.h>

// For sleep().
#include <unistd.h>


//
// --- Constants and types -----------------------------------------------------
//

#define PARAMETERS_FILENAME          "input.general"
#define OPERATIONS_FILENAME          "input.operations"
#define INPUT_COMMENT_CHAR           '#'
#define OUTPUT_COMMENT_CHAR          '%'

#define BLIS_FILE_PREFIX_STR         "libblis_test"
#define BLIS_FILEDATA_PREFIX_STR     "blis"

#define INPUT_BUFFER_SIZE            256
#define MAX_FILENAME_LENGTH          256
#define MAX_BINARY_NAME_LENGTH       256
#define MAX_FUNC_STRING_LENGTH       24
#define FLOPS_PER_UNIT_PERF          1e9

#define MAX_NUM_MSTORAGE             4
#define MAX_NUM_VSTORAGE             5
#define MAX_NUM_DATATYPES            4
#define MAX_NUM_PARAMETERS           7
#define MAX_NUM_DIMENSIONS           3
#define MAX_NUM_OPERANDS             5

#define MAX_PASS_STRING_LENGTH       32
#define BLIS_TEST_FAIL_STRING        "FAILURE"
#define BLIS_TEST_WARN_STRING        "MARGINAL"
#define BLIS_TEST_PASS_STRING        "PASS"

#define ON_FAILURE_IGNORE_CHAR       'i'
#define ON_FAILURE_SLEEP_CHAR        's'
#define ON_FAILURE_ABORT_CHAR        'a'

#define SECONDS_TO_SLEEP             3

#define DISABLE_ALL                  0
#define SPECIFY                      1
#define DISABLE                      0
#define ENABLE                       1


#define MAX_PARAM_VALS_PER_TYPE      4
#define BLIS_TEST_PARAM_SIDE_CHARS   "lr"
#define BLIS_TEST_PARAM_UPLO_CHARS   "lu"
#define BLIS_TEST_PARAM_UPLODE_CHARS "dlu"
#define BLIS_TEST_PARAM_TRANS_CHARS  "ncth"
#define BLIS_TEST_PARAM_CONJ_CHARS   "nc"
#define BLIS_TEST_PARAM_DIAG_CHARS   "nu"

#define NUM_PARAM_TYPES         6
typedef enum
{
	BLIS_TEST_PARAM_SIDE      = 0,
	BLIS_TEST_PARAM_UPLO      = 1,
	BLIS_TEST_PARAM_UPLODE    = 2,
	BLIS_TEST_PARAM_TRANS     = 3,
	BLIS_TEST_PARAM_CONJ      = 4,
	BLIS_TEST_PARAM_DIAG      = 5,
} param_t;


#define MAX_STORE_VALS_PER_TYPE      4
#define BLIS_TEST_MSTORE_CHARS       "crg"
#define BLIS_TEST_VSTORE_CHARS       "crji"


#define NUM_OPERAND_TYPES       2
typedef enum
{
	BLIS_TEST_MATRIX_OPERAND  = 0,
	BLIS_TEST_VECTOR_OPERAND  = 1,
} operand_t;


typedef enum
{
	BLIS_TEST_DIMS_MNK        = 0,
	BLIS_TEST_DIMS_MN         = 1,
	BLIS_TEST_DIMS_MK         = 2,
	BLIS_TEST_DIMS_M          = 3,
} dimset_t;


typedef enum
{
	BLIS_TEST_SEQ_FRONT_END   = 0,
	BLIS_TEST_MT_FRONT_END    = 1,
} mt_impl_t;



typedef struct
{
	unsigned int  n_repeats;
	unsigned int  n_mstorage;
	unsigned int  n_vstorage;
	char          storage[ NUM_OPERAND_TYPES ][ MAX_NUM_MSTORAGE + 1 ];
	unsigned int  mix_all_storage;
	int           gs_spacing;
	unsigned int  n_datatypes;
	char          datatype_char[ MAX_NUM_DATATYPES + 1 ];
	num_t         datatype[ MAX_NUM_DATATYPES + 1 ];
	dim_t         p_first;
	dim_t         p_max;
	dim_t         p_inc;
	char          reaction_to_failure;
	unsigned int  output_matlab_files;
} test_params_t;


typedef struct
{
	// parent test_ops_t struct
	struct test_ops_s*   ops;

	int           front_seq;
	dim_t         n_dims;
	dimset_t      dimset;
	int           dim_spec[ MAX_NUM_DIMENSIONS ];
	unsigned int  n_params;
	char          params[ MAX_NUM_PARAMETERS ];
	bool_t        test_done;

} test_op_t;


typedef struct test_ops_s
{
	// util
	test_op_t randv;
	test_op_t randm;

	// level-1v
	test_op_t addv;
	test_op_t axpyv;
	test_op_t copyv;
	test_op_t dotv;
	test_op_t dotxv;
	test_op_t fnormv;
	test_op_t scalv;
	test_op_t scal2v;
	test_op_t setv;
	test_op_t subv;
	
	// level-1m
	test_op_t addm;
	test_op_t axpym;
	test_op_t copym;
	test_op_t fnormm;
	test_op_t scalm;
	test_op_t scal2m;
	test_op_t setm;
	test_op_t subm;

	// level-2
	test_op_t gemv;
	test_op_t ger;
	test_op_t hemv;
	test_op_t her;
	test_op_t her2;
	test_op_t symv;
	test_op_t syr;
	test_op_t syr2;
	test_op_t trmv;
	test_op_t trsv;

	// level-3
	test_op_t gemm;
	test_op_t hemm;
	test_op_t herk;
	test_op_t her2k;
	test_op_t symm;
	test_op_t syrk;
	test_op_t syr2k;
	test_op_t trmm;
	test_op_t trmm3;
	test_op_t trsm;

} test_ops_t;


typedef struct
{
	double failwarn;
	double warnpass;
} thresh_t;


//
// --- Prototypes --------------------------------------------------------------
//

void libblis_test_utility_ops( test_params_t* params, test_ops_t* ops );
void libblis_test_level1m_ops( test_params_t* params, test_ops_t* ops );
void libblis_test_level1v_ops( test_params_t* params, test_ops_t* ops );
void libblis_test_level2_ops( test_params_t* params, test_ops_t* ops );
void libblis_test_level3_ops( test_params_t* params, test_ops_t* ops );

void libblis_test_read_params_file( char* input_filename, test_params_t* params );
void libblis_test_read_ops_file( char* input_filename, test_ops_t* ops );

void libblis_test_read_op_info( test_ops_t*  ops,
                                FILE*        input_stream,
                                dimset_t     dimset,
                                unsigned int n_params,
                                test_op_t*   op );

// --- Struct output ---

void libblis_test_output_params_struct( FILE* os, test_params_t* params );
void libblis_test_output_op_struct( FILE* os, test_op_t* op, char* op_str );

// --- Mapping ---

char*   libblis_test_get_string_for_result( double residual, num_t dt,
                                            thresh_t* thresh );
param_t libblis_test_get_param_type_for_char( char p_type );
operand_t libblis_test_get_operand_type_for_char( char o_type );
dim_t   libblis_test_get_n_dims_from_dimset( dimset_t dimset );
dim_t   libblis_test_get_dim_from_prob_size( int dim_spec, dim_t p_size );

// --- Parameter/storage string generation ---

void libblis_test_fill_param_strings( char*         p_str,
                                      char**        chars_for_param,
                                      unsigned int  n_params,
                                      unsigned int  n_param_combos,
                                      char**        pc_str );
void carryover( unsigned int* c,
                unsigned int* n_vals_for_param,
                unsigned int  n_params );

// --- Operation driver ---

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
                                             double* ) );    // residual

// --- Generate experiment string labels ---

void libblis_test_build_function_string( char*        prefix_str,
                                         char*        op_str,
                                         char         dt_char,
                                         unsigned int n_param_combos,
                                         char*        pc_str,
                                         char*        sc_str,
                                         char*        func_str );

void libblis_test_build_filename_string( char*        prefix_str,
                                         char*        op_str,
                                         char*        funcname_str );

void libblis_test_build_col_labels_string( test_op_t* op, char* l_str );

void fill_string_with_n_spaces( char* str, unsigned int n_spaces );

// --- Create object ---

void libblis_test_mobj_create( test_params_t* params, num_t dt, trans_t trans, char storage, dim_t m, dim_t n, obj_t* a );
void libblis_test_vobj_create( test_params_t* params, num_t dt, char storage, dim_t m, obj_t* x );

// --- Global string initialization ---

void libblis_test_init_strings( void );

// --- System wrappers ---

void libblis_test_sleep( void );
void libblis_test_abort( void );

// --- File I/O wrappers ---

void libblis_test_fopen_mfile( char* op_str, mt_impl_t impl, FILE** output_stream );
void libblis_test_fclose_mfile( FILE* output_stream );
void libblis_test_fopen_check_stream( char* filename_str, FILE* stream );

void libblis_test_read_next_line( char* buffer, FILE* input_stream );

// --- Custom fprintf-related ---

void libblis_test_fprintf( FILE* output_stream, char* message, ... );
void libblis_test_fprintf_c( FILE* output_stream, char* message, ... );
void libblis_test_printf_info( char* message, ... );
void libblis_test_printf_infoc( char* message, ... );
void libblis_test_printf_error( char* message, ... );

void libblis_test_parse_message( FILE* output_stream, char* message, va_list args );
void libblis_test_parse_command_line( int argc, char** argv );


//
// --- Test module headers -----------------------------------------------------
//

// Utility operations
#include "test_randv.h"
#include "test_randm.h"

// Level-1v
#include "test_addv.h"
#include "test_axpyv.h"
#include "test_copyv.h"
#include "test_dotv.h"
#include "test_dotxv.h"
#include "test_fnormv.h"
#include "test_scalv.h"
#include "test_scal2v.h"
#include "test_setv.h"
#include "test_subv.h"

// Level-1m
#include "test_addm.h"
#include "test_axpym.h"
#include "test_copym.h"
#include "test_fnormm.h"
#include "test_scalm.h"
#include "test_scal2m.h"
#include "test_setm.h"
#include "test_subm.h"

// Level-2
#include "test_gemv.h"
#include "test_ger.h"
#include "test_hemv.h"
#include "test_her.h"
#include "test_her2.h"
#include "test_symv.h"
#include "test_syr.h"
#include "test_syr2.h"
#include "test_trmv.h"
#include "test_trsv.h"

// Level-3
#include "test_gemm.h"
#include "test_hemm.h"
#include "test_herk.h"
#include "test_her2k.h"
#include "test_symm.h"
#include "test_syrk.h"
#include "test_syr2k.h"
#include "test_trmm.h"
#include "test_trmm3.h"
#include "test_trsm.h"

