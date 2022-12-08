#include "blis_utils.h"

using namespace std;

extern char libblis_test_pass_string[ MAX_PASS_STRING_LENGTH + 1 ];
extern char libblis_test_warn_string[ MAX_PASS_STRING_LENGTH + 1 ];
extern char libblis_test_fail_string[ MAX_PASS_STRING_LENGTH + 1 ];
extern char libblis_test_overflow_string[ MAX_PASS_STRING_LENGTH + 1 ];
extern char libblis_test_underflow_string[ MAX_PASS_STRING_LENGTH + 1 ];

void libblis_test_fprintf_c( FILE* output_stream, const char* message, ... );
void libblis_test_printf_infoc( const char* message, ... );

char libblis_test_binary_name[ MAX_BINARY_NAME_LENGTH + 1 ];

void carryover( unsigned int* c, unsigned int* n_vals_for_param, unsigned int  n_params );
void libblis_test_ceil_pow2( obj_t* alpha );
void libblis_test_parse_message( FILE* output_stream, const char* message, va_list args ) ;

unsigned int libblis_test_get_n_dims_from_dimset( dimset_t dimset ) ;

void bli_map_blis_to_netlib_trans( trans_t trans, char* blas_trans )
{
    if      ( trans == BLIS_NO_TRANSPOSE      ) *blas_trans = 'N';
    else if ( trans == BLIS_TRANSPOSE         ) *blas_trans = 'T';
    else if ( trans == BLIS_CONJ_NO_TRANSPOSE ) *blas_trans = 'C';
    else if ( trans == BLIS_CONJ_TRANSPOSE    ) *blas_trans = 'C';     //*blas_trans = 'H'
    else
    {
        bli_check_error_code( BLIS_INVALID_TRANS );
    }
}

void bli_param_map_char_to_blas_trans( char trans, trans_t* blas_trans )
{
    if      ( trans == 'n' || trans == 'N' ) *blas_trans = BLIS_NO_TRANSPOSE;
    else if ( trans == 't' || trans == 'T' ) *blas_trans = BLIS_TRANSPOSE;
    else if ( trans == 'c' || trans == 'C' ) *blas_trans = BLIS_CONJ_TRANSPOSE;
    else if ( trans == 'h' || trans == 'H' ) *blas_trans = BLIS_CONJ_NO_TRANSPOSE;
    else
    {
        bli_check_error_code( BLIS_INVALID_TRANS );
    }
}

void bli_param_map_char_to_herk_trans( char trans, trans_t* herk_trans )
{
    if      ( trans == 'n' || trans == 'N' ) *herk_trans = BLIS_NO_TRANSPOSE;
    else if ( trans == 'c' || trans == 'C' ) *herk_trans = BLIS_TRANSPOSE;
    else
    {
        bli_check_error_code( BLIS_INVALID_TRANS );
    }
}

void bli_param_map_char_to_syrk_trans( char trans, trans_t* syrk_trans )
{
    if      ( trans == 'n' || trans == 'N' ) *syrk_trans = BLIS_NO_TRANSPOSE;
    else if ( trans == 't' || trans == 'T' ) *syrk_trans = BLIS_TRANSPOSE;
    else if ( trans == 'c' || trans == 'C' ) *syrk_trans = BLIS_TRANSPOSE;
    else if ( trans == 'h' || trans == 'H' ) *syrk_trans = BLIS_NO_TRANSPOSE;
    else
    {
        bli_check_error_code( BLIS_INVALID_TRANS );
    }
}

void fill_string_with_n_spaces( char* str, unsigned int n_spaces )
{
   unsigned int i;

   // Initialze to empty string in case n_spaces == 0.
   sprintf( str, "%s", "" );

   for ( i = 0; i < n_spaces; ++i )
      sprintf( &str[i], " " );
}

void libblis_test_build_dims_string(test_op_t* op, tensor_t* dim, char* dims_str)
{
    // For level-1f experiments with fusing factors, we grab the fusing
    // factor from the op struct. We do something similar for micro-kernel
    // calls.
    if     ( op->dimset == BLIS_TEST_DIMS_MF )
    {
      sprintf( dims_str, " %5u %5u",
               ( unsigned int )dim->m,
               ( unsigned int ) op->dim_aux[0] );
    }
    else if( op->dimset == BLIS_TEST_DIMS_K )
    {
      sprintf( dims_str, " %5u %5u %5u",
               ( unsigned int ) op->dim_aux[0],
               ( unsigned int ) op->dim_aux[1],
               ( unsigned int )dim->m );
    }
    else if( op->dimset == BLIS_TEST_NO_DIMS )
    {
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
      //for ( i = 0; i < op->n_dims; ++i ) {
      if( (op->n_dims > 0) && (dim->m > 0) )
        sprintf( &dims_str[strlen(dims_str)], " %5u", ( unsigned int ) dim->m );
      if( (op->n_dims > 0) && (dim->n > 0) )
        sprintf( &dims_str[strlen(dims_str)], " %5u", ( unsigned int ) dim->n );
      if( (op->n_dims > 0) && (dim->k > 0) )
        sprintf( &dims_str[strlen(dims_str)], " %5u", ( unsigned int ) dim->k );
    }
}

void libblis_test_read_section_override( test_ops_t*  ops,
    FILE* input_stream, int* override )
{
  char  buffer[ INPUT_BUFFER_SIZE ];

  // Read the line for the section override switch.
  libblis_test_read_next_line( buffer, input_stream );
  sscanf( buffer, "%d ", override );
}

void libblis_test_read_op_info
     (
       test_ops_t*  ops,
       FILE*        input_stream,
       opid_t       opid,
       dimset_t     dimset,
       unsigned int n_params,
       test_op_t*   op
     )
{
  char  buffer[ INPUT_BUFFER_SIZE ];
  char  temp[ INPUT_BUFFER_SIZE ];
  unsigned int   i, p;

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
     libblis_test_printf_error( "Detected too many dimensions (%u) in input file to store.\n", op->n_dims );
  }

  //printf( "n_dims = %u\n", op->n_dims );

  // If there is at least one dimension for the current operation, read the
  // dimension specifications, which encode the actual dimensions or the
  // dimension ratios for each dimension.
  if( op->n_dims > 0 )  {
    libblis_test_read_next_line( buffer, input_stream );

    for( i = 0, p = 0; i < op->n_dims; ++i )
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
	unsigned int i;
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
    char api[50];

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

	if (params->api == API_CBLAS)
      sprintf(    api, "cblas" );
    else if(params->api == API_BLAS)
      sprintf(    api, "blas" );
    else
      sprintf(    api, "blis" );

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

	for ( i = 0; i < BLIS_NAT; ++i )
	{
    im = (ind_t)i;
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
	libblis_test_fprintf_c( os, "api interface                %s\n", api );
	libblis_test_fprintf_c( os, "permutation and combination  %s\n", (params->dimf == 1) ? "yes" : "no" );
	libblis_test_fprintf_c( os, "alpha and beta combination   %s\n", (params->nab == 0) ? "single value" : "multiple values" );
	libblis_test_fprintf_c( os, "integer bitexact test        %s\n", (params->bitextf == 1) ? "enabled" : "disabled" );
	libblis_test_fprintf_c( os, "print cases                  %s\n", (params->passflag == 1) ? "all" : "only failures" );
	libblis_test_fprintf_c( os, "bit-reproducibility          %s\n", (params->bitrp == 1) ? "enabled" : "disabled" );
	libblis_test_fprintf_c( os, "lpgemm memory-format order   %s\n", (params->op_t == 'p') ? "no-reorder" : "reorder" );
	libblis_test_fprintf_c( os, "-----------------------------------------------------------\n" );
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

		for ( i = BLIS_IND_FIRST; i < BLIS_IND_LAST+1; ++i )
		{
            im = (ind_t)i;
			if ( im != BLIS_1M && im != BLIS_NAT )
				params->ind_enable[ im ] = 0;
		}
	}
}

void libblis_test_output_op_struct( FILE* os, test_op_t* op, char* op_str )
{

  dimset_t dimset = op->dimset;

  if      ( dimset == BLIS_TEST_DIMS_MNK )	{
   libblis_test_fprintf_c( os, "%s m n k                  %d %d %d\n", op_str,
                                   op->dim_spec[0], op->dim_spec[1], op->dim_spec[2] );
  }
  else if ( dimset == BLIS_TEST_DIMS_MN )	{
    libblis_test_fprintf_c( os, "%s m n                    %d %d\n", op_str,
                                   op->dim_spec[0], op->dim_spec[1] );
  }
  else if ( dimset == BLIS_TEST_DIMS_MK )	{
    libblis_test_fprintf_c( os, "%s m k                    %d %d\n", op_str,
                                   op->dim_spec[0], op->dim_spec[1] );
  }
  else if ( dimset == BLIS_TEST_DIMS_M ||
            dimset == BLIS_TEST_DIMS_MF )	{
    libblis_test_fprintf_c( os, "%s m                      %d\n", op_str,
                                   op->dim_spec[0] );
  }
  else if ( dimset == BLIS_TEST_DIMS_K )	{
    libblis_test_fprintf_c( os, "%s k                      %d\n", op_str,
                                   op->dim_spec[0] );
  }
  else if ( dimset == BLIS_TEST_NO_DIMS )	{
   // Do nothing.
  }
  else	{
    libblis_test_printf_error( "Invalid dimension combination.\n" );
  }

  if ( op->n_params > 0 )
    libblis_test_fprintf_c( os, "%s operand params         %s\n", op_str, op->params );
  else
    libblis_test_fprintf_c( os, "%s operand params         %s\n", op_str, "(none)" );

  libblis_test_fprintf_c( os, "\n" );
  libblis_test_fprintf( os, "\n" );
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
  else  {
   r_val = BLIS_TEST_PARAM_SIDE;
   libblis_test_printf_error( "Invalid parameter character.\n" );
  }

  return r_val;
}

operand_t libblis_test_get_operand_type_for_char( char o_type )
{
  operand_t r_val;

  if      ( o_type == 'm' ) r_val = BLIS_TEST_MATRIX_OPERAND;
  else if ( o_type == 'v' ) r_val = BLIS_TEST_VECTOR_OPERAND;
  else  {
    r_val = BLIS_TEST_MATRIX_OPERAND;
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
  else	{
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

  for ( n_dims = 0; *cp != '\0'; ++n_dims )  {
    while ( isspace( *cp ) )  {
      ++cp;
    }

    while ( isdigit( *cp ) )  {
       ++cp;
    }
  }

  return n_dims;
}

dim_t libblis_test_get_dim_from_prob_size( int dim_spec, unsigned int p_size )
{
  dim_t dim;

  if ( dim_spec < 0 )
    dim = p_size / bli_abs(dim_spec);
  else
    dim = dim_spec;

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
  for ( i = 0; i < n_params; ++i ) 	{
    if ( p_spec_str[i] == '?' )
      n_vals_for_param[i] = strlen( chars_for_param[i] );
    else
      n_vals_for_param[i] = 1;
  }

  // Allocate an array with one digit per parameter. We will use
  // this array to keep track of our progress as we canonically move
  // though all possible parameter combinations.
  counter = ( unsigned int* ) malloc( n_params * sizeof( unsigned int ) );

  // Initialize all values in c to zero.
  for ( i = 0; i < n_params; ++i )
    counter[i] = 0;

  for ( pci = 0; pci < n_param_combos; ++pci ) 	{
    // Iterate backwards through each parameter string we create, since we
    // want to form (for example, if the parameters are transa and conjx:
    // (1) nn, (2) nc, (3) cn, (4) cc, (5) tn, (6) tc, (7) hn, (8) hc.
    for ( i = 0, pi = n_params - 1; i < n_params; --pi, ++i )		{
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
    if ( pci < n_param_combos - 1 ) {
      // Increment the least-most significant counter.
      counter[ n_params - 1 ]++;

      // Perform "carryover" if needed.
      carryover( &counter[ n_params - 1 ], &n_vals_for_param[ n_params - 1 ], n_params );
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
  if ( n_params == 1 )
    return;
  else  {
    if ( *c == *n_vals_for_param )  {
      *c = 0;
      *(c-1) += 1;
      carryover( c-1, n_vals_for_param-1, n_params-1 );
    }
  }
}

void libblis_test_vobj_randomize( test_params_t* params, bool normalize, obj_t* x )
{
  if( params->rand_method == BLIS_TEST_RAND_REAL_VALUES )
    bli_randv( x );
  else // if ( params->rand_method == BLIS_TEST_RAND_NARROW_POW2 )
    bli_randnv( x );

  if( normalize )
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

void libblis_test_build_filename_string( const char*  prefix_str,
                                         char*        op_str,
                                         char*        funcname_str )
{
  	sprintf( funcname_str, "%s_%s.m", prefix_str, op_str );
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

void libblis_test_read_next_line( char* buffer, FILE* input_stream )
{
  char temp[ INPUT_BUFFER_SIZE ];

  // We want to read at least one line, so we use a do-while loop.
  do  {
    // Read the next line into a temporary buffer and check success.
    if ( fgets( temp, INPUT_BUFFER_SIZE-1, input_stream ) == NULL )  {
      if ( feof( input_stream ) )
        libblis_test_printf_error( "Error reading input file: encountered unexpected EOF." );
      else
        libblis_test_printf_error( "Error (non-EOF) reading input file." );
    }
  // We continue to read lines into buffer until the line is neither
  // commented nor blank.
  }while ( temp[0] == INPUT_COMMENT_CHAR || temp[0] == '\n' ||
      temp[0] == ' ' || temp[0] == '\t' );

  // Save the string in temp, up to first white space character, into buffer.
  //sscanf( temp, "%s ", buffer );
  strcpy( buffer, temp );

  //printf( "libblis_test_read_next_line() read: %s\n", buffer );
}

void libblis_test_fprintf( FILE* output_stream, const char* message, ... )
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

void libblis_test_fprintf_c( FILE* output_stream, const char* message, ... )
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

void libblis_test_printf_info( const char* message, ... )
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

void libblis_test_printf_infoc( const char* message, ... )
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

void libblis_test_printf_error( const char* message, ... )
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

void libblis_test_parse_message( FILE* output_stream, const char* message, va_list args )
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
	 for ( c = 0; message[c] != '\0'; )	{
    if ( message[c] != '%' )  {
      fprintf( output_stream, "%c", message[c] );
      c += 1;
    }
    else if ( message[c] == '%' && message[c+1] == '%' ) {// handle escaped '%' chars.
      fprintf( output_stream, "%c", message[c] );
      c += 2;
    }
    else {
      // Save the format string if there is one.
      format_spec[0] = '%';
      for ( c += 1, cf = 1; strchr( "udefsc", message[c] ) == NULL; ++c, ++cf )  {
        format_spec[cf] = message[c];
      }

      // Add the final type specifier, and null-terminate the string.
      format_spec[cf] = message[c];
      format_spec[cf+1] = '\0';

      // Switch based on type, since we can't predict what will
      // va_args() will return.
      switch ( message[c] )			{
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

void libblis_test_check_empty_problem( obj_t* c, double* resid )
{
	 if ( bli_obj_has_zero_dim( c ) )	{
    *resid = 0.0;
 	}
}

bool libblis_test_op_is_done( test_op_t* op )
{
	return op->test_done;
}

int libblis_test_op_is_disabled( test_op_t* op )
{
  int r_val;

  // If there was at least one individual override, then an op test is
  // disabled if it is NOT equal to ENABLE_ONLY. If there were no
  // individual overrides, then an op test is disabled if it is equal
  // to DISABLE.
  if ( op->ops->indiv_over == TRUE ) {
    if ( op->op_switch != ENABLE_ONLY )
      r_val = TRUE;
    else
      r_val = FALSE;
  }
  else {// if ( op->ops->indiv_over == FALSE )
    if ( op->op_switch == DISABLE )
      r_val = TRUE;
    else
      r_val = FALSE;
  }
  return r_val;
}

char* libblis_test_get_result(double resid, const thresh_t* thresh,
                                  char* dc_str, test_params_t* params )  {
  char* r_val;
  num_t dt;
  bli_param_map_char_to_blis_dt(dc_str[0], &dt );

  if(params->bitextf == 1 ) {
      r_val = libblis_test_pass_string;
      double dmx = 0.0;
      double dmn = 0.0;
      switch( dt )  {
        case BLIS_FLOAT :
        case BLIS_SCOMPLEX :
        {
          dmx = (double)(std::numeric_limits<float>::max)();;
          dmn = (double)(std::numeric_limits<float>::min)();;
          break;
        }
        case BLIS_DOUBLE :
        case BLIS_DCOMPLEX :
        {
          dmx = (std::numeric_limits<double>::max)();
          dmn = (std::numeric_limits<double>::min)();
          break;
        }
        default :
          bli_check_error_code( BLIS_INVALID_DATATYPE );
      }


      if( params->oruflw == BLIS_OVERFLOW ) {
        if(( bli_isnan( resid ) || bli_isinf( resid ) ||
           ( resid > dmx )) && ( resid != 0.0 ))
        {
            r_val = libblis_test_overflow_string;
        }
      }
      else if( params->oruflw == BLIS_UNDERFLOW ) {
        if(( bli_isnan( resid ) || bli_isinf( resid ) ||
           ( resid < dmn )) && ( resid != 0.0 ))
        {
            r_val = libblis_test_underflow_string;
        }
      }
      else { /* params->oruflw == BLIS_DEFAULT */
          if ( resid != 0 ) r_val = libblis_test_fail_string;
          else              r_val = libblis_test_pass_string;
      }
  }
  else if ( bli_isnan( resid ) || bli_isinf( resid ) )  {
    r_val = libblis_test_fail_string;
  }
  else {
     // Check the result against the thresholds.
    if      ( resid > thresh[dt].failwarn ) r_val = libblis_test_fail_string;
    else if ( resid > thresh[dt].warnpass ) r_val = libblis_test_warn_string;
    else                                    r_val = libblis_test_pass_string;
  }
  return r_val;
}

bool libblis_test_get_string_for_result( double resid, num_t dt,
                                const thresh_t* thresh, char *r_val ) {
  bool res;
  // Before checking against the thresholds, make sure the residual is
  // neither NaN nor Inf. (Note that bli_isnan() and bli_isinf() are
  // both simply wrappers to the isnan() and isinf() macros defined
  // defined in math.h.)
  if ( bli_isnan( resid ) || bli_isinf( resid ) )	{
    r_val = libblis_test_fail_string;
    res = false;
  }
  else	{
    // Check the result against the thresholds.
    if ( resid > thresh[dt].failwarn ) {
      r_val = libblis_test_fail_string;
      res = false;
    }
    else if ( resid > thresh[dt].warnpass ) {
      r_val = libblis_test_warn_string;
      res = true;
    }
    else  {
      r_val = libblis_test_pass_string;
      res = true;
    }
  }
  return res;
}

int libblis_test_util_is_disabled( test_op_t* op ) {
	if ( op->ops->util_over == DISABLE )
   return TRUE;
	else
   return FALSE;
}

int libblis_test_l1v_is_disabled( test_op_t* op ) {
	if ( op->ops->l1v_over == DISABLE )
   return TRUE;
	else
   return FALSE;
}

int libblis_test_l1m_is_disabled( test_op_t* op ) {
	if ( op->ops->l1m_over == DISABLE )
   return TRUE;
	else
   return FALSE;
}

int libblis_test_l1f_is_disabled( test_op_t* op ) {
	if ( op->ops->l1f_over == DISABLE )
   return TRUE;
	else
   return FALSE;
}

int libblis_test_l2_is_disabled( test_op_t* op )
{
   if( op->ops->l2_over == DISABLE )
       return TRUE;
   else
       return FALSE;
}

int libblis_test_l3ukr_is_disabled( test_op_t* op )
{
   if( op->ops->l3ukr_over == DISABLE )
       return TRUE;
   else
       return FALSE;
}

int libblis_test_l3_is_disabled( test_op_t* op )
{
    if( op->ops->l3_over == DISABLE )
        return TRUE;
    else
        return FALSE;
}

// ---
int libblis_test_dt_str_has_sp_char_str( int n, char* str )
{
  for ( int i = 0 ; i < n ; ++i )
  {
      if ( str[i] == 's' ||  str[i] == 'c' )
          return TRUE;
  }
  return FALSE;
}

int libblis_test_dt_str_has_sp_char( test_params_t* params )
{
  	 return libblis_test_dt_str_has_sp_char_str( params->n_datatypes,
	                                            params->datatype_char );
}

// ---
int libblis_test_dt_str_has_dp_char_str( int n, char* str )
{
   for ( int i = 0 ; i < n ; ++i )
   {
       if ( str[i] == 'd' || str[i] == 'z' )
         return TRUE;
   }
   return FALSE;
}

int libblis_test_dt_str_has_dp_char( test_params_t* params )
{
	   return libblis_test_dt_str_has_dp_char_str( params->n_datatypes,
	                                            params->datatype_char );
}

// ---
int libblis_test_dt_str_has_rd_char_str( int n, char* str )
{
    for ( int i = 0; i < n; ++i )
    {
        if ( str[i] == 's' || str[i] == 'd' )
            return TRUE;
    }
    return FALSE;
}

int libblis_test_dt_str_has_rd_char( test_params_t* params ) {
 	  return libblis_test_dt_str_has_rd_char_str( params->n_datatypes,
	                                            params->datatype_char );
}

// ---
int libblis_test_dt_str_has_cd_char_str( int n, char* str )
{
    for ( int i = 0; i < n; ++i )
    {
        if ( str[i] == 'c' ||  str[i] == 'z' )
            return TRUE;
    }
    return FALSE;
}

int libblis_test_dt_str_has_cd_char( test_params_t* params )
{
  	 return libblis_test_dt_str_has_cd_char_str( params->n_datatypes,
	                                            params->datatype_char );
}

// ---
unsigned int libblis_test_count_combos (
  unsigned int n_operands,
  char*        spec_str,
  char**       char_sets
)
{
    unsigned int n_combos = 1;

    for ( unsigned int i = 0; i < n_operands; ++i )
    {
        if ( spec_str[i] == '?' )
            n_combos *= strlen( char_sets[i] );
    }

    return n_combos;
}

char libblis_test_proj_dtchar_to_precchar( char dt_char )
{
    char r_val = dt_char;
    if     ( r_val == 'c' )
        r_val = 's';
    else if( r_val == 'z' )
     r_val = 'd';

    return r_val;
}

////////////////////////////////////////////////////////////////////////
#ifdef __GTESTSUITE_MALLOC_BUFFER__
void libblis_test_alloc_buffer( obj_t* a )
{
  dim_t  n_elem = 0;
  dim_t  m, n;
  siz_t  elem_size;
  siz_t  buffer_size;
  void*  p;
  inc_t  rs,cs,is;

  bli_obj_free( a );

  // Query the dimensions of the object we are allocating.
  m = bli_obj_length( a );
  n = bli_obj_width( a );
  rs = bli_obj_row_stride( a );
  cs = bli_obj_col_stride( a );
  is = bli_obj_imag_stride( a );

  // Query the size of one element.
  elem_size = bli_obj_elem_size( a );

  // Determine how much object to allocate.
  if ( m == 0 || n == 0 )
  {
     // For empty objects, set n_elem to zero. Row and column strides
     // should remain unchanged (because alignment is not needed).
     n_elem = 0;
  }
  else
  {
     // The number of elements to allocate is given by the distance from
     // the element with the lowest address (usually {0, 0}) to the element
     // with the highest address (usually {m-1, n-1}), plus one for the
     // highest element itself.
     n_elem = (m-1) * bli_abs( rs ) + (n-1) * bli_abs( cs ) + 1;
  }

  // Handle the special case where imaginary stride is larger than
  // normal.
  if ( bli_obj_is_complex( a ) )
  {
     // Notice that adding is/2 works regardless of whether the
     // imaginary stride is unit, something between unit and
     // 2*n_elem, or something bigger than 2*n_elem.
     n_elem = bli_abs( is ) / 2 + n_elem;
  }

  // Compute the size of the total buffer to be allocated, which includes
  // padding if the leading dimension was increased for alignment purposes.
  buffer_size = ( siz_t )n_elem * elem_size;

  // Allocate the buffer.
  p = malloc( buffer_size );

  // Set individual fields.
  bli_obj_set_buffer( p, a );

}
#endif

void libblis_test_obj_free( obj_t* x )
{
#ifdef __GTESTSUITE_MALLOC_BUFFER__
  void* p;
	// Don't dereference obj if it is NULL.
   if ( x != NULL )
   {
     p = (void *)bli_obj_buffer( x );
     free( p );
   }
#else
   bli_obj_free( x );
#endif
}

void libblis_test_vobj_create( test_params_t* params, num_t dt, char storage,
                                    dim_t m, obj_t* x )
{
	 dim_t gs = params->gs_spacing;

  // Column vector (unit stride)
  if ( storage == 'c' ) {
    bli_obj_create( dt, m, 1,  1,  m,    x );
#ifdef __GTESTSUITE_MALLOC_BUFFER__
    libblis_test_alloc_buffer( x );
#endif
  }
  // Row vector (unit stride)
  else if ( storage == 'r' ) {
    bli_obj_create( dt, 1, m,  m,  1,    x );
#ifdef __GTESTSUITE_MALLOC_BUFFER__
    libblis_test_alloc_buffer( x );
#endif
  }
  // Column vector (non-unit stride)
  else if ( storage == 'j' ) {
    bli_obj_create( dt, m, 1,  gs, gs*m, x );
#ifdef __GTESTSUITE_MALLOC_BUFFER__
    libblis_test_alloc_buffer( x );
#endif
  }
  // Row vector (non-unit stride)
  else if ( storage == 'i' ) {
    bli_obj_create( dt, 1, m,  gs*m, gs, x );
#ifdef __GTESTSUITE_MALLOC_BUFFER__
    libblis_test_alloc_buffer( x );
#endif
  }
  else	{
    libblis_test_printf_error( "Invalid storage character: %c\n", storage );
  }

}

void libblis_test_mobj_create( test_params_t* params, num_t dt, trans_t trans,
    char storage, dim_t m, dim_t n, obj_t* a )  {

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
    if ( storage == 'c' )
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

#ifdef __GTESTSUITE_MALLOC_BUFFER__
    libblis_test_alloc_buffer( a );
#endif

}

double libblis_test_vector_check( test_params_t* params, obj_t* y )
{
    double resid = 0.0;
    num_t dt       = bli_obj_dt( y );
    f77_int len    = bli_obj_vector_dim( y );
    f77_int incy   = bli_obj_vector_inc( y );

    vflg_t flg =   params->oruflw;

    switch( dt )
    {
        case BLIS_FLOAT :
        {
            float*   Y = (float*) bli_obj_buffer( y );
            resid = libblis_vector_check_real<float>( flg, len, incy, Y );
            break;
        }
        case BLIS_DOUBLE :
        {
            double*   Y = (double*) bli_obj_buffer( y );
            resid = libblis_vector_check_real<double>( flg, len, incy, Y );
            break;
        }
        case BLIS_SCOMPLEX :
        {
            scomplex* Y = (scomplex*) bli_obj_buffer( y );
            resid = libblis_vector_check_cmplx<scomplex, float>( flg, len, incy, Y );
            break;
        }
        case BLIS_DCOMPLEX :
        {
            dcomplex* Y = (dcomplex*) bli_obj_buffer( y );
            resid = libblis_vector_check_cmplx<dcomplex, double>( flg, len, incy, Y );
            break;
        }
        default :
            bli_check_error_code( BLIS_INVALID_DATATYPE );
    }
    return resid;
}

double libblis_test_matrix_check( test_params_t* params, obj_t* c )
{
    dim_t  rsc, csc;
    double resid = 0.0;
    num_t dt = bli_obj_dt( c );
    dim_t  M = bli_obj_length( c );
    dim_t  N = bli_obj_width( c );

    if( bli_obj_row_stride( c ) == 1 )
    {
      rsc = 1;
      csc = bli_obj_col_stride( c );
    }
    else
    {
      rsc = bli_obj_row_stride( c );
      csc = 1 ;
    }

    vflg_t flg =   params->oruflw;

    switch( dt )
    {
        case BLIS_FLOAT :
        {
            float* C = (float*) bli_obj_buffer( c );
            resid = libblis_matrix_check_real<float>( flg, C, M, N, rsc, csc );
            break;
        }
        case BLIS_DOUBLE :
        {
            double* C = (double*) bli_obj_buffer( c );
            resid = libblis_matrix_check_real<double>( flg, C, M, N, rsc, csc );
            break;
        }
        case BLIS_SCOMPLEX :
        {
            scomplex* C = (scomplex*) bli_obj_buffer( c );
            resid = libblis_matrix_check_cmplx<scomplex, float>( flg, C, M, N, rsc, csc );
            break;
        }
        case BLIS_DCOMPLEX :
        {
            dcomplex* C = (dcomplex*) bli_obj_buffer( c );
            resid = libblis_matrix_check_cmplx<dcomplex, double>( flg, C, M, N, rsc, csc );
            break;
        }
        default :
            bli_check_error_code( BLIS_INVALID_DATATYPE );
    }
    return resid;
}

double libblis_test_bitrp_vector( obj_t* x, obj_t* y, num_t dt )
{
    double resid = 0.0;
    f77_int len    = bli_obj_vector_dim( x );
    f77_int incy   = bli_obj_vector_inc( x );

    switch( dt )
    {
        case BLIS_FLOAT :
        {
            float* X = (float*) bli_obj_buffer( x );
            float* Y = (float*) bli_obj_buffer( y );
            resid = computediffrv<float>( len, incy, X, Y );
            break;
        }
        case BLIS_DOUBLE :
        {
            double* X = (double*) bli_obj_buffer( x );
            double* Y = (double*) bli_obj_buffer( y );
            resid = computediffrv<double>( len, incy, X, Y );
            break;
        }
        case BLIS_SCOMPLEX :
        {
            scomplex* X = (scomplex*) bli_obj_buffer( x );
            scomplex* Y = (scomplex*) bli_obj_buffer( y );
            resid = computediffiv<scomplex>( len, incy, X, Y );
            break;
        }
        case BLIS_DCOMPLEX :
        {
            dcomplex* X = (dcomplex*) bli_obj_buffer( x );
            dcomplex* Y = (dcomplex*) bli_obj_buffer( y );
            resid = computediffiv<dcomplex>( len, incy, X, Y );
            break;
        }
        default :
            bli_check_error_code( BLIS_INVALID_DATATYPE );
    }
    return resid;
}

double libblis_test_bitrp_matrix( obj_t* c, obj_t* r, num_t dt )
{
    dim_t  rsc, csc;
    double resid = 0.0;
    dim_t  M = bli_obj_length( c );
    dim_t  N = bli_obj_width( c );

    if( bli_obj_row_stride( c ) == 1 )
    {
        rsc = 1;
        csc = bli_obj_col_stride( c );
    }
    else
    {
        rsc = bli_obj_row_stride( c );
        csc = 1 ;
    }

    switch( dt )
    {
        case BLIS_FLOAT :
        {
            float* C = (float*) bli_obj_buffer( c );
            float* R = (float*) bli_obj_buffer( r );
            resid = computediffrm<float>( M, N, C, R, rsc, csc );
            break;
        }
        case BLIS_DOUBLE :
        {
            double* C = (double*) bli_obj_buffer( c );
            double* R = (double*) bli_obj_buffer( r );
            resid = computediffrm<double>( M, N, C, R, rsc, csc );
            break;
        }
        case BLIS_SCOMPLEX :
        {
            scomplex* C = (scomplex*) bli_obj_buffer( c );
            scomplex* R = (scomplex*) bli_obj_buffer( r );
            resid = computediffim<scomplex>( M, N, C, R, rsc, csc );
            break;
        }
        case BLIS_DCOMPLEX :
        {
            dcomplex* C = (dcomplex*) bli_obj_buffer( c );
            dcomplex* R = (dcomplex*) bli_obj_buffer( r );
            resid = computediffim<dcomplex>( M, N, C, R, rsc, csc );
            break;
        }
        default :
            bli_check_error_code( BLIS_INVALID_DATATYPE );
    }
    return resid;
}

void conjugate_tensor( obj_t* aa, num_t dt )
{
    dim_t rs, cs;
    dim_t m = bli_obj_length( aa );
    dim_t n = bli_obj_width( aa );
    rs = bli_obj_row_stride( aa ) ;
    cs = bli_obj_col_stride( aa ) ;

    switch( dt )
    {
        case BLIS_FLOAT :
          break;
        case BLIS_DOUBLE :
          break;
        case BLIS_SCOMPLEX :
        {
          scomplex*  aap  = (scomplex*) bli_obj_buffer( aa );
          conjugatematrix<scomplex>( aap, m, n, rs, cs );
          break;
        }
        case BLIS_DCOMPLEX :
        {
          dcomplex*  aap  = (dcomplex*) bli_obj_buffer( aa );
          conjugatematrix<dcomplex>( aap, m, n, rs, cs );
          break;
        }
        default :
          bli_check_error_code( BLIS_INVALID_DATATYPE );
    }
    return;
}
