#include "blis_test.h"
#include "blis_utils.h"
#include "lpgemm_utils.h"

#ifdef BLIS_ENABLE_ADDONS
static int8_t* aocl_reorder(int8_t* b, dim_t k, dim_t n, dim_t ldb) {
  siz_t b_reorder_buf_siz_req;
  b_reorder_buf_siz_req = aocl_get_reorder_buf_size_u8s8s16os16('B',k, n);
  int8_t* b_reorder = ( int8_t* )malloc( b_reorder_buf_siz_req );
  aocl_reorder_u8s8s16os16('B', b, b_reorder, k, n, ldb );
  return b_reorder;
}

static void aocl_gemm_driver_u8s8s16os16(
  uint32_t       n_repeats,
  char           stor_order,
  char           transa,
  char           transb,
  dim_t          m,
  dim_t          n,
  dim_t          k,
  int16_t        alpha,
  uint8_t*       a,
  dim_t          lda,
  char           reordera,
  int8_t*        b,
  dim_t          ldb,
  char           reorderb,
  int16_t        beta,
  int16_t*       c,
  dim_t          ldc,
  aocl_post_op*  post_op
) {
  uint32_t i;
  char storage = stor_order;
  for ( i = 0; i < n_repeats; i++ ) {
    memset( ( void* ) c, 0, sizeof( int16_t ) * (m * n) );

    aocl_gemm_u8s8s16os16(storage, transa, transb, m, n, k,
                          alpha, a, lda, reordera, b,
                          ldb, reorderb, beta, c, ldc, post_op );
/*
    aocl_gemm_u8s8s16os16(storage, (const char)transa, (const char)transb, (const dim_t)m, (const dim_t)n, (const dim_t)k,
                          (const int8_t)alpha, (const uint8_t *)a, (const dim_t)lda, (const char)reordera, (const int8_t *)b,
                          (const dim_t)ldb, (const char)reorderb, (const int8_t)beta, c, (const dim_t)ldc, post_op );
*/
  }
}

static void mat_mul_driver_u8s8s16os16(
  test_params_t* params,
  char           stor_order,
  char           transa,
  char           transb,
  dim_t          m,
  dim_t          n,
  dim_t          k,
  int16_t        alpha,
  uint8_t*       a,
  dim_t          lda,
  int8_t*        b,
  dim_t          ldb,
  int16_t        beta,
  int16_t*       c,
  dim_t          ldc,
  aocl_post_op*  post_op
) {
  char reordera;
  char reorderb;

  if ( ( params->op_t == 'p' ) || ( params->op_t == 'P' ) )
  {
    /* No reordering of B.*/
    reordera = 'n';
    reorderb = 'n';

    aocl_gemm_driver_u8s8s16os16(params->n_repeats, stor_order, transa, transb, m, n, k,
                       alpha, a, lda, reordera, b, ldb, reorderb, beta, c, ldc, post_op );
  }
  else if ( ( params->op_t == 'r' ) || ( params->op_t == 'R' ) )
  {
    /* Reorder B.*/
    reordera = 'n';
    reorderb = 'r';

    int8_t* b_reorder = aocl_reorder( b, k, n, ldb );

    aocl_gemm_driver_u8s8s16os16(params->n_repeats, stor_order, transa, transb, m, n, k,
              alpha, a, lda, reordera, b_reorder, ldb, reorderb, beta, c, ldc, post_op );

    free( b_reorder );
  }
}

static double mat_mul_accuracy_check_driver_u8s8s16os16
    (
      const char stor_order,
      dim_t    m,
      dim_t    n,
      dim_t    k,
      int16_t  alpha,
      uint8_t* a,
      dim_t    lda,
      int8_t*  b,
      dim_t    ldb,
      int16_t  beta,
      int16_t* c,
      dim_t    ldc,
      int16_t* c_ref,
      aocl_post_op*  post_op,
      bool     dscale_out
    )
{
  double resid = 0.0;
  dim_t rs_a = lda;
  dim_t cs_a = 1;
  dim_t rs_b = ldb;
  dim_t cs_b = 1;
  dim_t rs_c = ldc;
  dim_t cs_c = 1;

  if ( ( stor_order == 'C' ) || ( stor_order == 'c' ) )
  {
    rs_a = 1;
    cs_a = lda;
    rs_b = 1;
    cs_b = ldb;
    rs_c = 1;
    cs_c = ldc;
  }

  resid = mat_mul_accuracy_check_driver<uint8_t,int8_t,int16_t,int16_t,float>( m, n, k,
              alpha, a, rs_a, cs_a, b, rs_b, cs_b, beta, c, rs_c, cs_c, c_ref, post_op,
              dscale_out );

  return resid;
}
#endif

double libblis_test_op_gemm_u8s8s16os16 (
  test_params_t* params,
  char*          pc_str,
  char*          sc_str,
  tensor_t*      dim,
  atom_t         alpv,
  atom_t         betv
){
  double       resid = 0.0;
#ifdef BLIS_ENABLE_ADDONS
  dim_t        m, n, k;
  dim_t        lda, ldb, ldc;
  char*        post_ops_str = NULL;
  char*        post_ops_str_dest = NULL;
  aocl_post_op* post_op = NULL;
  char stor_order = sc_str[0];
  char transa = 'n';       //pc_str[0];
  char transb = 'n';       //pc_str[1];
  bool dscale_out = false;

  stor_order = ( ( stor_order == 'r' ) || ( stor_order == 'R' ) ||
			     ( stor_order == 'c' ) || ( stor_order == 'C' ) ) ?
				 	stor_order : 'r';

  // Map the dimension specifier to actual dimensions.
  m = dim->m;
  k = dim->k;
  n = dim->n;

  if( ( stor_order == 'r' ) || ( stor_order == 'R' ) )
  {
    lda = k; // a = mxk;
    ldb = n; // b = kxn;
    ldc = n; // c = mxn;
  }
  else if ( ( stor_order == 'C' ) || ( stor_order == 'c' ) )
  {
    lda = m;
    ldb = k;
    ldc = m;
    cout << "Column Major not supported" << endl;
    return resid;
  }


  /* Get 64 byte aligned memory.*/
  uint8_t* a = ( uint8_t* ) malloc( sizeof( uint8_t ) * m * k );

  int8_t* b = ( int8_t* ) malloc( sizeof( int8_t ) * n * k );

  int16_t* c = ( int16_t* ) malloc( sizeof( int16_t ) * m * n );
  memset( ( void* ) c, 0, sizeof( int16_t ) * m * n );

  int16_t* c_ref = ( int16_t* ) malloc( sizeof( int16_t ) * m * n );
  memset( ( void* ) c_ref, 0, sizeof( int16_t ) * m * n );

  int16_t alpha = 2;
  int16_t beta = 9;

  fill_array<uint8_t>( a, ( m * k ) );
  fill_array<int8_t>( b, ( k * n ) );

  if ( post_ops_str != NULL )
  {
    post_ops_str_dest = strdup( post_ops_str );
  }

  if( ( post_ops_str != NULL ) || ( dscale_out ) )
  {
    post_op = lpgemm_create_post_ops_struct<int16_t,float>
                           ( m, n, post_ops_str_dest, dscale_out );
    if ( post_op == NULL )
    {
      printf(" post op struct allocation failure, returning.n");
      return -1;
    }
  }

  mat_mul_driver_u8s8s16os16(params, stor_order, transa, transb, m, n, k, alpha,
                            a, lda, b, ldb, beta, c, ldc, post_op );

#ifndef __GTEST_VALGRIND_TEST__
  resid = mat_mul_accuracy_check_driver_u8s8s16os16( stor_order, m, n, k, alpha,
                            a, lda, b, ldb, beta, c, ldc, c_ref, post_op,
                            dscale_out );
#endif

  if ( post_op != NULL )
    lpgemm_destroy_post_ops_struct( post_op );

  // Free the test objects.
  if ( a != NULL )
    free( a );
  if ( b != NULL )
    free( b );
  if ( c != NULL )
    free( c );
  if ( c_ref != NULL )
    free( c_ref );

  if ( post_ops_str_dest != NULL )
    free( post_ops_str_dest );

#else
  cout << "CPU Arch do not support AVX512" << endl;
#endif

  return resid;
}