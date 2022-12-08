#include "blis_test.h"
#include "blis_utils.h"

void libblis_test_randm_impl(iface_t iface, obj_t* x );

double libblis_test_randm_check(test_params_t* params, obj_t* x );

double libblis_test_bitrp_randm(
  test_params_t* params,
  iface_t        iface,
  obj_t*         x,
  num_t          dt
) {
  double resid = 0.0;
    resid = libblis_test_matrix_check(params, x);
  return resid;
}

double libblis_test_op_randm (
  test_params_t* params,
  iface_t        iface,
  char*          dc_str,
  char*          sc_str,
  tensor_t*      dim
)
{
  num_t        datatype;
  dim_t        m, n;
  char         x_store;
  double       resid = 0.0;
  obj_t        x;

  // Use the datatype of the first char in the datatype combination string.
  bli_param_map_char_to_blis_dt( dc_str[0], &datatype );

  // Map the dimension specifier to actual dimensions.
  m = dim->m;
  n = dim->n;

  // Extract the storage character for each operand.
  x_store = sc_str[0];

  // Create the test objects.
  libblis_test_mobj_create( params, datatype, BLIS_NO_TRANSPOSE, x_store, m, n, &x );

  libblis_test_randm_impl( iface, &x );

#ifndef __GTEST_VALGRIND_TEST__
  if(params->bitrp) {
    resid = libblis_test_bitrp_randm( params, iface, &x, datatype);
  }
  else {
    resid = libblis_test_randm_check( params, &x );
  }
#endif

  libblis_test_check_empty_problem( &x, &resid );

  // Free the test objects.
  libblis_test_obj_free( &x );

  return abs(resid);
}

void libblis_test_randm_impl (
  iface_t   iface,
  obj_t*    x
)
{
  switch ( iface )
  {
    case BLIS_TEST_SEQ_FRONT_END:
      bli_randm( x );
    break;

    default:
      libblis_test_printf_error( "Invalid interface type.\n" );
  }
}

void absummval(dim_t m, dim_t n, float* x, inc_t rs_x, inc_t cs_x, float* sum_x)
{
  float  abs_chi1;
  float  sum;
  dim_t  i, j;

  bli_sset0s(sum);
  for ( j = 0; j < n; j++ ) {
    for ( i = 0; i < m; i++ ) {
      float* chi1 = x + i*rs_x + j*cs_x;
      bli_ssabval2s(*chi1, abs_chi1);
      bli_ssadds(abs_chi1, sum);
    }
  }

 	bli_sscopys(sum, *sum_x);
}

void absummval(dim_t m, dim_t n, double* x, inc_t rs_x, inc_t cs_x, double* sum_x)
{
  double  abs_chi1;
  double  sum;
  dim_t  i, j;

  bli_dset0s(sum);
  for ( j = 0; j < n; j++ ) {
    for ( i = 0; i < m; i++ ) {
      double* chi1 = x + i*rs_x + j*cs_x;
      bli_ddabval2s(*chi1, abs_chi1);
      bli_ddadds(abs_chi1, sum);
    }
  }

 	bli_ddcopys(sum, *sum_x);
}

void absummval(dim_t m, dim_t n, scomplex* x, inc_t rs_x, inc_t cs_x, float* sum_x)
{
  float  abs_chi1;
  float  sum;
  dim_t  i, j;

  bli_sset0s(sum);
  for ( j = 0; j < n; j++ ) {
    for ( i = 0; i < m; i++ ) {
      scomplex* chi1 = x + i*rs_x + j*cs_x;
      bli_csabval2s(*chi1, abs_chi1);
      bli_ssadds(abs_chi1, sum);
    }
  }

 	bli_sscopys(sum, *sum_x);
}

void absummval(dim_t m, dim_t n, dcomplex* x, inc_t rs_x, inc_t cs_x, double* sum_x)
{
  double  abs_chi1;
  double  sum;
  dim_t  i, j;

  bli_dset0s(sum);
  for ( j = 0; j < n; j++ ) {
    for ( i = 0; i < m; i++ ) {
      dcomplex* chi1 = x + i*rs_x + j*cs_x;
      bli_zdabval2s(*chi1, abs_chi1);
      bli_ddadds(abs_chi1, sum);
    }
  }

 	bli_ddcopys(sum, *sum_x);
}

template <typename T, typename U>
void absumm ( obj_t* x, obj_t* sum_x )
{
  dim_t     m         = bli_obj_length( x );
  dim_t     n         = bli_obj_width( x );

  T*        buf_x     =(T*) bli_obj_buffer_at_off( x );
  inc_t     rs_x      = bli_obj_row_stride( x );
  inc_t     cs_x      = bli_obj_col_stride( x );

  U*        buf_sum_x = (U*)bli_obj_buffer_at_off( sum_x );

  // Invoke the function.
  absummval( m, n, buf_x, rs_x, cs_x, buf_sum_x );
}

double libblis_test_randm_check( test_params_t* params, obj_t* x )
{
  num_t  dt      = bli_obj_dt( x );
  num_t  dt_real = bli_obj_dt_proj_to_real( x );
  dim_t  m_x     = bli_obj_length( x );
  dim_t  n_x     = bli_obj_width( x );
  obj_t  sum;

  //
  // The two most likely ways that randm would fail is if all elements
  // were zero, or if all elements were greater than or equal to one.
  // We check both of these conditions by computing the sum of the
  // absolute values of the elements of x.
  //

  double resid = 0.0;

  bli_obj_scalar_init_detached( dt_real, &sum );

  switch( dt )  {
    case BLIS_FLOAT :
    {
      absumm<float, float>( x, &sum );
      break;
    }
    case BLIS_DOUBLE :
    {
      absumm<double, double>( x, &sum );
      break;
    }
    case BLIS_SCOMPLEX :
    {
      absumm<scomplex, float>( x, &sum );
      break;
    }
    case BLIS_DCOMPLEX :
    {
      absumm<dcomplex, double>( x, &sum );
      break;
    }
    default :
      bli_check_error_code( BLIS_INVALID_DATATYPE );
  }

  if ( bli_is_float( dt_real )) {
    float*  sum_x = (float*)bli_obj_buffer_at_off( &sum );

    if      ( *sum_x == *bli_d0         ) resid = 1.0;
    else if ( *sum_x >= 2.0 * m_x * n_x ) resid = 2.0;
  }
  else /* if ( bli_is_double( dt_real ) )*/  {
    double* sum_x = (double*)bli_obj_buffer_at_off( &sum );

    if      ( *sum_x == *bli_d0         ) resid = 1.0;
    else if ( *sum_x >= 2.0 * m_x * n_x ) resid = 2.0;
  }

  return resid;
}