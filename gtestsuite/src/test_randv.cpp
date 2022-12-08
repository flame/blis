#include "blis_test.h"
#include "blis_utils.h"

void libblis_test_randv_impl(iface_t iface, obj_t* x );

double libblis_test_randv_check(test_params_t* params, obj_t* x );

double libblis_test_bitrp_randv(
  test_params_t* params,
  iface_t        iface,
  obj_t*         x,
  num_t          dt
)
{
  double resid = 0.0;
    resid = libblis_test_vector_check(params, x);
  return resid;
}

double libblis_test_op_randv (
  test_params_t* params,
  iface_t        iface,
  char*          dc_str,
  char*          sc_str,
  tensor_t*      dim
)
{
  num_t        datatype;
  dim_t        m;
  char         x_store;
  obj_t        x;
  double       resid = 0.0;

  // Use the datatype of the first char in the datatype combination string.
  bli_param_map_char_to_blis_dt( dc_str[0], &datatype );

	 // Map the dimension specifier to an actual dimension.
  m = dim->m;

  // Extract the storage character for each operand.
  x_store = sc_str[0];

  // Create the test objects.
  libblis_test_vobj_create( params, datatype, x_store, m, &x );

  libblis_test_randv_impl( iface, &x );

#ifndef __GTEST_VALGRIND_TEST__
  if(params->bitrp) {
    resid = libblis_test_bitrp_randv( params, iface, &x, datatype);
  }
  else {
    resid = libblis_test_randv_check( params, &x );
  }
#endif

  libblis_test_check_empty_problem( &x, &resid );

  // Free the test objects.
  libblis_test_obj_free( &x );

  return abs(resid);
}

void libblis_test_randv_impl(
  iface_t iface,
  obj_t*  x
)
{
  switch ( iface )
  {
    case BLIS_TEST_SEQ_FRONT_END:
      bli_randv( x );
    break;

    default:
      libblis_test_printf_error( "Invalid interface type.\n" );
  }
}

double libblis_test_randv_check (test_params_t* params, obj_t* x )
{
  num_t  dt_real = bli_obj_dt_proj_to_real( x );
  dim_t  m_x     = bli_obj_vector_dim( x );
  obj_t  sum;

  double resid = 0.0;

  bli_obj_scalar_init_detached( dt_real, &sum );

  bli_norm1v( x, &sum );

  if (bli_is_float( dt_real )) {
    float*  sum_x = (float*)bli_obj_buffer_at_off( &sum );

    if      ( *sum_x == *bli_d0   ) resid = 1.0;
    else if ( *sum_x >= 2.0 * m_x ) resid = 2.0;
  }
  else /* if ( bli_is_double(dt_real )) */ {
    double* sum_x = (double*)bli_obj_buffer_at_off( &sum );

    if      ( *sum_x == *bli_d0   ) resid = 1.0;
    else if ( *sum_x >= 2.0 * m_x ) resid = 2.0;
  }

  return resid;
}