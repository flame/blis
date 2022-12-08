#include "blis_test.h"
#include "blis_utils.h"

// Local prototypes.
void libblis_test_setv_deps (
  thread_data_t* tdata,
  test_params_t* params,
  test_op_t*     op
);

void libblis_test_setv_impl (
  iface_t   iface,
  obj_t*    beta,
  obj_t*    x
);

double libblis_test_setv_check (
  test_params_t* params,
  obj_t*         beta,
  obj_t*         x
);

double libblis_ref_setv(
  test_params_t* params,
  obj_t*         beta,
  obj_t*         x
) {
  double resid = 0.0;

  if((params->bitextf == 0) && (params->oruflw == BLIS_DEFAULT)) {
    resid = libblis_test_setv_check( params, beta, x );
  }
  return resid;
}

double libblis_test_bitrp_setv(
  test_params_t* params,
  iface_t        iface,
  obj_t*         beta,
  obj_t*         x,
  obj_t*         r,
  num_t          dt
){

  double resid = 0.0;
	 unsigned int n_repeats = params->n_repeats;
	 unsigned int i;

  for(i = 0; i < n_repeats; i++) {
    libblis_test_setv_impl( iface, beta, r );
    resid = libblis_test_bitrp_matrix(x, r, dt);
  }

  return resid;
}

double libblis_test_op_setv (
  test_params_t* params,
  iface_t        iface,
  char*          dc_str,
  char*          pc_str,
  char*          sc_str,
  tensor_t*      dim
){
  num_t        datatype;
  dim_t        m;
  obj_t        beta;
  obj_t        x;
  double       resid = 0.0;

  // Use the datatype of the first char in the datatype combination string.
  bli_param_map_char_to_blis_dt( dc_str[0], &datatype );

  // Map the dimension specifier to an actual dimension.
  m = dim->m;

  // Create test scalars.
  bli_obj_scalar_init_detached( datatype, &beta );

  // Create test operands (vectors and/or matrices).
  libblis_test_vobj_create( params, datatype, sc_str[0], m, &x );

	 // Initialize beta to unit.
 	bli_copysc( &BLIS_ONE, &beta );

  if((params->bitextf == 0) && (params->oruflw == BLIS_DEFAULT)) {
    libblis_test_vobj_randomize( params, FALSE, &x );
  }
  else {
    libblis_test_vobj_irandomize( params, &x );
  }

  libblis_test_setv_impl( iface, &beta, &x );

#ifndef __GTEST_VALGRIND_TEST__
  if(params->bitrp) {
    obj_t r;

    libblis_test_vobj_create( params, datatype, sc_str[0], m, &r );

    resid = libblis_test_bitrp_setv( params, iface, &beta, &x, &r, datatype);

    bli_obj_free( &r );
  }
  else {
    resid = libblis_ref_setv( params, &beta, &x);
  }
#endif
  // Zero out performance and residual if output vector is empty.
  libblis_test_check_empty_problem( &x, &resid );

  // Free the test objects.
  libblis_test_obj_free( &x );

  return abs(resid);
}

void libblis_test_setv_impl (
  iface_t   iface,
  obj_t*    beta,
  obj_t*    x
) {
  switch ( iface ) {
    case BLIS_TEST_SEQ_FRONT_END:
      bli_setv( beta, x );
      break;

    default:
      libblis_test_printf_error( "Invalid interface type.\n" );
  }
}

double libblis_test_setv_check (
  test_params_t* params,
  obj_t*         beta,
  obj_t*         x
) {
  num_t dt_x     = bli_obj_dt( x );
  dim_t m_x      = bli_obj_vector_dim( x );
  inc_t inc_x    = bli_obj_vector_inc( x );
  void* buf_x    = (void*)bli_obj_buffer_at_off( x );
  void* buf_beta = (void*)bli_obj_buffer_for_1x1( dt_x, beta );
  dim_t i;

  double resid = 0.0;

  //
  // The easiest way to check that setv was successful is to confirm
  // that each element of x is equal to beta.
  //

  if      ( bli_obj_is_float( x ) ) {
    float*    chi1      = (float*)buf_x;
    float*    beta_cast = (float*)buf_beta;

    for ( i = 0; i < m_x; ++i ) {
      if ( !bli_seq( *chi1, *beta_cast ) ) {
        resid = 1.0;
        return resid;
      }
      chi1 += inc_x;
    }
  }
  else if ( bli_obj_is_double( x ) ) {
    double*   chi1      = (double*)buf_x;
    double*   beta_cast = (double*)buf_beta;

    for ( i = 0; i < m_x; ++i ) {
      if ( !bli_deq( *chi1, *beta_cast ) ) {
        resid = 1.0;
        return resid;
      }
      chi1 += inc_x;
    }
  }
  else if ( bli_obj_is_scomplex( x ) ) {
    scomplex* chi1      = (scomplex*)buf_x;
    scomplex* beta_cast = (scomplex*)buf_beta;

    for ( i = 0; i < m_x; ++i ) {
      if ( !bli_ceq( *chi1, *beta_cast ) ) {
        resid = 1.0;
        return resid;
      }
      chi1 += inc_x;
    }
  }
  else /* if ( bli_obj_is_dcomplex( x ) ) */{
    dcomplex* chi1      = (dcomplex*)buf_x;
    dcomplex* beta_cast = (dcomplex*)buf_beta;

    for ( i = 0; i < m_x; ++i ) {
      if ( !bli_zeq( *chi1, *beta_cast ) ) {
        resid = 1.0;
        return resid;
      }
      chi1 += inc_x;
    }
  }
  return resid;
}