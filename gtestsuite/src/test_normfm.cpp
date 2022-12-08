#include "blis_test.h"
#include "blis_utils.h"
#include "test_normfm.h"

// Local prototypes.
void libblis_test_normfm_deps(
  thread_data_t* tdata,
  test_params_t* params,
  test_op_t*     op
);

void libblis_test_normfm_impl(
  iface_t   iface,
  obj_t*    x,
  obj_t*    norm
);

double libblis_test_normfm_check(
  test_params_t* params,
  obj_t*         beta,
  obj_t*         x,
  obj_t*         norm
);

double libblis_ref_normfm(
  test_params_t* params,
  obj_t*  beta,
  obj_t*  x,
  obj_t*  norm
) {
  double resid = 0.0;

  if((params->bitextf == 0) && (params->oruflw == BLIS_DEFAULT)) {
    resid = libblis_test_normfm_check( params, beta, x, norm );
  }
  else {
    if(params->oruflw == BLIS_DEFAULT) {
      resid = libblis_test_inormfm_check( params, x, norm);
    }
    else {
      resid = libblis_test_matrix_check(params, x);
    }
  }
  return resid;
}

double libblis_test_bitrp_normfm(
  test_params_t* params,
  iface_t        iface,
  obj_t*         x,
  obj_t*         norm,
  obj_t*         r,
  num_t          dt
) {
  double resid = 0.0;
	 unsigned int n_repeats = params->n_repeats;
	 unsigned int i;
  num_t dt_real = bli_dt_proj_to_real( dt );

  for(i = 0; i < n_repeats; i++) {
    bli_obj_scalar_init_detached( dt_real,  r );
    libblis_test_normfm_impl( iface, x, r );
    resid = libblis_test_bitrp_matrix(norm, r, dt);
  }
  return resid;
}

double libblis_test_op_normfm (
  test_params_t* params,
  iface_t        iface,
  char*          dc_str,
  char*          pc_str,
  char*          sc_str,
  tensor_t*      dim
){
  num_t        datatype;
  num_t        dt_real;
  dim_t        m, n;
  obj_t        beta, norm;
  obj_t        x;
  double       resid = 0.0;

  // Use the datatype of the first char in the datatype combination string.
  bli_param_map_char_to_blis_dt( dc_str[0], &datatype );

  // Compute the real projection of the chosen datatype.
  dt_real = bli_dt_proj_to_real( datatype );

  // Map the dimension specifier to actual dimensions.
  m = dim->m;
  n = dim->n;

  // Create test scalars.
  bli_obj_scalar_init_detached( datatype, &beta );
  bli_obj_scalar_init_detached( dt_real,  &norm );

  // Create test operands (vectors and/or matrices).
  libblis_test_mobj_create( params, datatype, BLIS_NO_TRANSPOSE,
                            sc_str[0], m, n, &x );

  if((params->bitextf == 0) && (params->oruflw == BLIS_DEFAULT)) {
    // Initialize beta to 2 - 2i.
    bli_setsc( 2.0, -2.0, &beta );
    // Set all elements of x to beta.
    bli_setm( &beta, &x );
  }
  else {
    libblis_test_mobj_irandomize( params, &x );
  }

  libblis_test_normfm_impl( iface, &x, &norm );

#ifndef __GTEST_VALGRIND_TEST__
  if(params->bitrp) {
    obj_t r;

    resid = libblis_test_bitrp_normfm( params, iface, &x, &norm, &r, datatype);

    bli_obj_free( &r );
  }
  else {
    resid = libblis_ref_normfm( params, &beta, &x, &norm );
  }
#endif

  // Zero out performance and residual if input matrix is empty.
  libblis_test_check_empty_problem( &x, &resid );

  // Free the test objects.
  libblis_test_obj_free( &x );

  return abs(resid);
}

void libblis_test_normfm_impl(
  iface_t   iface,
  obj_t*    x,
  obj_t*    norm
){
  switch ( iface )
  {
    case BLIS_TEST_SEQ_FRONT_END:
      bli_normfm( x, norm );
      break;

    default:
      libblis_test_printf_error( "Invalid interface type.\n" );
  }
}

double libblis_test_normfm_check (
  test_params_t* params,
  obj_t*         beta,
  obj_t*         x,
  obj_t*         norm
){
  num_t  dt_real = bli_obj_dt_proj_to_real( x );
  dim_t  m       = bli_obj_length( x );
  dim_t  n       = bli_obj_width( x );

  obj_t  m_r, n_r, temp_r;

  double junk;
  double resid = 0.0;
  //
  // Pre-conditions:
  // - x is set to beta.
  // Note:
  // - beta should have a non-zero imaginary component in the complex
  //   cases in order to more fully exercise the implementation.
  //
  // Under these conditions, we assume that the implementation for
  //
  //   norm := normfm( x )
  //
  // is functioning correctly if
  //
  //   norm = sqrt( absqsc( beta ) * m * n )
  //
  // where m and n are the dimensions of x.
  //

  bli_obj_scalar_init_detached( dt_real, &temp_r );
  bli_obj_scalar_init_detached( dt_real, &m_r );
  bli_obj_scalar_init_detached( dt_real, &n_r );

  bli_setsc( ( double )m, 0.0, &m_r );
  bli_setsc( ( double )n, 0.0, &n_r );

  bli_absqsc( beta, &temp_r );
  bli_mulsc( &m_r, &temp_r );
  bli_mulsc( &n_r, &temp_r );
  bli_sqrtsc( &temp_r, &temp_r );
  bli_subsc( &temp_r, norm );

  bli_getsc( norm, &resid, &junk );

  return resid;
}