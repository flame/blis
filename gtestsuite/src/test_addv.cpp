#include "blis_test.h"
#include "blis_utils.h"
#include "test_addv.h"

// Local prototypes.
void libblis_test_addv_deps(thread_data_t* tdata,
                            test_params_t* params, test_op_t* op );

void libblis_test_addv_impl (iface_t iface, obj_t* x, obj_t* y);

double libblis_test_addv_check (
  test_params_t* params,
  obj_t*         alpha,
  obj_t*         beta,
  obj_t*         x,
  obj_t*         y
);

double libblis_ref_addv(
  test_params_t* params,
  obj_t*  alpha,
  obj_t*  beta,
  obj_t*  x,
  obj_t*  y,
  obj_t*  y_save
) {
  double resid = 0.0;

  if((params->bitextf == 0) && (params->oruflw == BLIS_DEFAULT)) {
    resid = libblis_test_addv_check(params, alpha, beta, x, y );
  }
  else {
   if(params->oruflw == BLIS_DEFAULT) {
      resid = libblis_test_iaddv_check(params, alpha, beta, x, y, y_save);
    }
    else {
      resid = libblis_test_vector_check(params, y);
    }
  }
  return resid;
}

double libblis_test_bitrp_addv(
  test_params_t* params,
  iface_t        iface,
  obj_t*         x,
  obj_t*         y,
  obj_t*         y_orig,
  obj_t*         r,
  num_t          dt
) {
  double resid = 0.0;
	 unsigned int n_repeats = params->n_repeats;
	 unsigned int i;

  for(i = 0; i < n_repeats; i++) {
    bli_copyv( y_orig, r );
    libblis_test_addv_impl( iface, x, r );
    resid = libblis_test_bitrp_vector(y, r, dt);
  }
  return resid;
}

double libblis_test_op_addv (
  test_params_t* params,
  iface_t        iface,
  char*          dc_str,
  char*          pc_str,
  char*          sc_str,
  tensor_t*      dim
){
  num_t        datatype;
  dim_t        m;
  conj_t       conjx;
  obj_t        alpha, beta;
  obj_t        x, y, y_save;
  double       resid = 0.0;

  // Use the datatype of the first char in the datatype combination string.
  bli_param_map_char_to_blis_dt( dc_str[0], &datatype );

  // Map the dimension specifier to an actual dimension.
  m = dim->m;

  // Map parameter characters to BLIS constants.
  bli_param_map_char_to_blis_conj( pc_str[0], &conjx );

  // Create test scalars.
  bli_obj_scalar_init_detached( datatype, &alpha );
  bli_obj_scalar_init_detached( datatype, &beta );

  // Create test operands (vectors and/or matrices).
  libblis_test_vobj_create( params, datatype, sc_str[0], m, &x );
  libblis_test_vobj_create( params, datatype, sc_str[1], m, &y );
  libblis_test_vobj_create( params, datatype, sc_str[1], m, &y_save );

  // Initialize alpha and beta.
  bli_setsc( -1.0, -1.0, &alpha );
  bli_setsc(  3.0,  3.0, &beta );

  // Set x and y to alpha and beta, respectively.
  bli_setv( &alpha, &x );
  bli_setv( &beta,  &y );

  // Apply the parameters.
  bli_obj_set_conj( conjx, &x );

  bli_copyv( &y, &y_save );

  libblis_test_addv_impl( iface, &x, &y );

#ifndef __GTEST_VALGRIND_TEST__
  if(params->bitrp) {
    obj_t r;

    libblis_test_vobj_create( params, datatype, sc_str[1], m, &r );

    resid = libblis_test_bitrp_addv( params, iface, &x, &y, &y_save,
                                                            &r, datatype);
    bli_obj_free( &r );
  }
  else {
    resid = libblis_ref_addv( params, &alpha, &beta, &x, &y, &y_save );
  }
#endif

  // Zero out performance and residual if output vector is empty.
  libblis_test_check_empty_problem( &y, &resid );

  // Free the test objects.
  libblis_test_obj_free( &x );
  libblis_test_obj_free( &y );
  libblis_test_obj_free( &y_save );

  return abs(resid);
}

void libblis_test_addv_impl (
  iface_t   iface,
  obj_t*    x,
  obj_t*    y
){

  switch ( iface )
  {
    case BLIS_TEST_SEQ_FRONT_END:
      bli_addv( x, y );
      break;

    default:
      libblis_test_printf_error( "Invalid interface type.\n" );
  }
}

double libblis_test_addv_check (
  test_params_t* params,
  obj_t*         alpha,
  obj_t*         beta,
  obj_t*         x,
  obj_t*         y
) {
  num_t  dt      = bli_obj_dt( x );
  num_t  dt_real = bli_obj_dt_proj_to_real( x );
  dim_t  m       = bli_obj_vector_dim( x );

  conj_t conjx   = bli_obj_conj_status( x );

  obj_t  aplusb;
  obj_t  alpha_conj;
  obj_t  norm_r, m_r, temp_r;

  double junk;
  double resid = 0.0;

  //
  // Pre-conditions:
  // - x is set to alpha.
  // - y_orig is set to beta.
  // Note:
  // - alpha and beta should have non-zero imaginary components in the
  //   complex cases in order to more fully exercise the implementation.
  //
  // Under these conditions, we assume that the implementation for
  //
  //   y := y_orig + conjx(x)
  //
  // is functioning correctly if
  //
  //   normfv(y) - sqrt( absqsc( beta + conjx(alpha) ) * m )
  //
  // is negligible.
  //

  bli_obj_scalar_init_detached( dt,      &aplusb );
  bli_obj_scalar_init_detached( dt_real, &temp_r );
  bli_obj_scalar_init_detached( dt_real, &norm_r );
  bli_obj_scalar_init_detached( dt_real, &m_r );

  bli_obj_scalar_init_detached_copy_of( dt, conjx, alpha, &alpha_conj );

  bli_normfv( y, &norm_r );

  bli_copysc( beta, &aplusb );
  bli_addsc( &alpha_conj, &aplusb );

  bli_setsc( ( double )m, 0.0, &m_r );

  bli_absqsc( &aplusb, &temp_r );
  bli_mulsc( &m_r, &temp_r );
  bli_sqrtsc( &temp_r, &temp_r );
  bli_subsc( &temp_r, &norm_r );

  bli_getsc( &norm_r, &resid, &junk );

  return resid;
}