#include "blis_test.h"
#include "blis_utils.h"
#include "test_xpbym.h"

// Local prototypes.
void libblis_test_xpbym_deps(
  thread_data_t* tdata,
  test_params_t* params,
  test_op_t*     op
);

void libblis_test_xpbym_impl(
  iface_t   iface,
  obj_t*    x,
  obj_t*    beta,
  obj_t*    y
);

double libblis_test_xpbym_check(
  test_params_t* params,
  obj_t*         x,
  obj_t*         beta,
  obj_t*         y,
  obj_t*         y_save
);

double libblis_ref_xpbym(
  test_params_t* params,
  obj_t*         x,
  obj_t*         beta,
  obj_t*         y,
  obj_t*         y_orig
){
  double resid = 0.0;

  if((params->bitextf == 0) && (params->oruflw == BLIS_DEFAULT)) {
 	  // Perform checks.
    resid = libblis_test_xpbym_check( params, x, beta, y, y_orig );
  }
  else {
    resid = libblis_test_ixpbym_check( params, x, beta, y, y_orig );
  }
  return resid;
}

double libblis_test_bitrp_xpbym(
  test_params_t* params,
  iface_t        iface,
  obj_t*         x,
  obj_t*         beta,
  obj_t*         y,
  obj_t*         y_orig,
  obj_t*         r,
  num_t          dt
) {
  double resid = 0.0;
	 unsigned int n_repeats = params->n_repeats;
	 unsigned int i;

  for(i = 0; i < n_repeats; i++) {
    bli_copym( y_orig, r );
    libblis_test_xpbym_impl( iface, x, beta, r );
    resid = libblis_test_bitrp_vector(y, r, dt);
  }
  return resid;
}

double libblis_test_op_xpbym (
  test_params_t* params,
  iface_t        iface,
  char*          dc_str,
  char*          pc_str,
  char*          sc_str,
  tensor_t*      dim,
  atom_t         alpha
){
  num_t        datatype;
  dim_t        m, n;
  trans_t      transx;
  obj_t        x, beta, y;
  obj_t        y_save;
  double       resid = 0.0;

  // Use the datatype of the first char in the datatype combination string.
  bli_param_map_char_to_blis_dt( dc_str[0], &datatype );

  // Map the dimension specifier to actual dimensions.
   m = dim->m;
   n = dim->n;

  // Map parameter characters to BLIS constants.
  bli_param_map_char_to_blis_trans( pc_str[0], &transx );

  // Create test scalars.
  bli_obj_scalar_init_detached( datatype, &beta );

  // Create test operands (vectors and/or matrices).
  libblis_test_mobj_create( params, datatype, transx,
                            sc_str[0], m, n, &x );
  libblis_test_mobj_create( params, datatype, BLIS_NO_TRANSPOSE,
                            sc_str[0], m, n, &y );
  libblis_test_mobj_create( params, datatype, BLIS_NO_TRANSPOSE,
                            sc_str[0], m, n, &y_save );

  // Set beta.
  if ( bli_obj_is_real( &y ) )
    bli_setsc( -2.0,  0.0, &beta );
  else
    bli_setsc(  0.0, -2.0, &beta );

  // Randomize and save y.
  if((params->bitextf == 0) && (params->oruflw == BLIS_DEFAULT)) {
    libblis_test_mobj_randomize( params, FALSE, &x );
    libblis_test_mobj_randomize( params, FALSE, &y );
  } else {
    libblis_test_mobj_irandomize( params, &x );
    libblis_test_mobj_irandomize( params, &y );
  }
  bli_copym( &y, &y_save );

  // Apply the parameters.
  bli_obj_set_conjtrans( transx, &x );

  libblis_test_xpbym_impl( iface, &x, &beta, &y );

#ifndef __GTEST_VALGRIND_TEST__
  if(params->bitrp) {
    obj_t r;

    libblis_test_mobj_create( params, datatype, BLIS_NO_TRANSPOSE,
                                                sc_str[0], m, n, &r );

    resid = libblis_test_bitrp_xpbym(params, iface, &x, &beta, &y,
                                                    &y_save, &r, datatype);

    bli_obj_free( &r );
  }
  else {
    resid = libblis_ref_xpbym( params, &x, &beta, &y, &y_save );
  }
#endif

  // Zero out performance and residual if output matrix is empty.
  libblis_test_check_empty_problem( &y, &resid );

  // Free the test objects.
  libblis_test_obj_free( &x );
  libblis_test_obj_free( &y );
  libblis_test_obj_free( &y_save );

  return abs(resid);
}

void libblis_test_xpbym_impl(
  iface_t   iface,
  obj_t*    x,
  obj_t*    beta,
  obj_t*    y
){
  switch ( iface ){
    case BLIS_TEST_SEQ_FRONT_END:
      bli_xpbym( x, beta, y );
      break;

    default:
      libblis_test_printf_error( "Invalid interface type.\n" );
  }
}

double libblis_test_xpbym_check(
  test_params_t* params,
  obj_t*         x,
  obj_t*         beta,
  obj_t*         y,
  obj_t*         y_orig
){
  num_t  dt      = bli_obj_dt( y );
  num_t  dt_real = bli_obj_dt_proj_to_real( y );

  dim_t  m       = bli_obj_length( y );
  dim_t  n       = bli_obj_width( y );

  obj_t  x_temp, y_temp;
  obj_t  norm;

  double junk;
  double resid = 0.0;

  //
  // Pre-conditions:
  // - x is randomized.
  // - y_orig is randomized.
  // Note:
  // - alpha should have a non-zero imaginary component in the complex
  //   cases in order to more fully exercise the implementation.
  //
  // Under these conditions, we assume that the implementation for
  //
  //   y := beta * y_orig + conjx(x)
  //
  // is functioning correctly if
  //
  //   normfm( y - ( beta * y_orig + conjx(x) ) )
  //
  // is negligible.
  //

  bli_obj_scalar_init_detached( dt_real, &norm );

  bli_obj_create( dt, m, n, 0, 0, &x_temp );
  bli_obj_create( dt, m, n, 0, 0, &y_temp );

  bli_copym( x,      &x_temp );
  bli_copym( y_orig, &y_temp );

  bli_scalm( beta, &y_temp );
  bli_addm( &x_temp, &y_temp );

  bli_subm( &y_temp, y );
  bli_normfm( y, &norm );
  bli_getsc( &norm, &resid, &junk );

  bli_obj_free( &x_temp );
  bli_obj_free( &y_temp );

  return resid;
}