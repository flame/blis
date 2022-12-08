#include "blis_test.h"
#include "blis_utils.h"
#include "test_copym.h"

// Local prototypes.
void libblis_test_copym_deps(
  thread_data_t* tdata,
  test_params_t* params,
  test_op_t*     op
);

void libblis_test_copym_impl (
  iface_t   iface,
  obj_t*    x,
  obj_t*    y
);

double libblis_test_copym_check (
  test_params_t* params,
  obj_t*         x,
  obj_t*         y
);

double libblis_ref_copym(
  test_params_t* params,
  obj_t*         x,
  obj_t*         y,
  obj_t*         y_save
) {
  double resid = 0.0;

  if((params->bitextf == 0) && (params->oruflw == BLIS_DEFAULT)) {
 	  // Perform checks.
    resid = libblis_test_copym_check( params, x, y );
  }
  else {
   if(params->oruflw == BLIS_DEFAULT) {
      resid = libblis_test_icopym_check( params, x, y, y_save );
    }
    else {
      resid = libblis_test_matrix_check(params, y);
    }
  }
  return resid;
}

double libblis_test_bitrp_copym(
  test_params_t* params,
  iface_t        iface,
  obj_t*         x,
  obj_t*         y,
  obj_t*         r,
  num_t          dt
) {
  double resid = 0.0;
  unsigned int n_repeats = params->n_repeats;
  unsigned int i;

  for(i = 0; i < n_repeats; i++) {
    bli_setm( &BLIS_ONE, r );
    libblis_test_copym_impl( iface, x, r );
    resid = libblis_test_bitrp_matrix(y, r, dt);
  }
  return resid;
}

double libblis_test_op_copym (
  test_params_t* params,
  iface_t        iface,
  char*          dc_str,
  char*          pc_str,
  char*          sc_str,
  tensor_t*      dim
){
  num_t        datatype;
  dim_t        m, n;
  trans_t      transx;
  obj_t        x, y, y_save;
  double       resid = 0.0;

  // Use the datatype of the first char in the datatype combination string.
  bli_param_map_char_to_blis_dt( dc_str[0], &datatype );

  // Map the dimension specifier to actual dimensions.
  m = dim->m;
  n = dim->n;

  // Map parameter characters to BLIS constants.
  bli_param_map_char_to_blis_trans( pc_str[0], &transx );

  // Create test operands (vectors and/or matrices).
  libblis_test_mobj_create( params, datatype, transx,
                           sc_str[0], m, n, &x );
  libblis_test_mobj_create( params, datatype, BLIS_NO_TRANSPOSE,
                           sc_str[1], m, n, &y );
  libblis_test_mobj_create( params, datatype, BLIS_NO_TRANSPOSE,
                           sc_str[1], m, n, &y_save );

  // Randomize x and set y to one.
  if((params->bitextf == 0) && (params->oruflw == BLIS_DEFAULT)) {
    libblis_test_mobj_randomize( params, FALSE, &x );
  }
  else {
    libblis_test_mobj_irandomize( params, &x );
  }

  bli_setm( &BLIS_ONE, &y );

  // Apply the parameters.
  bli_obj_set_conjtrans( transx, &x );

  bli_copym( &y, &y_save );

  libblis_test_copym_impl( iface, &x, &y );

#ifndef __GTEST_VALGRIND_TEST__
  if(params->bitrp) {
    resid = libblis_test_bitrp_copym( params, iface, &x, &y, &y_save, datatype);
  }
  else {
    resid = libblis_ref_copym( params, &x, &y, &y_save);
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

void libblis_test_copym_impl(
  iface_t   iface,
  obj_t*    x,
  obj_t*    y
) {
  switch ( iface )
  {
    case BLIS_TEST_SEQ_FRONT_END:
      bli_copym( x, y );
      break;

    default:
      libblis_test_printf_error( "Invalid interface type.\n" );
  }
}

double libblis_test_copym_check(
  test_params_t* params,
  obj_t*         x,
  obj_t*         y
) {
  num_t  dt_real = bli_obj_dt_proj_to_real( x );

  obj_t  norm_y_r;

  double junk;

  double resid = 0.0;

  //
  // Pre-conditions:
  // - x is randomized.
  //
  // Under these conditions, we assume that the implementation for
  //
  //   y := conjx(x)
  //
  // is functioning correctly if
  //
  //   normfm( y - conjx(x) )
  //
  // is negligible.
  //

  bli_obj_scalar_init_detached( dt_real, &norm_y_r );

  bli_subm( x, y );

  bli_normfm( y, &norm_y_r );

  bli_getsc( &norm_y_r, &resid, &junk );

  return resid;
}