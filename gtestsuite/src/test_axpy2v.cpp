#include "blis_test.h"
#include "blis_utils.h"
#include "test_axpy2v.h"

// Local prototypes.
void libblis_test_axpy2v_deps (
  thread_data_t* tdata,
  test_params_t* params,
  test_op_t*     op
);

void libblis_test_axpy2v_impl (
  iface_t   iface,
  obj_t*    alpha1,
  obj_t*    alpha2,
  obj_t*    x,
  obj_t*    y,
  obj_t*    z,
  cntx_t*   cntx
);

double libblis_test_axpy2v_check (
  test_params_t* params,
  obj_t*         alpha1,
  obj_t*         alpha2,
  obj_t*         x,
  obj_t*         y,
  obj_t*         z,
  obj_t*         z_orig
);

double libblis_ref_axpy2v (
  test_params_t* params,
  obj_t*         alpha1,
  obj_t*         alpha2,
  obj_t*         x,
  obj_t*         y,
  obj_t*         z,
  obj_t*         z_orig
){
  double resid = 0.0;

  if((params->bitextf == 0) && (params->oruflw == BLIS_DEFAULT)) {
    resid = libblis_test_axpy2v_check( params, alpha1, alpha2, x, y, z, z_orig );
  }
  else {
    if(params->oruflw == BLIS_DEFAULT) {
      resid = libblis_test_iaxpy2v_check( params, alpha1, alpha2, x, y, z, z_orig );
    }
    else {
      resid = libblis_test_vector_check(params, y);
    }
  }
  return resid;
}

double libblis_test_bitrp_axpy2v(
  test_params_t* params,
  iface_t        iface,
  obj_t*         alpha1,
  obj_t*         alpha2,
  obj_t*         x,
  obj_t*         y,
  obj_t*         z,
  cntx_t*        cntx,
  obj_t*         z_orig,
  obj_t*         r,
  num_t          dt
) {
  double resid = 0.0;
	 unsigned int n_repeats = params->n_repeats;
	 unsigned int i;

  for(i = 0; i < n_repeats; i++) {
    bli_copyv( z_orig, r );
    libblis_test_axpy2v_impl( iface, alpha1, alpha2, x, y, r, cntx);
    resid = libblis_test_bitrp_vector(z, r, dt);
  }
  return resid;
}

double libblis_test_op_axpy2v (
  test_params_t* params,
  iface_t        iface,
  char*          dc_str,
  char*          pc_str,
  char*          sc_str,
  tensor_t*      dim,
  atom_t         alpv,
  atom_t         betv
) {
  num_t        datatype;
  dim_t        m;
  conj_t       conjx, conjy;
  obj_t        alpha1, alpha2, x, y, z;
  obj_t        z_save;
  cntx_t*      cntx;
  double       resid = 0.0;

  // Query a context.
  cntx = bli_gks_query_cntx();

  // Use the datatype of the first char in the datatype combination string.
  bli_param_map_char_to_blis_dt( dc_str[0], &datatype );

  // Map the dimension specifier to an actual dimension.
  m = dim->m;

  // Map parameter characters to BLIS constants.
  bli_param_map_char_to_blis_conj( pc_str[0], &conjx );
  bli_param_map_char_to_blis_conj( pc_str[1], &conjy );

  // Create test scalars.
  bli_obj_scalar_init_detached( datatype, &alpha1 );
  bli_obj_scalar_init_detached( datatype, &alpha2 );

  // Create test operands (vectors and/or matrices).
  libblis_test_vobj_create( params, datatype, sc_str[0], m, &x );
  libblis_test_vobj_create( params, datatype, sc_str[1], m, &y );
  libblis_test_vobj_create( params, datatype, sc_str[2], m, &z );
  libblis_test_vobj_create( params, datatype, sc_str[2], m, &z_save );

  // Randomize x and y, and save y.
  if((params->bitextf == 0) && (params->oruflw == BLIS_DEFAULT)) {
    // Set alpha.
    if ( bli_obj_is_real( &z ) )	{
      bli_setsc(  alpv.real,  0.0, &alpha1 );
      bli_setsc(  betv.real,  0.0, &alpha2 );
    }
    else	{
      bli_setsc(  alpv.real,  (alpv.real/0.8), &alpha1 );
      bli_setsc(  betv.real,  (betv.real/1.2), &alpha2 );
    }
    libblis_test_vobj_randomize( params, TRUE, &x );
    libblis_test_vobj_randomize( params, TRUE, &y );
    libblis_test_vobj_randomize( params, TRUE, &z );
  } else {
    int32_t xx = (int32_t)alpv.real;
    int32_t yy = (int32_t)betv.real;
    if ( bli_obj_is_real( &z ) )	{
      bli_setsc( (double)xx,  0.0, &alpha1 );
      bli_setsc( (double)yy,  0.0, &alpha2 );
    }
    else	{
      int32_t ac = (int32_t)(xx/0.8);
      int32_t bc = (int32_t)(yy/1.0);
      bli_setsc( (double)xx, (double)ac, &alpha1 );
      bli_setsc( (double)yy, (double)bc, &alpha2 );
    }
    libblis_test_vobj_irandomize( params, &x );
    libblis_test_vobj_irandomize( params, &y );
    libblis_test_vobj_irandomize( params, &z );
  }

  bli_copyv( &z, &z_save );

  // Apply the parameters.
  bli_obj_set_conj( conjx, &x );
  bli_obj_set_conj( conjy, &y );

  libblis_test_axpy2v_impl( iface, &alpha1, &alpha2, &x, &y, &z, cntx);

#ifndef __GTEST_VALGRIND_TEST__
  if(params->bitrp) {
    obj_t r;

    libblis_test_vobj_create( params, datatype, sc_str[2], m, &r );

    resid = libblis_test_bitrp_axpy2v( params, iface, &alpha1,
                  &alpha2, &x, &y, &z, cntx, &z_save, &r, datatype);

    bli_obj_free( &r );
  }
  else {
    resid = libblis_ref_axpy2v( params, &alpha1, &alpha2, &x, &y, &z, &z_save );
  }
#endif

  // Zero out performance and residual if output vector is empty.
  libblis_test_check_empty_problem( &z, &resid );

  // Free the test objects.
  libblis_test_obj_free( &x );
  libblis_test_obj_free( &y );
  libblis_test_obj_free( &z );
  libblis_test_obj_free( &z_save );

  return abs(resid);
}

void libblis_test_axpy2v_impl (
  iface_t   iface,
  obj_t*    alpha1,
  obj_t*    alpha2,
  obj_t*    x,
  obj_t*    y,
  obj_t*    z,
  cntx_t*   cntx
){
  switch ( iface )
  {
    case BLIS_TEST_SEQ_FRONT_END:
      bli_axpy2v_ex( alpha1, alpha2, x, y, z, cntx, NULL );
      break;

    default:
      libblis_test_printf_error( "Invalid interface type.\n" );
  }
}

double libblis_test_axpy2v_check (
  test_params_t* params,
  obj_t*         alpha1,
  obj_t*         alpha2,
  obj_t*         x,
  obj_t*         y,
  obj_t*         z,
  obj_t*         z_orig
) {
  num_t  dt      = bli_obj_dt( z );
  num_t  dt_real = bli_obj_dt_proj_to_real( z );

  dim_t  m       = bli_obj_vector_dim( z );

  obj_t  x_temp, y_temp, z_temp;
  obj_t  norm;

  double junk;
  double resid = 0.0;
  //
  // Pre-conditions:
  // - x is randomized.
  // - y is randomized.
  // - z_orig is randomized.
  // Note:
  // - alpha1, alpha2 should have a non-zero imaginary component in the
  //   complex cases in order to more fully exercise the implementation.
  //
  // Under these conditions, we assume that the implementation for
  //
  //   z := z_orig + alpha1 * conjx(x) + alpha2 * conjy(y)
  //
  // is functioning correctly if
  //
  //   normfv( z - v )
  //
  // is negligible, where v contains z as computed by two calls to axpyv.
  //

  bli_obj_scalar_init_detached( dt_real, &norm );

  bli_obj_create( dt, m, 1, 0, 0, &x_temp );
  bli_obj_create( dt, m, 1, 0, 0, &y_temp );
  bli_obj_create( dt, m, 1, 0, 0, &z_temp );

  bli_copyv( x,      &x_temp );
  bli_copyv( y,      &y_temp );
  bli_copyv( z_orig, &z_temp );

  bli_scalv( alpha1, &x_temp );
  bli_scalv( alpha2, &y_temp );
  bli_addv( &x_temp, &z_temp );
  bli_addv( &y_temp, &z_temp );

  bli_subv( &z_temp, z );
  bli_normfv( z, &norm );
  bli_getsc( &norm, &resid, &junk );

  bli_obj_free( &x_temp );
  bli_obj_free( &y_temp );
  bli_obj_free( &z_temp );

  return resid;
}