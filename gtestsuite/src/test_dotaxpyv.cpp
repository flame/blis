#include "blis_test.h"
#include "blis_utils.h"
#include "test_dotaxpyv.h"

// Local prototypes.
void libblis_test_dotaxpyv_deps (
  thread_data_t* tdata,
  test_params_t* params,
  test_op_t*     op
);

void libblis_test_dotaxpyv_impl (
  iface_t   iface,
  obj_t*    alpha,
  obj_t*    xt,
  obj_t*    x,
  obj_t*    y,
  obj_t*    rho,
  obj_t*    z,
  cntx_t*   cntx
);

double libblis_test_dotaxpyv_check (
  test_params_t* params,
  obj_t*         alpha,
  obj_t*         xt,
  obj_t*         x,
  obj_t*         y,
  obj_t*         rho,
  obj_t*         z,
  obj_t*         z_orig
);

double libblis_ref_dotaxpyv (
  test_params_t* params,
  obj_t*         alpha,
  obj_t*         xt,
  obj_t*         x,
  obj_t*         y,
  obj_t*         rho,
  obj_t*         z,
  obj_t*         z_orig
){
  double resid = 0.0;

  if((params->bitextf == 0) && (params->oruflw == BLIS_DEFAULT)) {
    resid = libblis_test_dotaxpyv_check( params, alpha, xt, x, y, rho, z, z_orig );
  }
  else {
    if(params->oruflw == BLIS_DEFAULT) {
      resid = libblis_test_idotaxpyv_check( params, alpha, xt, x, y, rho, z, z_orig );
    }
    else {
      resid = libblis_test_vector_check(params, z);
    }
  }
  return resid;
}

double libblis_test_bitrp_dotxaxpyf(
  test_params_t* params,
  iface_t        iface,
  obj_t*         alpha,
  obj_t*         xt,
  obj_t*         x,
  obj_t*         y,
  obj_t*         rho,
  obj_t*         z,
  obj_t*         z_orig,
  cntx_t*        cntx,
  obj_t*         r,
  num_t          dt
) {
  double resid = 0.0;
	 unsigned int n_repeats = params->n_repeats;
	 unsigned int i;
  obj_t rh;

	 for(i = 0; i < n_repeats; i++) {
		  bli_copysc( &BLIS_MINUS_ONE, &rh );
		  bli_copyv( z_orig, r );
    libblis_test_dotaxpyv_impl( iface, alpha, xt, x, y, rho, r, cntx );
    resid = libblis_test_bitrp_vector(&rh, rho, dt);
    resid += libblis_test_bitrp_vector(z, r, dt);
  }

  return resid;
}

double libblis_test_op_dotaxpyv (
  test_params_t* params,
  iface_t        iface,
  char*          dc_str,
  char*          pc_str,
  char*          sc_str,
  tensor_t*      dim,
  atom_t         alpv
) {
  num_t        datatype;
  dim_t        m;
  conj_t       conjxt, conjx, conjy;
  conj_t       conjconjxty;
  obj_t        alpha, xt, x, y, rho, z;
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
  bli_param_map_char_to_blis_conj( pc_str[0], &conjxt );
  bli_param_map_char_to_blis_conj( pc_str[1], &conjx );
  bli_param_map_char_to_blis_conj( pc_str[2], &conjy );

  // Create test scalars.
  bli_obj_scalar_init_detached( datatype, &alpha );
  bli_obj_scalar_init_detached( datatype, &rho );

  // Create test operands (vectors and/or matrices).
  libblis_test_vobj_create( params, datatype, sc_str[0], m, &x );
  libblis_test_vobj_create( params, datatype, sc_str[1], m, &y );
  libblis_test_vobj_create( params, datatype, sc_str[2], m, &z );
  libblis_test_vobj_create( params, datatype, sc_str[2], m, &z_save );

  // Randomize x and z, and save z.
  if((params->bitextf == 0) && (params->oruflw == BLIS_DEFAULT)) {
    // Set alpha.
    if ( bli_obj_is_real( &z ) )	{
      bli_setsc(  alpv.real,  0.0, &alpha );
    }
    else	{
      bli_setsc(  alpv.real,  (alpv.real/0.8), &alpha );
    }
    libblis_test_vobj_randomize( params, TRUE, &x );
    libblis_test_vobj_randomize( params, TRUE, &z );
  } else {
    int32_t xx = (int32_t)alpv.real;
    if ( bli_obj_is_real( &z ) )	{
      bli_setsc( (double)xx,  0.0, &alpha );
    }
    else	{
      int32_t ac = (int32_t)(xx/0.8);
      bli_setsc( (double)xx, (double)ac, &alpha );
    }
    libblis_test_vobj_irandomize( params, &x );
    libblis_test_vobj_irandomize( params, &z );
  }

  bli_copyv( &z, &z_save );

  // Create an alias to x for xt. (Note that it doesn't actually need to be
  // transposed.)
  bli_obj_alias_to( &x, &xt );

  // Determine whether to make a copy of x with or without conjugation.
  //
  //  conjx conjy  ~conjx^conjy   y is initialized as
  //  n     n      c              y = conj(x)
  //  n     c      n              y = x
  //  c     n      n              y = x
  //  c     c      c              y = conj(x)
  //
  conjconjxty = bli_apply_conj( conjxt, conjy );
  conjconjxty = bli_conj_toggled( conjconjxty );
  bli_obj_set_conj( conjconjxty, &xt );
  bli_copyv( &xt, &y );

  // Apply the parameters.
  bli_obj_set_conj( conjxt, &xt );
  bli_obj_set_conj( conjx,  &x );
  bli_obj_set_conj( conjy,  &y );

  libblis_test_dotaxpyv_impl( iface, &alpha, &xt, &x, &y, &rho, &z, cntx );

#ifndef __GTEST_VALGRIND_TEST__
  if(params->bitrp) {
    obj_t r;

    libblis_test_vobj_create( params, datatype, sc_str[2], m, &r );

    resid = libblis_test_bitrp_dotxaxpyf(params, iface, &alpha,
                    &xt, &x, &y, &rho, &z, &z_save, cntx, &r, datatype);

    bli_obj_free( &r );
  }
  else {
    resid = libblis_ref_dotaxpyv( params, &alpha, &xt,
                                          &x, &y, &rho, &z, &z_save );
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

void libblis_test_dotaxpyv_impl (
  iface_t   iface,
  obj_t*    alpha,
  obj_t*    xt,
  obj_t*    x,
  obj_t*    y,
  obj_t*    rho,
  obj_t*    z,
  cntx_t*   cntx
){
  switch ( iface )
  {
    case BLIS_TEST_SEQ_FRONT_END:
      bli_dotaxpyv_ex( alpha, xt, x, y, rho, z, cntx, NULL );
      break;

    default:
      libblis_test_printf_error( "Invalid interface type.\n" );
  }
}

double libblis_test_dotaxpyv_check (
  test_params_t* params,
  obj_t*         alpha,
  obj_t*         xt,
  obj_t*         x,
  obj_t*         y,
  obj_t*         rho,
  obj_t*         z,
  obj_t*         z_orig
) {
  num_t  dt      = bli_obj_dt( z );
  num_t  dt_real = bli_obj_dt_proj_to_real( z );

  dim_t  m       = bli_obj_vector_dim( z );

  obj_t  rho_temp;

  obj_t  z_temp;
  obj_t  norm_z;

  double resid1, resid2;
  double junk;
  double resid = 0.0;

  //
  // Pre-conditions:
  // - x is randomized.
  // - y is randomized.
  // - z_orig is randomized.
  // - xt is an alias to x.
  // Note:
  // - alpha should have a non-zero imaginary component in the complex
  //   cases in order to more fully exercise the implementation.
  //
  // Under these conditions, we assume that the implementation for
  //
  //   rho := conjxt(x^T) conjy(y)
  //   z := z_orig + alpha * conjx(x)
  //
  // is functioning correctly if
  //
  //   ( rho - rho_temp )
  //
  // and
  //
  //   normfv( z - z_temp )
  //
  // are negligible, where rho_temp and z_temp contain rho and z as
  // computed by dotv and axpyv, respectively.
  //

  bli_obj_scalar_init_detached( dt,      &rho_temp );
  bli_obj_scalar_init_detached( dt_real, &norm_z );

  bli_obj_create( dt, m, 1, 0, 0, &z_temp );
  bli_copyv( z_orig, &z_temp );


  bli_dotv( xt, y, &rho_temp );
  bli_axpyv( alpha, x, &z_temp );


  bli_subsc( rho, &rho_temp );
  bli_getsc( &rho_temp, &resid1, &junk );

  bli_subv( &z_temp, z );
  bli_normfv( z, &norm_z );
  bli_getsc( &norm_z, &resid2, &junk );

  resid = bli_fmaxabs( resid1, resid2 );

  bli_obj_free( &z_temp );

  return resid;
}