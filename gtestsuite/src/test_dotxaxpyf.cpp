#include "blis_test.h"
#include "blis_utils.h"
#include "test_dotxaxpyf.h"

// Local prototypes.
void libblis_test_dotxaxpyf_deps (
  thread_data_t* tdata,
  test_params_t* params,
  test_op_t*     op
);

void libblis_test_dotxaxpyf_impl (
  iface_t   iface,
  obj_t*    alpha,
  obj_t*    at,
  obj_t*    a,
  obj_t*    w,
  obj_t*    x,
  obj_t*    beta,
  obj_t*    y,
  obj_t*    z,
  cntx_t*   cntx
);

double libblis_test_dotxaxpyf_check (
  test_params_t* params,
  obj_t*         alpha,
  obj_t*         at,
  obj_t*         a,
  obj_t*         w,
  obj_t*         x,
  obj_t*         beta,
  obj_t*         y,
  obj_t*         z,
  obj_t*         y_orig,
  obj_t*         z_orig
);

double libblis_ref_dotxaxpyf (
  test_params_t* params,
  obj_t*         alpha,
  obj_t*         at,
  obj_t*         a,
  obj_t*         w,
  obj_t*         x,
  obj_t*         beta,
  obj_t*         y,
  obj_t*         z,
  obj_t*         y_orig,
  obj_t*         z_orig
){
  double resid = 0.0;

  if((params->bitextf == 0) && (params->oruflw == BLIS_DEFAULT)) {
    resid = libblis_test_dotxaxpyf_check( params, alpha, at, a, w, x,
                                              beta, y, z, y_orig, z_orig);
  }
  else {
    if(params->oruflw == BLIS_DEFAULT) {
      resid = libblis_test_idotxaxpyf_check( params, alpha, at, a, w, x,
                                              beta, y, z, y_orig, z_orig);
    }
    else {
      resid = libblis_test_vector_check(params, y);
      resid = libblis_test_vector_check(params, z);
    }
  }
  return resid;
}

double libblis_test_bitrp_dotxaxpyf(
  test_params_t* params,
  iface_t        iface,
  obj_t*         alpha,
  obj_t*         at,
  obj_t*         a,
  obj_t*         w,
  obj_t*         x,
  obj_t*         beta,
  obj_t*         y,
  obj_t*         z,
  cntx_t*        cntx,
  obj_t*         y_orig,
  obj_t*         z_orig,
  obj_t*         r,
  obj_t*         v,
  num_t          dt
) {
  double resid = 0.0;
	 unsigned int n_repeats = params->n_repeats;
	 unsigned int i;

  for(i = 0; i < n_repeats; i++) {
    bli_copyv( y_orig, r );
    bli_copyv( z_orig, v );
    libblis_test_dotxaxpyf_impl( iface, alpha, at, a, w, x,
                                                   beta, r, v, cntx );
    resid = libblis_test_bitrp_vector(y, r, dt);
    resid += libblis_test_bitrp_vector(z, v, dt);
  }
  return resid;
}

double libblis_test_op_dotxaxpyf (
  test_params_t* params,
  test_op_t*     op,
  iface_t        iface,
  char*          dc_str,
  char*          pc_str,
  char*          sc_str,
  tensor_t*      dim,
  atom_t         alpv,
  atom_t         betv
) {
  num_t        datatype;
  dim_t        m, b_n;
  conj_t       conjat, conja, conjw, conjx;
  obj_t        alpha, at, a, w, x, beta, y, z;
  obj_t        y_save, z_save;
  cntx_t*      cntx;
  double       resid = 0.0;

  // Query a context.
  cntx = bli_gks_query_cntx();

  // Use the datatype of the first char in the datatype combination string.
  bli_param_map_char_to_blis_dt( dc_str[0], &datatype );

  // Map the dimension specifier to an actual dimension.
  m = dim->m;

  // Query the operation's fusing factor for the current datatype.
  b_n = bli_cntx_get_blksz_def_dt( datatype, BLIS_XF, cntx );

  // Store the fusing factor so that the driver can retrieve the value
  // later when printing results.
  op->dim_aux[0] = b_n;

  // Map parameter characters to BLIS constants.
  bli_param_map_char_to_blis_conj( pc_str[0], &conjat );
  bli_param_map_char_to_blis_conj( pc_str[1], &conja );
  bli_param_map_char_to_blis_conj( pc_str[2], &conjw );
  bli_param_map_char_to_blis_conj( pc_str[3], &conjx );

  // Create test scalars.
  bli_obj_scalar_init_detached( datatype, &alpha );
  bli_obj_scalar_init_detached( datatype, &beta );

  // Create test operands (vectors and/or matrices).
  libblis_test_mobj_create( params, datatype, BLIS_NO_TRANSPOSE,
                                              sc_str[0], m, b_n, &a );
  libblis_test_vobj_create( params, datatype, sc_str[1], m, &w );
  libblis_test_vobj_create( params, datatype, sc_str[2], b_n, &x );
  libblis_test_vobj_create( params, datatype, sc_str[3], b_n, &y );
  libblis_test_vobj_create( params, datatype, sc_str[3], b_n, &y_save );
  libblis_test_vobj_create( params, datatype, sc_str[4], m, &z );
  libblis_test_vobj_create( params, datatype, sc_str[4], m, &z_save );

  // Randomize A, w, x, y, and z, and save y and z.
  if((params->bitextf == 0) && (params->oruflw == BLIS_DEFAULT)) {
    // Set alpha.
    if ( bli_obj_is_real( &y ) )	{
      bli_setsc(  alpv.real,  0.0, &alpha );
      bli_setsc(  betv.real,  0.0, &beta );
    }
    else	{
      bli_setsc(  alpv.real,  (alpv.real/0.8), &alpha );
      bli_setsc(  betv.real,  (betv.real/1.2), &beta );
    }
    libblis_test_mobj_randomize( params, FALSE, &a );
    libblis_test_vobj_randomize( params, FALSE, &w );
    libblis_test_vobj_randomize( params, FALSE, &x );
    libblis_test_vobj_randomize( params, FALSE, &y );
    libblis_test_vobj_randomize( params, FALSE, &z );
  } else {
    int32_t xx = (int32_t)alpv.real;
    int32_t yy = (int32_t)betv.real;
    if ( bli_obj_is_real( &z ) )	{
      bli_setsc( (double)xx,  0.0, &alpha );
      bli_setsc( (double)yy,  0.0, &beta );
    }
    else	{
      int32_t ac = (int32_t)(xx/0.8);
      int32_t bc = (int32_t)(yy/1.0);
      bli_setsc( (double)xx, (double)ac, &alpha );
      bli_setsc( (double)yy, (double)bc, &beta );
    }
    libblis_test_mobj_irandomize( params, &a );
    libblis_test_vobj_irandomize( params, &w );
    libblis_test_vobj_irandomize( params, &x );
    libblis_test_vobj_irandomize( params, &y );
    libblis_test_vobj_irandomize( params, &z );
  }

  bli_copyv( &y, &y_save );
  bli_copyv( &z, &z_save );

  // Create an alias to a for at. (Note that it should NOT actually be
  // marked for transposition since the transposition is part of the dotxf
  // subproblem.)
  bli_obj_alias_to( &a, &at );

  // Apply the parameters.
  bli_obj_set_conj( conjat, &at );
  bli_obj_set_conj( conja, &a );
  bli_obj_set_conj( conjw, &w );
  bli_obj_set_conj( conjx, &x );

  libblis_test_dotxaxpyf_impl( iface, &alpha, &at, &a, &w, &x,
                                                   &beta, &y, &z, cntx );
#ifndef __GTEST_VALGRIND_TEST__
  if(params->bitrp) {
    obj_t r,v;

    libblis_test_vobj_create( params, datatype, sc_str[3], b_n, &r );

    libblis_test_vobj_create( params, datatype, sc_str[4], m, &v );

    resid = libblis_test_bitrp_dotxaxpyf(params, iface, &alpha, &at, &a, &w,
                  &x, &beta, &y, &z, cntx, &y_save, &z_save, &r, &v, datatype);

    bli_obj_free( &r );
    bli_obj_free( &v );
  }
  else {
    resid = libblis_ref_dotxaxpyf( params, &alpha, &at, &a, &w, &x,
                                         &beta, &y, &z, &y_save, &z_save);
  }
#endif

  // Zero out performance and residual if either output vector is empty.
  libblis_test_check_empty_problem( &y, &resid );
  libblis_test_check_empty_problem( &z, &resid );

  // Free the test objects.
  libblis_test_obj_free( &a );
  libblis_test_obj_free( &w );
  libblis_test_obj_free( &x );
  libblis_test_obj_free( &y );
  libblis_test_obj_free( &z );
  libblis_test_obj_free( &y_save );
  libblis_test_obj_free( &z_save );

  return abs(resid);
}

void libblis_test_dotxaxpyf_impl (
  iface_t   iface,
  obj_t*    alpha,
  obj_t*    at,
  obj_t*    a,
  obj_t*    w,
  obj_t*    x,
  obj_t*    beta,
  obj_t*    y,
  obj_t*    z,
  cntx_t*   cntx
) {
  switch ( iface )
  {
    case BLIS_TEST_SEQ_FRONT_END:
      bli_dotxaxpyf_ex( alpha, at, a, w, x, beta, y, z, cntx, NULL );
      break;

    default:
      libblis_test_printf_error( "Invalid interface type.\n" );
  }
}

double libblis_test_dotxaxpyf_check (
  test_params_t* params,
  obj_t*         alpha,
  obj_t*         at,
  obj_t*         a,
  obj_t*         w,
  obj_t*         x,
  obj_t*         beta,
  obj_t*         y,
  obj_t*         z,
  obj_t*         y_orig,
  obj_t*         z_orig
) {
  num_t  dt      = bli_obj_dt( y );
  num_t  dt_real = bli_obj_dt_proj_to_real( y );

  dim_t  m       = bli_obj_vector_dim( z );
  dim_t  b_n     = bli_obj_vector_dim( y );

  dim_t  i;

  obj_t  a1, chi1, psi1, v, q;
  obj_t  alpha_chi1;
  obj_t  norm;

  double resid1, resid2;
  double junk;
  double resid = 0.0;
  //
  // Pre-conditions:
  // - a is randomized.
  // - w is randomized.
  // - x is randomized.
  // - y is randomized.
  // - z is randomized.
  // - at is an alias to a.
  // Note:
  // - alpha and beta should have a non-zero imaginary component in the
  //   complex cases in order to more fully exercise the implementation.
  //
  // Under these conditions, we assume that the implementation for
  //
  //   y := beta * y_orig + alpha * conjat(A^T) * conjw(w)
  //   z :=        z_orig + alpha * conja(A)    * conjx(x)
  //
  // is functioning correctly if
  //
  //   normfv( y - v )
  //
  // and
  //
  //   normfv( z - q )
  //
  // are negligible, where v and q contain y and z as computed by repeated
  // calls to dotxv and axpyv, respectively.
  //

  bli_obj_scalar_init_detached( dt_real, &norm );
  bli_obj_scalar_init_detached( dt,      &alpha_chi1 );

  bli_obj_create( dt, b_n, 1, 0, 0, &v );
  bli_obj_create( dt, m,   1, 0, 0, &q );

  bli_copyv( y_orig, &v );
  bli_copyv( z_orig, &q );

  // v := beta * v + alpha * conjat(at) * conjw(w)
  for ( i = 0; i < b_n; ++i ) {
    bli_acquire_mpart_l2r( BLIS_SUBPART1, i, 1, at, &a1 );
    bli_acquire_vpart_f2b( BLIS_SUBPART1, i, 1, &v, &psi1 );

    bli_dotxv( alpha, &a1, w, beta, &psi1 );
  }

  // q := q + alpha * conja(a) * conjx(x)
  for ( i = 0; i < b_n; ++i ) {
    bli_acquire_mpart_l2r( BLIS_SUBPART1, i, 1, a, &a1 );
    bli_acquire_vpart_f2b( BLIS_SUBPART1, i, 1, x, &chi1 );

    bli_copysc( &chi1, &alpha_chi1 );
    bli_mulsc( alpha, &alpha_chi1 );

    bli_axpyv( &alpha_chi1, &a1, &q );
  }


  bli_subv( y, &v );
  bli_normfv( &v, &norm );
  bli_getsc( &norm, &resid1, &junk );

  bli_subv( z, &q );
  bli_normfv( &q, &norm );
  bli_getsc( &norm, &resid2, &junk );


  resid = bli_fmaxabs( resid1, resid2 );

  bli_obj_free( &v );
  bli_obj_free( &q );

  return resid;
}