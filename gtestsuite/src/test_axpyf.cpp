#include "blis_test.h"
#include "blis_utils.h"
#include "test_axpyf.h"

// Local prototypes.
void libblis_test_axpyf_deps (
  thread_data_t* tdata,
  test_params_t* params,
  test_op_t*     op
);

void libblis_test_axpyf_impl (
  iface_t   iface,
  obj_t*    alpha,
  obj_t*    a,
  obj_t*    x,
  obj_t*    y,
  cntx_t*   cntx
);

double libblis_test_axpyf_check (
  test_params_t* params,
  obj_t*         alpha,
  obj_t*         a,
  obj_t*         x,
  obj_t*         y,
  obj_t*         y_orig
);

double libblis_ref_axpyf (
  test_params_t* params,
  obj_t*         alpha,
  obj_t*         a,
  obj_t*         x,
  obj_t*         y,
  obj_t*         y_orig
){
  double resid = 0.0;

  if((params->bitextf == 0) && (params->oruflw == BLIS_DEFAULT)) {
    resid = libblis_test_axpyf_check( params, alpha, a, x, y, y_orig );
  }
  else {
    if(params->oruflw == BLIS_DEFAULT) {
      resid = libblis_test_iaxpyf_check( params, alpha, a, x, y, y_orig );
    }
    else {
      resid = libblis_test_vector_check(params, y);
    }
  }
  return resid;
}

double libblis_test_bitrp_axpyf(
  test_params_t* params,
  iface_t        iface,
  obj_t*         alpha,
  obj_t*         a,
  obj_t*         x,
  obj_t*         y,
  cntx_t*        cntx,
  obj_t*         y_orig,
  obj_t*         r,
  num_t          dt
) {
  double resid = 0.0;
	 unsigned int n_repeats = params->n_repeats;
	 unsigned int i;

  for(i = 0; i < n_repeats; i++) {
    bli_copyv( y_orig, r );
    libblis_test_axpyf_impl( iface, alpha, a, x, r, cntx );
    resid = libblis_test_bitrp_vector(y, r, dt);
  }
  return resid;
}

double libblis_test_op_axpyf (
  test_params_t* params,
  test_op_t*     op,
  iface_t        iface,
  char*          dc_str,
  char*          pc_str,
  char*          sc_str,
  tensor_t*      dim,
  atom_t         alpv
) {
  num_t        datatype;
  dim_t        m, b_n;
  conj_t       conja, conjx;
  obj_t        alpha, a, x, y;
  obj_t        y_save;
  cntx_t*      cntx;
  double       resid = 0.0;

  // Query a context.
  cntx = bli_gks_query_cntx();

  // Use the datatype of the first char in the datatype combination string.
  bli_param_map_char_to_blis_dt( dc_str[0], &datatype );

  // Map the dimension specifier to an actual dimension.
  m = dim->m;

  // Query the operation's fusing factor for the current datatype.
  b_n = bli_cntx_get_blksz_def_dt( datatype, BLIS_AF, cntx );

  // Store the fusing factor so that the driver can retrieve the value
  // later when printing results.
  op->dim_aux[0] = b_n;

  // Map parameter characters to BLIS constants.
  bli_param_map_char_to_blis_conj( pc_str[0], &conja );
  bli_param_map_char_to_blis_conj( pc_str[1], &conjx );

  // Create test scalars.
  bli_obj_scalar_init_detached( datatype, &alpha );

  // Create test operands (vectors and/or matrices).
  libblis_test_mobj_create( params, datatype, BLIS_NO_TRANSPOSE,
                                              sc_str[0], m, b_n, &a );
  libblis_test_vobj_create( params, datatype, sc_str[1], b_n, &x );
  libblis_test_vobj_create( params, datatype, sc_str[2], m, &y );
  libblis_test_vobj_create( params, datatype, sc_str[2], m, &y_save );

  // Set alpha.
  if ( bli_obj_is_real( &y ) )   {
    bli_setsc( -1.0,  0.0, &alpha );
  }
  else   {
    bli_setsc(  0.0, -1.0, &alpha );
  }

  // Randomize A, x, and y, and save y.
  if((params->bitextf == 0) && (params->oruflw == BLIS_DEFAULT)) {
    libblis_test_mobj_randomize( params, FALSE, &a );
    libblis_test_vobj_randomize( params, FALSE, &x );
    libblis_test_vobj_randomize( params, FALSE, &y );
  } else {
    libblis_test_mobj_irandomize( params, &a );
    libblis_test_vobj_irandomize( params, &x );
    libblis_test_vobj_irandomize( params, &y );
  }

  bli_copyv( &y, &y_save );

  // Apply the parameters.
  bli_obj_set_conj( conja, &a );
  bli_obj_set_conj( conjx, &x );

  libblis_test_axpyf_impl( iface, &alpha, &a, &x, &y, cntx );

#ifndef __GTEST_VALGRIND_TEST__
  if(params->bitrp) {
    obj_t r;

    libblis_test_vobj_create( params, datatype, sc_str[2], m, &r );

    resid = libblis_test_bitrp_axpyf( params, iface, &alpha, &a,
                                       &x, &y, cntx, &y_save, &r, datatype);

    bli_obj_free( &r );
  }
  else {
    resid = libblis_ref_axpyf( params, &alpha, &a, &x, &y, &y_save );
  }
#endif

  // Zero out performance and residual if output vector is empty.
  libblis_test_check_empty_problem( &y, &resid );

  // Free the test objects.
  libblis_test_obj_free( &a );
  libblis_test_obj_free( &x );
  libblis_test_obj_free( &y );
  libblis_test_obj_free( &y_save );

  return abs(resid);
}

void libblis_test_axpyf_impl (
  iface_t   iface,
  obj_t*    alpha,
  obj_t*    a,
  obj_t*    x,
  obj_t*    y,
  cntx_t*   cntx
) {
  switch ( iface )
  {
    case BLIS_TEST_SEQ_FRONT_END:
      bli_axpyf_ex( alpha, a, x, y, cntx, NULL );
      break;

    default:
      libblis_test_printf_error( "Invalid interface type.\n" );
  }
}

double libblis_test_axpyf_check (
  test_params_t* params,
  obj_t*         alpha,
  obj_t*         a,
  obj_t*         x,
  obj_t*         y,
  obj_t*         y_orig
) {
  num_t  dt      = bli_obj_dt( y );
  num_t  dt_real = bli_obj_dt_proj_to_real( y );

  dim_t  m       = bli_obj_vector_dim( y );
  dim_t  b_n     = bli_obj_width( a );

  dim_t  i;

  obj_t  a1, chi1, v;
  obj_t  alpha_chi1;
  obj_t  norm;

  double junk;
  double resid = 0.0;

  //
  // Pre-conditions:
  // - a is randomized.
  // - x is randomized.
  // - y is randomized.
  // Note:
  // - alpha should have a non-zero imaginary component in the complex
  //   cases in order to more fully exercise the implementation.
  //
  // Under these conditions, we assume that the implementation for
  //
  //   y := y_orig + alpha * conja(A) * conjx(x)
  //
  // is functioning correctly if
  //
  //   normfv( y - v )
  //
  // is negligible, where v contains y as computed by repeated calls to
  // axpyv.
  //

  bli_obj_scalar_init_detached( dt_real, &norm );
  bli_obj_scalar_init_detached( dt,      &alpha_chi1 );

  bli_obj_create( dt, m,   1, 0, 0, &v );

  bli_copyv( y_orig, &v );

  for ( i = 0; i < b_n; ++i ) {
    bli_acquire_mpart_l2r( BLIS_SUBPART1, i, 1, a, &a1 );
    bli_acquire_vpart_f2b( BLIS_SUBPART1, i, 1, x, &chi1 );

    bli_copysc( &chi1, &alpha_chi1 );
    bli_mulsc( alpha, &alpha_chi1 );

    bli_axpyv( &alpha_chi1, &a1, &v );
  }

  bli_subv( y, &v );
  bli_normfv( &v, &norm );
  bli_getsc( &norm, &resid, &junk );

  bli_obj_free( &v );

  return resid;
}