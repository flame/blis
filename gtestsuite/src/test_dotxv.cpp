#include "blis_test.h"
#include "blis_utils.h"
#include "test_dotxv.h"

// Local prototypes.
void libblis_test_dotxv_deps (
  thread_data_t* tdata,
  test_params_t* params,
  test_op_t*     op
);

void libblis_test_dotxv_impl (
  iface_t   iface,
  obj_t*    alpha,
  obj_t*    x,
  obj_t*    y,
  obj_t*    beta,
  obj_t*    rho
);

double libblis_test_dotxv_check (
  test_params_t* params,
  obj_t*         alpha,
  obj_t*         x,
  obj_t*         y,
  obj_t*         beta,
  obj_t*         rho,
  obj_t*         rho_orig
);

double libblis_ref_dotxv(
  test_params_t* params,
  obj_t*         alpha,
  obj_t*         x,
  obj_t*         y,
  obj_t*         beta,
  obj_t*         rho,
  obj_t*         rho_orig
){
  double resid = 0.0;

  if((params->bitextf == 0) && (params->oruflw == BLIS_DEFAULT)) {
    resid = libblis_test_dotxv_check( params, alpha, x, y, beta, rho,
                                                               rho_orig );
  }
  else {
   if(params->oruflw == BLIS_DEFAULT) {
      resid = libblis_test_idotxv_check( params, alpha, x, y, beta, rho, rho_orig );
    }
    else {
      resid = libblis_test_vector_check(params, rho);
    }
  }
  return resid;
}

double libblis_test_bitrp_dotxv(
  test_params_t* params,
  iface_t        iface,
  obj_t*         alpha,
  obj_t*         x,
  obj_t*         y,
  obj_t*         beta,
  obj_t*         rho,
  obj_t*         rho_orig,
  obj_t*         r,
  num_t          dt
) {
  double resid = 0.0;
	 unsigned int n_repeats = params->n_repeats;
	 unsigned int i;

  for(i = 0; i < n_repeats; i++) {
    bli_copysc( rho_orig, r );
    libblis_test_dotxv_impl( iface, alpha, x, y, beta, r );
    resid = libblis_test_bitrp_vector(rho, r, dt);
  }
  return resid;
}

double libblis_test_op_dotxv (
  test_params_t* params,
  iface_t        iface,
  char*          dc_str,
  char*          pc_str,
  char*          sc_str,
  tensor_t*      dim,
  atom_t         alpv,
  atom_t         betv
){
  num_t        datatype;
  dim_t        m;
  conj_t       conjx, conjy, conjconjxy;
  obj_t        alpha, x, y, beta, rho, rho_save;
  double       resid = 0.0;

  // Use the datatype of the first char in the datatype combination string.
  bli_param_map_char_to_blis_dt( dc_str[0], &datatype );

  // Map the dimension specifier to an actual dimension.
  m = dim->m;

  // Map parameter characters to BLIS constants.
  bli_param_map_char_to_blis_conj( pc_str[0], &conjx );
  bli_param_map_char_to_blis_conj( pc_str[1], &conjy );

  // Create test scalars.
  bli_obj_scalar_init_detached( datatype, &alpha );
  bli_obj_scalar_init_detached( datatype, &beta );
  bli_obj_scalar_init_detached( datatype, &rho );
  bli_obj_scalar_init_detached( datatype, &rho_save );

  // Create test operands (vectors and/or matrices).
  libblis_test_vobj_create( params, datatype, sc_str[0], m, &x );
  libblis_test_vobj_create( params, datatype, sc_str[1], m, &y );

  // Initialize alpha, beta, and rho.
  bli_copysc( &BLIS_ONE, &alpha );
  bli_copysc( &BLIS_ZERO, &beta );
  bli_copysc( &BLIS_MINUS_ONE, &rho );
  bli_copysc( &rho, &rho_save );

  // Randomize x.
  if((params->bitextf == 0) && (params->oruflw == BLIS_DEFAULT)) {
    libblis_test_vobj_randomize( params, TRUE, &x );
  } else {
    libblis_test_vobj_irandomize( params, &x );
  }

  // Determine whether to make a copy of x with or without conjugation.
  //
  //  conjx conjy  ~conjx^conjy   y is initialized as
  //  n     n      c              y = conj(x)
  //  n     c      n              y = x
  //  c     n      n              y = x
  //  c     c      c              y = conj(x)
  //
  conjconjxy = bli_apply_conj( conjx, conjy );
  conjconjxy = bli_conj_toggled( conjconjxy );
  bli_obj_set_conj( conjconjxy, &x );
  bli_copyv( &x, &y );

  // Apply the parameters.
  bli_obj_set_conj( conjx, &x );
  bli_obj_set_conj( conjy, &y );

  libblis_test_dotxv_impl( iface, &alpha, &x, &y, &beta, &rho );

#ifndef __GTEST_VALGRIND_TEST__
  if(params->bitrp) {
    obj_t r;

    bli_copysc( &rho, &r );

    resid = libblis_test_bitrp_dotxv( params, iface, &alpha,
                           &x, &y, &beta, &rho, &rho_save, &r, datatype);

    bli_obj_free( &r );
  }
  else {
    resid = libblis_ref_dotxv( params, &alpha, &x, &y, &beta, &rho, &rho_save );
  }
#endif

  // Zero out performance and residual if output scalar is empty.
  libblis_test_check_empty_problem( &rho, &resid );

  // Free the test objects.
  libblis_test_obj_free( &x );
  libblis_test_obj_free( &y );

  return abs(resid);
}

void libblis_test_dotxv_impl (
  iface_t   iface,
  obj_t*    alpha,
  obj_t*    x,
  obj_t*    y,
  obj_t*    beta,
  obj_t*    rho
) {
  switch ( iface )
  {
    case BLIS_TEST_SEQ_FRONT_END:
      bli_dotxv( alpha, x, y, beta, rho );
      break;

    default:
      libblis_test_printf_error( "Invalid interface type.\n" );
  }
}



double libblis_test_dotxv_check (
  test_params_t* params,
  obj_t*         alpha,
  obj_t*         x,
  obj_t*         y,
  obj_t*         beta,
  obj_t*         rho,
  obj_t*         rho_orig
) {
  num_t  dt_real = bli_obj_dt_proj_to_real( y );

  obj_t  rho_r, rho_i;
  obj_t  norm_x_r, norm_xy_r;
  obj_t  temp_r;

  double zero;
  double junk;
  double resid = 0.0;
  //
  // Pre-conditions:
  // - x is randomized.
  // - y is equal to conjx(conjy(x)).
  // - alpha must be real-valued.
  // - beta must be zero.
  // Note:
  // - We forgo fully exercising beta scaling in order to simplify the
  //   test.
  //
  // Under these conditions, we assume that the implementation for
  //
  //   rho := beta * rho_orig + alpha * conjx(x^T) conjy(y)
  //
  // is functioning correctly if
  //
  //   sqrtsc( rho.real ) - sqrtsc( alpha ) * normfv( x )
  //
  // and
  //
  //   rho.imag
  //
  // are negligible.
  //

  bli_obj_scalar_init_detached( dt_real, &rho_r );
  bli_obj_scalar_init_detached( dt_real, &rho_i );
  bli_obj_scalar_init_detached( dt_real, &norm_x_r );
  bli_obj_scalar_init_detached( dt_real, &norm_xy_r );
  bli_obj_scalar_init_detached( dt_real, &temp_r );

  bli_copysc( alpha, &temp_r );
  bli_sqrtsc( &temp_r, &temp_r );

  bli_normfv( x, &norm_x_r );
  bli_mulsc( &temp_r, &norm_x_r );

  bli_unzipsc( rho, &rho_r, &rho_i );

  bli_sqrtsc( &rho_r, &norm_xy_r );

  bli_subsc( &norm_x_r, &norm_xy_r );
  bli_getsc( &norm_xy_r, &resid, &junk );
  bli_getsc( &rho_i,     &zero, &junk );

  resid = bli_fmaxabs( resid, zero );

  return resid;
}