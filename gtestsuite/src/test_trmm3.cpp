#include "blis_test.h"
#include "blis_utils.h"
#include "test_trmm3.h"

using namespace std;

// Local prototypes.
void libblis_test_trmm3_deps (
  thread_data_t* tdata,
  test_params_t* params,
  test_op_t*     op
);

void libblis_test_trmm3_impl(
  iface_t   iface,
  side_t    side,
  obj_t*    alpha,
  obj_t*    a,
  obj_t*    b,
  obj_t*    beta,
  obj_t*    c
);

double libblis_test_trmm3_check(
  test_params_t* params,
  side_t         side,
  obj_t*         alpha,
  obj_t*         a,
  obj_t*         b,
  obj_t*         beta,
  obj_t*         c,
  obj_t*         c_orig
);

void libblis_api_trmm3(
  test_params_t* params,
  iface_t        iface,
  side_t         side,
  obj_t*         alpha,
  obj_t*         a,
  obj_t*         b,
  obj_t*         beta,
  obj_t*         c
)
{
  libblis_test_trmm3_impl( iface, side, alpha, a, b, beta, c );
  return ;
}

double libblis_ref_trmm3(
  test_params_t* params,
  side_t         side,
  obj_t*         alpha,
  obj_t*         a,
  obj_t*         b,
  obj_t*         beta,
  obj_t*         c,
  obj_t*         c_save,
  num_t          dt
) {

  double resid = 0.0;
  double *betap = (double *)bli_obj_buffer( beta );

  if ((params->nanf) && (*betap == 0)) {
    resid = libblis_check_nan_trmm3(c, dt );
  }
  else if((params->bitextf == 0) && (params->oruflw == BLIS_DEFAULT)) {
	   resid = libblis_test_trmm3_check( params, side, alpha, a, b, beta, c, c_save );
  }
  else {
    if(params->oruflw == BLIS_DEFAULT) {
	     resid = libblis_test_itrmm3_check( params, side, alpha, a, b, beta, c, c_save );
    }
    else {
      resid = libblis_test_matrix_check(params, c);
    }
  }
  return resid;
}

double libblis_test_bitrp_trmm3(
  test_params_t* params,
  iface_t        iface,
  side_t         side,
  obj_t*         alpha,
  obj_t*         a,
  obj_t*         b,
  obj_t*         beta,
  obj_t*         c,
  obj_t*         c_orig,
  obj_t*         r,
  num_t          dt
) {
  double resid = 0.0;
	 unsigned int n_repeats = params->n_repeats;
	 unsigned int i;

  for(i = 0; i < n_repeats; i++) {
    bli_copym( c_orig, r );
		  libblis_test_trmm3_impl( iface, side, alpha, a, b, beta, r );
    resid = libblis_test_bitrp_matrix(c, r, dt);
  }
  return resid;
}

double libblis_test_op_trmm3 (
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
  dim_t        m, n;
  dim_t        mn_side;
  side_t       side;
  uplo_t       uploa;
  trans_t      transa, transb;
  diag_t       diaga;
  obj_t        alpha, a, b, beta, c;
  obj_t        c_save;
  double       resid = 0.0;

  // Use the datatype of the first char in the datatype combination string.
  bli_param_map_char_to_blis_dt( dc_str[0], &datatype );

  // Map the dimension specifier to actual dimensions.
  m = dim->m;
  n = dim->n;

  // Map parameter characters to BLIS constants.
  bli_param_map_char_to_blis_side ( pc_str[0], &side );
  bli_param_map_char_to_blis_uplo ( pc_str[1], &uploa );
  bli_param_map_char_to_blis_trans( pc_str[2], &transa );
  bli_param_map_char_to_blis_diag ( pc_str[3], &diaga );
  bli_param_map_char_to_blis_trans( pc_str[4], &transb );

  // Create test scalars.
  bli_obj_scalar_init_detached( datatype, &alpha );
  bli_obj_scalar_init_detached( datatype, &beta );

  // Create test operands (vectors and/or matrices).
  bli_set_dim_with_side( side, m, n, &mn_side );
  libblis_test_mobj_create( params, datatype, transa,
                            sc_str[1], mn_side, mn_side, &a );
  libblis_test_mobj_create( params, datatype, transb,
                            sc_str[2], m,       n,       &b );
  libblis_test_mobj_create( params, datatype, BLIS_NO_TRANSPOSE,
                            sc_str[0], m,       n,       &c );
  libblis_test_mobj_create( params, datatype, BLIS_NO_TRANSPOSE,
                            sc_str[0], m,       n,       &c_save );

   // Set the structure and uplo properties of A.
   bli_obj_set_struc( BLIS_TRIANGULAR, &a );
   bli_obj_set_uplo( uploa, &a );

  // Set alpha and beta.
  if((params->bitextf == 0) && (params->oruflw == BLIS_DEFAULT)) {
    if ( bli_obj_is_real( &c ) )	{
      bli_setsc(  alpv.real,  0.0, &alpha );
      bli_setsc(  betv.real,  0.0, &beta );
    }
    else	{
      bli_setsc(  alpv.real,  (alpv.real/0.8), &alpha );
      bli_setsc(  betv.real,  (betv.real/1.2), &beta );
    }
    // Randomize A, B, and C, and save C.
    libblis_test_mobj_randomize( params, TRUE, &a );
    libblis_test_mobj_randomize( params, TRUE, &b );
    libblis_test_mobj_randomize( params, TRUE, &c );
  }
  else {
    int32_t x = alpv.real;
    int32_t y = betv.real;
    if ( bli_obj_is_real( &c ) )	{
      bli_setsc( (double)x,  0.0, &alpha );
      bli_setsc( (double)y,  0.0, &beta );
    }
    else	{
      int32_t ac = (int32_t)(x/0.8);
      int32_t bc = (int32_t)(y/1.0);
      bli_setsc( (double)x, (double)ac, &alpha );
      bli_setsc( (double)y, (double)bc, &beta );
    }
    libblis_test_mobj_irandomize( params, &a );
    libblis_test_mobj_irandomize( params, &b );
    libblis_test_mobj_irandomize( params, &c );
  }

  if ((params->nanf) && (betv.real == 0) ) {
    test_fillbuffmem(&c, datatype );
  }

  // Randomize A, make it densely triangular.
  bli_mktrim( &a );

  bli_obj_set_conjtrans( transa, &a );
  bli_obj_set_diag( diaga, &a );
  bli_obj_set_conjtrans( transb, &b );

  //Copy c to c_save
  bli_copym( &c, &c_save );

  libblis_api_trmm3(params, iface, side, &alpha, &a, &b, &beta, &c );

#ifndef __GTEST_VALGRIND_TEST__
  if(params->bitrp) {
    obj_t r;

    libblis_test_mobj_create( params, datatype, BLIS_NO_TRANSPOSE,
                                                sc_str[0], m, n, &r );
    resid = libblis_test_bitrp_trmm3( params, iface, side,&alpha, &a, &b,
                                            &beta, &c, &c_save, &r, datatype);
    bli_obj_free( &r );
  }
  else {
    resid = libblis_ref_trmm3(params, side, &alpha, &a, &b, &beta,
                                              &c, &c_save, datatype );
  }
#endif

  // Zero out performance and residual if output matrix is empty.
  libblis_test_check_empty_problem( &c, &resid );

  // Free the test objects.
  libblis_test_obj_free( &a );
  libblis_test_obj_free( &b );
  libblis_test_obj_free( &c );
  libblis_test_obj_free( &c_save );

  return abs(resid);
}

void libblis_test_trmm3_impl (
  iface_t   iface,
  side_t    side,
  obj_t*    alpha,
  obj_t*    a,
  obj_t*    b,
  obj_t*    beta,
  obj_t*    c
)
{
  switch( iface ) {
    case BLIS_TEST_SEQ_FRONT_END:
	  bli_trmm3( side, alpha, a, b, beta, c );
    break;

    default:
      libblis_test_printf_error( "Invalid interface type.\n" );
  }
}

double libblis_test_trmm3_check (
  test_params_t* params,
  side_t         side,
  obj_t*         alpha,
  obj_t*         a,
  obj_t*         b,
  obj_t*         beta,
  obj_t*         c,
  obj_t*         c_orig
)
{
  num_t  dt      = bli_obj_dt( c );
  num_t  dt_real = bli_obj_dt_proj_to_real( c );

  dim_t  m       = bli_obj_length( c );
  dim_t  n       = bli_obj_width( c );

  obj_t  norm;
  obj_t  t, v, w, z;

  double junk;
  double resid = 0.0;

  // Pre-conditions:
  // - a is randomized and triangular.
  // - b is randomized.
  // - c_orig is randomized.
  // Note:
  // - alpha and beta should have non-zero imaginary components in the
  //   complex cases in order to more fully exercise the implementation.
  //
  // Under these conditions, we assume that the implementation for
  //
  //   C := beta * C_orig + alpha * transa(A) * transb(B)    (side = left)
  //   C := beta * C_orig + alpha * transb(B) * transa(A)    (side = right)
  //
  // is functioning correctly if
  //
  //   normfv( v - z )
  //
  // is negligible, where
  //
  //   v = C * t
  //
  //   z = ( beta * C_orig + alpha * transa(A) * transb(B) ) * t     (side = left)
  //     = beta * C_orig * t + alpha * transa(A) * transb(B) * t
  //     = beta * C_orig * t + alpha * transa(A) * w
  //     = beta * C_orig * t + z
  //
  //   z = ( beta * C_orig + alpha * transb(B) * transa(A) ) * t     (side = right)
  //     = beta * C_orig * t + alpha * transb(B) * transa(A) * t
  //     = beta * C_orig * t + alpha * transb(B) * w
  //     = beta * C_orig * t + z

  bli_obj_scalar_init_detached( dt_real, &norm );

  if ( bli_is_left( side ) )
  {
    bli_obj_create( dt, n, 1, 0, 0, &t );
    bli_obj_create( dt, m, 1, 0, 0, &v );
    bli_obj_create( dt, m, 1, 0, 0, &w );
    bli_obj_create( dt, m, 1, 0, 0, &z );
  }
  else // else if ( bli_is_left( side ) )
  {
    bli_obj_create( dt, n, 1, 0, 0, &t );
    bli_obj_create( dt, m, 1, 0, 0, &v );
    bli_obj_create( dt, n, 1, 0, 0, &w );
    bli_obj_create( dt, m, 1, 0, 0, &z );
  }

  libblis_test_vobj_randomize( params, TRUE, &t );

  bli_gemv( &BLIS_ONE, c, &t, &BLIS_ZERO, &v );

  if ( bli_is_left( side ) )
  {
    bli_gemv( &BLIS_ONE, b, &t, &BLIS_ZERO, &w );
    bli_trmv( alpha, a, &w );
    bli_copyv( &w, &z );
  }
  else
  {
    bli_copyv( &t, &w );
    bli_trmv( &BLIS_ONE, a, &w );
    bli_gemv( alpha, b, &w, &BLIS_ZERO, &z );
  }

  bli_gemv( beta, c_orig, &t, &BLIS_ONE, &z );

  bli_subv( &z, &v );
  bli_normfv( &v, &norm );
  bli_getsc( &norm, &resid, &junk );

  bli_obj_free( &t );
  bli_obj_free( &v );
  bli_obj_free( &w );
  bli_obj_free( &z );

  return resid;
}