#include "blis_test.h"
#include "blis_utils.h"
#include "test_scalv.h"

// Local prototypes.
void libblis_test_scalv_deps (
  thread_data_t* tdata,
  test_params_t* params,
  test_op_t*     op
);

void libblis_test_scalv_impl (
  iface_t   iface,
  obj_t*    beta,
  obj_t*    y
);

double libblis_test_scalv_check (
  test_params_t* params,
  obj_t*         beta,
  obj_t*         y,
  obj_t*         y_orig
);

double cblas_scalv(
  f77_int    mm,
  obj_t*     beta,
  obj_t*     x,
  f77_int    incx,
  num_t      dt
){
  switch( dt )  {
    case BLIS_FLOAT :
    {
      float*  betap  = (float*) bli_obj_buffer( beta );
      float*  xp     = (float*) bli_obj_buffer( x );
      cblas_sscal( mm, *betap, xp, incx );
      break;
    }
    case BLIS_DOUBLE :
    {
      double*  betap  = (double*) bli_obj_buffer( beta );
      double*  xp     = (double*) bli_obj_buffer( x );
      cblas_dscal( mm, *betap, xp, incx );
      break;
    }
    case BLIS_SCOMPLEX :
    {
      scomplex* betap  = (scomplex*) bli_obj_buffer( beta );
      scomplex*  xp    = (scomplex*) bli_obj_buffer( x );
      cblas_cscal( mm, betap, xp, incx );
      break;
    }
    case BLIS_DCOMPLEX :
    {
      dcomplex* betap  = (dcomplex*) bli_obj_buffer( beta );
      dcomplex*  xp    = (dcomplex*) bli_obj_buffer( x );
      cblas_zscal( mm, betap, xp, incx );
      break;
    }
    default :
      bli_check_error_code( BLIS_INVALID_DATATYPE );
  }
  return 0;
}

double cblas_cscalv(
  f77_int    mm,
  obj_t*     beta,
  obj_t*     x,
  f77_int    incx,
  num_t      dt
){
  switch( dt )  {
    case BLIS_SCOMPLEX :
    {
      float* betap     = (float*) bli_obj_buffer( beta );
      scomplex*  xp    = (scomplex*) bli_obj_buffer( x );
      cblas_csscal( mm, *betap, xp, incx );
      break;
    }
    case BLIS_DCOMPLEX :
    {
      double* betap    = (double*) bli_obj_buffer( beta );
      dcomplex*  xp    = (dcomplex*) bli_obj_buffer( x );
      cblas_zdscal( mm, *betap, xp, incx );
      break;
    }
    default :
      bli_check_error_code( BLIS_INVALID_DATATYPE );
  }
  return 0;
}

double blas_scalv(
  f77_int    mm,
  obj_t*     beta,
  obj_t*     x,
  f77_int    incx,
  num_t      dt
){
  switch( dt )  {
    case BLIS_FLOAT :
    {
      float*  betap  = (float*) bli_obj_buffer( beta );
      float*  xp     = (float*) bli_obj_buffer( x );
      sscal_( &mm, betap, xp, &incx );
      break;
    }
    case BLIS_DOUBLE :
    {
      double*  betap  = (double*) bli_obj_buffer( beta );
      double*  xp     = (double*) bli_obj_buffer( x );
      dscal_( &mm, betap, xp, &incx );
      break;
    }
    case BLIS_SCOMPLEX :
    {
      scomplex*  betap = (scomplex*) bli_obj_buffer( beta );
      scomplex*  xp    = (scomplex*) bli_obj_buffer( x );
      cscal_( &mm, betap, xp, &incx );
      break;
    }
    case BLIS_DCOMPLEX :
    {
      dcomplex*  betap = (dcomplex*) bli_obj_buffer( beta );
      dcomplex*  xp    = (dcomplex*) bli_obj_buffer( x );
      zscal_( &mm, betap, xp, &incx );
      break;
    }
    default :
      bli_check_error_code( BLIS_INVALID_DATATYPE );
  }
  return 0;
}

double blas_cscalv(
  f77_int    mm,
  obj_t*     beta,
  obj_t*     x,
  f77_int    incx,
  num_t      dt
){
  switch( dt )  {
    case BLIS_SCOMPLEX :
    {
      float*  betap   = (float*) bli_obj_buffer( beta );
      scomplex*  xp   = (scomplex*) bli_obj_buffer( x );
      csscal_( &mm, betap, xp, &incx );
      break;
    }
    case BLIS_DCOMPLEX :
    {
      double*  betap = (double*) bli_obj_buffer( beta );
      dcomplex*  xp  = (dcomplex*) bli_obj_buffer( x );
      zdscal_( &mm, betap, xp, &incx );
      break;
    }
    default :
      bli_check_error_code( BLIS_INVALID_DATATYPE );
  }
  return 0;
}

void libblis_api_scalv(
  test_params_t* params,
  iface_t        iface,
  obj_t*         beta,
  obj_t*         x,
  num_t          dt
){
  if(params->api == API_BLIS) {
    libblis_test_scalv_impl( iface, beta, x );
  }
  else { /*CLBAS  || BLAS */
    dim_t  m     = bli_obj_vector_dim( x );
    f77_int incx = bli_obj_vector_inc( x );

    if(bli_obj_has_conj(beta)) {
       conjugate_tensor(beta, dt);
       bli_obj_set_conj( BLIS_NO_CONJUGATE, beta );
    }

    if( params->mixed_precision == 0 ) {
      if( params->api == API_CBLAS ) {
        cblas_scalv( m, beta, x, incx, dt );
      } else {
        blas_scalv( m, beta, x, incx, dt );
      }
    }
    else{
      if( params->api == API_CBLAS ) {
        cblas_cscalv( m, beta, x, incx, dt );
      } else {
        blas_cscalv( m, beta, x, incx, dt );
      }
    }
  }
  return ;
}

double libblis_ref_scalv(
  test_params_t* params,
  obj_t*  beta,
  obj_t*  y,
  obj_t*  y_save
) {
  double resid = 0.0;

  if((params->bitextf == 0) && (params->oruflw == BLIS_DEFAULT)) {
    resid = libblis_test_scalv_check( params, beta, y, y_save);
  }
  else {
    if(params->oruflw == BLIS_DEFAULT) {
      resid = libblis_test_iscalv_check( params, beta, y, y_save);
    }
    else {
      resid = libblis_test_vector_check(params, y);
    }
  }
  return resid;
}

double libblis_test_bitrp_scalv(
  test_params_t* params,
  iface_t        iface,
  obj_t*         beta,
  obj_t*         y,
  obj_t*         y_orig,
  obj_t*         r,
  num_t          dt
) {
  double resid = 0.0;
  unsigned int n_repeats = params->n_repeats;
  unsigned int i;

  for(i = 0 ; i < n_repeats ; i++) {
    bli_copyv( y_orig, r );
    libblis_test_scalv_impl( iface, beta, r );
    resid = libblis_test_bitrp_vector(y, r, dt);
  }
  return resid;
}

double libblis_test_op_scalv_md (
  test_params_t* params,
  iface_t        iface,
  char*          dc_str,
  char*          pc_str,
  char*          sc_str,
  tensor_t*      dim,
  atom_t         alpv
){
  num_t        dt_beta, dt_y;
  dim_t        m;
  conj_t       conjbeta;
  obj_t        beta, y;
  obj_t        y_save;
  double       resid = 0.0;
  obj_t        dbeta;

	// Decode the datatype combination string.
	bli_param_map_char_to_blis_dt( dc_str[0], &dt_y );
	bli_param_map_char_to_blis_dt( dc_str[1], &dt_beta );

  // Map the dimension specifier to an actual dimension.
  m = dim->m;

  // Map parameter characters to BLIS constants.
  bli_param_map_char_to_blis_conj( pc_str[0], &conjbeta );

  // Create test scalars.
  bli_obj_scalar_init_detached( dt_beta, &beta );
  bli_obj_scalar_init_detached( dt_beta, &dbeta );

  // Create test operands (vectors and/or matrices).
  libblis_test_vobj_create( params, dt_y, sc_str[0], m, &y );
  libblis_test_vobj_create( params, dt_y, sc_str[0], m, &y_save );

  if((params->bitextf == 0) && (params->oruflw == BLIS_DEFAULT)) {
    if ( bli_obj_is_real( &beta ) )
      bli_setsc( -2.0,  0.0, &beta );
    else
      bli_setsc(  0.0, -2.0, &beta );

    // Randomize x.
    libblis_test_vobj_randomize( params, FALSE, &y );
  }
  else{
    if ( bli_obj_is_real( &beta ) )
      bli_setsc( -2.0,  0.0, &beta );
    else
      bli_setsc(  0.0, -2.0, &beta );

    // Randomize x.
    libblis_test_vobj_irandomize( params, &y );
  }

  bli_copyv( &y, &y_save );

  // Apply the parameters.
  bli_obj_set_conj( conjbeta, &beta );

  bli_copysc( &beta, &dbeta );

  libblis_api_scalv( params, iface, &dbeta, &y, dt_y );

#ifndef __GTEST_VALGRIND_TEST__
  if(params->bitrp) {
    obj_t r;

    libblis_test_vobj_create( params, dt_y, sc_str[0], m, &r );


    resid = libblis_test_bitrp_scalv( params, iface, &beta, &y,
                                                &y_save, &r, dt_y);

    bli_obj_free( &r );
  }
  else {
    resid = libblis_ref_scalv( params, &beta, &y, &y_save);
  }
#endif

  // Zero out performance and residual if output vector is empty.
  libblis_test_check_empty_problem( &y, &resid );

  // Free the test objects.
  libblis_test_obj_free( &y );
  libblis_test_obj_free( &y_save );

  return resid;
}

double libblis_test_op_scalv (
  test_params_t* params,
  iface_t        iface,
  char*          dc_str,
  char*          pc_str,
  char*          sc_str,
  tensor_t*      dim,
  atom_t         alpv
){
  num_t        datatype;
  dim_t        m;
  conj_t       conjbeta;
  obj_t        beta, y;
  obj_t        y_save;
  double       resid = 0.0;
  obj_t        dbeta;

	// Use a different function to handle mixed datatypes.
	if ( params->mixed_domain || params->mixed_precision )
	{
      resid = libblis_test_op_scalv_md( params, iface, dc_str,
                                pc_str, sc_str, dim, alpv );
		return resid;
	}

  // Use the datatype of the first char in the datatype combination string.
  bli_param_map_char_to_blis_dt( dc_str[0], &datatype );

  // Map the dimension specifier to an actual dimension.
  m = dim->m;

  // Map parameter characters to BLIS constants.
  bli_param_map_char_to_blis_conj( pc_str[0], &conjbeta );

  // Create test scalars.
  bli_obj_scalar_init_detached( datatype, &beta );
  bli_obj_scalar_init_detached( datatype, &dbeta );

  // Create test operands (vectors and/or matrices).
  libblis_test_vobj_create( params, datatype, sc_str[0], m, &y );
  libblis_test_vobj_create( params, datatype, sc_str[0], m, &y_save );

  if((params->bitextf == 0) && (params->oruflw == BLIS_DEFAULT)) {
    if ( bli_obj_is_real( &beta ) )
      bli_setsc( -2.0,  0.0, &beta );
    else
      bli_setsc(  0.0, -2.0, &beta );

    // Randomize x.
    libblis_test_vobj_randomize( params, FALSE, &y );
  }
  else{
    if ( bli_obj_is_real( &beta ) )
      bli_setsc( -2.0,  0.0, &beta );
    else
      bli_setsc(  0.0, -2.0, &beta );

    // Randomize x.
    libblis_test_vobj_irandomize( params, &y );
  }

  bli_copyv( &y, &y_save );

  // Apply the parameters.
  bli_obj_set_conj( conjbeta, &beta );

  bli_copysc( &beta, &dbeta );

  libblis_api_scalv( params, iface, &dbeta, &y, datatype );

#ifndef __GTEST_VALGRIND_TEST__
  if(params->bitrp) {
    obj_t r;

    libblis_test_vobj_create( params, datatype, sc_str[0], m, &r );


    resid = libblis_test_bitrp_scalv( params, iface, &beta, &y,
                                                &y_save, &r, datatype);

    bli_obj_free( &r );
  }
  else {
    resid = libblis_ref_scalv( params, &beta, &y, &y_save);
  }
#endif

  // Zero out performance and residual if output vector is empty.
  libblis_test_check_empty_problem( &y, &resid );

  // Free the test objects.
  libblis_test_obj_free( &y );
  libblis_test_obj_free( &y_save );

  return abs(resid);
}

void libblis_test_scalv_impl (
  iface_t   iface,
  obj_t*    beta,
  obj_t*    y
) {
  switch ( iface )
  {
    case BLIS_TEST_SEQ_FRONT_END:
      bli_scalv( beta, y );
      break;

    default:
      libblis_test_printf_error( "Invalid interface type.\n" );
  }
}

double libblis_test_scalv_check (
  test_params_t* params,
  obj_t*         beta,
  obj_t*         y,
  obj_t*         y_orig
) {
  num_t  dt      = bli_obj_dt( y );
  num_t  dt_real = bli_obj_dt_proj_to_real( y );

  dim_t  m       = bli_obj_vector_dim( y );

  obj_t  norm_y_r;
  obj_t  nbeta;

  obj_t  y2;

  double junk;

  double resid = 0.0;

  //
  // Pre-conditions:
  // - y_orig is randomized.
  // Note:
  // - beta should have a non-zero imaginary component in the complex
  //   cases in order to more fully exercise the implementation.
  //
  // Under these conditions, we assume that the implementation for
  //
  //   y := conjbeta(beta) * y_orig
  //
  // is functioning correctly if
  //
  //   normfv( y + -conjbeta(beta) * y_orig )
  //
  // is negligible.
  //

  bli_obj_create( dt, m, 1, 0, 0, &y2 );
  bli_copyv( y_orig, &y2 );

  bli_obj_scalar_init_detached( dt,      &nbeta );
  bli_obj_scalar_init_detached( dt_real, &norm_y_r );

  bli_copysc( beta, &nbeta );
  bli_mulsc( &BLIS_MINUS_ONE, &nbeta );

  bli_scalv( &nbeta, &y2 );
  bli_addv( &y2, y );

  bli_normfv( y, &norm_y_r );

  bli_getsc( &norm_y_r, &resid, &junk );

  bli_obj_free( &y2 );

  return resid;
}