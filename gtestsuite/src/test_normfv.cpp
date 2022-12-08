#include "blis_test.h"
#include "blis_utils.h"
#include "test_normfv.h"

// Local prototypes.
void libblis_test_normfv_deps (
  thread_data_t* tdata,
  test_params_t* params,
  test_op_t*     op
);

void libblis_test_normfv_impl (
  iface_t   iface,
  obj_t*    x,
  obj_t*    norm
);

double libblis_test_normfv_check (
  test_params_t* params,
  obj_t*         beta,
  obj_t*         x,
  obj_t*         norm
);

double cblas_normfv(
  f77_int    mm,
  obj_t*     x,
  f77_int    incx,
  obj_t*     norm,
  num_t      dt
){
  switch( dt )  {
    case BLIS_FLOAT :
    {
      float*  nrm2   = (float*) bli_obj_buffer( norm );
      float*  xp     = (float*) bli_obj_buffer( x );
      *nrm2 = cblas_snrm2( mm, xp, incx );
      break;
    }
    case BLIS_DOUBLE :
    {
      double*  nrm2   = (double*) bli_obj_buffer( norm );
      double*  xp     = (double*) bli_obj_buffer( x );
      *nrm2 = cblas_dnrm2( mm, xp, incx );
      break;
    }
    case BLIS_SCOMPLEX :
    {
      float* nrm2     = (float*) bli_obj_buffer( norm );
      scomplex* xp    = (scomplex*) bli_obj_buffer( x );
      *nrm2 = cblas_scnrm2( mm, xp, incx );
      break;
    }
    case BLIS_DCOMPLEX :
    {
      double* nrm2    = (double*) bli_obj_buffer( norm );
      dcomplex* xp    = (dcomplex*) bli_obj_buffer( x );
      *nrm2 = cblas_dznrm2( mm, xp, incx );
      break;
    }
    default :
      bli_check_error_code( BLIS_INVALID_DATATYPE );
  }
  return 0;
}

double blas_normfv(
  f77_int    mm,
  obj_t*     x,
  f77_int    incx,
  obj_t*     norm,
  num_t      dt
){
  switch( dt )  {
    case BLIS_FLOAT :
    {
      float*  nrm2   = (float*) bli_obj_buffer( norm );
      float*  xp     = (float*) bli_obj_buffer( x );
      *nrm2 = snrm2_( &mm, xp, &incx );
      break;
    }
    case BLIS_DOUBLE :
    {
      double*  nrm2   = (double*) bli_obj_buffer( norm );
      double*  xp     = (double*) bli_obj_buffer( x );
      *nrm2 = dnrm2_( &mm, xp, &incx );
      break;
    }
    case BLIS_SCOMPLEX :
    {
      float* nrm2     = (float*) bli_obj_buffer( norm );
      scomplex* xp    = (scomplex*) bli_obj_buffer( x );
      *nrm2 = scnrm2_( &mm, xp, &incx );
      break;
    }
    case BLIS_DCOMPLEX :
    {
      double* nrm2    = (double*) bli_obj_buffer( norm );
      dcomplex* xp    = (dcomplex*) bli_obj_buffer( x );
      *nrm2 = dznrm2_( &mm, xp, &incx );
      break;
    }
    default :
      bli_check_error_code( BLIS_INVALID_DATATYPE );
  }
  return 0;
}

void libblis_api_normfv(
  test_params_t* params,
  iface_t        iface,
  obj_t*         x,
  obj_t*         norm,
  num_t          dt
){
  if(params->api == API_BLIS) {
    libblis_test_normfv_impl( iface, x, norm );
  }
  else { /*CLBAS  || BLAS */
    f77_int  m   = bli_obj_vector_dim( x );
    f77_int incx = bli_obj_vector_inc( x );

    if( params->api == API_CBLAS ) {
      cblas_normfv( m, x, incx, norm, dt );
    } else {
      blas_normfv( m, x, incx, norm, dt );
    }
  }
  return ;
}

double libblis_ref_normfv(
  test_params_t* params,
  obj_t*         beta,
  obj_t*         x,
  obj_t*         norm
) {
  double resid = 0.0;

  if((params->bitextf == 0) && (params->oruflw == BLIS_DEFAULT)) {
    resid = libblis_test_normfv_check( params, beta, x, norm);
  }
  else {
    if(params->oruflw == BLIS_DEFAULT) {
      resid = libblis_test_inormfv_check( params, beta, x, norm);
    }
    else {
      resid = libblis_test_vector_check(params, x);
    }
  }
  return resid;
}

double libblis_test_bitrp_normfv(
  test_params_t* params,
  iface_t        iface,
  obj_t*         x,
  obj_t*         norm,
  obj_t*         r,
  num_t          dt
) {
  double resid = 0.0;
	 unsigned int n_repeats = params->n_repeats;
	 unsigned int i;
  num_t dt_real = bli_dt_proj_to_real( dt );

  for(i = 0; i < n_repeats; i++) {
    bli_obj_scalar_init_detached( dt_real,  r );
    libblis_test_normfv_impl( iface, x, r );
    resid = libblis_test_bitrp_vector(norm, r, dt);
  }
  return resid;
}

double libblis_test_op_normfv (
  test_params_t* params,
  iface_t        iface,
  char*          dc_str,
  char*          pc_str,
  char*          sc_str,
  tensor_t*      dim
) {
  num_t        datatype;
  num_t        dt_real;
  dim_t        m;
  obj_t        beta, norm;
  obj_t        x;
  double       resid = 0.0;

  // Use the datatype of the first char in the datatype combination string.
  bli_param_map_char_to_blis_dt( dc_str[0], &datatype );

  // Compute the real projection of the chosen datatype.
  dt_real = bli_dt_proj_to_real( datatype );

  // Map the dimension specifier to an actual dimension.
  m = dim->m;

  // Create test scalars.
  bli_obj_scalar_init_detached( datatype, &beta );
  bli_obj_scalar_init_detached( dt_real,  &norm );

  // Create test operands (vectors and/or matrices).
  libblis_test_vobj_create( params, datatype, sc_str[0], m, &x );

  // Initialize beta to 2 - 2i.
  bli_setsc( 2.0, -2.0, &beta );

  // Set all elements of x to beta.
  bli_setv( &beta, &x );

  libblis_api_normfv( params, iface, &x, &norm, datatype );

#ifndef __GTEST_VALGRIND_TEST__
  if(params->bitrp) {
    obj_t r;

    resid = libblis_test_bitrp_normfv( params, iface, &x, &norm, &r, datatype);

    bli_obj_free( &r );
  }
  else {
    resid = libblis_ref_normfv( params, &beta, &x, &norm);
  }
#endif

  // Zero out performance and residual if input vector is empty.
  libblis_test_check_empty_problem( &x, &resid );

  // Free the test objects.
  libblis_test_obj_free( &x );

  return abs(resid);
}

void libblis_test_normfv_impl (
  iface_t   iface,
  obj_t*    x,
  obj_t*    norm
){
  switch ( iface )
  {
    case BLIS_TEST_SEQ_FRONT_END:
      bli_normfv( x, norm );
      break;

    default:
      libblis_test_printf_error( "Invalid interface type.\n" );
  }
}

double libblis_test_normfv_check (
  test_params_t* params,
  obj_t*         beta,
  obj_t*         x,
  obj_t*         norm
) {
  num_t  dt_real = bli_obj_dt_proj_to_real( x );
  dim_t  m       = bli_obj_vector_dim( x );

  obj_t  m_r, temp_r;

  double junk;
  double resid = 0.0;
  //
  // Pre-conditions:
  // - x is set to beta.
  // Note:
  // - beta should have a non-zero imaginary component in the complex
  //   cases in order to more fully exercise the implementation.
  //
  // Under these conditions, we assume that the implementation for
  //
  //   norm := normfv( x )
  //
  // is functioning correctly if
  //
  //   norm = sqrt( absqsc( beta ) * m )
  //
  // where m is the length of x.
  //

  bli_obj_scalar_init_detached( dt_real, &temp_r );
  bli_obj_scalar_init_detached( dt_real, &m_r );

  bli_setsc( ( double )m, 0.0, &m_r );

  bli_absqsc( beta, &temp_r );
  bli_mulsc( &m_r, &temp_r );
  bli_sqrtsc( &temp_r, &temp_r );
  bli_subsc( &temp_r, norm );

  bli_getsc( norm, &resid, &junk );

  return resid;
}