#include "blis_test.h"
#include "blis_utils.h"
#include "test_dotv.h"

// Local prototypes.
void libblis_test_dotv_deps (
  thread_data_t* tdata,
  test_params_t* params,
  test_op_t*     op
);

void libblis_test_dotv_impl (
  iface_t   iface,
  obj_t*    x,
  obj_t*    y,
  obj_t*    rho
);

double libblis_test_dotv_check (
  test_params_t* params,
  obj_t*         x,
  obj_t*         y,
  obj_t*         rho
);

double cblas_dotv(
  f77_int    m,
  obj_t*     x,
  f77_int    incx,
  obj_t*     y,
  f77_int    incy,
  obj_t*     rho,
  num_t      dt
){
  switch( dt )  {
    case BLIS_FLOAT :
    {
      float*  xp     = (float*) bli_obj_buffer( x );
      float*  yp     = (float*) bli_obj_buffer( y );
      float*  resp   = (float*) bli_obj_buffer( rho );
      *resp = cblas_sdot( m, xp, incx, yp, incy );
      break;
    }
    case BLIS_DOUBLE :
    {
      double*  xp    = (double*) bli_obj_buffer( x );
      double*  yp    = (double*) bli_obj_buffer( y );
      double*  resp  = (double*) bli_obj_buffer( rho );
      *resp = cblas_ddot( m, xp, incx, yp, incy );
      break;
    }
    case BLIS_SCOMPLEX :
    {
      scomplex*  xp    = (scomplex*) bli_obj_buffer( x );
      scomplex*  yp    = (scomplex*) bli_obj_buffer( y );
      scomplex*  resp  = (scomplex*) bli_obj_buffer( rho );
      cblas_cdotu_sub( m, xp, incx, yp, incy, resp );
      break;
    }
    case BLIS_DCOMPLEX :
    {
      dcomplex*  xp    = (dcomplex*) bli_obj_buffer( x );
      dcomplex*  yp    = (dcomplex*) bli_obj_buffer( y );
      dcomplex*  resp  = (dcomplex*) bli_obj_buffer( rho );
      cblas_zdotu_sub( m, xp, incx, yp, incy, resp );
      break;
    }
    default :
      bli_check_error_code( BLIS_INVALID_DATATYPE );
  }
  return 0;
}

double blas_dotv(
  f77_int    m,
  obj_t*     x,
  f77_int    incx,
  obj_t*     y,
  f77_int    incy,
  obj_t*     rho,
  num_t      dt
){
  switch( dt )  {
    case BLIS_FLOAT :
    {
      float*  xp     = (float*) bli_obj_buffer( x );
      float*  yp     = (float*) bli_obj_buffer( y );
      float*  resp   = (float*) bli_obj_buffer( rho );
      *resp = sdot_( &m, xp, &incx, yp, &incy );
      break;
    }
    case BLIS_DOUBLE :
    {
      double*  xp    = (double*) bli_obj_buffer( x );
      double*  yp    = (double*) bli_obj_buffer( y );
      double*  resp  = (double*) bli_obj_buffer( rho );
      *resp = ddot_( &m, xp, &incx, yp, &incy );
      break;
    }
    case BLIS_SCOMPLEX :
    {
      scomplex*  xp    = (scomplex*) bli_obj_buffer( x );
      scomplex*  yp    = (scomplex*) bli_obj_buffer( y );
      scomplex*  resp  = (scomplex*) bli_obj_buffer( rho );
#ifdef BLIS_DISABLE_COMPLEX_RETURN_INTEL
      *resp = cdotu_( &m, xp, &incx, yp, &incy );
#else
      cdotu_( resp, &m, xp, &incx, yp, &incy );
#endif // BLIS_DISABLE_COMPLEX_RETURN_INTEL ...
      break;
    }
    case BLIS_DCOMPLEX :
    {
      dcomplex*  xp    = (dcomplex*) bli_obj_buffer( x );
      dcomplex*  yp    = (dcomplex*) bli_obj_buffer( y );
      dcomplex*  resp  = (dcomplex*) bli_obj_buffer( rho );
#ifdef BLIS_DISABLE_COMPLEX_RETURN_INTEL
      *resp = zdotu_( &m, xp, &incx, yp, &incy );
#else
      zdotu_( resp, &m, xp, &incx, yp, &incy );
#endif // BLIS_DISABLE_COMPLEX_RETURN_INTEL ...
      break;
    }
    default :
      bli_check_error_code( BLIS_INVALID_DATATYPE );
  }
  return 0;
}

void libblis_api_dotv(
  test_params_t* params,
  iface_t        iface,
  obj_t*         x,
  obj_t*         y,
  obj_t*         rho,
  num_t          dt
){
  if(params->api == API_BLIS) {
    libblis_test_dotv_impl( iface, x, y, rho );
  }
  else { /*CLBAS  || BLAS */
    dim_t  m     = bli_obj_vector_dim( x );
    f77_int incx = bli_obj_vector_inc( x );
    f77_int incy = bli_obj_vector_inc( y );

    if(bli_obj_has_conj(x)) {
       conjugate_tensor(x, dt);
       bli_obj_set_conj( BLIS_NO_CONJUGATE, x );
    }

    if(bli_obj_has_conj(y)) {
       conjugate_tensor(y, dt);
       bli_obj_set_conj( BLIS_NO_CONJUGATE, y );
    }

    if( params->api == API_CBLAS ) {
      cblas_dotv( m, x, incx, y, incy, rho, dt );
    } else {
      blas_dotv( m, x, incx, y, incy, rho, dt );
    }
  }
  return ;
}

double libblis_ref_dotv(
  test_params_t* params,
  obj_t*         x,
  obj_t*         y,
  obj_t*         rho
) {
  double resid = 0.0;

  if((params->bitextf == 0) && (params->oruflw == BLIS_DEFAULT)) {
    resid = libblis_test_dotv_check( params, x, y, rho );
  }
  else {
    if(params->oruflw == BLIS_DEFAULT) {
      resid = libblis_test_idotv_check( params, x, y, rho );
    }
    else {
      resid = libblis_test_vector_check(params, rho);
    }
  }
  return resid;
}

double libblis_test_bitrp_dotv(
  test_params_t* params,
  iface_t        iface,
  obj_t*         x,
  obj_t*         y,
  obj_t*         rho,
  obj_t*         rh,
  num_t          dt
) {
  double resid = 0.0;
	 unsigned int n_repeats = params->n_repeats;
	 unsigned int i;

  for(i = 0; i < n_repeats; i++) {
    bli_copysc( &BLIS_MINUS_ONE, rh );
    libblis_test_dotv_impl( iface, x, y, rh );
    resid = libblis_test_bitrp_vector(rho, rh, dt);
  }
  return resid;
}

double libblis_test_op_dotv (
  test_params_t* params,
  iface_t        iface,
  char*          dc_str,
  char*          pc_str,
  char*          sc_str,
  tensor_t*      dim
){
  num_t        datatype;
  dim_t        m;
  conj_t       conjx, conjy, conjconjxy;
  obj_t        x, y, rho;
  double       resid = 0.0;
  obj_t        xx, yy;

  // Use the datatype of the first char in the datatype combination string.
  bli_param_map_char_to_blis_dt( dc_str[0], &datatype );

  // Map the dimension specifier to actual dimensions.
  m = dim->m;

  // Map parameter characters to BLIS constants.
  bli_param_map_char_to_blis_conj( pc_str[0], &conjx );
  bli_param_map_char_to_blis_conj( pc_str[1], &conjy );

  // Create test scalars.
  bli_obj_scalar_init_detached( datatype, &rho );

  // Create test operands (vectors and/or matrices).
  libblis_test_vobj_create( params, datatype, sc_str[0], m, &x );
  libblis_test_vobj_create( params, datatype, sc_str[0], m, &xx );
  libblis_test_vobj_create( params, datatype, sc_str[1], m, &y );
  libblis_test_vobj_create( params, datatype, sc_str[1], m, &yy );

  // Randomize x.
  if((params->bitextf == 0) && (params->oruflw == BLIS_DEFAULT)) {
    libblis_test_vobj_randomize( params, TRUE, &x );
  } else {
    libblis_test_vobj_irandomize( params, &x );
  }

  // Determine whether to make a copy of x with or without conjugation.
  //  conjx conjy  ~conjx^conjy   y is initialized as
  //  n     n      c              y = conj(x)
  //  n     c      n              y = x
  //  c     n      n              y = x
  //  c     c      c              y = conj(x)

  conjconjxy = bli_apply_conj( conjx, conjy );
  conjconjxy = bli_conj_toggled( conjconjxy );
  bli_obj_set_conj( conjconjxy, &x );
  bli_copyv( &x, &y );

  // Apply the parameters.
  bli_obj_set_conj( conjx, &x );
  bli_obj_set_conj( conjy, &y );

  bli_copysc( &BLIS_MINUS_ONE, &rho );

  bli_copyv( &x, &xx );
  bli_copyv( &y, &yy );

  libblis_api_dotv( params, iface, &xx, &yy, &rho, datatype);

#ifndef __GTEST_VALGRIND_TEST__
  if(params->bitrp) {
    obj_t r;

    bli_obj_scalar_init_detached( datatype, &r );

    resid = libblis_test_bitrp_dotv( params, iface, &x, &y, &rho, &r, datatype);

    bli_obj_free( &r );
  }
  else {
    resid = libblis_ref_dotv( params, &x, &y, &rho );
  }
#endif

  // Zero out performance and residual if output scalar is empty.
  libblis_test_check_empty_problem( &rho, &resid );

  // Free the test objects.
  libblis_test_obj_free( &x );
  libblis_test_obj_free( &xx );
  libblis_test_obj_free( &y );
  libblis_test_obj_free( &yy );

  return abs(resid);
}

void libblis_test_dotv_impl (
  iface_t   iface,
  obj_t*    x,
  obj_t*    y,
  obj_t*    rho
) {
  switch ( iface )
  {
    case BLIS_TEST_SEQ_FRONT_END:
      bli_dotv( x, y, rho );
      break;

    default:
      libblis_test_printf_error( "Invalid interface type.\n" );
  }
}

double libblis_test_dotv_check (
  test_params_t* params,
  obj_t*         x,
  obj_t*         y,
  obj_t*         rho
){
  num_t  dt_real = bli_obj_dt_proj_to_real( y );

  obj_t  rho_r, rho_i;
  obj_t  norm_x, norm_xy;

  double zero;
  double junk;
  double resid = 0.0;

  //
  // Pre-conditions:
  // - x is randomized.
  // - y is equal to conj(conjx(conjy(x))).
  //
  // Under these conditions, we assume that the implementation for
  //
  //   rho := conjx(x^T) conjy(y)
  //
  // is functioning correctly if
  //
  //   sqrtsc( rho.real ) - normfv( x )
  //
  // and
  //
  //   rho.imag
  //
  // are negligible.
  //

  bli_obj_scalar_init_detached( dt_real, &rho_r );
  bli_obj_scalar_init_detached( dt_real, &rho_i );
  bli_obj_scalar_init_detached( dt_real, &norm_x );
  bli_obj_scalar_init_detached( dt_real, &norm_xy );

  bli_normfv( x, &norm_x );

  bli_unzipsc( rho, &rho_r, &rho_i );

  bli_sqrtsc( &rho_r, &norm_xy );

  bli_subsc( &norm_x, &norm_xy );
  bli_getsc( &norm_xy, &resid, &junk );
  bli_getsc( &rho_i,   &zero, &junk );

  resid = bli_fmaxabs( resid, zero );

  return resid;
}