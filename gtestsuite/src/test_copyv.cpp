#include "blis_test.h"
#include "blis_utils.h"
#include "test_copyv.h"

// Local prototypes.
void libblis_test_copyv_deps (
  thread_data_t* tdata,
  test_params_t* params,
  test_op_t*     op
);

void libblis_test_copyv_impl (
  iface_t   iface,
  obj_t*    x,
  obj_t*    y
);

double libblis_test_copyv_check (
  test_params_t* params,
  obj_t*         x,
  obj_t*         y
);

double cblas_copyv(
  f77_int    m,
  obj_t*     x,
  f77_int    incx,
  obj_t*     y,
  f77_int    incy,
  num_t      dt
){
  switch( dt )  {
    case BLIS_FLOAT :
    {
      float*  xp     = (float*) bli_obj_buffer( x );
      float*  yp     = (float*) bli_obj_buffer( y );
      cblas_scopy( m, xp, incx, yp, incy );
      break;
    }
    case BLIS_DOUBLE :
    {
      double*  xp    = (double*) bli_obj_buffer( x );
      double*  yp    = (double*) bli_obj_buffer( y );
      cblas_dcopy( m, xp, incx, yp, incy );
      break;
    }
    case BLIS_SCOMPLEX :
    {
      scomplex*  xp    = (scomplex*) bli_obj_buffer( x );
      scomplex*  yp    = (scomplex*) bli_obj_buffer( y );
      cblas_ccopy( m, xp, incx, yp, incy );
      break;
    }
    case BLIS_DCOMPLEX :
    {
      dcomplex*  xp    = (dcomplex*) bli_obj_buffer( x );
      dcomplex*  yp    = (dcomplex*) bli_obj_buffer( y );
      cblas_zcopy( m, xp, incx, yp, incy );
      break;
    }
    default :
      bli_check_error_code( BLIS_INVALID_DATATYPE );
  }
  return 0;
}

double blas_copyv(
  f77_int    m,
  obj_t*     x,
  f77_int    incx,
  obj_t*     y,
  f77_int    incy,
  num_t      dt
){
  switch( dt )  {
    case BLIS_FLOAT :
    {
      float*  xp     = (float*) bli_obj_buffer( x );
      float*  yp     = (float*) bli_obj_buffer( y );
      scopy_( &m, xp, &incx, yp, &incy );
      break;
    }
    case BLIS_DOUBLE :
    {
      double*  xp    = (double*) bli_obj_buffer( x );
      double*  yp    = (double*) bli_obj_buffer( y );
      dcopy_( &m, xp, &incx, yp, &incy );
      break;
    }
    case BLIS_SCOMPLEX :
    {
      scomplex*  xp    = (scomplex*) bli_obj_buffer( x );
      scomplex*  yp    = (scomplex*) bli_obj_buffer( y );
      ccopy_( &m, xp, &incx, yp, &incy );
      break;
    }
    case BLIS_DCOMPLEX :
    {
      dcomplex*  xp    = (dcomplex*) bli_obj_buffer( x );
      dcomplex*  yp    = (dcomplex*) bli_obj_buffer( y );
      zcopy_( &m, xp, &incx, yp, &incy );
      break;
    }
    default :
      bli_check_error_code( BLIS_INVALID_DATATYPE );
  }
  return 0;
}

void libblis_api_copyv(
  test_params_t* params,
  iface_t        iface,
  obj_t*         x,
  obj_t*         y,
  num_t          dt
){
  if(params->api == API_BLIS) {
    libblis_test_copyv_impl( iface, x, y );
  }
  else { /*CLBAS  || BLAS */
    dim_t  m     = bli_obj_vector_dim( x );
    f77_int incx = bli_obj_vector_inc( x );
    f77_int incy = bli_obj_vector_inc( y );

    if(bli_obj_has_conj(x)) {
       conjugate_tensor(x, dt);
       bli_obj_set_conj( BLIS_NO_CONJUGATE, x );
    }

    if( params->api == API_CBLAS ) {
      cblas_copyv( m, x, incx, y, incy, dt );
    } else {
      blas_copyv( m, x, incx, y, incy, dt );
    }
  }
  return ;
}

double libblis_ref_copyv(
  test_params_t* params,
  obj_t*         x,
  obj_t*         y,
  obj_t*         y_save
) {
  double resid = 0.0;

  if((params->bitextf == 0) && (params->oruflw == BLIS_DEFAULT)) {
 	  // Perform checks.
    resid = libblis_test_copyv_check( params, x, y );
  }
  else {
   if(params->oruflw == BLIS_DEFAULT) {
      resid = libblis_test_icopyv_check( params, x, y, y_save );
    }
    else {
      resid = libblis_test_vector_check(params, y);
    }
  }
  return resid;
}

double libblis_test_bitrp_copyv(
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
    bli_setv( &BLIS_ONE, r );
    libblis_test_copyv_impl( iface, x, r );
    resid = libblis_test_bitrp_vector(y, r, dt);
  }
  return resid;
}

double libblis_test_op_copyv (
  test_params_t* params,
  iface_t        iface,
  char*          dc_str,
  char*          pc_str,
  char*          sc_str,
  tensor_t*      dim
){
  num_t        datatype;
  dim_t        m;
  conj_t       conjx;
  obj_t        x, y, y_save;
  double       resid = 0.0;
  obj_t        xx;

  // Use the datatype of the first char in the datatype combination string.
  bli_param_map_char_to_blis_dt( dc_str[0], &datatype );

  // Map the dimension specifier to actual dimensions.
  m = dim->m;

  // Map parameter characters to BLIS constants.
  bli_param_map_char_to_blis_conj( pc_str[0], &conjx );

  // Create test operands (vectors and/or matrices).
  libblis_test_vobj_create( params, datatype, sc_str[0], m, &x );
  libblis_test_vobj_create( params, datatype, sc_str[0], m, &xx );
  libblis_test_vobj_create( params, datatype, sc_str[1], m, &y );
  libblis_test_vobj_create( params, datatype, sc_str[1], m, &y_save );


  // Randomize x and set y to one.
  if((params->bitextf == 0) && (params->oruflw == BLIS_DEFAULT)) {
    libblis_test_vobj_randomize( params, FALSE, &x );
  }
  else {
    libblis_test_vobj_irandomize( params, &x );
  }

  bli_setv( &BLIS_ONE, &y );

  // Apply the parameters.
  bli_obj_set_conj( conjx, &x );

  bli_copyv( &y, &y_save );

  bli_copyv( &x, &xx );

  libblis_api_copyv( params, iface, &xx, &y, datatype);

#ifndef __GTEST_VALGRIND_TEST__
  if(params->bitrp) {
    resid = libblis_test_bitrp_copyv( params, iface, &x, &y, &y_save, datatype);
  }
  else {
    resid = libblis_ref_copyv( params, &x, &y, &y_save );
  }
#endif

  // Zero out performance and residual if output vector is empty.
  libblis_test_check_empty_problem( &y, &resid );

  // Free the test objects.
  libblis_test_obj_free( &x );
  libblis_test_obj_free( &xx );
  libblis_test_obj_free( &y );
  libblis_test_obj_free( &y_save );

  return abs(resid);
}

void libblis_test_copyv_impl (
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

double libblis_test_copyv_check (
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
  //   normfv( y - conjx(x) )
  //
  // is negligible.
  //

  bli_obj_scalar_init_detached( dt_real, &norm_y_r );

  bli_subv( x, y );

  bli_normfv( y, &norm_y_r );

  bli_getsc( &norm_y_r, &resid, &junk );

  return resid;
}