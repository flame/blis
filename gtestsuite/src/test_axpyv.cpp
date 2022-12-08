#include "blis_test.h"
#include "blis_utils.h"
#include "test_axpyv.h"

// Local prototypes.
void libblis_test_axpyv_deps (
  thread_data_t* tdata,
  test_params_t* params,
  test_op_t*     op
);

void libblis_test_axpyv_impl (
  iface_t   iface,
  obj_t*    alpha,
  obj_t*    x,
  obj_t*    y
);

double libblis_test_axpyv_check (
  test_params_t* params,
  obj_t*         alpha,
  obj_t*         x,
  obj_t*         y,
  obj_t*         y_orig
);

double cblas_axpyv(
  f77_int    m,
  obj_t*     alpha,
  obj_t*     x,
  f77_int    incx,
  obj_t*     y,
  f77_int    incy,
  num_t      dt
){
  switch( dt )  {
    case BLIS_FLOAT :
    {
      float*  alphap  = (float*) bli_obj_buffer( alpha );
      float*  xp      = (float*) bli_obj_buffer( x );
      float*  yp      = (float*) bli_obj_buffer( y );
      cblas_saxpy( m, *alphap, xp, incx, yp, incy );
      break;
    }
    case BLIS_DOUBLE :
    {
      double*  alphap = (double*) bli_obj_buffer( alpha );
      double*  xp     = (double*) bli_obj_buffer( x );
      double*  yp     = (double*) bli_obj_buffer( y );
      cblas_daxpy( m, *alphap, xp, incx, yp, incy );
      break;
    }
    case BLIS_SCOMPLEX :
    {
      scomplex*  alphap = (scomplex*) bli_obj_buffer( alpha );
      scomplex*  xp     = (scomplex*) bli_obj_buffer( x );
      scomplex*  yp     = (scomplex*) bli_obj_buffer( y );
      cblas_caxpy( m, alphap, xp, incx, yp, incy );
      break;
    }
    case BLIS_DCOMPLEX :
    {
      dcomplex*  alphap = (dcomplex*) bli_obj_buffer( alpha );
      dcomplex*  xp     = (dcomplex*) bli_obj_buffer( x );
      dcomplex*  yp     = (dcomplex*) bli_obj_buffer( y );
      cblas_zaxpy( m, alphap, xp, incx, yp, incy );
      break;
    }
    default :
      bli_check_error_code( BLIS_INVALID_DATATYPE );
  }
  return 0;
}

double blas_axpyv(
  f77_int    m,
  obj_t*     alpha,
  obj_t*     x,
  f77_int    incx,
  obj_t*     y,
  f77_int    incy,
  num_t      dt
){
  switch( dt )  {
    case BLIS_FLOAT :
    {
      float*  alphap    = (float*) bli_obj_buffer( alpha );
      float*  xp        = (float*) bli_obj_buffer( x );
      float*  yp        = (float*) bli_obj_buffer( y );
      saxpy_( &m, alphap, xp, &incx, yp, &incy );
      break;
    }
    case BLIS_DOUBLE :
    {
      double*  alphap   = (double*) bli_obj_buffer( alpha );
      double*  xp       = (double*) bli_obj_buffer( x );
      double*  yp       = (double*) bli_obj_buffer( y );
      daxpy_( &m, alphap, xp, &incx, yp, &incy );
      break;
    }
    case BLIS_SCOMPLEX :
    {
      scomplex*  alphap = (scomplex*) bli_obj_buffer( alpha );
      scomplex*  xp     = (scomplex*) bli_obj_buffer( x );
      scomplex*  yp     = (scomplex*) bli_obj_buffer( y );
      caxpy_( &m, alphap, xp, &incx, yp, &incy );
      break;
    }
    case BLIS_DCOMPLEX :
    {
      dcomplex*  alphap = (dcomplex*) bli_obj_buffer( alpha );
      dcomplex*  xp     = (dcomplex*) bli_obj_buffer( x );
      dcomplex*  yp     = (dcomplex*) bli_obj_buffer( y );
      zaxpy_( &m, alphap, xp, &incx, yp, &incy );
      break;
    }
    default :
      bli_check_error_code( BLIS_INVALID_DATATYPE );
  }
  return 0;
}

void libblis_api_axpyv(
  test_params_t* params,
  iface_t        iface,
  obj_t*         alpha,
  obj_t*         x,
  obj_t*         y,
  num_t          dt
){
  if(params->api == API_BLIS) {
    libblis_test_axpyv_impl( iface, alpha, x, y );
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
      cblas_axpyv( m, alpha, x, incx, y, incy, dt );
    } else {
      blas_axpyv( m, alpha, x, incx, y, incy, dt );
    }
  }
  return ;
}

double libblis_ref_axpyv(
  test_params_t* params,
  obj_t*  alpha,
  obj_t*  x,
  obj_t*  y,
  obj_t*  y_save
) {
  double resid = 0.0;

  if((params->bitextf == 0) && (params->oruflw == BLIS_DEFAULT)) {
 	  // Perform checks.
    resid = libblis_test_axpyv_check( params, alpha, x, y, y_save );
  }
  else {
    if(params->oruflw == BLIS_DEFAULT) {
      resid = libblis_test_iaxpyv_check( params, alpha, x, y, y_save );
    }
    else {
      resid = libblis_test_vector_check(params, y);
    }
  }
  return resid;
}

double libblis_test_bitrp_axpyv(
  test_params_t* params,
  iface_t        iface,
  obj_t*         alpha,
  obj_t*         x,
  obj_t*         y,
  obj_t*         y_save,
  obj_t*         r,
  num_t          dt
) {
  double resid = 0.0;
	 unsigned int n_repeats = params->n_repeats;
	 unsigned int i;

  for(i = 0; i < n_repeats; i++) {
    bli_copyv( y_save, r );
    libblis_test_axpyv_impl( iface, alpha, x, r );
    resid = libblis_test_bitrp_vector(y, r, dt);
  }
  return resid;
}

double libblis_test_op_axpyv (
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
  conj_t       conjx;
  obj_t        alpha, x, y;
  obj_t        y_save;
  double       resid = 0.0;
  obj_t        xx;

  // Use the datatype of the first char in the datatype combination string.
  bli_param_map_char_to_blis_dt( dc_str[0], &datatype );

  // Map the dimension specifier to actual dimensions.
  m = dim->m;

  // Map parameter characters to BLIS constants.
  bli_param_map_char_to_blis_conj( pc_str[0], &conjx );

  // Create test scalars.
  bli_obj_scalar_init_detached( datatype, &alpha );

  // Create test operands (vectors and/or matrices).
  libblis_test_vobj_create( params, datatype, sc_str[0], m, &x );
  libblis_test_vobj_create( params, datatype, sc_str[0], m, &xx );
  libblis_test_vobj_create( params, datatype, sc_str[1], m, &y );
  libblis_test_vobj_create( params, datatype, sc_str[1], m, &y_save );

  // Set alpha.
  if ( bli_obj_is_real( &y ) )
   bli_setsc( -2.0,  0.0, &alpha );
  else
   bli_setsc(  0.0, -2.0, &alpha );

  // Randomize x and y, and save y.
  if((params->bitextf == 0) && (params->oruflw == BLIS_DEFAULT)) {
    libblis_test_vobj_randomize( params, FALSE, &x );
    libblis_test_vobj_randomize( params, FALSE, &y );
  } else {
    libblis_test_vobj_irandomize( params, &x );
    libblis_test_vobj_irandomize( params, &y );
  }

  bli_copyv( &y, &y_save );

  // Apply the parameters.
  bli_obj_set_conj( conjx, &x );

  bli_copyv( &x, &xx );

  libblis_api_axpyv( params, iface, &alpha, &xx, &y, datatype);

#ifndef __GTEST_VALGRIND_TEST__
  if(params->bitrp) {
    obj_t r;

    libblis_test_vobj_create( params, datatype, sc_str[1], m, &r );

    resid = libblis_test_bitrp_axpyv( params, iface, &alpha, &x,
                                              &y, &y_save, &r, datatype);
    bli_obj_free( &r );
  }
  else {
    resid = libblis_ref_axpyv( params, &alpha, &x, &y, &y_save );
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

void libblis_test_axpyv_impl (
  iface_t   iface,
  obj_t*    alpha,
  obj_t*    x,
  obj_t*    y
) {
  switch ( iface )
  {
    case BLIS_TEST_SEQ_FRONT_END:
      bli_axpyv( alpha, x, y );
      break;

    default:
      libblis_test_printf_error( "Invalid interface type.\n" );
  }
}

double libblis_test_axpyv_check (
  test_params_t* params,
  obj_t*         alpha,
  obj_t*         x,
  obj_t*         y,
  obj_t*         y_orig
){
  num_t  dt      = bli_obj_dt( y );
  num_t  dt_real = bli_obj_dt_proj_to_real( y );

  dim_t  m       = bli_obj_vector_dim( y );

  obj_t  x_temp, y_temp;
  obj_t  norm;

  double junk;
  double resid = 0.0;
  //
  // Pre-conditions:
  // - x is randomized.
  // - y_orig is randomized.
  // Note:
  // - alpha should have a non-zero imaginary component in the complex
  //   cases in order to more fully exercise the implementation.
  //
  // Under these conditions, we assume that the implementation for
  //
  //   y := y_orig + alpha * conjx(x)
  //
  // is functioning correctly if
  //
  //   normfv( y - ( y_orig + alpha * conjx(x) ) )
  //
  // is negligible.
  //

  bli_obj_scalar_init_detached( dt_real, &norm );

  bli_obj_create( dt, m, 1, 0, 0, &x_temp );
  bli_obj_create( dt, m, 1, 0, 0, &y_temp );

  bli_copyv( x,      &x_temp );
  bli_copyv( y_orig, &y_temp );

  bli_scalv( alpha, &x_temp );
  bli_addv( &x_temp, &y_temp );

  bli_subv( &y_temp, y );
  bli_normfv( y, &norm );
  bli_getsc( &norm, &resid, &junk );

  bli_obj_free( &x_temp );
  bli_obj_free( &y_temp );

  return resid;
}