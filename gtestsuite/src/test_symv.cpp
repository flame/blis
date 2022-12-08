#include "blis_test.h"
#include "blis_utils.h"
#include "test_symv.h"

// Local prototypes.
void libblis_test_symv_deps(
  thread_data_t* tdata,
  test_params_t* params,
  test_op_t*     op
);

void libblis_test_symv_impl(
  iface_t   iface,
  obj_t*    alpha,
  obj_t*    a,
  obj_t*    x,
  obj_t*    beta,
  obj_t*    y
);

double libblis_test_symv_check(
  test_params_t* params,
  obj_t*         alpha,
  obj_t*         a,
  obj_t*         x,
  obj_t*         beta,
  obj_t*         y,
  obj_t*         y_orig
);

void cblas_symv(
  uplo_t  uploa,
  f77_int m,
  obj_t*  alpha,
  obj_t*  a,
  f77_int lda,
  obj_t*  x,
  f77_int incx,
  obj_t*  beta,
  obj_t*  y,
  f77_int incy,
  num_t   dt
){
  enum CBLAS_ORDER cblas_order;
 	enum CBLAS_UPLO  cblas_uplo ;

  if ( bli_obj_row_stride( a ) == 1 )
    cblas_order = CblasColMajor;
  else
    cblas_order = CblasRowMajor;

  if( bli_is_upper( uploa ) )
    cblas_uplo = CblasUpper;
  else
    cblas_uplo = CblasLower;

  switch( dt )  {
    case BLIS_FLOAT :
    {
      float*   alphap = (float*) bli_obj_buffer( alpha );
      float*   ap     = (float*) bli_obj_buffer( a );
      float*   xp     = (float*) bli_obj_buffer( x );
      float*   betap  = (float*) bli_obj_buffer( beta );
      float*   yp     = (float*) bli_obj_buffer( y );
      cblas_ssymv(cblas_order, cblas_uplo, m, *alphap, ap, lda, xp, incx,
                                                        *betap, yp, incy);
      break;
    }
    case BLIS_DOUBLE :
    {
      double*  alphap = (double*) bli_obj_buffer( alpha );
      double*  ap     = (double*) bli_obj_buffer( a );
      double*  xp     = (double*) bli_obj_buffer( x );
      double*  betap  = (double*) bli_obj_buffer( beta );
      double*  yp     = (double*) bli_obj_buffer( y );
      cblas_dsymv(cblas_order, cblas_uplo, m, *alphap, ap, lda, xp, incx,
                                                        *betap, yp, incy);
      break;
    }
    case BLIS_SCOMPLEX :
    {
      scomplex*  alphap = (scomplex*) bli_obj_buffer( alpha );
      scomplex*  ap     = (scomplex*) bli_obj_buffer( a );
      scomplex*  xp     = (scomplex*) bli_obj_buffer( x );
      scomplex*  betap  = (scomplex*) bli_obj_buffer( beta );
      scomplex*  yp     = (scomplex*) bli_obj_buffer( y );
      cblas_chemv(cblas_order, cblas_uplo, m, alphap, ap, lda, xp, incx,
                                                        betap, yp, incy);
      break;
    }
    case BLIS_DCOMPLEX :
    {
      dcomplex*  alphap = (dcomplex*) bli_obj_buffer( alpha );
      dcomplex*  ap     = (dcomplex*) bli_obj_buffer( a );
      dcomplex*  xp     = (dcomplex*) bli_obj_buffer( x );
      dcomplex*  betap  = (dcomplex*) bli_obj_buffer( beta );
      dcomplex*  yp     = (dcomplex*) bli_obj_buffer( y );
      cblas_zhemv(cblas_order, cblas_uplo, m, alphap, ap, lda, xp, incx,
                                                        betap, yp, incy);
      break;
    }
    default:
      bli_check_error_code( BLIS_INVALID_DATATYPE );
  }
}

void blas_symv(
  f77_char f77_uploa,
  f77_int m,
  obj_t*  alpha,
  obj_t*  a,
  f77_int lda,
  obj_t*  x,
  f77_int incx,
  obj_t*  beta,
  obj_t*  y,
  f77_int incy,
  num_t   dt
){

  switch( dt )  {
    case BLIS_FLOAT :
    {
      float*   alphap = (float*) bli_obj_buffer( alpha );
      float*   ap     = (float*) bli_obj_buffer( a );
      float*   xp     = (float*) bli_obj_buffer( x );
      float*   betap  = (float*) bli_obj_buffer( beta );
      float*   yp     = (float*) bli_obj_buffer( y );
      ssymv_(&f77_uploa, &m, alphap, ap, (f77_int*)&lda, xp, &incx,
                                                        betap, yp, &incy);
      break;
    }
    case BLIS_DOUBLE :
    {
      double*  alphap = (double*) bli_obj_buffer( alpha );
      double*  ap     = (double*) bli_obj_buffer( a );
      double*  xp     = (double*) bli_obj_buffer( x );
      double*  betap  = (double*) bli_obj_buffer( beta );
      double*  yp     = (double*) bli_obj_buffer( y );
      dsymv_(&f77_uploa, &m, alphap, ap, (f77_int*)&lda, xp, &incx,
                                                        betap, yp, &incy);
      break;
    }
    case BLIS_SCOMPLEX :
    {
      scomplex*  alphap = (scomplex*) bli_obj_buffer( alpha );
      scomplex*  ap     = (scomplex*) bli_obj_buffer( a );
      scomplex*  xp     = (scomplex*) bli_obj_buffer( x );
      scomplex*  betap  = (scomplex*) bli_obj_buffer( beta );
      scomplex*  yp     = (scomplex*) bli_obj_buffer( y );
      chemv_(&f77_uploa, &m, alphap, ap, (f77_int*)&lda, xp, &incx,
                                                        betap, yp, &incy);
      break;
    }
    case BLIS_DCOMPLEX :
    {
      dcomplex*  alphap = (dcomplex*) bli_obj_buffer( alpha );
      dcomplex*  ap     = (dcomplex*) bli_obj_buffer( a );
      dcomplex*  xp     = (dcomplex*) bli_obj_buffer( x );
      dcomplex*  betap  = (dcomplex*) bli_obj_buffer( beta );
      dcomplex*  yp     = (dcomplex*) bli_obj_buffer( y );
      zhemv_(&f77_uploa, &m, alphap, ap, (f77_int*)&lda, xp, &incx,
                                                        betap, yp, &incy);
      break;
    }
    default:
      bli_check_error_code( BLIS_INVALID_DATATYPE );
  }
}

void libblis_api_symv(
  test_params_t* params,
  iface_t        iface,
  obj_t*         alpha,
  obj_t*         a,
  obj_t*         x,
  obj_t*         beta,
  obj_t*         y
){

  if(params->api == API_BLIS) {
    libblis_test_symv_impl( iface, alpha, a, x, beta, y);
  }
  else { /*CLBAS  || BLAS */
    num_t   dt    = bli_obj_dt( a );
    uplo_t  uploa = bli_obj_uplo( a );
    f77_int mm    = bli_obj_length( a );
    f77_int incx  = bli_obj_vector_inc( x );
    f77_int incy  = bli_obj_vector_inc( y );
    f77_int lda ;

   if ( bli_obj_row_stride( a ) == 1 ) {
      lda    = bli_obj_col_stride( a );
    } else {
      lda    = bli_obj_row_stride( a );
    }

    if(params->ldf == 1) {
      lda = lda + params->ld[0];
    }

    if( bli_obj_has_conj(a) ) {
       conjugate_tensor(a, dt);
    }
    if( bli_obj_has_conj(x) ) {
       conjugate_tensor(x, dt);
    }

    if(params->api == API_CBLAS) {
     	cblas_symv(uploa, mm, alpha, a, lda, x, incx, beta, y, incy, dt );
    }
    else { /**/
      f77_char f77_uploa;
      if ( bli_obj_row_stride( a ) == 1 ){
        bli_param_map_blis_to_netlib_uplo( uploa, &f77_uploa );
        blas_symv(f77_uploa, mm, alpha, a, lda, x, incx, beta, y, incy, dt );
      }
      else {
        if( uploa == BLIS_UPPER)
          uploa = BLIS_LOWER;
        else if(uploa == BLIS_LOWER)
          uploa = BLIS_UPPER;

        bli_param_map_blis_to_netlib_uplo( uploa, &f77_uploa );
        blas_symv(f77_uploa, mm, alpha, a, lda, x, incx, beta, y, incy, dt );
      }
    }
  }
  return ;
}

double libblis_ref_symv(
  test_params_t* params,
  obj_t*         alpha,
  obj_t*         a,
  obj_t*         x,
  obj_t*         beta,
  obj_t*         y,
  obj_t*         y_orig
){
  double resid = 0.0;

  if((params->bitextf == 0) && (params->oruflw == BLIS_DEFAULT)) {
    resid = libblis_test_symv_check( params, alpha, a, x, beta, y, y_orig);
  }
  else {
    if(params->oruflw == BLIS_DEFAULT) {
      resid = libblis_test_isymv_check( params, alpha, a, x, beta, y, y_orig);
    }
    else {
      resid = libblis_test_vector_check(params, y);
    }
  }
  return resid;
}

double libblis_test_bitrp_symv(
  test_params_t* params,
  iface_t        iface,
  obj_t*         alpha,
  obj_t*         a,
  obj_t*         x,
  obj_t*         beta,
  obj_t*         y,
  obj_t*         y_orig,
  obj_t*         r,
  num_t          dt
) {
  double resid = 0.0;
	 unsigned int n_repeats = params->n_repeats;
	 unsigned int i;

  for(i = 0; i < n_repeats; i++) {
    bli_copyv( y_orig, r );
    libblis_test_symv_impl( iface, alpha, a, x, beta, r );
    resid = libblis_test_bitrp_vector(y, r, dt);
  }
  return resid;
}

double libblis_test_op_symv (
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
  uplo_t       uploa;
  conj_t       conja;
  conj_t       conjx;
  obj_t        alpha, a, x, beta, y;
  obj_t        y_save;
  double       resid = 0.0;
  obj_t        aa, xx;

  // Use the datatype of the first char in the datatype combination string.
  bli_param_map_char_to_blis_dt( dc_str[0], &datatype );

  // Map the dimension specifier to an actual dimension.
  m = dim->m;

  // Map parameter characters to BLIS constants.
  bli_param_map_char_to_blis_uplo( pc_str[0], &uploa );
  bli_param_map_char_to_blis_conj( pc_str[1], &conja );
  bli_param_map_char_to_blis_conj( pc_str[2], &conjx );

  // Create test scalars.
  bli_obj_scalar_init_detached( datatype, &alpha );
  bli_obj_scalar_init_detached( datatype, &beta );

  // Create test operands (vectors and/or matrices).
  libblis_test_mobj_create( params, datatype, BLIS_NO_TRANSPOSE,
                            sc_str[0], m, m, &a );
  libblis_test_mobj_create( params, datatype, BLIS_NO_TRANSPOSE,
                            sc_str[0], m, m, &aa );
  libblis_test_vobj_create( params, datatype,
                            sc_str[1], m,    &x );
  libblis_test_vobj_create( params, datatype,
                            sc_str[1], m,    &xx );
  libblis_test_vobj_create( params, datatype,
                            sc_str[2], m,    &y );
  libblis_test_vobj_create( params, datatype,
                            sc_str[2], m,    &y_save );

  // Set the structure and uplo properties of A.
  bli_obj_set_struc( BLIS_SYMMETRIC, &a );
  bli_obj_set_uplo( uploa, &a );

  bli_obj_set_struc( BLIS_SYMMETRIC, &aa );
  bli_obj_set_uplo( uploa, &aa );

  // Set alpha and beta.
  if((params->bitextf == 0) && (params->oruflw == BLIS_DEFAULT)) {
    if ( bli_obj_is_real( &y ) ){
      bli_setsc(  1.0,  0.0, &alpha );
      bli_setsc( -1.0,  0.0, &beta );
    }
    else{
      bli_setsc(  0.5,  0.5, &alpha );
      bli_setsc( -0.5,  0.5, &beta );
    }
    libblis_test_mobj_randomize( params, TRUE, &a );
    libblis_test_vobj_randomize( params, TRUE, &x );
    libblis_test_vobj_randomize( params, TRUE, &y );
  }
  else {
    int32_t xx = (int32_t) 1.0;
    int32_t yy = (int32_t)-1.0;
    if ( bli_obj_is_real( &y ) ){
      bli_setsc( xx,  0.0, &alpha );
      bli_setsc( yy,  0.0, &beta );
    }
    else{
      xx = (int32_t)(xx/0.8);
      yy = (int32_t)(yy/1.5);
      bli_setsc( xx, (xx+yy), &alpha );
      bli_setsc( yy, (xx-yy), &beta );
    }
    libblis_test_mobj_irandomize( params, &a );
    libblis_test_vobj_irandomize( params, &x );
    libblis_test_vobj_irandomize( params, &y );
  }

  // Randomize A, make it densely symmetric, and zero the unstored triangle
  // to ensure the implementation reads only from the stored region.
  bli_mksymm( &a );
  bli_mktrim( &a );

  // Randomize x and y, and save y.
  bli_copyv( &y, &y_save );

  bli_copym( &a, &aa );
  bli_copyv( &x, &xx );

  // Apply the remaining parameters.
  bli_obj_set_conj( conja, &a );
  bli_obj_set_conj( conjx, &x );

  bli_obj_set_conj( conja, &aa );
  bli_obj_set_conj( conjx, &xx );

  libblis_api_symv(params, iface, &alpha, &aa, &xx, &beta, &y );

#ifndef __GTEST_VALGRIND_TEST__
  if(params->bitrp) {
    obj_t r;

    libblis_test_vobj_create( params, datatype, sc_str[2], m, &r );

    resid = libblis_test_bitrp_symv( params, iface, &alpha, &a, &x,
                                   &beta, &y, &y_save, &r, datatype);

    bli_obj_free( &r );
  }
  else {
    resid = libblis_ref_symv( params, &alpha, &a, &x, &beta, &y, &y_save);
  }
#endif

  // Zero out performance and residual if output vector is empty.
  libblis_test_check_empty_problem( &y, &resid );

  // Free the test objects.
  libblis_test_obj_free( &a );
  libblis_test_obj_free( &aa );
  libblis_test_obj_free( &x );
  libblis_test_obj_free( &xx );
  libblis_test_obj_free( &y );
  libblis_test_obj_free( &y_save );

  return abs(resid);
}

void libblis_test_symv_impl(
  iface_t   iface,
  obj_t*    alpha,
  obj_t*    a,
  obj_t*    x,
  obj_t*    beta,
  obj_t*    y
){
  switch ( iface ){
    case BLIS_TEST_SEQ_FRONT_END:
      bli_symv( alpha, a, x, beta, y );
      break;

    default:
       libblis_test_printf_error( "Invalid interface type.\n" );
  }
}

double libblis_test_symv_check(
  test_params_t* params,
  obj_t*         alpha,
  obj_t*         a,
  obj_t*         x,
  obj_t*         beta,
  obj_t*         y,
  obj_t*         y_orig
){
  num_t  dt      = bli_obj_dt( y );
  num_t  dt_real = bli_obj_dt_proj_to_real( y );

  dim_t  m       = bli_obj_vector_dim( y );

  obj_t  v;
  obj_t  norm;

  double junk;
  double resid = 0.0;
  //
  // Pre-conditions:
  // - a is randomized and symmetric.
  // - x is randomized.
  // - y_orig is randomized.
  // Note:
  // - alpha and beta should have non-zero imaginary components in the
  //   complex cases in order to more fully exercise the implementation.
  //
  // Under these conditions, we assume that the implementation for
  //
  //   y := beta * y_orig + alpha * conja(A) * conjx(x)
  //
  // is functioning correctly if
  //
  //   normfv( y - v )
  //
  // is negligible, where
  //
  //   v = beta * y_orig + alpha * conja(A_dense) * x
  //

  bli_obj_scalar_init_detached( dt_real, &norm );

  bli_obj_create( dt, m, 1, 0, 0, &v );

  bli_copyv( y_orig, &v );

  bli_mksymm( a );
  bli_obj_set_struc( BLIS_GENERAL, a );
  bli_obj_set_uplo( BLIS_DENSE, a );

  bli_gemv( alpha, a, x, beta, &v );

  bli_subv( &v, y );
  bli_normfv( y, &norm );
  bli_getsc( &norm, &resid, &junk );

  bli_obj_free( &v );

  return resid;
}