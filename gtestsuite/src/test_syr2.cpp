#include "blis_test.h"
#include "blis_utils.h"
#include "test_syr2.h"

// Local prototypes.
void libblis_test_syr2_deps(
  thread_data_t* tdata,
  test_params_t* params,
  test_op_t*     op
);

void libblis_test_syr2_impl(
  iface_t   iface,
  obj_t*    alpha,
  obj_t*    x,
  obj_t*    y,
  obj_t*    a
);

double libblis_test_syr2_check(
  test_params_t* params,
  obj_t*         alpha,
  obj_t*         x,
  obj_t*         y,
  obj_t*         a,
  obj_t*         a_orig
);

void cblas_syr2(
  uplo_t  uploa,
  f77_int m,
  obj_t*  alpha,
  obj_t*  x,
  f77_int incx,
  obj_t*  y,
  f77_int incy,
  obj_t*  a,
  f77_int lda,
  num_t   dt
){
  enum CBLAS_UPLO  cblas_uplo;
  enum CBLAS_ORDER cblas_order;
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
      float*   yp     = (float*) bli_obj_buffer( y );
      cblas_ssyr2(cblas_order, cblas_uplo, m, *alphap, xp, incx,
                                                       yp, incy, ap, lda);
      break;
    }
    case BLIS_DOUBLE :
    {
      double*  alphap = (double*) bli_obj_buffer( alpha );
      double*  ap     = (double*) bli_obj_buffer( a );
      double*  xp     = (double*) bli_obj_buffer( x );
      double*  yp     = (double*) bli_obj_buffer( y );
      cblas_dsyr2(cblas_order, cblas_uplo, m, *alphap, xp, incx,
                                                       yp, incy, ap, lda);
      break;
    }
    case BLIS_SCOMPLEX :
    {
      scomplex*  alphap = (scomplex*) bli_obj_buffer( alpha );
      scomplex*  ap     = (scomplex*) bli_obj_buffer( a );
      scomplex*  xp     = (scomplex*) bli_obj_buffer( x );
      scomplex*  yp     = (scomplex*) bli_obj_buffer( y );
      cblas_cher2(cblas_order, cblas_uplo, m, alphap, xp, incx,
                                                       yp, incy, ap, lda);
      break;
    }
    case BLIS_DCOMPLEX :
    {
      dcomplex*  alphap = (dcomplex*) bli_obj_buffer( alpha );
      dcomplex*  ap     = (dcomplex*) bli_obj_buffer( a );
      dcomplex*  xp     = (dcomplex*) bli_obj_buffer( x );
      dcomplex*  yp     = (dcomplex*) bli_obj_buffer( y );
      cblas_zher2(cblas_order, cblas_uplo, m, alphap, xp, incx,
                                                       yp, incy, ap, lda);
      break;
    }
    default:
      bli_check_error_code( BLIS_INVALID_DATATYPE );
  }
}

void blas_syr2(
  f77_char f77_uploa,
  f77_int m,
  obj_t*  alpha,
  obj_t*  x,
  f77_int incx,
  obj_t*  y,
  f77_int incy,
  obj_t*  a,
  f77_int lda,
  num_t   dt
){

  switch( dt )  {
    case BLIS_FLOAT :
    {
      float*   alphap = (float*) bli_obj_buffer( alpha );
      float*   ap     = (float*) bli_obj_buffer( a );
      float*   xp     = (float*) bli_obj_buffer( x );
      float*   yp     = (float*) bli_obj_buffer( y );
      ssyr2_(&f77_uploa, &m, alphap, xp, &incx, yp, &incy, ap, (f77_int*)&lda);
      break;
    }
    case BLIS_DOUBLE :
    {
      double*  alphap = (double*) bli_obj_buffer( alpha );
      double*  ap     = (double*) bli_obj_buffer( a );
      double*  xp     = (double*) bli_obj_buffer( x );
      double*   yp    = (double*) bli_obj_buffer( y );
      dsyr2_(&f77_uploa, &m, alphap, xp, &incx, yp, &incy, ap, (f77_int*)&lda);
      break;
    }
    case BLIS_SCOMPLEX :
    {
      scomplex*  alphap = (scomplex*) bli_obj_buffer( alpha );
      scomplex*  ap     = (scomplex*) bli_obj_buffer( a );
      scomplex*  xp     = (scomplex*) bli_obj_buffer( x );
      scomplex*  yp     = (scomplex*) bli_obj_buffer( y );
      cher2_(&f77_uploa, &m, alphap, xp, &incx, yp, &incy, ap, (f77_int*)&lda);
      break;
    }
    case BLIS_DCOMPLEX :
    {
      dcomplex*  alphap = (dcomplex*) bli_obj_buffer( alpha );
      dcomplex*  ap     = (dcomplex*) bli_obj_buffer( a );
      dcomplex*  xp     = (dcomplex*) bli_obj_buffer( x );
      dcomplex*  yp     = (dcomplex*) bli_obj_buffer( y );
      zher2_(&f77_uploa, &m, alphap, xp, &incx, yp, &incy, ap, (f77_int*)&lda);
      break;
    }
    default:
      bli_check_error_code( BLIS_INVALID_DATATYPE );
  }
}

void libblis_api_syr2(
  test_params_t* params,
  iface_t        iface,
  obj_t*         alpha,
  obj_t*         x,
  obj_t*         y,
  obj_t*         a,
  num_t          dt
){

  if(params->api == API_BLIS) {
    libblis_test_syr2_impl( iface, alpha, x, y, a );
  }
  else { /*CLBAS  || BLAS */
    uplo_t  uploa = bli_obj_uplo( a );
    f77_int  mm   = bli_obj_length( a );
    f77_int  incx = bli_obj_vector_inc( x );
    f77_int  incy = bli_obj_vector_inc( y );
    f77_int  lda ;

   if ( bli_obj_row_stride( a ) == 1 ) {
      lda    = bli_obj_col_stride( a );
    } else {
      lda    = bli_obj_row_stride( a );
    }

    if(params->ldf == 1) {
      lda = lda + params->ld[0];
    }

    if(params->api == API_CBLAS) {
     	cblas_syr2(uploa, mm, alpha, x, incx, y, incy, a, lda, dt );
    }
    else { /**/
      f77_char f77_uploa;
      if ( bli_obj_row_stride( a ) == 1 ){
        bli_param_map_blis_to_netlib_uplo( uploa, &f77_uploa );
        blas_syr2(f77_uploa, mm, alpha, x, incx, y, incy, a, lda, dt );
      }
      else {
        if( uploa == BLIS_UPPER)
          uploa = BLIS_LOWER;
        else if(uploa == BLIS_LOWER)
          uploa = BLIS_UPPER;

        conjugate_tensor(x, dt);
        conjugate_tensor(y, dt);
        bli_param_map_blis_to_netlib_uplo( uploa, &f77_uploa );
        blas_syr2(f77_uploa, mm, alpha, x, incx, y, incy, a, lda, dt );
      }
    }
  }
  return ;
}

double libblis_ref_syr2(
  test_params_t* params,
  obj_t*         alpha,
  obj_t*         x,
  obj_t*         y,
  obj_t*         a,
  obj_t*         a_orig
){
  double resid = 0.0;

  if((params->bitextf == 0) && (params->oruflw == BLIS_DEFAULT)) {
    resid = libblis_test_syr2_check( params, alpha, x, y, a, a_orig);
  }
  else {
    if(params->oruflw == BLIS_DEFAULT) {
      resid = libblis_test_isyr2_check( params, alpha, x, y, a, a_orig);
    }
    else {
      resid = libblis_test_matrix_check(params, a);
    }
  }
  return resid;
}

double libblis_test_bitrp_syr2(
  test_params_t* params,
  iface_t        iface,
  obj_t*         alpha,
  obj_t*         x,
  obj_t*         y,
  obj_t*         a,
  obj_t*         a_orig,
  obj_t*         r,
  num_t          dt
) {
  double resid = 0.0;
	 unsigned int n_repeats = params->n_repeats;
	 unsigned int i;

  for(i = 0; i < n_repeats; i++) {
    bli_copym( a_orig, r );
    bli_mksymm( r );
    bli_mktrim( r );
    libblis_test_syr2_impl( iface, alpha, x, y, r);
    resid = libblis_test_bitrp_matrix(a, r, dt);
  }
  return resid;
}

double libblis_test_op_syr2 (
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
  uplo_t       uploa;
  conj_t       conjx, conjy;
  obj_t        alpha, x, y, a;
  obj_t        a_save;
  double       resid = 0.0;
  obj_t        xx, yy;

  // Use the datatype of the first char in the datatype combination string.
  bli_param_map_char_to_blis_dt( dc_str[0], &datatype );

  // Map the dimension specifier to an actual dimension.
  m = dim->m;

  // Map parameter characters to BLIS constants.
  bli_param_map_char_to_blis_uplo( pc_str[0], &uploa );
  bli_param_map_char_to_blis_conj( pc_str[1], &conjx );
  bli_param_map_char_to_blis_conj( pc_str[2], &conjy );

  // Create test scalars.
  bli_obj_scalar_init_detached( datatype, &alpha );

  // Create test operands (vectors and/or matrices).
  libblis_test_vobj_create( params, datatype,
                          sc_str[0], m,    &x );
  libblis_test_vobj_create( params, datatype,
                          sc_str[0], m,    &xx );
  libblis_test_vobj_create( params, datatype,
                          sc_str[1], m,    &y );
  libblis_test_vobj_create( params, datatype,
                          sc_str[1], m,    &yy );
  libblis_test_mobj_create( params, datatype, BLIS_NO_TRANSPOSE,
                          sc_str[2], m, m, &a );
  libblis_test_mobj_create( params, datatype, BLIS_NO_TRANSPOSE,
                          sc_str[2], m, m, &a_save );

  // Set the structure and uplo properties of A.
  bli_obj_set_struc( BLIS_SYMMETRIC, &a );
  bli_obj_set_uplo( uploa, &a );

  // Set alpha.
  if((params->bitextf == 0) && (params->oruflw == BLIS_DEFAULT)) {
    if ( bli_obj_is_real( &x ) ) {
      bli_setsc( alpv.real,  0.0, &alpha );
    }
    else {
      bli_setsc( alpv.real, alpv.imag, &alpha );
    }
    // Randomize x and y.
    libblis_test_vobj_randomize( params, TRUE, &x );
    libblis_test_vobj_randomize( params, TRUE, &y );
    libblis_test_mobj_randomize( params, TRUE, &a );
  }
  else{
    int32_t xx = (int32_t)alpv.real;
    if ( bli_obj_is_real( &x ) ) {
      bli_setsc( (double)xx,  0.0, &alpha );
    }
    else {
      int32_t ax = (int32_t)(xx/0.8);
      bli_setsc( (double)xx, (double)ax, &alpha );
    }

    libblis_test_vobj_irandomize( params, &x );
    libblis_test_vobj_irandomize( params, &y );
    libblis_test_mobj_irandomize( params, &a );
  }

  // Randomize A, make it densely symmetric, and zero the unstored triangle
  // to ensure the implementation is reads only from the stored region.
  bli_mksymm( &a );
  bli_mktrim( &a );

  // Save A and set its structure and uplo properties.
  bli_obj_set_struc( BLIS_SYMMETRIC, &a_save );
  bli_obj_set_uplo( uploa, &a_save );
  bli_copym( &a, &a_save );
  bli_mksymm( &a_save );
  bli_mktrim( &a_save );

  // Apply the remaining parameters.
  bli_obj_set_conj( conjx, &x );
  bli_obj_set_conj( conjy, &y );

  bli_copyv( &x, &xx );
  bli_copyv( &y, &yy );

  libblis_api_syr2( params, iface, &alpha, &xx, &yy, &a, datatype);

#ifndef __GTEST_VALGRIND_TEST__
  if(params->bitrp) {
    obj_t r;

    libblis_test_mobj_create( params, datatype, BLIS_NO_TRANSPOSE,
                                                 sc_str[2], m, m, &r );
    bli_obj_set_struc( BLIS_SYMMETRIC, &r );
    bli_obj_set_uplo( uploa, &r );

    resid = libblis_test_bitrp_syr2( params, iface, &alpha, &x, &y,
                                              &a, &a_save, &r, datatype);
    bli_obj_free( &r );
  }
  else {
    resid = libblis_ref_syr2( params, &alpha, &x, &y, &a, &a_save);
  }
#endif
  // Zero out performance and residual if output matrix is empty.
  libblis_test_check_empty_problem( &a, &resid );

  // Free the test objects.
  libblis_test_obj_free( &x );
  libblis_test_obj_free( &xx );
  libblis_test_obj_free( &y );
  libblis_test_obj_free( &yy );
  libblis_test_obj_free( &a );
  libblis_test_obj_free( &a_save );

  return abs(resid);
}

void libblis_test_syr2_impl(
  iface_t   iface,
  obj_t*    alpha,
  obj_t*    x,
  obj_t*    y,
  obj_t*    a
){
  switch ( iface ){
    case BLIS_TEST_SEQ_FRONT_END:
      bli_syr2( alpha, x, y, a );
      break;

    default:
      libblis_test_printf_error( "Invalid interface type.\n" );
  }
}

double libblis_test_syr2_check(
  test_params_t* params,
  obj_t*         alpha,
  obj_t*         x,
  obj_t*         y,
  obj_t*         a,
  obj_t*         a_orig
){
  num_t  dt      = bli_obj_dt( a );
  num_t  dt_real = bli_obj_dt_proj_to_real( a );

  dim_t  m_a     = bli_obj_length( a );

  obj_t  xt, yt;
  obj_t  t, v, w1, w2;
  obj_t  rho, norm;

  double junk;
  double resid = 0.0;
  //
  // Pre-conditions:
  // - x is randomized.
  // - y is randomized.
  // - a is randomized and symmetric.
  // Note:
  // - alpha should have a non-zero imaginary component in the
  //   complex cases in order to more fully exercise the implementation.
  //
  // Under these conditions, we assume that the implementation for
  //
  //   A := A_orig + alpha * conjx(x) * conjy(y)^T + alpha * conjy(y) * conjx(x)^T
  //
  // is functioning correctly if
  //
  //   normfv( v - w )
  //
  // is negligible, where
  //
  //   v = A * t
  //   w = ( A_orig + alpha * conjx(x) * conjy(y)^T + alpha * conjy(y) * conjx(x)^T ) * t
  //     = A_orig * t + alpha * conjx(x) * conjy(y)^T * t + alpha * conjy(y) * conjx(x)^T * t
  //     = A_orig * t + alpha * conjx(x) * conjy(y)^T * t + alpha * conjy(y) * rho
  //     = A_orig * t + alpha * conjx(x) * conjy(y)^T * t + w1
  //     = A_orig * t + alpha * conjx(x) * rho            + w1
  //     = A_orig * t + w2                                + w1
  //

  bli_mksymm( a );
  bli_mksymm( a_orig );
  bli_obj_set_struc( BLIS_GENERAL, a );
  bli_obj_set_struc( BLIS_GENERAL, a_orig );
  bli_obj_set_uplo( BLIS_DENSE, a );
  bli_obj_set_uplo( BLIS_DENSE, a_orig );

  bli_obj_scalar_init_detached( dt,      &rho );
  bli_obj_scalar_init_detached( dt_real, &norm );

  bli_obj_create( dt, m_a, 1, 0, 0, &t );
  bli_obj_create( dt, m_a, 1, 0, 0, &v );
  bli_obj_create( dt, m_a, 1, 0, 0, &w1 );
  bli_obj_create( dt, m_a, 1, 0, 0, &w2 );

  bli_obj_alias_to( x, &xt );
  bli_obj_alias_to( y, &yt );

  libblis_test_vobj_randomize( params, TRUE, &t );

  bli_gemv( &BLIS_ONE, a, &t, &BLIS_ZERO, &v );

  bli_dotv( &xt, &t, &rho );
  bli_mulsc( alpha, &rho );
  bli_scal2v( &rho, y, &w1 );

  bli_dotv( &yt, &t, &rho );
  bli_mulsc( alpha, &rho );
  bli_scal2v( &rho, x, &w2 );

  bli_addv( &w2, &w1 );

  bli_gemv( &BLIS_ONE, a_orig, &t, &BLIS_ONE, &w1 );

  bli_subv( &w1, &v );
  bli_normfv( &v, &norm );
  bli_getsc( &norm, &resid, &junk );

  bli_obj_free( &t );
  bli_obj_free( &v );
  bli_obj_free( &w1 );
  bli_obj_free( &w2 );

  return resid;
}