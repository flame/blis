#include "blis_test.h"
#include "blis_utils.h"
#include "test_gemv.h"

// Local prototypes.
void libblis_test_gemv_deps (
  thread_data_t* tdata,
  test_params_t* params,
  test_op_t*     op
);

void libblis_test_gemv_impl (
  iface_t   iface,
  obj_t*    alpha,
  obj_t*    a,
  obj_t*    x,
  obj_t*    beta,
  obj_t*    y
);

double libblis_test_gemv_check (
  test_params_t* params,
  obj_t*         kappa,
  obj_t*         alpha,
  obj_t*         a,
  obj_t*         x,
  obj_t*         beta,
  obj_t*         y,
  obj_t*         y_orig
);

void cblas_gemv(
  f77_int m,
  f77_int n,
  obj_t*  alpha,
  obj_t*  a,
  f77_int lda,
  obj_t*  x,
  f77_int incx,
  obj_t*  beta,
  obj_t*  y,
  f77_int incy,
  trans_t transa,
  num_t   dt
){
  enum CBLAS_ORDER     cblas_order;
  enum CBLAS_TRANSPOSE cblas_trans;

  if ( bli_obj_row_stride( a ) == 1 )
    cblas_order = CblasColMajor;
  else
    cblas_order = CblasRowMajor;

  if( bli_is_trans(transa ) )
    cblas_trans = CblasTrans;
  else if( bli_is_conjtrans(transa ) )
    cblas_trans = CblasConjTrans;
  else
    cblas_trans = CblasNoTrans;

  switch( dt )  {
    case BLIS_FLOAT :
    {
      float*   alphap = (float*) bli_obj_buffer( alpha );
      float*   ap     = (float*) bli_obj_buffer( a );
      float*   xp     = (float*) bli_obj_buffer( x );
      float*   betap  = (float*) bli_obj_buffer( beta );
      float*   yp     = (float*) bli_obj_buffer( y );
      cblas_sgemv( cblas_order, cblas_trans, m, n,
                      *alphap, ap, lda, xp, incx, *betap, yp, incy );
      break;
    }
    case BLIS_DOUBLE :
    {
      double*  alphap = (double*) bli_obj_buffer( alpha );
      double*  ap     = (double*) bli_obj_buffer( a );
      double*  xp     = (double*) bli_obj_buffer( x );
      double*  betap  = (double*) bli_obj_buffer( beta );
      double*  yp     = (double*) bli_obj_buffer( y );
      cblas_dgemv( cblas_order, cblas_trans, m, n,
                      *alphap, ap, lda, xp, incx, *betap, yp, incy );
      break;
    }
    case BLIS_SCOMPLEX :
    {
      scomplex*  alphap = (scomplex*) bli_obj_buffer( alpha );
      scomplex*  ap     = (scomplex*) bli_obj_buffer( a );
      scomplex*  xp     = (scomplex*) bli_obj_buffer( x );
      scomplex*  betap  = (scomplex*) bli_obj_buffer( beta );
      scomplex*  yp     = (scomplex*) bli_obj_buffer( y );
      cblas_cgemv( cblas_order, cblas_trans, m, n,
                        alphap, ap, lda, xp, incx, betap, yp, incy );

      break;
    }
    case BLIS_DCOMPLEX :
    {
      dcomplex*  alphap = (dcomplex*) bli_obj_buffer( alpha );
      dcomplex*  ap     = (dcomplex*) bli_obj_buffer( a );
      dcomplex*  xp     = (dcomplex*) bli_obj_buffer( x );
      dcomplex*  betap  = (dcomplex*) bli_obj_buffer( beta );
      dcomplex*  yp     = (dcomplex*) bli_obj_buffer( y );
      cblas_zgemv( cblas_order, cblas_trans, m, n,
                        alphap, ap, lda, xp, incx, betap, yp, incy );
      break;
    }
    default :
      bli_check_error_code( BLIS_INVALID_DATATYPE );
  }
}

void gemv_(f77_char f77_trans, f77_int m, f77_int n, scomplex* alpha,
scomplex* ap, f77_int lda, scomplex* xp, f77_int incx, scomplex* beta,
scomplex* yp, f77_int incy){
  cgemv_( &f77_trans, &m, &n, alpha, ap, &lda, xp, &incx, beta, yp, &incy);
}

void gemv_(f77_char f77_trans, f77_int m, f77_int n, dcomplex* alpha,
dcomplex* ap, f77_int lda, dcomplex* xp, f77_int incx, dcomplex* beta,
dcomplex* yp, f77_int incy){
  zgemv_( &f77_trans, &m, &n, alpha, ap, &lda, xp, &incx, beta, yp, &incy);
}

void blas_gemv(
  trans_t transa,
  f77_int  m,
  f77_int  n,
  obj_t*   alpha,
  obj_t*   a,
  f77_int  lda,
  obj_t*   x,
  f77_int  incx,
  obj_t*   beta,
  obj_t*   y,
  f77_int  incy,
  num_t    dt
){
  f77_char f77_trans;

  bli_param_map_blis_to_netlib_trans( transa, &f77_trans );

  switch( dt )  {
    case BLIS_FLOAT :
    {
      float*   alphap = (float*) bli_obj_buffer( alpha );
      float*   ap     = (float*) bli_obj_buffer( a );
      float*   xp     = (float*) bli_obj_buffer( x );
      float*   betap  = (float*) bli_obj_buffer( beta );
      float*   yp     = (float*) bli_obj_buffer( y );
      sgemv_( &f77_trans, &m, &n, alphap, ap, &lda, xp,
                                 &incx, betap, yp, &incy );
      break;
    }
    case BLIS_DOUBLE :
    {
      double*  alphap = (double*) bli_obj_buffer( alpha );
      double*  ap     = (double*) bli_obj_buffer( a );
      double*  xp     = (double*) bli_obj_buffer( x );
      double*  betap  = (double*) bli_obj_buffer( beta );
      double*  yp     = (double*) bli_obj_buffer( y );
      dgemv_( &f77_trans, &m, &n, alphap, ap, &lda, xp,
                                 &incx, betap, yp, &incy );
      break;
    }
    case BLIS_SCOMPLEX :
    {
      scomplex*  alphap = (scomplex*) bli_obj_buffer( alpha );
      scomplex*  ap     = (scomplex*) bli_obj_buffer( a );
      scomplex*  xp     = (scomplex*) bli_obj_buffer( x );
      scomplex*  betap  = (scomplex*) bli_obj_buffer( beta );
      scomplex*  yp     = (scomplex*) bli_obj_buffer( y );
      cgemv_( &f77_trans, &m, &n, alphap, ap, &lda, xp,
                                  &incx, betap, yp, &incy);
      break;
    }
    case BLIS_DCOMPLEX :
    {
      dcomplex*  alphap = (dcomplex*) bli_obj_buffer( alpha );
      dcomplex*  ap     = (dcomplex*) bli_obj_buffer( a );
      dcomplex*  xp     = (dcomplex*) bli_obj_buffer( x );
      dcomplex*  betap  = (dcomplex*) bli_obj_buffer( beta );
      dcomplex*  yp     = (dcomplex*) bli_obj_buffer( y );
      zgemv_( &f77_trans, &m, &n, alphap, ap, &lda, xp,
                                 &incx, betap, yp, &incy );
      break;
    }
    default :
      bli_check_error_code( BLIS_INVALID_DATATYPE );
  }
}

void libblis_api_gemv(
  test_params_t* params,
  iface_t iface,
  obj_t*  alpha,
  obj_t*  a,
  obj_t*  x,
  obj_t*  beta,
  obj_t*  y,
  num_t   dt
) {

  if(params->api == API_BLIS) {
    libblis_test_gemv_impl( iface, alpha, a, x, beta, y );
  }
  else { /*CLBAS  || BLAS */
    f77_int mm     = bli_obj_length( a );
    f77_int nn     = bli_obj_width( a );
    f77_int incx   = bli_obj_vector_inc( x );
    f77_int incy   = bli_obj_vector_inc( y );
    trans_t transa = bli_obj_conjtrans_status( a );
    f77_int lda ;

    if ( bli_obj_row_stride( a ) == 1 ) {
      lda    = bli_obj_col_stride( a );
    } else {
      lda    = bli_obj_row_stride( a );
    }

    if(params->ldf == 1) {
      lda = lda + params->ld[0];
    }

    if(bli_obj_has_notrans(a) && bli_obj_has_conj(a)) {
       conjugate_tensor(a, dt);
       transa = bli_obj_onlytrans_status( a );
    }

    if(bli_obj_has_conj(x)) {
       conjugate_tensor(x, dt);
    }

    if(params->api == API_CBLAS) {
      cblas_gemv(mm, nn, alpha, a, lda, x, incx, beta, y, incy, transa, dt);
    }
    else { /**/
      if( bli_obj_row_stride( a ) == 1 ) {
        blas_gemv(transa, mm, nn, alpha, a, lda, x, incx, beta, y, incy, dt);
      }
      else {
        blas_gemv(transa, nn, mm, alpha, a, lda, x, incx, beta, y, incy, dt);
      }
    }

    if(bli_obj_has_conj(x)) {
       conjugate_tensor(x, dt);
    }
  }
  return ;
}

double libblis_ref_gemv(
  test_params_t* params,
  obj_t*         kappa,
  obj_t*         alpha,
  obj_t*         a,
  obj_t*         x,
  obj_t*         beta,
  obj_t*         y,
  obj_t*         y_orig,
  num_t          dt
){
  double resid = 0.0;

  if((params->bitextf == 0) && (params->oruflw == BLIS_DEFAULT)) {
 	  // Perform checks.
    resid = libblis_test_gemv_check( params, kappa, alpha,
                                                a, x, beta, y, y_orig);
  }
  else {
   if(params->oruflw == BLIS_DEFAULT) {
      resid = libblis_test_igemv_check( alpha,
                                       a, x, beta, y, y_orig, dt);
    }
    else {
      resid = libblis_test_vector_check(params, y);
    }
  }
  return resid;
}

double libblis_test_bitrp_gemv(
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
    libblis_test_gemv_impl( iface, alpha, a, x, beta, r );
    resid = libblis_test_bitrp_vector(y, r, dt);
  }
  return resid;
}


double libblis_test_op_gemv (
  test_params_t* params,
  iface_t        iface,
  char*          dc_str,
  char*          pc_str,
  char*          sc_str,
  tensor_t*      dim,
  atom_t         alpv,
  atom_t         betv
){
  num_t   datatype;
  dim_t   m, n;
  trans_t transa;
  conj_t  conjx;
  obj_t   kappa;
  obj_t   alpha, a, x, beta, y;
  obj_t   y_save;
  double  resid = 0.0;
  obj_t   aa;

  // Use the datatype of the first char in the datatype combination string.
  bli_param_map_char_to_blis_dt( dc_str[0], &datatype );

  // Map the dimension specifier to actual dimensions.
  m = dim->m;
  n = dim->n;

  // Map parameter characters to BLIS constants.
  if(params->api != API_BLIS) {
    bli_param_map_char_to_blas_trans( pc_str[0], &transa );
  } else {
    bli_param_map_char_to_blis_trans( pc_str[0], &transa );
  }

  bli_param_map_char_to_blis_conj( pc_str[1], &conjx );

  // Create test scalars.
  bli_obj_scalar_init_detached( datatype, &kappa );
  bli_obj_scalar_init_detached( datatype, &alpha );
  bli_obj_scalar_init_detached( datatype, &beta );

  // Create test operands (vectors and/or matrices).
  libblis_test_mobj_create( params, datatype, transa,
                            sc_str[0], m, n, &a );
  libblis_test_mobj_create( params, datatype, transa,
                            sc_str[0], m, n, &aa );
  libblis_test_vobj_create( params, datatype,
                            sc_str[1], n,    &x );
  libblis_test_vobj_create( params, datatype,
                            sc_str[2], m,    &y );
  libblis_test_vobj_create( params, datatype,
                            sc_str[2], m,    &y_save );

  // Set alpha and beta.
  if((params->bitextf == 0) && (params->oruflw == BLIS_DEFAULT)) {
    if ( bli_obj_is_real( &y ) ) {
      bli_setsc(alpv.real, 0.0, &alpha);
      bli_setsc(betv.real, 0.0, &beta);
    }
    else {
      bli_setsc(alpv.real, (alpv.real/0.8), &alpha);
      bli_setsc(betv.real, (betv.real/1.2), &beta);
    }

    // Randomize x and y, and save y.
    libblis_test_vobj_randomize( params, TRUE, &x );
    libblis_test_vobj_randomize( params, TRUE, &y );
  }
  else{
    int32_t xx = (int32_t)alpv.real;
    int32_t yy = (int32_t)betv.real;
    if ( bli_obj_is_real( &y ) ) {
      bli_setsc( (double)xx,  0.0, &alpha );
      bli_setsc( (double)yy,  0.0, &beta );
    }
    else {
      // For syrk, both alpha and beta may be complex since, unlike herk,
      // C is symmetric in both the real and complex cases.
      int32_t ac = (int32_t)(xx/0.8);
      int32_t bc = (int32_t)(yy/1.0);
      bli_setsc( (double)xx, (double)ac, &alpha );
      bli_setsc( (double)yy, (double)bc, &beta );
    }

    // Randomize x and y, and save y.
    libblis_test_vobj_irandomize( params, &x );
    libblis_test_vobj_irandomize( params, &y );
  }
  // Initialize diagonal of matrix A.
  bli_setsc( 2.0, -1.0, &kappa );
  bli_setm( &BLIS_ZERO, &a );
  bli_setd( &kappa, &a );

  bli_copym( &a, &aa );

  // Apply the parameters.
  bli_obj_set_conjtrans( transa, &a );
  bli_obj_set_conj( conjx, &x );

  bli_copyv( &y, &y_save );

  bli_obj_set_conjtrans( transa, &aa );

  libblis_api_gemv(params, iface, &alpha, &aa, &x, &beta, &y, datatype );

#ifndef __GTEST_VALGRIND_TEST__
  if(params->bitrp) {
    obj_t r;

    libblis_test_vobj_create( params, datatype, sc_str[2], m, &r );

    resid = libblis_test_bitrp_gemv(params, iface, &alpha, &a, &x, &beta,
                                                 &y, &y_save, &r, datatype );
    bli_obj_free( &r );
  }
  else {
    resid = libblis_ref_gemv(params, &kappa, &alpha, &a, &x, &beta,
                                                 &y, &y_save, datatype );
  }
#endif

  // Zero out performance and residual if output vector is empty.
  libblis_test_check_empty_problem( &y, &resid );

  // Free the test objects.
  libblis_test_obj_free( &a );
  libblis_test_obj_free( &aa );
  libblis_test_obj_free( &x );
  libblis_test_obj_free( &y );
  libblis_test_obj_free( &y_save );

  return abs(resid);
}

void libblis_test_gemv_impl (
  iface_t   iface,
  obj_t*    alpha,
  obj_t*    a,
  obj_t*    x,
  obj_t*    beta,
  obj_t*    y
) {
	switch ( iface )	{
   case BLIS_TEST_SEQ_FRONT_END:
     bli_gemv( alpha, a, x, beta, y );
   break;

   default:
   libblis_test_printf_error( "Invalid interface type.\n" );
	}
}

double libblis_test_gemv_check (
  test_params_t* params,
  obj_t*         kappa,
  obj_t*         alpha,
  obj_t*         a,
  obj_t*         x,
  obj_t*         beta,
  obj_t*         y,
  obj_t*         y_orig
) {
  num_t  dt      = bli_obj_dt( y );
  num_t  dt_real = bli_obj_dt_proj_to_real( y );

  conj_t conja   = bli_obj_conj_status( a );

  dim_t  n_x     = bli_obj_vector_dim( x );
  dim_t  m_y     = bli_obj_vector_dim( y );

  dim_t  min_m_n = bli_min( m_y, n_x );

  obj_t  x_temp, y_temp;
  obj_t  kappac, norm;
  obj_t  xT_temp, yT_temp, yT;

  double junk;
  double resid = 0.0;

  //
  // Pre-conditions:
  // - a is initialized to kappa along the diagonal.
  // - x is randomized.
  // - y_orig is randomized.
  // Note:
  // - alpha, beta, and kappa should have non-zero imaginary components in
  //   the complex cases in order to more fully exercise the implementation.
  //
  // Under these conditions, we assume that the implementation for
  //
  //   y := beta * y_orig + alpha * transa(A) * conjx(x)
  //
  // is functioning correctly if
  //
  //   normfv( y - z )
  //
  // is negligible, where
  //
  //   z = beta * y_orig + alpha * conja(kappa) * x
  //

  bli_obj_scalar_init_detached_copy_of( dt, conja, kappa, &kappac );
  bli_obj_scalar_init_detached( dt_real, &norm );

  bli_obj_create( dt, n_x, 1, 0, 0, &x_temp );
  bli_obj_create( dt, m_y, 1, 0, 0, &y_temp );

  bli_copyv( x,      &x_temp );
  bli_copyv( y_orig, &y_temp );

  bli_acquire_vpart_f2b( BLIS_SUBPART1, 0, min_m_n,
                         &x_temp, &xT_temp );
  bli_acquire_vpart_f2b( BLIS_SUBPART1, 0, min_m_n,
                         &y_temp, &yT_temp );
  bli_acquire_vpart_f2b( BLIS_SUBPART1, 0, min_m_n,
                         y, &yT );

  bli_scalv( &kappac, &xT_temp );
  bli_scalv( beta, &yT_temp );
  bli_axpyv( alpha, &xT_temp, &yT_temp );

  bli_subv( &yT_temp, &yT );
  bli_normfv( &yT, &norm );
  bli_getsc( &norm, &resid, &junk );

  bli_obj_free( &x_temp );
  bli_obj_free( &y_temp );

  return resid;
}