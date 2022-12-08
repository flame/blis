#include "blis_test.h"
#include "blis_utils.h"
#include "test_ger.h"

// Local prototypes.
void libblis_test_ger_deps(
  thread_data_t* tdata,
  test_params_t* params,
  test_op_t*     op
);

void libblis_test_ger_impl(
  iface_t   iface,
  obj_t*    alpha,
  obj_t*    x,
  obj_t*    y,
  obj_t*    a
);

double libblis_test_ger_check(
  test_params_t* params,
  obj_t*         alpha,
  obj_t*         x,
  obj_t*         y,
  obj_t*         a,
  obj_t*         a_orig
);

void cblas_ger(
  f77_int m,
  f77_int n,
  obj_t*  alpha,
  obj_t*  x,
  f77_int incx,
  obj_t*  y,
  f77_int incy,
  obj_t*  a,
  f77_int lda,
  num_t   dt
){

  enum CBLAS_ORDER cblas_order;
  if ( bli_obj_row_stride( a ) == 1 )
    cblas_order = CblasColMajor;
  else
    cblas_order = CblasRowMajor;

  switch( dt )  {
    case BLIS_FLOAT :
    {
      float*   alphap = (float*) bli_obj_buffer( alpha );
      float*   ap     = (float*) bli_obj_buffer( a );
      float*   xp     = (float*) bli_obj_buffer( x );
      float*   yp     = (float*) bli_obj_buffer( y );
      cblas_sger(cblas_order, m, n, *alphap, xp, incx, yp, incy, ap, lda );
      break;
    }
    case BLIS_DOUBLE :
    {
      double*  alphap = (double*) bli_obj_buffer( alpha );
      double*  ap     = (double*) bli_obj_buffer( a );
      double*  xp     = (double*) bli_obj_buffer( x );
      double*  yp     = (double*) bli_obj_buffer( y );
      cblas_dger(cblas_order, m, n, *alphap, xp, incx, yp, incy, ap, lda );
      break;
    }
    case BLIS_SCOMPLEX :
    {
      scomplex*  alphap = (scomplex*) bli_obj_buffer( alpha );
      scomplex*  ap     = (scomplex*) bli_obj_buffer( a );
      scomplex*  xp     = (scomplex*) bli_obj_buffer( x );
      scomplex*  yp     = (scomplex*) bli_obj_buffer( y );
      if( bli_obj_has_conj(x) != bli_obj_has_conj(y)) {
        cblas_cgerc(cblas_order, m, n, alphap, xp, incx, yp, incy, ap, lda );
      }
      else{
        cblas_cgeru(cblas_order, m, n, alphap, xp, incx, yp, incy, ap, lda );
      }
      break;
    }
    case BLIS_DCOMPLEX :
    {
      dcomplex*  alphap = (dcomplex*) bli_obj_buffer( alpha );
      dcomplex*  ap     = (dcomplex*) bli_obj_buffer( a );
      dcomplex*  xp     = (dcomplex*) bli_obj_buffer( x );
      dcomplex*  yp     = (dcomplex*) bli_obj_buffer( y );
      if( bli_obj_has_conj(x) != bli_obj_has_conj(y)) {
        cblas_zgerc(cblas_order, m, n, alphap, xp, incx, yp, incy, ap, lda );
      }
      else{
        cblas_zgeru(cblas_order, m, n, alphap, xp, incx, yp, incy, ap, lda );
      }
      break;
    }
    default:
      bli_check_error_code( BLIS_INVALID_DATATYPE );
  }
}

void blas_ger(
  f77_int m,
  f77_int n,
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
      sger_(&m, &n, alphap, xp, &incx, yp, &incy, ap, &lda );
      break;
    }
    case BLIS_DOUBLE :
    {
      double*  alphap = (double*) bli_obj_buffer( alpha );
      double*  ap     = (double*) bli_obj_buffer( a );
      double*  xp     = (double*) bli_obj_buffer( x );
      double*  yp     = (double*) bli_obj_buffer( y );
      dger_(&m, &n, alphap, xp, &incx, yp, &incy, ap, &lda );
      break;
    }
    case BLIS_SCOMPLEX :
    {
      scomplex*  alphap = (scomplex*) bli_obj_buffer( alpha );
      scomplex*  ap     = (scomplex*) bli_obj_buffer( a );
      scomplex*  xp     = (scomplex*) bli_obj_buffer( x );
      scomplex*  yp     = (scomplex*) bli_obj_buffer( y );
      if( bli_obj_has_conj(x) != bli_obj_has_conj(y)) {
        cgerc_(&m, &n, alphap, xp, &incx, yp, &incy, ap, &lda );
      }
      else {
        cgeru_(&m, &n, alphap, xp, &incx, yp, &incy, ap, &lda );
      }
      break;
    }
    case BLIS_DCOMPLEX :
    {
      dcomplex*  alphap = (dcomplex*) bli_obj_buffer( alpha );
      dcomplex*  ap     = (dcomplex*) bli_obj_buffer( a );
      dcomplex*  xp     = (dcomplex*) bli_obj_buffer( x );
      dcomplex*  yp     = (dcomplex*) bli_obj_buffer( y );
      if( bli_obj_has_conj(x) != bli_obj_has_conj(y)) {
        zgerc_(&m, &n, alphap, xp, &incx, yp, &incy, ap, &lda );
      }
      else{
        zgeru_(&m, &n, alphap, xp, &incx, yp, &incy, ap, &lda );
      }
      break;
    }
    default :
      bli_check_error_code( BLIS_INVALID_DATATYPE );
  }
}

void libblis_api_ger(
  test_params_t* params,
  iface_t        iface,
  obj_t*         alpha,
  obj_t*         x,
  obj_t*         y,
  obj_t*         a,
  num_t          dt
){

  if(params->api == API_BLIS) {
    libblis_test_ger_impl( iface, alpha, x, y, a );
  }
  else { /*CLBAS  || BLAS */
    f77_int  mm   = bli_obj_length( a );
    f77_int  nn   = bli_obj_width( a );
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
     	cblas_ger(mm, nn, alpha, x, incx, y, incy, a, lda, dt );
    }
    else { /**/
      if ( bli_obj_row_stride( a ) == 1 ){
        blas_ger(mm, nn, alpha, x, incx, y, incy, a, lda, dt );
      }
      else {
        blas_ger(nn, mm, alpha, y, incy, x, incx, a, lda, dt );
      }
    }
  }
  return ;
}

double libblis_ref_ger(
  test_params_t* params,
  obj_t*         alpha,
  obj_t*         x,
  obj_t*         y,
  obj_t*         a,
  obj_t*         a_save
) {
  double resid = 0.0;

  if((params->bitextf == 0) && (params->oruflw == BLIS_DEFAULT)) {
 	  // Perform checks.
    resid = libblis_test_ger_check( params, alpha, x, y, a, a_save);
  }
  else {
    resid = libblis_test_iger_check( params, alpha, x, y, a, a_save);
  }
  return resid;
}

double libblis_test_bitrp_ger(
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
    bli_copyv( a_orig, r );
    libblis_test_ger_impl( iface, alpha, x, y, r );
    resid = libblis_test_bitrp_vector(a, r, dt);
  }
  return resid;
}

double libblis_test_op_ger (
  test_params_t* params,
  iface_t        iface,
  char*          dc_str,
  char*          pc_str,
  char*          sc_str,
  tensor_t*      dim,
  atom_t         alpv
){
  num_t        datatype;
  dim_t        m, n;
  conj_t       conjx, conjy;
  obj_t        alpha, x, y, a;
  obj_t        a_save;
  double       resid = 0.0;

  // Use the datatype of the first char in the datatype combination string.
  bli_param_map_char_to_blis_dt( dc_str[0], &datatype );

  // Map the dimension specifier to actual dimensions.
  m = dim->m;
  n = dim->n;

  // Map parameter characters to BLIS constants.
  bli_param_map_char_to_blis_conj( pc_str[0], &conjx );
  bli_param_map_char_to_blis_conj( pc_str[1], &conjy );

  // Create test scalars.
  bli_obj_scalar_init_detached( datatype, &alpha );

  // Create test operands (vectors and/or matrices).
  libblis_test_vobj_create( params, datatype,
                            sc_str[0], m,    &x );
  libblis_test_vobj_create( params, datatype,
                            sc_str[1], n,    &y );
  libblis_test_mobj_create( params, datatype, BLIS_NO_TRANSPOSE,
                            sc_str[2], m, n, &a );
  libblis_test_mobj_create( params, datatype, BLIS_NO_TRANSPOSE,
                            sc_str[2], m, n, &a_save );

  // Set alpha.
  if((params->bitextf == 0) && (params->oruflw == BLIS_DEFAULT)) {
    if ( bli_obj_is_real( &y ) ) {
      //bli_setsc(alpv, 0.0, &alpha);
      bli_setsc( -1.0,  1.0, &alpha );
    }
    else {
      //bli_setsc(alpv, (alpv/0.8), &alpha);
      bli_setsc( -1.0,  1.0, &alpha );
    }

    // Randomize x and y.
    libblis_test_vobj_randomize( params, TRUE, &x );
    libblis_test_vobj_randomize( params, TRUE, &y );
  }
  else{
    int32_t xx = (int32_t)alpv.real;
    if ( bli_obj_is_real( &y ) ) {
      bli_setsc( (double)xx,  0.0, &alpha );
    }
    else {
      // For syrk, both alpha and beta may be complex since, unlike herk,
      // C is symmetric in both the real and complex cases.
      int32_t ac = (int32_t)(xx/0.8);
      bli_setsc( (double)xx, (double)ac, &alpha );
    }

    // Randomize x and y, and save y.
    libblis_test_vobj_irandomize( params, &x );
    libblis_test_vobj_irandomize( params, &y );
  }

  // Initialize A to identity and save.
  bli_setm( &BLIS_ZERO, &a );
  bli_setd( &BLIS_ONE,  &a );
  bli_copym( &a, &a_save );

  // Apply the parameters.
  bli_obj_set_conj( conjx, &x );
  bli_obj_set_conj( conjy, &y );

  // Perform checks.
  libblis_api_ger(params, iface, &alpha, &x, &y, &a, datatype );

#ifndef __GTEST_VALGRIND_TEST__
  if(params->bitrp) {
    obj_t r;

    libblis_test_mobj_create( params, datatype, BLIS_NO_TRANSPOSE,
                            sc_str[2], m, n, &r );

    resid = libblis_test_bitrp_ger( params, iface, &alpha, &x, &y, &a,
                                              &a_save, &r, datatype );

    bli_obj_free( &r );
  }
  else {
    resid = libblis_ref_ger( params, &alpha, &x, &y, &a, &a_save);
  }
#endif

  // Zero out performance and residual if output matrix is empty.
  libblis_test_check_empty_problem( &a, &resid );

  // Free the test objects.
  libblis_test_obj_free( &x );
  libblis_test_obj_free( &y );
  libblis_test_obj_free( &a );
  libblis_test_obj_free( &a_save );

  return abs(resid);
}

void libblis_test_ger_impl(
  iface_t   iface,
  obj_t*    alpha,
  obj_t*    x,
  obj_t*    y,
  obj_t*    a
){
  switch( iface )
  {
    case BLIS_TEST_SEQ_FRONT_END:
      bli_ger( alpha, x, y, a );
      break;

    default:
      libblis_test_printf_error( "Invalid interface type.\n" );
  }
}

double libblis_test_ger_check(
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
  dim_t  n_a     = bli_obj_width( a );

  obj_t  t, v, w;
  obj_t  rho, norm;

  double junk;
  double resid = 0.0;

  //
  // Pre-conditions:
  // - x is randomized.
  // - y is randomized.
  // - a is identity.
  // Note:
  // - alpha should have a non-zero imaginary component in the
  //   complex cases in order to more fully exercise the implementation.
  //
  // Under these conditions, we assume that the implementation for
  //
  //   A := A_orig + alpha * conjx(x) * conjy(y)
  //
  // is functioning correctly if
  //
  //   normfv( v - w )
  //
  // is negligible, where
  //
  //   v = A * t
  //   w = ( A_orig + alpha * conjx(x) * conjy(y)^T ) * t
  //     =   A_orig * t + alpha * conjx(x) * conjy(y)^T * t
  //     =   A_orig * t + alpha * conjx(x) * rho
  //     =   A_orig * t + w
  //

  bli_obj_scalar_init_detached( dt,      &rho );
  bli_obj_scalar_init_detached( dt_real, &norm );

  bli_obj_create( dt, n_a, 1, 0, 0, &t );
  bli_obj_create( dt, m_a, 1, 0, 0, &v );
  bli_obj_create( dt, m_a, 1, 0, 0, &w );

  libblis_test_vobj_randomize( params, TRUE, &t );

  bli_gemv( &BLIS_ONE, a, &t, &BLIS_ZERO, &v );

  bli_dotv( y, &t, &rho );
  bli_mulsc( alpha, &rho );
  bli_scal2v( &rho, x, &w );
  bli_gemv( &BLIS_ONE, a_orig, &t, &BLIS_ONE, &w );

  bli_subv( &w, &v );
  bli_normfv( &v, &norm );
  bli_getsc( &norm, &resid, &junk );

  bli_obj_free( &t );
  bli_obj_free( &v );
  bli_obj_free( &w );

  return resid;
}