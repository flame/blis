#include "blis_test.h"
#include "blis_utils.h"
#include "test_trsv.h"

// Local prototypes.
void libblis_test_trsv_deps(
  thread_data_t* tdata,
  test_params_t* params,
  test_op_t*     op
);

void libblis_test_trsv_impl(
  iface_t   iface,
  obj_t*    alpha,
  obj_t*    a,
  obj_t*    x
);

double libblis_test_trsv_check(
  test_params_t* params,
  obj_t*         alpha,
  obj_t*         a,
  obj_t*         x,
  obj_t*         x_orig
);

void libblis_alphax(
  obj_t*  alpha,
  f77_int m,
  obj_t*  x,
  f77_int incx,
  num_t   dt
){
  int i, ix = 0;
	  switch( dt )  {
    case BLIS_FLOAT :
    {
      float*   Alpha  = (float*) bli_obj_buffer( alpha );
      float*   xp     = (float*) bli_obj_buffer( x );
      for(i = 0 ; i < m ; i++) {
        xp[ix] = (*Alpha * xp[ix]);
        ix = ix + incx;
      }
      break;
    }
    case BLIS_DOUBLE :
    {
      double*  Alpha  = (double*) bli_obj_buffer( alpha );
      double*  xp     = (double*) bli_obj_buffer( x );
      for(i = 0 ; i < m ; i++) {
        xp[ix] = (*Alpha * xp[ix]);
        ix = ix + incx;
      }
      break;
    }
    case BLIS_SCOMPLEX :
    {
      scomplex*  Alpha  = (scomplex*) bli_obj_buffer( alpha );
      scomplex*  xp     = (scomplex*) bli_obj_buffer( x );
      for(i = 0 ; i < m ; i++) {
        xp[ix] = mulc<scomplex>(*Alpha , xp[ix]);
        ix = ix + incx;
      }
      break;
    }
    case BLIS_DCOMPLEX :
    {
      dcomplex*  Alpha  = (dcomplex*) bli_obj_buffer( alpha );
      dcomplex*  xp     = (dcomplex*) bli_obj_buffer( x );
      for(i = 0 ; i < m ; i++) {
        xp[ix] = mulc<dcomplex>(*Alpha , xp[ix]);
        ix = ix + incx;
      }
      break;
    }
    default:
      bli_check_error_code( BLIS_INVALID_DATATYPE );
  }
}

void cblas_trsv(
  uplo_t  uploa,
  trans_t transa,
  diag_t  diaga,
  f77_int m,
  obj_t*  a,
  f77_int lda,
  obj_t*  x,
  f77_int incx,
  num_t   dt
){
	 enum CBLAS_ORDER cblas_order;
 	enum CBLAS_UPLO cblas_uploa;
  enum CBLAS_DIAG cblas_diaga;
  enum CBLAS_TRANSPOSE cblas_transa;

  if(bli_obj_row_stride(a) == 1 )
    cblas_order = CblasColMajor;
  else
    cblas_order = CblasRowMajor;

  if(bli_is_upper(uploa))
    cblas_uploa = CblasUpper;
  else
    cblas_uploa = CblasLower;

  if( bli_is_trans( transa ) )
    cblas_transa = CblasTrans;
  else if( bli_is_conjtrans( transa ) )
    cblas_transa = CblasConjTrans;
  else
    cblas_transa = CblasNoTrans;

  if(bli_is_unit_diag(diaga))
    cblas_diaga = CblasUnit;
  else
    cblas_diaga = CblasNonUnit;

	  switch( dt )  {
    case BLIS_FLOAT :
    {
      float*   ap     = (float*) bli_obj_buffer( a );
      float*   xp     = (float*) bli_obj_buffer( x );
      cblas_strsv(cblas_order, cblas_uploa, cblas_transa, cblas_diaga,
                                                m, ap, lda, xp, incx );
      break;
    }
    case BLIS_DOUBLE :
    {
      double*  ap     = (double*) bli_obj_buffer( a );
      double*  xp     = (double*) bli_obj_buffer( x );
      cblas_dtrsv(cblas_order, cblas_uploa, cblas_transa, cblas_diaga,
                                                m, ap, lda, xp, incx );
      break;
    }
    case BLIS_SCOMPLEX :
    {
      scomplex*  ap     = (scomplex*) bli_obj_buffer( a );
      scomplex*  xp     = (scomplex*) bli_obj_buffer( x );
      cblas_ctrsv(cblas_order, cblas_uploa, cblas_transa, cblas_diaga,
                                                m, ap, lda, xp, incx );
      break;
    }
    case BLIS_DCOMPLEX :
    {
      dcomplex*  ap     = (dcomplex*) bli_obj_buffer( a );
      dcomplex*  xp     = (dcomplex*) bli_obj_buffer( x );
      cblas_ztrsv(cblas_order, cblas_uploa, cblas_transa, cblas_diaga,
                                                m, ap, lda, xp, incx );
      break;
    }
    default:
      bli_check_error_code( BLIS_INVALID_DATATYPE );
  }
}

void blas_trsv(
  uplo_t  uploa,
  trans_t transa,
  diag_t  diaga,
  f77_int m,
  obj_t*  a,
  f77_int lda,
  obj_t*  x,
  f77_int incx,
  num_t   dt
){
  f77_char f77_uploa;
  f77_char f77_transa;
  f77_char f77_diaga;
  trans_t trans;

  if( bli_obj_row_stride( a ) == 1 ){
    if ( transa == BLIS_TRANSPOSE )              trans = BLIS_TRANSPOSE;
    else if ( transa == BLIS_CONJ_TRANSPOSE )    trans = BLIS_CONJ_TRANSPOSE;
    else /*if(transa == BLIS_NO_TRANSPOSE)*/     trans = BLIS_NO_TRANSPOSE;
  }
  else {
    if( uploa == BLIS_UPPER)
      uploa = BLIS_LOWER;
    else if(uploa == BLIS_LOWER)
      uploa = BLIS_UPPER;
    if(transa == BLIS_NO_TRANSPOSE)               trans = BLIS_TRANSPOSE;
    else if(transa == BLIS_TRANSPOSE)             trans = BLIS_NO_TRANSPOSE;
    else  /*if ( transa == BLIS_CONJ_TRANSPOSE)*/ trans = BLIS_NO_TRANSPOSE;
  }

  bli_param_map_blis_to_netlib_uplo( uploa, &f77_uploa );
 	bli_param_map_blis_to_netlib_trans( trans, &f77_transa );
 	bli_param_map_blis_to_netlib_diag( diaga, &f77_diaga );

  switch( dt )  {
    case BLIS_FLOAT :
    {
      float*   ap     = (float*) bli_obj_buffer( a );
      float*   xp     = (float*) bli_obj_buffer( x );
      strsv_(&f77_uploa, &f77_transa, &f77_diaga, &m, ap, (f77_int*)&lda, xp, &incx);
      break;
    }
    case BLIS_DOUBLE :
    {
      double*  ap     = (double*) bli_obj_buffer( a );
      double*  xp     = (double*) bli_obj_buffer( x );
      dtrsv_(&f77_uploa, &f77_transa, &f77_diaga, &m, ap, (f77_int*)&lda, xp, &incx);
      break;
    }
    case BLIS_SCOMPLEX :
    {
      scomplex*  ap     = (scomplex*) bli_obj_buffer( a );
      scomplex*  xp     = (scomplex*) bli_obj_buffer( x );
      ctrsv_(&f77_uploa, &f77_transa, &f77_diaga, &m, ap, (f77_int*)&lda, xp, &incx);
      break;
    }
    case BLIS_DCOMPLEX :
    {
      dcomplex*  ap     = (dcomplex*) bli_obj_buffer( a );
      dcomplex*  xp     = (dcomplex*) bli_obj_buffer( x );
      ztrsv_(&f77_uploa, &f77_transa, &f77_diaga, &m, ap, (f77_int*)&lda, xp, &incx);
      break;
    }
    default:
      bli_check_error_code( BLIS_INVALID_DATATYPE );
  }
}

void libblis_api_trsv(
  test_params_t* params,
  iface_t        iface,
  obj_t*         alpha,
  obj_t*         a,
  obj_t*         x,
  num_t          dt
){

  if(params->api == API_BLIS) {
    libblis_test_trsv_impl( iface, alpha, a, x );
  }
  else { /*CLBAS  || BLAS */
    uplo_t  uploa  = bli_obj_uplo( a );
    trans_t transa = bli_obj_conjtrans_status( a );
    diag_t  diaga  = bli_obj_diag( a );
    f77_int mm     = bli_obj_length( a );
    f77_int incx   = bli_obj_vector_inc( x );
    f77_int lda ;

   if ( bli_obj_row_stride( a ) == 1 ) {
      lda    = bli_obj_col_stride( a );
    } else {
      lda    = bli_obj_row_stride( a );
    }

    if(params->ldf == 1) {
      lda = lda + params->ld[0];
    }

    libblis_alphax(alpha, mm, x, incx, dt);

    if(bli_obj_has_notrans(a) && bli_obj_has_conj(a)) {
       conjugate_tensor(a, dt);
       transa = bli_obj_onlytrans_status( a );
    }

    if(params->api == API_CBLAS) {
     	cblas_trsv(uploa, transa, diaga, mm, a, lda, x, incx, dt );
    }
    else { /**/
      if ( bli_obj_row_stride( a ) == 1 ){
        blas_trsv(uploa, transa, diaga, mm, a, lda, x, incx, dt );
      }
      else {
        blas_trsv(uploa, transa, diaga, mm, a, lda, x, incx, dt );
      }
    }
  }
  return ;
}

double libblis_ref_trsv(
  test_params_t* params,
  obj_t*         alpha,
  obj_t*         a,
  obj_t*         x,
  obj_t*         x_orig
){
  double resid = 0.0;

  if((params->bitextf == 0) && (params->oruflw == BLIS_DEFAULT)) {
    resid = libblis_test_trsv_check( params, alpha, a, x, x_orig);
  }
  else {
    if(params->oruflw == BLIS_DEFAULT) {
      resid = libblis_test_itrsv_check( params, alpha, a, x, x_orig);
    }
    else {
      resid = libblis_test_vector_check(params, x);
    }
  }
  return resid;
}

double libblis_test_bitrp_trsv(
  test_params_t* params,
  iface_t        iface,
  obj_t*         alpha,
  obj_t*         a,
  obj_t*         x,
  obj_t*         x_orig,
  obj_t*         r,
  num_t          dt
) {
  double resid = 0.0;
	 unsigned int n_repeats = params->n_repeats;
	 unsigned int i;

  for(i = 0; i < n_repeats; i++) {
    bli_copyv( x_orig, r );
    libblis_test_trsv_impl( iface, alpha, a, x );
    resid = libblis_test_bitrp_vector(x, r, dt);
  }
  return resid;
}

double libblis_test_op_trsv (
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
  trans_t      transa;
  diag_t       diaga;
  obj_t        alpha, a, x;
  obj_t        x_save;
  double       resid = 0.0;
  obj_t        aa;

  // Use the datatype of the first char in the datatype combination string.
  bli_param_map_char_to_blis_dt( dc_str[0], &datatype );

  // Map the dimension specifier to an actual dimension.
  m = dim->m;

  // Map parameter characters to BLIS constants.
  bli_param_map_char_to_blis_uplo( pc_str[0], &uploa );
  bli_param_map_char_to_blis_trans( pc_str[1], &transa );
  bli_param_map_char_to_blis_diag( pc_str[2], &diaga );

  // Create test scalars.
  bli_obj_scalar_init_detached( datatype, &alpha );

  // Create test operands (vectors and/or matrices).
  libblis_test_mobj_create( params, datatype, BLIS_NO_TRANSPOSE,
                            sc_str[0], m, m, &a );
  libblis_test_mobj_create( params, datatype, BLIS_NO_TRANSPOSE,
                            sc_str[0], m, m, &aa );
  libblis_test_vobj_create( params, datatype,
                            sc_str[1], m,    &x );
  libblis_test_vobj_create( params, datatype,
                            sc_str[1], m,    &x_save );

  // Set the structure and uplo properties of A.
  bli_obj_set_struc( BLIS_TRIANGULAR, &a );
  bli_obj_set_uplo( uploa, &a );

  bli_obj_set_struc( BLIS_TRIANGULAR, &aa );
  bli_obj_set_uplo( uploa, &aa );

  if((params->bitextf == 0) && (params->oruflw == BLIS_DEFAULT)) {
    if ( bli_obj_is_real( &x ) )	{
      bli_setsc(  alpv.real,  0.0, &alpha );
    }
    else	{
      bli_setsc(  alpv.real,  (alpv.real/0.8), &alpha );
    }
    // Randomize A, make it densely triangular.
    libblis_test_mobj_randomize( params, TRUE, &a );
    libblis_test_vobj_randomize( params, TRUE, &x );
  }
  else{
    int32_t xx = (int32_t)alpv.real;
    if ( bli_obj_is_real( &x ) )	{
      bli_setsc( (double)xx,  0.0, &alpha );
    }
    else	{
      int32_t ac = (int32_t)(xx/0.8);
      bli_setsc( (double)xx, (double)ac, &alpha );
    }
    // Randomize A, make it densely triangular.
    libblis_test_mobj_irandomize( params, &a );
    libblis_test_vobj_irandomize( params, &x );
  }

  // Randomize A, load the diagonal, make it densely triangular.
  libblis_test_mobj_load_diag( params, &a );
  bli_mktrim( &a );
  bli_copyv( &x, &x_save );

  bli_copym( &a, &aa );

  // Apply the remaining parameters.
  bli_obj_set_conjtrans( transa, &a );
  bli_obj_set_diag( diaga, &a );

  bli_obj_set_conjtrans( transa, &aa );
  bli_obj_set_diag( diaga, &aa );

  libblis_api_trsv(params, iface, &alpha, &aa, &x, datatype);

#ifndef __GTEST_VALGRIND_TEST__
  if(params->bitrp) {
    obj_t r;

    libblis_test_vobj_create( params, datatype, sc_str[1], m, &r );

    resid = libblis_test_bitrp_trsv( params, iface, &alpha, &a, &x,
                                               &x_save, &r, datatype);

    bli_obj_free( &r );
  }
  else {
    resid = libblis_ref_trsv( params, &alpha, &a, &x, &x_save );
  }
#endif

  // Zero out performance and residual if output vector is empty.
  libblis_test_check_empty_problem( &x, &resid );

  // Free the test objects.
  libblis_test_obj_free( &a );
  libblis_test_obj_free( &aa );
  libblis_test_obj_free( &x );
  libblis_test_obj_free( &x_save );

  return resid;
}

void libblis_test_trsv_impl(
  iface_t   iface,
  obj_t*    alpha,
  obj_t*    a,
  obj_t*    x
){
  switch ( iface ){
    case BLIS_TEST_SEQ_FRONT_END:
      bli_trsv( alpha, a, x );
      break;

    default:
      libblis_test_printf_error( "Invalid interface type.\n" );
  }
}

double libblis_test_trsv_check(
  test_params_t* params,
  obj_t*         alpha,
  obj_t*         a,
  obj_t*         x,
  obj_t*         x_orig
){
  num_t   dt      = bli_obj_dt( x );
  num_t   dt_real = bli_obj_dt_proj_to_real( x );

  dim_t   m       = bli_obj_vector_dim( x );

  uplo_t  uploa   = bli_obj_uplo( a );
  trans_t transa  = bli_obj_conjtrans_status( a );

  obj_t   alpha_inv;
  obj_t   a_local, y;
  obj_t   norm;

  double  junk;
  double  resid = 0.0;

  //
  // Pre-conditions:
  // - a is randomized and triangular.
  // - x is randomized.
  // Note:
  // - alpha should have a non-zero imaginary component in the
  //   complex cases in order to more fully exercise the implementation.
  //
  // Under these conditions, we assume that the implementation for
  //
  //   x := alpha * inv(transa(A)) * x_orig
  //
  // is functioning correctly if
  //
  //   normfv( y - x_orig )
  //
  // is negligible, where
  //
  //   y = inv(alpha) * transa(A_dense) * x
  //

  bli_obj_scalar_init_detached( dt,      &alpha_inv );
  bli_obj_scalar_init_detached( dt_real, &norm );

  bli_copysc( &BLIS_ONE, &alpha_inv );
  bli_divsc( alpha, &alpha_inv );

  bli_obj_create( dt, m, 1, 0, 0, &y );
  bli_obj_create( dt, m, m, 0, 0, &a_local );

  bli_obj_set_struc( BLIS_TRIANGULAR, &a_local );
  bli_obj_set_uplo( uploa, &a_local );
  bli_obj_toggle_uplo_if_trans( transa, &a_local );
  bli_copym( a, &a_local );
  bli_mktrim( &a_local );

  bli_obj_set_struc( BLIS_GENERAL, &a_local );
  bli_obj_set_uplo( BLIS_DENSE, &a_local );

  bli_gemv( &alpha_inv, &a_local, x, &BLIS_ZERO, &y );

  bli_subv( x_orig, &y );
  bli_normfv( &y, &norm );
  bli_getsc( &norm, &resid, &junk );

  bli_obj_free( &y );
  bli_obj_free( &a_local );

  return resid;
}