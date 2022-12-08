#include "blis_test.h"
#include "blis_utils.h"
#include "test_trmm.h"

// Local prototypes.
void libblis_test_trmm_deps (
  thread_data_t* tdata,
  test_params_t* params,
  test_op_t*     op
);

void libblis_test_trmm_impl (
  iface_t iface,
  side_t  side,
  obj_t*  alpha,
  obj_t*  a,
  obj_t*  b
);

double libblis_test_trmm_check (
  test_params_t* params,
  side_t         side,
  obj_t*         alpha,
  obj_t*         a,
  obj_t*         b,
  obj_t*         b_orig
);

double cblas_trmm(
  f77_int mm,
  f77_int nn,
  f77_int lda,
  f77_int ldb,
  obj_t*  a,
  obj_t*  b,
  obj_t*  alpha,
  uplo_t  uploa,
  side_t  side,
  diag_t  diaga,
  trans_t transa,
  num_t   dt
){
  enum CBLAS_ORDER     cblas_order;
  enum CBLAS_UPLO      cblas_uplo;
  enum CBLAS_SIDE      cblas_side;
  enum CBLAS_DIAG      cblas_diag;
  enum CBLAS_TRANSPOSE cblas_transa;

  if ( bli_obj_row_stride( b ) == 1 )
    cblas_order = CblasColMajor;
  else
    cblas_order = CblasRowMajor;

  if( bli_is_trans( transa ) )
    cblas_transa = CblasTrans;
  else if( bli_is_conjtrans( transa ) )
    cblas_transa = CblasConjTrans;
  else
    cblas_transa = CblasNoTrans;

  if(bli_is_upper(uploa))
    cblas_uplo = CblasUpper;
  else
    cblas_uplo = CblasLower;

  if(bli_is_left(side))
    cblas_side = CblasLeft;
  else
    cblas_side = CblasRight;

  if(bli_is_unit_diag(diaga))
    cblas_diag = CblasUnit;
  else
    cblas_diag = CblasNonUnit;

  switch( dt )  {
    case BLIS_FLOAT :
    {
      float*   alphap = (float*) bli_obj_buffer( alpha );
      float*   ap     = (float*) bli_obj_buffer( a );
      float*   bp     = (float*) bli_obj_buffer( b );
      cblas_strmm( cblas_order, cblas_side, cblas_uplo, cblas_transa,
                       cblas_diag, mm, nn, *alphap, ap, lda, bp, ldb );
      break;
    }
    case BLIS_DOUBLE :
    {
      double*  alphap = (double*) bli_obj_buffer( alpha );
      double*  ap     = (double*) bli_obj_buffer( a );
      double*  bp     = (double*) bli_obj_buffer( b );
      cblas_dtrmm( cblas_order, cblas_side, cblas_uplo, cblas_transa,
                       cblas_diag, mm, nn, *alphap, ap, lda, bp, ldb );
      break;
    }
    case BLIS_SCOMPLEX :
    {
      scomplex*  alphap = (scomplex*) bli_obj_buffer( alpha );
      scomplex*  ap     = (scomplex*) bli_obj_buffer( a );
      scomplex*  bp     = (scomplex*) bli_obj_buffer( b );
      cblas_ctrmm( cblas_order, cblas_side, cblas_uplo, cblas_transa,
                       cblas_diag, mm, nn, alphap, ap, lda, bp, ldb );
      break;
    }
    case BLIS_DCOMPLEX :
    {
      dcomplex*  alphap = (dcomplex*) bli_obj_buffer( alpha );
      dcomplex*  ap     = (dcomplex*) bli_obj_buffer( a );
      dcomplex*  bp     = (dcomplex*) bli_obj_buffer( b );
      cblas_ztrmm( cblas_order, cblas_side, cblas_uplo, cblas_transa,
                       cblas_diag, mm, nn, alphap, ap, lda, bp, ldb );
      break;
    }
    default :
      bli_check_error_code( BLIS_INVALID_DATATYPE );
  }
  return 0;
}

double blas_trmm(
  f77_int mm,
  f77_int nn,
  f77_int lda,
  f77_int ldb,
  obj_t*  a,
  obj_t*  b,
  obj_t*  alpha,
  uplo_t  uploa,
  side_t  side,
  diag_t  diaga,
  trans_t transa,
  num_t   dt
){
  f77_char f77_side;
  f77_char f77_uploa;
  f77_char f77_transa;
  f77_char f77_diaga;

  bli_param_map_blis_to_netlib_side( side, &f77_side );
  bli_param_map_blis_to_netlib_uplo( uploa, &f77_uploa );
  bli_param_map_blis_to_netlib_trans( transa, &f77_transa );
  bli_param_map_blis_to_netlib_diag( diaga, &f77_diaga );

  switch( dt )  {
    case BLIS_FLOAT :
    {
      float*   alphap = (float*) bli_obj_buffer( alpha );
      float*   ap     = (float*) bli_obj_buffer( a );
      float*   bp     = (float*) bli_obj_buffer( b );
      strmm_( &f77_side, &f77_uploa, &f77_transa, &f77_diaga,
                          &mm, &nn, alphap, ap, &lda, bp, &ldb );
      break;
    }
    case BLIS_DOUBLE :
    {
      double*  alphap = (double*) bli_obj_buffer( alpha );
      double*  ap     = (double*) bli_obj_buffer( a );
      double*  bp     = (double*) bli_obj_buffer( b );
      dtrmm_( &f77_side, &f77_uploa, &f77_transa, &f77_diaga,
                          &mm, &nn, alphap, ap, &lda, bp, &ldb );
      break;
    }
    case BLIS_SCOMPLEX :
    {
      scomplex*  alphap = (scomplex*) bli_obj_buffer( alpha );
      scomplex*  ap     = (scomplex*) bli_obj_buffer( a );
      scomplex*  bp     = (scomplex*) bli_obj_buffer( b );
      ctrmm_( &f77_side, &f77_uploa, &f77_transa, &f77_diaga,
                          &mm, &nn, alphap, ap, &lda, bp, &ldb );
      break;
    }
    case BLIS_DCOMPLEX :
    {
      dcomplex*  alphap = (dcomplex*) bli_obj_buffer( alpha );
      dcomplex*  ap     = (dcomplex*) bli_obj_buffer( a );
      dcomplex*  bp     = (dcomplex*) bli_obj_buffer( b );
      ztrmm_( &f77_side, &f77_uploa, &f77_transa, &f77_diaga,
                          &mm, &nn, alphap, ap, &lda, bp, &ldb );
      break;
    }
    default :
      bli_check_error_code( BLIS_INVALID_DATATYPE );
  }
  return 0;
}

void libblis_api_trmm(
  test_params_t* params,
  iface_t        iface,
  side_t         side,
  obj_t*         alpha,
  obj_t*         a,
  obj_t*         b,
  num_t          dt
) {
  if(params->api == API_BLIS) {
    libblis_test_trmm_impl( iface, side, alpha, a, b );
  }
  else { /*CLBAS  || BLAS */
    uplo_t  uploa  = bli_obj_uplo( a );
    trans_t transa = bli_obj_conjtrans_status( a );
    diag_t  diaga  = bli_obj_diag( a );
    f77_int mm     = bli_obj_length( b );
    f77_int nn     = bli_obj_width( b );
    f77_int lda, ldb;

    if ( bli_obj_row_stride( a ) == 1 )
      lda    = bli_obj_col_stride( a );
    else
      lda    = bli_obj_row_stride( a );

    if ( bli_obj_row_stride( b ) == 1 )
      ldb    = bli_obj_col_stride( b );
    else
      ldb    = bli_obj_row_stride( b );

    if(params->ldf == 1) {
      lda = lda + params->ld[0];
      ldb = ldb + params->ld[1];
    }

    if(bli_obj_has_notrans(a) && bli_obj_has_conj(a)) {
       conjugate_tensor(a, dt);
       transa = bli_obj_onlytrans_status( a );
    }

    if(params->api == API_CBLAS) {
      cblas_trmm(mm, nn, lda, ldb, a, b, alpha,
                                  uploa, side, diaga, transa, dt);
    } else { /**/
      if( bli_obj_row_stride( a ) == 1 ) {
        blas_trmm(mm, nn, lda, ldb, a, b, alpha,
                                uploa, side, diaga, transa, dt);
      }
      else {
        if( side == BLIS_LEFT)
          side = BLIS_RIGHT;
        else if(side == BLIS_RIGHT)
          side = BLIS_LEFT;

        if( uploa == BLIS_UPPER)
          uploa = BLIS_LOWER;
        else if(uploa == BLIS_LOWER)
          uploa = BLIS_UPPER;

        blas_trmm(nn, mm, lda, ldb, a, b, alpha,
                                uploa, side, diaga, transa, dt);
      }
    }
  }
  return ;
}

double libblis_ref_trmm(
  test_params_t* params,
  side_t         side,
  obj_t*         alpha,
  obj_t*         a,
  obj_t*         b,
  obj_t*         b_orig,
  num_t          dt
){

  double resid = 0.0;

  if (params->nanf) {
    resid = libblis_check_nan_trmm(b, dt );
  }
  else if((params->bitextf == 0) && (params->oruflw == BLIS_DEFAULT)) {
    resid = libblis_test_trmm_check(params, side, alpha, a, b, b_orig);
  }
  else {
    if(params->oruflw == BLIS_DEFAULT) {
      resid = libblis_test_itrmm_check(params, side, alpha, a, b, b_orig, dt);
    }
    else {
      resid = libblis_test_matrix_check(params, b);
    }
  }
  return resid;
}

double libblis_test_bitrp_trmm(
  test_params_t* params,
  iface_t        iface,
  side_t         side,
  obj_t*         alpha,
  obj_t*         a,
  obj_t*         b,
  obj_t*         b_orig,
  obj_t*         r,
  num_t          dt
) {
  double resid = 0.0;
	 unsigned int n_repeats = params->n_repeats;
	 unsigned int i;

  for(i = 0; i < n_repeats; i++) {
    bli_copym( b_orig, r );
    libblis_test_trmm_impl( iface, side, alpha, a, r );
    resid = libblis_test_bitrp_matrix(b, r, dt);
  }
  return resid;
}

double libblis_test_op_trmm (
  test_params_t* params,
  iface_t        iface,
  char*          dc_str,
  char*          pc_str,
  char*          sc_str,
  tensor_t*      dim,
  atom_t         alpv
) {
  num_t        datatype;
  dim_t        m, n;
  dim_t        mn_side;
  side_t       side;
  uplo_t       uploa;
  trans_t      transa;
  diag_t       diaga;
  obj_t        alpha, a, b;
  obj_t        b_save;
  double       resid = 0.0;
  obj_t        aa;

  // Use the datatype of the first char in the datatype combination string.
  bli_param_map_char_to_blis_dt( dc_str[0], &datatype );

  // Map the dimension specifier to actual dimensions.
  m = dim->m;
  n = dim->n;

  // Map parameter characters to BLIS constants.
  bli_param_map_char_to_blis_side( pc_str[0], &side );
  bli_param_map_char_to_blis_uplo( pc_str[1], &uploa );
  bli_param_map_char_to_blis_diag( pc_str[3], &diaga );

  if(params->api == API_BLIS) {
    bli_param_map_char_to_blis_trans( pc_str[2], &transa );
  } else {
    bli_param_map_char_to_blas_trans( pc_str[2], &transa );
  }

  // Create test scalars.
  bli_obj_scalar_init_detached( datatype, &alpha );

  // Create test operands (vectors and/or matrices).
  bli_set_dim_with_side( side, m, n, &mn_side );
  libblis_test_mobj_create( params, datatype, transa,
                        sc_str[1], mn_side, mn_side, &a );
  libblis_test_mobj_create( params, datatype, transa,
                        sc_str[1], mn_side, mn_side, &aa );
  libblis_test_mobj_create( params, datatype, BLIS_NO_TRANSPOSE,
                        sc_str[0], m,       n,       &b );
  libblis_test_mobj_create( params, datatype, BLIS_NO_TRANSPOSE,
                        sc_str[0], m,       n,       &b_save );

  // Set the structure and uplo properties of A.
  bli_obj_set_struc( BLIS_TRIANGULAR, &a );
  bli_obj_set_uplo( uploa, &a );

  bli_obj_set_struc( BLIS_TRIANGULAR, &aa );
  bli_obj_set_uplo( uploa, &aa );

  // Set alpha.
  if((params->bitextf == 0) && (params->oruflw == BLIS_DEFAULT)) {
    if ( bli_obj_is_real( &b ) )
      bli_setsc(  alpv.real,  0.0, &alpha );
    else
      bli_setsc(  alpv.real,  0.0, &alpha );

    // Randomize A, load the diagonal, make it densely triangular.
    libblis_test_mobj_randomize( params, TRUE, &a );

    // Randomize B and save B.
    libblis_test_mobj_randomize( params, TRUE, &b );
  }
  else {
    int32_t x = (int32_t)alpv.real;
    if ( bli_obj_is_real( &b ) )	{
      bli_setsc( (double)x,  0.0, &alpha );
    }
    else	{
      int32_t ac = (int32_t)(x/0.8);
      bli_setsc( (double)x, (double)ac, &alpha );
    }

    // Randomize A, load the diagonal, make it densely triangular
    libblis_test_mobj_irandomize( params, &a );

    // Randomize B and save B.
    libblis_test_mobj_irandomize( params, &b );
  }

  if (params->nanf) {
    test_fillbuffmem( &b, datatype );
    test_fillbuffmem_diag( &b, datatype );
  }

  bli_mktrim( &a );

  //Copy b to b_save
  bli_copym( &a, &aa );
  bli_copym( &b, &b_save );

  // Apply the remaining parameters.
  bli_obj_set_conjtrans( transa, &a );
  bli_obj_set_diag( diaga, &a );

  bli_obj_set_conjtrans( transa, &aa );
  bli_obj_set_diag( diaga, &aa );

  libblis_api_trmm(params, iface, side, &alpha, &aa, &b, datatype);

#ifndef __GTEST_VALGRIND_TEST__
  if(params->bitrp) {
    obj_t r;

    libblis_test_mobj_create( params, datatype, BLIS_NO_TRANSPOSE,
                                                     sc_str[0], m, n, &r );
    resid = libblis_test_bitrp_trmm(params, iface, side, &alpha, &a,
                                                &b, &b_save, &r, datatype);
    bli_obj_free( &r );
  }
  else {
    resid = libblis_ref_trmm(params, side, &alpha, &a, &b, &b_save, datatype);
  }
#endif

  // Zero out performance and residual if output matrix is empty.
  libblis_test_check_empty_problem( &b, &resid );

  // Free the test objects.
  libblis_test_obj_free( &a );
  libblis_test_obj_free( &aa );
  libblis_test_obj_free( &b );
  libblis_test_obj_free( &b_save );

  return abs(resid);
}

void libblis_test_trmm_impl(
  iface_t   iface,
  side_t    side,
  obj_t*    alpha,
  obj_t*    a,
  obj_t*    b
) {
  switch( iface ) {
    case BLIS_TEST_SEQ_FRONT_END:
      bli_trmm( side, alpha, a, b );
    break;

    default:
      libblis_test_printf_error( "Invalid interface type.\n" );
  }
}

double libblis_test_trmm_check (
  test_params_t* params,
  side_t         side,
  obj_t*         alpha,
  obj_t*         a,
  obj_t*         b,
  obj_t*         b_orig
) {
  num_t  dt      = bli_obj_dt( b );
  num_t  dt_real = bli_obj_dt_proj_to_real( b );

  dim_t  m       = bli_obj_length( b );
  dim_t  n       = bli_obj_width( b );

  obj_t  norm;
  obj_t  t, v, w, z;

  double junk;
  double resid = 0.0;

  // Pre-conditions:
  // - a is randomized and triangular.
  // - b_orig is randomized.
  // Note:
  // - alpha should have a non-zero imaginary component in the
  //   complex cases in order to more fully exercise the implementation.
  //
  // Under these conditions, we assume that the implementation for
  //
  //   B := alpha * transa(A) * B_orig    (side = left)
  //   B := alpha * B_orig * transa(A)    (side = right)
  //
  // is functioning correctly if
  //
  //   normfv( v - z )
  //
  // is negligible, where
  //
  //   v = B * t
  //
  //   z = ( alpha * transa(A) * B ) * t     (side = left)
  //     = alpha * transa(A) * B * t
  //     = alpha * transa(A) * w
  //
  //   z = ( alpha * B * transa(A) ) * t     (side = right)
  //     = alpha * B * transa(A) * t
  //     = alpha * B * w

  bli_obj_scalar_init_detached( dt_real, &norm );

  if ( bli_is_left( side ) )
  {
    bli_obj_create( dt, n, 1, 0, 0, &t );
    bli_obj_create( dt, m, 1, 0, 0, &v );
    bli_obj_create( dt, m, 1, 0, 0, &w );
    bli_obj_create( dt, m, 1, 0, 0, &z );
  }
  else // else if ( bli_is_left( side ) )
  {
    bli_obj_create( dt, n, 1, 0, 0, &t );
    bli_obj_create( dt, m, 1, 0, 0, &v );
    bli_obj_create( dt, n, 1, 0, 0, &w );
    bli_obj_create( dt, m, 1, 0, 0, &z );
  }

  libblis_test_vobj_randomize( params, TRUE, &t );

  bli_gemv( &BLIS_ONE, b, &t, &BLIS_ZERO, &v );

  if ( bli_is_left( side ) )
  {
    bli_gemv( &BLIS_ONE, b_orig, &t, &BLIS_ZERO, &w );
    bli_trmv( alpha, a, &w );
    bli_copyv( &w, &z );
  }
  else
  {
    bli_copyv( &t, &w );
    bli_trmv( &BLIS_ONE, a, &w );
    bli_gemv( alpha, b_orig, &w, &BLIS_ZERO, &z );
  }

  bli_subv( &z, &v );
  bli_normfv( &v, &norm );
  bli_getsc( &norm, &resid, &junk );

  bli_obj_free( &t );
  bli_obj_free( &v );
  bli_obj_free( &w );
  bli_obj_free( &z );

  return resid;
}
