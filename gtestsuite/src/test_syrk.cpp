#include "blis_test.h"
#include "blis_utils.h"
#include "test_syrk.h"

void libblis_test_syrk_impl (
  iface_t   iface,
  obj_t*    alpha,
  obj_t*    a,
  obj_t*    beta,
  obj_t*    c
);

double libblis_test_syrk_check (
  test_params_t* params,
  obj_t*         alpha,
  obj_t*         a,
  obj_t*         beta,
  obj_t*         c,
  obj_t*         c_orig
);

double cblas_syrk(
	 uplo_t  uploc,
  trans_t transa,
  f77_int n,
  f77_int k,
  obj_t*  alpha,
  obj_t*  a,
  f77_int lda,
  obj_t*  beta,
  obj_t*  c,
  f77_int ldc,
  num_t   dt
){
  enum CBLAS_ORDER     cblas_order;
  enum CBLAS_TRANSPOSE cblas_trans;
 	enum CBLAS_UPLO      cblas_uplo;

  if ( bli_obj_row_stride( c ) == 1 )
    cblas_order = CblasColMajor;
  else
    cblas_order = CblasRowMajor;

  if( bli_is_upper( uploc ) )
    cblas_uplo = CblasUpper;
  else
    cblas_uplo = CblasLower;

  if( bli_is_trans( transa ) )
    cblas_trans = CblasTrans;
  else
    cblas_trans = CblasNoTrans;

  switch( dt )  {
    case BLIS_FLOAT :
    {
      float*   alphap = (float*) bli_obj_buffer( alpha );
      float*   ap     = (float*) bli_obj_buffer( a );
      float*   betap  = (float*) bli_obj_buffer( beta );
      float*   cp     = (float*) bli_obj_buffer( c );
      cblas_ssyrk( cblas_order, cblas_uplo, cblas_trans, n, k,
                                 *alphap, ap, lda, *betap, cp, ldc );
      break;
    }
    case BLIS_DOUBLE :
    {
      double*  alphap = (double*) bli_obj_buffer( alpha );
      double*  ap     = (double*) bli_obj_buffer( a );
      double*  betap  = (double*) bli_obj_buffer( beta );
      double*  cp     = (double*) bli_obj_buffer( c );
      cblas_dsyrk( cblas_order, cblas_uplo, cblas_trans, n, k,
                                 *alphap, ap, lda, *betap, cp, ldc );
      break;
    }
    case BLIS_SCOMPLEX :
    {
      scomplex*  alphap = (scomplex*) bli_obj_buffer( alpha );
      scomplex*  ap     = (scomplex*) bli_obj_buffer( a );
      scomplex*  betap  = (scomplex*) bli_obj_buffer( beta );
      scomplex*  cp     = (scomplex*) bli_obj_buffer( c );
      cblas_csyrk( cblas_order, cblas_uplo, cblas_trans, n, k,
                                   alphap, ap, lda, betap, cp, ldc );
      break;
    }
    case BLIS_DCOMPLEX :
    {
      dcomplex*  alphap = (dcomplex*) bli_obj_buffer( alpha );
      dcomplex*  ap     = (dcomplex*) bli_obj_buffer( a );
      dcomplex*  betap  = (dcomplex*) bli_obj_buffer( beta );
      dcomplex*  cp     = (dcomplex*) bli_obj_buffer( c );
      cblas_zsyrk( cblas_order, cblas_uplo, cblas_trans, n, k,
                                   alphap, ap, lda, betap, cp, ldc );
      break;
    }
    default :
      bli_check_error_code( BLIS_INVALID_DATATYPE );
  }
  return 0;
}

double blas_syrk(
  uplo_t   uploc,
  f77_char f77_transa,
  f77_int  n,
  f77_int  k,
  obj_t*   alpha,
  obj_t*   a,
  f77_int  lda,
  obj_t*   beta,
  obj_t*   c,
  f77_int  ldc,
  num_t    dt
){
  f77_char f77_uploc;

  bli_param_map_blis_to_netlib_uplo( uploc, &f77_uploc );
  switch( dt )  {
    case BLIS_FLOAT :
    {
      float*   alphap = (float*) bli_obj_buffer( alpha );
      float*   ap     = (float*) bli_obj_buffer( a );
      float*   betap  = (float*) bli_obj_buffer( beta );
      float*   cp     = (float*) bli_obj_buffer( c );
      ssyrk_( &f77_uploc,	&f77_transa, &n, &k, alphap, ap, (f77_int*)&lda,
                                               betap, cp, (f77_int*)&ldc );
      break;
    }
    case BLIS_DOUBLE :
    {
      double*  alphap = (double*) bli_obj_buffer( alpha );
      double*  ap     = (double*) bli_obj_buffer( a );
      double*  betap  = (double*) bli_obj_buffer( beta );
      double*  cp     = (double*) bli_obj_buffer( c );
      dsyrk_( &f77_uploc,	&f77_transa, &n, &k, alphap, ap,
                              (f77_int*)&lda, betap, cp, (f77_int*)&ldc );
      break;
    }
    case BLIS_SCOMPLEX :
    {
      scomplex*  alphap = (scomplex*) bli_obj_buffer( alpha );
      scomplex*  ap     = (scomplex*) bli_obj_buffer( a );
      scomplex*  betap  = (scomplex*) bli_obj_buffer( beta );
      scomplex*  cp     = (scomplex*) bli_obj_buffer( c );
      csyrk_( &f77_uploc,	&f77_transa, &n, &k, alphap, ap,
                              (f77_int*)&lda, betap, cp, (f77_int*)&ldc );
      break;
    }
    case BLIS_DCOMPLEX :
    {
      dcomplex*  alphap = (dcomplex*) bli_obj_buffer( alpha );
      dcomplex*  ap     = (dcomplex*) bli_obj_buffer( a );
      dcomplex*  betap  = (dcomplex*) bli_obj_buffer( beta );
      dcomplex*  cp     = (dcomplex*) bli_obj_buffer( c );
      zsyrk_( &f77_uploc,	&f77_transa, &n, &k, alphap, ap,
                              (f77_int*)&lda, betap, cp, (f77_int*)&ldc );
      break;
    }
    default :
      bli_check_error_code( BLIS_INVALID_DATATYPE );
  }
  return 0;
}

void libblis_api_syrk(
  test_params_t* params,
  iface_t   iface,
  obj_t*    alpha,
  obj_t*    a,
  obj_t*    beta,
  obj_t*    c,
  num_t     dt
){

  if(params->api == API_BLIS) {
    libblis_test_syrk_impl( iface, alpha, a, beta, c );
  }
  else { /*CLBAS  || BLAS */
    f77_int  m     = bli_obj_length( c );
    f77_int  k     = bli_obj_width_after_trans( a );
    uplo_t uploc   = bli_obj_uplo( c );
    trans_t transa = bli_obj_onlytrans_status( a );
    f77_int  lda, ldc;

  if( bli_obj_is_col_stored( c ) ) {
      lda    = bli_obj_col_stride( a );
      ldc    = bli_obj_col_stride( c );
    } else {
      lda    = bli_obj_row_stride( a );
      ldc    = bli_obj_row_stride( c );
    }

    if(params->ldf == 1) {
      lda = lda + params->ld[0];
      ldc = ldc + params->ld[2];
    }

    if(params->api == API_CBLAS) {
      cblas_syrk(uploc, transa, m, k, alpha, a, lda, beta, c, ldc, dt);
    }
    else { /**/
      f77_char f77_transa;
      if(bli_obj_is_col_stored( c )) {
        if(transa == BLIS_TRANSPOSE)                f77_transa='T';
        else if ( transa == BLIS_CONJ_TRANSPOSE )   f77_transa='N';
        else /*if ( transa == BLIS_NO_TRANSPOSE )*/ f77_transa='N';

        blas_syrk(uploc, f77_transa, m, k, alpha, a, lda,
                                               beta, c, ldc, dt);
      }
      else {
        if(transa == BLIS_TRANSPOSE)                f77_transa='N';
        else if ( transa == BLIS_CONJ_TRANSPOSE )   f77_transa='N';
        else /*if ( transa == BLIS_NO_TRANSPOSE )*/ f77_transa='T';

        if( uploc == BLIS_UPPER)
          uploc = BLIS_LOWER;
        else if(uploc == BLIS_LOWER)
          uploc = BLIS_UPPER;

        blas_syrk(uploc, f77_transa, m, k, alpha, a, lda,
                                               beta, c, ldc, dt);
      }
    }
  }
  return ;
}

double libblis_ref_syrk(
  test_params_t* params,
  obj_t*         alpha,
  obj_t*         a,
  obj_t*         beta,
  obj_t*         c,
  obj_t*         c_orig,
  num_t          dt
){
  double resid = 0.0;
//  double *betap = (double *)bli_obj_buffer( beta );

  if (params->nanf) {
    resid = libblis_check_nan_syrk(c, dt );
  }
  else if((params->bitextf == 0) && (params->oruflw == BLIS_DEFAULT)) {
 	  // Perform checks.
	   resid = libblis_test_syrk_check( params, alpha, a, beta, c, c_orig);
  }
  else {
    if(params->oruflw == BLIS_DEFAULT) {
      resid = libblis_test_isyrk_check( params, alpha, a, beta, c, c_orig);
    }
    else {
      resid = libblis_test_matrix_check(params, c);
    }
  }
  return resid;
}

double libblis_test_bitrp_syrk(
  test_params_t* params,
  iface_t        iface,
  obj_t*         alpha,
  obj_t*         a,
  obj_t*         beta,
  obj_t*         c,
  obj_t*         c_orig,
  obj_t*         r,
  num_t          dt
) {
  double resid = 0.0;
	 unsigned int n_repeats = params->n_repeats;
	 unsigned int i;

  for(i = 0; i < n_repeats; i++) {
    bli_copym( c_orig, r );
    libblis_test_syrk_impl( iface, alpha, a, beta, r );
    resid = libblis_test_bitrp_matrix(c, r, dt);
  }
  return resid;
}

double libblis_test_op_syrk (
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
  dim_t   m, k;
  uplo_t  uploc;
  trans_t transa;
  obj_t   alpha, a, beta, c;
  obj_t   c_save;
  double  resid = 0.0;
  obj_t   aa;

  // Use the datatype of the first char in the datatype combination string.
  bli_param_map_char_to_blis_dt( dc_str[0], &datatype );

  // Map the dimension specifier to actual dimensions.
  m = dim->m;
  k = dim->n;

    // Map parameter characters to BLIS constants.
  bli_param_map_char_to_blis_uplo( pc_str[0], &uploc );
  bli_param_map_char_to_syrk_trans( pc_str[1], &transa );

  // Create test scalars.
  bli_obj_scalar_init_detached( datatype, &alpha );
  bli_obj_scalar_init_detached( datatype, &beta );

  // Create test operands (vectors and/or matrices).
  libblis_test_mobj_create( params, datatype, transa,
                            sc_str[1], m, k, &a );
  libblis_test_mobj_create( params, datatype, transa,
                            sc_str[1], m, k, &aa );
  libblis_test_mobj_create( params, datatype, BLIS_NO_TRANSPOSE,
                            sc_str[0], m, m, &c );
  libblis_test_mobj_create( params, datatype, BLIS_NO_TRANSPOSE,
                            sc_str[0], m, m, &c_save );

  // Set the structure and uplo properties of C.
  bli_obj_set_struc( BLIS_SYMMETRIC, &c );
  bli_obj_set_uplo( uploc, &c );

  // Set alpha and beta.
  if((params->bitextf == 0) && (params->oruflw == BLIS_DEFAULT)) {
    if ( bli_obj_is_real( &c ) ) {
      bli_setsc(alpv.real, 0.0, &alpha);
      bli_setsc(betv.real, 0.0, &beta);
    }
    else {
      // For syrk, both alpha and beta may be complex since, unlike herk,
      // C is symmetric in both the real and complex cases.
      bli_setsc(alpv.real, (alpv.real/0.8), &alpha);
      bli_setsc(betv.real, (betv.real/1.2), &beta);
    }
    // Randomize A.
    libblis_test_mobj_randomize( params, TRUE, &a );

    libblis_test_mobj_randomize( params, TRUE, &c );
  }
  else {
    int32_t x = (int32_t)alpv.real;
    int32_t y = (int32_t)betv.real;
    if ( bli_obj_is_real( &c ) ) {
      bli_setsc( (double)x,  0.0, &alpha );
      bli_setsc( (double)y,  0.0, &beta );
    }
    else {
      // For syrk, both alpha and beta may be complex since, unlike herk,
      // C is symmetric in both the real and complex cases.
      int32_t ac = (int32_t)(x/0.8);
      int32_t bc = (int32_t)(y/1.0);
      bli_setsc( (double)x, (double)ac, &alpha );
      bli_setsc( (double)y, (double)bc, &beta );
    }
    // Randomize A.
    libblis_test_mobj_irandomize( params, &a );

    libblis_test_mobj_irandomize( params, &c );
  }

  bli_mksymm( &c );
  bli_mktrim( &c );

  // Save C and set its structure and uplo properties.
  bli_obj_set_struc( BLIS_SYMMETRIC, &c_save );
  bli_obj_set_uplo( uploc, &c_save );
  bli_copym( &c, &c_save );

  bli_mksymm( &c_save );
  bli_mktrim( &c_save );

  bli_copym( &a, &aa );

  // Apply the remaining parameters.
  bli_obj_set_conjtrans( transa, &a );

  bli_obj_set_conjtrans( transa, &aa );

  libblis_api_syrk(params, iface, &alpha, &aa, &beta, &c, datatype);

#ifndef __GTEST_VALGRIND_TEST__
  if(params->bitrp) {
    obj_t r;

    libblis_test_mobj_create( params, datatype, BLIS_NO_TRANSPOSE,
                                                 sc_str[0], m, m, &r );
    resid = libblis_test_bitrp_syrk( params, iface, &alpha, &a, &beta,
                                                &c, &c_save, &r, datatype);
    bli_obj_free( &r );
  }
  else {
    resid = libblis_ref_syrk(params, &alpha, &a, &beta, &c, &c_save, datatype);
  }
#endif

	 // Zero out performance and residual if output matrix is empty.
	 libblis_test_check_empty_problem( &c, &resid );

  // Free the test objects.
  libblis_test_obj_free( &a );
  libblis_test_obj_free( &aa );
  libblis_test_obj_free( &c );
  libblis_test_obj_free( &c_save );

  return abs(resid);
}

void libblis_test_syrk_impl(
  iface_t   iface,
  obj_t*    alpha,
  obj_t*    a,
  obj_t*    beta,
  obj_t*    c
) {

 	switch ( iface ) {
    case BLIS_TEST_SEQ_FRONT_END:
      bli_syrk( alpha, a, beta, c );
      break;

    default:
      libblis_test_printf_error( "Invalid interface type.\n" );
  }
}

double libblis_test_syrk_check (
  test_params_t* params,
  obj_t*         alpha,
  obj_t*         a,
  obj_t*         beta,
  obj_t*         c,
  obj_t*         c_orig
){
  num_t  dt      = bli_obj_dt( c );
  num_t  dt_real = bli_obj_dt_proj_to_real( c );

  dim_t  m       = bli_obj_length( c );
  dim_t  k       = bli_obj_width_after_trans( a );

  obj_t  at;
  obj_t  norm;
  obj_t  t, v, w, z;

  double junk;
  double resid = 0.0;
  //
  // Pre-conditions:
  // - a is randomized.
  // - c_orig is randomized and symmetric.
  // Note:
  // - alpha and beta should have non-zero imaginary components in the
  //   complex cases in order to more fully exercise the implementation.
  //
  // Under these conditions, we assume that the implementation for
  //
  //   C := beta * C_orig + alpha * transa(A) * transa(A)^T
  //
  // is functioning correctly if
  //
  //   normfv( v - z )
  //
  // is negligible, where
  //
  //   v = C * t
  //   z = ( beta * C_orig + alpha * transa(A) * transa(A)^T ) * t
  //     = beta * C_orig * t + alpha * transa(A) * transa(A)^T * t
  //     = beta * C_orig * t + alpha * transa(A) * w
  //     = beta * C_orig * t + z
  //

  bli_obj_alias_with_trans( BLIS_TRANSPOSE, a, &at );

  bli_obj_scalar_init_detached( dt_real, &norm );

  bli_obj_create( dt, m, 1, 0, 0, &t );
  bli_obj_create( dt, m, 1, 0, 0, &v );
  bli_obj_create( dt, k, 1, 0, 0, &w );
  bli_obj_create( dt, m, 1, 0, 0, &z );

  libblis_test_vobj_randomize( params, TRUE, &t );

  bli_symv( &BLIS_ONE, c, &t, &BLIS_ZERO, &v );

  bli_gemv( &BLIS_ONE, &at, &t, &BLIS_ZERO, &w );
  bli_gemv( alpha, a, &w, &BLIS_ZERO, &z );
  bli_symv( beta, c_orig, &t, &BLIS_ONE, &z );

  bli_subv( &z, &v );
  bli_normfv( &v, &norm );
  bli_getsc( &norm, &resid, &junk );

  bli_obj_free( &t );
  bli_obj_free( &v );
  bli_obj_free( &w );
  bli_obj_free( &z );

  return resid;
}
