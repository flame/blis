#include "blis_test.h"
#include "blis_utils.h"
#include "test_syr2k.h"

using namespace std;

// Local prototypes.
void libblis_test_syr2k_impl(
  iface_t   iface,
  obj_t*    alpha,
  obj_t*    a,
  obj_t*    b,
  obj_t*    beta,
  obj_t*    c
);

double libblis_test_syr2k_check(
  test_params_t* params,
  obj_t*         alpha,
  obj_t*         a,
  obj_t*         b,
  obj_t*         beta,
  obj_t*         c,
  obj_t*         c_orig
);

double cblas_syr2k(
  uplo_t     uploc,
  trans_t    trans,
  f77_int    mm,
  f77_int    kk,
  obj_t*     alpha,
  obj_t*     a,
  f77_int    lda,
  obj_t*     b,
  f77_int    ldb,
  obj_t*     beta,
  obj_t*     c,
  f77_int    ldc,
  num_t      dt
){
  enum CBLAS_ORDER     cblas_order;
  enum CBLAS_UPLO      cblas_uploc;
  enum CBLAS_TRANSPOSE cblas_trans;

  if( bli_is_trans( trans ) )
    cblas_trans = CblasTrans;
  else
    cblas_trans = CblasNoTrans;

  if ( bli_obj_row_stride( c ) == 1 )
    cblas_order = CblasColMajor;
  else
    cblas_order = CblasRowMajor;

  if(bli_is_upper(uploc))
    cblas_uploc = CblasUpper;
  else
    cblas_uploc = CblasLower;

  switch( dt )  {
    case BLIS_FLOAT :
    {
      float*   alphap = (float*) bli_obj_buffer( alpha );
      float*   ap     = (float*) bli_obj_buffer( a );
      float*   bp     = (float*) bli_obj_buffer( b );
      float*   betap  = (float*) bli_obj_buffer( beta );
      float*   cp     = (float*) bli_obj_buffer( c );
      cblas_ssyr2k( cblas_order, cblas_uploc, cblas_trans, mm, kk, *alphap,
                                     ap, lda, bp, ldb, *betap, cp, ldc );
      break;
    }
    case BLIS_DOUBLE :
    {
      double*  alphap = (double*) bli_obj_buffer( alpha );
      double*  ap     = (double*) bli_obj_buffer( a );
      double*  bp     = (double*) bli_obj_buffer( b );
      double*  betap  = (double*) bli_obj_buffer( beta );
      double*  cp     = (double*) bli_obj_buffer( c );
      cblas_dsyr2k( cblas_order, cblas_uploc, cblas_trans, mm, kk, *alphap,
                                     ap, lda, bp, ldb, *betap, cp, ldc );
      break;
    }
    case BLIS_SCOMPLEX :
    {
      scomplex*  alphap = (scomplex*) bli_obj_buffer( alpha );
      scomplex*  ap     = (scomplex*) bli_obj_buffer( a );
      scomplex*  bp     = (scomplex*) bli_obj_buffer( b );
      scomplex*  betap  = (scomplex*) bli_obj_buffer( beta );
      scomplex*  cp     = (scomplex*) bli_obj_buffer( c );
      cblas_csyr2k( cblas_order, cblas_uploc, cblas_trans, mm, kk, alphap,
                                      ap, lda, bp, ldb, betap, cp, ldc );
      break;
    }
    case BLIS_DCOMPLEX :
    {
      dcomplex*  alphap = (dcomplex*) bli_obj_buffer( alpha );
      dcomplex*  ap     = (dcomplex*) bli_obj_buffer( a );
      dcomplex*  bp     = (dcomplex*) bli_obj_buffer( b );
      dcomplex*  betap  = (dcomplex*) bli_obj_buffer( beta );
      dcomplex*  cp     = (dcomplex*) bli_obj_buffer( c );
      cblas_zsyr2k( cblas_order, cblas_uploc, cblas_trans, mm, kk, alphap,
                                      ap, lda, bp, ldb, betap, cp, ldc );
      break;
    }
    default :
      bli_check_error_code( BLIS_INVALID_DATATYPE );
  }

  return 0;
}

double blas_syr2k(
  uplo_t     uploc,
  trans_t    trans,
  f77_int    mm,
  f77_int    kk,
  obj_t*     alpha,
  obj_t*     a,
  f77_int    lda,
  obj_t*     b,
  f77_int    ldb,
  obj_t*     beta,
  obj_t*     c,
  f77_int    ldc,
  num_t      dt
){

  f77_char f77_uploc;
  f77_char f77_trans;

  bli_param_map_blis_to_netlib_uplo( uploc, &f77_uploc );
  bli_param_map_blis_to_netlib_trans( trans, &f77_trans );

  switch( dt )  {
    case BLIS_FLOAT :
    {
      float*   alphap = (float*) bli_obj_buffer( alpha );
      float*   ap     = (float*) bli_obj_buffer( a );
      float*   bp     = (float*) bli_obj_buffer( b );
      float*   betap  = (float*) bli_obj_buffer( beta );
      float*   cp     = (float*) bli_obj_buffer( c );
      ssyr2k_( &f77_uploc, &f77_trans, &mm, &kk, alphap, ap, (f77_int*)&lda,
                             bp, (f77_int*)&ldb, betap, cp, (f77_int*)&ldc );
      break;
    }
    case BLIS_DOUBLE :
    {
      double*  alphap = (double*) bli_obj_buffer( alpha );
      double*  ap     = (double*) bli_obj_buffer( a );
      double*  bp     = (double*) bli_obj_buffer( b );
      double*  betap  = (double*) bli_obj_buffer( beta );
      double*  cp     = (double*) bli_obj_buffer( c );
      dsyr2k_( &f77_uploc, &f77_trans, &mm, &kk, alphap, ap,(f77_int*)&lda,
                             bp, (f77_int*)&ldb, betap, cp, (f77_int*)&ldc );
      break;
    }
    case BLIS_SCOMPLEX :
    {
      scomplex*  alphap = (scomplex*) bli_obj_buffer( alpha );
      scomplex*  ap     = (scomplex*) bli_obj_buffer( a );
      scomplex*  bp     = (scomplex*) bli_obj_buffer( b );
      scomplex*  betap  = (scomplex*) bli_obj_buffer( beta );
      scomplex*  cp     = (scomplex*) bli_obj_buffer( c );
      csyr2k_( &f77_uploc, &f77_trans, &mm, &kk, alphap, ap, (f77_int*)&lda,
                            bp, (f77_int*)&ldb, betap, cp, (f77_int*)&ldc );
      break;
    }
    case BLIS_DCOMPLEX :
    {
      dcomplex*  alphap = (dcomplex*) bli_obj_buffer( alpha );
      dcomplex*  ap     = (dcomplex*) bli_obj_buffer( a );
      dcomplex*  bp     = (dcomplex*) bli_obj_buffer( b );
      dcomplex*  betap  = (dcomplex*) bli_obj_buffer( beta );
      dcomplex*  cp     = (dcomplex*) bli_obj_buffer( c );
      zsyr2k_( &f77_uploc, &f77_trans, &mm, &kk, alphap, ap, (f77_int*)&lda,
                            bp, (f77_int*)&ldb, betap, cp, (f77_int*)&ldc );
      break;
    }
    default :
      bli_check_error_code( BLIS_INVALID_DATATYPE );
  }
  return 0;
}

void libblis_api_syr2k(
  test_params_t* params,
  iface_t        iface,
  obj_t*         alpha,
  obj_t*         a,
  obj_t*         b,
  obj_t*         beta,
  obj_t*         c,
  num_t          dt
){
  if(params->api == API_BLIS) {
    libblis_test_syr2k_impl( iface, alpha, a, b, beta, c );
  }
  else { /*CLBAS  || BLAS */
    uplo_t  uploc  = bli_obj_uplo( c );
    dim_t  m       = bli_obj_length( c );
    dim_t  k       = bli_obj_width_after_trans( a );
    trans_t trans  = bli_obj_onlytrans_status( a );
    f77_int lda, ldb, ldc;

   if( bli_obj_row_stride( c ) == 1 ) {
      lda    = bli_obj_col_stride( a );
      ldb    = bli_obj_col_stride( b );
      ldc    = bli_obj_col_stride( c );
    } else {
      lda    = bli_obj_row_stride( a );
      ldb    = bli_obj_row_stride( b );
      ldc    = bli_obj_row_stride( c );
    }

    if(params->ldf == 1) {
      lda = lda + params->ld[0];
      ldb = ldb + params->ld[1];
      ldc = ldc + params->ld[2];
    }

    if(params->api == API_CBLAS) {
      cblas_syr2k( uploc, trans, m, k, alpha, a, lda, b, ldb, beta, c, ldc, dt );
    } else {
      if( bli_obj_row_stride( c ) == 1 ) {
        blas_syr2k( uploc, trans, m, k, alpha, a, lda, b, ldb, beta, c, ldc, dt );
      }
      else {
        if( uploc == BLIS_UPPER)
          uploc = BLIS_LOWER;
        else if(uploc == BLIS_LOWER)
          uploc = BLIS_UPPER;

        if( trans == BLIS_NO_TRANSPOSE)
          trans = BLIS_TRANSPOSE;
        else if(trans == BLIS_TRANSPOSE)
          trans = BLIS_NO_TRANSPOSE;

        blas_syr2k( uploc, trans, m, k, alpha, a, lda, b, ldb, beta, c, ldc, dt );
      }
    }
  }
  return ;
}

double libblis_ref_syr2k(
  test_params_t* params,
  obj_t*         alpha,
  obj_t*         a,
  obj_t*         b,
  obj_t*         beta,
  obj_t*         c,
  obj_t*         c_save,
  num_t          dt
) {

  double resid = 0.0;
  double *betap = (double *)bli_obj_buffer( beta );

  if ((params->nanf) && (*betap == 0)) {
    resid = libblis_check_nan_syr2k(c, dt );
  }
  else if((params->bitextf == 0) && (params->oruflw == BLIS_DEFAULT)) {
	   resid = libblis_test_syr2k_check( params, alpha, a, b, beta, c, c_save );
  }
  else {
    if(params->oruflw == BLIS_DEFAULT) {
	     resid = libblis_test_isyr2k_check( params, alpha, a, b, beta, c, c_save );
    }
    else {
      resid = libblis_test_matrix_check(params, c);
    }
  }
  return resid;
}

double libblis_test_bitrp_syr2k(
  test_params_t* params,
  iface_t        iface,
  obj_t*         alpha,
  obj_t*         a,
  obj_t*         b,
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
		  libblis_test_syr2k_impl( iface, alpha, a, b, beta, r );
    resid = libblis_test_bitrp_matrix(c, r, dt);
  }
  return resid;
}

double libblis_test_op_syr2k (
  test_params_t* params,
  iface_t        iface,
  char*          dc_str,
  char*          pc_str,
  char*          sc_str,
  tensor_t*      dim,
  atom_t         alpv,
  atom_t         betv
) {
  num_t   datatype;
  dim_t   m, k;
  uplo_t  uploc;
  trans_t trans;
  obj_t   alpha, a, b, beta, c;
  obj_t   c_save;
  double  resid = 0.0;
  obj_t   aa, bb;

  // Use the datatype of the first char in the datatype combination string.
  bli_param_map_char_to_blis_dt( dc_str[0], &datatype );

  // Map the dimension specifier to actual dimensions.
  m = dim->m;
  k = dim->n;

  // Map parameter characters to BLIS constants.
  bli_param_map_char_to_blis_uplo( pc_str[0], &uploc );
  bli_param_map_char_to_syrk_trans( pc_str[1], &trans );

  // Create test scalars.
  bli_obj_scalar_init_detached( datatype, &alpha );
  bli_obj_scalar_init_detached( datatype, &beta );

  // Create test operands (vectors and/or matrices).
  libblis_test_mobj_create( params, datatype, trans,
                            sc_str[1], m, k, &a );
  libblis_test_mobj_create( params, datatype, trans,
                            sc_str[1], m, k, &aa );
  libblis_test_mobj_create( params, datatype, trans,
                            sc_str[2], m, k, &b );
  libblis_test_mobj_create( params, datatype, trans,
                            sc_str[2], m, k, &bb );
  libblis_test_mobj_create( params, datatype, BLIS_NO_TRANSPOSE,
                            sc_str[0], m, m, &c );
  libblis_test_mobj_create( params, datatype, BLIS_NO_TRANSPOSE,
                            sc_str[0], m, m, &c_save );

  // Set the structure and uplo properties of C.
  bli_obj_set_struc( BLIS_SYMMETRIC, &c );
  bli_obj_set_uplo( uploc, &c );

  // Set alpha and beta.
  // For syr2k, both alpha and beta may be complex since, unlike her2k,
  // C is symmetric in both the real and complex cases.
  if((params->bitextf == 0) && (params->oruflw == BLIS_DEFAULT)) {
    if ( bli_obj_is_real( &c ) )	{
      bli_setsc(  alpv.real,  0.0, &alpha );
      bli_setsc(  betv.real,  0.0, &beta );
    }
    else	{
      bli_setsc(  alpv.real,  (alpv.real/0.8), &alpha );
      bli_setsc(  betv.real,  0.0, &beta );
    }
    // Randomize A, B, and C, and save C.
    libblis_test_mobj_randomize( params, TRUE, &a );
    libblis_test_mobj_randomize( params, TRUE, &b );
    libblis_test_mobj_randomize( params, TRUE, &c );
  }
  else {
    int32_t x = (int32_t)1.0; 	 //alpv.real;
    int32_t y = (int32_t)1.0;    //betv.real;
    if ( bli_obj_is_real( &c ) )	{
      bli_setsc( (double)x,  0.0, &alpha );
      bli_setsc( (double)y,  0.0, &beta );
    }
    else	{
      int32_t ac = (int32_t)(x/0.8);
      int32_t bc = (int32_t)(y/1.0);
      bli_setsc( (double)x, (double)ac, &alpha );
      bli_setsc( (double)y, (double)bc, &beta );
    }
    libblis_test_mobj_irandomize( params, &a );
    libblis_test_mobj_irandomize( params, &b );
    libblis_test_mobj_irandomize( params, &c );
  }

  if ((params->nanf) && (betv.real == 0) ) {
    test_fillbuffmem(&c, datatype );
  }

  // Randomize A, make it densely symmetric, and zero the unstored triangle
  // to ensure the implementation is reads only from the stored region.
  bli_mksymm( &c );
  bli_mktrim( &c );

  // Save C and set its structure and uplo properties.
  bli_obj_set_struc( BLIS_SYMMETRIC, &c_save );
  bli_obj_set_uplo( uploc, &c_save );
  bli_copym( &c, &c_save );
  bli_mksymm( &c_save );
  bli_mktrim( &c_save );

  // Apply the remaining parameters.
  bli_copym( &a, &aa );
  bli_copym( &b, &bb );

  bli_obj_set_conjtrans( trans, &a );
  bli_obj_set_conjtrans( trans, &b );

  bli_obj_set_conjtrans( trans, &aa );
  bli_obj_set_conjtrans( trans, &bb );

  libblis_api_syr2k(params, iface, &alpha, &aa, &bb, &beta, &c, datatype );

#ifndef __GTEST_VALGRIND_TEST__
  if(params->bitrp) {
    obj_t r;
    libblis_test_mobj_create( params, datatype, BLIS_NO_TRANSPOSE,
                                                       sc_str[0], m, m, &r );

    resid = libblis_test_bitrp_syr2k( params, iface, &alpha, &a, &b,
                                            &beta, &c, &c_save, &r, datatype);
    bli_obj_free( &r );
  }
  else {
    resid = libblis_ref_syr2k(params, &alpha, &a, &b, &beta,
                                              &c, &c_save, datatype );
  }
#endif

  // Zero out performance and residual if output matrix is empty.
  libblis_test_check_empty_problem( &c, &resid );

  // Free the test objects.
  libblis_test_obj_free( &a );
  libblis_test_obj_free( &aa );
  libblis_test_obj_free( &b );
  libblis_test_obj_free( &bb );
  libblis_test_obj_free( &c );
  libblis_test_obj_free( &c_save );

  return abs(resid);
}

void libblis_test_syr2k_impl
     (
       iface_t   iface,
       obj_t*    alpha,
       obj_t*    a,
       obj_t*    b,
       obj_t*    beta,
       obj_t*    c
     )
{
  switch ( iface )
  {
   case BLIS_TEST_SEQ_FRONT_END:
       bli_syr2k( alpha, a, b, beta, c );
   break;

   default:
       libblis_test_printf_error( "Invalid interface type.\n" );
  }
}

double libblis_test_syr2k_check
     (
       test_params_t* params,
       obj_t*         alpha,
       obj_t*         a,
       obj_t*         b,
       obj_t*         beta,
       obj_t*         c,
       obj_t*         c_orig
     )
{
  num_t  dt      = bli_obj_dt( c );
  num_t  dt_real = bli_obj_dt_proj_to_real( c );

  dim_t  m       = bli_obj_length( c );
  dim_t  k       = bli_obj_width_after_trans( a );

  obj_t  at, bt;
  obj_t  norm;
  obj_t  t, v, w1, w2, z;

  double junk;
  double resid = 0.0;

  //
  // Pre-conditions:
  // - a is randomized.
  // - b is randomized.
  // - c_orig is randomized and symmetric.
  // Note:
  // - alpha and beta should have non-zero imaginary components in the
  //   complex cases in order to more fully exercise the implementation.
  //
  // Under these conditions, we assume that the implementation for
  //
  //   C := beta * C_orig + alpha * transa(A) * transb(B)^T + alpha * transb(B) * transa(A)^T
  //
  // is functioning correctly if
  //
  //   normfv( v - z )
  //
  // is negligible, where
  //
  //   v = C * t
  //   z = ( beta * C_orig + alpha * transa(A) * transb(B)^T + alpha * transb(B) * transa(A)^T ) * t
  //     = beta * C_orig * t + alpha * transa(A) * transb(B)^T * t + alpha * transb(B) * transa(A)^T * t
  //     = beta * C_orig * t + alpha * transa(A) * transb(B)^T * t + alpha * transb(B) * w2
  //     = beta * C_orig * t + alpha * transa(A) * w1              + alpha * transb(B) * w2
  //     = beta * C_orig * t + alpha * transa(A) * w1              + z
  //     = beta * C_orig * t + z
  //

  bli_obj_alias_with_trans( BLIS_TRANSPOSE, a, &at );
  bli_obj_alias_with_trans( BLIS_TRANSPOSE, b, &bt );

  bli_obj_scalar_init_detached( dt_real, &norm );

  bli_obj_create( dt, m, 1, 0, 0, &t );
  bli_obj_create( dt, m, 1, 0, 0, &v );
  bli_obj_create( dt, k, 1, 0, 0, &w1 );
  bli_obj_create( dt, k, 1, 0, 0, &w2 );
  bli_obj_create( dt, m, 1, 0, 0, &z );

  libblis_test_vobj_randomize( params, TRUE, &t );

  bli_symv( &BLIS_ONE, c, &t, &BLIS_ZERO, &v );

  bli_gemv( &BLIS_ONE, &at, &t, &BLIS_ZERO, &w2 );
  bli_gemv( &BLIS_ONE, &bt, &t, &BLIS_ZERO, &w1 );
  bli_gemv( alpha, a, &w1, &BLIS_ZERO, &z );
  bli_gemv( alpha, b, &w2, &BLIS_ONE, &z );
  bli_symv( beta, c_orig, &t, &BLIS_ONE, &z );

  bli_subv( &z, &v );
  bli_normfv( &v, &norm );
  bli_getsc( &norm, &resid, &junk );

  bli_obj_free( &t );
  bli_obj_free( &v );
  bli_obj_free( &w1 );
  bli_obj_free( &w2 );
  bli_obj_free( &z );

  return resid;
}

