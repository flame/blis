#include "blis_test.h"
#include "blis_utils.h"
#include "test_symm.h"

using namespace std;

// Local prototypes.
void libblis_test_symm_deps (
  thread_data_t* tdata,
  test_params_t* params,
  test_op_t*     op
);

void libblis_test_symm_impl(
  iface_t   iface,
  side_t    side,
  obj_t*    alpha,
  obj_t*    a,
  obj_t*    b,
  obj_t*    beta,
  obj_t*    c
);

double libblis_test_symm_check(
  test_params_t* params,
  side_t         side,
  obj_t*         alpha,
  obj_t*         a,
  obj_t*         b,
  obj_t*         beta,
  obj_t*         c,
  obj_t*         c_orig
);

double cblas_symm(
  side_t     side,
  uplo_t     uploa,
  f77_int    mm,
  f77_int    nn,
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
  enum CBLAS_UPLO      cblas_uplo;
  enum CBLAS_SIDE      cblas_side;

  if ( bli_obj_row_stride( c ) == 1 )
    cblas_order = CblasColMajor;
  else
    cblas_order = CblasRowMajor;

  if(bli_is_upper(uploa))
    cblas_uplo = CblasUpper;
  else
    cblas_uplo = CblasLower;

  if(bli_is_left(side))
    cblas_side = CblasLeft;
  else
    cblas_side = CblasRight;

  switch( dt )  {
    case BLIS_FLOAT :
    {
      float*   alphap = (float*) bli_obj_buffer( alpha );
      float*   ap     = (float*) bli_obj_buffer( a );
      float*   bp     = (float*) bli_obj_buffer( b );
      float*   betap  = (float*) bli_obj_buffer( beta );
      float*   cp     = (float*) bli_obj_buffer( c );
      cblas_ssymm( cblas_order, cblas_side, cblas_uplo, mm, nn, *alphap,
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
      cblas_dsymm( cblas_order, cblas_side, cblas_uplo, mm, nn, *alphap,
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
      cblas_csymm( cblas_order, cblas_side, cblas_uplo, mm, nn, alphap,
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
      cblas_zsymm( cblas_order, cblas_side, cblas_uplo, mm, nn, alphap,
                                      ap, lda, bp, ldb, betap, cp, ldc );
      break;
    }
    default :
      bli_check_error_code( BLIS_INVALID_DATATYPE );
  }

  return 0;
}

double blas_symm(
  side_t     side,
  uplo_t     uploa,
  f77_int    mm,
  f77_int    nn,
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

  f77_char f77_side;
  f77_char f77_uploa;

  bli_param_map_blis_to_netlib_side( side, &f77_side );
  bli_param_map_blis_to_netlib_uplo( uploa, &f77_uploa );

  switch( dt )  {
    case BLIS_FLOAT :
    {
      float*   alphap = (float*) bli_obj_buffer( alpha );
      float*   ap     = (float*) bli_obj_buffer( a );
      float*   bp     = (float*) bli_obj_buffer( b );
      float*   betap  = (float*) bli_obj_buffer( beta );
      float*   cp     = (float*) bli_obj_buffer( c );
      ssymm_( &f77_side, &f77_uploa, &mm, &nn, alphap, ap, (f77_int*)&lda,
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
      dsymm_( &f77_side, &f77_uploa, &mm, &nn, alphap, ap, (f77_int*)&lda,
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
      csymm_( &f77_side, &f77_uploa, &mm, &nn, alphap, ap, (f77_int*)&lda,
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
      zsymm_( &f77_side, &f77_uploa, &mm, &nn, alphap, ap, (f77_int*)&lda,
                           bp, (f77_int*)&ldb, betap, cp, (f77_int*)&ldc );
      break;
    }
    default :
      bli_check_error_code( BLIS_INVALID_DATATYPE );
  }
  return 0;
}

void libblis_api_symm(
  test_params_t* params,
  iface_t        iface,
  side_t         side,
  obj_t*         alpha,
  obj_t*         a,
  obj_t*         b,
  obj_t*         beta,
  obj_t*         c,
  num_t          dt
){
  if(params->api == API_BLIS) {
		  libblis_test_symm_impl( iface, side, alpha, a, b, beta, c );
  }
  else { /*CLBAS  || BLAS */
    uplo_t  uploa  = bli_obj_uplo( a );
    f77_int mm     = bli_obj_length( c );
    f77_int nn     = bli_obj_width( c );
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
      cblas_symm( side, uploa, mm, nn, alpha, a, lda, b, ldb, beta, c, ldc, dt );
    } else { /**/
      if( bli_obj_row_stride( c ) == 1 ) {
        blas_symm( side, uploa, mm, nn, alpha, a, lda, b, ldb, beta, c, ldc, dt );
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

        blas_symm( side, uploa, nn, mm, alpha, a, lda, b, ldb, beta, c, ldc, dt );
      }
    }
  }
  return ;
}

double libblis_ref_symm(
  test_params_t* params,
  side_t         side,
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
    resid = libblis_check_nan_symm(c, dt );
  }
  else if((params->bitextf == 0) && (params->oruflw == BLIS_DEFAULT)) {
	   resid = libblis_test_symm_check( params, side, alpha, a, b, beta, c, c_save );
  }
  else {
    if(params->oruflw == BLIS_DEFAULT) {
	     resid = libblis_test_isymm_check( params, side, alpha, a, b, beta, c, c_save );
    }
    else {
      resid = libblis_test_matrix_check(params, c);
    }
  }
  return resid;
}

double libblis_test_bitrp_symm(
  test_params_t* params,
  iface_t        iface,
  side_t         side,
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
		  libblis_test_symm_impl( iface, side, alpha, a, b, beta, r );
    resid = libblis_test_bitrp_matrix(c, r, dt);
  }
  return resid;
}

double libblis_test_op_symm (
  test_params_t* params,
  iface_t        iface,
  char*          dc_str,
  char*          pc_str,
  char*          sc_str,
  tensor_t*      dim,
  atom_t         alpv,
  atom_t         betv
) {
  num_t        datatype;
  dim_t        m, n;
  dim_t        mn_side;
  side_t       side;
  uplo_t       uploa;
  obj_t        alpha, a, b, beta, c;
  obj_t        c_save;
  double       resid = 0.0;

  // Use the datatype of the first char in the datatype combination string.
  bli_param_map_char_to_blis_dt( dc_str[0], &datatype );

  // Map the dimension specifier to actual dimensions.
  m = dim->m;
  n = dim->n;

  // Map parameter characters to BLIS constants.
  bli_param_map_char_to_blis_side( pc_str[0], &side );
  bli_param_map_char_to_blis_uplo( pc_str[1], &uploa );

  // Create test scalars.
  bli_obj_scalar_init_detached( datatype, &alpha );
  bli_obj_scalar_init_detached( datatype, &beta );

   // Create test operands (vectors and/or matrices).
   bli_set_dim_with_side( side, m, n, &mn_side );
   libblis_test_mobj_create( params, datatype, BLIS_NO_TRANSPOSE,
                             sc_str[1], mn_side, mn_side, &a );
   libblis_test_mobj_create( params, datatype, BLIS_NO_TRANSPOSE,
                             sc_str[2], m,       n,       &b );
   libblis_test_mobj_create( params, datatype, BLIS_NO_TRANSPOSE,
                             sc_str[0], m,       n,       &c );
   libblis_test_mobj_create( params, datatype, BLIS_NO_TRANSPOSE,
                             sc_str[0], m,       n,       &c_save );

   // Set the structure and uplo properties of A.
   bli_obj_set_struc( BLIS_SYMMETRIC, &a );
   bli_obj_set_uplo( uploa, &a );

  // Set alpha and beta.
  if((params->bitextf == 0) && (params->oruflw == BLIS_DEFAULT)) {
    if ( bli_obj_is_real( &c ) )	{
      bli_setsc(  alpv.real,  0.0, &alpha );
      bli_setsc(  betv.real,  0.0, &beta );
    }
    else	{
      bli_setsc(  alpv.real,  (alpv.real/0.8), &alpha );
      bli_setsc(  betv.real,  (betv.real/1.2), &beta );
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
  // to ensure the implementation reads only from the stored region.
  bli_mksymm( &a );
  bli_mktrim( &a );

  //Copy c to c_save
  bli_copym( &c, &c_save );

  libblis_api_symm(params, iface, side, &alpha, &a, &b, &beta, &c, datatype );

#ifndef __GTEST_VALGRIND_TEST__
  if(params->bitrp) {
    obj_t r;

    libblis_test_mobj_create( params, datatype, BLIS_NO_TRANSPOSE,
                                                sc_str[0], m, n, &r );
    resid = libblis_test_bitrp_symm( params, iface, side,&alpha, &a, &b,
                                            &beta, &c, &c_save, &r, datatype);
    bli_obj_free( &r );
  }
  else {
    resid = libblis_ref_symm(params, side, &alpha, &a, &b, &beta,
                                              &c, &c_save, datatype );
  }
#endif

  // Zero out performance and residual if output matrix is empty.
  libblis_test_check_empty_problem( &c, &resid );

  // Free the test objects.
  libblis_test_obj_free( &a );
  libblis_test_obj_free( &b );
  libblis_test_obj_free( &c );
  libblis_test_obj_free( &c_save );

  return abs(resid);
}

void libblis_test_symm_impl (
  iface_t   iface,
  side_t    side,
  obj_t*    alpha,
  obj_t*    a,
  obj_t*    b,
  obj_t*    beta,
  obj_t*    c
){
  switch ( iface )	{
    case BLIS_TEST_SEQ_FRONT_END:
	    bli_symm( side, alpha, a, b, beta, c );
  break;

  default:
    libblis_test_printf_error( "Invalid interface type.\n" );
  }
}

double libblis_test_symm_check (
  test_params_t* params,
  side_t         side,
  obj_t*         alpha,
  obj_t*         a,
  obj_t*         b,
  obj_t*         beta,
  obj_t*         c,
  obj_t*         c_orig
) {
  num_t  dt      = bli_obj_dt( c );
  num_t  dt_real = bli_obj_dt_proj_to_real( c );

  dim_t  m       = bli_obj_length( c );
  dim_t  n       = bli_obj_width( c );

  obj_t  norm;
  obj_t  t, v, w, z;

  double junk;
  double resid = 0.0;
  //
  // Pre-conditions:
  // - a is randomized and symmetric.
  // - b is randomized.
  // - c_orig is randomized.
  // Note:
  // - alpha and beta should have non-zero imaginary components in the
  //   complex cases in order to more fully exercise the implementation.
  //
  // Under these conditions, we assume that the implementation for
  //
  //   C := beta * C_orig + alpha * conja(A) * transb(B)    (side = left)
  //   C := beta * C_orig + alpha * transb(B) * conja(A)    (side = right)
  //
  // is functioning correctly if
  //
  //   normfv( v - z )
  //
  // is negligible, where
  //
  //   v = C * t
  //
  //   z = ( beta * C_orig + alpha * conja(A) * transb(B) ) * t     (side = left)
  //     = beta * C_orig * t + alpha * conja(A) * transb(B) * t
  //     = beta * C_orig * t + alpha * conja(A) * w
  //     = beta * C_orig * t + z
  //
  //   z = ( beta * C_orig + alpha * transb(B) * conja(A) ) * t     (side = right)
  //     = beta * C_orig * t + alpha * transb(B) * conja(A) * t
  //     = beta * C_orig * t + alpha * transb(B) * w
  //     = beta * C_orig * t + z

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

  bli_gemv( &BLIS_ONE, c, &t, &BLIS_ZERO, &v );

  if ( bli_is_left( side ) )
  {
    bli_gemv( &BLIS_ONE, b, &t, &BLIS_ZERO, &w );
    bli_symv( alpha, a, &w, &BLIS_ZERO, &z );
  }
  else
  {
    bli_symv( &BLIS_ONE, a, &t, &BLIS_ZERO, &w );
    bli_gemv( alpha, b, &w, &BLIS_ZERO, &z );
  }

  bli_gemv( beta, c_orig, &t, &BLIS_ONE, &z );

  bli_subv( &z, &v );
  bli_normfv( &v, &norm );
  bli_getsc( &norm, &resid, &junk );

  bli_obj_free( &t );
  bli_obj_free( &v );
  bli_obj_free( &w );
  bli_obj_free( &z );

  return resid;
}