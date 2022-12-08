#include "blis_test.h"
#include "blis_utils.h"
#include "test_her.h"

// Local prototypes.
void libblis_test_her_deps(
  thread_data_t* tdata,
  test_params_t* params,
  test_op_t*     op
);

void libblis_test_her_impl(
  iface_t   iface,
  obj_t*    alpha,
  obj_t*    x,
  obj_t*    a
);

double libblis_test_her_check(
  test_params_t* params,
  obj_t*         alpha,
  obj_t*         x,
  obj_t*         a,
  obj_t*         a_orig
);

void cblas_her(
  uplo_t  uploa,
  f77_int m,
  obj_t*  alpha,
  obj_t*  x,
  f77_int incx,
  obj_t*  a,
  f77_int lda,
  num_t   dt
){
 	enum CBLAS_UPLO  cblas_uplo;
	 enum CBLAS_ORDER cblas_order = CblasColMajor;

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
      cblas_ssyr(cblas_order, cblas_uplo, m, *alphap, xp, incx, ap, lda);
      break;
    }
    case BLIS_DOUBLE :
    {
      double*  alphap = (double*) bli_obj_buffer( alpha );
      double*  ap     = (double*) bli_obj_buffer( a );
      double*  xp     = (double*) bli_obj_buffer( x );
      cblas_dsyr(cblas_order, cblas_uplo, m, *alphap, xp, incx, ap, lda);
      break;
    }
    case BLIS_SCOMPLEX :
    {
      float*   alphap   = (float*) bli_obj_buffer( alpha );
      scomplex*  ap     = (scomplex*) bli_obj_buffer( a );
      scomplex*  xp     = (scomplex*) bli_obj_buffer( x );
      cblas_cher(cblas_order, cblas_uplo, m, *alphap, xp, incx, ap, lda);
      break;
    }
    case BLIS_DCOMPLEX :
    {
      double*  alphap   = (double*) bli_obj_buffer( alpha );
      dcomplex*  ap     = (dcomplex*) bli_obj_buffer( a );
      dcomplex*  xp     = (dcomplex*) bli_obj_buffer( x );
      cblas_zher(cblas_order, cblas_uplo, m, *alphap, xp, incx, ap, lda);
      break;
    }
    default:
      bli_check_error_code( BLIS_INVALID_DATATYPE );
  }
}

void blas_her(
  f77_char f77_uploa,
  f77_int m,
  obj_t*  alpha,
  obj_t*  x,
  f77_int incx,
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
      ssyr_(&f77_uploa, &m, alphap, xp, &incx, ap, (f77_int*)&lda );
      break;
    }
    case BLIS_DOUBLE :
    {
      double*  alphap = (double*) bli_obj_buffer( alpha );
      double*  ap     = (double*) bli_obj_buffer( a );
      double*  xp     = (double*) bli_obj_buffer( x );
      dsyr_(&f77_uploa, &m, alphap, xp, &incx, ap, (f77_int*)&lda );
      break;
    }
    case BLIS_SCOMPLEX :
    {
      float*   alphap   = (float*) bli_obj_buffer( alpha );
      scomplex*  ap     = (scomplex*) bli_obj_buffer( a );
      scomplex*  xp     = (scomplex*) bli_obj_buffer( x );
      cher_(&f77_uploa, &m, alphap, xp, &incx, ap, (f77_int*)&lda );
      break;
    }
    case BLIS_DCOMPLEX :
    {
      double*  alphap   = (double*) bli_obj_buffer( alpha );
      dcomplex*  ap     = (dcomplex*) bli_obj_buffer( a );
      dcomplex*  xp     = (dcomplex*) bli_obj_buffer( x );
      zher_(&f77_uploa, &m, alphap, xp, &incx, ap, (f77_int*)&lda );
      break;
    }
    default:
      bli_check_error_code( BLIS_INVALID_DATATYPE );
  }
}

void libblis_api_her(
  test_params_t* params,
  iface_t        iface,
  obj_t*         alpha,
  obj_t*         x,
  obj_t*         a,
  num_t          dt
){

  if(params->api == API_BLIS) {
    libblis_test_her_impl( iface, alpha, x, a );
  }
  else { /*CLBAS  || BLAS */
    uplo_t  uploa = bli_obj_uplo( a );
    f77_int  mm   = bli_obj_length( a );
    f77_int  incx = bli_obj_vector_inc( x );
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
     	cblas_her(uploa, mm, alpha, x, incx, a, lda, dt );
    }
    else { /**/
      f77_char f77_uploa;
      if ( bli_obj_row_stride( a ) == 1 ){
        bli_param_map_blis_to_netlib_uplo( uploa, &f77_uploa );
        blas_her(f77_uploa, mm, alpha, x, incx, a, lda, dt );
      }
      else {
        if( uploa == BLIS_UPPER)
          uploa = BLIS_LOWER;
        else if(uploa == BLIS_LOWER)
          uploa = BLIS_UPPER;

        conjugate_tensor(x, dt);
        bli_param_map_blis_to_netlib_uplo( uploa, &f77_uploa );
        blas_her(f77_uploa, mm, alpha, x, incx, a, lda, dt );
      }
    }
  }
  return ;
}

double libblis_ref_her(
  test_params_t* params,
  obj_t*         alpha,
  obj_t*         x,
  obj_t*         a,
  obj_t*         a_orig
) {
  double resid = 0.0;

  if((params->bitextf == 0) && (params->oruflw == BLIS_DEFAULT)) {
    resid = libblis_test_her_check( params, alpha, x, a, a_orig );
  }
  else {
    if(params->oruflw == BLIS_DEFAULT) {
      resid = libblis_test_iher_check( params, alpha, x, a, a_orig );
    }
    else {
      resid = libblis_test_matrix_check(params, a);
    }
  }
  return resid;
}

double libblis_test_bitrp_her(
  test_params_t* params,
  iface_t        iface,
  obj_t*         alpha,
  obj_t*         x,
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
    bli_mktrim( r );
    libblis_test_her_impl( iface, alpha, x, r );
    resid = libblis_test_bitrp_matrix(a, r, dt);
  }
  return resid;
}

double libblis_test_op_her (
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
  conj_t       conjx;
  obj_t        alpha, x, a;
  obj_t        a_save;
  double       resid = 0.0;
  obj_t        xx;

  // Use the datatype of the first char in the datatype combination string.
  bli_param_map_char_to_blis_dt( dc_str[0], &datatype );

  // Map the dimension specifier to an actual dimension.
  m = dim->m;

  // Map parameter characters to BLIS constants.
  bli_param_map_char_to_blis_uplo( pc_str[0], &uploa );
  bli_param_map_char_to_blis_conj( pc_str[1], &conjx );

  // Create test scalars.
  bli_obj_scalar_init_detached( datatype, &alpha );

  // Create test operands (vectors and/or matrices).
  libblis_test_vobj_create( params, datatype,
                            sc_str[0], m,    &x );
  libblis_test_vobj_create( params, datatype,
                            sc_str[0], m,    &xx );
  libblis_test_mobj_create( params, datatype, BLIS_NO_TRANSPOSE,
                            sc_str[1], m, m, &a );
  libblis_test_mobj_create( params, datatype, BLIS_NO_TRANSPOSE,
                            sc_str[1], m, m, &a_save );

  // Set the structure and uplo properties of A.
  bli_obj_set_struc( BLIS_HERMITIAN, &a );
  bli_obj_set_uplo( uploa, &a );

  // Set alpha.
  if((params->bitextf == 0) && (params->oruflw == BLIS_DEFAULT)) {
    bli_setsc( alpv.real, 0.0, &alpha );
    // Randomize x.
    libblis_test_vobj_randomize( params, TRUE, &x );
    libblis_test_mobj_randomize( params, TRUE, &a );
  }
  else{
    int32_t xx = (int32_t)alpv.real;
    bli_setsc( (double)xx, (double)0.0, &alpha );
    // Randomize x.
    libblis_test_vobj_irandomize( params, &x );
    libblis_test_mobj_irandomize( params, &a );
  }

  // Randomize A, make it densely Hermitian, and zero the unstored triangle
  // to ensure the implementation is reads only from the stored region.
  bli_mkherm( &a );
  bli_mktrim( &a );

  // Save A and set its structure and uplo properties.
  bli_obj_set_struc( BLIS_HERMITIAN, &a_save );
  bli_obj_set_uplo( uploa, &a_save );
  bli_copym( &a, &a_save );
  bli_mktrim( &a_save );

  // Apply the remaining parameters.
  bli_obj_set_conj( conjx, &x );

  bli_copyv( &x, &xx );

  libblis_api_her(params, iface, &alpha, &xx, &a, datatype );

#ifndef __GTEST_VALGRIND_TEST__
  if(params->bitrp) {
    obj_t r;

    libblis_test_mobj_create( params, datatype, BLIS_NO_TRANSPOSE,
                                                         sc_str[1], m, m, &r );
    bli_obj_set_struc( BLIS_HERMITIAN, &r );
    bli_obj_set_uplo( uploa, &r );

    resid = libblis_test_bitrp_her( params, iface, &alpha, &x, &a, &a_save,
                                                               &r, datatype);
    bli_obj_free( &r );
  }
  else {
    resid = libblis_ref_her( params, &alpha, &x, &a, &a_save );
  }
#endif

  // Zero out performance and residual if output matrix is empty.
  libblis_test_check_empty_problem( &a, &resid );

  // Free the test objects.
  libblis_test_obj_free( &x );
  libblis_test_obj_free( &xx );
  libblis_test_obj_free( &a );
  libblis_test_obj_free( &a_save );

  return abs(resid);
}

void libblis_test_her_impl(
  iface_t   iface,
  obj_t*    alpha,
  obj_t*    x,
  obj_t*    a
){
  switch ( iface )
  {
    case BLIS_TEST_SEQ_FRONT_END:
      bli_her( alpha, x, a );
      break;

    default:
      libblis_test_printf_error( "Invalid interface type.\n" );
  }
}

double libblis_test_her_check(
  test_params_t* params,
  obj_t*         alpha,
  obj_t*         x,
  obj_t*         a,
  obj_t*         a_orig
){
  num_t  dt      = bli_obj_dt( a );
  num_t  dt_real = bli_obj_dt_proj_to_real( a );

  dim_t  m_a     = bli_obj_length( a );

  obj_t  xh, t, v, w;
  obj_t  rho, norm;

  double junk;
  double resid = 0.0;

  //
  // Pre-conditions:
  // - x is randomized.
  // - a is randomized and Hermitian.
  // Note:
  // - alpha must be real-valued.
  //
  // Under these conditions, we assume that the implementation for
  //
  //   A := A_orig + alpha * conjx(x) * conjx(x)^H
  //
  // is functioning correctly if
  //
  //   normfv( v - w )
  //
  // is negligible, where
  //
  //   v = A * t
  //   w = ( A_orig + alpha * conjx(x) * conjx(x)^H ) * t
  //     =   A_orig * t + alpha * conjx(x) * conjx(x)^H * t
  //     =   A_orig * t + alpha * conjx(x) * rho
  //     =   A_orig * t + w
  //

  bli_mkherm( a );
  bli_mkherm( a_orig );
  bli_obj_set_struc( BLIS_GENERAL, a );
  bli_obj_set_struc( BLIS_GENERAL, a_orig );
  bli_obj_set_uplo( BLIS_DENSE, a );
  bli_obj_set_uplo( BLIS_DENSE, a_orig );

  bli_obj_scalar_init_detached( dt,      &rho );
  bli_obj_scalar_init_detached( dt_real, &norm );

  bli_obj_create( dt, m_a, 1, 0, 0, &t );
  bli_obj_create( dt, m_a, 1, 0, 0, &v );
  bli_obj_create( dt, m_a, 1, 0, 0, &w );

  bli_obj_alias_with_conj( BLIS_CONJUGATE, x, &xh );

  libblis_test_vobj_randomize( params, TRUE, &t );

  bli_gemv( &BLIS_ONE, a, &t, &BLIS_ZERO, &v );

  bli_dotv( &xh, &t, &rho );
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