#include "blis_test.h"
#include "blis_utils.h"
#include "test_gemmt.h"

void libblis_test_gemmt_impl(
  iface_t iface,
  obj_t *alpha,
  obj_t *a,
  obj_t *b,
  obj_t *beta,
  obj_t *c
);

double libblis_test_gemmt_check(
  test_params_t *params,
  obj_t *alpha,
  obj_t *a,
  obj_t *b,
  obj_t *beta,
  obj_t *c,
  obj_t *c_orig
);

double cblas_gemmt(
  test_params_t* params,
  uplo_t         uploc,
  f77_int        n,
  f77_int        k,
  f77_int        lda,
  f77_int        ldb,
  f77_int        ldc,
  obj_t*         a,
  obj_t*         b,
  obj_t*         c,
  obj_t*         alpha,
  obj_t*         beta,
  num_t          dt,
  trans_t        transa,
  trans_t        transb
){
  enum CBLAS_ORDER     cblas_order;
  enum CBLAS_TRANSPOSE cblas_transa;
  enum CBLAS_TRANSPOSE cblas_transb;
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
    cblas_transa = CblasTrans;
  else if( bli_is_conjtrans( transa ) )
    cblas_transa = CblasConjTrans;
  else
    cblas_transa = CblasNoTrans;

  if( bli_is_trans( transb ) )
    cblas_transb = CblasTrans;
  else if( bli_is_conjtrans( transb ) )
    cblas_transb = CblasConjTrans;
  else
    cblas_transb = CblasNoTrans;

  switch( dt )  {
    case BLIS_FLOAT :
    {
      float*   alphap = (float*) bli_obj_buffer( alpha );
      float*   ap     = (float*) bli_obj_buffer( a );
      float*   bp     = (float*) bli_obj_buffer( b );
      float*   betap  = (float*) bli_obj_buffer( beta );
      float*   cp     = (float*) bli_obj_buffer( c );
      cblas_sgemmt( cblas_order, cblas_uplo, cblas_transa, cblas_transb,
   				              n, k, *alphap, ap, lda, bp, ldb, *betap, cp, ldc );
      break;
    }
    case BLIS_DOUBLE :
    {
      double*  alphap = (double*) bli_obj_buffer( alpha );
      double*  ap     = (double*) bli_obj_buffer( a );
      double*  bp     = (double*) bli_obj_buffer( b );
      double*  betap  = (double*) bli_obj_buffer( beta );
      double*  cp     = (double*) bli_obj_buffer( c );
      cblas_dgemmt( cblas_order, cblas_uplo, cblas_transa, cblas_transb,
   				              n, k, *alphap, ap, lda, bp, ldb, *betap, cp, ldc );
      break;
    }
    case BLIS_SCOMPLEX :
    {
      scomplex*  alphap = (scomplex*) bli_obj_buffer( alpha );
      scomplex*  ap     = (scomplex*) bli_obj_buffer( a );
      scomplex*  bp     = (scomplex*) bli_obj_buffer( b );
      scomplex*  betap  = (scomplex*) bli_obj_buffer( beta );
      scomplex*  cp     = (scomplex*) bli_obj_buffer( c );
      cblas_cgemmt( cblas_order, cblas_uplo, cblas_transa, cblas_transb,
   				              n, k, alphap, ap, lda, bp, ldb, betap, cp, ldc );
      break;
    }
    case BLIS_DCOMPLEX :
    {
      dcomplex*  alphap = (dcomplex*) bli_obj_buffer( alpha );
      dcomplex*  ap     = (dcomplex*) bli_obj_buffer( a );
      dcomplex*  bp     = (dcomplex*) bli_obj_buffer( b );
      dcomplex*  betap  = (dcomplex*) bli_obj_buffer( beta );
      dcomplex*  cp     = (dcomplex*) bli_obj_buffer( c );
      cblas_zgemmt( cblas_order, cblas_uplo, cblas_transa, cblas_transb,
   				              n, k, alphap, ap, lda, bp, ldb, betap, cp, ldc );
      break;
    }
    default :
      bli_check_error_code( BLIS_INVALID_DATATYPE );
  }

  return 0;
}

double blas_gemmt(
  f77_char f77_uploc,
  f77_int  n,
  f77_int  k,
  f77_int  lda,
  f77_int  ldb,
  f77_int  ldc,
  obj_t*   a,
  obj_t*   b,
  obj_t*   c,
  obj_t*   alpha,
  obj_t*   beta,
  num_t    dt,
  f77_char f77_transa,
  f77_char f77_transb
){

  switch( dt )  {
    case BLIS_FLOAT :
    {
      float*   alphap = (float*) bli_obj_buffer( alpha );
      float*   ap     = (float*) bli_obj_buffer( a );
      float*   bp     = (float*) bli_obj_buffer( b );
      float*   betap  = (float*) bli_obj_buffer( beta );
      float*   cp     = (float*) bli_obj_buffer( c );
      sgemmt_( &f77_uploc,	&f77_transa, &f77_transb, &n, &k, alphap, ap,
           (f77_int*)&lda, bp, (f77_int*)&ldb, betap, cp, (f77_int*)&ldc );
      break;
    }
    case BLIS_DOUBLE :
    {
      double*  alphap = (double*) bli_obj_buffer( alpha );
      double*  ap     = (double*) bli_obj_buffer( a );
      double*  bp     = (double*) bli_obj_buffer( b );
      double*  betap  = (double*) bli_obj_buffer( beta );
      double*  cp     = (double*) bli_obj_buffer( c );
      dgemmt_( &f77_uploc,	&f77_transa, &f77_transb, &n, &k, alphap, ap,
           (f77_int*)&lda, bp, (f77_int*)&ldb, betap, cp, (f77_int*)&ldc );
      break;
    }
    case BLIS_SCOMPLEX :
    {
      scomplex*  alphap = (scomplex*) bli_obj_buffer( alpha );
      scomplex*  ap     = (scomplex*) bli_obj_buffer( a );
      scomplex*  bp     = (scomplex*) bli_obj_buffer( b );
      scomplex*  betap  = (scomplex*) bli_obj_buffer( beta );
      scomplex*  cp     = (scomplex*) bli_obj_buffer( c );
      cgemmt_( &f77_uploc,	&f77_transa, &f77_transb, &n, &k, alphap, ap,
           (f77_int*)&lda, bp, (f77_int*)&ldb, betap, cp, (f77_int*)&ldc );
      break;
    }
    case BLIS_DCOMPLEX :
    {
      dcomplex*  alphap = (dcomplex*) bli_obj_buffer( alpha );
      dcomplex*  ap     = (dcomplex*) bli_obj_buffer( a );
      dcomplex*  bp     = (dcomplex*) bli_obj_buffer( b );
      dcomplex*  betap  = (dcomplex*) bli_obj_buffer( beta );
      dcomplex*  cp     = (dcomplex*) bli_obj_buffer( c );
      zgemmt_( &f77_uploc,	&f77_transa, &f77_transb, &n, &k, alphap, ap,
           (f77_int*)&lda, bp, (f77_int*)&ldb, betap, cp, (f77_int*)&ldc );
      break;
    }
    default :
      bli_check_error_code( BLIS_INVALID_DATATYPE );
  }
  return 0;
}

void libblis_api_gemmt(
  test_params_t* params,
  iface_t        iface,
  obj_t          *alpha,
  obj_t          *a,
  obj_t          *b,
  obj_t          *beta,
  obj_t          *c,
  num_t          dt
) {

  if(params->api == API_BLIS) {
    libblis_test_gemmt_impl(iface, alpha, a, b, beta, c);
  }
  else { /*CLBAS  || BLAS */
    f77_int kk     = bli_obj_width_after_trans( a );
    f77_int nn     = bli_obj_width( c );
    uplo_t  uploc  = bli_obj_uplo( c );
    trans_t transa = bli_obj_conjtrans_status( a );
    trans_t transb = bli_obj_conjtrans_status( b );
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

    if(bli_obj_has_notrans(a) && bli_obj_has_conj(a)) {
       conjugate_tensor(a, dt);
       transa = bli_obj_onlytrans_status( a );
    }

    if(bli_obj_has_notrans(b) && bli_obj_has_conj(b)) {
       conjugate_tensor(b, dt);
       transb = bli_obj_onlytrans_status( b );
    }

    if(params->api == API_CBLAS) {
      cblas_gemmt( params, uploc, nn, kk, lda, ldb, ldc, a, b, c,
                        alpha, beta, dt, transa, transb);
    } else { /**/
      f77_char f77_transa;
      f77_char f77_transb;
      f77_char f77_uploc;

      if(transa == BLIS_TRANSPOSE)                f77_transa='T';
      else if ( transa == BLIS_CONJ_TRANSPOSE )   f77_transa='C';
      else /*if ( transa == BLIS_NO_TRANSPOSE )*/ f77_transa='N';

      if(transb == BLIS_TRANSPOSE)                f77_transb='T';
      else if ( transb == BLIS_CONJ_TRANSPOSE )   f77_transb='C';
      else /*if ( transb == BLIS_NO_TRANSPOSE )*/ f77_transb='N';

      if( bli_obj_row_stride( c ) == 1 ) {
        bli_param_map_blis_to_netlib_uplo( uploc, &f77_uploc );
        blas_gemmt(f77_uploc, nn, kk, lda, ldb, ldc, a, b, c,
                      alpha, beta, dt, f77_transa, f77_transb);
      }else {
        if( uploc == BLIS_UPPER)
          uploc = BLIS_LOWER;
        else if(uploc == BLIS_LOWER)
          uploc = BLIS_UPPER;

        bli_param_map_blis_to_netlib_uplo( uploc, &f77_uploc );
        blas_gemmt(f77_uploc, nn, kk, ldb, lda, ldc, b, a, c,
                      alpha, beta, dt, f77_transb, f77_transa);
      }
    }
  }
  return ;
}

double libblis_ref_gemmt(
  test_params_t* params,
  obj_t        * alpha,
  obj_t        * a,
  obj_t        * b,
  obj_t        * beta,
  obj_t        * c,
  obj_t        * c_ref,
  obj_t        * c_orig
){
  double resid = 0.0;
  double *betap = (double *)bli_obj_buffer( beta );

  if ((params->nanf) && (*betap == 0)) {
    resid = libblis_check_nan_gemmt( c );
  }
  else if((params->bitextf == 0) && (params->oruflw == BLIS_DEFAULT)) {
    resid = libblis_test_gemmt_check(params, alpha, a, b, beta, c, c_ref);
  }
  else {
    if(params->oruflw == BLIS_DEFAULT) {
      resid = libblis_test_igemmt_check(params, alpha, a, b, beta, c, c_orig);
    }
    else {
      resid = libblis_test_matrix_check(params, c);
    }
  }
  return resid;
}

double libblis_test_bitrp_gemmt(
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
    libblis_test_gemmt_impl(iface, alpha, a, b, beta, r);
    resid = libblis_test_bitrp_matrix(c, r, dt);
  }
  return resid;
}

double libblis_test_op_gemmt (
  test_params_t *params,
  iface_t       iface,
  char          *dc_str,
  char          *pc_str,
  char          *sc_str,
  tensor_t*      dim,
  atom_t         alpv,
  atom_t         betv
){
  num_t   datatype;
  dim_t   m, k;
  uplo_t  uploc;
  trans_t transa, transb;
  obj_t   alpha, a, b, beta;
  obj_t   c, c_ref, c_org_tri, c_result_tri, c_save;
  double  resid = 0.0;
  obj_t   aa,bb;

  // Use the datatype of the first char in the datatype combination string.
  bli_param_map_char_to_blis_dt(dc_str[0], &datatype);

  // Map the dimension specifier to actual dimensions.
  m = dim->m;
  k = dim->n;

  // Map parameter characters to BLIS constants.
  bli_param_map_char_to_blis_uplo(pc_str[0], &uploc);
  bli_param_map_char_to_blis_trans(pc_str[1], &transa);
  bli_param_map_char_to_blis_trans(pc_str[2], &transb);

  // Create test scalars.
  bli_obj_scalar_init_detached(datatype, &alpha);
  bli_obj_scalar_init_detached(datatype, &beta);

  // Create test operands (vectors and/or matrices).
  libblis_test_mobj_create(params, datatype, transa,
         sc_str[1], m, k, &a);
  libblis_test_mobj_create(params, datatype, transa,
         sc_str[1], m, k, &aa);
  libblis_test_mobj_create(params, datatype, transb,
         sc_str[2], k, m, &b);
  libblis_test_mobj_create(params, datatype, transb,
         sc_str[2], k, m, &bb);
  libblis_test_mobj_create(params, datatype, BLIS_NO_TRANSPOSE,
         sc_str[0], m, m, &c);
  libblis_test_mobj_create(params, datatype, BLIS_NO_TRANSPOSE,
         sc_str[0], m, m, &c_save);
  libblis_test_mobj_create(params, datatype, BLIS_NO_TRANSPOSE,
         sc_str[0], m, m, &c_ref);
  libblis_test_mobj_create(params, datatype, BLIS_NO_TRANSPOSE,
         sc_str[0], m, m, &c_org_tri);
  libblis_test_mobj_create(params, datatype, BLIS_NO_TRANSPOSE,
         sc_str[0], m, m, &c_result_tri);

  // Set alpha and beta.
  if((params->bitextf == 0) && (params->oruflw == BLIS_DEFAULT)) {
    if (bli_obj_is_real(&c)) {
      bli_setsc(alpv.real, 0.0, &alpha);
      bli_setsc(betv.real, 0.0, &beta);
    }
    else {
      // For gemmt, both alpha and beta may be complex since, unlike herk,
      // C is symmetric in both the real and complex cases.
      bli_setsc(alpv.real, (alpv.real/0.8), &alpha);
      bli_setsc(betv.real, (betv.real/1.2), &beta);
    }
    // Randomize A and B
    libblis_test_mobj_randomize(params, TRUE, &a);
    libblis_test_mobj_randomize(params, TRUE, &b);

    // Generate random input matrix
    libblis_test_mobj_randomize(params, TRUE, &c);
  }
  else {
    int32_t x = (int32_t)alpv.real;
    int32_t y = (int32_t)betv.real;
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

    // Randomize A and B
    libblis_test_mobj_irandomize( params, &a );
    libblis_test_mobj_irandomize( params, &b );

    // Generate random input matrix
    libblis_test_mobj_irandomize( params, &c );
  }

  bli_copym( &a, &aa );
  bli_copym( &b, &bb );

  // Apply the remaining parameters.
  // We need to do this before we create the referece matrix
  bli_obj_set_conjtrans(transa, &a);
  bli_obj_set_conjtrans(transb, &b);

  // Create the requried copies before setting the uplo attribute
  bli_copym(&c, &c_save);
  bli_copym(&c, &c_org_tri);
  bli_copym(&c, &c_result_tri);
  bli_obj_set_uplo(uploc, &c);
  bli_obj_set_uplo(uploc, &c_save);

  // Create c_org_tri matrix using setm operation, this matrix will
  // have original values from input matrix "c" for all elements outside
  // triangle selected for GEMMT operation.
  bli_obj_set_uplo(uploc, &c_org_tri); // Set to request uplo to set all elemnts in triangle to zero
  bli_setm(&BLIS_ZERO, &c_org_tri);
  bli_obj_toggle_uplo(&c_org_tri); // Toggle uplo now so that untouched triangle is active.

  // GEMMT output is same as GEMM for the triangle selected by uplo
  // So we want to extract this triangle from complete GEMM results
  // We do this by setting the uplo and converting the results
  // to triangluer matrix.
  // Perform gemm operation on original inputs
  bli_gemm(&alpha, &a, &b, &beta, &c_result_tri);
  // Set the values in other triangle to zero by converting it to trianguler matrix
  bli_obj_set_uplo(uploc, &c_result_tri);
  bli_mktrim(&c_result_tri);

  // Now we have two matrices with opposite triangles set to zero
  // c_result_tri: It has output of GEMM in selected triangle (including diagonal)
  //               Rest of its elements are set to zero.
  // c_org_tri: It has values from orignal C matrix in the non-selected triangle
  //            Rest of the elements including diagonal are set to zero
  // The result of the GEMMT operation will be combined matrix of thse two matrics
  // So add them togher
  bli_setm(&BLIS_ZERO, &c_ref); // Both matrices we are going to add, have uplo settings
           // Clear the destination matrix to avoid partial updates
  bli_copym(&c_org_tri, &c_ref);
  bli_addm(&c_result_tri, &c_ref);

  if ((params->nanf)) {
    test_fillbuffmem(&c, datatype );
  }

  //Copy c to c_save
  bli_copym( &c, &c_save );

  bli_obj_set_conjtrans( transa, &aa );
  bli_obj_set_conjtrans( transb, &bb );

  libblis_api_gemmt(params, iface, &alpha, &aa, &bb, &beta, &c, datatype);

#ifndef __GTEST_VALGRIND_TEST__
  if(params->bitrp) {
    obj_t r;

    libblis_test_mobj_create(params, datatype, BLIS_NO_TRANSPOSE,
                                                     sc_str[0], m, m, &r);

    resid = libblis_test_bitrp_gemmt( params, iface, &alpha, &a, &b, &beta,
                                                &c, &c_save, &r, datatype);
    bli_obj_free( &r );
  }
  else {
    resid = libblis_ref_gemmt(params, &alpha, &a, &b, &beta, &c, &c_ref, &c_save);
  }
#endif

  // Zero out performance and residual if output matrix is empty.
  libblis_test_check_empty_problem(&c, &resid);

  // Free the test objects.
  libblis_test_obj_free(&a);
  libblis_test_obj_free(&aa);
  libblis_test_obj_free(&b);
  libblis_test_obj_free(&bb);
  libblis_test_obj_free(&c);
  libblis_test_obj_free(&c_ref);
  libblis_test_obj_free(&c_org_tri);
  libblis_test_obj_free(&c_result_tri);
  libblis_test_obj_free(&c_save);

  return abs(resid);
}

void libblis_test_gemmt_impl(
  iface_t iface,
  obj_t *alpha,
  obj_t *a,
  obj_t *b,
  obj_t *beta,
  obj_t *c
) {
  switch (iface) {
    case BLIS_TEST_SEQ_FRONT_END:
      bli_gemmt(alpha, a, b, beta, c);
    break;

    default:
      libblis_test_printf_error("Invalid interface type.\n");
  }
}

double libblis_test_gemmt_check(
  test_params_t *params,
  obj_t *alpha,
  obj_t *a,
  obj_t *b,
  obj_t *beta,
  obj_t *c,
  obj_t *c_orig
) {
  num_t dt = bli_obj_dt(c);
  num_t dt_real = bli_obj_dt_proj_to_real(c);

  dim_t m = bli_obj_length(c);

  obj_t norm;
  obj_t t, v, z;

  double junk;
  double resid = 0.0;
  //
  // Pre-conditions:
  // - a is randomized.
  // - b is randomized.
  // - c is randomized with uplo set
  //
  // Note:
  // - alpha and beta should have non-zero imaginary components in the
  //   complex cases in order to more fully exercise the implementation.
  //
  // Under these conditions, we assume that the implementation for
  //
  //   C := beta * C_orig + alpha * transa(A) * transa(B)
  //
  // is functioning correctly if
  //
  //   normfv( v - z )
  //
  // is negligible, where
  //
  //   v = C * t
  //   z = C * C_reference
  //
  //

  bli_obj_scalar_init_detached(dt_real, &norm);

  bli_obj_create(dt, m, 1, 0, 0, &t);
  bli_obj_create(dt, m, 1, 0, 0, &v);
  bli_obj_create(dt, m, 1, 0, 0, &z);

  libblis_test_vobj_randomize(params, TRUE, &t);

  // Ensure result metrix has only selected triangle.
  // Calculate V = C * t
  bli_gemv(&BLIS_ONE, c, &t, &BLIS_ZERO, &v);
  bli_gemv(&BLIS_ONE, c_orig, &t, &BLIS_ZERO, &z);

  // Find the norm
  bli_subv(&z, &v);
  bli_normfv(&v, &norm);
  bli_getsc(&norm, &resid, &junk);

  bli_obj_free(&t);
  bli_obj_free(&v);
  bli_obj_free(&z);

  return resid;
}