#include <complex>
#include "blis_test.h"
#include "blis_utils.h"
#include "test_gemmt.h"

using namespace std;

//*  ==========================================================================
//*> GEMMT performs one of the matrix-matrix operations
//*>    C := beta * C + alpha * transa(A) * transb(B)
//*  ==========================================================================

void libblis_gemv_check(trans_t transA , dim_t M, dim_t N, float* Alpha,
  float* A, dim_t rsa, dim_t csa, bool conja, float* X, dim_t incx, bool conjx,
  float* Beta, float* Y, dim_t incy) {
  libblis_igemv_check<float, int32_t>(transA, M,  N, Alpha, A, rsa, csa,
                                                       X, incx, Beta, Y, incy);
  return;
}

void libblis_gemv_check(trans_t transA , dim_t M, dim_t N, double* Alpha,
  double* A, dim_t rsa, dim_t csa, bool conja, double* X, dim_t incx,
  bool conjx, double* Beta, double* Y, dim_t incy) {
  libblis_igemv_check<double, int64_t>(transA, M,  N, Alpha, A, rsa, csa,
                                                       X, incx, Beta, Y, incy);
  return;
}

void libblis_gemv_check(trans_t transA , dim_t M, dim_t N, scomplex* Alpha,
  scomplex* A, dim_t rsa, dim_t csa, bool conja, scomplex* X, dim_t incx,
  bool conjx, scomplex* Beta, scomplex* Y, dim_t incy) {
  libblis_icgemv_check<scomplex, int32_t>(transA, M,  N, Alpha, A, rsa, csa,
                                          conja, X, incx, Beta, Y, incy, conjx);
  return;
}
void libblis_gemv_check(trans_t transA , dim_t M, dim_t N, dcomplex* Alpha,
  dcomplex* A, dim_t rsa, dim_t csa, bool conja, dcomplex* X, dim_t incx,
  bool conjx, dcomplex* Beta, dcomplex* Y, dim_t incy) {
  libblis_icgemv_check<dcomplex, int64_t>(transA, M,  N, Alpha, A, rsa, csa,
                                          conja, X, incx, Beta, Y, incy, conjx);
  return;
}


void libblis_gemm_check(dim_t M, dim_t N, dim_t K, float* Alpha, float* A,
  dim_t rsa, dim_t csa, bool conja, float* B, dim_t rsb, dim_t csb, bool conjb,
  float* Beta, float* C, dim_t rsc, dim_t csc) {
  libblis_igemm_check<float, int32_t>(M, N, K, Alpha, A, rsa, csa, B, rsb,
                                                     csb, Beta, C, rsc, csc);
  return;
}

void libblis_gemm_check(dim_t M, dim_t N, dim_t K, double* Alpha, double* A,
  dim_t rsa, dim_t csa, bool conja, double* B, dim_t rsb, dim_t csb, bool conjb,
  double* Beta, double* C, dim_t rsc, dim_t csc) {
  libblis_igemm_check<double, int64_t>(M, N, K, Alpha, A, rsa, csa, B, rsb,
                                                     csb, Beta, C, rsc, csc);
  return;
}

void libblis_gemm_check(dim_t M, dim_t N, dim_t K, scomplex* Alpha, scomplex* A,
  dim_t rsa, dim_t csa, bool conja, scomplex* B, dim_t rsb, dim_t csb, bool conjb,
  scomplex* Beta, scomplex* C, dim_t rsc, dim_t csc) {
  libblis_icgemm_check<scomplex, int32_t>(M, N, K, Alpha, A, rsa, csa, conja,
                                        B, rsb, csb, conjb, Beta, C, rsc, csc);
  return;
}

void libblis_gemm_check(dim_t M, dim_t N, dim_t K, dcomplex* Alpha, dcomplex* A,
  dim_t rsa, dim_t csa, bool conja, dcomplex* B, dim_t rsb, dim_t csb, bool conjb,
  dcomplex* Beta, dcomplex* C, dim_t rsc, dim_t csc) {
  libblis_icgemm_check<dcomplex, int64_t>(M, N, K, Alpha, A, rsa, csa, conja,
                                        B, rsb, csb, conjb, Beta, C, rsc, csc);
  return;
}

#define CROSSOVER_GEMMT 24

dim_t rec_split(dim_t n, num_t dt) {
  dim_t res = 0;

  switch( dt )  {
    case BLIS_FLOAT :
    {
      res = ((n >= 32) ? ((n + 16) / 32) * 16 : n / 2);
      break;
    }
    case BLIS_DOUBLE :
    {
      res = ((n >= 16) ? ((n + 8) / 16) * 8 : n / 2);
      break;
    }
    case BLIS_SCOMPLEX :
    {
      res = ((n >= 16) ? ((n + 8) / 16) * 8 : n / 2);
      break;
    }
    case BLIS_DCOMPLEX :
    {
      res = ((n >= 8) ? ((n + 4) / 8) * 4 : n / 2);
      break;
    }
    default :
      bli_check_error_code( BLIS_INVALID_DATATYPE );
  }
  return res;
}

/** sgemmt's unblocked compute kernel */
template <typename T, typename U>
static void gemmt_rec2(uplo_t uploc, trans_t transA, trans_t transB,
  dim_t n, dim_t k, T* alpha,  T* A,  dim_t ldA, bool conja, T* B,
  dim_t ldB, bool conjb, T* beta, T* C, dim_t ldC ) {

  dim_t incB, incC;
  dim_t rsa, csa;
  dim_t i;

  rsa = (transA == BLIS_NO_TRANSPOSE) ? 1 : ldA ;
  csa = (transA == BLIS_NO_TRANSPOSE) ? ldA : 1 ;
  incB = (transB == BLIS_NO_TRANSPOSE) ? 1 : ldB ;
  incC = 1;

  for (i = 0; i < n; i++) {
    // A_0
    // A_i
    T * A_0 = A;
    T * A_i = A + ((transA == BLIS_NO_TRANSPOSE) ? i : ldA * i);

    // * B_i *
    T * B_i = B + ((transB == BLIS_NO_TRANSPOSE) ? ldB * i : i);

    // * C_0i *
    // * C_ii *
    T * C_0i = C + ldC * i;
    T * C_ii = C + ldC * i + i;

    if (uploc == BLIS_LOWER) {
      int nmi = n - i;
      if (transA == BLIS_NO_TRANSPOSE)
        libblis_gemv_check(transA, nmi, k, alpha, A_i, rsa, csa, conja, B_i, incB, conjb, beta, C_ii, incC);
      else
        libblis_gemv_check(transA, k, nmi, alpha, A_i, rsa, csa, conja, B_i, incB, conjb, beta, C_ii, incC);
    } else {
      int ip1 = i + 1;
      if (transA == BLIS_NO_TRANSPOSE)
        libblis_gemv_check(transA, ip1, k, alpha, A_0, rsa, csa, conja, B_i, incB, conjb, beta, C_0i, incC);
      else
        libblis_gemv_check(transA, k, ip1, alpha, A_0, rsa, csa, conja, B_i, incB, conjb, beta, C_0i, incC);
    }
  }
}

/** sgemmt's recursive compute kernel */
template <typename T, typename U>
static void gemmt_rec(uplo_t uploc,  trans_t transA,  trans_t transB,
  dim_t n, dim_t k, T* alpha,  T* A,  dim_t ldA, bool cfA, T* B,
  dim_t ldB, bool cfB, T* beta, T* C, dim_t ldC, num_t dt ) {
  if (n <= max(CROSSOVER_GEMMT, 1)) {
    // Unblocked
    gemmt_rec2<T,U>(uploc, transA, transB, n, k, alpha, A, ldA, cfA,
                                          B, ldB, cfB, beta, C, ldC);
    return;
  }

  dim_t  rsa, csa;
  dim_t  rsb, csb;
  dim_t  rsc, csc;

  rsa = (transA == BLIS_NO_TRANSPOSE) ? 1 : ldA ;
  csa = (transA == BLIS_NO_TRANSPOSE) ? ldA : 1 ;
  rsb = (transB == BLIS_NO_TRANSPOSE) ? 1 : ldB ;
  csb = (transB == BLIS_NO_TRANSPOSE) ? ldB : 1 ;
  rsc = 1  ;
  csc = ldC;

  // Splitting
  dim_t n1 = rec_split(n, dt);       //SREC_SPLIT(n);
  dim_t n2 = n - n1;

  // A_T
  // A_B
  T * A_T = A;
  T * A_B = A + ((transA == BLIS_NO_TRANSPOSE) ? n1 : ldA * n1);

  // B_L B_R
  T * B_L = B;
  T * B_R = B + ((transB == BLIS_NO_TRANSPOSE) ? ldB * n1 : n1);

  // C_TL C_TR
  // C_BL C_BR
  T * C_TL = C;
  T * C_TR = C + ldC * n1;
  T * C_BL = C            + n1;
  T * C_BR = C + ldC * n1 + n1;

  // recursion(C_TL)
  gemmt_rec<T,U>(uploc, transA, transB, n1, k, alpha, A_T, ldA, cfA, B_L, ldB,
                                               cfB, beta, C_TL, ldC, dt);

  if (uploc == BLIS_LOWER)
    // C_BL = alpha A_B B_L + beta C_BL
    libblis_gemm_check(n2, n1, k, alpha, A_B, rsa, csa, cfA,
                               B_L, rsb, csb, cfB, beta, C_BL, rsc, csc);
  else
    // C_TR = alpha A_T B_R + beta C_TR
    libblis_gemm_check(n1, n2, k, alpha, A_T, rsa, csa, cfA,
                               B_R, rsb, csb, cfB, beta, C_TR, rsc, csc);

  // recursion(C_BR)
  gemmt_rec<T,U>(uploc, transA, transB, n2, k, alpha, A_B, ldA, cfA, B_R, ldB,
                                               cfB, beta, C_BR, ldC, dt);
}

double computediff(dim_t n,dim_t k, float *act, float *ref, dim_t rsc, dim_t csc) {
  return computediffrm(n, k, act, ref, rsc, csc);
}

double computediff(dim_t n,dim_t k, double *act, double *ref, dim_t rsc, dim_t csc) {
  return computediffrm(n, k, act, ref, rsc, csc);
}

double computediff(dim_t n,dim_t k, scomplex *act, scomplex *ref, dim_t rsc, dim_t csc) {
  return computediffim(n, k, act, ref, rsc, csc);
}
double computediff(dim_t n,dim_t k, dcomplex *act, dcomplex *ref, dim_t rsc, dim_t csc) {
  return computediffim(n, k, act, ref, rsc, csc);
}

/** GEMMT computes a matrix-matrix product with general matrices but updates
 * only the upper or lower triangular part of the result matrix.
 * */
template <typename T, typename U>
double libblis_igemmt_check(
  test_params_t* params,
  obj_t*  alpha,
  obj_t*  a,
  obj_t*  b,
  obj_t*  beta,
  obj_t*  c,
  obj_t*  c_orig,
  num_t   dt
){
  dim_t k        = bli_obj_width_after_trans( a );
  dim_t n        = bli_obj_width( c );
  uplo_t  uploc  = bli_obj_uplo( c );
  trans_t transA = bli_obj_onlytrans_status( a );
  trans_t transB = bli_obj_onlytrans_status( b );
  dim_t lda, ldb, ldc;
  dim_t  rsa, csa;
  dim_t  rsb, csb;
  dim_t  rsc, csc;

  bool crsf = bli_obj_is_row_stored( c );

  if ( crsf ) {
    rsa = transA ? bli_obj_col_stride( a ) : bli_obj_row_stride( a ) ;
    csa = transA ? bli_obj_row_stride( a ) : bli_obj_col_stride( a ) ;
    rsb = transB ? bli_obj_col_stride( b ) : bli_obj_row_stride( b ) ;
    csb = transB ? bli_obj_row_stride( b ) : bli_obj_col_stride( b ) ;
    rsc = bli_obj_row_stride( c_orig ) ;
    csc = 1 ;
    lda = transA ? csa : rsa ;
    ldb = transB ? csb : rsb ;
    ldc = rsc ;
  } else {
    rsa = transA ? bli_obj_col_stride( a ) : bli_obj_row_stride( a ) ;
    csa = transA ? bli_obj_row_stride( a ) : bli_obj_col_stride( a ) ;
    rsb = transB ? bli_obj_col_stride( b ) : bli_obj_row_stride( b ) ;
    csb = transB ? bli_obj_row_stride( b ) : bli_obj_col_stride( b ) ;
    rsc = 1 ;
    csc = bli_obj_col_stride( c_orig ) ; ;
    lda = transA ? rsa : csa ;
    ldb = transB ? rsb : csb ;
    ldc = csc ;
  }

  T* A       = (T*) bli_obj_buffer( a );
  T* B       = (T*) bli_obj_buffer( b );
  T* C       = (T*) bli_obj_buffer( c_orig );
  T* Alpha   = (T*) bli_obj_buffer( alpha );
  T* Beta    = (T*) bli_obj_buffer( beta );
  bool conja = bli_obj_has_conj( a );
  bool conjb = bli_obj_has_conj( b );

  if(bli_obj_has_conj(a)) {
     conjugate_tensor(a, dt);
     transA = bli_obj_onlytrans_status( a );
     conja  = false;
  }

  if(bli_obj_has_conj(b)) {
     conjugate_tensor(b, dt);
     transB = bli_obj_onlytrans_status( b );
     conjb  = false;
  }

  // Recursive kernel
  if( !crsf ) {
    gemmt_rec<T,U>(uploc, transA, transB, n, k, Alpha, A, lda,
                                  conja, B, ldb, conjb, Beta, C, ldc, dt);
  }else {
    if( uploc == BLIS_UPPER)
      uploc = BLIS_LOWER;
    else if(uploc == BLIS_LOWER)
      uploc = BLIS_UPPER;

    gemmt_rec<T,U>(uploc, transB, transA, n, k, Alpha, B, ldb,
                                  conjb, A, lda, conja, Beta, C, ldc, dt);
  }

  T* CC = (T*) bli_obj_buffer( c );

  double resid = 0.0;
  resid = computediff(n, k, C, CC, rsc, csc);

  return resid;
}

double libblis_test_igemmt_check(
  test_params_t *params,
  obj_t *alpha,
  obj_t *a,
  obj_t *b,
  obj_t *beta,
  obj_t *c,
  obj_t *c_orig
) {
  double resid = 0.0;
  num_t  dt    = bli_obj_dt(c);

  switch( dt )  {
    case BLIS_FLOAT :
    {
      resid = libblis_igemmt_check<float, int32_t>( params, alpha, a, b, beta,
                                       c, c_orig, dt );
      break;
    }
    case BLIS_DOUBLE :
    {
      resid = libblis_igemmt_check<double, int64_t>( params, alpha, a, b, beta,
                                       c, c_orig, dt );
      break;
    }
    case BLIS_SCOMPLEX :
    {
      resid = libblis_igemmt_check<scomplex, int32_t>( params, alpha, a, b, beta,
                                       c, c_orig, dt );
      break;
    }
    case BLIS_DCOMPLEX :
    {
      resid = libblis_igemmt_check<dcomplex, int64_t>( params, alpha, a, b, beta,
                                       c, c_orig, dt );
      break;
    }
    default :
      bli_check_error_code( BLIS_INVALID_DATATYPE );
  }
  return resid;
}


template <typename T>
double libblis_check_nan_real( dim_t rsc, dim_t csc, obj_t* c ) {
  dim_t  M = bli_obj_length( c );
  dim_t  N = bli_obj_width( c );
  dim_t  i,j;
  double resid = 0.0;
  T* C = (T*) bli_obj_buffer( c );

  for( i = 0 ; i < M ; i++ ) {
    for( j = 0 ; j < N ; j++ ) {
      auto tv = C[ i*rsc + j*csc ];
      if ( bli_isnan( tv )) {
        resid = tv ;
        break;
      }
    }
  }
  return resid;
}

template <typename U, typename T>
double libblis_check_nan_complex( dim_t rsc, dim_t csc, obj_t* c ) {
  dim_t  M = bli_obj_length( c );
  dim_t  N = bli_obj_width( c );
  dim_t  i,j;
  double resid = 0.0;
  U* C = (U*) bli_obj_buffer( c );

  for( i = 0 ; i < M ; i++ ) {
    for( j = 0 ; j < N ; j++ ) {
      auto tv = C[ i*rsc + j*csc ];
      if ( bli_isnan( tv.real ) || bli_isnan( tv.imag )) {
        resid = bli_isnan( tv.real ) ? tv.real : tv.imag;
        break;
      }
    }
  }
  return resid;
}

double libblis_check_nan_gemmt(obj_t* c) {
  dim_t  rsc, csc;
  double resid = 0.0;

  num_t dt = bli_obj_dt(c);
  if( bli_obj_row_stride( c ) == 1 ) {
    rsc = 1;
    csc = bli_obj_col_stride( c );
  } else {
    rsc = bli_obj_row_stride( c );
    csc = 1 ;
  }

  switch( dt )  {
    case BLIS_FLOAT:
    {
      resid = libblis_check_nan_real<float>( rsc, csc, c );
      break;
    }
    case BLIS_DOUBLE:
    {
      resid = libblis_check_nan_real<double>( rsc, csc, c );
      break;
    }
    case BLIS_SCOMPLEX:
    {
      resid = libblis_check_nan_complex<scomplex, float>( rsc, csc, c );
      break;
    }
    case BLIS_DCOMPLEX:
    {
      resid = libblis_check_nan_complex<dcomplex, double>( rsc, csc, c );
      break;
    }
    default :
      bli_check_error_code( BLIS_INVALID_DATATYPE );
  }

  return resid;
}


