#include "blis_test.h"
#include "blis_utils.h"
#include "test_trmv.h"

using namespace std;

//*  ==========================================================================
//*> TRMV  performs one of the matrix-vector operations
//*>    x := alpha * transa(A) * x
//*> where x is an n element vector and  A is an n by n unit, or non-unit,
//*> upper or lower triangular matrix.
//*  ==========================================================================

template <typename T, typename U>
void libblis_itrmv_check(uplo_t uploa, trans_t transa, diag_t diaga,
      T* alpha, dim_t N, T* A, dim_t rsa, dim_t csa, T* X, dim_t incx){

  T Alpha = *alpha;
  T tmp;
  int i, ix, j, jx, kx;
  bool NOTRANS, NOUNIT;

  if (N == 0)
    return;

  NOTRANS = (transa == BLIS_NO_TRANSPOSE);
  NOUNIT  = (diaga == BLIS_NONUNIT_DIAG);

  if (incx > 0) {
    kx = 0;
  }
  else {
    kx = 1 - (N * incx);
  }

  if(NOTRANS) {
    //* Form  x := A*x.
    if(uploa == BLIS_UPPER){
      jx = kx;
      for(j = 0 ; j < N ; j++){
        tmp = Alpha*X[jx];
        ix = kx;
        for(i = 0 ; i < j ; i++) {
          X[ix] = X[ix] + tmp*A[i*rsa + j*csa];
          ix = ix + incx;
        }
        if (NOUNIT)
          tmp = tmp*A[j*rsa + j*csa];

        X[jx] = tmp;
        jx = jx + incx;
      }
    }
    else{
      kx = kx + (N - 1)*incx;
      jx = kx;
      for(j = (N-1) ; j >= 0 ; j--){
        tmp = Alpha*X[jx];
        ix = kx;
        for(i = (N-1) ; i > j ; i--){
          X[ix] = X[ix] + tmp*A[i*rsa + j*csa];
          ix = ix - incx;
        }
        if(NOUNIT)
          tmp = tmp*A[j*rsa + j*csa];

        X[jx] = tmp;
        jx = jx - incx;
      }
    }
  }
  else {
    //* Form  x := A**T*x.
    if(uploa == BLIS_UPPER){
      jx = kx + (N - 1)*incx;
      for(j = (N-1) ; j >= 0 ; j--){
        tmp = X[jx];
        ix = jx;
        if(NOUNIT)
          tmp = tmp*A[j*rsa + j*csa];
        for(i = (j-1) ; i >= 0 ; i--) {
          ix = ix - incx;
          tmp = tmp + A[i*rsa + j*csa]*X[ix];
        }
        X[jx] = Alpha*tmp;
        jx = jx - incx;
      }
    }
    else{
      jx = kx;
      for(j = 0 ; j < N ; j++){
          tmp = X[jx];
          ix = jx;
          if (NOUNIT)
            tmp = tmp*A[j*rsa + j*csa];
        for(i = (j+1) ; i < N ; i++){
          ix = ix + incx;
          tmp = tmp + X[ix]*A[i*rsa + j*csa];
        }
        X[jx] = Alpha*tmp;
        jx = jx + incx;
      }
    }
  }
  return;
}

template <typename T, typename U>
void libblis_ictrmv_check(uplo_t uploa, trans_t transa, diag_t diaga,
T* alpha, dim_t N, T* A, dim_t rsa, dim_t csa, bool conja, T* X, dim_t incx){

  T Alpha = *alpha;
  T tmp;
  int i, ix, j, jx, kx;
  bool NOTRANS, NOUNIT;

  if (N == 0)
    return;

  NOTRANS = (transa == BLIS_NO_TRANSPOSE);
  NOUNIT  = (diaga == BLIS_NONUNIT_DIAG);

  if (incx > 0) {
    kx = 0;
  }
  else {
    kx = 1 - (N * incx);
  }

  if(conja) {
    for(i = 0 ; i < N ; i++) {
      for(j = 0 ; j < N ; j++) {
        A[i*rsa + j*csa] = conjugate<T>(A[i*rsa + j*csa]);
      }
    }
  }

  if(NOTRANS){
    //* Form  x := A*x.
    if(uploa == BLIS_UPPER){
      jx = kx;
      for(j = 0 ; j < N ; j++){
        tmp = mulc<T>(Alpha , X[jx]);
        ix = kx;
        for(i = 0 ; i < j ; i++) {
          X[ix] = addc<T>(X[ix] , mulc<T>(tmp , A[i*rsa + j*csa]));
          ix = ix + incx;
        }
        if (NOUNIT)
          tmp = mulc<T>(tmp , A[j*rsa + j*csa]);

        X[jx] = tmp;
        jx = jx + incx;
      }
    }
    else{
      kx = kx + (N - 1)*incx;
      jx = kx;
      for(j = (N-1) ; j >= 0 ; j--){
        tmp = mulc<T>(Alpha , X[jx]);
        ix = kx;
        for(i = (N-1) ; i > j ; i--){
          X[ix] = addc<T>(X[ix] , mulc<T>(tmp , A[i*rsa + j*csa]));
          ix = ix - incx;
        }
        if(NOUNIT)
          tmp = mulc<T>(tmp , A[j*rsa + j*csa]);

        X[jx] = tmp;
        jx = jx - incx;
      }
    }
  }
  else {
    //* Form  x := A**T*x.
    if(uploa == BLIS_UPPER){
      jx = kx + (N - 1)*incx;
      for(j = (N-1) ; j >= 0 ; j--){
        tmp = X[jx];
        ix = jx;
        if(NOUNIT)
          tmp = mulc<T>(tmp , A[j*rsa + j*csa]);
        for(i = (j-1) ; i >= 0 ; i--) {
          ix = ix - incx;
          tmp = addc<T>(tmp , mulc<T>(A[i*rsa + j*csa] , X[ix]));
        }
        X[jx] = mulc<T>(Alpha , tmp);
        jx = jx - incx;
      }
    }
    else{
      jx = kx;
      for(j = 0 ; j < N ; j++){
          tmp = X[jx];
          ix = jx;
          if (NOUNIT)
            tmp = mulc<T>(tmp , A[j*rsa + j*csa]);
        for(i = (j+1) ; i < N ; i++){
          ix = ix + incx;
          tmp = addc<T>(tmp , mulc<T>(X[ix] , A[i*rsa + j*csa]));
        }
        X[jx] = mulc<T>(Alpha , tmp);
        jx = jx + incx;
      }
    }
  }
  return;
}

double libblis_test_itrmv_check(
  test_params_t* params,
  obj_t*         alpha,
  obj_t*         a,
  obj_t*         x,
  obj_t*         x_orig
){
  num_t dt       = bli_obj_dt( x );
  dim_t M        = bli_obj_length( a );
  dim_t incx     = bli_obj_vector_inc( x );
  dim_t rsa      = bli_obj_row_stride( a );
  dim_t csa      = bli_obj_col_stride( a );
  uplo_t uploa   = bli_obj_uplo( a );
  bool conja     = bli_obj_has_conj( a );
  trans_t transa = bli_obj_onlytrans_status( a );
  diag_t diaga   = bli_obj_diag( a );
  double resid   = 0.0;

  switch( dt )  {
    case BLIS_FLOAT :
    {
      float*   X       = (float*) bli_obj_buffer( x_orig );
      float*   Alpha   = (float*) bli_obj_buffer( alpha );
      float*   A       = (float*) bli_obj_buffer( a );
      float*   XX      = (float*) bli_obj_buffer( x );
      libblis_itrmv_check<float, int32_t>(uploa, transa, diaga, Alpha,
                                                M, A, rsa, csa, X, incx);
      resid = computediffrv(M, incx, XX, X);
      break;
    }
    case BLIS_DOUBLE :
    {
      double*   X      = (double*) bli_obj_buffer( x_orig );
      double*   Alpha  = (double*) bli_obj_buffer( alpha );
      double*   A      = (double*) bli_obj_buffer( a );
      double*   XX     = (double*) bli_obj_buffer( x );
      libblis_itrmv_check<double, int64_t>(uploa, transa, diaga, Alpha,
                                                M, A, rsa, csa, X, incx);
      resid = computediffrv(M, incx, XX, X);
      break;
    }
    case BLIS_SCOMPLEX :
    {
      scomplex*   X     = (scomplex*) bli_obj_buffer( x_orig );
      scomplex*   Alpha = (scomplex*) bli_obj_buffer( alpha );
      scomplex*   A     = (scomplex*) bli_obj_buffer( a );
      scomplex*   XX    = (scomplex*) bli_obj_buffer( x );
      libblis_ictrmv_check<scomplex, int32_t>(uploa, transa, diaga, Alpha,
                                           M, A, rsa, csa, conja, X, incx);
      resid = computediffiv(M, incx, XX, X);
      break;
    }
    case BLIS_DCOMPLEX :
    {
      dcomplex*   X     = (dcomplex*) bli_obj_buffer( x_orig );
      dcomplex*   Alpha = (dcomplex*) bli_obj_buffer( alpha );
      dcomplex*   A     = (dcomplex*) bli_obj_buffer( a );
      dcomplex*   XX    = (dcomplex*) bli_obj_buffer( x );
      libblis_ictrmv_check<dcomplex, int64_t>(uploa, transa, diaga, Alpha,
                                           M, A, rsa, csa, conja, X, incx);
      resid = computediffiv(M, incx, XX, X);
      break;
    }
    default :
      bli_check_error_code( BLIS_INVALID_DATATYPE );
  }
  return resid;
}

