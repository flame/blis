#include "blis_test.h"
#include "blis_utils.h"
#include "test_trsv.h"

using namespace std;

//*  ==========================================================================
//*> TRSV Solves a triangular system of equations with a single value for the
//*>        right side
//*>    x := alpha * inv(transa(A)) * x_orig
//*> where b and x are n element vectors and A is an n by n unit, or
//*> non-unit, upper or lower triangular matrix.
//*  ==========================================================================

template <typename T, typename U>
void libblis_itrsv_check(uplo_t uploa, trans_t transa, diag_t diaga,
      T* alpha, dim_t N, T* A, dim_t rsa, dim_t csa, T* X, dim_t incx){

  T Alpha = alpha[0];
  T tmp;
  int i, ix, j, jx, kx;
  bool NOTRANS, NOUNIT;

  if(N == 0)
    return;

  NOTRANS = (transa == BLIS_NO_TRANSPOSE);
  NOUNIT  = (diaga == BLIS_NONUNIT_DIAG);

  //*     Set up the start point in X if the increment is not unity. This
  //*     will be  ( N - 1 )*incx  too small for descending loops.
  if (incx > 0) {
    kx = 0;
  }
  else {
    kx = 1 - (N * incx);
  }

  ix = 0;
  for(i = 0 ; i < N ; i++) {
    X[ix] = (Alpha * X[ix]);
    ix = ix + incx;
  }

  //*     Start the operations. In this version the elements of A are
  //*     accessed sequentially with one pass through A.
  if(NOTRANS){
    //* Form  x := inv( A )*x.
    if(uploa == BLIS_UPPER){
      kx = kx + (N - 1)*incx;
      ix = kx;
      for(i = (N-1) ; i >= 0 ; i--){
        tmp = 0;
        jx = (ix+1);
        for(j = (i+1) ; j < N ; j++){
          tmp = tmp + X[jx]*A[i*rsa + j*csa];
          jx = jx + incx;
        }
        tmp = (X[ix] - tmp);
        if(NOUNIT)
          tmp = (tmp/A[i*rsa + i*csa]);
        X[ix] = tmp;
        ix = ix - incx;
      }
    }
    else{
      ix = kx;
      for(i = 0 ; i < N ; i++){
        tmp = 0;
        jx = kx;
        for(j = 0 ; j < i ; j++ ){
          tmp = tmp + (X[jx]*A[i*rsa + j*csa]);
          jx = jx + incx;
        }
        tmp = (X[ix] - tmp);
        if(NOUNIT)
          tmp = (tmp/A[i*rsa + i*csa]);
        X[ix] = tmp;
        ix = ix + incx;
      }
    }
  }
  else{
    //* Form  x := inv( A**T )*x.
    if(uploa == BLIS_UPPER){
      ix = kx;
      for(i = 0 ; i < N ; i++){
        if(NOUNIT)
          X[ix] = (X[ix]/A[i*rsa + i*csa]);
        tmp = X[ix];
        jx  = ix;
        for(j = (i+1) ; j < N ; j++){
          jx = jx + incx;
          X[jx] = X[jx] - (tmp * A[i*rsa + j*csa]);
        }
        ix = ix + incx;
      }
    }
    else{
      ix = kx + (N - 1)*incx;
      for(i = (N-1) ; i >= 0 ; i--){
        if(NOUNIT)
          X[ix] = (X[ix]/A[i*rsa + i*csa]);
        tmp = X[ix];
        jx = ix;
        for(j = (i-1) ; j >= 0 ; j--){
          jx = jx - incx;
          X[jx] = X[jx] - (tmp * A[i*rsa + j*csa]);
        }
        ix = ix - incx;
      }
    }
  }
  return;
}

template <typename T, typename U>
void libblis_ictrsv_check(uplo_t uploa, trans_t transa, diag_t diaga,
T* alpha, dim_t N, T* A, dim_t rsa, dim_t csa, bool conja, T* X, dim_t incx){

  T Alpha = *alpha;
  T tmp;
  int i, ix, j, jx, kx;
  bool NOTRANS, NOUNIT;

  if (N == 0)
    return;

  NOTRANS = (transa == BLIS_NO_TRANSPOSE);
  NOUNIT  = (diaga == BLIS_NONUNIT_DIAG);

  //*     Set up the start point in X if the increment is not unity. This
  //*     will be  ( N - 1 )*incx  too small for descending loops.
  if (incx > 0) {
    kx = 0;
  }
  else {
    kx = 1 - (N * incx);
  }

  ix = 0;
  for(i = 0 ; i < N ; i++) {
    X[ix] = mulc<T>(Alpha , X[ix]);
    ix = ix + incx;
  }

  if(conja) {
    for(i = 0 ; i < N ; i++) {
      for(j = 0 ; j < N ; j++) {
        A[i*rsa + j*csa] = conjugate<T>(A[i*rsa + j*csa]);
      }
    }
  }

  if(NOTRANS){
    //* Form  x := inv( A )*x.
    if(uploa == BLIS_UPPER){
      kx = kx + (N - 1)*incx;
      ix = kx;
      for(i = (N-1) ; i >= 0 ; i--){
        tmp = {0.0,0.0};
        jx = (ix+1);
        for(j = (i+1) ; j < N ; j++){
          tmp = addc<T>(tmp , mulc<T>(X[jx] , A[i*rsa + j*csa]));
          jx = jx + incx;
        }
        tmp = subc<T>(X[ix] , tmp);
        if(NOUNIT)
          tmp = divc<T,U>(tmp , A[i*rsa + i*csa]);
        X[ix] = tmp;
        ix = ix - incx;
      }
    }
    else{
      ix = kx;
      for(i = 0 ; i < N ; i++){
        tmp = {0.0,0.0};
        jx = kx;
        for(j = 0 ; j < i ; j++ ){
          tmp = addc<T>(tmp , mulc<T>(X[jx] , A[i*rsa + j*csa]));
          jx = jx + incx;
        }
        tmp = subc<T>(X[ix] , tmp);
        if(NOUNIT)
          tmp = divc<T,U>(tmp , A[i*rsa + i*csa]);
        X[ix] = tmp;
        ix = ix + incx;
      }
    }
  }
  else{
    //* Form  x := inv( A**T )*x.
    if(uploa == BLIS_UPPER){
      ix = kx;
      for(i = 0 ; i < N ; i++){
        if(NOUNIT)
          X[ix] = divc<T,U>(X[ix] , A[i*rsa + i*csa]);
        tmp = X[ix];
        jx  = ix;
        for(j = (i+1) ; j < N ; j++){
          jx = jx + incx;
          X[jx] = subc<T>(X[jx] , mulc<T>(tmp , A[i*rsa + j*csa]));
        }
        ix = ix + incx;
      }
    }
    else{
      ix = kx + (N - 1)*incx;
      for(i = (N-1) ; i >= 0 ; i--){
        if(NOUNIT)
          X[ix] = divc<T,U>(X[ix] , A[i*rsa + i*csa]);
        tmp = X[ix];
        jx = ix;
        for(j = (i-1) ; j >= 0 ; j--){
          jx = jx - incx;
          X[jx] = subc<T>(X[jx] , mulc<T>(tmp , A[i*rsa + j*csa]));
        }
        ix = ix - incx;
      }
    }
  }
  return;
}

double libblis_test_itrsv_check(
  test_params_t* params,
  obj_t*         alpha,
  obj_t*         a,
  obj_t*         x,
  obj_t*         x_orig
){
  num_t dt       = bli_obj_dt( x );
  dim_t M        = bli_obj_length( a );
  dim_t incx     = bli_obj_vector_inc( x );
  dim_t rsa      = bli_obj_row_stride( a ) ;
  dim_t csa      = bli_obj_col_stride( a ) ;
  uplo_t uploa   = bli_obj_uplo( a );
  bool conja     = bli_obj_has_conj( a );
  trans_t transa = bli_obj_onlytrans_status( a );
  diag_t diaga   = bli_obj_diag( a );
  double resid   = 0.0;

  switch( dt )  {
    case BLIS_FLOAT :
    {
      float*   X       = (float*) bli_obj_buffer( x_orig );
      float*   A       = (float*) bli_obj_buffer( a );
      float*   Alpha   = (float*) bli_obj_buffer( alpha );
      float*   XX      = (float*) bli_obj_buffer( x );
      libblis_itrsv_check<float, int32_t>(uploa, transa, diaga, Alpha,
                                                M, A, rsa, csa, X, incx);
      resid = computediffrv(M, incx, XX, X);
      break;
    }
    case BLIS_DOUBLE :
    {
      double*   X      = (double*) bli_obj_buffer( x_orig );
      double*   A      = (double*) bli_obj_buffer( a );
      double*   Alpha  = (double*) bli_obj_buffer( alpha );
      double*   XX     = (double*) bli_obj_buffer( x );
      libblis_itrsv_check<double, int64_t>(uploa, transa, diaga, Alpha,
                                                 M, A, rsa, csa, X, incx);
      resid = computediffrv(M, incx, XX, X);
      break;
    }
    case BLIS_SCOMPLEX :
    {
      scomplex*   X     = (scomplex*) bli_obj_buffer( x_orig );
      scomplex*   A     = (scomplex*) bli_obj_buffer( a );
      scomplex*   Alpha = (scomplex*) bli_obj_buffer( alpha );
      scomplex*   XX    = (scomplex*) bli_obj_buffer( x );
      libblis_ictrsv_check<scomplex, float>(uploa, transa, diaga, Alpha,
                                           M, A, rsa, csa, conja, X, incx);
      resid = computediffiv(M, incx, XX, X);
      break;
    }
    case BLIS_DCOMPLEX :
    {
      dcomplex*   X     = (dcomplex*) bli_obj_buffer( x_orig );
      dcomplex*   A     = (dcomplex*) bli_obj_buffer( a );
      dcomplex*   Alpha = (dcomplex*) bli_obj_buffer( alpha );
      dcomplex*   XX    = (dcomplex*) bli_obj_buffer( x );
      libblis_ictrsv_check<dcomplex, double>(uploa, transa, diaga, Alpha,
                                           M, A, rsa, csa, conja, X, incx);
      resid = computediffiv(M, incx, XX, X);
      break;
    }
    default :
      bli_check_error_code( BLIS_INVALID_DATATYPE );
  }
  return resid;
}

