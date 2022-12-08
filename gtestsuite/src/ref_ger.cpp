#include "blis_test.h"
#include "blis_utils.h"
#include "test_ger.h"

using namespace std;

//*  ==========================================================================
//*> GER performs the rank 1 operation
//*>    A := alpha*x*y**T + A,
//*> where alpha is a scalar, x is an m element vector, y is an n element
//*> vector and A is an m by n matrix.
//*  ==========================================================================

template <typename T, typename U>
void libblis_iger_check(dim_t M, dim_t N, T *alpha, T *X, dim_t incx,
                          T* Y, dim_t incy, T* A, dim_t rsa, dim_t csa) {
  T Alpha = alpha[0];
  T temp;
  dim_t i, ix, j, jy, kx;
  T ZERO = 0.0;

  if ((M == 0) || (N == 0) ||
    (Alpha == ZERO))
      return;

  if (incy > 0) {
    jy = 0;
  }
  else {
    jy = 1 - (N - 1)*incy;
  }

  if (incx > 0) {
    kx = 0;
  }
  else {
    kx = 1 - (M - 1)*incx;
  }

  for(j = 0; j < N ; j++) {
    if (Y[jy] != ZERO) {
      temp = Alpha * Y[jy];
      ix = kx;
      for(i = 0 ; i <  M ; i++) {
        A[i*rsa + j *csa] = A[i*rsa + j *csa] + temp * X[ix];
        ix = ix + incx;
      }
    }
    jy = jy + incy;
  }
  return;
}

template <typename T, typename U>
void libblis_icger_check(dim_t M, dim_t N, T *alpha, T *X, dim_t incx,
  bool conjx, T* Y, dim_t incy, bool conjy, T* A, dim_t rsa, dim_t csa) {

  T Alpha = alpha[0];
  T temp;
  dim_t i, ix, j, jy, kx;
  T ZERO = {0.0 , 0.0};

  if ((M == 0) || (N == 0) ||
    ((Alpha.real == ZERO.real) &&(Alpha.imag == ZERO.imag)))
      return;

  ix = 0;
  if(conjx) {
    for(i = 0 ; i < M ; i++) {
      X[ix] = conjugate<T>(X[ix]);
      ix = ix + incx;
    }
  }

  jy = 0;
  if(conjy) {
    for(j = 0; j < N ; j++) {
      Y[jy] = conjugate<T>(Y[jy]);
      jy = jy + incy;
    }
  }

  if (incy > 0) {
    jy = 0;
  }
  else {
    jy = 1 - (N - 1)*incy;
  }

  if (incx > 0) {
    kx = 0;
  }
  else {
    kx = 1 - (M - 1)*incx;
  }

  for(j = 0; j < N ; j++) {
    if ((Y[jy].real != ZERO.real) || (Y[jy].imag != ZERO.imag)) {
      temp = mulc<T>(Alpha , Y[jy]);
      ix = kx;
      for(i = 0 ; i <  M ; i++) {
        A[i*rsa + j*csa] = addc<T>(A[i*rsa + j*csa] , mulc<T>(temp , X[ix]));
        ix = ix + incx;
      }
    }
    jy = jy + incy;
  }
  return;
}

double libblis_test_iger_check(
  test_params_t* params,
  obj_t*         alpha,
  obj_t*         x,
  obj_t*         y,
  obj_t*         a,
  obj_t*         a_orig
){
  num_t dt   = bli_obj_dt( x );
  dim_t M    = bli_obj_length( a );
  dim_t N    = bli_obj_width( a );
  dim_t incx = bli_obj_vector_inc( x );
  dim_t incy = bli_obj_vector_inc( y );
  bool conjx = bli_obj_has_conj( x );
  bool conjy = bli_obj_has_conj( y );
  dim_t rsa = bli_obj_row_stride( a ) ;
  dim_t csa = bli_obj_col_stride( a ) ;
  double resid = 0.0;

  switch( dt )  {
    case BLIS_FLOAT :
    {
      float*   Alpha    = (float*) bli_obj_buffer( alpha );
      float*   A        = (float*) bli_obj_buffer( a_orig );
      float*   X        = (float*) bli_obj_buffer( x );
      float*   Y        = (float*) bli_obj_buffer( y );
      float*   AA       = (float*) bli_obj_buffer( a );
      libblis_iger_check<float, int32_t>(M, N, Alpha, X, incx,
                                                      Y, incy, A, rsa, csa);
      resid = computediffrm(M, N, AA, A, rsa, csa);
      break;
    }
    case BLIS_DOUBLE :
    {
      double*   Alpha   = (double*) bli_obj_buffer( alpha );
      double*   A       = (double*) bli_obj_buffer( a_orig );
      double*   X       = (double*) bli_obj_buffer( x );
      double*   Y       = (double*) bli_obj_buffer( y );
      double*   AA      = (double*) bli_obj_buffer( a );
      libblis_iger_check<double, int64_t>(M, N, Alpha, X, incx,
                                                      Y, incy, A, rsa, csa);
      resid = computediffrm(M, N, AA, A, rsa, csa);
      break;
    }
    case BLIS_SCOMPLEX :
    {
      scomplex*   Alpha = (scomplex*) bli_obj_buffer( alpha );
      scomplex*   A     = (scomplex*) bli_obj_buffer( a_orig );
      scomplex*   X     = (scomplex*) bli_obj_buffer( x );
      scomplex*   Y     = (scomplex*) bli_obj_buffer( y );
      scomplex*   AA    = (scomplex*) bli_obj_buffer( a );
      libblis_icger_check<scomplex, int32_t>(M, N, Alpha, X, incx, conjx,
                                                Y, incy, conjy, A, rsa, csa);
      resid = computediffim(M, N, AA, A, rsa, csa);
      break;
    }
    case BLIS_DCOMPLEX :
    {
      dcomplex*   Alpha = (dcomplex*) bli_obj_buffer( alpha );
      dcomplex*   A     = (dcomplex*) bli_obj_buffer( a_orig );
      dcomplex*   X     = (dcomplex*) bli_obj_buffer( x );
      dcomplex*   Y     = (dcomplex*) bli_obj_buffer( y );
      dcomplex*   AA    = (dcomplex*) bli_obj_buffer( a );
      libblis_icger_check<dcomplex, int64_t>(M, N, Alpha, X, incx, conjx,
                                                Y, incy, conjy, A, rsa, csa);
      resid = computediffim(M, N, AA, A, rsa, csa);
      break;
    }
    default :
      bli_check_error_code( BLIS_INVALID_DATATYPE );
  }

  return resid;
}