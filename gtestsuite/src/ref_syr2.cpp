#include "blis_test.h"
#include "blis_utils.h"
#include "test_syr2.h"

using namespace std;

//*  ==========================================================================
//*> SYR2  performs the symmetric rank 2 operation
//*>    A := alpha*x*y**T + alpha*y*x**T + A,
//*> where alpha is a scalar, x and y are n element vectors and A is an n
//*> by n symmetric matrix.
//*  ==========================================================================

template <typename T, typename U>
void libblis_isyr2_check(uplo_t uploa, dim_t N, T* alpha, T* X, dim_t incx,
                             T* Y, dim_t incy, T* A, dim_t rsa, dim_t csa) {

  T ZERO  = 0.0;
  T Alpha = alpha[0];
  T tmp1, tmp2;
  int i, ix, iy, j, jx, jy, kx, ky;

  if ((N == 0) || (Alpha == ZERO))
    return;

  if (incx > 0) {
    kx = 0;
  }
  else {
    kx = 1 - (N * incx);
  }

  if (incy > 0) {
    ky = 0;
  }
  else {
    ky = 1 - (N * incy);
  }
  jx = kx;
  jy = ky;

  if(uploa == BLIS_UPPER) {
    //* Form  A  when A is stored in the upper triangle.
    for(j = 0 ; j < N ; j++) {
      if ((X[jx] != ZERO) || (Y[jy] != ZERO)) {
        tmp1 = Alpha * Y[jy];
        tmp2 = Alpha * X[jx];
        ix = kx;
        iy = ky;
        for(i = 0 ; i <= j ; i++) {
          A[i*rsa + j*csa] = A[i*rsa + j*csa] + (X[ix] * tmp1) + (Y[iy] * tmp2);
          ix = ix + incx;
          iy = iy + incy;
        }
      }
      jx = jx + incx;
      jy = jy + incy;
    }
  }
  else {
    //* Form  A  when A is stored in the lower triangle.
    for(j = 0 ; j < N ; j++) {
      if((X[jx] != ZERO) || (Y[jy] != ZERO)) {
        tmp1 = Alpha * Y[jy];
        tmp2 = Alpha * X[jx];
        ix = jx;
        iy = jy;
        for(i = j ; i < N ; i++) {
          A[i*rsa + j*csa] = A[i*rsa + j*csa] + (X[ix] * tmp1) + (Y[iy] * tmp2);
          ix = ix + incx;
          iy = iy + incy;
        }
      }
      jx = jx + incx;
      jy = jy + incy;
    }
  }
    return;
}

template <typename T, typename U>
void libblis_icsyr2_check(uplo_t uploa, dim_t N, T* alpha, T* X, dim_t incx,
      bool conjx, T* Y, dim_t incy, bool conjy, T* A, dim_t rsa, dim_t csa) {

  T ZERO  = {0.0, 0.0};
  T Alpha = *alpha;
  T tmp1, tmp2;
  int i, ix, iy, j, jx, jy, kx, ky;

  if((N == 0) || ((Alpha.real == ZERO.real) && (Alpha.imag == ZERO.imag)))
    return;

  if (incx > 0) {
    kx = 0;
  }
  else {
    kx = 1 - (N * incx);
  }

  if (incy > 0) {
    ky = 0;
  }
  else {
    ky = 1 - (N * incy);
  }
  jx = kx;
  jy = ky;

  if(conjx) {
    ix = 0;
    for(i = 0 ; i < N ; i++) {
      X[ix] = conjugate<T>(X[ix]);
      ix = ix + incx;
    }
  }

  if(conjy) {
    iy = 0;
    for(i = 0 ; i < N ; i++) {
      Y[iy] = conjugate<T>(Y[iy]);
      iy = iy + incy;
    }
  }

  T p1, p2, p;
  if(uploa == BLIS_UPPER) {
    //* Form  A  when A is stored in the upper triangle.
    for(j = 0 ; j < N ; j++) {
      tmp1 = mulc<T>(Alpha , Y[jy]);
      tmp2 = mulc<T>(Alpha , X[jx]);
      ix = kx;
      iy = ky;
      for(i = 0 ; i <= j ; i++) {
        p1 = mulc<T>(X[ix] , tmp1);
        p2 = mulc<T>(Y[iy] , tmp2);
        p  = addc<T>(p1 , p2);
        A[i*rsa + j*csa] = addc<T>(A[i*rsa + j*csa] , p);
        ix = ix + incx;
        iy = iy + incy;
      }
      jx = jx + incx;
      jy = jy + incy;
    }
  }
  else {
    //* Form  A  when A is stored in the lower triangle.
    for(j = 0 ; j < N ; j++) {
      tmp1 = mulc<T>(Alpha , Y[jy]);
      tmp2 = mulc<T>(Alpha , X[jx]);
      ix = jx;
      iy = jy;
      for(i = j ; i < N ; i++) {
        p1 = mulc<T>(X[ix] , tmp1);
        p2 = mulc<T>(Y[iy] , tmp2);
        p  = addc<T>(p1 , p2);
        A[i*rsa + j*csa] = addc<T>(A[i*rsa + j*csa] , p);
        ix = ix + incx;
        iy = iy + incy;
      }
      jx = jx + incx;
      jy = jy + incy;
    }
  }
  return;
}

double libblis_test_isyr2_check(
  test_params_t* params,
  obj_t*         alpha,
  obj_t*         x,
  obj_t*         y,
  obj_t*         a,
  obj_t*         a_orig
){

  num_t dt     = bli_obj_dt( x );
  uplo_t uploa = bli_obj_uplo( a );
  dim_t M      = bli_obj_length( a );
  dim_t N      = bli_obj_width( a );
  dim_t incx   = bli_obj_vector_inc( x );
  dim_t incy   = bli_obj_vector_inc( y );
  bool conjx   = bli_obj_has_conj( x );
  bool conjy   = bli_obj_has_conj( y );
  dim_t rsa    = bli_obj_row_stride( a ) ;
  dim_t csa    = bli_obj_col_stride( a ) ;
  double resid = 0.0;

  switch( dt )  {
    case BLIS_FLOAT :
    {
      float*   Alpha    = (float*) bli_obj_buffer( alpha );
      float*   A        = (float*) bli_obj_buffer( a_orig );
      float*   X        = (float*) bli_obj_buffer( x );
      float*   Y        = (float*) bli_obj_buffer( y );
      float*   AA       = (float*) bli_obj_buffer( a );
      libblis_isyr2_check<float, int32_t>(uploa, M, Alpha, X, incx,
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
      libblis_isyr2_check<double, int64_t>(uploa, M, Alpha, X, incx,
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
      libblis_icsyr2_check<scomplex, int32_t>(uploa, M, Alpha, X, incx, conjx,
                                               Y, incy, conjy, A, rsa, csa);
      resid = computediffim(M, N, AA, A, rsa, csa);
      break;
    }
    case BLIS_DCOMPLEX :
    {
      dcomplex*   Alpha = (dcomplex*) bli_obj_buffer( alpha );
      dcomplex*   A     = (dcomplex*) bli_obj_buffer( a_orig );
      dcomplex*   X     = (dcomplex*) bli_obj_buffer( x );
      dcomplex*   Y    = (dcomplex*) bli_obj_buffer( y );
      dcomplex*   AA    = (dcomplex*) bli_obj_buffer( a );
      libblis_icsyr2_check<dcomplex, int64_t>(uploa, M, Alpha, X, incx, conjx,
                                               Y, incy, conjy, A, rsa, csa);
      resid = computediffim(M, N, AA, A, rsa, csa);
      break;
    }
    default :
      bli_check_error_code( BLIS_INVALID_DATATYPE );
  }
  return resid;
}


