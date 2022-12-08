#include "blis_test.h"
#include "blis_utils.h"
#include "test_hemv.h"

using namespace std;

//*  ==========================================================================
//*> HEMV performs the matrix-vector  operation
//*>    y := alpha*A*x + beta*y
//*> where alpha and beta are scalars, x and y are n element vectors and
//*> A is an n by n hermitian matrix.
//*  ==========================================================================

template <typename T, typename U>
void libblis_ihemv_check(uplo_t uploa, dim_t M, T* alpha, T* A,
   dim_t rsa, dim_t csa, T* X, dim_t incx, T* beta, T* Y, dim_t incy) {
  T ONE = 1.0;
  T ZERO = 0.0;
  T Alpha = alpha[0];
  T Beta = beta[0];
  T tmp1, tmp2;
  dim_t i, ix, iy, j, jx, jy, kx, ky;

  if ((M == 0) ||
    ((Alpha == ZERO) && (Beta == ONE)))
      return ;

  //*     Set up the start points in  X  and  Y.
  if (incx > 0) {
    kx = 0;
  }
  else {
    kx = 1 - (M * incx);
  }
  if (incy > 0) {
    ky = 0;
  }
  else {
    ky = 1 - (M * incy);
  }

  //*     First form  y := beta*y.
  if (Beta != ONE) {
    iy = ky;
    if (Beta == ZERO) {
      for(i = 0 ; i < M ; i++) {
        Y[iy] = ZERO;
        iy = iy + incy;
      }
    }
    else {
      for(i = 0 ; i < M ; i++) {
        Y[iy] = (Beta * Y[iy]);
        iy = iy + incy;
      }
    }
  }

  if (Alpha == ZERO)
    return;

  T tmp = 0.0 ;
  if(uploa == BLIS_UPPER) {
    //* Form  y  when A is stored in upper triangle.
    jx = kx;
    jy = ky;
    for(j = 0 ; j < M ; j++) {
      tmp1 = (Alpha * X[jx]);
      tmp2 = ZERO;
      ix = kx;
      iy = ky;
      for(i = 0 ; i < j ; i++) {
        tmp = A[i*rsa + j*csa];
        Y[iy] = Y[iy] + (tmp1 * tmp);
        tmp2 = tmp2 + (tmp * X[ix]);
        ix = ix + incx;
        iy = iy + incy;
      }
      tmp = A[j*rsa + j*csa];
      Y[jy] = Y[jy] + (tmp1 * tmp) + (Alpha * tmp2);
      jx = jx + incx;
      jy = jy + incy;
    }
  }
  else {
    //* Form  y  when A is stored in lower triangle.
    jx = kx;
    jy = ky;
    for(j = 0 ; j < M ; j++) {
      tmp1 = (Alpha * X[jx]);
      tmp = A[j*rsa + j*csa];
      tmp2 = ZERO;
      Y[jy] = Y[jy] + (tmp1 * tmp);
      ix = jx;
      iy = jy;
      for(i = (j+1) ; i < M ; i++) {
        ix = ix + incx;
        iy = iy + incy;
        tmp = A[i*rsa + j*csa];
        Y[iy] = Y[iy] + (tmp1 * tmp);
        tmp2 = tmp2 + (tmp * X[ix]);
      }
      Y[jy] = Y[jy] + (Alpha * tmp2);
      jx = jx + incx;
      jy = jy + incy;
    }
  }

  return;
}

template <typename T, typename U>
void libblis_ichemv_check(uplo_t uploa, dim_t M, T* alpha, T* A, dim_t rsa,
dim_t csa, bool conja, T* X, dim_t incx, bool conjx, T* beta, T* Y, dim_t incy) {
  T ONE   = { 1.0, 0.0 };
  T ZERO  = { 0.0, 0.0 };
  T Alpha = *alpha;
  T Beta  = *beta;
  T tmp1, tmp2;
  dim_t i, ix, iy, j, jx, jy, kx, ky;

  if ((M == 0) ||
    ((Alpha.real == ZERO.real) && (Beta.real == ONE.real)))
      return ;

  //*     Set up the start points in  X  and  Y.
  if (incx > 0) {
    kx = 0;
  }
  else {
    kx = 1 - (M * incx);
  }
  if (incy > 0) {
    ky = 0;
  }
  else {
    ky = 1 - (M * incy);
  }

  //*     First form  y := beta*y.
  if((Beta.real != ONE.real) && (Beta.imag != ONE.imag)) {
    iy = ky;
    if((Beta.real != ZERO.real) && (Beta.imag != ZERO.imag)) {
      for(i = 0 ; i < M ; i++) {
        Y[iy] = ZERO;
        iy = iy + incy;
      }
    }
    else {
      for(i = 0 ; i < M ; i++) {
        Y[iy] = mulc<T>(Beta , Y[iy]);
        iy = iy + incy;
      }
    }
  }

  if((Alpha.real == ZERO.real) && (Alpha.imag == ZERO.imag))
    return;

  if(conjx) {
    ix = 0;
    for(i = 0 ; i < M ; i++) {
      X[ix] = conjugate<T>(X[ix]);
      ix = ix + incx;
    }
  }

  if(conja) {
    for(i = 0 ; i < M ; i++) {
      for(j = 0 ; j < M ; j++) {
        A[i*rsa + j*csa] = conjugate<T>(A[i*rsa + j*csa]);
      }
    }
  }

  T tmp = {0.0, 0.0};
  if(uploa == BLIS_UPPER) {
    //* Form  y  when A is stored in upper triangle.
    jx = kx;
    jy = ky;
    for(j = 0 ; j < M ; j++) {
      tmp1 = mulc<T>(Alpha , X[jx]);
      tmp2 = ZERO;
      ix = kx;
      iy = ky;
      for(i = 0 ; i < j ; i++) {
        tmp = A[i*rsa + j*csa];
        Y[iy] = addc<T>(Y[iy] , mulc<T>(tmp1 , tmp));
        tmp2  = addc<T>(tmp2 , mulc<T>(conjugate<T>(tmp) , X[ix]));
        ix = ix + incx;
        iy = iy + incy;
      }
      tmp = A[j*rsa + j*csa];
      tmp = addc<T>(mulc<T>(tmp1 , real<T>(tmp)) , mulc<T>(Alpha , tmp2));
      Y[jy] = addc<T>(Y[jy] , tmp );
      jx = jx + incx;
      jy = jy + incy;
    }
  }
  else {
    //* Form  y  when A is stored in lower triangle.
    jx = kx;
    jy = ky;
    for(j = 0 ; j < M ; j++) {
      tmp1 = mulc<T>(Alpha , X[jx]);
      tmp  = A[j*rsa + j*csa];
      tmp2 = ZERO;
      Y[jy] = addc<T>(Y[jy] , mulc<T>(tmp1 , real<T>(tmp)));
      ix = jx;
      iy = jy;
      for(i = (j+1) ; i < M ; i++) {
        ix = ix + incx;
        iy = iy + incy;
        tmp = A[i*rsa + j*csa];
        Y[iy] = addc<T>(Y[iy] , mulc<T>(tmp1 , tmp));
        tmp2  = addc<T>(tmp2 , mulc<T>(conjugate<T>(tmp) , X[ix]));
      }
      Y[jy] = addc<T>(Y[jy] , mulc<T>(Alpha , tmp2));
      jx = jx + incx;
      jy = jy + incy;
    }
  }

  return;
}

double libblis_test_ihemv_check(
  test_params_t* params,
  obj_t*         alpha,
  obj_t*         a,
  obj_t*         x,
  obj_t*         beta,
  obj_t*         y,
  obj_t*         y_orig
){
  num_t  dt    = bli_obj_dt( a );
  uplo_t uploa = bli_obj_uplo( a );
  dim_t M      = bli_obj_length( a );
  dim_t rsa    = bli_obj_row_stride( a );
  dim_t csa    = bli_obj_col_stride( a );
  bool conja   = bli_obj_has_conj( a );
  dim_t incx   = bli_obj_vector_inc( x );
  dim_t incy   = bli_obj_vector_inc( y );
  bool conjx   = bli_obj_has_conj( x );
  double resid = 0.0;

  switch( dt )  {
    case BLIS_FLOAT :
    {
      float*   Alpha = (float*) bli_obj_buffer( alpha );
      float*   A     = (float*) bli_obj_buffer( a );
      float*   X     = (float*) bli_obj_buffer( x );
      float*   Beta  = (float*) bli_obj_buffer( beta );
      float*   Y     = (float*) bli_obj_buffer( y_orig );
      float*   YY    = (float*) bli_obj_buffer( y );
      libblis_ihemv_check<float, int32_t>(uploa, M, Alpha, A, rsa, csa,
                                                 X, incx, Beta, Y, incy);
      resid = computediffrv(M, incy, YY, Y);
      break;
    }
    case BLIS_DOUBLE :
    {
      double*   Alpha   = (double*) bli_obj_buffer( alpha );
      double*   A       = (double*) bli_obj_buffer( a );
      double*   X       = (double*) bli_obj_buffer( x );
      double*   Beta    = (double*) bli_obj_buffer( beta );
      double*   Y       = (double*) bli_obj_buffer( y_orig );
      double*   YY      = (double*) bli_obj_buffer( y );
      libblis_ihemv_check<double, int64_t>(uploa, M, Alpha, A, rsa, csa,
                                                 X, incx, Beta, Y, incy);
      resid = computediffrv(M, incy, YY, Y);
      break;
    }
    case BLIS_SCOMPLEX :
    {
      scomplex*   Alpha = (scomplex*) bli_obj_buffer( alpha );
      scomplex*   A     = (scomplex*) bli_obj_buffer( a );
      scomplex*   X     = (scomplex*) bli_obj_buffer( x );
      scomplex*   Beta  = (scomplex*) bli_obj_buffer( beta );
      scomplex*   Y     = (scomplex*) bli_obj_buffer( y_orig );
      scomplex*   YY    = (scomplex*) bli_obj_buffer( y );
      libblis_ichemv_check<scomplex, int32_t>(uploa, M, Alpha, A, rsa, csa,
                                    conja, X, incx, conjx, Beta, Y, incy);
      resid = computediffiv(M, incy, YY, Y);
      break;
    }
    case BLIS_DCOMPLEX :
    {
      dcomplex*   Alpha = (dcomplex*) bli_obj_buffer( alpha );
      dcomplex*   A     = (dcomplex*) bli_obj_buffer( a );
      dcomplex*   X     = (dcomplex*) bli_obj_buffer( x );
      dcomplex*   Beta  = (dcomplex*) bli_obj_buffer( beta );
      dcomplex*   Y     = (dcomplex*) bli_obj_buffer( y_orig );
      dcomplex*   YY    = (dcomplex*) bli_obj_buffer( y );
      libblis_ichemv_check<dcomplex, int64_t>(uploa, M, Alpha, A, rsa, csa,
                                    conja, X, incx, conjx, Beta, Y, incy);
      resid = computediffiv(M, incy, YY, Y);
      break;
    }
    default :
      bli_check_error_code( BLIS_INVALID_DATATYPE );
  }

  return resid;
}