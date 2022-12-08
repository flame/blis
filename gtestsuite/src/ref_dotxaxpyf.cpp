#include "blis_test.h"
#include "blis_utils.h"
#include "test_dotxaxpyf.h"

using namespace std;

//*  ==========================================================================
//*> DOTXAXPYF performs fused operations
//*>     y := beta * y + alpha * conjat(A^T) * conjw(w)
//*>     z :=        z + alpha * conja(A)    * conjx(x)
//*>    where A is an m x b matrix, w and z are vectors of length m,
//*>    x and y are vectors of length b, and alpha and beta are scalars.
//*>    The kernel is implemented as a fusion of calls to dotxf and axpyf.
//*  ==========================================================================

template <typename T, typename U>
void libblis_idotxaxpyf_check(dim_t M, dim_t N, T* alpha, T* AT, T* A, dim_t rsa,
                 dim_t csa, T* W, dim_t incw, T* X, dim_t incx, T* beta, T* Y,
                 dim_t incy, T* Z, dim_t incz) {

  dim_t i, j, iw, ix, iy, iz;
  T Alpha = alpha[0];
  T Beta  = beta[0];
  T temp;
  if((M == 0) || (N == 0)) {
    return;
  }

  //y := beta * y + alpha * conjat(A^T) * conjw(w)
  iy = 0;
  for(j = 0 ; j < N ; j++) {
    iw = 0;
    temp = 0.0;
    for(i = 0 ; i < M ; i++) {
      temp += W[iw] * AT[i*rsa + j*csa];
      iw = iw + incw;
    }
    temp = Alpha * temp;
    Y[iy] = (Y[iy] * Beta) + temp;
    iy = iy + incy;
  }

  //z :=        z + alpha * conja(A)    * conjx(x)
  ix = 0;
  for(j = 0 ; j < N ; j++) {
    temp = Alpha * X[ix];
    iz = 0;
    for(i = 0 ; i < M ; i++) {
      Z[iz] = Z[iz] + temp * A[i*rsa + j*csa];
      iz = iz + incz;
    }
    ix = ix + incx;
  }

  return;
}

template <typename T, typename U>
void libblis_icdotxaxpyf_check(dim_t M, dim_t N, T* alpha, T* AT, bool conjat,
   T* A, dim_t rsa, dim_t csa, bool conja, T* W, dim_t incw, bool conjw, T* X,
   dim_t incx, bool conjx, T* beta, T* Y, dim_t incy, T* Z, dim_t incz) {

  dim_t i, j, iw, ix, iy, iz;
  //T ONE  = {1.0 , 0.0};
  T ZERO = {0.0 , 0.0};
  T Alpha = *alpha;
  T Beta  = *beta;
  T temp;

  if((M == 0) || (N == 0)) {
    return;
  }

  if(conjw) {
    iw = 0;
    for(i = 0 ; i < M ; i++) {
      W[iw] = conjugate<T>(W[iw]);
      iw = iw + incw;
    }
  }

  if(conjat) {
    for(j = 0 ; j < N ; j++) {
      for(i = 0 ; i < M ; i++) {
        AT[i*rsa + j*csa] = conjugate<T>(AT[i*rsa + j*csa]);
      }
    }
  }

  //y := beta * y + alpha * conjat(A^T) * conjw(w)
  iy = 0;
  for(j = 0 ; j < N ; j++) {
    iw = 0;
    temp = ZERO;
    for(i = 0 ; i < M ; i++) {
      temp = addc<T>(temp , mulc<T>(W[iw] , AT[i*rsa + j*csa]));
      iw = iw + incw;
    }
    temp = mulc<T>(Alpha , temp);
    Y[iy] = addc<T>(temp , mulc<T>(Y[iy] , Beta));
    iy = iy + incy;
  }

  //z := z + alpha * conja(A) * conjx(x)
  if(conjx) {
    ix = 0;
    for(i = 0 ; i < N ; i++) {
      X[ix] = conjugate<T>(X[ix]);
      ix = ix + incx;
    }
  }

  if(conja != conjat) {
    for(j = 0 ; j < N ; j++) {
      for(i = 0 ; i < M ; i++) {
        A[i*rsa + j*csa] = conjugate<T>(A[i*rsa + j*csa]);
      }
    }
  }

  //z :=        z + alpha * conja(A)    * conjx(x)
  ix = 0;
  for(j = 0 ; j < N ; j++) {
    temp = mulc<T>(Alpha , X[ix]);
    iz = 0;
    for(i = 0 ; i < M ; i++) {
      Z[iz] = addc<T>(Z[iz] , mulc<T>(temp , A[i*rsa + j*csa]));
      iz = iz + incz;
    }
    ix = ix + incx;
  }

  return;
}

double libblis_test_idotxaxpyf_check (
  test_params_t* params,
  obj_t*         alpha,
  obj_t*         at,
  obj_t*         a,
  obj_t*         w,
  obj_t*         x,
  obj_t*         beta,
  obj_t*         y,
  obj_t*         z,
  obj_t*         y_orig,
  obj_t*         z_orig
) {
  num_t  dt     = bli_obj_dt( a );
  dim_t  M      = bli_obj_vector_dim( z );
  dim_t  N      = bli_obj_vector_dim( y );
  f77_int rsa   = bli_obj_row_stride( a ) ;
  f77_int csa   = bli_obj_col_stride( a ) ;
  f77_int incw  = bli_obj_vector_inc( w );
  f77_int incx  = bli_obj_vector_inc( x );
  f77_int incy  = bli_obj_vector_inc( y );
  f77_int incz  = bli_obj_vector_inc( z );
  bool conjat   = bli_obj_has_conj( at );
  bool conja    = bli_obj_has_conj( a );
  bool conjw    = bli_obj_has_conj( w );
  bool conjx    = bli_obj_has_conj( x );
  double resid  = 0.0;

  switch( dt )  {
    case BLIS_FLOAT :
    {
      float*   Alpha    = (float*) bli_obj_buffer( alpha );
      float*   AT       = (float*) bli_obj_buffer( at );
      float*   A        = (float*) bli_obj_buffer( a );
      float*   W        = (float*) bli_obj_buffer( w );
      float*   X        = (float*) bli_obj_buffer( x );
      float*   Beta     = (float*) bli_obj_buffer( beta );
      float*   Y        = (float*) bli_obj_buffer( y_orig );
      float*   Z        = (float*) bli_obj_buffer( z_orig );
      float*   YY       = (float*) bli_obj_buffer( y );
      float*   ZZ       = (float*) bli_obj_buffer( z );
      libblis_idotxaxpyf_check<float, int32_t>(M, N, Alpha, AT,
              A, rsa, csa, W, incw, X, incx, Beta, Y, incy, Z, incz);
      resid  = computediffrv(M, incz, ZZ, Z);
      resid += computediffrv(N, incy, YY, Y);
      break;
    }
    case BLIS_DOUBLE :
    {
      double*   Alpha   = (double*) bli_obj_buffer( alpha );
      double*   AT      = (double*) bli_obj_buffer( at );
      double*   A       = (double*) bli_obj_buffer( a );
      double*   W       = (double*) bli_obj_buffer( w );
      double*   X       = (double*) bli_obj_buffer( x );
      double*   Beta    = (double*) bli_obj_buffer( beta );
      double*   Y       = (double*) bli_obj_buffer( y_orig );
      double*   Z       = (double*) bli_obj_buffer( z_orig );
      double*   YY      = (double*) bli_obj_buffer( y );
      double*   ZZ      = (double*) bli_obj_buffer( z );
      libblis_idotxaxpyf_check<double, int64_t>(M, N, Alpha, AT,
              A, rsa, csa, W, incw, X, incx, Beta, Y, incy, Z, incz);
      resid  = computediffrv(M, incz, ZZ, Z);
      resid += computediffrv(N, incy, YY, Y);
      break;
    }
    case BLIS_SCOMPLEX :
    {
      scomplex*   Alpha = (scomplex*) bli_obj_buffer( alpha );
      scomplex*   AT    = (scomplex*) bli_obj_buffer( at );
      scomplex*   A     = (scomplex*) bli_obj_buffer( a );
      scomplex*   W     = (scomplex*) bli_obj_buffer( w );
      scomplex*   X     = (scomplex*) bli_obj_buffer( x );
      scomplex*   Beta  = (scomplex*) bli_obj_buffer( beta );
      scomplex*   Y     = (scomplex*) bli_obj_buffer( y_orig );
      scomplex*   Z     = (scomplex*) bli_obj_buffer( z_orig );
      scomplex*   YY    = (scomplex*) bli_obj_buffer( y );
      scomplex*   ZZ    = (scomplex*) bli_obj_buffer( z );
      libblis_icdotxaxpyf_check<scomplex, int32_t>(M, N, Alpha, AT, conjat,
              A, rsa, csa, conja, W, incw, conjw, X, incx, conjx, Beta,
                                                        Y, incy, Z, incz);
      resid  = computediffiv(M, incz, ZZ, Z);
      resid += computediffiv(N, incy, YY, Y);
      break;
    }
    case BLIS_DCOMPLEX :
    {
      dcomplex*   Alpha = (dcomplex*) bli_obj_buffer( alpha );
      dcomplex*   AT    = (dcomplex*) bli_obj_buffer( at );
      dcomplex*   A     = (dcomplex*) bli_obj_buffer( a );
      dcomplex*   W     = (dcomplex*) bli_obj_buffer( w );
      dcomplex*   X     = (dcomplex*) bli_obj_buffer( x );
      dcomplex*   Beta  = (dcomplex*) bli_obj_buffer( beta );
      dcomplex*   Y     = (dcomplex*) bli_obj_buffer( y_orig );
      dcomplex*   Z     = (dcomplex*) bli_obj_buffer( z_orig );
      dcomplex*   YY    = (dcomplex*) bli_obj_buffer( y );
      dcomplex*   ZZ    = (dcomplex*) bli_obj_buffer( z );
      libblis_icdotxaxpyf_check<dcomplex, int64_t>(M, N, Alpha, AT, conjat,
              A, rsa, csa, conja, W, incw, conjw, X, incx, conjx, Beta,
                                                        Y, incy, Z, incz);
      resid  = computediffiv(M, incz, ZZ, Z);
      resid += computediffiv(N, incy, YY, Y);
      break;
    }
    default :
      bli_check_error_code( BLIS_INVALID_DATATYPE );
  }

  return abs(resid);
}

