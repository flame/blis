#include "blis_test.h"
#include "blis_utils.h"
#include "test_dotaxpyv.h"

using namespace std;

//*  ==========================================================================
//*> DOTAXPYV performs fused operations
//*>    rho := conjxt(x^T) * conjy(y)
//*>      y   := y + alpha * conjx(x)
//*>    where x, y, and z are vectors of length m and alpha and rho are scalars.
//*>    The kernel is implemented as a fusion of calls to dotv and axpyv
//*  ==========================================================================

template <typename T, typename U>
T libblis_idotaxpyv_check(dim_t len, T* alpha, T* XT, dim_t incxt,
     T* X, dim_t incx, T* Y, dim_t incy, T* Z, dim_t incz ) {

  dim_t i, ixt, ix, iy, iz;
  T ONE = 1.0 ;
  T ZERO = 0.0 ;
  T Alpha = alpha[0];
  T pr = 0.0;
  if (len == 0) {
    return pr;
  }

  ixt = 0;
  iy = 0;
  for(i = 0 ; i < len ; i++) {
    pr = pr + XT[ixt] * Y[iy];
    ixt = ixt + incxt;
    iy = iy + incy;
  }

  if (Alpha != ONE) {
    ix = 0;
    if (Alpha == ZERO) {
      for(i = 0 ; i < len ; i++) {
        X[ix] = ZERO;
        ix = ix + incx;
      }
    }
    else {
      for(i = 0 ; i < len ; i++) {
        X[ix] = Alpha*X[ix];
        ix = ix + incx;
      }
    }
  }

  ix = 0;
  iz = 0;
  for(i = 0 ; i < len ; i++) {
    Z[iz] = Z[iz] + X[ix];
    ix = ix + incx;
    iz = iz + incz;
  }

  return pr;
}

template <typename T, typename U>
T libblis_icdotaxpyv_check(dim_t len, T* alpha, T* XT, dim_t incxt, bool cfxt,
     T* X, dim_t incx, bool cfx, T* Y, dim_t incy, bool cfy, T* Z, dim_t incz ) {

  dim_t i, ixt, ix, iy, iz;
  T ONE  = {1.0 , 0.0};
  T ZERO = {0.0 , 0.0};
  T Alpha = *alpha;

  T pr = {0.0, 0.0};
  if (len == 0) {
      return pr;
  }

  if(cfxt) {
    ixt = 0;
    for(i = 0 ; i < len ; i++) {
      XT[ixt] = conjugate<T>(XT[ixt]);
      ixt = ixt + incxt;
    }
  }

  if(cfy) {
    iy = 0;
    for(i = 0 ; i < len ; i++) {
      Y[iy] = conjugate<T>(Y[iy]);
      iy = iy + incy;
    }
  }

  ixt = 0;
  iy = 0;
  for(i = 0 ; i < len ; i++) {
    pr = addc<T>(pr, mulc<T>(Y[iy] , XT[ixt]));
    ixt = ixt + incxt;
    iy  = iy + incy;
  }

  if(cfx != cfxt) {
    ix = 0;
    for(i = 0 ; i < len ; i++) {
      X[ix] = conjugate<T>(X[ix]);
      ix = ix + incx;
    }
  }

  if ((Alpha.real != ONE.real) && (Alpha.imag != ONE.imag)) {
    ix = 0;
    if ((Alpha.real == ZERO.real) && (Alpha.imag == ZERO.imag)) {
      for(i = 0 ; i < len ; i++) {
        X[ix] = ZERO;
        ix = ix + incx;
      }
    }
    else {
      for(i = 0 ; i < len ; i++) {
        X[ix] = mulc<T>(Alpha, X[ix]);
        ix = ix + incx;
      }
    }
  }

  ix = 0;
  iz = 0;
  for(i = 0 ; i < len ; i++) {
    Z[iz] = addc<T>(Z[iz] , X[ix]);
    ix = ix + incx;
    iz = iz + incz;
  }

  return pr;

}

double libblis_test_idotaxpyv_check (
  test_params_t* params,
  obj_t*         alpha,
  obj_t*         xt,
  obj_t*         x,
  obj_t*         y,
  obj_t*         rho_orig,
  obj_t*         z,
  obj_t*         z_orig
) {
  num_t  dt     = bli_obj_dt( y );
  dim_t  M      = bli_obj_vector_dim( z );
  f77_int incxt = bli_obj_vector_inc( xt );
  f77_int incx  = bli_obj_vector_inc( x );
  f77_int incy  = bli_obj_vector_inc( y );
  f77_int incz  = bli_obj_vector_inc( z );
  bool cfxt     = bli_obj_has_conj( xt );
  bool cfx      = bli_obj_has_conj( x );
  bool cfy      = bli_obj_has_conj( y );
  double r1,r2,resid ;
  r1 = r2 = resid  = 0.0;

  switch( dt )  {
    case BLIS_FLOAT :
    {
      float*   Alpha    = (float*) bli_obj_buffer( alpha );
      float*   X        = (float*) bli_obj_buffer( x );
      float*   XT       = (float*) bli_obj_buffer( xt );
      float*   Y        = (float*) bli_obj_buffer( y );
      float*   Z        = (float*) bli_obj_buffer( z_orig );
      float*   ZZ       = (float*) bli_obj_buffer( z );
      float*   av       = (float*) bli_obj_buffer( rho_orig );
      float ref = libblis_idotaxpyv_check<float, int32_t>( M, Alpha,
                                   XT, incxt, X, incx, Y, incy, Z, incz );
      r1 = computediffrv(M, incy, ZZ, Z);
      r2 = (*av - ref);
      break;
    }
    case BLIS_DOUBLE :
    {
      double*   Alpha    = (double*) bli_obj_buffer( alpha );
      double*   X        = (double*) bli_obj_buffer( x );
      double*   XT       = (double*) bli_obj_buffer( xt );
      double*   Y        = (double*) bli_obj_buffer( y );
      double*   Z        = (double*) bli_obj_buffer( z_orig );
      double*   ZZ       = (double*) bli_obj_buffer( z );
      double*   av       = (double*) bli_obj_buffer( rho_orig );
      double ref = libblis_idotaxpyv_check<double, int64_t>( M, Alpha,
                                   XT, incxt, X, incx, Y, incy, Z, incz );
      r1 = computediffrv(M, incy, ZZ, Z);
      r2 = (*av - ref);
      break;
    }
    case BLIS_SCOMPLEX :
    {
      scomplex*   Alpha  = (scomplex*) bli_obj_buffer( alpha );
      scomplex*   X      = (scomplex*) bli_obj_buffer( x );
      scomplex*   XT     = (scomplex*) bli_obj_buffer( xt );
      scomplex*   Y      = (scomplex*) bli_obj_buffer( y );
      scomplex*   Z      = (scomplex*) bli_obj_buffer( z_orig );
      scomplex*   ZZ     = (scomplex*) bli_obj_buffer( z );
      scomplex*   av     = (scomplex*) bli_obj_buffer( rho_orig );
      scomplex ref = libblis_icdotaxpyv_check<scomplex, int32_t>( M, Alpha,
                   XT, incxt, cfxt, X, incx, cfx, Y, incy, cfy, Z, incz );
      r1 = computediffiv(M, incy, ZZ, Z);
      r2 = ((*av).real - ref.real);
      r2 +=((*av).imag - ref.imag);
      break;
    }
    case BLIS_DCOMPLEX :
    {
      dcomplex*   Alpha  = (dcomplex*) bli_obj_buffer( alpha );
      dcomplex*   X      = (dcomplex*) bli_obj_buffer( x );
      dcomplex*   XT     = (dcomplex*) bli_obj_buffer( xt );
      dcomplex*   Y      = (dcomplex*) bli_obj_buffer( y );
      dcomplex*   Z      = (dcomplex*) bli_obj_buffer( z_orig );
      dcomplex*   ZZ     = (dcomplex*) bli_obj_buffer( z );
      dcomplex*   av     = (dcomplex*) bli_obj_buffer( rho_orig );
      dcomplex ref = libblis_icdotaxpyv_check<dcomplex, int64_t>( M, Alpha,
                   XT, incxt, cfxt, X, incx, cfx, Y, incy, cfy, Z, incz );
      r1 = computediffiv(M, incy, ZZ, Z);
      r2 = ((*av).real - ref.real);
      r2 +=((*av).imag - ref.imag);
      break;
    }
    default :
      bli_check_error_code( BLIS_INVALID_DATATYPE );
  }

  resid = abs(bli_fmaxabs( r1, r2 ));
  return resid;
}