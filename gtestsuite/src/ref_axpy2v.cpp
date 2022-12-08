#include "blis_test.h"
#include "blis_utils.h"
#include "test_axpy2v.h"

using namespace std;

//*  ==========================================================================
//*> AXPY2V performs vector operations
//*>    z := y + alphax * conjx(x) + alphay * conjy(y)
//*>    where x, y, and z are vectors of length m. The kernel is implemented
//*>    as a fused pair of calls to axpyv.
//*  ==========================================================================

template <typename T, typename U>
void libblis_iaxpy2v_check(dim_t len, T* alphax, T* alphay, T* X, dim_t incx,
                                        T* Y, dim_t incy, T* Z, dim_t incz) {
  dim_t i, ix, iy, iz;
  T ONE, ZERO;
  ONE = 1.0 ;
  ZERO = 0.0 ;
  T Alphax = alphax[0];
  T Alphay = alphay[0];

  if (len == 0){
    return;
  }

  if (Alphax != ONE) {
    ix = 0;
    if (Alphax == ZERO) {
      for(i = 0 ; i < len ; i++) {
        X[ix] = ZERO;
        ix = ix + incx;
      }
    }
    else {
      for(i = 0 ; i < len ; i++) {
        X[ix] = Alphax * X[ix];
        ix = ix + incx;
      }
    }
  }

  if (Alphay != ONE) {
    iy = 0;
    if (Alphay == ZERO) {
      for(i = 0 ; i < len ; i++) {
        Y[iy] = ZERO;
        iy = iy + incy;
      }
    }
    else {
      for(i = 0 ; i < len ; i++) {
        Y[iy] = Alphay * Y[iy];
        iy = iy + incy;
      }
    }
  }

  ix = 0;
  iy = 0;
  iz = 0;
  for(i = 0 ; i < len ; i++) {
    Z[iz] = Z[iz] + X[ix] + Y[iy] ;
    ix = ix + incx;
    iy = iy + incy;
    iz = iz + incz;
  }

  return;
}

template <typename T, typename U>
void libblis_icaxpy2v_check(dim_t len, T* alphax, T* alphay, T* X, dim_t incx,
                      bool cfx, T* Y, dim_t incy, bool cfy, T* Z, dim_t incz) {

  dim_t i, ix, iy, iz;
  T ONE, ZERO;
  ONE  = {1.0 , 0.0};
  ZERO = {0.0 , 0.0};
  T Alphax = *alphax;
  T Alphay = *alphay;

  if (len == 0) {
    return;
  }

  if(cfx) {
    ix = 0;
    for(i = 0 ; i < len ; i++) {
      X[ix] = conjugate<T>(X[ix]);
      ix = ix + incx;
    }
  }

  if(cfy) {
    iy = 0;
    for(i = 0 ; i < len ; i++) {
      Y[iy] = conjugate<T>(Y[iy]);
      iy = iy + incy;
    }
  }

  if ((Alphax.real != ONE.real) && (Alphax.imag != ONE.imag)) {
    ix = 0;
    if ((Alphax.real == ZERO.real) && (Alphax.imag == ZERO.imag)) {
      for(i = 0; i < len ; i++) {
        X[ix] = ZERO;
        ix = ix + incx;
      }
    }
    else {
      for(i = 0 ; i < len ; i++) {
        X[ix] = mulc<T>(Alphax , X[ix]);
        ix = ix + incx;
      }
    }
  }

  if ((Alphay.real != ONE.real) && (Alphay.imag != ONE.imag)) {
    iy = 0;
    if ((Alphay.real == ZERO.real) && (Alphay.imag == ZERO.imag)) {
      for(i = 0; i < len ; i++) {
        Y[iy] = ZERO;
        iy = iy + incy;
      }
    }
    else {
      for(i = 0 ; i < len ; i++) {
        Y[iy] = mulc<T>(Alphay , Y[iy]);
        iy = iy + incy;
      }
    }
  }

  ix = 0;
  iy = 0;
  iz = 0;
  for(i = 0 ; i < len ; i++) {
    auto xx = X[ix];
    auto yy = Y[iy];
    auto zz = Z[iz];
    zz.real = zz.real + xx.real + yy.real ;
    zz.imag = zz.imag + xx.imag + yy.imag ;
    Z[iz] = zz;
    ix = ix + incx;
    iy = iy + incy;
    iz = iz + incz;
  }

  return;
}

double libblis_test_iaxpy2v_check (
  test_params_t* params,
  obj_t*         alphax,
  obj_t*         alphay,
  obj_t*         x,
  obj_t*         y,
  obj_t*         z,
  obj_t*         z_orig
) {
  num_t  dt    = bli_obj_dt( z );
  dim_t  M     = bli_obj_vector_dim( z );
  f77_int incx = bli_obj_vector_inc( x );
  f77_int incy = bli_obj_vector_inc( y );
  bool cfx     = bli_obj_has_conj( x );
  bool cfy     = bli_obj_has_conj( y );
  f77_int incz = bli_obj_vector_inc( z );
  double resid = 0.0;

  switch( dt )  {
    case BLIS_FLOAT :
    {
      float*   Alphax   = (float*) bli_obj_buffer( alphax );
      float*   Alphay   = (float*) bli_obj_buffer( alphay );
      float*   X        = (float*) bli_obj_buffer( x );
      float*   Y        = (float*) bli_obj_buffer( y );
      float*   Z        = (float*) bli_obj_buffer( z_orig );
      float*   ZZ       = (float*) bli_obj_buffer( z );
      libblis_iaxpy2v_check<float, int32_t>( M, Alphax, Alphay, X, incx,
                                                        Y, incy, Z, incz );
      resid = computediffrv(M, incz, ZZ, Z);
      break;
    }
    case BLIS_DOUBLE :
    {
      double*   Alphax  = (double*) bli_obj_buffer( alphax );
      double*   Alphay  = (double*) bli_obj_buffer( alphay );
      double*   X       = (double*) bli_obj_buffer( x );
      double*   Y       = (double*) bli_obj_buffer( y );
      double*   Z       = (double*) bli_obj_buffer( z_orig );
      double*   ZZ      = (double*) bli_obj_buffer( z );
      libblis_iaxpy2v_check<double, int64_t>( M, Alphax, Alphay, X, incx,
                                                        Y, incy, Z, incz );
      resid = computediffrv(M, incz, ZZ, Z);
      break;
    }
    case BLIS_SCOMPLEX :
    {
      scomplex*   Alphax = (scomplex*) bli_obj_buffer( alphax );
      scomplex*   Alphay = (scomplex*) bli_obj_buffer( alphay );
      scomplex*   X      = (scomplex*) bli_obj_buffer( x );
      scomplex*   Y      = (scomplex*) bli_obj_buffer( y );
      scomplex*   Z      = (scomplex*) bli_obj_buffer( z_orig );
      scomplex*   ZZ     = (scomplex*) bli_obj_buffer( z );
      libblis_icaxpy2v_check<scomplex, int32_t>( M, Alphax, Alphay, X, incx,
                                                cfx, Y, incy, cfy, Z, incz );
      resid = computediffiv(M, incz, ZZ, Z);
      break;
    }
    case BLIS_DCOMPLEX :
    {
      dcomplex*   Alphax = (dcomplex*) bli_obj_buffer( alphax );
      dcomplex*   Alphay = (dcomplex*) bli_obj_buffer( alphay );
      dcomplex*   X      = (dcomplex*) bli_obj_buffer( x );
      dcomplex*   Y      = (dcomplex*) bli_obj_buffer( y );
      dcomplex*   Z      = (dcomplex*) bli_obj_buffer( z_orig );
      dcomplex*   ZZ     = (dcomplex*) bli_obj_buffer( z );
      libblis_icaxpy2v_check<dcomplex, int64_t>( M, Alphax, Alphay, X, incx,
                                                cfx, Y, incy, cfy, Z, incz );
      resid = computediffiv(M, incz, ZZ, Z);
      break;
    }
    default :
      bli_check_error_code( BLIS_INVALID_DATATYPE );
  }

  return resid;
}

