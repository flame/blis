#include "blis_test.h"
#include "blis_utils.h"
#include "test_dotv.h"

using namespace std;

//*  ==========================================================================
//*> DOTV performs vector operations
//*>    rho := conjx(x)^T * conjy(y)
//*>    where x and y are vectors of length n, and rho is a scalar.
//*  ==========================================================================

template <typename T, typename U>
T libblis_idotv_check(dim_t len, T* X, dim_t incx, T* Y, dim_t incy) {

  dim_t i, ix, iy;
  T pr = 0.0;
  if (len == 0) {
    return pr;
  }

  ix = 0;
  iy = 0;
  for(i = 0 ; i < len ; i++) {
    pr = pr + X[ix] * Y[iy];
    ix = ix + incx;
    iy = iy + incy;
  }

  return pr;
}

template <typename T, typename U>
T libblis_icdotv_check(dim_t len, T* X, dim_t incx, T* Y,
                                        dim_t incy, bool cfx, bool cfy) {
  dim_t i, ix, iy;
  T pr = {0.0, 0.0};
  if (len == 0) {
      return pr;
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

  ix = 0;
  iy = 0;
  for(i = 0 ; i < len ; i++) {
    pr = addc<T>(pr, mulc<T>(Y[iy] , X[ix]));
    ix = ix + incx;
    iy = iy + incy;
  }

  return pr;
}

double libblis_test_idotv_check(
  test_params_t* params,
  obj_t*  x,
  obj_t*  y,
  obj_t*  rho
) {
  num_t  dt    = bli_obj_dt( x );
  dim_t  M     = bli_obj_vector_dim( x );
  bool cfx     = bli_obj_has_conj( x );
  bool cfy     = bli_obj_has_conj( y );
  f77_int incx = bli_obj_vector_inc( x );
  f77_int incy = bli_obj_vector_inc( y );
  double resid = 0.0;

  switch( dt )  {
    case BLIS_FLOAT :
    {
      float*   X        = (float*) bli_obj_buffer( x );
      float*   Y        = (float*) bli_obj_buffer( y );
      float*   av       = (float*) bli_obj_internal_scalar_buffer( rho );
      float ref = libblis_idotv_check<float, int32_t>( M, X, incx, Y, incy );
      resid = (*av - ref);
      break;
    }
    case BLIS_DOUBLE :
    {
      double*   X       = (double*) bli_obj_buffer( x );
      double*   Y       = (double*) bli_obj_buffer( y );
      double*   av      = (double*) bli_obj_internal_scalar_buffer( rho );
      double ref = libblis_idotv_check<double, int64_t>( M, X, incx, Y, incy );
      resid = (*av - ref);
      break;
    }
    case BLIS_SCOMPLEX :
    {
      scomplex*   X     = (scomplex*) bli_obj_buffer( x );
      scomplex*   Y     = (scomplex*) bli_obj_buffer( y );
      scomplex*   av    = (scomplex*) bli_obj_internal_scalar_buffer( rho );
      scomplex ref = libblis_icdotv_check<scomplex, int32_t>( M, X, incx,
                                                         Y, incy, cfx, cfy );
      resid = ((*av).real - ref.real);
      resid +=((*av).imag - ref.imag);
      break;
    }
    case BLIS_DCOMPLEX :
    {
      dcomplex*   X     = (dcomplex*) bli_obj_buffer( x );
      dcomplex*   Y     = (dcomplex*) bli_obj_buffer( y );
      dcomplex*   av    = (dcomplex*) bli_obj_internal_scalar_buffer( rho );
      dcomplex ref = libblis_icdotv_check<dcomplex, int64_t>( M, X, incx,
                                                         Y, incy, cfx, cfy );
      resid = ((*av).real - ref.real);
      resid +=((*av).imag - ref.imag);
      break;
    }
    default :
      bli_check_error_code( BLIS_INVALID_DATATYPE );
  }

  return resid;
}