#include "blis_test.h"
#include "blis_utils.h"
#include "test_axpyv.h"

using namespace std;

//*  ==========================================================================
//*> AXPYV performs vector operations
//*>    y := y + alpha * conjx(x)
//*>    where x and y are vectors of length n, and alpha is a scalar
//*  ==========================================================================

template <typename T, typename U>
void libblis_iaxpyv_check(dim_t len, T* alpha, T* X, dim_t incx,
                                               T* Y, dim_t incy) {

  dim_t i, ix, iy;
  T ONE, ZERO;
  ONE = 1.0 ;
  ZERO = 0.0 ;
  T Alpha = alpha[0];

  if (len == 0){
      return;
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
  iy = 0;
  for(i = 0 ; i < len ; i++) {
    Y[iy] = Y[iy] + X[ix];
    ix = ix + incx;
    iy = iy + incy;
  }

  return;
}

template <typename T, typename U>
void libblis_icaxpyv_check(dim_t len, T* alpha, T* X, dim_t incx,
                                               T* Y, dim_t incy, bool cfx) {
  dim_t i, ix, iy;
  T ONE, ZERO;
  ONE  = {1.0 , 0.0};
  ZERO = {0.0 , 0.0};
  T Alpha = *alpha;

  if(len == 0){
      return;
  }

  if(cfx) {
    ix = 0;
    for(i = 0 ; i < len ; i++) {
      X[ix] = conjugate<T>(X[ix]);
      ix = ix + incx;
    }
  }

  /* First form  y := beta*y. */
  if (Alpha.real != ONE.real) {
    ix = 0;
    for(i = 0 ; i < len ; i++) {
      X[ix] = mulc<T>(Alpha , X[ix]);
      ix = ix + incx;
    }
  }

  ix = 0;
  iy = 0;
  for(i = 0 ; i < len ; i++) {
    Y[iy] = addc<T>(Y[iy] , X[ix]);
    ix = ix + incx;
    iy = iy + incy;
  }

  return;
}

double libblis_test_iaxpyv_check(
  test_params_t* params,
  obj_t*  alpha,
  obj_t*  x,
  obj_t*  y,
  obj_t*  y_orig
) {
  num_t  dt    = bli_obj_dt( x );
  dim_t  M     = bli_obj_vector_dim( x );
  bool cfx     = bli_obj_has_conj( x );
  f77_int incx = bli_obj_vector_inc( x );
  f77_int incy = bli_obj_vector_inc( y_orig );
  double resid = 0.0;

  switch( dt )  {
    case BLIS_FLOAT :
    {
      float*   Alpha    = (float*) bli_obj_buffer( alpha );
      float*   X        = (float*) bli_obj_buffer( x );
      float*   Y        = (float*) bli_obj_buffer( y_orig );
      float*   YY       = (float*) bli_obj_buffer( y );
      libblis_iaxpyv_check<float, int32_t>( M, Alpha, X, incx,
                                                       Y, incy );
      resid = computediffrv(M, incy, YY, Y);
      break;
    }
    case BLIS_DOUBLE :
    {
      double*   Alpha   = (double*) bli_obj_buffer( alpha );
      double*   X       = (double*) bli_obj_buffer( x );
      double*   Y       = (double*) bli_obj_buffer( y_orig );
      double*   YY      = (double*) bli_obj_buffer( y );
      libblis_iaxpyv_check<double, int64_t>( M, Alpha, X, incx,
                                                        Y, incy );
      resid = computediffrv(M, incy, YY, Y);
      break;
    }
    case BLIS_SCOMPLEX :
    {
      scomplex*   Alpha = (scomplex*) bli_obj_buffer( alpha );
      scomplex*   X     = (scomplex*) bli_obj_buffer( x );
      scomplex*   Y     = (scomplex*) bli_obj_buffer( y_orig );
      scomplex*   YY    = (scomplex*) bli_obj_buffer( y );
      libblis_icaxpyv_check<scomplex, int32_t>( M, Alpha, X, incx,
                                                        Y, incy, cfx );

      resid = computediffiv(M, incy, YY, Y);
      break;
    }
    case BLIS_DCOMPLEX :
    {
      dcomplex*   Alpha = (dcomplex*) bli_obj_buffer( alpha );
      dcomplex*   X     = (dcomplex*) bli_obj_buffer( x );
      dcomplex*   Y     = (dcomplex*) bli_obj_buffer( y_orig );
      dcomplex*   YY    = (dcomplex*) bli_obj_buffer( y );
      libblis_icaxpyv_check<dcomplex, int64_t>( M, Alpha, X, incx,
                                                        Y, incy, cfx );
      resid = computediffiv(M, incy, YY, Y);
      break;
    }
    default :
      bli_check_error_code( BLIS_INVALID_DATATYPE );
  }

  return resid;
}
