#include "blis_test.h"
#include "blis_utils.h"
#include "test_scal2v.h"

using namespace std;

//*  ==========================================================================
//*> SCAL2V performs vector operations
//*>     y := alpha * conjx(x)
//*>    where x is a vector of length n, and alpha is a scalar.
//*  ==========================================================================

template <typename T, typename U>
void libblis_iscal2v_check(dim_t len, T* alpha, T* X, dim_t incx,
                                                T* Y, dim_t incy) {
  dim_t i, ix, iy;
  //T ONE = 1.0 ;
  T ZERO = 0.0 ;
  T Alpha = alpha[0];

  if (len == 0){
      return;
  }

  ix = 0;
  iy = 0;
  if (Alpha == ZERO) {
    for(i = 0 ; i < len ; i++) {
      Y[iy] = ZERO;
      iy = iy + incy;
    }
  }
  else {
    for(i = 0 ; i < len ; i++) {
      Y[iy] = Alpha * X[ix];
      iy = iy + incy;
      ix = ix + incx;
    }
  }

  return;
}

template <typename T, typename U>
void libblis_icscal2v_check(dim_t len, T* alpha, T* X, dim_t incx,
                                               T* Y, dim_t incy, bool cfx) {
  dim_t i, ix, iy;
  //T ONE  = {1.0 , 0.0} ;
  T ZERO = {0.0 , 0.0} ;
  T Alpha = *alpha;

  if(len == 0) {
      return;
  }

 ix = 0;
  if(cfx) {
    for(i = 0 ; i < len ; i++) {
      X[ix] = conjugate<T>(X[ix]);
      ix = ix + incx;
    }
  }

  ix = 0;
  iy = 0;
  if ((Alpha.real == ZERO.real) && (Alpha.imag == ZERO.imag)) {
    for(i = 0; i < len ; i++) {
      Y[iy] = ZERO;
      iy = iy + incy;
    }
  }
  else {
    for(i = 0 ; i < len ; i++) {
      Y[iy] = mulc<T>(Alpha , X[ix]);
      ix = ix + incx;
      iy = iy + incy;
    }
  }

  return;
}

double libblis_test_iscal2v_check(
  test_params_t* params,
  obj_t*  alpha,
  obj_t*  x,
  obj_t*  y,
  obj_t*  y_orig
) {
  num_t  dt    = bli_obj_dt( x );
  dim_t  M     = bli_obj_vector_dim( x );
  f77_int incx = bli_obj_vector_inc( x );
  f77_int incy = bli_obj_vector_inc( y_orig );
  bool cfx     = bli_obj_has_conj( x );
  double resid = 0.0;

  switch( dt )  {
    case BLIS_FLOAT :
    {
      float*   Alpha    = (float*) bli_obj_buffer( alpha );
      float*   X        = (float*) bli_obj_buffer( x );
      float*   Y        = (float*) bli_obj_buffer( y_orig );
      float*   YY       = (float*) bli_obj_buffer( y );
      libblis_iscal2v_check<float, int32_t>(M, Alpha, X, incx, Y, incy );
      resid = computediffrv(M, incx, YY, Y);
      break;
    }
    case BLIS_DOUBLE :
    {
      double*   Alpha   = (double*) bli_obj_buffer( alpha );
      double*   X       = (double*) bli_obj_buffer( x );
      double*   Y       = (double*) bli_obj_buffer( y_orig );
      double*   YY      = (double*) bli_obj_buffer( y );
      libblis_iscal2v_check<double, int64_t>(M, Alpha, X, incx, Y, incy );
      resid = computediffrv(M, incx, YY, Y);
      break;
    }
    case BLIS_SCOMPLEX :
    {
      scomplex*   Alpha = (scomplex*) bli_obj_buffer( alpha );
      scomplex*   X     = (scomplex*) bli_obj_buffer( x );
      scomplex*   Y     = (scomplex*) bli_obj_buffer( y_orig );
      scomplex*   YY    = (scomplex*) bli_obj_buffer( y );
      libblis_icscal2v_check<scomplex, int32_t>(M, Alpha, X, incx,
                                                          Y, incy, cfx );
      resid = computediffiv(M, incx, YY, Y);
      break;
    }
    case BLIS_DCOMPLEX :
    {
      dcomplex*   Alpha = (dcomplex*) bli_obj_buffer( alpha );
      dcomplex*   X     = (dcomplex*) bli_obj_buffer( x );
      dcomplex*   Y     = (dcomplex*) bli_obj_buffer( y_orig );
      dcomplex*   YY    = (dcomplex*) bli_obj_buffer( y );
      libblis_icscal2v_check<dcomplex, int64_t>(M, Alpha, X, incx,
                                                          Y, incy, cfx );
      resid = computediffiv(M, incx, YY, Y);
      break;
    }
    default :
      bli_check_error_code( BLIS_INVALID_DATATYPE );
  }

  return resid;
}

