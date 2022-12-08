#include "blis_test.h"
#include "blis_utils.h"
#include "test_dotxv.h"

using namespace std;

//*  ==========================================================================
//*> DOTXV performs vector operations
//*>    rho := beta * rho + alpha * conjx(x)^T * conjy(y)
//*>    where x and y are vectors of length n, and alpha, beta, and rho are scalars.
//*  ==========================================================================

template <typename T, typename U>
void libblis_idotxv_check(dim_t len, T* alpha, T* X, dim_t incx,
                                  T* beta, T* Y, dim_t incy, T* rhorig ) {
  dim_t i, ix, iy;
  T ONE, ZERO;
  ONE = 1.0 ;
  ZERO = 0.0 ;
  T Alpha = alpha[0];
  T Beta  = beta[0];
  T rho   = *rhorig;

  if(len == 0) {
    return;
  }

  rho = rho * Beta;

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
        X[ix] = Alpha * X[ix];
        ix = ix + incx;
      }
    }
  }

  ix = 0;
  iy = 0;
  for(i = 0 ; i < len ; i++) {
    rho = rho + X[ix] * Y[iy];
    ix  = ix + incx;
    iy  = iy + incy;
  }

  *rhorig = rho;
  return;
}

template <typename T, typename U>
void libblis_icdotxv_check(dim_t len, T* alpha, T* X, dim_t incx,  T* beta,
                         T* Y, dim_t incy, T* rhorig, bool cfx, bool cfy) {
  dim_t i, ix, iy;
  T ONE, ZERO;
  ONE  = {1.0 , 0.0};
  ZERO = {0.0 , 0.0};
  T Alpha = *alpha;
  T Beta  = *beta;
  T rho   = *rhorig;

  if (len == 0) {
    return;
  }

  rho = mulc<T>(rho , Beta);

  if(cfx) {
    ix = 0;
    for(i = 0 ; i < len ; i++) {
      X[ix] = conjugate<T>(X[ix]);
      ix = ix + incx;
    }
  }

  if((Alpha.real != ONE.real) && (Alpha.imag != ONE.imag)) {
    ix = 0;
    if((Alpha.real == ZERO.real) && (Alpha.imag == ZERO.imag)) {
      for(i = 0 ; i < len ; i++) {
        X[ix] = ZERO;
        ix = ix + incx;
      }
    }
    else {
      for(i = 0 ; i < len ; i++) {
        X[ix] = mulc<T>(Alpha , X[ix]);
        ix = ix + incx;
      }
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
    rho = addc<T>(rho, mulc<T>(Y[iy] , X[ix]));
    ix = ix + incx;
    iy = iy + incy;
  }

  *rhorig = rho;
  return;
}

double libblis_test_idotxv_check(
  test_params_t* params,
  obj_t*  alpha,
  obj_t*  x,
  obj_t*  y,
  obj_t*  beta,
  obj_t*  rho,
  obj_t*  rho_orig
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
      float*   Alpha    = (float*) bli_obj_buffer( alpha );
      float*   X        = (float*) bli_obj_buffer( x );
      float*   Beta     = (float*) bli_obj_buffer( beta );
      float*   Y        = (float*) bli_obj_buffer( y );
      float*   rhorig   = (float*) bli_obj_internal_scalar_buffer( rho_orig );
      float*   rhp      = (float*) bli_obj_internal_scalar_buffer( rho );
      libblis_idotxv_check<float, int32_t>( M, Alpha, X, incx,
                                                   Beta, Y, incy, rhorig );
      resid = (*rhp - *rhorig);
      break;
    }
    case BLIS_DOUBLE :
    {
      double*   Alpha   = (double*) bli_obj_buffer( alpha );
      double*   X       = (double*) bli_obj_buffer( x );
      double*   Beta    = (double*) bli_obj_buffer( beta );
      double*   Y       = (double*) bli_obj_buffer( y );
      double*   rhorig  = (double*) bli_obj_internal_scalar_buffer( rho_orig );
      double*   rhp     = (double*) bli_obj_internal_scalar_buffer( rho );
      libblis_idotxv_check<double, int64_t>( M, Alpha, X, incx,
                                                   Beta, Y, incy, rhorig );
      resid = (*rhp - *rhorig);
      break;
    }
    case BLIS_SCOMPLEX :
    {
      scomplex*   Alpha  = (scomplex*) bli_obj_buffer( alpha );
      scomplex*   X      = (scomplex*) bli_obj_buffer( x );
      scomplex*   Beta   = (scomplex*) bli_obj_buffer( beta );
      scomplex*   Y      = (scomplex*) bli_obj_buffer( y );
      scomplex*   rhorig = (scomplex*) bli_obj_internal_scalar_buffer( rho_orig );
      scomplex*   rhp    = (scomplex*) bli_obj_internal_scalar_buffer( rho );
      libblis_icdotxv_check<scomplex, int32_t>( M, Alpha, X, incx,
                                         Beta, Y, incy, rhorig, cfx, cfy );
      resid  = ((*rhp).real - (*rhorig).real);
      resid += ((*rhp).imag - (*rhorig).imag);
      break;
    }
    case BLIS_DCOMPLEX :
    {
      dcomplex*   Alpha  = (dcomplex*) bli_obj_buffer( alpha );
      dcomplex*   X      = (dcomplex*) bli_obj_buffer( x );
      dcomplex*   Beta   = (dcomplex*) bli_obj_buffer( beta );
      dcomplex*   Y      = (dcomplex*) bli_obj_buffer( y );
      dcomplex*   rhorig = (dcomplex*) bli_obj_internal_scalar_buffer( rho_orig );
      dcomplex*   rhp    = (dcomplex*) bli_obj_internal_scalar_buffer( rho );
      libblis_icdotxv_check<dcomplex, int64_t>( M, Alpha, X, incx,
                                         Beta, Y, incy, rhorig, cfx, cfy );
      resid  = ((*rhp).real - (*rhorig).real);
      resid += ((*rhp).imag - (*rhorig).imag);
      break;
    }
    default :
      bli_check_error_code( BLIS_INVALID_DATATYPE );
  }

  return abs(resid);
}

