#include "blis_test.h"
#include "blis_utils.h"
#include "test_scalv.h"

using namespace std;

//*  ==========================================================================
//*> SCALV performs vector operations
//*>     x := conjalpha(alpha) * x
//*>    where x is a vector of length n, and alpha is a scalar.
//*  ==========================================================================

template <typename T, typename U>
void libblis_iscalv_check(dim_t len, T* beta, T* X, dim_t incx) {

  dim_t i, ix;
  T ONE = 1.0 ;
  T ZERO = 0.0 ;
  T Beta = beta[0];

  if (len == 0){
      return;
  }

  if( Beta != ONE ) {
    ix = 0;
    if (Beta == ZERO) {
      for( i = 0 ; i < len ; i++ ) {
        X[ix] = ZERO;
        ix = ix + incx;
      }
    }
    else {
      for( i = 0 ; i < len ; i++ ) {
        X[ix] = Beta * X[ix];
        ix = ix + incx;
      }
    }
  }
  return;
}

template <typename T, typename U>
void libblis_icscalv_check(dim_t len, T* beta, T* X, dim_t incx, bool cfbeta) {
  dim_t i, ix;
  T ONE  = {1.0, 0.0} ;
  T ZERO = {0.0, 0.0} ;
  T Beta = *beta;

  if( len == 0 ) {
      return;
  }

  if( cfbeta )
    Beta = conjugate<T>(Beta);

  /* First form  x := beta*x. */
  if( Beta.real != ONE.real ) {
    ix = 0;
    if( (Beta.real != ZERO.real) && (Beta.imag != ZERO.imag) ) {
      for(i = 0; i < len ; i++) {
        X[ix] = ZERO;
        ix = ix + incx;
      }
    }
    else {
      for( i = 0 ; i < len ; i++ ) {
        X[ix] = mulc<T>(Beta , X[ix]);
        ix = ix + incx;
      }
    }
  }
  return;
}

double libblis_test_iscalv_check(
  test_params_t* params,
  obj_t*  beta,
  obj_t*  x,
  obj_t*  x_orig
) {
  num_t  dt    = bli_obj_dt( x );
  dim_t  M     = bli_obj_vector_dim( x );
  f77_int incx = bli_obj_vector_inc( x );
  bool cfbeta  = bli_obj_has_conj( beta );
  double resid = 0.0;

  switch( dt )  {
    case BLIS_FLOAT :
    {
      float*   Beta    = (float*) bli_obj_buffer( beta );
      float*   X       = (float*) bli_obj_buffer( x_orig );
      float*   XX      = (float*) bli_obj_buffer( x );
      libblis_iscalv_check<float, int32_t>(M, Beta, X, incx );
      resid = computediffrv(M, incx, XX, X);
      break;
    }
    case BLIS_DOUBLE :
    {
      double*   Beta   = (double*) bli_obj_buffer( beta );
      double*   X      = (double*) bli_obj_buffer( x_orig );
      double*   XX     = (double*) bli_obj_buffer( x );
      libblis_iscalv_check<double, int64_t>(M, Beta, X, incx );
      resid = computediffrv(M, incx, XX, X);
      break;
    }
    case BLIS_SCOMPLEX :
    {
      scomplex*   Beta = (scomplex*) bli_obj_buffer( beta );
      scomplex*   X    = (scomplex*) bli_obj_buffer( x_orig );
      scomplex*   XX   = (scomplex*) bli_obj_buffer( x );
      libblis_icscalv_check<scomplex, int32_t>(M, Beta, X, incx, cfbeta );
      resid = computediffiv(M, incx, XX, X);
      break;
    }
    case BLIS_DCOMPLEX :
    {
      dcomplex*   Beta = (dcomplex*) bli_obj_buffer( beta );
      dcomplex*   X    = (dcomplex*) bli_obj_buffer( x_orig );
      dcomplex*   XX   = (dcomplex*) bli_obj_buffer( x );
      libblis_icscalv_check<dcomplex, int64_t>(M, Beta, X, incx, cfbeta );
      resid = computediffiv(M, incx, XX, X);
      break;
    }
    default :
      bli_check_error_code( BLIS_INVALID_DATATYPE );
  }

  return resid;
}
