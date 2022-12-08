#include "blis_test.h"
#include "blis_utils.h"
#include "test_normfv.h"

using namespace std;

//*  ==========================================================================
//*> NORMFV performs vector operations
//*>    Compute the Frobenius norm (bli_?normfv())
//*>    of the elements in a vector x of length n. The resulting norm is stored to norm
//*  ==========================================================================

template <typename T, typename U>
T libblis_inormfv_check(dim_t len, T* X, dim_t incx ) {
  dim_t i, ix;
  T sum = 0.0;
  T norm = 0.0;

  if (len == 0){
      return norm;
  }

  ix = 0;
  for(i = 0 ; i < len ; i++) {
    sum += X[ix] * X[ix];
    ix = ix + incx;
  }

  norm = sqrt( abs(sum) );

 	return norm;
}

template <typename T, typename U>
U libblis_icnormfv_check(dim_t len, T* X, dim_t incx ) {
  dim_t i, ix;
  T rr = { 0.0, 0.0 };
  U norm = 0.0;
  if(len == 0) {
      return norm;
  }

  ix = 0;
  for(i = 0 ; i < len ; i++) {
    //rr = addc<T>(rr, mulc<T>(X[ix] , X[ix]));
    auto a = X[ix];
    rr.real += a.real * a.real;
    rr.imag += a.imag * a.imag;
    ix = ix + incx;
  }

  U r = rr.real + rr.imag;
  norm = sqrt( abs(r) );

 	return norm;
}

double libblis_test_inormfv_check(
  test_params_t* params,
  obj_t*  alpha,
  obj_t*  x,
  obj_t*  n
) {
  num_t  dt    = bli_obj_dt( x );
  dim_t  M     = bli_obj_vector_dim( x );
  f77_int incx = bli_obj_vector_inc( x );
  double resid = 0.0;

  switch( dt )  {
    case BLIS_FLOAT :
    {
      float* X    = (float*) bli_obj_buffer( x );
      float* av   = (float*) bli_obj_internal_scalar_buffer( n );
      float rv    = libblis_inormfv_check<float, int32_t>(M, X, incx );
      resid = (double)(abs(rv - *av)/abs(rv));
      break;
    }
    case BLIS_DOUBLE :
    {
      double* X   = (double*) bli_obj_buffer( x );
      double* av  = (double*) bli_obj_internal_scalar_buffer( n );
      double rv   = libblis_inormfv_check<double, int64_t>(M, X, incx );
      resid = (double)(abs(rv - *av)/abs(rv));
      break;
    }
    case BLIS_SCOMPLEX :
    {
      scomplex* X  = (scomplex*) bli_obj_buffer( x );
      float* av    = (float*) bli_obj_internal_scalar_buffer( n );
      float  rv = libblis_icnormfv_check<scomplex, float>(M, X, incx );
      resid = (double)(abs(rv - *av)/abs(rv));
      break;
    }
    case BLIS_DCOMPLEX :
    {
      dcomplex* X  = (dcomplex*) bli_obj_buffer( x );
      double* av   = (double*) bli_obj_internal_scalar_buffer( n );
      double  rv = libblis_icnormfv_check<dcomplex, double>(M, X, incx );
      resid = (double)(abs(rv - *av)/abs(rv));
      break;
    }
    default :
      bli_check_error_code( BLIS_INVALID_DATATYPE );
  }

  return resid;
}
