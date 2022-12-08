#include "blis_test.h"
#include "blis_utils.h"
#include "test_amaxv.h"

using namespace std;

//*  ==========================================================================
//*> Given a vector of length n, return the zero-based index index of
//*> the element of vector x that contains the largest absolute value
//*> (or, in the complex domain, the largest complex modulus).
//*  ==========================================================================

template <typename T>
dim_t libblis_iamaxv_check(dim_t len, T* X, dim_t incx) {

  dim_t i, ix, iamax = 0;

  if (len == 0) {
      return 0;
  }

  ix = 0;
  T smax = abs(X[ix]);
  for(i = 0 ; i < len ; i++) {
    if(abs(X[ix]) > smax) {
      iamax = i;
      smax = abs(X[ix]);
    }
    ix = ix + incx;
  }

  return iamax;
}

template <typename T, typename U>
dim_t libblis_icamaxv_check(dim_t len, T* X, dim_t incx) {

  dim_t i, ix, iamax = 0;
  if (len == 0) {
      return 0;
  }

  ix = 0;
  U smax = abscomplex(X[ix]);
  for(i = 0 ; i < len ; i++) {
    if(abscomplex(X[ix]) > smax) {
      iamax = i;
      smax = abscomplex(X[ix]);
    }
    ix = ix + incx;
  }

  return iamax;
}

double libblis_test_iamaxv_check(
  test_params_t* params,
  obj_t*  x,
  obj_t*  index
){
  num_t  dt    = bli_obj_dt( x );
  dim_t  M     = bli_obj_vector_dim( x );
  f77_int incx = bli_obj_vector_inc( x );
  double resid = 0.0;
  dim_t ind = 0;

  switch( dt )  {
    case BLIS_FLOAT :
    {
      float*  X      = (float*) bli_obj_buffer( x );
      f77_int* indx  = (f77_int*) bli_obj_buffer( index );
      ind   = libblis_iamaxv_check<float>( M, X, incx );
      resid = (double)(*indx - ind);
      break;
    }
    case BLIS_DOUBLE :
    {
      double*   X     = (double*) bli_obj_buffer( x );
      f77_int* indx   = (f77_int*) bli_obj_buffer( index );
      ind   = libblis_iamaxv_check<double>( M, X, incx );
      resid = (double)(*indx - ind);
      break;
    }
    case BLIS_SCOMPLEX :
    {
      scomplex*   X  = (scomplex*) bli_obj_buffer( x );
      f77_int* indx  = (f77_int*) bli_obj_buffer( index );
      ind   = libblis_icamaxv_check<scomplex, float>( M, X, incx );
      resid = (double)(*indx - ind);
      break;
    }
    case BLIS_DCOMPLEX :
    {
      dcomplex*   X   = (dcomplex*) bli_obj_buffer( x );
      f77_int* indx   = (f77_int*) bli_obj_buffer( index );
      ind   = libblis_icamaxv_check<dcomplex, double>( M, X, incx );
      resid = (double)(*indx - ind);
      break;
    }
    default :
      bli_check_error_code( BLIS_INVALID_DATATYPE );
  }

  return resid;
}
