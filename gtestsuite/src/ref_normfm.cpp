#include "blis_test.h"
#include "blis_utils.h"
#include "test_normfm.h"

using namespace std;

//*  ==========================================================================
//*> NORMFM performs matrix operation
//*>    Compute the Frobenius norm (bli_?normfm())
//*>    of the elements in an m x n matrix A. The resulting norm is stored to norm
//*  ==========================================================================

template <typename T, typename U>
T libblis_inormfm_check(dim_t M, dim_t N, T* X, dim_t rsx, dim_t csx ) {

  dim_t i, j;
  T sum = 0.0;
  T norm = 0.0;

  if ((M == 0) || (N == 0)) {
      return norm;
  }

  for(i = 0 ; i < M ; i++) {
    for(j = 0 ; j < N ; j++) {
      sum +=  X[i*rsx + j*csx] * X[i*rsx + j*csx];
    }
  }

  norm = sqrt( abs(sum) );

 	return norm;
}

template <typename T, typename U>
U libblis_icnormfm_check(dim_t M, dim_t N, T* X, dim_t rsx, dim_t csx ) {

  dim_t i, j;
  T rr = { 0.0, 0.0 };
  U norm = 0.0;

  if ((M == 0) || (N == 0)) {
      return norm;
  }

  for(i = 0 ; i < M ; i++) {
    for(j = 0 ; j < N ; j++) {
      auto a = X[i*rsx + j*csx];
      rr.real += a.real * a.real;
      rr.imag += a.imag * a.imag;
    }
  }

  U r = rr.real + rr.imag;
  norm = sqrt( abs(r) );

 	return norm;
}

double libblis_test_inormfm_check(
  test_params_t* params,
  obj_t*         x,
  obj_t*         norm
){
  num_t  dt    = bli_obj_dt( x );
  dim_t  M     = bli_obj_length( x );
  dim_t  N     = bli_obj_width( x );
  dim_t  rsx   = bli_obj_row_stride( x ) ;
  dim_t  csx   = bli_obj_col_stride( x ) ;
  double resid = 0.0;

  switch( dt )  {
    case BLIS_FLOAT :
    {
      float* X    = (float*) bli_obj_buffer( x );
      float* av   = (float*) bli_obj_internal_scalar_buffer( norm );
      float rv    = libblis_inormfm_check<float, int32_t>(M, N, X, rsx, csx);
      resid = (double)(abs(rv - *av)/abs(rv));
      break;
    }
    case BLIS_DOUBLE :
    {
      double* X   = (double*) bli_obj_buffer( x );
      double* av  = (double*) bli_obj_internal_scalar_buffer( norm );
      double rv   = libblis_inormfm_check<double, int64_t>(M, N, X, rsx, csx);
      resid = (double)(abs(rv - *av)/abs(rv));
      break;
    }
    case BLIS_SCOMPLEX :
    {
      scomplex* X  = (scomplex*) bli_obj_buffer( x );
      float* av    = (float*) bli_obj_internal_scalar_buffer( norm );
      float  rv = libblis_icnormfm_check<scomplex, float>(M, N, X, rsx, csx);
      resid = (double)(abs(rv - *av)/abs(rv));
      break;
    }
    case BLIS_DCOMPLEX :
    {
      dcomplex* X  = (dcomplex*) bli_obj_buffer( x );
      double* av   = (double*) bli_obj_internal_scalar_buffer( norm );
      double  rv = libblis_icnormfm_check<dcomplex, double>(M, N, X, rsx, csx);
      resid = (double)(abs(rv - *av)/abs(rv));
      break;
    }
    default :
      bli_check_error_code( BLIS_INVALID_DATATYPE );
  }
  return abs(resid);
}
