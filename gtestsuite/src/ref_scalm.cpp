#include "blis_test.h"
#include "blis_utils.h"
#include "test_scalm.h"

using namespace std;

//*  ==========================================================================
//*> SCALM performs matrix operations
//*>     A := conjalpha(alpha) * A
//*>    where A is an m x n matrix, and alpha is a scalar.
//*  ==========================================================================

template <typename T, typename U>
void libblis_iscalm_check(dim_t M, dim_t N, T* alpha,
                             T* X, dim_t rsx, dim_t csx ) {

  dim_t i, j;
  T ONE = 1.0 ;
  T ZERO = 0.0 ;
  T Alpha = alpha[0];

  if ((M == 0) || (N == 0)) {
    return;
  }

  if (Alpha != ONE) {
    if (Alpha == ZERO) {
      for(i = 0 ; i < M ; i++) {
        for(j = 0 ; j < N ; j++) {
          X[i*rsx + j*csx] = ZERO;
        }
      }
    }
    else {
      for(i = 0 ; i < M ; i++) {
        for(j = 0 ; j < N ; j++) {
          X[i*rsx + j*csx] = Alpha * X[i*rsx + j*csx];
        }
      }
    }
  }

  return;
}

template <typename T, typename U>
void libblis_icscalm_check(dim_t M, dim_t N, T* alpha,
                             T* X, dim_t rsx, dim_t csx, bool cfalpha) {
  dim_t i, j;
  T ONE  = {1.0, 0.0} ;
  T ZERO = {0.0, 0.0} ;
  T Alpha = *alpha;

  if ((M == 0) || (N == 0)) {
    return;
  }

  if(cfalpha)
    Alpha = conjugate<T>(Alpha);

  /* First form  x := Alpha*x. */
  if ((Alpha.real != ONE.real) && (Alpha.imag != ONE.imag)) {
    if ((Alpha.real != ZERO.real) && (Alpha.imag != ZERO.imag)) {
      for(i = 0 ; i < M ; i++) {
        for(j = 0 ; j < N ; j++) {
          X[i*rsx + j*csx] = ZERO;
        }
      }
    }
    else {
      for(i = 0 ; i < M ; i++) {
        for(j = 0 ; j < N ; j++) {
          X[i*rsx + j*csx] = mulc<T>(Alpha , X[i*rsx + j*csx]);
        }
      }
    }
  }

  return;
}

double libblis_test_iscalm_check(
  test_params_t* params,
  obj_t*  alpha,
  obj_t*  x,
  obj_t*  x_orig
) {
  num_t  dt    = bli_obj_dt( x );
  dim_t  M     = bli_obj_length( x );
  dim_t  N     = bli_obj_width( x );
  dim_t  rsx   = bli_obj_row_stride( x ) ;
  dim_t  csx   = bli_obj_col_stride( x ) ;
  bool cfalpha = bli_obj_has_conj( alpha );
  double resid = 0.0;

  switch( dt )  {
    case BLIS_FLOAT :
    {
      float*   Alpha    = (float*) bli_obj_buffer( alpha );
      float*   X        = (float*) bli_obj_buffer( x_orig );
      float*   XX       = (float*) bli_obj_buffer( x );
      libblis_iscalm_check<float, int32_t>(M, N, Alpha, X, rsx, csx);
      resid = computediffrm(M, N, XX, X, rsx, csx);
      break;
    }
    case BLIS_DOUBLE :
    {
      double*   Alpha   = (double*) bli_obj_buffer( alpha );
      double*   X       = (double*) bli_obj_buffer( x_orig );
      double*   XX      = (double*) bli_obj_buffer( x );
      libblis_iscalm_check<double, int64_t>(M, N, Alpha, X, rsx, csx);
      resid = computediffrm(M, N, XX, X, rsx, csx);
      break;
    }
    case BLIS_SCOMPLEX :
    {
      scomplex*   Alpha = (scomplex*) bli_obj_buffer( alpha );
      scomplex*   X     = (scomplex*) bli_obj_buffer( x_orig );
      scomplex*   XX    = (scomplex*) bli_obj_buffer( x );
      libblis_icscalm_check<scomplex, int32_t>(M, N, Alpha, X, rsx, csx, cfalpha);
      resid = computediffim(M, N, XX, X, rsx, csx);
      break;
    }
    case BLIS_DCOMPLEX :
    {
      dcomplex*   Alpha = (dcomplex*) bli_obj_buffer( alpha );
      dcomplex*   X     = (dcomplex*) bli_obj_buffer( x_orig );
      dcomplex*   XX    = (dcomplex*) bli_obj_buffer( x );
      libblis_icscalm_check<dcomplex, int64_t>(M, N, Alpha, X, rsx, csx, cfalpha);
      resid = computediffim(M, N, XX, X, rsx, csx);
      break;
    }
    default :
      bli_check_error_code( BLIS_INVALID_DATATYPE );
  }

  return resid;
}

