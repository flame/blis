#include "blis_test.h"
#include "blis_utils.h"
#include "test_axpym.h"

using namespace std;

//*  ==========================================================================
//*> AXPYM performs matrix operations
//*>    B := B + alpha * transa(A)
//*>    where B is an m x n matrix.
//*  ==========================================================================

template <typename T, typename U>
void libblis_iaxpym_check(dim_t M, dim_t N, T* alpha,
     T* X, dim_t rsx, dim_t csx, T* Y, dim_t rsy, dim_t csy, T* YY) {

  dim_t i, j;
  T Alpha = alpha[0];

  if ((M == 0) || (N == 0)) {
      return;
  }

  for(i = 0 ; i < M ; i++) {
    for(j = 0 ; j < N ; j++) {
      Y[i*rsy + j*csy] = Y[i*rsy + j*csy] + ( Alpha * X[i*rsx + j*csx] );
    }
  }

  return;
}

template <typename T, typename U>
void libblis_icaxpym_check(dim_t M, dim_t N, T* alpha,
     T* X, dim_t rsx, dim_t csx, conj_t conjx, T* Y, dim_t rsy, dim_t csy) {

  dim_t i, j;
  T Alpha = *alpha;

  if ((M == 0) || (N == 0)) {
      return;
  }

  if(conjx) {
    for(i = 0 ; i < M ; i++) {
      for(j = 0 ; j < N ; j++) {
        X[i*rsx + j*csx] = conjugate<T>(X[i*rsx + j*csx]);
      }
    }
  }

  for(i = 0 ; i < M ; i++) {
    for(j = 0 ; j < N ; j++) {
      Y[i*rsy + j*csy] = addc<T>(Y[i*rsy + j*csy] , mulc<T>(Alpha , X[i*rsx + j*csx]));
    }
  }

  return;
}

double libblis_test_iaxpym_check(
  test_params_t* params,
  obj_t*         alpha,
  obj_t*         x,
  obj_t*         y,
  obj_t*         y_orig
){
  num_t  dt    = bli_obj_dt( x );
  bool  transx = bli_obj_has_trans( x );
  conj_t conjx = bli_obj_conj_status( x );
  dim_t  M     = bli_obj_length( y );
  dim_t  N     = bli_obj_width( y );
  dim_t  rsy   = bli_obj_row_stride( y ) ;
  dim_t  csy   = bli_obj_col_stride( y ) ;
  double resid = 0.0;
  dim_t  rsx, csx;

  if( bli_obj_is_col_stored( x ) ) {
    rsx = transx ? bli_obj_col_stride( x ) : bli_obj_row_stride( x ) ;
    csx = transx ? bli_obj_row_stride( x ) : bli_obj_col_stride( x ) ;
  } else {
    rsx = transx ? bli_obj_col_stride( x ) : bli_obj_row_stride( x ) ;
    csx = transx ? bli_obj_row_stride( x ) : bli_obj_col_stride( x ) ;
  }

  switch( dt )  {
    case BLIS_FLOAT :
    {
      float*   Alpha    = (float*) bli_obj_buffer( alpha );
      float*   X        = (float*) bli_obj_buffer( x );
      float*   Y        = (float*) bli_obj_buffer( y_orig );
      float*   YY       = (float*) bli_obj_buffer( y );
      libblis_iaxpym_check<float, int32_t>( M, N, Alpha, X, rsx, csx,
                                                 Y, rsy, csy, YY );
      resid = computediffrm(M, N, YY, Y, rsy, csy);
      break;
    }
    case BLIS_DOUBLE :
    {
      double*   Alpha   = (double*) bli_obj_buffer( alpha );
      double*   X       = (double*) bli_obj_buffer( x );
      double*   Y       = (double*) bli_obj_buffer( y_orig );
      double*   YY      = (double*) bli_obj_buffer( y );
      libblis_iaxpym_check<double, int64_t>( M, N, Alpha, X, rsx, csx,
                                                  Y, rsy, csy, YY );
      resid = computediffrm(M, N, YY, Y, rsy, csy);
      break;
    }
    case BLIS_SCOMPLEX :
    {
      scomplex*   Alpha = (scomplex*) bli_obj_buffer( alpha );
      scomplex*   X     = (scomplex*) bli_obj_buffer( x );
      scomplex*   Y     = (scomplex*) bli_obj_buffer( y_orig );
      scomplex*   YY    = (scomplex*) bli_obj_buffer( y );
      libblis_icaxpym_check<scomplex, int32_t>( M, N, Alpha, X, rsx, csx,
                                              conjx, Y, rsy, csy );
      resid = computediffim(M, N, YY, Y, rsy, csy);
      break;
    }
    case BLIS_DCOMPLEX :
    {
      dcomplex*   Alpha = (dcomplex*) bli_obj_buffer( alpha );
      dcomplex*   X     = (dcomplex*) bli_obj_buffer( x );
      dcomplex*   Y     = (dcomplex*) bli_obj_buffer( y_orig );
      dcomplex*   YY    = (dcomplex*) bli_obj_buffer( y );
      libblis_icaxpym_check<dcomplex, int64_t>( M, N, Alpha, X, rsx, csx,
                                              conjx, Y, rsy, csy );
      resid = computediffim(M, N, YY, Y, rsy, csy);
      break;
    }
    default :
      bli_check_error_code( BLIS_INVALID_DATATYPE );
  }

  return resid;
}

