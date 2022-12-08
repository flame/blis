#include "blis_test.h"
#include "blis_utils.h"
#include "test_syrk.h"

using namespace std;

//*  ==========================================================================
//*> C := alpha*A*A**T + beta*C,
//*>      or
//*> C := alpha*A**T*A + beta*C,
//*  ==========================================================================

template <typename T, typename U>
void libblis_isyrk_check(uplo_t uplo, trans_t trans, dim_t N, dim_t K,
  T* alpha, T* A, dim_t rsa, dim_t csa, T* beta, T* C, dim_t rsc, dim_t csc) {

  //* .. Local Scalars ..
  T tmp;
  dim_t i, j, l;
  bool UPPER, NOTRANS;
  T Alpha = alpha[0];
  T Beta  = beta[0];

  //* .. Parameters ..
  T ONE, ZERO;
  ONE  = 1.0 ;
  ZERO = 0.0 ;

  UPPER    = (uplo == BLIS_UPPER);
  NOTRANS  = (trans == BLIS_NO_TRANSPOSE) || (trans == BLIS_CONJ_NO_TRANSPOSE);

  //* Quick return if possible.
  if((N == 0) ||
    (((Alpha == ZERO) || (K == 0)) && (Beta == ONE))) {
      return;
  }

  //*     And when  alpha.eq.zero.
  if (Alpha == ZERO) {
    if (UPPER) {
      if (Beta == ZERO) {
        for(j = 0 ; j < N; j++) {
          for(i = 0 ; i <= j ; i++) {
            C[i*rsc + j*csc] = ZERO;
          }
        }
      }
      else {
        for(j = 0 ; j < N ; j++) {
          for(i = 0; i <= j ; i++) {
            C[i*rsc + j*csc] = Beta*C[i*rsc + j*csc];
          }
        }
      }
    }
    else {
      if (Beta == ZERO) {
        for(j = 0 ; j < N ; j++) {
          for(i = j ; i < N ; i++) {
            C[i*rsc + j*csc] = ZERO;
          }
        }
      }
      else {
        for(j = 0 ; j < N ; j++) {
          for(i = j ; i < N ; i++) {
            C[i*rsc + j*csc] = Beta*C[i*rsc + j*csc];
          }
        }
      }
    }
    return;
  }

  //* Start the operations.
  if(NOTRANS) {
    //*        Form  C := alpha*A*A**T + beta*C.
    if (UPPER) {
        if(Beta == ZERO) {
          for(j = 0 ; j < N ; j++) {
            for(i = 0 ; i <= j ; i++) {
              C[i*rsc + j*csc] = ZERO;
            }
          }
        }
        else if(Beta != ONE) {
          for(j = 0 ; j < N ; j++) {
            for(i = 0 ; i <= j ; i++) {
              C[i*rsc + j*csc] = Beta*C[i*rsc + j*csc];
            }
          }
        }

        for(j = 0 ; j < N ; j++) {
          for(l = 0; l < K ; l++) {
            if (A[j*rsa + l*csa] != ZERO) {
              tmp = Alpha*A[j*rsa + l*csa];
              for(i = 0 ; i <= j ; i++) {
                C[i*rsc + j*csc] = C[i*rsc + j*csc] + tmp*A[i*rsa + l*csa];
              }
            }
          }
        }
    }
    else {
      if(Beta == ZERO) {
        for(j = 0; j < N ; j++) {
          for(i = j ; i < N; i++) {
            C[i*rsc + j*csc] = ZERO;
          }
        }
      }
      else if (Beta != ONE) {
        for(j = 0; j < N ; j++) {
          for(i = j; i < N; i++) {
            C[i*rsc + j*csc] = Beta*C[i*rsc + j*csc];
          }
        }
      }

      for(j = 0; j < N ; j++) {
        for(l = 0; l < K ; l++) {
          if (A[j*rsa + l*csa] != ZERO) {
            tmp = Alpha*A[j*rsa + l*csa];
            for(i = j ; i < N; i++) {
              C[i*rsc + j*csc] = C[i*rsc + j*csc] + tmp*A[i*rsa + l*csa];
            }
          }
        }
      }
    }
  }
  else {
    //*        Form  C := alpha*A**T*A + beta*C.
    if (UPPER) {
      for(j = 0 ; j < N ; j++) {
        for(i = 0 ; i <= j ; i++) {
          tmp = ZERO;
          for(l = 0; l < K ; l++) {
            tmp = tmp + A[l*rsa + i*csa]*A[l*rsa + j*csa];
          }
          if (Beta == ZERO) {
            C[i*rsc + j*csc] = Alpha*tmp;
          }
          else {
            C[i*rsc + j*csc] = Alpha*tmp + Beta*C[i*rsc + j*csc];
          }
        }
      }
    }
    else {
      for(j = 0 ; j < N ; j++) {
        for(i = j ; i < N ; i++) {
          tmp = ZERO;
          for(l = 0 ; l < K ; l++) {
            tmp = tmp + A[l*rsa + i*csa]*A[l*rsa + j*csa];
          }
          if (Beta == ZERO) {
            C[i*rsc + j*csc] = Alpha*tmp;
          }
          else {
            C[i*rsc + j*csc] = Alpha*tmp + Beta*C[i*rsc + j*csc];
          }
        }
      }
    }
  }
  return;
}

template <typename T, typename U>
void libblis_icsyrk_check(uplo_t uplo, trans_t trans, dim_t N, dim_t K,
  T* alpha, T* A, dim_t rsa, dim_t csa, T* beta, T* C, dim_t rsc, dim_t csc) {

  //* .. Local Scalars ..
  T tmp;
  dim_t i, j, l;
  bool UPPER, NOTRANS;
  T Alpha = *alpha;
  T Beta  = *beta;

  //* .. Parameters ..
  T ONE, ZERO;
  ONE  = {1.0 , 0.0};
  ZERO = {0.0 , 0.0};

  UPPER    = (uplo == BLIS_UPPER);
  NOTRANS  = (trans == BLIS_NO_TRANSPOSE) || (trans == BLIS_CONJ_NO_TRANSPOSE);

  //* Quick return if possible.
  if((N == 0) ||
    (((Alpha.real == ZERO.real) || (K == 0)) && (Beta.real == ONE.real))) {
      return;
  }

  //*     And when  alpha.eq.zero.
  if((Alpha.real == ZERO.real) && (Alpha.imag == ZERO.imag)){
    if (UPPER) {
      if((Beta.real == ZERO.real)&&(Beta.imag == ZERO.imag)) {
        for(j = 0 ; j < N; j++) {
          for(i = 0 ; i <= j ; i++) {
            C[i*rsc + j*csc] = ZERO;
          }
        }
      }
      else {
        for(j = 0 ; j < N ; j++) {
          for(i = 0; i <= j ; i++) {
            C[i*rsc + j*csc] = mulc<T>(Beta , C[i*rsc + j*csc]);
          }
        }
      }
    }
    else {
      if((Beta.real == ZERO.real)&&(Beta.imag == ZERO.imag)) {
        for(j = 0 ; j < N ; j++) {
          for(i = j ; i < N ; i++) {
            C[i*rsc + j*csc] = ZERO;
          }
        }
      }
      else {
        for(j = 0 ; j < N ; j++) {
          for(i = j ; i < N ; i++) {
            C[i*rsc + j*csc] = mulc<T>(Beta , C[i*rsc + j*csc]);
          }
        }
      }
    }
    return;
  }

  //* Start the operations.
  if(NOTRANS) {
    //*        Form  C := alpha*A*A**T + beta*C.
    if (UPPER) {
        if((Beta.real == ZERO.real)||(Beta.imag == ZERO.imag)) {
          for(j = 0 ; j < N ; j++) {
            for(i = 0 ; i <= j ; i++) {
              C[i*rsc + j*csc] = ZERO;
            }
          }
        }
        else if((Beta.real != ONE.real)||(Beta.imag != ONE.imag)) {
          for(j = 0 ; j < N ; j++) {
            for(i = 0 ; i <= j ; i++) {
              C[i*rsc + j*csc] = mulc<T>(Beta , C[i*rsc + j*csc]);
            }
          }
        }

        for(j = 0 ; j < N ; j++) {
          for(l = 0; l < K ; l++) {
            if((A[j*rsa + l*csa].real != ZERO.real) || (A[j*rsa + l*csa].imag != ZERO.imag)) {
              tmp = mulc<T>(Alpha , A[j*rsa + l*csa]);
              for(i = 0 ; i <= j ; i++) {
                C[i*rsc + j*csc] = addc<T>(C[i*rsc + j*csc] , mulc<T>(tmp , A[i*rsa + l*csa]));
              }
            }
          }
        }
    }
    else {
      if((Beta.real == ZERO.real)||(Beta.imag == ZERO.imag)) {
        for(j = 0; j < N ; j++) {
          for(i = j ; i < N; i++) {
            C[i*rsc + j*csc] = ZERO;
          }
        }
      }
      else if((Beta.real != ONE.real) || (Beta.imag != ONE.imag)){
        for(j = 0; j < N ; j++) {
          for(i = j; i < N; i++) {
            C[i*rsc + j*csc] = mulc<T>(Beta , C[i*rsc + j*csc]);
          }
        }
      }

      for(j = 0; j < N ; j++) {
        for(l = 0; l < K ; l++) {
          if((A[j*rsa + l*csa].real != ZERO.real)||(A[j*rsa + l*csa].imag != ZERO.imag)) {
            tmp = mulc<T>(Alpha , A[j*rsa + l*csa]);
            for(i = j ; i < N; i++) {
              C[i*rsc + j*csc] = addc<T>(C[i*rsc + j*csc] , mulc<T>(tmp , A[i*rsa + l*csa]));
            }
          }
        }
      }
    }
  }
  else {
    //*        Form  C := alpha*A**T*A + beta*C.
    if (UPPER) {
      for(j = 0 ; j < N ; j++) {
        for(i = 0 ; i <= j ; i++) {
          tmp = ZERO;
          for(l = 0; l < K ; l++) {
            tmp = addc<T>(tmp , mulc<T>(A[l*rsa + i*csa] , A[l*rsa + j*csa]));
          }
          if((Beta.real == ZERO.real) ||(Beta.imag == ZERO.imag)){
            C[i*rsc + j*csc] = mulc<T>(Alpha , tmp);
          }
          else {
            C[i*rsc + j*csc] = addc<T>(mulc<T>(Alpha , tmp) , mulc<T>(Beta , C[i*rsc + j*csc]));
          }
        }
      }
    }
    else {
      for(j = 0 ; j < N ; j++) {
        for(i = j ; i < N ; i++) {
          tmp = ZERO;
          for(l = 0 ; l < K ; l++) {
            tmp = addc<T>(tmp , mulc<T>(A[l*rsa + i*csa] , A[l*rsa + j*csa]));
          }
          if((Beta.real == ZERO.real) || (Beta.imag == ZERO.imag)) {
            C[i*rsc + j*csc] = mulc<T>(Alpha , tmp);
          }
          else {
            C[i*rsc + j*csc] = addc<T>(mulc<T>(Alpha , tmp) , mulc<T>(Beta , C[i*rsc + j*csc]));
          }
        }
      }
    }
  }
  return;
}

double libblis_test_isyrk_check(
  test_params_t* params,
  obj_t*         alpha,
  obj_t*         a,
  obj_t*         beta,
  obj_t*         c,
  obj_t*         c_orig
){
  num_t dt       = bli_obj_dt( a );
  dim_t M        = bli_obj_length( c );
  dim_t K        = bli_obj_width_after_trans( a );
  uplo_t uploc   = bli_obj_uplo( c );
  trans_t transa = bli_obj_onlytrans_status( a );
  double resid  = 0.0;
  dim_t  rsa, csa;
  dim_t  rsc, csc;

  rsa = bli_obj_row_stride( a ) ;
  csa = bli_obj_col_stride( a ) ;
  rsc = bli_obj_row_stride( c ) ;
  csc = bli_obj_col_stride( c ) ;

  f77_int  lda;
  if( bli_obj_is_col_stored( c ) ) {
    lda    = bli_obj_col_stride( a );
  } else {
    lda    = bli_obj_row_stride( a );
  }
  int nrowa;
  if (transa == BLIS_NO_TRANSPOSE) {
    nrowa = M;
  } else {
    nrowa = K;
  }

  if( lda < max(1, nrowa) ) {
    return resid;
  }

  switch( dt )  {
    case BLIS_FLOAT :
    {
      float*   Alpha    = (float*) bli_obj_buffer( alpha );
      float*   A        = (float*) bli_obj_buffer( a );
      float*   Beta     = (float*) bli_obj_buffer( beta );
      float*   C        = (float*) bli_obj_buffer( c_orig );
      float*   CC       = (float*) bli_obj_buffer( c );
      libblis_isyrk_check<float, int32_t>(uploc, transa, M, K, Alpha,
                                       A, rsa, csa, Beta, C, rsc, csc);
      resid = computediffrm(M, M, CC, C, rsc, csc);
      break;
    }
    case BLIS_DOUBLE :
    {
      double*   Alpha   = (double*) bli_obj_buffer( alpha );
      double*   A       = (double*) bli_obj_buffer( a );
      double*   Beta    = (double*) bli_obj_buffer( beta );
      double*   C       = (double*) bli_obj_buffer( c_orig );
      double*   CC      = (double*) bli_obj_buffer( c );
      libblis_isyrk_check<double, int64_t>(uploc, transa, M, K, Alpha,
                                       A, rsa, csa, Beta, C, rsc, csc);
      resid = computediffrm(M, M, CC, C, rsc, csc);
      break;
    }
    case BLIS_SCOMPLEX :
    {
      scomplex*   Alpha = (scomplex*) bli_obj_buffer( alpha );
      scomplex*   A     = (scomplex*) bli_obj_buffer( a );
      scomplex*   Beta  = (scomplex*) bli_obj_buffer( beta );
      scomplex*   C     = (scomplex*) bli_obj_buffer( c_orig );
      scomplex*   CC    = (scomplex*) bli_obj_buffer( c );
      libblis_icsyrk_check<scomplex, int32_t>(uploc, transa, M, K, Alpha,
                                          A, rsa, csa, Beta, C, rsc, csc);
      resid = computediffim(M, M, CC, C, rsc, csc);
      break;
    }
    case BLIS_DCOMPLEX :
    {
      dcomplex*   Alpha = (dcomplex*) bli_obj_buffer( alpha );
      dcomplex*   A     = (dcomplex*) bli_obj_buffer( a );
      dcomplex*   Beta  = (dcomplex*) bli_obj_buffer( beta );
      dcomplex*   C     = (dcomplex*) bli_obj_buffer( c_orig );
      dcomplex*   CC    = (dcomplex*) bli_obj_buffer( c );
      libblis_icsyrk_check<dcomplex, int64_t>(uploc, transa, M, K, Alpha,
                                          A, rsa, csa, Beta, C, rsc, csc);
      resid = computediffim(M, M, CC, C, rsc, csc);
      break;
    }
    default :
      bli_check_error_code( BLIS_INVALID_DATATYPE );
  }

  return resid;
}

template <typename T>
double libblis_check_nan_real( dim_t rsc, dim_t csc, obj_t* c ) {
  dim_t  M = bli_obj_length( c );
  dim_t  N = bli_obj_width( c );
  dim_t  i,j;
  double resid = 0.0;
  T* C = (T*) bli_obj_buffer( c );

  for( i = 0 ; i < M ; i++ ) {
    for( j = 0 ; j < N ; j++ ) {
      auto tv = C[ i*rsc + j*csc ];
      if ( bli_isnan( tv )) {
        resid = tv ;
        break;
      }
    }
  }
  return resid;
}

template <typename U, typename T>
double libblis_check_nan_complex( dim_t rsc, dim_t csc, obj_t* c ) {
  dim_t  M = bli_obj_length( c );
  dim_t  N = bli_obj_width( c );
  dim_t  i,j;
  double resid = 0.0;
  U* C = (U*) bli_obj_buffer( c );

  for( i = 0 ; i < M ; i++ ) {
    for( j = 0 ; j < N ; j++ ) {
      auto tv = C[ i*rsc + j*csc ];
      if ( bli_isnan( tv.real ) || bli_isnan( tv.imag )) {
        resid = bli_isnan( tv.real ) ? tv.real : tv.imag;
        break;
      }
    }
  }
  return resid;
}

double libblis_check_nan_syrk(obj_t* c, num_t dt ) {
  dim_t  rsc, csc;
  double resid = 0.0;

  if( bli_obj_is_col_stored( c ) ) {
    rsc = 1;
    csc = bli_obj_col_stride( c );
  } else {
    rsc = bli_obj_row_stride( c );
    csc = 1 ;
  }

  switch( dt )  {
    case BLIS_FLOAT:
    {
      resid = libblis_check_nan_real<float>( rsc, csc, c );
      break;
    }
    case BLIS_DOUBLE:
    {
      resid = libblis_check_nan_real<double>( rsc, csc, c );
      break;
    }
    case BLIS_SCOMPLEX:
    {
      resid = libblis_check_nan_complex<scomplex, float>( rsc, csc, c );
      break;
    }
    case BLIS_DCOMPLEX:
    {
      resid = libblis_check_nan_complex<dcomplex, double>( rsc, csc, c );
      break;
    }
    default :
      bli_check_error_code( BLIS_INVALID_DATATYPE );
  }

  return resid;
}

