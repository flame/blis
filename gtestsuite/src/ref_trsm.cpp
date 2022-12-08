#include "blis_test.h"
#include "blis_utils.h"
#include "test_trsm.h"

using namespace std;

//*  ==========================================================================
//*> TRSM  solves one of the matrix equations
//*>    op( A )*X = alpha*B,   or   X*op( A ) = alpha*B,
//*> where alpha is a scalar, X and B are m by n matrices, A is a unit, or
//*> non-unit,  upper or lower triangular matrix  and  op( A )  is one  of
//*>    op( A ) = A   or   op( A ) = A**T.
//*> The matrix X is overwritten on B.
//*  ==========================================================================

template <typename T, typename U>
void libblis_ictrsm_check(side_t side, uplo_t uplo, trans_t transa,
  diag_t diaga, dim_t M, dim_t N, T Alpha, T* A, dim_t rsa, dim_t csa,
  bool conja, T* B, dim_t rsb, dim_t csb){

  T tmp;
  dim_t i, j, k;
  bool LSIDE, NOUNIT, UPPER, NOTRANS;
  T ONE, ZERO;
  ONE  = {1.0 , 0.0};
  ZERO = {0.0 , 0.0};

  //*     Test the input parameters.
  LSIDE   = (side == BLIS_LEFT);
  NOTRANS = (transa == BLIS_NO_TRANSPOSE) || (transa == BLIS_CONJ_NO_TRANSPOSE);
  NOUNIT  = (diaga == BLIS_NONUNIT_DIAG);
  UPPER   = (uplo == BLIS_UPPER);

  //* Quick return if possible.
  if ((M == 0) || (N == 0) )
    return;

  //* And when  alpha.eq.zero.
  if ((Alpha.real == ZERO.real) && (Alpha.imag == ZERO.imag)) {
    for(i = 0; i < M ; i++) {
      for(j = 0; j < N ; j++) {
        B[i*rsb+ j*csb] = ZERO;
      }
    }
    return;
  }

  if(conja) {
    dim_t dim;
    if (LSIDE)         dim = M;
    else               dim = N;
    for( i = 0 ; i < dim ; i++ ) {
      for( j = 0 ; j < dim ; j++ ) {
        A[i*rsa + j*csa] = conjugate<T>(A[i*rsa + j*csa]);
      }
    }
  }

  if((Alpha.real != ONE.real)&&(Alpha.imag != ONE.imag)){
    for(i = 0; i < M; i++) {
      for(j = 0 ; j < N ; j++) {
        B[i*rsb + j*csb] = mulc<T>(Alpha , B[i*rsb + j*csb]);
      }
    }
  }

  //* Start the operations.
  if (LSIDE) {
    if (NOTRANS) {
      //* Form  B := alpha*inv( A )*B.
      if (UPPER) { /* AuXB : LUN */
        for(j = 0 ; j < N ; j++) {
          for(k = M; k-- ; ) {
            if((B[k*rsb + j*csb].real != ZERO.real) || (B[k*rsb + j*csb].imag != ZERO.imag)) {
              if (NOUNIT) B[k*rsb + j*csb] = divc<T,U>(B[k*rsb + j*csb] , A[k*rsa + k*csa]);
              for(i = 0 ; i < k ; i++) {
                B[i*rsb + j*csb] = subc<T>(B[i*rsb + j*csb] , mulc<T>(B[k*rsb + j*csb] , A[i*rsa + k*csa]));
              }
            }
          }
        }
      }
      else {
        for(j = 0 ; j < N ; j++) {  /* AlXB : LLN */
          for(k = 0 ; k < M ; k++) {
            if ((B[k*rsb + j*csb].real != ZERO.real) || (B[k*rsb + j*csb].imag != ZERO.imag)) {
              if (NOUNIT) B[k*rsb + j*csb] = divc<T,U>(B[k*rsb + j*csb] , A[k*rsa + k*csa]);
              for(i=(k+1) ; i < M ; i++) {
                B[i*rsb + j*csb] = subc<T>(B[i*rsb + j*csb] , mulc<T>(B[k*rsb + j*csb] , A[i*rsa + k*csa]));
              }
            }
          }
        }
      }
    }
    else {
      //* Form  B := alpha*inv( A**T )*B.
      if (UPPER) {
        for(j = 0 ; j < N ; j++) {  /* AutXB : LUT */
          for(i = 0 ; i < M ; i++) {
            tmp = B[i*rsb + j*csb];
            for(k = 0 ; k < i ; k++) {
              tmp = subc<T>(tmp , mulc<T>(A[k*rsa + i*csa] , B[k*rsb + j*csb]));
            }
            if (NOUNIT) tmp = divc<T,U>(tmp , A[i*rsa + i*csa]);
            B[i*rsb + j*csb] = tmp;
          }
        }
      }
      else {
        for(j = 0 ; j < N ; j++) {  /* AltXB : LLT */
          for(i = M ; i-- ;) {
            tmp = B[i*rsb + j*csb];
            for(k = (i+1) ; k < M ; k++) {
              tmp = subc<T>(tmp , mulc<T>(A[k*rsa + i*csa] , B[k*rsb + j*csb]));
            }
            if (NOUNIT) tmp = divc<T,U>(tmp , A[i*rsa + i*csa]);
            B[i*rsb + j*csb] = tmp;
          }
        }
      }
    }
  }
  else {
    if(NOTRANS) {
      //* Form  B := alpha*B*inv( A ).
      if (UPPER) {
        for(j = 0 ; j < N ; j++) {  /* XAuB : RUN */
          for(k = 0 ; k < j ; k++) {
            if ((A[k*rsa + j*csa].real != ZERO.real)||(A[k*rsa + j*csa].imag != ZERO.imag)) {
              for(i = 0 ; i < M ; i++) {
                B[i*rsb + j*csb] = subc<T>(B[i*rsb + j*csb] , mulc<T>(A[k*rsa + j*csa] , B[i*rsb + k*csb]));
              }
            }
          }
          if (NOUNIT) {
            tmp = divc<T,U>(ONE , A[j*rsa + j*csa]);
            for(i = 0 ; i < M ; i++) {
              B[i*rsb + j*csb] = mulc<T>(tmp , B[i*rsb + j*csb]);
            }
          }
        }
      }
      else {
        for(j = N; j-- ; ) {  /* XAlB : RLN */
          for(k = (j+1) ; k < N ; k++) {
            if((A[k*rsa + j*csa].real != ZERO.real)||(A[k*rsa + j*csa].imag != ZERO.imag)) {
              for(i = 0 ; i < M ; i++) {
                B[i*rsb + j*csb] = subc<T>(B[i*rsb + j*csb] , mulc<T>(A[k*rsa + j*csa] , B[i*rsb + k*csb]));
              }
            }
          }
          if (NOUNIT) {
            tmp = divc<T,U>(ONE , A[j*rsa + j*csa]);
            for(i = 0 ; i < M ; i++) {
              B[i*rsb + j*csb] = mulc<T>(tmp , B[i*rsb + j*csb]);
            }
          }
        }
      }
    }
    else {
      //* Form  B := alpha*B*inv( A**T ).
      if (UPPER) {  /* XAutB : RUT */
        for(k = N ; k-- ; ) {
          if (NOUNIT) {
            tmp = divc<T,U>(ONE , A[k*rsa + k*csa]);
            for(i = 0 ; i < M ; i++) {
              B[i*rsb + k*csb] = mulc<T>(tmp , B[i*rsb + k*csb]);
            }
          }
          for(j = 0 ; j < k; j++) {
            if((A[j*rsa + k*csa].real != ZERO.real)||(A[j*rsa + k*csa].imag != ZERO.imag)) {
              tmp = A[j*rsa + k*csa];
              for(i = 0 ; i < M ; i++) {
                B[i*rsb + j*csb] = subc<T>(B[i*rsb + j*csb] , mulc<T>(tmp , B[i*rsb + k*csb]));
              }
            }
          }
        }
      }
      else {  /* XAltB : RLT */
        for(k = 0 ; k < N; k++) {
          if (NOUNIT) {
            tmp = divc<T,U>(ONE , A[k*rsa + k*csa]);
            for(i = 0 ; i < M ; i++) {
              B[i*rsb + k*csb] = mulc<T>(tmp , B[i*rsb + k*csb]);
            }
          }
          for(j = (k+1) ; j < N ; j++) {
            if((A[j*rsa + k*csa].real != ZERO.real)||(A[j*rsa + k*csa].imag != ZERO.imag)) {
              tmp = A[j*rsa + k*csa];
              for(i = 0 ; i < M; i++) {
                B[i*rsb + j*csb] = subc<T>(B[i*rsb + j*csb] , mulc<T>(tmp , B[i*rsb + k*csb]));
              }
            }
          }
        }
      }
    }
  }
  return;
}

template <typename T, typename U>
void libblis_itrsm_check(side_t side, uplo_t uploa, trans_t transa,
  diag_t diaga, dim_t M, dim_t N, T Alpha, T* A, dim_t rsa, dim_t csa,
  T* B, dim_t rsb, dim_t csb) {

  T tmp;
  dim_t i, j, k;
  bool LSIDE, UPPER;
  bool NOTRANS, NOUNIT;
  T ONE = 1.0;
  T ZERO = 0.0;

  LSIDE   = (side == BLIS_LEFT);
  NOTRANS = (transa == BLIS_NO_TRANSPOSE) || (transa == BLIS_CONJ_NO_TRANSPOSE);
  NOUNIT  = (diaga == BLIS_NONUNIT_DIAG);
  UPPER   = (uploa == BLIS_UPPER);

  if((M == 0) || (N == 0))
    return;

  if (Alpha == ZERO) {
    for(i = 0 ; i < M ; i++) {
      for(j = 0 ; j < N; j++) {
        B[i*rsb + j*csb] = ZERO;
      }
    }
    return;
  }

  if (Alpha != ONE) {
    for(i = 0; i < M; i++) {
      for(j = 0 ; j < N ; j++) {
        B[i*rsb + j*csb] = Alpha*B[i*rsb + j*csb];
      }
    }
  }

  //* Start the operations.
  if (LSIDE) {
    if (NOTRANS) {
      //* Form  B := alpha*inv( A )*B.
      if (UPPER) { /* AuXB : LUN */
        for(j = 0 ; j < N ; j++) {
          for(k = M; k-- ; ) {
            if (B[k*rsb + j*csb] != ZERO) {
              if (NOUNIT) B[k*rsb + j*csb] = B[k*rsb + j*csb]/A[k*rsa + k*csa];
              for(i = 0 ; i < k ; i++) {
                B[i*rsb + j*csb] = B[i*rsb + j*csb] - B[k*rsb + j*csb]*A[i*rsa + k*csa];
              }
            }
          }
        }
      }
      else {
        for(j = 0 ; j < N ; j++) {  /* AlXB : LLN */
          for(k = 0 ; k < M ; k++) {
            if (B[k*rsb + j*csb] != ZERO) {
              if (NOUNIT) B[k*rsb + j*csb] = B[k*rsb + j*csb]/A[k*rsa + k*csa];
              for(i=(k+1) ; i < M ; i++) {
                B[i*rsb + j*csb] = B[i*rsb + j*csb] - (B[k*rsb + j*csb]*A[i*rsa + k*csa]);
              }
            }
          }
        }
      }
    }
    else {
      //* Form  B := alpha*inv( A**T )*B.
      if (UPPER) {
        for(j = 0 ; j < N ; j++) {  /* AutXB : LUT */
          for(i = 0 ; i < M ; i++) {
            tmp = B[i*rsb + j*csb];
            for(k = 0 ; k < i ; k++) {
              tmp = tmp - A[k*rsa + i*csa]*B[k*rsb + j*csb];
            }
            if (NOUNIT) tmp = tmp/A[i*rsa + i*csa];
            B[i*rsb + j*csb] = tmp;
          }
        }
      }
      else {
        for(j = 0 ; j < N ; j++) {  /* AltXB : LLT */
          for(i = M ; i-- ;) {
            tmp = B[i*rsb + j*csb];
            for(k = (i+1) ; k < M ; k++) {
              tmp = tmp - A[k*rsa + i*csa]*B[k*rsb + j*csb];
            }
            if (NOUNIT) tmp = tmp/A[i*rsa + i*csa];
            B[i*rsb + j*csb] = tmp;
          }
        }
      }
    }
  }
  else {
    if(NOTRANS) {
      //* Form  B := alpha*B*inv( A ).
      if (UPPER) {
        for(j = 0 ; j < N ; j++) {  /* XAuB : RUN */
          for(k = 0 ; k < j ; k++) {
            if (A[k*rsa + j*csa] != ZERO) {
              for(i = 0 ; i < M ; i++) {
                B[i*rsb + j*csb] = B[i*rsb + j*csb] - A[k*rsa + j*csa]*B[i*rsb + k*csb];
              }
            }
          }
          if (NOUNIT) {
            tmp = ONE/A[j*rsa + j*csa];
            for(i = 0 ; i < M ; i++) {
              B[i*rsb + j*csb] = tmp*B[i*rsb + j*csb];
            }
          }
        }
      }
      else {
        for(j = N; j-- ; ) {  /* XAlB : RLN */
          for(k = (j+1) ; k < N ; k++) {
            if (A[k*rsa + j*csa] != ZERO) {
              for(i = 0 ; i < M ; i++) {
                B[i*rsb + j*csb] = B[i*rsb + j*csb] - A[k*rsa + j*csa]*B[i*rsb + k*csb];
              }
            }
          }
          if (NOUNIT) {
            tmp = ONE/A[j*rsa + j*csa];
            for(i = 0 ; i < M ; i++) {
              B[i*rsb + j*csb] = tmp*B[i*rsb + j*csb];
            }
          }
        }
      }
    }
    else {
      //* Form  B := alpha*B*inv( A**T ).
      if (UPPER) {  /* XAutB : RUT */
        for(k = N ; k-- ; ) {
          if (NOUNIT) {
            tmp = ONE/A[k*rsa + k*csa];
            for(i = 0 ; i < M ; i++) {
              B[i*rsb + k*csb] = tmp*B[i*rsb + k*csb];
            }
          }
          for(j = 0 ; j < k; j++) {
            if (A[j*rsa + k*csa] != ZERO) {
              tmp = A[j*rsa + k*csa];
              for(i = 0 ; i < M ; i++) {
                B[i*rsb + j*csb] = B[i*rsb + j*csb] - tmp*B[i*rsb + k*csb];
              }
            }
          }
        }
      }
      else {  /* XAltB : RLT */
        for(k = 0 ; k < N; k++) {
          if (NOUNIT) {
            tmp = ONE/A[k*rsa + k*csa];
            for(i = 0 ; i < M ; i++) {
              B[i*rsb + k*csb] = tmp*B[i*rsb + k*csb];
            }
          }
          for(j = (k+1) ; j < N ; j++) {
            if (A[j*rsa + k*csa] != ZERO) {
              tmp = A[j*rsa + k*csa];
              for(i = 0 ; i < M; i++) {
                B[i*rsb + j*csb] = B[i*rsb + j*csb] - tmp*B[i*rsb + k*csb];
              }
            }
          }
        }
      }
    }
  }
  return;
}

double libblis_test_itrsm_check(
  test_params_t* params,
  side_t         side,
  obj_t*         alpha,
  obj_t*         a,
  obj_t*         b,
  obj_t*         b_orig,
  num_t          dt
){
  dim_t M        = bli_obj_length( b_orig );
  dim_t N        = bli_obj_width( b_orig );
  uplo_t uploa   = bli_obj_uplo( a );
  trans_t transa = bli_obj_onlytrans_status( a );
  bool conja     = bli_obj_has_conj( a );
  diag_t diaga   = bli_obj_diag( a );
  dim_t rsa, csa;
  dim_t rsb, csb;
  double resid = 0.0;

  rsa = bli_obj_row_stride( a ) ;
  csa = bli_obj_col_stride( a ) ;
  rsb = bli_obj_row_stride( b ) ;
  csb = bli_obj_col_stride( b ) ;

  switch( dt )  {
    case BLIS_FLOAT :
    {
      float*   Alpha  = (float*) bli_obj_buffer( alpha );
      float*   A      = (float*) bli_obj_buffer( a );
      float*   B      = (float*) bli_obj_buffer( b_orig );
      float*   BB     = (float*) bli_obj_buffer( b );
      libblis_itrsm_check<float, int32_t>(side, uploa, transa,
                       diaga, M, N, *Alpha, A, rsa, csa, B, rsb, csb );
      resid = computediffrm(M, N, BB, B, rsb, csb);
      break;
    }
    case BLIS_DOUBLE :
    {
      double*   Alpha = (double*) bli_obj_buffer( alpha );
      double*   A     = (double*) bli_obj_buffer( a );
      double*   B     = (double*) bli_obj_buffer( b_orig );
      double*   BB    = (double*) bli_obj_buffer( b );
      libblis_itrsm_check<double, int64_t>(side, uploa, transa,
                       diaga, M, N, *Alpha, A, rsa, csa, B, rsb, csb );
      resid = computediffrm(M, N, BB, B, rsb, csb);
      break;
    }
    case BLIS_SCOMPLEX :
    {
      scomplex*   Alpha = (scomplex*) bli_obj_buffer( alpha );
      scomplex*   A     = (scomplex*) bli_obj_buffer( a );
      scomplex*   B     = (scomplex*) bli_obj_buffer( b_orig );
      scomplex*   BB    = (scomplex*) bli_obj_buffer( b );
      libblis_ictrsm_check<scomplex, float>(side, uploa, transa,
                    diaga, M, N, *Alpha, A, rsa, csa, conja, B, rsb, csb );
      resid = computediffim(M, N, BB, B, rsb, csb);
      break;
    }
    case BLIS_DCOMPLEX :
    {
      dcomplex*   Alpha = (dcomplex*) bli_obj_buffer( alpha );
      dcomplex*   A     = (dcomplex*) bli_obj_buffer( a );
      dcomplex*   B     = (dcomplex*) bli_obj_buffer( b_orig );
      dcomplex*   BB    = (dcomplex*) bli_obj_buffer( b );
      libblis_ictrsm_check<dcomplex, double>(side, uploa, transa,
                    diaga, M, N, *Alpha, A, rsa, csa, conja, B, rsb, csb );
      resid = computediffim(M, N, BB, B, rsb, csb);
      break;
    }
    default :
      bli_check_error_code( BLIS_INVALID_DATATYPE );
  }
  return resid;
}

template <typename T>
double libblis_check_nan_real( dim_t rs, dim_t cs, obj_t* b ) {
  dim_t  M = bli_obj_length( b );
  dim_t  N = bli_obj_width( b );
  dim_t  i,j;
  double resid = 0.0;
  T* B = (T*) bli_obj_buffer( b );

  for( i = 0 ; i < M ; i++ ) {
    for( j = 0 ; j < N ; j++ ) {
      auto tv = B[ i*rs + j*cs ];
      if ( bli_isnan( tv )) {
        resid = tv ;
        break;
      }
    }
  }
  return resid;
}

template <typename T>
double libblis_check_nan_complex( dim_t rs, dim_t cs, obj_t* b ) {
  dim_t  M = bli_obj_length( b );
  dim_t  N = bli_obj_width( b );
  dim_t  i,j;
  double resid = 0.0;
  T* B = (T*) bli_obj_buffer( b );

  for( i = 0 ; i < M ; i++ ) {
    for( j = 0 ; j < N ; j++ ) {
      auto tv = B[ i*rs + j*cs ];
      if ( bli_isnan( tv.real ) || bli_isnan( tv.imag )) {
        resid = bli_isnan( tv.real ) ? tv.real : tv.imag;
        break;
      }
    }
  }
  return resid;
}

double libblis_check_nan_trsm(obj_t* b, num_t dt ) {
  dim_t  rsc, csc;
  double resid = 0.0;

  if( bli_obj_row_stride( b ) == 1 ) {
    rsc = 1;
    csc = bli_obj_col_stride( b );
  } else {
    rsc = bli_obj_row_stride( b );
    csc = 1 ;
  }

  switch( dt )  {
    case BLIS_FLOAT:
    {
      resid = libblis_check_nan_real<float>( rsc, csc, b );
      break;
    }
    case BLIS_DOUBLE:
    {
      resid = libblis_check_nan_real<double>( rsc, csc, b );
      break;
    }
    case BLIS_SCOMPLEX:
    {
      resid = libblis_check_nan_complex<scomplex>( rsc, csc, b );
      break;
    }
    case BLIS_DCOMPLEX:
    {
      resid = libblis_check_nan_complex<dcomplex>( rsc, csc, b );
      break;
    }
    default :
      bli_check_error_code( BLIS_INVALID_DATATYPE );
  }

  return resid;
}

