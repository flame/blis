#include "blis_test.h"
#include "blis_utils.h"
#include "test_herk.h"

using namespace std;

//*  ==========================================================================
//*> HERK  performs one of the hermitian rank k operations
//*>    C := alpha*A*A**H + beta*C,
//*> or
//*>    C := alpha*A**H*A + beta*C,
//*> where  alpha and beta  are  real scalars,  C is an  n by n  hermitian
//*> matrix and  A  is an  n by k  matrix in the  first case and a  k by n
//*> matrix in the second case.
//*  ==========================================================================

template <typename T>
void libblis_iherk_check( uplo_t uplo, trans_t trans, dim_t N, dim_t K,
  T Alpha, T* A, dim_t rsa, dim_t csa, T Beta, T* C, dim_t rsc, dim_t csc )
{
  T tmp, rtmp;
  dim_t i, j, l;
  bool UPPER, NOTRANS;

  T ONE  = 1.0;
  T ZERO = 0.0;

  UPPER    = (uplo == BLIS_UPPER);
  NOTRANS  = (trans == BLIS_NO_TRANSPOSE) || (trans == BLIS_CONJ_NO_TRANSPOSE);

  if( (N == 0) || (( Alpha == ZERO || K == 0) && Beta == ONE ) )
    return;

  //*     And when  alpha.eq.zero.
  if( Alpha == ZERO )
  {
    if( UPPER )
    {
      if( Beta == ZERO )
      {
        for( j = 0 ; j < N ; j++ )
        {
          for( i = 0 ; i <= j ; i++ )
          {
            C[i*rsc + j*csc] = ZERO;
          }
        }
      }
      else
      {
        for( j = 0 ; j < N ; j++ )
        {
          for( i = 0 ; i < j ; i++ )
          {
            C[i*rsc + j*csc] = Beta*C[i*rsc + j*csc];
          }
          C[j*rsc + j*csc] = Beta*(C[j*rsc + j*csc]);
        }
      }
    }
    else
    {
      if( Beta == ZERO )
      {
        for( j = 0 ; j < N ; j++ )
        {
          for( i = j ; i < N ; i++ )
          {
            C[i*rsc + j*csc] = ZERO;
          }
        }
      }
      else
      {
        for( j = 0 ; j < N ; j++ )
        {
          C[j*rsc + j*csc] = Beta*C[j*rsc + j*csc];
          for( i = (j+1) ; i < N ; i++ )
          {
            C[i*rsc + j*csc] = Beta*C[i*rsc + j*csc];
          }
        }
      }
    }
    return;
  }

  //*     Start the operations.
  if( NOTRANS )
  {
    //* Form  C := alpha*A*A**H + beta*C.
    if( UPPER )
    {
      for( j = 0; j < N ; j++ )
      {
        if( Beta == ZERO )
        {
          for( i = 0 ; i <= j ; i++ )
          {
            C[i*rsc + j*csc] = ZERO;
          }
        }
        else if( Beta != ONE )
        {
          for(i = 0 ; i < j ; i++ )
          {
            C[i*rsc + j*csc] = Beta*C[i*rsc + j*csc];
          }
          C[j*rsc + j*csc] = Beta*C[j*rsc + j*csc];
        }
        for( l = 0 ; l < K ; l++ )
        {
          if( A[j*rsa + l*csa] != ZERO )
          {
            tmp = Alpha*A[j*rsa + l*csa] ;
            for( i = 0 ; i < j ; i++ )
            {
              C[i*rsc + j*csc] = C[i*rsc + j*csc] + tmp*A[i*rsa + l*csa];
            }
            C[j*rsc + j*csc] = C[j*rsc + j*csc] + tmp*A[i*rsa + l*csa];
          }
        }
      }
    }
    else
    {
      for( j = 0; j < N ; j++ )
      {
        if( Beta == ZERO )
        {
          for( i = j ; i < N; i++ )
          {
            C[i*rsc + j*csc] = ZERO;
          }
        }
        else if( Beta != ONE )
        {
          C[j*rsc + j*csc] = Beta*C[j*rsc + j*csc];
          for(i = (j+1) ; i < N ; i++ )
          {
            C[i*rsc + j*csc] = Beta*C[i*rsc + j*csc];
          }
        }
        for( l = 0 ; l < K ; l++ )
        {
          if( A[j*rsa + l*csa] != ZERO )
          {
            tmp = Alpha*A[j*rsa + l*csa];
            C[j*rsc + j*csc] = C[j*rsc + j*csc] + tmp*A[j*rsa + l*csa];
            for( i = (j+1) ; i < N ; i++ )
            {
              C[i*rsc + j*csc] = C[i*rsc + j*csc] + tmp*A[i*rsa + l*csa];
            }
          }
        }
      }
    }
  }
  else
  {
    //*        Form  C := alpha*A**H*A + beta*C.
    if( UPPER )
    {
      for( j = 0; j < N ; j++ )
      {
        for( i = 0 ; i < j ; i++ )
        {
          tmp = ZERO;
          for( l = 0 ; l < K ; l++ )
          {
            tmp = tmp + A[l*rsa + i*csa]*A[l*rsa + j*csa];
          }
          if( Beta == ZERO )
          {
            C[i*rsc + j*csc] = Alpha*tmp;
          }
          else
          {
            C[i*rsc + j*csc] = Alpha*tmp + Beta*C[i*rsc + j*csc];
          }
        }
        rtmp = ZERO;
        for( l = 0 ; l < K ; l++ )
        {
          rtmp = rtmp + A[l*rsa + j*csa]*A[l*rsa + j*csa];
        }
        if( Beta == ZERO )
        {
          C[j*rsc + j*csc] = Alpha*rtmp;
        }
        else
        {
          C[j*rsc + j*csc] = Alpha*rtmp + Beta*C[j*rsc + j*csc];
        }
      }
    }
    else
    {
      for( j = 0; j < N ; j++ )
      {
        rtmp = ZERO;
        for( l = 0 ; l < K ; l++ )
        {
          rtmp = rtmp + A[l*rsa + j*csa]*A[l*rsa + j*csa];
        }
        if( Beta == ZERO )
        {
          C[j*rsc + j*csc] = Alpha*rtmp;
        }
        else
        {
          C[j*rsc + j*csc] = Alpha*rtmp + Beta*C[j*rsc + j*csc];
        }
        for( i = (j+1) ; i < N ; i++ )
        {
          tmp = ZERO;
          for( l = 0 ; l < K ; l++ )
          {
            tmp = tmp + A[l*rsa + i*csa]*A[l*rsa + j*csa];
          }
          if( Beta == ZERO )
          {
            C[i*rsc + j*csc] = Alpha*tmp;
          }
          else
          {
            C[i*rsc + j*csc] = Alpha*tmp + Beta*C[i*rsc + j*csc];
          }
        }
      }
    }
  }
  return;
}

template <typename T, typename U>
void libblis_icherk_check(uplo_t uplo, trans_t trans, dim_t N, dim_t K,
  T Alpha,T* A, dim_t rsa, dim_t csa, T Beta, T* C, dim_t rsc, dim_t csc)
{
  T tmp;
  T rtmp;
  dim_t i, j, l;
  bool UPPER, NOTRANS;
  T ONE  = {1.0 , 0.0};
  T ZERO = {0.0 , 0.0};

  UPPER    = (uplo == BLIS_UPPER);
  NOTRANS  = (trans == BLIS_NO_TRANSPOSE) || (trans == BLIS_CONJ_NO_TRANSPOSE);

  //* Quick return if possible.
  if( (N == 0) ||
    (((Alpha.real == ZERO.real) || (K == 0)) && (Beta.real == ONE.real)) )
  {
    return;
  }

//*     And when  alpha.eq.zero.
  if( Alpha.real == ZERO.real )
  {
    if( UPPER )
    {
      if( Beta.real == ZERO.real )
      {
        for( j = 0 ; j < N; j++ )
        {
          for( i = 0 ; i <= j ; i++ )
          {
            C[i*rsc + j*csc] = ZERO;
          }
        }
      }
      else
      {
        for( j = 0 ; j < N ; j++ )
        {
          for( i = 0; i < j ; i++ )
          {
            C[i*rsc + j*csc] = mulc<T>(Beta , C[i*rsc + j*csc]);
          }
          C[j*rsc + j*csc] = mulc<T>(Beta , real<T>(C[j*rsc + j*csc]));
        }
      }
    }
    else
    {
      if( Beta.real == ZERO.real )
      {
        for( j = 0 ; j < N ; j++ )
        {
          for( i = j ; i < N ; i++ )
          {
            C[i*rsc + j*csc] = ZERO;
          }
        }
      }
      else
      {
        for( j = 0 ; j < N ; j++ )
        {
          C[j*rsc + j*csc] = mulc<T>(Beta , real<T>(C[j*rsc + j*csc]));
          for( i = (j+1) ; i < N ; i++ )
          {
            C[i*rsc + j*csc] = mulc<T>(Beta , C[i*rsc + j*csc]);
          }
        }
      }
    }
    return;
  }

  //* Start the operations.
  if( NOTRANS )
  {
    //*Form  C := alpha*A*A**H + beta*C.
    if( UPPER )
    {
      for( j = 0 ; j < N ; j++ )
      {
        if( Beta.real == ZERO.real )
        {
          for( i = 0 ; i <= j ; i++ )
          {
            C[i*rsc + j*csc] = ZERO;
          }
        }
        else if( Beta.real != ONE.real )
        {
          for( i = 0 ; i < j ; i++ )
          {
            C[i*rsc + j*csc] = mulc<T>(Beta , C[i*rsc + j*csc]);
          }
          C[j*rsc + j*csc] = mulc<T>(Beta , real<T>(C[j*rsc + j*csc]));
        }
        else
        {
          C[j*rsc + j*csc] = real<T>(C[j*rsc + j*csc]);
        }

        for( l = 0; l < K ; l++ )
        {
          if((A[j*rsa + l*csa].real != ZERO.real) || (A[j*rsa + l*csa].imag != ZERO.imag))
          {
            tmp = mulc<T>(Alpha , conjugate<T>(A[j*rsa + l*csa]));
            for( i = 0 ; i < j ; i++ )
            {
              C[i*rsc + j*csc] = addc<T>(C[i*rsc + j*csc] , mulc<T>(tmp , A[i*rsa + l*csa]));
            }
            C[j*rsc + j*csc] = addc<T>(real<T>(C[j*rsc + j*csc]) , real<T>(mulc<T>(tmp ,A[i*rsa + l*csa])));
          }
        }
      }
    }
    else
    {
      for( j = 0; j < N ; j++ )
      {
        if( Beta.real == ZERO.real )
        {
          for( i = j ; i < N ; i++ )
          {
            C[j*rsc + j*csc] = ZERO;
          }
        }
        else if( Beta.real != ONE.real )
        {
          C[j*rsc + j*csc] = mulc<T>(Beta ,real<T>(C[j*rsc + j*csc]));
          for( i = (j+1) ; i < N ; i++ )
          {
            C[i*rsc + j*csc] = mulc<T>(Beta , C[i*rsc + j*csc]);
          }
        }
        else
        {
          C[j*rsc + j*csc] = real<T>(C[j*rsc + j*csc]);
        }

        for( l = 0; l < K ; l++ )
        {
          if( (A[j*rsa + l*csa].real != ZERO.real)||(A[j*rsa + l*csa].imag != ZERO.imag) )
          {
            tmp = mulc<T>(Alpha , conjugate<T>(A[j*rsa + l*csa]));
            C[j*rsc + j*csc] = addc<T>(real<T>(C[j*rsc + j*csc]) , real<T>(mulc<T>(tmp , A[j*rsa + l*csa])));
            for( i = (j+1) ; i < N; i++ )
            {
              C[i*rsc + j*csc] = addc<T>(C[i*rsc + j*csc] , mulc<T>(tmp , A[i*rsa + l*csa]));
            }
          }
        }
      }
    }
  }
  else
  {
    //* Form  C := alpha*A**H*A + beta*C.
    if( UPPER )
    {
      for( j = 0 ; j < N ; j++ )
      {
          for( i = 0 ; i < j ; i++ )
          {
            tmp = ZERO;
            for( l = 0 ; l < K ; l++ )
            {
              tmp = addc<T>(tmp , mulc<T>(conjugate<T>(A[l*rsa + i*csa]) , A[l*rsa + j*csa]));
            }
            if( Beta.real == ZERO.real )
            {
              C[i*rsc + j*csc] = mulc<T>(Alpha , tmp);
            }
            else
            {
              C[i*rsc + j*csc] = addc<T>(mulc<T>(Alpha , tmp) , mulc<T>(Beta , C[i*rsc + j*csc]));
            }
        }
        rtmp = ZERO;
        for( l = 0 ; l < K ; l++ )
        {
          rtmp = addc<T>(rtmp , mulc<T>(conjugate<T>(A[l*rsa + j*csa]) , A[l*rsa + j*csa]));
        }
        if( Beta.real == ZERO.real )
        {
          C[j*rsc + j*csc] = mulc<T>(Alpha , rtmp);
        }
        else
        {
          C[j*rsc + j*csc] = addc<T>(mulc<T>(Alpha , rtmp) , mulc<T>(Beta , real<T>(C[j*rsc + j*csc])));
        }
      }
    }
    else
    {
      for( j = 0 ; j < N ; j++ )
      {
        rtmp = ZERO;
        for( l = 0 ; l < K ; l++ )
        {
          rtmp = addc<T>(rtmp , mulc<T>(conjugate<T>(A[l*rsa + j*csa]) , A[l*rsa + j*csa]));
        }
        if( Beta.real == ZERO.real )
        {
          C[j*rsc + j*csc] = mulc<T>(Alpha , rtmp);
        }
        else
        {
          C[j*rsc + j*csc] = addc<T>(mulc<T>(Alpha , rtmp) , mulc<T>(Beta , real<T>(C[j*rsc + j*csc])));
        }
        for( i = (j+1) ; i < N ; i++ )
        {
          tmp = ZERO;
          for( l = 0 ; l < K ; l++ )
          {
            tmp = addc<T>(tmp , mulc<T>(conjugate<T>(A[l*rsa + i*csa]) , A[l*rsa + j*csa]));
          }
          if( Beta.real == ZERO.real )
          {
             C[i*rsc + j*csc] = mulc<T>(Alpha , tmp);
          }
          else
          {
             C[i*rsc + j*csc] = addc<T>(mulc<T>(Alpha , tmp) , mulc<T>(Beta , C[i*rsc + j*csc]));
          }
        }
      }
    }
  }
  return;
}

double libblis_test_iherk_check(
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
  uplo_t uplo    = bli_obj_uplo( c );
  trans_t trans  = bli_obj_onlytrans_status( a );
  double resid   = 0.0;
  dim_t  rsa, csa;
  dim_t  rsc, csc;
  f77_int lda;

  rsa = bli_obj_row_stride( a ) ;
  csa = bli_obj_col_stride( a ) ;
  rsc = bli_obj_row_stride( c ) ;
  csc = bli_obj_col_stride( c ) ;

   if( bli_obj_is_col_stored( c ) ) {
     lda    = bli_obj_col_stride( a );
   } else {
     lda    = bli_obj_row_stride( a );
   }

   int nrowa;
   if (trans == BLIS_NO_TRANSPOSE) {
     nrowa = M;
   } else {
     nrowa = K;
   }

   if (lda < max(1, nrowa)) {
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
      libblis_iherk_check<float>( uplo, trans, M, K, *Alpha,
                                       A, rsa, csa, *Beta, C, rsc, csc );
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
      libblis_iherk_check<double>(uplo, trans, M, K, *Alpha,
                                       A, rsa, csa, *Beta, C, rsc, csc );
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
      Alpha->imag = 0.0 ;
      Beta->imag  = 0.0 ;
      libblis_icherk_check<scomplex, float>(uplo, trans, M, K, *Alpha,
                                        A, rsa, csa, *Beta, C, rsc, csc);
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
      Alpha->imag = 0.0 ;
      Beta->imag  = 0.0 ;
      libblis_icherk_check<dcomplex, double>(uplo, trans, M, K, *Alpha,
                                        A, rsa, csa, *Beta, C, rsc, csc);
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

double libblis_check_nan_herk(obj_t* c, num_t dt ) {
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

