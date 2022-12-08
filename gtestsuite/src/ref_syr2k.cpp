#include "blis_test.h"
#include "blis_utils.h"
#include "test_syr2k.h"

using namespace std;

//*  ==========================================================================
//*> SYR2K  performs one of the symmetric rank 2k operations
//*>    C := alpha*A*B**T + alpha*B*A**T + beta*C,
//*> or
//*>    C := alpha*A**T*B + alpha*B**T*A + beta*C,
//*> where  alpha and beta  are scalars, C is an  n by n  symmetric matrix
//*> and  A and B  are  n by k  matrices  in the  first  case  and  k by n
//*> matrices in the second case.
//*  ==========================================================================

template <typename T>
void libblis_isyr2k_check( uplo_t uplo, trans_t trans, dim_t N, dim_t K,
  T Alpha, T* A, dim_t rsa, dim_t csa, T* B, dim_t rsb, dim_t csb, T Beta,
  T* C, dim_t rsc, dim_t csc )
{
    T tmp1, tmp2;
    int i, j, l;
    bool UPPER, NOTRANS;

    T ONE  = 1.0 ;
    T ZERO = 0.0 ;

    //*     Test the input parameters.
    UPPER   = ( uplo == BLIS_UPPER );
    NOTRANS = ( trans == BLIS_NO_TRANSPOSE );

    if( N == 0 || (( Alpha == ZERO || K == 0 ) && Beta == ONE ))
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
                    for( i = 0 ; i <= j ; i++ )
                    {
                        C[i*rsc + j*csc] = Beta*C[i*rsc + j*csc];
                    }
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
                    for( i = j ; i < N ; i++ )
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
        //* C := alpha*A*B**T + alpha*B*A**T + C.
        if( UPPER )
        {
            for( j = 0 ; j < N ; j++ )
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
                    for( i = 0 ; i <= j ; i++ )
                    {
                        C[i*rsc + j*csc] = Beta*C[i*rsc + j*csc];
                    }
                }
                for( l = 0 ; l < K ; l++ )
                {
                    if( (A[j*rsa + l*csa] != ZERO) || (B[j*rsb + l*csb] != ZERO) )
                    {
                        tmp1 = Alpha*B[j*rsb + l*csb];
                        tmp2 = Alpha*A[j*rsa + l*csa];
                        for( i = 0 ; i <= j ; i++ )
                        {
                            C[i*rsc + j*csc] = C[i*rsc + j*csc] + A[i*rsa + l*csa]*tmp1 +  B[i*rsb + l*csb]*tmp2;
                        }
                    }
                }
            }
        }
        else
        {
            for( j = 0 ; j < N ; j++ )
            {
                if( Beta == ZERO )
                {
                    for( i = j ; i < N ; i++ )
                    {
                        C[i*rsc + j*csc] = ZERO;
                    }
                }
                else if( Beta != ONE )
                {
                    for( i = j ; i < N ; i++ )
                    {
                        C[i*rsc + j*csc] = Beta*C[i*rsc + j*csc];
                    }
                }
                for( l = 0 ; l < K ; l++ )
                {
                    if( (A[j*rsa + l*csa] != ZERO) || (B[j*rsb + l*csb] != ZERO) )
                    {
                        tmp1 = Alpha*B[j*rsb + l*csb];
                        tmp2 = Alpha*A[j*rsa + l*csa];
                        for( i = j; i < N ; i++ )
                        {
                            C[i*rsc + j*csc] = C[i*rsc + j*csc] + A[i*rsa + l*csa]*tmp1 + B[i*rsb + l*csb]*tmp2;
                        }
                    }
                }
            }
        }
    }
    else
    {
        //* C := alpha*A**T*B + alpha*B**T*A + C.
        if( UPPER )
        {
            for( j = 0 ; j < N ; j++ )
            {
                for( i = 0 ; i <= j ; i++ )
                {
                    tmp1 = ZERO;
                    tmp2 = ZERO;
                    for( l = 0 ; l < K ; l++ )
                    {
                        tmp1 = tmp1 + A[l*rsa + i*csa]*B[l*rsb + j*csb];
                        tmp2 = tmp2 + B[l*rsb + i*csb]*A[l*rsa + j*csa];
                    }
                    if( Beta == ZERO )
                    {
                        C[i*rsc + j*csc] = Alpha*tmp1 + Alpha*tmp2;
                    }
                    else
                    {
                        C[i*rsc + j*csc] = Beta*C[i*rsc + j*csc] + Alpha*tmp1 + Alpha*tmp2;
                    }
                }
            }
        }
        else
        {
            for( j = 0 ; j < N ; j++ )
            {
                for( i = j ; i < N ; i++ )
                {
                    tmp1 = ZERO;
                    tmp2 = ZERO;
                    for( l = 0 ; l < K ; l++ )
                    {
                        tmp1 = tmp1 + A[l*rsa + i*csa]*B[l*rsb + j*csb];
                        tmp2 = tmp2 + B[l*rsb + i*csb]*A[l*rsa + j*csa];
                    }
                    if( Beta == ZERO )
                    {
                        C[i*rsc + j*csc] = Alpha*tmp1 + Alpha*tmp2;
                    }
                    else
                    {
                        C[i*rsc + j*csc] = Beta*C[i*rsc + j*csc] + Alpha*tmp1 + Alpha*tmp2;
                    }
                }
            }
        }
    }
    return;
}

template <typename T, typename U>
void libblis_icsyr2k_check( uplo_t uplo, trans_t trans, dim_t N, dim_t K,
  T Alpha, T* A, dim_t rsa, dim_t csa, T* B, dim_t rsb, dim_t csb, T Beta,
  T* C, dim_t rsc, dim_t csc )
{
    T tmp1, tmp2;
    T tmpa, tmpb;
    int i, j, l;
    bool UPPER, NOTRANS;

    T ONE  = { 1.0 , 0.0 };
    T ZERO = { 0.0 , 0.0 };

    //*     Test the input parameters.
    UPPER   = (uplo == BLIS_UPPER);
    NOTRANS = (trans == BLIS_NO_TRANSPOSE);

    if( N == 0 || (( Alpha.real == ZERO.real || K == 0 ) && Beta.real == ONE.real ))
      return;

    //*     And when  alpha.eq.zero.
    if( Alpha.real == ZERO.real )
    {
        if( UPPER )
        {
            if( Beta.real == ZERO.real )
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
                    for( i = 0 ; i <= j ; i++)
                    {
                        C[i*rsc + j*csc] = mulc<T>(Beta , C[i*rsc + j*csc]);
                    }
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
                    for( i = j ; i < N ; i++ )
                    {
                        C[i*rsc + j*csc] = mulc<T>(Beta , C[i*rsc + j*csc]);
                    }
                }
            }
        }
        return;
    }

    //*     Start the operations.
    if( NOTRANS )
    {
        //*        Form  C := alpha*A*B**H + conjg( alpha )*B*A**H + C.
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
                else if( (Beta.real != ONE.real) || (Beta.imag != ONE.imag) )
                {
                    for( i = 0 ; i <= j ; i++ )
                    {
                        C[i*rsc + j*csc] = mulc<T>(Beta , C[i*rsc + j*csc]);
                    }
                }
                for( l = 0 ; l < K ; l++ )
                {
                   if( ((A[j*rsa + l*csa].real != ZERO.real) || (A[j*rsa + l*csa].imag != ZERO.imag))
                     || ((B[j*rsb + l*csb].real != ZERO.real) || (B[j*rsb + l*csb].imag != ZERO.imag)) )
                    {
                        tmp1 = mulc<T>(Alpha , B[j*rsb + l*csb]);
                        tmp2 = mulc<T>(Alpha , A[j*rsa + l*csa]);
                        for( i = 0 ; i <= j ; i++)
                        {
                            tmpa = mulc<T>(A[i*rsa + l*csa] , tmp1);
                            tmpb = mulc<T>(B[i*rsb + l*csb] , tmp2);
                            tmpa = addc<T>(tmpa , tmpb);
                            C[i*rsc + j*csc] = addc<T>(C[i*rsc + j*csc] , tmpa);
                        }
                    }
                }
            }
        }
        else
        {
            for( j = 0 ; j < N ; j++ )
            {
                if( (Beta.real == ZERO.real) || (Beta.imag == ZERO.imag) )
                {
                    for( i = j ; i < N ; i++ )
                    {
                        C[i*rsc + j*csc] = ZERO;
                    }
                }
                else if( (Beta.real != ONE.real) || (Beta.imag != ONE.imag) )
                {
                    for( i = j ; i < N ; i++ )
                    {
                        C[i*rsc + j*csc] = mulc<T>(Beta , C[i*rsc + j*csc]);
                    }
                }
                for( l = 0 ; l < K ; l++ )
                {
                    if( ((A[j*rsa + l*csa].real != ZERO.real) || (A[j*rsa + l*csa].imag != ZERO.imag))
                     || ((B[j*rsb + l*csb].real != ZERO.real) || (B[j*rsb + l*csb].imag != ZERO.imag)) )
                    {
                        tmp1 = mulc<T>(Alpha , B[j*rsb + l*csb]);
                        tmp2 = mulc<T>(Alpha , A[j*rsa + l*csa]);
                        for( i = j ; i < N ; i++ )
                        {
                            tmpa = mulc<T>(A[i*rsa + l*csa] , tmp1);
                            tmpb = mulc<T>(B[i*rsb + l*csb] , tmp2);
                            tmpa = addc<T>(tmpa, tmpb);
                            C[i*rsc + j*csc] = addc<T>(C[i*rsc + j*csc] , tmpa);
                        }
                    }
                }
            }
        }
    }
    else
    {
        //* Form  C := alpha*A**T*B + alpha*B**T*A + C.
        if( UPPER )
        {
            for( j = 0 ; j < N ; j++ )
            {
                for( i = 0 ; i <= j ; i++ )
                {
                    tmp1 = ZERO;
                    tmp2 = ZERO;
                    for( l = 0 ; l < K ; l++ )
                    {
                        tmp1 = addc<T>(tmp1 , mulc<T>(A[l*rsa + i*csa] , B[l*rsb + j*csb]));
                        tmp2 = addc<T>(tmp2 , mulc<T>(B[l*rsb + i*csb] , A[l*rsa + j*csa]));
                    }
                    if( (Beta.real == ZERO.real) || (Beta.imag == ZERO.imag) )
                    {
                        C[i*rsc + j*csc] = addc<T>(mulc<T>(Alpha , tmp1) , mulc<T>(Alpha ,tmp2));
                    }
                    else
                    {
                        tmpa = mulc<T>(Alpha , tmp1);
                        tmpb = mulc<T>(Alpha , tmp2);
                        tmpa = addc<T>(tmpa , tmpb);
                        C[i*rsc + j*csc] = addc<T>(mulc<T>(Beta , C[i*rsc + j*csc]) ,tmpa);
                    }
                 }
             }
        }
        else
        {
            for( j = 0 ; j < N ; j++ )
            {
                for( i = j ; i < N ; i++ )
                {
                    tmp1 = ZERO;
                    tmp2 = ZERO;
                    for( l = 0 ; l < K ; l++ )
                    {
                        tmp1 = addc<T>(tmp1 , mulc<T>(A[l*rsa + i*csa] , B[l*rsb + j*csb]));
                        tmp2 = addc<T>(tmp2 , mulc<T>(B[l*rsb + i*csb] , A[l*rsa + j*csa]));
                    }
                    if( (Beta.real == ZERO.real) || (Beta.imag == ZERO.imag) )
                    {
                        C[i*rsc + j*csc] = addc<T>(mulc<T>(Alpha , tmp1) , mulc<T>(Alpha , tmp2));
                    }
                    else
                    {
                        tmpa = mulc<T>(Alpha , tmp1);
                        tmpb = mulc<T>(Alpha , tmp2);
                        tmpa = addc<T>(tmpa, tmpb);
                        C[i*rsc + j*csc] = addc<T>(mulc<T>(Beta , C[i*rsc + j*csc]) , tmpa);
                    }
                }
            }
        }
    }
    return;
}

double libblis_test_isyr2k_check
    (
      test_params_t* params,
      obj_t*         alpha,
      obj_t*         a,
      obj_t*         b,
      obj_t*         beta,
      obj_t*         c,
      obj_t*         c_orig
    )
{
    num_t dt      = bli_obj_dt( c );
    uplo_t uploc  = bli_obj_uplo( c );
    dim_t  M      = bli_obj_length( c );
    dim_t  K      = bli_obj_width_after_trans( a );
    trans_t trans = bli_obj_onlytrans_status( a );
    dim_t rsa     = bli_obj_row_stride( a ) ;
    dim_t csa     = bli_obj_col_stride( a ) ;
    dim_t rsb     = bli_obj_row_stride( b ) ;
    dim_t csb     = bli_obj_col_stride( b ) ;
    dim_t rsc     = bli_obj_row_stride( c ) ;
    dim_t csc     = bli_obj_col_stride( c ) ;
    double resid  = 0.0;
    f77_int lda, ldb, ldc;

   if( bli_obj_is_col_stored( c ) ) {
     lda  = bli_obj_col_stride( a );
     ldb  = bli_obj_col_stride( b );
     ldc  = bli_obj_col_stride( c );
   } else {
     lda  = bli_obj_row_stride( a );
     ldb  = bli_obj_row_stride( b );
     ldc  = bli_obj_row_stride( c );
   }

   int nrowa;
   if (trans == BLIS_NO_TRANSPOSE) {
     nrowa = M;
   } else {
     nrowa = K;
   }

   if( lda < max(1, nrowa) ) {
     return resid;
   }
   if( ldb < max(1, nrowa) ) {
     return resid;
   }
   if( ldc < max(1, (int)M) ) {
     return resid;
   }

    switch( dt )  {
        case BLIS_FLOAT :
        {
            float*   Alpha = (float*) bli_obj_buffer( alpha );
            float*   A     = (float*) bli_obj_buffer( a );
            float*   B     = (float*) bli_obj_buffer( b );
            float*   Beta  = (float*) bli_obj_buffer( beta );
            float*   C     = (float*) bli_obj_buffer( c_orig );
            float*   CC    = (float*) bli_obj_buffer( c );
            libblis_isyr2k_check<float>(uploc, trans, M, K, *Alpha, A,
                                 rsa, csa, B, rsb, csb, *Beta, C, rsc, csc );
            resid = computediffrm(M, M, CC, C, rsc, csc);
			break;
        }
        case BLIS_DOUBLE :
        {
            double*  Alpha = (double*) bli_obj_buffer( alpha );
            double*  A     = (double*) bli_obj_buffer( a );
            double*  B     = (double*) bli_obj_buffer( b );
            double*  Beta  = (double*) bli_obj_buffer( beta );
            double*  C     = (double*) bli_obj_buffer( c_orig );
            double*  CC    = (double*) bli_obj_buffer( c );
            libblis_isyr2k_check<double>(uploc, trans, M, K, *Alpha, A,
                                 rsa, csa, B, rsb, csb, *Beta, C, rsc, csc );
            resid = computediffrm(M, M, CC, C, rsc, csc);
        }
            break;
        case BLIS_SCOMPLEX :
        {
            scomplex*   Alpha = (scomplex*) bli_obj_buffer( alpha );
            scomplex*   A     = (scomplex*) bli_obj_buffer( a );
            scomplex*   B     = (scomplex*) bli_obj_buffer( b );
            scomplex*   Beta  = (scomplex*) bli_obj_buffer( beta );
            scomplex*   C     = (scomplex*) bli_obj_buffer( c_orig );
            scomplex*   CC    = (scomplex*) bli_obj_buffer( c );
            libblis_icsyr2k_check<scomplex, float>(uploc, trans, M, K, *Alpha,
                                A, rsa, csa, B, rsb, csb, *Beta, C, rsc, csc );
            resid = computediffim(M, M, CC, C, rsc, csc);
            break;
        }
        case BLIS_DCOMPLEX :
        {
            dcomplex*   Alpha = (dcomplex*) bli_obj_buffer( alpha );
            dcomplex*   A     = (dcomplex*) bli_obj_buffer( a );
            dcomplex*   B     = (dcomplex*) bli_obj_buffer( b );
            dcomplex*   Beta  = (dcomplex*) bli_obj_buffer( beta );
            dcomplex*   C     = (dcomplex*) bli_obj_buffer( c_orig );
            dcomplex*   CC    = (dcomplex*) bli_obj_buffer( c );
            libblis_icsyr2k_check<dcomplex, double>(uploc, trans, M, K, *Alpha,
                                A, rsa, csa, B, rsb, csb, *Beta, C, rsc, csc );
            resid = computediffim(M, M, CC, C, rsc, csc);
            break;
        }
        default :
            bli_check_error_code( BLIS_INVALID_DATATYPE );
    }
    return abs(resid);
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

double libblis_check_nan_syr2k(obj_t* c, num_t dt ) {
  dim_t  rsc, csc;
  double resid = 0.0;

  if( bli_obj_row_stride( c ) == 1 ) {
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
      resid = libblis_check_nan_complex<scomplex>( rsc, csc, c );
      break;
    }
    case BLIS_DCOMPLEX:
    {
      resid = libblis_check_nan_complex<dcomplex>( rsc, csc, c );
      break;
    }
    default :
      bli_check_error_code( BLIS_INVALID_DATATYPE );
  }

  return resid;
}