#include "blis_test.h"
#include "blis_utils.h"
#include "test_hemm.h"

using namespace std;

//*  ==========================================================================
//*> HEMM  performs one of the matrix-matrix operations
//*>    C := alpha*A*B + beta*C,
//*> or
//*>    C := alpha*B*A + beta*C,
//*> where alpha and beta are scalars, A is an hermitian matrix and  B and
//*> C are m by n matrices.
//*  ==========================================================================

template <typename T>
void libblis_ihemm_check(side_t side, uplo_t uplo, dim_t M, dim_t N,
  T Alpha, T* A, dim_t rsa, dim_t csa, T* B, dim_t rsb, dim_t csb, T Beta,
  T* C, dim_t rsc, dim_t csc )
{
    T ONE = 1.0;
    T ZERO = 0.0;
    T tmp1, tmp2;
    bool LSIDE, UPPER;
    dim_t i, j, k;

    //*     Test the input parameters.
    LSIDE   = (side == BLIS_LEFT);
    UPPER   = (uplo == BLIS_UPPER);

    if( (M == 0 || N == 0) || ( Alpha == ZERO && Beta == ONE ) )
      return;

    //*     And when  Alpha.eq.zero.
    if( Alpha == ZERO )
    {
        if( Beta == ZERO )
        {
            for( j = 0 ; j < N ; j++ )
            {
                for( i = 0 ; i < M ; i++ )
                {
                    C[i*rsc + j*csc] = ZERO;
                }
            }
        }
        else
        {
            for( j = 0 ; j < N ; j++ )
            {
                for( i = 0 ; i < M ; i++ )
                {
                    C[i*rsc + j*csc] = Beta*C[i*rsc + j*csc];
                }
            }
        }
        return;
    }

    //*     Start the operations.
    if( LSIDE )
    {
        //* Form  C := Alpha*A*B + Beta*C.
        if( UPPER )
        {
            for( j = 0 ; j < N ; j++ )
            {
                for( i = 0 ; i < M ; i++ )
                {
                    tmp1 = Alpha*B[i*rsb + j*csb];
                    tmp2 = ZERO;
                    for( k = 0 ; k < i ; k++ )
                    {
                        C[k*rsc + j*csc] = C[k*rsc + j*csc] + tmp1*A[k*rsa + i*csa];
                        tmp2 = tmp2 + B[k*rsb + j*csb] * A[k*rsa + i*csa];
                    }
                    if (Beta == ZERO)
                    {
                        C[i*rsc + j*csc] = tmp1*A[i*rsa + i*csa] + Alpha*tmp2;
                    }
                    else
                    {
                        C[i*rsc + j*csc] = Beta*C[i*rsc + j*csc] + tmp1*A[i*rsa + i*csa] + Alpha*tmp2;
                    }
                }
            }
        }
        else
        {
            for( j = 0 ; j < N ; j++ )
            {
                for( i = (M-1) ; i >= 0 ; i-- )
                {
                    tmp1 = Alpha*B[i*rsb + j*csb];
                    tmp2 = ZERO;
                    for( k = (i+1) ; k < M ; k++ )
                    {
                        C[k*rsc + j*csc] = C[k*rsc + j*csc] + tmp1*A[k*rsa + i*csa];
                        tmp2 = tmp2 + B[k*rsb + j*csb]*A[k*rsa + i*csa];
                    }
                    if (Beta == ZERO)
                    {
                        C[i*rsc + j*csc] = tmp1*A[i*rsa + i*csa] + Alpha*tmp2;
                    }
                    else
                    {
                        C[i*rsc + j*csc] = Beta*C[i*rsc + j*csc] + tmp1*A[i*rsa + i*csa] + Alpha*tmp2;
                    }
                }
            }
        }
    }
    else
    {
        //* Form  C := Alpha*B*A + Beta*C.
        for( j = 0 ; j < N ; j++ )
        {
            tmp1 = Alpha*A[j*rsa + j*csa];
            if( Beta == ZERO )
            {
                for(i = 0 ; i < M ; i++)
                {
                    C[i*rsc + j*csc] = tmp1*B[i*rsb + j*csb];
                }
            }
            else
            {
                for(i = 0 ; i < M ; i++)
                {
                    C[i*rsc + j*csc] = Beta*C[i*rsc + j*csc] + tmp1*B[i*rsb + j*csb];
                }
            }
            for( k = 0 ; k < j ; k++ )
            {
                if( UPPER )
                {
                    tmp1 = Alpha*A[k*rsa + j*csa];
                }
                else
                {
                    tmp1 = Alpha*A[j*rsa + k*csa];
                }
                for(i = 0 ; i < M ; i++)
                {
                    C[i*rsc + j*csc] = C[i*rsc + j*csc] + tmp1*B[i*rsb + k*csb];
                }
            }
            for( k = (j+1) ; k < N ; k++ )
            {
                if( UPPER )
                {
                    tmp1 = Alpha*A[j*rsa + k*csa];
                }
                else
                {
                    tmp1 = Alpha*A[k*rsa + j*csa];
                }
                for(i = 0 ; i < M ; i++)
                {
                    C[i*rsc + j*csc] = C[i*rsc + j*csc] + tmp1*B[i*rsb + k*csb];
                }
            }
        }
    }
    return;
}

template <typename T, typename U>
void libblis_ichemm_check(side_t side, uplo_t uplo, dim_t M, dim_t N,
  T Alpha, T* A, dim_t rsa, dim_t csa, T* B, dim_t rsb, dim_t csb, T Beta,
  T* C, dim_t rsc, dim_t csc )
{
    T ONE  = {1.0 , 0.0};
    T ZERO = {0.0 , 0.0};
    T tmp1, tmp2;
    bool LSIDE, UPPER;
    dim_t i, j, k;

    //*     Test the input parameters.
    LSIDE   = (side == BLIS_LEFT);
    UPPER   = (uplo == BLIS_UPPER);

    if( (M == 0 || N == 0) || ( Alpha.real == ZERO.real && Beta.real == ONE.real ) )
      return;

    //*     And when  Alpha.eq.zero.
    if( Alpha.real == ZERO.real )
    {
        if( Beta.real == ZERO.real )
        {
            for( j = 0 ; j < N ; j++ )
            {
                for( i = 0 ; i < M ; i++ )
                {
                    C[i*rsc + j*csc] = ZERO;
                }
            }
        }
        else
        {
            for( j = 0 ; j < N ; j++ )
            {
                for( i = 0 ; i < M ; i++ )
                {
                    C[i*rsc + j*csc] = mulc<T>(Beta , C[i*rsc + j*csc]);
                }
            }
        }
        return;
    }

    //*     Start the operations.
    if( LSIDE )
    {
        //* Form  C := Alpha*A*B + Beta*C.
        if( UPPER )
        {
            for( j = 0 ; j < N ; j++ )
            {
                for( i = 0 ; i < M ; i++ )
                {
                    tmp1 = mulc<T>(Alpha , B[i*rsb + j*csb]);
                    tmp2 = ZERO;
                    for( k = 0 ; k < i ; k++ )
                    {
                        C[k*rsc + j*csc] = addc<T>(C[k*rsc + j*csc] , mulc<T>(tmp1 , A[k*rsa + i*csa]));
                        tmp2 = addc<T>(tmp2 , mulc<T>(B[k*rsb + j*csb] , conjugate<T>(A[k*rsa + i*csa])));
                    }
                    if (Beta.real == ZERO.real)
                    {
                        C[i*rsc + j*csc] = addc<T>(mulc<T>(tmp1 , real<T>(A[i*rsa + i*csa])) , mulc<T>(Alpha , tmp2));
                    }
                    else
                    {
                        tmp2 = addc<T>(mulc<T>(tmp1 , real<T>(A[i*rsa + i*csa])) , mulc<T>(Alpha , tmp2));
                        C[i*rsc + j*csc] = addc<T>(mulc<T>(Beta , C[i*rsc + j*csc]) , tmp2);
                        //C[i*rsc + j*csc] = mulc<T>(Beta , C[i*rsc + j*csc]) + mulc<T>(tmp1 , real<T>(A[i*rsa + i*csa])) + mulc<T>(Alpha , tmp2);
                    }
                }
            }
        }
        else
        {
            for( j = 0 ; j < N ; j++ )
            {
                for( i = (M-1) ; i >= 0 ; i-- )
                {
                    tmp1 = mulc<T>(Alpha , B[i*rsb + j*csb]);
                    tmp2 = ZERO;
                    for( k = (i+1) ; k < M ; k++ )
                    {
                        C[k*rsc + j*csc] = addc<T>(C[k*rsc + j*csc] , mulc<T>(tmp1 , A[k*rsa + i*csc]));
                        tmp2 = addc<T>(tmp2 , mulc<T>(B[k*rsb + j*csb] , conjugate<T>(A[k*rsa + i*csa])));
                    }
                    if (Beta.real == ZERO.real)
                    {
                        C[i*rsc + j*csc] = addc<T>(mulc<T>(tmp1 , real<T>(A[i*rsa + i*csa])) , mulc<T>(Alpha , tmp2));
                    }
                    else
                    {
                        tmp2 = addc<T>(mulc<T>(tmp1 , real<T>(A[i*rsa + i*csa])) , mulc<T>(Alpha , tmp2));
                        C[i*rsc + j*csc] = addc<T>(mulc<T>(Beta , C[i*rsc + j*csc]) , tmp2);
                        //C[i*rsc + j*csc] = mulc<T>(Beta , C[i*rsc + j*csc]) + mulc<T>(tmp1 , real<T>(A[i*rsa + i*csa])) + mulc<T>(Alpha , tmp2);
                    }
                }
            }
        }
    }
    else
    {
        //* Form  C := Alpha*B*A + Beta*C.
        for( j = 0 ; j < N ; j++ )
        {
            tmp1 = mulc<T>(Alpha , real<T>(A[j*rsa + j*csa]));
            if (Beta.real == ZERO.real)
            {
                for(i = 0 ; i < M ; i++)
                {
                    C[i*rsc + j*csc] = mulc<T>(tmp1 , B[i*rsb + j*csb]);
                }
            }
            else
            {
                for(i = 0 ; i < M ; i++)
                {
                    C[i*rsc + j*csc] = addc<T>(mulc<T>(Beta , C[i*rsc + j*csc]) , mulc<T>(tmp1 , B[i*rsb + j*csb]));
                }
            }
            for( k = 0 ; k < j ; k++ )
            {
                if( UPPER )
                {
                    tmp1 = mulc<T>(Alpha , A[k*rsa + j*csa]);
                }
                else
                {
                    tmp1 = mulc<T>(Alpha , conjugate<T>(A[j*rsa + k*csa]));
                }
                for(i = 0 ; i < M ; i++)
                {
                    C[i*rsc + j*csc] = addc<T>(C[i*rsc + j*csc] , mulc<T>(tmp1 , B[i*rsb + k*csb]));
                }
            }
            for( k = (j+1) ; k < N ; k++ )
            {
                if( UPPER )
                {
                    tmp1 = mulc<T>(Alpha , conjugate<T>(A[j*rsa + k*csa]));
                }
                else
                {
                    tmp1 = mulc<T>(Alpha , A[k*rsa + j*csa]);
                }
                for(i = 0 ; i < M ; i++)
                {
                    C[i*rsc + j*csc] = addc<T>(C[i*rsc + j*csc] , mulc<T>(tmp1 , B[i*rsb + k*csb]));
                }
            }
        }
    }
    return;
}

double libblis_test_ihemm_check
    (
      test_params_t* params,
      side_t         side,
      obj_t*         alpha,
      obj_t*         a,
      obj_t*         b,
      obj_t*         beta,
      obj_t*         c,
      obj_t*         c_orig
    )
{

    num_t dt     = bli_obj_dt( a );
    uplo_t uploa = bli_obj_uplo( a );
    dim_t M      = bli_obj_length( c );
    dim_t N      = bli_obj_width( c );
    dim_t rsa    = bli_obj_row_stride( a ) ;
    dim_t csa    = bli_obj_col_stride( a ) ;
    dim_t rsb    = bli_obj_row_stride( b ) ;
    dim_t csb    = bli_obj_col_stride( b ) ;
    dim_t rsc    = bli_obj_row_stride( c ) ;
    dim_t csc    = bli_obj_col_stride( c ) ;
    double resid = 0.0;

    switch( dt )  {
        case BLIS_FLOAT :
        {
            float*   Alpha = (float*) bli_obj_buffer( alpha );
            float*   A     = (float*) bli_obj_buffer( a );
            float*   B     = (float*) bli_obj_buffer( b );
            float*   Beta  = (float*) bli_obj_buffer( beta );
            float*   C     = (float*) bli_obj_buffer( c_orig );
            float*   CC    = (float*) bli_obj_buffer( c );
            libblis_ihemm_check<float>(side, uploa, M, N, *Alpha, A, rsa, csa,
                                            B, rsb, csb, *Beta, C, rsc, csc );
            resid = computediffrm(M, N, CC, C, rsc, csc);
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
            libblis_ihemm_check<double>(side, uploa, M, N, *Alpha, A, rsa, csa,
                                            B, rsb, csb, *Beta, C, rsc, csc );
            resid = computediffrm(M, N, CC, C, rsc, csc);
        }
            break;
        case BLIS_SCOMPLEX :
        {
            scomplex*   Alpha = (scomplex*) bli_obj_buffer( alpha );
            scomplex*   Beta  = (scomplex*) bli_obj_buffer( beta );
            scomplex*   A     = (scomplex*) bli_obj_buffer( a );
            scomplex*   B     = (scomplex*) bli_obj_buffer( b );
            scomplex*   C     = (scomplex*) bli_obj_buffer( c_orig );
            scomplex*   CC    = (scomplex*) bli_obj_buffer( c );
            libblis_ichemm_check<scomplex, float>(side, uploa, M, N, *Alpha,
                             A, rsa, csa, B, rsb, csb, *Beta, C, rsc, csc );
            resid = computediffim(M, N, CC, C, rsc, csc);
            break;
        }
        case BLIS_DCOMPLEX :
        {
            dcomplex*   Alpha = (dcomplex*) bli_obj_buffer( alpha );
            dcomplex*   Beta  = (dcomplex*) bli_obj_buffer( beta );
            dcomplex*   A     = (dcomplex*) bli_obj_buffer( a );
            dcomplex*   B     = (dcomplex*) bli_obj_buffer( b );
            dcomplex*   C     = (dcomplex*) bli_obj_buffer( c_orig );
            dcomplex*   CC    = (dcomplex*) bli_obj_buffer( c );
            libblis_ichemm_check<dcomplex, double>(side, uploa, M, N, *Alpha,
                             A, rsa, csa, B, rsb, csb, *Beta, C, rsc, csc );
            resid = computediffim(M, N, CC, C, rsc, csc);
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

double libblis_check_nan_hemm(obj_t* c, num_t dt ) {
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