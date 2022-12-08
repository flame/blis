#include "blis_test.h"
#include "blis_utils.h"
#include "test_trmm3.h"

using namespace std;

//*  ==========================================================================
//*> TRMM  performs one of the matrix-matrix operations
//*>    C := beta * C_orig + alpha * transa(A) * transb(B)
//*> or
//*>    C := beta * C_orig + alpha * transb(B) * transa(A)
//*> where alpha and beta are scalars, A is an triangular matrix and  B and
//*> C are m by n matrices.
//*  ==========================================================================

template <typename T>
void libblis_itrmm3_check(side_t side, uplo_t uplo, diag_t diaga,
  dim_t M, dim_t N, T Alpha, T* A, dim_t rsa, dim_t csa,
  T* B, dim_t rsb, dim_t csb, T Beta, T* C, dim_t rsc, dim_t csc )
{
    T ONE = 1.0;
    T ZERO = 0.0;
    T tmp;
    bool LSIDE, UPPER, UNITDA;
    dim_t i, j, k;

    //*     Test the input parameters.
    LSIDE   = ( side == BLIS_LEFT  );
    UPPER   = ( uplo == BLIS_UPPER );
    UNITDA  = ( diaga == BLIS_UNIT_DIAG );

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

    if( UNITDA )
    {
        dim_t dim;
        if( LSIDE )        dim = M;
        else               dim = N;
        for( i = 0 ; i < dim ; i++ )
        {
            for( j = 0 ; j < dim ; j++ )
            {
                if( i==j )
                    A[i*rsa + j*csa] = ONE ;
            }
        }
    }

    //*     Start the operations.
    if( LSIDE )
    {
        //* Form  C := beta * C_orig + alpha * transa(A) * transb(B)
        if( UPPER )
        {
            for( j = 0 ; j < N ; j++ )
            {
                for( i = 0 ; i < M ; i++ )
                {
                    tmp = ZERO;
                    for( k = i ; k < M ; k++ )
                    {
                        tmp += A[i*rsa + k*csa] * B[k*rsb + j*csb];
                    }
                    if( Beta == ZERO )
                    {
                        C[i*rsc + j*csc] = Alpha*tmp;
                    }
                    else
                    {
                        C[i*rsc + j*csc] = Beta*C[i*rsc + j*csc] + Alpha*tmp;
                    }
                }
            }
        }
        else
        {
            for( j = 0 ; j < N ; j++ )
            {
                for( i = 0 ; i < M ; i++ )
                {
                    tmp = ZERO;
                    for( k = 0 ; k <= i ; k++ )
                    {
                        tmp += A[i*rsa + k*csa] * B[k*rsb + j*csb];
                    }
                    if( Beta == ZERO )
                    {
                        C[i*rsc + j*csc] = Alpha*tmp;
                    }
                    else
                    {
                        C[i*rsc + j*csc] = Beta*C[i*rsc + j*csc] + Alpha*tmp;
                    }
                }
            }
        }
    }
    else
    {
        //* C := beta * C_orig + alpha * transb(B) * transa(A)
        if( UPPER )
        {
            for( i = 0 ; i < M ; i++ )
            {
                for( j = 0 ; j < N ; j++ )
                {
                    tmp = ZERO ;
                    for( k = 0 ; k <= j ; k++ )
                    {
                        tmp += B[i*rsb + k*csb]* A[k*rsa + j*csa];
                    }
                    if( Beta == ZERO )
                    {
                        C[i*rsc + j*csc] = Alpha*tmp;
                    }
                    else
                    {
                        C[i*rsc + j*csc] = Beta*C[i*rsc + j*csc] + Alpha*tmp;
                    }
                }
            }
        }
        else
        {
            for( i = 0 ; i < M ; i++ )
            {
                for( j = 0 ; j < N ; j++ )
                {
                    tmp = ZERO ;
                    for( k = j ; k < N ; k++ )
                    {
                        tmp += B[i*rsb + k*csb]* A[k*rsa + j*csa];
                    }
                    if( Beta == ZERO )
                    {
                        C[i*rsc + j*csc] = Alpha*tmp;
                    }
                    else
                    {
                        C[i*rsc + j*csc] = Beta*C[i*rsc + j*csc] + Alpha*tmp;
                    }
                }
            }
        }
    }
    return;
}

template <typename T, typename U>
void libblis_ictrmm3_check(side_t side, uplo_t uplo, diag_t diaga, dim_t M,
 dim_t N, T Alpha, T* A, dim_t rsa, dim_t csa, bool conja, T* B, dim_t rsb,
 dim_t csb, bool conjb, T Beta, T* C, dim_t rsc, dim_t csc )
{
    T ONE  = {1.0 , 0.0};
    T ZERO = {0.0 , 0.0};
    T tmp;
    bool LSIDE, UPPER, UNITDA;
    dim_t i, j, k;

    //*     Test the input parameters.
    LSIDE   = ( side == BLIS_LEFT  );
    UPPER   = ( uplo == BLIS_UPPER );
    UNITDA  = ( diaga == BLIS_UNIT_DIAG );

    if( (M == 0 || N == 0) || ( Alpha.real == ZERO.real && Beta.real == ONE.real ) )
      return;

    if( conja )
    {
        dim_t dim;
        if( LSIDE )        dim = M;
        else               dim = N;
        for( i = 0 ; i < dim ; i++ )
        {
            for( j = 0 ; j < dim ; j++ )
            {
                A[i*rsa + j*csa] = conjugate<T>(A[i*rsa + j*csa]);
            }
        }
    }

    if( conjb )
    {
        for( j = 0 ; j < N ; j++ )
        {
            for( i = 0 ; i < M ; i++ )
            {
                B[i*rsc + j*csc] = conjugate<T>(B[i*rsc + j*csc]);
            }
        }
    }

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

    if( UNITDA )
    {
        dim_t dim;
        if( LSIDE )        dim = M;
        else               dim = N;
        for( i = 0 ; i < dim ; i++ )
        {
            for( j = 0 ; j < dim ; j++ )
            {
                if( i==j )
                    A[i*rsa + j*csa] = ONE ;
            }
        }
    }

    //*     Start the operations.
    if( LSIDE )
    {
        //* Form  C := beta * C_orig + alpha * transa(A) * transb(B)
        if( UPPER )
        {
            for( j = 0 ; j < N ; j++ )
            {
                for( i = 0 ; i < M ; i++ )
                {
                    tmp = ZERO;
                    for( k = i ; k < M ; k++ )
                    {
                        tmp = addc<T>(tmp , mulc<T>(A[i*rsa + k*csa] , B[k*rsb + j*csb]));
                    }
                    if( (Beta.real == ZERO.real) || (Beta.imag == ZERO.imag) )
                    {
                        C[i*rsc + j*csc] = mulc<T>(Alpha , tmp);
                    }
                    else
                    {
                        C[i*rsc + j*csc] = addc<T>(mulc<T>(Beta , C[i*rsc + j*csc]) , mulc<T>(Alpha , tmp));
                    }
                }
            }
        }
        else
        {
            for( j = 0 ; j < N ; j++ )
            {
                for( i = 0 ; i < M ; i++ )
                {
                    tmp = ZERO;
                    for( k = 0 ; k <= i ; k++ )
                    {
                        tmp = addc<T>(tmp , mulc<T>(A[i*rsa + k*csa] , B[k*rsb + j*csb]));
                    }
                    if( (Beta.real == ZERO.real) || (Beta.imag == ZERO.imag) )
                    {
                        C[i*rsc + j*csc] = mulc<T>(Alpha , tmp);
                    }
                    else
                    {
                        C[i*rsc + j*csc] = addc<T>(mulc<T>(Beta , C[i*rsc + j*csc]) , mulc<T>(Alpha , tmp));
                    }
                }
            }
        }
    }
    else
    {
        //* C := beta * C_orig + alpha * transb(B) * transa(A)
        if( UPPER )
        {
            for( i = 0 ; i < M ; i++ )
            {
                for( j = 0 ; j < N ; j++ )
                {
                    tmp = ZERO ;
                    for( k = 0 ; k <= j ; k++ )
                    {
                        tmp = addc<T>(tmp , mulc<T>(B[i*rsb + k*csb] , A[k*rsa + j*csa]));
                    }
                    if( (Beta.real == ZERO.real) || (Beta.imag == ZERO.imag) )
                    {
                        C[i*rsc + j*csc] = mulc<T>(Alpha , tmp);
                    }
                    else
                    {
                        C[i*rsc + j*csc] = addc<T>(mulc<T>(Beta , C[i*rsc + j*csc]) , mulc<T>(Alpha , tmp));
                    }
                }
            }
        }
        else
        {
            for( i = 0 ; i < M ; i++ )
            {
                for( j = 0 ; j < N ; j++ )
                {
                    tmp = ZERO ;
                    for( k = j ; k < N ; k++ )
                    {
                        tmp = addc<T>(tmp , mulc<T>(B[i*rsb + k*csb] , A[k*rsa + j*csa]));
                    }
                    if( (Beta.real == ZERO.real) || (Beta.imag == ZERO.imag) )
                    {
                        C[i*rsc + j*csc] = mulc<T>(Alpha , tmp);
                    }
                    else
                    {
                        C[i*rsc + j*csc] = addc<T>(mulc<T>(Beta , C[i*rsc + j*csc]) , mulc<T>(Alpha , tmp));
                    }
                }
            }
        }
    }
    return;
}

double libblis_test_itrmm3_check
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

    num_t dt       = bli_obj_dt( a );
    uplo_t uploa   = bli_obj_uplo( a );
    diag_t diaga   = bli_obj_diag( a );
    dim_t M        = bli_obj_length( c );
    dim_t N        = bli_obj_width( c );
    bool conja     = bli_obj_has_conj( a );
    bool conjb     = bli_obj_has_conj( b );
    trans_t transa = bli_obj_onlytrans_status( a );
    trans_t transb = bli_obj_onlytrans_status( b );
    dim_t  rsa, csa;
    dim_t  rsb, csb;
    dim_t  rsc, csc;
    double resid   = 0.0;

    if( bli_obj_row_stride( c ) == 1 )
    {
      rsa = transa ? bli_obj_col_stride( a ) : bli_obj_row_stride( a ) ;
      csa = transa ? bli_obj_row_stride( a ) : bli_obj_col_stride( a ) ;
      rsb = transb ? bli_obj_col_stride( b ) : bli_obj_row_stride( b ) ;
      csb = transb ? bli_obj_row_stride( b ) : bli_obj_col_stride( b ) ;
      rsc = 1;
      csc = bli_obj_col_stride( c_orig );
    }
    else
    {
      rsa = transa ? bli_obj_col_stride( a ) : bli_obj_row_stride( a ) ;
      csa = transa ? bli_obj_row_stride( a ) : bli_obj_col_stride( a ) ;
      rsb = transb ? bli_obj_col_stride( b ) : bli_obj_row_stride( b ) ;
      csb = transb ? bli_obj_row_stride( b ) : bli_obj_col_stride( b ) ;
      rsc = bli_obj_row_stride( c_orig );
      csc = 1 ;
    }

    if( transa ) {
      if( bli_obj_is_upper_or_lower( a ) ) {
        bli_obj_toggle_uplo( a );
      }
      uploa   = bli_obj_uplo( a );
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
            libblis_itrmm3_check<float>(side, uploa, diaga, M, N, *Alpha,
                                A, rsa, csa, B, rsb, csb, *Beta, C, rsc, csc );
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
            libblis_itrmm3_check<double>(side, uploa, diaga, M, N, *Alpha,
                                A, rsa, csa, B, rsb, csb, *Beta, C, rsc, csc );
            resid = computediffrm(M, N, CC, C, rsc, csc);
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
            libblis_ictrmm3_check<scomplex, float>(side, uploa, diaga, M, N,
            *Alpha, A, rsa, csa, conja, B, rsb, csb, conjb, *Beta, C, rsc, csc );
            resid = computediffim(M, N, CC, C, rsc, csc);
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
            libblis_ictrmm3_check<dcomplex, double>(side, uploa, diaga, M, N,
            *Alpha, A, rsa, csa, conja, B, rsb, csb, conjb, *Beta, C, rsc, csc );
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

double libblis_check_nan_trmm3(obj_t* c, num_t dt ) {
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