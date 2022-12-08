#include "blis_test.h"
#include "blis_utils.h"
#include "test_trmm.h"

using namespace std;

//*  ==========================================================================
//*> TRMM  performs one of the matrix-matrix operations
//*>    B := alpha*op( A )*B,   or   B := alpha*B*op( A )
//*> where  alpha  is a scalar,  B  is an m by n matrix,  A  is a unit, or
//*> non-unit,  upper or lower triangular matrix  and  op( A )  is one  of
//*>    op( A ) = A   or   op( A ) = A**T   or   op( A ) = A**H.
//*  ==========================================================================

template <typename T, typename U>
void libblis_itrmm_check(side_t side, uplo_t uplo, trans_t transa,
  diag_t diag, dim_t M, dim_t N, T Alpha, T* A, dim_t rsa, dim_t csa,
  T* B, dim_t rsb, dim_t csb)
{
    T tmp;
    int i, j, k;
    bool LSIDE, NOUNIT, UPPER, NOTRANSA;

    T ONE  = 1.0;
    T ZERO = 0.0;

    LSIDE    = ( side == BLIS_LEFT );
    NOTRANSA = ( transa == BLIS_NO_TRANSPOSE );
    NOUNIT   = ( diag == BLIS_NONUNIT_DIAG );
    UPPER    = ( uplo == BLIS_UPPER );

    if( M == 0 || N == 0 )
      return;

    if( Alpha == ZERO )
    {
        for( j = 0 ; j < N ; j++ )
        {
            for( i = 0 ; i < M ; i++ )
            {
                B[i*rsb + j*csb] = ZERO;
            }
        }
        return;
    }


    if( LSIDE )
    {
        if( NOTRANSA )
        {
            //* Form  B := alpha*A*B.
            if( UPPER )
            {
                for( j = 0 ; j < N ; j++ )
                {
                    for( k = 0 ; k < M ; k++ )
                    {
                        if( B[k*rsb + j*csb] != ZERO )
                        {
                            tmp = Alpha*B[k*rsb + j*csb];
                            for( i = 0 ;  i < k ; i++ )
                            {
                                B[i*rsb + j*csb] = B[i*rsb + j*csb] + tmp*A[i*rsa + k*csa];
                            }
                            if( NOUNIT )
                                tmp = tmp*A[k*rsa + k*csa];
                            B[k*rsb + j*csb] = tmp;
                        }
                    }
                }
            }
            else
            {
                for( j = 0; j < N ; j++ )
                {
                    for( k = (M-1) ; k >= 0 ; k-- )
                    {
                        if( B[k*rsb + j*csb] != ZERO )
                        {
                            tmp = Alpha*B[k*rsb + j*csb];
                            B[k*rsb + j*csb] = tmp;
                            if( NOUNIT )
                                B[k*rsb + j*csb] = B[k*rsb + j*csb]*A[k*rsa + k*csa];
                            for( i = (k+1) ; i < M ; i++ )
                            {
                                B[i*rsb + j*csb] = B[i*rsb + j*csb] + (tmp * A[i*rsa + k*csa]);
                            }
                        }
                    }
                }
            }
        }
        else
        {
            //*           Form  B := alpha*A**T*B.
            if( UPPER )
            {
                for( j = 0; j < N ; j++ )
                {
                    for( i = (M-1) ; i >= 0 ; i-- )
                    {
                        tmp = B[i*rsb + j*csb];
                        if( NOUNIT )
                            tmp = tmp*A[i*rsa + i*csa];
                        for( k = 0 ; k < i ; k++ )
                        {
                            tmp = tmp + A[k*rsa + i*csa]*B[k*rsb + j*csb];
                        }
                        B[i*rsb + j*csb] = Alpha*tmp;
                    }
                }
            }
            else
            {
                for( j = 0; j < N ; j++ )
                {
                    for( i = 0 ; i < M ; i++ )
                    {
                        tmp = B[i*rsb + j*csb];
                        if( NOUNIT )
                            tmp = tmp*A[i*rsa + i*csa];
                        for( k =(i+1) ; k < M ; k++ )
                        {
                            tmp = tmp + A[k*rsa + i*csa]*B[k*rsb + j*csb];
                        }
                        B[i*rsb + j*csb] = Alpha*tmp;
                    }
                }
            }
        }
    }
    else
    {
        if( NOTRANSA )
        {
            //*  Form  B := alpha*B*A.
            if( UPPER )
            {
                for( j = (N-1) ; j >= 0 ; j-- )
                {
                    tmp = Alpha;
                    if( NOUNIT )
                        tmp = tmp*A[j*rsa + j*csa];
                    for( i = 0 ; i < M ; i++ )
                    {
                        B[i*rsb + j*csb] = tmp*B[i*rsb + j*csb];
                    }
                    for( k = 0 ; k < j ; k++ )
                    {
                        if( A[k*rsa + j*csa] != ZERO )
                        {
                            tmp = Alpha*A[k*rsa + j*csa];
                            for( i = 0 ; i < M ; i++ )
                            {
                                B[i*rsb + j*csb] = B[i*rsb + j*csb] + tmp*B[i*rsb + k*csb];
                            }
                        }
                    }
                }
            }
            else
            {
                for( j = 0; j < N ; j++ )
                {
                    tmp = Alpha;
                    if( NOUNIT )
                        tmp = tmp*A[j*rsa + j*csa];
                    for( i = 0 ; i < M ; i++ )
                    {
                        B[i*rsb + j*csb] = tmp*B[i*rsb + j*csb];
                    }
                    for( k =(j+1) ; k < N ; k++ )
                    {
                        if( A[k*rsa + j*csa] != ZERO )
                        {
                            tmp = Alpha*A[k*rsa + j*csa];
                            for( i = 0 ; i < M ; i++ )
                            {
                                B[i*rsb + j*csb] = B[i*rsb + j*csb] + tmp*B[i*rsb + k*csb];
                            }
                        }
                    }
                }
            }
        }
        else
        {
            //* Form  B := alpha*B*A**T.
            if( UPPER )
            {
                for( k = 0 ; k < N ; k++ )
                {
                    for( j = 0 ; j < k ; j++ )
                    {
                        if( A[j*rsa + k*csa] != ZERO )
                        {
                            tmp = Alpha*A[j*rsa + k*csa];
                            for( i = 0 ; i < M ; i++ )
                            {
                                B[i*rsb + j*csb] = B[i*rsb + j*csb] + tmp*B[i*rsb + k*csb];
                            }
                        }
                    }
                    tmp = Alpha;
                    if( NOUNIT )
                        tmp = tmp*A[k*rsa + k*csa];
                    if( tmp != ONE )
                    {
                        for( i = 0 ; i < M ; i++ )
                        {
                            B[i*rsb + k*csb] = tmp*B[i*rsb + k*csb];
                        }
                    }
                }
            }
            else
            {
                for( k = (N-1) ; k >= 0 ; k-- )
                {
                    for( j = (k+1) ; j < N ; j++ )
                    {
                        if( A[j*rsa + k*csa] != ZERO )
                        {
                            tmp = Alpha*A[j*rsa + k*csa];
                            for( i = 0 ; i < M ; i++ )
                            {
                                B[i*rsb + j*csb] = B[i*rsb + j*csb] + tmp*B[i*rsb + k*csb];
                            }
                        }
                    }
                    tmp = Alpha;
                    if( NOUNIT )
                        tmp = tmp*A[k*rsa + k*csa];
                    if( tmp != ONE )
                    {
                        for( i = 0 ; i < M ; i++ )
                        {
                            B[i*rsb + k*csb] = tmp*B[i*rsb + k*csb];
                        }
                    }
                }
            }
        }
    }
    return;
}

template <typename T, typename U>
void libblis_ictrmm_check(side_t side, uplo_t uplo, trans_t transa,
  diag_t diag, dim_t M, dim_t N, T Alpha, T* A, dim_t rsa, dim_t csa,
  bool conja, T* B, dim_t rsb, dim_t csb)
{
    T tmp;
    int i, j, k;
    bool LSIDE, NOTRANSA, NOUNIT, UPPER;

    T ONE  = { 1.0 , 0.0 };
    T ZERO = { 0.0 , 0.0 };

    //*     Test the input parameters.
    LSIDE    = ( side == BLIS_LEFT );
    NOTRANSA = ( transa == BLIS_NO_TRANSPOSE );
    NOUNIT   = ( diag == BLIS_NONUNIT_DIAG );
    UPPER    = ( uplo == BLIS_UPPER );

    if( M == 0 || N == 0 )
      return;

    if( (Alpha.real == ZERO.real) || (Alpha.imag == ZERO.imag) )
    {
        for( j = 0 ; j < N ; j++ )
        {
            for( i = 0 ; i < M ; i++ )
            {
                B[i*rsb + j*csb] = ZERO;
            }
        }
        return;
    }

    if( conja )
    {
        dim_t dim;
        if (LSIDE)         dim = M;
        else               dim = N;
        for( i = 0 ; i < dim ; i++ )
        {
            for( j = 0 ; j < dim ; j++ )
            {
                A[i*rsa + j*csa] = conjugate<T>(A[i*rsa + j*csa]);
            }
        }
    }

    if( LSIDE )
    {
        if( NOTRANSA )
        {
            //* Form  B := alpha*A*B.
            if( UPPER )
            {
                for( j = 0 ; j < N ; j++ )
                {
                    for( k = 0 ; k < M ; k++ )
                    {
                        if( (B[k*rsb + j*csb].real != ZERO.real) || (B[k*rsb + j*csb].imag != ZERO.imag) )
                        {
                            tmp = mulc<T>(Alpha , B[k*rsb + j*csb]);
                            for( i = 0 ;  i < k ; i++ )
                            {
                                B[i*rsb + j*csb] = addc<T>(B[i*rsb + j*csb] , mulc<T>(tmp , A[i*rsa + k*csa]));
                            }
                            if( NOUNIT )
                                tmp = mulc<T>(tmp , A[k*rsa + k*csa]);
                            B[k*rsb + j*csb] = tmp;
                        }
                    }
                }
            }
            else
            {
                for( j = 0 ; j < N ; j++ )
                {
                    for( k = (M-1) ; k >= 0 ; k-- )
                    {
                        if( (B[k*rsb + j*csb].real != ZERO.real) || (B[k*rsb + j*csb].imag != ZERO.imag) )
                        {
                            tmp = mulc<T>(Alpha , B[k*rsb + j*csb]);
                            B[k*rsb + j*csb] = tmp;
                            if( NOUNIT )
                                B[k*rsb + j*csb] = mulc<T>(B[k*rsb + j*csb] , A[k*rsa + k*csa]);
                            for( i = (k+1) ; i < M ; i++ )
                            {
                                B[i*rsb + j*csb] = addc<T>(B[i*rsb + j*csb] , mulc<T>(tmp , A[i*rsa + k*csa]));
                            }
                        }
                    }
                }
            }
        }
        else
        {
            //* Form  B := alpha*A**T*B   or   B := alpha*A**H*B.
            if( UPPER )
            {
                for( j = 0; j < N ; j++ )
                {
                    for( i = (M-1) ; i >= 0 ; i-- )
                    {
                        tmp = B[i*rsb + j*csb];
                        if( NOUNIT )
                            tmp = mulc<T>(tmp , A[i*rsa + i*csa]);
                        for( k = 0 ; k < i ; k++ )
                        {
                            tmp = addc<T>(tmp , mulc<T>(A[k*rsa + i*csa] , B[k*rsb + j*csb]));
                        }
                        B[i*rsb + j*csb] = mulc<T>(Alpha , tmp);
                    }
                }
            }
            else
            {
                for( j = 0; j < N ; j++ )
                {
                    for( i = 0 ; i < M ; i++ )
                    {
                        tmp = B[i*rsb + j*csb];
                        if( NOUNIT )
                            tmp = mulc<T>(tmp , A[i*rsa + i*csa]);
                        for( k =(i+1) ; k < M ; k++ )
                        {
                            tmp = addc<T>(tmp , mulc<T>(A[k*rsa + i*csa] , B[k*rsb + j*csb]));
                        }
                        B[i*rsb + j*csb] = mulc<T>(Alpha , tmp);
                    }
                }
            }
        }
    }
    else
    {
        if( NOTRANSA )
        {
            //*  Form  B := alpha*B*A.
            if( UPPER )
            {
                for( j = (N-1) ; j >= 0 ; j-- )
                {
                    tmp = Alpha;
                    if( NOUNIT )
                        tmp = mulc<T>(tmp , A[j*rsa + j*csa]);
                    for( i = 0 ; i < M ; i++ )
                    {
                        B[i*rsb + j*csb] = mulc<T>(tmp , B[i*rsb + j*csb]);
                    }
                    for( k = 0 ; k < j ; k++ )
                    {
                        if( (A[k*rsa + j*csa].real != ZERO.real)||(A[k*rsa + j*csa].imag != ZERO.imag) )
                        {
                            tmp = mulc<T>(Alpha , A[k*rsa + j*csa]);
                            for( i = 0 ; i < M ; i++ )
                            {
                                B[i*rsb + j*csb] = addc<T>(B[i*rsb + j*csb] , mulc<T>(tmp , B[i*rsb + k*csb]));
                            }
                        }
                    }
                }
            }
            else
            {
                for( j = 0; j < N ; j++ )
                {
                    tmp = Alpha;
                    if( NOUNIT )
                        tmp = mulc<T>(tmp , A[j*rsa + j*csa]);
                    for( i = 0 ; i < M ; i++ )
                    {
                        B[i*rsb + j*csb] = mulc<T>(tmp , B[i*rsb + j*csb]);
                    }
                    for( k =(j+1) ; k < N ; k++ )
                    {
                        if( (A[k*rsa + j*csa].real != ZERO.real)||(A[k*rsa + j*csa].imag != ZERO.imag) )
                        {
                            tmp = mulc<T>(Alpha , A[k*rsa + j*csa]);
                            for( i = 0 ; i < M ; i++ )
                            {
                                B[i*rsb + j*csb] = addc<T>(B[i*rsb + j*csb] , mulc<T>(tmp , B[i*rsb + k*csb]));
                            }
                        }
                    }
                }
            }
        }
        else
        {
            //* Form  B := alpha*B*A**T   or   B := alpha*B*A**H.
            if( UPPER )
            {
                for( k = 0 ; k < N ; k++ )
                {
                    for( j = 0 ; j < k ; j++ )
                    {
                        if( (A[j*rsa + k*csa].real != ZERO.real)||(A[j*rsa + k*csa].imag != ZERO.imag) )
                        {
                            tmp = mulc<T>(Alpha , A[j*rsa + k*csa]);
                            for( i = 0 ; i < M ; i++ )
                            {
                                B[i*rsb + j*csb] = addc<T>(B[i*rsb + j*csb] , mulc<T>(tmp , B[i*rsb + k*csb]));
                            }
                        }
                    }
                    tmp = Alpha;
                    if( NOUNIT )
                        tmp = mulc<T>(tmp , A[k*rsa + k*csa]);
                    if( (tmp.real != ONE.real) || (tmp.imag != ONE.imag) )
                    {
                        for( i = 0 ; i < M ; i++ )
                        {
                            B[i*rsb + k*csb] = mulc<T>(tmp , B[i*rsb + k*csb]);
                        }
                    }
                }
            }
            else
            {
                for( k = (N-1) ; k >= 0 ; k-- )
                {
                    for( j = (k+1) ; j < N ; j++ )
                    {
                        if( (A[j*rsa + k*csa].real != ZERO.real)||(A[j*rsa + k*csa].imag != ZERO.imag) )
                        {
                            tmp = mulc<T>(Alpha , A[j*rsa + k*csa]);
                            for( i = 0 ; i < M ; i++ )
                            {
                                B[i*rsb + j*csb] = addc<T>(B[i*rsb + j*csb] , mulc<T>(tmp , B[i*rsb + k*csb]));
                            }
                        }
                    }
                    tmp = Alpha;
                    if( NOUNIT )
                        tmp = mulc<T>(tmp , A[k*rsa + k*csa]);
                    if( (tmp.real != ONE.real) || (tmp.imag != ONE.imag) )
                    {
                        for( i = 0 ; i < M ; i++ )
                        {
                            B[i*rsb + k*csb] = mulc<T>(tmp , B[i*rsb + k*csb]);
                        }
                    }
                }
            }
        }
    }
    return;
}

double libblis_test_itrmm_check(
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
      libblis_itrmm_check<float, int32_t>(side, uploa, transa,
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
      libblis_itrmm_check<double, int64_t>(side, uploa, transa,
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
      libblis_ictrmm_check<scomplex, float>(side, uploa, transa,
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
      libblis_ictrmm_check<dcomplex, double>(side, uploa, transa,
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

double libblis_check_nan_trmm(obj_t* b, num_t dt ) {
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

