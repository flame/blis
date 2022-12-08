#include "blis_test.h"
#include "blis_utils.h"
#include "test_gemm.h"

using namespace std;

//*  ==========================================================================
//*> GEMM  performs one of the matrix-matrix operations
//*>    C := alpha*op( A )*op( B ) + beta*C,
//*> where  op( X ) is one of
//*>    op( X ) = X   or   op( X ) = X**T   or   op( X ) = X**H,
//*> alpha and beta are scalars, and A, B and C are matrices, with op( A )
//*> an m by k matrix,  op( B )  a  k by n matrix and  C an m by n matrix.
/*
Reference GEMM implemenation C = C*Beta + Alpha*A*B
Row major A=mxk , B=kxn and C=mxn , lda=rsa=k, csa=1, ldb=rsb=n,csb=1, ldc=rsc=n,csc=1
Col major A=mxk , B=kxn and C=mxn , rsa=1, lda=csa=m, rsb=1,ldb=csb=k, rsc=1,ldc=csc=m
*/
//*  ==========================================================================

template <typename T, typename U>
void libblis_igemm_check(dim_t M, dim_t N, dim_t K, T *alpha, T *A,
  dim_t rsa, dim_t csa, T *B, dim_t rsb, dim_t csb, T* beta,
  T *C, dim_t rsc, dim_t csc){

  T Alpha = alpha[0];
  T Beta  = beta[0];
  int  i,j,k;

  if(( Alpha != 0.)  && ( Beta != 0. )) {
    for( i = 0 ; i < M ; i++ ) {
      for( j = 0 ; j < N ; j++ ) {
       T sum = 0.0;
        for( k = 0 ; k < K ; k++ ) {
          sum += A[i*rsa + k*csa] * B[k*rsb + j*csb];
        }
        sum = ((Beta * C[i*rsc + j*csc]) + (Alpha * sum));
        C[i*rsc + j*csc] = sum;
      }
    }
  }
  else if(( Alpha != 0.)  && ( Beta == 0. )) {
    for( i = 0 ; i < M ; i++ ) {
      for( j = 0 ; j < N ; j++ ) {
        T sum = 0.0;
        for( k = 0 ; k < K ; k++ ) {
          sum += A[i*rsa + k*csa] * B[k*rsb + j*csb];
        }
        sum = (Alpha * sum);
        C[i*rsc + j*csc] = sum;
      }
    }
  }
  else if(( Alpha == 0.)  && ( Beta != 0. )) {
    for( i = 0 ; i < M ; i++ ) {
      for( j = 0 ; j < N ; j++ ) {
        T sum = (Beta * C[ i*rsc + j*csc ]);
        C[i*rsc + j*csc] = sum;
      }
    }
  }
  else /*if(( Alpha == 0.) && ( Beta == 0. ))*/ {
    //
  }

  return;
}

template <typename T, typename U>
void libblis_icgemm_check(dim_t M, dim_t N, dim_t K, T *alpha,
  T *A, dim_t rsa, dim_t csa, bool conja, T *B, dim_t rsb, dim_t csb,
  bool conjb, T* beta, T *C, dim_t rsc, dim_t csc){

  T Alpha = *alpha;
  T Beta  = *beta;
  int  i,j,k;

  if(conja) {
    for( i = 0 ; i < M ; i++ ) {
      for( k = 0 ; k < K ; k++ ) {
        A[i*rsa + k*csa] = conjugate<T>(A[i*rsa + k*csa]);
      }
    }
  }
  if(conjb) {
    for( k = 0 ; k < K ; k++ ) {
      for( j = 0 ; j < N ; j++ ) {
        B[k*rsb + j*csb] = conjugate<T>(B[k*rsb + j*csb]);
      }
    }
  }

  if((Alpha.real != 0.)  && (Beta.real != 0.)) {
    for( i = 0 ; i < M ; i++ ) {
      for( j = 0 ; j < N ; j++ ) {
        T sum = {0.0, 0.0};
        for( k = 0 ; k < K ; k++ ) {
          T aa = A[i*rsa + k*csa];
          T bb = B[k*rsb + j*csb];
          sum = addc<T>(sum , mulc<T>(aa , bb));
        }
        T xc = C[i*rsc + j*csc];
        sum = mulc<T>(Alpha,sum);
        xc  = mulc<T>(Beta,xc);
        C[i*rsc + j*csc] = addc<T>(xc , sum);
      }
    }
  }
  else if(( Alpha.real != 0.)  && ( Beta.real == 0. )) {
    for( i = 0 ; i < M ; i++ ) {
      for( j = 0 ; j < N ; j++ ) {
        T sum = {0.0, 0.0};
        for( k = 0 ; k < K ; k++ ) {
          T aa = A[i*rsa + k*csa];
          T bb = B[k*rsb + j*csb];
          sum = addc<T>(sum , mulc<T>(aa , bb));
        }
        sum = mulc<T>(Alpha,sum);
        C[i*rsc + j*csc] = sum;
      }
    }
  }
  else if(( Alpha.real == 0.)  && ( Beta.real != 0. )) {
    for( i = 0 ; i < M ; i++ ) {
      for( j = 0 ; j < N ; j++ ) {
        T sum ;
        T cc = C[i*rsc + j*csc];
        sum = mulc<T>(Beta,cc);
        C[i*rsc + j*csc] = sum;
      }
    }
  } else /*if(( Alpha == 0.) && ( Beta == 0. ))*/ {
    //
  }
  return ;
}

double libblis_test_igemm_check(
  test_params_t* params,
  obj_t*         alpha,
  obj_t*         a,
  obj_t*         b,
  obj_t*         beta,
  obj_t*         c,
  obj_t*         c_orig,
  num_t          dt
){
  dim_t  M = bli_obj_length( c_orig );
  dim_t  N = bli_obj_width( c_orig );
  dim_t  K = bli_obj_width_after_trans( a );
  dim_t  rsa, csa;
  dim_t  rsb, csb;
  dim_t  rsc, csc;
  bool conja = bli_obj_has_conj( a );
  bool conjb = bli_obj_has_conj( b );
  trans_t transA = bli_obj_onlytrans_status( a );
  trans_t transB = bli_obj_onlytrans_status( b );
  double resid   = 0.0;

  if( bli_obj_row_stride( c ) == 1 ) {
    rsa = transA ? bli_obj_col_stride( a ) : bli_obj_row_stride( a ) ;
    csa = transA ? bli_obj_row_stride( a ) : bli_obj_col_stride( a ) ;
    rsb = transB ? bli_obj_col_stride( b ) : bli_obj_row_stride( b ) ;
    csb = transB ? bli_obj_row_stride( b ) : bli_obj_col_stride( b ) ;
    rsc = 1;
    csc = bli_obj_col_stride( c_orig );
  } else {
    rsa = transA ? bli_obj_col_stride( a ) : bli_obj_row_stride( a ) ;
    csa = transA ? bli_obj_row_stride( a ) : bli_obj_col_stride( a ) ;
    rsb = transB ? bli_obj_col_stride( b ) : bli_obj_row_stride( b ) ;
    csb = transB ? bli_obj_row_stride( b ) : bli_obj_col_stride( b ) ;
    rsc = bli_obj_row_stride( c_orig );
    csc = 1 ;
  }

  switch( dt )  {
    case BLIS_FLOAT :
    {
      float*   Alpha    = (float*) bli_obj_buffer( alpha );
      float*   A        = (float*) bli_obj_buffer( a );
      float*   B        = (float*) bli_obj_buffer( b );
      float*   Beta     = (float*) bli_obj_buffer( beta );
      float*   C        = (float*) bli_obj_buffer( c_orig );
      float*   CC       = (float*) bli_obj_buffer( c );
      libblis_igemm_check<float, int32_t>(M, N, K, Alpha, A, rsa, csa,
                                       B, rsb, csb, Beta, C, rsc, csc);
      resid = computediffrm<float>(M, N, CC, C, rsc, csc);
      break;
    }
    case BLIS_DOUBLE :
    {
      double*   Alpha    = (double*) bli_obj_buffer( alpha );
      double*   A        = (double*) bli_obj_buffer( a );
      double*   B        = (double*) bli_obj_buffer( b );
      double*   Beta     = (double*) bli_obj_buffer( beta );
      double*   C        = (double*) bli_obj_buffer( c_orig );
      double*   CC       = (double*) bli_obj_buffer( c );
      libblis_igemm_check<double, int64_t>(M, N, K, Alpha, A, rsa, csa,
                                       B, rsb, csb, Beta, C, rsc, csc);
      resid = computediffrm<double>(M, N, CC, C, rsc, csc);
      break;
    }
    case BLIS_SCOMPLEX :
    {
      scomplex*   Alpha    = (scomplex*) bli_obj_buffer( alpha );
      scomplex*   A        = (scomplex*) bli_obj_buffer( a );
      scomplex*   B        = (scomplex*) bli_obj_buffer( b );
      scomplex*   Beta     = (scomplex*) bli_obj_buffer( beta );
      scomplex*   C        = (scomplex*) bli_obj_buffer( c_orig );
      scomplex*   CC       = (scomplex*) bli_obj_buffer( c );
      libblis_icgemm_check<scomplex, int32_t>(M, N, K, Alpha, A, rsa, csa,
                             conja, B, rsb, csb, conjb, Beta, C, rsc, csc);
      resid = computediffim<scomplex>(M, N, CC, C, rsc, csc);
      break;
    }
    case BLIS_DCOMPLEX :
    {
      dcomplex*   Alpha    = (dcomplex*) bli_obj_buffer( alpha );
      dcomplex*   A        = (dcomplex*) bli_obj_buffer( a );
      dcomplex*   B        = (dcomplex*) bli_obj_buffer( b );
      dcomplex*   Beta     = (dcomplex*) bli_obj_buffer( beta );
      dcomplex*   C        = (dcomplex*) bli_obj_buffer( c_orig );
      dcomplex*   CC       = (dcomplex*) bli_obj_buffer( c );
      libblis_icgemm_check<dcomplex, int64_t>(M, N, K, Alpha, A, rsa, csa,
                             conja, B, rsb, csb, conjb, Beta, C, rsc, csc);
      resid = computediffim<dcomplex>(M, N, CC, C, rsc, csc);
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
  T* C = (T*) bli_obj_buffer( c );
  double resid = 0.0;

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

template <typename T>
double libblis_check_nan_complex( dim_t rsc, dim_t csc, obj_t* c ) {

  dim_t  M = bli_obj_length( c );
  dim_t  N = bli_obj_width( c );
  dim_t  i,j;
  T* C = (T*) bli_obj_buffer( c );
  double resid = 0.0;

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

double libblis_check_nan_gemm(obj_t* c, num_t dt ) {
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

