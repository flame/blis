#include "blis_test.h"
#include "blis_utils.h"
#include "test_gemv.h"

using namespace std;

//*  ==========================================================================
//*> GEMV performs one of the matrix-vector operations
//*>    y := alpha*A*x + beta*y,   or   y := alpha*A**T*x + beta*y,   or
//*>    y := alpha*A**H*x + beta*y,
//*  ==========================================================================

template <typename T, typename U>
void libblis_igemv_check(trans_t transA , dim_t M, dim_t N, T* alpha, T* A,
         dim_t rsa, dim_t csa, T* X, dim_t incx, T* beta, T* Y, dim_t incy) {
  T ONE, ZERO;
  T temp;
  dim_t i, ix, iy, j, jx, jy, kx, ky, lenx, leny;
  bool NOTRANSA;

  ONE = 1.0 ;
  ZERO = 0.0 ;
  T Alpha = alpha[0];
  T Beta  = beta[0];

  if (((M == 0) || (N == 0)) ||
    ((Alpha == ZERO) && (Beta == ONE))) {
      return;
  }

  NOTRANSA = ((transA == BLIS_NO_TRANSPOSE) || (transA == BLIS_CONJ_NO_TRANSPOSE));

  /*   Set  lenx  and  leny, the lengths of the vectors x and y,
  and set up the start points in  X  and  Y. */
  if (NOTRANSA) {
    lenx = N;
    leny = M;
  }
  else {
    lenx = M;
    leny = N;
  }

  if (incx > 0) {
    kx = 0;
  }
  else {
    kx = 1 - (lenx - 1) * incx;
  }

  if (incy > 0) {
    ky = 0;
  }
  else {
    ky = 1 - (leny - 1) * incy;
  }

  //*     Start the operations. Here, the elements of A are
  //*     accessed sequentially with one pass through A.
  //*     First form  y := beta*y.
  if (Beta != ONE) {
    iy = ky;
    if (Beta == ZERO) {
      for(i = 0 ; i < leny ; i++) {
        Y[iy] = ZERO;
        iy = iy + incy;
      }
    }
    else {
      for(i = 0 ; i < leny ; i++) {
        Y[iy] = Beta*Y[iy];
        iy = iy + incy;
      }
    }
  }

  if(Alpha == ZERO)
    return;

  if(NOTRANSA) {
    /*  Form  y := alpha*A*x + y.*/
    jx = kx;
    for(j = 0 ; j < N ; j++) {
      temp = Alpha*X[jx];
      iy = ky;
      for(i = 0 ; i < M ; i++) {
        Y[iy] = Y[iy] + temp * A[i*rsa + j*csa];
        iy = iy + incy;
      }
      jx = jx + incx;
    }
  }
  else {
    //*        Form  y := alpha*A**T*x + y.
    jy = ky;
    for(i = 0 ; i < N ; i++) {
      temp = ZERO;
      ix = kx;
      for(j = 0 ; j < M ; j++) {
        temp = temp + A[i*rsa + j*csa] * X[ix];
        ix = ix + incx;
      }
      Y[jy] = Y[jy] + Alpha*temp;
      jy = jy + incy;
    }
  }
  return;
}

template <typename T, typename U>
void libblis_icgemv_check(trans_t transA , dim_t M, dim_t N, T* alpha, T* A,
         dim_t rsa, dim_t csa, bool conja, T* X, dim_t incx, T* beta, T* Y,
         dim_t incy, bool  conjx) {
  T ONE;
  T ZERO;
  T temp;
  dim_t i, ix, iy, j, jx, jy, kx, ky, lenx, leny;
  bool NOTRANSA;

  ONE  = {1.0 , 0.0};
  ZERO = {0.0 , 0.0};
  T Alpha = *alpha;
  T Beta  = *beta;

  if (((M == 0) || (N == 0)) ||
    ((Alpha.real == ZERO.real) && (Beta.real == ONE.real))) {
    return ;
  }

  NOTRANSA = ((transA == BLIS_NO_TRANSPOSE) || (transA == BLIS_CONJ_NO_TRANSPOSE));

  /*     Set  lenx  and  leny, the lengths of the vectors x and y,
       and set up the start points in  X  and  Y. */
  if(NOTRANSA) {
    lenx = N;
    leny = M;
  }
  else {
    lenx = M;
    leny = N;
  }

  if (incx > 0) {
    kx = 0;
  }
  else{
    kx = 1 - (lenx - 1) * incx;
  }

  if (incy > 0) {
    ky = 0;
  }
  else {
    ky = 1 - (leny - 1)*incy;
  }

  if (Alpha.real == ZERO.real)
    return;

  if( conja ) {
    for(i = 0; i < leny ; i++) {
      for(j = 0 ; j < lenx ; j++) {
        A[i*rsa + j*csa] = conjugate<T>(A[i*rsa + j*csa]);
      }
    }
  }

  if(conjx) {
    ix = kx;
    for(i = 0 ; i < lenx ; i++) {
      X[ix] = conjugate<T>(X[ix]);
      ix = ix + incx;
    }
  }

  /* First form  y := beta*y. */
  if (Beta.real != ONE.real) {
    iy = ky;
    if (Beta.real == ZERO.real) {
      for(i = 0; i < leny ; i++) {
        Y[iy] = ZERO;
        iy = iy + incy;
      }
    }
    else {
      for(i = 0 ; i < leny ; i++) {
        Y[iy] = mulc<T>(Beta , Y[iy]);
        iy = iy + incy;
      }
    }
  }

  if (NOTRANSA) {
    /* Form  y := alpha*A*x + y. */
    jx = kx;
    for(j = 0; j < N ; j++) {
      temp = mulc<T>(Alpha , X[jx]);
      iy = ky;
      for(i = 0; i < M ; i++) {
        Y[iy] = addc<T>(Y[iy] , mulc<T>(temp , A[i*rsa + j*csa]));
        iy = iy + incy;
      }
      jx = jx + incx;
    }
  }
  else {
    /* Form  y := alpha*A**T*x + y  or  y := alpha*A**H*x + y. */
    jy = ky;
    for(i = 0 ; i < N ; i++) {
      temp = ZERO;
      ix = kx;
      for(j = 0 ; j < M ; j++) {
        temp = addc<T>(temp , mulc<T>(A[i*rsa + j*csa] , X[ix]));
        ix = ix + incx;
      }
      Y[jy] = addc<T>(Y[jy] , mulc<T>(Alpha , temp));
      jy = jy + incy;
    }
  }
  return;
}

double libblis_test_igemv_check(
  obj_t*  alpha,
  obj_t*  a,
  obj_t*  x,
  obj_t*  beta,
  obj_t*  y,
  obj_t*  y_orig,
  num_t   dt
){
  double resid = 0.0;
  f77_int  rsa, csa;
  trans_t transA = bli_obj_onlytrans_status( a );
  f77_int M      = transA ? bli_obj_vector_dim( x )     : bli_obj_vector_dim( y_orig );
  f77_int N      = transA ? bli_obj_vector_dim( y_orig ): bli_obj_vector_dim( x );
  f77_int incx   = bli_obj_vector_inc( x );
  f77_int incy   = bli_obj_vector_inc( y_orig );
  f77_int len    = bli_obj_vector_dim( y_orig );
  bool cfx       = bli_obj_has_conj( x );
  bool cfa       = bli_obj_has_conj( a );
  bool sf        = bli_obj_is_col_stored( a );

  if( sf ) {
    rsa = bli_obj_has_trans( a ) ? bli_obj_col_stride( a ) : bli_obj_row_stride( a ) ;
    csa = bli_obj_has_trans( a ) ? bli_obj_row_stride( a ) : bli_obj_col_stride( a ) ;
  } else {
    rsa = bli_obj_has_trans( a ) ? bli_obj_col_stride( a ) : bli_obj_row_stride( a ) ;
    csa = bli_obj_has_trans( a ) ? bli_obj_row_stride( a ) : bli_obj_col_stride( a ) ;

    if(transA == BLIS_NO_TRANSPOSE)               transA = BLIS_TRANSPOSE;
    else if(transA == BLIS_TRANSPOSE)             transA = BLIS_NO_TRANSPOSE;
    else if ( transA == BLIS_CONJ_NO_TRANSPOSE)   transA = BLIS_CONJ_TRANSPOSE;
    else /*if ( transa == BLIS_CONJ_TRANSPOSE )*/ transA = BLIS_CONJ_NO_TRANSPOSE;
    M = M ^ N;
    N = M ^ N;
    M = M ^ N;
  }

  switch( dt )  {
    case BLIS_FLOAT :
    {
      float*   Alpha    = (float*) bli_obj_buffer( alpha );
      float*   A        = (float*) bli_obj_buffer( a );
      float*   Beta     = (float*) bli_obj_buffer( beta );
      float*   X        = (float*) bli_obj_buffer( x );
      float*   Y        = (float*) bli_obj_buffer( y_orig );
      libblis_igemv_check<float, int32_t>(transA, M,  N, Alpha, A, rsa, csa,
                                                 X, incx, Beta, Y, incy );
      float* YY         = (float*) bli_obj_buffer( y );
      resid = computediffrv(len, incy, YY, Y);
      break;
    }
    case BLIS_DOUBLE :
    {
      double*   Alpha   = (double*) bli_obj_buffer( alpha );
      double*   A       = (double*) bli_obj_buffer( a );
      double*   Beta    = (double*) bli_obj_buffer( beta );
      double*   X       = (double*) bli_obj_buffer( x );
      double*   Y       = (double*) bli_obj_buffer( y_orig );
      libblis_igemv_check<double, int64_t>(transA, M,  N, Alpha, A, rsa, csa,
                                                 X, incx, Beta, Y, incy );
      double*   YY        = (double*) bli_obj_buffer( y );
      resid = computediffrv(len, incy, YY, Y);
      break;
    }
    case BLIS_SCOMPLEX :
    {
      scomplex*   Alpha = (scomplex*) bli_obj_buffer( alpha );
      scomplex*   A     = (scomplex*) bli_obj_buffer( a );
      scomplex*   Beta  = (scomplex*) bli_obj_buffer( beta );
      scomplex*   X     = (scomplex*) bli_obj_buffer( x );
      scomplex*   Y     = (scomplex*) bli_obj_buffer( y_orig );
      libblis_icgemv_check<scomplex, int32_t>(transA, M,  N, Alpha, A, rsa,
                                       csa, cfa, X, incx, Beta, Y, incy, cfx );
      scomplex*   YY        = (scomplex*) bli_obj_buffer( y );
      resid = computediffiv(len, incy, YY, Y);
      break;
    }
    case BLIS_DCOMPLEX :
    {
      dcomplex*   Alpha = (dcomplex*) bli_obj_buffer( alpha );
      dcomplex*   A     = (dcomplex*) bli_obj_buffer( a );
      dcomplex*   Beta  = (dcomplex*) bli_obj_buffer( beta );
      dcomplex*   X     = (dcomplex*) bli_obj_buffer( x );
      dcomplex*   Y     = (dcomplex*) bli_obj_buffer( y_orig );
      libblis_icgemv_check<dcomplex, int64_t>(transA, M,  N, Alpha, A, rsa,
                                       csa, cfa, X, incx, Beta, Y, incy, cfx );
      dcomplex*   YY        = (dcomplex*) bli_obj_buffer( y );
      resid = computediffiv(len, incy, YY, Y);
      break;
    }
    default :
      bli_check_error_code( BLIS_INVALID_DATATYPE );
  }

  return resid;
}

