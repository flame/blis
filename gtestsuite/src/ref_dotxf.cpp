#include "blis_test.h"
#include "blis_utils.h"
#include "test_dotxf.h"

using namespace std;

//*  ==========================================================================
//*> DOTXF performs vector operations
//*>    y := beta * y_orig + alpha * conjat(A^T) * conjx(x)
//*>    where A is an m x b matrix, and y and x are vectors.
//*>    The kernel is implemented as a fused series of calls to dotxv
//*>    where b is less than or equal to an implementation-dependent fusing
//*>    factor specific to dotxf
//*  ==========================================================================

template <typename T, typename U>
void libblis_idotxf_check(dim_t M, dim_t N, T* alpha, T* A, dim_t rsa,
                   dim_t csa, T* X, dim_t incx, T* beta, T* Y, dim_t incy) {

  dim_t i, j, ix, iy;
  T Alpha = alpha[0];
  T Beta  = beta[0];
  T temp;
  if((M == 0) || (N == 0)) {
    return;
  }

  iy = 0;
  for(j = 0 ; j < M ; j++) {
    ix = 0;
    temp = 0.0;
    for(i = 0 ; i < N ; i++) {
      temp += X[ix] * A[i*rsa + j*csa];
      ix = ix + incx;
    }
    temp = Alpha * temp;
    Y[iy] = (Y[iy] * Beta) + temp;
    iy = iy + incy;
  }

  return;
}

template <typename T, typename U>
void libblis_icdotxf_check(dim_t M, dim_t N, T* alpha, T* A, dim_t rsa,
dim_t csa, bool cfa, T* X, dim_t incx, bool cfx, T* beta, T* Y, dim_t incy ) {

  dim_t i, j, ix, iy;
  //T ONE  = {1.0 , 0.0};
  T ZERO = {0.0 , 0.0};
  T Alpha = *alpha;
  T Beta  = *beta;
  T temp;

  if((M == 0) || (N == 0)) {
    return;
  }

  if(cfx) {
    ix = 0;
    for(i = 0 ; i < N ; i++) {
      X[ix] = conjugate<T>(X[ix]);
      ix = ix + incx;
    }
  }

  if(cfa) {
    for(j = 0 ; j < M ; j++) {
      for(i = 0 ; i < N ; i++) {
        A[i*rsa + j*csa] = conjugate<T>(A[i*rsa + j*csa]);
      }
    }
  }

  iy = 0;
  for(j = 0 ; j < M ; j++) {
    ix = 0;
    temp = ZERO;
    for(i = 0 ; i < N ; i++) {
      temp = addc<T>(temp , mulc<T>(X[ix] , A[i*rsa + j*csa]));
      ix = ix + incx;
    }
    temp = mulc<T>(Alpha , temp);
    Y[iy] = addc<T>(temp , mulc<T>(Y[iy] , Beta));
    iy = iy + incy;
  }

  return;
}

double libblis_test_idotxf_check (
  test_params_t* params,
  obj_t*         alpha,
  obj_t*         a,
  obj_t*         x,
  obj_t*         beta,
  obj_t*         y,
  obj_t*         y_orig
) {
  num_t  dt    = bli_obj_dt( y );
  f77_int incx = bli_obj_vector_inc( x );
  f77_int incy = bli_obj_vector_inc( y );
  bool cfx     = bli_obj_has_conj( x );
  bool cfa     = bli_obj_has_conj( a );
  double resid = 0.0;

  //martix transpose
  dim_t  N     = bli_obj_vector_dim( x );
  dim_t  M     = bli_obj_vector_dim( y );
  f77_int rsa  = bli_obj_row_stride( a );
  f77_int csa  = bli_obj_col_stride( a );

  switch( dt )  {
    case BLIS_FLOAT :
    {
      float*   Alpha    = (float*) bli_obj_buffer( alpha );
      float*   A        = (float*) bli_obj_buffer( a );
      float*   X        = (float*) bli_obj_buffer( x );
      float*   Beta     = (float*) bli_obj_buffer( beta );
      float*   Y        = (float*) bli_obj_buffer( y_orig );
      float*   YY       = (float*) bli_obj_buffer( y );
      libblis_idotxf_check<float, int32_t>( M, N, Alpha, A, rsa, csa,
                                                    X, incx, Beta, Y, incy );
      resid = computediffrv(M, incy, YY, Y);
      break;
    }
    case BLIS_DOUBLE :
    {
      double*   Alpha   = (double*) bli_obj_buffer( alpha );
      double*   A       = (double*) bli_obj_buffer( a );
      double*   X       = (double*) bli_obj_buffer( x );
      double*   Beta    = (double*) bli_obj_buffer( beta );
      double*   Y       = (double*) bli_obj_buffer( y_orig );
      double*   YY      = (double*) bli_obj_buffer( y );
      libblis_idotxf_check<double, int64_t>( M, N, Alpha, A, rsa, csa,
                                                    X, incx, Beta, Y, incy );
      resid = computediffrv(M, incy, YY, Y);
      break;
    }
    case BLIS_SCOMPLEX :
    {
      scomplex*   Alpha = (scomplex*) bli_obj_buffer( alpha );
      scomplex*   A     = (scomplex*) bli_obj_buffer( a );
      scomplex*   X     = (scomplex*) bli_obj_buffer( x );
      scomplex*   Beta  = (scomplex*) bli_obj_buffer( beta );
      scomplex*   Y     = (scomplex*) bli_obj_buffer( y_orig );
      scomplex*   YY    = (scomplex*) bli_obj_buffer( y );
      libblis_icdotxf_check<scomplex, int32_t>( M, N, Alpha, A, rsa, csa,
                                          cfa, X, incx, cfx, Beta, Y, incy );
      resid = computediffiv(M, incy, YY, Y);
      break;
    }
    case BLIS_DCOMPLEX :
    {
      dcomplex*   Alpha = (dcomplex*) bli_obj_buffer( alpha );
      dcomplex*   A     = (dcomplex*) bli_obj_buffer( a );
      dcomplex*   X     = (dcomplex*) bli_obj_buffer( x );
      dcomplex*   Beta  = (dcomplex*) bli_obj_buffer( beta );
      dcomplex*   Y     = (dcomplex*) bli_obj_buffer( y_orig );
      dcomplex*   YY    = (dcomplex*) bli_obj_buffer( y );
      libblis_icdotxf_check<dcomplex, int64_t>( M, N, Alpha, A, rsa, csa,
                                          cfa, X, incx, cfx, Beta, Y, incy );
      resid = computediffiv(M, incy, YY, Y);
      break;
    }
    default :
      bli_check_error_code( BLIS_INVALID_DATATYPE );
  }

  return resid;
}

