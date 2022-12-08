#include "blis_test.h"
#include "blis_utils.h"
#include "test_axpyv.h"

using namespace std;

//*  ==========================================================================
//*> AXPYF performs vector operations
//*>    y := y + alpha * conja(A) * conjx(x)
//*>    where A is an m x b matrix, and y and x are vectors.
//*>    The kernel  is implemented as a fused series of calls to axpyv
//*>    where b is less than or equal to an implementation-dependent
//*>    fusing factor specific to axpyf
//*  ==========================================================================

template <typename T, typename U>
void libblis_iaxpyf_check(dim_t M, dim_t N, T* alpha, T* A, dim_t rsa,
                             dim_t csa, T* X, dim_t incx, T* Y, dim_t incy) {
  dim_t i, j, ix, iy;
  T Alpha = alpha[0];
  T temp;
  if((M == 0) || (N == 0)) {
    return;
  }

  ix = 0;
  for(j = 0 ; j < N ; j++) {
    temp = Alpha * X[ix];
    iy = 0;
    for(i = 0 ; i < M ; i++) {
      Y[iy] = Y[iy] + temp * A[i*rsa + j*csa];
      iy = iy + incy;
    }
    ix = ix + incx;
  }

  return;
}

template <typename T, typename U>
void libblis_icaxpyf_check(dim_t M, dim_t N, T* alpha, T* A, dim_t rsa,
        dim_t csa, bool cfa, T* X, dim_t incx, bool cfx, T* Y, dim_t incy ) {

  dim_t i, j, ix, iy;
  T Alpha = *alpha;
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
    for(j = 0 ; j < N ; j++) {
      for(i = 0 ; i < M ; i++) {
        A[i*rsa + j*csa] = conjugate<T>(A[i*rsa + j*csa]);
      }
    }
  }

  ix = 0;
  for(j = 0 ; j < N ; j++) {
    temp = mulc<T>(Alpha , X[ix]);
    iy = 0;
    for(i = 0 ; i < M ; i++) {
      Y[iy] = addc<T>(Y[iy] , mulc<T>(temp , A[i*rsa + j*csa]));
      iy = iy + incy;
    }
    ix = ix + incx;
  }

  return;
}

double libblis_test_iaxpyf_check (
  test_params_t* params,
  obj_t*         alpha,
  obj_t*         a,
  obj_t*         x,
  obj_t*         y,
  obj_t*         y_orig
) {
  num_t  dt    = bli_obj_dt( y );
  dim_t  M     = bli_obj_vector_dim( y );
  dim_t  N     = bli_obj_width( a );
  f77_int incx = bli_obj_vector_inc( x );
  f77_int incy = bli_obj_vector_inc( y );
  bool cfx     = bli_obj_has_conj( x );
  bool cfa     = bli_obj_has_conj( a );
  f77_int rsa  = bli_obj_row_stride( a ) ;
  f77_int csa  = bli_obj_col_stride( a ) ;
  double resid = 0.0;

  switch( dt )  {
    case BLIS_FLOAT :
    {
      float*   Alpha    = (float*) bli_obj_buffer( alpha );
      float*   A        = (float*) bli_obj_buffer( a );
      float*   X        = (float*) bli_obj_buffer( x );
      float*   Y        = (float*) bli_obj_buffer( y_orig );
      float*   YY       = (float*) bli_obj_buffer( y );
      libblis_iaxpyf_check<float, int32_t>( M,  N, Alpha, A, rsa, csa,
                                                    X, incx, Y, incy );
      resid = computediffrv(M, incy, YY, Y);
      break;
    }
    case BLIS_DOUBLE :
    {
      double*   Alpha   = (double*) bli_obj_buffer( alpha );
      double*   A       = (double*) bli_obj_buffer( a );
      double*   X       = (double*) bli_obj_buffer( x );
      double*   Y       = (double*) bli_obj_buffer( y_orig );
      double*   YY      = (double*) bli_obj_buffer( y );
      libblis_iaxpyf_check<double, int64_t>( M,  N, Alpha, A, rsa, csa,
                                                    X, incx, Y, incy );
      resid = computediffrv(M, incy, YY, Y);
      break;
    }
    case BLIS_SCOMPLEX :
    {
      scomplex*   Alpha = (scomplex*) bli_obj_buffer( alpha );
      scomplex*   A     = (scomplex*) bli_obj_buffer( a );
      scomplex*   X     = (scomplex*) bli_obj_buffer( x );
      scomplex*   Y     = (scomplex*) bli_obj_buffer( y_orig );
      scomplex*   YY    = (scomplex*) bli_obj_buffer( y );
      libblis_icaxpyf_check<scomplex, int32_t>( M,  N, Alpha, A, rsa, csa,
                                               cfa, X, incx, cfx, Y, incy );
      resid = computediffiv(M, incy, YY, Y);
      break;
    }
    case BLIS_DCOMPLEX :
    {
      dcomplex*   Alpha = (dcomplex*) bli_obj_buffer( alpha );
      dcomplex*   A     = (dcomplex*) bli_obj_buffer( a );
      dcomplex*   X     = (dcomplex*) bli_obj_buffer( x );
      dcomplex*   Y     = (dcomplex*) bli_obj_buffer( y_orig );
      dcomplex*   YY    = (dcomplex*) bli_obj_buffer( y );
      libblis_icaxpyf_check<dcomplex, int64_t>( M,  N, Alpha, A, rsa, csa,
                                               cfa, X, incx, cfx, Y, incy );
      resid = computediffiv(M, incy, YY, Y);
      break;
    }
    default :
      bli_check_error_code( BLIS_INVALID_DATATYPE );
  }

  return resid;
}

