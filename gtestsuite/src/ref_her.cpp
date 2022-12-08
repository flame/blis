#include "blis_test.h"
#include "blis_utils.h"
#include "test_her.h"

using namespace std;

//*  ==========================================================================
//*> HER performs the hermitian rank 1 operation
//*>    A := alpha*x*x**H + A
//*>  where alpha is a real scalar, x is an n element vector and A is an
//*>  n by n hermitian matrix.
//*  ==========================================================================

template <typename T, typename U>
void libblis_iher_check(uplo_t uploa, dim_t N, T* alpha, T* X, dim_t incx,
                                               T* A, dim_t rsa, dim_t csa) {
  T ZERO = 0.0;
  T Alpha = alpha[0];
  T temp;
  int i, ix, j, jx, kx;

  if((N == 0) || (Alpha == ZERO))
    return;

  /* Set the start point in X if the increment is not unity. */
  if (incx > 0) {
    kx = 0;
  }
  else {
    kx = 1 - (N * incx);
  }

  if(uploa == BLIS_UPPER) {
    /* Form  A  when A is stored in upper triangle. */
    jx = kx;
    for(j = 0 ; j < N ; j++) {
      if (X[jx] != ZERO) {
        temp = Alpha * X[jx];
        ix = kx;
        for(i = 0 ; i <= j ; i++) {
          A[i*rsa + j*csa] = A[i*rsa + j*csa] + (X[ix] * temp);
          ix = ix + incx;
        }
      }
      jx = jx + incx;
    }
  }
  else {
    /* Form  A  when A is stored in lower triangle. */
    jx = kx;
    for(j = 0; j < N ; j++) {
      if (X[jx] != ZERO) {
        temp = Alpha * X[jx];
        ix = jx;
        for(i = j ; i < N ; i++) {
          A[i*rsa + j*csa] = A[i*rsa + j*csa] + (X[ix] * temp);
          ix = ix + incx;
        }
      }
      jx = jx + incx;
    }
  }

  return;
}

template <typename T, typename U>
void libblis_icher_check(uplo_t uploa, dim_t N, T* alpha, T* X, dim_t incx,
                                    bool conjx, T* A, dim_t rsa, dim_t csa) {
  T ZERO  = {0.0 , 0.0};
  T Alpha = alpha[0];
  T temp;
  int i, ix, j, jx, kx;

  if ((N == 0) || ((Alpha.real == ZERO.real) && (Alpha.imag == ZERO.imag)))
   return;

  if (incx > 0) {
    kx = 0;
  }
  else {
    kx = 1 - (N * incx);
  }

  if(conjx) {
    ix = 0;
    for(i = 0 ; i < N ; i++) {
      X[ix] = conjugate<T>(X[ix]);
      ix = ix + incx;
    }
  }

  if(uploa == BLIS_UPPER) {
    /* Form  A  when A is stored in upper triangle. */
    jx = kx;
    for(j = 0 ; j < N ; j++) {
      if ((X[jx].real != ZERO.real) || (X[jx].imag != ZERO.imag)) {
        temp = mulc<T>(Alpha , conjugate<T>(X[jx]));
        ix = kx;
        for(i = 0 ; i < j ; i++) {
          A[i*rsa + j*csa] = addc<T>(A[i*rsa + j*csa] , mulc<T>(X[ix] , temp));
          ix = ix + incx;
        }
        A[j*rsa + j*csa] = real<T>(addc<T>(A[j*rsa + j*csa] , mulc<T>(X[jx] , temp)));
      }
      else {
        A[j*rsa + j*csa] = real<T>(A[j*rsa + j*csa]);
      }
      jx = jx + incx;
    }
  }
  else {
    /* Form  A  when A is stored in lower triangle. */
    jx = kx;
    for(j = 0; j < N ; j++) {
      if ((X[jx].real != ZERO.real) || (X[jx].imag != ZERO.imag)) {
        temp = mulc<T>(Alpha , conjugate<T>(X[jx]));
        A[j*rsa + j*csa] = real<T>(addc<T>(A[j*rsa + j*csa] , mulc<T>(temp , X[jx])));
        ix = jx;
        for( i = (j+1) ; i < N ; i++) {
          ix = ix + incx;
          A[i*rsa + j*csa] = addc<T>(A[i*rsa + j*csa] , mulc<T>(X[ix] , temp));
        }
      }
      else {
        A[j*rsa + j*csa] = real<T>(A[j*rsa + j*csa]);
      }
      jx = jx + incx;
    }
  }

  return;
}

double libblis_test_iher_check(
  test_params_t* params,
  obj_t*         alpha,
  obj_t*         x,
  obj_t*         a,
  obj_t*         a_orig
){

  num_t dt     = bli_obj_dt( x );
  uplo_t uploa = bli_obj_uplo( a );
  dim_t M      = bli_obj_length( a );
  dim_t N      = bli_obj_width( a );
  dim_t incx   = bli_obj_vector_inc( x );
  bool conjx   = bli_obj_has_conj( x );
  dim_t rsa    = bli_obj_row_stride( a ) ;
  dim_t csa    = bli_obj_col_stride( a ) ;
  double resid = 0.0;

  switch( dt )  {
    case BLIS_FLOAT :
    {
      float*   Alpha    = (float*) bli_obj_buffer( alpha );
      float*   A        = (float*) bli_obj_buffer( a_orig );
      float*   X        = (float*) bli_obj_buffer( x );
      float*   AA       = (float*) bli_obj_buffer( a );
      libblis_iher_check<float, int32_t>(uploa, M, Alpha, X, incx,
                                                          A, rsa, csa);
      resid = computediffrm(M, N, AA, A, rsa, csa);
      break;
    }
    case BLIS_DOUBLE :
    {
      double*   Alpha   = (double*) bli_obj_buffer( alpha );
      double*   A       = (double*) bli_obj_buffer( a_orig );
      double*   X       = (double*) bli_obj_buffer( x );
      double*   AA      = (double*) bli_obj_buffer( a );
      libblis_iher_check<double, int64_t>(uploa, M, Alpha, X, incx,
                                                           A, rsa, csa);
      resid = computediffrm(M, N, AA, A, rsa, csa);
      break;
    }
    case BLIS_SCOMPLEX :
    {
      scomplex*   Alpha = (scomplex*) bli_obj_buffer( alpha );
      scomplex*   A     = (scomplex*) bli_obj_buffer( a_orig );
      scomplex*   X     = (scomplex*) bli_obj_buffer( x );
      scomplex*   AA    = (scomplex*) bli_obj_buffer( a );
      libblis_icher_check<scomplex, float>(uploa, M, Alpha, X, incx, conjx,
                                                              A, rsa, csa);
      resid = computediffim(M, N, AA, A, rsa, csa);
      break;
    }
    case BLIS_DCOMPLEX :
    {
      dcomplex*   Alpha = (dcomplex*) bli_obj_buffer( alpha );
      dcomplex*   A     = (dcomplex*) bli_obj_buffer( a_orig );
      dcomplex*   X     = (dcomplex*) bli_obj_buffer( x );
      dcomplex*   AA    = (dcomplex*) bli_obj_buffer( a );
      libblis_icher_check<dcomplex, double>(uploa, M, Alpha, X, incx, conjx,
                                                              A, rsa, csa);
      resid = computediffim(M, N, AA, A, rsa, csa);
      break;
    }
    default :
      bli_check_error_code( BLIS_INVALID_DATATYPE );
  }
  return resid;
}

