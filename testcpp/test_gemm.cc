/*

   BLISPP
   C++ test driver for BLIS CPP gemm routine and reference blis gemm routine.

   Copyright (C) 2019, Advanced Micro Devices, Inc.

   Redistribution and use in source and binary forms, with or without
   modification, are permitted provided that the following conditions are
   met:
    - Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.
    - Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.
    - Neither the name(s) of the copyright holder(s) nor the names of its
      contributors may be used to endorse or promote products derived
      from this software without specific prior written permission.

   THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
   "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
   LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
   A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
   HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
   SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
   LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
   DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
   THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
   (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
   OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

*/

#include <complex>
#include <iostream>
#include "blis.hh"
#include "test_gemm.hh"

using namespace blis;
using namespace std;
#define PRINT
/*
 * Test application assumes matrices to be column major, non-transposed
 */
void ref_gemm(num_t dt , int64_t m, int64_t n, int64_t k,
    void * alpha,
    void *A,
    void *B,
    void * beta,
    void *C )

{
   obj_t obj_a, obj_b, obj_c;
   obj_t obj_alpha, obj_beta;
   bli_obj_create_with_attached_buffer( dt, 1, 1, alpha, 1,1,&obj_alpha );
   bli_obj_create_with_attached_buffer( dt, 1, 1, beta,  1,1,&obj_beta );
   bli_obj_create_with_attached_buffer( dt, m, k, A, 1,m,&obj_a );
   bli_obj_create_with_attached_buffer( dt, k, n, B,1,k,&obj_b );
   bli_obj_create_with_attached_buffer( dt, m, n, C, 1,m,&obj_c );

   bli_obj_set_conjtrans( BLIS_NO_TRANSPOSE, &obj_a );
   bli_obj_set_conjtrans( BLIS_NO_TRANSPOSE, &obj_b );
   bli_gemm( &obj_alpha,
             &obj_a,
             &obj_b,
             &obj_beta,
             &obj_c );
	
}
template< typename T >
void test_gemm(  ) 
{
    int    i, j, p;
    T *A, *B, *C, *C_ref;
    T alpha, beta;
    int m,n,k;
    int    lda, ldb, ldc, ldc_ref;

    alpha = 1.0;
    beta = 0.0;
    m = 5;
    k = 4;
    n = 6;

    A    = new T[m * k];
    B    = new T[k * n];

    lda = m;
    ldb = k;
    ldc     = m;
    ldc_ref = m;
    C    = new T[ldc * n];
    C_ref= new T[m * n];

    srand48 (time(NULL));

    // Randonly generate points in [ 0, 1 ].
    for ( p = 0; p < k; p ++ ) {
        for ( i = 0; i < m; i ++ ) {
            A( i, p ) = (T)( drand48() );	
        }
    }
    for ( j = 0; j < n; j ++ ) {
        for ( p = 0; p < k; p ++ ) {
            B( p, j ) = (T)( drand48() );
        }
    }

    for ( j = 0; j < n; j ++ ) {
        for ( i = 0; i < m; i ++ ) {
            C_ref( i, j ) = (T)( 0.0 );	
                C( i, j ) = (T)( 0.0 );	
        }
    }
#ifdef PRINT
    bl_dgemm_printmatrix(A, lda ,m,k);
    bl_dgemm_printmatrix(B, ldb ,k,n);
    bl_dgemm_printmatrix(C, ldc ,m,n);
#endif
	blis::gemm(
	    CblasColMajor,
	    CblasNoTrans,
	    CblasNoTrans,
            m,
            n,
            k,
	    alpha,
            A,
            lda,
            B,
            ldb,
	    beta,
            C,
            ldc
            );

#ifdef PRINT
    bl_dgemm_printmatrix(C, ldc ,m,n);
#endif
   if(is_same<T, float>::value)
       ref_gemm(BLIS_FLOAT , m, n, k, &alpha, A, B, &beta, C_ref);
   else if(is_same<T, double>::value)
       ref_gemm(BLIS_DOUBLE , m, n, k, &alpha, A, B, &beta, C_ref);
   else if(is_same<T, complex<float>>::value)
       ref_gemm(BLIS_SCOMPLEX , m, n, k, &alpha, A, B, &beta, C_ref);
   else if(is_same<T, complex<double>>::value)
       ref_gemm(BLIS_DCOMPLEX , m, n, k, &alpha, A, B, &beta, C_ref);

#ifdef PRINT
    bl_dgemm_printmatrix(C_ref, ldc_ref ,m,n);
#endif
    if(computeError(ldc, ldc_ref, m, n, C, C_ref )==1)
	    printf("%s TEST FAIL\n" ,__func__);
    else
	    printf("%s TEST PASS\n" , __func__);



    delete[]( A     );
    delete[]( B     );
    delete[]( C     );
    delete[]( C_ref );
}

// -----------------------------------------------------------------------------
int main( int argc, char** argv )
{
    test_gemm<double>( );
    test_gemm<float>( );
    test_gemm<complex<float>>( );
    test_gemm<complex<double>>( );
    return 0;

}
