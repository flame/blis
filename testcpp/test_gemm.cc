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
<<<<<<< HEAD
#include "blis.hh"
#include "test.hh"
=======
#include <string.h>
#include <unistd.h>
#include "blis.hh"
>>>>>>> Code Cleanup done; Test code updated to add performance measurement

using namespace blis;
using namespace std;
//#define PRINT
#define ALPHA 1.0
#define BETA 0.0
#define M 5
#define N 6
#define K 4

/*
 * Test application assumes matrices to be column major, non-transposed
 */
template< typename T >
void ref_gemm(int64_t m, int64_t n, int64_t k,
    T * alpha,
    T *A,
    T *B,
    T * beta,
    T *C )

{
   obj_t obj_a, obj_b, obj_c;
   obj_t obj_alpha, obj_beta;
   num_t dt;
   if(is_same<T, float>::value)
       dt = BLIS_FLOAT;
   else if(is_same<T, double>::value)
       dt = BLIS_DOUBLE;
   else if(is_same<T, complex<float>>::value)
       dt = BLIS_SCOMPLEX;
   else if(is_same<T, complex<double>>::value)
       dt = BLIS_DCOMPLEX;

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
    T *A, *B, *C, *C_ref;
    T alpha, beta;
    int m,n,k;
    int    lda, ldb, ldc, ldc_ref;

    alpha = ALPHA;
    beta = BETA;
    m = M;
    k = K;
    n = N;

    lda = m;
    ldb = k;
    ldc     = m;
    ldc_ref = m;
    srand (time(NULL));
    allocate_init_buffer(A , m , k);
    allocate_init_buffer(B , k , n);
    allocate_init_buffer(C , m , n);
    copy_buffer(C, C_ref , m ,n);

#ifdef PRINT
    printmatrix(A, lda ,m,k , (char *)"A");
    printmatrix(B, ldb ,k,n, (char *)"B");
    printmatrix(C, ldc ,m,n, (char *)"C");
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
    printmatrix(C,ldc ,m,n , (char *)"C output");
#endif
   ref_gemm(m, n, k, &alpha, A, B, &beta, C_ref);

#ifdef PRINT
    printmatrix(C_ref, ldc_ref ,m,n, (char *)"C ref output");
#endif
    if(computeErrorM(ldc, ldc_ref, m, n, C, C_ref )==1)
	    printf("%s TEST FAIL\n" , __PRETTY_FUNCTION__ );
    else
	    printf("%s TEST PASS\n" , __PRETTY_FUNCTION__ );



    delete[]( A     );
    delete[]( B     );
    delete[]( C     );
    delete[]( C_ref );
}

// -----------------------------------------------------------------------------
int main( int argc, char** argv )
{
<<<<<<< HEAD
    test_gemm<double>( );
    test_gemm<float>( );
    test_gemm<complex<float>>( );
    test_gemm<complex<double>>( );
    return 0;
=======
	int M, N, K, lda, ldb, ldc;
	double a_d[DIM * DIM] = { 1.111, 2.222, 3.333, 4.444 };
	double b_d[DIM * DIM] = { 5.555, 6.666, 7.777, 8.888 };
	double c_d[DIM * DIM];
	double alpha_d, beta_d;
	float a_f[DIM * DIM] = { 1.1, 2.2, 3.3, 4.4 };
	float b_f[DIM * DIM] = { 5.5, 6.6, 7.7, 8.8 };
	float c_f[DIM * DIM];
	float alpha_f, beta_f;
	std::complex<float> a_c[DIM * DIM]={{1, 2},{3, 4},{5,6},{7,8}};
	std::complex<float> b_c[DIM * DIM]={{1, 2},{3, 4},{5,6},{7,8}};
	std::complex<float> c_c[DIM * DIM];
	std::complex<float> alpha_c, beta_c;
	std::complex<double> a_z[DIM * DIM]={{1.1, 2.2},{3.3, 4.4},{5.5,6.6},{7.7,8.8}};
	std::complex<double> b_z[DIM * DIM]={{1.1, 2.2},{3.3, 4.4},{5.5,6.6},{7.7,8.8}};
	std::complex<double> c_z[DIM * DIM];
	std::complex<double> alpha_z, beta_z;
	M = DIM;
	N = M;
	K = M;
	lda = M;
	ldb = K;
	ldc = M;
	alpha_d = 1.0;
	beta_d = 0.0;
	alpha_f = 1.0;
	beta_f = 0.0;
	alpha_c = {1.0,1.0};
	beta_c = {0.0,0.0};
	alpha_z = {1.0,1.0};
	beta_z = {0.0,0.0};

	/*cblis_sgemm*/	
	cout<<"a_f= \n";
	print_matrix<float>(a_f , M , K);
	cout<<"b_f= \n";
	print_matrix<float>(b_f , K , N);
	blis::gemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, N, K, alpha_f, a_f,
			lda, b_f, ldb, beta_f, c_f, ldc);
	cout<<"c_f= \n";
	print_matrix<float>(c_f , M , N);


	/*cblis_dgemm*/	
	printf("a_d = \n");
	print_matrix<double>(a_d , M , K);
	printf("b_d = \n");
	print_matrix<double>(b_d , K , N);
	blis::gemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, N, K, alpha_d, a_d,
			lda, b_d, ldb, beta_d, c_d, ldc);
	printf("c_d = \n");
	print_matrix<double>(c_d , M , N);


	/*cblis_cgemm*/	
	printf("a_c = \n");
	print_matrix<std::complex<float>>(a_c , M , K);
	printf("b_c = \n");
	print_matrix<std::complex<float>>(b_c , K , N);
	blis::gemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, N, K, alpha_c, a_c,
			lda, b_c, ldb, beta_c, c_c, ldc);
	printf("c_c = \n");
	print_matrix<std::complex<float>>(c_c , M , N);


	/*cblis_zgemm*/	
	printf("a_z = \n");
	print_matrix<std::complex<double>>(a_z , M , K);
	printf("b_z = \n");
	print_matrix<std::complex<double>>(b_z , K , N);
	blis::gemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, N, K, alpha_z, a_z,
			lda, b_z, ldb, beta_z, c_z, ldc);
	printf("c_z = \n");
	print_matrix<std::complex<double>>(c_z , M , N);
	return 0;
>>>>>>> Code Cleanup done; Test code updated to add performance measurement

}
