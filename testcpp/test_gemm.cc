#include <complex>

#include <stdio.h>
#include <iostream>
#include <string.h>
#include <unistd.h>
#include "blis.hh"

using namespace std;

#define DIM 2
template <typename T>
void print_matrix(T * matrix , int m , int n)
{
	for ( int L=0; L < m; L ++ ) {
		for ( int J = 0; J < n; J ++ ) {
			cout<< matrix[L * n + J]<<" ";
		}
		cout<<"\n";
	}

}
// -----------------------------------------------------------------------------
int main( int argc, char** argv )
{
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

}
