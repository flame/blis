/*

   BLISPP
   C++ test driver for BLIS CPP gemm routine and reference cblas gemm routine.

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

void test_dgemm(  ) 
{
    int    i, j, p;
    double *A, *B, *C, *C_ref;
    double alpha, beta;
    double flops;
    double ref_beg, ref_time, bl_dgemm_beg, bl_dgemm_time;
    int    nrepeats;
    int m,n,k;
    int    lda, ldb, ldc, ldc_ref;
    double ref_rectime, bl_dgemm_rectime;

    alpha = 1.0;
    beta = 0.0;
    m = 5;
    k = 6;
    n = 4;

    A    = new double[m * k];
    B    = new double[k * n];

    lda = m;
    ldb = k;
    ldc     = m;
    ldc_ref = m;
    C    = new double[ldc * n];
    C_ref= new double[m * n];

    nrepeats = 3;

    srand48 (time(NULL));

    // Randonly generate points in [ 0, 1 ].
    for ( p = 0; p < k; p ++ ) {
        for ( i = 0; i < m; i ++ ) {
            A( i, p ) = (double)( drand48() );	
        }
    }
    for ( j = 0; j < n; j ++ ) {
        for ( p = 0; p < k; p ++ ) {
            B( p, j ) = (double)( drand48() );
        }
    }

    for ( j = 0; j < n; j ++ ) {
        for ( i = 0; i < m; i ++ ) {
            C_ref( i, j ) = (double)( 0.0 );	
                C( i, j ) = (double)( 0.0 );	
        }
    }
#ifdef PRINT
    bl_dgemm_printmatrix(A, lda ,m,k);
    bl_dgemm_printmatrix(B, ldb ,k,n);
    bl_dgemm_printmatrix(C, ldc ,m,n);
#endif
    for ( i = 0; i < nrepeats; i ++ ) {
        bl_dgemm_beg = bl_clock();
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
        bl_dgemm_time = bl_clock() - bl_dgemm_beg;

        if ( i == 0 ) {
            bl_dgemm_rectime = bl_dgemm_time;
        } else {
            bl_dgemm_rectime = bl_dgemm_time < bl_dgemm_rectime ? bl_dgemm_time : bl_dgemm_rectime;
        }
    }

#ifdef PRINT
    bl_dgemm_printmatrix(C, ldc ,m,n);
#endif
    for ( i = 0; i < nrepeats; i ++ ) {
        ref_beg = bl_clock();
	cblas_dgemm(
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
            C_ref,
            ldc_ref);
        ref_time = bl_clock() - ref_beg;

        if ( i == 0 ) {
            ref_rectime = ref_time;
        } else {
            ref_rectime = ref_time < ref_rectime ? ref_time : ref_rectime;
        }
    }

#ifdef PRINT
    bl_dgemm_printmatrix(C_ref, ldc_ref ,m,n);
#endif
    if(computeError(ldc, ldc_ref, m, n, C, C_ref )==1)
	    printf("%s TEST FAIL\n" ,__func__);
    else
	    printf("%s TEST PASS\n" , __func__);


    // Compute overall floating point operations.
    flops = ( m * n / ( 1000.0 * 1000.0 * 1000.0 ) ) * ( 2 * k );

    printf( "%5d\t %5d\t %5d\t %5.2lf\t %5.2lf\n", 
            m, n, k, flops / bl_dgemm_rectime, flops / ref_rectime );

    free( A     );
    free( B     );
    free( C     );
    free( C_ref );
}
void test_zgemm(  )
{
    int    i, j, p;
    std::complex<double> *A, *B, *C, *C_ref;
    std::complex<double> alpha, beta;
    double flops;
    double ref_beg, ref_time, bl_dgemm_beg, bl_dgemm_time;
    int    nrepeats;
    int m,n,k;
    int    lda, ldb, ldc, ldc_ref;
    double ref_rectime, bl_dgemm_rectime;

    alpha = 1.0;
    beta = 0.0;
    m = 5;
    k = 6;
    n = 4;

    A    = new complex<double>[m * k];
    B    = new complex<double>[k * n];

    lda = m;
    ldb = k;
    ldc     = m;
    ldc_ref = m;
    C    = new complex<double>[ldc * n];
    C_ref= new complex<double>[m * n];
    nrepeats = 3;

    srand48 (time(NULL));

    // Randonly generate points in [ 0, 1 ].
    for ( p = 0; p < k; p ++ ) {
        for ( i = 0; i < m; i ++ ) {
            A( i, p ) = (complex<double>)( drand48() );
        }
    }
    for ( j = 0; j < n; j ++ ) {
        for ( p = 0; p < k; p ++ ) {
            B( p, j ) = (complex<double>)( drand48() );
        }
    }

    for ( j = 0; j < n; j ++ ) {
        for ( i = 0; i < m; i ++ ) {
            C_ref( i, j ) = (complex<double>)( 0.0 );
                C( i, j ) = (complex<double>)( 0.0 );
        }
    }
#ifdef PRINT
    bl_dgemm_printmatrix(A, lda ,m,k);
    bl_dgemm_printmatrix(B, ldb ,k,n);
    bl_dgemm_printmatrix(C, ldc ,m,n);
#endif
    for ( i = 0; i < nrepeats; i ++ ) {
        bl_dgemm_beg = bl_clock();
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

        bl_dgemm_time = bl_clock() - bl_dgemm_beg;

           
	if ( i == 0 ) {
            bl_dgemm_rectime = bl_dgemm_time;
        } else {
            bl_dgemm_rectime = bl_dgemm_time < bl_dgemm_rectime ? bl_dgemm_time : bl_dgemm_rectime;
        }
    }

#ifdef PRINT
    bl_dgemm_printmatrix(C, ldc ,m,n);
#endif
    for ( i = 0; i < nrepeats; i ++ ) {
        ref_beg = bl_clock();
        cblas_zgemm(
            CblasColMajor,
            CblasNoTrans,
            CblasNoTrans,
            m,
            n,
            k,
            &alpha,
            A,
            lda,
            B,
            ldb,
            &beta,
            C_ref,
            ldc_ref);
        ref_time = bl_clock() - ref_beg;

        if ( i == 0 ) {
            ref_rectime = ref_time;
        } else {
            ref_rectime = ref_time < ref_rectime ? ref_time : ref_rectime;
        }
    }

#ifdef PRINT
    bl_dgemm_printmatrix(C_ref, ldc_ref ,m,n);
#endif
    if(computeError(ldc, ldc_ref, m, n, C, C_ref )==1)
            printf("%s TEST FAIL\n" ,__func__);
    else
            printf("%s TEST PASS\n" , __func__);


    // Compute overall floating point operations.
    flops = ( m * n / ( 1000.0 * 1000.0 * 1000.0 ) ) * ( 2 * k );

    printf( "%5d\t %5d\t %5d\t %5.2lf\t %5.2lf\n",
            m, n, k, flops / bl_dgemm_rectime, flops / ref_rectime );

    free( A     );
    free( B     );
    free( C     );
    free( C_ref );
}

// -----------------------------------------------------------------------------
int main( int argc, char** argv )
{
    test_dgemm( );
    test_zgemm( );
    return 0;

}
