/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.

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

#include "blis.h"
#include <dlfcn.h>
#include "level3/ref_trmm3.h"

/*
 * ==========================================================================
 * TRMM3  performs one of the matrix-matrix operations
 *    C := beta * C_orig + alpha * transa(A) * transb(B)
 * or
 *    C := beta * C_orig + alpha * transb(B) * transa(A)
 * where alpha and beta are scalars, A is an triangular matrix
 * and  B and C are m by n matrices.
 * ==========================================================================
 */

namespace testinghelpers {

template <typename T>
void ref_trmm3( char storage, char side, char uploa, char trnsa, char diaga,
    char trnsb, gtint_t M, gtint_t N, T alpha, T *A, gtint_t lda,
    T *B, gtint_t ldb, T beta, T *C, gtint_t ldc )
{

    T one ;
    T zero;
    T tmp;
    dim_t i, j, k;
    initone<T>(one);
    initzero<T>(zero);

    //*     Test the input parameters.
    bool lside  = ( testinghelpers::chksideleft( side ) );
    bool upper  = ( testinghelpers::chkupper( uploa ) );
    bool unitdg = ( testinghelpers::chkunitdiag( diaga ) );
    bool transa = ( testinghelpers::chktrans( trnsa ) );
    bool transb = ( testinghelpers::chktrans( trnsb ) );
    bool conja  = ( testinghelpers::chktransconj( trnsa ) );
    bool conjb  = ( testinghelpers::chktransconj( trnsb ) );

    dim_t rsa,csa;
    dim_t rsb,csb;
    dim_t rsc,csc;

    if( (storage == 'c') || (storage == 'C') ) {
        rsa = transa ? lda : 1 ;
        csa = transa ? 1 : lda ;
        rsb = transb ? ldb : 1 ;
        csb = transb ? 1 : ldb ;
        rsc = 1 ;
        csc = ldc ;
    }
    else {
        rsa = transa ? 1 : lda ;
        csa = transa ? lda : 1 ;
        rsb = transb ? 1 : ldb ;
        csb = transb ? ldb : 1 ;
        rsc = ldc ;
        csc = 1 ;
    }

    if( (M == 0 || N == 0) || ( alpha == zero && beta == one ) )
      return;

    if( transa ) {
      upper   = !upper;
    }

    gtint_t mn;
    if( lside )  mn = M;
    else         mn = N;

    if( conja )
    {
        testinghelpers::conj<T>( storage, A, mn, mn, lda );
    }

    if( conjb )
    {
        testinghelpers::conj<T>( storage, B, M, N, lda );
    }

    //*     And when  alpha.eq.zero.
    if( alpha == zero )
    {
        if( beta == zero )
        {
            for( j = 0 ; j < N ; j++ )
            {
                for( i = 0 ; i < M ; i++ )
                {
                    C[i*rsc + j*csc] = zero;
                }
            }
        }
        else
        {
            for( j = 0 ; j < N ; j++ )
            {
                for( i = 0 ; i < M ; i++ )
                {
                    C[i*rsc + j*csc] = beta*C[i*rsc + j*csc];
                }
            }
        }
        return;
    }

    if( unitdg )
    {
        for( i = 0 ; i < mn ; i++ )
        {
            for( j = 0 ; j < mn ; j++ )
            {
                if( i==j )
                    A[i*rsa + j*csa] = one ;
            }
        }
    }

    //*     Start the operations.
    if( lside )
    {
        //* Form  C := beta * C_orig + alpha * transa(A) * transb(B)
        if( upper )
        {
            for( j = 0 ; j < N ; j++ )
            {
                for( i = 0 ; i < M ; i++ )
                {
                    tmp = zero;
                    for( k = i ; k < M ; k++ )
                    {
                        auto val = A[i*rsa + k*csa] * B[k*rsb + j*csb];
                        tmp = tmp + val ;
                    }
                    if( beta == zero )
                    {
                        C[i*rsc + j*csc] = alpha*tmp;
                    }
                    else
                    {
                        C[i*rsc + j*csc] = beta*C[i*rsc + j*csc] + alpha*tmp;
                    }
                }
            }
        }
        else
        {
            for( j = 0 ; j < N ; j++ )
            {
                for( i = 0 ; i < M ; i++ )
                {
                    tmp = zero;
                    for( k = 0 ; k <= i ; k++ )
                    {
                        auto val = A[i*rsa + k*csa] * B[k*rsb + j*csb];
                        tmp = tmp + val ;
                    }
                    if( beta == zero )
                    {
                        C[i*rsc + j*csc] = alpha*tmp;
                    }
                    else
                    {
                        C[i*rsc + j*csc] = beta*C[i*rsc + j*csc] + alpha*tmp;
                    }
                }
            }
        }
    }
    else
    {
        //* C := beta * C_orig + alpha * transb(B) * transa(A)
        if( upper )
        {
            for( i = 0 ; i < M ; i++ )
            {
                for( j = 0 ; j < N ; j++ )
                {
                    tmp = zero ;
                    for( k = 0 ; k <= j ; k++ )
                    {
                        auto val = B[i*rsb + k*csb]* A[k*rsa + j*csa];
                        tmp = tmp + val ;
                    }
                    if( beta == zero )
                    {
                        C[i*rsc + j*csc] = alpha*tmp;
                    }
                    else
                    {
                        C[i*rsc + j*csc] = beta*C[i*rsc + j*csc] + alpha*tmp;
                    }
                }
            }
        }
        else
        {
            for( i = 0 ; i < M ; i++ )
            {
                for( j = 0 ; j < N ; j++ )
                {
                    tmp = zero ;
                    for( k = j ; k < N ; k++ )
                    {
                        auto val = B[i*rsb + k*csb]* A[k*rsa + j*csa];
                        tmp = tmp + val ;
                    }
                    if( beta == zero )
                    {
                        C[i*rsc + j*csc] = alpha*tmp;
                    }
                    else
                    {
                        C[i*rsc + j*csc] = beta*C[i*rsc + j*csc] + alpha*tmp;
                    }
                }
            }
        }
    }
    return;
}

// Explicit template instantiations
template void ref_trmm3<float>( char, char, char, char, char, char, gtint_t, gtint_t,
                    float, float *, gtint_t, float *, gtint_t, float, float *, gtint_t );
template void ref_trmm3<double>( char, char, char, char, char, char, gtint_t, gtint_t,
               double, double *, gtint_t, double *, gtint_t, double, double *, gtint_t );
template void ref_trmm3<scomplex>( char, char, char, char, char, char, gtint_t, gtint_t,
     scomplex, scomplex *, gtint_t, scomplex *, gtint_t, scomplex, scomplex *, gtint_t );
template void ref_trmm3<dcomplex>( char, char, char, char, char, char, gtint_t, gtint_t,
     dcomplex, dcomplex *, gtint_t, dcomplex *, gtint_t, dcomplex, dcomplex *, gtint_t );

} //end of namespace testinghelpers
