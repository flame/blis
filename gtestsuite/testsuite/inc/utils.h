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

#pragma once

#pragma once
#include "blis.h"
#include "common/testing_helpers.h"

/*
 * ==========================================================================
 * MKHERM
 * Make an m x m matrix A explicitly Hermitian by copying the conjugate
 * of the triangle specified by uploa to the opposite triangle. Imaginary
 * components of diagonal elements are explicitly set to zero.
 * It is assumed that the diagonal offset of A is zero.
 * ==========================================================================
 */
template<typename T>
static void mkherm( char storage, char uplo, gtint_t n, T* ap, gtint_t lda )
{
    uplo_t  uploa;

    // Map parameter characters to BLIS constants.
    testinghelpers::char_to_blis_uplo ( uplo, &uploa );

    dim_t rsa,csa;
    rsa=csa=1;
    /* a = n x n   */
    if( (storage == 'c') || (storage == 'C') )
        csa = lda ;
    else
        rsa = lda ;

    if constexpr (std::is_same<T, float>::value)
        bli_smkherm( uploa, n, ap, rsa, csa );
    else if constexpr (std::is_same<T, double>::value)
        bli_dmkherm( uploa, n, ap, rsa, csa );
    else if constexpr (std::is_same<T, scomplex>::value)
        bli_cmkherm( uploa, n, ap, rsa, csa );
    else if constexpr (std::is_same<T, dcomplex>::value)
        bli_zmkherm( uploa, n, ap, rsa, csa );
    else

        throw std::runtime_error("Error in utils.h: Invalid typename in mkherm().");
}

/*
 * ==========================================================================
 * MKSYMM
 * Make an m x m matrix A explicitly symmetric by copying the triangle
 * specified by uploa to the opposite triangle.
 * It is assumed that the diagonal offset of A is zero.
 * ==========================================================================
 */

template<typename T>
static void mksymm( char storage, char uplo, gtint_t n, T* ap, gtint_t lda )
{
    uplo_t  uploa;

    // Map parameter characters to BLIS constants.
    testinghelpers::char_to_blis_uplo ( uplo, &uploa );

    dim_t rsa,csa;
    rsa=csa=1;
    /* a = n x n   */
    if( (storage == 'c') || (storage == 'C') )
        csa = lda ;
    else
        rsa = lda ;

    if constexpr (std::is_same<T, float>::value)
        bli_smksymm( uploa, n, ap, rsa, csa );
    else if constexpr (std::is_same<T, double>::value)
        bli_dmksymm( uploa, n, ap, rsa, csa );
    else if constexpr (std::is_same<T, scomplex>::value)
        bli_cmksymm( uploa, n, ap, rsa, csa );
    else if constexpr (std::is_same<T, dcomplex>::value)
        bli_zmksymm( uploa, n, ap, rsa, csa );
    else

        throw std::runtime_error("Error in utils.h: Invalid typename in mksymm().");
}

/*
 * ==========================================================================
 * MKTRIM
 * Make an m x m matrix A explicitly triangular by preserving the triangle
 * specified by uploa and zeroing the elements in the opposite triangle.
 * It is assumed that the diagonal offset of A is zero
 * ==========================================================================
 */
template<typename T>
static void mktrim( char storage, char uplo, gtint_t n, T* ap, gtint_t lda )
{
    uplo_t  uploa;

    // Map parameter characters to BLIS constants.
    testinghelpers::char_to_blis_uplo ( uplo, &uploa );

    dim_t rsa,csa;
    rsa=csa=1;
    /* a = n x n   */
    if( (storage == 'c') || (storage == 'C') )
        csa = lda ;
    else
        rsa = lda ;

    if constexpr (std::is_same<T, float>::value)
        bli_smktrim( uploa, n, ap, rsa, csa );
    else if constexpr (std::is_same<T, double>::value)
        bli_dmktrim( uploa, n, ap, rsa, csa );
    else if constexpr (std::is_same<T, scomplex>::value)
        bli_cmktrim( uploa, n, ap, rsa, csa );
    else if constexpr (std::is_same<T, dcomplex>::value)
        bli_zmktrim( uploa, n, ap, rsa, csa );
    else

        throw std::runtime_error("Error in utils.h: Invalid typename in mktrim().");
}

template<typename T>
static void print( T x, const char *spec ) {
    if constexpr (testinghelpers::type_info<T>::is_real)
        printf(spec, x);
    else {
        printf( spec, x.real );
        if(x.imag < 0)    printf( " -" );
        else              printf( " +" );
        printf( spec, abs(x.imag) );
        printf( " " );
    }
}

template<typename T>
void printmat( const char *mat, char storage, gtint_t m, gtint_t n, T *a, gtint_t ld, const char *spec )
{
    dim_t i, j;
    dim_t rs,cs;
    rs=cs=1;
    T val;
    if( (storage == 'c') || (storage == 'C') )
        cs = ld ;
    else
        rs = ld ;

    std::cout <<"matrix : " <<  mat <<  std::endl;

    for ( i = 0; i < m; i++ )
    {
        for ( j = 0; j < n; j++ )
        {
            val = a[i*rs + j*cs];
            print<T>(val,spec);
            printf( " " );
        }
        printf( "\n" );
    }
    printf( "\n" );
}

template<typename T>
void printvec( const char *vec, gtint_t n, T *x, gtint_t incx, const char *spec )
{
    dim_t i, idx;
    T val;

    std::cout <<"vector : " <<  vec <<  std::endl;

    for ( i = 0; i < n; i++ )
    {
        idx = (incx > 0) ? (i * incx) : ( - ( n - i - 1 ) * incx );
        val = x[idx];
        print<T>(val,spec);
        printf( " " );
    }
    printf( "\n\n" );
}

