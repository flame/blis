/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2022, Advanced Micro Devices, Inc. All rights reserved.

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

#include "immintrin.h"
#include "blis.h"

/**
 * Optimized implementation of ZHER for lower triangular row stored &
 * upper triangular column stored matrix.
 * This kernel performs:
 * A := A + conj?(alpha) * conj?(x) * conj?(x)^H
 * where,
 *      A is an m x m hermitian matrix stored in upper/lower triangular
 *      x is a vector of length m
 *      alpha is a scalar
 */
void bli_zher_zen_int_var1
(
    uplo_t             uplo,
    conj_t             conjx,
    conj_t             conjh,
    dim_t              m,
    dcomplex* restrict alpha,
    dcomplex* restrict x, inc_t incx,
    dcomplex* restrict c, inc_t rs_c, inc_t cs_c,
    cntx_t*   restrict cntx
)
{
    double xcR, xcI;
    double xhermcR, xhermcI;
    double alphaR;
    double interR, interI;

    dcomplex* xc;
    dcomplex* xhermc;
    dcomplex* cc;

    __m256d alphaRv;
    __m256d ymm0, ymm1, ymm4, ymm5;
    __m256d ymm6, ymm7, ymm8, ymm9, ymm10, ymm11;
    __m256d ymm0_shuf, ymm1_shuf;
    __m256d conj_mulv;

    dim_t conj_multiplier;

    inc_t rs_ct, cs_ct;
    dim_t i = 0;
    dim_t j = 0;

    alphaR = alpha->real;

    // The algorithm is expressed in terms of lower triangular case;
    // the upper triangular case is supported by swapping the row and column
    // strides of A & toggling the conj parameter.
    if ( bli_is_lower( uplo ) )
    {
        rs_ct = rs_c;
        cs_ct = cs_c;
    }
    else /* if ( bli_is_upper( uplo ) ) */
    {
        rs_ct = cs_c;
        cs_ct = rs_c;
        conjx = bli_apply_conj( conjh, conjx );
    }

    // Enabling conj_multiplier for scalar multiplication based on conjx
    if ( !bli_is_conj(conjx) ) conj_multiplier =  1;
    else                       conj_multiplier = -1;

    // Broadcasting real values of alpha based on conjx
    // alphaRv = aR aR aR aR
    if ( bli_is_conj( conjx ) ) alphaRv = _mm256_broadcast_sd( &alphaR );
    else                        alphaRv = _mm256_set_pd( -alphaR, alphaR, -alphaR, alphaR );

    conj_mulv = _mm256_set_pd( conj_multiplier, -1 * conj_multiplier, conj_multiplier, -1 * conj_multiplier );

    /********* DIAGONAL ELEMENTS *********/
    // Solving for the diagonal elements using a scalar loop
    for ( i = 0; i < m; i++ )
    {
        xc = x + i*incx;
        xcR = xc->real;
        xcI = xc->imag;
        xhermcR = xc->real;
        xhermcI = xc->imag;

        xcR = alphaR * xcR;
        xcI = alphaR * xcI;
        interR = xcR * xhermcR + xcI * xhermcI;

        cc = c + (i)*rs_ct + (i)*cs_ct;
        cc->real += interR;
        cc->imag = 0;
    }

    // Vectorized loop
    for ( i = 0; ( i + 3 ) < m; i += 4 )
    {
        // Loading elements of x to ymm0-1 for computing xherm vector
        // ymm0 = x0R x0I x1R x1I
        // ymm1 = x2R x2I x3R x3I
        ymm0 = _mm256_loadu_pd( (double*)(x + i*incx) );
        ymm1 = _mm256_loadu_pd( (double*)(x + (i + 2)*incx) );

        // Scaling xherm vector with alpha
        // alphaRv        = aR      aR     aR      aR
        // ymm0           = x0R    -x0I    x1R    -x1I
        // ymm1           = x2R    -x2I    x3R    -x3I
        // ymm0 * alphaRv = aR.x0R -aR.x0I aR.x1R -aR.x1I
        // ymm1 * alphaRv = aR.x2R -aR.x2I aR.x3R -aR.x3I
        ymm0 = _mm256_mul_pd( ymm0, alphaRv );
        ymm1 = _mm256_mul_pd( ymm1, alphaRv );

        // Shuffling xherm vector for multiplication with x vector
        // ymm0_shuf = -x0I x0R -x1I x1R
        // ymm1_shuf = -x2I x2R -x3I x3R
        ymm0_shuf = _mm256_permute_pd( ymm0, 5 );
        ymm1_shuf = _mm256_permute_pd( ymm1, 5 );

        /********* TRIANGULAR BLOCK *********/
        // Solving the corner elements of the triangular block
        // using scalar multiplication
        xc = x + (i + 1)*incx;
        xcR = xc->real;
        xcI = conj_multiplier * xc->imag;

        xhermc = x + (i)*incx;
        xhermcR = xhermc->real;
        xhermcI = -1 * conj_multiplier * xhermc->imag;

        xcR = alphaR * xcR;
        xcI = alphaR * xcI;
        interR = xcR * xhermcR - xcI * xhermcI;
        interI = xcR * xhermcI + xcI * xhermcR;

        cc = c + (i + 1)*rs_ct + (i + 0)*cs_ct;
        cc->real += interR;
        cc->imag += interI;

        xc = x + (i + 3)*incx;
        xcR = xc->real;
        xcI = conj_multiplier * xc->imag;

        xhermc = x + (i + 2)*incx;
        xhermcR = xhermc->real;
        xhermcI = -1 * conj_multiplier * xhermc->imag;

        xcR = alphaR * xcR;
        xcI = alphaR * xcI;
        interR = xcR * xhermcR - xcI * xhermcI;
        interI = xcR * xhermcI + xcI * xhermcR;

        cc = c + (i + 3)*rs_ct + (i + 2)*cs_ct;
        cc->real += interR;
        cc->imag += interI;

        // Solving the 2x2 square tile inside the triangular block
        // using intrinsics
        // Broadcasting elements from x to ymm4-5
        // ymm4 = x2R x2I x2R x2I
        // ymm5 = x3R x3I x3R x3I
        ymm4 = _mm256_broadcast_pd( (__m128d const*)( x + (i + 2)*incx ) );
        ymm5 = _mm256_broadcast_pd( (__m128d const*)( x + (i + 3)*incx ) );

        // Loading a tile from matrix
        // ymm10 = c20R c20I c21R c21I
        // ymm11 = c30R c30I c31R c31I
        ymm10 = _mm256_loadu_pd( (double*)( c + (i + 2)*rs_ct + (i)*cs_ct ) );
        ymm11 = _mm256_loadu_pd( (double*)( c + (i + 3)*rs_ct + (i)*cs_ct ) );

        // Separating the real & imaginary parts of x into ymm4-7
        // ymm6 -> imag of ymm4
        // ymm4 -> real of ymm4
        ymm6 = _mm256_permute_pd( ymm4, 15 );
        ymm4 = _mm256_permute_pd( ymm4, 0 );
        ymm7 = _mm256_permute_pd( ymm5, 15 );
        ymm5 = _mm256_permute_pd( ymm5, 0 );

        // Applying conjugate to elements of x vector
        ymm6 = _mm256_mul_pd( ymm6, conj_mulv );
        ymm7 = _mm256_mul_pd( ymm7, conj_mulv );

        // Multiplying x vector with x hermitian vector
        // and adding the result to the corresponding tile
        ymm8 = _mm256_mul_pd( ymm4, ymm0 );
        ymm8 = _mm256_fmadd_pd( ymm6, ymm0_shuf, ymm8 );
        ymm10 = _mm256_add_pd( ymm10, ymm8 );

        ymm9 = _mm256_mul_pd( ymm5, ymm0 );
        ymm9 = _mm256_fmadd_pd( ymm7, ymm0_shuf, ymm9 );
        ymm11 = _mm256_add_pd( ymm11, ymm9 );

        // Storing back the results to the matrix
        _mm256_storeu_pd( (double*)( c + (i + 2)*rs_ct + (i)*cs_ct ), ymm10 );
        _mm256_storeu_pd( (double*)( c + (i + 3)*rs_ct + (i)*cs_ct ), ymm11 );

        /********* SQUARE BLOCK *********/
        // Solving a 4x4 square block of matrix using intrinsics
        for ( j = (i + 4); (j + 3) < m; j += 4)
        {
            // Broadcasting elements from x to ymm4-5
            ymm4 = _mm256_broadcast_pd( (__m128d const*)( x + (j    )*incx ) );
            ymm5 = _mm256_broadcast_pd( (__m128d const*)( x + (j + 1)*incx ) );

            // Loading a tile from matrix
            ymm10 = _mm256_loadu_pd( (double*)( c + j*rs_ct + (i    )*cs_ct ) );
            ymm11 = _mm256_loadu_pd( (double*)( c + j*rs_ct + (i + 2)*cs_ct ) );

            // Separating the real & imaginary parts of x into ymm4-7
            // ymm6 -> imag of ymm4
            // ymm4 -> real of ymm4
            ymm6 = _mm256_permute_pd( ymm4, 15 );
            ymm4 = _mm256_permute_pd( ymm4, 0 );
            ymm7 = _mm256_permute_pd( ymm5, 15 );
            ymm5 = _mm256_permute_pd( ymm5, 0 );

            // Applying conjugate to elements of x vector
            ymm6 = _mm256_mul_pd( ymm6, conj_mulv );
            ymm7 = _mm256_mul_pd( ymm7, conj_mulv );

            // Multiplying x vector with x hermitian vector
            // and adding the result to the corresponding tile
            ymm8 = _mm256_mul_pd( ymm4, ymm0 );
            ymm8 = _mm256_fmadd_pd( ymm6, ymm0_shuf, ymm8 );
            ymm10 = _mm256_add_pd( ymm10, ymm8 );

            ymm9 = _mm256_mul_pd( ymm4, ymm1 );
            ymm9 = _mm256_fmadd_pd( ymm6, ymm1_shuf, ymm9 );
            ymm11 = _mm256_add_pd( ymm11, ymm9 );

            // Storing back the results to the matrix
            _mm256_storeu_pd
            (
              (double*)( c + (j)*rs_ct + (i)*cs_ct ),
              ymm10
            );
            _mm256_storeu_pd
            (
              (double*)( c + (j)*rs_ct + (i + 2)*cs_ct ),
              ymm11
            );

            // Loading a tile from matrix
            ymm10 = _mm256_loadu_pd
                    (
                      (double*)( c + (j + 1)*rs_ct + (i)*cs_ct )
                    );
            ymm11 = _mm256_loadu_pd
                    (
                      (double*)( c + (j + 1)*rs_ct + (i + 2)*cs_ct )
                    );

            // Multiplying x vector with x hermitian vector
            // and adding the result to the corresponding tile
            ymm8 = _mm256_mul_pd( ymm5, ymm0 );
            ymm8 = _mm256_fmadd_pd( ymm7, ymm0_shuf, ymm8 );
            ymm10 = _mm256_add_pd( ymm10, ymm8 );

            ymm9 = _mm256_mul_pd( ymm5, ymm1 );
            ymm9 = _mm256_fmadd_pd( ymm7, ymm1_shuf, ymm9 );
            ymm11 = _mm256_add_pd( ymm11, ymm9 );

            // Storing back the results to the matrix
            _mm256_storeu_pd
            (
              (double*)( c + (j + 1)*rs_ct + (i)*cs_ct ),
              ymm10
            );
            _mm256_storeu_pd
            (
              (double*)( c + (j + 1)*rs_ct + (i + 2)*cs_ct ),
              ymm11
            );

            // Broadcasting elements from x to ymm4-5
            ymm4 = _mm256_broadcast_pd( (__m128d const*)( x + (j + 2)*incx ) );
            ymm5 = _mm256_broadcast_pd( (__m128d const*)( x + (j + 3)*incx ) );

            // Loading a tile from matrix
            ymm10 = _mm256_loadu_pd
                    (
                      (double*)( c + (j + 2)*rs_ct + (i)*cs_ct )
                    );
            ymm11 = _mm256_loadu_pd
                    (
                      (double*)( c + (j + 2)*rs_ct + (i + 2)*cs_ct )
                    );

            // Separating the real & imaginary parts of x into ymm4-7
            // ymm6 -> imag of ymm4
            // ymm4 -> real of ymm4
            ymm6 = _mm256_permute_pd( ymm4, 15 );
            ymm4 = _mm256_permute_pd( ymm4, 0 );
            ymm7 = _mm256_permute_pd( ymm5, 15 );
            ymm5 = _mm256_permute_pd( ymm5, 0 );

            // Applying conjugate to elements of x vector
            ymm6 = _mm256_mul_pd( ymm6, conj_mulv );
            ymm7 = _mm256_mul_pd( ymm7, conj_mulv );

            // Multiplying x vector with x hermitian vector
            // and adding the result to the corresponding tile
            ymm8 = _mm256_mul_pd( ymm4, ymm0 );
            ymm8 = _mm256_fmadd_pd( ymm6, ymm0_shuf, ymm8 );
            ymm10 = _mm256_add_pd( ymm10, ymm8 );

            ymm9 = _mm256_mul_pd( ymm4, ymm1 );
            ymm9 = _mm256_fmadd_pd( ymm6, ymm1_shuf, ymm9 );
            ymm11 = _mm256_add_pd( ymm11, ymm9 );

            // Storing back the results to the matrix
            _mm256_storeu_pd
            (
              (double*)( c + (j + 2)*rs_ct + (i)*cs_ct ),
              ymm10
            );
            _mm256_storeu_pd
            (
              (double*)( c + (j + 2)*rs_ct + (i + 2)*cs_ct ),
              ymm11
            );

            // Loading a tile from matrix
            ymm10 = _mm256_loadu_pd
                    (
                      (double*)( c + (j + 3)*rs_ct + (i)*cs_ct )
                    );
            ymm11 = _mm256_loadu_pd
                    (
                      (double*)( c + (j + 3)*rs_ct + (i + 2)*cs_ct )
                    );

            // Multiplying x vector with x hermitian vector
            // and adding the result to the corresponding tile
            ymm8 = _mm256_mul_pd( ymm5, ymm0 );
            ymm8 = _mm256_fmadd_pd( ymm7, ymm0_shuf, ymm8 );
            ymm10 = _mm256_add_pd( ymm10, ymm8 );

            ymm9 = _mm256_mul_pd( ymm5, ymm1 );
            ymm9 = _mm256_fmadd_pd( ymm7, ymm1_shuf, ymm9 );
            ymm11 = _mm256_add_pd( ymm11, ymm9 );

            // Storing back the results to the matrix
            _mm256_storeu_pd
            (
              (double*)( c + (j + 3)*rs_ct + (i)*cs_ct ),
              ymm10
            );
            _mm256_storeu_pd
            (
              (double*)( c + (j + 3)*rs_ct + (i + 2)*cs_ct ),
              ymm11
            );
        }

        // Solving a 2x2 square block of matrix using intrinsics
        for ( ; (j + 1) < m; j += 2)
        {
            // Broadcasting elements from x to ymm4-5
            ymm4 = _mm256_broadcast_pd( (__m128d const*)( x + (j)*incx ) );
            ymm5 = _mm256_broadcast_pd( (__m128d const*)( x + (j + 1)*incx ) );

            // Loading a tile from matrix
            ymm10 = _mm256_loadu_pd( (double*)( c + j*rs_ct + (i)*cs_ct ) );
            ymm11 = _mm256_loadu_pd( (double*)( c + j*rs_ct + (i + 2)*cs_ct ) );

            // Separating the real & imaginary parts of x into ymm4-7
            // ymm6 -> imag of ymm4
            // ymm4 -> real of ymm4
            ymm6 = _mm256_permute_pd( ymm4, 15 );
            ymm4 = _mm256_permute_pd( ymm4, 0 );
            ymm7 = _mm256_permute_pd( ymm5, 15 );
            ymm5 = _mm256_permute_pd( ymm5, 0 );

            // Applying conjugate to elements of x vector
            ymm6 = _mm256_mul_pd( ymm6, conj_mulv );
            ymm7 = _mm256_mul_pd( ymm7, conj_mulv );

            // Multiplying x vector with x hermitian vector
            // and adding the result to the corresponding tile
            ymm8 = _mm256_mul_pd( ymm4, ymm0 );
            ymm8 = _mm256_fmadd_pd( ymm6, ymm0_shuf, ymm8 );
            ymm10 = _mm256_add_pd( ymm10, ymm8 );

            ymm9 = _mm256_mul_pd( ymm4, ymm1 );
            ymm9 = _mm256_fmadd_pd( ymm6, ymm1_shuf, ymm9 );
            ymm11 = _mm256_add_pd( ymm11, ymm9 );

            // Storing back the results to the matrix
            _mm256_storeu_pd
            (
              (double*)( c + (j)*rs_ct + (i)*cs_ct ),
              ymm10
            );
            _mm256_storeu_pd
            (
              (double*)( c + (j)*rs_ct + (i + 2)*cs_ct ),
              ymm11
            );

            // Loading a tile from matrix
            ymm10 = _mm256_loadu_pd
                    (
                      (double*)( c + (j + 1)*rs_ct + (i)*cs_ct )
                    );
            ymm11 = _mm256_loadu_pd
                    (
                      (double*)( c + (j + 1)*rs_ct + (i + 2)*cs_ct )
                    );

            // Multiplying x vector with x hermitian vector
            // and adding the result to the corresponding tile
            ymm8 = _mm256_mul_pd( ymm5, ymm0 );
            ymm8 = _mm256_fmadd_pd( ymm7, ymm0_shuf, ymm8 );
            ymm10 = _mm256_add_pd( ymm10, ymm8 );

            ymm9 = _mm256_mul_pd( ymm5, ymm1 );
            ymm9 = _mm256_fmadd_pd( ymm7, ymm1_shuf, ymm9 );
            ymm11 = _mm256_add_pd( ymm11, ymm9 );

            // Storing back the results to the matrix
            _mm256_storeu_pd
            (
              (double*)( c + (j + 1)*rs_ct + (i)*cs_ct ),
              ymm10
            );
            _mm256_storeu_pd
            (
               (double*)( c + (j + 1)*rs_ct + (i + 2)*cs_ct ),
               ymm11
            );
        }

        for ( ; j < m; j++ )
        {
            // Broadcasting elements from x to ymm4-5
            ymm4 = _mm256_broadcast_pd( (__m128d const*)( x + (j)*incx ) );

            // Loading a tile from matrix
            ymm10 = _mm256_loadu_pd( (double*)( c + j*rs_ct + (i)*cs_ct ) );
            ymm11 = _mm256_loadu_pd( (double*)( c + j*rs_ct + (i + 2)*cs_ct ) );

            // Separating the real & imaginary parts of x into ymm4-7
            // ymm6 -> imag of ymm4
            // ymm4 -> real of ymm4
            ymm6 = _mm256_permute_pd( ymm4, 15 );
            ymm4 = _mm256_permute_pd( ymm4, 0 );
            ymm7 = _mm256_permute_pd( ymm5, 15 );
            ymm5 = _mm256_permute_pd( ymm5, 0 );

            // Applying conjugate to elements of x vector
            ymm6 = _mm256_mul_pd( ymm6, conj_mulv );
            ymm7 = _mm256_mul_pd( ymm7, conj_mulv );

            // Multiplying x vector with x hermitian vector
            // and adding the result to the corresponding tile
            ymm8 = _mm256_mul_pd( ymm4, ymm0 );
            ymm8 = _mm256_fmadd_pd( ymm6, ymm0_shuf, ymm8 );
            ymm10 = _mm256_add_pd( ymm10, ymm8 );

            ymm9 = _mm256_mul_pd( ymm4, ymm1 );
            ymm9 = _mm256_fmadd_pd( ymm6, ymm1_shuf, ymm9 );
            ymm11 = _mm256_add_pd( ymm11, ymm9 );

            // Storing back the results to the matrix
            _mm256_storeu_pd
            (
              (double*)( c + (j)*rs_ct + (i)*cs_ct ),
              ymm10
            );
            _mm256_storeu_pd
            (
              (double*)( c + (j)*rs_ct + (i + 2)*cs_ct ),
              ymm11
            );
        }
    }

    // Solving the remaining blocks of matrix
    for ( ; ( i + 1 ) < m; i += 2 )
    {
        // Solving the corner elements
        xc = x + (i + 1)*incx;
        xcR = xc->real;
        xcI = conj_multiplier * xc->imag;

        xhermc = x + i*incx;
        xhermcR = xhermc->real;
        xhermcI = -1 * conj_multiplier * xhermc->imag;

        xcR = alphaR * xcR;
        xcI = alphaR * xcI;
        interR = xcR * xhermcR - xcI * xhermcI;
        interI = xcR * xhermcI + xcI * xhermcR;

        cc = c + (i + 1)*rs_ct + i*cs_ct;
        cc->real += interR;
        cc->imag += interI;

        // Loading elements of x to ymm0 for computing xherm vector
        ymm0 = _mm256_loadu_pd( (double*)( x + i*incx ) );

        // Scaling xherm vector with alpha
        ymm0 = _mm256_mul_pd( ymm0, alphaRv );

        // Shuffling xherm vector for multiplication with x vector
        ymm0_shuf = _mm256_permute_pd( ymm0, 5 );

        /********* SQUARE BLOCK *********/
        // Solving a 2x2 square block of matrix using intrinsics
        for ( j = ( i + 2 ); j < m; j++ )
        {
            // Broadcasting elements from x to ymm4
            ymm4 = _mm256_broadcast_pd( (__m128d const*)( x + (j)*incx ) );

            // Loading a tile from matrix
            ymm10 = _mm256_loadu_pd( (double*)( c + (j)*rs_ct + (i)*cs_ct ) );

            // Separating the real & imaginary parts of x into ymm4-7
            // ymm6 -> imag of ymm4
            // ymm4 -> real of ymm4
            ymm6 = _mm256_permute_pd( ymm4, 15 );
            ymm4 = _mm256_permute_pd( ymm4, 0 );

            // Applying conjugate to elements of x vector
            ymm6 = _mm256_mul_pd( ymm6, conj_mulv );

            // Multiplying x vector with x hermitian vector
            // and adding the result to the corresponding tile
            ymm8 = _mm256_mul_pd( ymm4, ymm0 );
            ymm8 = _mm256_fmadd_pd( ymm6, ymm0_shuf, ymm8 );
            ymm10 = _mm256_add_pd( ymm10, ymm8 );

            // Storing back the results to the matrix
            _mm256_storeu_pd( (double*)( c + (j)*rs_ct + (i)*cs_ct ), ymm10 );
        }
    }
}

/**
 * Optimized implementation of ZHER for lower triangular column stored &
 * upper triangular row stored matrix.
 * This kernel performs:
 * A := A + conj?(alpha) * conj?(x) * conj?(x)^H
 * where,
 * 		A is an m x m hermitian matrix stored in upper/lower triangular
 * 		x is a vector of length m
 * 		alpha is a scalar
 */
void bli_zher_zen_int_var2
(
    uplo_t  uplo,
    conj_t  conjx,
    conj_t  conjh,
    dim_t   m,
    dcomplex*  alpha,
    dcomplex*  x, inc_t incx,
    dcomplex*  c, inc_t rs_c, inc_t cs_c,
    cntx_t* cntx
)
{
    double xcR, xcI;
    double xhermcR, xhermcI;
    double alphaR;
    double interR, interI;

    dcomplex* xc;
    dcomplex* xhermc;
    dcomplex* cc;

    __m256d alphaRv;
    __m256d ymm0, ymm1, ymm2, ymm3, ymm4, ymm5;
    __m256d ymm6, ymm7, ymm8, ymm9, ymm10, ymm11;
    __m256d ymm0_shuf, ymm1_shuf, ymm2_shuf, ymm3_shuf;

    dim_t conj_multiplier;

    inc_t rs_ct, cs_ct;
    dim_t i = 0;
    dim_t j = 0;

    alphaR = alpha->real;

    // The algorithm is expressed in terms of lower triangular case;
    // the upper triangular case is supported by swapping the row and column
    // strides of A & toggling the conj parameter.
    if ( bli_is_lower( uplo ) )
    {
        rs_ct = rs_c;
        cs_ct = cs_c;
    }
    else /* if ( bli_is_upper( uplo ) ) */
    {
        rs_ct = cs_c;
        cs_ct = rs_c;
        conjx = bli_apply_conj( conjh, conjx );
    }

    // Enabling conj_multiplier for scalar multiplication based on conjx
    if ( !bli_is_conj(conjx) ) conj_multiplier =  1;
    else                       conj_multiplier = -1;

    // Broadcasting real values of alpha based on conjx
    // alphaRv = aR aR aR aR
    if ( bli_is_conj( conjx ) ) alphaRv = _mm256_broadcast_sd( &alphaR );
    else                        alphaRv = _mm256_set_pd( -alphaR, alphaR, -alphaR, alphaR );

    __m256d conj_mulv = _mm256_set_pd
                        (
                          conj_multiplier,
                          -1 * conj_multiplier,
                          conj_multiplier,
                          -1 * conj_multiplier
                        );

    /********* DIAGONAL ELEMENTS *********/
    // Solving for the diagonal elements using a scalar loop
    for ( i = 0; i < m; i++ )
    {
        xc = x + i*incx;
        xcR = xc->real;
        xcI = xc->imag;
        xhermcR = xc->real;
        xhermcI = xc->imag;

        xcR = alphaR * xcR;
        xcI = alphaR * xcI;
        interR = xcR * xhermcR + xcI * xhermcI;

        cc = c + (i)*rs_ct + (i)*cs_ct;
        cc->real += interR;
        cc->imag = 0;
    }

    // Vectorized loop
    for ( i = 0; ( i + 3 ) < m; i += 4 )
    {
        // Broadcasting elements of x to ymm0-1 for computing xherm vector
        // ymm0 = x0R x0I x1R x1I
        ymm0 = _mm256_broadcast_pd( (__m128d const*)( x + i*incx ) );
        ymm1 = _mm256_broadcast_pd( (__m128d const*)( x + (i + 1)*incx ) );
        ymm2 = _mm256_broadcast_pd( (__m128d const*)( x + (i + 2)*incx ) );
        ymm3 = _mm256_broadcast_pd( (__m128d const*)( x + (i + 3)*incx ) );

        // Scaling xherm vector with alpha
        // alphaRv        = aR      aR     aR      aR
        // ymm0           = x0R    -x0I    x1R    -x1I
        // ymm0 * alphaRv = aR.x0R -aR.x0I aR.x1R -aR.x1I
        ymm0 = _mm256_mul_pd( ymm0, alphaRv );
        ymm1 = _mm256_mul_pd( ymm1, alphaRv );
        ymm2 = _mm256_mul_pd( ymm2, alphaRv );
        ymm3 = _mm256_mul_pd( ymm3, alphaRv );

        // Shuffling xherm vector for multiplication with x vector
        // ymm0_shuf = -x0I x0R -x1I x1R
        ymm0_shuf = _mm256_permute_pd( ymm0, 5 );
        ymm1_shuf = _mm256_permute_pd( ymm1, 5 );
        ymm2_shuf = _mm256_permute_pd( ymm2, 5 );
        ymm3_shuf = _mm256_permute_pd( ymm3, 5 );

        /********* TRIANGULAR BLOCK *********/
        // Solving the corner elements of the triangular block
        // using scalar multiplication
        xc = x + (i + 1)*incx;
        xcR = xc->real;
        xcI = conj_multiplier * xc->imag;

        xhermc = x + (i)*incx;
        xhermcR = xhermc->real;
        xhermcI = -1 * conj_multiplier * xhermc->imag;

        xcR = alphaR * xcR;
        xcI = alphaR * xcI;
        interR = xcR * xhermcR - xcI * xhermcI;
        interI = xcR * xhermcI + xcI * xhermcR;

        cc = c + (i + 1)*rs_ct + (i + 0)*cs_ct;
        cc->real += interR;
        cc->imag += interI;

        xc = x + (i + 3)*incx;
        xcR = xc->real;
        xcI = conj_multiplier * xc->imag;

        xhermc = x + (i + 2)*incx;
        xhermcR = xhermc->real;
        xhermcI = -1 * conj_multiplier * xhermc->imag;

        xcR = alphaR * xcR;
        xcI = alphaR * xcI;
        interR = xcR * xhermcR - xcI * xhermcI;
        interI = xcR * xhermcI + xcI * xhermcR;

        cc = c + (i + 3)*rs_ct + (i + 2)*cs_ct;
        cc->real += interR;
        cc->imag += interI;

        // Solving the 2x2 square tile inside the triangular block
        // using intrinsics
        // Loading elements from x to ymm4
        // ymm4 = x2R x2I x2R x2I
        ymm4 = _mm256_loadu_pd( (double*)( x + (i + 2)*incx ) );

        // Loading a tile from matrix
        // ymm10 = c20R c20I c21R c21I
        // ymm11 = c30R c30I c31R c31I
        ymm10 = _mm256_loadu_pd
                (
                  (double*)( c + (i + 2)*rs_ct + (i)*cs_ct )
                );
        ymm11 = _mm256_loadu_pd
                (
                  (double*)( c + (i + 2)*rs_ct + (i + 1)*cs_ct )
                );

        // Separating the real & imaginary parts of x into ymm4-7
        // ymm6 -> imag of ymm4
        // ymm4 -> real of ymm4
        ymm6 = _mm256_permute_pd( ymm4, 15 );
        ymm4 = _mm256_permute_pd( ymm4, 0 );

        // Applying conjugate to elements of x vector
        ymm6 = _mm256_mul_pd( ymm6, conj_mulv );

        // Multiplying x vector with x hermitian vector
        // and adding the result to the corresponding tile
        ymm8 = _mm256_mul_pd( ymm4, ymm0 );
        ymm8 = _mm256_fmadd_pd( ymm6, ymm0_shuf, ymm8 );
        ymm10 = _mm256_add_pd( ymm10, ymm8 );

        ymm9 = _mm256_mul_pd( ymm4, ymm1 );
        ymm9 = _mm256_fmadd_pd( ymm6, ymm1_shuf, ymm9 );
        ymm11 = _mm256_add_pd( ymm11, ymm9 );

        // Storing back the results to the matrix
        _mm256_storeu_pd
        (
          (double*)( c + (i + 2)*rs_ct + (i)*cs_ct ),
          ymm10
        );
        _mm256_storeu_pd
        (
          (double*)( c + (i + 2)*rs_ct + (i + 1)*cs_ct ),
          ymm11
        );

        /********* SQUARE BLOCK *********/
        // Solving a 4x4 square block of matrix using intrinsics
        for ( j = (i + 4); (j + 3) < m; j += 4)
        {
            // Loading elements from x to ymm4-5
            ymm4 = _mm256_loadu_pd( (double*)( x + j*incx ) );
            ymm5 = _mm256_loadu_pd( (double*)( x + (j + 2)*incx ) );

            // Separating the real & imaginary parts of x into ymm4-7
            // ymm6 -> imag of ymm4
            // ymm4 -> real of ymm4
            ymm6 = _mm256_permute_pd( ymm4, 15 );
            ymm4 = _mm256_permute_pd( ymm4, 0 );
            ymm7 = _mm256_permute_pd( ymm5, 15 );
            ymm5 = _mm256_permute_pd( ymm5, 0 );

            // Applying conjugate to elements of x vector
            ymm6 = _mm256_mul_pd( ymm6, conj_mulv );
            ymm7 = _mm256_mul_pd( ymm7, conj_mulv );

            // Loading a tile from matrix
            ymm10 = _mm256_loadu_pd
                    (
                      (double*)( c + (j)*rs_ct + (i)*cs_ct )
                    );
            ymm11 = _mm256_loadu_pd
                    (
                      (double*)( c + (j + 2)*rs_ct + (i)*cs_ct )
                    );

            // Multiplying x vector with x hermitian vector
            // and adding the result to the corresponding tile
            ymm8 = _mm256_mul_pd( ymm4, ymm0 );
            ymm9 = _mm256_mul_pd( ymm5, ymm0 );
            ymm8 = _mm256_fmadd_pd( ymm6, ymm0_shuf, ymm8 );
            ymm9 = _mm256_fmadd_pd( ymm7, ymm0_shuf, ymm9 );
            ymm10 = _mm256_add_pd( ymm10, ymm8 );
            ymm11 = _mm256_add_pd( ymm11, ymm9 );

            // Storing back the results to the matrix
            _mm256_storeu_pd
            (
              (double*)( c + (j)*rs_ct + (i)*cs_ct ),
              ymm10
            );
            _mm256_storeu_pd
            (
              (double*)( c + (j + 2)*rs_ct + (i)*cs_ct ),
              ymm11
            );

            // Loading a tile from matrix
            ymm10 = _mm256_loadu_pd
                    (
                      (double*)( c + (j)*rs_ct + (i + 1)*cs_ct )
                    );
            ymm11 = _mm256_loadu_pd
                    (
                      (double*)( c + (j + 2)*rs_ct + (i + 1)*cs_ct )
                    );

            // Multiplying x vector with x hermitian vector
            // and adding the result to the corresponding tile
            ymm8 = _mm256_mul_pd( ymm4, ymm1 );
            ymm9 = _mm256_mul_pd( ymm5, ymm1 );
            ymm8 = _mm256_fmadd_pd( ymm6, ymm1_shuf, ymm8 );
            ymm9 = _mm256_fmadd_pd( ymm7, ymm1_shuf, ymm9 );
            ymm10 = _mm256_add_pd( ymm10, ymm8 );
            ymm11 = _mm256_add_pd( ymm11, ymm9 );

            // Storing back the results to the matrix
            _mm256_storeu_pd
            (
              (double*)( c + (j)*rs_ct + (i + 1)*cs_ct ),
              ymm10
            );
            _mm256_storeu_pd
            (
              (double*)( c + (j + 2)*rs_ct + (i + 1)*cs_ct ),
              ymm11
            );

            // Loading a tile from matrix
            ymm10 = _mm256_loadu_pd
                    (
                      (double*)( c + (j)*rs_ct + (i + 2)*cs_ct )
                    );
            ymm11 = _mm256_loadu_pd
                    (
                      (double*)( c + (j + 2)*rs_ct + (i + 2)*cs_ct )
                    );

            // Multiplying x vector with x hermitian vector
            // and adding the result to the corresponding tile
            ymm8 = _mm256_mul_pd( ymm4, ymm2 );
            ymm9 = _mm256_mul_pd( ymm5, ymm2 );
            ymm8 = _mm256_fmadd_pd( ymm6, ymm2_shuf, ymm8 );
            ymm9 = _mm256_fmadd_pd( ymm7, ymm2_shuf, ymm9 );
            ymm10 = _mm256_add_pd( ymm10, ymm8 );
            ymm11 = _mm256_add_pd( ymm11, ymm9 );

            // Storing back the results to the matrix
            _mm256_storeu_pd
            (
              (double*)( c + (j)*rs_ct + (i + 2)*cs_ct ),
              ymm10
            );
            _mm256_storeu_pd
            (
              (double*)( c + (j + 2)*rs_ct + (i + 2)*cs_ct ),
              ymm11
            );

            // Loading a tile from matrix
            ymm10 = _mm256_loadu_pd
                    (
                      (double*)( c + (j)*rs_ct + (i + 3)*cs_ct )
                    );
            ymm11 = _mm256_loadu_pd
                    (
                      (double*)( c + (j + 2)*rs_ct + (i + 3)*cs_ct )
                    );

            // Multiplying x vector with x hermitian vector
            // and adding the result to the corresponding tile
            ymm8 = _mm256_mul_pd( ymm4, ymm3 );
            ymm9 = _mm256_mul_pd( ymm5, ymm3 );
            ymm8 = _mm256_fmadd_pd( ymm6, ymm3_shuf, ymm8 );
            ymm9 = _mm256_fmadd_pd( ymm7, ymm3_shuf, ymm9 );
            ymm10 = _mm256_add_pd( ymm10, ymm8 );
            ymm11 = _mm256_add_pd( ymm11, ymm9 );

            // Storing back the results to the matrix
            _mm256_storeu_pd
            (
              (double*)( c + (j)*rs_ct + (i + 3)*cs_ct ),
              ymm10
            );
            _mm256_storeu_pd
            (
              (double*)( c + (j + 2)*rs_ct + (i + 3)*cs_ct ),
              ymm11
            );
        }

        // Solving a 2x2 square block of matrix using intrinsics
        for ( ; (j + 1) < m; j += 2)
        {
            // Loading elements from x to ymm4
            ymm4 = _mm256_loadu_pd( (double*)( x + j*incx ) );

            // Separating the real & imaginary parts of x into ymm4-7
            // ymm6 -> imag of ymm4
            // ymm4 -> real of ymm4
            ymm6 = _mm256_permute_pd( ymm4, 15 );
            ymm4 = _mm256_permute_pd( ymm4, 0 );

            // Applying conjugate to elements of x vector
            ymm6 = _mm256_mul_pd( ymm6, conj_mulv );

            // Loading a tile from matrix
            ymm10 = _mm256_loadu_pd( (double*)( c + (j)*rs_ct + (i)*cs_ct ) );

            // Multiplying x vector with x hermitian vector
            // and adding the result to the corresponding tile
            ymm8 = _mm256_mul_pd( ymm4, ymm0 );
            ymm8 = _mm256_fmadd_pd( ymm6, ymm0_shuf, ymm8 );
            ymm10 = _mm256_add_pd( ymm10, ymm8 );

            // Storing back the results to the matrix
            _mm256_storeu_pd( (double*)( c + (j)*rs_ct + (i)*cs_ct ), ymm10 );

            // Loading a tile from matrix
            ymm10 = _mm256_loadu_pd( (double*)( c + j*rs_ct + (i + 1)*cs_ct ) );

            // Multiplying x vector with x hermitian vector
            // and adding the result to the corresponding tile
            ymm8 = _mm256_mul_pd( ymm4, ymm1 );
            ymm8 = _mm256_fmadd_pd( ymm6, ymm1_shuf, ymm8 );
            ymm10 = _mm256_add_pd( ymm10, ymm8 );

            // Storing back the results to the matrix
            _mm256_storeu_pd( (double*)( c + j*rs_ct + (i + 1)*cs_ct ), ymm10 );

            // Loading a tile from matrix
            ymm10 = _mm256_loadu_pd( (double*)( c + j*rs_ct + (i + 2)*cs_ct ) );

            // Multiplying x vector with x hermitian vector
            // and adding the result to the corresponding tile
            ymm8 = _mm256_mul_pd( ymm4, ymm2 );
            ymm8 = _mm256_fmadd_pd( ymm6, ymm2_shuf, ymm8 );
            ymm10 = _mm256_add_pd( ymm10, ymm8 );

            // Storing back the results to the matrix
            _mm256_storeu_pd( (double*)( c + j*rs_ct + (i + 2)*cs_ct ), ymm10 );

            // Loading a tile from matrix
            ymm10 = _mm256_loadu_pd( (double*)( c + j*rs_ct + (i + 3)*cs_ct ) );

            // Multiplying x vector with x hermitian vector
            // and adding the result to the corresponding tile
            ymm8 = _mm256_mul_pd( ymm4, ymm3 );
            ymm8 = _mm256_fmadd_pd( ymm6, ymm3_shuf, ymm8 );
            ymm10 = _mm256_add_pd( ymm10, ymm8 );

            // Storing back the results to the matrix
            _mm256_storeu_pd( (double*)( c + j*rs_ct + (i + 3)*cs_ct ), ymm10 );
        }

        // Calculating for the remaining elements using scalar code
        for ( ; j < m; j++ )
        {
            xc = x + j*incx;
            xcR = xc->real;
            xcI = conj_multiplier * xc->imag;

            xhermc = x + i*incx;
            xhermcR = xhermc->real;
            xhermcI = -1 * conj_multiplier * xhermc->imag;

            xcR = alphaR * xcR;
            xcI = alphaR * xcI;
            interR = xcR * xhermcR - xcI * xhermcI;
            interI = xcR * xhermcI + xcI * xhermcR;

            // c + ((alpha * x) * xherm)
            cc = c + (j)*rs_ct + (i)*cs_ct;
            cc->real += interR;
            cc->imag += interI;

            xc = x + j*incx;
            xcR = xc->real;
            xcI = conj_multiplier * xc->imag;

            xhermc = x + (i + 1)*incx;
            xhermcR = xhermc->real;
            xhermcI = -1 * conj_multiplier * xhermc->imag;

            xcR = alphaR * xcR;
            xcI = alphaR * xcI;
            interR = xcR * xhermcR - xcI * xhermcI;
            interI = xcR * xhermcI + xcI * xhermcR;

            // c + ((alpha * x) * xherm)
            cc = c + (j)*rs_ct + (i + 1)*cs_ct;
            cc->real += interR;
            cc->imag += interI;

            xc = x + j*incx;
            xcR = xc->real;
            xcI = conj_multiplier * xc->imag;

            xhermc = x + (i + 2)*incx;
            xhermcR = xhermc->real;
            xhermcI = -1 * conj_multiplier * xhermc->imag;

            xcR = alphaR * xcR;
            xcI = alphaR * xcI;
            interR = xcR * xhermcR - xcI * xhermcI;
            interI = xcR * xhermcI + xcI * xhermcR;

            // c + ((alpha * x) * xherm)
            cc = c + (j)*rs_ct + (i + 2)*cs_ct;
            cc->real += interR;
            cc->imag += interI;

            xc = x + j*incx;
            xcR = xc->real;
            xcI = conj_multiplier * xc->imag;

            xhermc = x + (i + 3)*incx;
            xhermcR = xhermc->real;
            xhermcI = -1 * conj_multiplier * xhermc->imag;

            xcR = alphaR * xcR;
            xcI = alphaR * xcI;
            interR = xcR * xhermcR - xcI * xhermcI;
            interI = xcR * xhermcI + xcI * xhermcR;

            // c + ((alpha * x) * xherm)
            cc = c + (j)*rs_ct + (i + 3)*cs_ct;
            cc->real += interR;
            cc->imag += interI;
        }
    }

    for ( ; ( i + 1 ) < m; i += 2 )
    {
        /********* TRIANGULAR BLOCK *********/
        // Solving the corner elements of the triangular block
        // using scalar multiplication
        xc = x + (i + 1)*incx;
        xcR = xc->real;
        xcI = conj_multiplier * xc->imag;

        xhermc = x + i*incx;
        xhermcR = xhermc->real;
        xhermcI = -1 * conj_multiplier * xhermc->imag;

        xcR = alphaR * xcR;
        xcI = alphaR * xcI;
        interR = xcR * xhermcR - xcI * xhermcI;
        interI = xcR * xhermcI + xcI * xhermcR;

        cc = c + (i + 1)*rs_ct + i*cs_ct;
        cc->real += interR;
        cc->imag += interI;

        // Solving the remaining elements in square block
        // using scalar code
        for ( j = (i + 2); j < m; j++ )
        {
            xc = x + j*incx;
            xcR = xc->real;
            xcI = conj_multiplier * xc->imag;

            xhermc = x + i*incx;
            xhermcR = xhermc->real;
            xhermcI = -1 * conj_multiplier * xhermc->imag;

            xcR = alphaR * xcR;
            xcI = alphaR * xcI;
            interR = xcR * xhermcR - xcI * xhermcI;
            interI = xcR * xhermcI + xcI * xhermcR;

            // c + ((alpha * x) * xherm)
            cc = c + (j)*rs_ct + (i)*cs_ct;
            cc->real += interR;
            cc->imag += interI;

            xc = x + j*incx;
            xcR = xc->real;
            xcI = conj_multiplier * xc->imag;

            xhermc = x + (i + 1)*incx;
            xhermcR = xhermc->real;
            xhermcI = -1 * conj_multiplier * xhermc->imag;

            xcR = alphaR * xcR;
            xcI = alphaR * xcI;
            interR = xcR * xhermcR - xcI * xhermcI;
            interI = xcR * xhermcI + xcI * xhermcR;

            // c + ((alpha * x) * xherm)
            cc = c + (j)*rs_ct + (i + 1)*cs_ct;
            cc->real += interR;
            cc->imag += interI;
        }
    }
}