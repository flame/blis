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

/* Union data structure to access AVX registers
   One 256-bit AVX register holds 8 SP elements. */
typedef union
{
    __m256  v;
    float   f[8] __attribute__((aligned(64)));
} v8sf_t;

/* Union data structure to access AVX registers
*  One 256-bit AVX register holds 4 DP elements. */
typedef union
{
    __m256d v;
    __m128d xmm[2];
    double  d[4] __attribute__((aligned(64)));
} v4df_t;

typedef union
{
    __m128d v;
    double  d[2] __attribute__((aligned(64)));
} v2df_t;


void bli_saxpyf_zen_int_6
     (
       conj_t           conja,
       conj_t           conjx,
       dim_t            m,
       dim_t            b_n,
       float* restrict alpha,
       float* restrict a, inc_t inca, inc_t lda,
       float* restrict x, inc_t incx,
       float* restrict y, inc_t incy,
       cntx_t* restrict cntx
     )
{
    const dim_t      fuse_fac       = 6;
    const dim_t      n_elem_per_reg = 8;

    dim_t            i;

    float* restrict a0;
    float* restrict y0;

    v8sf_t           chi0v, chi1v, chi2v, chi3v;
    v8sf_t           chi4v,chi5v;

    v8sf_t           a00v, a01v;

    v8sf_t           y0v;

    float           chi0, chi1, chi2, chi3;
    float           chi4,chi5;

    // If either dimension is zero, or if alpha is zero, return early.
    if ( bli_zero_dim2( m, b_n ) || bli_seq0( *alpha ) ) return;

    // If b_n is not equal to the fusing factor, then perform the entire
    // operation as a loop over axpyv.
    if ( b_n != fuse_fac )
    {
        saxpyv_ker_ft f = bli_cntx_get_l1v_ker_dt( BLIS_FLOAT, BLIS_AXPYV_KER, cntx );

        for ( i = 0; i < b_n; ++i )
        {
            float* a1   = a + (0  )*inca + (i  )*lda;
            float* chi1 = x + (i  )*incx;
            float* y1   = y + (0  )*incy;
            float  alpha_chi1;

            bli_scopycjs( conjx, *chi1, alpha_chi1 );
            bli_sscals( *alpha, alpha_chi1 );

            f
            (
              conja,
              m,
              &alpha_chi1,
              a1, inca,
              y1, incy,
              cntx
            );
        }

        return;
    }

    // At this point, we know that b_n is exactly equal to the fusing factor.
    a0   = a + 0*lda;
    y0   = y;

    // Scale each chi scalar by alpha.
    chi0 = *( x + 0*incx )*(*alpha);
    chi1 = *( x + 1*incx )*(*alpha);
    chi2 = *( x + 2*incx )*(*alpha);
    chi3 = *( x + 3*incx )*(*alpha);
    chi4 = *( x + 4*incx )*(*alpha);
    chi5 = *( x + 5*incx )*(*alpha);

    // Broadcast the (alpha*chi?) scalars to all elements of vector registers.
    chi0v.v = _mm256_broadcast_ss( &chi0 );
    chi1v.v = _mm256_broadcast_ss( &chi1 );
    chi2v.v = _mm256_broadcast_ss( &chi2 );
    chi3v.v = _mm256_broadcast_ss( &chi3 );
    chi4v.v = _mm256_broadcast_ss( &chi4 );
    chi5v.v = _mm256_broadcast_ss( &chi5 );

    // If there are vectorized iterations, perform them with vector
    // instructions.
    if ( inca == 1 && incy == 1 )
    {
        for( i=0; (i + 7) < m; i += 8 )
        {
            // Load the input values.
            y0v.v = _mm256_loadu_ps( y0 + 0*n_elem_per_reg );

            //Col_0
            a00v.v = _mm256_loadu_ps( a0 + 0*n_elem_per_reg );
            y0v.v = _mm256_fmadd_ps( a00v.v, chi0v.v, y0v.v );  // perform : y += alpha * x;

            //Col_1
            a01v.v = _mm256_loadu_ps( a0 + 1*lda );
            y0v.v = _mm256_fmadd_ps( a01v.v, chi1v.v, y0v.v );

            //Col_2
            a00v.v = _mm256_loadu_ps( a0 + 2*lda );
            y0v.v = _mm256_fmadd_ps( a00v.v, chi2v.v, y0v.v );

            //Col_3
            a01v.v = _mm256_loadu_ps( a0 + 3*lda );
            y0v.v = _mm256_fmadd_ps( a01v.v, chi3v.v, y0v.v );

            //Col_4
            a00v.v = _mm256_loadu_ps( a0 + 4*lda );
            y0v.v = _mm256_fmadd_ps( a00v.v, chi4v.v, y0v.v );

            //Col_5
            a01v.v = _mm256_loadu_ps( a0 + 5*lda );
            y0v.v = _mm256_fmadd_ps( a01v.v, chi5v.v, y0v.v );

            // Store the output.
            _mm256_storeu_ps( (y0 + 0*n_elem_per_reg), y0v.v );

            y0 += n_elem_per_reg;
            a0 += n_elem_per_reg;
        }
        // If there are leftover iterations, perform them with scalar code.
        for ( ; (i + 0) < m ; ++i )
        {
            float       y0c = *y0;

            const float a0c = *a0;
            const float a1c = *(a0+ 1*lda);
            const float a2c = *(a0+ 2*lda);
            const float a3c = *(a0+ 3*lda);
            const float a4c = *(a0+ 4*lda);
            const float a5c = *(a0+ 5*lda);

            y0c += chi0 * a0c;
            y0c += chi1 * a1c;
            y0c += chi2 * a2c;
            y0c += chi3 * a3c;
            y0c += chi4 * a4c;
            y0c += chi5 * a5c;

            *y0 = y0c;

            a0 += 1;
            y0 += 1;
        }
    }
    else
    {
        for ( i = 0; (i + 0) < m ; ++i )
        {
            float       y0c = *y0;
            const float a0c = *a0;
            const float a1c = *(a0+ 1*lda);
            const float a2c = *(a0+ 2*lda);
            const float a3c = *(a0+ 3*lda);
            const float a4c = *(a0+ 4*lda);
            const float a5c = *(a0+ 5*lda);

            y0c += chi0 * a0c;
            y0c += chi1 * a1c;
            y0c += chi2 * a2c;
            y0c += chi3 * a3c;
            y0c += chi4 * a4c;
            y0c += chi5 * a5c;

            *y0 = y0c;

            a0 += inca;
            y0 += incy;
        }
    }
}
