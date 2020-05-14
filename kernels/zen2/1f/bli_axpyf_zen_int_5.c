/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2020, Advanced Micro Devices, Inc.

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
    double  d[4] __attribute__((aligned(64)));
} v4df_t;

// -----------------------------------------------------------------------------

void bli_daxpyf_zen_int_5
     (
       conj_t           conja,
       conj_t           conjx,
       dim_t            m,
       dim_t            b_n,
       double* restrict alpha,
       double* restrict a, inc_t inca, inc_t lda,
       double* restrict x, inc_t incx,
       double* restrict y, inc_t incy,
       cntx_t* restrict cntx
     )
{
    const dim_t      fuse_fac       = 5;

    const dim_t      n_elem_per_reg = 4;
    const dim_t      n_iter_unroll  = 2;

    dim_t            i;

    double* restrict a0;
    double* restrict a1;
    double* restrict a2;
    double* restrict a3;
    double* restrict a4;

    double* restrict y0;

    v4df_t           chi0v, chi1v, chi2v, chi3v;
    v4df_t           chi4v;

    v4df_t           a00v, a01v, a02v, a03v;
    v4df_t           a04v;

    v4df_t           a10v, a11v, a12v, a13v;
    v4df_t           a14v;

    v4df_t           y0v, y1v;

    double           chi0, chi1, chi2, chi3;
    double           chi4;

    // If either dimension is zero, or if alpha is zero, return early.
    if ( bli_zero_dim2( m, b_n ) || PASTEMAC(d,eq0)( *alpha ) ) return;

    // If b_n is not equal to the fusing factor, then perform the entire
    // operation as a loop over axpyv.
    if ( b_n != fuse_fac )
    {
#ifdef BLIS_CONFIG_ZEN2
        for ( i = 0; i < b_n; ++i )
        {
            double* a1   = a + (0  )*inca + (i  )*lda;
            double* chi1 = x + (i  )*incx;
            double* y1   = y + (0  )*incy;
            double  alpha_chi1;

            PASTEMAC(d,copycjs)( conjx, *chi1, alpha_chi1 );
            PASTEMAC(d,scals)( *alpha, alpha_chi1 );

            bli_daxpyv_zen_int10
            (
              conja,
              m,
              &alpha_chi1,
              a1, inca,
              y1, incy,
              cntx
            );
        }

#else
        daxpyv_ker_ft f = bli_cntx_get_l1v_ker_dt( BLIS_DOUBLE, BLIS_AXPYV_KER, cntx );

        for ( i = 0; i < b_n; ++i )
        {
            double* a1   = a + (0  )*inca + (i  )*lda;
            double* chi1 = x + (i  )*incx;
            double* y1   = y + (0  )*incy;
            double  alpha_chi1;

            PASTEMAC(d,copycjs)( conjx, *chi1, alpha_chi1 );
            PASTEMAC(d,scals)( *alpha, alpha_chi1 );

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

#endif
        return;
    }

    // At this point, we know that b_n is exactly equal to the fusing factor.

    a0   = a + 0*lda;
    a1   = a + 1*lda;
    a2   = a + 2*lda;
    a3   = a + 3*lda;
    a4   = a + 4*lda;
    y0   = y;

    chi0 = *( x + 0*incx );
    chi1 = *( x + 1*incx );
    chi2 = *( x + 2*incx );
    chi3 = *( x + 3*incx );
    chi4 = *( x + 4*incx );


    // Scale each chi scalar by alpha.
    PASTEMAC(d,scals)( *alpha, chi0 );
    PASTEMAC(d,scals)( *alpha, chi1 );
    PASTEMAC(d,scals)( *alpha, chi2 );
    PASTEMAC(d,scals)( *alpha, chi3 );
    PASTEMAC(d,scals)( *alpha, chi4 );

    // Broadcast the (alpha*chi?) scalars to all elements of vector registers.
    chi0v.v = _mm256_broadcast_sd( &chi0 );
    chi1v.v = _mm256_broadcast_sd( &chi1 );
    chi2v.v = _mm256_broadcast_sd( &chi2 );
    chi3v.v = _mm256_broadcast_sd( &chi3 );
    chi4v.v = _mm256_broadcast_sd( &chi4 );

    // If there are vectorized iterations, perform them with vector
    // instructions.
    if ( inca == 1 && incy == 1 )
    {
        for ( i = 0; (i + 7) < m; i += 8 )
        {
            // Load the input values.
            y0v.v = _mm256_loadu_pd( y0 + 0*n_elem_per_reg );
            y1v.v = _mm256_loadu_pd( y0 + 1*n_elem_per_reg );

            a00v.v = _mm256_loadu_pd( a0 + 0*n_elem_per_reg );
            a10v.v = _mm256_loadu_pd( a0 + 1*n_elem_per_reg );

            a01v.v = _mm256_loadu_pd( a1 + 0*n_elem_per_reg );
            a11v.v = _mm256_loadu_pd( a1 + 1*n_elem_per_reg );

            a02v.v = _mm256_loadu_pd( a2 + 0*n_elem_per_reg );
            a12v.v = _mm256_loadu_pd( a2 + 1*n_elem_per_reg );

            a03v.v = _mm256_loadu_pd( a3 + 0*n_elem_per_reg );
            a13v.v = _mm256_loadu_pd( a3 + 1*n_elem_per_reg );

            a04v.v = _mm256_loadu_pd( a4 + 0*n_elem_per_reg );
            a14v.v = _mm256_loadu_pd( a4 + 1*n_elem_per_reg );

            // perform : y += alpha * x;
            y0v.v = _mm256_fmadd_pd( a00v.v, chi0v.v, y0v.v );
            y1v.v = _mm256_fmadd_pd( a10v.v, chi0v.v, y1v.v );

            y0v.v = _mm256_fmadd_pd( a01v.v, chi1v.v, y0v.v );
            y1v.v = _mm256_fmadd_pd( a11v.v, chi1v.v, y1v.v );

            y0v.v = _mm256_fmadd_pd( a02v.v, chi2v.v, y0v.v );
            y1v.v = _mm256_fmadd_pd( a12v.v, chi2v.v, y1v.v );

            y0v.v = _mm256_fmadd_pd( a03v.v, chi3v.v, y0v.v );
            y1v.v = _mm256_fmadd_pd( a13v.v, chi3v.v, y1v.v );

            y0v.v = _mm256_fmadd_pd( a04v.v, chi4v.v, y0v.v );
            y1v.v = _mm256_fmadd_pd( a14v.v, chi4v.v, y1v.v );


            // Store the output.
            _mm256_storeu_pd( (y0 + 0*n_elem_per_reg), y0v.v );
            _mm256_storeu_pd( (y0 + 1*n_elem_per_reg), y1v.v );

            y0 += n_iter_unroll * n_elem_per_reg;
            a0 += n_iter_unroll * n_elem_per_reg;
            a1 += n_iter_unroll * n_elem_per_reg;
            a2 += n_iter_unroll * n_elem_per_reg;
            a3 += n_iter_unroll * n_elem_per_reg;
            a4 += n_iter_unroll * n_elem_per_reg;
        }

        for( ; (i + 3) < m; i += 4 )
        {
            // Load the input values.
            y0v.v = _mm256_loadu_pd( y0 + 0*n_elem_per_reg );

            a00v.v = _mm256_loadu_pd( a0 + 0*n_elem_per_reg );
            a01v.v = _mm256_loadu_pd( a1 + 0*n_elem_per_reg );
            a02v.v = _mm256_loadu_pd( a2 + 0*n_elem_per_reg );
            a03v.v = _mm256_loadu_pd( a3 + 0*n_elem_per_reg );
            a04v.v = _mm256_loadu_pd( a4 + 0*n_elem_per_reg );


            // perform : y += alpha * x;
            y0v.v = _mm256_fmadd_pd( a00v.v, chi0v.v, y0v.v );
            y0v.v = _mm256_fmadd_pd( a01v.v, chi1v.v, y0v.v );
            y0v.v = _mm256_fmadd_pd( a02v.v, chi2v.v, y0v.v );
            y0v.v = _mm256_fmadd_pd( a03v.v, chi3v.v, y0v.v );
            y0v.v = _mm256_fmadd_pd( a04v.v, chi4v.v, y0v.v );

            // Store the output.
            _mm256_storeu_pd( (y0 + 0*n_elem_per_reg), y0v.v );

            y0 += n_elem_per_reg;
            a0 += n_elem_per_reg;
            a1 += n_elem_per_reg;
            a2 += n_elem_per_reg;
            a3 += n_elem_per_reg;
            a4 += n_elem_per_reg;
        }
    
        // If there are leftover iterations, perform them with scalar code.
        for ( ; (i + 0) < m ; ++i )
        {
            double       y0c = *y0;

            const double a0c = *a0;
            const double a1c = *a1;
            const double a2c = *a2;
            const double a3c = *a3;
            const double a4c = *a4;

            y0c += chi0 * a0c;
            y0c += chi1 * a1c;
            y0c += chi2 * a2c;
            y0c += chi3 * a3c;
            y0c += chi4 * a4c;

            *y0 = y0c;

            a0 += 1;
            a1 += 1;
            a2 += 1;
            a3 += 1;
            a4 += 1;
            y0 += 1;
        }
    }
    else
    {
        for ( i = 0; (i + 0) < m ; ++i )
        {
            double       y0c = *y0;

            const double a0c = *a0;
            const double a1c = *a1;
            const double a2c = *a2;
            const double a3c = *a3;
            const double a4c = *a4;

            y0c += chi0 * a0c;
            y0c += chi1 * a1c;
            y0c += chi2 * a2c;
            y0c += chi3 * a3c;
            y0c += chi4 * a4c;

            *y0 = y0c;

            a0 += inca;
            a1 += inca;
            a2 += inca;
            a3 += inca;
            a4 += inca; 
            y0 += incy;
        }

    }
}

