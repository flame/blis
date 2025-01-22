/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2020 - 2022, Advanced Micro Devices, Inc. All rights reserved.

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


void bli_saxpyf_zen_int_5
     (
             conj_t  conja,
             conj_t  conjx,
             dim_t   m,
             dim_t   b_n,
       const void*   alpha0,
       const void*   a0, inc_t inca, inc_t lda,
       const void*   x0, inc_t incx,
             void*   y0, inc_t incy,
       const cntx_t* cntx
     )
{
	const float* restrict alpha = alpha0;
	const float* restrict a     = a0;
	const float* restrict x     = x0;
	      float* restrict y     = y0;

    const dim_t      fuse_fac       = 5;

    const dim_t      n_elem_per_reg = 8;
    const dim_t      n_iter_unroll  = 2;

    dim_t            i;

    v8sf_t           chi0v, chi1v, chi2v, chi3v;
    v8sf_t           chi4v;

    v8sf_t           a00v, a01v, a02v, a03v;
    v8sf_t           a04v;

    v8sf_t           a10v, a11v, a12v, a13v;
    v8sf_t           a14v;

    v8sf_t           y0v, y1v;

    float           chi0, chi1, chi2, chi3;
    float           chi4;

    // If either dimension is zero, or if alpha is zero, return early.
    if ( bli_zero_dim2( m, b_n ) || bli_seq0( *alpha ) ) return;

    // If b_n is not equal to the fusing factor, then perform the entire
    // operation as a loop over axpyv.
    if ( b_n != fuse_fac )
    {
        if ( cntx == NULL ) cntx = ( cntx_t* )bli_gks_query_cntx();

        axpyv_ker_ft f = bli_cntx_get_ukr_dt( BLIS_FLOAT, BLIS_AXPYV_KER, cntx );

        for ( i = 0; i < b_n; ++i )
        {
            const float* restrict ap1   = a + (0  )*inca + (i  )*lda;
            const float* restrict chi1 = x + (i  )*incx;
                  float* restrict y1   = y + (0  )*incy;
                  float           alpha_chi1;

            bli_scopycjs( conjx, *chi1, alpha_chi1 );
            bli_sscals( *alpha, alpha_chi1 );

            f
            (
              conja,
              m,
              &alpha_chi1,
              ap1, inca,
              y1, incy,
              cntx
            );
        }

        return;
    }

    // At this point, we know that b_n is exactly equal to the fusing factor.

    const float* restrict ap0   = a + 0*lda;
    const float* restrict ap1   = a + 1*lda;
    const float* restrict ap2   = a + 2*lda;
    const float* restrict ap3   = a + 3*lda;
    const float* restrict ap4   = a + 4*lda;
          float*          yp   = y;

    chi0 = *( x + 0*incx );
    chi1 = *( x + 1*incx );
    chi2 = *( x + 2*incx );
    chi3 = *( x + 3*incx );
    chi4 = *( x + 4*incx );


    // Scale each chi scalar by alpha.
    bli_sscals( *alpha, chi0 );
    bli_sscals( *alpha, chi1 );
    bli_sscals( *alpha, chi2 );
    bli_sscals( *alpha, chi3 );
    bli_sscals( *alpha, chi4 );

    // Broadcast the (alpha*chi?) scalars to all elements of vector registers.
    chi0v.v = _mm256_broadcast_ss( &chi0 );
    chi1v.v = _mm256_broadcast_ss( &chi1 );
    chi2v.v = _mm256_broadcast_ss( &chi2 );
    chi3v.v = _mm256_broadcast_ss( &chi3 );
    chi4v.v = _mm256_broadcast_ss( &chi4 );

    // If there are vectorized iterations, perform them with vector
    // instructions.
    if ( inca == 1 && incy == 1 )
    {
        for ( i = 0; (i + 15) < m; i += 16 )
        {
            // Load the input values.
            y0v.v = _mm256_loadu_ps( yp + 0*n_elem_per_reg );
            y1v.v = _mm256_loadu_ps( yp + 1*n_elem_per_reg );

            a00v.v = _mm256_loadu_ps( ap0 + 0*n_elem_per_reg );
            a10v.v = _mm256_loadu_ps( ap0 + 1*n_elem_per_reg );

            a01v.v = _mm256_loadu_ps( ap1 + 0*n_elem_per_reg );
            a11v.v = _mm256_loadu_ps( ap1 + 1*n_elem_per_reg );

            a02v.v = _mm256_loadu_ps( ap2 + 0*n_elem_per_reg );
            a12v.v = _mm256_loadu_ps( ap2 + 1*n_elem_per_reg );

            a03v.v = _mm256_loadu_ps( ap3 + 0*n_elem_per_reg );
            a13v.v = _mm256_loadu_ps( ap3 + 1*n_elem_per_reg );

            a04v.v = _mm256_loadu_ps( ap4 + 0*n_elem_per_reg );
            a14v.v = _mm256_loadu_ps( ap4 + 1*n_elem_per_reg );

            // perform : y += alpha * x;
            y0v.v = _mm256_fmadd_ps( a00v.v, chi0v.v, y0v.v );
            y1v.v = _mm256_fmadd_ps( a10v.v, chi0v.v, y1v.v );

            y0v.v = _mm256_fmadd_ps( a01v.v, chi1v.v, y0v.v );
            y1v.v = _mm256_fmadd_ps( a11v.v, chi1v.v, y1v.v );

            y0v.v = _mm256_fmadd_ps( a02v.v, chi2v.v, y0v.v );
            y1v.v = _mm256_fmadd_ps( a12v.v, chi2v.v, y1v.v );

            y0v.v = _mm256_fmadd_ps( a03v.v, chi3v.v, y0v.v );
            y1v.v = _mm256_fmadd_ps( a13v.v, chi3v.v, y1v.v );

            y0v.v = _mm256_fmadd_ps( a04v.v, chi4v.v, y0v.v );
            y1v.v = _mm256_fmadd_ps( a14v.v, chi4v.v, y1v.v );


            // Store the output.
            _mm256_storeu_ps( (yp + 0*n_elem_per_reg), y0v.v );
            _mm256_storeu_ps( (yp + 1*n_elem_per_reg), y1v.v );

            yp += n_iter_unroll * n_elem_per_reg;
            ap0 += n_iter_unroll * n_elem_per_reg;
            ap1 += n_iter_unroll * n_elem_per_reg;
            ap2 += n_iter_unroll * n_elem_per_reg;
            ap3 += n_iter_unroll * n_elem_per_reg;
            ap4 += n_iter_unroll * n_elem_per_reg;
        }

        for( ; (i + 7) < m; i += 8 )
        {
            // Load the input values.
            y0v.v = _mm256_loadu_ps( yp + 0*n_elem_per_reg );

            a00v.v = _mm256_loadu_ps( ap0 + 0*n_elem_per_reg );
            a01v.v = _mm256_loadu_ps( ap1 + 0*n_elem_per_reg );
            a02v.v = _mm256_loadu_ps( ap2 + 0*n_elem_per_reg );
            a03v.v = _mm256_loadu_ps( ap3 + 0*n_elem_per_reg );
            a04v.v = _mm256_loadu_ps( ap4 + 0*n_elem_per_reg );


            // perform : y += alpha * x;
            y0v.v = _mm256_fmadd_ps( a00v.v, chi0v.v, y0v.v );
            y0v.v = _mm256_fmadd_ps( a01v.v, chi1v.v, y0v.v );
            y0v.v = _mm256_fmadd_ps( a02v.v, chi2v.v, y0v.v );
            y0v.v = _mm256_fmadd_ps( a03v.v, chi3v.v, y0v.v );
            y0v.v = _mm256_fmadd_ps( a04v.v, chi4v.v, y0v.v );

            // Store the output.
            _mm256_storeu_ps( (yp + 0*n_elem_per_reg), y0v.v );

            yp += n_elem_per_reg;
            ap0 += n_elem_per_reg;
            ap1 += n_elem_per_reg;
            ap2 += n_elem_per_reg;
            ap3 += n_elem_per_reg;
            ap4 += n_elem_per_reg;
        }

        // If there are leftover iterations, perform them with scalar code.
        for ( ; (i + 0) < m ; ++i )
        {
            double       y0c = *yp;

            const float a0c = *ap0;
            const float a1c = *ap1;
            const float a2c = *ap2;
            const float a3c = *ap3;
            const float a4c = *ap4;

            y0c += chi0 * a0c;
            y0c += chi1 * a1c;
            y0c += chi2 * a2c;
            y0c += chi3 * a3c;
            y0c += chi4 * a4c;

            *yp = y0c;

            ap0 += 1;
            ap1 += 1;
            ap2 += 1;
            ap3 += 1;
            ap4 += 1;
            yp += 1;
        }
    }
    else
    {
        for ( i = 0; (i + 0) < m ; ++i )
        {
            double       y0c = *yp;

            const float a0c = *ap0;
            const float a1c = *ap1;
            const float a2c = *ap2;
            const float a3c = *ap3;
            const float a4c = *ap4;

            y0c += chi0 * a0c;
            y0c += chi1 * a1c;
            y0c += chi2 * a2c;
            y0c += chi3 * a3c;
            y0c += chi4 * a4c;

            *yp = y0c;

            ap0 += inca;
            ap1 += inca;
            ap2 += inca;
            ap3 += inca;
            ap4 += inca;
            yp += incy;
        }

    }
}


// -----------------------------------------------------------------------------

void bli_daxpyf_zen_int_5
     (
             conj_t  conja,
             conj_t  conjx,
             dim_t   m,
             dim_t   b_n,
       const void*   alpha0,
       const void*   a0, inc_t inca, inc_t lda,
       const void*   x0, inc_t incx,
             void*   y0, inc_t incy,
       const cntx_t* cntx
     )
{
	const double* restrict alpha = alpha0;
	const double* restrict a     = a0;
	const double* restrict x     = x0;
	      double* restrict y     = y0;

    const dim_t      fuse_fac       = 5;

    const dim_t      n_elem_per_reg = 4;
    const dim_t      n_iter_unroll  = 2;

    dim_t            i;

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
    if ( bli_zero_dim2( m, b_n ) || bli_deq0( *alpha ) ) return;

    // If b_n is not equal to the fusing factor, then perform the entire
    // operation as a loop over axpyv.
    if ( b_n != fuse_fac )
    {
        if ( cntx == NULL ) cntx = ( cntx_t* )bli_gks_query_cntx();

        axpyv_ker_ft f = bli_cntx_get_ukr_dt( BLIS_DOUBLE, BLIS_AXPYV_KER, cntx );

        for ( i = 0; i < b_n; ++i )
        {
            const double* restrict ap1   = a + (0  )*inca + (i  )*lda;
            const double* restrict chi1 = x + (i  )*incx;
                  double* restrict y1   = y + (0  )*incy;
                  double           alpha_chi1;

            bli_dcopycjs( conjx, *chi1, alpha_chi1 );
            bli_dscals( *alpha, alpha_chi1 );

            f
            (
              conja,
              m,
              &alpha_chi1,
              ap1, inca,
              y1, incy,
              cntx
            );
        }

        return;
    }

    // At this point, we know that b_n is exactly equal to the fusing factor.

    const double* restrict ap0   = a + 0*lda;
    const double* restrict ap1   = a + 1*lda;
    const double* restrict ap2   = a + 2*lda;
    const double* restrict ap3   = a + 3*lda;
    const double* restrict ap4   = a + 4*lda;
          double* restrict yp   = y;

    chi0 = *( x + 0*incx );
    chi1 = *( x + 1*incx );
    chi2 = *( x + 2*incx );
    chi3 = *( x + 3*incx );
    chi4 = *( x + 4*incx );


    // Scale each chi scalar by alpha.
    bli_dscals( *alpha, chi0 );
    bli_dscals( *alpha, chi1 );
    bli_dscals( *alpha, chi2 );
    bli_dscals( *alpha, chi3 );
    bli_dscals( *alpha, chi4 );

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
            y0v.v = _mm256_loadu_pd( yp + 0*n_elem_per_reg );
            y1v.v = _mm256_loadu_pd( yp + 1*n_elem_per_reg );

            a00v.v = _mm256_loadu_pd( ap0 + 0*n_elem_per_reg );
            a10v.v = _mm256_loadu_pd( ap0 + 1*n_elem_per_reg );

            a01v.v = _mm256_loadu_pd( ap1 + 0*n_elem_per_reg );
            a11v.v = _mm256_loadu_pd( ap1 + 1*n_elem_per_reg );

            a02v.v = _mm256_loadu_pd( ap2 + 0*n_elem_per_reg );
            a12v.v = _mm256_loadu_pd( ap2 + 1*n_elem_per_reg );

            a03v.v = _mm256_loadu_pd( ap3 + 0*n_elem_per_reg );
            a13v.v = _mm256_loadu_pd( ap3 + 1*n_elem_per_reg );

            a04v.v = _mm256_loadu_pd( ap4 + 0*n_elem_per_reg );
            a14v.v = _mm256_loadu_pd( ap4 + 1*n_elem_per_reg );

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
            _mm256_storeu_pd( (double *)(yp + 0*n_elem_per_reg), y0v.v );
            _mm256_storeu_pd( (double *)(yp + 1*n_elem_per_reg), y1v.v );

            yp += n_iter_unroll * n_elem_per_reg;
            ap0 += n_iter_unroll * n_elem_per_reg;
            ap1 += n_iter_unroll * n_elem_per_reg;
            ap2 += n_iter_unroll * n_elem_per_reg;
            ap3 += n_iter_unroll * n_elem_per_reg;
            ap4 += n_iter_unroll * n_elem_per_reg;
        }

        for( ; (i + 3) < m; i += 4 )
        {
            // Load the input values.
            y0v.v = _mm256_loadu_pd( yp + 0*n_elem_per_reg );

            a00v.v = _mm256_loadu_pd( ap0 + 0*n_elem_per_reg );
            a01v.v = _mm256_loadu_pd( ap1 + 0*n_elem_per_reg );
            a02v.v = _mm256_loadu_pd( ap2 + 0*n_elem_per_reg );
            a03v.v = _mm256_loadu_pd( ap3 + 0*n_elem_per_reg );
            a04v.v = _mm256_loadu_pd( ap4 + 0*n_elem_per_reg );


            // perform : y += alpha * x;
            y0v.v = _mm256_fmadd_pd( a00v.v, chi0v.v, y0v.v );
            y0v.v = _mm256_fmadd_pd( a01v.v, chi1v.v, y0v.v );
            y0v.v = _mm256_fmadd_pd( a02v.v, chi2v.v, y0v.v );
            y0v.v = _mm256_fmadd_pd( a03v.v, chi3v.v, y0v.v );
            y0v.v = _mm256_fmadd_pd( a04v.v, chi4v.v, y0v.v );

            // Store the output.
            _mm256_storeu_pd( (yp + 0*n_elem_per_reg), y0v.v );

            yp += n_elem_per_reg;
            ap0 += n_elem_per_reg;
            ap1 += n_elem_per_reg;
            ap2 += n_elem_per_reg;
            ap3 += n_elem_per_reg;
            ap4 += n_elem_per_reg;
        }

        // If there are leftover iterations, perform them with scalar code.
        for ( ; (i + 0) < m ; ++i )
        {
            double       y0c = *yp;

            const double a0c = *ap0;
            const double a1c = *ap1;
            const double a2c = *ap2;
            const double a3c = *ap3;
            const double a4c = *ap4;

            y0c += chi0 * a0c;
            y0c += chi1 * a1c;
            y0c += chi2 * a2c;
            y0c += chi3 * a3c;
            y0c += chi4 * a4c;

            *yp = y0c;

            ap0 += 1;
            ap1 += 1;
            ap2 += 1;
            ap3 += 1;
            ap4 += 1;
            yp += 1;
        }
    }
    else
    {
        for ( i = 0; (i + 0) < m ; ++i )
        {
            double       y0c = *yp;

            const double a0c = *ap0;
            const double a1c = *ap1;
            const double a2c = *ap2;
            const double a3c = *ap3;
            const double a4c = *ap4;

            y0c += chi0 * a0c;
            y0c += chi1 * a1c;
            y0c += chi2 * a2c;
            y0c += chi3 * a3c;
            y0c += chi4 * a4c;

            *yp = y0c;

            ap0 += inca;
            ap1 += inca;
            ap2 += inca;
            ap3 += inca;
            ap4 += inca;
            yp += incy;
        }

    }
}

// -----------------------------------------------------------------------------

void bli_daxpyf_zen_int_16x2
     (
             conj_t  conja,
             conj_t  conjx,
             dim_t   m,
             dim_t   b_n,
       const void*   alpha0,
       const void*   a0, inc_t inca, inc_t lda,
       const void*   x0, inc_t incx,
             void*   y0, inc_t incy,
       const cntx_t* cntx
     )
{
	const double* restrict alpha = alpha0;
	const double* restrict a     = a0;
	const double* restrict x     = x0;
	      double* restrict y     = y0;

    const dim_t      fuse_fac       = 2;

    const dim_t      n_elem_per_reg = 4;
    const dim_t      n_iter_unroll  = 4;

    dim_t            i;

    v4df_t           chi0v, chi1v;

    v4df_t           a00v, a01v;

    v4df_t           a10v, a11v;

    v4df_t           a20v, a21v;

    v4df_t           a30v, a31v;

    v4df_t           y0v, y1v, y2v, y3v;

    double           chi0, chi1;

    v2df_t           a40v, a41v;

    v2df_t           y4v; 

    // If either dimension is zero, or if alpha is zero, return early.
    if ( bli_zero_dim2( m, b_n ) || bli_deq0( *alpha ) ) return;

    // If b_n is not equal to the fusing factor, then perform the entire
    // operation as a loop over axpyv.
    if ( b_n != fuse_fac )
    {
        if ( cntx == NULL ) cntx = ( cntx_t* )bli_gks_query_cntx();

        axpyv_ker_ft f = bli_cntx_get_ukr_dt( BLIS_DOUBLE, BLIS_AXPYV_KER, cntx );

        for ( i = 0; i < b_n; ++i )
        {
            const double* restrict ap1   = a + (0  )*inca + (i  )*lda;
            const double* restrict chi1 = x + (i  )*incx;
                  double* restrict y1   = y + (0  )*incy;
                  double           alpha_chi1;

            bli_dcopycjs( conjx, *chi1, alpha_chi1 );
            bli_dscals( *alpha, alpha_chi1 );

            f
            (
              conja,
              m,
              &alpha_chi1,
              ap1, inca,
              y1, incy,
              cntx
            );
        }

        return;
    }

    // At this point, we know that b_n is exactly equal to the fusing factor.

    const double* restrict ap0   = a + 0*lda;
    const double* restrict ap1   = a + 1*lda;

          double* restrict yp   = y;

    chi0 = *( x + 0*incx );
    chi1 = *( x + 1*incx );


    // Scale each chi scalar by alpha.
    bli_dscals( *alpha, chi0 );
    bli_dscals( *alpha, chi1 );

    // Broadcast the (alpha*chi?) scalars to all elements of vector registers.
    chi0v.v = _mm256_broadcast_sd( &chi0 );
    chi1v.v = _mm256_broadcast_sd( &chi1 );

    // If there are vectorized iterations, perform them with vector
    // instructions.
    if ( inca == 1 && incy == 1 )
    {
        for ( i = 0; (i + 15) < m; i += 16 )
        {
            // Load the input values.
            y0v.v = _mm256_loadu_pd( yp + 0*n_elem_per_reg );
            y1v.v = _mm256_loadu_pd( yp + 1*n_elem_per_reg );
            y2v.v = _mm256_loadu_pd( yp + 2*n_elem_per_reg );
            y3v.v = _mm256_loadu_pd( yp + 3*n_elem_per_reg );

            a00v.v = _mm256_loadu_pd( ap0 + 0*n_elem_per_reg );
            a10v.v = _mm256_loadu_pd( ap0 + 1*n_elem_per_reg );
            a20v.v = _mm256_loadu_pd( ap0 + 2*n_elem_per_reg );
            a30v.v = _mm256_loadu_pd( ap0 + 3*n_elem_per_reg );

            a01v.v = _mm256_loadu_pd( ap1 + 0*n_elem_per_reg );
            a11v.v = _mm256_loadu_pd( ap1 + 1*n_elem_per_reg );
            a21v.v = _mm256_loadu_pd( ap1 + 2*n_elem_per_reg );
            a31v.v = _mm256_loadu_pd( ap1 + 3*n_elem_per_reg );

            // perform : y += alpha * x;
            y0v.v = _mm256_fmadd_pd( a00v.v, chi0v.v, y0v.v );
            y1v.v = _mm256_fmadd_pd( a10v.v, chi0v.v, y1v.v );
            y2v.v = _mm256_fmadd_pd( a20v.v, chi0v.v, y2v.v );
            y3v.v = _mm256_fmadd_pd( a30v.v, chi0v.v, y3v.v );

            y0v.v = _mm256_fmadd_pd( a01v.v, chi1v.v, y0v.v );
            y1v.v = _mm256_fmadd_pd( a11v.v, chi1v.v, y1v.v );
            y2v.v = _mm256_fmadd_pd( a21v.v, chi1v.v, y2v.v );
            y3v.v = _mm256_fmadd_pd( a31v.v, chi1v.v, y3v.v );

            // Store the output.
            _mm256_storeu_pd( (double *)(yp + 0*n_elem_per_reg), y0v.v );
            _mm256_storeu_pd( (double *)(yp + 1*n_elem_per_reg), y1v.v );
            _mm256_storeu_pd( (double *)(yp + 2*n_elem_per_reg), y2v.v );
            _mm256_storeu_pd( (double *)(yp + 3*n_elem_per_reg), y3v.v );

            yp += n_iter_unroll * n_elem_per_reg;
            ap0 += n_iter_unroll * n_elem_per_reg;
            ap1 += n_iter_unroll * n_elem_per_reg;
        }

        for ( ; (i + 11) < m; i += 12 )
        {
            // Load the input values.
            y0v.v = _mm256_loadu_pd( yp + 0*n_elem_per_reg );
            y1v.v = _mm256_loadu_pd( yp + 1*n_elem_per_reg );
            y2v.v = _mm256_loadu_pd( yp + 2*n_elem_per_reg );

            a00v.v = _mm256_loadu_pd( ap0 + 0*n_elem_per_reg );
            a10v.v = _mm256_loadu_pd( ap0 + 1*n_elem_per_reg );
            a20v.v = _mm256_loadu_pd( ap0 + 2*n_elem_per_reg );

            a01v.v = _mm256_loadu_pd( ap1 + 0*n_elem_per_reg );
            a11v.v = _mm256_loadu_pd( ap1 + 1*n_elem_per_reg );
            a21v.v = _mm256_loadu_pd( ap1 + 2*n_elem_per_reg );

            // perform : y += alpha * x;
            y0v.v = _mm256_fmadd_pd( a00v.v, chi0v.v, y0v.v );
            y1v.v = _mm256_fmadd_pd( a10v.v, chi0v.v, y1v.v );
            y2v.v = _mm256_fmadd_pd( a20v.v, chi0v.v, y2v.v );

            y0v.v = _mm256_fmadd_pd( a01v.v, chi1v.v, y0v.v );
            y1v.v = _mm256_fmadd_pd( a11v.v, chi1v.v, y1v.v );
            y2v.v = _mm256_fmadd_pd( a21v.v, chi1v.v, y2v.v );

            // Store the output.
            _mm256_storeu_pd( (double *)(yp + 0*n_elem_per_reg), y0v.v );
            _mm256_storeu_pd( (double *)(yp + 1*n_elem_per_reg), y1v.v );
            _mm256_storeu_pd( (double *)(yp + 2*n_elem_per_reg), y2v.v );

            yp += 3 * n_elem_per_reg;
            ap0 += 3 * n_elem_per_reg;
            ap1 += 3 * n_elem_per_reg;
        }
        for ( ; (i + 7) < m; i += 8 )
        {
            // Load the input values.
            y0v.v = _mm256_loadu_pd( yp + 0*n_elem_per_reg );
            y1v.v = _mm256_loadu_pd( yp + 1*n_elem_per_reg );

            a00v.v = _mm256_loadu_pd( ap0 + 0*n_elem_per_reg );
            a10v.v = _mm256_loadu_pd( ap0 + 1*n_elem_per_reg );

            a01v.v = _mm256_loadu_pd( ap1 + 0*n_elem_per_reg );
            a11v.v = _mm256_loadu_pd( ap1 + 1*n_elem_per_reg );

            // perform : y += alpha * x;
            y0v.v = _mm256_fmadd_pd( a00v.v, chi0v.v, y0v.v );
            y1v.v = _mm256_fmadd_pd( a10v.v, chi0v.v, y1v.v );

            y0v.v = _mm256_fmadd_pd( a01v.v, chi1v.v, y0v.v );
            y1v.v = _mm256_fmadd_pd( a11v.v, chi1v.v, y1v.v );

            // Store the output.
            _mm256_storeu_pd( (double *)(yp + 0*n_elem_per_reg), y0v.v );
            _mm256_storeu_pd( (double *)(yp + 1*n_elem_per_reg), y1v.v );

            yp += 2 * n_elem_per_reg;
            ap0 += 2 * n_elem_per_reg;
            ap1 += 2 * n_elem_per_reg;
        }

        for ( ; (i + 3) < m; i += 4 )
        {
            // Load the input values.
            y0v.v = _mm256_loadu_pd( yp + 0*n_elem_per_reg );

            a00v.v = _mm256_loadu_pd( ap0 + 0*n_elem_per_reg );

            a01v.v = _mm256_loadu_pd( ap1 + 0*n_elem_per_reg );

            // perform : y += alpha * x;
            y0v.v = _mm256_fmadd_pd( a00v.v, chi0v.v, y0v.v );

            y0v.v = _mm256_fmadd_pd( a01v.v, chi1v.v, y0v.v );

            // Store the output.
            _mm256_storeu_pd( (double *)(yp + 0*n_elem_per_reg), y0v.v );

            yp += n_elem_per_reg;
            ap0 += n_elem_per_reg;
            ap1 += n_elem_per_reg;
        }

        for ( ; (i + 1) < m; i += 2 )
        {
            // Load the input values.
            y4v.v = _mm_loadu_pd( yp + 0*n_elem_per_reg );

            a40v.v = _mm_loadu_pd( ap0 + 0*n_elem_per_reg );

            a41v.v = _mm_loadu_pd( ap1 + 0*n_elem_per_reg );

            // perform : y += alpha * x;
            y4v.v = _mm_fmadd_pd( a40v.v, chi0v.xmm[0], y4v.v );

            y4v.v = _mm_fmadd_pd( a41v.v, chi1v.xmm[0], y4v.v );

            // Store the output.
            _mm_storeu_pd( (double *)(yp + 0*n_elem_per_reg), y4v.v );

            yp += 2;
            ap0 += 2;
            ap1 += 2;
        }

        // If there are leftover iterations, perform them with scalar code.
        for ( ; (i + 0) < m ; ++i )
        {
            double       y0c = *yp;

            const double a0c = *ap0;
            const double a1c = *ap1;

            y0c += chi0 * a0c;
            y0c += chi1 * a1c;

            *yp = y0c;

            ap0 += 1;
            ap1 += 1;
            yp += 1;
        }
    }
    else
    {
        for ( i = 0; (i + 0) < m ; ++i )
        {
            double       y0c = *yp;

            const double a0c = *ap0;
            const double a1c = *ap1;

            y0c += chi0 * a0c;
            y0c += chi1 * a1c;

            *yp = y0c;

            ap0 += inca;
            ap1 += inca;
            yp += incy;
        }

    }
}

// -----------------------------------------------------------------------------

void bli_daxpyf_zen_int_16x4
     (
             conj_t  conja,
             conj_t  conjx,
             dim_t   m,
             dim_t   b_n,
       const void*   alpha0,
       const void*   a0, inc_t inca, inc_t lda,
       const void*   x0, inc_t incx,
             void*   y0, inc_t incy,
       const cntx_t* cntx
     )
{
	const double* restrict alpha = alpha0;
	const double* restrict a     = a0;
	const double* restrict x     = x0;
	      double* restrict y     = y0;

    const dim_t      fuse_fac       = 4;

    const dim_t      n_elem_per_reg = 4;
    const dim_t      n_iter_unroll  = 4;

    dim_t            i;

    v4df_t           chi0v, chi1v, chi2v, chi3v;

    v4df_t           a00v, a01v, a02v, a03v;

    v4df_t           a10v, a11v, a12v, a13v;

    v4df_t           a20v, a21v, a22v, a23v;

    v4df_t           a30v, a31v, a32v, a33v;

    v4df_t           y0v, y1v, y2v, y3v;

    double           chi0, chi1, chi2, chi3;

    v2df_t           y4v;

    v2df_t           a40v, a41v, a42v, a43v;

    // If either dimension is zero, or if alpha is zero, return early.
    if ( bli_zero_dim2( m, b_n ) || bli_deq0( *alpha ) ) return;

    // If b_n is not equal to the fusing factor, then perform the entire
    // operation as a loop over axpyv.
    if ( b_n != fuse_fac )
    {
        if ( cntx == NULL ) cntx = ( cntx_t* )bli_gks_query_cntx();

        axpyv_ker_ft f = bli_cntx_get_ukr_dt( BLIS_DOUBLE, BLIS_AXPYV_KER, cntx );

        for ( i = 0; i < b_n; ++i )
        {
            const double* restrict ap1   = a + (0  )*inca + (i  )*lda;
            const double* restrict chi1 = x + (i  )*incx;
                  double* restrict y1   = y + (0  )*incy;
                  double           alpha_chi1;

            bli_dcopycjs( conjx, *chi1, alpha_chi1 );
            bli_dscals( *alpha, alpha_chi1 );

            f
            (
              conja,
              m,
              &alpha_chi1,
              ap1, inca,
              y1, incy,
              cntx
            );
        }

        return;
    }

    // At this point, we know that b_n is exactly equal to the fusing factor.

    const double* restrict ap0   = a + 0*lda;
    const double* restrict ap1   = a + 1*lda;
    const double* restrict ap2   = a + 2*lda;
    const double* restrict ap3   = a + 3*lda;

          double* restrict yp   = y;

    chi0 = *( x + 0*incx );
    chi1 = *( x + 1*incx );
    chi2 = *( x + 2*incx );
    chi3 = *( x + 3*incx );

    // Scale each chi scalar by alpha.
    bli_dscals( *alpha, chi0 );
    bli_dscals( *alpha, chi1 );
    bli_dscals( *alpha, chi2 );
    bli_dscals( *alpha, chi3 );

    // Broadcast the (alpha*chi?) scalars to all elements of vector registers.
    chi0v.v = _mm256_broadcast_sd( &chi0 );
    chi1v.v = _mm256_broadcast_sd( &chi1 );
    chi2v.v = _mm256_broadcast_sd( &chi2 );
    chi3v.v = _mm256_broadcast_sd( &chi3 );

    // If there are vectorized iterations, perform them with vector
    // instructions.
    if ( inca == 1 && incy == 1 )
    {
        for ( i = 0; (i + 15) < m; i += 16 )
        {
            // Load the input values.
            y0v.v = _mm256_loadu_pd( yp + 0*n_elem_per_reg );
            y1v.v = _mm256_loadu_pd( yp + 1*n_elem_per_reg );
            y2v.v = _mm256_loadu_pd( yp + 2*n_elem_per_reg );
            y3v.v = _mm256_loadu_pd( yp + 3*n_elem_per_reg );

            a00v.v = _mm256_loadu_pd( ap0 + 0*n_elem_per_reg );
            a10v.v = _mm256_loadu_pd( ap0 + 1*n_elem_per_reg );
            a20v.v = _mm256_loadu_pd( ap0 + 2*n_elem_per_reg );
            a30v.v = _mm256_loadu_pd( ap0 + 3*n_elem_per_reg );

            a01v.v = _mm256_loadu_pd( ap1 + 0*n_elem_per_reg );
            a11v.v = _mm256_loadu_pd( ap1 + 1*n_elem_per_reg );
            a21v.v = _mm256_loadu_pd( ap1 + 2*n_elem_per_reg );
            a31v.v = _mm256_loadu_pd( ap1 + 3*n_elem_per_reg );

            a02v.v = _mm256_loadu_pd( ap2 + 0*n_elem_per_reg );
            a12v.v = _mm256_loadu_pd( ap2 + 1*n_elem_per_reg );
            a22v.v = _mm256_loadu_pd( ap2 + 2*n_elem_per_reg );
            a32v.v = _mm256_loadu_pd( ap2 + 3*n_elem_per_reg );

            a03v.v = _mm256_loadu_pd( ap3 + 0*n_elem_per_reg );
            a13v.v = _mm256_loadu_pd( ap3 + 1*n_elem_per_reg );
            a23v.v = _mm256_loadu_pd( ap3 + 2*n_elem_per_reg );
            a33v.v = _mm256_loadu_pd( ap3 + 3*n_elem_per_reg );

            // perform : y += alpha * x;
            y0v.v = _mm256_fmadd_pd( a00v.v, chi0v.v, y0v.v );
            y1v.v = _mm256_fmadd_pd( a10v.v, chi0v.v, y1v.v );
            y2v.v = _mm256_fmadd_pd( a20v.v, chi0v.v, y2v.v );
            y3v.v = _mm256_fmadd_pd( a30v.v, chi0v.v, y3v.v );

            y0v.v = _mm256_fmadd_pd( a01v.v, chi1v.v, y0v.v );
            y1v.v = _mm256_fmadd_pd( a11v.v, chi1v.v, y1v.v );
            y2v.v = _mm256_fmadd_pd( a21v.v, chi1v.v, y2v.v );
            y3v.v = _mm256_fmadd_pd( a31v.v, chi1v.v, y3v.v );

            y0v.v = _mm256_fmadd_pd( a02v.v, chi2v.v, y0v.v );
            y1v.v = _mm256_fmadd_pd( a12v.v, chi2v.v, y1v.v );
            y2v.v = _mm256_fmadd_pd( a22v.v, chi2v.v, y2v.v );
            y3v.v = _mm256_fmadd_pd( a32v.v, chi2v.v, y3v.v );

            y0v.v = _mm256_fmadd_pd( a03v.v, chi3v.v, y0v.v );
            y1v.v = _mm256_fmadd_pd( a13v.v, chi3v.v, y1v.v );
            y2v.v = _mm256_fmadd_pd( a23v.v, chi3v.v, y2v.v );
            y3v.v = _mm256_fmadd_pd( a33v.v, chi3v.v, y3v.v );

            // Store the output.
            _mm256_storeu_pd( (double *)(yp + 0*n_elem_per_reg), y0v.v );
            _mm256_storeu_pd( (double *)(yp + 1*n_elem_per_reg), y1v.v );
            _mm256_storeu_pd( (double *)(yp + 2*n_elem_per_reg), y2v.v );
            _mm256_storeu_pd( (double *)(yp + 3*n_elem_per_reg), y3v.v );

            yp += n_iter_unroll * n_elem_per_reg;
            ap0 += n_iter_unroll * n_elem_per_reg;
            ap1 += n_iter_unroll * n_elem_per_reg;
            ap2 += n_iter_unroll * n_elem_per_reg;
            ap3 += n_iter_unroll * n_elem_per_reg;
        }

        for ( ; (i + 11) < m; i += 12 )
        {
            // Load the input values.
            y0v.v = _mm256_loadu_pd( yp + 0*n_elem_per_reg );
            y1v.v = _mm256_loadu_pd( yp + 1*n_elem_per_reg );
            y2v.v = _mm256_loadu_pd( yp + 2*n_elem_per_reg );

            a00v.v = _mm256_loadu_pd( ap0 + 0*n_elem_per_reg );
            a10v.v = _mm256_loadu_pd( ap0 + 1*n_elem_per_reg );
            a20v.v = _mm256_loadu_pd( ap0 + 2*n_elem_per_reg );

            a01v.v = _mm256_loadu_pd( ap1 + 0*n_elem_per_reg );
            a11v.v = _mm256_loadu_pd( ap1 + 1*n_elem_per_reg );
            a21v.v = _mm256_loadu_pd( ap1 + 2*n_elem_per_reg );

            a02v.v = _mm256_loadu_pd( ap2 + 0*n_elem_per_reg );
            a12v.v = _mm256_loadu_pd( ap2 + 1*n_elem_per_reg );
            a22v.v = _mm256_loadu_pd( ap2 + 2*n_elem_per_reg );

            a03v.v = _mm256_loadu_pd( ap3 + 0*n_elem_per_reg );
            a13v.v = _mm256_loadu_pd( ap3 + 1*n_elem_per_reg );
            a23v.v = _mm256_loadu_pd( ap3 + 2*n_elem_per_reg );

            // perform : y += alpha * x;
            y0v.v = _mm256_fmadd_pd( a00v.v, chi0v.v, y0v.v );
            y1v.v = _mm256_fmadd_pd( a10v.v, chi0v.v, y1v.v );
            y2v.v = _mm256_fmadd_pd( a20v.v, chi0v.v, y2v.v );

            y0v.v = _mm256_fmadd_pd( a01v.v, chi1v.v, y0v.v );
            y1v.v = _mm256_fmadd_pd( a11v.v, chi1v.v, y1v.v );
            y2v.v = _mm256_fmadd_pd( a21v.v, chi1v.v, y2v.v );

            y0v.v = _mm256_fmadd_pd( a02v.v, chi2v.v, y0v.v );
            y1v.v = _mm256_fmadd_pd( a12v.v, chi2v.v, y1v.v );
            y2v.v = _mm256_fmadd_pd( a22v.v, chi2v.v, y2v.v );

            y0v.v = _mm256_fmadd_pd( a03v.v, chi3v.v, y0v.v );
            y1v.v = _mm256_fmadd_pd( a13v.v, chi3v.v, y1v.v );
            y2v.v = _mm256_fmadd_pd( a23v.v, chi3v.v, y2v.v );

            // Store the output.
            _mm256_storeu_pd( (double *)(yp + 0*n_elem_per_reg), y0v.v );
            _mm256_storeu_pd( (double *)(yp + 1*n_elem_per_reg), y1v.v );
            _mm256_storeu_pd( (double *)(yp + 2*n_elem_per_reg), y2v.v );

            yp += 3 * n_elem_per_reg;
            ap0 += 3 * n_elem_per_reg;
            ap1 += 3 * n_elem_per_reg;
            ap2 += 3 * n_elem_per_reg;
            ap3 += 3 * n_elem_per_reg;
        }

        for ( ; (i + 7) < m; i += 8 )
        {
            // Load the input values.
            y0v.v = _mm256_loadu_pd( yp + 0*n_elem_per_reg );
            y1v.v = _mm256_loadu_pd( yp + 1*n_elem_per_reg );

            a00v.v = _mm256_loadu_pd( ap0 + 0*n_elem_per_reg );
            a10v.v = _mm256_loadu_pd( ap0 + 1*n_elem_per_reg );

            a01v.v = _mm256_loadu_pd( ap1 + 0*n_elem_per_reg );
            a11v.v = _mm256_loadu_pd( ap1 + 1*n_elem_per_reg );

            a02v.v = _mm256_loadu_pd( ap2 + 0*n_elem_per_reg );
            a12v.v = _mm256_loadu_pd( ap2 + 1*n_elem_per_reg );

            a03v.v = _mm256_loadu_pd( ap3 + 0*n_elem_per_reg );
            a13v.v = _mm256_loadu_pd( ap3 + 1*n_elem_per_reg );

            // perform : y += alpha * x;
            y0v.v = _mm256_fmadd_pd( a00v.v, chi0v.v, y0v.v );
            y1v.v = _mm256_fmadd_pd( a10v.v, chi0v.v, y1v.v );

            y0v.v = _mm256_fmadd_pd( a01v.v, chi1v.v, y0v.v );
            y1v.v = _mm256_fmadd_pd( a11v.v, chi1v.v, y1v.v );

            y0v.v = _mm256_fmadd_pd( a02v.v, chi2v.v, y0v.v );
            y1v.v = _mm256_fmadd_pd( a12v.v, chi2v.v, y1v.v );

            y0v.v = _mm256_fmadd_pd( a03v.v, chi3v.v, y0v.v );
            y1v.v = _mm256_fmadd_pd( a13v.v, chi3v.v, y1v.v );

            // Store the output.
            _mm256_storeu_pd( (double *)(yp + 0*n_elem_per_reg), y0v.v );
            _mm256_storeu_pd( (double *)(yp + 1*n_elem_per_reg), y1v.v );

            yp += 2 * n_elem_per_reg;
            ap0 += 2 * n_elem_per_reg;
            ap1 += 2 * n_elem_per_reg;
            ap2 += 2 * n_elem_per_reg;
            ap3 += 2 * n_elem_per_reg;
        }


        for ( ; (i + 3) < m; i += 4)
        {
            // Load the input values.
            y0v.v = _mm256_loadu_pd( yp + 0*n_elem_per_reg );

            a00v.v = _mm256_loadu_pd( ap0 + 0*n_elem_per_reg );

            a01v.v = _mm256_loadu_pd( ap1 + 0*n_elem_per_reg );

            a02v.v = _mm256_loadu_pd( ap2 + 0*n_elem_per_reg );

            a03v.v = _mm256_loadu_pd( ap3 + 0*n_elem_per_reg );

            // perform : y += alpha * x;
            y0v.v = _mm256_fmadd_pd( a00v.v, chi0v.v, y0v.v );

            y0v.v = _mm256_fmadd_pd( a01v.v, chi1v.v, y0v.v );

            y0v.v = _mm256_fmadd_pd( a02v.v, chi2v.v, y0v.v );

            y0v.v = _mm256_fmadd_pd( a03v.v, chi3v.v, y0v.v );

            // Store the output.
            _mm256_storeu_pd( (double *)(yp + 0*n_elem_per_reg), y0v.v );

            yp += n_elem_per_reg;
            ap0 += n_elem_per_reg;
            ap1 += n_elem_per_reg;
            ap2 += n_elem_per_reg;
            ap3 += n_elem_per_reg;
        }
#if 1
        for ( ; (i + 1) < m; i += 2)
        {

            // Load the input values.
            y4v.v  = _mm_loadu_pd( yp + 0*n_elem_per_reg );

            a40v.v = _mm_loadu_pd( ap0 + 0*n_elem_per_reg );

            a41v.v = _mm_loadu_pd( ap1 + 0*n_elem_per_reg );

            a42v.v = _mm_loadu_pd( ap2 + 0*n_elem_per_reg );

            a43v.v = _mm_loadu_pd( ap3 + 0*n_elem_per_reg );

            // perform : y += alpha * x;
            y4v.v = _mm_fmadd_pd( a40v.v, chi0v.xmm[0], y4v.v );

            y4v.v = _mm_fmadd_pd( a41v.v, chi1v.xmm[0], y4v.v );

            y4v.v = _mm_fmadd_pd( a42v.v, chi2v.xmm[0], y4v.v );

            y4v.v = _mm_fmadd_pd( a43v.v, chi3v.xmm[0], y4v.v );

            // Store the output.
            _mm_storeu_pd( (double *)(yp + 0*n_elem_per_reg), y4v.v );

            yp += 2;
            ap0 += 2;
            ap1 += 2;
            ap2 += 2;
            ap3 += 2;
        }
#endif
        // If there are leftover iterations, perform them with scalar code.
        for ( ; (i + 0) < m ; ++i )
        {
            double       y0c = *yp;

            const double a0c = *ap0;
            const double a1c = *ap1;
            const double a2c = *ap2;
            const double a3c = *ap3;

            y0c += chi0 * a0c;
            y0c += chi1 * a1c;
            y0c += chi2 * a2c;
            y0c += chi3 * a3c;

            *yp = y0c;

            ap0 += 1;
            ap1 += 1;
            ap2 += 1;
            ap3 += 1;

            yp += 1;
        }
    }
    else
    {
        for ( i = 0; (i + 0) < m ; ++i )
        {
            double       y0c = *yp;

            const double a0c = *ap0;
            const double a1c = *ap1;
            const double a2c = *ap2;
            const double a3c = *ap3;

            y0c += chi0 * a0c;
            y0c += chi1 * a1c;
            y0c += chi2 * a2c;
            y0c += chi3 * a3c;

            *yp = y0c;

            ap0 += inca;
            ap1 += inca;
            ap2 += inca;
            ap3 += inca;

            yp += incy;
        }

    }
}


