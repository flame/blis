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


void bli_saxpyf_zen_int_5
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
    const dim_t      fuse_fac       = 5;

    const dim_t      n_elem_per_reg = 8;
    const dim_t      n_iter_unroll  = 2;

    dim_t            i;

    float* restrict a0;
    float* restrict a1;
    float* restrict a2;
    float* restrict a3;
    float* restrict a4;

    float* restrict y0;

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
#ifdef BLIS_CONFIG_ZEN2
        for ( i = 0; i < b_n; ++i )
        {
            float* a1   = a + (0  )*inca + (i  )*lda;
            float* chi1 = x + (i  )*incx;
            float* y1   = y + (0  )*incy;
            float  alpha_chi1;

            bli_scopycjs( conjx, *chi1, alpha_chi1 );
            bli_sscals( *alpha, alpha_chi1 );

            bli_saxpyv_zen_int10
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
            y0v.v = _mm256_loadu_ps( y0 + 0*n_elem_per_reg );
            y1v.v = _mm256_loadu_ps( y0 + 1*n_elem_per_reg );

            a00v.v = _mm256_loadu_ps( a0 + 0*n_elem_per_reg );
            a10v.v = _mm256_loadu_ps( a0 + 1*n_elem_per_reg );

            a01v.v = _mm256_loadu_ps( a1 + 0*n_elem_per_reg );
            a11v.v = _mm256_loadu_ps( a1 + 1*n_elem_per_reg );

            a02v.v = _mm256_loadu_ps( a2 + 0*n_elem_per_reg );
            a12v.v = _mm256_loadu_ps( a2 + 1*n_elem_per_reg );

            a03v.v = _mm256_loadu_ps( a3 + 0*n_elem_per_reg );
            a13v.v = _mm256_loadu_ps( a3 + 1*n_elem_per_reg );

            a04v.v = _mm256_loadu_ps( a4 + 0*n_elem_per_reg );
            a14v.v = _mm256_loadu_ps( a4 + 1*n_elem_per_reg );

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
            _mm256_storeu_ps( (y0 + 0*n_elem_per_reg), y0v.v );
            _mm256_storeu_ps( (y0 + 1*n_elem_per_reg), y1v.v );

            y0 += n_iter_unroll * n_elem_per_reg;
            a0 += n_iter_unroll * n_elem_per_reg;
            a1 += n_iter_unroll * n_elem_per_reg;
            a2 += n_iter_unroll * n_elem_per_reg;
            a3 += n_iter_unroll * n_elem_per_reg;
            a4 += n_iter_unroll * n_elem_per_reg;
        }

        for( ; (i + 7) < m; i += 8 )
        {
            // Load the input values.
            y0v.v = _mm256_loadu_ps( y0 + 0*n_elem_per_reg );

            a00v.v = _mm256_loadu_ps( a0 + 0*n_elem_per_reg );
            a01v.v = _mm256_loadu_ps( a1 + 0*n_elem_per_reg );
            a02v.v = _mm256_loadu_ps( a2 + 0*n_elem_per_reg );
            a03v.v = _mm256_loadu_ps( a3 + 0*n_elem_per_reg );
            a04v.v = _mm256_loadu_ps( a4 + 0*n_elem_per_reg );


            // perform : y += alpha * x;
            y0v.v = _mm256_fmadd_ps( a00v.v, chi0v.v, y0v.v );
            y0v.v = _mm256_fmadd_ps( a01v.v, chi1v.v, y0v.v );
            y0v.v = _mm256_fmadd_ps( a02v.v, chi2v.v, y0v.v );
            y0v.v = _mm256_fmadd_ps( a03v.v, chi3v.v, y0v.v );
            y0v.v = _mm256_fmadd_ps( a04v.v, chi4v.v, y0v.v );

            // Store the output.
            _mm256_storeu_ps( (y0 + 0*n_elem_per_reg), y0v.v );

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

            const float a0c = *a0;
            const float a1c = *a1;
            const float a2c = *a2;
            const float a3c = *a3;
            const float a4c = *a4;

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

            const float a0c = *a0;
            const float a1c = *a1;
            const float a2c = *a2;
            const float a3c = *a3;
            const float a4c = *a4;

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
    if ( bli_zero_dim2( m, b_n ) || bli_deq0( *alpha ) ) return;

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

            bli_dcopycjs( conjx, *chi1, alpha_chi1 );
            bli_dscals( *alpha, alpha_chi1 );

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

            bli_dcopycjs( conjx, *chi1, alpha_chi1 );
            bli_dscals( *alpha, alpha_chi1 );

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
            _mm256_storeu_pd( (double *)(y0 + 0*n_elem_per_reg), y0v.v );
            _mm256_storeu_pd( (double *)(y0 + 1*n_elem_per_reg), y1v.v );

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


// -----------------------------------------------------------------------------

void bli_caxpyf_zen_int_5
     (
       conj_t           conja,
       conj_t           conjx,
       dim_t            m,
       dim_t            b_n,
       scomplex* restrict alpha,
       scomplex* restrict a, inc_t inca, inc_t lda,
       scomplex* restrict x, inc_t incx,
       scomplex* restrict y, inc_t incy,
       cntx_t* restrict cntx
     )
{
    const dim_t         fuse_fac       = 5;

    const dim_t         n_elem_per_reg = 4;

    dim_t               i = 0;
    dim_t               setPlusOne = 1;

    v8sf_t              chi0v, chi1v, chi2v, chi3v, chi4v;
    v8sf_t              chi5v, chi6v, chi7v, chi8v, chi9v;

    v8sf_t              a00v, a01v, a02v, a03v, a04v;
    v8sf_t              a05v, a06v, a07v, a08v, a09v;
#if 0
    v8sf_t              a10v, a11v, a12v, a13v, a14v;
    v8sf_t              a15v, a16v, a17v, a18v, a19v;
    v8sf_t              y1v;
#endif
    v8sf_t              y0v;
    v8sf_t              setMinus, setPlus;

    scomplex* restrict  a0;
    scomplex* restrict  a1;
    scomplex* restrict  a2;
    scomplex* restrict  a3;
    scomplex* restrict  a4;

    scomplex* restrict  y0;

    scomplex            chi0;
    scomplex            chi1;
    scomplex            chi2;
    scomplex            chi3;
    scomplex            chi4;

    if ( bli_is_conj(conja) ){
        setPlusOne = -1;
    }

    // If either dimension is zero, or if alpha is zero, return early.
    if ( bli_zero_dim2( m, b_n ) || bli_ceq0( *alpha ) ) return;

    // If b_n is not equal to the fusing factor, then perform the entire
    // operation as a loop over axpyv.
    if ( b_n != fuse_fac )
    {
#ifdef BLIS_CONFIG_EPYC
        for ( i = 0; i < b_n; ++i )
        {
            scomplex* a1   = a + (0  )*inca + (i  )*lda;
            scomplex* chi1 = x + (i  )*incx;
            scomplex* y1   = y + (0  )*incy;
            scomplex  alpha_chi1;

            bli_ccopycjs( conjx, *chi1, alpha_chi1 );
            bli_cscals( *alpha, alpha_chi1 );

            bli_caxpyv_zen_int5
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
        caxpyv_ker_ft f = bli_cntx_get_l1v_ker_dt( BLIS_SCOMPLEX, BLIS_AXPYV_KER, cntx );

        for ( i = 0; i < b_n; ++i )
        {
            scomplex* a1   = a + (0  )*inca + (i  )*lda;
            scomplex* chi1 = x + (i  )*incx;
            scomplex* y1   = y + (0  )*incy;
            scomplex  alpha_chi1;

            bli_ccopycjs( conjx, *chi1, alpha_chi1 );
            bli_cscals( *alpha, alpha_chi1 );

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

    scomplex *pchi0 = x + 0*incx ;
    scomplex *pchi1 = x + 1*incx ;
    scomplex *pchi2 = x + 2*incx ;
    scomplex *pchi3 = x + 3*incx ;
    scomplex *pchi4 = x + 4*incx ;

    bli_ccopycjs( conjx, *pchi0, chi0 );
    bli_ccopycjs( conjx, *pchi1, chi1 );
    bli_ccopycjs( conjx, *pchi2, chi2 );
    bli_ccopycjs( conjx, *pchi3, chi3 );
    bli_ccopycjs( conjx, *pchi4, chi4 );

    // Scale each chi scalar by alpha.
    bli_cscals( *alpha, chi0 );
    bli_cscals( *alpha, chi1 );
    bli_cscals( *alpha, chi2 );
    bli_cscals( *alpha, chi3 );
    bli_cscals( *alpha, chi4 );

    // Broadcast the (alpha*chi?) scalars to all elements of vector registers.
    chi0v.v = _mm256_broadcast_ss( &chi0.real );
    chi1v.v = _mm256_broadcast_ss( &chi1.real );
    chi2v.v = _mm256_broadcast_ss( &chi2.real );
    chi3v.v = _mm256_broadcast_ss( &chi3.real );
    chi4v.v = _mm256_broadcast_ss( &chi4.real );

    chi5v.v = _mm256_broadcast_ss( &chi0.imag );
    chi6v.v = _mm256_broadcast_ss( &chi1.imag );
    chi7v.v = _mm256_broadcast_ss( &chi2.imag );
    chi8v.v = _mm256_broadcast_ss( &chi3.imag );
    chi9v.v = _mm256_broadcast_ss( &chi4.imag );

    // If there are vectorized iterations, perform them with vector
    // instructions.
    if ( inca == 1 && incy == 1 )
    {
        setMinus.v = _mm256_set_ps( -1, 1, -1, 1, -1, 1, -1, 1 );

        setPlus.v = _mm256_set1_ps( 1 );
        if ( bli_is_conj(conja) ){
            setPlus.v = _mm256_set_ps( -1, 1, -1, 1, -1, 1, -1, 1 );
        }

        /*
         y := y + alpha * conja(A) * conjx(x)

          nn
          (ar + ai) (xr + xi)
          ar * xr - ai * xi
          ar * xi + ai * xr

         cc : (ar - ai) (xr - xi)
          ar * xr - ai * xi
          -(ar * xi + ai * xr)

          nc : (ar + ai) (xr - xi)
           ar * xr + ai * xi
          -(ar * xi - ai * xr)

          cn : (ar - ai) (xr + xi)
           ar * xr + ai * xi
          ar * xi - ai * xr

        */

        i = 0;
#if 0 //Low performance
        for( i = 0; (i + 7) < m; i += 8 )
        {
            // Load the input values.
            y0v.v = _mm256_loadu_ps( (float*) (y0 + 0*n_elem_per_reg ));
            y1v.v = _mm256_loadu_ps( (float*) (y0 + 1*n_elem_per_reg ));

            a00v.v = _mm256_loadu_ps( (float*) (a0 + 0*n_elem_per_reg ));
            a10v.v = _mm256_loadu_ps( (float*) (a0 + 1*n_elem_per_reg ));

            a01v.v = _mm256_loadu_ps( (float*) (a1 + 0*n_elem_per_reg ));
            a11v.v = _mm256_loadu_ps( (float*) (a1 + 1*n_elem_per_reg ));

            a02v.v = _mm256_loadu_ps( (float*) (a2 + 0*n_elem_per_reg ));
            a12v.v = _mm256_loadu_ps( (float*) (a2 + 1*n_elem_per_reg ));

            a03v.v = _mm256_loadu_ps( (float*) (a3 + 0*n_elem_per_reg ));
            a13v.v = _mm256_loadu_ps( (float*) (a3 + 1*n_elem_per_reg ));

            a04v.v = _mm256_loadu_ps( (float*) (a4 + 0*n_elem_per_reg ));
            a14v.v = _mm256_loadu_ps( (float*) (a4 + 1*n_elem_per_reg ));

            a00v.v = _mm256_mul_ps( a00v.v, setPlus.v );
            a01v.v = _mm256_mul_ps( a01v.v, setPlus.v );
            a02v.v = _mm256_mul_ps( a02v.v, setPlus.v );
            a03v.v = _mm256_mul_ps( a03v.v, setPlus.v );
            a04v.v = _mm256_mul_ps( a04v.v, setPlus.v );

            a05v.v = _mm256_mul_ps( a00v.v, setMinus.v );
            a06v.v = _mm256_mul_ps( a01v.v, setMinus.v );
            a07v.v = _mm256_mul_ps( a02v.v, setMinus.v );
            a08v.v = _mm256_mul_ps( a03v.v, setMinus.v );
            a09v.v = _mm256_mul_ps( a04v.v, setMinus.v );

            a05v.v = _mm256_permute_ps( a05v.v, 0xB1 );
            a06v.v = _mm256_permute_ps( a06v.v, 0xB1 );
            a07v.v = _mm256_permute_ps( a07v.v, 0xB1 );
            a08v.v = _mm256_permute_ps( a08v.v, 0xB1 );
            a09v.v = _mm256_permute_ps( a09v.v, 0xB1 );

            a10v.v = _mm256_mul_ps( a10v.v, setPlus.v );
            a11v.v = _mm256_mul_ps( a11v.v, setPlus.v );
            a12v.v = _mm256_mul_ps( a12v.v, setPlus.v );
            a13v.v = _mm256_mul_ps( a13v.v, setPlus.v );
            a14v.v = _mm256_mul_ps( a14v.v, setPlus.v );

            a15v.v = _mm256_mul_ps( a10v.v, setMinus.v );
            a16v.v = _mm256_mul_ps( a11v.v, setMinus.v );
            a17v.v = _mm256_mul_ps( a12v.v, setMinus.v );
            a18v.v = _mm256_mul_ps( a13v.v, setMinus.v );
            a19v.v = _mm256_mul_ps( a14v.v, setMinus.v );

            a15v.v = _mm256_permute_ps( a15v.v, 0xB1 );
            a16v.v = _mm256_permute_ps( a16v.v, 0xB1 );
            a17v.v = _mm256_permute_ps( a17v.v, 0xB1 );
            a18v.v = _mm256_permute_ps( a18v.v, 0xB1 );
            a19v.v = _mm256_permute_ps( a19v.v, 0xB1 );

            // perform : y += alpha * x;
            y0v.v = _mm256_fmadd_ps( a00v.v, chi0v.v, y0v.v );
            y0v.v = _mm256_fmadd_ps( a01v.v, chi1v.v, y0v.v );
            y0v.v = _mm256_fmadd_ps( a02v.v, chi2v.v, y0v.v );
            y0v.v = _mm256_fmadd_ps( a03v.v, chi3v.v, y0v.v );
            y0v.v = _mm256_fmadd_ps( a04v.v, chi4v.v, y0v.v );

            y0v.v = _mm256_fmadd_ps( a05v.v, chi5v.v, y0v.v );
            y0v.v = _mm256_fmadd_ps( a06v.v, chi6v.v, y0v.v );
            y0v.v = _mm256_fmadd_ps( a07v.v, chi7v.v, y0v.v );
            y0v.v = _mm256_fmadd_ps( a08v.v, chi8v.v, y0v.v );
            y0v.v = _mm256_fmadd_ps( a09v.v, chi9v.v, y0v.v );

            // For next 4 elements perform : y += alpha * x;
            y1v.v = _mm256_fmadd_ps( a10v.v, chi0v.v, y1v.v );
            y1v.v = _mm256_fmadd_ps( a11v.v, chi1v.v, y1v.v );
            y1v.v = _mm256_fmadd_ps( a12v.v, chi2v.v, y1v.v );
            y1v.v = _mm256_fmadd_ps( a13v.v, chi3v.v, y1v.v );
            y1v.v = _mm256_fmadd_ps( a14v.v, chi4v.v, y1v.v );

            y1v.v = _mm256_fmadd_ps( a15v.v, chi5v.v, y1v.v );
            y1v.v = _mm256_fmadd_ps( a16v.v, chi6v.v, y1v.v );
            y1v.v = _mm256_fmadd_ps( a17v.v, chi7v.v, y1v.v );
            y1v.v = _mm256_fmadd_ps( a18v.v, chi8v.v, y1v.v );
            y1v.v = _mm256_fmadd_ps( a19v.v, chi9v.v, y1v.v );

            // Store the output.
            _mm256_storeu_ps( (float *)(y0 + 0*n_elem_per_reg), y0v.v );
            _mm256_storeu_ps( (float *)(y0 + 1*n_elem_per_reg), y1v.v );

            y0 += n_elem_per_reg * n_iter_unroll;
            a0 += n_elem_per_reg * n_iter_unroll;
            a1 += n_elem_per_reg * n_iter_unroll;
            a2 += n_elem_per_reg * n_iter_unroll;
            a3 += n_elem_per_reg * n_iter_unroll;
            a4 += n_elem_per_reg * n_iter_unroll;
        }
#endif
        for( ; (i + 3) < m; i += 4 )
        {
            // Load the input values.
            y0v.v = _mm256_loadu_ps( (float*) (y0 + 0*n_elem_per_reg ));

            a00v.v = _mm256_loadu_ps( (float*) (a0 + 0*n_elem_per_reg ));
            a01v.v = _mm256_loadu_ps( (float*) (a1 + 0*n_elem_per_reg ));
            a02v.v = _mm256_loadu_ps( (float*) (a2 + 0*n_elem_per_reg ));
            a03v.v = _mm256_loadu_ps( (float*) (a3 + 0*n_elem_per_reg ));
            a04v.v = _mm256_loadu_ps( (float*) (a4 + 0*n_elem_per_reg ));

            a00v.v = _mm256_mul_ps( a00v.v, setPlus.v );
            a01v.v = _mm256_mul_ps( a01v.v, setPlus.v );
            a02v.v = _mm256_mul_ps( a02v.v, setPlus.v );
            a03v.v = _mm256_mul_ps( a03v.v, setPlus.v );
            a04v.v = _mm256_mul_ps( a04v.v, setPlus.v );

            a05v.v = _mm256_mul_ps( a00v.v, setMinus.v );
            a06v.v = _mm256_mul_ps( a01v.v, setMinus.v );
            a07v.v = _mm256_mul_ps( a02v.v, setMinus.v );
            a08v.v = _mm256_mul_ps( a03v.v, setMinus.v );
            a09v.v = _mm256_mul_ps( a04v.v, setMinus.v );

            a05v.v = _mm256_permute_ps( a05v.v, 0xB1 );
            a06v.v = _mm256_permute_ps( a06v.v, 0xB1 );
            a07v.v = _mm256_permute_ps( a07v.v, 0xB1 );
            a08v.v = _mm256_permute_ps( a08v.v, 0xB1 );
            a09v.v = _mm256_permute_ps( a09v.v, 0xB1 );

            // perform : y += alpha * x;
            y0v.v = _mm256_fmadd_ps( a00v.v, chi0v.v, y0v.v );
            y0v.v = _mm256_fmadd_ps( a01v.v, chi1v.v, y0v.v );
            y0v.v = _mm256_fmadd_ps( a02v.v, chi2v.v, y0v.v );
            y0v.v = _mm256_fmadd_ps( a03v.v, chi3v.v, y0v.v );
            y0v.v = _mm256_fmadd_ps( a04v.v, chi4v.v, y0v.v );

            y0v.v = _mm256_fmadd_ps( a05v.v, chi5v.v, y0v.v );
            y0v.v = _mm256_fmadd_ps( a06v.v, chi6v.v, y0v.v );
            y0v.v = _mm256_fmadd_ps( a07v.v, chi7v.v, y0v.v );
            y0v.v = _mm256_fmadd_ps( a08v.v, chi8v.v, y0v.v );
            y0v.v = _mm256_fmadd_ps( a09v.v, chi9v.v, y0v.v );

            // Store the output.
            _mm256_storeu_ps( (float *)(y0 + 0*n_elem_per_reg), y0v.v );

            y0 += n_elem_per_reg ;
            a0 += n_elem_per_reg ;
            a1 += n_elem_per_reg ;
            a2 += n_elem_per_reg ;
            a3 += n_elem_per_reg ;
            a4 += n_elem_per_reg ;
        }

        // If there are leftover iterations, perform them with scalar code.
        for ( ; (i + 0) < m ; ++i )
        {
            scomplex       y0c = *y0;

            const scomplex a0c = *a0;
            const scomplex a1c = *a1;
            const scomplex a2c = *a2;
            const scomplex a3c = *a3;
            const scomplex a4c = *a4;

            y0c.real += chi0.real * a0c.real - chi0.imag * a0c.imag * setPlusOne;
            y0c.real += chi1.real * a1c.real - chi1.imag * a1c.imag * setPlusOne;
            y0c.real += chi2.real * a2c.real - chi2.imag * a2c.imag * setPlusOne;
            y0c.real += chi3.real * a3c.real - chi3.imag * a3c.imag * setPlusOne;
            y0c.real += chi4.real * a4c.real - chi4.imag * a4c.imag * setPlusOne;

            y0c.imag += chi0.imag * a0c.real + chi0.real * a0c.imag * setPlusOne;
            y0c.imag += chi1.imag * a1c.real + chi1.real * a1c.imag * setPlusOne;
            y0c.imag += chi2.imag * a2c.real + chi2.real * a2c.imag * setPlusOne;
            y0c.imag += chi3.imag * a3c.real + chi3.real * a3c.imag * setPlusOne;
            y0c.imag += chi4.imag * a4c.real + chi4.real * a4c.imag * setPlusOne;

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
        for ( ; (i + 0) < m ; ++i )
        {
            scomplex       y0c = *y0;
            const scomplex a0c = *a0;
            const scomplex a1c = *a1;
            const scomplex a2c = *a2;
            const scomplex a3c = *a3;
            const scomplex a4c = *a4;

            y0c.real += chi0.real * a0c.real - chi0.imag * a0c.imag * setPlusOne;
            y0c.real += chi1.real * a1c.real - chi1.imag * a1c.imag * setPlusOne;
            y0c.real += chi2.real * a2c.real - chi2.imag * a2c.imag * setPlusOne;
            y0c.real += chi3.real * a3c.real - chi3.imag * a3c.imag * setPlusOne;
            y0c.real += chi4.real * a4c.real - chi4.imag * a4c.imag * setPlusOne;

            y0c.imag += chi0.imag * a0c.real + chi0.real * a0c.imag * setPlusOne;
            y0c.imag += chi1.imag * a1c.real + chi1.real * a1c.imag * setPlusOne;
            y0c.imag += chi2.imag * a2c.real + chi2.real * a2c.imag * setPlusOne;
            y0c.imag += chi3.imag * a3c.real + chi3.real * a3c.imag * setPlusOne;
            y0c.imag += chi4.imag * a4c.real + chi4.real * a4c.imag * setPlusOne;

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


// -----------------------------------------------------------------------------

void bli_zaxpyf_zen_int_5
     (
       conj_t           conja,
       conj_t           conjx,
       dim_t            m,
       dim_t            b_n,
       dcomplex* restrict alpha,
       dcomplex* restrict a, inc_t inca, inc_t lda,
       dcomplex* restrict x, inc_t incx,
       dcomplex* restrict y, inc_t incy,
       cntx_t* restrict cntx
     )
{
    const dim_t              fuse_fac       = 5;

    const dim_t              n_elem_per_reg = 2;
    const dim_t              n_iter_unroll  = 2;

    dim_t                    i = 0;
    dim_t                    setPlusOne = 1;

    v4df_t                   chi0v, chi1v, chi2v, chi3v, chi4v;
    v4df_t                   chi5v, chi6v, chi7v, chi8v, chi9v;

    v4df_t                   a00v, a01v, a02v, a03v, a04v;
    v4df_t                   a05v, a06v, a07v, a08v, a09v;

    v4df_t                   a10v, a11v, a12v, a13v, a14v;
    v4df_t                   a15v, a16v, a17v, a18v, a19v;

    v4df_t                   y0v, y1v;
    v4df_t                   setMinus, setPlus;

    dcomplex                 chi0, chi1, chi2, chi3, chi4;
    dcomplex* restrict       a0;
    dcomplex* restrict       a1;
    dcomplex* restrict       a2;
    dcomplex* restrict       a3;
    dcomplex* restrict       a4;

    dcomplex*                restrict y0;


    if ( bli_is_conj(conja) ){
        setPlusOne = -1;
    }

    // If either dimension is zero, or if alpha is zero, return early.
    if ( bli_zero_dim2( m, b_n ) || bli_zeq0( *alpha ) ) return;

    // If b_n is not equal to the fusing factor, then perform the entire
    // operation as a loop over axpyv.
    if ( b_n != fuse_fac )
    {
#ifdef BLIS_CONFIG_EPYC
        for ( i = 0; i < b_n; ++i )
        {
            dcomplex* a1   = a + (0  )*inca + (i  )*lda;
            dcomplex* chi1 = x + (i  )*incx;
            dcomplex* y1   = y + (0  )*incy;
            dcomplex  alpha_chi1;

            bli_zcopycjs( conjx, *chi1, alpha_chi1 );
            bli_zscals( *alpha, alpha_chi1 );

            bli_zaxpyv_zen_int5
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
        zaxpyv_ker_ft f = bli_cntx_get_l1v_ker_dt( BLIS_DCOMPLEX, BLIS_AXPYV_KER, cntx );

        for ( i = 0; i < b_n; ++i )
        {
            dcomplex* a1   = a + (0  )*inca + (i  )*lda;
            dcomplex* chi1 = x + (i  )*incx;
            dcomplex* y1   = y + (0  )*incy;
            dcomplex  alpha_chi1;

            bli_zcopycjs( conjx, *chi1, alpha_chi1 );
            bli_zscals( *alpha, alpha_chi1 );

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

    dcomplex *pchi0 = x + 0*incx ;
    dcomplex *pchi1 = x + 1*incx ;
    dcomplex *pchi2 = x + 2*incx ;
    dcomplex *pchi3 = x + 3*incx ;
    dcomplex *pchi4 = x + 4*incx ;

    bli_zcopycjs( conjx, *pchi0, chi0 );
    bli_zcopycjs( conjx, *pchi1, chi1 );
    bli_zcopycjs( conjx, *pchi2, chi2 );
    bli_zcopycjs( conjx, *pchi3, chi3 );
    bli_zcopycjs( conjx, *pchi4, chi4 );

    // Scale each chi scalar by alpha.
    bli_zscals( *alpha, chi0 );
    bli_zscals( *alpha, chi1 );
    bli_zscals( *alpha, chi2 );
    bli_zscals( *alpha, chi3 );
    bli_zscals( *alpha, chi4 );

    // Broadcast the (alpha*chi?) scalars to all elements of vector registers.
    chi0v.v = _mm256_broadcast_sd( &chi0.real );
    chi1v.v = _mm256_broadcast_sd( &chi1.real );
    chi2v.v = _mm256_broadcast_sd( &chi2.real );
    chi3v.v = _mm256_broadcast_sd( &chi3.real );
    chi4v.v = _mm256_broadcast_sd( &chi4.real );

    chi5v.v = _mm256_broadcast_sd( &chi0.imag );
    chi6v.v = _mm256_broadcast_sd( &chi1.imag );
    chi7v.v = _mm256_broadcast_sd( &chi2.imag );
    chi8v.v = _mm256_broadcast_sd( &chi3.imag );
    chi9v.v = _mm256_broadcast_sd( &chi4.imag );

    // If there are vectorized iterations, perform them with vector
    // instructions.
    if ( inca == 1 && incy == 1 )
    {
         setMinus.v = _mm256_set_pd( -1, 1, -1, 1 );

         setPlus.v = _mm256_set1_pd( 1 );
         if ( bli_is_conj(conja) ){
             setPlus.v = _mm256_set_pd( -1, 1, -1, 1 );
         }

         /*
          y := y + alpha * conja(A) * conjx(x)

           nn
           (ar + ai) (xr + xi)
           ar * xr - ai * xi
           ar * xi + ai * xr

          cc : (ar - ai) (xr - xi)
           ar * xr - ai * xi
           -(ar * xi + ai * xr)

           nc : (ar + ai) (xr - xi)
            ar * xr + ai * xi
           -(ar * xi - ai * xr)

           cn : (ar - ai) (xr + xi)
            ar * xr + ai * xi
           ar * xi - ai * xr

         */

        for( i = 0; (i + 3) < m; i += 4 )
        {
            // Load the input values.
            y0v.v = _mm256_loadu_pd( (double*) (y0 + 0*n_elem_per_reg ));
            y1v.v = _mm256_loadu_pd( (double*) (y0 + 1*n_elem_per_reg ));

            a00v.v = _mm256_loadu_pd( (double*) (a0 + 0*n_elem_per_reg ));
            a10v.v = _mm256_loadu_pd( (double*) (a0 + 1*n_elem_per_reg ));

            a01v.v = _mm256_loadu_pd( (double*) (a1 + 0*n_elem_per_reg ));
            a11v.v = _mm256_loadu_pd( (double*) (a1 + 1*n_elem_per_reg ));

            a02v.v = _mm256_loadu_pd( (double*) (a2 + 0*n_elem_per_reg ));
            a12v.v = _mm256_loadu_pd( (double*) (a2 + 1*n_elem_per_reg ));

            a03v.v = _mm256_loadu_pd( (double*) (a3 + 0*n_elem_per_reg ));
            a13v.v = _mm256_loadu_pd( (double*) (a3 + 1*n_elem_per_reg ));

            a04v.v = _mm256_loadu_pd( (double*) (a4 + 0*n_elem_per_reg ));
            a14v.v = _mm256_loadu_pd( (double*) (a4 + 1*n_elem_per_reg ));

            a00v.v = _mm256_mul_pd( a00v.v, setPlus.v );
            a01v.v = _mm256_mul_pd( a01v.v, setPlus.v );
            a02v.v = _mm256_mul_pd( a02v.v, setPlus.v );
            a03v.v = _mm256_mul_pd( a03v.v, setPlus.v );
            a04v.v = _mm256_mul_pd( a04v.v, setPlus.v );

            a05v.v = _mm256_mul_pd( a00v.v, setMinus.v );
            a06v.v = _mm256_mul_pd( a01v.v, setMinus.v );
            a07v.v = _mm256_mul_pd( a02v.v, setMinus.v );
            a08v.v = _mm256_mul_pd( a03v.v, setMinus.v );
            a09v.v = _mm256_mul_pd( a04v.v, setMinus.v );

            a05v.v = _mm256_permute_pd( a05v.v, 5 );
            a06v.v = _mm256_permute_pd( a06v.v, 5 );
            a07v.v = _mm256_permute_pd( a07v.v, 5 );
            a08v.v = _mm256_permute_pd( a08v.v, 5 );
            a09v.v = _mm256_permute_pd( a09v.v, 5 );

            a10v.v = _mm256_mul_pd( a10v.v, setPlus.v );
            a11v.v = _mm256_mul_pd( a11v.v, setPlus.v );
            a12v.v = _mm256_mul_pd( a12v.v, setPlus.v );
            a13v.v = _mm256_mul_pd( a13v.v, setPlus.v );
            a14v.v = _mm256_mul_pd( a14v.v, setPlus.v );

            a15v.v = _mm256_mul_pd( a10v.v, setMinus.v );
            a16v.v = _mm256_mul_pd( a11v.v, setMinus.v );
            a17v.v = _mm256_mul_pd( a12v.v, setMinus.v );
            a18v.v = _mm256_mul_pd( a13v.v, setMinus.v );
            a19v.v = _mm256_mul_pd( a14v.v, setMinus.v );

            a15v.v = _mm256_permute_pd( a15v.v, 5 );
            a16v.v = _mm256_permute_pd( a16v.v, 5 );
            a17v.v = _mm256_permute_pd( a17v.v, 5 );
            a18v.v = _mm256_permute_pd( a18v.v, 5 );
            a19v.v = _mm256_permute_pd( a19v.v, 5 );

            // perform : y += alpha * x;
            y0v.v = _mm256_fmadd_pd( a00v.v, chi0v.v, y0v.v );
            y0v.v = _mm256_fmadd_pd( a01v.v, chi1v.v, y0v.v );
            y0v.v = _mm256_fmadd_pd( a02v.v, chi2v.v, y0v.v );
            y0v.v = _mm256_fmadd_pd( a03v.v, chi3v.v, y0v.v );
            y0v.v = _mm256_fmadd_pd( a04v.v, chi4v.v, y0v.v );

            y0v.v = _mm256_fmadd_pd( a05v.v, chi5v.v, y0v.v );
            y0v.v = _mm256_fmadd_pd( a06v.v, chi6v.v, y0v.v );
            y0v.v = _mm256_fmadd_pd( a07v.v, chi7v.v, y0v.v );
            y0v.v = _mm256_fmadd_pd( a08v.v, chi8v.v, y0v.v );
            y0v.v = _mm256_fmadd_pd( a09v.v, chi9v.v, y0v.v );

            // For next 4 elements perform : y += alpha * x;
            y1v.v = _mm256_fmadd_pd( a10v.v, chi0v.v, y1v.v );
            y1v.v = _mm256_fmadd_pd( a11v.v, chi1v.v, y1v.v );
            y1v.v = _mm256_fmadd_pd( a12v.v, chi2v.v, y1v.v );
            y1v.v = _mm256_fmadd_pd( a13v.v, chi3v.v, y1v.v );
            y1v.v = _mm256_fmadd_pd( a14v.v, chi4v.v, y1v.v );

            y1v.v = _mm256_fmadd_pd( a15v.v, chi5v.v, y1v.v );
            y1v.v = _mm256_fmadd_pd( a16v.v, chi6v.v, y1v.v );
            y1v.v = _mm256_fmadd_pd( a17v.v, chi7v.v, y1v.v );
            y1v.v = _mm256_fmadd_pd( a18v.v, chi8v.v, y1v.v );
            y1v.v = _mm256_fmadd_pd( a19v.v, chi9v.v, y1v.v );

            // Store the output.
            _mm256_storeu_pd( (double*) (y0 + 0*n_elem_per_reg), y0v.v );
            _mm256_storeu_pd( (double*) (y0 + 1*n_elem_per_reg), y1v.v );

            y0 += n_elem_per_reg * n_iter_unroll;
            a0 += n_elem_per_reg * n_iter_unroll;
            a1 += n_elem_per_reg * n_iter_unroll;
            a2 += n_elem_per_reg * n_iter_unroll;
            a3 += n_elem_per_reg * n_iter_unroll;
            a4 += n_elem_per_reg * n_iter_unroll;
        }
        for(  ; (i + 1) < m; i += 2 )
        {
            // Load the input values.
            y0v.v = _mm256_loadu_pd( (double*) (y0 + 0*n_elem_per_reg ));

            a00v.v = _mm256_loadu_pd( (double*)(a0 + 0*n_elem_per_reg) );
            a01v.v = _mm256_loadu_pd( (double*)(a1 + 0*n_elem_per_reg) );
            a02v.v = _mm256_loadu_pd( (double*)(a2 + 0*n_elem_per_reg) );
            a03v.v = _mm256_loadu_pd( (double*)(a3 + 0*n_elem_per_reg) );
            a04v.v = _mm256_loadu_pd( (double*)(a4 + 0*n_elem_per_reg) );

            a00v.v = _mm256_mul_pd( a00v.v, setPlus.v );
            a01v.v = _mm256_mul_pd( a01v.v, setPlus.v );
            a02v.v = _mm256_mul_pd( a02v.v, setPlus.v );
            a03v.v = _mm256_mul_pd( a03v.v, setPlus.v );
            a04v.v = _mm256_mul_pd( a04v.v, setPlus.v );

            a05v.v = _mm256_mul_pd( a00v.v, setMinus.v );
            a06v.v = _mm256_mul_pd( a01v.v, setMinus.v );
            a07v.v = _mm256_mul_pd( a02v.v, setMinus.v );
            a08v.v = _mm256_mul_pd( a03v.v, setMinus.v );
            a09v.v = _mm256_mul_pd( a04v.v, setMinus.v );

            a05v.v = _mm256_permute_pd( a05v.v, 5 );
            a06v.v = _mm256_permute_pd( a06v.v, 5 );
            a07v.v = _mm256_permute_pd( a07v.v, 5 );
            a08v.v = _mm256_permute_pd( a08v.v, 5 );
            a09v.v = _mm256_permute_pd( a09v.v, 5 );

            // perform : y += alpha * x;
            y0v.v = _mm256_fmadd_pd( a00v.v, chi0v.v, y0v.v );
            y0v.v = _mm256_fmadd_pd( a01v.v, chi1v.v, y0v.v );
            y0v.v = _mm256_fmadd_pd( a02v.v, chi2v.v, y0v.v );
            y0v.v = _mm256_fmadd_pd( a03v.v, chi3v.v, y0v.v );
            y0v.v = _mm256_fmadd_pd( a04v.v, chi4v.v, y0v.v );

            y0v.v = _mm256_fmadd_pd( a05v.v, chi5v.v, y0v.v );
            y0v.v = _mm256_fmadd_pd( a06v.v, chi6v.v, y0v.v );
            y0v.v = _mm256_fmadd_pd( a07v.v, chi7v.v, y0v.v );
            y0v.v = _mm256_fmadd_pd( a08v.v, chi8v.v, y0v.v );
            y0v.v = _mm256_fmadd_pd( a09v.v, chi9v.v, y0v.v );

            // Store the output.
            _mm256_storeu_pd( (double *)(y0 + 0*n_elem_per_reg), y0v.v );

            y0 += n_elem_per_reg ;
            a0 += n_elem_per_reg ;
            a1 += n_elem_per_reg ;
            a2 += n_elem_per_reg ;
            a3 += n_elem_per_reg ;
            a4 += n_elem_per_reg ;
        }
        // If there are leftover iterations, perform them with scalar code.
        for ( ; (i + 0) < m ; ++i )
        {
            dcomplex       y0c = *y0;

            const dcomplex a0c = *a0;
            const dcomplex a1c = *a1;
            const dcomplex a2c = *a2;
            const dcomplex a3c = *a3;
            const dcomplex a4c = *a4;

            y0c.real += chi0.real * a0c.real - chi0.imag * a0c.imag * setPlusOne;
            y0c.real += chi1.real * a1c.real - chi1.imag * a1c.imag * setPlusOne;
            y0c.real += chi2.real * a2c.real - chi2.imag * a2c.imag * setPlusOne;
            y0c.real += chi3.real * a3c.real - chi3.imag * a3c.imag * setPlusOne;
            y0c.real += chi4.real * a4c.real - chi4.imag * a4c.imag * setPlusOne;

            y0c.imag += chi0.imag * a0c.real + chi0.real * a0c.imag * setPlusOne;
            y0c.imag += chi1.imag * a1c.real + chi1.real * a1c.imag * setPlusOne;
            y0c.imag += chi2.imag * a2c.real + chi2.real * a2c.imag * setPlusOne;
            y0c.imag += chi3.imag * a3c.real + chi3.real * a3c.imag * setPlusOne;
            y0c.imag += chi4.imag * a4c.real + chi4.real * a4c.imag * setPlusOne;

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
         for ( ; (i + 0) < m ; ++i )
        {
            dcomplex       y0c = *y0;

            const dcomplex a0c = *a0;
            const dcomplex a1c = *a1;
            const dcomplex a2c = *a2;
            const dcomplex a3c = *a3;
            const dcomplex a4c = *a4;

            y0c.real += chi0.real * a0c.real - chi0.imag * a0c.imag * setPlusOne;
            y0c.real += chi1.real * a1c.real - chi1.imag * a1c.imag * setPlusOne;
            y0c.real += chi2.real * a2c.real - chi2.imag * a2c.imag * setPlusOne;
            y0c.real += chi3.real * a3c.real - chi3.imag * a3c.imag * setPlusOne;
            y0c.real += chi4.real * a4c.real - chi4.imag * a4c.imag * setPlusOne;

            y0c.imag += chi0.imag * a0c.real + chi0.real * a0c.imag * setPlusOne;
            y0c.imag += chi1.imag * a1c.real + chi1.real * a1c.imag * setPlusOne;
            y0c.imag += chi2.imag * a2c.real + chi2.real * a2c.imag * setPlusOne;
            y0c.imag += chi3.imag * a3c.real + chi3.real * a3c.imag * setPlusOne;
            y0c.imag += chi4.imag * a4c.real + chi4.real * a4c.imag * setPlusOne;

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

