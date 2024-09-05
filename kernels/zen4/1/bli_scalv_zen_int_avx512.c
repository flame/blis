/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2023 - 2024, Advanced Micro Devices, Inc. All rights reserved.

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
#include <immintrin.h>

/*
    Functionality
    -------------

    This function scales a single precision floating-point vector by an element of the
    same type.

    x := conjalpha(alpha) * x

    Function Signature
    -------------------

    * 'conjalpha' - Variable specified if alpha needs to be conjugated
    * 'n' - Length of the array passed
    * 'alpha' - Pointer to the element by which the vector is to be scaled
    * 'x' - Float pointer pointing to an array
    * 'incx' - Stride to point to the next element in the array
    * 'cntx' - BLIS context object

    Exception
    ----------

    None

    Deviation from BLAS
    --------------------

    Setv is used when alpha=0 unless a negative value of n is supplied.
    This only occurs in calls from BLAS and CBLAS scal APIs.

    Undefined behaviour
    -------------------

    None

*/
void bli_sscalv_zen_int_avx512
        (
          conj_t conjalpha,
          dim_t n,
          float *restrict alpha,
          float *restrict x, inc_t incx,
          cntx_t *restrict cntx
        )
{
    // If the vector dimension is zero, or if alpha is unit, return early.
    if ( bli_zero_dim1( n ) || PASTEMAC(s,eq1)( *alpha ) ) return;

    // If alpha is zero, use setv if not called from BLAS scal itself (indicated by n being negative).
    if ( PASTEMAC(s,eq0)( *alpha ) && n > 0 )
    {
        float *zero = bli_s0;
        if (cntx == NULL) cntx = bli_gks_query_cntx();
        ssetv_ker_ft f = bli_cntx_get_l1v_ker_dt(BLIS_FLOAT, BLIS_SETV_KER, cntx);

        f
        (
          BLIS_NO_CONJUGATE,
          n,
          zero,
          x, incx,
          cntx
        );

        return;
    }

    dim_t n0 = bli_abs(n);

    dim_t i = 0;
    float *restrict x0 = x;

    if (incx == 1)
    {
        // Number of float in AVX-512
        const dim_t n_elem_per_reg = 16;

        __m512 xv[8], alphav;
        alphav = _mm512_set1_ps(*alpha);

        for (i = 0; (i + 127) < n0; i += 128)
        {
            // Loading the input values
            xv[0] = _mm512_loadu_ps(x0 + 0 * n_elem_per_reg);
            xv[1] = _mm512_loadu_ps(x0 + 1 * n_elem_per_reg);
            xv[2] = _mm512_loadu_ps(x0 + 2 * n_elem_per_reg);
            xv[3] = _mm512_loadu_ps(x0 + 3 * n_elem_per_reg);
            xv[4] = _mm512_loadu_ps(x0 + 4 * n_elem_per_reg);
            xv[5] = _mm512_loadu_ps(x0 + 5 * n_elem_per_reg);
            xv[6] = _mm512_loadu_ps(x0 + 6 * n_elem_per_reg);
            xv[7] = _mm512_loadu_ps(x0 + 7 * n_elem_per_reg);

            // perform : x := alpha * x;
            xv[0] = _mm512_mul_ps(alphav, xv[0]);
            xv[1] = _mm512_mul_ps(alphav, xv[1]);
            xv[2] = _mm512_mul_ps(alphav, xv[2]);
            xv[3] = _mm512_mul_ps(alphav, xv[3]);

            _mm512_storeu_ps((x0 + 0 * n_elem_per_reg), xv[0]);
            _mm512_storeu_ps((x0 + 1 * n_elem_per_reg), xv[1]);
            _mm512_storeu_ps((x0 + 2 * n_elem_per_reg), xv[2]);
            _mm512_storeu_ps((x0 + 3 * n_elem_per_reg), xv[3]);

            xv[4] = _mm512_mul_ps(alphav, xv[4]);
            xv[5] = _mm512_mul_ps(alphav, xv[5]);
            xv[6] = _mm512_mul_ps(alphav, xv[6]);
            xv[7] = _mm512_mul_ps(alphav, xv[7]);

            _mm512_storeu_ps((x0 + 4 * n_elem_per_reg), xv[4]);
            _mm512_storeu_ps((x0 + 5 * n_elem_per_reg), xv[5]);
            _mm512_storeu_ps((x0 + 6 * n_elem_per_reg), xv[6]);
            _mm512_storeu_ps((x0 + 7 * n_elem_per_reg), xv[7]);

            x0 += 8 * n_elem_per_reg;
        }

        for (; (i + 63) < n0; i += 64)
        {
            // Loading the input values
            xv[0] = _mm512_loadu_ps(x0 + 0 * n_elem_per_reg);
            xv[1] = _mm512_loadu_ps(x0 + 1 * n_elem_per_reg);
            xv[2] = _mm512_loadu_ps(x0 + 2 * n_elem_per_reg);
            xv[3] = _mm512_loadu_ps(x0 + 3 * n_elem_per_reg);

            // perform : x := alpha * x;
            xv[0] = _mm512_mul_ps(alphav, xv[0]);
            xv[1] = _mm512_mul_ps(alphav, xv[1]);
            xv[2] = _mm512_mul_ps(alphav, xv[2]);
            xv[3] = _mm512_mul_ps(alphav, xv[3]);

            _mm512_storeu_ps((x0 + 0 * n_elem_per_reg), xv[0]);
            _mm512_storeu_ps((x0 + 1 * n_elem_per_reg), xv[1]);
            _mm512_storeu_ps((x0 + 2 * n_elem_per_reg), xv[2]);
            _mm512_storeu_ps((x0 + 3 * n_elem_per_reg), xv[3]);

            x0 += 4 * n_elem_per_reg;
        }

        for (; (i + 31) < n0; i += 32)
        {
            // Loading the input values
            xv[0] = _mm512_loadu_ps(x0 + 0 * n_elem_per_reg);
            xv[1] = _mm512_loadu_ps(x0 + 1 * n_elem_per_reg);

            // perform : x := alpha * x;
            xv[0] = _mm512_mul_ps(alphav, xv[0]);
            xv[1] = _mm512_mul_ps(alphav, xv[1]);

            _mm512_storeu_ps((x0 + 0 * n_elem_per_reg), xv[0]);
            _mm512_storeu_ps((x0 + 1 * n_elem_per_reg), xv[1]);

            x0 += 2 * n_elem_per_reg;
        }

        for (; (i + 15) < n0; i += 16)
        {
            // Loading the input values
            xv[0] = _mm512_loadu_ps(x0 + 0 * n_elem_per_reg);

            // perform : x := alpha * x;
            xv[0] = _mm512_mul_ps(alphav, xv[0]);

            _mm512_storeu_ps((x0 + 0 * n_elem_per_reg), xv[0]);

            x0 += n_elem_per_reg;
        }

        for (; (i + 7) < n0; i += 8)
        {
            // Loading the input values
            __m256 x_vec = _mm256_loadu_ps(x0);

            // perform : x := alpha * x;
            x_vec = _mm256_mul_ps(_mm256_set1_ps(*alpha), x_vec);

            // Store the output.
            _mm256_storeu_ps(x0, x_vec);

            x0 += 8;
        }

        /*
            Issue vzeroupper instruction to clear upper lanes of ymm registers.
            This avoids a performance penalty caused by false dependencies when
            transitioning from from AVX to SSE instructions (which may occur
            later, especially if BLIS is compiled with -mfpmath=sse).
        */
        _mm256_zeroupper();

        for (; (i + 3) < n0; i += 4)
        {
            // Loading the input values
            __m128 x_vec = _mm_loadu_ps(x0);

            // perform : x := alpha * x;
            x_vec = _mm_mul_ps(_mm_set1_ps(*alpha), x_vec);

            // Store the output.
            _mm_storeu_ps(x0, x_vec);

            x0 += 4;
        }
    }

    const float alphac = *alpha;

    for (; i < n0; ++i)
    {
        *x0 *= alphac;

        x0 += incx;
    }
}

// --------------------------------------------------------------------------------------

/*
    Functionality
    -------------

    This function scales a double precision floating-point vector by an element of the
    same type.

    x := conjalpha(alpha) * x

    Function Signature
    -------------------

    * 'conjalpha' - Variable specified if alpha needs to be conjugated
    * 'n' - Length of the array passed
    * 'alpha' - Pointer to the element by which the vector is to be scaled
    * 'x' - Double pointer pointing to an array
    * 'incx' - Stride to point to the next element in the array
    * 'cntx' - BLIS context object

    Exception
    ----------

    None

    Deviation from BLAS
    --------------------

    Setv is used when alpha=0 unless a negative value of n is supplied.
    This only occurs in calls from BLAS and CBLAS scal APIs.

    Undefined behaviour
    -------------------

    None

*/
BLIS_EXPORT_BLIS void bli_dscalv_zen_int_avx512
        (
          conj_t conjalpha,
          dim_t n,
          double *restrict alpha,
          double *restrict x, inc_t incx,
          cntx_t *restrict cntx
        )
{
    // If the vector dimension is zero, or if alpha is unit, return early.
    if ( bli_zero_dim1( n ) || PASTEMAC(d,eq1)( *alpha ) ) return;

    // If alpha is zero, use setv if not called from BLAS scal itself (indicated by n being negative).
    if ( PASTEMAC(d,eq0)( *alpha ) && n > 0 )
    {
        double *zero = bli_d0;
        if (cntx == NULL) cntx = bli_gks_query_cntx();
        dsetv_ker_ft f = bli_cntx_get_l1v_ker_dt(BLIS_DOUBLE, BLIS_SETV_KER, cntx);

        f
        (
          BLIS_NO_CONJUGATE,
          n,
          zero,
          x, incx,
          cntx
        );

        return;
    }

    dim_t n0 = bli_abs(n);

    dim_t i = 0;
    double *restrict x0;

    // Initialize local pointers.
    x0 = x;

    if (incx == 1)
    {
        // Number of double in AVX-512
        const dim_t n_elem_per_reg = 8;

        __m512d alphav;
        alphav = _mm512_set1_pd(*alpha);
        __m512d xv[8];

        for (i = 0; (i + 63) < n0; i += 64)
        {
            // Loading the input values
            xv[0] = _mm512_loadu_pd(x0 + 0 * n_elem_per_reg);
            xv[1] = _mm512_loadu_pd(x0 + 1 * n_elem_per_reg);
            xv[2] = _mm512_loadu_pd(x0 + 2 * n_elem_per_reg);
            xv[3] = _mm512_loadu_pd(x0 + 3 * n_elem_per_reg);
            xv[4] = _mm512_loadu_pd(x0 + 4 * n_elem_per_reg);
            xv[5] = _mm512_loadu_pd(x0 + 5 * n_elem_per_reg);
            xv[6] = _mm512_loadu_pd(x0 + 6 * n_elem_per_reg);
            xv[7] = _mm512_loadu_pd(x0 + 7 * n_elem_per_reg);

            // perform : x := alpha * x;
            xv[0] = _mm512_mul_pd(alphav, xv[0]);
            xv[1] = _mm512_mul_pd(alphav, xv[1]);
            xv[2] = _mm512_mul_pd(alphav, xv[2]);
            xv[3] = _mm512_mul_pd(alphav, xv[3]);

            _mm512_storeu_pd((x0 + 0 * n_elem_per_reg), xv[0]);
            _mm512_storeu_pd((x0 + 1 * n_elem_per_reg), xv[1]);
            _mm512_storeu_pd((x0 + 2 * n_elem_per_reg), xv[2]);
            _mm512_storeu_pd((x0 + 3 * n_elem_per_reg), xv[3]);

            xv[4] = _mm512_mul_pd(alphav, xv[4]);
            xv[5] = _mm512_mul_pd(alphav, xv[5]);
            xv[6] = _mm512_mul_pd(alphav, xv[6]);
            xv[7] = _mm512_mul_pd(alphav, xv[7]);

            _mm512_storeu_pd((x0 + 4 * n_elem_per_reg), xv[4]);
            _mm512_storeu_pd((x0 + 5 * n_elem_per_reg), xv[5]);
            _mm512_storeu_pd((x0 + 6 * n_elem_per_reg), xv[6]);
            _mm512_storeu_pd((x0 + 7 * n_elem_per_reg), xv[7]);

            x0 += 8 * n_elem_per_reg;
        }

        for (; (i + 31) < n0; i += 32)
        {
            // Loading the input values
            xv[0] = _mm512_loadu_pd(x0 + 0 * n_elem_per_reg);
            xv[1] = _mm512_loadu_pd(x0 + 1 * n_elem_per_reg);
            xv[2] = _mm512_loadu_pd(x0 + 2 * n_elem_per_reg);
            xv[3] = _mm512_loadu_pd(x0 + 3 * n_elem_per_reg);

            // perform : x := alpha * x;
            xv[0] = _mm512_mul_pd(alphav, xv[0]);
            xv[1] = _mm512_mul_pd(alphav, xv[1]);
            xv[2] = _mm512_mul_pd(alphav, xv[2]);
            xv[3] = _mm512_mul_pd(alphav, xv[3]);

            _mm512_storeu_pd((x0 + 0 * n_elem_per_reg), xv[0]);
            _mm512_storeu_pd((x0 + 1 * n_elem_per_reg), xv[1]);
            _mm512_storeu_pd((x0 + 2 * n_elem_per_reg), xv[2]);
            _mm512_storeu_pd((x0 + 3 * n_elem_per_reg), xv[3]);

            x0 += 4 * n_elem_per_reg;
        }

        for (; (i + 15) < n0; i += 16)
        {
            // Loading the input values
            xv[0] = _mm512_loadu_pd(x0 + 0 * n_elem_per_reg);
            xv[1] = _mm512_loadu_pd(x0 + 1 * n_elem_per_reg);

            // perform : x := alpha * x;
            xv[0] = _mm512_mul_pd(alphav, xv[0]);
            xv[1] = _mm512_mul_pd(alphav, xv[1]);

            _mm512_storeu_pd((x0 + 0 * n_elem_per_reg), xv[0]);
            _mm512_storeu_pd((x0 + 1 * n_elem_per_reg), xv[1]);

            x0 += 2 * n_elem_per_reg;
        }

        for (; (i + 7) < n0; i += 8)
        {
            // Loading the input values
            xv[0] = _mm512_loadu_pd(x0 + 0 * n_elem_per_reg);

            // perform : x := alpha * x;
            xv[0] = _mm512_mul_pd(alphav, xv[0]);

            _mm512_storeu_pd((x0 + 0 * n_elem_per_reg), xv[0]);

            x0 += n_elem_per_reg;
        }

        for (; (i + 3) < n0; i += 4)
        {
            // Loading the input values
            __m256d x_vec = _mm256_loadu_pd(x0);

            // perform : x := alpha * x;
            x_vec = _mm256_mul_pd(_mm256_set1_pd(*alpha), x_vec);

            // Store the output.
            _mm256_storeu_pd(x0, x_vec);

            x0 += 4;
        }

        /*
           Issue vzeroupper instruction to clear upper lanes of ymm registers.
           This avoids a performance penalty caused by false dependencies when
           transitioning from from AVX to SSE instructions (which may occur
           later, especially if BLIS is compiled with -mfpmath=sse).
       */
        _mm256_zeroupper();

        for (; (i + 1) < n0; i += 2)
        {
            // Loading the input values
            __m128d x_vec = _mm_loadu_pd(x0);

            // perform : x := alpha * x;
            x_vec = _mm_mul_pd(_mm_set1_pd(*alpha), x_vec);

            // Store the output.
            _mm_storeu_pd(x0, x_vec);

            x0 += 2;
        }
    }

    const double alphac = *alpha;

    for (; i < n0; ++i)
    {
        *x0 *= alphac;

        x0 += incx;
    }
}

/*
    Functionality
    -------------

    This function scales a double complex vector by an element of the
    type double.

    x := conjalpha(alpha) * x

    Function Signature
    -------------------

    * 'conjalpha' - Variable specified if alpha needs to be conjugated
    * 'n' - Length of the array passed
    * 'alpha' - Pointer to the element by which the vector is to be scaled
    * 'x' - Double complex pointer pointing to an array
    * 'incx' - Stride to point to the next element in the array
    * 'cntx' - BLIS context object

    Exception
    ----------

    None

    Deviation from BLAS
    --------------------

    Setv is used when alpha=0 unless a negative value of n is supplied.
    This only occurs in calls from BLAS and CBLAS scal APIs.

    Undefined behaviour
    -------------------

    None

*/
void bli_zdscalv_zen_int_avx512
     (
       conj_t           conjalpha,
       dim_t            n,
       dcomplex* restrict alpha,
       dcomplex* restrict x, inc_t incx,
       cntx_t* restrict cntx
     )
{
    /*
        This kernel only performs the computation
        when alpha is double from the BLAS layer
        alpha is passed as double complex to adhere
        to function pointer definition in BLIS
    */

    // If the vector dimension is zero, or if alpha is unit, return early.
    if ( bli_zero_dim1( n ) || PASTEMAC(z,eq1)( *alpha ) ) return;

    // If alpha is zero, use setv if not called from BLAS scal itself (indicated by n being negative).
    if ( PASTEMAC(z,eq0)( *alpha ) && n > 0 )
    {
        // Expert interface of setv is invoked when alpha is zero
        dcomplex *zero = bli_z0;

        /* When alpha is zero all the element in x are set to zero */
        PASTEMAC2(z, setv, BLIS_TAPI_EX_SUF)
        (
            BLIS_NO_CONJUGATE,
            n,
            zero,
            x, incx,
            cntx,
            NULL);

        return;
    }

    dim_t n0 = bli_abs(n);

    const double alphac = (*alpha).real;

    dim_t i = 0;

    double *restrict x0 = (double *)x;

    if (incx == 1)
    {
        __m512d alphav, xv[4];
        const dim_t n_elem_per_reg = 8; // number of elements per register

        alphav = _mm512_set1_pd(alphac);

        for (; (i + 15) < n0; i += 16)
        {
            xv[0] = _mm512_loadu_pd(x0);
            xv[1] = _mm512_loadu_pd(x0 + n_elem_per_reg);
            xv[2] = _mm512_loadu_pd(x0 + 2 * n_elem_per_reg);
            xv[3] = _mm512_loadu_pd(x0 + 3 * n_elem_per_reg);

            xv[0] = _mm512_mul_pd(alphav, xv[0]);
            xv[1] = _mm512_mul_pd(alphav, xv[1]);
            xv[2] = _mm512_mul_pd(alphav, xv[2]);
            xv[3] = _mm512_mul_pd(alphav, xv[3]);

            _mm512_storeu_pd(x0, xv[0]);
            _mm512_storeu_pd(x0 + n_elem_per_reg, xv[1]);
            _mm512_storeu_pd(x0 + 2 * n_elem_per_reg, xv[2]);
            _mm512_storeu_pd(x0 + 3 * n_elem_per_reg, xv[3]);

            x0 += 4 * n_elem_per_reg;
        }

        for (; (i + 7) < n0; i += 8)
        {
            xv[0] = _mm512_loadu_pd(x0);
            xv[1] = _mm512_loadu_pd(x0 + n_elem_per_reg);

            xv[0] = _mm512_mul_pd(alphav, xv[0]);
            xv[1] = _mm512_mul_pd(alphav, xv[1]);

            _mm512_storeu_pd(x0, xv[0]);
            _mm512_storeu_pd(x0 + n_elem_per_reg, xv[1]);

            x0 += 2 * n_elem_per_reg;
        }

        for (; (i + 3) < n0; i += 4)
        {
            xv[0] = _mm512_loadu_pd(x0);

            xv[0] = _mm512_mul_pd(alphav, xv[0]);

            _mm512_storeu_pd(x0, xv[0]);

            x0 += n_elem_per_reg;
        }

        for (; (i + 1) < n0; i += 2)
        {
            __m256d xv = _mm256_loadu_pd(x0);

            __m256d alphav = _mm256_set1_pd(alphac);

            xv = _mm256_mul_pd(alphav, xv);

            _mm256_storeu_pd(x0, xv);

            x0 += 4;
        }

        // Issue vzeroupper instruction to clear upper lanes of ymm registers.
        // This avoids a performance penalty caused by false dependencies when
        // transitioning from AVX to SSE instructions (which may occur as soon
        // as the n_left cleanup loop below if BLIS is compiled with
        // -mfpmath=sse).
        _mm256_zeroupper();
    }

    /* In double complex data type the computation of
    unit stride elements can still be vectorized using SSE*/
    __m128d alpha_reg, x_vec;

    alpha_reg = _mm_set1_pd((*alpha).real);

    for (; i < n0; ++i)
    {
        x_vec = _mm_loadu_pd(x0);

        x_vec = _mm_mul_pd(x_vec, alpha_reg);

        _mm_storeu_pd(x0, x_vec);

        x0 += 2 * incx;
    }
}


#define MICRO_OP( r0, r1, r2, r3 ) \
    /**
     * Loading 8 scomplex (16 float) elements from x to each zmm register.
     * xv[0] = x0R x0I x1R x1I x2R x2I x3R x3I ...
     */ \
    xv[r0] = _mm512_loadu_ps( x0 + r0*n_elem_per_reg ); \
    xv[r1] = _mm512_loadu_ps( x0 + r1*n_elem_per_reg ); \
    xv[r2] = _mm512_loadu_ps( x0 + r2*n_elem_per_reg ); \
    xv[r3] = _mm512_loadu_ps( x0 + r3*n_elem_per_reg ); \
    \
    /**
     * Using itermediate ZMM register to interchange real and imaginary
     * values of each element in xv register.
     * inter[0] = x0I x0R x1I x1R x2I x2R x3I x3R...
     */ \
    inter[r0] = _mm512_permute_ps( xv[r0], 0xB1 ); \
    inter[r1] = _mm512_permute_ps( xv[r1], 0xB1 ); \
    inter[r2] = _mm512_permute_ps( xv[r2], 0xB1 ); \
    inter[r3] = _mm512_permute_ps( xv[r3], 0xB1 ); \
    \
    /**
     * Scaling intermediate vector with imaginary part of alpha.
     * inter[0] = inter[0] * alphaI
     *          = x0I*alphaI x0R*alphaI x1I*alphaI x1R*alphaI ...
     */ \
    \
    inter[r0] = _mm512_mul_ps( inter[r0], alphaIv ); \
    inter[r1] = _mm512_mul_ps( inter[r1], alphaIv ); \
    inter[r2] = _mm512_mul_ps( inter[r2], alphaIv ); \
    inter[r3] = _mm512_mul_ps( inter[r3], alphaIv ); \
    \
    /**
     * Scaling xv with real part of alpha and doing alternatively sub-add of
     * the scaled intermediate register. The fmaddsub operation will
     * alternatively add and subtract elements in inter[0] from alphaRv*xv[0].
     * xv[0]    = xv[0] * alphaR -/+ inter[0]
     *          = x0R*alphaR - x0I*alphaI x0I*alphaR + x0R*alphaI
     *            x1R*alphaR - x1I*alphaI x1I*alphaR + x1R*alphaI ...
     */ \
    xv[r0] = _mm512_fmaddsub_ps( alphaRv, xv[r0], inter[r0] ); \
    xv[r1] = _mm512_fmaddsub_ps( alphaRv, xv[r1], inter[r1] ); \
    xv[r2] = _mm512_fmaddsub_ps( alphaRv, xv[r2], inter[r2] ); \
    xv[r3] = _mm512_fmaddsub_ps( alphaRv, xv[r3], inter[r3] ); \
    \
    /**
     * Storing the scaled vector back to x0.
     */ \
    _mm512_storeu_ps( x0 + r0*n_elem_per_reg, xv[r0] ); \
    _mm512_storeu_ps( x0 + r1*n_elem_per_reg, xv[r1] ); \
    _mm512_storeu_ps( x0 + r2*n_elem_per_reg, xv[r2] ); \
    _mm512_storeu_ps( x0 + r3*n_elem_per_reg, xv[r3] );

/*
    Functionality
    -------------

    This function scales a single complex vector by an element of the
    type single complex.

    x := conjalpha(alpha) * x

    Function Signature
    -------------------

    * 'conjalpha' - Variable specified if alpha needs to be conjugated
    * 'n' - Length of the array passed
    * 'alpha' - Pointer to the element by which the vector is to be scaled
    * 'x' - Single complex pointer pointing to an array
    * 'incx' - Stride to point to the next element in the array
    * 'cntx' - BLIS context object

    Exception
    ----------

    None

    Deviation from BLAS
    --------------------

    1. The kernel invokes SETV when alpha scalar is zero and explicitly sets all
       elements to zero thus, not propagating any NaNs/Infs.

    Undefined behaviour
    -------------------

    None

*/
void bli_cscalv_zen_int_avx512
     (
       conj_t             conjalpha,
       dim_t              n,
       scomplex* restrict alpha,
       scomplex* restrict x, inc_t incx,
       cntx_t*   restrict cntx
     )
{
    // If the vector dimension is zero, or if alpha is unit, return early.
    if ( bli_zero_dim1( n ) || PASTEMAC(c,eq1)( *alpha ) ) return;

    // If alpha is zero, use setv if not called from BLAS scal itself (indicated by n being negative).
    if ( PASTEMAC(c,eq0)( *alpha ) && n > 0 )
    {
        // Expert interface of setv is invoked when alpha is zero
        scomplex *zero = bli_c0;

        /* When alpha is zero all the element in x are set to zero */
        PASTEMAC2(c, setv, BLIS_TAPI_EX_SUF)
        (
          BLIS_NO_CONJUGATE,
          n,
          zero,
          x, incx,
          cntx,
          NULL
        );

        return;
    }

    dim_t n0 = bli_abs(n);

    dim_t i = 0;
    scomplex alpha_conj;
    float* restrict x0 = (float*) x;

    // Performs conjugation of alpha based on conjalpha.
    PASTEMAC(c,copycjs)( conjalpha, *alpha, alpha_conj );

    const float alphaR = alpha_conj.real;
    const float alphaI = alpha_conj.imag;

    if ( incx == 1 )
    {
        // number of elements per register.
        const dim_t n_elem_per_reg = 16;

        __m512 alphaRv, alphaIv;

        // Broadcast real and imaginary values of alpha.
        alphaRv = _mm512_set1_ps( alphaR );
        alphaIv = _mm512_set1_ps( alphaI );

        /**
         * General Algorithm:
         *
         * Broadcasting real and imaginary parts of alpha scalar to separate
         * zmm registers, alphaRv and alphaIv, respectively.
         * alphaRv  = alphaR alphaR alphaR alphaR ...
         * alphaIv  = alphaI alphaI alphaI alphaI ...
         *
         * Loading 8 scomplex (16 float) elements from x to each zmm register.
         * xv[0] = x0R x0I x1R x1I x2R x2I x3R x3I ...
         *
         * Using itermediate ZMM register to interchange real and imaginary
         * values of each element in xv register.
         * inter[0] = x0I x0R x1I x1R x2I x2R x3I x3R...
         *
         * Scaling the intermediate register with imaginary part of alpha.
         * inter[0] = inter[0] * alphaI
         *          = x0I*alphaI x0R*alphaI x1I*alphaI x1R*alphaI ...
         *
         * Scaling xv with real part of alpha and doing alternatively sub-add of
         * the scaled intermediate register.
         * xv[0]    = xv[0] * alphaR -/+ inter[0]
         *          = x0R*alphaR - x0I*alphaI x0I*alphaR + x0R*alphaI
         *            x1R*alphaR - x1I*alphaI x1I*alphaR + x1R*alphaI ...
         */

        // Processing 96 scomplex elements (192 floats) per iteration
        for ( ; (i + 95) < n0; i += 96 )
        {
            __m512 xv[12], inter[12];

            MICRO_OP( 0, 1, 2, 3 )

            MICRO_OP( 4, 5, 6, 7 )

            MICRO_OP( 8, 9, 10, 11 )

            // Incrementing x0 by 12*n_elem_per_reg, 192 floats
            // or 96 scomplex elements.
            x0 += 12 * n_elem_per_reg;
        }

        // Processing 64 scomplex elements (128 floats) per iteration
        for ( ; (i + 63) < n0; i += 64 )
        {
            __m512 xv[8], inter[8];

            MICRO_OP( 0, 1, 2, 3 )

            MICRO_OP( 4, 5, 6, 7 )

            // Incrementing x0 by 8*n_elem_per_reg, 128 floats
            // or 64 scomplex elements.
            x0 += 8 * n_elem_per_reg;
        }

        // Processing 32 scomplex elements (64 floats) per iteration
        for ( ; (i + 31) < n0; i += 32 )
        {
            __m512 xv[4], inter[4];

            MICRO_OP( 0, 1, 2, 3 )

            // Incrementing x0 by 4*n_elem_per_reg, 64 floats
            // or 32 scomplex elements.
            x0 += 4 * n_elem_per_reg;
        }

        // Processing 16 scomplex elements (32 floats) per iteration
        for ( ; (i + 15) < n0; i += 16 )
        {
            __m512 xv[2], inter[2];

            // Loading 8 scomplex (16 float) elements from x to each
            // zmm register.
            // xv[0] = x0R x0I x1R x1I x2R x2I x3R x3I ...
            xv[0] = _mm512_loadu_ps( x0 );
            xv[1] = _mm512_loadu_ps( x0 + 1*n_elem_per_reg );

            // Permuting xv and storing into intermediate vector.
            // inter[0] = x0I x0R x1I x1R x2I x2R x3I x3R...
            inter[0] = _mm512_permute_ps( xv[0], 0xB1 );
            inter[1] = _mm512_permute_ps( xv[1], 0xB1 );

            // Scaling intermediate vector with imaginary part of alpha.
            // inter[0] = inter[0] * alphaI
            //          = x0I*alphaI x0R*alphaI x1I*alphaI x1R*alphaI ...
            inter[0] = _mm512_mul_ps( inter[0], alphaIv );
            inter[1] = _mm512_mul_ps( inter[1], alphaIv );

            // Performing the fmaddsub operation to get resultant x scaled by
            // alpha. The fmaddsub operation will alternatively add and subtract
            // elements in inter[0] from alphaRv*xv[0].
            // xv[0]    = xv[0] * alphaR -/+ inter[0]
            //          = x0R*alphaR - x0I*alphaI x0I*alphaR + x0R*alphaI
            //            x1R*alphaR - x1I*alphaI x1I*alphaR + x1R*alphaI ...
            xv[0] = _mm512_fmaddsub_ps( alphaRv, xv[0], inter[0] );
            xv[1] = _mm512_fmaddsub_ps( alphaRv, xv[1], inter[1] );

            // Storing the scaled vector back to x0.
            _mm512_storeu_ps( x0, xv[0] );
            _mm512_storeu_ps( x0 + 1*n_elem_per_reg, xv[1] );

            // Incrementing x0 by 2*n_elem_per_reg, 32 floats
            // or 16 scomplex elements.
            x0 += 2 * n_elem_per_reg;
        }

        // Processing 8 scomplex elements (16 floats) per iteration
        for ( ; (i + 7) < n0; i += 8 )
        {
            __m512 xv[1], inter[1];

            // Loading 8 scomplex (16 float) elements from x to each
            // zmm register.
            // xv[0] = x0R x0I x1R x1I x2R x2I x3R x3I ...
            xv[0] = _mm512_loadu_ps( x0 );

            // Permuting xv and storing into intermediate zmm register.
            // inter[0] = x0I x0R x1I x1R x2I x2R x3I x3R...
            inter[0] = _mm512_permute_ps( xv[0], 0xB1 );

            // Scaling intermediate register with imaginary part of alpha.
            // inter[0] = inter[0] * alphaI
            //          = x0I*alphaI x0R*alphaI x1I*alphaI x1R*alphaI ...
            inter[0] = _mm512_mul_ps( inter[0], alphaIv );

            // Performing the fmaddsub operation to get resultant x scaled by
            // alpha. The fmaddsub operation will alternatively add and subtract
            // elements in inter[0] from alphaRv*xv[0].
            // xv[0]    = xv[0] * alphaR -/+ inter[0]
            //          = x0R*alphaR - x0I*alphaI x0I*alphaR + x0R*alphaI
            //            x1R*alphaR - x1I*alphaI x1I*alphaR + x1R*alphaI ...
            xv[0] = _mm512_fmaddsub_ps( alphaRv, xv[0], inter[0] );

            // Storing the scaled vector back to x0.
            _mm512_storeu_ps( x0, xv[0] );

            // Incrementing x0 by n_elem_per_reg, 16 floats
            // or 8 scomplex elements.
            x0 += n_elem_per_reg;
        }

        // Processing remaining elements, if any.
        if ( i < n0 )
        {
            // Setting the mask bit based on remaining elements.
            // Since each scomplex element corresponds to 2 floats,
            // we need to load and store 2*(n0-i) elements.

            __mmask16 mask = ( 1 << ( 2 * ( n0 - i ) ) ) - 1;

            __m512 xv, temp;

            xv = _mm512_maskz_loadu_ps( mask, x0 );

            temp = _mm512_permute_ps( xv, 0xB1 );

            temp = _mm512_mul_ps( alphaIv, temp );

            xv = _mm512_fmaddsub_ps( alphaRv, xv, temp );

            _mm512_mask_storeu_ps( x0, mask, xv );
        }
    }
    else    // if ( incx != 1 )
    {
        const float alphaR = alpha_conj.real;
        const float alphaI = alpha_conj.imag;

        float x0R, x0I;
        for (; i < n0; ++i)
        {
            x0R = *(x0);
            x0I = *(x0 + 1);

            *(x0)     = x0R * alphaR - x0I * alphaI;
            *(x0 + 1) = x0R * alphaI + x0I * alphaR;

            x0 += 2*incx;
        }
    }
}

/*
    Functionality
    -------------

    This function scales a double complex vector by an element of the
    type double complex.

    x := conjalpha(alpha) * x

    Function Signature
    -------------------

    * 'conjalpha' - Variable specified if alpha needs to be conjugated
    * 'n' - Length of the array passed
    * 'alpha' - Pointer to the element by which the vector is to be scaled
    * 'x' - Double complex pointer pointing to an array
    * 'incx' - Stride to point to the next element in the array
    * 'cntx' - BLIS context object

    Exception
    ----------

    None

    Deviation from BLAS
    --------------------

    Setv is used when alpha=0 unless a negative value of n is supplied.
    This only occurs in calls from BLAS and CBLAS scal APIs.

    Undefined behaviour
    -------------------

    None

*/
void bli_zscalv_zen_int_avx512
     (
       conj_t           conjalpha,
       dim_t            n,
       dcomplex* restrict alpha,
       dcomplex* restrict x, inc_t incx,
       cntx_t*   restrict cntx
     )
{
    // If the vector dimension is zero, or if alpha is unit, return early.
    if ( bli_zero_dim1( n ) || PASTEMAC(z,eq1)( *alpha ) ) return;

    // If alpha is zero, use setv if not called from BLAS scal itself (indicated by n being negative).
    if (PASTEMAC(z,eq0)( *alpha ) && n > 0 )
    {
        // Expert interface of setv is invoked when alpha is zero
        dcomplex *zero = bli_z0;

        /* When alpha is zero all the element in x are set to zero */
        PASTEMAC2(z, setv, BLIS_TAPI_EX_SUF)
        (
            BLIS_NO_CONJUGATE,
            n,
            zero,
            x, incx,
            cntx,
            NULL);

        return;
    }

    dim_t n0 = bli_abs(n);

    dim_t i = 0;
    dcomplex alpha_conj;
    double *restrict x0 = (double *)x;

    // Performs conjugation of alpha based on conjalpha
    PASTEMAC(z, copycjs)(conjalpha, *alpha, alpha_conj)

    const double alphaR = alpha_conj.real;
    const double alphaI = alpha_conj.imag;

    if (incx == 1)
    {
        __m512d alphaRv, alphaIv;
        const dim_t n_elem_per_reg = 8;     // number of elements per register

        // Broadcast real and imaginary values of alpha to separate registers.
        // alphaRv = alphaR alphaR alphaR alphaR ...
        // alphaIv = alphaI alphaI alphaI alphaI ...
        alphaRv = _mm512_set1_pd(alphaR);
        alphaIv = _mm512_set1_pd(alphaI);

        /**
         * General Algorithm:
         *
         * alphaRv = alphaR alphaR alphaR alphaR ...
         * alphaIv = alphaI alphaI alphaI alphaI ...
         *
         * xv[0]   = x0R x0I x1R x1I ...
         * temp[0] = x0I x0R x1I x1R ...
         * temp[0] = temp[0] * xv[0]
         *         = x0I*alphaI x0R*alphaI x1I*alphaI x1R*alphaI ...
         * xv[0]   = xv[0] * alphaR + temp[0]
         *         = x0R*alphaR + x0I*alphaI x0I*alphaR + x0R*alphaI
         *           x1R*alphaR + x1I*alphaI x1I*alphaR + x1R*alphaI ...
        */

        // Processing 48 dcomplex elements per iteration.
        for (; (i + 47) < n0; i += 48)
        {
            __m512d xv[12], temp[12];

            // Load elements from x vector.
            xv[0] = _mm512_loadu_pd(x0);
            xv[1] = _mm512_loadu_pd(x0 + n_elem_per_reg);
            xv[2] = _mm512_loadu_pd(x0 + 2 * n_elem_per_reg);
            xv[3] = _mm512_loadu_pd(x0 + 3 * n_elem_per_reg);

            // Operation: xv -> xv'
            // xv  = y0R y0I y1R y1I ...
            // xv' = y0I y0R y1I y1R ...
            temp[0] = _mm512_permute_pd(xv[0], 0x55);
            temp[1] = _mm512_permute_pd(xv[1], 0x55);
            temp[2] = _mm512_permute_pd(xv[2], 0x55);
            temp[3] = _mm512_permute_pd(xv[3], 0x55);

            // Operation: temp = temp * alphaIv
            // temp = x0I*alphaI x0R*alphaI x1I*alphaI x1R*alphaI ...
            temp[0] = _mm512_mul_pd(alphaIv, temp[0]);
            temp[1] = _mm512_mul_pd(alphaIv, temp[1]);
            temp[2] = _mm512_mul_pd(alphaIv, temp[2]);
            temp[3] = _mm512_mul_pd(alphaIv, temp[3]);

            // Operation: xv[0] = xv[0] * alphaR + temp[0]
            // xv[0] = x0R*alphaR + x0I*alphaI x0I*alphaR + x0R*alphaI
            //         x1R*alphaR + x1I*alphaI x1I*alphaR + x1R*alphaI ...
            xv[0] = _mm512_fmaddsub_pd(alphaRv, xv[0], temp[0]);
            xv[1] = _mm512_fmaddsub_pd(alphaRv, xv[1], temp[1]);
            xv[2] = _mm512_fmaddsub_pd(alphaRv, xv[2], temp[2]);
            xv[3] = _mm512_fmaddsub_pd(alphaRv, xv[3], temp[3]);

            // Store result to memory.
            _mm512_storeu_pd(x0, xv[0]);
            _mm512_storeu_pd(x0 + n_elem_per_reg, xv[1]);
            _mm512_storeu_pd(x0 + 2 * n_elem_per_reg, xv[2]);
            _mm512_storeu_pd(x0 + 3 * n_elem_per_reg, xv[3]);

            xv[4] = _mm512_loadu_pd(x0 + 4 * n_elem_per_reg);
            xv[5] = _mm512_loadu_pd(x0 + 5 * n_elem_per_reg);
            xv[6] = _mm512_loadu_pd(x0 + 6 * n_elem_per_reg);
            xv[7] = _mm512_loadu_pd(x0 + 7 * n_elem_per_reg);

            temp[4] = _mm512_permute_pd(xv[4], 0x55);
            temp[5] = _mm512_permute_pd(xv[5], 0x55);
            temp[6] = _mm512_permute_pd(xv[6], 0x55);
            temp[7] = _mm512_permute_pd(xv[7], 0x55);

            temp[4] = _mm512_mul_pd(alphaIv, temp[4]);
            temp[5] = _mm512_mul_pd(alphaIv, temp[5]);
            temp[6] = _mm512_mul_pd(alphaIv, temp[6]);
            temp[7] = _mm512_mul_pd(alphaIv, temp[7]);

            xv[4] = _mm512_fmaddsub_pd(alphaRv, xv[4], temp[4]);
            xv[5] = _mm512_fmaddsub_pd(alphaRv, xv[5], temp[5]);
            xv[6] = _mm512_fmaddsub_pd(alphaRv, xv[6], temp[6]);
            xv[7] = _mm512_fmaddsub_pd(alphaRv, xv[7], temp[7]);

            _mm512_storeu_pd(x0 + 4 * n_elem_per_reg, xv[4]);
            _mm512_storeu_pd(x0 + 5 * n_elem_per_reg, xv[5]);
            _mm512_storeu_pd(x0 + 6 * n_elem_per_reg, xv[6]);
            _mm512_storeu_pd(x0 + 7 * n_elem_per_reg, xv[7]);

            xv[8] = _mm512_loadu_pd(x0 + 8 * n_elem_per_reg);
            xv[9] = _mm512_loadu_pd(x0 + 9 * n_elem_per_reg);
            xv[10] = _mm512_loadu_pd(x0 + 10 * n_elem_per_reg);
            xv[11] = _mm512_loadu_pd(x0 + 11 * n_elem_per_reg);

            temp[8] = _mm512_permute_pd(xv[8], 0x55);
            temp[9] = _mm512_permute_pd(xv[9], 0x55);
            temp[10] = _mm512_permute_pd(xv[10], 0x55);
            temp[11] = _mm512_permute_pd(xv[11], 0x55);

            temp[8] = _mm512_mul_pd(alphaIv, temp[8]);
            temp[9] = _mm512_mul_pd(alphaIv, temp[9]);
            temp[10] = _mm512_mul_pd(alphaIv, temp[10]);
            temp[11] = _mm512_mul_pd(alphaIv, temp[11]);

            xv[8] = _mm512_fmaddsub_pd(alphaRv, xv[8], temp[8]);
            xv[9] = _mm512_fmaddsub_pd(alphaRv, xv[9], temp[9]);
            xv[10] = _mm512_fmaddsub_pd(alphaRv, xv[10], temp[10]);
            xv[11] = _mm512_fmaddsub_pd(alphaRv, xv[11], temp[11]);

            _mm512_storeu_pd(x0 + 8 * n_elem_per_reg, xv[8]);
            _mm512_storeu_pd(x0 + 9 * n_elem_per_reg, xv[9]);
            _mm512_storeu_pd(x0 + 10 * n_elem_per_reg, xv[10]);
            _mm512_storeu_pd(x0 + 11 * n_elem_per_reg, xv[11]);

            // Increment x0 vector pointer.
            x0 += 12 * n_elem_per_reg;
        }

        // Processing 32 dcomplex elements per iteration.
        for (; (i + 31) < n0; i += 32)
        {
            __m512d xv[8], temp[8];
            xv[0] = _mm512_loadu_pd(x0);
            xv[1] = _mm512_loadu_pd(x0 + n_elem_per_reg);
            xv[2] = _mm512_loadu_pd(x0 + 2 * n_elem_per_reg);
            xv[3] = _mm512_loadu_pd(x0 + 3 * n_elem_per_reg);

            temp[0] = _mm512_permute_pd(xv[0], 0x55);
            temp[1] = _mm512_permute_pd(xv[1], 0x55);
            temp[2] = _mm512_permute_pd(xv[2], 0x55);
            temp[3] = _mm512_permute_pd(xv[3], 0x55);

            temp[0] = _mm512_mul_pd(alphaIv, temp[0]);
            temp[1] = _mm512_mul_pd(alphaIv, temp[1]);
            temp[2] = _mm512_mul_pd(alphaIv, temp[2]);
            temp[3] = _mm512_mul_pd(alphaIv, temp[3]);

            xv[0] = _mm512_fmaddsub_pd(alphaRv, xv[0], temp[0]);
            xv[1] = _mm512_fmaddsub_pd(alphaRv, xv[1], temp[1]);
            xv[2] = _mm512_fmaddsub_pd(alphaRv, xv[2], temp[2]);
            xv[3] = _mm512_fmaddsub_pd(alphaRv, xv[3], temp[3]);

            _mm512_storeu_pd(x0, xv[0]);
            _mm512_storeu_pd(x0 + n_elem_per_reg, xv[1]);
            _mm512_storeu_pd(x0 + 2 * n_elem_per_reg, xv[2]);
            _mm512_storeu_pd(x0 + 3 * n_elem_per_reg, xv[3]);

            xv[4] = _mm512_loadu_pd(x0 + 4 * n_elem_per_reg);
            xv[5] = _mm512_loadu_pd(x0 + 5 * n_elem_per_reg);
            xv[6] = _mm512_loadu_pd(x0 + 6 * n_elem_per_reg);
            xv[7] = _mm512_loadu_pd(x0 + 7 * n_elem_per_reg);

            temp[4] = _mm512_permute_pd(xv[4], 0x55);
            temp[5] = _mm512_permute_pd(xv[5], 0x55);
            temp[6] = _mm512_permute_pd(xv[6], 0x55);
            temp[7] = _mm512_permute_pd(xv[7], 0x55);

            temp[4] = _mm512_mul_pd(alphaIv, temp[4]);
            temp[5] = _mm512_mul_pd(alphaIv, temp[5]);
            temp[6] = _mm512_mul_pd(alphaIv, temp[6]);
            temp[7] = _mm512_mul_pd(alphaIv, temp[7]);

            xv[4] = _mm512_fmaddsub_pd(alphaRv, xv[4], temp[4]);
            xv[5] = _mm512_fmaddsub_pd(alphaRv, xv[5], temp[5]);
            xv[6] = _mm512_fmaddsub_pd(alphaRv, xv[6], temp[6]);
            xv[7] = _mm512_fmaddsub_pd(alphaRv, xv[7], temp[7]);

            _mm512_storeu_pd(x0 + 4 * n_elem_per_reg, xv[4]);
            _mm512_storeu_pd(x0 + 5 * n_elem_per_reg, xv[5]);
            _mm512_storeu_pd(x0 + 6 * n_elem_per_reg, xv[6]);
            _mm512_storeu_pd(x0 + 7 * n_elem_per_reg, xv[7]);

            x0 += 8 * n_elem_per_reg;
        }

        // Processing 16 dcomplex elements per iteration.
        for (; (i + 15) < n0; i += 16)
        {
            __m512d xv[4], temp[4];
            xv[0] = _mm512_loadu_pd(x0);
            xv[1] = _mm512_loadu_pd(x0 + n_elem_per_reg);
            xv[2] = _mm512_loadu_pd(x0 + 2 * n_elem_per_reg);
            xv[3] = _mm512_loadu_pd(x0 + 3 * n_elem_per_reg);

            temp[0] = _mm512_permute_pd(xv[0], 0x55);
            temp[1] = _mm512_permute_pd(xv[1], 0x55);
            temp[2] = _mm512_permute_pd(xv[2], 0x55);
            temp[3] = _mm512_permute_pd(xv[3], 0x55);

            temp[0] = _mm512_mul_pd(alphaIv, temp[0]);
            temp[1] = _mm512_mul_pd(alphaIv, temp[1]);
            temp[2] = _mm512_mul_pd(alphaIv, temp[2]);
            temp[3] = _mm512_mul_pd(alphaIv, temp[3]);

            xv[0] = _mm512_fmaddsub_pd(alphaRv, xv[0], temp[0]);
            xv[1] = _mm512_fmaddsub_pd(alphaRv, xv[1], temp[1]);
            xv[2] = _mm512_fmaddsub_pd(alphaRv, xv[2], temp[2]);
            xv[3] = _mm512_fmaddsub_pd(alphaRv, xv[3], temp[3]);

            _mm512_storeu_pd(x0, xv[0]);
            _mm512_storeu_pd(x0 + n_elem_per_reg, xv[1]);
            _mm512_storeu_pd(x0 + 2 * n_elem_per_reg, xv[2]);
            _mm512_storeu_pd(x0 + 3 * n_elem_per_reg, xv[3]);

            x0 += 4 * n_elem_per_reg;
        }

        // Processing 8 dcomplex elements per iteration.
        for (; (i + 7) < n0; i += 8)
        {
            __m512d xv[2], temp[2];
            xv[0] = _mm512_loadu_pd(x0);
            xv[1] = _mm512_loadu_pd(x0 + n_elem_per_reg);

            temp[0] = _mm512_permute_pd(xv[0], 0x55);
            temp[1] = _mm512_permute_pd(xv[1], 0x55);

            temp[0] = _mm512_mul_pd(alphaIv, temp[0]);
            temp[1] = _mm512_mul_pd(alphaIv, temp[1]);

            xv[0] = _mm512_fmaddsub_pd(alphaRv, xv[0], temp[0]);
            xv[1] = _mm512_fmaddsub_pd(alphaRv, xv[1], temp[1]);

            _mm512_storeu_pd(x0, xv[0]);
            _mm512_storeu_pd(x0 + n_elem_per_reg, xv[1]);

            x0 += 2 * n_elem_per_reg;
        }

        // Processing 4 dcomplex elements per iteration.
        for (; (i + 3) < n0; i += 4)
        {
            __m512d xv, temp;
            xv = _mm512_loadu_pd(x0);

            temp = _mm512_permute_pd(xv, 0x55);

            temp = _mm512_mul_pd(alphaIv, temp);

            xv = _mm512_fmaddsub_pd(alphaRv, xv, temp);

            _mm512_storeu_pd(x0, xv);

            x0 += n_elem_per_reg;
        }

        // Processing the remainder elements.
        if( i < n0 )
        {
            // Setting the mask bit based on remaining elements
            // Since each dcomplex elements corresponds to 2 doubles
            // we need to load and store 2*(n0-i) elements.

            __mmask8 mask = ( 1 << ( 2 * ( n0 - i ) ) ) - 1;

            __m512d xv, temp, zero;
            zero = _mm512_setzero_pd();

            xv = _mm512_mask_loadu_pd( zero, mask, x0 );

            temp = _mm512_permute_pd( xv, 0x55 );

            temp = _mm512_mul_pd( alphaIv, temp );

            xv = _mm512_fmaddsub_pd( alphaRv, xv, temp );

            _mm512_mask_storeu_pd( x0, mask, xv );
        }
    }
    else    // Non-unit increment.
    {
        __m128d alphaRv, alphaIv, x_vec, temp;

        alphaRv = _mm_loaddup_pd(&alphaR);
        alphaIv = _mm_loaddup_pd(&alphaI);

        for (; i < n0; ++i)
        {
            x_vec = _mm_loadu_pd(x0);

            temp = _mm_shuffle_pd(x_vec, x_vec, 0x1);

            temp = _mm_mul_pd(alphaIv, temp);
            x_vec = _mm_fmaddsub_pd(alphaRv, x_vec, temp);

            _mm_storeu_pd(x0, x_vec);

            x0 += 2 * incx;
        }
    }
}
