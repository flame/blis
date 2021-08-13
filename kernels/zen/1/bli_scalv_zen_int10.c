/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2017 - 2021, Advanced Micro Devices, Inc. All rights reserved.
   Copyright (C) 2018, The University of Texas at Austin.

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

void bli_sscalv_zen_int10
     (
       conj_t           conjalpha,
       dim_t            n,
       float*  restrict alpha,
       float*  restrict x, inc_t incx,
       cntx_t* restrict cntx
     )
{
	const dim_t      n_elem_per_reg = 8;

	dim_t            i;

	float*  restrict x0;

	__m256           alphav;
	__m256           xv[10];
	__m256           zv[10];

	// If the vector dimension is zero, or if alpha is unit, return early.
	if ( bli_zero_dim1( n ) || PASTEMAC(s,eq1)( *alpha ) ) return;

	// If alpha is zero, use setv.
	if ( PASTEMAC(s,eq0)( *alpha ) )
	{
		float* zero = bli_s0;
#ifdef BLIS_CONFIG_EPYC
		bli_ssetv_zen_int
		(
		  BLIS_NO_CONJUGATE,
		  n,
		  zero,
		  x, incx,
		  cntx
		);
#else
		ssetv_ker_ft f = bli_cntx_get_l1v_ker_dt( BLIS_FLOAT, BLIS_SETV_KER, cntx );
		f
		(
		  BLIS_NO_CONJUGATE,
		  n,
		  zero,
		  x, incx,
		  cntx
		);
#endif
		return;
	}

	// Initialize local pointers.
	x0 = x;

	if ( incx == 1 )
	{
		// Broadcast the alpha scalar to all elements of a vector register.
		alphav = _mm256_broadcast_ss( alpha );

		for ( i = 0; (i + 79) < n; i += 80 )
		{
			// Load the input values.
			xv[0] = _mm256_loadu_ps( x0 + 0*n_elem_per_reg );
			xv[1] = _mm256_loadu_ps( x0 + 1*n_elem_per_reg );
			xv[2] = _mm256_loadu_ps( x0 + 2*n_elem_per_reg );
			xv[3] = _mm256_loadu_ps( x0 + 3*n_elem_per_reg );
			xv[4] = _mm256_loadu_ps( x0 + 4*n_elem_per_reg );
			xv[5] = _mm256_loadu_ps( x0 + 5*n_elem_per_reg );
			xv[6] = _mm256_loadu_ps( x0 + 6*n_elem_per_reg );
			xv[7] = _mm256_loadu_ps( x0 + 7*n_elem_per_reg );
			xv[8] = _mm256_loadu_ps( x0 + 8*n_elem_per_reg );
			xv[9] = _mm256_loadu_ps( x0 + 9*n_elem_per_reg );

			// perform : x := alpha * x;
			zv[0] = _mm256_mul_ps( alphav, xv[0] );
			zv[1] = _mm256_mul_ps( alphav, xv[1] );
			zv[2] = _mm256_mul_ps( alphav, xv[2] );
			zv[3] = _mm256_mul_ps( alphav, xv[3] );
			zv[4] = _mm256_mul_ps( alphav, xv[4] );
			zv[5] = _mm256_mul_ps( alphav, xv[5] );
			zv[6] = _mm256_mul_ps( alphav, xv[6] );
			zv[7] = _mm256_mul_ps( alphav, xv[7] );
			zv[8] = _mm256_mul_ps( alphav, xv[8] );
			zv[9] = _mm256_mul_ps( alphav, xv[9] );

			// Store the output.
			_mm256_storeu_ps( (x0 + 0*n_elem_per_reg), zv[0] );
			_mm256_storeu_ps( (x0 + 1*n_elem_per_reg), zv[1] );
			_mm256_storeu_ps( (x0 + 2*n_elem_per_reg), zv[2] );
			_mm256_storeu_ps( (x0 + 3*n_elem_per_reg), zv[3] );
			_mm256_storeu_ps( (x0 + 4*n_elem_per_reg), zv[4] );
			_mm256_storeu_ps( (x0 + 5*n_elem_per_reg), zv[5] );
			_mm256_storeu_ps( (x0 + 6*n_elem_per_reg), zv[6] );
			_mm256_storeu_ps( (x0 + 7*n_elem_per_reg), zv[7] );
			_mm256_storeu_ps( (x0 + 8*n_elem_per_reg), zv[8] );
			_mm256_storeu_ps( (x0 + 9*n_elem_per_reg), zv[9] );

			x0 += 10*n_elem_per_reg;
		}

		for ( ; (i + 39) < n; i += 40 )
		{
			// Load the input values.
			xv[0] = _mm256_loadu_ps( x0 + 0*n_elem_per_reg );
			xv[1] = _mm256_loadu_ps( x0 + 1*n_elem_per_reg );
			xv[2] = _mm256_loadu_ps( x0 + 2*n_elem_per_reg );
			xv[3] = _mm256_loadu_ps( x0 + 3*n_elem_per_reg );
			xv[4] = _mm256_loadu_ps( x0 + 4*n_elem_per_reg );

			// perform : x := alpha * x;
			zv[0] = _mm256_mul_ps( alphav, xv[0] );
			zv[1] = _mm256_mul_ps( alphav, xv[1] );
			zv[2] = _mm256_mul_ps( alphav, xv[2] );
			zv[3] = _mm256_mul_ps( alphav, xv[3] );
			zv[4] = _mm256_mul_ps( alphav, xv[4] );

			// Store the output.
			_mm256_storeu_ps( (x0 + 0*n_elem_per_reg), zv[0] );
			_mm256_storeu_ps( (x0 + 1*n_elem_per_reg), zv[1] );
			_mm256_storeu_ps( (x0 + 2*n_elem_per_reg), zv[2] );
			_mm256_storeu_ps( (x0 + 3*n_elem_per_reg), zv[3] );
			_mm256_storeu_ps( (x0 + 4*n_elem_per_reg), zv[4] );

			x0 += 5*n_elem_per_reg;
		}

		for ( ; (i + 31) < n; i += 32 )
		{
			// Load the input values.
			xv[0] = _mm256_loadu_ps( x0 + 0*n_elem_per_reg );
			xv[1] = _mm256_loadu_ps( x0 + 1*n_elem_per_reg );
			xv[2] = _mm256_loadu_ps( x0 + 2*n_elem_per_reg );
			xv[3] = _mm256_loadu_ps( x0 + 3*n_elem_per_reg );

			// perform : x := alpha * x;
			zv[0] = _mm256_mul_ps( alphav, xv[0] );
			zv[1] = _mm256_mul_ps( alphav, xv[1] );
			zv[2] = _mm256_mul_ps( alphav, xv[2] );
			zv[3] = _mm256_mul_ps( alphav, xv[3] );

			// Store the output.
			_mm256_storeu_ps( (x0 + 0*n_elem_per_reg), zv[0] );
			_mm256_storeu_ps( (x0 + 1*n_elem_per_reg), zv[1] );
			_mm256_storeu_ps( (x0 + 2*n_elem_per_reg), zv[2] );
			_mm256_storeu_ps( (x0 + 3*n_elem_per_reg), zv[3] );

			x0 += 4*n_elem_per_reg;
		}

		for ( ; (i + 15) < n; i += 16 )
		{
			// Load the input values.
			xv[0] = _mm256_loadu_ps( x0 + 0*n_elem_per_reg );
			xv[1] = _mm256_loadu_ps( x0 + 1*n_elem_per_reg );

			// perform : x := alpha * x;
			zv[0] = _mm256_mul_ps( alphav, xv[0] );
			zv[1] = _mm256_mul_ps( alphav, xv[1] );

			// Store the output.
			_mm256_storeu_ps( (x0 + 0*n_elem_per_reg), zv[0] );
			_mm256_storeu_ps( (x0 + 1*n_elem_per_reg), zv[1] );

			x0 += 2*n_elem_per_reg;
		}

		for ( ; (i + 7) < n; i += 8 )
		{
			// Load the input values.
			xv[0] = _mm256_loadu_ps( x0 + 0*n_elem_per_reg );

			// perform : x := alpha * x;
			zv[0] = _mm256_mul_ps( alphav, xv[0] );

			// Store the output.
			_mm256_storeu_ps( (x0 + 0*n_elem_per_reg), zv[0] );

			x0 += 1*n_elem_per_reg;
		}

		for ( ; (i + 0) < n; i += 1 )
		{
			*x0 *= *alpha;

			x0 += 1;
		}
	}
	else
	{
		const float alphac = *alpha;

		for ( i = 0; i < n; ++i )
		{
			*x0 *= alphac;

			x0 += incx;
		}
	}
}

// -----------------------------------------------------------------------------

void bli_dscalv_zen_int10
     (
       conj_t           conjalpha,
       dim_t            n,
       double* restrict alpha,
       double* restrict x, inc_t incx,
       cntx_t* restrict cntx
     )
{
	const dim_t      n_elem_per_reg = 4;

	dim_t            i;

	double* restrict x0;

	__m256d          alphav;
	__m256d          xv[10];
	__m256d          zv[10];

	// If the vector dimension is zero, or if alpha is unit, return early.
	if ( bli_zero_dim1( n ) || PASTEMAC(d,eq1)( *alpha ) ) return;

	// If alpha is zero, use setv.
	if ( PASTEMAC(d,eq0)( *alpha ) )
	{
		double* zero = bli_d0;
#ifdef BLIS_CONFIG_EPYC
		bli_dsetv_zen_int
		(
		  BLIS_NO_CONJUGATE,
		  n,
		  zero,
		  x, incx,
		  cntx
		);
#else
		dsetv_ker_ft f = bli_cntx_get_l1v_ker_dt( BLIS_DOUBLE, BLIS_SETV_KER, cntx );

		f
		(
		  BLIS_NO_CONJUGATE,
		  n,
		  zero,
		  x, incx,
		  cntx
		);
#endif
		return;
	}

	// Initialize local pointers.
	x0 = x;

	if ( incx == 1 )
	{
		// Broadcast the alpha scalar to all elements of a vector register.
		alphav = _mm256_broadcast_sd( alpha );

		for ( i = 0; (i + 39) < n; i += 40 )
		{
			// Load the input values.
			xv[0] = _mm256_loadu_pd( x0 + 0*n_elem_per_reg );
			xv[1] = _mm256_loadu_pd( x0 + 1*n_elem_per_reg );
			xv[2] = _mm256_loadu_pd( x0 + 2*n_elem_per_reg );
			xv[3] = _mm256_loadu_pd( x0 + 3*n_elem_per_reg );
			xv[4] = _mm256_loadu_pd( x0 + 4*n_elem_per_reg );
			xv[5] = _mm256_loadu_pd( x0 + 5*n_elem_per_reg );
			xv[6] = _mm256_loadu_pd( x0 + 6*n_elem_per_reg );
			xv[7] = _mm256_loadu_pd( x0 + 7*n_elem_per_reg );
			xv[8] = _mm256_loadu_pd( x0 + 8*n_elem_per_reg );
			xv[9] = _mm256_loadu_pd( x0 + 9*n_elem_per_reg );

			// perform : x := alpha * x;
			zv[0] = _mm256_mul_pd( alphav, xv[0] );
			zv[1] = _mm256_mul_pd( alphav, xv[1] );
			zv[2] = _mm256_mul_pd( alphav, xv[2] );
			zv[3] = _mm256_mul_pd( alphav, xv[3] );
			zv[4] = _mm256_mul_pd( alphav, xv[4] );
			zv[5] = _mm256_mul_pd( alphav, xv[5] );
			zv[6] = _mm256_mul_pd( alphav, xv[6] );
			zv[7] = _mm256_mul_pd( alphav, xv[7] );
			zv[8] = _mm256_mul_pd( alphav, xv[8] );
			zv[9] = _mm256_mul_pd( alphav, xv[9] );

			// Store the output.
			_mm256_storeu_pd( (x0 + 0*n_elem_per_reg), zv[0] );
			_mm256_storeu_pd( (x0 + 1*n_elem_per_reg), zv[1] );
			_mm256_storeu_pd( (x0 + 2*n_elem_per_reg), zv[2] );
			_mm256_storeu_pd( (x0 + 3*n_elem_per_reg), zv[3] );
			_mm256_storeu_pd( (x0 + 4*n_elem_per_reg), zv[4] );
			_mm256_storeu_pd( (x0 + 5*n_elem_per_reg), zv[5] );
			_mm256_storeu_pd( (x0 + 6*n_elem_per_reg), zv[6] );
			_mm256_storeu_pd( (x0 + 7*n_elem_per_reg), zv[7] );
			_mm256_storeu_pd( (x0 + 8*n_elem_per_reg), zv[8] );
			_mm256_storeu_pd( (x0 + 9*n_elem_per_reg), zv[9] );

			x0 += 10*n_elem_per_reg;
		}

		for ( ; (i + 19) < n; i += 20 )
		{
			// Load the input values.
			xv[0] = _mm256_loadu_pd( x0 + 0*n_elem_per_reg );
			xv[1] = _mm256_loadu_pd( x0 + 1*n_elem_per_reg );
			xv[2] = _mm256_loadu_pd( x0 + 2*n_elem_per_reg );
			xv[3] = _mm256_loadu_pd( x0 + 3*n_elem_per_reg );
			xv[4] = _mm256_loadu_pd( x0 + 4*n_elem_per_reg );

			// perform : x := alpha * x;
			zv[0] = _mm256_mul_pd( alphav, xv[0] );
			zv[1] = _mm256_mul_pd( alphav, xv[1] );
			zv[2] = _mm256_mul_pd( alphav, xv[2] );
			zv[3] = _mm256_mul_pd( alphav, xv[3] );
			zv[4] = _mm256_mul_pd( alphav, xv[4] );

			// Store the output.
			_mm256_storeu_pd( (x0 + 0*n_elem_per_reg), zv[0] );
			_mm256_storeu_pd( (x0 + 1*n_elem_per_reg), zv[1] );
			_mm256_storeu_pd( (x0 + 2*n_elem_per_reg), zv[2] );
			_mm256_storeu_pd( (x0 + 3*n_elem_per_reg), zv[3] );
			_mm256_storeu_pd( (x0 + 4*n_elem_per_reg), zv[4] );

			x0 += 5*n_elem_per_reg;
		}

		for ( ; (i + 15) < n; i += 16 )
		{
			// Load the input values.
			xv[0] = _mm256_loadu_pd( x0 + 0*n_elem_per_reg );
			xv[1] = _mm256_loadu_pd( x0 + 1*n_elem_per_reg );
			xv[2] = _mm256_loadu_pd( x0 + 2*n_elem_per_reg );
			xv[3] = _mm256_loadu_pd( x0 + 3*n_elem_per_reg );

			// perform : x := alpha * x;
			zv[0] = _mm256_mul_pd( alphav, xv[0] );
			zv[1] = _mm256_mul_pd( alphav, xv[1] );
			zv[2] = _mm256_mul_pd( alphav, xv[2] );
			zv[3] = _mm256_mul_pd( alphav, xv[3] );

			// Store the output.
			_mm256_storeu_pd( (x0 + 0*n_elem_per_reg), zv[0] );
			_mm256_storeu_pd( (x0 + 1*n_elem_per_reg), zv[1] );
			_mm256_storeu_pd( (x0 + 2*n_elem_per_reg), zv[2] );
			_mm256_storeu_pd( (x0 + 3*n_elem_per_reg), zv[3] );

			x0 += 4*n_elem_per_reg;
		}

		for ( ; (i + 7) < n; i += 8 )
		{
			// Load the input values.
			xv[0] = _mm256_loadu_pd( x0 + 0*n_elem_per_reg );
			xv[1] = _mm256_loadu_pd( x0 + 1*n_elem_per_reg );

			// perform : x := alpha * x;
			zv[0] = _mm256_mul_pd( alphav, xv[0] );
			zv[1] = _mm256_mul_pd( alphav, xv[1] );

			// Store the output.
			_mm256_storeu_pd( (x0 + 0*n_elem_per_reg), zv[0] );
			_mm256_storeu_pd( (x0 + 1*n_elem_per_reg), zv[1] );

			x0 += 2*n_elem_per_reg;
		}

		for ( ; (i + 3) < n; i += 4 )
		{
			// Load the input values.
			xv[0] = _mm256_loadu_pd( x0 + 0*n_elem_per_reg );

			// perform : x := alpha * x;
			zv[0] = _mm256_mul_pd( alphav, xv[0] );

			// Store the output.
			_mm256_storeu_pd( (x0 + 0*n_elem_per_reg), zv[0] );

			x0 += 1*n_elem_per_reg;
		}

		for ( ; (i + 0) < n; i += 1 )
		{
			*x0 *= *alpha;

			x0 += 1;
		}
	}
	else
	{
		const double alphac = *alpha;

		for ( i = 0; i < n; ++i )
		{
			*x0 *= alphac;

			x0 += incx;
		}
	}
}

// -----------------------------------------------------------------------------

void bli_cscalv_zen_int10
     (
       conj_t              conjalpha,
       dim_t               n,
       scomplex*  restrict alpha,
       scomplex*  restrict x,
       inc_t               incx,
       cntx_t*    restrict cntx
     )
{
    AOCL_DTL_TRACE_ENTRY(AOCL_DTL_LEVEL_TRACE_4)

    const dim_t      n_elem_per_reg = 8;

    dim_t            i;

    float*  restrict x0;
    float*  restrict alpha0;
    float alphaR, alphaI;

    __m256           alphaRv;
    __m256           alphaIv;
    __m256           xv[10];
    __m256           x_sufv[10];

    conj_t conjx_use = conjalpha;

    // If the vector dimension is zero, or if alpha is unit, return early.
    if ( bli_zero_dim1( n ) || PASTEMAC(c,eq1)( *alpha ) ) return;

    // If alpha is zero, use setv.
    if ( PASTEMAC(c,eq0)( *alpha ) )
    {
        scomplex* zero = bli_c0;
        if (cntx == NULL)
            cntx = bli_gks_query_cntx();
        csetv_ker_ft f = bli_cntx_get_l1v_ker_dt( BLIS_SCOMPLEX, BLIS_SETV_KER, cntx );
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

    // Initialize local pointers.
    x0 = (float*)x;
    alpha0 = (float*)alpha;

    alphaR = alpha->real;
    alphaI = alpha->imag;

    if ( incx == 1 )
    {
        // Broadcast the alpha scalar to all elements of a vector register.
        if ( !bli_is_conj (conjx_use) ) // If BLIS_NO_CONJUGATE
        {
            alphaRv = _mm256_broadcast_ss( &alphaR );
            alphaIv = _mm256_set_ps(alphaI, -alphaI, alphaI, -alphaI, alphaI, -alphaI, alphaI, -alphaI);
        }
        else
        {
            alphaIv = _mm256_broadcast_ss( &alphaI );
            alphaRv = _mm256_set_ps(-alphaR, alphaR, -alphaR, alphaR, -alphaR, alphaR, -alphaR, alphaR);
        }

        /*
        = (alpha_r + alpha_i) * (x_r + x_i)
        = alpha_r*x_r + alpha_r*x_i + alpha_i*x_r + (-alpha_i*x_i)
        = (alpha_r*x_r - alpha_i*x_i) + (alpha_r*x_i + alpha_i*x_r)I

        x      = x_r , x_i , x_r , x_i , x_r , x_i , x_r , x_i
        x_suf  = x_i , x_r , x_i , x_r , x_i , x_r , x_i , x_r
        alphaR = ar  , ar  , ar  , ar  , ar  , ar  , ar  , ar
        alphaI = -ai , ai  ,-ai  , ai  ,-ai  , ai  ,-ai,  ai

        step 1) Load x.
        step 2) Shuffle x.
        step 3) mul x <= x*alphaR    =>  ar*x_r , ar*x_i
        step 4) fma x <= x_suf*alphaI + x  => (-ai*x_i , ai*x_r) + (ar*x_r , ar*x_i)
                                           =>  (ar*x_r - ai*x_i), (ar*x_i + ai*x_r )
        */

        for ( i = 0; (i + 39) < n; i += 40 )
        {
            // Load the input values.
            xv[0] = _mm256_loadu_ps( x0 + 0*n_elem_per_reg );
            xv[1] = _mm256_loadu_ps( x0 + 1*n_elem_per_reg );
            xv[2] = _mm256_loadu_ps( x0 + 2*n_elem_per_reg );
            xv[3] = _mm256_loadu_ps( x0 + 3*n_elem_per_reg );
            xv[4] = _mm256_loadu_ps( x0 + 4*n_elem_per_reg );
            xv[5] = _mm256_loadu_ps( x0 + 5*n_elem_per_reg );
            xv[6] = _mm256_loadu_ps( x0 + 6*n_elem_per_reg );
            xv[7] = _mm256_loadu_ps( x0 + 7*n_elem_per_reg );
            xv[8] = _mm256_loadu_ps( x0 + 8*n_elem_per_reg );
            xv[9] = _mm256_loadu_ps( x0 + 9*n_elem_per_reg );

            // x     =  xr0 , xi0, xr1, xi1 ....
            // x_suf =  xi0 , xr0, xi1, xr1 ....
             x_sufv[0] = _mm256_permute_ps( xv[0], 0xB1);
            x_sufv[1] = _mm256_permute_ps( xv[1], 0xB1);
            x_sufv[2] = _mm256_permute_ps( xv[2], 0xB1);
            x_sufv[3] = _mm256_permute_ps( xv[3], 0xB1);
            x_sufv[4] = _mm256_permute_ps( xv[4], 0xB1);
            x_sufv[5] = _mm256_permute_ps( xv[5], 0xB1);
            x_sufv[6] = _mm256_permute_ps( xv[6], 0xB1);
            x_sufv[7] = _mm256_permute_ps( xv[7], 0xB1);
            x_sufv[8] = _mm256_permute_ps( xv[8], 0xB1);
            x_sufv[9] = _mm256_permute_ps( xv[9], 0xB1);

            // mul x <= x*alphaR
            // aphhaR =   ar  ,   ar  ,   ar  ,  ar , ....
            // x      =   xr  ,   xi  ,   xr  ,  xi , ....
            // mul    =  ar*xr, ar*xi , ar*xr , ar*xi, ....
            xv[0] = _mm256_mul_ps( alphaRv, xv[0] );
            xv[1] = _mm256_mul_ps( alphaRv, xv[1] );
            xv[2] = _mm256_mul_ps( alphaRv, xv[2] );
            xv[3] = _mm256_mul_ps( alphaRv, xv[3] );
            xv[4] = _mm256_mul_ps( alphaRv, xv[4] );
            xv[5] = _mm256_mul_ps( alphaRv, xv[5] );
            xv[6] = _mm256_mul_ps( alphaRv, xv[6] );
            xv[7] = _mm256_mul_ps( alphaRv, xv[7] );
            xv[8] = _mm256_mul_ps( alphaRv, xv[8] );
            xv[9] = _mm256_mul_ps( alphaRv, xv[9] );

            // fma x <= x_suf*alphaI + x
            // alphaI = -ai   ,  ai   , -ai   ,  ai ....
            // X suf  =  xi   ,  xr   ,  xi   ,  xr ....
            // mul    = -ai*xi, ai*xr , -ai*xi,  ai*xi ....
            // add x  =  ar*xr - ai*xi, ar*xi + ai*xr, ....
            xv[0] = _mm256_fmadd_ps( alphaIv, x_sufv[0], xv[0] );
            xv[1] = _mm256_fmadd_ps( alphaIv, x_sufv[1], xv[1] );
            xv[2] = _mm256_fmadd_ps( alphaIv, x_sufv[2], xv[2] );
            xv[3] = _mm256_fmadd_ps( alphaIv, x_sufv[3], xv[3] );
            xv[4] = _mm256_fmadd_ps( alphaIv, x_sufv[4], xv[4] );
            xv[5] = _mm256_fmadd_ps( alphaIv, x_sufv[5], xv[5] );
            xv[6] = _mm256_fmadd_ps( alphaIv, x_sufv[6], xv[6] );
            xv[7] = _mm256_fmadd_ps( alphaIv, x_sufv[7], xv[7] );
            xv[8] = _mm256_fmadd_ps( alphaIv, x_sufv[8], xv[8] );
            xv[9] = _mm256_fmadd_ps( alphaIv, x_sufv[9], xv[9] );

            // Store the output.
            _mm256_storeu_ps( (x0 + 0*n_elem_per_reg), xv[0] );
            _mm256_storeu_ps( (x0 + 1*n_elem_per_reg), xv[1] );
            _mm256_storeu_ps( (x0 + 2*n_elem_per_reg), xv[2] );
            _mm256_storeu_ps( (x0 + 3*n_elem_per_reg), xv[3] );
            _mm256_storeu_ps( (x0 + 4*n_elem_per_reg), xv[4] );
            _mm256_storeu_ps( (x0 + 5*n_elem_per_reg), xv[5] );
            _mm256_storeu_ps( (x0 + 6*n_elem_per_reg), xv[6] );
            _mm256_storeu_ps( (x0 + 7*n_elem_per_reg), xv[7] );
            _mm256_storeu_ps( (x0 + 8*n_elem_per_reg), xv[8] );
            _mm256_storeu_ps( (x0 + 9*n_elem_per_reg), xv[9] );

            x0 += 10*n_elem_per_reg;
        }

        for ( ; (i + 19) < n; i += 20 )
        {
            // Load the input values.
            xv[0] = _mm256_loadu_ps( x0 + 0*n_elem_per_reg );
            xv[1] = _mm256_loadu_ps( x0 + 1*n_elem_per_reg );
            xv[2] = _mm256_loadu_ps( x0 + 2*n_elem_per_reg );
            xv[3] = _mm256_loadu_ps( x0 + 3*n_elem_per_reg );
            xv[4] = _mm256_loadu_ps( x0 + 4*n_elem_per_reg );

            // x     =  xr0 , xi0, xr1, xi1 ....
            // x_suf =  xi0 , xr0, xi1, xr1 ....
            x_sufv[0] = _mm256_permute_ps( xv[0], 0xB1);
            x_sufv[1] = _mm256_permute_ps( xv[1], 0xB1);
            x_sufv[2] = _mm256_permute_ps( xv[2], 0xB1);
            x_sufv[3] = _mm256_permute_ps( xv[3], 0xB1);
            x_sufv[4] = _mm256_permute_ps( xv[4], 0xB1);

            // mul x <= x*alphaR
            // aphhaR =   ar  ,   ar  ,   ar  ,  ar , ....
            // x      =   xr  ,   xi  ,   xr  ,  xi , ....
            // mul    =  ar*xr, ar*xi , ar*xr , ar*xi, ....
            xv[0] = _mm256_mul_ps( alphaRv, xv[0] );
            xv[1] = _mm256_mul_ps( alphaRv, xv[1] );
            xv[2] = _mm256_mul_ps( alphaRv, xv[2] );
            xv[3] = _mm256_mul_ps( alphaRv, xv[3] );
            xv[4] = _mm256_mul_ps( alphaRv, xv[4] );

            // fma x <= x_suf*alphaI + x
            // alphaI = -ai   ,  ai   , -ai   ,  ai ....
            // X      =  xi   ,  xr   ,  xi   ,  xr ....
            // mul    = -ai*xi, ai*xr , -ai*xi,  ai*xi ....
            // add x  =  ar*xr - ai*xi, ar*xi + ai*xr,
            xv[0] = _mm256_fmadd_ps( alphaIv, x_sufv[0], xv[0] );
            xv[1] = _mm256_fmadd_ps( alphaIv, x_sufv[1], xv[1] );
            xv[2] = _mm256_fmadd_ps( alphaIv, x_sufv[2], xv[2] );
            xv[3] = _mm256_fmadd_ps( alphaIv, x_sufv[3], xv[3] );
            xv[4] = _mm256_fmadd_ps( alphaIv, x_sufv[4], xv[4] );

            // Store the output.
            _mm256_storeu_ps( (x0 + 0*n_elem_per_reg), xv[0] );
            _mm256_storeu_ps( (x0 + 1*n_elem_per_reg), xv[1] );
            _mm256_storeu_ps( (x0 + 2*n_elem_per_reg), xv[2] );
            _mm256_storeu_ps( (x0 + 3*n_elem_per_reg), xv[3] );
            _mm256_storeu_ps( (x0 + 4*n_elem_per_reg), xv[4] );

            x0 += 5*n_elem_per_reg;
        }

        for ( ; (i + 15) < n; i += 16 )
        {
            // Load the input values.
            xv[0] = _mm256_loadu_ps( x0 + 0*n_elem_per_reg );
            xv[1] = _mm256_loadu_ps( x0 + 1*n_elem_per_reg );
            xv[2] = _mm256_loadu_ps( x0 + 2*n_elem_per_reg );
            xv[3] = _mm256_loadu_ps( x0 + 3*n_elem_per_reg );

            // x     =  xr0 , xi0, xr1, xi1 ....
            // x_suf =  xi0 , xr0, xi1, xr1 ....
             x_sufv[0] = _mm256_permute_ps( xv[0], 0xB1);
            x_sufv[1] = _mm256_permute_ps( xv[1], 0xB1);
            x_sufv[2] = _mm256_permute_ps( xv[2], 0xB1);
            x_sufv[3] = _mm256_permute_ps( xv[3], 0xB1);

            // mul x <= x*alphaR
            // aphhaR =   ar  ,   ar  ,   ar  ,  ar , ....
            // x      =   xr  ,   xi  ,   xr  ,  xi , ....
            // mul    =  ar*xr, ar*xi , ar*xr , ar*xi, ....
            xv[0] = _mm256_mul_ps( alphaRv, xv[0] );
            xv[1] = _mm256_mul_ps( alphaRv, xv[1] );
            xv[2] = _mm256_mul_ps( alphaRv, xv[2] );
            xv[3] = _mm256_mul_ps( alphaRv, xv[3] );

            // fma x <= x_suf*alphaI + x
            // alphaI = -ai   ,  ai   , -ai   ,  ai ....
            // X      =  xi   ,  xr   ,  xi   ,  xr ....
            // mul    = -ai*xi, ai*xr , -ai*xi,  ai*xi ....
            // add x  =  ar*xr - ai*xi, ar*xi + ai*xr,
            xv[0] = _mm256_fmadd_ps( alphaIv, x_sufv[0], xv[0] );
            xv[1] = _mm256_fmadd_ps( alphaIv, x_sufv[1], xv[1] );
            xv[2] = _mm256_fmadd_ps( alphaIv, x_sufv[2], xv[2] );
            xv[3] = _mm256_fmadd_ps( alphaIv, x_sufv[3], xv[3] );

            // Store the output.
            _mm256_storeu_ps( (x0 + 0*n_elem_per_reg), xv[0] );
            _mm256_storeu_ps( (x0 + 1*n_elem_per_reg), xv[1] );
            _mm256_storeu_ps( (x0 + 2*n_elem_per_reg), xv[2] );
            _mm256_storeu_ps( (x0 + 3*n_elem_per_reg), xv[3] );

            x0 += 4*n_elem_per_reg;
        }

        for ( ; (i + 7) < n; i += 8 )
        {
            // Load the input values.
            xv[0] = _mm256_loadu_ps( x0 + 0*n_elem_per_reg );
            xv[1] = _mm256_loadu_ps( x0 + 1*n_elem_per_reg );

            // x     =  xr0 , xi0, xr1, xi1 ....
            // x_suf =  xi0 , xr0, xi1, xr1 ....
             x_sufv[0] = _mm256_permute_ps( xv[0], 0xB1);
            x_sufv[1] = _mm256_permute_ps( xv[1], 0xB1);

            // mul x <= x*alphaR
            // aphhaR =   ar  ,   ar  ,   ar  ,  ar , ....
            // x      =   xr  ,   xi  ,   xr  ,  xi , ....
            // mul    =  ar*xr, ar*xi , ar*xr , ar*xi, ....
            xv[0] = _mm256_mul_ps( alphaRv, xv[0] );
            xv[1] = _mm256_mul_ps( alphaRv, xv[1] );

            // fma x <= x_suf*alphaI + x
            // alphaI = -ai   ,  ai   , -ai   ,  ai ....
            // X      =  xi   ,  xr   ,  xi   ,  xr ....
            // mul    = -ai*xi, ai*xr , -ai*xi,  ai*xi ....
            // add x  =  ar*xr - ai*xi, ar*xi + ai*xr,
            xv[0] = _mm256_fmadd_ps( alphaIv, x_sufv[0], xv[0] );
            xv[1] = _mm256_fmadd_ps( alphaIv, x_sufv[1], xv[1] );

            // Store the output.
            _mm256_storeu_ps( (x0 + 0*n_elem_per_reg), xv[0] );
            _mm256_storeu_ps( (x0 + 1*n_elem_per_reg), xv[1] );

            x0 += 2*n_elem_per_reg;
        }

        for ( ; (i + 3) < n; i += 4 )
        {
            // Load the input values.
            xv[0] = _mm256_loadu_ps( x0 + 0*n_elem_per_reg );

            // x     =  xr0 , xi0, xr1, xi1 ....
            // x_suf =  xi0 , xr0, xi1, xr1 ....
             x_sufv[0] = _mm256_permute_ps( xv[0], 0xB1);

            // mul x <= x*alphaR
            // aphhaR =   ar  ,   ar  ,   ar  ,  ar , ....
            // x      =   xr  ,   xi  ,   xr  ,  xi , ....
            // mul    =  ar*xr, ar*xi , ar*xr , ar*xi, ....
            xv[0] = _mm256_mul_ps( alphaRv, xv[0] );

            // fma x <= x_suf*alphaI + x
            // alphaI = -ai   ,  ai   , -ai   ,  ai ....
            // X      =  xi   ,  xr   ,  xi   ,  xr ....
            // mul    = -ai*xi, ai*xr , -ai*xi,  ai*xi ....
            // add x  =  ar*xr - ai*xi, ar*xi + ai*xr,
            xv[0] = _mm256_fmadd_ps( alphaIv, x_sufv[0], xv[0] );

            // Store the output.
            _mm256_storeu_ps( (x0 + 0*n_elem_per_reg), xv[0] );

            x0 += 1*n_elem_per_reg;
        }

        for ( ; (i + 0) < n; i += 1 )
        {
            float real;

            // real part: ( aR.xR - aIxI )
            real   = *alpha0 * (*x0) - (*(alpha0 + 1)) * (*(x0+1));
            // img part: ( aR.xI + aI.xR )
            *(x0 + 1) = *alpha0 * (*(x0+1)) +  (*(alpha0 + 1)) * (*x0);

            *x0 = real;

            x0 += 2;
        }
    }
    else
    {
        const float alphar = *alpha0;
        const float alphai = *(alpha0 + 1);

        if ( !bli_is_conj(conjx_use) ) //  BLIS_NO_CONJUGATE
        {
            for ( i = 0; i < n; ++i )
            {
                const float x0c = *x0;
                const float x1c = *( x0+1 );

                *x0       = alphar * x0c - alphai * x1c;
                *(x0 + 1) = alphar * x1c + alphai * x0c;

                x0 += incx*2;
            }
        }
        else //  BLIS_CONJUGATE
        {
            for ( i = 0; i < n; ++i )
            {
                const float x0c = *x0;
                const float x1c = *( x0+1 );

                *x0        = alphar * x0c + alphai * x1c;
                *(x0 + 1)  = alphai * x0c - alphar * x1c;

                x0 += incx*2;
            }
        }
    }
}

// -----------------------------------------------------------------------------

void bli_zscalv_zen_int10
     (
       conj_t              conjalpha,
       dim_t               n,
       dcomplex*  restrict alpha,
       dcomplex*  restrict x,
       inc_t               incx,
       cntx_t*    restrict cntx
     )
{
    AOCL_DTL_TRACE_ENTRY(AOCL_DTL_LEVEL_TRACE_4)

    const dim_t      n_elem_per_reg = 4;

    dim_t            i;

    double*  restrict x0;
    double*  restrict alpha0;
    double alphaR, alphaI;

    __m256d           alphaRv;
    __m256d           alphaIv;
    __m256d           xv[10];
    __m256d           x_sufv[10];

    conj_t conjx_use = conjalpha;

    // If the vector dimension is zero, or if alpha is unit, return early.
    if ( bli_zero_dim1( n ) || PASTEMAC(z,eq1)( *alpha ) ) return;

    // If alpha is zero, use setv.
    if ( PASTEMAC(z,eq0)( *alpha ) )
    {
        dcomplex* zero = bli_z0;

        if (cntx == NULL)
            cntx = bli_gks_query_cntx();
        zsetv_ker_ft f = bli_cntx_get_l1v_ker_dt( BLIS_DCOMPLEX, BLIS_SETV_KER, cntx );
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

    // Initialize local pointers.
    x0 = (double*)x;
    alpha0 = (double*)alpha;

    alphaR = alpha->real;
    alphaI = alpha->imag;

    if ( incx == 1 )
    {
        // Broadcast the alpha scalar to all elements of a vector register.
        if ( !bli_is_conj (conjx_use) ) // If BLIS_NO_CONJUGATE
        {
            alphaRv = _mm256_broadcast_sd( &alphaR );
            alphaIv = _mm256_set_pd(alphaI, -alphaI, alphaI, -alphaI);
        }
        else
        {
            alphaIv = _mm256_broadcast_sd( &alphaI );
            alphaRv = _mm256_set_pd(alphaR, -alphaR, alphaR, -alphaR);
        }

        /*
        = (alpha_r + alpha_i) * (x_r + x_i)
        = alpha_r*x_r + alpha_r*x_i + alpha_i*x_r + (-alpha_i*x_i)
        = (alpha_r*x_r - alpha_i*x_i) + (alpha_r*x_i + alpha_i*x_r)I

        x      = x_r , x_i , x_r , x_i , x_r , x_i , x_r , x_i
        x_suf  = x_i , x_r , x_i , x_r , x_i , x_r , x_i , x_r
        alphaR = ar  , ar  , ar  , ar  , ar  , ar  , ar  , ar
        alphaI = -ai , ai  ,-ai  , ai  ,-ai  , ai  ,-ai,  ai

        step 1) Load x.
        step 2) Shuffle x.
        step 3) mul x <= x*alphaR    =>  ar*x_r , ar*x_i
        step 4) fma x <= x_suf*alphaI + x  => (-ai*x_i , ai*x_r) + (ar*x_r , ar*x_i)
                                           =>  (ar*x_r - ai*x_i), (ar*x_i + ai*x_r )
        */

        for ( i = 0; (i + 19) < n; i += 20 )
        {
            // Load the input values.
            xv[0] = _mm256_loadu_pd( x0 + 0*n_elem_per_reg );
            xv[1] = _mm256_loadu_pd( x0 + 1*n_elem_per_reg );
            xv[2] = _mm256_loadu_pd( x0 + 2*n_elem_per_reg );
            xv[3] = _mm256_loadu_pd( x0 + 3*n_elem_per_reg );
            xv[4] = _mm256_loadu_pd( x0 + 4*n_elem_per_reg );
            xv[5] = _mm256_loadu_pd( x0 + 5*n_elem_per_reg );
            xv[6] = _mm256_loadu_pd( x0 + 6*n_elem_per_reg );
            xv[7] = _mm256_loadu_pd( x0 + 7*n_elem_per_reg );
            xv[8] = _mm256_loadu_pd( x0 + 8*n_elem_per_reg );
            xv[9] = _mm256_loadu_pd( x0 + 9*n_elem_per_reg );

            // x     =  xr0 , xi0, xr1, xi1 ....
            // x_suf =  xi0 , xr0, xi1, xr1 ....
            x_sufv[0] = _mm256_permute_pd( xv[0], 5);
            x_sufv[1] = _mm256_permute_pd( xv[1], 5);
            x_sufv[2] = _mm256_permute_pd( xv[2], 5);
            x_sufv[3] = _mm256_permute_pd( xv[3], 5);
            x_sufv[4] = _mm256_permute_pd( xv[4], 5);
            x_sufv[5] = _mm256_permute_pd( xv[5], 5);
            x_sufv[6] = _mm256_permute_pd( xv[6], 5);
            x_sufv[7] = _mm256_permute_pd( xv[7], 5);
            x_sufv[8] = _mm256_permute_pd( xv[8], 5);
            x_sufv[9] = _mm256_permute_pd( xv[9], 5);

            // mul x <= x*alphaR
            // aphhaR =   ar  ,   ar  ,   ar  ,  ar , ....
            // x      =   xr  ,   xi  ,   xr  ,  xi , ....
            // mul    =  ar*xr, ar*xi , ar*xr , ar*xi, ....
            xv[0] = _mm256_mul_pd( alphaRv, xv[0] );
            xv[1] = _mm256_mul_pd( alphaRv, xv[1] );
            xv[2] = _mm256_mul_pd( alphaRv, xv[2] );
            xv[3] = _mm256_mul_pd( alphaRv, xv[3] );
            xv[4] = _mm256_mul_pd( alphaRv, xv[4] );
            xv[5] = _mm256_mul_pd( alphaRv, xv[5] );
            xv[6] = _mm256_mul_pd( alphaRv, xv[6] );
            xv[7] = _mm256_mul_pd( alphaRv, xv[7] );
            xv[8] = _mm256_mul_pd( alphaRv, xv[8] );
            xv[9] = _mm256_mul_pd( alphaRv, xv[9] );

            // fma x <= x_suf*alphaI + x
            // alphaI = -ai   ,  ai   , -ai   ,  ai ....
            // X suf  =  xi   ,  xr   ,  xi   ,  xr ....
            // mul    = -ai*xi, ai*xr , -ai*xi,  ai*xi ....
            // add x  =  ar*xr - ai*xi, ar*xi + ai*xr, ....
            xv[0] = _mm256_fmadd_pd( alphaIv, x_sufv[0], xv[0] );
            xv[1] = _mm256_fmadd_pd( alphaIv, x_sufv[1], xv[1] );
            xv[2] = _mm256_fmadd_pd( alphaIv, x_sufv[2], xv[2] );
            xv[3] = _mm256_fmadd_pd( alphaIv, x_sufv[3], xv[3] );
            xv[4] = _mm256_fmadd_pd( alphaIv, x_sufv[4], xv[4] );
            xv[5] = _mm256_fmadd_pd( alphaIv, x_sufv[5], xv[5] );
            xv[6] = _mm256_fmadd_pd( alphaIv, x_sufv[6], xv[6] );
            xv[7] = _mm256_fmadd_pd( alphaIv, x_sufv[7], xv[7] );
            xv[8] = _mm256_fmadd_pd( alphaIv, x_sufv[8], xv[8] );
            xv[9] = _mm256_fmadd_pd( alphaIv, x_sufv[9], xv[9] );

            // Store the output.
            _mm256_storeu_pd( (x0 + 0*n_elem_per_reg), xv[0] );
            _mm256_storeu_pd( (x0 + 1*n_elem_per_reg), xv[1] );
            _mm256_storeu_pd( (x0 + 2*n_elem_per_reg), xv[2] );
            _mm256_storeu_pd( (x0 + 3*n_elem_per_reg), xv[3] );
            _mm256_storeu_pd( (x0 + 4*n_elem_per_reg), xv[4] );
            _mm256_storeu_pd( (x0 + 5*n_elem_per_reg), xv[5] );
            _mm256_storeu_pd( (x0 + 6*n_elem_per_reg), xv[6] );
            _mm256_storeu_pd( (x0 + 7*n_elem_per_reg), xv[7] );
            _mm256_storeu_pd( (x0 + 8*n_elem_per_reg), xv[8] );
            _mm256_storeu_pd( (x0 + 9*n_elem_per_reg), xv[9] );

            x0 += 10*n_elem_per_reg;
        }

        for ( ; (i + 9) < n; i += 10 )
        {
            // Load the input values.
            xv[0] = _mm256_loadu_pd( x0 + 0*n_elem_per_reg );
            xv[1] = _mm256_loadu_pd( x0 + 1*n_elem_per_reg );
            xv[2] = _mm256_loadu_pd( x0 + 2*n_elem_per_reg );
            xv[3] = _mm256_loadu_pd( x0 + 3*n_elem_per_reg );
            xv[4] = _mm256_loadu_pd( x0 + 4*n_elem_per_reg );

            // x     =  xr0 , xi0, xr1, xi1
            // x_suf =  xi0 , xr0, xi1, xr1
            x_sufv[0] = _mm256_permute_pd( xv[0], 5);
            x_sufv[1] = _mm256_permute_pd( xv[1], 5);
            x_sufv[2] = _mm256_permute_pd( xv[2], 5);
            x_sufv[3] = _mm256_permute_pd( xv[3], 5);
            x_sufv[4] = _mm256_permute_pd( xv[4], 5);

            // mul x <= x*alphaR
            // aphhaR =   ar  ,   ar  ,   ar  ,  ar
            // x      =   xr  ,   xi  ,   xr  ,  xi
            // mul    =  ar*xr, ar*xi , ar*xr , ar*xi
            xv[0] = _mm256_mul_pd( alphaRv, xv[0] );
            xv[1] = _mm256_mul_pd( alphaRv, xv[1] );
            xv[2] = _mm256_mul_pd( alphaRv, xv[2] );
            xv[3] = _mm256_mul_pd( alphaRv, xv[3] );
            xv[4] = _mm256_mul_pd( alphaRv, xv[4] );

            // fma x <= x_suf*alphaI + x
            // alphaI = -ai   ,  ai   , -ai   , ai
            // X      =  xi   ,  xr   ,  xi   , xr
            // mul    = -ai*xi, ai*xr , -ai*xi, ai*xi
            // add x  =  ar*xr - ai*xi, ar*xi + ai*xr,
            xv[0] = _mm256_fmadd_pd( alphaIv, x_sufv[0], xv[0] );
            xv[1] = _mm256_fmadd_pd( alphaIv, x_sufv[1], xv[1] );
            xv[2] = _mm256_fmadd_pd( alphaIv, x_sufv[2], xv[2] );
            xv[3] = _mm256_fmadd_pd( alphaIv, x_sufv[3], xv[3] );
            xv[4] = _mm256_fmadd_pd( alphaIv, x_sufv[4], xv[4] );

            // Store the output.
            _mm256_storeu_pd( (x0 + 0*n_elem_per_reg), xv[0] );
            _mm256_storeu_pd( (x0 + 1*n_elem_per_reg), xv[1] );
            _mm256_storeu_pd( (x0 + 2*n_elem_per_reg), xv[2] );
            _mm256_storeu_pd( (x0 + 3*n_elem_per_reg), xv[3] );
            _mm256_storeu_pd( (x0 + 4*n_elem_per_reg), xv[4] );

            x0 += 5*n_elem_per_reg;
        }

        for ( ; (i + 7) < n; i += 8 )
        {
            // Load the input values.
            xv[0] = _mm256_loadu_pd( x0 + 0*n_elem_per_reg );
            xv[1] = _mm256_loadu_pd( x0 + 1*n_elem_per_reg );
            xv[2] = _mm256_loadu_pd( x0 + 2*n_elem_per_reg );
            xv[3] = _mm256_loadu_pd( x0 + 3*n_elem_per_reg );

            // x     =  xr0 , xi0, xr1, xi1 ....
            // x_suf =  xi0 , xr0, xi1, xr1 ....
            x_sufv[0] = _mm256_permute_pd( xv[0], 5);
            x_sufv[1] = _mm256_permute_pd( xv[1], 5);
            x_sufv[2] = _mm256_permute_pd( xv[2], 5);
            x_sufv[3] = _mm256_permute_pd( xv[3], 5);

            // mul x <= x*alphaR
            // aphhaR =   ar  ,   ar  ,   ar  ,  ar , ....
            // x      =   xr  ,   xi  ,   xr  ,  xi , ....
            // mul    =  ar*xr, ar*xi , ar*xr , ar*xi, ....
            xv[0] = _mm256_mul_pd( alphaRv, xv[0] );
            xv[1] = _mm256_mul_pd( alphaRv, xv[1] );
            xv[2] = _mm256_mul_pd( alphaRv, xv[2] );
            xv[3] = _mm256_mul_pd( alphaRv, xv[3] );

            // fma x <= x_suf*alphaI + x
            // alphaI = -ai   ,  ai   , -ai   ,  ai ....
            // X      =  xi   ,  xr   ,  xi   ,  xr ....
            // mul    = -ai*xi, ai*xr , -ai*xi,  ai*xi ....
            // add x  =  ar*xr - ai*xi, ar*xi + ai*xr,
            xv[0] = _mm256_fmadd_pd( alphaIv, x_sufv[0], xv[0] );
            xv[1] = _mm256_fmadd_pd( alphaIv, x_sufv[1], xv[1] );
            xv[2] = _mm256_fmadd_pd( alphaIv, x_sufv[2], xv[2] );
            xv[3] = _mm256_fmadd_pd( alphaIv, x_sufv[3], xv[3] );

            // Store the output.
            _mm256_storeu_pd( (x0 + 0*n_elem_per_reg), xv[0] );
            _mm256_storeu_pd( (x0 + 1*n_elem_per_reg), xv[1] );
            _mm256_storeu_pd( (x0 + 2*n_elem_per_reg), xv[2] );
            _mm256_storeu_pd( (x0 + 3*n_elem_per_reg), xv[3] );

            x0 += 4*n_elem_per_reg;
        }


        for ( ; (i + 3) < n; i += 4 )
        {
            // Load the input values.
            xv[0] = _mm256_loadu_pd( x0 + 0*n_elem_per_reg );
            xv[1] = _mm256_loadu_pd( x0 + 1*n_elem_per_reg );

            // x     =  xr0 , xi0, xr1, xi1 ....
            // x_suf =  xi0 , xr0, xi1, xr1 ....
            x_sufv[0] = _mm256_permute_pd( xv[0], 5);
            x_sufv[1] = _mm256_permute_pd( xv[1], 5);

            // mul x <= x*alphaR
            // aphhaR =   ar  ,   ar  ,   ar  ,  ar , ....
            // x      =   xr  ,   xi  ,   xr  ,  xi , ....
            // mul    =  ar*xr, ar*xi , ar*xr , ar*xi, ....
            xv[0] = _mm256_mul_pd( alphaRv, xv[0] );
            xv[1] = _mm256_mul_pd( alphaRv, xv[1] );

            // fma x <= x_suf*alphaI + x
            // alphaI = -ai   ,  ai   , -ai   ,  ai ....
            // X      =  xi   ,  xr   ,  xi   ,  xr ....
            // mul    = -ai*xi, ai*xr , -ai*xi,  ai*xi ....
            // add x  =  ar*xr - ai*xi, ar*xi + ai*xr,
            xv[0] = _mm256_fmadd_pd( alphaIv, x_sufv[0], xv[0] );
            xv[1] = _mm256_fmadd_pd( alphaIv, x_sufv[1], xv[1] );

            // Store the output.
            _mm256_storeu_pd( (x0 + 0*n_elem_per_reg), xv[0] );
            _mm256_storeu_pd( (x0 + 1*n_elem_per_reg), xv[1] );

            x0 += 2*n_elem_per_reg;
        }

        for ( ; (i + 1) < n; i += 2 )
        {
            // Load the input values.
            xv[0] = _mm256_loadu_pd( x0 + 0*n_elem_per_reg );

            // x     =  xr0 , xi0, xr1, xi1 ....
            // x_suf =  xi0 , xr0, xi1, xr1 ....
             x_sufv[0] = _mm256_permute_pd( xv[0], 5);

            // mul x <= x*alphaR
            // aphhaR =   ar  ,   ar  ,   ar  ,  ar , ....
            // x      =   xr  ,   xi  ,   xr  ,  xi , ....
            // mul    =  ar*xr, ar*xi , ar*xr , ar*xi, ....
            xv[0] = _mm256_mul_pd( alphaRv, xv[0] );

            // fma x <= x_suf*alphaI + x
            // alphaI = -ai   ,  ai   , -ai   ,  ai ....
            // X      =  xi   ,  xr   ,  xi   ,  xr ....
            // mul    = -ai*xi, ai*xr , -ai*xi,  ai*xi ....
            // add x  =  ar*xr - ai*xi, ar*xi + ai*xr,
            xv[0] = _mm256_fmadd_pd( alphaIv, x_sufv[0], xv[0] );

            // Store the output.
            _mm256_storeu_pd( (x0 + 0*n_elem_per_reg), xv[0] );

            x0 += 1*n_elem_per_reg;
        }

        for ( ; (i + 0) < n; i += 1 )
        {
            double real;

            // real part: ( aR.xR - aIxI )
            real   = *alpha0 * (*x0) - (*(alpha0 + 1)) * (*(x0+1));
            // img part: ( aR.xI + aI.xR )
            *(x0 + 1) = *alpha0 * (*(x0+1)) +  (*(alpha0 + 1)) * (*x0);

            *x0 = real;

            x0 += 2;
        }
    }
    else
    {
        const double alphar = *alpha0;
        const double alphai = *(alpha0 + 1);

        if ( !bli_is_conj(conjx_use) ) //  BLIS_NO_CONJUGATE
        {
            for ( i = 0; i < n; ++i )
            {
                const double x0c = *x0;
                const double x1c = *( x0 + 1 );

                *x0       = alphar * x0c - alphai * x1c;
                *(x0 + 1) = alphar * x1c + alphai * x0c;

                x0 += incx*2;
            }
        }
        else //  BLIS_CONJUGATE
        {
            for ( i = 0; i < n; ++i )
            {
                const double x0c = *x0;
                const double x1c = *( x0 + 1 );

                *x0        = alphar * x0c + alphai * x1c;
                *(x0 + 1)  = alphai * x0c - alphar * x1c;

                x0 += incx*2;
            }
        }
    }
}
