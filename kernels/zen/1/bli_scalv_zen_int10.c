/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2017 - 2021, Advanced Micro Devices, Inc. All rights reserved.
   Copyright (C) 2018, The University of Texas at Austin

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

	dim_t            i = 0;

	float*  restrict x0;

	__m256           alphav;
	__m256           xv[16];
	__m256           zv[16];

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
		dim_t option;

		// Unroll and the loop used is picked based on the input size.
		if( n < 300)
		{
			option = 2;
		}
		else if( n < 500)
		{
			option = 1;
		}
		else 
		{
			option = 0;
		}

		switch(option)
		{
			case 0:

				for ( ; (i + 127) < n; i += 128 )
				{
					//Load the input values
					xv[0] = _mm256_loadu_ps( x0 + 0*n_elem_per_reg );
					xv[1] = _mm256_loadu_ps( x0 + 1*n_elem_per_reg );
					xv[2] = _mm256_loadu_ps( x0 + 2*n_elem_per_reg );
					xv[3] = _mm256_loadu_ps( x0 + 3*n_elem_per_reg );

					// Perform : x := alpha * x;
					zv[0] = _mm256_mul_ps( alphav, xv[0] );
					zv[1] = _mm256_mul_ps( alphav, xv[1] );
					zv[2] = _mm256_mul_ps( alphav, xv[2] );
					zv[3] = _mm256_mul_ps( alphav, xv[3] );

					// Store the result
					_mm256_storeu_ps( (x0 + 0*n_elem_per_reg), zv[0] );
					_mm256_storeu_ps( (x0 + 1*n_elem_per_reg), zv[1] );
					_mm256_storeu_ps( (x0 + 2*n_elem_per_reg), zv[2] );
					_mm256_storeu_ps( (x0 + 3*n_elem_per_reg), zv[3] );
					
					xv[4] = _mm256_loadu_ps( x0 + 4*n_elem_per_reg );
					xv[5] = _mm256_loadu_ps( x0 + 5*n_elem_per_reg );
					xv[6] = _mm256_loadu_ps( x0 + 6*n_elem_per_reg );
					xv[7] = _mm256_loadu_ps( x0 + 7*n_elem_per_reg );

					zv[4] = _mm256_mul_ps( alphav, xv[4] );
					zv[5] = _mm256_mul_ps( alphav, xv[5] );
					zv[6] = _mm256_mul_ps( alphav, xv[6] );
					zv[7] = _mm256_mul_ps( alphav, xv[7] );

					_mm256_storeu_ps( (x0 + 4*n_elem_per_reg), zv[4] );
					_mm256_storeu_ps( (x0 + 5*n_elem_per_reg), zv[5] );
					_mm256_storeu_ps( (x0 + 6*n_elem_per_reg), zv[6] );
					_mm256_storeu_ps( (x0 + 7*n_elem_per_reg), zv[7] );
					
					xv[8] = _mm256_loadu_ps( x0 + 8*n_elem_per_reg );
					xv[9] = _mm256_loadu_ps( x0 + 9*n_elem_per_reg );
					xv[10] = _mm256_loadu_ps( x0 + 10*n_elem_per_reg );
					xv[11] = _mm256_loadu_ps( x0 + 11*n_elem_per_reg );

					zv[8] = _mm256_mul_ps( alphav, xv[8] );
					zv[9] = _mm256_mul_ps( alphav, xv[9] );
					zv[10] = _mm256_mul_ps( alphav, xv[10] );
					zv[11] = _mm256_mul_ps( alphav, xv[11] );
					
					_mm256_storeu_ps( (x0 + 8*n_elem_per_reg), zv[8] );
					_mm256_storeu_ps( (x0 + 9*n_elem_per_reg), zv[9] );
					_mm256_storeu_ps( (x0 + 10*n_elem_per_reg), zv[10] );
					_mm256_storeu_ps( (x0 + 11*n_elem_per_reg), zv[11] );

					xv[12] = _mm256_loadu_ps( x0 + 12*n_elem_per_reg );
					xv[13] = _mm256_loadu_ps( x0 + 13*n_elem_per_reg );
					xv[14] = _mm256_loadu_ps( x0 + 14*n_elem_per_reg );
					xv[15] = _mm256_loadu_ps( x0 + 15*n_elem_per_reg );

					zv[12] = _mm256_mul_ps( alphav, xv[12] );
					zv[13] = _mm256_mul_ps( alphav, xv[13] );
					zv[14] = _mm256_mul_ps( alphav, xv[14] );
					zv[15] = _mm256_mul_ps( alphav, xv[15] );

					_mm256_storeu_ps( (x0 + 12*n_elem_per_reg), zv[12] );
					_mm256_storeu_ps( (x0 + 13*n_elem_per_reg), zv[13] );
					_mm256_storeu_ps( (x0 + 14*n_elem_per_reg), zv[14] );
					_mm256_storeu_ps( (x0 + 15*n_elem_per_reg), zv[15] );
					
					x0 += 16*n_elem_per_reg;
				}
	
			case 1 :

				for ( ; (i + 95) < n; i += 96 )
				{
					xv[0] = _mm256_loadu_ps( x0 + 0*n_elem_per_reg );
					xv[1] = _mm256_loadu_ps( x0 + 1*n_elem_per_reg );
					xv[2] = _mm256_loadu_ps( x0 + 2*n_elem_per_reg );
					xv[3] = _mm256_loadu_ps( x0 + 3*n_elem_per_reg );

					zv[0] = _mm256_mul_ps( alphav, xv[0] );
					zv[1] = _mm256_mul_ps( alphav, xv[1] );
					zv[2] = _mm256_mul_ps( alphav, xv[2] );
					zv[3] = _mm256_mul_ps( alphav, xv[3] );

					_mm256_storeu_ps( (x0 + 0*n_elem_per_reg), zv[0] );
					_mm256_storeu_ps( (x0 + 1*n_elem_per_reg), zv[1] );
					_mm256_storeu_ps( (x0 + 2*n_elem_per_reg), zv[2] );
					_mm256_storeu_ps( (x0 + 3*n_elem_per_reg), zv[3] );
					
					xv[4] = _mm256_loadu_ps( x0 + 4*n_elem_per_reg );
					xv[5] = _mm256_loadu_ps( x0 + 5*n_elem_per_reg );
					xv[6] = _mm256_loadu_ps( x0 + 6*n_elem_per_reg );
					xv[7] = _mm256_loadu_ps( x0 + 7*n_elem_per_reg );
					
					zv[4] = _mm256_mul_ps( alphav, xv[4] );
					zv[5] = _mm256_mul_ps( alphav, xv[5] );
					zv[6] = _mm256_mul_ps( alphav, xv[6] );
					zv[7] = _mm256_mul_ps( alphav, xv[7] );
					
					_mm256_storeu_ps( (x0 + 4*n_elem_per_reg), zv[4] );
					_mm256_storeu_ps( (x0 + 5*n_elem_per_reg), zv[5] );
					_mm256_storeu_ps( (x0 + 6*n_elem_per_reg), zv[6] );
					_mm256_storeu_ps( (x0 + 7*n_elem_per_reg), zv[7] );
					
					xv[8] = _mm256_loadu_ps( x0 + 8*n_elem_per_reg );
					xv[9] = _mm256_loadu_ps( x0 + 9*n_elem_per_reg );
					xv[10] = _mm256_loadu_ps( x0 + 10*n_elem_per_reg );
					xv[11] = _mm256_loadu_ps( x0 + 11*n_elem_per_reg );
					
					zv[8] = _mm256_mul_ps( alphav, xv[8] );
					zv[9] = _mm256_mul_ps( alphav, xv[9] );
					zv[10] = _mm256_mul_ps( alphav, xv[10] );
					zv[11] = _mm256_mul_ps( alphav, xv[11] );
					
					_mm256_storeu_ps( (x0 + 8*n_elem_per_reg), zv[8] );
					_mm256_storeu_ps( (x0 + 9*n_elem_per_reg), zv[9] );
					_mm256_storeu_ps( (x0 + 10*n_elem_per_reg), zv[10] );
					_mm256_storeu_ps( (x0 + 11*n_elem_per_reg), zv[11] );
					
					x0 += 12*n_elem_per_reg;
				}

			case 2:
		
				for ( ; (i + 47) < n; i += 48 )
				{
					xv[0] = _mm256_loadu_ps( x0 + 0*n_elem_per_reg );
					xv[1] = _mm256_loadu_ps( x0 + 1*n_elem_per_reg );
					xv[2] = _mm256_loadu_ps( x0 + 2*n_elem_per_reg );

					zv[0] = _mm256_mul_ps( alphav, xv[0] );
					zv[1] = _mm256_mul_ps( alphav, xv[1] );
					zv[2] = _mm256_mul_ps( alphav, xv[2] );
					
					_mm256_storeu_ps( (x0 + 0*n_elem_per_reg), zv[0] );
					_mm256_storeu_ps( (x0 + 1*n_elem_per_reg), zv[1] );
					_mm256_storeu_ps( (x0 + 2*n_elem_per_reg), zv[2] );
					
					xv[3] = _mm256_loadu_ps( x0 + 3*n_elem_per_reg );
					xv[4] = _mm256_loadu_ps( x0 + 4*n_elem_per_reg );
					xv[5] = _mm256_loadu_ps( x0 + 5*n_elem_per_reg );

					zv[3] = _mm256_mul_ps( alphav, xv[3] );
					zv[4] = _mm256_mul_ps( alphav, xv[4] );
					zv[5] = _mm256_mul_ps( alphav, xv[5] );

					_mm256_storeu_ps( (x0 + 3*n_elem_per_reg), zv[3] );
					_mm256_storeu_ps( (x0 + 4*n_elem_per_reg), zv[4] );
					_mm256_storeu_ps( (x0 + 5*n_elem_per_reg), zv[5] );

					x0 += 6*n_elem_per_reg;
				}

				for ( ; (i + 23) < n; i += 24 )
				{
					xv[0] = _mm256_loadu_ps( x0 + 0*n_elem_per_reg );
					xv[1] = _mm256_loadu_ps( x0 + 1*n_elem_per_reg );
					xv[2] = _mm256_loadu_ps( x0 + 2*n_elem_per_reg );

					zv[0] = _mm256_mul_ps( alphav, xv[0] );
					zv[1] = _mm256_mul_ps( alphav, xv[1] );
					zv[2] = _mm256_mul_ps( alphav, xv[2] );

					_mm256_storeu_ps( (x0 + 0*n_elem_per_reg), zv[0] );
					_mm256_storeu_ps( (x0 + 1*n_elem_per_reg), zv[1] );
					_mm256_storeu_ps( (x0 + 2*n_elem_per_reg), zv[2] );

					x0 += 3*n_elem_per_reg;
				}

				for ( ; (i + 7) < n; i += 8 )
				{
					xv[0] = _mm256_loadu_ps( x0 + 0*n_elem_per_reg );

					zv[0] = _mm256_mul_ps( alphav, xv[0] );

					_mm256_storeu_ps( (x0 + 0*n_elem_per_reg), zv[0] );

					x0 += 1*n_elem_per_reg;
				}

				for ( ; (i + 0) < n; i += 1 )
				{
					*x0 *= *alpha;

					x0 += 1;
				}
		}
	}
	else
	{
		const float alphac = *alpha;

		for ( ; i < n; ++i )
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

	dim_t            i = 0;

	double* restrict x0;

	__m256d          alphav;
	__m256d          xv[16];
	__m256d          zv[16];

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
		dim_t option;		

		// Unroll and the loop used is picked based on the input size.
		if(n < 200)
		{
			option = 2;
		}
		else if(n < 500)
		{
			option = 1;
		}
		else
		{
			option = 0;
		}

		switch(option)
		{
			case 0:

				for (; (i + 63) < n; i += 64 )
				{
					xv[0] = _mm256_loadu_pd( x0 + 0*n_elem_per_reg );
					xv[1] = _mm256_loadu_pd( x0 + 1*n_elem_per_reg );
					xv[2] = _mm256_loadu_pd( x0 + 2*n_elem_per_reg );
					xv[3] = _mm256_loadu_pd( x0 + 3*n_elem_per_reg );

					zv[0] = _mm256_mul_pd( alphav, xv[0] );
					zv[1] = _mm256_mul_pd( alphav, xv[1] );
					zv[2] = _mm256_mul_pd( alphav, xv[2] );
					zv[3] = _mm256_mul_pd( alphav, xv[3] );

					_mm256_storeu_pd( (x0 + 0*n_elem_per_reg), zv[0] );
					_mm256_storeu_pd( (x0 + 1*n_elem_per_reg), zv[1] );
					_mm256_storeu_pd( (x0 + 2*n_elem_per_reg), zv[2] );
					_mm256_storeu_pd( (x0 + 3*n_elem_per_reg), zv[3] );

					xv[4] = _mm256_loadu_pd( x0 + 4*n_elem_per_reg );
					xv[5] = _mm256_loadu_pd( x0 + 5*n_elem_per_reg );
					xv[6] = _mm256_loadu_pd( x0 + 6*n_elem_per_reg );
					xv[7] = _mm256_loadu_pd( x0 + 7*n_elem_per_reg );
					
					zv[4] = _mm256_mul_pd( alphav, xv[4] );
					zv[5] = _mm256_mul_pd( alphav, xv[5] );
					zv[6] = _mm256_mul_pd( alphav, xv[6] );
					zv[7] = _mm256_mul_pd( alphav, xv[7] );

					_mm256_storeu_pd( (x0 + 4*n_elem_per_reg), zv[4] );
					_mm256_storeu_pd( (x0 + 5*n_elem_per_reg), zv[5] );
					_mm256_storeu_pd( (x0 + 6*n_elem_per_reg), zv[6] );
					_mm256_storeu_pd( (x0 + 7*n_elem_per_reg), zv[7] );

					xv[8] = _mm256_loadu_pd( x0 + 8*n_elem_per_reg );
					xv[9] = _mm256_loadu_pd( x0 + 9*n_elem_per_reg );
					xv[10] = _mm256_loadu_pd( x0 + 10*n_elem_per_reg );
					xv[11] = _mm256_loadu_pd( x0 + 11*n_elem_per_reg );

					zv[8] = _mm256_mul_pd( alphav, xv[8] );
					zv[9] = _mm256_mul_pd( alphav, xv[9] );
					zv[10] = _mm256_mul_pd( alphav, xv[10] );
					zv[11] = _mm256_mul_pd( alphav, xv[11] );

					_mm256_storeu_pd( (x0 + 8*n_elem_per_reg), zv[8] );
					_mm256_storeu_pd( (x0 + 9*n_elem_per_reg), zv[9] );
					_mm256_storeu_pd( (x0 + 10*n_elem_per_reg), zv[10] );
					_mm256_storeu_pd( (x0 + 11*n_elem_per_reg), zv[11] );
					
					xv[12] = _mm256_loadu_pd( x0 + 12*n_elem_per_reg );
					xv[13] = _mm256_loadu_pd( x0 + 13*n_elem_per_reg );
					xv[14] = _mm256_loadu_pd( x0 + 14*n_elem_per_reg );
					xv[15] = _mm256_loadu_pd( x0 + 15*n_elem_per_reg );
					
					zv[12] = _mm256_mul_pd( alphav, xv[12] );
					zv[13] = _mm256_mul_pd( alphav, xv[13] );
					zv[14] = _mm256_mul_pd( alphav, xv[14] );
					zv[15] = _mm256_mul_pd( alphav, xv[15] );

					_mm256_storeu_pd( (x0 + 12*n_elem_per_reg), zv[12] );
					_mm256_storeu_pd( (x0 + 13*n_elem_per_reg), zv[13] );
					_mm256_storeu_pd( (x0 + 14*n_elem_per_reg), zv[14] );
					_mm256_storeu_pd( (x0 + 15*n_elem_per_reg), zv[15] );

					x0 += 16*n_elem_per_reg;
				}

				for (; (i + 47) < n; i += 48 )
				{
					xv[0] = _mm256_loadu_pd( x0 + 0*n_elem_per_reg );
					xv[1] = _mm256_loadu_pd( x0 + 1*n_elem_per_reg );
					xv[2] = _mm256_loadu_pd( x0 + 2*n_elem_per_reg );
					xv[3] = _mm256_loadu_pd( x0 + 3*n_elem_per_reg );

					zv[0] = _mm256_mul_pd( alphav, xv[0] );
					zv[1] = _mm256_mul_pd( alphav, xv[1] );
					zv[2] = _mm256_mul_pd( alphav, xv[2] );
					zv[3] = _mm256_mul_pd( alphav, xv[3] );
					
					_mm256_storeu_pd( (x0 + 0*n_elem_per_reg), zv[0] );
					_mm256_storeu_pd( (x0 + 1*n_elem_per_reg), zv[1] );
					_mm256_storeu_pd( (x0 + 2*n_elem_per_reg), zv[2] );
					_mm256_storeu_pd( (x0 + 3*n_elem_per_reg), zv[3] );
					
					xv[4] = _mm256_loadu_pd( x0 + 4*n_elem_per_reg );
					xv[5] = _mm256_loadu_pd( x0 + 5*n_elem_per_reg );
					xv[6] = _mm256_loadu_pd( x0 + 6*n_elem_per_reg );
					xv[7] = _mm256_loadu_pd( x0 + 7*n_elem_per_reg );
					
					zv[4] = _mm256_mul_pd( alphav, xv[4] );
					zv[5] = _mm256_mul_pd( alphav, xv[5] );
					zv[6] = _mm256_mul_pd( alphav, xv[6] );
					zv[7] = _mm256_mul_pd( alphav, xv[7] );

					_mm256_storeu_pd( (x0 + 4*n_elem_per_reg), zv[4] );
					_mm256_storeu_pd( (x0 + 5*n_elem_per_reg), zv[5] );
					_mm256_storeu_pd( (x0 + 6*n_elem_per_reg), zv[6] );
					_mm256_storeu_pd( (x0 + 7*n_elem_per_reg), zv[7] );

					xv[8] = _mm256_loadu_pd( x0 + 8*n_elem_per_reg );
					xv[9] = _mm256_loadu_pd( x0 + 9*n_elem_per_reg );
					xv[10] = _mm256_loadu_pd( x0 + 10*n_elem_per_reg );
					xv[11] = _mm256_loadu_pd( x0 + 11*n_elem_per_reg );
					
					zv[8] = _mm256_mul_pd( alphav, xv[8] );
					zv[9] = _mm256_mul_pd( alphav, xv[9] );
					zv[10] = _mm256_mul_pd( alphav, xv[10] );
					zv[11] = _mm256_mul_pd( alphav, xv[11] );
					
					_mm256_storeu_pd( (x0 + 8*n_elem_per_reg), zv[8] );
					_mm256_storeu_pd( (x0 + 9*n_elem_per_reg), zv[9] );
					_mm256_storeu_pd( (x0 + 10*n_elem_per_reg), zv[10] );
					_mm256_storeu_pd( (x0 + 11*n_elem_per_reg), zv[11] );
					
					x0 += 12*n_elem_per_reg;
				}

			case 1:

				for (; (i + 31) < n; i += 32 )
				{
					xv[0] = _mm256_loadu_pd( x0 + 0*n_elem_per_reg );
					xv[1] = _mm256_loadu_pd( x0 + 1*n_elem_per_reg );
					xv[2] = _mm256_loadu_pd( x0 + 2*n_elem_per_reg );
					xv[3] = _mm256_loadu_pd( x0 + 3*n_elem_per_reg );
				
					zv[0] = _mm256_mul_pd( alphav, xv[0] );
					zv[1] = _mm256_mul_pd( alphav, xv[1] );
					zv[2] = _mm256_mul_pd( alphav, xv[2] );
					zv[3] = _mm256_mul_pd( alphav, xv[3] );
					
					_mm256_storeu_pd( (x0 + 0*n_elem_per_reg), zv[0] );
					_mm256_storeu_pd( (x0 + 1*n_elem_per_reg), zv[1] );
					_mm256_storeu_pd( (x0 + 2*n_elem_per_reg), zv[2] );
					_mm256_storeu_pd( (x0 + 3*n_elem_per_reg), zv[3] );
					
					xv[4] = _mm256_loadu_pd( x0 + 4*n_elem_per_reg );
					xv[5] = _mm256_loadu_pd( x0 + 5*n_elem_per_reg );
					xv[6] = _mm256_loadu_pd( x0 + 6*n_elem_per_reg );
					xv[7] = _mm256_loadu_pd( x0 + 7*n_elem_per_reg );
					
					zv[4] = _mm256_mul_pd( alphav, xv[4] );
					zv[5] = _mm256_mul_pd( alphav, xv[5] );
					zv[6] = _mm256_mul_pd( alphav, xv[6] );
					zv[7] = _mm256_mul_pd( alphav, xv[7] );
					
					_mm256_storeu_pd( (x0 + 4*n_elem_per_reg), zv[4] );
					_mm256_storeu_pd( (x0 + 5*n_elem_per_reg), zv[5] );
					_mm256_storeu_pd( (x0 + 6*n_elem_per_reg), zv[6] );
					_mm256_storeu_pd( (x0 + 7*n_elem_per_reg), zv[7] );
					
					x0 += 8*n_elem_per_reg;
				}
				
			case 2:

				for ( ; (i + 11) < n; i += 12 )
				{
					xv[0] = _mm256_loadu_pd( x0 + 0*n_elem_per_reg );
					xv[1] = _mm256_loadu_pd( x0 + 1*n_elem_per_reg );
					xv[2] = _mm256_loadu_pd( x0 + 2*n_elem_per_reg );
					
					zv[0] = _mm256_mul_pd( alphav, xv[0] );
					zv[1] = _mm256_mul_pd( alphav, xv[1] );
					zv[2] = _mm256_mul_pd( alphav, xv[2] );
					
					_mm256_storeu_pd( (x0 + 0*n_elem_per_reg), zv[0] );
					_mm256_storeu_pd( (x0 + 1*n_elem_per_reg), zv[1] );
					_mm256_storeu_pd( (x0 + 2*n_elem_per_reg), zv[2] );
					
					x0 += 3*n_elem_per_reg;
				}

				for ( ; (i + 3) < n; i += 4 )
				{
					xv[0] = _mm256_loadu_pd( x0 + 0*n_elem_per_reg );

					zv[0] = _mm256_mul_pd( alphav, xv[0] );

					_mm256_storeu_pd( (x0 + 0*n_elem_per_reg), zv[0] );

					x0 += 1*n_elem_per_reg;
				}

				for ( ; (i + 0) < n; i += 1 )
				{
					*x0 *= *alpha;

					x0 += 1;
				}
		}
	}
	else
	{
		const double alphac = *alpha;

		for ( ; i < n; ++i )
		{
			*x0 *= alphac;

			x0 += incx;
		}
	}
}

