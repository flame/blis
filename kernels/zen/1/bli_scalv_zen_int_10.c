/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2017 - 2026, Advanced Micro Devices, Inc. All rights reserved.
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

#define BLIS_ASM_SYNTAX_ATT
#include "bli_x86_asm_macros.h"

// -----------------------------------------------------------------------------

void bli_sscalv_zen_int_10
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

	// If alpha is zero, use setv if not called from BLAS scal itself (indicated by n being negative).
	if ( PASTEMAC(s,eq0)( *alpha ) && n > 0 )
	{
		float* zero = bli_s0;
		if ( cntx == NULL ) cntx = bli_gks_query_cntx();
		ssetv_ker_ft f = bli_cntx_get_l1v_ker_dt( BLIS_FLOAT, BLIS_SETV_KER, cntx );
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

	// Initialize local pointers.
	x0 = x;

	if ( incx == 1 )
	{
		// Broadcast the alpha scalar to all elements of a vector register.
		alphav = _mm256_broadcast_ss( alpha );
		dim_t option;

		// Unroll and the loop used is picked based on the input size.
		if( n0 < 300)
		{
			option = 2;
		}
		else if( n0 < 500)
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

				for ( ; (i + 127) < n0; i += 128 )
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

				for ( ; (i + 95) < n0; i += 96 )
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
		
				for ( ; (i + 47) < n0; i += 48 )
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

				for ( ; (i + 23) < n0; i += 24 )
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

				for ( ; (i + 7) < n0; i += 8 )
				{
					xv[0] = _mm256_loadu_ps( x0 + 0*n_elem_per_reg );

					zv[0] = _mm256_mul_ps( alphav, xv[0] );

					_mm256_storeu_ps( (x0 + 0*n_elem_per_reg), zv[0] );

					x0 += 1*n_elem_per_reg;
				}

				for ( ; (i + 0) < n0; i += 1 )
				{
					*x0 *= *alpha;

					x0 += 1;
				}
		}
	}
	else
	{
		const float alphac = *alpha;

		for ( ; i < n0; ++i )
		{
			*x0 *= alphac;

			x0 += incx;
		}
	}
}

// -----------------------------------------------------------------------------

BLIS_EXPORT_BLIS void bli_dscalv_zen_int_10
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

	// If alpha is zero, use setv if not called from BLAS scal itself (indicated by n being negative).
	if ( PASTEMAC(d,eq0)( *alpha ) && n > 0 )
	{
		double* zero = bli_d0;
		if ( cntx == NULL ) cntx = bli_gks_query_cntx();
		dsetv_ker_ft f = bli_cntx_get_l1v_ker_dt( BLIS_DOUBLE, BLIS_SETV_KER, cntx );

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

	// Initialize local pointers.
	x0 = x;

	if ( incx == 1 )
	{
		// Broadcast the alpha scalar to all elements of a vector register.
		alphav = _mm256_broadcast_sd( alpha );
		dim_t option;		

		// Unroll and the loop used is picked based on the input size.
		if(n0 < 200)
		{
			option = 2;
		}
		else if(n0 < 500)
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

				for (; (i + 63) < n0; i += 64 )
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

				for (; (i + 47) < n0; i += 48 )
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

				for (; (i + 31) < n0; i += 32 )
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

				for ( ; (i + 11) < n0; i += 12 )
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

				for ( ; (i + 3) < n0; i += 4 )
				{
					xv[0] = _mm256_loadu_pd( x0 + 0*n_elem_per_reg );

					zv[0] = _mm256_mul_pd( alphav, xv[0] );

					_mm256_storeu_pd( (x0 + 0*n_elem_per_reg), zv[0] );

					x0 += 1*n_elem_per_reg;
				}

				for ( ; (i + 0) < n0; i += 1 )
				{
					*x0 *= *alpha;

					x0 += 1;
				}
		}
	}
	else
	{
		const double alphac = *alpha;

		for ( ; i < n0; ++i )
		{
			*x0 *= alphac;

			x0 += incx;
		}
	}
}

void bli_zdscalv_zen_int_10
     (
       conj_t           conjalpha,
       dim_t            n,
       dcomplex* restrict alpha,
       dcomplex* restrict x, inc_t incx,
       cntx_t* restrict cntx
     )
{
	// If the vector dimension is zero, or if alpha is unit, return early.
	if ( bli_zero_dim1( n ) || PASTEMAC(z,eq1)( *alpha )) return;

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

	dim_t i = 0;
	const dim_t n_elem_per_reg = 4;    // number of elements per register

	double* restrict x0 = (double*) x;

	/*
		This kernel only performs the computation
		when alpha is double from the BLAS layer
		alpha is passed as double complex to adhere
		to function pointer definition in BLIS
	*/
	const double alphac = (*alpha).real;

	if ( incx == 1 )
	{
		__m256d alphav;
		__m256d xv[15];

		alphav = _mm256_broadcast_sd( &alphac );

		for ( ; ( i + 29 ) < n0; i += 30 )
		{
			xv[0] = _mm256_loadu_pd( x0 );
			xv[1] = _mm256_loadu_pd( x0 + n_elem_per_reg );
			xv[2] = _mm256_loadu_pd( x0 + 2 * n_elem_per_reg );
			xv[3] = _mm256_loadu_pd( x0 + 3 * n_elem_per_reg );
			xv[4] = _mm256_loadu_pd( x0 + 4 * n_elem_per_reg );
			xv[5] = _mm256_loadu_pd( x0 + 5 * n_elem_per_reg );
			xv[6] = _mm256_loadu_pd( x0 + 6 * n_elem_per_reg );
			xv[7] = _mm256_loadu_pd( x0 + 7 * n_elem_per_reg );
			xv[8] = _mm256_loadu_pd( x0 + 8 * n_elem_per_reg );
			xv[9] = _mm256_loadu_pd( x0 + 9 * n_elem_per_reg );
			xv[10] = _mm256_loadu_pd( x0 + 10 * n_elem_per_reg );
			xv[11] = _mm256_loadu_pd( x0 + 11 * n_elem_per_reg );
			xv[12] = _mm256_loadu_pd( x0 + 12 * n_elem_per_reg );
			xv[13] = _mm256_loadu_pd( x0 + 13 * n_elem_per_reg );
			xv[14] = _mm256_loadu_pd( x0 + 14 * n_elem_per_reg );			

			xv[0] = _mm256_mul_pd( alphav, xv[0] );
			xv[1] = _mm256_mul_pd( alphav, xv[1] );
			xv[2] = _mm256_mul_pd( alphav, xv[2] );
			xv[3] = _mm256_mul_pd( alphav, xv[3] );
			xv[4] = _mm256_mul_pd( alphav, xv[4] );
			xv[5] = _mm256_mul_pd( alphav, xv[5] );
			xv[6] = _mm256_mul_pd( alphav, xv[6] );
			xv[7] = _mm256_mul_pd( alphav, xv[7] );
			xv[8] = _mm256_mul_pd( alphav, xv[8] );
			xv[9] = _mm256_mul_pd( alphav, xv[9] );
			xv[10] = _mm256_mul_pd( alphav, xv[10] );
			xv[11] = _mm256_mul_pd( alphav, xv[11] );
			xv[12] = _mm256_mul_pd( alphav, xv[12] );
			xv[13] = _mm256_mul_pd( alphav, xv[13] );
			xv[14] = _mm256_mul_pd( alphav, xv[14] );
			
			_mm256_storeu_pd( x0, xv[0] );
			_mm256_storeu_pd( (x0 + n_elem_per_reg), xv[1] );
			_mm256_storeu_pd( (x0 + 2*n_elem_per_reg), xv[2] );
			_mm256_storeu_pd( (x0 + 3*n_elem_per_reg), xv[3] );
			_mm256_storeu_pd( (x0 + 4*n_elem_per_reg), xv[4] );
			_mm256_storeu_pd( (x0 + 5*n_elem_per_reg), xv[5] );
			_mm256_storeu_pd( (x0 + 6*n_elem_per_reg), xv[6] );
			_mm256_storeu_pd( (x0 + 7*n_elem_per_reg), xv[7] );
			_mm256_storeu_pd( (x0 + 8*n_elem_per_reg), xv[8] );
			_mm256_storeu_pd( (x0 + 9*n_elem_per_reg), xv[9] );
			_mm256_storeu_pd( (x0 + 10*n_elem_per_reg), xv[10] );
			_mm256_storeu_pd( (x0 + 11*n_elem_per_reg), xv[11] );
			_mm256_storeu_pd( (x0 + 12*n_elem_per_reg), xv[12] );
			_mm256_storeu_pd( (x0 + 13*n_elem_per_reg), xv[13] );
			_mm256_storeu_pd( (x0 + 14*n_elem_per_reg), xv[14] );

			x0 += 15 * n_elem_per_reg;
		}

		for ( ; ( i + 23 ) < n0; i += 24 )
		{
			xv[0] = _mm256_loadu_pd( x0 );
			xv[1] = _mm256_loadu_pd( x0 + n_elem_per_reg );
			xv[2] = _mm256_loadu_pd( x0 + 2 * n_elem_per_reg );
			xv[3] = _mm256_loadu_pd( x0 + 3 * n_elem_per_reg );
			xv[4] = _mm256_loadu_pd( x0 + 4 * n_elem_per_reg );
			xv[5] = _mm256_loadu_pd( x0 + 5 * n_elem_per_reg );
			xv[6] = _mm256_loadu_pd( x0 + 6 * n_elem_per_reg );
			xv[7] = _mm256_loadu_pd( x0 + 7 * n_elem_per_reg );
			xv[8] = _mm256_loadu_pd( x0 + 8 * n_elem_per_reg );
			xv[9] = _mm256_loadu_pd( x0 + 9 * n_elem_per_reg );
			xv[10] = _mm256_loadu_pd( x0 + 10 * n_elem_per_reg );
			xv[11] = _mm256_loadu_pd( x0 + 11 * n_elem_per_reg );

			xv[0] = _mm256_mul_pd( alphav, xv[0] );
			xv[1] = _mm256_mul_pd( alphav, xv[1] );
			xv[2] = _mm256_mul_pd( alphav, xv[2] );
			xv[3] = _mm256_mul_pd( alphav, xv[3] );
			xv[4] = _mm256_mul_pd( alphav, xv[4] );
			xv[5] = _mm256_mul_pd( alphav, xv[5] );
			xv[6] = _mm256_mul_pd( alphav, xv[6] );
			xv[7] = _mm256_mul_pd( alphav, xv[7] );
			xv[8] = _mm256_mul_pd( alphav, xv[8] );
			xv[9] = _mm256_mul_pd( alphav, xv[9] );
			xv[10] = _mm256_mul_pd( alphav, xv[10] );
			xv[11] = _mm256_mul_pd( alphav, xv[11] );
			
			_mm256_storeu_pd( x0, xv[0] );
			_mm256_storeu_pd( (x0 + n_elem_per_reg), xv[1] );
			_mm256_storeu_pd( (x0 + 2*n_elem_per_reg), xv[2] );
			_mm256_storeu_pd( (x0 + 3*n_elem_per_reg), xv[3] );
			_mm256_storeu_pd( (x0 + 4*n_elem_per_reg), xv[4] );
			_mm256_storeu_pd( (x0 + 5*n_elem_per_reg), xv[5] );
			_mm256_storeu_pd( (x0 + 6*n_elem_per_reg), xv[6] );
			_mm256_storeu_pd( (x0 + 7*n_elem_per_reg), xv[7] );
			_mm256_storeu_pd( (x0 + 8*n_elem_per_reg), xv[8] );
			_mm256_storeu_pd( (x0 + 9*n_elem_per_reg), xv[9] );
			_mm256_storeu_pd( (x0 + 10*n_elem_per_reg), xv[10] );
			_mm256_storeu_pd( (x0 + 11*n_elem_per_reg), xv[11] );

			x0 += 12 * n_elem_per_reg;
		}

		for ( ; ( i + 15 ) < n0; i += 16 )
		{
			xv[0] = _mm256_loadu_pd( x0 );
			xv[1] = _mm256_loadu_pd( x0 + n_elem_per_reg );
			xv[2] = _mm256_loadu_pd( x0 + 2 * n_elem_per_reg );
			xv[3] = _mm256_loadu_pd( x0 + 3 * n_elem_per_reg );
			xv[4] = _mm256_loadu_pd( x0 + 4 * n_elem_per_reg );
			xv[5] = _mm256_loadu_pd( x0 + 5 * n_elem_per_reg );
			xv[6] = _mm256_loadu_pd( x0 + 6 * n_elem_per_reg );
			xv[7] = _mm256_loadu_pd( x0 + 7 * n_elem_per_reg );

			xv[0] = _mm256_mul_pd( alphav, xv[0] );
			xv[1] = _mm256_mul_pd( alphav, xv[1] );
			xv[2] = _mm256_mul_pd( alphav, xv[2] );
			xv[3] = _mm256_mul_pd( alphav, xv[3] );
			xv[4] = _mm256_mul_pd( alphav, xv[4] );
			xv[5] = _mm256_mul_pd( alphav, xv[5] );
			xv[6] = _mm256_mul_pd( alphav, xv[6] );
			xv[7] = _mm256_mul_pd( alphav, xv[7] );
			
			_mm256_storeu_pd( x0, xv[0] );
			_mm256_storeu_pd( (x0 + n_elem_per_reg), xv[1] );
			_mm256_storeu_pd( (x0 + 2*n_elem_per_reg), xv[2] );
			_mm256_storeu_pd( (x0 + 3*n_elem_per_reg), xv[3] );
			_mm256_storeu_pd( (x0 + 4*n_elem_per_reg), xv[4] );
			_mm256_storeu_pd( (x0 + 5*n_elem_per_reg), xv[5] );
			_mm256_storeu_pd( (x0 + 6*n_elem_per_reg), xv[6] );
			_mm256_storeu_pd( (x0 + 7*n_elem_per_reg), xv[7] );

			x0 += 8 * n_elem_per_reg;
		}

		for ( ; ( i + 7 ) < n0; i += 8 )
		{
			xv[0] = _mm256_loadu_pd( x0 );
			xv[1] = _mm256_loadu_pd( x0 + n_elem_per_reg );
			xv[2] = _mm256_loadu_pd( x0 + 2 * n_elem_per_reg );
			xv[3] = _mm256_loadu_pd( x0 + 3 * n_elem_per_reg );

			xv[0] = _mm256_mul_pd( alphav, xv[0] );
			xv[1] = _mm256_mul_pd( alphav, xv[1] );
			xv[2] = _mm256_mul_pd( alphav, xv[2] );
			xv[3] = _mm256_mul_pd( alphav, xv[3] );
			
			_mm256_storeu_pd( x0, xv[0] );
			_mm256_storeu_pd( (x0 + n_elem_per_reg), xv[1] );
			_mm256_storeu_pd( (x0 + 2*n_elem_per_reg), xv[2] );
			_mm256_storeu_pd( (x0 + 3*n_elem_per_reg), xv[3] );

			x0 += 4 * n_elem_per_reg;
		}

		for ( ; ( i + 3 ) < n0; i += 4 )
		{
			xv[0] = _mm256_loadu_pd( x0 );
			xv[1] = _mm256_loadu_pd( x0 + n_elem_per_reg );

			xv[0] = _mm256_mul_pd( alphav, xv[0] );
			xv[1] = _mm256_mul_pd( alphav, xv[1] );
			
			_mm256_storeu_pd( x0, xv[0] );
			_mm256_storeu_pd( (x0 + n_elem_per_reg), xv[1] );

			x0 += 2 * n_elem_per_reg;
		}

		for ( ; ( i + 1 ) < n0; i += 2 )
		{
			xv[0] = _mm256_loadu_pd( x0 );

			xv[0] = _mm256_mul_pd( alphav, xv[0] );

			_mm256_storeu_pd( x0, xv[0] );

			x0 += n_elem_per_reg;
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

void bli_cscalv_zen_int
	(
	conj_t           conjalpha,
	dim_t            n,
	scomplex* restrict alpha,
	scomplex* restrict x, inc_t incx,
	cntx_t* restrict cntx
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
			NULL);

		return;
	}

	// Convert absolute value of n to 64-bit unsigned integer
	uint64_t n0 = (uint64_t)bli_abs(n);

	scomplex alpha_conj;
	float *x0 = (float *)x;

	// Performs conjugation of alpha based on conjalpha
	PASTEMAC(c, copycjs)(conjalpha, *alpha, alpha_conj);

	float real = alpha_conj.real;
	float imag = alpha_conj.imag;

	// Convert stride to 64-bit for inline assembly
	int64_t incx0 = (int64_t)incx;

	// Mask table for vmaskmovps: indexed by number of remaining complex elements (1-3)
	// Each complex element = 2 floats, so masks activate 2, 4, or 6 lanes respectively
	// High bit set (0x80000000) means that lane is active for load/store
	static const uint32_t mask_table[4][8] __attribute__((aligned(32))) = {
		{0, 0, 0, 0, 0, 0, 0, 0},                                      					// offset 0: unused
		{0x80000000, 0x80000000, 0, 0, 0, 0, 0, 0},                   					// offset 1: 2 lanes (1 complex)
		{0x80000000, 0x80000000, 0x80000000, 0x80000000, 0, 0, 0, 0}, 					// offset 2: 4 lanes (2 complex)
		{0x80000000, 0x80000000, 0x80000000, 0x80000000, 0x80000000, 0x80000000, 0, 0} 	// offset 3: 6 lanes (3 complex)
	};

	// Handling computation for both unit-strided and non-unit-strided vectors in inline asm.
	// Performs: x := alpha * x, where alpha = real + i*imag and x is a complex vector
	begin_asm()

	/*
		rax -> pointer to x vector
		rsi -> number of complex elements remaining
		rdx -> stride (incx)
		r11 -> byte offset for non-unit stride
		ymm8 -> broadcast real part of alpha
		ymm9 -> broadcast imaginary part of alpha
	*/

	// ----------------------------------------------------------------
	// Load base pointer and parameters
	// ----------------------------------------------------------------
	mov(var(x0), rax)                    // rax = base address of x vector
	mov(var(n0), rsi)                    // rsi = number of complex elements
	mov(var(incx0), rdx)                 // rdx = stride value (incx)

	// Broadcast alpha components to all lanes of YMM registers
	vbroadcastss(var(real), ymm8)        // ymm8 = [real, real, real, real, real, real, real, real]
	vbroadcastss(var(imag), ymm9)        // ymm9 = [imag, imag, imag, imag, imag, imag, imag, imag]

	// Check if unit stride; if not, jump to scalar loop
	cmp(imm(1), rdx)                     // compare incx with 1
	jne(.CSCALV_SCALAR_LOOP)             // if incx != 1, handle with scalar loop

	// ================================================================
	// Unit-stride vectorized loops
	// ================================================================

	// ----------------------------------------------------------------
	// Process 16 complex elements (32 floats, 128 bytes) per iteration
	// ----------------------------------------------------------------
	label(.CSCALV_LOOP16)
	cmp(imm(16), rsi)                    // check if remaining elements >= 16
	jl(.CSCALV_LOOP8)                    // if less, go to next smaller block

	// Load 16 complex elements (32 floats) into 4 YMM registers
	vmovups(mem(rax, 0*32), ymm0)        // ymm0 = x[i+0:i+3] (4 complex = 8 floats)
	vmovups(mem(rax, 1*32), ymm1)        // ymm1 = x[i+4:i+7]
	vmovups(mem(rax, 2*32), ymm2)        // ymm2 = x[i+8:i+11]
	vmovups(mem(rax, 3*32), ymm3)        // ymm3 = x[i+12:i+15]

	// Multiply x by imaginary part: imag * x
	vmulps(ymm9, ymm0, ymm10)            // ymm10 = imag * x[i+0:i+3]
	vmulps(ymm9, ymm1, ymm11)            // ymm11 = imag * x[i+4:i+7]
	vmulps(ymm9, ymm2, ymm12)            // ymm12 = imag * x[i+8:i+11]
	vmulps(ymm9, ymm3, ymm13)            // ymm13 = imag * x[i+12:i+15]

	// Swap real and imaginary parts (permute with pattern 0xB1 = 10110001)
	vpermilps(imm(0xB1), ymm10, ymm10)   // swap adjacent pairs in ymm10
	vpermilps(imm(0xB1), ymm11, ymm11)   // swap adjacent pairs in ymm11
	vpermilps(imm(0xB1), ymm12, ymm12)   // swap adjacent pairs in ymm12
	vpermilps(imm(0xB1), ymm13, ymm13)   // swap adjacent pairs in ymm13

	// Fused multiply-add-subtract: real*x +/- swapped(imag*x)
	vfmaddsub231ps(ymm0, ymm8, ymm10)    // ymm10 = real*x[i+0:i+3] +/- swapped(imag*x[i+0:i+3])
	vfmaddsub231ps(ymm1, ymm8, ymm11)    // ymm11 = real*x[i+4:i+7] +/- swapped(imag*x[i+4:i+7])
	vfmaddsub231ps(ymm2, ymm8, ymm12)    // ymm12 = real*x[i+8:i+11] +/- swapped(imag*x[i+8:i+11])
	vfmaddsub231ps(ymm3, ymm8, ymm13)    // ymm13 = real*x[i+12:i+15] +/- swapped(imag*x[i+12:i+15])

	// Store results back to memory
	vmovups(ymm10, mem(rax, 0*32))       // x[i+0:i+3] = ymm10
	vmovups(ymm11, mem(rax, 1*32))       // x[i+4:i+7] = ymm11
	vmovups(ymm12, mem(rax, 2*32))       // x[i+8:i+11] = ymm12
	vmovups(ymm13, mem(rax, 3*32))       // x[i+12:i+15] = ymm13

	add(imm(4*32), rax)                  // rax += 128 bytes (16 complex elements)
	sub(imm(16), rsi)                    // reduce remaining elements by 16
	jmp(.CSCALV_LOOP16)                  // repeat for next 16 elements

	// ----------------------------------------------------------------
	// Process 8 complex elements (16 floats, 64 bytes)
	// ----------------------------------------------------------------
	label(.CSCALV_LOOP8)
	cmp(imm(8), rsi)                     // check if remaining elements >= 8
	jl(.CSCALV_LOOP4)                    // if less, go to next smaller block

	// Load 8 complex elements into 2 YMM registers
	vmovups(mem(rax, 0*32), ymm0)        // ymm0 = x[i+0:i+3]
	vmovups(mem(rax, 1*32), ymm1)        // ymm1 = x[i+4:i+7]

	// Multiply by imaginary part
	vmulps(ymm9, ymm0, ymm6)             // ymm6 = imag * x[i+0:i+3]
	vmulps(ymm9, ymm1, ymm7)             // ymm7 = imag * x[i+4:i+7]

	// Swap real and imaginary parts
	vpermilps(imm(0xB1), ymm6, ymm6)     // swap adjacent pairs in ymm6
	vpermilps(imm(0xB1), ymm7, ymm7)     // swap adjacent pairs in ymm7

	// Fused multiply-add-subtract
	vfmaddsub231ps(ymm0, ymm8, ymm6)     // ymm6 = real*x[i+0:i+3] +/- swapped(imag*x[i+0:i+3])
	vfmaddsub231ps(ymm1, ymm8, ymm7)     // ymm7 = real*x[i+4:i+7] +/- swapped(imag*x[i+4:i+7])

	// Store results
	vmovups(ymm6, mem(rax, 0*32))        // x[i+0:i+3] = ymm6
	vmovups(ymm7, mem(rax, 1*32))        // x[i+4:i+7] = ymm7

	add(imm(2*32), rax)                  // rax += 64 bytes (8 complex elements)
	sub(imm(8), rsi)                     // reduce remaining elements by 8

	// ----------------------------------------------------------------
	// Process 4 complex elements (8 floats, 32 bytes)
	// ----------------------------------------------------------------
	label(.CSCALV_LOOP4)
	cmp(imm(4), rsi)                     // check if remaining elements >= 4
	jl(.CSCALV_FRINGE)                   // if less, handle fringe elements

	// Load 4 complex elements into 1 YMM register
	vmovups(mem(rax, 0*32), ymm0)        // ymm0 = x[i+0:i+3]

	// Multiply by imaginary part
	vmulps(ymm9, ymm0, ymm4)             // ymm4 = imag * x[i+0:i+3]

	// Swap real and imaginary parts
	vpermilps(imm(0xB1), ymm4, ymm4)     // swap adjacent pairs in ymm4

	// Fused multiply-add-subtract
	vfmaddsub231ps(ymm0, ymm8, ymm4)     // ymm4 = real*x[i+0:i+3] +/- swapped(imag*x[i+0:i+3])

	// Store result
	vmovups(ymm4, mem(rax, 0*32))        // x[i+0:i+3] = ymm4

	add(imm(1*32), rax)                  // rax += 32 bytes (4 complex elements)
	sub(imm(4), rsi)                     // reduce remaining elements by 4

	// ----------------------------------------------------------------
	// Handle fringe: 1-3 remaining complex elements (2-6 floats) using YMM with masking
	// ----------------------------------------------------------------
	label(.CSCALV_FRINGE)
	test(rsi, rsi)                       // check if any elements remaining
	jz(.CSCALV_DONE)                     // if zero, we're done

	// Build mask based on remaining elements (rsi = 1, 2, or 3)
	// Load mask from mask_table[rsi]
	mov(rsi, rcx)                        // rcx = copy of remaining elements count
	lea(var(mask_table), r10)            // r10 = address of mask_table
	sal(imm(5), rcx)                     // rcx *= 32 (each mask is 32 bytes = 8 ints * 4 bytes)
	add(rcx, r10)                        // r10 = mask_table + offset
	vmovdqu(mem(r10), ymm15)             // ymm15 = mask for vmaskmovps

	// Masked load of remaining elements
	vmaskmovps(mem(rax), ymm15, ymm0)    // ymm0 = masked load (1-3 complex elements)

	// Multiply by imaginary part
	vmulps(ymm9, ymm0, ymm4)             // ymm4 = imag * x

	// Swap real and imaginary parts
	vpermilps(imm(0xB1), ymm4, ymm4)     // swap adjacent pairs

	// Fused multiply-add-subtract
	vfmaddsub231ps(ymm0, ymm8, ymm4)     // ymm4 = real*x +/- swapped(imag*x)

	// Masked store of results
	vmaskmovps(ymm4, ymm15, mem(rax))    // masked store back to memory
	jmp(.CSCALV_DONE)                    // done with all elements

	// ================================================================
	// Non-unit stride scalar loop
	// ================================================================
	label(.CSCALV_SCALAR_LOOP)
	lea(mem(, rdx, 8), r11)              // r11 = incx * sizeof(scomplex) = incx * 8 bytes

	label(.CSCALV_SCALAR_ITER)
	test(rsi, rsi)                       // check if any elements remaining
	jz(.CSCALV_DONE)                     // if zero, we're done

	// Load real and imaginary parts
	vmovss(mem(rax), xmm0)               // xmm0 = x_real
	vmovss(mem(rax, 4), xmm1)            // xmm1 = x_imag

	// Compute complex multiplication
	vmulss(xmm9, xmm1, xmm2)             // xmm2 = imag * x_imag
	vmulss(xmm9, xmm0, xmm3)             // xmm3 = imag * x_real

	vfmsub231ss(xmm0, xmm8, xmm2)        // xmm2 = real*x_real - imag*x_imag
	vfmadd231ss(xmm1, xmm8, xmm3)        // xmm3 = real*x_imag + imag*x_real

	// Store results
	vmovss(xmm2, mem(rax))               // x[i].real = xmm2
	vmovss(xmm3, mem(rax, 4))            // x[i].imag = xmm3

	add(r11, rax)                        // rax += incx * 8 (move to next element)
	sub(imm(1), rsi)                     // reduce remaining elements by 1
	jmp(.CSCALV_SCALAR_ITER)             // repeat for next element

	label(.CSCALV_DONE)
	vzeroupper()                         // clear upper 128 bits of all YMM registers

	end_asm
	(
		: // Output operands (none)
		: // Input operands
		  [x0] 		"m" (x0),
		  [n0] 		"m" (n0),
		  [incx0] 	"m" (incx0),
		  [real] 	"m" (real),
		  [imag] 	"m" (imag),
		  [mask_table] 	"m" (mask_table)
		: // Clobbered registers
		  "rax",     "rsi",     "rdx",     "rcx",
		  "r10",     "r11",     "ymm0",    "ymm1",
		  "ymm2",    "ymm3",    "ymm4",    "ymm6",
		  "ymm7",    "ymm8",    "ymm9",    "ymm10",
		  "ymm11",   "ymm12",   "ymm13",   "ymm15",
		  "cc",      "memory"
	)
}

void bli_zscalv_zen_int
	(
	conj_t           conjalpha,
	dim_t            n,
	dcomplex* restrict alpha,
	dcomplex* restrict x, inc_t incx,
	cntx_t* restrict cntx
	)
{
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

	dim_t i = 0;
	dcomplex alpha_conj;
	double *x0 = (double *)x;

	// Performs conjugation of alpha based on conjalpha
	PASTEMAC(z, copycjs)(conjalpha, *alpha, alpha_conj)

	double real = alpha_conj.real;
	double imag = alpha_conj.imag;

	/*When incx is 1 and n0 >= 2 it is possible to use AVX2 instructions*/
	if (incx == 1 && n0 >= 2)
	{
		dim_t const n_elem_per_reg = 4;

		__m256d alpha_real_ymm, alpha_imag_ymm;

		alpha_real_ymm = _mm256_broadcast_sd(&real);
		alpha_imag_ymm = _mm256_broadcast_sd(&imag);

		__m256d x_vec_ymm[4], temp_ymm[8];

		/*  Code logic

			Consider,
			x1= a1 + ib1, x2 = a1 + ib2
			alpha = p + iq

			Vector values
			x_vec_ymm = a1, b1, a2, b2
			alpha_real_ymm = p, p, p, p
			alpha_imag_ymm = q, q, q, q

			Computation

			All real values
			temp_1 = x_vec_ymm * alpha_real_ymm = a1p, b1p, a2p, b2p

			All imaginary values
			temp_2 = x_vec_ymm * alpha_imag_ymm = a1q, b1q, a2q, b2q

			permute temp_2 to get

			b1q, a1q, b2q, a2q

			addsub temp_1 and temp_2 to get the final result
			and then store
		*/

		for (; (i + 7) < n0; i += 8)
		{
			x_vec_ymm[0] = _mm256_loadu_pd(x0);
			x_vec_ymm[1] = _mm256_loadu_pd(x0 + n_elem_per_reg);
			x_vec_ymm[2] = _mm256_loadu_pd(x0 + 2 * n_elem_per_reg);
			x_vec_ymm[3] = _mm256_loadu_pd(x0 + 3 * n_elem_per_reg);

			temp_ymm[0] = _mm256_mul_pd(x_vec_ymm[0], alpha_imag_ymm);
			temp_ymm[1] = _mm256_mul_pd(x_vec_ymm[1], alpha_imag_ymm);
			temp_ymm[2] = _mm256_mul_pd(x_vec_ymm[2], alpha_imag_ymm);
			temp_ymm[3] = _mm256_mul_pd(x_vec_ymm[3], alpha_imag_ymm);

			temp_ymm[4] = _mm256_permute_pd(temp_ymm[0], 0b0101);
			temp_ymm[5] = _mm256_permute_pd(temp_ymm[1], 0b0101);
			temp_ymm[6] = _mm256_permute_pd(temp_ymm[2], 0b0101);
			temp_ymm[7] = _mm256_permute_pd(temp_ymm[3], 0b0101);

			/*
				a[i+63:i] := alpha_real * b[i+63:i] - c[i+63:i] for odd indices
				a[i+63:i] := alpha_real * b[i+63:i] + c[i+63:i] for even indices
			*/
			temp_ymm[0] = _mm256_fmaddsub_pd(x_vec_ymm[0], alpha_real_ymm, temp_ymm[4]);
			temp_ymm[1] = _mm256_fmaddsub_pd(x_vec_ymm[1], alpha_real_ymm, temp_ymm[5]);
			temp_ymm[2] = _mm256_fmaddsub_pd(x_vec_ymm[2], alpha_real_ymm, temp_ymm[6]);
			temp_ymm[3] = _mm256_fmaddsub_pd(x_vec_ymm[3], alpha_real_ymm, temp_ymm[7]);

			_mm256_storeu_pd(x0, temp_ymm[0]);
			_mm256_storeu_pd(x0 + n_elem_per_reg, temp_ymm[1]);
			_mm256_storeu_pd(x0 + 2 * n_elem_per_reg, temp_ymm[2]);
			_mm256_storeu_pd(x0 + 3 * n_elem_per_reg, temp_ymm[3]);

			x0 += 4 * n_elem_per_reg;
		}

		for (; (i + 3) < n0; i += 4)
		{
			x_vec_ymm[0] = _mm256_loadu_pd(x0);
			x_vec_ymm[1] = _mm256_loadu_pd(x0 + n_elem_per_reg);

			temp_ymm[0] = _mm256_mul_pd(x_vec_ymm[0], alpha_imag_ymm);
			temp_ymm[1] = _mm256_mul_pd(x_vec_ymm[1], alpha_imag_ymm);

			temp_ymm[2] = _mm256_permute_pd(temp_ymm[0], 0b0101);
			temp_ymm[3] = _mm256_permute_pd(temp_ymm[1], 0b0101);

			temp_ymm[0] = _mm256_fmaddsub_pd(x_vec_ymm[0], alpha_real_ymm, temp_ymm[2]);
			temp_ymm[1] = _mm256_fmaddsub_pd(x_vec_ymm[1], alpha_real_ymm, temp_ymm[3]);

			_mm256_storeu_pd(x0, temp_ymm[0]);
			_mm256_storeu_pd(x0 + n_elem_per_reg, temp_ymm[1]);

			x0 += 2 * n_elem_per_reg;
		}

		for (; (i + 1) < n0; i += 2)
		{
			x_vec_ymm[0] = _mm256_loadu_pd(x0);

			temp_ymm[0] = _mm256_mul_pd(x_vec_ymm[0], alpha_imag_ymm);

			temp_ymm[1] = _mm256_permute_pd(temp_ymm[0], 0b0101);

			temp_ymm[0] = _mm256_fmaddsub_pd(x_vec_ymm[0], alpha_real_ymm, temp_ymm[1]);

			_mm256_storeu_pd(x0, temp_ymm[0]);

			x0 += n_elem_per_reg;
		}

		// Issue vzeroupper instruction to clear upper lanes of ymm registers.
		// This avoids a performance penalty caused by false dependencies when
		// transitioning from AVX to SSE instructions (which may occur later,
		// especially if BLIS is compiled with -mfpmath=sse).
		_mm256_zeroupper();
	}

	/* In double complex data type the computation of
	unit stride elements can still be vectorized using SSE*/
	__m128d temp_xmm[2], alpha_real_xmm, alpha_imag_xmm, x_vec_xmm;

	alpha_real_xmm = _mm_set1_pd(real);
	alpha_imag_xmm = _mm_set1_pd(imag);

	for (; i < n0; i++)
	{
		x_vec_xmm = _mm_loadu_pd(x0);

		temp_xmm[0] = _mm_permute_pd(x_vec_xmm, 0b01);

		temp_xmm[1] = _mm_mul_pd(temp_xmm[0], alpha_imag_xmm);

		temp_xmm[0] = _mm_fmaddsub_pd(x_vec_xmm, alpha_real_xmm, temp_xmm[1]);

		_mm_storeu_pd(x0, temp_xmm[0]);

		x0 += 2 * incx;
	}
}
