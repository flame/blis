/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2022 - 2024, Advanced Micro Devices, Inc. All rights reserved.

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

/* Union DS to access AVX registers */
/* One 256-bit AVX register holds 8 SP elements */
typedef union
{
	__m256  v;
	float   f[8] __attribute__((aligned(64)));
} v8sf_t;

/* One 256-bit AVX register holds 4 DP elements */
typedef union
{
	__m256d v;
	double  d[4] __attribute__((aligned(64)));
} v4df_t;

/**
 * saxpbyv kernel performs the axpbyv operation.
 * y := beta * y + alpha * conjx(x)
 * where,
 * 		x & y are single precision vectors of length n.
 * 		alpha & beta are scalars.
 */
void bli_saxpbyv_zen_int10
	 (
	   conj_t           conjx,
	   dim_t            n,
	   float*  restrict alpha,
	   float*  restrict x, inc_t incx,
	   float*  restrict beta,
	   float*  restrict y, inc_t incy,
	   cntx_t* restrict cntx
	 )
{
	AOCL_DTL_TRACE_ENTRY(AOCL_DTL_LEVEL_TRACE_4)

	// Redirecting to other L1 kernels based on alpha and beta values
	// If alpha is 0, we call SSCALV
	// This kernel would further reroute based on few other combinations
	// of alpha and beta. They are as follows :
	// When alpha = 0 :
	// 		When beta = 0 --> SSETV
	// 		When beta = 1 --> Early return
	// 		When beta = !( 0 or 1 ) --> SSCALV
	if ( bli_seq0( *alpha ) )
	{
		bli_sscalv_zen_int10
		(
		  BLIS_NO_CONJUGATE,
		  n,
		  beta,
		  y, incy,
		  cntx
		);

		AOCL_DTL_TRACE_EXIT(AOCL_DTL_LEVEL_TRACE_4)
		return;
	}

	// If beta is 0, we call SSCAL2V
	// This kernel would further reroute based on few other combinations
	// of alpha and beta. They are as follows :
	// When beta = 0 :
	// 		When alpha = 0 --> SSETV
	// 		When alpha = 1 --> SCOPYV
	// 		When alpha = !( 0 or 1 ) --> SSCAL2V
	else if ( bli_seq0( *beta ) )
	{
		bli_sscal2v_zen_int
		(
		  conjx,
		  n,
		  alpha,
		  x, incx,
		  y, incy,
		  cntx
		);

		AOCL_DTL_TRACE_EXIT(AOCL_DTL_LEVEL_TRACE_4)
		return;
	}

	// If beta is 1, we have 2 scenarios for rerouting
	// 		When alpha = 1 --> SADDV
	// 		When alpha = !( 0 or 1 ) --> SAXPYV
	else if ( bli_seq1( *beta ) )
	{
		if( bli_seq1( *alpha ) )
		{
			bli_saddv_zen_int
			(
			  conjx,
			  n,
			  x, incx,
			  y, incy,
			  cntx
			);
		}
		else
		{
			bli_saxpyv_zen_int
			(
			  conjx,
			  n,
			  alpha,
			  x, incx,
			  y, incy,
			  cntx
			);
		}

		AOCL_DTL_TRACE_EXIT(AOCL_DTL_LEVEL_TRACE_4)
		return;
	}

	const dim_t n_elem_per_reg  = 8;    // number of elements per register

	dim_t i = 0;          // iterator

	float* restrict x0;
	float* restrict y0;

	v8sf_t alphav;
	v8sf_t betav;
	v8sf_t yv[10];

	bool is_alpha_one = bli_seq1( *alpha );

	// initialize local pointers
	x0 = x;
	y0 = y;

	if( incx == 1 && incy == 1 )
	{
		// Broadcasting beta onto a YMM register
		betav.v = _mm256_broadcast_ss( beta );

		if( is_alpha_one ) // Scale y with beta and add x to it
		{
			// Processing 80 elements per loop, 10 FMAs
			for ( ; ( i + 79 ) < n; i += 80 )
			{
				// Loading input values
				yv[0].v =  _mm256_loadu_ps( x0 + 0*n_elem_per_reg );
				yv[1].v =  _mm256_loadu_ps( x0 + 1*n_elem_per_reg );
				yv[2].v =  _mm256_loadu_ps( x0 + 2*n_elem_per_reg );
				yv[3].v =  _mm256_loadu_ps( x0 + 3*n_elem_per_reg );
				yv[4].v =  _mm256_loadu_ps( x0 + 4*n_elem_per_reg );
				yv[5].v =  _mm256_loadu_ps( x0 + 5*n_elem_per_reg );
				yv[6].v =  _mm256_loadu_ps( x0 + 6*n_elem_per_reg );
				yv[7].v =  _mm256_loadu_ps( x0 + 7*n_elem_per_reg );
				yv[8].v =  _mm256_loadu_ps( x0 + 8*n_elem_per_reg );
				yv[9].v =  _mm256_loadu_ps( x0 + 9*n_elem_per_reg );

				// y := beta * y + x
				yv[0].v = _mm256_fmadd_ps
							(
								betav.v,
								_mm256_loadu_ps( y0 + 0*n_elem_per_reg ),
								yv[0].v
							);
				yv[1].v = _mm256_fmadd_ps
							(
								betav.v,
								_mm256_loadu_ps( y0 + 1*n_elem_per_reg ),
								yv[1].v
							);
				yv[2].v = _mm256_fmadd_ps
							(
								betav.v,
								_mm256_loadu_ps( y0 + 2*n_elem_per_reg ),
								yv[2].v
							);
				yv[3].v = _mm256_fmadd_ps
							(
								betav.v,
								_mm256_loadu_ps( y0 + 3*n_elem_per_reg ),
								yv[3].v
							);
				yv[4].v = _mm256_fmadd_ps
							(
								betav.v,
								_mm256_loadu_ps( y0 + 4*n_elem_per_reg ),
								yv[4].v
							);
				yv[5].v = _mm256_fmadd_ps
							(
								betav.v,
								_mm256_loadu_ps( y0 + 5*n_elem_per_reg ),
								yv[5].v
							);
				yv[6].v = _mm256_fmadd_ps
							(
								betav.v,
								_mm256_loadu_ps( y0 + 6*n_elem_per_reg ),
								yv[6].v
							);
				yv[7].v = _mm256_fmadd_ps
							(
								betav.v,
								_mm256_loadu_ps( y0 + 7*n_elem_per_reg ),
								yv[7].v
							);
				yv[8].v = _mm256_fmadd_ps
							(
								betav.v,
								_mm256_loadu_ps( y0 + 8*n_elem_per_reg ),
								yv[8].v
							);
				yv[9].v = _mm256_fmadd_ps
							(
								betav.v,
								_mm256_loadu_ps( y0 + 9*n_elem_per_reg ),
								yv[9].v
							);

				// Storing the output
				_mm256_storeu_ps( ( y0 + 0*n_elem_per_reg ), yv[0].v );
				_mm256_storeu_ps( ( y0 + 1*n_elem_per_reg ), yv[1].v );
				_mm256_storeu_ps( ( y0 + 2*n_elem_per_reg ), yv[2].v );
				_mm256_storeu_ps( ( y0 + 3*n_elem_per_reg ), yv[3].v );
				_mm256_storeu_ps( ( y0 + 4*n_elem_per_reg ), yv[4].v );
				_mm256_storeu_ps( ( y0 + 5*n_elem_per_reg ), yv[5].v );
				_mm256_storeu_ps( ( y0 + 6*n_elem_per_reg ), yv[6].v );
				_mm256_storeu_ps( ( y0 + 7*n_elem_per_reg ), yv[7].v );
				_mm256_storeu_ps( ( y0 + 8*n_elem_per_reg ), yv[8].v );
				_mm256_storeu_ps( ( y0 + 9*n_elem_per_reg ), yv[9].v );

				x0 += 10 * n_elem_per_reg;
				y0 += 10 * n_elem_per_reg;
			}

			// Processing 40 elements per loop, 5 FMAs
			for ( ; ( i + 39 ) < n; i += 40 )
			{
				// Loading input values
				yv[0].v =  _mm256_loadu_ps( x0 + 0*n_elem_per_reg );
				yv[1].v =  _mm256_loadu_ps( x0 + 1*n_elem_per_reg );
				yv[2].v =  _mm256_loadu_ps( x0 + 2*n_elem_per_reg );
				yv[3].v =  _mm256_loadu_ps( x0 + 3*n_elem_per_reg );
				yv[4].v =  _mm256_loadu_ps( x0 + 4*n_elem_per_reg );

				// y := beta * y + x
				yv[0].v = _mm256_fmadd_ps
							(
								betav.v,
								_mm256_loadu_ps( y0 + 0*n_elem_per_reg ),
								yv[0].v
							);
				yv[1].v = _mm256_fmadd_ps
							(
								betav.v,
								_mm256_loadu_ps( y0 + 1*n_elem_per_reg ),
								yv[1].v
							);
				yv[2].v = _mm256_fmadd_ps
							(
								betav.v,
								_mm256_loadu_ps( y0 + 2*n_elem_per_reg ),
								yv[2].v
							);
				yv[3].v = _mm256_fmadd_ps
							(
								betav.v,
								_mm256_loadu_ps( y0 + 3*n_elem_per_reg ),
								yv[3].v
							);
				yv[4].v = _mm256_fmadd_ps
							(
								betav.v,
								_mm256_loadu_ps( y0 + 4*n_elem_per_reg ),
								yv[4].v
							);

				// Storing the output
				_mm256_storeu_ps( ( y0 + 0*n_elem_per_reg ), yv[0].v );
				_mm256_storeu_ps( ( y0 + 1*n_elem_per_reg ), yv[1].v );
				_mm256_storeu_ps( ( y0 + 2*n_elem_per_reg ), yv[2].v );
				_mm256_storeu_ps( ( y0 + 3*n_elem_per_reg ), yv[3].v );
				_mm256_storeu_ps( ( y0 + 4*n_elem_per_reg ), yv[4].v );

				x0 += 5 * n_elem_per_reg;
				y0 += 5 * n_elem_per_reg;
			}

			// Processing 32 elements per loop, 4 FMAs
			for ( ; ( i + 31 ) < n; i += 32 )
			{
				// Loading input values
				yv[0].v =  _mm256_loadu_ps( x0 + 0*n_elem_per_reg );
				yv[1].v =  _mm256_loadu_ps( x0 + 1*n_elem_per_reg );
				yv[2].v =  _mm256_loadu_ps( x0 + 2*n_elem_per_reg );
				yv[3].v =  _mm256_loadu_ps( x0 + 3*n_elem_per_reg );

				// y := beta * y + x
				yv[0].v = _mm256_fmadd_ps
							(
								betav.v,
								_mm256_loadu_ps( y0 + 0*n_elem_per_reg ),
								yv[0].v
							);
				yv[1].v = _mm256_fmadd_ps
							(
								betav.v,
								_mm256_loadu_ps( y0 + 1*n_elem_per_reg ),
								yv[1].v
							);
				yv[2].v = _mm256_fmadd_ps
							(
								betav.v,
								_mm256_loadu_ps( y0 + 2*n_elem_per_reg ),
								yv[2].v
							);
				yv[3].v = _mm256_fmadd_ps
							(
								betav.v,
								_mm256_loadu_ps( y0 + 3*n_elem_per_reg ),
								yv[3].v
							);

				// Storing the output
				_mm256_storeu_ps( ( y0 + 0*n_elem_per_reg ), yv[0].v );
				_mm256_storeu_ps( ( y0 + 1*n_elem_per_reg ), yv[1].v );
				_mm256_storeu_ps( ( y0 + 2*n_elem_per_reg ), yv[2].v );
				_mm256_storeu_ps( ( y0 + 3*n_elem_per_reg ), yv[3].v );

				x0 += 4 * n_elem_per_reg;
				y0 += 4 * n_elem_per_reg;
			}

			// Processing 16 elements per loop, 2 FMAs
			for ( ; ( i + 15 ) < n; i += 16 )
			{
				// Loading input values
				yv[0].v =  _mm256_loadu_ps( x0 + 0*n_elem_per_reg );
				yv[1].v =  _mm256_loadu_ps( x0 + 1*n_elem_per_reg );

				// y := beta * y + x
				yv[0].v = _mm256_fmadd_ps
							(
								betav.v,
								_mm256_loadu_ps( y0 + 0*n_elem_per_reg ),
								yv[0].v
							);
				yv[1].v = _mm256_fmadd_ps
							(
								betav.v,
								_mm256_loadu_ps( y0 + 1*n_elem_per_reg ),
								yv[1].v
							);

				// Storing the output
				_mm256_storeu_ps( ( y0 + 0*n_elem_per_reg ), yv[0].v );
				_mm256_storeu_ps( ( y0 + 1*n_elem_per_reg ), yv[1].v );


				x0 += 2 * n_elem_per_reg;
				y0 += 2 * n_elem_per_reg;
			}

			// Processing 8 elements per loop, 1 FMA
			for ( ; ( i + 7 ) < n; i += 8 )
			{
				// Loading input values
				yv[0].v =  _mm256_loadu_ps( x0 + 0*n_elem_per_reg );

				// y := beta * y + x
				yv[0].v = _mm256_fmadd_ps
							(
								betav.v,
								_mm256_loadu_ps( y0 + 0*n_elem_per_reg ),
								yv[0].v
							);

				// Storing the output
				_mm256_storeu_ps( ( y0 + 0*n_elem_per_reg ), yv[0].v );

				x0 += 1 * n_elem_per_reg;
				y0 += 1 * n_elem_per_reg;
			}
		}
		else
		{
			// Broadcasting alpha onto a YMM register
			alphav.v = _mm256_broadcast_ss( alpha );

			// Processing 80 elements per loop, 10 FMAs and MULs
			for ( i = 0; ( i + 79 ) < n; i += 80 )
			{
				// loading input values
				yv[0].v =  _mm256_loadu_ps( y0 + 0*n_elem_per_reg );
				yv[1].v =  _mm256_loadu_ps( y0 + 1*n_elem_per_reg );
				yv[2].v =  _mm256_loadu_ps( y0 + 2*n_elem_per_reg );
				yv[3].v =  _mm256_loadu_ps( y0 + 3*n_elem_per_reg );
				yv[4].v =  _mm256_loadu_ps( y0 + 4*n_elem_per_reg );
				yv[5].v =  _mm256_loadu_ps( y0 + 5*n_elem_per_reg );
				yv[6].v =  _mm256_loadu_ps( y0 + 6*n_elem_per_reg );
				yv[7].v =  _mm256_loadu_ps( y0 + 7*n_elem_per_reg );
				yv[8].v =  _mm256_loadu_ps( y0 + 8*n_elem_per_reg );
				yv[9].v =  _mm256_loadu_ps( y0 + 9*n_elem_per_reg );

				// y' := beta * y
				yv[0].v = 	_mm256_mul_ps( betav.v, yv[0].v );
				yv[1].v = 	_mm256_mul_ps( betav.v, yv[1].v );
				yv[2].v = 	_mm256_mul_ps( betav.v, yv[2].v );
				yv[3].v = 	_mm256_mul_ps( betav.v, yv[3].v );
				yv[4].v = 	_mm256_mul_ps( betav.v, yv[4].v );
				yv[5].v = 	_mm256_mul_ps( betav.v, yv[5].v );
				yv[6].v = 	_mm256_mul_ps( betav.v, yv[6].v );
				yv[7].v = 	_mm256_mul_ps( betav.v, yv[7].v );
				yv[8].v = 	_mm256_mul_ps( betav.v, yv[8].v );
				yv[9].v = 	_mm256_mul_ps( betav.v, yv[9].v );

				// y := y' + alpha * x
				yv[0].v = _mm256_fmadd_ps
							(
								alphav.v,
								_mm256_loadu_ps( x0 + 0*n_elem_per_reg ),
								yv[0].v
							);
				yv[1].v = _mm256_fmadd_ps
							(
								alphav.v,
								_mm256_loadu_ps( x0 + 1*n_elem_per_reg ),
								yv[1].v
							);
				yv[2].v = _mm256_fmadd_ps
							(
								alphav.v,
								_mm256_loadu_ps( x0 + 2*n_elem_per_reg ),
								yv[2].v
							);
				yv[3].v = _mm256_fmadd_ps
							(
								alphav.v,
								_mm256_loadu_ps( x0 + 3*n_elem_per_reg ),
								yv[3].v
							);
				yv[4].v = _mm256_fmadd_ps
							(
								alphav.v,
								_mm256_loadu_ps( x0 + 4*n_elem_per_reg ),
								yv[4].v
							);
				yv[5].v = _mm256_fmadd_ps
							(
								alphav.v,
								_mm256_loadu_ps( x0 + 5*n_elem_per_reg ),
								yv[5].v
							);
				yv[6].v = _mm256_fmadd_ps
							(
								alphav.v,
								_mm256_loadu_ps( x0 + 6*n_elem_per_reg ),
								yv[6].v
							);
				yv[7].v = _mm256_fmadd_ps
							(
								alphav.v,
								_mm256_loadu_ps( x0 + 7*n_elem_per_reg ),
								yv[7].v
							);
				yv[8].v = _mm256_fmadd_ps
							(
								alphav.v,
								_mm256_loadu_ps( x0 + 8*n_elem_per_reg ),
								yv[8].v
							);
				yv[9].v = _mm256_fmadd_ps
							(
								alphav.v,
								_mm256_loadu_ps( x0 + 9*n_elem_per_reg ),
								yv[9].v
							);

				// storing the output
				_mm256_storeu_ps( ( y0 + 0*n_elem_per_reg ), yv[0].v );
				_mm256_storeu_ps( ( y0 + 1*n_elem_per_reg ), yv[1].v );
				_mm256_storeu_ps( ( y0 + 2*n_elem_per_reg ), yv[2].v );
				_mm256_storeu_ps( ( y0 + 3*n_elem_per_reg ), yv[3].v );
				_mm256_storeu_ps( ( y0 + 4*n_elem_per_reg ), yv[4].v );
				_mm256_storeu_ps( ( y0 + 5*n_elem_per_reg ), yv[5].v );
				_mm256_storeu_ps( ( y0 + 6*n_elem_per_reg ), yv[6].v );
				_mm256_storeu_ps( ( y0 + 7*n_elem_per_reg ), yv[7].v );
				_mm256_storeu_ps( ( y0 + 8*n_elem_per_reg ), yv[8].v );
				_mm256_storeu_ps( ( y0 + 9*n_elem_per_reg ), yv[9].v );

				x0 += 10 * n_elem_per_reg;
				y0 += 10 * n_elem_per_reg;
			}

			// Processing 40 elements per loop, 5 FMAs and MULs
			for ( ; ( i + 39 ) < n; i += 40 )
			{
				// loading input values
				yv[0].v = _mm256_loadu_ps( y0 + 0*n_elem_per_reg );
				yv[1].v = _mm256_loadu_ps( y0 + 1*n_elem_per_reg );
				yv[2].v = _mm256_loadu_ps( y0 + 2*n_elem_per_reg );
				yv[3].v = _mm256_loadu_ps( y0 + 3*n_elem_per_reg );
				yv[4].v = _mm256_loadu_ps( y0 + 4*n_elem_per_reg );

				// y' := beta * y
				yv[0].v = _mm256_mul_ps( betav.v, yv[0].v );
				yv[1].v = _mm256_mul_ps( betav.v, yv[1].v );
				yv[2].v = _mm256_mul_ps( betav.v, yv[2].v );
				yv[3].v = _mm256_mul_ps( betav.v, yv[3].v );
				yv[4].v = _mm256_mul_ps( betav.v, yv[4].v );

				// y := y' + alpha * x
				yv[0].v = _mm256_fmadd_ps
							(
								alphav.v,
								_mm256_loadu_ps( x0 + 0*n_elem_per_reg ),
								yv[0].v
							);
				yv[1].v = _mm256_fmadd_ps
							(
								alphav.v,
								_mm256_loadu_ps( x0 + 1*n_elem_per_reg ),
								yv[1].v
							);
				yv[2].v = _mm256_fmadd_ps
							(
								alphav.v,
								_mm256_loadu_ps( x0 + 2*n_elem_per_reg ),
								yv[2].v
							);
				yv[3].v = _mm256_fmadd_ps
							(
								alphav.v,
								_mm256_loadu_ps( x0 + 3*n_elem_per_reg ),
								yv[3].v
							);
				yv[4].v = _mm256_fmadd_ps
							(
								alphav.v,
								_mm256_loadu_ps( x0 + 4*n_elem_per_reg ),
								yv[4].v
							);

				// storing the output
				_mm256_storeu_ps( ( y0 + 0*n_elem_per_reg ), yv[0].v );
				_mm256_storeu_ps( ( y0 + 1*n_elem_per_reg ), yv[1].v );
				_mm256_storeu_ps( ( y0 + 2*n_elem_per_reg ), yv[2].v );
				_mm256_storeu_ps( ( y0 + 3*n_elem_per_reg ), yv[3].v );
				_mm256_storeu_ps( ( y0 + 4*n_elem_per_reg ), yv[4].v );

				x0 += 5 * n_elem_per_reg;
				y0 += 5 * n_elem_per_reg;
			}

			// Processing 32 elements per loop, 4 FMAs and MULs
			for ( ; ( i + 31 ) < n; i += 32 )
			{
				// loading input values
				yv[0].v = _mm256_loadu_ps( y0 + 0*n_elem_per_reg );
				yv[1].v = _mm256_loadu_ps( y0 + 1*n_elem_per_reg );
				yv[2].v = _mm256_loadu_ps( y0 + 2*n_elem_per_reg );
				yv[3].v = _mm256_loadu_ps( y0 + 3*n_elem_per_reg );

				// y' := beta * y
				yv[0].v = _mm256_mul_ps( betav.v, yv[0].v );
				yv[1].v = _mm256_mul_ps( betav.v, yv[1].v );
				yv[2].v = _mm256_mul_ps( betav.v, yv[2].v );
				yv[3].v = _mm256_mul_ps( betav.v, yv[3].v );

				// y := y' + alpha * x
				yv[0].v = _mm256_fmadd_ps
							(
								alphav.v,
								_mm256_loadu_ps( x0 + 0*n_elem_per_reg ),
								yv[0].v
							);
				yv[1].v = _mm256_fmadd_ps
							(
								alphav.v,
								_mm256_loadu_ps( x0 + 1*n_elem_per_reg ),
								yv[1].v
							);
				yv[2].v = _mm256_fmadd_ps
							(
								alphav.v,
								_mm256_loadu_ps( x0 + 2*n_elem_per_reg ),
								yv[2].v
							);
				yv[3].v = _mm256_fmadd_ps
							(
								alphav.v,
								_mm256_loadu_ps( x0 + 3*n_elem_per_reg ),
								yv[3].v
							);

				// storing the output
				_mm256_storeu_ps( ( y0 + 0*n_elem_per_reg ), yv[0].v );
				_mm256_storeu_ps( ( y0 + 1*n_elem_per_reg ), yv[1].v );
				_mm256_storeu_ps( ( y0 + 2*n_elem_per_reg ), yv[2].v );
				_mm256_storeu_ps( ( y0 + 3*n_elem_per_reg ), yv[3].v );

				x0 += 4 * n_elem_per_reg;
				y0 += 4 * n_elem_per_reg;
			}

			// Processing 16 elements per loop, 2 FMAs and MULs
			for ( ; ( i + 15 ) < n; i += 16 )
			{
				// loading input values
				yv[0].v = _mm256_loadu_ps( y0 + 0*n_elem_per_reg );
				yv[1].v = _mm256_loadu_ps( y0 + 1*n_elem_per_reg );

				// y' := beta * y
				yv[0].v = _mm256_mul_ps( betav.v, yv[0].v );
				yv[1].v = _mm256_mul_ps( betav.v, yv[1].v );

				// y := y' + alpha * x
				yv[0].v = _mm256_fmadd_ps
							(
								alphav.v,
								_mm256_loadu_ps( x0 + 0*n_elem_per_reg ),
								yv[0].v
							);
				yv[1].v = _mm256_fmadd_ps
							(
								alphav.v,
								_mm256_loadu_ps( x0 + 1*n_elem_per_reg ),
								yv[1].v
							);

				// storing the output
				_mm256_storeu_ps( ( y0 + 0*n_elem_per_reg ), yv[0].v );
				_mm256_storeu_ps( ( y0 + 1*n_elem_per_reg ), yv[1].v );

				x0 += 2 * n_elem_per_reg;
				y0 += 2 * n_elem_per_reg;
			}

			// Processing 8 elements per loop, 1 FMA and MUL
			for ( ; ( i + 7 ) < n; i += 8 )
			{
				// loading input values
				yv[0].v = _mm256_loadu_ps( y0 + 0*n_elem_per_reg );

				// y' := y := beta * y
				yv[0].v = _mm256_mul_ps( betav.v, yv[0].v );

				// y := y' + alpha * x
				yv[0].v = _mm256_fmadd_ps
							(
								alphav.v,
								_mm256_loadu_ps( x0 + 0*n_elem_per_reg ),
								yv[0].v
							);

				// storing the output
				_mm256_storeu_ps( ( y0 + 0*n_elem_per_reg ), yv[0].v );

				x0 += 1 * n_elem_per_reg;
				y0 += 1 * n_elem_per_reg;
			}
		}

		// Issue vzeroupper instruction to clear upper lanes of ymm registers.
		// This avoids a performance penalty caused by false dependencies when
		// transitioning from AVX to SSE instructions (which may occur as soon
		// as the n_left cleanup loop below if BLIS is compiled with
		// -mfpmath=sse).
		_mm256_zeroupper();
	}

	if( is_alpha_one )
	{
		for ( ; i < n; ++i )
		{
			*y0 = (*beta) * (*y0) + (*x0);

			x0 += incx;
			y0 += incy;
		}
	}
	else
	{
		for ( ; i < n; ++i )
		{
			*y0 = (*beta) * (*y0) + (*alpha) * (*x0);

			x0 += incx;
			y0 += incy;
		}
	}

	AOCL_DTL_TRACE_EXIT(AOCL_DTL_LEVEL_TRACE_4)
}

/**
 * daxpbyv kernel performs the axpbyv operation.
 * y := beta * y + alpha * conjx(x)
 * where,
 * 		x & y are double precision vectors of length n.
 * 		alpha & beta are scalars.
 */
void bli_daxpbyv_zen_int10
	 (
	   conj_t           conjx,
	   dim_t            n,
	   double* restrict alpha,
	   double* restrict x, inc_t incx,
	   double* restrict beta,
	   double* restrict y, inc_t incy,
	   cntx_t* restrict cntx
	 )
{
	AOCL_DTL_TRACE_ENTRY(AOCL_DTL_LEVEL_TRACE_4)

	// Redirecting to other L1 kernels based on alpha and beta values
	// If alpha is 0, we call DSCALV
	// This kernel would further reroute based on few other combinations
	// of alpha and beta. They are as follows :
	// When alpha = 0 :
	// 		When beta = 0 --> DSETV
	// 		When beta = 1 --> Early return
	// 		When beta = !( 0 or 1 ) --> DSCALV
	if ( bli_deq0( *alpha ) )
	{
		bli_dscalv_zen_int10
		(
		  BLIS_NO_CONJUGATE,
		  n,
		  beta,
		  y, incy,
		  cntx
		);

		AOCL_DTL_TRACE_EXIT(AOCL_DTL_LEVEL_TRACE_4)
		return;
	}

	// If beta is 0, we call DSCAL2V
	// This kernel would further reroute based on few other combinations
	// of alpha and beta. They are as follows :
	// When beta = 0 :
	// 		When alpha = 0 --> DSETV
	// 		When alpha = 1 --> DCOPYV
	// 		When alpha = !( 0 or 1 ) --> DSCAL2V
	else if ( bli_deq0( *beta ) )
	{
		bli_dscal2v_zen_int
		(
		  conjx,
		  n,
		  alpha,
		  x, incx,
		  y, incy,
		  cntx
		);

		AOCL_DTL_TRACE_EXIT(AOCL_DTL_LEVEL_TRACE_4)
		return;
	}

	// If beta is 1, we have 2 scenarios for rerouting
	// 		When alpha = 1 --> DADDV
	// 		When alpha = !( 0 or 1 ) --> DAXPYV
	else if ( bli_deq1( *beta ) )
	{
		if( bli_deq1( *alpha ) )
		{
			bli_daddv_zen_int
			(
			  conjx,
			  n,
			  x, incx,
			  y, incy,
			  cntx
			);
		}
		else
		{
			bli_daxpyv_zen_int
			(
			  conjx,
			  n,
			  alpha,
			  x, incx,
			  y, incy,
			  cntx
			);
		}

		AOCL_DTL_TRACE_EXIT(AOCL_DTL_LEVEL_TRACE_4)
		return;
	}

	const dim_t n_elem_per_reg  = 4;    // number of elements per register

	dim_t i = 0;          // iterator

	double* restrict x0;
	double* restrict y0;

	v4df_t alphav;
	v4df_t betav;
	v4df_t yv[10];

	bool is_alpha_one = bli_seq1( *alpha );

	// initialize local pointers
	x0 = x;
	y0 = y;

	if( incx == 1 && incy == 1 )
	{
		// Broadcasting beta onto a YMM register
		betav.v = _mm256_broadcast_sd( beta );

		if( is_alpha_one ) // Scale y with beta and add x to it
		{
			// Processing 40 elements per loop, 10 FMAs
			for ( ; ( i + 39 ) < n; i += 40 )
			{
				// Loading input values
				yv[0].v =  _mm256_loadu_pd( x0 + 0*n_elem_per_reg );
				yv[1].v =  _mm256_loadu_pd( x0 + 1*n_elem_per_reg );
				yv[2].v =  _mm256_loadu_pd( x0 + 2*n_elem_per_reg );
				yv[3].v =  _mm256_loadu_pd( x0 + 3*n_elem_per_reg );
				yv[4].v =  _mm256_loadu_pd( x0 + 4*n_elem_per_reg );
				yv[5].v =  _mm256_loadu_pd( x0 + 5*n_elem_per_reg );
				yv[6].v =  _mm256_loadu_pd( x0 + 6*n_elem_per_reg );
				yv[7].v =  _mm256_loadu_pd( x0 + 7*n_elem_per_reg );
				yv[8].v =  _mm256_loadu_pd( x0 + 8*n_elem_per_reg );
				yv[9].v =  _mm256_loadu_pd( x0 + 9*n_elem_per_reg );

				// y := beta * y + x
				yv[0].v = _mm256_fmadd_pd
							(
								betav.v,
								_mm256_loadu_pd( y0 + 0*n_elem_per_reg ),
								yv[0].v
							);
				yv[1].v = _mm256_fmadd_pd
							(
								betav.v,
								_mm256_loadu_pd( y0 + 1*n_elem_per_reg ),
								yv[1].v
							);
				yv[2].v = _mm256_fmadd_pd
							(
								betav.v,
								_mm256_loadu_pd( y0 + 2*n_elem_per_reg ),
								yv[2].v
							);
				yv[3].v = _mm256_fmadd_pd
							(
								betav.v,
								_mm256_loadu_pd( y0 + 3*n_elem_per_reg ),
								yv[3].v
							);
				yv[4].v = _mm256_fmadd_pd
							(
								betav.v,
								_mm256_loadu_pd( y0 + 4*n_elem_per_reg ),
								yv[4].v
							);
				yv[5].v = _mm256_fmadd_pd
							(
								betav.v,
								_mm256_loadu_pd( y0 + 5*n_elem_per_reg ),
								yv[5].v
							);
				yv[6].v = _mm256_fmadd_pd
							(
								betav.v,
								_mm256_loadu_pd( y0 + 6*n_elem_per_reg ),
								yv[6].v
							);
				yv[7].v = _mm256_fmadd_pd
							(
								betav.v,
								_mm256_loadu_pd( y0 + 7*n_elem_per_reg ),
								yv[7].v
							);
				yv[8].v = _mm256_fmadd_pd
							(
								betav.v,
								_mm256_loadu_pd( y0 + 8*n_elem_per_reg ),
								yv[8].v
							);
				yv[9].v = _mm256_fmadd_pd
							(
								betav.v,
								_mm256_loadu_pd( y0 + 9*n_elem_per_reg ),
								yv[9].v
							);

				// Storing the output
				_mm256_storeu_pd( ( y0 + 0*n_elem_per_reg ), yv[0].v );
				_mm256_storeu_pd( ( y0 + 1*n_elem_per_reg ), yv[1].v );
				_mm256_storeu_pd( ( y0 + 2*n_elem_per_reg ), yv[2].v );
				_mm256_storeu_pd( ( y0 + 3*n_elem_per_reg ), yv[3].v );
				_mm256_storeu_pd( ( y0 + 4*n_elem_per_reg ), yv[4].v );
				_mm256_storeu_pd( ( y0 + 5*n_elem_per_reg ), yv[5].v );
				_mm256_storeu_pd( ( y0 + 6*n_elem_per_reg ), yv[6].v );
				_mm256_storeu_pd( ( y0 + 7*n_elem_per_reg ), yv[7].v );
				_mm256_storeu_pd( ( y0 + 8*n_elem_per_reg ), yv[8].v );
				_mm256_storeu_pd( ( y0 + 9*n_elem_per_reg ), yv[9].v );

				x0 += 10 * n_elem_per_reg;
				y0 += 10 * n_elem_per_reg;
			}

			// Processing 20 elements per loop, 5 FMAs
			for ( ; ( i + 19 ) < n; i += 20 )
			{
				// Loading input values
				yv[0].v =  _mm256_loadu_pd( x0 + 0*n_elem_per_reg );
				yv[1].v =  _mm256_loadu_pd( x0 + 1*n_elem_per_reg );
				yv[2].v =  _mm256_loadu_pd( x0 + 2*n_elem_per_reg );
				yv[3].v =  _mm256_loadu_pd( x0 + 3*n_elem_per_reg );
				yv[4].v =  _mm256_loadu_pd( x0 + 4*n_elem_per_reg );

				// y := beta * y + x
				yv[0].v = _mm256_fmadd_pd
							(
								betav.v,
								_mm256_loadu_pd( y0 + 0*n_elem_per_reg ),
								yv[0].v
							);
				yv[1].v = _mm256_fmadd_pd
							(
								betav.v,
								_mm256_loadu_pd( y0 + 1*n_elem_per_reg ),
								yv[1].v
							);
				yv[2].v = _mm256_fmadd_pd
							(
								betav.v,
								_mm256_loadu_pd( y0 + 2*n_elem_per_reg ),
								yv[2].v
							);
				yv[3].v = _mm256_fmadd_pd
							(
								betav.v,
								_mm256_loadu_pd( y0 + 3*n_elem_per_reg ),
								yv[3].v
							);
				yv[4].v = _mm256_fmadd_pd
							(
								betav.v,
								_mm256_loadu_pd( y0 + 4*n_elem_per_reg ),
								yv[4].v
							);

				// Storing the output
				_mm256_storeu_pd( ( y0 + 0*n_elem_per_reg ), yv[0].v );
				_mm256_storeu_pd( ( y0 + 1*n_elem_per_reg ), yv[1].v );
				_mm256_storeu_pd( ( y0 + 2*n_elem_per_reg ), yv[2].v );
				_mm256_storeu_pd( ( y0 + 3*n_elem_per_reg ), yv[3].v );
				_mm256_storeu_pd( ( y0 + 4*n_elem_per_reg ), yv[4].v );

				x0 += 5 * n_elem_per_reg;
				y0 += 5 * n_elem_per_reg;
			}

			// Processing 16 elements per loop, 4 FMAs
			for ( ; ( i + 15 ) < n; i += 16 )
			{
				// Loading input values
				yv[0].v =  _mm256_loadu_pd( x0 + 0*n_elem_per_reg );
				yv[1].v =  _mm256_loadu_pd( x0 + 1*n_elem_per_reg );
				yv[2].v =  _mm256_loadu_pd( x0 + 2*n_elem_per_reg );
				yv[3].v =  _mm256_loadu_pd( x0 + 3*n_elem_per_reg );

				// y := beta * y + x
				yv[0].v = _mm256_fmadd_pd
							(
								betav.v,
								_mm256_loadu_pd( y0 + 0*n_elem_per_reg ),
								yv[0].v
							);
				yv[1].v = _mm256_fmadd_pd
							(
								betav.v,
								_mm256_loadu_pd( y0 + 1*n_elem_per_reg ),
								yv[1].v
							);
				yv[2].v = _mm256_fmadd_pd
							(
								betav.v,
								_mm256_loadu_pd( y0 + 2*n_elem_per_reg ),
								yv[2].v
							);
				yv[3].v = _mm256_fmadd_pd
							(
								betav.v,
								_mm256_loadu_pd( y0 + 3*n_elem_per_reg ),
								yv[3].v
							);

				// Storing the output
				_mm256_storeu_pd( ( y0 + 0*n_elem_per_reg ), yv[0].v );
				_mm256_storeu_pd( ( y0 + 1*n_elem_per_reg ), yv[1].v );
				_mm256_storeu_pd( ( y0 + 2*n_elem_per_reg ), yv[2].v );
				_mm256_storeu_pd( ( y0 + 3*n_elem_per_reg ), yv[3].v );

				x0 += 4 * n_elem_per_reg;
				y0 += 4 * n_elem_per_reg;
			}

			// Processing 8 elements per loop, 2 FMAs
			for ( ; ( i + 7 ) < n; i += 8 )
			{
				// Loading input values
				yv[0].v =  _mm256_loadu_pd( x0 + 0*n_elem_per_reg );
				yv[1].v =  _mm256_loadu_pd( x0 + 1*n_elem_per_reg );

				// y := beta * y + x
				yv[0].v = _mm256_fmadd_pd
							(
								betav.v,
								_mm256_loadu_pd( y0 + 0*n_elem_per_reg ),
								yv[0].v
							);
				yv[1].v = _mm256_fmadd_pd
							(
								betav.v,
								_mm256_loadu_pd( y0 + 1*n_elem_per_reg ),
								yv[1].v
							);

				// Storing the output
				_mm256_storeu_pd( ( y0 + 0*n_elem_per_reg ), yv[0].v );
				_mm256_storeu_pd( ( y0 + 1*n_elem_per_reg ), yv[1].v );


				x0 += 2 * n_elem_per_reg;
				y0 += 2 * n_elem_per_reg;
			}

			// Processing 4 elements per loop, 1 FMA
			for ( ; ( i + 3 ) < n; i += 4 )
			{
				// Loading input values
				yv[0].v =  _mm256_loadu_pd( x0 + 0*n_elem_per_reg );

				// y := beta * y + x
				yv[0].v = _mm256_fmadd_pd
							(
								betav.v,
								_mm256_loadu_pd( y0 + 0*n_elem_per_reg ),
								yv[0].v
							);

				// Storing the output
				_mm256_storeu_pd( ( y0 + 0*n_elem_per_reg ), yv[0].v );

				x0 += 1 * n_elem_per_reg;
				y0 += 1 * n_elem_per_reg;
			}
		}
		else
		{
			// Broadcasting alpha onto a YMM register
			alphav.v = _mm256_broadcast_sd( alpha );

			// Processing 40 elements per loop, 10 FMAs and MULs
			for ( i = 0; ( i + 39 ) < n; i += 40 )
			{
				// loading input values
				yv[0].v =  _mm256_loadu_pd( y0 + 0*n_elem_per_reg );
				yv[1].v =  _mm256_loadu_pd( y0 + 1*n_elem_per_reg );
				yv[2].v =  _mm256_loadu_pd( y0 + 2*n_elem_per_reg );
				yv[3].v =  _mm256_loadu_pd( y0 + 3*n_elem_per_reg );
				yv[4].v =  _mm256_loadu_pd( y0 + 4*n_elem_per_reg );
				yv[5].v =  _mm256_loadu_pd( y0 + 5*n_elem_per_reg );
				yv[6].v =  _mm256_loadu_pd( y0 + 6*n_elem_per_reg );
				yv[7].v =  _mm256_loadu_pd( y0 + 7*n_elem_per_reg );
				yv[8].v =  _mm256_loadu_pd( y0 + 8*n_elem_per_reg );
				yv[9].v =  _mm256_loadu_pd( y0 + 9*n_elem_per_reg );

				// y' := beta * y
				yv[0].v = 	_mm256_mul_pd( betav.v, yv[0].v );
				yv[1].v = 	_mm256_mul_pd( betav.v, yv[1].v );
				yv[2].v = 	_mm256_mul_pd( betav.v, yv[2].v );
				yv[3].v = 	_mm256_mul_pd( betav.v, yv[3].v );
				yv[4].v = 	_mm256_mul_pd( betav.v, yv[4].v );
				yv[5].v = 	_mm256_mul_pd( betav.v, yv[5].v );
				yv[6].v = 	_mm256_mul_pd( betav.v, yv[6].v );
				yv[7].v = 	_mm256_mul_pd( betav.v, yv[7].v );
				yv[8].v = 	_mm256_mul_pd( betav.v, yv[8].v );
				yv[9].v = 	_mm256_mul_pd( betav.v, yv[9].v );

				// y := y' + alpha * x
				yv[0].v = _mm256_fmadd_pd
							(
								alphav.v,
								_mm256_loadu_pd( x0 + 0*n_elem_per_reg ),
								yv[0].v
							);
				yv[1].v = _mm256_fmadd_pd
							(
								alphav.v,
								_mm256_loadu_pd( x0 + 1*n_elem_per_reg ),
								yv[1].v
							);
				yv[2].v = _mm256_fmadd_pd
							(
								alphav.v,
								_mm256_loadu_pd( x0 + 2*n_elem_per_reg ),
								yv[2].v
							);
				yv[3].v = _mm256_fmadd_pd
							(
								alphav.v,
								_mm256_loadu_pd( x0 + 3*n_elem_per_reg ),
								yv[3].v
							);
				yv[4].v = _mm256_fmadd_pd
							(
								alphav.v,
								_mm256_loadu_pd( x0 + 4*n_elem_per_reg ),
								yv[4].v
							);
				yv[5].v = _mm256_fmadd_pd
							(
								alphav.v,
								_mm256_loadu_pd( x0 + 5*n_elem_per_reg ),
								yv[5].v
							);
				yv[6].v = _mm256_fmadd_pd
							(
								alphav.v,
								_mm256_loadu_pd( x0 + 6*n_elem_per_reg ),
								yv[6].v
							);
				yv[7].v = _mm256_fmadd_pd
							(
								alphav.v,
								_mm256_loadu_pd( x0 + 7*n_elem_per_reg ),
								yv[7].v
							);
				yv[8].v = _mm256_fmadd_pd
							(
								alphav.v,
								_mm256_loadu_pd( x0 + 8*n_elem_per_reg ),
								yv[8].v
							);
				yv[9].v = _mm256_fmadd_pd
							(
								alphav.v,
								_mm256_loadu_pd( x0 + 9*n_elem_per_reg ),
								yv[9].v
							);

				// storing the output
				_mm256_storeu_pd( ( y0 + 0*n_elem_per_reg ), yv[0].v );
				_mm256_storeu_pd( ( y0 + 1*n_elem_per_reg ), yv[1].v );
				_mm256_storeu_pd( ( y0 + 2*n_elem_per_reg ), yv[2].v );
				_mm256_storeu_pd( ( y0 + 3*n_elem_per_reg ), yv[3].v );
				_mm256_storeu_pd( ( y0 + 4*n_elem_per_reg ), yv[4].v );
				_mm256_storeu_pd( ( y0 + 5*n_elem_per_reg ), yv[5].v );
				_mm256_storeu_pd( ( y0 + 6*n_elem_per_reg ), yv[6].v );
				_mm256_storeu_pd( ( y0 + 7*n_elem_per_reg ), yv[7].v );
				_mm256_storeu_pd( ( y0 + 8*n_elem_per_reg ), yv[8].v );
				_mm256_storeu_pd( ( y0 + 9*n_elem_per_reg ), yv[9].v );

				x0 += 10 * n_elem_per_reg;
				y0 += 10 * n_elem_per_reg;
			}

			// Processing 20 elements per loop, 5 FMAs and MULs
			for ( ; ( i + 19 ) < n; i += 20 )
			{
				// loading input values
				yv[0].v = _mm256_loadu_pd( y0 + 0*n_elem_per_reg );
				yv[1].v = _mm256_loadu_pd( y0 + 1*n_elem_per_reg );
				yv[2].v = _mm256_loadu_pd( y0 + 2*n_elem_per_reg );
				yv[3].v = _mm256_loadu_pd( y0 + 3*n_elem_per_reg );
				yv[4].v = _mm256_loadu_pd( y0 + 4*n_elem_per_reg );

				// y' := beta * y
				yv[0].v = _mm256_mul_pd( betav.v, yv[0].v );
				yv[1].v = _mm256_mul_pd( betav.v, yv[1].v );
				yv[2].v = _mm256_mul_pd( betav.v, yv[2].v );
				yv[3].v = _mm256_mul_pd( betav.v, yv[3].v );
				yv[4].v = _mm256_mul_pd( betav.v, yv[4].v );

				// y := y' + alpha * x
				yv[0].v = _mm256_fmadd_pd
							(
								alphav.v,
								_mm256_loadu_pd( x0 + 0*n_elem_per_reg ),
								yv[0].v
							);
				yv[1].v = _mm256_fmadd_pd
							(
								alphav.v,
								_mm256_loadu_pd( x0 + 1*n_elem_per_reg ),
								yv[1].v
							);
				yv[2].v = _mm256_fmadd_pd
							(
								alphav.v,
								_mm256_loadu_pd( x0 + 2*n_elem_per_reg ),
								yv[2].v
							);
				yv[3].v = _mm256_fmadd_pd
							(
								alphav.v,
								_mm256_loadu_pd( x0 + 3*n_elem_per_reg ),
								yv[3].v
							);
				yv[4].v = _mm256_fmadd_pd
							(
								alphav.v,
								_mm256_loadu_pd( x0 + 4*n_elem_per_reg ),
								yv[4].v
							);

				// storing the output
				_mm256_storeu_pd( ( y0 + 0*n_elem_per_reg ), yv[0].v );
				_mm256_storeu_pd( ( y0 + 1*n_elem_per_reg ), yv[1].v );
				_mm256_storeu_pd( ( y0 + 2*n_elem_per_reg ), yv[2].v );
				_mm256_storeu_pd( ( y0 + 3*n_elem_per_reg ), yv[3].v );
				_mm256_storeu_pd( ( y0 + 4*n_elem_per_reg ), yv[4].v );

				x0 += 5 * n_elem_per_reg;
				y0 += 5 * n_elem_per_reg;
			}

			// Processing 16 elements per loop, 4 FMAs and MULs
			for ( ; ( i + 15 ) < n; i += 16 )
			{
				// loading input values
				yv[0].v = _mm256_loadu_pd( y0 + 0*n_elem_per_reg );
				yv[1].v = _mm256_loadu_pd( y0 + 1*n_elem_per_reg );
				yv[2].v = _mm256_loadu_pd( y0 + 2*n_elem_per_reg );
				yv[3].v = _mm256_loadu_pd( y0 + 3*n_elem_per_reg );

				// y' := beta * y
				yv[0].v = _mm256_mul_pd( betav.v, yv[0].v );
				yv[1].v = _mm256_mul_pd( betav.v, yv[1].v );
				yv[2].v = _mm256_mul_pd( betav.v, yv[2].v );
				yv[3].v = _mm256_mul_pd( betav.v, yv[3].v );

				// y := y' + alpha * x
				yv[0].v = _mm256_fmadd_pd
							(
								alphav.v,
								_mm256_loadu_pd( x0 + 0*n_elem_per_reg ),
								yv[0].v
							);
				yv[1].v = _mm256_fmadd_pd
							(
								alphav.v,
								_mm256_loadu_pd( x0 + 1*n_elem_per_reg ),
								yv[1].v
							);
				yv[2].v = _mm256_fmadd_pd
							(
								alphav.v,
								_mm256_loadu_pd( x0 + 2*n_elem_per_reg ),
								yv[2].v
							);
				yv[3].v = _mm256_fmadd_pd
							(
								alphav.v,
								_mm256_loadu_pd( x0 + 3*n_elem_per_reg ),
								yv[3].v
							);

				// storing the output
				_mm256_storeu_pd( ( y0 + 0*n_elem_per_reg ), yv[0].v );
				_mm256_storeu_pd( ( y0 + 1*n_elem_per_reg ), yv[1].v );
				_mm256_storeu_pd( ( y0 + 2*n_elem_per_reg ), yv[2].v );
				_mm256_storeu_pd( ( y0 + 3*n_elem_per_reg ), yv[3].v );

				x0 += 4 * n_elem_per_reg;
				y0 += 4 * n_elem_per_reg;
			}

			// Processing 8 elements per loop, 2 FMAs and MULs
			for ( ; ( i + 7 ) < n; i += 8 )
			{
				// loading input values
				yv[0].v = _mm256_loadu_pd( y0 + 0*n_elem_per_reg );
				yv[1].v = _mm256_loadu_pd( y0 + 1*n_elem_per_reg );

				// y' := beta * y
				yv[0].v = _mm256_mul_pd( betav.v, yv[0].v );
				yv[1].v = _mm256_mul_pd( betav.v, yv[1].v );

				// y := y' + alpha * x
				yv[0].v = _mm256_fmadd_pd
							(
								alphav.v,
								_mm256_loadu_pd( x0 + 0*n_elem_per_reg ),
								yv[0].v
							);
				yv[1].v = _mm256_fmadd_pd
							(
								alphav.v,
								_mm256_loadu_pd( x0 + 1*n_elem_per_reg ),
								yv[1].v
							);

				// storing the output
				_mm256_storeu_pd( ( y0 + 0*n_elem_per_reg ), yv[0].v );
				_mm256_storeu_pd( ( y0 + 1*n_elem_per_reg ), yv[1].v );

				x0 += 2 * n_elem_per_reg;
				y0 += 2 * n_elem_per_reg;
			}

			// Processing 4 elements per loop, 1 FMA and MUL
			for ( ; ( i + 3 ) < n; i += 4 )
			{
				// loading input values
				yv[0].v = _mm256_loadu_pd( y0 + 0*n_elem_per_reg );

				// y' := y := beta * y
				yv[0].v = _mm256_mul_pd( betav.v, yv[0].v );

				// y := y' + alpha * x
				yv[0].v = _mm256_fmadd_pd
							(
								alphav.v,
								_mm256_loadu_pd( x0 + 0*n_elem_per_reg ),
								yv[0].v
							);

				// storing the output
				_mm256_storeu_pd( ( y0 + 0*n_elem_per_reg ), yv[0].v );

				x0 += 1 * n_elem_per_reg;
				y0 += 1 * n_elem_per_reg;
			}
		}

		// Issue vzeroupper instruction to clear upper lanes of ymm registers.
		// This avoids a performance penalty caused by false dependencies when
		// transitioning from AVX to SSE instructions (which may occur as soon
		// as the n_left cleanup loop below if BLIS is compiled with
		// -mfpmath=sse).
		_mm256_zeroupper();
	}

	// Handling fringe cases or non-unit strided inputs
	if( is_alpha_one )
	{
		for ( ; i < n; ++i )
		{
			*y0 = (*beta) * (*y0) + (*x0);

			x0 += incx;
			y0 += incy;
		}
	}
	else
	{
		for ( ; i < n; ++i )
		{
			*y0 = (*beta) * (*y0) + (*alpha) * (*x0);

			x0 += incx;
			y0 += incy;
		}
	}

	AOCL_DTL_TRACE_EXIT(AOCL_DTL_LEVEL_TRACE_4)
}
