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
void bli_saxpbyv_zen_int
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

	const dim_t     n_elem_per_reg  = 8;    // number of elements per register
	const dim_t     n_iter_unroll   = 4;    // num of registers per iteration

	dim_t           i = 0;          // iterator

	float* restrict x0;
	float* restrict y0;

	v8sf_t          alphav;
	v8sf_t          betav;
	v8sf_t          yv[4];

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

				x0 += n_elem_per_reg * n_iter_unroll;
				y0 += n_elem_per_reg * n_iter_unroll;
			}
		}
		else
		{
			// Broadcasting alpha onto a YMM register
			alphav.v = _mm256_broadcast_ss( alpha );

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

				x0 += n_elem_per_reg * n_iter_unroll;
				y0 += n_elem_per_reg * n_iter_unroll;
			}
		}

		// Issue vzeroupper instruction to clear upper lanes of ymm registers.
		// This avoids a performance penalty caused by false dependencies when
		// transitioning from AVX to SSE instructions (which may occur as soon
		// as the n_left cleanup loop below if BLIS is compiled with
		// -mfpmath=sse).
		_mm256_zeroupper();
	}

	// Handling fringe cases or non-unit strides
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
void bli_daxpbyv_zen_int
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

	const dim_t     n_elem_per_reg  = 4;    // number of elements per register
	const dim_t     n_iter_unroll   = 4;    // number of registers per iteration

	dim_t           i = 0;          // iterator

	double* restrict x0;
	double* restrict y0;

	v4df_t          alphav;
	v4df_t          betav;
	v4df_t          yv[4];

	bool is_alpha_one = bli_deq1( *alpha );

	// initialize local pointers
	x0 = x;
	y0 = y;
	
	if ( incx == 1 && incy == 1 )
	{
		// Broadcasting beta onto a YMM register
		betav.v = _mm256_broadcast_sd( beta );

		if( is_alpha_one ) // Scale y with beta and add x to it
		{
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

				x0 += n_elem_per_reg * n_iter_unroll;
				y0 += n_elem_per_reg * n_iter_unroll;
			}
		}
		else
		{
			// Broadcasting alpha onto a YMM register
			alphav.v = _mm256_broadcast_sd( alpha );

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

				x0 += n_elem_per_reg * n_iter_unroll;
				y0 += n_elem_per_reg * n_iter_unroll;
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

/**
 * caxpbyv kernel performs the axpbyv operation.
 * y := beta * y + alpha * conjx(x)
 * where,
 * 		x & y are simple complex vectors of length n.
 * 		alpha & beta are scalars.
 */
void bli_caxpbyv_zen_int
	 (
	   conj_t             conjx,
	   dim_t              n,
	   scomplex* restrict alpha,
	   scomplex* restrict x, inc_t incx,
	   scomplex* restrict beta,
	   scomplex* restrict y, inc_t incy,
	   cntx_t*   restrict cntx
	 )
{
	AOCL_DTL_TRACE_ENTRY(AOCL_DTL_LEVEL_TRACE_4)

	// Redirecting to other L1 kernels based on alpha and beta values
	// If alpha is 0, we call CSCALV
	// This kernel would further reroute based on few other combinations
	// of alpha and beta. They are as follows :
	// When alpha = 0 :
	// 		When beta = 0 --> CSETV
	// 		When beta = 1 --> Early return
	// 		When beta = !( 0 or 1 ) --> CSCALV
	if ( bli_ceq0( *alpha ) )
	{
		bli_cscalv_zen_int
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

	// If beta is 0, we call CSCAL2V
	// This kernel would further reroute based on few other combinations
	// of alpha and beta. They are as follows :
	// When beta = 0 :
	// 		When alpha = 0 --> CSETV
	// 		When alpha = 1 --> CCOPYV
	// 		When alpha = !( 0 or 1 ) --> CSCAL2V
	else if ( bli_ceq0( *beta ) )
	{
		bli_cscal2v_zen_int
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
	// 		When alpha = 1 --> CADDV
	// 		When alpha = !( 0 or 1 ) --> CAXPYV
	else if ( bli_ceq1( *beta ) )
	{
		if( bli_ceq1( *alpha ) )
		{
			bli_caddv_zen_int
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
			bli_caxpyv_zen_int5
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

	dim_t i = 0; // iterator

	// Local pointers to x and y vectors
	float*  restrict x0;
	float*  restrict y0;

	// Boolean to check if alpha is 1
	bool is_alpha_one = bli_ceq1( *alpha );

	// Variables to store real and imaginary components of alpha and beta
	float alphaR, alphaI, betaR, betaI;

	// Initializing the local pointers
	x0  = ( float* ) x;
	y0  = ( float* ) y;

	alphaR = alpha->real;
	alphaI = alpha->imag;
	betaR  = beta->real;
	betaI  = beta->imag;

	// In case of unit strides for x and y vectors
	if ( incx == 1 && incy == 1 )
	{
		// Number of float precision elements in a YMM register
		const dim_t  n_elem_per_reg = 8;

		// Scratch registers
		__m256 xv[4];
		__m256 yv[4];
		__m256 iv[4];

		// Vectors to store real and imaginary components of beta
		__m256 betaRv, betaIv;

		// Broadcasting real and imaginary components of beta onto the registers
		betaRv = _mm256_broadcast_ss( &betaR );
		betaIv = _mm256_broadcast_ss( &betaI );

		if( is_alpha_one )
		{
			__m256 reg_one = _mm256_set1_ps(1.0f);
			iv[0] = _mm256_setzero_ps();

			// Converting reg_one to have {1.0, -1.0, 1.0, -1.0, ...}
			// This is needed in case we have t0 conjugate X vector
			if( bli_is_conj( conjx ) )
			{
				reg_one = _mm256_fmsubadd_ps( reg_one, iv[0], reg_one );
			}
			// Processing 16 elements per loop, 8 FMAs
			for ( ; ( i + 15 ) < n; i += 16 )
			{
				// Load the y vector, 16 elements in total
				// yv =  yR1  yI1  yR2  yI2 ...
				yv[0] = _mm256_loadu_ps( y0 );
				yv[1] = _mm256_loadu_ps( y0 + 1 * n_elem_per_reg );
				yv[2] = _mm256_loadu_ps( y0 + 2 * n_elem_per_reg );
				yv[3] = _mm256_loadu_ps( y0 + 3 * n_elem_per_reg );

				// Load the x vector, 16 elements in total
				// xv = xR1  xI1  xR2  xI2 ...
				xv[0] = _mm256_loadu_ps( x0 );
				xv[1] = _mm256_loadu_ps( x0 + 1 * n_elem_per_reg );
				xv[2] = _mm256_loadu_ps( x0 + 2 * n_elem_per_reg );
				xv[3] = _mm256_loadu_ps( x0 + 3 * n_elem_per_reg );

				// Permute the vectors from y for the required compute
				// iv =  yI1  yR1  yI2  yR2 ...
				iv[0] = _mm256_permute_ps( yv[0], 0xB1 );
				iv[1] = _mm256_permute_ps( yv[1], 0xB1 );
				iv[2] = _mm256_permute_ps( yv[2], 0xB1 );
				iv[3] = _mm256_permute_ps( yv[3], 0xB1 );

				// Scale the permuted vectors with imaginary component of beta
				// iv = betaIv * yv
				//    = yI1.bI, yR1.bI, yI2.bI, yR2.bI, ...
				iv[0] = _mm256_mul_ps( betaIv, iv[0] );
				iv[1] = _mm256_mul_ps( betaIv, iv[1] );
				iv[2] = _mm256_mul_ps( betaIv, iv[2] );
				iv[3] = _mm256_mul_ps( betaIv, iv[3] );

				// Using fmaddsub to scale with real component of beta
				// and sub/add to iv
				// yv = betaRv * yv -/+ iv
				//    = yR1.bR - yI1.bI, yI1.bR + yR1.bI, ...
				yv[0] = _mm256_fmaddsub_ps( betaRv, yv[0], iv[0] );
				yv[1] = _mm256_fmaddsub_ps( betaRv, yv[1], iv[1] );
				yv[2] = _mm256_fmaddsub_ps( betaRv, yv[2], iv[2] );
				yv[3] = _mm256_fmaddsub_ps( betaRv, yv[3], iv[3] );

				// Adding X conjugate to it
				yv[0] = _mm256_fmadd_ps( reg_one, xv[0], yv[0] );
				yv[1] = _mm256_fmadd_ps( reg_one, xv[1], yv[1] );
				yv[2] = _mm256_fmadd_ps( reg_one, xv[2], yv[2] );
				yv[3] = _mm256_fmadd_ps( reg_one, xv[3], yv[3] );

				// Storing the result to memory
				_mm256_storeu_ps( ( y0 ), yv[0] );
				_mm256_storeu_ps( ( y0 + 1 * n_elem_per_reg ), yv[1] );
				_mm256_storeu_ps( ( y0 + 2 * n_elem_per_reg ), yv[2] );
				_mm256_storeu_ps( ( y0 + 3 * n_elem_per_reg ), yv[3] );

				// Adjusting the pointers for the next iteration
				y0 += 4 * n_elem_per_reg;
				x0 += 4 * n_elem_per_reg;
			}

			// Processing 12 elements per loop, 12 FMAs
			for ( ; ( i + 11 ) < n; i += 12 )
			{
				// Load the y vector, 12 elements in total
				// yv =  yR1  yI1  yR2  yI2 ...
				yv[0] = _mm256_loadu_ps( y0 );
				yv[1] = _mm256_loadu_ps( y0 + 1 * n_elem_per_reg );
				yv[2] = _mm256_loadu_ps( y0 + 2 * n_elem_per_reg );

				// Load the x vector, 12 elements in total
				// xv = xR1  xI1  xR2  xI2
				xv[0] = _mm256_loadu_ps( x0 );
				xv[1] = _mm256_loadu_ps( x0 + 1 * n_elem_per_reg );
				xv[2] = _mm256_loadu_ps( x0 + 2 * n_elem_per_reg );

				// Permute the vectors from y for the required compute
				// iv =  yI1  yR1  yI2  yR2 ...
				iv[0] = _mm256_permute_ps( yv[0], 0xB1 );
				iv[1] = _mm256_permute_ps( yv[1], 0xB1 );
				iv[2] = _mm256_permute_ps( yv[2], 0xB1 );

				// Scale the permuted vectors with imaginary component of beta
				// iv = betaIv * yv
				//    = yI1.bI, yR1.bI, yI2.bI, yR2.bI, ...
				iv[0] = _mm256_mul_ps( betaIv, iv[0] );
				iv[1] = _mm256_mul_ps( betaIv, iv[1] );
				iv[2] = _mm256_mul_ps( betaIv, iv[2] );

				// Using fmaddsub to scale with real component of beta
				// and sub/add to iv
				// yv = betaRv * yv -/+ iv
				//    = yR1.bR - yI1.bI, yI1.bR + yR1.bI, ...
				yv[0] = _mm256_fmaddsub_ps( betaRv, yv[0], iv[0] );
				yv[1] = _mm256_fmaddsub_ps( betaRv, yv[1], iv[1] );
				yv[2] = _mm256_fmaddsub_ps( betaRv, yv[2], iv[2] );

				// Adding X conjugate to it
				yv[0] = _mm256_fmadd_ps( reg_one, xv[0], yv[0] );
				yv[1] = _mm256_fmadd_ps( reg_one, xv[1], yv[1] );
				yv[2] = _mm256_fmadd_ps( reg_one, xv[2], yv[2] );

				// Storing the result to memory
				_mm256_storeu_ps( ( y0 ), yv[0] );
				_mm256_storeu_ps( ( y0 + 1 * n_elem_per_reg ), yv[1] );
				_mm256_storeu_ps( ( y0 + 2 * n_elem_per_reg ), yv[2] );

				// Adjusting the pointers for the next iteration
				y0 += 3 * n_elem_per_reg;
				x0 += 3 * n_elem_per_reg;
			}

			// Processing 8 elements per loop, 8 FMAs
			for ( ; ( i + 7 ) < n; i += 8 )
			{
				// Load the y vector, 8 elements in total
				// yv =  yR1  yI1  yR2  yI2 ...
				yv[0] = _mm256_loadu_ps( y0 );
				yv[1] = _mm256_loadu_ps( y0 + 1 * n_elem_per_reg );

				// Load the x vector, 8 elements in total
				// xv = xR1  xI1  xR2  xI2 ...
				xv[0] = _mm256_loadu_ps( x0 );
				xv[1] = _mm256_loadu_ps( x0 + 1 * n_elem_per_reg );

				// Permute the vectors from y for the required compute
				// iv =  yI1  yR1  yI2  yR2
				iv[0] = _mm256_permute_ps( yv[0], 0xB1 );
				iv[1] = _mm256_permute_ps( yv[1], 0xB1 );

				// Scale the permuted vectors with imaginary component of beta
				// iv = betaIv * yv
				//    = yI1.bI, yR1.bI, yI2.bI, yR2.bI, ...
				iv[0] = _mm256_mul_ps( betaIv, iv[0] );
				iv[1] = _mm256_mul_ps( betaIv, iv[1] );

				// Using fmaddsub to scale with real component of beta
				// and sub/add to iv
				// yv = betaRv * yv -/+ iv
				//    = yR1.bR - yI1.bI, yI1.bR + yR1.bI, ...
				yv[0] = _mm256_fmaddsub_ps( betaRv, yv[0], iv[0] );
				yv[1] = _mm256_fmaddsub_ps( betaRv, yv[1], iv[1] );

				// Adding X conjugate to it
				yv[0] = _mm256_fmadd_ps( reg_one, xv[0], yv[0] );
				yv[1] = _mm256_fmadd_ps( reg_one, xv[1], yv[1] );

				// Storing the result to memory
				_mm256_storeu_ps( ( y0 ), yv[0] );
				_mm256_storeu_ps( ( y0 + 1 * n_elem_per_reg ), yv[1] );

				// Adjusting the pointers for the next iteration
				y0 += 2 * n_elem_per_reg;
				x0 += 2 * n_elem_per_reg;
			}

			// Processing 4 elements per loop, 4 FMAs
			for ( ; ( i + 3 ) < n; i += 4 )
			{
				// Load the y vector, 4 elements in total
				// yv =  yR1  yI1  yR2  yI2 ...
				yv[0] = _mm256_loadu_ps( y0 );

				// Load the x vector, 4 elements in total
				// xv = xR1  xI1  xR2  xI2 ...
				xv[0] = _mm256_loadu_ps( x0 );

				// Permute the vectors from y for the required compute
				// iv =  yI1  yR1  yI2  yR2 ...
				iv[0] = _mm256_permute_ps( yv[0], 0xB1 );

				// Scale the permuted vectors with imaginary component of beta
				// iv = betaIv * yv
				//    = yI1.bI, yR1.bI, yI2.bI, yR2.bI, ...
				iv[0] = _mm256_mul_ps( betaIv, iv[0] );

				// Using fmaddsub to scale with real component of beta
				// and sub/add to iv
				// yv = betaRv * yv -/+ iv
				//    = yR1.bR - yI1.bI, yI1.bR + yR1.bI, ...
				yv[0] = _mm256_fmaddsub_ps( betaRv, yv[0], iv[0] );

				// Adding X conjugate to it
				yv[0] = _mm256_fmadd_ps( reg_one, xv[0], yv[0] );

				// Storing the result to memory
				_mm256_storeu_ps( ( y0 ), yv[0] );

				// Adjusting the pointers for the next iteration
				y0 += 1 * n_elem_per_reg;
				x0 += 1 * n_elem_per_reg;
			}
		}
		else
		{
			// Scratch registers for storing real and imaginary components of alpha
			__m256 alphaRv, alphaIv;

			iv[0] = _mm256_setzero_ps();

			alphaRv = _mm256_broadcast_ss( &alphaR );
			alphaIv = _mm256_broadcast_ss( &alphaI );

			// The changes on alphaRv and alphaIv are as follows :
			// If conjugate is required:
			//		alphaRv =  aR  -aR  aR  -aR
			// Else :
			//		alphaIv =  -aI  aI  -aI  aI
			if( bli_is_conj( conjx ) )
			{
				alphaRv = _mm256_fmsubadd_ps( iv[0], iv[0], alphaRv );
			}
			else
			{
				alphaIv = _mm256_addsub_ps( iv[0], alphaIv );
			}

			// Processing 16 elements per loop, 16 FMAs
			for ( i = 0; ( i + 15 ) < n; i += 16 )
			{
				// Load the y vector, 16 elements in total
				// yv =  yR1  yI1  yR2  yI2 ...
				yv[0] = _mm256_loadu_ps( y0 );
				yv[1] = _mm256_loadu_ps( y0 + 1 * n_elem_per_reg );
				yv[2] = _mm256_loadu_ps( y0 + 2 * n_elem_per_reg );
				yv[3] = _mm256_loadu_ps( y0 + 3 * n_elem_per_reg );

				// Load the x vector, 16 elements in total
				// xv = xR1  xI1  xR2  xI2 ...
				xv[0] = _mm256_loadu_ps( x0 );
				xv[1] = _mm256_loadu_ps( x0 + 1 * n_elem_per_reg );
				xv[2] = _mm256_loadu_ps( x0 + 2 * n_elem_per_reg );
				xv[3] = _mm256_loadu_ps( x0 + 3 * n_elem_per_reg );

				// Permute the vectors from y for the required compute
				// iv =  yI1  yR1  yI2  yR2 ...
				iv[0] = _mm256_permute_ps( yv[0], 0xB1 );
				iv[1] = _mm256_permute_ps( yv[1], 0xB1 );
				iv[2] = _mm256_permute_ps( yv[2], 0xB1 );
				iv[3] = _mm256_permute_ps( yv[3], 0xB1 );

				// Scale the permuted vectors with imaginary component of beta
				// iv = betaIv * yv
				//    = yI1.bI, yR1.bI, yI2.bI, yR2.bI, ...
				iv[0] = _mm256_mul_ps( betaIv, iv[0] );
				iv[1] = _mm256_mul_ps( betaIv, iv[1] );
				iv[2] = _mm256_mul_ps( betaIv, iv[2] );
				iv[3] = _mm256_mul_ps( betaIv, iv[3] );

				// Using fmaddsub to scale with real component of beta
				// and sub/add to iv
				// yv = betaRv * yv -/+ iv
				//    = yR1.bR - yI1.bI, yI1.bR + yR1.bI, ...
				yv[0] = _mm256_fmaddsub_ps( betaRv, yv[0], iv[0] );
				yv[1] = _mm256_fmaddsub_ps( betaRv, yv[1], iv[1] );
				yv[2] = _mm256_fmaddsub_ps( betaRv, yv[2], iv[2] );
				yv[3] = _mm256_fmaddsub_ps( betaRv, yv[3], iv[3] );

				// Permute the loaded vectors from x for the required compute
				// xv' =  xI1  xR1  xI2  xR2 ...
				iv[0] = _mm256_permute_ps( xv[0], 0xB1 );
				iv[1] = _mm256_permute_ps( xv[1], 0xB1 );
				iv[2] = _mm256_permute_ps( xv[2], 0xB1 );
				iv[3] = _mm256_permute_ps( xv[3], 0xB1 );

				// yv = alphaRv * xv + yv
				//    = yR1.bR - yR1.bI + xR1.aR, yI1.bR + yI1.bI + xI1.aR, ...
				yv[0] = _mm256_fmadd_ps( alphaRv, xv[0], yv[0] );
				yv[1] = _mm256_fmadd_ps( alphaRv, xv[1], yv[1] );
				yv[2] = _mm256_fmadd_ps( alphaRv, xv[2], yv[2] );
				yv[3] = _mm256_fmadd_ps( alphaRv, xv[3], yv[3] );

				// yv = alphaIv * iv + yv
				//    = yR1.bR - yR1.bI - xI1.aI, yI1.bR + yI1.bI + xR1.aI, ...
				yv[0] = _mm256_fmadd_ps( alphaIv, iv[0], yv[0] );
				yv[1] = _mm256_fmadd_ps( alphaIv, iv[1], yv[1] );
				yv[2] = _mm256_fmadd_ps( alphaIv, iv[2], yv[2] );
				yv[3] = _mm256_fmadd_ps( alphaIv, iv[3], yv[3] );

				// Storing the result to memory
				_mm256_storeu_ps( ( y0 ), yv[0] );
				_mm256_storeu_ps( ( y0 + 1 * n_elem_per_reg ), yv[1] );
				_mm256_storeu_ps( ( y0 + 2 * n_elem_per_reg ), yv[2] );
				_mm256_storeu_ps( ( y0 + 3 * n_elem_per_reg ), yv[3] );

				// Adjusting the pointers for the next iteration
				y0 += 4 * n_elem_per_reg;
				x0 += 4 * n_elem_per_reg;
			}

			// Processing 12 elements per loop, 12 FMAs
			for ( ; ( i + 11 ) < n; i += 12 )
			{
				// Load the y vector, 12 elements in total
				// yv =  yR1  yI1  yR2  yI2 ...
				yv[0] = _mm256_loadu_ps( y0 );
				yv[1] = _mm256_loadu_ps( y0 + 1 * n_elem_per_reg );
				yv[2] = _mm256_loadu_ps( y0 + 2 * n_elem_per_reg );

				// Load the x vector, 12 elements in total
				// xv = xR1  xI1  xR2  xI2 ...
				xv[0] = _mm256_loadu_ps( x0 );
				xv[1] = _mm256_loadu_ps( x0 + 1 * n_elem_per_reg );
				xv[2] = _mm256_loadu_ps( x0 + 2 * n_elem_per_reg );

				// Permute the vectors from y for the required compute
				// iv =  yI1  yR1  yI2  yR2 ...
				iv[0] = _mm256_permute_ps( yv[0], 0xB1 );
				iv[1] = _mm256_permute_ps( yv[1], 0xB1 );
				iv[2] = _mm256_permute_ps( yv[2], 0xB1 );

				// Scale the permuted vectors with imaginary component of beta
				// iv = betaIv * yv
				//    = yI1.bI, yR1.bI, yI2.bI, yR2.bI, ...`
				iv[0] = _mm256_mul_ps( betaIv, iv[0] );
				iv[1] = _mm256_mul_ps( betaIv, iv[1] );
				iv[2] = _mm256_mul_ps( betaIv, iv[2] );

				// Using fmaddsub to scale with real component of beta
				// and sub/add to iv
				// yv = betaRv * yv -/+ iv
				//    = yR1.bR - yI1.bI, yI1.bR + yR1.bI, ...
				yv[0] = _mm256_fmaddsub_ps( betaRv, yv[0], iv[0] );
				yv[1] = _mm256_fmaddsub_ps( betaRv, yv[1], iv[1] );
				yv[2] = _mm256_fmaddsub_ps( betaRv, yv[2], iv[2] );

				// Permute the loaded vectors from x for the required compute
				// xv' =  xI1  xR1  xI2  xR2 ...
				iv[0] = _mm256_permute_ps( xv[0], 0xB1 );
				iv[1] = _mm256_permute_ps( xv[1], 0xB1 );
				iv[2] = _mm256_permute_ps( xv[2], 0xB1 );

				// yv = alphaRv * xv + yv
				//    = yR1.bR - yR1.bI + xR1.aR, yI1.bR + yI1.bI + xI1.aR, ...
				yv[0] = _mm256_fmadd_ps( alphaRv, xv[0], yv[0] );
				yv[1] = _mm256_fmadd_ps( alphaRv, xv[1], yv[1] );
				yv[2] = _mm256_fmadd_ps( alphaRv, xv[2], yv[2] );

				// yv = alphaIv * iv + yv
				//    = yR1.bR - yR1.bI - xI1.aI, yI1.bR + yI1.bI + xR1.aI, ...
				yv[0] = _mm256_fmadd_ps( alphaIv, iv[0], yv[0] );
				yv[1] = _mm256_fmadd_ps( alphaIv, iv[1], yv[1] );
				yv[2] = _mm256_fmadd_ps( alphaIv, iv[2], yv[2] );

				// Storing the result to memory
				_mm256_storeu_ps( ( y0 ), yv[0] );
				_mm256_storeu_ps( ( y0 + 1 * n_elem_per_reg ), yv[1] );
				_mm256_storeu_ps( ( y0 + 2 * n_elem_per_reg ), yv[2] );

				// Adjusting the pointers for the next iteration
				y0 += 3 * n_elem_per_reg;
				x0 += 3 * n_elem_per_reg;
			}

			// Processing 8 elements per loop, 8 FMAs
			for ( ; ( i + 7 ) < n; i += 8 )
			{
				// Load the y vector, 8 elements in total
				// yv =  yR1  yI1  yR2  yI2 ...
				yv[0] = _mm256_loadu_ps( y0 );
				yv[1] = _mm256_loadu_ps( y0 + 1 * n_elem_per_reg );

				// Load the x vector, 8 elements in total
				// xv = xR1  xI1  xR2  xI2 ...
				xv[0] = _mm256_loadu_ps( x0 );
				xv[1] = _mm256_loadu_ps( x0 + 1 * n_elem_per_reg );

				// Permute the vectors from y for the required compute
				// iv =  yI1  yR1  yI2  yR2 ...
				iv[0] = _mm256_permute_ps( yv[0], 0xB1 );
				iv[1] = _mm256_permute_ps( yv[1], 0xB1 );

				// Scale the permuted vectors with imaginary component of beta
				// iv = betaIv * yv
				//    = yI1.bI, yR1.bI, yI2.bI, yR2.bI, ...
				iv[0] = _mm256_mul_ps( betaIv, iv[0] );
				iv[1] = _mm256_mul_ps( betaIv, iv[1] );

				// Using fmaddsub to scale with real component of beta
				// and sub/add to iv
				// yv = betaRv * yv -/+ iv
				//    = yR1.bR - yI1.bI, yI1.bR + yR1.bI, ...
				yv[0] = _mm256_fmaddsub_ps( betaRv, yv[0], iv[0] );
				yv[1] = _mm256_fmaddsub_ps( betaRv, yv[1], iv[1] );

				// Permute the loaded vectors from x for the required compute
				// xv' =  xI1  xR1  xI2  xR2
				iv[0] = _mm256_permute_ps( xv[0], 0xB1 );
				iv[1] = _mm256_permute_ps( xv[1], 0xB1 );

				// yv = alphaRv * xv + yv
				//    = yR1.bR - yR1.bI + xR1.aR, yI1.bR + yI1.bI + xI1.aR, ...
				yv[0] = _mm256_fmadd_ps( alphaRv, xv[0], yv[0] );
				yv[1] = _mm256_fmadd_ps( alphaRv, xv[1], yv[1] );

				// yv = alphaIv * iv + yv
				//    = yR1.bR - yR1.bI - xI1.aI, yI1.bR + yI1.bI + xR1.aI, ...
				yv[0] = _mm256_fmadd_ps( alphaIv, iv[0], yv[0] );
				yv[1] = _mm256_fmadd_ps( alphaIv, iv[1], yv[1] );

				// Storing the result to memory
				_mm256_storeu_ps( ( y0 ), yv[0] );
				_mm256_storeu_ps( ( y0 + 1 * n_elem_per_reg ), yv[1] );

				// Adjusting the pointers for the next iteration
				y0 += 2 * n_elem_per_reg;
				x0 += 2 * n_elem_per_reg;
			}

			// Processing 4 elements per loop, 4 FMAs
			for ( ; ( i + 3 ) < n; i += 4 )
			{
				// Load the y vector, 4 elements in total
				// yv =  yR1  yI1  yR2  yI2 ...
				yv[0] = _mm256_loadu_ps( y0 );

				// Load the x vector, 4 elements in total
				// xv = xR1  xI1  xR2  xI2 ...
				xv[0] = _mm256_loadu_ps( x0 );

				// Permute the vectors from y for the required compute
				// iv =  yI1  yR1  yI2  yR2 ...
				iv[0] = _mm256_permute_ps( yv[0], 0xB1 );

				// Scale the permuted vectors with imaginary component of beta
				// iv = betaIv * yv
				//    = yI1.bI, yR1.bI, yI2.bI, yR2.bI, ...
				iv[0] = _mm256_mul_ps( betaIv, iv[0] );

				// Using fmaddsub to scale with real component of beta
				// and sub/add to iv
				// yv = betaRv * yv -/+ iv
				//    = yR1.bR - yI1.bI, yI1.bR + yR1.bI, ...
				yv[0] = _mm256_fmaddsub_ps( betaRv, yv[0], iv[0] );

				// Permute the loaded vectors from x for the required compute
				// xv' =  xI1  xR1  xI2  xR2 ...
				iv[0] = _mm256_permute_ps( xv[0], 0xB1 );

				// yv = alphaRv * xv + yv
				//    = yR1.bR - yR1.bI + xR1.aR, yI1.bR + yI1.bI + xI1.aR, ...
				yv[0] = _mm256_fmadd_ps( alphaRv, xv[0], yv[0] );

				// yv = alphaIv * iv + yv
				//    = yR1.bR - yR1.bI - xI1.aI, yI1.bR + yI1.bI + xR1.aI, ...
				yv[0] = _mm256_fmadd_ps( alphaIv, iv[0], yv[0] );

				// Storing the result to memory
				_mm256_storeu_ps( ( y0 ), yv[0] );

				// Adjusting the pointers for the next iteration
				y0 += 1 * n_elem_per_reg;
				x0 += 1 * n_elem_per_reg;
			}
		}

		// Issue vzeroupper instruction to clear upper lanes of ymm registers.
		// This avoids a performance penalty caused by false dependencies when
		// transitioning from AVX to SSE instructions.
		_mm256_zeroupper();
	}

	// Handling fringe cases or non-unit-strides
	if ( is_alpha_one )
	{
		if( bli_is_conj( conjx ) )
		{
			for( ; i < n; i += 1 )
			{
				scomplex temp;
				temp.real = ( betaR * (*y0) ) - ( betaI * (*(y0 + 1)) ) + (*x0);
				temp.imag = ( betaR * (*(y0 + 1)) ) + ( betaI * (*y0) ) - (*(x0 + 1));

				(*y0) = temp.real;
				(*(y0 + 1)) = temp.imag;
				
				x0 += 2 * incx;
				y0 += 2 * incy;
			}
		}
		else
		{
			for( ; i < n; i += 1 )
			{
				scomplex temp;
				temp.real = ( betaR * (*y0) ) - ( betaI * (*(y0 + 1)) ) + (*x0);
				temp.imag = ( betaR * (*(y0 + 1)) ) + ( betaI * (*y0) ) + (*(x0 + 1));

				(*y0) = temp.real;
				(*(y0 + 1)) = temp.imag;
				
				x0 += 2 * incx;
				y0 += 2 * incy;
			}
		}
	}
	else
	{
		if( bli_is_conj( conjx ) )
		{
			for( ; i < n; i += 1 )
			{
				scomplex temp;
				temp.real = ( betaR * (*y0) ) - ( betaI * (*(y0 + 1)) ) +
										( alphaR * (*x0) ) + ( alphaI * (*(x0 + 1)) );
				temp.imag = ( betaR * (*(y0 + 1)) ) + ( betaI * (*y0) ) -
										( alphaR * (*(x0 + 1)) ) + ( alphaI * (*x0) );

				(*y0) = temp.real;
				(*(y0 + 1)) = temp.imag;
				
				x0 += 2 * incx;
				y0 += 2 * incy;
			}
		}
		else
		{
			for( ; i < n; i += 1 )
			{
				scomplex temp;
				temp.real = ( betaR * (*y0) ) - ( betaI * (*(y0 + 1)) ) +
										( alphaR * (*x0) ) - ( alphaI * (*(x0 + 1)) );
				temp.imag = ( betaR * (*(y0 + 1)) ) + ( betaI * (*y0) ) +
										( alphaR * (*(x0 + 1)) ) + ( alphaI * (*x0) );

				(*y0) = temp.real;
				(*(y0 + 1)) = temp.imag;
				
				x0 += 2 * incx;
				y0 += 2 * incy;
			}
		}
	}

	AOCL_DTL_TRACE_EXIT(AOCL_DTL_LEVEL_TRACE_4)
}

/**
 * zaxpbyv kernel performs the axpbyv operation.
 * y := beta * y + alpha * conjx(x)
 * where,
 * 		x & y are double complex vectors of length n.
 * 		alpha & beta are scalars.
 */
void bli_zaxpbyv_zen_int
	 (
	   conj_t           conjx,
	   dim_t            n,
	   dcomplex* restrict alpha,
	   dcomplex* restrict x, inc_t incx,
	   dcomplex* restrict beta,
	   dcomplex* restrict y, inc_t incy,
	   cntx_t*   restrict cntx
	 )
{
	AOCL_DTL_TRACE_ENTRY(AOCL_DTL_LEVEL_TRACE_4)

	// Redirecting to other L1 kernels based on alpha and beta values
	// If alpha is 0, we call ZSCALV
	// This kernel would further reroute based on few other combinations
	// of alpha and beta. They are as follows :
	// When alpha = 0 :
	// 		When beta = 0 --> ZSETV
	// 		When beta = 1 --> Early return
	// 		When beta = !( 0 or 1 ) --> ZSCALV
	if ( bli_ceq0( *alpha ) )
	{
		bli_zscalv_zen_int
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

	// If beta is 0, we call ZSCAL2V
	// This kernel would further reroute based on few other combinations
	// of alpha and beta. They are as follows :
	// When beta = 0 :
	// 		When alpha = 0 --> ZSETV
	// 		When alpha = 1 --> ZCOPYV
	// 		When alpha = !( 0 or 1 ) --> ZSCAL2V
	else if ( bli_ceq0( *beta ) )
	{
		bli_zscal2v_zen_int
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
	// 		When alpha = 1 --> ZADDV
	// 		When alpha = !( 0 or 1 ) --> ZAXPYV
	else if ( bli_ceq1( *beta ) )
	{
		if( bli_ceq1( *alpha ) )
		{
			bli_zaddv_zen_int
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
			bli_zaxpyv_zen_int5
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

	dim_t i = 0; // iterator

	// Local pointers to x and y vectors
	double*  restrict x0;
	double*  restrict y0;

	// Boolean to check if alpha is 1
	bool is_alpha_one = bli_zeq1( *alpha );

	// Variables to store real and imaginary components of alpha and beta
	double alphaR, alphaI, betaR, betaI;

	// Initializing the local pointers
	x0  = ( double* ) x;
	y0  = ( double* ) y;

	alphaR = alpha->real;
	alphaI = alpha->imag;
	betaR  = beta->real;
	betaI  = beta->imag;

	// In case of unit strides for x and y vectors
	if ( incx == 1 && incy == 1 )
	{
		// Number of double precision elements in a YMM register
		const dim_t  n_elem_per_reg = 4;

		// Scratch registers
		__m256d xv[4];
		__m256d yv[4];
		__m256d iv[4];
		// Vectors to store real and imaginary components of beta
		__m256d betaRv, betaIv;

		// Broadcasting real and imaginary components of beta onto the registers
		betaRv = _mm256_broadcast_sd( &betaR );
		betaIv = _mm256_broadcast_sd( &betaI );

		if( is_alpha_one )
		{
			__m256d reg_one = _mm256_set1_pd(1.0);
			iv[0] = _mm256_setzero_pd();

			// Converting reg_one to have {1.0, -1.0, 1.0, -1.0}
			// This is needed in case we have t0 conjugate X vector
			if( bli_is_conj( conjx ) )
			{
				reg_one = _mm256_fmsubadd_pd( reg_one, iv[0], reg_one );
			}
			// Processing 8 elements per loop, 8 FMAs
			for ( i = 0; ( i + 7 ) < n; i += 8 )
			{
				// Load the y vector, 8 elements in total
				// yv =  yR1  yI1  yR2  yI2
				yv[0] = _mm256_loadu_pd( y0 );
				yv[1] = _mm256_loadu_pd( y0 + 1 * n_elem_per_reg );
				yv[2] = _mm256_loadu_pd( y0 + 2 * n_elem_per_reg );
				yv[3] = _mm256_loadu_pd( y0 + 3 * n_elem_per_reg );

				// Load the x vector, 8 elements in total
				// xv = xR1  xI1  xR2  xI2
				xv[0] = _mm256_loadu_pd( x0 );
				xv[1] = _mm256_loadu_pd( x0 + 1 * n_elem_per_reg );
				xv[2] = _mm256_loadu_pd( x0 + 2 * n_elem_per_reg );
				xv[3] = _mm256_loadu_pd( x0 + 3 * n_elem_per_reg );

				// Permute the vectors from y for the required compute
				// iv =  yI1  yR1  yI2  yR2
				iv[0] = _mm256_permute_pd( yv[0], 0x5 );
				iv[1] = _mm256_permute_pd( yv[1], 0x5 );
				iv[2] = _mm256_permute_pd( yv[2], 0x5 );
				iv[3] = _mm256_permute_pd( yv[3], 0x5 );

				// Scale the permuted vectors with imaginary component of beta
				// iv = betaIv * yv
				//    = yI1.bI, yR1.bI, yI2.bI, yR2.bI, ...
				iv[0] = _mm256_mul_pd( betaIv, iv[0] );
				iv[1] = _mm256_mul_pd( betaIv, iv[1] );
				iv[2] = _mm256_mul_pd( betaIv, iv[2] );
				iv[3] = _mm256_mul_pd( betaIv, iv[3] );

				// Using fmaddsub to scale with real component of beta
				// and sub/add to iv
				// yv = betaRv * yv -/+ iv
				//    = yR1.bR - yI1.bI, yI1.bR + yR1.bI, ...
				yv[0] = _mm256_fmaddsub_pd( betaRv, yv[0], iv[0] );
				yv[1] = _mm256_fmaddsub_pd( betaRv, yv[1], iv[1] );
				yv[2] = _mm256_fmaddsub_pd( betaRv, yv[2], iv[2] );
				yv[3] = _mm256_fmaddsub_pd( betaRv, yv[3], iv[3] );

				// Adding X conjugate to it
				yv[0] = _mm256_fmadd_pd( reg_one, xv[0], yv[0] );
				yv[1] = _mm256_fmadd_pd( reg_one, xv[1], yv[1] );
				yv[2] = _mm256_fmadd_pd( reg_one, xv[2], yv[2] );
				yv[3] = _mm256_fmadd_pd( reg_one, xv[3], yv[3] );

				// Storing the result to memory
				_mm256_storeu_pd( ( y0 ), yv[0] );
				_mm256_storeu_pd( ( y0 + 1 * n_elem_per_reg ), yv[1] );
				_mm256_storeu_pd( ( y0 + 2 * n_elem_per_reg ), yv[2] );
				_mm256_storeu_pd( ( y0 + 3 * n_elem_per_reg ), yv[3] );

				// Adjusting the pointers for the next iteration
				y0 += 4 * n_elem_per_reg;
				x0 += 4 * n_elem_per_reg;
			}

			// Processing 6 elements per loop, 6 FMAs
			for ( ; ( i + 5 ) < n; i += 6 )
			{
				// Load the y vector, 6 elements in total
				// yv =  yR1  yI1  yR2  yI2
				yv[0] = _mm256_loadu_pd( y0 );
				yv[1] = _mm256_loadu_pd( y0 + 1 * n_elem_per_reg );
				yv[2] = _mm256_loadu_pd( y0 + 2 * n_elem_per_reg );

				// Load the x vector, 6 elements in total
				// xv = xR1  xI1  xR2  xI2
				xv[0] = _mm256_loadu_pd( x0 );
				xv[1] = _mm256_loadu_pd( x0 + 1 * n_elem_per_reg );
				xv[2] = _mm256_loadu_pd( x0 + 2 * n_elem_per_reg );

				// Permute the vectors from y for the required compute
				// iv =  yI1  yR1  yI2  yR2
				iv[0] = _mm256_permute_pd( yv[0], 0x5 );
				iv[1] = _mm256_permute_pd( yv[1], 0x5 );
				iv[2] = _mm256_permute_pd( yv[2], 0x5 );

				// Scale the permuted vectors with imaginary component of beta
				// iv = betaIv * yv
				//    = yI1.bI, yR1.bI, yI2.bI, yR2.bI, ...
				iv[0] = _mm256_mul_pd( betaIv, iv[0] );
				iv[1] = _mm256_mul_pd( betaIv, iv[1] );
				iv[2] = _mm256_mul_pd( betaIv, iv[2] );

				// Using fmaddsub to scale with real component of beta
				// and sub/add to iv
				// yv = betaRv * yv -/+ iv
				//    = yR1.bR - yI1.bI, yI1.bR + yR1.bI, ...
				yv[0] = _mm256_fmaddsub_pd( betaRv, yv[0], iv[0] );
				yv[1] = _mm256_fmaddsub_pd( betaRv, yv[1], iv[1] );
				yv[2] = _mm256_fmaddsub_pd( betaRv, yv[2], iv[2] );

				// Adding X conjugate to it
				yv[0] = _mm256_fmadd_pd( reg_one, xv[0], yv[0] );
				yv[1] = _mm256_fmadd_pd( reg_one, xv[1], yv[1] );
				yv[2] = _mm256_fmadd_pd( reg_one, xv[2], yv[2] );

				// Storing the result to memory
				_mm256_storeu_pd( ( y0 ), yv[0] );
				_mm256_storeu_pd( ( y0 + 1 * n_elem_per_reg ), yv[1] );
				_mm256_storeu_pd( ( y0 + 2 * n_elem_per_reg ), yv[2] );

				// Adjusting the pointers for the next iteration
				y0 += 3 * n_elem_per_reg;
				x0 += 3 * n_elem_per_reg;
			}

			// Processing 4 elements per loop, 4 FMAs
			for ( ; ( i + 3 ) < n; i += 4 )
			{
				// Load the y vector, 4 elements in total
				// yv =  yR1  yI1  yR2  yI2
				yv[0] = _mm256_loadu_pd( y0 );
				yv[1] = _mm256_loadu_pd( y0 + 1 * n_elem_per_reg );

				// Load the x vector, 4 elements in total
				// xv = xR1  xI1  xR2  xI2
				xv[0] = _mm256_loadu_pd( x0 );
				xv[1] = _mm256_loadu_pd( x0 + 1 * n_elem_per_reg );

				// Permute the vectors from y for the required compute
				// iv =  yI1  yR1  yI2  yR2
				iv[0] = _mm256_permute_pd( yv[0], 0x5 );
				iv[1] = _mm256_permute_pd( yv[1], 0x5 );

				// Scale the permuted vectors with imaginary component of beta
				// iv = betaIv * yv
				//    = yI1.bI, yR1.bI, yI2.bI, yR2.bI, ...
				iv[0] = _mm256_mul_pd( betaIv, iv[0] );
				iv[1] = _mm256_mul_pd( betaIv, iv[1] );

				// Using fmaddsub to scale with real component of beta
				// and sub/add to iv
				// yv = betaRv * yv -/+ iv
				//    = yR1.bR - yI1.bI, yI1.bR + yR1.bI, ...
				yv[0] = _mm256_fmaddsub_pd( betaRv, yv[0], iv[0] );
				yv[1] = _mm256_fmaddsub_pd( betaRv, yv[1], iv[1] );

				// Adding X conjugate to it
				yv[0] = _mm256_fmadd_pd( reg_one, xv[0], yv[0] );
				yv[1] = _mm256_fmadd_pd( reg_one, xv[1], yv[1] );

				// Storing the result to memory
				_mm256_storeu_pd( ( y0 ), yv[0] );
				_mm256_storeu_pd( ( y0 + 1 * n_elem_per_reg ), yv[1] );

				// Adjusting the pointers for the next iteration
				y0 += 2 * n_elem_per_reg;
				x0 += 2 * n_elem_per_reg;
			}

			// Processing 2 elements per loop, 2 FMAs
			for ( ; ( i + 1 ) < n; i += 2 )
			{
				// Load the y vector, 2 elements in total
				// yv =  yR1  yI1  yR2  yI2
				yv[0] = _mm256_loadu_pd( y0 );

				// Load the x vector, 2 elements in total
				// xv = xR1  xI1  xR2  xI2
				xv[0] = _mm256_loadu_pd( x0 );

				// Permute the vectors from y for the required compute
				// iv =  yI1  yR1  yI2  yR2
				iv[0] = _mm256_permute_pd( yv[0], 0x5 );

				// Scale the permuted vectors with imaginary component of beta
				// iv = betaIv * yv
				//    = yI1.bI, yR1.bI, yI2.bI, yR2.bI, ...
				iv[0] = _mm256_mul_pd( betaIv, iv[0] );

				// Using fmaddsub to scale with real component of beta
				// and sub/add to iv
				// yv = betaRv * yv -/+ iv
				//    = yR1.bR - yI1.bI, yI1.bR + yR1.bI, ...
				yv[0] = _mm256_fmaddsub_pd( betaRv, yv[0], iv[0] );

				// Adding X conjugate to it
				yv[0] = _mm256_fmadd_pd( reg_one, xv[0], yv[0] );

				// Storing the result to memory
				_mm256_storeu_pd( ( y0 ), yv[0] );

				// Adjusting the pointers for the next iteration
				y0 += 1 * n_elem_per_reg;
				x0 += 1 * n_elem_per_reg;
			}
		}
		else
		{
			// Scratch registers for storing real and imaginary components of alpha
			__m256d alphaRv, alphaIv;

			iv[0] = _mm256_setzero_pd();

			alphaRv = _mm256_broadcast_sd( &alphaR );
			alphaIv = _mm256_broadcast_sd( &alphaI );

			// The changes on alphaRv and alphaIv are as follows :
			// If conjugate is required:
			//		alphaRv =  aR  -aR  aR  -aR
			// Else :
			//		alphaIv =  -aI  aI  -aI  aI
			if( bli_is_conj( conjx ) )
			{
				alphaRv = _mm256_fmsubadd_pd( iv[0], iv[0], alphaRv );
			}
			else
			{
				alphaIv = _mm256_addsub_pd( iv[0], alphaIv );
			}

			// Processing 8 elements per loop, 8 FMAs
			for ( i = 0; ( i + 7 ) < n; i += 8 )
			{
				// Load the y vector, 8 elements in total
				// yv =  yR1  yI1  yR2  yI2
				yv[0] = _mm256_loadu_pd( y0 );
				yv[1] = _mm256_loadu_pd( y0 + 1 * n_elem_per_reg );
				yv[2] = _mm256_loadu_pd( y0 + 2 * n_elem_per_reg );
				yv[3] = _mm256_loadu_pd( y0 + 3 * n_elem_per_reg );

				// Load the x vector, 8 elements in total
				// xv = xR1  xI1  xR2  xI2
				xv[0] = _mm256_loadu_pd( x0 );
				xv[1] = _mm256_loadu_pd( x0 + 1 * n_elem_per_reg );
				xv[2] = _mm256_loadu_pd( x0 + 2 * n_elem_per_reg );
				xv[3] = _mm256_loadu_pd( x0 + 3 * n_elem_per_reg );

				// Permute the vectors from y for the required compute
				// iv =  yI1  yR1  yI2  yR2
				iv[0] = _mm256_permute_pd( yv[0], 0x5 );
				iv[1] = _mm256_permute_pd( yv[1], 0x5 );
				iv[2] = _mm256_permute_pd( yv[2], 0x5 );
				iv[3] = _mm256_permute_pd( yv[3], 0x5 );

				// Scale the permuted vectors with imaginary component of beta
				// iv = betaIv * yv
				//    = yI1.bI, yR1.bI, yI2.bI, yR2.bI, ...
				iv[0] = _mm256_mul_pd( betaIv, iv[0] );
				iv[1] = _mm256_mul_pd( betaIv, iv[1] );
				iv[2] = _mm256_mul_pd( betaIv, iv[2] );
				iv[3] = _mm256_mul_pd( betaIv, iv[3] );

				// Using fmaddsub to scale with real component of beta
				// and sub/add to iv
				// yv = betaRv * yv -/+ iv
				//    = yR1.bR - yI1.bI, yI1.bR + yR1.bI, ...
				yv[0] = _mm256_fmaddsub_pd( betaRv, yv[0], iv[0] );
				yv[1] = _mm256_fmaddsub_pd( betaRv, yv[1], iv[1] );
				yv[2] = _mm256_fmaddsub_pd( betaRv, yv[2], iv[2] );
				yv[3] = _mm256_fmaddsub_pd( betaRv, yv[3], iv[3] );

				// Permute the loaded vectors from x for the required compute
				// xv' =  xI1  xR1  xI2  xR2
				iv[0] = _mm256_permute_pd( xv[0], 0x5 );
				iv[1] = _mm256_permute_pd( xv[1], 0x5 );
				iv[2] = _mm256_permute_pd( xv[2], 0x5 );
				iv[3] = _mm256_permute_pd( xv[3], 0x5 );

				// yv = alphaRv * xv + yv
				//    = yR1.bR - yR1.bI + xR1.aR, yI1.bR + yI1.bI + xI1.aR, ...
				yv[0] = _mm256_fmadd_pd( alphaRv, xv[0], yv[0] );
				yv[1] = _mm256_fmadd_pd( alphaRv, xv[1], yv[1] );
				yv[2] = _mm256_fmadd_pd( alphaRv, xv[2], yv[2] );
				yv[3] = _mm256_fmadd_pd( alphaRv, xv[3], yv[3] );

				// yv = alphaIv * iv + yv
				//    = yR1.bR - yR1.bI - xI1.aI, yI1.bR + yI1.bI + xR1.aI, ...
				yv[0] = _mm256_fmadd_pd( alphaIv, iv[0], yv[0] );
				yv[1] = _mm256_fmadd_pd( alphaIv, iv[1], yv[1] );
				yv[2] = _mm256_fmadd_pd( alphaIv, iv[2], yv[2] );
				yv[3] = _mm256_fmadd_pd( alphaIv, iv[3], yv[3] );

				// Storing the result to memory
				_mm256_storeu_pd( ( y0 ), yv[0] );
				_mm256_storeu_pd( ( y0 + 1 * n_elem_per_reg ), yv[1] );
				_mm256_storeu_pd( ( y0 + 2 * n_elem_per_reg ), yv[2] );
				_mm256_storeu_pd( ( y0 + 3 * n_elem_per_reg ), yv[3] );

				// Adjusting the pointers for the next iteration
				y0 += 4 * n_elem_per_reg;
				x0 += 4 * n_elem_per_reg;
			}

			// Processing 6 elements per loop, 6 FMAs
			for ( ; ( i + 5 ) < n; i += 6 )
			{
				// Load the y vector, 6 elements in total
				// yv =  yR1  yI1  yR2  yI2
				yv[0] = _mm256_loadu_pd( y0 );
				yv[1] = _mm256_loadu_pd( y0 + 1 * n_elem_per_reg );
				yv[2] = _mm256_loadu_pd( y0 + 2 * n_elem_per_reg );

				// Load the x vector, 6 elements in total
				// xv = xR1  xI1  xR2  xI2
				xv[0] = _mm256_loadu_pd( x0 );
				xv[1] = _mm256_loadu_pd( x0 + 1 * n_elem_per_reg );
				xv[2] = _mm256_loadu_pd( x0 + 2 * n_elem_per_reg );

				// Permute the vectors from y for the required compute
				// iv =  yI1  yR1  yI2  yR2
				iv[0] = _mm256_permute_pd( yv[0], 0x5 );
				iv[1] = _mm256_permute_pd( yv[1], 0x5 );
				iv[2] = _mm256_permute_pd( yv[2], 0x5 );

				// Scale the permuted vectors with imaginary component of beta
				// iv = betaIv * yv
				//    = yI1.bI, yR1.bI, yI2.bI, yR2.bI, ...`
				iv[0] = _mm256_mul_pd( betaIv, iv[0] );
				iv[1] = _mm256_mul_pd( betaIv, iv[1] );
				iv[2] = _mm256_mul_pd( betaIv, iv[2] );

				// Using fmaddsub to scale with real component of beta
				// and sub/add to iv
				// yv = betaRv * yv -/+ iv
				//    = yR1.bR - yI1.bI, yI1.bR + yR1.bI, ...
				yv[0] = _mm256_fmaddsub_pd( betaRv, yv[0], iv[0] );
				yv[1] = _mm256_fmaddsub_pd( betaRv, yv[1], iv[1] );
				yv[2] = _mm256_fmaddsub_pd( betaRv, yv[2], iv[2] );

				// Permute the loaded vectors from x for the required compute
				// xv' =  xI1  xR1  xI2  xR2
				iv[0] = _mm256_permute_pd( xv[0], 0x5 );
				iv[1] = _mm256_permute_pd( xv[1], 0x5 );
				iv[2] = _mm256_permute_pd( xv[2], 0x5 );

				// yv = alphaRv * xv + yv
				//    = yR1.bR - yR1.bI + xR1.aR, yI1.bR + yI1.bI + xI1.aR, ...
				yv[0] = _mm256_fmadd_pd( alphaRv, xv[0], yv[0] );
				yv[1] = _mm256_fmadd_pd( alphaRv, xv[1], yv[1] );
				yv[2] = _mm256_fmadd_pd( alphaRv, xv[2], yv[2] );

				// yv = alphaIv * iv + yv
				//    = yR1.bR - yR1.bI - xI1.aI, yI1.bR + yI1.bI + xR1.aI, ...
				yv[0] = _mm256_fmadd_pd( alphaIv, iv[0], yv[0] );
				yv[1] = _mm256_fmadd_pd( alphaIv, iv[1], yv[1] );
				yv[2] = _mm256_fmadd_pd( alphaIv, iv[2], yv[2] );

				// Storing the result to memory
				_mm256_storeu_pd( ( y0 ), yv[0] );
				_mm256_storeu_pd( ( y0 + 1 * n_elem_per_reg ), yv[1] );
				_mm256_storeu_pd( ( y0 + 2 * n_elem_per_reg ), yv[2] );

				// Adjusting the pointers for the next iteration
				y0 += 3 * n_elem_per_reg;
				x0 += 3 * n_elem_per_reg;
			}

			// Processing 4 elements per loop, 4 FMAs
			for ( ; ( i + 3 ) < n; i += 4 )
			{
				// Load the y vector, 4 elements in total
				// yv =  yR1  yI1  yR2  yI2
				yv[0] = _mm256_loadu_pd( y0 );
				yv[1] = _mm256_loadu_pd( y0 + 1 * n_elem_per_reg );

				// Load the x vector, 4 elements in total
				// xv = xR1  xI1  xR2  xI2
				xv[0] = _mm256_loadu_pd( x0 );
				xv[1] = _mm256_loadu_pd( x0 + 1 * n_elem_per_reg );

				// Permute the vectors from y for the required compute
				// iv =  yI1  yR1  yI2  yR2
				iv[0] = _mm256_permute_pd( yv[0], 0x5 );
				iv[1] = _mm256_permute_pd( yv[1], 0x5 );

				// Scale the permuted vectors with imaginary component of beta
				// iv = betaIv * yv
				//    = yI1.bI, yR1.bI, yI2.bI, yR2.bI, ...
				iv[0] = _mm256_mul_pd( betaIv, iv[0] );
				iv[1] = _mm256_mul_pd( betaIv, iv[1] );

				// Using fmaddsub to scale with real component of beta
				// and sub/add to iv
				// yv = betaRv * yv -/+ iv
				//    = yR1.bR - yI1.bI, yI1.bR + yR1.bI, ...
				yv[0] = _mm256_fmaddsub_pd( betaRv, yv[0], iv[0] );
				yv[1] = _mm256_fmaddsub_pd( betaRv, yv[1], iv[1] );

				// Permute the loaded vectors from x for the required compute
				// xv' =  xI1  xR1  xI2  xR2
				iv[0] = _mm256_permute_pd( xv[0], 0x5 );
				iv[1] = _mm256_permute_pd( xv[1], 0x5 );

				// yv = alphaRv * xv + yv
				//    = yR1.bR - yR1.bI + xR1.aR, yI1.bR + yI1.bI + xI1.aR, ...
				yv[0] = _mm256_fmadd_pd( alphaRv, xv[0], yv[0] );
				yv[1] = _mm256_fmadd_pd( alphaRv, xv[1], yv[1] );

				// yv = alphaIv * iv + yv
				//    = yR1.bR - yR1.bI - xI1.aI, yI1.bR + yI1.bI + xR1.aI, ...
				yv[0] = _mm256_fmadd_pd( alphaIv, iv[0], yv[0] );
				yv[1] = _mm256_fmadd_pd( alphaIv, iv[1], yv[1] );

				// Storing the result to memory
				_mm256_storeu_pd( ( y0 ), yv[0] );
				_mm256_storeu_pd( ( y0 + 1 * n_elem_per_reg ), yv[1] );

				// Adjusting the pointers for the next iteration
				y0 += 2 * n_elem_per_reg;
				x0 += 2 * n_elem_per_reg;
			}

			// Processing 2 elements per loop, 2 FMAs
			for ( ; ( i + 1 ) < n; i += 2 )
			{
				// Load the y vector, 2 elements in total
				// yv =  yR1  yI1  yR2  yI2
				yv[0] = _mm256_loadu_pd( y0 );

				// Load the x vector, 2 elements in total
				// xv = xR1  xI1  xR2  xI2
				xv[0] = _mm256_loadu_pd( x0 );

				// Permute the vectors from y for the required compute
				// iv =  yI1  yR1  yI2  yR2
				iv[0] = _mm256_permute_pd( yv[0], 0x5 );

				// Scale the permuted vectors with imaginary component of beta
				// iv = betaIv * yv
				//    = yI1.bI, yR1.bI, yI2.bI, yR2.bI, ...
				iv[0] = _mm256_mul_pd( betaIv, iv[0] );

				// Using fmaddsub to scale with real component of beta
				// and sub/add to iv
				// yv = betaRv * yv -/+ iv
				//    = yR1.bR - yI1.bI, yI1.bR + yR1.bI, ...
				yv[0] = _mm256_fmaddsub_pd( betaRv, yv[0], iv[0] );

				// Permute the loaded vectors from x for the required compute
				// xv' =  xI1  xR1  xI2  xR2
				iv[0] = _mm256_permute_pd( xv[0], 0x5 );

				// yv = alphaRv * xv + yv
				//    = yR1.bR - yR1.bI + xR1.aR, yI1.bR + yI1.bI + xI1.aR, ...
				yv[0] = _mm256_fmadd_pd( alphaRv, xv[0], yv[0] );

				// yv = alphaIv * iv + yv
				//    = yR1.bR - yR1.bI - xI1.aI, yI1.bR + yI1.bI + xR1.aI, ...
				yv[0] = _mm256_fmadd_pd( alphaIv, iv[0], yv[0] );

				// Storing the result to memory
				_mm256_storeu_pd( ( y0 ), yv[0] );

				// Adjusting the pointers for the next iteration
				y0 += 1 * n_elem_per_reg;
				x0 += 1 * n_elem_per_reg;
			}
		}

		// Issue vzeroupper instruction to clear upper lanes of ymm registers.
		// This avoids a performance penalty caused by false dependencies when
		// transitioning from AVX to SSE instructions.
		_mm256_zeroupper();

	}

	// Scratch registers to be used in case of non-unit strides or fringe case of 1.
	__m128d x_elem, y_elem, x_perm, y_perm;
	__m128d betaRv, betaIv;

	// Broadcasting real and imag parts of beta onto 128 bit registers
	betaRv = _mm_set1_pd( betaR );
	betaIv = _mm_set1_pd( betaI );

	// Changing betaIv to { -bI  bI } for the compute
	x_elem = _mm_setzero_pd();
	betaIv = _mm_addsub_pd( x_elem, betaIv );

	if ( is_alpha_one )
	{
		__m128d reg_one = _mm_set1_pd(1.0);

		if( bli_is_conj( conjx ) )
		{
			reg_one = _mm_addsub_pd( x_elem, reg_one );
			reg_one = _mm_permute_pd( reg_one, 0x1 );
		}

		// Iterate over y, one element at a time
		for ( ; i < n; i += 1 )
		{
			// Load an element from x and y
			// y_elem =  yR1  yI1
			// x_elem =  xR1  xI1
			y_elem = _mm_loadu_pd( y0 );
			x_elem = _mm_loadu_pd( x0 );

			// Permute y in accordance to its compute
			// y_perm =  yI1  yR1
			y_perm = _mm_permute_pd( y_elem, 0x1 );

			// Scale y_perm by the imaginary
			// component of beta
			// y_perm =  -yI1.bI, yR1.bI
			y_perm = _mm_mul_pd( betaIv, y_perm );

			// Use fmadd to scale with real component of
			// beta and add with intermediate result
			// y_elem =  yR1.bR - yI1.bI, yI1.bR + yR1.bI
			y_elem = _mm_fmadd_pd( betaRv, y_elem, y_perm );

			y_elem = _mm_fmadd_pd( reg_one, x_elem, y_elem );

			// Storing the result to memory
			_mm_storeu_pd( y0, y_elem );

			// Adjusting the pointer for the next iteration
			x0 += incx * 2;
			y0 += incy * 2;
		}
	}
	else
	{
		// Scratch registers to store real and imaginary components
		// of alpha onto XMM registers
		__m128d alphaRv, alphaIv;

		// Broadcasting real and imaginary components of alpha
		x_elem = _mm_setzero_pd();
		alphaRv = _mm_loaddup_pd( &alphaR );
		alphaIv = _mm_loaddup_pd( &alphaI );

		// The changes on alphaRv and alphaIv are as follows :
		// If conjugate is required:
		//		alphaRv =  aR  -aR
		// Else :
		//		alphaIv =  -aI  aI
		if( bli_is_conj( conjx ) )
		{
			alphaRv = _mm_addsub_pd( x_elem, alphaRv );
			alphaRv = _mm_permute_pd( alphaRv, 0x1 );
		}
		else
		{
			alphaIv = _mm_addsub_pd( x_elem, alphaIv );
		}

		// Iterating over x and y vectors, on element at a time
		for ( ; i < n; i += 1 )
		{
			// Load an element from x and y
			// y_elem =  yR1  yI1
			// x_elem =  xR1  xI1
			y_elem = _mm_loadu_pd( y0 );
			x_elem = _mm_loadu_pd( x0 );

			// Permute y in accordance to its compute
			// y_perm =  yI1  yR1
			// x_perm =  xR1  xI1
			y_perm = _mm_permute_pd( y_elem, 0x1 );
			x_perm = _mm_permute_pd( x_elem, 0x1 );

			// Scale y_perm and x_perm by the imaginary
			// component of beta and alpha
			// y_perm =  -yI1.bI, yR1.bI
			// x_perm =  -xI1.aI, xR1.aI
			y_perm = _mm_mul_pd( betaIv, y_perm );
			x_perm = _mm_mul_pd( alphaIv, x_perm );

			// Use fmadd to scale with y_elem with
			// real component of beta and add with
			// intermediate result. Similarly do
			// for x_elem.
			// y_elem =  yR1.bR - yI1.bI, yI1.bR + yR1.bI
			// x_elem =  xR1.aR - xI1.aI, xI1.aR + xR1.aI
			y_elem = _mm_fmadd_pd( betaRv, y_elem, y_perm );
			x_elem = _mm_fmadd_pd( alphaRv, x_elem, x_perm );

			// Add the computed x and y vectors, store on y.
			y_elem = _mm_add_pd( y_elem, x_elem );

			// Storing the result to memory
			_mm_storeu_pd( y0, y_elem );

			// Adjusting the pointer for the next iteration
			x0 += incx * 2;
			y0 += incy * 2;
		}
	}

	AOCL_DTL_TRACE_EXIT(AOCL_DTL_LEVEL_TRACE_4)
}
