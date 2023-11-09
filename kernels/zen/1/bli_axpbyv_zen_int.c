/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2022 - 2023, Advanced Micro Devices, Inc. All rights reserved.

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
 * 		alpha & beta are scalers.
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
	const dim_t     n_elem_per_reg  = 8;    // number of elements per register
	const dim_t     n_iter_unroll   = 4;    // num of registers per iteration

	dim_t           i;          // iterator

	float* restrict x0;
	float* restrict y0;

	v8sf_t          alphav;
	v8sf_t          betav;
	v8sf_t          y0v, y1v, y2v, y3v;

	/* if the vector dimension is zero, or if alpha & beta are zero,
	   return early. */
	if ( bli_zero_dim1( n ) ||
		 ( PASTEMAC( s, eq0 )( *alpha ) && PASTEMAC( s, eq0 )( *beta ) ) )
		 return;

	// initialize local pointers
	x0 = x;
	y0 = y;

	if ( incx == 1 && incy == 1 )
	{
		// broadcast alpha & beta to all elements of respective vector registers
		alphav.v = _mm256_broadcast_ss( alpha );
		betav.v  = _mm256_broadcast_ss( beta );

		// unrolling and vectorizing
		for ( i = 0; ( i + 31 ) < n; i += 32 )
		{
			// loading input y
			y0v.v = _mm256_loadu_ps( y0 + 0*n_elem_per_reg );
			y1v.v = _mm256_loadu_ps( y0 + 1*n_elem_per_reg );
			y2v.v = _mm256_loadu_ps( y0 + 2*n_elem_per_reg );
			y3v.v = _mm256_loadu_ps( y0 + 3*n_elem_per_reg );

			// y' := y := beta * y
			y0v.v = _mm256_mul_ps( betav.v, y0v.v );
			y1v.v = _mm256_mul_ps( betav.v, y1v.v );
			y2v.v = _mm256_mul_ps( betav.v, y2v.v );
			y3v.v = _mm256_mul_ps( betav.v, y3v.v );

			// y := y' + alpha * x
			y0v.v = _mm256_fmadd_ps
					(
					  alphav.v,
					  _mm256_loadu_ps( x0 + 0*n_elem_per_reg ),
					  y0v.v
					);
			y1v.v = _mm256_fmadd_ps
					(
					  alphav.v,
					  _mm256_loadu_ps( x0 + 1*n_elem_per_reg ),
					  y1v.v
					);
			y2v.v = _mm256_fmadd_ps
					(
					  alphav.v,
					  _mm256_loadu_ps( x0 + 2*n_elem_per_reg ),
					  y2v.v
					);
			y3v.v = _mm256_fmadd_ps
					(
					  alphav.v,
					  _mm256_loadu_ps( x0 + 3*n_elem_per_reg ),
					  y3v.v
					);

			// storing the output
			_mm256_storeu_ps( ( y0 + 0*n_elem_per_reg ), y0v.v );
			_mm256_storeu_ps( ( y0 + 1*n_elem_per_reg ), y1v.v );
			_mm256_storeu_ps( ( y0 + 2*n_elem_per_reg ), y2v.v );
			_mm256_storeu_ps( ( y0 + 3*n_elem_per_reg ), y3v.v );

			x0 += n_elem_per_reg * n_iter_unroll;
			y0 += n_elem_per_reg * n_iter_unroll;
		}
		
		// Issue vzeroupper instruction to clear upper lanes of ymm registers.
		// This avoids a performance penalty caused by false dependencies when
		// transitioning from AVX to SSE instructions (which may occur as soon
		// as the n_left cleanup loop below if BLIS is compiled with
		// -mfpmath=sse).
		_mm256_zeroupper();

		// if there are leftover iterations, perform them with scaler code
		for ( ; i < n; ++i )
		{
			*y0 = ( (*alpha) * (*x0) ) + ( (*beta) * (*y0) );

			x0 += incx;
			y0 += incy;
		}
	}
	else
	{
		// for non-unit increments, use scaler code
		for ( i = 0; i < n; ++i )
		{
			*y0 = ( (*alpha) * (*x0) ) + ( (*beta) * (*y0) );

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
 * 		alpha & beta are scalers.
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
	const dim_t     n_elem_per_reg  = 4;    // number of elements per register
	const dim_t     n_iter_unroll   = 4;    // number of registers per iteration

	dim_t           i;          // iterator

	double* restrict x0;
	double* restrict y0;

	v4df_t          alphav;
	v4df_t          betav;
	v4df_t          y0v, y1v, y2v, y3v;

	/* if the vector dimension is zero, or if alpha & beta are zero,
	   return early. */
	if ( bli_zero_dim1( n ) ||
		 ( PASTEMAC( s, eq0 )( *alpha ) && PASTEMAC( s, eq0 )( *beta ) ) )
	{
		AOCL_DTL_TRACE_EXIT(AOCL_DTL_LEVEL_TRACE_4)
		return;
	}

	// initialize local pointers
	x0 = x;
	y0 = y;
	
	if ( incx == 1 && incy == 1 )
	{
		// broadcast alpha & beta to all elements of respective vector registers
		alphav.v = _mm256_broadcast_sd( alpha );
		betav.v  = _mm256_broadcast_sd( beta );

		// unrolling and vectorizing
		for ( i = 0; ( i + 15 ) < n; i += 16 )
		{
			// loading input y
			y0v.v = _mm256_loadu_pd( y0 + 0*n_elem_per_reg );
			y1v.v = _mm256_loadu_pd( y0 + 1*n_elem_per_reg );
			y2v.v = _mm256_loadu_pd( y0 + 2*n_elem_per_reg );
			y3v.v = _mm256_loadu_pd( y0 + 3*n_elem_per_reg );

			// y' := y := beta * y
			y0v.v = _mm256_mul_pd( betav.v, y0v.v );
			y1v.v = _mm256_mul_pd( betav.v, y1v.v );
			y2v.v = _mm256_mul_pd( betav.v, y2v.v );
			y3v.v = _mm256_mul_pd( betav.v, y3v.v );
			
			// y := y' + alpha * x
			//   := beta * y + alpha * x
			y0v.v = _mm256_fmadd_pd
					(
					  alphav.v,
					  _mm256_loadu_pd( x0 + 0*n_elem_per_reg ),
					  y0v.v
					);
			y1v.v = _mm256_fmadd_pd
					(
					  alphav.v,
					  _mm256_loadu_pd( x0 + 1*n_elem_per_reg ),
					  y1v.v
					);
			y2v.v = _mm256_fmadd_pd
					(
					  alphav.v,
					  _mm256_loadu_pd( x0 + 2*n_elem_per_reg ),
					  y2v.v
					);
			y3v.v = _mm256_fmadd_pd
					(
					  alphav.v,
					  _mm256_loadu_pd( x0 + 3*n_elem_per_reg ),
					  y3v.v
					);

			// storing the output
			_mm256_storeu_pd( ( y0 + 0*n_elem_per_reg ), y0v.v );
			_mm256_storeu_pd( ( y0 + 1*n_elem_per_reg ), y1v.v );
			_mm256_storeu_pd( ( y0 + 2*n_elem_per_reg ), y2v.v );
			_mm256_storeu_pd( ( y0 + 3*n_elem_per_reg ), y3v.v );

			x0 += n_elem_per_reg * n_iter_unroll;
			y0 += n_elem_per_reg * n_iter_unroll;
		}

		// Issue vzeroupper instruction to clear upper lanes of ymm registers.
		// This avoids a performance penalty caused by false dependencies when
		// transitioning from AVX to SSE instructions (which may occur as soon
		// as the n_left cleanup loop below if BLIS is compiled with
		// -mfpmath=sse).
		_mm256_zeroupper();

		// if there are leftover iterations, perform them with scaler code
		for ( ; i < n; ++i )
		{
			*y0 = ( (*alpha) * (*x0) ) + ( (*beta) * (*y0) );

			x0 += incx;
			y0 += incy;
		}
	}
	else
	{
		// for non-unit increments, use scaler code
		for ( i = 0; i < n; ++i )
		{
			*y0 = ( (*alpha) * (*x0) ) + ( (*beta) * (*y0) );

			x0 += incx;
			y0 += incy;
		}
	}
}

/**
 * caxpbyv kernel performs the axpbyv operation.
 * y := beta * y + alpha * conjx(x)
 * where,
 * 		x & y are simple complex vectors of length n.
 * 		alpha & beta are scalers.
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
	const dim_t      n_elem_per_reg = 8;    // number of elements per register

	dim_t            i;     // iterator

	float*  restrict x0;
	float*  restrict y0;

	float alphaR, alphaI, betaR, betaI;

	__m256 alphaRv;
	__m256 alphaIv;
	__m256 betaRv;
	__m256 betaIv;
	__m256 xv[4];
	__m256 yv[4];
	__m256 iv[4];   // intermediate registers

	conj_t conjx_use = conjx;
	
	/* if the vector dimension is zero, or if alpha & beta are zero,
	   return early. */
	if ( bli_zero_dim1( n ) ||
		 ( PASTEMAC( c, eq0 )( *alpha ) && PASTEMAC( c, eq0 )( *beta ) ) )
	{
		AOCL_DTL_TRACE_EXIT(AOCL_DTL_LEVEL_TRACE_4)
		return;
	}

	// initialize local pointers
	x0     = ( float* ) x;
	y0     = ( float* ) y;

	alphaR = alpha->real;
	alphaI = alpha->imag;
	betaR  = beta->real;
	betaI  = beta->imag;

	if ( incx == 1 && incy == 1 )
	{
		//---------- Scalar algorithm BLIS_NO_CONJUGATE -------------
		// y = beta*y + alpha*x
		// y = ( bR + ibI ) * ( yR + iyI ) + ( aR + iaI ) * ( xR + ixI )
		// y = bR.yR + ibR.yI + ibI.yR - ibIyI + aR.xR + iaR.xI + iaI.xR - aI.xI
		// y =   ( bR.yR - bI.yI + aR.xR - aI.xI ) +
		//	   i ( bR.yI + bI.yR + aR.xI + aI.xR )

		// SIMD Algorithm BLIS_NO_CONJUGATE
		// yv  =  yR1  yI1  yR2  yI2  yR3  yI3  yR4  yI4
		// yv' =  yI1  yR1  yI2  yR2  yI3  yR3  yI4  yR4
		// xv  =  xR1  xI1  xR2  xI2  xR3  xI3  xR4  xI4
		// xv' =  xI1  xR1  xI2  xR2  xI3  xR3  xI4  xR4
		// arv =  aR   aR   aR   aR   aR   aR   aR   aR
		// aiv = -aI   aI  -aI   aI  -aI   aI  -aI   aI
		// brv =  bR   bR   bR   bR   bR   bR   bR   bR
		// biv = -bI   bI  -bI   bI  -bI   bI  -bI   bI
		
		// step 1: iv = brv * iv
		// step 2: shuffle yv -> yv'
		// step 3: FMA yv = biv * yv' + iv
		// step 4: iv = arv * xv
		// step 5: shuffle xv -> xv'
		// step 6: FMA yv = aiv * xv' + iv

		//---------- Scalar algorithm BLIS_CONJUGATE -------------
		// y = beta*y + alpha*conj(x)
		// y = ( bR + ibI ) * ( yR + iyI ) + ( aR + iaI ) * ( xR - ixI )
		// y = bR.yR + ibR.yI + ibI.yR - bI.yI + aR.xR - iaR.xI + iaI.xR + aI.xI
		// y =   ( bR.yR - bI.yI + aR.xR + aI.xI ) +
		//	   i ( bR.yI + bI.yR - aR.xI + aI.xR )

		// SIMD Algorithm BLIS_CONJUGATE
		// yv  =  yR1  yI1  yR2  yI2  yR3  yI3  yR4  yI4
		// yv' =  yI1  yR1  yI2  yR2  yI3  yR3  yI4  yR4
		// xv  =  xR1  xI1  xR2  xI2  xR3  xI3  xR4  xI4
		// xv' =  xI1  xR1  xI2  xR2  xI3  xR3  xI4  xR4
		// arv =  aR  -aR   aR  -aR   aR  -aR   aR  -aR
		// aiv =  aI   aI   aI   aI   aI   aI   aI   aI
		// brv =  bR   bR   bR   bR   bR   bR   bR   bR
		// biv = -bI   bI  -bI   bI  -bI   bI  -bI   bI
		//
		// step 1: iv = brv * iv
		// step 2: shuffle yv -> yv'
		// step 3: FMA yv = biv * yv' + iv
		// step 4: iv = arv * xv
		// step 5: shuffle xv -> xv'
		// step 6: FMA yv = aiv * xv' + iv

		// broadcast alpha & beta to all elements of respective vector registers
		if ( !bli_is_conj( conjx ) )    // If BLIS_NO_CONJUGATE
		{
			// alphaRv =  aR   aR   aR   aR   aR   aR   aR   aR
			// alphaIv = -aI   aI  -aI   aI  -aI   aI  -aI   aI
			// betaRv  =  bR   bR   bR   bR   bR   bR   bR   bR
			// betaIv  = -bI   bI  -bI   bI  -bI   bI  -bI   bI
			alphaRv = _mm256_broadcast_ss( &alphaR );
			alphaIv = _mm256_set_ps
					  (
						alphaI, -alphaI, alphaI, -alphaI,
					    alphaI, -alphaI, alphaI, -alphaI
					  );
			betaRv  = _mm256_broadcast_ss( &betaR );
			betaIv  = _mm256_set_ps
					  (
						betaI, -betaI, betaI, -betaI,
						betaI, -betaI, betaI, -betaI
					  );
		}
		else
		{
			// alphaRv =  aR  -aR   aR  -aR   aR  -aR   aR  -aR
			// alphaIv =  aI   aI   aI   aI   aI   aI   aI   aI
			// betaRv  =  bR   bR   bR   bR   bR   bR   bR   bR
			// betaIv  = -bI   bI  -bI   bI  -bI   bI  -bI   bI
			alphaRv = _mm256_set_ps
					  (
						-alphaR, alphaR, -alphaR, alphaR,
						-alphaR, alphaR, -alphaR, alphaR
					  );
			alphaIv = _mm256_broadcast_ss( &alphaI );
			betaRv  = _mm256_broadcast_ss( &betaR );
			betaIv  = _mm256_set_ps
					  (
						betaI, -betaI, betaI, -betaI,
						betaI, -betaI, betaI, -betaI
					  );
		}

		// Processing 16 elements per loop, 8 FMAs
		for ( i = 0; ( i + 15 ) < n; i += 16 )
		{
			// xv = xR1  xI1  xR2  xI2  xR3  xI3  xR4  xI4
			xv[0] = _mm256_loadu_ps( x0 + 0*n_elem_per_reg );
			xv[1] = _mm256_loadu_ps( x0 + 1*n_elem_per_reg );
			xv[2] = _mm256_loadu_ps( x0 + 2*n_elem_per_reg );
			xv[3] = _mm256_loadu_ps( x0 + 3*n_elem_per_reg );

			// yv  =  yR1  yI1  yR2  yI2  yR3  yI3  yR4  yI4
			yv[0] = _mm256_loadu_ps( y0 + 0*n_elem_per_reg );
			yv[1] = _mm256_loadu_ps( y0 + 1*n_elem_per_reg );
			yv[2] = _mm256_loadu_ps( y0 + 2*n_elem_per_reg );
			yv[3] = _mm256_loadu_ps( y0 + 3*n_elem_per_reg );

			// iv = betaRv * yv
			//    = yR1.bR, yI1.bR, yR2.bR, yI2.bR, ...
			iv[0] = _mm256_mul_ps( betaRv, yv[0] );
			iv[1] = _mm256_mul_ps( betaRv, yv[1] );
			iv[2] = _mm256_mul_ps( betaRv, yv[2] );
			iv[3] = _mm256_mul_ps( betaRv, yv[3] );

			// yv' =  yI1  yR1  yI2  yR2  yI3  yR3  yI4  yR4
			yv[0] = _mm256_permute_ps( yv[0], 0xB1);
			yv[1] = _mm256_permute_ps( yv[1], 0xB1);
			yv[2] = _mm256_permute_ps( yv[2], 0xB1);
			yv[3] = _mm256_permute_ps( yv[3], 0xB1);
			
			// yv = betaIv * yv' + iv
			//    = yR1.bR - yI1.bI, yI1.bR + yR1.bI, ...
			yv[0] = _mm256_fmadd_ps( betaIv, yv[0], iv[0] );
			yv[1] = _mm256_fmadd_ps( betaIv, yv[1], iv[1] );
			yv[2] = _mm256_fmadd_ps( betaIv, yv[2], iv[2] );
			yv[3] = _mm256_fmadd_ps( betaIv, yv[3], iv[3] );

			// iv = alphaRv * xv
			//    = xR1.aR, xI1.aR, xR2.aR, xI2.aR, ...
			iv[0] = _mm256_mul_ps( alphaRv, xv[0] );
			iv[1] = _mm256_mul_ps( alphaRv, xv[1] );
			iv[2] = _mm256_mul_ps( alphaRv, xv[2] );
			iv[3] = _mm256_mul_ps( alphaRv, xv[3] );

			// xv' =  xI1  xR1  xI2  xR2  xI3  xR3  xI4  xR4
			xv[0] = _mm256_permute_ps( xv[0], 0xB1);
			xv[1] = _mm256_permute_ps( xv[1], 0xB1);
			xv[2] = _mm256_permute_ps( xv[2], 0xB1);
			xv[3] = _mm256_permute_ps( xv[3], 0xB1);

			// yv = alphaIv * xv + yv
			//    = yR1.bR - yR1.bI - xR1.aI, yI1.bR + yI1.bI + xI1.aI, ...
			iv[0] = _mm256_fmadd_ps( alphaIv, xv[0], iv[0] );
			iv[1] = _mm256_fmadd_ps( alphaIv, xv[1], iv[1] );
			iv[2] = _mm256_fmadd_ps( alphaIv, xv[2], iv[2] );
			iv[3] = _mm256_fmadd_ps( alphaIv, xv[3], iv[3] );

			yv[0] = _mm256_add_ps( yv[0], iv[0] );
			yv[1] = _mm256_add_ps( yv[1], iv[1] );
			yv[2] = _mm256_add_ps( yv[2], iv[2] );
			yv[3] = _mm256_add_ps( yv[3], iv[3] );

			_mm256_storeu_ps( (y0 + 0*n_elem_per_reg), yv[0] );
			_mm256_storeu_ps( (y0 + 1*n_elem_per_reg), yv[1] );
			_mm256_storeu_ps( (y0 + 2*n_elem_per_reg), yv[2] );
			_mm256_storeu_ps( (y0 + 3*n_elem_per_reg), yv[3] );

			y0 += 4*n_elem_per_reg;
			x0 += 4*n_elem_per_reg;
		}

		// Processing 12 elements per loop, 6 FMAs
		for ( ; ( i + 11 ) < n; i += 12 )
		{
			// xv = xR1  xI1  xR2  xI2  xR3  xI3  xR4  xI4
			xv[0] = _mm256_loadu_ps( x0 + 0*n_elem_per_reg );
			xv[1] = _mm256_loadu_ps( x0 + 1*n_elem_per_reg );
			xv[2] = _mm256_loadu_ps( x0 + 2*n_elem_per_reg );

			// yv  =  yR1  yI1  yR2  yI2  yR3  yI3  yR4  yI4
			yv[0] = _mm256_loadu_ps( y0 + 0*n_elem_per_reg );
			yv[1] = _mm256_loadu_ps( y0 + 1*n_elem_per_reg );
			yv[2] = _mm256_loadu_ps( y0 + 2*n_elem_per_reg );

			// iv = betaRv * yv
			//    = yR1.bR, yI1.bR, yR2.bR, yI2.bR, ...
			iv[0] = _mm256_mul_ps( betaRv, yv[0] );
			iv[1] = _mm256_mul_ps( betaRv, yv[1] );
			iv[2] = _mm256_mul_ps( betaRv, yv[2] );

			// yv' =  yI1  yR1  yI2  yR2  yI3  yR3  yI4  yR4
			yv[0] = _mm256_permute_ps( yv[0], 0xB1);
			yv[1] = _mm256_permute_ps( yv[1], 0xB1);
			yv[2] = _mm256_permute_ps( yv[2], 0xB1);
			
			// yv = betaIv * yv' + iv
			//    = yR1.bR - yI1.bI, yI1.bR + yR1.bI, ...
			yv[0] = _mm256_fmadd_ps( betaIv, yv[0], iv[0] );
			yv[1] = _mm256_fmadd_ps( betaIv, yv[1], iv[1] );
			yv[2] = _mm256_fmadd_ps( betaIv, yv[2], iv[2] );

			// iv = alphaRv * xv
			//    = xR1.aR, xI1.aR, xR2.aR, xI2.aR, ...
			iv[0] = _mm256_mul_ps( alphaRv, xv[0] );
			iv[1] = _mm256_mul_ps( alphaRv, xv[1] );
			iv[2] = _mm256_mul_ps( alphaRv, xv[2] );

			// xv' =  xI1  xR1  xI2  xR2  xI3  xR3  xI4  xR4
			xv[0] = _mm256_permute_ps( xv[0], 0xB1);
			xv[1] = _mm256_permute_ps( xv[1], 0xB1);
			xv[2] = _mm256_permute_ps( xv[2], 0xB1);

			// yv = alphaIv * xv + yv
			//    = yR1.bR - yR1.bI - xR1.aI, yI1.bR + yI1.bI + xI1.aI, ...
			iv[0] = _mm256_fmadd_ps( alphaIv, xv[0], iv[0] );
			iv[1] = _mm256_fmadd_ps( alphaIv, xv[1], iv[1] );
			iv[2] = _mm256_fmadd_ps( alphaIv, xv[2], iv[2] );

			yv[0] = _mm256_add_ps( yv[0], iv[0] );
			yv[1] = _mm256_add_ps( yv[1], iv[1] );
			yv[2] = _mm256_add_ps( yv[2], iv[2] );

			_mm256_storeu_ps( (y0 + 0*n_elem_per_reg), yv[0] );
			_mm256_storeu_ps( (y0 + 1*n_elem_per_reg), yv[1] );
			_mm256_storeu_ps( (y0 + 2*n_elem_per_reg), yv[2] );

			y0 += 3*n_elem_per_reg;
			x0 += 3*n_elem_per_reg;
		}

		// Processing 16 elements per loop, 8 FMAs
		for ( ; ( i + 7 ) < n; i += 8 )
		{
			// xv = xR1  xI1  xR2  xI2  xR3  xI3  xR4  xI4
			xv[0] = _mm256_loadu_ps( x0 + 0*n_elem_per_reg );
			xv[1] = _mm256_loadu_ps( x0 + 1*n_elem_per_reg );

			// yv = yR1  yI1  yR2  yI2  yR3  yI3  yR4  yI4
			yv[0] = _mm256_loadu_ps( y0 + 0*n_elem_per_reg );
			yv[1] = _mm256_loadu_ps( y0 + 1*n_elem_per_reg );

			// iv = betaRv * yv
			//    = yR1.bR, yI1.bR, yR2.bR, yI2.bR, ...
			iv[0] = _mm256_mul_ps( betaRv, yv[0] );
			iv[1] = _mm256_mul_ps( betaRv, yv[1] );

			// yv' = yI1  yR1  yI2  yR2  yI3  yR3  yI4  yR4
			yv[0] = _mm256_permute_ps( yv[0], 0xB1);
			yv[1] = _mm256_permute_ps( yv[1], 0xB1);

			// yv = betaIv * yv' + iv
			//    = yR1.bR - yI1.bI, yI1.bR + yR1.bI, ...
			yv[0] = _mm256_fmadd_ps( betaIv, yv[0], iv[0] );
			yv[1] = _mm256_fmadd_ps( betaIv, yv[1], iv[1] );

			// iv = alphaRv * xv
			//    = xR1.aR, xI1.aR, xR2.aR, xI2.aR, ...
			iv[0] = _mm256_mul_ps( alphaRv, xv[0] );
			iv[1] = _mm256_mul_ps( alphaRv, xv[1] );

			// xv' =  xI1  xR1  xI2  xR2  xI3  xR3  xI4  xR4
			xv[0] = _mm256_permute_ps( xv[0], 0xB1);
			xv[1] = _mm256_permute_ps( xv[1], 0xB1);

			// yv = alphaIv * xv + yv
			//    = yR1.bR - yR1.bI - xR1.aI, yI1.bR + yI1.bI + xI1.aI, ...
			iv[0] = _mm256_fmadd_ps( alphaIv, xv[0], iv[0] );
			iv[1] = _mm256_fmadd_ps( alphaIv, xv[1], iv[1] );

			yv[0] = _mm256_add_ps( yv[0], iv[0] );
			yv[1] = _mm256_add_ps( yv[1], iv[1] );

			_mm256_storeu_ps( (y0 + 0*n_elem_per_reg), yv[0] );
			_mm256_storeu_ps( (y0 + 1*n_elem_per_reg), yv[1] );

			y0 += 2*n_elem_per_reg;
			x0 += 2*n_elem_per_reg;
		}

		// Issue vzeroupper instruction to clear upper lanes of ymm registers.
		// This avoids a performance penalty caused by false dependencies when
		// transitioning from AVX to SSE instructions (which may occur as soon
		// as the n_left cleanup loop below if BLIS is compiled with
		// -mfpmath=sse).
		_mm256_zeroupper();

		if ( !bli_is_conj( conjx_use ) )
		{
			for ( ; i < n ; ++i )
			{
				const float yRc = *y0;
				const float yIc = *( y0 + 1 );

				*y0       = ( betaR  * yRc ) - ( betaI  * yIc ) +
							( alphaR * (*x0) ) - ( alphaI * (*(x0 + 1)) );
				*(y0 + 1) = ( betaR  * yIc ) + ( betaI  * yRc ) +
							( alphaR * (*(x0 + 1)) ) + ( alphaI * (*x0) );

				x0 += 2;
				y0 += 2;
			}
		}
		else
		{
			for ( ; i < n ; ++i )
			{
				const float yRc = *y0;
				const float yIc = *( y0 + 1 );

				*y0       = ( betaR  * yRc ) - ( betaI  * yIc ) +
							( alphaR * (*x0) ) + ( alphaI * (*(x0 + 1)) );
				*(y0 + 1) = ( betaR  * yIc ) + ( betaI  * yRc ) -
							( alphaR * (*(x0 + 1)) ) + ( alphaI * (*x0) );

				x0 += 2;
				y0 += 2;
			}
		}
	}
	else
	{
		// for non-unit increments, use scaler code
		if ( !bli_is_conj( conjx_use ) )
		{
			for ( i = 0; i < n ; ++i )
			{
				const float yRc = *y0;
				const float yIc = *( y0 + 1 );

				// yReal = ( bR.yR - bI.yI + aR.xR - aI.xI )
				*y0       = ( betaR  * yRc ) - ( betaI  * yIc ) +
							( alphaR * (*x0) ) - ( alphaI * (*(x0 + 1)) );
				// yImag = ( bR.yI + bI.yR + aR.xI + aI.xR )
				*(y0 + 1) = ( betaR  * yIc ) + ( betaI  * yRc ) +
							( alphaR * (*(x0 + 1)) ) + ( alphaI * (*x0) );

				x0 += incx * 2;
				y0 += incy * 2;
			}
		}
		else
		{
			for ( i = 0; i < n ; ++i )
			{
				const float yRc = *y0;
				const float yIc = *( y0 + 1 );

				// yReal = ( bR.yR - bI.yI + aR.xR - aI.xI )
				*y0       = ( betaR  * yRc ) - ( betaI  * yIc ) +
							( alphaR * (*x0) ) + ( alphaI * (*(x0 + 1)) );
				// yImag = ( bR.yI + bI.yR + aR.xI + aI.xR )
				*(y0 + 1) = ( betaR  * yIc ) + ( betaI  * yRc ) -
							( alphaR * (*(x0 + 1)) ) + ( alphaI * (*x0) );

				x0 += incx * 2;
				y0 += incy * 2;
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
 * 		alpha & beta are scalers.
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

	dim_t i = 0; // iterator

	// Local pointers to x and y vectors
	double*  restrict x0;
	double*  restrict y0;

	// Variables to store real and imaginary components of alpha and beta
	double alphaR, alphaI, betaR, betaI;

	// Local variable to store the conjugate type
	conj_t conjx_use = conjx;

	/* If the vector dimension is zero, return early. */
	if ( bli_zero_dim1( n ) )
	{
		AOCL_DTL_TRACE_EXIT(AOCL_DTL_LEVEL_TRACE_4)
		return;
	}

	// Initializing the local pointers
	x0  = ( double* ) x;
	y0  = ( double* ) y;

	alphaR = alpha->real;
	alphaI = alpha->imag;
	betaR  = beta->real;
	betaI  = beta->imag;

	// Vectors to store real and imaginary components of beta
	__m256d betaRv, betaIv;

	// Broadcasting real and imaginary components of beta onto the registers
	betaRv = _mm256_broadcast_sd( &betaR );
	betaIv = _mm256_broadcast_sd( &betaI );

	// Initializing a variable to classify the type of the computation
	bool is_alpha_zero = bli_zeq0( *alpha );

	// In case of unit strides for x and y vectors
	if ( incx == 1 && incy == 1 )
	{
		// Number of double precision elements in a YMM register
		const dim_t  n_elem_per_reg = 4;

		// Scratch registers
		__m256d xv[4];
		__m256d yv[4];
		__m256d iv[4];

		// In case of alpha being 0, we just need to scale y by beta
		if( is_alpha_zero )
		{
			// Processing 8 elements per loop, 8 FMAs
			for ( i = 0; ( i + 7 ) < n; i += 8 )
			{
				// Load the y vector, 8 elements in total
				// yv =  yR1  yI1  yR2  yI2
				yv[0] = _mm256_loadu_pd( y0 );
				yv[1] = _mm256_loadu_pd( y0 + 1 * n_elem_per_reg );
				yv[2] = _mm256_loadu_pd( y0 + 2 * n_elem_per_reg );
				yv[3] = _mm256_loadu_pd( y0 + 3 * n_elem_per_reg );

				// Permute the loaded vectors for the required compute
				// xv =  yI1  yR1  yI2  yR2
				xv[0] = _mm256_permute_pd( yv[0], 5 );
				xv[1] = _mm256_permute_pd( yv[1], 5 );
				xv[2] = _mm256_permute_pd( yv[2], 5 );
				xv[3] = _mm256_permute_pd( yv[3], 5 );

				// Scale the permuted vectors with imaginary component of beta
				// iv =  yI1  yR1  yI2  yR2
				iv[0] = _mm256_mul_pd( betaIv, xv[0] );
				iv[1] = _mm256_mul_pd( betaIv, xv[1] );
				iv[2] = _mm256_mul_pd( betaIv, xv[2] );
				iv[3] = _mm256_mul_pd( betaIv, xv[3] );

				// Using fmaddsub to scale with real component of beta and sub/add to iv
				// yv = betaRv * yv -/+ iv
				//    = yR1.bR - yI1.bI, yI1.bR + yR1.bI, ...
				yv[0] = _mm256_fmaddsub_pd( betaRv, yv[0], iv[0] );
				yv[1] = _mm256_fmaddsub_pd( betaRv, yv[1], iv[1] );
				yv[2] = _mm256_fmaddsub_pd( betaRv, yv[2], iv[2] );
				yv[3] = _mm256_fmaddsub_pd( betaRv, yv[3], iv[3] );

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

				// Permute the loaded vectors for the required compute
				// xv =  yI1  yR1  yI2  yR2
				xv[0] = _mm256_permute_pd( yv[0], 5 );
				xv[1] = _mm256_permute_pd( yv[1], 5 );
				xv[2] = _mm256_permute_pd( yv[2], 5 );

				// Scale the permuted vectors with imaginary component of beta
				// iv =  yI1  yR1  yI2  yR2
				iv[0] = _mm256_mul_pd( betaIv, xv[0] );
				iv[1] = _mm256_mul_pd( betaIv, xv[1] );
				iv[2] = _mm256_mul_pd( betaIv, xv[2] );

				// Using fmaddsub to scale with real component of beta
				// and sub/add to iv
				// yv = betaRv * yv -/+ iv
				//    = yR1.bR - yI1.bI, yI1.bR + yR1.bI, ...
				yv[0] = _mm256_fmaddsub_pd( betaRv, yv[0], iv[0] );
				yv[1] = _mm256_fmaddsub_pd( betaRv, yv[1], iv[1] );
				yv[2] = _mm256_fmaddsub_pd( betaRv, yv[2], iv[2] );

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

				// Permute the loaded vectors for the required compute
				// xv =  yI1  yR1  yI2  yR2
				xv[0] = _mm256_permute_pd( yv[0], 5 );
				xv[1] = _mm256_permute_pd( yv[1], 5 );

				// Scale the permuted vectors with imaginary component of beta
				// iv =  yI1.bI, yR1.bI, yI2.bI, yR2.bI
				iv[0] = _mm256_mul_pd( betaIv, xv[0] );
				iv[1] = _mm256_mul_pd( betaIv, xv[1] );

				// Using fmaddsub to scale with real component of beta
				// and sub/add to iv
				// yv = betaRv * yv -/+ iv
				//    = yR1.bR - yI1.bI, yI1.bR + yR1.bI, ...
				yv[0] = _mm256_fmaddsub_pd( betaRv, yv[0], iv[0] );
				yv[1] = _mm256_fmaddsub_pd( betaRv, yv[1], iv[1] );

				// Storing the result to memory
				_mm256_storeu_pd( ( y0 ), yv[0] );
				_mm256_storeu_pd( ( y0 + 1 * n_elem_per_reg ), yv[1] );

				// Adjusting the pointers for the next iteration
				y0 += 2 * n_elem_per_reg;
				x0 += 2 * n_elem_per_reg;
			}

			// Processing 2 elements per loop, 3 FMAs
			for ( ; ( i + 1 ) < n; i += 2 )
			{
				// Load the y vector, 2 elements in total
				// yv =  yR1  yI1  yR2  yI2
				yv[0] = _mm256_loadu_pd( y0 );

				// Permute the loaded vectors for the required compute
				// xv =  yI1  yR1  yI2  yR2
				xv[0] = _mm256_permute_pd( yv[0], 5 );

				// Scale the permuted vectors with imaginary component of beta
				// iv =  yI1  yR1  yI2  yR2
				iv[0] = _mm256_mul_pd( betaIv, xv[0] );

				// Using fmaddsub to scale with real component of beta
				// and sub/add to iv
				// yv = betaRv * yv -/+ iv
				//    = yR1.bR - yI1.bI, yI1.bR + yR1.bI, ...
				yv[0] = _mm256_fmaddsub_pd( betaRv, yv[0], iv[0] );

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
			if( bli_is_conj( conjx_use ) )
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
				// Load the y vector, 6 elements in total
				// yv =  yR1  yI1  yR2  yI2
				yv[0] = _mm256_loadu_pd( y0 );
				yv[1] = _mm256_loadu_pd( y0 + 1 * n_elem_per_reg );
				yv[2] = _mm256_loadu_pd( y0 + 2 * n_elem_per_reg );
				yv[3] = _mm256_loadu_pd( y0 + 3 * n_elem_per_reg );

				// Load the x vector, 6 elements in total
				// xv = xR1  xI1  xR2  xI2
				xv[0] = _mm256_loadu_pd( x0 );
				xv[1] = _mm256_loadu_pd( x0 + 1 * n_elem_per_reg );
				xv[2] = _mm256_loadu_pd( x0 + 2 * n_elem_per_reg );
				xv[3] = _mm256_loadu_pd( x0 + 3 * n_elem_per_reg );

				// Permute the vectors from y for the required compute
				// iv =  yI1  yR1  yI2  yR2
				iv[0] = _mm256_permute_pd( yv[0], 5 );
				iv[1] = _mm256_permute_pd( yv[1], 5 );
				iv[2] = _mm256_permute_pd( yv[2], 5 );
				iv[3] = _mm256_permute_pd( yv[3], 5 );

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
				iv[0] = _mm256_permute_pd( xv[0], 5 );
				iv[1] = _mm256_permute_pd( xv[1], 5 );
				iv[2] = _mm256_permute_pd( xv[2], 5 );
				iv[3] = _mm256_permute_pd( xv[3], 5 );

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
				iv[0] = _mm256_permute_pd( yv[0], 5 );
				iv[1] = _mm256_permute_pd( yv[1], 5 );
				iv[2] = _mm256_permute_pd( yv[2], 5 );

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
				iv[0] = _mm256_permute_pd( xv[0], 5 );
				iv[1] = _mm256_permute_pd( xv[1], 5 );
				iv[2] = _mm256_permute_pd( xv[2], 5 );

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
				// Load the y vector, 6 elements in total
				// yv =  yR1  yI1  yR2  yI2
				yv[0] = _mm256_loadu_pd( y0 );
				yv[1] = _mm256_loadu_pd( y0 + 1 * n_elem_per_reg );

				// Load the x vector, 6 elements in total
				// xv = xR1  xI1  xR2  xI2
				xv[0] = _mm256_loadu_pd( x0 );
				xv[1] = _mm256_loadu_pd( x0 + 1 * n_elem_per_reg );

				// Permute the vectors from y for the required compute
				// iv =  yI1  yR1  yI2  yR2
				iv[0] = _mm256_permute_pd( yv[0], 5 );
				iv[1] = _mm256_permute_pd( yv[1], 5 );

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
				iv[0] = _mm256_permute_pd( xv[0], 5 );
				iv[1] = _mm256_permute_pd( xv[1], 5 );

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

			// Processing 2 elements per loop, 3 FMAs
			for ( ; ( i + 1 ) < n; i += 2 )
			{
				// Load the y vector, 6 elements in total
				// yv =  yR1  yI1  yR2  yI2
				yv[0] = _mm256_loadu_pd( y0 );

				// Load the x vector, 6 elements in total
				// xv = xR1  xI1  xR2  xI2
				xv[0] = _mm256_loadu_pd( x0 );

				// Permute the vectors from y for the required compute
				// iv =  yI1  yR1  yI2  yR2
				iv[0] = _mm256_permute_pd( yv[0], 5 );

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
				iv[0] = _mm256_permute_pd( xv[0], 5 );

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
	__m128d betaRv_128, betaIv_128;

	// Casting the lower 128-bit lanes from betaRv and betaIv to its 128-bit alternative
	// registers to avoid redundant broadcasts.
	betaRv_128 = _mm256_castpd256_pd128( betaRv );
	betaIv_128 = _mm256_castpd256_pd128( betaIv );

	// NOTE : We cannot similarly use _mm256_castpd256_pd128 to avoid loading alpha
	//		  since alpha is loaded onto its YMM rgeisters on requirement basis.
	//        In case of directly falling to this compute(non-unit stride cases),
	//        alpha wouldn't have been loaded onto any YMM reigsters.

	// Changing betaIv_128 to { -bI  bI } for the compute
	x_elem = _mm_setzero_pd();
	betaIv_128 = _mm_addsub_pd( x_elem, betaIv_128 );

	// In case of alpha being 0, we just need to scale y by beta
	if ( is_alpha_zero )
	{
		// Iterate over y, one element at a time
		for ( ; i < n; i += 1 )
		{
			// Load an element from y
			// y_elem =  yR1  yI1
			y_elem = _mm_loadu_pd( y0 );

			// Permute y in accordance to its compute
			// y_perm =  yI1  yR1
			y_perm = _mm_permute_pd( y_elem, 0x1 );

			// Scale y_perm by the imaginary
			// component of beta
			// y_perm =  -yI1.bI, yR1.bI
			y_perm = _mm_mul_pd( betaIv_128, y_perm );

			// Use fmadd to scale with real component of
			// beta and add with intermediate result
			// y_elem =  yR1.bR - yI1.bI, yI1.bR + yR1.bI
			y_elem = _mm_fmadd_pd( betaRv_128, y_elem, y_perm );

			// Storing the result to memory
			_mm_storeu_pd( y0, y_elem );

			// Adjusting the pointer for the next iteration
			y0 += incy * 2;
		}
	}
	else
	{
		// Scratch registers to store real and imaginary components
		// of alpha onto XMM registers
		__m128d alphaRv_128, alphaIv_128;

		// Broadcasting real and imaginary components of alpha
		x_elem = _mm_setzero_pd();
		alphaRv_128 = _mm_loaddup_pd( &alphaR );
		alphaIv_128 = _mm_loaddup_pd( &alphaI );

		// The changes on alphaRv_128 and alphaIv_128 are as follows :
		// If conjugate is required:
		//		alphaRv_128 =  aR  -aR
		// Else :
		//		alphaIv_128 =  -aI  aI
		if( bli_is_conj( conjx_use ) )
		{
			alphaRv_128 = _mm_addsub_pd( x_elem, alphaRv_128 );
			alphaRv_128 = _mm_permute_pd( alphaRv_128, 0x1 );
		}
		else
		{
			alphaIv_128 = _mm_addsub_pd( x_elem, alphaIv_128 );
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
			y_perm = _mm_mul_pd( betaIv_128, y_perm );
			x_perm = _mm_mul_pd( alphaIv_128, x_perm );

			// Use fmadd to scale with y_elem with
			// real component of beta and add with
			// intermediate result. Similarly do
			// for x_elem.
			// y_elem =  yR1.bR - yI1.bI, yI1.bR + yR1.bI
			// x_elem =  xR1.aR - xI1.aI, xI1.aR + xR1.aI
			y_elem = _mm_fmadd_pd( betaRv_128, y_elem, y_perm );
			x_elem = _mm_fmadd_pd( alphaRv_128, x_elem, x_perm );

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
