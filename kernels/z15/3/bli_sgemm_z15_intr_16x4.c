/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2026, IBM Corp.

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

#include <vecintrin.h>

#define Z15_SGEMM_16X4_STEP() \
	do \
	{ \
		const __vector float a0 = vec_xl(  0, a_cast ); \
		const __vector float a1 = vec_xl( 16, a_cast ); \
		const __vector float a2 = vec_xl( 32, a_cast ); \
		const __vector float a3 = vec_xl( 48, a_cast ); \
		const __vector float b0 = vec_splats( b_cast[0] ); \
		const __vector float b1 = vec_splats( b_cast[1] ); \
		const __vector float b2 = vec_splats( b_cast[2] ); \
		const __vector float b3 = vec_splats( b_cast[3] ); \
		c00 = vec_madd( a0, b0, c00 ); \
		c01 = vec_madd( a0, b1, c01 ); \
		c02 = vec_madd( a0, b2, c02 ); \
		c03 = vec_madd( a0, b3, c03 ); \
		c10 = vec_madd( a1, b0, c10 ); \
		c11 = vec_madd( a1, b1, c11 ); \
		c12 = vec_madd( a1, b2, c12 ); \
		c13 = vec_madd( a1, b3, c13 ); \
		c20 = vec_madd( a2, b0, c20 ); \
		c21 = vec_madd( a2, b1, c21 ); \
		c22 = vec_madd( a2, b2, c22 ); \
		c23 = vec_madd( a2, b3, c23 ); \
		c30 = vec_madd( a3, b0, c30 ); \
		c31 = vec_madd( a3, b1, c31 ); \
		c32 = vec_madd( a3, b2, c32 ); \
		c33 = vec_madd( a3, b3, c33 ); \
		a_cast += 16; \
		b_cast += 4; \
	} while ( 0 )

void bli_sgemm_z15_intr_16x4
     (
             dim_t      m,
             dim_t      n,
             dim_t      k,
       const void*      alpha,
       const void*      a,
       const void*      b,
       const void*      beta,
             void*      c, inc_t rs_c, inc_t cs_c,
       const auxinfo_t* data,
       const cntx_t*    cntx
     )
{
	GEMM_UKR_SETUP_CT( s, 16, 4, false );

	const float  alpha_val = *( const float* )alpha;
	const float  beta_val  = *( const float* )beta;
	const float* restrict a_cast = a;
	const float* restrict b_cast = b;
	      float* restrict c_cast = c;

	const __vector float zero = vec_splats( 0.0f );
	__vector float c00 = zero;
	__vector float c10 = zero;
	__vector float c20 = zero;
	__vector float c30 = zero;
	__vector float c01 = zero;
	__vector float c11 = zero;
	__vector float c21 = zero;
	__vector float c31 = zero;
	__vector float c02 = zero;
	__vector float c12 = zero;
	__vector float c22 = zero;
	__vector float c32 = zero;
	__vector float c03 = zero;
	__vector float c13 = zero;
	__vector float c23 = zero;
	__vector float c33 = zero;

	const dim_t k_iter = k / 2;
	const dim_t k_left = k % 2;

	for ( dim_t p = 0; p < k_iter; ++p )
	{
		Z15_SGEMM_16X4_STEP();
		Z15_SGEMM_16X4_STEP();
	}

	if ( k_left )
	{
		Z15_SGEMM_16X4_STEP();
	}

	if ( alpha_val != 1.0f )
	{
		const __vector float alpha_v = vec_splats( alpha_val );

		c00 *= alpha_v;
		c01 *= alpha_v;
		c02 *= alpha_v;
		c03 *= alpha_v;
		c10 *= alpha_v;
		c11 *= alpha_v;
		c12 *= alpha_v;
		c13 *= alpha_v;
		c20 *= alpha_v;
		c21 *= alpha_v;
		c22 *= alpha_v;
		c23 *= alpha_v;
		c30 *= alpha_v;
		c31 *= alpha_v;
		c32 *= alpha_v;
		c33 *= alpha_v;
	}

	float* restrict c0 = c_cast;
	float* restrict c1 = c0 + cs_c;
	float* restrict c2 = c1 + cs_c;
	float* restrict c3 = c2 + cs_c;

	if ( beta_val != 0.0f )
	{
		const __vector float beta_v = vec_splats( beta_val );

		c00 = vec_madd( beta_v, vec_xl(  0, c0 ), c00 );
		c10 = vec_madd( beta_v, vec_xl( 16, c0 ), c10 );
		c20 = vec_madd( beta_v, vec_xl( 32, c0 ), c20 );
		c30 = vec_madd( beta_v, vec_xl( 48, c0 ), c30 );
		c01 = vec_madd( beta_v, vec_xl(  0, c1 ), c01 );
		c11 = vec_madd( beta_v, vec_xl( 16, c1 ), c11 );
		c21 = vec_madd( beta_v, vec_xl( 32, c1 ), c21 );
		c31 = vec_madd( beta_v, vec_xl( 48, c1 ), c31 );
		c02 = vec_madd( beta_v, vec_xl(  0, c2 ), c02 );
		c12 = vec_madd( beta_v, vec_xl( 16, c2 ), c12 );
		c22 = vec_madd( beta_v, vec_xl( 32, c2 ), c22 );
		c32 = vec_madd( beta_v, vec_xl( 48, c2 ), c32 );
		c03 = vec_madd( beta_v, vec_xl(  0, c3 ), c03 );
		c13 = vec_madd( beta_v, vec_xl( 16, c3 ), c13 );
		c23 = vec_madd( beta_v, vec_xl( 32, c3 ), c23 );
		c33 = vec_madd( beta_v, vec_xl( 48, c3 ), c33 );
	}

	vec_xst( c00,  0, c0 );
	vec_xst( c10, 16, c0 );
	vec_xst( c20, 32, c0 );
	vec_xst( c30, 48, c0 );
	vec_xst( c01,  0, c1 );
	vec_xst( c11, 16, c1 );
	vec_xst( c21, 32, c1 );
	vec_xst( c31, 48, c1 );
	vec_xst( c02,  0, c2 );
	vec_xst( c12, 16, c2 );
	vec_xst( c22, 32, c2 );
	vec_xst( c32, 48, c2 );
	vec_xst( c03,  0, c3 );
	vec_xst( c13, 16, c3 );
	vec_xst( c23, 32, c3 );
	vec_xst( c33, 48, c3 );

	GEMM_UKR_FLUSH_CT( s );

	( void )data;
	( void )cntx;
}

#undef Z15_SGEMM_16X4_STEP
