/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2014, The University of Texas at Austin
   Copyright (C) 2021, The University of Tokyo

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

#include <arm_acle.h>
#include <arm_sme.h>
#include "blis.h"

__arm_new( "za" ) __arm_locally_streaming void bli_sgemm_armsme_int_2SVLx2SVL
	(
			  dim_t      m,
			  dim_t      n,
			  dim_t      k,
		const void*      alpha,
		const void*      a,
		const void*      b,
		const void*      beta,
			  void*		 c, inc_t rs_c, inc_t cs_c,
		const auxinfo_t* data,
		const cntx_t*    cntx
	) 
{
	uint64_t SVL = svcntsw();

	GEMM_UKR_SETUP_CT_AMBI( s, 2 * SVL, 2 * SVL, false );

	float *a_ = (float *)a;
	float *b_ = (float *)b;

	float *a_next = (float *)bli_auxinfo_next_a( data );
	float *b_next = (float *)bli_auxinfo_next_b( data );

	float *c_ = (float *)c;

	const uint64_t result_tile_TL_corner_ = 0;
	const uint64_t result_tile_TR_corner_ = result_tile_TL_corner_ + SVL;

	if ( cs_c != 1 )
	{
		__pldx( 1, 1, 0,
			(float *)&c_[result_tile_TL_corner_ + ( ( ( 0 + 0 ) * cs_c ) )] );
		__pldx( 1, 1, 0,
			(float *)&c_[result_tile_TL_corner_ + ( ( ( 1 + 0 ) * cs_c ) )] );
		__pldx( 1, 1, 0,
			(float *)&c_[result_tile_TL_corner_ + ( ( ( 2 + 0 ) * cs_c ) )] );
		__pldx( 1, 1, 0,
			(float *)&c_[result_tile_TL_corner_ + ( ( ( 3 + 0 ) * cs_c ) )] );

		__pldx( 1, 1, 0,
			(float *)&c_[result_tile_TL_corner_ + ( ( ( 4 + 0 ) * cs_c ) )] );
		__pldx( 1, 1, 0,
			(float *)&c_[result_tile_TL_corner_ + ( ( ( 5 + 0 ) * cs_c ) )] );
		__pldx( 1, 1, 0,
			(float *)&c_[result_tile_TL_corner_ + ( ( ( 6 + 0 ) * cs_c ) )] );
		__pldx( 1, 1, 0,
			(float *)&c_[result_tile_TL_corner_ + ( ( ( 7 + 0 ) * cs_c ) )] );

		__pldx( 1, 1, 0,
			(float *)&c_[result_tile_TR_corner_ + ( ( ( 0 + 0 ) * cs_c ) )] );
		__pldx( 1, 1, 0,
			(float *)&c_[result_tile_TR_corner_ + ( ( ( 1 + 0 ) * cs_c ) )] );
		__pldx( 1, 1, 0,
			(float *)&c_[result_tile_TR_corner_ + ( ( ( 2 + 0 ) * cs_c ) )] );
		__pldx( 1, 1, 0,
			(float *)&c_[result_tile_TR_corner_ + ( ( ( 3 + 0 ) * cs_c ) )] );

		__pldx( 1, 1, 0,
			(float *)&c_[result_tile_TR_corner_ + ( ( ( 4 + 0 ) * cs_c ) )] );
		__pldx( 1, 1, 0,
			(float *)&c_[result_tile_TR_corner_ + ( ( ( 5 + 0 ) * cs_c ) )] );
		__pldx( 1, 1, 0,
			(float *)&c_[result_tile_TR_corner_ + ( ( ( 6 + 0 ) * cs_c ) )] );
		__pldx( 1, 1, 0,
			(float *)&c_[result_tile_TR_corner_ + ( ( ( 7 + 0 ) * cs_c ) )] );
	}
	else
	{
		__pldx( 1, 1, 0,
			(float *)&c_[result_tile_TL_corner_ + ( ( ( 0 + 0 ) * rs_c ) )] );
		__pldx( 1, 1, 0,
			(float *)&c_[result_tile_TL_corner_ + ( ( ( 1 + 0 ) * rs_c ) )] );
		__pldx( 1, 1, 0,
			(float *)&c_[result_tile_TL_corner_ + ( ( ( 2 + 0 ) * rs_c ) )] );
		__pldx( 1, 1, 0,
			(float *)&c_[result_tile_TL_corner_ + ( ( ( 3 + 0 ) * rs_c ) )] );

		__pldx( 1, 1, 0,
			(float *)&c_[result_tile_TL_corner_ + ( ( ( 4 + 0 ) * rs_c ) )] );
		__pldx( 1, 1, 0,
			(float *)&c_[result_tile_TL_corner_ + ( ( ( 5 + 0 ) * rs_c ) )] );
		__pldx( 1, 1, 0,
			(float *)&c_[result_tile_TL_corner_ + ( ( ( 6 + 0 ) * rs_c ) )] );
		__pldx( 1, 1, 0,
			(float *)&c_[result_tile_TL_corner_ + ( ( ( 7 + 0 ) * rs_c ) )] );

		__pldx( 1, 1, 0,
			(float *)&c_[result_tile_TR_corner_ + ( ( ( 0 + 0 ) * rs_c ) )] );
		__pldx( 1, 1, 0,
			(float *)&c_[result_tile_TR_corner_ + ( ( ( 1 + 0 ) * rs_c ) )] );
		__pldx( 1, 1, 0,
			(float *)&c_[result_tile_TR_corner_ + ( ( ( 2 + 0 ) * rs_c ) )] );
		__pldx( 1, 1, 0,
			(float *)&c_[result_tile_TR_corner_ + ( ( ( 3 + 0 ) * rs_c ) )] );

		__pldx( 1, 1, 0,
			(float *)&c_[result_tile_TR_corner_ + ( ( ( 4 + 0 ) * rs_c ) )] );
		__pldx( 1, 1, 0,
			(float *)&c_[result_tile_TR_corner_ + ( ( ( 5 + 0 ) * rs_c ) )] );
		__pldx( 1, 1, 0,
			(float *)&c_[result_tile_TR_corner_ + ( ( ( 6 + 0 ) * rs_c ) )] );
		__pldx( 1, 1, 0,
			(float *)&c_[result_tile_TR_corner_ + ( ( ( 7 + 0 ) * rs_c ) )] );
	}

	svzero_za();

	uint64_t k_;
	uint64_t k_iter = k / 8;
	uint64_t k_left = k % 8;

	for ( k_ = 0; k_ < k_iter; k_++ )
	{
		svfloat32x4_t zL00 = svld1_f32_x4( svptrue_c32(),
			(float32_t *)( &a_[0] ) );
		svfloat32x4_t zR00 = svld1_f32_x4( svptrue_c32(),
			(float32_t *)( &b_[0] ) );

		svmopa_za32_m( 0, svptrue_b32(), svptrue_b32(), svget4( zL00, 0 ),
			svget4( zR00, 0 ) );
		svmopa_za32_m( 1, svptrue_b32(), svptrue_b32(), svget4( zL00, 1 ),
			svget4( zR00, 0 ) );

		__pldx( 0, 1, 1, (float *)&a_next[0] );
		__pldx( 0, 1, 1, (float *)&b_next[0] );

		svmopa_za32_m( 2, svptrue_b32(), svptrue_b32(), svget4( zL00, 0 ),
			svget4( zR00, 1 ) );
		svmopa_za32_m( 3, svptrue_b32(), svptrue_b32(), svget4( zL00, 1 ),
			svget4( zR00, 1 ) );

		svmopa_za32_m( 0, svptrue_b32(), svptrue_b32(), svget4( zL00, 2 ),
			svget4( zR00, 2 ) );
		svmopa_za32_m( 1, svptrue_b32(), svptrue_b32(), svget4( zL00, 3 ),
			svget4( zR00, 2 ) );

		svmopa_za32_m( 2, svptrue_b32(), svptrue_b32(), svget4( zL00, 2 ),
			svget4( zR00, 3 ) );
		svmopa_za32_m( 3, svptrue_b32(), svptrue_b32(), svget4( zL00, 3 ),
			svget4( zR00, 3 ) );

		svfloat32x4_t zL02 = svld1_f32_x4( svptrue_c32(),
			(float32_t *)( &a_[( 4 * SVL )] ) );
		svfloat32x4_t zR02 = svld1_f32_x4( svptrue_c32(),
			(float32_t *)( &b_[( 4 * SVL )] ) );

		svmopa_za32_m( 0, svptrue_b32(), svptrue_b32(), svget4( zL02, 0 ),
			svget4( zR02, 0 ) );
		svmopa_za32_m( 1, svptrue_b32(), svptrue_b32(), svget4( zL02, 1 ),
			svget4( zR02, 0 ) );

		__pldx( 0, 1, 1, (float *)&a_next[4 * SVL] );
		__pldx( 0, 1, 1, (float *)&b_next[4 * SVL] );

		svmopa_za32_m( 2, svptrue_b32(), svptrue_b32(), svget4( zL02, 0 ),
			svget4( zR02, 1 ) );
		svmopa_za32_m( 3, svptrue_b32(), svptrue_b32(), svget4( zL02, 1 ),
			svget4( zR02, 1 ) );

		svmopa_za32_m( 0, svptrue_b32(), svptrue_b32(), svget4( zL02, 2 ),
			svget4( zR02, 2 ) );
		svmopa_za32_m( 1, svptrue_b32(), svptrue_b32(), svget4( zL02, 3 ),
			svget4( zR02, 2 ) );
		svmopa_za32_m( 2, svptrue_b32(), svptrue_b32(), svget4( zL02, 2 ),
			svget4( zR02, 3 ) );
		svmopa_za32_m( 3, svptrue_b32(), svptrue_b32(), svget4( zL02, 3 ),
			svget4( zR02, 3 ) );

		svfloat32x4_t zL04 = svld1_f32_x4( svptrue_c32(),
			(float32_t *)( &a_[8 * SVL] ) );
		svfloat32x4_t zR04 = svld1_f32_x4( svptrue_c32(),
			(float32_t *)( &b_[8 * SVL] ) );

		svmopa_za32_m( 0, svptrue_b32(), svptrue_b32(), svget4( zL04, 0 ),
			svget4( zR04, 0 ) );
		svmopa_za32_m( 1, svptrue_b32(), svptrue_b32(), svget4( zL04, 1 ),
			svget4( zR04, 0 ) );

		__pldx( 0, 1, 1, (float *)&a_next[8 * SVL] );
		__pldx( 0, 1, 1, (float *)&b_next[8 * SVL] );

		svmopa_za32_m( 2, svptrue_b32(), svptrue_b32(), svget4( zL04, 0 ),
			svget4( zR04, 1 ) );
		svmopa_za32_m( 3, svptrue_b32(), svptrue_b32(), svget4( zL04, 1 ),
			svget4( zR04, 1 ) );

		svmopa_za32_m( 0, svptrue_b32(), svptrue_b32(), svget4( zL04, 2 ),
			svget4( zR04, 2 ) );
		svmopa_za32_m( 1, svptrue_b32(), svptrue_b32(), svget4( zL04, 3 ),
			svget4( zR04, 2 ) );

		svmopa_za32_m( 2, svptrue_b32(), svptrue_b32(), svget4( zL04, 2 ),
			svget4( zR04, 3 ) );
		svmopa_za32_m( 3, svptrue_b32(), svptrue_b32(), svget4( zL04, 3 ),
			svget4( zR04, 3 ) );

		svfloat32x4_t zL06 = svld1_f32_x4( svptrue_c32(),
			(float32_t *)( &a_[( 12 * SVL )] ) );
		svfloat32x4_t zR06 = svld1_f32_x4( svptrue_c32(),
			(float32_t *)( &b_[( 12 * SVL )] ) );

		svmopa_za32_m( 0, svptrue_b32(), svptrue_b32(), svget4( zL06, 0 ),
			svget4( zR06, 0 ) );
		svmopa_za32_m( 1, svptrue_b32(), svptrue_b32(), svget4( zL06, 1 ),
			svget4( zR06, 0 ) );

		__pldx( 0, 1, 1, (float *)&a_next[12 * SVL] );
		__pldx( 0, 1, 1, (float *)&b_next[12 * SVL] );

		svmopa_za32_m( 2, svptrue_b32(), svptrue_b32(), svget4( zL06, 0 ),
			svget4( zR06, 1 ) );
		svmopa_za32_m( 3, svptrue_b32(), svptrue_b32(), svget4( zL06, 1 ),
			svget4( zR06, 1 ) );

		svmopa_za32_m( 0, svptrue_b32(), svptrue_b32(), svget4( zL06, 2 ),
			svget4( zR06, 2 ) );
		svmopa_za32_m( 1, svptrue_b32(), svptrue_b32(), svget4( zL06, 3 ),
			svget4( zR06, 2 ) );

		svmopa_za32_m( 2, svptrue_b32(), svptrue_b32(), svget4( zL06, 2 ),
			svget4( zR06, 3 ) );
		svmopa_za32_m( 3, svptrue_b32(), svptrue_b32(), svget4( zL06, 3 ),
			svget4( zR06, 3 ) );

		a_ += ( 2 * 8 * SVL );
		b_ += ( 2 * 8 * SVL );

		a_next += ( 2 * 8 * SVL );
		b_next += ( 2 * 8 * SVL );
	}

	for ( k_ = 0; k_ < k_left; k_ += 1 )
	{
		svfloat32x2_t zL00 = svld1_f32_x2( svptrue_c32(),
			(float32_t *)( &a_[0] ) );
		svfloat32x2_t zR00 = svld1_f32_x2( svptrue_c32(),
			(float32_t *)( &b_[0] ) );

		svmopa_za32_m( 0, svptrue_b32(), svptrue_b32(), svget2( zL00, 0 ),
			svget2( zR00, 0 ) );
		svmopa_za32_m( 1, svptrue_b32(), svptrue_b32(), svget2( zL00, 1 ),
			svget2( zR00, 0 ) );

		svmopa_za32_m( 2, svptrue_b32(), svptrue_b32(), svget2( zL00, 0 ),
			svget2( zR00, 1 ) );
		svmopa_za32_m( 3, svptrue_b32(), svptrue_b32(), svget2( zL00, 1 ),
			svget2( zR00, 1 ) );

		a_ += ( 2 * SVL );
		b_ += ( 2 * SVL );
	}

	// Store ZA to matResult.

	const uint64_t result_tile_TL_corner = 0;

	float beta_ = *(float *)beta;
	float alpha_ = *(float *)alpha;

	svfloat32_t zbeta = svdup_f32( beta_ );
	svfloat32_t zalpha = svdup_f32( alpha_ );

	if ( rs_c == 1 )
	{
		const uint64_t result_tile_TR_corner = SVL * cs_c;

		if ( beta_ == 0 )
		{
			for ( uint64_t tcol = 0; tcol < SVL; tcol += 4 )
			{
				// Read ZA slices into Z regs
				svfloat32_t z0 = svread_ver_za32_m( z0, svptrue_b32(),
					/* tile: */ 0, /* slice: */ tcol + 0 );
				svfloat32_t z1 = svread_ver_za32_m( z1, svptrue_b32(),
					/* tile: */ 1, /* slice: */ tcol + 0 );
				svfloat32_t z2 = svread_ver_za32_m( z2, svptrue_b32(),
					/* tile: */ 2, /* slice: */ tcol + 0 );
				svfloat32_t z3 = svread_ver_za32_m( z3, svptrue_b32(),
					/* tile: */ 3, /* slice: */ tcol + 0 );

				// Scale Z regs by broadcast alpha
				z0 = svmul_f32_z( svptrue_b32(), z0, zalpha );
				z1 = svmul_f32_z( svptrue_b32(), z1, zalpha );
				z2 = svmul_f32_z( svptrue_b32(), z2, zalpha );
				z3 = svmul_f32_z( svptrue_b32(), z3, zalpha );

				// Store full result into C
				svfloat32x2_t z400 = svcreate2( z0, z1 );
				svst1_f32_x2( svptrue_c32(),
					&c_[result_tile_TL_corner + ( tcol + 0 ) * cs_c], z400 );

				svfloat32x2_t z600 = svcreate2( z2, z3 );
				svst1_f32_x2( svptrue_c32(),
					&c_[result_tile_TR_corner + ( tcol + 0 ) * cs_c], z600 );

				// Repeat unfolded x4
				z0 = svread_ver_za32_m( z0, svptrue_b32(),
					/* tile: */ 0, /* slice: */ tcol + 1 );
				z1 = svread_ver_za32_m( z1, svptrue_b32(),
					/* tile: */ 1, /* slice: */ tcol + 1 );
				z2 = svread_ver_za32_m( z2, svptrue_b32(),
					/* tile: */ 2, /* slice: */ tcol + 1 );
				z3 = svread_ver_za32_m( z3, svptrue_b32(),
					/* tile: */ 3, /* slice: */ tcol + 1 );

				z0 = svmul_f32_z( svptrue_b32(), z0, zalpha );
				z1 = svmul_f32_z( svptrue_b32(), z1, zalpha );
				z2 = svmul_f32_z( svptrue_b32(), z2, zalpha );
				z3 = svmul_f32_z( svptrue_b32(), z3, zalpha );

				z400 = svcreate2( z0, z1 );
				svst1_f32_x2( svptrue_c32(),
					&c_[result_tile_TL_corner + ( tcol + 1 ) * cs_c], z400 );

				z600 = svcreate2( z2, z3 );
				svst1_f32_x2( svptrue_c32(),
					&c_[result_tile_TR_corner + ( tcol + 1 ) * cs_c], z600 );

				z0 = svread_ver_za32_m( z0, svptrue_b32(),
					/* tile: */ 0, /* slice: */ tcol + 2 );
				z1 = svread_ver_za32_m( z1, svptrue_b32(),
					/* tile: */ 1, /* slice: */ tcol + 2 );
				z2 = svread_ver_za32_m( z2, svptrue_b32(),
					/* tile: */ 2, /* slice: */ tcol + 2 );
				z3 = svread_ver_za32_m( z3, svptrue_b32(),
					/* tile: */ 3, /* slice: */ tcol + 2 );

				z0 = svmul_f32_z( svptrue_b32(), z0, zalpha );
				z1 = svmul_f32_z( svptrue_b32(), z1, zalpha );
				z2 = svmul_f32_z( svptrue_b32(), z2, zalpha );
				z3 = svmul_f32_z( svptrue_b32(), z3, zalpha );

				z400 = svcreate2( z0, z1 );
				svst1_f32_x2( svptrue_c32(),
					&c_[result_tile_TL_corner + ( tcol + 2 ) * cs_c], z400 );

				z600 = svcreate2( z2, z3 );
				svst1_f32_x2( svptrue_c32(),
					&c_[result_tile_TR_corner + ( tcol + 2 ) * cs_c], z600 );

				z0 = svread_ver_za32_m( z0, svptrue_b32(),
					/* tile: */ 0, /* slice: */ tcol + 3 );
				z1 = svread_ver_za32_m( z1, svptrue_b32(),
					/* tile: */ 1, /* slice: */ tcol + 3 );
				z2 = svread_ver_za32_m( z2, svptrue_b32(),
					/* tile: */ 2, /* slice: */ tcol + 3 );
				z3 = svread_ver_za32_m( z3, svptrue_b32(),
					/* tile: */ 3, /* slice: */ tcol + 3 );

				z0 = svmul_f32_z( svptrue_b32(), z0, zalpha );
				z1 = svmul_f32_z( svptrue_b32(), z1, zalpha );
				z2 = svmul_f32_z( svptrue_b32(), z2, zalpha );
				z3 = svmul_f32_z( svptrue_b32(), z3, zalpha );

				z400 = svcreate2( z0, z1 );
				svst1_f32_x2( svptrue_c32(),
					&c_[result_tile_TL_corner + ( tcol + 3 ) * cs_c], z400 );

				z600 = svcreate2( z2, z3 );
				svst1_f32_x2( svptrue_c32(),
					&c_[result_tile_TR_corner + ( tcol + 3 ) * cs_c], z600 );
			}
		}

		// beta != 0
		else
		{
			for ( uint64_t tcol = 0; tcol < SVL; tcol += 4 )
			{
				// Read ZA slices into Z regs
				svfloat32_t z0 = svread_ver_za32_m( z0, svptrue_b32(),
					/* tile: */ 0, /* slice: */ tcol + 0 );
				svfloat32_t z1 = svread_ver_za32_m( z1, svptrue_b32(),
					/* tile: */ 1, /* slice: */ tcol + 0 );
				svfloat32_t z2 = svread_ver_za32_m( z2, svptrue_b32(),
					/* tile: */ 2, /* slice: */ tcol + 0 );
				svfloat32_t z3 = svread_ver_za32_m( z3, svptrue_b32(),
					/* tile: */ 3, /* slice: */ tcol + 0 );

				// Scale Z regs by broadcast alpha
				svfloat32_t z00 = svmul_f32_z( svptrue_b32(), z0, zalpha );
				svfloat32_t z10 = svmul_f32_z( svptrue_b32(), z1, zalpha );
				svfloat32_t z20 = svmul_f32_z( svptrue_b32(), z2, zalpha );
				svfloat32_t z30 = svmul_f32_z( svptrue_b32(), z3, zalpha );

				// Load C into Z regs
				svfloat32x2_t zq5 = svld1_f32_x2( svptrue_c32(),
					&c_[result_tile_TL_corner + ( ( ( tcol + 0 ) * cs_c ) )] );
				svfloat32x2_t zq6 = svld1_f32_x2( svptrue_c32(),
					&c_[result_tile_TR_corner + ( ( ( tcol + 0 ) * cs_c ) )] );

				// Scale Z regs by broadcast beta
				svfloat32_t z40 = svmla_m( svptrue_b32(), z00, svget2( zq5, 0 ),
					zbeta );
				svfloat32_t z50 = svmla_m( svptrue_b32(), z10, svget2( zq5, 1 ),
					zbeta );
				svfloat32_t z60 = svmla_m( svptrue_b32(), z20, svget2( zq6, 0 ),
					zbeta );
				svfloat32_t z70 = svmla_m( svptrue_b32(), z30, svget2( zq6, 1 ),
					zbeta );

				// Store full result into C
				svfloat32x2_t z400 = svcreate2( z40, z50 );
				svst1_f32_x2( svptrue_c32(),
					&c_[result_tile_TL_corner + ( tcol + 0 ) * cs_c], z400 );

				svfloat32x2_t z600 = svcreate2( z60, z70 );
				svst1_f32_x2( svptrue_c32(),
					&c_[result_tile_TR_corner + ( tcol + 0 ) * cs_c], z600 );

				// Repeat unfolded x4
				svfloat32_t z01 = svread_ver_za32_m( z01, svptrue_b32(),
					/* tile: */ 0, /* slice: */ tcol + 1 );
				svfloat32_t z11 = svread_ver_za32_m( z11, svptrue_b32(),
					/* tile: */ 1, /* slice: */ tcol + 1 );
				svfloat32_t z21 = svread_ver_za32_m( z21, svptrue_b32(),
					/* tile: */ 2, /* slice: */ tcol + 1 );
				svfloat32_t z31 = svread_ver_za32_m( z31, svptrue_b32(),
					/* tile: */ 3, /* slice: */ tcol + 1 );

				svfloat32_t z02 = svmul_f32_z( svptrue_b32(), z01, zalpha );
				svfloat32_t z12 = svmul_f32_z( svptrue_b32(), z11, zalpha );
				svfloat32_t z22 = svmul_f32_z( svptrue_b32(), z21, zalpha );
				svfloat32_t z32 = svmul_f32_z( svptrue_b32(), z31, zalpha );

				svfloat32x2_t zq51 = svld1_f32_x2( svptrue_c32(),
					&c_[result_tile_TL_corner + ( ( ( tcol + 1 ) * cs_c ) )] );
				svfloat32x2_t zq61 = svld1_f32_x2( svptrue_c32(),
					&c_[result_tile_TR_corner + ( ( ( tcol + 1 ) * cs_c ) )] );

				svfloat32_t z401 = svmla_m( svptrue_b32(), z02,
					svget2( zq51, 0 ), zbeta );
				svfloat32_t z501 = svmla_m( svptrue_b32(), z12,
					svget2( zq51, 1 ), zbeta );
				svfloat32_t z601 = svmla_m( svptrue_b32(), z22,
					svget2( zq61, 0 ), zbeta );
				svfloat32_t z701 = svmla_m( svptrue_b32(), z32,
					svget2( zq61, 1 ), zbeta );

				svfloat32x2_t z4001 = svcreate2( z401, z501 );
				svst1_f32_x2( svptrue_c32(),
					&c_[result_tile_TL_corner + ( tcol + 1 ) * cs_c], z4001 );

				svfloat32x2_t z6001 = svcreate2( z601, z701 );
				svst1_f32_x2( svptrue_c32(),
					&c_[result_tile_TR_corner + ( tcol + 1 ) * cs_c], z6001 );

				svfloat32_t z03 = svread_ver_za32_m( z03, svptrue_b32(),
					/* tile: */ 0, /* slice: */ tcol + 2 );
				svfloat32_t z13 = svread_ver_za32_m( z13, svptrue_b32(),
					/* tile: */ 1, /* slice: */ tcol + 2 );
				svfloat32_t z23 = svread_ver_za32_m( z23, svptrue_b32(),
					/* tile: */ 2, /* slice: */ tcol + 2 );
				svfloat32_t z33 = svread_ver_za32_m( z33, svptrue_b32(),
					/* tile: */ 3, /* slice: */ tcol + 2 );

				svfloat32_t z04 = svmul_f32_z( svptrue_b32(), z03, zalpha );
				svfloat32_t z14 = svmul_f32_z( svptrue_b32(), z13, zalpha );
				svfloat32_t z24 = svmul_f32_z( svptrue_b32(), z23, zalpha );
				svfloat32_t z34 = svmul_f32_z( svptrue_b32(), z33, zalpha );

				svfloat32x2_t zq52 = svld1_f32_x2( svptrue_c32(),
					&c_[result_tile_TL_corner + ( ( ( tcol + 2 ) * cs_c ) )] );
				svfloat32x2_t zq62 = svld1_f32_x2( svptrue_c32(),
					&c_[result_tile_TR_corner + ( ( ( tcol + 2 ) * cs_c ) )] );

				svfloat32_t z402 = svmla_m( svptrue_b32(), z04,
					svget2( zq52, 0 ), zbeta );
				svfloat32_t z502 = svmla_m( svptrue_b32(), z14,
					svget2( zq52, 1 ), zbeta );
				svfloat32_t z602 = svmla_m( svptrue_b32(), z24,
					svget2( zq62, 0 ), zbeta );
				svfloat32_t z702 = svmla_m( svptrue_b32(), z34,
					svget2( zq62, 1 ), zbeta );

				svfloat32x2_t z4002 = svcreate2( z402, z502 );
				svst1_f32_x2( svptrue_c32(),
					&c_[result_tile_TL_corner + ( tcol + 2 ) * cs_c], z4002 );

				svfloat32x2_t z6002 = svcreate2( z602, z702 );
				svst1_f32_x2( svptrue_c32(),
					&c_[result_tile_TR_corner + ( tcol + 2 ) * cs_c], z6002 );

				svfloat32_t z05 = svread_ver_za32_m( z05, svptrue_b32(),
					/* tile: */ 0, /* slice: */ tcol + 3 );
				svfloat32_t z15 = svread_ver_za32_m( z15, svptrue_b32(),
					/* tile: */ 1, /* slice: */ tcol + 3 );
				svfloat32_t z25 = svread_ver_za32_m( z25, svptrue_b32(),
					/* tile: */ 2, /* slice: */ tcol + 3 );
				svfloat32_t z35 = svread_ver_za32_m( z35, svptrue_b32(),
					/* tile: */ 3, /* slice: */ tcol + 3 );

				svfloat32_t z06 = svmul_f32_z( svptrue_b32(), z05, zalpha );
				svfloat32_t z16 = svmul_f32_z( svptrue_b32(), z15, zalpha );
				svfloat32_t z26 = svmul_f32_z( svptrue_b32(), z25, zalpha );
				svfloat32_t z36 = svmul_f32_z( svptrue_b32(), z35, zalpha );

				svfloat32x2_t zq53 = svld1_f32_x2( svptrue_c32(),
					&c_[result_tile_TL_corner + ( ( ( tcol + 3 ) * cs_c ) )] );
				svfloat32x2_t zq63 = svld1_f32_x2( svptrue_c32(),
					&c_[result_tile_TR_corner + ( ( ( tcol + 3 ) * cs_c ) )] );

				svfloat32_t z403 = svmla_m( svptrue_b32(), z06,
					svget2( zq53, 0 ), zbeta );
				svfloat32_t z503 = svmla_m( svptrue_b32(), z16,
					svget2( zq53, 1 ), zbeta );
				svfloat32_t z603 = svmla_m( svptrue_b32(), z26,
					svget2( zq63, 0 ), zbeta );
				svfloat32_t z703 = svmla_m( svptrue_b32(), z36,
					svget2( zq63, 1 ), zbeta );

				svfloat32x2_t z4003 = svcreate2( z403, z503 );
				svst1_f32_x2( svptrue_c32(),
					&c_[result_tile_TL_corner + ( tcol + 3 ) * cs_c], z4003 );

				svfloat32x2_t z6003 = svcreate2( z603, z703 );
				svst1_f32_x2( svptrue_c32(),
					&c_[result_tile_TR_corner + ( tcol + 3 ) * cs_c], z6003 );
			}
		}
	}
	else
	{
		const uint64_t result_tile_BL_corner = SVL * rs_c;

		if ( beta_ == 0 )
		{
			for ( uint64_t tcol = 0; tcol < SVL; tcol += 4 )
			{
				// Read ZA slices into Z regs
				svfloat32_t z0 = svread_hor_za32_m( z0, svptrue_b32(),
					/* tile: */ 0, /* slice: */ tcol + 0 );
				svfloat32_t z1 = svread_hor_za32_m( z1, svptrue_b32(),
					/* tile: */ 1, /* slice: */ tcol + 0 );
				svfloat32_t z2 = svread_hor_za32_m( z2, svptrue_b32(),
					/* tile: */ 2, /* slice: */ tcol + 0 );
				svfloat32_t z3 = svread_hor_za32_m( z3, svptrue_b32(),
					/* tile: */ 3, /* slice: */ tcol + 0 );

				// Scale Z regs by broadcast alpha
				z0 = svmul_f32_z( svptrue_b32(), z0, zalpha );
				z1 = svmul_f32_z( svptrue_b32(), z1, zalpha );
				z2 = svmul_f32_z( svptrue_b32(), z2, zalpha );
				z3 = svmul_f32_z( svptrue_b32(), z3, zalpha );

				// Store full result into C
				svfloat32x2_t z400 = svcreate2( z0, z2 );
				svst1_f32_x2( svptrue_c32(),
					&c_[result_tile_TL_corner + ( tcol + 0 ) * rs_c], z400 );

				svfloat32x2_t z600 = svcreate2( z1, z3 );
				svst1_f32_x2( svptrue_c32(),
					&c_[result_tile_BL_corner + ( tcol + 0 ) * rs_c], z600 );

				// Repeat unfolded x4
				z0 = svread_hor_za32_m( z0, svptrue_b32(),
					/* tile: */ 0, /* slice: */ tcol + 1 );
				z1 = svread_hor_za32_m( z1, svptrue_b32(),
					/* tile: */ 1, /* slice: */ tcol + 1 );
				z2 = svread_hor_za32_m( z2, svptrue_b32(),
					/* tile: */ 2, /* slice: */ tcol + 1 );
				z3 = svread_hor_za32_m( z3, svptrue_b32(),
					/* tile: */ 3, /* slice: */ tcol + 1 );

				z0 = svmul_f32_z( svptrue_b32(), z0, zalpha );
				z1 = svmul_f32_z( svptrue_b32(), z1, zalpha );
				z2 = svmul_f32_z( svptrue_b32(), z2, zalpha );
				z3 = svmul_f32_z( svptrue_b32(), z3, zalpha );

				z400 = svcreate2( z0, z2 );
				svst1_f32_x2( svptrue_c32(),
					&c_[result_tile_TL_corner + ( tcol + 1 ) * rs_c], z400 );

				z600 = svcreate2( z1, z3 );
				svst1_f32_x2( svptrue_c32(),
					&c_[result_tile_BL_corner + ( tcol + 1 ) * rs_c], z600 );

				z0 = svread_hor_za32_m( z0, svptrue_b32(),
					/* tile: */ 0, /* slice: */ tcol + 2 );
				z1 = svread_hor_za32_m( z1, svptrue_b32(),
					/* tile: */ 1, /* slice: */ tcol + 2 );
				z2 = svread_hor_za32_m( z2, svptrue_b32(),
					/* tile: */ 2, /* slice: */ tcol + 2 );
				z3 = svread_hor_za32_m( z3, svptrue_b32(),
					/* tile: */ 3, /* slice: */ tcol + 2 );

				z0 = svmul_f32_z( svptrue_b32(), z0, zalpha );
				z1 = svmul_f32_z( svptrue_b32(), z1, zalpha );
				z2 = svmul_f32_z( svptrue_b32(), z2, zalpha );
				z3 = svmul_f32_z( svptrue_b32(), z3, zalpha );

				z400 = svcreate2( z0, z2 );
				svst1_f32_x2( svptrue_c32(),
					&c_[result_tile_TL_corner + ( tcol + 2 ) * rs_c], z400 );

				z600 = svcreate2( z1, z3 );
				svst1_f32_x2( svptrue_c32(),
					&c_[result_tile_BL_corner + ( tcol + 2 ) * rs_c], z600 );

				z0 = svread_hor_za32_m( z0, svptrue_b32(),
					/* tile: */ 0, /* slice: */ tcol + 3 );
				z1 = svread_hor_za32_m( z1, svptrue_b32(),
					/* tile: */ 1, /* slice: */ tcol + 3 );
				z2 = svread_hor_za32_m( z2, svptrue_b32(),
					/* tile: */ 2, /* slice: */ tcol + 3 );
				z3 = svread_hor_za32_m( z3, svptrue_b32(),
					/* tile: */ 3, /* slice: */ tcol + 3 );

				z0 = svmul_f32_z( svptrue_b32(), z0, zalpha );
				z1 = svmul_f32_z( svptrue_b32(), z1, zalpha );
				z2 = svmul_f32_z( svptrue_b32(), z2, zalpha );
				z3 = svmul_f32_z( svptrue_b32(), z3, zalpha );

				z400 = svcreate2( z0, z2 );
				svst1_f32_x2( svptrue_c32(),
					&c_[result_tile_TL_corner + ( tcol + 3 ) * rs_c], z400 );

				z600 = svcreate2( z1, z3 );
				svst1_f32_x2( svptrue_c32(),
					&c_[result_tile_BL_corner + ( tcol + 3 ) * rs_c], z600 );
			}
		}
		else
		{
			for ( uint64_t tcol = 0; tcol < SVL; tcol += 4 )
			{
				// Read ZA slices into Z regs
				svfloat32_t z0 = svread_hor_za32_m( z0, svptrue_b32(),
					/* tile: */ 0, /* slice: */ tcol + 0 );
				svfloat32_t z1 = svread_hor_za32_m( z1, svptrue_b32(),
					/* tile: */ 1, /* slice: */ tcol + 0 );
				svfloat32_t z2 = svread_hor_za32_m( z2, svptrue_b32(),
					/* tile: */ 2, /* slice: */ tcol + 0 );
				svfloat32_t z3 = svread_hor_za32_m( z3, svptrue_b32(),
					/* tile: */ 3, /* slice: */ tcol + 0 );

				// Scale Z regs by broadcast alpha
				svfloat32_t z00 = svmul_f32_z( svptrue_b32(), z0, zalpha );
				svfloat32_t z10 = svmul_f32_z( svptrue_b32(), z1, zalpha );
				svfloat32_t z20 = svmul_f32_z( svptrue_b32(), z2, zalpha );
				svfloat32_t z30 = svmul_f32_z( svptrue_b32(), z3, zalpha );

				// Load C into Z regs
				svfloat32x2_t zq5 = svld1_f32_x2( svptrue_c32(),
					&c_[result_tile_TL_corner + ( ( ( tcol + 0 ) * rs_c ) )] );
				svfloat32x2_t zq6 = svld1_f32_x2( svptrue_c32(),
					&c_[result_tile_BL_corner + ( ( ( tcol + 0 ) * rs_c ) )] );

				// Scale Z regs by broadcast beta
				svfloat32_t z40 = svmla_m( svptrue_b32(), z00, svget2( zq5, 0 ),
					zbeta );
				svfloat32_t z50 = svmla_m( svptrue_b32(), z10, svget2( zq6, 0 ),
					zbeta );
				svfloat32_t z60 = svmla_m( svptrue_b32(), z20, svget2( zq5, 1 ),
					zbeta );
				svfloat32_t z70 = svmla_m( svptrue_b32(), z30, svget2( zq6, 1 ),
					zbeta );

				// Store full result into C
				svfloat32x2_t z400 = svcreate2( z40, z60 );
				svst1_f32_x2( svptrue_c32(),
					&c_[result_tile_TL_corner + ( tcol + 0 ) * rs_c], z400 );

				svfloat32x2_t z600 = svcreate2( z50, z70 );
				svst1_f32_x2( svptrue_c32(),
					&c_[result_tile_BL_corner + ( tcol + 0 ) * rs_c], z600 );

				// Repeat unfolded x4
				svfloat32_t z01 = svread_hor_za32_m( z01, svptrue_b32(),
					/* tile: */ 0, /* slice: */ tcol + 1 );
				svfloat32_t z11 = svread_hor_za32_m( z11, svptrue_b32(),
					/* tile: */ 1, /* slice: */ tcol + 1 );
				svfloat32_t z21 = svread_hor_za32_m( z21, svptrue_b32(),
					/* tile: */ 2, /* slice: */ tcol + 1 );
				svfloat32_t z31 = svread_hor_za32_m( z31, svptrue_b32(),
					/* tile: */ 3, /* slice: */ tcol + 1 );

				svfloat32_t z02 = svmul_f32_z( svptrue_b32(), z01, zalpha );
				svfloat32_t z12 = svmul_f32_z( svptrue_b32(), z11, zalpha );
				svfloat32_t z22 = svmul_f32_z( svptrue_b32(), z21, zalpha );
				svfloat32_t z32 = svmul_f32_z( svptrue_b32(), z31, zalpha );

				svfloat32x2_t zq51 = svld1_f32_x2( svptrue_c32(),
					&c_[result_tile_TL_corner + ( ( ( tcol + 1 ) * rs_c ) )] );
				svfloat32x2_t zq61 = svld1_f32_x2( svptrue_c32(),
					&c_[result_tile_BL_corner + ( ( ( tcol + 1 ) * rs_c ) )] );

				svfloat32_t z401 = svmla_m( svptrue_b32(), z02,
					svget2( zq51, 0 ), zbeta );
				svfloat32_t z501 = svmla_m( svptrue_b32(), z12,
					svget2( zq61, 0 ), zbeta );
				svfloat32_t z601 = svmla_m( svptrue_b32(), z22,
					svget2( zq51, 1 ), zbeta );
				svfloat32_t z701 = svmla_m( svptrue_b32(), z32,
					svget2( zq61, 1 ), zbeta );

				svfloat32x2_t z4001 = svcreate2( z401, z601 );
				svst1_f32_x2( svptrue_c32(),
					&c_[result_tile_TL_corner + ( tcol + 1 ) * rs_c], z4001 );

				svfloat32x2_t z6001 = svcreate2( z501, z701 );
				svst1_f32_x2( svptrue_c32(),
					&c_[result_tile_BL_corner + ( tcol + 1 ) * rs_c], z6001 );

				svfloat32_t z03 = svread_hor_za32_m( z03, svptrue_b32(),
					/* tile: */ 0, /* slice: */ tcol + 2 );
				svfloat32_t z13 = svread_hor_za32_m( z13, svptrue_b32(),
					/* tile: */ 1, /* slice: */ tcol + 2 );
				svfloat32_t z23 = svread_hor_za32_m( z23, svptrue_b32(),
					/* tile: */ 2, /* slice: */ tcol + 2 );
				svfloat32_t z33 = svread_hor_za32_m( z33, svptrue_b32(),
					/* tile: */ 3, /* slice: */ tcol + 2 );

				svfloat32_t z04 = svmul_f32_z( svptrue_b32(), z03, zalpha );
				svfloat32_t z14 = svmul_f32_z( svptrue_b32(), z13, zalpha );
				svfloat32_t z24 = svmul_f32_z( svptrue_b32(), z23, zalpha );
				svfloat32_t z34 = svmul_f32_z( svptrue_b32(), z33, zalpha );

				svfloat32x2_t zq52 = svld1_f32_x2( svptrue_c32(),
					&c_[result_tile_TL_corner + ( ( ( tcol + 2 ) * rs_c ) )] );
				svfloat32x2_t zq62 = svld1_f32_x2( svptrue_c32(),
					&c_[result_tile_BL_corner + ( ( ( tcol + 2 ) * rs_c ) )] );

				svfloat32_t z402 = svmla_m( svptrue_b32(), z04,
					svget2( zq52, 0 ), zbeta );
				svfloat32_t z502 = svmla_m( svptrue_b32(), z14,
					svget2( zq62, 0 ), zbeta );
				svfloat32_t z602 = svmla_m( svptrue_b32(), z24,
					svget2( zq52, 1 ), zbeta );
				svfloat32_t z702 = svmla_m( svptrue_b32(), z34,
					svget2( zq62, 1 ), zbeta );

				svfloat32x2_t z4002 = svcreate2( z402, z602 );
				svst1_f32_x2( svptrue_c32(),
					&c_[result_tile_TL_corner + ( tcol + 2 ) * rs_c], z4002 );

				svfloat32x2_t z6002 = svcreate2( z502, z702 );
				svst1_f32_x2( svptrue_c32(),
					&c_[result_tile_BL_corner + ( tcol + 2 ) * rs_c], z6002 );

				svfloat32_t z05 = svread_hor_za32_m( z05, svptrue_b32(),
					/* tile: */ 0, /* slice: */ tcol + 3 );
				svfloat32_t z15 = svread_hor_za32_m( z15, svptrue_b32(),
					/* tile: */ 1, /* slice: */ tcol + 3 );
				svfloat32_t z25 = svread_hor_za32_m( z25, svptrue_b32(),
					/* tile: */ 2, /* slice: */ tcol + 3 );
				svfloat32_t z35 = svread_hor_za32_m( z35, svptrue_b32(),
					/* tile: */ 3, /* slice: */ tcol + 3 );

				svfloat32_t z06 = svmul_f32_z( svptrue_b32(), z05, zalpha );
				svfloat32_t z16 = svmul_f32_z( svptrue_b32(), z15, zalpha );
				svfloat32_t z26 = svmul_f32_z( svptrue_b32(), z25, zalpha );
				svfloat32_t z36 = svmul_f32_z( svptrue_b32(), z35, zalpha );

				svfloat32x2_t zq53 = svld1_f32_x2( svptrue_c32(),
					&c_[result_tile_TL_corner + ( ( ( tcol + 3 ) * rs_c ) )] );
				svfloat32x2_t zq63 = svld1_f32_x2( svptrue_c32(),
					&c_[result_tile_BL_corner + ( ( ( tcol + 3 ) * rs_c ) )] );

				svfloat32_t z403 = svmla_m( svptrue_b32(), z06,
					svget2( zq53, 0 ), zbeta );
				svfloat32_t z503 = svmla_m( svptrue_b32(), z16,
					svget2( zq63, 0 ), zbeta );
				svfloat32_t z603 = svmla_m( svptrue_b32(), z26,
					svget2( zq53, 1 ), zbeta );
				svfloat32_t z703 = svmla_m( svptrue_b32(), z36,
					svget2( zq63, 1 ), zbeta );

				svfloat32x2_t z4003 = svcreate2( z403, z603 );
				svst1_f32_x2( svptrue_c32(),
					&c_[result_tile_TL_corner + ( tcol + 3 ) * rs_c], z4003 );

				svfloat32x2_t z6003 = svcreate2( z503, z703 );
				svst1_f32_x2( svptrue_c32(),
					&c_[result_tile_BL_corner + ( tcol + 3 ) * rs_c], z6003 );
			}
		}
	}

	GEMM_UKR_FLUSH_CT( s );

	return;
}

__arm_new( "za" ) __arm_locally_streaming void bli_dgemm_armsme_int_4SVLx2SVL
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
	uint64_t SVL = svcntsd();

	GEMM_UKR_SETUP_CT_AMBI( d, 4 * SVL, 2 * SVL, false );

	double *a_ = (double *)a;
	double *b_ = (double *)b;
	double *c_ = (double *)c;

	svzero_za();

	uint64_t k_;
	uint64_t k_iter = k / 4;
	uint64_t k_left = k % 4;

	for ( k_ = 0; k_ < k_iter; k_++ )
	{
		// Loads
		svfloat64x4_t zL00 = svld1_f64_x4( svptrue_c32(),
			(float64_t *)( &a_[0] ) );
		svfloat64x2_t zR00 = svld1_f64_x2( svptrue_c32(),
			(float64_t *)( &b_[0] ) );

		svmopa_za64_m( 0, svptrue_b32(), svptrue_b32(), svget4( zL00, 0 ),
			svget2( zR00, 0 ) );
		svmopa_za64_m( 1, svptrue_b32(), svptrue_b32(), svget4( zL00, 1 ),
			svget2( zR00, 0 ) );

		svmopa_za64_m( 2, svptrue_b32(), svptrue_b32(), svget4( zL00, 2 ),
			svget2( zR00, 0 ) );
		svmopa_za64_m( 3, svptrue_b32(), svptrue_b32(), svget4( zL00, 3 ),
			svget2( zR00, 0 ) );

		svmopa_za64_m( 4, svptrue_b32(), svptrue_b32(), svget4( zL00, 0 ),
			svget2( zR00, 1 ) );
		svmopa_za64_m( 5, svptrue_b32(), svptrue_b32(), svget4( zL00, 1 ),
			svget2( zR00, 1 ) );

		svmopa_za64_m( 6, svptrue_b32(), svptrue_b32(), svget4( zL00, 2 ),
			svget2( zR00, 1 ) );
		svmopa_za64_m( 7, svptrue_b32(), svptrue_b32(), svget4( zL00, 3 ),
			svget2( zR00, 1 ) );

		svfloat64x4_t zL01 = svld1_f64_x4( svptrue_c32(),
			(float64_t *)( &a_[( 4 * SVL )] ) );
		svfloat64x2_t zR01 = svld1_f64_x2( svptrue_c32(),
			(float64_t *)( &b_[( 2 * SVL )] ) );

		svmopa_za64_m( 0, svptrue_b32(), svptrue_b32(), svget4( zL01, 0 ),
			svget2( zR01, 0 ) );
		svmopa_za64_m( 1, svptrue_b32(), svptrue_b32(), svget4( zL01, 1 ),
			svget2( zR01, 0 ) );

		svmopa_za64_m( 2, svptrue_b32(), svptrue_b32(), svget4( zL01, 2 ),
			svget2( zR01, 0 ) );
		svmopa_za64_m( 3, svptrue_b32(), svptrue_b32(), svget4( zL01, 3 ),
			svget2( zR01, 0 ) );

		svmopa_za64_m( 4, svptrue_b32(), svptrue_b32(), svget4( zL01, 0 ),
			svget2( zR01, 1 ) );
		svmopa_za64_m( 5, svptrue_b32(), svptrue_b32(), svget4( zL01, 1 ),
			svget2( zR01, 1 ) );

		svmopa_za64_m( 6, svptrue_b32(), svptrue_b32(), svget4( zL01, 2 ),
			svget2( zR01, 1 ) );
		svmopa_za64_m( 7, svptrue_b32(), svptrue_b32(), svget4( zL01, 3 ),
			svget2( zR01, 1 ) );

		svfloat64x4_t zL02 = svld1_f64_x4( svptrue_c32(),
			(float64_t *)( &a_[( 8 * SVL )] ) );
		svfloat64x2_t zR02 = svld1_f64_x2( svptrue_c32(),
			(float64_t *)( &b_[( 4 * SVL )] ) );

		svmopa_za64_m( 0, svptrue_b32(), svptrue_b32(), svget4( zL02, 0 ),
			svget2( zR02, 0 ) );
		svmopa_za64_m( 1, svptrue_b32(), svptrue_b32(), svget4( zL02, 1 ),
			svget2( zR02, 0 ) );

		svmopa_za64_m( 2, svptrue_b32(), svptrue_b32(), svget4( zL02, 2 ),
			svget2( zR02, 0 ) );
		svmopa_za64_m( 3, svptrue_b32(), svptrue_b32(), svget4( zL02, 3 ),
			svget2( zR02, 0 ) );

		svmopa_za64_m( 4, svptrue_b32(), svptrue_b32(), svget4( zL02, 0 ),
			svget2( zR02, 1 ) );
		svmopa_za64_m( 5, svptrue_b32(), svptrue_b32(), svget4( zL02, 1 ),
			svget2( zR02, 1 ) );

		svmopa_za64_m( 6, svptrue_b32(), svptrue_b32(), svget4( zL02, 2 ),
			svget2( zR02, 1 ) );
		svmopa_za64_m( 7, svptrue_b32(), svptrue_b32(), svget4( zL02, 3 ),
			svget2( zR02, 1 ) );

		svfloat64x4_t zL03 = svld1_f64_x4( svptrue_c32(),
			(float64_t *)( &a_[( 12 * SVL )] ) );
		svfloat64x2_t zR03 = svld1_f64_x2( svptrue_c32(),
			(float64_t *)( &b_[( 6 * SVL )] ) );

		svmopa_za64_m( 0, svptrue_b32(), svptrue_b32(), svget4( zL03, 0 ),
			svget2( zR03, 0 ) );
		svmopa_za64_m( 1, svptrue_b32(), svptrue_b32(), svget4( zL03, 1 ),
			svget2( zR03, 0 ) );

		svmopa_za64_m( 2, svptrue_b32(), svptrue_b32(), svget4( zL03, 2 ),
			svget2( zR03, 0 ) );
		svmopa_za64_m( 3, svptrue_b32(), svptrue_b32(), svget4( zL03, 3 ),
			svget2( zR03, 0 ) );

		svmopa_za64_m( 4, svptrue_b32(), svptrue_b32(), svget4( zL03, 0 ),
			svget2( zR03, 1 ) );
		svmopa_za64_m( 5, svptrue_b32(), svptrue_b32(), svget4( zL03, 1 ),
			svget2( zR03, 1 ) );

		svmopa_za64_m( 6, svptrue_b32(), svptrue_b32(), svget4( zL03, 2 ),
			svget2( zR03, 1 ) );
		svmopa_za64_m( 7, svptrue_b32(), svptrue_b32(), svget4( zL03, 3 ),
			svget2( zR03, 1 ) );

		a_ += ( 2 * 8 * SVL );
		b_ += ( 8 * SVL );
	}

	for ( k_ = 0; k_ < k_left; k_ += 1 )
	{
		svfloat64x4_t zL00 = svld1_f64_x4( svptrue_c32(),
			(float64_t *)( &a_[0] ) );
		svfloat64x2_t zR00 = svld1_f64_x2( svptrue_c32(),
			(float64_t *)( &b_[0] ) );

		svmopa_za64_m( 0, svptrue_b32(), svptrue_b32(), svget4( zL00, 0 ),
			svget2( zR00, 0 ) );
		svmopa_za64_m( 1, svptrue_b32(), svptrue_b32(), svget4( zL00, 1 ),
			svget2( zR00, 0 ) );

		svmopa_za64_m( 2, svptrue_b32(), svptrue_b32(), svget4( zL00, 2 ),
			svget2( zR00, 0 ) );
		svmopa_za64_m( 3, svptrue_b32(), svptrue_b32(), svget4( zL00, 3 ),
			svget2( zR00, 0 ) );

		svmopa_za64_m( 4, svptrue_b32(), svptrue_b32(), svget4( zL00, 0 ),
			svget2( zR00, 1 ) );
		svmopa_za64_m( 5, svptrue_b32(), svptrue_b32(), svget4( zL00, 1 ),
			svget2( zR00, 1 ) );

		svmopa_za64_m( 6, svptrue_b32(), svptrue_b32(), svget4( zL00, 2 ),
			svget2( zR00, 1 ) );
		svmopa_za64_m( 7, svptrue_b32(), svptrue_b32(), svget4( zL00, 3 ),
			svget2( zR00, 1 ) );

		a_ += ( 4 * SVL );
		b_ += ( 2 * SVL );
	}

	double beta_ = *(double *)beta;
	double alpha_ = *(double *)alpha;

	const uint64_t result_tile_TL_corner = 0;

	svfloat64_t zbeta = svdup_f64( beta_ );
	svfloat64_t zalpha = svdup_f64( alpha_ );

	if ( rs_c == 1 )
	{
		const uint64_t result_tile_TR_corner = SVL * cs_c;

		if ( beta_ == 0 )
		{
			for ( uint64_t tcol = 0; tcol < SVL; tcol += 4 )
			{
				// Read ZA slices into Z regs
				svfloat64_t z0 = svread_ver_za64_m( z0, svptrue_b32(),
					/* tile: */ 0, /* slice: */ tcol + 0 );
				svfloat64_t z1 = svread_ver_za64_m( z1, svptrue_b32(),
					/* tile: */ 1, /* slice: */ tcol + 0 );
				svfloat64_t z2 = svread_ver_za64_m( z2, svptrue_b32(),
					/* tile: */ 2, /* slice: */ tcol + 0 );
				svfloat64_t z3 = svread_ver_za64_m( z3, svptrue_b32(),
					/* tile: */ 3, /* slice: */ tcol + 0 );
				svfloat64_t z4 = svread_ver_za64_m( z3, svptrue_b32(),
					/* tile: */ 4, /* slice: */ tcol + 0 );
				svfloat64_t z5 = svread_ver_za64_m( z3, svptrue_b32(),
					/* tile: */ 5, /* slice: */ tcol + 0 );
				svfloat64_t z6 = svread_ver_za64_m( z3, svptrue_b32(),
					/* tile: */ 6, /* slice: */ tcol + 0 );
				svfloat64_t z7 = svread_ver_za64_m( z3, svptrue_b32(),
					/* tile: */ 7, /* slice: */ tcol + 0 );

				// Scale Z regs by broadcast alpha
				z0 = svmul_f64_z( svptrue_b32(), z0, zalpha );
				z1 = svmul_f64_z( svptrue_b32(), z1, zalpha );
				z2 = svmul_f64_z( svptrue_b32(), z2, zalpha );
				z3 = svmul_f64_z( svptrue_b32(), z3, zalpha );
				z4 = svmul_f64_z( svptrue_b32(), z4, zalpha );
				z5 = svmul_f64_z( svptrue_b32(), z5, zalpha );
				z6 = svmul_f64_z( svptrue_b32(), z6, zalpha );
				z7 = svmul_f64_z( svptrue_b32(), z7, zalpha );

				// Store full result into C
				svfloat64x4_t z400 = svcreate4( z0, z1, z2, z3 );
				svst1_f64_x4( svptrue_c32(),
					&c_[result_tile_TL_corner + ( tcol + 0 ) * cs_c], z400 );

				svfloat64x4_t z600 = svcreate4( z4, z5, z6, z7 );
				svst1_f64_x4( svptrue_c32(),
					&c_[result_tile_TR_corner + ( tcol + 0 ) * cs_c], z600 );

				// tcol + 1
				z0 = svread_ver_za64_m( z0, svptrue_b32(),
					/* tile: */ 0, /* slice: */ tcol + 1 );
				z1 = svread_ver_za64_m( z1, svptrue_b32(),
					/* tile: */ 1, /* slice: */ tcol + 1 );
				z2 = svread_ver_za64_m( z2, svptrue_b32(),
					/* tile: */ 2, /* slice: */ tcol + 1 );
				z3 = svread_ver_za64_m( z3, svptrue_b32(),
					/* tile: */ 3, /* slice: */ tcol + 1 );
				z4 = svread_ver_za64_m( z3, svptrue_b32(),
					/* tile: */ 4, /* slice: */ tcol + 1 );
				z5 = svread_ver_za64_m( z3, svptrue_b32(),
					/* tile: */ 5, /* slice: */ tcol + 1 );
				z6 = svread_ver_za64_m( z3, svptrue_b32(),
					/* tile: */ 6, /* slice: */ tcol + 1 );
				z7 = svread_ver_za64_m( z3, svptrue_b32(),
					/* tile: */ 7, /* slice: */ tcol + 1 );

				z0 = svmul_f64_z( svptrue_b32(), z0, zalpha );
				z1 = svmul_f64_z( svptrue_b32(), z1, zalpha );
				z2 = svmul_f64_z( svptrue_b32(), z2, zalpha );
				z3 = svmul_f64_z( svptrue_b32(), z3, zalpha );
				z4 = svmul_f64_z( svptrue_b32(), z4, zalpha );
				z5 = svmul_f64_z( svptrue_b32(), z5, zalpha );
				z6 = svmul_f64_z( svptrue_b32(), z6, zalpha );
				z7 = svmul_f64_z( svptrue_b32(), z7, zalpha );

				z400 = svcreate4( z0, z1, z2, z3 );
				svst1_f64_x4( svptrue_c32(),
					&c_[result_tile_TL_corner + ( tcol + 1 ) * cs_c], z400 );

				z600 = svcreate4( z4, z5, z6, z7 );
				svst1_f64_x4( svptrue_c32(),
					&c_[result_tile_TR_corner + ( tcol + 1 ) * cs_c], z600 );

				// tcol + 2
				z0 = svread_ver_za64_m( z0, svptrue_b32(),
					/* tile: */ 0, /* slice: */ tcol + 2 );
				z1 = svread_ver_za64_m( z1, svptrue_b32(),
					/* tile: */ 1, /* slice: */ tcol + 2 );
				z2 = svread_ver_za64_m( z2, svptrue_b32(),
					/* tile: */ 2, /* slice: */ tcol + 2 );
				z3 = svread_ver_za64_m( z3, svptrue_b32(),
					/* tile: */ 3, /* slice: */ tcol + 2 );
				z4 = svread_ver_za64_m( z3, svptrue_b32(),
					/* tile: */ 4, /* slice: */ tcol + 2 );
				z5 = svread_ver_za64_m( z3, svptrue_b32(),
					/* tile: */ 5, /* slice: */ tcol + 2 );
				z6 = svread_ver_za64_m( z3, svptrue_b32(),
					/* tile: */ 6, /* slice: */ tcol + 2 );
				z7 = svread_ver_za64_m( z3, svptrue_b32(),
					/* tile: */ 7, /* slice: */ tcol + 2 );

				z0 = svmul_f64_z( svptrue_b32(), z0, zalpha );
				z1 = svmul_f64_z( svptrue_b32(), z1, zalpha );
				z2 = svmul_f64_z( svptrue_b32(), z2, zalpha );
				z3 = svmul_f64_z( svptrue_b32(), z3, zalpha );
				z4 = svmul_f64_z( svptrue_b32(), z4, zalpha );
				z5 = svmul_f64_z( svptrue_b32(), z5, zalpha );
				z6 = svmul_f64_z( svptrue_b32(), z6, zalpha );
				z7 = svmul_f64_z( svptrue_b32(), z7, zalpha );

				z400 = svcreate4( z0, z1, z2, z3 );
				svst1_f64_x4( svptrue_c32(),
					&c_[result_tile_TL_corner + ( tcol + 2 ) * cs_c], z400 );

				z600 = svcreate4( z4, z5, z6, z7 );
				svst1_f64_x4( svptrue_c32(),
					&c_[result_tile_TR_corner + ( tcol + 2 ) * cs_c], z600 );

				// tcol + 3
				z0 = svread_ver_za64_m( z0, svptrue_b32(),
					/* tile: */ 0, /* slice: */ tcol + 3 );
				z1 = svread_ver_za64_m( z1, svptrue_b32(),
					/* tile: */ 1, /* slice: */ tcol + 3 );
				z2 = svread_ver_za64_m( z2, svptrue_b32(),
					/* tile: */ 2, /* slice: */ tcol + 3 );
				z3 = svread_ver_za64_m( z3, svptrue_b32(),
					/* tile: */ 3, /* slice: */ tcol + 3 );
				z4 = svread_ver_za64_m( z3, svptrue_b32(),
					/* tile: */ 4, /* slice: */ tcol + 3 );
				z5 = svread_ver_za64_m( z3, svptrue_b32(),
					/* tile: */ 5, /* slice: */ tcol + 3 );
				z6 = svread_ver_za64_m( z3, svptrue_b32(),
					/* tile: */ 6, /* slice: */ tcol + 3 );
				z7 = svread_ver_za64_m( z3, svptrue_b32(),
					/* tile: */ 7, /* slice: */ tcol + 3 );

				z0 = svmul_f64_z( svptrue_b32(), z0, zalpha );
				z1 = svmul_f64_z( svptrue_b32(), z1, zalpha );
				z2 = svmul_f64_z( svptrue_b32(), z2, zalpha );
				z3 = svmul_f64_z( svptrue_b32(), z3, zalpha );
				z4 = svmul_f64_z( svptrue_b32(), z4, zalpha );
				z5 = svmul_f64_z( svptrue_b32(), z5, zalpha );
				z6 = svmul_f64_z( svptrue_b32(), z6, zalpha );
				z7 = svmul_f64_z( svptrue_b32(), z7, zalpha );

				z400 = svcreate4( z0, z1, z2, z3 );
				svst1_f64_x4( svptrue_c32(),
					&c_[result_tile_TL_corner + ( tcol + 3 ) * cs_c], z400 );

				z600 = svcreate4( z4, z5, z6, z7 );
				svst1_f64_x4( svptrue_c32(),
					&c_[result_tile_TR_corner + ( tcol + 3 ) * cs_c], z600 );
			}
		}
		else
		{
			for ( uint64_t tcol = 0; tcol < SVL; tcol += 4 )
			{
				// Read ZA slices into Z regs
				svfloat64_t z0 = svread_ver_za64_m( z0, svptrue_b32(),
					/* tile: */ 0, /* slice: */ tcol + 0 );
				svfloat64_t z1 = svread_ver_za64_m( z1, svptrue_b32(),
					/* tile: */ 1, /* slice: */ tcol + 0 );
				svfloat64_t z2 = svread_ver_za64_m( z2, svptrue_b32(),
					/* tile: */ 2, /* slice: */ tcol + 0 );
				svfloat64_t z3 = svread_ver_za64_m( z3, svptrue_b32(),
					/* tile: */ 3, /* slice: */ tcol + 0 );
				svfloat64_t z4 = svread_ver_za64_m( z3, svptrue_b32(),
					/* tile: */ 4, /* slice: */ tcol + 0 );
				svfloat64_t z5 = svread_ver_za64_m( z3, svptrue_b32(),
					/* tile: */ 5, /* slice: */ tcol + 0 );
				svfloat64_t z6 = svread_ver_za64_m( z3, svptrue_b32(),
					/* tile: */ 6, /* slice: */ tcol + 0 );
				svfloat64_t z7 = svread_ver_za64_m( z3, svptrue_b32(),
					/* tile: */ 7, /* slice: */ tcol + 0 );

				// Scale Z regs by broadcast alpha
				z0 = svmul_f64_z( svptrue_b32(), z0, zalpha );
				z1 = svmul_f64_z( svptrue_b32(), z1, zalpha );
				z2 = svmul_f64_z( svptrue_b32(), z2, zalpha );
				z3 = svmul_f64_z( svptrue_b32(), z3, zalpha );
				z4 = svmul_f64_z( svptrue_b32(), z4, zalpha );
				z5 = svmul_f64_z( svptrue_b32(), z5, zalpha );
				z6 = svmul_f64_z( svptrue_b32(), z6, zalpha );
				z7 = svmul_f64_z( svptrue_b32(), z7, zalpha );

				// Load C into Z regs
				svfloat64x4_t zq5 = svld1_f64_x4( svptrue_c32(),
					&c_[result_tile_TL_corner + ( ( ( tcol + 0 ) * cs_c ) )] );
				svfloat64x4_t zq6 = svld1_f64_x4( svptrue_c32(),
					&c_[result_tile_TR_corner + ( ( ( tcol + 0 ) * cs_c ) )] );

				// Scale Z regs by broadcast beta
				svfloat64_t z40 = svmla_m( svptrue_b32(), z0, svget4( zq5, 0 ),
					zbeta );
				svfloat64_t z50 = svmla_m( svptrue_b32(), z1, svget4( zq5, 1 ),
					zbeta );
				svfloat64_t z60 = svmla_m( svptrue_b32(), z2, svget4( zq5, 2 ),
					zbeta );
				svfloat64_t z70 = svmla_m( svptrue_b32(), z3, svget4( zq5, 3 ),
					zbeta );
				svfloat64_t z80 = svmla_m( svptrue_b32(), z4, svget4( zq6, 0 ),
					zbeta );
				svfloat64_t z90 = svmla_m( svptrue_b32(), z5, svget4( zq6, 1 ),
					zbeta );
				svfloat64_t za0 = svmla_m( svptrue_b32(), z6, svget4( zq6, 2 ),
					zbeta );
				svfloat64_t zb0 = svmla_m( svptrue_b32(), z7, svget4( zq6, 3 ),
					zbeta );

				// Store full result into C
				svfloat64x4_t z400 = svcreate4( z40, z50, z60, z70 );
				svst1_f64_x4( svptrue_c32(),
					&c_[result_tile_TL_corner + ( tcol + 0 ) * cs_c], z400 );

				svfloat64x4_t z600 = svcreate4( z80, z90, za0, zb0 );
				svst1_f64_x4( svptrue_c32(),
					&c_[result_tile_TR_corner + ( tcol + 0 ) * cs_c], z600 );

				// tcol + 1
				z0 = svread_ver_za64_m( z0, svptrue_b32(),
					/* tile: */ 0, /* slice: */ tcol + 1 );
				z1 = svread_ver_za64_m( z1, svptrue_b32(),
					/* tile: */ 1, /* slice: */ tcol + 1 );
				z2 = svread_ver_za64_m( z2, svptrue_b32(),
					/* tile: */ 2, /* slice: */ tcol + 1 );
				z3 = svread_ver_za64_m( z3, svptrue_b32(),
					/* tile: */ 3, /* slice: */ tcol + 1 );
				z4 = svread_ver_za64_m( z3, svptrue_b32(),
					/* tile: */ 4, /* slice: */ tcol + 1 );
				z5 = svread_ver_za64_m( z3, svptrue_b32(),
					/* tile: */ 5, /* slice: */ tcol + 1 );
				z6 = svread_ver_za64_m( z3, svptrue_b32(),
					/* tile: */ 6, /* slice: */ tcol + 1 );
				z7 = svread_ver_za64_m( z3, svptrue_b32(),
					/* tile: */ 7, /* slice: */ tcol + 1 );

				z0 = svmul_f64_z( svptrue_b32(), z0, zalpha );
				z1 = svmul_f64_z( svptrue_b32(), z1, zalpha );
				z2 = svmul_f64_z( svptrue_b32(), z2, zalpha );
				z3 = svmul_f64_z( svptrue_b32(), z3, zalpha );
				z4 = svmul_f64_z( svptrue_b32(), z4, zalpha );
				z5 = svmul_f64_z( svptrue_b32(), z5, zalpha );
				z6 = svmul_f64_z( svptrue_b32(), z6, zalpha );
				z7 = svmul_f64_z( svptrue_b32(), z7, zalpha );

				zq5 = svld1_f64_x4( svptrue_c32(),
					&c_[result_tile_TL_corner + ( ( ( tcol + 1 ) * cs_c ) )] );
				zq6 = svld1_f64_x4( svptrue_c32(),
					&c_[result_tile_TR_corner + ( ( ( tcol + 1 ) * cs_c ) )] );

				z40 = svmla_m( svptrue_b32(), z0, svget4( zq5, 0 ), zbeta );
				z50 = svmla_m( svptrue_b32(), z1, svget4( zq5, 1 ), zbeta );
				z60 = svmla_m( svptrue_b32(), z2, svget4( zq5, 2 ), zbeta );
				z70 = svmla_m( svptrue_b32(), z3, svget4( zq5, 3 ), zbeta );
				z80 = svmla_m( svptrue_b32(), z4, svget4( zq6, 0 ), zbeta );
				z90 = svmla_m( svptrue_b32(), z5, svget4( zq6, 1 ), zbeta );
				za0 = svmla_m( svptrue_b32(), z6, svget4( zq6, 2 ), zbeta );
				zb0 = svmla_m( svptrue_b32(), z7, svget4( zq6, 3 ), zbeta );

				z400 = svcreate4( z40, z50, z60, z70 );
				svst1_f64_x4( svptrue_c32(),
					&c_[result_tile_TL_corner + ( tcol + 1 ) * cs_c], z400 );

				z600 = svcreate4( z80, z90, za0, zb0 );
				svst1_f64_x4( svptrue_c32(),
					&c_[result_tile_TR_corner + ( tcol + 1 ) * cs_c], z600 );

				// tcol + 2
				z0 = svread_ver_za64_m( z0, svptrue_b32(),
					/* tile: */ 0, /* slice: */ tcol + 2 );
				z1 = svread_ver_za64_m( z1, svptrue_b32(),
					/* tile: */ 1, /* slice: */ tcol + 2 );
				z2 = svread_ver_za64_m( z2, svptrue_b32(),
					/* tile: */ 2, /* slice: */ tcol + 2 );
				z3 = svread_ver_za64_m( z3, svptrue_b32(),
					/* tile: */ 3, /* slice: */ tcol + 2 );
				z4 = svread_ver_za64_m( z3, svptrue_b32(),
					/* tile: */ 4, /* slice: */ tcol + 2 );
				z5 = svread_ver_za64_m( z3, svptrue_b32(),
					/* tile: */ 5, /* slice: */ tcol + 2 );
				z6 = svread_ver_za64_m( z3, svptrue_b32(),
					/* tile: */ 6, /* slice: */ tcol + 2 );
				z7 = svread_ver_za64_m( z3, svptrue_b32(),
					/* tile: */ 7, /* slice: */ tcol + 2 );

				z0 = svmul_f64_z( svptrue_b32(), z0, zalpha );
				z1 = svmul_f64_z( svptrue_b32(), z1, zalpha );
				z2 = svmul_f64_z( svptrue_b32(), z2, zalpha );
				z3 = svmul_f64_z( svptrue_b32(), z3, zalpha );
				z4 = svmul_f64_z( svptrue_b32(), z4, zalpha );
				z5 = svmul_f64_z( svptrue_b32(), z5, zalpha );
				z6 = svmul_f64_z( svptrue_b32(), z6, zalpha );
				z7 = svmul_f64_z( svptrue_b32(), z7, zalpha );

				zq5 = svld1_f64_x4( svptrue_c32(),
					&c_[result_tile_TL_corner + ( ( ( tcol + 2 ) * cs_c ) )] );
				zq6 = svld1_f64_x4( svptrue_c32(),
					&c_[result_tile_TR_corner + ( ( ( tcol + 2 ) * cs_c ) )] );

				z40 = svmla_m( svptrue_b32(), z0, svget4( zq5, 0 ), zbeta );
				z50 = svmla_m( svptrue_b32(), z1, svget4( zq5, 1 ), zbeta );
				z60 = svmla_m( svptrue_b32(), z2, svget4( zq5, 2 ), zbeta );
				z70 = svmla_m( svptrue_b32(), z3, svget4( zq5, 3 ), zbeta );
				z80 = svmla_m( svptrue_b32(), z4, svget4( zq6, 0 ), zbeta );
				z90 = svmla_m( svptrue_b32(), z5, svget4( zq6, 1 ), zbeta );
				za0 = svmla_m( svptrue_b32(), z6, svget4( zq6, 2 ), zbeta );
				zb0 = svmla_m( svptrue_b32(), z7, svget4( zq6, 3 ), zbeta );

				z400 = svcreate4( z40, z50, z60, z70 );
				svst1_f64_x4( svptrue_c32(),
					&c_[result_tile_TL_corner + ( tcol + 2 ) * cs_c], z400 );

				z600 = svcreate4( z80, z90, za0, zb0 );
				svst1_f64_x4( svptrue_c32(),
					&c_[result_tile_TR_corner + ( tcol + 2 ) * cs_c], z600 );

				// tcol + 3
				z0 = svread_ver_za64_m( z0, svptrue_b32(),
					/* tile: */ 0, /* slice: */ tcol + 3 );
				z1 = svread_ver_za64_m( z1, svptrue_b32(),
					/* tile: */ 1, /* slice: */ tcol + 3 );
				z2 = svread_ver_za64_m( z2, svptrue_b32(),
					/* tile: */ 2, /* slice: */ tcol + 3 );
				z3 = svread_ver_za64_m( z3, svptrue_b32(),
					/* tile: */ 3, /* slice: */ tcol + 3 );
				z4 = svread_ver_za64_m( z3, svptrue_b32(),
					/* tile: */ 4, /* slice: */ tcol + 3 );
				z5 = svread_ver_za64_m( z3, svptrue_b32(),
					/* tile: */ 5, /* slice: */ tcol + 3 );
				z6 = svread_ver_za64_m( z3, svptrue_b32(),
					/* tile: */ 6, /* slice: */ tcol + 3 );
				z7 = svread_ver_za64_m( z3, svptrue_b32(),
					/* tile: */ 7, /* slice: */ tcol + 3 );

				z0 = svmul_f64_z( svptrue_b32(), z0, zalpha );
				z1 = svmul_f64_z( svptrue_b32(), z1, zalpha );
				z2 = svmul_f64_z( svptrue_b32(), z2, zalpha );
				z3 = svmul_f64_z( svptrue_b32(), z3, zalpha );
				z4 = svmul_f64_z( svptrue_b32(), z4, zalpha );
				z5 = svmul_f64_z( svptrue_b32(), z5, zalpha );
				z6 = svmul_f64_z( svptrue_b32(), z6, zalpha );
				z7 = svmul_f64_z( svptrue_b32(), z7, zalpha );

				zq5 = svld1_f64_x4( svptrue_c32(),
					&c_[result_tile_TL_corner + ( ( ( tcol + 3 ) * cs_c ) )] );
				zq6 = svld1_f64_x4( svptrue_c32(),
					&c_[result_tile_TR_corner + ( ( ( tcol + 3 ) * cs_c ) )] );

				z40 = svmla_m( svptrue_b32(), z0, svget4( zq5, 0 ), zbeta );
				z50 = svmla_m( svptrue_b32(), z1, svget4( zq5, 1 ), zbeta );
				z60 = svmla_m( svptrue_b32(), z2, svget4( zq5, 2 ), zbeta );
				z70 = svmla_m( svptrue_b32(), z3, svget4( zq5, 3 ), zbeta );
				z80 = svmla_m( svptrue_b32(), z4, svget4( zq6, 0 ), zbeta );
				z90 = svmla_m( svptrue_b32(), z5, svget4( zq6, 1 ), zbeta );
				za0 = svmla_m( svptrue_b32(), z6, svget4( zq6, 2 ), zbeta );
				zb0 = svmla_m( svptrue_b32(), z7, svget4( zq6, 3 ), zbeta );

				z400 = svcreate4( z40, z50, z60, z70 );
				svst1_f64_x4( svptrue_c32(),
					&c_[result_tile_TL_corner + ( tcol + 3 ) * cs_c], z400 );

				z600 = svcreate4( z80, z90, za0, zb0 );
				svst1_f64_x4( svptrue_c32(),
					&c_[result_tile_TR_corner + ( tcol + 3 ) * cs_c], z600 );
			}
		}
	}
	else
	{
		const uint64_t result_tile_TR_corner = SVL * rs_c;
		const uint64_t result_tile_BL_corner = SVL * 2 * rs_c;
		const uint64_t result_tile_BR_corner = SVL * 3 * rs_c;

		if ( beta_ == 0 )
		{
			for ( uint64_t tcol = 0; tcol < SVL; tcol += 4 )
			{
				// Read ZA slices into Z regs
				svfloat64_t z0 = svread_hor_za64_m( z0, svptrue_b32(),
					/* tile: */ 0, /* slice: */ tcol + 0 );
				svfloat64_t z1 = svread_hor_za64_m( z1, svptrue_b32(),
					/* tile: */ 1, /* slice: */ tcol + 0 );
				svfloat64_t z2 = svread_hor_za64_m( z2, svptrue_b32(),
					/* tile: */ 2, /* slice: */ tcol + 0 );
				svfloat64_t z3 = svread_hor_za64_m( z3, svptrue_b32(),
					/* tile: */ 3, /* slice: */ tcol + 0 );
				svfloat64_t z4 = svread_hor_za64_m( z4, svptrue_b32(),
					/* tile: */ 4, /* slice: */ tcol + 0 );
				svfloat64_t z5 = svread_hor_za64_m( z5, svptrue_b32(),
					/* tile: */ 5, /* slice: */ tcol + 0 );
				svfloat64_t z6 = svread_hor_za64_m( z6, svptrue_b32(),
					/* tile: */ 6, /* slice: */ tcol + 0 );
				svfloat64_t z7 = svread_hor_za64_m( z7, svptrue_b32(),
					/* tile: */ 7, /* slice: */ tcol + 0 );

				// Scale Z regs by broadcast alpha
				z0 = svmul_f64_z( svptrue_b32(), z0, zalpha );
				z1 = svmul_f64_z( svptrue_b32(), z1, zalpha );
				z2 = svmul_f64_z( svptrue_b32(), z2, zalpha );
				z3 = svmul_f64_z( svptrue_b32(), z3, zalpha );
				z4 = svmul_f64_z( svptrue_b32(), z4, zalpha );
				z5 = svmul_f64_z( svptrue_b32(), z5, zalpha );
				z6 = svmul_f64_z( svptrue_b32(), z6, zalpha );
				z7 = svmul_f64_z( svptrue_b32(), z7, zalpha );

				// Store full result into C
				svfloat64x2_t z400 = svcreate2( z0, z4 );
				svst1_f64_x2( svptrue_c32(),
					&c_[result_tile_TL_corner + ( tcol + 0 ) * rs_c], z400 );

				svfloat64x2_t z600 = svcreate2( z1, z5 );
				svst1_f64_x2( svptrue_c32(),
					&c_[result_tile_TR_corner + ( tcol + 0 ) * rs_c], z600 );
				svfloat64x2_t z700 = svcreate2( z2, z6 );
				svst1_f64_x2( svptrue_c32(),
					&c_[result_tile_BL_corner + ( tcol + 0 ) * rs_c], z700 );

				svfloat64x2_t z800 = svcreate2( z3, z7 );
				svst1_f64_x2( svptrue_c32(),
					&c_[result_tile_BR_corner + ( tcol + 0 ) * rs_c], z800 );

				// tcol + 1
				z0 = svread_hor_za64_m( z0, svptrue_b32(),
					/* tile: */ 0, /* slice: */ tcol + 1 );
				z1 = svread_hor_za64_m( z1, svptrue_b32(),
					/* tile: */ 1, /* slice: */ tcol + 1 );
				z2 = svread_hor_za64_m( z2, svptrue_b32(),
					/* tile: */ 2, /* slice: */ tcol + 1 );
				z3 = svread_hor_za64_m( z3, svptrue_b32(),
					/* tile: */ 3, /* slice: */ tcol + 1 );
				z4 = svread_hor_za64_m( z4, svptrue_b32(),
					/* tile: */ 4, /* slice: */ tcol + 1 );
				z5 = svread_hor_za64_m( z5, svptrue_b32(),
					/* tile: */ 5, /* slice: */ tcol + 1 );
				z6 = svread_hor_za64_m( z6, svptrue_b32(),
					/* tile: */ 6, /* slice: */ tcol + 1 );
				z7 = svread_hor_za64_m( z7, svptrue_b32(),
					/* tile: */ 7, /* slice: */ tcol + 1 );

				z0 = svmul_f64_z( svptrue_b32(), z0, zalpha );
				z1 = svmul_f64_z( svptrue_b32(), z1, zalpha );
				z2 = svmul_f64_z( svptrue_b32(), z2, zalpha );
				z3 = svmul_f64_z( svptrue_b32(), z3, zalpha );
				z4 = svmul_f64_z( svptrue_b32(), z4, zalpha );
				z5 = svmul_f64_z( svptrue_b32(), z5, zalpha );
				z6 = svmul_f64_z( svptrue_b32(), z6, zalpha );
				z7 = svmul_f64_z( svptrue_b32(), z7, zalpha );

				z400 = svcreate2( z0, z4 );
				svst1_f64_x2( svptrue_c32(),
					&c_[result_tile_TL_corner + ( tcol + 1 ) * rs_c], z400 );

				z600 = svcreate2( z1, z5 );
				svst1_f64_x2( svptrue_c32(),
					&c_[result_tile_TR_corner + ( tcol + 1 ) * rs_c], z600 );

				z700 = svcreate2( z2, z6 );
				svst1_f64_x2( svptrue_c32(),
					&c_[result_tile_BL_corner + ( tcol + 1 ) * rs_c], z700 );

				z800 = svcreate2( z3, z7 );
				svst1_f64_x2( svptrue_c32(),
					&c_[result_tile_BR_corner + ( tcol + 1 ) * rs_c], z800 );

				// tcol + 2
				z0 = svread_hor_za64_m( z0, svptrue_b32(),
					/* tile: */ 0, /* slice: */ tcol + 2 );
				z1 = svread_hor_za64_m( z1, svptrue_b32(),
					/* tile: */ 1, /* slice: */ tcol + 2 );
				z2 = svread_hor_za64_m( z2, svptrue_b32(),
					/* tile: */ 2, /* slice: */ tcol + 2 );
				z3 = svread_hor_za64_m( z3, svptrue_b32(),
					/* tile: */ 3, /* slice: */ tcol + 2 );
				z4 = svread_hor_za64_m( z4, svptrue_b32(),
					/* tile: */ 4, /* slice: */ tcol + 2 );
				z5 = svread_hor_za64_m( z5, svptrue_b32(),
					/* tile: */ 5, /* slice: */ tcol + 2 );
				z6 = svread_hor_za64_m( z6, svptrue_b32(),
					/* tile: */ 6, /* slice: */ tcol + 2 );
				z7 = svread_hor_za64_m( z7, svptrue_b32(),
					/* tile: */ 7, /* slice: */ tcol + 2 );

				z0 = svmul_f64_z( svptrue_b32(), z0, zalpha );
				z1 = svmul_f64_z( svptrue_b32(), z1, zalpha );
				z2 = svmul_f64_z( svptrue_b32(), z2, zalpha );
				z3 = svmul_f64_z( svptrue_b32(), z3, zalpha );
				z4 = svmul_f64_z( svptrue_b32(), z4, zalpha );
				z5 = svmul_f64_z( svptrue_b32(), z5, zalpha );
				z6 = svmul_f64_z( svptrue_b32(), z6, zalpha );
				z7 = svmul_f64_z( svptrue_b32(), z7, zalpha );

				z400 = svcreate2( z0, z4 );
				svst1_f64_x2( svptrue_c32(),
					&c_[result_tile_TL_corner + ( tcol + 2 ) * rs_c], z400 );

				z600 = svcreate2( z1, z5 );
				svst1_f64_x2( svptrue_c32(),
					&c_[result_tile_TR_corner + ( tcol + 2 ) * rs_c], z600 );

				z700 = svcreate2( z2, z6 );
				svst1_f64_x2( svptrue_c32(),
					&c_[result_tile_BL_corner + ( tcol + 2 ) * rs_c], z700 );

				z800 = svcreate2( z3, z7 );
				svst1_f64_x2( svptrue_c32(),
					&c_[result_tile_BR_corner + ( tcol + 2 ) * rs_c], z800 );

				// tcol + 3
				z0 = svread_hor_za64_m( z0, svptrue_b32(),
					/* tile: */ 0, /* slice: */ tcol + 3 );
				z1 = svread_hor_za64_m( z1, svptrue_b32(),
					/* tile: */ 1, /* slice: */ tcol + 3 );
				z2 = svread_hor_za64_m( z2, svptrue_b32(),
					/* tile: */ 2, /* slice: */ tcol + 3 );
				z3 = svread_hor_za64_m( z3, svptrue_b32(),
					/* tile: */ 3, /* slice: */ tcol + 3 );
				z4 = svread_hor_za64_m( z4, svptrue_b32(),
					/* tile: */ 4, /* slice: */ tcol + 3 );
				z5 = svread_hor_za64_m( z5, svptrue_b32(),
					/* tile: */ 5, /* slice: */ tcol + 3 );
				z6 = svread_hor_za64_m( z6, svptrue_b32(),
					/* tile: */ 6, /* slice: */ tcol + 3 );
				z7 = svread_hor_za64_m( z7, svptrue_b32(),
					/* tile: */ 7, /* slice: */ tcol + 3 );

				z0 = svmul_f64_z( svptrue_b32(), z0, zalpha );
				z1 = svmul_f64_z( svptrue_b32(), z1, zalpha );
				z2 = svmul_f64_z( svptrue_b32(), z2, zalpha );
				z3 = svmul_f64_z( svptrue_b32(), z3, zalpha );
				z4 = svmul_f64_z( svptrue_b32(), z4, zalpha );
				z5 = svmul_f64_z( svptrue_b32(), z5, zalpha );
				z6 = svmul_f64_z( svptrue_b32(), z6, zalpha );
				z7 = svmul_f64_z( svptrue_b32(), z7, zalpha );

				z400 = svcreate2( z0, z4 );
				svst1_f64_x2( svptrue_c32(),
					&c_[result_tile_TL_corner + ( tcol + 3 ) * rs_c], z400 );

				z600 = svcreate2( z1, z5 );
				svst1_f64_x2( svptrue_c32(),
					&c_[result_tile_TR_corner + ( tcol + 3 ) * rs_c], z600 );

				z700 = svcreate2( z2, z6 );
				svst1_f64_x2( svptrue_c32(),
					&c_[result_tile_BL_corner + ( tcol + 3 ) * rs_c], z700 );

				z800 = svcreate2( z3, z7 );
				svst1_f64_x2( svptrue_c32(),
					&c_[result_tile_BR_corner + ( tcol + 3 ) * rs_c], z800 );
			}
		}
		else
		{
			for ( uint64_t tcol = 0; tcol < SVL; tcol += 4 )
			{
				// Read ZA slices into Z regs
				svfloat64_t z0 = svread_hor_za64_m( z0, svptrue_b32(),
					/* tile: */ 0, /* slice: */ tcol + 0 );
				svfloat64_t z1 = svread_hor_za64_m( z1, svptrue_b32(),
					/* tile: */ 1, /* slice: */ tcol + 0 );
				svfloat64_t z2 = svread_hor_za64_m( z2, svptrue_b32(),
					/* tile: */ 2, /* slice: */ tcol + 0 );
				svfloat64_t z3 = svread_hor_za64_m( z3, svptrue_b32(),
					/* tile: */ 3, /* slice: */ tcol + 0 );
				svfloat64_t z4 = svread_hor_za64_m( z4, svptrue_b32(),
					/* tile: */ 4, /* slice: */ tcol + 0 );
				svfloat64_t z5 = svread_hor_za64_m( z5, svptrue_b32(),
					/* tile: */ 5, /* slice: */ tcol + 0 );
				svfloat64_t z6 = svread_hor_za64_m( z6, svptrue_b32(),
					/* tile: */ 6, /* slice: */ tcol + 0 );
				svfloat64_t z7 = svread_hor_za64_m( z7, svptrue_b32(),
					/* tile: */ 7, /* slice: */ tcol + 0 );

				// Scale Z regs by broadcast alpha
				z0 = svmul_f64_z( svptrue_b32(), z0, zalpha );
				z1 = svmul_f64_z( svptrue_b32(), z1, zalpha );
				z2 = svmul_f64_z( svptrue_b32(), z2, zalpha );
				z3 = svmul_f64_z( svptrue_b32(), z3, zalpha );
				z4 = svmul_f64_z( svptrue_b32(), z4, zalpha );
				z5 = svmul_f64_z( svptrue_b32(), z5, zalpha );
				z6 = svmul_f64_z( svptrue_b32(), z6, zalpha );
				z7 = svmul_f64_z( svptrue_b32(), z7, zalpha );

				// Load C into Z regs
				svfloat64x2_t zq5 = svld1_f64_x2( svptrue_c32(),
					&c_[result_tile_TL_corner + ( ( ( tcol + 0 ) * rs_c ) )] );
				svfloat64x2_t zq6 = svld1_f64_x2( svptrue_c32(),
					&c_[result_tile_TR_corner + ( ( ( tcol + 0 ) * rs_c ) )] );
				svfloat64x2_t zq7 = svld1_f64_x2( svptrue_c32(),
					&c_[result_tile_BL_corner + ( ( ( tcol + 0 ) * rs_c ) )] );
				svfloat64x2_t zq8 = svld1_f64_x2( svptrue_c32(),
					&c_[result_tile_BR_corner + ( ( ( tcol + 0 ) * rs_c ) )] );

				// Scale Z regs by broadcast beta (reordered ZA tiles to match
				//  horizontal order)
				svfloat64_t z40 = svmla_m( svptrue_b32(), z0, svget2( zq5, 0 ),
					zbeta );
				svfloat64_t z50 = svmla_m( svptrue_b32(), z4, svget2( zq5, 1 ),
					zbeta );
				svfloat64_t z60 = svmla_m( svptrue_b32(), z1, svget2( zq6, 0 ),
					zbeta );
				svfloat64_t z70 = svmla_m( svptrue_b32(), z5, svget2( zq6, 1 ),
					zbeta );
				svfloat64_t z80 = svmla_m( svptrue_b32(), z2, svget2( zq7, 0 ),
					zbeta );
				svfloat64_t z90 = svmla_m( svptrue_b32(), z6, svget2( zq7, 1 ),
					zbeta );
				svfloat64_t za0 = svmla_m( svptrue_b32(), z3, svget2( zq8, 0 ),
					zbeta );
				svfloat64_t zb0 = svmla_m( svptrue_b32(), z7, svget2( zq8, 1 ),
					zbeta );

				// Store full result into C
				svfloat64x2_t z400 = svcreate2( z40, z50 );
				svst1_f64_x2( svptrue_c32(),
					&c_[result_tile_TL_corner + ( tcol + 0 ) * rs_c], z400 );

				svfloat64x2_t z600 = svcreate2( z60, z70 );
				svst1_f64_x2( svptrue_c32(),
					&c_[result_tile_TR_corner + ( tcol + 0 ) * rs_c], z600 );

				svfloat64x2_t z700 = svcreate2( z80, z90 );
				svst1_f64_x2( svptrue_c32(),
					&c_[result_tile_BL_corner + ( tcol + 0 ) * rs_c], z700 );

				svfloat64x2_t z800 = svcreate2( za0, zb0 );
				svst1_f64_x2( svptrue_c32(),
					&c_[result_tile_BR_corner + ( tcol + 0 ) * rs_c], z800 );

				// tcol + 1
				z0 = svread_hor_za64_m( z0, svptrue_b32(),
					/* tile: */ 0, /* slice: */ tcol + 1 );
				z1 = svread_hor_za64_m( z1, svptrue_b32(),
					/* tile: */ 1, /* slice: */ tcol + 1 );
				z2 = svread_hor_za64_m( z2, svptrue_b32(),
					/* tile: */ 2, /* slice: */ tcol + 1 );
				z3 = svread_hor_za64_m( z3, svptrue_b32(),
					/* tile: */ 3, /* slice: */ tcol + 1 );
				z4 = svread_hor_za64_m( z4, svptrue_b32(),
					/* tile: */ 4, /* slice: */ tcol + 1 );
				z5 = svread_hor_za64_m( z5, svptrue_b32(),
					/* tile: */ 5, /* slice: */ tcol + 1 );
				z6 = svread_hor_za64_m( z6, svptrue_b32(),
					/* tile: */ 6, /* slice: */ tcol + 1 );
				z7 = svread_hor_za64_m( z7, svptrue_b32(),
					/* tile: */ 7, /* slice: */ tcol + 1 );

				z0 = svmul_f64_z( svptrue_b32(), z0, zalpha );
				z1 = svmul_f64_z( svptrue_b32(), z1, zalpha );
				z2 = svmul_f64_z( svptrue_b32(), z2, zalpha );
				z3 = svmul_f64_z( svptrue_b32(), z3, zalpha );
				z4 = svmul_f64_z( svptrue_b32(), z4, zalpha );
				z5 = svmul_f64_z( svptrue_b32(), z5, zalpha );
				z6 = svmul_f64_z( svptrue_b32(), z6, zalpha );
				z7 = svmul_f64_z( svptrue_b32(), z7, zalpha );

				zq5 = svld1_f64_x2( svptrue_c32(),
					&c_[result_tile_TL_corner + ( ( ( tcol + 1 ) * rs_c ) )] );
				zq6 = svld1_f64_x2( svptrue_c32(),
					&c_[result_tile_TR_corner + ( ( ( tcol + 1 ) * rs_c ) )] );
				zq7 = svld1_f64_x2( svptrue_c32(),
					&c_[result_tile_BL_corner + ( ( ( tcol + 1 ) * rs_c ) )] );
				zq8 = svld1_f64_x2( svptrue_c32(),
					&c_[result_tile_BR_corner + ( ( ( tcol + 1 ) * rs_c ) )] );

				z40 = svmla_m( svptrue_b32(), z0, svget2( zq5, 0 ), zbeta );
				z50 = svmla_m( svptrue_b32(), z4, svget2( zq5, 1 ), zbeta );
				z60 = svmla_m( svptrue_b32(), z1, svget2( zq6, 0 ), zbeta );
				z70 = svmla_m( svptrue_b32(), z5, svget2( zq6, 1 ), zbeta );
				z80 = svmla_m( svptrue_b32(), z2, svget2( zq7, 0 ), zbeta );
				z90 = svmla_m( svptrue_b32(), z6, svget2( zq7, 1 ), zbeta );
				za0 = svmla_m( svptrue_b32(), z3, svget2( zq8, 0 ), zbeta );
				zb0 = svmla_m( svptrue_b32(), z7, svget2( zq8, 1 ), zbeta );

				z400 = svcreate2( z40, z50 );
				svst1_f64_x2( svptrue_c32(),
					&c_[result_tile_TL_corner + ( tcol + 1 ) * rs_c], z400 );

				z600 = svcreate2( z60, z70 );
				svst1_f64_x2( svptrue_c32(),
					&c_[result_tile_TR_corner + ( tcol + 1 ) * rs_c], z600 );

				z700 = svcreate2( z80, z90 );
				svst1_f64_x2( svptrue_c32(),
					&c_[result_tile_BL_corner + ( tcol + 1 ) * rs_c], z700 );

				z800 = svcreate2( za0, zb0 );
				svst1_f64_x2( svptrue_c32(),
					&c_[result_tile_BR_corner + ( tcol + 1 ) * rs_c], z800 );

				// tcol + 2
				z0 = svread_hor_za64_m( z0, svptrue_b32(),
					/* tile: */ 0, /* slice: */ tcol + 2 );
				z1 = svread_hor_za64_m( z1, svptrue_b32(),
					/* tile: */ 1, /* slice: */ tcol + 2 );
				z2 = svread_hor_za64_m( z2, svptrue_b32(),
					/* tile: */ 2, /* slice: */ tcol + 2 );
				z3 = svread_hor_za64_m( z3, svptrue_b32(),
					/* tile: */ 3, /* slice: */ tcol + 2 );
				z4 = svread_hor_za64_m( z4, svptrue_b32(),
					/* tile: */ 4, /* slice: */ tcol + 2 );
				z5 = svread_hor_za64_m( z5, svptrue_b32(),
					/* tile: */ 5, /* slice: */ tcol + 2 );
				z6 = svread_hor_za64_m( z6, svptrue_b32(),
					/* tile: */ 6, /* slice: */ tcol + 2 );
				z7 = svread_hor_za64_m( z7, svptrue_b32(),
					/* tile: */ 7, /* slice: */ tcol + 2 );

				z0 = svmul_f64_z( svptrue_b32(), z0, zalpha );
				z1 = svmul_f64_z( svptrue_b32(), z1, zalpha );
				z2 = svmul_f64_z( svptrue_b32(), z2, zalpha );
				z3 = svmul_f64_z( svptrue_b32(), z3, zalpha );
				z4 = svmul_f64_z( svptrue_b32(), z4, zalpha );
				z5 = svmul_f64_z( svptrue_b32(), z5, zalpha );
				z6 = svmul_f64_z( svptrue_b32(), z6, zalpha );
				z7 = svmul_f64_z( svptrue_b32(), z7, zalpha );

				zq5 = svld1_f64_x2( svptrue_c32(),
					&c_[result_tile_TL_corner + ( ( ( tcol + 2 ) * rs_c ) )] );
				zq6 = svld1_f64_x2( svptrue_c32(),
					&c_[result_tile_TR_corner + ( ( ( tcol + 2 ) * rs_c ) )] );
				zq7 = svld1_f64_x2( svptrue_c32(),
					&c_[result_tile_BL_corner + ( ( ( tcol + 2 ) * rs_c ) )] );
				zq8 = svld1_f64_x2( svptrue_c32(),
					&c_[result_tile_BR_corner + ( ( ( tcol + 2 ) * rs_c ) )] );

				z40 = svmla_m( svptrue_b32(), z0, svget2( zq5, 0 ), zbeta );
				z50 = svmla_m( svptrue_b32(), z4, svget2( zq5, 1 ), zbeta );
				z60 = svmla_m( svptrue_b32(), z1, svget2( zq6, 0 ), zbeta );
				z70 = svmla_m( svptrue_b32(), z5, svget2( zq6, 1 ), zbeta );
				z80 = svmla_m( svptrue_b32(), z2, svget2( zq7, 0 ), zbeta );
				z90 = svmla_m( svptrue_b32(), z6, svget2( zq7, 1 ), zbeta );
				za0 = svmla_m( svptrue_b32(), z3, svget2( zq8, 0 ), zbeta );
				zb0 = svmla_m( svptrue_b32(), z7, svget2( zq8, 1 ), zbeta );

				z400 = svcreate2( z40, z50 );
				svst1_f64_x2( svptrue_c32(),
					&c_[result_tile_TL_corner + ( tcol + 2 ) * rs_c], z400 );

				z600 = svcreate2( z60, z70 );
				svst1_f64_x2( svptrue_c32(),
					&c_[result_tile_TR_corner + ( tcol + 2 ) * rs_c], z600 );

				z700 = svcreate2( z80, z90 );
				svst1_f64_x2( svptrue_c32(),
					&c_[result_tile_BL_corner + ( tcol + 2 ) * rs_c], z700 );

				z800 = svcreate2( za0, zb0 );
				svst1_f64_x2( svptrue_c32(),
					&c_[result_tile_BR_corner + ( tcol + 2 ) * rs_c], z800 );

				// tcol + 3
				z0 = svread_hor_za64_m( z0, svptrue_b32(),
					/* tile: */ 0, /* slice: */ tcol + 3 );
				z1 = svread_hor_za64_m( z1, svptrue_b32(),
					/* tile: */ 1, /* slice: */ tcol + 3 );
				z2 = svread_hor_za64_m( z2, svptrue_b32(),
					/* tile: */ 2, /* slice: */ tcol + 3 );
				z3 = svread_hor_za64_m( z3, svptrue_b32(),
					/* tile: */ 3, /* slice: */ tcol + 3 );
				z4 = svread_hor_za64_m( z4, svptrue_b32(),
					/* tile: */ 4, /* slice: */ tcol + 3 );
				z5 = svread_hor_za64_m( z5, svptrue_b32(),
					/* tile: */ 5, /* slice: */ tcol + 3 );
				z6 = svread_hor_za64_m( z6, svptrue_b32(),
					/* tile: */ 6, /* slice: */ tcol + 3 );
				z7 = svread_hor_za64_m( z7, svptrue_b32(),
					/* tile: */ 7, /* slice: */ tcol + 3 );

				z0 = svmul_f64_z( svptrue_b32(), z0, zalpha );
				z1 = svmul_f64_z( svptrue_b32(), z1, zalpha );
				z2 = svmul_f64_z( svptrue_b32(), z2, zalpha );
				z3 = svmul_f64_z( svptrue_b32(), z3, zalpha );
				z4 = svmul_f64_z( svptrue_b32(), z4, zalpha );
				z5 = svmul_f64_z( svptrue_b32(), z5, zalpha );
				z6 = svmul_f64_z( svptrue_b32(), z6, zalpha );
				z7 = svmul_f64_z( svptrue_b32(), z7, zalpha );

				zq5 = svld1_f64_x2( svptrue_c32(),
					&c_[result_tile_TL_corner + ( ( ( tcol + 3 ) * rs_c ) )] );
				zq6 = svld1_f64_x2( svptrue_c32(),
					&c_[result_tile_TR_corner + ( ( ( tcol + 3 ) * rs_c ) )] );
				zq7 = svld1_f64_x2( svptrue_c32(),
					&c_[result_tile_BL_corner + ( ( ( tcol + 3 ) * rs_c ) )] );
				zq8 = svld1_f64_x2( svptrue_c32(),
					&c_[result_tile_BR_corner + ( ( ( tcol + 3 ) * rs_c ) )] );

				z40 = svmla_m( svptrue_b32(), z0, svget2( zq5, 0 ), zbeta );
				z50 = svmla_m( svptrue_b32(), z4, svget2( zq5, 1 ), zbeta );
				z60 = svmla_m( svptrue_b32(), z1, svget2( zq6, 0 ), zbeta );
				z70 = svmla_m( svptrue_b32(), z5, svget2( zq6, 1 ), zbeta );
				z80 = svmla_m( svptrue_b32(), z2, svget2( zq7, 0 ), zbeta );
				z90 = svmla_m( svptrue_b32(), z6, svget2( zq7, 1 ), zbeta );
				za0 = svmla_m( svptrue_b32(), z3, svget2( zq8, 0 ), zbeta );
				zb0 = svmla_m( svptrue_b32(), z7, svget2( zq8, 1 ), zbeta );

				z400 = svcreate2( z40, z50 );
				svst1_f64_x2( svptrue_c32(),
					&c_[result_tile_TL_corner + ( tcol + 3 ) * rs_c], z400 );

				z600 = svcreate2( z60, z70 );
				svst1_f64_x2( svptrue_c32(),
					&c_[result_tile_TR_corner + ( tcol + 3 ) * rs_c], z600 );

				z700 = svcreate2( z80, z90 );
				svst1_f64_x2( svptrue_c32(),
					&c_[result_tile_BL_corner + ( tcol + 3 ) * rs_c], z700 );

				z800 = svcreate2( za0, zb0 );
				svst1_f64_x2( svptrue_c32(),
					&c_[result_tile_BR_corner + ( tcol + 3 ) * rs_c], z800 );
			}
		}
	}
    
	GEMM_UKR_FLUSH_CT( d );

	return;
}



