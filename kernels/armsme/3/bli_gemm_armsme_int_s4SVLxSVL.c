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

#include <arm_sme.h>
#include "blis.h"

__arm_new( "za" ) __arm_locally_streaming void bli_sgemm_armsme_int_4SVLxSVL
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
	uint64_t SVL = svcntsw();

	GEMM_UKR_SETUP_CT_AMBI( s, 4 * SVL, SVL, false );

	float* a_ = (float*)a;
	float* b_ = (float*)b;

	const void* a_next = bli_auxinfo_next_a( data );
	const void* b_next = bli_auxinfo_next_b( data );

	float* c_ = (float*)c;

	svzero_za();

	uint64_t k_;
	uint64_t k_iter = k / 4;
	uint64_t k_left = k % 4;

	for ( k_ = 0; k_ < k_iter; k_++ )
	{
		// Loads
		svfloat32x4_t zL00 = svld1_f32_x4( svptrue_c32(),
			(float32_t*)( &b_[0] ) );

		svfloat32x4_t zR00 = svld1_f32_x4( svptrue_c32(),
			(float32_t*)( &a_[0] ) );

		svmopa_za32_m( 0, svptrue_b32(), svptrue_b32(), svget4( zL00, 0 ),
			svget4( zR00, 0 ) );
		svmopa_za32_m( 1, svptrue_b32(), svptrue_b32(), svget4( zL00, 0 ),
			svget4( zR00, 1 ) );

		svfloat32x4_t zR01 = svld1_f32_x4( svptrue_c32(),
			(float32_t*)( &a_[( 4 * SVL )] ) );

		svmopa_za32_m( 2, svptrue_b32(), svptrue_b32(), svget4( zL00, 0 ),
			svget4( zR00, 2 ) );
		svmopa_za32_m( 3, svptrue_b32(), svptrue_b32(), svget4( zL00, 0 ),
			svget4( zR00, 3 ) );

		svmopa_za32_m( 0, svptrue_b32(), svptrue_b32(), svget4( zL00, 1 ),
			svget4( zR01, 0 ) );
		svmopa_za32_m( 1, svptrue_b32(), svptrue_b32(), svget4( zL00, 1 ),
			svget4( zR01, 1 ) );

		svfloat32x4_t zR02 = svld1_f32_x4( svptrue_c32(),
			(float32_t*)( &a_[2 * ( 4 * SVL )] ) );

		svmopa_za32_m( 2, svptrue_b32(), svptrue_b32(), svget4( zL00, 1 ),
			svget4( zR01, 2 ) );
		svmopa_za32_m( 3, svptrue_b32(), svptrue_b32(), svget4( zL00, 1 ),
			svget4( zR01, 3 ) );

		svmopa_za32_m( 0, svptrue_b32(), svptrue_b32(), svget4( zL00, 2 ),
			svget4( zR02, 0 ) );
		svmopa_za32_m( 1, svptrue_b32(), svptrue_b32(), svget4( zL00, 2 ),
			svget4( zR02, 1 ) );

		svfloat32x4_t zR03 = svld1_f32_x4( svptrue_c32(),
			(float32_t*)( &a_[3 * ( 4 * SVL )] ) );

		svmopa_za32_m( 2, svptrue_b32(), svptrue_b32(), svget4( zL00, 2 ),
			svget4( zR02, 2 ) );
		svmopa_za32_m( 3, svptrue_b32(), svptrue_b32(), svget4( zL00, 2 ),
			svget4( zR02, 3 ) );

		svmopa_za32_m( 0, svptrue_b32(), svptrue_b32(), svget4( zL00, 3 ),
			svget4( zR03, 0 ) );
		svprfb( svptrue_b32(), (float*)&a_next, 0 );
		svmopa_za32_m( 1, svptrue_b32(), svptrue_b32(), svget4( zL00, 3 ),
			svget4( zR03, 1 ) );

		svprfb( svptrue_b32(), (float*)&b_next, 0 );
		svmopa_za32_m( 2, svptrue_b32(), svptrue_b32(), svget4( zL00, 3 ),
			svget4( zR03, 2 ) );
		svmopa_za32_m( 3, svptrue_b32(), svptrue_b32(), svget4( zL00, 3 ),
			svget4( zR03, 3 ) );

		b_ += ( 4 * SVL );
		a_ += ( 4 * 4 * SVL );
	}

	for ( k_ = 0; k_ < k_left; k_ += 1 )
	{
		svfloat32_t zL00 = svld1_f32( svptrue_b32(), (float32_t*)( &b_[0] ) );
		svfloat32x4_t zR00 = svld1_f32_x4( svptrue_c32(),
			(float32_t*)( &a_[0] ) );

		svmopa_za32_m( 0, svptrue_b32(), svptrue_b32(), zL00,
			svget4( zR00, 0 ) );
		svmopa_za32_m( 1, svptrue_b32(), svptrue_b32(), zL00,
			svget4( zR00, 1 ) );

		svmopa_za32_m( 2, svptrue_b32(), svptrue_b32(), zL00,
			svget4( zR00, 2 ) );
		svmopa_za32_m( 3, svptrue_b32(), svptrue_b32(), zL00,
			svget4( zR00, 3 ) );

		b_ += ( SVL );
		a_ += ( 4 * SVL );
	}

	float beta_ = *(float*)beta;
	float alpha_ = *(float*)alpha;

	const uint64_t result_tile_TL_corner = 0;
	const uint64_t result_tile_BL_corner = SVL * rs_c;
	const uint64_t result_tile_TR_corner = SVL * 2 * rs_c;
	const uint64_t result_tile_BR_corner = SVL * 3 * rs_c;

	svfloat32_t zbeta = svdup_f32( beta_ );
	svfloat32_t zalpha = svdup_f32( alpha_ );

	if ( cs_c == 1 )
	{
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
				svst1( svptrue_b32(),
					&c_[result_tile_TL_corner + ( tcol + 0 ) * rs_c], z0 );
				svst1( svptrue_b32(),
					&c_[result_tile_BL_corner + ( tcol + 0 ) * rs_c], z1 );
				svst1( svptrue_b32(),
					&c_[result_tile_TR_corner + ( tcol + 0 ) * rs_c], z2 );
				svst1( svptrue_b32(),
					&c_[result_tile_BR_corner + ( tcol + 0 ) * rs_c], z3 );

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

				svst1( svptrue_b32(),
					&c_[result_tile_TL_corner + ( tcol + 1 ) * rs_c], z0 );
				svst1( svptrue_b32(),
					&c_[result_tile_BL_corner + ( tcol + 1 ) * rs_c], z1 );
				svst1( svptrue_b32(),
					&c_[result_tile_TR_corner + ( tcol + 1 ) * rs_c], z2 );
				svst1( svptrue_b32(),
					&c_[result_tile_BR_corner + ( tcol + 1 ) * rs_c], z3 );

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

				svst1( svptrue_b32(),
					&c_[result_tile_TL_corner + ( tcol + 2 ) * rs_c], z0 );
				svst1( svptrue_b32(),
					&c_[result_tile_BL_corner + ( tcol + 2 ) * rs_c], z1 );
				svst1( svptrue_b32(),
					&c_[result_tile_TR_corner + ( tcol + 2 ) * rs_c], z2 );
				svst1( svptrue_b32(),
					&c_[result_tile_BR_corner + ( tcol + 2 ) * rs_c], z3 );

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

				svst1( svptrue_b32(),
					&c_[result_tile_TL_corner + ( tcol + 3 ) * rs_c], z0 );
				svst1( svptrue_b32(),
					&c_[result_tile_BL_corner + ( tcol + 3 ) * rs_c], z1 );
				svst1( svptrue_b32(),
					&c_[result_tile_TR_corner + ( tcol + 3 ) * rs_c], z2 );
				svst1( svptrue_b32(),
					&c_[result_tile_BR_corner + ( tcol + 3 ) * rs_c], z3 );
			}
		}
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
				z0 = svmul_f32_z( svptrue_b32(), z0, zalpha );
				z1 = svmul_f32_z( svptrue_b32(), z1, zalpha );
				z2 = svmul_f32_z( svptrue_b32(), z2, zalpha );
				z3 = svmul_f32_z( svptrue_b32(), z3, zalpha );

				// Load C into Z regs
				svfloat32_t z4 = svld1_f32( svptrue_b32(),
					&c_[result_tile_TL_corner + ( tcol + 0 ) * rs_c] );
				svfloat32_t z5 = svld1_f32( svptrue_b32(),
					&c_[result_tile_BL_corner + ( tcol + 0 ) * rs_c] );
				svfloat32_t z6 = svld1_f32( svptrue_b32(),
					&c_[result_tile_TR_corner + ( tcol + 0 ) * rs_c] );
				svfloat32_t z7 = svld1_f32( svptrue_b32(),
					&c_[result_tile_BR_corner + ( tcol + 0 ) * rs_c] );

				// Scale Z regs by broadcast beta
				z4 = svmla_m( svptrue_b32(), z0, z4, zbeta );
				z5 = svmla_m( svptrue_b32(), z1, z5, zbeta );
				z6 = svmla_m( svptrue_b32(), z2, z6, zbeta );
				z7 = svmla_m( svptrue_b32(), z3, z7, zbeta );

				// Store full result into C
				svst1( svptrue_b32(),
					&c_[result_tile_TL_corner + ( tcol + 0 ) * rs_c], z4 );
				svst1( svptrue_b32(),
					&c_[result_tile_BL_corner + ( tcol + 0 ) * rs_c], z5 );
				svst1( svptrue_b32(),
					&c_[result_tile_TR_corner + ( tcol + 0 ) * rs_c], z6 );
				svst1( svptrue_b32(),
					&c_[result_tile_BR_corner + ( tcol + 0 ) * rs_c], z7 );

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

				z4 = svld1_f32( svptrue_b32(),
					&c_[result_tile_TL_corner + ( tcol + 1 ) * rs_c] );
				z5 = svld1_f32( svptrue_b32(),
					&c_[result_tile_BL_corner + ( tcol + 1 ) * rs_c] );
				z6 = svld1_f32( svptrue_b32(),
					&c_[result_tile_TR_corner + ( tcol + 1 ) * rs_c] );
				z7 = svld1_f32( svptrue_b32(),
					&c_[result_tile_BR_corner + ( tcol + 1 ) * rs_c] );

				z4 = svmla_m( svptrue_b32(), z0, z4, zbeta );
				z5 = svmla_m( svptrue_b32(), z1, z5, zbeta );
				z6 = svmla_m( svptrue_b32(), z2, z6, zbeta );
				z7 = svmla_m( svptrue_b32(), z3, z7, zbeta );

				svst1( svptrue_b32(),
					&c_[result_tile_TL_corner + ( tcol + 1 ) * rs_c], z4 );
				svst1( svptrue_b32(),
					&c_[result_tile_BL_corner + ( tcol + 1 ) * rs_c], z5 );
				svst1( svptrue_b32(),
					&c_[result_tile_TR_corner + ( tcol + 1 ) * rs_c], z6 );
				svst1( svptrue_b32(),
					&c_[result_tile_BR_corner + ( tcol + 1 ) * rs_c], z7 );

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

				z4 = svld1_f32( svptrue_b32(),
					&c_[result_tile_TL_corner + ( tcol + 2 ) * rs_c] );
				z5 = svld1_f32( svptrue_b32(),
					&c_[result_tile_BL_corner + ( tcol + 2 ) * rs_c] );
				z6 = svld1_f32( svptrue_b32(),
					&c_[result_tile_TR_corner + ( tcol + 2 ) * rs_c] );
				z7 = svld1_f32( svptrue_b32(),
					&c_[result_tile_BR_corner + ( tcol + 2 ) * rs_c] );

				z4 = svmla_m( svptrue_b32(), z0, z4, zbeta );
				z5 = svmla_m( svptrue_b32(), z1, z5, zbeta );
				z6 = svmla_m( svptrue_b32(), z2, z6, zbeta );
				z7 = svmla_m( svptrue_b32(), z3, z7, zbeta );

				svst1( svptrue_b32(),
					&c_[result_tile_TL_corner + ( tcol + 2 ) * rs_c], z4 );
				svst1( svptrue_b32(),
					&c_[result_tile_BL_corner + ( tcol + 2 ) * rs_c], z5 );
				svst1( svptrue_b32(),
					&c_[result_tile_TR_corner + ( tcol + 2 ) * rs_c], z6 );
				svst1( svptrue_b32(),
					&c_[result_tile_BR_corner + ( tcol + 2 ) * rs_c], z7 );

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

				z4 = svld1_f32( svptrue_b32(),
					&c_[result_tile_TL_corner + ( tcol + 3 ) * rs_c] );
				z5 = svld1_f32( svptrue_b32(),
					&c_[result_tile_BL_corner + ( tcol + 3 ) * rs_c] );
				z6 = svld1_f32( svptrue_b32(),
					&c_[result_tile_TR_corner + ( tcol + 3 ) * rs_c] );
				z7 = svld1_f32( svptrue_b32(),
					&c_[result_tile_BR_corner + ( tcol + 3 ) * rs_c] );

				z4 = svmla_m( svptrue_b32(), z0, z4, zbeta );
				z5 = svmla_m( svptrue_b32(), z1, z5, zbeta );
				z6 = svmla_m( svptrue_b32(), z2, z6, zbeta );
				z7 = svmla_m( svptrue_b32(), z3, z7, zbeta );

				svst1( svptrue_b32(),
					&c_[result_tile_TL_corner + ( tcol + 3 ) * rs_c], z4 );
				svst1( svptrue_b32(),
					&c_[result_tile_BL_corner + ( tcol + 3 ) * rs_c], z5 );
				svst1( svptrue_b32(),
					&c_[result_tile_TR_corner + ( tcol + 3 ) * rs_c], z6 );
				svst1( svptrue_b32(),
					&c_[result_tile_BR_corner + ( tcol + 3 ) * rs_c], z7 );
			}
		}
	}
	else
	{
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
				svfloat32x4_t z4w = svcreate4( z0, z1, z2, z3 );
				svst1_f32_x4( svptrue_c32(),
					&c_[result_tile_TL_corner + ( tcol + 0 ) * cs_c], z4w );

				z0 = svread_hor_za32_m( z0, svptrue_b32(),
					/* tile: */ 0, /* slice: */ tcol + 1 );
				z1 = svread_hor_za32_m( z1, svptrue_b32(),
					/* tile: */ 1, /* slice: */ tcol + 1 );
				z2 = svread_hor_za32_m( z2, svptrue_b32(),
					/* tile: */ 2, /* slice: */ tcol + 1 );
				z3 = svread_hor_za32_m( z3, svptrue_b32(),
					/* tile: */ 3, /* slice: */ tcol + 1 );

				// Repeat unfolded x4
				svfloat32_t z4 = svmul_f32_z( svptrue_b32(), z0, zalpha );
				svfloat32_t z5 = svmul_f32_z( svptrue_b32(), z1, zalpha );
				svfloat32_t z6 = svmul_f32_z( svptrue_b32(), z2, zalpha );
				svfloat32_t z7 = svmul_f32_z( svptrue_b32(), z3, zalpha );

				svfloat32x4_t z5w = svcreate4( z4, z5, z6, z7 );
				svst1_f32_x4( svptrue_c32(),
					&c_[result_tile_TL_corner + ( tcol + 1 ) * cs_c], z5w );

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

				z4w = svcreate4( z0, z1, z2, z3 );
				svst1_f32_x4( svptrue_c32(),
					&c_[result_tile_TL_corner + ( tcol + 2 ) * cs_c], z4w );

				z0 = svread_hor_za32_m( z0, svptrue_b32(),
					/* tile: */ 0, /* slice: */ tcol + 3 );
				z1 = svread_hor_za32_m( z1, svptrue_b32(),
					/* tile: */ 1, /* slice: */ tcol + 3 );
				z2 = svread_hor_za32_m( z2, svptrue_b32(),
					/* tile: */ 2, /* slice: */ tcol + 3 );
				z3 = svread_hor_za32_m( z3, svptrue_b32(),
					/* tile: */ 3, /* slice: */ tcol + 3 );

				z4 = svmul_f32_z( svptrue_b32(), z0, zalpha );
				z5 = svmul_f32_z( svptrue_b32(), z1, zalpha );
				z6 = svmul_f32_z( svptrue_b32(), z2, zalpha );
				z7 = svmul_f32_z( svptrue_b32(), z3, zalpha );

				z5w = svcreate4( z4, z5, z6, z7 );
				svst1_f32_x4( svptrue_c32(),
					&c_[result_tile_TL_corner + ( tcol + 3 ) * cs_c], z5w );
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
				z0 = svmul_f32_z( svptrue_b32(), z0, zalpha );
				z1 = svmul_f32_z( svptrue_b32(), z1, zalpha );
				z2 = svmul_f32_z( svptrue_b32(), z2, zalpha );
				z3 = svmul_f32_z( svptrue_b32(), z3, zalpha );

				// Load C into Z regs
				svfloat32x4_t z4q = svld1_f32_x4( svptrue_c32(),
					&c_[result_tile_TL_corner + ( tcol + 0 ) * cs_c] );

				// Scale Z regs by broadcast beta
				svfloat32_t z4 = svmla_m( svptrue_b32(), z0, svget4( z4q, 0 ),
					zbeta );
				svfloat32_t z5 = svmla_m( svptrue_b32(), z1, svget4( z4q, 1 ),
					zbeta );
				svfloat32_t z6 = svmla_m( svptrue_b32(), z2, svget4( z4q, 2 ),
					zbeta );
				svfloat32_t z7 = svmla_m( svptrue_b32(), z3, svget4( z4q, 3 ),
					zbeta );

				// Store full result into C
				svfloat32x4_t z4w = svcreate4( z4, z5, z6, z7 );
				svst1_f32_x4( svptrue_c32(),
					&c_[result_tile_TL_corner + ( tcol + 0 ) * cs_c], z4w );

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

				svfloat32x4_t z5q = svld1_f32_x4( svptrue_c32(),
					&c_[result_tile_TL_corner + ( tcol + 1 ) * cs_c] );

				svfloat32_t z8 = svmla_m( svptrue_b32(), z0, svget4( z5q, 0 ),
					zbeta );
				svfloat32_t z9 = svmla_m( svptrue_b32(), z1, svget4( z5q, 1 ),
					zbeta );
				svfloat32_t z10 = svmla_m( svptrue_b32(), z2, svget4( z5q, 2 ),
					zbeta );
				svfloat32_t z11 = svmla_m( svptrue_b32(), z3, svget4( z5q, 3 ),
					zbeta );

				svfloat32x4_t z5w = svcreate4( z8, z9, z10, z11 );
				svst1_f32_x4( svptrue_c32(),
					&c_[result_tile_TL_corner + ( tcol + 1 ) * cs_c], z5w );

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

				z4q = svld1_f32_x4( svptrue_c32(),
					&c_[result_tile_TL_corner + ( tcol + 2 ) * cs_c] );

				z4 = svmla_m( svptrue_b32(), z0, svget4( z4q, 0 ), zbeta );
				z5 = svmla_m( svptrue_b32(), z1, svget4( z4q, 1 ), zbeta );
				z6 = svmla_m( svptrue_b32(), z2, svget4( z4q, 2 ), zbeta );
				z7 = svmla_m( svptrue_b32(), z3, svget4( z4q, 3 ), zbeta );

				z4w = svcreate4( z4, z5, z6, z7 );
				svst1_f32_x4( svptrue_c32(),
					&c_[result_tile_TL_corner + ( tcol + 2 ) * cs_c], z4w );

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

				z5q = svld1_f32_x4( svptrue_c32(),
					&c_[result_tile_TL_corner + ( tcol + 3 ) * cs_c] );

				z8 = svmla_m( svptrue_b32(), z0, svget4( z5q, 0 ), zbeta );
				z9 = svmla_m( svptrue_b32(), z1, svget4( z5q, 1 ), zbeta );
				z10 = svmla_m( svptrue_b32(), z2, svget4( z5q, 2 ), zbeta );
				z11 = svmla_m( svptrue_b32(), z3, svget4( z5q, 3 ), zbeta );

				z5w = svcreate4( z8, z9, z10, z11 );
				svst1_f32_x4( svptrue_c32(),
					&c_[result_tile_TL_corner + ( tcol + 3 ) * cs_c], z5w );
			}
		}
	}

	GEMM_UKR_FLUSH_CT( s );

	return;
}
__arm_new( "za" ) __arm_locally_streaming void bli_dgemm_armsme_int_8SVLxSVL
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

	GEMM_UKR_SETUP_CT_AMBI( d, 8 * SVL, SVL, false );

	double* a_ = (double*)a;
	double* b_ = (double*)b;
	double* c_ = (double*)c;

	svzero_za();

	uint64_t k_;
	uint64_t k_iter = k / 4;
	uint64_t k_left = k % 4;

	for ( k_ = 0; k_ < k_iter; k_++ )
	{
		// Loads
		svfloat64x4_t zL00 = svld1_f64_x4( svptrue_c32(),
			(float64_t*)( &a_[0] ) );
		svfloat64x4_t zL01 = svld1_f64_x4( svptrue_c32(),
			(float64_t*)( &a_[4 * SVL] ) );
		svfloat64_t zR00 = svld1_f64( svptrue_b32(), (float64_t*)( &b_[0] ) );

		svmopa_za64_m( 0, svptrue_b32(), svptrue_b32(), svget4( zL00, 0 ),
			zR00 );
		svmopa_za64_m( 1, svptrue_b32(), svptrue_b32(), svget4( zL00, 1 ),
			zR00 );

		svmopa_za64_m( 2, svptrue_b32(), svptrue_b32(), svget4( zL00, 2 ),
			zR00 );
		svmopa_za64_m( 3, svptrue_b32(), svptrue_b32(), svget4( zL00, 3 ),
			zR00 );

		svmopa_za64_m( 4, svptrue_b32(), svptrue_b32(), svget4( zL01, 0 ),
			zR00 );
		svmopa_za64_m( 5, svptrue_b32(), svptrue_b32(), svget4( zL01, 1 ),
			zR00 );

		svmopa_za64_m( 6, svptrue_b32(), svptrue_b32(), svget4( zL01, 2 ),
			zR00 );
		svmopa_za64_m( 7, svptrue_b32(), svptrue_b32(), svget4( zL01, 3 ),
			zR00 );

		svfloat64x4_t zL02 = svld1_f64_x4( svptrue_c32(),
			(float64_t*)( &a_[8 * SVL] ) );
		svfloat64x4_t zL03 = svld1_f64_x4( svptrue_c32(),
			(float64_t*)( &a_[12 * SVL] ) );
		svfloat64_t zR01 = svld1_f64( svptrue_b32(),
			(float64_t*)( &b_[1 * SVL] ) );

		svmopa_za64_m( 0, svptrue_b32(), svptrue_b32(), svget4( zL02, 0 ),
			zR01 );
		svmopa_za64_m( 1, svptrue_b32(), svptrue_b32(), svget4( zL02, 1 ),
			zR01 );

		svmopa_za64_m( 2, svptrue_b32(), svptrue_b32(), svget4( zL02, 2 ),
			zR01 );
		svmopa_za64_m( 3, svptrue_b32(), svptrue_b32(), svget4( zL02, 3 ),
			zR01 );

		svmopa_za64_m( 4, svptrue_b32(), svptrue_b32(), svget4( zL03, 0 ),
			zR01 );
		svmopa_za64_m( 5, svptrue_b32(), svptrue_b32(), svget4( zL03, 1 ),
			zR01 );

		svmopa_za64_m( 6, svptrue_b32(), svptrue_b32(), svget4( zL03, 2 ),
			zR01 );
		svmopa_za64_m( 7, svptrue_b32(), svptrue_b32(), svget4( zL03, 3 ),
			zR01 );

		svfloat64x4_t zL04 = svld1_f64_x4( svptrue_c32(),
			(float64_t*)( &a_[16 * SVL] ) );
		svfloat64x4_t zL05 = svld1_f64_x4( svptrue_c32(),
			(float64_t*)( &a_[20 * SVL] ) );
		svfloat64_t zR02 = svld1_f64( svptrue_b32(),
			(float64_t*)( &b_[2 * SVL] ) );

		svmopa_za64_m( 0, svptrue_b32(), svptrue_b32(), svget4( zL04, 0 ),
			zR02 );
		svmopa_za64_m( 1, svptrue_b32(), svptrue_b32(), svget4( zL04, 1 ),
			zR02 );

		svmopa_za64_m( 2, svptrue_b32(), svptrue_b32(), svget4( zL04, 2 ),
			zR02 );
		svmopa_za64_m( 3, svptrue_b32(), svptrue_b32(), svget4( zL04, 3 ),
			zR02 );

		svmopa_za64_m( 4, svptrue_b32(), svptrue_b32(), svget4( zL05, 0 ),
			zR02 );
		svmopa_za64_m( 5, svptrue_b32(), svptrue_b32(), svget4( zL05, 1 ),
			zR02 );

		svmopa_za64_m( 6, svptrue_b32(), svptrue_b32(), svget4( zL05, 2 ),
			zR02 );
		svmopa_za64_m( 7, svptrue_b32(), svptrue_b32(), svget4( zL05, 3 ),
			zR02 );

		svfloat64x4_t zL06 = svld1_f64_x4( svptrue_c32(),
			(float64_t*)( &a_[24 * SVL] ) );
		svfloat64x4_t zL07 = svld1_f64_x4( svptrue_c32(),
			(float64_t*)( &a_[28 * SVL] ) );
		svfloat64_t zR03 = svld1_f64( svptrue_b32(),
			(float64_t*)( &b_[3 * SVL] ) );

		svmopa_za64_m( 0, svptrue_b32(), svptrue_b32(), svget4( zL06, 0 ),
			zR03 );
		svmopa_za64_m( 1, svptrue_b32(), svptrue_b32(), svget4( zL06, 1 ),
			zR03 );

		svmopa_za64_m( 2, svptrue_b32(), svptrue_b32(), svget4( zL06, 2 ),
			zR03 );
		svmopa_za64_m( 3, svptrue_b32(), svptrue_b32(), svget4( zL06, 3 ),
			zR03 );

		svmopa_za64_m( 4, svptrue_b32(), svptrue_b32(), svget4( zL07, 0 ),
			zR03 );
		svmopa_za64_m( 5, svptrue_b32(), svptrue_b32(), svget4( zL07, 1 ),
			zR03 );

		svmopa_za64_m( 6, svptrue_b32(), svptrue_b32(), svget4( zL07, 2 ),
			zR03 );
		svmopa_za64_m( 7, svptrue_b32(), svptrue_b32(), svget4( zL07, 3 ),
			zR03 );

		a_ += ( 4 * 8 * SVL );
		b_ += ( 4 * SVL );
	}

	for ( k_ = 0; k_ < k_left; k_ += 1 )
	{
		svfloat64x4_t zL00 = svld1_f64_x4( svptrue_c32(),
			(float64_t*)( &a_[0] ) );
		svfloat64x4_t zL01 = svld1_f64_x4( svptrue_c32(),
			(float64_t*)( &a_[4 * SVL] ) );
		svfloat64_t zR00 = svld1_f64( svptrue_b32(), (float64_t*)( &b_[0] ) );

		svmopa_za64_m( 0, svptrue_b32(), svptrue_b32(), svget4( zL00, 0 ),
			zR00 );
		svmopa_za64_m( 1, svptrue_b32(), svptrue_b32(), svget4( zL00, 1 ),
			zR00 );

		svmopa_za64_m( 2, svptrue_b32(), svptrue_b32(), svget4( zL00, 2 ),
			zR00 );
		svmopa_za64_m( 3, svptrue_b32(), svptrue_b32(), svget4( zL00, 3 ),
			zR00 );

		svmopa_za64_m( 4, svptrue_b32(), svptrue_b32(), svget4( zL01, 0 ),
			zR00 );
		svmopa_za64_m( 5, svptrue_b32(), svptrue_b32(), svget4( zL01, 1 ),
			zR00 );

		svmopa_za64_m( 6, svptrue_b32(), svptrue_b32(), svget4( zL01, 2 ),
			zR00 );
		svmopa_za64_m( 7, svptrue_b32(), svptrue_b32(), svget4( zL01, 3 ),
			zR00 );

		a_ += ( 8 * SVL );
		b_ += ( SVL );
	}

	double beta_ = *(double*)beta;
	double alpha_ = *(double*)alpha;

	const uint64_t result_tile_TL_corner = 0;

	svfloat64_t zbeta = svdup_f64( beta_ );
	svfloat64_t zalpha = svdup_f64( alpha_ );

	if ( rs_c == 1 )
	{
		const uint64_t result_tile_TR_corner = 4 * SVL;

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
		const uint64_t result_tile_1 = SVL * rs_c;
		const uint64_t result_tile_2 = SVL * 2 * rs_c;
		const uint64_t result_tile_3 = SVL * 3 * rs_c;
		const uint64_t result_tile_4 = SVL * 4 * rs_c;
		const uint64_t result_tile_5 = SVL * 5 * rs_c;
		const uint64_t result_tile_6 = SVL * 6 * rs_c;
		const uint64_t result_tile_7 = SVL * 7 * rs_c;

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
				svst1_f64( svptrue_b32(),
					&c_[result_tile_TL_corner + ( tcol + 0 ) * rs_c], z0 );
				svst1_f64( svptrue_b32(),
					&c_[result_tile_1 + ( tcol + 0 ) * rs_c], z1 );
				svst1_f64( svptrue_b32(),
					&c_[result_tile_2 + ( tcol + 0 ) * rs_c], z2 );
				svst1_f64( svptrue_b32(),
					&c_[result_tile_3 + ( tcol + 0 ) * rs_c], z3 );
				svst1_f64( svptrue_b32(),
					&c_[result_tile_4 + ( tcol + 0 ) * rs_c], z4 );
				svst1_f64( svptrue_b32(),
					&c_[result_tile_5 + ( tcol + 0 ) * rs_c], z5 );
				svst1_f64( svptrue_b32(),
					&c_[result_tile_6 + ( tcol + 0 ) * rs_c], z6 );
				svst1_f64( svptrue_b32(),
					&c_[result_tile_7 + ( tcol + 0 ) * rs_c], z7 );

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

				svst1_f64( svptrue_b32(),
					&c_[result_tile_TL_corner + ( tcol + 1 ) * rs_c], z0 );
				svst1_f64( svptrue_b32(),
					&c_[result_tile_1 + ( tcol + 1 ) * rs_c], z1 );
				svst1_f64( svptrue_b32(),
					&c_[result_tile_2 + ( tcol + 1 ) * rs_c], z2 );
				svst1_f64( svptrue_b32(),
					&c_[result_tile_3 + ( tcol + 1 ) * rs_c], z3 );
				svst1_f64( svptrue_b32(),
					&c_[result_tile_4 + ( tcol + 1 ) * rs_c], z4 );
				svst1_f64( svptrue_b32(),
					&c_[result_tile_5 + ( tcol + 1 ) * rs_c], z5 );
				svst1_f64( svptrue_b32(),
					&c_[result_tile_6 + ( tcol + 1 ) * rs_c], z6 );
				svst1_f64( svptrue_b32(),
					&c_[result_tile_7 + ( tcol + 1 ) * rs_c], z7 );

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

				svst1_f64( svptrue_b32(),
					&c_[result_tile_TL_corner + ( tcol + 2 ) * rs_c], z0 );
				svst1_f64( svptrue_b32(),
					&c_[result_tile_1 + ( tcol + 2 ) * rs_c], z1 );
				svst1_f64( svptrue_b32(),
					&c_[result_tile_2 + ( tcol + 2 ) * rs_c], z2 );
				svst1_f64( svptrue_b32(),
					&c_[result_tile_3 + ( tcol + 2 ) * rs_c], z3 );
				svst1_f64( svptrue_b32(),
					&c_[result_tile_4 + ( tcol + 2 ) * rs_c], z4 );
				svst1_f64( svptrue_b32(),
					&c_[result_tile_5 + ( tcol + 2 ) * rs_c], z5 );
				svst1_f64( svptrue_b32(),
					&c_[result_tile_6 + ( tcol + 2 ) * rs_c], z6 );
				svst1_f64( svptrue_b32(),
					&c_[result_tile_7 + ( tcol + 2 ) * rs_c], z7 );

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

				svst1_f64( svptrue_b32(),
					&c_[result_tile_TL_corner + ( tcol + 3 ) * rs_c], z0 );
				svst1_f64( svptrue_b32(),
					&c_[result_tile_1 + ( tcol + 3 ) * rs_c], z1 );
				svst1_f64( svptrue_b32(),
					&c_[result_tile_2 + ( tcol + 3 ) * rs_c], z2 );
				svst1_f64( svptrue_b32(),
					&c_[result_tile_3 + ( tcol + 3 ) * rs_c], z3 );
				svst1_f64( svptrue_b32(),
					&c_[result_tile_4 + ( tcol + 3 ) * rs_c], z4 );
				svst1_f64( svptrue_b32(),
					&c_[result_tile_5 + ( tcol + 3 ) * rs_c], z5 );
				svst1_f64( svptrue_b32(),
					&c_[result_tile_6 + ( tcol + 3 ) * rs_c], z6 );
				svst1_f64( svptrue_b32(),
					&c_[result_tile_7 + ( tcol + 3 ) * rs_c], z7 );
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
				svfloat64_t zq0 = svld1_f64( svptrue_b32(),
					&c_[result_tile_TL_corner + ( ( ( tcol + 0 ) * rs_c ) )] );
				svfloat64_t zq1 = svld1_f64( svptrue_b32(),
					&c_[result_tile_1 + ( ( ( tcol + 0 ) * rs_c ) )] );
				svfloat64_t zq2 = svld1_f64( svptrue_b32(),
					&c_[result_tile_2 + ( ( ( tcol + 0 ) * rs_c ) )] );
				svfloat64_t zq3 = svld1_f64( svptrue_b32(),
					&c_[result_tile_3 + ( ( ( tcol + 0 ) * rs_c ) )] );
				svfloat64_t zq4 = svld1_f64( svptrue_b32(),
					&c_[result_tile_4 + ( ( ( tcol + 0 ) * rs_c ) )] );
				svfloat64_t zq5 = svld1_f64( svptrue_b32(),
					&c_[result_tile_5 + ( ( ( tcol + 0 ) * rs_c ) )] );
				svfloat64_t zq6 = svld1_f64( svptrue_b32(),
					&c_[result_tile_6 + ( ( ( tcol + 0 ) * rs_c ) )] );
				svfloat64_t zq7 = svld1_f64( svptrue_b32(),
					&c_[result_tile_7 + ( ( ( tcol + 0 ) * rs_c ) )] );

				// Scale Z regs by broadcast beta
				svfloat64_t z00 = svmla_m( svptrue_b32(), z0, zq0, zbeta );
				svfloat64_t z10 = svmla_m( svptrue_b32(), z1, zq1, zbeta );
				svfloat64_t z20 = svmla_m( svptrue_b32(), z2, zq2, zbeta );
				svfloat64_t z30 = svmla_m( svptrue_b32(), z3, zq3, zbeta );
				svfloat64_t z40 = svmla_m( svptrue_b32(), z4, zq4, zbeta );
				svfloat64_t z50 = svmla_m( svptrue_b32(), z5, zq5, zbeta );
				svfloat64_t z60 = svmla_m( svptrue_b32(), z6, zq6, zbeta );
				svfloat64_t z70 = svmla_m( svptrue_b32(), z7, zq7, zbeta );

				// Store full result into C
				svst1_f64( svptrue_b32(),
					&c_[result_tile_TL_corner + ( tcol + 0 ) * rs_c], z00 );
				svst1_f64( svptrue_b32(),
					&c_[result_tile_1 + ( tcol + 0 ) * rs_c], z10 );
				svst1_f64( svptrue_b32(),
					&c_[result_tile_2 + ( tcol + 0 ) * rs_c], z20 );
				svst1_f64( svptrue_b32(),
					&c_[result_tile_3 + ( tcol + 0 ) * rs_c], z30 );
				svst1_f64( svptrue_b32(),
					&c_[result_tile_4 + ( tcol + 0 ) * rs_c], z40 );
				svst1_f64( svptrue_b32(),
					&c_[result_tile_5 + ( tcol + 0 ) * rs_c], z50 );
				svst1_f64( svptrue_b32(),
					&c_[result_tile_6 + ( tcol + 0 ) * rs_c], z60 );
				svst1_f64( svptrue_b32(),
					&c_[result_tile_7 + ( tcol + 0 ) * rs_c], z70 );

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

				zq0 = svld1_f64( svptrue_b32(),
					&c_[result_tile_TL_corner + ( ( ( tcol + 1 ) * rs_c ) )] );
				zq1 = svld1_f64( svptrue_b32(),
					&c_[result_tile_1 + ( ( ( tcol + 1 ) * rs_c ) )] );
				zq2 = svld1_f64( svptrue_b32(),
					&c_[result_tile_2 + ( ( ( tcol + 1 ) * rs_c ) )] );
				zq3 = svld1_f64( svptrue_b32(),
					&c_[result_tile_3 + ( ( ( tcol + 1 ) * rs_c ) )] );
				zq4 = svld1_f64( svptrue_b32(),
					&c_[result_tile_4 + ( ( ( tcol + 1 ) * rs_c ) )] );
				zq5 = svld1_f64( svptrue_b32(),
					&c_[result_tile_5 + ( ( ( tcol + 1 ) * rs_c ) )] );
				zq6 = svld1_f64( svptrue_b32(),
					&c_[result_tile_6 + ( ( ( tcol + 1 ) * rs_c ) )] );
				zq7 = svld1_f64( svptrue_b32(),
					&c_[result_tile_7 + ( ( ( tcol + 1 ) * rs_c ) )] );

				z00 = svmla_m( svptrue_b32(), z0, zq0, zbeta );
				z10 = svmla_m( svptrue_b32(), z1, zq1, zbeta );
				z20 = svmla_m( svptrue_b32(), z2, zq2, zbeta );
				z30 = svmla_m( svptrue_b32(), z3, zq3, zbeta );
				z40 = svmla_m( svptrue_b32(), z4, zq4, zbeta );
				z50 = svmla_m( svptrue_b32(), z5, zq5, zbeta );
				z60 = svmla_m( svptrue_b32(), z6, zq6, zbeta );
				z70 = svmla_m( svptrue_b32(), z7, zq7, zbeta );

				svst1_f64( svptrue_b32(),
					&c_[result_tile_TL_corner + ( tcol + 1 ) * rs_c], z00 );
				svst1_f64( svptrue_b32(),
					&c_[result_tile_1 + ( tcol + 1 ) * rs_c], z10 );
				svst1_f64( svptrue_b32(),
					&c_[result_tile_2 + ( tcol + 1 ) * rs_c], z20 );
				svst1_f64( svptrue_b32(),
					&c_[result_tile_3 + ( tcol + 1 ) * rs_c], z30 );
				svst1_f64( svptrue_b32(),
					&c_[result_tile_4 + ( tcol + 1 ) * rs_c], z40 );
				svst1_f64( svptrue_b32(),
					&c_[result_tile_5 + ( tcol + 1 ) * rs_c], z50 );
				svst1_f64( svptrue_b32(),
					&c_[result_tile_6 + ( tcol + 1 ) * rs_c], z60 );
				svst1_f64( svptrue_b32(),
					&c_[result_tile_7 + ( tcol + 1 ) * rs_c], z70 );

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

				zq0 = svld1_f64( svptrue_b32(),
					&c_[result_tile_TL_corner + ( ( ( tcol + 2 ) * rs_c ) )] );
				zq1 = svld1_f64( svptrue_b32(),
					&c_[result_tile_1 + ( ( ( tcol + 2 ) * rs_c ) )] );
				zq2 = svld1_f64( svptrue_b32(),
					&c_[result_tile_2 + ( ( ( tcol + 2 ) * rs_c ) )] );
				zq3 = svld1_f64( svptrue_b32(),
					&c_[result_tile_3 + ( ( ( tcol + 2 ) * rs_c ) )] );
				zq4 = svld1_f64( svptrue_b32(),
					&c_[result_tile_4 + ( ( ( tcol + 2 ) * rs_c ) )] );
				zq5 = svld1_f64( svptrue_b32(),
					&c_[result_tile_5 + ( ( ( tcol + 2 ) * rs_c ) )] );
				zq6 = svld1_f64( svptrue_b32(),
					&c_[result_tile_6 + ( ( ( tcol + 2 ) * rs_c ) )] );
				zq7 = svld1_f64( svptrue_b32(),
					&c_[result_tile_7 + ( ( ( tcol + 2 ) * rs_c ) )] );

				z00 = svmla_m( svptrue_b32(), z0, zq0, zbeta );
				z10 = svmla_m( svptrue_b32(), z1, zq1, zbeta );
				z20 = svmla_m( svptrue_b32(), z2, zq2, zbeta );
				z30 = svmla_m( svptrue_b32(), z3, zq3, zbeta );
				z40 = svmla_m( svptrue_b32(), z4, zq4, zbeta );
				z50 = svmla_m( svptrue_b32(), z5, zq5, zbeta );
				z60 = svmla_m( svptrue_b32(), z6, zq6, zbeta );
				z70 = svmla_m( svptrue_b32(), z7, zq7, zbeta );

				svst1_f64( svptrue_b32(),
					&c_[result_tile_TL_corner + ( tcol + 2 ) * rs_c], z00 );
				svst1_f64( svptrue_b32(),
					&c_[result_tile_1 + ( tcol + 2 ) * rs_c], z10 );
				svst1_f64( svptrue_b32(),
					&c_[result_tile_2 + ( tcol + 2 ) * rs_c], z20 );
				svst1_f64( svptrue_b32(),
					&c_[result_tile_3 + ( tcol + 2 ) * rs_c], z30 );
				svst1_f64( svptrue_b32(),
					&c_[result_tile_4 + ( tcol + 2 ) * rs_c], z40 );
				svst1_f64( svptrue_b32(),
					&c_[result_tile_5 + ( tcol + 2 ) * rs_c], z50 );
				svst1_f64( svptrue_b32(),
					&c_[result_tile_6 + ( tcol + 2 ) * rs_c], z60 );
				svst1_f64( svptrue_b32(),
					&c_[result_tile_7 + ( tcol + 2 ) * rs_c], z70 );

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

				zq0 = svld1_f64( svptrue_b32(),
					&c_[result_tile_TL_corner + ( ( ( tcol + 3 ) * rs_c ) )] );
				zq1 = svld1_f64( svptrue_b32(),
					&c_[result_tile_1 + ( ( ( tcol + 3 ) * rs_c ) )] );
				zq2 = svld1_f64( svptrue_b32(),
					&c_[result_tile_2 + ( ( ( tcol + 3 ) * rs_c ) )] );
				zq3 = svld1_f64( svptrue_b32(),
					&c_[result_tile_3 + ( ( ( tcol + 3 ) * rs_c ) )] );
				zq4 = svld1_f64( svptrue_b32(),
					&c_[result_tile_4 + ( ( ( tcol + 3 ) * rs_c ) )] );
				zq5 = svld1_f64( svptrue_b32(),
					&c_[result_tile_5 + ( ( ( tcol + 3 ) * rs_c ) )] );
				zq6 = svld1_f64( svptrue_b32(),
					&c_[result_tile_6 + ( ( ( tcol + 3 ) * rs_c ) )] );
				zq7 = svld1_f64( svptrue_b32(),
					&c_[result_tile_7 + ( ( ( tcol + 3 ) * rs_c ) )] );

				z00 = svmla_m( svptrue_b32(), z0, zq0, zbeta );
				z10 = svmla_m( svptrue_b32(), z1, zq1, zbeta );
				z20 = svmla_m( svptrue_b32(), z2, zq2, zbeta );
				z30 = svmla_m( svptrue_b32(), z3, zq3, zbeta );
				z40 = svmla_m( svptrue_b32(), z4, zq4, zbeta );
				z50 = svmla_m( svptrue_b32(), z5, zq5, zbeta );
				z60 = svmla_m( svptrue_b32(), z6, zq6, zbeta );
				z70 = svmla_m( svptrue_b32(), z7, zq7, zbeta );

				svst1_f64( svptrue_b32(),
					&c_[result_tile_TL_corner + ( tcol + 3 ) * rs_c], z00 );
				svst1_f64( svptrue_b32(),
					&c_[result_tile_1 + ( tcol + 3 ) * rs_c], z10 );
				svst1_f64( svptrue_b32(),
					&c_[result_tile_2 + ( tcol + 3 ) * rs_c], z20 );
				svst1_f64( svptrue_b32(),
					&c_[result_tile_3 + ( tcol + 3 ) * rs_c], z30 );
				svst1_f64( svptrue_b32(),
					&c_[result_tile_4 + ( tcol + 3 ) * rs_c], z40 );
				svst1_f64( svptrue_b32(),
					&c_[result_tile_5 + ( tcol + 3 ) * rs_c], z50 );
				svst1_f64( svptrue_b32(),
					&c_[result_tile_6 + ( tcol + 3 ) * rs_c], z60 );
				svst1_f64( svptrue_b32(),
					&c_[result_tile_7 + ( tcol + 3 ) * rs_c], z70 );
			}
		}
	}

	GEMM_UKR_FLUSH_CT( d );

	return;
}

