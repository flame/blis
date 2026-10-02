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

__arm_new( "za" ) __arm_locally_streaming void bli_dgemm_armsme_int_2SVLx4SVL
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

	GEMM_UKR_SETUP_CT_AMBI( d, 2 * SVL, 4 * SVL, false );

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
			(float64_t*)( &b_[0] ) );
		svfloat64x2_t zR00 = svld1_f64_x2( svptrue_c32(),
			(float64_t*)( &a_[0] ) );

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
			(float64_t*)( &b_[( 4 * SVL )] ) );
		svfloat64x2_t zR01 = svld1_f64_x2( svptrue_c32(),
			(float64_t*)( &a_[( 2 * SVL )] ) );

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
			(float64_t*)( &b_[( 8 * SVL )] ) );
		svfloat64x2_t zR02 = svld1_f64_x2( svptrue_c32(),
			(float64_t*)( &a_[( 4 * SVL )] ) );

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
			(float64_t*)( &b_[( 12 * SVL )] ) );
		svfloat64x2_t zR03 = svld1_f64_x2( svptrue_c32(),
			(float64_t*)( &a_[( 6 * SVL )] ) );

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

		a_ += ( 8 * SVL );
		b_ += ( 2 * 8 * SVL );
	}

	for ( k_ = 0; k_ < k_left; k_ += 1 )
	{
		svfloat64x4_t zL00 = svld1_f64_x4( svptrue_c32(),
			(float64_t*)( &b_[0] ) );
		svfloat64x2_t zR00 = svld1_f64_x2( svptrue_c32(),
			(float64_t*)( &a_[0] ) );

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

		a_ += ( 2 * SVL );
		b_ += ( 4 * SVL );
	}

	double beta_ = *(double*)beta;
	double alpha_ = *(double*)alpha;

	const uint64_t result_tile_TL_corner = 0;

	svfloat64_t zbeta = svdup_f64( beta_ );
	svfloat64_t zalpha = svdup_f64( alpha_ );

	if ( cs_c == 1 )
	{
		const uint64_t result_tile_TR_corner = SVL * rs_c;

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
					&c_[result_tile_TL_corner + ( tcol + 0 ) * rs_c], z400 );

				svfloat64x4_t z600 = svcreate4( z4, z5, z6, z7 );
				svst1_f64_x4( svptrue_c32(),
					&c_[result_tile_TR_corner + ( tcol + 0 ) * rs_c], z600 );

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
					&c_[result_tile_TL_corner + ( tcol + 1 ) * rs_c], z400 );

				z600 = svcreate4( z4, z5, z6, z7 );
				svst1_f64_x4( svptrue_c32(),
					&c_[result_tile_TR_corner + ( tcol + 1 ) * rs_c], z600 );

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
					&c_[result_tile_TL_corner + ( tcol + 2 ) * rs_c], z400 );

				z600 = svcreate4( z4, z5, z6, z7 );
				svst1_f64_x4( svptrue_c32(),
					&c_[result_tile_TR_corner + ( tcol + 2 ) * rs_c], z600 );

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
					&c_[result_tile_TL_corner + ( tcol + 3 ) * rs_c], z400 );

				z600 = svcreate4( z4, z5, z6, z7 );
				svst1_f64_x4( svptrue_c32(),
					&c_[result_tile_TR_corner + ( tcol + 3 ) * rs_c], z600 );
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
					&c_[result_tile_TL_corner + ( ( ( tcol + 0 ) * rs_c ) )] );
				svfloat64x4_t zq6 = svld1_f64_x4( svptrue_c32(),
					&c_[result_tile_TR_corner + ( ( ( tcol + 0 ) * rs_c ) )] );

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
					&c_[result_tile_TL_corner + ( tcol + 0 ) * rs_c], z400 );

				svfloat64x4_t z600 = svcreate4( z80, z90, za0, zb0 );
				svst1_f64_x4( svptrue_c32(),
					&c_[result_tile_TR_corner + ( tcol + 0 ) * rs_c], z600 );

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
					&c_[result_tile_TL_corner + ( ( ( tcol + 1 ) * rs_c ) )] );
				zq6 = svld1_f64_x4( svptrue_c32(),
					&c_[result_tile_TR_corner + ( ( ( tcol + 1 ) * rs_c ) )] );

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
					&c_[result_tile_TL_corner + ( tcol + 1 ) * rs_c], z400 );

				z600 = svcreate4( z80, z90, za0, zb0 );
				svst1_f64_x4( svptrue_c32(),
					&c_[result_tile_TR_corner + ( tcol + 1 ) * rs_c], z600 );

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
					&c_[result_tile_TL_corner + ( ( ( tcol + 2 ) * rs_c ) )] );
				zq6 = svld1_f64_x4( svptrue_c32(),
					&c_[result_tile_TR_corner + ( ( ( tcol + 2 ) * rs_c ) )] );

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
					&c_[result_tile_TL_corner + ( tcol + 2 ) * rs_c], z400 );

				z600 = svcreate4( z80, z90, za0, zb0 );
				svst1_f64_x4( svptrue_c32(),
					&c_[result_tile_TR_corner + ( tcol + 2 ) * rs_c], z600 );

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
					&c_[result_tile_TL_corner + ( ( ( tcol + 3 ) * rs_c ) )] );
				zq6 = svld1_f64_x4( svptrue_c32(),
					&c_[result_tile_TR_corner + ( ( ( tcol + 3 ) * rs_c ) )] );

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
					&c_[result_tile_TL_corner + ( tcol + 3 ) * rs_c], z400 );

				z600 = svcreate4( z80, z90, za0, zb0 );
				svst1_f64_x4( svptrue_c32(),
					&c_[result_tile_TR_corner + ( tcol + 3 ) * rs_c], z600 );
			}
		}
	}
	else
	{
		const uint64_t result_tile_TR_corner = SVL * cs_c;
		const uint64_t result_tile_BL_corner = SVL * 2 * cs_c;
		const uint64_t result_tile_BR_corner = SVL * 3 * cs_c;

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
					&c_[result_tile_TL_corner + ( tcol + 0 ) * cs_c], z400 );

				svfloat64x2_t z600 = svcreate2( z1, z5 );
				svst1_f64_x2( svptrue_c32(),
					&c_[result_tile_TR_corner + ( tcol + 0 ) * cs_c], z600 );
				svfloat64x2_t z700 = svcreate2( z2, z6 );
				svst1_f64_x2( svptrue_c32(),
					&c_[result_tile_BL_corner + ( tcol + 0 ) * cs_c], z700 );

				svfloat64x2_t z800 = svcreate2( z3, z7 );
				svst1_f64_x2( svptrue_c32(),
					&c_[result_tile_BR_corner + ( tcol + 0 ) * cs_c], z800 );

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
					&c_[result_tile_TL_corner + ( tcol + 1 ) * cs_c], z400 );

				z600 = svcreate2( z1, z5 );
				svst1_f64_x2( svptrue_c32(),
					&c_[result_tile_TR_corner + ( tcol + 1 ) * cs_c], z600 );

				z700 = svcreate2( z2, z6 );
				svst1_f64_x2( svptrue_c32(),
					&c_[result_tile_BL_corner + ( tcol + 1 ) * cs_c], z700 );

				z800 = svcreate2( z3, z7 );
				svst1_f64_x2( svptrue_c32(),
					&c_[result_tile_BR_corner + ( tcol + 1 ) * cs_c], z800 );

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
					&c_[result_tile_TL_corner + ( tcol + 2 ) * cs_c], z400 );

				z600 = svcreate2( z1, z5 );
				svst1_f64_x2( svptrue_c32(),
					&c_[result_tile_TR_corner + ( tcol + 2 ) * cs_c], z600 );

				z700 = svcreate2( z2, z6 );
				svst1_f64_x2( svptrue_c32(),
					&c_[result_tile_BL_corner + ( tcol + 2 ) * cs_c], z700 );

				z800 = svcreate2( z3, z7 );
				svst1_f64_x2( svptrue_c32(),
					&c_[result_tile_BR_corner + ( tcol + 2 ) * cs_c], z800 );

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
					&c_[result_tile_TL_corner + ( tcol + 3 ) * cs_c], z400 );

				z600 = svcreate2( z1, z5 );
				svst1_f64_x2( svptrue_c32(),
					&c_[result_tile_TR_corner + ( tcol + 3 ) * cs_c], z600 );

				z700 = svcreate2( z2, z6 );
				svst1_f64_x2( svptrue_c32(),
					&c_[result_tile_BL_corner + ( tcol + 3 ) * cs_c], z700 );

				z800 = svcreate2( z3, z7 );
				svst1_f64_x2( svptrue_c32(),
					&c_[result_tile_BR_corner + ( tcol + 3 ) * cs_c], z800 );
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
					&c_[result_tile_TL_corner + ( ( ( tcol + 0 ) * cs_c ) )] );
				svfloat64x2_t zq6 = svld1_f64_x2( svptrue_c32(),
					&c_[result_tile_TR_corner + ( ( ( tcol + 0 ) * cs_c ) )] );
				svfloat64x2_t zq7 = svld1_f64_x2( svptrue_c32(),
					&c_[result_tile_BL_corner + ( ( ( tcol + 0 ) * cs_c ) )] );
				svfloat64x2_t zq8 = svld1_f64_x2( svptrue_c32(),
					&c_[result_tile_BR_corner + ( ( ( tcol + 0 ) * cs_c ) )] );

				// Scale Z regs by broadcast beta (reordered ZA tiles to match
				// horizontal order)
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
					&c_[result_tile_TL_corner + ( tcol + 0 ) * cs_c], z400 );

				svfloat64x2_t z600 = svcreate2( z60, z70 );
				svst1_f64_x2( svptrue_c32(),
					&c_[result_tile_TR_corner + ( tcol + 0 ) * cs_c], z600 );
				svfloat64x2_t z700 = svcreate2( z80, z90 );
				svst1_f64_x2( svptrue_c32(),
					&c_[result_tile_BL_corner + ( tcol + 0 ) * cs_c], z700 );

				svfloat64x2_t z800 = svcreate2( za0, zb0 );
				svst1_f64_x2( svptrue_c32(),
					&c_[result_tile_BR_corner + ( tcol + 0 ) * cs_c], z800 );

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
					&c_[result_tile_TL_corner + ( ( ( tcol + 1 ) * cs_c ) )] );
				zq6 = svld1_f64_x2( svptrue_c32(),
					&c_[result_tile_TR_corner + ( ( ( tcol + 1 ) * cs_c ) )] );
				zq7 = svld1_f64_x2( svptrue_c32(),
					&c_[result_tile_BL_corner + ( ( ( tcol + 1 ) * cs_c ) )] );
				zq8 = svld1_f64_x2( svptrue_c32(),
					&c_[result_tile_BR_corner + ( ( ( tcol + 1 ) * cs_c ) )] );

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
					&c_[result_tile_TL_corner + ( tcol + 1 ) * cs_c], z400 );

				z600 = svcreate2( z60, z70 );
				svst1_f64_x2( svptrue_c32(),
					&c_[result_tile_TR_corner + ( tcol + 1 ) * cs_c], z600 );

				z700 = svcreate2( z80, z90 );
				svst1_f64_x2( svptrue_c32(),
					&c_[result_tile_BL_corner + ( tcol + 1 ) * cs_c], z700 );

				z800 = svcreate2( za0, zb0 );
				svst1_f64_x2( svptrue_c32(),
					&c_[result_tile_BR_corner + ( tcol + 1 ) * cs_c], z800 );

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
					&c_[result_tile_TL_corner + ( ( ( tcol + 2 ) * cs_c ) )] );
				zq6 = svld1_f64_x2( svptrue_c32(),
					&c_[result_tile_TR_corner + ( ( ( tcol + 2 ) * cs_c ) )] );
				zq7 = svld1_f64_x2( svptrue_c32(),
					&c_[result_tile_BL_corner + ( ( ( tcol + 2 ) * cs_c ) )] );
				zq8 = svld1_f64_x2( svptrue_c32(),
					&c_[result_tile_BR_corner + ( ( ( tcol + 2 ) * cs_c ) )] );

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
					&c_[result_tile_TL_corner + ( tcol + 2 ) * cs_c], z400 );

				z600 = svcreate2( z60, z70 );
				svst1_f64_x2( svptrue_c32(),
					&c_[result_tile_TR_corner + ( tcol + 2 ) * cs_c], z600 );

				z700 = svcreate2( z80, z90 );
				svst1_f64_x2( svptrue_c32(),
					&c_[result_tile_BL_corner + ( tcol + 2 ) * cs_c], z700 );

				z800 = svcreate2( za0, zb0 );
				svst1_f64_x2( svptrue_c32(),
					&c_[result_tile_BR_corner + ( tcol + 2 ) * cs_c], z800 );

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
					&c_[result_tile_TL_corner + ( ( ( tcol + 3 ) * cs_c ) )] );
				zq6 = svld1_f64_x2( svptrue_c32(),
					&c_[result_tile_TR_corner + ( ( ( tcol + 3 ) * cs_c ) )] );
				zq7 = svld1_f64_x2( svptrue_c32(),
					&c_[result_tile_BL_corner + ( ( ( tcol + 3 ) * cs_c ) )] );
				zq8 = svld1_f64_x2( svptrue_c32(),
					&c_[result_tile_BR_corner + ( ( ( tcol + 3 ) * cs_c ) )] );

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
					&c_[result_tile_TL_corner + ( tcol + 3 ) * cs_c], z400 );

				z600 = svcreate2( z60, z70 );
				svst1_f64_x2( svptrue_c32(),
					&c_[result_tile_TR_corner + ( tcol + 3 ) * cs_c], z600 );

				z700 = svcreate2( z80, z90 );
				svst1_f64_x2( svptrue_c32(),
					&c_[result_tile_BL_corner + ( tcol + 3 ) * cs_c], z700 );

				z800 = svcreate2( za0, zb0 );
				svst1_f64_x2( svptrue_c32(),
					&c_[result_tile_BR_corner + ( tcol + 3 ) * cs_c], z800 );
			}
		}
	}
	GEMM_UKR_FLUSH_CT( d );

	return;
}

