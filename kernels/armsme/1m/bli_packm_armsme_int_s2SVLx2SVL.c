/*
 *
 * BLIS An object-based framework for developing high-performance BLAS-like
 * libraries.
 *
 * Copyright (C) 2014, The University of Texas at Austin Copyright (C) 2020,
 * Linaro Limited
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are
 * met: - Redistributions of source code must retain the above copyright
 * notice, this list of conditions and the following disclaimer. -
 * Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution. - Neither the
 * name(s) of the copyright holder(s) nor the names of its contributors may
 * be used to endorse or promote products derived from this software without
 * specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS
 * IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
 * THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
 * LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
 * NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 */

#include <arm_sme.h>
#include <arm_sve.h>

#include "blis.h"

// MACROS FOR FALLTHROUGH LOGIC

// 1. Core Read & Shuffle Logic
#define READ_AND_SHUFFLE_VG4_1( tcol, zq0_, zq2_ )                          \
	svfloat32x4_t zq0 = svread_ver_za32_f32_vg4( 0, tcol );                 \
	svfloat32x4_t zq2 = svread_ver_za32_f32_vg4( 2, tcol );                 \
	zq0_ = svcreate4( svget4( zq0, 0 ), svget4( zq2, 0 ), svget4( zq0, 1 ), \
		svget4( zq2, 1 ) );                                                 \
	zq2_ = svcreate4( svget4( zq0, 2 ), svget4( zq2, 2 ), svget4( zq0, 3 ), \
		svget4( zq2, 3 ) );

#define READ_AND_SHUFFLE_VG4_2( tcol, zq1_, zq3_ )                          \
	svfloat32x4_t zq1 = svread_ver_za32_f32_vg4( 1, tcol );                 \
	svfloat32x4_t zq3 = svread_ver_za32_f32_vg4( 3, tcol );                 \
	zq1_ = svcreate4( svget4( zq1, 0 ), svget4( zq3, 0 ), svget4( zq1, 1 ), \
		svget4( zq3, 1 ) );                                                 \
	zq3_ = svcreate4( svget4( zq1, 2 ), svget4( zq3, 2 ), svget4( zq1, 3 ), \
		svget4( zq3, 3 ) );

#define READ_AND_SHUFFLE_VG2_1( tcol, zq0_ )                                \
	svfloat32x2_t zq0 = svread_ver_za32_f32_vg2( 0, tcol );                 \
	svfloat32x2_t zq2 = svread_ver_za32_f32_vg2( 2, tcol );                 \
	zq0_ = svcreate4( svget2( zq0, 0 ), svget2( zq2, 0 ), svget2( zq0, 1 ), \
		svget2( zq2, 1 ) );

#define READ_AND_SHUFFLE_VG2_2( tcol, zq1_ )                                \
	svfloat32x2_t zq1 = svread_ver_za32_f32_vg2( 1, tcol );                 \
	svfloat32x2_t zq3 = svread_ver_za32_f32_vg2( 3, tcol );                 \
	zq1_ = svcreate4( svget2( zq1, 0 ), svget2( zq3, 0 ), svget2( zq1, 1 ), \
		svget2( zq3, 1 ) );

#define READ_AND_SHUFFLE_1( tcol, zq0_ )                                \
	svfloat32_t zq0 = svread_ver_za32_m( zq0, svptrue_b32(), 0, tcol ); \
	svfloat32_t zq2 = svread_ver_za32_m( zq2, svptrue_b32(), 2, tcol ); \
	zq0_ = svcreate2( zq0, zq2 );

#define READ_AND_SHUFFLE_2( tcol, zq1_ )                                \
	svfloat32_t zq1 = svread_ver_za32_m( zq1, svptrue_b32(), 1, tcol ); \
	svfloat32_t zq3 = svread_ver_za32_m( zq3, svptrue_b32(), 3, tcol ); \
	zq1_ = svcreate2( zq1, zq3 );

// 2. Execution Blocks combined with storing

// [FULL] Stores 8 Vectors
#define OP_VG4_1( tcol, p_ )                    \
	{                                           \
		svcount_t p0 = svptrue_c32();           \
		svfloat32x4_t z0, z1;                   \
		READ_AND_SHUFFLE_VG4_1( tcol, z0, z1 ); \
		svst1( p0, &p_[0], z0 );                \
		svst1( p0, &p_[4 * SVL], z1 );          \
		p_ += ( 8 * SVL );                      \
	}

#define OP_VG4_2( tcol, p_ )                    \
	{                                           \
		svcount_t p0 = svptrue_c32();           \
		svfloat32x4_t z0, z1;                   \
		READ_AND_SHUFFLE_VG4_2( tcol, z0, z1 ); \
		svst1( p0, &p_[0], z0 );                \
		svst1( p0, &p_[4 * SVL], z1 );          \
		p_ += ( 8 * SVL );                      \
	}

// [TAIL VG2] Stores 4 Vectors
#define OP_TAIL_VG2_1( tcol, p_ )           \
	{                                       \
		svcount_t p0 = svptrue_c32();       \
		svfloat32x4_t z0;                   \
		READ_AND_SHUFFLE_VG2_1( tcol, z0 ); \
		svst1( p0, &p_[0], z0 );            \
		p_ += ( 4 * SVL );                  \
	}

#define OP_TAIL_VG2_2( tcol, p_ )           \
	{                                       \
		svcount_t p0 = svptrue_c32();       \
		svfloat32x4_t z0;                   \
		READ_AND_SHUFFLE_VG2_2( tcol, z0 ); \
		svst1( p0, &p_[0], z0 );            \
		p_ += ( 4 * SVL );                  \
	}

// [TAIL] Stores 2 Vectors
#define OP_TAIL_1( tcol, p_ )           \
	{                                   \
		svcount_t p0 = svptrue_c32();   \
		svfloat32x2_t z0;               \
		READ_AND_SHUFFLE_1( tcol, z0 ); \
		svst1( p0, &p_[0], z0 );        \
		p_ += ( 2 * SVL );              \
	}

#define OP_TAIL_2( tcol, p_ )           \
	{                                   \
		svcount_t p0 = svptrue_c32();   \
		svfloat32x2_t z0;               \
		READ_AND_SHUFFLE_2( tcol, z0 ); \
		svst1( p0, &p_[0], z0 );        \
		p_ += ( 2 * SVL );              \
	}

__arm_new( "za" ) __arm_locally_streaming void bli_spackm_armsme_int_2SVLx2SVL
	(
		conj_t conja,
		pack_t schema,
		dim_t cdim_,
		dim_t cdim_max,
		dim_t cdim_bcast,
		dim_t n_,
		dim_t n_max_,
		const void *kappa,
		const void *a, inc_t inca_, inc_t lda_,
		void *p, inc_t ldp_,
		const void *params,
		const cntx_t * cntx
	)
{
	const int64_t cdim = cdim_;
	const int64_t n = n_;
	const int64_t inca = inca_;
	const int64_t lda = lda_;
	const int64_t ldp = ldp_;

	float* restrict a_ = (float*)a;
	float* restrict p_ = (float*)p;

	uint64_t SVL = svcntsw();

	svfloat32x2_t tmp;

	const float* restrict alpha1 = a;
	float* restrict pi1 = p;

	const bool gs = ( inca != 1 && lda != 1 );

	if ( cdim_bcast == 1 && !gs )
	{
		if ( bli_seq1( *( (float*)kappa ) ) )
		{
			if ( inca == 1 )
			// continous memory.packA style
			{
				svbool_t p0 = svwhilelt_b32( (int64_t)0, cdim );
				svbool_t p1 = svwhilelt_b32( (int64_t)SVL, cdim );

				for ( dim_t k = n; k != 0; --k )
				{
					svfloat32_t z0 = svld1_f32( p0, alpha1 + 0 * SVL );
					svfloat32_t z1 = svld1_f32( p1, alpha1 + 1 * SVL );

					tmp = svcreate2( z0, z1 );

					svst1_f32_x2( svptrue_c32(), pi1, tmp );

					alpha1 += lda;
					pi1 += ldp;
				}
			}
			else
			{
				{
					for ( uint64_t col = 0; col < n; col += 2 * SVL )
					{
						int64_t valid_cols = n - col;

						// Determine total valid rows for this vertical block
						// (max 2 * SVL)
						int64_t valid_rows = ( cdim % ( 2 * SVL ) == 0 ) ?
							( 2 * SVL ) :
							( cdim % ( 2 * SVL ) );

						// Generate the 2 standard SVE column predicates for the
						// pairs of left and right tiles
						svbool_t pc0 = svwhilelt_b32( (int64_t)( 0 * SVL ),
							valid_cols );
						svbool_t pc1 = svwhilelt_b32( (int64_t)( 1 * SVL ),
							valid_cols );

						svcount_t p_all = svptrue_c32();

						if ( valid_cols >= 2 * SVL && valid_rows >= 2 * SVL )
						{
							// FAST PATH: Perfect 2*SVL x 2*SVL block
							for ( uint64_t trow = 0; trow < SVL; trow += 4 )
							{
								const uint64_t tile_UL_corner = (trow)*inca +
									col;

								// Group 1 (Tiles 0 and 1)
								svfloat32x2_t zp0 = svld1_f32_x2( p_all,
									&a_[tile_UL_corner + 0 * inca] );
								svfloat32x2_t zp1 = svld1_f32_x2( p_all,
									&a_[tile_UL_corner + 1 * inca] );
								svfloat32x2_t zp2 = svld1_f32_x2( p_all,
									&a_[tile_UL_corner + 2 * inca] );
								svfloat32x2_t zp3 = svld1_f32_x2( p_all,
									&a_[tile_UL_corner + 3 * inca] );

								const uint64_t tile_BL_corner = tile_UL_corner +
									inca * SVL;

								// Group 1 (Tiles 2 and 3)
								svfloat32x2_t zp4 = svld1_f32_x2( p_all,
									&a_[tile_BL_corner + 0 * inca] );
								svfloat32x2_t zp5 = svld1_f32_x2( p_all,
									&a_[tile_BL_corner + 1 * inca] );
								svfloat32x2_t zp6 = svld1_f32_x2( p_all,
									&a_[tile_BL_corner + 2 * inca] );
								svfloat32x2_t zp7 = svld1_f32_x2( p_all,
									&a_[tile_BL_corner + 3 * inca] );

								// Shuffle into x4 tuples
								svfloat32x4_t zq0 = svcreate4( svget2( zp0, 0 ),
									svget2( zp1, 0 ), svget2( zp2, 0 ),
									svget2( zp3, 0 ) );
								svfloat32x4_t zq1 = svcreate4( svget2( zp0, 1 ),
									svget2( zp1, 1 ), svget2( zp2, 1 ),
									svget2( zp3, 1 ) );
								svfloat32x4_t zq2 = svcreate4( svget2( zp4, 0 ),
									svget2( zp5, 0 ), svget2( zp6, 0 ),
									svget2( zp7, 0 ) );
								svfloat32x4_t zq3 = svcreate4( svget2( zp4, 1 ),
									svget2( zp5, 1 ), svget2( zp6, 1 ),
									svget2( zp7, 1 ) );

								// ZA writes
								svwrite_hor_za32_f32_vg4( 0, trow, zq0 );
								svwrite_hor_za32_f32_vg4( 1, trow, zq1 );
								svwrite_hor_za32_f32_vg4( 2, trow, zq2 );
								svwrite_hor_za32_f32_vg4( 3, trow, zq3 );
							}
						}
						else
						{
							// SAFE PATH: Matrix edge
							for ( uint64_t trow = 0; trow < SVL; trow += 4 )
							{
								const uint64_t tile_UL_corner = (trow)*inca +
									col;
								const uint64_t tile_BL_corner = tile_UL_corner +
									inca * SVL;

								// 1. Create undefined default vectors
								svfloat32_t undef_v = svundef_f32();
								svfloat32x2_t undef_x2 = svcreate2( undef_v,
									undef_v );

								// 2. Default all load arrays to empty
								svfloat32x2_t zp0 = undef_x2, zp1 = undef_x2,
											  zp2 = undef_x2, zp3 = undef_x2;
								svfloat32x2_t zp4 = undef_x2, zp5 = undef_x2,
											  zp6 = undef_x2, zp7 = undef_x2;

								// 3. Calculate rows left independently for the
								// top and bottom block
								int64_t rows_left_top = valid_rows - trow;
								int64_t rows_left_bot = valid_rows -
									( SVL + trow );

								// 4. Load top rows (writes to tiles 0,1)
								if ( rows_left_top > 0 )
									zp0 = svcreate2(
										svld1_f32( pc0,
											&a_[tile_UL_corner + 0 * inca] ),
										svld1_f32( pc1,
											&a_[tile_UL_corner + 0 * inca +
												SVL] ) );
								if ( rows_left_top > 1 )
									zp1 = svcreate2(
										svld1_f32( pc0,
											&a_[tile_UL_corner + 1 * inca] ),
										svld1_f32( pc1,
											&a_[tile_UL_corner + 1 * inca +
												SVL] ) );
								if ( rows_left_top > 2 )
									zp2 = svcreate2(
										svld1_f32( pc0,
											&a_[tile_UL_corner + 2 * inca] ),
										svld1_f32( pc1,
											&a_[tile_UL_corner + 2 * inca +
												SVL] ) );
								if ( rows_left_top > 3 )
									zp3 = svcreate2(
										svld1_f32( pc0,
											&a_[tile_UL_corner + 3 * inca] ),
										svld1_f32( pc1,
											&a_[tile_UL_corner + 3 * inca +
												SVL] ) );

								// 5. Load bottom rows (writes to tiles 2, 3)
								if ( rows_left_bot > 0 )
									zp4 = svcreate2(
										svld1_f32( pc0,
											&a_[tile_BL_corner + 0 * inca] ),
										svld1_f32( pc1,
											&a_[tile_BL_corner + 0 * inca +
												SVL] ) );
								if ( rows_left_bot > 1 )
									zp5 = svcreate2(
										svld1_f32( pc0,
											&a_[tile_BL_corner + 1 * inca] ),
										svld1_f32( pc1,
											&a_[tile_BL_corner + 1 * inca +
												SVL] ) );
								if ( rows_left_bot > 2 )
									zp6 = svcreate2(
										svld1_f32( pc0,
											&a_[tile_BL_corner + 2 * inca] ),
										svld1_f32( pc1,
											&a_[tile_BL_corner + 2 * inca +
												SVL] ) );
								if ( rows_left_bot > 3 )
									zp7 = svcreate2(
										svld1_f32( pc0,
											&a_[tile_BL_corner + 3 * inca] ),
										svld1_f32( pc1,
											&a_[tile_BL_corner + 3 * inca +
												SVL] ) );

								// 6. Shuffle into x4 tuples
								svfloat32x4_t zq0 = svcreate4( svget2( zp0, 0 ),
									svget2( zp1, 0 ), svget2( zp2, 0 ),
									svget2( zp3, 0 ) );
								svfloat32x4_t zq1 = svcreate4( svget2( zp0, 1 ),
									svget2( zp1, 1 ), svget2( zp2, 1 ),
									svget2( zp3, 1 ) );
								svfloat32x4_t zq2 = svcreate4( svget2( zp4, 0 ),
									svget2( zp5, 0 ), svget2( zp6, 0 ),
									svget2( zp7, 0 ) );
								svfloat32x4_t zq3 = svcreate4( svget2( zp4, 1 ),
									svget2( zp5, 1 ), svget2( zp6, 1 ),
									svget2( zp7, 1 ) );

								// 7. Write into ZA
								svwrite_hor_za32_f32_vg4( 0, trow, zq0 );
								svwrite_hor_za32_f32_vg4( 1, trow, zq1 );
								svwrite_hor_za32_f32_vg4( 2, trow, zq2 );
								svwrite_hor_za32_f32_vg4( 3, trow, zq3 );
							}
						}
						// Check if we are at the edge and fewer than
						// 2 * SVL columns remain
						if ( col + ( 2 * SVL ) > n )
						{
							// Total columns left to process in this tail.
							// Range: [1, 2*SVL - 1]
							int total_rem = n - col;

							// Split total_rem into columns for Tile Pair 0/2
							// (rem1) and 1/3 (rem2) Each vertical tile pair has
							// a width of SVL.
							int rem1 = ( total_rem > (int)SVL ) ? (int)SVL :
																  total_rem;
							int rem2 = ( total_rem > (int)SVL ) ?
								( total_rem - (int)SVL ) :
								0;

							// PART 1: Process Tiles 0 & 2
							if ( rem1 > 0 )
							{
								int tcol = 0;
								int n4 = rem1 >> 2;

								if ( n4 > 0 )
								{
									int i = ( n4 + 3 ) >> 2;
									// Duff's Device unrolling VG4 operations
									switch ( n4 & 3 )
									{
									case 0:
										do
										{
											OP_VG4_1( tcol, p_ );
											tcol += 4;
										case 3:
											OP_VG4_1( tcol, p_ );
											tcol += 4;
										case 2:
											OP_VG4_1( tcol, p_ );
											tcol += 4;
										case 1:
											OP_VG4_1( tcol, p_ );
											tcol += 4;
										} while ( --i > 0 );
									}
								}

								// Handle remaining 1, 2, or 3 columns
								switch ( rem1 & 3 )
								{
								case 3:
									OP_TAIL_VG2_1( tcol, p_ );
									tcol += 2;
									OP_TAIL_1( tcol, p_ );
									break;
								case 2:
									OP_TAIL_VG2_1( tcol, p_ );
									break;
								case 1:
									OP_TAIL_1( tcol, p_ );
									break;
								default:
									break;
								}
							}

							// PART 2: Process Tiles 1 & 3
							if ( rem2 > 0 )
							{
								int tcol = 0;
								int n4 = rem2 >> 2;

								if ( n4 > 0 )
								{
									int i = ( n4 + 3 ) >> 2;
									// Duff's Device unrolling VG4 operations
									switch ( n4 & 3 )
									{
									case 0:
										do
										{
											OP_VG4_2( tcol, p_ );
											tcol += 4;
										case 3:
											OP_VG4_2( tcol, p_ );
											tcol += 4;
										case 2:
											OP_VG4_2( tcol, p_ );
											tcol += 4;
										case 1:
											OP_VG4_2( tcol, p_ );
											tcol += 4;
										} while ( --i > 0 );
									}
								}

								// Handle remaining 1, 2, or 3 columns
								switch ( rem2 & 3 )
								{
								case 3:
									OP_TAIL_VG2_2( tcol, p_ );
									tcol += 2;
									OP_TAIL_2( tcol, p_ );
									break;
								case 2:
									OP_TAIL_VG2_2( tcol, p_ );
									break;
								case 1:
									OP_TAIL_2( tcol, p_ );
									break;
								default:
									break;
								}
							}
						}

						else
						{
							// Read - as - columns and store
							for ( uint64_t tcol = 0; tcol < SVL; tcol += 4 )
							{
								svcount_t p0 = svptrue_c32();

								// Each svread_ver reads 4 columns of the
								// tile(SVL).
								svfloat32x4_t zq0 = svread_ver_za32_f32_vg4(
									/* tile: */ 0, /* slice: */ tcol );
								svfloat32x4_t zq2 = svread_ver_za32_f32_vg4(
									/* tile: */ 2, /* slice: */ tcol );

								svfloat32x4_t zq1 = svread_ver_za32_f32_vg4(
									/* tile: */ 1, /* slice: */ tcol );
								svfloat32x4_t zq3 = svread_ver_za32_f32_vg4(
									/* tile: */ 3, /* slice: */ tcol );

								svfloat32x4_t zq0_ = svcreate4( 
									svget4( zq0, 0 ), svget4( zq2, 0 ),
									svget4( zq0, 1 ), svget4( zq2, 1 ) );

								svfloat32x4_t zq1_ = svcreate4( 
									svget4( zq0, 2 ), svget4( zq2, 2 ),
									svget4( zq0, 3 ), svget4( zq2, 3 ) );

								svfloat32x4_t zq2_ = svcreate4( 
									svget4( zq1, 0 ), svget4( zq3, 0 ),
									svget4( zq1, 1 ), svget4( zq3, 1 ) );

								svfloat32x4_t zq3_ = svcreate4( 
									svget4( zq1, 2 ), svget4( zq3, 2 ),
									svget4( zq1, 3 ), svget4( zq3, 3 ) );

								svst1( p0, &p_[0], zq0_ );
								svst1( p0, &p_[4 * SVL], zq1_ );
								svst1( p0, &p_[2 * SVL * SVL], zq2_ );
								svst1( p0, &p_[2 * SVL * SVL + 4 * SVL], zq3_ );
								p_ += ( 8 * SVL );
							}
							p_ += ( 2 * SVL * SVL );
						}
					}
				}

				p_ = (float*)p;
			}
		}
		else
		{
			bli_sscal2bbs_mxn
				(
				 conja,
				 cdim_,
				 n_,
				 kappa,
				 a, inca, lda,
				 p_, cdim_bcast, ldp
				);

		}
	}
	else
	{
		bli_sscal2bbs_mxn
			(
			 conja,
			 cdim_,
			 n_,
			 kappa,
			 a, inca, lda,
			 p_, cdim_bcast, ldp
			);
	}

	bli_sset0s_edge
		(
		 cdim_ * cdim_bcast, cdim_max * cdim_bcast,
		 n_, n_max_,
		 p_, ldp
		);
}
