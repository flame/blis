/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2014, The University of Texas at Austin
   Copyright (C) 2020, Linaro Limited

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
#include <arm_sve.h>

#include "blis.h"

// MACROS FOR FALLTHROUGH LOGIC

// PATH 1

// 1. Core Read, Shuffle & Store Logic for Pairs
// Reads 4 columns from Top(TA) and Bottom(TB) tiles
// Interleaves: [TB_0, TA_0, TB_1, TA_1] ...
#define OP_VG4_PAIR( TA, TB, TCOL, P_PTR )                              \
	{                                                                   \
		svcount_t p_all = svptrue_c64();                                \
		svfloat64x4_t zA = svread_ver_za64_f64_vg4( TA, TCOL );         \
		svfloat64x4_t zB = svread_ver_za64_f64_vg4( TB, TCOL );         \
                                                                        \
		/* Shuffle Cols 0 & 1: [B0, A0, B1, A1] */                      \
		svfloat64x4_t z0 = svcreate4( svget4( zB, 0 ), svget4( zA, 0 ), \
			svget4( zB, 1 ), svget4( zA, 1 ) );                         \
		/* Shuffle Cols 2 & 3: [B2, A2, B3, A3] */                      \
		svfloat64x4_t z1 = svcreate4( svget4( zB, 2 ), svget4( zA, 2 ), \
			svget4( zB, 3 ), svget4( zA, 3 ) );                         \
                                                                        \
		svst1( p_all, P_PTR, z0 );                                      \
		svst1( p_all, P_PTR + 4 * SVL, z1 );                            \
		P_PTR += ( 8 * SVL );                                           \
	}

#define OP_VG2_PAIR( TA, TB, TCOL, P_PTR )                              \
	{                                                                   \
		svcount_t p_all = svptrue_c64();                                \
		svfloat64x2_t zA = svread_ver_za64_f64_vg2( TA, TCOL );         \
		svfloat64x2_t zB = svread_ver_za64_f64_vg2( TB, TCOL );         \
                                                                        \
		/* Shuffle Cols 0 & 1 */                                        \
		svfloat64x4_t z0 = svcreate4( svget2( zB, 0 ), svget2( zA, 0 ), \
			svget2( zB, 1 ), svget2( zA, 1 ) );                         \
                                                                        \
		svst1( p_all, P_PTR, z0 );                                      \
		P_PTR += ( 4 * SVL );                                           \
	}

#define OP_VG1_PAIR( TA, TB, TCOL, P_PTR )                          \
	{                                                               \
		svbool_t p_true = svptrue_b64();                            \
		svcount_t p_cnt = svptrue_c64();                            \
                                                                    \
		svfloat64_t vA = svread_ver_za64_m( vA, p_true, TA, TCOL ); \
		svfloat64_t vB = svread_ver_za64_m( vB, p_true, TB, TCOL ); \
                                                                    \
		/* Store as pair [B, A] (2 vectors total) */                \
		svfloat64x2_t z0 = svcreate2( vB, vA );                     \
		svst1( p_cnt, P_PTR, z0 );                                  \
		P_PTR += ( 2 * SVL );                                       \
	}

// 2. Duff's Device Logic for a Partial Pair Tile
#define PROCESS_PARTIAL_PAIR( TA, TB, REM, P_BASE )  \
	{                                                \
		int tcol = 0;                                \
		double* p_curr = P_BASE;                     \
		int n4 = REM >> 2;                           \
                                                     \
		if ( n4 > 0 )                                \
		{                                            \
			switch ( n4 & 3 )                        \
			{                                        \
			case 0:                                  \
				OP_VG4_PAIR( TA, TB, tcol, p_curr ); \
				tcol += 4;                           \
			case 3:                                  \
				OP_VG4_PAIR( TA, TB, tcol, p_curr ); \
				tcol += 4;                           \
			case 2:                                  \
				OP_VG4_PAIR( TA, TB, tcol, p_curr ); \
				tcol += 4;                           \
			case 1:                                  \
				OP_VG4_PAIR( TA, TB, tcol, p_curr ); \
				tcol += 4;                           \
			}                                        \
		}                                            \
		switch ( REM & 3 )                           \
		{                                            \
		case 3:                                      \
			OP_VG2_PAIR( TA, TB, tcol, p_curr );     \
			tcol += 2;                               \
			OP_VG1_PAIR( TA, TB, tcol, p_curr );     \
			break;                                   \
		case 2:                                      \
			OP_VG2_PAIR( TA, TB, tcol, p_curr );     \
			break;                                   \
		case 1:                                      \
			OP_VG1_PAIR( TA, TB, tcol, p_curr );     \
			break;                                   \
		default:                                     \
			break;                                   \
		}                                            \
		P_BASE += ( 2 * SVL * SVL );                 \
	}

// 3. Logic for a Full Pair Tile
#define PROCESS_FULL_PAIR( TA, TB, P_BASE )         \
	{                                               \
		double* p_curr = P_BASE;                    \
		for ( int tcol = 0; tcol < SVL; tcol += 4 ) \
		{                                           \
			OP_VG4_PAIR( TA, TB, tcol, p_curr );    \
		}                                           \
		P_BASE += ( 2 * SVL * SVL );                \
	}

// PATH 2

// 1. Core Read, Shuffle & Store Logic for Groups
// Reads 4 columns from 4 Tiles (T0..T3) and stores interleaved
#define OP_VG4_GROUP( T0, T1, T2, T3, TCOL, P_PTR )                       \
	{                                                                     \
		svcount_t p_all = svptrue_c64();                                  \
		svfloat64x4_t z0 = svread_ver_za64_f64_vg4( T0, TCOL );           \
		svfloat64x4_t z1 = svread_ver_za64_f64_vg4( T1, TCOL );           \
		svfloat64x4_t z2 = svread_ver_za64_f64_vg4( T2, TCOL );           \
		svfloat64x4_t z3 = svread_ver_za64_f64_vg4( T3, TCOL );           \
                                                                          \
		/* Shuffle and Store Column 0 */                                  \
		svfloat64x4_t res0 = svcreate4( svget4( z0, 0 ), svget4( z1, 0 ), \
			svget4( z2, 0 ), svget4( z3, 0 ) );                           \
		svst1( p_all, P_PTR, res0 );                                      \
                                                                          \
		/* Shuffle and Store Column 1 */                                  \
		svfloat64x4_t res1 = svcreate4( svget4( z0, 1 ), svget4( z1, 1 ), \
			svget4( z2, 1 ), svget4( z3, 1 ) );                           \
		svst1( p_all, P_PTR + 4 * SVL, res1 );                            \
                                                                          \
		/* Shuffle and Store Column 2 */                                  \
		svfloat64x4_t res2 = svcreate4( svget4( z0, 2 ), svget4( z1, 2 ), \
			svget4( z2, 2 ), svget4( z3, 2 ) );                           \
		svst1( p_all, P_PTR + 8 * SVL, res2 );                            \
                                                                          \
		/* Shuffle and Store Column 3 */                                  \
		svfloat64x4_t res3 = svcreate4( svget4( z0, 3 ), svget4( z1, 3 ), \
			svget4( z2, 3 ), svget4( z3, 3 ) );                           \
		svst1( p_all, P_PTR + 12 * SVL, res3 );                           \
                                                                          \
		P_PTR += ( 16 * SVL );                                            \
	}

#define OP_VG2_GROUP( T0, T1, T2, T3, TCOL, P_PTR )                       \
	{                                                                     \
		svcount_t p_all = svptrue_c64();                                  \
		svfloat64x2_t z0 = svread_ver_za64_f64_vg2( T0, TCOL );           \
		svfloat64x2_t z1 = svread_ver_za64_f64_vg2( T1, TCOL );           \
		svfloat64x2_t z2 = svread_ver_za64_f64_vg2( T2, TCOL );           \
		svfloat64x2_t z3 = svread_ver_za64_f64_vg2( T3, TCOL );           \
                                                                          \
		/* Shuffle and Store Column 0 */                                  \
		svfloat64x4_t res0 = svcreate4( svget2( z0, 0 ), svget2( z1, 0 ), \
			svget2( z2, 0 ), svget2( z3, 0 ) );                           \
		svst1( p_all, P_PTR, res0 );                                      \
                                                                          \
		/* Shuffle and Store Column 1 */                                  \
		svfloat64x4_t res1 = svcreate4( svget2( z0, 1 ), svget2( z1, 1 ), \
			svget2( z2, 1 ), svget2( z3, 1 ) );                           \
		svst1( p_all, P_PTR + 4 * SVL, res1 );                            \
                                                                          \
		P_PTR += ( 8 * SVL );                                             \
	}

#define OP_VG1_GROUP( T0, T1, T2, T3, TCOL, P_PTR )                 \
	{                                                               \
		svbool_t p_true = svptrue_b64();                            \
		svcount_t p_cnt = svptrue_c64();                            \
                                                                    \
		svfloat64_t z0 = svread_ver_za64_m( z0, p_true, T0, TCOL ); \
		svfloat64_t z1 = svread_ver_za64_m( z1, p_true, T1, TCOL ); \
		svfloat64_t z2 = svread_ver_za64_m( z2, p_true, T2, TCOL ); \
		svfloat64_t z3 = svread_ver_za64_m( z3, p_true, T3, TCOL ); \
                                                                    \
		/* Shuffle and Store */                                     \
		svfloat64x4_t res0 = svcreate4( z0, z1, z2, z3 );           \
		svst1( p_cnt, P_PTR, res0 );                                \
                                                                    \
		P_PTR += ( 4 * SVL );                                       \
	}

// 2. Logic for all group cases
#define PROCESS_GROUP( T0, T1, T2, T3, REM, P_BASE )      \
	{                                                     \
		int tcol = 0;                                     \
		int local_rem = REM;                              \
		double* p_curr = P_BASE;                          \
                                                          \
		while ( local_rem >= 4 )                          \
		{                                                 \
			OP_VG4_GROUP( T0, T1, T2, T3, tcol, p_curr ); \
			tcol += 4;                                    \
			local_rem -= 4;                               \
		}                                                 \
		if ( local_rem >= 2 )                             \
		{                                                 \
			OP_VG2_GROUP( T0, T1, T2, T3, tcol, p_curr ); \
			tcol += 2;                                    \
			local_rem -= 2;                               \
		}                                                 \
		if ( local_rem >= 1 )                             \
		{                                                 \
			OP_VG1_GROUP( T0, T1, T2, T3, tcol, p_curr ); \
		}                                                 \
	}

__arm_new( "za" ) __arm_locally_streaming void bli_dpackm_armsme_int_4SVLx2SVL
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

	double* restrict a_ = (double*)a;
	double* restrict p_ = (double*)p;

	uint64_t SVL = svcntsd();

	svfloat64x4_t tmp;
	svfloat64x2_t tmp2;

	const double* restrict alpha1 = a;
	double* restrict pi1 = p;

	const bool gs = ( inca != 1 && lda != 1 );

	if ( !gs && cdim_bcast )
	{
		if ( bli_deq1( *( (double*)kappa ) ) )
		{
			if ( inca == 1 && ldp == 4 * SVL )
			// continous memory.packA style
			{
				svbool_t p0 = svwhilelt_b64( (int64_t)0, cdim );
				svbool_t p1 = svwhilelt_b64( (int64_t)SVL, cdim );
				svbool_t p2 = svwhilelt_b64( (int64_t)( 2 * SVL ), cdim );
				svbool_t p3 = svwhilelt_b64( (int64_t)( 3 * SVL ), cdim );

				for ( dim_t k = n; k != 0; --k )
				{
					svfloat64_t z0 = svld1_f64( p0, alpha1 + 0 * SVL );
					svfloat64_t z1 = svld1_f64( p1, alpha1 + 1 * SVL );
					svfloat64_t z2 = svld1_f64( p2, alpha1 + 2 * SVL );
					svfloat64_t z3 = svld1_f64( p3, alpha1 + 3 * SVL );

					tmp = svcreate4( z0, z1, z2, z3 );

					svst1_f64_x4( svptrue_c64(), pi1, tmp );

					alpha1 += lda;
					pi1 += ldp;
				}
			}
			if ( inca == 1 && ldp == 2 * SVL )
			// continous memory.packA style
			{
				svbool_t p0 = svwhilelt_b64( (int64_t)0, cdim );
				svbool_t p1 = svwhilelt_b64( (int64_t)SVL, cdim );
				for ( dim_t k = n; k != 0; --k )
				{
					svfloat64_t z0 = svld1_f64( p0, alpha1 + 0 * SVL );
					svfloat64_t z1 = svld1_f64( p1, alpha1 + 1 * SVL );

					tmp2 = svcreate2( z0, z1 );

					svst1_f64_x2( svptrue_c64(), pi1, tmp2 );

					alpha1 += lda;
					pi1 += ldp;
				}
			}
			else if ( inca != 1 && ldp == 2 * SVL )
			{
				for ( uint64_t col = 0; col < n; col += 4 * SVL )
				{
					int64_t valid_cols = n - col;

					// Determine total valid rows for this vertical block
					// (max 2 * SVL)
					int64_t valid_rows = ( cdim % ( 2 * SVL ) == 0 ) ?
						( 2 * SVL ) :
						( cdim % ( 2 * SVL ) );

					// Generate the 4 standard SVE column predicates for the
					// left-to-right f64 chunks
					svbool_t pc0 = svwhilelt_b64( (int64_t)( 0 * SVL ),
						valid_cols );
					svbool_t pc1 = svwhilelt_b64( (int64_t)( 1 * SVL ),
						valid_cols );
					svbool_t pc2 = svwhilelt_b64( (int64_t)( 2 * SVL ),
						valid_cols );
					svbool_t pc3 = svwhilelt_b64( (int64_t)( 3 * SVL ),
						valid_cols );

					svcount_t p_all = svptrue_c64();

					if ( valid_cols >= 4 * SVL && valid_rows >= 2 * SVL )
					{
						// FAST PATH: Perfect 2*SVL x 4*SVL block
						for ( uint64_t trow = 0; trow < SVL; trow += 4 )
						{
							const uint64_t tile_UL_corner = (trow)*inca + col;
							const uint64_t tile_UR_corner = tile_UL_corner +
								inca * SVL;

							// Group 1 (Tiles 4 through 7)
							svfloat64x4_t zp0 = svld1_f64_x4( p_all,
								&a_[tile_UL_corner + 0 * inca] );
							svfloat64x4_t zp1 = svld1_f64_x4( p_all,
								&a_[tile_UL_corner + 1 * inca] );
							svfloat64x4_t zp2 = svld1_f64_x4( p_all,
								&a_[tile_UL_corner + 2 * inca] );
							svfloat64x4_t zp3 = svld1_f64_x4( p_all,
								&a_[tile_UL_corner + 3 * inca] );

							svfloat64x4_t zq0 = svcreate4( svget4( zp0, 0 ),
								svget4( zp1, 0 ), svget4( zp2, 0 ),
								svget4( zp3, 0 ) );
							svfloat64x4_t zq1 = svcreate4( svget4( zp0, 1 ),
								svget4( zp1, 1 ), svget4( zp2, 1 ),
								svget4( zp3, 1 ) );
							svfloat64x4_t zq2 = svcreate4( svget4( zp0, 2 ),
								svget4( zp1, 2 ), svget4( zp2, 2 ),
								svget4( zp3, 2 ) );
							svfloat64x4_t zq3 = svcreate4( svget4( zp0, 3 ),
								svget4( zp1, 3 ), svget4( zp2, 3 ),
								svget4( zp3, 3 ) );

							svwrite_hor_za64_f64_vg4( 4, trow, zq0 );
							svwrite_hor_za64_f64_vg4( 5, trow, zq1 );
							svwrite_hor_za64_f64_vg4( 6, trow, zq2 );
							svwrite_hor_za64_f64_vg4( 7, trow, zq3 );

							// Group 2 (Tiles 0 through 3)
							svfloat64x4_t zp4 = svld1_f64_x4( p_all,
								&a_[tile_UR_corner + 0 * inca] );
							svfloat64x4_t zp5 = svld1_f64_x4( p_all,
								&a_[tile_UR_corner + 1 * inca] );
							svfloat64x4_t zp6 = svld1_f64_x4( p_all,
								&a_[tile_UR_corner + 2 * inca] );
							svfloat64x4_t zp7 = svld1_f64_x4( p_all,
								&a_[tile_UR_corner + 3 * inca] );

							svfloat64x4_t zq4 = svcreate4( svget4( zp4, 0 ),
								svget4( zp5, 0 ), svget4( zp6, 0 ),
								svget4( zp7, 0 ) );
							svfloat64x4_t zq5 = svcreate4( svget4( zp4, 1 ),
								svget4( zp5, 1 ), svget4( zp6, 1 ),
								svget4( zp7, 1 ) );
							svfloat64x4_t zq6 = svcreate4( svget4( zp4, 2 ),
								svget4( zp5, 2 ), svget4( zp6, 2 ),
								svget4( zp7, 2 ) );
							svfloat64x4_t zq7 = svcreate4( svget4( zp4, 3 ),
								svget4( zp5, 3 ), svget4( zp6, 3 ),
								svget4( zp7, 3 ) );

							svwrite_hor_za64_f64_vg4( 0, trow, zq4 );
							svwrite_hor_za64_f64_vg4( 1, trow, zq5 );
							svwrite_hor_za64_f64_vg4( 2, trow, zq6 );
							svwrite_hor_za64_f64_vg4( 3, trow, zq7 );
						}
					}
					else
					{
						// SAFE PATH: Matrix edge
						for ( uint64_t trow = 0; trow < SVL; trow += 4 )
						{
							const uint64_t tile_UL_corner = (trow)*inca + col;
							const uint64_t tile_UR_corner = tile_UL_corner +
								inca * SVL;

							// 1. Create undefined default vectors
							svfloat64_t undef_v = svundef_f64();
							svfloat64x4_t undef_x4 = svcreate4( undef_v,
								undef_v, undef_v, undef_v );

							// 2. Default all load arrays to empty to guarantee
							// safety
							svfloat64x4_t zp0 = undef_x4, zp1 = undef_x4,
										  zp2 = undef_x4, zp3 = undef_x4;
							svfloat64x4_t zp4 = undef_x4, zp5 = undef_x4,
										  zp6 = undef_x4, zp7 = undef_x4;

							// 3. Calculate rows left independently for the top
							// and bottom block
							int64_t rows_left_top = valid_rows - trow;
							int64_t rows_left_bot = valid_rows - ( SVL + trow );

							// 4. Load top rows (writes to tiles 4, 5, 6, 7)
							if ( rows_left_top > 0 )
							{
								zp0 = svcreate4( svld1_f64( pc0,
													 &a_[tile_UL_corner +
														 0 * inca + 0 * SVL] ),
									svld1_f64( pc1,
										&a_[tile_UL_corner + 0 * inca +
											1 * SVL] ),
									svld1_f64( pc2,
										&a_[tile_UL_corner + 0 * inca +
											2 * SVL] ),
									svld1_f64( pc3,
										&a_[tile_UL_corner + 0 * inca +
											3 * SVL] ) );
							}
							if ( rows_left_top > 1 )
							{
								zp1 = svcreate4( svld1_f64( pc0,
													 &a_[tile_UL_corner +
														 1 * inca + 0 * SVL] ),
									svld1_f64( pc1,
										&a_[tile_UL_corner + 1 * inca +
											1 * SVL] ),
									svld1_f64( pc2,
										&a_[tile_UL_corner + 1 * inca +
											2 * SVL] ),
									svld1_f64( pc3,
										&a_[tile_UL_corner + 1 * inca +
											3 * SVL] ) );
							}
							if ( rows_left_top > 2 )
							{
								zp2 = svcreate4( svld1_f64( pc0,
													 &a_[tile_UL_corner +
														 2 * inca + 0 * SVL] ),
									svld1_f64( pc1,
										&a_[tile_UL_corner + 2 * inca +
											1 * SVL] ),
									svld1_f64( pc2,
										&a_[tile_UL_corner + 2 * inca +
											2 * SVL] ),
									svld1_f64( pc3,
										&a_[tile_UL_corner + 2 * inca +
											3 * SVL] ) );
							}
							if ( rows_left_top > 3 )
							{
								zp3 = svcreate4( svld1_f64( pc0,
													 &a_[tile_UL_corner +
														 3 * inca + 0 * SVL] ),
									svld1_f64( pc1,
										&a_[tile_UL_corner + 3 * inca +
											1 * SVL] ),
									svld1_f64( pc2,
										&a_[tile_UL_corner + 3 * inca +
											2 * SVL] ),
									svld1_f64( pc3,
										&a_[tile_UL_corner + 3 * inca +
											3 * SVL] ) );
							}

							// 5. Load bottom rows (writes to tiles 0, 1, 2, 3)
							if ( rows_left_bot > 0 )
							{
								zp4 = svcreate4( svld1_f64( pc0,
													 &a_[tile_UR_corner +
														 0 * inca + 0 * SVL] ),
									svld1_f64( pc1,
										&a_[tile_UR_corner + 0 * inca +
											1 * SVL] ),
									svld1_f64( pc2,
										&a_[tile_UR_corner + 0 * inca +
											2 * SVL] ),
									svld1_f64( pc3,
										&a_[tile_UR_corner + 0 * inca +
											3 * SVL] ) );
							}
							if ( rows_left_bot > 1 )
							{
								zp5 = svcreate4( svld1_f64( pc0,
													 &a_[tile_UR_corner +
														 1 * inca + 0 * SVL] ),
									svld1_f64( pc1,
										&a_[tile_UR_corner + 1 * inca +
											1 * SVL] ),
									svld1_f64( pc2,
										&a_[tile_UR_corner + 1 * inca +
											2 * SVL] ),
									svld1_f64( pc3,
										&a_[tile_UR_corner + 1 * inca +
											3 * SVL] ) );
							}
							if ( rows_left_bot > 2 )
							{
								zp6 = svcreate4( svld1_f64( pc0,
													 &a_[tile_UR_corner +
														 2 * inca + 0 * SVL] ),
									svld1_f64( pc1,
										&a_[tile_UR_corner + 2 * inca +
											1 * SVL] ),
									svld1_f64( pc2,
										&a_[tile_UR_corner + 2 * inca +
											2 * SVL] ),
									svld1_f64( pc3,
										&a_[tile_UR_corner + 2 * inca +
											3 * SVL] ) );
							}
							if ( rows_left_bot > 3 )
							{
								zp7 = svcreate4( svld1_f64( pc0,
													 &a_[tile_UR_corner +
														 3 * inca + 0 * SVL] ),
									svld1_f64( pc1,
										&a_[tile_UR_corner + 3 * inca +
											1 * SVL] ),
									svld1_f64( pc2,
										&a_[tile_UR_corner + 3 * inca +
											2 * SVL] ),
									svld1_f64( pc3,
										&a_[tile_UR_corner + 3 * inca +
											3 * SVL] ) );
							}

							// 6. Shuffle into x4 tuples
							svfloat64x4_t zq0 = svcreate4( svget4( zp0, 0 ),
								svget4( zp1, 0 ), svget4( zp2, 0 ),
								svget4( zp3, 0 ) );
							svfloat64x4_t zq1 = svcreate4( svget4( zp0, 1 ),
								svget4( zp1, 1 ), svget4( zp2, 1 ),
								svget4( zp3, 1 ) );
							svfloat64x4_t zq2 = svcreate4( svget4( zp0, 2 ),
								svget4( zp1, 2 ), svget4( zp2, 2 ),
								svget4( zp3, 2 ) );
							svfloat64x4_t zq3 = svcreate4( svget4( zp0, 3 ),
								svget4( zp1, 3 ), svget4( zp2, 3 ),
								svget4( zp3, 3 ) );

							svfloat64x4_t zq4 = svcreate4( svget4( zp4, 0 ),
								svget4( zp5, 0 ), svget4( zp6, 0 ),
								svget4( zp7, 0 ) );
							svfloat64x4_t zq5 = svcreate4( svget4( zp4, 1 ),
								svget4( zp5, 1 ), svget4( zp6, 1 ),
								svget4( zp7, 1 ) );
							svfloat64x4_t zq6 = svcreate4( svget4( zp4, 2 ),
								svget4( zp5, 2 ), svget4( zp6, 2 ),
								svget4( zp7, 2 ) );
							svfloat64x4_t zq7 = svcreate4( svget4( zp4, 3 ),
								svget4( zp5, 3 ), svget4( zp6, 3 ),
								svget4( zp7, 3 ) );

							// 7. Write into ZA
							svwrite_hor_za64_f64_vg4( 4, trow, zq0 );
							svwrite_hor_za64_f64_vg4( 5, trow, zq1 );
							svwrite_hor_za64_f64_vg4( 6, trow, zq2 );
							svwrite_hor_za64_f64_vg4( 7, trow, zq3 );

							svwrite_hor_za64_f64_vg4( 0, trow, zq4 );
							svwrite_hor_za64_f64_vg4( 1, trow, zq5 );
							svwrite_hor_za64_f64_vg4( 2, trow, zq6 );
							svwrite_hor_za64_f64_vg4( 3, trow, zq7 );
						}
					}
					// Check if we are at the edge and fewer than
					// 4 * SVL columns remain
					if ( col + ( 4 * SVL ) > n )
					{
						int total_rem = n - col;

						// --- PAIR 1: Tiles 0 and 4 ---
						if ( total_rem >= (int)SVL )
						{
							PROCESS_FULL_PAIR( 0, 4, p_ );
							total_rem -= SVL;
						}
						else
						{
							PROCESS_PARTIAL_PAIR( 0, 4, total_rem, p_ );
							total_rem = 0;
						}

						// --- PAIR 2: Tiles 1 and 5 ---
						if ( total_rem > 0 )
						{
							if ( total_rem >= (int)SVL )
							{
								PROCESS_FULL_PAIR( 1, 5, p_ );
								total_rem -= SVL;
							}
							else
							{
								PROCESS_PARTIAL_PAIR( 1, 5, total_rem, p_ );
								total_rem = 0;
							}
						}

						// --- PAIR 3: Tiles 2 and 6 ---
						if ( total_rem > 0 )
						{
							if ( total_rem >= (int)SVL )
							{
								PROCESS_FULL_PAIR( 2, 6, p_ );
								total_rem -= SVL;
							}
							else
							{
								PROCESS_PARTIAL_PAIR( 2, 6, total_rem, p_ );
								total_rem = 0;
							}
						}

						// --- PAIR 4: Tiles 3 and 7 ---
						if ( total_rem > 0 )
						{
							PROCESS_PARTIAL_PAIR( 3, 7, total_rem, p_ );
						}
					}
					else
					{
						// Read - as - columns and store
						for ( uint64_t tcol = 0; tcol < SVL; tcol += 4 )
						{
							svcount_t p0 = svptrue_c32();

							// Each svread_ver reads 4 columns of the tile(SVL).
							svfloat64x4_t zq0 = svread_ver_za64_f64_vg4(
								/* tile: */ 0, /* slice: */ tcol );
							svfloat64x4_t zq2 = svread_ver_za64_f64_vg4(
								/* tile: */ 1, /* slice: */ tcol );

							svfloat64x4_t zq1 = svread_ver_za64_f64_vg4(
								/* tile: */ 4, /* slice: */ tcol );
							svfloat64x4_t zq3 = svread_ver_za64_f64_vg4(
								/* tile: */ 5, /* slice: */ tcol );

							svfloat64x4_t zq0_ = svcreate4( svget4( zq1, 0 ),
								svget4( zq0, 0 ), svget4( zq1, 1 ),
								svget4( zq0, 1 ) );

							svfloat64x4_t zq1_ = svcreate4( svget4( zq1, 2 ),
								svget4( zq0, 2 ), svget4( zq1, 3 ),
								svget4( zq0, 3 ) );

							svfloat64x4_t zq2_ = svcreate4( svget4( zq3, 0 ),
								svget4( zq2, 0 ), svget4( zq3, 1 ),
								svget4( zq2, 1 ) );

							svfloat64x4_t zq3_ = svcreate4( svget4( zq3, 2 ),
								svget4( zq2, 2 ), svget4( zq3, 3 ),
								svget4( zq2, 3 ) );

							svst1( p0, &p_[0], zq0_ );
							svst1( p0, &p_[4 * SVL], zq1_ );
							svst1( p0, &p_[2 * SVL * SVL], zq2_ );
							svst1( p0, &p_[2 * SVL * SVL + 4 * SVL], zq3_ );

							// Each svread_ver reads 4 columns of the tile(SVL).
							svfloat64x4_t zq4 = svread_ver_za64_f64_vg4(
								/* tile: */ 2, /* slice: */ tcol );
							svfloat64x4_t zq5 = svread_ver_za64_f64_vg4(
								/* tile: */ 6, /* slice: */ tcol );

							svfloat64x4_t zq6 = svread_ver_za64_f64_vg4(
								/* tile: */ 3, /* slice: */ tcol );
							svfloat64x4_t zq7 = svread_ver_za64_f64_vg4(
								/* tile: */ 7, /* slice: */ tcol );

							svfloat64x4_t zq4_ = svcreate4( svget4( zq5, 0 ),
								svget4( zq4, 0 ), svget4( zq5, 1 ),
								svget4( zq4, 1 ) );

							svfloat64x4_t zq5_ = svcreate4( svget4( zq5, 2 ),
								svget4( zq4, 2 ), svget4( zq5, 3 ),
								svget4( zq4, 3 ) );

							svfloat64x4_t zq6_ = svcreate4( svget4( zq7, 0 ),
								svget4( zq6, 0 ), svget4( zq7, 1 ),
								svget4( zq6, 1 ) );

							svfloat64x4_t zq7_ = svcreate4( svget4( zq7, 2 ),
								svget4( zq6, 2 ), svget4( zq7, 3 ),
								svget4( zq6, 3 ) );

							svst1( p0, &p_[4 * SVL * SVL], zq4_ );
							svst1( p0, &p_[4 * SVL * SVL + 4 * SVL], zq5_ );
							svst1( p0, &p_[6 * SVL * SVL], zq6_ );
							svst1( p0, &p_[6 * SVL * SVL + 4 * SVL], zq7_ );

							p_ += ( 8 * SVL );
						}
						p_ += ( 6 * SVL * SVL );
					}
				}

				p_ = (double*)p;
			}
			else if ( inca != 1 && ldp == 4 * SVL )
			{
				for ( uint64_t col = 0; col < n; col += 2 * SVL )
				{
					int64_t valid_cols = n - col;

					// Determine total valid rows for this vertical block
					// (max 4 * SVL)
					int64_t valid_rows = ( cdim % ( 4 * SVL ) == 0 ) ?
						( 4 * SVL ) :
						( cdim % ( 4 * SVL ) );

					// Generate the 2 standard SVE column predicates for the
					// left and right f64 chunks
					svbool_t pc0 = svwhilelt_b64( (int64_t)( 0 * SVL ),
						valid_cols );
					svbool_t pc1 = svwhilelt_b64( (int64_t)( 1 * SVL ),
						valid_cols );

					svcount_t p_all = svptrue_c64();

					if ( valid_cols >= 2 * SVL && valid_rows >= 4 * SVL )
					{
						// FAST PATH: Perfect 4*SVL x 2*SVL block
						for ( uint64_t trow = 0; trow < SVL; trow += 4 )
						{
							const uint64_t tile_UL_corner = (trow)*inca + col;
							const uint64_t tile_UR_corner = tile_UL_corner +
								inca * SVL;
							const uint64_t tile_BL_corner = tile_UL_corner +
								inca * 2 * SVL;
							const uint64_t tile_BR_corner = tile_UL_corner +
								inca * 3 * SVL;

							// Group 1 (Tiles 0 & 4)
							svfloat64x2_t zp0 = svld1_f64_x2( p_all,
								&a_[tile_UL_corner + 0 * inca] );
							svfloat64x2_t zp1 = svld1_f64_x2( p_all,
								&a_[tile_UL_corner + 1 * inca] );
							svfloat64x2_t zp2 = svld1_f64_x2( p_all,
								&a_[tile_UL_corner + 2 * inca] );
							svfloat64x2_t zp3 = svld1_f64_x2( p_all,
								&a_[tile_UL_corner + 3 * inca] );

							// Group 2 (Tiles 1 & 5)
							svfloat64x2_t zp4 = svld1_f64_x2( p_all,
								&a_[tile_UR_corner + 0 * inca] );
							svfloat64x2_t zp5 = svld1_f64_x2( p_all,
								&a_[tile_UR_corner + 1 * inca] );
							svfloat64x2_t zp6 = svld1_f64_x2( p_all,
								&a_[tile_UR_corner + 2 * inca] );
							svfloat64x2_t zp7 = svld1_f64_x2( p_all,
								&a_[tile_UR_corner + 3 * inca] );

							svfloat64x4_t zq0 = svcreate4( svget2( zp0, 0 ),
								svget2( zp1, 0 ), svget2( zp2, 0 ),
								svget2( zp3, 0 ) );
							svfloat64x4_t zq1 = svcreate4( svget2( zp0, 1 ),
								svget2( zp1, 1 ), svget2( zp2, 1 ),
								svget2( zp3, 1 ) );
							svfloat64x4_t zq2 = svcreate4( svget2( zp4, 0 ),
								svget2( zp5, 0 ), svget2( zp6, 0 ),
								svget2( zp7, 0 ) );
							svfloat64x4_t zq3 = svcreate4( svget2( zp4, 1 ),
								svget2( zp5, 1 ), svget2( zp6, 1 ),
								svget2( zp7, 1 ) );

							svwrite_hor_za64_f64_vg4( 0, trow, zq0 );
							svwrite_hor_za64_f64_vg4( 4, trow, zq1 );
							svwrite_hor_za64_f64_vg4( 1, trow, zq2 );
							svwrite_hor_za64_f64_vg4( 5, trow, zq3 );

							// Group 3 (Tiles 2 & 6)
							svfloat64x2_t zp8 = svld1_f64_x2( p_all,
								&a_[tile_BL_corner + 0 * inca] );
							svfloat64x2_t zp9 = svld1_f64_x2( p_all,
								&a_[tile_BL_corner + 1 * inca] );
							svfloat64x2_t zp10 = svld1_f64_x2( p_all,
								&a_[tile_BL_corner + 2 * inca] );
							svfloat64x2_t zp11 = svld1_f64_x2( p_all,
								&a_[tile_BL_corner + 3 * inca] );

							// Group 4 (Tiles 3 & 7)
							svfloat64x2_t zp12 = svld1_f64_x2( p_all,
								&a_[tile_BR_corner + 0 * inca] );
							svfloat64x2_t zp13 = svld1_f64_x2( p_all,
								&a_[tile_BR_corner + 1 * inca] );
							svfloat64x2_t zp14 = svld1_f64_x2( p_all,
								&a_[tile_BR_corner + 2 * inca] );
							svfloat64x2_t zp15 = svld1_f64_x2( p_all,
								&a_[tile_BR_corner + 3 * inca] );

							svfloat64x4_t zq4 = svcreate4( svget2( zp8, 0 ),
								svget2( zp9, 0 ), svget2( zp10, 0 ),
								svget2( zp11, 0 ) );
							svfloat64x4_t zq5 = svcreate4( svget2( zp8, 1 ),
								svget2( zp9, 1 ), svget2( zp10, 1 ),
								svget2( zp11, 1 ) );
							svfloat64x4_t zq6 = svcreate4( svget2( zp12, 0 ),
								svget2( zp13, 0 ), svget2( zp14, 0 ),
								svget2( zp15, 0 ) );
							svfloat64x4_t zq7 = svcreate4( svget2( zp12, 1 ),
								svget2( zp13, 1 ), svget2( zp14, 1 ),
								svget2( zp15, 1 ) );

							svwrite_hor_za64_f64_vg4( 2, trow, zq4 );
							svwrite_hor_za64_f64_vg4( 6, trow, zq5 );
							svwrite_hor_za64_f64_vg4( 3, trow, zq6 );
							svwrite_hor_za64_f64_vg4( 7, trow, zq7 );
						}
					}
					else
					{
						// SAFE PATH: Matrix edge
						for ( uint64_t trow = 0; trow < SVL; trow += 4 )
						{
							const uint64_t tile_UL_corner = (trow)*inca + col;
							const uint64_t tile_UR_corner = tile_UL_corner +
								inca * SVL;
							const uint64_t tile_BL_corner = tile_UL_corner +
								inca * 2 * SVL;
							const uint64_t tile_BR_corner = tile_UL_corner +
								inca * 3 * SVL;

							// 1. Create undefined default vectors
							svfloat64_t undef_v = svundef_f64();
							svfloat64x2_t undef_x2 = svcreate2( undef_v,
								undef_v );

							// 2. Default all load arrays to empty to guarantee
							// safety
							svfloat64x2_t zp0 = undef_x2, zp1 = undef_x2,
										  zp2 = undef_x2, zp3 = undef_x2;
							svfloat64x2_t zp4 = undef_x2, zp5 = undef_x2,
										  zp6 = undef_x2, zp7 = undef_x2;
							svfloat64x2_t zp8 = undef_x2, zp9 = undef_x2,
										  zp10 = undef_x2, zp11 = undef_x2;
							svfloat64x2_t zp12 = undef_x2, zp13 = undef_x2,
										  zp14 = undef_x2, zp15 = undef_x2;

							// 3. Calculate rows left independently for all 4
							// vertical groups
							int64_t rows_left_0 = valid_rows - trow;
							int64_t rows_left_1 = valid_rows - ( SVL + trow );
							int64_t rows_left_2 = valid_rows -
								( 2 * SVL + trow );
							int64_t rows_left_3 = valid_rows -
								( 3 * SVL + trow );

							// 4. Load Group 1 (writes to tiles 0 and 4)
							if ( rows_left_0 > 0 )
								zp0 = svcreate2(
									svld1_f64( pc0,
										&a_[tile_UL_corner + 0 * inca] ),
									svld1_f64( pc1,
										&a_[tile_UL_corner + 0 * inca +
											SVL] ) );
							if ( rows_left_0 > 1 )
								zp1 = svcreate2(
									svld1_f64( pc0,
										&a_[tile_UL_corner + 1 * inca] ),
									svld1_f64( pc1,
										&a_[tile_UL_corner + 1 * inca +
											SVL] ) );
							if ( rows_left_0 > 2 )
								zp2 = svcreate2(
									svld1_f64( pc0,
										&a_[tile_UL_corner + 2 * inca] ),
									svld1_f64( pc1,
										&a_[tile_UL_corner + 2 * inca +
											SVL] ) );
							if ( rows_left_0 > 3 )
								zp3 = svcreate2(
									svld1_f64( pc0,
										&a_[tile_UL_corner + 3 * inca] ),
									svld1_f64( pc1,
										&a_[tile_UL_corner + 3 * inca +
											SVL] ) );

							// 5. Load Group 2 (writes to tiles 1 and 5)
							if ( rows_left_1 > 0 )
								zp4 = svcreate2(
									svld1_f64( pc0,
										&a_[tile_UR_corner + 0 * inca] ),
									svld1_f64( pc1,
										&a_[tile_UR_corner + 0 * inca +
											SVL] ) );
							if ( rows_left_1 > 1 )
								zp5 = svcreate2(
									svld1_f64( pc0,
										&a_[tile_UR_corner + 1 * inca] ),
									svld1_f64( pc1,
										&a_[tile_UR_corner + 1 * inca +
											SVL] ) );
							if ( rows_left_1 > 2 )
								zp6 = svcreate2(
									svld1_f64( pc0,
										&a_[tile_UR_corner + 2 * inca] ),
									svld1_f64( pc1,
										&a_[tile_UR_corner + 2 * inca +
											SVL] ) );
							if ( rows_left_1 > 3 )
								zp7 = svcreate2(
									svld1_f64( pc0,
										&a_[tile_UR_corner + 3 * inca] ),
									svld1_f64( pc1,
										&a_[tile_UR_corner + 3 * inca +
											SVL] ) );

							// 6. Load Group 3 (writes to tiles 2 and 6)
							if ( rows_left_2 > 0 )
								zp8 = svcreate2(
									svld1_f64( pc0,
										&a_[tile_BL_corner + 0 * inca] ),
									svld1_f64( pc1,
										&a_[tile_BL_corner + 0 * inca +
											SVL] ) );
							if ( rows_left_2 > 1 )
								zp9 = svcreate2(
									svld1_f64( pc0,
										&a_[tile_BL_corner + 1 * inca] ),
									svld1_f64( pc1,
										&a_[tile_BL_corner + 1 * inca +
											SVL] ) );
							if ( rows_left_2 > 2 )
								zp10 = svcreate2(
									svld1_f64( pc0,
										&a_[tile_BL_corner + 2 * inca] ),
									svld1_f64( pc1,
										&a_[tile_BL_corner + 2 * inca +
											SVL] ) );
							if ( rows_left_2 > 3 )
								zp11 = svcreate2(
									svld1_f64( pc0,
										&a_[tile_BL_corner + 3 * inca] ),
									svld1_f64( pc1,
										&a_[tile_BL_corner + 3 * inca +
											SVL] ) );

							// 7. Load Group 4 (writes to tiles 3 and 7)
							if ( rows_left_3 > 0 )
								zp12 = svcreate2(
									svld1_f64( pc0,
										&a_[tile_BR_corner + 0 * inca] ),
									svld1_f64( pc1,
										&a_[tile_BR_corner + 0 * inca +
											SVL] ) );
							if ( rows_left_3 > 1 )
								zp13 = svcreate2(
									svld1_f64( pc0,
										&a_[tile_BR_corner + 1 * inca] ),
									svld1_f64( pc1,
										&a_[tile_BR_corner + 1 * inca +
											SVL] ) );
							if ( rows_left_3 > 2 )
								zp14 = svcreate2(
									svld1_f64( pc0,
										&a_[tile_BR_corner + 2 * inca] ),
									svld1_f64( pc1,
										&a_[tile_BR_corner + 2 * inca +
											SVL] ) );
							if ( rows_left_3 > 3 )
								zp15 = svcreate2(
									svld1_f64( pc0,
										&a_[tile_BR_corner + 3 * inca] ),
									svld1_f64( pc1,
										&a_[tile_BR_corner + 3 * inca +
											SVL] ) );

							// 8. Shuffle into x4 tuples
							svfloat64x4_t zq0 = svcreate4( svget2( zp0, 0 ),
								svget2( zp1, 0 ), svget2( zp2, 0 ),
								svget2( zp3, 0 ) );
							svfloat64x4_t zq1 = svcreate4( svget2( zp0, 1 ),
								svget2( zp1, 1 ), svget2( zp2, 1 ),
								svget2( zp3, 1 ) );
							svfloat64x4_t zq2 = svcreate4( svget2( zp4, 0 ),
								svget2( zp5, 0 ), svget2( zp6, 0 ),
								svget2( zp7, 0 ) );
							svfloat64x4_t zq3 = svcreate4( svget2( zp4, 1 ),
								svget2( zp5, 1 ), svget2( zp6, 1 ),
								svget2( zp7, 1 ) );

							// 9. Write into ZA
							svwrite_hor_za64_f64_vg4( 0, trow, zq0 );
							svwrite_hor_za64_f64_vg4( 4, trow, zq1 );
							svwrite_hor_za64_f64_vg4( 1, trow, zq2 );
							svwrite_hor_za64_f64_vg4( 5, trow, zq3 );

							svfloat64x4_t zq4 = svcreate4( svget2( zp8, 0 ),
								svget2( zp9, 0 ), svget2( zp10, 0 ),
								svget2( zp11, 0 ) );
							svfloat64x4_t zq5 = svcreate4( svget2( zp8, 1 ),
								svget2( zp9, 1 ), svget2( zp10, 1 ),
								svget2( zp11, 1 ) );
							svfloat64x4_t zq6 = svcreate4( svget2( zp12, 0 ),
								svget2( zp13, 0 ), svget2( zp14, 0 ),
								svget2( zp15, 0 ) );
							svfloat64x4_t zq7 = svcreate4( svget2( zp12, 1 ),
								svget2( zp13, 1 ), svget2( zp14, 1 ),
								svget2( zp15, 1 ) );

							svwrite_hor_za64_f64_vg4( 2, trow, zq4 );
							svwrite_hor_za64_f64_vg4( 6, trow, zq5 );
							svwrite_hor_za64_f64_vg4( 3, trow, zq6 );
							svwrite_hor_za64_f64_vg4( 7, trow, zq7 );
						}
					}
					// Check if we are at the edge where fewer than
					// 2 * SVL columns remain
					if ( col + ( 2 * SVL ) > n )
					{
						int total_rem = n - col;

						// --- GROUP 1: Tiles 0, 1, 2, 3 ---
						int rem_g1 = ( total_rem > (int)SVL ) ? (int)SVL :
																total_rem;

						PROCESS_GROUP( 0, 1, 2, 3, rem_g1, &p_[0] );

						// --- GROUP 2: Tiles 4, 5, 6, 7 ---
						if ( total_rem > (int)SVL )
						{
							int rem_g2 = total_rem - (int)SVL;
							PROCESS_GROUP( 4, 5, 6, 7, rem_g2,
								&p_[4 * SVL * SVL] );
						}
					}
					else
					{
						// Read - as - columns and store
						for ( uint64_t tcol = 0; tcol < SVL; tcol += 4 )
						{
							svcount_t p0 = svptrue_c32();

							// Each svread_ver reads 4 columns of the tile(SVL).
							svfloat64x4_t zq0 = svread_ver_za64_f64_vg4(
								/* tile: */ 0, /* slice: */ tcol );
							svfloat64x4_t zq2 = svread_ver_za64_f64_vg4(
								/* tile: */ 2, /* slice: */ tcol );

							svfloat64x4_t zq1 = svread_ver_za64_f64_vg4(
								/* tile: */ 1, /* slice: */ tcol );
							svfloat64x4_t zq3 = svread_ver_za64_f64_vg4(
								/* tile: */ 3, /* slice: */ tcol );

							svfloat64x4_t zq0_ = svcreate4( svget4( zq0, 0 ),
								svget4( zq1, 0 ), svget4( zq2, 0 ),
								svget4( zq3, 0 ) );

							svfloat64x4_t zq1_ = svcreate4( svget4( zq0, 1 ),
								svget4( zq1, 1 ), svget4( zq2, 1 ),
								svget4( zq3, 1 ) );

							svfloat64x4_t zq2_ = svcreate4( svget4( zq0, 2 ),
								svget4( zq1, 2 ), svget4( zq2, 2 ),
								svget4( zq3, 2 ) );

							svfloat64x4_t zq3_ = svcreate4( svget4( zq0, 3 ),
								svget4( zq1, 3 ), svget4( zq2, 3 ),
								svget4( zq3, 3 ) );

							svst1( p0, &p_[0], zq0_ );
							svst1( p0, &p_[4 * SVL], zq1_ );
							svst1( p0, &p_[8 * SVL], zq2_ );
							svst1( p0, &p_[12 * SVL], zq3_ );

							// Each svread_ver reads 4 columns of the tile(SVL).
							svfloat64x4_t zq4 = svread_ver_za64_f64_vg4(
								/* tile: */ 4, /* slice: */ tcol );
							svfloat64x4_t zq5 = svread_ver_za64_f64_vg4(
								/* tile: */ 5, /* slice: */ tcol );

							svfloat64x4_t zq6 = svread_ver_za64_f64_vg4(
								/* tile: */ 6, /* slice: */ tcol );
							svfloat64x4_t zq7 = svread_ver_za64_f64_vg4(
								/* tile: */ 7, /* slice: */ tcol );

							svfloat64x4_t zq4_ = svcreate4( svget4( zq4, 0 ),
								svget4( zq5, 0 ), svget4( zq6, 0 ),
								svget4( zq7, 0 ) );

							svfloat64x4_t zq5_ = svcreate4( svget4( zq4, 1 ),
								svget4( zq5, 1 ), svget4( zq6, 1 ),
								svget4( zq7, 1 ) );

							svfloat64x4_t zq6_ = svcreate4( svget4( zq4, 2 ),
								svget4( zq5, 2 ), svget4( zq6, 2 ),
								svget4( zq7, 2 ) );

							svfloat64x4_t zq7_ = svcreate4( svget4( zq4, 3 ),
								svget4( zq5, 3 ), svget4( zq6, 3 ),
								svget4( zq7, 3 ) );

							svst1( p0, &p_[4 * SVL * SVL], zq4_ );
							svst1( p0, &p_[4 * SVL * SVL + 4 * SVL], zq5_ );
							svst1( p0, &p_[4 * SVL * SVL + 8 * SVL], zq6_ );
							svst1( p0, &p_[4 * SVL * SVL + 12 * SVL], zq7_ );

							p_ += ( 2 * SVL * SVL );
						}
						p_ += ( 4 * SVL * SVL );
					}
				}

				p_ = (double*)p;
			}
		}
		else
		{
			bli_dscal2bbs_mxn
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
		bli_dscal2bbs_mxn
			(
			 conja,
			 cdim_,
			 n_,
			 kappa,
			 a, inca, lda,
			 p_, cdim_bcast, ldp
			);
	}

	bli_dset0s_edge
		(
		 cdim_ * cdim_bcast, cdim_max * cdim_bcast,
		 n_, n_max_,
		 p_, ldp
		);
}
