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

// 1. Core Read & Store Logic
#define OP_VG4( TILE, TCOL, P_PTR )                              \
	{                                                            \
		svcount_t p_all = svptrue_c64();                         \
		svfloat64x4_t z = svread_ver_za64_f64_vg4( TILE, TCOL ); \
		svst1( p_all, P_PTR, z );                                \
		P_PTR += ( 4 * SVL );                                    \
	}

#define OP_VG2( TILE, TCOL, P_PTR )                              \
	{                                                            \
		svcount_t p_all = svptrue_c64();                         \
		svfloat64x2_t z = svread_ver_za64_f64_vg2( TILE, TCOL ); \
		svst1( p_all, P_PTR, z );                                \
		P_PTR += ( 2 * SVL );                                    \
	}

#define OP_VG1( TILE, TCOL, P_PTR )                                 \
	{                                                               \
		svbool_t p_true = svptrue_b64();                            \
		svfloat64_t z = svread_ver_za64_m( z, p_true, TILE, TCOL ); \
		svst1( p_true, P_PTR, z );                                  \
		P_PTR += ( 1 * SVL );                                       \
	}

// 2. Duff's Device Logic for a Partial Tile
#define PROCESS_PARTIAL_TILE( TILE_ID, REM, P_BASE )       \
	{                                                      \
		int tcol = 0;                                      \
		double* p_curr = P_BASE;                           \
		int n4 = REM >> 2; /* Number of full VG4 blocks */ \
                                                           \
		if ( n4 > 0 )                                      \
		{                                                  \
			switch ( n4 & 3 )                              \
			{                                              \
			case 0:                                        \
				OP_VG4( TILE_ID, tcol, p_curr );           \
				tcol += 4;                                 \
			case 3:                                        \
				OP_VG4( TILE_ID, tcol, p_curr );           \
				tcol += 4;                                 \
			case 2:                                        \
				OP_VG4( TILE_ID, tcol, p_curr );           \
				tcol += 4;                                 \
			case 1:                                        \
				OP_VG4( TILE_ID, tcol, p_curr );           \
				tcol += 4;                                 \
			}                                              \
		}                                                  \
                                                           \
		/* Handle remaining 1, 2, or 3 columns */          \
		switch ( REM & 3 )                                 \
		{                                                  \
		case 3:                                            \
			OP_VG2( TILE_ID, tcol, p_curr );               \
			tcol += 2;                                     \
			OP_VG1( TILE_ID, tcol, p_curr );               \
			break;                                         \
		case 2:                                            \
			OP_VG2( TILE_ID, tcol, p_curr );               \
			break;                                         \
		case 1:                                            \
			OP_VG1( TILE_ID, tcol, p_curr );               \
			break;                                         \
		default:                                           \
			break;                                         \
		}                                                  \
	}

// 3. Logic for a Full Tile
#define PROCESS_FULL_TILE( TILE_ID, P_BASE )        \
	{                                               \
		double* p_curr = P_BASE;                    \
		for ( int tcol = 0; tcol < SVL; tcol += 4 ) \
		{                                           \
			OP_VG4( TILE_ID, tcol, p_curr );        \
		}                                           \
	}

// PATH 2

// 1. Core Read, Shuffle & Store Logic
#define OP_SHUFFLED_VG4( TCOL, P_PTR )                              \
	{                                                               \
		svcount_t p_all = svptrue_c64();                            \
		svfloat64x4_t zq0 = svread_ver_za64_f64_vg4( 0, TCOL );     \
		svfloat64x4_t zq1 = svread_ver_za64_f64_vg4( 1, TCOL );     \
		svfloat64x4_t zq2 = svread_ver_za64_f64_vg4( 2, TCOL );     \
		svfloat64x4_t zq3 = svread_ver_za64_f64_vg4( 3, TCOL );     \
		svfloat64x4_t zq4 = svread_ver_za64_f64_vg4( 4, TCOL );     \
		svfloat64x4_t zq5 = svread_ver_za64_f64_vg4( 5, TCOL );     \
		svfloat64x4_t zq6 = svread_ver_za64_f64_vg4( 6, TCOL );     \
		svfloat64x4_t zq7 = svread_ver_za64_f64_vg4( 7, TCOL );     \
                                                                    \
		/* Shuffle and Store Column 0 (Offset 0) */                 \
		svfloat64x4_t z_c0_top = svcreate4( svget4( zq0, 0 ),       \
			svget4( zq1, 0 ), svget4( zq2, 0 ), svget4( zq3, 0 ) ); \
		svfloat64x4_t z_c0_bot = svcreate4( svget4( zq4, 0 ),       \
			svget4( zq5, 0 ), svget4( zq6, 0 ), svget4( zq7, 0 ) ); \
		svst1( p_all, P_PTR, z_c0_top );                            \
		svst1( p_all, P_PTR + 4 * SVL, z_c0_bot );                  \
                                                                    \
		/* Shuffle and Store Column 1 (Offset 1*SVL*SVL) */         \
		svfloat64x4_t z_c1_top = svcreate4( svget4( zq0, 1 ),       \
			svget4( zq1, 1 ), svget4( zq2, 1 ), svget4( zq3, 1 ) ); \
		svfloat64x4_t z_c1_bot = svcreate4( svget4( zq4, 1 ),       \
			svget4( zq5, 1 ), svget4( zq6, 1 ), svget4( zq7, 1 ) ); \
		svst1( p_all, P_PTR + 1 * SVL * SVL, z_c1_top );            \
		svst1( p_all, P_PTR + 1 * SVL * SVL + 4 * SVL, z_c1_bot );  \
                                                                    \
		/* Shuffle and Store Column 2 (Offset 2*SVL*SVL) */         \
		svfloat64x4_t z_c2_top = svcreate4( svget4( zq0, 2 ),       \
			svget4( zq1, 2 ), svget4( zq2, 2 ), svget4( zq3, 2 ) ); \
		svfloat64x4_t z_c2_bot = svcreate4( svget4( zq4, 2 ),       \
			svget4( zq5, 2 ), svget4( zq6, 2 ), svget4( zq7, 2 ) ); \
		svst1( p_all, P_PTR + 2 * SVL * SVL, z_c2_top );            \
		svst1( p_all, P_PTR + 2 * SVL * SVL + 4 * SVL, z_c2_bot );  \
                                                                    \
		/* Shuffle and Store Column 3 (Offset 3*SVL*SVL) */         \
		svfloat64x4_t z_c3_top = svcreate4( svget4( zq0, 3 ),       \
			svget4( zq1, 3 ), svget4( zq2, 3 ), svget4( zq3, 3 ) ); \
		svfloat64x4_t z_c3_bot = svcreate4( svget4( zq4, 3 ),       \
			svget4( zq5, 3 ), svget4( zq6, 3 ), svget4( zq7, 3 ) ); \
		svst1( p_all, P_PTR + 3 * SVL * SVL, z_c3_top );            \
		svst1( p_all, P_PTR + 3 * SVL * SVL + 4 * SVL, z_c3_bot );  \
                                                                    \
		P_PTR += ( 4 * SVL * SVL );                                 \
	}

#define OP_SHUFFLED_VG2( TCOL, P_PTR )                              \
	{                                                               \
		svcount_t p_all = svptrue_c64();                            \
		svfloat64x2_t zq0 = svread_ver_za64_f64_vg2( 0, TCOL );     \
		svfloat64x2_t zq1 = svread_ver_za64_f64_vg2( 1, TCOL );     \
		svfloat64x2_t zq2 = svread_ver_za64_f64_vg2( 2, TCOL );     \
		svfloat64x2_t zq3 = svread_ver_za64_f64_vg2( 3, TCOL );     \
		svfloat64x2_t zq4 = svread_ver_za64_f64_vg2( 4, TCOL );     \
		svfloat64x2_t zq5 = svread_ver_za64_f64_vg2( 5, TCOL );     \
		svfloat64x2_t zq6 = svread_ver_za64_f64_vg2( 6, TCOL );     \
		svfloat64x2_t zq7 = svread_ver_za64_f64_vg2( 7, TCOL );     \
                                                                    \
		svfloat64x4_t z_c0_top = svcreate4( svget2( zq0, 0 ),       \
			svget2( zq1, 0 ), svget2( zq2, 0 ), svget2( zq3, 0 ) ); \
		svfloat64x4_t z_c0_bot = svcreate4( svget2( zq4, 0 ),       \
			svget2( zq5, 0 ), svget2( zq6, 0 ), svget2( zq7, 0 ) ); \
		svst1( p_all, P_PTR, z_c0_top );                            \
		svst1( p_all, P_PTR + 4 * SVL, z_c0_bot );                  \
                                                                    \
		svfloat64x4_t z_c1_top = svcreate4( svget2( zq0, 1 ),       \
			svget2( zq1, 1 ), svget2( zq2, 1 ), svget2( zq3, 1 ) ); \
		svfloat64x4_t z_c1_bot = svcreate4( svget2( zq4, 1 ),       \
			svget2( zq5, 1 ), svget2( zq6, 1 ), svget2( zq7, 1 ) ); \
		svst1( p_all, P_PTR + 1 * SVL * SVL, z_c1_top );            \
		svst1( p_all, P_PTR + 1 * SVL * SVL + 4 * SVL, z_c1_bot );  \
                                                                    \
		P_PTR += ( 2 * SVL * SVL );                                 \
	}

#define OP_SHUFFLED_VG1( TCOL, P_PTR )                             \
	{                                                              \
		svbool_t p_true = svptrue_b64();                           \
		svcount_t p_cnt = svptrue_c64();                           \
		svfloat64_t ud = svundef_f64();                            \
		svfloat64_t z0 = svread_ver_za64_m( ud, p_true, 0, TCOL ); \
		svfloat64_t z1 = svread_ver_za64_m( ud, p_true, 1, TCOL ); \
		svfloat64_t z2 = svread_ver_za64_m( ud, p_true, 2, TCOL ); \
		svfloat64_t z3 = svread_ver_za64_m( ud, p_true, 3, TCOL ); \
		svfloat64_t z4 = svread_ver_za64_m( ud, p_true, 4, TCOL ); \
		svfloat64_t z5 = svread_ver_za64_m( ud, p_true, 5, TCOL ); \
		svfloat64_t z6 = svread_ver_za64_m( ud, p_true, 6, TCOL ); \
		svfloat64_t z7 = svread_ver_za64_m( ud, p_true, 7, TCOL ); \
                                                                   \
		svfloat64x4_t z_top = svcreate4( z0, z1, z2, z3 );         \
		svfloat64x4_t z_bot = svcreate4( z4, z5, z6, z7 );         \
		svst1( p_cnt, P_PTR, z_top );                              \
		svst1( p_cnt, P_PTR + 4 * SVL, z_bot );                    \
                                                                   \
		P_PTR += ( 1 * SVL * SVL );                                \
	}

__arm_new( "za" ) __arm_locally_streaming void bli_dpackm_armsme_int_SVLx8SVL
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

	svfloat64x4_t tmp, tmp2;
	svfloat64_t tmp3;

	const double* restrict alpha1 = a;
	double* restrict pi1 = p;

	const bool gs = ( inca != 1 && lda != 1 );
	if ( !gs && cdim_bcast )
	{
		if ( bli_deq1( *( (double*)kappa ) ) )
		{
			if ( inca == 1 && ldp == 8 * SVL )
			// continous memory.packA style
			{
				svbool_t p0 = svwhilelt_b64( (int64_t)0, cdim );
				svbool_t p1 = svwhilelt_b64( (int64_t)( 1 * SVL ), cdim );
				svbool_t p2 = svwhilelt_b64( (int64_t)( 2 * SVL ), cdim );
				svbool_t p3 = svwhilelt_b64( (int64_t)( 3 * SVL ), cdim );
				svbool_t p4 = svwhilelt_b64( (int64_t)( 4 * SVL ), cdim );
				svbool_t p5 = svwhilelt_b64( (int64_t)( 5 * SVL ), cdim );
				svbool_t p6 = svwhilelt_b64( (int64_t)( 6 * SVL ), cdim );
				svbool_t p7 = svwhilelt_b64( (int64_t)( 7 * SVL ), cdim );

				for ( dim_t k = n; k != 0; --k )
				{
					svfloat64_t z0 = svld1_f64( p0, alpha1 + 0 * SVL );
					svfloat64_t z1 = svld1_f64( p1, alpha1 + 1 * SVL );
					svfloat64_t z2 = svld1_f64( p2, alpha1 + 2 * SVL );
					svfloat64_t z3 = svld1_f64( p3, alpha1 + 3 * SVL );
					svfloat64_t z4 = svld1_f64( p4, alpha1 + 4 * SVL );
					svfloat64_t z5 = svld1_f64( p5, alpha1 + 5 * SVL );
					svfloat64_t z6 = svld1_f64( p6, alpha1 + 6 * SVL );
					svfloat64_t z7 = svld1_f64( p7, alpha1 + 7 * SVL );

					tmp = svcreate4( z0, z1, z2, z3 );
					tmp2 = svcreate4( z4, z5, z6, z7 );

					svst1_f64_x4( svptrue_c64(), pi1, tmp );
					svst1_f64_x4( svptrue_c64(), pi1 + 4 * SVL, tmp2 );

					alpha1 += lda;
					pi1 += ldp;
				}
			}
			if ( inca == 1 && ldp == SVL )
			// continous memory.packA style
			{
				svbool_t p0 = svwhilelt_b64( (int64_t)0, cdim );
				for ( dim_t k = n; k != 0; --k )
				{
					tmp3 = svld1_f64( p0, alpha1 );
					svst1_f64( svptrue_b64(), pi1, tmp3 );

					alpha1 += lda;
					pi1 += ldp;
				}
			}
			else if ( inca != 1 && ldp == SVL )
			{
				for ( uint64_t col = 0; col < n; col += 8 * SVL )
				{
					int64_t valid_cols = n - col;

					// Determine total valid rows for this vertical block
					// (max SVL)
					int64_t valid_rows = ( cdim % SVL == 0 ) ? SVL :
															   ( cdim % SVL );

					// Generate the 8 standard SVE column predicates for the
					// safe edge-case loads
					svbool_t pc0 = svwhilelt_b64( (int64_t)( 0 * SVL ),
						valid_cols );
					svbool_t pc1 = svwhilelt_b64( (int64_t)( 1 * SVL ),
						valid_cols );
					svbool_t pc2 = svwhilelt_b64( (int64_t)( 2 * SVL ),
						valid_cols );
					svbool_t pc3 = svwhilelt_b64( (int64_t)( 3 * SVL ),
						valid_cols );
					svbool_t pc4 = svwhilelt_b64( (int64_t)( 4 * SVL ),
						valid_cols );
					svbool_t pc5 = svwhilelt_b64( (int64_t)( 5 * SVL ),
						valid_cols );
					svbool_t pc6 = svwhilelt_b64( (int64_t)( 6 * SVL ),
						valid_cols );
					svbool_t pc7 = svwhilelt_b64( (int64_t)( 7 * SVL ),
						valid_cols );

					svcount_t p_all = svptrue_c64();

					for ( uint64_t trow = 0; trow < SVL; trow += 4 )
					{
						const uint64_t tile_UL_corner = (trow)*inca + col;

						// 1. Create undefined default vectors
						svfloat64_t undef_v = svundef_f64();
						svfloat64x4_t undef_x4 = svcreate4( undef_v, undef_v,
							undef_v, undef_v );

						// 2. Default all load arrays to empty
						svfloat64x4_t zp0 = undef_x4, zp1 = undef_x4,
									  zp2 = undef_x4, zp3 = undef_x4;
						svfloat64x4_t zp4 = undef_x4, zp5 = undef_x4,
									  zp6 = undef_x4, zp7 = undef_x4;

						// 3. Calculate rows left for all tiles
						int64_t rows_left = valid_rows - trow;

						// 4. Loads
						if ( valid_cols >= 8 * SVL )
						{
							// FAST PATH: All 8*SVL columns exist
							if ( rows_left > 0 )
							{
								zp0 = svld1_f64_x4( p_all,
									&a_[tile_UL_corner + 0 * inca] );
								zp4 = svld1_f64_x4( p_all,
									&a_[tile_UL_corner + 0 * inca + 4 * SVL] );
							}
							if ( rows_left > 1 )
							{
								zp1 = svld1_f64_x4( p_all,
									&a_[tile_UL_corner + 1 * inca] );
								zp5 = svld1_f64_x4( p_all,
									&a_[tile_UL_corner + 1 * inca + 4 * SVL] );
							}
							if ( rows_left > 2 )
							{
								zp2 = svld1_f64_x4( p_all,
									&a_[tile_UL_corner + 2 * inca] );
								zp6 = svld1_f64_x4( p_all,
									&a_[tile_UL_corner + 2 * inca + 4 * SVL] );
							}
							if ( rows_left > 3 )
							{
								zp3 = svld1_f64_x4( p_all,
									&a_[tile_UL_corner + 3 * inca] );
								zp7 = svld1_f64_x4( p_all,
									&a_[tile_UL_corner + 3 * inca + 4 * SVL] );
							}
						}
						else
						{
							// SAFE PATH: Matrix edge
							if ( rows_left > 0 )
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
								zp4 = svcreate4( svld1_f64( pc4,
													 &a_[tile_UL_corner +
														 0 * inca + 4 * SVL] ),
									svld1_f64( pc5,
										&a_[tile_UL_corner + 0 * inca +
											5 * SVL] ),
									svld1_f64( pc6,
										&a_[tile_UL_corner + 0 * inca +
											6 * SVL] ),
									svld1_f64( pc7,
										&a_[tile_UL_corner + 0 * inca +
											7 * SVL] ) );
							}
							if ( rows_left > 1 )
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
								zp5 = svcreate4( svld1_f64( pc4,
													 &a_[tile_UL_corner +
														 1 * inca + 4 * SVL] ),
									svld1_f64( pc5,
										&a_[tile_UL_corner + 1 * inca +
											5 * SVL] ),
									svld1_f64( pc6,
										&a_[tile_UL_corner + 1 * inca +
											6 * SVL] ),
									svld1_f64( pc7,
										&a_[tile_UL_corner + 1 * inca +
											7 * SVL] ) );
							}
							if ( rows_left > 2 )
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
								zp6 = svcreate4( svld1_f64( pc4,
													 &a_[tile_UL_corner +
														 2 * inca + 4 * SVL] ),
									svld1_f64( pc5,
										&a_[tile_UL_corner + 2 * inca +
											5 * SVL] ),
									svld1_f64( pc6,
										&a_[tile_UL_corner + 2 * inca +
											6 * SVL] ),
									svld1_f64( pc7,
										&a_[tile_UL_corner + 2 * inca +
											7 * SVL] ) );
							}
							if ( rows_left > 3 )
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
								zp7 = svcreate4( svld1_f64( pc4,
													 &a_[tile_UL_corner +
														 3 * inca + 4 * SVL] ),
									svld1_f64( pc5,
										&a_[tile_UL_corner + 3 * inca +
											5 * SVL] ),
									svld1_f64( pc6,
										&a_[tile_UL_corner + 3 * inca +
											6 * SVL] ),
									svld1_f64( pc7,
										&a_[tile_UL_corner + 3 * inca +
											7 * SVL] ) );
							}
						}

						// 5. Shuffle into x4 tuples
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

						// 6. Write into ZA
						svwrite_hor_za64_f64_vg4( 0, trow, zq0 );
						svwrite_hor_za64_f64_vg4( 1, trow, zq1 );
						svwrite_hor_za64_f64_vg4( 2, trow, zq2 );
						svwrite_hor_za64_f64_vg4( 3, trow, zq3 );
						svwrite_hor_za64_f64_vg4( 4, trow, zq4 );
						svwrite_hor_za64_f64_vg4( 5, trow, zq5 );
						svwrite_hor_za64_f64_vg4( 6, trow, zq6 );
						svwrite_hor_za64_f64_vg4( 7, trow, zq7 );
					}
					// Check if we are at the edge and fewer than
					// 8 * SVL columns remain
					if ( col + ( 8 * SVL ) > n )
					{
						int total_rem = n - col;

						// --- TILE 0 ---
						if ( total_rem >= (int)SVL )
						{
							PROCESS_FULL_TILE( 0, &p_[0] );
							total_rem -= SVL;
						}
						else
						{
							PROCESS_PARTIAL_TILE( 0, total_rem, &p_[0] );
							total_rem = 0;
						}

						// --- TILE 1 ---
						if ( total_rem > 0 )
						{
							if ( total_rem >= (int)SVL )
							{
								PROCESS_FULL_TILE( 1, &p_[SVL * SVL] );
								total_rem -= SVL;
							}
							else
							{
								PROCESS_PARTIAL_TILE( 1, total_rem,
									&p_[SVL * SVL] );
								total_rem = 0;
							}
						}

						// --- TILE 2 ---
						if ( total_rem > 0 )
						{
							if ( total_rem >= (int)SVL )
							{
								PROCESS_FULL_TILE( 2, &p_[2 * SVL * SVL] );
								total_rem -= SVL;
							}
							else
							{
								PROCESS_PARTIAL_TILE( 2, total_rem,
									&p_[2 * SVL * SVL] );
								total_rem = 0;
							}
						}

						// --- TILE 3 ---
						if ( total_rem > 0 )
						{
							if ( total_rem >= (int)SVL )
							{
								PROCESS_FULL_TILE( 3, &p_[3 * SVL * SVL] );
								total_rem -= SVL;
							}
							else
							{
								PROCESS_PARTIAL_TILE( 3, total_rem,
									&p_[3 * SVL * SVL] );
								total_rem = 0;
							}
						}
						// --- TILE 4 ---
						if ( total_rem > 0 )
						{
							if ( total_rem >= (int)SVL )
							{
								PROCESS_FULL_TILE( 4, &p_[4 * SVL * SVL] );
								total_rem -= SVL;
							}
							else
							{
								PROCESS_PARTIAL_TILE( 4, total_rem,
									&p_[4 * SVL * SVL] );
								total_rem = 0;
							}
						}

						// --- TILE 5 ---
						if ( total_rem > 0 )
						{
							if ( total_rem >= (int)SVL )
							{
								PROCESS_FULL_TILE( 5, &p_[5 * SVL * SVL] );
								total_rem -= SVL;
							}
							else
							{
								PROCESS_PARTIAL_TILE( 5, total_rem,
									&p_[5 * SVL * SVL] );
								total_rem = 0;
							}
						}

						// --- TILE 6 ---
						if ( total_rem > 0 )
						{
							if ( total_rem >= (int)SVL )
							{
								PROCESS_FULL_TILE( 6, &p_[6 * SVL * SVL] );
								total_rem -= SVL;
							}
							else
							{
								PROCESS_PARTIAL_TILE( 6, total_rem,
									&p_[6 * SVL * SVL] );
								total_rem = 0;
							}
						}

						// --- TILE 7 ---
						if ( total_rem > 0 )
						{
							PROCESS_PARTIAL_TILE( 7, total_rem,
								&p_[7 * SVL * SVL] );
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

							svst1( p0, &p_[0], zq0 );
							svst1( p0, &p_[SVL * SVL], zq1 );
							svst1( p0, &p_[2 * SVL * SVL], zq2 );
							svst1( p0, &p_[3 * SVL * SVL], zq3 );

							// Each svread_ver reads 4 columns of the tile(SVL).
							svfloat64x4_t zq4 = svread_ver_za64_f64_vg4(
								/* tile: */ 4, /* slice: */ tcol );
							svfloat64x4_t zq5 = svread_ver_za64_f64_vg4(
								/* tile: */ 5, /* slice: */ tcol );

							svfloat64x4_t zq6 = svread_ver_za64_f64_vg4(
								/* tile: */ 6, /* slice: */ tcol );
							svfloat64x4_t zq7 = svread_ver_za64_f64_vg4(
								/* tile: */ 7, /* slice: */ tcol );

							svst1( p0, &p_[4 * SVL * SVL], zq4 );
							svst1( p0, &p_[5 * SVL * SVL], zq5 );
							svst1( p0, &p_[6 * SVL * SVL], zq6 );
							svst1( p0, &p_[7 * SVL * SVL], zq7 );

							p_ += ( 4 * SVL );
						}
						p_ += ( 7 * SVL * SVL );
					}
				}

				p_ = (double*)p;
			}
			else if ( inca != 1 && ldp == 8 * SVL )
			{
				for ( uint64_t col = 0; col < n; col += SVL )
				{
					int64_t valid_cols = n - col;

					// Determine total valid rows for this vertical block
					// (max 8 * SVL)
					int64_t valid_rows = ( cdim % ( 8 * SVL ) == 0 ) ?
						( 8 * SVL ) :
						( cdim % ( 8 * SVL ) );

					// Generate a standard SVE column predicate for the safe
					// edge-case loads
					svbool_t p_col = svwhilelt_b64( (int64_t)0, valid_cols );
					svbool_t p_all = svptrue_b64();

					if ( valid_cols >= SVL && valid_rows >= 8 * SVL )
					{
						// FAST PATH: Perfect 8*SVL x SVL block
						for ( uint64_t trow = 0; trow < SVL; trow += 4 )
						{
							const uint64_t tile_0 = (trow)*inca + col;
							const uint64_t tile_1 = tile_0 + inca * SVL;
							const uint64_t tile_2 = tile_0 + inca * 2 * SVL;
							const uint64_t tile_3 = tile_0 + inca * 3 * SVL;
							const uint64_t tile_4 = tile_0 + inca * 4 * SVL;
							const uint64_t tile_5 = tile_0 + inca * 5 * SVL;
							const uint64_t tile_6 = tile_0 + inca * 6 * SVL;
							const uint64_t tile_7 = tile_0 + inca * 7 * SVL;

							svfloat64x4_t zq0 = svcreate4(
								svld1_f64( p_all, &a_[tile_0 + 0 * inca] ),
								svld1_f64( p_all, &a_[tile_0 + 1 * inca] ),
								svld1_f64( p_all, &a_[tile_0 + 2 * inca] ),
								svld1_f64( p_all, &a_[tile_0 + 3 * inca] ) );

							svfloat64x4_t zq1 = svcreate4(
								svld1_f64( p_all, &a_[tile_1 + 0 * inca] ),
								svld1_f64( p_all, &a_[tile_1 + 1 * inca] ),
								svld1_f64( p_all, &a_[tile_1 + 2 * inca] ),
								svld1_f64( p_all, &a_[tile_1 + 3 * inca] ) );

							svfloat64x4_t zq2 = svcreate4(
								svld1_f64( p_all, &a_[tile_2 + 0 * inca] ),
								svld1_f64( p_all, &a_[tile_2 + 1 * inca] ),
								svld1_f64( p_all, &a_[tile_2 + 2 * inca] ),
								svld1_f64( p_all, &a_[tile_2 + 3 * inca] ) );

							svfloat64x4_t zq3 = svcreate4(
								svld1_f64( p_all, &a_[tile_3 + 0 * inca] ),
								svld1_f64( p_all, &a_[tile_3 + 1 * inca] ),
								svld1_f64( p_all, &a_[tile_3 + 2 * inca] ),
								svld1_f64( p_all, &a_[tile_3 + 3 * inca] ) );

							svwrite_hor_za64_f64_vg4( 0, trow, zq0 );
							svwrite_hor_za64_f64_vg4( 1, trow, zq1 );
							svwrite_hor_za64_f64_vg4( 2, trow, zq2 );
							svwrite_hor_za64_f64_vg4( 3, trow, zq3 );

							svfloat64x4_t zq4 = svcreate4(
								svld1_f64( p_all, &a_[tile_4 + 0 * inca] ),
								svld1_f64( p_all, &a_[tile_4 + 1 * inca] ),
								svld1_f64( p_all, &a_[tile_4 + 2 * inca] ),
								svld1_f64( p_all, &a_[tile_4 + 3 * inca] ) );

							svfloat64x4_t zq5 = svcreate4(
								svld1_f64( p_all, &a_[tile_5 + 0 * inca] ),
								svld1_f64( p_all, &a_[tile_5 + 1 * inca] ),
								svld1_f64( p_all, &a_[tile_5 + 2 * inca] ),
								svld1_f64( p_all, &a_[tile_5 + 3 * inca] ) );

							svfloat64x4_t zq6 = svcreate4(
								svld1_f64( p_all, &a_[tile_6 + 0 * inca] ),
								svld1_f64( p_all, &a_[tile_6 + 1 * inca] ),
								svld1_f64( p_all, &a_[tile_6 + 2 * inca] ),
								svld1_f64( p_all, &a_[tile_6 + 3 * inca] ) );

							svfloat64x4_t zq7 = svcreate4(
								svld1_f64( p_all, &a_[tile_7 + 0 * inca] ),
								svld1_f64( p_all, &a_[tile_7 + 1 * inca] ),
								svld1_f64( p_all, &a_[tile_7 + 2 * inca] ),
								svld1_f64( p_all, &a_[tile_7 + 3 * inca] ) );

							svwrite_hor_za64_f64_vg4( 4, trow, zq4 );
							svwrite_hor_za64_f64_vg4( 5, trow, zq5 );
							svwrite_hor_za64_f64_vg4( 6, trow, zq6 );
							svwrite_hor_za64_f64_vg4( 7, trow, zq7 );
						}
					}
					else
					{
						// SAFE PATH: Matrix edge
						for ( uint64_t trow = 0; trow < SVL; trow += 4 )
						{
							// 1. Create undefined default vectors
							svfloat64_t undef_v = svundef_f64();
							svfloat64x4_t undef_x4 = svcreate4( undef_v,
								undef_v, undef_v, undef_v );

							// 2. Default all load arrays to empty
							svfloat64x4_t zq0 = undef_x4, zq1 = undef_x4,
										  zq2 = undef_x4, zq3 = undef_x4;
							svfloat64x4_t zq4 = undef_x4, zq5 = undef_x4,
										  zq6 = undef_x4, zq7 = undef_x4;

							const uint64_t tile_0 = (trow)*inca + col;
							const uint64_t tile_1 = tile_0 + inca * SVL;
							const uint64_t tile_2 = tile_0 + inca * 2 * SVL;
							const uint64_t tile_3 = tile_0 + inca * 3 * SVL;
							const uint64_t tile_4 = tile_0 + inca * 4 * SVL;
							const uint64_t tile_5 = tile_0 + inca * 5 * SVL;
							const uint64_t tile_6 = tile_0 + inca * 6 * SVL;
							const uint64_t tile_7 = tile_0 + inca * 7 * SVL;

							// 3. Calculate rows left independently for each
							// tile
							int64_t rows_left_t0 = valid_rows -
								( 0 * SVL + trow );
							int64_t rows_left_t1 = valid_rows -
								( 1 * SVL + trow );
							int64_t rows_left_t2 = valid_rows -
								( 2 * SVL + trow );
							int64_t rows_left_t3 = valid_rows -
								( 3 * SVL + trow );
							int64_t rows_left_t4 = valid_rows -
								( 4 * SVL + trow );
							int64_t rows_left_t5 = valid_rows -
								( 5 * SVL + trow );
							int64_t rows_left_t6 = valid_rows -
								( 6 * SVL + trow );
							int64_t rows_left_t7 = valid_rows -
								( 7 * SVL + trow );

							// 4. Loads for each tile
							if ( rows_left_t0 > 0 )
							{
								zq0 = svcreate4( ( rows_left_t0 > 0 ) ?
										svld1_f64( p_col,
											&a_[tile_0 + 0 * inca] ) :
										undef_v,
									( rows_left_t0 > 1 ) ?
										svld1_f64( p_col,
											&a_[tile_0 + 1 * inca] ) :
										undef_v,
									( rows_left_t0 > 2 ) ?
										svld1_f64( p_col,
											&a_[tile_0 + 2 * inca] ) :
										undef_v,
									( rows_left_t0 > 3 ) ?
										svld1_f64( p_col,
											&a_[tile_0 + 3 * inca] ) :
										undef_v );
							}

							if ( rows_left_t1 > 0 )
							{
								zq1 = svcreate4( ( rows_left_t1 > 0 ) ?
										svld1_f64( p_col,
											&a_[tile_1 + 0 * inca] ) :
										undef_v,
									( rows_left_t1 > 1 ) ?
										svld1_f64( p_col,
											&a_[tile_1 + 1 * inca] ) :
										undef_v,
									( rows_left_t1 > 2 ) ?
										svld1_f64( p_col,
											&a_[tile_1 + 2 * inca] ) :
										undef_v,
									( rows_left_t1 > 3 ) ?
										svld1_f64( p_col,
											&a_[tile_1 + 3 * inca] ) :
										undef_v );
							}

							if ( rows_left_t2 > 0 )
							{
								zq2 = svcreate4( ( rows_left_t2 > 0 ) ?
										svld1_f64( p_col,
											&a_[tile_2 + 0 * inca] ) :
										undef_v,
									( rows_left_t2 > 1 ) ?
										svld1_f64( p_col,
											&a_[tile_2 + 1 * inca] ) :
										undef_v,
									( rows_left_t2 > 2 ) ?
										svld1_f64( p_col,
											&a_[tile_2 + 2 * inca] ) :
										undef_v,
									( rows_left_t2 > 3 ) ?
										svld1_f64( p_col,
											&a_[tile_2 + 3 * inca] ) :
										undef_v );
							}

							if ( rows_left_t3 > 0 )
							{
								zq3 = svcreate4( ( rows_left_t3 > 0 ) ?
										svld1_f64( p_col,
											&a_[tile_3 + 0 * inca] ) :
										undef_v,
									( rows_left_t3 > 1 ) ?
										svld1_f64( p_col,
											&a_[tile_3 + 1 * inca] ) :
										undef_v,
									( rows_left_t3 > 2 ) ?
										svld1_f64( p_col,
											&a_[tile_3 + 2 * inca] ) :
										undef_v,
									( rows_left_t3 > 3 ) ?
										svld1_f64( p_col,
											&a_[tile_3 + 3 * inca] ) :
										undef_v );
							}

							if ( rows_left_t4 > 0 )
							{
								zq4 = svcreate4( ( rows_left_t4 > 0 ) ?
										svld1_f64( p_col,
											&a_[tile_4 + 0 * inca] ) :
										undef_v,
									( rows_left_t4 > 1 ) ?
										svld1_f64( p_col,
											&a_[tile_4 + 1 * inca] ) :
										undef_v,
									( rows_left_t4 > 2 ) ?
										svld1_f64( p_col,
											&a_[tile_4 + 2 * inca] ) :
										undef_v,
									( rows_left_t4 > 3 ) ?
										svld1_f64( p_col,
											&a_[tile_4 + 3 * inca] ) :
										undef_v );
							}

							if ( rows_left_t5 > 0 )
							{
								zq5 = svcreate4( ( rows_left_t5 > 0 ) ?
										svld1_f64( p_col,
											&a_[tile_5 + 0 * inca] ) :
										undef_v,
									( rows_left_t5 > 1 ) ?
										svld1_f64( p_col,
											&a_[tile_5 + 1 * inca] ) :
										undef_v,
									( rows_left_t5 > 2 ) ?
										svld1_f64( p_col,
											&a_[tile_5 + 2 * inca] ) :
										undef_v,
									( rows_left_t5 > 3 ) ?
										svld1_f64( p_col,
											&a_[tile_5 + 3 * inca] ) :
										undef_v );
							}

							if ( rows_left_t6 > 0 )
							{
								zq6 = svcreate4( ( rows_left_t6 > 0 ) ?
										svld1_f64( p_col,
											&a_[tile_6 + 0 * inca] ) :
										undef_v,
									( rows_left_t6 > 1 ) ?
										svld1_f64( p_col,
											&a_[tile_6 + 1 * inca] ) :
										undef_v,
									( rows_left_t6 > 2 ) ?
										svld1_f64( p_col,
											&a_[tile_6 + 2 * inca] ) :
										undef_v,
									( rows_left_t6 > 3 ) ?
										svld1_f64( p_col,
											&a_[tile_6 + 3 * inca] ) :
										undef_v );
							}

							if ( rows_left_t7 > 0 )
							{
								zq7 = svcreate4( ( rows_left_t7 > 0 ) ?
										svld1_f64( p_col,
											&a_[tile_7 + 0 * inca] ) :
										undef_v,
									( rows_left_t7 > 1 ) ?
										svld1_f64( p_col,
											&a_[tile_7 + 1 * inca] ) :
										undef_v,
									( rows_left_t7 > 2 ) ?
										svld1_f64( p_col,
											&a_[tile_7 + 2 * inca] ) :
										undef_v,
									( rows_left_t7 > 3 ) ?
										svld1_f64( p_col,
											&a_[tile_7 + 3 * inca] ) :
										undef_v );
							}

							// 5. Write into ZA
							svwrite_hor_za64_f64_vg4( 0, trow, zq0 );
							svwrite_hor_za64_f64_vg4( 1, trow, zq1 );
							svwrite_hor_za64_f64_vg4( 2, trow, zq2 );
							svwrite_hor_za64_f64_vg4( 3, trow, zq3 );
							svwrite_hor_za64_f64_vg4( 4, trow, zq4 );
							svwrite_hor_za64_f64_vg4( 5, trow, zq5 );
							svwrite_hor_za64_f64_vg4( 6, trow, zq6 );
							svwrite_hor_za64_f64_vg4( 7, trow, zq7 );
						}
					}

					// Check if we are at the edge where fewer than
					// SVL columns remain
					if ( col + SVL > n )
					{
						int rem = n - col;
						int tcol = 0;

						// 1. Process as many full VG4 blocks as possible
						while ( rem >= 4 )
						{
							OP_SHUFFLED_VG4( tcol, p_ );
							tcol += 4;
							rem -= 4;
						}

						// 2. Process a VG2 block if remaining
						if ( rem >= 2 )
						{
							OP_SHUFFLED_VG2( tcol, p_ );
							tcol += 2;
							rem -= 2;
						}

						// 3. Process the last column if remaining
						if ( rem >= 1 )
						{
							OP_SHUFFLED_VG1( tcol, p_ );
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
							svst1( p0, &p_[SVL * SVL], zq1_ );
							svst1( p0, &p_[2 * SVL * SVL], zq2_ );
							svst1( p0, &p_[3 * SVL * SVL], zq3_ );

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

							svst1( p0, &p_[4 * SVL], zq4_ );
							svst1( p0, &p_[SVL * SVL + 4 * SVL], zq5_ );
							svst1( p0, &p_[2 * SVL * SVL + 4 * SVL], zq6_ );
							svst1( p0, &p_[3 * SVL * SVL + 4 * SVL], zq7_ );

							p_ += ( 4 * SVL * SVL );
						}
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
