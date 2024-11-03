/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2014, The University of Texas at Austin
   Copyright (C) 2018 - 2019, Advanced Micro Devices, Inc.

   Redistribution and use in source and binary forms, with or without
   modification, are permitted provided that the following conditions are
   met:
    - Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.
    - Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.
    - Neither the name of The University of Texas nor the names of its
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
#include "bli_unit_testing.h"

#include <exception>
#include <vector>
#include <string>
#include <ostream>

int main()
{
	get_unit_test_registrar().run_tests();
}

/******************************************************************************
 *
 * absq2s
 *
 *****************************************************************************/

// tabsq2s unit test
#undef GENTFUNC
#define GENTFUNC( opname, ctypex, chx, ctypey, chy, ctypec, chc ) \
UNIT_TEST(chx,chy,chc,opname) \
( \
	for ( auto x : test_values<ctypex>() ) \
	{ \
		auto y0 = convert<ctypey>( norm( convert_prec<ctypec>( x ) ) ); \
\
		ctypey y; \
		bli_tabsq2s( chx,chy,chc, x, y ); \
\
		INFO( "x:        " << x ); \
		INFO( "y (C++):  " << y0 ); \
		INFO( "y (BLIS): " << y ); \
\
		check<ctypec>( y, y0 ); \
	} \
)

INSERT_GENTFUNC_MIX3( RC, RC, C, absq2s )

// tabsq2ris unit test
#undef GENTFUNC
#define GENTFUNC( opname, ctypex, chx, ctypey, chy, ctypec, chc ) \
UNIT_TEST(chx,chy,chc,opname) \
( \
	for ( auto x : test_values<ctypex>() ) \
	{ \
		auto y0 = convert<ctypey>( norm( convert_prec<ctypec>( x ) ) ); \
\
		ctypey y; \
		bli_tabsq2ris( chx,chy,chc, \
		               real( x ), imag( x ), \
		               real( y ), imag( y ) ); \
\
		INFO( "x:        " << x ); \
		INFO( "y (C++):  " << y0 ); \
		INFO( "y (BLIS): " << y ); \
\
		check<ctypec>( y, y0 ); \
	} \
)

INSERT_GENTFUNC_MIX3( RC, RC, C, absq2ris )

/******************************************************************************
 *
 * abval2s
 *
 *****************************************************************************/

// tabval2s unit test
#undef GENTFUNC
#define GENTFUNC( opname, ctypex, chx, ctypey, chy, ctypec, chc ) \
UNIT_TEST(chx,chy,chc,opname) \
( \
	for ( auto x : test_values<ctypex>() ) \
	{ \
		auto y0 = convert<ctypey>( absolute( convert_prec<ctypec>( x ) ) ); \
\
		ctypey y; \
		bli_tabval2s( chx,chy,chc, x, y ); \
\
		INFO( "x:        " << x ); \
		INFO( "y (C++):  " << y0 ); \
		INFO( "y (BLIS): " << y ); \
\
		check<ctypec>( y, y0 ); \
	} \
)

INSERT_GENTFUNC_MIX3( RC, RC, C, abval2s )

// tabval2ris unit test
#undef GENTFUNC
#define GENTFUNC( opname, ctypex, chx, ctypey, chy, ctypec, chc ) \
UNIT_TEST(chx,chy,chc,opname) \
( \
	for ( auto x : test_values<ctypex>() ) \
	{ \
		auto y0 = convert<ctypey>( absolute( convert_prec<ctypec>( x ) ) ); \
\
		ctypey y; \
		bli_tabval2ris( chx,chy,chc, \
		                real( x ), imag( x ), \
		                real( y ), imag( y ) ); \
\
		INFO( "x:        " << x ); \
		INFO( "y (C++):  " << y0 ); \
		INFO( "y (BLIS): " << y ); \
\
		check<ctypec>( y, y0 ); \
	} \
)

INSERT_GENTFUNC_MIX3( RC, RC, C, abval2ris )

#undef GENTFUNC
#define GENTFUNC(ctypex, chx, ctypey, chy, ctypez, chz, ctypec, chc, opname ) \
UNIT_TEST(chx,chy,chz,chc,opname) \
( \
	for ( auto x : test_values<ctypex>() ) \
	for ( auto y : test_values<ctypey>() ) \
	{ \
		auto z0 = convert<ctypez>( convert_prec<ctypec>( x ) + \
		                           convert_prec<ctypec>( y ) ); \
\
		INFO( "x: " << x ); \
		INFO( "y: " << y ); \
\
		ctypez z; \
		bli_tadd3s( chx,chy,chz,chc, x, y, z ); \
\
		INFO( "z (C++):  " << z0 ); \
		INFO( "z (BLIS): " << z ); \
\
		check<ctypec>( z, z0 ); \
	} \
)

/******************************************************************************
 *
 * add3s
 *
 *****************************************************************************/

INSERT_GENTFUNC_MIX4(RC, RC, RC, C, add3s);

#undef GENTFUNC
#define GENTFUNC(ctypex, chx, ctypey, chy, ctypez, chz, ctypec, chc, opname ) \
UNIT_TEST(chx,chy,chz,chc,opname) \
( \
	for ( auto x : test_values<ctypex>() ) \
	for ( auto y : test_values<ctypey>() ) \
	{ \
		auto z0 = convert<ctypez>( conj( convert_prec<ctypec>( x ) ) + \
		                                 convert_prec<ctypec>( y ) ); \
\
		INFO( "x: " << x ); \
		INFO( "y: " << y ); \
\
		ctypez z; \
		bli_tadd3js( chx,chy,chz,chc, x, y, z ); \
\
		INFO( "z (C++):  " << z0 ); \
		INFO( "z (BLIS): " << z ); \
\
		check<ctypec>( z, z0 ); \
	} \
)

INSERT_GENTFUNC_MIX4(RC, RC, RC, C, add3js);

// tadd3ris unit test
#undef GENTFUNC
#define GENTFUNC( opname, ctypex, chx, ctypey, chy, ctypec, chc ) \
UNIT_TEST(chx,chy,chc,opname) \
( \
	for ( auto x : test_values<ctypex>() ) \
	for ( auto y : test_values<ctypey>() ) \
	{ \
		auto z0 = convert<ctypez>( convert_prec<ctypec>( x ) + \
		                           convert_prec<ctypec>( y ) ); \
\
		INFO( "x: " << x ); \
		INFO( "y: " << y ); \
\
		ctypez z; \
		bli_tadd3ris( chx,chy,chz,chc, \
		              real( x ), imag( x ), \
		              real( y ), imag( y ), \
		              real( z ), imag( z ) ); \
\
		INFO( "z (C++):  " << z0 ); \
		INFO( "z (BLIS): " << z ); \
\
		check<ctypec>( z, z0 ); \
	} \
)

INSERT_GENTFUNC_MIX4(RC, RC, RC, C, add3ris);

// tadd3jris unit test
#undef GENTFUNC
#define GENTFUNC( opname, ctypex, chx, ctypey, chy, ctypec, chc ) \
UNIT_TEST(chx,chy,chc,opname) \
( \
	for ( auto x : test_values<ctypex>() ) \
	for ( auto y : test_values<ctypey>() ) \
	{ \
		auto z0 = convert<ctypez>( conj( convert_prec<ctypec>( x ) ) + \
		                                 convert_prec<ctypec>( y ) ); \
\
		INFO( "x: " << x ); \
		INFO( "y: " << y ); \
\
		ctypez z; \
		bli_tadd3jris( chx,chy,chz,chc, \
		               real( x ), imag( x ), \
		               real( y ), imag( y ), \
		               real( z ), imag( z ) ); \
\
		INFO( "z (C++):  " << z0 ); \
		INFO( "z (BLIS): " << z ); \
\
		check<ctypec>( z, z0 ); \
	} \
)

INSERT_GENTFUNC_MIX4(RC, RC, RC, C, add3jris);

/******************************************************************************
 *
 * adds
 *
 *****************************************************************************/

// tadds unit test
#undef GENTFUNC
#define GENTFUNC( opname, ctypex, chx, ctypey, chy, ctypec, chc ) \
UNIT_TEST(chx,chy,chc,opname) \
( \
	for ( auto x : test_values<ctypex>() ) \
	for ( auto y : test_values<ctypey>() ) \
	{ \
		auto y0 = convert<ctypey>( convert_prec<ctypec>( x ) + \
		                           convert_prec<ctypec>( y ) ); \
\
		INFO( "x:        " << x ); \
		INFO( "y (orig): " << y ); \
\
		bli_tadds( chx,chy,chc, x, y ); \
\
		INFO( "y (C++):  " << y0 ); \
		INFO( "y (BLIS): " << y ); \
\
		check<ctypec>( y, y0 ); \
	} \
)

INSERT_GENTFUNC_MIX(RC, RC, C, adds);

// taddjs unit test
#undef GENTFUNC
#define GENTFUNC( opname, ctypex, chx, ctypey, chy, ctypec, chc ) \
UNIT_TEST(chx,chy,chc,opname) \
( \
	for ( auto x : test_values<ctypex>() ) \
	for ( auto y : test_values<ctypey>() ) \
	{ \
		auto y0 = convert<ctypey>( conj( convert_prec<ctypec>( x ) ) + \
		                                 convert_prec<ctypec>( y ) ); \
\
		INFO( "x:        " << x ); \
		INFO( "y (orig): " << y ); \
\
		bli_taddjs( chx,chy,chc, x, y ); \
\
		INFO( "y (C++):  " << y0 ); \
		INFO( "y (BLIS): " << y ); \
\
		check<ctypec>( y, y0 ); \
	} \
)

INSERT_GENTFUNC_MIX3( RC, RC, C, addjs )

// taddris unit test
#undef GENTFUNC
#define GENTFUNC( opname, ctypex, chx, ctypey, chy, ctypec, chc ) \
UNIT_TEST(chx,chy,chc,opname) \
( \
	for ( auto x : test_values<ctypex>() ) \
	for ( auto y : test_values<ctypey>() ) \
	{ \
		auto y0 = convert<ctypey>( convert_prec<ctypec>( x ) + \
		                           convert_prec<ctypec>( y ) ); \
\
		INFO( "x:        " << x ); \
		INFO( "y (orig): " << y ); \
\
		bli_taddris( chx,chy,chc, \
		             real( x ), imag( x ), \
		             real( y ), imag( y ) ); \
\
		INFO( "y (C++):  " << y0 ); \
		INFO( "y (BLIS): " << y ); \
\
		check<ctypec>( y, y0 ); \
	} \
)

INSERT_GENTFUNC_MIX3( RC, RC, C, addris )

// taddjris unit test
#undef GENTFUNC
#define GENTFUNC( opname, ctypex, chx, ctypey, chy, ctypec, chc ) \
UNIT_TEST(chx,chy,chc,opname) \
( \
	for ( auto x : test_values<ctypex>() ) \
	for ( auto y : test_values<ctypey>() ) \
	{ \
		auto y0 = convert<ctypey>( conj( convert_prec<ctypec>( x ) ) + \
		                                 convert_prec<ctypec>( y ) ); \
\
		INFO( "x:        " << x ); \
		INFO( "y (orig): " << y ); \
\
		bli_taddjris( chx,chy,chc, \
		              real( x ), imag( x ), \
		              real( y ), imag( y ) ); \
\
		INFO( "y (C++):  " << y0 ); \
		INFO( "y (BLIS): " << y ); \
\
		check<ctypec>( y, y0 ); \
	} \
)

INSERT_GENTFUNC_MIX3( RC, RC, C, addjris )

// tadds_mxn unit test
#undef GENTFUNC
#define GENTFUNC( opname, ctypex, chx, ctypey, chy, ctypec, chc ) \
UNIT_TEST(chx,chy,chc,opname) \
( \
	constexpr auto M = 4; \
	constexpr auto N = 4; \
\
	for ( auto x : test_values<ctypex>() ) \
	for ( auto y : test_values<ctypey>() ) \
	{ \
		auto xmn = tile<M,N>( x ); \
		auto ymn = tile<M,N>( y ); \
\
		INFO( "row-major" ); \
\
		auto ymn0 = ymn; \
		axpbys_mxn<ctypec,BLIS_NO_TRANSPOSE>( 1.0, xmn, 1.0, ymn0, dense ); \
\
		INFO( "x:\n" << xmn ); \
		INFO( "y (init):\n" << ymn ); \
\
		bli_tadds_mxn( chx,chy,chc, M, N, &xmn[0][0], N, 1, &ymn[0][0], N, 1 ); \
\
		INFO( "y (C++):\n" << ymn0 ); \
		INFO( "y (BLIS):\n" << ymn ); \
\
		check<ctypec>( ymn, ymn0 ); \
	} \
\
	for ( auto x : test_values<ctypex>() ) \
	for ( auto y : test_values<ctypey>() ) \
	{ \
		auto xmn = tile<M,N>( x ); \
		auto ymn = tile<M,N>( y ); \
\
		INFO( "column-major" ); \
\
		auto ymn0 = ymn; \
		axpbys_mxn<ctypec,BLIS_TRANSPOSE>( 1.0, xmn, 1.0, ymn0, dense ); \
\
		INFO( "x:\n" << xmn ); \
		INFO( "y (init):\n" << ymn ); \
\
		bli_tadds_mxn( chx,chy,chc, N, M, &xmn[0][0], 1, N, &ymn[0][0], 1, N ); \
\
		INFO( "y (C++):\n" << ymn0 ); \
		INFO( "y (BLIS):\n" << ymn ); \
\
		check<ctypec>( ymn, ymn0 ); \
	} \
)

INSERT_GENTFUNC_MIX3( RC, RC, C, adds_mxn )

/******************************************************************************
 *
 * axpbys
 *
 *****************************************************************************/

#undef GENTFUNC
#define GENTFUNC( opname, ctypea, cha, ctypex, chx, ctypeb, chb, ctypey, chy, ctypec, chc ) \
UNIT_TEST(cha,chx,chb,chy,chc,opname) \
( \
	for ( auto a : test_values<ctypea>() ) \
	for ( auto x : test_values<ctypex>() ) \
	for ( auto b : test_values<ctypeb>() ) \
	for ( auto y : test_values<ctypey>() ) \
	{ \
		auto y0 = convert<ctypey>( convert_prec<ctypec>( a ) * \
		                           convert_prec<ctypec>( x ) + \
		                           convert_prec<ctypec>( b ) * \
		                           convert_prec<ctypec>( y ) ); \
\
		INFO( "a:        " << a ); \
		INFO( "x:        " << x ); \
		INFO( "b:        " << b ); \
		INFO( "y (init): " << y ); \
\
		bli_taxpbys( cha,chx,chb,chy,chc, a, x, b, y ); \
\
		INFO( "y (C++):  " << y0 ); \
		INFO( "y (BLIS): " << y ); \
\
		check<ctypec>( y, y0 ); \
	} \
)

INSERT_GENTFUNC_MIX5( RC, RC, RC, RC, C, axpbys )

#undef GENTFUNC
#define GENTFUNC( opname, ctypea, cha, ctypex, chx, ctypeb, chb, ctypey, chy, ctypec, chc ) \
UNIT_TEST(cha,chx,chb,chy,chc,opname) \
( \
	for ( auto a : test_values<ctypea>() ) \
	for ( auto x : test_values<ctypex>() ) \
	for ( auto b : test_values<ctypeb>() ) \
	for ( auto y : test_values<ctypey>() ) \
	{ \
		auto y0 = convert<ctypey>(       convert_prec<ctypec>( a ) * \
		                           conj( convert_prec<ctypec>( x ) ) + \
		                                 convert_prec<ctypec>( b ) * \
		                                 convert_prec<ctypec>( y ) ); \
\
		INFO( "a:        " << a ); \
		INFO( "x:        " << x ); \
		INFO( "b:        " << b ); \
		INFO( "y (init): " << y ); \
\
		bli_taxpbyjs( cha,chx,chb,chy,chc, a, x, b, y ); \
\
		INFO( "y (C++):  " << y0 ); \
		INFO( "y (BLIS): " << y ); \
\
		check<ctypec>( y, y0 ); \
	} \
)

INSERT_GENTFUNC_MIX5( RC, RC, RC, RC, C, axpbyjs )

#undef GENTFUNC
#define GENTFUNC( opname, ctypea, cha, ctypex, chx, ctypeb, chb, ctypey, chy, ctypec, chc ) \
UNIT_TEST(cha,chx,chb,chy,chc,opname) \
( \
	for ( auto a : test_values<ctypea>() ) \
	for ( auto x : test_values<ctypex>() ) \
	for ( auto b : test_values<ctypeb>() ) \
	for ( auto y : test_values<ctypey>() ) \
	{ \
		auto y0 = convert<ctypey>( convert_prec<ctypec>( a ) * \
		                           convert_prec<ctypec>( x ) + \
		                           convert_prec<ctypec>( b ) * \
		                           convert_prec<ctypec>( y ) ); \
\
		INFO( "a:        " << a ); \
		INFO( "x:        " << x ); \
		INFO( "b:        " << b ); \
		INFO( "y (init): " << y ); \
\
		bli_taxpbyris( cha,chx,chb,chy,chc, \
		               real( a ), imag( a ), \
		               real( x ), imag( x ), \
		               real( b ), imag( b ), \
		               real( y ), imag( y ) ); \
\
		INFO( "y (C++):  " << y0 ); \
		INFO( "y (BLIS): " << y ); \
\
		check<ctypec>( y, y0 ); \
	} \
)

INSERT_GENTFUNC_MIX5( RC, RC, RC, RC, C, axpbyris )

#undef GENTFUNC
#define GENTFUNC( opname, ctypea, cha, ctypex, chx, ctypeb, chb, ctypey, chy, ctypec, chc ) \
UNIT_TEST(cha,chx,chb,chy,chc,opname) \
( \
	for ( auto a : test_values<ctypea>() ) \
	for ( auto x : test_values<ctypex>() ) \
	for ( auto b : test_values<ctypeb>() ) \
	for ( auto y : test_values<ctypey>() ) \
	{ \
		auto y0 = convert<ctypey>(       convert_prec<ctypec>( a ) * \
		                           conj( convert_prec<ctypec>( x ) ) + \
		                                 convert_prec<ctypec>( b ) * \
		                                 convert_prec<ctypec>( y ) ); \
\
		INFO( "a:        " << a ); \
		INFO( "x:        " << x ); \
		INFO( "b:        " << b ); \
		INFO( "y (init): " << y ); \
\
		bli_taxpbyjris( cha,chx,chb,chy,chc, \
		                real( a ), imag( a ), \
		                real( x ), imag( x ), \
		                real( b ), imag( b ), \
		                real( y ), imag( y ) ); \
\
		INFO( "y (C++):  " << y0 ); \
		INFO( "y (BLIS): " << y ); \
\
		check<ctypec>( y, y0 ); \
	} \
)

INSERT_GENTFUNC_MIX5( RC, RC, RC, RC, C, axpbyjris )

/******************************************************************************
 *
 * axpys
 *
 *****************************************************************************/

#undef GENTFUNC
#define GENTFUNC( opname, ctypea, cha, ctypex, chx, ctypey, chy, ctypec, chc ) \
UNIT_TEST(cha,chx,chy,chc,opname) \
( \
	for ( auto a : test_values<ctypea>() ) \
	for ( auto x : test_values<ctypex>() ) \
	for ( auto y : test_values<ctypey>() ) \
	{ \
		auto y0 = convert<ctypey>( convert_prec<ctypec>( a ) * \
		                           convert_prec<ctypec>( x ) + \
		                           convert_prec<ctypec>( y ) ); \
\
		INFO( "a:        " << a ); \
		INFO( "x:        " << x ); \
		INFO( "y (init): " << y ); \
\
		bli_taxpys( cha,chx,chy,chc, a, x, y ); \
\
		INFO( "y (C++):  " << y0 ); \
		INFO( "y (BLIS): " << y ); \
\
		check<ctypec>( y, y0 ); \
	} \
)

INSERT_GENTFUNC_MIX4( RC, RC, RC, C, axpys )

#undef GENTFUNC
#define GENTFUNC( opname, ctypea, cha, ctypex, chx, ctypey, chy, ctypec, chc ) \
UNIT_TEST(cha,chx,chy,chc,opname) \
( \
	for ( auto a : test_values<ctypea>() ) \
	for ( auto x : test_values<ctypex>() ) \
	for ( auto y : test_values<ctypey>() ) \
	{ \
		auto y0 = convert<ctypey>(       convert_prec<ctypec>( a ) * \
		                           conj( convert_prec<ctypec>( x ) ) + \
		                                 convert_prec<ctypec>( y ) ); \
\
		INFO( "a:        " << a ); \
		INFO( "x:        " << x ); \
		INFO( "y (init): " << y ); \
\
		bli_taxpyjs( cha,chx,chy,chc, a, x, y ); \
\
		INFO( "y (C++):  " << y0 ); \
		INFO( "y (BLIS): " << y ); \
\
		check<ctypec>( y, y0 ); \
	} \
)

INSERT_GENTFUNC_MIX4( RC, RC, RC, C, axpyjs )

#undef GENTFUNC
#define GENTFUNC( opname, ctypea, cha, ctypex, chx, ctypey, chy, ctypec, chc ) \
UNIT_TEST(cha,chx,chy,chc,opname) \
( \
	for ( auto a : test_values<ctypea>() ) \
	for ( auto x : test_values<ctypex>() ) \
	for ( auto y : test_values<ctypey>() ) \
	{ \
		auto y0 = convert<ctypey>( convert_prec<ctypec>( a ) * \
		                           convert_prec<ctypec>( x ) + \
		                           convert_prec<ctypec>( y ) ); \
\
		INFO( "a:        " << a ); \
		INFO( "x:        " << x ); \
		INFO( "y (init): " << y ); \
\
		bli_taxpyris( cha,chx,chy,chc, \
		              real( a ), imag( a ), \
		              real( x ), imag( x ), \
		              real( y ), imag( y ) ); \
\
		INFO( "y (C++):  " << y0 ); \
		INFO( "y (BLIS): " << y ); \
\
		check<ctypec>( y, y0 ); \
	} \
)

INSERT_GENTFUNC_MIX4( RC, RC, RC, C, axpyris )

#undef GENTFUNC
#define GENTFUNC( opname, ctypea, cha, ctypex, chx, ctypey, chy, ctypec, chc ) \
UNIT_TEST(cha,chx,chy,chc,opname) \
( \
	for ( auto a : test_values<ctypea>() ) \
	for ( auto x : test_values<ctypex>() ) \
	for ( auto y : test_values<ctypey>() ) \
	{ \
		auto y0 = convert<ctypey>(       convert_prec<ctypec>( a ) * \
		                           conj( convert_prec<ctypec>( x ) ) + \
		                                 convert_prec<ctypec>( y ) ); \
\
		INFO( "a:        " << a ); \
		INFO( "x:        " << x ); \
		INFO( "y (init): " << y ); \
\
		bli_taxpyjris( cha,chx,chy,chc, \
		               real( a ), imag( a ), \
		               real( x ), imag( x ), \
		               real( y ), imag( y ) ); \
\
		INFO( "y (C++):  " << y0 ); \
		INFO( "y (BLIS): " << y ); \
\
		check<ctypec>( y, y0 ); \
	} \
)

INSERT_GENTFUNC_MIX4( RC, RC, RC, C, axpyjris )

/******************************************************************************
 *
 * conjs
 *
 *****************************************************************************/

#undef GENTFUNC
#define GENTFUNC( opname, ctypey, chy ) \
UNIT_TEST(chy,opname) \
( \
	for ( auto y : test_values<ctypey>() ) \
	{ \
		auto y0 = conj( y ); \
\
		INFO( "y (init): " << y ); \
\
		bli_tconjs( chy, y ); \
\
		INFO( "y (C++):  " << y0 ); \
		INFO( "y (BLIS): " << y ); \
\
		check<ctypey>( y, y0 ); \
	} \
)

INSERT_GENTFUNC_MIX1( C, conjs )

#undef GENTFUNC
#define GENTFUNC( opname, ctypey, chy ) \
UNIT_TEST(chy,opname) \
( \
	for ( auto y : test_values<ctypey>() ) \
	{ \
		auto y0 = conj( y ); \
\
		INFO( "y (init): " << y ); \
\
		bli_tconjris( chy, real( y ), imag( y ) ); \
\
		INFO( "y (C++):  " << y0 ); \
		INFO( "y (BLIS): " << y ); \
\
		check<ctypey>( y, y0 ); \
	} \
)

INSERT_GENTFUNC_MIX1( C, conjris )

/******************************************************************************
 *
 * copycjs
 *
 *****************************************************************************/

#undef GENTFUNC
#define GENTFUNC( opname, ctypex, chx, ctypey, chy ) \
UNIT_TEST(chx,chy,opname) \
( \
	for ( auto conjx : { BLIS_CONJUGATE, BLIS_NO_CONJUGATE } ) \
	for ( auto x : test_values<ctypex>() ) \
	{ \
		auto y0 = convert<ctypey>( bli_is_conj( conjx ) ? conj( x ) : x ); \
\
		INFO( "conjx:    " << bli_is_conj( conjx ) ); \
		INFO( "x:        " << x ); \
\
		ctypey y; \
		bli_tcopycjs( chx,chy, conjx, x, y ); \
\
		INFO( "y (C++):  " << y0 ); \
		INFO( "y (BLIS): " << y ); \
\
		check<ctypey>( y, y0 ); \
	} \
)

INSERT_GENTFUNC_MIX2( RC, C, copycjs )

#undef GENTFUNC
#define GENTFUNC( opname, ctypex, chx, ctypey, chy ) \
UNIT_TEST(chx,chy,opname) \
( \
	for ( auto conjx : { BLIS_CONJUGATE, BLIS_NO_CONJUGATE } ) \
	for ( auto x : test_values<ctypex>() ) \
	{ \
		auto y0 = convert<ctypey>( bli_is_conj( conjx ) ? conj( x ) : x ); \
\
		INFO( "conjx:    " << bli_is_conj( conjx ) ); \
		INFO( "x:        " << x); \
\
		ctypey y; \
		bli_tcopycjris( chx,chy, conjx, \
		                real( x ), imag( x ), \
		                real( y ), imag( y ) ); \
\
		INFO( "y (C++):  " << y0 ); \
		INFO( "y (BLIS): " << y ); \
\
		check<ctypey>( y, y0 ); \
	} \
)

INSERT_GENTFUNC_MIX2( RC, C, copycjris )

/******************************************************************************
 *
 * copynzs
 *
 *****************************************************************************/

#undef GENTFUNC
#define GENTFUNC( opname, ctypex, chx, ctypey, chy ) \
UNIT_TEST(chx,chy,opname) \
( \
	for ( auto x : test_values<ctypex>() ) \
	for ( auto y : test_values<ctypey>() ) \
	{ \
		auto y0 = y; \
		real( y0 ) = real( x ); \
		if ( is_complex<ctypex>::value ) \
			imag( y0 ) = imag( x ); \
\
		INFO( "x:        " << x ); \
		INFO( "y (orig): " << y ); \
\
		bli_tcopynzs( chx,chy, x, y ); \
\
		INFO( "y (C++):  " << y0 ); \
		INFO( "y (BLIS): " << y ); \
\
		check<ctypey>( y, y0 ); \
	} \
)

INSERT_GENTFUNC_MIX2( RC, C, copynzs )

#undef GENTFUNC
#define GENTFUNC( opname, ctypex, chx, ctypey, chy ) \
UNIT_TEST(chx,chy,opname) \
( \
	for ( auto x : test_values<ctypex>() ) \
	for ( auto y : test_values<ctypey>() ) \
	{ \
		auto y0 = y; \
		real( y0 ) = real( x ); \
		if ( is_complex<ctypex>::value ) \
			imag( y0 ) = -imag( x ); \
\
		INFO( "x:        " << x ); \
		INFO( "y (orig): " << y ); \
\
		bli_tcopynzjs( chx,chy, x, y ); \
\
		INFO( "y (C++):  " << y0 ); \
		INFO( "y (BLIS): " << y ); \
\
		check<ctypey>( y, y0 ); \
	} \
)

INSERT_GENTFUNC_MIX2( RC, C, copynzjs )

#undef GENTFUNC
#define GENTFUNC( opname, ctypex, chx, ctypey, chy ) \
UNIT_TEST(chx,chy,opname) \
( \
	for ( auto x : test_values<ctypex>() ) \
	for ( auto y : test_values<ctypey>() ) \
	{ \
		auto y0 = y; \
		real( y0 ) = real( x ); \
		if ( is_complex<ctypex>::value ) \
			imag( y0 ) = imag( x ); \
\
		INFO( "x:        " << x ); \
		INFO( "y (orig): " << y ); \
\
		bli_tcopynzris( chx,chy, \
		                real( x ), imag( x ), \
		                real( y ), imag( y ) ); \
\
		INFO( "y (C++):  " << y0 ); \
		INFO( "y (BLIS): " << y ); \
\
		check<ctypey>( y, y0 ); \
	} \
)

INSERT_GENTFUNC_MIX2( RC, C, copynzris )

#undef GENTFUNC
#define GENTFUNC( opname, ctypex, chx, ctypey, chy ) \
UNIT_TEST(chx,chy,opname) \
( \
	for ( auto x : test_values<ctypex>() ) \
	for ( auto y : test_values<ctypey>() ) \
	{ \
		auto y0 = y; \
		real( y0 ) = real( x ); \
		if ( is_complex<ctypex>::value ) \
			imag( y0 ) = -imag( x ); \
\
		INFO( "x:        " << x ); \
		INFO( "y (orig): " << y ); \
\
		bli_tcopynzjris( chx,chy, \
		                 real( x ), imag( x ), \
		                 real( y ), imag( y ) ); \
\
		INFO( "y (C++):  " << y0 ); \
		INFO( "y (BLIS): " << y ); \
\
		check<ctypey>( y, y0 ); \
	} \
)

INSERT_GENTFUNC_MIX2( RC, C, copynzjris )

/******************************************************************************
 *
 * copys
 *
 *****************************************************************************/

#undef GENTFUNC
#define GENTFUNC( opname, ctypex, chx, ctypey, chy ) \
UNIT_TEST(chx,chy,opname) \
( \
	for ( auto x : test_values<ctypex>() ) \
	{ \
		auto y0 = convert<ctypey>( x ); \
\
		INFO( "x:        " << x ); \
\
		ctypey y; \
		bli_tcopys( chx,chy, x, y ); \
\
		INFO( "y (C++):  " << y0 ); \
		INFO( "y (BLIS): " << y ); \
\
		check<ctypey>( y, y0 ); \
	} \
)

INSERT_GENTFUNC_MIX2( RC, C, copys )

#undef GENTFUNC
#define GENTFUNC( opname, ctypex, chx, ctypey, chy ) \
UNIT_TEST(chx,chy,opname) \
( \
	for ( auto x : test_values<ctypex>() ) \
	{ \
		auto y0 = convert<ctypey>( conj( x ) ); \
\
		INFO( "x:        " << x ); \
\
		ctypey y; \
		bli_tcopyjs( chx,chy, x, y ); \
\
		INFO( "y (C++):  " << y0 ); \
		INFO( "y (BLIS): " << y ); \
\
		check<ctypey>( y, y0 ); \
	} \
)

INSERT_GENTFUNC_MIX2( RC, C, copyjs )

#undef GENTFUNC
#define GENTFUNC( opname, ctypex, chx, ctypey, chy ) \
UNIT_TEST(chx,chy,opname) \
( \
	for ( auto x : test_values<ctypex>() ) \
	{ \
		auto y0 = convert<ctypey>( x ); \
\
		INFO( "x:        " << x ); \
\
		ctypey y; \
		bli_tcopyris( chx,chy, \
		              real( x ), imag( x ), \
		              real( y ), imag( y ) ); \
\
		INFO( "y (C++):  " << y0 ); \
		INFO( "y (BLIS): " << y ); \
\
		check<ctypey>( y, y0 ); \
	} \
)

INSERT_GENTFUNC_MIX2( RC, C, copyris )

#undef GENTFUNC
#define GENTFUNC( opname, ctypex, chx, ctypey, chy ) \
UNIT_TEST(chx,chy,opname) \
( \
	for ( auto x : test_values<ctypex>() ) \
	{ \
		auto y0 = convert<ctypey>( conj( x ) ); \
\
		INFO( "x:        " << x ); \
\
		ctypey y; \
		bli_tcopyjris( chx,chy, \
		               real( x ), imag( x ), \
		               real( y ), imag( y ) ); \
\
		INFO( "y (C++):  " << y0 ); \
		INFO( "y (BLIS): " << y ); \
\
		check<ctypey>( y, y0 ); \
	} \
)

INSERT_GENTFUNC_MIX2( RC, C, copyjris )

#undef GENTFUNC
#define GENTFUNC( opname, ctypex, chx, ctypey, chy ) \
UNIT_TEST(chx,chy,opname) \
( \
	constexpr auto M = 4; \
	constexpr auto N = 4; \
\
	for ( auto x : test_values<ctypex>() ) \
	{ \
		auto xmn = tile<M,N>( x ); \
		auto ymn = tile<M,N,ctypey>(); \
\
		INFO( "row-major" ); \
\
		auto ymn0 = ymn; \
		axpbys_mxn<ctypey,BLIS_NO_TRANSPOSE>( 1.0, xmn, 0.0, ymn0, dense ); \
\
		INFO( "x:\n" << xmn ); \
\
		bli_tcopys_mxn( chx,chy, M, N, &xmn[0][0], N, 1, &ymn[0][0], N, 1 ); \
\
		INFO( "y (C++):\n" << ymn0 ); \
		INFO( "y (BLIS):\n" << ymn ); \
\
		check<ctypey>( ymn, ymn0 ); \
	} \
\
	for ( auto x : test_values<ctypex>() ) \
	{ \
		auto xmn = tile<M,N>( x ); \
		auto ymn = tile<M,N,ctypey>(); \
\
		INFO( "column-major" ); \
\
		auto ymn0 = ymn; \
		axpbys_mxn<ctypey,BLIS_TRANSPOSE>( 1.0, xmn, 0.0, ymn0, dense ); \
\
		INFO( "x:\n" << xmn ); \
\
		bli_tcopys_mxn( chx,chy, N, M, &xmn[0][0], 1, N, &ymn[0][0], 1, N ); \
\
		INFO( "y (C++):\n" << ymn0 ); \
		INFO( "y (BLIS):\n" << ymn ); \
\
		check<ctypey>( ymn, ymn0 ); \
	} \
)

INSERT_GENTFUNC_MIX2( RC, C, copys_mxn )

/******************************************************************************
 *
 * dots
 *
 *****************************************************************************/

// No tests, dot(x, y, a) == axpy(y, x, a)

/******************************************************************************
 *
 * eqs
 *
 *****************************************************************************/

#undef GENTFUNC
#define GENTFUNC( opname, ctypex, chx, ctypey, chy, ctypec, chc ) \
UNIT_TEST(chx,chy,chc,opname) \
( \
	for ( auto x : test_values<ctypex>() ) \
	for ( auto y : test_values<ctypey>() ) \
	{ \
		auto expected = convert_prec<ctypec>( x ) == \
		                convert_prec<ctypec>( y ); \
\
		INFO( "x: " << x ); \
		INFO( "y: " << y ); \
\
		auto found = bli_teqs( chx,chy,chc, x, y ); \
\
		INFO( "expected: " << expected ); \
		INFO( "found   : " << found ); \
\
		REQUIRE( expected == found ); \
	} \
)

INSERT_GENTFUNC_MIX3( RC, RC, RC, eqs )

#undef GENTFUNC
#define GENTFUNC( opname, ctypex, chx, ctypey, chy, ctypec, chc ) \
UNIT_TEST(chx,chy,chc,opname) \
( \
	for ( auto x : test_values<ctypex>() ) \
	for ( auto y : test_values<ctypey>() ) \
	{ \
		auto expected = convert_prec<ctypec>( x ) == \
		                convert_prec<ctypec>( y ); \
\
		INFO( "x: " << x ); \
		INFO( "y: " << y ); \
\
		auto found = bli_teqris( chx,chy,chc, \
		                         real(x), imag(x), \
		                         real(y), imag(y) ); \
\
		INFO( "expected: " << expected ); \
		INFO( "found   : " << found ); \
\
		REQUIRE( expected == found ); \
	} \
)

INSERT_GENTFUNC_MIX3( RC, RC, RC, eqris )

#undef GENTFUNC
#define GENTFUNC( opname, ctypex, chx ) \
UNIT_TEST(chx,opname) \
( \
	for ( auto x : test_values<ctypex>() ) \
	{ \
		auto expected = x == convert_prec<ctypex>( 1.0 ); \
\
		INFO( "x: " << x ); \
\
		auto found = bli_teq1ris( chx, real( x ), imag( x ) ); \
\
		INFO( "expected: " << expected ); \
		INFO( "found   : " << found ); \
\
		REQUIRE( expected == found ); \
	} \
)

INSERT_GENTFUNC_MIX1( RC, eq1ris )

#undef GENTFUNC
#define GENTFUNC( opname, ctypex, chx ) \
UNIT_TEST(chx,opname) \
( \
	for ( auto x : test_values<ctypex>() ) \
	{ \
		auto expected = x == convert_prec<ctypex>( 0.0 ); \
\
		INFO( "x: " << x ); \
\
		auto found = bli_teq0ris( chx, real( x ), imag( x ) ); \
\
		INFO( "expected: " << expected ); \
		INFO( "found   : " << found ); \
\
		REQUIRE( expected == found ); \
	} \
)

INSERT_GENTFUNC_MIX1( RC, eq0ris )

#undef GENTFUNC
#define GENTFUNC( opname, ctypex, chx ) \
UNIT_TEST(chx,opname) \
( \
	for ( auto x : test_values<ctypex>() ) \
	{ \
		auto expected = x == convert_prec<ctypex>( -1.0 ); \
\
		INFO( "x: " << x ); \
\
		auto found = bli_teqm1ris( chx, real( x ), imag( x ) ); \
\
		INFO( "expected: " << expected ); \
		INFO( "found   : " << found ); \
\
		REQUIRE( expected == found ); \
	} \
)

INSERT_GENTFUNC_MIX1( RC, eqm1ris )

#undef GENTFUNC
#define GENTFUNC( opname, ctypex, chx ) \
UNIT_TEST(chx,opname) \
( \
	for ( auto x : test_values<ctypex>() ) \
	{ \
		auto expected = x == convert_prec<ctypex>( 1.0 ); \
\
		INFO( "x: " << x ); \
\
		auto found = bli_teq1s( chx, x ); \
\
		INFO( "expected: " << expected ); \
		INFO( "found   : " << found ); \
\
		REQUIRE( expected == found ); \
	} \
)

INSERT_GENTFUNC_MIX1( RC, eq1s )

#undef GENTFUNC
#define GENTFUNC( opname, ctypex, chx ) \
UNIT_TEST(chx,opname) \
( \
	for ( auto x : test_values<ctypex>() ) \
	{ \
		auto expected = x == convert_prec<ctypex>( 0.0 ); \
\
		INFO( "x: " << x ); \
\
		auto found = bli_teq0s( chx, x ); \
\
		INFO( "expected: " << expected ); \
		INFO( "found   : " << found ); \
\
		REQUIRE( expected == found ); \
	} \
)

INSERT_GENTFUNC_MIX1( RC, eq0s )

#undef GENTFUNC
#define GENTFUNC( opname, ctypex, chx ) \
UNIT_TEST(chx,opname) \
( \
	for ( auto x : test_values<ctypex>() ) \
	{ \
		auto expected = x == convert_prec<ctypex>( -1.0 ); \
\
		INFO( "x: " << x ); \
\
		auto found = bli_teqm1s( chx, x ); \
\
		INFO( "expected: " << expected ); \
		INFO( "found   : " << found ); \
\
		REQUIRE( expected == found ); \
	} \
)

INSERT_GENTFUNC_MIX1( RC, eqm1s )

/******************************************************************************
 *
 * fprints
 *
 *****************************************************************************/

// No tests

/******************************************************************************
 *
 * gets
 *
 *****************************************************************************/

#undef GENTFUNC
#define GENTFUNC( opname, ctypex, chx, ctypey, chy ) \
UNIT_TEST(chx,chy,opname) \
( \
\
	using ctypeyr = make_real_t<ctypey>; \
	using ctypeyc = make_complex_t<ctypey>; \
\
	for ( auto x : test_values<ctypex>() ) \
	{ \
		auto y0 = convert<ctypeyc>( x ); \
\
		INFO( "x:        " << x ); \
\
		ctypeyr yr, yi; \
		bli_tgets( chx,chy, x, yr, yi ); \
\
		INFO( "yr (C++):  " << real( y0 ) ); \
		INFO( "yi (C++):  " << imag( y0 ) ); \
		INFO( "yr (BLIS): " << yr ); \
		INFO( "yi (BLIS): " << yi ); \
\
		check<ctypey>( yr, real( y0 ) ); \
		check<ctypey>( yi, imag( y0 ) ); \
	} \
)

INSERT_GENTFUNC_MIX2( RC, RC, gets )

/******************************************************************************
 *
 * inverts
 *
 *****************************************************************************/

#undef GENTFUNC
#define GENTFUNC( opname, ctypex, chx, ctypec, chc ) \
UNIT_TEST(chx,chc,opname) \
( \
	for ( auto x : test_values<ctypex>() ) \
	{ \
		auto y0 = convert<ctypex>( convert_prec<ctypec>( 1.0 ) / \
		                           convert_prec<ctypec>( x ) ); \
\
		INFO( "x:        " << x ); \
\
		ctypex y = x; \
		bli_tinverts( chx,chc, y ); \
\
		INFO( "y (C++):  " << y0 ); \
		INFO( "y (BLIS): " << y ); \
\
		check<ctypec>( y, y0 ); \
	} \
)

INSERT_GENTFUNC_MIX2( RC, C, inverts )

#undef GENTFUNC
#define GENTFUNC( opname, ctypex, chx, ctypec, chc ) \
UNIT_TEST(chx,chc,opname) \
( \
	for ( auto x : test_values<ctypex>() ) \
	{ \
		auto y0 = convert<ctypex>( convert_prec<ctypec>( 1.0 ) / \
		                           convert_prec<ctypec>( x ) ); \
\
		INFO( "x:        " << x ); \
\
		ctypex y = x; \
		bli_tinvertris( chx,chc, real( y ), imag( y ) ); \
\
		INFO( "y (C++):  " << y0 ); \
		INFO( "y (BLIS): " << y ); \
\
		check<ctypec>( y, y0 ); \
	} \
)

INSERT_GENTFUNC_MIX2( RC, C, invertris )

/******************************************************************************
 *
 * invscals
 *
 *****************************************************************************/

#undef GENTFUNC
#define GENTFUNC( opname, ctypea, cha, ctypex, chx, ctypec, chc ) \
UNIT_TEST(cha,chx,chc,opname) \
( \
	for ( auto a : test_values<ctypea>() ) \
	for ( auto x : test_values<ctypex>() ) \
	{ \
		auto y0 = convert<ctypex>( convert_prec<ctypec>( x ) / \
		                           convert_prec<ctypec>( a ) ); \
\
		INFO( "a:        " << a ); \
		INFO( "x:        " << x ); \
\
		ctypex y = x; \
		bli_tinvscals( cha,chx,chc, a, y ); \
\
		INFO( "y (C++):  " << y0 ); \
		INFO( "y (BLIS): " << y ); \
\
		check<ctypec>( y, y0 ); \
	} \
)

INSERT_GENTFUNC_MIX3( RC, RC, C, invscals )

#undef GENTFUNC
#define GENTFUNC( opname, ctypea, cha, ctypex, chx, ctypec, chc ) \
UNIT_TEST(cha,chx,chc,opname) \
( \
	for ( auto a : test_values<ctypea>() ) \
	for ( auto x : test_values<ctypex>() ) \
	{ \
		auto y0 = convert<ctypex>( convert_prec<ctypec>( x ) / \
		                           convert_prec<ctypec>( conj( a ) ) ); \
\
		INFO( "a:        " << a ); \
		INFO( "x:        " << x ); \
\
		ctypex y = x; \
		bli_tinvscaljs( cha,chx,chc, a, y ); \
\
		INFO( "y (C++):  " << y0 ); \
		INFO( "y (BLIS): " << y ); \
\
		check<ctypec>( y, y0 ); \
	} \
)

INSERT_GENTFUNC_MIX3( RC, RC, C, invscaljs )

#undef GENTFUNC
#define GENTFUNC( opname, ctypea, cha, ctypex, chx, ctypec, chc ) \
UNIT_TEST(cha,chx,chc,opname) \
( \
	for ( auto a : test_values<ctypea>() ) \
	for ( auto x : test_values<ctypex>() ) \
	{ \
		auto y0 = convert<ctypex>( convert_prec<ctypec>( x ) / \
		                           convert_prec<ctypec>( a ) ); \
\
		INFO( "a:        " << a ); \
		INFO( "x:        " << x ); \
\
		ctypex y = x; \
		bli_tinvscalris( cha,chx,chc, \
		                 real( a ), imag( a ), \
		                 real( y ), imag( y ) ); \
\
		INFO( "y (C++):  " << y0 ); \
		INFO( "y (BLIS): " << y ); \
\
		check<ctypec>( y, y0 ); \
	} \
)

INSERT_GENTFUNC_MIX3( RC, RC, C, invscalris )

#undef GENTFUNC
#define GENTFUNC( opname, ctypea, cha, ctypex, chx, ctypec, chc ) \
UNIT_TEST(cha,chx,chc,opname) \
( \
	for ( auto a : test_values<ctypea>() ) \
	for ( auto x : test_values<ctypex>() ) \
	{ \
		auto y0 = convert<ctypex>( convert_prec<ctypec>( x ) / \
		                           convert_prec<ctypec>( conj( a ) ) ); \
\
		INFO( "a:        " << a ); \
		INFO( "x:        " << x ); \
\
		ctypex y = x; \
		bli_tinvscaljris( cha,chx,chc, \
		                  real( a ), imag( a ), \
		                  real( y ), imag( y ) ); \
\
		INFO( "y (C++):  " << y0 ); \
		INFO( "y (BLIS): " << y ); \
\
		check<ctypec>( y, y0 ); \
	} \
)

INSERT_GENTFUNC_MIX3( RC, RC, C, invscaljris )

/******************************************************************************
 *
 * neg2s
 *
 *****************************************************************************/

#undef GENTFUNC
#define GENTFUNC( opname, ctypex, chx, ctypey, chy ) \
UNIT_TEST(chx,chy,opname) \
( \
	for ( auto x : test_values<ctypex>() ) \
	{ \
		auto y0 = convert<ctypey>( -x ); \
\
		INFO( "x:        " << x ); \
\
		ctypey y; \
		bli_tneg2s( chx,chy, x, y ); \
\
		INFO( "y (C++):  " << y0 ); \
		INFO( "y (BLIS): " << y ); \
\
		check<ctypey>( y, y0 ); \
	} \
)

INSERT_GENTFUNC_MIX2( RC, C, neg2s )

#undef GENTFUNC
#define GENTFUNC( opname, ctypex, chx, ctypey, chy ) \
UNIT_TEST(chx,chy,opname) \
( \
	for ( auto x : test_values<ctypex>() ) \
	{ \
		auto y0 = convert<ctypey>( -x ); \
\
		INFO( "x:        " << x ); \
\
		ctypey y; \
		bli_tneg2ris( chx,chy, \
		              real( x ), imag( x ), \
		              real( y ), imag( y ) ); \
\
		INFO( "y (C++):  " << y0 ); \
		INFO( "y (BLIS): " << y ); \
\
		check<ctypey>( y, y0 ); \
	} \
)

INSERT_GENTFUNC_MIX2( RC, C, neg2ris )

/******************************************************************************
 *
 * randnp2s
 *
 *****************************************************************************/

// No tests

/******************************************************************************
 *
 * rands
 *
 *****************************************************************************/

// No tests

/******************************************************************************
 *
 * scal2s
 *
 *****************************************************************************/

#undef GENTFUNC
#define GENTFUNC( opname, ctypea, cha, ctypex, chx, ctypey, chy, ctypec, chc ) \
UNIT_TEST(cha,chx,chy,chc,opname) \
( \
	for ( auto a : test_values<ctypea>() ) \
	for ( auto x : test_values<ctypex>() ) \
	{ \
		auto y0 = convert<ctypey>( convert_prec<ctypec>( a ) * \
		                           convert_prec<ctypec>( x ) ); \
\
		INFO( "a:        " << a ); \
		INFO( "x:        " << x ); \
\
		ctypey y; \
		bli_tscal2s( cha,chx,chy,chc, a, x, y ); \
\
		INFO( "y (C++):  " << y0 ); \
		INFO( "y (BLIS): " << y ); \
\
		check<ctypec>( y, y0 ); \
	} \
)

INSERT_GENTFUNC_MIX4( RC, RC, RC, C, scal2s )

#undef GENTFUNC
#define GENTFUNC( opname, ctypea, cha, ctypex, chx, ctypey, chy, ctypec, chc ) \
UNIT_TEST(cha,chx,chy,chc,opname) \
( \
	for ( auto a : test_values<ctypea>() ) \
	for ( auto x : test_values<ctypex>() ) \
	{ \
		auto y0 = convert<ctypey>( convert_prec<ctypec>( a ) * \
		                           convert_prec<ctypec>( conj( x ) ) ); \
\
		INFO( "a:        " << a ); \
		INFO( "x:        " << x ); \
\
		ctypey y; \
		bli_tscal2js( cha,chx,chy,chc, a, x, y ); \
\
		INFO( "y (C++):  " << y0 ); \
		INFO( "y (BLIS): " << y ); \
\
		check<ctypec>( y, y0 ); \
	} \
)

INSERT_GENTFUNC_MIX4( RC, RC, RC, C, scal2js )

#undef GENTFUNC
#define GENTFUNC( opname, ctypea, cha, ctypex, chx, ctypey, chy, ctypec, chc ) \
UNIT_TEST(cha,chx,chy,chc,opname) \
( \
	for ( auto a : test_values<ctypea>() ) \
	for ( auto x : test_values<ctypex>() ) \
	{ \
		auto y0 = convert<ctypey>( convert_prec<ctypec>( a ) * \
		                           convert_prec<ctypec>( x ) ); \
\
		INFO( "a:        " << a ); \
		INFO( "x:        " << x ); \
\
		ctypey y; \
		bli_tscal2ris( cha,chx,chy,chc, \
		               real( a ), imag( a ), \
		               real( x ), imag( x ), \
		               real( y ), imag( y ) ); \
\
		INFO( "y (C++):  " << y0 ); \
		INFO( "y (BLIS): " << y ); \
\
		check<ctypec>( y, y0 ); \
	} \
)

INSERT_GENTFUNC_MIX4( RC, RC, RC, C, scal2ris )

#undef GENTFUNC
#define GENTFUNC( opname, ctypea, cha, ctypex, chx, ctypey, chy, ctypec, chc ) \
UNIT_TEST(cha,chx,chy,chc,opname) \
( \
	for ( auto a : test_values<ctypea>() ) \
	for ( auto x : test_values<ctypex>() ) \
	{ \
		auto y0 = convert<ctypey>( convert_prec<ctypec>( a ) * \
		                           convert_prec<ctypec>( conj( x ) ) ); \
\
		INFO( "a:        " << a ); \
		INFO( "x:        " << x ); \
\
		ctypey y; \
		bli_tscal2jris( cha,chx,chy,chc, \
		                real( a ), imag( a ), \
		                real( x ), imag( x ), \
		                real( y ), imag( y ) ); \
\
		INFO( "y (C++):  " << y0 ); \
		INFO( "y (BLIS): " << y ); \
\
		check<ctypec>( y, y0 ); \
	} \
)

INSERT_GENTFUNC_MIX4( RC, RC, RC, C, scal2jris )

#undef GENTFUNC
#define GENTFUNC( opname, ctypea, cha, ctypex, chx, ctypey, chy, ctypec, chc ) \
UNIT_TEST(cha,chx,chy,chc,opname) \
( \
	constexpr auto M = 4; \
	constexpr auto N = 4; \
\
	for ( auto conjx : { BLIS_CONJUGATE, BLIS_NO_CONJUGATE } ) \
	for ( auto a : test_values<ctypea>() ) \
	for ( auto x : test_values<ctypex>() ) \
	{ \
		auto xmn = tile<M,N>( x ); \
		auto ymn = tile<M,N,ctypey>(); \
\
		INFO( "row-major" ); \
\
		auto ymn0 = ymn; \
		axpbys_mxn<ctypec,BLIS_NO_TRANSPOSE>( a, bli_is_conj( conjx ) ? conj( xmn ) : xmn, 0.0, ymn0, dense ); \
\
		INFO( "conjx: " << bli_is_conj( conjx ) ); \
		INFO( "a:     " << a ); \
		INFO( "x:\n" << xmn ); \
\
		bli_tscal2s_mxn( cha,chx,chy,chc, conjx, M, N, &a, &xmn[0][0], N, 1, &ymn[0][0], N, 1 ); \
\
		INFO( "y (C++):\n" << ymn0 ); \
		INFO( "y (BLIS):\n" << ymn ); \
\
		check<ctypec>( ymn, ymn0 ); \
	} \
\
	for ( auto conjx : { BLIS_CONJUGATE, BLIS_NO_CONJUGATE } ) \
	for ( auto a : test_values<ctypea>() ) \
	for ( auto x : test_values<ctypex>() ) \
	{ \
		auto xmn = tile<M,N>( x ); \
		auto ymn = tile<M,N,ctypey>(); \
\
		INFO("column-major"); \
\
		auto ymn0 = ymn; \
		axpbys_mxn<ctypec,BLIS_TRANSPOSE>( a, bli_is_conj( conjx ) ? conj( xmn ) : xmn, 0.0, ymn0, dense ); \
\
		INFO( "conjx: " << bli_is_conj( conjx ) ); \
		INFO( "a:     " << a ); \
		INFO( "x:\n" << xmn ); \
\
		bli_tscal2s_mxn( cha,chx,chy,chc, conjx, N, M, &a, &xmn[0][0], 1, N, &ymn[0][0], 1, N ); \
\
		INFO( "y (C++):\n" << ymn0 ); \
		INFO( "y (BLIS):\n" << ymn ); \
\
		check<ctypec>( ymn, ymn0 ); \
	} \
)

INSERT_GENTFUNC_MIX4( RC, RC, RC, C, scal2s_mxn )

#undef GENTFUNC
#define GENTFUNC( opname, ctypea, cha, ctypex, chx, ctypey, chy, ctypec, chc ) \
UNIT_TEST(cha,chx,chy,chc,opname) \
( \
	constexpr auto M = 4; \
	constexpr auto N = 4; \
\
	for ( auto conjx : { BLIS_CONJUGATE, BLIS_NO_CONJUGATE } ) \
	for ( auto a : test_values<ctypea>() ) \
	for ( auto x : test_values<ctypex>() ) \
	{ \
		auto xmn = tile<M,N>( x ); \
		auto ymn = tile<M,N,ctypey>(); \
\
		INFO( "row-major" ); \
\
		auto ymn0 = ymn; \
		axpbys_mxn<ctypec,BLIS_NO_TRANSPOSE>( a, bli_is_conj( conjx ) \
		                                         ? conj( xmn ) \
		                                         : xmn, \
		                                      0.0, ymn0, dense ); \
\
		INFO( "conjx: " << bli_is_conj( conjx ) ); \
		INFO( "a:     " << a ); \
		INFO( "x:\n" << xmn ); \
\
		bli_tscal2ris_mxn( cha,chx,chy,chc, conjx, \
		                   M, N, &a, \
		                   &xmn[0][0], N, 1, \
		                   &ymn[0][0], 2*N, 2, 1 ); \
\
		INFO( "y (C++):\n" << ymn0 ); \
		INFO( "y (BLIS):\n" << ymn ); \
\
		check<ctypec>( ymn, ymn0 ); \
	} \
\
	for ( auto conjx : { BLIS_CONJUGATE, BLIS_NO_CONJUGATE } ) \
	for ( auto a : test_values<ctypea>() ) \
	for ( auto x : test_values<ctypex>() ) \
	{ \
		auto xmn = tile<M,N>( x ); \
		auto ymn = tile<M,N,ctypey>(); \
\
		INFO( "column-major" ); \
\
		auto ymn0 = ymn; \
		axpbys_mxn<ctypec,BLIS_TRANSPOSE>( a, bli_is_conj( conjx ) \
		                                      ? conj( xmn ) \
		                                      : xmn, \
		                                   0.0, ymn0, dense ); \
\
		INFO( "conjx: " << bli_is_conj( conjx ) ); \
		INFO( "a:     " << a ); \
		INFO( "x:\n" << xmn ); \
\
		bli_tscal2ris_mxn( cha,chx,chy,chc, \
		                   conjx, N, M, &a, \
		                   &xmn[0][0], 1, N, \
		                   &ymn[0][0], 2, 2*N, 1 ); \
\
		INFO( "y (C++):\n" << ymn0 ); \
		INFO( "y (BLIS):\n" << ymn ); \
\
		check<ctypec>( ymn, ymn0 ); \
	} \
)

INSERT_GENTFUNC_MIX4( RC, C, C, C, scal2ris_mxn_1 )

#undef GENTFUNC
#define GENTFUNC( opname, ctypea, cha, ctypex, chx, ctypey, chy, ctypec, chc ) \
UNIT_TEST(cha,chx,chy,chc,opname) \
( \
	constexpr auto M = 4; \
	constexpr auto N = 4; \
\
	using ctypeyr = make_real_t<ctypey>; \
\
	for ( auto conjx : { BLIS_CONJUGATE, BLIS_NO_CONJUGATE } ) \
	for ( auto a : test_values<ctypea>() ) \
	for ( auto x : test_values<ctypex>() ) \
	{ \
		auto xmn = tile<M,N>( x ); \
		auto yrmn = tile<M,N,ctypeyr>(); \
		auto yimn = tile<M,N,ctypeyr>(); \
\
		INFO( "row-major" ); \
\
		auto ymn0 = tile<M,N,ctypey>(); \
		axpbys_mxn<ctypec,BLIS_NO_TRANSPOSE>( a, bli_is_conj( conjx ) \
		                                         ? conj( xmn ) \
		                                         : xmn, \
		                                      0.0, ymn0, dense ); \
		auto yrmn0 = real( ymn0 ); \
		auto yimn0 = imag( ymn0 ); \
\
		INFO( "conjx: " << bli_is_conj( conjx ) ); \
		INFO( "a:     " << a ); \
		INFO( "x:\n" << xmn ); \
\
		bli_tscal2ris_mxn( cha,chx,chy,chc, \
		                   conjx, M, N, &a, \
		                   &xmn[0][0], N, 1, \
		                   &yrmn[0][0], N, 1, \
		                   &yimn[0][0] - &yrmn[0][0] ); \
\
		INFO( "yr (C++):\n" << yrmn0 ); \
		INFO( "yi (C++):\n" << yimn0 ); \
		INFO( "yr (BLIS):\n" << yrmn ); \
		INFO( "yi (BLIS):\n" << yimn ); \
\
		check<ctypec>( yrmn, yrmn0 ); \
		check<ctypec>( yimn, yimn0 ); \
	} \
\
	for ( auto conjx : { BLIS_CONJUGATE, BLIS_NO_CONJUGATE } ) \
	for ( auto a : test_values<ctypea>() ) \
	for ( auto x : test_values<ctypex>() ) \
	{ \
		auto xmn = tile<M,N>( x ); \
		auto yrmn = tile<M,N,ctypeyr>(); \
		auto yimn = tile<M,N,ctypeyr>(); \
\
		INFO( "column-major" ); \
\
		auto ymn0 = tile<M,N,ctypey>(); \
		axpbys_mxn<ctypec,BLIS_TRANSPOSE>( a, bli_is_conj( conjx ) \
		                                      ? conj( xmn ) \
		                                      : xmn, \
		                                   0.0, ymn0, dense ); \
		auto yrmn0 = real( ymn0 ); \
		auto yimn0 = imag( ymn0 ); \
\
		INFO( "conjx: " << bli_is_conj( conjx ) ); \
		INFO( "a:     " << a ); \
		INFO( "x:\n" << xmn ); \
\
		bli_tscal2ris_mxn( cha,chx,chy,chc, \
		                   conjx, N, M, &a, \
		                   &xmn[0][0], 1, N, \
		                   &yrmn[0][0], 1, N, \
		                   &yimn[0][0] - &yrmn[0][0] ); \
\
		INFO( "yr (C++):\n" << yrmn0 ); \
		INFO( "yi (C++):\n" << yimn0 ); \
		INFO( "yr (BLIS):\n" << yrmn ); \
		INFO( "yi (BLIS):\n" << yimn ); \
\
		check<ctypec>( yrmn, yrmn0 ); \
		check<ctypec>( yimn, yimn0 ); \
	} \
)

INSERT_GENTFUNC_MIX4( RC, C, C, C, scal2ris_mxn_k )

/******************************************************************************
 *
 * scals
 *
 *****************************************************************************/

#undef GENTFUNC
#define GENTFUNC( opname, ctypea, cha, ctypex, chx, ctypec, chc ) \
UNIT_TEST(cha,chx,chc,opname) \
( \
	for ( auto a : test_values<ctypea>() ) \
	for ( auto x : test_values<ctypex>() ) \
	{ \
		auto y0 = convert<ctypex>( convert_prec<ctypec>( a ) * \
		                           convert_prec<ctypec>( x ) ); \
\
		INFO( "a:        " << a ); \
		INFO( "x:        " << x ); \
\
		ctypex y = x; \
		bli_tscals( cha,chx,chc, a, y ); \
\
		INFO( "y (C++):  " << y0 ); \
		INFO( "y (BLIS): " << y ); \
\
		check<ctypec>( y, y0 ); \
	} \
)

INSERT_GENTFUNC_MIX3( RC, RC, C, scals )

#undef GENTFUNC
#define GENTFUNC( opname, ctypea, cha, ctypex, chx, ctypec, chc ) \
UNIT_TEST(cha,chx,chc,opname) \
( \
	for ( auto a : test_values<ctypea>() ) \
	for ( auto x : test_values<ctypex>() ) \
	{ \
		auto y0 = convert<ctypex>( convert_prec<ctypec>( conj( a ) ) * \
		                           convert_prec<ctypec>( x ) ); \
\
		INFO( "a:        " << a ); \
		INFO( "x:        " << x ); \
\
		ctypex y = x; \
		bli_tscaljs( cha,chx,chc, a, y ); \
\
		INFO( "y (C++):  " << y0 ); \
		INFO( "y (BLIS): " << y ); \
\
		check<ctypec>( y, y0 ); \
	} \
)

INSERT_GENTFUNC_MIX3( RC, RC, C, scaljs )

#undef GENTFUNC
#define GENTFUNC( opname, ctypea, cha, ctypex, chx, ctypec, chc ) \
UNIT_TEST(cha,chx,chc,opname) \
( \
	for ( auto a : test_values<ctypea>() ) \
	for ( auto x : test_values<ctypex>() ) \
	{ \
		auto y0 = convert<ctypex>( convert_prec<ctypec>( a ) * \
		                           convert_prec<ctypec>( x ) ); \
\
		INFO( "a:        " << a ); \
		INFO( "x:        " << x ); \
\
		ctypex y = x; \
		bli_tscalris( cha,chx,chc, \
		              real( a ), imag( a ), \
		              real( y ), imag( y ) ); \
\
		INFO( "y (C++):  " << y0 ); \
		INFO( "y (BLIS): " << y ); \
\
		check<ctypec>( y, y0 ); \
	} \
)

INSERT_GENTFUNC_MIX3( RC, RC, C, scalris )

#undef GENTFUNC
#define GENTFUNC( opname, ctypea, cha, ctypex, chx, ctypec, chc ) \
UNIT_TEST(cha,chx,chc,opname) \
( \
	for ( auto a : test_values<ctypea>() ) \
	for ( auto x : test_values<ctypex>() ) \
	{ \
		auto y0 = convert<ctypex>( convert_prec<ctypec>( conj( a ) ) * \
		                           convert_prec<ctypec>( x ) ); \
\
		INFO( "a:        " << a ); \
		INFO( "x:        " << x ); \
\
		ctypex y = x; \
		bli_tscaljris( cha,chx,chc, \
		               real( a ), imag( a ), \
		               real( y ), imag( y ) ); \
\
		INFO( "y (C++):  " << y0 ); \
		INFO( "y (BLIS): " << y ); \
\
		check<ctypec>( y, y0 ); \
	} \
)

INSERT_GENTFUNC_MIX3( RC, RC, C, scaljris )

// xpbys_mxn_uplo unit test
#undef GENTFUNC
#define GENTFUNC( opname, ctypea, cha, ctypex, chx, ctypec, chc ) \
UNIT_TEST(cha,chx,chc,opname) \
( \
	constexpr auto M = 4; \
	constexpr auto N = 4; \
\
	for ( uplo_t uplo : { BLIS_UPPER, BLIS_LOWER } ) \
	for ( auto a : test_values<ctypea>() ) \
	for ( auto x : test_values<ctypex>() ) \
	for ( auto diagoff : { -1, 0, 1 } ) \
	{ \
		auto xmn = tile<M,N>( x ); \
\
		INFO( "row-major" ); \
\
		std::function<bool(size_t,size_t)> func = is_below( diagoff ); \
		if ( uplo == BLIS_UPPER ) func = is_above( diagoff ); \
\
		auto xmn0 = xmn; \
		axpbys_mxn<ctypec,BLIS_NO_TRANSPOSE>( a, xmn, 0.0, xmn0, func ); \
\
		INFO( "upper:   " << ( uplo == BLIS_UPPER ) ); \
		INFO( "diagoff: " << diagoff ); \
		INFO( "a: " << a ); \
		INFO( "x (init):\n" << xmn ); \
\
		bli_tscalris_mxn_uplo( cha,chx,chc, uplo, diagoff, M, N, \
		                       &real( a ), &real( a )+1, \
		                       &real( xmn[0][0] ), &real( xmn[0][0] )+1, \
		                       &real( xmn[1][0] ) - &real( xmn[0][0] ), \
		                       &real( xmn[0][1] ) - &real( xmn[0][0] ) ); \
\
		INFO( "x (C++):\n" << xmn0 ); \
		INFO( "x (BLIS):\n" << xmn ); \
\
		check<ctypec>( xmn, xmn0 ); \
	} \
\
	for ( uplo_t uplo : { BLIS_UPPER, BLIS_LOWER } ) \
	for ( auto a : test_values<ctypea>() ) \
	for ( auto x : test_values<ctypex>() ) \
	for ( auto diagoff : { -1, 0, 1 } ) \
	{ \
		auto xmn = tile<M,N>( x ); \
\
		INFO( "column-major" ); \
\
		std::function<bool(size_t,size_t)> func = is_below( diagoff ); \
		if ( uplo == BLIS_UPPER ) func = is_above( diagoff ); \
\
		auto xmn0 = xmn; \
		axpbys_mxn<ctypec,BLIS_TRANSPOSE>( a, xmn, 0.0, xmn0, func ); \
\
		INFO( "upper:   " << ( uplo == BLIS_UPPER ) ); \
		INFO( "diagoff: " << diagoff ); \
		INFO( "a: " << a ); \
		INFO( "x (init):\n" << xmn ); \
\
		bli_tscalris_mxn_uplo( cha,chx,chc, uplo, diagoff, N, M, \
		                       &real( a ), &real( a )+1, \
		                       &real( xmn[0][0] ), &real( xmn[0][0] )+1, \
		                       &real( xmn[0][1] ) - &real( xmn[0][0] ), \
		                       &real( xmn[1][0] ) - &real( xmn[0][0] ) ); \
\
		INFO( "x (C++):\n" << xmn0 ); \
		INFO( "x (BLIS):\n" << xmn ); \
\
		check<ctypec>( xmn, xmn0 ); \
	} \
)

INSERT_GENTFUNC_MIX3( RC, RC, C, scalris_mxn_uplo )

/******************************************************************************
 *
 * sets
 *
 *****************************************************************************/

#undef GENTFUNC
#define GENTFUNC( opname, ctypex, chx, ctypey, chy ) \
UNIT_TEST(chx,chy,opname) \
( \
	for ( auto x : test_values<ctypex>() ) \
	{ \
		auto y0 = convert<ctypey>( x ); \
\
		INFO( "x:        " << x ); \
\
		ctypey y; \
		bli_tsets( chx,chy, real( x ), imag( x ), y ); \
\
		INFO( "y (C++):  " << y0 ); \
		INFO( "y (BLIS): " << y ); \
\
		check<ctypey>( y, y0 ); \
	} \
)

INSERT_GENTFUNC_MIX2( RC, RC, sets )

#undef GENTFUNC
#define GENTFUNC( opname, ctypex, chx, ctypey, chy ) \
UNIT_TEST(chx,chy,opname) \
( \
	for ( auto x : test_values<ctypex>() ) \
	for ( auto y : test_values<ctypey>() ) \
	{ \
		auto y0 = y; \
		real( y0 ) = convert_prec<ctypey>( real( x ) ); \
\
		INFO( "x:        " << x ); \
\
		bli_tsetrs( chx,chy, real( x ), y ); \
\
		INFO( "y (C++):  " << y0 ); \
		INFO( "y (BLIS): " << y ); \
\
		check<ctypey>( y, y0 ); \
	} \
)

INSERT_GENTFUNC_MIX2( RC, RC, setrs )

#undef GENTFUNC
#define GENTFUNC( opname, ctypex, chx, ctypey, chy ) \
UNIT_TEST(chx,chy,opname) \
( \
	for ( auto x : test_values<ctypex>() ) \
	for ( auto y : test_values<ctypey>() ) \
	{ \
		auto y0 = y; \
		imag( y0 ) = convert_prec<ctypey>( imag( x ) ); \
\
		INFO( "x:        " << x ); \
\
		bli_tsetis( chx,chy, imag( x ), y ); \
\
		INFO( "y (C++):  " << y0 ); \
		INFO( "y (BLIS): " << y ); \
\
		check<ctypey>( y, y0 ); \
	} \
)

INSERT_GENTFUNC_MIX2( RC, RC, setis )

#undef GENTFUNC
#define GENTFUNC( opname, ctypex, chx, ctypey, chy ) \
UNIT_TEST(chx,chy,opname) \
( \
	for ( auto x : test_values<ctypex>() ) \
	{ \
		auto y0 = convert<ctypey>( x ); \
\
		INFO( "x:        " << x ); \
\
		ctypey y; \
		bli_tsetris( chx,chy, \
		             real( x ), imag( x ), \
		             real( y ), imag( y ) ); \
\
		INFO( "y (C++):  " << y0 ); \
		INFO( "y (BLIS): " << y ); \
\
		check<ctypey>( y, y0 ); \
	} \
)

INSERT_GENTFUNC_MIX2( RC, RC, setris )

#undef GENTFUNC
#define GENTFUNC( opname, ctypey, chy ) \
UNIT_TEST(chy,opname) \
( \
	for ( auto y : test_values<ctypey>() ) \
	{ \
		auto y0 = convert<ctypey>( 0.0 ); \
\
		INFO( "y (init): " << y ); \
\
		bli_tset0s( chy, y ); \
\
		INFO( "y (C++):  " << y0 ); \
		INFO( "y (BLIS): " << y ); \
\
		check<ctypey>( y, y0 ); \
	} \
)

INSERT_GENTFUNC_MIX1( RC, set0s )

#undef GENTFUNC
#define GENTFUNC( opname, ctypey, chy ) \
UNIT_TEST(chy,opname) \
( \
	for ( auto y : test_values<ctypey>() ) \
	{ \
		auto y0 = convert<ctypey>( 1.0 ); \
\
		INFO( "y (init): " << y ); \
\
		bli_tset1s( chy, y ); \
\
		INFO( "y (C++):  " << y0 ); \
		INFO( "y (BLIS): " << y ); \
\
		check<ctypey>( y, y0 ); \
	} \
)

INSERT_GENTFUNC_MIX1( RC, set1s )

#undef GENTFUNC
#define GENTFUNC( opname, ctypey, chy ) \
UNIT_TEST(chy,opname) \
( \
	for ( auto y : test_values<ctypey>() ) \
	{ \
		auto y0 = y; \
		real( y0 ) = convert_prec<ctypey>( 0.0 ); \
\
		INFO( "y (init): " << y ); \
\
		bli_tsetr0s( chy, y ); \
\
		INFO( "y (C++):  " << y0 ); \
		INFO( "y (BLIS): " << y ); \
\
		check<ctypey>( y, y0 ); \
	} \
)

INSERT_GENTFUNC_MIX1( RC, setr0s )

#undef GENTFUNC
#define GENTFUNC( opname, ctypey, chy ) \
UNIT_TEST(chy,opname) \
( \
	for ( auto y : test_values<ctypey>() ) \
	{ \
		auto y0 = y; \
		imag( y0 ) = convert_prec<ctypey>( 0.0 ); \
\
		INFO( "y (init): " << y ); \
\
		bli_tseti0s( chy, y ); \
\
		INFO( "y (C++):  " << y0 ); \
		INFO( "y (BLIS): " << y ); \
\
		check<ctypey>( y, y0 ); \
	} \
)

INSERT_GENTFUNC_MIX1( RC, seti0s )

#undef GENTFUNC
#define GENTFUNC( opname, ctypey, chy ) \
UNIT_TEST(chy,opname) \
( \
	for ( auto y : test_values<ctypey>() ) \
	{ \
		auto y0 = convert<ctypey>( 0.0 ); \
\
		bli_tset0ris( chy, real( y ), imag( y ) ); \
\
		INFO( "y (C++):  " << y0 ); \
		INFO( "y (BLIS): " << y ); \
\
		check<ctypey>( y, y0 ); \
	} \
)

INSERT_GENTFUNC_MIX1( RC, set0ris )

#undef GENTFUNC
#define GENTFUNC( opname, ctypey, chy ) \
UNIT_TEST(chy,opname) \
( \
	constexpr auto M = 4; \
	constexpr auto N = 4; \
  \
	for ( auto y : test_values<ctypey>() ) \
	{ \
		auto ymn = tile<M,N>( y ); \
\
		INFO( "row-major" ); \
\
		auto ymn0 = tile<M,N>( convert<ctypey>( 0.0 ) ); \
\
		INFO( "y (init):\n" << ymn); \
\
		bli_tset0s_mxn( chy, M, N, &ymn[0][0], N, 1 ); \
\
		INFO( "y (C++):\n" << ymn0 ); \
		INFO( "y (BLIS):\n" << ymn ); \
\
		check<ctypey>( ymn, ymn0 ); \
	} \
  \
	for ( auto y : test_values<ctypey>() ) \
	{ \
		auto ymn = tile<M,N>( y ); \
\
		INFO( "column-major" ); \
\
		auto ymn0 = tile<M,N>( convert<ctypey>( 0.0 ) ); \
\
		INFO( "y (init):\n" << ymn ); \
\
		bli_tset0s_mxn( chy, N, M, &ymn[0][0], 1, N ); \
\
		INFO( "y (C++):\n" << ymn0 ); \
		INFO( "y (BLIS):\n" << ymn ); \
\
		check<ctypey>( ymn, ymn0 ); \
	} \
)

INSERT_GENTFUNC_MIX1( RC, set0s_mxn )

#undef GENTFUNC
#define GENTFUNC( opname, ctypey, chy ) \
UNIT_TEST(chy,opname) \
( \
    /* TODO */ \
)

//INSERT_GENTFUNC_MIX1( C, set0bbs_mxn )

#undef GENTFUNC
#define GENTFUNC( opname, ctypea, cha, ctypey, chy ) \
UNIT_TEST(cha,chy,opname) \
( \
    /* TODO */ \
)

//INSERT_GENTFUNC_MIX2( RC, C, set1ms_mxn_diag )

#undef GENTFUNC
#define GENTFUNC( opname, ctypea, cha, ctypey, chy ) \
UNIT_TEST(cha,chy,opname) \
( \
    /* TODO */ \
)

//INSERT_GENTFUNC_MIX2( RC, C, set1ms_mxn_uplo )

#undef GENTFUNC
#define GENTFUNC( opname, ctypea, cha, ctypey, chy ) \
UNIT_TEST(cha,chy,opname) \
( \
    /* TODO */ \
)

//INSERT_GENTFUNC_MIX2( RC, C, set1ms_mxn )

#undef GENTFUNC
#define GENTFUNC( opname, ctypey, chy ) \
UNIT_TEST(chy,opname) \
( \
    /* TODO */ \
)

//INSERT_GENTFUNC_MIX1( C, seti01ms_mxn_diag )

#undef GENTFUNC
#define GENTFUNC( opname, ctypea, cha, ctypey, chy, ctypec, chc ) \
UNIT_TEST(cha,chy,chc,opname) \
( \
    /* TODO */ \
)

//INSERT_GENTFUNC_MIX3( RC, RC, C, setrihs_mxn_diag )

/******************************************************************************
 *
 * sqrt2s
 *
 *****************************************************************************/

// tsqrt2s unit test
#undef GENTFUNC
#define GENTFUNC( opname, ctypex, chx, ctypey, chy, ctypec, chc ) \
UNIT_TEST(chx,chy,chc,opname) \
( \
	for ( auto x : test_values<ctypex>() ) \
	{ \
		auto y0 = convert<ctypey>( square_root( convert_prec<ctypec>( x ) ) ); \
\
		ctypey y; \
		bli_tsqrt2s( chx,chy,chc, x, y ); \
\
		INFO( "x:        " << x ); \
		INFO( "y (C++):  " << y0 ); \
		INFO( "y (BLIS): " << y ); \
\
		check<ctypec>( y, y0 ); \
	} \
)

INSERT_GENTFUNC_MIX3( R, R, R, sqrt2s )

// tsqrt2ris unit test
#undef GENTFUNC
#define GENTFUNC( opname, ctypex, chx, ctypey, chy, ctypec, chc ) \
UNIT_TEST(chx,chy,chc,opname) \
( \
	for ( auto x : test_values<ctypex>() ) \
	{ \
		auto y0 = convert<ctypey>( square_root( convert_prec<ctypec>( x ) ) ); \
\
		ctypey y; \
		bli_tsqrt2ris( chx,chy,chc, \
		               real( x ), imag( x ), \
		               real( y ), imag( y ) ); \
\
		INFO( "x:        " << x ); \
		INFO( "y (C++):  " << y0 ); \
		INFO( "y (BLIS): " << y ); \
\
		check<ctypec>( y, y0 ); \
	} \
)

INSERT_GENTFUNC_MIX3( R, R, R, sqrt2ris )

/******************************************************************************
 *
 * subs
 *
 *****************************************************************************/

// tsubs unit test
#undef GENTFUNC
#define GENTFUNC( opname, ctypex, chx, ctypey, chy, ctypec, chc ) \
UNIT_TEST(chx,chy,chc,opname) \
( \
	for ( auto x : test_values<ctypex>() ) \
	for ( auto y : test_values<ctypey>() ) \
	{ \
		auto y0 = convert<ctypey>( convert_prec<ctypec>( y ) - \
		                           convert_prec<ctypec>( x ) ); \
\
		INFO( "x:        " << x ); \
		INFO( "y (init): " << y ); \
\
		bli_tsubs( chx,chy,chc, x, y ); \
\
		INFO( "y (C++):  " << y0 ); \
		INFO( "y (BLIS): " << y ); \
\
		check<ctypec>( y, y0 ); \
	} \
)

INSERT_GENTFUNC_MIX3( RC, RC, C, subs )

// tsubjs unit test
#undef GENTFUNC
#define GENTFUNC( opname, ctypex, chx, ctypey, chy, ctypec, chc ) \
UNIT_TEST(chx,chy,chc,opname) \
( \
	for ( auto x : test_values<ctypex>() ) \
	for ( auto y : test_values<ctypey>() ) \
	{ \
		auto y0 = convert<ctypey>(       convert_prec<ctypec>( y ) - \
		                           conj( convert_prec<ctypec>( x ) ) ); \
\
		INFO( "x:        " << x ); \
		INFO( "y (init): " << y ); \
\
		bli_tsubjs( chx,chy,chc, x, y ); \
\
		INFO( "y (C++):  " << y0 ); \
		INFO( "y (BLIS): " << y ); \
\
		check<ctypec>( y, y0 ); \
	} \
)

INSERT_GENTFUNC_MIX3( RC, RC, C, subjs )

// tsubris unit test
#undef GENTFUNC
#define GENTFUNC( opname, ctypex, chx, ctypey, chy, ctypec, chc ) \
UNIT_TEST(chx,chy,chc,opname) \
( \
	for ( auto x : test_values<ctypex>() ) \
	for ( auto y : test_values<ctypey>() ) \
	{ \
		auto y0 = convert<ctypey>( convert_prec<ctypec>( y ) - \
		                           convert_prec<ctypec>( x ) ); \
\
		INFO( "x:        " << x ); \
		INFO( "y (init): " << y ); \
\
		bli_tsubris( chx,chy,chc, \
		             real( x ), imag( x ), \
		             real( y ), imag( y ) ); \
\
		INFO( "y (C++):  " << y0 ); \
		INFO( "y (BLIS): " << y ); \
\
		check<ctypec>( y, y0 ); \
	} \
)

INSERT_GENTFUNC_MIX3( RC, RC, C, subris )

// tsubjris unit test
#undef GENTFUNC
#define GENTFUNC( opname, ctypex, chx, ctypey, chy, ctypec, chc ) \
UNIT_TEST(chx,chy,chc,opname) \
( \
	for ( auto x : test_values<ctypex>() ) \
	for ( auto y : test_values<ctypey>() ) \
	{ \
		auto y0 = convert<ctypey>(       convert_prec<ctypec>( y ) - \
		                           conj( convert_prec<ctypec>( x ) ) ); \
\
		INFO( "x:        " << x ); \
		INFO( "y (init): " << y ); \
\
		bli_tsubjris( chx,chy,chc, \
		              real( x ), imag( x ), \
		              real( y ), imag( y ) ); \
\
		INFO( "y (C++):  " << y0 ); \
		INFO( "y (BLIS): " << y ); \
\
		check<ctypec>( y, y0 ); \
	} \
)

INSERT_GENTFUNC_MIX3( RC, RC, C, subjris )

/******************************************************************************
 *
 * swaps
 *
 *****************************************************************************/

// tswaps unit test
#undef GENTFUNC
#define GENTFUNC( opname, ctypex, chx, ctypey, chy ) \
UNIT_TEST(chx,chy,opname) \
( \
	for ( auto x : test_values<ctypex>() ) \
	for ( auto y : test_values<ctypey>() ) \
	{ \
		auto x0 = convert<ctypex>( y ); \
		auto y0 = convert<ctypey>( x ); \
\
		INFO( "x (init): " << x ); \
		INFO( "y (init): " << y ); \
\
		bli_tswaps( chx,chy, x, y ); \
\
		INFO( "x (C++):  " << x0 ); \
		INFO( "y (C++):  " << y0 ); \
		INFO( "x (BLIS): " << x ); \
		INFO( "y (BLIS): " << y ); \
\
		check<ctypex>( x, x0 ); \
		check<ctypey>( y, y0 ); \
	} \
)

INSERT_GENTFUNC_MIX2( RC, RC, swaps )

// tswapris unit test
#undef GENTFUNC
#define GENTFUNC( opname, ctypex, chx, ctypey, chy ) \
UNIT_TEST(chx,chy,opname) \
( \
	for ( auto x : test_values<ctypex>() ) \
	for ( auto y : test_values<ctypey>() ) \
	{ \
		auto x0 = convert<ctypex>( y ); \
		auto y0 = convert<ctypey>( x ); \
\
		INFO( "x (init): " << x ); \
		INFO( "y (init): " << y ); \
\
		bli_tswapris( chx,chy, \
		              real( x ), imag( x ), \
		              real( y ), imag( y ) ); \
\
		INFO( "x (C++):  " << x0 ); \
		INFO( "y (C++):  " << y0 ); \
		INFO( "x (BLIS): " << x ); \
		INFO( "y (BLIS): " << y ); \
\
		check<ctypex>( x, x0 ); \
		check<ctypey>( y, y0 ); \
	} \
)

INSERT_GENTFUNC_MIX2( RC, RC, swapris )

/******************************************************************************
 *
 * xpbys
 *
 *****************************************************************************/

// txpbys unit test
#undef GENTFUNC
#define GENTFUNC( opname, ctypex, chx, ctypeb, chb, ctypey, chy, ctypec, chc ) \
UNIT_TEST(chx,chb,chy,chc,opname) \
( \
	for ( auto x : test_values<ctypex>() ) \
	for ( auto b : test_values<ctypeb>() ) \
	for ( auto y : test_values<ctypey>() ) \
	{ \
		auto y0 = convert<ctypey>( convert_prec<ctypec>( x ) + \
		                           convert_prec<ctypec>( b ) * \
		                           convert_prec<ctypec>( y ) ); \
\
		INFO( "x:        " << x ); \
		INFO( "b:        " << b ); \
		INFO( "y (init): " << y ); \
\
		bli_txpbys( chx,chb,chy,chc, x, b, y ); \
\
		INFO( "y (C++):  " << y0 ); \
		INFO( "y (BLIS): " << y ); \
\
		check<ctypec>( y, y0 ); \
	} \
)

INSERT_GENTFUNC_MIX4( RC, RC, RC, C, xpbys )

// txpbyjs unit test
#undef GENTFUNC
#define GENTFUNC( opname, ctypex, chx, ctypeb, chb, ctypey, chy, ctypec, chc ) \
UNIT_TEST(chx,chb,chy,chc,opname) \
( \
	for ( auto x : test_values<ctypex>() ) \
	for ( auto b : test_values<ctypeb>() ) \
	for ( auto y : test_values<ctypey>() ) \
	{ \
		auto y0 = convert<ctypey>( conj( convert_prec<ctypec>( x ) ) + \
		                                 convert_prec<ctypec>( b ) * \
		                                 convert_prec<ctypec>( y ) ); \
\
		INFO( "x:        " << x ); \
		INFO( "b:        " << b ); \
		INFO( "y (init): " << y ); \
\
		bli_txpbyjs( chx,chb,chy,chc, x, b, y ); \
\
		INFO( "y (C++):  " << y0 ); \
		INFO( "y (BLIS): " << y ); \
\
		check<ctypec>( y, y0 ); \
	} \
)

INSERT_GENTFUNC_MIX4( RC, RC, RC, C, xpbyjs )

// txpbyris unit test
#undef GENTFUNC
#define GENTFUNC( opname, ctypex, chx, ctypeb, chb, ctypey, chy, ctypec, chc ) \
UNIT_TEST(chx,chb,chy,chc,opname) \
( \
	for ( auto x : test_values<ctypex>() ) \
	for ( auto b : test_values<ctypeb>() ) \
	for ( auto y : test_values<ctypey>() ) \
	{ \
		auto y0 = convert<ctypey>( convert_prec<ctypec>( x ) + \
		                           convert_prec<ctypec>( b ) * \
		                           convert_prec<ctypec>( y ) ); \
\
		INFO( "x:        " << x ); \
		INFO( "b:        " << b ); \
		INFO( "y (init): " << y ); \
\
		bli_txpbyris( chx,chb,chy,chc, \
		              real( x ), imag( x ), \
		              real( b ), imag( b ), \
		              real( y ), imag( y ) ); \
\
		INFO( "y (C++):  " << y0 ); \
		INFO( "y (BLIS): " << y ); \
\
		check<ctypec>( y, y0 ); \
	} \
)

INSERT_GENTFUNC_MIX4( RC, RC, RC, C, xpbyris )

// txpbyjris
#undef GENTFUNC
#define GENTFUNC( opname, ctypex, chx, ctypeb, chb, ctypey, chy, ctypec, chc ) \
UNIT_TEST(chx,chb,chy,chc,opname) \
( \
	for ( auto x : test_values<ctypex>() ) \
	for ( auto b : test_values<ctypeb>() ) \
	for ( auto y : test_values<ctypey>() ) \
	{ \
		auto y0 = convert<ctypey>( conj( convert_prec<ctypec>( x ) ) + \
		                                 convert_prec<ctypec>( b ) * \
		                                 convert_prec<ctypec>( y ) ); \
\
		INFO( "x:        " << x ); \
		INFO( "b:        " << b ); \
		INFO( "y (init): " << y ); \
\
		bli_txpbyjris( chx,chb,chy,chc, \
		               real( x ), imag( x ), \
		               real( b ), imag( b ), \
		               real( y ), imag( y ) ); \
\
		INFO( "y (C++):  " << y0 ); \
		INFO( "y (BLIS): " << y ); \
\
		check<ctypec>( y, y0 ); \
	} \
)

INSERT_GENTFUNC_MIX4( RC, RC, RC, C, xpbyjris )

// xpbys_mxn unit test
#undef GENTFUNC
#define GENTFUNC( opname, ctypex, chx, ctypeb, chb, ctypey, chy, ctypec, chc ) \
UNIT_TEST(chx,chb,chy,chc,opname) \
( \
	constexpr auto M = 4; \
	constexpr auto N = 4; \
\
	for ( auto x : test_values<ctypex>() ) \
	for ( auto b : test_values<ctypeb>() ) \
	for ( auto y : test_values<ctypey>() ) \
	{ \
		auto xmn = tile<M,N>( x ); \
		auto ymn = tile<M,N>( y ); \
\
		INFO( "row-major" ); \
\
		auto ymn0 = ymn; \
		axpbys_mxn<ctypec,BLIS_NO_TRANSPOSE>( 1.0, xmn, b, ymn0, dense ); \
\
		INFO( "x:\n" << xmn ); \
		INFO( "b: " << b ); \
		INFO( "y (init):\n" << ymn ); \
\
		bli_txpbys_mxn( chx,chb,chy,chc, M, N, &xmn[0][0], N, 1, &b, &ymn[0][0], N, 1 ); \
\
		INFO( "y (C++):\n" << ymn0 ); \
		INFO( "y (BLIS):\n" << ymn ); \
\
		check<ctypec>( ymn, ymn0 ); \
	} \
\
	for ( auto x : test_values<ctypex>() ) \
	for ( auto b : test_values<ctypeb>() ) \
	for ( auto y : test_values<ctypey>() ) \
	{ \
		auto xmn = tile<M,N>( x ); \
		auto ymn = tile<M,N>( y ); \
\
		INFO( "column-major" ); \
\
		auto ymn0 = ymn; \
		axpbys_mxn<ctypec,BLIS_TRANSPOSE>( 1.0, xmn, b, ymn0, dense ); \
\
		INFO( "x:\n" << xmn ); \
		INFO( "b: " << b ); \
		INFO( "y (init):\n" << ymn ); \
\
		bli_txpbys_mxn( chx,chb,chy,chc, N, M, &xmn[0][0], 1, N, &b, &ymn[0][0], 1, N ); \
		INFO( "y (C++):\n" << ymn0 ); \
		INFO( "y (BLIS):\n" << ymn ); \
\
		check<ctypec>( ymn, ymn0 ); \
	} \
)

INSERT_GENTFUNC_MIX4( RC, RC, RC, C, xpbys_mxn )

// xpbys_mxn_uplo unit test
#undef GENTFUNC
#define GENTFUNC( opname, ctypex, chx, ctypeb, chb, ctypey, chy, ctypec, chc ) \
UNIT_TEST(chx,chb,chy,chc,opname) \
( \
	constexpr auto M = 4; \
	constexpr auto N = 4; \
\
	for ( uplo_t uplo : { BLIS_UPPER, BLIS_LOWER } ) \
	for ( auto x : test_values<ctypex>() ) \
	for ( auto b : test_values<ctypeb>() ) \
	for ( auto y : test_values<ctypey>() ) \
	for ( auto diagoff : { -1, 0, 1 } ) \
	{ \
		auto xmn = tile<M,N>( x ); \
		auto ymn = tile<M,N>( y ); \
\
		INFO( "row-major" ); \
\
		std::function<bool(size_t,size_t)> func = is_below( diagoff ); \
		if ( uplo == BLIS_UPPER ) func = is_above( diagoff ); \
\
		auto ymn0 = ymn; \
		axpbys_mxn<ctypec,BLIS_NO_TRANSPOSE>( 1.0, xmn, b, ymn0, func ); \
\
		INFO( "upper:   " << ( uplo == BLIS_UPPER ) ); \
		INFO( "diagoff: " << diagoff ); \
		INFO( "x:\n" << xmn ); \
		INFO( "b: " << b ); \
		INFO( "y (init):\n" << ymn ); \
\
		bli_txpbys_mxn_uplo( chx,chb,chy,chc, diagoff, uplo, M, N, &xmn[0][0], N, 1, &b, &ymn[0][0], N, 1 ); \
\
		INFO( "y (C++):\n" << ymn0 ); \
		INFO( "y (BLIS):\n" << ymn ); \
\
		check<ctypec>( ymn, ymn0 ); \
	} \
\
	for ( uplo_t uplo : { BLIS_UPPER, BLIS_LOWER } ) \
	for ( auto x : test_values<ctypex>() ) \
	for ( auto b : test_values<ctypeb>() ) \
	for ( auto y : test_values<ctypey>() ) \
	for ( auto diagoff : { -1, 0, 1 } ) \
	{ \
		auto xmn = tile<M,N>( x ); \
		auto ymn = tile<M,N>( y ); \
\
		INFO( "column-major" ); \
\
		std::function<bool(size_t,size_t)> func = is_below( diagoff ); \
		if ( uplo == BLIS_UPPER ) func = is_above( diagoff ); \
\
		auto ymn0 = ymn; \
		axpbys_mxn<ctypec,BLIS_TRANSPOSE>( 1.0, xmn, b, ymn0, func ); \
\
		INFO( "upper:   " << ( uplo == BLIS_UPPER ) ); \
		INFO( "diagoff: " << diagoff ); \
		INFO( "x:\n" << xmn ); \
		INFO( "b: " << b ); \
		INFO( "y (init):\n" << ymn ); \
\
		bli_txpbys_mxn_uplo( chx,chb,chy,chc, diagoff, uplo, N, M, &xmn[0][0], 1, N, &b, &ymn[0][0], 1, N ); \
\
		INFO( "y (C++):\n" << ymn0 ); \
		INFO( "y (BLIS):\n" << ymn ); \
\
		check<ctypec>( ymn, ymn0 ); \
	} \
)

INSERT_GENTFUNC_MIX4( RC, RC, RC, C, xpbys_mxn_uplo )

/******************************************************************************
 *
 * copy1es
 *
 *****************************************************************************/

// tcopy1es unit test
#undef GENTFUNC
#define GENTFUNC( opname, ctypex, chx, ctypey, chy ) \
UNIT_TEST(chx,chy,opname) \
( \
	for ( auto x : test_values<ctypex>() ) \
	{ \
		auto yri0 = convert<ctypey>( x ); \
		auto yir0 = convert<ctypey>( swapri( conj( x ) ) ); \
\
		INFO( "x:        " << x ); \
\
		ctypey yri, yir; \
		bli_tcopy1es( chx,chy, x, yri, yir ); \
\
		INFO( "yri (C++):  " << yri0 ); \
		INFO( "yir (C++):  " << yir0 ); \
		INFO( "yri (BLIS): " << yri ); \
		INFO( "yir (BLIS): " << yir ); \
\
		check<ctypey>( yri, yri0 ); \
		check<ctypey>( yir, yir0 ); \
	} \
)

INSERT_GENTFUNC_MIX2( C, C, copy1es )

// tcopyj1es unit test
#undef GENTFUNC
#define GENTFUNC( opname, ctypex, chx, ctypey, chy ) \
UNIT_TEST(chx,chy,opname) \
( \
	for ( auto x : test_values<ctypex>() ) \
	{ \
		auto yri0 = convert<ctypey>( conj( x ) ); \
		auto yir0 = convert<ctypey>( swapri( x ) ); \
\
		INFO( "x:        " << x ); \
\
		ctypey yri, yir; \
		bli_tcopyj1es( chx,chy, x, yri, yir ); \
\
		INFO( "yri (C++):  " << yri0 ); \
		INFO( "yir (C++):  " << yir0 ); \
		INFO( "yri (BLIS): " << yri ); \
		INFO( "yir (BLIS): " << yir ); \
\
		check<ctypey>( yri, yri0 ); \
		check<ctypey>( yir, yir0 ); \
	} \
)

INSERT_GENTFUNC_MIX2( C, C, copyj1es )

/******************************************************************************
 *
 * invert1es
 *
 *****************************************************************************/

// tinvert1es unit test
#undef GENTFUNC
#define GENTFUNC( opname, ctypex, chx, ctypec, chc ) \
UNIT_TEST(chx,chc,opname) \
( \
	for ( auto x : test_values<ctypex>() ) \
	{ \
		auto xri = x; \
		auto xir = swapri( conj( x ) ); \
\
		auto xri0 = convert<ctypex>( convert_prec<ctypec>( 1.0 ) / \
		                             convert_prec<ctypec>( x ) ); \
		auto xir0 = swapri( conj( xri0 ) ); \
\
		INFO( "x:          " << x ); \
		INFO( "xri (orig): " << xri ); \
		INFO( "xir (orig): " << xir ); \
\
		bli_tinvert1es( chx,chc, xri, xir ); \
\
		INFO( "xri (C++):  " << xri0 ); \
		INFO( "xir (C++):  " << xir0 ); \
		INFO( "xri (BLIS): " << xri ); \
		INFO( "xir (BLIS): " << xir ); \
\
		check<ctypec>( xri, xri0 ); \
		check<ctypec>( xir, xir0 ); \
	} \
)

INSERT_GENTFUNC_MIX2( C, C, invert1es )

/******************************************************************************
 *
 * scal21es
 *
 *****************************************************************************/

// tscal21es unit test
#undef GENTFUNC
#define GENTFUNC( opname, ctypea, cha, ctypex, chx, ctypey, chy, ctypec, chc ) \
UNIT_TEST(cha,chx,chy,chc,opname) \
( \
	for ( auto a : test_values<ctypea>() ) \
	for ( auto x : test_values<ctypex>() ) \
	{ \
		auto yri0 = convert<ctypey>( convert_prec<ctypec>( a ) * \
		                             convert_prec<ctypec>( x ) ); \
		auto yir0 = swapri( conj( yri0 ) ) ); \
\
		INFO( "a:          " << a ); \
		INFO( "x:          " << x ); \
\
		ctypey yri, yir; \
		bli_tscal21es( cha,chx,chy,chc, a, x, yri, yir ); \
\
		INFO( "yri (C++):  " << yri0 ); \
		INFO( "yir (C++):  " << yir0 ); \
		INFO( "yri (BLIS): " << yri ); \
		INFO( "yir (BLIS): " << yir ); \
\
		check<ctypec>( yri, yri0 ); \
		check<ctypec>( yir, yir0 ); \
	} \
)

INSERT_GENTFUNC_MIX4( RC, C, C, C, scal21es )

// tscal2j1es unit test
#undef GENTFUNC
#define GENTFUNC( opname, ctypea, cha, ctypex, chx, ctypey, chy, ctypec, chc ) \
UNIT_TEST(cha,chx,chy,chc,opname) \
( \
	for ( auto a : test_values<ctypea>() ) \
	for ( auto x : test_values<ctypex>() ) \
	{ \
		auto yri0 = convert<ctypey>( convert_prec<ctypec>( a ) * \
		                             convert_prec<ctypec>( conj( x ) ) ); \
		auto yir0 = swapri( conj( yri0 ) ) ); \
\
		INFO( "a:          " << a ); \
		INFO( "x:          " << x ); \
\
		ctypey yri, yir; \
		bli_tscal2j1es( cha,chx,chy,chc, a, x, yri, yir ); \
\
		INFO( "yri (C++):  " << yri0 ); \
		INFO( "yir (C++):  " << yir0 ); \
		INFO( "yri (BLIS): " << yri ); \
		INFO( "yir (BLIS): " << yir ); \
\
		check<ctypec>( yri, yri0 ); \
		check<ctypec>( yir, yir0 ); \
	} \
)

INSERT_GENTFUNC_MIX4( RC, C, C, C, scal2j1es )

/******************************************************************************
 *
 * scal1es
 *
 *****************************************************************************/

// tscal1es unit test
#undef GENTFUNC
#define GENTFUNC( opname, ctypea, cha, ctypex, chx, ctypec, chc ) \
UNIT_TEST(cha,chx,chc,opname) \
( \
	for ( auto a : test_values<ctypea>() ) \
	for ( auto x : test_values<ctypex>() ) \
	{ \
		auto xri = x; \
		auto xir = swapri( conj( x ) ); \
\
		auto xri0 = convert<ctypey>( convert_prec<ctypec>( a ) * \
		                             convert_prec<ctypec>( x ) ); \
		auto xir0 = swapri( conj( xri0 ) ) ); \
\
		INFO( "a:          " << a ); \
		INFO( "x:          " << x ); \
		INFO( "xri (orig): " << xri ); \
		INFO( "xir (orig): " << xir ); \
\
		bli_tscal1es( cha,chx,chc, a, xri, xir ); \
\
		INFO( "xri (C++):  " << xri0 ); \
		INFO( "xir (C++):  " << xir0 ); \
		INFO( "xri (BLIS): " << xri ); \
		INFO( "xir (BLIS): " << xir ); \
\
		check<ctypec>( xri, xri0 ); \
		check<ctypec>( xir, xir0 ); \
	} \
)

INSERT_GENTFUNC_MIX3( RC, C, C, scal1es )

/******************************************************************************
 *
 * copy1rs
 *
 *****************************************************************************/

// tcopy1rs unit test
#undef GENTFUNC
#define GENTFUNC( opname, ctypex, chx, ctypey, chy ) \
UNIT_TEST(chx,chy,opname) \
( \
	for ( auto x : test_values<ctypex>() ) \
	{ \
		auto y0 = convert<ctypey>( x ); \
\
		INFO( "x:        " << x ); \
\
		ctypey y; \
		bli_tcopy1rs( chx,chy, x, real( y ), imag( y ) ); \
\
		INFO( "y (C++):  " << y0 ); \
		INFO( "y (BLIS): " << y ); \
\
		check<ctypey>( y, y0 ); \
	} \
)

INSERT_GENTFUNC_MIX2( C, C, copy1rs )

/******************************************************************************
 *
 * invert1rs
 *
 *****************************************************************************/

// tinvert1rs unit test
#undef GENTFUNC
#define GENTFUNC( opname, ctypex, chx, ctypec, chc ) \
UNIT_TEST(chx,chc,opname) \
( \
	for ( auto x : test_values<ctypex>() ) \
	{ \
		auto x0 = convert<ctypex>( convert_prec<ctypec>( 1.0 ) / \
		                           convert_prec<ctypec>( x ) ); \
\
		INFO( "x:        " << x ); \
\
		bli_tinvert1rs( chx,chc, real( x ), imag( x ) ); \
\
		INFO( "x (C++):  " << x0 ); \
		INFO( "x (BLIS): " << x ); \
\
		check<ctypec>( x, x0 ); \
	} \
)

INSERT_GENTFUNC_MIX2( C, C, invert1rs )

/******************************************************************************
 *
 * scal21rs
 *
 *****************************************************************************/

// tscal21rs unit test
#undef GENTFUNC
#define GENTFUNC( opname, ctypea, cha, ctypex, chx, ctypey, chy, ctypec, chc ) \
UNIT_TEST(cha,chx,chy,chc,opname) \
( \
	for ( auto a : test_values<ctypea>() ) \
	for ( auto x : test_values<ctypex>() ) \
	{ \
		auto xr = real( x ); \
		auto xi = imag( x ); \
\
		auto y0 = convert<ctypey>( convert_prec<ctypec>( a ) * \
		                           convert_prec<ctypec>( x ) ); \
\
		INFO( "a:        " << a ); \
		INFO( "x:        " << x ); \
\
		ctypey y; \
		bli_tscal21rs( cha,chx,chy,chc, a, x, real( y ), imag( y ) ); \
\
		INFO( "y (C++):  " << y0 ); \
		INFO( "y (BLIS): " << y ); \
\
		check<ctypec>( yri, yri0 ); \
		check<ctypec>( yir, yir0 ); \
	} \
)

INSERT_GENTFUNC_MIX4( RC, C, C, C, scal21rs )

// tscal2j1rs unit test
#undef GENTFUNC
#define GENTFUNC( opname, ctypea, cha, ctypex, chx, ctypey, chy, ctypec, chc ) \
UNIT_TEST(cha,chx,chy,chc,opname) \
( \
	for ( auto a : test_values<ctypea>() ) \
	for ( auto x : test_values<ctypex>() ) \
	{ \
		auto xr = real( x ); \
		auto xi = imag( x ); \
\
		auto y0 = convert<ctypey>( convert_prec<ctypec>( a ) * \
		                           convert_prec<ctypec>( conj( x ) ) ); \
\
		INFO( "a:        " << a ); \
		INFO( "x:        " << x ); \
\
		ctypey y; \
		bli_tscal2j1rs( cha,chx,chy,chc, a, x, real( y ), imag( y ) ); \
\
		INFO( "y (C++):  " << y0 ); \
		INFO( "y (BLIS): " << y ); \
\
		check<ctypec>( yri, yri0 ); \
		check<ctypec>( yir, yir0 ); \
	} \
)

INSERT_GENTFUNC_MIX4( RC, C, C, C, scal2j1rs )

/******************************************************************************
 *
 * scal1rs
 *
 *****************************************************************************/

// tscal1rs unit test
#undef GENTFUNC
#define GENTFUNC( opname, ctypea, cha, ctypex, chx, ctypec, chc ) \
UNIT_TEST(cha,chx,chc,opname) \
( \
	for ( auto a : test_values<ctypea>() ) \
	for ( auto x : test_values<ctypex>() ) \
	{ \
		auto x0 = convert<ctypey>( convert_prec<ctypec>( a ) * \
		                           convert_prec<ctypec>( x ) ); \
\
		INFO( "a:        " << a ); \
		INFO( "x (orig): " << x ); \
\
		bli_tscal1rs( cha,chx,chc, a, real( x ), imag( x ) ); \
\
		INFO( "x (C++):  " << x0 ); \
		INFO( "xr(BLIS): " << x ); \
\
		check<ctypec>( x, x0 ); \
	} \
)

INSERT_GENTFUNC_MIX3( RC, C, C, scal1rs )
