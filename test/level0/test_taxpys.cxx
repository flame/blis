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

#include "test_l0.hpp"

/******************************************************************************
 *
 * axpys
 *
 *****************************************************************************/

#undef GENTFUNC
#define GENTFUNC( opname, ctypea, cha, ctypex, chx, ctypey, chy, ctypec, chc ) \
UNIT_TEST(cha,chx,chy,chc,opname) \
( \
	for ( const auto a : test_values<ctypea>() ) \
	for ( const auto x : test_values<ctypex>() ) \
	for (       auto y : test_values<ctypey>() ) \
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

INSERT_GENTFUNC_MIX4( RC, RC, RC, R, axpys )

#undef GENTFUNC
#define GENTFUNC( opname, ctypea, cha, ctypex, chx, ctypec, chc ) \
UNIT_TEST(cha,chx,chc,opname) \
( \
	for ( const auto a : test_values<ctypea>() ) \
	for (       auto x : test_values<ctypex>() ) \
	{ \
		auto x0 = convert<ctypex>( convert_prec<ctypec>( a ) * \
		                           convert_prec<ctypec>( x ) + \
		                           convert_prec<ctypec>( x ) ); \
\
		INFO( "a:        " << a ); \
		INFO( "x:        " << x ); \
\
		bli_taxpys( cha,chx,chx,chc, a, x, x ); \
\
		INFO( "x (C++):  " << x0 ); \
		INFO( "x (BLIS): " << x ); \
\
		check<ctypec>( x, x0 ); \
	} \
)

INSERT_GENTFUNC_MIX3( RC, RC, R, axpys_inplace )

#undef GENTFUNC
#define GENTFUNC( opname, ctypea, cha, ctypex, chx, ctypey, chy, ctypec, chc ) \
UNIT_TEST(cha,chx,chy,chc,opname) \
( \
	for ( const auto a : test_values<ctypea>() ) \
	for ( const auto x : test_values<ctypex>() ) \
	for (       auto y : test_values<ctypey>() ) \
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

INSERT_GENTFUNC_MIX4( RC, RC, RC, R, axpyjs )

#undef GENTFUNC
#define GENTFUNC( opname, ctypea, cha, ctypex, chx, ctypey, chy, ctypec, chc ) \
UNIT_TEST(cha,chx,chy,chc,opname) \
( \
	for ( const auto a : test_values<ctypea>() ) \
	for ( const auto x : test_values<ctypex>() ) \
	for (       auto y : test_values<ctypey>() ) \
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

INSERT_GENTFUNC_MIX4( RC, RC, RC, R, axpyris )

#undef GENTFUNC
#define GENTFUNC( opname, ctypea, cha, ctypex, chx, ctypey, chy, ctypec, chc ) \
UNIT_TEST(cha,chx,chy,chc,opname) \
( \
	for ( const auto a : test_values<ctypea>() ) \
	for ( const auto x : test_values<ctypex>() ) \
	for (       auto y : test_values<ctypey>() ) \
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

INSERT_GENTFUNC_MIX4( RC, RC, RC, R, axpyjris )
