/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2014, The University of Texas at Austin

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

#ifndef BLIS_AXPBYS_MXN_H
#define BLIS_AXPBYS_MXN_H

// axpbys_mxn

// Notes:
// - The first char encodes the type of a.
// - The second char encodes the type of x.
// - The third char encodes the type of b.
// - The fourth char encodes the type of y.
// - We only implement cases where typeof(a) == type(x) && typeof(b) == typeof(y).

#undef  BLIS_ENABLE_CR_CASES
#define BLIS_ENABLE_CR_CASES 0

// -- bli_????axpbys_mxn --

#undef  GENTFUNC2
#define GENTFUNC2( ctypex, ctypey, chx, chy, opname, kername ) \
\
BLIS_INLINE void PASTEMAC(chx,chx,chy,chy,opname) \
     ( \
       const dim_t   m, \
       const dim_t   n, \
       const ctypex* alpha, \
       const ctypex* x, inc_t rs_x, inc_t cs_x, \
       const ctypey* beta, \
             ctypey* y, inc_t rs_y, inc_t cs_y  \
     ) \
{ \
	/* If beta is zero, overwrite y with alpha*x (in case y has infs or NaNs). */ \
	if ( PASTEMAC(chy,eq0)( *beta ) ) \
	{ \
		PASTEMAC(chx,chx,chy,scal2s_mxn)( BLIS_NO_CONJUGATE, m, n, alpha, x, rs_x, cs_x, y, rs_y, cs_y ); \
		return; \
	} \
\
	if      ( BLIS_ENABLE_CR_CASES && rs_x == 1 && rs_y == 1 ) \
	{ \
		for ( dim_t jj = 0; jj < n; ++jj ) \
		for ( dim_t ii = 0; ii < m; ++ii ) \
		PASTEMAC(chx,chx,chy,chy,kername) \
		( \
		  *alpha, *(x + ii + jj*cs_x), \
		  *beta,  *(y + ii + jj*cs_y) \
		); \
	} \
	else if ( BLIS_ENABLE_CR_CASES && cs_x == 1 && cs_y == 1 ) \
	{ \
		for ( dim_t ii = 0; ii < m; ++ii ) \
		for ( dim_t jj = 0; jj < n; ++jj ) \
		PASTEMAC(chx,chx,chy,chy,kername) \
		( \
		  *alpha, *(x + ii*rs_x + jj), \
		  *beta,  *(y + ii*rs_y + jj) \
		); \
	} \
	else \
	{ \
		for ( dim_t jj = 0; jj < n; ++jj ) \
		for ( dim_t ii = 0; ii < m; ++ii ) \
		PASTEMAC(chx,chx,chy,chy,kername) \
		( \
		  *alpha, *(x + ii*rs_x + jj*cs_x), \
		  *beta,  *(y + ii*rs_y + jj*cs_y) \
		); \
	} \
}

INSERT_GENTFUNC2_BASIC ( axpbys_mxn, axpbys )
INSERT_GENTFUNC2_MIX_DP( axpbys_mxn, axpbys )


// -- bli_?axpbys_mxn --

#undef  GENTFUNC
#define GENTFUNC( ctype, ch, opname ) \
\
BLIS_INLINE void PASTEMAC(ch,opname) \
     ( \
       const dim_t  m, \
       const dim_t  n, \
       const ctype* alpha, \
       const ctype* x, inc_t rs_x, inc_t cs_x, \
       const ctype* beta, \
             ctype* y, inc_t rs_y, inc_t cs_y  \
     ) \
{ \
    PASTEMAC(ch,ch,ch,ch,opname)( m, n, alpha, x, rs_x, cs_x, beta, y, rs_y, cs_y ); \
}

INSERT_GENTFUNC_BASIC( axpbys_mxn )


#endif
