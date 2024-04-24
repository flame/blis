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

#ifndef BLIS_SCAL2S_MXN_H
#define BLIS_SCAL2S_MXN_H

// scal2s_mxn

// Notes:
// - The first char encodes the type of a.
// - The second char encodes the type of x.
// - The third char encodes the type of y.
// - We only implement cases where typeof(a) == type(x).

#undef  BLIS_ENABLE_CR_CASES
#define BLIS_ENABLE_CR_CASES 0

// -- bli_???scal2s_mxn --

#undef  GENTFUNC2
#define GENTFUNC2( ctypex, ctypey, chx, chy, opname, kername ) \
\
BLIS_INLINE void PASTEMAC(chx,chx,chy,opname) \
     ( \
       const conj_t  conjx, \
       const dim_t   m, \
       const dim_t   n, \
       const ctypex* alpha, \
       const ctypex* x, inc_t rs_x, inc_t cs_x, \
             ctypey* y, inc_t rs_y, inc_t cs_y  \
     ) \
{ \
	if ( bli_is_conj( conjx ) ) \
	{ \
		for ( dim_t j = 0; j < n; ++j ) \
		{ \
			const ctypex* restrict xj = x + j*cs_x; \
			      ctypey* restrict yj = y + j*cs_y; \
\
			for ( dim_t i = 0; i < m; ++i ) \
			{ \
				const ctypex* restrict xij = xj + i*rs_x; \
				      ctypey* restrict yij = yj + i*rs_y; \
\
				PASTEMAC(chx,chx,chy,scal2js)( *alpha, *xij, *yij ); \
			} \
		} \
	} \
	else /* if ( bli_is_noconj( conjx ) ) */ \
	{ \
		for ( dim_t j = 0; j < n; ++j ) \
		{ \
			const ctypex* restrict xj = x + j*cs_x; \
			      ctypey* restrict yj = y + j*cs_y; \
\
			for ( dim_t i = 0; i < m; ++i ) \
			{ \
				const ctypex* restrict xij = xj + i*rs_x; \
				      ctypey* restrict yij = yj + i*rs_y; \
\
				PASTEMAC(chx,chx,chy,scal2s)( *alpha, *xij, *yij ); \
			} \
		} \
	} \
}

INSERT_GENTFUNC2_BASIC ( scal2s_mxn, scal2s )
INSERT_GENTFUNC2_MIX_DP( scal2s_mxn, scal2s )


// -- bli_?scal2s_mxn --

#undef  GENTFUNC
#define GENTFUNC( ctype, ch, opname ) \
\
BLIS_INLINE void PASTEMAC(ch,opname) \
     ( \
       const conj_t conjx, \
       const dim_t  m, \
       const dim_t  n, \
       const ctype* alpha, \
       const ctype* x, inc_t rs_x, inc_t cs_x, \
             ctype* y, inc_t rs_y, inc_t cs_y  \
     ) \
{ \
	PASTEMAC(ch,ch,ch,opname)( conjx, m, n, alpha, x, rs_x, cs_x, y, rs_y, cs_y ); \
}

INSERT_GENTFUNC_BASIC( scal2s_mxn )

#endif
