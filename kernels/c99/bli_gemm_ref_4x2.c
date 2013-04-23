/*

   BLIS    
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2013, The University of Texas

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


#undef  GENTFUNC
#define GENTFUNC( ctype, ch, varname, kername ) \
\
void PASTEMAC(ch,varname)( \
                           dim_t           k, \
                           ctype* restrict alpha, \
                           ctype* restrict a, \
                           ctype* restrict b, \
                           ctype* restrict beta, \
                           ctype* restrict c, inc_t rs_c, inc_t cs_c, \
                           ctype* restrict a_next, \
                           ctype* restrict b_next  \
                         ) \
{ \
	ctype  a0; \
	ctype  a1; \
	ctype  a2; \
	ctype  a3; \
\
	ctype  b0,   b1; \
\
	ctype  ab00, ab01; \
	ctype  ab10, ab11; \
	ctype  ab20, ab21; \
	ctype  ab30, ab31; \
\
	ctype* c00, * c01; \
	ctype* c10, * c11; \
	ctype* c20, * c21; \
	ctype* c30, * c31; \
\
	dim_t  i; \
\
\
	c00 = (c + 0*rs_c + 0*cs_c); \
	c10 = (c + 1*rs_c + 0*cs_c); \
	c20 = (c + 2*rs_c + 0*cs_c); \
	c30 = (c + 3*rs_c + 0*cs_c); \
\
	c01 = (c + 0*rs_c + 1*cs_c); \
	c11 = (c + 1*rs_c + 1*cs_c); \
	c21 = (c + 2*rs_c + 1*cs_c); \
	c31 = (c + 3*rs_c + 1*cs_c); \
\
	PASTEMAC(ch,set0s)( ab00 ); \
	PASTEMAC(ch,set0s)( ab10 ); \
	PASTEMAC(ch,set0s)( ab20 ); \
	PASTEMAC(ch,set0s)( ab30 ); \
\
	PASTEMAC(ch,set0s)( ab01 ); \
	PASTEMAC(ch,set0s)( ab11 ); \
	PASTEMAC(ch,set0s)( ab21 ); \
	PASTEMAC(ch,set0s)( ab31 ); \
\
	for ( i = 0; i < k; ++i ) \
	{ \
		a0 = *(a + 0); \
		a1 = *(a + 1); \
		a2 = *(a + 2); \
		a3 = *(a + 3); \
\
		b0 = *(b + 0); \
		b1 = *(b + 2); \
\
		PASTEMAC(ch,dots)( a0, b0, ab00 ); \
		PASTEMAC(ch,dots)( a1, b0, ab10 ); \
		PASTEMAC(ch,dots)( a2, b0, ab20 ); \
		PASTEMAC(ch,dots)( a3, b0, ab30 ); \
\
		PASTEMAC(ch,dots)( a0, b1, ab01 ); \
		PASTEMAC(ch,dots)( a1, b1, ab11 ); \
		PASTEMAC(ch,dots)( a2, b1, ab21 ); \
		PASTEMAC(ch,dots)( a3, b1, ab31 ); \
\
		a += 4; \
		b += 4; \
	} \
\
	if ( PASTEMAC(ch,eq0)( *beta ) ) \
	{ \
		PASTEMAC(ch,set0s)( *c00 ); \
		PASTEMAC(ch,set0s)( *c10 ); \
		PASTEMAC(ch,set0s)( *c20 ); \
		PASTEMAC(ch,set0s)( *c30 ); \
\
		PASTEMAC(ch,set0s)( *c01 ); \
		PASTEMAC(ch,set0s)( *c11 ); \
		PASTEMAC(ch,set0s)( *c21 ); \
		PASTEMAC(ch,set0s)( *c31 ); \
	} \
	else \
	{ \
		PASTEMAC(ch,scals)( *beta, *c00 ); \
		PASTEMAC(ch,scals)( *beta, *c10 ); \
		PASTEMAC(ch,scals)( *beta, *c20 ); \
		PASTEMAC(ch,scals)( *beta, *c30 ); \
\
		PASTEMAC(ch,scals)( *beta, *c01 ); \
		PASTEMAC(ch,scals)( *beta, *c11 ); \
		PASTEMAC(ch,scals)( *beta, *c21 ); \
		PASTEMAC(ch,scals)( *beta, *c31 ); \
	} \
\
	PASTEMAC(ch,dots)( *alpha, ab00, *c00 ); \
	PASTEMAC(ch,dots)( *alpha, ab10, *c10 ); \
	PASTEMAC(ch,dots)( *alpha, ab20, *c20 ); \
	PASTEMAC(ch,dots)( *alpha, ab30, *c30 ); \
\
	PASTEMAC(ch,dots)( *alpha, ab01, *c01 ); \
	PASTEMAC(ch,dots)( *alpha, ab11, *c11 ); \
	PASTEMAC(ch,dots)( *alpha, ab21, *c21 ); \
	PASTEMAC(ch,dots)( *alpha, ab31, *c31 ); \
}

INSERT_GENTFUNC_BASIC( gemm_ref_4x2, gemm_ref_4x2 )

