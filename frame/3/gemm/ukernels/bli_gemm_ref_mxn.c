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
                           ctype* restrict c, inc_t rs_c, inc_t cs_c \
                         ) \
{ \
	const dim_t     m     = PASTEMAC(ch,mr); \
	const dim_t     n     = PASTEMAC(ch,nr); \
\
	const inc_t     cs_a  = PASTEMAC(ch,packmr); \
\
	const inc_t     rs_b  = PASTEMAC(ch,packnr); \
\
	const inc_t     rs_ab = 1; \
	const inc_t     cs_ab = PASTEMAC(ch,mr); \
\
	dim_t           k0, j0, i0; \
\
	ctype           ab[ PASTEMAC(ch,mr) * \
	                    PASTEMAC(ch,nr) ]; \
	ctype* restrict ab00; \
	ctype           a0; \
	ctype           b0; \
\
\
	/* Initialize the accumulator elements in ab to zero. */ \
	for ( i0 = 0; i0 < m * n; ++i0 ) \
	{ \
		PASTEMAC(ch,set0s)( *(ab + i0) ); \
	} \
\
	/* Perform a series of k rank-1 updates into ab. */ \
	for ( k0 = 0; k0 < k; ++k0 ) \
	{ \
		ab00 = ab; \
\
		for ( j0 = 0; j0 < n; ++j0 ) \
		{ \
			b0 = *(b + j0); \
\
			for ( i0 = 0; i0 < m; ++i0 ) \
			{ \
				a0 = *(a + i0); \
\
				PASTEMAC(ch,dots)( a0, \
				                   b0, \
				                   *ab00 ); \
				ab00 += rs_ab; \
			} \
		} \
\
		a += cs_a; \
		b += rs_b; \
	} \
\
	/* Scale the result in ab by alpha. */ \
	for ( i0 = 0; i0 < m * n; ++i0 ) \
	{ \
		PASTEMAC(ch,scals)( *alpha, *(ab + i0) ); \
	} \
\
	/* If beta is zero, overwrite c with the scaled result in ab. Otherwise,
	   scale by beta and then add the scaled redult in ab. */ \
	if ( PASTEMAC(ch,eq0)( *beta ) ) \
	{ \
		PASTEMAC(ch,copys_mxn)( m, \
		                        n, \
		                        ab, rs_ab, cs_ab, \
		                        c,  rs_c,  cs_c ); \
	} \
	else \
	{ \
		PASTEMAC(ch,xpbys_mxn)( m, \
		                        n, \
		                        ab, rs_ab, cs_ab, \
		                        beta, \
		                        c,  rs_c,  cs_c ); \
	} \
}

INSERT_GENTFUNC_BASIC( gemm_ref_mxn, gemm_ref_mxn )

