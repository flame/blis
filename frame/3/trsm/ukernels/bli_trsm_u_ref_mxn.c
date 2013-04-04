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
                           ctype* restrict a, \
                           ctype* restrict b, \
                           ctype* restrict bd, \
                           ctype* restrict c, inc_t rs_c, inc_t cs_c \
                         ) \
{ \
	const dim_t     MR    = PASTEMAC2(ch,varname,_mr); \
	const dim_t     NR    = PASTEMAC2(ch,varname,_nr); \
\
	const dim_t     m     = MR; \
	const dim_t     n     = NR; \
\
	const inc_t     rs_a  = 1; \
	const inc_t     cs_a  = MR; \
\
	const inc_t     rs_b  = NR; \
	const inc_t     cs_b  = 1; \
\
	dim_t           iter, i, j, k; \
	dim_t           n_behind; \
\
	ctype* restrict alpha11; \
	ctype* restrict a12t; \
	ctype* restrict alpha12; \
	ctype* restrict X2; \
	ctype* restrict x1; \
	ctype* restrict x21; \
	ctype* restrict chi21; \
	ctype* restrict chi11; \
	ctype* restrict gamma11; \
	ctype           rho11; \
\
	for ( iter = 0; iter < m; ++iter ) \
	{ \
		i        = m - iter - 1; \
		n_behind = iter; \
		alpha11  = a + (i  )*rs_a + (i  )*cs_a; \
		a12t     = a + (i  )*rs_a + (i+1)*cs_a; \
		x1       = b + (i  )*rs_b + (0  )*cs_b; \
		X2       = b + (i+1)*rs_b + (0  )*cs_b; \
\
		/* x1 = x1 - a12t * X2; */ \
		/* x1 = x1 / alpha11; */ \
		for ( j = 0; j < n; ++j ) \
		{ \
			chi11   = x1 + (0  )*rs_b + (j  )*cs_b; \
			x21     = X2 + (0  )*rs_b + (j  )*cs_b; \
			gamma11 = c  + (i  )*rs_c + (j  )*cs_c; \
\
			/* chi11 = chi11 - a12t * x21; */ \
			PASTEMAC(ch,set0s)( rho11 ); \
			for ( k = 0; k < n_behind; ++k ) \
			{ \
				alpha12 = a12t + (k  )*cs_a; \
				chi21   = x21  + (k  )*rs_b; \
\
				PASTEMAC(ch,axpys)( *alpha12, *chi21, rho11 ); \
			} \
			PASTEMAC(ch,subs)( rho11, *chi11 ); \
\
			/* chi11 = chi11 / alpha11; */ \
			/* NOTE: 1.0/alpha11 is stored instead of alpha11, so we
			   need to multiply rather than divide. */ \
			PASTEMAC(ch,scals)( *alpha11, *chi11 ); \
\
			/* Output final result to matrix C. */ \
			PASTEMAC(ch,copys)( *chi11, *gamma11 ); \
		} \
	} \
}

INSERT_GENTFUNC_BASIC( trsm_u_ref_mxn, trsm_u_ref_mxn )

