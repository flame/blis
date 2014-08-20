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
    - Neither the name of The University of Texas at Austin nor the names
      of its contributors may be used to endorse or promote products
      derived from this software without specific prior written permission.

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
#define GENTFUNC( ctype, ch, varname ) \
\
void PASTEMAC(ch,varname)( \
                           ctype* restrict a, \
                           ctype* restrict b, \
                           ctype* restrict c, inc_t rs_c, inc_t cs_c, \
                           auxinfo_t*      data  \
                         ) \
{ \
	const dim_t     m     = PASTEMAC(ch,mr); \
	const dim_t     n     = PASTEMAC(ch,nr); \
\
	const inc_t     rs_a  = 1; \
	const inc_t     cs_a  = PASTEMAC(ch,packmr); \
\
	const inc_t     rs_b  = PASTEMAC(ch,packnr); \
	const inc_t     cs_b  = 1; \
\
	dim_t           iter, i, j, l; \
	dim_t           n_behind; \
\
	for ( iter = 0; iter < m; ++iter ) \
	{ \
		i        = iter; \
		n_behind = i; \
\
		ctype* restrict alpha11  = a + (i  )*rs_a + (i  )*cs_a; \
		ctype* restrict a10t     = a + (i  )*rs_a + (0  )*cs_a; \
		ctype* restrict B0       = b + (0  )*rs_b + (0  )*cs_b; \
		ctype* restrict b1       = b + (i  )*rs_b + (0  )*cs_b; \
\
		/* b1 = b1 - a10t * B0; */ \
		/* b1 = b1 / alpha11; */ \
		for ( j = 0; j < n; ++j ) \
		{ \
			ctype* restrict b01     = B0 + (0  )*rs_b + (j  )*cs_b; \
			ctype* restrict beta11  = b1 + (0  )*rs_b + (j  )*cs_b; \
			ctype* restrict gamma11 = c  + (i  )*rs_c + (j  )*cs_c; \
			ctype           beta11c = *beta11; \
			ctype           rho11; \
\
			/* beta11 = beta11 - a10t * b01; */ \
			PASTEMAC(ch,set0s)( rho11 ); \
			for ( l = 0; l < n_behind; ++l ) \
			{ \
				ctype* restrict alpha10 = a10t + (l  )*cs_a; \
				ctype* restrict beta01  = b01  + (l  )*rs_b; \
\
				PASTEMAC(ch,axpys)( *alpha10, *beta01, rho11 ); \
			} \
			PASTEMAC(ch,subs)( rho11, beta11c ); \
\
			/* beta11 = beta11 / alpha11; */ \
			/* NOTE: The INVERSE of alpha11 (1.0/alpha11) is stored instead
			   of alpha11, so we can multiply rather than divide. We store 
			   the inverse of alpha11 intentionally to avoid expensive
			   division instructions within the micro-kernel. */ \
			PASTEMAC(ch,scals)( *alpha11, beta11c ); \
\
			/* Output final result to matrix c. */ \
			PASTEMAC(ch,copys)( beta11c, *gamma11 ); \
\
			/* Store the local value back to b11. */ \
			PASTEMAC(ch,copys)( beta11c, *beta11 ); \
		} \
	} \
}

INSERT_GENTFUNC_BASIC0( trsm_l_ukr_ref )

