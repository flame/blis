/*

   BLIS    
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2012, The University of Texas

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

#include "blis2.h"

#define FUNCPTR_T unpackm_cxk_fp

typedef void (*FUNCPTR_T)(
                           conj_t  conjp,
                           dim_t   n,
                           void*   beta,
                           void*   p,
                           void*   a, inc_t inca, inc_t lda
                         );

static FUNCPTR_T ftypes[10][BLIS_NUM_FP_TYPES] =
{
	// panel width = 0
	{
		NULL,
		NULL,
		NULL,
		NULL,
	},
	// panel width = 1
	{
		NULL,
		NULL,
		NULL,
		NULL,
	},
	// panel width = 2
	{
		PASTEMAC(s,unpackm_2xk),
		PASTEMAC(c,unpackm_2xk),
		PASTEMAC(d,unpackm_2xk),
		PASTEMAC(z,unpackm_2xk),
	},
	// panel width = 3
	{
		NULL,
		NULL,
		NULL,
		NULL,
	},
	// panel width = 4
	{
		PASTEMAC(s,unpackm_4xk),
		PASTEMAC(c,unpackm_4xk),
		PASTEMAC(d,unpackm_4xk),
		PASTEMAC(z,unpackm_4xk),
	},
	// panel width = 5
	{
		NULL,
		NULL,
		NULL,
		NULL,
	},
	// panel width = 6
	{
		NULL,
		NULL,
		NULL,
		NULL,
	},
	// panel width = 7
	{
		NULL,
		NULL,
		NULL,
		NULL,
	},
	// panel width = 8
	{
		NULL,
		NULL,
		NULL,
		NULL,
	},
	// panel width = 9
	{
		NULL,
		NULL,
		NULL,
		NULL,
	}
};




#undef  GENTFUNC
#define GENTFUNC( ctype, ch, opname, varname ) \
\
void PASTEMAC(ch,varname)( \
                           conj_t  conjp, \
                           dim_t   m, \
                           dim_t   n, \
                           void*   beta, \
                           void*   p,             inc_t ldp, \
                           void*   a, inc_t inca, inc_t lda  \
                         ) \
{ \
	dim_t     panel_dim; \
	num_t     dt; \
	FUNCPTR_T f; \
\
	/* The panel dimension is always equal to the leading dimension of p. */ \
	panel_dim = ldp; \
\
	/* Acquire the datatype for the current function. */ \
	dt = PASTEMAC(ch,type); \
\
	/* Index into the array to extract the correct function pointer. */ \
	f = ftypes[panel_dim][dt]; \
\
	/* If there exists a kernel implementation for the panel dimension
	   provided, and the "width" of the panel is equal to the leading
	   dimension, we invoke the implementation. Otherwise, we use scal2m.
	   By using scal2m to handle edge cases (where m < panel_dim), we
	   allow the kernel implementations to remain very simple. */ \
	if ( f != NULL && m == panel_dim ) \
	{ \
		f( conjp, \
		   n, \
		   beta, \
		   p, \
		   a, inca, lda ); \
	} \
	else \
	{ \
		/* Treat the panel as m x n and column-stored (unit row stride).*/ \
		PASTEMAC3(ch,ch,ch,scal2m)( 0, \
		                            BLIS_NONUNIT_DIAG, \
		                            BLIS_DENSE, \
		                            conjp, \
		                            m, \
		                            n, \
		                            beta, \
		                            p, 1,    ldp, \
		                            a, inca, lda ); \
	} \
}

INSERT_GENTFUNC_BASIC( unpackm_cxk, unpackm_cxk )

