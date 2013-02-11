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

#include "blis2.h"

#define FUNCPTR_T packm_cxk_fp

typedef void (*FUNCPTR_T)(
                           conj_t  conja,
                           dim_t   n,
                           void*   beta,
                           void*   a, inc_t inca, inc_t lda,
                           void*   p
                         );

#undef  FUNCPTR_ARRAY_LENGTH
#define FUNCPTR_ARRAY_LENGTH 18

#undef  GENARRAY
#define GENARRAY( kername2,  kername4,  kername6,  kername8,   \
                  kername10, kername12, kername14, kername16 ) \
\
static FUNCPTR_T ftypes[FUNCPTR_ARRAY_LENGTH][BLIS_NUM_FP_TYPES] = \
{ \
	/* panel width = 0 */ \
	{ \
		NULL, NULL, NULL, NULL, \
	}, \
	/* panel width = 1 */ \
	{ \
		NULL, NULL, NULL, NULL, \
	}, \
	/* panel width = 2 */ \
	{ \
		PASTEMAC(s,kername2), \
		PASTEMAC(c,kername2), \
		PASTEMAC(d,kername2), \
		PASTEMAC(z,kername2), \
	}, \
	/* panel width = 3 */ \
	{ \
		NULL, NULL, NULL, NULL, \
	}, \
	/* panel width = 4 */ \
	{ \
		PASTEMAC(s,kername4), \
		PASTEMAC(c,kername4), \
		PASTEMAC(d,kername4), \
		PASTEMAC(z,kername4), \
	}, \
	/* panel width = 5 */ \
	{ \
		NULL, NULL, NULL, NULL, \
	}, \
	/* panel width = 6 */ \
	{ \
		PASTEMAC(s,kername6), \
		PASTEMAC(c,kername6), \
		PASTEMAC(d,kername6), \
		PASTEMAC(z,kername6), \
	}, \
	/* panel width = 7 */ \
	{ \
		NULL, NULL, NULL, NULL, \
	}, \
	/* panel width = 8 */ \
	{ \
		PASTEMAC(s,kername8), \
		PASTEMAC(c,kername8), \
		PASTEMAC(d,kername8), \
		PASTEMAC(z,kername8), \
	}, \
	/* panel width = 9 */ \
	{ \
		NULL, NULL, NULL, NULL, \
	}, \
	/* panel width = 10 */ \
	{ \
		PASTEMAC(s,kername10), \
		PASTEMAC(c,kername10), \
		PASTEMAC(d,kername10), \
		PASTEMAC(z,kername10), \
	}, \
	/* panel width = 11 */ \
	{ \
		NULL, NULL, NULL, NULL, \
	}, \
	/* panel width = 12 */ \
	{ \
		PASTEMAC(s,kername12), \
		PASTEMAC(c,kername12), \
		PASTEMAC(d,kername12), \
		PASTEMAC(z,kername12), \
	}, \
	/* panel width = 13 */ \
	{ \
		NULL, NULL, NULL, NULL, \
	}, \
	/* panel width = 14 */ \
	{ \
		PASTEMAC(s,kername14), \
		PASTEMAC(c,kername14), \
		PASTEMAC(d,kername14), \
		PASTEMAC(z,kername14), \
	}, \
	/* panel width = 15 */ \
	{ \
		NULL, NULL, NULL, NULL, \
	}, \
	/* panel width = 16 */ \
	{ \
		PASTEMAC(s,kername16), \
		PASTEMAC(c,kername16), \
		PASTEMAC(d,kername16), \
		PASTEMAC(z,kername16), \
	}, \
	/* panel width = 17 */ \
	{ \
		NULL, NULL, NULL, NULL, \
	} \
};

GENARRAY( PACKM_2XK_KERNEL,
          PACKM_4XK_KERNEL,
          PACKM_6XK_KERNEL,
          PACKM_8XK_KERNEL,
          PACKM_10XK_KERNEL,
          PACKM_12XK_KERNEL,
          PACKM_14XK_KERNEL,
          PACKM_16XK_KERNEL ) 



#undef  GENTFUNC
#define GENTFUNC( ctype, ch, opname, scal2vker ) \
\
void PASTEMAC(ch,opname)( \
                          conj_t  conja, \
                          dim_t   m, \
                          dim_t   n, \
                          void*   beta, \
                          void*   a, inc_t inca, inc_t lda, \
                          void*   p,             inc_t ldp  \
                        ) \
{ \
	dim_t     panel_dim; \
	num_t     dt; \
	FUNCPTR_T f; \
\
	/* If the panel dimension is unit, then we recognize that this allows
	   the kernel to reduce to a copyv, so we call that kernel directly. */ \
	if ( m == 1 ) \
	{ \
		PASTEMAC3(ch,ch,ch,scal2vker)( conja, \
		                               n, \
		                               beta, \
		                               a, lda, \
		                               p, ldp ); \
		return; \
	} \
\
	/* The panel dimension is always equal to the leading dimension of p. */ \
	panel_dim = ldp; \
\
	/* Acquire the datatype for the current function. */ \
	dt = PASTEMAC(ch,type); \
\
	/* Index into the array to extract the correct function pointer.
	   If the panel dimension is too big to be within the array of
	   explicitly handled kernels, then we treat that kernel the same
	   as if it were in range but unimplemented. */ \
	if ( panel_dim < FUNCPTR_ARRAY_LENGTH ) f = ftypes[panel_dim][dt]; \
	else                                    f = NULL; \
\
	/* If there exists a kernel implementation for the panel dimension
	   provided, and the "width" of the panel is equal to the leading
	   dimension, we invoke the implementation. Otherwise, we use scal2m.
	   By using scal2m to handle edge cases (where m < panel_dim), we
	   allow the kernel implementations to remain very simple. */ \
	if ( f != NULL && m == panel_dim ) \
	{ \
		f( conja, \
		   n, \
		   beta, \
		   a, inca, lda, \
		   p ); \
	} \
	else \
	{ \
		/* Treat the panel as m x n and column-stored (unit row stride). */ \
		PASTEMAC3(ch,ch,ch,scal2m)( 0, \
		                            BLIS_NONUNIT_DIAG, \
		                            BLIS_DENSE, \
		                            conja, \
		                            m, \
		                            n, \
		                            beta, \
		                            a, inca, lda, \
		                            p, 1,    ldp ); \
	} \
}

INSERT_GENTFUNC_BASIC( packm_cxk, SCAL2V_KERNEL )

