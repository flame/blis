/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2014, The University of Texas at Austin
   Copyright (C) 2020, Advanced Micro Devices, Inc. All rights reserved.

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

#include "blis.h"

#undef  GENTFUNC
#define GENTFUNC( ctype, ch, varname ) \
\
void PASTEMAC(ch,varname) \
     ( \
       conj_t  conjx, \
       conj_t  conjy, \
       dim_t   m, \
       dim_t   n, \
       ctype*  alpha, \
       ctype*  x, inc_t incx, \
       ctype*  y, inc_t incy, \
       ctype*  a, inc_t rs_a, inc_t cs_a, \
       cntx_t* cntx  \
     ) \
{ \
	AOCL_DTL_TRACE_ENTRY(AOCL_DTL_LEVEL_TRACE_3) \
\
	const num_t dt = PASTEMAC(ch,type); \
\
	ctype*  a1t; \
	ctype*  chi1; \
	ctype*  y1; \
	ctype   alpha_chi1; \
	dim_t   i; \
\
	PASTECH(ch,axpyv_ker_ft) kfp_av; \
\
	/* Query the context for the kernel function pointer. */ \
	kfp_av = bli_cntx_get_l1v_ker_dt( dt, BLIS_AXPYV_KER, cntx ); \
\
	for ( i = 0; i < m; ++i ) \
	{ \
		a1t  = a + (i  )*rs_a + (0  )*cs_a; \
		chi1 = x + (i  )*incx; \
		y1   = y + (0  )*incy; \
\
		/* a1t = a1t + alpha * chi1 * y; */ \
		PASTEMAC(ch,copycjs)( conjx, *chi1, alpha_chi1 ); \
		PASTEMAC(ch,scals)( *alpha, alpha_chi1 ); \
\
		kfp_av \
		( \
		  conjy, \
		  n, \
		  &alpha_chi1, \
		  y1,  incy, \
		  a1t, cs_a, \
		  cntx  \
		); \
	} \
	AOCL_DTL_TRACE_EXIT(AOCL_DTL_LEVEL_TRACE_3) \
\
}

INSERT_GENTFUNC_BASIC0( ger_unb_var1 )

