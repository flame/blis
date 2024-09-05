/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2014, The University of Texas at Austin
   Copyright (C) 2020 - 2024, Advanced Micro Devices, Inc. All rights reserved.

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


//
// Define BLAS-to-BLIS interfaces.
//
#undef  GENTFUNCSCAL
#define GENTFUNCSCAL( ftype_x, ftype_a, chx, cha, chau, blasname, blisname ) \
\
void PASTEF772S(chx,cha,blasname) \
     ( \
       const f77_int* n, \
       const ftype_a* alpha, \
       ftype_x* x, const f77_int* incx  \
     ) \
{ \
	AOCL_DTL_TRACE_ENTRY(AOCL_DTL_LEVEL_TRACE_1) \
\
	/* Initialize BLIS. */ \
	bli_init_auto(); \
\
	dim_t n0 = (dim_t)(*n); \
	ftype_x *x0 = x; \
	inc_t incx0 = (inc_t)(*incx); \
\
	if ((n0 <= 0) || (alpha == NULL) || (incx0 <= 0) || PASTEMAC(chau, eq1)(*alpha)) \
	{ \
		AOCL_DTL_TRACE_EXIT(AOCL_DTL_LEVEL_TRACE_1); \
		/* Finalize BLIS. */ \
		bli_finalize_auto(); \
		return ; \
	} \
\
	/* NOTE: We do not natively implement BLAS's csscal/zdscal in BLIS.
	   that is, we just always sub-optimally implement those cases
	   by casting alpha to ctype_x (potentially the complex domain) and
	   using the homogeneous datatype instance according to that type. */ \
	ftype_x  alpha_cast; \
	PASTEMAC2(cha,chx,copys)( *alpha, alpha_cast ); \
\
	/* Call BLIS interface. */ \
	/* Pass size as negative to stipulate don't use SETV when alpha=0 */ \
	PASTEMAC2(chx,blisname,BLIS_TAPI_EX_SUF) \
	( \
	  BLIS_NO_CONJUGATE, \
	  -n0, \
	  &alpha_cast, \
	  x0, incx0, \
	  NULL, \
	  NULL  \
	); \
\
	AOCL_DTL_TRACE_EXIT(AOCL_DTL_LEVEL_TRACE_1) \
	/* Finalize BLIS. */ \
	bli_finalize_auto(); \
}\
IF_BLIS_ENABLE_BLAS(\
void PASTEF772(chx,cha,blasname) \
     ( \
       const f77_int* n, \
       const ftype_a* alpha, \
       ftype_x* x, const f77_int* incx  \
     ) \
{ \
  PASTEF772S(chx,cha,blasname)( n, alpha, x, incx ); \
} \
)

INSERT_GENTFUNCSCAL_BLAS( scal, scalv )
