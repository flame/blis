/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2014, The University of Texas at Austin
   Copyright (C) 2020 - 2025, Advanced Micro Devices, Inc. All rights reserved.

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
#undef  GENTFUNCR2
#define GENTFUNCR2( ftype_x, ftype_r, chx, chr, blasname, blisname ) \
\
ftype_r PASTEF772S(chr,chx,blasname) \
     ( \
       const f77_int* n, \
       const ftype_x* x, const f77_int* incx  \
     ) \
{ \
	/* Initialize BLIS. */ \
	bli_init_auto(); \
\
	AOCL_DTL_TRACE_ENTRY(AOCL_DTL_LEVEL_TRACE_1) \
	AOCL_DTL_LOG_NRM2_INPUTS(AOCL_DTL_LEVEL_TRACE_1,*MKSTR(chx),*n, *incx);\
\
	dim_t    n0; \
	ftype_x* x0; \
	inc_t    incx0; \
	ftype_r  norm; \
\
	/* Convert/typecast negative values of n to zero. */ \
	bli_convert_blas_dim1( *n, n0 ); \
\
	/* If the input increments are negative, adjust the pointers so we can
	   use positive increments instead. */ \
	bli_convert_blas_incv( n0, (ftype_x*)x, *incx, x0, incx0 ); \
\
	/* Call BLIS interface. */ \
	PASTEMAC2(chx,blisname,BLIS_TAPI_EX_SUF) \
	( \
	  n0, \
	  x0, incx0, \
	  &norm, \
	  NULL, \
	  NULL  \
	); \
\
	AOCL_DTL_LOG_NRM2_STATS(AOCL_DTL_LEVEL_TRACE_1, *MKSTR(chx), *n); \
	AOCL_DTL_TRACE_EXIT(AOCL_DTL_LEVEL_TRACE_1) \
	/* Finalize BLIS. */ \
	bli_finalize_auto(); \
\
	return norm; \
}\
\
IF_BLIS_ENABLE_BLAS(\
ftype_r PASTEF772(chr,chx,blasname) \
     ( \
       const f77_int* n, \
       const ftype_x* x, const f77_int* incx  \
     ) \
{ \
  return PASTEF772S(chr,chx,blasname)( n, x, incx );\
} \
)

INSERT_GENTFUNCR2_BLAS( nrm2, normfv )

