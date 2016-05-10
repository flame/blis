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
void PASTEMAC(ch,varname) \
     ( \
       conj_t          conjat, \
       conj_t          conja, \
       conj_t          conjw, \
       conj_t          conjx, \
       dim_t           m, \
       dim_t           b_n, \
       ctype* restrict alpha, \
       ctype* restrict a, inc_t inca, inc_t lda, \
       ctype* restrict w, inc_t incw, \
       ctype* restrict x, inc_t incx, \
       ctype* restrict beta, \
       ctype* restrict y, inc_t incy, \
       ctype* restrict z, inc_t incz, \
       cntx_t*         cntx  \
     ) \
{ \
	/* A is m x n.                   */ \
	/* y = beta * y + alpha * A^T w; */ \
	/* z =        z + alpha * A   x; */ \
\
	/* Query the context for the kernel function pointer. */ \
	const num_t          dt     = PASTEMAC(ch,type); \
	PASTECH(ch,dotxf_ft) kfp_df = bli_cntx_get_l1f_ker_dt( dt, BLIS_DOTXF_KER, cntx ); \
	PASTECH(ch,axpyf_ft) kfp_af = bli_cntx_get_l1f_ker_dt( dt, BLIS_AXPYF_KER, cntx ); \
\
	kfp_df \
	( \
	  conjat, \
	  conjw, \
	  m, \
	  b_n, \
	  alpha, \
	  a, inca, lda, \
	  w, incw, \
	  beta, \
	  y, incy, \
	  cntx  \
	); \
\
	kfp_af \
	( \
	  conja, \
	  conjx, \
	  m, \
	  b_n, \
	  alpha, \
	  a, inca, lda, \
	  x, incx, \
	  z, incz, \
	  cntx  \
	); \
}

INSERT_GENTFUNC_BASIC0( dotxaxpyf_ref_var2 )

