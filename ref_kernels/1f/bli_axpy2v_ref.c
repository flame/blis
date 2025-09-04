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
#define GENTFUNC( ctype, ch, opname, arch, suf ) \
\
void PASTEMAC3(ch,opname,arch,suf) \
     ( \
       conj_t           conjx, \
       conj_t           conjy, \
       dim_t            n, \
       ctype*  restrict alphax, \
       ctype*  restrict alphay, \
       ctype*  restrict x, inc_t incx, \
       ctype*  restrict y, inc_t incy, \
       ctype*  restrict z, inc_t incz, \
       cntx_t* restrict cntx  \
     ) \
{ \
	if ( bli_zero_dim1( n ) ) return; \
\
	if ( incz == 1 && incx == 1 && incy == 1 ) \
	{ \
		ctype chic, psic; \
\
		if ( bli_is_noconj( conjx ) ) \
		{ \
			if ( bli_is_noconj( conjy ) ) \
			{ \
				PRAGMA_SIMD \
				for ( dim_t i = 0; i < n; ++i ) \
				{ \
					PASTEMAC(ch,axpys)( *alphax, x[i], z[i] ); \
					PASTEMAC(ch,axpys)( *alphay, y[i], z[i] ); \
				} \
			} \
			else /* if ( bli_is_conj( conjy ) ) */ \
			{ \
				PRAGMA_SIMD \
				for ( dim_t i = 0; i < n; ++i ) \
				{ \
					PASTEMAC(ch,axpys)( *alphax, x[i], z[i] ); \
					PASTEMAC(ch,copyjs)( y[i], psic ); \
					PASTEMAC(ch,axpys)( *alphay, psic, z[i] ); \
				} \
			} \
		} \
		else /* if ( bli_is_conj( conjx ) ) */ \
		{ \
			if ( bli_is_noconj( conjy ) ) \
			{ \
				PRAGMA_SIMD \
				for ( dim_t i = 0; i < n; ++i ) \
				{ \
					PASTEMAC(ch,copyjs)( x[i], chic ); \
					PASTEMAC(ch,axpys)( *alphax, chic, z[i] ); \
					PASTEMAC(ch,axpys)( *alphay, y[i], z[i] ); \
				} \
			} \
			else /* if ( bli_is_conj( conjy ) ) */ \
			{ \
				PRAGMA_SIMD \
				for ( dim_t i = 0; i < n; ++i ) \
				{ \
					PASTEMAC(ch,copyjs)( x[i], chic ); \
					PASTEMAC(ch,axpys)( *alphax, chic, z[i] ); \
					PASTEMAC(ch,copyjs)( y[i], psic ); \
					PASTEMAC(ch,axpys)( *alphay, psic, z[i] ); \
				} \
			} \
		} \
	} \
	else \
	{ \
		/* Query the context for the kernel function pointer. */ \
		const num_t              dt     = PASTEMAC(ch,type); \
		PASTECH(ch,axpyv_ker_ft) kfp_av \
		= \
		bli_cntx_get_l1v_ker_dt( dt, BLIS_AXPYV_KER, cntx ); \
\
		kfp_av \
		( \
		  conjx, \
		  n, \
		  alphax, \
		  x, incx, \
		  z, incz, \
		  cntx  \
		); \
\
		kfp_av \
		( \
		  conjy, \
		  n, \
		  alphay, \
		  y, incy, \
		  z, incz, \
		  cntx  \
		); \
	} \
}

INSERT_GENTFUNC_BASIC2( axpy2v, BLIS_CNAME_INFIX, BLIS_REF_SUFFIX )

