/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2014, The University of Texas at Austin
   Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.

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


#define PACKM_BODY( ctype, ch, pragma, cdim, inca, op ) \
\
do \
{ \
	for ( dim_t k = n; k != 0; --k ) \
	{ \
		pragma \
		for ( dim_t mn = 0; mn < cdim; mn++ ) \
		for ( dim_t d = 0; d < dfac; d++ ) \
			PASTEMAC(ch,op)( kappa_cast, *(alpha1 + mn*inca), *(pi1 + mn*dfac + d) ); \
\
		alpha1 += lda; \
		pi1    += ldp; \
	} \
} while(0)


#undef  GENTFUNC
#define GENTFUNC( ctype, ch, opname, mnr0, bb0, arch, suf ) \
\
void PASTEMAC3(ch,opname,arch,suf) \
     ( \
             conj_t  conja, \
             pack_t  schema, \
             dim_t   cdim, \
             dim_t   n, \
             dim_t   n_max, \
       const void*   kappa, \
       const void*   a, inc_t inca, inc_t lda, \
             void*   p,             inc_t ldp, \
       const cntx_t* cntx  \
     ) \
{ \
	const dim_t     mnr        = PASTECH2(mnr0, _, ch); \
	const num_t     dt         = PASTEMAC(ch,type); \
	const dim_t     cdim_max   = bli_cntx_get_blksz_def_dt( dt, mnr0, cntx ); \
	const dim_t     dfac       = PASTECH2(bb0, _, ch); \
\
	      ctype           kappa_cast = *( ctype* )kappa; \
	const ctype* restrict alpha1     = a; \
	      ctype* restrict pi1        = p; \
\
	if ( cdim == mnr && mnr != -1 ) \
	{ \
		if ( inca == 1 ) \
		{ \
			if ( bli_is_conj( conja ) ) PACKM_BODY( ctype, ch, PRAGMA_SIMD, mnr, 1, scal2js ); \
			else                        PACKM_BODY( ctype, ch, PRAGMA_SIMD, mnr, 1, scal2s ); \
		} \
		else \
		{ \
			if ( bli_is_conj( conja ) ) PACKM_BODY( ctype, ch, PRAGMA_SIMD, mnr, inca, scal2js ); \
			else                        PACKM_BODY( ctype, ch, PRAGMA_SIMD, mnr, inca, scal2s ); \
		} \
	} \
	else /* if ( cdim < mnr ) */ \
	{ \
		if ( bli_is_conj( conja ) ) PACKM_BODY( ctype, ch, , cdim, inca, scal2js ); \
		else                        PACKM_BODY( ctype, ch, , cdim, inca, scal2s ); \
	} \
\
	PASTEMAC(ch,set0s_edge) \
	( \
	  cdim*dfac, cdim_max*dfac, \
	  n, n_max, \
	  p, ldp  \
	); \
}

INSERT_GENTFUNC_BASIC( packm_mrxk, BLIS_MR, BLIS_BBM, BLIS_CNAME_INFIX, BLIS_REF_SUFFIX )
INSERT_GENTFUNC_BASIC( packm_nrxk, BLIS_NR, BLIS_BBN, BLIS_CNAME_INFIX, BLIS_REF_SUFFIX )

#undef  GENTFUNC
#define GENTFUNC( ctype, ch, opname, mnr, arch, suf ) \
\
void PASTEMAC3(ch,opname,arch,suf) \
     ( \
       conj_t           conja, \
       pack_t           schema, \
       dim_t            cdim, \
       dim_t            n, \
       dim_t            n_max, \
       ctype*  restrict kappa, \
       ctype*  restrict a, inc_t inca, inc_t lda, \
       ctype*  restrict p,             inc_t ldp, \
       cntx_t* restrict cntx \
     ) \
{ \
    ctype* restrict kappa_cast = kappa; \
    ctype* restrict alpha1     = a; \
    ctype* restrict pi1        = p; \
\
	if ( cdim == mnr ) \
	{ \
		if ( PASTEMAC(ch,eq1)( *kappa_cast ) ) \
		{ \
			if ( bli_is_conj( conja ) ) \
			{ \
				for ( dim_t k = n; k != 0; --k ) \
				{ \
					PASTEMAC(ch,copyjs)( *(alpha1 + 0*inca), *(pi1 + 0) ); \
					PASTEMAC(ch,copyjs)( *(alpha1 + 1*inca), *(pi1 + 1) ); \
					PASTEMAC(ch,copyjs)( *(alpha1 + 2*inca), *(pi1 + 2) ); \
					PASTEMAC(ch,copyjs)( *(alpha1 + 3*inca), *(pi1 + 3) ); \
					PASTEMAC(ch,copyjs)( *(alpha1 + 4*inca), *(pi1 + 4) ); \
					PASTEMAC(ch,copyjs)( *(alpha1 + 5*inca), *(pi1 + 5) ); \
					PASTEMAC(ch,copyjs)( *(alpha1 + 6*inca), *(pi1 + 6) ); \
					PASTEMAC(ch,copyjs)( *(alpha1 + 7*inca), *(pi1 + 7) ); \
					PASTEMAC(ch,copyjs)( *(alpha1 + 8*inca), *(pi1 + 8) ); \
					PASTEMAC(ch,copyjs)( *(alpha1 + 9*inca), *(pi1 + 9) ); \
					PASTEMAC(ch,copyjs)( *(alpha1 +10*inca), *(pi1 +10) ); \
					PASTEMAC(ch,copyjs)( *(alpha1 +11*inca), *(pi1 +11) ); \
					PASTEMAC(ch,copyjs)( *(alpha1 +12*inca), *(pi1 +12) ); \
					PASTEMAC(ch,copyjs)( *(alpha1 +13*inca), *(pi1 +13) ); \
					PASTEMAC(ch,copyjs)( *(alpha1 +14*inca), *(pi1 +14) ); \
					PASTEMAC(ch,copyjs)( *(alpha1 +15*inca), *(pi1 +15) ); \
					PASTEMAC(ch,copyjs)( *(alpha1 +16*inca), *(pi1 +16) ); \
					PASTEMAC(ch,copyjs)( *(alpha1 +17*inca), *(pi1 +17) ); \
					PASTEMAC(ch,copyjs)( *(alpha1 +18*inca), *(pi1 +18) ); \
					PASTEMAC(ch,copyjs)( *(alpha1 +19*inca), *(pi1 +19) ); \
					PASTEMAC(ch,copyjs)( *(alpha1 +20*inca), *(pi1 +20) ); \
					PASTEMAC(ch,copyjs)( *(alpha1 +21*inca), *(pi1 +21) ); \
					PASTEMAC(ch,copyjs)( *(alpha1 +22*inca), *(pi1 +22) ); \
					PASTEMAC(ch,copyjs)( *(alpha1 +23*inca), *(pi1 +23) ); \
					PASTEMAC(ch,copyjs)( *(alpha1 +24*inca), *(pi1 +24) ); \
					PASTEMAC(ch,copyjs)( *(alpha1 +25*inca), *(pi1 +25) ); \
					PASTEMAC(ch,copyjs)( *(alpha1 +26*inca), *(pi1 +26) ); \
					PASTEMAC(ch,copyjs)( *(alpha1 +27*inca), *(pi1 +27) ); \
					PASTEMAC(ch,copyjs)( *(alpha1 +28*inca), *(pi1 +28) ); \
					PASTEMAC(ch,copyjs)( *(alpha1 +29*inca), *(pi1 +29) ); \
					PASTEMAC(ch,copyjs)( *(alpha1 +30*inca), *(pi1 +30) ); \
					PASTEMAC(ch,copyjs)( *(alpha1 +31*inca), *(pi1 +31) ); \
\
					alpha1 += lda; \
					pi1    += ldp; \
				} \
			} \
			else \
			{ \
				for ( dim_t k = n; k != 0; --k ) \
				{ \
					PASTEMAC(ch,copys)( *(alpha1 + 0*inca), *(pi1 + 0) ); \
					PASTEMAC(ch,copys)( *(alpha1 + 1*inca), *(pi1 + 1) ); \
					PASTEMAC(ch,copys)( *(alpha1 + 2*inca), *(pi1 + 2) ); \
					PASTEMAC(ch,copys)( *(alpha1 + 3*inca), *(pi1 + 3) ); \
					PASTEMAC(ch,copys)( *(alpha1 + 4*inca), *(pi1 + 4) ); \
					PASTEMAC(ch,copys)( *(alpha1 + 5*inca), *(pi1 + 5) ); \
					PASTEMAC(ch,copys)( *(alpha1 + 6*inca), *(pi1 + 6) ); \
					PASTEMAC(ch,copys)( *(alpha1 + 7*inca), *(pi1 + 7) ); \
					PASTEMAC(ch,copys)( *(alpha1 + 8*inca), *(pi1 + 8) ); \
					PASTEMAC(ch,copys)( *(alpha1 + 9*inca), *(pi1 + 9) ); \
					PASTEMAC(ch,copys)( *(alpha1 +10*inca), *(pi1 +10) ); \
					PASTEMAC(ch,copys)( *(alpha1 +11*inca), *(pi1 +11) ); \
					PASTEMAC(ch,copys)( *(alpha1 +12*inca), *(pi1 +12) ); \
					PASTEMAC(ch,copys)( *(alpha1 +13*inca), *(pi1 +13) ); \
					PASTEMAC(ch,copys)( *(alpha1 +14*inca), *(pi1 +14) ); \
					PASTEMAC(ch,copys)( *(alpha1 +15*inca), *(pi1 +15) ); \
					PASTEMAC(ch,copys)( *(alpha1 +16*inca), *(pi1 +16) ); \
					PASTEMAC(ch,copys)( *(alpha1 +17*inca), *(pi1 +17) ); \
					PASTEMAC(ch,copys)( *(alpha1 +18*inca), *(pi1 +18) ); \
					PASTEMAC(ch,copys)( *(alpha1 +19*inca), *(pi1 +19) ); \
					PASTEMAC(ch,copys)( *(alpha1 +20*inca), *(pi1 +20) ); \
					PASTEMAC(ch,copys)( *(alpha1 +21*inca), *(pi1 +21) ); \
					PASTEMAC(ch,copys)( *(alpha1 +22*inca), *(pi1 +22) ); \
					PASTEMAC(ch,copys)( *(alpha1 +23*inca), *(pi1 +23) ); \
					PASTEMAC(ch,copys)( *(alpha1 +24*inca), *(pi1 +24) ); \
					PASTEMAC(ch,copys)( *(alpha1 +25*inca), *(pi1 +25) ); \
					PASTEMAC(ch,copys)( *(alpha1 +26*inca), *(pi1 +26) ); \
					PASTEMAC(ch,copys)( *(alpha1 +27*inca), *(pi1 +27) ); \
					PASTEMAC(ch,copys)( *(alpha1 +28*inca), *(pi1 +28) ); \
					PASTEMAC(ch,copys)( *(alpha1 +29*inca), *(pi1 +29) ); \
					PASTEMAC(ch,copys)( *(alpha1 +30*inca), *(pi1 +30) ); \
					PASTEMAC(ch,copys)( *(alpha1 +31*inca), *(pi1 +31) ); \
\
					alpha1 += lda; \
					pi1    += ldp; \
				} \
			} \
		} \
		else \
		{ \
			if ( bli_is_conj( conja ) ) \
			{ \
				for ( dim_t k = n; k != 0; --k ) \
				{ \
					PASTEMAC(ch,scal2js)( *kappa_cast, *(alpha1 + 0*inca), *(pi1 + 0) ); \
					PASTEMAC(ch,scal2js)( *kappa_cast, *(alpha1 + 1*inca), *(pi1 + 1) ); \
					PASTEMAC(ch,scal2js)( *kappa_cast, *(alpha1 + 2*inca), *(pi1 + 2) ); \
					PASTEMAC(ch,scal2js)( *kappa_cast, *(alpha1 + 3*inca), *(pi1 + 3) ); \
					PASTEMAC(ch,scal2js)( *kappa_cast, *(alpha1 + 4*inca), *(pi1 + 4) ); \
					PASTEMAC(ch,scal2js)( *kappa_cast, *(alpha1 + 5*inca), *(pi1 + 5) ); \
					PASTEMAC(ch,scal2js)( *kappa_cast, *(alpha1 + 6*inca), *(pi1 + 6) ); \
					PASTEMAC(ch,scal2js)( *kappa_cast, *(alpha1 + 7*inca), *(pi1 + 7) ); \
					PASTEMAC(ch,scal2js)( *kappa_cast, *(alpha1 + 8*inca), *(pi1 + 8) ); \
					PASTEMAC(ch,scal2js)( *kappa_cast, *(alpha1 + 9*inca), *(pi1 + 9) ); \
					PASTEMAC(ch,scal2js)( *kappa_cast, *(alpha1 +10*inca), *(pi1 +10) ); \
					PASTEMAC(ch,scal2js)( *kappa_cast, *(alpha1 +11*inca), *(pi1 +11) ); \
					PASTEMAC(ch,scal2js)( *kappa_cast, *(alpha1 +12*inca), *(pi1 +12) ); \
					PASTEMAC(ch,scal2js)( *kappa_cast, *(alpha1 +13*inca), *(pi1 +13) ); \
					PASTEMAC(ch,scal2js)( *kappa_cast, *(alpha1 +14*inca), *(pi1 +14) ); \
					PASTEMAC(ch,scal2js)( *kappa_cast, *(alpha1 +15*inca), *(pi1 +15) ); \
					PASTEMAC(ch,scal2js)( *kappa_cast, *(alpha1 +16*inca), *(pi1 +16) ); \
					PASTEMAC(ch,scal2js)( *kappa_cast, *(alpha1 +17*inca), *(pi1 +17) ); \
					PASTEMAC(ch,scal2js)( *kappa_cast, *(alpha1 +18*inca), *(pi1 +18) ); \
					PASTEMAC(ch,scal2js)( *kappa_cast, *(alpha1 +19*inca), *(pi1 +19) ); \
					PASTEMAC(ch,scal2js)( *kappa_cast, *(alpha1 +20*inca), *(pi1 +20) ); \
					PASTEMAC(ch,scal2js)( *kappa_cast, *(alpha1 +21*inca), *(pi1 +21) ); \
					PASTEMAC(ch,scal2js)( *kappa_cast, *(alpha1 +22*inca), *(pi1 +22) ); \
					PASTEMAC(ch,scal2js)( *kappa_cast, *(alpha1 +23*inca), *(pi1 +23) ); \
					PASTEMAC(ch,scal2js)( *kappa_cast, *(alpha1 +24*inca), *(pi1 +24) ); \
					PASTEMAC(ch,scal2js)( *kappa_cast, *(alpha1 +25*inca), *(pi1 +25) ); \
					PASTEMAC(ch,scal2js)( *kappa_cast, *(alpha1 +26*inca), *(pi1 +26) ); \
					PASTEMAC(ch,scal2js)( *kappa_cast, *(alpha1 +27*inca), *(pi1 +27) ); \
					PASTEMAC(ch,scal2js)( *kappa_cast, *(alpha1 +28*inca), *(pi1 +28) ); \
					PASTEMAC(ch,scal2js)( *kappa_cast, *(alpha1 +29*inca), *(pi1 +29) ); \
					PASTEMAC(ch,scal2js)( *kappa_cast, *(alpha1 +30*inca), *(pi1 +30) ); \
					PASTEMAC(ch,scal2js)( *kappa_cast, *(alpha1 +31*inca), *(pi1 +31) ); \
\
					alpha1 += lda; \
					pi1    += ldp; \
				} \
			} \
			else \
			{ \
				for ( dim_t k = n; k != 0; --k ) \
				{ \
					PASTEMAC(ch,scal2s)( *kappa_cast, *(alpha1 + 0*inca), *(pi1 + 0) ); \
					PASTEMAC(ch,scal2s)( *kappa_cast, *(alpha1 + 1*inca), *(pi1 + 1) ); \
					PASTEMAC(ch,scal2s)( *kappa_cast, *(alpha1 + 2*inca), *(pi1 + 2) ); \
					PASTEMAC(ch,scal2s)( *kappa_cast, *(alpha1 + 3*inca), *(pi1 + 3) ); \
					PASTEMAC(ch,scal2s)( *kappa_cast, *(alpha1 + 4*inca), *(pi1 + 4) ); \
					PASTEMAC(ch,scal2s)( *kappa_cast, *(alpha1 + 5*inca), *(pi1 + 5) ); \
					PASTEMAC(ch,scal2s)( *kappa_cast, *(alpha1 + 6*inca), *(pi1 + 6) ); \
					PASTEMAC(ch,scal2s)( *kappa_cast, *(alpha1 + 7*inca), *(pi1 + 7) ); \
					PASTEMAC(ch,scal2s)( *kappa_cast, *(alpha1 + 8*inca), *(pi1 + 8) ); \
					PASTEMAC(ch,scal2s)( *kappa_cast, *(alpha1 + 9*inca), *(pi1 + 9) ); \
					PASTEMAC(ch,scal2s)( *kappa_cast, *(alpha1 +10*inca), *(pi1 +10) ); \
					PASTEMAC(ch,scal2s)( *kappa_cast, *(alpha1 +11*inca), *(pi1 +11) ); \
					PASTEMAC(ch,scal2s)( *kappa_cast, *(alpha1 +12*inca), *(pi1 +12) ); \
					PASTEMAC(ch,scal2s)( *kappa_cast, *(alpha1 +13*inca), *(pi1 +13) ); \
					PASTEMAC(ch,scal2s)( *kappa_cast, *(alpha1 +14*inca), *(pi1 +14) ); \
					PASTEMAC(ch,scal2s)( *kappa_cast, *(alpha1 +15*inca), *(pi1 +15) ); \
					PASTEMAC(ch,scal2s)( *kappa_cast, *(alpha1 +16*inca), *(pi1 +16) ); \
					PASTEMAC(ch,scal2s)( *kappa_cast, *(alpha1 +17*inca), *(pi1 +17) ); \
					PASTEMAC(ch,scal2s)( *kappa_cast, *(alpha1 +18*inca), *(pi1 +18) ); \
					PASTEMAC(ch,scal2s)( *kappa_cast, *(alpha1 +19*inca), *(pi1 +19) ); \
					PASTEMAC(ch,scal2s)( *kappa_cast, *(alpha1 +20*inca), *(pi1 +20) ); \
					PASTEMAC(ch,scal2s)( *kappa_cast, *(alpha1 +21*inca), *(pi1 +21) ); \
					PASTEMAC(ch,scal2s)( *kappa_cast, *(alpha1 +22*inca), *(pi1 +22) ); \
					PASTEMAC(ch,scal2s)( *kappa_cast, *(alpha1 +23*inca), *(pi1 +23) ); \
					PASTEMAC(ch,scal2s)( *kappa_cast, *(alpha1 +24*inca), *(pi1 +24) ); \
					PASTEMAC(ch,scal2s)( *kappa_cast, *(alpha1 +25*inca), *(pi1 +25) ); \
					PASTEMAC(ch,scal2s)( *kappa_cast, *(alpha1 +26*inca), *(pi1 +26) ); \
					PASTEMAC(ch,scal2s)( *kappa_cast, *(alpha1 +27*inca), *(pi1 +27) ); \
					PASTEMAC(ch,scal2s)( *kappa_cast, *(alpha1 +28*inca), *(pi1 +28) ); \
					PASTEMAC(ch,scal2s)( *kappa_cast, *(alpha1 +29*inca), *(pi1 +29) ); \
					PASTEMAC(ch,scal2s)( *kappa_cast, *(alpha1 +30*inca), *(pi1 +30) ); \
					PASTEMAC(ch,scal2s)( *kappa_cast, *(alpha1 +31*inca), *(pi1 +31) ); \
\
					alpha1 += lda; \
					pi1    += ldp; \
				} \
			} \
		} \
	} \
	else /* if ( cdim < mnr ) */ \
	{ \
		PASTEMAC2(ch,scal2m,BLIS_TAPI_EX_SUF) \
		( \
		  0, \
		  BLIS_NONUNIT_DIAG, \
		  BLIS_DENSE, \
		  ( trans_t )conja, \
		  cdim, \
		  n, \
		  kappa, \
		  a, inca, lda, \
		  p,    1, ldp, \
		  cntx, \
		  NULL  \
		); \
\
		/* if ( cdim < mnr ) */ \
		{ \
			const dim_t     i      = cdim; \
			const dim_t     m_edge = mnr - cdim; \
			const dim_t     n_edge = n_max; \
			ctype* restrict p_cast = p; \
			ctype* restrict p_edge = p_cast + (i  )*1; \
\
			PASTEMAC(ch,set0s_mxn) \
			( \
			  m_edge, \
			  n_edge, \
			  p_edge, 1, ldp  \
			); \
		} \
	} \
\
	if ( n < n_max ) \
	{ \
		const dim_t     j      = n; \
		const dim_t     m_edge = mnr; \
		const dim_t     n_edge = n_max - n; \
		ctype* restrict p_cast = p; \
		ctype* restrict p_edge = p_cast + (j  )*ldp; \
\
		PASTEMAC(ch,set0s_mxn) \
		( \
		  m_edge, \
		  n_edge, \
		  p_edge, 1, ldp  \
		); \
	} \
}

INSERT_GENTFUNC_BASIC3( packm_32xk, 32, BLIS_CNAME_INFIX, BLIS_REF_SUFFIX )


