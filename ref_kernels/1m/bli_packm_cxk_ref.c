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


// 2022 Jun 13 - loops rerolled, compiled under gcc with -O2 -- Leick Robinson
//               To force rolled loops, define BLIS_FORCE_ROLL_PACKM_REF_KERNEL


#include "blis.h"


#ifdef BLIS_FORCE_ROLL_PACKM_REF_KERNEL
// When this cpp macro is enabled, the following pragmas prevent the compiler
// from performing loop unrolling (i.e., the packm loops are left "rolled").
#pragma GCC push_options
#pragma GCC optimize ("O2")
#endif


#if 0

#undef  GENTFUNC
#define GENTFUNC( ctype, ch, opname, mnr, arch, suf ) \
\
void PASTEMAC3(ch,opname,arch,suf) \
     ( \
             conj_t  conja, \
             pack_t  schema, \
             dim_t   cdim, \
             dim_t   n, \
             dim_t   n_max, \
       const ctype*  kappa, \
       const ctype*  a, inc_t inca, inc_t lda, \
             ctype*  p,             inc_t ldp, \
       const cntx_t* cntx  \
     ) \
{ \
	const ctype* restrict kappa_cast = kappa; \
	const ctype* restrict alpha1     = a; \
	      ctype* restrict pi1        = p; \
\
	dim_t           n_iter     = n / 4; \
	dim_t           n_left     = n % 4; \
\
	if ( cdim == mnr ) \
	{ \
		if ( PASTEMAC(ch,eq1)( *kappa_cast ) ) \
		{ \
			if ( bli_is_conj( conja ) ) \
			{ \
				for ( dim_t k = n; k != 0; --k ) \
				{ \
                    for (int mn = 0; mn < 2; ++mn) { \
					    PASTEMAC(ch,copyjs)( *(alpha1 + mn*inca), *(pi1 + mn) ); \
                    } \
\
					alpha1 += lda; \
					pi1    += ldp; \
				} \
			} \
			else \
			{ \
				for ( ; n_iter != 0; --n_iter ) \
				{ \
                    for (int incr = 0; incr < 4; ++incr) { \
                        for (int mn = 0; mn < 2; ++mn) { \
					        PASTEMAC2(ch,ch,copys)( *(alpha1 + mn*inca + incr*lda), *(pi1 + mn + incr*ldp) ); \
                        } \
                    } \
\
					alpha1 += 4*lda; \
					pi1    += 4*ldp; \
				} \
\
				for ( ; n_left != 0; --n_left ) \
				{ \
                    for (int mn = 0; mn < 2; ++mn) { \
					    PASTEMAC2(ch,ch,copys)( *(alpha1 + mn*inca), *(pi1 + mn) ); \
                    } \
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
                    for (int mn = 0; mn < 2; ++mn) { \
					    PASTEMAC(ch,scal2js)( *kappa_cast, *(alpha1 + mn*inca), *(pi1 + mn) ); \
                    } \
\
					alpha1 += lda; \
					pi1    += ldp; \
				} \
			} \
			else \
			{ \
				for ( dim_t k = n; k != 0; --k ) \
				{ \
                    for (int mn = 0; mn < 2; ++mn) { \
					    PASTEMAC(ch,scal2s)( *kappa_cast, *(alpha1 + mn*inca), *(pi1 + mn) ); \
                    } \
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

INSERT_GENTFUNC_BASIC( packm_2xk, 2, BLIS_CNAME_INFIX, BLIS_REF_SUFFIX )



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
	dim_t           n_iter     = n / 4; \
	dim_t           n_left     = n % 4; \
\
	if ( cdim == mnr ) \
	{ \
		if ( PASTEMAC(ch,eq1)( *kappa_cast ) ) \
		{ \
			if ( bli_is_conj( conja ) ) \
			{ \
				for ( dim_t k = n; k != 0; --k ) \
				{ \
                    for (int mn = 0; mn < 3; ++mn) { \
					    PASTEMAC2(ch,ch,copyjs)( *(alpha1 + mn*inca), *(pi1 + mn) ); \
                    } \
\
					alpha1 += lda; \
					pi1    += ldp; \
				} \
			} \
			else \
			{ \
				for ( ; n_iter != 0; --n_iter ) \
				{ \
                    for (int incr = 0; incr < 4; ++incr) { \
                        for (int mn = 0; mn < 3; ++mn) { \
					        PASTEMAC2(ch,ch,copys)( *(alpha1 + mn*inca + incr*lda), *(pi1 + mn + incr*ldp) ); \
                        } \
                    } \
\
					alpha1 += 4*lda; \
					pi1    += 4*ldp; \
				} \
\
				for ( ; n_left != 0; --n_left ) \
				{ \
                    for (int mn = 0; mn < 3; ++mn) { \
					    PASTEMAC2(ch,ch,copys)( *(alpha1 + mn*inca), *(pi1 + mn) ); \
                    } \
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
                    for (int mn = 0; mn < 3; ++mn) { \
					    PASTEMAC3(ch,ch,ch,scal2js)( *kappa_cast, *(alpha1 + mn*inca), *(pi1 + mn) ); \
                    } \
\
					alpha1 += lda; \
					pi1    += ldp; \
				} \
			} \
			else \
			{ \
				for ( dim_t k = n; k != 0; --k ) \
				{ \
                    for (int mn = 0; mn < 3; ++mn) { \
					    PASTEMAC3(ch,ch,ch,scal2s)( *kappa_cast, *(alpha1 + mn*inca), *(pi1 + mn) ); \
                    } \
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

INSERT_GENTFUNC_BASIC( packm_3xk, 3, BLIS_CNAME_INFIX, BLIS_REF_SUFFIX )



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
	dim_t           n_iter     = n / 2; \
	dim_t           n_left     = n % 2; \
\
	if ( cdim == mnr ) \
	{ \
		if ( PASTEMAC(ch,eq1)( *kappa_cast ) ) \
		{ \
			if ( bli_is_conj( conja ) ) \
			{ \
				for ( dim_t k = n; k != 0; --k ) \
				{ \
                    for (int mn = 0; mn < 4; ++mn) { \
					    PASTEMAC2(ch,ch,copyjs)( *(alpha1 + mn*inca), *(pi1 + mn) ); \
                    } \
\
					alpha1 += lda; \
					pi1    += ldp; \
				} \
			} \
			else \
			{ \
				for ( ; n_iter != 0; --n_iter ) \
				{ \
                    for (int incr = 0; incr < 2; ++incr) { \
                        for (int mn = 0; mn < 4; ++mn) { \
					        PASTEMAC2(ch,ch,copys)( *(alpha1 + mn*inca + incr*lda), *(pi1 + mn + incr*ldp) ); \
                        } \
                    } \
\
					alpha1 += 2*lda; \
					pi1    += 2*ldp; \
				} \
\
				for ( ; n_left != 0; --n_left ) \
				{ \
                    for (int mn = 0; mn < 4; ++mn) { \
					    PASTEMAC2(ch,ch,copys)( *(alpha1 + mn*inca), *(pi1 + mn) ); \
                    } \
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
                    for (int mn = 0; mn < 4; ++mn) { \
					    PASTEMAC3(ch,ch,ch,scal2js)( *kappa_cast, *(alpha1 + mn*inca), *(pi1 + mn) ); \
                    } \
\
					alpha1 += lda; \
					pi1    += ldp; \
				} \
			} \
			else \
			{ \
				for ( dim_t k = n; k != 0; --k ) \
				{ \
                    for (int mn = 0; mn < 4; ++mn) { \
					    PASTEMAC3(ch,ch,ch,scal2s)( *kappa_cast, *(alpha1 + mn*inca), *(pi1 + mn) ); \
                    } \
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

INSERT_GENTFUNC_BASIC( packm_4xk, 4, BLIS_CNAME_INFIX, BLIS_REF_SUFFIX )



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
                    for (int mn = 0; mn < 6; ++mn) { \
					    PASTEMAC(ch,copyjs)( *(alpha1 + mn*inca), *(pi1 + mn) ); \
                    } \
\
					alpha1 += lda; \
					pi1    += ldp; \
				} \
			} \
			else \
			{ \
				for ( dim_t k = n; k != 0; --k ) \
				{ \
                    for (int mn = 0; mn < 6; ++mn) { \
					    PASTEMAC(ch,copys)( *(alpha1 + mn*inca), *(pi1 + mn) ); \
                    } \
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
                    for (int mn = 0; mn < 6; ++mn) { \
					    PASTEMAC(ch,scal2js)( *kappa_cast, *(alpha1 + mn*inca), *(pi1 + mn) ); \
                    } \
\
					alpha1 += lda; \
					pi1    += ldp; \
				} \
			} \
			else \
			{ \
				for ( dim_t k = n; k != 0; --k ) \
				{ \
                    for (int mn = 0; mn < 6; ++mn) { \
					    PASTEMAC(ch,scal2s)( *kappa_cast, *(alpha1 + mn*inca), *(pi1 + mn) ); \
                    } \
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

INSERT_GENTFUNC_BASIC( packm_6xk, 6, BLIS_CNAME_INFIX, BLIS_REF_SUFFIX )



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
	dim_t           n_iter     = n / 2; \
	dim_t           n_left     = n % 2; \
\
	if ( cdim == mnr ) \
	{ \
		if ( PASTEMAC(ch,eq1)( *kappa_cast ) ) \
		{ \
			if ( bli_is_conj( conja ) ) \
			{ \
				for ( dim_t k = n; k != 0; --k ) \
				{ \
                    for (int mn = 0; mn < 8; ++mn) { \
					    PASTEMAC(ch,copyjs)( *(alpha1 + mn*inca), *(pi1 + mn) ); \
                    } \
\
					alpha1 += lda; \
					pi1    += ldp; \
				} \
			} \
			else \
			{ \
				for ( ; n_iter != 0; --n_iter ) \
				{ \
                    for (int incr = 0; incr < 2; ++incr) { \
                        for (int mn = 0; mn < 8; ++mn) { \
					        PASTEMAC(ch,copys)( *(alpha1 + mn*inca + incr*lda), *(pi1 + mn + incr*ldp) ); \
                        } \
                    } \
\
					alpha1 += 2*lda; \
					pi1    += 2*ldp; \
				} \
\
				for ( ; n_left != 0; --n_left ) \
				{ \
                    for (int mn = 0; mn < 8; ++mn) { \
					    PASTEMAC(ch,copys)( *(alpha1 + mn*inca), *(pi1 + mn) ); \
                    } \
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
                    for (int mn = 0; mn < 8; ++mn) { \
					    PASTEMAC(ch,scal2js)( *kappa_cast, *(alpha1 + mn*inca), *(pi1 + mn) ); \
                    } \
\
					alpha1 += lda; \
					pi1    += ldp; \
				} \
			} \
			else \
			{ \
				for ( dim_t k = n; k != 0; --k ) \
				{ \
                    for (int mn = 0; mn < 8; ++mn) { \
					    PASTEMAC(ch,scal2s)( *kappa_cast, *(alpha1 + mn*inca), *(pi1 + mn) ); \
                    } \
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

INSERT_GENTFUNC_BASIC( packm_8xk, 8, BLIS_CNAME_INFIX, BLIS_REF_SUFFIX )



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
                    for (int mn = 0; mn < 10; ++mn) { \
					    PASTEMAC(ch,copyjs)( *(alpha1 + mn*inca), *(pi1 + mn) ); \
                    } \
\
					alpha1 += lda; \
					pi1    += ldp; \
				} \
			} \
			else \
			{ \
				for ( dim_t k = n; k != 0; --k ) \
				{ \
                    for (int mn = 0; mn < 10; ++mn) { \
					    PASTEMAC(ch,copys)( *(alpha1 + mn*inca), *(pi1 + mn) ); \
                    } \
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
                    for (int mn = 0; mn < 10; ++mn) { \
					    PASTEMAC(ch,scal2js)( *kappa_cast, *(alpha1 + mn*inca), *(pi1 + mn) ); \
                    } \
\
					alpha1 += lda; \
					pi1    += ldp; \
				} \
			} \
			else \
			{ \
				for ( dim_t k = n; k != 0; --k ) \
				{ \
                    for (int mn = 0; mn < 10; ++mn) { \
					    PASTEMAC(ch,scal2s)( *kappa_cast, *(alpha1 + mn*inca), *(pi1 + mn) ); \
                    } \
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

INSERT_GENTFUNC_BASIC( packm_10xk, 10, BLIS_CNAME_INFIX, BLIS_REF_SUFFIX )



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
                    for (int mn = 0; mn < 12; ++mn) { \
					    PASTEMAC(ch,copyjs)( *(alpha1 + mn*inca), *(pi1 + mn) ); \
                    } \
\
					alpha1 += lda; \
					pi1    += ldp; \
				} \
			} \
			else \
			{ \
				for ( dim_t k = n; k != 0; --k ) \
				{ \
                    for (int mn = 0; mn < 12; ++mn) { \
					    PASTEMAC(ch,copys)( *(alpha1 + mn*inca), *(pi1 + mn) ); \
                    } \
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
                    for (int mn = 0; mn < 12; ++mn) { \
					    PASTEMAC(ch,scal2js)( *kappa_cast, *(alpha1 + mn*inca), *(pi1 + mn) ); \
                    } \
\
					alpha1 += lda; \
					pi1    += ldp; \
				} \
			} \
			else \
			{ \
				for ( dim_t k = n; k != 0; --k ) \
				{ \
                    for (int mn = 0; mn < 12; ++mn) { \
					    PASTEMAC(ch,scal2s)( *kappa_cast, *(alpha1 + mn*inca), *(pi1 + mn) ); \
                    } \
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

INSERT_GENTFUNC_BASIC( packm_12xk, 12, BLIS_CNAME_INFIX, BLIS_REF_SUFFIX )



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
                    for (int mn = 0; mn < 14; ++mn) { \
					    PASTEMAC(ch,copyjs)( *(alpha1 + mn*inca), *(pi1 + mn) ); \
                    } \
\
					alpha1 += lda; \
					pi1    += ldp; \
				} \
			} \
			else \
			{ \
				for ( dim_t k = n; k != 0; --k ) \
				{ \
                    for (int mn = 0; mn < 14; ++mn) { \
					    PASTEMAC(ch,copys)( *(alpha1 + mn*inca), *(pi1 + mn) ); \
                    } \
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
                    for (int mn = 0; mn < 14; ++mn) { \
					    PASTEMAC(ch,scal2js)( *kappa_cast, *(alpha1 + mn*inca), *(pi1 + mn) ); \
                    } \
\
					alpha1 += lda; \
					pi1    += ldp; \
				} \
			} \
			else \
			{ \
				for ( dim_t k = n; k != 0; --k ) \
				{ \
                    for (int mn = 0; mn < 14; ++mn) { \
					    PASTEMAC(ch,scal2s)( *kappa_cast, *(alpha1 + mn*inca), *(pi1 + mn) ); \
                    } \
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

INSERT_GENTFUNC_BASIC( packm_14xk, 14, BLIS_CNAME_INFIX, BLIS_REF_SUFFIX )



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
                    for (int mn = 0; mn < 16; ++mn) { \
					    PASTEMAC(ch,copyjs)( *(alpha1 + mn*inca), *(pi1 + mn) ); \
                    } \
\
					alpha1 += lda; \
					pi1    += ldp; \
				} \
			} \
			else \
			{ \
				for ( dim_t k = n; k != 0; --k ) \
				{ \
                    for (int mn = 0; mn < 16; ++mn) { \
					    PASTEMAC(ch,copys)( *(alpha1 + mn*inca), *(pi1 + mn) ); \
                    } \
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
                    for (int mn = 0; mn < 16; ++mn) { \
					    PASTEMAC(ch,scal2js)( *kappa_cast, *(alpha1 + mn*inca), *(pi1 + mn) ); \
                    } \
\
					alpha1 += lda; \
					pi1    += ldp; \
				} \
			} \
			else \
			{ \
				for ( dim_t k = n; k != 0; --k ) \
				{ \
                    for (int mn = 0; mn < 16; ++mn) { \
					    PASTEMAC(ch,scal2s)( *kappa_cast, *(alpha1 + mn*inca), *(pi1 + mn) ); \
                    } \
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

INSERT_GENTFUNC_BASIC( packm_16xk, 16, BLIS_CNAME_INFIX, BLIS_REF_SUFFIX )



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
                    for (int mn = 0; mn < 24; ++mn) { \
					    PASTEMAC(ch,copyjs)( *(alpha1 + mn*inca), *(pi1 + mn) ); \
                    } \
\
					alpha1 += lda; \
					pi1    += ldp; \
				} \
			} \
			else \
			{ \
				for ( dim_t k = n; k != 0; --k ) \
				{ \
                    for (int mn = 0; mn < 24; ++mn) { \
					    PASTEMAC(ch,copys)( *(alpha1 + mn*inca), *(pi1 + mn) ); \
                    } \
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
                    for (int mn = 0; mn < 24; ++mn) { \
					    PASTEMAC(ch,scal2js)( *kappa_cast, *(alpha1 + mn*inca), *(pi1 + mn) ); \
                    } \
\
					alpha1 += lda; \
					pi1    += ldp; \
				} \
			} \
			else \
			{ \
				for ( dim_t k = n; k != 0; --k ) \
				{ \
                    for (int mn = 0; mn < 24; ++mn) { \
					    PASTEMAC(ch,scal2s)( *kappa_cast, *(alpha1 + mn*inca), *(pi1 + mn) ); \
                    } \
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

INSERT_GENTFUNC_BASIC( packm_24xk, 24, BLIS_CNAME_INFIX, BLIS_REF_SUFFIX )


#else

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

// ---

#define PACKM_BODY_1( ctype, ch, pragma, cdim, inca, op ) \
\
do \
{ \
	for ( dim_t k = n; k != 0; --k ) \
	{ \
		pragma \
		for ( dim_t mn = 0; mn < cdim; mn++ ) \
		for ( dim_t d = 0; d < dfac; d++ ) \
			PASTEMAC(ch,op)( *(alpha1 + mn*inca), *(pi1 + mn*dfac + d) ); \
\
		alpha1 += lda; \
		pi1    += ldp; \
	} \
} while(0)

// ---

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
			if ( PASTEMAC(ch,eq1)( kappa_cast ) ) \
			{ \
				if ( bli_is_conj( conja ) ) PACKM_BODY_1( ctype, ch, PRAGMA_SIMD, mnr, 1, copyjs ); \
				else                        PACKM_BODY_1( ctype, ch, PRAGMA_SIMD, mnr, 1, copys ); \
			} \
			else \
			{ \
				if ( bli_is_conj( conja ) ) PACKM_BODY( ctype, ch, PRAGMA_SIMD, mnr, 1, scal2js ); \
				else                        PACKM_BODY( ctype, ch, PRAGMA_SIMD, mnr, 1, scal2s ); \
			} \
		} \
		else \
		{ \
			if ( PASTEMAC(ch,eq1)( kappa_cast ) ) \
			{ \
				if ( bli_is_conj( conja ) ) PACKM_BODY_1( ctype, ch, PRAGMA_SIMD, mnr, inca, copyjs ); \
				else                        PACKM_BODY_1( ctype, ch, PRAGMA_SIMD, mnr, inca, copys ); \
			} \
			else \
			{ \
				if ( bli_is_conj( conja ) ) PACKM_BODY( ctype, ch, PRAGMA_SIMD, mnr, inca, scal2js ); \
				else                        PACKM_BODY( ctype, ch, PRAGMA_SIMD, mnr, inca, scal2s ); \
			} \
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


#endif
