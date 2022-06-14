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

#define bli_scopys_ik(kappa, alpha, beta) bli_scopys(alpha, beta)
#define bli_dcopys_ik(kappa, alpha, beta) bli_dcopys(alpha, beta)
#define bli_ccopys_ik(kappa, alpha, beta) bli_ccopys(alpha, beta)
#define bli_zcopys_ik(kappa, alpha, beta) bli_zcopys(alpha, beta)

#define bli_scopyjs_ik(kappa, alpha, beta) bli_scopyjs(alpha, beta)
#define bli_dcopyjs_ik(kappa, alpha, beta) bli_dcopyjs(alpha, beta)
#define bli_ccopyjs_ik(kappa, alpha, beta) bli_ccopyjs(alpha, beta)
#define bli_zcopyjs_ik(kappa, alpha, beta) bli_zcopyjs(alpha, beta)

#define PACK1(ch,op,i) \
PASTEMAC(ch,op)( *kappa_cast, *(alpha1 + (i)*inca), *(pi1 + (i)) );

#define PACK2(ch,op,i) \
PACK1(ch,op,i+0) \
PACK1(ch,op,i+1)

#define PACK3(ch,op,i) \
PACK2(ch,op,i+0) \
PACK1(ch,op,i+2)

#define PACK4(ch,op,i) \
PACK2(ch,op,i+0) \
PACK2(ch,op,i+2)

#define PACK6(ch,op,i) \
PACK4(ch,op,i+0) \
PACK2(ch,op,i+4)

#define PACK8(ch,op,i) \
PACK4(ch,op,i+0) \
PACK4(ch,op,i+4)

#define PACK10(ch,op,i) \
PACK6(ch,op,i+0) \
PACK4(ch,op,i+6)

#define PACK12(ch,op,i) \
PACK6(ch,op,i+0) \
PACK6(ch,op,i+6)

#define PACK14(ch,op,i) \
PACK8(ch,op,i+0) \
PACK6(ch,op,i+8)

#define PACK16(ch,op,i) \
PACK8(ch,op,i+0) \
PACK8(ch,op,i+8)

#define PACK24(ch,op,i) \
PACK16(ch,op,i+0) \
PACK8(ch,op,i+16)

#define PACK_(ch,op,n) PACK##n(ch,op,0)

#define COPY(ch,n) PACK_(ch,copys_ik,n)
#define COPYJ(ch,n) PACK_(ch,copyjs_ik,n)
#define SCAL2(ch,n) PACK_(ch,scal2s,n)
#define SCAL2J(ch,n) PACK_(ch,scal2js,n)

#undef  GENTFUNC
#define GENTFUNC( ctype, ch, opname, mnr0, bb0, arch, suf ) \
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
    dim_t           mnr        = PASTECH2(mnr0, _, ch); \
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
					COPYJ(ch,PASTECH2(mnr0, _, ch)); \
\
					alpha1 += lda; \
					pi1    += ldp; \
				} \
			} \
			else \
			{ \
				for ( dim_t k = n; k != 0; --k ) \
				{ \
					COPY(ch,PASTECH2(mnr0, _, ch)); \
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
					SCAL2J(ch,PASTECH2(mnr0, _, ch)); \
\
					alpha1 += lda; \
					pi1    += ldp; \
				} \
			} \
			else \
			{ \
				for ( dim_t k = n; k != 0; --k ) \
				{ \
					SCAL2(ch,PASTECH2(mnr0, _, ch)); \
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

INSERT_GENTFUNC_BASIC4( packm_mrxk, BLIS_MR, BLIS_BBM, BLIS_CNAME_INFIX, BLIS_REF_SUFFIX )
INSERT_GENTFUNC_BASIC4( packm_nrxk, BLIS_NR, BLIS_BBN, BLIS_CNAME_INFIX, BLIS_REF_SUFFIX )

