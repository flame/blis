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

#undef  GENTFUNCCO
#define GENTFUNCCO( ctype, ctype_r, ch, chr, opname ) \
\
void PASTEMAC(ch,opname) \
     ( \
       conj_t  conja, \
       pack_t  schema, \
       dim_t   panel_dim, \
       dim_t   panel_len, \
       ctype*  kappa, \
       ctype*  a, inc_t inca, inc_t lda, \
       ctype*  p,             inc_t ldp, \
       cntx_t* cntx  \
     ) \
{ \
	num_t     dt     = PASTEMAC(ch,type); \
	l1mkr_t   ker_id = panel_dim; \
\
	PASTECH2(ch,opname,_ker_ft) f; \
\
	/* Query the context for the packm kernel corresponding to the current
	   panel dimension, or kernel id. If the id is invalid, the function will
	   return NULL. */ \
	f = bli_cntx_get_packm_ker_dt( dt, ker_id, cntx ); \
\
	/* If there exists a kernel implementation for the micro-panel dimension
	   provided, we invoke the implementation. Otherwise, we use scal2m. */ \
	if ( f != NULL ) \
	{ \
		f \
		( \
		  conja, \
		  schema, \
		  panel_len, \
		  kappa, \
		  a, inca, lda, \
		  p,       ldp, \
		  cntx  \
		); \
	} \
	else \
	{ \
		ctype*   restrict kappa_cast = ( ctype*   )kappa; \
		ctype*   restrict a_r        = ( ctype*   )a; \
		ctype_r* restrict p_r        = ( ctype_r* )p; \
		const dim_t       inca1      = inca; \
		const dim_t       lda1       = lda; \
		const dim_t       ldp1       = ldp; \
		dim_t             i, j; \
\
		/* Treat the micro-panel as panel_dim x panel_len and column-stored
		   (unit row stride). */ \
\
		if ( bli_is_ro_packed( schema ) ) \
		{ \
			if ( bli_is_conj( conja ) ) \
			{ \
				for ( j = 0; j < panel_len; ++j ) \
				{ \
					for ( i = 0; i < panel_dim; ++i ) \
					{ \
						ctype*   restrict alpha11 = a_r + (i  )*inca1 + (j  )*lda1; \
						ctype_r* restrict pi11_r  = p_r + (i  )*1     + (j  )*ldp1; \
\
						PASTEMAC(ch,scal2jros) \
						( \
						  *kappa_cast, \
						  *alpha11, \
						  *pi11_r  \
						); \
					} \
				} \
			} \
			else /* if ( bli_is_noconj( conja ) ) */ \
			{ \
				for ( j = 0; j < panel_len; ++j ) \
				{ \
					for ( i = 0; i < panel_dim; ++i ) \
					{ \
						ctype*   restrict alpha11 = a_r + (i  )*inca1 + (j  )*lda1; \
						ctype_r* restrict pi11_r  = p_r + (i  )*1     + (j  )*ldp1; \
\
						PASTEMAC(ch,scal2ros) \
						( \
						  *kappa_cast, \
						  *alpha11, \
						  *pi11_r  \
						); \
					} \
				} \
			} \
		} \
		else if ( bli_is_io_packed( schema ) ) \
		{ \
			if ( bli_is_conj( conja ) ) \
			{ \
				for ( j = 0; j < panel_len; ++j ) \
				{ \
					for ( i = 0; i < panel_dim; ++i ) \
					{ \
						ctype*   restrict alpha11 = a_r + (i  )*inca1 + (j  )*lda1; \
						ctype_r* restrict pi11_r  = p_r + (i  )*1     + (j  )*ldp1; \
\
						PASTEMAC(ch,scal2jios) \
						( \
						  *kappa_cast, \
						  *alpha11, \
						  *pi11_r  \
						); \
					} \
				} \
			} \
			else /* if ( bli_is_noconj( conja ) ) */ \
			{ \
				for ( j = 0; j < panel_len; ++j ) \
				{ \
					for ( i = 0; i < panel_dim; ++i ) \
					{ \
						ctype*   restrict alpha11 = a_r + (i  )*inca1 + (j  )*lda1; \
						ctype_r* restrict pi11_r  = p_r + (i  )*1     + (j  )*ldp1; \
\
						PASTEMAC(ch,scal2ios) \
						( \
						  *kappa_cast, \
						  *alpha11, \
						  *pi11_r  \
						); \
					} \
				} \
			} \
		} \
		else /* if ( bli_is_rpi_packed( schema ) ) */ \
		{ \
			if ( bli_is_conj( conja ) ) \
			{ \
				for ( j = 0; j < panel_len; ++j ) \
				{ \
					for ( i = 0; i < panel_dim; ++i ) \
					{ \
						ctype*   restrict alpha11 = a_r + (i  )*inca1 + (j  )*lda1; \
						ctype_r* restrict pi11_r  = p_r + (i  )*1     + (j  )*ldp1; \
\
						PASTEMAC(ch,scal2jrpis) \
						( \
						  *kappa_cast, \
						  *alpha11, \
						  *pi11_r  \
						); \
					} \
				} \
			} \
			else /* if ( bli_is_noconj( conja ) ) */ \
			{ \
				for ( j = 0; j < panel_len; ++j ) \
				{ \
					for ( i = 0; i < panel_dim; ++i ) \
					{ \
						ctype*   restrict alpha11 = a_r + (i  )*inca1 + (j  )*lda1; \
						ctype_r* restrict pi11_r  = p_r + (i  )*1     + (j  )*ldp1; \
\
						PASTEMAC(ch,scal2rpis) \
						( \
						  *kappa_cast, \
						  *alpha11, \
						  *pi11_r  \
						); \
					} \
				} \
			} \
		} \
	} \
}

INSERT_GENTFUNCCO_BASIC0( packm_cxk_rih )

