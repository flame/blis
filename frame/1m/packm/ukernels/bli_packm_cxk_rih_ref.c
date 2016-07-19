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
#define GENTFUNCCO( ctype, ctype_r, ch, chr, varname ) \
\
void PASTEMAC(ch,varname) \
     ( \
       conj_t         conja, \
       pack_t         schema, \
       dim_t          n, \
       void* restrict kappa, \
       void* restrict a, inc_t inca, inc_t lda, \
       void* restrict p,             inc_t ldp  \
     ) \
{ \
	const inc_t       inca2      = 2 * inca; \
	const inc_t       lda2       = 2 * lda; \
\
	ctype*            kappa_cast = kappa; \
	ctype*   restrict alpha1     = a; \
	ctype_r* restrict alpha1_r   = ( ctype_r* )a; \
	ctype_r* restrict alpha1_i   = ( ctype_r* )a + 1; \
	ctype_r* restrict pi1_r      = ( ctype_r* )p; \
\
\
	if ( bli_is_ro_packed( schema ) ) \
	{ \
		if ( PASTEMAC(ch,eq1)( *kappa_cast ) ) \
		{ \
			/* This works regardless of conja since we are only copying
			   the real part. */ \
			{ \
				for ( ; n != 0; --n ) \
				{ \
					PASTEMAC(chr,copys)( *(alpha1_r + 0*inca2), *(pi1_r + 0) ); \
					PASTEMAC(chr,copys)( *(alpha1_r + 1*inca2), *(pi1_r + 1) ); \
\
					alpha1_r += lda2; \
					pi1_r    += ldp; \
				} \
			} \
		} \
		else \
		{ \
			if ( bli_is_conj( conja ) ) \
			{ \
				for ( ; n != 0; --n ) \
				{ \
					PASTEMAC(ch,scal2jros)( *kappa_cast, *(alpha1 + 0*inca), *(pi1_r + 0) ); \
					PASTEMAC(ch,scal2jros)( *kappa_cast, *(alpha1 + 1*inca), *(pi1_r + 1) ); \
\
					alpha1 += lda; \
					pi1_r  += ldp; \
				} \
			} \
			else \
			{ \
				for ( ; n != 0; --n ) \
				{ \
					PASTEMAC(ch,scal2ros)( *kappa_cast, *(alpha1 + 0*inca), *(pi1_r + 0) ); \
					PASTEMAC(ch,scal2ros)( *kappa_cast, *(alpha1 + 1*inca), *(pi1_r + 1) ); \
\
					alpha1 += lda; \
					pi1_r  += ldp; \
				} \
			} \
		} \
	} \
	else if ( bli_is_io_packed( schema ) ) \
	{ \
		if ( PASTEMAC(ch,eq1)( *kappa_cast ) ) \
		{ \
			if ( bli_is_conj( conja ) ) \
			{ \
				for ( ; n != 0; --n ) \
				{ \
					PASTEMAC(chr,copys)( -*(alpha1_i + 0*inca2), *(pi1_r + 0) ); \
					PASTEMAC(chr,copys)( -*(alpha1_i + 1*inca2), *(pi1_r + 1) ); \
\
					alpha1_i += lda2; \
					pi1_r    += ldp; \
				} \
			} \
			else \
			{ \
				for ( ; n != 0; --n ) \
				{ \
					PASTEMAC(chr,copys)( *(alpha1_i + 0*inca2), *(pi1_r + 0) ); \
					PASTEMAC(chr,copys)( *(alpha1_i + 1*inca2), *(pi1_r + 1) ); \
\
					alpha1_i += lda2; \
					pi1_r    += ldp; \
				} \
			} \
		} \
		else \
		{ \
			if ( bli_is_conj( conja ) ) \
			{ \
				for ( ; n != 0; --n ) \
				{ \
					PASTEMAC(ch,scal2jios)( *kappa_cast, *(alpha1 + 0*inca), *(pi1_r + 0) ); \
					PASTEMAC(ch,scal2jios)( *kappa_cast, *(alpha1 + 1*inca), *(pi1_r + 1) ); \
\
					alpha1 += lda; \
					pi1_r  += ldp; \
				} \
			} \
			else \
			{ \
				for ( ; n != 0; --n ) \
				{ \
					PASTEMAC(ch,scal2ios)( *kappa_cast, *(alpha1 + 0*inca), *(pi1_r + 0) ); \
					PASTEMAC(ch,scal2ios)( *kappa_cast, *(alpha1 + 1*inca), *(pi1_r + 1) ); \
\
					alpha1 += lda; \
					pi1_r  += ldp; \
				} \
			} \
		} \
	} \
	else /* if ( bli_is_rpi_packed( schema ) ) */ \
	{ \
		if ( PASTEMAC(ch,eq1)( *kappa_cast ) ) \
		{ \
			if ( bli_is_conj( conja ) ) \
			{ \
				for ( ; n != 0; --n ) \
				{ \
					PASTEMAC(chr,add3s)( *(alpha1_r + 0*inca2), -*(alpha1_i + 0*inca2), *(pi1_r + 0) ); \
					PASTEMAC(chr,add3s)( *(alpha1_r + 1*inca2), -*(alpha1_i + 1*inca2), *(pi1_r + 1) ); \
\
					alpha1_r += lda2; \
					alpha1_i += lda2; \
					pi1_r    += ldp; \
				} \
			} \
			else \
			{ \
				for ( ; n != 0; --n ) \
				{ \
					PASTEMAC(chr,add3s)( *(alpha1_r + 0*inca2), *(alpha1_i + 0*inca2), *(pi1_r + 0) ); \
					PASTEMAC(chr,add3s)( *(alpha1_r + 1*inca2), *(alpha1_i + 1*inca2), *(pi1_r + 1) ); \
\
					alpha1_r += lda2; \
					alpha1_i += lda2; \
					pi1_r    += ldp; \
				} \
			} \
		} \
		else \
		{ \
			if ( bli_is_conj( conja ) ) \
			{ \
				for ( ; n != 0; --n ) \
				{ \
					PASTEMAC(ch,scal2jrpis)( *kappa_cast, *(alpha1 + 0*inca), *(pi1_r + 0) ); \
					PASTEMAC(ch,scal2jrpis)( *kappa_cast, *(alpha1 + 1*inca), *(pi1_r + 1) ); \
\
					alpha1 += lda; \
					pi1_r  += ldp; \
				} \
			} \
			else \
			{ \
				for ( ; n != 0; --n ) \
				{ \
					PASTEMAC(ch,scal2rpis)( *kappa_cast, *(alpha1 + 0*inca), *(pi1_r + 0) ); \
					PASTEMAC(ch,scal2rpis)( *kappa_cast, *(alpha1 + 1*inca), *(pi1_r + 1) ); \
\
					alpha1 += lda; \
					pi1_r  += ldp; \
				} \
			} \
		} \
	} \
}

INSERT_GENTFUNCCO_BASIC0( packm_2xk_rih_ref )



#undef  GENTFUNCCO
#define GENTFUNCCO( ctype, ctype_r, ch, chr, varname ) \
\
void PASTEMAC(ch,varname) \
     ( \
       conj_t         conja, \
       pack_t         schema, \
       dim_t          n, \
       void* restrict kappa, \
       void* restrict a, inc_t inca, inc_t lda, \
       void* restrict p,             inc_t ldp  \
     ) \
{ \
	const inc_t       inca2      = 2 * inca; \
	const inc_t       lda2       = 2 * lda; \
\
	ctype*            kappa_cast = kappa; \
	ctype*   restrict alpha1     = a; \
	ctype_r* restrict alpha1_r   = ( ctype_r* )a; \
	ctype_r* restrict alpha1_i   = ( ctype_r* )a + 1; \
	ctype_r* restrict pi1_r      = ( ctype_r* )p; \
\
\
	if ( bli_is_ro_packed( schema ) ) \
	{ \
		if ( PASTEMAC(ch,eq1)( *kappa_cast ) ) \
		{ \
			/* This works regardless of conja since we are only copying
			   the real part. */ \
			{ \
				for ( ; n != 0; --n ) \
				{ \
					PASTEMAC(chr,copys)( *(alpha1_r + 0*inca2), *(pi1_r + 0) ); \
					PASTEMAC(chr,copys)( *(alpha1_r + 1*inca2), *(pi1_r + 1) ); \
					PASTEMAC(chr,copys)( *(alpha1_r + 2*inca2), *(pi1_r + 2) ); \
					PASTEMAC(chr,copys)( *(alpha1_r + 3*inca2), *(pi1_r + 3) ); \
\
					alpha1_r += lda2; \
					pi1_r    += ldp; \
				} \
			} \
		} \
		else \
		{ \
			if ( bli_is_conj( conja ) ) \
			{ \
				for ( ; n != 0; --n ) \
				{ \
					PASTEMAC(ch,scal2jros)( *kappa_cast, *(alpha1 + 0*inca), *(pi1_r + 0) ); \
					PASTEMAC(ch,scal2jros)( *kappa_cast, *(alpha1 + 1*inca), *(pi1_r + 1) ); \
					PASTEMAC(ch,scal2jros)( *kappa_cast, *(alpha1 + 2*inca), *(pi1_r + 2) ); \
					PASTEMAC(ch,scal2jros)( *kappa_cast, *(alpha1 + 3*inca), *(pi1_r + 3) ); \
\
					alpha1 += lda; \
					pi1_r  += ldp; \
				} \
			} \
			else \
			{ \
				for ( ; n != 0; --n ) \
				{ \
					PASTEMAC(ch,scal2ros)( *kappa_cast, *(alpha1 + 0*inca), *(pi1_r + 0) ); \
					PASTEMAC(ch,scal2ros)( *kappa_cast, *(alpha1 + 1*inca), *(pi1_r + 1) ); \
					PASTEMAC(ch,scal2ros)( *kappa_cast, *(alpha1 + 2*inca), *(pi1_r + 2) ); \
					PASTEMAC(ch,scal2ros)( *kappa_cast, *(alpha1 + 3*inca), *(pi1_r + 3) ); \
\
					alpha1 += lda; \
					pi1_r  += ldp; \
				} \
			} \
		} \
	} \
	else if ( bli_is_io_packed( schema ) ) \
	{ \
		if ( PASTEMAC(ch,eq1)( *kappa_cast ) ) \
		{ \
			if ( bli_is_conj( conja ) ) \
			{ \
				for ( ; n != 0; --n ) \
				{ \
					PASTEMAC(chr,copys)( -*(alpha1_i + 0*inca2), *(pi1_r + 0) ); \
					PASTEMAC(chr,copys)( -*(alpha1_i + 1*inca2), *(pi1_r + 1) ); \
					PASTEMAC(chr,copys)( -*(alpha1_i + 2*inca2), *(pi1_r + 2) ); \
					PASTEMAC(chr,copys)( -*(alpha1_i + 3*inca2), *(pi1_r + 3) ); \
\
					alpha1_i += lda2; \
					pi1_r    += ldp; \
				} \
			} \
			else \
			{ \
				for ( ; n != 0; --n ) \
				{ \
					PASTEMAC(chr,copys)( *(alpha1_i + 0*inca2), *(pi1_r + 0) ); \
					PASTEMAC(chr,copys)( *(alpha1_i + 1*inca2), *(pi1_r + 1) ); \
					PASTEMAC(chr,copys)( *(alpha1_i + 2*inca2), *(pi1_r + 2) ); \
					PASTEMAC(chr,copys)( *(alpha1_i + 3*inca2), *(pi1_r + 3) ); \
\
					alpha1_i += lda2; \
					pi1_r    += ldp; \
				} \
			} \
		} \
		else \
		{ \
			if ( bli_is_conj( conja ) ) \
			{ \
				for ( ; n != 0; --n ) \
				{ \
					PASTEMAC(ch,scal2jios)( *kappa_cast, *(alpha1 + 0*inca), *(pi1_r + 0) ); \
					PASTEMAC(ch,scal2jios)( *kappa_cast, *(alpha1 + 1*inca), *(pi1_r + 1) ); \
					PASTEMAC(ch,scal2jios)( *kappa_cast, *(alpha1 + 2*inca), *(pi1_r + 2) ); \
					PASTEMAC(ch,scal2jios)( *kappa_cast, *(alpha1 + 3*inca), *(pi1_r + 3) ); \
\
					alpha1 += lda; \
					pi1_r  += ldp; \
				} \
			} \
			else \
			{ \
				for ( ; n != 0; --n ) \
				{ \
					PASTEMAC(ch,scal2ios)( *kappa_cast, *(alpha1 + 0*inca), *(pi1_r + 0) ); \
					PASTEMAC(ch,scal2ios)( *kappa_cast, *(alpha1 + 1*inca), *(pi1_r + 1) ); \
					PASTEMAC(ch,scal2ios)( *kappa_cast, *(alpha1 + 2*inca), *(pi1_r + 2) ); \
					PASTEMAC(ch,scal2ios)( *kappa_cast, *(alpha1 + 3*inca), *(pi1_r + 3) ); \
\
					alpha1 += lda; \
					pi1_r  += ldp; \
				} \
			} \
		} \
	} \
	else /* if ( bli_is_rpi_packed( schema ) ) */ \
	{ \
		if ( PASTEMAC(ch,eq1)( *kappa_cast ) ) \
		{ \
			if ( bli_is_conj( conja ) ) \
			{ \
				for ( ; n != 0; --n ) \
				{ \
					PASTEMAC(chr,add3s)( *(alpha1_r + 0*inca2), -*(alpha1_i + 0*inca2), *(pi1_r + 0) ); \
					PASTEMAC(chr,add3s)( *(alpha1_r + 1*inca2), -*(alpha1_i + 1*inca2), *(pi1_r + 1) ); \
					PASTEMAC(chr,add3s)( *(alpha1_r + 2*inca2), -*(alpha1_i + 2*inca2), *(pi1_r + 2) ); \
					PASTEMAC(chr,add3s)( *(alpha1_r + 3*inca2), -*(alpha1_i + 3*inca2), *(pi1_r + 3) ); \
\
					alpha1_r += lda2; \
					alpha1_i += lda2; \
					pi1_r    += ldp; \
				} \
			} \
			else \
			{ \
				for ( ; n != 0; --n ) \
				{ \
					PASTEMAC(chr,add3s)( *(alpha1_r + 0*inca2), *(alpha1_i + 0*inca2), *(pi1_r + 0) ); \
					PASTEMAC(chr,add3s)( *(alpha1_r + 1*inca2), *(alpha1_i + 1*inca2), *(pi1_r + 1) ); \
					PASTEMAC(chr,add3s)( *(alpha1_r + 2*inca2), *(alpha1_i + 2*inca2), *(pi1_r + 2) ); \
					PASTEMAC(chr,add3s)( *(alpha1_r + 3*inca2), *(alpha1_i + 3*inca2), *(pi1_r + 3) ); \
\
					alpha1_r += lda2; \
					alpha1_i += lda2; \
					pi1_r    += ldp; \
				} \
			} \
		} \
		else \
		{ \
			if ( bli_is_conj( conja ) ) \
			{ \
				for ( ; n != 0; --n ) \
				{ \
					PASTEMAC(ch,scal2jrpis)( *kappa_cast, *(alpha1 + 0*inca), *(pi1_r + 0) ); \
					PASTEMAC(ch,scal2jrpis)( *kappa_cast, *(alpha1 + 1*inca), *(pi1_r + 1) ); \
					PASTEMAC(ch,scal2jrpis)( *kappa_cast, *(alpha1 + 2*inca), *(pi1_r + 2) ); \
					PASTEMAC(ch,scal2jrpis)( *kappa_cast, *(alpha1 + 3*inca), *(pi1_r + 3) ); \
\
					alpha1 += lda; \
					pi1_r  += ldp; \
				} \
			} \
			else \
			{ \
				for ( ; n != 0; --n ) \
				{ \
					PASTEMAC(ch,scal2rpis)( *kappa_cast, *(alpha1 + 0*inca), *(pi1_r + 0) ); \
					PASTEMAC(ch,scal2rpis)( *kappa_cast, *(alpha1 + 1*inca), *(pi1_r + 1) ); \
					PASTEMAC(ch,scal2rpis)( *kappa_cast, *(alpha1 + 2*inca), *(pi1_r + 2) ); \
					PASTEMAC(ch,scal2rpis)( *kappa_cast, *(alpha1 + 3*inca), *(pi1_r + 3) ); \
\
					alpha1 += lda; \
					pi1_r  += ldp; \
				} \
			} \
		} \
	} \
}

INSERT_GENTFUNCCO_BASIC0( packm_4xk_rih_ref )



#undef  GENTFUNCCO
#define GENTFUNCCO( ctype, ctype_r, ch, chr, varname ) \
\
void PASTEMAC(ch,varname) \
     ( \
       conj_t         conja, \
       pack_t         schema, \
       dim_t          n, \
       void* restrict kappa, \
       void* restrict a, inc_t inca, inc_t lda, \
       void* restrict p,             inc_t ldp  \
     ) \
{ \
	const inc_t       inca2      = 2 * inca; \
	const inc_t       lda2       = 2 * lda; \
\
	ctype*            kappa_cast = kappa; \
	ctype*   restrict alpha1     = a; \
	ctype_r* restrict alpha1_r   = ( ctype_r* )a; \
	ctype_r* restrict alpha1_i   = ( ctype_r* )a + 1; \
	ctype_r* restrict pi1_r      = ( ctype_r* )p; \
\
\
	if ( bli_is_ro_packed( schema ) ) \
	{ \
		if ( PASTEMAC(ch,eq1)( *kappa_cast ) ) \
		{ \
			/* This works regardless of conja since we are only copying
			   the real part. */ \
			{ \
				for ( ; n != 0; --n ) \
				{ \
					PASTEMAC(chr,copys)( *(alpha1_r + 0*inca2), *(pi1_r + 0) ); \
					PASTEMAC(chr,copys)( *(alpha1_r + 1*inca2), *(pi1_r + 1) ); \
					PASTEMAC(chr,copys)( *(alpha1_r + 2*inca2), *(pi1_r + 2) ); \
					PASTEMAC(chr,copys)( *(alpha1_r + 3*inca2), *(pi1_r + 3) ); \
					PASTEMAC(chr,copys)( *(alpha1_r + 4*inca2), *(pi1_r + 4) ); \
					PASTEMAC(chr,copys)( *(alpha1_r + 5*inca2), *(pi1_r + 5) ); \
\
					alpha1_r += lda2; \
					pi1_r    += ldp; \
				} \
			} \
		} \
		else \
		{ \
			if ( bli_is_conj( conja ) ) \
			{ \
				for ( ; n != 0; --n ) \
				{ \
					PASTEMAC(ch,scal2jros)( *kappa_cast, *(alpha1 + 0*inca), *(pi1_r + 0) ); \
					PASTEMAC(ch,scal2jros)( *kappa_cast, *(alpha1 + 1*inca), *(pi1_r + 1) ); \
					PASTEMAC(ch,scal2jros)( *kappa_cast, *(alpha1 + 2*inca), *(pi1_r + 2) ); \
					PASTEMAC(ch,scal2jros)( *kappa_cast, *(alpha1 + 3*inca), *(pi1_r + 3) ); \
					PASTEMAC(ch,scal2jros)( *kappa_cast, *(alpha1 + 4*inca), *(pi1_r + 4) ); \
					PASTEMAC(ch,scal2jros)( *kappa_cast, *(alpha1 + 5*inca), *(pi1_r + 5) ); \
\
					alpha1 += lda; \
					pi1_r  += ldp; \
				} \
			} \
			else \
			{ \
				for ( ; n != 0; --n ) \
				{ \
					PASTEMAC(ch,scal2ros)( *kappa_cast, *(alpha1 + 0*inca), *(pi1_r + 0) ); \
					PASTEMAC(ch,scal2ros)( *kappa_cast, *(alpha1 + 1*inca), *(pi1_r + 1) ); \
					PASTEMAC(ch,scal2ros)( *kappa_cast, *(alpha1 + 2*inca), *(pi1_r + 2) ); \
					PASTEMAC(ch,scal2ros)( *kappa_cast, *(alpha1 + 3*inca), *(pi1_r + 3) ); \
					PASTEMAC(ch,scal2ros)( *kappa_cast, *(alpha1 + 4*inca), *(pi1_r + 4) ); \
					PASTEMAC(ch,scal2ros)( *kappa_cast, *(alpha1 + 5*inca), *(pi1_r + 5) ); \
\
					alpha1 += lda; \
					pi1_r  += ldp; \
				} \
			} \
		} \
	} \
	else if ( bli_is_io_packed( schema ) ) \
	{ \
		if ( PASTEMAC(ch,eq1)( *kappa_cast ) ) \
		{ \
			if ( bli_is_conj( conja ) ) \
			{ \
				for ( ; n != 0; --n ) \
				{ \
					PASTEMAC(chr,copys)( -*(alpha1_i + 0*inca2), *(pi1_r + 0) ); \
					PASTEMAC(chr,copys)( -*(alpha1_i + 1*inca2), *(pi1_r + 1) ); \
					PASTEMAC(chr,copys)( -*(alpha1_i + 2*inca2), *(pi1_r + 2) ); \
					PASTEMAC(chr,copys)( -*(alpha1_i + 3*inca2), *(pi1_r + 3) ); \
					PASTEMAC(chr,copys)( -*(alpha1_i + 4*inca2), *(pi1_r + 4) ); \
					PASTEMAC(chr,copys)( -*(alpha1_i + 5*inca2), *(pi1_r + 5) ); \
\
					alpha1_i += lda2; \
					pi1_r    += ldp; \
				} \
			} \
			else \
			{ \
				for ( ; n != 0; --n ) \
				{ \
					PASTEMAC(chr,copys)( *(alpha1_i + 0*inca2), *(pi1_r + 0) ); \
					PASTEMAC(chr,copys)( *(alpha1_i + 1*inca2), *(pi1_r + 1) ); \
					PASTEMAC(chr,copys)( *(alpha1_i + 2*inca2), *(pi1_r + 2) ); \
					PASTEMAC(chr,copys)( *(alpha1_i + 3*inca2), *(pi1_r + 3) ); \
					PASTEMAC(chr,copys)( *(alpha1_i + 4*inca2), *(pi1_r + 4) ); \
					PASTEMAC(chr,copys)( *(alpha1_i + 5*inca2), *(pi1_r + 5) ); \
\
					alpha1_i += lda2; \
					pi1_r    += ldp; \
				} \
			} \
		} \
		else \
		{ \
			if ( bli_is_conj( conja ) ) \
			{ \
				for ( ; n != 0; --n ) \
				{ \
					PASTEMAC(ch,scal2jios)( *kappa_cast, *(alpha1 + 0*inca), *(pi1_r + 0) ); \
					PASTEMAC(ch,scal2jios)( *kappa_cast, *(alpha1 + 1*inca), *(pi1_r + 1) ); \
					PASTEMAC(ch,scal2jios)( *kappa_cast, *(alpha1 + 2*inca), *(pi1_r + 2) ); \
					PASTEMAC(ch,scal2jios)( *kappa_cast, *(alpha1 + 3*inca), *(pi1_r + 3) ); \
					PASTEMAC(ch,scal2jios)( *kappa_cast, *(alpha1 + 4*inca), *(pi1_r + 4) ); \
					PASTEMAC(ch,scal2jios)( *kappa_cast, *(alpha1 + 5*inca), *(pi1_r + 5) ); \
\
					alpha1 += lda; \
					pi1_r  += ldp; \
				} \
			} \
			else \
			{ \
				for ( ; n != 0; --n ) \
				{ \
					PASTEMAC(ch,scal2ios)( *kappa_cast, *(alpha1 + 0*inca), *(pi1_r + 0) ); \
					PASTEMAC(ch,scal2ios)( *kappa_cast, *(alpha1 + 1*inca), *(pi1_r + 1) ); \
					PASTEMAC(ch,scal2ios)( *kappa_cast, *(alpha1 + 2*inca), *(pi1_r + 2) ); \
					PASTEMAC(ch,scal2ios)( *kappa_cast, *(alpha1 + 3*inca), *(pi1_r + 3) ); \
					PASTEMAC(ch,scal2ios)( *kappa_cast, *(alpha1 + 4*inca), *(pi1_r + 4) ); \
					PASTEMAC(ch,scal2ios)( *kappa_cast, *(alpha1 + 5*inca), *(pi1_r + 5) ); \
\
					alpha1 += lda; \
					pi1_r  += ldp; \
				} \
			} \
		} \
	} \
	else /* if ( bli_is_rpi_packed( schema ) ) */ \
	{ \
		if ( PASTEMAC(ch,eq1)( *kappa_cast ) ) \
		{ \
			if ( bli_is_conj( conja ) ) \
			{ \
				for ( ; n != 0; --n ) \
				{ \
					PASTEMAC(chr,add3s)( *(alpha1_r + 0*inca2), -*(alpha1_i + 0*inca2), *(pi1_r + 0) ); \
					PASTEMAC(chr,add3s)( *(alpha1_r + 1*inca2), -*(alpha1_i + 1*inca2), *(pi1_r + 1) ); \
					PASTEMAC(chr,add3s)( *(alpha1_r + 2*inca2), -*(alpha1_i + 2*inca2), *(pi1_r + 2) ); \
					PASTEMAC(chr,add3s)( *(alpha1_r + 3*inca2), -*(alpha1_i + 3*inca2), *(pi1_r + 3) ); \
					PASTEMAC(chr,add3s)( *(alpha1_r + 4*inca2), -*(alpha1_i + 4*inca2), *(pi1_r + 4) ); \
					PASTEMAC(chr,add3s)( *(alpha1_r + 5*inca2), -*(alpha1_i + 5*inca2), *(pi1_r + 5) ); \
\
					alpha1_r += lda2; \
					alpha1_i += lda2; \
					pi1_r    += ldp; \
				} \
			} \
			else \
			{ \
				for ( ; n != 0; --n ) \
				{ \
					PASTEMAC(chr,add3s)( *(alpha1_r + 0*inca2), *(alpha1_i + 0*inca2), *(pi1_r + 0) ); \
					PASTEMAC(chr,add3s)( *(alpha1_r + 1*inca2), *(alpha1_i + 1*inca2), *(pi1_r + 1) ); \
					PASTEMAC(chr,add3s)( *(alpha1_r + 2*inca2), *(alpha1_i + 2*inca2), *(pi1_r + 2) ); \
					PASTEMAC(chr,add3s)( *(alpha1_r + 3*inca2), *(alpha1_i + 3*inca2), *(pi1_r + 3) ); \
					PASTEMAC(chr,add3s)( *(alpha1_r + 4*inca2), *(alpha1_i + 4*inca2), *(pi1_r + 4) ); \
					PASTEMAC(chr,add3s)( *(alpha1_r + 5*inca2), *(alpha1_i + 5*inca2), *(pi1_r + 5) ); \
\
					alpha1_r += lda2; \
					alpha1_i += lda2; \
					pi1_r    += ldp; \
				} \
			} \
		} \
		else \
		{ \
			if ( bli_is_conj( conja ) ) \
			{ \
				for ( ; n != 0; --n ) \
				{ \
					PASTEMAC(ch,scal2jrpis)( *kappa_cast, *(alpha1 + 0*inca), *(pi1_r + 0) ); \
					PASTEMAC(ch,scal2jrpis)( *kappa_cast, *(alpha1 + 1*inca), *(pi1_r + 1) ); \
					PASTEMAC(ch,scal2jrpis)( *kappa_cast, *(alpha1 + 2*inca), *(pi1_r + 2) ); \
					PASTEMAC(ch,scal2jrpis)( *kappa_cast, *(alpha1 + 3*inca), *(pi1_r + 3) ); \
					PASTEMAC(ch,scal2jrpis)( *kappa_cast, *(alpha1 + 4*inca), *(pi1_r + 4) ); \
					PASTEMAC(ch,scal2jrpis)( *kappa_cast, *(alpha1 + 5*inca), *(pi1_r + 5) ); \
\
					alpha1 += lda; \
					pi1_r  += ldp; \
				} \
			} \
			else \
			{ \
				for ( ; n != 0; --n ) \
				{ \
					PASTEMAC(ch,scal2rpis)( *kappa_cast, *(alpha1 + 0*inca), *(pi1_r + 0) ); \
					PASTEMAC(ch,scal2rpis)( *kappa_cast, *(alpha1 + 1*inca), *(pi1_r + 1) ); \
					PASTEMAC(ch,scal2rpis)( *kappa_cast, *(alpha1 + 2*inca), *(pi1_r + 2) ); \
					PASTEMAC(ch,scal2rpis)( *kappa_cast, *(alpha1 + 3*inca), *(pi1_r + 3) ); \
					PASTEMAC(ch,scal2rpis)( *kappa_cast, *(alpha1 + 4*inca), *(pi1_r + 4) ); \
					PASTEMAC(ch,scal2rpis)( *kappa_cast, *(alpha1 + 5*inca), *(pi1_r + 5) ); \
\
					alpha1 += lda; \
					pi1_r  += ldp; \
				} \
			} \
		} \
	} \
}

INSERT_GENTFUNCCO_BASIC0( packm_6xk_rih_ref )



#undef  GENTFUNCCO
#define GENTFUNCCO( ctype, ctype_r, ch, chr, varname ) \
\
void PASTEMAC(ch,varname) \
     ( \
       conj_t         conja, \
       pack_t         schema, \
       dim_t          n, \
       void* restrict kappa, \
       void* restrict a, inc_t inca, inc_t lda, \
       void* restrict p,             inc_t ldp  \
     ) \
{ \
	const inc_t       inca2      = 2 * inca; \
	const inc_t       lda2       = 2 * lda; \
\
	ctype*            kappa_cast = kappa; \
	ctype*   restrict alpha1     = a; \
	ctype_r* restrict alpha1_r   = ( ctype_r* )a; \
	ctype_r* restrict alpha1_i   = ( ctype_r* )a + 1; \
	ctype_r* restrict pi1_r      = ( ctype_r* )p; \
\
\
	if ( bli_is_ro_packed( schema ) ) \
	{ \
		if ( PASTEMAC(ch,eq1)( *kappa_cast ) ) \
		{ \
			/* This works regardless of conja since we are only copying
			   the real part. */ \
			{ \
				for ( ; n != 0; --n ) \
				{ \
					PASTEMAC(chr,copys)( *(alpha1_r + 0*inca2), *(pi1_r + 0) ); \
					PASTEMAC(chr,copys)( *(alpha1_r + 1*inca2), *(pi1_r + 1) ); \
					PASTEMAC(chr,copys)( *(alpha1_r + 2*inca2), *(pi1_r + 2) ); \
					PASTEMAC(chr,copys)( *(alpha1_r + 3*inca2), *(pi1_r + 3) ); \
					PASTEMAC(chr,copys)( *(alpha1_r + 4*inca2), *(pi1_r + 4) ); \
					PASTEMAC(chr,copys)( *(alpha1_r + 5*inca2), *(pi1_r + 5) ); \
					PASTEMAC(chr,copys)( *(alpha1_r + 6*inca2), *(pi1_r + 6) ); \
					PASTEMAC(chr,copys)( *(alpha1_r + 7*inca2), *(pi1_r + 7) ); \
\
					alpha1_r += lda2; \
					pi1_r    += ldp; \
				} \
			} \
		} \
		else \
		{ \
			if ( bli_is_conj( conja ) ) \
			{ \
				for ( ; n != 0; --n ) \
				{ \
					PASTEMAC(ch,scal2jros)( *kappa_cast, *(alpha1 + 0*inca), *(pi1_r + 0) ); \
					PASTEMAC(ch,scal2jros)( *kappa_cast, *(alpha1 + 1*inca), *(pi1_r + 1) ); \
					PASTEMAC(ch,scal2jros)( *kappa_cast, *(alpha1 + 2*inca), *(pi1_r + 2) ); \
					PASTEMAC(ch,scal2jros)( *kappa_cast, *(alpha1 + 3*inca), *(pi1_r + 3) ); \
					PASTEMAC(ch,scal2jros)( *kappa_cast, *(alpha1 + 4*inca), *(pi1_r + 4) ); \
					PASTEMAC(ch,scal2jros)( *kappa_cast, *(alpha1 + 5*inca), *(pi1_r + 5) ); \
					PASTEMAC(ch,scal2jros)( *kappa_cast, *(alpha1 + 6*inca), *(pi1_r + 6) ); \
					PASTEMAC(ch,scal2jros)( *kappa_cast, *(alpha1 + 7*inca), *(pi1_r + 7) ); \
\
					alpha1 += lda; \
					pi1_r  += ldp; \
				} \
			} \
			else \
			{ \
				for ( ; n != 0; --n ) \
				{ \
					PASTEMAC(ch,scal2ros)( *kappa_cast, *(alpha1 + 0*inca), *(pi1_r + 0) ); \
					PASTEMAC(ch,scal2ros)( *kappa_cast, *(alpha1 + 1*inca), *(pi1_r + 1) ); \
					PASTEMAC(ch,scal2ros)( *kappa_cast, *(alpha1 + 2*inca), *(pi1_r + 2) ); \
					PASTEMAC(ch,scal2ros)( *kappa_cast, *(alpha1 + 3*inca), *(pi1_r + 3) ); \
					PASTEMAC(ch,scal2ros)( *kappa_cast, *(alpha1 + 4*inca), *(pi1_r + 4) ); \
					PASTEMAC(ch,scal2ros)( *kappa_cast, *(alpha1 + 5*inca), *(pi1_r + 5) ); \
					PASTEMAC(ch,scal2ros)( *kappa_cast, *(alpha1 + 6*inca), *(pi1_r + 6) ); \
					PASTEMAC(ch,scal2ros)( *kappa_cast, *(alpha1 + 7*inca), *(pi1_r + 7) ); \
\
					alpha1 += lda; \
					pi1_r  += ldp; \
				} \
			} \
		} \
	} \
	else if ( bli_is_io_packed( schema ) ) \
	{ \
		if ( PASTEMAC(ch,eq1)( *kappa_cast ) ) \
		{ \
			if ( bli_is_conj( conja ) ) \
			{ \
				for ( ; n != 0; --n ) \
				{ \
					PASTEMAC(chr,copys)( -*(alpha1_i + 0*inca2), *(pi1_r + 0) ); \
					PASTEMAC(chr,copys)( -*(alpha1_i + 1*inca2), *(pi1_r + 1) ); \
					PASTEMAC(chr,copys)( -*(alpha1_i + 2*inca2), *(pi1_r + 2) ); \
					PASTEMAC(chr,copys)( -*(alpha1_i + 3*inca2), *(pi1_r + 3) ); \
					PASTEMAC(chr,copys)( -*(alpha1_i + 4*inca2), *(pi1_r + 4) ); \
					PASTEMAC(chr,copys)( -*(alpha1_i + 5*inca2), *(pi1_r + 5) ); \
					PASTEMAC(chr,copys)( -*(alpha1_i + 6*inca2), *(pi1_r + 6) ); \
					PASTEMAC(chr,copys)( -*(alpha1_i + 7*inca2), *(pi1_r + 7) ); \
\
					alpha1_i += lda2; \
					pi1_r    += ldp; \
				} \
			} \
			else \
			{ \
				for ( ; n != 0; --n ) \
				{ \
					PASTEMAC(chr,copys)( *(alpha1_i + 0*inca2), *(pi1_r + 0) ); \
					PASTEMAC(chr,copys)( *(alpha1_i + 1*inca2), *(pi1_r + 1) ); \
					PASTEMAC(chr,copys)( *(alpha1_i + 2*inca2), *(pi1_r + 2) ); \
					PASTEMAC(chr,copys)( *(alpha1_i + 3*inca2), *(pi1_r + 3) ); \
					PASTEMAC(chr,copys)( *(alpha1_i + 4*inca2), *(pi1_r + 4) ); \
					PASTEMAC(chr,copys)( *(alpha1_i + 5*inca2), *(pi1_r + 5) ); \
					PASTEMAC(chr,copys)( *(alpha1_i + 6*inca2), *(pi1_r + 6) ); \
					PASTEMAC(chr,copys)( *(alpha1_i + 7*inca2), *(pi1_r + 7) ); \
\
					alpha1_i += lda2; \
					pi1_r    += ldp; \
				} \
			} \
		} \
		else \
		{ \
			if ( bli_is_conj( conja ) ) \
			{ \
				for ( ; n != 0; --n ) \
				{ \
					PASTEMAC(ch,scal2jios)( *kappa_cast, *(alpha1 + 0*inca), *(pi1_r + 0) ); \
					PASTEMAC(ch,scal2jios)( *kappa_cast, *(alpha1 + 1*inca), *(pi1_r + 1) ); \
					PASTEMAC(ch,scal2jios)( *kappa_cast, *(alpha1 + 2*inca), *(pi1_r + 2) ); \
					PASTEMAC(ch,scal2jios)( *kappa_cast, *(alpha1 + 3*inca), *(pi1_r + 3) ); \
					PASTEMAC(ch,scal2jios)( *kappa_cast, *(alpha1 + 4*inca), *(pi1_r + 4) ); \
					PASTEMAC(ch,scal2jios)( *kappa_cast, *(alpha1 + 5*inca), *(pi1_r + 5) ); \
					PASTEMAC(ch,scal2jios)( *kappa_cast, *(alpha1 + 6*inca), *(pi1_r + 6) ); \
					PASTEMAC(ch,scal2jios)( *kappa_cast, *(alpha1 + 7*inca), *(pi1_r + 7) ); \
\
					alpha1 += lda; \
					pi1_r  += ldp; \
				} \
			} \
			else \
			{ \
				for ( ; n != 0; --n ) \
				{ \
					PASTEMAC(ch,scal2ios)( *kappa_cast, *(alpha1 + 0*inca), *(pi1_r + 0) ); \
					PASTEMAC(ch,scal2ios)( *kappa_cast, *(alpha1 + 1*inca), *(pi1_r + 1) ); \
					PASTEMAC(ch,scal2ios)( *kappa_cast, *(alpha1 + 2*inca), *(pi1_r + 2) ); \
					PASTEMAC(ch,scal2ios)( *kappa_cast, *(alpha1 + 3*inca), *(pi1_r + 3) ); \
					PASTEMAC(ch,scal2ios)( *kappa_cast, *(alpha1 + 4*inca), *(pi1_r + 4) ); \
					PASTEMAC(ch,scal2ios)( *kappa_cast, *(alpha1 + 5*inca), *(pi1_r + 5) ); \
					PASTEMAC(ch,scal2ios)( *kappa_cast, *(alpha1 + 6*inca), *(pi1_r + 6) ); \
					PASTEMAC(ch,scal2ios)( *kappa_cast, *(alpha1 + 7*inca), *(pi1_r + 7) ); \
\
					alpha1 += lda; \
					pi1_r  += ldp; \
				} \
			} \
		} \
	} \
	else /* if ( bli_is_rpi_packed( schema ) ) */ \
	{ \
		if ( PASTEMAC(ch,eq1)( *kappa_cast ) ) \
		{ \
			if ( bli_is_conj( conja ) ) \
			{ \
				for ( ; n != 0; --n ) \
				{ \
					PASTEMAC(chr,add3s)( *(alpha1_r + 0*inca2), -*(alpha1_i + 0*inca2), *(pi1_r + 0) ); \
					PASTEMAC(chr,add3s)( *(alpha1_r + 1*inca2), -*(alpha1_i + 1*inca2), *(pi1_r + 1) ); \
					PASTEMAC(chr,add3s)( *(alpha1_r + 2*inca2), -*(alpha1_i + 2*inca2), *(pi1_r + 2) ); \
					PASTEMAC(chr,add3s)( *(alpha1_r + 3*inca2), -*(alpha1_i + 3*inca2), *(pi1_r + 3) ); \
					PASTEMAC(chr,add3s)( *(alpha1_r + 4*inca2), -*(alpha1_i + 4*inca2), *(pi1_r + 4) ); \
					PASTEMAC(chr,add3s)( *(alpha1_r + 5*inca2), -*(alpha1_i + 5*inca2), *(pi1_r + 5) ); \
					PASTEMAC(chr,add3s)( *(alpha1_r + 6*inca2), -*(alpha1_i + 6*inca2), *(pi1_r + 6) ); \
					PASTEMAC(chr,add3s)( *(alpha1_r + 7*inca2), -*(alpha1_i + 7*inca2), *(pi1_r + 7) ); \
\
					alpha1_r += lda2; \
					alpha1_i += lda2; \
					pi1_r    += ldp; \
				} \
			} \
			else \
			{ \
				for ( ; n != 0; --n ) \
				{ \
					PASTEMAC(chr,add3s)( *(alpha1_r + 0*inca2), *(alpha1_i + 0*inca2), *(pi1_r + 0) ); \
					PASTEMAC(chr,add3s)( *(alpha1_r + 1*inca2), *(alpha1_i + 1*inca2), *(pi1_r + 1) ); \
					PASTEMAC(chr,add3s)( *(alpha1_r + 2*inca2), *(alpha1_i + 2*inca2), *(pi1_r + 2) ); \
					PASTEMAC(chr,add3s)( *(alpha1_r + 3*inca2), *(alpha1_i + 3*inca2), *(pi1_r + 3) ); \
					PASTEMAC(chr,add3s)( *(alpha1_r + 4*inca2), *(alpha1_i + 4*inca2), *(pi1_r + 4) ); \
					PASTEMAC(chr,add3s)( *(alpha1_r + 5*inca2), *(alpha1_i + 5*inca2), *(pi1_r + 5) ); \
					PASTEMAC(chr,add3s)( *(alpha1_r + 6*inca2), *(alpha1_i + 6*inca2), *(pi1_r + 6) ); \
					PASTEMAC(chr,add3s)( *(alpha1_r + 7*inca2), *(alpha1_i + 7*inca2), *(pi1_r + 7) ); \
\
					alpha1_r += lda2; \
					alpha1_i += lda2; \
					pi1_r    += ldp; \
				} \
			} \
		} \
		else \
		{ \
			if ( bli_is_conj( conja ) ) \
			{ \
				for ( ; n != 0; --n ) \
				{ \
					PASTEMAC(ch,scal2jrpis)( *kappa_cast, *(alpha1 + 0*inca), *(pi1_r + 0) ); \
					PASTEMAC(ch,scal2jrpis)( *kappa_cast, *(alpha1 + 1*inca), *(pi1_r + 1) ); \
					PASTEMAC(ch,scal2jrpis)( *kappa_cast, *(alpha1 + 2*inca), *(pi1_r + 2) ); \
					PASTEMAC(ch,scal2jrpis)( *kappa_cast, *(alpha1 + 3*inca), *(pi1_r + 3) ); \
					PASTEMAC(ch,scal2jrpis)( *kappa_cast, *(alpha1 + 4*inca), *(pi1_r + 4) ); \
					PASTEMAC(ch,scal2jrpis)( *kappa_cast, *(alpha1 + 5*inca), *(pi1_r + 5) ); \
					PASTEMAC(ch,scal2jrpis)( *kappa_cast, *(alpha1 + 6*inca), *(pi1_r + 6) ); \
					PASTEMAC(ch,scal2jrpis)( *kappa_cast, *(alpha1 + 7*inca), *(pi1_r + 7) ); \
\
					alpha1 += lda; \
					pi1_r  += ldp; \
				} \
			} \
			else \
			{ \
				for ( ; n != 0; --n ) \
				{ \
					PASTEMAC(ch,scal2rpis)( *kappa_cast, *(alpha1 + 0*inca), *(pi1_r + 0) ); \
					PASTEMAC(ch,scal2rpis)( *kappa_cast, *(alpha1 + 1*inca), *(pi1_r + 1) ); \
					PASTEMAC(ch,scal2rpis)( *kappa_cast, *(alpha1 + 2*inca), *(pi1_r + 2) ); \
					PASTEMAC(ch,scal2rpis)( *kappa_cast, *(alpha1 + 3*inca), *(pi1_r + 3) ); \
					PASTEMAC(ch,scal2rpis)( *kappa_cast, *(alpha1 + 4*inca), *(pi1_r + 4) ); \
					PASTEMAC(ch,scal2rpis)( *kappa_cast, *(alpha1 + 5*inca), *(pi1_r + 5) ); \
					PASTEMAC(ch,scal2rpis)( *kappa_cast, *(alpha1 + 6*inca), *(pi1_r + 6) ); \
					PASTEMAC(ch,scal2rpis)( *kappa_cast, *(alpha1 + 7*inca), *(pi1_r + 7) ); \
\
					alpha1 += lda; \
					pi1_r  += ldp; \
				} \
			} \
		} \
	} \
}

INSERT_GENTFUNCCO_BASIC0( packm_8xk_rih_ref )



#undef  GENTFUNCCO
#define GENTFUNCCO( ctype, ctype_r, ch, chr, varname ) \
\
void PASTEMAC(ch,varname) \
     ( \
       conj_t         conja, \
       pack_t         schema, \
       dim_t          n, \
       void* restrict kappa, \
       void* restrict a, inc_t inca, inc_t lda, \
       void* restrict p,             inc_t ldp  \
     ) \
{ \
	const inc_t       inca2      = 2 * inca; \
	const inc_t       lda2       = 2 * lda; \
\
	ctype*            kappa_cast = kappa; \
	ctype*   restrict alpha1     = a; \
	ctype_r* restrict alpha1_r   = ( ctype_r* )a; \
	ctype_r* restrict alpha1_i   = ( ctype_r* )a + 1; \
	ctype_r* restrict pi1_r      = ( ctype_r* )p; \
\
\
	if ( bli_is_ro_packed( schema ) ) \
	{ \
		if ( PASTEMAC(ch,eq1)( *kappa_cast ) ) \
		{ \
			/* This works regardless of conja since we are only copying
			   the real part. */ \
			{ \
				for ( ; n != 0; --n ) \
				{ \
					PASTEMAC(chr,copys)( *(alpha1_r + 0*inca2), *(pi1_r + 0) ); \
					PASTEMAC(chr,copys)( *(alpha1_r + 1*inca2), *(pi1_r + 1) ); \
					PASTEMAC(chr,copys)( *(alpha1_r + 2*inca2), *(pi1_r + 2) ); \
					PASTEMAC(chr,copys)( *(alpha1_r + 3*inca2), *(pi1_r + 3) ); \
					PASTEMAC(chr,copys)( *(alpha1_r + 4*inca2), *(pi1_r + 4) ); \
					PASTEMAC(chr,copys)( *(alpha1_r + 5*inca2), *(pi1_r + 5) ); \
					PASTEMAC(chr,copys)( *(alpha1_r + 6*inca2), *(pi1_r + 6) ); \
					PASTEMAC(chr,copys)( *(alpha1_r + 7*inca2), *(pi1_r + 7) ); \
					PASTEMAC(chr,copys)( *(alpha1_r + 8*inca2), *(pi1_r + 8) ); \
					PASTEMAC(chr,copys)( *(alpha1_r + 9*inca2), *(pi1_r + 9) ); \
\
					alpha1_r += lda2; \
					pi1_r    += ldp; \
				} \
			} \
		} \
		else \
		{ \
			if ( bli_is_conj( conja ) ) \
			{ \
				for ( ; n != 0; --n ) \
				{ \
					PASTEMAC(ch,scal2jros)( *kappa_cast, *(alpha1 + 0*inca), *(pi1_r + 0) ); \
					PASTEMAC(ch,scal2jros)( *kappa_cast, *(alpha1 + 1*inca), *(pi1_r + 1) ); \
					PASTEMAC(ch,scal2jros)( *kappa_cast, *(alpha1 + 2*inca), *(pi1_r + 2) ); \
					PASTEMAC(ch,scal2jros)( *kappa_cast, *(alpha1 + 3*inca), *(pi1_r + 3) ); \
					PASTEMAC(ch,scal2jros)( *kappa_cast, *(alpha1 + 4*inca), *(pi1_r + 4) ); \
					PASTEMAC(ch,scal2jros)( *kappa_cast, *(alpha1 + 5*inca), *(pi1_r + 5) ); \
					PASTEMAC(ch,scal2jros)( *kappa_cast, *(alpha1 + 6*inca), *(pi1_r + 6) ); \
					PASTEMAC(ch,scal2jros)( *kappa_cast, *(alpha1 + 7*inca), *(pi1_r + 7) ); \
					PASTEMAC(ch,scal2jros)( *kappa_cast, *(alpha1 + 8*inca), *(pi1_r + 8) ); \
					PASTEMAC(ch,scal2jros)( *kappa_cast, *(alpha1 + 9*inca), *(pi1_r + 9) ); \
\
					alpha1 += lda; \
					pi1_r  += ldp; \
				} \
			} \
			else \
			{ \
				for ( ; n != 0; --n ) \
				{ \
					PASTEMAC(ch,scal2ros)( *kappa_cast, *(alpha1 + 0*inca), *(pi1_r + 0) ); \
					PASTEMAC(ch,scal2ros)( *kappa_cast, *(alpha1 + 1*inca), *(pi1_r + 1) ); \
					PASTEMAC(ch,scal2ros)( *kappa_cast, *(alpha1 + 2*inca), *(pi1_r + 2) ); \
					PASTEMAC(ch,scal2ros)( *kappa_cast, *(alpha1 + 3*inca), *(pi1_r + 3) ); \
					PASTEMAC(ch,scal2ros)( *kappa_cast, *(alpha1 + 4*inca), *(pi1_r + 4) ); \
					PASTEMAC(ch,scal2ros)( *kappa_cast, *(alpha1 + 5*inca), *(pi1_r + 5) ); \
					PASTEMAC(ch,scal2ros)( *kappa_cast, *(alpha1 + 6*inca), *(pi1_r + 6) ); \
					PASTEMAC(ch,scal2ros)( *kappa_cast, *(alpha1 + 7*inca), *(pi1_r + 7) ); \
					PASTEMAC(ch,scal2ros)( *kappa_cast, *(alpha1 + 8*inca), *(pi1_r + 8) ); \
					PASTEMAC(ch,scal2ros)( *kappa_cast, *(alpha1 + 9*inca), *(pi1_r + 9) ); \
\
					alpha1 += lda; \
					pi1_r  += ldp; \
				} \
			} \
		} \
	} \
	else if ( bli_is_io_packed( schema ) ) \
	{ \
		if ( PASTEMAC(ch,eq1)( *kappa_cast ) ) \
		{ \
			if ( bli_is_conj( conja ) ) \
			{ \
				for ( ; n != 0; --n ) \
				{ \
					PASTEMAC(chr,copys)( -*(alpha1_i + 0*inca2), *(pi1_r + 0) ); \
					PASTEMAC(chr,copys)( -*(alpha1_i + 1*inca2), *(pi1_r + 1) ); \
					PASTEMAC(chr,copys)( -*(alpha1_i + 2*inca2), *(pi1_r + 2) ); \
					PASTEMAC(chr,copys)( -*(alpha1_i + 3*inca2), *(pi1_r + 3) ); \
					PASTEMAC(chr,copys)( -*(alpha1_i + 4*inca2), *(pi1_r + 4) ); \
					PASTEMAC(chr,copys)( -*(alpha1_i + 5*inca2), *(pi1_r + 5) ); \
					PASTEMAC(chr,copys)( -*(alpha1_i + 6*inca2), *(pi1_r + 6) ); \
					PASTEMAC(chr,copys)( -*(alpha1_i + 7*inca2), *(pi1_r + 7) ); \
					PASTEMAC(chr,copys)( -*(alpha1_i + 8*inca2), *(pi1_r + 8) ); \
					PASTEMAC(chr,copys)( -*(alpha1_i + 9*inca2), *(pi1_r + 9) ); \
\
					alpha1_i += lda2; \
					pi1_r    += ldp; \
				} \
			} \
			else \
			{ \
				for ( ; n != 0; --n ) \
				{ \
					PASTEMAC(chr,copys)( *(alpha1_i + 0*inca2), *(pi1_r + 0) ); \
					PASTEMAC(chr,copys)( *(alpha1_i + 1*inca2), *(pi1_r + 1) ); \
					PASTEMAC(chr,copys)( *(alpha1_i + 2*inca2), *(pi1_r + 2) ); \
					PASTEMAC(chr,copys)( *(alpha1_i + 3*inca2), *(pi1_r + 3) ); \
					PASTEMAC(chr,copys)( *(alpha1_i + 4*inca2), *(pi1_r + 4) ); \
					PASTEMAC(chr,copys)( *(alpha1_i + 5*inca2), *(pi1_r + 5) ); \
					PASTEMAC(chr,copys)( *(alpha1_i + 6*inca2), *(pi1_r + 6) ); \
					PASTEMAC(chr,copys)( *(alpha1_i + 7*inca2), *(pi1_r + 7) ); \
					PASTEMAC(chr,copys)( *(alpha1_i + 8*inca2), *(pi1_r + 8) ); \
					PASTEMAC(chr,copys)( *(alpha1_i + 9*inca2), *(pi1_r + 9) ); \
\
					alpha1_i += lda2; \
					pi1_r    += ldp; \
				} \
			} \
		} \
		else \
		{ \
			if ( bli_is_conj( conja ) ) \
			{ \
				for ( ; n != 0; --n ) \
				{ \
					PASTEMAC(ch,scal2jios)( *kappa_cast, *(alpha1 + 0*inca), *(pi1_r + 0) ); \
					PASTEMAC(ch,scal2jios)( *kappa_cast, *(alpha1 + 1*inca), *(pi1_r + 1) ); \
					PASTEMAC(ch,scal2jios)( *kappa_cast, *(alpha1 + 2*inca), *(pi1_r + 2) ); \
					PASTEMAC(ch,scal2jios)( *kappa_cast, *(alpha1 + 3*inca), *(pi1_r + 3) ); \
					PASTEMAC(ch,scal2jios)( *kappa_cast, *(alpha1 + 4*inca), *(pi1_r + 4) ); \
					PASTEMAC(ch,scal2jios)( *kappa_cast, *(alpha1 + 5*inca), *(pi1_r + 5) ); \
					PASTEMAC(ch,scal2jios)( *kappa_cast, *(alpha1 + 6*inca), *(pi1_r + 6) ); \
					PASTEMAC(ch,scal2jios)( *kappa_cast, *(alpha1 + 7*inca), *(pi1_r + 7) ); \
					PASTEMAC(ch,scal2jios)( *kappa_cast, *(alpha1 + 8*inca), *(pi1_r + 8) ); \
					PASTEMAC(ch,scal2jios)( *kappa_cast, *(alpha1 + 9*inca), *(pi1_r + 9) ); \
\
					alpha1 += lda; \
					pi1_r  += ldp; \
				} \
			} \
			else \
			{ \
				for ( ; n != 0; --n ) \
				{ \
					PASTEMAC(ch,scal2ios)( *kappa_cast, *(alpha1 + 0*inca), *(pi1_r + 0) ); \
					PASTEMAC(ch,scal2ios)( *kappa_cast, *(alpha1 + 1*inca), *(pi1_r + 1) ); \
					PASTEMAC(ch,scal2ios)( *kappa_cast, *(alpha1 + 2*inca), *(pi1_r + 2) ); \
					PASTEMAC(ch,scal2ios)( *kappa_cast, *(alpha1 + 3*inca), *(pi1_r + 3) ); \
					PASTEMAC(ch,scal2ios)( *kappa_cast, *(alpha1 + 4*inca), *(pi1_r + 4) ); \
					PASTEMAC(ch,scal2ios)( *kappa_cast, *(alpha1 + 5*inca), *(pi1_r + 5) ); \
					PASTEMAC(ch,scal2ios)( *kappa_cast, *(alpha1 + 6*inca), *(pi1_r + 6) ); \
					PASTEMAC(ch,scal2ios)( *kappa_cast, *(alpha1 + 7*inca), *(pi1_r + 7) ); \
					PASTEMAC(ch,scal2ios)( *kappa_cast, *(alpha1 + 8*inca), *(pi1_r + 8) ); \
					PASTEMAC(ch,scal2ios)( *kappa_cast, *(alpha1 + 9*inca), *(pi1_r + 9) ); \
\
					alpha1 += lda; \
					pi1_r  += ldp; \
				} \
			} \
		} \
	} \
	else /* if ( bli_is_rpi_packed( schema ) ) */ \
	{ \
		if ( PASTEMAC(ch,eq1)( *kappa_cast ) ) \
		{ \
			if ( bli_is_conj( conja ) ) \
			{ \
				for ( ; n != 0; --n ) \
				{ \
					PASTEMAC(chr,add3s)( *(alpha1_r + 0*inca2), -*(alpha1_i + 0*inca2), *(pi1_r + 0) ); \
					PASTEMAC(chr,add3s)( *(alpha1_r + 1*inca2), -*(alpha1_i + 1*inca2), *(pi1_r + 1) ); \
					PASTEMAC(chr,add3s)( *(alpha1_r + 2*inca2), -*(alpha1_i + 2*inca2), *(pi1_r + 2) ); \
					PASTEMAC(chr,add3s)( *(alpha1_r + 3*inca2), -*(alpha1_i + 3*inca2), *(pi1_r + 3) ); \
					PASTEMAC(chr,add3s)( *(alpha1_r + 4*inca2), -*(alpha1_i + 4*inca2), *(pi1_r + 4) ); \
					PASTEMAC(chr,add3s)( *(alpha1_r + 5*inca2), -*(alpha1_i + 5*inca2), *(pi1_r + 5) ); \
					PASTEMAC(chr,add3s)( *(alpha1_r + 6*inca2), -*(alpha1_i + 6*inca2), *(pi1_r + 6) ); \
					PASTEMAC(chr,add3s)( *(alpha1_r + 7*inca2), -*(alpha1_i + 7*inca2), *(pi1_r + 7) ); \
					PASTEMAC(chr,add3s)( *(alpha1_r + 8*inca2), -*(alpha1_i + 8*inca2), *(pi1_r + 8) ); \
					PASTEMAC(chr,add3s)( *(alpha1_r + 9*inca2), -*(alpha1_i + 9*inca2), *(pi1_r + 9) ); \
\
					alpha1_r += lda2; \
					alpha1_i += lda2; \
					pi1_r    += ldp; \
				} \
			} \
			else \
			{ \
				for ( ; n != 0; --n ) \
				{ \
					PASTEMAC(chr,add3s)( *(alpha1_r + 0*inca2), *(alpha1_i + 0*inca2), *(pi1_r + 0) ); \
					PASTEMAC(chr,add3s)( *(alpha1_r + 1*inca2), *(alpha1_i + 1*inca2), *(pi1_r + 1) ); \
					PASTEMAC(chr,add3s)( *(alpha1_r + 2*inca2), *(alpha1_i + 2*inca2), *(pi1_r + 2) ); \
					PASTEMAC(chr,add3s)( *(alpha1_r + 3*inca2), *(alpha1_i + 3*inca2), *(pi1_r + 3) ); \
					PASTEMAC(chr,add3s)( *(alpha1_r + 4*inca2), *(alpha1_i + 4*inca2), *(pi1_r + 4) ); \
					PASTEMAC(chr,add3s)( *(alpha1_r + 5*inca2), *(alpha1_i + 5*inca2), *(pi1_r + 5) ); \
					PASTEMAC(chr,add3s)( *(alpha1_r + 6*inca2), *(alpha1_i + 6*inca2), *(pi1_r + 6) ); \
					PASTEMAC(chr,add3s)( *(alpha1_r + 7*inca2), *(alpha1_i + 7*inca2), *(pi1_r + 7) ); \
					PASTEMAC(chr,add3s)( *(alpha1_r + 8*inca2), *(alpha1_i + 8*inca2), *(pi1_r + 8) ); \
					PASTEMAC(chr,add3s)( *(alpha1_r + 9*inca2), *(alpha1_i + 9*inca2), *(pi1_r + 9) ); \
\
					alpha1_r += lda2; \
					alpha1_i += lda2; \
					pi1_r    += ldp; \
				} \
			} \
		} \
		else \
		{ \
			if ( bli_is_conj( conja ) ) \
			{ \
				for ( ; n != 0; --n ) \
				{ \
					PASTEMAC(ch,scal2jrpis)( *kappa_cast, *(alpha1 + 0*inca), *(pi1_r + 0) ); \
					PASTEMAC(ch,scal2jrpis)( *kappa_cast, *(alpha1 + 1*inca), *(pi1_r + 1) ); \
					PASTEMAC(ch,scal2jrpis)( *kappa_cast, *(alpha1 + 2*inca), *(pi1_r + 2) ); \
					PASTEMAC(ch,scal2jrpis)( *kappa_cast, *(alpha1 + 3*inca), *(pi1_r + 3) ); \
					PASTEMAC(ch,scal2jrpis)( *kappa_cast, *(alpha1 + 4*inca), *(pi1_r + 4) ); \
					PASTEMAC(ch,scal2jrpis)( *kappa_cast, *(alpha1 + 5*inca), *(pi1_r + 5) ); \
					PASTEMAC(ch,scal2jrpis)( *kappa_cast, *(alpha1 + 6*inca), *(pi1_r + 6) ); \
					PASTEMAC(ch,scal2jrpis)( *kappa_cast, *(alpha1 + 7*inca), *(pi1_r + 7) ); \
					PASTEMAC(ch,scal2jrpis)( *kappa_cast, *(alpha1 + 8*inca), *(pi1_r + 8) ); \
					PASTEMAC(ch,scal2jrpis)( *kappa_cast, *(alpha1 + 9*inca), *(pi1_r + 9) ); \
\
					alpha1 += lda; \
					pi1_r  += ldp; \
				} \
			} \
			else \
			{ \
				for ( ; n != 0; --n ) \
				{ \
					PASTEMAC(ch,scal2rpis)( *kappa_cast, *(alpha1 + 0*inca), *(pi1_r + 0) ); \
					PASTEMAC(ch,scal2rpis)( *kappa_cast, *(alpha1 + 1*inca), *(pi1_r + 1) ); \
					PASTEMAC(ch,scal2rpis)( *kappa_cast, *(alpha1 + 2*inca), *(pi1_r + 2) ); \
					PASTEMAC(ch,scal2rpis)( *kappa_cast, *(alpha1 + 3*inca), *(pi1_r + 3) ); \
					PASTEMAC(ch,scal2rpis)( *kappa_cast, *(alpha1 + 4*inca), *(pi1_r + 4) ); \
					PASTEMAC(ch,scal2rpis)( *kappa_cast, *(alpha1 + 5*inca), *(pi1_r + 5) ); \
					PASTEMAC(ch,scal2rpis)( *kappa_cast, *(alpha1 + 6*inca), *(pi1_r + 6) ); \
					PASTEMAC(ch,scal2rpis)( *kappa_cast, *(alpha1 + 7*inca), *(pi1_r + 7) ); \
					PASTEMAC(ch,scal2rpis)( *kappa_cast, *(alpha1 + 8*inca), *(pi1_r + 8) ); \
					PASTEMAC(ch,scal2rpis)( *kappa_cast, *(alpha1 + 9*inca), *(pi1_r + 9) ); \
\
					alpha1 += lda; \
					pi1_r  += ldp; \
				} \
			} \
		} \
	} \
}

INSERT_GENTFUNCCO_BASIC0( packm_10xk_rih_ref )



#undef  GENTFUNCCO
#define GENTFUNCCO( ctype, ctype_r, ch, chr, varname ) \
\
void PASTEMAC(ch,varname) \
     ( \
       conj_t         conja, \
       pack_t         schema, \
       dim_t          n, \
       void* restrict kappa, \
       void* restrict a, inc_t inca, inc_t lda, \
       void* restrict p,             inc_t ldp  \
     ) \
{ \
	const inc_t       inca2      = 2 * inca; \
	const inc_t       lda2       = 2 * lda; \
\
	ctype*            kappa_cast = kappa; \
	ctype*   restrict alpha1     = a; \
	ctype_r* restrict alpha1_r   = ( ctype_r* )a; \
	ctype_r* restrict alpha1_i   = ( ctype_r* )a + 1; \
	ctype_r* restrict pi1_r      = ( ctype_r* )p; \
\
\
	if ( bli_is_ro_packed( schema ) ) \
	{ \
		if ( PASTEMAC(ch,eq1)( *kappa_cast ) ) \
		{ \
			/* This works regardless of conja since we are only copying
			   the real part. */ \
			{ \
				for ( ; n != 0; --n ) \
				{ \
					PASTEMAC(chr,copys)( *(alpha1_r + 0*inca2), *(pi1_r + 0) ); \
					PASTEMAC(chr,copys)( *(alpha1_r + 1*inca2), *(pi1_r + 1) ); \
					PASTEMAC(chr,copys)( *(alpha1_r + 2*inca2), *(pi1_r + 2) ); \
					PASTEMAC(chr,copys)( *(alpha1_r + 3*inca2), *(pi1_r + 3) ); \
					PASTEMAC(chr,copys)( *(alpha1_r + 4*inca2), *(pi1_r + 4) ); \
					PASTEMAC(chr,copys)( *(alpha1_r + 5*inca2), *(pi1_r + 5) ); \
					PASTEMAC(chr,copys)( *(alpha1_r + 6*inca2), *(pi1_r + 6) ); \
					PASTEMAC(chr,copys)( *(alpha1_r + 7*inca2), *(pi1_r + 7) ); \
					PASTEMAC(chr,copys)( *(alpha1_r + 8*inca2), *(pi1_r + 8) ); \
					PASTEMAC(chr,copys)( *(alpha1_r + 9*inca2), *(pi1_r + 9) ); \
					PASTEMAC(chr,copys)( *(alpha1_r +10*inca2), *(pi1_r +10) ); \
					PASTEMAC(chr,copys)( *(alpha1_r +11*inca2), *(pi1_r +11) ); \
\
					alpha1_r += lda2; \
					pi1_r    += ldp; \
				} \
			} \
		} \
		else \
		{ \
			if ( bli_is_conj( conja ) ) \
			{ \
				for ( ; n != 0; --n ) \
				{ \
					PASTEMAC(ch,scal2jros)( *kappa_cast, *(alpha1 + 0*inca), *(pi1_r + 0) ); \
					PASTEMAC(ch,scal2jros)( *kappa_cast, *(alpha1 + 1*inca), *(pi1_r + 1) ); \
					PASTEMAC(ch,scal2jros)( *kappa_cast, *(alpha1 + 2*inca), *(pi1_r + 2) ); \
					PASTEMAC(ch,scal2jros)( *kappa_cast, *(alpha1 + 3*inca), *(pi1_r + 3) ); \
					PASTEMAC(ch,scal2jros)( *kappa_cast, *(alpha1 + 4*inca), *(pi1_r + 4) ); \
					PASTEMAC(ch,scal2jros)( *kappa_cast, *(alpha1 + 5*inca), *(pi1_r + 5) ); \
					PASTEMAC(ch,scal2jros)( *kappa_cast, *(alpha1 + 6*inca), *(pi1_r + 6) ); \
					PASTEMAC(ch,scal2jros)( *kappa_cast, *(alpha1 + 7*inca), *(pi1_r + 7) ); \
					PASTEMAC(ch,scal2jros)( *kappa_cast, *(alpha1 + 8*inca), *(pi1_r + 8) ); \
					PASTEMAC(ch,scal2jros)( *kappa_cast, *(alpha1 + 9*inca), *(pi1_r + 9) ); \
					PASTEMAC(ch,scal2jros)( *kappa_cast, *(alpha1 +10*inca), *(pi1_r +10) ); \
					PASTEMAC(ch,scal2jros)( *kappa_cast, *(alpha1 +11*inca), *(pi1_r +11) ); \
\
					alpha1 += lda; \
					pi1_r  += ldp; \
				} \
			} \
			else \
			{ \
				for ( ; n != 0; --n ) \
				{ \
					PASTEMAC(ch,scal2ros)( *kappa_cast, *(alpha1 + 0*inca), *(pi1_r + 0) ); \
					PASTEMAC(ch,scal2ros)( *kappa_cast, *(alpha1 + 1*inca), *(pi1_r + 1) ); \
					PASTEMAC(ch,scal2ros)( *kappa_cast, *(alpha1 + 2*inca), *(pi1_r + 2) ); \
					PASTEMAC(ch,scal2ros)( *kappa_cast, *(alpha1 + 3*inca), *(pi1_r + 3) ); \
					PASTEMAC(ch,scal2ros)( *kappa_cast, *(alpha1 + 4*inca), *(pi1_r + 4) ); \
					PASTEMAC(ch,scal2ros)( *kappa_cast, *(alpha1 + 5*inca), *(pi1_r + 5) ); \
					PASTEMAC(ch,scal2ros)( *kappa_cast, *(alpha1 + 6*inca), *(pi1_r + 6) ); \
					PASTEMAC(ch,scal2ros)( *kappa_cast, *(alpha1 + 7*inca), *(pi1_r + 7) ); \
					PASTEMAC(ch,scal2ros)( *kappa_cast, *(alpha1 + 8*inca), *(pi1_r + 8) ); \
					PASTEMAC(ch,scal2ros)( *kappa_cast, *(alpha1 + 9*inca), *(pi1_r + 9) ); \
					PASTEMAC(ch,scal2ros)( *kappa_cast, *(alpha1 +10*inca), *(pi1_r +10) ); \
					PASTEMAC(ch,scal2ros)( *kappa_cast, *(alpha1 +11*inca), *(pi1_r +11) ); \
\
					alpha1 += lda; \
					pi1_r  += ldp; \
				} \
			} \
		} \
	} \
	else if ( bli_is_io_packed( schema ) ) \
	{ \
		if ( PASTEMAC(ch,eq1)( *kappa_cast ) ) \
		{ \
			if ( bli_is_conj( conja ) ) \
			{ \
				for ( ; n != 0; --n ) \
				{ \
					PASTEMAC(chr,copys)( -*(alpha1_i + 0*inca2), *(pi1_r + 0) ); \
					PASTEMAC(chr,copys)( -*(alpha1_i + 1*inca2), *(pi1_r + 1) ); \
					PASTEMAC(chr,copys)( -*(alpha1_i + 2*inca2), *(pi1_r + 2) ); \
					PASTEMAC(chr,copys)( -*(alpha1_i + 3*inca2), *(pi1_r + 3) ); \
					PASTEMAC(chr,copys)( -*(alpha1_i + 4*inca2), *(pi1_r + 4) ); \
					PASTEMAC(chr,copys)( -*(alpha1_i + 5*inca2), *(pi1_r + 5) ); \
					PASTEMAC(chr,copys)( -*(alpha1_i + 6*inca2), *(pi1_r + 6) ); \
					PASTEMAC(chr,copys)( -*(alpha1_i + 7*inca2), *(pi1_r + 7) ); \
					PASTEMAC(chr,copys)( -*(alpha1_i + 8*inca2), *(pi1_r + 8) ); \
					PASTEMAC(chr,copys)( -*(alpha1_i + 9*inca2), *(pi1_r + 9) ); \
					PASTEMAC(chr,copys)( -*(alpha1_i +10*inca2), *(pi1_r +10) ); \
					PASTEMAC(chr,copys)( -*(alpha1_i +11*inca2), *(pi1_r +11) ); \
\
					alpha1_i += lda2; \
					pi1_r    += ldp; \
				} \
			} \
			else \
			{ \
				for ( ; n != 0; --n ) \
				{ \
					PASTEMAC(chr,copys)( *(alpha1_i + 0*inca2), *(pi1_r + 0) ); \
					PASTEMAC(chr,copys)( *(alpha1_i + 1*inca2), *(pi1_r + 1) ); \
					PASTEMAC(chr,copys)( *(alpha1_i + 2*inca2), *(pi1_r + 2) ); \
					PASTEMAC(chr,copys)( *(alpha1_i + 3*inca2), *(pi1_r + 3) ); \
					PASTEMAC(chr,copys)( *(alpha1_i + 4*inca2), *(pi1_r + 4) ); \
					PASTEMAC(chr,copys)( *(alpha1_i + 5*inca2), *(pi1_r + 5) ); \
					PASTEMAC(chr,copys)( *(alpha1_i + 6*inca2), *(pi1_r + 6) ); \
					PASTEMAC(chr,copys)( *(alpha1_i + 7*inca2), *(pi1_r + 7) ); \
					PASTEMAC(chr,copys)( *(alpha1_i + 8*inca2), *(pi1_r + 8) ); \
					PASTEMAC(chr,copys)( *(alpha1_i + 9*inca2), *(pi1_r + 9) ); \
					PASTEMAC(chr,copys)( *(alpha1_i +10*inca2), *(pi1_r +10) ); \
					PASTEMAC(chr,copys)( *(alpha1_i +11*inca2), *(pi1_r +11) ); \
\
					alpha1_i += lda2; \
					pi1_r    += ldp; \
				} \
			} \
		} \
		else \
		{ \
			if ( bli_is_conj( conja ) ) \
			{ \
				for ( ; n != 0; --n ) \
				{ \
					PASTEMAC(ch,scal2jios)( *kappa_cast, *(alpha1 + 0*inca), *(pi1_r + 0) ); \
					PASTEMAC(ch,scal2jios)( *kappa_cast, *(alpha1 + 1*inca), *(pi1_r + 1) ); \
					PASTEMAC(ch,scal2jios)( *kappa_cast, *(alpha1 + 2*inca), *(pi1_r + 2) ); \
					PASTEMAC(ch,scal2jios)( *kappa_cast, *(alpha1 + 3*inca), *(pi1_r + 3) ); \
					PASTEMAC(ch,scal2jios)( *kappa_cast, *(alpha1 + 4*inca), *(pi1_r + 4) ); \
					PASTEMAC(ch,scal2jios)( *kappa_cast, *(alpha1 + 5*inca), *(pi1_r + 5) ); \
					PASTEMAC(ch,scal2jios)( *kappa_cast, *(alpha1 + 6*inca), *(pi1_r + 6) ); \
					PASTEMAC(ch,scal2jios)( *kappa_cast, *(alpha1 + 7*inca), *(pi1_r + 7) ); \
					PASTEMAC(ch,scal2jios)( *kappa_cast, *(alpha1 + 8*inca), *(pi1_r + 8) ); \
					PASTEMAC(ch,scal2jios)( *kappa_cast, *(alpha1 + 9*inca), *(pi1_r + 9) ); \
					PASTEMAC(ch,scal2jios)( *kappa_cast, *(alpha1 +10*inca), *(pi1_r +10) ); \
					PASTEMAC(ch,scal2jios)( *kappa_cast, *(alpha1 +11*inca), *(pi1_r +11) ); \
\
					alpha1 += lda; \
					pi1_r  += ldp; \
				} \
			} \
			else \
			{ \
				for ( ; n != 0; --n ) \
				{ \
					PASTEMAC(ch,scal2ios)( *kappa_cast, *(alpha1 + 0*inca), *(pi1_r + 0) ); \
					PASTEMAC(ch,scal2ios)( *kappa_cast, *(alpha1 + 1*inca), *(pi1_r + 1) ); \
					PASTEMAC(ch,scal2ios)( *kappa_cast, *(alpha1 + 2*inca), *(pi1_r + 2) ); \
					PASTEMAC(ch,scal2ios)( *kappa_cast, *(alpha1 + 3*inca), *(pi1_r + 3) ); \
					PASTEMAC(ch,scal2ios)( *kappa_cast, *(alpha1 + 4*inca), *(pi1_r + 4) ); \
					PASTEMAC(ch,scal2ios)( *kappa_cast, *(alpha1 + 5*inca), *(pi1_r + 5) ); \
					PASTEMAC(ch,scal2ios)( *kappa_cast, *(alpha1 + 6*inca), *(pi1_r + 6) ); \
					PASTEMAC(ch,scal2ios)( *kappa_cast, *(alpha1 + 7*inca), *(pi1_r + 7) ); \
					PASTEMAC(ch,scal2ios)( *kappa_cast, *(alpha1 + 8*inca), *(pi1_r + 8) ); \
					PASTEMAC(ch,scal2ios)( *kappa_cast, *(alpha1 + 9*inca), *(pi1_r + 9) ); \
					PASTEMAC(ch,scal2ios)( *kappa_cast, *(alpha1 +10*inca), *(pi1_r +10) ); \
					PASTEMAC(ch,scal2ios)( *kappa_cast, *(alpha1 +11*inca), *(pi1_r +11) ); \
\
					alpha1 += lda; \
					pi1_r  += ldp; \
				} \
			} \
		} \
	} \
	else /* if ( bli_is_rpi_packed( schema ) ) */ \
	{ \
		if ( PASTEMAC(ch,eq1)( *kappa_cast ) ) \
		{ \
			if ( bli_is_conj( conja ) ) \
			{ \
				for ( ; n != 0; --n ) \
				{ \
					PASTEMAC(chr,add3s)( *(alpha1_r + 0*inca2), -*(alpha1_i + 0*inca2), *(pi1_r + 0) ); \
					PASTEMAC(chr,add3s)( *(alpha1_r + 1*inca2), -*(alpha1_i + 1*inca2), *(pi1_r + 1) ); \
					PASTEMAC(chr,add3s)( *(alpha1_r + 2*inca2), -*(alpha1_i + 2*inca2), *(pi1_r + 2) ); \
					PASTEMAC(chr,add3s)( *(alpha1_r + 3*inca2), -*(alpha1_i + 3*inca2), *(pi1_r + 3) ); \
					PASTEMAC(chr,add3s)( *(alpha1_r + 4*inca2), -*(alpha1_i + 4*inca2), *(pi1_r + 4) ); \
					PASTEMAC(chr,add3s)( *(alpha1_r + 5*inca2), -*(alpha1_i + 5*inca2), *(pi1_r + 5) ); \
					PASTEMAC(chr,add3s)( *(alpha1_r + 6*inca2), -*(alpha1_i + 6*inca2), *(pi1_r + 6) ); \
					PASTEMAC(chr,add3s)( *(alpha1_r + 7*inca2), -*(alpha1_i + 7*inca2), *(pi1_r + 7) ); \
					PASTEMAC(chr,add3s)( *(alpha1_r + 8*inca2), -*(alpha1_i + 8*inca2), *(pi1_r + 8) ); \
					PASTEMAC(chr,add3s)( *(alpha1_r + 9*inca2), -*(alpha1_i + 9*inca2), *(pi1_r + 9) ); \
					PASTEMAC(chr,add3s)( *(alpha1_r +10*inca2), -*(alpha1_i +10*inca2), *(pi1_r +10) ); \
					PASTEMAC(chr,add3s)( *(alpha1_r +11*inca2), -*(alpha1_i +11*inca2), *(pi1_r +11) ); \
\
					alpha1_r += lda2; \
					alpha1_i += lda2; \
					pi1_r    += ldp; \
				} \
			} \
			else \
			{ \
				for ( ; n != 0; --n ) \
				{ \
					PASTEMAC(chr,add3s)( *(alpha1_r + 0*inca2), *(alpha1_i + 0*inca2), *(pi1_r + 0) ); \
					PASTEMAC(chr,add3s)( *(alpha1_r + 1*inca2), *(alpha1_i + 1*inca2), *(pi1_r + 1) ); \
					PASTEMAC(chr,add3s)( *(alpha1_r + 2*inca2), *(alpha1_i + 2*inca2), *(pi1_r + 2) ); \
					PASTEMAC(chr,add3s)( *(alpha1_r + 3*inca2), *(alpha1_i + 3*inca2), *(pi1_r + 3) ); \
					PASTEMAC(chr,add3s)( *(alpha1_r + 4*inca2), *(alpha1_i + 4*inca2), *(pi1_r + 4) ); \
					PASTEMAC(chr,add3s)( *(alpha1_r + 5*inca2), *(alpha1_i + 5*inca2), *(pi1_r + 5) ); \
					PASTEMAC(chr,add3s)( *(alpha1_r + 6*inca2), *(alpha1_i + 6*inca2), *(pi1_r + 6) ); \
					PASTEMAC(chr,add3s)( *(alpha1_r + 7*inca2), *(alpha1_i + 7*inca2), *(pi1_r + 7) ); \
					PASTEMAC(chr,add3s)( *(alpha1_r + 8*inca2), *(alpha1_i + 8*inca2), *(pi1_r + 8) ); \
					PASTEMAC(chr,add3s)( *(alpha1_r + 9*inca2), *(alpha1_i + 9*inca2), *(pi1_r + 9) ); \
					PASTEMAC(chr,add3s)( *(alpha1_r +10*inca2), *(alpha1_i +10*inca2), *(pi1_r +10) ); \
					PASTEMAC(chr,add3s)( *(alpha1_r +11*inca2), *(alpha1_i +11*inca2), *(pi1_r +11) ); \
\
					alpha1_r += lda2; \
					alpha1_i += lda2; \
					pi1_r    += ldp; \
				} \
			} \
		} \
		else \
		{ \
			if ( bli_is_conj( conja ) ) \
			{ \
				for ( ; n != 0; --n ) \
				{ \
					PASTEMAC(ch,scal2jrpis)( *kappa_cast, *(alpha1 + 0*inca), *(pi1_r + 0) ); \
					PASTEMAC(ch,scal2jrpis)( *kappa_cast, *(alpha1 + 1*inca), *(pi1_r + 1) ); \
					PASTEMAC(ch,scal2jrpis)( *kappa_cast, *(alpha1 + 2*inca), *(pi1_r + 2) ); \
					PASTEMAC(ch,scal2jrpis)( *kappa_cast, *(alpha1 + 3*inca), *(pi1_r + 3) ); \
					PASTEMAC(ch,scal2jrpis)( *kappa_cast, *(alpha1 + 4*inca), *(pi1_r + 4) ); \
					PASTEMAC(ch,scal2jrpis)( *kappa_cast, *(alpha1 + 5*inca), *(pi1_r + 5) ); \
					PASTEMAC(ch,scal2jrpis)( *kappa_cast, *(alpha1 + 6*inca), *(pi1_r + 6) ); \
					PASTEMAC(ch,scal2jrpis)( *kappa_cast, *(alpha1 + 7*inca), *(pi1_r + 7) ); \
					PASTEMAC(ch,scal2jrpis)( *kappa_cast, *(alpha1 + 8*inca), *(pi1_r + 8) ); \
					PASTEMAC(ch,scal2jrpis)( *kappa_cast, *(alpha1 + 9*inca), *(pi1_r + 9) ); \
					PASTEMAC(ch,scal2jrpis)( *kappa_cast, *(alpha1 +10*inca), *(pi1_r +10) ); \
					PASTEMAC(ch,scal2jrpis)( *kappa_cast, *(alpha1 +11*inca), *(pi1_r +11) ); \
\
					alpha1 += lda; \
					pi1_r  += ldp; \
				} \
			} \
			else \
			{ \
				for ( ; n != 0; --n ) \
				{ \
					PASTEMAC(ch,scal2rpis)( *kappa_cast, *(alpha1 + 0*inca), *(pi1_r + 0) ); \
					PASTEMAC(ch,scal2rpis)( *kappa_cast, *(alpha1 + 1*inca), *(pi1_r + 1) ); \
					PASTEMAC(ch,scal2rpis)( *kappa_cast, *(alpha1 + 2*inca), *(pi1_r + 2) ); \
					PASTEMAC(ch,scal2rpis)( *kappa_cast, *(alpha1 + 3*inca), *(pi1_r + 3) ); \
					PASTEMAC(ch,scal2rpis)( *kappa_cast, *(alpha1 + 4*inca), *(pi1_r + 4) ); \
					PASTEMAC(ch,scal2rpis)( *kappa_cast, *(alpha1 + 5*inca), *(pi1_r + 5) ); \
					PASTEMAC(ch,scal2rpis)( *kappa_cast, *(alpha1 + 6*inca), *(pi1_r + 6) ); \
					PASTEMAC(ch,scal2rpis)( *kappa_cast, *(alpha1 + 7*inca), *(pi1_r + 7) ); \
					PASTEMAC(ch,scal2rpis)( *kappa_cast, *(alpha1 + 8*inca), *(pi1_r + 8) ); \
					PASTEMAC(ch,scal2rpis)( *kappa_cast, *(alpha1 + 9*inca), *(pi1_r + 9) ); \
					PASTEMAC(ch,scal2rpis)( *kappa_cast, *(alpha1 +10*inca), *(pi1_r +10) ); \
					PASTEMAC(ch,scal2rpis)( *kappa_cast, *(alpha1 +11*inca), *(pi1_r +11) ); \
\
					alpha1 += lda; \
					pi1_r  += ldp; \
				} \
			} \
		} \
	} \
}

INSERT_GENTFUNCCO_BASIC0( packm_12xk_rih_ref )



#undef  GENTFUNCCO
#define GENTFUNCCO( ctype, ctype_r, ch, chr, varname ) \
\
void PASTEMAC(ch,varname) \
     ( \
       conj_t         conja, \
       pack_t         schema, \
       dim_t          n, \
       void* restrict kappa, \
       void* restrict a, inc_t inca, inc_t lda, \
       void* restrict p,             inc_t ldp  \
     ) \
{ \
	const inc_t       inca2      = 2 * inca; \
	const inc_t       lda2       = 2 * lda; \
\
	ctype*            kappa_cast = kappa; \
	ctype*   restrict alpha1     = a; \
	ctype_r* restrict alpha1_r   = ( ctype_r* )a; \
	ctype_r* restrict alpha1_i   = ( ctype_r* )a + 1; \
	ctype_r* restrict pi1_r      = ( ctype_r* )p; \
\
\
	if ( bli_is_ro_packed( schema ) ) \
	{ \
		if ( PASTEMAC(ch,eq1)( *kappa_cast ) ) \
		{ \
			/* This works regardless of conja since we are only copying
			   the real part. */ \
			{ \
				for ( ; n != 0; --n ) \
				{ \
					PASTEMAC(chr,copys)( *(alpha1_r + 0*inca2), *(pi1_r + 0) ); \
					PASTEMAC(chr,copys)( *(alpha1_r + 1*inca2), *(pi1_r + 1) ); \
					PASTEMAC(chr,copys)( *(alpha1_r + 2*inca2), *(pi1_r + 2) ); \
					PASTEMAC(chr,copys)( *(alpha1_r + 3*inca2), *(pi1_r + 3) ); \
					PASTEMAC(chr,copys)( *(alpha1_r + 4*inca2), *(pi1_r + 4) ); \
					PASTEMAC(chr,copys)( *(alpha1_r + 5*inca2), *(pi1_r + 5) ); \
					PASTEMAC(chr,copys)( *(alpha1_r + 6*inca2), *(pi1_r + 6) ); \
					PASTEMAC(chr,copys)( *(alpha1_r + 7*inca2), *(pi1_r + 7) ); \
					PASTEMAC(chr,copys)( *(alpha1_r + 8*inca2), *(pi1_r + 8) ); \
					PASTEMAC(chr,copys)( *(alpha1_r + 9*inca2), *(pi1_r + 9) ); \
					PASTEMAC(chr,copys)( *(alpha1_r +10*inca2), *(pi1_r +10) ); \
					PASTEMAC(chr,copys)( *(alpha1_r +11*inca2), *(pi1_r +11) ); \
					PASTEMAC(chr,copys)( *(alpha1_r +12*inca2), *(pi1_r +12) ); \
					PASTEMAC(chr,copys)( *(alpha1_r +13*inca2), *(pi1_r +13) ); \
\
					alpha1_r += lda2; \
					pi1_r    += ldp; \
				} \
			} \
		} \
		else \
		{ \
			if ( bli_is_conj( conja ) ) \
			{ \
				for ( ; n != 0; --n ) \
				{ \
					PASTEMAC(ch,scal2jros)( *kappa_cast, *(alpha1 + 0*inca), *(pi1_r + 0) ); \
					PASTEMAC(ch,scal2jros)( *kappa_cast, *(alpha1 + 1*inca), *(pi1_r + 1) ); \
					PASTEMAC(ch,scal2jros)( *kappa_cast, *(alpha1 + 2*inca), *(pi1_r + 2) ); \
					PASTEMAC(ch,scal2jros)( *kappa_cast, *(alpha1 + 3*inca), *(pi1_r + 3) ); \
					PASTEMAC(ch,scal2jros)( *kappa_cast, *(alpha1 + 4*inca), *(pi1_r + 4) ); \
					PASTEMAC(ch,scal2jros)( *kappa_cast, *(alpha1 + 5*inca), *(pi1_r + 5) ); \
					PASTEMAC(ch,scal2jros)( *kappa_cast, *(alpha1 + 6*inca), *(pi1_r + 6) ); \
					PASTEMAC(ch,scal2jros)( *kappa_cast, *(alpha1 + 7*inca), *(pi1_r + 7) ); \
					PASTEMAC(ch,scal2jros)( *kappa_cast, *(alpha1 + 8*inca), *(pi1_r + 8) ); \
					PASTEMAC(ch,scal2jros)( *kappa_cast, *(alpha1 + 9*inca), *(pi1_r + 9) ); \
					PASTEMAC(ch,scal2jros)( *kappa_cast, *(alpha1 +10*inca), *(pi1_r +10) ); \
					PASTEMAC(ch,scal2jros)( *kappa_cast, *(alpha1 +11*inca), *(pi1_r +11) ); \
					PASTEMAC(ch,scal2jros)( *kappa_cast, *(alpha1 +12*inca), *(pi1_r +12) ); \
					PASTEMAC(ch,scal2jros)( *kappa_cast, *(alpha1 +13*inca), *(pi1_r +13) ); \
\
					alpha1 += lda; \
					pi1_r  += ldp; \
				} \
			} \
			else \
			{ \
				for ( ; n != 0; --n ) \
				{ \
					PASTEMAC(ch,scal2ros)( *kappa_cast, *(alpha1 + 0*inca), *(pi1_r + 0) ); \
					PASTEMAC(ch,scal2ros)( *kappa_cast, *(alpha1 + 1*inca), *(pi1_r + 1) ); \
					PASTEMAC(ch,scal2ros)( *kappa_cast, *(alpha1 + 2*inca), *(pi1_r + 2) ); \
					PASTEMAC(ch,scal2ros)( *kappa_cast, *(alpha1 + 3*inca), *(pi1_r + 3) ); \
					PASTEMAC(ch,scal2ros)( *kappa_cast, *(alpha1 + 4*inca), *(pi1_r + 4) ); \
					PASTEMAC(ch,scal2ros)( *kappa_cast, *(alpha1 + 5*inca), *(pi1_r + 5) ); \
					PASTEMAC(ch,scal2ros)( *kappa_cast, *(alpha1 + 6*inca), *(pi1_r + 6) ); \
					PASTEMAC(ch,scal2ros)( *kappa_cast, *(alpha1 + 7*inca), *(pi1_r + 7) ); \
					PASTEMAC(ch,scal2ros)( *kappa_cast, *(alpha1 + 8*inca), *(pi1_r + 8) ); \
					PASTEMAC(ch,scal2ros)( *kappa_cast, *(alpha1 + 9*inca), *(pi1_r + 9) ); \
					PASTEMAC(ch,scal2ros)( *kappa_cast, *(alpha1 +10*inca), *(pi1_r +10) ); \
					PASTEMAC(ch,scal2ros)( *kappa_cast, *(alpha1 +11*inca), *(pi1_r +11) ); \
					PASTEMAC(ch,scal2ros)( *kappa_cast, *(alpha1 +12*inca), *(pi1_r +12) ); \
					PASTEMAC(ch,scal2ros)( *kappa_cast, *(alpha1 +13*inca), *(pi1_r +13) ); \
\
					alpha1 += lda; \
					pi1_r  += ldp; \
				} \
			} \
		} \
	} \
	else if ( bli_is_io_packed( schema ) ) \
	{ \
		if ( PASTEMAC(ch,eq1)( *kappa_cast ) ) \
		{ \
			if ( bli_is_conj( conja ) ) \
			{ \
				for ( ; n != 0; --n ) \
				{ \
					PASTEMAC(chr,copys)( -*(alpha1_i + 0*inca2), *(pi1_r + 0) ); \
					PASTEMAC(chr,copys)( -*(alpha1_i + 1*inca2), *(pi1_r + 1) ); \
					PASTEMAC(chr,copys)( -*(alpha1_i + 2*inca2), *(pi1_r + 2) ); \
					PASTEMAC(chr,copys)( -*(alpha1_i + 3*inca2), *(pi1_r + 3) ); \
					PASTEMAC(chr,copys)( -*(alpha1_i + 4*inca2), *(pi1_r + 4) ); \
					PASTEMAC(chr,copys)( -*(alpha1_i + 5*inca2), *(pi1_r + 5) ); \
					PASTEMAC(chr,copys)( -*(alpha1_i + 6*inca2), *(pi1_r + 6) ); \
					PASTEMAC(chr,copys)( -*(alpha1_i + 7*inca2), *(pi1_r + 7) ); \
					PASTEMAC(chr,copys)( -*(alpha1_i + 8*inca2), *(pi1_r + 8) ); \
					PASTEMAC(chr,copys)( -*(alpha1_i + 9*inca2), *(pi1_r + 9) ); \
					PASTEMAC(chr,copys)( -*(alpha1_i +10*inca2), *(pi1_r +10) ); \
					PASTEMAC(chr,copys)( -*(alpha1_i +11*inca2), *(pi1_r +11) ); \
					PASTEMAC(chr,copys)( -*(alpha1_i +12*inca2), *(pi1_r +12) ); \
					PASTEMAC(chr,copys)( -*(alpha1_i +13*inca2), *(pi1_r +13) ); \
\
					alpha1_i += lda2; \
					pi1_r    += ldp; \
				} \
			} \
			else \
			{ \
				for ( ; n != 0; --n ) \
				{ \
					PASTEMAC(chr,copys)( *(alpha1_i + 0*inca2), *(pi1_r + 0) ); \
					PASTEMAC(chr,copys)( *(alpha1_i + 1*inca2), *(pi1_r + 1) ); \
					PASTEMAC(chr,copys)( *(alpha1_i + 2*inca2), *(pi1_r + 2) ); \
					PASTEMAC(chr,copys)( *(alpha1_i + 3*inca2), *(pi1_r + 3) ); \
					PASTEMAC(chr,copys)( *(alpha1_i + 4*inca2), *(pi1_r + 4) ); \
					PASTEMAC(chr,copys)( *(alpha1_i + 5*inca2), *(pi1_r + 5) ); \
					PASTEMAC(chr,copys)( *(alpha1_i + 6*inca2), *(pi1_r + 6) ); \
					PASTEMAC(chr,copys)( *(alpha1_i + 7*inca2), *(pi1_r + 7) ); \
					PASTEMAC(chr,copys)( *(alpha1_i + 8*inca2), *(pi1_r + 8) ); \
					PASTEMAC(chr,copys)( *(alpha1_i + 9*inca2), *(pi1_r + 9) ); \
					PASTEMAC(chr,copys)( *(alpha1_i +10*inca2), *(pi1_r +10) ); \
					PASTEMAC(chr,copys)( *(alpha1_i +11*inca2), *(pi1_r +11) ); \
					PASTEMAC(chr,copys)( *(alpha1_i +12*inca2), *(pi1_r +12) ); \
					PASTEMAC(chr,copys)( *(alpha1_i +13*inca2), *(pi1_r +13) ); \
\
					alpha1_i += lda2; \
					pi1_r    += ldp; \
				} \
			} \
		} \
		else \
		{ \
			if ( bli_is_conj( conja ) ) \
			{ \
				for ( ; n != 0; --n ) \
				{ \
					PASTEMAC(ch,scal2jios)( *kappa_cast, *(alpha1 + 0*inca), *(pi1_r + 0) ); \
					PASTEMAC(ch,scal2jios)( *kappa_cast, *(alpha1 + 1*inca), *(pi1_r + 1) ); \
					PASTEMAC(ch,scal2jios)( *kappa_cast, *(alpha1 + 2*inca), *(pi1_r + 2) ); \
					PASTEMAC(ch,scal2jios)( *kappa_cast, *(alpha1 + 3*inca), *(pi1_r + 3) ); \
					PASTEMAC(ch,scal2jios)( *kappa_cast, *(alpha1 + 4*inca), *(pi1_r + 4) ); \
					PASTEMAC(ch,scal2jios)( *kappa_cast, *(alpha1 + 5*inca), *(pi1_r + 5) ); \
					PASTEMAC(ch,scal2jios)( *kappa_cast, *(alpha1 + 6*inca), *(pi1_r + 6) ); \
					PASTEMAC(ch,scal2jios)( *kappa_cast, *(alpha1 + 7*inca), *(pi1_r + 7) ); \
					PASTEMAC(ch,scal2jios)( *kappa_cast, *(alpha1 + 8*inca), *(pi1_r + 8) ); \
					PASTEMAC(ch,scal2jios)( *kappa_cast, *(alpha1 + 9*inca), *(pi1_r + 9) ); \
					PASTEMAC(ch,scal2jios)( *kappa_cast, *(alpha1 +10*inca), *(pi1_r +10) ); \
					PASTEMAC(ch,scal2jios)( *kappa_cast, *(alpha1 +11*inca), *(pi1_r +11) ); \
					PASTEMAC(ch,scal2jios)( *kappa_cast, *(alpha1 +12*inca), *(pi1_r +12) ); \
					PASTEMAC(ch,scal2jios)( *kappa_cast, *(alpha1 +13*inca), *(pi1_r +13) ); \
\
					alpha1 += lda; \
					pi1_r  += ldp; \
				} \
			} \
			else \
			{ \
				for ( ; n != 0; --n ) \
				{ \
					PASTEMAC(ch,scal2ios)( *kappa_cast, *(alpha1 + 0*inca), *(pi1_r + 0) ); \
					PASTEMAC(ch,scal2ios)( *kappa_cast, *(alpha1 + 1*inca), *(pi1_r + 1) ); \
					PASTEMAC(ch,scal2ios)( *kappa_cast, *(alpha1 + 2*inca), *(pi1_r + 2) ); \
					PASTEMAC(ch,scal2ios)( *kappa_cast, *(alpha1 + 3*inca), *(pi1_r + 3) ); \
					PASTEMAC(ch,scal2ios)( *kappa_cast, *(alpha1 + 4*inca), *(pi1_r + 4) ); \
					PASTEMAC(ch,scal2ios)( *kappa_cast, *(alpha1 + 5*inca), *(pi1_r + 5) ); \
					PASTEMAC(ch,scal2ios)( *kappa_cast, *(alpha1 + 6*inca), *(pi1_r + 6) ); \
					PASTEMAC(ch,scal2ios)( *kappa_cast, *(alpha1 + 7*inca), *(pi1_r + 7) ); \
					PASTEMAC(ch,scal2ios)( *kappa_cast, *(alpha1 + 8*inca), *(pi1_r + 8) ); \
					PASTEMAC(ch,scal2ios)( *kappa_cast, *(alpha1 + 9*inca), *(pi1_r + 9) ); \
					PASTEMAC(ch,scal2ios)( *kappa_cast, *(alpha1 +10*inca), *(pi1_r +10) ); \
					PASTEMAC(ch,scal2ios)( *kappa_cast, *(alpha1 +11*inca), *(pi1_r +11) ); \
					PASTEMAC(ch,scal2ios)( *kappa_cast, *(alpha1 +12*inca), *(pi1_r +12) ); \
					PASTEMAC(ch,scal2ios)( *kappa_cast, *(alpha1 +13*inca), *(pi1_r +13) ); \
\
					alpha1 += lda; \
					pi1_r  += ldp; \
				} \
			} \
		} \
	} \
	else /* if ( bli_is_rpi_packed( schema ) ) */ \
	{ \
		if ( PASTEMAC(ch,eq1)( *kappa_cast ) ) \
		{ \
			if ( bli_is_conj( conja ) ) \
			{ \
				for ( ; n != 0; --n ) \
				{ \
					PASTEMAC(chr,add3s)( *(alpha1_r + 0*inca2), -*(alpha1_i + 0*inca2), *(pi1_r + 0) ); \
					PASTEMAC(chr,add3s)( *(alpha1_r + 1*inca2), -*(alpha1_i + 1*inca2), *(pi1_r + 1) ); \
					PASTEMAC(chr,add3s)( *(alpha1_r + 2*inca2), -*(alpha1_i + 2*inca2), *(pi1_r + 2) ); \
					PASTEMAC(chr,add3s)( *(alpha1_r + 3*inca2), -*(alpha1_i + 3*inca2), *(pi1_r + 3) ); \
					PASTEMAC(chr,add3s)( *(alpha1_r + 4*inca2), -*(alpha1_i + 4*inca2), *(pi1_r + 4) ); \
					PASTEMAC(chr,add3s)( *(alpha1_r + 5*inca2), -*(alpha1_i + 5*inca2), *(pi1_r + 5) ); \
					PASTEMAC(chr,add3s)( *(alpha1_r + 6*inca2), -*(alpha1_i + 6*inca2), *(pi1_r + 6) ); \
					PASTEMAC(chr,add3s)( *(alpha1_r + 7*inca2), -*(alpha1_i + 7*inca2), *(pi1_r + 7) ); \
					PASTEMAC(chr,add3s)( *(alpha1_r + 8*inca2), -*(alpha1_i + 8*inca2), *(pi1_r + 8) ); \
					PASTEMAC(chr,add3s)( *(alpha1_r + 9*inca2), -*(alpha1_i + 9*inca2), *(pi1_r + 9) ); \
					PASTEMAC(chr,add3s)( *(alpha1_r +10*inca2), -*(alpha1_i +10*inca2), *(pi1_r +10) ); \
					PASTEMAC(chr,add3s)( *(alpha1_r +11*inca2), -*(alpha1_i +11*inca2), *(pi1_r +11) ); \
					PASTEMAC(chr,add3s)( *(alpha1_r +12*inca2), -*(alpha1_i +12*inca2), *(pi1_r +12) ); \
					PASTEMAC(chr,add3s)( *(alpha1_r +13*inca2), -*(alpha1_i +13*inca2), *(pi1_r +13) ); \
\
					alpha1_r += lda2; \
					alpha1_i += lda2; \
					pi1_r    += ldp; \
				} \
			} \
			else \
			{ \
				for ( ; n != 0; --n ) \
				{ \
					PASTEMAC(chr,add3s)( *(alpha1_r + 0*inca2), *(alpha1_i + 0*inca2), *(pi1_r + 0) ); \
					PASTEMAC(chr,add3s)( *(alpha1_r + 1*inca2), *(alpha1_i + 1*inca2), *(pi1_r + 1) ); \
					PASTEMAC(chr,add3s)( *(alpha1_r + 2*inca2), *(alpha1_i + 2*inca2), *(pi1_r + 2) ); \
					PASTEMAC(chr,add3s)( *(alpha1_r + 3*inca2), *(alpha1_i + 3*inca2), *(pi1_r + 3) ); \
					PASTEMAC(chr,add3s)( *(alpha1_r + 4*inca2), *(alpha1_i + 4*inca2), *(pi1_r + 4) ); \
					PASTEMAC(chr,add3s)( *(alpha1_r + 5*inca2), *(alpha1_i + 5*inca2), *(pi1_r + 5) ); \
					PASTEMAC(chr,add3s)( *(alpha1_r + 6*inca2), *(alpha1_i + 6*inca2), *(pi1_r + 6) ); \
					PASTEMAC(chr,add3s)( *(alpha1_r + 7*inca2), *(alpha1_i + 7*inca2), *(pi1_r + 7) ); \
					PASTEMAC(chr,add3s)( *(alpha1_r + 8*inca2), *(alpha1_i + 8*inca2), *(pi1_r + 8) ); \
					PASTEMAC(chr,add3s)( *(alpha1_r + 9*inca2), *(alpha1_i + 9*inca2), *(pi1_r + 9) ); \
					PASTEMAC(chr,add3s)( *(alpha1_r +10*inca2), *(alpha1_i +10*inca2), *(pi1_r +10) ); \
					PASTEMAC(chr,add3s)( *(alpha1_r +11*inca2), *(alpha1_i +11*inca2), *(pi1_r +11) ); \
					PASTEMAC(chr,add3s)( *(alpha1_r +12*inca2), *(alpha1_i +12*inca2), *(pi1_r +12) ); \
					PASTEMAC(chr,add3s)( *(alpha1_r +13*inca2), *(alpha1_i +13*inca2), *(pi1_r +13) ); \
\
					alpha1_r += lda2; \
					alpha1_i += lda2; \
					pi1_r    += ldp; \
				} \
			} \
		} \
		else \
		{ \
			if ( bli_is_conj( conja ) ) \
			{ \
				for ( ; n != 0; --n ) \
				{ \
					PASTEMAC(ch,scal2jrpis)( *kappa_cast, *(alpha1 + 0*inca), *(pi1_r + 0) ); \
					PASTEMAC(ch,scal2jrpis)( *kappa_cast, *(alpha1 + 1*inca), *(pi1_r + 1) ); \
					PASTEMAC(ch,scal2jrpis)( *kappa_cast, *(alpha1 + 2*inca), *(pi1_r + 2) ); \
					PASTEMAC(ch,scal2jrpis)( *kappa_cast, *(alpha1 + 3*inca), *(pi1_r + 3) ); \
					PASTEMAC(ch,scal2jrpis)( *kappa_cast, *(alpha1 + 4*inca), *(pi1_r + 4) ); \
					PASTEMAC(ch,scal2jrpis)( *kappa_cast, *(alpha1 + 5*inca), *(pi1_r + 5) ); \
					PASTEMAC(ch,scal2jrpis)( *kappa_cast, *(alpha1 + 6*inca), *(pi1_r + 6) ); \
					PASTEMAC(ch,scal2jrpis)( *kappa_cast, *(alpha1 + 7*inca), *(pi1_r + 7) ); \
					PASTEMAC(ch,scal2jrpis)( *kappa_cast, *(alpha1 + 8*inca), *(pi1_r + 8) ); \
					PASTEMAC(ch,scal2jrpis)( *kappa_cast, *(alpha1 + 9*inca), *(pi1_r + 9) ); \
					PASTEMAC(ch,scal2jrpis)( *kappa_cast, *(alpha1 +10*inca), *(pi1_r +10) ); \
					PASTEMAC(ch,scal2jrpis)( *kappa_cast, *(alpha1 +11*inca), *(pi1_r +11) ); \
					PASTEMAC(ch,scal2jrpis)( *kappa_cast, *(alpha1 +12*inca), *(pi1_r +12) ); \
					PASTEMAC(ch,scal2jrpis)( *kappa_cast, *(alpha1 +13*inca), *(pi1_r +13) ); \
\
					alpha1 += lda; \
					pi1_r  += ldp; \
				} \
			} \
			else \
			{ \
				for ( ; n != 0; --n ) \
				{ \
					PASTEMAC(ch,scal2rpis)( *kappa_cast, *(alpha1 + 0*inca), *(pi1_r + 0) ); \
					PASTEMAC(ch,scal2rpis)( *kappa_cast, *(alpha1 + 1*inca), *(pi1_r + 1) ); \
					PASTEMAC(ch,scal2rpis)( *kappa_cast, *(alpha1 + 2*inca), *(pi1_r + 2) ); \
					PASTEMAC(ch,scal2rpis)( *kappa_cast, *(alpha1 + 3*inca), *(pi1_r + 3) ); \
					PASTEMAC(ch,scal2rpis)( *kappa_cast, *(alpha1 + 4*inca), *(pi1_r + 4) ); \
					PASTEMAC(ch,scal2rpis)( *kappa_cast, *(alpha1 + 5*inca), *(pi1_r + 5) ); \
					PASTEMAC(ch,scal2rpis)( *kappa_cast, *(alpha1 + 6*inca), *(pi1_r + 6) ); \
					PASTEMAC(ch,scal2rpis)( *kappa_cast, *(alpha1 + 7*inca), *(pi1_r + 7) ); \
					PASTEMAC(ch,scal2rpis)( *kappa_cast, *(alpha1 + 8*inca), *(pi1_r + 8) ); \
					PASTEMAC(ch,scal2rpis)( *kappa_cast, *(alpha1 + 9*inca), *(pi1_r + 9) ); \
					PASTEMAC(ch,scal2rpis)( *kappa_cast, *(alpha1 +10*inca), *(pi1_r +10) ); \
					PASTEMAC(ch,scal2rpis)( *kappa_cast, *(alpha1 +11*inca), *(pi1_r +11) ); \
					PASTEMAC(ch,scal2rpis)( *kappa_cast, *(alpha1 +12*inca), *(pi1_r +12) ); \
					PASTEMAC(ch,scal2rpis)( *kappa_cast, *(alpha1 +13*inca), *(pi1_r +13) ); \
\
					alpha1 += lda; \
					pi1_r  += ldp; \
				} \
			} \
		} \
	} \
}

INSERT_GENTFUNCCO_BASIC0( packm_14xk_rih_ref )



#undef  GENTFUNCCO
#define GENTFUNCCO( ctype, ctype_r, ch, chr, varname ) \
\
void PASTEMAC(ch,varname) \
     ( \
       conj_t         conja, \
       pack_t         schema, \
       dim_t          n, \
       void* restrict kappa, \
       void* restrict a, inc_t inca, inc_t lda, \
       void* restrict p,             inc_t ldp  \
     ) \
{ \
	const inc_t       inca2      = 2 * inca; \
	const inc_t       lda2       = 2 * lda; \
\
	ctype*            kappa_cast = kappa; \
	ctype*   restrict alpha1     = a; \
	ctype_r* restrict alpha1_r   = ( ctype_r* )a; \
	ctype_r* restrict alpha1_i   = ( ctype_r* )a + 1; \
	ctype_r* restrict pi1_r      = ( ctype_r* )p; \
\
\
	if ( bli_is_ro_packed( schema ) ) \
	{ \
		if ( PASTEMAC(ch,eq1)( *kappa_cast ) ) \
		{ \
			/* This works regardless of conja since we are only copying
			   the real part. */ \
			{ \
				for ( ; n != 0; --n ) \
				{ \
					PASTEMAC(chr,copys)( *(alpha1_r + 0*inca2), *(pi1_r + 0) ); \
					PASTEMAC(chr,copys)( *(alpha1_r + 1*inca2), *(pi1_r + 1) ); \
					PASTEMAC(chr,copys)( *(alpha1_r + 2*inca2), *(pi1_r + 2) ); \
					PASTEMAC(chr,copys)( *(alpha1_r + 3*inca2), *(pi1_r + 3) ); \
					PASTEMAC(chr,copys)( *(alpha1_r + 4*inca2), *(pi1_r + 4) ); \
					PASTEMAC(chr,copys)( *(alpha1_r + 5*inca2), *(pi1_r + 5) ); \
					PASTEMAC(chr,copys)( *(alpha1_r + 6*inca2), *(pi1_r + 6) ); \
					PASTEMAC(chr,copys)( *(alpha1_r + 7*inca2), *(pi1_r + 7) ); \
					PASTEMAC(chr,copys)( *(alpha1_r + 8*inca2), *(pi1_r + 8) ); \
					PASTEMAC(chr,copys)( *(alpha1_r + 9*inca2), *(pi1_r + 9) ); \
					PASTEMAC(chr,copys)( *(alpha1_r +10*inca2), *(pi1_r +10) ); \
					PASTEMAC(chr,copys)( *(alpha1_r +11*inca2), *(pi1_r +11) ); \
					PASTEMAC(chr,copys)( *(alpha1_r +12*inca2), *(pi1_r +12) ); \
					PASTEMAC(chr,copys)( *(alpha1_r +13*inca2), *(pi1_r +13) ); \
					PASTEMAC(chr,copys)( *(alpha1_r +14*inca2), *(pi1_r +14) ); \
					PASTEMAC(chr,copys)( *(alpha1_r +15*inca2), *(pi1_r +15) ); \
\
					alpha1_r += lda2; \
					pi1_r    += ldp; \
				} \
			} \
		} \
		else \
		{ \
			if ( bli_is_conj( conja ) ) \
			{ \
				for ( ; n != 0; --n ) \
				{ \
					PASTEMAC(ch,scal2jros)( *kappa_cast, *(alpha1 + 0*inca), *(pi1_r + 0) ); \
					PASTEMAC(ch,scal2jros)( *kappa_cast, *(alpha1 + 1*inca), *(pi1_r + 1) ); \
					PASTEMAC(ch,scal2jros)( *kappa_cast, *(alpha1 + 2*inca), *(pi1_r + 2) ); \
					PASTEMAC(ch,scal2jros)( *kappa_cast, *(alpha1 + 3*inca), *(pi1_r + 3) ); \
					PASTEMAC(ch,scal2jros)( *kappa_cast, *(alpha1 + 4*inca), *(pi1_r + 4) ); \
					PASTEMAC(ch,scal2jros)( *kappa_cast, *(alpha1 + 5*inca), *(pi1_r + 5) ); \
					PASTEMAC(ch,scal2jros)( *kappa_cast, *(alpha1 + 6*inca), *(pi1_r + 6) ); \
					PASTEMAC(ch,scal2jros)( *kappa_cast, *(alpha1 + 7*inca), *(pi1_r + 7) ); \
					PASTEMAC(ch,scal2jros)( *kappa_cast, *(alpha1 + 8*inca), *(pi1_r + 8) ); \
					PASTEMAC(ch,scal2jros)( *kappa_cast, *(alpha1 + 9*inca), *(pi1_r + 9) ); \
					PASTEMAC(ch,scal2jros)( *kappa_cast, *(alpha1 +10*inca), *(pi1_r +10) ); \
					PASTEMAC(ch,scal2jros)( *kappa_cast, *(alpha1 +11*inca), *(pi1_r +11) ); \
					PASTEMAC(ch,scal2jros)( *kappa_cast, *(alpha1 +12*inca), *(pi1_r +12) ); \
					PASTEMAC(ch,scal2jros)( *kappa_cast, *(alpha1 +13*inca), *(pi1_r +13) ); \
					PASTEMAC(ch,scal2jros)( *kappa_cast, *(alpha1 +14*inca), *(pi1_r +14) ); \
					PASTEMAC(ch,scal2jros)( *kappa_cast, *(alpha1 +15*inca), *(pi1_r +15) ); \
\
					alpha1 += lda; \
					pi1_r  += ldp; \
				} \
			} \
			else \
			{ \
				for ( ; n != 0; --n ) \
				{ \
					PASTEMAC(ch,scal2ros)( *kappa_cast, *(alpha1 + 0*inca), *(pi1_r + 0) ); \
					PASTEMAC(ch,scal2ros)( *kappa_cast, *(alpha1 + 1*inca), *(pi1_r + 1) ); \
					PASTEMAC(ch,scal2ros)( *kappa_cast, *(alpha1 + 2*inca), *(pi1_r + 2) ); \
					PASTEMAC(ch,scal2ros)( *kappa_cast, *(alpha1 + 3*inca), *(pi1_r + 3) ); \
					PASTEMAC(ch,scal2ros)( *kappa_cast, *(alpha1 + 4*inca), *(pi1_r + 4) ); \
					PASTEMAC(ch,scal2ros)( *kappa_cast, *(alpha1 + 5*inca), *(pi1_r + 5) ); \
					PASTEMAC(ch,scal2ros)( *kappa_cast, *(alpha1 + 6*inca), *(pi1_r + 6) ); \
					PASTEMAC(ch,scal2ros)( *kappa_cast, *(alpha1 + 7*inca), *(pi1_r + 7) ); \
					PASTEMAC(ch,scal2ros)( *kappa_cast, *(alpha1 + 8*inca), *(pi1_r + 8) ); \
					PASTEMAC(ch,scal2ros)( *kappa_cast, *(alpha1 + 9*inca), *(pi1_r + 9) ); \
					PASTEMAC(ch,scal2ros)( *kappa_cast, *(alpha1 +10*inca), *(pi1_r +10) ); \
					PASTEMAC(ch,scal2ros)( *kappa_cast, *(alpha1 +11*inca), *(pi1_r +11) ); \
					PASTEMAC(ch,scal2ros)( *kappa_cast, *(alpha1 +12*inca), *(pi1_r +12) ); \
					PASTEMAC(ch,scal2ros)( *kappa_cast, *(alpha1 +13*inca), *(pi1_r +13) ); \
					PASTEMAC(ch,scal2ros)( *kappa_cast, *(alpha1 +14*inca), *(pi1_r +14) ); \
					PASTEMAC(ch,scal2ros)( *kappa_cast, *(alpha1 +15*inca), *(pi1_r +15) ); \
\
					alpha1 += lda; \
					pi1_r  += ldp; \
				} \
			} \
		} \
	} \
	else if ( bli_is_io_packed( schema ) ) \
	{ \
		if ( PASTEMAC(ch,eq1)( *kappa_cast ) ) \
		{ \
			if ( bli_is_conj( conja ) ) \
			{ \
				for ( ; n != 0; --n ) \
				{ \
					PASTEMAC(chr,copys)( -*(alpha1_i + 0*inca2), *(pi1_r + 0) ); \
					PASTEMAC(chr,copys)( -*(alpha1_i + 1*inca2), *(pi1_r + 1) ); \
					PASTEMAC(chr,copys)( -*(alpha1_i + 2*inca2), *(pi1_r + 2) ); \
					PASTEMAC(chr,copys)( -*(alpha1_i + 3*inca2), *(pi1_r + 3) ); \
					PASTEMAC(chr,copys)( -*(alpha1_i + 4*inca2), *(pi1_r + 4) ); \
					PASTEMAC(chr,copys)( -*(alpha1_i + 5*inca2), *(pi1_r + 5) ); \
					PASTEMAC(chr,copys)( -*(alpha1_i + 6*inca2), *(pi1_r + 6) ); \
					PASTEMAC(chr,copys)( -*(alpha1_i + 7*inca2), *(pi1_r + 7) ); \
					PASTEMAC(chr,copys)( -*(alpha1_i + 8*inca2), *(pi1_r + 8) ); \
					PASTEMAC(chr,copys)( -*(alpha1_i + 9*inca2), *(pi1_r + 9) ); \
					PASTEMAC(chr,copys)( -*(alpha1_i +10*inca2), *(pi1_r +10) ); \
					PASTEMAC(chr,copys)( -*(alpha1_i +11*inca2), *(pi1_r +11) ); \
					PASTEMAC(chr,copys)( -*(alpha1_i +12*inca2), *(pi1_r +12) ); \
					PASTEMAC(chr,copys)( -*(alpha1_i +13*inca2), *(pi1_r +13) ); \
					PASTEMAC(chr,copys)( -*(alpha1_i +14*inca2), *(pi1_r +14) ); \
					PASTEMAC(chr,copys)( -*(alpha1_i +15*inca2), *(pi1_r +15) ); \
\
					alpha1_i += lda2; \
					pi1_r    += ldp; \
				} \
			} \
			else \
			{ \
				for ( ; n != 0; --n ) \
				{ \
					PASTEMAC(chr,copys)( *(alpha1_i + 0*inca2), *(pi1_r + 0) ); \
					PASTEMAC(chr,copys)( *(alpha1_i + 1*inca2), *(pi1_r + 1) ); \
					PASTEMAC(chr,copys)( *(alpha1_i + 2*inca2), *(pi1_r + 2) ); \
					PASTEMAC(chr,copys)( *(alpha1_i + 3*inca2), *(pi1_r + 3) ); \
					PASTEMAC(chr,copys)( *(alpha1_i + 4*inca2), *(pi1_r + 4) ); \
					PASTEMAC(chr,copys)( *(alpha1_i + 5*inca2), *(pi1_r + 5) ); \
					PASTEMAC(chr,copys)( *(alpha1_i + 6*inca2), *(pi1_r + 6) ); \
					PASTEMAC(chr,copys)( *(alpha1_i + 7*inca2), *(pi1_r + 7) ); \
					PASTEMAC(chr,copys)( *(alpha1_i + 8*inca2), *(pi1_r + 8) ); \
					PASTEMAC(chr,copys)( *(alpha1_i + 9*inca2), *(pi1_r + 9) ); \
					PASTEMAC(chr,copys)( *(alpha1_i +10*inca2), *(pi1_r +10) ); \
					PASTEMAC(chr,copys)( *(alpha1_i +11*inca2), *(pi1_r +11) ); \
					PASTEMAC(chr,copys)( *(alpha1_i +12*inca2), *(pi1_r +12) ); \
					PASTEMAC(chr,copys)( *(alpha1_i +13*inca2), *(pi1_r +13) ); \
					PASTEMAC(chr,copys)( *(alpha1_i +14*inca2), *(pi1_r +14) ); \
					PASTEMAC(chr,copys)( *(alpha1_i +15*inca2), *(pi1_r +15) ); \
\
					alpha1_i += lda2; \
					pi1_r    += ldp; \
				} \
			} \
		} \
		else \
		{ \
			if ( bli_is_conj( conja ) ) \
			{ \
				for ( ; n != 0; --n ) \
				{ \
					PASTEMAC(ch,scal2jios)( *kappa_cast, *(alpha1 + 0*inca), *(pi1_r + 0) ); \
					PASTEMAC(ch,scal2jios)( *kappa_cast, *(alpha1 + 1*inca), *(pi1_r + 1) ); \
					PASTEMAC(ch,scal2jios)( *kappa_cast, *(alpha1 + 2*inca), *(pi1_r + 2) ); \
					PASTEMAC(ch,scal2jios)( *kappa_cast, *(alpha1 + 3*inca), *(pi1_r + 3) ); \
					PASTEMAC(ch,scal2jios)( *kappa_cast, *(alpha1 + 4*inca), *(pi1_r + 4) ); \
					PASTEMAC(ch,scal2jios)( *kappa_cast, *(alpha1 + 5*inca), *(pi1_r + 5) ); \
					PASTEMAC(ch,scal2jios)( *kappa_cast, *(alpha1 + 6*inca), *(pi1_r + 6) ); \
					PASTEMAC(ch,scal2jios)( *kappa_cast, *(alpha1 + 7*inca), *(pi1_r + 7) ); \
					PASTEMAC(ch,scal2jios)( *kappa_cast, *(alpha1 + 8*inca), *(pi1_r + 8) ); \
					PASTEMAC(ch,scal2jios)( *kappa_cast, *(alpha1 + 9*inca), *(pi1_r + 9) ); \
					PASTEMAC(ch,scal2jios)( *kappa_cast, *(alpha1 +10*inca), *(pi1_r +10) ); \
					PASTEMAC(ch,scal2jios)( *kappa_cast, *(alpha1 +11*inca), *(pi1_r +11) ); \
					PASTEMAC(ch,scal2jios)( *kappa_cast, *(alpha1 +12*inca), *(pi1_r +12) ); \
					PASTEMAC(ch,scal2jios)( *kappa_cast, *(alpha1 +13*inca), *(pi1_r +13) ); \
					PASTEMAC(ch,scal2jios)( *kappa_cast, *(alpha1 +14*inca), *(pi1_r +14) ); \
					PASTEMAC(ch,scal2jios)( *kappa_cast, *(alpha1 +15*inca), *(pi1_r +15) ); \
\
					alpha1 += lda; \
					pi1_r  += ldp; \
				} \
			} \
			else \
			{ \
				for ( ; n != 0; --n ) \
				{ \
					PASTEMAC(ch,scal2ios)( *kappa_cast, *(alpha1 + 0*inca), *(pi1_r + 0) ); \
					PASTEMAC(ch,scal2ios)( *kappa_cast, *(alpha1 + 1*inca), *(pi1_r + 1) ); \
					PASTEMAC(ch,scal2ios)( *kappa_cast, *(alpha1 + 2*inca), *(pi1_r + 2) ); \
					PASTEMAC(ch,scal2ios)( *kappa_cast, *(alpha1 + 3*inca), *(pi1_r + 3) ); \
					PASTEMAC(ch,scal2ios)( *kappa_cast, *(alpha1 + 4*inca), *(pi1_r + 4) ); \
					PASTEMAC(ch,scal2ios)( *kappa_cast, *(alpha1 + 5*inca), *(pi1_r + 5) ); \
					PASTEMAC(ch,scal2ios)( *kappa_cast, *(alpha1 + 6*inca), *(pi1_r + 6) ); \
					PASTEMAC(ch,scal2ios)( *kappa_cast, *(alpha1 + 7*inca), *(pi1_r + 7) ); \
					PASTEMAC(ch,scal2ios)( *kappa_cast, *(alpha1 + 8*inca), *(pi1_r + 8) ); \
					PASTEMAC(ch,scal2ios)( *kappa_cast, *(alpha1 + 9*inca), *(pi1_r + 9) ); \
					PASTEMAC(ch,scal2ios)( *kappa_cast, *(alpha1 +10*inca), *(pi1_r +10) ); \
					PASTEMAC(ch,scal2ios)( *kappa_cast, *(alpha1 +11*inca), *(pi1_r +11) ); \
					PASTEMAC(ch,scal2ios)( *kappa_cast, *(alpha1 +12*inca), *(pi1_r +12) ); \
					PASTEMAC(ch,scal2ios)( *kappa_cast, *(alpha1 +13*inca), *(pi1_r +13) ); \
					PASTEMAC(ch,scal2ios)( *kappa_cast, *(alpha1 +14*inca), *(pi1_r +14) ); \
					PASTEMAC(ch,scal2ios)( *kappa_cast, *(alpha1 +15*inca), *(pi1_r +15) ); \
\
					alpha1 += lda; \
					pi1_r  += ldp; \
				} \
			} \
		} \
	} \
	else /* if ( bli_is_rpi_packed( schema ) ) */ \
	{ \
		if ( PASTEMAC(ch,eq1)( *kappa_cast ) ) \
		{ \
			if ( bli_is_conj( conja ) ) \
			{ \
				for ( ; n != 0; --n ) \
				{ \
					PASTEMAC(chr,add3s)( *(alpha1_r + 0*inca2), -*(alpha1_i + 0*inca2), *(pi1_r + 0) ); \
					PASTEMAC(chr,add3s)( *(alpha1_r + 1*inca2), -*(alpha1_i + 1*inca2), *(pi1_r + 1) ); \
					PASTEMAC(chr,add3s)( *(alpha1_r + 2*inca2), -*(alpha1_i + 2*inca2), *(pi1_r + 2) ); \
					PASTEMAC(chr,add3s)( *(alpha1_r + 3*inca2), -*(alpha1_i + 3*inca2), *(pi1_r + 3) ); \
					PASTEMAC(chr,add3s)( *(alpha1_r + 4*inca2), -*(alpha1_i + 4*inca2), *(pi1_r + 4) ); \
					PASTEMAC(chr,add3s)( *(alpha1_r + 5*inca2), -*(alpha1_i + 5*inca2), *(pi1_r + 5) ); \
					PASTEMAC(chr,add3s)( *(alpha1_r + 6*inca2), -*(alpha1_i + 6*inca2), *(pi1_r + 6) ); \
					PASTEMAC(chr,add3s)( *(alpha1_r + 7*inca2), -*(alpha1_i + 7*inca2), *(pi1_r + 7) ); \
					PASTEMAC(chr,add3s)( *(alpha1_r + 8*inca2), -*(alpha1_i + 8*inca2), *(pi1_r + 8) ); \
					PASTEMAC(chr,add3s)( *(alpha1_r + 9*inca2), -*(alpha1_i + 9*inca2), *(pi1_r + 9) ); \
					PASTEMAC(chr,add3s)( *(alpha1_r +10*inca2), -*(alpha1_i +10*inca2), *(pi1_r +10) ); \
					PASTEMAC(chr,add3s)( *(alpha1_r +11*inca2), -*(alpha1_i +11*inca2), *(pi1_r +11) ); \
					PASTEMAC(chr,add3s)( *(alpha1_r +12*inca2), -*(alpha1_i +12*inca2), *(pi1_r +12) ); \
					PASTEMAC(chr,add3s)( *(alpha1_r +13*inca2), -*(alpha1_i +13*inca2), *(pi1_r +13) ); \
					PASTEMAC(chr,add3s)( *(alpha1_r +14*inca2), -*(alpha1_i +14*inca2), *(pi1_r +14) ); \
					PASTEMAC(chr,add3s)( *(alpha1_r +15*inca2), -*(alpha1_i +15*inca2), *(pi1_r +15) ); \
\
					alpha1_r += lda2; \
					alpha1_i += lda2; \
					pi1_r    += ldp; \
				} \
			} \
			else \
			{ \
				for ( ; n != 0; --n ) \
				{ \
					PASTEMAC(chr,add3s)( *(alpha1_r + 0*inca2), *(alpha1_i + 0*inca2), *(pi1_r + 0) ); \
					PASTEMAC(chr,add3s)( *(alpha1_r + 1*inca2), *(alpha1_i + 1*inca2), *(pi1_r + 1) ); \
					PASTEMAC(chr,add3s)( *(alpha1_r + 2*inca2), *(alpha1_i + 2*inca2), *(pi1_r + 2) ); \
					PASTEMAC(chr,add3s)( *(alpha1_r + 3*inca2), *(alpha1_i + 3*inca2), *(pi1_r + 3) ); \
					PASTEMAC(chr,add3s)( *(alpha1_r + 4*inca2), *(alpha1_i + 4*inca2), *(pi1_r + 4) ); \
					PASTEMAC(chr,add3s)( *(alpha1_r + 5*inca2), *(alpha1_i + 5*inca2), *(pi1_r + 5) ); \
					PASTEMAC(chr,add3s)( *(alpha1_r + 6*inca2), *(alpha1_i + 6*inca2), *(pi1_r + 6) ); \
					PASTEMAC(chr,add3s)( *(alpha1_r + 7*inca2), *(alpha1_i + 7*inca2), *(pi1_r + 7) ); \
					PASTEMAC(chr,add3s)( *(alpha1_r + 8*inca2), *(alpha1_i + 8*inca2), *(pi1_r + 8) ); \
					PASTEMAC(chr,add3s)( *(alpha1_r + 9*inca2), *(alpha1_i + 9*inca2), *(pi1_r + 9) ); \
					PASTEMAC(chr,add3s)( *(alpha1_r +10*inca2), *(alpha1_i +10*inca2), *(pi1_r +10) ); \
					PASTEMAC(chr,add3s)( *(alpha1_r +11*inca2), *(alpha1_i +11*inca2), *(pi1_r +11) ); \
					PASTEMAC(chr,add3s)( *(alpha1_r +12*inca2), *(alpha1_i +12*inca2), *(pi1_r +12) ); \
					PASTEMAC(chr,add3s)( *(alpha1_r +13*inca2), *(alpha1_i +13*inca2), *(pi1_r +13) ); \
					PASTEMAC(chr,add3s)( *(alpha1_r +14*inca2), *(alpha1_i +14*inca2), *(pi1_r +14) ); \
					PASTEMAC(chr,add3s)( *(alpha1_r +15*inca2), *(alpha1_i +15*inca2), *(pi1_r +15) ); \
\
					alpha1_r += lda2; \
					alpha1_i += lda2; \
					pi1_r    += ldp; \
				} \
			} \
		} \
		else \
		{ \
			if ( bli_is_conj( conja ) ) \
			{ \
				for ( ; n != 0; --n ) \
				{ \
					PASTEMAC(ch,scal2jrpis)( *kappa_cast, *(alpha1 + 0*inca), *(pi1_r + 0) ); \
					PASTEMAC(ch,scal2jrpis)( *kappa_cast, *(alpha1 + 1*inca), *(pi1_r + 1) ); \
					PASTEMAC(ch,scal2jrpis)( *kappa_cast, *(alpha1 + 2*inca), *(pi1_r + 2) ); \
					PASTEMAC(ch,scal2jrpis)( *kappa_cast, *(alpha1 + 3*inca), *(pi1_r + 3) ); \
					PASTEMAC(ch,scal2jrpis)( *kappa_cast, *(alpha1 + 4*inca), *(pi1_r + 4) ); \
					PASTEMAC(ch,scal2jrpis)( *kappa_cast, *(alpha1 + 5*inca), *(pi1_r + 5) ); \
					PASTEMAC(ch,scal2jrpis)( *kappa_cast, *(alpha1 + 6*inca), *(pi1_r + 6) ); \
					PASTEMAC(ch,scal2jrpis)( *kappa_cast, *(alpha1 + 7*inca), *(pi1_r + 7) ); \
					PASTEMAC(ch,scal2jrpis)( *kappa_cast, *(alpha1 + 8*inca), *(pi1_r + 8) ); \
					PASTEMAC(ch,scal2jrpis)( *kappa_cast, *(alpha1 + 9*inca), *(pi1_r + 9) ); \
					PASTEMAC(ch,scal2jrpis)( *kappa_cast, *(alpha1 +10*inca), *(pi1_r +10) ); \
					PASTEMAC(ch,scal2jrpis)( *kappa_cast, *(alpha1 +11*inca), *(pi1_r +11) ); \
					PASTEMAC(ch,scal2jrpis)( *kappa_cast, *(alpha1 +12*inca), *(pi1_r +12) ); \
					PASTEMAC(ch,scal2jrpis)( *kappa_cast, *(alpha1 +13*inca), *(pi1_r +13) ); \
					PASTEMAC(ch,scal2jrpis)( *kappa_cast, *(alpha1 +14*inca), *(pi1_r +14) ); \
					PASTEMAC(ch,scal2jrpis)( *kappa_cast, *(alpha1 +15*inca), *(pi1_r +15) ); \
\
					alpha1 += lda; \
					pi1_r  += ldp; \
				} \
			} \
			else \
			{ \
				for ( ; n != 0; --n ) \
				{ \
					PASTEMAC(ch,scal2rpis)( *kappa_cast, *(alpha1 + 0*inca), *(pi1_r + 0) ); \
					PASTEMAC(ch,scal2rpis)( *kappa_cast, *(alpha1 + 1*inca), *(pi1_r + 1) ); \
					PASTEMAC(ch,scal2rpis)( *kappa_cast, *(alpha1 + 2*inca), *(pi1_r + 2) ); \
					PASTEMAC(ch,scal2rpis)( *kappa_cast, *(alpha1 + 3*inca), *(pi1_r + 3) ); \
					PASTEMAC(ch,scal2rpis)( *kappa_cast, *(alpha1 + 4*inca), *(pi1_r + 4) ); \
					PASTEMAC(ch,scal2rpis)( *kappa_cast, *(alpha1 + 5*inca), *(pi1_r + 5) ); \
					PASTEMAC(ch,scal2rpis)( *kappa_cast, *(alpha1 + 6*inca), *(pi1_r + 6) ); \
					PASTEMAC(ch,scal2rpis)( *kappa_cast, *(alpha1 + 7*inca), *(pi1_r + 7) ); \
					PASTEMAC(ch,scal2rpis)( *kappa_cast, *(alpha1 + 8*inca), *(pi1_r + 8) ); \
					PASTEMAC(ch,scal2rpis)( *kappa_cast, *(alpha1 + 9*inca), *(pi1_r + 9) ); \
					PASTEMAC(ch,scal2rpis)( *kappa_cast, *(alpha1 +10*inca), *(pi1_r +10) ); \
					PASTEMAC(ch,scal2rpis)( *kappa_cast, *(alpha1 +11*inca), *(pi1_r +11) ); \
					PASTEMAC(ch,scal2rpis)( *kappa_cast, *(alpha1 +12*inca), *(pi1_r +12) ); \
					PASTEMAC(ch,scal2rpis)( *kappa_cast, *(alpha1 +13*inca), *(pi1_r +13) ); \
					PASTEMAC(ch,scal2rpis)( *kappa_cast, *(alpha1 +14*inca), *(pi1_r +14) ); \
					PASTEMAC(ch,scal2rpis)( *kappa_cast, *(alpha1 +15*inca), *(pi1_r +15) ); \
\
					alpha1 += lda; \
					pi1_r  += ldp; \
				} \
			} \
		} \
	} \
}

INSERT_GENTFUNCCO_BASIC0( packm_16xk_rih_ref )



#undef  GENTFUNCCO
#define GENTFUNCCO( ctype, ctype_r, ch, chr, varname ) \
\
void PASTEMAC(ch,varname) \
     ( \
       conj_t         conja, \
       pack_t         schema, \
       dim_t          n, \
       void* restrict kappa, \
       void* restrict a, inc_t inca, inc_t lda, \
       void* restrict p,             inc_t ldp  \
     ) \
{ \
	const inc_t       inca2      = 2 * inca; \
	const inc_t       lda2       = 2 * lda; \
\
	ctype*            kappa_cast = kappa; \
	ctype*   restrict alpha1     = a; \
	ctype_r* restrict alpha1_r   = ( ctype_r* )a; \
	ctype_r* restrict alpha1_i   = ( ctype_r* )a + 1; \
	ctype_r* restrict pi1_r      = ( ctype_r* )p; \
\
\
	if ( bli_is_ro_packed( schema ) ) \
	{ \
		if ( PASTEMAC(ch,eq1)( *kappa_cast ) ) \
		{ \
			/* This works regardless of conja since we are only copying
			   the real part. */ \
			{ \
				for ( ; n != 0; --n ) \
				{ \
					PASTEMAC(chr,copys)( *(alpha1_r + 0*inca2), *(pi1_r + 0) ); \
					PASTEMAC(chr,copys)( *(alpha1_r + 1*inca2), *(pi1_r + 1) ); \
					PASTEMAC(chr,copys)( *(alpha1_r + 2*inca2), *(pi1_r + 2) ); \
					PASTEMAC(chr,copys)( *(alpha1_r + 3*inca2), *(pi1_r + 3) ); \
					PASTEMAC(chr,copys)( *(alpha1_r + 4*inca2), *(pi1_r + 4) ); \
					PASTEMAC(chr,copys)( *(alpha1_r + 5*inca2), *(pi1_r + 5) ); \
					PASTEMAC(chr,copys)( *(alpha1_r + 6*inca2), *(pi1_r + 6) ); \
					PASTEMAC(chr,copys)( *(alpha1_r + 7*inca2), *(pi1_r + 7) ); \
					PASTEMAC(chr,copys)( *(alpha1_r + 8*inca2), *(pi1_r + 8) ); \
					PASTEMAC(chr,copys)( *(alpha1_r + 9*inca2), *(pi1_r + 9) ); \
					PASTEMAC(chr,copys)( *(alpha1_r +10*inca2), *(pi1_r +10) ); \
					PASTEMAC(chr,copys)( *(alpha1_r +11*inca2), *(pi1_r +11) ); \
					PASTEMAC(chr,copys)( *(alpha1_r +12*inca2), *(pi1_r +12) ); \
					PASTEMAC(chr,copys)( *(alpha1_r +13*inca2), *(pi1_r +13) ); \
					PASTEMAC(chr,copys)( *(alpha1_r +14*inca2), *(pi1_r +14) ); \
					PASTEMAC(chr,copys)( *(alpha1_r +15*inca2), *(pi1_r +15) ); \
					PASTEMAC(chr,copys)( *(alpha1_r +16*inca2), *(pi1_r +16) ); \
					PASTEMAC(chr,copys)( *(alpha1_r +17*inca2), *(pi1_r +17) ); \
					PASTEMAC(chr,copys)( *(alpha1_r +18*inca2), *(pi1_r +18) ); \
					PASTEMAC(chr,copys)( *(alpha1_r +19*inca2), *(pi1_r +19) ); \
					PASTEMAC(chr,copys)( *(alpha1_r +20*inca2), *(pi1_r +20) ); \
					PASTEMAC(chr,copys)( *(alpha1_r +21*inca2), *(pi1_r +21) ); \
					PASTEMAC(chr,copys)( *(alpha1_r +22*inca2), *(pi1_r +22) ); \
					PASTEMAC(chr,copys)( *(alpha1_r +23*inca2), *(pi1_r +23) ); \
					PASTEMAC(chr,copys)( *(alpha1_r +24*inca2), *(pi1_r +24) ); \
					PASTEMAC(chr,copys)( *(alpha1_r +25*inca2), *(pi1_r +25) ); \
					PASTEMAC(chr,copys)( *(alpha1_r +26*inca2), *(pi1_r +26) ); \
					PASTEMAC(chr,copys)( *(alpha1_r +27*inca2), *(pi1_r +27) ); \
					PASTEMAC(chr,copys)( *(alpha1_r +28*inca2), *(pi1_r +28) ); \
					PASTEMAC(chr,copys)( *(alpha1_r +29*inca2), *(pi1_r +29) ); \
\
					alpha1_r += lda2; \
					pi1_r    += ldp; \
				} \
			} \
		} \
		else \
		{ \
			if ( bli_is_conj( conja ) ) \
			{ \
				for ( ; n != 0; --n ) \
				{ \
					PASTEMAC(ch,scal2jros)( *kappa_cast, *(alpha1 + 0*inca), *(pi1_r + 0) ); \
					PASTEMAC(ch,scal2jros)( *kappa_cast, *(alpha1 + 1*inca), *(pi1_r + 1) ); \
					PASTEMAC(ch,scal2jros)( *kappa_cast, *(alpha1 + 2*inca), *(pi1_r + 2) ); \
					PASTEMAC(ch,scal2jros)( *kappa_cast, *(alpha1 + 3*inca), *(pi1_r + 3) ); \
					PASTEMAC(ch,scal2jros)( *kappa_cast, *(alpha1 + 4*inca), *(pi1_r + 4) ); \
					PASTEMAC(ch,scal2jros)( *kappa_cast, *(alpha1 + 5*inca), *(pi1_r + 5) ); \
					PASTEMAC(ch,scal2jros)( *kappa_cast, *(alpha1 + 6*inca), *(pi1_r + 6) ); \
					PASTEMAC(ch,scal2jros)( *kappa_cast, *(alpha1 + 7*inca), *(pi1_r + 7) ); \
					PASTEMAC(ch,scal2jros)( *kappa_cast, *(alpha1 + 8*inca), *(pi1_r + 8) ); \
					PASTEMAC(ch,scal2jros)( *kappa_cast, *(alpha1 + 9*inca), *(pi1_r + 9) ); \
					PASTEMAC(ch,scal2jros)( *kappa_cast, *(alpha1 +10*inca), *(pi1_r +10) ); \
					PASTEMAC(ch,scal2jros)( *kappa_cast, *(alpha1 +11*inca), *(pi1_r +11) ); \
					PASTEMAC(ch,scal2jros)( *kappa_cast, *(alpha1 +12*inca), *(pi1_r +12) ); \
					PASTEMAC(ch,scal2jros)( *kappa_cast, *(alpha1 +13*inca), *(pi1_r +13) ); \
					PASTEMAC(ch,scal2jros)( *kappa_cast, *(alpha1 +14*inca), *(pi1_r +14) ); \
					PASTEMAC(ch,scal2jros)( *kappa_cast, *(alpha1 +15*inca), *(pi1_r +15) ); \
					PASTEMAC(ch,scal2jros)( *kappa_cast, *(alpha1 +16*inca), *(pi1_r +16) ); \
					PASTEMAC(ch,scal2jros)( *kappa_cast, *(alpha1 +17*inca), *(pi1_r +17) ); \
					PASTEMAC(ch,scal2jros)( *kappa_cast, *(alpha1 +18*inca), *(pi1_r +18) ); \
					PASTEMAC(ch,scal2jros)( *kappa_cast, *(alpha1 +19*inca), *(pi1_r +19) ); \
					PASTEMAC(ch,scal2jros)( *kappa_cast, *(alpha1 +20*inca), *(pi1_r +20) ); \
					PASTEMAC(ch,scal2jros)( *kappa_cast, *(alpha1 +21*inca), *(pi1_r +21) ); \
					PASTEMAC(ch,scal2jros)( *kappa_cast, *(alpha1 +22*inca), *(pi1_r +22) ); \
					PASTEMAC(ch,scal2jros)( *kappa_cast, *(alpha1 +23*inca), *(pi1_r +23) ); \
					PASTEMAC(ch,scal2jros)( *kappa_cast, *(alpha1 +24*inca), *(pi1_r +24) ); \
					PASTEMAC(ch,scal2jros)( *kappa_cast, *(alpha1 +25*inca), *(pi1_r +25) ); \
					PASTEMAC(ch,scal2jros)( *kappa_cast, *(alpha1 +26*inca), *(pi1_r +26) ); \
					PASTEMAC(ch,scal2jros)( *kappa_cast, *(alpha1 +27*inca), *(pi1_r +27) ); \
					PASTEMAC(ch,scal2jros)( *kappa_cast, *(alpha1 +28*inca), *(pi1_r +28) ); \
					PASTEMAC(ch,scal2jros)( *kappa_cast, *(alpha1 +29*inca), *(pi1_r +29) ); \
\
					alpha1 += lda; \
					pi1_r  += ldp; \
				} \
			} \
			else \
			{ \
				for ( ; n != 0; --n ) \
				{ \
					PASTEMAC(ch,scal2ros)( *kappa_cast, *(alpha1 + 0*inca), *(pi1_r + 0) ); \
					PASTEMAC(ch,scal2ros)( *kappa_cast, *(alpha1 + 1*inca), *(pi1_r + 1) ); \
					PASTEMAC(ch,scal2ros)( *kappa_cast, *(alpha1 + 2*inca), *(pi1_r + 2) ); \
					PASTEMAC(ch,scal2ros)( *kappa_cast, *(alpha1 + 3*inca), *(pi1_r + 3) ); \
					PASTEMAC(ch,scal2ros)( *kappa_cast, *(alpha1 + 4*inca), *(pi1_r + 4) ); \
					PASTEMAC(ch,scal2ros)( *kappa_cast, *(alpha1 + 5*inca), *(pi1_r + 5) ); \
					PASTEMAC(ch,scal2ros)( *kappa_cast, *(alpha1 + 6*inca), *(pi1_r + 6) ); \
					PASTEMAC(ch,scal2ros)( *kappa_cast, *(alpha1 + 7*inca), *(pi1_r + 7) ); \
					PASTEMAC(ch,scal2ros)( *kappa_cast, *(alpha1 + 8*inca), *(pi1_r + 8) ); \
					PASTEMAC(ch,scal2ros)( *kappa_cast, *(alpha1 + 9*inca), *(pi1_r + 9) ); \
					PASTEMAC(ch,scal2ros)( *kappa_cast, *(alpha1 +10*inca), *(pi1_r +10) ); \
					PASTEMAC(ch,scal2ros)( *kappa_cast, *(alpha1 +11*inca), *(pi1_r +11) ); \
					PASTEMAC(ch,scal2ros)( *kappa_cast, *(alpha1 +12*inca), *(pi1_r +12) ); \
					PASTEMAC(ch,scal2ros)( *kappa_cast, *(alpha1 +13*inca), *(pi1_r +13) ); \
					PASTEMAC(ch,scal2ros)( *kappa_cast, *(alpha1 +14*inca), *(pi1_r +14) ); \
					PASTEMAC(ch,scal2ros)( *kappa_cast, *(alpha1 +15*inca), *(pi1_r +15) ); \
					PASTEMAC(ch,scal2ros)( *kappa_cast, *(alpha1 +16*inca), *(pi1_r +16) ); \
					PASTEMAC(ch,scal2ros)( *kappa_cast, *(alpha1 +17*inca), *(pi1_r +17) ); \
					PASTEMAC(ch,scal2ros)( *kappa_cast, *(alpha1 +18*inca), *(pi1_r +18) ); \
					PASTEMAC(ch,scal2ros)( *kappa_cast, *(alpha1 +19*inca), *(pi1_r +19) ); \
					PASTEMAC(ch,scal2ros)( *kappa_cast, *(alpha1 +20*inca), *(pi1_r +20) ); \
					PASTEMAC(ch,scal2ros)( *kappa_cast, *(alpha1 +21*inca), *(pi1_r +21) ); \
					PASTEMAC(ch,scal2ros)( *kappa_cast, *(alpha1 +22*inca), *(pi1_r +22) ); \
					PASTEMAC(ch,scal2ros)( *kappa_cast, *(alpha1 +23*inca), *(pi1_r +23) ); \
					PASTEMAC(ch,scal2ros)( *kappa_cast, *(alpha1 +24*inca), *(pi1_r +24) ); \
					PASTEMAC(ch,scal2ros)( *kappa_cast, *(alpha1 +25*inca), *(pi1_r +25) ); \
					PASTEMAC(ch,scal2ros)( *kappa_cast, *(alpha1 +26*inca), *(pi1_r +26) ); \
					PASTEMAC(ch,scal2ros)( *kappa_cast, *(alpha1 +27*inca), *(pi1_r +27) ); \
					PASTEMAC(ch,scal2ros)( *kappa_cast, *(alpha1 +28*inca), *(pi1_r +28) ); \
					PASTEMAC(ch,scal2ros)( *kappa_cast, *(alpha1 +29*inca), *(pi1_r +29) ); \
\
					alpha1 += lda; \
					pi1_r  += ldp; \
				} \
			} \
		} \
	} \
	else if ( bli_is_io_packed( schema ) ) \
	{ \
		if ( PASTEMAC(ch,eq1)( *kappa_cast ) ) \
		{ \
			if ( bli_is_conj( conja ) ) \
			{ \
				for ( ; n != 0; --n ) \
				{ \
					PASTEMAC(chr,copys)( -*(alpha1_i + 0*inca2), *(pi1_r + 0) ); \
					PASTEMAC(chr,copys)( -*(alpha1_i + 1*inca2), *(pi1_r + 1) ); \
					PASTEMAC(chr,copys)( -*(alpha1_i + 2*inca2), *(pi1_r + 2) ); \
					PASTEMAC(chr,copys)( -*(alpha1_i + 3*inca2), *(pi1_r + 3) ); \
					PASTEMAC(chr,copys)( -*(alpha1_i + 4*inca2), *(pi1_r + 4) ); \
					PASTEMAC(chr,copys)( -*(alpha1_i + 5*inca2), *(pi1_r + 5) ); \
					PASTEMAC(chr,copys)( -*(alpha1_i + 6*inca2), *(pi1_r + 6) ); \
					PASTEMAC(chr,copys)( -*(alpha1_i + 7*inca2), *(pi1_r + 7) ); \
					PASTEMAC(chr,copys)( -*(alpha1_i + 8*inca2), *(pi1_r + 8) ); \
					PASTEMAC(chr,copys)( -*(alpha1_i + 9*inca2), *(pi1_r + 9) ); \
					PASTEMAC(chr,copys)( -*(alpha1_i +10*inca2), *(pi1_r +10) ); \
					PASTEMAC(chr,copys)( -*(alpha1_i +11*inca2), *(pi1_r +11) ); \
					PASTEMAC(chr,copys)( -*(alpha1_i +12*inca2), *(pi1_r +12) ); \
					PASTEMAC(chr,copys)( -*(alpha1_i +13*inca2), *(pi1_r +13) ); \
					PASTEMAC(chr,copys)( -*(alpha1_i +14*inca2), *(pi1_r +14) ); \
					PASTEMAC(chr,copys)( -*(alpha1_i +15*inca2), *(pi1_r +15) ); \
					PASTEMAC(chr,copys)( -*(alpha1_i +16*inca2), *(pi1_r +16) ); \
					PASTEMAC(chr,copys)( -*(alpha1_i +17*inca2), *(pi1_r +17) ); \
					PASTEMAC(chr,copys)( -*(alpha1_i +18*inca2), *(pi1_r +18) ); \
					PASTEMAC(chr,copys)( -*(alpha1_i +19*inca2), *(pi1_r +19) ); \
					PASTEMAC(chr,copys)( -*(alpha1_i +20*inca2), *(pi1_r +20) ); \
					PASTEMAC(chr,copys)( -*(alpha1_i +21*inca2), *(pi1_r +21) ); \
					PASTEMAC(chr,copys)( -*(alpha1_i +22*inca2), *(pi1_r +22) ); \
					PASTEMAC(chr,copys)( -*(alpha1_i +23*inca2), *(pi1_r +23) ); \
					PASTEMAC(chr,copys)( -*(alpha1_i +24*inca2), *(pi1_r +24) ); \
					PASTEMAC(chr,copys)( -*(alpha1_i +25*inca2), *(pi1_r +25) ); \
					PASTEMAC(chr,copys)( -*(alpha1_i +26*inca2), *(pi1_r +26) ); \
					PASTEMAC(chr,copys)( -*(alpha1_i +27*inca2), *(pi1_r +27) ); \
					PASTEMAC(chr,copys)( -*(alpha1_i +28*inca2), *(pi1_r +28) ); \
					PASTEMAC(chr,copys)( -*(alpha1_i +29*inca2), *(pi1_r +29) ); \
\
					alpha1_i += lda2; \
					pi1_r    += ldp; \
				} \
			} \
			else \
			{ \
				for ( ; n != 0; --n ) \
				{ \
					PASTEMAC(chr,copys)( *(alpha1_i + 0*inca2), *(pi1_r + 0) ); \
					PASTEMAC(chr,copys)( *(alpha1_i + 1*inca2), *(pi1_r + 1) ); \
					PASTEMAC(chr,copys)( *(alpha1_i + 2*inca2), *(pi1_r + 2) ); \
					PASTEMAC(chr,copys)( *(alpha1_i + 3*inca2), *(pi1_r + 3) ); \
					PASTEMAC(chr,copys)( *(alpha1_i + 4*inca2), *(pi1_r + 4) ); \
					PASTEMAC(chr,copys)( *(alpha1_i + 5*inca2), *(pi1_r + 5) ); \
					PASTEMAC(chr,copys)( *(alpha1_i + 6*inca2), *(pi1_r + 6) ); \
					PASTEMAC(chr,copys)( *(alpha1_i + 7*inca2), *(pi1_r + 7) ); \
					PASTEMAC(chr,copys)( *(alpha1_i + 8*inca2), *(pi1_r + 8) ); \
					PASTEMAC(chr,copys)( *(alpha1_i + 9*inca2), *(pi1_r + 9) ); \
					PASTEMAC(chr,copys)( *(alpha1_i +10*inca2), *(pi1_r +10) ); \
					PASTEMAC(chr,copys)( *(alpha1_i +11*inca2), *(pi1_r +11) ); \
					PASTEMAC(chr,copys)( *(alpha1_i +12*inca2), *(pi1_r +12) ); \
					PASTEMAC(chr,copys)( *(alpha1_i +13*inca2), *(pi1_r +13) ); \
					PASTEMAC(chr,copys)( *(alpha1_i +14*inca2), *(pi1_r +14) ); \
					PASTEMAC(chr,copys)( *(alpha1_i +15*inca2), *(pi1_r +15) ); \
					PASTEMAC(chr,copys)( *(alpha1_i +16*inca2), *(pi1_r +16) ); \
					PASTEMAC(chr,copys)( *(alpha1_i +17*inca2), *(pi1_r +17) ); \
					PASTEMAC(chr,copys)( *(alpha1_i +18*inca2), *(pi1_r +18) ); \
					PASTEMAC(chr,copys)( *(alpha1_i +19*inca2), *(pi1_r +19) ); \
					PASTEMAC(chr,copys)( *(alpha1_i +20*inca2), *(pi1_r +20) ); \
					PASTEMAC(chr,copys)( *(alpha1_i +21*inca2), *(pi1_r +21) ); \
					PASTEMAC(chr,copys)( *(alpha1_i +22*inca2), *(pi1_r +22) ); \
					PASTEMAC(chr,copys)( *(alpha1_i +23*inca2), *(pi1_r +23) ); \
					PASTEMAC(chr,copys)( *(alpha1_i +24*inca2), *(pi1_r +24) ); \
					PASTEMAC(chr,copys)( *(alpha1_i +25*inca2), *(pi1_r +25) ); \
					PASTEMAC(chr,copys)( *(alpha1_i +26*inca2), *(pi1_r +26) ); \
					PASTEMAC(chr,copys)( *(alpha1_i +27*inca2), *(pi1_r +27) ); \
					PASTEMAC(chr,copys)( *(alpha1_i +28*inca2), *(pi1_r +28) ); \
					PASTEMAC(chr,copys)( *(alpha1_i +29*inca2), *(pi1_r +29) ); \
\
					alpha1_i += lda2; \
					pi1_r    += ldp; \
				} \
			} \
		} \
		else \
		{ \
			if ( bli_is_conj( conja ) ) \
			{ \
				for ( ; n != 0; --n ) \
				{ \
					PASTEMAC(ch,scal2jios)( *kappa_cast, *(alpha1 + 0*inca), *(pi1_r + 0) ); \
					PASTEMAC(ch,scal2jios)( *kappa_cast, *(alpha1 + 1*inca), *(pi1_r + 1) ); \
					PASTEMAC(ch,scal2jios)( *kappa_cast, *(alpha1 + 2*inca), *(pi1_r + 2) ); \
					PASTEMAC(ch,scal2jios)( *kappa_cast, *(alpha1 + 3*inca), *(pi1_r + 3) ); \
					PASTEMAC(ch,scal2jios)( *kappa_cast, *(alpha1 + 4*inca), *(pi1_r + 4) ); \
					PASTEMAC(ch,scal2jios)( *kappa_cast, *(alpha1 + 5*inca), *(pi1_r + 5) ); \
					PASTEMAC(ch,scal2jios)( *kappa_cast, *(alpha1 + 6*inca), *(pi1_r + 6) ); \
					PASTEMAC(ch,scal2jios)( *kappa_cast, *(alpha1 + 7*inca), *(pi1_r + 7) ); \
					PASTEMAC(ch,scal2jios)( *kappa_cast, *(alpha1 + 8*inca), *(pi1_r + 8) ); \
					PASTEMAC(ch,scal2jios)( *kappa_cast, *(alpha1 + 9*inca), *(pi1_r + 9) ); \
					PASTEMAC(ch,scal2jios)( *kappa_cast, *(alpha1 +10*inca), *(pi1_r +10) ); \
					PASTEMAC(ch,scal2jios)( *kappa_cast, *(alpha1 +11*inca), *(pi1_r +11) ); \
					PASTEMAC(ch,scal2jios)( *kappa_cast, *(alpha1 +12*inca), *(pi1_r +12) ); \
					PASTEMAC(ch,scal2jios)( *kappa_cast, *(alpha1 +13*inca), *(pi1_r +13) ); \
					PASTEMAC(ch,scal2jios)( *kappa_cast, *(alpha1 +14*inca), *(pi1_r +14) ); \
					PASTEMAC(ch,scal2jios)( *kappa_cast, *(alpha1 +15*inca), *(pi1_r +15) ); \
					PASTEMAC(ch,scal2jios)( *kappa_cast, *(alpha1 +16*inca), *(pi1_r +16) ); \
					PASTEMAC(ch,scal2jios)( *kappa_cast, *(alpha1 +17*inca), *(pi1_r +17) ); \
					PASTEMAC(ch,scal2jios)( *kappa_cast, *(alpha1 +18*inca), *(pi1_r +18) ); \
					PASTEMAC(ch,scal2jios)( *kappa_cast, *(alpha1 +19*inca), *(pi1_r +19) ); \
					PASTEMAC(ch,scal2jios)( *kappa_cast, *(alpha1 +20*inca), *(pi1_r +20) ); \
					PASTEMAC(ch,scal2jios)( *kappa_cast, *(alpha1 +21*inca), *(pi1_r +21) ); \
					PASTEMAC(ch,scal2jios)( *kappa_cast, *(alpha1 +22*inca), *(pi1_r +22) ); \
					PASTEMAC(ch,scal2jios)( *kappa_cast, *(alpha1 +23*inca), *(pi1_r +23) ); \
					PASTEMAC(ch,scal2jios)( *kappa_cast, *(alpha1 +24*inca), *(pi1_r +24) ); \
					PASTEMAC(ch,scal2jios)( *kappa_cast, *(alpha1 +25*inca), *(pi1_r +25) ); \
					PASTEMAC(ch,scal2jios)( *kappa_cast, *(alpha1 +26*inca), *(pi1_r +26) ); \
					PASTEMAC(ch,scal2jios)( *kappa_cast, *(alpha1 +27*inca), *(pi1_r +27) ); \
					PASTEMAC(ch,scal2jios)( *kappa_cast, *(alpha1 +28*inca), *(pi1_r +28) ); \
					PASTEMAC(ch,scal2jios)( *kappa_cast, *(alpha1 +29*inca), *(pi1_r +29) ); \
\
					alpha1 += lda; \
					pi1_r  += ldp; \
				} \
			} \
			else \
			{ \
				for ( ; n != 0; --n ) \
				{ \
					PASTEMAC(ch,scal2ios)( *kappa_cast, *(alpha1 + 0*inca), *(pi1_r + 0) ); \
					PASTEMAC(ch,scal2ios)( *kappa_cast, *(alpha1 + 1*inca), *(pi1_r + 1) ); \
					PASTEMAC(ch,scal2ios)( *kappa_cast, *(alpha1 + 2*inca), *(pi1_r + 2) ); \
					PASTEMAC(ch,scal2ios)( *kappa_cast, *(alpha1 + 3*inca), *(pi1_r + 3) ); \
					PASTEMAC(ch,scal2ios)( *kappa_cast, *(alpha1 + 4*inca), *(pi1_r + 4) ); \
					PASTEMAC(ch,scal2ios)( *kappa_cast, *(alpha1 + 5*inca), *(pi1_r + 5) ); \
					PASTEMAC(ch,scal2ios)( *kappa_cast, *(alpha1 + 6*inca), *(pi1_r + 6) ); \
					PASTEMAC(ch,scal2ios)( *kappa_cast, *(alpha1 + 7*inca), *(pi1_r + 7) ); \
					PASTEMAC(ch,scal2ios)( *kappa_cast, *(alpha1 + 8*inca), *(pi1_r + 8) ); \
					PASTEMAC(ch,scal2ios)( *kappa_cast, *(alpha1 + 9*inca), *(pi1_r + 9) ); \
					PASTEMAC(ch,scal2ios)( *kappa_cast, *(alpha1 +10*inca), *(pi1_r +10) ); \
					PASTEMAC(ch,scal2ios)( *kappa_cast, *(alpha1 +11*inca), *(pi1_r +11) ); \
					PASTEMAC(ch,scal2ios)( *kappa_cast, *(alpha1 +12*inca), *(pi1_r +12) ); \
					PASTEMAC(ch,scal2ios)( *kappa_cast, *(alpha1 +13*inca), *(pi1_r +13) ); \
					PASTEMAC(ch,scal2ios)( *kappa_cast, *(alpha1 +14*inca), *(pi1_r +14) ); \
					PASTEMAC(ch,scal2ios)( *kappa_cast, *(alpha1 +15*inca), *(pi1_r +15) ); \
					PASTEMAC(ch,scal2ios)( *kappa_cast, *(alpha1 +16*inca), *(pi1_r +16) ); \
					PASTEMAC(ch,scal2ios)( *kappa_cast, *(alpha1 +17*inca), *(pi1_r +17) ); \
					PASTEMAC(ch,scal2ios)( *kappa_cast, *(alpha1 +18*inca), *(pi1_r +18) ); \
					PASTEMAC(ch,scal2ios)( *kappa_cast, *(alpha1 +19*inca), *(pi1_r +19) ); \
					PASTEMAC(ch,scal2ios)( *kappa_cast, *(alpha1 +20*inca), *(pi1_r +20) ); \
					PASTEMAC(ch,scal2ios)( *kappa_cast, *(alpha1 +21*inca), *(pi1_r +21) ); \
					PASTEMAC(ch,scal2ios)( *kappa_cast, *(alpha1 +22*inca), *(pi1_r +22) ); \
					PASTEMAC(ch,scal2ios)( *kappa_cast, *(alpha1 +23*inca), *(pi1_r +23) ); \
					PASTEMAC(ch,scal2ios)( *kappa_cast, *(alpha1 +24*inca), *(pi1_r +24) ); \
					PASTEMAC(ch,scal2ios)( *kappa_cast, *(alpha1 +25*inca), *(pi1_r +25) ); \
					PASTEMAC(ch,scal2ios)( *kappa_cast, *(alpha1 +26*inca), *(pi1_r +26) ); \
					PASTEMAC(ch,scal2ios)( *kappa_cast, *(alpha1 +27*inca), *(pi1_r +27) ); \
					PASTEMAC(ch,scal2ios)( *kappa_cast, *(alpha1 +28*inca), *(pi1_r +28) ); \
					PASTEMAC(ch,scal2ios)( *kappa_cast, *(alpha1 +29*inca), *(pi1_r +29) ); \
\
					alpha1 += lda; \
					pi1_r  += ldp; \
				} \
			} \
		} \
	} \
	else /* if ( bli_is_rpi_packed( schema ) ) */ \
	{ \
		if ( PASTEMAC(ch,eq1)( *kappa_cast ) ) \
		{ \
			if ( bli_is_conj( conja ) ) \
			{ \
				for ( ; n != 0; --n ) \
				{ \
					PASTEMAC(chr,add3s)( *(alpha1_r + 0*inca2), -*(alpha1_i + 0*inca2), *(pi1_r + 0) ); \
					PASTEMAC(chr,add3s)( *(alpha1_r + 1*inca2), -*(alpha1_i + 1*inca2), *(pi1_r + 1) ); \
					PASTEMAC(chr,add3s)( *(alpha1_r + 2*inca2), -*(alpha1_i + 2*inca2), *(pi1_r + 2) ); \
					PASTEMAC(chr,add3s)( *(alpha1_r + 3*inca2), -*(alpha1_i + 3*inca2), *(pi1_r + 3) ); \
					PASTEMAC(chr,add3s)( *(alpha1_r + 4*inca2), -*(alpha1_i + 4*inca2), *(pi1_r + 4) ); \
					PASTEMAC(chr,add3s)( *(alpha1_r + 5*inca2), -*(alpha1_i + 5*inca2), *(pi1_r + 5) ); \
					PASTEMAC(chr,add3s)( *(alpha1_r + 6*inca2), -*(alpha1_i + 6*inca2), *(pi1_r + 6) ); \
					PASTEMAC(chr,add3s)( *(alpha1_r + 7*inca2), -*(alpha1_i + 7*inca2), *(pi1_r + 7) ); \
					PASTEMAC(chr,add3s)( *(alpha1_r + 8*inca2), -*(alpha1_i + 8*inca2), *(pi1_r + 8) ); \
					PASTEMAC(chr,add3s)( *(alpha1_r + 9*inca2), -*(alpha1_i + 9*inca2), *(pi1_r + 9) ); \
					PASTEMAC(chr,add3s)( *(alpha1_r +10*inca2), -*(alpha1_i +10*inca2), *(pi1_r +10) ); \
					PASTEMAC(chr,add3s)( *(alpha1_r +11*inca2), -*(alpha1_i +11*inca2), *(pi1_r +11) ); \
					PASTEMAC(chr,add3s)( *(alpha1_r +12*inca2), -*(alpha1_i +12*inca2), *(pi1_r +12) ); \
					PASTEMAC(chr,add3s)( *(alpha1_r +13*inca2), -*(alpha1_i +13*inca2), *(pi1_r +13) ); \
					PASTEMAC(chr,add3s)( *(alpha1_r +14*inca2), -*(alpha1_i +14*inca2), *(pi1_r +14) ); \
					PASTEMAC(chr,add3s)( *(alpha1_r +15*inca2), -*(alpha1_i +15*inca2), *(pi1_r +15) ); \
					PASTEMAC(chr,add3s)( *(alpha1_r +16*inca2), -*(alpha1_i +16*inca2), *(pi1_r +16) ); \
					PASTEMAC(chr,add3s)( *(alpha1_r +17*inca2), -*(alpha1_i +17*inca2), *(pi1_r +17) ); \
					PASTEMAC(chr,add3s)( *(alpha1_r +18*inca2), -*(alpha1_i +18*inca2), *(pi1_r +18) ); \
					PASTEMAC(chr,add3s)( *(alpha1_r +19*inca2), -*(alpha1_i +19*inca2), *(pi1_r +19) ); \
					PASTEMAC(chr,add3s)( *(alpha1_r +20*inca2), -*(alpha1_i +20*inca2), *(pi1_r +20) ); \
					PASTEMAC(chr,add3s)( *(alpha1_r +21*inca2), -*(alpha1_i +21*inca2), *(pi1_r +21) ); \
					PASTEMAC(chr,add3s)( *(alpha1_r +22*inca2), -*(alpha1_i +22*inca2), *(pi1_r +22) ); \
					PASTEMAC(chr,add3s)( *(alpha1_r +23*inca2), -*(alpha1_i +23*inca2), *(pi1_r +23) ); \
					PASTEMAC(chr,add3s)( *(alpha1_r +24*inca2), -*(alpha1_i +24*inca2), *(pi1_r +24) ); \
					PASTEMAC(chr,add3s)( *(alpha1_r +25*inca2), -*(alpha1_i +25*inca2), *(pi1_r +25) ); \
					PASTEMAC(chr,add3s)( *(alpha1_r +26*inca2), -*(alpha1_i +26*inca2), *(pi1_r +26) ); \
					PASTEMAC(chr,add3s)( *(alpha1_r +27*inca2), -*(alpha1_i +27*inca2), *(pi1_r +27) ); \
					PASTEMAC(chr,add3s)( *(alpha1_r +28*inca2), -*(alpha1_i +28*inca2), *(pi1_r +28) ); \
					PASTEMAC(chr,add3s)( *(alpha1_r +29*inca2), -*(alpha1_i +29*inca2), *(pi1_r +29) ); \
\
					alpha1_r += lda2; \
					alpha1_i += lda2; \
					pi1_r    += ldp; \
				} \
			} \
			else \
			{ \
				for ( ; n != 0; --n ) \
				{ \
					PASTEMAC(chr,add3s)( *(alpha1_r + 0*inca2), *(alpha1_i + 0*inca2), *(pi1_r + 0) ); \
					PASTEMAC(chr,add3s)( *(alpha1_r + 1*inca2), *(alpha1_i + 1*inca2), *(pi1_r + 1) ); \
					PASTEMAC(chr,add3s)( *(alpha1_r + 2*inca2), *(alpha1_i + 2*inca2), *(pi1_r + 2) ); \
					PASTEMAC(chr,add3s)( *(alpha1_r + 3*inca2), *(alpha1_i + 3*inca2), *(pi1_r + 3) ); \
					PASTEMAC(chr,add3s)( *(alpha1_r + 4*inca2), *(alpha1_i + 4*inca2), *(pi1_r + 4) ); \
					PASTEMAC(chr,add3s)( *(alpha1_r + 5*inca2), *(alpha1_i + 5*inca2), *(pi1_r + 5) ); \
					PASTEMAC(chr,add3s)( *(alpha1_r + 6*inca2), *(alpha1_i + 6*inca2), *(pi1_r + 6) ); \
					PASTEMAC(chr,add3s)( *(alpha1_r + 7*inca2), *(alpha1_i + 7*inca2), *(pi1_r + 7) ); \
					PASTEMAC(chr,add3s)( *(alpha1_r + 8*inca2), *(alpha1_i + 8*inca2), *(pi1_r + 8) ); \
					PASTEMAC(chr,add3s)( *(alpha1_r + 9*inca2), *(alpha1_i + 9*inca2), *(pi1_r + 9) ); \
					PASTEMAC(chr,add3s)( *(alpha1_r +10*inca2), *(alpha1_i +10*inca2), *(pi1_r +10) ); \
					PASTEMAC(chr,add3s)( *(alpha1_r +11*inca2), *(alpha1_i +11*inca2), *(pi1_r +11) ); \
					PASTEMAC(chr,add3s)( *(alpha1_r +12*inca2), *(alpha1_i +12*inca2), *(pi1_r +12) ); \
					PASTEMAC(chr,add3s)( *(alpha1_r +13*inca2), *(alpha1_i +13*inca2), *(pi1_r +13) ); \
					PASTEMAC(chr,add3s)( *(alpha1_r +14*inca2), *(alpha1_i +14*inca2), *(pi1_r +14) ); \
					PASTEMAC(chr,add3s)( *(alpha1_r +15*inca2), *(alpha1_i +15*inca2), *(pi1_r +15) ); \
					PASTEMAC(chr,add3s)( *(alpha1_r +16*inca2), *(alpha1_i +16*inca2), *(pi1_r +16) ); \
					PASTEMAC(chr,add3s)( *(alpha1_r +17*inca2), *(alpha1_i +17*inca2), *(pi1_r +17) ); \
					PASTEMAC(chr,add3s)( *(alpha1_r +18*inca2), *(alpha1_i +18*inca2), *(pi1_r +18) ); \
					PASTEMAC(chr,add3s)( *(alpha1_r +19*inca2), *(alpha1_i +19*inca2), *(pi1_r +19) ); \
					PASTEMAC(chr,add3s)( *(alpha1_r +20*inca2), *(alpha1_i +20*inca2), *(pi1_r +20) ); \
					PASTEMAC(chr,add3s)( *(alpha1_r +21*inca2), *(alpha1_i +21*inca2), *(pi1_r +21) ); \
					PASTEMAC(chr,add3s)( *(alpha1_r +22*inca2), *(alpha1_i +22*inca2), *(pi1_r +22) ); \
					PASTEMAC(chr,add3s)( *(alpha1_r +23*inca2), *(alpha1_i +23*inca2), *(pi1_r +23) ); \
					PASTEMAC(chr,add3s)( *(alpha1_r +24*inca2), *(alpha1_i +24*inca2), *(pi1_r +24) ); \
					PASTEMAC(chr,add3s)( *(alpha1_r +25*inca2), *(alpha1_i +25*inca2), *(pi1_r +25) ); \
					PASTEMAC(chr,add3s)( *(alpha1_r +26*inca2), *(alpha1_i +26*inca2), *(pi1_r +26) ); \
					PASTEMAC(chr,add3s)( *(alpha1_r +27*inca2), *(alpha1_i +27*inca2), *(pi1_r +27) ); \
					PASTEMAC(chr,add3s)( *(alpha1_r +28*inca2), *(alpha1_i +28*inca2), *(pi1_r +28) ); \
					PASTEMAC(chr,add3s)( *(alpha1_r +29*inca2), *(alpha1_i +29*inca2), *(pi1_r +29) ); \
\
					alpha1_r += lda2; \
					alpha1_i += lda2; \
					pi1_r    += ldp; \
				} \
			} \
		} \
		else \
		{ \
			if ( bli_is_conj( conja ) ) \
			{ \
				for ( ; n != 0; --n ) \
				{ \
					PASTEMAC(ch,scal2jrpis)( *kappa_cast, *(alpha1 + 0*inca), *(pi1_r + 0) ); \
					PASTEMAC(ch,scal2jrpis)( *kappa_cast, *(alpha1 + 1*inca), *(pi1_r + 1) ); \
					PASTEMAC(ch,scal2jrpis)( *kappa_cast, *(alpha1 + 2*inca), *(pi1_r + 2) ); \
					PASTEMAC(ch,scal2jrpis)( *kappa_cast, *(alpha1 + 3*inca), *(pi1_r + 3) ); \
					PASTEMAC(ch,scal2jrpis)( *kappa_cast, *(alpha1 + 4*inca), *(pi1_r + 4) ); \
					PASTEMAC(ch,scal2jrpis)( *kappa_cast, *(alpha1 + 5*inca), *(pi1_r + 5) ); \
					PASTEMAC(ch,scal2jrpis)( *kappa_cast, *(alpha1 + 6*inca), *(pi1_r + 6) ); \
					PASTEMAC(ch,scal2jrpis)( *kappa_cast, *(alpha1 + 7*inca), *(pi1_r + 7) ); \
					PASTEMAC(ch,scal2jrpis)( *kappa_cast, *(alpha1 + 8*inca), *(pi1_r + 8) ); \
					PASTEMAC(ch,scal2jrpis)( *kappa_cast, *(alpha1 + 9*inca), *(pi1_r + 9) ); \
					PASTEMAC(ch,scal2jrpis)( *kappa_cast, *(alpha1 +10*inca), *(pi1_r +10) ); \
					PASTEMAC(ch,scal2jrpis)( *kappa_cast, *(alpha1 +11*inca), *(pi1_r +11) ); \
					PASTEMAC(ch,scal2jrpis)( *kappa_cast, *(alpha1 +12*inca), *(pi1_r +12) ); \
					PASTEMAC(ch,scal2jrpis)( *kappa_cast, *(alpha1 +13*inca), *(pi1_r +13) ); \
					PASTEMAC(ch,scal2jrpis)( *kappa_cast, *(alpha1 +14*inca), *(pi1_r +14) ); \
					PASTEMAC(ch,scal2jrpis)( *kappa_cast, *(alpha1 +15*inca), *(pi1_r +15) ); \
					PASTEMAC(ch,scal2jrpis)( *kappa_cast, *(alpha1 +16*inca), *(pi1_r +16) ); \
					PASTEMAC(ch,scal2jrpis)( *kappa_cast, *(alpha1 +17*inca), *(pi1_r +17) ); \
					PASTEMAC(ch,scal2jrpis)( *kappa_cast, *(alpha1 +18*inca), *(pi1_r +18) ); \
					PASTEMAC(ch,scal2jrpis)( *kappa_cast, *(alpha1 +19*inca), *(pi1_r +19) ); \
					PASTEMAC(ch,scal2jrpis)( *kappa_cast, *(alpha1 +20*inca), *(pi1_r +20) ); \
					PASTEMAC(ch,scal2jrpis)( *kappa_cast, *(alpha1 +21*inca), *(pi1_r +21) ); \
					PASTEMAC(ch,scal2jrpis)( *kappa_cast, *(alpha1 +22*inca), *(pi1_r +22) ); \
					PASTEMAC(ch,scal2jrpis)( *kappa_cast, *(alpha1 +23*inca), *(pi1_r +23) ); \
					PASTEMAC(ch,scal2jrpis)( *kappa_cast, *(alpha1 +24*inca), *(pi1_r +24) ); \
					PASTEMAC(ch,scal2jrpis)( *kappa_cast, *(alpha1 +25*inca), *(pi1_r +25) ); \
					PASTEMAC(ch,scal2jrpis)( *kappa_cast, *(alpha1 +26*inca), *(pi1_r +26) ); \
					PASTEMAC(ch,scal2jrpis)( *kappa_cast, *(alpha1 +27*inca), *(pi1_r +27) ); \
					PASTEMAC(ch,scal2jrpis)( *kappa_cast, *(alpha1 +28*inca), *(pi1_r +28) ); \
					PASTEMAC(ch,scal2jrpis)( *kappa_cast, *(alpha1 +29*inca), *(pi1_r +29) ); \
\
					alpha1 += lda; \
					pi1_r  += ldp; \
				} \
			} \
			else \
			{ \
				for ( ; n != 0; --n ) \
				{ \
					PASTEMAC(ch,scal2rpis)( *kappa_cast, *(alpha1 + 0*inca), *(pi1_r + 0) ); \
					PASTEMAC(ch,scal2rpis)( *kappa_cast, *(alpha1 + 1*inca), *(pi1_r + 1) ); \
					PASTEMAC(ch,scal2rpis)( *kappa_cast, *(alpha1 + 2*inca), *(pi1_r + 2) ); \
					PASTEMAC(ch,scal2rpis)( *kappa_cast, *(alpha1 + 3*inca), *(pi1_r + 3) ); \
					PASTEMAC(ch,scal2rpis)( *kappa_cast, *(alpha1 + 4*inca), *(pi1_r + 4) ); \
					PASTEMAC(ch,scal2rpis)( *kappa_cast, *(alpha1 + 5*inca), *(pi1_r + 5) ); \
					PASTEMAC(ch,scal2rpis)( *kappa_cast, *(alpha1 + 6*inca), *(pi1_r + 6) ); \
					PASTEMAC(ch,scal2rpis)( *kappa_cast, *(alpha1 + 7*inca), *(pi1_r + 7) ); \
					PASTEMAC(ch,scal2rpis)( *kappa_cast, *(alpha1 + 8*inca), *(pi1_r + 8) ); \
					PASTEMAC(ch,scal2rpis)( *kappa_cast, *(alpha1 + 9*inca), *(pi1_r + 9) ); \
					PASTEMAC(ch,scal2rpis)( *kappa_cast, *(alpha1 +10*inca), *(pi1_r +10) ); \
					PASTEMAC(ch,scal2rpis)( *kappa_cast, *(alpha1 +11*inca), *(pi1_r +11) ); \
					PASTEMAC(ch,scal2rpis)( *kappa_cast, *(alpha1 +12*inca), *(pi1_r +12) ); \
					PASTEMAC(ch,scal2rpis)( *kappa_cast, *(alpha1 +13*inca), *(pi1_r +13) ); \
					PASTEMAC(ch,scal2rpis)( *kappa_cast, *(alpha1 +14*inca), *(pi1_r +14) ); \
					PASTEMAC(ch,scal2rpis)( *kappa_cast, *(alpha1 +15*inca), *(pi1_r +15) ); \
					PASTEMAC(ch,scal2rpis)( *kappa_cast, *(alpha1 +16*inca), *(pi1_r +16) ); \
					PASTEMAC(ch,scal2rpis)( *kappa_cast, *(alpha1 +17*inca), *(pi1_r +17) ); \
					PASTEMAC(ch,scal2rpis)( *kappa_cast, *(alpha1 +18*inca), *(pi1_r +18) ); \
					PASTEMAC(ch,scal2rpis)( *kappa_cast, *(alpha1 +19*inca), *(pi1_r +19) ); \
					PASTEMAC(ch,scal2rpis)( *kappa_cast, *(alpha1 +20*inca), *(pi1_r +20) ); \
					PASTEMAC(ch,scal2rpis)( *kappa_cast, *(alpha1 +21*inca), *(pi1_r +21) ); \
					PASTEMAC(ch,scal2rpis)( *kappa_cast, *(alpha1 +22*inca), *(pi1_r +22) ); \
					PASTEMAC(ch,scal2rpis)( *kappa_cast, *(alpha1 +23*inca), *(pi1_r +23) ); \
					PASTEMAC(ch,scal2rpis)( *kappa_cast, *(alpha1 +24*inca), *(pi1_r +24) ); \
					PASTEMAC(ch,scal2rpis)( *kappa_cast, *(alpha1 +25*inca), *(pi1_r +25) ); \
					PASTEMAC(ch,scal2rpis)( *kappa_cast, *(alpha1 +26*inca), *(pi1_r +26) ); \
					PASTEMAC(ch,scal2rpis)( *kappa_cast, *(alpha1 +27*inca), *(pi1_r +27) ); \
					PASTEMAC(ch,scal2rpis)( *kappa_cast, *(alpha1 +28*inca), *(pi1_r +28) ); \
					PASTEMAC(ch,scal2rpis)( *kappa_cast, *(alpha1 +29*inca), *(pi1_r +29) ); \
\
					alpha1 += lda; \
					pi1_r  += ldp; \
				} \
			} \
		} \
	} \
}

INSERT_GENTFUNCCO_BASIC0( packm_30xk_rih_ref )

