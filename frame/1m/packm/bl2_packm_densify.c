/*

   BLIS    
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2012, The University of Texas

   Redistribution and use in source and binary forms, with or without
   modification, are permitted provided that the following conditions are
   met:
    - Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.
    - Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.
    - Neither the name of The University of Texas nor the names of its
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

#include "blis2.h"

#undef  GENTFUNC
#define GENTFUNC( ctype, ch, opname, varname ) \
\
void PASTEMAC(ch,varname)( \
                           struc_t strucc, \
                           doff_t  diagoffc, \
                           uplo_t  uploc, \
                           trans_t transc, \
                           dim_t   m, \
                           dim_t   n, \
                           void*   beta, \
                           void*   c, inc_t rs_c, inc_t cs_c, \
                           void*   p, inc_t rs_p, inc_t cs_p \
                         ) \
{ \
	ctype* beta_cast = beta; \
	ctype* c_cast    = c; \
	ctype* p_cast    = p; \
	ctype* zero      = PASTEMAC(ch,0); \
\
	/* If uploc is upper or lower, then the structure of c is necessarily
	   non-dense (ie: Hermitian, symmetric, or triangular, where part of the
	   matrix is unstored). In these cases, when indicated by the densify
	   parameter, we want to fill in the unstored part of the matrix. How
	   this is done depends on the structure of c. */ \
	{ \
		/* The Hermitian and symmetric cases are almost identical, so we
		   handle them in one conditional block. */ \
		if ( bl2_is_hermitian( strucc ) || bl2_is_symmetric( strucc ) ) \
		{ \
			/* First we must reflect the region referenced to the opposite
			   side of the diagonal. */ \
			c_cast = c_cast + diagoffc * ( doff_t )cs_c + \
			                 -diagoffc * ( doff_t )rs_c; \
			bl2_negate_diag_offset( diagoffc ); \
			bl2_toggle_trans( transc ); \
			if      ( bl2_is_upper( uploc ) ) diagoffc += 1; \
			else if ( bl2_is_lower( uploc ) ) diagoffc -= 1; \
\
			/* If c is Hermitian, we need to apply a conjugation when
			   copying the region opposite the diagonal. */ \
			if ( bl2_is_hermitian( strucc ) ) \
				bl2_toggle_conj( transc ); \
\
			/* Copy the data from the region opposite the diagonal of c
			   (as specified by the original value of diagoffc). Notice
			   that we use a diag parameter of non-unit since we can
			   assume nothing about the neighboring off-diagonal. */ \
			PASTEMAC3(ch,ch,ch,scal2m)( diagoffc, \
			                            BLIS_NONUNIT_DIAG, \
			                            uploc, \
			                            transc, \
			                            m, \
			                            n, \
			                            beta_cast, \
			                            c_cast, rs_c, cs_c, \
			                            p_cast, rs_p, cs_p ); \
		} \
		else /* if ( bl2_is_triangular( strucc ) ) */ \
		{ \
			doff_t diagoffp = diagoffc; \
			uplo_t uplop    = uploc; \
\
			/* For this step we need the uplo and diagonal offset of p, which
			   we can derive from the parameters given. */ \
			if ( bl2_does_trans( transc ) ) \
			{ \
				bl2_negate_diag_offset( diagoffp ); \
				bl2_toggle_uplo( uplop ); \
			} \
\
			/* For triangular matrices, we wish to reference the region
			   strictly opposite the diagonal of c. This amounts to 
			   shifting the diagonal offset and then toggling uploc. */ \
			if      ( bl2_is_upper( uplop ) ) diagoffp -= 1; \
			else if ( bl2_is_lower( uplop ) ) diagoffp += 1; \
			bl2_toggle_uplo( uplop ); \
\
			/* Set the region opposite the diagonal of p to zero. */ \
			PASTEMAC2(ch,ch,setm)( diagoffp, \
			                       BLIS_NONUNIT_DIAG, \
			                       uplop, \
			                       m, \
			                       n, \
			                       zero, \
			                       p_cast, rs_p, cs_p ); \
		} \
	} \
}

INSERT_GENTFUNC_BASIC( packm_densify, packm_densify )

