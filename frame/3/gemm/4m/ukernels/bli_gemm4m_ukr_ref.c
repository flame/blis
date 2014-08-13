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
   THEORY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
   (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
   OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

*/

#include "blis.h"

#undef  GENTFUNCCO
#define GENTFUNCCO( ctype, ctype_r, ch, chr, varname, gemmukr ) \
\
void PASTEMAC(ch,varname)( \
                           dim_t           k, \
                           ctype* restrict alpha, \
                           ctype* restrict a, \
                           ctype* restrict b, \
                           ctype* restrict beta, \
                           ctype* restrict c, inc_t rs_c, inc_t cs_c, \
                           auxinfo_t*      data \
                         ) \
{ \
	ctype_r           ct_r[ PASTEMAC(chr,mr) * \
	                        PASTEMAC(chr,nr) ] \
	                   __attribute__((aligned(BLIS_STACK_BUF_ALIGN_SIZE))); \
	ctype_r           ct_i[ PASTEMAC(chr,mr) * \
	                        PASTEMAC(chr,nr) ] \
	                   __attribute__((aligned(BLIS_STACK_BUF_ALIGN_SIZE))); \
	const inc_t       rs_ct     = 1; \
	const inc_t       cs_ct     = PASTEMAC(chr,mr); \
\
\
	const dim_t       m         = PASTEMAC(chr,mr); \
	const dim_t       n         = PASTEMAC(chr,nr); \
\
	const inc_t       ps_a      = bli_auxinfo_ps_a( data ); \
	const inc_t       ps_b      = bli_auxinfo_ps_b( data ); \
\
	ctype_r* restrict a_r       = ( ctype_r* )a; \
	ctype_r* restrict a_i       = ( ctype_r* )a + ps_a; \
\
	ctype_r* restrict b_r       = ( ctype_r* )b; \
	ctype_r* restrict b_i       = ( ctype_r* )b + ps_b; \
\
	ctype_r* restrict c_r       = ( ctype_r* )c; \
	ctype_r* restrict c_i       = ( ctype_r* )c + 1; \
\
	const inc_t       rs_c2     = 2 * rs_c; \
	const inc_t       cs_c2     = 2 * cs_c; \
\
	ctype_r* restrict one_r     = PASTEMAC(chr,1); \
	ctype_r* restrict zero_r    = PASTEMAC(chr,0); \
\
	ctype_r           alpha_r   = PASTEMAC(ch,real)( *alpha ); \
	ctype_r           alpha_i   = PASTEMAC(ch,imag)( *alpha ); \
\
	ctype_r           m_alpha_r = -PASTEMAC(ch,real)( *alpha ); \
\
	void*             a_next    = bli_auxinfo_next_a( data ); \
	void*             b_next    = bli_auxinfo_next_b( data ); \
\
	dim_t             i, j; \
\
\
	/* SAFETY CHECK: The higher level implementation should never
	   allow an alpha with non-zero imaginary component to be passed
	   in, because it can't be applied properly using the 4m method.
	   If alpha is not real, then something is very wrong. */ \
	if ( !PASTEMAC(chr,eq0)( alpha_i ) ) \
		bli_check_error_code( BLIS_NOT_YET_IMPLEMENTED ); \
\
\
	/* c.r = beta.r * c.r + alpha.r * a.r * b.r
	                      - alpha.r * a.i * b.i;
	   c.i = beta.r * c.i + alpha.r * a.r * b.i
	                      + alpha.r * a.i * b.r; */ \
\
	bli_auxinfo_set_next_ab( a_r, b_i, *data ); \
\
	/* ct.r = alpha.r * a.r * b.r; */ \
	PASTEMAC(chr,gemmukr)( k, \
	                       &alpha_r, \
	                       a_r, \
	                       b_r, \
	                       zero_r, \
	                       ct_r, rs_ct, cs_ct, \
	                       data ); \
\
	bli_auxinfo_set_next_ab( a_i, b_r, *data ); \
\
	/* ct.i = alpha.r * a.r * b.i; */ \
	PASTEMAC(chr,gemmukr)( k, \
	                       &alpha_r, \
	                       a_r, \
	                       b_i, \
	                       zero_r, \
	                       ct_i, rs_ct, cs_ct, \
	                       data ); \
\
	bli_auxinfo_set_next_ab( a_i, b_i, *data ); \
\
	/* ct.i += alpha.r * a.i * b.r; */ \
	PASTEMAC(chr,gemmukr)( k, \
	                       &alpha_r, \
	                       a_i, \
	                       b_r, \
	                       one_r, \
	                       ct_i, rs_ct, cs_ct, \
	                       data ); \
\
	bli_auxinfo_set_next_ab( a_next, b_next, *data ); \
\
	/* ct.r += -alpha.r * a.i * b.i; */ \
	PASTEMAC(chr,gemmukr)( k, \
	                       &m_alpha_r, \
	                       a_i, \
	                       b_i, \
	                       one_r, \
	                       ct_r, rs_ct, cs_ct, \
	                       data ); \
\
\
	/* Accumulate the final result in ct back to c. */ \
	if ( PASTEMAC(ch,eq1)( *beta ) ) \
	{ \
		for ( j = 0; j < n; ++j ) \
		for ( i = 0; i < m; ++i ) \
		PASTEMAC(ch,addris)( *(ct_r + i*rs_ct + j*cs_ct), \
		                     *(ct_i + i*rs_ct + j*cs_ct), \
		                     *(c_r  + i*rs_c2 + j*cs_c2), \
		                     *(c_i  + i*rs_c2 + j*cs_c2) ); \
	} \
	else if ( PASTEMAC(ch,eq0)( *beta ) ) \
	{ \
		for ( j = 0; j < n; ++j ) \
		for ( i = 0; i < m; ++i ) \
		PASTEMAC(ch,copyris)( *(ct_r + i*rs_ct + j*cs_ct), \
		                      *(ct_i + i*rs_ct + j*cs_ct), \
		                      *(c_r  + i*rs_c2 + j*cs_c2), \
		                      *(c_i  + i*rs_c2 + j*cs_c2) ); \
	} \
	else \
	{ \
		ctype_r beta_r = PASTEMAC(ch,real)( *beta ); \
		ctype_r beta_i = PASTEMAC(ch,imag)( *beta ); \
\
		for ( j = 0; j < n; ++j ) \
		for ( i = 0; i < m; ++i ) \
		PASTEMAC(ch,xpbyris)( *(ct_r + i*rs_ct + j*cs_ct), \
		                      *(ct_i + i*rs_ct + j*cs_ct), \
		                      beta_r, \
		                      beta_i, \
		                      *(c_r  + i*rs_c2 + j*cs_c2), \
		                      *(c_i  + i*rs_c2 + j*cs_c2) ); \
	} \
}

INSERT_GENTFUNCCO_BASIC( gemm4m_ukr_ref, GEMM_UKERNEL )

