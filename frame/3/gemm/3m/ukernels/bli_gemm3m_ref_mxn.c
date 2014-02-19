/*

   BLIS    
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2014, The University of Texas

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
	ctype_r           ab_r[ PASTEMAC(chr,mr) * \
	                        PASTEMAC(chr,nr) ] \
	                   __attribute__((aligned(BLIS_STACK_BUF_ALIGN_SIZE))); \
	ctype_r           ab_i[ PASTEMAC(chr,mr) * \
	                        PASTEMAC(chr,nr) ] \
	                   __attribute__((aligned(BLIS_STACK_BUF_ALIGN_SIZE))); \
	const inc_t       rs_ab     = 1; \
	const inc_t       cs_ab     = PASTEMAC(chr,mr); \
\
\
	const dim_t       m         = PASTEMAC(chr,mr); \
	const dim_t       n         = PASTEMAC(chr,nr); \
\
	const inc_t       ps_a      = ( bli_auxinfo_ps_a( data ) * 2 ) / 3; \
	const inc_t       ps_b      = ( bli_auxinfo_ps_b( data ) * 2 ) / 3; \
\
	ctype_r* restrict a_r       = ( ctype_r* )a; \
	ctype_r* restrict a_i       = ( ctype_r* )a +   ps_a; \
	ctype_r* restrict a_ri      = ( ctype_r* )a + 2*ps_a; \
\
	ctype_r* restrict b_r       = ( ctype_r* )b; \
	ctype_r* restrict b_i       = ( ctype_r* )b +   ps_b; \
	ctype_r* restrict b_ri      = ( ctype_r* )b + 2*ps_b; \
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
	ctype_r* restrict alpha_r   = &PASTEMAC(ch,real)( *alpha ); \
	ctype_r* restrict alpha_i   = &PASTEMAC(ch,imag)( *alpha ); \
\
	ctype_r* restrict beta_r    = &PASTEMAC(ch,real)( *beta ); \
	ctype_r* restrict beta_i    = &PASTEMAC(ch,imag)( *beta ); \
\
	void*             a_next    = bli_auxinfo_next_a( data ); \
	void*             b_next    = bli_auxinfo_next_b( data ); \
\
	dim_t             i, j; \
\
\
	/* SAFETY CHECK: The higher level implementation should never
	   allow an alpha with non-zero imaginary component to be passed
	   in, because it can't be applied properly using the 3m method.
	   If alpha is not real, then something is very wrong. */ \
	if ( !PASTEMAC(chr,eq0)( *alpha_i ) ) \
		bli_check_error_code( BLIS_NOT_YET_IMPLEMENTED ); \
\
\
	/* Copy the contents of c to a temporary buffer ct. */ \
	if ( !PASTEMAC(chr,eq0)( *beta_i ) ) \
	{ \
		/* We can handle a non-zero imaginary component on beta, but to do
		   so we have to manually scale c and then use beta == 1 for the
		   micro-kernel calls. */ \
		for ( i = 0; i < m; ++i ) \
		for ( j = 0; j < n; ++j ) \
		PASTEMAC(ch,scal2ris)( *beta_r, \
		                       *beta_i, \
		                       *(c_r  + i*rs_c2 + j*cs_c2), \
		                       *(c_i  + i*rs_c2 + j*cs_c2), \
		                       *(ct_r + i*rs_ct + j*cs_ct), \
		                       *(ct_i + i*rs_ct + j*cs_ct) ); \
\
		/* Use beta.r == 1.0. */ \
		beta_r = one_r; \
	} \
	else \
	{ \
		/* Copy c to ct without scaling. */ \
		for ( j = 0; j < n; ++j ) \
		for ( i = 0; i < m; ++i ) \
		PASTEMAC(ch,copyris)( *(c_r  + i*rs_c2 + j*cs_c2), \
		                      *(c_i  + i*rs_c2 + j*cs_c2), \
		                      *(ct_r + i*rs_ct + j*cs_ct), \
		                      *(ct_i + i*rs_ct + j*cs_ct) ); \
	} \
\
\
	/* c.r = beta.r * c.r                          + a.r * b.r - a.i * b.i;
	   c.i = beta.r * c.i + (a.r + a.i)(b.r + b.i) - a.r * b.r - a.i * b.i; */ \
\
	bli_auxinfo_set_next_ab( a_i, b_i, *data ); \
\
	/* ab.r = a.r * b.r; */ \
	PASTEMAC(chr,gemmukr)( k, \
	                       alpha_r, \
	                       a_r, \
	                       b_r, \
	                       zero_r, \
	                       ab_r, rs_ab, cs_ab, \
	                       data ); \
\
	bli_auxinfo_set_next_ab( a_ri, b_ri, *data ); \
\
	/* ab.i = a.i * b.i; */ \
	PASTEMAC(chr,gemmukr)( k, \
	                       alpha_r, \
	                       a_i, \
	                       b_i, \
	                       zero_r, \
	                       ab_i, rs_ab, cs_ab, \
	                       data ); \
\
	bli_auxinfo_set_next_ab( a_next, b_next, *data ); \
\
	/* ct.i = a.ri * b.ri; */ \
	PASTEMAC(chr,gemmukr)( k, \
	                       alpha_r, \
	                       a_ri, \
	                       b_ri, \
	                       beta_r, \
	                       ct_i, rs_ct, cs_ct, \
	                       data ); \
\
\
	/* ct.r = beta.r * ct.r + ab.r;
	   ct.r =          ct.r - ab.i;
	   ct.i =          ct.i - ab.r;
	   ct.i =          ct.i - ab.i; */ \
	for ( j = 0; j < n; ++j ) \
	for ( i = 0; i < m; ++i ) \
	{ \
		ctype_r alphabeta_r = *(ab_r + i*rs_ab + j*cs_ab); \
		ctype_r alphabeta_i = *(ab_i + i*rs_ab + j*cs_ab); \
		ctype_r gammat_r    = *(ct_r + i*rs_ct + j*cs_ct); \
		ctype_r gammat_i    = *(ct_i + i*rs_ct + j*cs_ct); \
\
		PASTEMAC(chr,scals)( *beta_r, gammat_r ); \
\
		PASTEMAC(chr,adds)( alphabeta_r, gammat_r ); \
		PASTEMAC(chr,subs)( alphabeta_i, gammat_r ); \
		PASTEMAC(chr,subs)( alphabeta_r, gammat_i ); \
		PASTEMAC(chr,subs)( alphabeta_i, gammat_i ); \
\
		/* Store the local values (from ct) back to c. */ \
		PASTEMAC(ch,copyris)( gammat_r, \
		                      gammat_i, \
		                      *(c_r + i*rs_c2 + j*cs_c2), \
		                      *(c_i + i*rs_c2 + j*cs_c2) ); \
	} \
}

INSERT_GENTFUNCCO_BASIC( gemm3m_ref_mxn, GEMM_UKERNEL )

