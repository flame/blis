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
#define GENTFUNCCO( ctype, ctype_r, ch, chr, varname, gemmukr, trsmukr ) \
\
void PASTEMAC(ch,varname)( \
                           dim_t           k, \
                           ctype* restrict alpha, \
                           ctype* restrict a12, \
                           ctype* restrict a11, \
                           ctype* restrict b21, \
                           ctype* restrict b11, \
                           ctype* restrict c11, inc_t rs_c, inc_t cs_c, \
                           auxinfo_t*      data  \
                         ) \
{ \
	const dim_t       m           = PASTEMAC(chr,mr); \
	const dim_t       n           = PASTEMAC(chr,nr); \
\
	const inc_t       is_a        = bli_auxinfo_is_a( data ); \
	const inc_t       is_b        = bli_auxinfo_is_b( data ); \
\
	ctype_r* restrict a11_r       = ( ctype_r* )a11; \
\
	ctype_r* restrict a12_r       = ( ctype_r* )a12; \
	ctype_r* restrict a12_i       = ( ctype_r* )a12 + is_a; \
\
	ctype_r* restrict b11_r       = ( ctype_r* )b11; \
	ctype_r* restrict b11_i       = ( ctype_r* )b11 + is_b; \
\
	ctype_r* restrict b21_r       = ( ctype_r* )b21; \
	ctype_r* restrict b21_i       = ( ctype_r* )b21 + is_b; \
\
	const inc_t       rs_b        = PASTEMAC(chr,packnr); \
	const inc_t       cs_b        = 1; \
\
	ctype_r* restrict one_r       = PASTEMAC(chr,1); \
	ctype_r* restrict minus_one_r = PASTEMAC(chr,m1); \
\
	ctype_r           alpha_r     = PASTEMAC(ch,real)( *alpha ); \
	ctype_r           alpha_i     = PASTEMAC(ch,imag)( *alpha ); \
\
	void*             a_next      = bli_auxinfo_next_a( data ); \
	void*             b_next      = bli_auxinfo_next_b( data ); \
\
	dim_t             i, j; \
\
/*
PASTEMAC(chr,fprintm)( stdout, "gemmtrsm4m_ukr: a1112p_r", m, k+m, \
                       a11_r, 1,      PASTEMAC(chr,packmr), "%4.1f", "" ); \
PASTEMAC(chr,fprintm)( stdout, "gemmtrsm4m_ukr: a1112p_i", m, k+m, \
                       a11_r+is_a, 1, PASTEMAC(chr,packmr), "%4.1f", "" ); \
PASTEMAC(chr,fprintm)( stdout, "gemmtrsm4m_ukr: b1121p_r", k+m, n, \
                       b11_r,      PASTEMAC(chr,packnr), 1, "%4.1f", "" ); \
PASTEMAC(chr,fprintm)( stdout, "gemmtrsm4m_ukr: b1121p_i", k+m, n, \
                       b11_r+is_b, PASTEMAC(chr,packnr), 1, "%4.1f", "" ); \
*/ \
 \
\
	/* Copy the contents of c to a temporary buffer ct. */ \
	if ( !PASTEMAC(chr,eq0)( alpha_i ) ) \
	{ \
		/* We can handle a non-zero imaginary component on alpha, but to do
		   so we have to manually scale b and then use alpha == 1 for the
		   micro-kernel calls. */ \
		for ( i = 0; i < m; ++i ) \
		for ( j = 0; j < n; ++j ) \
		PASTEMAC(ch,scalris)( alpha_r, \
		                      alpha_i, \
		                      *(b11_r + i*rs_b + j*cs_b), \
		                      *(b11_i + i*rs_b + j*cs_b) ); \
\
		/* Use alpha.r == 1.0. */ \
		alpha_r = *one_r; \
	} \
\
\
	/* b11.r = alpha.r * b11.r - ( a12.r * b21.r - a12.i * b21.i );
	   b11.i = alpha.r * b11.r - ( a12.r * b21.i + a12.i * b21.r ); */ \
\
	bli_auxinfo_set_next_ab( a12_r, b21_i, *data ); \
\
	/* b11.r = alpha.r * b11.r - a12.r * b21.r; */ \
	PASTEMAC(chr,gemmukr)( k, \
	                       minus_one_r, \
	                       a12_r, \
	                       b21_r, \
	                       &alpha_r, \
	                       b11_r, rs_b, cs_b, \
	                       data ); \
\
	bli_auxinfo_set_next_ab( a12_i, b21_r, *data ); \
\
	/* b11.i = alpha.r * b11.i - a12.r * b21.i; */ \
	PASTEMAC(chr,gemmukr)( k, \
	                       minus_one_r, \
	                       a12_r, \
	                       b21_i, \
	                       &alpha_r, \
	                       b11_i, rs_b, cs_b, \
	                       data ); \
\
	bli_auxinfo_set_next_ab( a12_i, b21_i, *data ); \
\
	/* b11.i =     1.0 * b11.i - a12.i * b21.r; */ \
	PASTEMAC(chr,gemmukr)( k, \
	                       minus_one_r, \
	                       a12_i, \
	                       b21_r, \
	                       one_r, \
	                       b11_i, rs_b, cs_b, \
	                       data ); \
\
	bli_auxinfo_set_next_ab( a_next, b_next, *data ); \
\
	/* b11.r =     1.0 * b11.r + a12.i * b21.i; */ \
	PASTEMAC(chr,gemmukr)( k, \
	                       one_r, \
	                       a12_i, \
	                       b21_i, \
	                       one_r, \
	                       b11_r, rs_b, cs_b, \
	                       data ); \
\
\
	/* b11 = inv(a11) * b11;
	   c11 = b11; */ \
	PASTEMAC(ch,trsmukr)( a11_r, \
	                      b11_r, \
	                      c11, rs_c, cs_c, \
	                      data ); \
}

INSERT_GENTFUNCCO_BASIC2( gemmtrsm4m_u_ukr_ref, GEMM_UKERNEL, TRSM4M_U_UKERNEL )

