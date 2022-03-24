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
#define GENTFUNC( ctype, ch, opname, arch, suf, trsmkerid ) \
\
void PASTEMAC3(ch,opname,arch,suf) \
     ( \
       dim_t               m, \
       dim_t               n, \
       dim_t               k, \
       ctype*     restrict alpha, \
       ctype*     restrict a1x, \
       ctype*     restrict a11, \
       ctype*     restrict bx1, \
       ctype*     restrict b11, \
       ctype*     restrict c11, inc_t rs_c, inc_t cs_c, \
       auxinfo_t* restrict data, \
       cntx_t*    restrict cntx  \
     ) \
{ \
	const num_t     dt     = PASTEMAC(ch,type); \
\
	const dim_t     mr     = bli_cntx_get_blksz_def_dt( dt, BLIS_MR, cntx ); \
	const dim_t     nr     = bli_cntx_get_blksz_def_dt( dt, BLIS_NR, cntx ); \
\
	const inc_t     packnr = bli_cntx_get_blksz_max_dt( dt, BLIS_NR, cntx ); \
\
	const inc_t     rs_b   = packnr; \
	const inc_t     cs_b   = 1; \
\
	ctype*          minus_one = PASTEMAC(ch,m1); \
\
	PASTECH(ch,gemm_ukr_ft) \
	              gemm_ukr = bli_cntx_get_l3_nat_ukr_dt( dt, BLIS_GEMM_UKR, cntx ); \
	PASTECH(ch,trsm_ukr_ft) \
	              trsm_ukr = bli_cntx_get_l3_nat_ukr_dt( dt, trsmkerid, cntx ); \
\
	ctype           ct[ BLIS_STACK_BUF_MAX_SIZE \
	                    / sizeof( ctype ) ] \
	                    __attribute__((aligned(BLIS_STACK_BUF_ALIGN_SIZE))); \
	/* FGVZ: Should we be querying the preference of BLIS_GEMMTRSM_?_UKR
	   instead? */ \
	const bool      col_pref    = bli_cntx_l3_vir_ukr_prefers_cols_dt( dt, BLIS_GEMM_UKR, cntx ); \
	const inc_t     rs_ct       = ( col_pref ? 1 : nr ); \
	const inc_t     cs_ct       = ( col_pref ? mr : 1 ); \
\
	const bool      use_ct      = ( m < mr || n < nr ); \
\
	ctype* restrict c11_use     = c11; \
	inc_t           rs_c_use    = rs_c; \
	inc_t           cs_c_use    = cs_c; \
\
	if ( use_ct ) \
	{ \
		c11_use  = ct; \
		rs_c_use = rs_ct; \
		cs_c_use = cs_ct; \
	} \
\
	/* lower: b11 = alpha * b11 - a10 * b01; */ \
	/* upper: b11 = alpha * b11 - a12 * b21; */ \
	gemm_ukr \
	( \
	  m, \
	  n, \
	  k, \
	  minus_one, \
	  a1x, \
	  bx1, \
	  alpha, \
	  b11, rs_b, cs_b, \
	  data, \
	  cntx  \
	); \
\
	/* b11 = inv(a11) * b11;
	   c11 = b11; */ \
	trsm_ukr \
	( \
	  a11, \
	  b11, \
	  c11_use, rs_c_use, cs_c_use, \
	  data, \
	  cntx  \
	); \
\
	if ( use_ct ) \
	{ \
		PASTEMAC(ch,copys_mxn) \
		( \
		  m, n, \
		  ct,  rs_ct, cs_ct, \
		  c11, rs_c,  cs_c  \
		); \
	} \
\
/*
PASTEMAC(d,fprintm)( stdout, "gemmtrsm_ukr: b0111p_r after", k+3, 8, \
                     ( double* )b01,     2*PASTEMAC(ch,packnr), 2, "%4.1f", "" ); \
PASTEMAC(d,fprintm)( stdout, "gemmtrsm_ukr: b0111p_i after", k+3, 8, \
                     ( double* )b01 + 1, 2*PASTEMAC(ch,packnr), 2, "%4.1f", "" ); \
*/ \
}

INSERT_GENTFUNC_BASIC3( gemmtrsm_l, BLIS_CNAME_INFIX, BLIS_REF_SUFFIX, BLIS_TRSM_L_UKR )
INSERT_GENTFUNC_BASIC3( gemmtrsm_u, BLIS_CNAME_INFIX, BLIS_REF_SUFFIX, BLIS_TRSM_U_UKR )

