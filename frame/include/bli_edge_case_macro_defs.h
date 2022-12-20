/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2021, The University of Texas at Austin

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

#ifndef BLIS_EDGE_CASE_MACRO_DEFS_H
#define BLIS_EDGE_CASE_MACRO_DEFS_H


// Helper macros for edge-case handling within gemm microkernels.

#define GEMM_UKR_SETUP_CT_PRE(ch,mr,nr,row_major) \
\
	PASTEMAC(ch,ctype)* restrict _beta   = beta; \
	PASTEMAC(ch,ctype)* restrict _c      = c; \
	const inc_t                  _rs_c   = rs_c; \
	const inc_t                  _cs_c   = cs_c; \
	PASTEMAC(ch,ctype)           _ct[ BLIS_STACK_BUF_MAX_SIZE / sizeof( PASTEMAC(ch,ctype) ) ] \
	                                  __attribute__((aligned(BLIS_STACK_BUF_ALIGN_SIZE))); \
	const inc_t                  _rs_ct  = row_major ? nr :  1; \
	const inc_t                  _cs_ct  = row_major ?  1 : mr;

#define GEMM_UKR_SETUP_CT_POST(ch) \
\
	PASTEMAC(ch,ctype) _zero; \
	PASTEMAC(ch,set0s)( _zero ); \
	\
	if ( _use_ct ) \
	{ \
		c = _ct; \
		rs_c = _rs_ct; \
		cs_c = _cs_ct; \
		beta = &_zero; \
	}

#define GEMM_UKR_SETUP_CT(ch,mr,nr,row_major) \
\
	GEMM_UKR_SETUP_CT_PRE(ch,mr,nr,row_major); \
	const bool _use_ct = ( row_major ? cs_c != 1 : rs_c != 1 ) || \
	                     m != mr || n != nr; \
	GEMM_UKR_SETUP_CT_POST(ch);

#define GEMM_UKR_SETUP_CT_AMBI(ch,mr,nr,row_major) \
\
	GEMM_UKR_SETUP_CT_PRE(ch,mr,nr,row_major); \
	const bool _use_ct = ( cs_c != 1 && rs_c != 1 ) || \
	                     m != mr || n != nr; \
	GEMM_UKR_SETUP_CT_POST(ch);

#define GEMM_UKR_SETUP_CT_ANY(ch,mr,nr,row_major) \
\
	GEMM_UKR_SETUP_CT_PRE(ch,mr,nr,row_major); \
	const bool _use_ct = m != mr || n != nr; \
	GEMM_UKR_SETUP_CT_POST(ch);

#define GEMM_UKR_SETUP_CT_ALIGNED(ch,mr,nr,row_major,alignment) \
\
	GEMM_UKR_SETUP_CT_PRE(ch,mr,nr,row_major); \
	const bool _use_ct = ( row_major ? cs_c != 1 : rs_c != 1 ) || \
	                     m != mr || n != nr || \
	                     ( (uintptr_t)_c % alignment ) || \
	                     ( ( ( row_major ? _rs_c : _cs_c )*sizeof( PASTEMAC(ch,ctype) ) ) % alignment ); \
	GEMM_UKR_SETUP_CT_POST(ch);

#define GEMM_UKR_FLUSH_CT(ch) \
\
	if ( _use_ct ) \
	{ \
		PASTEMAC(ch,xpbys_mxn) \
		( \
		  m, n, \
		  _ct, _rs_ct, _cs_ct, \
		  _beta, \
		  _c,  _rs_c,  _cs_c \
		); \
	} \



// Debug versions of the above macros in which auxinfo_t.ct is used
// instead of a local ct.

#define GEMM_UKR_SETUP_AUXCT_PRE(ch,mr,nr,row_major) \
\
	PASTEMAC(ch,ctype)* restrict _beta   = beta; \
	PASTEMAC(ch,ctype)* restrict _c      = c; \
	const inc_t                  _rs_c   = rs_c; \
	const inc_t                  _cs_c   = cs_c; \
	PASTEMAC(ch,ctype)* restrict _ct     = bli_auxinfo_ct( data ); \
	const inc_t                  _rs_ct  = row_major ? nr :  1; \
	const inc_t                  _cs_ct  = row_major ?  1 : mr;

#define GEMM_UKR_SETUP_AUXCT_POST(ch) \
\
	PASTEMAC(ch,ctype) _zero; \
	PASTEMAC(ch,set0s)( _zero ); \
	\
	if ( _use_ct ) \
	{ \
		c = _ct; \
		rs_c = _rs_ct; \
		cs_c = _cs_ct; \
		beta = &_zero; \
	}

#define GEMM_UKR_SETUP_AUXCT(ch,mr,nr,row_major) \
\
	GEMM_UKR_SETUP_AUXCT_PRE(ch,mr,nr,row_major); \
	const bool _use_ct = ( row_major ? cs_c != 1 : rs_c != 1 ) || \
	                     m != mr || n != nr; \
	GEMM_UKR_SETUP_AUXCT_POST(ch);

#define GEMM_UKR_SETUP_AUXCT_AMBI(ch,mr,nr,row_major) \
\
	GEMM_UKR_SETUP_AUXCT_PRE(ch,mr,nr,row_major); \
	const bool _use_ct = ( cs_c != 1 && rs_c != 1 ) || \
	                     m != mr || n != nr; \
	GEMM_UKR_SETUP_AUXCT_POST(ch);

#define GEMM_UKR_SETUP_AUXCT_ANY(ch,mr,nr,row_major) \
\
	GEMM_UKR_SETUP_AUXCT_PRE(ch,mr,nr,row_major); \
	const bool _use_ct = m != mr || n != nr; \
	GEMM_UKR_SETUP_AUXCT_POST(ch);

#define GEMM_UKR_SETUP_AUXCT_ALIGNED(ch,mr,nr,row_major,alignment) \
\
	GEMM_UKR_SETUP_AUXCT_PRE(ch,mr,nr,row_major); \
	const bool _use_ct = ( row_major ? cs_c != 1 : rs_c != 1 ) || \
	                     m != mr || n != nr || \
	                     ( (uintptr_t)_c % alignment ) || \
	                     ( ( ( row_major ? _rs_c : _cs_c )*sizeof( PASTEMAC(ch,ctype) ) ) % alignment ); \
	GEMM_UKR_SETUP_AUXCT_POST(ch);

#define GEMM_UKR_FLUSH_AUXCT(ch) \
\
	if ( _use_ct ) \
	{ \
		PASTEMAC(ch,xpbys_mxn) \
		( \
		  m, n, \
		  _ct, _rs_ct, _cs_ct, \
		  _beta, \
		  _c,  _rs_c,  _cs_c \
		); \
	} \


#endif

