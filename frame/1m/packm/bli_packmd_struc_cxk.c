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

#undef  GENTFUNC2RO
#define GENTFUNC2RO( ctypec_r, ctype_c, ctypep_r, ctypep, chc_r, chc, chp_r, chp, varname ) \
GENTFUNC2RO_( ctypec_r, ctypec_r, ctypep_r, ctypep_r, chc_r, chc_r, chp_r, chp_r, varname ) \
GENTFUNC2RO_( ctypec_r, ctypec,   ctypep_r, ctypep,   chc_r, chc,   chp_r, chp,   varname )

#undef  GENTFUNC2RO_
#define GENTFUNC2RO_( ctypec_r, ctype_c, ctypep_r, ctypep, chc_r, chc, chp_r, chp, varname ) \
\
void PASTEMAC(chc,chp,varname) \
     ( \
             struc_t strucc, \
             diag_t  diagc, \
             uplo_t  uploc, \
             conj_t  conjc, \
             pack_t  schema, \
             bool    invdiag, \
             dim_t   panel_dim, \
             dim_t   panel_len, \
             dim_t   panel_dim_max, \
             dim_t   panel_len_max, \
             dim_t   panel_dim_off, \
             dim_t   panel_len_off, \
             dim_t   panel_bcast, \
       const void*   kappa, \
       const void*   c, inc_t incc, inc_t ldc, \
             void*   p,             inc_t ldp, \
       const void*   params_, \
       const cntx_t* cntx \
     ) \
{ \
	num_t dt_c          = PASTEMAC(chc,type); \
	num_t dt_p          = PASTEMAC(chp,type); \
	dim_t dt_c_size     = bli_dt_size( dt_c ); \
\
	ukr_t cxk_ker_id    = BLIS_PACKMD_KER; \
\
	packmd_cxk_ker_ft f_cxk = bli_cntx_get_ukr2_dt( dt_c, dt_p, cxk_ker_id, cntx ); \
\
	const gemmd_params* params = ( const gemmd_params* )params_; \
\
	      inc_t incd = params->incd; \
	const char* d    = ( const char* )params->d + panel_len_off*incd*dt_c_size; \
\
	/* For general matrices, pack and return early */ \
	if ( bli_is_general( strucc ) ) \
	{ \
		f_cxk \
		( \
		  conjc, \
		  schema, \
		  panel_dim, \
		  panel_dim_max, \
		  panel_bcast, \
		  panel_len, \
		  panel_len_max, \
		  kappa, \
		  c, incc, ldc, \
		  d, incd, \
		  p,       ldp, \
		  params, \
		  cntx \
		); \
		return; \
	} \
\
	bli_check_error_code( BLIS_NOT_YET_IMPLEMENTED ); \
}

INSERT_GENTFUNC2RO( packmd_struc_cxk )
INSERT_GENTFUNC2RO_MIX_P( packmd_struc_cxk )

