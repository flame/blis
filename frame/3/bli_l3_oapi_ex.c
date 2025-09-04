/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2021, The University of Texas at Austin
   Copyright (C) 2024, Advanced Micro Devices, Inc. All rights reserved.

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

//
// Define object-based interfaces (expert).
//

#undef  GENFRONT
#define GENFRONT( opname ) \
\
void PASTEMAC(opname,BLIS_OAPI_EX_SUF) \
     ( \
       obj_t*  alpha, \
       obj_t*  a, \
       obj_t*  b, \
       obj_t*  beta, \
       obj_t*  c, \
       cntx_t* cntx, \
       rntm_t* rntm  \
     ) \
{ \
	AOCL_DTL_TRACE_ENTRY(AOCL_DTL_LEVEL_TRACE_2) \
	bli_init_once(); \
\
	/* If C has a zero dimension, return early.	*/	\
	if ( bli_obj_has_zero_dim( c ) ) {\
		AOCL_DTL_TRACE_EXIT(AOCL_DTL_LEVEL_TRACE_2) \
		return;									 \
	}\
\
	/* if alpha or A or B has a zero dimension, \
	   scale C by beta and return early. */ \
	if ( bli_obj_equals( alpha, &BLIS_ZERO ) || \
	     bli_obj_has_zero_dim( a ) || \
	     bli_obj_has_zero_dim( b ) ) \
	{\
		bli_scalm( beta, c ); \
		AOCL_DTL_TRACE_EXIT(AOCL_DTL_LEVEL_TRACE_2)	\
		return;\
	}\
\
	/* If the rntm is non-NULL, it may indicate that we should forgo sup
	   handling altogether. */ \
	bool enable_sup = TRUE; \
	if ( rntm != NULL ) enable_sup = bli_rntm_l3_sup( rntm ); \
\
	if ( enable_sup ) \
	{ \
		/* Execute the small/unpacked oapi handler. If it finds that the problem
		   does not fall within the thresholds that define "small", or for some
		   other reason decides not to use the small/unpacked implementation,
		   the function returns with BLIS_FAILURE, which causes execution to
		   proceed towards the conventional implementation. */ \
		err_t result = PASTEMAC(opname,sup)( alpha, a, b, beta, c, cntx, rntm ); \
		if ( result == BLIS_SUCCESS ) \
		{ \
			AOCL_DTL_TRACE_EXIT(AOCL_DTL_LEVEL_TRACE_2)	\
			return; \
		} \
	} \
\
	/* Initialize a local runtime with global settings if necessary. Note
	   that in the case that a runtime is passed in, we make a local copy. */ \
	rntm_t rntm_l; \
	if ( rntm == NULL ) { bli_rntm_init_from_global( &rntm_l ); rntm = &rntm_l; } \
	else                { rntm_l = *rntm;                       rntm = &rntm_l; } \
\
	/* Default to using native execution. */ \
	num_t dt = bli_obj_dt( c ); \
	ind_t im = BLIS_NAT; \
\
	/* If each matrix operand has a complex storage datatype, try to get an
	   induced method (if one is available and enabled). NOTE: Allowing
	   precisions to vary while using 1m, which is what we do here, is unique
	   to gemm; other level-3 operations use 1m only if all storage datatypes
	   are equal (and they ignore the computation precision). */ \
	if ( bli_obj_is_complex( c ) && \
	     bli_obj_is_complex( a ) && \
	     bli_obj_is_complex( b ) ) \
	{ \
		/* Find the highest priority induced method that is both enabled and
		   available for the current operation. (If an induced method is
		   available but not enabled, or simply unavailable, BLIS_NAT will
		   be returned here.) */ \
		im = PASTEMAC(opname,ind_find_avail)( dt ); \
	} \
\
	/* If necessary, obtain a valid context from the gks using the induced
	   method id determined above. */ \
	if ( cntx == NULL ) cntx = bli_gks_query_ind_cntx( im, dt ); \
\
	/* Check the operands. */ \
	if ( bli_error_checking_is_enabled() ) \
		PASTEMAC(opname,_check)( alpha, a, b, beta, c, cntx ); \
\
	/* Invoke the operation's front-end and request the default control tree. */ \
	PASTEMAC(opname,_front)( alpha, a, b, beta, c, cntx, rntm, NULL ); \
\
	AOCL_DTL_TRACE_EXIT(AOCL_DTL_LEVEL_TRACE_2)	\
}

// If a sandbox was enabled, we forgo defining bli_gemm_ex() since it will be
// defined in the sandbox environment.
#ifndef BLIS_ENABLE_SANDBOX
GENFRONT( gemm )
#endif

#undef  GENFRONT
#define GENFRONT( opname ) \
\
void PASTEMAC(opname,BLIS_OAPI_EX_SUF) \
     ( \
       obj_t*  alpha, \
       obj_t*  a, \
       obj_t*  b, \
       obj_t*  beta, \
       obj_t*  c, \
       cntx_t* cntx, \
       rntm_t* rntm  \
     ) \
{ \
	AOCL_DTL_TRACE_ENTRY(AOCL_DTL_LEVEL_TRACE_2) \
	bli_init_once(); \
\
	/* If C has a zero dimension, return early.	*/	\
	if ( bli_obj_has_zero_dim( c ) ) {\
		AOCL_DTL_TRACE_EXIT(AOCL_DTL_LEVEL_TRACE_2) \
		return;									 \
	}\
\
	/* if alpha or A or B has a zero dimension, \
	   scale C by beta and return early. */ \
	if ( bli_obj_equals( alpha, &BLIS_ZERO ) || \
	     bli_obj_has_zero_dim( a ) || \
	     bli_obj_has_zero_dim( b ) ) \
	{\
		bli_scalm( beta, c ); \
		AOCL_DTL_TRACE_EXIT(AOCL_DTL_LEVEL_TRACE_2)	\
		return;\
	}\
\
	/* If the rntm is non-NULL, it may indicate that we should forgo sup
	   handling altogether. */ \
	bool enable_sup = TRUE; \
	if ( rntm != NULL ) enable_sup = bli_rntm_l3_sup( rntm ); \
\
	if ( enable_sup ) \
	{ \
		/* Execute the small/unpacked oapi handler. If it finds that the problem
		   does not fall within the thresholds that define "small", or for some
		   other reason decides not to use the small/unpacked implementation,
		   the function returns with BLIS_FAILURE, which causes execution to
		   proceed towards the conventional implementation. */ \
		err_t result = PASTEMAC(opname,sup)( alpha, a, b, beta, c, cntx, rntm ); \
		if ( result == BLIS_SUCCESS ) \
		{\
			AOCL_DTL_TRACE_EXIT(AOCL_DTL_LEVEL_TRACE_2) \
			return; \
		} \
	} \
\
	/* Initialize a local runtime with global settings if necessary. Note
	   that in the case that a runtime is passed in, we make a local copy. */ \
	rntm_t rntm_l; \
	if ( rntm == NULL ) { bli_rntm_init_from_global( &rntm_l ); rntm = &rntm_l; } \
	else                { rntm_l = *rntm;                       rntm = &rntm_l; } \
\
	/* Default to using native execution. */ \
	num_t dt = bli_obj_dt( c ); \
	ind_t im = BLIS_NAT; \
\
	/* If all matrix operands are complex and of the same storage datatype, try
	   to get an induced method (if one is available and enabled). */ \
	if ( bli_obj_dt( a ) == bli_obj_dt( c ) && \
	     bli_obj_dt( b ) == bli_obj_dt( c ) && \
	     bli_obj_is_complex( c ) ) \
	{ \
		/* Find the highest priority induced method that is both enabled and
		   available for the current operation. (If an induced method is
		   available but not enabled, or simply unavailable, BLIS_NAT will
		   be returned here.) */ \
		im = PASTEMAC(opname,ind_find_avail)( dt ); \
	} \
\
	/* If necessary, obtain a valid context from the gks using the induced
	   method id determined above. */ \
	if ( cntx == NULL ) cntx = bli_gks_query_ind_cntx( im, dt ); \
\
	/* Check the operands. */ \
	if ( bli_error_checking_is_enabled() ) \
		PASTEMAC(opname,_check)( alpha, a, b, beta, c, cntx ); \
\
	/* Invoke the operation's front-end and request the default control tree. */ \
	PASTEMAC(opname,_front)( alpha, a, b, beta, c, cntx, rntm, NULL ); \
\
	AOCL_DTL_TRACE_EXIT(AOCL_DTL_LEVEL_TRACE_2)	\
}

GENFRONT( gemmt )

#undef  GENFRONT
#define GENFRONT( opname ) \
\
void PASTEMAC(opname,BLIS_OAPI_EX_SUF) \
     ( \
       obj_t*  alpha, \
       obj_t*  a, \
       obj_t*  b, \
       obj_t*  beta, \
       obj_t*  c, \
       cntx_t* cntx, \
       rntm_t* rntm  \
     ) \
{ \
	AOCL_DTL_TRACE_ENTRY(AOCL_DTL_LEVEL_TRACE_2) \
	bli_init_once(); \
\
	/* Initialize a local runtime with global settings if necessary. Note
	   that in the case that a runtime is passed in, we make a local copy. */ \
	rntm_t rntm_l; \
	if ( rntm == NULL ) { bli_rntm_init_from_global( &rntm_l ); rntm = &rntm_l; } \
	else                { rntm_l = *rntm;                       rntm = &rntm_l; } \
\
	/* Default to using native execution. */ \
	num_t dt = bli_obj_dt( c ); \
	ind_t im = BLIS_NAT; \
\
	/* If all matrix operands are complex and of the same storage datatype, try
	   to get an induced method (if one is available and enabled). */ \
	if ( bli_obj_dt( a ) == bli_obj_dt( c ) && \
	     bli_obj_dt( b ) == bli_obj_dt( c ) && \
	     bli_obj_is_complex( c ) ) \
	{ \
		/* Find the highest priority induced method that is both enabled and
		   available for the current operation. (If an induced method is
		   available but not enabled, or simply unavailable, BLIS_NAT will
		   be returned here.) */ \
		im = PASTEMAC(opname,ind_find_avail)( dt ); \
	} \
\
	/* If necessary, obtain a valid context from the gks using the induced
	   method id determined above. */ \
	if ( cntx == NULL ) cntx = bli_gks_query_ind_cntx( im, dt ); \
\
	/* Check the operands. */ \
	if ( bli_error_checking_is_enabled() ) \
		PASTEMAC(opname,_check)( alpha, a, b, beta, c, cntx ); \
\
	/* Invoke the operation's front-end and request the default control tree. */ \
	PASTEMAC(opname,_front)( alpha, a, b, beta, c, cntx, rntm, NULL ); \
\
	AOCL_DTL_TRACE_EXIT(AOCL_DTL_LEVEL_TRACE_2)	\
}

GENFRONT( her2k )
GENFRONT( syr2k )


#undef  GENFRONT
#define GENFRONT( opname ) \
\
void PASTEMAC(opname,BLIS_OAPI_EX_SUF) \
     ( \
       side_t  side, \
       obj_t*  alpha, \
       obj_t*  a, \
       obj_t*  b, \
       obj_t*  beta, \
       obj_t*  c, \
       cntx_t* cntx, \
       rntm_t* rntm  \
     ) \
{ \
	AOCL_DTL_TRACE_ENTRY(AOCL_DTL_LEVEL_TRACE_2) \
	bli_init_once(); \
\
	/* Initialize a local runtime with global settings if necessary. Note
	   that in the case that a runtime is passed in, we make a local copy. */ \
	rntm_t rntm_l; \
	if ( rntm == NULL ) { bli_rntm_init_from_global( &rntm_l ); rntm = &rntm_l; } \
	else                { rntm_l = *rntm;                       rntm = &rntm_l; } \
\
	/* Default to using native execution. */ \
	num_t dt = bli_obj_dt( c ); \
	ind_t im = BLIS_NAT; \
\
	/* If all matrix operands are complex and of the same storage datatype, try
	   to get an induced method (if one is available and enabled). */ \
	if ( bli_obj_dt( a ) == bli_obj_dt( c ) && \
	     bli_obj_dt( b ) == bli_obj_dt( c ) && \
	     bli_obj_is_complex( c ) ) \
	{ \
		/* Find the highest priority induced method that is both enabled and
		   available for the current operation. (If an induced method is
		   available but not enabled, or simply unavailable, BLIS_NAT will
		   be returned here.) */ \
		im = PASTEMAC(opname,ind_find_avail)( dt ); \
	} \
\
	/* If necessary, obtain a valid context from the gks using the induced
	   method id determined above. */ \
	if ( cntx == NULL ) cntx = bli_gks_query_ind_cntx( im, dt ); \
\
	/* Check the operands. */ \
	if ( bli_error_checking_is_enabled() ) \
		PASTEMAC(opname,_check)( side, alpha, a, b, beta, c, cntx ); \
\
	/* Invoke the operation's front-end and request the default control tree. */ \
	PASTEMAC(opname,_front)( side, alpha, a, b, beta, c, cntx, rntm, NULL ); \
\
	AOCL_DTL_TRACE_EXIT(AOCL_DTL_LEVEL_TRACE_2)	\
}

GENFRONT( hemm )
GENFRONT( symm )
GENFRONT( trmm3 )


#undef  GENFRONT
#define GENFRONT( opname ) \
\
void PASTEMAC(opname,BLIS_OAPI_EX_SUF) \
     ( \
       obj_t*  alpha, \
       obj_t*  a, \
       obj_t*  beta, \
       obj_t*  c, \
       cntx_t* cntx, \
       rntm_t* rntm  \
     ) \
{ \
	AOCL_DTL_TRACE_ENTRY(AOCL_DTL_LEVEL_TRACE_2) \
	bli_init_once(); \
\
	/* If C has a zero dimension, return early. */ \
	if ( bli_obj_has_zero_dim( c ) ) {\
		AOCL_DTL_TRACE_EXIT(AOCL_DTL_LEVEL_TRACE_2) \
		return; \
	} \
\
	/* If alpha or A or B has a zero dimension, \
	   scale C by beta and return early. */ \
\
	if( bli_obj_equals( alpha, &BLIS_ZERO ) || \
	    bli_obj_has_zero_dim( a ) ) \
	{ \
		bli_scalm( beta, c ); \
		AOCL_DTL_TRACE_EXIT(AOCL_DTL_LEVEL_TRACE_2) \
		return; \
	} \
\
	/* If the rntm is non-NULL, it may indicate that we should forgo SUP handling altogether. */ \
	bool enable_sup = TRUE; \
	if( rntm != NULL ) enable_sup = bli_rntm_l3_sup( rntm ); \
\
	if( enable_sup ) \
	{ \
		/* Execute the small/unpacked oapi handler.
		   If it finds that the problem does not fall within the
		   thresholds that define "small", or for some other reason
		   decides not to use the small/unpacked implementation,
		   the function returns with BLIS_FAILURE, which causes excution
		   to proceed forward towards conventional implementation, */ \
\
		err_t result = PASTEMAC(opname, sup) ( alpha, a, beta, c, cntx, rntm ); \
		if( result == BLIS_SUCCESS ) { \
			AOCL_DTL_TRACE_EXIT(AOCL_DTL_LEVEL_TRACE_2) \
			return; \
		} \
	} \
\
	/* Initialize a local runtime with global settings if necessary. Note
	   that in the case that a runtime is passed in, we make a local copy. */ \
	rntm_t rntm_l; \
	if ( rntm == NULL ) { bli_rntm_init_from_global( &rntm_l ); rntm = &rntm_l; } \
	else                { rntm_l = *rntm;                       rntm = &rntm_l; } \
\
	/* Default to using native execution. */ \
	num_t dt = bli_obj_dt( c ); \
	ind_t im = BLIS_NAT; \
\
	/* If all matrix operands are complex and of the same storage datatype, try
	   to get an induced method (if one is available and enabled). */ \
	if ( bli_obj_dt( a ) == bli_obj_dt( c ) && \
	     bli_obj_is_complex( c ) ) \
	{ \
		/* Find the highest priority induced method that is both enabled and
		   available for the current operation. (If an induced method is
		   available but not enabled, or simply unavailable, BLIS_NAT will
		   be returned here.) */ \
		im = PASTEMAC(opname,ind_find_avail)( dt ); \
	} \
\
	/* If necessary, obtain a valid context from the gks using the induced
	   method id determined above. */ \
	if ( cntx == NULL ) cntx = bli_gks_query_ind_cntx( im, dt ); \
\
	/* Check the operands. */ \
	if ( bli_error_checking_is_enabled() ) \
		PASTEMAC(opname,_check)( alpha, a, beta, c, cntx ); \
\
	/* Invoke the operation's front-end and request the default control tree. */ \
	PASTEMAC(opname,_front)( alpha, a, beta, c, cntx, rntm, NULL ); \
\
	AOCL_DTL_TRACE_EXIT(AOCL_DTL_LEVEL_TRACE_2)	\
}

GENFRONT( syrk )

#undef  GENFRONT
#define GENFRONT( opname ) \
\
void PASTEMAC(opname,BLIS_OAPI_EX_SUF) \
     ( \
       obj_t*  alpha, \
       obj_t*  a, \
       obj_t*  beta, \
       obj_t*  c, \
       cntx_t* cntx, \
       rntm_t* rntm  \
     ) \
{ \
	AOCL_DTL_TRACE_ENTRY(AOCL_DTL_LEVEL_TRACE_2) \
	bli_init_once(); \
\
	/* Initialize a local runtime with global settings if necessary. Note
	   that in the case that a runtime is passed in, we make a local copy. */ \
	rntm_t rntm_l; \
	if ( rntm == NULL ) { bli_rntm_init_from_global( &rntm_l ); rntm = &rntm_l; } \
	else                { rntm_l = *rntm;                       rntm = &rntm_l; } \
\
	/* Default to using native execution. */ \
	num_t dt = bli_obj_dt( c ); \
	ind_t im = BLIS_NAT; \
\
	/* If all matrix operands are complex and of the same storage datatype, try
	   to get an induced method (if one is available and enabled). */ \
	if ( bli_obj_dt( a ) == bli_obj_dt( c ) && \
	     bli_obj_is_complex( c ) ) \
	{ \
		/* Find the highest priority induced method that is both enabled and
		   available for the current operation. (If an induced method is
		   available but not enabled, or simply unavailable, BLIS_NAT will
		   be returned here.) */ \
		im = PASTEMAC(opname,ind_find_avail)( dt ); \
	} \
\
	/* If necessary, obtain a valid context from the gks using the induced
	   method id determined above. */ \
	if ( cntx == NULL ) cntx = bli_gks_query_ind_cntx( im, dt ); \
\
	/* Check the operands. */ \
	if ( bli_error_checking_is_enabled() ) \
		PASTEMAC(opname,_check)( alpha, a, beta, c, cntx ); \
\
	/* Invoke the operation's front-end and request the default control tree. */ \
	PASTEMAC(opname,_front)( alpha, a, beta, c, cntx, rntm, NULL ); \
\
	AOCL_DTL_TRACE_EXIT(AOCL_DTL_LEVEL_TRACE_2)	\
}

GENFRONT( herk )


#undef  GENFRONT
#define GENFRONT( opname ) \
\
void PASTEMAC(opname,BLIS_OAPI_EX_SUF) \
     ( \
       side_t  side, \
       obj_t*  alpha, \
       obj_t*  a, \
       obj_t*  b, \
       cntx_t* cntx, \
       rntm_t* rntm  \
     ) \
{ \
	AOCL_DTL_TRACE_ENTRY(AOCL_DTL_LEVEL_TRACE_2) \
	bli_init_once(); \
\
	/* Initialize a local runtime with global settings if necessary. Note
	   that in the case that a runtime is passed in, we make a local copy. */ \
	rntm_t rntm_l; \
	if ( rntm == NULL ) { bli_rntm_init_from_global( &rntm_l ); rntm = &rntm_l; } \
	else                { rntm_l = *rntm;                       rntm = &rntm_l; } \
\
	/* Default to using native execution. */ \
	num_t dt = bli_obj_dt( b ); \
	ind_t im = BLIS_NAT; \
\
	/* If all matrix operands are complex and of the same storage datatype, try
	   to get an induced method (if one is available and enabled). */ \
	if ( bli_obj_dt( a ) == bli_obj_dt( b ) && \
	     bli_obj_is_complex( b ) ) \
	{ \
		/* Find the highest priority induced method that is both enabled and
		   available for the current operation. (If an induced method is
		   available but not enabled, or simply unavailable, BLIS_NAT will
		   be returned here.) */ \
		im = PASTEMAC(opname,ind_find_avail)( dt ); \
	} \
\
	/* If necessary, obtain a valid context from the gks using the induced
	   method id determined above. */ \
	if ( cntx == NULL ) cntx = bli_gks_query_ind_cntx( im, dt ); \
\
	/* Check the operands. */ \
	if ( bli_error_checking_is_enabled() ) \
		PASTEMAC(opname,_check)( side, alpha, a, b, cntx ); \
\
	/* Invoke the operation's front-end and request the default control tree. */ \
	PASTEMAC(opname,_front)( side, alpha, a, b, cntx, rntm, NULL ); \
\
	AOCL_DTL_TRACE_EXIT(AOCL_DTL_LEVEL_TRACE_2)	\
}

GENFRONT( trmm )
GENFRONT( trsm )

