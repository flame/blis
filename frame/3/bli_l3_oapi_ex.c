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

#include "blis.h"

//
// Define object-based interfaces (expert).
//

// If a sandbox was enabled, we forgo defining bli_gemm_ex() since it will be
// defined in the sandbox environment.
#ifndef BLIS_ENABLE_SANDBOX

void PASTEMAC(gemm,BLIS_OAPI_EX_SUF)
     (
       const obj_t*  alpha,
       const obj_t*  a,
       const obj_t*  b,
       const obj_t*  beta,
       const obj_t*  c,
       const cntx_t* cntx,
       const rntm_t* rntm
     )
{
	bli_init_once();

	// If C has a zero dimension, return early.
	if ( bli_obj_has_zero_dim( c ) ) return;

	// If alpha is zero, or if A or B has a zero dimension, scale C by beta
	// and return early.
	if ( bli_obj_equals( alpha, &BLIS_ZERO ) ||
	     bli_obj_has_zero_dim( a ) ||
	     bli_obj_has_zero_dim( b ) )
	{
		bli_scalm( beta, c );
		return;
	}

	// If the rntm is non-NULL, it may indicate that we should forgo sup
	// handling altogether.
	bool enable_sup = TRUE;
	if ( rntm != NULL ) enable_sup = bli_rntm_l3_sup( rntm );

	if ( enable_sup )
	{
		// Execute the small/unpacked oapi handler. If it finds that the problem
		// does not fall within the thresholds that define "small", or for some
		// other reason decides not to use the small/unpacked implementation,
		// the function returns with BLIS_FAILURE, which causes execution to
		// proceed towards the conventional implementation.
		err_t result = bli_gemmsup( alpha, a, b, beta, c, cntx, rntm );
		if ( result == BLIS_SUCCESS )
		{
			return;
		}
	}

	// Initialize a local runtime with global settings if necessary. Note
	// that in the case that a runtime is passed in, we make a local copy.
	rntm_t rntm_l;
	if ( rntm == NULL ) { bli_rntm_init_from_global( &rntm_l ); }
	else                { rntm_l = *rntm;                       }

	// Default to using native execution.
	num_t dt = bli_obj_dt( c );
	ind_t im = BLIS_NAT;

	// If each matrix operand has a complex storage datatype, try to get an
	// induced method (if one is available and enabled). NOTE: Allowing
	// precisions to vary while using 1m, which is what we do here, is unique
	// to gemm; other level-3 operations use 1m only if all storage datatypes
	// are equal (and they ignore the computation precision).
	if ( bli_obj_is_complex( c ) &&
	     bli_obj_is_complex( a ) &&
	     bli_obj_is_complex( b ) )
	{
		// Find the highest priority induced method that is both enabled and
		// available for the current operation. (If an induced method is
		// available but not enabled, or simply unavailable, BLIS_NAT will
		// be returned here.)
		im = bli_gemmind_find_avail( dt );
	}

	// If necessary, obtain a valid context from the gks using the induced
	// method id determined above.
	if ( cntx == NULL ) cntx = bli_gks_query_ind_cntx( im );

	// Check the operands.
	if ( bli_error_checking_is_enabled() )
		bli_gemm_check( alpha, a, b, beta, c, cntx );

	// Invoke the operation's front-end and request the default control tree.
	bli_gemm_front( alpha, a, b, beta, c, cntx, &rntm_l );
}

#endif


void PASTEMAC(gemmt,BLIS_OAPI_EX_SUF)
     (
       const obj_t*  alpha,
       const obj_t*  a,
       const obj_t*  b,
       const obj_t*  beta,
       const obj_t*  c,
       const cntx_t* cntx,
       const rntm_t* rntm
     )
{
	bli_init_once();

	// If C has a zero dimension, return early.
	if ( bli_obj_has_zero_dim( c ) ) return;

	// If alpha is zero, or if A or B has a zero dimension, scale C by beta
	// and return early.
	if ( bli_obj_equals( alpha, &BLIS_ZERO ) ||
	     bli_obj_has_zero_dim( a ) ||
	     bli_obj_has_zero_dim( b ) )
	{
		bli_scalm( beta, c );
		return;
	}

	// Initialize a local runtime with global settings if necessary. Note
	// that in the case that a runtime is passed in, we make a local copy.
	rntm_t rntm_l;
	if ( rntm == NULL ) { bli_rntm_init_from_global( &rntm_l ); }
	else                { rntm_l = *rntm;                       }

	// Default to using native execution.
	num_t dt = bli_obj_dt( c );
	ind_t im = BLIS_NAT;

	// If all matrix operands are complex and of the same storage datatype, try
	// to get an induced method (if one is available and enabled).
	if ( bli_obj_dt( a ) == bli_obj_dt( c ) &&
	     bli_obj_dt( b ) == bli_obj_dt( c ) &&
	     bli_obj_is_complex( c ) )
	{
		// Find the highest priority induced method that is both enabled and
		// available for the current operation. (If an induced method is
		// available but not enabled, or simply unavailable, BLIS_NAT will
		// be returned here.)
		im = bli_gemmtind_find_avail( dt );
	}

	// If necessary, obtain a valid context from the gks using the induced
	// method id determined above.
	if ( cntx == NULL ) cntx = bli_gks_query_ind_cntx( im );

	// Check the operands.
	if ( bli_error_checking_is_enabled() )
		bli_gemmt_check( alpha, a, b, beta, c, cntx );

	// Invoke the operation's front-end and request the default control tree.
	bli_gemmt_front( alpha, a, b, beta, c, cntx, &rntm_l );
}


void PASTEMAC(her2k,BLIS_OAPI_EX_SUF)
     (
       const obj_t*  alpha,
       const obj_t*  a,
       const obj_t*  b,
       const obj_t*  beta,
       const obj_t*  c,
       const cntx_t* cntx,
       const rntm_t* rntm
     )
{
	bli_init_once();

	obj_t ah;
	obj_t bh;
	obj_t alphah;

	// Check parameters.
	if ( bli_error_checking_is_enabled() )
		bli_her2k_check( alpha, a, b, beta, c, cntx );

	bli_obj_alias_to( alpha, &alphah );
	bli_obj_toggle_conj( &alphah );

	bli_obj_alias_to( a, &ah );
	bli_obj_toggle_trans( &ah );
	bli_obj_toggle_conj( &ah );

	bli_obj_alias_to( b, &bh );
	bli_obj_toggle_trans( &bh );
	bli_obj_toggle_conj( &bh );

	// Invoke gemmt twice, using beta only the first time.
	PASTEMAC(gemmt,BLIS_OAPI_EX_SUF)(   alpha, a, &bh,      beta, c, cntx, rntm );
	PASTEMAC(gemmt,BLIS_OAPI_EX_SUF)( &alphah, b, &ah, &BLIS_ONE, c, cntx, rntm );

	// The Hermitian rank-2k product was computed as alpha*A*B'+alpha'*B*A', even for
	// the diagonal elements. Mathematically, the imaginary components of
	// diagonal elements of a Hermitian rank-2k product should always be
	// zero. However, in practice, they sometimes accumulate meaningless
	// non-zero values. To prevent this, we explicitly set those values
	// to zero before returning.
	bli_setid( &BLIS_ZERO, c );
}


void PASTEMAC(syr2k,BLIS_OAPI_EX_SUF)
     (
       const obj_t*  alpha,
       const obj_t*  a,
       const obj_t*  b,
       const obj_t*  beta,
       const obj_t*  c,
       const cntx_t* cntx,
       const rntm_t* rntm
     )
{
	bli_init_once();

	obj_t at;
	obj_t bt;

	// Check parameters.
	if ( bli_error_checking_is_enabled() )
		bli_syr2k_check( alpha, a, b, beta, c, cntx );

	bli_obj_alias_to( b, &bt );
	bli_obj_toggle_trans( &bt );

	bli_obj_alias_to( a, &at );
	bli_obj_toggle_trans( &at );

	// Invoke gemmt twice, using beta only the first time.
	PASTEMAC(gemmt,BLIS_OAPI_EX_SUF)( alpha, a, &bt,      beta, c, cntx, rntm );
	PASTEMAC(gemmt,BLIS_OAPI_EX_SUF)( alpha, b, &at, &BLIS_ONE, c, cntx, rntm );
}


void PASTEMAC(hemm,BLIS_OAPI_EX_SUF)
     (
             side_t  side,
       const obj_t*  alpha,
       const obj_t*  a,
       const obj_t*  b,
       const obj_t*  beta,
       const obj_t*  c,
       const cntx_t* cntx,
       const rntm_t* rntm
     )
{
	bli_init_once();

	// Initialize a local runtime with global settings if necessary. Note
	// that in the case that a runtime is passed in, we make a local copy.
	rntm_t rntm_l;
	if ( rntm == NULL ) { bli_rntm_init_from_global( &rntm_l ); }
	else                { rntm_l = *rntm;                       }

	// Default to using native execution.
	num_t dt = bli_obj_dt( c );
	ind_t im = BLIS_NAT;

	// If all matrix operands are complex and of the same storage datatype, try
	// to get an induced method (if one is available and enabled).
	if ( bli_obj_dt( a ) == bli_obj_dt( c ) &&
	     bli_obj_dt( b ) == bli_obj_dt( c ) &&
	     bli_obj_is_complex( c ) )
	{
		// Find the highest priority induced method that is both enabled and
		// available for the current operation. (If an induced method is
		// available but not enabled, or simply unavailable, BLIS_NAT will
		// be returned here.)
		im = bli_hemmind_find_avail( dt );
	}

	// If necessary, obtain a valid context from the gks using the induced
	// method id determined above.
	if ( cntx == NULL ) cntx = bli_gks_query_ind_cntx( im );

	// Check the operands.
	if ( bli_error_checking_is_enabled() )
		bli_hemm_check( side, alpha, a, b, beta, c, cntx );

	// Invoke the operation's front-end and request the default control tree.
	bli_hemm_front( side, alpha, a, b, beta, c, cntx, &rntm_l );
}


void PASTEMAC(symm,BLIS_OAPI_EX_SUF)
     (
             side_t  side,
       const obj_t*  alpha,
       const obj_t*  a,
       const obj_t*  b,
       const obj_t*  beta,
       const obj_t*  c,
       const cntx_t* cntx,
       const rntm_t* rntm
     )
{
	bli_init_once();

	// Initialize a local runtime with global settings if necessary. Note
	// that in the case that a runtime is passed in, we make a local copy.
	rntm_t rntm_l;
	if ( rntm == NULL ) { bli_rntm_init_from_global( &rntm_l ); }
	else                { rntm_l = *rntm;                       }

	// Default to using native execution.
	num_t dt = bli_obj_dt( c );
	ind_t im = BLIS_NAT;

	// If all matrix operands are complex and of the same storage datatype, try
	// to get an induced method (if one is available and enabled).
	if ( bli_obj_dt( a ) == bli_obj_dt( c ) &&
	     bli_obj_dt( b ) == bli_obj_dt( c ) &&
	     bli_obj_is_complex( c ) )
	{
		// Find the highest priority induced method that is both enabled and
		// available for the current operation. (If an induced method is
		// available but not enabled, or simply unavailable, BLIS_NAT will
		// be returned here.)
		im = bli_symmind_find_avail( dt );
	}

	// If necessary, obtain a valid context from the gks using the induced
	// method id determined above.
	if ( cntx == NULL ) cntx = bli_gks_query_ind_cntx( im );

	// Check the operands.
	if ( bli_error_checking_is_enabled() )
		bli_symm_check( side, alpha, a, b, beta, c, cntx );

	// Invoke the operation's front-end and request the default control tree.
	bli_symm_front( side, alpha, a, b, beta, c, cntx, &rntm_l );
}


void PASTEMAC(trmm3,BLIS_OAPI_EX_SUF)
     (
             side_t  side,
       const obj_t*  alpha,
       const obj_t*  a,
       const obj_t*  b,
       const obj_t*  beta,
       const obj_t*  c,
       const cntx_t* cntx,
       const rntm_t* rntm
     )
{
	bli_init_once();

	// Initialize a local runtime with global settings if necessary. Note
	// that in the case that a runtime is passed in, we make a local copy.
	rntm_t rntm_l;
	if ( rntm == NULL ) { bli_rntm_init_from_global( &rntm_l ); }
	else                { rntm_l = *rntm;                       }

	// Default to using native execution.
	num_t dt = bli_obj_dt( c );
	ind_t im = BLIS_NAT;

	// If all matrix operands are complex and of the same storage datatype, try
	// to get an induced method (if one is available and enabled).
	if ( bli_obj_dt( a ) == bli_obj_dt( c ) &&
	     bli_obj_dt( b ) == bli_obj_dt( c ) &&
	     bli_obj_is_complex( c ) )
	{
		// Find the highest priority induced method that is both enabled and
		// available for the current operation. (If an induced method is
		// available but not enabled, or simply unavailable, BLIS_NAT will
		// be returned here.)
		im = bli_trmm3ind_find_avail( dt );
	}

	// If necessary, obtain a valid context from the gks using the induced
	// method id determined above.
	if ( cntx == NULL ) cntx = bli_gks_query_ind_cntx( im );

	// Check the operands.
	if ( bli_error_checking_is_enabled() )
		bli_trmm3_check( side, alpha, a, b, beta, c, cntx );

	// Invoke the operation's front-end and request the default control tree.
	bli_trmm3_front( side, alpha, a, b, beta, c, cntx, &rntm_l );
}


void PASTEMAC(herk,BLIS_OAPI_EX_SUF)
     (
       const obj_t*  alpha,
       const obj_t*  a,
       const obj_t*  beta,
       const obj_t*  c,
       const cntx_t* cntx,
       const rntm_t* rntm
     )
{
	bli_init_once();

	obj_t ah;

	// Check parameters.
	if ( bli_error_checking_is_enabled() )
		bli_herk_check( alpha, a, beta, c, cntx );

	bli_obj_alias_to( a, &ah );
	bli_obj_toggle_trans( &ah );
	bli_obj_toggle_conj( &ah );

	PASTEMAC(gemmt,BLIS_OAPI_EX_SUF)( alpha, a, &ah, beta, c, cntx, rntm );

	// The Hermitian rank-k product was computed as Re(alpha)*A*A', even for the
	// diagonal elements. Mathematically, the imaginary components of
	// diagonal elements of a Hermitian rank-k product should always be
	// zero. However, in practice, they sometimes accumulate meaningless
	// non-zero values. To prevent this, we explicitly set those values
	// to zero before returning.
	bli_setid( &BLIS_ZERO, c );
}


void PASTEMAC(syrk,BLIS_OAPI_EX_SUF)
     (
       const obj_t*  alpha,
       const obj_t*  a,
       const obj_t*  beta,
       const obj_t*  c,
       const cntx_t* cntx,
       const rntm_t* rntm
     )
{
	bli_init_once();

	obj_t at;

	// Check parameters.
	if ( bli_error_checking_is_enabled() )
		bli_syrk_check( alpha, a, beta, c, cntx );

	bli_obj_alias_to( a, &at );
	bli_obj_toggle_trans( &at );

	PASTEMAC(gemmt,BLIS_OAPI_EX_SUF)( alpha, a, &at, beta, c, cntx, rntm );
}


void PASTEMAC(trmm,BLIS_OAPI_EX_SUF)
     (
             side_t  side,
       const obj_t*  alpha,
       const obj_t*  a,
       const obj_t*  b,
       const cntx_t* cntx,
       const rntm_t* rntm
     )
{
	bli_init_once();

	// Initialize a local runtime with global settings if necessary. Note
	// that in the case that a runtime is passed in, we make a local copy.
	rntm_t rntm_l;
	if ( rntm == NULL ) { bli_rntm_init_from_global( &rntm_l ); }
	else                { rntm_l = *rntm;                       }

	// Default to using native execution.
	num_t dt = bli_obj_dt( b );
	ind_t im = BLIS_NAT;

	// If all matrix operands are complex and of the same storage datatype, try
	// to get an induced method (if one is available and enabled).
	if ( bli_obj_dt( a ) == bli_obj_dt( b ) &&
	     bli_obj_is_complex( b ) )
	{
		// Find the highest priority induced method that is both enabled and
		// available for the current operation. (If an induced method is
		// available but not enabled, or simply unavailable, BLIS_NAT will
		// be returned here.)
		im = bli_trmmind_find_avail( dt );
	}

	// If necessary, obtain a valid context from the gks using the induced
	// method id determined above.
	if ( cntx == NULL ) cntx = bli_gks_query_ind_cntx( im );

	// Check the operands.
	if ( bli_error_checking_is_enabled() )
		bli_trmm_check( side, alpha, a, b, cntx );

	// Invoke the operation's front-end and request the default control tree.
	bli_trmm_front( side, alpha, a, b, cntx, &rntm_l );
}


void PASTEMAC(trsm,BLIS_OAPI_EX_SUF)
     (
             side_t  side,
       const obj_t*  alpha,
       const obj_t*  a,
       const obj_t*  b,
       const cntx_t* cntx,
       const rntm_t* rntm
     )
{
	bli_init_once();

	// Initialize a local runtime with global settings if necessary. Note
	// that in the case that a runtime is passed in, we make a local copy.
	rntm_t rntm_l;
	if ( rntm == NULL ) { bli_rntm_init_from_global( &rntm_l ); }
	else                { rntm_l = *rntm;                       }

	// Default to using native execution.
	num_t dt = bli_obj_dt( b );
	ind_t im = BLIS_NAT;

	// If all matrix operands are complex and of the same storage datatype, try
	// to get an induced method (if one is available and enabled).
	if ( bli_obj_dt( a ) == bli_obj_dt( b ) &&
	     bli_obj_is_complex( b ) )
	{
		// Find the highest priority induced method that is both enabled and
		// available for the current operation. (If an induced method is
		// available but not enabled, or simply unavailable, BLIS_NAT will
		// be returned here.)
		im = bli_trsmind_find_avail( dt );
	}

	// If necessary, obtain a valid context from the gks using the induced
	// method id determined above.
	if ( cntx == NULL ) cntx = bli_gks_query_ind_cntx( im );

	// Check the operands.
	if ( bli_error_checking_is_enabled() )
		bli_trsm_check( side, alpha, a, b, cntx );

	// Invoke the operation's front-end and request the default control tree.
	bli_trsm_front( side, alpha, a, b, cntx, &rntm_l );
}
