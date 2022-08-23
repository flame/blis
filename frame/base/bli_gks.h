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

#ifndef BLIS_GKS_H
#define BLIS_GKS_H

bool                   bli_gks_is_init( void );
void                   bli_gks_mark_init( void );
void                   bli_gks_mark_uninit( void );

err_t                  bli_gks_init( void );
err_t                  bli_gks_finalize( void );

void                   bli_gks_init_index( void );

err_t                  bli_gks_lookup_nat_cntx( arch_t id, const cntx_t** cntx );
err_t                  bli_gks_lookup_ind_cntx( arch_t id, ind_t ind, const cntx_t** cntx );
const cntx_t* const *  bli_gks_lookup_id( arch_t id );
err_t                  bli_gks_register_cntx( arch_t id, void_fp nat_fp, void_fp ref_fp, void_fp ind_fp );

BLIS_EXPORT_BLIS err_t bli_gks_query_cntx( const cntx_t** cntx );
BLIS_EXPORT_BLIS err_t bli_gks_query_nat_cntx( const cntx_t** cntx );

err_t                  bli_gks_query_cntx_noinit( const cntx_t** cntx );

BLIS_EXPORT_BLIS err_t bli_gks_query_ind_cntx( ind_t ind, const cntx_t** cntx );

BLIS_EXPORT_BLIS err_t bli_gks_init_ref_cntx( cntx_t* cntx );

err_t                  bli_gks_cntx_l3_nat_ukr_is_ref( num_t dt, ukr_t ukr_id, const cntx_t* cntx, bool* is_ref );

BLIS_EXPORT_BLIS err_t bli_gks_l3_ukr_impl_string( ukr_t ukr, ind_t method, num_t dt, const char** str );
BLIS_EXPORT_BLIS err_t bli_gks_l3_ukr_impl_type( ukr_t ukr, ind_t method, num_t dt, kimpl_t* ki );

//
// -- cntx_t* query convenience wrapper ----------------------------------------
//

BLIS_INLINE err_t bli_gks_query_cntx_if_null( const cntx_t** cntx )
{
	err_t r_val;

	if ( *cntx == NULL )
	{
		r_val = bli_gks_query_nat_cntx( cntx );
		bli_check_return_if_failure( r_val );
	}

	return BLIS_SUCCESS;
}

BLIS_INLINE err_t bli_gks_query_ind_cntx_if_null( ind_t im, const cntx_t** cntx )
{
	err_t r_val;

	if ( *cntx == NULL )
	{
		r_val = bli_gks_query_ind_cntx( im, cntx );
		bli_check_return_if_failure( r_val );
	}

	return BLIS_SUCCESS;
}

#endif

