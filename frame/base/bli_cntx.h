/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2014, The University of Texas at Austin
   Copyright (C) 2016, Hewlett Packard Enterprise Development LP
   Copyright (C) 2019, Advanced Micro Devices, Inc.

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

#ifndef BLIS_CNTX_H
#define BLIS_CNTX_H


// Context object type (defined in bli_type_defs.h)

/*
typedef struct cntx_s
{
	blksz_t   blkszs[ BLIS_NUM_BLKSZS ];
	bszid_t   bmults[ BLIS_NUM_BLKSZS ];

	func_t    ukrs[ BLIS_NUM_UKRS ];
	mbool_t   ukr_prefs[ BLIS_NUM_UKR_PREFS ];

	void_fp   l3_sup_handlers[ BLIS_NUM_LEVEL3_OPS ];

	ind_t     method;

} cntx_t;
*/

// -----------------------------------------------------------------------------

//
// -- cntx_t query (complex) ---------------------------------------------------
//

BLIS_INLINE const blksz_t* bli_cntx_get_blksz( bszid_t bs_id, const cntx_t* cntx )
{
	// Return the address of the blksz_t identified by bs_id.
	return &cntx->blkszs[ bs_id ];
}

BLIS_INLINE dim_t bli_cntx_get_blksz_def_dt( num_t dt, bszid_t bs_id, const cntx_t* cntx )
{
	const blksz_t* blksz  = bli_cntx_get_blksz( bs_id, cntx );
	dim_t          bs_dt  = bli_blksz_get_def( dt, blksz );

	// Return the main (default) blocksize value for the datatype given.
	return bs_dt;
}

BLIS_INLINE dim_t bli_cntx_get_blksz_max_dt( num_t dt, bszid_t bs_id, const cntx_t* cntx )
{
	const blksz_t* blksz  = bli_cntx_get_blksz( bs_id, cntx );
	dim_t          bs_dt  = bli_blksz_get_max( dt, blksz );

	// Return the auxiliary (maximum) blocksize value for the datatype given.
	return bs_dt;
}

// -----------------------------------------------------------------------------

BLIS_INLINE const func_t* bli_cntx_get_ukrs( ukr_t ukr_id, const cntx_t* cntx )
{
	return &cntx->ukrs[ ukr_id ];
}

BLIS_INLINE void_fp bli_cntx_get_ukr_dt( num_t dt, ukr_t ukr_id, const cntx_t* cntx )
{
	const func_t* func = bli_cntx_get_ukrs( ukr_id, cntx );

	return bli_func_get_dt( dt, func );
}

// -----------------------------------------------------------------------------

BLIS_INLINE const mbool_t* bli_cntx_get_ukr_prefs( ukr_pref_t pref_id, const cntx_t* cntx )
{
	return &cntx->ukr_prefs[ pref_id ];
}

BLIS_INLINE bool bli_cntx_get_ukr_prefs_dt( num_t dt, ukr_pref_t ukr_id, const cntx_t* cntx )
{
	const mbool_t* mbool = bli_cntx_get_ukr_prefs( ukr_id, cntx );

	return ( bool )bli_mbool_get_dt( dt, mbool );
}

// -----------------------------------------------------------------------------

BLIS_INLINE bool bli_cntx_l3_sup_thresh_is_met( num_t dt, dim_t m, dim_t n, dim_t k, const cntx_t* cntx )
{
	if ( m < bli_cntx_get_blksz_def_dt( dt, BLIS_MT, cntx ) ) return TRUE;
	if ( n < bli_cntx_get_blksz_def_dt( dt, BLIS_NT, cntx ) ) return TRUE;
	if ( k < bli_cntx_get_blksz_def_dt( dt, BLIS_KT, cntx ) ) return TRUE;

	return FALSE;
}

// -----------------------------------------------------------------------------

BLIS_INLINE void_fp bli_cntx_get_l3_sup_handler( opid_t op, const cntx_t* cntx )
{
	return cntx->l3_sup_handlers[ op ];
}

// -----------------------------------------------------------------------------

//
// -- cntx_t modification (complex) --------------------------------------------
//

// NOTE: The framework does not use any of the following functions. We provide
// them in order to facilitate creating/modifying custom contexts.

BLIS_INLINE void bli_cntx_set_blksz( bszid_t bs_id, blksz_t* blksz, cntx_t* cntx )
{
	cntx->blkszs[ bs_id ] = *blksz;
}

BLIS_INLINE void bli_cntx_set_blksz_def_dt( num_t dt, bszid_t bs_id, dim_t bs, cntx_t* cntx )
{
	bli_blksz_set_def( bs, dt, &cntx->blkszs[ bs_id ] );
}

BLIS_INLINE void bli_cntx_set_blksz_max_dt( num_t dt, bszid_t bs_id, dim_t bs, cntx_t* cntx )
{
	bli_blksz_set_max( bs, dt, &cntx->blkszs[ bs_id ]);
}

BLIS_INLINE void bli_cntx_set_ukr( ukr_t ukr_id, const func_t* func, cntx_t* cntx )
{
	cntx->ukrs[ ukr_id ] = *func;
}

BLIS_INLINE void bli_cntx_set_ukr_dt( void_fp fp, num_t dt, ukr_t ker_id, cntx_t* cntx )
{
	bli_func_set_dt( fp, dt, &cntx->ukrs[ ker_id ] );
}

BLIS_INLINE void bli_cntx_set_ukr_pref( ukr_pref_t ukr_id, mbool_t* prefs, cntx_t* cntx )
{
	cntx->ukr_prefs[ ukr_id ] = *prefs;
}

BLIS_INLINE void_fp bli_cntx_get_l3_sup_ker_dt( num_t dt, stor3_t stor_id, const cntx_t* cntx )
{
	ukr_t ukr_id = bli_stor3_ukr( stor_id );

	return bli_cntx_get_ukr_dt( dt, ukr_id, cntx );
}

BLIS_INLINE dim_t bli_cntx_get_l3_sup_blksz_def_dt( num_t dt, bszid_t bs_id, const cntx_t* cntx )
{
	switch ( bs_id )
	{
		case BLIS_MR: bs_id = BLIS_MR_SUP; break;
		case BLIS_NR: bs_id = BLIS_NR_SUP; break;
		case BLIS_KR: bs_id = BLIS_KR_SUP; break;
		case BLIS_MC: bs_id = BLIS_MC_SUP; break;
		case BLIS_NC: bs_id = BLIS_NC_SUP; break;
		case BLIS_KC: bs_id = BLIS_KC_SUP; break;
		default: break;
	};

	return bli_cntx_get_blksz_def_dt( dt, bs_id, cntx );
}

BLIS_INLINE dim_t bli_cntx_get_l3_sup_blksz_max_dt( num_t dt, bszid_t bs_id, const cntx_t* cntx )
{
	switch ( bs_id )
	{
		case BLIS_MR: bs_id = BLIS_MR_SUP; break;
		case BLIS_NR: bs_id = BLIS_NR_SUP; break;
		case BLIS_KR: bs_id = BLIS_KR_SUP; break;
		case BLIS_MC: bs_id = BLIS_MC_SUP; break;
		case BLIS_NC: bs_id = BLIS_NC_SUP; break;
		case BLIS_KC: bs_id = BLIS_KC_SUP; break;
		default: break;
	};

	return bli_cntx_get_blksz_max_dt( dt, bs_id, cntx );
}

// -----------------------------------------------------------------------------

// Function prototypes

BLIS_EXPORT_BLIS void bli_cntx_clear( cntx_t* cntx );

BLIS_EXPORT_BLIS void bli_cntx_set_blkszs( cntx_t* cntx, ... );

BLIS_EXPORT_BLIS void bli_cntx_set_ind_blkszs( ind_t method, num_t dt, cntx_t* cntx, ... );

BLIS_EXPORT_BLIS void bli_cntx_set_ukrs( cntx_t* cntx, ... );
BLIS_EXPORT_BLIS void bli_cntx_set_ukr_prefs( cntx_t* cntx, ... );

BLIS_EXPORT_BLIS void bli_cntx_print( const cntx_t* cntx );

BLIS_EXPORT_BLIS void bli_cntx_set_l3_sup_handlers( cntx_t* cntx, ... );


#endif

