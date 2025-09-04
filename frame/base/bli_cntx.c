/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2014, The University of Texas at Austin
   Copyright (C) 2020 - 2023, Advanced Micro Devices, Inc. All rights reserved.

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

void bli_cntx_clear( cntx_t* cntx )
{
	// Fill the entire cntx_t structure with zeros.
	memset( ( void* )cntx, 0, sizeof( cntx_t ) );
}

// -----------------------------------------------------------------------------

void bli_cntx_set_blkszs( cntx_t* cntx, ... )
{
	// This function can be called from the bli_cntx_init_*() function for
	// a particular architecture if the kernel developer wishes to use
	// non-default blocksizes. It should be called after
	// bli_cntx_init_<subconfig>_ref() so that the context begins with
	// default blocksizes across all datatypes.

	/* Example prototypes:

	   void bli_cntx_set_blkszs
	   (
	     cntx_t* cntx,
	     bszid_t bs0_id, blksz_t* blksz0, bszid_t bm0_id,
	     bszid_t bs1_id, blksz_t* blksz1, bszid_t bm1_id,
	     bszid_t bs2_id, blksz_t* blksz2, bszid_t bm2_id,
	     ...,
	     BLIS_VA_END
	   );
	*/

	// Save the execution type into the context.
	bli_cntx_set_method( BLIS_NAT, cntx );

	// Query the context for the addresses of:
	// - the blocksize object array
	// - the blocksize multiple array
	blksz_t* cntx_blkszs = cntx->blkszs;
	bszid_t* cntx_bmults = cntx->bmults;

	// Initialize variable argument environment.
	va_list args;
	va_start( args, cntx );

	// Process blocksizes until we get a BLIS_VA_END.
	while ( true )
	{
		int bs_id0 = va_arg( args, int );

		// If we find a bszid_t id of BLIS_VA_END, then we are done.
		if ( bs_id0 == BLIS_VA_END ) break;

		// Here, we query the variable argument list for:
		// - the bszid_t of the blocksize we're about to process (already done),
		// - the address of the blksz_t object,
		// - the bszid_t of the multiple we need to associate with
		//   the blksz_t object.
		bszid_t  bs_id = ( bszid_t  )bs_id0;
		blksz_t* blksz = ( blksz_t* )va_arg( args, blksz_t* );
		bszid_t  bm_id = ( bszid_t  )va_arg( args, bszid_t  );

		// Copy the blksz_t object contents into the appropriate
		// location within the context's blksz_t array. Do the same
		// for the blocksize multiple id.
		//cntx_blkszs[ bs_id ] = *blksz;
		//bli_blksz_copy( blksz, cntx_blksz );
		blksz_t* cntx_blksz = &cntx_blkszs[ bs_id ];
		bli_blksz_copy_if_pos( blksz, cntx_blksz );

		// Copy the blocksize multiple id into the context.
		cntx_bmults[ bs_id ] = bm_id;
	}

	// Shutdown variable argument environment and clean up stack.
	va_end( args );
}

// -----------------------------------------------------------------------------

void bli_cntx_set_ind_blkszs( ind_t method, num_t dt, cntx_t* cntx, ... )
{
	/* Example prototypes:

	   void bli_gks_cntx_set_ind_blkszs
	   (
	     ind_t   method != BLIS_NAT,
	     num_t   dt,
	     cntx_t* cntx,
	     bszid_t bs0_id, dim_t def_scalr0, dim_t max_scalr0,
	     bszid_t bs1_id, dim_t def_scalr1, dim_t max_scalr1,
	     bszid_t bs2_id, dim_t def_scalr2, dim_t max_scalr2,
	     ...,
	     BLIS_VA_END
	   );

		NOTE: This function modifies an existing context that is presumed
		to have been initialized for native execution.
	*/

	// Project the given datatype to the real domain. This will be used later on.
	num_t dt_real = bli_dt_proj_to_real( dt );

	// Return early if called with BLIS_NAT.
	if ( method == BLIS_NAT ) return;

	// Save the execution type into the context.
	bli_cntx_set_method( method, cntx );

	// Initialize variable argument environment.
	va_list args;
	va_start( args, cntx );

	// Process blocksizes until we get a BLIS_VA_END.
	while ( true )
	{
		int bs_id0 = va_arg( args, int );

		// If we find a bszid_t id of BLIS_VA_END, then we are done.
		if ( bs_id0 == BLIS_VA_END ) break;

		// Here, we query the variable argument list for:
		// - the bszid_t of the blocksize we're about to process (already done),
		// - the scalars we wish to apply to the real blocksizes to
		//   come up with the induced complex blocksizes (for default
		//   and maximum blocksizes).
		bszid_t bs_id = ( bszid_t )bs_id0;
		double  dsclr = ( double  )va_arg( args, double );
		double  msclr = ( double  )va_arg( args, double );

		// Query the context for the blksz_t object assoicated with the
		// current blocksize id, and also query the object corresponding
		// to the blocksize multiple.
		blksz_t* cntx_blksz = ( blksz_t* )bli_cntx_get_blksz( bs_id, cntx );

		// Copy the real domain value of the blksz_t object into the
		// corresponding complex domain slot of the same object.
		bli_blksz_copy_dt( dt_real, cntx_blksz, dt, cntx_blksz );

		// If the default blocksize scalar is non-unit, we need to scale
		// the complex domain default blocksizes.
		if ( dsclr != 1.0 )
		{
			// Read the current blocksize id, blocksize multiple id,
			// and blocksize scalar.
			bszid_t  bs_id = bszids[ i ];
			double   dsclr = dsclrs[ i ];
			double   msclr = msclrs[ i ];

			//blksz_t* cntx_blksz = &cntx_blkszs[ bs_id ];

			// Query the context for the blksz_t object assoicated with the
			// current blocksize id, and also query the object corresponding
			// to the blocksize multiple.
			blksz_t* cntx_blksz = bli_cntx_get_blksz( bs_id, cntx );
			blksz_t* cntx_trsm_blksz = bli_cntx_get_trsm_blksz( bs_id, cntx );

			// Copy the real domain value of the blksz_t object into the
			// corresponding complex domain slot of the same object.
			bli_blksz_copy_dt( dt_real, cntx_blksz, dt, cntx_blksz );
			bli_blksz_copy_dt( dt_real, cntx_blksz, dt, cntx_trsm_blksz );

			// If the default blocksize scalar is non-unit, we need to scale
			// the complex domain default blocksizes.
			if ( dsclr != 1.0 )
			{
				// Scale the default blocksize value corresponding to the given
				// datatype.
				bli_blksz_scale_def( 1, ( dim_t )dsclr, dt, cntx_blksz );
				bli_blksz_scale_def( 1, ( dim_t )dsclr, dt, cntx_trsm_blksz );
			}

			// Similarly, if the maximum blocksize scalar is non-unit, we need
			// to scale the complex domain maximum blocksizes.
			if ( msclr != 1.0 )
			{
				// Scale the maximum blocksize value corresponding to the given
				// datatype.
				bli_blksz_scale_max( 1, ( dim_t )msclr, dt, cntx_blksz );
				bli_blksz_scale_max( 1, ( dim_t )msclr, dt, cntx_trsm_blksz );
			}
		}

		// Similarly, if the maximum blocksize scalar is non-unit, we need
		// to scale the complex domain maximum blocksizes.
		if ( msclr != 1.0 )
		{
			// Scale the maximum blocksize value corresponding to the given
			// datatype.
			bli_blksz_scale_max( 1, ( dim_t )msclr, dt, cntx_blksz );
		}
	}

	// Shutdown variable argument environment and clean up stack.
	va_end( args );
}

// -----------------------------------------------------------------------------

void bli_cntx_set_ukrs( cntx_t* cntx , ... )
{
	// This function can be called from the bli_cntx_init_*() function for
	// a particular architecture if the kernel developer wishes to use
	// non-default microkernels. It should be called after
	// bli_cntx_init_<subconfig>_ref() so that the context begins with
	// default microkernels across all datatypes.

	/* Example prototypes:

	   void bli_cntx_set_ukrs
	   (
	     cntx_t* cntx,
	     ukr_t ukr0_id, num_t dt0, void_fp ukr0_fp,
	     ukr_t ukr1_id, num_t dt1, void_fp ukr1_fp,
	     ukr_t ukr2_id, num_t dt2, void_fp ukr2_fp,
	     ...,
	     BLIS_VA_END
	   );
	*/

	// Query the context for the address of the ukernel func_t array
	func_t*  cntx_ukrs = cntx->ukrs;

	// Initialize variable argument environment.
	va_list   args;
	va_start( args, cntx );

	// Process ukernels until BLIS_VA_END is reached.
	while ( true )
	{
		const int ukr_id0 = va_arg( args, int );

		// If we find a ukernel id of BLIS_VA_END, then we are done.
		if ( ukr_id0 == BLIS_VA_END ) break;

		// Here, we query the variable argument list for:
		// - the ukr_t of the kernel we're about to process (already done),
		// - the datatype of the kernel, and
		// - the kernel function pointer
		const ukr_t   ukr_id = ( ukr_t   )ukr_id0;
		const num_t   ukr_dt = ( num_t   )va_arg( args, num_t   );
		      void_fp ukr_fp = ( void_fp )va_arg( args, void_fp );

		// Index into the func_t and mbool_t for the current kernel id
		// being processed.
		func_t* ukrs = &cntx_ukrs[ ukr_id ];

		// Store the ukernel function pointer into the context.
		// Notice that we redundantly store the native
		// ukernel address in both the native and virtual ukernel slots
		// in the context. This is standard practice when creating a
		// native context. (Induced method contexts will overwrite the
		// virtual function pointer with the address of the appropriate
		// virtual ukernel.)
		bli_func_set_dt( ukr_fp, ukr_dt, ukrs );

		// Locate the virtual ukernel func_t pointer that corresponds to the
		// ukernel id provided by the caller.
		switch ( ukr_id )
		{
			case BLIS_GEMM_UKR:       ukrs = &cntx_ukrs[ BLIS_GEMM_VIR_UKR ]; break;
			case BLIS_GEMMTRSM_L_UKR: ukrs = &cntx_ukrs[ BLIS_GEMMTRSM_L_VIR_UKR ]; break;
			case BLIS_GEMMTRSM_U_UKR: ukrs = &cntx_ukrs[ BLIS_GEMMTRSM_U_VIR_UKR ]; break;
			case BLIS_TRSM_L_UKR:     ukrs = &cntx_ukrs[ BLIS_TRSM_L_VIR_UKR ]; break;
			case BLIS_TRSM_U_UKR:     ukrs = &cntx_ukrs[ BLIS_TRSM_U_VIR_UKR ]; break;
			default:                  ukrs = NULL; break;
		};

		if ( ukrs )
			bli_func_set_dt( ukr_fp, ukr_dt, ukrs );
	}

	// Shutdown variable argument environment and clean up stack.
	va_end( args );
}

// -----------------------------------------------------------------------------

void bli_cntx_set_ukr_prefs( cntx_t* cntx , ... )
{
	// This function can be called from the bli_cntx_init_*() function for
	// a particular architecture if the kernel developer wishes to use
	// non-default microkernel preferences. It should be called after
	// bli_cntx_init_<subconfig>_ref() so that the context begins with
	// default preferences across all datatypes.

	/* Example prototypes:

	   void bli_cntx_set_ukr_prefs
	   (
	     cntx_t* cntx,
	     ukr_pref_t ukr_pref0_id, num_t dt0, bool ukr_pref0,
	     ukr_pref_t ukr_pref1_id, num_t dt1, bool ukr_pref1,
	     ukr_pref_t ukr_pref2_id, num_t dt2, bool ukr_pref2,
	     ...,
	     BLIS_VA_END
	   );
	*/

	// Query the context for the address of the ukernel preference mbool_t array
	mbool_t* cntx_ukr_prefs = cntx->ukr_prefs;

	// Initialize variable argument environment.
	va_start( args, n_bs );

	// Process n_bs tuples.
	for ( i = 0; i < n_bs; ++i )
	{
		// Here, we query the variable argument list for:
		// - the bszid_t of the blocksize we're about to process,
		// - the address of the blksz_t object.
		bszid_t  bs_id = ( bszid_t  )va_arg( args, bszid_t  );
		blksz_t* blksz = ( blksz_t* )va_arg( args, blksz_t* );

		// Store the values in our temporary arrays.
		bszids[ i ] = bs_id;
		blkszs[ i ] = blksz;
	}

	// The last argument should be the context pointer.
	cntx_t* cntx = ( cntx_t* )va_arg( args, cntx_t* );

	// Shutdown variable argument environment and clean up stack.
	va_end( args );

	// -- End variable argument section --

	// Query the context for the addresses of:
	// - the blocksize object array
	blksz_t* cntx_l3_sup_blkszs = bli_cntx_l3_sup_blkszs_buf( cntx );

	// Now that we have the context address, we want to copy the values
	// from the temporary buffers into the corresponding buffers in the
	// context. Notice that the blksz_t* pointers were saved, rather than
	// the objects themselves, but we copy the contents of the objects
	// when copying into the context.

	// Process each blocksize id tuple provided.
	for ( i = 0; i < n_bs; ++i )
	{
		// Read the current blocksize id, blksz_t* pointer, blocksize
		// multiple id, and blocksize scalar.
		bszid_t  bs_id = bszids[ i ];
		blksz_t* blksz = blkszs[ i ];

		blksz_t* cntx_l3_sup_blksz = &cntx_l3_sup_blkszs[ bs_id ];

		// Copy the blksz_t object contents into the appropriate
		// location within the context's blksz_t array.
		//cntx_l3_sup_blkszs[ bs_id ] = *blksz;
		//bli_blksz_copy( blksz, cntx_l3_sup_blksz );
		bli_blksz_copy_if_pos( blksz, cntx_l3_sup_blksz );
	}

	// Free the temporary local arrays.

	#ifdef BLIS_ENABLE_MEM_TRACING
	printf( "bli_cntx_set_blkszs(): " );
	#endif
	bli_free_intl( blkszs );

	#ifdef BLIS_ENABLE_MEM_TRACING
	printf( "bli_cntx_set_blkszs(): " );
	#endif
	bli_free_intl( bszids );
}

// -----------------------------------------------------------------------------

void bli_cntx_set_l3_sup_tri_blkszs( dim_t n_bs, ... )
{
	// This function can be called from the bli_cntx_init_*() function for
	// a particular architecture if the kernel developer wishes to use
	// non-default l3 sup blocksizes for operations that deal with triangular
	// matrices. It should be called after bli_cntx_init_defaults() so that 
	// the context begins with default blocksizes across all datatypes.

	/* Example prototypes:

	   void bli_cntx_set_blkszs
	   (
	     dim_t   n_bs,
	     bszid_t bs0_id, blksz_t* blksz0,
	     bszid_t bs1_id, blksz_t* blksz1,
	     bszid_t bs2_id, blksz_t* blksz2,
	     ...
	     cntx_t* cntx
	   );
	*/

	va_list   args;
	dim_t     i;
	err_t     r_val;

	// Allocate some temporary local arrays.
	#ifdef BLIS_ENABLE_MEM_TRACING
	printf( "bli_cntx_set_blkszs(): " );
	#endif
	bszid_t* bszids = bli_malloc_intl( n_bs * sizeof( bszid_t  ), &r_val );

	#ifdef BLIS_ENABLE_MEM_TRACING
	printf( "bli_cntx_set_blkszs(): " );
	#endif
	blksz_t** blkszs = bli_malloc_intl( n_bs * sizeof( blksz_t* ), &r_val );

	// -- Begin variable argument section --

	// Initialize variable argument environment.
	va_start( args, n_bs );

	// Process n_bs tuples.
	for ( i = 0; i < n_bs; ++i )
	{
		// Here, we query the variable argument list for:
		// - the bszid_t of the blocksize we're about to process,
		// - the address of the blksz_t object.
		bszid_t  bs_id = ( bszid_t  )va_arg( args, bszid_t  );
		blksz_t* blksz = ( blksz_t* )va_arg( args, blksz_t* );

		// Store the values in our temporary arrays.
		bszids[ i ] = bs_id;
		blkszs[ i ] = blksz;
	}

	// The last argument should be the context pointer.
	cntx_t* cntx = ( cntx_t* )va_arg( args, cntx_t* );

	// Shutdown variable argument environment and clean up stack.
	va_end( args );

	// -- End variable argument section --

	// Query the context for the addresses of:
	// - the blocksize object array
	blksz_t* cntx_l3_sup_tri_blkszs = bli_cntx_l3_sup_tri_blkszs_buf( cntx );

	// Now that we have the context address, we want to copy the values
	// from the temporary buffers into the corresponding buffers in the
	// context. Notice that the blksz_t* pointers were saved, rather than
	// the objects themselves, but we copy the contents of the objects
	// when copying into the context.

	// Process each blocksize id tuple provided.
	for ( i = 0; i < n_bs; ++i )
	{
		// Read the current blocksize id, blksz_t* pointer, blocksize
		// multiple id, and blocksize scalar.
		bszid_t  bs_id = bszids[ i ];
		blksz_t* blksz = blkszs[ i ];

		blksz_t* cntx_l3_sup_tri_blksz = &cntx_l3_sup_tri_blkszs[ bs_id ];

		// Copy the blksz_t object contents into the appropriate
		// location within the context's blksz_t array.
		//cntx_l3_sup_blkszs[ bs_id ] = *blksz;
		//bli_blksz_copy( blksz, cntx_l3_sup_blksz );
		bli_blksz_copy_if_pos( blksz, cntx_l3_sup_tri_blksz );
	}

	// Free the temporary local arrays.

	#ifdef BLIS_ENABLE_MEM_TRACING
	printf( "bli_cntx_set_blkszs(): " );
	#endif
	bli_free_intl( blkszs );

	#ifdef BLIS_ENABLE_MEM_TRACING
	printf( "bli_cntx_set_blkszs(): " );
	#endif
	bli_free_intl( bszids );
}

// -----------------------------------------------------------------------------

void bli_cntx_set_l3_sup_kers( dim_t n_ukrs, ... )
{
	// This function can be called from the bli_cntx_init_*() function for
	// a particular architecture if the kernel developer wishes to use
	// non-default level-3 microkernels for small/unpacked matrices. It
	// should be called after bli_cntx_init_defaults() so that the context
	// begins with default sup micro/millikernels across all datatypes.

	/* Example prototypes:

	   void bli_cntx_set_l3_sup_kers
	   (
	     dim_t   n_ukrs,
	     stor3_t stor_id0, num_t dt0, void* ukr0_fp, bool pref0,
	     stor3_t stor_id1, num_t dt1, void* ukr1_fp, bool pref1,
	     stor3_t stor_id2, num_t dt2, void* ukr2_fp, bool pref2,
	     ...
	     cntx_t* cntx
	   );
	*/

	va_list   args;
	va_start( args, cntx );

	// Process ukernel preferences until BLIS_VA_END is reached.
	while ( true )
	{
		// Here, we query the variable argument list for:
		// - the stor3_t storage case being assigned to the kernel we're
		//   about to process,
		// - the datatype of the kernel,
		// - the kernel function pointer, and
		// - the kernel function storage preference
		// that we need to store to the context.
		const stor3_t  st3_id   = ( stor3_t )va_arg( args, stor3_t );
		const num_t    ukr_dt   = ( num_t   )va_arg( args, num_t   );
		      void*    ukr_fp   = ( void*   )va_arg( args, void*   );
		const bool     ukr_pref = ( bool    )va_arg( args, int     );

		// Store the values in our temporary arrays.
		st3_ids[ i ]   = st3_id;
		ukr_dts[ i ]   = ukr_dt;
		ukr_fps[ i ]   = ukr_fp;
		ukr_prefs[ i ] = ukr_pref;
	}

	// The last argument should be the context pointer.
	cntx_t* cntx = ( cntx_t* )va_arg( args, cntx_t* );

	// Shutdown variable argument environment and clean up stack.
	va_end( args );

	// -- End variable argument section --

	// Query the context for the addresses of:
	// - the l3 small/unpacked ukernel func_t array
	// - the l3 small/unpacked ukernel preferences array
	func_t*  cntx_l3_sup_kers       = bli_cntx_l3_sup_kers_buf( cntx );
	mbool_t* cntx_l3_sup_kers_prefs = bli_cntx_l3_sup_kers_prefs_buf( cntx );

	// Now that we have the context address, we want to copy the values
	// from the temporary buffers into the corresponding buffers in the
	// context.

#if 0
	dim_t sup_map[ BLIS_NUM_LEVEL3_SUP_UKRS ][2];

	// Create the small/unpacked ukernel mappings:
	// - rv -> rrr 0, rcr 2
	// - rg -> rrc 1, rcc 3
	// - cv -> ccr 6, ccc 7
	// - cg -> crr 4, crc 5
	// - rd -> rrc 1
	// - cd -> crc 5
	// - rc -> rcc 3
	// - cr -> crr 4
	// - gx -> xxx 8
	// NOTE: We only need to set one slot in the context l3_sup_kers array
	// for the general-stride/generic ukernel type, but since the loop below
	// needs to be set up to set two slots to accommodate the RV, RG, CV, and
	// CG, ukernel types, we will just be okay with the GX ukernel being set
	// redundantly. (The RD, CD, CR, and RC ukernel types are set redundantly
	// for the same reason.)
	sup_map[ BLIS_GEMMSUP_RV_UKR ][0] = BLIS_RRR;
	sup_map[ BLIS_GEMMSUP_RV_UKR ][1] = BLIS_RCR;
	sup_map[ BLIS_GEMMSUP_RG_UKR ][0] = BLIS_RRC;
	sup_map[ BLIS_GEMMSUP_RG_UKR ][1] = BLIS_RCC;
	sup_map[ BLIS_GEMMSUP_CV_UKR ][0] = BLIS_CCR;
	sup_map[ BLIS_GEMMSUP_CV_UKR ][1] = BLIS_CCC;
	sup_map[ BLIS_GEMMSUP_CG_UKR ][0] = BLIS_CRR;
	sup_map[ BLIS_GEMMSUP_CG_UKR ][1] = BLIS_CRC;

	sup_map[ BLIS_GEMMSUP_RD_UKR ][0] = BLIS_RRC;
	sup_map[ BLIS_GEMMSUP_RD_UKR ][1] = BLIS_RRC;
	sup_map[ BLIS_GEMMSUP_CD_UKR ][0] = BLIS_CRC;
	sup_map[ BLIS_GEMMSUP_CD_UKR ][1] = BLIS_CRC;

	sup_map[ BLIS_GEMMSUP_RC_UKR ][0] = BLIS_RCC;
	sup_map[ BLIS_GEMMSUP_RC_UKR ][1] = BLIS_RCC;
	sup_map[ BLIS_GEMMSUP_CR_UKR ][0] = BLIS_CRR;
	sup_map[ BLIS_GEMMSUP_CR_UKR ][1] = BLIS_CRR;

	sup_map[ BLIS_GEMMSUP_GX_UKR ][0] = BLIS_XXX;
	sup_map[ BLIS_GEMMSUP_GX_UKR ][1] = BLIS_XXX;
#endif

	// Process each blocksize id tuple provided.
	for ( i = 0; i < n_ukrs; ++i )
	{
		// Read the current stor3_t id, ukernel datatype, ukernel function
		// pointer, and ukernel preference.
		const stor3_t st3_id   = st3_ids[ i ];
		const num_t   ukr_dt   = ukr_dts[ i ];
		      void*   ukr_fp   = ukr_fps[ i ];
		const bool    ukr_pref = ukr_prefs[ i ];

		// Index to the func_t and mbool_t for the current stor3_t id
		// being processed.
		func_t*  ukrs   = &cntx_l3_sup_kers[ st3_id ];
		mbool_t* prefs  = &cntx_l3_sup_kers_prefs[ st3_id ];

		// Store the ukernel function pointer and preference values into
		// the stor3_t location in the context.
		bli_func_set_dt( ukr_fp, ukr_dt, ukrs );
		bli_mbool_set_dt( ukr_pref, ukr_dt, prefs );
	}

	// Free the temporary local arrays.
	#ifdef BLIS_ENABLE_MEM_TRACING
	printf( "bli_cntx_set_l3_sup_kers(): " );
	#endif
	bli_free_intl( st3_ids );

	#ifdef BLIS_ENABLE_MEM_TRACING
	printf( "bli_cntx_set_l3_sup_kers(): " );
	#endif
	bli_free_intl( ukr_dts );

	#ifdef BLIS_ENABLE_MEM_TRACING
	printf( "bli_cntx_set_l3_sup_kers(): " );
	#endif
	bli_free_intl( ukr_fps );

	#ifdef BLIS_ENABLE_MEM_TRACING
	printf( "bli_cntx_set_l3_sup_kers(): " );
	#endif
	bli_free_intl( ukr_prefs );
}

// -----------------------------------------------------------------------------

void bli_cntx_set_l3_sup_tri_kers( dim_t n_ukrs, ... )
{
	// This function can be called from the bli_cntx_init_*() function for
	// a particular architecture if the kernel developer wishes to use
	// non-default level-3 microkernels for small/unpacked matrices for operations
	// that deal with triangular matrices. It should be called after 
	// bli_cntx_init_defaults() so that the context begins with default sup 
	// micro/millikernels across all datatypes.

	/* Example prototypes:

	   void bli_cntx_set_l3_sup_kers
	   (
	     dim_t   n_ukrs,
	     stor3_t stor_id0, num_t dt0, void* ukr0_fp, bool pref0,
	     stor3_t stor_id1, num_t dt1, void* ukr1_fp, bool pref1,
	     stor3_t stor_id2, num_t dt2, void* ukr2_fp, bool pref2,
	     ...
	     cntx_t* cntx
	   );
	*/

	va_list   args;
	dim_t     i;
	err_t     r_val;

	// Allocate some temporary local arrays.

	#ifdef BLIS_ENABLE_MEM_TRACING
	printf( "bli_cntx_set_l3_sup_tri_kers(): " );
	#endif
	stor3_t* st3_ids   = bli_malloc_intl( n_ukrs * sizeof( stor3_t ), &r_val );

	#ifdef BLIS_ENABLE_MEM_TRACING
	printf( "bli_cntx_set_l3_sup_tri_kers(): " );
	#endif
	num_t*   ukr_dts   = bli_malloc_intl( n_ukrs * sizeof( num_t   ), &r_val );

	#ifdef BLIS_ENABLE_MEM_TRACING
	printf( "bli_cntx_set_l3_sup_tri_kers(): " );
	#endif
	void**   ukr_fps   = bli_malloc_intl( n_ukrs * sizeof( void*   ), &r_val );

	#ifdef BLIS_ENABLE_MEM_TRACING
	printf( "bli_cntx_set_l3_sup_tri_kers(): " );
	#endif
	bool*    ukr_prefs = bli_malloc_intl( n_ukrs * sizeof( bool    ), &r_val );

	// -- Begin variable argument section --

	// Initialize variable argument environment.
	va_start( args, n_ukrs );

	// Process n_ukrs tuples.
	for ( i = 0; i < n_ukrs; ++i )
	{
		// Here, we query the variable argument list for:
		// - the stor3_t storage case being assigned to the kernel we're
		//   about to process,
		// - the datatype of the kernel,
		// - the kernel function pointer, and
		// - the kernel function storage preference
		// that we need to store to the context.
		const stor3_t  st3_id   = ( stor3_t )va_arg( args, stor3_t );
		const num_t    ukr_dt   = ( num_t   )va_arg( args, num_t   );
		      void*    ukr_fp   = ( void*   )va_arg( args, void*   );
		const bool     ukr_pref = ( bool    )va_arg( args, int     );

		// Store the values in our temporary arrays.
		st3_ids[ i ]   = st3_id;
		ukr_dts[ i ]   = ukr_dt;
		ukr_fps[ i ]   = ukr_fp;
		ukr_prefs[ i ] = ukr_pref;
	}

	// The last argument should be the context pointer.
	cntx_t* cntx = ( cntx_t* )va_arg( args, cntx_t* );

	// Shutdown variable argument environment and clean up stack.
	va_end( args );

	// -- End variable argument section --

	// Query the context for the addresses of:
	// - the l3 small/unpacked ukernel func_t array
	// - the l3 small/unpacked ukernel preferences array
	func_t*  cntx_l3_sup_tri_kers       = bli_cntx_l3_sup_tri_kers_buf( cntx );
	mbool_t* cntx_l3_sup_tri_kers_prefs = bli_cntx_l3_sup_tri_kers_prefs_buf( cntx );

	// Process each blocksize id tuple provided.
	for ( i = 0; i < n_ukrs; ++i )
	{
		// Read the current stor3_t id, ukernel datatype, ukernel function
		// pointer, and ukernel preference.
		const stor3_t st3_id   = st3_ids[ i ];
		const num_t   ukr_dt   = ukr_dts[ i ];
		      void*   ukr_fp   = ukr_fps[ i ];
		const bool    ukr_pref = ukr_prefs[ i ];

		// Index to the func_t and mbool_t for the current stor3_t id
		// being processed.
		func_t*  ukrs   = &cntx_l3_sup_tri_kers[ st3_id ];
		mbool_t* prefs  = &cntx_l3_sup_tri_kers_prefs[ st3_id ];

		// Store the ukernel function pointer and preference values into
		// the stor3_t location in the context.
		bli_func_set_dt( ukr_fp, ukr_dt, ukrs );
		bli_mbool_set_dt( ukr_pref, ukr_dt, prefs );
	}

	// Free the temporary local arrays.
	#ifdef BLIS_ENABLE_MEM_TRACING
	printf( "bli_cntx_set_l3_sup_tri_kers(): " );
	#endif
	bli_free_intl( st3_ids );

	#ifdef BLIS_ENABLE_MEM_TRACING
	printf( "bli_cntx_set_l3_sup_tri_kers(): " );
	#endif
	bli_free_intl( ukr_dts );

	#ifdef BLIS_ENABLE_MEM_TRACING
	printf( "bli_cntx_set_l3_sup_tri_kers(): " );
	#endif
	bli_free_intl( ukr_fps );

	#ifdef BLIS_ENABLE_MEM_TRACING
	printf( "bli_cntx_set_l3_sup_tri_kers(): " );
	#endif
	bli_free_intl( ukr_prefs );
}

// -----------------------------------------------------------------------------
void bli_cntx_set_trsm_blkszs( dim_t n_bs, ... )
{
	// This function should be called from the bli_cntx_init_*() function for
	//  zen family architectures to set TRSM blocksizes. It should be called after
	// bli_cntx_init_defaults() so that the context begins with default
	// blocksizes across all datatypes.

	/* Example prototypes:

	   void bli_cntx_set_trsm_blkszs
	   (
	     dim_t   n_bs,
	     bszid_t bs0_id, blksz_t* blksz0,
	     bszid_t bs1_id, blksz_t* blksz1,
	     bszid_t bs2_id, blksz_t* blksz2,
	     ...
	     cntx_t* cntx
	   );
	*/

	va_list   args;
	dim_t     i;
	err_t     r_val;

	// Allocate some temporary local arrays.
	#ifdef BLIS_ENABLE_MEM_TRACING
	printf( "bli_cntx_set_blkszs(): " );
	#endif
	bszid_t* bszids = bli_malloc_intl( n_bs * sizeof( bszid_t  ), &r_val );

	#ifdef BLIS_ENABLE_MEM_TRACING
	printf( "bli_cntx_set_blkszs(): " );
	#endif
	blksz_t** blkszs = bli_malloc_intl( n_bs * sizeof( blksz_t* ), &r_val );

	// -- Begin variable argument section --

	// Initialize variable argument environment.
	va_start( args, n_bs );

	// Process n_bs tuples.
	for ( i = 0; i < n_bs; ++i )
	{
		// Here, we query the variable argument list for:
		// - the bszid_t of the blocksize we're about to process,
		// - the address of the blksz_t object.
		bszid_t  bs_id = ( bszid_t  )va_arg( args, bszid_t  );
		blksz_t* blksz = ( blksz_t* )va_arg( args, blksz_t* );

		// Store the values in our temporary arrays.
		bszids[ i ] = bs_id;
		blkszs[ i ] = blksz;
	}

	// The last argument should be the context pointer.
	cntx_t* cntx = ( cntx_t* )va_arg( args, cntx_t* );

	// Shutdown variable argument environment and clean up stack.
	va_end( args );

	// -- End variable argument section --

	// Query the context for the addresses of:
	// - the blocksize object array
	blksz_t* cntx_l3_trsm_blkszs = bli_cntx_trsm_blkszs_buf( cntx );

	// Now that we have the context address, we want to copy the values
	// from the temporary buffers into the corresponding buffers in the
	// context. Notice that the blksz_t* pointers were saved, rather than
	// the objects themselves, but we copy the contents of the objects
	// when copying into the context.

	// Process each blocksize id tuple provided.
	for ( i = 0; i < n_bs; ++i )
	{
		// Read the current blocksize id, blksz_t* pointer, blocksize
		// multiple id, and blocksize scalar.
		bszid_t  bs_id = bszids[ i ];
		blksz_t* blksz = blkszs[ i ];

		blksz_t* cntx_l3_trsm_blksz = &cntx_l3_trsm_blkszs[ bs_id ];

		// Copy the blksz_t object contents into the appropriate
		// location within the context's blksz_t array.
		//cntx_trsm_blkszs[ bs_id ] = *blksz;
		//bli_blksz_copy( blksz, cntx_trsm_blksz );
		bli_blksz_copy_if_pos( blksz, cntx_l3_trsm_blksz );
	}

	// Free the temporary local arrays.

	#ifdef BLIS_ENABLE_MEM_TRACING
	printf( "bli_cntx_set_blkszs(): " );
	#endif
	bli_free_intl( blkszs );

	#ifdef BLIS_ENABLE_MEM_TRACING
	printf( "bli_cntx_set_blkszs(): " );
	#endif
	bli_free_intl( bszids );
}

// -----------------------------------------------------------------------------

void bli_cntx_set_l1f_kers( dim_t n_kers, ... )
{
	// This function can be called from the bli_cntx_init_*() function for
	// a particular architecture if the kernel developer wishes to use
	// non-default level-1f kernels. It should be called after
	// bli_cntx_init_defaults() so that the context begins with default l1f
	// kernels across all datatypes.

	/* Example prototypes:

	   void bli_cntx_set_l1f_kers
	   (
	     dim_t   n_ukrs,
	     l1fkr_t ker0_id, num_t ker0_dt, void_fp ker0_fp,
	     l1fkr_t ker1_id, num_t ker1_dt, void_fp ker1_fp,
	     l1fkr_t ker2_id, num_t ker2_dt, void_fp ker2_fp,
	     ...
	     cntx_t* cntx
	   );
	*/

	va_list   args;
	dim_t     i;
	err_t     r_val;

	// Allocate some temporary local arrays.

	#ifdef BLIS_ENABLE_MEM_TRACING
	printf( "bli_cntx_set_l1f_kers(): " );
	#endif
	l1fkr_t* ker_ids   = bli_malloc_intl( n_kers * sizeof( l1fkr_t ), &r_val );

	#ifdef BLIS_ENABLE_MEM_TRACING
	printf( "bli_cntx_set_l1f_kers(): " );
	#endif
	num_t*   ker_dts   = bli_malloc_intl( n_kers * sizeof( num_t   ), &r_val );

	#ifdef BLIS_ENABLE_MEM_TRACING
	printf( "bli_cntx_set_l1f_kers(): " );
	#endif
	void_fp* ker_fps   = bli_malloc_intl( n_kers * sizeof( void_fp ), &r_val );

	// -- Begin variable argument section --

		// If we find a ukernel pref id of BLIS_VA_END, then we are done.
		if ( ukr_pref_id0 == BLIS_VA_END ) break;

		// Here, we query the variable argument list for:
		// - the ukr_t of the kernel we're about to process (already done),
		// - the datatype of the kernel, and
		// - the kernel function pointer
		const ukr_pref_t ukr_pref_id = ( ukr_pref_t )ukr_pref_id0;
		const num_t      ukr_pref_dt = ( num_t      )va_arg( args, num_t );
		const bool       ukr_pref    = ( bool       )va_arg( args, int );

		// Index into the func_t and mbool_t for the current kernel id
		// being processed.
		mbool_t* ukr_prefs = &cntx_ukr_prefs[ ukr_pref_id ];

		// Store the ukernel preference value into the context.
		bli_mbool_set_dt( ukr_pref, ukr_pref_dt, ukr_prefs );
	}

	// Shutdown variable argument environment and clean up stack.
	va_end( args );
}

// -----------------------------------------------------------------------------

void bli_cntx_set_l3_thresh_funcs( dim_t n_funcs, ... )
{
	// This function can be called from the bli_cntx_init_*() function for
	// a particular architecture if the kernel developer wishes to use
	// non-default level-3 threshold functions. It should be called after
	// bli_cntx_init_defaults() so that the context begins with default
	// functionss across all operations.

	/* Example prototypes:

	   void bli_cntx_set_l3_thresh_funcs
	   (
	     dim_t   n_funcs,
	     opid_t op1_id, void_fp ker0_fp,
	     opid_t op2_id, void_fp ker1_fp,
	     opid_t op2_id, void_fp ker2_fp,
	     ...
	     cntx_t* cntx
	   );
	*/

	va_list   args;
	dim_t     i;
	err_t     r_val;

	// Allocate some temporary local arrays.

	#ifdef BLIS_ENABLE_MEM_TRACING
	printf( "bli_cntx_set_l3_thresh_funcs(): " );
	#endif
	opid_t* func_ids   = bli_malloc_intl( n_funcs * sizeof( opid_t ), &r_val );

	#ifdef BLIS_ENABLE_MEM_TRACING
	printf( "bli_cntx_set_l3_thresh_funcs(): " );
	#endif
	void_fp* func_fps   = bli_malloc_intl( n_funcs * sizeof( void_fp ), &r_val );

	// -- Begin variable argument section --

	// Initialize variable argument environment.
	va_start( args, n_funcs );

	// Process n_funcs tuples.
	for ( i = 0; i < n_funcs; ++i )
	{
		// Here, we query the variable argument list for:
		// - the opid_t of the function we're about to process,
		// - the function pointer
		// that we need to store to the context.
		const opid_t  op_id   = ( opid_t )va_arg( args, opid_t );
		      void_fp  func_fp   = ( void_fp )va_arg( args, void_fp );

		// Store the values in our temporary arrays.
		func_ids[ i ]   = op_id;
		func_fps[ i ]   = func_fp;
	}

	// The last argument should be the context pointer.
	cntx_t* cntx = ( cntx_t* )va_arg( args, cntx_t* );

	// Shutdown variable argument environment and clean up stack.
	va_end( args );

	// -- End variable argument section --

	// Query the context for the address of:
	// - the level-3 threshold func array
	void_fp* cntx_l3_thresh_funcs = bli_cntx_l3_thresh_funcs_buf( cntx );

	// Now that we have the context address, we want to copy the values
	// from the temporary buffers into the corresponding buffers in the
	// context.

	// Process each blocksize id tuple provided.
	for ( i = 0; i < n_funcs; ++i )
	{
		// Read the current func id, and function pointer.
		const opid_t  func_id   = func_ids[ i ];
		      void_fp func_fp   = func_fps[ i ];

		// Store function pointer in cntx
		cntx_l3_thresh_funcs[ func_id ] = func_fp;

	}

	// Free the temporary local arrays.

	#ifdef BLIS_ENABLE_MEM_TRACING
	printf( "bli_cntx_set_l3_thresh_funcs(): " );
	#endif
	bli_free_intl( func_ids );

	#ifdef BLIS_ENABLE_MEM_TRACING
	printf( "bli_cntx_set_l3_thresh_funcs(): " );
	#endif
	bli_free_intl( func_fps );
}

// -----------------------------------------------------------------------------

void bli_cntx_set_packm_kers( dim_t n_kers, ... )
{
	// This function can be called from the bli_cntx_init_*() function for
	// a particular architecture if the kernel developer wishes to use
	// non-default level-3 operation handler for small/unpacked matrices. It
	// should be called after bli_cntx_init_<subconfig>_ref() so that the
	// context begins with default sup handlers across all datatypes.

	/* Example prototypes:

	   void bli_cntx_set_l3_sup_handlers
	   (
	     cntx_t* cntx
	     opid_t  op0_id, void_fp handler0_fp,
	     opid_t  op1_id, void_fp handler1_fp,
	     opid_t  op2_id, void_fp handler2_fp,
	     ...,
	     BLIS_VA_END
	   );
	*/

	// Query the context for the address of the l3 sup handlers array.
	void_fp* cntx_l3_sup_handlers = cntx->l3_sup_handlers;

	// Initialize variable argument environment.
	va_list   args;
	va_start( args, cntx );

	// Process sup handlers until BLIS_VA_END is reached.
	while ( true )
	{
		const int op_id0 = va_arg( args, int );

		// If we find an operation id of BLIS_VA_END, then we are done.
		if ( op_id0 == BLIS_VA_END ) break;

		// Here, we query the variable argument list for:
		// - the opid_t of the operation we're about to process,
		// - the sup handler function pointer
		const opid_t  op_id = ( opid_t  )op_id0;
		      void_fp op_fp = ( void_fp )va_arg( args, void_fp );

		// Store the sup handler function pointer into the slot for the
		// specified operation id.
		cntx_l3_sup_handlers[ op_id ] = op_fp;
	}

	// Shutdown variable argument environment and clean up stack.
	va_end( args );
}

// -----------------------------------------------------------------------------

void bli_cntx_print( const cntx_t* cntx )
{
	dim_t i;

	// Print the values stored in the blksz_t objects.
	printf( "                               s                d                c                z\n" );

	for ( i = 0; i < BLIS_NUM_BLKSZS; ++i )
	{
		printf( "blksz/mult %2lu:  %13lu/%2lu %13lu/%2lu %13lu/%2lu %13lu/%2lu\n",
		         ( unsigned long )i,
		         ( unsigned long )bli_cntx_get_blksz_def_dt( BLIS_FLOAT,    i, cntx ),
		         ( unsigned long )bli_cntx_get_bmult_dt    ( BLIS_FLOAT,    i, cntx ),
		         ( unsigned long )bli_cntx_get_blksz_def_dt( BLIS_DOUBLE,   i, cntx ),
		         ( unsigned long )bli_cntx_get_bmult_dt    ( BLIS_DOUBLE,   i, cntx ),
		         ( unsigned long )bli_cntx_get_blksz_def_dt( BLIS_SCOMPLEX, i, cntx ),
		         ( unsigned long )bli_cntx_get_bmult_dt    ( BLIS_SCOMPLEX, i, cntx ),
		         ( unsigned long )bli_cntx_get_blksz_def_dt( BLIS_DCOMPLEX, i, cntx ),
		         ( unsigned long )bli_cntx_get_bmult_dt    ( BLIS_DCOMPLEX, i, cntx )
		      );
	}

	for ( i = 0; i < BLIS_NUM_UKRS; ++i )
	{
		const func_t* ukr = bli_cntx_get_ukrs( i, cntx );

		printf( "ukr %2lu:  %16p %16p %16p %16p\n",
		        ( unsigned long )i,
		        bli_func_get_dt( BLIS_FLOAT,    ukr ),
		        bli_func_get_dt( BLIS_DOUBLE,   ukr ),
		        bli_func_get_dt( BLIS_SCOMPLEX, ukr ),
		        bli_func_get_dt( BLIS_DCOMPLEX, ukr )
		      );
	}

	for ( i = 0; i < BLIS_NUM_UKR_PREFS; ++i )
	{
		const mbool_t* ukr_pref = bli_cntx_get_ukr_prefs( i, cntx );

		printf( "ukr pref %2lu:  %d %d %d %d\n",
		        ( unsigned long )i,
		        bli_mbool_get_dt( BLIS_FLOAT,    ukr_pref ),
		        bli_mbool_get_dt( BLIS_DOUBLE,   ukr_pref ),
		        bli_mbool_get_dt( BLIS_SCOMPLEX, ukr_pref ),
		        bli_mbool_get_dt( BLIS_DCOMPLEX, ukr_pref )
		      );
	}

	{
		ind_t method = bli_cntx_method( cntx );

		printf( "ind method   : %lu\n", ( unsigned long )method );
	}
}

