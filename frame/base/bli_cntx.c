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
   THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
   (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
   OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

*/

#include "blis.h"

#if 0
//
// NOTE: Since these functions currently do nothing, they are defined
// as empty macros in bli_cntx.
//
void bli_cntx_obj_create( cntx_t* cntx )
{
	// Since cntx_t objects contain statically-allocated arrays,
	// we don't need to do anything in order to create the cntx_t
	// instance.
}

void bli_cntx_obj_free( cntx_t* cntx )
{
	// Just as we don't need to do anything in order to create a
	// cntx_t instance, we don't need to do anything to destory
	// one.
}
#endif

void bli_cntx_obj_clear( cntx_t* cntx )
{
	// Fill the entire cntx_t structure with zeros.
	memset( ( void* )cntx, 0, sizeof( cntx ) );
}

void bli_cntx_init( cntx_t* cntx )
{
	// This function initializes a "universal" context that is pre-loaded
	// with kernel addresses for all level-1v, -1f, and -3 kernels, in
	// addition to all level-1f and -3 blocksizes.

	bli_gks_cntx_set_l3_nat_ukr( BLIS_GEMM_UKR, cntx );
	bli_gks_cntx_set_l3_nat_ukr( BLIS_GEMMTRSM_L_UKR, cntx );
	bli_gks_cntx_set_l3_nat_ukr( BLIS_GEMMTRSM_U_UKR, cntx );
	bli_gks_cntx_set_l3_nat_ukr( BLIS_TRSM_L_UKR, cntx );
	bli_gks_cntx_set_l3_nat_ukr( BLIS_TRSM_U_UKR, cntx );

	bli_gks_cntx_set_blkszs( BLIS_NAT, 6,
	                         BLIS_NC, BLIS_NR,
	                         BLIS_KC, BLIS_KR,
	                         BLIS_MC, BLIS_KR,
	                         BLIS_NR, BLIS_NR,
	                         BLIS_MR, BLIS_MR,
	                         BLIS_KR, BLIS_KR,
	                         cntx );

	bli_gks_cntx_set_l1f_ker( BLIS_AXPY2V_KER, cntx );
	bli_gks_cntx_set_l1f_ker( BLIS_DOTAXPYV_KER, cntx );
	bli_gks_cntx_set_l1f_ker( BLIS_AXPYF_KER, cntx );
	bli_gks_cntx_set_l1f_ker( BLIS_DOTXF_KER, cntx );
	bli_gks_cntx_set_l1f_ker( BLIS_DOTXAXPYF_KER, cntx );

	bli_gks_cntx_set_blkszs( BLIS_NAT, 3,
	                         BLIS_AF, BLIS_AF,
	                         BLIS_DF, BLIS_DF,
	                         BLIS_XF, BLIS_XF,
	                         cntx );

	bli_gks_cntx_set_l1v_ker( BLIS_ADDV_KER, cntx );
	bli_gks_cntx_set_l1v_ker( BLIS_AXPYV_KER, cntx );
	bli_gks_cntx_set_l1v_ker( BLIS_COPYV_KER, cntx );
	bli_gks_cntx_set_l1v_ker( BLIS_DOTV_KER, cntx );
	bli_gks_cntx_set_l1v_ker( BLIS_DOTXV_KER, cntx );
	bli_gks_cntx_set_l1v_ker( BLIS_INVERTV_KER, cntx );
	bli_gks_cntx_set_l1v_ker( BLIS_SCALV_KER, cntx );
	bli_gks_cntx_set_l1v_ker( BLIS_SCAL2V_KER, cntx );
	bli_gks_cntx_set_l1v_ker( BLIS_SETV_KER, cntx );
	bli_gks_cntx_set_l1v_ker( BLIS_SUBV_KER, cntx );
	bli_gks_cntx_set_l1v_ker( BLIS_SWAPV_KER, cntx );
}

// -----------------------------------------------------------------------------

blksz_t* bli_cntx_get_blksz( bszid_t bs_id,
                             cntx_t* cntx )
{
	blksz_t* blkszs = bli_cntx_blkszs_buf( cntx );
	blksz_t* blksz  = &blkszs[ bs_id ];

	// Return the address of the blksz_t identified by bs_id.
	return blksz;
}

#if 0
dim_t bli_cntx_get_blksz_def_dt( num_t   dt,
                                 bszid_t bs_id,
                                 cntx_t* cntx )
{
	blksz_t* blkszs = bli_cntx_blkszs_buf( cntx );
	blksz_t* blksz  = &blkszs[ bs_id ];

	// Return the default blocksize value for the datatype given.
	return bli_blksz_get_def( dt, blksz );
}

dim_t bli_cntx_get_blksz_max_dt( num_t   dt,
                                 bszid_t bs_id,
                                 cntx_t* cntx )
{
	blksz_t* blkszs = bli_cntx_blkszs_buf( cntx );
	blksz_t* blksz  = &blkszs[ bs_id ];

	// Return the default blocksize value for the datatype given.
	return bli_blksz_get_max( dt, blksz );
}
#endif

blksz_t* bli_cntx_get_bmult( bszid_t bs_id,
                             cntx_t* cntx )
{
	blksz_t* blkszs = bli_cntx_blkszs_buf( cntx );
	bszid_t* bmults = bli_cntx_bmults_buf( cntx );
	bszid_t  bm_id  = bmults[ bs_id ];
	blksz_t* bmult  = &blkszs[ bm_id ];

	// Return the address of the blksz_t identified by the multiple for
	// the blocksize corresponding to bs_id.
	return bmult;
}

#if 0
dim_t bli_cntx_get_bmult_dt( num_t   dt,
                             bszid_t bs_id,
                             cntx_t* cntx )
{
	blksz_t* bmult = bli_cntx_get_bmult( bs_id, cntx );

	return bli_blksz_get_def( dt, bmult );
}
#endif

func_t* bli_cntx_get_l3_ukr( l3ukr_t ukr_id,
                             cntx_t* cntx )
{
	func_t* l3_vir_ukrs = bli_cntx_l3_vir_ukrs_buf( cntx );
	func_t* l3_nat_ukrs = bli_cntx_l3_nat_ukrs_buf( cntx );
	func_t* l3_ukrs;
	func_t* l3_ukr;

	// If the context was set up for non-native (ie: induced) execution,
	// the virtual ukernel func_t's will contain the appropriate function
	// pointers. Otherwise, we use the native ukernel func_t's.
	if ( bli_cntx_method( cntx ) != BLIS_NAT ) l3_ukrs = l3_vir_ukrs;
	else                                       l3_ukrs = l3_nat_ukrs;

	// Index into the func_t array chosen above using the ukr_id.
	l3_ukr = &l3_ukrs[ ukr_id ];

	// Return the address of the func_t identified by ukr_id.
	return l3_ukr;
}

#if 0
void* bli_cntx_get_l3_ukr_dt( num_t   dt,
                              l3ukr_t ukr_id,
                              cntx_t* cntx )
{
	func_t* l3_vir_ukrs = bli_cntx_l3_vir_ukrs_buf( cntx );
	func_t* l3_nat_ukrs = bli_cntx_l3_nat_ukrs_buf( cntx );
	func_t* l3_ukrs;
	func_t* l3_ukr;

	// If the context was set up for non-native (ie: induced) execution,
	// the virtual ukernel func_t's will contain the appropriate function
	// pointers. Otherwise, we use the native ukernel func_t's.
	if ( bli_cntx_method( cntx ) != BLIS_NAT ) l3_ukrs = l3_vir_ukrs;
	else                                       l3_ukrs = l3_nat_ukrs;

	// Index into the func_t array chosen above using the ukr_id.
	l3_ukr = &l3_ukrs[ ukr_id ];

	return bli_func_get_dt( dt, l3_ukr );
}
#endif

func_t* bli_cntx_get_l3_vir_ukr( l3ukr_t ukr_id,
                                 cntx_t* cntx )
{
	func_t* l3_vir_ukrs = bli_cntx_l3_vir_ukrs_buf( cntx );
	func_t* l3_vir_ukr  = &l3_vir_ukrs[ ukr_id ];

	// Return the address of the virtual level-3 micro-kernel func_t
	// identified by ukr_id.
	return l3_vir_ukr;
}

#if 0
void* bli_cntx_get_l3_vir_ukr_dt( num_t   dt,
                                  l3ukr_t ukr_id,
                                  cntx_t* cntx )
{
	func_t* l3_vir_ukrs = bli_cntx_l3_vir_ukrs_buf( cntx );
	func_t* l3_vir_ukr  = &l3_vir_ukrs[ ukr_id ];

	// Return the address of the virtual level-3 micro-kernel func_t
	// identified by ukr_id.
	return bli_func_get_dt( dt, l3_vir_ukr );
}
#endif

func_t* bli_cntx_get_l3_nat_ukr( l3ukr_t ukr_id,
                                 cntx_t* cntx )
{
	func_t* l3_nat_ukrs = bli_cntx_l3_nat_ukrs_buf( cntx );
	func_t* l3_nat_ukr  = &l3_nat_ukrs[ ukr_id ];

	// Return the address of the native level-3 micro-kernel func_t
	// identified by ukr_id.
	return l3_nat_ukr;
}

#if 0
void* bli_cntx_get_l3_nat_ukr_dt( num_t   dt,
                                  l3ukr_t ukr_id,
                                  cntx_t* cntx )
{
	func_t* l3_nat_ukrs = bli_cntx_l3_nat_ukrs_buf( cntx );
	func_t* l3_nat_ukr  = &l3_nat_ukrs[ ukr_id ];

	// Return the address of the native level-3 micro-kernel func_t
	// identified by ukr_id.
	return bli_func_get_dt( dt, l3_nat_ukr );
}
#endif

func_t* bli_cntx_get_l1f_ker( l1fkr_t ker_id,
                              cntx_t* cntx )
{
	func_t* l1f_kers = bli_cntx_l1f_kers_buf( cntx );
	func_t* l1f_ker  = &l1f_kers[ ker_id ];

	// Return the address of the level-1f kernel func_t identified by
	// ker_id.
	return l1f_ker;
}

#if 0
void* bli_cntx_get_l1f_ker_dt( num_t   dt,
                               l1fkr_t ker_id,
                               cntx_t* cntx )
{
	func_t* l1f_kers = bli_cntx_l1f_kers_buf( cntx );
	func_t* l1f_ker  = &l1f_kers[ ker_id ];

	return bli_func_get_dt( dt, l1f_ker );
}
#endif

func_t* bli_cntx_get_l1v_ker( l1vkr_t ker_id,
                              cntx_t* cntx )
{
	func_t* l1v_kers = bli_cntx_l1v_kers_buf( cntx );
	func_t* l1v_ker  = &l1v_kers[ ker_id ];

	// Return the address of the level-1v kernel func_t identified by
	// ker_id.
	return l1v_ker;
}

#if 0
void* bli_cntx_get_l1v_ker_dt( num_t   dt,
                               l1vkr_t ker_id,
                               cntx_t* cntx )
{
	func_t* l1v_kers = bli_cntx_l1v_kers_buf( cntx );
	func_t* l1v_ker  = &l1v_kers[ ker_id ];

	return bli_func_get_dt( dt, l1v_ker );
}
#endif

mbool_t* bli_cntx_get_l3_nat_ukr_prefs( l3ukr_t ukr_id,
                                        cntx_t* cntx )
{
	mbool_t* l3_nat_ukrs_prefs = bli_cntx_l3_nat_ukrs_prefs_buf( cntx );
	mbool_t* l3_nat_ukrs_pref  = &l3_nat_ukrs_prefs[ ukr_id ];

	// Return the address of the native kernel func_t identified by ukr_id.
	return l3_nat_ukrs_pref;
}

func_t* bli_cntx_get_packm_ukr( cntx_t* cntx )
{
	func_t* packm_ukrs = bli_cntx_packm_ukrs( cntx );

	// Return the address of the func_t that contains the packm ukernels.
	return packm_ukrs;
}

#if 0
ind_t bli_cntx_get_ind_method( cntx_t* cntx )
{
	return bli_cntx_method( cntx );
}

pack_t bli_cntx_get_pack_schema_a( cntx_t* cntx )
{
	return bli_cntx_schema_a( cntx );
}

pack_t bli_cntx_get_pack_schema_b( cntx_t* cntx )
{
	return bli_cntx_schema_b( cntx );
}
#endif

// -----------------------------------------------------------------------------

#if 1
//
// NOTE: This function is disabled because:
// - we currently do not have any need to set a context direclty with
//   blksz_t objects
// - it may be broken; it needs to be synced up with the corresponding
//   function in bli_gks.c.
//
void bli_cntx_set_blkszs( ind_t method, dim_t n_bs, ... )
{
	/* Example prototypes:

	   void
	   bli_cntx_set_blkszs(

	             ind_t   method = BLIS_NAT,
	             dim_t   n_bs,
	             bszid_t bs0_id, blksz_t* blksz0, bszid_t bm0_id,
	             bszid_t bs1_id, blksz_t* blksz1, bszid_t bm1_id,
	             bszid_t bs2_id, blksz_t* blksz2, bszid_t bm2_id,
	             ...
	             cntx_t* cntx );

	   void
	   bli_cntx_set_blkszs(

	             ind_t   method != BLIS_NAT,
	             dim_t   n_bs,
	             bszid_t bs0_id, blksz_t* blksz0, bszid_t bm0_id, dim_t scalr0,
	             bszid_t bs1_id, blksz_t* blksz1, bszid_t bm1_id, dim_t scalr1,
	             bszid_t bs2_id, blksz_t* blksz2, bszid_t bm2_id, dim_t scalr2,
	             ...
	             cntx_t* cntx );
	*/
	va_list   args;
	dim_t     i;

	bszid_t*  bszids;
	blksz_t** blkszs;
	bszid_t*  bmults;
	dim_t*    scalrs;

	cntx_t*   cntx;

	blksz_t*  cntx_blkszs;
	bszid_t*  cntx_bmults;


	// Allocate some temporary local arrays.
	bszids = bli_malloc_intl( n_bs * sizeof( bszid_t  ) );
	blkszs = bli_malloc_intl( n_bs * sizeof( blksz_t* ) );
	bmults = bli_malloc_intl( n_bs * sizeof( bszid_t  ) );
	scalrs = bli_malloc_intl( n_bs * sizeof( dim_t    ) );

	// -- Begin variable argument section --

	// Initialize variable argument environment.
	va_start( args, n_bs );

	// Handle native and induced method cases separately.
	if ( method == BLIS_NAT )
	{
		// Process n_bs tuples.
		for ( i = 0; i < n_bs; ++i )
		{
			// Here, we query the variable argument list for:
			// - the bszid_t of the blocksize we're about to process,
			// - the address of the blksz_t object, and
			// - the bszid_t of the multiple we need to associate with
			//   the blksz_t object.
			const bszid_t  bs_id = va_arg( args, bszid_t  );
			      blksz_t* blksz = va_arg( args, blksz_t* );
			const bszid_t  bm_id = va_arg( args, bszid_t  );

			// Store the values in our temporary arrays.
			bszids[ i ] = bs_id;
			blkszs[ i ] = blksz;
			bmults[ i ] = bm_id;
		}
	}
	else // if induced method execution was indicated
	{
		// Process n_bs tuples.
		for ( i = 0; i < n_bs; ++i )
		{
			// Here, we query the variable argument list for:
			// - the bszid_t of the blocksize we're about to process,
			// - the address of the blksz_t object, and
			// - the bszid_t of the multiple we need to associate with
			//   the blksz_t object.
			// - the scalar we wish to apply to the real blocksizes to
			//   come up with the induced complex blocksizes.
			const bszid_t  bs_id = va_arg( args, bszid_t  );
			      blksz_t* blksz = va_arg( args, blksz_t* );
			const bszid_t  bm_id = va_arg( args, bszid_t  );
			const dim_t    scalr = va_arg( args, dim_t    );

			// Store the values in our temporary arrays.
			bszids[ i ] = bs_id;
			blkszs[ i ] = blksz;
			bmults[ i ] = bm_id;
			scalrs[ i ] = scalr;
		}
	}

	// The last argument should be the context pointer.
	cntx = va_arg( args, cntx_t* );

	// Shutdown variable argument environment and clean up stack.
	va_end( args );

	// -- End variable argument section --

	// Save the execution type into the context.
	bli_cntx_set_method( method, cntx );

	// Query the context for the addresses of:
	// - the blocksize object array
	// - the blocksize multiple array
	cntx_blkszs = bli_cntx_blkszs_buf( cntx );
	cntx_bmults = bli_cntx_bmults_buf( cntx );

	// Now that we have the context address, we want to copy the values
	// from the temporary buffers into the corresponding buffers in the
	// context. Notice that the blksz_t* pointers were saved, rather than
	// the objects themselves, but we copy the contents of the objects
	// when copying into the context.

	// Handle native and induced method cases separately.
	if ( method == BLIS_NAT )
	{
		// Process each blocksize id tuple provided.
		for ( i = 0; i < n_bs; ++i )
		{
			// Read the current blocksize id, blksz_t* pointer, blocksize
			// multiple id, and blocksize scalar.
			const bszid_t  bs_id = bszids[ i ];
			const bszid_t  bm_id = bmults[ i ];

			      blksz_t* blksz = blkszs[ i ];

			      blksz_t* cntx_blksz = &cntx_blkszs[ bs_id ];

			// Copy the blksz_t object contents into the appropriate
			// location within the context's blksz_t array. Do the same
			// for the blocksize multiple id.
			//cntx_blkszs[ bs_id ] = *blksz;
			bli_blksz_copy( blksz, cntx_blksz );

			// Copy the blocksize multiple id into the context.
			cntx_bmults[ bs_id ] = bm_id;
		}
	}
	else
	{
		// Process each blocksize id tuple provided.
		for ( i = 0; i < n_bs; ++i )
		{
			// Read the current blocksize id, blksz_t pointer, blocksize
			// multiple id, and blocksize scalar.
			const bszid_t  bs_id = bszids[ i ];
			const bszid_t  bm_id = bmults[ i ];
			const dim_t    scalr = scalrs[ i ];

			      blksz_t* blksz = blkszs[ i ];
			      blksz_t* bmult = blkszs[ i ];

			      blksz_t* cntx_blksz = &cntx_blkszs[ bs_id ];

			// Copy the real domain values of the source blksz_t object into
			// the context, duplicating into the complex domain fields.
			bli_blksz_copy_dt( BLIS_FLOAT,  blksz, BLIS_FLOAT,    cntx_blksz );
			bli_blksz_copy_dt( BLIS_DOUBLE, blksz, BLIS_DOUBLE,   cntx_blksz );
			bli_blksz_copy_dt( BLIS_FLOAT,  blksz, BLIS_SCOMPLEX, cntx_blksz );
			bli_blksz_copy_dt( BLIS_DOUBLE, blksz, BLIS_DCOMPLEX, cntx_blksz );

			// The next steps apply only to cache blocksizes, and not register
			// blocksizes (ie: they only apply to blocksizes for which the
			// blocksize multiple id is different than the blocksize id) and
			// only when the scalar provided is non-unit.
			if ( bs_id != bm_id && scalr != 1 ) 
			{
				// Scale the complex domain values in the blocksize object.
				bli_blksz_scale_dt_by( 1, scalr, BLIS_SCOMPLEX, cntx_blksz );
				bli_blksz_scale_dt_by( 1, scalr, BLIS_DCOMPLEX, cntx_blksz );

				// Finally, round the newly-scaled blocksizes down to their
				// respective multiples.
				bli_blksz_reduce_dt_to( BLIS_FLOAT,  bmult, BLIS_SCOMPLEX, cntx_blksz );
				bli_blksz_reduce_dt_to( BLIS_DOUBLE, bmult, BLIS_DCOMPLEX, cntx_blksz );
			}

			// Copy the blocksize multiple id into the context.
			cntx_bmults[ bs_id ] = bm_id;
		}
	}

	// Free the temporary local arrays.
	bli_free_intl( blkszs );
	bli_free_intl( bszids );
	bli_free_intl( bmults );
	bli_free_intl( scalrs );
}
#endif

// -----------------------------------------------------------------------------

void bli_cntx_set_blksz( bszid_t  bs_id,
                         blksz_t* blksz,
                         bszid_t  mult_id,
                         cntx_t*  cntx )
{
	blksz_t* blkszs = bli_cntx_blkszs_buf( cntx );
	bszid_t* bmults = bli_cntx_bmults_buf( cntx );

	// Copy the blocksize object into the specified location within
	// the context's blocksize array.
	blkszs[ bs_id ] = *blksz;

	// Assign the blocksize multiple id to the corresponding location
	// in the context's blocksize multiple array.
	bmults[ bs_id ] = mult_id;
}

void bli_cntx_set_l3_vir_ukr( l3ukr_t ukr_id,
                              func_t* func,
                              cntx_t* cntx )
{
	func_t* l3_vir_ukrs = bli_cntx_l3_vir_ukrs_buf( cntx );

	// Copy the function object into the specified location within
	// the context's virtual level-3 ukernel array.
	l3_vir_ukrs[ ukr_id ] = *func;
}

void bli_cntx_set_l3_nat_ukr( l3ukr_t ukr_id,
                              func_t* func,
                              cntx_t* cntx )
{
	func_t* l3_nat_ukrs = bli_cntx_l3_nat_ukrs_buf( cntx );

	// Copy the function object into the specified location within
	// the context's native level-3 ukernel array.
	l3_nat_ukrs[ ukr_id ] = *func;
}

void bli_cntx_set_l3_nat_ukr_prefs( l3ukr_t  ukr_id,
                                    mbool_t* prefs,
                                    cntx_t*  cntx )
{
	mbool_t* l3_nat_ukrs_prefs = bli_cntx_l3_nat_ukrs_prefs_buf( cntx );

	// Copy the mbool_t into the specified location within
	// the context's native level-3 ukernel preference array.
	l3_nat_ukrs_prefs[ ukr_id ] = *prefs;
}

void bli_cntx_set_l1f_ker( l1fkr_t ker_id,
                           func_t* func,
                           cntx_t* cntx )
{
	func_t* l1f_kers = bli_cntx_l1f_kers_buf( cntx );

	// Copy the function object into the specified location within
	// the context's level-1f kernel array.
	l1f_kers[ ker_id ] = *func;
}

void bli_cntx_set_l1v_ker( l1vkr_t ker_id,
                           func_t* func,
                           cntx_t* cntx )
{
	func_t* l1v_kers = bli_cntx_l1v_kers_buf( cntx );

	// Copy the function object into the specified location within
	// the context's level-1v kernel array.
	l1v_kers[ ker_id ] = *func;
}

void bli_cntx_set_packm_ukr( func_t* func,
                             cntx_t* cntx )
{
	func_t* packm_ukrs = bli_cntx_packm_ukrs( cntx );

	// Copy the function object into the context's packm ukernel object.
	*packm_ukrs = *func;
}

void bli_cntx_set_ind_method( ind_t   method,
                              cntx_t* cntx )
{
	bli_cntx_set_method( method, cntx );
}

void bli_cntx_set_pack_schema_ab( pack_t  schema_a,
                                  pack_t  schema_b,
                                  cntx_t* cntx )
{
	bli_cntx_set_schema_a( schema_a, cntx );
	bli_cntx_set_schema_b( schema_b, cntx );
}

void bli_cntx_set_pack_schema_a( pack_t  schema_a,
                                 cntx_t* cntx )
{
	bli_cntx_set_schema_a( schema_a, cntx );
}

void bli_cntx_set_pack_schema_b( pack_t  schema_b,
                                 cntx_t* cntx )
{
	bli_cntx_set_schema_b( schema_b, cntx );
}

void bli_cntx_set_pack_schema_c( pack_t  schema_c,
                                 cntx_t* cntx )
{
	bli_cntx_set_schema_c( schema_c, cntx );
}

// -----------------------------------------------------------------------------

bool_t bli_cntx_l3_nat_ukr_prefers_rows_dt( num_t   dt,
                                            l3ukr_t ukr_id,
                                            cntx_t* cntx )
{
	mbool_t* ukrs_prefs = bli_cntx_get_l3_nat_ukr_prefs( ukr_id, cntx );
	bool_t   ukr_prefs  = bli_mbool_get_dt( dt, ukrs_prefs );

	// A ukernel preference of TRUE means the ukernel prefers row
	// storage.
	return ukr_prefs == TRUE;
}

bool_t bli_cntx_l3_nat_ukr_prefers_cols_dt( num_t   dt,
                                            l3ukr_t ukr_id,
                                            cntx_t* cntx )
{
	mbool_t* ukrs_prefs = bli_cntx_get_l3_nat_ukr_prefs( ukr_id, cntx );
	bool_t   ukr_prefs  = bli_mbool_get_dt( dt, ukrs_prefs );

	// A ukernel preference of FALSE means the ukernel prefers column
	// storage.
	return ukr_prefs == FALSE;
}

bool_t bli_cntx_l3_nat_ukr_prefers_storage_of( obj_t*  obj,
                                               l3ukr_t ukr_id,
                                               cntx_t* cntx )
{
	return !bli_cntx_l3_nat_ukr_dislikes_storage_of( obj, ukr_id, cntx );
}

bool_t bli_cntx_l3_nat_ukr_dislikes_storage_of( obj_t*  obj,
                                                l3ukr_t ukr_id,
                                                cntx_t* cntx )
{
	const num_t  dt    = bli_obj_datatype( *obj );
	const bool_t ukr_prefers_rows
	                   = bli_cntx_l3_nat_ukr_prefers_rows_dt( dt, ukr_id, cntx );
	const bool_t ukr_prefers_cols
	                   = bli_cntx_l3_nat_ukr_prefers_cols_dt( dt, ukr_id, cntx );
	bool_t       r_val = FALSE;

	if      ( bli_obj_is_row_stored( *obj ) && ukr_prefers_cols ) r_val = TRUE;
	else if ( bli_obj_is_col_stored( *obj ) && ukr_prefers_rows ) r_val = TRUE;

	return r_val;
}

// -----------------------------------------------------------------------------

void bli_cntx_print( cntx_t* cntx )
{
	dim_t i;

	// Print the values stored in the blksz_t objects.
	printf( "                               s                d                c                z\n" );
#if 0
	//for ( i = 0; i < BLIS_NUM_BLKSZS; ++i )
	for ( i = 0; i < 6; ++i )
	{
		printf( "blksz/mult %2lu:  %13lu/%2lu %13lu/%2lu %13lu/%2lu %13lu/%2lu\n",
		         i,
		         bli_cntx_get_blksz_def_dt( BLIS_FLOAT,    i, cntx ),
		         bli_cntx_get_bmult_dt    ( BLIS_FLOAT,    i, cntx ),
		         bli_cntx_get_blksz_def_dt( BLIS_DOUBLE,   i, cntx ),
		         bli_cntx_get_bmult_dt    ( BLIS_DOUBLE,   i, cntx ),
		         bli_cntx_get_blksz_def_dt( BLIS_SCOMPLEX, i, cntx ),
		         bli_cntx_get_bmult_dt    ( BLIS_SCOMPLEX, i, cntx ),
		         bli_cntx_get_blksz_def_dt( BLIS_DCOMPLEX, i, cntx ),
		         bli_cntx_get_bmult_dt    ( BLIS_DCOMPLEX, i, cntx )
		      );
	}
#endif


	for ( i = 0; i < BLIS_NUM_LEVEL3_UKRS; ++i )
	{
		func_t* ukr = bli_cntx_get_l3_vir_ukr( i, cntx );

		printf( "l3 vir ukr %2lu:  %16p %16p %16p %16p\n",
		        i,
		        bli_func_get_dt( BLIS_FLOAT,    ukr ),
		        bli_func_get_dt( BLIS_DOUBLE,   ukr ),
		        bli_func_get_dt( BLIS_SCOMPLEX, ukr ),
		        bli_func_get_dt( BLIS_DCOMPLEX, ukr )
		      );
	}

	for ( i = 0; i < BLIS_NUM_LEVEL3_UKRS; ++i )
	{
		func_t* ukr = bli_cntx_get_l3_nat_ukr( i, cntx );

		printf( "l3 nat ukr %2lu:  %16p %16p %16p %16p\n",
		        i,
		        bli_func_get_dt( BLIS_FLOAT,    ukr ),
		        bli_func_get_dt( BLIS_DOUBLE,   ukr ),
		        bli_func_get_dt( BLIS_SCOMPLEX, ukr ),
		        bli_func_get_dt( BLIS_DCOMPLEX, ukr )
		      );
	}

	for ( i = 0; i < BLIS_NUM_LEVEL1F_KERS; ++i )
	{
		func_t* ker = bli_cntx_get_l1f_ker( i, cntx );

		printf( "l1f ker    %2lu:  %16p %16p %16p %16p\n",
		        i,
		        bli_func_get_dt( BLIS_FLOAT,    ker ),
		        bli_func_get_dt( BLIS_DOUBLE,   ker ),
		        bli_func_get_dt( BLIS_SCOMPLEX, ker ),
		        bli_func_get_dt( BLIS_DCOMPLEX, ker )
		      );
	}

	for ( i = 0; i < BLIS_NUM_LEVEL1V_KERS; ++i )
	{
		func_t* ker = bli_cntx_get_l1v_ker( i, cntx );

		printf( "l1v ker    %2lu:  %16p %16p %16p %16p\n",
		        i,
		        bli_func_get_dt( BLIS_FLOAT,    ker ),
		        bli_func_get_dt( BLIS_DOUBLE,   ker ),
		        bli_func_get_dt( BLIS_SCOMPLEX, ker ),
		        bli_func_get_dt( BLIS_DCOMPLEX, ker )
		      );
	}

	{
		func_t* ukr = bli_cntx_get_packm_ukr( cntx );

		printf( "packm ker    :  %16p %16p %16p %16p\n",
		        bli_func_get_dt( BLIS_FLOAT,    ukr ),
		        bli_func_get_dt( BLIS_DOUBLE,   ukr ),
		        bli_func_get_dt( BLIS_SCOMPLEX, ukr ),
		        bli_func_get_dt( BLIS_DCOMPLEX, ukr )
		      );
	}

	{
		ind_t method = bli_cntx_get_ind_method( cntx );

		printf( "ind method   : %lu\n", ( guint_t )method );
	}
}
















