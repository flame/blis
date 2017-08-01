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
void bli_cntx_create( cntx_t* cntx )
{
	// Since cntx_t objects contain statically-allocated arrays,
	// we don't need to do anything in order to create the cntx_t
	// instance.
}

void bli_cntx_free( cntx_t* cntx )
{
	// Just as we don't need to do anything in order to create a
	// cntx_t instance, we don't need to do anything to destory
	// one.
}
#endif

void bli_cntx_clear( cntx_t* cntx )
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

blksz_t* bli_cntx_get_blksz
     (
       bszid_t bs_id,
       cntx_t* cntx
     )
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

blksz_t* bli_cntx_get_bmult
     (
       bszid_t bs_id,
       cntx_t* cntx
     )
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

func_t* bli_cntx_get_l3_ukr
     (
       l3ukr_t ukr_id,
       cntx_t* cntx
     )
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

func_t* bli_cntx_get_l3_vir_ukr
     (
       l3ukr_t ukr_id,
       cntx_t* cntx
     )
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

func_t* bli_cntx_get_l3_nat_ukr
     (
       l3ukr_t ukr_id,
       cntx_t* cntx
     )
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

func_t* bli_cntx_get_l1f_ker
     (
       l1fkr_t ker_id,
       cntx_t* cntx
     )
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

func_t* bli_cntx_get_l1v_ker
     (
       l1vkr_t ker_id,
       cntx_t* cntx
     )
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

mbool_t* bli_cntx_get_l3_nat_ukr_prefs
     (
       l3ukr_t ukr_id,
       cntx_t* cntx
     )
{
	mbool_t* l3_nat_ukrs_prefs = bli_cntx_l3_nat_ukrs_prefs_buf( cntx );
	mbool_t* l3_nat_ukrs_pref  = &l3_nat_ukrs_prefs[ ukr_id ];

	// Return the address of the native kernel func_t identified by ukr_id.
	return l3_nat_ukrs_pref;
}

func_t* bli_cntx_get_packm_ker
     (
       l1mkr_t ker_id,
       cntx_t* cntx
     )
{
	func_t* packm_kers = bli_cntx_packm_kers_buf( cntx );
	func_t* packm_ker  = &packm_kers[ ker_id ];

	// Return the address of the func_t that contains the packm ukernels.
	return packm_ker;
}

func_t* bli_cntx_get_unpackm_ker
     (
       l1mkr_t ker_id,
       cntx_t* cntx
     )
{
	func_t* unpackm_kers = bli_cntx_unpackm_kers_buf( cntx );
	func_t* unpackm_ker  = &unpackm_kers[ ker_id ];

	// Return the address of the func_t that contains the unpackm ukernels.
	return unpackm_ker;
}

#if 0
ind_t bli_cntx_get_ind_method( cntx_t* cntx )
{
	return bli_cntx_method( cntx );
}

pack_t bli_cntx_get_pack_schema_a_block( cntx_t* cntx )
{
	return bli_cntx_schema_a_block( cntx );
}

pack_t bli_cntx_get_pack_schema_b_panel( cntx_t* cntx )
{
	return bli_cntx_schema_b_panel( cntx );
}

pack_t bli_cntx_get_pack_schema_c_panel( cntx_t* cntx )
{
	return bli_cntx_schema_c_panel( cntx );
}

bool_t bli_cntx_get_ukr_anti_pref( cntx_t* cntx )
{
	return bli_cntx_anti_pref( cntx );
}
#endif

dim_t bli_cntx_get_num_threads( cntx_t* cntx )
{
	return bli_cntx_jc_way( cntx ) *
	       bli_cntx_pc_way( cntx ) *
	       bli_cntx_ic_way( cntx ) *
	       bli_cntx_jr_way( cntx ) *
	       bli_cntx_ir_way( cntx );
}

dim_t bli_cntx_get_num_threads_in
     (
       cntx_t* cntx,
       cntl_t* cntl
     )
{
	dim_t n_threads_in = 1;

	for ( ; cntl != NULL; cntl = bli_cntl_sub_node( cntl ) )
	{
		bszid_t bszid = bli_cntl_bszid( cntl );
		dim_t   cur_way;

		// We assume bszid is in {KR,MR,NR,MC,KC,NR} if it is not
		// BLIS_NO_PART.
		if ( bszid != BLIS_NO_PART )
			cur_way = bli_cntx_way_for_bszid( bszid, cntx );
		else
			cur_way = 1;

		n_threads_in *= cur_way;
	}

	return n_threads_in;
}

// -----------------------------------------------------------------------------

void bli_cntx_set_blkszs( ind_t method, dim_t n_bs, ... )
{
	/* Example prototypes:

	   void bli_cntx_set_blkszs
	   (
	     ind_t   method = BLIS_NAT,
	     dim_t   n_bs,
	     bszid_t bs0_id, blksz_t* blksz0, bszid_t bm0_id,
	     bszid_t bs1_id, blksz_t* blksz1, bszid_t bm1_id,
	     bszid_t bs2_id, blksz_t* blksz2, bszid_t bm2_id,
	     ...
	     cntx_t* cntx
	   );

	   void bli_cntx_set_blkszs
	   (
	     ind_t   method != BLIS_NAT,
	     dim_t   n_bs,
	     bszid_t bs0_id, blksz_t* blksz0, bszid_t bm0_id, dim_t def_scalr0, dim_t max_scalr0,
	     bszid_t bs1_id, blksz_t* blksz1, bszid_t bm1_id, dim_t def_scalr1, dim_t max_scalr1,
	     bszid_t bs2_id, blksz_t* blksz2, bszid_t bm2_id, dim_t def_scalr2, dim_t max_scalr2,
	     ...
	     cntx_t* cntx
	   );
	*/
	va_list   args;
	dim_t     i;

	bszid_t*  bszids;
	blksz_t** blkszs;
	bszid_t*  bmults;
	double*   dsclrs;
	double*   msclrs;

	cntx_t*   cntx;

	blksz_t*  cntx_blkszs;
	bszid_t*  cntx_bmults;


	// Allocate some temporary local arrays.
	bszids = bli_malloc_intl( n_bs * sizeof( bszid_t  ) );
	blkszs = bli_malloc_intl( n_bs * sizeof( blksz_t* ) );
	bmults = bli_malloc_intl( n_bs * sizeof( bszid_t  ) );
	dsclrs = bli_malloc_intl( n_bs * sizeof( double   ) );
	msclrs = bli_malloc_intl( n_bs * sizeof( double   ) );

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
			// - the bszid_t of the multiple
			// that we need to associate with the blksz_t object.
			bszid_t  bs_id = va_arg( args, bszid_t  );
			blksz_t* blksz = va_arg( args, blksz_t* );
			bszid_t  bm_id = va_arg( args, bszid_t  );

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
			// - the address of the blksz_t object,
			// - the bszid_t of the multiple, and
			// - the scalars we wish to apply to the real blocksizes to
			//   come up with the induced complex blocksizes (for default
			//   and maximum blocksizes).
			bszid_t  bs_id = va_arg( args, bszid_t  );
			blksz_t* blksz = va_arg( args, blksz_t* );
			bszid_t  bm_id = va_arg( args, bszid_t  );
			double   dsclr = va_arg( args, double   );
			double   msclr = va_arg( args, double   );

			// Store the values in our temporary arrays.
			bszids[ i ] = bs_id;
			blkszs[ i ] = blksz;
			bmults[ i ] = bm_id;
			dsclrs[ i ] = dsclr;
			msclrs[ i ] = msclr;
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
			bszid_t  bs_id = bszids[ i ];
			bszid_t  bm_id = bmults[ i ];

			blksz_t* blksz = blkszs[ i ];

			blksz_t* cntx_blksz = &cntx_blkszs[ bs_id ];

			// Copy the blksz_t object contents into the appropriate
			// location within the context's blksz_t array. Do the same
			// for the blocksize multiple id.
			//cntx_blkszs[ bs_id ] = *blksz;
			//bli_blksz_copy_smart( blksz, cntx_blksz );
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
			bszid_t  bs_id = bszids[ i ];
			bszid_t  bm_id = bmults[ i ];
			double   dsclr = dsclrs[ i ];
			double   msclr = msclrs[ i ];

			blksz_t* blksz = blkszs[ i ];
			blksz_t* bmult = blkszs[ i ];

			blksz_t* cntx_blksz = &cntx_blkszs[ bs_id ];

			// Copy the real domain values of the source blksz_t object into
			// the context, duplicating into the complex domain fields.
			bli_blksz_copy_dt( BLIS_FLOAT,  blksz, BLIS_FLOAT,    cntx_blksz );
			bli_blksz_copy_dt( BLIS_DOUBLE, blksz, BLIS_DOUBLE,   cntx_blksz );
			bli_blksz_copy_dt( BLIS_FLOAT,  blksz, BLIS_SCOMPLEX, cntx_blksz );
			bli_blksz_copy_dt( BLIS_DOUBLE, blksz, BLIS_DCOMPLEX, cntx_blksz );

			// If the default blocksize scalar is non-unit, we need to scale
			// the complex domain default blocksizes.
			if ( dsclr != 1.0 )
			{
				// Scale the complex domain default blocksize values in the
				// blocksize object.
				bli_blksz_scale_def( 1, ( dim_t )dsclr, BLIS_SCOMPLEX, cntx_blksz );
				bli_blksz_scale_def( 1, ( dim_t )dsclr, BLIS_DCOMPLEX, cntx_blksz );

				if ( bs_id != bm_id )
				{
					// Round the newly-scaled blocksizes down to their multiple.
					// (Note that both the default and maximum blocksize values
					// must be a multiple of the same blocksize multiple.) Also,
					// note that this is only done when the blocksize id is not
					// equal to the blocksize multiple id (ie: we don't round
					// down scaled register blocksizes since they are their own
					// multiples).
					bli_blksz_reduce_def_to( BLIS_FLOAT,  bmult, BLIS_SCOMPLEX, cntx_blksz );
					bli_blksz_reduce_def_to( BLIS_DOUBLE, bmult, BLIS_DCOMPLEX, cntx_blksz );
				}
			}

			// Similarly, if the maximum blocksize scalar is non-unit, we need
			// to scale the complex domain maximum blocksizes.
			if ( msclr != 1.0 )
			{
				// Scale the complex domain maximum blocksize values in the
				// blocksize object.
				bli_blksz_scale_max( 1, ( dim_t )msclr, BLIS_SCOMPLEX, cntx_blksz );
				bli_blksz_scale_max( 1, ( dim_t )msclr, BLIS_DCOMPLEX, cntx_blksz );

				if ( bs_id != bm_id )
				{
					// Round the newly-scaled blocksizes down to their multiple.
					// (Note that both the default and maximum blocksize values
					// must be a multiple of the same blocksize multiple.) Also,
					// note that this is only done when the blocksize id is not
					// equal to the blocksize multiple id (ie: we don't round
					// down scaled register blocksizes since they are their own
					// multiples).
					bli_blksz_reduce_max_to( BLIS_FLOAT,  bmult, BLIS_SCOMPLEX, cntx_blksz );
					bli_blksz_reduce_max_to( BLIS_DOUBLE, bmult, BLIS_DCOMPLEX, cntx_blksz );
				}
			}

			// Copy the blocksize multiple id into the context.
			cntx_bmults[ bs_id ] = bm_id;
		}
	}

	// Free the temporary local arrays.
	bli_free_intl( blkszs );
	bli_free_intl( bszids );
	bli_free_intl( bmults );
	bli_free_intl( dsclrs );
	bli_free_intl( msclrs );
}

// -----------------------------------------------------------------------------

void bli_cntx_set_blksz
     (
       bszid_t  bs_id,
       blksz_t* blksz,
       bszid_t  mult_id,
       cntx_t*  cntx
     )
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

// -----------------------------------------------------------------------------

void bli_cntx_set_l3_nat_ukrs( dim_t n_ukrs, ... )
{
	/* Example prototypes:

	   void bli_cntx_set_l3_nat_ukrs
	   (
	     dim_t   n_ukrs,
	     l3ukr_t ukr0_id, num_t dt0, void* ukr0_fp, bool_t pref0,
	     l3ukr_t ukr1_id, num_t dt1, void* ukr1_fp, bool_t pref1,
	     l3ukr_t ukr2_id, num_t dt2, void* ukr2_fp, bool_t pref2,
	     ...
	     cntx_t* cntx
	   );
	*/
	va_list   args;
	dim_t     i;

	// Allocate some temporary local arrays.
	l3ukr_t* ukr_ids   = bli_malloc_intl( n_ukrs * sizeof( l3ukr_t ) );
	num_t*   ukr_dts   = bli_malloc_intl( n_ukrs * sizeof( num_t   ) );
	void**   ukr_fps   = bli_malloc_intl( n_ukrs * sizeof( void*   ) );
	bool_t*  ukr_prefs = bli_malloc_intl( n_ukrs * sizeof( bool_t  ) );

	// -- Begin variable argument section --

	// Initialize variable argument environment.
	va_start( args, n_ukrs );

	// Process n_ukrs tuples.
	for ( i = 0; i < n_ukrs; ++i )
	{
		// Here, we query the variable argument list for:
		// - the l3ukr_t of the kernel we're about to process,
		// - the datatype of the kernel,
		// - the kernel function pointer, and
		// - the kernel function storage preference
		// that we need to store to the context.
		const l3ukr_t  ukr_id   = va_arg( args, l3ukr_t );
		const num_t    ukr_dt   = va_arg( args, num_t   );
		      void*    ukr_fp   = va_arg( args, void*   );
		const bool_t   ukr_pref = va_arg( args, bool_t  );

		// Store the values in our temporary arrays.
		ukr_ids[ i ]   = ukr_id;
		ukr_dts[ i ]   = ukr_dt;
		ukr_fps[ i ]   = ukr_fp;
		ukr_prefs[ i ] = ukr_pref;
	}

	// The last argument should be the context pointer.
	cntx_t* cntx = va_arg( args, cntx_t* );

	// Shutdown variable argument environment and clean up stack.
	va_end( args );

	// -- End variable argument section --

	// Query the context for the addresses of:
	// - the l3 native ukernel func_t array
	// - the l3 native ukernel preferences array
	func_t*  cntx_l3_nat_ukrs       = bli_cntx_l3_nat_ukrs_buf( cntx );
	mbool_t* cntx_l3_nat_ukrs_prefs = bli_cntx_l3_nat_ukrs_prefs_buf( cntx );

	// Now that we have the context address, we want to copy the values
	// from the temporary buffers into the corresponding buffers in the
	// context.

	// Process each blocksize id tuple provided.
	for ( i = 0; i < n_ukrs; ++i )
	{
		// Read the current blocksize id, blksz_t* pointer, blocksize
		// multiple id, and blocksize scalar.
		const l3ukr_t ukr_id   = ukr_ids[ i ];
		const num_t   ukr_dt   = ukr_dts[ i ];
		      void*   ukr_fp   = ukr_fps[ i ];
		const bool_t  ukr_pref = ukr_prefs[ i ];

		// Index into the func_t and mbool_t for the current kernel id
		// being processed.
		func_t*       ukrs   = &cntx_l3_nat_ukrs[ ukr_id ];
		mbool_t*      prefs  = &cntx_l3_nat_ukrs_prefs[ ukr_id ];

		// Store the ukernel function pointer and preference values into
		// the context.
		bli_func_set_dt( ukr_fp, ukr_dt, ukrs );
		bli_mbool_set_dt( ukr_pref, ukr_dt, prefs );
	}

	// Free the temporary local arrays.
	bli_free_intl( ukr_ids );
	bli_free_intl( ukr_dts );
	bli_free_intl( ukr_fps );
	bli_free_intl( ukr_prefs );
}

// -----------------------------------------------------------------------------

void bli_cntx_set_l3_nat_ukr
     (
       l3ukr_t ukr_id,
       func_t* func,
       cntx_t* cntx
     )
{
	func_t* l3_nat_ukrs = bli_cntx_l3_nat_ukrs_buf( cntx );

	// Copy the function object into the specified location within
	// the context's native level-3 ukernel array.
	l3_nat_ukrs[ ukr_id ] = *func;
}

void bli_cntx_set_l3_nat_ukr_prefs
     (
       l3ukr_t  ukr_id,
       mbool_t* prefs,
       cntx_t*  cntx
     )
{
	mbool_t* l3_nat_ukrs_prefs = bli_cntx_l3_nat_ukrs_prefs_buf( cntx );

	// Copy the mbool_t into the specified location within
	// the context's native level-3 ukernel preference array.
	l3_nat_ukrs_prefs[ ukr_id ] = *prefs;
}

void bli_cntx_set_l3_vir_ukr
     (
       l3ukr_t ukr_id,
       func_t* func,
       cntx_t* cntx
     )
{
	func_t* l3_vir_ukrs = bli_cntx_l3_vir_ukrs_buf( cntx );

	// Copy the function object into the specified location within
	// the context's virtual level-3 ukernel array.
	l3_vir_ukrs[ ukr_id ] = *func;
}

void bli_cntx_set_l1f_ker
     (
       l1fkr_t ker_id,
       func_t* func,
       cntx_t* cntx
     )
{
	func_t* l1f_kers = bli_cntx_l1f_kers_buf( cntx );

	// Copy the function object into the specified location within
	// the context's level-1f kernel array.
	l1f_kers[ ker_id ] = *func;
}

void bli_cntx_set_l1v_ker
     (
       l1vkr_t ker_id,
       func_t* func,
       cntx_t* cntx
     )
{
	func_t* l1v_kers = bli_cntx_l1v_kers_buf( cntx );

	// Copy the function object into the specified location within
	// the context's level-1v kernel array.
	l1v_kers[ ker_id ] = *func;
}

// -----------------------------------------------------------------------------

void bli_cntx_set_packm_kers( dim_t n_kers, ... )
{
	/* Example prototypes:

	   void bli_cntx_set_packm_kers
	   (
	     dim_t   n_ukrs,
	     l1mkr_t ker0_id, num_t ker0_dt, void* ker0_fp,
	     l1mkr_t ker1_id, num_t ker1_dt, void* ker1_fp,
	     l1mkr_t ker2_id, num_t ker2_dt, void* ker2_fp,
	     ...
	     cntx_t* cntx
	   );
	*/
	va_list   args;
	dim_t     i;

	// Allocate some temporary local arrays.
	l1mkr_t* ker_ids   = bli_malloc_intl( n_kers * sizeof( l1mkr_t ) );
	num_t*   ker_dts   = bli_malloc_intl( n_kers * sizeof( num_t   ) );
	void**   ker_fps   = bli_malloc_intl( n_kers * sizeof( void*   ) );

	// -- Begin variable argument section --

	// Initialize variable argument environment.
	va_start( args, n_kers );

	// Process n_kers tuples.
	for ( i = 0; i < n_kers; ++i )
	{
		// Here, we query the variable argument list for:
		// - the l1mkr_t of the kernel we're about to process,
		// - the datatype of the kernel, and
		// - the kernel function pointer
		// that we need to store to the context.
		const l1mkr_t  ker_id   = va_arg( args, l1mkr_t );
		const num_t    ker_dt   = va_arg( args, num_t   );
		      void*    ker_fp   = va_arg( args, void*   );

		// Store the values in our temporary arrays.
		ker_ids[ i ]   = ker_id;
		ker_dts[ i ]   = ker_dt;
		ker_fps[ i ]   = ker_fp;
	}

	// The last argument should be the context pointer.
	cntx_t* cntx = va_arg( args, cntx_t* );

	// Shutdown variable argument environment and clean up stack.
	va_end( args );

	// -- End variable argument section --

	// Query the context for the address of:
	// - the packm kernels func_t array
	func_t* cntx_packm_kers = bli_cntx_packm_kers_buf( cntx );

	// Now that we have the context address, we want to copy the values
	// from the temporary buffers into the corresponding buffers in the
	// context.

	// Process each blocksize id tuple provided.
	for ( i = 0; i < n_kers; ++i )
	{
		// Read the current blocksize id, blksz_t* pointer, blocksize
		// multiple id, and blocksize scalar.
		const l1mkr_t ker_id   = ker_ids[ i ];
		const num_t   ker_dt   = ker_dts[ i ];
		      void*   ker_fp   = ker_fps[ i ];

		// Index into the func_t and mbool_t for the current kernel id
		// being processed.
		func_t*       kers     = &cntx_packm_kers[ ker_id ];

		// Store the ukernel function pointer and preference values into
		// the context.
		bli_func_set_dt( ker_fp, ker_dt, kers );
	}

	// Free the temporary local arrays.
	bli_free_intl( ker_ids );
	bli_free_intl( ker_dts );
	bli_free_intl( ker_fps );
}

// -----------------------------------------------------------------------------

void bli_cntx_set_packm_ker
     (
       l1mkr_t ker_id,
       func_t* func,
       cntx_t* cntx
     )
{
	func_t* packm_kers = bli_cntx_packm_kers_buf( cntx );

	// Copy the function object into the specified location within
	// the context's packm kernel array.
	packm_kers[ ker_id ] = *func;
}

// -----------------------------------------------------------------------------

void bli_cntx_set_ind_method
     (
       ind_t   method,
       cntx_t* cntx
     )
{
	bli_cntx_set_method( method, cntx );
}

void bli_cntx_set_pack_schema_ab_blockpanel
     (
       pack_t  schema_a,
       pack_t  schema_b,
       cntx_t* cntx
     )
{
	bli_cntx_set_schema_a_block( schema_a, cntx );
	bli_cntx_set_schema_b_panel( schema_b, cntx );
}

void bli_cntx_set_pack_schema_a_block
     (
       pack_t  schema_a,
       cntx_t* cntx
     )
{
	bli_cntx_set_schema_a_block( schema_a, cntx );
}

void bli_cntx_set_pack_schema_b_panel
     (
       pack_t  schema_b,
       cntx_t* cntx
     )
{
	bli_cntx_set_schema_b_panel( schema_b, cntx );
}

void bli_cntx_set_pack_schema_c_panel
     (
       pack_t  schema_c,
       cntx_t* cntx
     )
{
	bli_cntx_set_schema_c_panel( schema_c, cntx );
}

#if 0
void bli_cntx_set_ukr_anti_pref( bool_t  anti_pref,
                                 cntx_t* cntx )
{
	bli_cntx_set_anti_pref( anti_pref, cntx );
}
#endif

void bli_cntx_set_thrloop_from_env
     (
       opid_t  l3_op,
       side_t  side,
       cntx_t* cntx,
       dim_t   m,
       dim_t   n,
       dim_t   k
     )
{
	dim_t jc, pc, ic, jr, ir;

#ifdef BLIS_ENABLE_MULTITHREADING

	int nthread = bli_thread_get_env( "BLIS_NUM_THREADS", -1 );

	if ( nthread == -1 )
	    nthread = bli_thread_get_env( "OMP_NUM_THREADS", -1 );

	if ( nthread < 1 ) nthread = 1;

    bli_partition_2x2( nthread, m*BLIS_DEFAULT_M_THREAD_RATIO,
                                n*BLIS_DEFAULT_N_THREAD_RATIO, &ic, &jc );

    for ( ir = BLIS_DEFAULT_MR_THREAD_MAX ; ir > 1 ; ir-- )
    {
        if ( ic % ir == 0 )
        {
            ic /= ir;
            break;
        }
    }

    for ( jr = BLIS_DEFAULT_NR_THREAD_MAX ; jr > 1 ; jr-- )
    {
        if ( jc % jr == 0 )
        {
            jc /= jr;
            break;
        }
    }

    pc = 1;

    dim_t jc_env = bli_thread_get_env( "BLIS_JC_NT", -1 );
    dim_t ic_env = bli_thread_get_env( "BLIS_IC_NT", -1 );
    dim_t jr_env = bli_thread_get_env( "BLIS_JR_NT", -1 );
    dim_t ir_env = bli_thread_get_env( "BLIS_IR_NT", -1 );

    if (jc_env != -1 || ic_env != -1 || jr_env != -1 || ir_env != -1)
    {
        jc = (jc_env == -1 ? 1 : jc_env);
        ic = (ic_env == -1 ? 1 : ic_env);
        jr = (jr_env == -1 ? 1 : jr_env);
        ir = (ir_env == -1 ? 1 : ir_env);
    }

#else

	jc = 1;
	pc = 1;
	ic = 1;
	jr = 1;
	ir = 1;

#endif

	if ( l3_op == BLIS_TRMM )
	{
		// We reconfigure the paralelism from trmm_r due to a dependency in
		// the jc loop. (NOTE: This dependency does not exist for trmm3 )
		if ( bli_is_right( side ) )
		{
			bli_cntx_set_thrloop
			(
			  1,
			  pc,
			  ic,
			  jr * jc,
			  ir,
			  cntx
			);
		}
		else // if ( bli_is_left( side ) )
		{
			bli_cntx_set_thrloop
			(
			  jc,
			  pc,
			  ic,
			  jr,
			  ir,
			  cntx
			);
		}
	}
	else if ( l3_op == BLIS_TRSM )
	{
		if ( bli_is_right( side ) )
		{
			bli_cntx_set_thrloop
			(
			  1,
			  1,
			  ic * pc * jc * ir * jr,
			  1,
			  1,
			  cntx
			);
		}
		else // if ( bli_is_left( side ) )
		{
			bli_cntx_set_thrloop
			(
			  1,
			  1,
			  1,
			  ic * pc * jc * jr * ir,
			  1,
			  cntx
			);
		}
	}
	else // if ( l3_op == BLIS_TRSM )
	{
		bli_cntx_set_thrloop
		(
		  jc,
		  pc,
		  ic,
		  jr,
		  ir,
		  cntx
		);
	}
}


// -----------------------------------------------------------------------------

bool_t bli_cntx_l3_nat_ukr_prefers_rows_dt
     (
       num_t   dt,
       l3ukr_t ukr_id,
       cntx_t* cntx
     )
{
	mbool_t* ukrs_prefs = bli_cntx_get_l3_nat_ukr_prefs( ukr_id, cntx );
	bool_t   ukr_prefs  = bli_mbool_get_dt( dt, ukrs_prefs );

	// A ukernel preference of TRUE means the ukernel prefers row
	// storage.
	return ukr_prefs == TRUE;
}

bool_t bli_cntx_l3_nat_ukr_prefers_cols_dt
     (
       num_t   dt,
       l3ukr_t ukr_id,
       cntx_t* cntx
     )
{
	mbool_t* ukrs_prefs = bli_cntx_get_l3_nat_ukr_prefs( ukr_id, cntx );
	bool_t   ukr_prefs  = bli_mbool_get_dt( dt, ukrs_prefs );

	// A ukernel preference of FALSE means the ukernel prefers column
	// storage.
	return ukr_prefs == FALSE;
}

bool_t bli_cntx_l3_nat_ukr_prefers_storage_of
     (
       obj_t*  obj,
       l3ukr_t ukr_id,
       cntx_t* cntx
     )
{
	return !bli_cntx_l3_nat_ukr_dislikes_storage_of( obj, ukr_id, cntx );
}

bool_t bli_cntx_l3_nat_ukr_dislikes_storage_of
     (
       obj_t*  obj,
       l3ukr_t ukr_id,
       cntx_t* cntx
     )
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

bool_t bli_cntx_l3_nat_ukr_eff_prefers_storage_of
     (
       obj_t*  obj,
       l3ukr_t ukr_id,
       cntx_t* cntx
     )
{
	bool_t r_val = bli_cntx_l3_nat_ukr_prefers_storage_of( obj, ukr_id, cntx );

	// If the anti-preference is set, negate the result.
	if ( bli_cntx_anti_pref( cntx ) ) r_val = !r_val;

	return r_val;
}

bool_t bli_cntx_l3_nat_ukr_eff_dislikes_storage_of
     (
       obj_t*  obj,
       l3ukr_t ukr_id,
       cntx_t* cntx
     )
{
	bool_t r_val = bli_cntx_l3_nat_ukr_dislikes_storage_of( obj, ukr_id, cntx );

	// If the anti-preference is set, negate the result.
	if ( bli_cntx_anti_pref( cntx ) ) r_val = !r_val;

	return r_val;
}

// -----------------------------------------------------------------------------

bool_t bli_cntx_l3_ukr_prefers_rows_dt
     (
       num_t   dt,
       l3ukr_t ukr_id,
       cntx_t* cntx
     )
{
	// Reference the ukr storage preferences of the corresponding real
	// micro-kernel for induced methods.
	if ( bli_cntx_get_ind_method( cntx ) != BLIS_NAT )
	    dt = bli_datatype_proj_to_real( dt );

	return bli_cntx_l3_nat_ukr_prefers_rows_dt( dt, ukr_id, cntx );
}

bool_t bli_cntx_l3_ukr_prefers_cols_dt
     (
       num_t   dt,
       l3ukr_t ukr_id,
       cntx_t* cntx
     )
{
	// Reference the ukr storage preferences of the corresponding real
	// micro-kernel for induced methods.
	if ( bli_cntx_get_ind_method( cntx ) != BLIS_NAT )
	    dt = bli_datatype_proj_to_real( dt );

	return bli_cntx_l3_nat_ukr_prefers_cols_dt( dt, ukr_id, cntx );
}

bool_t bli_cntx_l3_ukr_prefers_storage_of
     (
       obj_t*  obj,
       l3ukr_t ukr_id,
       cntx_t* cntx
     )
{
	return !bli_cntx_l3_ukr_dislikes_storage_of( obj, ukr_id, cntx );
}

bool_t bli_cntx_l3_ukr_dislikes_storage_of
     (
       obj_t*  obj,
       l3ukr_t ukr_id,
       cntx_t* cntx
     )
{
	num_t dt = bli_obj_datatype( *obj );

	const bool_t ukr_prefers_rows
	                   = bli_cntx_l3_ukr_prefers_rows_dt( dt, ukr_id, cntx );
	const bool_t ukr_prefers_cols
	                   = bli_cntx_l3_ukr_prefers_cols_dt( dt, ukr_id, cntx );
	bool_t       r_val = FALSE;

	if      ( bli_obj_is_row_stored( *obj ) && ukr_prefers_cols ) r_val = TRUE;
	else if ( bli_obj_is_col_stored( *obj ) && ukr_prefers_rows ) r_val = TRUE;

	return r_val;
}

bool_t bli_cntx_l3_ukr_eff_prefers_storage_of
     (
       obj_t*  obj,
       l3ukr_t ukr_id,
       cntx_t* cntx
     )
{
	bool_t r_val = bli_cntx_l3_ukr_prefers_storage_of( obj, ukr_id, cntx );

	// If the anti-preference is set, negate the result.
	if ( bli_cntx_anti_pref( cntx ) ) r_val = !r_val;

	return r_val;
}

bool_t bli_cntx_l3_ukr_eff_dislikes_storage_of
     (
       obj_t*  obj,
       l3ukr_t ukr_id,
       cntx_t* cntx
     )
{
	bool_t r_val = bli_cntx_l3_ukr_dislikes_storage_of( obj, ukr_id, cntx );

	// If the anti-preference is set, negate the result.
	if ( bli_cntx_anti_pref( cntx ) ) r_val = !r_val;

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
		ind_t method = bli_cntx_get_ind_method( cntx );

		printf( "ind method   : %lu\n", ( guint_t )method );
	}
}

