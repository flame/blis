/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2014, The University of Texas at Austin
   Copyright (C) 2018 - 2019, Advanced Micro Devices, Inc.

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
		bli_blksz_copy_if_nonneg( blksz, cntx_blksz );

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
			// Scale the default blocksize value corresponding to the given
			// datatype.
			bli_blksz_scale_def( 1, ( dim_t )dsclr, dt, cntx_blksz );
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
	va_list   args;
	va_start( args, cntx );

	// Process ukernel preferences until BLIS_VA_END is reached.
	while ( true )
	{
		const int ukr_pref_id0 = va_arg( args, int );

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

void bli_cntx_set_l3_sup_handlers( cntx_t* cntx, ... )
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

