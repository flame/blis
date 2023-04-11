/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2014, The University of Texas at Austin
   Copyright (C) 2018-2020, Advanced Micro Devices, Inc.

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

// The array of cntx_t* pointers to cache modified contexts used by induced
// methods.
static cntx_t** gks[ BLIS_NUM_ARCHS ];

// The array of function pointers holding the registered context initialization
// functions for induced methods.
static void_fp  cntx_ind_init[ BLIS_NUM_ARCHS ];

// The array of function pointers holding the registered context initialization
// functions for reference kernels.
static void_fp  cntx_ref_init[ BLIS_NUM_ARCHS ];

// Define a function pointer type for context initialization functions.
typedef void (*nat_cntx_init_ft)( cntx_t* cntx );
typedef void (*ref_cntx_init_ft)( cntx_t* cntx );
typedef void (*ind_cntx_init_ft)( ind_t method, cntx_t* cntx );

// Cached copies of the pointers to the native and induced contexts for the
// active subconfiguration. When BLIS_ENABLE_GKS_CACHING is enabled, these
// pointers will be set once and then reused to fulfill subsequent context
// queries.
static cntx_t* cached_cntx_nat = NULL;
static cntx_t* cached_cntx_ind = NULL;

// -----------------------------------------------------------------------------

void bli_gks_init( void )
{
	{
		// Initialize the internal data structure we use to track registered
		// contexts.
		bli_gks_init_index();

		// Register a context for each architecture that was #define'd in
		// bli_config.h.

		// -- Intel architectures ----------------------------------------------

#ifdef BLIS_CONFIG_SKX
		bli_gks_register_cntx( BLIS_ARCH_SKX,         bli_cntx_init_skx,
		                                              bli_cntx_init_skx_ref,
		                                              bli_cntx_init_skx_ind );
#endif
#ifdef BLIS_CONFIG_KNL
		bli_gks_register_cntx( BLIS_ARCH_KNL,         bli_cntx_init_knl,
		                                              bli_cntx_init_knl_ref,
		                                              bli_cntx_init_knl_ind );
#endif
#ifdef BLIS_CONFIG_KNC
		bli_gks_register_cntx( BLIS_ARCH_KNC,         bli_cntx_init_knc,
		                                              bli_cntx_init_knc_ref,
		                                              bli_cntx_init_knc_ind );
#endif
#ifdef BLIS_CONFIG_HASWELL
		bli_gks_register_cntx( BLIS_ARCH_HASWELL,     bli_cntx_init_haswell,
		                                              bli_cntx_init_haswell_ref,
		                                              bli_cntx_init_haswell_ind );
#endif
#ifdef BLIS_CONFIG_SANDYBRIDGE
		bli_gks_register_cntx( BLIS_ARCH_SANDYBRIDGE, bli_cntx_init_sandybridge,
		                                              bli_cntx_init_sandybridge_ref,
		                                              bli_cntx_init_sandybridge_ind );
#endif
#ifdef BLIS_CONFIG_PENRYN
		bli_gks_register_cntx( BLIS_ARCH_PENRYN,      bli_cntx_init_penryn,
		                                              bli_cntx_init_penryn_ref,
		                                              bli_cntx_init_penryn_ind );
#endif

		// -- AMD architectures ------------------------------------------------

#ifdef BLIS_CONFIG_ZEN3
		bli_gks_register_cntx( BLIS_ARCH_ZEN3,        bli_cntx_init_zen3,
		                                              bli_cntx_init_zen3_ref,
		                                              bli_cntx_init_zen3_ind );
#endif
#ifdef BLIS_CONFIG_ZEN2
		bli_gks_register_cntx( BLIS_ARCH_ZEN2,        bli_cntx_init_zen2,
		                                              bli_cntx_init_zen2_ref,
		                                              bli_cntx_init_zen2_ind );
#endif
#ifdef BLIS_CONFIG_ZEN
		bli_gks_register_cntx( BLIS_ARCH_ZEN,         bli_cntx_init_zen,
		                                              bli_cntx_init_zen_ref,
		                                              bli_cntx_init_zen_ind );
#endif
#ifdef BLIS_CONFIG_EXCAVATOR
		bli_gks_register_cntx( BLIS_ARCH_EXCAVATOR,   bli_cntx_init_excavator,
		                                              bli_cntx_init_excavator_ref,
		                                              bli_cntx_init_excavator_ind );
#endif
#ifdef BLIS_CONFIG_STEAMROLLER
		bli_gks_register_cntx( BLIS_ARCH_STEAMROLLER, bli_cntx_init_steamroller,
		                                              bli_cntx_init_steamroller_ref,
		                                              bli_cntx_init_steamroller_ind );
#endif
#ifdef BLIS_CONFIG_PILEDRIVER
		bli_gks_register_cntx( BLIS_ARCH_PILEDRIVER,  bli_cntx_init_piledriver,
		                                              bli_cntx_init_piledriver_ref,
		                                              bli_cntx_init_piledriver_ind );
#endif
#ifdef BLIS_CONFIG_BULLDOZER
		bli_gks_register_cntx( BLIS_ARCH_BULLDOZER,   bli_cntx_init_bulldozer,
		                                              bli_cntx_init_bulldozer_ref,
		                                              bli_cntx_init_bulldozer_ind );
#endif

		// -- ARM architectures ------------------------------------------------

		// -- ARM-SVE --
#ifdef BLIS_CONFIG_ARMSVE
		bli_gks_register_cntx( BLIS_ARCH_ARMSVE,      bli_cntx_init_armsve,
		                                              bli_cntx_init_armsve_ref,
		                                              bli_cntx_init_armsve_ind );
#endif
#ifdef BLIS_CONFIG_A64FX
		bli_gks_register_cntx( BLIS_ARCH_A64FX,       bli_cntx_init_a64fx,
		                                              bli_cntx_init_a64fx_ref,
		                                              bli_cntx_init_a64fx_ind );
#endif

		// -- ARM-NEON (4 pipes x 128-bit vectors) --
#ifdef BLIS_CONFIG_FIRESTORM
		bli_gks_register_cntx( BLIS_ARCH_FIRESTORM,   bli_cntx_init_firestorm,
		                                              bli_cntx_init_firestorm_ref,
		                                              bli_cntx_init_firestorm_ind );
#endif

		// -- ARM (2 pipes x 128-bit vectors) --
#ifdef BLIS_CONFIG_THUNDERX2
		bli_gks_register_cntx( BLIS_ARCH_THUNDERX2,   bli_cntx_init_thunderx2,
		                                              bli_cntx_init_thunderx2_ref,
		                                              bli_cntx_init_thunderx2_ind );
#endif
#ifdef BLIS_CONFIG_CORTEXA57
		bli_gks_register_cntx( BLIS_ARCH_CORTEXA57,   bli_cntx_init_cortexa57,
		                                              bli_cntx_init_cortexa57_ref,
		                                              bli_cntx_init_cortexa57_ind );
#endif
#ifdef BLIS_CONFIG_CORTEXA53
		bli_gks_register_cntx( BLIS_ARCH_CORTEXA53,   bli_cntx_init_cortexa53,
		                                              bli_cntx_init_cortexa53_ref,
		                                              bli_cntx_init_cortexa53_ind );
#endif

		// -- ARM (older 32-bit microarchitectures) --
#ifdef BLIS_CONFIG_CORTEXA15
		bli_gks_register_cntx( BLIS_ARCH_CORTEXA15,   bli_cntx_init_cortexa15,
		                                              bli_cntx_init_cortexa15_ref,
		                                              bli_cntx_init_cortexa15_ind );
#endif
#ifdef BLIS_CONFIG_CORTEXA9
		bli_gks_register_cntx( BLIS_ARCH_CORTEXA9,    bli_cntx_init_cortexa9,
		                                              bli_cntx_init_cortexa9_ref,
		                                              bli_cntx_init_cortexa9_ind );
#endif

		// -- IBM architectures ------------------------------------------------

#ifdef BLIS_CONFIG_POWER10
		bli_gks_register_cntx( BLIS_ARCH_POWER10,     bli_cntx_init_power10,
		                                              bli_cntx_init_power10_ref,
		                                              bli_cntx_init_power10_ind );
#endif
#ifdef BLIS_CONFIG_POWER9
		bli_gks_register_cntx( BLIS_ARCH_POWER9,      bli_cntx_init_power9,
		                                              bli_cntx_init_power9_ref,
		                                              bli_cntx_init_power9_ind );
#endif
#ifdef BLIS_CONFIG_POWER7
		bli_gks_register_cntx( BLIS_ARCH_POWER7,      bli_cntx_init_power7,
		                                              bli_cntx_init_power7_ref,
		                                              bli_cntx_init_power7_ind );
#endif
#ifdef BLIS_CONFIG_BGQ
		bli_gks_register_cntx( BLIS_ARCH_BGQ,         bli_cntx_init_bgq,
		                                              bli_cntx_init_bgq_ref,
		                                              bli_cntx_init_bgq_ind );
#endif

		// -- RISC-V architectures --------------------------------------------

#ifdef BLIS_CONFIG_RV32I
		bli_gks_register_cntx( BLIS_ARCH_RV32I,       bli_cntx_init_rv32i,
		                                              bli_cntx_init_rv32i_ref,
		                                              bli_cntx_init_rv32i_ind );
#endif

#ifdef BLIS_CONFIG_RV64I
		bli_gks_register_cntx( BLIS_ARCH_RV64I,       bli_cntx_init_rv64i,
		                                              bli_cntx_init_rv64i_ref,
		                                              bli_cntx_init_rv64i_ind );
#endif

#ifdef BLIS_CONFIG_RV32IV
		bli_gks_register_cntx( BLIS_ARCH_RV32IV,      bli_cntx_init_rv32iv,
		                                              bli_cntx_init_rv32iv_ref,
		                                              bli_cntx_init_rv32iv_ind );
#endif

#ifdef BLIS_CONFIG_RV64IV
		bli_gks_register_cntx( BLIS_ARCH_RV64IV,      bli_cntx_init_rv64iv,
		                                              bli_cntx_init_rv64iv_ref,
		                                              bli_cntx_init_rv64iv_ind );
#endif

		// -- Generic architectures --------------------------------------------

#ifdef BLIS_CONFIG_GENERIC
		bli_gks_register_cntx( BLIS_ARCH_GENERIC,     bli_cntx_init_generic,
		                                              bli_cntx_init_generic_ref,
		                                              bli_cntx_init_generic_ind );
#endif
	}

#ifdef BLIS_ENABLE_GKS_CACHING
	// Deep-query and cache the native and induced method contexts so they are
	// ready to go when needed (by BLIS or the application). Notice that we use
	// the _noinit() APIs, which skip their internal calls to bli_init_once().
	// The reasons: (1) Skipping that call is necessary to prevent an infinite
	// loop since the current function, bli_gks_init(), is called from within
	// bli_init_once(); and (2) we can guarantee that the gks has been
	// initialized given that bli_gks_init() is about to return.
	cached_cntx_nat = ( cntx_t* )bli_gks_query_nat_cntx_noinit();
	cached_cntx_ind = ( cntx_t* )bli_gks_query_ind_cntx_noinit( BLIS_1M );
#endif
}

// -----------------------------------------------------------------------------

void bli_gks_finalize( void )
{
	arch_t id;
	ind_t  ind;

	// BEGIN CRITICAL SECTION
	// NOTE: This critical section is implicit. We assume this function is only
	// called from within the critical section within bli_finalize().
	{

		// Iterate over the architectures in the gks array.
		for ( id = 0; id < BLIS_NUM_ARCHS; ++id )
		{
			cntx_t** gks_id = gks[ id ];

			// Only consider context arrays for architectures that were allocated
			// in the first place.
			if ( gks_id != NULL )
			{
				// Iterate over the induced methods in the current sub-array
				// referenced by cntx_pp.
				for ( ind = 0; ind < BLIS_NUM_IND_METHODS; ++ind )
				{
					cntx_t* gks_id_ind = gks_id[ ind ];

					// If the current context was allocated, free it.
					if ( gks_id_ind != NULL )
					{
						#ifdef BLIS_ENABLE_MEM_TRACING
						printf( "bli_gks_finalize(): cntx for ind_t %d: ", ( int )ind );
						#endif

						bli_free_intl( gks_id_ind );
					}
				}

				#ifdef BLIS_ENABLE_MEM_TRACING
				printf( "bli_gks_finalize(): gks for arch_t %d: ", ( int )id );
				#endif

				// Free the array of BLIS_NUM_IND_METHODS cntx* elements.
				bli_free_intl( gks_id );
			}
		}

	}
	// END CRITICAL SECTION

#ifdef BLIS_ENABLE_GKS_CACHING
	// Clear the cached pointers to the native and induced contexts.
	cached_cntx_nat = NULL;
	cached_cntx_ind = NULL;
#endif
}

// -----------------------------------------------------------------------------

void bli_gks_init_index( void )
{
	// This function is called by bli_gks_init(). It simply initializes all
	// architecture id elements of the internal arrays to NULL.

	const size_t gks_size = sizeof( cntx_t* ) * BLIS_NUM_ARCHS;
	const size_t fpa_size = sizeof( void_fp ) * BLIS_NUM_ARCHS;

	// Set every entry in gks and context init function pointer arrays to
	// zero/NULL. This is done so that later on we know which ones were
	// allocated.
	memset( gks,           0, gks_size );
	memset( cntx_ref_init, 0, fpa_size );
	memset( cntx_ind_init, 0, fpa_size );
}

// -----------------------------------------------------------------------------

const cntx_t* bli_gks_lookup_nat_cntx
     (
       arch_t id
     )
{
	// Return the address of the (native) context for a given architecture id.
	// This function assumes the architecture has already been registered.

	return bli_gks_lookup_ind_cntx( id, BLIS_NAT );
}

// -----------------------------------------------------------------------------

const cntx_t* bli_gks_lookup_ind_cntx
     (
       arch_t id,
       ind_t  ind
     )
{
	// Return the address of the context for a given architecture id and
	// induced method. This function assumes the architecture has already
	// been registered. Note that this function returns NULL if the induced
	// method hasn't yet been called (and thus its context pointer is still
	// NULL).

	// Sanity check: verify that the arch_t id is valid.
	if ( bli_error_checking_is_enabled() )
	{
		err_t e_val = bli_check_valid_arch_id( id );
		bli_check_error_code( e_val );
	}

	// Index into the array of context pointers for the given architecture id,
	// and then index into the subarray for the given induced method.
	cntx_t** gks_id     = gks[ id ];
	cntx_t*  gks_id_ind = gks_id[ ind ];

	// Return the context pointer at gks_id_ind.
	return gks_id_ind;
}

// -----------------------------------------------------------------------------

const cntx_t* const * bli_gks_lookup_id
     (
       arch_t id
     )
{
	// Return the address of the array of context pointers for a given
	// architecture id. This function is only used for sanity check purposes
	// to ensure that the underlying data structures for a particular id are
	// initialized.

	// Index into the array of context pointers for the given architecture id.
	cntx_t** gks_id = gks[ id ];

	// Return the context pointer at gks_id_ind.
	return ( const cntx_t* const * )gks_id;
}

// -----------------------------------------------------------------------------

void bli_gks_register_cntx
     (
       arch_t  id,
       void_fp nat_fp,
       void_fp ref_fp,
       void_fp ind_fp
     )
{
	err_t r_val;

	// This function is called by bli_gks_init() for each architecture that
	// will be supported by BLIS. It takes an architecture id and three
	// function pointers, one to a function that initializes a native context
	// (supplied by the kernel developer), one to a function that initializes
	// a reference context (with function pointers specific to the architecture
	// associated with id), and one to a function that initializes a
	// context for use with induced methods (again, with function pointers
	// to the architecture). The latter two functions are automatically
	// generated by the framework. Unlike with native contexts, we don't
	// actually store the induced contexts until that induced method is
	// called, and we don't ever store reference contexts. For this reason, we
	// can get away with only storing the pointers to the initialization
	// functions for those latter two types of contexts, which we can then
	// call at a later time when those contexts are needed.

	// Sanity check: verify that the arch_t id is valid.
	if ( bli_error_checking_is_enabled() )
	{
		err_t e_val = bli_check_valid_arch_id( id );
		bli_check_error_code( e_val );
	}

	nat_cntx_init_ft f = nat_fp;

	// First, store the function pointers to the context initialization
	// functions for reference kernels and induced method execution. The
	// former will be used whenever we need to obtain reference kernels and
	// latter will be used later on if the user calls a level-3 function
	// with induced execution enabled.
	cntx_ref_init[ id ] = ref_fp;
	cntx_ind_init[ id ] = ind_fp;

	// If the the context array pointer isn't NULL, then it means the given
	// architecture id has already registered (and the underlying memory
	// allocations and context initializations have already been performed).
	// This is really just a safety feature to prevent memory leaks; this
	// early return should never occur, because the caller should never try
	// to register with an architecture id that has already been registered.
	if ( gks[ id ] != NULL ) return;

	#ifdef BLIS_ENABLE_MEM_TRACING
	printf( "bli_gks_register_cntx(): " );
	#endif

	// At this point, we know the pointer to the array of cntx_t* is NULL and
	// needs to be allocated. Allocate the memory and initialize it to
	// zeros/NULL, storing the address of the alloacted memory at the element
	// for the current architecture id.
	gks[ id ] = bli_calloc_intl( sizeof( cntx_t* ) * BLIS_NUM_IND_METHODS, &r_val );

	// Alias the allocated array for readability.
	cntx_t** gks_id = gks[ id ];

	#ifdef BLIS_ENABLE_MEM_TRACING
	printf( "bli_gks_register_cntx(): " );
	#endif

	// Allocate memory for a single context and store the address at
	// the element in the gks[ id ] array that is reserved for native
	// execution.
	gks_id[ BLIS_NAT ] = bli_calloc_intl( sizeof( cntx_t ), &r_val );

	// Alias the allocated context address for readability.
	cntx_t* gks_id_nat = gks_id[ BLIS_NAT ];

	// Call the context initialization function on the element of the newly
	// allocated array corresponding to native execution.
	f( gks_id_nat );

	// Verify that cache blocksizes are whole multiples of register blocksizes.
	// Specifically, verify that:
	//   - MC is a whole multiple of MR.
	//   - NC is a whole multiple of NR.
	//   - KC is a whole multiple of KR.
	// These constraints are enforced because it makes it easier to handle diagonals
	// in the macro-kernel implementations. Additionally, we optionally verify that:
	//   - MC is a whole multiple of NR.
	//   - NC is a whole multiple of MR.
	// These latter constraints, guarded by #ifndef BLIS_RELAX_MCNR_NCMR_CONSTRAINTS
	// below, are only enforced when we wish to be able to handle the trsm right-
	// side case handling that swaps A and B, so that B is the triangular matrix,
	// with NR blocking used to pack A and MR blocking used to pack B, with the
	// arguments to the gemmtrsm microkernel swapped at the last minute, as the
	// kernel is called.
	err_t e_val;

	const blksz_t* mc = bli_cntx_get_blksz( BLIS_MC, gks_id_nat );
	const blksz_t* nc = bli_cntx_get_blksz( BLIS_NC, gks_id_nat );
	const blksz_t* kc = bli_cntx_get_blksz( BLIS_KC, gks_id_nat );
	const blksz_t* mr = bli_cntx_get_blksz( BLIS_MR, gks_id_nat );
	const blksz_t* nr = bli_cntx_get_blksz( BLIS_NR, gks_id_nat );
	const blksz_t* kr = bli_cntx_get_blksz( BLIS_KR, gks_id_nat );

	e_val = bli_check_valid_mc_mod_mult( mc, mr ); bli_check_error_code( e_val );
	e_val = bli_check_valid_nc_mod_mult( nc, nr ); bli_check_error_code( e_val );
	e_val = bli_check_valid_kc_mod_mult( kc, kr ); bli_check_error_code( e_val );
#ifndef BLIS_RELAX_MCNR_NCMR_CONSTRAINTS
	e_val = bli_check_valid_mc_mod_mult( mc, nr ); bli_check_error_code( e_val );
	e_val = bli_check_valid_nc_mod_mult( nc, mr ); bli_check_error_code( e_val );
#endif

	// Verify that the register blocksizes in the context are sufficiently large
	// relative to the maximum stack buffer size defined at configure-time.
	e_val = bli_check_sufficient_stack_buf_size( gks_id_nat );
	bli_check_error_code( e_val );
}

// -----------------------------------------------------------------------------

const cntx_t* bli_gks_query_cntx( void )
{
	return bli_gks_query_nat_cntx();
}

// -----------------------------------------------------------------------------

const cntx_t* bli_gks_query_nat_cntx( void )
{
	bli_init_once();

#ifdef BLIS_ENABLE_GKS_CACHING

	// Return a pointer to the context for native execution that was deep-
	// queried and cached at the end of bli_gks_init().
	return cached_cntx_nat;

#else

	// Deep-query and return the address of a context for native execution.
	return bli_gks_query_nat_cntx_impl();

#endif
}

const cntx_t* bli_gks_query_nat_cntx_noinit( void )
{
	// NOTE: This function purposefully avoids calling bli_init_once() so that
	// it is safe to call during inititalization.

	return bli_gks_query_nat_cntx_impl();
}

// -----------------------------------------------------------------------------

const cntx_t* bli_gks_query_nat_cntx_impl( void )
{
	// Return the address of the native context for the architecture id
	// corresponding to the current hardware, as determined by
	// bli_arch_query_id().

	// Query the architecture id.
	arch_t id = bli_arch_query_id();

	// Use the architecture id to look up a pointer to its context.
	const cntx_t* cntx = bli_gks_lookup_nat_cntx( id );

	return cntx;
}

// -----------------------------------------------------------------------------

const cntx_t* bli_gks_query_ind_cntx
     (
       ind_t ind
     )
{
	bli_init_once();

#ifdef BLIS_ENABLE_GKS_CACHING

	// If for some reason the native context was requested, we return its
	// address instead of the one for induced execution.
	if ( ind == BLIS_NAT ) return cached_cntx_nat;

	// Return a pointer to the context for the induced method that was deep-
	// queried and cached at the end of bli_gks_init().
	return cached_cntx_ind;

#else

	// Deep-query and return the address of a context for the requested induced
	// method. (In this case, caching never takes place since it was disabled
	// at configure-time.)
	return bli_gks_query_ind_cntx_impl( ind );

#endif
}

const cntx_t* bli_gks_query_ind_cntx_noinit
     (
       ind_t ind
     )
{
	// NOTE: This function purposefully avoids calling bli_init_once() so that
	// it is safe to call during inititalization.

	return bli_gks_query_ind_cntx_impl( ind );
}

// -----------------------------------------------------------------------------

// A mutex to allow synchronous access to the gks when it needs to be updated
// with a new entry corresponding to a context for an ind_t value.
static bli_pthread_mutex_t gks_mutex = BLIS_PTHREAD_MUTEX_INITIALIZER;

const cntx_t* bli_gks_query_ind_cntx_impl
     (
       ind_t ind
     )
{
	cntx_t* gks_id_ind;
	err_t r_val;


	// Return the address of a context that will be suited for executing a
	// level-3 operation via the requested induced method (and datatype) for
	// the architecture id corresponding to the current hardware, as
	// determined by bli_arch_query_id().

	// This function is called when a level-3 operation via induced method is
	// called, e.g. bli_gemm1m(). If this is the first time that induced method
	// is being executed since bli_gks_init(), the necessary context structure
	// is allocated. If this is not the first time a context for the requested
	// induced method was queried, then the memory will already be allocated
	// and initialized, and the previous cntx_t struct will be overwritten.
	// The function will then return the address to the newly-initialized (or
	// previously-allocated-but-reinitialized) cntx_t struct. Note that some of
	// this function must be executed with mutual exclusion to ensure thread
	// safety and deterministic behavior.

	// Query the architecture id.
	arch_t id = bli_arch_query_id();

	// Sanity check: verify that the arch_t id is valid.
	if ( bli_error_checking_is_enabled() )
	{
		err_t e_val = bli_check_valid_arch_id( id );
		bli_check_error_code( e_val );
	}

	// NOTE: These initial statements can reside outside of the critical section
	// because gks[ id ] should have already been allocated, and the native
	// context in that array should have already been allocated/initialized.

	// Query the gks for the array of context pointers corresponding to the
	// given architecture id.
	cntx_t** gks_id     = gks[ id ];
	cntx_t*  gks_id_nat = gks_id[ BLIS_NAT ];

	// If for some reason the native context was requested, we can return
	// its address early.
	if ( ind == BLIS_NAT ) return gks_id_nat;

	// This function assumes that the architecture idenified by id has
	// already been registered with the gks (which guarantees that
	// gks[ id ] is non-NULL and gks[ id ][ BLIS_NAT ] is also non-NULL
	// and refers to a context initialized with valid data).

	// Acquire the mutex protecting the gks.
	bli_pthread_mutex_lock( &gks_mutex );

	// BEGIN CRITICAL SECTION
	{
		// Alias for readability the element of gks_id associated with the
		// requested induced method.
		gks_id_ind = gks_id[ ind ];

		// If the context pointer is NULL, then we know we must allocate and
		// then initialize the context before returning its address.
		if ( gks_id_ind == NULL )
		{
			// If gks_id_ind is NULL, then we know we must allocate and then
			// initialize the context, storing its address back to
			// gks_id[ ind ].
			gks_id_ind    = bli_calloc_intl( sizeof( cntx_t ), &r_val );
			gks_id[ ind ] = gks_id_ind;
		}

		// Before we can call the induced method context initialization
		// function on the newly allocated structure, we must first copy
		// over the contents of the native context. If a previous context
		// was already copied, this will overwrite those previous values.
		*gks_id_ind = *gks_id_nat;

		// Use the architecture id to look up the function pointer to the
		// context initialization function for induced methods.
		ind_cntx_init_ft f = cntx_ind_init[ id ];

		// Now we modify the context (so that it contains the proper values
		// for its induced method) by calling the context initialization
		// function for the current induced method. (That function assumes
		// that the context is pre-initialized with values for native
		// execution.)
		f( ind, gks_id_ind );
	}
	// END CRITICAL SECTION

	// Release the mutex protecting the gks.
	bli_pthread_mutex_unlock( &gks_mutex );

	// Return the address of the newly-allocated/initialized context.
	return gks_id_ind;

}

// -----------------------------------------------------------------------------

void bli_gks_init_ref_cntx
    (
      cntx_t* cntx
    )
{
	// Query the architecture id.
	arch_t id = bli_arch_query_id();

	// Sanity check: verify that the arch_t id is valid.
	if ( bli_error_checking_is_enabled() )
	{
		err_t e_val = bli_check_valid_arch_id( id );
		bli_check_error_code( e_val );
	}

	// Obtain the function pointer to the context initialization function for
	// reference kernels.
	ref_cntx_init_ft f = cntx_ref_init[ id ];

	// Initialize the caller's context with reference kernels and related values.
	f( cntx );
}

// -----------------------------------------------------------------------------

bool bli_gks_cntx_l3_nat_ukr_is_ref
     (
             num_t   dt,
             ukr_t   ukr_id,
       const cntx_t* cntx
     )
{
	cntx_t ref_cntx;

	// Initialize a context with reference kernels for the arch_t id queried
	// via bli_arch_query_id().
	bli_gks_init_ref_cntx( &ref_cntx );

	// Query each context for the micro-kernel function pointer for the
	// specified datatype.
	void_fp ref_fp = bli_cntx_get_ukr_dt( dt, ukr_id, &ref_cntx );
	void_fp fp     = bli_cntx_get_ukr_dt( dt, ukr_id, cntx );

	// Return the result.
	return fp == ref_fp;
}

//
// -- level-3 micro-kernel implementation strings ------------------------------
//

static const char* bli_gks_l3_ukr_impl_str[BLIS_NUM_UKR_IMPL_TYPES] =
{
	"refrnce",
	"virtual",
	"optimzd",
	"notappl",
};

// -----------------------------------------------------------------------------

const char* bli_gks_l3_ukr_impl_string( ukr_t ukr, ind_t method, num_t dt )
{
	kimpl_t ki;

	// Query the context for the current induced method and datatype, and
	// then query the ukernel function pointer for the given datatype from
	// that context.
	const cntx_t* cntx = bli_gks_query_ind_cntx( method );
	void_fp fp         = bli_cntx_get_ukr_dt( dt, ukr, cntx );

	// Check whether the ukernel function pointer is NULL for the given
	// datatype. If it is NULL, return the string for not applicable.
	// Otherwise, query the ukernel implementation type using the method
	// provided and return the associated string.
	if ( fp == NULL )
		ki = BLIS_NOTAPPLIC_UKERNEL;
	else
		ki = bli_gks_l3_ukr_impl_type( ukr, method, dt );

	return bli_gks_l3_ukr_impl_str[ ki ];
}

#if 0
char* bli_gks_l3_ukr_avail_impl_string( ukr_t ukr, num_t dt )
{
	opid_t  oper;
	ind_t   method;
	kimpl_t ki;

	// We need to decide which operation we will use to query the
	// current available induced method. If the ukr type given is
	// BLIS_GEMM_UKR, we use gemm. Otherwise, we use trsm (since
	// the four other defined ukr types are trsm-related).
	if ( ukr == BLIS_GEMM_UKR ) oper = BLIS_GEMM;
	else                        oper = BLIS_TRSM;

	// Query the current available induced method using the
	// chosen operation id type.
	method = bli_l3_ind_oper_find_avail( oper, dt );

	// Query the ukernel implementation type using the current
	// available method.
	ki = bli_gks_l3_ukr_impl_type( ukr, method, dt );

	return bli_ukr_impl_str[ ki ];
}
#endif

kimpl_t bli_gks_l3_ukr_impl_type( ukr_t ukr, ind_t method, num_t dt )
{
	// If the current available induced method is not native, it
	// must be virtual.
	if ( method != BLIS_NAT ) return BLIS_VIRTUAL_UKERNEL;
	else
	{
		// If the current available induced method for the gemm
		// operation is native, then it might be reference or
		// optimized. To determine which, we compare the
		// datatype-specific function pointer within the ukrs
		// object corresponding to the current available induced
		// method to the typed function pointer within the known
		// reference ukrs object.

		// Query the architecture id.
		arch_t id = bli_arch_query_id();

		// Sanity check: verify that the arch_t id is valid.
		if ( bli_error_checking_is_enabled() )
		{
			err_t e_val = bli_check_valid_arch_id( id );
			bli_check_error_code( e_val );
		}

		// Query the native context from the gks.
		const cntx_t* nat_cntx = bli_gks_lookup_nat_cntx( id );

		if ( bli_gks_cntx_l3_nat_ukr_is_ref( dt, ukr, nat_cntx ) )
			return BLIS_REFERENCE_UKERNEL;
		else
			return BLIS_OPTIMIZED_UKERNEL;
	}
}

