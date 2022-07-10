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

#ifdef BLIS_CONFIGURETIME_CPUID

  // NOTE: If you need to make any changes to this cpp branch, it's probably
  // the case that you also need to modify bli_arch.c, bli_cpuid.c, and
  // bli_env.c. Don't forget to update these other files as needed!

  // The BLIS_ENABLE_SYSTEM macro must be defined so that the correct cpp
  // branch in bli_system.h is processed. (This macro is normally defined in
  // bli_config.h.)
  #define BLIS_ENABLE_SYSTEM

  // Use C-style static inline functions for any static inline functions that
  // happen to be defined by the headers below. (This macro is normally defined
  // in bli_config_macro_defs.h.)
  #define BLIS_INLINE static

  // Since we're not building a shared library, we can forgo the use of the
  // BLIS_EXPORT_BLIS annotations by #defining them to be nothing. (This macro
  // is normally defined in bli_config_macro_defs.h.)
  #define BLIS_EXPORT_BLIS

  #include "bli_system.h"
  #include "bli_type_defs.h"
  #include "bli_arch.h"
  #include "bli_cpuid.h"
  #include "bli_env.h"
#else
  #include "blis.h"
#endif

// -----------------------------------------------------------------------------

// The arch_t id for the currently running hardware. We initialize to -1,
// which will be overwritten upon calling bli_arch_set_id().
static arch_t the_id = -1;

err_t bli_arch_query_id( arch_t* id )
{
#ifndef BLIS_CONFIGURETIME_CPUID

	// Make sure the arch_t id has been set.
	err_t r_val = bli_arch_set_id_once();
	bli_check_return_if_failure( r_val );

#else

	// configure's cpuid driver doesn't look at error codes, so we don't
	// bother returning it.
	bli_arch_set_id_once();

#endif

	// Set the arch_t pointer.
	*id = the_id;

	return BLIS_SUCCESS;
}

// -----------------------------------------------------------------------------

static bli_pthread_mutex_t arch_id_mutex = BLIS_PTHREAD_MUTEX_INITIALIZER;

err_t bli_arch_set_id_once( void )
{
#ifndef BLIS_CONFIGURETIME_CPUID

	err_t r_val = BLIS_SUCCESS;

	// If the arch_t id hasn't already been set, continue to the
	// mutex acquisition step.
	if ( the_id == -1 )
	{
		// Acquire the mutex protecting initialization.
		bli_pthread_mutex_lock( &arch_id_mutex );

		// Check the arch_t id again now that we've obtained the lock.
		if ( the_id == -1 )
		{
			r_val = bli_arch_set_id();
		}

		// Release the mutex protecting initialization.
		bli_pthread_mutex_unlock( &arch_id_mutex );

		// If the previous function failed, return the error code immediately.
		bli_check_return_if_failure( r_val );
	}

#else

	// configure's cpuid driver doesn't look at error codes, so we don't
	// bother returning it.
	bli_arch_set_id();

#endif

	return BLIS_SUCCESS;
}

// -----------------------------------------------------------------------------

err_t bli_arch_set_id( void )
{
	// Check the environment variable BLIS_ARCH_DEBUG to see if the user
	// requested that we echo the result of the subconfiguration selection.
	bool do_logging = bli_env_get_var( "BLIS_ARCH_DEBUG", 0 );
	bli_arch_set_logging( do_logging );

	// Check the environment variable BLIS_ARCH_TYPE to see if the user
	// requested that we use a specific subconfiguration.
	dim_t req_id = bli_env_get_var( "BLIS_ARCH_TYPE", -1 );

#ifndef BLIS_CONFIGURETIME_CPUID
	// If req_id is -1, it indicates that BLIS_ARCH_TYPE was NOT set. This means
	// we must query the hardware for the id (or read whatever value was hard-
	// coded at configure-time).
	if ( req_id == -1 )
#endif
	{
		// BLIS_ARCH_TYPE was unset. Proceed with normal subconfiguration
		// selection behavior.

		// Architecture families.
		#if defined BLIS_FAMILY_INTEL64 || \
		    defined BLIS_FAMILY_AMD64   || \
		    defined BLIS_FAMILY_X86_64  || \
		    defined BLIS_FAMILY_ARM64   || \
		    defined BLIS_FAMILY_ARM32
		req_id = bli_cpuid_query_id();
		#endif

		// Intel microarchitectures.
		#ifdef BLIS_FAMILY_SKX
		req_id = BLIS_ARCH_SKX;
		#endif
		#ifdef BLIS_FAMILY_KNL
		req_id = BLIS_ARCH_KNL;
		#endif
		#ifdef BLIS_FAMILY_KNC
		req_id = BLIS_ARCH_KNC;
		#endif
		#ifdef BLIS_FAMILY_HASWELL
		req_id = BLIS_ARCH_HASWELL;
		#endif
		#ifdef BLIS_FAMILY_SANDYBRIDGE
		req_id = BLIS_ARCH_SANDYBRIDGE;
		#endif
		#ifdef BLIS_FAMILY_PENRYN
		req_id = BLIS_ARCH_PENRYN;
		#endif

		// AMD microarchitectures.
		#ifdef BLIS_FAMILY_ZEN3
		req_id = BLIS_ARCH_ZEN3;
		#endif
		#ifdef BLIS_FAMILY_ZEN2
		req_id = BLIS_ARCH_ZEN2;
		#endif
		#ifdef BLIS_FAMILY_ZEN
		req_id = BLIS_ARCH_ZEN;
		#endif
		#ifdef BLIS_FAMILY_EXCAVATOR
		req_id = BLIS_ARCH_EXCAVATOR;
		#endif
		#ifdef BLIS_FAMILY_STEAMROLLER
		req_id = BLIS_ARCH_STEAMROLLER;
		#endif
		#ifdef BLIS_FAMILY_PILEDRIVER
		req_id = BLIS_ARCH_PILEDRIVER;
		#endif
		#ifdef BLIS_FAMILY_BULLDOZER
		req_id = BLIS_ARCH_BULLDOZER;
		#endif

		// ARM microarchitectures.
		#ifdef BLIS_FAMILY_ARMSVE
		req_id = BLIS_ARCH_ARMSVE;
		#endif
		#ifdef BLIS_FAMILY_A64FX
		req_id = BLIS_ARCH_A64FX;
		#endif
		#ifdef BLIS_FAMILY_FIRESTORM
		req_id = BLIS_ARCH_FIRESTORM;
		#endif
		#ifdef BLIS_FAMILY_THUNDERX2
		req_id = BLIS_ARCH_THUNDERX2;
		#endif
		#ifdef BLIS_FAMILY_CORTEXA57
		req_id = BLIS_ARCH_CORTEXA57;
		#endif
		#ifdef BLIS_FAMILY_CORTEXA53
		req_id = BLIS_ARCH_CORTEXA53;
		#endif
		#ifdef BLIS_FAMILY_CORTEXA15
		req_id = BLIS_ARCH_CORTEXA15;
		#endif
		#ifdef BLIS_FAMILY_CORTEXA9
		req_id = BLIS_ARCH_CORTEXA9;
		#endif

		// IBM microarchitectures.
		#ifdef BLIS_FAMILY_POWER10
		req_id = BLIS_ARCH_POWER10;
		#endif
		#ifdef BLIS_FAMILY_POWER9
		req_id = BLIS_ARCH_POWER9;
		#endif
		#ifdef BLIS_FAMILY_POWER7
		req_id = BLIS_ARCH_POWER7;
		#endif
		#ifdef BLIS_FAMILY_BGQ
		req_id = BLIS_ARCH_BGQ;
		#endif

		// Generic microarchitecture.
		#ifdef BLIS_FAMILY_GENERIC
		req_id = BLIS_ARCH_GENERIC;
		#endif
	}

#ifndef BLIS_CONFIGURETIME_CPUID
	// If req_id was set to an invalid arch_t value (ie: outside the range
	// [0,BLIS_NUM_ARCHS-1]), output an error message and abort.
	if ( bli_error_checking_is_enabled() )
	{
		err_t e_val = bli_check_valid_arch_id( req_id );
		bli_check_return_error_code( e_val );
	}

	// At this point, we know that req_id is in the valid range, but we
	// don't yet know if it refers to a context that was actually
	// initialized. Query the address of an internal context data structure
	// corresponding to req_id. This pointer will be NULL if the associated
	// subconfig is not available.
	const cntx_t* const * req_cntx = bli_gks_lookup_id( req_id );

	// This function checks the context pointer and aborts with a useful
	// error message if the pointer is found to be NULL.
	if ( bli_error_checking_is_enabled() )
	{
		err_t e_val = bli_check_initialized_gks_cntx( req_cntx );
		bli_check_return_error_code( e_val );
	}
#endif

	// Finally, we can be confident that req_id (1) is in range and (2)
	// refers to a context that has been initialized. So we cache it to
	// the local static variable.
	the_id = req_id;

	//printf( "blis_arch_query_id(): the_id = %u\n", id );
	//exit(1);

	if ( bli_arch_get_logging() )
	{
		// Query the string associated with the detected/chosen arch_t id.
		// Note that we don't need to check the error code returned by
		// bli_arch_string() since we've already confirmed that the arch_t
		// id is valid.
		const char* str;
		bli_arch_string( the_id, &str );

		fprintf( stderr, "libblis: selecting sub-configuration '%s'.\n", str );
	}

	return BLIS_SUCCESS;
}

// -----------------------------------------------------------------------------

// NOTE: This string array must be kept up-to-date with the arch_t
// enumeration that is typedef'ed in bli_type_defs.h. That is, the
// index order of each string should correspond to the implied/assigned
// enum value given to the corresponding BLIS_ARCH_ value.
static const char* config_name[ BLIS_NUM_ARCHS ] =
{
    "skx",
    "knl",
    "knc",
    "haswell",
    "sandybridge",
    "penryn",

    "zen3",
    "zen2",
    "zen",
    "excavator",
    "steamroller",
    "piledriver",
    "bulldozer",

    "armsve",
    "a64fx",
    "firestorm",
    "thunderx2",
    "cortexa57",
    "cortexa53",
    "cortexa15",
    "cortexa9",

    "power10",
    "power9",
    "power7",
    "bgq",

    "generic"
};

err_t bli_arch_string( arch_t id, const char** str )
{
#ifndef BLIS_CONFIGURETIME_CPUID
	err_t r_val = BLIS_SUCCESS;

	r_val = bli_check_valid_arch_id( id );
	bli_check_return_error_code( r_val );
#endif

	// If the caller passed in a valid id, index into the string array and
	// "return" the appropriate pointer.
	*str = config_name[ id ];

	return BLIS_SUCCESS;
}

// -----------------------------------------------------------------------------

static bool arch_dolog = 0;

void bli_arch_set_logging( bool dolog )
{
	arch_dolog = dolog;
}

bool bli_arch_get_logging( void )
{
	return arch_dolog;
}

void bli_arch_log( const char* fmt, ... )
{
	const char prefix[] = "libblis: ";
	int  n_chars  = strlen( prefix ) + strlen( fmt ) + 1;

	if ( bli_arch_get_logging() && fmt )
	{
		char* prefix_fmt = malloc( n_chars );

		snprintf( prefix_fmt, n_chars, "%s%s", prefix, fmt );

		va_list ap;
		va_start( ap, fmt );
		vfprintf( stderr, prefix_fmt, ap );
		va_end( ap );

		free( prefix_fmt );
	}
}

