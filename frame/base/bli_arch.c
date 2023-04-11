/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2014, The University of Texas at Austin
   Copyright (C) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

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
  #define BLIS_INLINE static
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

// The arch and model ids for the currently running hardware. We initialize
// to -1, which will be overwritten upon calling bli_arch_set_id().
static arch_t actual_arch_id = -1;
static model_t actual_model_id = -1;

// The arch and model ids for the currently running hardware, or the values
// the user specifies to use. We initialize to -1, which will be overwritten
// upon calling bli_arch_set_id().
static arch_t arch_id = -1;
static model_t model_id = -1;

// Variable used to communicate if user has set '__blis_arch_type_name' between
// bli_arch_set_id() and bli_arch_check_id()
static dim_t __attribute__ ((unused)) req_id = -1;

arch_t bli_arch_query_id( void )
{
	bli_arch_set_id_once();
	bli_arch_check_id_once();

	// Simply return the id that was previously cached.
	return arch_id;
}

model_t bli_model_query_id( void )
{
	bli_arch_set_id_once();
	bli_arch_check_id_once();

	// Simply return the model_id that was previously cached.
	return model_id;
}

model_t bli_init_model_query_id( void )
{
	bli_arch_set_id_once();

	// Simply return the model_id that was previously cached.
	return model_id;
}

// -----------------------------------------------------------------------------

// A pthread structure used in pthread_once(). pthread_once() is guaranteed to
// execute exactly once among all threads that pass in this control object.
static bli_pthread_once_t once_id_init = BLIS_PTHREAD_ONCE_INIT;
static bli_pthread_once_t once_id_check = BLIS_PTHREAD_ONCE_INIT;

void bli_arch_set_id_once( void )
{
#ifndef BLIS_CONFIGURETIME_CPUID
	bli_pthread_once( &once_id_init, bli_arch_set_id );
#endif
}

void bli_arch_check_id_once( void )
{
#ifndef BLIS_CONFIGURETIME_CPUID
	bli_pthread_once( &once_id_check, bli_arch_check_id );
#endif
}

// -----------------------------------------------------------------------------

void bli_arch_set_id( void )
{
	// Check the environment variable BLIS_ARCH_DEBUG to see if the user
	// requested that we echo the result of the subconfiguration selection.
	bool do_logging = bli_env_get_var( "BLIS_ARCH_DEBUG", 0 );
	bli_arch_set_logging( do_logging );

	// Get actual hardware arch and model ids.
	actual_arch_id = bli_cpuid_query_id();
	actual_model_id = bli_cpuid_query_model_id( actual_arch_id );

	// DISABLE_BLIS_ARCH_TYPE and BLIS_CONFIGURETIME_CPUID seem similar but
	// have different use cases:
	// * BLIS_CONFIGURETIME_CPUID is used by the "configure auto" option to
	//   select a single code path, and affects other parts of the code.
	// * DISABLE_BLIS_ARCH_TYPE disables user selection of code path here in
	//   builds with multiple code paths.

#ifndef DISABLE_BLIS_ARCH_TYPE
	// Check the environment variable (that "__blis_arch_type_name" is
	// defined to be) to see if the user requested that we use a specific
	// subconfiguration. "__blis_arch_type_name" will be defined by the
	// configure command in bli_config.h, with the default name of BLIS_ARCH_TYPE
	req_id = bli_env_get_var_arch_type( __blis_arch_type_name, -1 );

#ifndef BLIS_CONFIGURETIME_CPUID
	if ( req_id != -1 )
	{
		// BLIS_ARCH_TYPE was set. Cautiously check whether its value is usable.

		// If req_id was set to an invalid arch_t value (ie: outside the range
		// [1,BLIS_NUM_ARCHS-1]), output an error message and abort.
		if ( bli_error_checking_is_enabled() )
		{
			err_t e_val = bli_check_valid_arch_id( req_id );
			bli_check_error_code( e_val );
		}

		// Check again context actually initialized deferred to
		// bli_arch_check_id() called later.

		// For now, we can only be confident that req_id is in range.
		arch_id = req_id;
	}
	else
#endif

#endif
	{
		// BLIS_ARCH_TYPE was unset. Proceed with normal subconfiguration
		// selection behavior.

		// Architecture families.
		#if defined BLIS_FAMILY_INTEL64      || \
		    defined BLIS_FAMILY_AMDZEN       || \
		    defined BLIS_FAMILY_AMD64_LEGACY || \
		    defined BLIS_FAMILY_X86_64       || \
		    defined BLIS_FAMILY_ARM64        || \
		    defined BLIS_FAMILY_ARM32
		arch_id = actual_arch_id;
		#endif

		// Intel microarchitectures.
		#ifdef BLIS_FAMILY_SKX
		arch_id = BLIS_ARCH_SKX;
		#endif
		#ifdef BLIS_FAMILY_KNL
		arch_id = BLIS_ARCH_KNL;
		#endif
		#ifdef BLIS_FAMILY_KNC
		arch_id = BLIS_ARCH_KNC;
		#endif
		#ifdef BLIS_FAMILY_HASWELL
		arch_id = BLIS_ARCH_HASWELL;
		#endif
		#ifdef BLIS_FAMILY_SANDYBRIDGE
		arch_id = BLIS_ARCH_SANDYBRIDGE;
		#endif
		#ifdef BLIS_FAMILY_PENRYN
		arch_id = BLIS_ARCH_PENRYN;
		#endif

		// AMD microarchitectures.
		#ifdef BLIS_FAMILY_ZEN4
		arch_id = BLIS_ARCH_ZEN4;
		#endif
		#ifdef BLIS_FAMILY_ZEN3
		arch_id = BLIS_ARCH_ZEN3;
		#endif
		#ifdef BLIS_FAMILY_ZEN2
		arch_id = BLIS_ARCH_ZEN2;
		#endif
		#ifdef BLIS_FAMILY_ZEN
		arch_id = BLIS_ARCH_ZEN;
		#endif
		#ifdef BLIS_FAMILY_EXCAVATOR
		arch_id = BLIS_ARCH_EXCAVATOR;
		#endif
		#ifdef BLIS_FAMILY_STEAMROLLER
		arch_id = BLIS_ARCH_STEAMROLLER;
		#endif
		#ifdef BLIS_FAMILY_PILEDRIVER
		arch_id = BLIS_ARCH_PILEDRIVER;
		#endif
		#ifdef BLIS_FAMILY_BULLDOZER
		arch_id = BLIS_ARCH_BULLDOZER;
		#endif

		// ARM microarchitectures.
		#ifdef BLIS_FAMILY_THUNDERX2
		arch_id = BLIS_ARCH_THUNDERX2;
		#endif
		#ifdef BLIS_FAMILY_CORTEXA57
		arch_id = BLIS_ARCH_CORTEXA57;
		#endif
		#ifdef BLIS_FAMILY_CORTEXA53
		arch_id = BLIS_ARCH_CORTEXA53;
		#endif
		#ifdef BLIS_FAMILY_CORTEXA15
		arch_id = BLIS_ARCH_CORTEXA15;
		#endif
		#ifdef BLIS_FAMILY_CORTEXA9
		arch_id = BLIS_ARCH_CORTEXA9;
		#endif

		// IBM microarchitectures.
		#ifdef BLIS_FAMILY_POWER10
		arch_id = BLIS_ARCH_POWER10;
		#endif
		#ifdef BLIS_FAMILY_POWER9
		arch_id = BLIS_ARCH_POWER9;
		#endif
		#ifdef BLIS_FAMILY_POWER7
		arch_id = BLIS_ARCH_POWER7;
		#endif
		#ifdef BLIS_FAMILY_BGQ
		arch_id = BLIS_ARCH_BGQ;
		#endif

		// Generic microarchitecture.
		#ifdef BLIS_FAMILY_GENERIC
		arch_id = BLIS_ARCH_GENERIC;
		#endif
	}


#ifndef DISABLE_BLIS_MODEL_TYPE
	// Check the environment variable (that "__blis_model_type_name" is
	// defined to be) to see if the user requested that we use a specific
	// subconfiguration. "__blis_model_type_name" will be defined by the
	// configure command in bli_config.h, with the default name of BLIS_MODEL_TYPE
	dim_t req_model = bli_env_get_var_model_type( __blis_model_type_name, -1 );

#ifndef BLIS_CONFIGURETIME_CPUID
	if ( req_model != -1 )
	{
		// BLIS_MODEL_TYPE was set. Cautiously check whether its value is usable.
		// Assume here that arch_id is valid.

		// If req_model was set to an invalid model_t value (ie: both outside
		// the range appropriate for the given architecture and not default),
		// set to default value and continue.
		if ( bli_error_checking_is_enabled() )
		{
			err_t e_val = bli_check_valid_model_id( arch_id, req_model );
			if (e_val != BLIS_SUCCESS)
			{
				req_model = BLIS_MODEL_DEFAULT;
				e_val = BLIS_SUCCESS;
			}
			bli_check_error_code( e_val );
		}

		// We can now be confident that req_model is in range for the
		// selected architecture, or it has been reset to be default.
		model_id = req_model;
	}
	else
#endif

#endif
	{
		// BLIS_MODEL_TYPE was unset. Proceed with normal subconfiguration
		// selection behavior, based on value of architecture id selected
		// above. Unlike for arch_id, we cannot simply use actual_model_id
		// here, as we need to choose model_id based on the arch_id we are
		// using, which could be different to actual_arch_id.

		model_id = bli_cpuid_query_model_id( arch_id );
	}

	//printf( "blis_arch_query_id(): arch_id, model_id = %u, %u\n", arch_id, model_id );
	//exit(1);
}

void bli_arch_check_id( void )
{
	bli_arch_set_id_once();

	// Check arch value against configured options. Only needed
	// if user has set it. This function will also do the
	// logging of chosen arch and model (if desired).

	// DISABLE_BLIS_ARCH_TYPE and BLIS_CONFIGURETIME_CPUID seem similar but
	// have different use cases:
	// * BLIS_CONFIGURETIME_CPUID is used by the "configure auto" option to
	//   select a single code path, and affects other parts of the code.
	// * DISABLE_BLIS_ARCH_TYPE disables user selection of code path here in
	//   builds with multiple code paths.

#ifndef DISABLE_BLIS_ARCH_TYPE

#ifndef BLIS_CONFIGURETIME_CPUID
	if ( req_id != -1 )
	{
		// BLIS_ARCH_TYPE was set. Cautiously check whether its value is usable.

		// In BLAS1 and BLAS2 routines, bli_init_auto() may not have been
		// called, so ensure cntx has been initialized here.
		bli_gks_init_once();

		// At this point, we know that req_id is in the valid range, but we
		// don't yet know if it refers to a context that was actually
		// initialized. Query the address of an internal context data structure
		// corresponding to req_id. This pointer will be NULL if the associated
		// subconfig is not available.
		cntx_t** req_cntx = bli_gks_lookup_id( req_id );

		// This function checks the context pointer and aborts with a useful
		// error message if the pointer is found to be NULL.
		if ( bli_error_checking_is_enabled() )
		{
			err_t e_val = bli_check_initialized_gks_cntx( req_cntx );
			bli_check_error_code( e_val );
		}

		// Finally, we can be confident that req_id (1) is in range and (2)
		// refers to a context that has been initialized.
		arch_id = req_id;
	}
#endif

#endif

	if ( bli_arch_get_logging() )
        {
		if ( model_id == BLIS_MODEL_DEFAULT )
		{
			fprintf( stderr, "libblis: selecting sub-configuration '%s'.\n",
				 bli_arch_string( arch_id ) );
		}
		else
		{
			fprintf( stderr, "libblis: selecting sub-configuration '%s', model '%s'.\n",
				 bli_arch_string( arch_id ), bli_model_string( model_id ) );
		}
        }

	//printf( "blis_arch_check_id(): arch_id, model_id = %u, %u\n", arch_id, model_id );
	//exit(1);
}

// -----------------------------------------------------------------------------

// NOTE: This string array must be kept up-to-date with the arch_t
// enumeration that is typedef'ed in bli_type_defs.h. That is, the
// index order of each string should correspond to the implied/assigned
// enum value given to the corresponding BLIS_ARCH_ value.
// This must also be kept up-to-date with the bli_env_get_var_arch_type()
// function in bli_env.c
static char* config_name[ BLIS_NUM_ARCHS ] =
{
    "error",

    "generic",

    "skx",
    "knl",
    "knc",
    "haswell",
    "sandybridge",
    "penryn",

    "zen4",
    "zen3",
    "zen2",
    "zen",
    "excavator",
    "steamroller",
    "piledriver",
    "bulldozer",

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

char* bli_arch_string( arch_t id )
{
	return config_name[ id ];
}

// NOTE: This string array must be kept up-to-date with the model_t
// enumeration that is typedef'ed in bli_type_defs.h. That is, the
// index order of each string should correspond to the implied/assigned
// enum value given to the corresponding BLIS_model_ value.
// This must also be kept up-to-date with the bli_env_get_var_model_type()
// function in bli_env.c
static char* model_name[ BLIS_NUM_MODELS ] =
{
    "error",

    "default",

    "Genoa",
    "Bergamo",
    "Genoa-X",

    "Milan",
    "Milan-X",

    "default"
};

char* bli_model_string( model_t id )
{
	return model_name[ id ];
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

void bli_arch_log( char* fmt, ... )
{
	char prefix[] = "libblis: ";
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

