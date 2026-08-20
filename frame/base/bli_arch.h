/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2014, The University of Texas at Austin
   Copyright (C) 2018 - 2026, Advanced Micro Devices, Inc. All rights reserved.

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

#ifndef BLIS_ARCH_H
#define BLIS_ARCH_H

BLIS_EXPORT_BLIS bool bli_aocl_enable_instruction_query( void );

BLIS_EXPORT_BLIS arch_t bli_arch_query_id( void );

BLIS_EXPORT_BLIS model_t bli_model_query_id( void );
BLIS_EXPORT_BLIS model_t bli_init_model_query_id( void );

BLIS_EXPORT_BLIS char*  bli_arch_string( arch_t id );
BLIS_EXPORT_BLIS char*  bli_model_string( model_t id );

// Map a resolved arch_t to a coarse ISA capability tier for kernel dispatch.
//
// IMPORTANT: this is the single source of truth for "which SIMD tier does
// this Zen arch belong to". When a new Zen arch is added to arch_t, add its
// one case here -- do NOT reintroduce raw arch_id comparisons in kernel
// files. A switch (rather than a range test on the enum's numeric values)
// is used deliberately: the arch_t comment warns that values may be
// added/inserted OR rearranged, so an unrecognized/future arch falls
// through to BLIS_ISA_TIER_GENERIC (correct-but-unoptimized) instead of
// being silently misrouted to the wrong kernel.
BLIS_INLINE isa_tier_t bli_arch_isa_tier( arch_t arch_id )
{
	switch ( arch_id )
	{
		case BLIS_ARCH_ZEN6:
		case BLIS_ARCH_ZEN5:
		case BLIS_ARCH_ZEN4:
			return BLIS_ISA_TIER_AVX512;

		case BLIS_ARCH_ZEN3:
		case BLIS_ARCH_ZEN2:
		case BLIS_ARCH_ZEN:
			return BLIS_ISA_TIER_AVX2;

		// Intel / ARM / Power / GENERIC / ERROR -> reference path.
		default:
			return BLIS_ISA_TIER_GENERIC;
	}
}

#if defined(BLIS_IS_BUILDING_LIBRARY) || defined(BLIS_CONFIGURETIME_CPUID)

extern arch_t g_arch_id;
extern model_t g_model_id;

extern bli_pthread_once_t once_id_check;
extern bli_pthread_once_t once_id_init;

void bli_arch_set_id( void );
void bli_arch_check_id( void );

void bli_arch_set_logging( bool dolog );
bool bli_arch_get_logging( void );
void bli_arch_log( char*, ... );

BLIS_INLINE arch_t bli_arch_query_id_internal( void )
{

#if defined BLIS_FAMILY_INTEL64      || \
    defined BLIS_FAMILY_AMDZEN       || \
    defined BLIS_FAMILY_AMD64_LEGACY || \
    defined BLIS_FAMILY_X86_64       || \
    defined BLIS_FAMILY_ARM64        || \
    defined BLIS_FAMILY_ARM32

	// For builds with multiple sub-configurations use the global value
	// that will reflect dynamic dispatch, subject to any user override
	// via environment variables.
  #ifndef BLIS_CONFIGURETIME_CPUID
	bli_pthread_once( &once_id_check, bli_arch_check_id );
  #endif
	// Simply return the id that was previously cached.
	return g_arch_id;

#else

  #if defined BLIS_FAMILY_TO_ARCH_VALUE
	// For single sub-configuration builds, get value from header file
	arch_t l_arch_id = BLIS_FAMILY_TO_ARCH_VALUE;
  #elif defined BLIS_CONFIGURETIME_CPUID
	// For "auto" build, initialize BLIS_FAMILY_TO_ARCH_VALUE to
	// generic as starting point for use in architecture detection.
	// BLIS will then determine the correct architecture and get
	// the correct BLIS_FAMILY_TO_ARCH_VALUE from the relevant
	// sub-configuration header file.
	arch_t l_arch_id = BLIS_ARCH_GENERIC;
  #else
	// No fallback if BLIS_FAMILY_TO_ARCH_VALUE is not set in
	// the relevant config bli_family header file
	#error "BLIS_FAMILY_TO_ARCH_VALUE not defined in relevant config bli_family header file"
  #endif
	return l_arch_id;

#endif

}

BLIS_INLINE model_t bli_model_query_id_internal( void )
{
#ifndef BLIS_CONFIGURETIME_CPUID
	bli_pthread_once( &once_id_check, bli_arch_check_id );
#endif
	// Simply return the model_id that was previously cached.
	return g_model_id;
}

BLIS_INLINE model_t bli_init_model_query_id_internal( void )
{
#ifndef BLIS_CONFIGURETIME_CPUID
	bli_pthread_once( &once_id_init, bli_arch_set_id );
#endif
	// Simply return the model_id that was previously cached.
	return g_model_id;
}

#endif // BLIS_IS_BUILDING_LIBRARY

#endif // BLIS_ARCH_H

