/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2014, The University of Texas at Austin
   Copyright (C) 2018, Advanced Micro Devices, Inc.

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

#if 0
  // Used only during standalone testing of ARM support.
  #include "bli_system.h"
  #include "bli_type_defs.h"
  #include "bli_cpuid.h"
  #undef __x86_64__
  #undef _M_X64
  #undef __i386
  #undef _M_IX86
  #define __arm__
#endif

#ifndef BLIS_CONFIGURETIME_CPUID
  #include "blis.h"
#else
  #include "bli_system.h"
  #include "bli_type_defs.h"
  #include "bli_cpuid.h"
#endif

// -----------------------------------------------------------------------------

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386) || defined(_M_IX86)

arch_t bli_cpuid_query_id( void )
{
	uint32_t vendor, family, model, features;

	// Call the CPUID instruction and parse its results into a family id,
	// model id, and a feature bit field. The return value encodes the
	// vendor.
	vendor = bli_cpuid_query( &family, &model, &features );

	if ( vendor == VENDOR_INTEL )
	{
		// Check for each Intel configuration that is enabled, check for that
		// microarchitecture. We check from most recent to most dated.
#ifdef BLIS_CONFIG_SKX
		if ( bli_cpuid_is_skx( family, model, features ) )
			return BLIS_ARCH_SKX;
#endif
#ifdef BLIS_CONFIG_KNL
		if ( bli_cpuid_is_knl( family, model, features ) )
			return BLIS_ARCH_KNL;
#endif
#ifdef BLIS_CONFIG_HASWELL
		if ( bli_cpuid_is_haswell( family, model, features ) )
			return BLIS_ARCH_HASWELL;
#endif
#ifdef BLIS_CONFIG_SANDYBRIDGE
		if ( bli_cpuid_is_sandybridge( family, model, features ) )
			return BLIS_ARCH_SANDYBRIDGE;
#endif
#ifdef BLIS_CONFIG_PENRYN
		if ( bli_cpuid_is_penryn( family, model, features ) )
			return BLIS_ARCH_PENRYN;
#endif
		// If none of the other sub-configurations were detected, return
		// the 'generic' arch_t id value.
		return BLIS_ARCH_GENERIC;
	}
	else if ( vendor == VENDOR_AMD )
	{

		// Check for each AMD configuration that is enabled, check for that
		// microarchitecture. We check from most recent to most dated.
#ifdef BLIS_CONFIG_ZEN
		if ( bli_cpuid_is_zen( family, model, features ) )
			return BLIS_ARCH_ZEN;
#endif
#ifdef BLIS_CONFIG_EXCAVATOR
		if ( bli_cpuid_is_excavator( family, model, features ) )
			return BLIS_ARCH_EXCAVATOR;
#endif
#ifdef BLIS_CONFIG_STEAMROLLER
		if ( bli_cpuid_is_steamroller( family, model, features ) )
			return BLIS_ARCH_STEAMROLLER;
#endif
#ifdef BLIS_CONFIG_PILEDRIVER
		if ( bli_cpuid_is_piledriver( family, model, features ) )
			return BLIS_ARCH_PILEDRIVER;
#endif
#ifdef BLIS_CONFIG_BULLDOZER
		if ( bli_cpuid_is_bulldozer( family, model, features ) )
			return BLIS_ARCH_BULLDOZER;
#endif
		// If none of the other sub-configurations were detected, return
		// the 'generic' arch_t id value.
		return BLIS_ARCH_GENERIC;
	}
	else if ( vendor == VENDOR_UNKNOWN )
	{
		return BLIS_ARCH_GENERIC;
	}

	return BLIS_ARCH_GENERIC;
}

// -----------------------------------------------------------------------------

bool_t bli_cpuid_is_skx
     (
       uint32_t family,
       uint32_t model,
       uint32_t features
     )
{
	// Check for expected CPU features.
	const uint32_t expected = FEATURE_AVX      |
	                          FEATURE_FMA3     |
	                          FEATURE_AVX2     |
	                          FEATURE_AVX512F  |
	                          FEATURE_AVX512DQ |
	                          FEATURE_AVX512BW |
	                          FEATURE_AVX512VL ;


	int nvpu = vpu_count();

	if ( !bli_cpuid_has_features( features, expected ) || nvpu != 2 )
		return FALSE;

	return TRUE;
}

bool_t bli_cpuid_is_knl
     (
       uint32_t family,
       uint32_t model,
       uint32_t features
     )
{
	// Check for expected CPU features.
	const uint32_t expected = FEATURE_AVX     |
	                          FEATURE_FMA3    |
	                          FEATURE_AVX2    |
	                          FEATURE_AVX512F |
	                          FEATURE_AVX512PF;

	if ( !bli_cpuid_has_features( features, expected ) ) return FALSE;

	return TRUE;
}

bool_t bli_cpuid_is_haswell
     (
       uint32_t family,
       uint32_t model,
       uint32_t features
     )
{
	// Check for expected CPU features.
	const uint32_t expected = FEATURE_AVX  |
	                          FEATURE_FMA3 |
	                          FEATURE_AVX2;

	if ( !bli_cpuid_has_features( features, expected ) ) return FALSE;

	return TRUE;
}

bool_t bli_cpuid_is_sandybridge
     (
       uint32_t family,
       uint32_t model,
       uint32_t features
     )
{
	// Check for expected CPU features.
	const uint32_t expected = FEATURE_AVX;

	if ( !bli_cpuid_has_features( features, expected ) ) return FALSE;

	return TRUE;
}

bool_t bli_cpuid_is_penryn
     (
       uint32_t family,
       uint32_t model,
       uint32_t features
     )
{
	// Check for expected CPU features.
	const uint32_t expected = FEATURE_SSE3 |
	                          FEATURE_SSSE3;

	if ( !bli_cpuid_has_features( features, expected ) ) return FALSE;

	return TRUE;
}

// -----------------------------------------------------------------------------

bool_t bli_cpuid_is_zen
     (
       uint32_t family,
       uint32_t model,
       uint32_t features
     )
{
	// Check for expected CPU features.
	const uint32_t expected = FEATURE_AVX  |
	                          FEATURE_FMA3 |
	                          FEATURE_AVX2;

	if ( !bli_cpuid_has_features( features, expected ) ) return FALSE;

	// All Zen cores have a family of 0x17.
	if ( family != 0x17 ) return FALSE;

	// Finally, check for specific models:
	// - 0x00-0xff (THIS NEEDS UPDATING)
	const bool_t is_arch
	=
	( 0x00 <= model && model <= 0xff );

	if ( !is_arch ) return FALSE;

	return TRUE;
}

bool_t bli_cpuid_is_excavator
     (
       uint32_t family,
       uint32_t model,
       uint32_t features
     )
{
	// Check for expected CPU features.
	const uint32_t expected = FEATURE_AVX  |
	                          FEATURE_FMA3 |
	                          FEATURE_AVX2;

	if ( !bli_cpuid_has_features( features, expected ) ) return FALSE;

	// All Excavator cores have a family of 0x15.
	if ( family != 0x15 ) return FALSE;

	// Finally, check for specific models:
	// - 0x60-0x7f
	const bool_t is_arch
	=
	( 0x60 <= model && model <= 0x7f );

	if ( !is_arch ) return FALSE;

	return TRUE;
}

bool_t bli_cpuid_is_steamroller
     (
       uint32_t family,
       uint32_t model,
       uint32_t features
     )
{
	// Check for expected CPU features.
	const uint32_t expected = FEATURE_AVX  |
	                          FEATURE_FMA3 |
	                          FEATURE_FMA4;

	if ( !bli_cpuid_has_features( features, expected ) ) return FALSE;

	// All Steamroller cores have a family of 0x15.
	if ( family != 0x15 ) return FALSE;

	// Finally, check for specific models:
	// - 0x30-0x3f
	const bool_t is_arch
	=
	( 0x30 <= model && model <= 0x3f );

	if ( !is_arch ) return FALSE;

	return TRUE;
}

bool_t bli_cpuid_is_piledriver
     (
       uint32_t family,
       uint32_t model,
       uint32_t features
     )
{
	// Check for expected CPU features.
	const uint32_t expected = FEATURE_AVX  |
	                          FEATURE_FMA3 |
	                          FEATURE_FMA4;

	if ( !bli_cpuid_has_features( features, expected ) ) return FALSE;

	// All Piledriver cores have a family of 0x15.
	if ( family != 0x15 ) return FALSE;

	// Finally, check for specific models:
	// - 0x02
	// - 0x10-0x1f
	const bool_t is_arch
	=
	model == 0x02 || ( 0x10 <= model && model <= 0x1f );

	if ( !is_arch ) return FALSE;

	return TRUE;
}

bool_t bli_cpuid_is_bulldozer
     (
       uint32_t family,
       uint32_t model,
       uint32_t features
     )
{
	// Check for expected CPU features.
	const uint32_t expected = FEATURE_AVX |
	                          FEATURE_FMA4;

	if ( !bli_cpuid_has_features( features, expected ) ) return FALSE;

	// All Bulldozer cores have a family of 0x15.
	if ( family != 0x15 ) return FALSE;

	// Finally, check for specific models:
	// - 0x00
	// - 0x01
	const bool_t is_arch
	=
	( model == 0x00 || model == 0x01 );

	if ( !is_arch ) return FALSE;

	return TRUE;
}

#elif defined(__aarch64__) || defined(__arm__) || defined(_M_ARM)

arch_t bli_cpuid_query_id( void )
{
	uint32_t vendor, model, part, features;

	// Call the CPUID instruction and parse its results into a model id,
	// part id, and a feature bit field. The return value encodes the
	// vendor.
	vendor = bli_cpuid_query( &model, &part, &features );

	//printf( "vendor   = %u\n", vendor );
	//printf( "model    = %u\n", model );
	//printf( "part     = 0x%x\n", part );
	//printf( "features = %u\n", features );

	if ( vendor == VENDOR_ARM )
	{
		if ( model == MODEL_ARMV8 )
		{
			// Check for each ARMv8 configuration that is enabled, check for that
			// microarchitecture. We check from most recent to most dated.
#ifdef BLIS_CONFIG_CORTEXA57
			if ( bli_cpuid_is_cortexa57( model, part, features ) )
				return BLIS_ARCH_CORTEXA57;
#endif
			// If none of the other sub-configurations were detected, return
			// the 'generic' arch_t id value.
			return BLIS_ARCH_GENERIC;
		}
		else if ( model == MODEL_ARMV7 )
		{
			// Check for each ARMv7 configuration that is enabled, check for that
			// microarchitecture. We check from most recent to most dated.
#ifdef BLIS_CONFIG_CORTEXA15
			if ( bli_cpuid_is_cortexa15( model, part, features ) )
				return BLIS_ARCH_CORTEXA15;
#endif
#ifdef BLIS_CONFIG_CORTEXA9
			if ( bli_cpuid_is_cortexa9( model, part, features ) )
				return BLIS_ARCH_CORTEXA9;
#endif
			// If none of the other sub-configurations were detected, return
			// the 'generic' arch_t id value.
			return BLIS_ARCH_GENERIC;
		}
	}
	else if ( vendor == VENDOR_UNKNOWN )
	{
		return BLIS_ARCH_GENERIC;
	}

	return BLIS_ARCH_GENERIC;
}

bool_t bli_cpuid_is_cortexa57
     (
       uint32_t family,
       uint32_t model,
       uint32_t features
     )
{
	// Check for expected CPU features.
	const uint32_t expected = FEATURE_NEON;

	if ( !bli_cpuid_has_features( features, expected ) ) return FALSE;

	return TRUE;
}

bool_t bli_cpuid_is_cortexa53
     (
       uint32_t family,
       uint32_t model,
       uint32_t features
     )
{
	// Check for expected CPU features.
	const uint32_t expected = FEATURE_NEON;

	if ( !bli_cpuid_has_features( features, expected ) ) return FALSE;

	return TRUE;
}

bool_t bli_cpuid_is_cortexa15
     (
       uint32_t family,
       uint32_t model,
       uint32_t features
     )
{
	// Check for expected CPU features.
	const uint32_t expected = FEATURE_NEON;

	if ( !bli_cpuid_has_features( features, expected ) ) return FALSE;

	return TRUE;
}

bool_t bli_cpuid_is_cortexa9
     (
       uint32_t family,
       uint32_t model,
       uint32_t features
     )
{
	// Check for expected CPU features.
	const uint32_t expected = FEATURE_NEON;

	if ( !bli_cpuid_has_features( features, expected ) ) return FALSE;

	return TRUE;
}

#endif

// -----------------------------------------------------------------------------

//
// This section of the file was based off of cpuid.cxx from TBLIS [1].
//
// [1] https://github.com/devinamatthews/tblis
//

/*

   Copyright (C) 2017, The University of Texas at Austin
   Copyright (C) 2017, Devin Matthews
   Copyright (C) 2018, Advanced Micro Devices, Inc.

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

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386) || defined(_M_IX86)

enum
{
                                      // input register(s)     output register
	FEATURE_MASK_SSE3     = (1u<< 0), // cpuid[eax=1]         :ecx[0]
	FEATURE_MASK_SSSE3    = (1u<< 9), // cpuid[eax=1]         :ecx[9]
	FEATURE_MASK_SSE41    = (1u<<19), // cpuid[eax=1]         :ecx[19]
	FEATURE_MASK_SSE42    = (1u<<20), // cpuid[eax=1]         :ecx[20]
	FEATURE_MASK_AVX      = (1u<<28), // cpuid[eax=1]         :ecx[28]
	FEATURE_MASK_AVX2     = (1u<< 5), // cpuid[eax=7,ecx=0]   :ebx[5]
	FEATURE_MASK_FMA3     = (1u<<12), // cpuid[eax=1]         :ecx[12]
	FEATURE_MASK_FMA4     = (1u<<16), // cpuid[eax=0x80000001]:ecx[16]
	FEATURE_MASK_AVX512F  = (1u<<16), // cpuid[eax=7,ecx=0]   :ebx[16]
	FEATURE_MASK_AVX512DQ = (1u<<17), // cpuid[eax=7,ecx=0]   :ebx[17]
	FEATURE_MASK_AVX512PF = (1u<<26), // cpuid[eax=7,ecx=0]   :ebx[26]
	FEATURE_MASK_AVX512ER = (1u<<27), // cpuid[eax=7,ecx=0]   :ebx[27]
	FEATURE_MASK_AVX512CD = (1u<<28), // cpuid[eax=7,ecx=0]   :ebx[28]
	FEATURE_MASK_AVX512BW = (1u<<30), // cpuid[eax=7,ecx=0]   :ebx[30]
	FEATURE_MASK_AVX512VL = (1u<<31), // cpuid[eax=7,ecx=0]   :ebx[31]
	FEATURE_MASK_XGETBV   = (1u<<26)|
                            (1u<<27), // cpuid[eax=1]         :ecx[27:26]
	XGETBV_MASK_XMM       = 0x02u,    // xcr0[1]
	XGETBV_MASK_YMM       = 0x04u,    // xcr0[2]
	XGETBV_MASK_ZMM       = 0xe0u     // xcr0[7:5]
};


uint32_t bli_cpuid_query
     (
       uint32_t* family,
       uint32_t* model,
       uint32_t* features
     )
{
	uint32_t eax, ebx, ecx, edx;

	uint32_t old_model  = 0;
	uint32_t old_family = 0;
	uint32_t ext_model  = 0;
	uint32_t ext_family = 0;

	*family   = 0;
	*model    = 0;
	*features = 0;

	//fprintf( stderr, "checking cpuid\n" );

	uint32_t cpuid_max     = __get_cpuid_max( 0,           0 );
	uint32_t cpuid_max_ext = __get_cpuid_max( 0x80000000u, 0 );

	//fprintf( stderr, "max cpuid leaf: %d\n", cpuid_max );
	//fprintf( stderr, "max extended cpuid leaf: %08x\n", cpuid_max_ext );

	if ( cpuid_max < 1 ) return VENDOR_UNKNOWN;

	// The fourth '0' serves as the NULL-terminator for the vendor string.
	uint32_t vendor_string[4] = { 0, 0, 0, 0 };

	// This is actually a macro that modifies the last four operands,
	// hence why they are not passed by address.
	__cpuid( 0, eax, vendor_string[0],
	                 vendor_string[2],
	                 vendor_string[1] );

	// Check extended feature bits for post-AVX2 features.
	if ( cpuid_max >= 7 )
	{
		// This is actually a macro that modifies the last four operands,
		// hence why they are not passed by address.
		__cpuid_count( 7, 0, eax, ebx, ecx, edx );

		//fprintf( stderr, "cpuid leaf 7:\n" );
		//print_binary( eax );
		//print_binary( ebx );
		//print_binary( ecx );
		//print_binary( edx );

		if ( bli_cpuid_has_features( ebx, FEATURE_MASK_AVX2     ) ) *features |= FEATURE_AVX2;
		if ( bli_cpuid_has_features( ebx, FEATURE_MASK_AVX512F  ) ) *features |= FEATURE_AVX512F;
		if ( bli_cpuid_has_features( ebx, FEATURE_MASK_AVX512DQ ) ) *features |= FEATURE_AVX512DQ;
		if ( bli_cpuid_has_features( ebx, FEATURE_MASK_AVX512PF ) ) *features |= FEATURE_AVX512PF;
		if ( bli_cpuid_has_features( ebx, FEATURE_MASK_AVX512ER ) ) *features |= FEATURE_AVX512ER;
		if ( bli_cpuid_has_features( ebx, FEATURE_MASK_AVX512CD ) ) *features |= FEATURE_AVX512CD;
		if ( bli_cpuid_has_features( ebx, FEATURE_MASK_AVX512BW ) ) *features |= FEATURE_AVX512BW;
		if ( bli_cpuid_has_features( ebx, FEATURE_MASK_AVX512VL ) ) *features |= FEATURE_AVX512VL;
	}

	// Check extended processor info / features bits for AMD-specific features.
	if ( cpuid_max_ext >= 0x80000001u )
	{
		// This is actually a macro that modifies the last four operands,
		// hence why they are not passed by address.
		__cpuid( 0x80000001u, eax, ebx, ecx, edx );

		//fprintf(stderr, "extended cpuid leaf 0x80000001:\n");
		//print_binary(eax);
		//print_binary(ebx);
		//print_binary(ecx);
		//print_binary(edx);

		if ( bli_cpuid_has_features( ecx, FEATURE_MASK_FMA4 ) ) *features |= FEATURE_FMA4;
	}

	// Unconditionally check processor info / features bits.
	{
		// This is actually a macro that modifies the last four operands,
		// hence why they are not passed by address.
		__cpuid( 1, eax, ebx, ecx, edx );

		//fprintf(stderr, "cpuid leaf 1:\n");
		//print_binary(eax);
		//print_binary(ebx);
		//print_binary(ecx);
		//print_binary(edx);

		/*
		   cpuid(eax=1): eax[27:0]

			3: 0 - Stepping
			7: 4 - Model
		   11: 8 - Family
		   13:12 - Processor Type
		   19:16 - Extended Model
		   27:20 - Extended Family

		   Intel and AMD have suggested applications to display the family of a
		   CPU as the sum of the "Family" and the "Extended Family" fields shown
		   above, and the model as the sum of the "Model" and the 4-bit
		   left-shifted "Extended Model" fields. If "Family" is different than
		   6 or 15, only the "Family" and "Model" fields should be used while the
		   "Extended Family" and "Extended Model" bits are reserved. If "Family"
		   is set to 15, then "Extended Family" and the 4-bit left-shifted
		   "Extended Model" should be added to the respective base values, and if
		   "Family" is set to 6, then only the 4-bit left-shifted "Extended Model"
		   should be added to "Model".
		*/

		old_model  = ( eax >>  4 ) & ( 0xF  ); // bits 7:4
		old_family = ( eax >>  8 ) & ( 0xF  ); // bits 11:8

		ext_model  = ( eax >> 16 ) & ( 0xF  ); // bits 19:16
		ext_family = ( eax >> 20 ) & ( 0xFF ); // bits 27:20

		// Set the display model and family values based on the original family
		// value. See explanation above.
		if      ( old_family == 6 )
		{
			*model  = ( ext_model << 4 ) + old_model;
			*family =                      old_family;
		}
		else if ( old_family == 15 )
		{
			*model  = ( ext_model << 4 ) + old_model;
			*family = ( ext_family     ) + old_family;
		}
		else
		{
			*model  =                      old_model;
			*family =                      old_family;
		}

		// Check for SSE, AVX, and FMA3 features.
		if ( bli_cpuid_has_features( ecx, FEATURE_MASK_SSE3  ) ) *features |= FEATURE_SSE3;
		if ( bli_cpuid_has_features( ecx, FEATURE_MASK_SSSE3 ) ) *features |= FEATURE_SSSE3;
		if ( bli_cpuid_has_features( ecx, FEATURE_MASK_SSE41 ) ) *features |= FEATURE_SSE41;
		if ( bli_cpuid_has_features( ecx, FEATURE_MASK_SSE42 ) ) *features |= FEATURE_SSE42;
		if ( bli_cpuid_has_features( ecx, FEATURE_MASK_AVX   ) ) *features |= FEATURE_AVX;
		if ( bli_cpuid_has_features( ecx, FEATURE_MASK_FMA3  ) ) *features |= FEATURE_FMA3;

		// Check whether the hardware supports xsave/xrestor/xsetbv/xgetbv AND 
		// support for these is enabled by the OS. If so, then we proceed with
		// checking that various register-state saving features are available.
		if ( bli_cpuid_has_features( ecx, FEATURE_MASK_XGETBV ) )
		{
			uint32_t xcr = 0;

			// Call xgetbv to get xcr0 (the extended control register) copied
			// to [edx:eax]. This encodes whether software supports various
			// register state-saving features.
			__asm__ __volatile__
			(
				".byte 0x0F, 0x01, 0xD0"
				: "=a" (eax),
				  "=d" (edx)
				: "c"  (xcr)
				: "cc"
			);

			//fprintf(stderr, "xcr0:\n");
			//print_binary(eax);
			//print_binary(edx);

			//fprintf(stderr, "xgetbv: xmm: %d\n", bli_cpuid_has_features(eax, XGETBV_MASK_XMM));
			//fprintf(stderr, "xgetbv: ymm: %d\n", bli_cpuid_has_features(eax, XGETBV_MASK_XMM|
			//                                                XGETBV_MASK_YMM));
			//fprintf(stderr, "xgetbv: zmm: %d\n", bli_cpuid_has_features(eax, XGETBV_MASK_XMM|
			//                                                XGETBV_MASK_YMM|
			//                                                XGETBV_MASK_ZMM));

			// The OS can manage the state of 512-bit zmm (AVX-512) registers
			// only if the xcr[7:5] bits are set. If they are not set, then
			// clear all feature bits related to AVX-512. 
			if ( !bli_cpuid_has_features( eax, XGETBV_MASK_XMM |
				                               XGETBV_MASK_YMM |
				                               XGETBV_MASK_ZMM ) )
			{
				*features &= ~( FEATURE_AVX512F  |
				                FEATURE_AVX512DQ |
				                FEATURE_AVX512PF |
				                FEATURE_AVX512ER |
				                FEATURE_AVX512CD |
				                FEATURE_AVX512BW |
				                FEATURE_AVX512VL );
			}

			// The OS can manage the state of 256-bit ymm (AVX) registers
			// only if the xcr[2] bit is set. If it is not set, then
			// clear all feature bits related to AVX. 
			if ( !bli_cpuid_has_features( eax, XGETBV_MASK_XMM |
				                               XGETBV_MASK_YMM ) )
			{
				*features &= ~( FEATURE_AVX  |
				                FEATURE_AVX2 |
				                FEATURE_FMA3 |
				                FEATURE_FMA4 );
			}

			// The OS can manage the state of 128-bit xmm (SSE) registers
			// only if the xcr[1] bit is set. If it is not set, then
			// clear all feature bits related to SSE (which means the
			// entire bitfield is clear). 
			if ( !bli_cpuid_has_features( eax, XGETBV_MASK_XMM ) )
			{
				*features = 0;
			}
		}
		else
		{
			// If the hardware does not support xsave/xrestor/xsetbv/xgetbv,
			// OR these features are not enabled by the OS, then we clear
			// the bitfield, because it means that not even xmm support is
			// present.

			//fprintf(stderr, "xgetbv: no\n");
			features = 0;
		}
	}

	//fprintf(stderr, "vendor: %12s\n", vendor_string);
	//fprintf(stderr, "family: %d\n", family);
	//fprintf(stderr, "model: %d\n", model);
	//fprintf(stderr, "sse3: %d\n", bli_cpuid_has_features(features, FEATURE_SSE3));
	//fprintf(stderr, "ssse3: %d\n", bli_cpuid_has_features(features, FEATURE_SSSE3));
	//fprintf(stderr, "sse4.1: %d\n", bli_cpuid_has_features(features, FEATURE_SSE41));
	//fprintf(stderr, "sse4.2: %d\n", bli_cpuid_has_features(features, FEATURE_SSE42));
	//fprintf(stderr, "avx: %d\n", bli_cpuid_has_features(features, FEATURE_AVX));
	//fprintf(stderr, "avx2: %d\n", bli_cpuid_has_features(features, FEATURE_AVX2));
	//fprintf(stderr, "fma3: %d\n", bli_cpuid_has_features(features, FEATURE_FMA3));
	//fprintf(stderr, "fma4: %d\n", bli_cpuid_has_features(features, FEATURE_FMA4));
	//fprintf(stderr, "avx512f: %d\n", bli_cpuid_has_features(features, FEATURE_AVX512F));
	//fprintf(stderr, "avx512pf: %d\n", bli_cpuid_has_features(features, FEATURE_AVX512PF));
	//fprintf(stderr, "avx512dq: %d\n", bli_cpuid_has_features(features, FEATURE_AVX512DQ));

	// Check the vendor string and return a value to indicate Intel or AMD.
	if      ( strcmp( ( char* )vendor_string, "AuthenticAMD" ) == 0 )
		return VENDOR_AMD;
	else if ( strcmp( ( char* )vendor_string, "GenuineIntel" ) == 0 )
		return VENDOR_INTEL;
	else
		return VENDOR_UNKNOWN;
}

void get_cpu_name( char *cpu_name )
{
	uint32_t eax, ebx, ecx, edx;

	__cpuid( 0x80000002u, eax, ebx, ecx, edx );
	//printf("%x %x %x %x\n", eax, ebx, ecx, edx);

	*( uint32_t* )&cpu_name[0 + 0] = eax;
	*( uint32_t* )&cpu_name[0 + 4] = ebx;
	*( uint32_t* )&cpu_name[0 + 8] = ecx;
	*( uint32_t* )&cpu_name[0 +12] = edx;

	__cpuid( 0x80000003u, eax, ebx, ecx, edx );
	//printf("%x %x %x %x\n", eax, ebx, ecx, edx);

	*( uint32_t* )&cpu_name[16+ 0] = eax;
	*( uint32_t* )&cpu_name[16+ 4] = ebx;
	*( uint32_t* )&cpu_name[16+ 8] = ecx;
	*( uint32_t* )&cpu_name[16+12] = edx;

	__cpuid( 0x80000004u, eax, ebx, ecx, edx );
	//printf("%x %x %x %x\n", eax, ebx, ecx, edx);

	*( uint32_t* )&cpu_name[32+ 0] = eax;
	*( uint32_t* )&cpu_name[32+ 4] = ebx;
	*( uint32_t* )&cpu_name[32+ 8] = ecx;
	*( uint32_t* )&cpu_name[32+12] = edx;
}

int vpu_count( void )
{
	char  cpu_name[48] = {};
	char* loc;
	char  model_num[5];
	int   sku;

	get_cpu_name( cpu_name );

	if ( strstr( cpu_name, "Intel(R) Xeon(R)" ) != NULL )
	{
		loc = strstr( cpu_name, "Platinum" );
		if ( loc == NULL )
			loc = strstr( cpu_name, "Gold" );
		if ( loc == NULL )
			loc = strstr( cpu_name, "Silver" );
		if ( loc == NULL )
			loc = strstr( cpu_name, "Bronze" );
		if ( loc == NULL )
			loc = strstr( cpu_name, "W" );
		if ( loc == NULL )
			return -1;

		loc = strstr( loc+1, " " );
		if ( loc == NULL )
			return -1;

		strncpy( model_num, loc+1, 4 );
		model_num[4] = '\0';

		sku = atoi( model_num );

		if      ( 8199 >= sku && sku >= 8100 ) return 2;
		else if ( 6199 >= sku && sku >= 6100 ) return 2;
		else if (                sku == 5122 ) return 2;
		else if ( 5199 >= sku && sku >= 5100 ) return 1;
		else if ( 4199 >= sku && sku >= 4100 ) return 1;
		else if ( 3199 >= sku && sku >= 3100 ) return 1;
		else if ( 2199 >= sku && sku >= 2120 ) return 2;
		else if ( 2119 >= sku && sku >= 2100 ) return 1;
		else return -1;
	}
	else if ( strstr( cpu_name, "Intel(R) Core(TM) i9" ) != NULL )
	{
		return 1;
	}
	else if ( strstr( cpu_name, "Intel(R) Core(TM) i7" ) != NULL )
	{
		if ( strstr( cpu_name, "7800X" ) != NULL ||
		     strstr( cpu_name, "7820X" ) != NULL )
			return 1;
		else
			return -1;
	}
	else
	{
		return -1;
	}
}

#elif defined(__aarch64__) || defined(__arm__) || defined(_M_ARM)

uint32_t bli_cpuid_query
     (
       uint32_t* model,
       uint32_t* part,
       uint32_t* features
     )
{
	*model    = MODEL_UNKNOWN;
    *part     = 0;
	*features = 0;
    
#if 1
	const char* grep_str1 = "grep -m 1 Processor /proc/cpuinfo";
	const char* grep_str2 = "grep -m 1 'CPU part' /proc/cpuinfo";
	const char* grep_str3 = "grep -m 1 Features /proc/cpuinfo";
#else
	const char* grep_str1 = "grep -m 1 Processor ./proc_cpuinfo";
	const char* grep_str2 = "grep -m 1 'CPU part' ./proc_cpuinfo";
	const char* grep_str3 = "grep -m 1 Features ./proc_cpuinfo";
#endif

	FILE *fd1 = popen( grep_str1, "r");
	if ( !fd1 )
	{
        //printf("popen 1 failed\n");
		return VENDOR_ARM;
	}
	FILE *fd2 = popen( grep_str2, "r");
	if (!fd2)
	{
        //printf("popen 2 failed\n");
		pclose(fd1);
		return VENDOR_ARM;
	}
	FILE *fd3 = popen( grep_str3, "r");
	if (!fd3)
	{
        //printf("popen 3 failed\n");
		pclose(fd1);
		pclose(fd2);
		return VENDOR_ARM;
	}

	uint32_t n1, n2, n3;
	int      c;

	// First, discover how many chars are in each stream.
	for ( n1 = 0; (c = fgetc(fd1)) != EOF; ++n1 ) continue;
	for ( n2 = 0; (c = fgetc(fd2)) != EOF; ++n2 ) continue;
	for ( n3 = 0; (c = fgetc(fd3)) != EOF; ++n3 ) continue;

	//printf( "n1, n2, n3 = %u %u %u\n", n1, n2, n3 );

	// Close the streams.
	pclose( fd1 );
	pclose( fd2 );
	pclose( fd3 );

	// Allocate the correct amount of memory for each stream.
	char* proc_str = malloc( ( size_t )( n1 + 1 ) );
	char* ptno_str = malloc( ( size_t )( n2 + 1 ) );
	char* feat_str = malloc( ( size_t )( n3 + 1 ) );
    *proc_str = 0;
    *ptno_str = 0;
    *feat_str = 0;

	// Re-open the streams. Note that there is no need to check for errors
	// this time since we're assumign that the contents of /proc/cpuinfo
	// will be the same as before.
	fd1 = popen( grep_str1, "r");
	fd2 = popen( grep_str2, "r");
	fd3 = popen( grep_str3, "r");

	char* r_val;

	// Now read each stream in its entirety. Nothing should go wrong, but
	// if it does, bail out.
	r_val = fgets( proc_str, n1, fd1 );
	if ( n1 && r_val == NULL ) bli_abort();

	r_val = fgets( ptno_str, n2, fd2 );
	if ( n2 && r_val == NULL ) bli_abort();

	r_val = fgets( feat_str, n3, fd3 );
	if ( n3 && r_val == NULL ) bli_abort();

    //printf( "proc_str: %s\n", proc_str );
	//printf( "ptno_str: %s\n", ptno_str );
	//printf( "feat_str: %s\n", feat_str );

	// Close the streams.
	pclose( fd1 );
	pclose( fd2 );
	pclose( fd3 );

	// Parse the feature string to check for SIMD features.
	if ( strstr( feat_str, "neon"  ) != NULL ||
	     strstr( feat_str, "asimd" ) != NULL )
		*features |= FEATURE_NEON;
	//printf( "features var: %u\n", *features );

	// Parse the processor string to uncover the model.
	if      ( strstr( proc_str, "ARMv7"   ) != NULL )
		*model = MODEL_ARMV7;
	else if ( strstr( proc_str, "AArch64" ) != NULL ||
              strstr( proc_str, "ARMv8"   ) )
		*model = MODEL_ARMV8;
	//printf( "model: %u\n", *model );

	// Parse the part number string.
	r_val = strstr( ptno_str, "0x" );
    if ( r_val != NULL)
    {
	    *part = strtol( r_val, NULL, 16 );
    }
	//printf( "part#: %x\n", *part );

	return VENDOR_ARM;
}

#endif
