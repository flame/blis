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

#ifndef BLIS_CONFIGURETIME_CPUID
  #include "blis.h"
#else
  #include <stdlib.h>
  #include <string.h>
  #include <stdint.h>
  #include "bli_type_defs.h"
  #include "bli_arch.h"
  #include "bli_cpuid.h"
#endif

// -----------------------------------------------------------------------------

arch_t bli_arch_query_id( void )
{
	arch_t id = -1;

	// Architecture families.
#if defined BLIS_FAMILY_INTEL64 || \
    defined BLIS_FAMILY_AMD64 || \
    defined BLIS_FAMILY_X86_64
	id = bli_cpuid_query_id();
#endif

	// Intel microarchitectures.
#ifdef BLIS_FAMILY_SKX
	id = BLIS_ARCH_SKX;
#endif
#ifdef BLIS_FAMILY_KNL
	id = BLIS_ARCH_KNL;
#endif
#ifdef BLIS_FAMILY_KNC
	id = BLIS_ARCH_KNC;
#endif
#ifdef BLIS_FAMILY_HASWELL
	id = BLIS_ARCH_HASWELL;
#endif
#ifdef BLIS_FAMILY_SANDYBRIDGE
	id = BLIS_ARCH_SANDYBRIDGE;
#endif
#ifdef BLIS_FAMILY_PENRYN
	id = BLIS_ARCH_PENRYN;
#endif

	// AMD microarchitectures.
#ifdef BLIS_FAMILY_ZEN
	id = BLIS_ARCH_ZEN;
#endif
#ifdef BLIS_FAMILY_EXCAVATOR
	id = BLIS_ARCH_EXCAVATOR;
#endif
#ifdef BLIS_FAMILY_STEAMROLLER
	id = BLIS_ARCH_STEAMROLLER;
#endif
#ifdef BLIS_FAMILY_PILEDRIVER
	id = BLIS_ARCH_PILEDRIVER;
#endif
#ifdef BLIS_FAMILY_BULLDOZER
	id = BLIS_ARCH_BULLDOZER;
#endif

	// ARM microarchitectures.
#ifdef BLIS_FAMILY_CORTEXA57
	id = BLIS_ARCH_CORTEXA57;
#endif
#ifdef BLIS_FAMILY_CORTEXA15
	id = BLIS_ARCH_CORTEXA15;
#endif
#ifdef BLIS_FAMILY_CORTEXA9
	id = BLIS_ARCH_CORTEXA9;
#endif

	// IBM microarchitectures.
#ifdef BLIS_FAMILY_POWER7
	id = BLIS_ARCH_POWER7;
#endif
#ifdef BLIS_FAMILY_BGQ
	id = BLIS_ARCH_BGQ;
#endif

	// Generic microarchitecture.
#ifdef BLIS_FAMILY_GENERIC
	id = BLIS_ARCH_GENERIC;
#endif

	//printf( "blis_arch_query_id(): id = %u\n", id );
	//exit(1);

	return id;
}

// -----------------------------------------------------------------------------

// NOTE: This string array must be kept up-to-date with the arch_t
// enumeration that is typedef'ed in bli_type_defs.h. That is, the
// index order of each string should correspond to the implied/assigned
// enum value given to the corresponding BLIS_ARCH_ value.
static char* config_name[ BLIS_NUM_ARCHS ] =
{
    "skx",
    "knl",
    "knc",
    "haswell",
    "sandybridge",
    "penryn",

    "zen",
    "excavator",
    "steamroller",
    "piledriver",
    "bulldozer",

    "cortexa57",
    "cortexa15",
    "cortexa9",

    "power7",
    "bgq",

    "generic"
};

char* bli_arch_string( arch_t id )
{
	return config_name[ id ];
}

