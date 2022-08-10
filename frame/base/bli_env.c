/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2014, The University of Texas at Austin
   Copyright (C) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

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
  #include "bli_env.h"
#else
  #include "blis.h"
#endif

// -----------------------------------------------------------------------------

gint_t bli_env_get_var( const char* env, gint_t fallback )
{
	gint_t r_val;
	char*  str;

	// Query the environment variable and store the result in str.
	str = getenv( env );

	// Set the return value based on the string obtained from getenv().
	if ( str != NULL )
	{
		// If there was no error, convert the string to an integer and
		// prepare to return that integer.
		r_val = ( gint_t )strtol( str, NULL, 10 );
	}
	else
	{
		// If there was an error, use the "fallback" as the return value.
		r_val = fallback;
	}

	return r_val;
}

gint_t bli_env_get_var_arch_type( const char* env, gint_t fallback )
{
	gint_t r_val;
	char*  str;
	int i, size;

	// Query the environment variable and store the result in str.
	str = getenv( env );

	// Set the return value based on the string obtained from getenv().
	if ( str != NULL )
	{
		// If there was no error, convert the string to an integer and
		// prepare to return that integer.
		r_val = ( gint_t )strtol( str, NULL, 10 );

		if (r_val == 0)
		{
			// Could be deliberately 0 (now meaning an ERROR)
			// or a non-numeric value. We still allow direct
			// specification of integer value to select code
			// path. Non-zero integer values bypass this code
			// block and are handled as before. Here we look
			// for known meaningful names, and return 0 if
			// we cannot find a match. This code MUST be kept
			// in synch with arch_t enumeration in
			// bli_type_defs.h and array config_name in bli_arch.c

			// convert string to lowercase
			size = strlen(str);
			for (i=0;i<=size;i++)
			{
				str[i] = tolower(str[i]);
			}

			// Intel
			if (strcmp(str, "skx") == 0)
			{
				r_val = BLIS_ARCH_SKX;
			}
			else if (strcmp(str, "knl") == 0)
			{
				r_val = BLIS_ARCH_KNL;
			}
			else if (strcmp(str, "knc") == 0)
			{
				r_val = BLIS_ARCH_KNC;
			}
			else if (strcmp(str, "haswell") == 0)
			{
				r_val = BLIS_ARCH_HASWELL;
			}
			else if (strcmp(str, "sandybridge") == 0)
			{
				r_val = BLIS_ARCH_SANDYBRIDGE;
			}
			else if (strcmp(str, "penryn") == 0)
			{
				r_val = BLIS_ARCH_PENRYN;
			}
			// AMD
			else if (strcmp(str, "zen4") == 0)
			{
				r_val = BLIS_ARCH_ZEN4;
			}
			else if (strcmp(str, "zen3") == 0)
			{
				r_val = BLIS_ARCH_ZEN3;
			}
			else if (strcmp(str, "zen2") == 0)
			{
				r_val = BLIS_ARCH_ZEN2;
			}
			else if ((strcmp(str, "zen") == 0) ||
			         (strcmp(str, "zen1") == 0))
			{
				r_val = BLIS_ARCH_ZEN;
			}
			else if (strcmp(str, "excavator") == 0)
			{
				r_val = BLIS_ARCH_EXCAVATOR;
			}
			else if (strcmp(str, "steamroller") == 0)
			{
				r_val = BLIS_ARCH_STEAMROLLER;
			}
			else if (strcmp(str, "piledriver") == 0)
			{
				r_val = BLIS_ARCH_PILEDRIVER;
			}
			else if (strcmp(str, "bulldozer") == 0)
			{
				r_val = BLIS_ARCH_BULLDOZER;
			}
			// ARM
			else if (strcmp(str, "thunderx2") == 0)
			{
				r_val = BLIS_ARCH_THUNDERX2;
			}
			else if (strcmp(str, "cortexa57") == 0)
			{
				r_val = BLIS_ARCH_CORTEXA57;
			}
			else if (strcmp(str, "cortexa53") == 0)
			{
				r_val = BLIS_ARCH_CORTEXA53;
			}
			else if (strcmp(str, "cortexa15") == 0)
			{
				r_val = BLIS_ARCH_CORTEXA15;
			}
			else if (strcmp(str, "cortexa9") == 0)
			{
				r_val = BLIS_ARCH_CORTEXA9;
			}
			// IBM POWER
			else if (strcmp(str, "power10") == 0)
			{
				r_val = BLIS_ARCH_POWER10;
			}
			else if (strcmp(str, "power9") == 0)
			{
				r_val = BLIS_ARCH_POWER9;
			}
			else if (strcmp(str, "power7") == 0)
			{
				r_val = BLIS_ARCH_POWER7;
			}
			else if (strcmp(str, "bgq") == 0)
			{
				r_val = BLIS_ARCH_BGQ;
			}
			// Generic
			else if (strcmp(str, "generic") == 0)
			{
				r_val = BLIS_ARCH_GENERIC;
			}

			// No else case means we return r_val=0, i.e. this behaves
			// the same as generic bli_env_get_var().
		}
	}
	else
	{
		// If there was an error, use the "fallback" as the return value.
		r_val = fallback;
	}

	return r_val;
}

#if 0
#ifdef _MSC_VER
#define strerror_r(errno,buf,len) strerror_s(buf,len,errno)
#endif

void bli_env_set_var( const char* env, dim_t value )
{
	dim_t       r_val;
	char        value_str[32];
	const char* fs_32 = "%u";
	const char* fs_64 = "%lu";

	// Convert the string to an integer, but vary the format specifier
	// depending on the integer type size.
	if ( bli_info_get_int_type_size() == 32 ) sprintf( value_str, fs_32, value );
	else                                      sprintf( value_str, fs_64, value );

	// Set the environment variable using the string we just wrote to via
	// sprintf(). (The 'TRUE' argument means we want to overwrite the current
	// value if the environment variable already exists.)
	r_val = bli_setenv( env, value_str, TRUE );

	// Check the return value in case something went horribly wrong.
	if ( r_val == -1 )
	{
		char err_str[128];

		// Query the human-readable error string corresponding to errno.
		strerror_r( errno, err_str, 128 );

		// Print the error message.
		bli_print_msg( err_str, __FILE__, __LINE__ );
	}
}
#endif

