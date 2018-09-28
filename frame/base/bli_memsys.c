/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2014, The University of Texas at Austin
   Copyright (C) 2016, Hewlett Packard Enterprise Development LP

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

static membrk_t global_membrk;

// -----------------------------------------------------------------------------

membrk_t* bli_memsys_global_membrk( void )
{
	return &global_membrk;
}

// -----------------------------------------------------------------------------

void bli_memsys_init( void )
{
	// Query a native context so we have something to pass into
	// bli_membrk_init_pools(). We use BLIS_DOUBLE for the datatype,
	// but the dt argument is actually only used when initializing
	// contexts for induced methods.

	// NOTE: Instead of calling bli_gks_query_cntx(), we call
	// bli_gks_query_cntx_noinit() to avoid the call to bli_init_once().
	cntx_t* cntx_p = bli_gks_query_cntx_noinit();

	// Initialize the global membrk_t object and its memory pools.
	bli_membrk_init( cntx_p, &global_membrk );
}

void bli_memsys_finalize( void )
{
	// Finalize the global membrk_t object and its memory pools.
	bli_membrk_finalize( &global_membrk );
}

