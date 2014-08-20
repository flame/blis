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

void* bli_malloc( siz_t size )
{
	void* p = NULL;
	int   r_val;

	if ( size == 0 ) return NULL;

#if BLIS_HEAP_ADDR_ALIGN_SIZE == 1
	p = malloc( ( size_t )size );
#elif defined(_WIN32)
	p = _aligned_malloc( ( size_t )size,
	                     ( size_t )BLIS_HEAP_ADDR_ALIGN_SIZE );
#else
	r_val = posix_memalign( &p,
	                        ( size_t )BLIS_HEAP_ADDR_ALIGN_SIZE,
	                        ( size_t )size );

	if ( r_val != 0 ) bli_abort();
#endif

	if ( p == NULL ) bli_abort();

	return p;
}

void bli_free( void* p )
{
#if BLIS_HEAP_ADDR_ALIGN_SIZE == 1 || !defined(_WIN32)
	free( p );
#else
	_aligned_free( p );
#endif
}

