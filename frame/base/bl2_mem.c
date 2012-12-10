/*

   BLIS    
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2012, The University of Texas

   Redistribution and use in source and binary forms, with or without
   modification, are permitted provided that the following conditions are
   met:
    - Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.
    - Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.
    - Neither the name of The University of Texas nor the names of its
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

#include "blis2.h"

#define N_ELEM_SMEM ( BLIS_STATIC_MEM_POOL_SIZE / sizeof( double ) )

double  smem[ N_ELEM_SMEM ];

double* mc      = smem;
int     counter = 0;


// -- Memory Manager -----------------------------------------------------------

void bl2_mm_acquire_v( num_t  dt,
                       dim_t  m,
                       mem_t* mem )
{
	siz_t elem_size = bl2_datatype_size( dt );
	siz_t buf_size;

	// Compute the number of bytes needed for an m-length vector of type dt.
	buf_size       = m * elem_size;

	mem->buf       = bl2_malloc( buf_size );
	mem->m         = m;
	mem->n         = 1;
}

void bl2_mm_acquire_m( num_t  dt,
                       dim_t  m,
                       dim_t  n,
                       mem_t* mem )
{
	siz_t elem_size = bl2_datatype_size( dt );
	siz_t buf_size;

	// Compute the number of bytes needed for an m x n matrix of type dt.
	buf_size       = ( m * n ) * elem_size + BLIS_MAX_PREFETCH_BYTE_OFFSET;

#if 0
	mem->buf       = bl2_malloc( buf_size );
#else
	mem->buf       = bl2_malloc_s( buf_size );
#endif
	mem->m         = m;
	mem->n         = n;
}


void bl2_mm_release( mem_t* mem )
{
#if 0
	bl2_free( mem->buf );
#else
	bl2_free_s( mem->buf );
#endif
	mem->m         = 0;
	mem->n         = 0;
}


void* bl2_malloc_s( siz_t buf_size )
{
	void* rmem;
	siz_t padding;

	if ( ( siz_t )mc % BLIS_PAGE_SIZE != 0 )
	{
		// We assume mc begins 16 byte-aligned. If this assumption holds, then
		// mc % 4096 is also a multiple of 16 bytes. and so padding is also a 
		// multiple of 16 bytes. Thus, we don't need to adjust mc any further
		// after the following two lines.
		padding = BLIS_PAGE_SIZE - ( ( siz_t )mc % BLIS_PAGE_SIZE );
		mc += padding / sizeof( double );
	}

	rmem = ( void* )mc;
	mc += ( buf_size / sizeof( double ) );

	if ( mc >= smem + ( N_ELEM_SMEM ) )
		bl2_abort();

	++counter;

	return rmem;
}

void bl2_free_s( void* p )
{
	--counter;

	if ( counter == 0 )
		mc = smem;
}

void bl2_mm_clear_smem( void )
{
	dim_t n = N_ELEM_SMEM;
	dim_t i;

	for ( i = 0; i < n; ++i )
	{
		smem[i] = 0.0;
	}
}

// -- Memory pool implementation -----------------------------------------------

