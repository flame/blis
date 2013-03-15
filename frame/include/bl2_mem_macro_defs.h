/*

   BLIS    
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2013, The University of Texas

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

#ifndef BLIS_MEM_MACRO_DEFS_H
#define BLIS_MEM_MACRO_DEFS_H


// Mem entry query

#define bl2_mem_buffer( mem_p ) \
\
	( (mem_p)->buf )

#define bl2_mem_buf_type( mem_p ) \
\
    ( (mem_p)->buf_type )

#define bl2_mem_pool( mem_p ) \
\
    ( (mem_p)->pool )

#define bl2_mem_size( mem_p ) \
\
	( (mem_p)->size )

#define bl2_mem_length( mem_p ) \
\
	( (mem_p)->m )

#define bl2_mem_width( mem_p ) \
\
	( (mem_p)->n )

#define bl2_mem_elem_size( mem_p ) \
\
	( (mem_p)->elem_size )

#define bl2_mem_is_alloc( mem_p ) \
\
	( bl2_mem_buffer( mem_p ) != NULL )

#define bl2_mem_is_unalloc( mem_p ) \
\
	( bl2_mem_buffer( mem_p ) == NULL )


// Mem entry modification

#define bl2_mem_set_buffer( buf0, mem_p ) \
{ \
    mem_p->buf = buf0; \
}

#define bl2_mem_set_buf_type( buf_type0, mem_p ) \
{ \
    mem_p->buf_type = buf_type0; \
}

#define bl2_mem_set_pool( pool0, mem_p ) \
{ \
    mem_p->pool = pool0; \
}

#define bl2_mem_set_size( size0, mem_p ) \
{ \
    mem_p->size = size0; \
}

#define bl2_mem_set_length( m0, mem_p ) \
{ \
    mem_p->m = m0; \
}

#define bl2_mem_set_width( n0, mem_p ) \
{ \
    mem_p->n = n0; \
}

#define bl2_mem_set_elem_size( elem_size0, mem_p ) \
{ \
    mem_p->elem_size = elem_size0; \
}

#define bl2_mem_set_dims( m0, n0, mem_p ) \
{ \
    bl2_mem_set_length( m0, mem_p ); \
    bl2_mem_set_width( n0, mem_p ); \
}


// Allocate a mem_t object if it is unallocated, or update its dimensions
// if it is allocated. This macro is used for matrices.

#define bl2_mem_alloc_update_m( m_padded, n_padded, elem_size, buf_type, mem_p ) \
{ \
	bool_t needs_alloc; \
	siz_t  size_needed; \
\
	if ( bl2_mem_is_unalloc( mem_p ) ) \
	{ \
		/* If the mem_t object is currently unallocated (NULL), mark it for
		   allocation. */ \
		needs_alloc = TRUE; \
	} \
	else \
	{ \
		/* Compute the total buffer size needed. */ \
		size_needed = m_padded * n_padded * elem_size; \
\
		if ( size_needed <= bl2_mem_size( mem_p ) ) \
		{ \
			/* If the mem_t object is currently allocated, AND what is
			   allocated and available is equal to or greater than what is
			   needed, then set the dimensions according to how much we
			   need. This allows us to avoid unnecessarily releasing and
			   re-allocating when all we need is a subset of what is already
			   available. This case will occur when, for example, handling
			   both forward and backward edge cases. */ \
			bl2_mem_set_dims( m_padded, n_padded, mem_p ); \
\
			needs_alloc = FALSE; \
		} \
		else /* if ( bl2_mem_size( mem_p ) < size_needed ) */ \
		{ \
			/* If the mem_t object is currently allocated and smaller than is
			   needed, then something is very wrong, since the cache blocksizes
			   that drive the level-3 blocked algorithms are the same ones that
			   determine the sizes of the blocks within our memory allocator's
			   memory pools. This branch should never be executed. */ \
			bl2_abort(); \
\
			needs_alloc = FALSE; \
		} \
	} \
\
	if ( needs_alloc ) \
	{ \
		bl2_mem_acquire_m( m_padded, \
		                   n_padded, \
		                   elem_size, \
		                   buf_type, \
		                   mem_p ); \
	} \
} \


// Allocate a mem_t object if it is unallocated, or update its dimensions
// if it is allocated. This macro is used for vectors.

#define bl2_mem_alloc_update_v( m_padded, elem_size, mem_p ) \
{ \
	bool_t needs_alloc; \
	siz_t  size_needed; \
\
	if ( bl2_mem_is_unalloc( mem_p ) ) \
	{ \
		/* If the mem_t object is currently unallocated (NULL), mark it for
		   allocation. */ \
		needs_alloc = TRUE; \
	} \
	else \
	{ \
		/* Compute the total buffer size needed. */ \
		size_needed = m_padded * elem_size; \
\
		if ( size_needed <= bl2_mem_size( mem_p ) ) \
		{ \
			/* If the mem_t object is currently allocated, AND what is
			   allocated and available is equal to or larger than what is
			   needed, then set the dimension according to how much we
			   need. This allows us to avoid unnecessarily releasing and
			   re-allocating when all we need is a subset of what is already
			   available. This case will occur when, for example, handling
			   both forward and backward edge cases. */ \
			bl2_mem_set_dims( m_padded, 1, mem_p ); \
\
			needs_alloc = FALSE; \
		} \
		else /* if ( bl2_mem_size( mem_p ) < size_needed ) */ \
		{ \
			/* If the mem_t object is currently allocated and smaller than is
			   needed, then release the memory and re-allocate. */ \
			bl2_mem_release( mem_p ); \
\
			needs_alloc = TRUE; \
		} \
	} \
\
	if ( needs_alloc ) \
	{ \
		bl2_mem_acquire_v( m_padded, \
		                   elem_size, \
		                   mem_p ); \
	} \
} \



#endif 
