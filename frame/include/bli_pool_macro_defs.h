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

#ifndef BLIS_POOL_MACRO_DEFS_H
#define BLIS_POOL_MACRO_DEFS_H


// Pool entry query

#define bli_pool_block_ptrs( pool_p ) \
\
	( (pool_p)->block_ptrs )

#define bli_pool_num_blocks( pool_p ) \
\
	( (pool_p)->num_blocks )

#define bli_pool_block_size( pool_p ) \
\
	( (pool_p)->block_size )

#define bli_pool_top_index( pool_p ) \
\
	( (pool_p)->top_index )

#define bli_pool_is_exhausted( pool_p ) \
\
	( bli_pool_top_index( pool_p ) == -1 )


// Pool entry modification

#define bli_pool_set_block_ptrs( block_ptrs0, pool_p ) \
{ \
    (pool_p)->block_ptrs = block_ptrs0; \
}

#define bli_pool_set_num_blocks( num_blocks0, pool_p ) \
{ \
    (pool_p)->num_blocks = num_blocks0; \
}

#define bli_pool_set_block_size( block_size0, pool_p ) \
{ \
    (pool_p)->block_size = block_size0; \
}

#define bli_pool_set_top_index( top_index0, pool_p ) \
{ \
    (pool_p)->top_index = top_index0; \
}

#define bli_pool_dec_top_index( pool_p ) \
{ \
    ((pool_p)->top_index)--; \
}

#define bli_pool_inc_top_index( pool_p ) \
{ \
    ((pool_p)->top_index)++; \
}

#define bli_pool_init( num_blocks, block_size, block_ptrs, pool_p ) \
{ \
	bli_pool_set_num_blocks( num_blocks, pool_p ); \
	bli_pool_set_block_size( block_size, pool_p ); \
	bli_pool_set_block_ptrs( block_ptrs, pool_p ); \
	bli_pool_set_top_index( num_blocks - 1, pool_p ); \
}


#endif 
