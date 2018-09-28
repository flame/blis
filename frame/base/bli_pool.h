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

#ifndef BLIS_POOL_H
#define BLIS_POOL_H

// -- Pool block type --

/*
typedef struct
{
    void* buf_sys;
    void* buf_align;
} pblk_t;
*/

// -- Pool type --

/*
typedef struct
{
    pblk_t* block_ptrs;
    dim_t   block_ptrs_len;

    dim_t   top_index;
    dim_t   num_blocks;

    siz_t   block_size;
    siz_t   align_size;
} pool_t;
*/


// Pool block query

static void* bli_pblk_buf_sys( pblk_t* pblk )
{
    return pblk->buf_sys;
}

static void* bli_pblk_buf_align( pblk_t* pblk )
{
    return pblk->buf_align;
}

// Pool block modification

static void bli_pblk_set_buf_sys( void* buf_sys, pblk_t* pblk )
{
    pblk->buf_sys = buf_sys;
}

static void bli_pblk_set_buf_align( void* buf_align, pblk_t* pblk )
{
    pblk->buf_align = buf_align;
}

static void bli_pblk_clear( pblk_t* pblk )
{
	bli_pblk_set_buf_sys( NULL, pblk );
	bli_pblk_set_buf_align( NULL, pblk );
}


// Pool entry query

static pblk_t* bli_pool_block_ptrs( pool_t* pool )
{
	return pool->block_ptrs;
}

static dim_t bli_pool_block_ptrs_len( pool_t* pool )
{
	return pool->block_ptrs_len;
}

static dim_t bli_pool_num_blocks( pool_t* pool )
{
	return pool->num_blocks;
}

static siz_t bli_pool_block_size( pool_t* pool )
{
	return pool->block_size;
}

static siz_t bli_pool_align_size( pool_t* pool )
{
	return pool->align_size;
}

static dim_t bli_pool_top_index( pool_t* pool )
{
	return pool->top_index;
}

static bool_t bli_pool_is_exhausted( pool_t* pool )
{
	return ( bool_t )
	       ( bli_pool_top_index( pool ) == bli_pool_num_blocks( pool ) );
}

// Pool entry modification

static void bli_pool_set_block_ptrs( pblk_t* block_ptrs, pool_t* pool ) \
{
    pool->block_ptrs = block_ptrs;
}

static void bli_pool_set_block_ptrs_len( dim_t block_ptrs_len, pool_t* pool ) \
{
    pool->block_ptrs_len = block_ptrs_len;
}

static void bli_pool_set_num_blocks( dim_t num_blocks, pool_t* pool ) \
{
    pool->num_blocks = num_blocks;
}

static void bli_pool_set_block_size( siz_t block_size, pool_t* pool ) \
{
    pool->block_size = block_size;
}

static void bli_pool_set_align_size( siz_t align_size, pool_t* pool ) \
{
    pool->align_size = align_size;
}

static void bli_pool_set_top_index( dim_t top_index, pool_t* pool ) \
{
    pool->top_index = top_index;
}

// -----------------------------------------------------------------------------

void bli_pool_init( dim_t   num_blocks,
                    siz_t   block_size,
                    siz_t   align_size,
                    pool_t* pool );
void bli_pool_finalize( pool_t* pool );
void bli_pool_reinit( dim_t   num_blocks_new,
                      siz_t   block_size_new,
                      siz_t   align_size_new,
                      pool_t* pool );

void bli_pool_checkout_block( siz_t req_size, pblk_t* block, pool_t* pool );
void bli_pool_checkin_block( pblk_t* block, pool_t* pool );

void bli_pool_grow( dim_t num_blocks_add, pool_t* pool );
void bli_pool_shrink( dim_t num_blocks_sub, pool_t* pool );

void bli_pool_alloc_block( siz_t   block_size,
                           siz_t   align_size,
                           pblk_t* block );
void bli_pool_free_block( pblk_t* block );

void bli_pool_print( pool_t* pool );
void bli_pblk_print( pblk_t* pblk );

#endif

