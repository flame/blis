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

#ifndef BLIS_MEM_H
#define BLIS_MEM_H


// Mem entry query

static pblk_t* bli_mem_pblk( mem_t* mem )
{
	return &(mem->pblk);
}

static void* bli_mem_buffer( mem_t* mem )
{
	return bli_pblk_buf_align( bli_mem_pblk( mem ) );
}

static void* bli_mem_buf_sys( mem_t* mem )
{
	return bli_pblk_buf_sys( bli_mem_pblk( mem ) );
}

static packbuf_t bli_mem_buf_type( mem_t* mem )
{
	return mem->buf_type;
}

static pool_t* bli_mem_pool( mem_t* mem )
{
	return mem->pool;
}

static membrk_t* bli_mem_membrk( mem_t* mem )
{
	return mem->membrk;
}

static siz_t bli_mem_size( mem_t* mem )
{
	return mem->size;
}

static bool_t bli_mem_is_alloc( mem_t* mem )
{
	return ( bool_t )
	       ( bli_mem_buffer( mem ) != NULL );
}

static bool_t bli_mem_is_unalloc( mem_t* mem )
{
	return ( bool_t )
	       ( bli_mem_buffer( mem ) == NULL );
}


// Mem entry modification

static void bli_mem_set_pblk( pblk_t* pblk, mem_t* mem )
{
	mem->pblk = *pblk;
}

static void bli_mem_set_buffer( void* buf, mem_t* mem )
{
	bli_pblk_set_buf_align( buf, &(mem->pblk) );
}

static void bli_mem_set_buf_sys( void* buf, mem_t* mem )
{
	bli_pblk_set_buf_sys( buf, &(mem->pblk) );
}

static void bli_mem_set_buf_type( packbuf_t buf_type, mem_t* mem )
{
	mem->buf_type = buf_type;
}

static void bli_mem_set_pool( pool_t* pool, mem_t* mem )
{
	mem->pool = pool;
}

static void bli_mem_set_membrk( membrk_t* membrk, mem_t* mem )
{
	mem->membrk = membrk;
}

static void bli_mem_set_size( siz_t size, mem_t* mem )
{
	mem->size = size;
}

static void bli_mem_clear( mem_t* mem )
{
	bli_mem_set_buffer( NULL, mem );
	bli_mem_set_buf_sys( NULL, mem );
	bli_mem_set_pool( NULL, mem );
	bli_mem_set_size( 0, mem );
	bli_mem_set_membrk( NULL, mem );
}


#endif 
