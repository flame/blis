/*

   BLIS    
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2014, The University of Texas at Austin
   Copyright (C) 2016 Hewlett Packard Enterprise Development LP

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

#ifndef BLIS_MEMBRK_H
#define BLIS_MEMBRK_H


#define bli_membrk_pool( pool_index, membrk_p ) \
\
	( (membrk_p)->pools + (pool_index) )

#define bli_membrk_mutex( membrk_p ) \
\
	( &( (membrk_p)->mutex ) )

#define bli_membrk_malloc_fp( membrk_p ) \
\
	( (membrk_p)->malloc_fp )

#define bli_membrk_free_fp( membrk_p ) \
\
	( (membrk_p)->free_fp )

#define bli_membrk_set_malloc_fp( _malloc_fp, membrk_p ) \
{\
	(membrk_p)->malloc_fp = _malloc_fp; \
}

#define bli_membrk_set_free_fp( _free_fp, membrk_p ) \
{\
	(membrk_p)->free_fp = _free_fp; \
}

#define bli_membrk_lock( membrk_p ) \
{\
	bli_mutex_lock( &((membrk_p)->mutex) ); \
}

#define bli_membrk_unlock( membrk_p ) \
{\
	bli_mutex_unlock( &((membrk_p)->mutex) ); \
}

#define bli_membrk_malloc( size, membrk ) \
\
	/* Call the malloc()-style function in membrk. */ \
	((membrk)->malloc_fp)( size )

#define bli_membrk_free( buf_p, membrk ) \
\
	/* Call the free()-style function in membrk. */ \
	((membrk)->free_fp)( buf_p )


// -----------------------------------------------------------------------------

void bli_membrk_init
     (
       cntx_t*   cntx,
       membrk_t* membrk
     );
void bli_membrk_finalize
     (
       membrk_t* membrk
     );

void bli_membrk_acquire_m
     (
       membrk_t* membrk,
       siz_t     req_size,
       packbuf_t buf_type,
       mem_t*    mem
     );

void bli_membrk_acquire_v
     (
       membrk_t* membrk,
       siz_t     req_size,
       mem_t*    mem
     );

void bli_membrk_release
     (
       mem_t* mem
     );

siz_t bli_membrk_pool_size
     (
       membrk_t* membrk,
       packbuf_t buf_type
     );

// ----------------------------------------------------------------------------

void bli_membrk_init_pools
     (
       cntx_t*   cntx,
       membrk_t* membrk
     );
void bli_membrk_reinit_pools
     (
       cntx_t*   cntx,
       membrk_t* membrk
     );
void bli_membrk_finalize_pools
     (
       membrk_t* membrk
     );

void bli_membrk_compute_pool_block_sizes
     (
       siz_t*  bs_a,
       siz_t*  bs_b,
       siz_t*  bs_c,
       cntx_t* cntx
     );
void bli_membrk_compute_pool_block_sizes_dt
     (
       num_t   dt,
       siz_t*  bs_a,
       siz_t*  bs_b,
       siz_t*  bs_c,
       cntx_t* cntx
     );

#endif

