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

#ifndef BLIS_MEM_MACRO_DEFS_H
#define BLIS_MEM_MACRO_DEFS_H


// Mem entry query

#define bli_mem_buffer( mem_p ) \
\
	( (mem_p)->buf )

#define bli_mem_buf_type( mem_p ) \
\
    ( (mem_p)->buf_type )

#define bli_mem_pool( mem_p ) \
\
    ( (mem_p)->pool )

#define bli_mem_size( mem_p ) \
\
	( (mem_p)->size )

#define bli_mem_is_alloc( mem_p ) \
\
	( bli_mem_buffer( mem_p ) != NULL )

#define bli_mem_is_unalloc( mem_p ) \
\
	( bli_mem_buffer( mem_p ) == NULL )


// Mem entry modification

#define bli_mem_set_buffer( buf0, mem_p ) \
{ \
    mem_p->buf = buf0; \
}

#define bli_mem_set_buf_type( buf_type0, mem_p ) \
{ \
    mem_p->buf_type = buf_type0; \
}

#define bli_mem_set_pool( pool0, mem_p ) \
{ \
    mem_p->pool = pool0; \
}

#define bli_mem_set_size( size0, mem_p ) \
{ \
    mem_p->size = size0; \
}


#endif 
