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

#include "bli_cntl_init.h"

typedef enum
{
	BLIS_UNBLOCKED = 0,
	BLIS_UNB_FUSED = 1,
	BLIS_UNB_OPT   = 1,
	BLIS_BLOCKED   = 2
} impl_t;

typedef enum
{
	BLIS_VARIANT1 = 0,
	BLIS_VARIANT2,
	BLIS_VARIANT3,
	BLIS_VARIANT4,
	BLIS_VARIANT5,
	BLIS_VARIANT6,
	BLIS_VARIANT7,
	BLIS_VARIANT8,
	BLIS_VARIANT9,
} varnum_t;


void bli_cntl_obj_free( void* cntl );



// -- Control tree accessor macros (common to many node types) --

#define cntl_impl_type( cntl )     cntl->impl_type
#define cntl_var_num( cntl )       cntl->var_num
#define cntl_blocksize( cntl )     cntl->b



// -- Control tree query macros --

#define cntl_is_noop( cntl ) \
\
	( cntl == NULL )

#define cntl_is_leaf( cntl ) \
\
	( cntl_impl_type( cntl ) != BLIS_BLOCKED )

#define cntl_is_blocked( cntl ) \
\
	( cntl_impl_type( cntl ) == BLIS_BLOCKED )

