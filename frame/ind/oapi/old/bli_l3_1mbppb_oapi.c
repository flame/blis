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

// -- gemmbp/gemmpb ------------------------------------------------------------

#undef  GENFRONT
#define GENFRONT( opname, imeth, alg ) \
\
void PASTEMAC2(opname,imeth,alg) \
     ( \
       obj_t*  alpha, \
       obj_t*  a, \
       obj_t*  b, \
       obj_t*  beta, \
       obj_t*  c  \
     ) \
{ \
	num_t   dt     = bli_obj_datatype( *c ); \
	cntx_t* cntx_p; \
	cntl_t* cntl_p; \
\
	/* If the objects are in the real domain, execute the native
	   implementation. */ \
	if ( bli_obj_is_real( *c ) ) \
	{ \
		PASTEMAC(opname,nat)( alpha, a, b, beta, c, NULL ); \
		return; \
	} \
\
	/* Query a context for the current induced method. This context is
	   managed and cached by the gks and should not be freed by the caller.
	   Note that we pass in the datatype because it will be needed when
	   bli_gks_query_ind_cntx() eventually calls the bli_ind_cntx_init()
	   family of functions. */ \
	cntx_p = bli_gks_query_ind_cntx( BLIS_1M, dt ); \
\
	/* We're not supposed to modify contexts given to us by the gks, but
	   we're going to cheat in order to change the context to one for bp
	   or pb. (The default is bp.) NOTE: THIS IS NOT THREAD-SAFE FOR
	   APPLICATION-LEVEL THREADING. */ \
	if ( PASTEMAC2(opname,imeth,alg) == \
	     PASTEMAC2(opname,imeth,pb) ) \
		PASTEMAC2(opname,pb,_cntx_init)( dt, cntx_p ); \
\
	/* Create a control tree for the current algorithm (bp or pb). */ \
	cntl_p = PASTEMAC2(opname,alg,_cntl_create)( BLIS_GEMM );  \
\
	/* Invoke the operation's front end using the context and control
	   tree we just created. */ \
	PASTEMAC(opname,_front)( alpha, a, b, beta, c, cntx_p, cntl_p ); \
\
	/* Free the control tree. Since the implementation will only make
	   copies of it (and not use it directly) we do not need to supply
	   a thread object. */ \
	bli_cntl_free( cntl_p, NULL ); \
}

// gemm
GENFRONT( gemm, 1m, bp )
GENFRONT( gemm, 1m, pb )

