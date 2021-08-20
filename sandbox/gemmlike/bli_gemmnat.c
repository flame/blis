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
    - Neither the name(s) of the copyright holder(s) nor the names of its
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

// Given the current architecture of BLIS sandboxes, bli_gemmnat() is the
// entry point to any sandbox implementation.

// NOTE: This function is implemented identically to the function that it
// overrides in frame/ind/oapi/bli_l3_nat_oapi.c. This means that we are
// forgoing the option of customizing the implementations that underlie
// bli_gemm() and bli_?gemm(). Any new code defined in this sandbox
// directory, however, will be included in the BLIS.

#include "blis.h"

#undef  GENFRONT
#define GENFRONT( opname, cname, imeth ) \
\
void PASTEMAC(opname,imeth) \
     ( \
       obj_t*  alpha, \
       obj_t*  a, \
       obj_t*  b, \
       obj_t*  beta, \
       obj_t*  c, \
       cntx_t* cntx, \
       rntm_t* rntm  \
     ) \
{ \
\
	/* A switch to easily toggle whether we use the sandbox implementation
	   of bls_gemm() as the implementation for bli_gemm(). (This allows for
	   easy testing of bls_gemm() via the testsuite.) */ \
	if ( 1 ) \
	{ \
		bls_gemm_ex( alpha, a, b, beta, c, cntx, rntm ); \
		return; \
	} \
\
	bli_init_once(); \
\
	/* Obtain a valid (native) context from the gks if necessary. */ \
	if ( cntx == NULL ) cntx = bli_gks_query_cntx(); \
\
	/* Initialize a local runtime with global settings if necessary. Note
	   that in the case that a runtime is passed in, we make a local copy. */ \
	rntm_t rntm_l; \
	if ( rntm == NULL ) { bli_rntm_init_from_global( &rntm_l ); rntm = &rntm_l; } \
	else                { rntm_l = *rntm;                       rntm = &rntm_l; } \
\
	/* Invoke the operation's front end. */ \
	PASTEMAC(opname,_front) \
	( \
	  alpha, a, b, beta, c, cntx, rntm, NULL \
	); \
}

GENFRONT( gemm, gemm, nat )
