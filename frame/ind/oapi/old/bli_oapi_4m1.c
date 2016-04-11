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

// Bring control trees into scope.
extern gemm_t* gemm_cntl;
extern trsm_t* trsm_l_cntl;
extern trsm_t* trsm_r_cntl;


// -- gemm/her2k/syr2k ---------------------------------------------------------

#undef  GENFRONT
#define GENFRONT( opname, cname, imeth, nstage ) \
\
void PASTEMAC(opname,imeth)( \
                             obj_t*  alpha, \
                             obj_t*  a, \
                             obj_t*  b, \
                             obj_t*  beta, \
                             obj_t*  c \
                           ) \
{ \
	cntx_t cntx; \
	dim_t  i; \
\
	/* If the objects are in the real domain, execute the native
	   implementation. */ \
	if ( bli_obj_is_real( *c ) ) \
	{ \
		PASTEMAC(opname,nat)( alpha, a, b, beta, c ); \
		return; \
	} \
\
	/* Initialize the context. */ \
	PASTEMAC2(cname,imeth,_cntx_stage)( i, &cntx ); \
\
	/* Some induced methods (e.g. 3mh and 4mh) execute in multiple
	   "stages". */ \
	for ( i = 0; i < nstage; ++i ) \
	{ \
		/* Prepare the context for the ith stage of computation. */ \
		PASTEMAC2(cname,imeth,_cntx_stage)( i, &cntx ); \
\
		/* Invoke the operation's front end with the appropriate control
		   tree. */ \
		PASTEMAC(opname,_front)( alpha, a, b, beta, c, &cntx, \
		                         PASTECH(cname,_cntl) ); \
	} \
\
	/* Finalize the context. */ \
	PASTEMAC2(cname,imeth,_cntx_finalize)( &cntx ); \
}

GENFRONT( gemm, gemm, 4m1, 1 )
GENFRONT( her2k, gemm, 4m1, 1 )
GENFRONT( syr2k, gemm, 4m1, 1 )


// -- hemm/symm/trmm3 ----------------------------------------------------------

#undef  GENFRONT
#define GENFRONT( opname, cname, imeth, nstage ) \
\
void PASTEMAC(opname,imeth)( \
                             side_t  side, \
                             obj_t*  alpha, \
                             obj_t*  a, \
                             obj_t*  b, \
                             obj_t*  beta, \
                             obj_t*  c \
                           ) \
{ \
	cntx_t cntx; \
	dim_t  i; \
\
	/* If the objects are in the real domain, execute the native
	   implementation. */ \
	if ( bli_obj_is_real( *c ) ) \
	{ \
		PASTEMAC(opname,nat)( side, alpha, a, b, beta, c ); \
		return; \
	} \
\
	/* Initialize the context. */ \
	PASTEMAC2(cname,imeth,_cntx_stage)( i, &cntx ); \
\
	/* Some induced methods (e.g. 3mh and 4mh) execute in multiple
	   "stages". */ \
	for ( i = 0; i < nstage; ++i ) \
	{ \
		/* Prepare the context for the ith stage of computation. */ \
		PASTEMAC2(cname,imeth,_cntx_stage)( i, &cntx ); \
\
		/* Invoke the operation's front end with the appropriate control
		   tree. */ \
		PASTEMAC(opname,_front)( side, alpha, a, b, beta, c, &cntx, \
		                         PASTECH(cname,_cntl) ); \
	} \
\
	/* Finalize the context. */ \
	PASTEMAC2(cname,imeth,_cntx_finalize)( &cntx ); \
}

GENFRONT( hemm, gemm, 4m1, 1 )
GENFRONT( symm, gemm, 4m1, 1 )
GENFRONT( trmm3, gemm, 4m1, 1 )


// -- herk/syrk ----------------------------------------------------------------

#undef  GENFRONT
#define GENFRONT( opname, cname, imeth, nstage ) \
\
void PASTEMAC(opname,imeth)( \
                             obj_t*  alpha, \
                             obj_t*  a, \
                             obj_t*  beta, \
                             obj_t*  c \
                           ) \
{ \
	cntx_t cntx; \
	dim_t  i; \
\
	/* If the objects are in the real domain, execute the native
	   implementation. */ \
	if ( bli_obj_is_real( *c ) ) \
	{ \
		PASTEMAC(opname,nat)( alpha, a, beta, c ); \
		return; \
	} \
\
	/* Initialize the context. */ \
	PASTEMAC2(cname,imeth,_cntx_stage)( i, &cntx ); \
\
	/* Some induced methods (e.g. 3mh and 4mh) execute in multiple
	   "stages". */ \
	for ( i = 0; i < nstage; ++i ) \
	{ \
		/* Prepare the context for the ith stage of computation. */ \
		PASTEMAC2(cname,imeth,_cntx_stage)( i, &cntx ); \
\
		/* Invoke the operation's front end with the appropriate control
		   tree. */ \
		PASTEMAC(opname,_front)( alpha, a, beta, c, &cntx, \
		                         PASTECH(cname,_cntl) ); \
	} \
\
	/* Finalize the context. */ \
	PASTEMAC2(cname,imeth,_cntx_finalize)( &cntx ); \
}

GENFRONT( herk, gemm, 4m1, 1 )
GENFRONT( syrk, gemm, 4m1, 1 )


// -- trmm ---------------------------------------------------------------------

#undef  GENFRONT
#define GENFRONT( opname, cname, imeth, nstage ) \
\
void PASTEMAC(opname,imeth)( \
                             side_t  side, \
                             obj_t*  alpha, \
                             obj_t*  a, \
                             obj_t*  b  \
                           ) \
{ \
	cntx_t cntx; \
	dim_t  i; \
\
	/* If the objects are in the real domain, execute the native
	   implementation. */ \
	if ( bli_obj_is_real( *b ) ) \
	{ \
		PASTEMAC(opname,nat)( side, alpha, a, b ); \
		return; \
	} \
\
	/* Initialize the context. */ \
	PASTEMAC2(cname,imeth,_cntx_stage)( i, &cntx ); \
\
	/* Some induced methods (e.g. 3mh and 4mh) execute in multiple
	   "stages". */ \
	for ( i = 0; i < nstage; ++i ) \
	{ \
		/* Prepare the context for the ith stage of computation. */ \
		PASTEMAC2(cname,imeth,_cntx_stage)( i, &cntx ); \
\
		/* Invoke the operation's front end with the appropriate control
		   tree. */ \
		PASTEMAC(opname,_front)( side, alpha, a, b, &cntx, \
		                         PASTECH(cname,_cntl) ); \
	} \
\
	/* Finalize the context. */ \
	PASTEMAC2(cname,imeth,_cntx_finalize)( &cntx ); \
}

GENFRONT( trmm, gemm, 4m1, 1 )


// -- trsm ---------------------------------------------------------------------

#undef  GENFRONT
#define GENFRONT( opname, cname, imeth, nstage ) \
\
void PASTEMAC(opname,imeth)( \
                             side_t  side, \
                             obj_t*  alpha, \
                             obj_t*  a, \
                             obj_t*  b  \
                           ) \
{ \
	cntx_t cntx; \
	dim_t  i; \
\
	/* If the objects are in the real domain, execute the native
	   implementation. */ \
	if ( bli_obj_is_real( *b ) ) \
	{ \
		PASTEMAC(opname,nat)( side, alpha, a, b ); \
		return; \
	} \
\
	/* Initialize the context. */ \
	PASTEMAC2(cname,imeth,_cntx_stage)( i, &cntx ); \
\
	/* Some induced methods (e.g. 3mh and 4mh) execute in multiple
	   "stages". */ \
	for ( i = 0; i < nstage; ++i ) \
	{ \
		/* Prepare the context for the ith stage of computation. */ \
		PASTEMAC2(cname,imeth,_cntx_stage)( i, &cntx ); \
\
		/* Invoke the operation's front end with the appropriate control
		   tree. */ \
		PASTEMAC(opname,_front)( side, alpha, a, b, &cntx, \
		                         PASTECH(cname,_l_cntl), \
		                         PASTECH(cname,_r_cntl) ); \
	} \
\
	/* Finalize the context. */ \
	PASTEMAC2(cname,imeth,_cntx_finalize)( &cntx ); \
}

GENFRONT( trsm, trsm, 4m1, 1 )

