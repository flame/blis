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
void PASTEMAC(opname,imeth) \
     ( \
       obj_t*  alpha, \
       obj_t*  a, \
       obj_t*  b, \
       obj_t*  beta, \
       obj_t*  c, \
       cntx_t* cntx  \
     ) \
{ \
	cntx_t* cntx_p; \
	dim_t   i; \
\
	obj_t*  beta_use = beta; \
\
	/* If the objects are in the real domain, execute the native
	   implementation. */ \
	if ( bli_obj_is_real( *c ) ) \
	{ \
		PASTEMAC(opname,nat)( alpha, a, b, beta, c, cntx ); \
		return; \
	} \
\
	/* Initialize a local context if the one provided is NULL. */ \
	bli_cntx_init_local_if2( cname, imeth, cntx, cntx_p ); \
\
	/* Some induced methods execute in multiple "stages". */ \
	for ( i = 0; i < nstage; ++i ) \
	{ \
		/* Prepare the context for the ith stage of computation. */ \
		PASTEMAC2(cname,imeth,_cntx_stage)( i, cntx_p ); \
\
		/* For multi-stage methods, use BLIS_ONE as beta after the first
		   stage. */ \
		if ( i > 0 ) beta_use = &BLIS_ONE; \
\
		/* Invoke the operation's front end with the appropriate control
		   tree. */ \
		PASTEMAC(opname,_front)( alpha, a, b, beta_use, c, cntx_p, \
		                         PASTECH(cname,_cntl) ); \
	} \
\
	/* Finalize the local context if it was initialized here. */ \
	bli_cntx_finalize_local_if2( cname, imeth, cntx ); \
}

// gemm
GENFRONT( gemm, gemm, 3mh, 3 )
GENFRONT( gemm, gemm, 3m3, 1 )
GENFRONT( gemm, gemm, 3m2, 1 )
GENFRONT( gemm, gemm, 3m1, 1 )
GENFRONT( gemm, gemm, 4mh, 4 )
GENFRONT( gemm, gemm, 4mb, 1 )
GENFRONT( gemm, gemm, 4m1, 1 )

// her2k
GENFRONT( her2k, gemm, 3mh, 3 )
//GENFRONT( her2k, gemm, 3m3, 1 ) // Not implemented.
//GENFRONT( her2k, gemm, 3m2, 1 ) // Not implemented.
GENFRONT( her2k, gemm, 3m1, 1 )
GENFRONT( her2k, gemm, 4mh, 4 )
//GENFRONT( her2k, gemm, 4mb, 1 ) // Not implemented.
GENFRONT( her2k, gemm, 4m1, 1 )

// syr2k
GENFRONT( syr2k, gemm, 3mh, 3 )
//GENFRONT( syr2k, gemm, 3m3, 1 ) // Not implemented.
//GENFRONT( syr2k, gemm, 3m2, 1 ) // Not implemented.
GENFRONT( syr2k, gemm, 3m1, 1 )
GENFRONT( syr2k, gemm, 4mh, 4 )
//GENFRONT( syr2k, gemm, 4mb, 1 ) // Not implemented.
GENFRONT( syr2k, gemm, 4m1, 1 )


// -- hemm/symm/trmm3 ----------------------------------------------------------

#undef  GENFRONT
#define GENFRONT( opname, cname, imeth, nstage ) \
\
void PASTEMAC(opname,imeth) \
     ( \
       side_t  side, \
       obj_t*  alpha, \
       obj_t*  a, \
       obj_t*  b, \
       obj_t*  beta, \
       obj_t*  c, \
       cntx_t* cntx  \
     ) \
{ \
	cntx_t* cntx_p; \
	dim_t   i; \
\
	obj_t*  beta_use = beta; \
\
	/* If the objects are in the real domain, execute the native
	   implementation. */ \
	if ( bli_obj_is_real( *c ) ) \
	{ \
		PASTEMAC(opname,nat)( side, alpha, a, b, beta, c, cntx ); \
		return; \
	} \
\
	/* Initialize a local context if the one provided is NULL. */ \
	bli_cntx_init_local_if2( cname, imeth, cntx, cntx_p ); \
\
	/* Some induced methods execute in multiple "stages". */ \
	for ( i = 0; i < nstage; ++i ) \
	{ \
		/* Prepare the context for the ith stage of computation. */ \
		PASTEMAC2(cname,imeth,_cntx_stage)( i, cntx_p ); \
\
		/* For multi-stage methods, use BLIS_ONE as beta after the first
		   stage. */ \
		if ( i > 0 ) beta_use = &BLIS_ONE; \
\
		/* Invoke the operation's front end with the appropriate control
		   tree. */ \
		PASTEMAC(opname,_front)( side, alpha, a, b, beta_use, c, cntx_p, \
		                         PASTECH(cname,_cntl) ); \
	} \
\
	/* Finalize the local context if it was initialized here. */ \
	bli_cntx_finalize_local_if2( cname, imeth, cntx ); \
}

// hemm
GENFRONT( hemm, gemm, 3mh, 3 )
//GENFRONT( hemm, gemm, 3m3, 1 ) // Not implemented.
//GENFRONT( hemm, gemm, 3m2, 1 ) // Not implemented.
GENFRONT( hemm, gemm, 3m1, 1 )
GENFRONT( hemm, gemm, 4mh, 4 )
//GENFRONT( hemm, gemm, 4mb, 1 ) // Not implemented.
GENFRONT( hemm, gemm, 4m1, 1 )

// symm
GENFRONT( symm, gemm, 3mh, 3 )
//GENFRONT( symm, gemm, 3m3, 1 ) // Not implemented.
//GENFRONT( symm, gemm, 3m2, 1 ) // Not implemented.
GENFRONT( symm, gemm, 3m1, 1 )
GENFRONT( symm, gemm, 4mh, 4 )
//GENFRONT( symm, gemm, 4mb, 1 ) // Not implemented.
GENFRONT( symm, gemm, 4m1, 1 )

// trmm3
GENFRONT( trmm3, gemm, 3mh, 3 )
//GENFRONT( trmm3, gemm, 3m3, 1 ) // Not implemented.
//GENFRONT( trmm3, gemm, 3m2, 1 ) // Not implemented.
GENFRONT( trmm3, gemm, 3m1, 1 )
GENFRONT( trmm3, gemm, 4mh, 4 )
//GENFRONT( trmm3, gemm, 4mb, 1 ) // Not implemented.
GENFRONT( trmm3, gemm, 4m1, 1 )


// -- herk/syrk ----------------------------------------------------------------

#undef  GENFRONT
#define GENFRONT( opname, cname, imeth, nstage ) \
\
void PASTEMAC(opname,imeth) \
     ( \
       obj_t*  alpha, \
       obj_t*  a, \
       obj_t*  beta, \
       obj_t*  c, \
       cntx_t* cntx  \
     ) \
{ \
	cntx_t* cntx_p; \
	dim_t   i; \
\
	obj_t*  beta_use = beta; \
\
	/* If the objects are in the real domain, execute the native
	   implementation. */ \
	if ( bli_obj_is_real( *c ) ) \
	{ \
		PASTEMAC(opname,nat)( alpha, a, beta, c, cntx ); \
		return; \
	} \
\
	/* Initialize a local context if the one provided is NULL. */ \
	bli_cntx_init_local_if2( cname, imeth, cntx, cntx_p ); \
\
	/* Some induced methods execute in multiple "stages". */ \
	for ( i = 0; i < nstage; ++i ) \
	{ \
		/* Prepare the context for the ith stage of computation. */ \
		PASTEMAC2(cname,imeth,_cntx_stage)( i, cntx_p ); \
\
		/* For multi-stage methods, use BLIS_ONE as beta after the first
		   stage. */ \
		if ( i > 0 ) beta_use = &BLIS_ONE; \
\
		/* Invoke the operation's front end with the appropriate control
		   tree. */ \
		PASTEMAC(opname,_front)( alpha, a, beta_use, c, cntx_p, \
		                         PASTECH(cname,_cntl) ); \
	} \
\
	/* Finalize the local context if it was initialized here. */ \
	bli_cntx_finalize_local_if2( cname, imeth, cntx ); \
}

// herk
GENFRONT( herk, gemm, 3mh, 3 )
//GENFRONT( herk, gemm, 3m3, 1 ) // Not implemented.
//GENFRONT( herk, gemm, 3m2, 1 ) // Not implemented.
GENFRONT( herk, gemm, 3m1, 1 )
GENFRONT( herk, gemm, 4mh, 4 )
//GENFRONT( herk, gemm, 4mb, 1 ) // Not implemented.
GENFRONT( herk, gemm, 4m1, 1 )

// syrk
GENFRONT( syrk, gemm, 3mh, 3 )
//GENFRONT( syrk, gemm, 3m3, 1 ) // Not implemented.
//GENFRONT( syrk, gemm, 3m2, 1 ) // Not implemented.
GENFRONT( syrk, gemm, 3m1, 1 )
GENFRONT( syrk, gemm, 4mh, 4 )
//GENFRONT( syrk, gemm, 4mb, 1 ) // Not implemented.
GENFRONT( syrk, gemm, 4m1, 1 )


// -- trmm ---------------------------------------------------------------------

#undef  GENFRONT
#define GENFRONT( opname, cname, imeth, nstage ) \
\
void PASTEMAC(opname,imeth) \
     ( \
       side_t  side, \
       obj_t*  alpha, \
       obj_t*  a, \
       obj_t*  b, \
       cntx_t* cntx  \
     ) \
{ \
	cntx_t* cntx_p; \
	dim_t   i; \
\
	/* If the objects are in the real domain, execute the native
	   implementation. */ \
	if ( bli_obj_is_real( *b ) ) \
	{ \
		PASTEMAC(opname,nat)( side, alpha, a, b, cntx ); \
		return; \
	} \
\
	/* Initialize a local context if the one provided is NULL. */ \
	bli_cntx_init_local_if2( cname, imeth, cntx, cntx_p ); \
\
	/* Some induced methods execute in multiple "stages". */ \
	for ( i = 0; i < nstage; ++i ) \
	{ \
		/* Prepare the context for the ith stage of computation. */ \
		PASTEMAC2(cname,imeth,_cntx_stage)( i, cntx_p ); \
\
		/* Invoke the operation's front end with the appropriate control
		   tree. */ \
		PASTEMAC(opname,_front)( side, alpha, a, b, cntx_p, \
		                         PASTECH(cname,_cntl) ); \
	} \
\
	/* Finalize the local context if it was initialized here. */ \
	bli_cntx_finalize_local_if2( cname, imeth, cntx ); \
}

// trmm
//GENFRONT( trmm, gemm, 3mh, 3 ) // Unimplementable.
//GENFRONT( trmm, gemm, 3m3, 1 ) // Unimplementable.
//GENFRONT( trmm, gemm, 3m2, 1 ) // Unimplementable.
GENFRONT( trmm, gemm, 3m1, 1 )
//GENFRONT( trmm, gemm, 4mh, 4 ) // Unimplementable.
//GENFRONT( trmm, gemm, 4mb, 1 ) // Unimplementable.
GENFRONT( trmm, gemm, 4m1, 1 )


// -- trsm ---------------------------------------------------------------------

#undef  GENFRONT
#define GENFRONT( opname, cname, imeth, nstage ) \
\
void PASTEMAC(opname,imeth) \
     ( \
       side_t  side, \
       obj_t*  alpha, \
       obj_t*  a, \
       obj_t*  b, \
       cntx_t* cntx  \
     ) \
{ \
	cntx_t* cntx_p; \
\
	/* If the objects are in the real domain, execute the native
	   implementation. */ \
	if ( bli_obj_is_real( *b ) ) \
	{ \
		PASTEMAC(opname,nat)( side, alpha, a, b, cntx ); \
		return; \
	} \
\
	/* Initialize a local context if the one provided is NULL. */ \
	bli_cntx_init_local_if2( cname, imeth, cntx, cntx_p ); \
\
	{ \
		/* NOTE: trsm cannot be implemented via any induced method that
		   needs to execute in stages (e.g. 3mh, 4mh). */ \
\
		/* Invoke the operation's front end with the appropriate control
		   tree. */ \
		PASTEMAC(opname,_front)( side, alpha, a, b, cntx_p, \
		                         PASTECH(cname,_l_cntl), \
		                         PASTECH(cname,_r_cntl) ); \
	} \
\
	/* Finalize the local context if it was initialized here. */ \
	bli_cntx_finalize_local_if2( cname, imeth, cntx ); \
}

// trsm
//GENFRONT( trmm, trsm, 3mh, 3 ) // Unimplementable.
//GENFRONT( trmm, trsm, 3m3, 1 ) // Unimplementable.
//GENFRONT( trmm, trsm, 3m2, 1 ) // Unimplementable.
GENFRONT( trsm, trsm, 3m1, 1 )
//GENFRONT( trmm, trsm, 4mh, 4 ) // Unimplementable.
//GENFRONT( trmm, trsm, 4mb, 1 ) // Unimplementable.
GENFRONT( trsm, trsm, 4m1, 1 )


//
// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------
//

