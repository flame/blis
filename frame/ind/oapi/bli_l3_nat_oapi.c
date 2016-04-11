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

// NOTE: The function definitions in this file can be consolidated with the
// definitions for the other induced methods. The only advantage of keeping
// them separate is that it allows us to avoid the very small loop overhead
// of executing one iteration of a for loop, plus the overhead of calling a
// function that does nothing (ie: the _cntx_init_stage() function).

// -- gemm/her2k/syr2k ---------------------------------------------------------

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
       cntx_t* cntx  \
     ) \
{ \
	cntx_t* cntx_p; \
\
	/* Initialize a local context if the one provided is NULL. */ \
	bli_cntx_init_local_if2( cname, imeth, cntx, cntx_p ); \
\
	/* Invoke the operation's front end with the appropriate control
	   tree. */ \
	PASTEMAC(opname,_front) \
	( \
	  alpha, a, b, beta, c, cntx_p, \
	  PASTECH(cname,_cntl) \
	); \
\
	/* Finalize the local context if it was initialized here. */ \
	bli_cntx_finalize_local_if2( cname, imeth, cntx ); \
}

GENFRONT( gemm, gemm, nat )
GENFRONT( her2k, gemm, nat )
GENFRONT( syr2k, gemm, nat )


// -- hemm/symm/trmm3 ----------------------------------------------------------

#undef  GENFRONT
#define GENFRONT( opname, cname, imeth ) \
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
\
	/* Initialize a local context if the one provided is NULL. */ \
	bli_cntx_init_local_if2( cname, imeth, cntx, cntx_p ); \
\
	/* Invoke the operation's front end with the appropriate control
	   tree. */ \
	PASTEMAC(opname,_front) \
	( \
	  side, alpha, a, b, beta, c, cntx_p, \
	  PASTECH(cname,_cntl) \
	); \
\
	/* Finalize the local context if it was initialized here. */ \
	bli_cntx_finalize_local_if2( cname, imeth, cntx ); \
}

GENFRONT( hemm, gemm, nat )
GENFRONT( symm, gemm, nat )
GENFRONT( trmm3, gemm, nat )


// -- herk/syrk ----------------------------------------------------------------

#undef  GENFRONT
#define GENFRONT( opname, cname, imeth ) \
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
\
	/* Initialize a local context if the one provided is NULL. */ \
	bli_cntx_init_local_if2( cname, imeth, cntx, cntx_p ); \
\
	/* Invoke the operation's front end with the appropriate control
	   tree. */ \
	PASTEMAC(opname,_front) \
	( \
	  alpha, a, beta, c, cntx_p, \
	  PASTECH(cname,_cntl) \
	); \
\
	/* Finalize the local context if it was initialized here. */ \
	bli_cntx_finalize_local_if2( cname, imeth, cntx ); \
}

GENFRONT( herk, gemm, nat )
GENFRONT( syrk, gemm, nat )


// -- trmm ---------------------------------------------------------------------

#undef  GENFRONT
#define GENFRONT( opname, cname, imeth ) \
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
	/* Initialize a local context if the one provided is NULL. */ \
	bli_cntx_init_local_if2( cname, imeth, cntx, cntx_p ); \
\
	/* Invoke the operation's front end with the appropriate control
	   tree. */ \
	PASTEMAC(opname,_front) \
	( \
	  side, alpha, a, b, cntx_p, \
	  PASTECH(cname,_cntl) \
	); \
\
	/* Finalize the local context if it was initialized here. */ \
	bli_cntx_finalize_local_if2( cname, imeth, cntx ); \
}

GENFRONT( trmm, gemm, nat )


// -- trsm ---------------------------------------------------------------------

#undef  GENFRONT
#define GENFRONT( opname, cname, imeth ) \
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
	/* Initialize a local context if the one provided is NULL. */ \
	bli_cntx_init_local_if2( cname, imeth, cntx, cntx_p ); \
\
	/* Invoke the operation's front end with the appropriate control
	   tree. */ \
	PASTEMAC(opname,_front) \
	( \
	  side, alpha, a, b, cntx_p, \
	  PASTECH(cname,_l_cntl), \
	  PASTECH(cname,_r_cntl) \
	); \
\
	/* Finalize the local context if it was initialized here. */ \
	bli_cntx_finalize_local_if2( cname, imeth, cntx ); \
}

GENFRONT( trsm, trsm, nat )

