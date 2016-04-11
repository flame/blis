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

// Guard the function definitions so that they are only compiled when
// #included from files that define the object API macros.
#ifdef BLIS_ENABLE_OAPI

//
// Define object-based interfaces.
//

#undef  GENFRONT
#define GENFRONT( opname ) \
\
void PASTEMAC(opname,EX_SUF) \
     ( \
       obj_t*  alpha, \
       obj_t*  a, \
       obj_t*  b, \
       obj_t*  beta, \
       obj_t*  c  \
       BLIS_OAPI_CNTX_PARAM  \
     ) \
{ \
	BLIS_OAPI_CNTX_DECL \
\
	/* Invoke the operation's "ind" function--its induced method front-end.
	   This function will call native execution for real domain problems.
	   For complex problems, it calls the highest priority induced method
	   that is available (ie: implemented and enabled), and if none are
	   enabled, it calls native execution. */ \
	PASTEMAC(opname,ind) \
	( \
	  alpha, \
	  a, \
	  b, \
	  beta, \
	  c, \
	  cntx  \
	); \
}

GENFRONT( gemm )
GENFRONT( her2k )
GENFRONT( syr2k )


#undef  GENFRONT
#define GENFRONT( opname ) \
\
void PASTEMAC(opname,EX_SUF) \
     ( \
       side_t  side, \
       obj_t*  alpha, \
       obj_t*  a, \
       obj_t*  b, \
       obj_t*  beta, \
       obj_t*  c  \
       BLIS_OAPI_CNTX_PARAM  \
     ) \
{ \
	BLIS_OAPI_CNTX_DECL \
\
	PASTEMAC(opname,ind) \
	( \
	  side, \
	  alpha, \
	  a, \
	  b, \
	  beta, \
	  c, \
	  cntx  \
	); \
}

GENFRONT( hemm )
GENFRONT( symm )
GENFRONT( trmm3 )


#undef  GENFRONT
#define GENFRONT( opname ) \
\
void PASTEMAC(opname,EX_SUF) \
     ( \
       obj_t*  alpha, \
       obj_t*  a, \
       obj_t*  beta, \
       obj_t*  c  \
       BLIS_OAPI_CNTX_PARAM  \
     ) \
{ \
	BLIS_OAPI_CNTX_DECL \
\
	PASTEMAC(opname,ind) \
	( \
	  alpha, \
	  a, \
	  beta, \
	  c, \
	  cntx  \
	); \
}

GENFRONT( herk )
GENFRONT( syrk )


#undef  GENFRONT
#define GENFRONT( opname ) \
\
void PASTEMAC(opname,EX_SUF) \
     ( \
       side_t  side, \
       obj_t*  alpha, \
       obj_t*  a, \
       obj_t*  b  \
       BLIS_OAPI_CNTX_PARAM  \
     ) \
{ \
	BLIS_OAPI_CNTX_DECL \
\
	PASTEMAC(opname,ind) \
	( \
	  side, \
	  alpha, \
	  a, \
	  b, \
	  cntx  \
	); \
}

GENFRONT( trmm )
GENFRONT( trsm )


#endif

