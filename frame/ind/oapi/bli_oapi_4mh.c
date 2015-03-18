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

extern gemm_t* gemm4mh_cntl_rr;
extern gemm_t* gemm4mh_cntl_ii;
extern gemm_t* gemm4mh_cntl_ri;
extern gemm_t* gemm4mh_cntl_ir;


// -- gemm ---------------------------------------------------------------------

#undef  GENFRONT
#define GENFRONT( opname, cname, imeth ) \
\
void PASTEMAC(opname,imeth)( \
                             obj_t*  alpha, \
                             obj_t*  a, \
                             obj_t*  b, \
                             obj_t*  beta, \
                             obj_t*  c \
                           ) \
{ \
	if ( bli_obj_is_real( *c ) ) \
	{ \
		PASTEMAC0(opname)( alpha, a, b, beta, c ); \
	} \
	else \
	{ \
		PASTEMAC(opname,_front)( alpha, a, b, beta,      c, \
		                         PASTECH2(cname,imeth,_cntl_rr) ); \
		PASTEMAC(opname,_front)( alpha, a, b, &BLIS_ONE, c, \
		                         PASTECH2(cname,imeth,_cntl_ii) ); \
		PASTEMAC(opname,_front)( alpha, a, b, &BLIS_ONE, c, \
		                         PASTECH2(cname,imeth,_cntl_ri) ); \
		PASTEMAC(opname,_front)( alpha, a, b, &BLIS_ONE, c, \
		                         PASTECH2(cname,imeth,_cntl_ir) ); \
	} \
}

GENFRONT( gemm, gemm, 4mh )


// -- hemm/symm/trmm3 ----------------------------------------------------------

#undef  GENFRONT
#define GENFRONT( opname, cname, imeth ) \
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
	if ( bli_obj_is_real( *c ) ) \
	{ \
		PASTEMAC0(opname)( side, alpha, a, b, beta, c ); \
	} \
	else \
	{ \
		PASTEMAC(opname,_front)( side, alpha, a, b, beta,      c, \
		                         PASTECH2(cname,imeth,_cntl_rr) ); \
		PASTEMAC(opname,_front)( side, alpha, a, b, &BLIS_ONE, c, \
		                         PASTECH2(cname,imeth,_cntl_ii) ); \
		PASTEMAC(opname,_front)( side, alpha, a, b, &BLIS_ONE, c, \
		                         PASTECH2(cname,imeth,_cntl_ri) ); \
		PASTEMAC(opname,_front)( side, alpha, a, b, &BLIS_ONE, c, \
		                         PASTECH2(cname,imeth,_cntl_ir) ); \
	} \
}

GENFRONT( hemm, gemm, 4mh )
GENFRONT( symm, gemm, 4mh )
GENFRONT( trmm3, gemm, 4mh )


// -- herk/syrk ----------------------------------------------------------------

#undef  GENFRONT
#define GENFRONT( opname, cname, imeth ) \
\
void PASTEMAC(opname,imeth)( \
                             obj_t*  alpha, \
                             obj_t*  a, \
                             obj_t*  beta, \
                             obj_t*  c \
                           ) \
{ \
	if ( bli_obj_is_real( *c ) ) \
	{ \
		PASTEMAC0(opname)( alpha, a, beta, c ); \
	} \
	else \
	{ \
		PASTEMAC(opname,_front)( alpha, a, beta,      c, \
		                         PASTECH2(cname,imeth,_cntl_rr) ); \
		PASTEMAC(opname,_front)( alpha, a, &BLIS_ONE, c, \
		                         PASTECH2(cname,imeth,_cntl_ii) ); \
		PASTEMAC(opname,_front)( alpha, a, &BLIS_ONE, c, \
		                         PASTECH2(cname,imeth,_cntl_ri) ); \
		PASTEMAC(opname,_front)( alpha, a, &BLIS_ONE, c, \
		                         PASTECH2(cname,imeth,_cntl_ir) ); \
	} \
}

GENFRONT( herk, gemm, 4mh )
GENFRONT( syrk, gemm, 4mh )


// -- her2k/syr2k --------------------------------------------------------------

#undef  GENFRONT
#define GENFRONT( opname, cname, imeth ) \
\
void PASTEMAC(opname,imeth)( \
                             obj_t*  alpha, \
                             obj_t*  a, \
                             obj_t*  b, \
                             obj_t*  beta, \
                             obj_t*  c \
                           ) \
{ \
	if ( bli_obj_is_real( *c ) ) \
	{ \
		PASTEMAC0(opname)( alpha, a, b, beta, c ); \
	} \
	else \
	{ \
		PASTEMAC(opname,_front)( alpha, a, b, beta,      c, \
		                         PASTECH2(cname,imeth,_cntl_rr) ); \
		PASTEMAC(opname,_front)( alpha, a, b, &BLIS_ONE, c, \
		                         PASTECH2(cname,imeth,_cntl_ii) ); \
		PASTEMAC(opname,_front)( alpha, a, b, &BLIS_ONE, c, \
		                         PASTECH2(cname,imeth,_cntl_ri) ); \
		PASTEMAC(opname,_front)( alpha, a, b, &BLIS_ONE, c, \
		                         PASTECH2(cname,imeth,_cntl_ir) ); \
	} \
}

GENFRONT( her2k, gemm, 4mh )
GENFRONT( syr2k, gemm, 4mh )


// -- trmm ---------------------------------------------------------------------

//
// 4mh is not applicable to trmm.
//

// -- trsm ---------------------------------------------------------------------

//
// 4mh is not applicable to trsm.
//
