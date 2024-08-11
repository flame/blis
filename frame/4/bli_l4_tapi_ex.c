/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2022, The University of Texas at Austin

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

#include "blis.h"

//
// Define BLAS-like interfaces with typed operands (expert).
//

#undef  GENTFUNC
#define GENTFUNC( ctype, ch, opname, struca ) \
\
void PASTEMAC(ch,opname,BLIS_OAPI_EX_SUF) \
     ( \
             uplo_t  uploa, \
             dim_t   m, \
             ctype*  a, inc_t rs_a, inc_t cs_a, \
       const cntx_t* cntx, \
             rntm_t* rntm  \
     ) \
{ \
	bli_init_once(); \
\
	const num_t dt = PASTEMAC(ch,type); \
\
	obj_t       ao = BLIS_OBJECT_INITIALIZER; \
\
	bli_obj_init_finish( dt, m, m, a, rs_a, cs_a, &ao ); \
\
	bli_obj_set_uplo( uploa, &ao ); \
\
	bli_obj_set_struc( struca, &ao ); \
\
	PASTEMAC(opname,BLIS_OAPI_EX_SUF) \
	( \
	  &ao, \
	  cntx, \
	  rntm  \
	); \
}

INSERT_GENTFUNC_BASIC( chol,   BLIS_HERMITIAN  )
INSERT_GENTFUNC_BASIC( ttmm,   BLIS_TRIANGULAR )
INSERT_GENTFUNC_BASIC( hpdinv, BLIS_HERMITIAN  )


#undef  GENTFUNC
#define GENTFUNC( ctype, ch, opname, struca ) \
\
void PASTEMAC(ch,opname,BLIS_OAPI_EX_SUF) \
     ( \
             uplo_t  uploa, \
             diag_t  diaga, \
             dim_t   m, \
             ctype*  a, inc_t rs_a, inc_t cs_a, \
       const cntx_t* cntx, \
             rntm_t* rntm  \
     ) \
{ \
	bli_init_once(); \
\
	const num_t dt = PASTEMAC(ch,type); \
\
	obj_t       ao = BLIS_OBJECT_INITIALIZER; \
\
	bli_obj_init_finish( dt, m, m, a, rs_a, cs_a, &ao ); \
\
	bli_obj_set_uplo( uploa, &ao ); \
	bli_obj_set_diag( diaga, &ao ); \
\
	bli_obj_set_struc( struca, &ao ); \
\
	PASTEMAC(opname,BLIS_OAPI_EX_SUF) \
	( \
	  &ao, \
	  cntx, \
	  rntm  \
	); \
}

INSERT_GENTFUNC_BASIC( trinv, BLIS_TRIANGULAR )

