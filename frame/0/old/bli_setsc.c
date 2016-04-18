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

typedef void (*FUNCPTR_T)(
                           double* zeta_r,
                           double* zeta_i,
                           void*   chi 
                         );

static FUNCPTR_T GENARRAY(ftypes,setsc);

//
// Define object-based interfaces.
//

#undef  GENFRONT
#define GENFRONT( opname ) \
\
void PASTEMAC0(opname)( \
                        double* zeta_r, \
                        double* zeta_i, \
                        obj_t*  chi  \
                      ) \
{ \
	num_t     dt_chi    = bli_obj_datatype( *chi ); \
\
	void*     buf_chi   = bli_obj_buffer_at_off( *chi ); \
\
	FUNCPTR_T f; \
\
	if ( bli_error_checking_is_enabled() ) \
	    PASTEMAC(opname,_check)( zeta_r, zeta_i, chi ); \
\
	/* Index into the type combination array to extract the correct
	   function pointer. */ \
	f = ftypes[dt_chi]; \
\
	/* Invoke the function. */ \
	f( \
	   zeta_r, \
	   zeta_i, \
	   buf_chi  \
	 ); \
}

GENFRONT( setsc )


//
// Define BLAS-like interfaces with typed operands.
//

#undef  GENTFUNC
#define GENTFUNC( ctype, ch, opname ) \
\
void PASTEMAC(ch,opname)( \
                          double* zeta_r, \
                          double* zeta_i  \
                          void*   chi, \
                        ) \
{ \
	ctype* chi_cast = chi; \
\
	PASTEMAC2(d,ch,sets)( *zeta_r, *zeta_i, *chi_cast ); \
}

INSERT_GENTFUNC_BASIC( setsc )

