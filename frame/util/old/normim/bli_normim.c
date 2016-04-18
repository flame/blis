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


//
// Define object-based interface.
//
#undef  GENFRONT
#define GENFRONT( opname, varname ) \
\
void PASTEMAC0(opname)( \
                        obj_t* x, \
                        obj_t* norm  \
                      ) \
{ \
    if ( bli_error_checking_is_enabled() ) \
        PASTEMAC(opname,_check)( x, norm ); \
\
    PASTEMAC0(varname)( x, \
                        norm ); \
}

GENFRONT( normim, normim_unb_var1 )


//
// Define BLAS-like interfaces.
//
#undef  GENTFUNCR
#define GENTFUNCR( ctype_x, ctype_xr, chx, chxr, opname, varname ) \
\
void PASTEMAC(chx,opname)( \
                           doff_t    diagoffx, \
                           diag_t    diagx, \
                           uplo_t    uplox, \
                           dim_t     m, \
                           dim_t     n, \
                           ctype_x*  x, inc_t rs_x, inc_t cs_x, \
                           ctype_xr* norm  \
                         ) \
{ \
	PASTEMAC(chx,varname)( diagoffx, \
	                       diagx, \
	                       uplox, \
	                       m, \
	                       n, \
	                       x, rs_x, cs_x, \
	                       norm ); \
}

INSERT_GENTFUNCR_BASIC( normim, normim_unb_var1 )

