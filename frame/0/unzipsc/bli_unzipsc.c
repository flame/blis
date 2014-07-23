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
void bli_unzipsc( obj_t* beta,
                  obj_t* chi_r,
                  obj_t* chi_i )
{
	if ( bli_error_checking_is_enabled() )
		bli_unzipsc_check( beta, chi_r, chi_i );

	bli_unzipsc_unb_var1( beta, chi_r, chi_i );
}


//
// Define BLAS-like interfaces.
//
#undef  GENTFUNCR
#define GENTFUNCR( ctype_b, ctype_br, chb, chbr, opname, varname ) \
\
void PASTEMAC2(chb,chbr,opname)( \
                                 ctype_b*  beta, \
                                 ctype_br* chi_r, \
                                 ctype_br* chi_i  \
                               ) \
{ \
	PASTEMAC(chb,varname)( beta, \
	                       chi_r, \
	                       chi_i ); \
}

INSERT_GENTFUNCR_BASIC( unzipsc, unzipsc_unb_var1 )

