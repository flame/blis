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

#include "blis.h"

//
// Define function pointer query interfaces.
//

#undef  GENFRONT
#define GENFRONT( opname ) \
\
GENARRAY_FPA( PASTECH(opname,BLIS_TAPI_EX_SUF,_vft), \
              PASTECH(opname,BLIS_TAPI_EX_SUF) ); \
\
PASTECH(opname,BLIS_TAPI_EX_SUF,_vft) \
PASTEMAC(opname,BLIS_TAPI_EX_SUF,_qfp)( num_t dt ) \
{ \
	return PASTECH(opname,BLIS_TAPI_EX_SUF,_fpa)[ dt ]; \
}

GENFRONT( gemv )
GENFRONT( ger )
GENFRONT( hemv )
GENFRONT( symv )
GENFRONT( shmv )
GENFRONT( skmv )
GENFRONT( her )
GENFRONT( syr )
GENFRONT( her2 )
GENFRONT( syr2 )
GENFRONT( shr2 )
GENFRONT( skr2 )
GENFRONT( trmv )
GENFRONT( trsv )

//
// Define function pointer query interfaces for level-2 implementations.
//

#undef  GENFRONT
#define GENFRONT( opname, varname ) \
\
GENARRAY_FPA( PASTECH(opname,_unb,_vft), \
              varname ); \
\
PASTECH(opname,_unb,_vft) \
PASTEMAC(varname,_qfp)( num_t dt ) \
{ \
	return PASTECH(varname,_fpa)[ dt ]; \
}

GENFRONT( gemv, gemv_unb_var1 )
GENFRONT( gemv, gemv_unb_var2 )
GENFRONT( gemv, gemv_unf_var1 )
GENFRONT( gemv, gemv_unf_var2 )

GENFRONT( ger, ger_unb_var1 )
GENFRONT( ger, ger_unb_var2 )

GENFRONT( hemv, hemv_unb_var1 )
GENFRONT( hemv, hemv_unb_var2 )
GENFRONT( hemv, hemv_unb_var3 )
GENFRONT( hemv, hemv_unb_var4 )
GENFRONT( hemv, hemv_unf_var1 )
GENFRONT( hemv, hemv_unf_var3 )
GENFRONT( hemv, hemv_unf_var1a )
GENFRONT( hemv, hemv_unf_var3a )

GENFRONT( her, her_unb_var1 )
GENFRONT( her, her_unb_var2 )

GENFRONT( her2, her2_unb_var1 )
GENFRONT( her2, her2_unb_var2 )
GENFRONT( her2, her2_unb_var3 )
GENFRONT( her2, her2_unb_var4 )
GENFRONT( her2, her2_unf_var1 )
GENFRONT( her2, her2_unf_var4 )

GENFRONT( trmv, trmv_unb_var1 )
GENFRONT( trmv, trmv_unb_var2 )
GENFRONT( trmv, trmv_unf_var1 )
GENFRONT( trmv, trmv_unf_var2 )

GENFRONT( trsv, trsv_unb_var1 )
GENFRONT( trsv, trsv_unb_var2 )
GENFRONT( trsv, trsv_unf_var1 )
GENFRONT( trsv, trsv_unf_var2 )

