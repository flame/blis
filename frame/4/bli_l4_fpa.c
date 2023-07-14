/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2022, The University of Texas at Austin
   Copyright (C) 2022, Oracle Labs, Oracle Corporation

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

#ifdef BLIS_ENABLE_LEVEL4

//
// Define function pointer query interfaces for typed APIs.
//

#undef  GENFRONT
#define GENFRONT( opname ) \
\
GENARRAY_FPA( PASTECH2(opname,BLIS_TAPI_EX_SUF,_vft), \
              PASTECH(opname,BLIS_TAPI_EX_SUF) ); \
\
PASTECH2(opname,BLIS_TAPI_EX_SUF,_vft) \
PASTEMAC2(opname,BLIS_TAPI_EX_SUF,_qfp)( num_t dt ) \
{ \
    return PASTECH2(opname,BLIS_TAPI_EX_SUF,_fpa)[ dt ]; \
}


//GENFRONT( chol )
//GENFRONT( trinv )
//GENFRONT( ttmm )
GENFRONT( hevd )
GENFRONT( rhevd )
GENFRONT( hevpinv )


//
// Define function pointer query interfaces for implementations.
//

#undef  GENFRONT
#define GENFRONT( opname, varname ) \
\
GENARRAY_FPA( PASTECH2(opname,_opt,_vft), \
              varname ); \
\
PASTECH2(opname,_opt,_vft) \
PASTEMAC(varname,_qfp)( num_t dt ) \
{ \
    return PASTECH(varname,_fpa)[ dt ]; \
}

// chol

GENFRONT( chol, chol_l_opt_var1 )
GENFRONT( chol, chol_l_opt_var2 )
GENFRONT( chol, chol_l_opt_var3 )

GENFRONT( chol, chol_u_opt_var1 )
GENFRONT( chol, chol_u_opt_var2 )
GENFRONT( chol, chol_u_opt_var3 )

// trinv

GENFRONT( trinv, trinv_l_opt_var1 )
GENFRONT( trinv, trinv_l_opt_var2 )
GENFRONT( trinv, trinv_l_opt_var3 )
GENFRONT( trinv, trinv_l_opt_var4 )

GENFRONT( trinv, trinv_u_opt_var1 )
GENFRONT( trinv, trinv_u_opt_var2 )
GENFRONT( trinv, trinv_u_opt_var3 )
GENFRONT( trinv, trinv_u_opt_var4 )

// ttmm

GENFRONT( ttmm, ttmm_l_opt_var1 )
GENFRONT( ttmm, ttmm_l_opt_var2 )
GENFRONT( ttmm, ttmm_l_opt_var3 )

GENFRONT( ttmm, ttmm_u_opt_var1 )
GENFRONT( ttmm, ttmm_u_opt_var2 )
GENFRONT( ttmm, ttmm_u_opt_var3 )


#endif
