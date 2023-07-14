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

#ifndef BLIS_L4_FPA_H
#define BLIS_L4_FPA_H

//
// Prototype function pointer query interfaces for typed APIs.
//

#undef  GENPROT
#define GENPROT( opname ) \
\
PASTECH2(opname,BLIS_TAPI_EX_SUF,_vft) \
PASTEMAC2(opname,BLIS_TAPI_EX_SUF,_qfp)( num_t dt );

//GENPROT( chol )
//GENPROT( trinv )
//GENPROT( ttmm )
GENPROT( hevd )
GENPROT( rhevd )
GENPROT( hevpinv )


//
// Prototype function pointer query interfaces for implementations.
//

#undef  GENPROT
#define GENPROT( opname, varname ) \
\
PASTECH2(opname,_opt,_vft) \
PASTEMAC(varname,_qfp)( num_t dt );

// chol

GENPROT( chol, chol_l_opt_var1 )
GENPROT( chol, chol_l_opt_var2 )
GENPROT( chol, chol_l_opt_var3 )

GENPROT( chol, chol_u_opt_var1 )
GENPROT( chol, chol_u_opt_var2 )
GENPROT( chol, chol_u_opt_var3 )

// trinv

GENPROT( trinv, trinv_l_opt_var1 )
GENPROT( trinv, trinv_l_opt_var2 )
GENPROT( trinv, trinv_l_opt_var3 )
GENPROT( trinv, trinv_l_opt_var4 )

GENPROT( trinv, trinv_u_opt_var1 )
GENPROT( trinv, trinv_u_opt_var2 )
GENPROT( trinv, trinv_u_opt_var3 )
GENPROT( trinv, trinv_u_opt_var4 )

// ttmm

GENPROT( ttmm, ttmm_l_opt_var1 )
GENPROT( ttmm, ttmm_l_opt_var2 )
GENPROT( ttmm, ttmm_l_opt_var3 )

GENPROT( ttmm, ttmm_u_opt_var1 )
GENPROT( ttmm, ttmm_u_opt_var2 )
GENPROT( ttmm, ttmm_u_opt_var3 )

#endif
