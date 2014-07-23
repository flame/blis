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
                        obj_t* y  \
                      ) \
{ \
    if ( bli_error_checking_is_enabled() ) \
        PASTEMAC(opname,_check)( x, y ); \
\
    PASTEMAC0(varname)( x, \
                        y ); \
}

GENFRONT( subv, subv_kernel )


//
// Define BLAS-like interfaces with homogeneous-typed operands.
//
#undef  GENTFUNC
#define GENTFUNC( ctype, ch, opname, varname ) \
\
void PASTEMAC(ch,opname)( \
                          conj_t conjx, \
                          dim_t  n, \
                          ctype* x, inc_t incx, \
                          ctype* y, inc_t incy \
                        ) \
{ \
	PASTEMAC2(ch,ch,varname)( conjx, \
	                          n, \
	                          x, incx, \
	                          y, incy ); \
}

INSERT_GENTFUNC_BASIC( subv, SUBV_KERNEL )


//
// Define BLAS-like interfaces with heterogeneous-typed operands.
//
#undef  GENTFUNC2
#define GENTFUNC2( ctype_x, ctype_y, chx, chy, opname, varname ) \
\
void PASTEMAC2(chx,chy,opname)( \
                                conj_t   conjx, \
                                dim_t    n, \
                                ctype_x* x, inc_t incx, \
                                ctype_y* y, inc_t incy \
                              ) \
{ \
	PASTEMAC2(chx,chy,varname)( conjx, \
	                            n, \
	                            x, incx, \
	                            y, incy ); \
}

INSERT_GENTFUNC2_BASIC( subv, SUBV_KERNEL )

#ifdef BLIS_ENABLE_MIXED_DOMAIN_SUPPORT
INSERT_GENTFUNC2_MIX_D( subv, SUBV_KERNEL )
#endif

#ifdef BLIS_ENABLE_MIXED_PRECISION_SUPPORT
INSERT_GENTFUNC2_MIX_P( subv, SUBV_KERNEL )
#endif

