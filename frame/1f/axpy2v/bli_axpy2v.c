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
                        obj_t* alpha1, \
                        obj_t* alpha2, \
                        obj_t* x, \
                        obj_t* y, \
                        obj_t* z  \
                      ) \
{ \
	if ( bli_error_checking_is_enabled() ) \
		PASTEMAC(opname,_check)( alpha1, alpha2, x, y, z ); \
\
	PASTEMAC0(varname)( alpha1, \
	                    alpha2, \
	                    x, \
	                    y, \
	                    z ); \
}

GENFRONT( axpy2v, axpy2v_kernel )


//
// Define BLAS-like interfaces with homogeneous-typed operands.
//
#undef  GENTFUNC
#define GENTFUNC( ctype, ch, opname, varname ) \
\
void PASTEMAC(ch,opname)( \
                          conj_t conjx, \
                          conj_t conjy, \
                          dim_t  n, \
                          ctype* alpha1, \
                          ctype* alpha2, \
                          ctype* x, inc_t incx, \
                          ctype* y, inc_t incy, \
                          ctype* z, inc_t incz \
                        ) \
{ \
	PASTEMAC3(ch,ch,ch,varname)( conjx, \
	                             conjy, \
	                             n, \
	                             alpha1, \
	                             alpha2, \
	                             x, incx, \
	                             y, incy, \
	                             z, incz ); \
}

INSERT_GENTFUNC_BASIC( axpy2v, AXPY2V_KERNEL )


//
// Define BLAS-like interfaces with heterogeneous-typed operands.
//
#undef  GENTFUNC3U12
#define GENTFUNC3U12( ctype_x, ctype_y, ctype_z, ctype_xy, chx, chy, chz, chxy, opname, varname ) \
\
void PASTEMAC3(chx,chy,chz,opname)( \
                                    conj_t    conjx, \
                                    conj_t    conjy, \
                                    dim_t     n, \
                                    ctype_xy* alpha1, \
                                    ctype_xy* alpha2, \
                                    ctype_x*  x, inc_t incx, \
                                    ctype_y*  y, inc_t incy, \
                                    ctype_z*  z, inc_t incz \
                                  ) \
{ \
	PASTEMAC3(chx,chy,chz,varname)( conjx, \
	                                conjy, \
	                                n, \
	                                alpha1, \
	                                alpha2, \
	                                x, incx, \
	                                y, incy, \
	                                z, incz ); \
}

// Define the basic set of functions unconditionally, and then also some
// mixed datatype functions if requested.
INSERT_GENTFUNC3U12_BASIC( axpy2v, AXPY2V_KERNEL )

#ifdef BLIS_ENABLE_MIXED_DOMAIN_SUPPORT
INSERT_GENTFUNC3U12_MIX_D( axpy2v, AXPY2V_KERNEL )
#endif

#ifdef BLIS_ENABLE_MIXED_PRECISION_SUPPORT
INSERT_GENTFUNC3U12_MIX_P( axpy2v, AXPY2V_KERNEL )
#endif

