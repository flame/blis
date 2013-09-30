/*

   BLIS    
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2013, The University of Texas

   Redistribution and use in source and binary forms, with or without
   modification, are permitted provided that the following conditions are
   met:
    - Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.
    - Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.
    - Neither the name of The University of Texas nor the names of its
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


#undef  GENTFUNC3U12
#define GENTFUNC3U12( ctype_x, ctype_y, ctype_z, ctype_xy, chx, chy, chz, chxy, varname, dotxvker, axpyvker ) \
\
void PASTEMAC3(chx,chy,chz,varname)( \
                                     conj_t conjxt, \
                                     conj_t conjx, \
                                     conj_t conjy, \
                                     dim_t  m, \
                                     void*  alpha, \
                                     void*  x, inc_t incx, \
                                     void*  y, inc_t incy, \
                                     void*  rho, \
                                     void*  z, inc_t incz \
                                   ) \
{ \
	ctype_xy* one        = PASTEMAC(chxy,1); \
	ctype_xy* zero       = PASTEMAC(chxy,0); \
	ctype_x*  alpha_cast = alpha; \
	ctype_x*  x_cast     = x; \
	ctype_y*  y_cast     = y; \
	ctype_xy* rho_cast   = rho; \
	ctype_z*  z_cast     = z; \
\
	PASTEMAC3(chx,chy,chxy,dotxvker)( conjxt, \
	                                  conjy, \
	                                  m, \
	                                  one, \
	                                  x_cast, incx, \
	                                  y_cast, incy, \
	                                  zero, \
	                                  rho_cast ); \
	PASTEMAC3(chx,chx,chz,axpyvker)( conjx, \
	                                 m, \
	                                 alpha_cast, \
	                                 x_cast, incx, \
	                                 z_cast, incz ); \
}

// Define the basic set of functions unconditionally, and then also some
// mixed datatype functions if requested.
INSERT_GENTFUNC3U12_BASIC2( dotaxpyv_unb_var1, DOTXV_KERNEL, AXPYV_KERNEL )

#ifdef BLIS_ENABLE_MIXED_DOMAIN_SUPPORT
INSERT_GENTFUNC3U12_MIX_D2( dotaxpyv_unb_var1, DOTXV_KERNEL, AXPYV_KERNEL )
#endif

#ifdef BLIS_ENABLE_MIXED_PRECISION_SUPPORT
INSERT_GENTFUNC3U12_MIX_P2( dotaxpyv_unb_var1, DOTXV_KERNEL, AXPYV_KERNEL )
#endif

