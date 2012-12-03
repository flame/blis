/*

   BLIS    
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2012, The University of Texas

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

#include "blis2.h"


#undef  GENTFUNC3U12
#define GENTFUNC3U12( ctype_a, ctype_b, ctype_c, ctype_ab, cha, chb, chc, chab, opname, varname ) \
\
void PASTEMAC3(cha,chb,chc,varname)( \
                                     conj_t conjat, \
                                     conj_t conja, \
                                     conj_t conjw, \
                                     conj_t conjx, \
                                     dim_t  m, \
                                     dim_t  b_n, \
                                     void*  alpha, \
                                     void*  a, inc_t inca, inc_t lda, \
                                     void*  w, inc_t incw, \
                                     void*  x, inc_t incx, \
                                     void*  beta, \
                                     void*  y, inc_t incy, \
                                     void*  z, inc_t incz \
                                   ) \
{ \
	ctype_ab* alpha_cast = alpha; \
	ctype_a*  a_cast     = a; \
	ctype_b*  w_cast     = w; \
	ctype_b*  x_cast     = x; \
	ctype_c*  beta_cast  = beta; \
	ctype_c*  y_cast     = y; \
	ctype_c*  z_cast     = z; \
	ctype_a*  a1; \
	ctype_b*  chi1; \
	ctype_b*  w1; \
	ctype_c*  psi1; \
	ctype_c*  z1; \
	ctype_b   conjx_chi1; \
	ctype_ab  alpha_chi1; \
	dim_t     i; \
\
	/* A is m x n.                   */ \
	/* y = beta * y + alpha * A^T w; */ \
	/* z =        z + alpha * A   x; */ \
	for ( i = 0; i < b_n; ++i ) \
	{ \
		a1   = a_cast + (0  )*inca + (i  )*lda; \
		w1   = w_cast + (0  )*incw; \
		psi1 = y_cast + (i  )*incy; \
\
		PASTEMAC3(cha,chb,chc,dotxv)( conjat, \
		                              conjw, \
	    	                          m, \
	        	                      alpha_cast, \
	            	                  a1, inca, \
	                	              w1, incw, \
	                    	          beta_cast, \
	                        	      psi1 ); \
	} \
\
	for ( i = 0; i < b_n; ++i ) \
	{ \
		a1   = a_cast + (0  )*inca + (i  )*lda; \
		chi1 = x_cast + (i  )*incx; \
		z1   = z_cast + (0  )*incz; \
\
		PASTEMAC2(chb,chb,copycjs)( conjx, *chi1, conjx_chi1 ); \
		PASTEMAC3(chab,chb,chab,scal2s)( *alpha_cast, conjx_chi1, alpha_chi1 ); \
		PASTEMAC3(chab,cha,chc,axpyv)( conja, \
	    	                           m, \
	            	                   &alpha_chi1, \
	                	               a1, inca, \
	                    	           z1, incz ); \
	} \
}

// Define the basic set of functions unconditionally, and then also some
// mixed datatype functions if requested.
INSERT_GENTFUNC3U12_BASIC( dotxaxpyf, dotxaxpyf_unb_var1 )

#ifdef BLIS_ENABLE_MIXED_DOMAIN_SUPPORT
INSERT_GENTFUNC3U12_MIX_D( dotxaxpyf, dotxaxpyf_unb_var1 )
#endif

#ifdef BLIS_ENABLE_MIXED_PRECISION_SUPPORT
INSERT_GENTFUNC3U12_MIX_P( dotxaxpyf, dotxaxpyf_unb_var1 )
#endif

