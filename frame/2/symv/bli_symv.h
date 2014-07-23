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

#include "bli_symv_check.h"


void bli_symv( obj_t*  alpha,
               obj_t*  a,
               obj_t*  x,
               obj_t*  beta,
               obj_t*  y );


//
// Prototype BLAS-like interfaces with homogeneous-typed operands.
//
#undef  GENTPROT
#define GENTPROT( ctype, ch, opname ) \
\
void PASTEMAC(ch,opname)( \
                          uplo_t  uploa, \
                          conj_t  conja, \
                          conj_t  conjx, \
                          dim_t   m, \
                          ctype*  alpha, \
                          ctype*  a, inc_t rs_a, inc_t cs_a, \
                          ctype*  x, inc_t incx, \
                          ctype*  beta, \
                          ctype*  y, inc_t incy  \
                        );

INSERT_GENTPROT_BASIC( symv )


//
// Prototype BLAS-like interfaces with heterogeneous-typed operands.
//
#undef  GENTPROT3U12
#define GENTPROT3U12( ctype_a, ctype_x, ctype_y, ctype_ax, cha, chx, chy, chax, opname ) \
\
void PASTEMAC3(cha,chx,chy,opname)( \
                                    uplo_t    uploa, \
                                    conj_t    conja, \
                                    conj_t    conjx, \
                                    dim_t     m, \
                                    ctype_ax* alpha, \
                                    ctype_a*  a, inc_t rs_a, inc_t cs_a, \
                                    ctype_x*  x, inc_t incx, \
                                    ctype_y*  beta, \
                                    ctype_y*  y, inc_t incy  \
                                  );

INSERT_GENTPROT3U12_BASIC( symv )

#ifdef BLIS_ENABLE_MIXED_DOMAIN_SUPPORT
INSERT_GENTPROT3U12_MIX_D( symv )
#endif

#ifdef BLIS_ENABLE_MIXED_PRECISION_SUPPORT
INSERT_GENTPROT3U12_MIX_P( symv )
#endif

