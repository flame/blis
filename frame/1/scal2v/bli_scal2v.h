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

#include "bli_scal2v_check.h"
#include "bli_scal2v_kernel.h"
#include "bli_scal2v_ref.h"


//
// Prototype object-based interface.
//
void bli_scal2v( obj_t* beta,
                 obj_t* x,
                 obj_t* y );


//
// Prototype BLAS-like interfaces with homogeneous-typed operands.
//
#undef  GENTPROT
#define GENTPROT( ctype, ch, opname ) \
\
void PASTEMAC(ch,opname)( \
                          conj_t conjx, \
                          dim_t  n, \
                          ctype* beta, \
                          ctype* x, inc_t incx, \
                          ctype* y, inc_t incy \
                        );

INSERT_GENTPROT_BASIC( scal2v )


//
// Prototype BLAS-like interfaces with heterogeneous-typed operands.
//
#undef  GENTPROT3
#define GENTPROT3( ctype_b, ctype_x, ctype_y, chb, chx, chy, opname ) \
\
void PASTEMAC3(chb,chx,chy,opname)( \
                                    conj_t   conjx, \
                                    dim_t    n, \
                                    ctype_b* beta, \
                                    ctype_x* x, inc_t incx, \
                                    ctype_y* y, inc_t incy \
                                  );

INSERT_GENTPROT3_BASIC( scal2v )

#ifdef BLIS_ENABLE_MIXED_DOMAIN_SUPPORT
INSERT_GENTPROT3_MIX_D( scal2v )
#endif

#ifdef BLIS_ENABLE_MIXED_PRECISION_SUPPORT
INSERT_GENTPROT3_MIX_P( scal2v )
#endif

