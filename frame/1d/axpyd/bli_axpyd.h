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

#include "bli_axpyd_check.h"
#include "bli_axpyd_unb_var1.h"


//
// Prototype object-based interface.
//
void bli_axpyd( obj_t* alpha,
                obj_t* x,
                obj_t* y );


//
// Prototype BLAS-like interfaces with homogeneous-typed operands.
//
#undef  GENTPROT
#define GENTPROT( ctype, ch, opname ) \
\
void PASTEMAC(ch,opname)( \
                          doff_t  diagoffx, \
                          diag_t  diagx, \
                          trans_t transx, \
                          dim_t   m, \
                          dim_t   n, \
                          ctype*  alpha, \
                          ctype*  x, inc_t rs_x, inc_t cs_x, \
                          ctype*  y, inc_t rs_y, inc_t cs_y \
                        );

INSERT_GENTPROT_BASIC( axpyd )


//
// Prototype BLAS-like interfaces with heterogeneous-typed operands.
//
#undef  GENTPROT3
#define GENTPROT3( ctype_a, ctype_x, ctype_y, cha, chx, chy, opname ) \
\
void PASTEMAC3(cha,chx,chy,opname)( \
                                    doff_t   diagoffx, \
                                    diag_t   diagx, \
                                    trans_t  transx, \
                                    dim_t    m, \
                                    dim_t    n, \
                                    ctype_a* alpha, \
                                    ctype_x* x, inc_t rs_x, inc_t cs_x, \
                                    ctype_y* y, inc_t rs_y, inc_t cs_y \
                                  );

INSERT_GENTPROT3_BASIC( axpyd )

#ifdef BLIS_ENABLE_MIXED_DOMAIN_SUPPORT
INSERT_GENTPROT3_MIX_D( axpyd )
#endif

#ifdef BLIS_ENABLE_MIXED_PRECISION_SUPPORT
INSERT_GENTPROT3_MIX_P( axpyd )
#endif

