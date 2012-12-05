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

#include "bl2_dotxaxpyf_unb_var1.h"


//
// Define fusing factors (if they are not already defined by the user
// in bl2_kernel.h).
//
#ifndef bl2_sdotxaxpyf_fuse_fac
#define bl2_sdotxaxpyf_fuse_fac BLIS_DEFAULT_FUSING_FACTOR_S
#endif
#ifndef bl2_ddotxaxpyf_fuse_fac
#define bl2_ddotxaxpyf_fuse_fac BLIS_DEFAULT_FUSING_FACTOR_D
#endif
#ifndef bl2_cdotxaxpyf_fuse_fac
#define bl2_cdotxaxpyf_fuse_fac BLIS_DEFAULT_FUSING_FACTOR_C
#endif
#ifndef bl2_zdotxaxpyf_fuse_fac
#define bl2_zdotxaxpyf_fuse_fac BLIS_DEFAULT_FUSING_FACTOR_Z
#endif


//
// Prototype BLAS-like interfaces with homogeneous-typed operands.
//
#undef  GENTPROT
#define GENTPROT( ctype, ch, opname ) \
\
void PASTEMAC(ch,opname)( \
                          conj_t conjat, \
                          conj_t conja, \
                          conj_t conjw, \
                          conj_t conjx, \
                          dim_t  m, \
                          dim_t  n, \
                          ctype* alpha, \
                          ctype* a, inc_t inca, inc_t lda, \
                          ctype* w, inc_t incw, \
                          ctype* x, inc_t incx, \
                          ctype* beta, \
                          ctype* y, inc_t incy, \
                          ctype* z, inc_t incz \
                        );

INSERT_GENTPROT_BASIC( dotxaxpyf )


//
// Prototype BLAS-like interfaces with heterogeneous-typed operands.
//
#undef  GENTPROT3U12
#define GENTPROT3U12( ctype_a, ctype_b, ctype_c, ctype_ab, cha, chb, chc, chab, opname ) \
\
void PASTEMAC3(cha,chb,chc,opname)( \
                                    conj_t    conjat, \
                                    conj_t    conja, \
                                    conj_t    conjw, \
                                    conj_t    conjx, \
                                    dim_t     m, \
                                    dim_t     n, \
                                    ctype_ab* alpha, \
                                    ctype_a*  a, inc_t inca, inc_t lda, \
                                    ctype_b*  w, inc_t incw, \
                                    ctype_b*  x, inc_t incx, \
                                    ctype_c*  beta, \
                                    ctype_c*  y, inc_t incy, \
                                    ctype_c*  z, inc_t incz \
                                  );


INSERT_GENTPROT3U12_BASIC( dotxaxpyf )

#ifdef BLIS_ENABLE_MIXED_DOMAIN_SUPPORT
INSERT_GENTPROT3U12_MIX_D( dotxaxpyf )
#endif

#ifdef BLIS_ENABLE_MIXED_PRECISION_SUPPORT
INSERT_GENTPROT3U12_MIX_P( dotxaxpyf )
#endif

