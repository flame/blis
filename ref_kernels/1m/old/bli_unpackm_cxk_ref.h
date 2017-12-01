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

#undef  GENTPROT
#define GENTPROT( ctype, ch, opname, arch, suf ) \
\
void PASTEMAC3(ch,opname,arch,suf) \
     ( \
       conj_t           conjp, \
       dim_t            n, \
       void*   restrict kappa, \
       void*   restrict p,             inc_t ldp, \
       void*   restrict a, inc_t inca, inc_t lda, \
       cntx_t* restrict cntx  \
     );

INSERT_GENTPROT_BASIC2( unpackm_2xk_ref, BLIS_CNAME_INFIX, BLIS_REF_SUFFIX )
INSERT_GENTPROT_BASIC2( unpackm_4xk_ref, BLIS_CNAME_INFIX, BLIS_REF_SUFFIX )
INSERT_GENTPROT_BASIC2( unpackm_6xk_ref, BLIS_CNAME_INFIX, BLIS_REF_SUFFIX )
INSERT_GENTPROT_BASIC2( unpackm_8xk_ref, BLIS_CNAME_INFIX, BLIS_REF_SUFFIX )
INSERT_GENTPROT_BASIC2( unpackm_10xk_ref, BLIS_CNAME_INFIX, BLIS_REF_SUFFIX )
INSERT_GENTPROT_BASIC2( unpackm_12xk_ref, BLIS_CNAME_INFIX, BLIS_REF_SUFFIX )
INSERT_GENTPROT_BASIC2( unpackm_14xk_ref, BLIS_CNAME_INFIX, BLIS_REF_SUFFIX )
INSERT_GENTPROT_BASIC2( unpackm_16xk_ref, BLIS_CNAME_INFIX, BLIS_REF_SUFFIX )

