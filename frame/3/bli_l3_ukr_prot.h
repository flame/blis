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

//
// Define template prototypes for level-3 micro-kernels.
//

#define GEMM_UKR_PROT( ctype, ch, opname ) GEMM_UKR_PROT2(ctype, ctype, ch, opname)

#define GEMM_UKR_PROT2( ctype_in, ctype_out, ch, opname ) \
\
void PASTEMAC(ch,opname) \
     ( \
       dim_t               m, \
       dim_t               n, \
       dim_t               k, \
       ctype_out* restrict alpha, \
       ctype_in*  restrict a, \
       ctype_in*  restrict b, \
       ctype_out* restrict beta, \
       ctype_out* restrict c, inc_t rs_c, inc_t cs_c, \
       auxinfo_t* restrict data, \
       cntx_t*    restrict cntx  \
     );


#define GEMMTRSM_UKR_PROT( ctype, ch, opname ) \
\
void PASTEMAC(ch,opname) \
     ( \
       dim_t               m, \
       dim_t               n, \
       dim_t               k, \
       ctype*     restrict alpha, \
       ctype*     restrict a1x, \
       ctype*     restrict a11, \
       ctype*     restrict bx1, \
       ctype*     restrict b11, \
       ctype*     restrict c11, inc_t rs_c, inc_t cs_c, \
       auxinfo_t* restrict data, \
       cntx_t*    restrict cntx  \
     );


#define TRSM_UKR_PROT( ctype, ch, opname ) \
\
void PASTEMAC(ch,opname) \
     ( \
       ctype*     restrict a, \
       ctype*     restrict b, \
       ctype*     restrict c, inc_t rs_c, inc_t cs_c, \
       auxinfo_t* restrict data, \
       cntx_t*    restrict cntx  \
     );

