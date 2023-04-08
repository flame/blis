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

// -- Level-3 native micro-kernel prototype redefinitions ----------------------

#ifdef BLIS_ENABLE_GEMM_MD

#if 0
#undef  GENTFUNCCO
#define GENTFUNCCO( ctype, ctype_r, ch, chr, opname, suf ) \
\
void PASTEMAC2(ch,opname,suf) \
     ( \
             dim_t      m, \
             dim_t      n, \
             dim_t      k, \
       const void*      alpha, \
       const void*      a, \
       const void*      b, \
       const void*      beta, \
             void*      c, inc_t rs_c, inc_t cs_c, \
             auxinfo_t* data, \
       const cntx_t*    cntx  \
     )
#endif

#undef  GENTPROTCO
#define GENTPROTCO( ctype, ctype_r, ch, chr, funcname, opname ) \
\
void PASTEMAC(ch,funcname) \
     ( \
       PASTECH(opname,_params), \
       BLIS_AUXINFO_PARAM, \
       BLIS_CNTX_PARAM  \
     );

INSERT_GENTPROTCO_BASIC( gemm_md_c2r_ref, gemm )


#endif
