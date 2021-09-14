/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2014, The University of Texas at Austin
   Copyright (C) 2020, Advanced Micro Devices, Inc.

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
// Prototype object-based interfaces.
//

BLIS_EXPORT_BLIS void bli_gemm_ex
     (
       obj_t*  alpha,
       obj_t*  a,
       obj_t*  b,
       obj_t*  beta,
       obj_t*  c,
       cntx_t* cntx,
       rntm_t* rntm
     );

BLIS_EXPORT_BLIS void bli_gemm
     (
       obj_t*  alpha,
       obj_t*  a,
       obj_t*  b,
       obj_t*  beta,
       obj_t*  c
     );

BLIS_EXPORT_BLIS void bli_gemmt_ex
     (
       obj_t*  alpha,
       obj_t*  a,
       obj_t*  b,
       obj_t*  beta,
       obj_t*  c,
       cntx_t* cntx,
       rntm_t* rntm
     );

BLIS_EXPORT_BLIS void bli_gemmt
     (
       obj_t*  alpha,
       obj_t*  a,
       obj_t*  b,
       obj_t*  beta,
       obj_t*  c
     );

#undef GENTDEF
#define GENTDEF(opname,ind) \
BLIS_EXPORT_BLIS void PASTEMAC(opname,ind) \
     ( \
       obj_t*  alpha, \
       obj_t*  a, \
       obj_t*  b, \
       obj_t*  beta, \
       obj_t*  c, \
       cntx_t* cntx, \
       rntm_t* rntm \
     );

GENTDEF(her2k,_ex);
GENTDEF(her2k,3mh);
GENTDEF(her2k,3m1);
GENTDEF(her2k,4mh);
GENTDEF(her2k,4m1);
GENTDEF(her2k,1m);
GENTDEF(her2k,nat);
GENTDEF(her2k,ind);

GENTDEF(syr2k,_ex);
GENTDEF(syr2k,3mh);
GENTDEF(syr2k,3m1);
GENTDEF(syr2k,4mh);
GENTDEF(syr2k,4m1);
GENTDEF(syr2k,1m);
GENTDEF(syr2k,nat);
GENTDEF(syr2k,ind);

BLIS_EXPORT_BLIS void bli_her2k
     (
       obj_t*  alpha,
       obj_t*  a,
       obj_t*  b,
       obj_t*  beta,
       obj_t*  c
     );

BLIS_EXPORT_BLIS void bli_syr2k
     (
       obj_t*  alpha,
       obj_t*  a,
       obj_t*  b,
       obj_t*  beta,
       obj_t*  c
     );

BLIS_EXPORT_BLIS void bli_hemm_ex
     (
       side_t  side,
       obj_t*  alpha,
       obj_t*  a,
       obj_t*  b,
       obj_t*  beta,
       obj_t*  c,
       cntx_t* cntx,
       rntm_t* rntm
     );
BLIS_EXPORT_BLIS void bli_hemm
     (
       side_t  side,
       obj_t*  alpha,
       obj_t*  a,
       obj_t*  b,
       obj_t*  beta,
       obj_t*  c
     );

BLIS_EXPORT_BLIS void bli_symm_ex
     (
       side_t  side,
       obj_t*  alpha,
       obj_t*  a,
       obj_t*  b,
       obj_t*  beta,
       obj_t*  c,
       cntx_t* cntx,
       rntm_t* rntm
     );
BLIS_EXPORT_BLIS void bli_symm
     (
       side_t  side,
       obj_t*  alpha,
       obj_t*  a,
       obj_t*  b,
       obj_t*  beta,
       obj_t*  c
     );

BLIS_EXPORT_BLIS void bli_trmm3_ex
     (
       side_t  side,
       obj_t*  alpha,
       obj_t*  a,
       obj_t*  b,
       obj_t*  beta,
       obj_t*  c,
       cntx_t* cntx,
       rntm_t* rntm
     );
BLIS_EXPORT_BLIS void bli_trmm3
     (
       side_t  side,
       obj_t*  alpha,
       obj_t*  a,
       obj_t*  b,
       obj_t*  beta,
       obj_t*  c
     );

#undef GENTDEF
#define GENTDEF(opname,ind) \
BLIS_EXPORT_BLIS void PASTEMAC(opname,ind) \
     ( \
       obj_t*  alpha, \
       obj_t*  a, \
       obj_t*  beta, \
       obj_t*  c, \
       cntx_t* cntx, \
       rntm_t* rntm \
     );

GENTDEF(herk,_ex);
GENTDEF(herk,3mh);
GENTDEF(herk,3m1);
GENTDEF(herk,4mh);
GENTDEF(herk,4m1);
GENTDEF(herk,1m);
GENTDEF(herk,nat);
GENTDEF(herk,ind);

GENTDEF(syrk,_ex);
GENTDEF(syrk,3mh);
GENTDEF(syrk,3m1);
GENTDEF(syrk,4mh);
GENTDEF(syrk,4m1);
GENTDEF(syrk,1m);
GENTDEF(syrk,nat);
GENTDEF(syrk,ind);

BLIS_EXPORT_BLIS void bli_herk
     (
       obj_t*  alpha,
       obj_t*  a,
       obj_t*  beta,
       obj_t*  c
     );

BLIS_EXPORT_BLIS void bli_syrk
     (
       obj_t*  alpha,
       obj_t*  a,
       obj_t*  beta,
       obj_t*  c
     );

BLIS_EXPORT_BLIS void bli_trmm_ex
     (
       side_t  side,
       obj_t*  alpha,
       obj_t*  a,
       obj_t*  b,
       cntx_t* cntx,
       rntm_t* rntm
     );

BLIS_EXPORT_BLIS void bli_trmm
     (
       side_t  side,
       obj_t*  alpha,
       obj_t*  a,
       obj_t*  b
     );

BLIS_EXPORT_BLIS void bli_trsm_ex
     (
       side_t  side,
       obj_t*  alpha,
       obj_t*  a,
       obj_t*  b,
       cntx_t* cntx,
       rntm_t* rntm
     );

BLIS_EXPORT_BLIS void bli_trsm
     (
       side_t  side,
       obj_t*  alpha,
       obj_t*  a,
       obj_t*  b
     );
