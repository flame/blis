/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.

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

// BLAS Extension APIs
/* ?gemm_pack.h */
/* BLAS interface to perform scaling and packing of the  */
/* matrix  to a packed matrix structure to be used in subsequent calls */
/* Datatype : s & d (single and double precision only supported) */
/* BLAS Extensions */
/* output is a packed buffer */

// Currently we are not adding blis interfaces - these BLAS interfaces will be available by default

#undef  GENTPROTRO
#define GENTPROTRO( ftype, ch, blasname ) \
\
IF_BLIS_ENABLE_BLAS(\
BLIS_EXPORT_BLAS void PASTEF77(ch,blasname) \
     ( \
       const f77_char* identifier, \
       const f77_char* trans, \
       const f77_int*  m, \
       const f77_int*  n, \
       const f77_int*  k, \
       const ftype*    alpha, \
       const ftype*    src, const f77_int*  pld, \
             ftype*    dest \
     ); \
)\
BLIS_EXPORT_BLAS void PASTEF77S(ch,blasname) \
     ( \
       const f77_char* identifier, \
       const f77_char* trans, \
       const f77_int*  m, \
       const f77_int*  n, \
       const f77_int*  k, \
       const ftype*    alpha, \
       const ftype*    src, const f77_int*  pld, \
             ftype*    dest \
     );

INSERT_GENTPROTRO_BLAS( gemm_pack )
