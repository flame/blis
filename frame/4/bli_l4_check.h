/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2022, The University of Texas at Austin
   Copyright (C) 2022, Oracle Labs, Oracle Corporation

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

#ifndef BLIS_L4_CHECK_H
#define BLIS_L4_CHECK_H

//
// Prototype object-based check functions.
//

void bli_chol_check
     (
       const obj_t*  a,
       const cntx_t* cntx
     );

void bli_trinv_check
     (
       const obj_t*  a,
       const cntx_t* cntx
     );

void bli_ttmm_check
     (
       const obj_t*  a,
       const cntx_t* cntx
     );

void bli_hpdinv_check
     (
       const obj_t*  a,
       const cntx_t* cntx
     );

void bli_hevd_check
     (
       const obj_t*  a,
       const obj_t*  v,
       const obj_t*  e,
       const cntx_t* cntx
     );

void bli_rhevd_check
     (
       const obj_t*  v,
       const obj_t*  e,
       const obj_t*  a,
       const cntx_t* cntx
     );

void bli_hevpinv_check
     (
             double  thresh,
       const obj_t*  a,
       const obj_t*  p,
       const cntx_t* cntx
     );

#endif
