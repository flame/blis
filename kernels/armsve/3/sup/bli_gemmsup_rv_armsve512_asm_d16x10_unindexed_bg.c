/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2014, The University of Texas at Austin
   Copyright (C) 2020, Dept. Physics, The University of Tokyo

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
#include "blis.h"

void bli_dgemmsup_rv_armsve512_asm_10x16_unindexed_bg
     (
       conj_t              conja,
       conj_t              conjb,
       dim_t               m0t,
       dim_t               n0t,
       dim_t               k0,
       double*    restrict alpha,
       double*    restrict at, inc_t rs_at0, inc_t cs_at0,
       double*    restrict bt, inc_t rs_bt0, inc_t cs_bt0,
       double*    restrict beta,
       double*    restrict ct, inc_t rs_ct0, inc_t cs_ct0,
       auxinfo_t* restrict datat,
       cntx_t*    restrict cntx
     )
{
  auxinfo_t data;
  bli_auxinfo_set_next_a( bli_auxinfo_next_b( datat ), &data );
  bli_auxinfo_set_next_b( bli_auxinfo_next_a( datat ), &data );
  bli_dgemmsup_cv_armsve512_asm_16x10_unindexed_bg
  (
    conjb, conja,
    n0t, m0t, k0,
    alpha,
    bt, cs_bt0, rs_bt0,
    at, cs_at0, rs_at0,
    beta,
    ct, cs_ct0, rs_ct0,
    &data,
    cntx
  );
}

