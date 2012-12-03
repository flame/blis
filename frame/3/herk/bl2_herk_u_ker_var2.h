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

#include "blis2.h"

//
// Define macro-kernel blocksizes.
//
// NOTE: These MR and NR values below MUST match the values that packm uses
// when initializing its control tree node.
//

#define bl2_sherk_u_ker_var2_dupb BLIS_DEFAULT_DUPLICATE_B
#define bl2_sherk_u_ker_var2_ndup BLIS_DEFAULT_NUM_DUPL_S
#define bl2_sherk_u_ker_var2_kc   BLIS_DEFAULT_KC_S
#define bl2_sherk_u_ker_var2_mr   BLIS_DEFAULT_MR_S
#define bl2_sherk_u_ker_var2_nr   BLIS_DEFAULT_NR_S

#define bl2_dherk_u_ker_var2_dupb BLIS_DEFAULT_DUPLICATE_B
#define bl2_dherk_u_ker_var2_ndup BLIS_DEFAULT_NUM_DUPL_D
#define bl2_dherk_u_ker_var2_kc   BLIS_DEFAULT_KC_D
#define bl2_dherk_u_ker_var2_mr   BLIS_DEFAULT_MR_D
#define bl2_dherk_u_ker_var2_nr   BLIS_DEFAULT_NR_D

#define bl2_cherk_u_ker_var2_dupb BLIS_DEFAULT_DUPLICATE_B
#define bl2_cherk_u_ker_var2_ndup BLIS_DEFAULT_NUM_DUPL_C
#define bl2_cherk_u_ker_var2_kc   BLIS_DEFAULT_KC_C
#define bl2_cherk_u_ker_var2_mr   BLIS_DEFAULT_MR_C
#define bl2_cherk_u_ker_var2_nr   BLIS_DEFAULT_NR_C

#define bl2_zherk_u_ker_var2_dupb BLIS_DEFAULT_DUPLICATE_B
#define bl2_zherk_u_ker_var2_ndup BLIS_DEFAULT_NUM_DUPL_Z
#define bl2_zherk_u_ker_var2_kc   BLIS_DEFAULT_KC_Z
#define bl2_zherk_u_ker_var2_mr   BLIS_DEFAULT_MR_Z
#define bl2_zherk_u_ker_var2_nr   BLIS_DEFAULT_NR_Z


//
// Prototype object-based interface.
//
void bl2_herk_u_ker_var2( obj_t*  alpha,
                          obj_t*  a,
                          obj_t*  b,
                          obj_t*  beta,
                          obj_t*  c,
                          herk_t* cntl );


//
// Prototype BLAS-like interfaces.
//
#undef  GENTPROT
#define GENTPROT( ctype, ch, varname ) \
\
void PASTEMAC(ch,varname)( \
                           doff_t  diagoffc, \
                           dim_t   m, \
                           dim_t   n, \
                           dim_t   k, \
                           void*   alpha, \
                           void*   a, inc_t rs_a, inc_t cs_a, inc_t ps_a, \
                           void*   b, inc_t rs_b, inc_t cs_b, inc_t ps_b, \
                           void*   beta, \
                           void*   c, inc_t rs_c, inc_t cs_c \
                         );

INSERT_GENTPROT_BASIC( herk_u_ker_var2 )

