/*

   BLIS    
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2013, The University of Texas

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
// Default register blocksizes and micro-kernel shapes
//
// NOTE: These MR and NR values below MUST match the values that packm uses
// when initializing its control tree node.
//
#include "bl2_gemm_4x2.h"
#include "bl2_dupl_kx2.h"

#define bl2_sher2k_u_ker_var3_ndup BLIS_DEFAULT_NUM_DUPL_S
#define bl2_sher2k_u_ker_var3_kc   BLIS_DEFAULT_KC_S
#define bl2_sher2k_u_ker_var3_mr   BLIS_DEFAULT_MR_S
#define bl2_sher2k_u_ker_var3_nr   BLIS_DEFAULT_NR_S
#define bl2_sher2k_u_ker_var3_ukr  bl2_sgemm_4x2
#define bl2_sher2k_u_ker_var3_dupl bl2_sdupl_kx2

#define bl2_dher2k_u_ker_var3_ndup BLIS_DEFAULT_NUM_DUPL_D
#define bl2_dher2k_u_ker_var3_kc   BLIS_DEFAULT_KC_D
#define bl2_dher2k_u_ker_var3_mr   BLIS_DEFAULT_MR_D
#define bl2_dher2k_u_ker_var3_nr   BLIS_DEFAULT_NR_D
#define bl2_dher2k_u_ker_var3_ukr  bl2_dgemm_4x2
#define bl2_dher2k_u_ker_var3_dupl bl2_ddupl_kx2

#define bl2_cher2k_u_ker_var3_ndup BLIS_DEFAULT_NUM_DUPL_C
#define bl2_cher2k_u_ker_var3_kc   BLIS_DEFAULT_KC_C
#define bl2_cher2k_u_ker_var3_mr   BLIS_DEFAULT_MR_C
#define bl2_cher2k_u_ker_var3_nr   BLIS_DEFAULT_NR_C
#define bl2_cher2k_u_ker_var3_ukr  bl2_cgemm_4x2
#define bl2_cher2k_u_ker_var3_dupl bl2_cdupl_kx2

#define bl2_zher2k_u_ker_var3_ndup BLIS_DEFAULT_NUM_DUPL_Z
#define bl2_zher2k_u_ker_var3_kc   BLIS_DEFAULT_KC_Z
#define bl2_zher2k_u_ker_var3_mr   BLIS_DEFAULT_MR_Z
#define bl2_zher2k_u_ker_var3_nr   BLIS_DEFAULT_NR_Z
#define bl2_zher2k_u_ker_var3_ukr  bl2_zgemm_4x2
#define bl2_zher2k_u_ker_var3_dupl bl2_zdupl_kx2



void bl2_her2k_u_ker_var3( obj_t*  alpha,
                           obj_t*  a,
                           obj_t*  bh,
                           obj_t*  alpha_conj,
                           obj_t*  b,
                           obj_t*  ah,
                           obj_t*  beta,
                           obj_t*  c,
                           her2k_t* cntl );


#undef  GENTPROT
#define GENTPROT( ctype, ch, varname ) \
\
void PASTEMAC(ch,varname)( \
                           doff_t  diagoffc, \
                           uplo_t  uploc, \
                           dim_t   m, \
                           dim_t   n, \
                           dim_t   k, \
                           void*   a,  inc_t ps_a, \
                           void*   bh, inc_t ps_bh, \
                           void*   b,  inc_t ps_b, \
                           void*   ah, inc_t ps_ah, \
                           void*   c,  inc_t rs_c, inc_t cs_c \
                         );

INSERT_GENTPROT_BASIC( her2k_u_ker_var3 )

