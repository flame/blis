/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.

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

// Defining separate static arrays to hold all the kernel info, based on the datatype
static gemmtiny_ukr_info_t cgemmtiny_ukr_avx512[] =
{
    { (void *)bli_cgemmsup_cv_zen4_asm_24x4m, (void *)bli_cpackm_zen4_asm_24xk, FALSE, FALSE, 24, 4 },
    { (void *)bli_cgemmsup_cv_zen4_asm_24x4m, (void *)bli_cpackm_zen4_asm_24xk, FALSE,  TRUE, 24, 4 },
    { (void *)bli_cgemmsup_cv_zen4_asm_24x4m, (void *)bli_cpackm_zen4_asm_24xk, FALSE, FALSE, 24, 4 },
    { (void *)bli_cgemmsup_cv_zen4_asm_24x4m, (void *)bli_cpackm_zen4_asm_24xk, FALSE, FALSE, 24, 4 },
    { (void *)bli_cgemmsup_cv_zen4_asm_24x4m, (void *)bli_cpackm_zen4_asm_24xk, FALSE, FALSE, 24, 4 },
    { (void *)bli_cgemmsup_cv_zen4_asm_24x4m, (void *)bli_cpackm_zen4_asm_24xk, FALSE,  TRUE, 24, 4 },
    { (void *)bli_cgemmsup_cv_zen4_asm_24x4m, (void *)bli_cpackm_zen4_asm_24xk, FALSE, FALSE, 24, 4 },
    { (void *)bli_cgemmsup_cv_zen4_asm_24x4m, (void *)bli_cpackm_zen4_asm_24xk, FALSE, FALSE, 24, 4 }
};

static gemmtiny_ukr_info_t zgemmtiny_ukr_avx512[] =
{
    { (void *)bli_zgemmsup_cv_zen4_asm_12x4m, (void *)bli_zpackm_zen4_asm_12xk, FALSE, FALSE, 12, 4 },
    { (void *)bli_zgemmsup_cd_zen4_asm_12x4m, (void *)bli_zpackm_zen4_asm_12xk, FALSE, FALSE, 12, 4 },
    { (void *)bli_zgemmsup_cv_zen4_asm_12x4m, (void *)bli_zpackm_zen4_asm_12xk, FALSE, FALSE, 12, 4 },
    { (void *)bli_zgemmsup_cv_zen4_asm_12x4m, (void *)bli_zpackm_zen4_asm_12xk, FALSE, FALSE, 12, 4 },
    { (void *)bli_zgemmsup_cv_zen4_asm_12x4m, (void *)bli_zpackm_zen4_asm_12xk, FALSE, FALSE, 12, 4 },
    { (void *)bli_zgemmsup_cd_zen4_asm_12x4m, (void *)bli_zpackm_zen4_asm_12xk, FALSE, FALSE, 12, 4 },
    { (void *)bli_zgemmsup_cv_zen4_asm_12x4m, (void *)bli_zpackm_zen4_asm_12xk, FALSE, FALSE, 12, 4 },
    { (void *)bli_zgemmsup_cv_zen4_asm_12x4m, (void *)bli_zpackm_zen4_asm_12xk, FALSE, FALSE, 12, 4 }
};

// Function macro that defines the bli_?gemmtiny_avx512_ukr_info functions
// These are used to acquire the kernel info at framework level
#undef  GENTFUNC
#define GENTFUNC( ftype, ch, tfuncname ) \
err_t PASTEMAC( ch, tfuncname ) \
      ( \
        stor3_t stor_id, \
        gemmtiny_ukr_info_t *fp_info \
      ) \
{ \
  /* Acquire the object based on stor_id */ \
  *fp_info = TINY_GEMM_AVX512(ch)[stor_id]; \
  /* If the kernel doesn't exist, return the appropriate signal */ \
  if( fp_info->ukr_fp == NULL ) \
  { \
   return BLIS_NOT_YET_IMPLEMENTED; \
  } \
  else if ( ( fp_info->pack_fp == NULL ) && ( fp_info->enable_pack == TRUE ) ) \
  { \
    return BLIS_NOT_YET_IMPLEMENTED; \
  } \
\
  return BLIS_SUCCESS; \
} \

GENTFUNC( scomplex, c, gemmtiny_avx512_ukr_info )
GENTFUNC( dcomplex, z, gemmtiny_avx512_ukr_info )
