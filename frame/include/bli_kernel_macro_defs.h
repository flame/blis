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

#ifndef BLIS_KERNEL_MACRO_DEFS_H
#define BLIS_KERNEL_MACRO_DEFS_H


// Redefine kernel blocksizes, defined in bli_kernel.h, to shorter
// names that can be derived via PASTEMAC macro.

// Cache blocksizes

#define bli_smc  BLIS_DEFAULT_MC_S 
#define bli_snc  BLIS_DEFAULT_NC_S
#define bli_skc  BLIS_DEFAULT_KC_S

#define bli_dmc  BLIS_DEFAULT_MC_D 
#define bli_dnc  BLIS_DEFAULT_NC_D
#define bli_dkc  BLIS_DEFAULT_KC_D

#define bli_cmc  BLIS_DEFAULT_MC_C 
#define bli_cnc  BLIS_DEFAULT_NC_C
#define bli_ckc  BLIS_DEFAULT_KC_C

#define bli_zmc  BLIS_DEFAULT_MC_Z 
#define bli_znc  BLIS_DEFAULT_NC_Z
#define bli_zkc  BLIS_DEFAULT_KC_Z

// Register blocksizes

#define bli_smr  BLIS_DEFAULT_MR_S 
#define bli_snr  BLIS_DEFAULT_NR_S
#define bli_skr  BLIS_DEFAULT_KR_S

#define bli_dmr  BLIS_DEFAULT_MR_D 
#define bli_dnr  BLIS_DEFAULT_NR_D
#define bli_dkr  BLIS_DEFAULT_KR_D

#define bli_cmr  BLIS_DEFAULT_MR_C 
#define bli_cnr  BLIS_DEFAULT_NR_C
#define bli_ckr  BLIS_DEFAULT_KR_C

#define bli_zmr  BLIS_DEFAULT_MR_Z 
#define bli_znr  BLIS_DEFAULT_NR_Z
#define bli_zkr  BLIS_DEFAULT_KR_Z

// Duplication

#define bli_sndup  BLIS_DEFAULT_NUM_DUPL_S
#define bli_dndup  BLIS_DEFAULT_NUM_DUPL_D
#define bli_cndup  BLIS_DEFAULT_NUM_DUPL_C
#define bli_zndup  BLIS_DEFAULT_NUM_DUPL_Z


#endif 
