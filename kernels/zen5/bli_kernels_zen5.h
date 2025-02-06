/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2024 - 2025, Advanced Micro Devices, Inc. All rights reserved.

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

// Dgemm sup RV kernels
GEMMSUP_KER_PROT( double,  d, gemmsup_rv_zen5_asm_24x8m)
GEMMSUP_KER_PROT( double,  d, gemmsup_rv_zen5_asm_24x7m)
GEMMSUP_KER_PROT( double,  d, gemmsup_rv_zen5_asm_24x6m)
GEMMSUP_KER_PROT( double,  d, gemmsup_rv_zen5_asm_24x5m)
GEMMSUP_KER_PROT( double,  d, gemmsup_rv_zen5_asm_24x4m)
GEMMSUP_KER_PROT( double,  d, gemmsup_rv_zen5_asm_24x3m)
GEMMSUP_KER_PROT( double,  d, gemmsup_rv_zen5_asm_24x2m)
GEMMSUP_KER_PROT( double,  d, gemmsup_rv_zen5_asm_24x1m)

// threshold functions
bool bli_cntx_gemmsup_thresh_is_met_zen5
(
    obj_t*  a,
    obj_t*  b,
    obj_t*  c,
    cntx_t* cntx
);

// dynamic blocksizes function
void bli_dynamic_blkszs_zen5
    (
      dim_t n_threads,
      cntx_t* cntx,
      num_t dt
    );

err_t bli_trsm_small_ZEN5
      (
        side_t side,
        obj_t  *alpha,
        obj_t  *a,
        obj_t  *b,
        cntx_t *cntx,
        cntl_t *cntl,
        bool   is_parallel
      );

TRSMSMALL_KER_PROT( d, trsm_small_XAltB_XAuB_ZEN5 )
TRSMSMALL_KER_PROT( d, trsm_small_XAutB_XAlB_ZEN5 )
TRSMSMALL_KER_PROT( d, trsm_small_AltXB_AuXB_ZEN5 )
TRSMSMALL_KER_PROT( d, trsm_small_AutXB_AlXB_ZEN5 )

TRSMSMALL_KER_PROT( z, trsm_small_XAltB_XAuB_ZEN5 )
TRSMSMALL_KER_PROT( z, trsm_small_XAutB_XAlB_ZEN5 )
TRSMSMALL_KER_PROT( z, trsm_small_AltXB_AuXB_ZEN5 )
TRSMSMALL_KER_PROT( z, trsm_small_AutXB_AlXB_ZEN5 )

#ifdef BLIS_ENABLE_OPENMP
err_t bli_trsm_small_mt_ZEN5
      (
        side_t side,
        obj_t  *alpha,
        obj_t  *a,
        obj_t  *b,
        cntx_t *cntx,
        cntl_t *cntl,
        bool   is_parallel
      );
#endif
