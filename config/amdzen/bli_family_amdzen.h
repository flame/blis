/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2021-2023, Advanced Micro Devices, Inc. All rights reserved.

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

#ifndef BLIS_FAMILY_AMDZEN_H
#define BLIS_FAMILY_AMDZEN_H

// By default, it is effective to parallelize the outer loops.
// Setting these macros to 1 will force JR and IR inner loops
// to be not paralleized.
//
#define BLIS_THREAD_MAX_IR      1
#define BLIS_THREAD_MAX_JR      1


#define BLIS_ENABLE_SMALL_MATRIX
#define BLIS_ENABLE_SMALL_MATRIX_TRSM


// This will select the threshold below which small matrix code will be called.
#define BLIS_SMALL_MATRIX_THRES        700
#define BLIS_SMALL_M_RECT_MATRIX_THRES 160
#define BLIS_SMALL_K_RECT_MATRIX_THRES 128

#define BLIS_SMALL_MATRIX_A_THRES_M_SYRK    96
#define BLIS_SMALL_MATRIX_A_THRES_N_SYRK    128

// When running HPL with pure MPI without DGEMM threading (Single-threaded
// BLIS), defining this macro as 1 yields better performance.
#define AOCL_BLIS_MULTIINSTANCE   0

/*
 * Override the block sizes in the context to the block sizes used
 * by AVX2 GEMM+TRSM kernels, this is needed in Zen4 context as default
 * GEMM kernels are AVX512 based and uses different block sizes.
 * 
 * This function should be called in TRSM path before performing
 * any packing operations. 
 * 
 * Also the context must be restored to default values by calling 
 * bli_zen4_restore_default_blkszs() before exiting TRSM Path
 */
BLIS_EXPORT_BLIS void bli_zen4_override_trsm_blkszs (cntx_t* cntx);

BLIS_EXPORT_BLIS void bli_zen4_override_gemmt_blkszs (cntx_t* cntx);

/*
 * Restore the block sizes to default values needed for zen4 context.
 *
 * This function should be called to restore the block sizes to there
 * default values if they where overriden by calling
 * bli_zen4_override_trsm_blkszs() to enable AVX2 GEMM kernels in the 
 * TRSM path.
 * 
 */
BLIS_EXPORT_BLIS void bli_zen4_restore_default_blkszs (cntx_t* cntx);

#endif

