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

#ifndef BLIS_CONFIG_ZEN5_H
#define BLIS_CONFIG_ZEN5_H

// Thresholds for ZGEMM Tiny code-paths
// This is specific to the micro-architecture
// The macros take the input dimensions and the transpose values for the GEMM API
// and define a condition that checks for entry based on these parameters
#define zgemm_tiny_zen5_thresh_avx2( transa, transb, m, n, k, is_parallel ) \
  /* In case of single-threaded request */ \
  ( ( !is_parallel ) && \
    /* Separate thresholds based on transpose value of A */ \
    ( ( bli_is_notrans( transa ) && ( m < 8 ) && ( n < 200 ) && ( k < 200 ) ) || \
      ( bli_is_trans( transa ) && ( m < 8 ) && ( n < 200 ) && ( k < 200 ) && ( k >= 8 ) ) ) ) || \
  /* In case of multi-threaded request */ \
  ( ( is_parallel ) && ( ( m * n * k ) < 5000 ) && ( k >= 16 ) )

#define zgemm_tiny_zen5_thresh_avx512( transa, transb, m, n, k, is_parallel ) \
  /* In case of single-threaded request */ \
  ( ( !is_parallel ) && \
    /* Separate thresholds based on transpose value of A */ \
    ( ( bli_is_notrans( transa ) && ( m < 200 ) && ( n < 200 ) && ( k < 200 ) ) || \
      ( bli_is_trans( transa ) && ( m < 200 ) && ( n < 200 ) && ( k < 200 ) && ( k >= 8 ) ) ) ) || \
  /* In case of multi-threaded request */ \
  ( ( is_parallel ) && ( ( m * n * k ) < 10000 ) && ( k >= 16 ) )

/* Defining the macro to be used for selecting the kernel at runtime */
#define ZEN5_UKR_SELECTOR( ch, transa, transb, m, n, k, stor_id, ukr_support, gemmtiny_ukr_info, is_parallel ) \
    if ( PASTECH3( ch, gemm_tiny, _zen5_thresh, _avx2 )( transa, transb, m, n, k, is_parallel ) ) \
      LOOKUP_AVX2_UKR( ch, stor_id, ukr_support, gemmtiny_ukr_info ) \
    else if ( PASTECH3( ch, gemm_tiny, _zen5_thresh, _avx512 )( transa, transb, m, n, k, is_parallel ) ) \
      LOOKUP_AVX512_UKR( ch, stor_id, ukr_support, gemmtiny_ukr_info ) \
    break;

#endif
