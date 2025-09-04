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

/* NOTE : The order in the macro naming is as follows : THRESH_{API}_{dt}_{codepath}_{march}_{isa}
          The case-sensitivity of {dt} is small, since it is passed from a higher layer(where is it
          used for function-name generation as well). Each of these main macros constitutes to using
          sub-macros, specific to the inputs conditions for path checking. */

/* NOTE : The thresholds for the Tiny-paths have been empirically determined, with datapoints generated
          based on the micro-kernel design. The thresholds are not exhaustive, and are subject to change
          based on the performance validation results. This path provides a lightweight framework as opposed
          to the SUP path, thereby favouring Tiny computation. */

// Thresholds for CGEMM Tiny code-paths
// This is specific to the micro-architecture
// The macros take the input dimensions and the transpose values for the GEMM API
// and define a condition that checks for entry based on these parameters

// Macro for checking if the request is for a single-threaded operation on CGEMM, for the AVX2 ISA
#define IS_TINY_NOT_PARALLEL_c_ZEN5_AVX2( transa, transb, m, n, k, is_parallel ) \
  ( 0 )

// Macro for checking if the request is for a multi-threaded operation on CGEMM, for the AVX2 ISA
#define IS_TINY_PARALLEL_c_ZEN5_AVX2( transa, transb, m, n, k, is_parallel ) \
  ( 0 )

// Macro for checking if the request is for a single-threaded operation on CGEMM, for the AVX512 ISA
#define IS_TINY_NOT_PARALLEL_c_ZEN5_AVX512( transa, transb, m, n, k, is_parallel ) \
  ( ( !is_parallel ) && \
    ( ( ( bli_is_notrans( transa ) && \
      ( ( ( k <= 48 ) && ( m <= 396 ) && ( n <= 396 ) ) || \
        ( ( k <= 288 ) && \
          ( ( ( m <= 360 ) && ( n <= 32 ) ) || \
            ( ( m <= 76 ) && ( n <= 224 ) ) ) ) ) ) ) || \
      ( ( bli_is_trans( transa ) && \
        ( ( ( n <= 48 ) && \
            ( ( ( m <= 368 ) && ( k <= 120 ) ) || \
              ( ( m <= 396 ) && ( k <= 92 ) ) || \
              ( ( m <= 192 ) && ( k <= 396 ) ) ) ) || \
          ( ( n <= 396 ) && \
            ( ( ( m <= 144 ) && ( k <= 48 ) ) || \
              ( ( m <= 72 ) && ( k <= 192 ) ) ) ) ) ) ) ) )

// Macro for checking if the request is for a multi-threaded operation on CGEMM, for the AVX512 ISA
#define IS_TINY_PARALLEL_c_ZEN5_AVX512( transa, transb, m, n, k, is_parallel ) \
  ( ( is_parallel ) && \
    ( ( ( bli_is_notrans( transa ) && \
        ( ( ( n <= 8 ) && \
            ( ( ( m <= 396 ) && ( k <= 32 ) ) || \
              ( ( m <= 48 ) && ( k <= 120 ) ) ) ) || \
          ( ( m <= 96 ) && ( n <= 28 ) && ( k <= 16 ) ) ) ) ) || \
      ( ( bli_is_trans( transa ) && \
        ( ( ( n <= 16 ) && \
            ( ( ( m <= 208 ) && ( k <= 24 ) ) || \
              ( ( m <= 396 ) && ( k <= 8 ) ) ) ) || \
          ( ( m <= 72 ) && ( n <= 8 ) && ( k <= 200 ) ) || \
          ( ( m <= 24 ) && ( n <= 100 ) && ( k <= 16 ) ) ) ) ) ) )

// Main macro to check entry to Tiny-CGEMM-ZEN5-AVX2
#define THRESH_GEMM_c_TINY_ZEN5_AVX2( transa, transb, m, n, k, is_parallel ) \
  /* In case of single-threaded request */ \
  IS_TINY_NOT_PARALLEL_c_ZEN5_AVX2( transa, transb, m, n, k, is_parallel ) || \
  /* In case of multi-threaded request */ \
  IS_TINY_PARALLEL_c_ZEN5_AVX2( transa, transb, m, n, k, is_parallel )

// Main macro to check entry to Tiny-CGEMM-ZEN5-AVX512
#define THRESH_GEMM_c_TINY_ZEN5_AVX512( transa, transb, m, n, k, is_parallel ) \
  /* In case of single-threaded request */ \
  IS_TINY_NOT_PARALLEL_c_ZEN5_AVX512( transa, transb, m, n, k, is_parallel ) || \
  /* In case of multi-threaded request */ \
  IS_TINY_PARALLEL_c_ZEN5_AVX512( transa, transb, m, n, k, is_parallel )

// Thresholds for ZGEMM Tiny code-paths
// This is specific to the micro-architecture
// The macros take the input dimensions and the transpose values for the GEMM API
// and define a condition that checks for entry based on these parameters

// Macro for checking if the request is for a single-threaded operation on ZGEMM, for the AVX2 ISA
#define IS_TINY_NOT_PARALLEL_z_ZEN5_AVX2( transa, transb, m, n, k, is_parallel ) \
  ( ( !is_parallel ) && \
    /* Separate thresholds based on transpose value of A */ \
    ( ( bli_is_notrans( transa ) && ( m < 8 ) && ( n < 200 ) && ( k < 200 ) ) || \
      ( bli_is_trans( transa ) && ( m < 8 ) && ( n < 200 ) && ( k < 200 ) && ( k >= 8 ) ) ) )

// Macro for checking if the request is for a multi-threaded operation on ZGEMM, for the AVX2 ISA
#define IS_TINY_PARALLEL_z_ZEN5_AVX2( transa, transb, m, n, k, is_parallel ) \
  ( ( is_parallel ) && \
    /* Separate thresholds based on transpose value of A */ \
    ( ( bli_is_notrans( transa ) && \
      ( ( m <= 4 ) && ( n <= 200 ) && ( k <= 8 ) ) ) || \
      ( bli_is_trans( transa ) && \
      ( ( m <= 4 ) && ( n >= 12 ) && ( n <= 200 ) && ( k <= 4 ) ) ) ) )

// Macro for checking if the request is for a single-threaded operation on ZGEMM, for the AVX512 ISA
#define IS_TINY_NOT_PARALLEL_z_ZEN5_AVX512( transa, transb, m, n, k, is_parallel ) \
  ( ( !is_parallel ) && \
    /* Separate thresholds based on transpose value of A */ \
    ( ( bli_is_notrans( transa ) && ( m < 200 ) && ( n < 200 ) && ( k < 200 ) ) || \
      ( bli_is_trans( transa ) && ( m < 200 ) && ( n < 200 ) && ( k < 200 ) && ( k >= 8 ) ) ) )

// Macro for checking if the request is for a multi-threaded operation on ZGEMM, for the AVX512 ISA
#define IS_TINY_PARALLEL_z_ZEN5_AVX512( transa, transb, m, n, k, is_parallel ) \
  ( ( is_parallel ) && \
    /* Separate thresholds based on transpose value of A */ \
    ( ( bli_is_notrans( transa ) && \
      ( ( m <= 200 ) && ( k <= 200 ) && ( ( ( n <= 16 ) && ( ( m * k ) <= 16000 ) ) || \
        ( ( n <= 16 ) && ( ( m * k ) <= 13000 ) ) ) ) ) || \
        ( bli_is_trans( transa ) && \
        ( ( m <= 200 ) && ( k <= 200 ) && ( ( ( n <= 16 ) && ( ( m * k ) <= 7000 ) ) || \
          ( ( n <= 16 ) && ( ( m * k ) <= 6000 ) ) ) ) ) ) )

// Main macro to check entry to Tiny-ZGEMM-ZEN5-AVX2
#define THRESH_GEMM_z_TINY_ZEN5_AVX2( transa, transb, m, n, k, is_parallel ) \
  /* In case of single-threaded request */ \
  IS_TINY_NOT_PARALLEL_z_ZEN5_AVX2( transa, transb, m, n, k, is_parallel ) || \
  /* In case of multi-threaded request */ \
  IS_TINY_PARALLEL_z_ZEN5_AVX2( transa, transb, m, n, k, is_parallel )

// Main macro to check entry to Tiny-ZGEMM-ZEN5-AVX512
#define THRESH_GEMM_z_TINY_ZEN5_AVX512( transa, transb, m, n, k, is_parallel ) \
  /* In case of single-threaded request */ \
  IS_TINY_NOT_PARALLEL_z_ZEN5_AVX512( transa, transb, m, n, k, is_parallel ) || \
  /* In case of multi-threaded request */ \
  IS_TINY_PARALLEL_z_ZEN5_AVX512( transa, transb, m, n, k, is_parallel )

/* Defining the macro to be used for selecting the kernel at runtime */
#define ZEN5_UKR_SELECTOR( ch, transa, transb, m, n, k, stor_id, ukr_support, gemmtiny_ukr_info, is_parallel ) \
    if ( PASTECH2( THRESH_GEMM_, ch, _TINY_ZEN5_AVX2 )( transa, transb, m, n, k, is_parallel ) ) \
      LOOKUP_AVX2_UKR( ch, stor_id, ukr_support, gemmtiny_ukr_info ) \
    else if ( PASTECH2( THRESH_GEMM_, ch, _TINY_ZEN5_AVX512 )( transa, transb, m, n, k, is_parallel ) ) \
      LOOKUP_AVX512_UKR( ch, stor_id, ukr_support, gemmtiny_ukr_info ) \
    break;

#endif
