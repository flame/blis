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

#ifndef BLIS_CONFIG_ZEN4_H
#define BLIS_CONFIG_ZEN4_H

/* NOTE : The order in the main macro naming is as follows : THRESH_{API}_{dt}_{codepath}_{march}_{isa}
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
// We support only when transa is 'N'
#define IS_TINY_NOT_PARALLEL_c_ZEN4_AVX2( transa, transb, m, n, k, is_parallel ) \
  ( ( !is_parallel ) && \
    ( bli_is_notrans( transa ) && ( m <= 72 ) && ( n <= 96 ) && ( k < 12 ) ) )

// Macro for checking if the request is for a multi-threaded operation on CGEMM, for the AVX2 ISA
// We support only when transa is 'N'
#define IS_TINY_PARALLEL_c_ZEN4_AVX2( transa, transb, m, n, k, is_parallel ) \
  ( ( is_parallel ) && \
    ( bli_is_notrans( transa ) && ( m <= 96 ) && ( n <= 96 ) && ( k <= 96 ) && \
      ( ( ( m * k ) <= 144 ) || ( ( n * k ) <= 144 ) || ( ( m * n ) <= 144 ) ) && \
      ( ( m * n * k ) <= 7200 ) ) )

// Macro for checking if the request is for a single-threaded operation on CGEMM, for the AVX512 ISA
#define IS_TINY_NOT_PARALLEL_c_ZEN4_AVX512( transa, transb, m, n, k, is_parallel ) \
  ( ( !is_parallel ) && \
    ( ( bli_is_notrans( transa ) && \
      ( ( m <= 396 ) && \
        ( ( ( n <= 312 ) && ( k <= 24 ) ) || \
          ( ( n <= 396 ) && ( k <= 12 ) ) || \
          ( ( n <= 136 ) && ( k <= 32 ) ) || \
          ( ( m <= 52 ) && ( n <= 396 ) && ( k <= 396 ) ) || \
          ( ( n <= 8 ) && ( k <= 396 ) ) || \
          ( ( n <= 16 ) && ( k <= 148 ) ) || \
          ( ( n <= 48 ) && ( k <= 68 ) ) ) ) ) || \
      ( bli_is_trans( transa ) && \
      ( ( ( m <= 396 ) && \
          ( ( ( n <= 256 ) && ( k <= 32 ) ) || \
            ( ( n <= 396 ) && ( k <= 16 ) ) ) ) || \
        ( ( m <= 72 ) && ( n <= 396 ) && ( k <= 32 ) ) || \
        ( ( n <= 32 ) && \
          ( ( ( m <= 192 ) && ( k <= 396 ) ) || ( ( m <= 396 ) && ( k <= 96 ) ) ) ) ) ) ) )

// Macro for checking if the request is for a multi-threaded operation on CGEMM, for the AVX512 ISA
#define IS_TINY_PARALLEL_c_ZEN4_AVX512( transa, transb, m, n, k, is_parallel ) \
  ( ( is_parallel ) && \
    ( ( bli_is_notrans( transa ) && \
      ( ( ( n <= 12 ) && \
          ( ( ( m <= 36 ) && ( k <= 396 ) ) || \
            ( ( m <= 48 ) && ( k <= 232 ) ) || \
            ( ( m <= 288 ) && ( n <= 8 ) && ( k <= 44 ) ) ) ) || \
        ( ( n <= 20 ) && \
          ( ( ( m <= 144 ) && ( k <= 24 ) ) || \
            ( ( m <= 28 ) && ( k <= 396 ) ) ) ) ) ) || \
      ( bli_is_trans( transa ) && \
      ( ( ( n <= 16 ) && \
          ( ( ( m <= 192 ) && ( k <= 32 ) ) || \
            ( ( m <= 72 ) && ( k <= 144 ) ) ) ) || \
        ( ( m <= 288 ) && ( n <= 40 ) && ( k <= 24 ) ) ) ) ) )

// Main macro to check entry to Tiny-CGEMM-ZEN4-AVX2
#define THRESH_GEMM_c_TINY_ZEN4_AVX2( transa, transb, m, n, k, is_parallel ) \
  /* In case of single-threaded request */ \
  IS_TINY_NOT_PARALLEL_c_ZEN4_AVX2( transa, transb, m, n, k, is_parallel ) || \
  /* In case of multi-threaded request */ \
  IS_TINY_PARALLEL_c_ZEN4_AVX2( transa, transb, m, n, k, is_parallel )

// Main macro to check entry to Tiny-CGEMM-ZEN4-AVX512
#define THRESH_GEMM_c_TINY_ZEN4_AVX512( transa, transb, m, n, k, is_parallel ) \
  /* In case of single-threaded request */ \
  IS_TINY_NOT_PARALLEL_c_ZEN4_AVX512( transa, transb, m, n, k, is_parallel ) || \
  /* In case of multi-threaded request */ \
  IS_TINY_PARALLEL_c_ZEN4_AVX512( transa, transb, m, n, k, is_parallel )

// Thresholds for ZGEMM Tiny code-paths
// This is specific to the micro-architecture
// The macros take the input dimensions and the transpose values for the GEMM API
// and define a condition that checks for entry based on these parameters

// Macro for checking if the request is for a single-threaded operation on ZGEMM, for the AVX2 ISA
#define IS_TINY_NOT_PARALLEL_z_ZEN4_AVX2( transa, transb, m, n, k, is_parallel ) \
   ( ( !is_parallel ) && \
    /* Separate thresholds based on transpose value of A */                              \
                                                                                         \
    /* An additional check (m % 2 == 0) is included in both branches. */                 \
	/* Now, only inputs with an even m dimension (divisible by 2) which are below the */ \
	/* threshold of avx2 kernel are routed to the AVX2 kernel */                         \
	/* Odd m values are routed to the AVX512 kernel. */                                  \
	/* Because avx2 kernels invokes gemv calls for m_left=1 */                           \
	/* (odd m dimension of matrix) */                                                    \
    /* The gemv function call adds overhead for very small sizes and results */          \
    /* in suboptimal performance. */                                                     \
    ( ( bli_is_notrans( transa ) && ( m < 60 ) && ( n >= 4 ) && ( n < 200 ) && ( k < 68 ) && (m % 2 == 0) ) || \
      ( bli_is_trans( transa ) && ( m < 200 ) && ( n < 200 ) && ( k < 200 ) && ( k >= 16 ) && (m % 2 == 0) ) ) )

// Macro for checking if the request is for a multi-threaded operation on ZGEMM, for the AVX2 ISA
#define IS_TINY_PARALLEL_z_ZEN4_AVX2( transa, transb, m, n, k, is_parallel ) \
  ( ( is_parallel ) && \
    /* Separate thresholds based on transpose value of A */ \
    ( ( bli_is_notrans( transa ) && \
      ( ( ( m <= 6 ) && ( n <= 80 ) && ( k <= 64 ) ) || \
        ( ( m <= 4 ) && ( n <= 200 ) && ( k <= 16 ) ) ) ) || \
      ( bli_is_trans( transa ) && \
      ( ( ( m <= 6 ) && ( n <= 40 ) && ( k <= 72 ) ) || ( ( m <= 12 ) && ( n <= 24 ) && ( k <= 44 ) ) ) ) ) )

// Macro for checking if the request is for a single-threaded operation on ZGEMM, for the AVX512 ISA
#define IS_TINY_NOT_PARALLEL_z_ZEN4_AVX512( transa, transb, m, n, k, is_parallel ) \
  ( ( !is_parallel ) && \
    /* Separate thresholds based on transpose value of A */                                               \
                                                                                                          \
    /* Threshold change to route all of the inputs to avx512 tiny path. */                                \
    /* Eliminating dependency of avx2 zgemm_small path if A, B matrix storage is 'N'(not transpose) or */ \
    /* 'T'(transpose). */                                                                                 \
                                                                                                          \
    /* Dependency on AVX2 zgemm_small path is eliminated for non-transpose and transpose storage. */      \
    /* For conjugate transpose cases, falling back to AVX2 is still allowed. */                           \
    ( ( bli_is_notrans( transa ) && ( m < 200 ) && ( n < 200 ) && ( k < 200 ) ) || \
      ( bli_is_trans( transa ) && ( m < 200 ) && ( n < 200 ) && ( k < 200 ) && ( k >= 8 ) ) ) )

// Macro for checking if the request is for a multi-threaded operation on ZGEMM, for the AVX512 ISA
#define IS_TINY_PARALLEL_z_ZEN4_AVX512( transa, transb, m, n, k, is_parallel ) \
  ( ( is_parallel ) && \
    /* A set of unified thresholds for all the cases */ \
    ( ( ( ( m <= 16 ) && ( n <= 16 ) && ( k <= 24 ) ) ) || \
    /* Separate thresholds based on transpose value of A */ \
      ( ( bli_is_notrans( transa ) && \
        ( ( ( n <= 16 ) && ( ( ( m <= 32 ) && ( k <= 80 ) ) || ( ( m <= 80 ) && ( k <= 20 ) ) ) ) || \
          ( ( k <= 8 ) && ( m <= 40 ) && ( n <= 40 ) ) ) ) || \
        ( bli_is_trans( transa ) && \
        ( ( ( n <= 16 ) && ( ( ( m <= 40 ) && ( k <= 40 ) ) || ( ( m <= 96 ) && ( k <= 12 ) ) || ( ( m <= 16 ) && ( k <= 96 ) ) ) ) || \
          ( ( k <= 8 ) && ( m <= 40 ) && ( n <= 40 ) ) ) ) ) ) )

// Main macro to check entry to Tiny-ZGEMM-ZEN4-AVX2
#define THRESH_GEMM_z_TINY_ZEN4_AVX2( transa, transb, m, n, k, is_parallel ) \
  /* In case of single-threaded request */ \
  IS_TINY_NOT_PARALLEL_z_ZEN4_AVX2( transa, transb, m, n, k, is_parallel ) || \
  /* In case of multi-threaded request */ \
  IS_TINY_PARALLEL_z_ZEN4_AVX2( transa, transb, m, n, k, is_parallel )

// Main macro to check entry to Tiny-ZGEMM-ZEN4-AVX512
#define THRESH_GEMM_z_TINY_ZEN4_AVX512( transa, transb, m, n, k, is_parallel ) \
  /* In case of single-threaded request */ \
  IS_TINY_NOT_PARALLEL_z_ZEN4_AVX512( transa, transb, m, n, k, is_parallel ) || \
  /* In case of multi-threaded request */ \
  IS_TINY_PARALLEL_z_ZEN4_AVX512( transa, transb, m, n, k, is_parallel )

/* Defining the macro to be used for selecting the kernel at runtime */
#define ZEN4_UKR_SELECTOR( ch, transa, transb, m, n, k, stor_id, ukr_support, gemmtiny_ukr_info, is_parallel ) \
    if ( PASTECH2( THRESH_GEMM_, ch, _TINY_ZEN4_AVX2 )( transa, transb, m, n, k, is_parallel ) ) \
      LOOKUP_AVX2_UKR( ch, stor_id, ukr_support, gemmtiny_ukr_info ) \
    else if ( PASTECH2( THRESH_GEMM_, ch, _TINY_ZEN4_AVX512 )( transa, transb, m, n, k, is_parallel ) ) \
      LOOKUP_AVX512_UKR( ch, stor_id, ukr_support, gemmtiny_ukr_info ) \
    break;

#endif
