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

#ifndef BLIS_CONFIG_ZEN_H
#define BLIS_CONFIG_ZEN_H

/* Thresholds for ZGEMM Tiny code-paths.
   This is based on the micro-architecture.
   For now, we are not defining the threshold for ZEN based architectures.
   Thus, it would take the subsequent paths(small, sup, native) */
#define zgemm_tiny_zen_thresh_avx2( transa, transb, m, n, k, is_parallel ) \
  ( 0 ) \

/* Defining the macro to be used for selecting the kernel at runtime */
#define ZEN_UKR_SELECTOR( ch, transa, transb, m, n, k, stor_id, ukr_support, gemmtiny_ukr_info, is_parallel ) \
    if ( PASTECH3( ch, gemm_tiny, _zen_thresh, _avx2 )( transa, transb, m, n, k, is_parallel ) ) \
      LOOKUP_AVX2_UKR( ch, stor_id, ukr_support, gemmtiny_ukr_info ) \
    break;

#endif
