/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2024, Advanced Micro Devices, Inc. All rights reserved.

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

err_t bli_dgemm_tiny
(
        trans_t transa,
        trans_t transb,
        dim_t  m,
        dim_t  n,
        dim_t  k,
        const double*    alpha,
        const double*    a, const inc_t rs_a0, const inc_t cs_a0,
        const double*    b, const inc_t rs_b0, const inc_t cs_b0,
        const double*    beta,
        double*    c, const inc_t rs_c0, const inc_t cs_c0
)
{
    // Query the architecture ID
    arch_t id = bli_arch_query_id();

    if(FALSE == bli_thread_get_is_parallel())
    {
        // Pick the kernel based on the architecture ID
        switch (id)
        {
          case BLIS_ARCH_ZEN5:
          case BLIS_ARCH_ZEN4:
#if defined(BLIS_FAMILY_ZEN5) || defined(BLIS_FAMILY_ZEN4) || defined(BLIS_FAMILY_AMDZEN) || defined(BLIS_FAMILY_X86_64)
              if(((m == n) && (m < 400) && (k < 1000)) ||
              ( (m != n) && (( ((m + n -k) < 1500) &&
              ((m + k-n) < 1500) && ((n + k-m) < 1500) ) ||
              ((n <= 100) && (k <=100)))))
              {
                  return bli_dgemm_tiny_24x8
                          (
                              1 * (transa == BLIS_CONJ_NO_TRANSPOSE),
                              1 * (transb == BLIS_CONJ_NO_TRANSPOSE),
                              transa,
                              transb,
                              m,
                              n,
                              k,
                              alpha,
                              a, rs_a0, cs_a0,
                              b, rs_b0, cs_b0,
                              beta,
                              c, rs_c0, cs_c0
                          );
              }
#endif
              break;

          case BLIS_ARCH_ZEN:
          case BLIS_ARCH_ZEN2:
          case BLIS_ARCH_ZEN3:
              if( ( (m <= 8)  || ( (m <= 1000) && (n <= 24) && (k >= 4) ) ) && (k <= 1500) )
              {
                  return bli_dgemm_tiny_6x8
                          (
                              1 * (transa == BLIS_CONJ_NO_TRANSPOSE),
                              1 * (transb == BLIS_CONJ_NO_TRANSPOSE),
                              transa,
                              transb,
                              m,
                              n,
                              k,
                              alpha,
                              a, rs_a0, cs_a0,
                              b, rs_b0, cs_b0,
                              beta,
                              c, rs_c0, cs_c0
                          );
              }
              break;
          default:
              return BLIS_FAILURE;
        }
    }

    return BLIS_FAILURE;
}
