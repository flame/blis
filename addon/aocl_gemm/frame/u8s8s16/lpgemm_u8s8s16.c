/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2022, Advanced Micro Devices, Inc. All rights reserved.

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
#include "lpgemm_u8s8s16.h"
#include "lpgemm_packb.h"
#include "lpgemm_6x32rowmajor.h"
#include "lpgemm_utils.h"
#include "lpgemm_config.h"

void lpgemm_rowvar_u8s8s16o16
  (
    const dim_t m,
    const dim_t n,
    const dim_t k,
    const uint8_t *a,
    const dim_t rs_a,
    const dim_t cs_a,
    const int8_t *b,
    const dim_t rs_b,
    const dim_t cs_b,
    int16_t *c,
    const dim_t rs_c,
    int16_t alpha,
    int16_t beta
  )
{
  // To Do: Constant declaration's to be moved to config files
  dim_t NC = 1024;
  dim_t KC = 1024;
  dim_t MC = 144;
  dim_t NR = 32;

  const int8_t *b_use;
  const uint8_t *a_use;

  for (dim_t jc = 0; jc < n; jc += NC)
  {
    dim_t nc0 = ((jc + NC) <= n) ? NC : (n % NC);

    for (dim_t pc = 0; pc < k; pc += KC)
    {
      int32_t beta0 = (pc == 0) ? beta : 1;
      dim_t kc0 = ((pc + KC) <= k) ? KC : (k % KC);

      int kc0_updated = kc0;

      // Making multiple of 2 to suit k in vpmaddubsw
      kc0_updated += (kc0_updated & 0x1);      

      // B part getting processed
      b_use = b + (jc * k) + (pc * nc0);

      for (dim_t ic = 0; ic < m; ic += MC)
      {
        dim_t mc0 = ((ic + MC) <= m) ? MC : (m % MC);

        a_use = a + (rs_a * ic) + (cs_a * pc);

        dim_t a_block_stride = rs_a;

        for (dim_t jr = 0; jr < nc0; jr += NR)
        {
          dim_t nr0 = ((jr + NR) <= nc0) ? NR : (nc0 % NR);

          // Calls for reorder B
          lpgemm_rowvar_u8s8s16o16_6x32(
              mc0, nr0, kc0,
              a_use, rs_a, cs_a, a_block_stride,
              (b_use + (jr * kc0_updated)), rs_b, cs_b,
              (c + (rs_c * ic) + jc + jr), rs_c, 1,
              alpha, beta0);
        }
      }
    }
  }
}
