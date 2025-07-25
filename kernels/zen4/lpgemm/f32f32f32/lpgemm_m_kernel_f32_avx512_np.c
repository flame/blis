/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2023 - 2025, Advanced Micro Devices, Inc. All rights reserved.

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
#include "immintrin.h"
#include "xmmintrin.h"
#include "blis.h"

#ifdef BLIS_ADDON_LPGEMM

#include "lpgemm_kernel_macros_f32.h"

#define MR 6
#define NR 64

LPGEMM_MAIN_KERN(float,float,float,f32f32f32of32_avx512_6x64m_np)
{
    //Call RD kernels if B is transposed
    if(rs_b == 1 && n0 != 1 )
    {
      lpgemm_rowvar_f32f32f32of32_avx512_6x64m_rd
        (
          m0, n0, k0,
          a, rs_a, cs_a, ps_a,
          b, rs_b, cs_b,
          c, rs_c, cs_c,
          alpha, beta,
          post_ops_list, post_ops_attr
        );
        return;
    }

    uint64_t n_left = n0 % 64;  //n0 is expected to be n0<=NR
    // First check whether this is a edge case in the n dimension.
    // If so, dispatch other 12x?m kernels, as needed.
    if ( n_left )
    {
        float*  cij = (float* )c;
        float*  bj  = (float* )b;
        float*  ai  = (float* )a;

        if ( 48 <= n_left )
        {
            const dim_t nr_cur = 48;

            lpgemm_rowvar_f32f32f32of32_avx512_6x48m_np
            (
              m0, k0,
              ai,  rs_a, cs_a, ps_a,
              bj,  rs_b, cs_b,
              cij, rs_c,
              alpha, beta,
              post_ops_list, post_ops_attr
            );

            cij += nr_cur*cs_c; bj += nr_cur*cs_b; n_left -= nr_cur;
			      post_ops_attr.post_op_c_j += 48;
        }

        if ( 32 <= n_left )
        {
            const dim_t nr_cur = 32;
            lpgemm_rowvar_f32f32f32of32_avx512_6x32m_np
            (
              m0, k0,
              ai,  rs_a, cs_a, ps_a,
              bj,  rs_b, cs_b,
              cij, rs_c,
              alpha, beta,
              post_ops_list, post_ops_attr
            );
            cij += nr_cur*cs_c; bj += nr_cur*cs_b; n_left -= nr_cur;
            post_ops_attr.post_op_c_j += 32;
        }

        if( n_left >= 16)
        {
          lpgemm_rowvar_f32f32f32of32_avx512_6x16m_np
            (
              m0, k0,
              ai,  rs_a, cs_a, ps_a,
              bj,  rs_b, cs_b,
              cij, rs_c,
              alpha, beta,
              post_ops_list, post_ops_attr );
              cij += 16*cs_c; bj += 16*cs_b; n_left -= 16;
              post_ops_attr.post_op_c_j += 16;
        }

        if( n_left >= 8)
        {
          dim_t nr_cur = n_left % 16;
          lpgemm_rowvar_f32f32f32of32_avx512_6xlt16m_np(m0, k0,
            ai,  rs_a, cs_a, ps_a,
            bj,  rs_b, cs_b,
            cij, rs_c,
            alpha, beta,
            n_left,
            post_ops_list, post_ops_attr );
            cij += nr_cur*cs_c; bj += nr_cur*cs_b; n_left -= nr_cur;
            post_ops_attr.post_op_c_j += nr_cur;
        }

        if( n_left > 0 )
        {
          lpgemm_rowvar_f32f32f32of32_6xlt8m_np(m0, k0,
            ai,  rs_a, cs_a, ps_a,
            bj,  rs_b, cs_b,
            cij, rs_c,
            alpha, beta,
            n_left,
            post_ops_list, post_ops_attr );
        }

      return;
    }

    // Typecast local copies of integers in case dim_t and inc_t are a
    // different size than is expected by load instructions.
    uint64_t k_iter = k0;

    uint64_t m_iter = m0 / 6;
    uint64_t m_left = m0 % 6;

    // Query the panel stride of A and convert it to units of bytes.
    if ( m_iter == 0 ){    goto consider_edge_cases; }

    __m512 zmm0, zmm1, zmm2, zmm3, zmm4, zmm5, zmm6, zmm7;
    __m512 zmm8, zmm9, zmm10, zmm11, zmm12, zmm13, zmm14, zmm15;
    __m512 zmm16, zmm17, zmm18, zmm19, zmm20, zmm21, zmm22, zmm23;
    __m512 zmm24, zmm25, zmm26, zmm27, zmm28, zmm29, zmm30, zmm31;

    /*Produce MRxNR outputs */
    for(dim_t m=0; m < m_iter; m++)
    {
      /* zero the accumulator registers */
      ZERO_ACC_ZMM_4_REG(zmm8, zmm9, zmm10, zmm11);
      ZERO_ACC_ZMM_4_REG(zmm12, zmm13, zmm14, zmm15);
      ZERO_ACC_ZMM_4_REG(zmm16, zmm17, zmm18, zmm19);
      ZERO_ACC_ZMM_4_REG(zmm20, zmm21, zmm22, zmm23);
      ZERO_ACC_ZMM_4_REG(zmm24, zmm25, zmm26, zmm27);
      ZERO_ACC_ZMM_4_REG(zmm28, zmm29, zmm30, zmm31);

      _mm256_zeroupper();

      float *abuf, *bbuf, *cbuf, *_cbuf;

      abuf = (float *)a + m * ps_a; // Move to next MRxKC in MCxKC (where MC>=MR)
      bbuf = (float *)b;  //Same KCxNR is used across different MRxKC in MCxKC
      cbuf = (float *)c + m * MR * rs_c; // Move to next MRXNR in output

      /*_mm_prefetch( (MR X NR) from C*/
      _mm_prefetch((cbuf + 0*rs_c), _MM_HINT_T0);
      _mm_prefetch((cbuf + 1*rs_c), _MM_HINT_T0);
      _mm_prefetch((cbuf + 2*rs_c), _MM_HINT_T0);
      _mm_prefetch((cbuf + 3*rs_c), _MM_HINT_T0);
      _mm_prefetch((cbuf + 4*rs_c), _MM_HINT_T0);
      _mm_prefetch((cbuf + 5*rs_c), _MM_HINT_T0);

      for(dim_t k = 0; k < k_iter; k++)
      {
        /*Load 32 elements from row0 of B*/
        zmm0 = _mm512_loadu_ps (bbuf );     //load 0-15 values from current row
        zmm1 = _mm512_loadu_ps (bbuf + 16); //load 16-31 values from current row

        /*Load Next 32 elements from row0 of B*/
        zmm6 = _mm512_loadu_ps (bbuf + 32); //load 32-47 from current row
        zmm7 = _mm512_loadu_ps (bbuf + 48); //load 48-63 from current row

        /*Broadcast col0 elements of 12 rows of A*/
        zmm2 = _mm512_set1_ps(*(abuf + 0*rs_a)); //broadcast c0r0
        zmm3 = _mm512_set1_ps(*(abuf + 1*rs_a)); //broadcast c0r1
        zmm4 = _mm512_set1_ps(*(abuf + 2*rs_a)); //broadcast c0r2
        zmm5 = _mm512_set1_ps(*(abuf + 3*rs_a)); //broadcast c0r3

        zmm8 = _mm512_fmadd_ps(zmm0, zmm2, zmm8);
        zmm9 = _mm512_fmadd_ps(zmm1, zmm2, zmm9);
        zmm10 = _mm512_fmadd_ps(zmm6, zmm2, zmm10);
        zmm11 = _mm512_fmadd_ps(zmm7, zmm2, zmm11);

        zmm12 = _mm512_fmadd_ps(zmm0, zmm3, zmm12);
        zmm13 = _mm512_fmadd_ps(zmm1, zmm3, zmm13);
        zmm14 = _mm512_fmadd_ps(zmm6, zmm3, zmm14);
        zmm15 = _mm512_fmadd_ps(zmm7, zmm3, zmm15);

        zmm2 = _mm512_set1_ps(*(abuf + 4*rs_a)); //broadcast c0r4
        zmm3 = _mm512_set1_ps(*(abuf + 5*rs_a)); //broadcast c0r5

        zmm16 = _mm512_fmadd_ps(zmm0, zmm4, zmm16);
        zmm17 = _mm512_fmadd_ps(zmm1, zmm4, zmm17);
        zmm18 = _mm512_fmadd_ps(zmm6, zmm4, zmm18);
        zmm19 = _mm512_fmadd_ps(zmm7, zmm4, zmm19);

        zmm20 = _mm512_fmadd_ps(zmm0, zmm5, zmm20);
        zmm21 = _mm512_fmadd_ps(zmm1, zmm5, zmm21);
        zmm22 = _mm512_fmadd_ps(zmm6, zmm5, zmm22);
        zmm23 = _mm512_fmadd_ps(zmm7, zmm5, zmm23);

        zmm24 = _mm512_fmadd_ps(zmm0, zmm2, zmm24);
        zmm25 = _mm512_fmadd_ps(zmm1, zmm2, zmm25);
        zmm26 = _mm512_fmadd_ps(zmm6, zmm2, zmm26);
        zmm27 = _mm512_fmadd_ps(zmm7, zmm2, zmm27);

        zmm28 = _mm512_fmadd_ps(zmm0, zmm3, zmm28);
        zmm29 = _mm512_fmadd_ps(zmm1, zmm3, zmm29);
        zmm30 = _mm512_fmadd_ps(zmm6, zmm3, zmm30);
        zmm31 = _mm512_fmadd_ps(zmm7, zmm3, zmm31);

        bbuf += rs_b;  //move b pointer to next row
        abuf += cs_a;  //move a pointer to next col

      }//kloop

      zmm0 = _mm512_set1_ps(alpha);
      ALPHA_MUL_ACC_ZMM_4_REG(zmm8,zmm9,zmm10,zmm11,zmm0)
      ALPHA_MUL_ACC_ZMM_4_REG(zmm12,zmm13,zmm14,zmm15,zmm0)
      ALPHA_MUL_ACC_ZMM_4_REG(zmm16,zmm17,zmm18,zmm19,zmm0)
      ALPHA_MUL_ACC_ZMM_4_REG(zmm20,zmm21,zmm22,zmm23,zmm0)
      ALPHA_MUL_ACC_ZMM_4_REG(zmm24,zmm25,zmm26,zmm27,zmm0)
      ALPHA_MUL_ACC_ZMM_4_REG(zmm28,zmm29,zmm30,zmm31,zmm0)

      if ( beta != 0 )
      {
        zmm3 = _mm512_set1_ps(beta);

        //load c and beta, convert to f32
        if ( ( post_ops_attr.buf_downscale != NULL ) &&
             ( post_ops_attr.is_first_k == TRUE ) )
        {
          //c[0, 0-15]
          BF16_F32_BETA_OP(zmm8, m, 0, 0, zmm0,zmm3);
          //c[0, 16-31]
          BF16_F32_BETA_OP(zmm9, m, 0, 1, zmm1,zmm3);
          //c[0, 32-47]
          BF16_F32_BETA_OP(zmm10, m, 0, 2, zmm0,zmm3);
          //c[0, 48-63]
          BF16_F32_BETA_OP(zmm11, m, 0, 3, zmm1,zmm3);
          //c[1, 0-15]
          BF16_F32_BETA_OP(zmm12, m, 1, 0, zmm0,zmm3);
          //c[1, 16-31]
          BF16_F32_BETA_OP(zmm13, m, 1, 1, zmm1,zmm3);
          //c[1, 32-47]
          BF16_F32_BETA_OP(zmm14, m, 1, 2, zmm0,zmm3);
          //c[1, 48-63]
          BF16_F32_BETA_OP(zmm15, m, 1, 3, zmm1,zmm3);
          //c[2, 0-15]
          BF16_F32_BETA_OP(zmm16, m, 2, 0, zmm0,zmm3);
          //c[2,16-31]
          BF16_F32_BETA_OP(zmm17, m, 2, 1, zmm1,zmm3);
          //c[2,32-47]
          BF16_F32_BETA_OP(zmm18, m, 2, 2, zmm0,zmm3);
          //c[2,48-63]
          BF16_F32_BETA_OP(zmm19, m, 2, 3, zmm1,zmm3);
          //c[3, 0-15]
          BF16_F32_BETA_OP(zmm20, m, 3, 0, zmm0,zmm3);
          //c[3,16-31]
          BF16_F32_BETA_OP(zmm21, m, 3, 1, zmm1,zmm3);
          //c[3,32-47]
          BF16_F32_BETA_OP(zmm22, m, 3, 2, zmm0,zmm3);
          //c[3,48-63]
          BF16_F32_BETA_OP(zmm23, m, 3, 3, zmm1,zmm3);
          //c[4, 0-15]
          BF16_F32_BETA_OP(zmm24, m, 4, 0, zmm0,zmm3);
          //c[4, 16-31]
          BF16_F32_BETA_OP(zmm25, m, 4, 1, zmm1,zmm3);
          //c[4, 32-47]
          BF16_F32_BETA_OP(zmm26, m, 4, 2, zmm0,zmm3);
          //c[4, 48-63]
          BF16_F32_BETA_OP(zmm27, m, 4, 3, zmm1,zmm3);
          //c[5, 0-15]
          BF16_F32_BETA_OP(zmm28, m, 5, 0, zmm0,zmm3);
          //c[5,16-31]
          BF16_F32_BETA_OP(zmm29, m, 5, 1, zmm1,zmm3);
          //c[5,32-47]
          BF16_F32_BETA_OP(zmm30, m, 5, 2, zmm0,zmm3);
          //c[5,48-63]
          BF16_F32_BETA_OP(zmm31, m, 5, 3, zmm1,zmm3);
        }
        else
        {
          _cbuf = cbuf;
          //load c and multiply with beta and
          //add to accumulator and store back

          zmm0 = _mm512_loadu_ps(_cbuf);
          zmm1 = _mm512_loadu_ps(_cbuf + 16);
          zmm8 = _mm512_fmadd_ps(zmm0, zmm3, zmm8);
          zmm9 = _mm512_fmadd_ps(zmm1, zmm3, zmm9);

          zmm0 = _mm512_loadu_ps(_cbuf + 32);
          zmm1 = _mm512_loadu_ps(_cbuf + 48);
          zmm10 = _mm512_fmadd_ps(zmm0, zmm3, zmm10);
          zmm11 = _mm512_fmadd_ps(zmm1, zmm3, zmm11);
          _cbuf += rs_c;

          zmm0 = _mm512_loadu_ps(_cbuf);
          zmm1 = _mm512_loadu_ps(_cbuf + 16);
          zmm12 = _mm512_fmadd_ps(zmm0, zmm3, zmm12);
          zmm13 = _mm512_fmadd_ps(zmm1, zmm3, zmm13);

          zmm0 = _mm512_loadu_ps(_cbuf + 32);
          zmm1 = _mm512_loadu_ps(_cbuf + 48);
          zmm14 = _mm512_fmadd_ps(zmm0, zmm3, zmm14);
          zmm15 = _mm512_fmadd_ps(zmm1, zmm3, zmm15);
          _cbuf += rs_c;

          zmm0 = _mm512_loadu_ps(_cbuf);
          zmm1 = _mm512_loadu_ps(_cbuf + 16);
          zmm16 = _mm512_fmadd_ps(zmm0, zmm3, zmm16);
          zmm17 = _mm512_fmadd_ps(zmm1, zmm3, zmm17);

          zmm0 = _mm512_loadu_ps(_cbuf + 32);
          zmm1 = _mm512_loadu_ps(_cbuf + 48);
          zmm18 = _mm512_fmadd_ps(zmm0, zmm3, zmm18);
          zmm19 = _mm512_fmadd_ps(zmm1, zmm3, zmm19);
          _cbuf += rs_c;

          zmm0 = _mm512_loadu_ps(_cbuf);
          zmm1 = _mm512_loadu_ps(_cbuf + 16);
          zmm20 = _mm512_fmadd_ps(zmm0, zmm3, zmm20);
          zmm21 = _mm512_fmadd_ps(zmm1, zmm3, zmm21);

          zmm0 = _mm512_loadu_ps(_cbuf + 32);
          zmm1 = _mm512_loadu_ps(_cbuf + 48);
          zmm22 = _mm512_fmadd_ps(zmm0, zmm3, zmm22);
          zmm23 = _mm512_fmadd_ps(zmm1, zmm3, zmm23);
          _cbuf += rs_c;

          zmm0 = _mm512_loadu_ps(_cbuf);
          zmm1 = _mm512_loadu_ps(_cbuf + 16);
          zmm24 = _mm512_fmadd_ps(zmm0, zmm3, zmm24);
          zmm25 = _mm512_fmadd_ps(zmm1, zmm3, zmm25);

          zmm0 = _mm512_loadu_ps(_cbuf + 32);
          zmm1 = _mm512_loadu_ps(_cbuf + 48);
          zmm26 = _mm512_fmadd_ps(zmm0, zmm3, zmm26);
          zmm27 = _mm512_fmadd_ps(zmm1, zmm3, zmm27);
          _cbuf += rs_c;

          zmm0 = _mm512_loadu_ps(_cbuf);
          zmm1 = _mm512_loadu_ps(_cbuf + 16);
          zmm28 = _mm512_fmadd_ps(zmm0, zmm3, zmm28);
          zmm29 = _mm512_fmadd_ps(zmm1, zmm3, zmm29);

          zmm0 = _mm512_loadu_ps(_cbuf + 32);
          zmm1 = _mm512_loadu_ps(_cbuf + 48);
          zmm30 = _mm512_fmadd_ps(zmm0, zmm3, zmm30);
          zmm31 = _mm512_fmadd_ps(zmm1, zmm3, zmm31);
        }
      }

      if ( ( post_ops_attr.buf_downscale != NULL ) &&
           ( post_ops_attr.is_last_k == TRUE ) )
      {
        uint32_t tlsb, rounded, temp[16] = {0};
        int i;
        bfloat16* dest;

        CVT_STORE_F32_BF16_MASK_AVX512(16, zmm8, 0, 0);
        CVT_STORE_F32_BF16_MASK_AVX512(16, zmm9, 0, 1);
        CVT_STORE_F32_BF16_MASK_AVX512(16, zmm10, 0, 2);
        CVT_STORE_F32_BF16_MASK_AVX512(16, zmm11, 0, 3);

        CVT_STORE_F32_BF16_MASK_AVX512(16, zmm12, 1, 0);
        CVT_STORE_F32_BF16_MASK_AVX512(16, zmm13, 1, 1);
        CVT_STORE_F32_BF16_MASK_AVX512(16, zmm14, 1, 2);
        CVT_STORE_F32_BF16_MASK_AVX512(16, zmm15, 1, 3);

        CVT_STORE_F32_BF16_MASK_AVX512(16, zmm16, 2, 0);
        CVT_STORE_F32_BF16_MASK_AVX512(16, zmm17, 2, 1);
        CVT_STORE_F32_BF16_MASK_AVX512(16, zmm18, 2, 2);
        CVT_STORE_F32_BF16_MASK_AVX512(16, zmm19, 2, 3);

        CVT_STORE_F32_BF16_MASK_AVX512(16, zmm20, 3, 0);
        CVT_STORE_F32_BF16_MASK_AVX512(16, zmm21, 3, 1);
        CVT_STORE_F32_BF16_MASK_AVX512(16, zmm22, 3, 2);
        CVT_STORE_F32_BF16_MASK_AVX512(16, zmm23, 3, 3);

        CVT_STORE_F32_BF16_MASK_AVX512(16, zmm24, 4, 0);
        CVT_STORE_F32_BF16_MASK_AVX512(16, zmm25, 4, 1);
        CVT_STORE_F32_BF16_MASK_AVX512(16, zmm26, 4, 2);
        CVT_STORE_F32_BF16_MASK_AVX512(16, zmm27, 4, 3);

        CVT_STORE_F32_BF16_MASK_AVX512(16, zmm28, 5, 0);
        CVT_STORE_F32_BF16_MASK_AVX512(16, zmm29, 5, 1);
        CVT_STORE_F32_BF16_MASK_AVX512(16, zmm30, 5, 2);
        CVT_STORE_F32_BF16_MASK_AVX512(16, zmm31, 5, 3);
      }
      else
      {
        _mm512_storeu_ps(cbuf, zmm8);
        _mm512_storeu_ps(cbuf + 16, zmm9);
        _mm512_storeu_ps(cbuf + 32, zmm10);
        _mm512_storeu_ps(cbuf + 48, zmm11);
        cbuf += rs_c;
        _mm512_storeu_ps(cbuf, zmm12);
        _mm512_storeu_ps(cbuf + 16, zmm13);
        _mm512_storeu_ps(cbuf + 32, zmm14);
        _mm512_storeu_ps(cbuf + 48, zmm15);
        cbuf += rs_c;
        _mm512_storeu_ps(cbuf, zmm16);
        _mm512_storeu_ps(cbuf + 16, zmm17);
        _mm512_storeu_ps(cbuf + 32, zmm18);
        _mm512_storeu_ps(cbuf + 48, zmm19);
        cbuf += rs_c;
        _mm512_storeu_ps(cbuf, zmm20);
        _mm512_storeu_ps(cbuf + 16, zmm21);
        _mm512_storeu_ps(cbuf + 32, zmm22);
        _mm512_storeu_ps(cbuf + 48, zmm23);
        cbuf += rs_c;
        _mm512_storeu_ps(cbuf, zmm24);
        _mm512_storeu_ps(cbuf + 16, zmm25);
        _mm512_storeu_ps(cbuf + 32, zmm26);
        _mm512_storeu_ps(cbuf + 48, zmm27);
        cbuf += rs_c;
        _mm512_storeu_ps(cbuf, zmm28);
        _mm512_storeu_ps(cbuf + 16, zmm29);
        _mm512_storeu_ps(cbuf + 32, zmm30);
        _mm512_storeu_ps(cbuf + 48, zmm31);
      }

      post_ops_attr.post_op_c_i += MR;
    }//mloop

    consider_edge_cases:

    // Handle edge cases in the m dimension, if they exist.
    if( m_left )
    {
        const dim_t      i_edge = m0 - ( dim_t )m_left;

        float*  restrict cij = (float *)c + i_edge*rs_c;
        float*  restrict ai  = (float *)a + m_iter*ps_a;
        float*  restrict bj  = (float *)b;

        lpgemm_m_fringe_f32_ker_ft ker_fps[6] =
        {
          NULL,
          lpgemm_rowvar_f32f32f32of32_avx512_1x64_np,
          lpgemm_rowvar_f32f32f32of32_avx512_2x64_np,
          lpgemm_rowvar_f32f32f32of32_avx512_3x64_np,
          lpgemm_rowvar_f32f32f32of32_avx512_4x64_np,
          lpgemm_rowvar_f32f32f32of32_avx512_5x64_np
        };

        lpgemm_m_fringe_f32_ker_ft ker_fp = ker_fps[ m_left ];

        ker_fp
        (
          k0,
          ai, rs_a, cs_a,
          bj, rs_b, cs_b,
          cij,rs_c,
          alpha, beta,
          post_ops_list, post_ops_attr
        );
        return;
    }
}

LPGEMM_N_FRINGE_KERN(float,float,float,f32f32f32of32_avx512_6x48m_np)
{

    // Typecast local copies of integers in case dim_t and inc_t are a
    // different size than is expected by load instructions.
    uint64_t k_iter = k0;

    uint64_t m_iter = m0 / 6;
    uint64_t m_left = m0 % 6;

    // Query the panel stride of A and convert it to units of bytes.
    if ( m_iter == 0 ){    goto consider_edge_cases; }

    __m512 zmm0, zmm1, zmm2, zmm3, zmm4, zmm5, zmm6;
    __m512 zmm8, zmm9, zmm10, zmm12, zmm13, zmm14;
    __m512 zmm16, zmm17, zmm18, zmm20, zmm21, zmm22;
    __m512 zmm24, zmm25, zmm26, zmm28, zmm29, zmm30, zmm31;

    /*Produce MRxNR outputs */
    for(dim_t m=0; m < m_iter; m++)
    {
      /* zero the accumulator registers */
      ZERO_ACC_ZMM_4_REG(zmm8, zmm9, zmm10, zmm12);
      ZERO_ACC_ZMM_4_REG(zmm13, zmm14,zmm16, zmm17);
      ZERO_ACC_ZMM_4_REG(zmm18, zmm20, zmm21, zmm22);
      ZERO_ACC_ZMM_4_REG(zmm24, zmm25, zmm26, zmm28);
      ZERO_ACC_ZMM_4_REG(zmm29, zmm30, zmm31, zmm2);

      _mm256_zeroupper();

      float *abuf, *bbuf, *cbuf, *_cbuf;

      abuf = (float *)a + m * ps_a; // Move to next MRxKC in MCxKC (where MC>=MR)
      bbuf = (float *)b;  //Same KCxNR is used across different MRxKC in MCxKC
      cbuf = (float *)c + m * MR * rs_c; // Move to next MRXNR in output

      /*_mm_prefetch( (MR X NR) from C*/
      _mm_prefetch((cbuf + 0*rs_c), _MM_HINT_T0);
      _mm_prefetch((cbuf + 1*rs_c), _MM_HINT_T0);
      _mm_prefetch((cbuf + 2*rs_c), _MM_HINT_T0);
      _mm_prefetch((cbuf + 3*rs_c), _MM_HINT_T0);
      _mm_prefetch((cbuf + 4*rs_c), _MM_HINT_T0);

      for(dim_t k = 0; k < k_iter; k++)
      {
        /*Load 32 elements from row0 of B*/
        zmm0 = _mm512_loadu_ps (bbuf );     //load 0-15 values from current row
        zmm1 = _mm512_loadu_ps (bbuf + 16); //load 16-31 values from current row

        /*Load Next 32 elements from row0 of B*/
        zmm6 = _mm512_loadu_ps (bbuf + 32); //load 32-47 from current row

        /*Broadcast col0 elements of 12 rows of A*/
        zmm2 = _mm512_set1_ps(*(abuf + 0*rs_a)); //broadcast c0r0
        zmm3 = _mm512_set1_ps(*(abuf + 1*rs_a)); //broadcast c0r1
        zmm4 = _mm512_set1_ps(*(abuf + 2*rs_a)); //broadcast c0r2
        zmm5 = _mm512_set1_ps(*(abuf + 3*rs_a)); //broadcast c0r3

        zmm8 = _mm512_fmadd_ps(zmm0, zmm2, zmm8);
        zmm9 = _mm512_fmadd_ps(zmm1, zmm2, zmm9);
        zmm10 = _mm512_fmadd_ps(zmm6, zmm2, zmm10);

        zmm12 = _mm512_fmadd_ps(zmm0, zmm3, zmm12);
        zmm13 = _mm512_fmadd_ps(zmm1, zmm3, zmm13);
        zmm14 = _mm512_fmadd_ps(zmm6, zmm3, zmm14);

        zmm2 = _mm512_set1_ps(*(abuf + 4*rs_a)); //broadcast c0r4
        zmm3 = _mm512_set1_ps(*(abuf + 5*rs_a)); //broadcast c0r5

        zmm16 = _mm512_fmadd_ps(zmm0, zmm4, zmm16);
        zmm17 = _mm512_fmadd_ps(zmm1, zmm4, zmm17);
        zmm18 = _mm512_fmadd_ps(zmm6, zmm4, zmm18);

        zmm20 = _mm512_fmadd_ps(zmm0, zmm5, zmm20);
        zmm21 = _mm512_fmadd_ps(zmm1, zmm5, zmm21);
        zmm22 = _mm512_fmadd_ps(zmm6, zmm5, zmm22);

        zmm24 = _mm512_fmadd_ps(zmm0, zmm2, zmm24);
        zmm25 = _mm512_fmadd_ps(zmm1, zmm2, zmm25);
        zmm26 = _mm512_fmadd_ps(zmm6, zmm2, zmm26);

        zmm28 = _mm512_fmadd_ps(zmm0, zmm3, zmm28);
        zmm29 = _mm512_fmadd_ps(zmm1, zmm3, zmm29);
        zmm30 = _mm512_fmadd_ps(zmm6, zmm3, zmm30);

        bbuf += rs_b;  //move b pointer to next row
        abuf += cs_a;  //move a pointer to next col

      }//kloop

      zmm0 = _mm512_set1_ps(alpha);
      ALPHA_MUL_ACC_ZMM_4_REG(zmm8,zmm9,zmm10,zmm12,zmm0)
      ALPHA_MUL_ACC_ZMM_4_REG(zmm13,zmm14,zmm16,zmm17,zmm0)
      ALPHA_MUL_ACC_ZMM_4_REG(zmm18,zmm20,zmm21,zmm22,zmm0)
      ALPHA_MUL_ACC_ZMM_4_REG(zmm24,zmm25,zmm26,zmm28,zmm0)
      ALPHA_MUL_ACC_ZMM_4_REG(zmm29,zmm30,zmm31,zmm2,zmm0)

      if ( beta != 0 )
      {
        zmm3 = _mm512_set1_ps(beta);

        //load c and beta, convert to f32
        if ( ( post_ops_attr.buf_downscale != NULL ) &&
             ( post_ops_attr.is_first_k == TRUE ) )
			  {
          //c[0, 0-15]
          BF16_F32_BETA_OP(zmm8, m, 0, 0, zmm0,zmm3);
          //c[0, 16-31]
          BF16_F32_BETA_OP(zmm9, m, 0, 1, zmm1,zmm3);
          //c[0, 32-47]
          BF16_F32_BETA_OP(zmm10, m, 0, 2, zmm0,zmm3);
          //c[1, 0-15]
          BF16_F32_BETA_OP(zmm12, m, 1, 0, zmm0,zmm3);
          //c[1, 16-31]
          BF16_F32_BETA_OP(zmm13, m, 1, 1, zmm1,zmm3);
          //c[1, 32-47]
          BF16_F32_BETA_OP(zmm14, m, 1, 2, zmm0,zmm3);
          //c[2, 0-15]
          BF16_F32_BETA_OP(zmm16, m, 2, 0, zmm0,zmm3);
          //c[2,16-31]
          BF16_F32_BETA_OP(zmm17, m, 2, 1, zmm1,zmm3);
          //c[2,32-47]
          BF16_F32_BETA_OP(zmm18, m, 2, 2, zmm0,zmm3);
          //c[3, 0-15]
          BF16_F32_BETA_OP(zmm20, m, 3, 0, zmm0,zmm3);
          //c[3,16-31]
          BF16_F32_BETA_OP(zmm21, m, 3, 1, zmm1,zmm3);
          //c[3,32-47]
          BF16_F32_BETA_OP(zmm22, m, 3, 2, zmm0,zmm3);
          //c[4, 0-15]
          BF16_F32_BETA_OP(zmm24, m, 4, 0, zmm0,zmm3);
          //c[4, 16-31]
          BF16_F32_BETA_OP(zmm25, m, 4, 1, zmm1,zmm3);
          //c[4, 32-47]
          BF16_F32_BETA_OP(zmm26, m, 4, 2, zmm0,zmm3);
          //c[5, 0-15]
          BF16_F32_BETA_OP(zmm28, m, 5, 0, zmm0,zmm3);
          //c[5,16-31]
          BF16_F32_BETA_OP(zmm29, m, 5, 1, zmm1,zmm3);
          //c[5,32-47]
          BF16_F32_BETA_OP(zmm30, m, 5, 2, zmm0,zmm3);
        }
        else
        {
          _cbuf = cbuf;
          //load c and multiply with beta and
          //add to accumulator and store back

          zmm0 = _mm512_loadu_ps(_cbuf);
          zmm1 = _mm512_loadu_ps(_cbuf + 16);
          zmm8 = _mm512_fmadd_ps(zmm0, zmm3, zmm8);
          zmm9 = _mm512_fmadd_ps(zmm1, zmm3, zmm9);

          zmm0 = _mm512_loadu_ps(_cbuf + 32);
          zmm10 = _mm512_fmadd_ps(zmm0, zmm3, zmm10);
          _cbuf += rs_c;

          zmm0 = _mm512_loadu_ps(_cbuf);
          zmm1 = _mm512_loadu_ps(_cbuf + 16);
          zmm12 = _mm512_fmadd_ps(zmm0, zmm3, zmm12);
          zmm13 = _mm512_fmadd_ps(zmm1, zmm3, zmm13);

          zmm0 = _mm512_loadu_ps(_cbuf + 32);
          zmm14 = _mm512_fmadd_ps(zmm0, zmm3, zmm14);
          _cbuf += rs_c;

          zmm0 = _mm512_loadu_ps(_cbuf);
          zmm1 = _mm512_loadu_ps(_cbuf + 16);
          zmm16 = _mm512_fmadd_ps(zmm0, zmm3, zmm16);
          zmm17 = _mm512_fmadd_ps(zmm1, zmm3, zmm17);

          zmm0 = _mm512_loadu_ps(_cbuf + 32);
          zmm18 = _mm512_fmadd_ps(zmm0, zmm3, zmm18);
          _cbuf += rs_c;

          zmm0 = _mm512_loadu_ps(_cbuf);
          zmm1 = _mm512_loadu_ps(_cbuf + 16);
          zmm20 = _mm512_fmadd_ps(zmm0, zmm3, zmm20);
          zmm21 = _mm512_fmadd_ps(zmm1, zmm3, zmm21);

          zmm0 = _mm512_loadu_ps(_cbuf + 32);
          zmm22 = _mm512_fmadd_ps(zmm0, zmm3, zmm22);
          _cbuf += rs_c;

          zmm0 = _mm512_loadu_ps(_cbuf);
          zmm1 = _mm512_loadu_ps(_cbuf + 16);
          zmm24 = _mm512_fmadd_ps(zmm0, zmm3, zmm24);
          zmm25 = _mm512_fmadd_ps(zmm1, zmm3, zmm25);

          zmm0 = _mm512_loadu_ps(_cbuf + 32);
          zmm26 = _mm512_fmadd_ps(zmm0, zmm3, zmm26);
          _cbuf += rs_c;

          zmm0 = _mm512_loadu_ps(_cbuf);
          zmm1 = _mm512_loadu_ps(_cbuf + 16);
          zmm28 = _mm512_fmadd_ps(zmm0, zmm3, zmm28);
          zmm29 = _mm512_fmadd_ps(zmm1, zmm3, zmm29);

          zmm0 = _mm512_loadu_ps(_cbuf + 32);
          zmm30 = _mm512_fmadd_ps(zmm0, zmm3, zmm30);
        }
      }

      // Store the results back to C.
      // If the post-ops buffer downscale is set, we will store the results
      // in bfloat16 format, otherwise we store them in float format.
      if ( ( post_ops_attr.buf_downscale != NULL ) &&
           ( post_ops_attr.is_last_k == TRUE ) )
      {
        uint32_t tlsb, rounded, temp[16] = {0};
        int i;
        bfloat16* dest;

        CVT_STORE_F32_BF16_MASK_AVX512(16, zmm8, 0, 0);
        CVT_STORE_F32_BF16_MASK_AVX512(16, zmm9, 0, 1);
        CVT_STORE_F32_BF16_MASK_AVX512(16, zmm10, 0, 2);

        CVT_STORE_F32_BF16_MASK_AVX512(16, zmm12, 1, 0);
        CVT_STORE_F32_BF16_MASK_AVX512(16, zmm13, 1, 1);
        CVT_STORE_F32_BF16_MASK_AVX512(16, zmm14, 1, 2);

        CVT_STORE_F32_BF16_MASK_AVX512(16, zmm16, 2, 0);
        CVT_STORE_F32_BF16_MASK_AVX512(16, zmm17, 2, 1);
        CVT_STORE_F32_BF16_MASK_AVX512(16, zmm18, 2, 2);

        CVT_STORE_F32_BF16_MASK_AVX512(16, zmm20, 3, 0);
        CVT_STORE_F32_BF16_MASK_AVX512(16, zmm21, 3, 1);
        CVT_STORE_F32_BF16_MASK_AVX512(16, zmm22, 3, 2);

        CVT_STORE_F32_BF16_MASK_AVX512(16, zmm24, 4, 0);
        CVT_STORE_F32_BF16_MASK_AVX512(16, zmm25, 4, 1);
        CVT_STORE_F32_BF16_MASK_AVX512(16, zmm26, 4, 2);

        CVT_STORE_F32_BF16_MASK_AVX512(16, zmm28, 5, 0);
        CVT_STORE_F32_BF16_MASK_AVX512(16, zmm29, 5, 1);
        CVT_STORE_F32_BF16_MASK_AVX512(16, zmm30, 5, 2);
      }
      else
      {
        _mm512_storeu_ps(cbuf, zmm8);
        _mm512_storeu_ps(cbuf + 16, zmm9);
        _mm512_storeu_ps(cbuf + 32, zmm10);
        cbuf += rs_c;
        _mm512_storeu_ps(cbuf, zmm12);
        _mm512_storeu_ps(cbuf + 16, zmm13);
        _mm512_storeu_ps(cbuf + 32, zmm14);
        cbuf += rs_c;
        _mm512_storeu_ps(cbuf, zmm16);
        _mm512_storeu_ps(cbuf + 16, zmm17);
        _mm512_storeu_ps(cbuf + 32, zmm18);
        cbuf += rs_c;
        _mm512_storeu_ps(cbuf, zmm20);
        _mm512_storeu_ps(cbuf + 16, zmm21);
        _mm512_storeu_ps(cbuf + 32, zmm22);
        cbuf += rs_c;
        _mm512_storeu_ps(cbuf, zmm24);
        _mm512_storeu_ps(cbuf + 16, zmm25);
        _mm512_storeu_ps(cbuf + 32, zmm26);
        cbuf += rs_c;
        _mm512_storeu_ps(cbuf, zmm28);
        _mm512_storeu_ps(cbuf + 16, zmm29);
        _mm512_storeu_ps(cbuf + 32, zmm30);
      }

      post_ops_attr.post_op_c_i += MR;
    }//mloop

    consider_edge_cases:

    // Handle edge cases in the m dimension, if they exist.
    if( m_left )
    {
        const dim_t      i_edge = m0 - ( dim_t )m_left;

        float*  restrict cij = (float *) c + i_edge*rs_c;
        float*  restrict ai  = (float *) a + m_iter*ps_a;
        float*  restrict bj  = (float *) b;

        lpgemm_m_fringe_f32_ker_ft ker_fps[6] =
        {
          NULL,
          lpgemm_rowvar_f32f32f32of32_avx512_1x48_np,
          lpgemm_rowvar_f32f32f32of32_avx512_2x48_np,
          lpgemm_rowvar_f32f32f32of32_avx512_3x48_np,
          lpgemm_rowvar_f32f32f32of32_avx512_4x48_np,
          lpgemm_rowvar_f32f32f32of32_avx512_5x48_np
        };

        lpgemm_m_fringe_f32_ker_ft ker_fp = ker_fps[ m_left ];

        ker_fp
        (
          k0,
          ai, rs_a, cs_a,
          bj, rs_b, cs_b,
          cij,rs_c,
          alpha, beta,
          post_ops_list, post_ops_attr
        );
        return;
    }
}


LPGEMM_N_FRINGE_KERN(float,float,float,f32f32f32of32_avx512_6x32m_np)
{

    // Typecast local copies of integers in case dim_t and inc_t are a
    // different size than is expected by load instructions.
    uint64_t k_iter = k0;

    uint64_t m_iter = m0 / 6;
    uint64_t m_left = m0 % 6;

    // Query the panel stride of A and convert it to units of bytes.
    if ( m_iter == 0 ){    goto consider_edge_cases; }

    __m512 zmm0, zmm1, zmm2, zmm3, zmm4, zmm5;
    __m512 zmm8, zmm9, zmm12, zmm13;
    __m512 zmm16, zmm17, zmm20, zmm21;
    __m512 zmm24, zmm25, zmm28, zmm29;

    /*Produce MRxNR outputs */
    for(dim_t m=0; m < m_iter; m++)
    {
      /* zero the accumulator registers */
      ZERO_ACC_ZMM_4_REG(zmm8, zmm9, zmm12, zmm13);
      ZERO_ACC_ZMM_4_REG(zmm16, zmm17, zmm20, zmm21);
      ZERO_ACC_ZMM_4_REG(zmm24, zmm25, zmm28, zmm29);

      _mm256_zeroupper();

      float *abuf, *bbuf, *cbuf, *_cbuf;

      abuf = (float *)a + m * ps_a; // Move to next MRxKC in MCxKC (where MC>=MR)
      bbuf = (float *)b;  //Same KCxNR is used across different MRxKC in MCxKC
      cbuf = (float *)c + m * MR * rs_c; // Move to next MRXNR in output

      for(dim_t k = 0; k < k_iter; k++)
      {
        /*Load 32 elements from row0 of B*/
        zmm0 = _mm512_loadu_ps (bbuf );     //load 0-15 values from current row
        zmm1 = _mm512_loadu_ps (bbuf + 16); //load 16-31 values from current row

       /*Broadcast col0 elements of 12 rows of A*/
        zmm2 = _mm512_set1_ps(*(abuf + 0*rs_a)); //broadcast c0r0
        zmm3 = _mm512_set1_ps(*(abuf + 1*rs_a)); //broadcast c0r1
        zmm4 = _mm512_set1_ps(*(abuf + 2*rs_a)); //broadcast c0r2
        zmm5 = _mm512_set1_ps(*(abuf + 3*rs_a)); //broadcast c0r3

        zmm8 = _mm512_fmadd_ps(zmm0, zmm2, zmm8);
        zmm9 = _mm512_fmadd_ps(zmm1, zmm2, zmm9);

        zmm12 = _mm512_fmadd_ps(zmm0, zmm3, zmm12);
        zmm13 = _mm512_fmadd_ps(zmm1, zmm3, zmm13);

        zmm2 = _mm512_set1_ps(*(abuf + 4*rs_a)); //broadcast c0r4
        zmm3 = _mm512_set1_ps(*(abuf + 5*rs_a)); //broadcast c0r5

        zmm16 = _mm512_fmadd_ps(zmm0, zmm4, zmm16);
        zmm17 = _mm512_fmadd_ps(zmm1, zmm4, zmm17);

        zmm20 = _mm512_fmadd_ps(zmm0, zmm5, zmm20);
        zmm21 = _mm512_fmadd_ps(zmm1, zmm5, zmm21);

        zmm24 = _mm512_fmadd_ps(zmm0, zmm2, zmm24);
        zmm25 = _mm512_fmadd_ps(zmm1, zmm2, zmm25);

        zmm28 = _mm512_fmadd_ps(zmm0, zmm3, zmm28);
        zmm29 = _mm512_fmadd_ps(zmm1, zmm3, zmm29);

        bbuf += rs_b;  //move b pointer to next row
        abuf += cs_a;  //move a pointer to next col

      }//kloop

      zmm0 = _mm512_set1_ps(alpha);

      ALPHA_MUL_ACC_ZMM_4_REG(zmm8,zmm9,zmm12,zmm13,zmm0)
      ALPHA_MUL_ACC_ZMM_4_REG(zmm16,zmm17,zmm20,zmm21,zmm0)
      ALPHA_MUL_ACC_ZMM_4_REG(zmm24,zmm25,zmm28,zmm29,zmm0)

      if ( beta != 0 )
      {
        zmm3 = _mm512_set1_ps(beta);

        //load c and beta, convert to f32
        if ( ( post_ops_attr.buf_downscale != NULL ) &&
				     ( post_ops_attr.is_first_k == TRUE ) )
			  {
          //c[0, 0-15]
          BF16_F32_BETA_OP(zmm8, m, 0, 0, zmm0,zmm3);
          //c[0, 16-31]
          BF16_F32_BETA_OP(zmm9, m, 0, 1, zmm1,zmm3);
          //c[1, 0-15]
          BF16_F32_BETA_OP(zmm12, m, 1, 0, zmm0,zmm3);
          //c[1, 16-31]
          BF16_F32_BETA_OP(zmm13, m, 1, 1, zmm1,zmm3);
          //c[2, 0-15]
          BF16_F32_BETA_OP(zmm16, m, 2, 0, zmm0,zmm3);
          //c[2,16-31]
          BF16_F32_BETA_OP(zmm17, m, 2, 1, zmm1,zmm3);
          //c[3, 0-15]
          BF16_F32_BETA_OP(zmm20, m, 3, 0, zmm0,zmm3);
          //c[3,16-31]
          BF16_F32_BETA_OP(zmm21, m, 3, 1, zmm1,zmm3);
          //c[4, 0-15]
          BF16_F32_BETA_OP(zmm24, m, 4, 0, zmm0,zmm3);
          //c[4, 16-31]
          BF16_F32_BETA_OP(zmm25, m, 4, 1, zmm1,zmm3);
          //c[5, 0-15]
          BF16_F32_BETA_OP(zmm28, m, 5, 0, zmm0,zmm3);
          //c[5,16-31]
          BF16_F32_BETA_OP(zmm29, m, 5, 1, zmm1,zmm3);
        }
        else
        {
          _cbuf = cbuf;
          //load c and multiply with beta and
          //add to accumulator and store back

          zmm0 = _mm512_loadu_ps(_cbuf);
          zmm1 = _mm512_loadu_ps(_cbuf + 16);
          zmm8 = _mm512_fmadd_ps(zmm0, zmm3, zmm8);
          zmm9 = _mm512_fmadd_ps(zmm1, zmm3, zmm9);
          _cbuf += rs_c;

          zmm0 = _mm512_loadu_ps(_cbuf);
          zmm1 = _mm512_loadu_ps(_cbuf + 16);
          zmm12 = _mm512_fmadd_ps(zmm0, zmm3, zmm12);
          zmm13 = _mm512_fmadd_ps(zmm1, zmm3, zmm13);
          _cbuf += rs_c;

          zmm0 = _mm512_loadu_ps(_cbuf);
          zmm1 = _mm512_loadu_ps(_cbuf + 16);
          zmm16 = _mm512_fmadd_ps(zmm0, zmm3, zmm16);
          zmm17 = _mm512_fmadd_ps(zmm1, zmm3, zmm17);
          _cbuf += rs_c;

          zmm0 = _mm512_loadu_ps(_cbuf);
          zmm1 = _mm512_loadu_ps(_cbuf + 16);
          zmm20 = _mm512_fmadd_ps(zmm0, zmm3, zmm20);
          zmm21 = _mm512_fmadd_ps(zmm1, zmm3, zmm21);
          _cbuf += rs_c;

          zmm0 = _mm512_loadu_ps(_cbuf);
          zmm1 = _mm512_loadu_ps(_cbuf + 16);
          zmm24 = _mm512_fmadd_ps(zmm0, zmm3, zmm24);
          zmm25 = _mm512_fmadd_ps(zmm1, zmm3, zmm25);
          _cbuf += rs_c;

          zmm0 = _mm512_loadu_ps(_cbuf);
          zmm1 = _mm512_loadu_ps(_cbuf + 16);
          zmm28 = _mm512_fmadd_ps(zmm0, zmm3, zmm28);
          zmm29 = _mm512_fmadd_ps(zmm1, zmm3, zmm29);
        }
      }

      // Store the results back to C.
      if ( ( post_ops_attr.buf_downscale != NULL ) &&
           ( post_ops_attr.is_last_k == TRUE ) )
      {
        uint32_t tlsb, rounded, temp[16] = {0};
        int i;
        bfloat16* dest;

        CVT_STORE_F32_BF16_MASK_AVX512(16, zmm8, 0, 0);
        CVT_STORE_F32_BF16_MASK_AVX512(16, zmm9, 0, 1);

        CVT_STORE_F32_BF16_MASK_AVX512(16, zmm12, 1, 0);
        CVT_STORE_F32_BF16_MASK_AVX512(16, zmm13, 1, 1);

        CVT_STORE_F32_BF16_MASK_AVX512(16, zmm16, 2, 0);
        CVT_STORE_F32_BF16_MASK_AVX512(16, zmm17, 2, 1);

        CVT_STORE_F32_BF16_MASK_AVX512(16, zmm20, 3, 0);
        CVT_STORE_F32_BF16_MASK_AVX512(16, zmm21, 3, 1);

        CVT_STORE_F32_BF16_MASK_AVX512(16, zmm24, 4, 0);
        CVT_STORE_F32_BF16_MASK_AVX512(16, zmm25, 4, 1);

        CVT_STORE_F32_BF16_MASK_AVX512(16, zmm28, 5, 0);
        CVT_STORE_F32_BF16_MASK_AVX512(16, zmm29, 5, 1);
      }
      else
      {
        _mm512_storeu_ps(cbuf, zmm8);
        _mm512_storeu_ps(cbuf + 16, zmm9);
        cbuf += rs_c;
        _mm512_storeu_ps(cbuf, zmm12);
        _mm512_storeu_ps(cbuf + 16, zmm13);
        cbuf += rs_c;
        _mm512_storeu_ps(cbuf, zmm16);
        _mm512_storeu_ps(cbuf + 16, zmm17);
        cbuf += rs_c;
        _mm512_storeu_ps(cbuf, zmm20);
        _mm512_storeu_ps(cbuf + 16, zmm21);
        cbuf += rs_c;
        _mm512_storeu_ps(cbuf, zmm24);
        _mm512_storeu_ps(cbuf + 16, zmm25);
        cbuf += rs_c;
        _mm512_storeu_ps(cbuf, zmm28);
        _mm512_storeu_ps(cbuf + 16, zmm29);
      }

      post_ops_attr.post_op_c_i += MR;
    }//mloop

    consider_edge_cases:

    // Handle edge cases in the m dimension, if they exist.
    if( m_left )
    {
        const dim_t      i_edge = m0 - ( dim_t )m_left;

        float*  restrict cij = (float *) c + i_edge*rs_c;
        float*  restrict ai  = (float *) a + m_iter*ps_a;
        float*  restrict bj  = (float *) b;

        lpgemm_m_fringe_f32_ker_ft ker_fps[6] =
        {
          NULL,
          lpgemm_rowvar_f32f32f32of32_avx512_1x32_np,
          lpgemm_rowvar_f32f32f32of32_avx512_2x32_np,
          lpgemm_rowvar_f32f32f32of32_avx512_3x32_np,
          lpgemm_rowvar_f32f32f32of32_avx512_4x32_np,
          lpgemm_rowvar_f32f32f32of32_avx512_5x32_np
        };

        lpgemm_m_fringe_f32_ker_ft ker_fp = ker_fps[ m_left ];

        ker_fp
        (
          k0,
          ai, rs_a, cs_a,
          bj, rs_b, cs_b,
          cij,rs_c,
          alpha, beta,
          post_ops_list, post_ops_attr
        );
        return;
    }
}


LPGEMM_N_FRINGE_KERN(float,float,float,f32f32f32of32_avx512_6x16m_np)
{

    // Typecast local copies of integers in case dim_t and inc_t are a
    // different size than is expected by load instructions.
    uint64_t k_iter = k0;

    uint64_t m_iter = m0 / 6;
    uint64_t m_left = m0 % 6;

    // Query the panel stride of A and convert it to units of bytes.
    if ( m_iter == 0 ){    goto consider_edge_cases; }

    __m512 zmm0, zmm2, zmm3, zmm4, zmm5;
    __m512 zmm8, zmm12, zmm16, zmm20, zmm24, zmm28;

    /*Produce MRxNR outputs */
    for(dim_t m=0; m < m_iter; m++)
    {
      /* zero the accumulator registers */
      ZERO_ACC_ZMM_4_REG(zmm8, zmm12, zmm16, zmm20);
      ZERO_ACC_ZMM_2_REG(zmm24, zmm28);

      _mm256_zeroupper();

      float *abuf, *bbuf, *cbuf, *_cbuf;

      abuf = (float *)a + m * ps_a; // Move to next MRxKC in MCxKC (where MC>=MR)
      bbuf = (float *)b;  //Same KCxNR is used across different MRxKC in MCxKC
      cbuf = (float *)c + m * MR * rs_c; // Move to next MRXNR in output

      for(dim_t k = 0; k < k_iter; k++)
      {
        /*Load 32 elements from row0 of B*/
        zmm0 = _mm512_loadu_ps (bbuf );     //load 0-15 values from current row

       /*Broadcast col0 elements of 12 rows of A*/
        zmm2 = _mm512_set1_ps(*(abuf + 0*rs_a)); //broadcast c0r0
        zmm3 = _mm512_set1_ps(*(abuf + 1*rs_a)); //broadcast c0r1
        zmm4 = _mm512_set1_ps(*(abuf + 2*rs_a)); //broadcast c0r2
        zmm5 = _mm512_set1_ps(*(abuf + 3*rs_a)); //broadcast c0r3

        zmm8 = _mm512_fmadd_ps(zmm0, zmm2, zmm8);

        zmm12 = _mm512_fmadd_ps(zmm0, zmm3, zmm12);

        zmm2 = _mm512_set1_ps(*(abuf + 4*rs_a)); //broadcast c0r4
        zmm3 = _mm512_set1_ps(*(abuf + 5*rs_a)); //broadcast c0r5

        zmm16 = _mm512_fmadd_ps(zmm0, zmm4, zmm16);

        zmm20 = _mm512_fmadd_ps(zmm0, zmm5, zmm20);

        zmm24 = _mm512_fmadd_ps(zmm0, zmm2, zmm24);

        zmm28 = _mm512_fmadd_ps(zmm0, zmm3, zmm28);

        bbuf += rs_b;  //move b pointer to next row
        abuf += cs_a;  //move a pointer to next col

      }//kloop

      zmm0 = _mm512_set1_ps(alpha);

      ALPHA_MUL_ACC_ZMM_4_REG(zmm8,zmm12, zmm16, zmm20, zmm0)
      ALPHA_MUL_ACC_ZMM_2_REG(zmm24, zmm28, zmm0)

      if ( beta != 0 )
      {
        zmm3 = _mm512_set1_ps(beta);

        //load c and beta, convert to f32
        if ( ( post_ops_attr.buf_downscale != NULL ) &&
				     ( post_ops_attr.is_first_k == TRUE ) )
			  {
          //c[0, 0-15]
          BF16_F32_BETA_OP(zmm8, m, 0, 0, zmm0,zmm3);
          //c[1, 0-15]
          BF16_F32_BETA_OP(zmm12, m, 1, 0, zmm0,zmm3);
          //c[2, 0-15]
          BF16_F32_BETA_OP(zmm16, m, 2, 0, zmm0,zmm3);
          //c[3, 0-15]
          BF16_F32_BETA_OP(zmm20, m, 3, 0, zmm0,zmm3);
          //c[4, 0-15]
          BF16_F32_BETA_OP(zmm24, m, 4, 0, zmm0,zmm3);
          //c[5, 0-15]
          BF16_F32_BETA_OP(zmm28, m, 5, 0, zmm0,zmm3);
        }
        else
        {
          _cbuf = cbuf;
          //load c and multiply with beta and
          //add to accumulator and store back

          zmm0 = _mm512_loadu_ps(_cbuf);
          zmm8 = _mm512_fmadd_ps(zmm0, zmm3, zmm8);
          _cbuf += rs_c;

          zmm0 = _mm512_loadu_ps(_cbuf);
          zmm12 = _mm512_fmadd_ps(zmm0, zmm3, zmm12);
          _cbuf += rs_c;

          zmm0 = _mm512_loadu_ps(_cbuf);
          zmm16 = _mm512_fmadd_ps(zmm0, zmm3, zmm16);
          _cbuf += rs_c;

          zmm0 = _mm512_loadu_ps(_cbuf);
          zmm20 = _mm512_fmadd_ps(zmm0, zmm3, zmm20);
          _cbuf += rs_c;

          zmm0 = _mm512_loadu_ps(_cbuf);
          zmm24 = _mm512_fmadd_ps(zmm0, zmm3, zmm24);
          _cbuf += rs_c;

          zmm0 = _mm512_loadu_ps(_cbuf);
          zmm28 = _mm512_fmadd_ps(zmm0, zmm3, zmm28);
        }
      }

      if ( ( post_ops_attr.buf_downscale != NULL ) &&
      ( post_ops_attr.is_last_k == TRUE ) )
      {
        uint32_t tlsb, rounded, temp[16] = {0};
        int i;
        bfloat16* dest;

        CVT_STORE_F32_BF16_MASK_AVX512(16, zmm8, 0, 0);

        CVT_STORE_F32_BF16_MASK_AVX512(16, zmm12, 1, 0);

        CVT_STORE_F32_BF16_MASK_AVX512(16, zmm16, 2, 0);

        CVT_STORE_F32_BF16_MASK_AVX512(16, zmm20, 3, 0);

        CVT_STORE_F32_BF16_MASK_AVX512(16, zmm24, 4, 0);

        CVT_STORE_F32_BF16_MASK_AVX512(16, zmm28, 5, 0);
      }
      else
      {
        _mm512_storeu_ps(cbuf, zmm8);
        cbuf += rs_c;
        _mm512_storeu_ps(cbuf, zmm12);
        cbuf += rs_c;
        _mm512_storeu_ps(cbuf, zmm16);
        cbuf += rs_c;
        _mm512_storeu_ps(cbuf, zmm20);
        cbuf += rs_c;
        _mm512_storeu_ps(cbuf, zmm24);
        cbuf += rs_c;
        _mm512_storeu_ps(cbuf, zmm28);
      }

      post_ops_attr.post_op_c_i += MR;
    }//mloop

    consider_edge_cases:

    // Handle edge cases in the m dimension, if they exist.
    if( m_left )
    {
        const dim_t      i_edge = m0 - ( dim_t )m_left;

        float*  restrict cij = (float *) c + i_edge*rs_c;
        float*  restrict ai  = (float *) a + m_iter*ps_a;
        float*  restrict bj  = (float *) b;

        lpgemm_m_fringe_f32_ker_ft ker_fps[6] =
        {
          NULL,
          lpgemm_rowvar_f32f32f32of32_avx512_1x16_np,
          lpgemm_rowvar_f32f32f32of32_avx512_2x16_np,
          lpgemm_rowvar_f32f32f32of32_avx512_3x16_np,
          lpgemm_rowvar_f32f32f32of32_avx512_4x16_np,
          lpgemm_rowvar_f32f32f32of32_avx512_5x16_np
        };

        lpgemm_m_fringe_f32_ker_ft ker_fp = ker_fps[ m_left ];

        ker_fp
        (
          k0,
          ai, rs_a, cs_a,
          bj, rs_b, cs_b,
          cij,rs_c,
          alpha, beta,
          post_ops_list, post_ops_attr
        );
        return;
    }

}


LPGEMM_N_LT_NR0_FRINGE_KERN(float,float,float,f32f32f32of32_avx512_6xlt16m_np)
{
    // Typecast local copies of integers in case dim_t and inc_t are a
    // different size than is expected by load instructions.
    uint64_t k_iter = k0;

    uint64_t m_iter = m0 / 6;
    uint64_t m_left = m0 % 6;

    // Query the panel stride of A and convert it to units of bytes.
    if ( m_iter == 0 ){    goto consider_edge_cases; }

    __m512 zmm0, zmm2, zmm3, zmm4, zmm5;
    __m512 zmm8, zmm12, zmm16, zmm20, zmm24, zmm28;

    __mmask16 mask16 = _cvtu32_mask16( 0xFFFF >> ( 16 - n0_rem ) );
    /*Produce MRxNR outputs */
    for(dim_t m=0; m < m_iter; m++)
    {
      /* zero the accumulator registers */
      ZERO_ACC_ZMM_4_REG(zmm8, zmm12, zmm16, zmm20);
      ZERO_ACC_ZMM_2_REG(zmm24, zmm28);

      _mm256_zeroupper();

      float *abuf, *bbuf, *cbuf, *_cbuf;

      abuf = (float *)a + m * ps_a; // Move to next MRxKC in MCxKC (where MC>=MR)
      bbuf = (float *)b;  //Same KCxNR is used across different MRxKC in MCxKC
      cbuf = (float *)c + m * MR * rs_c; // Move to next MRXNR in output

      for(dim_t k = 0; k < k_iter; k++)
      {
        /*Load 32 elements from row0 of B*/
        zmm0 = _mm512_maskz_loadu_ps (mask16, bbuf );     //load 0-15 values from current row

       /*Broadcast col0 elements of 12 rows of A*/
        zmm2 = _mm512_set1_ps(*(abuf + 0*rs_a)); //broadcast c0r0
        zmm3 = _mm512_set1_ps(*(abuf + 1*rs_a)); //broadcast c0r1
        zmm4 = _mm512_set1_ps(*(abuf + 2*rs_a)); //broadcast c0r2
        zmm5 = _mm512_set1_ps(*(abuf + 3*rs_a)); //broadcast c0r3

        zmm8 = _mm512_fmadd_ps(zmm0, zmm2, zmm8);

        zmm12 = _mm512_fmadd_ps(zmm0, zmm3, zmm12);

        zmm2 = _mm512_set1_ps(*(abuf + 4*rs_a)); //broadcast c0r4
        zmm3 = _mm512_set1_ps(*(abuf + 5*rs_a)); //broadcast c0r5

        zmm16 = _mm512_fmadd_ps(zmm0, zmm4, zmm16);

        zmm20 = _mm512_fmadd_ps(zmm0, zmm5, zmm20);

        zmm24 = _mm512_fmadd_ps(zmm0, zmm2, zmm24);

        zmm28 = _mm512_fmadd_ps(zmm0, zmm3, zmm28);

        bbuf += rs_b;  //move b pointer to next row
        abuf += cs_a;  //move a pointer to next col

      }//kloop

      zmm0 = _mm512_set1_ps(alpha);

      ALPHA_MUL_ACC_ZMM_4_REG(zmm8,zmm12, zmm16, zmm20, zmm0)
      ALPHA_MUL_ACC_ZMM_2_REG(zmm24, zmm28, zmm0)

      if ( beta != 0 )
      {
        zmm3 = _mm512_set1_ps(beta);

        //load c and beta, convert to f32
        if ( ( post_ops_attr.buf_downscale != NULL ) &&
				     ( post_ops_attr.is_first_k == TRUE ) )
			  {
          //c[0, 0-15]
          BF16_F32_BETA_OP_NLT16F_MASK(mask16, zmm8, 0, 0, zmm0,zmm3);
          //c[1, 0-15]
          BF16_F32_BETA_OP_NLT16F_MASK(mask16, zmm12, 1, 0, zmm0,zmm3);
          //c[2, 0-15]
          BF16_F32_BETA_OP_NLT16F_MASK(mask16, zmm16, 2, 0, zmm0,zmm3);
          //c[3, 0-15]
          BF16_F32_BETA_OP_NLT16F_MASK(mask16, zmm20, 3, 0, zmm0,zmm3);
          //c[4, 0-15]
          BF16_F32_BETA_OP_NLT16F_MASK(mask16, zmm24, 4, 0, zmm0,zmm3);
          //c[5, 0-15]
          BF16_F32_BETA_OP_NLT16F_MASK(mask16, zmm28, 5, 0, zmm0,zmm3);
        }
        else
        {
          _cbuf = cbuf;
          //load c and multiply with beta and
          //add to accumulator and store back

          zmm0 = _mm512_maskz_loadu_ps(mask16, _cbuf);
          zmm8 = _mm512_fmadd_ps(zmm0, zmm3, zmm8);
          _cbuf += rs_c;

          zmm0 = _mm512_maskz_loadu_ps(mask16, _cbuf);
          zmm12 = _mm512_fmadd_ps(zmm0, zmm3, zmm12);
          _cbuf += rs_c;

          zmm0 = _mm512_maskz_loadu_ps(mask16, _cbuf);
          zmm16 = _mm512_fmadd_ps(zmm0, zmm3, zmm16);
          _cbuf += rs_c;

          zmm0 = _mm512_maskz_loadu_ps(mask16, _cbuf);
          zmm20 = _mm512_fmadd_ps(zmm0, zmm3, zmm20);
          _cbuf += rs_c;

          zmm0 = _mm512_maskz_loadu_ps(mask16, _cbuf);
          zmm24 = _mm512_fmadd_ps(zmm0, zmm3, zmm24);
          _cbuf += rs_c;

          zmm0 = _mm512_maskz_loadu_ps(mask16, _cbuf);
          zmm28 = _mm512_fmadd_ps(zmm0, zmm3, zmm28);
        }
      }

      // Generate a mask16 of all 1's.
      __mmask16 mask_all1 = _cvtu32_mask16( 0xFFFF >> ( 16 - n0_rem ) );

      if ( ( post_ops_attr.buf_downscale != NULL ) &&
           ( post_ops_attr.is_last_k == TRUE ) )
      {
        uint32_t tlsb, rounded, temp[16] = {0};
        int i;
        bfloat16* dest;

        CVT_STORE_F32_BF16_MASK_AVX512(n0_rem, zmm8, 0, 0);

        CVT_STORE_F32_BF16_MASK_AVX512(n0_rem, zmm12, 1, 0);

        CVT_STORE_F32_BF16_MASK_AVX512(n0_rem, zmm16, 2, 0);

        CVT_STORE_F32_BF16_MASK_AVX512(n0_rem, zmm20, 3, 0);

        CVT_STORE_F32_BF16_MASK_AVX512(n0_rem, zmm24, 4, 0);

        CVT_STORE_F32_BF16_MASK_AVX512(n0_rem, zmm28, 5, 0);
      }
      else
      {
        _mm512_mask_storeu_ps(cbuf, mask_all1, zmm8);
        cbuf += rs_c;
        _mm512_mask_storeu_ps(cbuf, mask_all1, zmm12);
        cbuf += rs_c;
        _mm512_mask_storeu_ps(cbuf, mask_all1, zmm16);
        cbuf += rs_c;
        _mm512_mask_storeu_ps(cbuf, mask_all1, zmm20);
        cbuf += rs_c;
        _mm512_mask_storeu_ps(cbuf, mask_all1, zmm24);
        cbuf += rs_c;
        _mm512_mask_storeu_ps(cbuf, mask_all1, zmm28);
      }

      post_ops_attr.post_op_c_i += MR;
    }//mloop

    consider_edge_cases:

    // Handle edge cases in the m dimension, if they exist.
    if( m_left )
    {
        const dim_t      i_edge = m0 - ( dim_t )m_left;

        float*  restrict cij = (float *) c + i_edge*rs_c;
        float*  restrict ai  = (float *) a + m_iter*ps_a;
        float*  restrict bj  = (float *) b;

        lpgemm_mn_fringe_f32_mask_ker_ft ker_fps[6] =
        {
          NULL,
          lpgemm_rowvar_f32f32f32of32_avx512_1xlt16_np,
          lpgemm_rowvar_f32f32f32of32_avx512_2xlt16_np,
          lpgemm_rowvar_f32f32f32of32_avx512_3xlt16_np,
          lpgemm_rowvar_f32f32f32of32_avx512_4xlt16_np,
          lpgemm_rowvar_f32f32f32of32_avx512_5xlt16_np
        };

        lpgemm_mn_fringe_f32_mask_ker_ft ker_fp = ker_fps[ m_left ];

        ker_fp
        (
          k0,
          ai, rs_a, cs_a,
          bj, rs_b, cs_b,
          cij,rs_c,
          alpha, beta,
          n0_rem,
          post_ops_list, post_ops_attr
        );
        return;
    }
  }
#endif
