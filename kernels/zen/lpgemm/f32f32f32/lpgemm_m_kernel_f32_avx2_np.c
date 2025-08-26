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

#include "lpgemm_kernel_macros_f32_avx2.h"

#define MR 6
#define NR 16

LPGEMM_MAIN_KERN(float,float,float,f32f32f32of32_6x16m_np)
{
    //Call RD kernels if B is transposed
    if(rs_b == 1 && n0 != 1)
    {
        lpgemm_rowvar_f32f32f32of32_6x16m_rd
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

    uint64_t n_left = n0 % NR;  //n0 is expected to be n0<=NR
    // First check whether this is a edge case in the n dimension.
    // If so, dispatch other 6x?m kernels, as needed.
    if (n_left )
    {
        float*  cij = (float* )c;
        float*  bj  = (float* )b;
        float*  ai  = (float* )a;

        if ( 8 <= n_left )
        {
            const dim_t nr_cur = 8;

            lpgemm_rowvar_f32f32f32of32_6x8m_np
            (
              m0, k0,
              ai,  rs_a, cs_a, ps_a,
              bj,  rs_b, cs_b,
              cij, rs_c,
              alpha, beta,
              post_ops_list, post_ops_attr
            );

            cij += nr_cur*cs_c; bj += nr_cur*cs_b; n_left -= nr_cur;
            post_ops_attr.post_op_c_j += 8;
        }

        if ( 4 <= n_left )
        {
            const dim_t nr_cur = 4;

            lpgemm_rowvar_f32f32f32of32_6x4m_np
            (
              m0, k0,
              ai,  rs_a, cs_a, ps_a,
              bj,  rs_b, cs_b,
              cij, rs_c,
              alpha, beta,
              post_ops_list, post_ops_attr
            );
            cij += nr_cur*cs_c; bj += nr_cur*cs_b; n_left -= nr_cur;
            post_ops_attr.post_op_c_j += 4;
        }

        if ( 2 <= n_left )
        {
            const dim_t nr_cur = 2;

            lpgemm_rowvar_f32f32f32of32_6x2m_np
            (
              m0, k0,
              ai,  rs_a, cs_a, ps_a,
              bj,  rs_b, cs_b,
              cij, rs_c,
              alpha, beta,
              post_ops_list, post_ops_attr
            );
            cij += nr_cur*cs_c; bj += nr_cur*cs_b; n_left -= nr_cur;
            post_ops_attr.post_op_c_j += 2;
        }

        if ( 1 == n_left )
        {
            lpgemm_rowvar_f32f32f32of32_6x1m_np
            (
              m0, k0,
              ai,  rs_a, cs_a, ps_a,
              bj,  rs_b, cs_b,
              cij, rs_c,
              alpha, beta,
              post_ops_list, post_ops_attr
            );
        }

        return;
    }

    // Typecast local copies of integers in case dim_t and inc_t are a
    // different size than is expected by load instructions.
    uint64_t k_iter = (uint64_t)k0;

    uint64_t m_iter = (uint64_t)m0 / 6;
    uint64_t m_left = (uint64_t)m0 % 6;

    if ( m_iter == 0 ){    goto consider_edge_cases; }

    /*Declare the registers*/
    __m256 ymm0, ymm1, ymm2, ymm3;
    __m256 ymm4, ymm5, ymm6, ymm7;
    __m256 ymm8, ymm9, ymm10, ymm11;
    __m256 ymm12, ymm13, ymm14, ymm15;

    /*Produce MRxNR outputs */
    for(dim_t m=0; m < m_iter; m++)
    {
      /* zero the accumulator registers */
      ZERO_ACC_YMM_4_REG(ymm4, ymm5, ymm6, ymm7);
      ZERO_ACC_YMM_4_REG(ymm8,  ymm9,  ymm10, ymm11);
      ZERO_ACC_YMM_4_REG(ymm12, ymm13, ymm14, ymm15);

      float *abuf, *bbuf, *cbuf, *_cbuf;

      abuf = (float *)a + m * ps_a; // Move to next MRxKC in MCxKC (where MC>=MR)
      bbuf = (float *)b;  //Same KCxNR panel is used across MCxKC block
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
        /*Load 16 elements from row0 of B*/
        ymm0 = _mm256_loadu_ps(bbuf );
        ymm1 = _mm256_loadu_ps(bbuf + 8);
        bbuf += rs_b;  //move b pointer to next row

        ymm2 = _mm256_broadcast_ss((abuf + 0*rs_a)); //broadcast c0r0
        ymm3 = _mm256_broadcast_ss((abuf + 1*rs_a)); //broadcast c0r1

        ymm4 = _mm256_fmadd_ps(ymm0, ymm2, ymm4);
        ymm5 = _mm256_fmadd_ps(ymm1, ymm2, ymm5);
        ymm6 = _mm256_fmadd_ps(ymm0, ymm3, ymm6);
        ymm7 = _mm256_fmadd_ps(ymm1, ymm3, ymm7);

        ymm2 = _mm256_broadcast_ss((abuf + 2*rs_a)); //broadcast c0r2
        ymm3 = _mm256_broadcast_ss((abuf + 3*rs_a)); //broadcast c0r3

        ymm8 = _mm256_fmadd_ps(ymm0, ymm2, ymm8);
        ymm9 = _mm256_fmadd_ps(ymm1, ymm2, ymm9);
        ymm10 = _mm256_fmadd_ps(ymm0, ymm3, ymm10);
        ymm11 = _mm256_fmadd_ps(ymm1, ymm3, ymm11);

        ymm2 = _mm256_broadcast_ss((abuf + 4*rs_a)); //broadcast c0r4
        ymm3 = _mm256_broadcast_ss((abuf + 5*rs_a)); //broadcast c0r5
        abuf += cs_a;  //move a pointer to next col

        ymm12 = _mm256_fmadd_ps(ymm0, ymm2, ymm12);
        ymm13 = _mm256_fmadd_ps(ymm1, ymm2, ymm13);
        ymm14 = _mm256_fmadd_ps(ymm0, ymm3, ymm14);
        ymm15 = _mm256_fmadd_ps(ymm1, ymm3, ymm15);
      }//kloop

      ymm0 = _mm256_broadcast_ss(&(alpha));
      ALPHA_MUL_ACC_YMM_4_REG(ymm4,ymm5,ymm6,ymm7,ymm0)
      ALPHA_MUL_ACC_YMM_4_REG(ymm8,ymm9,ymm10,ymm11,ymm0)
      ALPHA_MUL_ACC_YMM_4_REG(ymm12,ymm13,ymm14,ymm15,ymm0)

      if ( beta != 0.0 )
      {
        ymm3 = _mm256_broadcast_ss(&(beta));

        if ( ( post_ops_attr.buf_downscale != NULL ) &&
          ( post_ops_attr.is_first_k == TRUE ) )
        {
          BF16_F32_C_BNZ_8(0,0,ymm0,ymm3,ymm4)
          BF16_F32_C_BNZ_8(0,1,ymm1,ymm3,ymm5)
          BF16_F32_C_BNZ_8(1,0,ymm0,ymm3,ymm6)
          BF16_F32_C_BNZ_8(1,1,ymm1,ymm3,ymm7)
          BF16_F32_C_BNZ_8(2,0,ymm0,ymm3,ymm8)
          BF16_F32_C_BNZ_8(2,1,ymm1,ymm3,ymm9)
          BF16_F32_C_BNZ_8(3,0,ymm0,ymm3,ymm10)
          BF16_F32_C_BNZ_8(3,1,ymm1,ymm3,ymm11)
          BF16_F32_C_BNZ_8(4,0,ymm0,ymm3,ymm12)
          BF16_F32_C_BNZ_8(4,1,ymm1,ymm3,ymm13)
          BF16_F32_C_BNZ_8(5,0,ymm0,ymm3,ymm14)
          BF16_F32_C_BNZ_8(5,1,ymm1,ymm3,ymm15)
        }
        else
        {
           _cbuf = cbuf;
          //load c and multiply with beta and
          //add to accumulator and store back
          F32_C_BNZ_8(_cbuf,rs_c,ymm0,ymm3,ymm4)
          F32_C_BNZ_8(_cbuf+8,rs_c,ymm1,ymm3,ymm5)
          _cbuf += rs_c;
          F32_C_BNZ_8(_cbuf,rs_c,ymm0,ymm3,ymm6)
          F32_C_BNZ_8(_cbuf+8,rs_c,ymm1,ymm3,ymm7)
          _cbuf += rs_c;
          F32_C_BNZ_8(_cbuf,rs_c,ymm0,ymm3,ymm8)
          F32_C_BNZ_8(_cbuf+8,rs_c,ymm1,ymm3,ymm9)
          _cbuf += rs_c;
          F32_C_BNZ_8(_cbuf,rs_c,ymm0,ymm3,ymm10)
          F32_C_BNZ_8(_cbuf+8,rs_c,ymm1,ymm3,ymm11)
          _cbuf += rs_c;
          F32_C_BNZ_8(_cbuf,rs_c,ymm0,ymm3,ymm12)
          F32_C_BNZ_8(_cbuf+8,rs_c,ymm1,ymm3,ymm13)
          _cbuf += rs_c;
          F32_C_BNZ_8(_cbuf,rs_c,ymm0,ymm3,ymm14)
          F32_C_BNZ_8(_cbuf+8,rs_c,ymm1,ymm3,ymm15)
        }
      }

      uint32_t tlsb, rounded, temp[8] = {0};
      int i;
      bfloat16* dest;

      if ( ( post_ops_attr.buf_downscale != NULL ) &&
        ( post_ops_attr.is_last_k == TRUE ) )
      {
        STORE_F32_BF16_YMM(ymm4, 0, 0, 8);
        STORE_F32_BF16_YMM(ymm5, 0, 1, 8);
        STORE_F32_BF16_YMM(ymm6, 1, 0, 8);
        STORE_F32_BF16_YMM(ymm7, 1, 1, 8);
        STORE_F32_BF16_YMM(ymm8, 2, 0, 8);
        STORE_F32_BF16_YMM(ymm9, 2, 1, 8);
        STORE_F32_BF16_YMM(ymm10, 3, 0, 8);
        STORE_F32_BF16_YMM(ymm11, 3, 1, 8);
        STORE_F32_BF16_YMM(ymm12, 4, 0, 8);
        STORE_F32_BF16_YMM(ymm13, 4, 1, 8);
        STORE_F32_BF16_YMM(ymm14, 5, 0, 8);
        STORE_F32_BF16_YMM(ymm15, 5, 1, 8);
      }
      else
      {
        _mm256_storeu_ps(cbuf, ymm4);
        _mm256_storeu_ps(cbuf + 8, ymm5);
        cbuf += rs_c;
        _mm256_storeu_ps(cbuf, ymm6);
        _mm256_storeu_ps(cbuf + 8, ymm7);
        cbuf += rs_c;
        _mm256_storeu_ps(cbuf, ymm8);
        _mm256_storeu_ps(cbuf + 8, ymm9);
        cbuf += rs_c;
        _mm256_storeu_ps(cbuf, ymm10);
        _mm256_storeu_ps(cbuf + 8, ymm11);
        cbuf += rs_c;
        _mm256_storeu_ps(cbuf, ymm12);
        _mm256_storeu_ps(cbuf + 8, ymm13);
        cbuf += rs_c;
        _mm256_storeu_ps(cbuf, ymm14);
        _mm256_storeu_ps(cbuf + 8, ymm15);
      }

      post_ops_attr.post_op_c_i += MR;
    }//mloop

    consider_edge_cases:

    // Handle edge cases in the m dimension, if they exist.
    if ( m_left )
    {
        const dim_t      i_edge = m0 - ( dim_t )m_left;

        float*  restrict cij = (float *) c + i_edge*rs_c;
        float*  restrict ai  = (float *) a + m_iter*ps_a;
        float*  restrict bj  = (float *) b;

        lpgemm_m_fringe_f32_ker_ft ker_fps[6] =
        {
          NULL,
          lpgemm_rowvar_f32f32f32of32_1x16_np,
          lpgemm_rowvar_f32f32f32of32_2x16_np,
          lpgemm_rowvar_f32f32f32of32_3x16_np,
          lpgemm_rowvar_f32f32f32of32_4x16_np,
          lpgemm_rowvar_f32f32f32of32_5x16_np
        };

        lpgemm_m_fringe_f32_ker_ft ker_fp = ker_fps[ m_left ];

        ker_fp
        (
          k0,
          ai, rs_a, cs_a,
          bj, rs_b, cs_b,
          cij, rs_c,
          alpha, beta,
          post_ops_list, post_ops_attr
        );
        return;
    }
}

LPGEMM_N_FRINGE_KERN(float,float,float,f32f32f32of32_6x8m_np)
{
    // Typecast local copies of integers in case dim_t and inc_t are a
    // different size than is expected by load instructions.
    uint64_t k_iter = (uint64_t)k0;

    uint64_t m_iter = (uint64_t)m0 / 6;
    uint64_t m_left = (uint64_t)m0 % 6;

    if ( m_iter == 0 ){    goto consider_edge_cases; }

    /*Declare the registers*/
    __m256 ymm0, ymm2, ymm3;
    __m256 ymm4, ymm6, ymm8, ymm10;
    __m256 ymm12, ymm13, ymm14, ymm15;

    /*Produce MRxNR outputs */
    for(dim_t m=0; m < m_iter; m++)
    {
      /* zero the accumulator registers */
      ZERO_ACC_YMM_4_REG(ymm4, ymm6, ymm8, ymm10);
      ZERO_ACC_YMM_4_REG(ymm12, ymm13, ymm14, ymm15);

      float *abuf, *bbuf, *cbuf, *_cbuf;

      abuf = (float *)a + m * ps_a; // Move to next MRxKC in MCxKC (where MC>=MR)
      bbuf = (float *)b;  //Same KCxNR panel is used across MCxKC block
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
        /*Load 8 elements from row0 of B*/
        ymm0 = _mm256_loadu_ps(bbuf );
        bbuf += rs_b;  //move b pointer to next row

        ymm2 = _mm256_broadcast_ss((abuf + 0*rs_a)); //broadcast c0r0
        ymm3 = _mm256_broadcast_ss((abuf + 1*rs_a)); //broadcast c0r1

        ymm4 = _mm256_fmadd_ps(ymm0, ymm2, ymm4);
        ymm6 = _mm256_fmadd_ps(ymm0, ymm3, ymm6);

        ymm2 = _mm256_broadcast_ss((abuf + 2*rs_a)); //broadcast c0r2
        ymm3 = _mm256_broadcast_ss((abuf + 3*rs_a)); //broadcast c0r3

        ymm8 = _mm256_fmadd_ps(ymm0, ymm2, ymm8);
        ymm10 = _mm256_fmadd_ps(ymm0, ymm3, ymm10);

        ymm2 = _mm256_broadcast_ss((abuf + 4*rs_a)); //broadcast c0r4
        ymm3 = _mm256_broadcast_ss((abuf + 5*rs_a)); //broadcast c0r5
        abuf += cs_a;  //move a pointer to next col

        ymm12 = _mm256_fmadd_ps(ymm0, ymm2, ymm12);
        ymm14 = _mm256_fmadd_ps(ymm0, ymm3, ymm14);
      }//kloop

      ymm0 = _mm256_broadcast_ss(&(alpha));
      ALPHA_MUL_ACC_YMM_4_REG(ymm4,ymm6,ymm8,ymm10,ymm0)
      ALPHA_MUL_ACC_YMM_4_REG(ymm12,ymm13,ymm14,ymm15,ymm0)

      if ( beta != 0.0 )
      {
        //load c and multiply with beta and
        //add to accumulator and store back
        ymm3 = _mm256_broadcast_ss(&(beta));
        if ( ( post_ops_attr.buf_downscale != NULL ) &&
          ( post_ops_attr.is_first_k == TRUE ) )
        {
          BF16_F32_C_BNZ_8(0,0,ymm0,ymm3,ymm4)
          BF16_F32_C_BNZ_8(1,0,ymm0,ymm3,ymm6)
          BF16_F32_C_BNZ_8(2,0,ymm0,ymm3,ymm8)
          BF16_F32_C_BNZ_8(3,0,ymm0,ymm3,ymm10)
          BF16_F32_C_BNZ_8(4,0,ymm0,ymm3,ymm12)
          BF16_F32_C_BNZ_8(5,0,ymm0,ymm3,ymm14)
        }
        else
        {
          _cbuf = cbuf;
          F32_C_BNZ_8(_cbuf,rs_c,ymm0,ymm3,ymm4)
          _cbuf += rs_c;
          F32_C_BNZ_8(_cbuf,rs_c,ymm0,ymm3,ymm6)
          _cbuf += rs_c;
          F32_C_BNZ_8(_cbuf,rs_c,ymm0,ymm3,ymm8)
          _cbuf += rs_c;
          F32_C_BNZ_8(_cbuf,rs_c,ymm0,ymm3,ymm10)
          _cbuf += rs_c;
          F32_C_BNZ_8(_cbuf,rs_c,ymm0,ymm3,ymm12)
          _cbuf += rs_c;
          F32_C_BNZ_8(_cbuf,rs_c,ymm0,ymm3,ymm14)
        }
      }

    uint32_t tlsb, rounded, temp[8] = {0};
    int i;
    bfloat16* dest;

    if ( ( post_ops_attr.buf_downscale != NULL ) &&
      ( post_ops_attr.is_last_k == TRUE ) )
    {
      STORE_F32_BF16_YMM(ymm4, 0, 0, 8)
      STORE_F32_BF16_YMM(ymm6, 1, 0, 8)
      STORE_F32_BF16_YMM(ymm8, 2, 0, 8)
      STORE_F32_BF16_YMM(ymm10, 3, 0, 8)
      STORE_F32_BF16_YMM(ymm12, 4, 0, 8)
      STORE_F32_BF16_YMM(ymm14, 5, 0, 8)
    }
    else
    {
      _mm256_storeu_ps(cbuf, ymm4);
      cbuf += rs_c;
      _mm256_storeu_ps(cbuf, ymm6);
      cbuf += rs_c;
      _mm256_storeu_ps(cbuf, ymm8);
      cbuf += rs_c;
      _mm256_storeu_ps(cbuf, ymm10);
      cbuf += rs_c;
      _mm256_storeu_ps(cbuf, ymm12);
      cbuf += rs_c;
      _mm256_storeu_ps(cbuf, ymm14);
    }

    post_ops_attr.post_op_c_i += MR;
    }//mloop

    consider_edge_cases:

    // Handle edge cases in the m dimension, if they exist.
    if ( m_left )
    {
        const dim_t      i_edge = m0 - ( dim_t )m_left;

        float*  restrict cij = (float *) c + i_edge*rs_c;
        float*  restrict ai  = (float *) a + m_iter*ps_a;
        float*  restrict bj  = (float *) b;

        lpgemm_m_fringe_f32_ker_ft ker_fps[6] =
        {
          NULL,
          lpgemm_rowvar_f32f32f32of32_1x8_np,
          lpgemm_rowvar_f32f32f32of32_2x8_np,
          lpgemm_rowvar_f32f32f32of32_3x8_np,
          lpgemm_rowvar_f32f32f32of32_4x8_np,
          lpgemm_rowvar_f32f32f32of32_5x8_np
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

LPGEMM_N_FRINGE_KERN(float,float,float,f32f32f32of32_6x4m_np)
{
    // Typecast local copies of integers in case dim_t and inc_t are a
    // different size than is expected by load instructions.
    uint64_t k_iter = (uint64_t)k0;

    uint64_t m_iter = (uint64_t)m0 / 6;
    uint64_t m_left = (uint64_t)m0 % 6;

    if ( m_iter == 0 ){    goto consider_edge_cases; }

    /*Declare the registers*/
    __m128 xmm0, xmm1, xmm2, xmm3;
    __m128 xmm4, xmm5, xmm6, xmm7;
    __m128 xmm8, xmm9;

    /*Produce MRxNR outputs */
    for(dim_t m=0; m < m_iter; m++)
    {
      /* zero the accumulator registers */
      ZERO_ACC_XMM_4_REG(xmm4,xmm5,xmm6,xmm7)
      ZERO_ACC_XMM_4_REG(xmm8,xmm9,xmm0,xmm1)

      float *abuf, *bbuf, *cbuf, *_cbuf;

      abuf = (float *)a + m * ps_a; // Move to next MRxKC in MCxKC (where MC>=MR)
      bbuf = (float *)b;  //Same KCxNR panel is used across MCxKC block
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
        /*Load 4 elements from row0 of B*/
        xmm0 = _mm_loadu_ps(bbuf );
        bbuf += rs_b;  //move b pointer to next row

        xmm1 = _mm_broadcast_ss((abuf + 0*rs_a)); //broadcast c0r0
        xmm2 = _mm_broadcast_ss((abuf + 1*rs_a)); //broadcast c0r0
        xmm3 = _mm_broadcast_ss((abuf + 2*rs_a)); //broadcast c0r0

        xmm4 = _mm_fmadd_ps(xmm0, xmm1, xmm4);
        xmm5 = _mm_fmadd_ps(xmm0, xmm2, xmm5);
        xmm6 = _mm_fmadd_ps(xmm0, xmm3, xmm6);

        xmm1 = _mm_broadcast_ss((abuf + 3*rs_a)); //broadcast c0r0
        xmm2 = _mm_broadcast_ss((abuf + 4*rs_a)); //broadcast c0r0
        xmm3 = _mm_broadcast_ss((abuf + 5*rs_a)); //broadcast c0r0
        abuf += cs_a;  //move a pointer to next col

        xmm7 = _mm_fmadd_ps(xmm0, xmm1, xmm7);
        xmm8 = _mm_fmadd_ps(xmm0, xmm2, xmm8);
        xmm9 = _mm_fmadd_ps(xmm0, xmm3, xmm9);
      }//kloop

      xmm0 = _mm_broadcast_ss(&(alpha));
      ALPHA_MUL_ACC_XMM_4_REG(xmm4,xmm5,xmm6,xmm7,xmm0)
      ALPHA_MUL_ACC_XMM_4_REG(xmm8,xmm9,xmm2,xmm3,xmm0)

      if ( beta != 0.0 )
      {

        xmm3 = _mm_broadcast_ss(&(beta));
        if ( ( post_ops_attr.buf_downscale != NULL ) &&
          ( post_ops_attr.is_first_k == TRUE ) )
        {
          BF16_F32_C_BNZ_4(0,0,xmm1,xmm3,xmm4)
          BF16_F32_C_BNZ_4(1,0,xmm1,xmm3,xmm5)
          BF16_F32_C_BNZ_4(2,0,xmm1,xmm3,xmm6)
          BF16_F32_C_BNZ_4(3,0,xmm1,xmm3,xmm7)
          BF16_F32_C_BNZ_4(4,0,xmm1,xmm3,xmm8)
          BF16_F32_C_BNZ_4(5,0,xmm1,xmm3,xmm9)
        }
        else
        {
          _cbuf = cbuf;
          //load c and multiply with beta and
          //add to accumulator and store back
          F32_C_BNZ_4(_cbuf,rs_c,xmm1,xmm3,xmm4)
          _cbuf += rs_c;
          F32_C_BNZ_4(_cbuf,rs_c,xmm1,xmm3,xmm5)
          _cbuf += rs_c;
          F32_C_BNZ_4(_cbuf,rs_c,xmm1,xmm3,xmm6)
          _cbuf += rs_c;
          F32_C_BNZ_4(_cbuf,rs_c,xmm1,xmm3,xmm7)
          _cbuf += rs_c;
          F32_C_BNZ_4(_cbuf,rs_c,xmm1,xmm3,xmm8)
          _cbuf += rs_c;
          F32_C_BNZ_4(_cbuf,rs_c,xmm1,xmm3,xmm9)
        }
      }

      uint32_t tlsb, rounded, temp[4] = {0};
      int i;
      bfloat16* dest;

      if ( ( post_ops_attr.buf_downscale != NULL ) &&
        ( post_ops_attr.is_last_k == TRUE ) )
      {
        STORE_F32_BF16_4XMM(xmm4, 0, 0)
        STORE_F32_BF16_4XMM(xmm5, 1, 0)
        STORE_F32_BF16_4XMM(xmm6, 2, 0)
        STORE_F32_BF16_4XMM(xmm7, 3, 0)
        STORE_F32_BF16_4XMM(xmm8, 4, 0)
        STORE_F32_BF16_4XMM(xmm9, 5, 0)
      }
      else
      {
        _mm_storeu_ps(cbuf, xmm4);
        cbuf += rs_c;
        _mm_storeu_ps(cbuf, xmm5);
        cbuf += rs_c;
        _mm_storeu_ps(cbuf, xmm6);
        cbuf += rs_c;
        _mm_storeu_ps(cbuf, xmm7);
        cbuf += rs_c;
        _mm_storeu_ps(cbuf, xmm8);
        cbuf += rs_c;
        _mm_storeu_ps(cbuf, xmm9);
      }

      post_ops_attr.post_op_c_i += MR;
    }//mloop

    consider_edge_cases:

    // Handle edge cases in the m dimension, if they exist.
    if ( m_left )
    {
        const dim_t      i_edge = m0 - ( dim_t )m_left;

        float*  restrict cij = (float *) c + i_edge*rs_c;
        float*  restrict ai  = (float *) a + m_iter*ps_a;
        float*  restrict bj  = (float *) b;

        lpgemm_m_fringe_f32_ker_ft ker_fps[6] =
        {
          NULL,
          lpgemm_rowvar_f32f32f32of32_1x4_np,
          lpgemm_rowvar_f32f32f32of32_2x4_np,
          lpgemm_rowvar_f32f32f32of32_3x4_np,
          lpgemm_rowvar_f32f32f32of32_4x4_np,
          lpgemm_rowvar_f32f32f32of32_5x4_np
        };

        lpgemm_m_fringe_f32_ker_ft ker_fp = ker_fps[ m_left ];

        ker_fp
        (
          k0,
          ai, rs_a, cs_a,
          bj, rs_b, cs_b,
          cij, rs_c,
          alpha, beta,
          post_ops_list, post_ops_attr
        );
        return;
    }
}

LPGEMM_N_FRINGE_KERN(float,float,float,f32f32f32of32_6x2m_np)
{
    // Typecast local copies of integers in case dim_t and inc_t are a
    // different size than is expected by load instructions.
    uint64_t k_iter = (uint64_t)k0;

    uint64_t m_iter = (uint64_t)m0 / 6;
    uint64_t m_left = (uint64_t)m0 % 6;

    if ( m_iter == 0 ){    goto consider_edge_cases; }

    /*Declare the registers*/
    __m128 xmm0, xmm1, xmm2, xmm3;
    __m128 xmm4, xmm5, xmm6, xmm7;
    __m128 xmm8, xmm9;

    /*Produce MRxNR outputs */
    for(dim_t m=0; m < m_iter; m++)
    {
      /* zero the accumulator registers */
      ZERO_ACC_XMM_4_REG(xmm4,xmm5,xmm6,xmm7)
      ZERO_ACC_XMM_4_REG(xmm8,xmm9,xmm0,xmm1)

      float *abuf, *bbuf, *cbuf, *_cbuf;

      abuf = (float *)a + m * ps_a; // Move to next MRxKC in MCxKC (where MC>=MR)
      bbuf = (float *)b;  //Same KCxNR panel is used across MCxKC block
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
        /*Load 2 elements from row0 of B*/
        xmm0 = ( __m128 )_mm_load_sd((const double*) bbuf );
        bbuf += rs_b;  //move b pointer to next row

        xmm1 = _mm_broadcast_ss((abuf + 0*rs_a)); //broadcast c0r0
        xmm2 = _mm_broadcast_ss((abuf + 1*rs_a)); //broadcast c0r0
        xmm3 = _mm_broadcast_ss((abuf + 2*rs_a)); //broadcast c0r0

        xmm4 = _mm_fmadd_ps(xmm0, xmm1, xmm4);
        xmm5 = _mm_fmadd_ps(xmm0, xmm2, xmm5);
        xmm6 = _mm_fmadd_ps(xmm0, xmm3, xmm6);

        xmm1 = _mm_broadcast_ss((abuf + 3*rs_a)); //broadcast c0r0
        xmm2 = _mm_broadcast_ss((abuf + 4*rs_a)); //broadcast c0r0
        xmm3 = _mm_broadcast_ss((abuf + 5*rs_a)); //broadcast c0r0
        abuf += cs_a;  //move a pointer to next col

        xmm7 = _mm_fmadd_ps(xmm0, xmm1, xmm7);
        xmm8 = _mm_fmadd_ps(xmm0, xmm2, xmm8);
        xmm9 = _mm_fmadd_ps(xmm0, xmm3, xmm9);
      }//kloop

      xmm0 = _mm_broadcast_ss(&(alpha));
      ALPHA_MUL_ACC_XMM_4_REG(xmm4,xmm5,xmm6,xmm7,xmm0)
      ALPHA_MUL_ACC_XMM_4_REG(xmm8,xmm9,xmm2,xmm3,xmm0)

      if ( beta != 0.0 )
      {
        xmm3 = _mm_broadcast_ss(&(beta));

        if ( ( post_ops_attr.buf_downscale != NULL ) &&
          ( post_ops_attr.is_first_k == TRUE ) )
        {
          BF16_F32_C_BNZ_2(0,0,xmm1,xmm3,xmm4)
          BF16_F32_C_BNZ_2(1,0,xmm1,xmm3,xmm5)
          BF16_F32_C_BNZ_2(2,0,xmm1,xmm3,xmm6)
          BF16_F32_C_BNZ_2(3,0,xmm1,xmm3,xmm7)
          BF16_F32_C_BNZ_2(4,0,xmm1,xmm3,xmm8)
          BF16_F32_C_BNZ_2(5,0,xmm1,xmm3,xmm9)
        }
        else
        {
          _cbuf = cbuf;
          //load c and multiply with beta and
          //add to accumulator and store back
          F32_C_BNZ_2(_cbuf,rs_c,xmm1,xmm3,xmm4)
          _cbuf += rs_c;
          F32_C_BNZ_2(_cbuf,rs_c,xmm1,xmm3,xmm5)
          _cbuf += rs_c;
          F32_C_BNZ_2(_cbuf,rs_c,xmm1,xmm3,xmm6)
          _cbuf += rs_c;
          F32_C_BNZ_2(_cbuf,rs_c,xmm1,xmm3,xmm7)
          _cbuf += rs_c;
          F32_C_BNZ_2(_cbuf,rs_c,xmm1,xmm3,xmm8)
          _cbuf += rs_c;
          F32_C_BNZ_2(_cbuf,rs_c,xmm1,xmm3,xmm9)
        }
      }

      uint32_t tlsb, rounded, temp[2] = {0};
      int i;
      bfloat16* dest;

      if ( ( post_ops_attr.buf_downscale != NULL ) &&
        ( post_ops_attr.is_last_k == TRUE ) )
      {
        STORE_F32_BF16_2XMM(xmm4, 0, 0)
        STORE_F32_BF16_2XMM(xmm5, 1, 0)
        STORE_F32_BF16_2XMM(xmm6, 2, 0)
        STORE_F32_BF16_2XMM(xmm7, 3, 0)
        STORE_F32_BF16_2XMM(xmm8, 4, 0)
        STORE_F32_BF16_2XMM(xmm9, 5, 0)
      }
      else
      {
        _mm_store_sd((double*)cbuf, ( __m128d )xmm4);
        cbuf += rs_c;
        _mm_store_sd((double*)cbuf, ( __m128d )xmm5);
        cbuf += rs_c;
        _mm_store_sd((double*)cbuf, ( __m128d )xmm6);
        cbuf += rs_c;
        _mm_store_sd((double*)cbuf, ( __m128d )xmm7);
        cbuf += rs_c;
        _mm_store_sd((double*)cbuf, ( __m128d )xmm8);
        cbuf += rs_c;
        _mm_store_sd((double*)cbuf, ( __m128d )xmm9);
      }

      post_ops_attr.post_op_c_i += MR;
    }//mloop

    consider_edge_cases:

    // Handle edge cases in the m dimension, if they exist.
    if ( m_left )
    {
        const dim_t      i_edge = m0 - ( dim_t )m_left;

        float*  restrict cij = (float *) c + i_edge*rs_c;
        float*  restrict ai  = (float *) a + m_iter*ps_a;
        float*  restrict bj  = (float *) b;

        lpgemm_m_fringe_f32_ker_ft ker_fps[6] =
        {
          NULL,
          lpgemm_rowvar_f32f32f32of32_1x2_np,
          lpgemm_rowvar_f32f32f32of32_2x2_np,
          lpgemm_rowvar_f32f32f32of32_3x2_np,
          lpgemm_rowvar_f32f32f32of32_4x2_np,
          lpgemm_rowvar_f32f32f32of32_5x2_np
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

LPGEMM_N_FRINGE_KERN(float,float,float,f32f32f32of32_6x1m_np)
{
    // Typecast local copies of integers in case dim_t and inc_t are a
    // different size than is expected by load instructions.
    uint64_t k_iter = (uint64_t)k0;

    uint64_t m_iter = (uint64_t)m0 / 6;
    uint64_t m_left = (uint64_t)m0 % 6;

    if ( m_iter == 0 ){    goto consider_edge_cases; }

    /*Declare the registers*/
    __m128 xmm0, xmm1, xmm2, xmm3;
    __m128 xmm4, xmm5, xmm6, xmm7;
    __m128 xmm8, xmm9;

    /*Produce MRxNR outputs */
    for(dim_t m=0; m < m_iter; m++)
    {
      /* zero the accumulator registers */
      ZERO_ACC_XMM_4_REG(xmm4,xmm5,xmm6,xmm7)
      ZERO_ACC_XMM_4_REG(xmm8,xmm9,xmm0,xmm1)

      float *abuf, *bbuf, *cbuf, *_cbuf;

      abuf = (float *)a + m * ps_a; // Move to next MRxKC in MCxKC (where MC>=MR)
      bbuf = (float *)b;  //Same KCxNR panel is used across MCxKC block
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
        /*Load 1 elements from row0 of B*/
        xmm0 = _mm_load_ss(bbuf );
        bbuf += rs_b;  //move b pointer to next row

        xmm1 = _mm_broadcast_ss((abuf + 0*rs_a)); //broadcast c0r0
        xmm2 = _mm_broadcast_ss((abuf + 1*rs_a)); //broadcast c0r0
        xmm3 = _mm_broadcast_ss((abuf + 2*rs_a)); //broadcast c0r0

        xmm4 = _mm_fmadd_ps(xmm0, xmm1, xmm4);
        xmm5 = _mm_fmadd_ps(xmm0, xmm2, xmm5);
        xmm6 = _mm_fmadd_ps(xmm0, xmm3, xmm6);

        xmm1 = _mm_broadcast_ss((abuf + 3*rs_a)); //broadcast c0r0
        xmm2 = _mm_broadcast_ss((abuf + 4*rs_a)); //broadcast c0r0
        xmm3 = _mm_broadcast_ss((abuf + 5*rs_a)); //broadcast c0r0
        abuf += cs_a;  //move a pointer to next col

        xmm7 = _mm_fmadd_ps(xmm0, xmm1, xmm7);
        xmm8 = _mm_fmadd_ps(xmm0, xmm2, xmm8);
        xmm9 = _mm_fmadd_ps(xmm0, xmm3, xmm9);
      }//kloop

      xmm0 = _mm_broadcast_ss(&(alpha));
      ALPHA_MUL_ACC_XMM_4_REG(xmm4,xmm5,xmm6,xmm7,xmm0)
      ALPHA_MUL_ACC_XMM_4_REG(xmm8,xmm9,xmm2,xmm3,xmm0)

      if ( beta != 0.0 )
      {
        xmm3 = _mm_broadcast_ss(&(beta));

        if ( ( post_ops_attr.buf_downscale != NULL ) &&
          ( post_ops_attr.is_first_k == TRUE ) )
        {
          BF16_F32_C_BNZ_1(0,0,xmm1,xmm3,xmm4)
          BF16_F32_C_BNZ_1(1,0,xmm1,xmm3,xmm5)
          BF16_F32_C_BNZ_1(2,0,xmm1,xmm3,xmm6)
          BF16_F32_C_BNZ_1(3,0,xmm1,xmm3,xmm7)
          BF16_F32_C_BNZ_1(4,0,xmm1,xmm3,xmm8)
          BF16_F32_C_BNZ_1(5,0,xmm1,xmm3,xmm9)
        }
        else
        {
          _cbuf = cbuf;
          //load c and multiply with beta and
          //add to accumulator and store back
          F32_C_BNZ_1(_cbuf,rs_c,xmm1,xmm3,xmm4)
          _cbuf += rs_c;
          F32_C_BNZ_1(_cbuf,rs_c,xmm1,xmm3,xmm5)
          _cbuf += rs_c;
          F32_C_BNZ_1(_cbuf,rs_c,xmm1,xmm3,xmm6)
          _cbuf += rs_c;
          F32_C_BNZ_1(_cbuf,rs_c,xmm1,xmm3,xmm7)
          _cbuf += rs_c;
          F32_C_BNZ_1(_cbuf,rs_c,xmm1,xmm3,xmm8)
          _cbuf += rs_c;
          F32_C_BNZ_1(_cbuf,rs_c,xmm1,xmm3,xmm9)
        }
      }

      uint32_t tlsb, rounded, temp[1] = {0};
      int i;
      bfloat16* dest;

      if ( ( post_ops_attr.buf_downscale != NULL ) &&
			 ( post_ops_attr.is_last_k == TRUE ) )
		  {
        STORE_F32_BF16_1XMM(xmm4, 0, 0)
        STORE_F32_BF16_1XMM(xmm5, 1, 0)
        STORE_F32_BF16_1XMM(xmm6, 2, 0)
        STORE_F32_BF16_1XMM(xmm7, 3, 0)
        STORE_F32_BF16_1XMM(xmm8, 4, 0)
        STORE_F32_BF16_1XMM(xmm9, 5, 0)
      }
      else
      {
        _mm_store_ss(cbuf, xmm4);
        cbuf += rs_c;
        _mm_store_ss(cbuf, xmm5);
        cbuf += rs_c;
        _mm_store_ss(cbuf, xmm6);
        cbuf += rs_c;
        _mm_store_ss(cbuf, xmm7);
        cbuf += rs_c;
        _mm_store_ss(cbuf, xmm8);
        cbuf += rs_c;
        _mm_store_ss(cbuf, xmm9);
      }

      post_ops_attr.post_op_c_i += MR;
    }//mloop

    consider_edge_cases:

    // Handle edge cases in the m dimension, if they exist.
    if ( m_left )
    {
        const dim_t      i_edge = m0 - ( dim_t )m_left;

        float*  restrict cij = (float *) c + i_edge*rs_c;
        float*  restrict ai  = (float *) a + m_iter*ps_a;
        float*  restrict bj  = (float *) b;

        lpgemm_m_fringe_f32_ker_ft ker_fps[6] =
        {
          NULL,
          lpgemm_rowvar_f32f32f32of32_1x1,
          lpgemm_rowvar_f32f32f32of32_2x1,
          lpgemm_rowvar_f32f32f32of32_3x1,
          lpgemm_rowvar_f32f32f32of32_4x1,
          lpgemm_rowvar_f32f32f32of32_5x1
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

/* Mask elements to specify how many elements to be loaded from C buffer */
static const int32_t mask[8][8] = { {0, 0, 0, 0, 0, 0, 0, 0}, //load no values, not used currently
                                    {-1, 0, 0, 0, 0, 0, 0, 0}, // load 1 value from memory
                                    {-1, -1, 0, 0, 0, 0, 0, 0}, // load 2 values from memory
                                    {-1, -1, -1, 0, 0, 0, 0, 0},
                                    {-1, -1, -1, -1, 0, 0, 0, 0},
                                    {-1, -1, -1, -1, -1, 0, 0, 0},
                                    {-1, -1, -1, -1, -1, -1, 0, 0},
                                    {-1, -1, -1, -1, -1, -1, -1, 0},
                                  };

LPGEMM_N_LT_NR0_FRINGE_KERN(float,float,float,f32f32f32of32_6xlt8m_np)
{
    // Typecast local copies of integers in case dim_t and inc_t are a
    // different size than is expected by load instructions.
    uint64_t k_iter = (uint64_t)k0;

    uint64_t m_iter = (uint64_t)m0 / 6;
    uint64_t m_left = (uint64_t)m0 % 6;

    if ( m_iter == 0 ){    goto consider_edge_cases; }

    /*Declare the registers*/
    __m256 ymm0, ymm2, ymm3;
    __m256 ymm4, ymm6, ymm8, ymm10;
    __m256 ymm12, ymm14;

    __m256i ymm_mask = _mm256_loadu_si256((__m256i*)mask[n0_rem]);
    /*Produce MRxNR outputs */
    for(dim_t m=0; m < m_iter; m++)
    {
      /* zero the accumulator registers */
      ZERO_ACC_YMM_4_REG(ymm4, ymm6, ymm8, ymm10);
      ZERO_ACC_YMM_2_REG(ymm12, ymm14);

      float *abuf, *bbuf, *cbuf, *_cbuf;

      abuf = (float *)a + m * ps_a; // Move to next MRxKC in MCxKC (where MC>=MR)
      bbuf = (float *)b;  //Same KCxNR panel is used across MCxKC block
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
        /*Load 8 elements from row0 of B*/
        ymm0 = _mm256_maskload_ps (bbuf, ymm_mask);
        bbuf += rs_b;  //move b pointer to next row

        ymm2 = _mm256_broadcast_ss((abuf + 0*rs_a)); //broadcast c0r0
        ymm3 = _mm256_broadcast_ss((abuf + 1*rs_a)); //broadcast c0r1

        ymm4 = _mm256_fmadd_ps(ymm0, ymm2, ymm4);
        ymm6 = _mm256_fmadd_ps(ymm0, ymm3, ymm6);

        ymm2 = _mm256_broadcast_ss((abuf + 2*rs_a)); //broadcast c0r2
        ymm3 = _mm256_broadcast_ss((abuf + 3*rs_a)); //broadcast c0r3

        ymm8 = _mm256_fmadd_ps(ymm0, ymm2, ymm8);
        ymm10 = _mm256_fmadd_ps(ymm0, ymm3, ymm10);

        ymm2 = _mm256_broadcast_ss((abuf + 4*rs_a)); //broadcast c0r4
        ymm3 = _mm256_broadcast_ss((abuf + 5*rs_a)); //broadcast c0r5
        abuf += cs_a;  //move a pointer to next col

        ymm12 = _mm256_fmadd_ps(ymm0, ymm2, ymm12);
        ymm14 = _mm256_fmadd_ps(ymm0, ymm3, ymm14);
      }//kloop

      ymm0 = _mm256_broadcast_ss(&(alpha));
      ALPHA_MUL_ACC_YMM_4_REG(ymm4,ymm6,ymm8,ymm10,ymm0)
      ALPHA_MUL_ACC_YMM_2_REG(ymm12,ymm14,ymm0)

      if ( beta != 0.0 )
      {
        //load c and multiply with beta and
        //add to accumulator and store back
        ymm3 = _mm256_broadcast_ss(&(beta));
        if ( ( post_ops_attr.buf_downscale != NULL ) &&
          ( post_ops_attr.is_first_k == TRUE ) )
        {
          BF16_F32_C_BNZ_8_MASK(0,0,ymm0,ymm3,ymm4, ymm_mask)
          BF16_F32_C_BNZ_8_MASK(1,0,ymm0,ymm3,ymm6, ymm_mask)
          BF16_F32_C_BNZ_8_MASK(2,0,ymm0,ymm3,ymm8, ymm_mask)
          BF16_F32_C_BNZ_8_MASK(3,0,ymm0,ymm3,ymm10, ymm_mask)
          BF16_F32_C_BNZ_8_MASK(4,0,ymm0,ymm3,ymm12, ymm_mask)
          BF16_F32_C_BNZ_8_MASK(5,0,ymm0,ymm3,ymm14, ymm_mask)
        }
        else
        {
          _cbuf = cbuf;
          F32_C_BNZ_8_MASK(_cbuf,rs_c,ymm0,ymm3,ymm4, ymm_mask)
          _cbuf += rs_c;
          F32_C_BNZ_8_MASK(_cbuf,rs_c,ymm0,ymm3,ymm6, ymm_mask)
          _cbuf += rs_c;
          F32_C_BNZ_8_MASK(_cbuf,rs_c,ymm0,ymm3,ymm8, ymm_mask)
          _cbuf += rs_c;
          F32_C_BNZ_8_MASK(_cbuf,rs_c,ymm0,ymm3,ymm10, ymm_mask)
          _cbuf += rs_c;
          F32_C_BNZ_8_MASK(_cbuf,rs_c,ymm0,ymm3,ymm12, ymm_mask)
          _cbuf += rs_c;
          F32_C_BNZ_8_MASK(_cbuf,rs_c,ymm0,ymm3,ymm14, ymm_mask)
        }
      }

      uint32_t tlsb, rounded, temp[8] = {0};
      int i;
      bfloat16* dest;

      if ( ( post_ops_attr.buf_downscale != NULL ) &&
        ( post_ops_attr.is_last_k == TRUE ) )
      {
        STORE_F32_BF16_YMM(ymm4, 0, 0, n0_rem)
        STORE_F32_BF16_YMM(ymm6, 1, 0, n0_rem)
        STORE_F32_BF16_YMM(ymm8, 2, 0, n0_rem)
        STORE_F32_BF16_YMM(ymm10, 3, 0, n0_rem)
        STORE_F32_BF16_YMM(ymm12, 4, 0, n0_rem)
        STORE_F32_BF16_YMM(ymm14, 5, 0, n0_rem)
      }
      else
      {
        _mm256_maskstore_ps(cbuf, ymm_mask, ymm4);
        cbuf += rs_c;
        _mm256_maskstore_ps(cbuf, ymm_mask, ymm6);
        cbuf += rs_c;
        _mm256_maskstore_ps(cbuf, ymm_mask, ymm8);
        cbuf += rs_c;
        _mm256_maskstore_ps(cbuf, ymm_mask, ymm10);
        cbuf += rs_c;
        _mm256_maskstore_ps(cbuf, ymm_mask, ymm12);
        cbuf += rs_c;
        _mm256_maskstore_ps(cbuf, ymm_mask, ymm14);
      }

      post_ops_attr.post_op_c_i += MR;
    }//mloop

    consider_edge_cases:

    // Handle edge cases in the m dimension, if they exist.
    if ( m_left )
    {
        const dim_t      i_edge = m0 - ( dim_t )m_left;

        float*  restrict cij = (float *) c + i_edge*rs_c;
        float*  restrict ai  = (float *) a + m_iter*ps_a;
        float*  restrict bj  = (float *) b;

        lpgemm_mn_fringe_f32_mask_ker_ft ker_fps[6] =
        {
          NULL,
          lpgemm_rowvar_f32f32f32of32_1xlt8_np,
          lpgemm_rowvar_f32f32f32of32_2xlt8_np,
          lpgemm_rowvar_f32f32f32of32_3xlt8_np,
          lpgemm_rowvar_f32f32f32of32_4xlt8_np,
          lpgemm_rowvar_f32f32f32of32_5xlt8_np
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
