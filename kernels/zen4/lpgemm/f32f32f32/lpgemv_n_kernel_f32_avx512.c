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
#include "immintrin.h"
#include "xmmintrin.h"
#include "blis.h"

#ifdef BLIS_ADDON_LPGEMM

#include "lpgemm_kernel_macros_f32.h"

#define LPGEMV_N_KERNEL_4_LOADS(zmm0, zmm1, zmm2, zmm3, paddr, stride) \
  zmm0 = _mm512_loadu_ps(paddr); \
  zmm1 = _mm512_loadu_ps(paddr + stride); \
  zmm2 = _mm512_loadu_ps(paddr + 2 * stride); \
  zmm3 = _mm512_loadu_ps(paddr + 3 * stride);

#define LPGEMV_N_KERNEL_4_MASKLOADS(zmm0, zmm1, zmm2, zmm3, zmm7, k1, paddr, stride) \
  zmm0 = _mm512_mask_loadu_ps(zmm7, k1, paddr);                                      \
  zmm1 = _mm512_mask_loadu_ps(zmm7, k1, paddr + stride);                             \
  zmm2 = _mm512_mask_loadu_ps(zmm7, k1, paddr + 2 * stride);                         \
  zmm3 = _mm512_mask_loadu_ps(zmm7, k1, paddr + 3 * stride);

#define LPGEMV_N_KERNEL_4_FMA(zmm8, zmm9, zmm10, zmm11, zmm6, zmm0, zmm1, zmm2, zmm3) \
  zmm8 = _mm512_fmadd_ps(zmm0, zmm6, zmm8); \
  zmm9 = _mm512_fmadd_ps(zmm1, zmm6, zmm9); \
  zmm10 = _mm512_fmadd_ps(zmm2, zmm6, zmm10); \
  zmm11 = _mm512_fmadd_ps(zmm3, zmm6, zmm11);

#define LPGEMV_ZMM2XMM(zmm0, zmm1, zmm2, zmm3, ymm0, ymm1, ymm2, ymm3, xmm0) \
  ymm0 = _mm256_add_ps(_mm512_extractf32x8_ps(zmm0, 0x0), \
                       _mm512_extractf32x8_ps(zmm0, 0x1)); \
  ymm1 = _mm256_add_ps(_mm512_extractf32x8_ps(zmm1, 0x0), \
                       _mm512_extractf32x8_ps(zmm1, 0x1)); \
  ymm0 = _mm256_hadd_ps(ymm0, ymm1); \
  ymm2 = _mm256_add_ps(_mm512_extractf32x8_ps(zmm2, 0x0), \
                       _mm512_extractf32x8_ps(zmm2, 0x1)); \
  ymm3 = _mm256_add_ps(_mm512_extractf32x8_ps(zmm3, 0x0), \
                       _mm512_extractf32x8_ps(zmm3, 0x1)); \
  ymm1 = _mm256_hadd_ps(ymm2, ymm3); \
  ymm0 = _mm256_hadd_ps(ymm0, ymm1); \
  xmm0 = _mm_add_ps(_mm256_extractf128_ps(ymm0, 0), _mm256_extractf128_ps(ymm0,1));

// When n=1 is load 16x1 from B and load MRx16 from A and perform dot product
//  to produce C output of MRX1. The vectorization is done in k loop and
//  the horizontal reduction done to produce one output from each
//  accumulator register
LPGEMV_N_EQ1_KERN( float, float, float, f32f32f32of32 )
{
  static void *post_ops_labels[] =
      {
          &&POST_OPS_6x64F_DISABLE,
          &&POST_OPS_BIAS_6x64F,
          &&POST_OPS_RELU_6x64F,
          &&POST_OPS_RELU_SCALE_6x64F,
          &&POST_OPS_GELU_TANH_6x64F,
          &&POST_OPS_GELU_ERF_6x64F,
          &&POST_OPS_CLIP_6x64F,
          NULL, // Virtual node for downscale, else segfault
          &&POST_OPS_MATRIX_ADD_6x64F,
          &&POST_OPS_SWISH_6x64F,
          &&POST_OPS_MATRIX_MUL_6x64F
      };

  // Strides are updated based on matrix packing/reordering.
  const float *a_use = NULL;
  const float *b_use = NULL;
  float *c_use = NULL;

  lpgemm_post_op_attr post_ops_attr = *(post_op_attr);

  for (dim_t mr = 0; mr < m0; mr += MR)
  {
    dim_t mr0 = bli_min((m0 - mr), MR);
    dim_t k_iter = k/16;
    dim_t k_rem = k & 0xF;

    //Create load mask for k fringe
    __mmask16 k1 = 0xFFFF;
    if (k_rem)
    {
      k1 = (0xFFFF >> (16 - k_rem));
    }

    // Create store mask for C for mr fringe
    __mmask16 k2 = 0xFFFF;
    if (mr0 < MR)
    {
      k2 = (0xFFFF >> (MR - mr0));
    }

    __m512 zmm0, zmm1, zmm2, zmm3, zmm4, zmm5, zmm6, zmm7;
    __m512 zmm8, zmm9, zmm10, zmm11, zmm12, zmm13, zmm14;
    __m512 zmm15, zmm16, zmm17, zmm18, zmm19, zmm20, zmm21;
    __m512 zmm22, zmm23, zmm24, zmm25, zmm26, zmm27, zmm28;
    __m512 zmm29, zmm30, zmm31;

    __m256 ymm0,ymm1,ymm2,ymm3,ymm4,ymm5,ymm6;
    __m128 xmm0, xmm1, xmm2, xmm3;

    ZERO_ACC_ZMM_4_REG(zmm0, zmm1, zmm2, zmm3);
    ZERO_ACC_ZMM_4_REG(zmm4, zmm5, zmm6, zmm7);
    /* zero the accumulator registers */
    ZERO_ACC_ZMM_4_REG(zmm8, zmm9, zmm10, zmm11);
    ZERO_ACC_ZMM_4_REG(zmm12, zmm13, zmm14, zmm15);
    ZERO_ACC_ZMM_4_REG(zmm16, zmm17, zmm18, zmm19);
    ZERO_ACC_ZMM_4_REG(zmm20, zmm21, zmm22, zmm23);
    ZERO_ACC_ZMM_4_REG(zmm24, zmm25, zmm26, zmm27);
    ZERO_ACC_ZMM_4_REG(zmm28, zmm29, zmm30, zmm31);
    ZERO_ACC_XMM_4_REG (xmm0,xmm1,xmm2,xmm3)

    //update pointers
    a_use = a + mr * rs_a;
    b_use = b;
    c_use = c + mr * rs_c;

    //prefetch C
    _mm_prefetch(c_use, _MM_HINT_T0);
    _mm_prefetch(b_use, _MM_HINT_T0);

    //Check for MR whether to process main kernel or mfringe kernel
    if (mr0 == MR)
    {
        //Dot product kernel
        for (dim_t k = 0; k < k_iter; k++)
        {
          zmm6 = _mm512_loadu_ps(b_use); // Load 0-15 in b[k+0 - k+15]
          b_use += 16;   // move b pointer to next 16 elements

          //Load 4x16 elements from row0-row3 of A
          LPGEMV_N_KERNEL_4_LOADS(zmm0, zmm1, zmm2, zmm3, a_use, rs_a)
          a_use += (4 * rs_a);

          // Load 4x16 elements from row3-row7 of A
          LPGEMV_N_KERNEL_4_LOADS(zmm24, zmm25, zmm26, zmm27, a_use, rs_a)
          a_use += (4 * rs_a);

          LPGEMV_N_KERNEL_4_FMA(zmm8, zmm9, zmm10, zmm11, zmm6, zmm0, zmm1, zmm2, zmm3)

          // Load 4x16 elements from row8-row11 of A
          LPGEMV_N_KERNEL_4_LOADS(zmm28, zmm29, zmm30, zmm31, a_use, rs_a)
          a_use += (4 * rs_a);

          // Load 4x16 elements from row12-row15 of A
          LPGEMV_N_KERNEL_4_LOADS(zmm0, zmm1, zmm2, zmm3, a_use, rs_a)
          a_use -= (12 * rs_a); //Update aptr back to move horizontally

          LPGEMV_N_KERNEL_4_FMA(zmm12, zmm13, zmm14, zmm15, zmm6, zmm24, zmm25, zmm26, zmm27)
          LPGEMV_N_KERNEL_4_FMA(zmm16, zmm17, zmm18, zmm19, zmm6, zmm28, zmm29, zmm30, zmm31)
          LPGEMV_N_KERNEL_4_FMA(zmm20, zmm21, zmm22, zmm23, zmm6, zmm0, zmm1, zmm2, zmm3)
          a_use += 16;
        }// kloop

        if(k_rem)
        {
          zmm6 = _mm512_mask_loadu_ps(zmm7, k1, b_use); // Load 0-15 in b[k+0 - k+15]

          // Load 4x16 elements from row0-row3 of A
          LPGEMV_N_KERNEL_4_MASKLOADS(zmm0, zmm1, zmm2, zmm3, zmm7, k1, a_use, rs_a)
          a_use += (4 * rs_a);

          LPGEMV_N_KERNEL_4_MASKLOADS(zmm24, zmm25, zmm26, zmm27, zmm7, k1, a_use, rs_a)
          a_use += (4 * rs_a);

          LPGEMV_N_KERNEL_4_FMA(zmm8, zmm9, zmm10, zmm11, zmm6, zmm0, zmm1, zmm2, zmm3)

          LPGEMV_N_KERNEL_4_MASKLOADS(zmm28, zmm29, zmm30, zmm31, zmm7, k1, a_use, rs_a)
          a_use += (4 * rs_a);

          LPGEMV_N_KERNEL_4_MASKLOADS(zmm0, zmm1, zmm2, zmm3, zmm7, k1, a_use, rs_a)

          LPGEMV_N_KERNEL_4_FMA(zmm12, zmm13, zmm14, zmm15, zmm6, zmm24, zmm25, zmm26, zmm27)
          LPGEMV_N_KERNEL_4_FMA(zmm16, zmm17, zmm18, zmm19, zmm6, zmm28, zmm29, zmm30, zmm31)
          LPGEMV_N_KERNEL_4_FMA(zmm20, zmm21, zmm22, zmm23, zmm6, zmm0, zmm1, zmm2, zmm3)
        }// kloop

        //Add the registers horizantally to get one
        LPGEMV_ZMM2XMM(zmm8, zmm9, zmm10, zmm11, ymm0, ymm1, ymm2, ymm3, xmm0)
        LPGEMV_ZMM2XMM(zmm12, zmm13, zmm14, zmm15, ymm4, ymm1, ymm2, ymm3, xmm1)
        LPGEMV_ZMM2XMM(zmm16, zmm17, zmm18, zmm19, ymm5, ymm1, ymm2, ymm3, xmm2)
        LPGEMV_ZMM2XMM(zmm20, zmm21, zmm22, zmm23, ymm6, ymm1, ymm2, ymm3, xmm3)

        //compose outputs into one zmm to perform post-ops
        zmm8 = _mm512_insertf32x4(zmm8, xmm0, 0);
        zmm8 = _mm512_insertf32x4(zmm8, xmm1, 1);
        zmm8 = _mm512_insertf32x4(zmm8, xmm2, 2);
        zmm8 = _mm512_insertf32x4(zmm8, xmm3, 3);
    }else
    {
      //Handle fringe cases when mr0 < MR
      const float *a_use_fringe = a_use;
      dim_t mr0_use = mr0;
      dim_t regidx = 0;

      // Dot product for mfringe 8
      if (mr0_use >= 8)
      {
        // Dot product kernel for mr0 == 8
        for (dim_t k = 0; k < k_iter; k++)
        {
          zmm6 = _mm512_loadu_ps(b_use); // Load 0-15 in b[k+0 - k+15]
          b_use += 16;                   // move b pointer to next 16 elements

          // Load 4x16 elements from row0-row3 of A
          LPGEMV_N_KERNEL_4_LOADS(zmm0, zmm1, zmm2, zmm3, a_use, rs_a)
          a_use += (4 * rs_a);

          // Load 4x16 elements from row3-row7 of A
          LPGEMV_N_KERNEL_4_LOADS(zmm24, zmm25, zmm26, zmm27, a_use, rs_a)
          a_use -= (4 * rs_a);

          //Perform FMA on two 4x16 block of A with 16x1
          LPGEMV_N_KERNEL_4_FMA(zmm8, zmm9, zmm10, zmm11, zmm6, zmm0, zmm1, zmm2, zmm3)
          LPGEMV_N_KERNEL_4_FMA(zmm12, zmm13, zmm14, zmm15, zmm6, zmm24, zmm25, zmm26, zmm27)
          a_use += 16;
        }

        if (k_rem)
        {
          zmm6 = _mm512_mask_loadu_ps(zmm7, k1, b_use); // Load 0-15 in b[k+0 - k+15]

          // Load 4x16 elements from row0-row3 of A
          LPGEMV_N_KERNEL_4_MASKLOADS(zmm0, zmm1, zmm2, zmm3, zmm7, k1, a_use, rs_a)
          a_use += (4 * rs_a);
          LPGEMV_N_KERNEL_4_MASKLOADS(zmm24, zmm25, zmm26, zmm27, zmm7, k1, a_use, rs_a)
          LPGEMV_N_KERNEL_4_FMA(zmm8, zmm9, zmm10, zmm11, zmm6, zmm0, zmm1, zmm2, zmm3)
          LPGEMV_N_KERNEL_4_FMA(zmm12, zmm13, zmm14, zmm15, zmm6, zmm24, zmm25, zmm26, zmm27)
        }

        //update pointers
        mr0_use -= 8;
        a_use = a_use_fringe + 8 * rs_a;
        a_use_fringe = a_use;
        b_use = b;

        //Horizontal add 8 zmm registers and get output into 2 xmm registers
        LPGEMV_ZMM2XMM(zmm8, zmm9, zmm10, zmm11, ymm0, ymm1, ymm2, ymm3, xmm0)
        LPGEMV_ZMM2XMM(zmm12, zmm13, zmm14, zmm15, ymm4, ymm1, ymm2, ymm3, xmm1)

        //insert xmm outputs into final output zmm8 reg
        zmm8 = _mm512_insertf32x4(zmm8, xmm0, 0);
        zmm8 = _mm512_insertf32x4(zmm8, xmm1, 1);
        regidx = 2;
      }

      // Dot product for mfringe 4
      if (mr0_use >= 4)
      {
        // Dot product kernel for mr0 == 8
        for (dim_t k = 0; k < k_iter; k++)
        {
          zmm6 = _mm512_loadu_ps(b_use); // Load 0-15 in b[k+0 - k+15]
          b_use += 16;                   // move b pointer to next 16 elements
          // Load 4x16 elements from row0-row3 of A
          LPGEMV_N_KERNEL_4_LOADS(zmm0, zmm1, zmm2, zmm3, a_use, rs_a)
          // Perform FMA on 4x16 block of A with 16x1
          LPGEMV_N_KERNEL_4_FMA(zmm16, zmm17, zmm18, zmm19, zmm6, zmm0, zmm1, zmm2, zmm3)
          a_use += 16;
        }

        if (k_rem)
        {
          zmm6 = _mm512_mask_loadu_ps(zmm7, k1, b_use); // Load 0-15 in b[k+0 - k+15]
          // Load 4x16 elements from row0-row3 of A
          LPGEMV_N_KERNEL_4_MASKLOADS(zmm0, zmm1, zmm2, zmm3, zmm7, k1, a_use, rs_a)
          LPGEMV_N_KERNEL_4_FMA(zmm16, zmm17, zmm18, zmm19, zmm6, zmm0, zmm1, zmm2, zmm3)
        }

        //update pointers
        mr0_use -= 4;
        a_use = a_use_fringe + 4 * rs_a;
        a_use_fringe = a_use;
        b_use = b;

        //Horizontal add 4 zmm reg and get the output into one xmm
        LPGEMV_ZMM2XMM(zmm16, zmm17, zmm18, zmm19, ymm5, ymm1, ymm2, ymm3, xmm2)

        //insert xmm outputs into final output zmm8 reg based on regidx
        if(regidx == 0) zmm8 = _mm512_insertf32x4(zmm8, xmm2, 0);
        else zmm8 = _mm512_insertf32x4(zmm8, xmm2, 2);
        regidx++;
      }

      // Dot product for  <= 3
      if (mr0_use)
      {
        // Dot product for m = 2
        if (mr0_use >= 2)
        {
          for (dim_t k = 0; k < k_iter; k++)
          {
              zmm6 = _mm512_loadu_ps(b_use); // Load 0-15 in b[k+0 - k+15]
              // Load 2x16 elements from row0-row1 of A
              zmm0 = _mm512_loadu_ps(a_use);
              zmm1 = _mm512_loadu_ps(a_use + rs_a);
              zmm20 = _mm512_fmadd_ps(zmm0, zmm6, zmm20);
              zmm21 = _mm512_fmadd_ps(zmm1, zmm6, zmm21);
              b_use += 16; // move b pointer to next 16 elements
              a_use += 16;
          }
          if (k_rem)
          {
            zmm6 = _mm512_mask_loadu_ps(zmm7, k1, b_use); // Load 0-15 in b[k+0 - k+15]
            zmm0 = _mm512_mask_loadu_ps(zmm7, k1, a_use); // Load 0-15 in b[k+0 - k+15]
            zmm1 = _mm512_mask_loadu_ps(zmm7, k1, a_use + rs_a); // Load 0-15 in b[k+0 - k+15]
            zmm20 = _mm512_fmadd_ps(zmm0, zmm6, zmm20);
            zmm21 = _mm512_fmadd_ps(zmm1, zmm6, zmm21);
          }
          mr0_use -= 2;
          a_use = a_use_fringe + 2 * rs_a;
          a_use_fringe = a_use;
          b_use = b;
        }

        // Dot product for m = 2
        if (mr0_use == 1)
        {
          for (dim_t k = 0; k < k_iter; k++)
          {
            zmm6 = _mm512_loadu_ps(b_use); // Load 0-15 in b[k+0 - k+15]
            zmm0 = _mm512_loadu_ps(a_use);
            zmm22 = _mm512_fmadd_ps(zmm0, zmm6, zmm22);
            b_use += 16; // move b pointer to next 16 elements
            a_use += 16;
          }

          if (k_rem)
          {
            zmm6 = _mm512_mask_loadu_ps(zmm7, k1, b_use);
            zmm0 = _mm512_mask_loadu_ps(zmm7, k1, a_use);
            zmm22 = _mm512_fmadd_ps(zmm0, zmm6, zmm22);
          }
          // When only fringe 1, update the registers to store in order
          if (!(mr0 & 0x2))  zmm20 = zmm22;
        }

        // Horizontal add 4 zmm reg and get the output into one xmm
        LPGEMV_ZMM2XMM(zmm20, zmm21, zmm22, zmm23, ymm6, ymm1, ymm2, ymm3, xmm3)

        // insert xmm outputs into final output zmm8 reg based on regidx
        if (regidx == 0) zmm8 = _mm512_insertf32x4(zmm8, xmm3, 0);
        else if(regidx == 1)  zmm8 = _mm512_insertf32x4(zmm8, xmm3, 1);
        else if (regidx == 2) zmm8 = _mm512_insertf32x4(zmm8, xmm3, 2);
        else zmm8 = _mm512_insertf32x4(zmm8, xmm3, 3);
      }
    }

    //Scale accumulated output with alpha
    zmm0 = _mm512_set1_ps(alpha);
    zmm8 = _mm512_mul_ps(zmm0, zmm8);

    if (beta != 0)
    {
      const float *_cbuf = c_use;

      //C = beta*C + alpha*A*B
      zmm3 = _mm512_set1_ps(beta);
      if (rs_c == 1)
      {
        zmm0 = _mm512_maskz_loadu_ps(k2, _cbuf);
      }else
      {
        //load C into zmm0
        float ctemp[16];
        for(dim_t i = 0; i < mr0; i++)
        {
          ctemp[i] = _cbuf[i * rs_c];
        }
        zmm0 = _mm512_maskz_loadu_ps(k2, ctemp);
      }
      zmm8 = _mm512_fmadd_ps(zmm0, zmm3, zmm8);
    }

    // Post Ops
    post_ops_attr.is_last_k = TRUE;
    lpgemm_post_op *post_ops_list_temp = post_op;
    POST_OP_LABEL_LASTK_SAFE_JUMP

  POST_OPS_BIAS_6x64F:
  {
      // If original output was columns major, then by the time
      // kernel sees it, the matrix would be accessed as if it were
      // transposed. Due to this the bias array will be accessed by
      // the ic index, and each bias element corresponds to an
      // entire row of the transposed output array, instead of an
      // entire column.
    zmm9 = _mm512_set1_ps(*((float *)post_ops_list_temp->op_args1));
    zmm8 = _mm512_add_ps(zmm9, zmm8);
    POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
  }
  POST_OPS_RELU_6x64F:
  {
    zmm1 = _mm512_setzero_ps();

    // c[0,0-15]
    zmm8 = _mm512_max_ps(zmm1, zmm8);

    POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
  }
  POST_OPS_RELU_SCALE_6x64F:
  {
    zmm1 = _mm512_setzero_ps();
    zmm2 =
        _mm512_set1_ps(*((float *)post_ops_list_temp->op_args2));

    __mmask16 relu_cmp_mask;

    // c[0, 0-15]
    RELU_SCALE_OP_F32S_AVX512(zmm8)

    POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
  }
  POST_OPS_GELU_TANH_6x64F:
  {
    __m512i zmm6;
    // c[0, 0-15]
    GELU_TANH_F32S_AVX512(zmm8, zmm0, zmm1, zmm2, zmm3, zmm4, zmm5, zmm6)

    POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
  }
  POST_OPS_GELU_ERF_6x64F:
  {
    // c[0, 0-15]
    GELU_ERF_F32S_AVX512(zmm8, zmm0, zmm1, zmm2)
    POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
  }
  POST_OPS_CLIP_6x64F:
  {
    zmm0 = _mm512_set1_ps(*(float *)post_ops_list_temp->op_args2);
    zmm1 = _mm512_set1_ps(*(float *)post_ops_list_temp->op_args3);

    // c[0, 0-15]
    CLIP_F32S_AVX512(zmm8, zmm0, zmm1)

    POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
  }
  POST_OPS_MATRIX_ADD_6x64F:
  {
    float *matptr = (float *)post_ops_list_temp->op_args1;
    zmm0 = _mm512_maskz_loadu_ps(k2, (matptr + post_ops_attr.post_op_c_i));
    zmm8 = _mm512_add_ps(zmm8, zmm0);
    POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
  }
   POST_OPS_MATRIX_MUL_6x64F:
  {
    float *matptr = (float *)post_ops_list_temp->op_args1;
    zmm0 = _mm512_maskz_loadu_ps(k2, (matptr + post_ops_attr.post_op_c_i));
    zmm8 = _mm512_mul_ps(zmm8, zmm0);
    POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
  }
  POST_OPS_SWISH_6x64F:
  {
    zmm7 =
        _mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args2 ) );
    __m512i ex_out;

    // c[0, 0-15]
    SWISH_F32_AVX512_DEF(zmm8, zmm7, zmm0, zmm1, zmm2, zmm3, zmm4, ex_out);
  }
  POST_OPS_6x64F_DISABLE:
  {
    if (rs_c == 1)
    {
      _mm512_mask_storeu_ps(c_use, k2, zmm8);
    }
    else
    {
      // Store ZMM8 into ctemp buffer and store back
      // element by element into output buffer at strides
      float ctemp[16];
      _mm512_mask_storeu_ps(ctemp, k2, zmm8);
      for (dim_t i = 0; i < mr0; i++)
      {
        c_use[i * rs_c] = ctemp[i];
      }
    }
    post_ops_attr.post_op_c_i += MR;
  }
  } // mr loop
}

#endif // BLIS_ADDON_LPGEMM
