/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2023 - 2024, Advanced Micro Devices, Inc. All rights reserved.

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

LPGEMM_MAIN_KERN(float,float,float,f32f32f32of32_avx512_6x64m)
{
    static void* post_ops_labels[] =
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

            lpgemm_rowvar_f32f32f32of32_avx512_6x48m
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

            lpgemm_rowvar_f32f32f32of32_avx512_6x32m
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

        if ( 16 <= n_left )
        {
            const dim_t nr_cur = 16;

            lpgemm_rowvar_f32f32f32of32_6x16m
            (
              m0, nr_cur, k0,
              ai,  rs_a, cs_a, ps_a,
              bj,  rs_b, cs_b,
              cij, rs_c, cs_c,
              alpha, beta,
              post_ops_list, post_ops_attr
            );
            cij += nr_cur*cs_c; bj += nr_cur*cs_b; n_left -= nr_cur;
			post_ops_attr.post_op_c_j += 16;
        }

        if ( 8 <= n_left )
        {
            const dim_t nr_cur = 8;

            lpgemm_rowvar_f32f32f32of32_6x8m
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

            lpgemm_rowvar_f32f32f32of32_6x4m
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
  
            lpgemm_rowvar_f32f32f32of32_6x2m
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
            lpgemm_rowvar_f32f32f32of32_6x1m
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
        _cbuf = cbuf;
        //load c and multiply with beta and 
        //add to accumulator and store back
        zmm3 = _mm512_set1_ps(beta);        

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
      // Post Ops
      lpgemm_post_op* post_ops_list_temp = post_ops_list;
      POST_OP_LABEL_LASTK_SAFE_JUMP

POST_OPS_BIAS_6x64F:
      {
        if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
             ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
        {
          zmm1 =
            _mm512_loadu_ps( ( float* )post_ops_list_temp->op_args1 +
              post_ops_attr.post_op_c_j + ( 0 * 16 ) );
          zmm2 =
            _mm512_loadu_ps( ( float* )post_ops_list_temp->op_args1 +
              post_ops_attr.post_op_c_j + ( 1 * 16 ) );
          zmm3 =
            _mm512_loadu_ps( ( float* )post_ops_list_temp->op_args1 +
              post_ops_attr.post_op_c_j + ( 2 * 16 ) );
          zmm4 =
            _mm512_loadu_ps( ( float* )post_ops_list_temp->op_args1 +
              post_ops_attr.post_op_c_j + ( 3 * 16 ) );

          // c[0,0-15]
          zmm8 = _mm512_add_ps( zmm1, zmm8 );

          // c[0, 16-31]
          zmm9 = _mm512_add_ps( zmm2, zmm9 );

          // c[0,32-47]
          zmm10 = _mm512_add_ps( zmm3, zmm10 );

          // c[0,48-63]
          zmm11 = _mm512_add_ps( zmm4, zmm11 );

          // c[1,0-15]
          zmm12 = _mm512_add_ps( zmm1, zmm12 );

          // c[1, 16-31]
          zmm13 = _mm512_add_ps( zmm2, zmm13 );

          // c[1,32-47]
          zmm14 = _mm512_add_ps( zmm3, zmm14 );

          // c[1,48-63]
          zmm15 = _mm512_add_ps( zmm4, zmm15 );

          // c[2,0-15]
          zmm16 = _mm512_add_ps( zmm1, zmm16 );

          // c[2, 16-31]
          zmm17 = _mm512_add_ps( zmm2, zmm17 );

          // c[2,32-47]
          zmm18 = _mm512_add_ps( zmm3, zmm18 );

          // c[2,48-63]
          zmm19 = _mm512_add_ps( zmm4, zmm19 );

          // c[3,0-15]
          zmm20 = _mm512_add_ps( zmm1, zmm20 );

          // c[3, 16-31]
          zmm21 = _mm512_add_ps( zmm2, zmm21 );

          // c[3,32-47]
          zmm22 = _mm512_add_ps( zmm3, zmm22 );

          // c[3,48-63]
          zmm23 = _mm512_add_ps( zmm4, zmm23 );

          // c[4,0-15]
          zmm24 = _mm512_add_ps( zmm1, zmm24 );

          // c[4, 16-31]
          zmm25 = _mm512_add_ps( zmm2, zmm25 );

          // c[4,32-47]
          zmm26 = _mm512_add_ps( zmm3, zmm26 );

          // c[4,48-63]
          zmm27 = _mm512_add_ps( zmm4, zmm27 );

          // c[5,0-15]
          zmm28 = _mm512_add_ps( zmm1, zmm28 );

          // c[5, 16-31]
          zmm29 = _mm512_add_ps( zmm2, zmm29 );

          // c[5,32-47]
          zmm30 = _mm512_add_ps( zmm3, zmm30 );

          // c[5,48-63]
          zmm31 = _mm512_add_ps( zmm4, zmm31 );
        }
        else
        {
          // If original output was columns major, then by the time
          // kernel sees it, the matrix would be accessed as if it were
          // transposed. Due to this the bias array will be accessed by
          // the ic index, and each bias element corresponds to an
          // entire row of the transposed output array, instead of an
          // entire column.
          zmm1 =
            _mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args1 +
              post_ops_attr.post_op_c_i + 0 ) );
          zmm2 =
            _mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args1 +
              post_ops_attr.post_op_c_i + 1 ) );
          zmm3 =
            _mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args1 +
              post_ops_attr.post_op_c_i + 2 ) );
          zmm4 =
            _mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args1 +
              post_ops_attr.post_op_c_i + 3 ) );
          zmm5 =
            _mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args1 +
              post_ops_attr.post_op_c_i + 4 ) );
          zmm6 =
            _mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args1 +
              post_ops_attr.post_op_c_i + 5 ) );

          // c[0,0-15]
          zmm8 = _mm512_add_ps( zmm1, zmm8 );

          // c[0, 16-31]
          zmm9 = _mm512_add_ps( zmm1, zmm9 );

          // c[0,32-47]
          zmm10 = _mm512_add_ps( zmm1, zmm10 );

          // c[0,48-63]
          zmm11 = _mm512_add_ps( zmm1, zmm11 );

          // c[1,0-15]
          zmm12 = _mm512_add_ps( zmm2, zmm12 );

          // c[1, 16-31]
          zmm13 = _mm512_add_ps( zmm2, zmm13 );

          // c[1,32-47]
          zmm14 = _mm512_add_ps( zmm2, zmm14 );

          // c[1,48-63]
          zmm15 = _mm512_add_ps( zmm2, zmm15 );

          // c[2,0-15]
          zmm16 = _mm512_add_ps( zmm3, zmm16 );

          // c[2, 16-31]
          zmm17 = _mm512_add_ps( zmm3, zmm17 );

          // c[2,32-47]
          zmm18 = _mm512_add_ps( zmm3, zmm18 );

          // c[2,48-63]
          zmm19 = _mm512_add_ps( zmm3, zmm19 );

          // c[3,0-15]
          zmm20 = _mm512_add_ps( zmm4, zmm20 );

          // c[3, 16-31]
          zmm21 = _mm512_add_ps( zmm4, zmm21 );

          // c[3,32-47]
          zmm22 = _mm512_add_ps( zmm4, zmm22 );

          // c[3,48-63]
          zmm23 = _mm512_add_ps( zmm4, zmm23 );

          // c[4,0-15]
          zmm24 = _mm512_add_ps( zmm5, zmm24 );

          // c[4, 16-31]
          zmm25 = _mm512_add_ps( zmm5, zmm25 );

          // c[4,32-47]
          zmm26 = _mm512_add_ps( zmm5, zmm26 );

          // c[4,48-63]
          zmm27 = _mm512_add_ps( zmm5, zmm27 );

          // c[5,0-15]
          zmm28 = _mm512_add_ps( zmm6, zmm28 );

          // c[5, 16-31]
          zmm29 = _mm512_add_ps( zmm6, zmm29 );

          // c[5,32-47]
          zmm30 = _mm512_add_ps( zmm6, zmm30 );

          // c[5,48-63]
          zmm31 = _mm512_add_ps( zmm6, zmm31 );
        }
        POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
      }
POST_OPS_RELU_6x64F:
      {
        zmm1 = _mm512_setzero_ps();

        // c[0,0-15]
        zmm8 = _mm512_max_ps( zmm1, zmm8 );

        // c[0, 16-31]
        zmm9 = _mm512_max_ps( zmm1, zmm9 );

        // c[0,32-47]
        zmm10 = _mm512_max_ps( zmm1, zmm10 );

        // c[0,48-63]
        zmm11 = _mm512_max_ps( zmm1, zmm11 );

        // c[1,0-15]
        zmm12 = _mm512_max_ps( zmm1, zmm12 );

        // c[1,16-31]
        zmm13 = _mm512_max_ps( zmm1, zmm13 );

        // c[1,32-47]
        zmm14 = _mm512_max_ps( zmm1, zmm14 );

        // c[1,48-63]
        zmm15 = _mm512_max_ps( zmm1, zmm15 );

        // c[2,0-15]
        zmm16 = _mm512_max_ps( zmm1, zmm16 );

        // c[2,16-31]
        zmm17 = _mm512_max_ps( zmm1, zmm17 );

        // c[2,32-47]
        zmm18 = _mm512_max_ps( zmm1, zmm18 );

        // c[2,48-63]
        zmm19 = _mm512_max_ps( zmm1, zmm19 );

        // c[3,0-15]
        zmm20 = _mm512_max_ps( zmm1, zmm20 );

        // c[3,16-31]
        zmm21 = _mm512_max_ps( zmm1, zmm21 );

        // c[3,32-47]
        zmm22 = _mm512_max_ps( zmm1, zmm22 );

        // c[3,48-63]
        zmm23 = _mm512_max_ps( zmm1, zmm23 );

        // c[4,0-15]
        zmm24 = _mm512_max_ps( zmm1, zmm24 );

        // c[4,16-31]
        zmm25 = _mm512_max_ps( zmm1, zmm25 );

        // c[4,32-47]
        zmm26 = _mm512_max_ps( zmm1, zmm26 );

        // c[4,48-63]
        zmm27 = _mm512_max_ps( zmm1, zmm27 );

        // c[5,0-15]
        zmm28 = _mm512_max_ps( zmm1, zmm28 );

        // c[5,16-31]
        zmm29 = _mm512_max_ps( zmm1, zmm29 );

        // c[5,32-47]
        zmm30 = _mm512_max_ps( zmm1, zmm30 );

        // c[5,48-63]
        zmm31 = _mm512_max_ps( zmm1, zmm31 );

        POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
      }
POST_OPS_RELU_SCALE_6x64F:
      {
        zmm1 = _mm512_setzero_ps();
        zmm2 =
          _mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args2 ) );

        __mmask16 relu_cmp_mask;

        // c[0, 0-15]
        RELU_SCALE_OP_F32S_AVX512(zmm8)

        // c[0, 16-31]
        RELU_SCALE_OP_F32S_AVX512(zmm9)

        // c[0, 32-47]
        RELU_SCALE_OP_F32S_AVX512(zmm10)

        // c[0, 48-63]
        RELU_SCALE_OP_F32S_AVX512(zmm11)

        // c[1, 0-15]
        RELU_SCALE_OP_F32S_AVX512(zmm12)

        // c[1, 16-31]
        RELU_SCALE_OP_F32S_AVX512(zmm13)

        // c[1, 32-47]
        RELU_SCALE_OP_F32S_AVX512(zmm14)

        // c[1, 48-63]
        RELU_SCALE_OP_F32S_AVX512(zmm15)

        // c[2, 0-15]
        RELU_SCALE_OP_F32S_AVX512(zmm16)

        // c[2, 16-31]
        RELU_SCALE_OP_F32S_AVX512(zmm17)

        // c[2, 32-47]
        RELU_SCALE_OP_F32S_AVX512(zmm18)

        // c[2, 48-63]
        RELU_SCALE_OP_F32S_AVX512(zmm19)

        // c[3, 0-15]
        RELU_SCALE_OP_F32S_AVX512(zmm20)

        // c[3, 16-31]
        RELU_SCALE_OP_F32S_AVX512(zmm21)

        // c[3, 32-47]
        RELU_SCALE_OP_F32S_AVX512(zmm22)

        // c[3, 48-63]
        RELU_SCALE_OP_F32S_AVX512(zmm23)

        // c[4, 0-15]
        RELU_SCALE_OP_F32S_AVX512(zmm24)

        // c[4, 16-31]
        RELU_SCALE_OP_F32S_AVX512(zmm25)

        // c[4, 32-47]
        RELU_SCALE_OP_F32S_AVX512(zmm26)

        // c[4, 48-63]
        RELU_SCALE_OP_F32S_AVX512(zmm27)

        // c[5, 0-15]
        RELU_SCALE_OP_F32S_AVX512(zmm28)

        // c[5, 16-31]
        RELU_SCALE_OP_F32S_AVX512(zmm29)

        // c[5, 32-47]
        RELU_SCALE_OP_F32S_AVX512(zmm30)

        // c[5, 48-63]
        RELU_SCALE_OP_F32S_AVX512(zmm31)

        POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
      }
POST_OPS_GELU_TANH_6x64F:
      {
        __m512i zmm6;
        // c[0, 0-15]
        GELU_TANH_F32S_AVX512(zmm8, zmm0, zmm1, zmm2, zmm3, zmm4, zmm5, zmm6)

        // c[0, 16-31]
        GELU_TANH_F32S_AVX512(zmm9, zmm0, zmm1, zmm2, zmm3, zmm4, zmm5, zmm6)

        // c[0, 32-47]
        GELU_TANH_F32S_AVX512(zmm10, zmm0, zmm1, zmm2, zmm3, zmm4, zmm5, zmm6)

        // c[0, 48-63]
        GELU_TANH_F32S_AVX512(zmm11, zmm0, zmm1, zmm2, zmm3, zmm4, zmm5, zmm6)

        // c[1, 0-15]
        GELU_TANH_F32S_AVX512(zmm12, zmm0, zmm1, zmm2, zmm3, zmm4, zmm5, zmm6)

        // c[1, 16-31]
        GELU_TANH_F32S_AVX512(zmm13, zmm0, zmm1, zmm2, zmm3, zmm4, zmm5, zmm6)

        // c[1, 32-47]
        GELU_TANH_F32S_AVX512(zmm14, zmm0, zmm1, zmm2, zmm3, zmm4, zmm5, zmm6)

        // c[1, 48-63]
        GELU_TANH_F32S_AVX512(zmm15, zmm0, zmm1, zmm2, zmm3, zmm4, zmm5, zmm6)

        // c[2, 0-15]
        GELU_TANH_F32S_AVX512(zmm16, zmm0, zmm1, zmm2, zmm3, zmm4, zmm5, zmm6)

        // c[2, 16-31]
        GELU_TANH_F32S_AVX512(zmm17, zmm0, zmm1, zmm2, zmm3, zmm4, zmm5, zmm6)

        // c[2, 32-47]
        GELU_TANH_F32S_AVX512(zmm18, zmm0, zmm1, zmm2, zmm3, zmm4, zmm5, zmm6)

        // c[2, 48-63]
        GELU_TANH_F32S_AVX512(zmm19, zmm0, zmm1, zmm2, zmm3, zmm4, zmm5, zmm6)

        // c[3, 0-15]
        GELU_TANH_F32S_AVX512(zmm20, zmm0, zmm1, zmm2, zmm3, zmm4, zmm5, zmm6)

        // c[3, 16-31]
        GELU_TANH_F32S_AVX512(zmm21, zmm0, zmm1, zmm2, zmm3, zmm4, zmm5, zmm6)

        // c[3, 32-47]
        GELU_TANH_F32S_AVX512(zmm22, zmm0, zmm1, zmm2, zmm3, zmm4, zmm5, zmm6)

        // c[3, 48-63]
        GELU_TANH_F32S_AVX512(zmm23, zmm0, zmm1, zmm2, zmm3, zmm4, zmm5, zmm6)

        // c[4, 0-15]
        GELU_TANH_F32S_AVX512(zmm24, zmm0, zmm1, zmm2, zmm3, zmm4, zmm5, zmm6)

        // c[4, 16-31]
        GELU_TANH_F32S_AVX512(zmm25, zmm0, zmm1, zmm2, zmm3, zmm4, zmm5, zmm6)

        // c[4, 32-47]
        GELU_TANH_F32S_AVX512(zmm26, zmm0, zmm1, zmm2, zmm3, zmm4, zmm5, zmm6)

        // c[4, 48-63]
        GELU_TANH_F32S_AVX512(zmm27, zmm0, zmm1, zmm2, zmm3, zmm4, zmm5, zmm6)

        // c[5, 0-15]
        GELU_TANH_F32S_AVX512(zmm28, zmm0, zmm1, zmm2, zmm3, zmm4, zmm5, zmm6)

        // c[5, 16-31]
        GELU_TANH_F32S_AVX512(zmm29, zmm0, zmm1, zmm2, zmm3, zmm4, zmm5, zmm6)

        // c[5, 32-47]
        GELU_TANH_F32S_AVX512(zmm30, zmm0, zmm1, zmm2, zmm3, zmm4, zmm5, zmm6)

        // c[5, 48-63]
        GELU_TANH_F32S_AVX512(zmm31, zmm0, zmm1, zmm2, zmm3, zmm4, zmm5, zmm6)

        POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
      }
POST_OPS_GELU_ERF_6x64F:
      {
        // c[0, 0-15]
        GELU_ERF_F32S_AVX512(zmm8, zmm0, zmm1, zmm2)

        // c[0, 16-31]
        GELU_ERF_F32S_AVX512(zmm9, zmm0, zmm1, zmm2)

        // c[0, 32-47]
        GELU_ERF_F32S_AVX512(zmm10, zmm0, zmm1, zmm2)

        // c[0, 48-63]
        GELU_ERF_F32S_AVX512(zmm11, zmm0, zmm1, zmm2)

        // c[1, 0-15]
        GELU_ERF_F32S_AVX512(zmm12, zmm0, zmm1, zmm2)

        // c[1, 16-31]
        GELU_ERF_F32S_AVX512(zmm13, zmm0, zmm1, zmm2)

        // c[1, 32-47]
        GELU_ERF_F32S_AVX512(zmm14, zmm0, zmm1, zmm2)

        // c[1, 48-63]
        GELU_ERF_F32S_AVX512(zmm15, zmm0, zmm1, zmm2)

        // c[2, 0-15]
        GELU_ERF_F32S_AVX512(zmm16, zmm0, zmm1, zmm2)

        // c[2, 16-31]
        GELU_ERF_F32S_AVX512(zmm17, zmm0, zmm1, zmm2)

        // c[2, 32-47]
        GELU_ERF_F32S_AVX512(zmm18, zmm0, zmm1, zmm2)

        // c[2, 48-63]
        GELU_ERF_F32S_AVX512(zmm19, zmm0, zmm1, zmm2)

        // c[3, 0-15]
        GELU_ERF_F32S_AVX512(zmm20, zmm0, zmm1, zmm2)

        // c[3, 16-31]
        GELU_ERF_F32S_AVX512(zmm21, zmm0, zmm1, zmm2)

        // c[3, 32-47]
        GELU_ERF_F32S_AVX512(zmm22, zmm0, zmm1, zmm2)

        // c[3, 48-63]
        GELU_ERF_F32S_AVX512(zmm23, zmm0, zmm1, zmm2)

        // c[4, 0-15]
        GELU_ERF_F32S_AVX512(zmm24, zmm0, zmm1, zmm2)

        // c[4, 16-31]
        GELU_ERF_F32S_AVX512(zmm25, zmm0, zmm1, zmm2)

        // c[4, 32-47]
        GELU_ERF_F32S_AVX512(zmm26, zmm0, zmm1, zmm2)

        // c[4, 48-63]
        GELU_ERF_F32S_AVX512(zmm27, zmm0, zmm1, zmm2)

        // c[5, 0-15]
        GELU_ERF_F32S_AVX512(zmm28, zmm0, zmm1, zmm2)

        // c[5, 16-31]
        GELU_ERF_F32S_AVX512(zmm29, zmm0, zmm1, zmm2)

        // c[5, 32-47]
        GELU_ERF_F32S_AVX512(zmm30, zmm0, zmm1, zmm2)

        // c[5, 48-63]
        GELU_ERF_F32S_AVX512(zmm31, zmm0, zmm1, zmm2)

        POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
      }
POST_OPS_CLIP_6x64F:
      {
        zmm0 = _mm512_set1_ps( *( float* )post_ops_list_temp->op_args2 );
        zmm1 = _mm512_set1_ps( *( float* )post_ops_list_temp->op_args3 );

        // c[0, 0-15]
        CLIP_F32S_AVX512(zmm8, zmm0, zmm1)

        // c[0, 16-31]
        CLIP_F32S_AVX512(zmm9, zmm0, zmm1)

        // c[0, 32-47]
        CLIP_F32S_AVX512(zmm10, zmm0, zmm1)

        // c[0, 48-63]
        CLIP_F32S_AVX512(zmm11, zmm0, zmm1)

        // c[1, 0-15]
        CLIP_F32S_AVX512(zmm12, zmm0, zmm1)

        // c[1, 16-31]
        CLIP_F32S_AVX512(zmm13, zmm0, zmm1)

        // c[1, 32-47]
        CLIP_F32S_AVX512(zmm14, zmm0, zmm1)

        // c[1, 48-63]
        CLIP_F32S_AVX512(zmm15, zmm0, zmm1)

        // c[2, 0-15]
        CLIP_F32S_AVX512(zmm16, zmm0, zmm1)

        // c[2, 16-31]
        CLIP_F32S_AVX512(zmm17, zmm0, zmm1)

        // c[2, 32-47]
        CLIP_F32S_AVX512(zmm18, zmm0, zmm1)

        // c[2, 48-63]
        CLIP_F32S_AVX512(zmm19, zmm0, zmm1)

        // c[3, 0-15]
        CLIP_F32S_AVX512(zmm20, zmm0, zmm1)

        // c[3, 16-31]
        CLIP_F32S_AVX512(zmm21, zmm0, zmm1)

        // c[3, 32-47]
        CLIP_F32S_AVX512(zmm22, zmm0, zmm1)

        // c[3, 48-63]
        CLIP_F32S_AVX512(zmm23, zmm0, zmm1)

        // c[4, 0-15]
        CLIP_F32S_AVX512(zmm24, zmm0, zmm1)

        // c[4, 16-31]
        CLIP_F32S_AVX512(zmm25, zmm0, zmm1)

        // c[4, 32-47]
        CLIP_F32S_AVX512(zmm26, zmm0, zmm1)

        // c[4, 48-63]
        CLIP_F32S_AVX512(zmm27, zmm0, zmm1)

        // c[5, 0-15]
        CLIP_F32S_AVX512(zmm28, zmm0, zmm1)

        // c[5, 16-31]
        CLIP_F32S_AVX512(zmm29, zmm0, zmm1)

        // c[5, 32-47]
        CLIP_F32S_AVX512(zmm30, zmm0, zmm1)

        // c[5, 48-63]
        CLIP_F32S_AVX512(zmm31, zmm0, zmm1)

        POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
      }
POST_OPS_MATRIX_ADD_6x64F:
      {
          dim_t ldm = *( dim_t* )post_ops_list_temp->op_args3;
          float* matptr = ( float* )post_ops_list_temp->op_args1;

          // c[0:0-15,16-31,32-47,48-63]
          F32_F32_MATRIX_ADD_4COL(zmm1,zmm2,zmm3,zmm4,0,8,9,10,11);

          // c[1:0-15,16-31,32-47,48-63]
          F32_F32_MATRIX_ADD_4COL(zmm1,zmm2,zmm3,zmm4,1,12,13,14,15);

          // c[2:0-15,16-31,32-47,48-63]
          F32_F32_MATRIX_ADD_4COL(zmm1,zmm2,zmm3,zmm4,2,16,17,18,19);

          // c[3:0-15,16-31,32-47,48-63]
          F32_F32_MATRIX_ADD_4COL(zmm1,zmm2,zmm3,zmm4,3,20,21,22,23);

          // c[4:0-15,16-31,32-47,48-63]
          F32_F32_MATRIX_ADD_4COL(zmm1,zmm2,zmm3,zmm4,4,24,25,26,27);

          // c[5:0-15,16-31,32-47,48-63]
          F32_F32_MATRIX_ADD_4COL(zmm1,zmm2,zmm3,zmm4,5,28,29,30,31);

          POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
      }
POST_OPS_MATRIX_MUL_6x64F:
      {
          dim_t ldm = *( dim_t* )post_ops_list_temp->op_args3;
          float* matptr = ( float* )post_ops_list_temp->op_args1;
          // c[0:0-15,16-31,32-47,48-63]
          F32_F32_MATRIX_MUL_4COL(zmm1,zmm2,zmm3,zmm4,0,8,9,10,11);

          // c[1:0-15,16-31,32-47,48-63]
          F32_F32_MATRIX_MUL_4COL(zmm1,zmm2,zmm3,zmm4,1,12,13,14,15);

          // c[2:0-15,16-31,32-47,48-63]
          F32_F32_MATRIX_MUL_4COL(zmm1,zmm2,zmm3,zmm4,2,16,17,18,19);

          // c[3:0-15,16-31,32-47,48-63]
          F32_F32_MATRIX_MUL_4COL(zmm1,zmm2,zmm3,zmm4,3,20,21,22,23);

          // c[4:0-15,16-31,32-47,48-63]
          F32_F32_MATRIX_MUL_4COL(zmm1,zmm2,zmm3,zmm4,4,24,25,26,27);

          // c[5:0-15,16-31,32-47,48-63]
          F32_F32_MATRIX_MUL_4COL(zmm1,zmm2,zmm3,zmm4,5,28,29,30,31);

          POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
      }      
POST_OPS_SWISH_6x64F:
      {
          zmm7 =
              _mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args2 ) );
          __m512i ex_out;

          // c[0, 0-15]
          SWISH_F32_AVX512_DEF(zmm8, zmm7, zmm0, zmm1, zmm2, zmm3, zmm4, ex_out);

          // c[0, 16-31]
          SWISH_F32_AVX512_DEF(zmm9, zmm7, zmm0, zmm1, zmm2, zmm3, zmm4, ex_out);

          // c[0, 32-47]
          SWISH_F32_AVX512_DEF(zmm10, zmm7, zmm0, zmm1, zmm2, zmm3, zmm4, ex_out);

          // c[0, 48-63]
          SWISH_F32_AVX512_DEF(zmm11, zmm7, zmm0, zmm1, zmm2, zmm3, zmm4, ex_out);

          // c[1, 0-15]
          SWISH_F32_AVX512_DEF(zmm12, zmm7, zmm0, zmm1, zmm2, zmm3, zmm4, ex_out);

          // c[1, 16-31]
          SWISH_F32_AVX512_DEF(zmm13, zmm7, zmm0, zmm1, zmm2, zmm3, zmm4, ex_out);

          // c[1, 32-47]
          SWISH_F32_AVX512_DEF(zmm14, zmm7, zmm0, zmm1, zmm2, zmm3, zmm4, ex_out);

          // c[1, 48-63]
          SWISH_F32_AVX512_DEF(zmm15, zmm7, zmm0, zmm1, zmm2, zmm3, zmm4, ex_out);

          // c[2, 0-15]
          SWISH_F32_AVX512_DEF(zmm16, zmm7, zmm0, zmm1, zmm2, zmm3, zmm4, ex_out);

          // c[2, 16-31]
          SWISH_F32_AVX512_DEF(zmm17, zmm7, zmm0, zmm1, zmm2, zmm3, zmm4, ex_out);

          // c[2, 32-47]
          SWISH_F32_AVX512_DEF(zmm18, zmm7, zmm0, zmm1, zmm2, zmm3, zmm4, ex_out);

          // c[2, 48-63]
          SWISH_F32_AVX512_DEF(zmm19, zmm7, zmm0, zmm1, zmm2, zmm3, zmm4, ex_out);

          // c[3, 0-15]
          SWISH_F32_AVX512_DEF(zmm20, zmm7, zmm0, zmm1, zmm2, zmm3, zmm4, ex_out);

          // c[3, 16-31]
          SWISH_F32_AVX512_DEF(zmm21, zmm7, zmm0, zmm1, zmm2, zmm3, zmm4, ex_out);

          // c[3, 32-47]
          SWISH_F32_AVX512_DEF(zmm22, zmm7, zmm0, zmm1, zmm2, zmm3, zmm4, ex_out);

          // c[3, 48-63]
          SWISH_F32_AVX512_DEF(zmm23, zmm7, zmm0, zmm1, zmm2, zmm3, zmm4, ex_out);

          // c[4, 0-15]
          SWISH_F32_AVX512_DEF(zmm24, zmm7, zmm0, zmm1, zmm2, zmm3, zmm4, ex_out);

          // c[4, 16-31]
          SWISH_F32_AVX512_DEF(zmm25, zmm7, zmm0, zmm1, zmm2, zmm3, zmm4, ex_out);

          // c[4, 32-47]
          SWISH_F32_AVX512_DEF(zmm26, zmm7, zmm0, zmm1, zmm2, zmm3, zmm4, ex_out);

          // c[4, 48-63]
          SWISH_F32_AVX512_DEF(zmm27, zmm7, zmm0, zmm1, zmm2, zmm3, zmm4, ex_out);

          // c[5, 0-15]
          SWISH_F32_AVX512_DEF(zmm28, zmm7, zmm0, zmm1, zmm2, zmm3, zmm4, ex_out);

          // c[5, 16-31]
          SWISH_F32_AVX512_DEF(zmm29, zmm7, zmm0, zmm1, zmm2, zmm3, zmm4, ex_out);

          // c[5, 32-47]
          SWISH_F32_AVX512_DEF(zmm30, zmm7, zmm0, zmm1, zmm2, zmm3, zmm4, ex_out);

          // c[5, 48-63]
          SWISH_F32_AVX512_DEF(zmm31, zmm7, zmm0, zmm1, zmm2, zmm3, zmm4, ex_out);

          POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
      }
POST_OPS_6x64F_DISABLE:
      ;

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
          lpgemm_rowvar_f32f32f32of32_avx512_1x64,
          lpgemm_rowvar_f32f32f32of32_avx512_2x64,
          lpgemm_rowvar_f32f32f32of32_avx512_3x64,
          lpgemm_rowvar_f32f32f32of32_avx512_4x64,
          lpgemm_rowvar_f32f32f32of32_avx512_5x64
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

LPGEMM_N_FRINGE_KERN(float,float,float,f32f32f32of32_avx512_6x48m)
{
    static void* post_ops_labels[] =
            {
              &&POST_OPS_6x48F_DISABLE,
              &&POST_OPS_BIAS_6x48F,
              &&POST_OPS_RELU_6x48F,
              &&POST_OPS_RELU_SCALE_6x48F,
              &&POST_OPS_GELU_TANH_6x48F,
              &&POST_OPS_GELU_ERF_6x48F,
              &&POST_OPS_CLIP_6x48F,
              NULL, // Virtual node for downscale, else segfault
              &&POST_OPS_MATRIX_ADD_6x48F,
              &&POST_OPS_SWISH_6x48F,
              &&POST_OPS_MATRIX_MUL_6x48F
            };
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
        _cbuf = cbuf; 
        //load c and multiply with beta and 
        //add to accumulator and store back
        zmm3 = _mm512_set1_ps(beta);

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

      // Post Ops
      lpgemm_post_op* post_ops_list_temp = post_ops_list;
      POST_OP_LABEL_LASTK_SAFE_JUMP

POST_OPS_BIAS_6x48F:
      {
        if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
             ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
        {
          zmm1 =
            _mm512_loadu_ps( ( float* )post_ops_list_temp->op_args1 +
              post_ops_attr.post_op_c_j + ( 0 * 16 ) );
          zmm2 =
            _mm512_loadu_ps( ( float* )post_ops_list_temp->op_args1 +
              post_ops_attr.post_op_c_j + ( 1 * 16 ) );
          zmm3 =
            _mm512_loadu_ps( ( float* )post_ops_list_temp->op_args1 +
              post_ops_attr.post_op_c_j + ( 2 * 16 ) );

          // c[0,0-15]
          zmm8 = _mm512_add_ps( zmm1, zmm8 );

          // c[0, 16-31]
          zmm9 = _mm512_add_ps( zmm2, zmm9 );

          // c[0,32-47]
          zmm10 = _mm512_add_ps( zmm3, zmm10 );

          // c[1,0-15]
          zmm12 = _mm512_add_ps( zmm1, zmm12 );

          // c[1, 16-31]
          zmm13 = _mm512_add_ps( zmm2, zmm13 );

          // c[1,32-47]
          zmm14 = _mm512_add_ps( zmm3, zmm14 );

          // c[2,0-15]
          zmm16 = _mm512_add_ps( zmm1, zmm16 );

          // c[2, 16-31]
          zmm17 = _mm512_add_ps( zmm2, zmm17 );

          // c[2,32-47]
          zmm18 = _mm512_add_ps( zmm3, zmm18 );

          // c[3,0-15]
          zmm20 = _mm512_add_ps( zmm1, zmm20 );

          // c[3, 16-31]
          zmm21 = _mm512_add_ps( zmm2, zmm21 );

          // c[3,32-47]
          zmm22 = _mm512_add_ps( zmm3, zmm22 );

          // c[4,0-15]
          zmm24 = _mm512_add_ps( zmm1, zmm24 );

          // c[4, 16-31]
          zmm25 = _mm512_add_ps( zmm2, zmm25 );

          // c[4,32-47]
          zmm26 = _mm512_add_ps( zmm3, zmm26 );

          // c[5,0-15]
          zmm28 = _mm512_add_ps( zmm1, zmm28 );

          // c[5, 16-31]
          zmm29 = _mm512_add_ps( zmm2, zmm29 );

          // c[5,32-47]
          zmm30 = _mm512_add_ps( zmm3, zmm30 );
        }
        else
        {
          // If original output was columns major, then by the time
          // kernel sees it, the matrix would be accessed as if it were
          // transposed. Due to this the bias array will be accessed by
          // the ic index, and each bias element corresponds to an
          // entire row of the transposed output array, instead of an
          // entire column.
          zmm1 =
            _mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args1 +
              post_ops_attr.post_op_c_i + 0 ) );
          zmm2 =
            _mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args1 +
              post_ops_attr.post_op_c_i + 1 ) );
          zmm3 =
            _mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args1 +
              post_ops_attr.post_op_c_i + 2 ) );
          zmm4 =
            _mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args1 +
              post_ops_attr.post_op_c_i + 3 ) );
          zmm5 =
            _mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args1 +
              post_ops_attr.post_op_c_i + 4 ) );
          zmm6 =
            _mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args1 +
              post_ops_attr.post_op_c_i + 5 ) );

          // c[0,0-15]
          zmm8 = _mm512_add_ps( zmm1, zmm8 );

          // c[0, 16-31]
          zmm9 = _mm512_add_ps( zmm1, zmm9 );

          // c[0,32-47]
          zmm10 = _mm512_add_ps( zmm1, zmm10 );

          // c[1,0-15]
          zmm12 = _mm512_add_ps( zmm2, zmm12 );

          // c[1, 16-31]
          zmm13 = _mm512_add_ps( zmm2, zmm13 );

          // c[1,32-47]
          zmm14 = _mm512_add_ps( zmm2, zmm14 );

          // c[2,0-15]
          zmm16 = _mm512_add_ps( zmm3, zmm16 );

          // c[2, 16-31]
          zmm17 = _mm512_add_ps( zmm3, zmm17 );

          // c[2,32-47]
          zmm18 = _mm512_add_ps( zmm3, zmm18 );

          // c[3,0-15]
          zmm20 = _mm512_add_ps( zmm4, zmm20 );

          // c[3, 16-31]
          zmm21 = _mm512_add_ps( zmm4, zmm21 );

          // c[3,32-47]
          zmm22 = _mm512_add_ps( zmm4, zmm22 );

          // c[4,0-15]
          zmm24 = _mm512_add_ps( zmm5, zmm24 );

          // c[4, 16-31]
          zmm25 = _mm512_add_ps( zmm5, zmm25 );

          // c[4,32-47]
          zmm26 = _mm512_add_ps( zmm5, zmm26 );

          // c[5,0-15]
          zmm28 = _mm512_add_ps( zmm6, zmm28 );

          // c[5, 16-31]
          zmm29 = _mm512_add_ps( zmm6, zmm29 );

          // c[5,32-47]
          zmm30 = _mm512_add_ps( zmm6, zmm30 );
        }
        POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
      }
POST_OPS_RELU_6x48F:
      {
        zmm1 = _mm512_setzero_ps();

        // c[0,0-15]
        zmm8 = _mm512_max_ps( zmm1, zmm8 );

        // c[0, 16-31]
        zmm9 = _mm512_max_ps( zmm1, zmm9 );

        // c[0,32-47]
        zmm10 = _mm512_max_ps( zmm1, zmm10 );

        // c[1,0-15]
        zmm12 = _mm512_max_ps( zmm1, zmm12 );

        // c[1,16-31]
        zmm13 = _mm512_max_ps( zmm1, zmm13 );

        // c[1,32-47]
        zmm14 = _mm512_max_ps( zmm1, zmm14 );

        // c[2,0-15]
        zmm16 = _mm512_max_ps( zmm1, zmm16 );

        // c[2,16-31]
        zmm17 = _mm512_max_ps( zmm1, zmm17 );

        // c[2,32-47]
        zmm18 = _mm512_max_ps( zmm1, zmm18 );

        // c[3,0-15]
        zmm20 = _mm512_max_ps( zmm1, zmm20 );

        // c[3,16-31]
        zmm21 = _mm512_max_ps( zmm1, zmm21 );

        // c[3,32-47]
        zmm22 = _mm512_max_ps( zmm1, zmm22 );

        // c[4,0-15]
        zmm24 = _mm512_max_ps( zmm1, zmm24 );

        // c[4,16-31]
        zmm25 = _mm512_max_ps( zmm1, zmm25 );

        // c[4,32-47]
        zmm26 = _mm512_max_ps( zmm1, zmm26 );

        // c[5,0-15]
        zmm28 = _mm512_max_ps( zmm1, zmm28 );

        // c[5,16-31]
        zmm29 = _mm512_max_ps( zmm1, zmm29 );

        // c[5,32-47]
        zmm30 = _mm512_max_ps( zmm1, zmm30 );

        POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
      }
POST_OPS_RELU_SCALE_6x48F:
      {
        zmm1 = _mm512_setzero_ps();
        zmm2 =
          _mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args2 ) );

        __mmask16 relu_cmp_mask;

        // c[0, 0-15]
        RELU_SCALE_OP_F32S_AVX512(zmm8)

        // c[0, 16-31]
        RELU_SCALE_OP_F32S_AVX512(zmm9)

        // c[0, 32-47]
        RELU_SCALE_OP_F32S_AVX512(zmm10)

        // c[1, 0-15]
        RELU_SCALE_OP_F32S_AVX512(zmm12)

        // c[1, 16-31]
        RELU_SCALE_OP_F32S_AVX512(zmm13)

        // c[1, 32-47]
        RELU_SCALE_OP_F32S_AVX512(zmm14)

        // c[2, 0-15]
        RELU_SCALE_OP_F32S_AVX512(zmm16)

        // c[2, 16-31]
        RELU_SCALE_OP_F32S_AVX512(zmm17)

        // c[2, 32-47]
        RELU_SCALE_OP_F32S_AVX512(zmm18)

        // c[3, 0-15]
        RELU_SCALE_OP_F32S_AVX512(zmm20)

        // c[3, 16-31]
        RELU_SCALE_OP_F32S_AVX512(zmm21)

        // c[3, 32-47]
        RELU_SCALE_OP_F32S_AVX512(zmm22)

        // c[4, 0-15]
        RELU_SCALE_OP_F32S_AVX512(zmm24)

        // c[4, 16-31]
        RELU_SCALE_OP_F32S_AVX512(zmm25)

        // c[4, 32-47]
        RELU_SCALE_OP_F32S_AVX512(zmm26)

        // c[5, 0-15]
        RELU_SCALE_OP_F32S_AVX512(zmm28)

        // c[5, 16-31]
        RELU_SCALE_OP_F32S_AVX512(zmm29)

        // c[5, 32-47]
        RELU_SCALE_OP_F32S_AVX512(zmm30)

        POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
      }
POST_OPS_GELU_TANH_6x48F:
      {
        __m512i zmm6;
        // c[0, 0-15]
        GELU_TANH_F32S_AVX512(zmm8, zmm0, zmm1, zmm2, zmm3, zmm4, zmm5, zmm6)

        // c[0, 16-31]
        GELU_TANH_F32S_AVX512(zmm9, zmm0, zmm1, zmm2, zmm3, zmm4, zmm5, zmm6)

        // c[0, 32-47]
        GELU_TANH_F32S_AVX512(zmm10, zmm0, zmm1, zmm2, zmm3, zmm4, zmm5, zmm6)

        // c[1, 0-15]
        GELU_TANH_F32S_AVX512(zmm12, zmm0, zmm1, zmm2, zmm3, zmm4, zmm5, zmm6)

        // c[1, 16-31]
        GELU_TANH_F32S_AVX512(zmm13, zmm0, zmm1, zmm2, zmm3, zmm4, zmm5, zmm6)

        // c[1, 32-47]
        GELU_TANH_F32S_AVX512(zmm14, zmm0, zmm1, zmm2, zmm3, zmm4, zmm5, zmm6)

        // c[2, 0-15]
        GELU_TANH_F32S_AVX512(zmm16, zmm0, zmm1, zmm2, zmm3, zmm4, zmm5, zmm6)

        // c[2, 16-31]
        GELU_TANH_F32S_AVX512(zmm17, zmm0, zmm1, zmm2, zmm3, zmm4, zmm5, zmm6)

        // c[2, 32-47]
        GELU_TANH_F32S_AVX512(zmm18, zmm0, zmm1, zmm2, zmm3, zmm4, zmm5, zmm6)

        // c[3, 0-15]
        GELU_TANH_F32S_AVX512(zmm20, zmm0, zmm1, zmm2, zmm3, zmm4, zmm5, zmm6)

        // c[3, 16-31]
        GELU_TANH_F32S_AVX512(zmm21, zmm0, zmm1, zmm2, zmm3, zmm4, zmm5, zmm6)

        // c[3, 32-47]
        GELU_TANH_F32S_AVX512(zmm22, zmm0, zmm1, zmm2, zmm3, zmm4, zmm5, zmm6)

        // c[4, 0-15]
        GELU_TANH_F32S_AVX512(zmm24, zmm0, zmm1, zmm2, zmm3, zmm4, zmm5, zmm6)

        // c[4, 16-31]
        GELU_TANH_F32S_AVX512(zmm25, zmm0, zmm1, zmm2, zmm3, zmm4, zmm5, zmm6)

        // c[4, 32-47]
        GELU_TANH_F32S_AVX512(zmm26, zmm0, zmm1, zmm2, zmm3, zmm4, zmm5, zmm6)

        // c[5, 0-15]
        GELU_TANH_F32S_AVX512(zmm28, zmm0, zmm1, zmm2, zmm3, zmm4, zmm5, zmm6)

        // c[5, 16-31]
        GELU_TANH_F32S_AVX512(zmm29, zmm0, zmm1, zmm2, zmm3, zmm4, zmm5, zmm6)

        // c[5, 32-47]
        GELU_TANH_F32S_AVX512(zmm30, zmm0, zmm1, zmm2, zmm3, zmm4, zmm5, zmm6)

        POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
      }
POST_OPS_GELU_ERF_6x48F:
      {
        // c[0, 0-15]
        GELU_ERF_F32S_AVX512(zmm8, zmm0, zmm1, zmm2)

        // c[0, 16-31]
        GELU_ERF_F32S_AVX512(zmm9, zmm0, zmm1, zmm2)

        // c[0, 32-47]
        GELU_ERF_F32S_AVX512(zmm10, zmm0, zmm1, zmm2)

        // c[1, 0-15]
        GELU_ERF_F32S_AVX512(zmm12, zmm0, zmm1, zmm2)

        // c[1, 16-31]
        GELU_ERF_F32S_AVX512(zmm13, zmm0, zmm1, zmm2)

        // c[1, 32-47]
        GELU_ERF_F32S_AVX512(zmm14, zmm0, zmm1, zmm2)

        // c[2, 0-15]
        GELU_ERF_F32S_AVX512(zmm16, zmm0, zmm1, zmm2)

        // c[2, 16-31]
        GELU_ERF_F32S_AVX512(zmm17, zmm0, zmm1, zmm2)

        // c[2, 32-47]
        GELU_ERF_F32S_AVX512(zmm18, zmm0, zmm1, zmm2)

        // c[3, 0-15]
        GELU_ERF_F32S_AVX512(zmm20, zmm0, zmm1, zmm2)

        // c[3, 16-31]
        GELU_ERF_F32S_AVX512(zmm21, zmm0, zmm1, zmm2)

        // c[3, 32-47]
        GELU_ERF_F32S_AVX512(zmm22, zmm0, zmm1, zmm2)

        // c[4, 0-15]
        GELU_ERF_F32S_AVX512(zmm24, zmm0, zmm1, zmm2)

        // c[4, 16-31]
        GELU_ERF_F32S_AVX512(zmm25, zmm0, zmm1, zmm2)

        // c[4, 32-47]
        GELU_ERF_F32S_AVX512(zmm26, zmm0, zmm1, zmm2)

        // c[5, 0-15]
        GELU_ERF_F32S_AVX512(zmm28, zmm0, zmm1, zmm2)

        // c[5, 16-31]
        GELU_ERF_F32S_AVX512(zmm29, zmm0, zmm1, zmm2)

        // c[5, 32-47]
        GELU_ERF_F32S_AVX512(zmm30, zmm0, zmm1, zmm2)

        POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
      }
POST_OPS_CLIP_6x48F:
      {
        zmm0 = _mm512_set1_ps( *( float* )post_ops_list_temp->op_args2 );
        zmm1 = _mm512_set1_ps( *( float* )post_ops_list_temp->op_args3 );

        // c[0, 0-15]
        CLIP_F32S_AVX512(zmm8, zmm0, zmm1)

        // c[0, 16-31]
        CLIP_F32S_AVX512(zmm9, zmm0, zmm1)

        // c[0, 32-47]
        CLIP_F32S_AVX512(zmm10, zmm0, zmm1)

        // c[1, 0-15]
        CLIP_F32S_AVX512(zmm12, zmm0, zmm1)

        // c[1, 16-31]
        CLIP_F32S_AVX512(zmm13, zmm0, zmm1)

        // c[1, 32-47]
        CLIP_F32S_AVX512(zmm14, zmm0, zmm1)

        // c[2, 0-15]
        CLIP_F32S_AVX512(zmm16, zmm0, zmm1)

        // c[2, 16-31]
        CLIP_F32S_AVX512(zmm17, zmm0, zmm1)

        // c[2, 32-47]
        CLIP_F32S_AVX512(zmm18, zmm0, zmm1)

        // c[3, 0-15]
        CLIP_F32S_AVX512(zmm20, zmm0, zmm1)

        // c[3, 16-31]
        CLIP_F32S_AVX512(zmm21, zmm0, zmm1)

        // c[3, 32-47]
        CLIP_F32S_AVX512(zmm22, zmm0, zmm1)

        // c[4, 0-15]
        CLIP_F32S_AVX512(zmm24, zmm0, zmm1)

        // c[4, 16-31]
        CLIP_F32S_AVX512(zmm25, zmm0, zmm1)

        // c[4, 32-47]
        CLIP_F32S_AVX512(zmm26, zmm0, zmm1)

        // c[5, 0-15]
        CLIP_F32S_AVX512(zmm28, zmm0, zmm1)

        // c[5, 16-31]
        CLIP_F32S_AVX512(zmm29, zmm0, zmm1)

        // c[5, 32-47]
        CLIP_F32S_AVX512(zmm30, zmm0, zmm1)

        POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
      }
POST_OPS_MATRIX_ADD_6x48F:
      {
          dim_t ldm = *( dim_t* )post_ops_list_temp->op_args3;
          float* matptr = ( float* )post_ops_list_temp->op_args1;

          // c[0:0-15,16-31,32-47]
          F32_F32_MATRIX_ADD_3COL(zmm1,zmm2,zmm3,0,8,9,10);

          // c[1:0-15,16-31,32-47]
          F32_F32_MATRIX_ADD_3COL(zmm1,zmm2,zmm3,1,12,13,14);

          // c[2:0-15,16-31,32-47]
          F32_F32_MATRIX_ADD_3COL(zmm1,zmm2,zmm3,2,16,17,18);

          // c[3:0-15,16-31,32-47]
          F32_F32_MATRIX_ADD_3COL(zmm1,zmm2,zmm3,3,20,21,22);

          // c[4:0-15,16-31,32-47]
          F32_F32_MATRIX_ADD_3COL(zmm1,zmm2,zmm3,4,24,25,26);

          // c[5:0-15,16-31,32-47]
          F32_F32_MATRIX_ADD_3COL(zmm1,zmm2,zmm3,5,28,29,30);

          POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
      }
POST_OPS_MATRIX_MUL_6x48F:
      {
          dim_t ldm = *( dim_t* )post_ops_list_temp->op_args3;
          float* matptr = ( float* )post_ops_list_temp->op_args1;

          // c[0:0-15,16-31,32-47]
          F32_F32_MATRIX_MUL_3COL(zmm1,zmm2,zmm3,0,8,9,10);

          // c[1:0-15,16-31,32-47]
          F32_F32_MATRIX_MUL_3COL(zmm1,zmm2,zmm3,1,12,13,14);

          // c[2:0-15,16-31,32-47]
          F32_F32_MATRIX_MUL_3COL(zmm1,zmm2,zmm3,2,16,17,18);

          // c[3:0-15,16-31,32-47]
          F32_F32_MATRIX_MUL_3COL(zmm1,zmm2,zmm3,3,20,21,22);

          // c[4:0-15,16-31,32-47]
          F32_F32_MATRIX_MUL_3COL(zmm1,zmm2,zmm3,4,24,25,26);

          // c[5:0-15,16-31,32-47]
          F32_F32_MATRIX_MUL_3COL(zmm1,zmm2,zmm3,5,28,29,30);

          POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
      }      
POST_OPS_SWISH_6x48F:
      {
          __m512 zmm7 =
              _mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args2 ) );
          __m512i ex_out;

          // c[0, 0-15]
          SWISH_F32_AVX512_DEF(zmm8, zmm7, zmm0, zmm1, zmm2, zmm3, zmm4, ex_out);

          // c[0, 16-31]
          SWISH_F32_AVX512_DEF(zmm9, zmm7, zmm0, zmm1, zmm2, zmm3, zmm4, ex_out);

          // c[0, 32-47]
          SWISH_F32_AVX512_DEF(zmm10, zmm7, zmm0, zmm1, zmm2, zmm3, zmm4, ex_out);

          // c[1, 0-15]
          SWISH_F32_AVX512_DEF(zmm12, zmm7, zmm0, zmm1, zmm2, zmm3, zmm4, ex_out);

          // c[1, 16-31]
          SWISH_F32_AVX512_DEF(zmm13, zmm7, zmm0, zmm1, zmm2, zmm3, zmm4, ex_out);

          // c[1, 32-47]
          SWISH_F32_AVX512_DEF(zmm14, zmm7, zmm0, zmm1, zmm2, zmm3, zmm4, ex_out);

          // c[2, 0-15]
          SWISH_F32_AVX512_DEF(zmm16, zmm7, zmm0, zmm1, zmm2, zmm3, zmm4, ex_out);

          // c[2, 16-31]
          SWISH_F32_AVX512_DEF(zmm17, zmm7, zmm0, zmm1, zmm2, zmm3, zmm4, ex_out);

          // c[2, 32-47]
          SWISH_F32_AVX512_DEF(zmm18, zmm7, zmm0, zmm1, zmm2, zmm3, zmm4, ex_out);

          // c[3, 0-15]
          SWISH_F32_AVX512_DEF(zmm20, zmm7, zmm0, zmm1, zmm2, zmm3, zmm4, ex_out);

          // c[3, 16-31]
          SWISH_F32_AVX512_DEF(zmm21, zmm7, zmm0, zmm1, zmm2, zmm3, zmm4, ex_out);

          // c[3, 32-47]
          SWISH_F32_AVX512_DEF(zmm22, zmm7, zmm0, zmm1, zmm2, zmm3, zmm4, ex_out);

          // c[4, 0-15]
          SWISH_F32_AVX512_DEF(zmm24, zmm7, zmm0, zmm1, zmm2, zmm3, zmm4, ex_out);

          // c[4, 16-31]
          SWISH_F32_AVX512_DEF(zmm25, zmm7, zmm0, zmm1, zmm2, zmm3, zmm4, ex_out);

          // c[4, 32-47]
          SWISH_F32_AVX512_DEF(zmm26, zmm7, zmm0, zmm1, zmm2, zmm3, zmm4, ex_out);

          // c[5, 0-15]
          SWISH_F32_AVX512_DEF(zmm28, zmm7, zmm0, zmm1, zmm2, zmm3, zmm4, ex_out);

          // c[5, 16-31]
          SWISH_F32_AVX512_DEF(zmm29, zmm7, zmm0, zmm1, zmm2, zmm3, zmm4, ex_out);

          // c[5, 32-47]
          SWISH_F32_AVX512_DEF(zmm30, zmm7, zmm0, zmm1, zmm2, zmm3, zmm4, ex_out);

          POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
      }
POST_OPS_6x48F_DISABLE:
      ;
 
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
          lpgemm_rowvar_f32f32f32of32_avx512_1x48,
          lpgemm_rowvar_f32f32f32of32_avx512_2x48,
          lpgemm_rowvar_f32f32f32of32_avx512_3x48,
          lpgemm_rowvar_f32f32f32of32_avx512_4x48,
          lpgemm_rowvar_f32f32f32of32_avx512_5x48
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

LPGEMM_N_FRINGE_KERN(float,float,float,f32f32f32of32_avx512_6x32m)
{
    static void* post_ops_labels[] =
            {
              &&POST_OPS_6x32F_DISABLE,
              &&POST_OPS_BIAS_6x32F,
              &&POST_OPS_RELU_6x32F,
              &&POST_OPS_RELU_SCALE_6x32F,
              &&POST_OPS_GELU_TANH_6x32F,
              &&POST_OPS_GELU_ERF_6x32F,
              &&POST_OPS_CLIP_6x32F,
              NULL, // Virtual node for downscale, else segfault
              &&POST_OPS_MATRIX_ADD_6x32F,
              &&POST_OPS_SWISH_6x32F,
              &&POST_OPS_MATRIX_MUL_6x32F
            };
    // Typecast local copies of integers in case dim_t and inc_t are a
    // different size than is expected by load instructions.
    uint64_t k_iter = k0;

    uint64_t m_iter = m0 / 6;
    uint64_t m_left = m0 % 6;

    // Query the panel stride of A and convert it to units of bytes.
    if ( m_iter == 0 ){    goto consider_edge_cases; }

    __m512 zmm0, zmm1, zmm2, zmm3, zmm4, zmm5;
    __m512 zmm6, zmm8, zmm9, zmm12, zmm13;
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
        _cbuf = cbuf; 
        //load c and multiply with beta and 
        //add to accumulator and store back
        zmm3 = _mm512_set1_ps(beta);

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

      // Post Ops
      lpgemm_post_op* post_ops_list_temp = post_ops_list;
      POST_OP_LABEL_LASTK_SAFE_JUMP

POST_OPS_BIAS_6x32F:
      {
        if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
             ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
        {
          zmm1 =
            _mm512_loadu_ps( ( float* )post_ops_list_temp->op_args1 +
              post_ops_attr.post_op_c_j + ( 0 * 16 ) );
          zmm2 =
            _mm512_loadu_ps( ( float* )post_ops_list_temp->op_args1 +
              post_ops_attr.post_op_c_j + ( 1 * 16 ) );

          // c[0,0-15]
          zmm8 = _mm512_add_ps( zmm1, zmm8 );

          // c[0, 16-31]
          zmm9 = _mm512_add_ps( zmm2, zmm9 );

          // c[1,0-15]
          zmm12 = _mm512_add_ps( zmm1, zmm12 );

          // c[1, 16-31]
          zmm13 = _mm512_add_ps( zmm2, zmm13 );

          // c[2,0-15]
          zmm16 = _mm512_add_ps( zmm1, zmm16 );

          // c[2, 16-31]
          zmm17 = _mm512_add_ps( zmm2, zmm17 );

          // c[3,0-15]
          zmm20 = _mm512_add_ps( zmm1, zmm20 );

          // c[3, 16-31]
          zmm21 = _mm512_add_ps( zmm2, zmm21 );

          // c[4,0-15]
          zmm24 = _mm512_add_ps( zmm1, zmm24 );

          // c[4, 16-31]
          zmm25 = _mm512_add_ps( zmm2, zmm25 );

          // c[5,0-15]
          zmm28 = _mm512_add_ps( zmm1, zmm28 );

          // c[5, 16-31]
          zmm29 = _mm512_add_ps( zmm2, zmm29 );
        }
        else
        {
          // If original output was columns major, then by the time
          // kernel sees it, the matrix would be accessed as if it were
          // transposed. Due to this the bias array will be accessed by
          // the ic index, and each bias element corresponds to an
          // entire row of the transposed output array, instead of an
          // entire column.
          zmm1 =
            _mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args1 +
              post_ops_attr.post_op_c_i + 0 ) );
          zmm2 =
            _mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args1 +
              post_ops_attr.post_op_c_i + 1 ) );
          zmm3 =
            _mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args1 +
              post_ops_attr.post_op_c_i + 2 ) );
          zmm4 =
            _mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args1 +
              post_ops_attr.post_op_c_i + 3 ) );
          zmm5 =
            _mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args1 +
              post_ops_attr.post_op_c_i + 4 ) );
          zmm6 =
            _mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args1 +
              post_ops_attr.post_op_c_i + 5 ) );

          // c[0,0-15]
          zmm8 = _mm512_add_ps( zmm1, zmm8 );

          // c[0, 16-31]
          zmm9 = _mm512_add_ps( zmm1, zmm9 );

          // c[1,0-15]
          zmm12 = _mm512_add_ps( zmm2, zmm12 );

          // c[1, 16-31]
          zmm13 = _mm512_add_ps( zmm2, zmm13 );

          // c[2,0-15]
          zmm16 = _mm512_add_ps( zmm3, zmm16 );

          // c[2, 16-31]
          zmm17 = _mm512_add_ps( zmm3, zmm17 );

          // c[3,0-15]
          zmm20 = _mm512_add_ps( zmm4, zmm20 );

          // c[3, 16-31]
          zmm21 = _mm512_add_ps( zmm4, zmm21 );

          // c[4,0-15]
          zmm24 = _mm512_add_ps( zmm5, zmm24 );

          // c[4, 16-31]
          zmm25 = _mm512_add_ps( zmm5, zmm25 );

          // c[5,0-15]
          zmm28 = _mm512_add_ps( zmm6, zmm28 );

          // c[5, 16-31]
          zmm29 = _mm512_add_ps( zmm6, zmm29 );
        }
        POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
      }
POST_OPS_RELU_6x32F:
      {
        zmm1 = _mm512_setzero_ps();

        // c[0,0-15]
        zmm8 = _mm512_max_ps( zmm1, zmm8 );

        // c[0, 16-31]
        zmm9 = _mm512_max_ps( zmm1, zmm9 );

        // c[1,0-15]
        zmm12 = _mm512_max_ps( zmm1, zmm12 );

        // c[1,16-31]
        zmm13 = _mm512_max_ps( zmm1, zmm13 );

        // c[2,0-15]
        zmm16 = _mm512_max_ps( zmm1, zmm16 );

        // c[2,16-31]
        zmm17 = _mm512_max_ps( zmm1, zmm17 );

        // c[3,0-15]
        zmm20 = _mm512_max_ps( zmm1, zmm20 );

        // c[3,16-31]
        zmm21 = _mm512_max_ps( zmm1, zmm21 );

        // c[4,0-15]
        zmm24 = _mm512_max_ps( zmm1, zmm24 );

        // c[4,16-31]
        zmm25 = _mm512_max_ps( zmm1, zmm25 );

        // c[5,0-15]
        zmm28 = _mm512_max_ps( zmm1, zmm28 );

        // c[5,16-31]
        zmm29 = _mm512_max_ps( zmm1, zmm29 );

        POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
      }
POST_OPS_RELU_SCALE_6x32F:
      {
        zmm1 = _mm512_setzero_ps();
        zmm2 =
          _mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args2 ) );

        __mmask16 relu_cmp_mask;

        // c[0, 0-15]
        RELU_SCALE_OP_F32S_AVX512(zmm8)

        // c[0, 16-31]
        RELU_SCALE_OP_F32S_AVX512(zmm9)

        // c[1, 0-15]
        RELU_SCALE_OP_F32S_AVX512(zmm12)

        // c[1, 16-31]
        RELU_SCALE_OP_F32S_AVX512(zmm13)

        // c[2, 0-15]
        RELU_SCALE_OP_F32S_AVX512(zmm16)

        // c[2, 16-31]
        RELU_SCALE_OP_F32S_AVX512(zmm17)

        // c[3, 0-15]
        RELU_SCALE_OP_F32S_AVX512(zmm20)

        // c[3, 16-31]
        RELU_SCALE_OP_F32S_AVX512(zmm21)

        // c[4, 0-15]
        RELU_SCALE_OP_F32S_AVX512(zmm24)

        // c[4, 16-31]
        RELU_SCALE_OP_F32S_AVX512(zmm25)

        // c[5, 0-15]
        RELU_SCALE_OP_F32S_AVX512(zmm28)

        // c[5, 16-31]
        RELU_SCALE_OP_F32S_AVX512(zmm29)

        POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
      }
POST_OPS_GELU_TANH_6x32F:
      {
        __m512i zmm6;
        // c[0, 0-15]
        GELU_TANH_F32S_AVX512(zmm8, zmm0, zmm1, zmm2, zmm3, zmm4, zmm5, zmm6)

        // c[0, 16-31]
        GELU_TANH_F32S_AVX512(zmm9, zmm0, zmm1, zmm2, zmm3, zmm4, zmm5, zmm6)

        // c[1, 0-15]
        GELU_TANH_F32S_AVX512(zmm12, zmm0, zmm1, zmm2, zmm3, zmm4, zmm5, zmm6)

        // c[1, 16-31]
        GELU_TANH_F32S_AVX512(zmm13, zmm0, zmm1, zmm2, zmm3, zmm4, zmm5, zmm6)

        // c[2, 0-15]
        GELU_TANH_F32S_AVX512(zmm16, zmm0, zmm1, zmm2, zmm3, zmm4, zmm5, zmm6)

        // c[2, 16-31]
        GELU_TANH_F32S_AVX512(zmm17, zmm0, zmm1, zmm2, zmm3, zmm4, zmm5, zmm6)

        // c[3, 0-15]
        GELU_TANH_F32S_AVX512(zmm20, zmm0, zmm1, zmm2, zmm3, zmm4, zmm5, zmm6)

        // c[3, 16-31]
        GELU_TANH_F32S_AVX512(zmm21, zmm0, zmm1, zmm2, zmm3, zmm4, zmm5, zmm6)

        // c[4, 0-15]
        GELU_TANH_F32S_AVX512(zmm24, zmm0, zmm1, zmm2, zmm3, zmm4, zmm5, zmm6)

        // c[4, 16-31]
        GELU_TANH_F32S_AVX512(zmm25, zmm0, zmm1, zmm2, zmm3, zmm4, zmm5, zmm6)

        // c[5, 0-15]
        GELU_TANH_F32S_AVX512(zmm28, zmm0, zmm1, zmm2, zmm3, zmm4, zmm5, zmm6)

        // c[5, 16-31]
        GELU_TANH_F32S_AVX512(zmm29, zmm0, zmm1, zmm2, zmm3, zmm4, zmm5, zmm6)

        POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
      }
POST_OPS_GELU_ERF_6x32F:
      {
        // c[0, 0-15]
        GELU_ERF_F32S_AVX512(zmm8, zmm0, zmm1, zmm2)

        // c[0, 16-31]
        GELU_ERF_F32S_AVX512(zmm9, zmm0, zmm1, zmm2)

        // c[1, 0-15]
        GELU_ERF_F32S_AVX512(zmm12, zmm0, zmm1, zmm2)

        // c[1, 16-31]
        GELU_ERF_F32S_AVX512(zmm13, zmm0, zmm1, zmm2)

        // c[2, 0-15]
        GELU_ERF_F32S_AVX512(zmm16, zmm0, zmm1, zmm2)

        // c[2, 16-31]
        GELU_ERF_F32S_AVX512(zmm17, zmm0, zmm1, zmm2)

        // c[3, 0-15]
        GELU_ERF_F32S_AVX512(zmm20, zmm0, zmm1, zmm2)

        // c[3, 16-31]
        GELU_ERF_F32S_AVX512(zmm21, zmm0, zmm1, zmm2)

        // c[4, 0-15]
        GELU_ERF_F32S_AVX512(zmm24, zmm0, zmm1, zmm2)

        // c[4, 16-31]
        GELU_ERF_F32S_AVX512(zmm25, zmm0, zmm1, zmm2)

        // c[5, 0-15]
        GELU_ERF_F32S_AVX512(zmm28, zmm0, zmm1, zmm2)

        // c[5, 16-31]
        GELU_ERF_F32S_AVX512(zmm29, zmm0, zmm1, zmm2)

        POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
      }
POST_OPS_CLIP_6x32F:
      {
        zmm0 = _mm512_set1_ps( *( float* )post_ops_list_temp->op_args2 );
        zmm1 = _mm512_set1_ps( *( float* )post_ops_list_temp->op_args3 );

        // c[0, 0-15]
        CLIP_F32S_AVX512(zmm8, zmm0, zmm1)

        // c[0, 16-31]
        CLIP_F32S_AVX512(zmm9, zmm0, zmm1)

        // c[1, 0-15]
        CLIP_F32S_AVX512(zmm12, zmm0, zmm1)

        // c[1, 16-31]
        CLIP_F32S_AVX512(zmm13, zmm0, zmm1)

        // c[2, 0-15]
        CLIP_F32S_AVX512(zmm16, zmm0, zmm1)

        // c[2, 16-31]
        CLIP_F32S_AVX512(zmm17, zmm0, zmm1)

        // c[3, 0-15]
        CLIP_F32S_AVX512(zmm20, zmm0, zmm1)

        // c[3, 16-31]
        CLIP_F32S_AVX512(zmm21, zmm0, zmm1)

        // c[4, 0-15]
        CLIP_F32S_AVX512(zmm24, zmm0, zmm1)

        // c[4, 16-31]
        CLIP_F32S_AVX512(zmm25, zmm0, zmm1)

        // c[5, 0-15]
        CLIP_F32S_AVX512(zmm28, zmm0, zmm1)

        // c[5, 16-31]
        CLIP_F32S_AVX512(zmm29, zmm0, zmm1)

        POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
      }
POST_OPS_MATRIX_ADD_6x32F:
      {
          dim_t ldm = *( dim_t* )post_ops_list_temp->op_args3;
          float* matptr = ( float* )post_ops_list_temp->op_args1;

          // c[0:0-15,16-31]
          F32_F32_MATRIX_ADD_2COL(zmm1,zmm2,0,8,9);

          // c[1:0-15,16-31]
          F32_F32_MATRIX_ADD_2COL(zmm1,zmm2,1,12,13);

          // c[2:0-15,16-31]
          F32_F32_MATRIX_ADD_2COL(zmm1,zmm2,2,16,17);

          // c[3:0-15,16-31]
          F32_F32_MATRIX_ADD_2COL(zmm1,zmm2,3,20,21);

          // c[4:0-15,16-31]
          F32_F32_MATRIX_ADD_2COL(zmm1,zmm2,4,24,25);

          // c[5:0-15,16-31]
          F32_F32_MATRIX_ADD_2COL(zmm1,zmm2,5,28,29);

          POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
      }
POST_OPS_MATRIX_MUL_6x32F:
      {
          dim_t ldm = *( dim_t* )post_ops_list_temp->op_args3;
          float* matptr = ( float* )post_ops_list_temp->op_args1;

          // c[0:0-15,16-31]
          F32_F32_MATRIX_MUL_2COL(zmm1,zmm2,0,8,9);

          // c[1:0-15,16-31]
          F32_F32_MATRIX_MUL_2COL(zmm1,zmm2,1,12,13);

          // c[2:0-15,16-31]
          F32_F32_MATRIX_MUL_2COL(zmm1,zmm2,2,16,17);

          // c[3:0-15,16-31]
          F32_F32_MATRIX_MUL_2COL(zmm1,zmm2,3,20,21);

          // c[4:0-15,16-31]
          F32_F32_MATRIX_MUL_2COL(zmm1,zmm2,4,24,25);

          // c[5:0-15,16-31]
          F32_F32_MATRIX_MUL_2COL(zmm1,zmm2,5,28,29);

          POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
      }      
POST_OPS_SWISH_6x32F:
      {
          __m512 zmm7 =
              _mm512_set1_ps( *( ( float* )post_ops_list_temp->op_args2 ) );
          __m512i ex_out;

          // c[0, 0-15]
          SWISH_F32_AVX512_DEF(zmm8, zmm7, zmm0, zmm1, zmm2, zmm3, zmm4, ex_out);

          // c[0, 16-31]
          SWISH_F32_AVX512_DEF(zmm9, zmm7, zmm0, zmm1, zmm2, zmm3, zmm4, ex_out);

          // c[1, 0-15]
          SWISH_F32_AVX512_DEF(zmm12, zmm7, zmm0, zmm1, zmm2, zmm3, zmm4, ex_out);

          // c[1, 16-31]
          SWISH_F32_AVX512_DEF(zmm13, zmm7, zmm0, zmm1, zmm2, zmm3, zmm4, ex_out);

          // c[2, 0-15]
          SWISH_F32_AVX512_DEF(zmm16, zmm7, zmm0, zmm1, zmm2, zmm3, zmm4, ex_out);

          // c[2, 16-31]
          SWISH_F32_AVX512_DEF(zmm17, zmm7, zmm0, zmm1, zmm2, zmm3, zmm4, ex_out);

          // c[3, 0-15]
          SWISH_F32_AVX512_DEF(zmm20, zmm7, zmm0, zmm1, zmm2, zmm3, zmm4, ex_out);

          // c[3, 16-31]
          SWISH_F32_AVX512_DEF(zmm21, zmm7, zmm0, zmm1, zmm2, zmm3, zmm4, ex_out);

          // c[4, 0-15]
          SWISH_F32_AVX512_DEF(zmm24, zmm7, zmm0, zmm1, zmm2, zmm3, zmm4, ex_out);

          // c[4, 16-31]
          SWISH_F32_AVX512_DEF(zmm25, zmm7, zmm0, zmm1, zmm2, zmm3, zmm4, ex_out);

          // c[5, 0-15]
          SWISH_F32_AVX512_DEF(zmm28, zmm7, zmm0, zmm1, zmm2, zmm3, zmm4, ex_out);

          // c[5, 16-31]
          SWISH_F32_AVX512_DEF(zmm29, zmm7, zmm0, zmm1, zmm2, zmm3, zmm4, ex_out);

          POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
      }
POST_OPS_6x32F_DISABLE:
      ;

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
          lpgemm_rowvar_f32f32f32of32_avx512_1x32,
          lpgemm_rowvar_f32f32f32of32_avx512_2x32,
          lpgemm_rowvar_f32f32f32of32_avx512_3x32,
          lpgemm_rowvar_f32f32f32of32_avx512_4x32,
          lpgemm_rowvar_f32f32f32of32_avx512_5x32
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
#endif
