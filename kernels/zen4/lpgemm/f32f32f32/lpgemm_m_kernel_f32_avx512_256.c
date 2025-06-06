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
#include "immintrin.h"
#include "xmmintrin.h"
#include "blis.h"

#ifdef BLIS_ADDON_LPGEMM

#include "../../../zen/lpgemm/f32f32f32/lpgemm_kernel_macros_f32_avx2.h"

#define MR 6
#define NR 64


LPGEMM_MAIN_KERN(float,float,float,f32f32f32of32_avx512_256_6x32m)
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
              &&POST_OPS_DOWNSCALE_6x32F,
              &&POST_OPS_MATRIX_ADD_6x32F,
              &&POST_OPS_SWISH_6x32F,
              &&POST_OPS_MATRIX_MUL_6x32F,
              &&POST_OPS_TANH_6x32F,
              &&POST_OPS_SIGMOID_6x32F
            };

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
    __m256 ymm16, ymm17, ymm18, ymm19;
    __m256 ymm20, ymm21, ymm22, ymm23;
    __m256 ymm24, ymm25, ymm26, ymm27;
    __m256 ymm28, ymm29, ymm30, ymm31;

    /*Produce MRxNR outputs */
    for(dim_t m=0; m < m_iter; m++)
    {
        /* zero the accumulator registers */
        ZERO_ACC_YMM_4_REG( ymm8, ymm9, ymm10, ymm11 );
        ZERO_ACC_YMM_4_REG( ymm12, ymm13, ymm14, ymm15 );
        ZERO_ACC_YMM_4_REG( ymm16, ymm17, ymm18, ymm19 );
        ZERO_ACC_YMM_4_REG( ymm20, ymm21, ymm22, ymm23 );
        ZERO_ACC_YMM_4_REG( ymm24, ymm25, ymm26, ymm27 );
        ZERO_ACC_YMM_4_REG( ymm28, ymm29, ymm30, ymm31 );

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
            ymm2 = _mm256_loadu_ps(bbuf + 16);
            ymm3 = _mm256_loadu_ps(bbuf + 24);

            bbuf += rs_b;  //move b pointer to next row

            ymm4 = _mm256_broadcast_ss((abuf + 0*rs_a)); //broadcast c0r0
            ymm5 = _mm256_broadcast_ss((abuf + 1*rs_a)); //broadcast c0r1

            ymm8  = _mm256_fmadd_ps(ymm0, ymm4, ymm8);
            ymm9  = _mm256_fmadd_ps(ymm1, ymm4, ymm9);
            ymm10 = _mm256_fmadd_ps(ymm2, ymm4, ymm10);
            ymm11 = _mm256_fmadd_ps(ymm3, ymm4, ymm11);

            ymm12 = _mm256_fmadd_ps(ymm0, ymm5, ymm12);
            ymm13 = _mm256_fmadd_ps(ymm1, ymm5, ymm13);
            ymm14 = _mm256_fmadd_ps(ymm2, ymm5, ymm14);
            ymm15 = _mm256_fmadd_ps(ymm3, ymm5, ymm15);

            ymm4 = _mm256_broadcast_ss((abuf + 2*rs_a)); //broadcast c0r2
            ymm5 = _mm256_broadcast_ss((abuf + 3*rs_a)); //broadcast c0r3

            ymm16 = _mm256_fmadd_ps(ymm0, ymm4, ymm16);
            ymm17 = _mm256_fmadd_ps(ymm1, ymm4, ymm17);
            ymm18 = _mm256_fmadd_ps(ymm2, ymm4, ymm18);
            ymm19 = _mm256_fmadd_ps(ymm3, ymm4, ymm19);

            ymm20 = _mm256_fmadd_ps(ymm0, ymm5, ymm20);
            ymm21 = _mm256_fmadd_ps(ymm1, ymm5, ymm21);
            ymm22 = _mm256_fmadd_ps(ymm2, ymm5, ymm22);
            ymm23 = _mm256_fmadd_ps(ymm3, ymm5, ymm23);

            ymm4 = _mm256_broadcast_ss((abuf + 4*rs_a)); //broadcast c0r4
            ymm5 = _mm256_broadcast_ss((abuf + 5*rs_a)); //broadcast c0r5
            abuf += cs_a;  //move a pointer to next col

            ymm24 = _mm256_fmadd_ps(ymm0, ymm4, ymm24);
            ymm25 = _mm256_fmadd_ps(ymm1, ymm4, ymm25);
            ymm26 = _mm256_fmadd_ps(ymm2, ymm4, ymm26);
            ymm27 = _mm256_fmadd_ps(ymm3, ymm4, ymm27);

            ymm28 = _mm256_fmadd_ps(ymm0, ymm5, ymm28);
            ymm29 = _mm256_fmadd_ps(ymm1, ymm5, ymm29);
            ymm30 = _mm256_fmadd_ps(ymm2, ymm5, ymm30);
            ymm31 = _mm256_fmadd_ps(ymm3, ymm5, ymm31);
        }//kloop

        ymm0 = _mm256_broadcast_ss(&(alpha));
        ALPHA_MUL_ACC_YMM_4_REG(ymm8,ymm9,ymm10,ymm11,ymm0)
        ALPHA_MUL_ACC_YMM_4_REG(ymm12,ymm13,ymm14,ymm15,ymm0)
        ALPHA_MUL_ACC_YMM_4_REG(ymm16,ymm17,ymm18,ymm19,ymm0)
        ALPHA_MUL_ACC_YMM_4_REG(ymm20,ymm21,ymm22,ymm23,ymm0)
        ALPHA_MUL_ACC_YMM_4_REG(ymm24,ymm25,ymm26,ymm27,ymm0)
        ALPHA_MUL_ACC_YMM_4_REG(ymm28,ymm29,ymm30,ymm31,ymm0)

        if( beta != 0.0 )
        {
            ymm4 = _mm256_broadcast_ss(&(beta));

            if ( ( post_ops_attr.buf_downscale != NULL ) &&
            ( post_ops_attr.is_first_k == TRUE ) )
            {
                BF16_F32_C_BNZ_8(0, 0, ymm0, ymm4, ymm8)
                BF16_F32_C_BNZ_8(0, 1, ymm1, ymm4, ymm9)
                BF16_F32_C_BNZ_8(0, 2, ymm2, ymm4, ymm10)
                BF16_F32_C_BNZ_8(0, 3, ymm3, ymm4, ymm11)

                BF16_F32_C_BNZ_8(1, 0, ymm0, ymm4, ymm12)
                BF16_F32_C_BNZ_8(1, 1, ymm1, ymm4, ymm13)
                BF16_F32_C_BNZ_8(1, 2, ymm2, ymm4, ymm14)
                BF16_F32_C_BNZ_8(1, 3, ymm3, ymm4, ymm15)

                BF16_F32_C_BNZ_8(2, 0, ymm0, ymm4, ymm16)
                BF16_F32_C_BNZ_8(2, 1, ymm1, ymm4, ymm17)
                BF16_F32_C_BNZ_8(2, 2, ymm2, ymm4, ymm18)
                BF16_F32_C_BNZ_8(2, 3, ymm3, ymm4, ymm19)

                BF16_F32_C_BNZ_8(3, 0, ymm0, ymm4, ymm20)
                BF16_F32_C_BNZ_8(3, 1, ymm1, ymm4, ymm21)
                BF16_F32_C_BNZ_8(3, 2, ymm2, ymm4, ymm22)
                BF16_F32_C_BNZ_8(3, 3, ymm3, ymm4, ymm23)

                BF16_F32_C_BNZ_8(4, 0, ymm0, ymm4, ymm24)
                BF16_F32_C_BNZ_8(4, 1, ymm1, ymm4, ymm25)
                BF16_F32_C_BNZ_8(4, 2, ymm2, ymm4, ymm26)
                BF16_F32_C_BNZ_8(4, 3, ymm3, ymm4, ymm27)

                BF16_F32_C_BNZ_8(5, 0, ymm0, ymm4, ymm28)
                BF16_F32_C_BNZ_8(5, 1, ymm1, ymm4, ymm29)
                BF16_F32_C_BNZ_8(5, 2, ymm2, ymm4, ymm30)
                BF16_F32_C_BNZ_8(5, 3, ymm3, ymm4, ymm31)
            }
            else
            {
                _cbuf = cbuf;
                //load c and multiply with beta and
                //add to accumulator and store back
                F32_C_BNZ_8(_cbuf,rs_c,ymm0,ymm4,ymm8)
                F32_C_BNZ_8(_cbuf+8,rs_c,ymm1,ymm4,ymm9)
                F32_C_BNZ_8(_cbuf+16,rs_c,ymm2,ymm4,ymm10)
                F32_C_BNZ_8(_cbuf+24,rs_c,ymm3,ymm4,ymm11)

                _cbuf += rs_c;

                F32_C_BNZ_8(_cbuf,rs_c,ymm0,ymm4,ymm12)
                F32_C_BNZ_8(_cbuf+8,rs_c,ymm1,ymm4,ymm13)
                F32_C_BNZ_8(_cbuf+16,rs_c,ymm2,ymm4,ymm14)
                F32_C_BNZ_8(_cbuf+24,rs_c,ymm3,ymm4,ymm15)

                _cbuf += rs_c;
                F32_C_BNZ_8(_cbuf,rs_c,ymm0,ymm4,ymm16)
                F32_C_BNZ_8(_cbuf+8,rs_c,ymm1,ymm4,ymm17)
                F32_C_BNZ_8(_cbuf+16,rs_c,ymm2,ymm4,ymm18)
                F32_C_BNZ_8(_cbuf+24,rs_c,ymm3,ymm4,ymm19)

                _cbuf += rs_c;

                F32_C_BNZ_8(_cbuf,rs_c,ymm0,ymm4,ymm20)
                F32_C_BNZ_8(_cbuf+8,rs_c,ymm1,ymm4,ymm21)
                F32_C_BNZ_8(_cbuf+16,rs_c,ymm2,ymm4,ymm22)
                F32_C_BNZ_8(_cbuf+24,rs_c,ymm3,ymm4,ymm23)

                _cbuf += rs_c;

                F32_C_BNZ_8(_cbuf,rs_c,ymm0,ymm4,ymm24)
                F32_C_BNZ_8(_cbuf+8,rs_c,ymm1,ymm4,ymm25)
                F32_C_BNZ_8(_cbuf+16,rs_c,ymm2,ymm4,ymm26)
                F32_C_BNZ_8(_cbuf+24,rs_c,ymm3,ymm4,ymm27)

                _cbuf += rs_c;
                F32_C_BNZ_8(_cbuf,rs_c,ymm0,ymm4,ymm28)
                F32_C_BNZ_8(_cbuf+8,rs_c,ymm1,ymm4,ymm29)
                F32_C_BNZ_8(_cbuf+16,rs_c,ymm2,ymm4,ymm30)
                F32_C_BNZ_8(_cbuf+24,rs_c,ymm3,ymm4,ymm31)
            }
        }

        // Post Ops
        lpgemm_post_op* post_ops_list_temp = post_ops_list;
        POST_OP_LABEL_LASTK_SAFE_JUMP

POST_OPS_BIAS_6x32F:
        {
            if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
                ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
            {
                if( post_ops_list_temp->stor_type == BF16 )
                {
                    BF16_F32_BIAS_LOAD_AVX2( ymm0, 0 )
                    BF16_F32_BIAS_LOAD_AVX2( ymm1, 1 )
                    BF16_F32_BIAS_LOAD_AVX2( ymm2, 2 )
                    BF16_F32_BIAS_LOAD_AVX2( ymm3, 3 )
                }
                else
                {
                    ymm0 = _mm256_loadu_ps( ( float* )post_ops_list_temp->op_args1 +
                            post_ops_attr.post_op_c_j + ( 0 * 8 ) );
                    ymm1 = _mm256_loadu_ps( ( float* )post_ops_list_temp->op_args1 +
                            post_ops_attr.post_op_c_j + ( 1 * 8 ) );
                    ymm2 = _mm256_loadu_ps( ( float* )post_ops_list_temp->op_args1 +
                            post_ops_attr.post_op_c_j + ( 2 * 8 ) );
                    ymm3 = _mm256_loadu_ps( ( float* )post_ops_list_temp->op_args1 +
                            post_ops_attr.post_op_c_j + ( 3 * 8 ) );
                }

                ymm8 = _mm256_add_ps(ymm8, ymm0);
                ymm9 = _mm256_add_ps(ymm9, ymm1);
                ymm10 = _mm256_add_ps(ymm10, ymm2);
                ymm11 = _mm256_add_ps(ymm11, ymm3);

                ymm12 = _mm256_add_ps(ymm12, ymm0);
                ymm13 = _mm256_add_ps(ymm13, ymm1);
                ymm14 = _mm256_add_ps(ymm14, ymm2);
                ymm15 = _mm256_add_ps(ymm15, ymm3);

                ymm16 = _mm256_add_ps(ymm16, ymm0);
                ymm17 = _mm256_add_ps(ymm17, ymm1);
                ymm18 = _mm256_add_ps(ymm18, ymm2);
                ymm19 = _mm256_add_ps(ymm19, ymm3);

                ymm20 = _mm256_add_ps(ymm20, ymm0);
                ymm21 = _mm256_add_ps(ymm21, ymm1);
                ymm22 = _mm256_add_ps(ymm22, ymm2);
                ymm23 = _mm256_add_ps(ymm23, ymm3);

                ymm24 = _mm256_add_ps(ymm24, ymm0);
                ymm25 = _mm256_add_ps(ymm25, ymm1);
                ymm26 = _mm256_add_ps(ymm26, ymm2);
                ymm27 = _mm256_add_ps(ymm27, ymm3);

                ymm28 = _mm256_add_ps(ymm28, ymm0);
                ymm29 = _mm256_add_ps(ymm29, ymm1);
                ymm30 = _mm256_add_ps(ymm30, ymm2);
                ymm31 = _mm256_add_ps(ymm31, ymm3);
            }
            else
            {
                if( post_ops_list_temp->stor_type == BF16 )
                {
                    BF16_F32_BIAS_BCAST_AVX2(ymm0,0)
                    BF16_F32_BIAS_BCAST_AVX2(ymm1,1)
                    BF16_F32_BIAS_BCAST_AVX2(ymm2,2)
                    BF16_F32_BIAS_BCAST_AVX2(ymm3,3)
                    BF16_F32_BIAS_BCAST_AVX2(ymm4,4)
                    BF16_F32_BIAS_BCAST_AVX2(ymm5,5)
                }
                else
                {
                    ymm0 = _mm256_broadcast_ss( ( float* )post_ops_list_temp->op_args1 +
                                                post_ops_attr.post_op_c_i + 0 );
                    ymm1 = _mm256_broadcast_ss( ( float* )post_ops_list_temp->op_args1 +
                                                post_ops_attr.post_op_c_i + 1 );
                    ymm2 = _mm256_broadcast_ss( ( float* )post_ops_list_temp->op_args1 +
                                                post_ops_attr.post_op_c_i + 2 );
                    ymm3 = _mm256_broadcast_ss( ( float* )post_ops_list_temp->op_args1 +
                                                post_ops_attr.post_op_c_i + 3 );
                    ymm4 = _mm256_broadcast_ss( ( float* )post_ops_list_temp->op_args1 +
                                                post_ops_attr.post_op_c_i + 4 );
                    ymm5 = _mm256_broadcast_ss( ( float* )post_ops_list_temp->op_args1 +
                                                post_ops_attr.post_op_c_i + 5 );
                }

                ymm8 = _mm256_add_ps(ymm8, ymm0);
                ymm9 = _mm256_add_ps(ymm9, ymm0);
                ymm10 = _mm256_add_ps(ymm10, ymm0);
                ymm11 = _mm256_add_ps(ymm11, ymm0);

                ymm12 = _mm256_add_ps(ymm12, ymm1);
                ymm13 = _mm256_add_ps(ymm13, ymm1);
                ymm14 = _mm256_add_ps(ymm14, ymm1);
                ymm15 = _mm256_add_ps(ymm15, ymm1);

                ymm16 = _mm256_add_ps(ymm16, ymm2);
                ymm17 = _mm256_add_ps(ymm17, ymm2);
                ymm18 = _mm256_add_ps(ymm18, ymm2);
                ymm19 = _mm256_add_ps(ymm19, ymm2);

                ymm20 = _mm256_add_ps(ymm20, ymm3);
                ymm21 = _mm256_add_ps(ymm21, ymm3);
                ymm22 = _mm256_add_ps(ymm22, ymm3);
                ymm23 = _mm256_add_ps(ymm23, ymm3);

                ymm24 = _mm256_add_ps(ymm24, ymm4);
                ymm25 = _mm256_add_ps(ymm25, ymm4);
                ymm26 = _mm256_add_ps(ymm26, ymm4);
                ymm27 = _mm256_add_ps(ymm27, ymm4);

                ymm28 = _mm256_add_ps(ymm28, ymm5);
                ymm29 = _mm256_add_ps(ymm29, ymm5);
                ymm30 = _mm256_add_ps(ymm30, ymm5);
                ymm31 = _mm256_add_ps(ymm31, ymm5);
            }

            POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
        }

POST_OPS_RELU_6x32F:
        {
            ymm0 = _mm256_setzero_ps();

            ymm8 = _mm256_max_ps(ymm8, ymm0);
            ymm9 = _mm256_max_ps(ymm9, ymm0);
            ymm10 = _mm256_max_ps(ymm10, ymm0);
            ymm11 = _mm256_max_ps(ymm11, ymm0);

            ymm12 = _mm256_max_ps(ymm12, ymm0);
            ymm13 = _mm256_max_ps(ymm13, ymm0);
            ymm14 = _mm256_max_ps(ymm14, ymm0);
            ymm15 = _mm256_max_ps(ymm15, ymm0);

            ymm16 = _mm256_max_ps(ymm16, ymm0);
            ymm17 = _mm256_max_ps(ymm17, ymm0);
            ymm18 = _mm256_max_ps(ymm18, ymm0);
            ymm19 = _mm256_max_ps(ymm19, ymm0);

            ymm20 = _mm256_max_ps(ymm20, ymm0);
            ymm21 = _mm256_max_ps(ymm21, ymm0);
            ymm22 = _mm256_max_ps(ymm22, ymm0);
            ymm23 = _mm256_max_ps(ymm23, ymm0);

            ymm24 = _mm256_max_ps(ymm24, ymm0);
            ymm25 = _mm256_max_ps(ymm25, ymm0);
            ymm26 = _mm256_max_ps(ymm26, ymm0);
            ymm27 = _mm256_max_ps(ymm27, ymm0);

            ymm28 = _mm256_max_ps(ymm28, ymm0);
            ymm29 = _mm256_max_ps(ymm29, ymm0);
            ymm30 = _mm256_max_ps(ymm30, ymm0);
            ymm31 = _mm256_max_ps(ymm31, ymm0);

            POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
        }

POST_OPS_RELU_SCALE_6x32F:
        {
            ymm0 = _mm256_broadcast_ss( ( float* )post_ops_list_temp->op_args2 );
            ymm1 = _mm256_setzero_ps();

            RELU_SCALE_OP_F32S_AVX2(ymm8, ymm0, ymm1, ymm2)
            RELU_SCALE_OP_F32S_AVX2(ymm9, ymm0, ymm1, ymm3)
            RELU_SCALE_OP_F32S_AVX2(ymm10, ymm0, ymm1, ymm4)
            RELU_SCALE_OP_F32S_AVX2(ymm11, ymm0, ymm1, ymm5)

            RELU_SCALE_OP_F32S_AVX2(ymm12, ymm0, ymm1, ymm2)
            RELU_SCALE_OP_F32S_AVX2(ymm13, ymm0, ymm1, ymm3)
            RELU_SCALE_OP_F32S_AVX2(ymm14, ymm0, ymm1, ymm4)
            RELU_SCALE_OP_F32S_AVX2(ymm15, ymm0, ymm1, ymm5)

            RELU_SCALE_OP_F32S_AVX2(ymm16, ymm0, ymm1, ymm2)
            RELU_SCALE_OP_F32S_AVX2(ymm17, ymm0, ymm1, ymm3)
            RELU_SCALE_OP_F32S_AVX2(ymm18, ymm0, ymm1, ymm4)
            RELU_SCALE_OP_F32S_AVX2(ymm19, ymm0, ymm1, ymm5)

            RELU_SCALE_OP_F32S_AVX2(ymm20, ymm0, ymm1, ymm2)
            RELU_SCALE_OP_F32S_AVX2(ymm21, ymm0, ymm1, ymm3)
            RELU_SCALE_OP_F32S_AVX2(ymm22, ymm0, ymm1, ymm4)
            RELU_SCALE_OP_F32S_AVX2(ymm23, ymm0, ymm1, ymm5)

            RELU_SCALE_OP_F32S_AVX2(ymm24, ymm0, ymm1, ymm2)
            RELU_SCALE_OP_F32S_AVX2(ymm25, ymm0, ymm1, ymm3)
            RELU_SCALE_OP_F32S_AVX2(ymm26, ymm0, ymm1, ymm4)
            RELU_SCALE_OP_F32S_AVX2(ymm27, ymm0, ymm1, ymm5)

            RELU_SCALE_OP_F32S_AVX2(ymm28, ymm0, ymm1, ymm2)
            RELU_SCALE_OP_F32S_AVX2(ymm29, ymm0, ymm1, ymm3)
            RELU_SCALE_OP_F32S_AVX2(ymm30, ymm0, ymm1, ymm4)
            RELU_SCALE_OP_F32S_AVX2(ymm31, ymm0, ymm1, ymm5)

            POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
        }
POST_OPS_GELU_TANH_6x32F:
        {
            __m256 dn, x_tanh;
            __m256i q;

            GELU_TANH_F32S_AVX2(ymm8, ymm0, ymm1, ymm2, ymm3, dn, x_tanh, q)
            GELU_TANH_F32S_AVX2(ymm9, ymm0, ymm1, ymm2, ymm3, dn, x_tanh, q)
            GELU_TANH_F32S_AVX2(ymm10, ymm0, ymm1, ymm2, ymm3, dn, x_tanh, q)
            GELU_TANH_F32S_AVX2(ymm11, ymm0, ymm1, ymm2, ymm3, dn, x_tanh, q)

            GELU_TANH_F32S_AVX2(ymm12, ymm0, ymm1, ymm2, ymm3, dn, x_tanh, q)
            GELU_TANH_F32S_AVX2(ymm13, ymm0, ymm1, ymm2, ymm3, dn, x_tanh, q)
            GELU_TANH_F32S_AVX2(ymm14, ymm0, ymm1, ymm2, ymm3, dn, x_tanh, q)
            GELU_TANH_F32S_AVX2(ymm15, ymm0, ymm1, ymm2, ymm3, dn, x_tanh, q)

            GELU_TANH_F32S_AVX2(ymm16, ymm0, ymm1, ymm2, ymm3, dn, x_tanh, q)
            GELU_TANH_F32S_AVX2(ymm17, ymm0, ymm1, ymm2, ymm3, dn, x_tanh, q)
            GELU_TANH_F32S_AVX2(ymm18, ymm0, ymm1, ymm2, ymm3, dn, x_tanh, q)
            GELU_TANH_F32S_AVX2(ymm19, ymm0, ymm1, ymm2, ymm3, dn, x_tanh, q)

            GELU_TANH_F32S_AVX2(ymm20, ymm0, ymm1, ymm2, ymm3, dn, x_tanh, q)
            GELU_TANH_F32S_AVX2(ymm21, ymm0, ymm1, ymm2, ymm3, dn, x_tanh, q)
            GELU_TANH_F32S_AVX2(ymm22, ymm0, ymm1, ymm2, ymm3, dn, x_tanh, q)
            GELU_TANH_F32S_AVX2(ymm23, ymm0, ymm1, ymm2, ymm3, dn, x_tanh, q)

            GELU_TANH_F32S_AVX2(ymm24, ymm0, ymm1, ymm2, ymm3, dn, x_tanh, q)
            GELU_TANH_F32S_AVX2(ymm25, ymm0, ymm1, ymm2, ymm3, dn, x_tanh, q)
            GELU_TANH_F32S_AVX2(ymm26, ymm0, ymm1, ymm2, ymm3, dn, x_tanh, q)
            GELU_TANH_F32S_AVX2(ymm27, ymm0, ymm1, ymm2, ymm3, dn, x_tanh, q)

            GELU_TANH_F32S_AVX2(ymm28, ymm0, ymm1, ymm2, ymm3, dn, x_tanh, q)
            GELU_TANH_F32S_AVX2(ymm29, ymm0, ymm1, ymm2, ymm3, dn, x_tanh, q)
            GELU_TANH_F32S_AVX2(ymm30, ymm0, ymm1, ymm2, ymm3, dn, x_tanh, q)
            GELU_TANH_F32S_AVX2(ymm31, ymm0, ymm1, ymm2, ymm3, dn, x_tanh, q)

            POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
        }
POST_OPS_GELU_ERF_6x32F:
        {
            GELU_ERF_F32S_AVX2(ymm8, ymm0, ymm1, ymm2)
            GELU_ERF_F32S_AVX2(ymm9, ymm0, ymm1, ymm2)
            GELU_ERF_F32S_AVX2(ymm10, ymm0, ymm1, ymm2)
            GELU_ERF_F32S_AVX2(ymm11, ymm0, ymm1, ymm2)

            GELU_ERF_F32S_AVX2(ymm12, ymm0, ymm1, ymm2)
            GELU_ERF_F32S_AVX2(ymm13, ymm0, ymm1, ymm2)
            GELU_ERF_F32S_AVX2(ymm14, ymm0, ymm1, ymm2)
            GELU_ERF_F32S_AVX2(ymm15, ymm0, ymm1, ymm2)

            GELU_ERF_F32S_AVX2(ymm16, ymm0, ymm1, ymm2)
            GELU_ERF_F32S_AVX2(ymm17, ymm0, ymm1, ymm2)
            GELU_ERF_F32S_AVX2(ymm18, ymm0, ymm1, ymm2)
            GELU_ERF_F32S_AVX2(ymm19, ymm0, ymm1, ymm2)

            GELU_ERF_F32S_AVX2(ymm20, ymm0, ymm1, ymm2)
            GELU_ERF_F32S_AVX2(ymm21, ymm0, ymm1, ymm2)
            GELU_ERF_F32S_AVX2(ymm22, ymm0, ymm1, ymm2)
            GELU_ERF_F32S_AVX2(ymm23, ymm0, ymm1, ymm2)

            GELU_ERF_F32S_AVX2(ymm24, ymm0, ymm1, ymm2)
            GELU_ERF_F32S_AVX2(ymm25, ymm0, ymm1, ymm2)
            GELU_ERF_F32S_AVX2(ymm26, ymm0, ymm1, ymm2)
            GELU_ERF_F32S_AVX2(ymm27, ymm0, ymm1, ymm2)

            GELU_ERF_F32S_AVX2(ymm28, ymm0, ymm1, ymm2)
            GELU_ERF_F32S_AVX2(ymm29, ymm0, ymm1, ymm2)
            GELU_ERF_F32S_AVX2(ymm30, ymm0, ymm1, ymm2)
            GELU_ERF_F32S_AVX2(ymm31, ymm0, ymm1, ymm2)

            POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
        }
POST_OPS_CLIP_6x32F:
        {
            ymm0 = _mm256_set1_ps( *( float* )post_ops_list_temp->op_args2 );
            ymm1 = _mm256_set1_ps( *( float* )post_ops_list_temp->op_args3 );

            CLIP_F32S_AVX2(ymm8, ymm0, ymm1)
            CLIP_F32S_AVX2(ymm9, ymm0, ymm1)
            CLIP_F32S_AVX2(ymm10, ymm0, ymm1)
            CLIP_F32S_AVX2(ymm11, ymm0, ymm1)

            CLIP_F32S_AVX2(ymm12, ymm0, ymm1)
            CLIP_F32S_AVX2(ymm13, ymm0, ymm1)
            CLIP_F32S_AVX2(ymm14, ymm0, ymm1)
            CLIP_F32S_AVX2(ymm15, ymm0, ymm1)

            CLIP_F32S_AVX2(ymm16, ymm0, ymm1)
            CLIP_F32S_AVX2(ymm17, ymm0, ymm1)
            CLIP_F32S_AVX2(ymm18, ymm0, ymm1)
            CLIP_F32S_AVX2(ymm19, ymm0, ymm1)

            CLIP_F32S_AVX2(ymm20, ymm0, ymm1)
            CLIP_F32S_AVX2(ymm21, ymm0, ymm1)
            CLIP_F32S_AVX2(ymm22, ymm0, ymm1)
            CLIP_F32S_AVX2(ymm23, ymm0, ymm1)

            CLIP_F32S_AVX2(ymm24, ymm0, ymm1)
            CLIP_F32S_AVX2(ymm25, ymm0, ymm1)
            CLIP_F32S_AVX2(ymm26, ymm0, ymm1)
            CLIP_F32S_AVX2(ymm27, ymm0, ymm1)

            CLIP_F32S_AVX2(ymm28, ymm0, ymm1)
            CLIP_F32S_AVX2(ymm29, ymm0, ymm1)
            CLIP_F32S_AVX2(ymm30, ymm0, ymm1)
            CLIP_F32S_AVX2(ymm31, ymm0, ymm1)

            POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
        }
POST_OPS_DOWNSCALE_6x32F:
        {
            __m256 selector1 = _mm256_setzero_ps();
            __m256 selector2 = _mm256_setzero_ps();
            __m256 selector3 = _mm256_setzero_ps();
            __m256 selector4 = _mm256_setzero_ps();
            __m256 selector5 = _mm256_setzero_ps();
            __m256 selector6 = _mm256_setzero_ps();

            __m256 zero_point0 = _mm256_setzero_ps();
            __m256 zero_point1 = _mm256_setzero_ps();
            __m256 zero_point2 = _mm256_setzero_ps();
            __m256 zero_point3 = _mm256_setzero_ps();
            __m256 zero_point4 = _mm256_setzero_ps();
            __m256 zero_point5 = _mm256_setzero_ps();

            bool is_bf16 = ( post_ops_list_temp->stor_type == BF16 ) ||
                    ( ( post_ops_list_temp->stor_type == NONE ) &&
                      ( post_ops_attr.c_stor_type == BF16 ) );

            if( post_ops_list_temp->scale_factor_len == 1 )
            {
                selector1 =
                        _mm256_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
                selector2 =
                        _mm256_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
                selector3 =
                        _mm256_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
                selector4 =
                        _mm256_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
                selector5 =
                        _mm256_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
                selector6 =
                        _mm256_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
            }

            if( *( (dim_t* )post_ops_list_temp->op_args3 ) == 1 )
            {
                if( is_bf16 == TRUE )
                {
                    BF16_F32_ZP_SCALAR_BCAST_AVX2(zero_point0)
                    BF16_F32_ZP_SCALAR_BCAST_AVX2(zero_point1)
                    BF16_F32_ZP_SCALAR_BCAST_AVX2(zero_point2)
                    BF16_F32_ZP_SCALAR_BCAST_AVX2(zero_point3)
                    BF16_F32_ZP_SCALAR_BCAST_AVX2(zero_point4)
                    BF16_F32_ZP_SCALAR_BCAST_AVX2(zero_point5)
                }
                else
                {
                    zero_point0 = _mm256_set1_ps( *( ( float* )post_ops_list_temp->op_args1 ) );
                    zero_point1 = _mm256_set1_ps( *( ( float* )post_ops_list_temp->op_args1 ) );
                    zero_point2 = _mm256_set1_ps( *( ( float* )post_ops_list_temp->op_args1 ) );
                    zero_point3 = _mm256_set1_ps( *( ( float* )post_ops_list_temp->op_args1 ) );
                    zero_point4 = _mm256_set1_ps( *( ( float* )post_ops_list_temp->op_args1 ) );
                    zero_point5 = _mm256_set1_ps( *( ( float* )post_ops_list_temp->op_args1 ) );
                }
            }

            if( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
            ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
            {
                if( post_ops_list_temp->scale_factor_len > 1 )
                {
                    selector1 = _mm256_loadu_ps ( ( float* )post_ops_list_temp->scale_factor +
                                post_ops_attr.post_op_c_j + ( 0 * 8 ) );
                    selector2 = _mm256_loadu_ps ( ( float* )post_ops_list_temp->scale_factor +
                                post_ops_attr.post_op_c_j + ( 1 * 8 ) );
                    selector3 = _mm256_loadu_ps ( ( float* )post_ops_list_temp->scale_factor +
                                post_ops_attr.post_op_c_j + ( 2 * 8 ) );
                    selector4 = _mm256_loadu_ps ( ( float* )post_ops_list_temp->scale_factor +
                                post_ops_attr.post_op_c_j + ( 3 * 8 ) );
                }
                if ( *( ( dim_t* )post_ops_list_temp->op_args3 ) > 1 )
                {
                    if( is_bf16 == TRUE )
                    {
                        BF16_F32_ZP_VECTOR_LOAD_AVX2(zero_point0,0)
                        BF16_F32_ZP_VECTOR_LOAD_AVX2(zero_point1,1)
                        BF16_F32_ZP_VECTOR_LOAD_AVX2(zero_point2,2)
                        BF16_F32_ZP_VECTOR_LOAD_AVX2(zero_point3,3)
                    }
                    else
                    {
                        zero_point0 = _mm256_loadu_ps ( (float* )post_ops_list_temp->op_args1 +
                                            post_ops_attr.post_op_c_j + ( 0 * 8 ) );
                        zero_point1 = _mm256_loadu_ps ( (float* )post_ops_list_temp->op_args1 +
                                            post_ops_attr.post_op_c_j + ( 1 * 8 ) );
                        zero_point2 = _mm256_loadu_ps ( (float* )post_ops_list_temp->op_args1 +
                                            post_ops_attr.post_op_c_j + ( 2 * 8 ) );
                        zero_point3 = _mm256_loadu_ps ( (float* )post_ops_list_temp->op_args1 +
                                            post_ops_attr.post_op_c_j + ( 3 * 8 ) );
                    }
                }
                F32_SCL_MULRND_AVX2(ymm8, selector1, zero_point0)
                F32_SCL_MULRND_AVX2(ymm9, selector2, zero_point1)
                F32_SCL_MULRND_AVX2(ymm10, selector3, zero_point2)
                F32_SCL_MULRND_AVX2(ymm11, selector4, zero_point3)

                F32_SCL_MULRND_AVX2(ymm12, selector1, zero_point0)
                F32_SCL_MULRND_AVX2(ymm13, selector2, zero_point1)
                F32_SCL_MULRND_AVX2(ymm14, selector3, zero_point2)
                F32_SCL_MULRND_AVX2(ymm15, selector4, zero_point3)

                F32_SCL_MULRND_AVX2(ymm16, selector1, zero_point0)
                F32_SCL_MULRND_AVX2(ymm17, selector2, zero_point1)
                F32_SCL_MULRND_AVX2(ymm18, selector3, zero_point2)
                F32_SCL_MULRND_AVX2(ymm19, selector4, zero_point3)

                F32_SCL_MULRND_AVX2(ymm20, selector1, zero_point0)
                F32_SCL_MULRND_AVX2(ymm21, selector2, zero_point1)
                F32_SCL_MULRND_AVX2(ymm22, selector3, zero_point2)
                F32_SCL_MULRND_AVX2(ymm23, selector4, zero_point3)

                F32_SCL_MULRND_AVX2(ymm24, selector1, zero_point0)
                F32_SCL_MULRND_AVX2(ymm25, selector2, zero_point1)
                F32_SCL_MULRND_AVX2(ymm26, selector3, zero_point2)
                F32_SCL_MULRND_AVX2(ymm27, selector4, zero_point3)

                F32_SCL_MULRND_AVX2(ymm28, selector1, zero_point0)
                F32_SCL_MULRND_AVX2(ymm29, selector2, zero_point1)
                F32_SCL_MULRND_AVX2(ymm30, selector3, zero_point2)
                F32_SCL_MULRND_AVX2(ymm31, selector4, zero_point3)
            }
            else
            {
                if( post_ops_list_temp->scale_factor_len > 1 )
                {
                    selector1 =
                        _mm256_set1_ps( *( (float* )post_ops_list_temp->scale_factor +
                                        post_ops_attr.post_op_c_i + 0 ) );
                    selector2 =
                        _mm256_set1_ps( *( (float* )post_ops_list_temp->scale_factor +
                                        post_ops_attr.post_op_c_i + 1 ) );
                    selector3 =
                        _mm256_set1_ps( *( (float* )post_ops_list_temp->scale_factor +
                                        post_ops_attr.post_op_c_i + 2 ) );
                    selector4 =
                        _mm256_set1_ps( *( (float* )post_ops_list_temp->scale_factor +
                                        post_ops_attr.post_op_c_i + 3 ) );
                    selector5 =
                        _mm256_set1_ps( *( (float* )post_ops_list_temp->scale_factor +
                                        post_ops_attr.post_op_c_i + 4 ) );
                    selector6 =
                        _mm256_set1_ps( *( (float* )post_ops_list_temp->scale_factor +
                                        post_ops_attr.post_op_c_i + 5 ) );
                }
                if( *( ( dim_t* )post_ops_list_temp->op_args3 ) > 1 )
                {
                    if( is_bf16 == TRUE )
                    {
                        BF16_F32_ZP_VECTOR_BCAST_AVX2(zero_point0,0)
                        BF16_F32_ZP_VECTOR_BCAST_AVX2(zero_point1,1)
                        BF16_F32_ZP_VECTOR_BCAST_AVX2(zero_point2,2)
                        BF16_F32_ZP_VECTOR_BCAST_AVX2(zero_point3,3)
                        BF16_F32_ZP_VECTOR_BCAST_AVX2(zero_point4,4)
                        BF16_F32_ZP_VECTOR_BCAST_AVX2(zero_point5,5)
                    }
                    else
                    {
                        zero_point0 = _mm256_set1_ps( *( ( float* )post_ops_list_temp->op_args1 +
                                        post_ops_attr.post_op_c_i + 0 ) );
                        zero_point1 = _mm256_set1_ps( *( ( float* )post_ops_list_temp->op_args1 +
                                        post_ops_attr.post_op_c_i + 1) );
                        zero_point2 = _mm256_set1_ps( *( ( float* )post_ops_list_temp->op_args1 +
                                        post_ops_attr.post_op_c_i + 2 ) );
                        zero_point3 = _mm256_set1_ps( *( ( float* )post_ops_list_temp->op_args1 +
                                        post_ops_attr.post_op_c_i + 3 ) );
                        zero_point4 = _mm256_set1_ps( *( ( float* )post_ops_list_temp->op_args1 +
                                        post_ops_attr.post_op_c_i + 4 ) );
                        zero_point5 = _mm256_set1_ps( *( ( float* )post_ops_list_temp->op_args1 +
                                        post_ops_attr.post_op_c_i + 5 ) );
                    }
                }

                F32_SCL_MULRND_AVX2(ymm8, selector1, zero_point0)
                F32_SCL_MULRND_AVX2(ymm9, selector1, zero_point0)
                F32_SCL_MULRND_AVX2(ymm10, selector1, zero_point0)
                F32_SCL_MULRND_AVX2(ymm11, selector1, zero_point0)

                F32_SCL_MULRND_AVX2(ymm12, selector2, zero_point1)
                F32_SCL_MULRND_AVX2(ymm13, selector2, zero_point1)
                F32_SCL_MULRND_AVX2(ymm14, selector2, zero_point1)
                F32_SCL_MULRND_AVX2(ymm15, selector2, zero_point1)

                F32_SCL_MULRND_AVX2(ymm16, selector3, zero_point2)
                F32_SCL_MULRND_AVX2(ymm17, selector3, zero_point2)
                F32_SCL_MULRND_AVX2(ymm18, selector3, zero_point2)
                F32_SCL_MULRND_AVX2(ymm19, selector3, zero_point2)

                F32_SCL_MULRND_AVX2(ymm20, selector4, zero_point3)
                F32_SCL_MULRND_AVX2(ymm21, selector4, zero_point3)
                F32_SCL_MULRND_AVX2(ymm22, selector4, zero_point3)
                F32_SCL_MULRND_AVX2(ymm23, selector4, zero_point3)

                F32_SCL_MULRND_AVX2(ymm24, selector5, zero_point4)
                F32_SCL_MULRND_AVX2(ymm25, selector5, zero_point4)
                F32_SCL_MULRND_AVX2(ymm26, selector5, zero_point4)
                F32_SCL_MULRND_AVX2(ymm27, selector5, zero_point4)

                F32_SCL_MULRND_AVX2(ymm28, selector6, zero_point5)
                F32_SCL_MULRND_AVX2(ymm29, selector6, zero_point5)
                F32_SCL_MULRND_AVX2(ymm30, selector6, zero_point5)
                F32_SCL_MULRND_AVX2(ymm31, selector6, zero_point5)
            }
            POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
        }
POST_OPS_MATRIX_ADD_6x32F:
        {
            dim_t ldm = *( dim_t* )post_ops_list_temp->op_args3;

            __m256 scl_fctr1 = _mm256_setzero_ps();
            __m256 scl_fctr2 = _mm256_setzero_ps();
            __m256 scl_fctr3 = _mm256_setzero_ps();
            __m256 scl_fctr4 = _mm256_setzero_ps();
            __m256 scl_fctr5 = _mm256_setzero_ps();
            __m256 scl_fctr6 = _mm256_setzero_ps();

            bool is_bf16 = ( post_ops_list_temp->stor_type == BF16 ) ||
                ( ( post_ops_list_temp->stor_type == NONE ) &&
                ( post_ops_attr.c_stor_type == BF16 ) );

            // Even though different registers are used for scalar in column and
            // row major case, all those registers will contain the same value.
            if ( post_ops_list_temp->scale_factor_len == 1 )
            {
                scl_fctr1 =
                    _mm256_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
                scl_fctr2 =
                    _mm256_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
                scl_fctr3 =
                    _mm256_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
                scl_fctr4 =
                    _mm256_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
                scl_fctr5 =
                    _mm256_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
                scl_fctr6 =
                    _mm256_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
            }
            else
            {
                if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
                    ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
                {
                    scl_fctr1 =
                    _mm256_loadu_ps( ( float* )post_ops_list_temp->scale_factor +
                        post_ops_attr.post_op_c_j + ( 0 * 16 ) );
                    scl_fctr2 =
                    _mm256_loadu_ps( ( float* )post_ops_list_temp->scale_factor +
                        post_ops_attr.post_op_c_j + ( 1 * 16 ) );
                    scl_fctr3 =
                    _mm256_loadu_ps( ( float* )post_ops_list_temp->scale_factor +
                        post_ops_attr.post_op_c_j + ( 2 * 16 ) );
                    scl_fctr4 =
                    _mm256_loadu_ps( ( float* )post_ops_list_temp->scale_factor +
                        post_ops_attr.post_op_c_j + ( 3 * 16 ) );
                }
                else
                {
                    scl_fctr1 =
                    _mm256_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
                        post_ops_attr.post_op_c_i + 0 ) );
                    scl_fctr2 =
                    _mm256_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
                        post_ops_attr.post_op_c_i + 1 ) );
                    scl_fctr3 =
                    _mm256_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
                        post_ops_attr.post_op_c_i + 2 ) );
                    scl_fctr4 =
                    _mm256_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
                        post_ops_attr.post_op_c_i + 3 ) );
                    scl_fctr5 =
                    _mm256_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
                        post_ops_attr.post_op_c_i + 4 ) );
                    scl_fctr6 =
                    _mm256_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
                        post_ops_attr.post_op_c_i + 5 ) );
                }
            }
            if ( is_bf16 == TRUE )
            {
                bfloat16* matptr = ( bfloat16* )post_ops_list_temp->op_args1;

                if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
                ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
                {
                    BF16_F32_MATRIX_ADD_4COL_YMM(ymm0,ymm1,ymm2,ymm3,scl_fctr1,
                                                scl_fctr2,scl_fctr3,scl_fctr4,
                                                0,8,9,10,11);
                    BF16_F32_MATRIX_ADD_4COL_YMM(ymm4,ymm5,ymm6,ymm7,scl_fctr1,
                                                scl_fctr2,scl_fctr3,scl_fctr4,
                                                1,12,13,14,15);
                    BF16_F32_MATRIX_ADD_4COL_YMM(ymm0,ymm1,ymm2,ymm3,scl_fctr1,
                                                scl_fctr2,scl_fctr3,scl_fctr4,
                                                2,16,17,18,19);
                    BF16_F32_MATRIX_ADD_4COL_YMM(ymm4,ymm5,ymm6,ymm7,scl_fctr1,
                                                scl_fctr2,scl_fctr3,scl_fctr4,
                                                3,20,21,22,23);
                    BF16_F32_MATRIX_ADD_4COL_YMM(ymm0,ymm1,ymm2,ymm3,scl_fctr1,
                                                scl_fctr2,scl_fctr3,scl_fctr4,
                                                4,24,25,26,27);
                    BF16_F32_MATRIX_ADD_4COL_YMM(ymm4,ymm5,ymm6,ymm7,scl_fctr1,
                                                scl_fctr2,scl_fctr3,scl_fctr4,
                                                5,28,29,30,31);
                }
                else
                {
                    BF16_F32_MATRIX_ADD_4COL_YMM(ymm0,ymm1,ymm2,ymm3,scl_fctr1,
                                                scl_fctr1,scl_fctr1,scl_fctr1,
                                                0,8,9,10,11);
                    BF16_F32_MATRIX_ADD_4COL_YMM(ymm4,ymm5,ymm6,ymm7,scl_fctr2,
                                                scl_fctr2,scl_fctr2,scl_fctr2,
                                                1,12,13,14,15);
                    BF16_F32_MATRIX_ADD_4COL_YMM(ymm0,ymm1,ymm2,ymm3,scl_fctr3,
                                                scl_fctr3,scl_fctr3,scl_fctr3,
                                                2,16,17,18,19);
                    BF16_F32_MATRIX_ADD_4COL_YMM(ymm4,ymm5,ymm6,ymm7,scl_fctr4,
                                                scl_fctr4,scl_fctr4,scl_fctr4,
                                                3,20,21,22,23);
                    BF16_F32_MATRIX_ADD_4COL_YMM(ymm0,ymm1,ymm2,ymm3,scl_fctr5,
                                                scl_fctr5,scl_fctr5,scl_fctr5,
                                                4,24,25,26,27);
                    BF16_F32_MATRIX_ADD_4COL_YMM(ymm4,ymm5,ymm6,ymm7,scl_fctr6,
                                                scl_fctr6,scl_fctr6,scl_fctr6,
                                                5,28,29,30,31);
                }
            }
            else
            {
                float* matptr = ( float* )post_ops_list_temp->op_args1;

                if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
                ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
                {
                    F32_F32_MATRIX_ADD_4COL_YMM(ymm0,ymm1,ymm2,ymm3,scl_fctr1,
                                                scl_fctr2,scl_fctr3,scl_fctr4,
                                                0,8,9,10,11);
                    F32_F32_MATRIX_ADD_4COL_YMM(ymm4,ymm5,ymm6,ymm7,scl_fctr1,
                                                scl_fctr2,scl_fctr3,scl_fctr4,
                                                1,12,13,14,15);
                    F32_F32_MATRIX_ADD_4COL_YMM(ymm0,ymm1,ymm2,ymm3,scl_fctr1,
                                                scl_fctr2,scl_fctr3,scl_fctr4,
                                                2,16,17,18,19);
                    F32_F32_MATRIX_ADD_4COL_YMM(ymm4,ymm5,ymm6,ymm7,scl_fctr1,
                                                scl_fctr2,scl_fctr3,scl_fctr4,
                                                3,20,21,22,23);
                    F32_F32_MATRIX_ADD_4COL_YMM(ymm0,ymm1,ymm2,ymm3,scl_fctr1,
                                                scl_fctr2,scl_fctr3,scl_fctr4,
                                                4,24,25,26,27);
                    F32_F32_MATRIX_ADD_4COL_YMM(ymm4,ymm5,ymm6,ymm7,scl_fctr1,
                                                scl_fctr2,scl_fctr3,scl_fctr4,
                                                5,28,29,30,31);
                }
                else
                {
                    F32_F32_MATRIX_ADD_4COL_YMM(ymm0,ymm1,ymm2,ymm3,scl_fctr1,
                                                scl_fctr1,scl_fctr1,scl_fctr1,
                                                0,8,9,10,11);
                    F32_F32_MATRIX_ADD_4COL_YMM(ymm4,ymm5,ymm6,ymm7,scl_fctr2,
                                                scl_fctr2,scl_fctr2,scl_fctr2,
                                                1,12,13,14,15);
                    F32_F32_MATRIX_ADD_4COL_YMM(ymm0,ymm1,ymm2,ymm3,scl_fctr3,
                                                scl_fctr3,scl_fctr3,scl_fctr3,
                                                2,16,17,18,19);
                    F32_F32_MATRIX_ADD_4COL_YMM(ymm4,ymm5,ymm6,ymm7,scl_fctr4,
                                                scl_fctr4,scl_fctr4,scl_fctr4,
                                                3,20,21,22,23);
                    F32_F32_MATRIX_ADD_4COL_YMM(ymm0,ymm1,ymm2,ymm3,scl_fctr5,
                                                scl_fctr5,scl_fctr5,scl_fctr5,
                                                4,24,25,26,27);
                    F32_F32_MATRIX_ADD_4COL_YMM(ymm4,ymm5,ymm6,ymm7,scl_fctr6,
                                                scl_fctr6,scl_fctr6,scl_fctr6,
                                                5,28,29,30,31);
                }
            }
            POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
        }
POST_OPS_MATRIX_MUL_6x32F:
        {
            dim_t ldm = *( dim_t* )post_ops_list_temp->op_args3;

            __m256 scl_fctr1 = _mm256_setzero_ps();
            __m256 scl_fctr2 = _mm256_setzero_ps();
            __m256 scl_fctr3 = _mm256_setzero_ps();
            __m256 scl_fctr4 = _mm256_setzero_ps();
            __m256 scl_fctr5 = _mm256_setzero_ps();
            __m256 scl_fctr6 = _mm256_setzero_ps();

            bool is_bf16 = ( post_ops_list_temp->stor_type == BF16 ) ||
                ( ( post_ops_list_temp->stor_type == NONE ) &&
                    ( post_ops_attr.c_stor_type == BF16 ) );

            // Even though different registers are used for scalar in column and
            // row major case, all those registers will contain the same value.
            if ( post_ops_list_temp->scale_factor_len == 1 )
            {
                scl_fctr1 =
                    _mm256_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
                scl_fctr2 =
                    _mm256_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
                scl_fctr3 =
                    _mm256_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
                scl_fctr4 =
                    _mm256_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
                scl_fctr5 =
                    _mm256_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
                scl_fctr6 =
                    _mm256_set1_ps( *( ( float* )post_ops_list_temp->scale_factor ) );
            }
            else
            {
                if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
                    ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
                {
                    scl_fctr1 =
                    _mm256_loadu_ps( ( float* )post_ops_list_temp->scale_factor +
                        post_ops_attr.post_op_c_j + ( 0 * 16 ) );
                    scl_fctr2 =
                    _mm256_loadu_ps( ( float* )post_ops_list_temp->scale_factor +
                        post_ops_attr.post_op_c_j + ( 1 * 16 ) );
                    scl_fctr3 =
                    _mm256_loadu_ps( ( float* )post_ops_list_temp->scale_factor +
                        post_ops_attr.post_op_c_j + ( 2 * 16 ) );
                    scl_fctr4 =
                    _mm256_loadu_ps( ( float* )post_ops_list_temp->scale_factor +
                        post_ops_attr.post_op_c_j + ( 3 * 16 ) );
                }
                else
                {
                    scl_fctr1 =
                    _mm256_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
                        post_ops_attr.post_op_c_i + 0 ) );
                    scl_fctr2 =
                    _mm256_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
                        post_ops_attr.post_op_c_i + 1 ) );
                    scl_fctr3 =
                    _mm256_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
                        post_ops_attr.post_op_c_i + 2 ) );
                    scl_fctr4 =
                    _mm256_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
                        post_ops_attr.post_op_c_i + 3 ) );
                    scl_fctr5 =
                    _mm256_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
                        post_ops_attr.post_op_c_i + 4 ) );
                    scl_fctr6 =
                    _mm256_set1_ps( *( ( float* )post_ops_list_temp->scale_factor +
                        post_ops_attr.post_op_c_i + 5 ) );
                }
            }
            if ( is_bf16 == TRUE )
            {
                bfloat16* matptr = ( bfloat16* )post_ops_list_temp->op_args1;

                if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
                ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
                {
                    BF16_F32_MATRIX_MUL_4COL(ymm0,ymm1,ymm2,ymm3,scl_fctr1,
                                        scl_fctr2,scl_fctr3,scl_fctr4,
                                        0,8,9,10,11);
                    BF16_F32_MATRIX_MUL_4COL(ymm4,ymm5,ymm6,ymm7,scl_fctr1,
                                        scl_fctr2,scl_fctr3,scl_fctr4,
                                        1,12,13,14,15);
                    BF16_F32_MATRIX_MUL_4COL(ymm0,ymm1,ymm2,ymm3,scl_fctr1,
                                        scl_fctr2,scl_fctr3,scl_fctr4,
                                        2,16,17,18,19);
                    BF16_F32_MATRIX_MUL_4COL(ymm4,ymm5,ymm6,ymm7,scl_fctr1,
                                        scl_fctr2,scl_fctr3,scl_fctr4,
                                        3,20,21,22,23);
                    BF16_F32_MATRIX_MUL_4COL(ymm0,ymm1,ymm2,ymm3,scl_fctr1,
                                        scl_fctr2,scl_fctr3,scl_fctr4,
                                        4,24,25,26,27);
                    BF16_F32_MATRIX_MUL_4COL(ymm4,ymm5,ymm6,ymm7,scl_fctr1,
                                        scl_fctr2,scl_fctr3,scl_fctr4,
                                        5,28,29,30,31);
                }
                else
                {
                    BF16_F32_MATRIX_MUL_4COL(ymm0,ymm1,ymm2,ymm3,scl_fctr1,
                                        scl_fctr1,scl_fctr1,scl_fctr1,
                                        0,8,9,10,11);
                    BF16_F32_MATRIX_MUL_4COL(ymm4,ymm5,ymm6,ymm7,scl_fctr2,
                                        scl_fctr2,scl_fctr2,scl_fctr2,
                                        1,12,13,14,15);
                    BF16_F32_MATRIX_MUL_4COL(ymm0,ymm1,ymm2,ymm3,scl_fctr3,
                                        scl_fctr3,scl_fctr3,scl_fctr3,
                                        2,16,17,18,19);
                    BF16_F32_MATRIX_MUL_4COL(ymm4,ymm5,ymm6,ymm7,scl_fctr4,
                                        scl_fctr4,scl_fctr4,scl_fctr4,
                                        3,20,21,22,23);
                    BF16_F32_MATRIX_MUL_4COL(ymm0,ymm1,ymm2,ymm3,scl_fctr5,
                                        scl_fctr5,scl_fctr5,scl_fctr5,
                                        4,24,25,26,27);
                    BF16_F32_MATRIX_MUL_4COL(ymm4,ymm5,ymm6,ymm7,scl_fctr6,
                                        scl_fctr6,scl_fctr6,scl_fctr6,
                                        5,28,29,30,31);
                }
            }
            else
            {
                float* matptr = ( float* )post_ops_list_temp->op_args1;

                if ( ( *( char* )post_ops_list_temp->op_args2 == 'r' ) ||
                    ( *( char* )post_ops_list_temp->op_args2 == 'R' ) )
                {
                    F32_F32_MATRIX_MUL_4COL_YMM(ymm0,ymm1,ymm2,ymm3,scl_fctr1,
                                                scl_fctr2,scl_fctr3,scl_fctr4,
                                                0,8,9,10,11);
                    F32_F32_MATRIX_MUL_4COL_YMM(ymm4,ymm5,ymm6,ymm7,scl_fctr1,
                                                scl_fctr2,scl_fctr3,scl_fctr4,
                                                1,12,13,14,15);
                    F32_F32_MATRIX_MUL_4COL_YMM(ymm0,ymm1,ymm2,ymm3,scl_fctr1,
                                                scl_fctr2,scl_fctr3,scl_fctr4,
                                                2,16,17,18,19);
                    F32_F32_MATRIX_MUL_4COL_YMM(ymm4,ymm5,ymm6,ymm7,scl_fctr1,
                                                scl_fctr2,scl_fctr3,scl_fctr4,
                                                3,20,21,22,23);
                    F32_F32_MATRIX_MUL_4COL_YMM(ymm0,ymm1,ymm2,ymm3,scl_fctr1,
                                                scl_fctr2,scl_fctr3,scl_fctr4,
                                                4,24,25,26,27);
                    F32_F32_MATRIX_MUL_4COL_YMM(ymm4,ymm5,ymm6,ymm7,scl_fctr1,
                                                scl_fctr2,scl_fctr3,scl_fctr4,
                                                5,28,29,30,31);
                }
                else
                {
                    F32_F32_MATRIX_MUL_4COL_YMM(ymm0,ymm1,ymm2,ymm3,scl_fctr1,
                                                scl_fctr1,scl_fctr1,scl_fctr1,
                                                0,8,9,10,11);
                    F32_F32_MATRIX_MUL_4COL_YMM(ymm4,ymm5,ymm6,ymm7,scl_fctr2,
                                                scl_fctr2,scl_fctr2,scl_fctr2,
                                                1,12,13,14,15);
                    F32_F32_MATRIX_MUL_4COL_YMM(ymm0,ymm1,ymm2,ymm3,scl_fctr3,
                                                scl_fctr3,scl_fctr3,scl_fctr3,
                                                2,16,17,18,19);
                    F32_F32_MATRIX_MUL_4COL_YMM(ymm4,ymm5,ymm6,ymm7,scl_fctr4,
                                                scl_fctr4,scl_fctr4,scl_fctr4,
                                                3,20,21,22,23);
                    F32_F32_MATRIX_MUL_4COL_YMM(ymm0,ymm1,ymm2,ymm3,scl_fctr5,
                                                scl_fctr5,scl_fctr5,scl_fctr5,
                                                4,24,25,26,27);
                    F32_F32_MATRIX_MUL_4COL_YMM(ymm4,ymm5,ymm6,ymm7,scl_fctr6,
                                                scl_fctr6,scl_fctr6,scl_fctr6,
                                                5,28,29,30,31);
                }
            }
            POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
        }
POST_OPS_SWISH_6x32F:
        {
            ymm0 =
            _mm256_broadcast_ss( ( float* )post_ops_list_temp->op_args2 );
            __m256 z, dn;
            __m256i ex_out;

            SWISH_F32_AVX2_DEF(ymm8, ymm0, ymm1, ymm2, ymm3, z, dn, ex_out)
            SWISH_F32_AVX2_DEF(ymm9, ymm0, ymm1, ymm2, ymm3, z, dn, ex_out)
            SWISH_F32_AVX2_DEF(ymm10, ymm0, ymm1, ymm2, ymm3, z, dn, ex_out)
            SWISH_F32_AVX2_DEF(ymm11, ymm0, ymm1, ymm2, ymm3, z, dn, ex_out)

            SWISH_F32_AVX2_DEF(ymm12, ymm0, ymm1, ymm2, ymm3, z, dn, ex_out)
            SWISH_F32_AVX2_DEF(ymm13, ymm0, ymm1, ymm2, ymm3, z, dn, ex_out)
            SWISH_F32_AVX2_DEF(ymm14, ymm0, ymm1, ymm2, ymm3, z, dn, ex_out)
            SWISH_F32_AVX2_DEF(ymm15, ymm0, ymm1, ymm2, ymm3, z, dn, ex_out)

            SWISH_F32_AVX2_DEF(ymm16, ymm0, ymm1, ymm2, ymm3, z, dn, ex_out)
            SWISH_F32_AVX2_DEF(ymm17, ymm0, ymm1, ymm2, ymm3, z, dn, ex_out)
            SWISH_F32_AVX2_DEF(ymm18, ymm0, ymm1, ymm2, ymm3, z, dn, ex_out)
            SWISH_F32_AVX2_DEF(ymm19, ymm0, ymm1, ymm2, ymm3, z, dn, ex_out)

            SWISH_F32_AVX2_DEF(ymm20, ymm0, ymm1, ymm2, ymm3, z, dn, ex_out)
            SWISH_F32_AVX2_DEF(ymm21, ymm0, ymm1, ymm2, ymm3, z, dn, ex_out)
            SWISH_F32_AVX2_DEF(ymm22, ymm0, ymm1, ymm2, ymm3, z, dn, ex_out)
            SWISH_F32_AVX2_DEF(ymm23, ymm0, ymm1, ymm2, ymm3, z, dn, ex_out)

            SWISH_F32_AVX2_DEF(ymm24, ymm0, ymm1, ymm2, ymm3, z, dn, ex_out)
            SWISH_F32_AVX2_DEF(ymm25, ymm0, ymm1, ymm2, ymm3, z, dn, ex_out)
            SWISH_F32_AVX2_DEF(ymm26, ymm0, ymm1, ymm2, ymm3, z, dn, ex_out)
            SWISH_F32_AVX2_DEF(ymm27, ymm0, ymm1, ymm2, ymm3, z, dn, ex_out)

            SWISH_F32_AVX2_DEF(ymm28, ymm0, ymm1, ymm2, ymm3, z, dn, ex_out)
            SWISH_F32_AVX2_DEF(ymm29, ymm0, ymm1, ymm2, ymm3, z, dn, ex_out)
            SWISH_F32_AVX2_DEF(ymm30, ymm0, ymm1, ymm2, ymm3, z, dn, ex_out)
            SWISH_F32_AVX2_DEF(ymm31, ymm0, ymm1, ymm2, ymm3, z, dn, ex_out)

            POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
        }
POST_OPS_TANH_6x32F:
        {
            __m256 dn;
            __m256i q;

            TANH_F32S_AVX2(ymm8, ymm0, ymm1, ymm2, ymm3, dn, q)
            TANH_F32S_AVX2(ymm9, ymm0, ymm1, ymm2, ymm3, dn, q)
            TANH_F32S_AVX2(ymm10, ymm0, ymm1, ymm2, ymm3, dn, q)
            TANH_F32S_AVX2(ymm11, ymm0, ymm1, ymm2, ymm3, dn, q)

            TANH_F32S_AVX2(ymm12, ymm0, ymm1, ymm2, ymm3, dn, q)
            TANH_F32S_AVX2(ymm13, ymm0, ymm1, ymm2, ymm3, dn, q)
            TANH_F32S_AVX2(ymm14, ymm0, ymm1, ymm2, ymm3, dn, q)
            TANH_F32S_AVX2(ymm15, ymm0, ymm1, ymm2, ymm3, dn, q)

            TANH_F32S_AVX2(ymm16, ymm0, ymm1, ymm2, ymm3, dn, q)
            TANH_F32S_AVX2(ymm17, ymm0, ymm1, ymm2, ymm3, dn, q)
            TANH_F32S_AVX2(ymm18, ymm0, ymm1, ymm2, ymm3, dn, q)
            TANH_F32S_AVX2(ymm19, ymm0, ymm1, ymm2, ymm3, dn, q)

            TANH_F32S_AVX2(ymm20, ymm0, ymm1, ymm2, ymm3, dn, q)
            TANH_F32S_AVX2(ymm21, ymm0, ymm1, ymm2, ymm3, dn, q)
            TANH_F32S_AVX2(ymm22, ymm0, ymm1, ymm2, ymm3, dn, q)
            TANH_F32S_AVX2(ymm23, ymm0, ymm1, ymm2, ymm3, dn, q)

            TANH_F32S_AVX2(ymm24, ymm0, ymm1, ymm2, ymm3, dn, q)
            TANH_F32S_AVX2(ymm25, ymm0, ymm1, ymm2, ymm3, dn, q)
            TANH_F32S_AVX2(ymm26, ymm0, ymm1, ymm2, ymm3, dn, q)
            TANH_F32S_AVX2(ymm27, ymm0, ymm1, ymm2, ymm3, dn, q)

            TANH_F32S_AVX2(ymm28, ymm0, ymm1, ymm2, ymm3, dn, q)
            TANH_F32S_AVX2(ymm29, ymm0, ymm1, ymm2, ymm3, dn, q)
            TANH_F32S_AVX2(ymm30, ymm0, ymm1, ymm2, ymm3, dn, q)
            TANH_F32S_AVX2(ymm31, ymm0, ymm1, ymm2, ymm3, dn, q)

            POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
        }
POST_OPS_SIGMOID_6x32F:
        {
            __m256 z, dn;
            __m256i ex_out;

            SIGMOID_F32_AVX2_DEF(ymm8, ymm1, ymm2, ymm3, z, dn, ex_out)
            SIGMOID_F32_AVX2_DEF(ymm9, ymm1, ymm2, ymm3, z, dn, ex_out)
            SIGMOID_F32_AVX2_DEF(ymm10, ymm1, ymm2, ymm3, z, dn, ex_out)
            SIGMOID_F32_AVX2_DEF(ymm11, ymm1, ymm2, ymm3, z, dn, ex_out)

            SIGMOID_F32_AVX2_DEF(ymm12, ymm1, ymm2, ymm3, z, dn, ex_out)
            SIGMOID_F32_AVX2_DEF(ymm13, ymm1, ymm2, ymm3, z, dn, ex_out)
            SIGMOID_F32_AVX2_DEF(ymm14, ymm1, ymm2, ymm3, z, dn, ex_out)
            SIGMOID_F32_AVX2_DEF(ymm15, ymm1, ymm2, ymm3, z, dn, ex_out)

            SIGMOID_F32_AVX2_DEF(ymm16, ymm1, ymm2, ymm3, z, dn, ex_out)
            SIGMOID_F32_AVX2_DEF(ymm17, ymm1, ymm2, ymm3, z, dn, ex_out)
            SIGMOID_F32_AVX2_DEF(ymm18, ymm1, ymm2, ymm3, z, dn, ex_out)
            SIGMOID_F32_AVX2_DEF(ymm19, ymm1, ymm2, ymm3, z, dn, ex_out)

            SIGMOID_F32_AVX2_DEF(ymm20, ymm1, ymm2, ymm3, z, dn, ex_out)
            SIGMOID_F32_AVX2_DEF(ymm21, ymm1, ymm2, ymm3, z, dn, ex_out)
            SIGMOID_F32_AVX2_DEF(ymm22, ymm1, ymm2, ymm3, z, dn, ex_out)
            SIGMOID_F32_AVX2_DEF(ymm23, ymm1, ymm2, ymm3, z, dn, ex_out)

            SIGMOID_F32_AVX2_DEF(ymm24, ymm1, ymm2, ymm3, z, dn, ex_out)
            SIGMOID_F32_AVX2_DEF(ymm25, ymm1, ymm2, ymm3, z, dn, ex_out)
            SIGMOID_F32_AVX2_DEF(ymm26, ymm1, ymm2, ymm3, z, dn, ex_out)
            SIGMOID_F32_AVX2_DEF(ymm27, ymm1, ymm2, ymm3, z, dn, ex_out)

            SIGMOID_F32_AVX2_DEF(ymm28, ymm1, ymm2, ymm3, z, dn, ex_out)
            SIGMOID_F32_AVX2_DEF(ymm29, ymm1, ymm2, ymm3, z, dn, ex_out)
            SIGMOID_F32_AVX2_DEF(ymm30, ymm1, ymm2, ymm3, z, dn, ex_out)
            SIGMOID_F32_AVX2_DEF(ymm31, ymm1, ymm2, ymm3, z, dn, ex_out)

            POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR
        }
POST_OPS_6x32F_DISABLE:
        {
            uint32_t tlsb, rounded, temp[8] = {0};
            int i;
            bfloat16* dest;

            if ( ( post_ops_attr.buf_downscale != NULL ) &&
                ( post_ops_attr.is_last_k == TRUE ) )
            {
                STORE_F32_BF16_YMM(ymm8, 0, 0, 8)
                STORE_F32_BF16_YMM(ymm9, 0, 1, 8)
                STORE_F32_BF16_YMM(ymm10, 0, 2, 8)
                STORE_F32_BF16_YMM(ymm11, 0, 3, 8)

                STORE_F32_BF16_YMM(ymm12, 1, 0, 8)
                STORE_F32_BF16_YMM(ymm13, 1, 1, 8)
                STORE_F32_BF16_YMM(ymm14, 1, 2, 8)
                STORE_F32_BF16_YMM(ymm15, 1, 3, 8)

                STORE_F32_BF16_YMM(ymm16, 2, 0, 8)
                STORE_F32_BF16_YMM(ymm17, 2, 1, 8)
                STORE_F32_BF16_YMM(ymm18, 2, 2, 8)
                STORE_F32_BF16_YMM(ymm19, 2, 3, 8)

                STORE_F32_BF16_YMM(ymm20, 3, 0, 8)
                STORE_F32_BF16_YMM(ymm21, 3, 1, 8)
                STORE_F32_BF16_YMM(ymm22, 3, 2, 8)
                STORE_F32_BF16_YMM(ymm23, 3, 3, 8)

                STORE_F32_BF16_YMM(ymm24, 4, 0, 8)
                STORE_F32_BF16_YMM(ymm25, 4, 1, 8)
                STORE_F32_BF16_YMM(ymm26, 4, 2, 8)
                STORE_F32_BF16_YMM(ymm27, 4, 3, 8)

                STORE_F32_BF16_YMM(ymm28, 5, 0, 8)
                STORE_F32_BF16_YMM(ymm29, 5, 1, 8)
                STORE_F32_BF16_YMM(ymm30, 5, 2, 8)
                STORE_F32_BF16_YMM(ymm31, 5, 3, 8)
            }
            else
            {
                _mm256_storeu_ps(cbuf, ymm8);
                _mm256_storeu_ps(cbuf + 8, ymm9);
                _mm256_storeu_ps(cbuf + 16, ymm10);
                _mm256_storeu_ps(cbuf + 24, ymm11);

                cbuf += rs_c;

                _mm256_storeu_ps(cbuf, ymm12);
                _mm256_storeu_ps(cbuf + 8, ymm13);
                _mm256_storeu_ps(cbuf + 16, ymm14);
                _mm256_storeu_ps(cbuf + 24, ymm15);

                cbuf += rs_c;

                _mm256_storeu_ps(cbuf, ymm16);
                _mm256_storeu_ps(cbuf + 8, ymm17);
                _mm256_storeu_ps(cbuf + 16, ymm18);
                _mm256_storeu_ps(cbuf + 24, ymm19);

                cbuf += rs_c;

                _mm256_storeu_ps(cbuf, ymm20);
                _mm256_storeu_ps(cbuf + 8, ymm21);
                _mm256_storeu_ps(cbuf + 16, ymm22);
                _mm256_storeu_ps(cbuf + 24, ymm23);

                cbuf += rs_c;

                _mm256_storeu_ps(cbuf, ymm24);
                _mm256_storeu_ps(cbuf + 8, ymm25);
                _mm256_storeu_ps(cbuf + 16, ymm26);
                _mm256_storeu_ps(cbuf + 24, ymm27);

                cbuf += rs_c;

                _mm256_storeu_ps(cbuf, ymm28);
                _mm256_storeu_ps(cbuf + 8, ymm29);
                _mm256_storeu_ps(cbuf + 16, ymm30);
                _mm256_storeu_ps(cbuf + 24, ymm31);
            }
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
          lpgemm_rowvar_f32f32f32of32_avx512_256_1x32,
          lpgemm_rowvar_f32f32f32of32_avx512_256_2x32,
          lpgemm_rowvar_f32f32f32of32_avx512_256_3x32,
          lpgemm_rowvar_f32f32f32of32_avx512_256_4x32,
          lpgemm_rowvar_f32f32f32of32_avx512_256_5x32
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

LPGEMM_MAIN_KERN(float,float,float,f32f32f32of32_avx512_256_6x64m)
{
    uint64_t n_left = n0 % NR;  //n0 is expected to be n0<=NR
    // First check whether this is a edge case in the n dimension.
    // If so, dispatch other 6x?m kernels, as needed.
    if (n_left )
    {
        float*  cij = (float* )c;
        float*  bj  = (float* )b;
        float*  ai  = (float* )a;

        if( 32 <= n_left )
        {
            const dim_t nr_cur = 32;
            lpgemm_rowvar_f32f32f32of32_avx512_256_6x32m
            (
              m0, nr_cur, k0,
              ai,  rs_a, cs_a, ps_a,
              bj,  rs_b, cs_b,
              cij, rs_c, cs_c,
              alpha, beta,
              post_ops_list, post_ops_attr
            );

            cij += nr_cur*cs_c; bj += nr_cur*cs_b; n_left -= nr_cur;
            post_ops_attr.post_op_c_j += 32;
        }
        if( 16 <= n_left )
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

    //dim_t post_op_c_i_copy = post_ops_attr.post_op_c_i;
    for( dim_t jj = 0; jj < n0; jj += 32 )
    {
        dim_t nr_cur = 32;
        //post_ops_attr.post_op_c_i = post_op_c_i_copy;
        lpgemm_rowvar_f32f32f32of32_avx512_256_6x32m
        (
          m0, nr_cur, k0,
          a,  rs_a, cs_a, ps_a,
          b + jj * cs_b,  rs_b, cs_b,
          c + jj * cs_c,  rs_c, cs_c,
          alpha, beta,
          post_ops_list, post_ops_attr
        );

        post_ops_attr.post_op_c_j += 32;
    }
}

#endif // BLIS_ADDON_LPGEMM