/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2022 - 2025, Advanced Micro Devices, Inc. All rights reserved.

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

#ifndef LPGEMM_POST_OPS_H
#define LPGEMM_POST_OPS_H

typedef enum
{
	POST_OPS_DISABLE = 0,
	POST_OPS_BIAS = 1,
	POST_OPS_RELU = 2,
	POST_OPS_RELU_SCALE = 3,
	POST_OPS_GELU_TANH = 4,
	POST_OPS_GELU_ERF = 5,
	POST_OPS_CLIP = 6,
	POST_OPS_DOWNSCALE = 7,
	POST_OPS_MATRIX_ADD = 8,
	POST_OPS_SWISH = 9,
	POST_OPS_MATRIX_MUL = 10,
	POST_OPS_TANH = 11,
	POST_OPS_SIGMOID = 12,
	POST_OPS_SUM = 13



} LPGEMM_POST_OP_CODE;

// Used as an internal structure.
typedef struct lpgemm_post_op_t
{
	uint64_t op_code;
	void* op_args1; // zero_point, bias, sum_buff
	void* op_args2; // alpha, storage order, sum_zero_point
	void* op_args3; // beta, zero_point_len
	void* scale_factor;
	dim_t scale_factor_len;
	bool is_power_of_2;
	uint64_t stor_type;
	uint64_t zp_stor_type;
	struct lpgemm_post_op_t* next;
} lpgemm_post_op;

// Used as an internal structure.
typedef struct lpgemm_pre_op_t
{
	uint64_t op_code;
	uint64_t group_size;
	void *scale_factor;
	uint64_t scale_factor_len;
	uint64_t scale_factor_type;
	void *zp;
	uint64_t zp_len;
	struct lpgemm_pre_op_t *next;
} lpgemm_pre_op;

typedef struct lpgemm_grp_post_op_attr_t
{
	void* a_scale_factor;
	uint64_t a_scale_factor_len;
	void* a_zp;
	uint64_t a_zp_len;
	void* b_scale_factor;
	uint64_t b_scale_factor_len;
	void* b_zp;
	uint64_t b_zp_len;
	uint64_t group_size;
	uint64_t grp_post_op_i;
	uint64_t grp_post_op_j;
	uint64_t grp_post_op_k;
	uint64_t grp_post_op_lda;
	uint64_t grp_post_op_ldb;
	uint64_t grp_post_op_sum_ld;
	AOCL_STORAGE_TYPE sf_stor_type;
	AOCL_STORAGE_TYPE zp_stor_type;
} lpgemm_grp_post_op_attr;

// Used as an internal structure
typedef struct lpgemm_group_post_op_t
{
	uint64_t group_size;
	void *a_scale_factor;
	uint64_t a_scale_factor_len;
	void* a_zp;
	uint64_t a_zp_len;
	void *b_scale_factor;
	uint64_t b_scale_factor_len;
	void* b_zp;
	uint64_t b_zp_len;
	AOCL_STORAGE_TYPE sf_stor_type;
	AOCL_STORAGE_TYPE zp_stor_type;
	struct lpgemm_group_post_op_t *next;
} lpgemm_group_post_op;

// Used as an internal structure.
typedef struct lpgemm_post_op_attr_t
{
	uint64_t post_op_c_i;
	uint64_t post_op_c_j;
	uint64_t rs_c_downscale;
	uint64_t cs_c_downscale;
	void* buf_downscale;
	uint64_t is_first_k;
	uint64_t is_last_k;
	uint64_t c_stor_type;
	uint64_t b_sum_offset;
	int32_t* b_col_sum_vec;
	int16_t* b_col_sum_vec_s16;
} lpgemm_post_op_attr;

typedef struct lpgemm_pre_op_attr_t
{
	void*     scale_factor;
	uint64_t  scale_factor_len;
	uint64_t  scale_factor_type;
	void*     zero_point;
	uint64_t  zero_point_len;
	uint64_t  pre_op_b_i;
	uint64_t  pre_op_b_j;
	uint64_t  group_size;
	uint64_t  pre_op_ld;
} lpgemm_pre_op_attr;
err_t lpgemm_translate_to_post_ops_list
     (
       aocl_post_op*   post_op_unparsed,
       lpgemm_post_op* post_op_list,
       void*           scale_buffer,
       void*           meta_arg,
       dim_t           m,
       dim_t           n
     );

err_t lpgemm_translate_to_pre_ops_list
	(
		aocl_pre_op *pre_op_unparsed,
		lpgemm_pre_op *pre_op_list,
		dim_t m,
		dim_t n,
		dim_t k
	);

err_t lpgemm_translate_to_group_postops_list
	(
		aocl_group_post_op *post_op_unparsed,
		lpgemm_group_post_op *post_op_list,
		dim_t m, dim_t n, dim_t k
	);

#define POST_OP_LABEL_LASTK_SAFE_JUMP \
		if ( ( post_ops_attr.is_last_k == TRUE ) && ( post_ops_list_temp != NULL ) ) \
		{ \
			goto *post_ops_labels[post_ops_list_temp->op_code]; \
		} \
		else \
		{ \
			goto *post_ops_labels[0]; \
		}

#define POST_OP_LABEL_LASTK_SAFE_JUMP_WITH_NEXT_PTR \
		post_ops_list_temp = post_ops_list_temp->next; \
		if ( post_ops_list_temp != NULL ) \
		{ \
			goto *post_ops_labels[post_ops_list_temp->op_code]; \
		} \
		else \
		{ \
			goto *post_ops_labels[0]; \
		}

#endif //LPGEMM_POST_OPS_H
