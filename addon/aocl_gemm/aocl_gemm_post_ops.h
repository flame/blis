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

#ifndef AOCL_GEMM_POST_OPS_H
#define AOCL_GEMM_POST_OPS_H

#define AOCL_MAX_POST_OPS 8
#define AOCL_MAX_PRE_OPS 1

typedef enum
{
	RELU = 0,
	PRELU = 1,
	GELU_TANH = 2,
	GELU_ERF = 3,
	CLIP = 4,
	SWISH = 5,
	TANH = 6,
	SIGMOID = 7,
} AOCL_ELT_ALGO_TYPE;

typedef enum
{
	SUM = 1,
	ELTWISE = 2,
	BIAS = 3,
	SCALE = 4,
	MATRIX_ADD = 5,
	MATRIX_MUL = 6,
} AOCL_POST_OP_TYPE;

typedef enum
{
	AOCL_GEMM_F32 = 0,
	AOCL_GEMM_BF16 = 1,
	AOCL_GEMM_INT8 = 2,
	AOCL_GEMM_UINT8 = 3,
	AOCL_GEMM_INT4 = 4,
	AOCL_GEMM_INT32 = 5,
	NULLTYPE = 6,
} AOCL_PARAMS_STORAGE_TYPES;

typedef struct
{
	void* alpha;
	void* beta;
	AOCL_ELT_ALGO_TYPE algo_type;
} aocl_eltwise_algo;

typedef struct
{
	bool is_power_of_2;
	void* scale_factor;
	void* buff;
	void* zero_point;
	dim_t scale_factor_len;
	dim_t zero_point_len;
	AOCL_PARAMS_STORAGE_TYPES zp_stor_type;
} aocl_post_op_sum; // Also use for scale.

typedef struct
{
	bool is_power_of_2;
	void* scale_factor;
	dim_t scale_factor_len;
	aocl_eltwise_algo algo;
} aocl_post_op_eltwise;

typedef struct
{
	void* bias;
	AOCL_PARAMS_STORAGE_TYPES stor_type;
} aocl_post_op_bias;

typedef struct
{
	void* matrix;
	void* scale_factor;
	dim_t scale_factor_len;
	dim_t ldm;
	AOCL_PARAMS_STORAGE_TYPES stor_type;
} aocl_post_op_matrix_add;

typedef struct
{
	void* matrix;
	void* scale_factor;
	dim_t scale_factor_len;
	dim_t ldm;
	AOCL_PARAMS_STORAGE_TYPES stor_type;
} aocl_post_op_matrix_mul;

typedef struct
{
	void* zero_point;
	//len should be one which is one or n i.e., one zp
	//per tensor or one zp per channel respectively
	dim_t zero_point_len;
	AOCL_PARAMS_STORAGE_TYPES zero_point_type;
} aocl_pre_op_zp;

typedef struct
{
	void* scale_factor;
	//len should be one which is one or n i.e., one sf
	//per tensor or one sf per channel respectively
	dim_t scale_factor_len;
	AOCL_PARAMS_STORAGE_TYPES scale_factor_type;
} aocl_pre_op_sf;

typedef struct
{
	aocl_pre_op_zp *b_zp;
	aocl_pre_op_sf *b_scl;
	dim_t seq_length;
	dim_t group_size;
} aocl_pre_op;

typedef struct
{
	dim_t group_size;
	dim_t seq_length;
	aocl_pre_op_sf *a_scl;
	aocl_pre_op_sf *b_scl;
	aocl_pre_op_zp *a_zp;
	aocl_pre_op_zp *b_zp;
} aocl_group_post_op;

typedef struct
{
	dim_t group_size;
} AOCL_SYMM_STAT_QUANT;

typedef struct
{
	aocl_post_op_sum* sum; // Multiple scale/sum allowed.
	aocl_post_op_eltwise* eltwise; // Multiple eltwise allowed.
	aocl_post_op_bias* bias;
	aocl_post_op_matrix_add* matrix_add;
	aocl_post_op_matrix_mul* matrix_mul;

	// eg: seq_length = 2
	dim_t seq_length;

	// eg: seq_vector[0] = BIAS, seq_vector[1] = ELTWISE means bias followed
	// by eltwise(relu, if AOCL_ELT_ALGO_TYPE = 1).
	AOCL_POST_OP_TYPE* seq_vector;

	//Pass pre-op structure also through post-ops
	aocl_pre_op  *pre_ops;

	aocl_group_post_op *post_op_grp;
	// To keep track of eltwise operations.
	dim_t num_eltwise;

} aocl_post_op;

#endif //AOCL_GEMM_POST_OPS_H
