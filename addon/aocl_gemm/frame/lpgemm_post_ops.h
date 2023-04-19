/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2022-23, Advanced Micro Devices, Inc. All rights reserved.

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
	POST_OPS_SUM = 8,
} LPGEMM_POST_OP_CODE;

// Used as an internal structure.
typedef struct lpgemm_post_op_t
{
	LPGEMM_POST_OP_CODE op_code;
	void* op_args1;
	void* op_args2; // alpha, zero_point, storage order
	void* op_args3; // beta, downscale buffer/original C matrix
	void* scale_factor;
	bool is_power_of_2;
	struct lpgemm_post_op_t* next;
} lpgemm_post_op;

// Used as an internal structure.
typedef struct lpgemm_post_op_attr_t
{
	dim_t post_op_c_i;
	dim_t post_op_c_j;
	dim_t rs_c_downscale;
	dim_t cs_c_downscale;
	void* buf_downscale;
	bool is_first_k;
	bool is_last_k;
	dim_t b_sum_offset;
	int32_t* b_col_sum_vec;
	int16_t* b_col_sum_vec_s16;
} lpgemm_post_op_attr;

void lpgemm_translate_to_post_ops_list
     (
       aocl_post_op*   post_op_unparsed,
       lpgemm_post_op* post_op_list,
       void*           scale_buffer,
       void*           meta_arg
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
