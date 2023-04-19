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

#ifndef AOCL_GEMM_POST_OPS_H
#define AOCL_GEMM_POST_OPS_H

#define AOCL_MAX_POST_OPS 5

typedef enum
{
	RELU = 0,
	PRELU = 1,
	GELU_TANH = 2,
	GELU_ERF = 3,
	CLIP = 4,
} AOCL_ELT_ALGO_TYPE;

typedef enum
{
	SUM = 1,
	ELTWISE = 2,
	BIAS = 3,
	SCALE = 4,
} AOCL_POST_OP_TYPE;

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
} aocl_post_op_sum; // Also use for scale.

typedef struct
{
	bool is_power_of_2;
	void* scale_factor;
	aocl_eltwise_algo algo;
} aocl_post_op_eltwise;

typedef struct
{
	void* bias;
} aocl_post_op_bias;

typedef struct
{
	aocl_post_op_sum sum;
	aocl_post_op_eltwise* eltwise; //Multiple eltwise allowed.
	aocl_post_op_bias bias;

	// eg: seq_length = 2
	dim_t seq_length;

	// eg: seq_vector[0] = BIAS, seq_vector[1] = ELTWISE means bias followed
	// by eltwise(relu, if AOCL_ELT_ALGO_TYPE = 1).
	AOCL_POST_OP_TYPE* seq_vector;
} aocl_post_op;

#endif //AOCL_GEMM_POST_OPS_H
