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

#include "blis.h"
#include "lpgemm_post_ops.h"
#include "lpgemm_types.h"


static inline AOCL_STORAGE_TYPE get_stor_type(AOCL_PARAMS_STORAGE_TYPES pstor_type)
{
	AOCL_STORAGE_TYPE stor_type = NONE;
	switch ( pstor_type )
	{
		case AOCL_GEMM_F32:
				stor_type = F32;
				break;
		case AOCL_GEMM_BF16:
				stor_type = BF16;
				break;
		case AOCL_GEMM_INT8:
				stor_type = S8;
				break;
		case AOCL_GEMM_UINT8:
				stor_type = U8;
				break;
		case AOCL_GEMM_INT32:
				stor_type = S32;
				break;
		default:
				break;
	}

	return stor_type;
}


BLIS_INLINE void lpgemm_set_pre_ops_node_params
     (
       lpgemm_pre_op* pre_op_node,
       dim_t group_size,
       void* zero_point,
       void* scale_factor,
       dim_t zero_point_len,
       dim_t scale_factor_len,
       dim_t scale_factor_type
     )
{
	pre_op_node->group_size = group_size;
	pre_op_node->scale_factor = scale_factor;
	pre_op_node->scale_factor_len = scale_factor_len;
	pre_op_node->zp = zero_point;
	pre_op_node->zp_len = zero_point_len;
	pre_op_node->scale_factor_type = scale_factor_type;
	pre_op_node->next = NULL;
}

BLIS_INLINE void lpgemm_set_group_post_ops_node_params
     (
       lpgemm_group_post_op* post_op_node,
       dim_t group_size,
       void* a_zero_point,
       void* a_scale_factor,
       dim_t a_zero_point_len,
       dim_t a_scale_factor_len,
       void* b_zero_point,
       void* b_scale_factor,
       dim_t b_zero_point_len,
       dim_t b_scale_factor_len,
	   AOCL_STORAGE_TYPE sf_stor_type,
	   AOCL_STORAGE_TYPE zp_stor_type
     )
{
	post_op_node->group_size = group_size;
	post_op_node->a_zp = a_zero_point;
	post_op_node->a_zp_len = a_zero_point_len;
	post_op_node->a_scale_factor = a_scale_factor;
	post_op_node->a_scale_factor_len = a_scale_factor_len;
	post_op_node->b_zp = b_zero_point;
	post_op_node->b_zp_len = b_zero_point_len;
	post_op_node->b_scale_factor = b_scale_factor;
	post_op_node->b_scale_factor_len = b_scale_factor_len;
	post_op_node->sf_stor_type = sf_stor_type;
	post_op_node->zp_stor_type = zp_stor_type;
	post_op_node->next = NULL;
}

err_t lpgemm_translate_to_group_postops_list
	(
	  aocl_group_post_op *post_op_unparsed,
	  lpgemm_group_post_op *post_op_list,
	  dim_t m,
	  dim_t n,
	  dim_t k
    )
{
	if( ( post_op_unparsed == NULL ) || ( post_op_unparsed->seq_length <= 0 ) )
	{
		lpgemm_set_group_post_ops_node_params
		(
			post_op_list, 0,
			NULL, NULL, 0, 0, NULL, NULL, 0, 0, NONE, NONE
		);

		return BLIS_SUCCESS;
	}

	for ( dim_t i = 0; i < post_op_unparsed->seq_length; ++i )
	{
		/* group_size that is non-multiple of 4 is supported only when group_size == k */
		dim_t group_size = post_op_unparsed->group_size;
		if( ( group_size == 0 ) || ( group_size > k ) || ( group_size == k ) ) group_size = k;
		else if(post_op_unparsed->group_size % 4  > 0 ) return BLIS_FAILURE;

		if ( post_op_unparsed->a_zp != NULL )
		{
			/* check for validity of pre-ops */
			if( ( ( post_op_unparsed->a_zp)->zero_point_len > 0 ) &&
			    ( ( post_op_unparsed->a_zp)->zero_point == NULL ) ) return BLIS_FAILURE;
		}

		if ( post_op_unparsed->a_scl != NULL )
		{
			if( ( ( post_op_unparsed->a_scl)->scale_factor_len > 0 ) &&
			    ( ( post_op_unparsed->a_scl)->scale_factor == NULL ) ) return BLIS_FAILURE;
		}

		if ( post_op_unparsed->b_zp != NULL )
		{
			/* check for validity of pre-ops */
			if( ( ( post_op_unparsed->b_zp)->zero_point_len > 0 ) &&
			    ( ( post_op_unparsed->b_zp)->zero_point == NULL ) ) return BLIS_FAILURE;
		}

		if ( post_op_unparsed->b_scl != NULL )
		{
			if( ( ( post_op_unparsed->b_scl)->scale_factor_len > 0 ) &&
			    ( ( post_op_unparsed->b_scl)->scale_factor == NULL ) ) return BLIS_FAILURE;
		}

		if( ( ( post_op_unparsed->a_scl )->scale_factor_type ) != ( ( post_op_unparsed->b_scl )->scale_factor_type ) )
		{
			bli_print_msg(" A and B scale factor type mismatch. Exiting..", __FILE__, __LINE__ );
			return BLIS_FAILURE;
		}

		// Not supporting zero-point for now.
		// if( ( ( post_op_unparsed->a_zp )->zero_point_type ) != ( ( post_op_unparsed->b_zp )->zero_point_type ) )
		// {
		// 	bli_print_msg(" A and B zero point type mismatch. Exiting..", __FILE__, __LINE__ );
		// 	return BLIS_FAILURE;
		// }

		AOCL_STORAGE_TYPE tmp_zp_stor_type = NONE; //get_stor_type( ( post_op_unparsed->a_zp )->zero_point_type );

		// At this point we are sure that sf and zp types of both matrices match.
		AOCL_STORAGE_TYPE tmp_sf_stor_type = get_stor_type( ( post_op_unparsed->a_scl )->scale_factor_type );

		lpgemm_set_group_post_ops_node_params
		(
			post_op_list,
			group_size,
			// A zero-point
			post_op_unparsed->a_zp == NULL ? NULL : ( post_op_unparsed->a_zp )->zero_point,
			// A scale factor
			( post_op_unparsed->a_scl )->scale_factor,
			// A zero-point length
			post_op_unparsed->a_zp == NULL ? 0 : ( post_op_unparsed->a_zp )->zero_point_len,
			// A scale factor length
			( post_op_unparsed->a_scl )->scale_factor_len,
			// B zero-point
			post_op_unparsed->b_zp == NULL ? NULL : ( post_op_unparsed->b_zp )->zero_point,
			// B scale factor
			( post_op_unparsed->b_scl )->scale_factor,
			// B zero-point length
			post_op_unparsed->b_zp == NULL ? 0 : ( post_op_unparsed->b_zp )->zero_point_len,
			// B scale factor length
			( post_op_unparsed->b_scl )->scale_factor_len,
			tmp_sf_stor_type,
			tmp_zp_stor_type
		);

		// Simulating linked link using an array.
		if ( i < ( post_op_unparsed->seq_length - 1 ) )
		{
			( post_op_list + i )->next = ( post_op_list + i + 1 );
		}
	}

	return BLIS_SUCCESS;
}

err_t lpgemm_translate_to_pre_ops_list
	(
	  aocl_pre_op *pre_op_unparsed,
	  lpgemm_pre_op *pre_op_list,
	  dim_t m,
	  dim_t n,
	  dim_t k
	)
{
	(void)(m);			  // Unused for now, potential to be used later.
	(void)(k);			  // Unused for now, potential to be used later.

	if ((pre_op_unparsed == NULL) || (pre_op_unparsed->seq_length <= 0))
	{
		lpgemm_set_pre_ops_node_params
		(
			pre_op_list, 0,
			NULL, NULL, 0, 0, NONE
		);

		return BLIS_SUCCESS;
	}

	if ((pre_op_unparsed->seq_length > AOCL_MAX_POST_OPS))
	{
		lpgemm_set_pre_ops_node_params
		(
			pre_op_list, 0,
			NULL, NULL, 0, 0, NONE
		);

		bli_print_msg(" Max supported pre-ops is 2, supplied input pre-ops"
					  " are more. Exiting..",
					  __FILE__, __LINE__);
		return BLIS_UNEXPECTED_VECTOR_DIM; // Error, seq length exceeds max pre ops permitted.
	}

	for (dim_t i = 0; i < pre_op_unparsed->seq_length; ++i)
	{

		/* odd group_size is supported only when group_size == k */
		dim_t group_size = pre_op_unparsed->group_size;
		if( ( group_size == 0 ) || ( group_size > k ) || ( group_size == k ) ) group_size = k;
		else if(pre_op_unparsed->group_size % 2  == 1 ) return BLIS_FAILURE;

		if (pre_op_unparsed->b_zp != NULL)
		{
			/* check for validity of pre-ops */
			if( ( ( pre_op_unparsed->b_zp)->zero_point_len > 0 ) &&
			    ( ( pre_op_unparsed->b_zp)->zero_point == NULL ) ) return BLIS_FAILURE;
		}

		if (pre_op_unparsed->b_scl!=NULL)
		{
			if( ( ( pre_op_unparsed->b_scl)->scale_factor_len > 0 ) &&
			    ( ( pre_op_unparsed->b_scl)->scale_factor == NULL ) ) return BLIS_FAILURE;

		}
		lpgemm_set_pre_ops_node_params
		(
			pre_op_list,
			group_size,
			pre_op_unparsed->b_zp==NULL? NULL: (pre_op_unparsed->b_zp)->zero_point,
			(pre_op_unparsed->b_scl)->scale_factor,
			pre_op_unparsed->b_zp==NULL? 0: (pre_op_unparsed->b_zp)->zero_point_len,
			(pre_op_unparsed->b_scl)->scale_factor_len,
			(pre_op_unparsed->b_scl)->scale_factor_type == AOCL_GEMM_BF16 ? BF16 : F32
		);

		// Simulating linked link using an array.
		if (i < (pre_op_unparsed->seq_length - 1))
		{
			(pre_op_list + i)->next = (pre_op_list + i + 1);
		}
	}

	return BLIS_SUCCESS;
}

BLIS_INLINE void lpgemm_set_node_params(
	lpgemm_post_op *post_op_node,
	LPGEMM_POST_OP_CODE op_code,
	void *op1,
	void *op2,
	void *op3,
	void *scale_factor,
	dim_t scale_factor_len,
	bool is_power_of_2,
	AOCL_STORAGE_TYPE stor_type,
	AOCL_STORAGE_TYPE zp_stor_type)
{
	post_op_node->op_code = op_code;
	post_op_node->op_args1 = op1;
	post_op_node->op_args2 = op2;
	post_op_node->op_args3 = op3;
	post_op_node->scale_factor = scale_factor;
	post_op_node->scale_factor_len = scale_factor_len;
	post_op_node->is_power_of_2 = is_power_of_2;
	post_op_node->stor_type = stor_type;
	post_op_node->zp_stor_type = zp_stor_type;
	post_op_node->next = NULL;
}

err_t lpgemm_translate_to_post_ops_list
     (
       aocl_post_op*   post_op_unparsed,
       lpgemm_post_op* post_op_list,
       void*           scale_buffer,
       void*           meta_arg,
       dim_t           m,
       dim_t           n
     )
{
	( void )( scale_buffer ); //Unused for now, potential to be used later.
	( void )( m ); //Unused for now, potential to be used later.

	if ( ( post_op_unparsed == NULL ) || ( post_op_unparsed->seq_length <= 0 ) )
	{
		lpgemm_set_node_params
		(
		  post_op_list, POST_OPS_DISABLE,
		  NULL, NULL, NULL, NULL, 0, FALSE, NONE, NONE
		);

		return BLIS_SUCCESS;
	}

	if ( ( post_op_unparsed->seq_length > AOCL_MAX_POST_OPS ) )
	{
		lpgemm_set_node_params
		(
		  post_op_list, POST_OPS_DISABLE,
		  NULL, NULL, NULL, NULL, 0, FALSE, NONE, NONE
		);

		bli_print_msg(" Max supported post-ops is 5, supplied input post-ops" \
						" are more. Exiting..", __FILE__, __LINE__ );
		return BLIS_UNEXPECTED_VECTOR_DIM; //Error, seq length exceeds max post ops permitted.
	}

	dim_t e_i = 0; // Multiple eltwise supported.
	dim_t s_i = 0; // Multiple sum/scale supported.
	dim_t b_i = 0; // Multiple bias supported.
	dim_t m_i = 0; // Multiple matrix add supported.
	dim_t mul_i = 0; // Multiple matrix mul supported.
	for ( dim_t i = 0; i < post_op_unparsed->seq_length; ++i )
	{
		// Dispatcher code
		switch ( *( post_op_unparsed->seq_vector + i ) )
		{
			case SUM:
					{
						lpgemm_set_node_params
						(
						  ( post_op_list + i ), POST_OPS_SUM,
						  ( post_op_unparsed->sum + s_i )->buff,
						  ( post_op_unparsed->sum + s_i )->zero_point,
						  NULL,
						  ( post_op_unparsed->sum + s_i )->scale_factor,
						  ( post_op_unparsed->sum + s_i )->scale_factor_len,
						  ( post_op_unparsed->sum + s_i )->is_power_of_2,
						  NONE, NONE
						);

						s_i += 1;
					}
					break;
			case ELTWISE:
					{
						LPGEMM_POST_OP_CODE tmp_code = POST_OPS_DISABLE;
						// Eltwise algo dispatcher.
						switch ( ( post_op_unparsed->eltwise + e_i )->algo.algo_type )
						{
							case RELU:
									tmp_code = POST_OPS_RELU;
									break;
							case PRELU:
									if( ( post_op_unparsed->eltwise + e_i )->algo.alpha == NULL )
									{
										bli_print_msg(" Post_op.alpha is NULL. Exiting..", __FILE__, __LINE__ );
										return BLIS_NULL_POINTER;
									}
									tmp_code = POST_OPS_RELU_SCALE;
									break;
							case GELU_TANH:
									tmp_code = POST_OPS_GELU_TANH;
									break;
							case GELU_ERF:
									tmp_code = POST_OPS_GELU_ERF;
									break;
							case CLIP:
									if( ( ( post_op_unparsed->eltwise + e_i )->algo.alpha == NULL ) ||
									    ( ( post_op_unparsed->eltwise + e_i )->algo.beta  == NULL ) )
									{
										bli_print_msg(" Post_op.clip min or max value is NULL. Exiting..", __FILE__, __LINE__ );
										return BLIS_NULL_POINTER;
									}
									tmp_code = POST_OPS_CLIP;
									break;
							case SWISH:
									if( ( post_op_unparsed->eltwise + e_i )->algo.alpha == NULL )
									{
										bli_print_msg(" Post_op.alpha is NULL. Exiting..", __FILE__, __LINE__ );
										return BLIS_NULL_POINTER;
									}
									tmp_code = POST_OPS_SWISH;
									break;
							case TANH:
									tmp_code = POST_OPS_TANH;
									break;
							case SIGMOID:
									tmp_code = POST_OPS_SIGMOID;
									break;
							default:
									break;
						}
						lpgemm_set_node_params
						(
						  ( post_op_list + i ), tmp_code,
						  NULL,
						  ( post_op_unparsed->eltwise + e_i )->algo.alpha,
						  ( post_op_unparsed->eltwise + e_i )->algo.beta,
						  ( post_op_unparsed->eltwise + e_i )->scale_factor,
						  ( post_op_unparsed->eltwise + e_i )->scale_factor_len,
						  ( post_op_unparsed->eltwise + e_i )->is_power_of_2,
						  NONE, NONE
						);
						e_i += 1;
					}
					break;
			case BIAS:
					{
						if( ( post_op_unparsed->bias + b_i )->bias == NULL )
						{
							bli_print_msg(" Post_op.bias is NULL. Exiting..", __FILE__, __LINE__ );
							return BLIS_NULL_POINTER;
						}
						AOCL_STORAGE_TYPE tmp_stor_type =
							get_stor_type( ( post_op_unparsed->bias + b_i )->stor_type );
						lpgemm_set_node_params
						(
						  ( post_op_list + i ), POST_OPS_BIAS,
						  ( post_op_unparsed->bias + b_i )->bias,
						  meta_arg, NULL, NULL, 0, FALSE, tmp_stor_type, NONE
						);

						b_i += 1;
					}
					break;
			case SCALE:
					{
						if ( ( ( post_op_unparsed->sum + s_i )->scale_factor_len > 0 ) &&
							 ( ( post_op_unparsed->sum + s_i )->scale_factor == NULL ) )
						{
							bli_print_msg(" Post_op.scale scale_factor is NULL. Exiting..",
											__FILE__, __LINE__ );
							return BLIS_NULL_POINTER;
						}
						if ( ( ( post_op_unparsed->sum + s_i )->zero_point_len > 0 ) &&
							 ( ( post_op_unparsed->sum + s_i )->zero_point == NULL ) )
						{
							bli_print_msg(" Post_op.scale zero_point is NULL. Exiting..",
											__FILE__, __LINE__ );
							return BLIS_NULL_POINTER;
						}
						if ( ( ( post_op_unparsed->sum + s_i )->scale_factor_len != 1 ) &&
							 ( ( post_op_unparsed->sum + s_i )->scale_factor_len < n ) )
						{
							bli_print_msg(" Post_op.scale scale factor length is < n." \
										  " Exiting..", __FILE__, __LINE__ );
							return BLIS_NULL_POINTER;
						}
						if ( ( ( post_op_unparsed->sum + s_i )->zero_point_len != 1 ) &&
							 ( ( post_op_unparsed->sum + s_i )->zero_point_len < n ) )
						{
							bli_print_msg(" Post_op.scale zero point length is < n." \
										  " Exiting..", __FILE__, __LINE__ );
							return BLIS_NULL_POINTER;
						}

						AOCL_STORAGE_TYPE tmp_zp_stor_type  =
							get_stor_type( ( post_op_unparsed->sum + s_i )->zp_stor_type );

						lpgemm_set_node_params
						(
						  ( post_op_list + i ), POST_OPS_DOWNSCALE,
						  ( post_op_unparsed->sum + s_i )->zero_point,
						  meta_arg, &( ( post_op_unparsed->sum + s_i )->zero_point_len ),
						  ( post_op_unparsed->sum + s_i )->scale_factor,
						  ( post_op_unparsed->sum + s_i )->scale_factor_len,
						  FALSE, NONE, tmp_zp_stor_type 
						);

						s_i += 1;
					}
					break;
			case MATRIX_ADD:
					{
						if ( ( ( post_op_unparsed->matrix_add + m_i )->matrix == NULL ) ||
							 ( ( post_op_unparsed->matrix_add + m_i )->ldm <= 0 ) )
						{
							bli_print_msg(" Post_op.matrix_add attributes are invalid. Exiting..",
											__FILE__, __LINE__ );
							return BLIS_NULL_POINTER;
						}
						AOCL_STORAGE_TYPE tmp_stor_type =
							get_stor_type( ( post_op_unparsed->matrix_add + m_i )->stor_type );

						lpgemm_set_node_params
						(
						  ( post_op_list + i ), POST_OPS_MATRIX_ADD,
						  ( post_op_unparsed->matrix_add + m_i )->matrix,
						  meta_arg, &( ( post_op_unparsed->matrix_add + m_i )->ldm ),
						  ( post_op_unparsed->matrix_add + m_i )->scale_factor,
						  ( post_op_unparsed->matrix_add + m_i )->scale_factor_len,
						  FALSE, tmp_stor_type, NONE
						);

						m_i += 1;
					}
					break;
			case MATRIX_MUL:
					{
						if ( ( ( post_op_unparsed->matrix_mul + mul_i )->matrix == NULL ) ||
							 ( ( post_op_unparsed->matrix_mul + mul_i )->ldm <= 0 ) )
						{
							bli_print_msg(" Post_op.matrix_mul attributes are invalid. Exiting..",
											__FILE__, __LINE__ );
							return BLIS_NULL_POINTER;
						}
						AOCL_STORAGE_TYPE tmp_stor_type =
							get_stor_type( ( post_op_unparsed->matrix_mul + mul_i )->stor_type );

						lpgemm_set_node_params
						(
						  ( post_op_list + i ), POST_OPS_MATRIX_MUL,
						  ( post_op_unparsed->matrix_mul + mul_i )->matrix,
						  meta_arg, &( ( post_op_unparsed->matrix_mul + mul_i )->ldm ),
						  ( post_op_unparsed->matrix_mul + mul_i )->scale_factor,
						  ( post_op_unparsed->matrix_mul + mul_i )->scale_factor_len,
						  FALSE, tmp_stor_type, NONE
						);

						mul_i += 1;
					}
					break;
			default:
					break;
		}

		// Simulating linked link using an array.
		if ( i < ( post_op_unparsed->seq_length - 1 ) )
		{
			( post_op_list + i )->next = ( post_op_list + i + 1);
		}
	}
	return BLIS_SUCCESS;
}
