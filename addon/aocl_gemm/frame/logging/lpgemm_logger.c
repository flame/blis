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

#include "blis.h"
#include "lpgemm_sys.h"
#include "lpgemm_logger.h"
#include "lpgemm_post_ops.h"
#include "lpgemm_types.h"
#include <string.h>

#ifdef AOCL_LPGEMM_LOGGER_SUPPORT

static bli_pthread_once_t once_check_lpgemm_logger_init = BLIS_PTHREAD_ONCE_INIT;

static bool lpgemm_logger_enabled = FALSE;

bool is_logger_enabled()
{
	return lpgemm_logger_enabled;
}

FILE* lpgemm_start_logger_fn(void)
{
	lpgemm_init_logger();

	FILE* fd = NULL;

	if ( lpgemm_logger_enabled == TRUE )
	{
		char log_file[255] = {0};
		sprintf( log_file, "%s_P%lu_T%lu%s",
				AOCL_LPGEMM_LOG_FILE_PRFX,
				lpgemm_getpid(), lpgemm_gettid(),
				AOCL_LPGEMM_LOG_FILE_EXT );

		fd = fopen( log_file, "a" );
	}

	return fd;
}

void lpgemm_stop_logger_fn( FILE* fd )
{
	if ( ( lpgemm_logger_enabled == TRUE ) && ( fd != NULL ) )
	{
		fflush( fd );
		fclose( fd );
	}
}

#define LPGEMM_POST_OPS_STR_COPY(ops_str, ops_str_len, p_str) \
	do \
	{ \
		char* c_ops_str = p_str; \
		size_t c_ops_str_len = strlen( c_ops_str ); \
		strcpy( ops_str + ops_str_len, c_ops_str ); \
		ops_str_len += c_ops_str_len; \
	} while ( 0 ); \

void lpgemm_get_pre_ops_str( aocl_post_op* post_ops, char* ops_str )
{
	if ( post_ops == NULL )
	{
		strcpy( ops_str, "none" );
		return;
	}

	aocl_pre_op* pre_ops = post_ops->pre_ops;
	if ( ( pre_ops == NULL ) || ( pre_ops->seq_length <= 0 ) )
	{
		strcpy( ops_str, "none" );
		return;
	}
	if ( ( pre_ops->seq_length > AOCL_MAX_POST_OPS ) )
	{
		strcpy( ops_str, "ops over-limit" );
		return;
	}

	size_t ops_str_len = 0;
	char* delim_str = "#";
	size_t delim_str_len = strlen( delim_str );

	LPGEMM_POST_OPS_STR_COPY( ops_str, ops_str_len, "group_sz=" );
	int written = sprintf( ( ops_str + ops_str_len ), "%ld", pre_ops->group_size );
	if ( written > 0 )
	{
		ops_str += written;
	}
	strcpy( ops_str + ops_str_len, delim_str );
	ops_str_len += delim_str_len;

	for (dim_t i = 0; i < pre_ops->seq_length; ++i)
	{
		LPGEMM_POST_OPS_STR_COPY( ops_str, ops_str_len, "scale=" );
		if ( ( pre_ops->b_scl ) != NULL )
		{
			if ( ( pre_ops->b_scl + i )->scale_factor_len == 1 )
			{
				LPGEMM_POST_OPS_STR_COPY( ops_str, ops_str_len, "scalar_scale_factor," );
			}
			else
			{
				LPGEMM_POST_OPS_STR_COPY( ops_str, ops_str_len, "vector_scale_factor," );
			}
		}

		if ( ( pre_ops->b_zp ) != NULL )
		{
			if ( ( pre_ops->b_zp + i )->zero_point_len == 1 )
			{
				LPGEMM_POST_OPS_STR_COPY( ops_str, ops_str_len, "scalar_zero_point," );
			}
			else
			{
				LPGEMM_POST_OPS_STR_COPY( ops_str, ops_str_len, "vector_zero_point," );
			}
		}

		strcpy( ops_str + ops_str_len, delim_str );
		ops_str_len += delim_str_len;
	}
}

void lpgemm_get_post_ops_str( aocl_post_op* post_ops, char* ops_str )
{
	if ( ( post_ops == NULL ) || ( post_ops->seq_length <= 0 ) )
	{
		strcpy( ops_str, "none" );
		return;
	}
	if ( ( post_ops->seq_length > AOCL_MAX_POST_OPS ) )
	{
		strcpy( ops_str, "ops over-limit" );
		return;
	}

	size_t ops_str_len = 0;
	dim_t e_i = 0; // Multiple eltwise supported.
	dim_t s_i = 0; // Multiple sum/scale supported.
	char* delim_str = "#";
	size_t delim_str_len = strlen( delim_str );
	for ( dim_t i = 0; i < post_ops->seq_length; ++i )
	{
		// Dispatcher code
		switch ( *( post_ops->seq_vector + i ) )
		{
			case ELTWISE:
				{
					LPGEMM_POST_OPS_STR_COPY( ops_str, ops_str_len, "eltwise=");
					// Eltwise algo dispatcher.
					switch ( ( post_ops->eltwise + e_i )->algo.algo_type )
					{
						case RELU:
							{
								LPGEMM_POST_OPS_STR_COPY( ops_str, ops_str_len, "relu");
							}
							break;
						case PRELU:
							{
								LPGEMM_POST_OPS_STR_COPY( ops_str, ops_str_len, "prelu" );
							}
							break;
						case GELU_TANH:
							{
								LPGEMM_POST_OPS_STR_COPY( ops_str, ops_str_len, "gelu_tanh" );
							}
							break;
						case GELU_ERF:
							{
								LPGEMM_POST_OPS_STR_COPY( ops_str, ops_str_len, "gelu_erf" );
							}
							break;
						case CLIP:
							{
								LPGEMM_POST_OPS_STR_COPY( ops_str, ops_str_len, "clip" );
							}
							break;
						case SWISH:
							{
								LPGEMM_POST_OPS_STR_COPY( ops_str, ops_str_len, "swish" );
							}
							break;
						case TANH:
							{
								LPGEMM_POST_OPS_STR_COPY( ops_str, ops_str_len, "tanh" );
							}
							break;
						case SIGMOID:
							{
								LPGEMM_POST_OPS_STR_COPY( ops_str, ops_str_len, "sigmoid" );
							}
							break;
						default:
							break;
					}
					e_i += 1;
				}
				break;
			case BIAS:
				{
					LPGEMM_POST_OPS_STR_COPY( ops_str, ops_str_len, "bias" );
				}
				break;
			case SCALE:
				{
					LPGEMM_POST_OPS_STR_COPY( ops_str, ops_str_len, "scale=" );
					if ( ( post_ops->sum + s_i )->scale_factor_len == 1 )
					{
						LPGEMM_POST_OPS_STR_COPY( ops_str, ops_str_len, "scalar_scale_factor," );
					}
					else
					{
						LPGEMM_POST_OPS_STR_COPY( ops_str, ops_str_len, "vector_scale_factor," );
					}

					if ( ( post_ops->sum + s_i )->zero_point_len == 1 )
					{
						LPGEMM_POST_OPS_STR_COPY( ops_str, ops_str_len, "scalar_zero_point," );
					}
					else
					{
						LPGEMM_POST_OPS_STR_COPY( ops_str, ops_str_len, "vector_zero_point," );
					}

					s_i += 1;
				}
				break;
			case MATRIX_ADD:
				{
					LPGEMM_POST_OPS_STR_COPY( ops_str, ops_str_len, "mat_add" );
				}
				break;
			case MATRIX_MUL:
				{
					LPGEMM_POST_OPS_STR_COPY( ops_str, ops_str_len, "mat_mul" );
				}
				break;
			default:
				break;
		}

		strcpy( ops_str + ops_str_len, delim_str );
		ops_str_len += delim_str_len;
	}
}

void lpgemm_write_logger_gemm_fn
     (
       FILE*         fd,
       char*         op_type,
       const char    order,
       const char    transa,
       const char    transb,
       const dim_t   m,
       const dim_t   n,
       const dim_t   k,
       const float   alpha,
       const dim_t   lda,
       const char    mem_format_a,
       const dim_t   ldb,
       const char    mem_format_b,
       const float   beta,
       const dim_t   ldc,
       aocl_post_op* post_op_unparsed
     )
{
	if ( ( lpgemm_logger_enabled == TRUE ) && ( fd != NULL ) )
	{
		char pre_ops_str[1024] = {0};
		lpgemm_get_pre_ops_str( post_op_unparsed, pre_ops_str );

		char post_ops_str[2048] = {0};
		lpgemm_get_post_ops_str( post_op_unparsed, post_ops_str );

		fprintf( fd, "%c %c %c %c %c %ld %ld %ld %ld %ld %ld "\
					"%s:pre_ops=[%s]:post_ops=[%s] %f %f ",
				order, transa, transb, mem_format_a, mem_format_b,
				m, n, k, lda, ldb, ldc,
				op_type, pre_ops_str, post_ops_str,
				alpha, beta );
	}
}

void batch_lpgemm_write_logger_gemm_fn
     (
       FILE*         fd,
       char*         op_type,
       const char*   order,
       const char*   transa,
       const char*   transb,
	   const dim_t   batch_size,
       const dim_t*  m,
       const dim_t*  n,
       const dim_t*  k,
       const float*  alpha,
       const dim_t*  lda,
       const char*   mem_format_a,
       const dim_t*  ldb,
       const char*   mem_format_b,
       const float*  beta,
       const dim_t*  ldc,
       aocl_post_op** post_op_unparsed
     )
{
	if ( ( lpgemm_logger_enabled == TRUE ) && ( fd != NULL ) )
	{
		char pre_ops_str[1024] = {0};

		char post_ops_str[2048] = {0};

		fprintf(fd, "%s:bs=%ld\n", op_type, batch_size);
		for( dim_t i = 0; i < batch_size; i++ )
		{
			lpgemm_get_pre_ops_str( post_op_unparsed[i], pre_ops_str );
			lpgemm_get_post_ops_str( post_op_unparsed[i], post_ops_str );
			fprintf( fd, "%c %c %c %c %c %ld %ld %ld %ld %ld %ld "\
						":pre_ops=[%s]:post_ops=[%s] %f %f\n",
					order[i], transa[i], transb[i], mem_format_a[i], mem_format_b[i],
					m[i], n[i], k[i], lda[i], ldb[i], ldc[i],
					pre_ops_str, post_ops_str,
					(float)(alpha[i]), (float)(beta[i]) );
		}
	}
}


void lpgemm_write_logger_time_break_fn( FILE* fd, double stime )
{
	if ( ( lpgemm_logger_enabled == TRUE ) && ( fd != NULL ) )
	{
		fprintf( fd, "time:%f \n", stime );
	}
}

void _lpgemm_init_logger()
{
	lpgemm_logger_enabled =
		bli_env_get_var( "AOCL_ENABLE_LPGEMM_LOGGER", FALSE );
}

void lpgemm_init_logger()
{
	bli_pthread_once
	(
	  &once_check_lpgemm_logger_init,
	  _lpgemm_init_logger
	);
}

#else

void lpgemm_init_logger()
{}

#endif
