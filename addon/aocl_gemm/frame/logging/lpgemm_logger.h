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

#ifndef LPGEMM_LOGGER_H
#define LPGEMM_LOGGER_H

#ifdef AOCL_LPGEMM_LOGGER_SUPPORT

#define AOCL_LPGEMM_LOG_FILE_PRFX "aocl_gemm_log"
#define AOCL_LPGEMM_LOG_FILE_EXT ".txt"

FILE* lpgemm_start_logger_fn(void);
void lpgemm_stop_logger_fn( FILE* fd );
void lpgemm_get_post_ops_str( aocl_post_op* post_ops, char* ops_str );
void lpgemm_get_pre_ops_str( aocl_post_op* post_ops, char* ops_str );
bool is_logger_enabled();
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
     );
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
     );
void lpgemm_write_logger_time_break_fn( FILE* fd, double stime );

#define LPGEMM_START_LOGGER() \
	FILE* fd = lpgemm_start_logger_fn(); \
	double aocl_lpgemm_logger_start_time = bli_clock(); \

#define LPGEMM_STOP_LOGGER() \
	double aocl_lpgemm_logger_stop_time = DBL_MAX; \
	aocl_lpgemm_logger_stop_time = \
			bli_clock_min_diff \
			( \
			  aocl_lpgemm_logger_stop_time, \
			  aocl_lpgemm_logger_start_time \
			); \
	lpgemm_write_logger_time_break_fn( fd, aocl_lpgemm_logger_stop_time ); \
	lpgemm_stop_logger_fn( fd ); \

#define LPGEMM_WRITE_LOGGER(...) \
	lpgemm_write_logger_gemm_fn( fd, __VA_ARGS__ ); \

#define BATCH_LPGEMM_WRITE_LOGGER( op_type, order, transa, transb, \
                                   batch_size, m, n, k, \
                                   alpha, lda, mem_format_a, \
                                   ldb, mem_format_b, beta, \
                                   ldc, post_op_unparsed ) \
{ \
  if ( ( is_logger_enabled() ) && ( fd != NULL ) ) \
	{ \
		char pre_ops_str[1024] = {0}; \
 \
		char post_ops_str[2048] = {0}; \
 \
		fprintf(fd, "%s:bs=%ld\n", op_type, batch_size); \
		for( dim_t i = 0; i < batch_size; i++ ) \
		{ \
			lpgemm_get_pre_ops_str( post_op_unparsed[i], pre_ops_str ); \
			lpgemm_get_post_ops_str( post_op_unparsed[i], post_ops_str ); \
			fprintf( fd, "%c %c %c %c %c %ld %ld %ld %ld %ld %ld "\
						":pre_ops=[%s]:post_ops=[%s] %f %f\n", \
					order[i], transa[i], transb[i], mem_format_a[i], mem_format_b[i], \
					m[i], n[i], k[i], lda[i], ldb[i], ldc[i],  \
					pre_ops_str, post_ops_str, \
					(float)(alpha[i]), (float)(beta[i]) ); \
		} \
	} \
}

#else

#define LPGEMM_START_LOGGER(...)

#define LPGEMM_STOP_LOGGER(...)

#define LPGEMM_WRITE_LOGGER(...)

#define BATCH_LPGEMM_WRITE_LOGGER(op_type, order, transa, transb, \
                                   batch_size, m, n, k, \
                                   alpha, lda, mem_format_a, \
                                   ldb, mem_format_b, beta, \
                                   ldc, post_op_unparsed)

#endif

void lpgemm_init_logger();

#endif //LPGEMM_LOGGER_H
