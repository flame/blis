/*===================================================================
 * File Name :  aocldtl_blis.h
 *
 * Description : BLIS library specific debug helpes.
 *
 * Copyright (C) 2020, Advanced Micro Devices, Inc. All rights reserved.
 *
 *==================================================================*/


#ifndef __AOCLDTL_BLIS_H
#define __AOCLDTL_BLIS_H

#include "blis.h"

#if AOCL_DTL_LOG_ENABLE
void AOCL_DTL_log_gemm_sizes(int8 loglevel,
                             obj_t* alpha,
                             obj_t* a,
                             obj_t* b,
                             obj_t* beta,
                             obj_t* c,
                             const char* filename,
                             const char* functionn_name,
                             int line);

void AOCL_DTL_log_trsm_sizes(int8 loglevel,
                             side_t side,
                             obj_t* alpha,
                             obj_t* a,
                             obj_t* b,
                             const char* filename,
                             const char* function_name,
                             int line);

void AOCL_DTL_log_gemmt_sizes(int8 loglevel,
                             obj_t* alpha,
                             obj_t* a,
                             obj_t* b,
                             obj_t* beta,
                             obj_t* c,
                             const char* filename,
                             const char* function_name,
                             int line);

// Level-2 Logging

void AOCL_DTL_log_gemv_sizes( int8 loglevel,
                              char dt_type,
                              const f77_char transa,
                              const f77_int  m,
                              const f77_int  n,
                              const double    alpha,
                              const f77_int lda,
                              const f77_int incx,
                              const double    beta,
                              const f77_int incy,
                              const char* filename,
                              const char* function_name,
                              int line);

#define AOCL_DTL_LOG_GEMM_INPUTS(loglevel, alpha, a, b, beta, c)    \
    AOCL_DTL_log_gemm_sizes(loglevel, alpha, a, b, beta, c, __FILE__, __FUNCTION__, __LINE__);

#define AOCL_DTL_LOG_TRSM_INPUTS(loglevel, side, alpha, a, b)     \
    AOCL_DTL_log_trsm_sizes(loglevel, side, alpha, a, b, __FILE__, __FUNCTION__, __LINE__);

#define AOCL_DTL_LOG_GEMMT_INPUTS(loglevel, alpha, a, b, beta, c)  \
    AOCL_DTL_log_gemmt_sizes(loglevel, alpha, a, b, beta, c,  __FILE__,__FUNCTION__,__LINE__);

#define AOCL_DTL_LOG_SCAL_INPUTS(loglevel, dt_type, alpha, n, incx )\
    AOCL_DTL_log_scal_sizes(loglevel, dt_type, alpha, n, incx,  __FILE__,__FUNCTION__,__LINE__);

#define AOCL_DTL_LOG_SWAP_INPUTS(loglevel, dt_type, n, incx, incy)\
    AOCL_DTL_log_swap_sizes(loglevel, dt_type, n, incx, incy,  __FILE__,__FUNCTION__,__LINE__);

// Level-2 Macros
#define AOCL_DTL_LOG_GEMV_INPUTS(loglevel, dt_type, transa, m, n, alp, lda, incx, beta, incy) \
  AOCL_DTL_log_gemv_sizes(loglevel, dt_type, transa, m, n, alp, lda, incx, beta, incy, __FILE__,\
                          __FUNCTION__, __LINE__);

#else

#define AOCL_DTL_LOG_GEMM_INPUTS(loglevel, alpha, a, b, beta, c)

#define AOCL_DTL_LOG_TRSM_INPUTS(loglevel, side, alpha, a, b)

#define AOCL_DTL_LOG_GEMMT_INPUTS(loglevel, alpha, a, b, beta, c)

#define AOCL_DTL_LOG_SCAL_INPUTS(loglevel, dt_type, alpha, n, incx )

#define AOCL_DTL_LOG_SWAP_INPUTS(loglevel, dt_type, n, incx, incy)

#define AOCL_DTL_LOG_GEMV_INPUTS(loglevel, dt_type, transa, m, n, alp, lda, incx, beta, incy)

#endif


#endif
