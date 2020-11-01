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

void AOCL_DTL_log_hemm_sizes(int8 loglevel,
                             char dt_type,
                             const f77_char side,
                             const f77_char uploa,
                             const f77_int  m,
                             const f77_int  n,
                             const void* alpha,
                             const f77_int lda,
                             const f77_int ldb,
                             const void* beta,
                             const f77_int ldc,
                             const char* filename,
                             const char* function_name,
                             int line);


// Level-2 Logging

void AOCL_DTL_log_gemv_sizes( int8 loglevel,
                              char dt_type,
                              const f77_char transa,
                              const f77_int  m,
                              const f77_int  n,
                              const void*    alpha,
                              const f77_int lda,
                              const f77_int incx,
                              const void*    beta,
                              const f77_int incy,
                              const char* filename,
                              const char* function_name,
                              int line);

void AOCL_DTL_log_ger_sizes( int8 loglevel,
                             char dt_type,
                             const f77_int m,
                             const f77_int n,
                             const void* alpha,
                             const f77_int incx,
                             const f77_int incy,
                             const f77_int lda,
                             const char* filename,
                             const char* function_name,
                             int line
                           );

void AOCL_DTL_log_hemv_sizes ( int8 loglevel,
                              char dt_type,
                              const f77_char uploa,
                              const f77_int  m,
                              const void* alpha,
                              const f77_int lda,
                              const f77_int incx,
                              const void* beta,
                              const f77_int incy,
                              const char* filename,
                              const char* function_name,
                              int line);

void AOCL_DTL_log_her2_sizes ( int8 loglevel,
                              char dt_type,
                              const f77_char uploa,
                              const f77_int  m,
                              const void* alpha,
                              const f77_int incx,
                              const f77_int incy,
                              const f77_int lda,
                              const char* filename,
                              const char* function_name,
                              int line);

// Level-1 Logging

void AOCL_DTL_log_copy_sizes( int8 loglevel,
                              char dt_type,
                              const f77_int n,
                              const f77_int incx,
                              const f77_int incy,
                              const char* filename,
                              const char* function_name,
                              int line);

void AOCL_DTL_log_amax_sizes ( int8 loglevel,
                              char dt_type,
                              const f77_int  n,
                              const f77_int incx,
                              const char* filename,
                              const char* function_name,
                              int line);

void AOCL_DTL_log_asum_sizes ( int8 loglevel,
                              char dt_type,
                              const f77_int  n,
                              const f77_int incx,
                              const char* filename,
                              const char* function_name,
                              int line);

void AOCL_DTL_log_axpby_sizes ( int8 loglevel,
                               char dt_type,
                               const f77_int  n,
                               const void* alpha,
                               const f77_int incx,
                               const void* beta,
                               const f77_int incy,
                               const char* filename,
                               const char* function_name,
                               int line);

void AOCL_DTL_log_axpy_sizes ( int8 loglevel,
                              char dt_type,
                              const f77_int  n,
                              const void* alpha,
                              const f77_int incx,
                              const f77_int incy,
                              const char* filename,
                              const char* function_name,
                              int line);

void AOCL_DTL_log_dotv_sizes( int8 loglevel,
                              char dt_type,
                              char transa,
                              const f77_int  n,
                              const f77_int incx,
                              const f77_int incy,
                              const char* filename,
                              const char* function_name,
                              int line
                              );

// Level-3 Macros
#define AOCL_DTL_LOG_GEMM_INPUTS(loglevel, alpha, a, b, beta, c)    \
    AOCL_DTL_log_gemm_sizes(loglevel, alpha, a, b, beta, c, __FILE__, __FUNCTION__, __LINE__);

#define AOCL_DTL_LOG_TRSM_INPUTS(loglevel, side, alpha, a, b)     \
    AOCL_DTL_log_trsm_sizes(loglevel, side, alpha, a, b, __FILE__, __FUNCTION__, __LINE__);

#define AOCL_DTL_LOG_GEMMT_INPUTS(loglevel, alpha, a, b, beta, c)  \
    AOCL_DTL_log_gemmt_sizes(loglevel, alpha, a, b, beta, c,  __FILE__,__FUNCTION__,__LINE__);

#define AOCL_DTL_LOG_HEMM_INPUTS(loglevel, dt_type, side, uplo, m, n, alpha, lda, ldb, beta, ldc)  \
    AOCL_DTL_log_hemm_sizes(loglevel, dt_type, side, uplo, m, n, alpha, lda, ldb, beta, ldc, \
                            __FILE__, __FUNCTION__, __LINE__);

#define AOCL_DTL_LOG_SCAL_INPUTS(loglevel, dt_type, alpha, n, incx )\
    AOCL_DTL_log_scal_sizes(loglevel, dt_type, alpha, n, incx,  __FILE__,__FUNCTION__,__LINE__);

#define AOCL_DTL_LOG_SWAP_INPUTS(loglevel, dt_type, n, incx, incy)\
    AOCL_DTL_log_swap_sizes(loglevel, dt_type, n, incx, incy,  __FILE__,__FUNCTION__,__LINE__);

// Level-2 Macros
#define AOCL_DTL_LOG_GEMV_INPUTS(loglevel, dt_type, transa, m, n, alp, lda, incx, beta, incy) \
    AOCL_DTL_log_gemv_sizes(loglevel, dt_type, transa, m, n, alp, lda, incx, beta, incy, __FILE__,\
                          __FUNCTION__, __LINE__);

#define AOCL_DTL_LOG_GER_INPUTS(loglevel, dt_type, m, n, alpha, incx, incy, lda) \
    AOCL_DTL_log_ger_sizes(loglevel, dt_type, m, n, alpha, incx, incy, lda, __FILE__, __FUNCTION__, __LINE__);

// Level-1 Macros
#define AOCL_DTL_LOG_COPY_INPUTS(loglevel, dt_type, n, incx, incy) \
    AOCL_DTL_log_copy_sizes(loglevel, dt_type, n, incx, incy, __FILE__, __FUNCTION__, __LINE__);

#define AOCL_DTL_LOG_HEMV_INPUTS(loglevel, dt_type, uploa, m, alpha, lda, incx, beta, incy) \
    AOCL_DTL_log_hemv_sizes(loglevel, dt_type, uploa, m, alpha, lda, incx, beta, incy, \
                          __FILE__, __FUNCTION__, __LINE__);

#define AOCL_DTL_LOG_HER2_INPUTS(loglevel, dt_type, uploa, m, alpha, incx, incy, lda) \
    AOCL_DTL_log_her2_sizes(loglevel, dt_type, uploa, m, alpha, incx, incy, lda, \
                          __FILE__, __FUNCTION__, __LINE__);

// Level-1 Macros
#define AOCL_DTL_LOG_AMAX_INPUTS(loglevel, dt_type, n, incx) \
    AOCL_DTL_log_amax_sizes(loglevel, dt_type, n, incx, __FILE__, __FUNCTION__, __LINE__);

#define AOCL_DTL_LOG_ASUM_INPUTS(loglevel, dt_type, n, incx) \
    AOCL_DTL_log_asum_sizes(loglevel, dt_type, n, incx, __FILE__, __FUNCTION__, __LINE__);

#define AOCL_DTL_LOG_AXPBY_INPUTS(loglevel, dt_type, n, alpha, incx, beta, incy) \
    AOCL_DTL_log_axpby_sizes(loglevel, dt_type, n, alpha, incx, beta, incy, __FILE__,\
                            __FUNCTION__, __LINE__);

#define AOCL_DTL_LOG_AXPY_INPUTS(loglevel, dt_type, n, alpha, incx, incy) \
    AOCL_DTL_log_axpy_sizes(loglevel, dt_type, n, alpha, incx, incy, __FILE__,\
                            __FUNCTION__, __LINE__);

#define AOCL_DTL_LOG_DOTV_INPUTS(loglevel, dt_type, transa, n, incx, incy) \
  AOCL_DTL_log_dotv_sizes(loglevel, dt_type, transa, n, incx, incy, __FILE__, __FUNCTION__, __LINE__); \

#else

#define AOCL_DTL_LOG_GEMM_INPUTS(loglevel, alpha, a, b, beta, c)

#define AOCL_DTL_LOG_TRSM_INPUTS(loglevel, side, alpha, a, b)

#define AOCL_DTL_LOG_GEMMT_INPUTS(loglevel, alpha, a, b, beta, c)

#define AOCL_DTL_LOG_HEMM_INPUTS(loglevel, dt_type, side, uplo, m, n, alpha, lda, ldb, beta, ldc)

#define AOCL_DTL_LOG_SCAL_INPUTS(loglevel, dt_type, alpha, n, incx )

#define AOCL_DTL_LOG_SWAP_INPUTS(loglevel, dt_type, n, incx, incy)

#define AOCL_DTL_LOG_GEMV_INPUTS(loglevel, dt_type, transa, m, n, alp, lda, incx, beta, incy)

#define AOCL_DTL_LOG_GER_INPUTS(loglevel, dt_type, m, n, alpha, incx, incy, lda)

#define AOCL_DTL_LOG_COPY_INPUTS(loglevel, dt_type, n, incx, incy)

#define AOCL_DTL_LOG_HEMV_INPUTS(loglevel, dt_type, uploa, m, alpha, lda, incx, beta, incy)

#define AOCL_DTL_LOG_HER2_INPUTS(loglevel, dt_type, uploa, m, alpha, incx, incy, lda)

#define AOCL_DTL_LOG_AMAX_INPUTS(loglevel, dt_type, n, incx)

#define AOCL_DTL_LOG_ASUM_INPUTS(loglevel, dt_type, n, incx)

#define AOCL_DTL_LOG_AXPBY_INPUTS(loglevel, dt_type, n, alpha, incx, beta, incy)

#define AOCL_DTL_LOG_AXPY_INPUTS(loglevel, dt_type, n, alpha, incx, incy)

#define AOCL_DTL_LOG_DOTV_INPUTS(loglevel, dt_type, transa, n, incx, incy)

#endif


#endif
