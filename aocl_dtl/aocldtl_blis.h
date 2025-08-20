/*===================================================================
 * File Name :  aocldtl_blis.h
 *
 * Description : BLIS library specific debug helpes.
 *
 * Copyright (C) 2020 - 2025, Advanced Micro Devices, Inc. All rights reserved.
 *
 *==================================================================*/


#ifndef __AOCLDTL_BLIS_H
#define __AOCLDTL_BLIS_H

#if AOCL_DTL_LOG_ENABLE
dim_t AOCL_get_requested_threads_count(void);

// Level-3 Logging

void AOCL_DTL_log_gemm_sizes(int8 loglevel,
                             char dt_type,
                             const f77_char transa,
                             const f77_char transb,
                             const f77_int m,
                             const f77_int n,
                             const f77_int k,
                             const void *alpha,
                             const f77_int lda,
                             const f77_int ldb,
                             const void *beta,
                             const f77_int ldc,
                             const char *filename,
                             const char *function_name,
                             int line);

void AOCL_DTL_log_gemm_stats(int8 loglevel,
                             char dt_type,
                             const f77_int m,
                             const f77_int n,
                             const f77_int k);

void AOCL_DTL_log_gemmt_sizes(int8 loglevel,
                             char dt_type,
                             char uplo,
                             char transa,
                             char transb,
                             const f77_int n,
                             const f77_int k,
                             const void* alpha,
                             const f77_int lda,
                             const f77_int ldb,
                             const void* beta,
                             const f77_int ldc,
                             const char* filename,
                             const char* function_name,
                             int line);

void AOCL_DTL_log_gemmt_stats(int8 loglevel,
                             char dt_type,
                             const f77_int n,
                             const f77_int k);

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

void AOCL_DTL_log_her2k_sizes(int8 loglevel,
                              char dt_type,
                              const f77_char uploc,
                              const f77_char transa,
                              const f77_int  m,
                              const f77_int  k,
                              const void*    alpha,
                              const f77_int lda,
                              const f77_int ldb,
                              const void*    beta,
                              const f77_int ldc,
                              const char* filename,
                              const char* function_name,
                              int line);

void AOCL_DTL_log_herk_sizes(int8 loglevel,
                             char dt_type,
                             const f77_char uploc,
                             const f77_char transa,
                             const f77_int  m,
                             const f77_int  k,
                             const void*   alpha,
                             const f77_int lda,
                             const void*  beta,
                             const f77_int ldc,
                             const char* filename,
                             const char* function_name,
                             int line);

void AOCL_DTL_log_symm_sizes(int8 loglevel,
                             char dt_type,
                             const f77_char side,
                             const f77_char uploa,
                             const f77_int  m,
                             const f77_int  n,
                             const void*    alpha,
                             const f77_int lda,
                             const f77_int ldb,
                             const void*    beta,
                             const f77_int ldc,
                             const char* filename,
                             const char* function_name,
                             int line);

void AOCL_DTL_log_syr2k_sizes(int8 loglevel,
                             char   dt_type,
                             const f77_char uploc,
                             const f77_char transa,
                             const f77_int  m,
                             const f77_int  k,
                             const void*    alpha,
                             const f77_int  lda,
                             const f77_int  ldb,
                             const void*    beta,
                             const f77_int  ldc,
                             const char*    filename,
                             const char*    function_name,
                             int  line);

void AOCL_DTL_log_syrk_sizes(int8 loglevel,
                             char dt_type,
                             const f77_char uploc,
                             const f77_char transa,
                             const f77_int  m,
                             const f77_int  k,
                             const void*    alpha,
                             const f77_int  lda,
                             const void*    beta,
                             const f77_int  ldc,
                             const char*    filename,
                             const char*    function_name,
                             int line);

void AOCL_DTL_log_trmm_sizes(int8 loglevel,
                             char dt_type,
                             const f77_char side,
                             const f77_char uploa,
                             const f77_char transa,
                             const f77_char diaga,
                             const f77_int  m,
                             const f77_int  n,
                             const void*    alpha,
                             const f77_int  lda,
                             const f77_int  ldb,
                             const char*    filename,
                             const char*    function_name,
                             int  line);

void AOCL_DTL_log_trsm_sizes(int8 loglevel,
                             char dt,
                             f77_char side,
                             f77_char uploa,
                             f77_char transa,
                             f77_char diaga,
                             const f77_int m,
                             const f77_int n,
                             const void* alpha,
                             f77_int lda,
                             f77_int ldb,
                             const char* filename,
                             const char* function_name,
                             int line);

void AOCL_DTL_log_trsm_stats(int8 loglevel,
                             char dt_type,
                             f77_char side,
                             const f77_int m,
                             const f77_int n);


// Level-3 Extension Logging

void AOCL_DTL_log_gemm3m_sizes(int8 loglevel,
                               char dt_type,
                               const f77_char transa,
                               const f77_char transb,
                               const f77_int m,
                               const f77_int n,
                               const f77_int k,
                               const void *alpha,
                               const f77_int lda,
                               const f77_int ldb,
                               const void *beta,
                               const f77_int ldc,
                               const char *filename,
                               const char *function_name,
                               int line);

void AOCL_DTL_log_gemm3m_stats(int8 loglevel,
                               char dt_type,
                               const f77_int m,
                               const f77_int n,
                               const f77_int k);

void AOCL_DTL_log_gemm_batch_sizes(int8 loglevel,
                                   char dt_type,
                                   const f77_int group_count,
                                   const char *filename,
                                   const char *function_name,
                                   int line);

void AOCL_DTL_log_gemm_get_size_sizes(int8 loglevel,
                             char dt_type,
                             const f77_char identifer,
                             const f77_int m,
                             const f77_int n,
                             const f77_int k,
                             const char *filename,
                             const char *function_name,
                             int line);

void AOCL_DTL_log_gemm_pack_sizes(int8 loglevel,
                             char dt_type,
                             const f77_char identifer,
                             const f77_char trans,
                             const f77_int m,
                             const f77_int n,
                             const f77_int k,
                             const void *alpha,
                             const f77_int pld,
                             const char *filename,
                             const char *function_name,
                             int line);

void AOCL_DTL_log_gemm_compute_sizes(int8 loglevel,
                             char dt_type,
                             const f77_char transa,
                             const f77_char transb,
                             const f77_int m,
                             const f77_int n,
                             const f77_int k,
                             const f77_int lda,
                             const f77_int ldb,
                             const void *beta,
                             const f77_int ldc,
                             const char *filename,
                             const char *function_name,
                             int line);

// Level-2 Logging

void AOCL_DTL_log_gemv_sizes(int8 loglevel,
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

void AOCL_DTL_log_ger_sizes(int8 loglevel,
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

void AOCL_DTL_log_hemv_sizes(int8 loglevel,
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

void AOCL_DTL_log_her2_sizes(int8 loglevel,
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

void AOCL_DTL_log_her_sizes(int8 loglevel,
                            char dt_type,
                            const f77_char uploa,
                            const f77_int  m,
                            const void* alpha,
                            const f77_int  incx,
                            const f77_int lda,
                            const char* filename,
                            const char* function_name,
                            int line);

void AOCL_DTL_log_symv_sizes(int8 loglevel,
                             char dt_type,
                             const f77_char uploa,
                             const f77_int  m,
                             const void*    alpha,
                             const f77_int lda,
                             const f77_int incx,
                             const void*    beta,
                             const f77_int incy,
                             const char* filename,
                             const char* function_name,
                             int line);

void AOCL_DTL_log_syr2_sizes(int8 loglevel,
                             char dt_type,
                             const f77_char uploa,
                             const f77_int  m,
                             const void*    alpha,
                             const f77_int  incx,
                             const f77_int  incy,
                             const f77_int  lda,
                             const char*    filename,
                             const char*    function_name,
                             int  line);

void AOCL_DTL_log_syr_sizes(int8 loglevel,
                            char dt_type,
                            const f77_char  uploa,
                            const f77_int   m,
                            const void*     alpha,
                            const f77_int   incx,
                            const f77_int   lda,
                            const char*     filename,
                            const char*     function_name,
                            int line);

void AOCL_DTL_log_trmv_sizes(int8 loglevel,
                             char dt_type,
                             const f77_char uploa,
                             const f77_char transa,
                             const f77_char diaga,
                             const f77_int m,
                             const f77_int lda,
                             const f77_int incx,
                             const char* filename,
                             const char* function_name,
                             int line);

void AOCL_DTL_log_trsv_sizes(int8 loglevel,
                             char dt_type,
                             const f77_char uploa,
                             const f77_char transa,
                             const f77_char diaga,
                             const f77_int m,
                             const f77_int lda,
                             const f77_int incx,
                             const char* filename,
                             const char* function_name,
                             int line);

// Level-2 Banded Logging

void AOCL_DTL_log_gbmv_sizes(int8 loglevel,
                             char dt_type,
                             const f77_char transa,
                             const f77_int  m,
                             const f77_int  n,
                             const f77_int  kl,
                             const f77_int  ku,
                             const void*    alpha,
                             const f77_int lda,
                             const f77_int incx,
                             const void*    beta,
                             const f77_int incy,
                             const char* filename,
                             const char* function_name,
                             int line);

void AOCL_DTL_log_hbmv_sizes(int8 loglevel,
                             char dt_type,
                             const f77_char uploa,
                             const f77_int  m,
                             const f77_int  k,
                             const void* alpha,
                             const f77_int lda,
                             const f77_int incx,
                             const void* beta,
                             const f77_int incy,
                             const char* filename,
                             const char* function_name,
                             int line);

void AOCL_DTL_log_sbmv_sizes(int8 loglevel,
                             char dt_type,
                             const f77_char uploa,
                             const f77_int  m,
                             const f77_int  k,
                             const void*    alpha,
                             const f77_int lda,
                             const f77_int incx,
                             const void*    beta,
                             const f77_int incy,
                             const char* filename,
                             const char* function_name,
                             int line);

void AOCL_DTL_log_tbmv_sizes(int8 loglevel,
                             char dt_type,
                             const f77_char uploa,
                             const f77_char transa,
                             const f77_char diaga,
                             const f77_int m,
                             const f77_int k,
                             const f77_int lda,
                             const f77_int incx,
                             const char* filename,
                             const char* function_name,
                             int line);

void AOCL_DTL_log_tbsv_sizes(int8 loglevel,
                             char dt_type,
                             const f77_char uploa,
                             const f77_char transa,
                             const f77_char diaga,
                             const f77_int m,
                             const f77_int k,
                             const f77_int lda,
                             const f77_int incx,
                             const char* filename,
                             const char* function_name,
                             int line);

// Level-2 Packed Logging

void AOCL_DTL_log_hpmv_sizes(int8 loglevel,
                             char dt_type,
                             const f77_char uploa,
                             const f77_int  m,
                             const void* alpha,
                             const f77_int incx,
                             const void* beta,
                             const f77_int incy,
                             const char* filename,
                             const char* function_name,
                             int line);

void AOCL_DTL_log_hpr2_sizes(int8 loglevel,
                             char dt_type,
                             const f77_char uploa,
                             const f77_int  m,
                             const void* alpha,
                             const f77_int incx,
                             const f77_int incy,
                             const char* filename,
                             const char* function_name,
                             int line);

void AOCL_DTL_log_hpr_sizes(int8 loglevel,
                            char dt_type,
                            const f77_char uploa,
                            const f77_int  m,
                            const void* alpha,
                            const f77_int  incx,
                            const char* filename,
                            const char* function_name,
                            int line);

void AOCL_DTL_log_spmv_sizes(int8 loglevel,
                             char dt_type,
                             const f77_char uploa,
                             const f77_int  m,
                             const void*    alpha,
                             const f77_int incx,
                             const void*    beta,
                             const f77_int incy,
                             const char* filename,
                             const char* function_name,
                             int line);

void AOCL_DTL_log_spr2_sizes(int8 loglevel,
                             char dt_type,
                             const f77_char uploa,
                             const f77_int  m,
                             const void*    alpha,
                             const f77_int  incx,
                             const f77_int  incy,
                             const char*    filename,
                             const char*    function_name,
                             int  line);

void AOCL_DTL_log_spr_sizes(int8 loglevel,
                            char dt_type,
                            const f77_char  uploa,
                            const f77_int   m,
                            const void*     alpha,
                            const f77_int   incx,
                            const char*     filename,
                            const char*     function_name,
                            int line);

void AOCL_DTL_log_tpmv_sizes(int8 loglevel,
                             char dt_type,
                             const f77_char uploa,
                             const f77_char transa,
                             const f77_char diaga,
                             const f77_int m,
                             const f77_int incx,
                             const char* filename,
                             const char* function_name,
                             int line);

void AOCL_DTL_log_tpsv_sizes(int8 loglevel,
                             char dt_type,
                             const f77_char uploa,
                             const f77_char transa,
                             const f77_char diaga,
                             const f77_int m,
                             const f77_int incx,
                             const char* filename,
                             const char* function_name,
                             int line);

// Level-2 plane rotations and modified Givens transformation Logging

void AOCL_DTL_log_rot_sizes(int8 loglevel,
                            char dt_type,
                            const f77_int m,
                            const f77_int incx,
                            const f77_int incy,
                            const void* c,
                            const void* s,
                            const char* filename,
                            const char* function_name,
                            int line);

void AOCL_DTL_log_rotg_sizes(int8 loglevel,
                             char dt_type,
                             const void* a,
                             const void* b,
                             const void* c,
                             const void* s,
                             const char* filename,
                             const char* function_name,
                             int line);

void AOCL_DTL_log_rotm_sizes(int8 loglevel,
                             char dt_type,
                             const f77_int m,
                             const f77_int incx,
                             const f77_int incy,
                             const void* param1,
                             const void* param2,
                             const void* param3,
                             const void* param4,
                             const void* param5,
                             const char* filename,
                             const char* function_name,
                             int line);

void AOCL_DTL_log_rotmg_sizes(int8 loglevel,
                              char dt_type,
                              const void* dd1,
                              const void* dd2,
                              const void* dx1,
                              const void* dy1,
                              const char* filename,
                              const char* function_name,
                              int line);

// Level-1 Logging

void AOCL_DTL_log_amin_sizes(int8 loglevel,
                             char dt_type,
                             const f77_int  n,
                             const f77_int incx,
                             const char* filename,
                             const char* function_name,
                             int line);

void AOCL_DTL_log_amax_sizes(int8 loglevel,
                             char dt_type,
                             const f77_int  n,
                             const f77_int incx,
                             const char* filename,
                             const char* function_name,
                             int line);

void AOCL_DTL_log_asum_sizes(int8 loglevel,
                             char dt_type,
                             const f77_int  n,
                             const f77_int incx,
                             const char* filename,
                             const char* function_name,
                             int line);

void AOCL_DTL_log_axpby_sizes(int8 loglevel,
                              char dt_type,
                              const f77_int  n,
                              const void* alpha,
                              const f77_int incx,
                              const void* beta,
                              const f77_int incy,
                              const char* filename,
                              const char* function_name,
                              int line);

void AOCL_DTL_log_axpy_sizes(int8 loglevel,
                             char dt_type,
                             const f77_int  n,
                             const void* alpha,
                             const f77_int incx,
                             const f77_int incy,
                             const char* filename,
                             const char* function_name,
                             int line);

void AOCL_DTL_log_copy_sizes(int8 loglevel,
                             char dt_type,
                             const f77_int n,
                             const f77_int incx,
                             const f77_int incy,
                             const char* filename,
                             const char* function_name,
                             int line);

void AOCL_DTL_log_dotv_sizes(int8 loglevel,
                             char dt_type,
                             const f77_char conjx,
                             const f77_int  n,
                             const f77_int incx,
                             const f77_int incy,
                             const char* filename,
                             const char* function_name,
                             int line
                             );

void AOCL_DTL_log_nrm2_sizes(int8 loglevel,
                             char dt_type,
                             const f77_int  n,
                             const f77_int  incx,
                             const char* filename,
                             const char* function_name,
                             int line);

void AOCL_DTL_log_nrm2_stats(int8 loglevel,
                             char dt_type,
                             const f77_int n);


void AOCL_DTL_log_scal_sizes(int8 loglevel,
                             char dt_type,
                             const void* alpha,
                             const f77_int  n,
                             const f77_int  incx,
                             const char* filename,
                             const char* function_name,
                             int line);

void AOCL_DTL_log_swap_sizes(int8 loglevel,
                             char dt_type,
                             const f77_int  n,
                             const f77_int  incx,
                             const f77_int  incy,
                             const char* filename,
                             const char* function_name,
                             int line);

// Matrix Copy and Transpose Logging

void AOCL_DTL_log_matadd_sizes(int8 loglevel,
                               char dt_type,
                               const f77_char transa,
                               const f77_char transb,
                               const f77_int  m,
                               const f77_int  n,
                               const void*    alpha,
                               const f77_int  lda,
                               const void*    beta,
                               const f77_int  ldb,
                               const f77_int  ldc,
                               const char* filename,
                               const char* function_name,
                               int line);

void AOCL_DTL_log_matcopy_sizes(int8 loglevel,
                                char dt_type,
                                const f77_char trans,
                                const f77_int  rows,
                                const f77_int  cols,
                                const void*    alpha,
                                const f77_int  lda,
                                const f77_int  ldb,
                                const char* filename,
                                const char* function_name,
                                int line);

void AOCL_DTL_log_matcopy2_sizes(int8 loglevel,
                                 char dt_type,
                                 const f77_char trans,
                                 const f77_int  rows,
                                 const f77_int  cols,
                                 const void*    alpha,
                                 const f77_int  lda,
                                 const f77_int  stridea,
                                 const f77_int  ldb,
                                 const f77_int  strideb,
                                 const char* filename,
                                 const char* function_name,
                                 int line);

// Level-3 Macros

#define AOCL_DTL_LOG_GEMM_INPUTS(loglevel, dt, transa, transb, m, n, k, alpha, lda, ldb, beta, ldc)    \
    if (gbIsLoggingEnabled) \
        AOCL_DTL_log_gemm_sizes(loglevel, dt, transa, transb, m, n, k, alpha, lda, ldb, beta, ldc, \
                                __FILE__, __FUNCTION__, __LINE__);

#define AOCL_DTL_LOG_GEMM_STATS(loglevel, dt_type, m, n, k)    \
    if (gbIsLoggingEnabled) \
        AOCL_DTL_log_gemm_stats(loglevel, dt_type, m, n, k);

#define AOCL_DTL_LOG_GEMMT_INPUTS(loglevel, dt, uplo, transa, transb, n, k, alpha, lda, ldb, beta, ldc)  \
    if (gbIsLoggingEnabled) \
        AOCL_DTL_log_gemmt_sizes(loglevel, dt, uplo, transa, transb, n, k, alpha, lda, ldb, beta, ldc, \
                                 __FILE__,__FUNCTION__,__LINE__);

#define AOCL_DTL_LOG_GEMMT_STATS(loglevel, dt_type, n, k)    \
    if (gbIsLoggingEnabled) \
        AOCL_DTL_log_gemmt_stats(loglevel, dt_type, n, k);

#define AOCL_DTL_LOG_HEMM_INPUTS(loglevel, dt_type, side, uplo, m, n, alpha, lda, ldb, beta, ldc)  \
    if (gbIsLoggingEnabled) \
        AOCL_DTL_log_hemm_sizes(loglevel, dt_type, side, uplo, m, n, alpha, lda, ldb, beta, ldc, \
                                __FILE__, __FUNCTION__, __LINE__);

#define AOCL_DTL_LOG_HER2K_INPUTS(loglevel, dt_type, uploc, transa, m, k, alpha, lda, ldb, beta, ldc)\
    if (gbIsLoggingEnabled) \
        AOCL_DTL_log_her2k_sizes(loglevel, dt_type, uploc, transa, m, k, alpha, lda, ldb, beta, ldc, __FILE__,\
                                __FUNCTION__, __LINE__);

#define AOCL_DTL_LOG_HERK_INPUTS(loglevel, dt_type, uploc, transa, m, k, alpha, lda, beta, ldc)\
    if (gbIsLoggingEnabled) \
        AOCL_DTL_log_herk_sizes(loglevel, dt_type, transa, uploc, m, k, alpha, lda, beta, ldc, __FILE__,\
                                __FUNCTION__, __LINE__);

#define AOCL_DTL_LOG_SYMM_INPUTS(loglevel, dt_type, side, uploa, m, n, alpha, lda, ldb, beta, ldc)\
    if (gbIsLoggingEnabled) \
        AOCL_DTL_log_symm_sizes(loglevel, dt_type, side, uploa, m, n, alpha, lda, ldb, beta, ldc, __FILE__,\
                                __FUNCTION__, __LINE__);

#define AOCL_DTL_LOG_SYR2K_INPUTS(loglevel, dt_type, uploc, transa, m, k, alpha, lda, ldb, beta, ldc) \
    if (gbIsLoggingEnabled) \
        AOCL_DTL_log_syr2k_sizes(loglevel, dt_type, uploc, transa, m, k, alpha, lda, ldb, beta,\
                                ldc, __FILE__, __FUNCTION__,__LINE__);

#define AOCL_DTL_LOG_SYRK_INPUTS(loglevel, dt_type, uploc, transa, m, k, alpha, lda, beta, ldc) \
    if (gbIsLoggingEnabled) \
        AOCL_DTL_log_syrk_sizes(loglevel, dt_type, uploc, transa, m, k, alpha, lda, beta, ldc, __FILE__,\
                                __FUNCTION__,__LINE__);

#define AOCL_DTL_LOG_TRMM_INPUTS(loglevel, dt_type, side, uploa, transa, diaga, m, n, alpha, lda, ldb) \
    if (gbIsLoggingEnabled) \
        AOCL_DTL_log_trmm_sizes(loglevel, dt_type, side, uploa, transa, diaga, m, n, alpha, lda, ldb, __FILE__,\
                                __FUNCTION__,__LINE__);

#define AOCL_DTL_LOG_TRSM_INPUTS(loglevel, dt, side, uploa, transa, diaga, m, n, alpha, lda, ldb)     \
    if (gbIsLoggingEnabled) \
        AOCL_DTL_log_trsm_sizes(loglevel, dt, side, uploa, transa, diaga, m, n, alpha, lda, ldb, \
                                __FILE__, __FUNCTION__, __LINE__);

#define AOCL_DTL_LOG_TRSM_STATS(loglevel, dt_type, side, m, n)    \
    if (gbIsLoggingEnabled) \
        AOCL_DTL_log_trsm_stats(loglevel, dt_type, side, m, n);

// Level-3 Extension Macros

#define AOCL_DTL_LOG_GEMM3M_INPUTS(loglevel, dt, transa, transb, m, n, k, alpha, lda, ldb, beta, ldc)    \
    if (gbIsLoggingEnabled) \
        AOCL_DTL_log_gemm3m_sizes(loglevel, dt, transa, transb, m, n, k, alpha, lda, ldb, beta, ldc, \
                                __FILE__, __FUNCTION__, __LINE__);

#define AOCL_DTL_LOG_GEMM3M_STATS(loglevel, dt_type, m, n, k)    \
    if (gbIsLoggingEnabled) \
        AOCL_DTL_log_gemm3m_stats(loglevel, dt_type, m, n, k);

#define AOCL_DTL_LOG_GEMM_BATCH_INPUTS(loglevel, dt, group_count) \
    if (gbIsLoggingEnabled) \
        AOCL_DTL_log_gemm_batch_sizes(loglevel, dt, group_count, \
                                         __FILE__, __FUNCTION__, __LINE__);

#define AOCL_DTL_LOG_GEMM_GET_SIZE_INPUTS(loglevel, dt, identifier, m, n, k) \
    if (gbIsLoggingEnabled) \
        AOCL_DTL_log_gemm_get_size_sizes(loglevel, dt, identifier, m, n, k, \
                                         __FILE__, __FUNCTION__, __LINE__);

#define AOCL_DTL_LOG_GEMM_PACK_INPUTS(loglevel, dt, identifier, trans, m, n, k, alpha, pld) \
    if (gbIsLoggingEnabled) \
        AOCL_DTL_log_gemm_pack_sizes(loglevel, dt, identifier, trans, m, n, k, alpha, pld, \
                                     __FILE__, __FUNCTION__, __LINE__);

#define AOCL_DTL_LOG_GEMM_COMPUTE_INPUTS(loglevel, dt, transa, transb, m, n, k, lda, ldb, beta, ldc) \
    if (gbIsLoggingEnabled) \
        AOCL_DTL_log_gemm_compute_sizes(loglevel, dt, transa, transb, m, n, k, lda, ldb, beta, ldc, \
                                        __FILE__, __FUNCTION__, __LINE__);

// Level-2 Macros

#define AOCL_DTL_LOG_GEMV_INPUTS(loglevel, dt_type, transa, m, n, alp, lda, incx, beta, incy) \
    if (gbIsLoggingEnabled) \
        AOCL_DTL_log_gemv_sizes(loglevel, dt_type, transa, m, n, alp, lda, incx, beta, incy, __FILE__,\
                            __FUNCTION__, __LINE__);

#define AOCL_DTL_LOG_GER_INPUTS(loglevel, dt_type, m, n, alpha, incx, incy, lda) \
    if (gbIsLoggingEnabled) \
        AOCL_DTL_log_ger_sizes(loglevel, dt_type, m, n, alpha, incx, incy, lda, __FILE__, __FUNCTION__, __LINE__);

#define AOCL_DTL_LOG_HEMV_INPUTS(loglevel, dt_type, uploa, m, alpha, lda, incx, beta, incy) \
    if (gbIsLoggingEnabled) \
        AOCL_DTL_log_hemv_sizes(loglevel, dt_type, uploa, m, alpha, lda, incx, beta, incy, \
                            __FILE__, __FUNCTION__, __LINE__);

#define AOCL_DTL_LOG_HER2_INPUTS(loglevel, dt_type, uploa, m, alpha, incx, incy, lda) \
    if (gbIsLoggingEnabled) \
        AOCL_DTL_log_her2_sizes(loglevel, dt_type, uploa, m, alpha, incx, incy, lda, \
                            __FILE__, __FUNCTION__, __LINE__);

#define AOCL_DTL_LOG_HER_INPUTS(loglevel, dt_type, uploa, m, alpha, incx, lda )\
    if (gbIsLoggingEnabled) \
        AOCL_DTL_log_her_sizes(loglevel, dt_type, uploa, m, alpha, incx, lda,  __FILE__,__FUNCTION__,__LINE__);

#define AOCL_DTL_LOG_SYMV_INPUTS(loglevel, dt_type, uploa, m, alpha, lda, incx, beta, incy)\
    if (gbIsLoggingEnabled) \
        AOCL_DTL_log_symv_sizes(loglevel, dt_type, uploa, m, alpha, lda, incx, beta, incy, __FILE__,\
                                __FUNCTION__, __LINE__);

#define AOCL_DTL_LOG_SYR2_INPUTS(loglevel, dt_type, uploa, m, alpha, incx, incy, lda) \
    if (gbIsLoggingEnabled) \
        AOCL_DTL_log_syr2_sizes(loglevel, dt_type, uploa, m, alpha, incx, incy, lda, __FILE__,\
                                __FUNCTION__,__LINE__);

#define AOCL_DTL_LOG_SYR_INPUTS(loglevel, dt_type, uploa, m, alpha, incx, lda) \
    if (gbIsLoggingEnabled) \
        AOCL_DTL_log_syr_sizes(loglevel, dt_type, uploa, m, alpha, incx, lda,\
                                __FILE__,__FUNCTION__,__LINE__);

#define AOCL_DTL_LOG_TRMV_INPUTS(loglevel, dt_type, uploa, transa, diaga, m, lda, incx) \
    if (gbIsLoggingEnabled) \
        AOCL_DTL_log_trmv_sizes(loglevel, dt_type, uploa, transa, diaga, m, lda, incx,\
                                __FILE__,__FUNCTION__,__LINE__);

#define AOCL_DTL_LOG_TRSV_INPUTS(loglevel, dt_type, uploa, transa, diaga, m, lda, incx ) \
    if (gbIsLoggingEnabled) \
        AOCL_DTL_log_trsv_sizes(loglevel, dt_type, uploa, transa, diaga, m, lda, incx,\
                                __FILE__,__FUNCTION__,__LINE__);

// Level-2 Banded Macros

#define AOCL_DTL_LOG_GBMV_INPUTS(loglevel, dt_type, transa, m, n, kl, ku, alp, lda, incx, beta, incy) \
    if (gbIsLoggingEnabled) \
        AOCL_DTL_log_gbmv_sizes(loglevel, dt_type, transa, m, n, kl, ku, alp, lda, incx, beta, incy, __FILE__,\
                            __FUNCTION__, __LINE__);

#define AOCL_DTL_LOG_HBMV_INPUTS(loglevel, dt_type, uploa, m, k, alpha, lda, incx, beta, incy) \
    if (gbIsLoggingEnabled) \
        AOCL_DTL_log_hbmv_sizes(loglevel, dt_type, uploa, m, k, alpha, lda, incx, beta, incy, \
                            __FILE__, __FUNCTION__, __LINE__);

#define AOCL_DTL_LOG_SBMV_INPUTS(loglevel, dt_type, uploa, m, k, alpha, lda, incx, beta, incy)\
    if (gbIsLoggingEnabled) \
        AOCL_DTL_log_sbmv_sizes(loglevel, dt_type, uploa, m, k, alpha, lda, incx, beta, incy, __FILE__,\
                                __FUNCTION__, __LINE__);

#define AOCL_DTL_LOG_TBMV_INPUTS(loglevel, dt_type, uploa, transa, diaga, m, k, lda, incx) \
    if (gbIsLoggingEnabled) \
        AOCL_DTL_log_tbmv_sizes(loglevel, dt_type, uploa, transa, diaga, m, k, lda, incx,\
                                __FILE__,__FUNCTION__,__LINE__);

#define AOCL_DTL_LOG_TBSV_INPUTS(loglevel, dt_type, uploa, transa, diaga, m, k, lda, incx) \
    if (gbIsLoggingEnabled) \
        AOCL_DTL_log_tbsv_sizes(loglevel, dt_type, uploa, transa, diaga, m, k, lda, incx,\
                                __FILE__,__FUNCTION__,__LINE__);

// Level-2 Packed Macros

#define AOCL_DTL_LOG_HPMV_INPUTS(loglevel, dt_type, uploa, m, alpha, incx, beta, incy) \
    if (gbIsLoggingEnabled) \
        AOCL_DTL_log_hpmv_sizes(loglevel, dt_type, uploa, m, alpha, incx, beta, incy, \
                            __FILE__, __FUNCTION__, __LINE__);

#define AOCL_DTL_LOG_HPR2_INPUTS(loglevel, dt_type, uploa, m, alpha, incx, incy) \
    if (gbIsLoggingEnabled) \
        AOCL_DTL_log_hpr2_sizes(loglevel, dt_type, uploa, m, alpha, incx, incy, \
                            __FILE__, __FUNCTION__, __LINE__);

#define AOCL_DTL_LOG_HPR_INPUTS(loglevel, dt_type, uploa, m, alpha, incx )\
    if (gbIsLoggingEnabled) \
        AOCL_DTL_log_hpr_sizes(loglevel, dt_type, uploa, m, alpha, incx,  __FILE__,__FUNCTION__,__LINE__);

#define AOCL_DTL_LOG_SPMV_INPUTS(loglevel, dt_type, uploa, m, alpha, incx, beta, incy)\
    if (gbIsLoggingEnabled) \
        AOCL_DTL_log_spmv_sizes(loglevel, dt_type, uploa, m, alpha, incx, beta, incy, __FILE__,\
                                __FUNCTION__, __LINE__);

#define AOCL_DTL_LOG_SPR2_INPUTS(loglevel, dt_type, uploa, m, alpha, incx, incy) \
    if (gbIsLoggingEnabled) \
        AOCL_DTL_log_spr2_sizes(loglevel, dt_type, uploa, m, alpha, incx, incy, __FILE__,\
                                __FUNCTION__,__LINE__);

#define AOCL_DTL_LOG_SPR_INPUTS(loglevel, dt_type, uploa, m, alpha, incx) \
    if (gbIsLoggingEnabled) \
        AOCL_DTL_log_spr_sizes(loglevel, dt_type, uploa, m, alpha, incx,\
                                __FILE__,__FUNCTION__,__LINE__);

#define AOCL_DTL_LOG_TPMV_INPUTS(loglevel, dt_type, uploa, transa, diaga, m, incx) \
    if (gbIsLoggingEnabled) \
        AOCL_DTL_log_tpmv_sizes(loglevel, dt_type, uploa, transa, diaga, m, incx,\
                                __FILE__,__FUNCTION__,__LINE__);

#define AOCL_DTL_LOG_TPSV_INPUTS(loglevel, dt_type, uploa, transa, diaga, m, incx ) \
    if (gbIsLoggingEnabled) \
        AOCL_DTL_log_tpsv_sizes(loglevel, dt_type, uploa, transa, diaga, m, incx,\
                                __FILE__,__FUNCTION__,__LINE__);

// Level-2 plane rotations and modified Givens transformation Macros

#define AOCL_DTL_LOG_ROT_INPUTS(loglevel, dt_type, m, incx, incy, c, s) \
    if (gbIsLoggingEnabled) \
        AOCL_DTL_log_rot_sizes(loglevel, dt_type, m, incx, incy, c, s, \
                               __FILE__,__FUNCTION__,__LINE__);

#define AOCL_DTL_LOG_ROTG_INPUTS(loglevel, dt_type, a, b, c, s) \
    if (gbIsLoggingEnabled) \
        AOCL_DTL_log_rotg_sizes(loglevel, dt_type, a, b, c, s, \
                                __FILE__,__FUNCTION__,__LINE__);

#define AOCL_DTL_LOG_ROTM_INPUTS(loglevel, dt_type, m, incx, incy, param1, param2, param3, param4, param5) \
    if (gbIsLoggingEnabled) \
        AOCL_DTL_log_rotm_sizes(loglevel, dt_type, m, incx, incy, param1, param2, param3, param4, param5, \
                                __FILE__,__FUNCTION__,__LINE__);

#define AOCL_DTL_LOG_ROTMG_INPUTS(loglevel, dt_type, dd1, dd2, dx1, dy1) \
    if (gbIsLoggingEnabled) \
        AOCL_DTL_log_rotmg_sizes(loglevel, dt_type, dd1, dd2, dx1, dy1, \
                                 __FILE__,__FUNCTION__,__LINE__);

// Level-1 Macros

#define AOCL_DTL_LOG_AMIN_INPUTS(loglevel, dt_type, n, incx) \
    if (gbIsLoggingEnabled) \
        AOCL_DTL_log_amin_sizes(loglevel, dt_type, n, incx, __FILE__, __FUNCTION__, __LINE__);

#define AOCL_DTL_LOG_AMAX_INPUTS(loglevel, dt_type, n, incx) \
    if (gbIsLoggingEnabled) \
        AOCL_DTL_log_amax_sizes(loglevel, dt_type, n, incx, __FILE__, __FUNCTION__, __LINE__);

#define AOCL_DTL_LOG_ASUM_INPUTS(loglevel, dt_type, n, incx) \
    if (gbIsLoggingEnabled) \
        AOCL_DTL_log_asum_sizes(loglevel, dt_type, n, incx, __FILE__, __FUNCTION__, __LINE__);

#define AOCL_DTL_LOG_AXPBY_INPUTS(loglevel, dt_type, n, alpha, incx, beta, incy) \
    if (gbIsLoggingEnabled) \
        AOCL_DTL_log_axpby_sizes(loglevel, dt_type, n, alpha, incx, beta, incy, __FILE__,\
                                __FUNCTION__, __LINE__);

#define AOCL_DTL_LOG_AXPY_INPUTS(loglevel, dt_type, n, alpha, incx, incy) \
    if (gbIsLoggingEnabled) \
        AOCL_DTL_log_axpy_sizes(loglevel, dt_type, n, alpha, incx, incy, __FILE__,\
                                __FUNCTION__, __LINE__);

#define AOCL_DTL_LOG_COPY_INPUTS(loglevel, dt_type, n, incx, incy) \
    if (gbIsLoggingEnabled) \
        AOCL_DTL_log_copy_sizes(loglevel, dt_type, n, incx, incy, __FILE__, __FUNCTION__, __LINE__);

#define AOCL_DTL_LOG_DOTV_INPUTS(loglevel, dt_type, conjx, n, incx, incy) \
    if (gbIsLoggingEnabled) \
        AOCL_DTL_log_dotv_sizes(loglevel, dt_type, conjx, n, incx, incy, __FILE__, __FUNCTION__, __LINE__); \

#define AOCL_DTL_LOG_NRM2_INPUTS(loglevel, dt_type, n, incx)\
    if (gbIsLoggingEnabled) \
        AOCL_DTL_log_nrm2_sizes(loglevel, dt_type, n, incx, __FILE__,__FUNCTION__,__LINE__);

#define AOCL_DTL_LOG_NRM2_STATS(loglevel, dt_type, n)    \
    if (gbIsLoggingEnabled) \
        AOCL_DTL_log_nrm2_stats(loglevel, dt_type, n);

#define AOCL_DTL_LOG_SCAL_INPUTS(loglevel, dt_type, alpha, n, incx )\
    if (gbIsLoggingEnabled) \
        AOCL_DTL_log_scal_sizes(loglevel, dt_type, alpha, n, incx,  __FILE__,__FUNCTION__,__LINE__);

#define AOCL_DTL_LOG_SWAP_INPUTS(loglevel, dt_type, n, incx, incy)\
    if (gbIsLoggingEnabled) \
        AOCL_DTL_log_swap_sizes(loglevel, dt_type, n, incx, incy,  __FILE__,__FUNCTION__,__LINE__);

// Matrix Copy and Transpose Macros

#define AOCL_DTL_LOG_MATADD_INPUTS(loglevel, dt_type, transa, transb, m, n, alpha, lda, beta, ldb, ldc ) \
    if (gbIsLoggingEnabled) \
        AOCL_DTL_log_matadd_sizes(loglevel, dt_type, transa, transb, m, n, alpha, lda, beta, ldb, ldc,  __FILE__,__FUNCTION__,__LINE__);

#define AOCL_DTL_LOG_MATCOPY_INPUTS(loglevel, dt_type, trans, rows, cols, alpha, lda, ldb ) \
    if (gbIsLoggingEnabled) \
        AOCL_DTL_log_matcopy_sizes(loglevel, dt_type, trans, rows, cols, alpha, lda, ldb,  __FILE__,__FUNCTION__,__LINE__);

#define AOCL_DTL_LOG_MATCOPY2_INPUTS(loglevel, dt_type, trans, rows, cols, alpha, lda, stridea, ldb, strideb ) \
    if (gbIsLoggingEnabled) \
        AOCL_DTL_log_matcopy2_sizes(loglevel, dt_type, trans, rows, cols, alpha, lda, stridea, ldb, strideb,  __FILE__,__FUNCTION__,__LINE__);

#else // AOCL_DTL_LOG_ENABLE

// Level-3 Macros

#define AOCL_DTL_LOG_GEMM_INPUTS(loglevel, dt, transa, transb, m, n, k, alpha, lda, ldb, beta, ldc)

#define AOCL_DTL_LOG_GEMM_STATS(loglevel, dt_type, m, n, k)

#define AOCL_DTL_LOG_GEMMT_INPUTS(loglevel, dt, uplo, transa, transb, n, k, alpha, lda, ldb, beta, ldc)

#define AOCL_DTL_LOG_GEMMT_STATS(loglevel, dt_type, n, k)

#define AOCL_DTL_LOG_HEMM_INPUTS(loglevel, dt_type, side, uplo, m, n, alpha, lda, ldb, beta, ldc)

#define AOCL_DTL_LOG_HER2K_INPUTS(loglevel, dt_type, uploc, transa, m, k, alpha, lda, ldb, beta, ldc)

#define AOCL_DTL_LOG_HERK_INPUTS(loglevel, dt_type, uploc, transa, m, k, alpha, lda, beta, ldc)

#define AOCL_DTL_LOG_SYMM_INPUTS(loglevel, dt_type, side, uploa, m, n, alpha, lda, ldb, beta, ldc)

#define AOCL_DTL_LOG_SYR2K_INPUTS(loglevel, dt_type, uploc, transa, m, k, alpha, lda, ldb, beta, ldc)

#define AOCL_DTL_LOG_SYRK_INPUTS(loglevel, dt_type, uploc, transa, m, k, alpha, lda, beta, ldc)

#define AOCL_DTL_LOG_TRMM_INPUTS(loglevel, dt_type, side, uploa, transa, diaga, m, n, alpha, lda, ldb)

#define AOCL_DTL_LOG_TRSM_INPUTS(loglevel, dt, side, uploa, transa, diaga, m, n, alpha, lda, ldb)

#define AOCL_DTL_LOG_TRSM_STATS(loglevel, dt_type, side, m, n)

// Level-3 Extension Macros

#define AOCL_DTL_LOG_GEMM3M_INPUTS(loglevel, dt, transa, transb, m, n, k, alpha, lda, ldb, beta, ldc)

#define AOCL_DTL_LOG_GEMM3M_STATS(loglevel, dt_type, m, n, k)

#define AOCL_DTL_LOG_GEMM_BATCH_INPUTS(loglevel, dt, group_count)

#define AOCL_DTL_LOG_GEMM_GET_SIZE_INPUTS(loglevel, dt, identifier, m, n, k)

#define AOCL_DTL_LOG_GEMM_PACK_INPUTS(loglevel, dt, identifier, trans, m, n, k, alpha, pld)

#define AOCL_DTL_LOG_GEMM_COMPUTE_INPUTS(loglevel, dt, transa, transb, m, n, k, lda, ldb, beta, ldc)

// Level-2 Macros

#define AOCL_DTL_LOG_GEMV_INPUTS(loglevel, dt_type, transa, m, n, alp, lda, incx, beta, incy)

#define AOCL_DTL_LOG_GER_INPUTS(loglevel, dt_type, m, n, alpha, incx, incy, lda)

#define AOCL_DTL_LOG_HEMV_INPUTS(loglevel, dt_type, uploa, m, alpha, lda, incx, beta, incy)

#define AOCL_DTL_LOG_HER2_INPUTS(loglevel, dt_type, uploa, m, alpha, incx, incy, lda)

#define AOCL_DTL_LOG_HER_INPUTS(loglevel, dt_type, uploa, m, alpha, incx, lda )

#define AOCL_DTL_LOG_SYMV_INPUTS(loglevel, dt_type, uploa, m, alpha, lda, incx, beta, incy)

#define AOCL_DTL_LOG_SYR2_INPUTS(loglevel, dt_type, uploa, m, alpha, incx, incy, lda)

#define AOCL_DTL_LOG_SYR_INPUTS(loglevel, dt_type, uploa, m, alpha, incx, lda)

#define AOCL_DTL_LOG_TRMV_INPUTS(loglevel, dt_type, uploa, transa, diaga, m, lda, incx)

#define AOCL_DTL_LOG_TRSV_INPUTS(loglevel, dt_type, uploa, transa, diaga, m, lda, incx )

// Level-2 Banded Macros

#define AOCL_DTL_LOG_GBMV_INPUTS(loglevel, dt_type, transa, m, n, kl, ku, alp, lda, incx, beta, incy)

#define AOCL_DTL_LOG_HBMV_INPUTS(loglevel, dt_type, uploa, m, k, alpha, lda, incx, beta, incy)

#define AOCL_DTL_LOG_SBMV_INPUTS(loglevel, dt_type, uploa, m, k, alpha, lda, incx, beta, incy)

#define AOCL_DTL_LOG_TBMV_INPUTS(loglevel, dt_type, uploa, transa, diaga, m, k, lda, incx)

#define AOCL_DTL_LOG_TBSV_INPUTS(loglevel, dt_type, uploa, transa, diaga, m, k, lda, incx)

// Level-2 Packed Macros

#define AOCL_DTL_LOG_HPMV_INPUTS(loglevel, dt_type, uploa, m, alpha, incx, beta, incy)

#define AOCL_DTL_LOG_HPR2_INPUTS(loglevel, dt_type, uploa, m, alpha, incx, incy)

#define AOCL_DTL_LOG_HPR_INPUTS(loglevel, dt_type, uploa, m, alpha, incx )

#define AOCL_DTL_LOG_SPMV_INPUTS(loglevel, dt_type, uploa, m, alpha, incx, beta, incy)

#define AOCL_DTL_LOG_SPR2_INPUTS(loglevel, dt_type, uploa, m, alpha, incx, incy)

#define AOCL_DTL_LOG_SPR_INPUTS(loglevel, dt_type, uploa, m, alpha, incx)

#define AOCL_DTL_LOG_TPMV_INPUTS(loglevel, dt_type, uploa, transa, diaga, m, incx)

#define AOCL_DTL_LOG_TPSV_INPUTS(loglevel, dt_type, uploa, transa, diaga, m, incx )

// Level-2 plane rotations and modified Givens transformation Macros

#define AOCL_DTL_LOG_ROT_INPUTS(loglevel, dt_type, m, incx, incy, c, s)

#define AOCL_DTL_LOG_ROTG_INPUTS(loglevel, dt_type, a, b, c, s)

#define AOCL_DTL_LOG_ROTM_INPUTS(loglevel, dt_type, m, incx, incy, param1, param2, param3, param4, param5) \

#define AOCL_DTL_LOG_ROTMG_INPUTS(loglevel, dt_type, dd1, dd2, dx1, dy1)

// Level-1 Macros

#define AOCL_DTL_LOG_AMIN_INPUTS(loglevel, dt_type, n, incx)

#define AOCL_DTL_LOG_AMAX_INPUTS(loglevel, dt_type, n, incx)

#define AOCL_DTL_LOG_ASUM_INPUTS(loglevel, dt_type, n, incx)

#define AOCL_DTL_LOG_AXPBY_INPUTS(loglevel, dt_type, n, alpha, incx, beta, incy)

#define AOCL_DTL_LOG_AXPY_INPUTS(loglevel, dt_type, n, alpha, incx, incy)

#define AOCL_DTL_LOG_COPY_INPUTS(loglevel, dt_type, n, incx, incy)

#define AOCL_DTL_LOG_DOTV_INPUTS(loglevel, dt_type, conjx, n, incx, incy)

#define AOCL_DTL_LOG_NRM2_INPUTS(loglevel, dt_type, n, incx)

#define AOCL_DTL_LOG_NRM2_STATS(loglevel, dt_type, n)

#define AOCL_DTL_LOG_SCAL_INPUTS(loglevel, dt_type, alpha, n, incx )

#define AOCL_DTL_LOG_SWAP_INPUTS(loglevel, dt_type, n, incx, incy)

// Matrix Copy and Transpose Macros

#define AOCL_DTL_LOG_MATADD_INPUTS(loglevel, dt_type, transa, transb, m, n, alpha, lda, beta, ldb, ldc )

#define AOCL_DTL_LOG_MATCOPY_INPUTS(loglevel, dt_type, trans, rows, cols, alpha, lda, ldb )

#define AOCL_DTL_LOG_MATCOPY2_INPUTS(loglevel, dt_type, trans, rows, cols, alpha, lda, stridea, ldb, strideb )

#endif // AOCL_DTL_LOG_ENABLE

#endif //ndef __AOCLDTL_BLIS_H
